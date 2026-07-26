#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <fstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/kpath/kpath_atlas.h"
#include "common/tensors/abstraction/kpath/kpath_film.h"
#include "common/tensors/abstraction/kpath/kpath_fill.h"
#include "common/tensors/abstraction/kpath/kpath_pipeline.h"
#include "common/tensors/abstraction/kpath/kpath_program.h"
#include "common/tensors/abstraction/kpath/kpath_raster.h"
#include "common/tensors/abstraction/kpath/kpath_raster_utils.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo_ir.h"
#include "mem_backend.h" // gp_mem_backend_torch_cuda_available() -- avoids needing <torch/torch.h> here
#include <filesystem>
#include <system_error>
#ifdef _WIN32
#  include <windows.h>
#endif
#ifndef NODUS_DEBUG_MOUSE_LISTENER
#define NODUS_DEBUG_MOUSE_LISTENER 0
#endif
#if NODUS_DEBUG_MOUSE_LISTENER
#define MOUSE_MOVE_DEBUGF(...) fprintf(stderr, __VA_ARGS__)
#else
#define MOUSE_MOVE_DEBUGF(...) do {} while (0)
#endif

#include "canvas_abi.h"

namespace {

// NOTE: no compatibility macros - use SDL3 API names directly.

namespace nt = nodus::tensors;
namespace nk = nodus::tensors::kpath;

static std::vector<nk::FilmHistogramDescriptor> default_film_histograms() {
    std::vector<nk::FilmHistogramDescriptor> descs(4);
    const std::array<const char*, 4> names = {"film_r", "film_g", "film_b", "film_a"};
    for (size_t i = 0; i < descs.size(); ++i) {
        descs[i].basis.name = names[i];
        if (i == 3) {
            descs[i].layer_reactance = {0.0f};
        } else {
            descs[i].layer_reactance = {1.0f};
        }
    }
    return descs;
}

struct CanvasHead {
    GP_CanvasContext* context = nullptr;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
    std::vector<uint8_t> upload_buffer;
    std::string workspace_path;
};

static constexpr size_t kMaxCanvasHeads = 8;
static constexpr const char* kDefaultCanvasWorkspacePath = "canvas_workspace.txt";
static constexpr const char* kFrontendCanvasStatePath = "frontend_canvases.txt";
static constexpr double kCanvasAutosaveIntervalSeconds = 2.0;

struct CanvasCollectionState {
    std::array<std::string, kMaxCanvasHeads> workspace_paths;
    std::array<bool, kMaxCanvasHeads> has_history{};
    size_t active_index = 0;
};
static std::string canvas_workspace_path_for_index(size_t index);
static void initialize_collection_state(CanvasCollectionState& state);
static bool load_frontend_collection_state(CanvasCollectionState& state);
static void save_frontend_collection_state(const CanvasCollectionState& state);
static constexpr int kDefaultCanvasWidth = 1000;
static constexpr int kDefaultCanvasHeight = 720;
static constexpr int kCanvasSwitchWindowMs = 3000;

// Kpath raster state used as an alternate render source inside the frontend shell.
static nt::AbstractTensorPool g_demo_tensor_pool(nt::AbstractTensorPool::Options{
    .clear_on_release = false,
    .cache_handles = false,
    .enable_shape_bucketing = false,
    .bucket_pow2_max = 0,
    .bucket_multiple = 0,
    .max_cached_handles_total = 0,
    .max_cached_handles_per_key = 0,
});
struct KpathDemoState {
    KpathDemoState() = default;
    KpathDemoState(const KpathDemoState&) = delete;
    KpathDemoState& operator=(const KpathDemoState&) = delete;
    KpathDemoState(KpathDemoState&&) = default;
    KpathDemoState& operator=(KpathDemoState&&) = default;
    nk::ArmatureProgram program;
    nk::ArmatureProgram base_program;
    nk::ArmatureProgram stats_program;
    nk::MachineControlConfig machine;
    nk::GaussianToolParams tool;
    nk::ProgramRasterTransform xform;
    nk::TensorCanvas2D energy;
    nk::TensorCanvas2D heat_energy;
    nk::TensorCanvas2D temp;
    nk::TensorCanvas2D stats_energy;
    nk::TensorCanvas2D stats_temp;
    nt::AbstractTensorPool::PooledTensor film_rgb;
    nt::AbstractTensorPool::PooledTensor film_exposures;
    nt::AbstractTensorPool::PooledTensor film_exposures_scratch;
    nk::FilmTensor film_tensor;
    std::vector<nk::FilmHistogramDescriptor> film_histograms;
    nt::AbstractTensorPool::PooledTensor flip;
    nt::AbstractTensor temp_state;
    nt::AbstractTensorPool::PooledTensor kernel_bank;
    nt::AbstractTensorPool::PooledTensor kernel_ids;
    nt::AbstractTensorPool::PooledTensor temp_kernel_ids;
    nt::AbstractTensorPool::PooledTensor diffusion_kernel;
    nt::TensorBackend* backend = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    bool ready = false;
    bool use_scatter = false;
    double avg_raster_ms = 0.0;
    uint32_t avg_count = 0;
    float phase = 0.0f;
    float phase_step = 0.05f;
    nk::Shaper shaper;
    bool shaper_ready = false;
    float radial_offset = -12.0f;
    float heat_decay_scale = 1.0f;
    float energy_scale = 1.0f;
    float exposure_gain = 2.5f;
    float film_decay = 0.94f;
    float heat_intensity = 1.0f;
    float heat_min_kelvin = 800.0f;
    float heat_max_kelvin = 6500.0f;
    bool normalize_energy = true;
    bool normalize_heat = true;
    struct CachedGlyphProgram {
        uint64_t outline_hash = 0;
        nk::GlyphOutline outline;
        nk::ArmatureProgram outline_program;
        nk::ArmatureProgram fill_program;
    };
    std::unordered_map<uint32_t, CachedGlyphProgram> glyph_cache;
    struct TimingProfile {
        double geom_ms = 0.0;
        double xform_ms = 0.0;
        double raster_ms = 0.0;
        double scatter_ms = 0.0;
        double scatter_plan_ms = 0.0;
        double scatter_clear_ms = 0.0;
        double scatter_map_ms = 0.0;
        double scatter_deposit_ms = 0.0;
        double scatter_temp_state_accum_ms = 0.0;
        double scatter_diffusion_prep_ms = 0.0;
        double scatter_diffusion_iter_ms = 0.0;
        double scatter_diffusion_copyback_ms = 0.0;
        double scatter_temp_state_writeback_ms = 0.0;
        double scatter_total_ms = 0.0;

        uint64_t scatter_program_points = 0;
        uint64_t scatter_plan_moments = 0;
        uint64_t scatter_unique_sites = 0;
        uint64_t scatter_site_activations = 0;
        uint64_t scatter_pixels_touched = 0;
        double stats_build_ms = 0.0;
        double stats_raster_ms = 0.0;
        double film_ms = 0.0;
        double film_map_ms = 0.0;
        double film_norm_energy_ms = 0.0;
        double film_norm_heat_ms = 0.0;
        double film_build_ms = 0.0;
        double film_scatter_ms = 0.0;
        double film_reduce_ms = 0.0;
        double film_pixel_ms = 0.0;
        double film_unmap_ms = 0.0;
        double film_total_ms = 0.0;
        double copy_ms = 0.0;
        double total_ms = 0.0;
    } timing;
};

struct FrontendResources {
    SDL_Window* window = nullptr;
    SDL_GLContext gl_context = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* canvas_texture = nullptr;
    int canvas_texture_width = 0;
    int canvas_texture_height = 0;
    // Device selection is tracked as a plain string; the actual torch::Device/
    // torch::Tensor types are never named here so this file never needs
    // <torch/torch.h> (see gp_mem_backend_torch_cuda_available()).
    std::string torch_device_name = "cpu";
    std::array<CanvasHead, kMaxCanvasHeads> canvas_heads{};
    size_t active_canvas_index = 0;
    CanvasCollectionState collection_state{};
    int canvas_width_hint = kDefaultCanvasWidth;
    int canvas_height_hint = kDefaultCanvasHeight;
    bool kpath_mode = false;
    KpathDemoState kpath_demo{};
    std::vector<uint8_t> kpath_pixels;
    // layout info updated on window resize and used for rendering/input translation
    struct FrontendLayout {
        int window_w = 0;
        int window_h = 0;
        int canvas_w = 0;   // display width for the canvas texture
        int canvas_h = 0;   // display height for the canvas texture
        int canvas_x = 0;   // top-left x where canvas is rendered (window coords)
        int canvas_y = 0;   // top-left y where canvas is rendered (window coords)
        int left_space = 0; // positive: pixels between window left and canvas left; negative: canvas extends left
        int right_space = 0; // positive: pixels between canvas right and window right
        int bottom_space = 0; // positive: pixels between canvas bottom and window bottom
        // high-precision mouse values (SDL3 provides floating-point coords/deltas)
        float mouse_fx = 0.0f;
        float mouse_fy = 0.0f;
        float mouse_fdx = 0.0f;
        float mouse_fdy = 0.0f;
    } layout;
};

struct CanvasTickSnapshot {
    std::array<GP_CanvasContext*, kMaxCanvasHeads> contexts{};
    std::array<size_t, kMaxCanvasHeads> head_indices{};
    size_t count = 0;
};

enum class DisplayMode {
    Interval = 0,
    Ratio = 1,
};

struct DisplayState {
    std::atomic<DisplayMode> mode{DisplayMode::Interval};
    std::atomic<double> interval{1.0 / 30.0};
    std::atomic<double> ratio{1.0};
    std::atomic<int64_t> ticks_since_display{0};
    std::atomic<double> last_display_time{0.0};
};

static std::string make_spiral_relgeo_ir(float phase, float radial_offset) {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(6);
    ss << "center = pt(0, 0);\n";
    ss << "pi_v = pi();\n";
    ss << "tau_v = mul(pi_v, 2.0);\n";
    ss << "radians_v = mul(tau_v, 5.0);\n";
    ss << "phase_v = " << phase << ";\n";
    ss << "radial_offset_v = " << radial_offset << ";\n";
    ss << "outer_radius = 220.0;\n";
    ss << "inner_radius = add(outer_radius, radial_offset_v);\n";
    ss << "phase_out_end = add(phase_v, radians_v);\n";
    ss << "path = path_begin();\n";
    ss << "path_spiral(path, center, outer_radius, radians_v, 720, phase_v, \"ccw\");\n";
    ss << "path_radial(path, center, outer_radius, inner_radius, phase_out_end, 32);\n";
    ss << "path_spiral(path, center, mul(inner_radius, -1.0), radians_v, 720, phase_out_end, \"cw\");\n";
    ss << "path_end(path, \"closed\", \"ccw\");\n";
    return ss.str();
}

static uint64_t hash_outline(const nk::GlyphOutline& outline) {
    uint64_t h = 14695981039346656037ull;
    auto mix = [&](const void* data, size_t len) {
        const uint8_t* b = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < len; ++i) {
            h ^= static_cast<uint64_t>(b[i]);
            h *= 1099511628211ull;
        }
    };
    mix(&outline.glyph_id, sizeof(outline.glyph_id));
    for (const auto& seg : outline.segments) {
        mix(&seg.op, sizeof(seg.op));
        mix(&seg.x1, sizeof(seg.x1));
        mix(&seg.y1, sizeof(seg.y1));
        mix(&seg.x2, sizeof(seg.x2));
        mix(&seg.y2, sizeof(seg.y2));
        mix(&seg.x3, sizeof(seg.x3));
        mix(&seg.y3, sizeof(seg.y3));
    }
    return h;
}

static void append_moveto_after_geometry(nk::ArmatureProgram& program, float safe_z) {
    if (program.points.empty()) return;
    const nk::ToolPoint& last = program.points.back();
    nk::ToolPoint move{};
    move.x = last.x;
    move.y = last.y;
    move.z = safe_z;
    move.engaged = false;
    program.points.push_back(move);
}

static std::string resolve_font_path() {
    if (const char* env = std::getenv("KPATH_TEST_FONT")) {
        std::filesystem::path p(env);
        if (std::filesystem::exists(p)) return p.string();
    }
#if defined(_WIN32)
    const std::vector<std::filesystem::path> candidates = {
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
        "C:/Windows/Fonts/seguisb.ttf"
    };
    for (const auto& c : candidates) if (std::filesystem::exists(c)) return c.string();
#endif
    return {};
}

static bool build_cached_program_from_outline(uint32_t glyph_id,
                                              const nk::GlyphOutline& outline,
                                              const nk::FillPlanConfig& fill_cfg,
                                              std::unordered_map<uint32_t, KpathDemoState::CachedGlyphProgram>& cache,
                                              nk::ArmatureProgram& out_program) {
    const uint64_t h = hash_outline(outline);
    auto& entry = cache[glyph_id];
    if (entry.outline_hash != h) {
        entry.outline_hash = h;
        entry.outline = outline;
        entry.outline_program.points.clear();
        entry.fill_program.points.clear();
        nk::append_glyph_outline_to_program(entry.outline_program, outline, 0.0f, 0.0f, 0.0f, 24);
        (void)nk::plan_fill_for_glyph_outline(outline, entry.fill_program, fill_cfg);
    }
    out_program.points.clear();
    out_program.points.reserve(entry.outline_program.points.size() + entry.fill_program.points.size());
    out_program.points.insert(out_program.points.end(),
                              entry.outline_program.points.begin(),
                              entry.outline_program.points.end());
    out_program.points.insert(out_program.points.end(),
                              entry.fill_program.points.begin(),
                              entry.fill_program.points.end());
    return !out_program.points.empty();
}

struct Mat4 {
    // Row-major 4x4 matrix.
    float m[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
};

struct Quat {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

static Mat4 mat4_identity() {
    return Mat4{};
}

static Mat4 mat4_from_quat_translation_scale(const Quat& q, float tx, float ty, float tz, float sx, float sy, float sz) {
    const float xx = q.x * q.x;
    const float yy = q.y * q.y;
    const float zz = q.z * q.z;
    const float xy = q.x * q.y;
    const float xz = q.x * q.z;
    const float yz = q.y * q.z;
    const float wx = q.w * q.x;
    const float wy = q.w * q.y;
    const float wz = q.w * q.z;

    Mat4 out;
    // Row-major rotation (right-handed).
    out.m[0] = (1.0f - 2.0f * (yy + zz)) * sx;
    out.m[1] = (2.0f * (xy + wz)) * sx;
    out.m[2] = (2.0f * (xz - wy)) * sx;
    out.m[3] = tx;

    out.m[4] = (2.0f * (xy - wz)) * sy;
    out.m[5] = (1.0f - 2.0f * (xx + zz)) * sy;
    out.m[6] = (2.0f * (yz + wx)) * sy;
    out.m[7] = ty;

    out.m[8] = (2.0f * (xz + wy)) * sz;
    out.m[9] = (2.0f * (yz - wx)) * sz;
    out.m[10] = (1.0f - 2.0f * (xx + yy)) * sz;
    out.m[11] = tz;

    out.m[12] = 0.0f;
    out.m[13] = 0.0f;
    out.m[14] = 0.0f;
    out.m[15] = 1.0f;
    return out;
}

static nk::ArmatureProgram transform_program(const nk::ArmatureProgram& src, const Mat4& mat) {
    nk::ArmatureProgram out;
    out.points.reserve(src.points.size());
    for (const auto& p : src.points) {
        nk::ToolPoint q = p;
        const float x = p.x;
        const float y = p.y;
        const float z = p.z;
        const float nx = mat.m[0] * x + mat.m[1] * y + mat.m[2] * z + mat.m[3];
        const float ny = mat.m[4] * x + mat.m[5] * y + mat.m[6] * z + mat.m[7];
        const float nz = mat.m[8] * x + mat.m[9] * y + mat.m[10] * z + mat.m[11];
        q.x = nx;
        q.y = ny;
        q.z = nz;
        out.points.push_back(q);
    }
    return out;
}

static void reset_kpath_demo(KpathDemoState& state) {
    state.program.points.clear();
    state.base_program.points.clear();
    state.stats_program.points.clear();
    state.energy = nk::TensorCanvas2D{};
    state.heat_energy = nk::TensorCanvas2D{};
    state.temp = nk::TensorCanvas2D{};
    state.stats_energy = nk::TensorCanvas2D{};
    state.stats_temp = nk::TensorCanvas2D{};
    state.film_rgb = nt::AbstractTensorPool::PooledTensor{};
    state.film_exposures = nt::AbstractTensorPool::PooledTensor{};
    state.film_exposures_scratch = nt::AbstractTensorPool::PooledTensor{};
    state.flip = nt::AbstractTensorPool::PooledTensor{};
    state.temp_state = nt::AbstractTensor{};
    state.kernel_bank = nt::AbstractTensorPool::PooledTensor{};
    state.kernel_ids = nt::AbstractTensorPool::PooledTensor{};
    state.temp_kernel_ids = nt::AbstractTensorPool::PooledTensor{};
    state.diffusion_kernel = nt::AbstractTensorPool::PooledTensor{};
    state.backend = nullptr;
    state.width = 0;
    state.height = 0;
    state.ready = false;
    state.use_scatter = false;
    state.avg_raster_ms = 0.0;
    state.avg_count = 0;
    state.phase = 0.0f;
    state.phase_step = 0.05f;
    state.timing = {};
    state.glyph_cache.clear();
}

static bool build_atlas_text_program(nk::Shaper& shaper,
                                     const std::vector<std::string>& lines,
                                     float start_x,
                                     float start_y,
                                     float line_height,
                                     nk::ArmatureProgram& out_program) {
    nk::AtlasBuilder builder;
    struct LinePlan {
        nk::TokenLayoutPlan plan;
        float x = 0.0f;
        float y = 0.0f;
    };
    std::vector<LinePlan> plans;
    plans.reserve(lines.size());

    float y = start_y;
    for (const auto& line : lines) {
        if (line.empty()) {
            y -= line_height;
            continue;
        }
        nk::CodepointSequence seq = nk::codepoints_from_utf8(line);
        nk::TokenLayoutPlan plan;
        if (!nk::build_token_from_sequence(builder, shaper, seq, plan)) {
            y -= line_height;
            continue;
        }
        plans.push_back(LinePlan{std::move(plan), start_x, y});
        y -= line_height;
    }

    if (plans.empty()) return false;

    nk::Atlas atlas = builder.finalize();
    out_program.points.clear();

    for (const auto& line : plans) {
        float pen_x = line.x;
        float pen_y = line.y;
        for (const auto& edge_id : line.plan.edges) {
            const nk::AtlasEdge& edge = atlas.edge(edge_id);
            const nk::NodeMetadata* meta = atlas.node_metadata(edge.dst);
            if (meta) {
                nk::append_glyph_outline_to_program(out_program,
                                                    meta->outline,
                                                    pen_x + edge.offset_x,
                                                    pen_y + edge.offset_y,
                                                    0.0f,
                                                    12);
            }
            pen_x += edge.advance_x;
            pen_y += edge.advance_y;
        }
    }

    return !out_program.points.empty();
}

static void add_canvas_inplace(nk::TensorCanvas2D& dst, const nk::TensorCanvas2D& src, float scale) {
    if (dst.width != src.width || dst.height != src.height) return;
    const size_t count = static_cast<size_t>(dst.width) * dst.height;
    for (size_t i = 0; i < count; ++i) {
        dst.values[i] += src.values[i] * scale;
    }
}

static bool build_spiral_program_from_relgeo(float phase,
                                             float radial_offset,
                                             std::unordered_map<uint32_t, KpathDemoState::CachedGlyphProgram>& cache,
                                             nk::ArmatureProgram& out_program) {
    const std::string ir = make_spiral_relgeo_ir(phase, radial_offset);
    nk::RelProgram rel;
    std::string err;
    if (!nk::relgeo_program_from_ir(ir, rel, &err)) {
        std::cerr << "RelGeo parse failed: " << err << "\n";
        return false;
    }
    nk::RelGlyph glyph;
    glyph.glyph_id = 1;
    glyph.program = std::move(rel);
    nk::GlyphOutline outline;
    if (!nk::compile_relglyph_outline(glyph, outline, 0.0f, 0.0f, 1.0f, &err)) {
        std::cerr << "RelGeo compile failed: " << err << "\n";
        return false;
    }
    nk::FillPlanConfig fill_cfg;
    fill_cfg.tool.tool_width = 6.0f;
    // Lower overlap to reduce redundant re-writing of the same areas.
    // High overlap is useful for quality/coverage, but it explodes path length.
    fill_cfg.tool.overlap = 0.25f;
    fill_cfg.tool.angle_degrees = 0.0f;
    return build_cached_program_from_outline(glyph.glyph_id, outline, fill_cfg, cache, out_program);
}

static bool initialize_kpath_demo(KpathDemoState& state, uint32_t width, uint32_t height) {
    reset_kpath_demo(state);
    state.width = std::max<uint32_t>(1, width);
    state.height = std::max<uint32_t>(1, height);

    nt::register_in_memory_backend(true);
    auto& backend_singleton = nt::in_memory_backend_singleton();
    state.backend = backend_singleton.name() ? static_cast<nt::TensorBackend*>(&backend_singleton) : nullptr;
    if (!state.backend) {
        std::cerr << "Kpath demo: in-memory backend unavailable\n";
        return false;
    }

    if (!build_spiral_program_from_relgeo(0.0f, state.radial_offset, state.glyph_cache, state.base_program)) {
        std::cerr << "Failed to build initial kpath RelGeo program\n";
        return false;
    }
    state.machine.step_px = 0.75f;
    state.machine.energy_per_px = 1.0f;
    state.machine.enable_thermal_guard = false;
    state.tool.sigma_px = 1.1f;

    state.xform = nk::plan_program_raster_transform_refined(state.base_program,
                                                            state.machine,
                                                            1.0f,
                                                            8.0f,
                                                            state.tool);
    state.xform.width_px = state.width;
    state.xform.height_px = state.height;
    state.xform.scale = 1.0f;
    state.xform.shift_x = static_cast<float>(state.width) * 0.5f;
    state.xform.shift_y = static_cast<float>(state.height) * 0.5f;

    state.energy.resize(state.width, state.height, 0.0f);
    state.heat_energy.resize(state.width, state.height, 0.0f);
    state.temp.resize(state.width, state.height, 0.0f);
    state.stats_energy.resize(state.width, state.height, 0.0f);
    state.stats_temp.resize(state.width, state.height, 0.0f);
    {
        state.film_histograms = default_film_histograms();
        if (!state.film_tensor.configure(state.width, state.height, state.film_histograms)) {
            std::cerr << "Kpath demo: failed to configure film histograms\n";
            return false;
        }

        nt::TensorDesc exposures_desc{};
        exposures_desc.dtype = nt::TensorDType::F32;
        exposures_desc.layout = nt::TensorLayout::Dense;
        exposures_desc.shape.dims = {state.height,
                                     state.width,
                                     state.film_tensor.total_channels()};
        state.film_exposures = g_demo_tensor_pool.acquire(exposures_desc, state.backend);
        state.film_exposures_scratch = g_demo_tensor_pool.acquire(exposures_desc, state.backend);
        if (!state.film_exposures.valid()) {
            std::cerr << "Kpath demo: failed to allocate film exposure tensor\n";
            return false;
        }
        if (!state.film_exposures_scratch.valid()) {
            std::cerr << "Kpath demo: failed to allocate film exposure scratch tensor\n";
            return false;
        }

        nt::TensorDesc agg_desc{};
        agg_desc.dtype = nt::TensorDType::F32;
        agg_desc.layout = nt::TensorLayout::Dense;
        agg_desc.shape.dims = {state.height,
                               state.width,
                               state.film_tensor.histogram_count()};
        state.film_rgb = g_demo_tensor_pool.acquire(agg_desc, state.backend);
        if (!state.film_rgb.valid()) {
            std::cerr << "Kpath demo: failed to allocate film accumulator tensor\n";
            return false;
        }
    }

    nk::ProgramScatterPlan scatter_plan;
    if (!nk::plan_program_scatter(state.base_program, state.machine, state.xform, &scatter_plan, true, true, state.backend)) {
        std::cerr << "Kpath demo: failed to plan scatter\n";
        return false;
    }
    nt::AbstractTensor dense_points = nk::coo_points_to_dense_f32(scatter_plan.points, state.backend);
    if (!dense_points.valid()) {
        std::cerr << "Kpath demo: failed to materialize scatter points\n";
        return false;
    }

    const uint32_t head_count = 4;
    const std::array<float, 4> kernel_weights = {1.0f, 0.85f, 0.7f, 0.55f};
    state.kernel_bank = nk::make_kernel_bank(g_demo_tensor_pool, state.backend, kernel_weights);
    state.kernel_ids = nk::make_kernel_ids_for_heads(g_demo_tensor_pool, dense_points, state.width, state.height, head_count, state.backend);
    state.temp_kernel_ids = nk::make_kernel_ids_for_heads(g_demo_tensor_pool, dense_points, state.width, state.height, head_count, state.backend);
    state.diffusion_kernel = nk::make_laplacian_kernel(g_demo_tensor_pool, state.backend);
    state.flip = nk::make_flip_tensor(g_demo_tensor_pool, state.backend, state.width, state.height);

    if (const std::string font_path = resolve_font_path(); !font_path.empty()) {
        state.shaper_ready = state.shaper.load_font(font_path, 16.0f);
        if (!state.shaper_ready) {
            std::cerr << "Kpath demo: failed to load font at " << font_path << "\n";
        }
    } else {
        std::cerr << "Kpath demo: no font found for stats overlay\n";
    }

    state.ready = state.kernel_bank.valid() && state.kernel_ids.valid() && state.diffusion_kernel.valid() && state.flip.valid();
    return state.ready;
}

static bool render_kpath_demo_frame(KpathDemoState& state, std::vector<uint8_t>& rgba_out) {
    if (!state.ready || !state.backend) return false;

    const auto frame_start = std::chrono::high_resolution_clock::now();
    state.phase += state.phase_step;
    const auto t_geom_start = std::chrono::high_resolution_clock::now();
    const float ang = state.phase;
    const float half = 0.5f * ang;
    Quat qz{};
    qz.z = std::sin(half);
    qz.w = std::cos(half);
    Mat4 transform = mat4_from_quat_translation_scale(qz, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
    state.program = transform_program(state.base_program, transform);
    append_moveto_after_geometry(state.program, state.machine.safe_z);
    const auto t_geom_end = std::chrono::high_resolution_clock::now();

    nk::ThermalSimConfig heat_cfg = nk::thermal_sim_from_preset(nk::ThermalMaterialPreset::SteelThin);
    heat_cfg.decay = std::clamp(heat_cfg.decay * state.heat_decay_scale, 0.0f, 0.999f);
    nk::MachineControlConfig machine = state.machine;
    machine.energy_per_px = state.machine.energy_per_px * state.energy_scale;
    machine.energy_to_temp = heat_cfg.heat_gain;
    nk::BeamToolParams gaussian_tool{};
    gaussian_tool.falloff = nk::BeamFalloffKind::Gaussian;
    gaussian_tool.sigma_px = state.tool.sigma_px;
    nk::BeamToolParams airy_tool{};
    airy_tool.falloff = nk::BeamFalloffKind::Airy;

    const auto t_xform_start = std::chrono::high_resolution_clock::now();
    state.xform = nk::plan_program_raster_transform_refined(state.program,
                                                            machine,
                                                            1.0f,
                                                            8.0f,
                                                            state.tool);
    state.xform.width_px = state.width;
    state.xform.height_px = state.height;
    state.xform.scale = 1.0f;
    state.xform.shift_x = static_cast<float>(state.width) * 0.5f;
    state.xform.shift_y = static_cast<float>(state.height) * 0.5f;
    const auto t_xform_end = std::chrono::high_resolution_clock::now();
    state.timing.geom_ms = std::chrono::duration<double, std::milli>(t_geom_end - t_geom_start).count();
    state.timing.xform_ms = std::chrono::duration<double, std::milli>(t_xform_end - t_xform_start).count();

    const auto t0 = std::chrono::high_resolution_clock::now();
    nk::ScatterTimingBreakdown scatter_breakdown;
    if (!state.use_scatter) {
        const auto t_raster_start = std::chrono::high_resolution_clock::now();
        nk::rasterize_program_with_kernel_transformed(state.program,
                                                      state.energy,
                                                      state.temp,
                                                      machine,
                                                      gaussian_tool,
                                                      state.xform);
        const auto t_raster_end = std::chrono::high_resolution_clock::now();
        const auto t_scatter_start = std::chrono::high_resolution_clock::now();
        if (!nk::rasterize_program_scatter_with_spatial_kernel(state.program,
                                                               state.heat_energy,
                                                               state.temp,
                                                               machine,
                                                               state.xform,
                                                               airy_tool,
                                                               &state.diffusion_kernel.tensor(),
                                                               heat_cfg,
                                                               &state.temp_state,
                                                               state.backend,
                                                               &scatter_breakdown)) {
            std::cerr << "Kpath demo: scatter rasterization failed (scatter-only=0, w="
                      << state.width << ", h=" << state.height << ")\n";
            return false;
        }
        const auto t_scatter_end = std::chrono::high_resolution_clock::now();
        state.timing.raster_ms = std::chrono::duration<double, std::milli>(t_raster_end - t_raster_start).count();
        state.timing.scatter_ms = std::chrono::duration<double, std::milli>(t_scatter_end - t_scatter_start).count();
    } else {
        const auto t_scatter_start = std::chrono::high_resolution_clock::now();
        if (!nk::rasterize_program_scatter_with_spatial_kernel(state.program,
                                                               state.energy,
                                                               state.temp,
                                                               machine,
                                                               state.xform,
                                                               airy_tool,
                                                               &state.diffusion_kernel.tensor(),
                                                               heat_cfg,
                                                               &state.temp_state,
                                                               state.backend,
                                                               &scatter_breakdown)) {
            std::cerr << "Kpath demo: scatter rasterization failed (scatter-only=1, w="
                      << state.width << ", h=" << state.height << ")\n";
            return false;
        }
        const auto t_scatter_end = std::chrono::high_resolution_clock::now();
        state.timing.raster_ms = 0.0;
        state.timing.scatter_ms = std::chrono::duration<double, std::milli>(t_scatter_end - t_scatter_start).count();
    }

    state.timing.scatter_plan_ms = scatter_breakdown.plan_ms;
    state.timing.scatter_clear_ms = scatter_breakdown.clear_ms;
    state.timing.scatter_map_ms = scatter_breakdown.map_ms;
    state.timing.scatter_deposit_ms = scatter_breakdown.deposit_ms;
    state.timing.scatter_temp_state_accum_ms = scatter_breakdown.temp_state_accum_ms;
    state.timing.scatter_diffusion_prep_ms = scatter_breakdown.diffusion_prep_ms;
    state.timing.scatter_diffusion_iter_ms = scatter_breakdown.diffusion_iter_ms;
    state.timing.scatter_diffusion_copyback_ms = scatter_breakdown.diffusion_copyback_ms;
    state.timing.scatter_temp_state_writeback_ms = scatter_breakdown.temp_state_writeback_ms;
    state.timing.scatter_total_ms = scatter_breakdown.total_ms;

    state.timing.scatter_program_points = scatter_breakdown.program_points;
    state.timing.scatter_plan_moments = scatter_breakdown.plan_moments;
    state.timing.scatter_unique_sites = scatter_breakdown.unique_sites;
    state.timing.scatter_site_activations = scatter_breakdown.site_activations;
    state.timing.scatter_pixels_touched = scatter_breakdown.pixels_touched;

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    state.avg_raster_ms = (state.avg_raster_ms * state.avg_count + ms) / static_cast<double>(state.avg_count + 1);
    state.avg_count = std::min<uint32_t>(state.avg_count + 1, 120u);

    const auto t_stats_build_start = std::chrono::high_resolution_clock::now();
    if (state.shaper_ready) {
        const float origin_x = -static_cast<float>(state.width) * 0.5f + 18.0f;
        const float origin_y = static_cast<float>(state.height) * 0.5f - 24.0f;
        const float line_height = 18.0f;
        char line_buf[128];
        std::vector<std::string> lines;
        lines.reserve(28);
        std::snprintf(line_buf, sizeof(line_buf), "kpath bench");
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "geom: %.2f ms", state.timing.geom_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "xform: %.2f ms", state.timing.xform_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "raster: %.2f ms", state.timing.raster_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "scatter: %.2f ms", state.timing.scatter_ms);
        lines.emplace_back(line_buf);
          std::snprintf(line_buf, sizeof(line_buf), "  prog points: %llu", static_cast<unsigned long long>(state.timing.scatter_program_points));
          lines.emplace_back(line_buf);
          std::snprintf(line_buf, sizeof(line_buf), "  plan moments: %llu", static_cast<unsigned long long>(state.timing.scatter_plan_moments));
          lines.emplace_back(line_buf);
          std::snprintf(line_buf, sizeof(line_buf), "  unique sites: %llu", static_cast<unsigned long long>(state.timing.scatter_unique_sites));
          lines.emplace_back(line_buf);
          std::snprintf(line_buf, sizeof(line_buf), "  site activations: %llu", static_cast<unsigned long long>(state.timing.scatter_site_activations));
          lines.emplace_back(line_buf);
          std::snprintf(line_buf, sizeof(line_buf), "  pixels touched: %llu", static_cast<unsigned long long>(state.timing.scatter_pixels_touched));
          lines.emplace_back(line_buf);
          const double eff_kernel = (state.timing.scatter_site_activations > 0)
                                                  ? (static_cast<double>(state.timing.scatter_pixels_touched) /
                                                      static_cast<double>(state.timing.scatter_site_activations))
                                                  : 0.0;
          const double eff_sites = (state.timing.scatter_plan_moments > 0)
                                                 ? (static_cast<double>(state.timing.scatter_unique_sites) /
                                                     static_cast<double>(state.timing.scatter_plan_moments))
                                                 : 0.0;
          const double avg_footprint = (state.timing.scatter_plan_moments > 0)
                                                      ? (static_cast<double>(state.timing.scatter_site_activations) /
                                                          static_cast<double>(state.timing.scatter_plan_moments))
                                                      : 0.0;
          std::snprintf(line_buf, sizeof(line_buf), "  eff(kernel): %.6f", eff_kernel);
          lines.emplace_back(line_buf);
          std::snprintf(line_buf, sizeof(line_buf), "  eff(sites): %.6f", eff_sites);
          lines.emplace_back(line_buf);
          std::snprintf(line_buf, sizeof(line_buf), "  avg footprint px/moment: %.3f", avg_footprint);
          lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  plan: %.2f ms", state.timing.scatter_plan_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  clear: %.2f ms", state.timing.scatter_clear_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  map: %.2f ms", state.timing.scatter_map_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  deposit: %.2f ms", state.timing.scatter_deposit_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  decay/state: %.2f ms", state.timing.scatter_temp_state_accum_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  diffusion prep: %.2f ms", state.timing.scatter_diffusion_prep_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  diffusion iters: %.2f ms", state.timing.scatter_diffusion_iter_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  diffusion copy: %.2f ms", state.timing.scatter_diffusion_copyback_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  temp writeback: %.2f ms", state.timing.scatter_temp_state_writeback_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "stats_build: %.2f ms", state.timing.stats_build_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "stats_raster: %.2f ms", state.timing.stats_raster_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "film: %.2f ms", state.timing.film_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  map: %.2f ms", state.timing.film_map_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  norm energy: %.2f ms", state.timing.film_norm_energy_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  norm heat: %.2f ms", state.timing.film_norm_heat_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  build: %.2f ms", state.timing.film_build_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  scatter: %.2f ms", state.timing.film_scatter_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  reduce: %.2f ms", state.timing.film_reduce_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  pixels: %.2f ms", state.timing.film_pixel_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "  unmap: %.2f ms", state.timing.film_unmap_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "copy: %.2f ms", state.timing.copy_ms);
        lines.emplace_back(line_buf);
        std::snprintf(line_buf, sizeof(line_buf), "total: %.2f ms", state.timing.total_ms);
        lines.emplace_back(line_buf);
        (void)build_atlas_text_program(state.shaper,
                                       lines,
                                       origin_x,
                                       origin_y,
                                       line_height,
                                       state.stats_program);
    }
    const auto t_stats_build_end = std::chrono::high_resolution_clock::now();
    const auto t_stats_raster_start = std::chrono::high_resolution_clock::now();
    if (state.shaper_ready && !state.stats_program.points.empty()) {
        nk::MachineControlConfig text_machine = state.machine;
        text_machine.step_px = 1.0f;
        nk::BeamToolParams text_tool = gaussian_tool;
        text_tool.sigma_px = 0.9f;
        nk::rasterize_program_with_kernel_transformed(state.stats_program,
                                                      state.stats_energy,
                                                      state.stats_temp,
                                                      text_machine,
                                                      text_tool,
                                                      state.xform);
        add_canvas_inplace(state.energy, state.stats_energy, 0.65f);
    }
    const auto t_stats_raster_end = std::chrono::high_resolution_clock::now();

    const nk::BeamHistogram beam_hist{};
    nk::BlackbodyResponseConfig heat_response{};
    heat_response.min_kelvin = state.heat_min_kelvin;
    heat_response.max_kelvin = state.heat_max_kelvin;
    heat_response.intensity = state.heat_intensity;
    const auto t_film_start = std::chrono::high_resolution_clock::now();
    nk::FilmTimingBreakdown film_breakdown;
    if (!nk::fill_flip_rgba_dual_tensor(state.energy,
                                        state.temp,
                                        state.film_tensor,
                                        state.film_exposures,
                                        state.film_exposures_scratch,
                                        state.film_rgb.tensor(),
                                        state.flip.tensor(),
                                        state.exposure_gain,
                                        state.film_decay,
                                        beam_hist,
                                        heat_response,
                                        state.normalize_energy,
                                        state.normalize_heat,
                                        1.0f,
                                        1.0f,
                                        &film_breakdown)) {
        std::cerr << "Kpath demo: film composite failed (w=" << state.width
                  << ", h=" << state.height << ")\n";
        return false;
    }
    const auto t_film_end = std::chrono::high_resolution_clock::now();
    state.timing.film_map_ms = film_breakdown.map_ms;
    state.timing.film_norm_energy_ms = film_breakdown.normalize_energy_ms;
    state.timing.film_norm_heat_ms = film_breakdown.normalize_heat_ms;
    state.timing.film_build_ms = film_breakdown.build_ms;
    state.timing.film_scatter_ms = film_breakdown.scatter_ms;
    state.timing.film_reduce_ms = film_breakdown.reduce_ms;
    state.timing.film_pixel_ms = film_breakdown.pixel_loop_ms;
    state.timing.film_unmap_ms = film_breakdown.unmap_ms;
    state.timing.film_total_ms = film_breakdown.total_ms;

    auto* mem = dynamic_cast<nt::InMemoryBackend*>(state.backend);
    if (!mem) {
        std::cerr << "Kpath demo: backend is not InMemoryBackend\n";
        return false;
    }
    void* src_v = nullptr;
    size_t src_bytes = 0;
    if (!mem->map(state.flip->handle(), &src_v, &src_bytes)) {
        const auto handle = state.flip->handle();
        std::cerr << "Kpath demo: failed to map flip tensor (handle_id="
                  << handle.id << ")\n";
        return false;
    }
    const size_t expected = static_cast<size_t>(state.width) * static_cast<size_t>(state.height) * sizeof(uint32_t);
    rgba_out.resize(expected);
    const auto t_copy_start = std::chrono::high_resolution_clock::now();
    std::memcpy(rgba_out.data(), src_v, expected);
    const auto t_copy_end = std::chrono::high_resolution_clock::now();
    mem->unmap(state.flip->handle());
    const auto frame_end = std::chrono::high_resolution_clock::now();

    state.timing.geom_ms = std::chrono::duration<double, std::milli>(t_geom_end - t_geom_start).count();
    state.timing.xform_ms = std::chrono::duration<double, std::milli>(t_xform_end - t_xform_start).count();
    state.timing.stats_build_ms = std::chrono::duration<double, std::milli>(t_stats_build_end - t_stats_build_start).count();
    state.timing.stats_raster_ms = std::chrono::duration<double, std::milli>(t_stats_raster_end - t_stats_raster_start).count();
    state.timing.film_ms = std::chrono::duration<double, std::milli>(t_film_end - t_film_start).count();
    state.timing.copy_ms = std::chrono::duration<double, std::milli>(t_copy_end - t_copy_start).count();
    state.timing.total_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    return true;
}

static bool ensure_kpath_ready(FrontendResources& resources, uint32_t width, uint32_t height) {
    if (resources.kpath_demo.ready &&
        resources.kpath_demo.width == width &&
        resources.kpath_demo.height == height) {
        return true;
    }
    resources.kpath_pixels.clear();
    return initialize_kpath_demo(resources.kpath_demo, width, height);
}

class CanvasTickController {
public:
    explicit CanvasTickController(FrontendResources& resources)
        : resources_(resources) {
        for (size_t i = 0; i < kMaxCanvasHeads; ++i) {
            head_thread_dt_[i].store(1.0 / 60.0);
            head_thread_ratio_[i].store(1.0);
            head_thread_free_spin_[i].store(false);
            head_gui_dt_[i].store(1.0 / 60.0);
            render_frame_counters_[i].store(0);
            display_params_[i].mode.store(DisplayMode::Interval);
            display_params_[i].interval.store(1.0 / 30.0);
            display_params_[i].ratio.store(1.0);
            display_params_[i].ticks_since_display.store(0);
            display_params_[i].last_display_time.store(0.0);
        }
        gui_delay_s_.store(1.0 / 60.0);
        thread_delay_s_.store(1.0 / 60.0);
    }

    ~CanvasTickController() { stop(); }

    void start() {
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true)) return;
        worker_ = std::thread(&CanvasTickController::thread_loop, this);
    }

    void stop() {
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false)) return;
        if (worker_.joinable()) worker_.join();
    }

    void record_render(float dt) {
        publish_snapshot();
        for (size_t i = 0; i < kMaxCanvasHeads; ++i) {
            if (!resources_.canvas_heads[i].context) continue;
            head_gui_dt_[i].store(dt > 0.0f ? dt : (1.0 / 60.0), std::memory_order_relaxed);
            render_frame_counters_[i].fetch_add(1, std::memory_order_relaxed);
        }
    }

    void set_global_gui_delay(double seconds) {
        gui_delay_s_.store(std::max(0.0, seconds), std::memory_order_relaxed);
    }

    void set_global_thread_delay(double seconds) {
        thread_delay_s_.store(std::max(0.0, seconds), std::memory_order_relaxed);
    }

    void set_canvas_thread_config(size_t head_idx, double dt, double ratio, bool free_spin) {
        if (head_idx >= kMaxCanvasHeads) return;
        head_thread_dt_[head_idx].store(std::max(0.0, dt), std::memory_order_relaxed);
        head_thread_ratio_[head_idx].store(std::max(0.0, ratio), std::memory_order_relaxed);
        head_thread_free_spin_[head_idx].store(free_spin, std::memory_order_relaxed);
    }

    void set_canvas_gui_dt(size_t head_idx, double dt) {
        if (head_idx >= kMaxCanvasHeads) return;
        head_gui_dt_[head_idx].store(std::max(0.0, dt), std::memory_order_relaxed);
    }

    double gui_delay_seconds() const {
        return gui_delay_s_.load(std::memory_order_relaxed);
    }

    enum class DisplayMode {
        Interval = 0,
        Ratio = 1,
    };

    void set_canvas_display_interval(size_t head_idx, double interval) {
        if (head_idx >= kMaxCanvasHeads) return;
        display_params_[head_idx].mode.store(DisplayMode::Interval, std::memory_order_relaxed);
        display_params_[head_idx].interval.store(std::max(0.0, interval), std::memory_order_relaxed);
    }

    void set_canvas_display_ratio(size_t head_idx, double ratio) {
        if (head_idx >= kMaxCanvasHeads) return;
        display_params_[head_idx].mode.store(DisplayMode::Ratio, std::memory_order_relaxed);
        display_params_[head_idx].ratio.store(std::max(1.0, ratio), std::memory_order_relaxed);
    }

    bool should_display(size_t head_idx, double now_seconds) const {
        if (head_idx >= kMaxCanvasHeads) return false;
        const DisplayState& state = display_params_[head_idx];
        DisplayMode mode = state.mode.load(std::memory_order_relaxed);
        if (mode == DisplayMode::Interval) {
            double interval = state.interval.load(std::memory_order_relaxed);
            double last = state.last_display_time.load(std::memory_order_relaxed);
            return interval <= 0.0 || (now_seconds - last >= interval);
        }
        double ratio = state.ratio.load(std::memory_order_relaxed);
        double ticks = static_cast<double>(state.ticks_since_display.load(std::memory_order_relaxed));
        return ticks >= ratio;
    }

    void mark_displayed(size_t head_idx, double now_seconds) {
        if (head_idx >= kMaxCanvasHeads) return;
        auto& state = display_params_[head_idx];
        state.last_display_time.store(now_seconds, std::memory_order_relaxed);
        state.ticks_since_display.store(0, std::memory_order_relaxed);
    }

private:
    struct DisplayState {
        std::atomic<DisplayMode> mode{DisplayMode::Interval};
        std::atomic<double> interval{1.0 / 30.0};
        std::atomic<double> ratio{1.0};
        std::atomic<int64_t> ticks_since_display{0};
        std::atomic<double> last_display_time{0.0};
    };

    void publish_snapshot() {
        CanvasTickSnapshot temp{};
        for (size_t i = 0; i < kMaxCanvasHeads; ++i) {
            if (!resources_.canvas_heads[i].context) continue;
            temp.contexts[temp.count] = resources_.canvas_heads[i].context;
            temp.head_indices[temp.count] = i;
            ++temp.count;
        }
        int next = 1 - snapshot_index_.load(std::memory_order_relaxed);
        snapshots_[next] = temp;
        snapshot_index_.store(next, std::memory_order_release);
    }

    CanvasTickSnapshot load_snapshot() const {
        return snapshots_[snapshot_index_.load(std::memory_order_acquire)];
    }

    void thread_loop() {
        std::array<double, kMaxCanvasHeads> accum{};
        std::array<uint64_t, kMaxCanvasHeads> last_frame_counts{};
        while (running_.load(std::memory_order_relaxed)) {
            CanvasTickSnapshot snapshot = load_snapshot();
            for (size_t entry = 0; entry < snapshot.count; ++entry) {
                size_t head_idx = snapshot.head_indices[entry];
                GP_CanvasContext* ctx = snapshot.contexts[entry];
                if (!ctx) continue;
                bool free_spin = head_thread_free_spin_[head_idx].load(std::memory_order_relaxed);
                double ratio = head_thread_ratio_[head_idx].load(std::memory_order_relaxed);
                double ticks_to_run = 0.0;
                if (free_spin) {
                    ticks_to_run = 1.0;
                } else {
                    uint64_t current = render_frame_counters_[head_idx].load(std::memory_order_relaxed);
                    uint64_t frames = current - last_frame_counts[head_idx];
                    if (frames == 0) continue;
                    accum[head_idx] += static_cast<double>(frames) * ratio;
                    ticks_to_run = std::floor(accum[head_idx]);
                    accum[head_idx] -= ticks_to_run;
                    last_frame_counts[head_idx] = current;
                    if (ticks_to_run <= 0.0) continue;
                }
                int paused = 1;
                if (gp_canvas_get_thread_manager_paused(ctx, &paused) && paused) continue;
                float dt = static_cast<float>(head_thread_dt_[head_idx].load(std::memory_order_relaxed));
                int tick_count = static_cast<int>(ticks_to_run);
                for (int t = 0; t < tick_count; ++t) {
                    gp_canvas_step(ctx, dt);
                }
                if (tick_count > 0) {
                    display_params_[head_idx].ticks_since_display.fetch_add(tick_count, std::memory_order_relaxed);
                }
            }
            double delay = thread_delay_s_.load(std::memory_order_relaxed);
            if (delay > 0.0) {
                std::this_thread::sleep_for(std::chrono::duration<double>(delay));
            }
        }
    }

    FrontendResources& resources_;
    std::array<CanvasTickSnapshot, 2> snapshots_{};
    std::atomic<int> snapshot_index_{0};
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::array<std::atomic<double>, kMaxCanvasHeads> head_thread_dt_{};
    std::array<std::atomic<double>, kMaxCanvasHeads> head_thread_ratio_{};
    std::array<std::atomic<bool>, kMaxCanvasHeads> head_thread_free_spin_{};
    std::array<std::atomic<double>, kMaxCanvasHeads> head_gui_dt_{};
    std::array<std::atomic<uint64_t>, kMaxCanvasHeads> render_frame_counters_{};
    std::array<DisplayState, kMaxCanvasHeads> display_params_{};
    std::atomic<double> gui_delay_s_{1.0 / 60.0};
    std::atomic<double> thread_delay_s_{1.0 / 60.0};
};

bool create_canvas_head(CanvasHead& head, int width, int height) {
    if (head.context) {
        return true;
    }
    head.context = gp_canvas_create(width, height);
    if (!head.context) {
        return false;
    }
    head.width = width;
    head.height = height;
    head.pixels.assign(static_cast<size_t>(width) * height * 4, 0u);
    return true;
}

void destroy_canvas_head(CanvasHead& head) {
    if (!head.context) {
        return;
    }
    gp_canvas_destroy(head.context);
    head.context = nullptr;
    head.pixels.clear();
    head.width = 0;
    head.height = 0;
    head.workspace_path.clear();
}

bool ensure_canvas_head_initialized(FrontendResources& resources, size_t index) {
    if (index >= kMaxCanvasHeads) return false;
    auto& head = resources.canvas_heads[index];
    if (head.context) return true;
    if (!create_canvas_head(head, resources.canvas_width_hint, resources.canvas_height_hint)) {
        return false;
    }
    std::string workspace_path = resources.collection_state.workspace_paths[index];
    if (workspace_path.empty()) {
        workspace_path = canvas_workspace_path_for_index(index);
    }
    head.workspace_path = workspace_path;
    resources.collection_state.workspace_paths[index] = workspace_path;
    if (!workspace_path.empty()) {
        gp_canvas_set_autosave(head.context, workspace_path.c_str(), kCanvasAutosaveIntervalSeconds);
        std::ifstream ifs(workspace_path);
        if (ifs.good()) {
            if (!gp_canvas_load_from_file(head.context, workspace_path.c_str())) {
                std::cerr << "Failed to load canvas workspace: " << workspace_path << "\n";
            }
        }
    }
    resources.canvas_width_hint = head.width;
    resources.canvas_height_hint = head.height;
    gp_canvas_set_thread_manager_paused(head.context, 0);
    resources.collection_state.has_history[index] = true;
    return true;
}

void destroy_canvas_heads(FrontendResources& resources) {
    for (auto& head : resources.canvas_heads) {
        destroy_canvas_head(head);
    }
}

bool initialize_canvas_heads(FrontendResources& resources) {
    size_t target = resources.collection_state.active_index;
    if (target >= kMaxCanvasHeads) {
        target = 0;
    }
    if (!ensure_canvas_head_initialized(resources, target)) {
        std::cerr << "Failed to create initial canvas head\n";
        return false;
    }
    resources.active_canvas_index = target;
    resources.collection_state.active_index = target;
    return true;
}

bool ensure_canvas_texture(FrontendResources& resources, int width, int height) {
    if (!resources.renderer || width <= 0 || height <= 0) return false;
    if (resources.canvas_texture &&
        width == resources.canvas_texture_width &&
        height == resources.canvas_texture_height) {
        return true;
    }
    if (resources.canvas_texture) {
        SDL_DestroyTexture(resources.canvas_texture);
        resources.canvas_texture = nullptr;
    }
    resources.canvas_texture = SDL_CreateTexture(
        resources.renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        width,
        height
    );
    if (!resources.canvas_texture) return false;
    // SDL3 texture querying differs across backends; skip detailed format
    // probe here — we already requested RGBA32 and will assume that layout.
    resources.canvas_texture_width = width;
    resources.canvas_texture_height = height;
    return true;
}

bool update_active_canvas_texture(FrontendResources& resources) {
    size_t index = resources.active_canvas_index;
    if (index >= resources.canvas_heads.size()) return false;
    auto& head = resources.canvas_heads[index];
    if (!head.context || head.width <= 0 || head.height <= 0) return false;
    if (!ensure_canvas_texture(resources, head.width, head.height)) return false;
    size_t buf_len = static_cast<size_t>(head.width) * head.height * 4u;
    if (head.pixels.size() < buf_len) head.pixels.resize(buf_len);
    if (!gp_canvas_raster_rgba(head.context, head.pixels.data(), static_cast<int32_t>(buf_len))) {
        return false;
    }
    int pitch = head.width * 4;
    SDL_UpdateTexture(resources.canvas_texture, nullptr, head.pixels.data(), pitch);
    return true;
}

bool initialize_window(FrontendResources& resources) {
    // Assume the caller (main) is responsible for primary SDL initialization.
    // Only initialize the video/events subsystems here if they aren't already.
    {
        const Uint32 required = SDL_INIT_VIDEO | SDL_INIT_EVENTS;
        if ((SDL_WasInit(required) & required) != required) {
            if (!SDL_Init(required)) {
                std::cerr << "SDL_Init (video/events) failed: " << SDL_GetError() << "\n";
                return false;
            }
        }
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    // Cast combined window flags to the integer type expected by SDL3 API.
    uint32_t win_flags = static_cast<uint32_t>(SDL_WINDOW_OPENGL) |
                         static_cast<uint32_t>(SDL_WINDOW_RESIZABLE) |
                         static_cast<uint32_t>(SDL_WINDOW_HIGH_PIXEL_DENSITY);
    // SDL3 offers a bounds-based CreateWindow overload; construct a rect and call that variant.
    SDL_Rect bounds;
    bounds.x = SDL_WINDOWPOS_CENTERED;
    bounds.y = SDL_WINDOWPOS_CENTERED;
    bounds.w = 1280;
    bounds.h = 720;
    // SDL3 CreateWindow signature is (title, width, height, flags).
    resources.window = SDL_CreateWindow(
        "Canvas Frontend Shell",
        bounds.w,
        bounds.h,
        win_flags
    );
    // position after creation if centering is desired
    if (resources.window) {
        SDL_SetWindowPosition(resources.window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    }

    if (!resources.window) {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
        return false;
    }

    resources.gl_context = SDL_GL_CreateContext(resources.window);
    if (!resources.gl_context) {
        std::cerr << "SDL_GL_CreateContext failed: " << SDL_GetError() << "\n";
        return false;
    }

    if (!SDL_GL_MakeCurrent(resources.window, resources.gl_context)) {
        std::cerr << "SDL_GL_MakeCurrent failed: " << SDL_GetError() << "\n";
        return false;
    }

    SDL_GL_SetSwapInterval(1);
    // SDL3 renderer creation signature changed to accept an optional driver name.
    // Pass nullptr to use the default accelerated renderer.
    resources.renderer = SDL_CreateRenderer(resources.window, /*driver_name=*/nullptr);
    if (!resources.renderer) {
        std::cerr << "SDL_CreateRenderer failed: " << SDL_GetError() << "\n";
        return false;
    }

    // Create a simple GL shader program to ensure GL functions are available.
    // We load GL entry points with SDL_GL_GetProcAddress at runtime.
    auto create_test_shader = [](FrontendResources& res) {
        using PFN_glCreateShader = GLuint(*)(GLenum);
        using PFN_glShaderSource = void(*)(GLuint, GLsizei, const GLchar* const*, const GLint*);
        using PFN_glCompileShader = void(*)(GLuint);
        using PFN_glGetShaderiv = void(*)(GLuint, GLenum, GLint*);
        using PFN_glGetShaderInfoLog = void(*)(GLuint, GLsizei, GLsizei*, GLchar*);
        using PFN_glCreateProgram = GLuint(*)();
        using PFN_glAttachShader = void(*)(GLuint, GLuint);
        using PFN_glLinkProgram = void(*)(GLuint);
        using PFN_glGetProgramiv = void(*)(GLuint, GLenum, GLint*);
        using PFN_glGetProgramInfoLog = void(*)(GLuint, GLsizei, GLsizei*, GLchar*);
        using PFN_glDeleteShader = void(*)(GLuint);
        using PFN_glUseProgram = void(*)(GLuint);

        PFN_glCreateShader glCreateShaderFP = (PFN_glCreateShader)SDL_GL_GetProcAddress("glCreateShader");
        PFN_glShaderSource glShaderSourceFP = (PFN_glShaderSource)SDL_GL_GetProcAddress("glShaderSource");
        PFN_glCompileShader glCompileShaderFP = (PFN_glCompileShader)SDL_GL_GetProcAddress("glCompileShader");
        PFN_glGetShaderiv glGetShaderivFP = (PFN_glGetShaderiv)SDL_GL_GetProcAddress("glGetShaderiv");
        PFN_glGetShaderInfoLog glGetShaderInfoLogFP = (PFN_glGetShaderInfoLog)SDL_GL_GetProcAddress("glGetShaderInfoLog");
        PFN_glCreateProgram glCreateProgramFP = (PFN_glCreateProgram)SDL_GL_GetProcAddress("glCreateProgram");
        PFN_glAttachShader glAttachShaderFP = (PFN_glAttachShader)SDL_GL_GetProcAddress("glAttachShader");
        PFN_glLinkProgram glLinkProgramFP = (PFN_glLinkProgram)SDL_GL_GetProcAddress("glLinkProgram");
        PFN_glGetProgramiv glGetProgramivFP = (PFN_glGetProgramiv)SDL_GL_GetProcAddress("glGetProgramiv");
        PFN_glGetProgramInfoLog glGetProgramInfoLogFP = (PFN_glGetProgramInfoLog)SDL_GL_GetProcAddress("glGetProgramInfoLog");
        PFN_glDeleteShader glDeleteShaderFP = (PFN_glDeleteShader)SDL_GL_GetProcAddress("glDeleteShader");
        PFN_glUseProgram glUseProgramFP = (PFN_glUseProgram)SDL_GL_GetProcAddress("glUseProgram");

        if (!glCreateShaderFP || !glShaderSourceFP || !glCompileShaderFP || !glGetShaderivFP || !glGetShaderInfoLogFP ||
            !glCreateProgramFP || !glAttachShaderFP || !glLinkProgramFP || !glGetProgramivFP || !glGetProgramInfoLogFP ||
            !glDeleteShaderFP || !glUseProgramFP) {
            std::cerr << "GL entry points not available; skipping shader test\n";
            return;
        }

        const char* vs_src = "#version 330 core\nlayout(location=0) in vec2 aPos; void main(){gl_Position=vec4(aPos,0.0,1.0);}";
        const char* fs_src = "#version 330 core\nout vec4 FragColor; void main(){FragColor=vec4(1.0,0.0,1.0,1.0);}";

        GLuint vs = glCreateShaderFP(GL_VERTEX_SHADER);
        glShaderSourceFP(vs, 1, &vs_src, nullptr);
        glCompileShaderFP(vs);
        GLint compiled = 0; glGetShaderivFP(vs, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            char buf[512]; GLsizei len = 0; glGetShaderInfoLogFP(vs, sizeof(buf), &len, buf); std::cerr << "Vertex shader compile failed: " << buf << "\n";
            glDeleteShaderFP(vs);
            return;
        }
        GLuint fs = glCreateShaderFP(GL_FRAGMENT_SHADER);
        glShaderSourceFP(fs, 1, &fs_src, nullptr);
        glCompileShaderFP(fs);
        glGetShaderivFP(fs, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            char buf[512]; GLsizei len = 0; glGetShaderInfoLogFP(fs, sizeof(buf), &len, buf); std::cerr << "Fragment shader compile failed: " << buf << "\n";
            glDeleteShaderFP(vs); glDeleteShaderFP(fs);
            return;
        }
        GLuint prog = glCreateProgramFP();
        glAttachShaderFP(prog, vs);
        glAttachShaderFP(prog, fs);
        glLinkProgramFP(prog);
        GLint linked = 0; glGetProgramivFP(prog, GL_LINK_STATUS, &linked);
        if (!linked) {
            char buf[512]; GLsizei len = 0; glGetProgramInfoLogFP(prog, sizeof(buf), &len, buf); std::cerr << "Program link failed: " << buf << "\n";
            glDeleteShaderFP(vs); glDeleteShaderFP(fs);
            return;
        }
        // Keep program bound for potential runtime usage.
        glUseProgramFP(prog);
        // Do not delete program now so it remains usable.
    };
    create_test_shader(resources);
    return true;
}

void cleanup(FrontendResources& resources) {
    destroy_canvas_heads(resources);
    if (resources.canvas_texture) {
        SDL_DestroyTexture(resources.canvas_texture);
        resources.canvas_texture = nullptr;
    }
    if (resources.renderer) {
        SDL_DestroyRenderer(resources.renderer);
        resources.renderer = nullptr;
    }
    if (resources.gl_context) {
        SDL_GL_DestroyContext(resources.gl_context);
    }
    if (resources.window) {
        SDL_DestroyWindow(resources.window);
    }
    SDL_Quit();
}

struct InputFilters {
    bool keyboard_enabled = true;
    bool mouse_enabled = true;
    bool gamepad_enabled = false;
    bool keyboard_filter_active = false;
    std::unordered_set<SDL_Scancode> keyboard_keys;
    bool mouse_filter_active = false;
    std::unordered_set<Uint8> mouse_buttons;
    bool gamepad_filter_active = false;
    std::unordered_set<SDL_GamepadButton> gamepad_buttons;
};

struct CanvasSwitchCombo {
    bool hunting = false;
    std::chrono::steady_clock::time_point start_time{};
    std::string digits;
    bool clone_from_active = false;
};

struct GamepadInfo {
    SDL_Gamepad* gamepad = nullptr;
    SDL_JoystickID instance_id = -1;
};

class InputDispatcher {
public:
    InputDispatcher(FrontendResources& resources, InputFilters filters);
    InputDispatcher(const InputDispatcher&) = delete;
    InputDispatcher& operator=(const InputDispatcher&) = delete;
    ~InputDispatcher();

    void handle_event(const SDL_Event& event);
    void update();
    void refresh_layout_from_window();
    // notify dispatcher that window size changed (window coords)
    void on_window_resized(int window_w, int window_h);

private:
    bool should_forward_keyboard(const SDL_KeyboardEvent& event) const;
    bool should_forward_mouse_button(const SDL_MouseButtonEvent& event) const;
    bool handle_combo_key(const SDL_KeyboardEvent& event);
    void maybe_complete_combo();
    void start_combo(bool clone_requested = false);
    void complete_combo();
    int scancode_to_digit(SDL_Scancode scan) const;
    void switch_to_canvas(size_t index);
    void add_gamepad(int device_index);
    void remove_gamepad(SDL_JoystickID instance_id);
    void update_modifier_state(const SDL_KeyboardEvent& event, bool pressed);

    FrontendResources& resources_;
    InputFilters filters_;
    std::vector<GamepadInfo> gamepads_;
    CanvasSwitchCombo combo_;
    bool shift_down_ = false;
    bool ctrl_down_ = false;
    bool alt_down_ = false;
    // track previous mouse float position to derive deltas when SDL3 doesn't provide deltas
    float prev_mouse_x_ = 0.0f;
    float prev_mouse_y_ = 0.0f;
    bool have_prev_mouse_ = false;
};

InputDispatcher::InputDispatcher(FrontendResources& resources, InputFilters filters)
    : resources_(resources), filters_(std::move(filters)) {}

InputDispatcher::~InputDispatcher() {
    for (auto& info : gamepads_) {
        if (info.gamepad) {
            SDL_CloseGamepad(info.gamepad);
            info.gamepad = nullptr;
        }
    }
}

void InputDispatcher::handle_event(const SDL_Event& event) {
    maybe_complete_combo();
    switch (event.type) {
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP: {
            update_modifier_state(event.key, event.type == SDL_EVENT_KEY_DOWN);
            // Ctrl+Shift+K toggles the kpath raster source.
            if (event.type == SDL_EVENT_KEY_DOWN &&
                event.key.scancode == SDL_SCANCODE_K &&
                ctrl_down_ && shift_down_) {
                if (!resources_.kpath_mode) {
                    const uint32_t target_w = static_cast<uint32_t>(std::max(1, resources_.layout.canvas_w > 0 ? resources_.layout.canvas_w : resources_.canvas_width_hint));
                    const uint32_t target_h = static_cast<uint32_t>(std::max(1, resources_.layout.canvas_h > 0 ? resources_.layout.canvas_h : resources_.canvas_height_hint));
                    if (ensure_kpath_ready(resources_, target_w, target_h)) {
                        resources_.kpath_mode = true;
                        resources_.canvas_texture_width = static_cast<int>(target_w);
                        resources_.canvas_texture_height = static_cast<int>(target_h);
                        ensure_canvas_texture(resources_, static_cast<int>(target_w), static_cast<int>(target_h));
                        refresh_layout_from_window();
                        std::cout << "Kpath raster mode enabled (Ctrl+Shift+K toggles, Space toggles scatter)\n";
                    } else {
                        std::cerr << "Failed to initialize kpath raster mode\n";
                    }
                } else {
                    resources_.kpath_mode = false;
                    // Restore canvas texture sizing to the active canvas dimensions.
                    auto& head = resources_.canvas_heads[resources_.active_canvas_index];
                    resources_.canvas_texture_width = head.width;
                    resources_.canvas_texture_height = head.height;
                    refresh_layout_from_window();
                    std::cout << "Kpath raster mode disabled; returning to canvas rendering\n";
                }
                return;
            }
            if (resources_.kpath_mode && event.type == SDL_EVENT_KEY_DOWN &&
                event.key.scancode == SDL_SCANCODE_SPACE) {
                resources_.kpath_demo.use_scatter = !resources_.kpath_demo.use_scatter;
                std::cout << "Kpath raster scatter mode "
                          << (resources_.kpath_demo.use_scatter ? "enabled" : "disabled") << "\n";
                return;
            }
            if (resources_.kpath_mode && event.type == SDL_EVENT_KEY_DOWN) {
                switch (event.key.scancode) {
                    case SDL_SCANCODE_LEFTBRACKET:
                        resources_.kpath_demo.heat_decay_scale = std::max(0.01f, resources_.kpath_demo.heat_decay_scale * 0.9f);
                        std::cout << "Kpath heat decay scale: " << resources_.kpath_demo.heat_decay_scale << "\n";
                        return;
                    case SDL_SCANCODE_RIGHTBRACKET:
                        resources_.kpath_demo.heat_decay_scale = std::min(10.0f, resources_.kpath_demo.heat_decay_scale * 1.1f);
                        std::cout << "Kpath heat decay scale: " << resources_.kpath_demo.heat_decay_scale << "\n";
                        return;
                    case SDL_SCANCODE_MINUS:
                        resources_.kpath_demo.energy_scale = std::max(0.01f, resources_.kpath_demo.energy_scale * 0.9f);
                        std::cout << "Kpath energy scale: " << resources_.kpath_demo.energy_scale << "\n";
                        return;
                    case SDL_SCANCODE_EQUALS:
                        resources_.kpath_demo.energy_scale = std::min(50.0f, resources_.kpath_demo.energy_scale * 1.1f);
                        std::cout << "Kpath energy scale: " << resources_.kpath_demo.energy_scale << "\n";
                        return;
                    case SDL_SCANCODE_O:
                        resources_.kpath_demo.exposure_gain = std::max(0.01f, resources_.kpath_demo.exposure_gain * 0.9f);
                        std::cout << "Kpath exposure gain: " << resources_.kpath_demo.exposure_gain << "\n";
                        return;
                    case SDL_SCANCODE_P:
                        resources_.kpath_demo.exposure_gain = std::min(20.0f, resources_.kpath_demo.exposure_gain * 1.1f);
                        std::cout << "Kpath exposure gain: " << resources_.kpath_demo.exposure_gain << "\n";
                        return;
                    case SDL_SCANCODE_K:
                        resources_.kpath_demo.film_decay = std::max(0.3f, resources_.kpath_demo.film_decay - 0.02f);
                        std::cout << "Kpath film decay: " << resources_.kpath_demo.film_decay << "\n";
                        return;
                    case SDL_SCANCODE_L:
                        resources_.kpath_demo.film_decay = std::min(0.99f, resources_.kpath_demo.film_decay + 0.02f);
                        std::cout << "Kpath film decay: " << resources_.kpath_demo.film_decay << "\n";
                        return;
                    case SDL_SCANCODE_H:
                        resources_.kpath_demo.heat_intensity = std::max(0.05f, resources_.kpath_demo.heat_intensity * 0.9f);
                        std::cout << "Kpath heat intensity: " << resources_.kpath_demo.heat_intensity << "\n";
                        return;
                    case SDL_SCANCODE_J:
                        resources_.kpath_demo.heat_intensity = std::min(10.0f, resources_.kpath_demo.heat_intensity * 1.1f);
                        std::cout << "Kpath heat intensity: " << resources_.kpath_demo.heat_intensity << "\n";
                        return;
                    case SDL_SCANCODE_COMMA:
                        resources_.kpath_demo.heat_max_kelvin = std::max(1500.0f, resources_.kpath_demo.heat_max_kelvin - 250.0f);
                        std::cout << "Kpath heat max kelvin: " << resources_.kpath_demo.heat_max_kelvin << "\n";
                        return;
                    case SDL_SCANCODE_PERIOD:
                        resources_.kpath_demo.heat_max_kelvin = std::min(12000.0f, resources_.kpath_demo.heat_max_kelvin + 250.0f);
                        std::cout << "Kpath heat max kelvin: " << resources_.kpath_demo.heat_max_kelvin << "\n";
                        return;
                    case SDL_SCANCODE_N:
                        resources_.kpath_demo.normalize_energy = !resources_.kpath_demo.normalize_energy;
                        std::cout << "Kpath normalize energy: " << (resources_.kpath_demo.normalize_energy ? "on" : "off") << "\n";
                        return;
                    case SDL_SCANCODE_M:
                        resources_.kpath_demo.normalize_heat = !resources_.kpath_demo.normalize_heat;
                        std::cout << "Kpath normalize heat: " << (resources_.kpath_demo.normalize_heat ? "on" : "off") << "\n";
                        return;
                    default:
                        break;
                }
            }
            if (resources_.kpath_mode) break;
            if (handle_combo_key(event.key)) {
                return;
            }
            if (!filters_.keyboard_enabled) break;
            if (!should_forward_keyboard(event.key)) break;
            auto* ctx = resources_.canvas_heads[resources_.active_canvas_index].context;
            if (!ctx) break;
            // SDL3 keyboard event exposes `scancode` on `event.key`.
            int sc = event.key.scancode;
            int keycode = 0;
            int mods = SDL_GetModState();
            gp_canvas_on_key(
                ctx,
                keycode,
                sc,
                event.type == SDL_EVENT_KEY_DOWN ? 1 : 0,
                mods
            );
            break;
        }
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP: {
            if (resources_.kpath_mode) break;
            if (!filters_.mouse_enabled) break;
            if (!should_forward_mouse_button(event.button)) break;
            auto* ctx = resources_.canvas_heads[resources_.active_canvas_index].context;
            if (!ctx) break;
            // Prefer high-precision stored coords when available; fall back to integer event values.
            float fx = resources_.layout.mouse_fx;
            float fy = resources_.layout.mouse_fy;
            float fdx = resources_.layout.mouse_fdx;
            float fdy = resources_.layout.mouse_fdy;
            if (!have_prev_mouse_) {
                fx = static_cast<float>(event.button.x);
                fy = static_cast<float>(event.button.y);
                fdx = 0.0f; fdy = 0.0f;
            }
            float lx = fx - static_cast<float>(resources_.layout.canvas_x);
            float ly = fy - static_cast<float>(resources_.layout.canvas_y);
            int button = static_cast<int>(event.button.button);
            if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                fprintf(stderr, "[DBG] frontend on_mouse_down ctx=%p lx=%f,ly=%f fdx=%f,fdy=%f btn=%d\n", (void*)ctx, lx, ly, fdx, fdy, button);
                gp_canvas_on_mouse_down(ctx, lx, ly, fdx, fdy, button);
            } else {
                fprintf(stderr, "[DBG] frontend on_mouse_up ctx=%p lx=%f,ly=%f fdx=%f,fdy=%f btn=%d\n", (void*)ctx, lx, ly, fdx, fdy, button);
                gp_canvas_on_mouse_up(ctx, lx, ly, fdx, fdy, button);
            }
            break;
        }
        case SDL_EVENT_MOUSE_MOTION: {
            if (resources_.kpath_mode) break;
            if (!filters_.mouse_enabled) break;
            auto* ctx = resources_.canvas_heads[resources_.active_canvas_index].context;
            if (!ctx) break;
            // SDL3 exposes floating-point mouse coords; capture those and derive deltas.
            float mx = static_cast<float>(event.motion.x);
            float my = static_cast<float>(event.motion.y);
            float mdx = 0.0f, mdy = 0.0f;
            if (have_prev_mouse_) {
                mdx = mx - prev_mouse_x_;
                mdy = my - prev_mouse_y_;
            }
            prev_mouse_x_ = mx;
            prev_mouse_y_ = my;
            have_prev_mouse_ = true;
            // store high-precision values into layout for consumers
            resources_.layout.mouse_fx = mx;
            resources_.layout.mouse_fy = my;
            resources_.layout.mouse_fdx = mdx;
            resources_.layout.mouse_fdy = mdy;
            // translate to canvas-local float coords and forward deltas
            float lx = mx - static_cast<float>(resources_.layout.canvas_x);
            float ly = my - static_cast<float>(resources_.layout.canvas_y);
            MOUSE_MOVE_DEBUGF("[DBG] frontend on_mouse_move ctx=%p lx=%f,ly=%f mdx=%f,mdy=%f\n", (void*)ctx, lx, ly, mdx, mdy);
            gp_canvas_on_mouse_move(ctx, lx, ly, mdx, mdy);
            break;
        }
        case SDL_EVENT_MOUSE_WHEEL: {
            if (resources_.kpath_mode) break;
            if (!filters_.mouse_enabled) break;
            auto* ctx = resources_.canvas_heads[resources_.active_canvas_index].context;
            if (!ctx) break;
            // Use stored high-precision cursor position when available
            float mx = resources_.layout.mouse_fx;
            float my = resources_.layout.mouse_fy;
            if (!have_prev_mouse_) {
                float fx = 0.0f, fy = 0.0f;
                SDL_GetMouseState(&fx, &fy);
                mx = fx;
                my = fy;
            }
            float mdx = resources_.layout.mouse_fdx;
            float mdy = resources_.layout.mouse_fdy;
            float lx = mx - static_cast<float>(resources_.layout.canvas_x);
            float ly = my - static_cast<float>(resources_.layout.canvas_y);
            float scroll = static_cast<float>(event.wheel.y);
            fprintf(stderr, "[DBG] frontend on_mouse_scroll ctx=%p lx=%f,ly=%f mdx=%f,mdy=%f scroll=%f\n", (void*)ctx, lx, ly, mdx, mdy, scroll);
            gp_canvas_on_mouse_scroll(ctx, lx, ly, mdx, mdy, scroll);
            break;
        }
        case SDL_EVENT_GAMEPAD_ADDED:
            add_gamepad(event.gdevice.which);
            break;
        case SDL_EVENT_GAMEPAD_REMOVED:
            remove_gamepad(event.gdevice.which);
            break;
        case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
        case SDL_EVENT_GAMEPAD_BUTTON_UP:
            if (!filters_.gamepad_enabled) break;
            if (!filters_.gamepad_filter_active ||
                filters_.gamepad_buttons.count(static_cast<SDL_GamepadButton>(event.gbutton.button)) > 0) {
                if (filters_.gamepad_filter_active) {
                    std::cout << "Gamepad button "
                              << static_cast<int>(event.gbutton.button)
                              << (event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN ? " down" : " up")
                              << "\n";
                }
            }
            break;
        default:
            break;
    }
}

void InputDispatcher::update_modifier_state(const SDL_KeyboardEvent& event, bool pressed) {
    switch (event.scancode) {
        case SDL_SCANCODE_LSHIFT:
        case SDL_SCANCODE_RSHIFT:
            shift_down_ = pressed;
            break;
        case SDL_SCANCODE_LCTRL:
        case SDL_SCANCODE_RCTRL:
            ctrl_down_ = pressed;
            break;
        case SDL_SCANCODE_LALT:
        case SDL_SCANCODE_RALT:
            alt_down_ = pressed;
            break;
        default:
            break;
    }
}

void InputDispatcher::refresh_layout_from_window() {
    if (!resources_.window) return;
    int ww = 0, wh = 0;
    SDL_GetWindowSize(resources_.window, &ww, &wh);
    on_window_resized(ww, wh);
}

bool InputDispatcher::should_forward_keyboard(const SDL_KeyboardEvent& event) const {
    if (!filters_.keyboard_filter_active) return true;
    return filters_.keyboard_keys.count(event.scancode) > 0;
}

bool InputDispatcher::should_forward_mouse_button(const SDL_MouseButtonEvent& event) const {
    if (!filters_.mouse_filter_active) return true;
    return filters_.mouse_buttons.count(event.button) > 0;
}

void InputDispatcher::update() {
    maybe_complete_combo();
}

void InputDispatcher::on_window_resized(int window_w, int window_h) {
    auto& layout = resources_.layout;
    layout.window_w = window_w;
    layout.window_h = window_h;
    int cw = resources_.canvas_texture_width > 0 ? resources_.canvas_texture_width : resources_.canvas_width_hint;
    int ch = resources_.canvas_texture_height > 0 ? resources_.canvas_texture_height : resources_.canvas_height_hint;
    layout.canvas_w = cw;
    layout.canvas_h = ch;
    layout.canvas_x = (window_w - cw) / 2;
    layout.canvas_y = 0;
    layout.left_space = layout.canvas_x;
    layout.right_space = window_w - (layout.canvas_x + layout.canvas_w);
    layout.bottom_space = window_h - (layout.canvas_y + layout.canvas_h);

    // Update each canvas head and notify canvas contexts of new logical size
    if (!resources_.kpath_mode) {
        for (auto& head : resources_.canvas_heads) {
            head.width = cw;
            head.height = ch;
            head.pixels.assign(static_cast<size_t>(std::max(1, cw)) * std::max(1, ch) * 4, 0u);
            if (head.context) {
                gp_canvas_set_size(head.context, cw, ch);
            }
        }
    }
    // Ensure the SDL texture matches the chosen canvas size
    ensure_canvas_texture(resources_, cw, ch);
}

void InputDispatcher::maybe_complete_combo() {
    if (!combo_.hunting) return;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - combo_.start_time).count();
    if (elapsed >= kCanvasSwitchWindowMs) {
        complete_combo();
    }
}

bool InputDispatcher::handle_combo_key(const SDL_KeyboardEvent& event) {
    if (event.type != SDL_EVENT_KEY_DOWN) return false;
    int digit = scancode_to_digit(event.scancode);
    if (event.scancode == SDL_SCANCODE_TAB &&
        shift_down_ && ctrl_down_) {
        if (!combo_.hunting) {
            start_combo(alt_down_);
        } else {
            combo_.start_time = std::chrono::steady_clock::now();
            combo_.clone_from_active = alt_down_;
        }
        return true;
    }
    if (combo_.hunting && digit >= 0) {
        if (combo_.digits.size() < 3) {
            combo_.digits.push_back(static_cast<char>('0' + digit));
            std::cout << "Canvas hunt saw digit " << digit << " (scancode " << event.scancode << " mods=" << SDL_GetModState() << ")\n";
        }
        return true;
    }
    if (combo_.hunting) {
        std::cout << "Canvas hunt ignored key sc=" << event.scancode << " mod=" << SDL_GetModState() << "\n";
    }
    return false;
}

void InputDispatcher::start_combo(bool clone_requested) {
    combo_.hunting = true;
    combo_.start_time = std::chrono::steady_clock::now();
    combo_.digits.clear();
    combo_.clone_from_active = clone_requested;
    std::cout << "Canvas hunt started\n";
}

void InputDispatcher::complete_combo() {
    if (!combo_.hunting) return;
    combo_.hunting = false;
    bool clone_requested = combo_.clone_from_active;
    combo_.clone_from_active = false;
    if (combo_.digits.empty()) return;
    int parsed = 0;
    for (char ch : combo_.digits) {
        parsed = parsed * 10 + (ch - '0');
        if (parsed > static_cast<int>(kMaxCanvasHeads)) {
            break;
        }
    }
    combo_.digits.clear();
    if (parsed <= 0 || parsed > static_cast<int>(kMaxCanvasHeads)) {
        return;
    }
    std::cout << "Canvas hunt complete target=" << parsed << "\n";
    size_t target_index = static_cast<size_t>(parsed - 1);
    if (clone_requested &&
        target_index < kMaxCanvasHeads &&
        target_index != resources_.active_canvas_index) {
        const auto& collection = resources_.collection_state;
        const std::string& source_path = collection.workspace_paths[collection.active_index];
        const std::string& target_path = collection.workspace_paths[target_index];
        if (!source_path.empty() && !target_path.empty()) {
            std::error_code ec;
            if (std::filesystem::exists(source_path, ec) && !ec) {
                std::filesystem::copy_file(source_path, target_path,
                    std::filesystem::copy_options::overwrite_existing, ec);
                if (!ec) {
                    resources_.collection_state.has_history[target_index] = true;
                    std::cout << "Cloned canvas " << collection.active_index
                              << " -> " << target_index << "\n";
                } else {
                    std::cerr << "Failed to clone canvas workspace: " << ec.message() << "\n";
                }
            }
        }
    }
    switch_to_canvas(target_index);
}

int InputDispatcher::scancode_to_digit(SDL_Scancode scan) const {
    switch (scan) {
        case SDL_SCANCODE_0: case SDL_SCANCODE_KP_0: return 0;
        case SDL_SCANCODE_1: case SDL_SCANCODE_KP_1: return 1;
        case SDL_SCANCODE_2: case SDL_SCANCODE_KP_2: return 2;
        case SDL_SCANCODE_3: case SDL_SCANCODE_KP_3: return 3;
        case SDL_SCANCODE_4: case SDL_SCANCODE_KP_4: return 4;
        case SDL_SCANCODE_5: case SDL_SCANCODE_KP_5: return 5;
        case SDL_SCANCODE_6: case SDL_SCANCODE_KP_6: return 6;
        case SDL_SCANCODE_7: case SDL_SCANCODE_KP_7: return 7;
        case SDL_SCANCODE_8: case SDL_SCANCODE_KP_8: return 8;
        case SDL_SCANCODE_9: case SDL_SCANCODE_KP_9: return 9;
        default: return -1;
    }
}

void InputDispatcher::switch_to_canvas(size_t index) {
    if (index >= kMaxCanvasHeads) return;
    if (!ensure_canvas_head_initialized(resources_, index)) {
        std::cerr << "Failed to prepare canvas head " << index << "\n";
        return;
    }
    resources_.active_canvas_index = index;
    resources_.collection_state.active_index = index;
    std::cout << "Switched to canvas head " << index << "\n";
}

void InputDispatcher::add_gamepad(int device_index) {
    SDL_Gamepad* gp = SDL_OpenGamepad(device_index);
    if (!gp) {
        std::cerr << "Gamepad open failed: " << SDL_GetError() << "\n";
        return;
    }
    SDL_Joystick* joy = SDL_GetGamepadJoystick(gp);
    if (!joy) {
        SDL_CloseGamepad(gp);
        return;
    }
    SDL_JoystickID instance_id = SDL_GetJoystickID(joy);
    if (instance_id < 0) {
        SDL_CloseGamepad(gp);
        return;
    }
    for (const auto& info : gamepads_) {
        if (info.instance_id == instance_id) {
            SDL_CloseGamepad(gp);
            return;
        }
    }
    gamepads_.push_back({gp, instance_id});
    std::cout << "Gamepad connected (instance " << instance_id << ")\n";
}

void InputDispatcher::remove_gamepad(SDL_JoystickID instance_id) {
    auto it = std::find_if(gamepads_.begin(), gamepads_.end(),
        [instance_id](const GamepadInfo& info) { return info.instance_id == instance_id; });
    if (it == gamepads_.end()) return;
    if (it->gamepad) {
        SDL_CloseGamepad(it->gamepad);
    }
    std::cout << "Gamepad disconnected (instance " << instance_id << ")\n";
    gamepads_.erase(it);
}

static std::string trim(std::string value) {
    auto begin = std::find_if(value.begin(), value.end(), [](unsigned char ch) { return !std::isspace(ch); });
    auto end = std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base();
    if (begin < end) {
        value = std::string(begin, end);
    } else {
        value.clear();
    }
    return value;
}

static std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

static std::vector<std::string> split_comma_list(const std::string& input) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;
    while (std::getline(stream, token, ',')) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

static bool parse_positive_int(const std::string& text, int& value) {
    value = 0;
    bool seen = false;
    for (char ch : text) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
        seen = true;
        value = value * 10 + (ch - '0');
    }
    return seen;
}

static std::optional<Uint8> mouse_button_from_token(const std::string& raw) {
    static const std::unordered_map<std::string, Uint8> kMouseButtonMap = {
        {"left", SDL_BUTTON_LEFT},
        {"right", SDL_BUTTON_RIGHT},
        {"middle", SDL_BUTTON_MIDDLE},
        {"x1", SDL_BUTTON_X1},
        {"x2", SDL_BUTTON_X2},
    };
    auto lower = to_lower(raw);
    if (auto it = kMouseButtonMap.find(lower); it != kMouseButtonMap.end()) {
        return it->second;
    }
        if (lower.rfind("button", 0) == 0) {
        int parsed = 0;
        if (parse_positive_int(lower.substr(6), parsed)) {
            if (parsed > 0 && parsed <= 8) {
                return static_cast<Uint8>(parsed);
            }
        }
    }
    int parsed = 0;
    if (parse_positive_int(lower, parsed) && parsed > 0 && parsed <= 8) {
        return static_cast<Uint8>(parsed);
    }
    return std::nullopt;
}

static std::optional<SDL_GamepadButton> gamepad_button_from_token(const std::string& raw) {
    static const std::unordered_map<std::string, SDL_GamepadButton> kGamepadButtonMap = {
        {"a", SDL_GAMEPAD_BUTTON_SOUTH},
        {"b", SDL_GAMEPAD_BUTTON_EAST},
        {"x", SDL_GAMEPAD_BUTTON_WEST},
        {"y", SDL_GAMEPAD_BUTTON_NORTH},
        {"back", SDL_GAMEPAD_BUTTON_BACK},
        {"guide", SDL_GAMEPAD_BUTTON_GUIDE},
        {"start", SDL_GAMEPAD_BUTTON_START},
        {"leftstick", SDL_GAMEPAD_BUTTON_LEFT_STICK},
        {"rightstick", SDL_GAMEPAD_BUTTON_RIGHT_STICK},
        {"leftshoulder", SDL_GAMEPAD_BUTTON_LEFT_SHOULDER},
        {"rightshoulder", SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER},
        {"dpup", SDL_GAMEPAD_BUTTON_DPAD_UP},
        {"dpdown", SDL_GAMEPAD_BUTTON_DPAD_DOWN},
        {"dpleft", SDL_GAMEPAD_BUTTON_DPAD_LEFT},
        {"dpright", SDL_GAMEPAD_BUTTON_DPAD_RIGHT},
        {"misc1", SDL_GAMEPAD_BUTTON_MISC1},
        {"paddle1", SDL_GAMEPAD_BUTTON_RIGHT_PADDLE1},
        {"paddle2", SDL_GAMEPAD_BUTTON_LEFT_PADDLE1},
        {"paddle3", SDL_GAMEPAD_BUTTON_RIGHT_PADDLE2},
        {"paddle4", SDL_GAMEPAD_BUTTON_LEFT_PADDLE2},
        {"touchpad", SDL_GAMEPAD_BUTTON_TOUCHPAD},
    };
    auto lower = to_lower(raw);
    if (auto it = kGamepadButtonMap.find(lower); it != kGamepadButtonMap.end()) {
        return it->second;
    }
    return std::nullopt;
}

static void apply_keyboard_filter(InputFilters& filters, const std::string& raw_value) {
    auto tokens = split_comma_list(raw_value);
    if (tokens.empty()) return;
    for (const auto& token : tokens) {
        if (token.empty()) continue;
        auto normalized = to_lower(token);
        if (normalized == "any" || normalized == "all") {
            filters.keyboard_filter_active = false;
            filters.keyboard_keys.clear();
            return;
        }
        if (normalized == "none") {
            filters.keyboard_filter_active = true;
            filters.keyboard_keys.clear();
            continue;
        }
        SDL_Scancode code = SDL_GetScancodeFromName(token.c_str());
        if (code == SDL_SCANCODE_UNKNOWN) {
            std::cerr << "Unknown keyboard scancode '" << token << "'\n";
            continue;
        }
        filters.keyboard_filter_active = true;
        filters.keyboard_keys.insert(code);
    }
}

static void apply_mouse_filter(InputFilters& filters, const std::string& raw_value) {
    auto tokens = split_comma_list(raw_value);
    if (tokens.empty()) return;
    for (const auto& token : tokens) {
        if (token.empty()) continue;
        auto normalized = to_lower(token);
        if (normalized == "any" || normalized == "all") {
            filters.mouse_filter_active = false;
            filters.mouse_buttons.clear();
            return;
        }
        if (normalized == "none") {
            filters.mouse_filter_active = true;
            filters.mouse_buttons.clear();
            continue;
        }
        if (auto button = mouse_button_from_token(token)) {
            filters.mouse_filter_active = true;
            filters.mouse_buttons.insert(*button);
        } else {
            std::cerr << "Unknown mouse button '" << token << "'\n";
        }
    }
}

static void apply_gamepad_filter(InputFilters& filters, const std::string& raw_value) {
    auto tokens = split_comma_list(raw_value);
    if (tokens.empty()) return;
    for (const auto& token : tokens) {
        if (token.empty()) continue;
        auto normalized = to_lower(token);
        if (normalized == "any" || normalized == "all") {
            filters.gamepad_filter_active = false;
            filters.gamepad_buttons.clear();
            return;
        }
        if (normalized == "none") {
            filters.gamepad_filter_active = true;
            filters.gamepad_buttons.clear();
            continue;
        }
        if (auto button = gamepad_button_from_token(token)) {
            filters.gamepad_filter_active = true;
            filters.gamepad_buttons.insert(*button);
        } else {
            std::cerr << "Unknown gamepad button '" << token << "'\n";
        }
    }
}

InputFilters parse_input_filters(int argc, char** argv) {
    InputFilters filters;
    const std::string keyboard_prefix = "--keyboard-filter=";
    const std::string mouse_prefix = "--mouse-filter=";
    const std::string gamepad_prefix = "--gamepad-filter=";
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--disable-keyboard") {
            filters.keyboard_enabled = false;
        } else if (arg == "--disable-mouse") {
            filters.mouse_enabled = false;
        } else if (arg == "--enable-gamepad") {
            filters.gamepad_enabled = true;
        } else if (arg == "--disable-gamepad") {
            filters.gamepad_enabled = false;
        } else if (arg.rfind(keyboard_prefix, 0) == 0) {
            apply_keyboard_filter(filters, arg.substr(keyboard_prefix.size()));
        } else if (arg.rfind(mouse_prefix, 0) == 0) {
            apply_mouse_filter(filters, arg.substr(mouse_prefix.size()));
        } else if (arg.rfind(gamepad_prefix, 0) == 0) {
            apply_gamepad_filter(filters, arg.substr(gamepad_prefix.size()));
        }
    }
    return filters;
}

struct FrontendOptions {
    bool show_help = false;
    bool verbose = false;
    std::string device = "auto"; // cpu, cuda, auto
    int width = kDefaultCanvasWidth;
    int height = kDefaultCanvasHeight;
    std::string workspace_path;
    // timing overrides: negative => not set
    double thread_delay = -1.0;
    double gui_delay = -1.0;

    struct CanvasConfig {
        int idx = 0;
        double dt = 1.0/60.0;
        double ratio = 1.0;
        bool free_spin = false;
        double display_interval = -1.0; // if >=0 set interval mode
        double display_ratio = -1.0; // if >=0 set ratio mode
        bool pause = false; // if true, set canvas thread manager paused
    };
    std::vector<CanvasConfig> canvas_configs;
};

static void print_usage(const char* progname) {
    std::cout << "Usage: " << progname << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help             Show this help message and exit\n";
    std::cout << "  --device=<cpu|cuda|auto>  Choose torch device (default: auto)\n";
    std::cout << "  --width=<pixels>       Initial canvas width (default: 1000)\n";
    std::cout << "  --height=<pixels>      Initial canvas height (default: 720)\n";
    std::cout << "  --workspace=<path>     Path for primary canvas workspace file\n";
    std::cout << "  --verbose              Enable verbose logging\n";
    std::cout << "  --thread-delay=<s>     Global thread loop sleep (seconds, default ~0.016)\n";
    std::cout << "  --gui-delay=<s>        Global GUI delay used by tick controller (seconds)\n";
    std::cout << "  --canvas-config=IDX:DT:RATIO:FREE  Per-canvas thread config (FREE=0/1)\n";
    std::cout << "  --canvas-display-interval=IDX:SEC  Set display interval for canvas IDX\n";
    std::cout << "  --canvas-display-ratio=IDX:RATIO   Set display ratio for canvas IDX\n";
    std::cout << "  --pause-canvas=IDX     Start canvas IDX with thread manager paused\n";
    std::cout << "  --disable-keyboard     Disable keyboard input (existing option)\n";
    std::cout << "  --disable-mouse        Disable mouse input (existing option)\n";
    std::cout << "  --enable-gamepad       Enable gamepad input (existing option)\n";
}

FrontendOptions parse_frontend_options(int argc, char** argv) {
    FrontendOptions opts;
    const std::string device_prefix = "--device=";
    const std::string width_prefix = "--width=";
    const std::string height_prefix = "--height=";
    const std::string workspace_prefix = "--workspace=";
    const std::string thread_delay_prefix = "--thread-delay=";
    const std::string gui_delay_prefix = "--gui-delay=";
    const std::string canvas_config_prefix = "--canvas-config="; // IDX:DT:RATIO:FREE
    const std::string canvas_display_interval_prefix = "--canvas-display-interval="; // IDX:SEC
    const std::string canvas_display_ratio_prefix = "--canvas-display-ratio="; // IDX:RATIO
    const std::string pause_canvas_prefix = "--pause-canvas="; // IDX
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            opts.show_help = true;
        } else if (arg == "--verbose") {
            opts.verbose = true;
        } else if (arg.rfind(device_prefix, 0) == 0) {
            opts.device = arg.substr(device_prefix.size());
        } else if (arg.rfind(width_prefix, 0) == 0) {
            opts.width = std::stoi(arg.substr(width_prefix.size()));
        } else if (arg.rfind(height_prefix, 0) == 0) {
            opts.height = std::stoi(arg.substr(height_prefix.size()));
        } else if (arg.rfind(workspace_prefix, 0) == 0) {
            opts.workspace_path = arg.substr(workspace_prefix.size());
        } else if (arg.rfind(thread_delay_prefix, 0) == 0) {
            opts.thread_delay = std::stod(arg.substr(thread_delay_prefix.size()));
        } else if (arg.rfind(gui_delay_prefix, 0) == 0) {
            opts.gui_delay = std::stod(arg.substr(gui_delay_prefix.size()));
        } else if (arg.rfind(canvas_config_prefix, 0) == 0) {
            std::string v = arg.substr(canvas_config_prefix.size());
            // expected IDX:DT:RATIO:FREE
            FrontendOptions::CanvasConfig cc;
            std::replace(v.begin(), v.end(), ':', ' ');
            std::istringstream iss(v);
            int free_i = 0;
            if ((iss >> cc.idx >> cc.dt >> cc.ratio >> free_i)) {
                cc.free_spin = (free_i != 0);
                opts.canvas_configs.push_back(cc);
            }
        } else if (arg.rfind(canvas_display_interval_prefix, 0) == 0) {
            std::string v = arg.substr(canvas_display_interval_prefix.size());
            std::replace(v.begin(), v.end(), ':', ' ');
            std::istringstream iss(v);
            FrontendOptions::CanvasConfig cc;
            if (iss >> cc.idx >> cc.display_interval) {
                opts.canvas_configs.push_back(cc);
            }
        } else if (arg.rfind(canvas_display_ratio_prefix, 0) == 0) {
            std::string v = arg.substr(canvas_display_ratio_prefix.size());
            std::replace(v.begin(), v.end(), ':', ' ');
            std::istringstream iss(v);
            FrontendOptions::CanvasConfig cc;
            if (iss >> cc.idx >> cc.display_ratio) {
                opts.canvas_configs.push_back(cc);
            }
        } else if (arg.rfind(pause_canvas_prefix, 0) == 0) {
            std::string v = arg.substr(pause_canvas_prefix.size());
            int idx = std::stoi(v);
            FrontendOptions::CanvasConfig cc; cc.idx = idx; cc.pause = true; opts.canvas_configs.push_back(cc);
        }
    }
    return opts;
}

static std::string canvas_workspace_path_for_index(size_t index) {
    if (index == 0) {
        return std::string(kDefaultCanvasWorkspacePath);
    }
    std::ostringstream ss;
    ss << "canvas_workspace_" << index << ".txt";
    return ss.str();
}

static void initialize_collection_state(CanvasCollectionState& state) {
    for (size_t i = 0; i < kMaxCanvasHeads; ++i) {
        state.workspace_paths[i] = canvas_workspace_path_for_index(i);
        state.has_history[i] = false;
    }
    state.active_index = 0;
}

static bool load_frontend_collection_state(CanvasCollectionState& state) {
    std::ifstream ifs(kFrontendCanvasStatePath);
    if (!ifs.good()) {
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;
        if (keyword == "ACTIVE") {
            size_t idx = 0;
            if (iss >> idx && idx < kMaxCanvasHeads) {
                state.active_index = idx;
            }
        } else if (keyword == "CANVAS") {
            size_t idx = 0;
            if (!(iss >> idx) || idx >= kMaxCanvasHeads) continue;
            state.workspace_paths[idx] = canvas_workspace_path_for_index(idx);
            state.has_history[idx] = true;
        }
    }
    return true;
}

static void save_frontend_collection_state(const CanvasCollectionState& state) {
    std::ofstream ofs(kFrontendCanvasStatePath, std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Unable to write frontend canvas state to " << kFrontendCanvasStatePath << "\n";
        return;
    }
    size_t active = state.active_index;
    if (active >= kMaxCanvasHeads) {
        active = 0;
    }
    ofs << "ACTIVE " << active << "\n";
    for (size_t i = 0; i < kMaxCanvasHeads; ++i) {
        if (!state.has_history[i]) continue;
        ofs << "CANVAS " << i << " " << canvas_workspace_path_for_index(i) << "\n";
    }
}

void pump_events(bool& running, InputDispatcher& dispatcher) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_EVENT_QUIT:
                running = false;
                break;
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                running = false;
                break;
            case SDL_EVENT_WINDOW_RESIZED:
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                // Let dispatcher query the window directly and update layout
                dispatcher.refresh_layout_from_window();
                break;
            default:
                break;
        }
        dispatcher.handle_event(event);
    }
}

void render_frame(FrontendResources& resources, CanvasTickController& controller) {
    if (!resources.renderer) return;
    bool rendered_kpath = false;
    if (resources.kpath_mode) {
        const uint32_t target_w = static_cast<uint32_t>(std::max(1, resources.canvas_texture_width > 0 ? resources.canvas_texture_width : resources.canvas_width_hint));
        const uint32_t target_h = static_cast<uint32_t>(std::max(1, resources.canvas_texture_height > 0 ? resources.canvas_texture_height : resources.canvas_height_hint));
        const bool kpath_ready = ensure_kpath_ready(resources, target_w, target_h);
        const bool texture_ready = kpath_ready &&
            ensure_canvas_texture(resources, static_cast<int>(target_w), static_cast<int>(target_h));
        const bool frame_ready = texture_ready &&
            render_kpath_demo_frame(resources.kpath_demo, resources.kpath_pixels);
        if (frame_ready) {
            int pitch = static_cast<int>(target_w) * 4;
            SDL_UpdateTexture(resources.canvas_texture, nullptr, resources.kpath_pixels.data(), pitch);
            rendered_kpath = true;
        } else {
            if (!kpath_ready) {
                std::cerr << "Kpath raster frame failed: ensure_kpath_ready returned false (w="
                          << target_w << ", h=" << target_h << ")\n";
            } else if (!texture_ready) {
                std::cerr << "Kpath raster frame failed: ensure_canvas_texture returned false (w="
                          << target_w << ", h=" << target_h << ")\n";
            } else {
                std::cerr << "Kpath raster frame failed: render_kpath_demo_frame returned false (w="
                          << target_w << ", h=" << target_h << ")\n";
            }
            std::cerr << "Kpath raster frame failed; disabling kpath mode\n";
            resources.kpath_mode = false;
            auto& head = resources.canvas_heads[resources.active_canvas_index];
            resources.canvas_texture_width = head.width;
            resources.canvas_texture_height = head.height;
        }
    }
    auto now = std::chrono::steady_clock::now();
    double now_seconds = std::chrono::duration<double>(now.time_since_epoch()).count();
    size_t active_idx = resources.active_canvas_index;
    if (!rendered_kpath) {
        bool should_update = controller.should_display(active_idx, now_seconds);
        if (should_update) {
            update_active_canvas_texture(resources);
            controller.mark_displayed(active_idx, now_seconds);
        }
    }

    // Clear full window background
    SDL_SetRenderDrawColor(resources.renderer, 20, 24, 34, 255);
    SDL_RenderClear(resources.renderer);
    // Render canvas texture at computed top-centered position
    if (resources.canvas_texture) {
        SDL_Rect dsti;
        dsti.x = resources.layout.canvas_x;
        dsti.y = resources.layout.canvas_y;
        dsti.w = resources.layout.canvas_w > 0 ? resources.layout.canvas_w : resources.canvas_texture_width;
        dsti.h = resources.layout.canvas_h > 0 ? resources.layout.canvas_h : resources.canvas_texture_height;
        SDL_FRect dstf;
        dstf.x = static_cast<float>(dsti.x);
        dstf.y = static_cast<float>(dsti.y);
        dstf.w = static_cast<float>(dsti.w);
        dstf.h = static_cast<float>(dsti.h);
        SDL_RenderTexture(resources.renderer, resources.canvas_texture, nullptr, &dstf);
    }
    SDL_RenderPresent(resources.renderer);
}

void prepare_torch(FrontendResources& resources, const FrontendOptions& opts) {
    // Decide device based on options and availability. The CUDA probe itself
    // runs inside mem_backend_torch.cpp (the sole translation unit that
    // includes <torch/torch.h>); this file only ever sees a plain int/string.
    std::string dev = to_lower(opts.device);
    bool cuda_available = gp_mem_backend_torch_cuda_available() != 0;
    if (dev == "cpu") {
        resources.torch_device_name = "cpu";
    } else if (dev == "cuda") {
        if (cuda_available) {
            resources.torch_device_name = "cuda";
        } else {
            std::cerr << "Requested CUDA but CUDA not available; falling back to CPU\n";
            resources.torch_device_name = "cpu";
        }
    } else { // auto or unknown -> prefer CUDA if available
        resources.torch_device_name = cuda_available ? "cuda" : "cpu";
    }

    // Apply display/workspace hints from options
    resources.canvas_width_hint = opts.width;
    resources.canvas_height_hint = opts.height;
    if (!opts.workspace_path.empty()) {
        resources.collection_state.workspace_paths[resources.collection_state.active_index] = opts.workspace_path;
    }

    if (opts.verbose) {
        std::cout << "Using Torch device: " << resources.torch_device_name << "\n";
        std::cout << "Canvas size hint: " << resources.canvas_width_hint << "x" << resources.canvas_height_hint << "\n";
        if (!opts.workspace_path.empty()) std::cout << "Workspace: " << opts.workspace_path << "\n";
    }

    // initialize layout with hints; actual window size will update later
    resources.layout.canvas_w = resources.canvas_width_hint;
    resources.layout.canvas_h = resources.canvas_height_hint;
}

}  // namespace

int main(int argc, char** argv) {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_GAMEPAD)) {
        std::cerr << "SDL_Init failed: " << SDL_GetError();
// Optionally include Win32 formatted/system-localized error text. Define
// `NODUS_PRINT_WIN32_ERROR_TEXT` at build time to enable; otherwise only
// SDL_GetError() is printed (safer for localization/CI environments).
#if defined(_WIN32) && defined(NODUS_PRINT_WIN32_ERROR_TEXT)
        DWORD err = GetLastError();
        if (err != 0) {
            char buf[512] = {0};
            FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                           nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), buf, sizeof(buf), nullptr);
            std::cerr << " (GetLastError=" << err << ": " << buf << ")";
        }
#endif
        std::cerr << "\n";
        return EXIT_FAILURE;
    }
    // SDL3 uses SDL_INIT_GAMEPAD to enable gamepad support; explicit
    // event state calls like SDL_GameControllerEventState are deprecated.
    FrontendOptions options = parse_frontend_options(argc, argv);
    if (options.show_help) {
        print_usage(argv && argv[0] ? argv[0] : "frontend_shell");
        return EXIT_SUCCESS;
    }

    InputFilters input_filters = parse_input_filters(argc, argv);

    FrontendResources resources;
    initialize_collection_state(resources.collection_state);
    if (load_frontend_collection_state(resources.collection_state)) {
        std::cout << "Restored frontend canvas collection (active="
                  << resources.collection_state.active_index << ")\n";
    }
    prepare_torch(resources, options);

    if (!initialize_window(resources)) {
        cleanup(resources);
        return EXIT_FAILURE;
    }

    if (!initialize_canvas_heads(resources)) {
        cleanup(resources);
        return EXIT_FAILURE;
    }

    std::cout << "Torch backend: " << resources.torch_device_name << "\n";

    constexpr float kCanvasStepDt = 1.0f / 60.0f;
    CanvasTickController tick_controller(resources);
    // apply CLI overrides if provided
    if (options.thread_delay >= 0.0) tick_controller.set_global_thread_delay(options.thread_delay);
    else tick_controller.set_global_thread_delay(kCanvasStepDt);
    if (options.gui_delay >= 0.0) tick_controller.set_global_gui_delay(options.gui_delay);
    else tick_controller.set_global_gui_delay(kCanvasStepDt);

    // apply per-canvas configs
    for (const auto& cc : options.canvas_configs) {
        if (cc.idx < 0 || cc.idx >= static_cast<int>(resources.canvas_heads.size())) continue;
        tick_controller.set_canvas_thread_config(static_cast<size_t>(cc.idx), cc.dt, cc.ratio, cc.free_spin);
        if (cc.display_interval >= 0.0) tick_controller.set_canvas_display_interval(static_cast<size_t>(cc.idx), cc.display_interval);
        if (cc.display_ratio >= 0.0) tick_controller.set_canvas_display_ratio(static_cast<size_t>(cc.idx), cc.display_ratio);
    }

    tick_controller.start();

    {
        InputDispatcher dispatcher(resources, input_filters);
        // initialize layout from current window size
        if (resources.window) {
            int ww = 0, wh = 0;
            SDL_GetWindowSize(resources.window, &ww, &wh);
            dispatcher.on_window_resized(ww, wh);
        }
            // apply pause flags to canvases if requested
            for (const auto& cc : options.canvas_configs) {
                if (cc.pause && cc.idx >= 0 && cc.idx < static_cast<int>(resources.canvas_heads.size())) {
                    auto& head = resources.canvas_heads[cc.idx];
                    if (head.context) gp_canvas_set_thread_manager_paused(head.context, 1);
                }
            }
            bool running = true;
        while (running) {
            pump_events(running, dispatcher);
            dispatcher.update();
            tick_controller.record_render(kCanvasStepDt);
            render_frame(resources, tick_controller);
            if (resources.kpath_mode) {
                SDL_Delay(0);
            } else {
                double gui_delay = tick_controller.gui_delay_seconds();
                if (gui_delay > 0.0) {
                    Uint32 delay_ms = static_cast<Uint32>(std::max(0.0, gui_delay * 1000.0));
                    SDL_Delay(delay_ms);
                } else {
                    SDL_Delay(0);
                }
            }
        }
        tick_controller.stop();
    }

    save_frontend_collection_state(resources.collection_state);

    cleanup(resources);
    return EXIT_SUCCESS;
}
