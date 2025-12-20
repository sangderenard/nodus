#include "canvas_abi.h"
#include "rope_sim.h"
#include "table_abi.h"
#include "raytrace_2d.h"
#include "stage_abi.h"
#include "text_render_helper.h"
#include "thread_manager.h"
#include "labels.h"

#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iterator>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <chrono>

// C-linkage prototypes for spline drawer implemented in table_abi.cpp
extern "C" void table_draw_rope_curve_blend_colored(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, const float* hues, int hue_count, int samples_per_segment, float hue_intensity);
extern "C" void table_draw_rope_curve_blend(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, uint8_t cr, uint8_t cg, uint8_t cb, uint8_t ca, int samples_per_segment);

// local minimal Color and draw helpers (self-contained)
struct Color { uint8_t r=0,g=0,b=0,a=255; };

struct ContactLight {
    Color col{0,0,0,0};
    float intensity = 0.0f;
    bool valid = false;
};

static inline void blend_pixel(uint8_t* dst, uint8_t sr, uint8_t sg, uint8_t sb, uint8_t sa) {
    if (!dst) return;
    float a = sa / 255.0f;
    if (a <= 0.0f) return;
    float inv = 1.0f - a;
    float dr = dst[0] / 255.0f;
    float dg = dst[1] / 255.0f;
    float db = dst[2] / 255.0f;
    float da = dst[3] / 255.0f;
    float srf = sr / 255.0f;
    float sgf = sg / 255.0f;
    float sbf = sb / 255.0f;
    float outa = a + da * inv;
    if (outa <= 0.0f) { dst[0]=dst[1]=dst[2]=dst[3]=0; return; }
    float out_r = (srf * a + dr * da * inv) / outa;
    float out_g = (sgf * a + dg * da * inv) / outa;
    float out_b = (sbf * a + db * da * inv) / outa;
    dst[0] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_r)) * 255.0f));
    dst[1] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_g)) * 255.0f));
    dst[2] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_b)) * 255.0f));
    dst[3] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, outa)) * 255.0f));
}

static void memset_rect(uint8_t* img, int w, int h, int pitch, int x0, int y0, int rw, int rh, Color c) {
    if (!img) return;
    int x1 = std::min(w, x0 + rw);
    int y1 = std::min(h, y0 + rh);
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    if (x0 >= x1 || y0 >= y1) return;
    for (int y = y0; y < y1; ++y) {
        uint8_t* row = img + y * pitch + x0 * 4;
        for (int x = x0; x < x1; ++x) {
            row[0] = c.r; row[1] = c.g; row[2] = c.b; row[3] = c.a; row += 4;
        }
    }
}

static void draw_circle(uint8_t* img, int w, int h, int pitch, int cx, int cy, int r, Color c) {
    if (!img) return;
    const int r2 = r * r;
    for (int dy = -r; dy <= r; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= h) continue;
        for (int dx = -r; dx <= r; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= w) continue;
            if (dx*dx + dy*dy > r2) continue;
            uint8_t* p = img + y * pitch + x * 4;
            p[0] = c.r; p[1] = c.g; p[2] = c.b; p[3] = c.a;
        }
    }
}

// draw soft blended filled circle (used for rope blobs)
static void draw_blob_blend(uint8_t* img, int w, int h, int pitch, int cx, int cy, int radius, Color c) {
    if (!img || radius <= 0) return;
    int r = radius;
    int r2 = r * r;
    int y0 = std::max(0, cy - r);
    int y1 = std::min(h - 1, cy + r);
    for (int y = y0; y <= y1; ++y) {
        int dy = y - cy;
        int dx_limit = static_cast<int>(std::floor(std::sqrt((double)r2 - double(dy * dy))));
        int x0 = std::max(0, cx - dx_limit);
        int x1 = std::min(w - 1, cx + dx_limit);
        for (int x = x0; x <= x1; ++x) {
            int dx = x - cx;
            int d2 = dx * dx + dy * dy;
            if (d2 > r2) continue;
            float t = 1.0f - (std::sqrt((float)d2) / (float)r);
            uint8_t sa = static_cast<uint8_t>(std::lround(c.a * t));
            uint8_t* dst = img + y * pitch + x * 4;
            blend_pixel(dst, c.r, c.g, c.b, sa);
        }
    }
}

static void draw_segment_kernel_glow(uint8_t* img, int w, int h, int pitch,
    float x1, float y1, float x2, float y2, float radius, Color c, float alpha_scale = 1.0f) {
    if (!img || radius <= 0.0f) return;
    float dx = x2 - x1;
    float dy = y2 - y1;
    float len2 = dx * dx + dy * dy;
    float r = radius;
    float rplus = r + 2.0f;
    int minx = static_cast<int>(std::floor(std::min(x1, x2) - rplus));
    int maxx = static_cast<int>(std::ceil(std::max(x1, x2) + rplus));
    int miny = static_cast<int>(std::floor(std::min(y1, y2) - rplus));
    int maxy = static_cast<int>(std::ceil(std::max(y1, y2) + rplus));
    minx = std::max(minx, 0);
    miny = std::max(miny, 0);
    maxx = std::min(maxx, w - 1);
    maxy = std::min(maxy, h - 1);

    float sigma = r * 0.5f;
    float inv2sig2 = 1.0f / (2.0f * sigma * sigma);
    for (int y = miny; y <= maxy; ++y) {
        for (int x = minx; x <= maxx; ++x) {
            float t = 0.0f;
            if (len2 > 1e-6f) {
                float vx = float(x) - x1;
                float vy = float(y) - y1;
                float proj = (vx * dx + vy * dy) / len2;
                t = std::clamp(proj, 0.0f, 1.0f);
            }
            float cxp = x1 + dx * t;
            float cyp = y1 + dy * t;
            float lx = float(x) - cxp;
            float ly = float(y) - cyp;
            float dist2 = lx * lx + ly * ly;
            if (dist2 > r * r) continue;
            float gauss = std::exp(-dist2 * inv2sig2);
            uint8_t sa = static_cast<uint8_t>(std::lround(float(c.a) * gauss * alpha_scale));
            if (sa == 0) continue;
            uint8_t* dst = img + y * pitch + x * 4;
            blend_pixel(dst, c.r, c.g, c.b, sa);
        }
    }
}

static float build_uniform_samples_along_polyline(const float* verts, int count, float spacing,
    std::vector<std::pair<float,float>>& out_samples, std::vector<float>& out_dist) {
    out_samples.clear();
    out_dist.clear();
    if (!verts || count < 2) return 0.0f;

    std::vector<float> prefix(static_cast<size_t>(count), 0.0f);
    for (int i = 1; i < count; ++i) {
        float dx = verts[i * 2 + 0] - verts[(i - 1) * 2 + 0];
        float dy = verts[i * 2 + 1] - verts[(i - 1) * 2 + 1];
        prefix[static_cast<size_t>(i)] = prefix[static_cast<size_t>(i - 1)] + std::sqrt(dx * dx + dy * dy);
    }
    float total_len = prefix.back();
    if (total_len <= 1e-5f) return 0.0f;

    out_samples.reserve(static_cast<size_t>(std::ceil(total_len / spacing)) + 2);
    out_dist.reserve(out_samples.capacity());

    int seg_idx = 0;
    float d = 0.0f;
    while (d < total_len) {
        while (seg_idx + 1 < count && prefix[static_cast<size_t>(seg_idx + 1)] < d) ++seg_idx;
        float seg_start = prefix[static_cast<size_t>(seg_idx)];
        float seg_end = prefix[static_cast<size_t>(seg_idx + 1)];
        float t = (seg_end > seg_start) ? (d - seg_start) / (seg_end - seg_start) : 0.0f;
        float x0 = verts[seg_idx * 2 + 0];
        float y0 = verts[seg_idx * 2 + 1];
        float x1 = verts[(seg_idx + 1) * 2 + 0];
        float y1 = verts[(seg_idx + 1) * 2 + 1];
        float sx = x0 + (x1 - x0) * t;
        float sy = y0 + (y1 - y0) * t;
        out_samples.emplace_back(sx, sy);
        out_dist.push_back(d);
        d += spacing;
    }
    out_samples.emplace_back(verts[(count - 1) * 2 + 0], verts[(count - 1) * 2 + 1]);
    out_dist.push_back(total_len);
    return total_len;
}

static void compute_light_mix_arrays(const std::vector<float>& mid_dist, float total_len, float ia0, float ib0, float decay,
    std::vector<float>& out_total, std::vector<float>& out_t) {
    size_t n = mid_dist.size();
    out_total.assign(n, 0.0f);
    out_t.assign(n, 0.0f);
    if (n == 0 || total_len <= 1e-6f) return;

    Eigen::Map<const Eigen::ArrayXf> d(mid_dist.data(), static_cast<int>(n));
    Eigen::ArrayXf u = d / total_len;
    Eigen::ArrayXf ia = ia0 * (-decay * u).exp();
    Eigen::ArrayXf ib = ib0 * (-decay * (1.0f - u)).exp();
    Eigen::ArrayXf total = ia + ib;
    Eigen::ArrayXf t = ib / (total + 1e-6f);

    Eigen::Map<Eigen::ArrayXf>(out_total.data(), static_cast<int>(n)) = total;
    Eigen::Map<Eigen::ArrayXf>(out_t.data(), static_cast<int>(n)) = t;
}

static void draw_rope_light_falloff(uint8_t* img, int w, int h, int pitch,
    const float* verts, int count, ContactLight a, ContactLight b, float decay, int radius) {
    if (!img || !verts || count < 2) return;
    if (!a.valid && !b.valid) return;
    if (!a.valid && b.valid) { a.col = b.col; a.intensity = 0.0f; }
    if (!b.valid && a.valid) { b.col = a.col; b.intensity = 0.0f; }

    float spacing = std::max(1.0f, float(radius) * 0.6f);
    std::vector<std::pair<float,float>> samples;
    std::vector<float> sample_dist;
    float total_len = build_uniform_samples_along_polyline(verts, count, spacing, samples, sample_dist);
    if (total_len <= 1e-5f) return;

    std::vector<float> mid_dist;
    mid_dist.reserve(samples.size() > 1 ? samples.size() - 1 : 0);
    for (size_t i = 0; i + 1 < sample_dist.size(); ++i) {
        mid_dist.push_back(0.5f * (sample_dist[i] + sample_dist[i + 1]));
    }

    std::vector<float> totals;
    std::vector<float> mix_t;
    compute_light_mix_arrays(mid_dist, total_len, a.intensity, b.intensity, decay, totals, mix_t);

    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float total = totals[i];
        if (total <= 1e-4f) continue;
        float t = mix_t[i];
        Color col;
        col.r = static_cast<uint8_t>(std::lround(float(a.col.r) * (1.0f - t) + float(b.col.r) * t));
        col.g = static_cast<uint8_t>(std::lround(float(a.col.g) * (1.0f - t) + float(b.col.g) * t));
        col.b = static_cast<uint8_t>(std::lround(float(a.col.b) * (1.0f - t) + float(b.col.b) * t));
        col.a = static_cast<uint8_t>(std::lround(220.0f * std::clamp(total, 0.0f, 1.0f)));
        draw_segment_kernel_glow(img, w, h, pitch, samples[i].first, samples[i].second, samples[i + 1].first, samples[i + 1].second, float(radius), col, 1.0f);
    }
}
// Per-canvas drag state (moved into the canvas object to avoid global map)
struct DragState {
    int dragging = 0;
    int module = -1;
    int offx = 0;
    int offy = 0;
    int panning = 0;
    int pan_last_x = 0;
    int pan_last_y = 0;
};

struct GP_CanvasContextImpl;
extern "C" int gp_canvas_add_edge_with_type(GP_CanvasContext* ctx_, const GP_CanvasEdgeDesc* desc, int type_id);

// Lightweight chat color struct used by canvas chat visuals
struct ChatCol { uint8_t r=40, g=40, b=40, a=255; };

// Forward-declare chat bg callback so it can be assigned earlier in the file
static void chat_bg_callback(void* user, int module_idx, int width, int height, uint8_t* out_rgba, int32_t out_pitch);

struct MolexLayoutInfo {
    int rows = 0;
    int cols = 0;
    std::vector<uint32_t> hashes;
};

static uint32_t hash_connector(int module_idx, bool is_input, int pin_number, int grid_row, int grid_col) {
    uint32_t h = static_cast<uint32_t>(module_idx + 1);
    h = (h * 0x9E3779B1u) ^ static_cast<uint32_t>(pin_number * 0x165667B1u);
    h ^= static_cast<uint32_t>(grid_row * 31) << 8;
    h ^= static_cast<uint32_t>(grid_col * 17) << 16;
    if (!is_input) h ^= 0xA5A5A5A5u;
    h = (h * 0x9E3779B1u) ^ 0xC2B2AE35u;
    return h ? h : 1;
}

static MolexLayoutInfo make_molex_layout(int module_idx, bool is_input, int count) {
    MolexLayoutInfo out;
    if (count <= 0) return out;
    const int max_cols = 8;
    float best_score = std::numeric_limits<float>::infinity();
    int best_cols = std::min(count, max_cols);
    int best_rows = (count + best_cols - 1) / best_cols;
    for (int cols = 1; cols <= std::min(count, max_cols); ++cols) {
        int rows = (count + cols - 1) / cols;
        int waste = cols * rows - count;
        float ratio = float(cols) / float(std::max(1, rows));
        float score = float(waste) + 4.0f * std::fabs(ratio - 3.0f);
        if (score < best_score || (std::fabs(score - best_score) < 1e-4f && cols > best_cols)) {
            best_score = score;
            best_cols = cols;
            best_rows = rows;
        }
    }
    out.cols = best_cols;
    out.rows = best_rows;
    out.hashes.reserve(static_cast<size_t>(count));
    for (int idx = 0; idx < count; ++idx) {
        int grid_row = idx / best_cols;
        int grid_col = idx % best_cols;
        uint32_t h = hash_connector(module_idx, is_input, idx + 1, grid_row, grid_col);
        out.hashes.push_back(h);
    }
    return out;
}


// Minimal internal canvas context implementation
struct GP_CanvasContextImpl {
    int width=0, height=0;
    std::vector<GP_CanvasModuleDesc> modules;
    // internal edge info bundles the desc, rope index, and per-edge hues
    struct EdgeInfo {
        GP_CanvasEdgeDesc desc;
        int rope_idx = -1;
        int type_id = 0; // 0 == wildcard / untyped
        std::vector<float> hues;
        float hue_intensity = 0.0f;
    };
    std::vector<EdgeInfo> edges;
    // optional attached table per module (aligned with `modules` by index)
    std::vector<GP_TableContext*> module_tables;
    std::vector<int> module_table_owned; // 1 if canvas should destroy
    // Optional stage per module (for light-sim demo modules).
    std::vector<GP_StageContext*> module_stages;
    std::vector<int> module_stage_owned; // 1 if canvas should destroy
    std::vector<int> module_is_stage;    // 1 if this module is a stage module
    std::vector<GP_TableImage> module_stage_images; // live image descriptors per stage module
    // per-module cached completed stage RGBA buffers (tight RGBA8 rows)
    std::vector<std::vector<uint8_t>> module_stage_cache_rgba;
    std::vector<int> module_stage_cache_w;
    std::vector<int> module_stage_cache_h;
    std::vector<int> module_stage_cache_pitch;
    std::vector<std::unique_ptr<std::mutex>> module_stage_cache_mu;
    std::vector<uint8_t> module_stage_integrator_mode; // integrator lock per stage module
    std::vector<std::vector<float>> module_stage_integrator_accum; // per-stage integrator accumulators
    struct ModuleBg {
        GP_CanvasModuleBgFn cb = nullptr;
        void* user = nullptr;
        int mode = 0;
        int rays = 512;
        int reflections = 2;
        float blur_sigma = 2.0f;
        int oversample = 2;
        float temporal_decay = 0.92f;
        float temporal_max = 1.0f;
        uint32_t temporal_frame = 0;
        float ray_exposure = 1.0f;
        float ray_bounce_decay = 0.75f;
        float ray_air_decay = 0.0025f;
        uint8_t table_alpha = 255;
        uint8_t table_alpha_ray = 48;
        int header_margin_px = 0; // reserved vertical strip for overlay UI (e.g., stage table header)
        Raytrace2D* ray = nullptr;
        std::vector<float> accum;
        std::vector<float> temporal;
        std::vector<uint8_t> scratch;
        std::vector<uint8_t> layer;
    };
    std::vector<ModuleBg> module_bg;
    // per-module IO counts (inputs, outputs) exposed in the control bar
    std::vector<int> module_io_in_count;
    std::vector<int> module_io_out_count;
    std::vector<std::vector<int>> module_io_input_rows;
    std::vector<std::vector<int>> module_io_output_rows;
    std::vector<MolexLayoutInfo> module_input_layout;
    std::vector<MolexLayoutInfo> module_output_layout;
    std::vector<std::vector<ModuleIORow>> module_io_rows;
    std::vector<std::vector<ModuleIORow>> module_table_rows;
    std::vector<std::unordered_map<int, std::vector<float>>> module_stack_snapshots;
    std::vector<std::vector<ModuleToolKind>> module_tool_stack;
    // optional per-module key-recorder state pointer
    std::vector<void*> module_key_recorder_state;
    std::vector<ModuleInputState> module_input_state;
    // per-module lightweight chat state (used to visually confirm rope traffic)
    std::vector<std::string> module_chat_text;
    std::vector<ChatCol> module_chat_color;
    std::vector<int> module_chat_ttl; // frames remaining to show chat highlight
    // cable style/hues
    int jacket_px = 4;
    int jacket_border = 2;
    std::vector<float> hues;
    float hue_intensity = 0.0f;
    // selection state for click-to-connect behavior. `anchor_x/anchor_y` are
    // pixel coordinates (canvas space) for the selected contact when the
    // selection originates from a table hitbox; they remain -1 when unset.
    struct Sel { int module = -1; int contact_idx = -1; int left = -1; int anchor_x = -1; int anchor_y = -1; } selected;
    // transient module index used by root-table action dispatch
    int dispatch_module_idx = -1;
    // per-canvas drag state (moved here to avoid a global map)
    DragState drag;
    // provisional rope index while user is selecting a contact and moving the mouse
    int prospective_rope_idx = -1;
    // rope simulation tuning parameters and UI bar height
    int rope_bar_h = 28; // extra bar above control bar
    int sim_segs = 8;
    float sim_slack = 0.0f;
    int sim_iters = 8;
    float sim_damping = 0.86f;
    float sim_maxforce = 800.0f;
    // tool selection state: separate groups (exclusive within group)
    // canvas tool group: 0 = select, 1 = new table, 2 = edge mode, 3 = new stage
    int selected_tool_canvas = 0;
    // edge tool group: 0=create, 1=destroy, 2=on-change, 3=continuous (-1 = none)
    int selected_tool_edge = 0;
    int edge_order_value = 0;
    int edge_order_tool_active = 0;
    // table tool group: 0 = neutral, 1 = select, 2 = menu
    int selected_tool_table = 0;
    bool tool_menu_open = false;
    int io_attachment_count = 1;
    // which module (if any) has keyboard/focus for table editing
    int focused_module = -1;
    // registered host windows (opaque pointers)
    std::vector<void*> windows;
    // mapping from window pointer to stable node id for backing graph
    std::unordered_map<void*, int> window_node_ids;
    // next unique node id for graph nodes
    int next_node_id = 1;
    // per-module node id (aligned with `modules`) or -1 if none
    std::vector<int> module_node_id;
    // graph node/contract representation
    struct NodeContract {
        int node_id = -1;
        int module_idx = -1; // which module this node belongs to (-1 if none)
        std::vector<int> input_types; // supported input type ids
        std::vector<int> output_types; // supported output type ids
    };
    std::vector<NodeContract> nodes;
    // UI control bar height (in canvas-local pixels)
    int control_bar_h = 28;
    // viewport offset (world origin visible at (0,0) in screen space)
    int offset_x = 0;
    int offset_y = 0;
    int scroll_x_needed = 0;
    int scroll_y_needed = 0;
    // optional table container (non-owning unless marked)
    GP_TableContext* container_table = nullptr;
    int container_table_owned = 0;
    int root_actions_installed = 0;
    // autosave parameters (path may be empty to disable)
    std::string autosave_path;
    double autosave_interval_s = 0.0;
    double autosave_accum_s = 0.0;
    std::unique_ptr<ThreadManager> thread_mgr;
    GP_CanvasContextImpl(int w, int h): width(w), height(h) {}
};

// (global drag map removed; each canvas has its own DragState member)

static GP_CanvasContextImpl* g_canvas_context_singleton = nullptr;

const std::vector<ModuleIORow>* canvas_get_module_io_rows(int module_idx) {
    if (!g_canvas_context_singleton) return nullptr;
    if (module_idx < 0 || module_idx >= static_cast<int>(g_canvas_context_singleton->module_io_rows.size())) return nullptr;
    return &g_canvas_context_singleton->module_io_rows[module_idx];
}

bool canvas_get_module_input_state(int module_idx, ModuleInputState* out_state) {
    if (!g_canvas_context_singleton || !out_state) return false;
    if (module_idx < 0 || module_idx >= static_cast<int>(g_canvas_context_singleton->module_input_state.size())) return false;
    *out_state = g_canvas_context_singleton->module_input_state[module_idx];
    return true;
}

void canvas_clear_module_input_pulses(int module_idx) {
    if (!g_canvas_context_singleton) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(g_canvas_context_singleton->module_input_state.size())) return;
    auto &state = g_canvas_context_singleton->module_input_state[module_idx];
    state.mouse_down = 0;
    state.mouse_up = 0;
    state.key_event = 0;
}

void canvas_set_module_stack_snapshot(int module_idx, int row_idx, const float* values, int count) {
    if (!g_canvas_context_singleton) return;
    if (module_idx < 0) return;
    if (module_idx >= static_cast<int>(g_canvas_context_singleton->module_stack_snapshots.size())) {
        g_canvas_context_singleton->module_stack_snapshots.resize(module_idx + 1);
    }
    auto &snapshots = g_canvas_context_singleton->module_stack_snapshots[module_idx];
    std::vector<float> data;
    if (values && count > 0) {
        data.assign(values, values + count);
    }
    snapshots[row_idx] = std::move(data);
}

static bool module_has_tool(const GP_CanvasContextImpl* ctx, int module_idx, ModuleToolKind tool) {
    if (!ctx) return false;
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_io_rows.size())) {
        const auto &rows = ctx->module_io_rows[module_idx];
        for (const auto &row : rows) {
            if (row.kind == ModuleRowKind::Tool && row.tool == tool) return true;
        }
    }
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_tool_stack.size())) {
        const auto &tools = ctx->module_tool_stack[module_idx];
        return std::find(tools.begin(), tools.end(), tool) != tools.end();
    }
    return false;
}

static void ensure_module_row_order(GP_CanvasContextImpl* ctx, int module_idx) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) ctx->module_io_rows.resize(module_idx + 1);
    auto &rows = ctx->module_io_rows[module_idx];
    if (!rows.empty()) return;
    if (module_idx >= static_cast<int>(ctx->module_io_input_rows.size())) ctx->module_io_input_rows.resize(module_idx + 1);
    if (module_idx >= static_cast<int>(ctx->module_io_output_rows.size())) ctx->module_io_output_rows.resize(module_idx + 1);
    const auto &input_rows = ctx->module_io_input_rows[module_idx];
    const auto &output_rows = ctx->module_io_output_rows[module_idx];
    for (int count : input_rows) {
        rows.push_back({ModuleRowKind::Input, 0, ModuleToolKind::None, std::clamp(count, 1, 32)});
    }
    if (module_idx < static_cast<int>(ctx->module_tool_stack.size())) {
        for (ModuleToolKind tool : ctx->module_tool_stack[module_idx]) {
            rows.push_back({ModuleRowKind::Tool, -1, tool, 0});
        }
    }
    for (int count : output_rows) {
        rows.push_back({ModuleRowKind::Output, 0, ModuleToolKind::None, std::clamp(count, 1, 32)});
    }
}

static void canvas_record_key_input(GP_CanvasContextImpl* ctx, int key, int action) {
    if (!ctx || action == 0) return;
    for (int mi = 0; mi < static_cast<int>(ctx->module_input_state.size()); ++mi) {
        if (!module_has_tool(ctx, mi, ModuleToolKind::KeyboardListener)) continue;
        if (mi < 0 || mi >= static_cast<int>(ctx->module_input_state.size())) continue;
        auto &state = ctx->module_input_state[mi];
        state.key = key;
        state.key_event = 1;
    }
}

static void canvas_record_mouse_input(GP_CanvasContextImpl* ctx, int x, int y, bool down, bool up) {
    if (!ctx) return;
    for (int mi = 0; mi < static_cast<int>(ctx->module_input_state.size()); ++mi) {
        if (!module_has_tool(ctx, mi, ModuleToolKind::MouseListener)) continue;
        if (mi < 0 || mi >= static_cast<int>(ctx->module_input_state.size())) continue;
        auto &state = ctx->module_input_state[mi];
        state.mouse_x = static_cast<float>(x);
        state.mouse_y = static_cast<float>(y);
        if (down) state.mouse_down = 1;
        if (up) state.mouse_up = 1;
    }
}


// NOTE: legacy per-side contact geometry removed. Contact hit/position
// information is authoritative from attached `GP_TableContext` hitboxes.

static inline int table_hit_contact_index(const GP_TableHitBox& hb) {
    if (hb.row_idx >= 0) return hb.row_idx;
    if (hb.part == GP_TABLE_HIT_LED_TABLE) return hb.aux1;
    return hb.aux0;
}

static int resolve_contact_index(const GP_CanvasContextImpl* ctx, int module_idx, const GP_TableHitBox& hb) {
    if (!ctx) return table_hit_contact_index(hb);
    const std::vector<ModuleIORow>* rows_ptr = nullptr;
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_table_rows.size()) &&
        !ctx->module_table_rows[module_idx].empty()) {
        rows_ptr = &ctx->module_table_rows[module_idx];
    } else if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_io_rows.size())) {
        rows_ptr = &ctx->module_io_rows[module_idx];
    }
    if (rows_ptr) {
        const auto &rows = *rows_ptr;
        if (hb.row_idx >= 0 && hb.row_idx < static_cast<int>(rows.size())) {
            const ModuleIORow &meta = rows[hb.row_idx];
            if (meta.kind == ModuleRowKind::Tool) return -1;
            int attachment_count = std::max(1, meta.attachment_count);
            int led_idx = 0;
            if (hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG) {
                led_idx = std::clamp(hb.aux0, 0, attachment_count - 1);
            } else if (hb.part == GP_TABLE_HIT_LED_TABLE) {
                led_idx = std::clamp(hb.aux1, 0, attachment_count - 1);
            }
            return meta.contact_idx + led_idx;
        }
    }
    return table_hit_contact_index(hb);
}

static int resolve_side_contact_index(const GP_CanvasContextImpl* ctx, int module_idx, bool is_input, int contact_idx) {
    if (is_input) return contact_idx;
    if (!ctx) return contact_idx;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_io_in_count.size())) return contact_idx;
    int offset = std::max(0, ctx->module_io_in_count[module_idx]);
    return contact_idx - offset;
}

struct CanvasBounds {
    int min_x = 0;
    int max_x = 0;
    int min_y = 0;
    int max_y = 0;
    bool has_any = false;
};

static CanvasBounds compute_canvas_bounds(const GP_CanvasContextImpl* ctx) {
    CanvasBounds b;
    if (!ctx) return b;
    if (!ctx->modules.empty()) {
        b.has_any = true;
        b.min_x = ctx->modules.front().x;
        b.max_x = ctx->modules.front().x + ctx->modules.front().w;
        b.min_y = ctx->modules.front().y;
        b.max_y = ctx->modules.front().y + ctx->modules.front().h;
        for (const auto& m : ctx->modules) {
            b.min_x = std::min(b.min_x, m.x);
            b.max_x = std::max(b.max_x, m.x + m.w);
            b.min_y = std::min(b.min_y, m.y);
            b.max_y = std::max(b.max_y, m.y + m.h);
        }
    } else {
        b.min_x = 0; b.max_x = ctx->width;
        b.min_y = 0; b.max_y = ctx->height;
    }
    return b;
}

static uint32_t lookup_molex_hash(const GP_CanvasContextImpl* ctx, int module_idx, bool is_input, int contact_idx) {
    if (!ctx || contact_idx < 0) return 0;
    const auto &layout = is_input ? ctx->module_input_layout : ctx->module_output_layout;
    if (module_idx < 0 || module_idx >= static_cast<int>(layout.size())) return 0;
    const MolexLayoutInfo &info = layout[module_idx];
    int side_idx = resolve_side_contact_index(ctx, module_idx, is_input, contact_idx);
    if (side_idx < 0 || side_idx >= static_cast<int>(info.hashes.size())) return 0;
    return info.hashes[side_idx];
}

static void clamp_offset_to_bounds(GP_CanvasContextImpl* ctx, const CanvasBounds& b) {
    if (!ctx) return;
    int view_w = std::max(1, ctx->width);
    int view_h = std::max(1, ctx->height);
    int min_off_x = std::min(0, b.min_x);
    int max_off_x = std::max(min_off_x, b.max_x - view_w);
    int min_off_y = std::min(0, b.min_y);
    int max_off_y = std::max(min_off_y, b.max_y - view_h);
    ctx->offset_x = std::clamp(ctx->offset_x, min_off_x, max_off_x);
    ctx->offset_y = std::clamp(ctx->offset_y, min_off_y, max_off_y);
    ctx->scroll_x_needed = (b.min_x < ctx->offset_x) || (b.max_x > ctx->offset_x + view_w);
    ctx->scroll_y_needed = (b.min_y < ctx->offset_y) || (b.max_y > ctx->offset_y + view_h);
}

static void sync_container_scroll(GP_CanvasContextImpl* ctx, const CanvasBounds& b) {
    if (!ctx || !ctx->container_table) return;
    float fx = 0.0f, fy = 0.0f;
    // tolerate older tables that only expose vertical scroll
    if (!gp_table_get_scroll_fraction_xy(ctx->container_table, &fx, &fy)) {
        gp_table_get_scroll_fraction(ctx->container_table, &fy);
        fx = 0.0f;
    }
    fx = std::clamp(fx, 0.0f, 1.0f);
    fy = std::clamp(fy, 0.0f, 1.0f);
    int view_w = std::max(1, ctx->width);
    int view_h = std::max(1, ctx->height);
    int min_off_x = std::min(0, b.min_x);
    int max_off_x = std::max(min_off_x, b.max_x - view_w);
    int min_off_y = std::min(0, b.min_y);
    int max_off_y = std::max(min_off_y, b.max_y - view_h);
    ctx->offset_x = min_off_x + static_cast<int>(std::lround(fx * float(max_off_x - min_off_x)));
    ctx->offset_y = min_off_y + static_cast<int>(std::lround(fy * float(max_off_y - min_off_y)));
    ctx->offset_x = std::clamp(ctx->offset_x, min_off_x, max_off_x);
    ctx->offset_y = std::clamp(ctx->offset_y, min_off_y, max_off_y);
    float out_fx = (max_off_x == min_off_x) ? 0.0f : float(ctx->offset_x - min_off_x) / float(max_off_x - min_off_x);
    float out_fy = (max_off_y == min_off_y) ? 0.0f : float(ctx->offset_y - min_off_y) / float(max_off_y - min_off_y);
    gp_table_set_scroll_fraction_xy(ctx->container_table, out_fx, out_fy);
}

static CanvasBounds update_canvas_scroll_state(GP_CanvasContextImpl* ctx, bool pull_from_container) {
    CanvasBounds b = compute_canvas_bounds(ctx);
    if (pull_from_container) sync_container_scroll(ctx, b);
    clamp_offset_to_bounds(ctx, b);
    if (ctx && ctx->container_table) {
        int view_w = std::max(1, ctx->width);
        int view_h = std::max(1, ctx->height);
        int min_off_x = std::min(0, b.min_x);
        int max_off_x = std::max(min_off_x, b.max_x - view_w);
        int min_off_y = std::min(0, b.min_y);
        int max_off_y = std::max(min_off_y, b.max_y - view_h);
        float fx = (max_off_x == min_off_x) ? 0.0f : float(ctx->offset_x - min_off_x) / float(max_off_x - min_off_x);
        float fy = (max_off_y == min_off_y) ? 0.0f : float(ctx->offset_y - min_off_y) / float(max_off_y - min_off_y);
        gp_table_set_scroll_fraction_xy(ctx->container_table, fx, fy);
    }
    return b;
}

enum CanvasActionId {
    CANVAS_ACT_SIM_SEGS_DEC = 2000,
    CANVAS_ACT_SIM_SEGS_INC = 2001,
    CANVAS_ACT_SIM_SLACK_DEC = 2002,
    CANVAS_ACT_SIM_SLACK_INC = 2003,
    CANVAS_ACT_TOOL_CANVAS_0 = 2010,
    CANVAS_ACT_TOOL_CANVAS_1 = 2011,
    CANVAS_ACT_TOOL_CANVAS_2 = 2012,
    CANVAS_ACT_TOOL_CANVAS_3 = 2013,
    CANVAS_ACT_TOOL_EDGE_0 = 2014,
    CANVAS_ACT_TOOL_EDGE_1 = 2015,
    CANVAS_ACT_TOOL_EDGE_2 = 2016,
    CANVAS_ACT_TOOL_EDGE_3 = 2017,
    CANVAS_ACT_EDGE_ORDER_DEC = 2018,
    CANVAS_ACT_EDGE_ORDER_INC = 2019,
    CANVAS_ACT_EDGE_ORDER_TOOL = 2020,
    CANVAS_ACT_TOOL_TABLE_0 = 2030,
    CANVAS_ACT_TOOL_TABLE_1 = 2031,
    CANVAS_ACT_TOOL_TABLE_2 = 2032,
    CANVAS_ACT_IO_COUNT_DEC = 2040,
    CANVAS_ACT_IO_COUNT_INC = 2041,
    CANVAS_ACT_IO_CONSUMER_ADD = 2042,
    CANVAS_ACT_IO_PRODUCER_ADD = 2043,
    CANVAS_ACT_MODULE_LED = 2050,
    CANVAS_ACT_MENU_TOOL_ADD = 2101,
    CANVAS_ACT_MENU_TOOL_SUB = 2102,
    CANVAS_ACT_MENU_TOOL_MUL = 2103,
    CANVAS_ACT_MENU_TOOL_DIV = 2104,
    CANVAS_ACT_MENU_TOOL_MOD = 2105,
    CANVAS_ACT_MENU_TOOL_KEYBOARD = 2106,
    CANVAS_ACT_MENU_TOOL_MOUSE = 2107,
    CANVAS_ACT_MENU_TOOL_STACK = 2108,
};

struct InputRayLight {
    float cx = 0.0f;
    float cy = 0.0f;
    float radius = 1.0f;
    float intensity = 0.0f;
    int contact_idx = -1;
};

static void ensure_module_bg_storage(GP_CanvasContextImpl::ModuleBg& bg, int w, int h, int oversample) {
    const int mw = std::max(0, w);
    const int mh = std::max(0, h);
    const int os = std::max(1, oversample);
    const int mw_hi = mw * os;
    const int mh_hi = mh * os;

    const size_t px = static_cast<size_t>(mw) * static_cast<size_t>(mh);
    const size_t px_hi = static_cast<size_t>(std::max(0, mw_hi)) * static_cast<size_t>(std::max(0, mh_hi));

    // Frame accum is cleared by the caller each render; preserve temporal across frames.
    if (bg.accum.size() != px) bg.accum.assign(px, 0.0f);
    if (bg.temporal.size() != px) bg.temporal.assign(px, 0.0f);
    if (bg.scratch.size() != px * 4u) bg.scratch.assign(px * 4u, 0);
    if (bg.layer.size() != px_hi * 4u) bg.layer.assign(px_hi * 4u, 0);

    if (!bg.ray && mw_hi > 0 && mh_hi > 0) {
        bg.ray = raytrace2d_create(mw_hi, mh_hi);
    } else if (bg.ray && mw_hi > 0 && mh_hi > 0) {
        raytrace2d_resize(bg.ray, mw_hi, mh_hi);
    }
}

static void blit_module_buffer(uint8_t* out_rgba, int w, int h, int pitch, int sx, int sy, int mw, int mh, const std::vector<uint8_t>& src) {
    if (!out_rgba || src.empty() || mw <= 0 || mh <= 0) return;
    for (int yy = 0; yy < mh; ++yy) {
        int dst_y = sy + yy;
        if (dst_y < 0 || dst_y >= h) continue;
        uint8_t* dst_row = out_rgba + dst_y * pitch;
        const uint8_t* src_row = src.data() + static_cast<size_t>(yy) * static_cast<size_t>(mw) * 4u;
        int dst_x0 = sx;
        int src_x0 = 0;
        int copy_w = mw;
        if (dst_x0 < 0) { src_x0 = -dst_x0; copy_w -= src_x0; dst_x0 = 0; }
        copy_w = std::min(copy_w, std::max(0, w - dst_x0));
        if (copy_w <= 0) continue;
        std::memcpy(dst_row + dst_x0 * 4, src_row + static_cast<size_t>(src_x0) * 4u, static_cast<size_t>(copy_w) * 4u);
    }
}

static void blit_module_buffer_alpha(uint8_t* out_rgba, int w, int h, int pitch, int sx, int sy, int mw, int mh, const std::vector<uint8_t>& src, uint8_t global_alpha) {
    if (!out_rgba || src.empty() || mw <= 0 || mh <= 0) return;
    if (global_alpha >= 255) {
        blit_module_buffer(out_rgba, w, h, pitch, sx, sy, mw, mh, src);
        return;
    }
    const float g = global_alpha / 255.0f;
    for (int yy = 0; yy < mh; ++yy) {
        int dst_y = sy + yy;
        if (dst_y < 0 || dst_y >= h) continue;
        uint8_t* dst_row = out_rgba + dst_y * pitch;
        const uint8_t* src_row = src.data() + static_cast<size_t>(yy) * static_cast<size_t>(mw) * 4u;
        int dst_x0 = sx;
        int src_x0 = 0;
        int copy_w = mw;
        if (dst_x0 < 0) { src_x0 = -dst_x0; copy_w -= src_x0; dst_x0 = 0; }
        copy_w = std::min(copy_w, std::max(0, w - dst_x0));
        if (copy_w <= 0) continue;
        uint8_t* dst = dst_row + dst_x0 * 4;
        const uint8_t* srcp = src_row + static_cast<size_t>(src_x0) * 4u;
        for (int xx = 0; xx < copy_w; ++xx) {
            const uint8_t sa = srcp[3];
            const float a = (sa / 255.0f) * g;
            if (a >= 0.999f) {
                dst[0] = srcp[0];
                dst[1] = srcp[1];
                dst[2] = srcp[2];
                dst[3] = 255;
            } else if (a > 0.001f) {
                const float inv = 1.0f - a;
                dst[0] = static_cast<uint8_t>(std::lround(srcp[0] * a + dst[0] * inv));
                dst[1] = static_cast<uint8_t>(std::lround(srcp[1] * a + dst[1] * inv));
                dst[2] = static_cast<uint8_t>(std::lround(srcp[2] * a + dst[2] * inv));
                dst[3] = 255;
            }
            dst += 4;
            srcp += 4;
        }
    }
}

// Blend source using its own alpha (no global alpha scaling).
static void blit_module_buffer_srcalpha(uint8_t* out_rgba, int w, int h, int pitch, int sx, int sy, int mw, int mh, const std::vector<uint8_t>& src) {
    if (!out_rgba || src.empty() || mw <= 0 || mh <= 0) return;
    for (int yy = 0; yy < mh; ++yy) {
        int dst_y = sy + yy;
        if (dst_y < 0 || dst_y >= h) continue;
        uint8_t* dst_row = out_rgba + dst_y * pitch;
        const uint8_t* src_row = src.data() + static_cast<size_t>(yy) * static_cast<size_t>(mw) * 4u;
        int dst_x0 = sx;
        int src_x0 = 0;
        int copy_w = mw;
        if (dst_x0 < 0) { src_x0 = -dst_x0; copy_w -= src_x0; dst_x0 = 0; }
        copy_w = std::min(copy_w, std::max(0, w - dst_x0));
        if (copy_w <= 0) continue;
        uint8_t* dst = dst_row + dst_x0 * 4;
        const uint8_t* srcp = src_row + static_cast<size_t>(src_x0) * 4u;
        for (int xx = 0; xx < copy_w; ++xx) {
            const float a = srcp[3] / 255.0f;
            if (a >= 0.999f) {
                dst[0] = srcp[0];
                dst[1] = srcp[1];
                dst[2] = srcp[2];
                dst[3] = 255;
            } else if (a > 0.001f) {
                const float inv = 1.0f - a;
                dst[0] = static_cast<uint8_t>(std::lround(srcp[0] * a + dst[0] * inv));
                dst[1] = static_cast<uint8_t>(std::lround(srcp[1] * a + dst[1] * inv));
                dst[2] = static_cast<uint8_t>(std::lround(srcp[2] * a + dst[2] * inv));
                dst[3] = 255;
            }
            dst += 4;
            srcp += 4;
        }
    }
}

static void blit_module_buffer_alpha_scaled(uint8_t* out_rgba, int w, int h, int pitch, int dst_x, int dst_y, int dst_w, int dst_h, const std::vector<uint8_t>& src, int src_w, int src_h, uint8_t global_alpha) {
    if (!out_rgba || src.empty() || dst_w <= 0 || dst_h <= 0 || src_w <= 0 || src_h <= 0) return;
    const float g = std::clamp(global_alpha, uint8_t(0), uint8_t(255)) / 255.0f;
    for (int dy = 0; dy < dst_h; ++dy) {
        int y = dst_y + dy;
        if (y < 0 || y >= h) continue;
        float syf = (dst_h > 1) ? (float(dy) / float(dst_h - 1)) * float(src_h - 1) : 0.0f;
        int syi = std::clamp(int(std::round(syf)), 0, src_h - 1);
        const uint8_t* src_row = src.data() + static_cast<size_t>(syi) * static_cast<size_t>(src_w) * 4u;
        for (int dx = 0; dx < dst_w; ++dx) {
            int x = dst_x + dx;
            if (x < 0 || x >= w) continue;
            float sxf = (dst_w > 1) ? (float(dx) / float(dst_w - 1)) * float(src_w - 1) : 0.0f;
            int sxi = std::clamp(int(std::round(sxf)), 0, src_w - 1);
            const uint8_t* sp = src_row + static_cast<size_t>(sxi) * 4u;
            float sa = (sp[3] / 255.0f) * g;
            if (sa <= 0.0f) continue;
            float inv = 1.0f - sa;
            uint8_t* dp = out_rgba + y * pitch + x * 4;
            for (int c = 0; c < 3; ++c) {
                float s = sp[c] / 255.0f;
                float d = dp[c] / 255.0f;
                dp[c] = static_cast<uint8_t>(std::lround((s * sa + d * inv) * 255.0f));
            }
            float da = dp[3] / 255.0f;
            dp[3] = static_cast<uint8_t>(std::lround((sa + da * inv) * 255.0f));
        }
    }
}

static void render_module_raytrace_bg(GP_CanvasContextImpl::ModuleBg& bg, int mw, int mh, const std::vector<InputRayLight>& lights) {
    const int os = std::max(1, bg.oversample);
    const int mw_hi = std::max(0, mw) * os;
    const int mh_hi = std::max(0, mh) * os;
    ensure_module_bg_storage(bg, mw, mh, os);
    if (!bg.ray) return;
    std::fill(bg.accum.begin(), bg.accum.end(), 0.0f);

    // Temporal accumulation: decay per redraw (not wall-clock).
    const float temporal_decay = std::clamp(bg.temporal_decay, 0.0f, 1.0f);
    const float temporal_max = std::max(0.0f, bg.temporal_max);

    if (!lights.empty() && mw > 0 && mh > 0 && mw_hi > 0 && mh_hi > 0) {
        ++bg.temporal_frame;
        raytrace2d_set_room(bg.ray, 0.0f, 0.0f, static_cast<float>(mw), static_cast<float>(mh));
        raytrace2d_set_params(bg.ray, bg.rays, bg.reflections, bg.blur_sigma);
        raytrace2d_set_attenuation(bg.ray, bg.ray_bounce_decay, bg.ray_air_decay);
        for (const auto& light : lights) {
            if (light.intensity <= 0.0f) continue;
            uint32_t seed = static_cast<uint32_t>(light.contact_idx + 1);
            seed = seed * 0x9E3779B1u;
            seed ^= (bg.temporal_frame * 0x85EBCA6Bu) + 0xC2B2AE35u;
            if (seed == 0) seed = 1u;
            raytrace2d_set_seed(bg.ray, seed);
            raytrace2d_set_light(bg.ray, light.cx, light.cy, light.radius);
            if (!raytrace2d_render_rgba(bg.ray, bg.layer.data(), static_cast<int32_t>(bg.layer.size()))) continue;

            // Downsample (nearest-neighbor) from oversampled buffer and accumulate.
            for (int y = 0; y < mh; ++y) {
                const int hy = y * os;
                const size_t row_hi = static_cast<size_t>(hy) * static_cast<size_t>(mw_hi);
                const size_t row_lo = static_cast<size_t>(y) * static_cast<size_t>(mw);
                for (int x = 0; x < mw; ++x) {
                    const int hx = x * os;
                    const size_t idx_hi = (row_hi + static_cast<size_t>(hx)) * 4u;
                    const size_t idx_lo = row_lo + static_cast<size_t>(x);
                    float v = bg.layer[idx_hi + 0] / 255.0f;
                    bg.accum[idx_lo] += v * light.intensity;
                }
            }
        }

        // Accumulate with decay and clamp to a max intensity.
        for (size_t i = 0; i < bg.temporal.size(); ++i) {
            float v = bg.temporal[i] * temporal_decay + bg.accum[i];
            bg.temporal[i] = std::min(temporal_max, v);
        }
    } else {
        // No lights: decay the existing accumulated field.
        for (size_t i = 0; i < bg.temporal.size(); ++i) {
            bg.temporal[i] *= temporal_decay;
        }
    }

    const float exposure = std::max(0.0f, bg.ray_exposure);
    const uint8_t base_r = 40, base_g = 40, base_b = 50;
    const float add = 160.0f;
    for (int y = 0; y < mh; ++y) {
        for (int x = 0; x < mw; ++x) {
            size_t idx = static_cast<size_t>(y) * static_cast<size_t>(mw) + static_cast<size_t>(x);
            float t = std::clamp(bg.temporal[idx] * exposure, 0.0f, 1.0f);
            uint8_t r = static_cast<uint8_t>(std::clamp(base_r + t * add, 0.0f, 255.0f));
            uint8_t g = static_cast<uint8_t>(std::clamp(base_g + t * add, 0.0f, 255.0f));
            uint8_t b = static_cast<uint8_t>(std::clamp(base_b + t * add, 0.0f, 255.0f));
            size_t o = idx * 4u;
            bg.scratch[o + 0] = r;
            bg.scratch[o + 1] = g;
            bg.scratch[o + 2] = b;
            bg.scratch[o + 3] = 255;
        }
    }
}

static const char* tool_label(ModuleToolKind tool) {
    switch (tool) {
        case ModuleToolKind::Add: return LABEL_TOOL_ADD;
        case ModuleToolKind::Subtract: return LABEL_TOOL_SUB;
        case ModuleToolKind::Multiply: return LABEL_TOOL_MUL;
        case ModuleToolKind::Divide: return LABEL_TOOL_DIV;
        case ModuleToolKind::Modulo: return LABEL_TOOL_MOD;
        case ModuleToolKind::KeyboardListener: return LABEL_TOOL_KEYBOARD;
        case ModuleToolKind::MouseListener: return LABEL_TOOL_MOUSE;
        case ModuleToolKind::StackDisplay: return LABEL_TOOL_STACK;
        case ModuleToolKind::None:
        default:
            return "";
    }
}

struct ToolMenuItem {
    int action_id = 0;
    const char* label = "";
    ModuleToolKind tool = ModuleToolKind::None;
};

static const ToolMenuItem kToolMenuItems[] = {
    { CANVAS_ACT_MENU_TOOL_ADD, LABEL_TOOL_ADD, ModuleToolKind::Add },
    { CANVAS_ACT_MENU_TOOL_SUB, LABEL_TOOL_SUB, ModuleToolKind::Subtract },
    { CANVAS_ACT_MENU_TOOL_MUL, LABEL_TOOL_MUL, ModuleToolKind::Multiply },
    { CANVAS_ACT_MENU_TOOL_DIV, LABEL_TOOL_DIV, ModuleToolKind::Divide },
    { CANVAS_ACT_MENU_TOOL_MOD, LABEL_TOOL_MOD, ModuleToolKind::Modulo },
    { CANVAS_ACT_MENU_TOOL_KEYBOARD, LABEL_TOOL_KEYBOARD, ModuleToolKind::KeyboardListener },
    { CANVAS_ACT_MENU_TOOL_MOUSE, LABEL_TOOL_MOUSE, ModuleToolKind::MouseListener },
    { CANVAS_ACT_MENU_TOOL_STACK, LABEL_TOOL_STACK, ModuleToolKind::StackDisplay },
};

struct ToolMenuLayout {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    int header_h = 0;
    int row_h = 0;
    int item_start_y = 0;
    int stack_start_y = 0;
    int stack_count = 0;
};

static ToolMenuLayout compute_tool_menu_layout(const GP_CanvasContextImpl* ctx) {
    ToolMenuLayout layout{};
    if (!ctx) return layout;
    const int padding = 8;
    const int margin = 12;
    const int header_h = 18;
    const int row_h = 22;
    const int w = 200;
    int stack_count = 0;
    if (ctx->focused_module >= 0 && ctx->focused_module < static_cast<int>(ctx->module_io_rows.size())) {
        stack_count = static_cast<int>(ctx->module_io_rows[ctx->focused_module].size());
    }
    int h = padding * 2 + header_h + static_cast<int>(std::size(kToolMenuItems)) * row_h;
    if (stack_count > 0) {
        h += padding + header_h + stack_count * row_h;
    }
    int x = std::max(margin, ctx->width - w - margin);
    int y = ctx->rope_bar_h + ctx->control_bar_h + margin;
    layout.x = x;
    layout.y = y;
    layout.w = w;
    layout.h = h;
    layout.header_h = header_h;
    layout.row_h = row_h;
    layout.item_start_y = y + padding + header_h;
    layout.stack_start_y = layout.item_start_y + static_cast<int>(std::size(kToolMenuItems)) * row_h + padding;
    layout.stack_count = stack_count;
    return layout;
}

static const GP_TableAction kCanvasRootActions[] = {
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SIM_SEGS_DEC, CANVAS_ACT_SIM_SEGS_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SIM_SEGS_INC, CANVAS_ACT_SIM_SEGS_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SIM_SLACK_DEC, CANVAS_ACT_SIM_SLACK_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SIM_SLACK_INC, CANVAS_ACT_SIM_SLACK_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_CANVAS_0, CANVAS_ACT_TOOL_CANVAS_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_CANVAS_1, CANVAS_ACT_TOOL_CANVAS_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_CANVAS_2, CANVAS_ACT_TOOL_CANVAS_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_CANVAS_3, CANVAS_ACT_TOOL_CANVAS_3 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_0, CANVAS_ACT_TOOL_EDGE_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_1, CANVAS_ACT_TOOL_EDGE_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_2, CANVAS_ACT_TOOL_EDGE_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_3, CANVAS_ACT_TOOL_EDGE_3 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EDGE_ORDER_DEC, CANVAS_ACT_EDGE_ORDER_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EDGE_ORDER_INC, CANVAS_ACT_EDGE_ORDER_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EDGE_ORDER_TOOL, CANVAS_ACT_EDGE_ORDER_TOOL },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_TABLE_0, CANVAS_ACT_TOOL_TABLE_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_TABLE_1, CANVAS_ACT_TOOL_TABLE_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_TABLE_2, CANVAS_ACT_TOOL_TABLE_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_IO_COUNT_DEC, CANVAS_ACT_IO_COUNT_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_IO_COUNT_INC, CANVAS_ACT_IO_COUNT_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_IO_CONSUMER_ADD, CANVAS_ACT_IO_CONSUMER_ADD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_IO_PRODUCER_ADD, CANVAS_ACT_IO_PRODUCER_ADD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_ADD, CANVAS_ACT_MENU_TOOL_ADD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_SUB, CANVAS_ACT_MENU_TOOL_SUB },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_MUL, CANVAS_ACT_MENU_TOOL_MUL },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_DIV, CANVAS_ACT_MENU_TOOL_DIV },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_MOD, CANVAS_ACT_MENU_TOOL_MOD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_KEYBOARD, CANVAS_ACT_MENU_TOOL_KEYBOARD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_MOUSE, CANVAS_ACT_MENU_TOOL_MOUSE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_STACK, CANVAS_ACT_MENU_TOOL_STACK },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_LED, GP_TABLE_ACTION_ANY, CANVAS_ACT_MODULE_LED },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_LED_ARG, GP_TABLE_ACTION_ANY, CANVAS_ACT_MODULE_LED },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_LED_TABLE, GP_TABLE_ACTION_ANY, CANVAS_ACT_MODULE_LED },
};

static int canvas_handle_module_led_hit(GP_CanvasContextImpl* ctx, int module_idx, const GP_TableHitBox& hit);
static void canvas_setup_stage_table(GP_TableContext* t, int w_px);
static void canvas_setup_stage_defaults(GP_StageContext* st, int w_px, int h_px);
static void stage_bg_callback(void* user, int module_idx, int width, int height, uint8_t* out_rgba, int32_t out_pitch);
static GP_TableContext* canvas_ensure_root_table(GP_CanvasContextImpl* ctx);
static void canvas_append_io_row(GP_CanvasContextImpl* ctx, int module_idx, bool is_input, int attachment_count);
static void sync_module_table_io_layout(GP_CanvasContextImpl* ctx, int module_idx);
static void canvas_push_tool_to_focused(GP_CanvasContextImpl* ctx, ModuleToolKind tool);

// Per-module recorder state stored by the canvas so it can be freed on table destroy.
struct KeyRecorderState {
    GP_CanvasContextImpl* canvas;
    int module_idx;
    uint64_t writer_key;
    int root_edge_idx;
};

// Create a simple table that records key presses. Attaches a key callback
// that appends a text row for each key press.
static void canvas_install_key_recorder_table(GP_CanvasContextImpl* ctx, int module_idx) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    // create table
    GP_TableContext* t = gp_table_create(nullptr);
    if (!t) return;
    // style
    GP_TableStyle st{};
    st.width_px = std::max(1, ctx->modules[module_idx].w);
    st.row_h_px = 20;
    st.name_w_px = 140;
    gp_table_set_style(t, &st);
    // one text column
    GP_TableColumn col{};
    col.kind = GP_TABLE_CELL_TEXT;
    col.width_px = std::max(64, st.width_px - st.name_w_px);
    col.align = 0;
    gp_table_set_columns(t, &col, 1);
    // initial header row
    GP_TableRow r{};
    memset(&r, 0, sizeof(r));
    r.kind = GP_TABLE_ROW_HEADER;
    r.depth = 0;
    r.expanded = 1;
    r.selected = 0;
    r.cell_count = 1;
    r.cells[0].kind = GP_TABLE_CELL_TEXT;
    std::snprintf(r.cells[0].text, sizeof(r.cells[0].text), "%s", LABEL_MENU_KEY_RECORDER);
    gp_table_set_rows(t, &r, 1);

    // We'll attach a key callback that both records the key locally and
    // publishes the key value onto a root-table edge (if present).
    // Create a small state object and store it per-module.
    KeyRecorderState* ks = new KeyRecorderState();
    ks->canvas = ctx; ks->module_idx = module_idx; ks->writer_key = 0; ks->root_edge_idx = -1;

    gp_table_set_key_callback(t, +[](void* user, int key, int scancode, int action, int mods) {
        KeyRecorderState* ks = reinterpret_cast<KeyRecorderState*>(user);
        if (!ks) return;
        printf("key-recorder callback: module=%d key=%d sc=%d action=%d mods=%d writer_key=%llu root_edge_idx=%d\n", ks->module_idx, key, scancode, action, mods, (unsigned long long)ks->writer_key, ks->root_edge_idx);
        GP_TableContext* tt = nullptr;
        // find the attached table for this module via canvas
        GP_CanvasContextImpl* c = ks->canvas;
        if (!c) return;
        int mi = ks->module_idx;
        if (mi < 0 || mi >= static_cast<int>(c->module_tables.size())) return;
        tt = c->module_tables[mi];
        if (!tt) return;
        // Only act on non-zero actions
        if (action == 0) return;
        // append local row
        int32_t n = gp_table_get_row_count(tt);
        std::vector<GP_TableRow> rows;
        rows.resize(static_cast<size_t>(n + 1));
        for (int32_t i = 0; i < n; ++i) {
            GP_TableRow tmp; memset(&tmp, 0, sizeof(tmp));
            if (gp_table_get_row(tt, i, &tmp)) rows[static_cast<size_t>(i)] = tmp;
        }
        GP_TableRow nr; memset(&nr, 0, sizeof(nr));
        nr.kind = GP_TABLE_ROW_DEVICE; nr.depth = 0; nr.expanded = 1; nr.selected = 0; nr.cell_count = 1;
        nr.cells[0].kind = GP_TABLE_CELL_TEXT;
        std::snprintf(nr.cells[0].text, sizeof(nr.cells[0].text), "K=%d S=%d A=%d M=%d", key, scancode, action, mods);
        rows[static_cast<size_t>(n)] = nr;
        gp_table_set_rows(tt, rows.data(), static_cast<int32_t>(rows.size()));

        // publish to root edge(s) if available. Use module's output count
        // to determine how many contact channels to publish to. This lets a
        // single recorder emit to multiple writer keys (contact_idx 0..N-1).
        GP_TableContext* root = canvas_ensure_root_table(c);
        if (!root) return;
        int out_count = 1;
        if (mi >= 0 && mi < static_cast<int>(c->module_io_out_count.size())) out_count = std::max(1, c->module_io_out_count[mi]);
        for (int ci = 0; ci < out_count; ++ci) {
            uint64_t writer_key = (static_cast<uint64_t>(static_cast<uint32_t>(mi)) << 32) |
                                  (static_cast<uint64_t>(static_cast<uint32_t>(ci)) << 16) |
                                  static_cast<uint64_t>(0);
            int edge_idx = -1;
            if (!gp_table_edge_index_for_key(root, writer_key, &edge_idx) || edge_idx < 0) {
                // no edge for this contact, skip
                //printf("key-recorder: no root edge for module=%d contact=%d key=%llu\n", ks->module_idx, ci, (unsigned long long)writer_key);
                continue;
            }
            float payload[1]; payload[0] = static_cast<float>(key);
            int dropped = 0;
            int ok = gp_table_edge_publish(root, edge_idx, writer_key, payload, 1, &dropped);
            printf("key-recorder publish: module=%d contact=%d edge=%d writer_key=%llu ok=%d dropped=%d payload=%f\n", ks->module_idx, ci, edge_idx, (unsigned long long)writer_key, ok, dropped, payload[0]);
            if (!ok) {
                printf("key-recorder publish FAILED: module=%d contact=%d edge=%d writer_key=%llu\n", ks->module_idx, ci, edge_idx, (unsigned long long)writer_key);
            }
        }
    }, ks);

    // store state so we can free it on table destroy
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_key_recorder_state.size())) ctx->module_key_recorder_state[module_idx] = reinterpret_cast<void*>(ks);
    printf("canvas_install_key_recorder_table: installed recorder for module=%d writer_key=%llu\n", module_idx, (unsigned long long)ks->writer_key);

    // attach to module and let canvas own it
    gp_canvas_attach_table(reinterpret_cast<GP_CanvasContext*>(ctx), module_idx, t, 1);
    ctx->focused_module = module_idx;
    if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) ctx->module_io_rows.resize(module_idx + 1);
    ctx->module_io_rows[module_idx].clear();
    if (module_idx >= static_cast<int>(ctx->module_table_rows.size())) ctx->module_table_rows.resize(module_idx + 1);
    ctx->module_table_rows[module_idx].clear();
    if (module_idx >= static_cast<int>(ctx->module_stack_snapshots.size())) ctx->module_stack_snapshots.resize(module_idx + 1);
    ctx->module_stack_snapshots[module_idx].clear();
    // If there's another module available, create a canvas/root edge from this module to the next module
    if (ctx->modules.size() > 1) {
        int target = (module_idx + 1) % static_cast<int>(ctx->modules.size());
        if (target != module_idx) {
            GP_CanvasEdgeDesc ed{};
            ed.a_module = module_idx; ed.a_contact_idx = 0;
            ed.b_module = target; ed.b_contact_idx = 0;
            // create a canvas-level edge which will ensure root-table FIFOs
            gp_canvas_add_edge_with_type(reinterpret_cast<GP_CanvasContext*>(ctx), &ed, /*type_id=*/0);
            // Try to synchronously create the corresponding root edge for immediate publishes
            GP_TableContext* root = canvas_ensure_root_table(ctx);
                if (root) {
                    uint64_t ka = (static_cast<uint64_t>(static_cast<uint32_t>(ed.a_module)) << 32) |
                                  (static_cast<uint64_t>(static_cast<uint32_t>(ed.a_contact_idx)) << 16) |
                                  static_cast<uint64_t>(0);
                    uint64_t kb = (static_cast<uint64_t>(static_cast<uint32_t>(ed.b_module)) << 32) |
                                  (static_cast<uint64_t>(static_cast<uint32_t>(ed.b_contact_idx)) << 16) |
                                  static_cast<uint64_t>(0);
                    gp_table_add_edge(root, ka, kb);
                    // update cached edge index and writer_key if state exists
                    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_key_recorder_state.size())) {
                        void* s = ctx->module_key_recorder_state[module_idx];
                        if (s) {
                            KeyRecorderState* ks2 = reinterpret_cast<KeyRecorderState*>(s);
                            // if writer_key wasn't set (==0), use the canonical 'ka' we just created
                            if (ks2->writer_key == 0) ks2->writer_key = ka;
                            int idx = -1;
                            if (gp_table_edge_index_for_key(root, ks2->writer_key, &idx) && idx >= 0) ks2->root_edge_idx = idx;
                        }
                    }
                    (void)0;
                }
        }
    }
}

static void canvas_install_root_actions(GP_CanvasContextImpl* ctx, GP_TableContext* table) {
    if (!ctx || !table) return;
    if (ctx->root_actions_installed) return;
    gp_table_set_action_callback(table, [](void* user, int32_t action_id, const GP_TableHitBox* hit) {
        auto* c = reinterpret_cast<GP_CanvasContextImpl*>(user);
        if (!c || !hit) return;
        switch (action_id) {
            case CANVAS_ACT_SIM_SEGS_DEC:
                c->sim_segs = std::max(2, c->sim_segs - 1);
                printf("gp_canvas_on_click: sim_segs=%d\n", c->sim_segs);
                break;
            case CANVAS_ACT_SIM_SEGS_INC:
                c->sim_segs = std::min(64, c->sim_segs + 1);
                printf("gp_canvas_on_click: sim_segs=%d\n", c->sim_segs);
                break;
            case CANVAS_ACT_SIM_SLACK_DEC:
                c->sim_slack = std::max(0.0f, c->sim_slack - 0.1f);
                printf("gp_canvas_on_click: sim_slack=%.2f\n", c->sim_slack);
                break;
            case CANVAS_ACT_SIM_SLACK_INC:
                c->sim_slack = std::min(8.0f, c->sim_slack + 0.1f);
                printf("gp_canvas_on_click: sim_slack=%.2f\n", c->sim_slack);
                break;
            case CANVAS_ACT_TOOL_CANVAS_0:
            case CANVAS_ACT_TOOL_CANVAS_1:
            case CANVAS_ACT_TOOL_CANVAS_2:
            case CANVAS_ACT_TOOL_CANVAS_3: {
                int tool = static_cast<int>(action_id - CANVAS_ACT_TOOL_CANVAS_0);
                if (c->selected_tool_canvas == tool) c->selected_tool_canvas = 0; else c->selected_tool_canvas = tool;
                printf("gp_canvas_on_click: canvas tool %d toggled -> selected_tool_canvas=%d\n", tool, c->selected_tool_canvas);
                break;
            }
            case CANVAS_ACT_TOOL_EDGE_0:
            case CANVAS_ACT_TOOL_EDGE_1:
            case CANVAS_ACT_TOOL_EDGE_2:
            case CANVAS_ACT_TOOL_EDGE_3: {
                int tool = static_cast<int>(action_id - CANVAS_ACT_TOOL_EDGE_0);
                if (c->selected_tool_edge == tool) c->selected_tool_edge = -1; else c->selected_tool_edge = tool;
                printf("gp_canvas_on_click: edge tool %d toggled -> selected_tool_edge=%d\n", tool, c->selected_tool_edge);
                break;
            }
            case CANVAS_ACT_EDGE_ORDER_DEC:
            case CANVAS_ACT_EDGE_ORDER_INC: {
                int delta = (action_id == CANVAS_ACT_EDGE_ORDER_INC) ? 1 : -1;
                c->edge_order_value = std::clamp(c->edge_order_value + delta, -4, 4);
                printf("gp_canvas_on_click: edge_order_value -> %d\n", c->edge_order_value);
                break;
            }
            case CANVAS_ACT_EDGE_ORDER_TOOL: {
                c->edge_order_tool_active = c->edge_order_tool_active ? 0 : 1;
                printf("gp_canvas_on_click: edge_order_tool_active -> %d\n", c->edge_order_tool_active);
                break;
            }
            
            case CANVAS_ACT_TOOL_TABLE_0:
            case CANVAS_ACT_TOOL_TABLE_1:
            case CANVAS_ACT_TOOL_TABLE_2: {
                // debug: table-tool button clicked (before state change)
                printf("gp_canvas_on_click: table-tool button clicked action_id=%d selected_tool_table(before)=%d\n", action_id, c->selected_tool_table);
                int tool = static_cast<int>(action_id - CANVAS_ACT_TOOL_TABLE_0);
                if (tool == 2) {
                    c->tool_menu_open = !c->tool_menu_open;
                    c->selected_tool_table = c->tool_menu_open ? tool : 0;
                } else {
                    if (c->selected_tool_table == tool) c->selected_tool_table = 0; else c->selected_tool_table = tool;
                    c->tool_menu_open = false;
                }
                printf("gp_canvas_on_click: table tool %d toggled -> selected_tool_table=%d\n", tool, c->selected_tool_table);
                break;
            }
            case CANVAS_ACT_IO_COUNT_DEC:
            case CANVAS_ACT_IO_COUNT_INC: {
                int delta = (action_id == CANVAS_ACT_IO_COUNT_INC) ? 1 : -1;
                c->io_attachment_count = std::clamp(c->io_attachment_count + delta, 1, 32);
                printf("gp_canvas_on_click: io_attachment_count -> %d\n", c->io_attachment_count);
                break;
            }
            case CANVAS_ACT_IO_CONSUMER_ADD:
            case CANVAS_ACT_IO_PRODUCER_ADD: {
                int focused = c->focused_module;
                if (focused >= 0 && focused < static_cast<int>(c->modules.size())) {
                    bool is_input = (action_id == CANVAS_ACT_IO_CONSUMER_ADD);
                    canvas_append_io_row(c, focused, is_input, c->io_attachment_count);
                    printf("gp_canvas_on_click: module=%d add %s row (attachments=%d)\n",
                        focused,
                        is_input ? "consumer" : "producer",
                        c->io_attachment_count);
                }
                break;
            }
            case CANVAS_ACT_MODULE_LED:
                if (c->dispatch_module_idx >= 0) {
                    canvas_handle_module_led_hit(c, c->dispatch_module_idx, *hit);
                }
                break;
            case CANVAS_ACT_MENU_TOOL_ADD:
                canvas_push_tool_to_focused(c, ModuleToolKind::Add);
                break;
            case CANVAS_ACT_MENU_TOOL_SUB:
                canvas_push_tool_to_focused(c, ModuleToolKind::Subtract);
                break;
            case CANVAS_ACT_MENU_TOOL_MUL:
                canvas_push_tool_to_focused(c, ModuleToolKind::Multiply);
                break;
            case CANVAS_ACT_MENU_TOOL_DIV:
                canvas_push_tool_to_focused(c, ModuleToolKind::Divide);
                break;
            case CANVAS_ACT_MENU_TOOL_MOD:
                canvas_push_tool_to_focused(c, ModuleToolKind::Modulo);
                break;
            case CANVAS_ACT_MENU_TOOL_KEYBOARD:
                canvas_push_tool_to_focused(c, ModuleToolKind::KeyboardListener);
                break;
            case CANVAS_ACT_MENU_TOOL_MOUSE:
                canvas_push_tool_to_focused(c, ModuleToolKind::MouseListener);
                break;
            case CANVAS_ACT_MENU_TOOL_STACK:
                canvas_push_tool_to_focused(c, ModuleToolKind::StackDisplay);
                break;
            default:
                break;
        }
    }, ctx);
    gp_table_set_actions(table, kCanvasRootActions, static_cast<int>(sizeof(kCanvasRootActions) / sizeof(kCanvasRootActions[0])));
    ctx->root_actions_installed = 1;
}

static GP_TableContext* canvas_ensure_root_table(GP_CanvasContextImpl* ctx) {
    if (!ctx) return nullptr;
    if (!ctx->container_table) {
        ctx->container_table = gp_table_create(nullptr);
        ctx->container_table_owned = 1;
        ctx->root_actions_installed = 0;
    }
    canvas_install_root_actions(ctx, ctx->container_table);
    return ctx->container_table;
}

static RopeSim* canvas_root_sim(GP_CanvasContextImpl* ctx) {
    if (!ctx || !ctx->container_table) return nullptr;
    return gp_table_get_rope_sim(ctx->container_table);
}

static RopeSim* canvas_require_root_sim(GP_CanvasContextImpl* ctx) {
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return nullptr;
    RopeSim* sim = gp_table_get_rope_sim(root);
    if (!sim) {
        sim = rope_sim_create(1024, 64);
        gp_table_attach_rope_sim(root, sim, 1);
    }
    return sim;
}

static void canvas_attach_tables_to_root_sim(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    RopeSim* sim = canvas_root_sim(ctx);
    if (!sim) return;
    for (size_t i = 0; i < ctx->module_tables.size(); ++i) {
        if (ctx->module_tables[i]) gp_table_attach_rope_sim(ctx->module_tables[i], sim, 0);
    }
}

static int canvas_dispatch_root_hit(GP_CanvasContextImpl* ctx, const GP_TableHitBox& hit) {
    if (!ctx) return 0;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return 0;
    return gp_table_dispatch_hit(root, &hit);
}

static int canvas_dispatch_root_action(GP_CanvasContextImpl* ctx, int action_id) {
    if (!ctx) return 0;
    GP_TableHitBox hb{};
    hb.part = GP_TABLE_HIT_CELL;
    hb.aux0 = action_id;
    return canvas_dispatch_root_hit(ctx, hb);
}

// Query attached table for input/output IO key counts. If table is null,
// returns zero counts.
static void get_table_io_counts(GP_TableContext* t, int &out_in_count, int &out_out_count) {
    out_in_count = 0; out_out_count = 0;
    if (!t) return;
    const int cap = 4096;
    std::vector<unsigned long long> keys(cap);
    int nin = gp_table_enumerate_io_keys(t, 0, keys.data(), cap);
    if (nin > 0) out_in_count = nin;
    int nout = gp_table_enumerate_io_keys(t, 1, keys.data(), cap);
    if (nout > 0) out_out_count = nout;
}

static void canvas_append_io_row(GP_CanvasContextImpl* ctx, int module_idx, bool is_input, int attachment_count) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[module_idx]) return;
    if (module_idx >= static_cast<int>(ctx->module_io_input_rows.size())) ctx->module_io_input_rows.resize(module_idx + 1);
    if (module_idx >= static_cast<int>(ctx->module_io_output_rows.size())) ctx->module_io_output_rows.resize(module_idx + 1);
    int count = std::clamp(attachment_count, 1, 32);
    ensure_module_row_order(ctx, module_idx);
    if (is_input) {
        ctx->module_io_input_rows[module_idx].push_back(count);
        if (module_idx < static_cast<int>(ctx->module_io_rows.size())) {
            ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Input, 0, ModuleToolKind::None, count});
        }
    } else {
        ctx->module_io_output_rows[module_idx].push_back(count);
        if (module_idx < static_cast<int>(ctx->module_io_rows.size())) {
            ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Output, 0, ModuleToolKind::None, count});
        }
    }
}

static uint64_t canvas_root_key_for_contact(int module_idx, int contact_idx) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) |
           (static_cast<uint64_t>(static_cast<uint32_t>(contact_idx)) << 16) |
           static_cast<uint64_t>(0);
}

static void canvas_collect_edges_for_contact(const GP_CanvasContextImpl* ctx, int module_idx, int contact_idx, std::vector<int> &out_indices) {
    out_indices.clear();
    if (!ctx) return;
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto &e = ctx->edges[i].desc;
        if ((e.a_module == module_idx && e.a_contact_idx == contact_idx) ||
            (e.b_module == module_idx && e.b_contact_idx == contact_idx)) {
            out_indices.push_back(static_cast<int>(i));
        }
    }
}

static void canvas_set_delta_mode_for_edge(GP_CanvasContextImpl* ctx, const GP_CanvasEdgeDesc &desc, bool delta_mode) {
    if (!ctx) return;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return;
    uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
    uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
    int edge_idx = -1;
    if (gp_table_edge_index_for_pair(root, ka, kb, &edge_idx)) {
        gp_table_edge_set_delta_mode(root, edge_idx, delta_mode ? 1 : 0);
    } else if (gp_table_edge_index_for_pair(root, kb, ka, &edge_idx)) {
        gp_table_edge_set_delta_mode(root, edge_idx, delta_mode ? 1 : 0);
    }
}

static void canvas_set_order_mode_for_edge(GP_CanvasContextImpl* ctx, const GP_CanvasEdgeDesc &desc, int order_mode) {
    if (!ctx) return;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return;
    uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
    uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
    int edge_idx = -1;
    if (gp_table_edge_index_for_pair(root, ka, kb, &edge_idx)) {
        gp_table_edge_set_order_mode(root, edge_idx, order_mode);
    } else if (gp_table_edge_index_for_pair(root, kb, ka, &edge_idx)) {
        gp_table_edge_set_order_mode(root, edge_idx, order_mode);
    }
}

static void canvas_remove_edge_at(GP_CanvasContextImpl* ctx, int edge_idx) {
    if (!ctx || edge_idx < 0 || edge_idx >= static_cast<int>(ctx->edges.size())) return;
    const auto desc = ctx->edges[static_cast<size_t>(edge_idx)].desc;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (root) {
        uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
        uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
        gp_table_remove_edge_pair(root, ka, kb);
    }
    ctx->edges.erase(ctx->edges.begin() + edge_idx);
}

static float point_segment_distance_sq(float px, float py, float ax, float ay, float bx, float by) {
    float vx = bx - ax;
    float vy = by - ay;
    float wx = px - ax;
    float wy = py - ay;
    float c1 = vx * wx + vy * wy;
    if (c1 <= 0.0f) {
        float dx = px - ax;
        float dy = py - ay;
        return dx * dx + dy * dy;
    }
    float c2 = vx * vx + vy * vy;
    if (c2 <= c1) {
        float dx = px - bx;
        float dy = py - by;
        return dx * dx + dy * dy;
    }
    float t = c1 / c2;
    float projx = ax + t * vx;
    float projy = ay + t * vy;
    float dx = px - projx;
    float dy = py - projy;
    return dx * dx + dy * dy;
}

static int canvas_pick_edge_by_rope(GP_CanvasContextImpl* ctx, int world_x, int world_y, float max_dist) {
    if (!ctx) return -1;
    RopeSim* sim = canvas_root_sim(ctx);
    if (!sim) return -1;
    float max_dist_sq = max_dist * max_dist;
    int best_edge = -1;
    float best_dist = max_dist_sq;
    for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
        int ridx = ctx->edges[ei].rope_idx;
        if (ridx < 0) continue;
        int vc = rope_sim_get_vertex_count(sim, ridx);
        if (vc < 2) continue;
        std::vector<float> verts(static_cast<size_t>(vc) * 2u);
        int got = rope_sim_get_vertices(sim, ridx, verts.data(), static_cast<int>(verts.size()));
        if (got < 2) continue;
        float dx0 = verts[0] - static_cast<float>(world_x);
        float dy0 = verts[1] - static_cast<float>(world_y);
        float dx1 = verts[(got - 1) * 2] - static_cast<float>(world_x);
        float dy1 = verts[(got - 1) * 2 + 1] - static_cast<float>(world_y);
        if ((dx0 * dx0 + dy0 * dy0) <= max_dist_sq) continue;
        if ((dx1 * dx1 + dy1 * dy1) <= max_dist_sq) continue;
        for (int vi = 0; vi < got - 1; ++vi) {
            float ax = verts[vi * 2];
            float ay = verts[vi * 2 + 1];
            float bx = verts[(vi + 1) * 2];
            float by = verts[(vi + 1) * 2 + 1];
            float dist_sq = point_segment_distance_sq(static_cast<float>(world_x), static_cast<float>(world_y), ax, ay, bx, by);
            if (dist_sq <= best_dist) {
                best_dist = dist_sq;
                best_edge = static_cast<int>(ei);
            }
        }
    }
}

static uint64_t canvas_root_key_for_contact(int module_idx, int contact_idx) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) |
           (static_cast<uint64_t>(static_cast<uint32_t>(contact_idx)) << 16) |
           static_cast<uint64_t>(0);
}

static void canvas_collect_edges_for_contact(const GP_CanvasContextImpl* ctx, int module_idx, int contact_idx, std::vector<int> &out_indices) {
    out_indices.clear();
    if (!ctx) return;
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto &e = ctx->edges[i].desc;
        if ((e.a_module == module_idx && e.a_contact_idx == contact_idx) ||
            (e.b_module == module_idx && e.b_contact_idx == contact_idx)) {
            out_indices.push_back(static_cast<int>(i));
        }
    }
}

static void canvas_set_delta_mode_for_edge(GP_CanvasContextImpl* ctx, const GP_CanvasEdgeDesc &desc, bool delta_mode) {
    if (!ctx) return;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return;
    uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
    uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
    int edge_idx = -1;
    if (gp_table_edge_index_for_pair(root, ka, kb, &edge_idx)) {
        gp_table_edge_set_delta_mode(root, edge_idx, delta_mode ? 1 : 0);
    } else if (gp_table_edge_index_for_pair(root, kb, ka, &edge_idx)) {
        gp_table_edge_set_delta_mode(root, edge_idx, delta_mode ? 1 : 0);
    }
}

static void canvas_set_order_mode_for_edge(GP_CanvasContextImpl* ctx, const GP_CanvasEdgeDesc &desc, int order_mode) {
    if (!ctx) return;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return;
    uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
    uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
    int edge_idx = -1;
    if (gp_table_edge_index_for_pair(root, ka, kb, &edge_idx)) {
        gp_table_edge_set_order_mode(root, edge_idx, order_mode);
    } else if (gp_table_edge_index_for_pair(root, kb, ka, &edge_idx)) {
        gp_table_edge_set_order_mode(root, edge_idx, order_mode);
    }
}

static void canvas_remove_edge_at(GP_CanvasContextImpl* ctx, int edge_idx) {
    if (!ctx || edge_idx < 0 || edge_idx >= static_cast<int>(ctx->edges.size())) return;
    const auto desc = ctx->edges[static_cast<size_t>(edge_idx)].desc;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (root) {
        uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
        uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
        gp_table_remove_edge_pair(root, ka, kb);
    }
    ctx->edges.erase(ctx->edges.begin() + edge_idx);
}

static float point_segment_distance_sq(float px, float py, float ax, float ay, float bx, float by) {
    float vx = bx - ax;
    float vy = by - ay;
    float wx = px - ax;
    float wy = py - ay;
    float c1 = vx * wx + vy * wy;
    if (c1 <= 0.0f) {
        float dx = px - ax;
        float dy = py - ay;
        return dx * dx + dy * dy;
    }
    float c2 = vx * vx + vy * vy;
    if (c2 <= c1) {
        float dx = px - bx;
        float dy = py - by;
        return dx * dx + dy * dy;
    }
    float t = c1 / c2;
    float projx = ax + t * vx;
    float projy = ay + t * vy;
    float dx = px - projx;
    float dy = py - projy;
    return dx * dx + dy * dy;
}

static int canvas_pick_edge_by_rope(GP_CanvasContextImpl* ctx, int world_x, int world_y, float max_dist) {
    if (!ctx) return -1;
    RopeSim* sim = canvas_root_sim(ctx);
    if (!sim) return -1;
    float max_dist_sq = max_dist * max_dist;
    int best_edge = -1;
    float best_dist = max_dist_sq;
    for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
        int ridx = ctx->edges[ei].rope_idx;
        if (ridx < 0) continue;
        int vc = rope_sim_get_vertex_count(sim, ridx);
        if (vc < 2) continue;
        std::vector<float> verts(static_cast<size_t>(vc) * 2u);
        int got = rope_sim_get_vertices(sim, ridx, verts.data(), static_cast<int>(verts.size()));
        if (got < 2) continue;
        float dx0 = verts[0] - static_cast<float>(world_x);
        float dy0 = verts[1] - static_cast<float>(world_y);
        float dx1 = verts[(got - 1) * 2] - static_cast<float>(world_x);
        float dy1 = verts[(got - 1) * 2 + 1] - static_cast<float>(world_y);
        if ((dx0 * dx0 + dy0 * dy0) <= max_dist_sq) continue;
        if ((dx1 * dx1 + dy1 * dy1) <= max_dist_sq) continue;
        for (int vi = 0; vi < got - 1; ++vi) {
            float ax = verts[vi * 2];
            float ay = verts[vi * 2 + 1];
            float bx = verts[(vi + 1) * 2];
            float by = verts[(vi + 1) * 2 + 1];
            float dist_sq = point_segment_distance_sq(static_cast<float>(world_x), static_cast<float>(world_y), ax, ay, bx, by);
            if (dist_sq <= best_dist) {
                best_dist = dist_sq;
                best_edge = static_cast<int>(ei);
            }
        }
    }
}

static uint64_t canvas_root_key_for_contact(int module_idx, int contact_idx) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) |
           (static_cast<uint64_t>(static_cast<uint32_t>(contact_idx)) << 16) |
           static_cast<uint64_t>(0);
}

static void canvas_collect_edges_for_contact(const GP_CanvasContextImpl* ctx, int module_idx, int contact_idx, std::vector<int> &out_indices) {
    out_indices.clear();
    if (!ctx) return;
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto &e = ctx->edges[i].desc;
        if ((e.a_module == module_idx && e.a_contact_idx == contact_idx) ||
            (e.b_module == module_idx && e.b_contact_idx == contact_idx)) {
            out_indices.push_back(static_cast<int>(i));
        }
    }
}

static void canvas_set_delta_mode_for_edge(GP_CanvasContextImpl* ctx, const GP_CanvasEdgeDesc &desc, bool delta_mode) {
    if (!ctx) return;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return;
    uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
    uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
    int edge_idx = -1;
    if (gp_table_edge_index_for_pair(root, ka, kb, &edge_idx)) {
        gp_table_edge_set_delta_mode(root, edge_idx, delta_mode ? 1 : 0);
    } else if (gp_table_edge_index_for_pair(root, kb, ka, &edge_idx)) {
        gp_table_edge_set_delta_mode(root, edge_idx, delta_mode ? 1 : 0);
    }
}

static void canvas_set_order_mode_for_edge(GP_CanvasContextImpl* ctx, const GP_CanvasEdgeDesc &desc, int order_mode) {
    if (!ctx) return;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return;
    uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
    uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
    int edge_idx = -1;
    if (gp_table_edge_index_for_pair(root, ka, kb, &edge_idx)) {
        gp_table_edge_set_order_mode(root, edge_idx, order_mode);
    } else if (gp_table_edge_index_for_pair(root, kb, ka, &edge_idx)) {
        gp_table_edge_set_order_mode(root, edge_idx, order_mode);
    }
}

static void canvas_remove_edge_at(GP_CanvasContextImpl* ctx, int edge_idx) {
    if (!ctx || edge_idx < 0 || edge_idx >= static_cast<int>(ctx->edges.size())) return;
    const auto desc = ctx->edges[static_cast<size_t>(edge_idx)].desc;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (root) {
        uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
        uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
        gp_table_remove_edge_pair(root, ka, kb);
    }
    ctx->edges.erase(ctx->edges.begin() + edge_idx);
}

static float point_segment_distance_sq(float px, float py, float ax, float ay, float bx, float by) {
    float vx = bx - ax;
    float vy = by - ay;
    float wx = px - ax;
    float wy = py - ay;
    float c1 = vx * wx + vy * wy;
    if (c1 <= 0.0f) {
        float dx = px - ax;
        float dy = py - ay;
        return dx * dx + dy * dy;
    }
    float c2 = vx * vx + vy * vy;
    if (c2 <= c1) {
        float dx = px - bx;
        float dy = py - by;
        return dx * dx + dy * dy;
    }
    float t = c1 / c2;
    float projx = ax + t * vx;
    float projy = ay + t * vy;
    float dx = px - projx;
    float dy = py - projy;
    return dx * dx + dy * dy;
}

static int canvas_pick_edge_by_rope(GP_CanvasContextImpl* ctx, int world_x, int world_y, float max_dist) {
    if (!ctx) return -1;
    RopeSim* sim = canvas_root_sim(ctx);
    if (!sim) return -1;
    float max_dist_sq = max_dist * max_dist;
    int best_edge = -1;
    float best_dist = max_dist_sq;
    for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
        int ridx = ctx->edges[ei].rope_idx;
        if (ridx < 0) continue;
        int vc = rope_sim_get_vertex_count(sim, ridx);
        if (vc < 2) continue;
        std::vector<float> verts(static_cast<size_t>(vc) * 2u);
        int got = rope_sim_get_vertices(sim, ridx, verts.data(), static_cast<int>(verts.size()));
        if (got < 2) continue;
        float dx0 = verts[0] - static_cast<float>(world_x);
        float dy0 = verts[1] - static_cast<float>(world_y);
        float dx1 = verts[(got - 1) * 2] - static_cast<float>(world_x);
        float dy1 = verts[(got - 1) * 2 + 1] - static_cast<float>(world_y);
        if ((dx0 * dx0 + dy0 * dy0) <= max_dist_sq) continue;
        if ((dx1 * dx1 + dy1 * dy1) <= max_dist_sq) continue;
        for (int vi = 0; vi < got - 1; ++vi) {
            float ax = verts[vi * 2];
            float ay = verts[vi * 2 + 1];
            float bx = verts[(vi + 1) * 2];
            float by = verts[(vi + 1) * 2 + 1];
            float dist_sq = point_segment_distance_sq(static_cast<float>(world_x), static_cast<float>(world_y), ax, ay, bx, by);
            if (dist_sq <= best_dist) {
                best_dist = dist_sq;
                best_edge = static_cast<int>(ei);
            }
        }
    }
    return best_edge;
}

// Ensure the attached table reflects the canvas' requested IO counts for the module.
// Creates columns/rows and LED cells (GP_TABLE_CELL_LEDS_ARG) to display counts.
static void sync_module_table_io_layout(GP_CanvasContextImpl* ctx, int module_idx) {
    // Simplified behavior: only update the focused module's attached table
    // to contain a single LED cell representing the input count from the
    // control-bar. This avoids complex layout logic while producing a
    // stateful LED cell that the existing table rasterizer will render.
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    // Stage modules get their own fixed IO layout with explicit ports.
    if (module_idx < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[module_idx]) {
        GP_TableContext* t = (module_idx < static_cast<int>(ctx->module_tables.size())) ? ctx->module_tables[module_idx] : nullptr;
        GP_StageContext* st = (module_idx < static_cast<int>(ctx->module_stages.size())) ? ctx->module_stages[module_idx] : nullptr;
        if (!t) return;
        // ensure IO counts for stage ports (one in, one out)
        if (module_idx >= static_cast<int>(ctx->module_io_in_count.size())) ctx->module_io_in_count.resize(module_idx + 1, 0);
        if (module_idx >= static_cast<int>(ctx->module_io_out_count.size())) ctx->module_io_out_count.resize(module_idx + 1, 0);
        ctx->module_io_in_count[module_idx] = 1;
        ctx->module_io_out_count[module_idx] = 1;

        // keep the stage table compact; give it a faint opaque backing so LEDs are visible
        GP_TableStyle st_style{};
        gp_table_get_style(t, &st_style);
        const auto &mod = ctx->modules[module_idx];
        st_style.width_px = std::max(1, mod.w);
        st_style.row_h_px = 18;
        // avoid pushing columns off-screen by keeping the name gutter tiny
        st_style.name_w_px = 8;
        // keep some translucency so the stage is still visible beneath the table surface
        st_style.bg_rgba[0] = 18; st_style.bg_rgba[1] = 18; st_style.bg_rgba[2] = 24; st_style.bg_rgba[3] = 180;
        st_style.bg_sel_rgba[0] = 28; st_style.bg_sel_rgba[1] = 28; st_style.bg_sel_rgba[2] = 36; st_style.bg_sel_rgba[3] = 200;
        gp_table_set_style(t, &st_style);

        // Column sizing: content + input LED + output LED. Keep totals within table width.
        int avail_w = std::max(16, st_style.width_px - st_style.name_w_px);
        GP_TableColumn cols[3];
        cols[0].kind = GP_TABLE_CELL_TEXT;
        cols[0].width_px = std::max(64, avail_w - 144);
        cols[0].align = 0;
        cols[1].kind = GP_TABLE_CELL_LEDS_ARG;
        cols[1].width_px = 72;
        cols[1].align = 0;
        cols[2].kind = GP_TABLE_CELL_LEDS_ARG;
        cols[2].width_px = 72;
        cols[2].align = 0;
        gp_table_set_columns(t, cols, 3);

        auto fill_text = [](GP_TableCell &c, const char* txt) {
            c.kind = GP_TABLE_CELL_TEXT;
            std::snprintf(c.text, sizeof(c.text), "%s", txt);
        };
        auto fill_led_single = [](GP_TableCell &c) {
            c.kind = GP_TABLE_CELL_LEDS_ARG;
            c.value = 1.0f;      // one LED
            c.flags = 1u;        // on mask (bit0)
            c.reserved0 = 1;     // active mask bit0
        };

        // Rows: input port, output port, and a flex "content" row that the stage
        // render is bound into as a GP_TABLE_CELL_IMAGE.
        GP_TableRow rows[3];
        std::memset(rows, 0, sizeof(rows));
        for (int i = 0; i < 3; ++i) {
            rows[i].kind = GP_TABLE_ROW_DEVICE;
            rows[i].depth = 0;
            rows[i].expanded = 1;
            rows[i].selected = 0;
            rows[i].cell_count = 3;
        }
        fill_text(rows[0].cells[0], LABEL_IO_INPUT);
        fill_led_single(rows[0].cells[1]);
        fill_text(rows[0].cells[2], "");
        fill_text(rows[1].cells[0], LABEL_IO_OUTPUT);
        fill_text(rows[1].cells[1], "");
        fill_led_single(rows[1].cells[2]);

        rows[2].cells[0].kind = GP_TABLE_CELL_IMAGE;
        rows[2].cells[0].image = nullptr;
        fill_text(rows[2].cells[1], "");
        fill_text(rows[2].cells[2], "");
        // reserved0 < 0 => flex row (absorbs remaining vertical space when rendered into a larger buffer)
        rows[2].reserved0 = -1;
        gp_table_set_rows(t, rows, 3);

        // update metadata so control surfaces know there are two contacts
        if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) ctx->module_io_rows.resize(module_idx + 1);
        ctx->module_io_rows[module_idx].clear();
        ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Input, 0, ModuleToolKind::None, 1});
        ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Output, 1, ModuleToolKind::None, 1});
        if (module_idx >= static_cast<int>(ctx->module_table_rows.size())) ctx->module_table_rows.resize(module_idx + 1);
        ctx->module_table_rows[module_idx].clear();
        if (module_idx >= static_cast<int>(ctx->module_stack_snapshots.size())) ctx->module_stack_snapshots.resize(module_idx + 1);
        ctx->module_stack_snapshots[module_idx].clear();

        auto bind_port = [&](int row_idx, int led_col_idx, bool is_output, int channel) {
            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(row_idx)) << 32) |
                (static_cast<uint64_t>(static_cast<uint32_t>(led_col_idx)) << 16) | // LED column index
                static_cast<uint64_t>(0); // single LED
            gp_table_set_key_type_hint(t, key, /*type_id=*/0, is_output ? 0 : 1, is_output ? 1 : 0);
            if (st) gp_table_enqueue_bind_stage_port(t, key, st, is_output ? 1 : 0, channel);
        };
        bind_port(0, /*led_col_idx=*/1, /*is_output=*/false, /*channel=*/0);
        bind_port(1, /*led_col_idx=*/2, /*is_output=*/true,  /*channel=*/0);
        return;
    }
    if (module_idx >= static_cast<int>(ctx->module_tables.size())) return;
    GP_TableContext* t = ctx->module_tables[module_idx];
    if (!t) return;
    // only touch the currently focused module to avoid clobbering other tables
    if (ctx->focused_module != module_idx) return;
    if (module_idx >= static_cast<int>(ctx->module_io_input_rows.size())) ctx->module_io_input_rows.resize(module_idx + 1);
    if (module_idx >= static_cast<int>(ctx->module_io_output_rows.size())) ctx->module_io_output_rows.resize(module_idx + 1);
    int legacy_in_count = (module_idx < static_cast<int>(ctx->module_io_in_count.size())) ? ctx->module_io_in_count[module_idx] : 0;
    int legacy_out_count = (module_idx < static_cast<int>(ctx->module_io_out_count.size())) ? ctx->module_io_out_count[module_idx] : 0;
    auto &input_rows = ctx->module_io_input_rows[module_idx];
    auto &output_rows = ctx->module_io_output_rows[module_idx];
    if (input_rows.empty() && legacy_in_count > 0) input_rows.assign(static_cast<size_t>(legacy_in_count), 1);
    if (output_rows.empty() && legacy_out_count > 0) output_rows.assign(static_cast<size_t>(legacy_out_count), 1);
    ensure_module_row_order(ctx, module_idx);
    if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) return;
    auto &ordered_rows = ctx->module_io_rows[module_idx];
    int in_count = 0;
    int out_count = 0;
    for (const auto &row : ordered_rows) {
        if (row.kind == ModuleRowKind::Input) in_count += std::max(0, row.attachment_count);
        if (row.kind == ModuleRowKind::Output) out_count += std::max(0, row.attachment_count);
    }
    if (module_idx >= static_cast<int>(ctx->module_io_in_count.size())) ctx->module_io_in_count.resize(module_idx + 1, 0);
    if (module_idx >= static_cast<int>(ctx->module_io_out_count.size())) ctx->module_io_out_count.resize(module_idx + 1, 0);
    ctx->module_io_in_count[module_idx] = in_count;
    ctx->module_io_out_count[module_idx] = out_count;

    // always rebuild the table layout to reflect ordered IO rows

    // If there's no attached table but the UI has non-zero IO counts, create
    // a canvas-owned table so the user sees the LED cells immediately.
    if (!t && (in_count > 0 || (module_idx < static_cast<int>(ctx->module_io_out_count.size()) && ctx->module_io_out_count[module_idx] > 0))) {
        GP_TableContext* nt = gp_table_create(nullptr);
        gp_table_set_prospective_mode(nt, 1);
        gp_table_prospective_set_params(nt, 8, 4.0f, 0.0f);
        t = nt;
    }
    if (!t) return;

    // Create columns: label, LED grid.
    GP_TableColumn cols[2];
    cols[0].kind = GP_TABLE_CELL_TEXT; cols[0].width_px = 80; cols[0].align = 0;
    cols[1].kind = GP_TABLE_CELL_LEDS_ARG; cols[1].width_px = 120; cols[1].align = 0;
    gp_table_set_columns(t, cols, 2);

    GP_TableStyle st{};
    gp_table_get_style(t, &st);
    int base_row_h = st.row_h_px > 0 ? st.row_h_px : 20;
    int stack_value_row_h = std::max(12, base_row_h - 6);

    int total_rows = static_cast<int>(ordered_rows.size());
    if (total_rows <= 0) {
        GP_TableRow prow{}; memset(&prow, 0, sizeof(prow));
        prow.kind = GP_TABLE_ROW_HEADER; prow.depth = 0; prow.expanded = 1; prow.selected = 0;
        prow.cell_count = 2;
        prow.cells[0].kind = GP_TABLE_CELL_TEXT;
        prow.cells[1].kind = GP_TABLE_CELL_TEXT;
        gp_table_set_rows(t, &prow, 1);
        if (module_idx >= static_cast<int>(ctx->module_table_rows.size())) ctx->module_table_rows.resize(module_idx + 1);
        ctx->module_table_rows[module_idx].clear();
        return;
    }

    auto fill_text_cell = [](GP_TableCell &cell, const char* text) {
        cell.kind = GP_TABLE_CELL_TEXT;
        size_t len = std::min<std::size_t>(std::strlen(text), sizeof(cell.text) - 1);
        std::memcpy(cell.text, text, len);
        cell.text[len] = '\0';
    };

    auto fill_led_cell = [](GP_TableCell &cell, int count) {
        cell.kind = GP_TABLE_CELL_LEDS_ARG;
        int clamped = std::clamp(count, 1, 32);
        cell.value = static_cast<float>(clamped);
        uint32_t mask = (clamped >= 32) ? 0xFFFFFFFFu : ((1u << clamped) - 1u);
        cell.flags = mask;
        cell.reserved0 = static_cast<int32_t>(mask);
    };

    std::vector<GP_TableRow> rows;
    rows.reserve(static_cast<size_t>(total_rows));
    std::vector<ModuleIORow> row_meta;
    row_meta.reserve(static_cast<size_t>(total_rows));

    auto append_row = [&](const char* label, ModuleRowKind kind, int contact_idx, int attachment_count, ModuleToolKind tool_kind) {
        GP_TableRow r{};
        memset(&r, 0, sizeof(r));
        r.kind = GP_TABLE_ROW_DEVICE;
        r.depth = 0;
        r.expanded = 1;
        r.selected = 0;
        r.cell_count = (kind == ModuleRowKind::Tool) ? 1 : 2;
        fill_text_cell(r.cells[0], label);
        if (kind != ModuleRowKind::Tool) {
            fill_led_cell(r.cells[1], attachment_count);
        }
        rows.push_back(r);
        row_meta.push_back({kind, contact_idx, tool_kind, attachment_count});
    };

    auto append_stack_display_rows = [&](int logical_row_idx) {
        GP_TableRow header{};
        memset(&header, 0, sizeof(header));
        header.kind = GP_TABLE_ROW_HEADER;
        header.depth = 0;
        header.expanded = 1;
        header.selected = 0;
        header.cell_count = 1;
        fill_text_cell(header.cells[0], LABEL_TOOL_STACK);
        rows.push_back(header);
        row_meta.push_back({ModuleRowKind::Tool, -1, ModuleToolKind::StackDisplay, 0});

        const std::vector<float>* snapshot = nullptr;
        if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_stack_snapshots.size())) {
            auto &snapshots = ctx->module_stack_snapshots[module_idx];
            auto it = snapshots.find(logical_row_idx);
            if (it != snapshots.end()) snapshot = &it->second;
        }
        if (!snapshot || snapshot->empty()) return;

        int display_idx = 0;
        for (auto it = snapshot->rbegin(); it != snapshot->rend(); ++it, ++display_idx) {
            GP_TableRow vr{};
            memset(&vr, 0, sizeof(vr));
            vr.kind = GP_TABLE_ROW_NOTE;
            vr.depth = 0;
            vr.expanded = 1;
            vr.selected = 0;
            vr.cell_count = 1;
            vr.reserved0 = stack_value_row_h;
            char buf[96];
            std::snprintf(buf, sizeof(buf), "%d: %.6g", display_idx, *it);
            fill_text_cell(vr.cells[0], buf);
            rows.push_back(vr);
            row_meta.push_back({ModuleRowKind::Tool, -1, ModuleToolKind::StackDisplay, 0});
        }
    };

    int input_contact = 0;
    int output_contact = 0;
    for (size_t row_idx = 0; row_idx < ordered_rows.size(); ++row_idx) {
        const auto &row = ordered_rows[row_idx];
        if (row.kind == ModuleRowKind::Input) {
            int attachments = std::clamp(row.attachment_count, 1, 32);
            append_row(LABEL_IO_INPUT, ModuleRowKind::Input, input_contact, attachments, ModuleToolKind::None);
            input_contact += attachments;
        } else if (row.kind == ModuleRowKind::Output) {
            int attachments = std::clamp(row.attachment_count, 1, 32);
            append_row(LABEL_IO_OUTPUT, ModuleRowKind::Output, in_count + output_contact, attachments, ModuleToolKind::None);
            output_contact += attachments;
        } else {
            if (row.tool == ModuleToolKind::StackDisplay) {
                append_stack_display_rows(static_cast<int>(row_idx));
            } else {
                append_row(tool_label(row.tool), ModuleRowKind::Tool, -1, 0, row.tool);
            }
        }
    }

    gp_table_set_rows(t, rows.data(), static_cast<int>(rows.size()));
    if (module_idx >= static_cast<int>(ctx->module_table_rows.size())) ctx->module_table_rows.resize(module_idx + 1);
    ctx->module_table_rows[module_idx] = row_meta;
        // Annotate LED keys for created rows with explicit input/output hints
        // so the table renderer's glow pass can recognize IO roles.
        for (int ri = 0; ri < static_cast<int>(rows.size()); ++ri) {
            ModuleRowKind kind = row_meta[ri].kind;
            if (kind == ModuleRowKind::Tool) continue;
            bool is_input = (kind == ModuleRowKind::Input);
            int col_idx = 1; // LED cell column
        if (col_idx >= rows[ri].cell_count) continue;
        const GP_TableCell &cell = rows[ri].cells[col_idx];
        int led_count = 9;
        if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
            int c = std::max(0, std::min(32, static_cast<int>(cell.value)));
            if (c == 0) c = static_cast<int>(cell.flags & 0xFFu);
            if (c == 0) c = 12;
            led_count = c;
        } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
            led_count = 8;
        }
        for (int li = 0; li < led_count; ++li) {
            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(ri)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(col_idx)) << 16) | static_cast<uint64_t>(static_cast<uint32_t>(li));
            gp_table_set_key_type_hint(t, key, 0, is_input ? 1 : 0, is_input ? 0 : 1);
        }
    }
    
    auto ensure_layout = [&](std::vector<MolexLayoutInfo> &arr) {
        if (module_idx >= static_cast<int>(arr.size())) arr.resize(module_idx + 1);
    };
    ensure_layout(ctx->module_input_layout);
    ensure_layout(ctx->module_output_layout);
    ctx->module_input_layout[module_idx] = make_molex_layout(module_idx, true, in_count);
    ctx->module_output_layout[module_idx] = make_molex_layout(module_idx, false, out_count);
    return;
}

static void canvas_push_tool_to_focused(GP_CanvasContextImpl* ctx, ModuleToolKind tool) {
    if (!ctx) return;
    int focused = ctx->focused_module;
    if (focused < 0 || focused >= static_cast<int>(ctx->modules.size())) return;
    if (focused >= static_cast<int>(ctx->module_tool_stack.size())) ctx->module_tool_stack.resize(focused + 1);
    ensure_module_row_order(ctx, focused);
    if (focused >= static_cast<int>(ctx->module_io_rows.size())) return;
    auto &rows = ctx->module_io_rows[focused];
    auto insert_tool_row = [&](size_t idx) {
        ModuleIORow row;
        row.kind = ModuleRowKind::Tool;
        row.contact_idx = -1;
        row.tool = tool;
        row.attachment_count = 0;
        rows.insert(rows.begin() + static_cast<ptrdiff_t>(idx), row);
    };
    if (tool == ModuleToolKind::KeyboardListener) {
        auto it = std::find_if(rows.begin(), rows.end(), [](const ModuleIORow &r) {
            return r.kind == ModuleRowKind::Tool && r.tool == ModuleToolKind::MouseListener;
        });
        if (it != rows.end()) {
            insert_tool_row(static_cast<size_t>(std::distance(rows.begin(), it) + 1));
        } else {
            rows.push_back({ModuleRowKind::Tool, -1, tool, 0});
        }
    } else if (tool == ModuleToolKind::MouseListener) {
        auto it = std::find_if(rows.begin(), rows.end(), [](const ModuleIORow &r) {
            return r.kind == ModuleRowKind::Tool && r.tool == ModuleToolKind::KeyboardListener;
        });
        if (it != rows.end()) {
            insert_tool_row(static_cast<size_t>(std::distance(rows.begin(), it)));
        } else {
            rows.push_back({ModuleRowKind::Tool, -1, tool, 0});
        }
    } else {
        rows.push_back({ModuleRowKind::Tool, -1, tool, 0});
    }
    auto &tools = ctx->module_tool_stack[focused];
    tools.clear();
    for (const auto &row : rows) {
        if (row.kind == ModuleRowKind::Tool) tools.push_back(row.tool);
    }
    sync_module_table_io_layout(ctx, focused);
    update_canvas_scroll_state(ctx, /*pull_from_container=*/false);
    ctx->tool_menu_open = false;
    ctx->selected_tool_table = 0;
}

static int canvas_handle_module_led_hit(GP_CanvasContextImpl* ctx, int module_idx, const GP_TableHitBox& found) {
    if (!ctx) return 0;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return 0;
    const auto& m = ctx->modules[module_idx];
    if (found.part != GP_TABLE_HIT_LED && found.part != GP_TABLE_HIT_LED_ARG && found.part != GP_TABLE_HIT_LED_TABLE) return 0;
    int ax = m.x + (found.x0 + found.x1) / 2;
    int ay = m.y + (found.y0 + found.y1) / 2;
    int contact_idx = resolve_contact_index(ctx, module_idx, found);
    if (contact_idx < 0) return 0;
    ctx->focused_module = module_idx;
    if (ctx->edge_order_tool_active) {
        std::vector<int> edge_indices;
        canvas_collect_edges_for_contact(ctx, module_idx, contact_idx, edge_indices);
        if (!edge_indices.empty()) {
            for (int idx : edge_indices) {
                if (idx < 0 || idx >= static_cast<int>(ctx->edges.size())) continue;
                canvas_set_order_mode_for_edge(ctx, ctx->edges[static_cast<size_t>(idx)].desc, ctx->edge_order_value);
            }
            ctx->selected.module = -1; ctx->selected.contact_idx = -1; ctx->selected.left = -1; ctx->selected.anchor_x = -1; ctx->selected.anchor_y = -1;
            if (ctx->prospective_rope_idx >= 0) ctx->prospective_rope_idx = -1;
            return 1;
        }
    }
    if (ctx->selected_tool_edge >= 0 && ctx->selected_tool_edge != 0) {
        std::vector<int> edge_indices;
        canvas_collect_edges_for_contact(ctx, module_idx, contact_idx, edge_indices);
        if (!edge_indices.empty()) {
            if (ctx->selected_tool_edge == 1) {
                std::sort(edge_indices.begin(), edge_indices.end(), std::greater<int>());
                for (int idx : edge_indices) canvas_remove_edge_at(ctx, idx);
            } else {
                bool delta_mode = (ctx->selected_tool_edge == 2);
                for (int idx : edge_indices) {
                    if (idx < 0 || idx >= static_cast<int>(ctx->edges.size())) continue;
                    canvas_set_delta_mode_for_edge(ctx, ctx->edges[static_cast<size_t>(idx)].desc, delta_mode);
                }
            }
            ctx->selected.module = -1; ctx->selected.contact_idx = -1; ctx->selected.left = -1; ctx->selected.anchor_x = -1; ctx->selected.anchor_y = -1;
            if (ctx->prospective_rope_idx >= 0) ctx->prospective_rope_idx = -1;
            return 1;
        }
    }
    // Determine producer/consumer using row metadata when available.
    bool is_producer = false;
    bool resolved_role = false;
    const std::vector<ModuleIORow>* rows_ptr = nullptr;
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_table_rows.size()) &&
        !ctx->module_table_rows[module_idx].empty()) {
        rows_ptr = &ctx->module_table_rows[module_idx];
    } else if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_io_rows.size())) {
        rows_ptr = &ctx->module_io_rows[module_idx];
    }
    if (rows_ptr) {
        const auto &rows = *rows_ptr;
        if (found.row_idx >= 0 && found.row_idx < static_cast<int>(rows.size())) {
            const auto &meta = rows[found.row_idx];
            if (meta.kind == ModuleRowKind::Input || meta.kind == ModuleRowKind::Output) {
                bool is_input = (meta.kind == ModuleRowKind::Input);
                is_producer = !is_input;
                resolved_role = true;
            }
        }
    }
    if (!resolved_role && module_idx < static_cast<int>(ctx->module_io_in_count.size()) && module_idx < static_cast<int>(ctx->module_io_out_count.size())) {
        int in_count = ctx->module_io_in_count[module_idx];
        int out_count = ctx->module_io_out_count[module_idx];
        int total = std::max(0, in_count) + std::max(0, out_count);
        if (total > 0 && contact_idx >= 0 && contact_idx < total) {
            bool is_input = (contact_idx < in_count);
            is_producer = !is_input;
            resolved_role = true;
        }
    }
    if (!resolved_role) {
        if (found.col_idx >= 0) is_producer = (found.col_idx != 0);
        else is_producer = (ax >= m.x + m.w / 2);
    }
    bool is_input = !is_producer;
    if (ctx->selected.module == -1) {
        ctx->selected.module = module_idx; ctx->selected.contact_idx = contact_idx; ctx->selected.left = is_producer ? 1 : 0;
        ctx->selected.anchor_x = ax; ctx->selected.anchor_y = ay;
        uint32_t connector_hash = lookup_molex_hash(ctx, module_idx, is_input, contact_idx);
        printf("gp_canvas_on_click: selecting table LED module=%d contact=%d producer=%d anchor=%d,%d hash=0x%08x\n",
            module_idx, contact_idx, ctx->selected.left, ax, ay, connector_hash);
        RopeSim* sim = canvas_require_root_sim(ctx);
        if (sim) {
            int ax0 = ax, ay0 = ay;
            int bx = ax, by = ay;
            int segs = ctx->sim_segs; float slack = ctx->sim_slack;
            ctx->prospective_rope_idx = rope_sim_add_rope(sim, static_cast<float>(ax0), static_cast<float>(ay0), static_cast<float>(bx), static_cast<float>(by), segs, slack);
            printf("gp_canvas_on_click: created prospective rope %d for table LED %d/%d\n", ctx->prospective_rope_idx, module_idx, contact_idx);
        }
        return 1;
    }
    bool sel_producer = (ctx->selected.left == 1);
    // allow connecting producer -> consumer only
    if (sel_producer && !is_producer) {
        GP_CanvasEdgeDesc e; e.a_module = ctx->selected.module; e.a_contact_idx = ctx->selected.contact_idx; e.b_module = module_idx; e.b_contact_idx = contact_idx;
        int ei = gp_canvas_add_edge_with_type(reinterpret_cast<GP_CanvasContext*>(ctx), &e, /*type_id=*/0);
        printf("gp_canvas_on_click: gp_canvas_add_edge returned %d\n", ei);
        if (ei >= 0 && ctx->prospective_rope_idx >= 0) { ctx->edges[ei].rope_idx = ctx->prospective_rope_idx; ctx->prospective_rope_idx = -1; }
        ctx->selected.module = -1; ctx->selected.contact_idx = -1; ctx->selected.left = -1; ctx->selected.anchor_x = -1; ctx->selected.anchor_y = -1;
        return 1;
    } else if (!sel_producer && is_producer) {
        GP_CanvasEdgeDesc e; e.a_module = module_idx; e.a_contact_idx = contact_idx; e.b_module = ctx->selected.module; e.b_contact_idx = ctx->selected.contact_idx;
        int ei = gp_canvas_add_edge_with_type(reinterpret_cast<GP_CanvasContext*>(ctx), &e, /*type_id=*/0);
        printf("gp_canvas_on_click: gp_canvas_add_edge returned %d\n", ei);
        if (ei >= 0 && ctx->prospective_rope_idx >= 0) { ctx->edges[ei].rope_idx = ctx->prospective_rope_idx; ctx->prospective_rope_idx = -1; }
        ctx->selected.module = -1; ctx->selected.contact_idx = -1; ctx->selected.left = -1; ctx->selected.anchor_x = -1; ctx->selected.anchor_y = -1;
        return 1;
    }
    // same direction: switch selection to this
    ctx->selected.module = module_idx; ctx->selected.contact_idx = contact_idx; ctx->selected.left = is_producer ? 1 : 0; ctx->selected.anchor_x = ax; ctx->selected.anchor_y = ay;
    return 1;
}
extern "C" GP_CanvasContext* gp_canvas_create(int width, int height) {
    GP_CanvasContextImpl* c = new GP_CanvasContextImpl(width, height);
    if (!g_canvas_context_singleton) g_canvas_context_singleton = c;
    canvas_ensure_root_table(c);
    c->thread_mgr = std::make_unique<ThreadManager>();
    c->thread_mgr->set_mode(ThreadManager::Mode::Scheduled);
    c->thread_mgr->start();
    // expose as global for table-layer integration
    ThreadManager::set_global(c->thread_mgr.get());
    return reinterpret_cast<GP_CanvasContext*>(c);
}

extern "C" void gp_canvas_destroy(GP_CanvasContext* ctx) {
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    if (!c) return;
    if (c->thread_mgr) {
        c->thread_mgr->stop();
        // clear global reference before destroying
        ThreadManager::set_global(nullptr);
        c->thread_mgr.reset();
    }
    // destroy any owned attached tables
    for (size_t i = 0; i < c->module_tables.size(); ++i) {
        if (c->module_tables[i] && c->module_table_owned[i]) {
            gp_table_destroy(c->module_tables[i]);
        }
    }
    // destroy any owned stages
    for (size_t i = 0; i < c->module_stages.size(); ++i) {
        if (c->module_stages[i] && i < c->module_stage_owned.size() && c->module_stage_owned[i]) {
            gp_stage_destroy(c->module_stages[i]);
        }
    }
    for (auto &bg : c->module_bg) {
        if (bg.ray) {
            raytrace2d_destroy(bg.ray);
            bg.ray = nullptr;
        }
    }
    if (c->container_table && c->container_table_owned) {
        gp_table_destroy(c->container_table);
    }
    delete c;
    if (g_canvas_context_singleton == c) g_canvas_context_singleton = nullptr;
}

extern "C" int gp_canvas_add_module(GP_CanvasContext* ctx_, const GP_CanvasModuleDesc* desc) {
    if (!ctx_ || !desc) return -1;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    int default_bg_mode = 0;
    for (const auto &bg : c->module_bg) {
        if (bg.mode == 1) { default_bg_mode = 1; break; }
    }
    GP_CanvasModuleDesc d = *desc;
    c->modules.push_back(d);
    c->module_tables.push_back(nullptr);
    c->module_table_owned.push_back(0);
    c->module_stages.push_back(nullptr);
    c->module_stage_owned.push_back(0);
    c->module_is_stage.push_back(0);
    c->module_stage_images.push_back(GP_TableImage{});
    c->module_stage_cache_rgba.emplace_back();
    c->module_stage_cache_w.push_back(0);
    c->module_stage_cache_h.push_back(0);
    c->module_stage_cache_pitch.push_back(0);
    c->module_stage_cache_mu.emplace_back(std::make_unique<std::mutex>());
    c->module_stage_integrator_mode.push_back(0);
    c->module_stage_integrator_accum.emplace_back();
    c->module_bg.emplace_back();
    c->module_bg.back().mode = default_bg_mode;
    c->module_io_in_count.push_back(0);
    c->module_io_out_count.push_back(0);
    c->module_io_input_rows.emplace_back();
    c->module_io_output_rows.emplace_back();
    c->module_input_layout.emplace_back();
    c->module_output_layout.emplace_back();
    c->module_io_rows.emplace_back();
    c->module_table_rows.emplace_back();
    c->module_stack_snapshots.emplace_back();
    c->module_tool_stack.emplace_back();
    c->module_key_recorder_state.push_back(nullptr);
    c->module_input_state.emplace_back();
    c->module_chat_text.push_back(std::string());
    c->module_chat_color.push_back(ChatCol{});
    c->module_chat_ttl.push_back(0);
    int new_idx = static_cast<int>(c->modules.size() - 1);
    // Ensure newly-added modules get a canvas-owned table so table-driven
    // hitboxes and dynamic LED cells work immediately instead of falling
    // back to legacy module contact geometry.
    gp_canvas_create_table(ctx_, new_idx);
    // Populate the table's IO layout to reflect current module IO counts
    // (this will add LED_ARG cells if module_io_in_count/out_count > 0).
    sync_module_table_io_layout(c, new_idx);
    update_canvas_scroll_state(c, /*pull_from_container=*/false);
    return new_idx;
}

extern "C" int gp_canvas_move_module(GP_CanvasContext* ctx_, int module_idx, int x, int y) {
    if (!ctx_) return -1;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->modules.size())) return -1;
    c->modules[module_idx].x = x;
    c->modules[module_idx].y = y;
    update_canvas_scroll_state(c, /*pull_from_container=*/false);
    return 1;
}

extern "C" int gp_canvas_on_click(GP_CanvasContext* ctx_, int x, int y) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    update_canvas_scroll_state(c, /*pull_from_container=*/true);
    int view_x = x;
    int view_y = y;
    int world_x = x + c->offset_x;
    int world_y = y + c->offset_y;
    printf("gp_canvas_on_click: click view=%d,%d world=%d,%d selected_module=%d selected_contact=%d selected_left=%d prospective_rope=%d\n", view_x, view_y, world_x, world_y, c->selected.module, c->selected.contact_idx, c->selected.left, c->prospective_rope_idx);
    // find contact under point
    const int pick_r = 8;
    // rope bar (top-most) - adjust sim parameters
    if (view_y >= 0 && view_y < c->rope_bar_h) {
        // simple left/right buttons: segs +/- at left, slack +/- at right
        int bw = std::max(8, c->rope_bar_h - 8);
        int spacing = 8;
        int bx = 8;
        // segs -
        if (view_x >= bx && view_x < bx + bw) {
            if (canvas_dispatch_root_action(c, CANVAS_ACT_SIM_SEGS_DEC)) return 1;
        }
        bx += bw + spacing;
        // segs +
        if (view_x >= bx && view_x < bx + bw) {
            if (canvas_dispatch_root_action(c, CANVAS_ACT_SIM_SEGS_INC)) return 1;
        }
        // slack -
        int bx2 = c->width - 8 - bw*2 - spacing;
        if (view_x >= bx2 && view_x < bx2 + bw) {
            if (canvas_dispatch_root_action(c, CANVAS_ACT_SIM_SLACK_DEC)) return 1;
        }
        // slack +
        bx2 += bw + spacing;
        if (view_x >= bx2 && view_x < bx2 + bw) {
            if (canvas_dispatch_root_action(c, CANVAS_ACT_SIM_SLACK_INC)) return 1;
        }
    }
    // check control bar button regions first — buttons are canvas-local coords (shifted down by rope_bar_h)
    if (view_y >= c->rope_bar_h && view_y < c->rope_bar_h + c->control_bar_h) {
        const int canvas_btn_count = 4;
        const int edge_btn_count = 4;
        const int table_btn_count = 3;
        const int spacing = 8;
        int by = c->rope_bar_h + 4;
        int bh = std::max(4, c->control_bar_h - 8);
        int bw = bh; // square buttons
        // left canvas group
        int bx = 8;
        int canvas_group_w = canvas_btn_count * (bw + spacing) - spacing;
        for (int bi = 0; bi < canvas_btn_count; ++bi) {
            int bx_i = bx + bi * (bw + spacing);
            if (view_x >= bx_i && view_x < bx_i + bw && view_y >= by && view_y < by + bh) {
                int action_id = CANVAS_ACT_TOOL_CANVAS_0 + bi;
                if (canvas_dispatch_root_action(c, action_id)) return 1;
            }
        }
        int bx_edge = bx + canvas_group_w + spacing * 2;
        for (int bi = 0; bi < edge_btn_count; ++bi) {
            int bx_i = bx_edge + bi * (bw + spacing);
            if (view_x >= bx_i && view_x < bx_i + bw && view_y >= by && view_y < by + bh) {
                int action_id = CANVAS_ACT_TOOL_EDGE_0 + bi;
                if (canvas_dispatch_root_action(c, action_id)) return 1;
            }
        }
        // right table group
        int group_width = table_btn_count * (bw + spacing) - spacing;
        int bx_r = std::max(8, c->width - 8 - group_width);
        for (int bi = 0; bi < table_btn_count; ++bi) {
            int bx_i = bx_r + bi * (bw + spacing);
            if (view_x >= bx_i && view_x < bx_i + bw && view_y >= by && view_y < by + bh) {
                int action_id = CANVAS_ACT_TOOL_TABLE_0 + bi;
                if (canvas_dispatch_root_action(c, action_id)) return 1;
            }
        }
        // IO controls to the left of table buttons: compute positions matching raster
        int io_base_x = bx_r;
        int nbw = bw;
        int num_w = std::max(24, nbw * 2);
        int gap = 10;
        int counter_total_w = nbw + gap + num_w + gap + nbw;
        int action_gap = 16;
        int pair_gap = 10;
        int pair_total_w = nbw * 2 + pair_gap;
        int order_num_w = std::max(28, nbw * 2);
        int order_total_w = nbw + gap + order_num_w + gap + nbw + gap + nbw;
        int order_left_x = io_base_x - counter_total_w - action_gap - pair_total_w - action_gap - order_total_w;
        int pair_left_x = io_base_x - counter_total_w - action_gap - pair_total_w;
        int bx_consumer = pair_left_x;
        int bx_producer = bx_consumer + nbw + pair_gap;
        if (view_y >= by && view_y < by + bh) {
            if (view_x >= bx_consumer && view_x < bx_consumer + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_IO_CONSUMER_ADD)) return 1;
            }
            if (view_x >= bx_producer && view_x < bx_producer + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_IO_PRODUCER_ADD)) return 1;
            }
        }
        if (view_y >= by && view_y < by + bh) {
            int bx_down = order_left_x;
            int bx_num = bx_down + nbw + gap;
            int bx_up = bx_num + order_num_w + gap;
            int bx_tool = bx_up + nbw + gap;
            if (view_x >= bx_down && view_x < bx_down + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_EDGE_ORDER_DEC)) return 1;
            }
            if (view_x >= bx_up && view_x < bx_up + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_EDGE_ORDER_INC)) return 1;
            }
            if (view_x >= bx_tool && view_x < bx_tool + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_EDGE_ORDER_TOOL)) return 1;
            }
        }
        // counter group positions
        int bx_minus = io_base_x - (nbw + gap + num_w + gap + nbw);
        int bx_num = bx_minus + nbw + gap;
        int bx_plus = bx_num + num_w + gap;
        if (view_y >= by && view_y < by + bh) {
            if (view_x >= bx_minus && view_x < bx_minus + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_IO_COUNT_DEC)) return 1;
            }
            if (view_x >= bx_plus && view_x < bx_plus + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_IO_COUNT_INC)) return 1;
            }
        }
    }
    if (c->tool_menu_open) {
        ToolMenuLayout layout = compute_tool_menu_layout(c);
        if (view_x >= layout.x && view_x < layout.x + layout.w && view_y >= layout.y && view_y < layout.y + layout.h) {
            int item_y = layout.item_start_y;
            for (size_t i = 0; i < std::size(kToolMenuItems); ++i) {
                int y0 = item_y + static_cast<int>(i) * layout.row_h;
                if (view_y >= y0 && view_y < y0 + layout.row_h) {
                    if (canvas_dispatch_root_action(c, kToolMenuItems[i].action_id)) return 1;
                }
            }
            return 1;
        }
        c->tool_menu_open = false;
        c->selected_tool_table = 0;
    }
    for (int mi = 0; mi < static_cast<int>(c->modules.size()); ++mi) {
        const auto &m = c->modules[mi];
        // If module has an attached table, ask the table for hit information
        // for clicks inside the module rect. If the table reports a LED hit,
        // map that hit into the canvas contact selection/rope creation flow
        // so clicks target the actual LED cells rendered by the table.
        if (mi < static_cast<int>(c->module_tables.size()) && c->module_tables[mi]) {
            if (world_x >= m.x && world_x < m.x + m.w && world_y >= m.y && world_y < m.y + m.h) {
                // Query the attached table for hitboxes without invoking
                // `gp_table_on_click` (which mutates table selection). This
                // lets the canvas resolve LED hits for edge-mode without
                // competing with the table's own selection logic.
                GP_TableContext* t = c->module_tables[mi];
                int lx = world_x - m.x;
                int ly = world_y - m.y;
                int tw = std::max(1, m.w);
                int th = std::max(1, m.h);
                std::vector<uint8_t> tmp(static_cast<size_t>(tw) * static_cast<size_t>(th) * 4);
                GP_TableGeom geom{};
                gp_table_get_geom(t, &geom);
                geom.width_px = tw; geom.height_px = th;
                const int hitcap = 1024;
                std::vector<GP_TableHitBox> hits(hitcap);
                int hits_written = 0;
                int ok = gp_table_render_rgba_with_state(t, nullptr, tmp.data(), static_cast<int32_t>(tmp.size()), &geom, hits.data(), hitcap, &hits_written);
                printf("gp_canvas_on_click: module=%d has_table=%d geom=%d,%d render_ok=%d hits_cap=%d hits_written=%d\n", mi, (t!=nullptr)?1:0, tw, th, ok, hitcap, hits_written);
                printf("gp_canvas_on_click: local point = %d,%d (module local lx,ly)\n", lx, ly);
                for (int hi = 0; hi < hits_written; ++hi) {
                    const auto &hbi = hits[hi];
                    printf("  hit[%d]=part=%d row=%d col=%d aux0=%d aux1=%d rect=%d,%d-%d,%d flags=0x%x\n", hi, hbi.part, hbi.row_idx, hbi.col_idx, hbi.aux0, hbi.aux1, hbi.x0, hbi.y0, hbi.x1, hbi.y1, hbi.flags);
                }
                if (ok && hits_written > 0) {
                    // find first hit containing local point
                    GP_TableHitBox found{}; bool found_any = false;
                    for (int hi = 0; hi < hits_written; ++hi) {
                        const GP_TableHitBox &hb = hits[hi];
                        if (lx >= hb.x0 && lx < hb.x1 && ly >= hb.y0 && ly < hb.y1) { found = hb; found_any = true; break; }
                    }
                    if (!found_any) {
                        printf("gp_canvas_on_click: module=%d table_hits_present=%d but none contain (%d,%d) local\n", mi, hits_written, lx, ly);
                        for (int hi = 0; hi < hits_written; ++hi) {
                            const GP_TableHitBox &hb = hits[hi];
                            printf("  hit[%d]=part=%d row=%d col=%d aux0=%d aux1=%d rect=%d,%d-%d,%d flags=0x%x\n", hi, hb.part, hb.row_idx, hb.col_idx, hb.aux0, hb.aux1, hb.x0, hb.y0, hb.x1, hb.y1, hb.flags);
                        }
                    }
                    if (found_any) {
                        c->dispatch_module_idx = mi;
                        int handled = canvas_dispatch_root_hit(c, found);
                        c->dispatch_module_idx = -1;
                        if (handled) return 1;
                        // If LED hit, handle canvas-level connection flow. In
                        // edge-drawing mode we avoid calling into the table so
                        // we don't toggle its internal selection state.
                        if (found.part == GP_TABLE_HIT_LED || found.part == GP_TABLE_HIT_LED_ARG || found.part == GP_TABLE_HIT_LED_TABLE) {
                            if (canvas_handle_module_led_hit(c, mi, found)) return 1;
                        } else {
                            // Non-LED hit: let table handle the click unless we're
                            // in canvas-level edge-only mode (tool index 2).
                            if (c->selected_tool_canvas != 2) {
                                // forward to table so it can perform its default actions
                                GP_TableHitBox out_hit{};
                                gp_table_on_click(t, lx, ly, &out_hit);
                                return 1;
                            }
                            // if edge-only mode, consume the hit but do not mutate table
                            return 1;
                        }
                    }
                }
            }
        }
    }
    if (c->edge_order_tool_active) {
        int edge_idx = canvas_pick_edge_by_rope(c, world_x, world_y, static_cast<float>(pick_r));
        if (edge_idx >= 0 && edge_idx < static_cast<int>(c->edges.size())) {
            canvas_set_order_mode_for_edge(c, c->edges[static_cast<size_t>(edge_idx)].desc, c->edge_order_value);
            return 1;
        }
    }
    if (c->selected_tool_edge >= 0 && c->selected_tool_edge != 0) {
        int edge_idx = canvas_pick_edge_by_rope(c, world_x, world_y, static_cast<float>(pick_r));
        if (edge_idx >= 0 && edge_idx < static_cast<int>(c->edges.size())) {
            if (c->selected_tool_edge == 1) {
                canvas_remove_edge_at(c, edge_idx);
            } else {
                bool delta_mode = (c->selected_tool_edge == 2);
                canvas_set_delta_mode_for_edge(c, c->edges[static_cast<size_t>(edge_idx)].desc, delta_mode);
            }
            return 1;
        }
    }
    // click not on any contact: clear selection
    if (c->selected.module != -1) printf("gp_canvas_on_click: clearing selection module=%d contact=%d\n", c->selected.module, c->selected.contact_idx);
    c->selected.module = -1; c->selected.contact_idx = -1; c->selected.left = -1; c->selected.anchor_x = -1; c->selected.anchor_y = -1;
    // discard any provisional rope
    if (c->prospective_rope_idx >= 0) {
        printf("gp_canvas_on_click: discarding prospective rope %d\n", c->prospective_rope_idx);
        c->prospective_rope_idx = -1;
    }

    // If a tool is active and the click is in empty space (not on any module),
    // spawn the tool's module.
    bool hit_module = false;
    for (int mi = 0; mi < static_cast<int>(c->modules.size()); ++mi) {
        const auto &m = c->modules[mi];
        if (world_x >= m.x && world_x < m.x + m.w && world_y >= m.y && world_y < m.y + m.h) { hit_module = true; break; }
    }
    if (!hit_module && c->selected_tool_canvas == 1) {
        GP_CanvasModuleDesc d{};
        int nx = world_x - 20;
        int ny = world_y - 16;
        d.x = nx; d.y = ny; d.w = 320; d.h = 240;
        std::memset(d.label, 0, sizeof(d.label));
        std::memcpy(d.label, LABEL_MODULE_TABLE, std::min<size_t>(std::strlen(LABEL_MODULE_TABLE), sizeof(d.label) - 1));
        int new_idx = gp_canvas_add_module(ctx_, &d);
        if (new_idx >= 0) {
            // Tool==1 => create a table at click position
            gp_canvas_create_table(ctx_, new_idx);
            printf("gp_canvas_on_click: spawned new table at %d,%d module=%d\n", nx, ny, new_idx);
            // focus the newly created module
            c->focused_module = new_idx;
        }
        // leave tool selected - user can toggle off with button
        return 1;
    }
    if (!hit_module && c->selected_tool_canvas == 3) {
        GP_CanvasModuleDesc d{};
        int nx = world_x - 20;
        int ny = world_y - 16;
        d.x = nx; d.y = ny; d.w = 320; d.h = 240;
        std::memset(d.label, 0, sizeof(d.label));
        std::memcpy(d.label, LABEL_MODULE_STAGE, std::min<size_t>(std::strlen(LABEL_MODULE_STAGE), sizeof(d.label) - 1));
        int new_idx = gp_canvas_add_module(ctx_, &d);
        if (new_idx >= 0) {
            // Mark as stage module and set up stage + frame table.
            if (new_idx < static_cast<int>(c->module_is_stage.size())) c->module_is_stage[new_idx] = 1;
            if (new_idx < static_cast<int>(c->module_stages.size())) {
                GP_StageContext* st = gp_stage_create(d.w, d.h, 2);
                c->module_stages[new_idx] = st;
                if (new_idx < static_cast<int>(c->module_stage_owned.size())) c->module_stage_owned[new_idx] = 1;
                canvas_setup_stage_defaults(st, d.w, d.h);
            }
            // Stage modules always expose one input and one output port.
            if (new_idx < static_cast<int>(c->module_io_in_count.size())) c->module_io_in_count[new_idx] = 1;
            if (new_idx < static_cast<int>(c->module_io_out_count.size())) c->module_io_out_count[new_idx] = 1;
            // Use module background callback for stage raster.
            if (new_idx < static_cast<int>(c->module_bg.size())) {
                c->module_bg[new_idx].cb = stage_bg_callback;
                c->module_bg[new_idx].user = c;
                // allow stage imagery to remain visible behind the table
                c->module_bg[new_idx].table_alpha = 192;
                c->module_bg[new_idx].table_alpha_ray = 192;
                c->module_bg[new_idx].header_margin_px = 0;   // content rendered inside table cells; no header gap needed
            }
            // Reconfigure the auto-created table as a minimal header strip.
            if (new_idx < static_cast<int>(c->module_tables.size()) && c->module_tables[new_idx]) {
                canvas_setup_stage_table(c->module_tables[new_idx], d.w);
                // Install stage IO LEDs immediately.
                sync_module_table_io_layout(c, new_idx);
            }
            printf("gp_canvas_on_click: spawned new stage at %d,%d module=%d\n", nx, ny, new_idx);
            c->focused_module = new_idx;
        }
        return 1;
    }

    // forward click to any attached table that contains the point
    for (int mi = 0; mi < static_cast<int>(c->modules.size()); ++mi) {
        const auto &m = c->modules[mi];
        if (world_x >= m.x && world_x < m.x + m.w && world_y >= m.y && world_y < m.y + m.h) {
            // focus this module when clicked
            c->focused_module = mi;
            // debug: report focus and current table-tool selection before any action
            printf("gp_canvas_on_click: focus set -> module=%d selected_tool_table=%d selected_tool_canvas=%d\n", mi, c->selected_tool_table, c->selected_tool_canvas);
            if (mi < static_cast<int>(c->module_tables.size()) && c->module_tables[mi] && c->selected_tool_canvas == 0) {
                // local coords; only forward clicks into attached tables when
                // canvas is in select/interaction mode (tool 0). In edge-mode
                // we performed non-mutating hit tests earlier and should avoid
                // letting the table change its own selection state here.
                int lx = world_x - m.x;
                int ly = world_y - m.y;
                GP_TableHitBox hb{};
                int ok = gp_table_on_click(c->module_tables[mi], lx, ly, &hb);
                if (ok) return 1;
            }
            break;
        }
    }
    return 0;
}

extern "C" int gp_canvas_on_mouse_down(GP_CanvasContext* ctx_, int x, int y) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    canvas_record_mouse_input(c, x, y, /*down=*/true, /*up=*/false);
    if (gp_canvas_on_click(ctx_, x, y)) return 1;
    int world_x = x + c->offset_x;
    int world_y = y + c->offset_y;
    // otherwise check for module hit to start dragging
    for (int mi = static_cast<int>(c->modules.size()) - 1; mi >= 0; --mi) {
        const auto &m = c->modules[mi];
        // Only start a drag if the mouse is within a small header area at the
        // top of the module. This prevents clicks on embedded table content or
        // contacts from immediately initiating a window move.
        int header_h = std::min(24, std::max(8, m.h / 6));
        if (world_x >= m.x && world_x < m.x + m.w && world_y >= m.y && world_y < m.y + header_h) {
            // start drag: record in per-canvas DragState
            c->drag.dragging = 1;
            c->drag.module = mi;
            c->drag.offx = world_x - m.x;
            c->drag.offy = world_y - m.y;
            // focus the module being dragged
            c->focused_module = mi;
            printf("gp_canvas_on_mouse_down: start drag canvas=%p module=%d off=%d,%d\n", (void*)c, mi, c->drag.offx, c->drag.offy);
            return 1;
        }
    }
    // In edge-tool (rightmost canvas tool) allow background drag to pan viewport.
    if (c->selected_tool_canvas == 2) {
        bool hit_module = false;
        for (const auto& m : c->modules) {
            if (world_x >= m.x && world_x < m.x + m.w && world_y >= m.y && world_y < m.y + m.h) { hit_module = true; break; }
        }
        if (!hit_module) {
            c->drag.dragging = 1;
            c->drag.panning = 1;
            c->drag.module = -1;
            c->drag.pan_last_x = x;
            c->drag.pan_last_y = y;
            printf("gp_canvas_on_mouse_down: start pan canvas=%p at view=%d,%d world=%d,%d\n", (void*)c, x, y, world_x, world_y);
            return 1;
        }
    }
    return 0;
}

extern "C" int gp_canvas_on_mouse_move(GP_CanvasContext* ctx_, int x, int y) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    canvas_record_mouse_input(c, x, y, /*down=*/false, /*up=*/false);
    if (!c->drag.panning) {
        update_canvas_scroll_state(c, /*pull_from_container=*/true);
    }
    bool handled = false;
    int world_x = x + c->offset_x;
    int world_y = y + c->offset_y;
    // update provisional rope endpoint to follow mouse — only if anchor known
    if (c->prospective_rope_idx >= 0 && c->selected.module >= 0) {
        if (c->selected.anchor_x >= 0 && c->selected.anchor_y >= 0) {
            int fx = c->selected.anchor_x;
            int fy = c->selected.anchor_y;
            RopeSim* sim = canvas_root_sim(c);
            if (sim) {
                rope_sim_move_endpoints(sim, c->prospective_rope_idx, static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(world_x), static_cast<float>(world_y));
            }
            handled = true;
        }
    }
    // handle viewport pan
    if (c->drag.dragging && c->drag.panning) {
        int dx = x - c->drag.pan_last_x;
        int dy = y - c->drag.pan_last_y;
        c->offset_x -= dx;
        c->offset_y -= dy;
        c->drag.pan_last_x = x;
        c->drag.pan_last_y = y;
        update_canvas_scroll_state(c, /*pull_from_container=*/false);
        handled = true;
    }
    // handle module drag if present (per-canvas drag state)
    DragState ds = c->drag;
    if (ds.dragging && ds.module >= 0) {
        int nx = world_x - ds.offx;
        int ny = world_y - ds.offy;
        c->modules[ds.module].x = nx;
        c->modules[ds.module].y = ny;
        printf("gp_canvas_on_mouse_move: canvas=%p module=%d -> %d,%d\n", (void*)c, ds.module, nx, ny);
        handled = true;
    }
    if (handled) {
        update_canvas_scroll_state(c, /*pull_from_container=*/false);
    }
    return handled ? 1 : 0;
}

extern "C" int gp_canvas_on_mouse_up(GP_CanvasContext* ctx_, int x, int y) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    canvas_record_mouse_input(c, x, y, /*down=*/false, /*up=*/true);
    if (!c->drag.dragging) return 0;
    c->drag.dragging = 0;
    c->drag.panning = 0;
    printf("gp_canvas_on_mouse_up: canvas=%p module=%d\n", (void*)c, c->drag.module);
    c->drag.module = -1;
    update_canvas_scroll_state(c, /*pull_from_container=*/false);
    return 1;
}

extern "C" int gp_canvas_on_key(GP_CanvasContext* ctx_, int key, int scancode, int action, int mods) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    printf("gp_canvas_on_key: canvas=%p key=%d scancode=%d action=%d mods=%d focused=%d\n", (void*)c, key, scancode, action, mods, c->focused_module);
    canvas_record_key_input(c, key, action);
    int mi = c->focused_module;
    if (mi >= 0 && mi < static_cast<int>(c->module_tables.size())) {
        GP_TableContext* t = c->module_tables[mi];
        if (t) {
            return gp_table_on_key(t, key, scancode, action, mods);
        }
    }
    return 0;
}

extern "C" int gp_canvas_set_offset(GP_CanvasContext* ctx_, int offx, int offy) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->offset_x = offx;
    c->offset_y = offy;
    update_canvas_scroll_state(c, /*pull_from_container=*/false);
    return 1;
}

extern "C" int gp_canvas_get_offset(GP_CanvasContext* ctx_, int* out_offx, int* out_offy) {
    if (!ctx_ || !out_offx || !out_offy) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    *out_offx = c->offset_x;
    *out_offy = c->offset_y;
    return 1;
}

extern "C" int gp_canvas_get_scroll_flags(GP_CanvasContext* ctx_, int* out_has_h, int* out_has_v) {
    if (!ctx_ || !out_has_h || !out_has_v) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    update_canvas_scroll_state(c, /*pull_from_container=*/false);
    *out_has_h = c->scroll_x_needed;
    *out_has_v = c->scroll_y_needed;
    return 1;
}

// create/destroy canvas-owned table helpers
extern "C" int gp_canvas_create_table(GP_CanvasContext* ctx_, int module_idx) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->modules.size())) return 0;
    if (c->module_tables[module_idx]) return 0; // already has a table
    GP_TableContext* t = gp_table_create(nullptr);
    if (!t) return 0;
    c->module_tables[module_idx] = t;
    c->module_table_owned[module_idx] = 1;
    // Tweak the table style for canvas-owned tables so row height is
    // compact and rows aren't vertically stretched to fill module height.
    // This helps keep LED hitboxes aligned with visual rows when modules
    // are taller than the table content.
    if (module_idx >= 0 && module_idx < static_cast<int>(c->modules.size())) {
        const auto &mod = c->modules[module_idx];
        GP_TableStyle st{};
        st.width_px = std::max(1, mod.w);
        st.row_h_px = 18; // slightly tighter than default 20
        st.indent_px = 14;
        st.expand_w_px = 12;
        st.name_w_px = 160;
        // Provide explicit colors to avoid zero/transparent defaults which
        // would produce fully transparent output when rendered into a
        // module buffer. These match the renderer's charcoal defaults.
        st.bg_rgba[0] = 40; st.bg_rgba[1] = 40; st.bg_rgba[2] = 50; st.bg_rgba[3] = 255;
        st.bg_sel_rgba[0] = 18; st.bg_sel_rgba[1] = 18; st.bg_sel_rgba[2] = 26; st.bg_sel_rgba[3] = 255;
        st.hdr_rgba[0] = 20; st.hdr_rgba[1] = 20; st.hdr_rgba[2] = 28; st.hdr_rgba[3] = 255;
        st.text_rgba[0] = 240; st.text_rgba[1] = 240; st.text_rgba[2] = 245; st.text_rgba[3] = 255;
        st.text_hdr_rgba[0] = 255; st.text_hdr_rgba[1] = 255; st.text_hdr_rgba[2] = 255; st.text_hdr_rgba[3] = 255;
        st.led_on_rgba[0] = 255; st.led_on_rgba[1] = 210; st.led_on_rgba[2] = 90; st.led_on_rgba[3] = 255;
        st.led_off_rgba[0] = 70; st.led_off_rgba[1] = 70; st.led_off_rgba[2] = 80; st.led_off_rgba[3] = 255;
        st.led_edge_rgba[0] = 255; st.led_edge_rgba[1] = 255; st.led_edge_rgba[2] = 255; st.led_edge_rgba[3] = 255;
        st.axis_bg_rgba[0] = 18; st.axis_bg_rgba[1] = 18; st.axis_bg_rgba[2] = 22; st.axis_bg_rgba[3] = 255;
        st.axis_tick_rgba[0] = 200; st.axis_tick_rgba[1] = 200; st.axis_tick_rgba[2] = 200; st.axis_tick_rgba[3] = 255;
        st.axis_val_rgba[0] = 255; st.axis_val_rgba[1] = 255; st.axis_val_rgba[2] = 140; st.axis_val_rgba[3] = 255;
        st.timer_rgba[0] = 110; st.timer_rgba[1] = 180; st.timer_rgba[2] = 255; st.timer_rgba[3] = 255;
        st.wave_bg_rgba[0] = 6; st.wave_bg_rgba[1] = 6; st.wave_bg_rgba[2] = 8; st.wave_bg_rgba[3] = 255;
        st.wave_fg_rgba[0] = 255; st.wave_fg_rgba[1] = 255; st.wave_fg_rgba[2] = 255; st.wave_fg_rgba[3] = 255;
        gp_table_set_style(t, &st);
    }
    // ensure module has a backing node in graph
    if (module_idx >= static_cast<int>(c->module_node_id.size())) c->module_node_id.resize(module_idx + 1, -1);
    if (c->module_node_id[module_idx] < 0) {
        int nid = c->next_node_id++;
        c->module_node_id[module_idx] = nid;
        GP_CanvasContextImpl::NodeContract nc; nc.node_id = nid; nc.module_idx = module_idx;
        c->nodes.push_back(std::move(nc));
    }
    // attach table to canvas sim
    RopeSim* sim = canvas_require_root_sim(c);
    if (sim) gp_table_attach_rope_sim(t, sim, 0);
    // enable prospective/live mode by default for embedded tables
    gp_table_set_prospective_mode(t, 1);
    gp_table_prospective_set_params(t, 8, 4.0f, 0.0f);
    // If the table has no IO sections and no existing columns, create empty
    // input/output edge sections on the table according to its side-reading
    // direction so canvases render empty input/output strips at the edges.
    // Only do this for truly blank tables to avoid clobbering caller-initialized tables.
    {
        GP_TableGeom gtmp{};
        int col_count_existing = 0;
        if (gp_table_get_geom(t, &gtmp)) {
            for (int ii = 0; ii < 8; ++ii) if (gtmp.col_w[ii] > 0) ++col_count_existing;
        }
        if (!gp_table_has_io_sections(t) && col_count_existing == 0) {
            int in_dir = 0, out_dir = 0;
            gp_table_get_side_reading_direction(t, 0, &in_dir);
            gp_table_get_side_reading_direction(t, 1, &out_dir);
            int dir = in_dir; // prefer input side direction as primary
            if (dir == 0 || dir == 2) {
                // horizontal layout: ensure an input column at left and output at right
                GP_TableColumn cols[2];
                cols[0].kind = GP_TABLE_CELL_LEDS;
                cols[0].width_px = 80;
                cols[0].align = 0;
                cols[1].kind = GP_TABLE_CELL_LEDS;
                cols[1].width_px = 80;
                cols[1].align = 0;
                if (dir == 2) { // RightToLeft -> swap so inputs appear on the right edge
                    GP_TableColumn tmp = cols[0]; cols[0] = cols[1]; cols[1] = tmp;
                }
                gp_table_set_columns(t, cols, 2);
                // leave rows empty for now
                gp_table_set_rows(t, nullptr, 0);
            } else {
                // vertical layout: create a single column and empty top/bottom rows
                GP_TableColumn col;
                col.kind = GP_TABLE_CELL_LEDS;
                col.width_px = 160;
                col.align = 0;
                gp_table_set_columns(t, &col, 1);
                GP_TableRow rows[2];
                memset(rows, 0, sizeof(rows));
                rows[0].kind = GP_TABLE_ROW_HEADER;
                rows[0].depth = 0;
                rows[0].expanded = 1;
                rows[0].selected = 0;
                rows[0].cell_count = 1;
                rows[0].cells[0].kind = GP_TABLE_CELL_LEDS;
                rows[1].kind = GP_TABLE_ROW_NOTE;
                rows[1].depth = 0;
                rows[1].expanded = 1;
                rows[1].selected = 0;
                rows[1].cell_count = 1;
                rows[1].cells[0].kind = GP_TABLE_CELL_LEDS;
                if (dir == 3) {
                    // BottomToTop: place inputs at bottom (swap)
                    GP_TableRow tmp = rows[0]; rows[0] = rows[1]; rows[1] = tmp;
                }
                gp_table_set_rows(t, rows, 2);
            }
        }
    }
    return 1;
}

extern "C" int gp_canvas_register_window(GP_CanvasContext* ctx_, void* window_ptr) {
    if (!ctx_ || !window_ptr) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // if already registered, noop
    if (c->window_node_ids.find(window_ptr) != c->window_node_ids.end()) {
        return 1;
    }
    int nid = c->next_node_id++;
    c->window_node_ids[window_ptr] = nid;
    c->windows.push_back(window_ptr);
    return 1;
}

extern "C" int gp_canvas_unregister_window(GP_CanvasContext* ctx_, void* window_ptr) {
    if (!ctx_ || !window_ptr) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    for (auto it = c->windows.begin(); it != c->windows.end(); ++it) {
        if (*it == window_ptr) { c->windows.erase(it); break; }
    }
    auto mit = c->window_node_ids.find(window_ptr);
    if (mit != c->window_node_ids.end()) c->window_node_ids.erase(mit);
    return 1;
}

extern "C" int gp_canvas_get_window_node_id(GP_CanvasContext* ctx_, void* window_ptr) {
    if (!ctx_ || !window_ptr) return -1;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    auto it = c->window_node_ids.find(window_ptr);
    if (it == c->window_node_ids.end()) return -1;
    return it->second;
}

extern "C" int gp_canvas_destroy_table(GP_CanvasContext* ctx_, int module_idx) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->modules.size())) return 0;
    GP_TableContext* t = c->module_tables[module_idx];
    int owned = c->module_table_owned[module_idx];
    c->module_tables[module_idx] = nullptr;
    c->module_table_owned[module_idx] = 0;
    if (t && owned) gp_table_destroy(t);
    // free any module key-recorder state
    if (module_idx >= 0 && module_idx < static_cast<int>(c->module_key_recorder_state.size())) {
        void* s = c->module_key_recorder_state[module_idx];
        if (s) {
            delete reinterpret_cast<KeyRecorderState*>(s);
            c->module_key_recorder_state[module_idx] = nullptr;
        }
    }
    // detach node mapping for this module
    if (module_idx >= 0 && module_idx < static_cast<int>(c->module_node_id.size())) {
        int nid = c->module_node_id[module_idx];
        c->module_node_id[module_idx] = -1;
        for (auto &n : c->nodes) {
            if (n.node_id == nid) { n.module_idx = -1; break; }
        }
    }
    return 1;
}

extern "C" int gp_canvas_step(GP_CanvasContext* ctx_, float dt) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    RopeSim* sim = canvas_root_sim(c);
    if (sim) {
        rope_sim_step(sim, dt, c->sim_maxforce, c->sim_iters, c->sim_damping);
    }
    // decay chat highlight TTLs and clear chat bg callback when expired
    for (size_t mi = 0; mi < c->modules.size(); ++mi) {
        if (mi < c->module_chat_ttl.size() && c->module_chat_ttl[mi] > 0) {
            c->module_chat_ttl[mi] -= 1;
            if (c->module_chat_ttl[mi] <= 0) {
                if (mi < static_cast<size_t>(c->module_bg.size())) {
                    if (c->module_bg[mi].cb == chat_bg_callback) {
                        c->module_bg[mi].cb = nullptr;
                        c->module_bg[mi].user = nullptr;
                        c->module_bg[mi].table_alpha = 255;
                    }
                }
            }
        }
    }
    if (c->thread_mgr) {
        ThreadManager::TickRequest req;
        req.dt = static_cast<double>(dt);
        req.root_table = canvas_ensure_root_table(c);
        req.modules.reserve(c->modules.size());
        for (int mi = 0; mi < static_cast<int>(c->modules.size()); ++mi) {
            ThreadManager::ModuleContract mod{};
            mod.module_idx = mi;
            mod.table = (mi >= 0 && mi < static_cast<int>(c->module_tables.size())) ? c->module_tables[mi] : nullptr;
            mod.in_count = (mi >= 0 && mi < static_cast<int>(c->module_io_in_count.size())) ? c->module_io_in_count[mi] : 0;
            mod.out_count = (mi >= 0 && mi < static_cast<int>(c->module_io_out_count.size())) ? c->module_io_out_count[mi] : 0;
            req.modules.push_back(mod);
        }
        req.edges.reserve(c->edges.size());
        for (size_t ei = 0; ei < c->edges.size(); ++ei) {
            ThreadManager::EdgeContract e{};
            e.edge_idx = static_cast<int32_t>(ei);
            e.type_id = c->edges[ei].type_id;
            e.a_module = c->edges[ei].desc.a_module;
            e.a_contact_idx = c->edges[ei].desc.a_contact_idx;
            e.b_module = c->edges[ei].desc.b_module;
            e.b_contact_idx = c->edges[ei].desc.b_contact_idx;
            req.edges.push_back(e);
        }
        // Populate stage tasks so ThreadManager can run stage work before table steps.
        req.stages.reserve(c->module_stages.size());
        for (int mi = 0; mi < static_cast<int>(c->module_stages.size()); ++mi) {
            GP_StageContext* st = c->module_stages[mi];
            if (!st) continue;
            ThreadManager::StageContract sc{};
            sc.module_idx = mi;
            sc.stage = reinterpret_cast<void*>(st);
            sc.table = (mi >= 0 && mi < static_cast<int>(c->module_tables.size())) ? c->module_tables[mi] : nullptr;
            // Ensure cache buffer is allocated to the stage size; UI rendering will read this.
            int w = std::max(1, c->modules[mi].w);
            int h = std::max(1, c->modules[mi].h);
            int pitch = w * 4;
            if (mi >= static_cast<int>(c->module_stage_cache_rgba.size())) {
                // should not happen, but guard
                c->module_stage_cache_rgba.resize(mi + 1);
                c->module_stage_cache_w.resize(mi + 1);
                c->module_stage_cache_h.resize(mi + 1);
                c->module_stage_cache_pitch.resize(mi + 1);
                c->module_stage_cache_mu.emplace_back(std::make_unique<std::mutex>());
            }
            {
                std::lock_guard<std::mutex> lk(*c->module_stage_cache_mu[mi]);
                if (c->module_stage_cache_w[mi] != w || c->module_stage_cache_h[mi] != h || c->module_stage_cache_pitch[mi] != pitch) {
                    c->module_stage_cache_w[mi] = w;
                    c->module_stage_cache_h[mi] = h;
                    c->module_stage_cache_pitch[mi] = pitch;
                    c->module_stage_cache_rgba[mi].assign(static_cast<size_t>(pitch) * static_cast<size_t>(h), 0u);
                }
            }
            sc.out_rgba = c->module_stage_cache_rgba[mi].empty() ? nullptr : c->module_stage_cache_rgba[mi].data();
            sc.out_pitch = c->module_stage_cache_pitch[mi];
            sc.width = c->module_stage_cache_w[mi];
            sc.height = c->module_stage_cache_h[mi];
            sc.cache_mu = c->module_stage_cache_mu[mi].get();
            req.stages.push_back(sc);
        }
        // Submit asynchronously so the UI thread is not blocked by stage/table work.
        c->thread_mgr->submit_tick(std::move(req), /*wait=*/false);
    }
    // autosave: accumulate dt and write canvas file when interval reached
    if (!c->autosave_path.empty() && c->autosave_interval_s > 0.0) {
        c->autosave_accum_s += static_cast<double>(dt);
        if (c->autosave_accum_s >= c->autosave_interval_s) {
            // attempt save; ignore failures
            gp_canvas_save_to_file(ctx_, c->autosave_path.c_str());
            c->autosave_accum_s = 0.0;
        }
    }
    return 1;
}

extern "C" void* gp_canvas_get_rope_sim(GP_CanvasContext* ctx_) {
    if (!ctx_) return nullptr;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    return reinterpret_cast<void*>(canvas_root_sim(c));
}

static void canvas_setup_stage_table(GP_TableContext* t, int w_px) {
    if (!t) return;
    GP_TableStyle st{};
    // Keep the table narrow/compact; it acts mostly as a header strip + focusable frame.
    st.width_px = std::max(1, w_px);
    st.row_h_px = 18;
    st.indent_px = 14;
    st.expand_w_px = 12;
    st.name_w_px = std::max(60, std::min(200, w_px / 2));
    st.bg_rgba[0] = 0; st.bg_rgba[1] = 0; st.bg_rgba[2] = 0; st.bg_rgba[3] = 0; // leave empty area transparent
    st.bg_sel_rgba[0] = 0; st.bg_sel_rgba[1] = 0; st.bg_sel_rgba[2] = 0; st.bg_sel_rgba[3] = 0;
    st.hdr_rgba[0] = 20; st.hdr_rgba[1] = 20; st.hdr_rgba[2] = 28; st.hdr_rgba[3] = 255;
    st.text_rgba[0] = 240; st.text_rgba[1] = 240; st.text_rgba[2] = 245; st.text_rgba[3] = 255;
    st.text_hdr_rgba[0] = 255; st.text_hdr_rgba[1] = 255; st.text_hdr_rgba[2] = 255; st.text_hdr_rgba[3] = 255;
    st.led_on_rgba[0] = 255; st.led_on_rgba[1] = 210; st.led_on_rgba[2] = 90; st.led_on_rgba[3] = 255;
    st.led_off_rgba[0] = 70; st.led_off_rgba[1] = 70; st.led_off_rgba[2] = 80; st.led_off_rgba[3] = 255;
    st.led_edge_rgba[0] = 255; st.led_edge_rgba[1] = 255; st.led_edge_rgba[2] = 255; st.led_edge_rgba[3] = 255;
    st.axis_bg_rgba[0] = 18; st.axis_bg_rgba[1] = 18; st.axis_bg_rgba[2] = 22; st.axis_bg_rgba[3] = 255;
    st.axis_tick_rgba[0] = 200; st.axis_tick_rgba[1] = 200; st.axis_tick_rgba[2] = 200; st.axis_tick_rgba[3] = 255;
    st.axis_val_rgba[0] = 255; st.axis_val_rgba[1] = 255; st.axis_val_rgba[2] = 140; st.axis_val_rgba[3] = 255;
    st.timer_rgba[0] = 110; st.timer_rgba[1] = 180; st.timer_rgba[2] = 255; st.timer_rgba[3] = 255;
    st.wave_bg_rgba[0] = 6; st.wave_bg_rgba[1] = 6; st.wave_bg_rgba[2] = 8; st.wave_bg_rgba[3] = 255;
    st.wave_fg_rgba[0] = 255; st.wave_fg_rgba[1] = 255; st.wave_fg_rgba[2] = 255; st.wave_fg_rgba[3] = 255;
    gp_table_set_style(t, &st);

    GP_TableColumn col{};
    col.kind = GP_TABLE_CELL_TEXT;
    col.width_px = st.width_px;
    col.align = 0;
    gp_table_set_columns(t, &col, 1);

    GP_TableRow row{};
    std::memset(&row, 0, sizeof(row));
    row.kind = GP_TABLE_ROW_HEADER;
    row.depth = 0;
    row.expanded = 1;
    row.selected = 0;
    row.cell_count = 1;
    row.cells[0].kind = GP_TABLE_CELL_TEXT;
    gp_table_set_rows(t, &row, 1);
}

static void canvas_setup_stage_defaults(GP_StageContext* st, int w_px, int h_px) {
    if (!st) return;
    gp_stage_resize(st, w_px, h_px);
    gp_stage_set_sampling(st, /*oversample=*/2, /*downsample=*/2);
    gp_stage_set_temporal(st, /*decay=*/0.93f, /*max_intensity=*/1.0f);
    gp_stage_set_ray_params(st, /*rays=*/700, /*max_reflections=*/4, /*blur_sigma=*/2.0f);
    gp_stage_set_ray_attenuation(st, /*bounce_decay=*/0.78f, /*distance_decay=*/0.0025f);
    gp_stage_set_exposure(st, 1.0f);
    gp_stage_set_depth(st, std::min<float>(float(std::min(w_px, h_px)), 140.0f));
    gp_stage_set_field_mode(st, GP_STAGE_FIELD_VECTOR6);

    uint8_t cool[4] = {110, 170, 255, 255};
    uint8_t warm[4] = {255, 180, 90, 255};
    gp_stage_set_layer_tint(st, 0, cool);
    gp_stage_set_layer_tint(st, 1, warm);

    gp_stage_clear_emitters3d(st);
    float depth = std::min<float>(float(std::min(w_px, h_px)), 140.0f);
    GP_StageEmitter3D a{};
    a.x = 0.25f * w_px;
    a.y = 0.35f * h_px;
    a.z = 0.30f * depth;
    a.radius = 6.0f;
    a.intensity = 0.75f;
    a.frequency = 0.015f;
    a.phase0 = 0.0f;
    gp_stage_add_emitter3d(st, &a);

    GP_StageEmitter3D b{};
    b.x = 0.72f * w_px;
    b.y = 0.60f * h_px;
    b.z = 0.75f * depth;
    b.radius = 7.0f;
    b.intensity = 0.65f;
    b.frequency = 0.012f;
    b.phase0 = 1.3f;
    gp_stage_add_emitter3d(st, &b);

    gp_stage_set_emission_mask_sampling(st, /*samples_per_frame=*/12, /*z=*/0.5f * depth, /*radius=*/3.0f, /*intensity=*/0.10f, /*frequency=*/0.02f, /*phase0=*/0.0f);
}

static constexpr unsigned long long kStageInputLedKey = ((unsigned long long)0 << 32) | ((unsigned long long)1 << 16) | 0ull;
static constexpr int kStageIntegratorSampleBudget = 256;
static constexpr uint32_t kStageIntegratorDefaultStride = 18u;
static constexpr float kIntegratorTemporalDecay = 0.995f;
static constexpr float kIntegratorTemporalMax = 4.0f;
static constexpr float kIntegratorExposure = 0.55f;

static std::vector<float>& stage_get_integrator_buffer(GP_CanvasContextImpl* ctx, int module_idx, int width, int height) {
    if (!ctx || module_idx < 0) {
        static std::vector<float> empty_buffer;
        empty_buffer.clear();
        return empty_buffer;
    }
    if (module_idx >= static_cast<int>(ctx->module_stage_integrator_accum.size())) {
        ctx->module_stage_integrator_accum.resize(module_idx + 1);
    }
    int stage_w = std::max(1, width);
    int stage_h = std::max(1, height);
    size_t need = static_cast<size_t>(stage_w) * static_cast<size_t>(stage_h) * 3u;
    auto &buf = ctx->module_stage_integrator_accum[module_idx];
    if (buf.size() != need) {
        buf.assign(need, 0.0f);
    }
    return buf;
}

static uint32_t stage_integrator_stride(GP_TableContext* table, int edge_idx) {
    if (!table || edge_idx < 0) return kStageIntegratorDefaultStride;
    GP_TableEdgeTensorSpec spec{};
    if (!gp_table_edge_get_tensor_spec(table, edge_idx, &spec)) return kStageIntegratorDefaultStride;
    uint64_t stride = 1;
    for (int32_t di = 0; di < spec.dim_count; ++di) {
        stride *= static_cast<uint64_t>(std::max(1, spec.dims[di]));
        if (stride > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return std::numeric_limits<uint32_t>::max();
        }
    }
    if (stride == 0) stride = kStageIntegratorDefaultStride;
    stride = std::max<uint64_t>(stride, kStageIntegratorDefaultStride);
    return static_cast<uint32_t>(std::min<uint64_t>(stride, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
}

static void render_stage_integrator_image(GP_CanvasContextImpl* ctx, int module_idx, GP_TableContext* table, int stage_w, int stage_h, std::vector<uint8_t>& tmp) {
    if (!ctx || module_idx < 0) return;
    auto &accum = stage_get_integrator_buffer(ctx, module_idx, stage_w, stage_h);
    float decay = std::clamp(kIntegratorTemporalDecay, 0.0f, 1.0f);
    if (decay < 1.0f) {
        for (float& v : accum) {
            v *= decay;
        }
    }

    if (table) {
        int edge_idx = -1;
        if (gp_table_edge_index_for_key(table, kStageInputLedKey, &edge_idx) && edge_idx >= 0) {
            uint32_t stride = stage_integrator_stride(table, edge_idx);
            std::vector<float> sample(stride);
            int32_t unread = 0;
            // Ensure integrator is subscribed as a reader; enqueue the subscription
            // so the manager applies it on the table's thread.
            gp_table_enqueue_edge_subscribe_ex(table, edge_idx, kStageInputLedKey, /*start_at_head=*/1);
            gp_table_edge_unread(table, edge_idx, kStageInputLedKey, &unread);
            int to_read = std::min<int>(std::max(0, unread), kStageIntegratorSampleBudget);
            for (int ri = 0; ri < to_read; ++ri) {
                int32_t written = 0;
                if (!gp_table_edge_consume(table, edge_idx, kStageInputLedKey, sample.data(), static_cast<int>(stride), &written) || written <= 0) {
                    break;
                }
                size_t plen = static_cast<size_t>(written);
                if (plen < 17) continue;
                float px = sample[0];
                float py = sample[1];
                int ix = static_cast<int>(std::lround(px));
                int iy = static_cast<int>(std::lround(py));
                ix = std::clamp(ix, 0, stage_w - 1);
                iy = std::clamp(iy, 0, stage_h - 1);
                size_t acc_idx = (static_cast<size_t>(iy) * static_cast<size_t>(stage_w) + static_cast<size_t>(ix)) * 3u;
                if (acc_idx + 2 >= accum.size()) continue;
                float weight = (plen > 17) ? sample[17] : 1.0f;
                float w = std::max(0.0f, weight);
                if (plen > 14) accum[acc_idx + 0] += sample[14] * w;
                if (plen > 15) accum[acc_idx + 1] += sample[15] * w;
                if (plen > 16) accum[acc_idx + 2] += sample[16] * w;
            }
        }
    }

    float clamp_max = std::max(0.0f, kIntegratorTemporalMax);
    float scale_max = (clamp_max > 0.0f) ? (1.0f / clamp_max) : 1.0f;
    float exposure = std::max(0.0f, kIntegratorExposure);
    size_t pixel_count = static_cast<size_t>(stage_w) * static_cast<size_t>(stage_h);
    if (tmp.size() < pixel_count * 4u) tmp.resize(pixel_count * 4u);
    for (size_t pi = 0; pi < pixel_count; ++pi) {
        size_t acc_idx = pi * 3u;
        size_t out_idx = pi * 4u;
        float r = accum[acc_idx + 0];
        float g = accum[acc_idx + 1];
        float b = accum[acc_idx + 2];
        if (clamp_max > 0.0f) {
            r = std::min(r, clamp_max);
            g = std::min(g, clamp_max);
            b = std::min(b, clamp_max);
        }
        float rr = std::clamp(r * exposure * scale_max, 0.0f, 1.0f);
        float gg = std::clamp(g * exposure * scale_max, 0.0f, 1.0f);
        float bb = std::clamp(b * exposure * scale_max, 0.0f, 1.0f);
        tmp[out_idx + 0] = static_cast<uint8_t>(std::lround(rr * 255.0f));
        tmp[out_idx + 1] = static_cast<uint8_t>(std::lround(gg * 255.0f));
        tmp[out_idx + 2] = static_cast<uint8_t>(std::lround(bb * 255.0f));
        tmp[out_idx + 3] = 255;
    }
}

static bool stage_module_has_input_connection(const GP_CanvasContextImpl* ctx, int module_idx, int in_count) {
    if (!ctx) return false;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return false;
    if (in_count <= 0) return false;
    for (const auto &edge : ctx->edges) {
        const auto &desc = edge.desc;
        if (desc.b_module == module_idx && desc.b_contact_idx >= 0 && desc.b_contact_idx < in_count) {
            return true;
        }
    }
    return false;
}

static void configure_stage_scanner_mode(GP_StageContext* st) {
    // Mirror defaults used during stage creation.
    gp_stage_set_temporal(st, /*decay=*/0.93f, /*max_intensity=*/1.0f);
    gp_stage_set_exposure(st, /*exposure=*/1.0f);
}

static void configure_stage_integrator_mode(GP_StageContext* st) {
    gp_stage_set_temporal(st, /*decay=*/0.995f, /*max_intensity=*/4.0f);
    gp_stage_set_exposure(st, /*exposure=*/0.55f);
}

static void apply_log_response(uint8_t* pix, size_t pixel_count) {
    if (!pix) return;
    constexpr float kLogScale = 18.0f;
    const float denom = std::log1p(kLogScale);
    for (size_t i = 0; i < pixel_count; ++i) {
        uint8_t* base = pix + i * 4;
        for (int ch = 0; ch < 3; ++ch) {
            float norm = static_cast<float>(base[ch]) / 255.0f;
            float mapped = std::log1p(norm * kLogScale) / denom;
            mapped = std::clamp(mapped, 0.0f, 1.0f);
            base[ch] = static_cast<uint8_t>(std::lround(mapped * 255.0f));
        }
        base[3] = 255;
    }
}

static void stage_bg_callback(void* user, int module_idx, int width, int height, uint8_t* out_rgba, int32_t out_pitch) {
    if (!user || !out_rgba || width <= 0 || height <= 0) return;
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(user);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->module_stages.size())) return;
    GP_StageContext* st = c->module_stages[module_idx];
    if (!st) return;
    int stage_h = std::max(1, height);
    int in_count = 0;
    if (module_idx >= 0 && module_idx < static_cast<int>(c->module_io_in_count.size())) {
        in_count = c->module_io_in_count[module_idx];
    }
    bool integrator_mode = stage_module_has_input_connection(c, module_idx, in_count);
    if (module_idx >= static_cast<int>(c->module_stage_integrator_mode.size())) {
        c->module_stage_integrator_mode.resize(module_idx + 1, 0);
    }
    bool was_integrator = (c->module_stage_integrator_mode[module_idx] != 0);
    if (integrator_mode != was_integrator) {
        if (integrator_mode) {
            configure_stage_integrator_mode(st);
            auto& accum = stage_get_integrator_buffer(c, module_idx, width, stage_h);
            std::fill(accum.begin(), accum.end(), 0.0f);
        } else {
            configure_stage_scanner_mode(st);
        }
        c->module_stage_integrator_mode[module_idx] = integrator_mode ? 1 : 0;
    }
    // render full module height (no header band baked into the stage image)
    gp_stage_resize(st, width, stage_h);
    std::vector<uint8_t> tmp;
    tmp.resize(static_cast<size_t>(width) * static_cast<size_t>(stage_h) * 4u);
    // UI should not run stage work. Use cached completed RGBA if available.
    if (integrator_mode) {
        GP_TableContext* table = c->container_table ? c->container_table : ((module_idx < static_cast<int>(c->module_tables.size())) ? c->module_tables[module_idx] : nullptr);
        // If cached texture exists and matches size, blit it; otherwise fall back
        // to rendering a subtle grid so UI shows something.
        bool blitted = false;
        if (module_idx >= 0 && module_idx < static_cast<int>(c->module_stage_cache_rgba.size())) {
            int cw = c->module_stage_cache_w[module_idx];
            int ch = c->module_stage_cache_h[module_idx];
            int cp = c->module_stage_cache_pitch[module_idx];
            if (cw == width && ch == stage_h && cp >= width * 4) {
                std::lock_guard<std::mutex> lk(*c->module_stage_cache_mu[module_idx]);
                const uint8_t* src = c->module_stage_cache_rgba[module_idx].data();
                if (src && !c->module_stage_cache_rgba[module_idx].empty()) {
                    for (int y = 0; y < height; ++y) {
                        uint8_t* row = out_rgba + y * out_pitch;
                        if (y < ch) {
                            const uint8_t* srow = src + static_cast<size_t>(y) * static_cast<size_t>(cp);
                            std::memcpy(row, srow, static_cast<size_t>(width) * 4);
                        } else {
                            std::memset(row, 0, static_cast<size_t>(width) * 4);
                        }
                    }
                    blitted = true;
                }
            }
        }
        if (!blitted) {
            gp_stage_render(st);
            render_stage_integrator_image(c, module_idx, table, width, stage_h, tmp);
        }
    } else {
        // Non-integrator mode: prefer cached texture, else copy latest stage image
        bool blitted = false;
        if (module_idx >= 0 && module_idx < static_cast<int>(c->module_stage_cache_rgba.size())) {
            int cw = c->module_stage_cache_w[module_idx];
            int ch = c->module_stage_cache_h[module_idx];
            int cp = c->module_stage_cache_pitch[module_idx];
            if (cw == width && ch == stage_h && cp >= width * 4) {
                std::lock_guard<std::mutex> lk(*c->module_stage_cache_mu[module_idx]);
                const uint8_t* src = c->module_stage_cache_rgba[module_idx].data();
                if (src && !c->module_stage_cache_rgba[module_idx].empty()) {
                    for (int y = 0; y < height; ++y) {
                        uint8_t* row = out_rgba + y * out_pitch;
                        if (y < ch) {
                            const uint8_t* srow = src + static_cast<size_t>(y) * static_cast<size_t>(cp);
                            std::memcpy(row, srow, static_cast<size_t>(width) * 4);
                        } else {
                            std::memset(row, 0, static_cast<size_t>(width) * 4);
                        }
                    }
                    blitted = true;
                }
            }
        }
        if (!blitted) {
            gp_stage_render(st);
            gp_stage_copy_rgba(st, tmp.data(), static_cast<int32_t>(tmp.size()));
        }
    }
    // If stage produced an all-zero frame (e.g., no emitters configured yet), paint a subtle fallback grid
    // so the module isn't rendered as a solid black box.
    bool all_zero = true;
    for (size_t i = 0; i + 3 < tmp.size(); i += 4) {
        if (tmp[i + 0] || tmp[i + 1] || tmp[i + 2]) { all_zero = false; break; }
    }
    if (all_zero) {
        for (int y = 0; y < stage_h; ++y) {
            for (int x = 0; x < width; ++x) {
                size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) * 4u + static_cast<size_t>(x) * 4u;
                uint8_t v = static_cast<uint8_t>(10 + ((x / 16 + y / 16) % 2) * 12);
                tmp[idx + 0] = v;
                tmp[idx + 1] = v;
                tmp[idx + 2] = v + 10;
                tmp[idx + 3] = 255;
            }
        }
    }
    if (integrator_mode) {
        apply_log_response(tmp.data(), static_cast<size_t>(stage_h) * static_cast<size_t>(width));
    }
    // copy stage image into out_rgba (no additional bands)
    for (int y = 0; y < height; ++y) {
        uint8_t* row = out_rgba + y * out_pitch;
        if (y < stage_h) {
            const uint8_t* src = tmp.data() + static_cast<size_t>(y) * width * 4;
            std::memcpy(row, src, static_cast<size_t>(width) * 4);
        } else {
            std::memset(row, 0, static_cast<size_t>(width) * 4);
        }
    }
}

// Simple chat background callback: fill module background with chat color
// and render the latest chat text (if present) using `text_render_helper`.
static void chat_bg_callback(void* user, int module_idx, int width, int height, uint8_t* out_rgba, int32_t out_pitch) {
    if (!user || !out_rgba || width <= 0 || height <= 0) return;
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(user);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->modules.size())) return;
    // pick color from chat state
    if (module_idx >= static_cast<int>(c->module_chat_color.size())) return;
    auto col = c->module_chat_color[module_idx];
    // fill background
    Color bc{col.r, col.g, col.b, col.a};
    memset_rect(out_rgba, width, height, out_pitch, 0, 0, width, height, bc);

    // render text bitmap and blit it with alpha
    if (module_idx < static_cast<int>(c->module_chat_text.size())) {
        const std::string &txt = c->module_chat_text[module_idx];
        if (!txt.empty()) {
            TextBitmap tb = render_text_to_rgba(txt, 1.0f, {255,255,255,255});
            if (tb.width > 0 && tb.height > 0 && !tb.pixels.empty()) {
                // convert pixels to uint8_t vector and blit at small inset
                std::vector<uint8_t> v(tb.pixels.begin(), tb.pixels.end());
                int px = 8;
                int py = 8;
                blit_module_buffer_srcalpha(out_rgba, width, height, out_pitch, px, py, tb.width, tb.height, v);
            }
        }
    }
}

extern "C" int gp_canvas_attach_table(GP_CanvasContext* ctx_, int module_idx, GP_TableContext* table, int take_ownership) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->modules.size())) return 0;
    // set table pointer and ownership
    c->module_tables[module_idx] = table;
    c->module_table_owned[module_idx] = take_ownership ? 1 : 0;
    // ensure module has a backing node in graph
    if (module_idx >= static_cast<int>(c->module_node_id.size())) c->module_node_id.resize(module_idx + 1, -1);
    if (c->module_node_id[module_idx] < 0) {
        int nid = c->next_node_id++;
        c->module_node_id[module_idx] = nid;
        GP_CanvasContextImpl::NodeContract nc; nc.node_id = nid; nc.module_idx = module_idx;
        c->nodes.push_back(std::move(nc));
    }
    if (table) {
        // attach table's rope sim to root table sim; do not transfer ownership
        RopeSim* sim = canvas_require_root_sim(c);
        // attach table's rope sim to root table sim; do not transfer ownership
        if (sim) gp_table_attach_rope_sim(table, sim, 0);
        // enable prospective/live mode by default for attached tables
        gp_table_set_prospective_mode(table, 1);
        gp_table_prospective_set_params(table, 8, 4.0f, 0.0f);
        // If the table has no IO sections and no existing columns, create empty
        // IO columns/rows so attached blank tables show input/output strips.
        GP_TableGeom gtmp{};
        int col_count_existing = 0;
        if (gp_table_get_geom(table, &gtmp)) {
            for (int ii = 0; ii < 8; ++ii) if (gtmp.col_w[ii] > 0) ++col_count_existing;
        }
        if (!gp_table_has_io_sections(table) && col_count_existing == 0) {
            int in_dir = 0, out_dir = 0;
            gp_table_get_side_reading_direction(table, 0, &in_dir);
            gp_table_get_side_reading_direction(table, 1, &out_dir);
            int dir = in_dir;
            if (dir == 0 || dir == 2) {
                GP_TableColumn cols[2];
                cols[0].kind = GP_TABLE_CELL_LEDS;
                cols[0].width_px = 80;
                cols[0].align = 0;
                cols[1].kind = GP_TABLE_CELL_LEDS;
                cols[1].width_px = 80;
                cols[1].align = 0;
                if (dir == 2) { GP_TableColumn tmp = cols[0]; cols[0] = cols[1]; cols[1] = tmp; }
                gp_table_set_columns(table, cols, 2);
                gp_table_set_rows(table, nullptr, 0);
            } else {
                GP_TableColumn col;
                col.kind = GP_TABLE_CELL_LEDS;
                col.width_px = 160;
                col.align = 0;
                gp_table_set_columns(table, &col, 1);
                GP_TableRow rows[2]; memset(rows, 0, sizeof(rows));
                rows[0].kind = GP_TABLE_ROW_HEADER; rows[0].depth = 0; rows[0].expanded = 1; rows[0].cell_count = 1; rows[0].cells[0].kind = GP_TABLE_CELL_LEDS;
                rows[1].kind = GP_TABLE_ROW_NOTE; rows[1].depth = 0; rows[1].expanded = 1; rows[1].cell_count = 1; rows[1].cells[0].kind = GP_TABLE_CELL_LEDS;
                if (dir == 3) { GP_TableRow tmp = rows[0]; rows[0] = rows[1]; rows[1] = tmp; }
                gp_table_set_rows(table, rows, 2);
            }
        }
        // Probe table IO keys and populate NodeContract input/output type lists
        // so the canvas knows what types this module exposes.
        if (module_idx >= 0) {
            // ensure nodes vector contains the contract for this module (created above)
            GP_CanvasContextImpl::NodeContract* found = nullptr;
            for (auto &n : c->nodes) if (n.module_idx == module_idx) { found = &n; break; }
            if (found) {
                found->input_types.clear();
                found->output_types.clear();
                const int cap = 4096;
                std::vector<unsigned long long> keys(cap);
                int nin = gp_table_enumerate_io_keys(table, 0, keys.data(), cap);
                if (nin > 0) {
                    std::unordered_set<int> in_types;
                    for (int i = 0; i < nin; ++i) {
                        unsigned long long k = keys[i];
                        int type_id = -1, is_in=0, is_out=0;
                        gp_table_get_key_type_hint(table, k, &type_id, &is_in, &is_out);
                        if (type_id >= 0) in_types.insert(type_id);
                    }
                    found->input_types.assign(in_types.begin(), in_types.end());
                }
                int nout = gp_table_enumerate_io_keys(table, 1, keys.data(), cap);
                if (nout > 0) {
                    std::unordered_set<int> out_types;
                    for (int i = 0; i < nout; ++i) {
                        unsigned long long k = keys[i];
                        int type_id = -1, is_in=0, is_out=0;
                        gp_table_get_key_type_hint(table, k, &type_id, &is_in, &is_out);
                        if (type_id >= 0) out_types.insert(type_id);
                    }
                    found->output_types.assign(out_types.begin(), out_types.end());
                }
            }
        }
    }
    return 1;
}

extern "C" int gp_canvas_set_templates_dir(const char* dir) {
    // forward to table helper
    return gp_table_set_library_dir(dir) ? 1 : 0;
}

extern "C" int gp_canvas_get_templates_dir(char* out_buf, int out_len) {
    return gp_table_get_library_dir(out_buf, out_len);
}

extern "C" int gp_canvas_set_container_table(GP_CanvasContext* ctx_, GP_TableContext* table, int take_ownership) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (c->container_table && c->container_table_owned) {
        gp_table_destroy(c->container_table);
    }
    c->container_table = table;
    c->container_table_owned = (table && take_ownership) ? 1 : 0;
    c->root_actions_installed = 0;
    if (!c->container_table) {
        canvas_ensure_root_table(c);
    } else {
        canvas_install_root_actions(c, c->container_table);
    }
    canvas_attach_tables_to_root_sim(c);
    update_canvas_scroll_state(c, /*pull_from_container=*/true);
    return 1;
}

extern "C" GP_TableContext* gp_canvas_get_container_table(GP_CanvasContext* ctx_) {
    if (!ctx_) return nullptr;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    return c->container_table;
}

extern "C" int gp_canvas_set_autosave(GP_CanvasContext* ctx_, const char* path, double interval_s) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!path || path[0] == '\0' || interval_s <= 0.0) {
        c->autosave_path.clear(); c->autosave_interval_s = 0.0; c->autosave_accum_s = 0.0; return 1;
    }
    c->autosave_path = std::string(path);
    c->autosave_interval_s = interval_s;
    c->autosave_accum_s = 0.0;
    return 1;
}

extern "C" int gp_canvas_get_autosave(GP_CanvasContext* ctx_, char* out_path, int out_len, double* out_interval_s) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (out_path && out_len > 0) {
        int32_t to_write = static_cast<int32_t>(std::min<size_t>(c->autosave_path.size(), static_cast<size_t>(out_len)));
        if (to_write > 0) memcpy(out_path, c->autosave_path.data(), static_cast<size_t>(to_write));
    }
    if (out_interval_s) *out_interval_s = c->autosave_interval_s;
    return 1;
}

extern "C" int gp_canvas_set_thread_manager_mode(GP_CanvasContext* ctx_, int mode) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!c->thread_mgr) return 0;
    ThreadManager::Mode m = (mode == 0) ? ThreadManager::Mode::FreeSpinning : ThreadManager::Mode::Scheduled;
    c->thread_mgr->set_mode(m);
    return 1;
}

extern "C" int gp_canvas_get_thread_manager_mode(GP_CanvasContext* ctx_, int* out_mode) {
    if (!ctx_ || !out_mode) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!c->thread_mgr) return 0;
    *out_mode = (c->thread_mgr->mode() == ThreadManager::Mode::FreeSpinning) ? 0 : 1;
    return 1;
}

extern "C" int gp_canvas_detach_table(GP_CanvasContext* ctx_, int module_idx) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->modules.size())) return 0;
    GP_TableContext* t = c->module_tables[module_idx];
    int owned = c->module_table_owned[module_idx];
    c->module_tables[module_idx] = nullptr;
    c->module_table_owned[module_idx] = 0;
    if (t && owned) gp_table_destroy(t);
    // detach node mapping for this module
    if (module_idx >= 0 && module_idx < static_cast<int>(c->module_node_id.size())) {
        int nid = c->module_node_id[module_idx];
        c->module_node_id[module_idx] = -1;
        for (auto &n : c->nodes) {
            if (n.node_id == nid) { n.module_idx = -1; break; }
        }
    }
    return 1;
}

extern "C" int gp_canvas_set_module_bg_callback(GP_CanvasContext* ctx_, int module_idx, GP_CanvasModuleBgFn cb, void* user) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->module_bg.size())) return 0;
    c->module_bg[module_idx].cb = cb;
    c->module_bg[module_idx].user = user;
    return 1;
}

extern "C" int gp_canvas_clear_module_bg_callback(GP_CanvasContext* ctx_, int module_idx) {
    return gp_canvas_set_module_bg_callback(ctx_, module_idx, nullptr, nullptr);
}

extern "C" int gp_canvas_set_module_bg_mode(GP_CanvasContext* ctx_, int module_idx, int mode) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->module_bg.size())) return 0;
    c->module_bg[module_idx].mode = mode;
    return 1;
}

extern "C" int gp_canvas_get_module_bg_mode(GP_CanvasContext* ctx_, int module_idx, int* out_mode) {
    if (!ctx_ || !out_mode || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->module_bg.size())) return 0;
    *out_mode = c->module_bg[module_idx].mode;
    return 1;
}

extern "C" int gp_canvas_set_module_raytrace_params(GP_CanvasContext* ctx_, int module_idx, int ray_count, int max_reflections, float blur_sigma) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->module_bg.size())) return 0;
    auto &bg = c->module_bg[module_idx];
    bg.rays = std::max(1, ray_count);
    bg.reflections = std::max(0, max_reflections);
    bg.blur_sigma = std::max(0.0f, blur_sigma);
    return 1;
}

extern "C" int gp_canvas_set_module_raytrace_tuning(GP_CanvasContext* ctx_, int module_idx, float bounce_decay, float distance_decay, float exposure) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->module_bg.size())) return 0;
    auto &bg = c->module_bg[module_idx];
    bg.ray_bounce_decay = std::clamp(bounce_decay, 0.0f, 1.0f);
    bg.ray_air_decay = std::max(0.0f, distance_decay);
    bg.ray_exposure = std::max(0.0f, exposure);
    return 1;
}

extern "C" int gp_canvas_set_module_table_alpha(GP_CanvasContext* ctx_, int module_idx, float alpha, float raytrace_alpha) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->module_bg.size())) return 0;
    auto &bg = c->module_bg[module_idx];
    float a = std::clamp(alpha, 0.0f, 1.0f);
    float ar = std::clamp(raytrace_alpha, 0.0f, 1.0f);
    bg.table_alpha = static_cast<uint8_t>(std::lround(a * 255.0f));
    bg.table_alpha_ray = static_cast<uint8_t>(std::lround(ar * 255.0f));
    return 1;
}

extern "C" int gp_canvas_add_edge(GP_CanvasContext* ctx_, const GP_CanvasEdgeDesc* desc) {
    // legacy: add untyped edge (type_id == 0)
    return gp_canvas_add_edge_with_type(ctx_, desc, 0);
}

// add an edge with an explicit type id. type_id == 0 means untyped/wildcard.
extern "C" int gp_canvas_add_edge_with_type(GP_CanvasContext* ctx_, const GP_CanvasEdgeDesc* desc, int type_id) {
    if (!ctx_ || !desc) return -1;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // basic bounds
    if (desc->a_module < 0 || desc->a_module >= static_cast<int>(c->modules.size())) return -1;
    if (desc->b_module < 0 || desc->b_module >= static_cast<int>(c->modules.size())) return -1;
    // If a type is specified, validate against module contracts: a_module must provide the type as output and b_module must accept as input.
    if (type_id != 0) {
        int na = (desc->a_module < static_cast<int>(c->module_node_id.size())) ? c->module_node_id[desc->a_module] : -1;
        int nb = (desc->b_module < static_cast<int>(c->module_node_id.size())) ? c->module_node_id[desc->b_module] : -1;
        if (na < 0 || nb < 0) return -1;
        // find node contracts
        GP_CanvasContextImpl::NodeContract* nodeA = nullptr;
        GP_CanvasContextImpl::NodeContract* nodeB = nullptr;
        for (auto &n : c->nodes) {
            if (n.node_id == na) nodeA = &n;
            if (n.node_id == nb) nodeB = &n;
        }
        if (!nodeA || !nodeB) return -1;
        bool a_supports = std::find(nodeA->output_types.begin(), nodeA->output_types.end(), type_id) != nodeA->output_types.end();
        bool b_supports = std::find(nodeB->input_types.begin(), nodeB->input_types.end(), type_id) != nodeB->input_types.end();
        if (!a_supports || !b_supports) {
            printf("gp_canvas_add_edge_with_type: type %d not supported by modules %d->%d\n", type_id, desc->a_module, desc->b_module);
            return -1;
        }
    }

    GP_CanvasContextImpl::EdgeInfo ei;
    ei.desc = *desc;
    ei.type_id = type_id;
    // copy canvas-level default hues into edge if available
    if (!c->hues.empty()) {
        ei.hues = c->hues;
        ei.hue_intensity = c->hue_intensity;
    }
    // Ensure container/root table has corresponding edge keys so the
    // root table (canonical graph) will host the data FIFOs for canvas
    // edges. This makes tables defer their graphs to the root canvas graph.
    GP_TableContext* root = canvas_ensure_root_table(c);
    if (root) {
        // determine required columns/rows to cover contact indices and modules
        int max_module = std::max(desc->a_module, desc->b_module);
        int max_contact = std::max(desc->a_contact_idx, desc->b_contact_idx);
        int cols_needed = std::min(8, std::max(1, max_contact + 1));
        int rows_needed = std::max(1, max_module + 1);
        // setup simple LED columns if the table appears empty or too small
        GP_TableGeom gtmp{};
        int col_count_existing = 0;
        if (gp_table_get_geom(root, &gtmp)) {
            for (int ii = 0; ii < 8; ++ii) if (gtmp.col_w[ii] > 0) ++col_count_existing;
        }
        if (col_count_existing < cols_needed) {
            GP_TableColumn cols[8];
            for (int i = 0; i < cols_needed; ++i) { cols[i].kind = GP_TABLE_CELL_LEDS; cols[i].width_px = 80; cols[i].align = 0; }
            gp_table_set_columns(root, cols, cols_needed);
        }
        // ensure rows exist
        std::vector<GP_TableRow> rows;
        rows.resize(rows_needed);
        for (int ri = 0; ri < rows_needed; ++ri) {
            std::memset(&rows[ri], 0, sizeof(GP_TableRow));
            rows[ri].kind = GP_TABLE_ROW_DEVICE;
            rows[ri].depth = 0;
            rows[ri].expanded = 1;
            rows[ri].cell_count = cols_needed;
            for (int ci = 0; ci < cols_needed && ci < 8; ++ci) rows[ri].cells[ci].kind = GP_TABLE_CELL_LEDS;
        }
        gp_table_set_rows(root, rows.data(), static_cast<int>(rows.size()));

        // compute keys for endpoints and add edge to root table so it creates FIFOs
        uint64_t ka = (static_cast<uint64_t>(static_cast<uint32_t>(desc->a_module)) << 32) |
                      (static_cast<uint64_t>(static_cast<uint32_t>(desc->a_contact_idx)) << 16) |
                      static_cast<uint64_t>(0);
        uint64_t kb = (static_cast<uint64_t>(static_cast<uint32_t>(desc->b_module)) << 32) |
                      (static_cast<uint64_t>(static_cast<uint32_t>(desc->b_contact_idx)) << 16) |
                      static_cast<uint64_t>(0);
        printf("gp_canvas_add_edge_with_type: creating root edge ka=0x%016llx kb=0x%016llx root=%p\n", (unsigned long long)ka, (unsigned long long)kb, (void*)root);
        // Enqueue edge addition to be applied by the manager thread.
        gp_table_enqueue_add_edge(root, ka, kb);
    }

    c->edges.push_back(std::move(ei));
    int idx = static_cast<int>(c->edges.size() - 1);
    printf("gp_canvas_add_edge_with_type: added edge %d type=%d a=%d.%d b=%d.%d\n", idx, type_id, desc->a_module, desc->a_contact_idx, desc->b_module, desc->b_contact_idx);
    return idx;
}

// Set the module's supported input/output type lists
extern "C" int gp_canvas_set_module_io_types(GP_CanvasContext* ctx_, int module_idx, const int* input_types, int input_count, const int* output_types, int output_count) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->modules.size())) return 0;
    int nid = -1;
    if (module_idx < static_cast<int>(c->module_node_id.size())) nid = c->module_node_id[module_idx];
    if (nid < 0) return 0;
    // find node
    for (auto &n : c->nodes) {
        if (n.node_id == nid) {
            n.input_types.clear(); n.output_types.clear();
            if (input_types && input_count > 0) n.input_types.assign(input_types, input_types + input_count);
            if (output_types && output_count > 0) n.output_types.assign(output_types, output_types + output_count);
            return 1;
        }
    }
    return 0;
}

extern "C" int gp_canvas_get_module_node_id(GP_CanvasContext* ctx_, int module_idx) {
    if (!ctx_ || module_idx < 0) return -1;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->module_node_id.size())) return -1;
    return c->module_node_id[module_idx];
}

// Persist canvas state to a simple line-based file format. This is intentionally
// lightweight and human readable so it's easy to edit by hand during development.
// Format (V1):
// CANVAS V1
// WIDTH HEIGHT CONTROL_BAR_H
// MODULE x y w h label
// EDGE a_module a_contact_idx b_module b_contact_idx type_id
// NODE node_id module_idx input_count [inputs...] output_count [outputs...]
// Lines may appear in any order; loader will reconstruct internal vectors.
extern "C" int gp_canvas_save_to_file(GP_CanvasContext* ctx_, const char* path) {
    if (!ctx_ || !path) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    std::ofstream ofs(path);
    if (!ofs.good()) return 0;
    ofs << "CANVAS V1\n";
    ofs << c->width << " " << c->height << " " << c->control_bar_h << "\n";
    ofs << "OFFSET " << c->offset_x << " " << c->offset_y << "\n";
    // modules
    for (size_t i = 0; i < c->modules.size(); ++i) {
        const auto &m = c->modules[i];
        // label may contain spaces; write as remainder of line
        ofs << "MODULE " << m.x << " " << m.y << " " << m.w << " " << m.h << " ";
        // write label as-is but escape newlines and backslashes
        std::string lbl(m.label, m.label + sizeof(m.label));
        // trim at first null
        size_t z = lbl.find('\0'); if (z != std::string::npos) lbl.resize(z);
        for (char ch : lbl) {
            if (ch == '\\') ofs << "\\\\";
            else if (ch == '\n') ofs << "\\n";
            else ofs << ch;
        }
        ofs << "\n";
    }
    // edges
    for (const auto &ei : c->edges) {
        const auto &e = ei.desc;
        ofs << "EDGE " << e.a_module << " " << e.a_contact_idx << " " << e.b_module << " " << e.b_contact_idx << " " << ei.type_id << "\n";
    }
    // nodes/contracts
    for (const auto &n : c->nodes) {
        ofs << "NODE " << n.node_id << " " << n.module_idx << " ";
        ofs << static_cast<int>(n.input_types.size());
        for (int t : n.input_types) ofs << " " << t;
        ofs << " " << static_cast<int>(n.output_types.size());
        for (int t : n.output_types) ofs << " " << t;
        ofs << "\n";
    }
    ofs.close();
    return 1;
}

// Load canvas state from file written by `gp_canvas_save_to_file`. The loader
// will clear current modules/edges/nodes and recreate them. Any canvas-owned
// tables will be destroyed. Returns 1 on success.
extern "C" int gp_canvas_load_from_file(GP_CanvasContext* ctx_, const char* path) {
    if (!ctx_ || !path) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    std::ifstream ifs(path);
    if (!ifs.good()) return 0;
    // clear existing modules/edges/nodes (destroy owned tables/stages)
    for (size_t i = 0; i < c->module_tables.size(); ++i) {
        if (c->module_tables[i] && c->module_table_owned[i]) gp_table_destroy(c->module_tables[i]);
    }
    for (size_t i = 0; i < c->module_stages.size(); ++i) {
        if (c->module_stages[i] && i < c->module_stage_owned.size() && c->module_stage_owned[i]) gp_stage_destroy(c->module_stages[i]);
    }
    c->modules.clear();
    c->module_tables.clear();
    c->module_table_owned.clear();
    c->module_stages.clear();
    c->module_stage_owned.clear();
    c->module_is_stage.clear();
    c->module_stage_images.clear();
    c->module_stage_integrator_mode.clear();
    c->module_stage_integrator_accum.clear();
    c->module_bg.clear();
    c->module_io_in_count.clear();
    c->module_io_out_count.clear();
    c->module_io_input_rows.clear();
    c->module_io_output_rows.clear();
    c->module_input_layout.clear();
    c->module_output_layout.clear();
    c->module_io_rows.clear();
    c->module_table_rows.clear();
    c->module_stack_snapshots.clear();
    c->module_input_state.clear();
    c->edges.clear();
    c->nodes.clear();
    c->module_node_id.clear();
    c->module_tool_stack.clear();

    std::string line;
    // optionally read header
    if (!std::getline(ifs, line)) return 0;
    if (line.rfind("CANVAS", 0) == 0) {
        // read dims
        if (!std::getline(ifs, line)) return 0;
        std::istringstream sh(line);
        int w,h,cbh; sh >> w >> h >> cbh;
        c->width = w; c->height = h; c->control_bar_h = cbh;
    } else {
        // no header; reset stream to beginning
        ifs.clear(); ifs.seekg(0);
    }

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tag; ss >> tag;
        if (tag == "MODULE") {
                GP_CanvasModuleDesc m{};
                ss >> m.x >> m.y >> m.w >> m.h;
            // remainder of line is label (may be empty)
            std::string rest;
            std::getline(ss, rest);
            // trim leading spaces
            if (!rest.empty() && rest[0] == ' ') rest.erase(0,1);
            // unescape backslashes and \n
            std::string lbl; lbl.reserve(rest.size());
            for (size_t i = 0; i < rest.size(); ++i) {
                char ch = rest[i];
                if (ch == '\\' && i + 1 < rest.size()) {
                    char nx = rest[i+1];
                    if (nx == 'n') { lbl.push_back('\n'); ++i; }
                    else { lbl.push_back(nx); ++i; }
                } else lbl.push_back(ch);
            }
            // copy into fixed label buffer
            std::memset(m.label, 0, sizeof(m.label));
            std::memcpy(m.label, lbl.c_str(), std::min<size_t>(lbl.size(), sizeof(m.label)-1));
            // append module
            c->modules.push_back(m);
            c->module_tables.push_back(nullptr);
            c->module_table_owned.push_back(0);
            c->module_bg.emplace_back();
            c->module_tool_stack.emplace_back();
            c->module_io_in_count.push_back(0);
            c->module_io_out_count.push_back(0);
            c->module_io_input_rows.emplace_back();
            c->module_io_output_rows.emplace_back();
            c->module_input_layout.emplace_back();
            c->module_output_layout.emplace_back();
            c->module_io_rows.emplace_back();
            c->module_table_rows.emplace_back();
            c->module_stack_snapshots.emplace_back();
            c->module_input_state.emplace_back();
            // assign node id for this module
            int nid = c->next_node_id++;
            c->module_node_id.push_back(nid);
            GP_CanvasContextImpl::NodeContract nc; nc.node_id = nid; nc.module_idx = static_cast<int>(c->modules.size() - 1);
            c->nodes.push_back(std::move(nc));
        } else if (tag == "EDGE") {
            GP_CanvasEdgeDesc e{}; int type_id = 0;
            ss >> e.a_module >> e.a_contact_idx >> e.b_module >> e.b_contact_idx >> type_id;
            GP_CanvasContextImpl::EdgeInfo ei; ei.desc = e; ei.type_id = type_id;
            if (!c->hues.empty()) { ei.hues = c->hues; ei.hue_intensity = c->hue_intensity; }
            c->edges.push_back(std::move(ei));
        } else if (tag == "NODE") {
            int nid = -1; int midx = -1; ss >> nid >> midx;
            GP_CanvasContextImpl::NodeContract *found = nullptr;
            for (auto &n : c->nodes) if (n.node_id == nid) { found = &n; break; }
            if (!found) { GP_CanvasContextImpl::NodeContract nc; nc.node_id = nid; nc.module_idx = midx; c->nodes.push_back(std::move(nc)); found = &c->nodes.back(); }
            int in_count = 0; ss >> in_count;
            for (int i = 0; i < in_count; ++i) { int t; ss >> t; found->input_types.push_back(t); }
            int out_count = 0; ss >> out_count;
            for (int i = 0; i < out_count; ++i) { int t; ss >> t; found->output_types.push_back(t); }
        } else if (tag == "OFFSET") {
            ss >> c->offset_x >> c->offset_y;
        }
    }
    update_canvas_scroll_state(c, /*pull_from_container=*/false);
    // done
    return 1;
}

extern "C" int gp_canvas_set_cable_style(GP_CanvasContext* ctx_, int jacket_px, int jacket_border) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->jacket_px = std::max(1, jacket_px);
    c->jacket_border = std::max(0, jacket_border);
    return 1;
}

extern "C" int gp_canvas_set_edge_hues(GP_CanvasContext* ctx_, const float* hues, int hue_count, float hue_intensity) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->hues.clear();
    c->hue_intensity = std::clamp(hue_intensity, 0.0f, 1.0f);
    if (!hues || hue_count <= 0) return 1;
    c->hues.assign(hues, hues + hue_count);
    // propagate to existing edges as a default palette (doesn't override per-edge custom hues)
    for (auto &ei : c->edges) {
        if (ei.hues.empty()) {
            ei.hues = c->hues;
            ei.hue_intensity = c->hue_intensity;
        }
    }
    return 1;
}

extern "C" int gp_canvas_raster_rgba(GP_CanvasContext* ctx_, uint8_t* out_rgba, int32_t out_len_bytes) {
    if (!ctx_ || !out_rgba) return 0;
    auto *ctx = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    int w = ctx->width;
    int h = ctx->height;
    int pitch = w * 4;
    if (out_len_bytes < w * h * 4) return 0;

    update_canvas_scroll_state(ctx, /*pull_from_container=*/true);

    // clear
    memset(out_rgba, 0, static_cast<size_t>(w) * h * 4);
    // draw control bar at top with toggle tool buttons
    int rb = ctx->rope_bar_h;
    if (rb > 0) {
        // draw rope sim bar at very top
        memset_rect(out_rgba, w, h, pitch, 0, 0, w, rb, Color{22,22,28,255});
        // draw simple controls: segs +/- at left, slack +/- at right, and display values
        int bw = std::max(4, rb - 8);
        int spacing = 8;
        int bx = 8; int byy = 4;
        memset_rect(out_rgba, w, h, pitch, bx, byy, bw, rb - 8, Color{60,60,72,255}); bx += bw + spacing;
        memset_rect(out_rgba, w, h, pitch, bx, byy, bw, rb - 8, Color{60,60,72,255});
        // slack buttons on right
        int bx2 = w - 8 - bw*2 - spacing; memset_rect(out_rgba, w, h, pitch, bx2, byy, bw, rb - 8, Color{60,60,72,255}); bx2 += bw + spacing; memset_rect(out_rgba, w, h, pitch, bx2, byy, bw, rb - 8, Color{60,60,72,255});
        // value text (render_text_to_rgba is available)
        {
            std::string s = std::string(LABEL_ROPE_SEGS_PREFIX) + std::to_string(ctx->sim_segs) + LABEL_ROPE_SLACK_PREFIX + std::to_string(ctx->sim_slack);
            auto bm = render_text_to_rgba(s, 1.0f, {220,220,220,255});
            if (!bm.pixels.empty()) {
                int tx = (w - bm.width) / 2;
                int ty = (rb - bm.height) / 2;
                for (int yy = 0; yy < bm.height; ++yy) {
                    int dst_y = ty + yy;
                    if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < bm.width; ++xx) {
                        int dst_x = tx + xx;
                        if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
        }
        // render labels for the small rope-bar buttons (Seg -, Seg +, Slack -, Slack +)
        {
            int bw = std::max(4, rb - 8);
            int spacing = 8;
            int bx = 8;
            int byy = 4;
            std::vector<std::string> lbls = {LABEL_ROPE_SEGS_DEC, LABEL_ROPE_SEGS_INC};
            for (int i = 0; i < 2; ++i) {
                int bx_i = bx + i * (bw + spacing);
                auto tb = render_text_to_rgba(lbls[i], 0.9f, {230,230,230,255});
                if (!tb.pixels.empty()) {
                    int tx = bx_i + (bw - tb.width) / 2;
                    int ty = byy + (rb - 8 - tb.height) / 2;
                    for (int yy = 0; yy < tb.height; ++yy) {
                        int dst_y = ty + yy;
                        if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < tb.width; ++xx) {
                            int dst_x = tx + xx;
                            if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &tb.pixels[(yy * tb.width + xx) * 4];
                            float sa = src[3] / 255.0f;
                            if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) {
                                for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                dst[3] = 255;
                            }
                        }
                    }
                }
            }
            int bx2 = w - 8 - bw*2 - spacing;
            std::vector<std::string> lbls2 = {LABEL_ROPE_SLACK_DEC, LABEL_ROPE_SLACK_INC};
            for (int i = 0; i < 2; ++i) {
                int bx_i = bx2 + i * (bw + spacing);
                auto tb = render_text_to_rgba(lbls2[i], 0.9f, {230,230,230,255});
                if (!tb.pixels.empty()) {
                    int tx = bx_i + (bw - tb.width) / 2;
                    int ty = byy + (rb - 8 - tb.height) / 2;
                    for (int yy = 0; yy < tb.height; ++yy) {
                        int dst_y = ty + yy;
                        if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < tb.width; ++xx) {
                            int dst_x = tx + xx;
                            if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &tb.pixels[(yy * tb.width + xx) * 4];
                            float sa = src[3] / 255.0f;
                            if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) {
                                for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                dst[3] = 255;
                            }
                        }
                    }
                }
            }
        }
    }
    int cbh = ctx->control_bar_h;
    if (cbh > 0) {
        memset_rect(out_rgba, w, h, pitch, 0, rb, w, cbh, Color{28,28,34,255});
        // two tool groups: canvas (left) and table (right)
        const int canvas_btn_count = 4;
        const int edge_btn_count = 4;
        const int table_btn_count = 3;
        const int spacing = 8;
        int bh = std::max(4, cbh - 8);
        int bw = bh; // square buttons
        // left (canvas) group
        int bx = 8; int by = rb + 4;
        int canvas_group_w = canvas_btn_count * (bw + spacing) - spacing;
        for (int bi = 0; bi < canvas_btn_count; ++bi) {
            int bx_i = bx + bi * (bw + spacing);
            bool selected = (ctx->selected_tool_canvas == bi);
            Color fill = selected ? Color{90,90,110,255} : Color{60,60,72,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, by, bw, bh, fill);
            // left/right border
            for (int oy = 0; oy < bh; ++oy) {
                int y = by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
            // render canvas tool labels (full text) and a short letter inside the button
            const char* canvas_labels[4] = {
                LABEL_CANVAS_TOOL_SELECT,
                LABEL_CANVAS_TOOL_NEW_TABLE,
                LABEL_CANVAS_TOOL_EDGE_MODE,
                LABEL_CANVAS_TOOL_NEW_STAGE
            };
            auto lbm = render_text_to_rgba(canvas_labels[bi], 0.95f, {240,240,240,255});
            const char* canvas_short[4] = {
                LABEL_CANVAS_TOOL_SHORT_SELECT,
                LABEL_CANVAS_TOOL_SHORT_TABLE,
                LABEL_CANVAS_TOOL_SHORT_EDGE,
                LABEL_CANVAS_TOOL_SHORT_STAGE
            };
            if (!lbm.pixels.empty()) {
                int tx = bx_i + (bw - lbm.width) / 2;
                int ty = by + (bh - lbm.height) / 2;
                for (int yy = 0; yy < lbm.height; ++yy) {
                    int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < lbm.width; ++xx) {
                        int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &lbm.pixels[(yy * lbm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
            // small centered letter inside the square button for quick ID
            auto small = render_text_to_rgba(canvas_short[bi], 1.1f, {240,240,240,255});
            if (!small.pixels.empty()) {
                int txs = bx_i + (bw - small.width) / 2;
                int tys = by + (bh - small.height) / 2;
                for (int yy = 0; yy < small.height; ++yy) {
                    int dst_y = tys + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < small.width; ++xx) {
                        int dst_x = txs + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &small.pixels[(yy * small.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
        }
        // middle (edge) group
        int bx_edge = bx + canvas_group_w + spacing * 2;
        for (int bi = 0; bi < edge_btn_count; ++bi) {
            int bx_i = bx_edge + bi * (bw + spacing);
            bool selected = (ctx->selected_tool_edge == bi);
            Color fill = selected ? Color{90,80,70,255} : Color{60,54,48,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, by, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
            const char* edge_labels[4] = {
                LABEL_EDGE_TOOL_CREATE,
                LABEL_EDGE_TOOL_DESTROY,
                LABEL_EDGE_TOOL_ON_CHANGE,
                LABEL_EDGE_TOOL_CONTINUOUS
            };
            auto lbm = render_text_to_rgba(edge_labels[bi], 0.85f, {235,228,218,255});
            const char* edge_short[4] = {
                LABEL_EDGE_TOOL_SHORT_CREATE,
                LABEL_EDGE_TOOL_SHORT_DESTROY,
                LABEL_EDGE_TOOL_SHORT_ON_CHANGE,
                LABEL_EDGE_TOOL_SHORT_CONTINUOUS
            };
            if (!lbm.pixels.empty()) {
                int tx = bx_i + (bw - lbm.width) / 2;
                int ty = by + (bh - lbm.height) / 2;
                for (int yy = 0; yy < lbm.height; ++yy) {
                    int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < lbm.width; ++xx) {
                        int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &lbm.pixels[(yy * lbm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
            auto small = render_text_to_rgba(edge_short[bi], 1.05f, {235,228,218,255});
            if (!small.pixels.empty()) {
                int txs = bx_i + (bw - small.width) / 2;
                int tys = by + (bh - small.height) / 2;
                for (int yy = 0; yy < small.height; ++yy) {
                    int dst_y = tys + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < small.width; ++xx) {
                        int dst_x = txs + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &small.pixels[(yy * small.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
        }
        // right (table) group
        int group_width = table_btn_count * (bw + spacing) - spacing;
        int bx_r = std::max(8, w - 8 - group_width);
        for (int bi = 0; bi < table_btn_count; ++bi) {
            int bx_i = bx_r + bi * (bw + spacing);
            Color fill = (ctx->selected_tool_table == bi) ? Color{90,70,90,255} : Color{60,50,60,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, by, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
            // render table tool labels (full text) and a short letter inside the button
            const char* table_labels[3] = {
                LABEL_TABLE_TOOL_SELECT,
                LABEL_TABLE_TOOL_EDIT,
                LABEL_TABLE_TOOL_MORE
            };
            auto lbm2 = render_text_to_rgba(table_labels[bi], 0.85f, {230,220,240,255});
            const char* table_short[3] = {
                LABEL_TABLE_TOOL_SHORT_SELECT,
                LABEL_TABLE_TOOL_SHORT_EDIT,
                LABEL_TABLE_TOOL_SHORT_MORE
            };
            if (!lbm2.pixels.empty()) {
                int tx = bx_i + (bw - lbm2.width) / 2;
                int ty = by + (bh - lbm2.height) / 2;
                for (int yy = 0; yy < lbm2.height; ++yy) {
                    int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < lbm2.width; ++xx) {
                        int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &lbm2.pixels[(yy * lbm2.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
                    // small centered letter inside the square button for quick ID
                    auto small2 = render_text_to_rgba(table_short[bi], 1.0f, {230,220,240,255});
                    if (!small2.pixels.empty()) {
                        int txs = bx_i + (bw - small2.width) / 2;
                        int tys = by + (bh - small2.height) / 2;
                        for (int yy = 0; yy < small2.height; ++yy) {
                            int dst_y = tys + yy; if (dst_y < 0 || dst_y >= h) continue;
                            for (int xx = 0; xx < small2.width; ++xx) {
                                int dst_x = txs + xx; if (dst_x < 0 || dst_x >= w) continue;
                                uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                const unsigned char* src = &small2.pixels[(yy * small2.width + xx) * 4];
                                float sa = src[3] / 255.0f;
                                if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                else if (sa > 0.001f) {
                                    for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                    dst[3] = 255;
                                }
                            }
                        }
                    }
            }
        }
        // IO counter shown to the left of the table buttons
        auto draw_io_group = [&](int base_x, int byy, int in_count, const char* label) {
            // minus box, number area, plus box
            int nbw = bw;
            int num_w = std::max(24, nbw * 2);
            int gap = 10;
            int bx_minus = base_x - (nbw + gap + num_w + gap + nbw);
            int bx_num = bx_minus + nbw + gap;
            int bx_plus = bx_num + num_w + gap;
            // minus
            memset_rect(out_rgba, w, h, pitch, bx_minus, byy, nbw, bh, Color{50,50,56,255});
            // number background
            memset_rect(out_rgba, w, h, pitch, bx_num, byy, num_w, bh, Color{36,36,42,255});
            // plus
            memset_rect(out_rgba, w, h, pitch, bx_plus, byy, nbw, bh, Color{50,50,56,255});
            // render number text centered
            std::string s = std::to_string(in_count);
            auto bm = render_text_to_rgba(s, 1.0f, {255,255,255,255});
            if (!bm.pixels.empty()) {
                int tx = bx_num + (num_w - bm.width) / 2;
                int ty = byy + (bh - bm.height) / 2;
                for (int yy = 0; yy < bm.height; ++yy) {
                    int dst_y = ty + yy;
                    if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < bm.width; ++xx) {
                        int dst_x = tx + xx;
                        if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
            // tiny label: render above number area
            if (label && label[0] != '\0') {
                auto lb = render_text_to_rgba(label, 0.75f, {200,200,200,255});
                if (!lb.pixels.empty()) {
                    int tx = bx_num + (num_w - lb.width) / 2;
                    int ty = byy - lb.height - 2; // place slightly above number
                    for (int yy = 0; yy < lb.height; ++yy) {
                        int dst_y = ty + yy;
                        if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < lb.width; ++xx) {
                            int dst_x = tx + xx;
                            if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &lb.pixels[(yy * lb.width + xx) * 4];
                            float sa = src[3] / 255.0f;
                            if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) {
                                for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                dst[3] = 255;
                            }
                        }
                    }
                }
            }
        };
        auto draw_io_action_pair = [&](int left_x, int byy, const char* left_label, const char* right_label) {
            int nbw = bw;
            int gap = 10;
            int bx_left = left_x;
            int bx_right = bx_left + nbw + gap;
            memset_rect(out_rgba, w, h, pitch, bx_left, byy, nbw, bh, Color{50,50,56,255});
            memset_rect(out_rgba, w, h, pitch, bx_right, byy, nbw, bh, Color{50,50,56,255});
            auto draw_label = [&](int bx, const char* label) {
                auto bm = render_text_to_rgba(label, 0.9f, {230,230,235,255});
                if (!bm.pixels.empty()) {
                    int tx = bx + (nbw - bm.width) / 2;
                    int ty = byy + (bh - bm.height) / 2;
                    for (int yy = 0; yy < bm.height; ++yy) {
                        int dst_y = ty + yy;
                        if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < bm.width; ++xx) {
                            int dst_x = tx + xx;
                            if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                            float sa = src[3] / 255.0f;
                            if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) {
                                for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                dst[3] = 255;
                            }
                        }
                    }
                }
            };
            if (left_label && left_label[0] != '\0') draw_label(bx_left, left_label);
            if (right_label && right_label[0] != '\0') draw_label(bx_right, right_label);
        };
        auto draw_edge_order_group = [&](int left_x, int byy, int value, bool tool_active) {
            int nbw = bw;
            int num_w = std::max(28, nbw * 2);
            int gap = 10;
            int tool_gap = 10;
            int bx_down = left_x;
            int bx_num = bx_down + nbw + gap;
            int bx_up = bx_num + num_w + gap;
            int bx_tool = bx_up + nbw + tool_gap;
            memset_rect(out_rgba, w, h, pitch, bx_down, byy, nbw, bh, Color{50,50,56,255});
            memset_rect(out_rgba, w, h, pitch, bx_num, byy, num_w, bh, Color{36,36,42,255});
            memset_rect(out_rgba, w, h, pitch, bx_up, byy, nbw, bh, Color{50,50,56,255});
            Color tool_fill = tool_active ? Color{90,80,70,255} : Color{60,54,48,255};
            memset_rect(out_rgba, w, h, pitch, bx_tool, byy, nbw, bh, tool_fill);
            auto draw_label = [&](int bx, const char* label, float scale, Color col) {
                auto bm = render_text_to_rgba(label, scale, {col.r, col.g, col.b, col.a});
                if (!bm.pixels.empty()) {
                    int tx = bx + (nbw - bm.width) / 2;
                    int ty = byy + (bh - bm.height) / 2;
                    for (int yy = 0; yy < bm.height; ++yy) {
                        int dst_y = ty + yy;
                        if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < bm.width; ++xx) {
                            int dst_x = tx + xx;
                            if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                            float sa = src[3] / 255.0f;
                            if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) {
                                for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                dst[3] = 255;
                            }
                        }
                    }
                }
            };
            draw_label(bx_down, LABEL_IO_MINUS, 1.0f, {230,230,235,255});
            draw_label(bx_up, LABEL_IO_PLUS, 1.0f, {230,230,235,255});
            draw_label(bx_tool, LABEL_EDGE_ORDER_TOOL, 0.8f, {235,228,218,255});
            std::string s = (value >= 0) ? ("+" + std::to_string(value)) : std::to_string(value);
            auto bm = render_text_to_rgba(s, 0.95f, {255,255,255,255});
            if (!bm.pixels.empty()) {
                int tx = bx_num + (num_w - bm.width) / 2;
                int ty = byy + (bh - bm.height) / 2;
                for (int yy = 0; yy < bm.height; ++yy) {
                    int dst_y = ty + yy;
                    if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < bm.width; ++xx) {
                        int dst_x = tx + xx;
                        if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
        };
        // compute left of table buttons start for groups placement
        int io_base_x = bx_r; // place IO groups to the left of the table buttons
        int io_by = by;
        int counter_value = std::max(1, ctx->io_attachment_count);
        int nbw = bw;
        int num_w = std::max(24, nbw * 2);
        int gap = 10;
        int counter_total_w = nbw + gap + num_w + gap + nbw;
        int action_gap = 16;
        int pair_gap = 10;
        int pair_total_w = nbw * 2 + pair_gap;
        int order_num_w = std::max(28, nbw * 2);
        int order_total_w = nbw + gap + order_num_w + gap + nbw + gap + nbw;
        int order_left_x = io_base_x - counter_total_w - action_gap - pair_total_w - action_gap - order_total_w;
        int pair_left_x = io_base_x - counter_total_w - action_gap - pair_total_w;
        draw_edge_order_group(order_left_x, io_by, ctx->edge_order_value, ctx->edge_order_tool_active != 0);
        draw_io_action_pair(pair_left_x, io_by, LABEL_IO_CONSUMER_SHORT, LABEL_IO_PRODUCER_SHORT);
        draw_io_group(io_base_x, io_by, counter_value, LABEL_IO_COUNT_SHORT);
    }

    // Prepare storage for per-module table hitboxes discovered during table rendering.
    std::vector<std::vector<GP_TableHitBox>> module_hitboxes(ctx->modules.size());
    std::vector<std::unordered_map<int, ContactLight>> module_contact_lights(ctx->modules.size());
    std::vector<std::vector<uint8_t>> module_tables(ctx->modules.size());

    // First pass: render tables and gather hitboxes/light info (no drawing yet).
    for (int mi = 0; mi < static_cast<int>(ctx->modules.size()); ++mi) {
        const auto &m = ctx->modules[mi];
        module_hitboxes[mi].clear();
        module_contact_lights[mi].clear();
        module_tables[mi].clear();
        if (mi < 0 || mi >= static_cast<int>(ctx->module_tables.size()) || !ctx->module_tables[mi]) continue;
        GP_TableContext* t = ctx->module_tables[mi];
        // Ensure per-module table layout is up-to-date before rendering it.
        sync_module_table_io_layout(ctx, mi);
        bool is_stage = (mi >= 0 && mi < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[mi]);
        int tw = std::max(1, m.w);
        int th = std::max(1, m.h);
        module_tables[mi].resize(static_cast<size_t>(tw) * th * 4);
        // Initialize the full module-sized buffer so blitting doesn't punch transparent holes
        // when the table's computed height is smaller than the module height.
        if (mi < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[mi]) {
            std::fill(module_tables[mi].begin(), module_tables[mi].end(), 0);
        } else {
            for (size_t i = 0; i + 3 < module_tables[mi].size(); i += 4) {
                module_tables[mi][i + 0] = 40;
                module_tables[mi][i + 1] = 40;
                module_tables[mi][i + 2] = 50;
                module_tables[mi][i + 3] = 255;
            }
        }
        GP_TableGeom geom{};
        // Provide the module-sized target geometry so the table raster
        // can compute row layout (image rows rely on the target height).
        geom.width_px = tw;
        geom.height_px = th;
        const int hitcap = 4096;
        std::vector<GP_TableHitBox> hits(hitcap);
        int hits_written = 0;
        // For stage modules we want the stage image to act as the full
        // module background. Render the table into a temporary buffer and
        // composite only its non-transparent pixels over the stage base so
        // transparent gutters preserve the stage imagery.
        if (mi < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[mi] && !ctx->module_stage_images[mi].rgba) {
            // no stage image available: fallback to regular render
            gp_table_render_rgba_with_state(t, nullptr, module_tables[mi].data(), static_cast<int32_t>(module_tables[mi].size()), &geom, hits.data(), hitcap, &hits_written);
        } else if (mi < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[mi]) {
            // Prefill module buffer with stage pixels if available.
            if (!ctx->module_stage_images[mi].rgba) {
                std::fill(module_tables[mi].begin(), module_tables[mi].end(), 0);
            } else {
                const uint8_t* src_stage = ctx->module_stage_images[mi].rgba;
                size_t bytes = static_cast<size_t>(tw) * static_cast<size_t>(th) * 4u;
                std::memcpy(module_tables[mi].data(), src_stage, bytes);
            }
            // render table into temp and composite
            std::vector<uint8_t> tmp_buf(module_tables[mi].size());
            int ok = gp_table_render_rgba_with_state(t, nullptr, tmp_buf.data(), static_cast<int32_t>(tmp_buf.size()), &geom, hits.data(), hitcap, &hits_written);
            if (ok) {
                size_t pixels = static_cast<size_t>(tw) * static_cast<size_t>(th);
                uint8_t* dst = module_tables[mi].data();
                const uint8_t* src = tmp_buf.data();
                for (size_t pi = 0; pi < pixels; ++pi) {
                    const uint8_t sa = src[pi * 4 + 3];
                    if (sa == 0) continue; // leave base (stage) pixel
                    if (sa >= 255) {
                        dst[pi * 4 + 0] = src[pi * 4 + 0];
                        dst[pi * 4 + 1] = src[pi * 4 + 1];
                        dst[pi * 4 + 2] = src[pi * 4 + 2];
                        dst[pi * 4 + 3] = 255;
                    } else {
                        float a = sa / 255.0f;
                        for (int cc = 0; cc < 3; ++cc) {
                            dst[pi * 4 + cc] = static_cast<uint8_t>(std::lround(src[pi * 4 + cc] * a + dst[pi * 4 + cc] * (1.0f - a)));
                        }
                        dst[pi * 4 + 3] = 255;
                    }
                }
            }
        } else {
            gp_table_render_rgba_with_state(t, nullptr, module_tables[mi].data(), static_cast<int32_t>(module_tables[mi].size()), &geom, hits.data(), hitcap, &hits_written);
        }
        if (hits_written > 0) {
            module_hitboxes[mi].assign(hits.begin(), hits.begin() + hits_written);
        }
        auto &light_map = module_contact_lights[mi];
        light_map.clear();
        GP_TableStyle st{};
        gp_table_get_style(t, &st);
        Color led_on{st.led_on_rgba[0], st.led_on_rgba[1], st.led_on_rgba[2], st.led_on_rgba[3]};
        int in_count = (mi >= 0 && mi < static_cast<int>(ctx->module_io_in_count.size())) ? ctx->module_io_in_count[mi] : 0;
        int out_count = (mi >= 0 && mi < static_cast<int>(ctx->module_io_out_count.size())) ? ctx->module_io_out_count[mi] : 0;
        for (int hi = 0; hi < hits_written; ++hi) {
            const auto &hb = hits[hi];
            if (hb.part != GP_TABLE_HIT_LED && hb.part != GP_TABLE_HIT_LED_ARG && hb.part != GP_TABLE_HIT_LED_TABLE) continue;
            int contact_idx = resolve_contact_index(ctx, mi, hb);
            if (contact_idx < 0) continue;
            int on = 0, active = 0;
            if (!gp_table_get_led_info(t, hb.row_idx, hb.col_idx, hb.aux0, &on, &active, nullptr, nullptr)) continue;
            float glow = 0.0f;
            gp_table_get_led_glow(t, hb.row_idx, hb.col_idx, hb.aux0, &glow);
            bool lit = (on != 0) || (active != 0);
            bool is_input = (contact_idx >= 0 && contact_idx < in_count);
            bool is_output = (contact_idx >= in_count && contact_idx >= 0 && contact_idx < (in_count + out_count));
            if (glow <= 0.0f && is_output && lit) glow = 0.35f;
            if (!lit) glow *= 0.3f;
            if (is_input) glow *= 0.35f;
            if (is_output && lit) glow = std::min(1.0f, glow * 1.25f + 0.15f);
            if (glow > 0.0f) {
                ContactLight cl;
                cl.col = led_on;
                cl.intensity = glow;
                cl.valid = true;
                light_map[contact_idx] = cl;
            }
        }
    }

    // ensure root rope sim exists
    RopeSim* sim = canvas_require_root_sim(ctx);

    std::vector<ContactLight> edge_light_a(ctx->edges.size());
    std::vector<ContactLight> edge_light_b(ctx->edges.size());
    const float rope_decay = 0.35f;
    const float rope_end_gain = std::exp(-rope_decay);
    auto is_input_contact = [&](int module_idx, int contact_idx) -> bool {
        if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_io_in_count.size())) {
            int in_count = ctx->module_io_in_count[module_idx];
            return (contact_idx >= 0 && contact_idx < in_count);
        }
        return false;
    };

    // update/create ropes for edges
    for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
        const auto &edge = ctx->edges[ei].desc;
        if (edge.a_module < 0 || edge.a_module >= static_cast<int>(ctx->modules.size())) continue;
        if (edge.b_module < 0 || edge.b_module >= static_cast<int>(ctx->modules.size())) continue;
        int ax, ay, bx, by;
        // compute contact positions. If the endpoint module has an attached table
        // and we captured hitboxes during rendering, prefer the table-provided
        // hitbox center for exact LED anchor coordinates. Fall back to legacy
        // computed positions otherwise.
        const GP_CanvasModuleDesc &ma = ctx->modules[edge.a_module];
        const GP_CanvasModuleDesc &mb = ctx->modules[edge.b_module];
        bool resolvedA = false, resolvedB = false;
        if (edge.a_module < static_cast<int>(module_hitboxes.size()) && !module_hitboxes[edge.a_module].empty()) {
            for (const auto &hb : module_hitboxes[edge.a_module]) {
                if ((hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG || hb.part == GP_TABLE_HIT_LED_TABLE) && resolve_contact_index(ctx, edge.a_module, hb) == edge.a_contact_idx) {
                    int local_x = (hb.x0 + hb.x1) / 2;
                    int local_y = (hb.y0 + hb.y1) / 2;
                    ax = ctx->modules[edge.a_module].x + local_x;
                    ay = ctx->modules[edge.a_module].y + local_y;
                    if (edge.a_module < static_cast<int>(module_contact_lights.size())) {
                        auto &map = module_contact_lights[edge.a_module];
                        auto it = map.find(edge.a_contact_idx);
                        if (it != map.end()) edge_light_a[ei] = it->second;
                    }
                    resolvedA = true; break;
                }
            }
        }
        if (edge.b_module < static_cast<int>(module_hitboxes.size()) && !module_hitboxes[edge.b_module].empty()) {
            for (const auto &hb : module_hitboxes[edge.b_module]) {
                if ((hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG || hb.part == GP_TABLE_HIT_LED_TABLE) && resolve_contact_index(ctx, edge.b_module, hb) == edge.b_contact_idx) {
                    int local_x = (hb.x0 + hb.x1) / 2;
                    int local_y = (hb.y0 + hb.y1) / 2;
                    bx = ctx->modules[edge.b_module].x + local_x;
                    by = ctx->modules[edge.b_module].y + local_y;
                    if (edge.b_module < static_cast<int>(module_contact_lights.size())) {
                        auto &map = module_contact_lights[edge.b_module];
                        auto it = map.find(edge.b_contact_idx);
                        if (it != map.end()) edge_light_b[ei] = it->second;
                    }
                    resolvedB = true; break;
                }
            }
        }
        if (resolvedA && resolvedB) {
            bool a_input = is_input_contact(edge.a_module, edge.a_contact_idx);
            bool b_input = is_input_contact(edge.b_module, edge.b_contact_idx);
            if (!a_input && b_input) {
                float src = edge_light_a[ei].valid ? edge_light_a[ei].intensity : 0.0f;
                float delivered = src * rope_end_gain;
                if (delivered > 0.0f) {
                    if (!edge_light_b[ei].valid) {
                        edge_light_b[ei].col = edge_light_a[ei].col;
                        edge_light_b[ei].valid = true;
                        edge_light_b[ei].intensity = delivered;
                    } else {
                        edge_light_b[ei].intensity = std::max(edge_light_b[ei].intensity, delivered);
                    }
                }
            }
        }
        // If either endpoint couldn't be resolved from table hitboxes, skip
        // rope creation for this edge - legacy side-based geometry removed.
        if (!resolvedA || !resolvedB) {
            ctx->edges[ei].rope_idx = -1;
            continue;
        }
        int ridx = ctx->edges[ei].rope_idx;
        if (ridx < 0) {
            int segs = ctx->sim_segs;
            float slack = ctx->sim_slack;
            int newr = sim ? rope_sim_add_rope(sim, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by), segs, slack) : -1;
            ctx->edges[ei].rope_idx = newr;
        } else {
            if (sim) rope_sim_move_endpoints(sim, ridx, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by));
        }
    }

    // step sim
    if (sim) rope_sim_step(sim, 1.0f/60.0f, ctx->sim_maxforce, ctx->sim_iters, ctx->sim_damping);

    // Second pass: draw module backgrounds (raytrace/callback) and blit tables.
    for (int mi = 0; mi < static_cast<int>(ctx->modules.size()); ++mi) {
        const auto &m = ctx->modules[mi];
        int sx = m.x - ctx->offset_x;
        int sy = m.y - ctx->offset_y;
        std::vector<InputRayLight> inputs;
        bool is_stage = (mi >= 0 && mi < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[mi]);
        bool bg_done = false;
        if (mi >= 0 && mi < static_cast<int>(ctx->module_bg.size())) {
            auto &bg = ctx->module_bg[mi];
            if (bg.cb) {
                ensure_module_bg_storage(bg, m.w, m.h, /*oversample=*/1);
                bg.cb(bg.user, mi, m.w, m.h, bg.scratch.data(), m.w * 4);
                if (!is_stage) {
                    blit_module_buffer(out_rgba, w, h, pitch, sx, sy, m.w, m.h, bg.scratch);
                    bg_done = true;
                } else {
                    // Stage modules: embed the stage scratch buffer inside the table's content
                    // image cell and re-render the table now that the stage buffer is current.
                    if (mi >= static_cast<int>(ctx->module_stage_images.size())) ctx->module_stage_images.resize(mi + 1);
                    GP_TableImage img{};
                    if (!bg.scratch.empty()) {
                        img.rgba = bg.scratch.data();
                        img.width_px = m.w;
                        img.height_px = m.h;
                        img.pitch_bytes = m.w * 4;
                    }
                    ctx->module_stage_images[mi] = img;
                    if (mi >= 0 && mi < static_cast<int>(ctx->module_tables.size()) && ctx->module_tables[mi]) {
                        GP_TableContext* t = ctx->module_tables[mi];
                        int rc = gp_table_get_row_count(t);
                        if (rc > 0) {
                            std::vector<GP_TableRow> rows(static_cast<size_t>(rc));
                            for (int ri = 0; ri < rc; ++ri) gp_table_get_row(t, ri, &rows[static_cast<size_t>(ri)]);
                            bool changed = false;
                            for (auto &r : rows) {
                                for (int ci = 0; ci < r.cell_count && ci < 8; ++ci) {
                                    if (r.cells[ci].kind == GP_TABLE_CELL_IMAGE) {
                                        r.cells[ci].image = img.rgba ? &ctx->module_stage_images[mi] : nullptr;
                                        // Allow the table to allocate the full module height for
                                        // the image content by setting the cell's reserved0
                                        // to the module height. Also set the row's
                                        // reserved0 to the module height so the row
                                        // requests the full module area (header + image)
                                        // instead of collapsing to the header band.
                                        if (img.rgba) {
                                            r.cells[ci].reserved0 = m.h;
                                            r.reserved0 = m.h;
                                        }
                                        changed = true;
                                    }
                                }
                            }
                            if (changed) gp_table_set_rows(t, rows.data(), rc);
                        }
                        if (mi >= 0 && mi < static_cast<int>(module_tables.size()) && !module_tables[mi].empty()) {
                            GP_TableGeom geom{};
                            geom.width_px = std::max(1, m.w);
                            geom.height_px = std::max(1, m.h);
                            // Debug: report computed table geom and row/cell reserved values
                            GP_TableGeom gtmp{};
                            if (gp_table_get_geom(t, &gtmp)) {
                                printf("DEBUG: before render: table rows=%d ctx_geom=%dx%d target_geom=%dx%d\n", rc, gtmp.width_px, gtmp.height_px, geom.width_px, geom.height_px);
                            } else {
                                printf("DEBUG: before render: table rows=%d target_geom=%dx%d (no ctx geom)\n", rc, geom.width_px, geom.height_px);
                            }
                            for (int ri = 0; ri < rc; ++ri) {
                                GP_TableRow r{};
                                gp_table_get_row(t, ri, &r);
                                printf("DEBUG: row %d kind=%d expanded=%d reserved0=%d cell_count=%d\n", ri, r.kind, r.expanded, r.reserved0, r.cell_count);
                                for (int ci = 0; ci < r.cell_count && ci < 8; ++ci) {
                                    const GP_TableCell &c = r.cells[ci];
                                    if (c.kind == GP_TABLE_CELL_IMAGE) printf("DEBUG:  cell %d is IMAGE reserved0=%d image=%p\n", ci, c.reserved0, (void*)c.image);
                                }
                            }

                            // Render table into a temp buffer and composite over the
                            // existing module buffer so the stage background (which
                            // lives in module_tables[mi]) remains visible where the
                            // table is transparent.
                            std::vector<uint8_t> tmp_buf(module_tables[mi].size());
                            int ok = gp_table_render_rgba_with_state(t, nullptr, tmp_buf.data(), static_cast<int32_t>(tmp_buf.size()), &geom, nullptr, 0, nullptr);
                            if (ok) {
                                int tw_local = std::max(1, m.w);
                                int th_local = std::max(1, m.h);
                                size_t count_nonzero = 0;
                                size_t pixels = static_cast<size_t>(tw_local) * static_cast<size_t>(th_local);
                                uint8_t* dst = module_tables[mi].data();
                                const uint8_t* src = tmp_buf.data();
                                for (size_t pi = 0; pi < pixels; ++pi) {
                                    const uint8_t sa = src[pi * 4 + 3];
                                    if (sa == 0) {
                                        // nothing drawn here; keep existing base
                                        if (dst[pi * 4 + 3] != 0) ++count_nonzero;
                                        continue;
                                    }
                                    if (sa >= 255) {
                                        dst[pi * 4 + 0] = src[pi * 4 + 0];
                                        dst[pi * 4 + 1] = src[pi * 4 + 1];
                                        dst[pi * 4 + 2] = src[pi * 4 + 2];
                                        dst[pi * 4 + 3] = 255;
                                        ++count_nonzero;
                                    } else {
                                        float a = sa / 255.0f;
                                        for (int cc = 0; cc < 3; ++cc) {
                                            dst[pi * 4 + cc] = static_cast<uint8_t>(std::lround(src[pi * 4 + cc] * a + dst[pi * 4 + cc] * (1.0f - a)));
                                        }
                                        dst[pi * 4 + 3] = 255;
                                        ++count_nonzero;
                                    }
                                }
                                printf("DEBUG: after render: nonzero_alpha_pixels=%zu of %zu\n", count_nonzero, pixels);
                            }
                        }
                    }
                }
            } else if (bg.mode == 1) {
                if (!is_stage) {
                    render_module_raytrace_bg(bg, m.w, m.h, inputs);
                    blit_module_buffer(out_rgba, w, h, pitch, sx, sy, m.w, m.h, bg.scratch);
                    bg_done = true;
                }
            }
        }
        if (!bg_done) {
            Color bgc{40,40,50,255};
            memset_rect(out_rgba, w, h, pitch, sx, sy, m.w, m.h, bgc);
        }
        if (mi >= 0 && mi < static_cast<int>(module_tables.size()) && !module_tables[mi].empty()) {
            uint8_t table_alpha = 255;
            if (mi >= 0 && mi < static_cast<int>(ctx->module_bg.size())) {
                const auto &bg = ctx->module_bg[mi];
                table_alpha = bg.table_alpha;
                if (bg.mode == 1) {
                    table_alpha = std::min(table_alpha, bg.table_alpha_ray);
                }
            }
            if (is_stage) {
                // Stage imagery is embedded into the table buffer; blend using the table's per-pixel alpha.
                blit_module_buffer_srcalpha(out_rgba, w, h, pitch, sx, sy, m.w, m.h, module_tables[mi]);
            } else if (table_alpha >= 255) {
                blit_module_buffer(out_rgba, w, h, pitch, sx, sy, m.w, m.h, module_tables[mi]);
            } else {
                blit_module_buffer_alpha(out_rgba, w, h, pitch, sx, sy, m.w, m.h, module_tables[mi], table_alpha);
            }
        }
        if (mi == ctx->focused_module) {
            Color fb{60,120,220,255};
            int t = 2;
            for (int dy = 1; dy <= t; ++dy) {
                int ytop = sy - dy;
                int ybot = sy + m.h - 1 + dy;
                if (ytop >= 0 && ytop < h) memset_rect(out_rgba, w, h, pitch, std::max(0, sx - dy), ytop, std::min(w, m.w + 2*dy), 1, fb);
                if (ybot >= 0 && ybot < h) memset_rect(out_rgba, w, h, pitch, std::max(0, sx - dy), ybot, std::min(w, m.w + 2*dy), 1, fb);
            }
            for (int dx = 1; dx <= t; ++dx) {
                int lx = sx - dx;
                int rx = sx + m.w - 1 + dx;
                if (lx >= 0 && lx < w) memset_rect(out_rgba, w, h, pitch, lx, std::max(0, sy - t), 1, std::min(h, m.h + 2*t), fb);
                if (rx >= 0 && rx < w) memset_rect(out_rgba, w, h, pitch, rx, std::max(0, sy - t), 1, std::min(h, m.h + 2*t), fb);
            }
        }
    }

    // Prefer manager-supplied immutable network snapshot for rendering the connection graph.
    // The snapshot contains only endpoint keys and edges (no UI data). Map endpoint keys
    // to module centers for visual placement (canvas module positions are UI-owned).
    GP_TableContext* root_table = canvas_ensure_root_table(ctx);
    std::shared_ptr<ThreadManager::NetworkSnapshot> snap;
    ThreadManager* tm = ThreadManager::global();
    if (tm && root_table) snap = tm->get_table_snapshot(root_table);
    if (snap && !snap->edges.empty()) {
        // Draw simple straight lines between module centers derived from endpoint keys.
        for (size_t ei = 0; ei < snap->edges.size(); ++ei) {
            const auto &se = snap->edges[ei];
            // Resolve endpoints using the canonical indices from the snapshot.
            if (se.a_idx >= snap->nodes.size() || se.b_idx >= snap->nodes.size()) continue;
            uint64_t a_key = snap->nodes[se.a_idx];
            uint64_t b_key = snap->nodes[se.b_idx];
            int a_module = static_cast<int>((a_key >> 32) & 0xFFFFFFFFu);
            int b_module = static_cast<int>((b_key >> 32) & 0xFFFFFFFFu);
            if (a_module < 0 || a_module >= static_cast<int>(ctx->modules.size())) continue;
            if (b_module < 0 || b_module >= static_cast<int>(ctx->modules.size())) continue;
            int ax = ctx->modules[static_cast<size_t>(a_module)].x + ctx->modules[static_cast<size_t>(a_module)].w / 2 - ctx->offset_x;
            int ay = ctx->modules[static_cast<size_t>(a_module)].y + ctx->modules[static_cast<size_t>(a_module)].h / 2 - ctx->offset_y;
            int bx = ctx->modules[static_cast<size_t>(b_module)].x + ctx->modules[static_cast<size_t>(b_module)].w / 2 - ctx->offset_x;
            int by = ctx->modules[static_cast<size_t>(b_module)].y + ctx->modules[static_cast<size_t>(b_module)].h / 2 - ctx->offset_y;
            // draw a simple line (Bresenham-ish) with light gray color
            Color col{200,200,200,180};
            int dx = std::abs(bx - ax), sx = ax < bx ? 1 : -1;
            int dy = -std::abs(by - ay), sy = ay < by ? 1 : -1;
            int err = dx + dy;
            int x0 = ax, y0 = ay;
            while (true) {
                if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h) {
                    uint8_t* px = out_rgba + (y0 * pitch) + (x0 * 4);
                    blend_pixel(px, col.r, col.g, col.b, col.a);
                }
                if (x0 == bx && y0 == by) break;
                int e2 = 2 * err;
                if (e2 >= dy) { err += dy; x0 += sx; }
                if (e2 <= dx) { err += dx; y0 += sy; }
            }
        }
    } else {
        // Fallback: legacy per-edge rope rendering using canvas-local rope sim/indices
        for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
            int ridx = ctx->edges[ei].rope_idx;
            if (ridx < 0) continue;
            int vc = sim ? rope_sim_get_vertex_count(sim, ridx) : 0;
            if (vc < 2) continue;
            std::vector<float> verts(static_cast<size_t>(vc) * 2);
            int got = sim ? rope_sim_get_vertices(sim, ridx, verts.data(), static_cast<int>(verts.size())) : 0;
            if (got <= 0) continue;
            std::vector<float> verts_view(static_cast<size_t>(got) * 2);
            for (int vi = 0; vi < got; ++vi) {
                verts_view[vi * 2 + 0] = verts[vi * 2 + 0] - static_cast<float>(ctx->offset_x);
                verts_view[vi * 2 + 1] = verts[vi * 2 + 1] - static_cast<float>(ctx->offset_y);
            }
            int jacket_px = ctx->jacket_px;
            int jacket_border = ctx->jacket_border;
            table_draw_rope_curve_blend(out_rgba, w, h, pitch, verts_view.data(), got, jacket_px, jacket_border, 200, 200, 200, 180, 3);
            int glow_r = std::max(2, jacket_px * 2);
            draw_rope_light_falloff(out_rgba, w, h, pitch, verts_view.data(), got, edge_light_a[ei], edge_light_b[ei], rope_decay, glow_r);
        }
    }

    // render provisional prospective rope (follows mouse) if present
    if (ctx->prospective_rope_idx >= 0) {
        int ridx = ctx->prospective_rope_idx;
        int vc = sim ? rope_sim_get_vertex_count(sim, ridx) : 0;
        if (vc >= 2) {
            std::vector<float> verts(static_cast<size_t>(vc) * 2);
            int got = sim ? rope_sim_get_vertices(sim, ridx, verts.data(), static_cast<int>(verts.size())) : 0;
            if (got > 0) {
                std::vector<float> verts_view(static_cast<size_t>(got) * 2);
                for (int vi = 0; vi < got; ++vi) {
                    verts_view[vi * 2 + 0] = verts[vi * 2 + 0] - static_cast<float>(ctx->offset_x);
                    verts_view[vi * 2 + 1] = verts[vi * 2 + 1] - static_cast<float>(ctx->offset_y);
                }
                int jacket_px = ctx->jacket_px;
                int jacket_border = ctx->jacket_border;
                const float* hues_ptr = ctx->hues.empty() ? nullptr : ctx->hues.data();
                int hue_count = static_cast<int>(ctx->hues.size());
                float hue_intensity = ctx->hue_intensity;
                int samples_per_segment = 3;
                (void)hues_ptr;
                (void)hue_count;
                (void)hue_intensity;
                table_draw_rope_curve_blend(out_rgba, w, h, pitch, verts_view.data(), got, jacket_px, jacket_border, 200, 200, 200, 180, samples_per_segment);
            }
        }
    }

    if (ctx->tool_menu_open) {
        ToolMenuLayout layout = compute_tool_menu_layout(ctx);
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y, layout.w, layout.h, Color{32,32,40,230});
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y, layout.w, 1, Color{90,90,110,255});
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y + layout.h - 1, layout.w, 1, Color{10,10,14,255});
        auto blit_text = [&](const char* text, int tx, int ty, float scale, Color col) {
            if (!text || text[0] == '\0') return;
            auto bm = render_text_to_rgba(text, scale, {col.r, col.g, col.b, col.a});
            if (bm.pixels.empty()) return;
            for (int yy = 0; yy < bm.height; ++yy) {
                int dst_y = ty + yy;
                if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < bm.width; ++xx) {
                    int dst_x = tx + xx;
                    if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        };
        int title_x = layout.x + 8;
        int title_y = layout.y + 6;
        blit_text(LABEL_MENU_TITLE, title_x, title_y, 0.95f, Color{220,220,230,255});
        int item_y = layout.item_start_y;
        for (size_t i = 0; i < std::size(kToolMenuItems); ++i) {
            int y0 = item_y + static_cast<int>(i) * layout.row_h;
            memset_rect(out_rgba, w, h, pitch, layout.x + 4, y0, layout.w - 8, layout.row_h - 2, Color{45,45,58,240});
            blit_text(kToolMenuItems[i].label, layout.x + 12, y0 + 3, 0.9f, Color{240,240,240,255});
        }
        if (layout.stack_count > 0) {
            int stack_title_y = layout.stack_start_y - layout.header_h + 2;
            blit_text(LABEL_MENU_STACK_TITLE, layout.x + 8, stack_title_y, 0.9f, Color{200,200,210,255});
            int stack_y = layout.stack_start_y;
            if (ctx->focused_module >= 0 && ctx->focused_module < static_cast<int>(ctx->module_io_rows.size())) {
                const auto &stack = ctx->module_io_rows[ctx->focused_module];
                for (size_t i = 0; i < stack.size(); ++i) {
                    int y0 = stack_y + static_cast<int>(i) * layout.row_h;
                    memset_rect(out_rgba, w, h, pitch, layout.x + 4, y0, layout.w - 8, layout.row_h - 2, Color{36,36,48,240});
                    char label[96];
                    label[0] = '\0';
                    const auto &row = stack[i];
                    if (row.kind == ModuleRowKind::Input) {
                        if (row.attachment_count > 1) {
                            std::snprintf(label, sizeof(label), "%s x%d", LABEL_IO_CONSUMER, row.attachment_count);
                        } else {
                            std::snprintf(label, sizeof(label), "%s", LABEL_IO_CONSUMER);
                        }
                    } else if (row.kind == ModuleRowKind::Output) {
                        if (row.attachment_count > 1) {
                            std::snprintf(label, sizeof(label), "%s x%d", LABEL_IO_PRODUCER, row.attachment_count);
                        } else {
                            std::snprintf(label, sizeof(label), "%s", LABEL_IO_PRODUCER);
                        }
                    } else {
                        std::snprintf(label, sizeof(label), "%s", tool_label(row.tool));
                    }
                    blit_text(label, layout.x + 12, y0 + 3, 0.85f, Color{210,210,220,255});
                }
            }
        }
    }

    return 1;
}
