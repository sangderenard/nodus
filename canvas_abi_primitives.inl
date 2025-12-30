// Deferred console logging
#include "console_logger.h"
#ifndef printf
#define printf(...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
#ifndef fprintf
#define fprintf(file, ...) CONSOLE_PRINTF(__VA_ARGS__)
#endif

// C-linkage prototypes for spline drawer implemented in table_abi.cpp
extern "C" void table_draw_rope_curve_blend_colored(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, const float* hues, int hue_count, int samples_per_segment, float hue_intensity);
extern "C" void table_draw_rope_curve_blend(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, uint8_t cr, uint8_t cg, uint8_t cb, uint8_t ca, int samples_per_segment);
extern "C" void table_draw_rope_polyline(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int radius, uint8_t cr, uint8_t cg, uint8_t cb, uint8_t ca);

static const char* kCanvasDefaultWorkspacePath = "canvas_workspace.txt";

// local minimal Color and draw helpers (self-contained)
struct Color { uint8_t r=0,g=0,b=0,a=255; };

// Contact lighting descriptor used when computing per-contact glow
struct ContactLight { Color col; float intensity = 0.0f; bool valid = false; };

static inline uint32_t pack_rgba(Color c) {
    return (static_cast<uint32_t>(c.r) << 24) |
           (static_cast<uint32_t>(c.g) << 16) |
           (static_cast<uint32_t>(c.b) << 8)  |
           static_cast<uint32_t>(c.a);
}

static inline Color unpack_rgba(uint32_t v) {
    Color c;
    c.r = static_cast<uint8_t>((v >> 24) & 0xFFu);
    c.g = static_cast<uint8_t>((v >> 16) & 0xFFu);
    c.b = static_cast<uint8_t>((v >> 8) & 0xFFu);
    c.a = static_cast<uint8_t>(v & 0xFFu);
    return c;
}

// Convert array of 4 floats [r,g,b,a] (0..1) to Color
static inline Color rgba_from_floats(const float* f) {
    Color c{0,0,0,255};
    if (!f) return c;
    auto to = [](float v)->uint8_t { return static_cast<uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
    c.r = to(f[0]); c.g = to(f[1]); c.b = to(f[2]); c.a = to(f[3]);
    return c;
}

// Blend src color (with alpha 0..255) over destination pixel in-place
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
    if (outa <= 0.0f) {
        dst[0] = dst[1] = dst[2] = dst[3] = 0;
        return;
    }
    float out_r = (srf * a + dr * da * inv) / outa;
    float out_g = (sgf * a + dg * da * inv) / outa;
    float out_b = (sbf * a + db * da * inv) / outa;
    dst[0] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_r)) * 255.0f));
    dst[1] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_g)) * 255.0f));
    dst[2] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_b)) * 255.0f));
    dst[3] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, outa)) * 255.0f));
}

// Simple rectangle fill helper (canvas-local copy). Kept static to avoid
// exposing linkage; mirrors the behavior used in table rendering.
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
            row[0] = c.r; row[1] = c.g; row[2] = c.b; row[3] = c.a;
            row += 4;
        }
    }
}

static constexpr int kSubgroupBinCount = 8;
// High bit flag reserved for pointer-mode handshake (outside palette bits)
static constexpr uint32_t kEdgeFlagPointerMode = 0x80000000u;
static constexpr float kColorWheelPi = 3.14159265358979323846f;

static inline Color hsv_to_color(float h, float s, float v, uint8_t a=255) {
    h = std::fmod(h, 1.0f);
    if (h < 0.0f) h += 1.0f;
    s = std::clamp(s, 0.0f, 1.0f);
    v = std::clamp(v, 0.0f, 1.0f);
    float c = v * s;
    float hp = h * 6.0f;
    float x = c * (1.0f - std::fabs(std::fmod(hp, 2.0f) - 1.0f));
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    if (hp < 1.0f) { r = c; g = x; }
    else if (hp < 2.0f) { r = x; g = c; }
    else if (hp < 3.0f) { g = c; b = x; }
    else if (hp < 4.0f) { g = x; b = c; }
    else if (hp < 5.0f) { r = x; b = c; }
    else { r = c; b = x; }
    float m = v - c;
    return Color{
        static_cast<uint8_t>(std::lround((r + m) * 255.0f)),
        static_cast<uint8_t>(std::lround((g + m) * 255.0f)),
        static_cast<uint8_t>(std::lround((b + m) * 255.0f)),
        a
    };
}

static inline float rgb_to_hue(const Color &c) {
    const float r = c.r / 255.0f;
    const float g = c.g / 255.0f;
    const float b = c.b / 255.0f;
    const float maxc = std::max({r, g, b});
    const float minc = std::min({r, g, b});
    const float delta = maxc - minc;
    if (delta <= 1e-6f) return 0.0f;
    float hue;
    if (maxc == r) {
        hue = std::fmod((g - b) / delta, 6.0f);
    } else if (maxc == g) {
        hue = ((b - r) / delta) + 2.0f;
    } else {
        hue = ((r - g) / delta) + 4.0f;
    }
    hue /= 6.0f;
    if (hue < 0.0f) hue += 1.0f;
    return hue;
}

static inline uint32_t subgroup_mask_for_index(int idx) {
    if (idx < 0 || idx >= kSubgroupBinCount) return 0u;
    return 1u << static_cast<uint32_t>(idx);
}

static inline uint32_t subgroup_palette_index(uint32_t flags) {
    uint32_t mask = (kSubgroupBinCount >= 32) ? 0xFFFFFFFFu : ((1u << kSubgroupBinCount) - 1u);
    return flags & mask;
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

        float spacing = std::max(1.0f, float(radius) * 0.6f); // spacing between samples
        std::vector<std::pair<float,float>> samples{};
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
    // overlay drag support
    int overlay_id = -1;
    int overlay_offx = 0; // offset from overlay left/top when drag started
    int overlay_offy = 0;
};

struct RopeIdEntry {
    GP_TableContext* table = nullptr;
    uint64_t rope_id = 0ull;
};

struct GP_CanvasContextImpl;

// Snapshot representation for persisted canvas-level meta-groups.
struct CanvasMetaVert {
    uint64_t rope_id = 0ull;
    int vertex_idx = 0;
};

struct CanvasMetaSnapshot {
    int module_idx = -1;
    int meta_slot = -1;
    std::vector<CanvasMetaVert> verts;
    int channel_group = 0;
    uint64_t anchor_rope_id = 0ull;
    int anchor_v = -1;
    unsigned int subgroup_flags = 0u;
    int ring_mode = 0;
    float ring_u = 0.0f;
    unsigned long long overlay_a = 0ull;
    unsigned long long overlay_b = 0ull;
    float overlay_x1 = 0.0f;
    float overlay_y1 = 0.0f;
    float overlay_x2 = 0.0f;
    float overlay_y2 = 0.0f;
    float confinement = 0.0f;
    unsigned long long mgid = 0ull;
    float dangling_len = 0.0f;
    unsigned int lasso_flags = 0u;
    int32_t lasso_widget_type = 0;
};
extern "C" int gp_canvas_add_edge_with_type(GP_CanvasContext* ctx_, const GP_CanvasEdgeDesc* desc, int type_id);

// Lightweight chat color struct used by canvas chat visuals
struct ChatCol { uint8_t r=40, g=40, b=40, a=255; };

// Forward-declare chat bg callback so it can be assigned earlier in the file
static void chat_bg_callback(void* user, int module_idx, int width, int height, uint8_t* out_rgba, int32_t out_pitch);

// Simple bounds structure representing the extents of modules on the canvas.
// Used to compute scroll/clamp state for the viewport.
struct CanvasBounds {
    int min_x = 0;
    int max_x = 0;
    int min_y = 0;
    int max_y = 0;
    bool has_any = false;
};

// NOTE: legacy per-side contact geometry removed. Contact hit/position
// information is authoritative from attached `GP_TableContext` hitboxes.

constexpr int kModuleDefaultWidth = 420;
constexpr int kModuleDefaultHeight = 620;
constexpr int kModulePreviewMinTableHeight = 120;
constexpr int kModuleLedPerSide = 2;
constexpr int kModuleTopPadding = 6;
constexpr int kModuleTopGap = 4;
constexpr int kModuleTitleRowH = 24;
constexpr int kModuleThumbRowH = 16;
constexpr int kModuleControlRowH = 24;
constexpr int kModuleLedRowH = 20;
const int kModuleExtraLedCount = 16;
const int kModuleExtraLedRows = 4;
const int kModuleFrameContactBase = 10000;
constexpr int kModuleFrameRowSend = -100;
constexpr int kModuleFrameRowReceive = -101;
constexpr int kModuleTopUiHeight =
    kModuleTopPadding * 2 +
    kModuleTitleRowH +
    kModuleThumbRowH +
    kModuleControlRowH +
    kModuleLedRowH * kModuleExtraLedRows +
    kModuleTopGap * (3 + (kModuleExtraLedRows - 1));
constexpr int kModuleColLeftLed = 0;
constexpr int kModuleColText = 1;
constexpr int kModuleColRightLed = 2;
constexpr int kModuleColCount = 3;

struct ModuleFrameLedGroup {
    std::array<std::array<GP_TableCell, kModuleExtraLedCount>, kModuleExtraLedRows> cells{};
};

struct ModuleFrameLink {
    // per-logical-row pointers: rows are arranged as [send-left, send-right, receive-left, receive-right]
    std::array<std::array<void*, kModuleExtraLedCount>, kModuleExtraLedRows> ptrs{};
    // stable per-port UUIDs parallel to `ptrs` for deterministic binding
    std::array<std::array<uint64_t, kModuleExtraLedCount>, kModuleExtraLedRows> port_uuids{};
    // convenience accessors for legacy-style single send/receive
    std::array<void*, kModuleExtraLedCount>& send_ptrs() { return ptrs[0]; }
    const std::array<void*, kModuleExtraLedCount>& send_ptrs() const { return ptrs[0]; }
    std::array<void*, kModuleExtraLedCount>& receive_ptrs() { return ptrs[2]; }
    const std::array<void*, kModuleExtraLedCount>& receive_ptrs() const { return ptrs[2]; }
    std::array<uint64_t, kModuleExtraLedCount>& send_port_uuids() { return port_uuids[0]; }
    const std::array<uint64_t, kModuleExtraLedCount>& send_port_uuids() const { return port_uuids[0]; }
    std::array<uint64_t, kModuleExtraLedCount>& receive_port_uuids() { return port_uuids[2]; }
    const std::array<uint64_t, kModuleExtraLedCount>& receive_port_uuids() const { return port_uuids[2]; }
};

struct ModulePreviewBuffer {
    std::vector<uint8_t> rgba;
    std::vector<GP_TableHitBox> hitboxes;
    int width_px = 0;
    int height_px = 0;
    int pitch_bytes = 0;
};

struct ModuleStackTail {
    std::array<float, 64> values{};
    std::atomic<int> count{0};
    std::atomic<uint32_t> seq{0};

    ModuleStackTail() = default;
    // disable copy: std::atomic is non-copyable
    ModuleStackTail(const ModuleStackTail&) = delete;
    ModuleStackTail& operator=(const ModuleStackTail&) = delete;

    // provide move semantics: copy underlying values and atomics via load/store
    ModuleStackTail(ModuleStackTail&& other) noexcept {
        values = other.values;
        count.store(other.count.load());
        seq.store(other.seq.load());
    }
    ModuleStackTail& operator=(ModuleStackTail&& other) noexcept {
        if (this != &other) {
            values = other.values;
            count.store(other.count.load());
            seq.store(other.seq.load());
        }
        return *this;
    }
};

static ModuleFrameLedGroup make_module_frame_led_group() {
    ModuleFrameLedGroup group{};
    for (int row = 0; row < kModuleExtraLedRows; ++row) {
        for (int idx = 0; idx < kModuleExtraLedCount; ++idx) {
            GP_TableCell &cell = group.cells[static_cast<size_t>(row)][static_cast<size_t>(idx)];
            std::memset(&cell, 0, sizeof(cell));
            cell.kind = GP_TABLE_CELL_LEDS;
            cell.value = 1.0f;
            cell.flags = 0u;
            cell.reserved0 = 0;
        }
    }
    return group;
}

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


