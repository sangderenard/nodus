#include "canvas_abi.h"
#include "rope_sim.h"
#include "table_abi.h"
#include "raytrace_2d.h"
#include "stage_abi.h"
#include "text_render_helper.h"
#include "thread_manager.h"
#include "module_library.h"
#include "module_library_actualizer.h"
#include "labels.h"
#include "module_preview.h"
#include "plugin_manager.h"
#include "tool_registry.h"
#include "tool_api.h"
#include "table_node_groups.h"

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
#include <map>
#include <array>
#include <atomic>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <Eigen/Dense>
#include <chrono>
#include <mutex>
#include <cstdlib>

// C-linkage prototypes for spline drawer implemented in table_abi.cpp
extern "C" void table_draw_rope_curve_blend_colored(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, const float* hues, int hue_count, int samples_per_segment, float hue_intensity);
extern "C" void table_draw_rope_curve_blend(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, uint8_t cr, uint8_t cg, uint8_t cb, uint8_t ca, int samples_per_segment);

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


// Minimal internal canvas context implementation
// Central ID hook type used by canvas-wide ID generation.
typedef uint64_t (*GP_CanvasIdHookFn)(GP_CanvasContext* ctx, uint64_t hint);
struct GP_CanvasContextImpl {
    int width=0, height=0;
    std::vector<GP_CanvasModuleDesc> modules;
    // internal edge info bundles the desc, rope index, and per-edge hues
    struct EdgeInfo {
        GP_CanvasEdgeDesc desc;
        int rope_idx = -1;
            uint64_t rope_uid = 0ull;
        int type_id = 0; // 0 == wildcard / untyped
        uint32_t subgroup_flags = 0;
        std::vector<float> hues;
        float hue_intensity = 0.0f;
        // Optional overlay keys: when non-zero, these 64-bit keys override
        // the usual module/contact-derived keys and reference custom canvas
        // overlays created by `gp_canvas_create_overlay_with_leds`.
        unsigned long long overlay_key_a = 0ull;
        unsigned long long overlay_key_b = 0ull;
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
    std::vector<int> module_sim_enabled; // per-module sim enable flag (1=simulate,0=skip)
    std::vector<int> module_skip; // per-module full-skip flag (1=skip entire module work)
    std::vector<int> module_exec_skip_count; // per-module execution cadence skip count (0 = every frame)
    std::vector<std::vector<int>> module_io_input_rows;
    std::vector<std::vector<int>> module_io_output_rows;
    std::vector<MolexLayoutInfo> module_input_layout;
    std::vector<MolexLayoutInfo> module_output_layout;
    std::vector<std::vector<ModuleIORow>> module_io_rows;
    // parallel structure to `module_io_rows` storing instantiated plugin tool
    // instances for plugin-origin rows; null entries indicate no instance.
    std::vector<std::vector<std::unique_ptr<ITool, std::function<void(ITool*)>>>> module_plugin_instances;
    std::vector<std::vector<ModuleIORow>> module_table_rows;
    std::vector<ModuleFrameLedGroup> module_frame_leds;
    std::vector<ModuleFrameLink> module_frame_links;
    std::vector<ModulePreviewBuffer> module_preview_buffers;
    std::vector<ModuleStackTail> module_stack_tail;
    std::vector<std::unordered_map<int, std::vector<float>>> module_stack_snapshots;
    std::vector<std::vector<ModuleToolKind>> module_tool_stack;
    // Action subscriber registry: map action_id -> list of (callback,user)
    std::unordered_map<int32_t, std::vector<std::pair<GP_CanvasActionSubscriberFn, void*>>> action_subscribers;
    std::mutex action_subscribers_mu;
    // Action -> bound ports (module,row,col,led,pending_ptr)
    // Per-binding edge payload published into root table FIFOs. Layout is
    // declared in `canvas_abi.h` as `EventPayload` and used by the manager.
    struct ActionBinding { int module_idx; int row; int col; int led_idx; void* pending_ptr; int root_edge_idx = -1; unsigned long long writer_key = 0ull; };
    std::unordered_map<int32_t, std::vector<ActionBinding>> action_port_bindings;
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
    // click-listen: when true, root-table click actions are captured rather
    // than immediately dispatched. `pending_action` holds an allocated
    // copy of the action intent and can be bound into module frame ptrs.
    struct PendingAction { int32_t action_id = 0; GP_TableHitBox hit{}; uint64_t aux_uid = 0ull; };
    bool click_listen_mode = false;
    PendingAction* pending_action = nullptr; // owned when non-null
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
    float sim_damping = 0.01f;
    float sim_maxforce = 800.0f;
    // global rope-sim frame-skip count (0 => step every frame).
    // The count is the number of frames to skip between steps (step every count+1 frames).
    int sim_frame_skip_count = 0;
    int sim_root_paused = 0; // toolbar-level pause for the root/global rope sim
    // tool selection state: separate groups (exclusive within group)
    // canvas tool group: 0 = select, 1 = new table, 2 = edge mode, 3 = new stage
    int selected_tool_canvas = 0;
    // edge tool group: 0=create, 1=destroy, 2=on-change, 3=continuous (-1 = none)
    int selected_tool_edge = 0;
    int edge_order_value = 0;
    int edge_order_tool_active = 0;
    // table tool group: 0 = neutral, 1 = select, 2 = menu
    int selected_tool_table = 0;
    // kpn tool group: 0..2 (K, P, N), -1 = none
    int selected_tool_kpn = -1;
    // fifo policy subgroup selector: bitmask of enabled subgroup flags
    uint32_t selected_tool_subgroup_flags = 0u;

    // lasso mode: when true, canvas will route mouse drags to the
    // meta-edge lasso handler (keyboard-toggleable). This is intentionally
    // independent from the toolbar/menu tool groups so it does not alter
    // visible toolbar layout.
    bool lasso_mode = false;
    // transient lasso path points (canvas-local coords). Collected while
    // `lasso_mode` is enabled and the user drags the mouse; processed on
    // mouse-up to create a meta-group in the attached table.
    std::vector<std::pair<float,float>> lasso_points;
    // optional lasso event callback (registered via gp_canvas_set_lasso_callback)
    GP_CanvasLassoFn lasso_cb = nullptr;
    void* lasso_cb_user = nullptr;
    // optional overlay button callback (registered via gp_canvas_set_overlay_button_callback)
    GP_CanvasOverlayButtonFn overlay_button_cb = nullptr;
    void* overlay_button_user = nullptr;
    // optional click-drag callback (registered via gp_canvas_set_click_drag_callback)
    GP_CanvasClickDragFn click_drag_cb = nullptr;
    void* click_drag_cb_user = nullptr;
    // transient click-drag tracking (independent of tool state)
    bool click_drag_active = false;
    float click_drag_start_x = 0.0f;
    float click_drag_start_y = 0.0f;
    std::array<std::array<float, 4>, kSubgroupBinCount> subgroup_base_rgba{};
    std::array<Color, kSubgroupBinCount> subgroup_base_colors{};
    std::array<std::atomic<uint32_t>, kSubgroupBinCount> subgroup_target_rgba{};
    std::array<Color, 1 << kSubgroupBinCount> subgroup_palette{};
    std::array<float, 1 << kSubgroupBinCount> subgroup_palette_hue{};
    struct ToolbarLedBox {
        int x0=0, y0=0, x1=0, y1=0; // view coords
        int wx0=0, wy0=0, wx1=0, wy1=0; // world coords (with offset)
        int subgroup_idx=0;
    };
    std::vector<ToolbarLedBox> toolbar_leds;
    bool tool_menu_open = false;
    // Custom canvas overlays keyed by small integer id. Overlays are
    // lightweight rectangle-only panes carrying two LED positions used for
    // anchoring ropes and interactions. Keys returned to callers encode the
    // overlay id and led index.
    struct OverlayEntry {
        int id = 0;
        float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
        unsigned long long key_a = 0ull;
        unsigned long long key_b = 0ull;
        // stable per-port UUIDs for the two contact positions
        uint64_t port_uuid_a = 0ull;
        uint64_t port_uuid_b = 0ull;
        // cached numeric-control rect in screen/view coords (pixels)
        int ctrl_x = -1, ctrl_y = -1, ctrl_w = 0, ctrl_h = 0;
        // authoritative meta-group binding: when an overlay represents a
        // meta-group widget, these fields are set so clicks map directly
        // to the table-side meta-group without rescanning ropes.
        GP_TableContext* meta_table = nullptr;
        GP_MetaGroup* meta_mg = nullptr;
    };
    std::unordered_map<int, OverlayEntry> overlays;
    // map port UUID -> (overlay id, led index 0/1)
    std::unordered_map<uint64_t, std::pair<int,int>> overlay_port_uuid_map;
    // map module port UUID -> (module_idx, row, led_idx)
    std::unordered_map<uint64_t, std::array<int,3>> module_port_uuid_map;
    // map overlay sentinel keys -> overlay id for fast lookup and to
    // ensure we never create duplicate overlays for the same keys
    std::unordered_map<unsigned long long, int> overlay_key_map;
    // pending snapshots waiting for overlays or backing tables to exist
    std::vector<CanvasMetaSnapshot> pending_meta_snapshots;
    // Post-load pending meta groups that need deterministic finalize pass
    struct PostLoadMetaPending { GP_TableContext* table; GP_MetaGroup* mg; float ring_u; int ring_mode; unsigned long long overlay_a; unsigned long long overlay_b; };
    std::vector<PostLoadMetaPending> post_load_meta_pending;
    // map canonical root table keys -> overlay sentinel keys so table-led
    // queries can resolve to overlay positions when appropriate.
    std::unordered_map<uint64_t, unsigned long long> canonical_to_overlay;
    // module UUID -> module_idx
    std::unordered_map<uint64_t, int> module_uuid_map;
    // mapping from persisted rope id -> (table, rope_id) for deterministic restore
    std::unordered_map<uint64_t, RopeIdEntry> rope_id_map;
    bool rope_map_dirty = false;
    // mapping from persisted meta-group id (lasso id) -> sim_group_idx
    std::unordered_map<uint64_t, int> lasso_id_map;
    // next stable port UUID (monotonic)
    uint64_t next_port_uuid = 1;
    // next stable module UUID (monotonic)
    uint64_t next_module_uuid = 1;
    // Central ID hook: callers may install a custom ID generator that
    // receives an optional hint. If null, `gp_canvas_generate_id` will
    // produce a default id using a simple hash+counter.
    GP_CanvasIdHookFn id_hook = nullptr;
    // fallback counter used by default generator to reduce colliding outputs
    std::atomic<uint64_t> next_id_counter{1};
    int next_overlay_id = 1;
    bool plugin_menu_open = false;
    int plugin_menu_module_idx = -1;
    std::vector<ModuleToolKind> plugin_tool_kinds;
    std::vector<std::string> plugin_tool_ids;
    std::vector<std::string> plugin_tool_labels;
    int io_attachment_count = 1;
    // number of send/receive row pairs to display (each pair == one grid-row)
    int module_frame_pair_count = 2;
    int table_tool_number = 1;
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
    // per-module stable UUIDs (monotonic 64-bit). 0 == unset
    std::vector<uint64_t> module_uuids;
    // graph node/contract representation
    struct NodeContract {
        int node_id = -1;
        int module_idx = -1; // which module this node belongs to (-1 if none)
        std::vector<int> input_types; // supported input type ids
        std::vector<int> output_types; // supported output type ids
    };
    std::vector<NodeContract> nodes;
    // UI control bar height (in canvas-local pixels)
    int control_bar_h = 56;
    // viewport offset (world origin visible at (0,0) in screen space)
    int offset_x = 0;
    int offset_y = 0;
    int scroll_x_needed = 0;
    int scroll_y_needed = 0;
    // optional table container (non-owning unless marked)
    GP_TableContext* container_table = nullptr;
    int container_table_owned = 0;
    int root_actions_installed = 0;
    int root_module_idx = -1; // synthetic module that mirrors the canvas root table
    // autosave parameters (path may be empty to disable)
    std::string autosave_path;
    double autosave_interval_s = 0.0;
    double autosave_accum_s = 0.0;
    bool thread_mgr_paused = true;
    int thread_mgr_delay_ms = 0;
    double thread_mgr_delay_accum_s = 0.0;
    std::unique_ptr<ThreadManager> thread_mgr;
    GP_CanvasContextImpl(int w, int h): width(w), height(h) {}
};

// (global drag map removed; each canvas has its own DragState member)

static GP_CanvasContextImpl* g_canvas_context_singleton = nullptr;

static void canvas_recompute_subgroup_palette(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    constexpr int kPaletteSize = 1 << kSubgroupBinCount;
    for (int mask = 0; mask < kPaletteSize; ++mask) {
        if (mask == 0) {
            Color neutral{120, 120, 130, 255};
            ctx->subgroup_palette[0] = neutral;
            ctx->subgroup_palette_hue[0] = 0.0f;
            continue;
        }
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_a = 0.0f;
        int count = 0;
        for (int i = 0; i < kSubgroupBinCount; ++i) {
            if ((mask & (1 << i)) == 0) continue;
            const Color base = ctx->subgroup_base_colors[static_cast<size_t>(i)];
            sum_r += static_cast<float>(base.r);
            sum_g += static_cast<float>(base.g);
            sum_b += static_cast<float>(base.b);
            sum_a += static_cast<float>(base.a);
            ++count;
        }
        if (count <= 0) count = 1;
        float inv = 1.0f / static_cast<float>(count);
        Color mixed{
            static_cast<uint8_t>(std::lround(std::clamp(sum_r * inv, 0.0f, 255.0f))),
            static_cast<uint8_t>(std::lround(std::clamp(sum_g * inv, 0.0f, 255.0f))),
            static_cast<uint8_t>(std::lround(std::clamp(sum_b * inv, 0.0f, 255.0f))),
            static_cast<uint8_t>(std::lround(std::clamp(sum_a * inv, 0.0f, 255.0f)))
        };
        ctx->subgroup_palette[static_cast<size_t>(mask)] = mixed;
        ctx->subgroup_palette_hue[static_cast<size_t>(mask)] = rgb_to_hue(mixed);
    }
}

static void canvas_init_subgroup_palette(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    for (int i = 0; i < kSubgroupBinCount; ++i) {
        float hue = static_cast<float>(i) / static_cast<float>(kSubgroupBinCount);
        Color base = hsv_to_color(hue, 0.9f, 0.95f, 255);
        ctx->subgroup_base_rgba[static_cast<size_t>(i)] = {float(base.r), float(base.g), float(base.b), float(base.a)};
        ctx->subgroup_base_colors[static_cast<size_t>(i)] = base;
        ctx->subgroup_target_rgba[static_cast<size_t>(i)].store(pack_rgba(base), std::memory_order_relaxed);
    }
    canvas_recompute_subgroup_palette(ctx);
}

static int canvas_resolve_rope_id_to_sim_index(GP_TableContext* table, uint64_t rope_id) {
    if (!table || rope_id == 0ull) return -1;
    if (!gp_table_get_rope_sim(table)) return -1;
    return gp_table_resolve_rope_id_to_sim_index(table, rope_id);
}

static uint64_t canvas_find_rope_id_for_sim(GP_CanvasContextImpl* c, GP_TableContext* table, int sim_idx) {
    if (!c || !table || sim_idx < 0) return 0ull;
    if (!gp_table_get_rope_sim(table)) return 0ull;
    for (const auto &kv : c->rope_id_map) {
        if (kv.second.table != table) continue;
        int resolved = gp_table_resolve_rope_id_to_sim_index(table, kv.second.rope_id);
        if (resolved == sim_idx) return kv.second.rope_id;
    }
    return 0ull;
}

static void canvas_refresh_rope_map(GP_CanvasContextImpl* c) {
    if (!c || !c->rope_map_dirty) return;
    std::unordered_set<GP_TableContext*> tables;
    tables.reserve(c->rope_id_map.size());
    for (const auto &kv : c->rope_id_map) {
        if (kv.second.table) tables.insert(kv.second.table);
    }
    for (auto *table : tables) {
        if (!table) continue;
        if (!gp_table_get_rope_sim(table)) continue;
        bool needs_rebuild = false;
        for (const auto &kv : c->rope_id_map) {
            if (kv.second.table != table) continue;
            if (gp_table_resolve_rope_id_to_sim_index(table, kv.second.rope_id) < 0) {
                needs_rebuild = true;
                break;
            }
        }
        if (!needs_rebuild) continue;
        int cnt = gp_table_get_rope_id_count(table);
        if (cnt <= 0) continue;
        std::vector<uint64_t> tmp(static_cast<size_t>(cnt));
        int got = gp_table_get_rope_ids(table, tmp.data(), cnt);
        if (got > 0) {
            gp_table_set_rope_ids_from_array(table, tmp.data(), got);
            printf("canvas: refreshed rope map for table=%p entries=%d\n", (void*)table, got);
        }
    }
    c->rope_map_dirty = false;
}

extern "C" int gp_canvas_register_table_rope_ids_from_array(GP_CanvasContext* ctx, GP_TableContext* table, const uint64_t* ids, int count) {
    if (!ctx || !ids || count <= 0) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    printf("gp_canvas_register_table_rope_ids_from_array: registering %d ids for table %p\n", count, (void*)table);
    for (int i = 0; i < count; ++i) {
        uint64_t id = ids[static_cast<size_t>(i)];
        if (id == 0ull) continue;
        c->rope_id_map[id] = RopeIdEntry{table, id};
        printf("  registered rope_id=%llu -> table=%p\n", (unsigned long long)id, (void*)table);
    }
    // Try to attach registered IDs to any canvas edges that map to this table.
    for (size_t ei = 0; ei < c->edges.size(); ++ei) {
        const auto &e = c->edges[ei];
        uint64_t ka = (static_cast<uint64_t>(static_cast<uint32_t>(e.desc.a_module)) << 32) |
                      (static_cast<uint64_t>(static_cast<uint32_t>(e.desc.a_contact_idx)) << 16) |
                      static_cast<uint64_t>(0);
        uint64_t kb = (static_cast<uint64_t>(static_cast<uint32_t>(e.desc.b_module)) << 32) |
                      (static_cast<uint64_t>(static_cast<uint32_t>(e.desc.b_contact_idx)) << 16) |
                      static_cast<uint64_t>(0);
        int edge_idx = -1;
        if (!gp_table_edge_index_for_pair(table, ka, kb, &edge_idx)) continue;
        if (edge_idx < 0 || edge_idx >= count) continue;
        uint64_t mapped_id = ids[static_cast<size_t>(edge_idx)];
        if (mapped_id == 0ull) continue;
        c->edges[ei].rope_uid = mapped_id; // keep existing edge field name
        // refresh rope map entry to ensure table association is recorded
        c->rope_id_map[mapped_id] = RopeIdEntry{table, mapped_id};
        printf("  bound canvas.edge[%zu] -> rope_id=%llu (table_edge=%d)\n", ei, (unsigned long long)mapped_id, edge_idx);
    }
    return 1;
}

// Central ID generation API. Callers should use this instead of
// incrementing per-field counters so the canvas can provide a uniform
// ID scheme via `gp_canvas_set_id_hook`.
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

extern "C" uint64_t gp_canvas_generate_id(GP_CanvasContext* ctx, uint64_t hint) {
    // Accept null ctx: use a global fallback counter so callers don't need
    // to special-case canvas availability. Also honor a pluggable id_hook
    // when a canvas is present.
    static std::atomic<uint64_t> s_fallback_counter{1};
    if (ctx) {
        GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
        if (c->id_hook) return c->id_hook(ctx, hint);
        uint64_t count = c->next_id_counter.fetch_add(1);
        uint64_t seed = (hint * 0x9e3779b97f4a7c15ULL) ^ count ^ reinterpret_cast<uint64_t>(ctx);
        uint64_t out = splitmix64(seed);
        if (out == 0ull) out = splitmix64(seed ^ 0xdeadbeefcafebabeULL);
        return out;
    } else {
        uint64_t count = s_fallback_counter.fetch_add(1);
        uint64_t seed = (hint * 0x9e3779b97f4a7c15ULL) ^ count;
        uint64_t out = splitmix64(seed);
        if (out == 0ull) out = splitmix64(seed ^ 0xdeadbeefcafebabeULL);
        return out;
    }
}

extern "C" GP_CanvasIdHookFn gp_canvas_set_id_hook(GP_CanvasContext* ctx, GP_CanvasIdHookFn hook) {
    if (!ctx) return nullptr;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    GP_CanvasIdHookFn prev = c->id_hook;
    c->id_hook = hook;
    return prev;
}

// Return edges whose both endpoints are module indices that map to `table`.
// For each matching canvas edge, optionally output the edge desc and the
// persistent rope id (if known). If `out_edges` or `out_rope_ids` are
// NULL, the function returns the required count. `max_entries` bounds the
// number of entries written when output buffers are provided.
extern "C" int gp_canvas_get_table_local_edges_and_rope_ids(GP_CanvasContext* ctx_, GP_TableContext* table, GP_CanvasEdgeDesc* out_edges, uint64_t* out_rope_ids, int max_entries) {
    if (!ctx_ || !table) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // determine module indices that point to the provided table
    std::vector<int> module_idxs;
    for (size_t mi = 0; mi < c->module_tables.size(); ++mi) {
        if (c->module_tables[mi] == table) module_idxs.push_back(static_cast<int>(mi));
    }
    // also treat the container_table as a special module index marker (-1)
    bool is_container = (c->container_table == table);
    auto module_belongs = [&](int mid) -> bool {
        if (mid < 0 && is_container) return true;
        for (int x : module_idxs) if (x == mid) return true;
        return false;
    };

    // first pass: count matching edges
    int count = 0;
    for (const auto &ei : c->edges) {
        int a = ei.desc.a_module;
        int b = ei.desc.b_module;
        if (module_belongs(a) && module_belongs(b)) ++count;
    }
    if (!out_edges && !out_rope_ids) return count;

    int written = 0;
    for (const auto &ei : c->edges) {
        if (written >= max_entries) break;
        int a = ei.desc.a_module;
        int b = ei.desc.b_module;
        if (!(module_belongs(a) && module_belongs(b))) continue;
        if (out_edges) out_edges[written] = ei.desc;
        uint64_t rid = 0ull;
        // prefer explicit edge rope uid if present; otherwise search rope_id_map
        if (ei.rope_uid != 0ull) rid = ei.rope_uid;
        else {
            rid = gp_canvas_find_persistent_rope_id(ctx_, table, ei.rope_idx);
        }
        if (out_rope_ids) out_rope_ids[written] = rid;
        ++written;
    }
    return written;
}

extern "C" int gp_canvas_table_meta_add_vertex_by_id(GP_CanvasContext* ctx, GP_TableContext* table, void* meta_mg, uint64_t rope_id, int vertex_idx) {
    if (!ctx || !table || !meta_mg || rope_id == 0ull) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    auto it = c->rope_id_map.find(rope_id);
    if (it == c->rope_id_map.end()) {
        printf("gp_canvas_table_meta_add_vertex_by_id: could not find mapping for rope_id=%llu\n", (unsigned long long)rope_id);
        return 0;
    }
    GP_TableContext* mapped_table = it->second.table;
    if (mapped_table != table) {
        printf("gp_canvas_table_meta_add_vertex_by_id: rope_id=%llu maps to a different table (%p) than target (%p)\n", (unsigned long long)rope_id, (void*)mapped_table, (void*)table);
        return 0;
    }
    if (!gp_table_get_rope_sim(table)) {
        printf("gp_canvas_table_meta_add_vertex_by_id: rope_id=%llu table=%p has no RopeSim attached yet; deferring\n", (unsigned long long)rope_id, (void*)table);
        return 0;
    }
    int mapped_idx = gp_table_resolve_rope_id_to_sim_index(table, rope_id);
    if (mapped_idx < 0) {
        printf("gp_canvas_table_meta_add_vertex_by_id: rope_id=%llu could not resolve sim index for table=%p\n", (unsigned long long)rope_id, (void*)table);
        return 0;
    }
    return gp_table_meta_add_vertex(table, reinterpret_cast<GP_MetaGroup*>(meta_mg), mapped_idx, vertex_idx);
}

extern "C" int gp_canvas_resolve_rope_id_to_index(GP_CanvasContext* ctx, GP_TableContext* table, uint64_t rope_id) {
    if (!ctx || !table || rope_id == 0ull) return -1;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    auto it = c->rope_id_map.find(rope_id);
    if (it == c->rope_id_map.end()) return -1;
    if (it->second.table != table) return -1;
    return canvas_resolve_rope_id_to_sim_index(table, rope_id);
}

extern "C" void gp_canvas_mark_rope_map_dirty(GP_CanvasContext* ctx) {
    if (!ctx) return;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    c->rope_map_dirty = true;
}

// Canonical search for persistent rope id for a given table+sim index.
extern "C" uint64_t gp_canvas_find_persistent_rope_id(GP_CanvasContext* ctx_, GP_TableContext* table, int sim_idx) {
    if (!ctx_ || !table || sim_idx < 0) return 0ull;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);

    // Helper: check a specific table for a mapping that refers to the same RopeSim
    auto check_table_for_sim = [&](GP_TableContext* other) -> uint64_t {
        if (!other) return 0ull;
        RopeSim* s1 = gp_table_get_rope_sim(table);
        RopeSim* s2 = gp_table_get_rope_sim(other);
        if (s1 == nullptr || s2 == nullptr) return 0ull;
        if (s1 != s2) return 0ull; // only meaningful if same sim attached

        // Prefer explicit per-table rope ids
        int cur_cnt = gp_table_get_rope_id_count(other);
        if (cur_cnt > 0 && sim_idx >= 0 && sim_idx < cur_cnt) {
            std::vector<uint64_t> tmp(static_cast<size_t>(cur_cnt));
            int got = gp_table_get_rope_ids(other, tmp.data(), cur_cnt);
            if (got > sim_idx && tmp[static_cast<size_t>(sim_idx)] != 0ull) return tmp[static_cast<size_t>(sim_idx)];
        }

        // Fall back to canvas registry entries that point to this table
        return canvas_find_rope_id_for_sim(c, other, sim_idx);
    };

    // 1) Local table
    // Prefer table-local persisted ids if available
    int local_cnt = gp_table_get_rope_id_count(table);
    if (local_cnt > 0 && sim_idx >= 0 && sim_idx < local_cnt) {
        std::vector<uint64_t> tmp(static_cast<size_t>(local_cnt));
        int got = gp_table_get_rope_ids(table, tmp.data(), local_cnt);
        if (got > sim_idx && tmp[static_cast<size_t>(sim_idx)] != 0ull) return tmp[static_cast<size_t>(sim_idx)];
    }

    // also consult canvas registry directly for exact (table,sim_idx) mapping
    uint64_t by_map = canvas_find_rope_id_for_sim(c, table, sim_idx);
    if (by_map != 0ull) return by_map;

    // 2) Up the chain: container table and root module table
    uint64_t found = 0ull;
    found = check_table_for_sim(c->container_table);
    if (found != 0ull) return found;
    if (c->root_module_idx >= 0 && c->root_module_idx < static_cast<int>(c->module_tables.size())) {
        found = check_table_for_sim(c->module_tables[c->root_module_idx]);
        if (found != 0ull) return found;
    }

    // 3) Breadth-first search across module graph
    int mod_count = static_cast<int>(c->modules.size());
    if (mod_count <= 0) return 0ull;
    // find starting modules that host `table`
    std::vector<int> start_mods;
    for (int mi = 0; mi < static_cast<int>(c->module_tables.size()); ++mi) {
        if (c->module_tables[mi] == table) start_mods.push_back(mi);
    }
    if (start_mods.empty()) {
        // fallback to root module if none found
        if (c->root_module_idx >= 0 && c->root_module_idx < static_cast<int>(c->module_tables.size())) start_mods.push_back(c->root_module_idx);
        else start_mods.push_back(0);
    }

    std::vector<char> visited(static_cast<size_t>(mod_count));
    std::deque<int> q;
    for (int s : start_mods) { if (s >= 0 && s < mod_count) { visited[static_cast<size_t>(s)] = 1; q.push_back(s); } }

    while (!q.empty()) {
        int cur = q.front(); q.pop_front();
        GP_TableContext* t = nullptr;
        if (cur >= 0 && cur < static_cast<int>(c->module_tables.size())) t = c->module_tables[cur];
        if (t) {
            uint64_t id = check_table_for_sim(t);
            if (id != 0ull) return id;
        }

        // enqueue neighbors from canvas edges graph
        for (const auto &ei : c->edges) {
            int a = ei.desc.a_module;
            int b = ei.desc.b_module;
            if (a == cur && b >= 0 && b < mod_count && !visited[static_cast<size_t>(b)]) { visited[static_cast<size_t>(b)] = 1; q.push_back(b); }
            else if (b == cur && a >= 0 && a < mod_count && !visited[static_cast<size_t>(a)]) { visited[static_cast<size_t>(a)] = 1; q.push_back(a); }
        }
    }

    return 0ull;
}

extern "C" uint64_t gp_canvas_generate_port_uuid(GP_CanvasContext* ctx) {
    if (!ctx) return 0ull;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    // Prefer central generator so ID scheme is uniform and pluggable.
    return gp_canvas_generate_id(ctx, 0ull);
}

extern "C" uint64_t gp_canvas_generate_module_uuid(GP_CanvasContext* ctx) {
    if (!ctx) return 0ull;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    return gp_canvas_generate_id(ctx, 0ull);
}

extern "C" int gp_canvas_set_module_uuid(GP_CanvasContext* ctx, int module_idx, uint64_t module_uuid) {
    if (!ctx || module_idx < 0) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    if (module_idx >= static_cast<int>(c->modules.size())) return 0;
    if (static_cast<int>(c->module_uuids.size()) <= module_idx) c->module_uuids.resize(c->modules.size());
    // erase previous mapping if present
    uint64_t prev = c->module_uuids[static_cast<size_t>(module_idx)];
    if (prev != 0ull) c->module_uuid_map.erase(prev);
    c->module_uuids[static_cast<size_t>(module_idx)] = module_uuid;
    if (module_uuid != 0ull) c->module_uuid_map[module_uuid] = module_idx;
    return 1;
}

extern "C" uint64_t gp_canvas_get_module_uuid(GP_CanvasContext* ctx, int module_idx) {
    if (!ctx || module_idx < 0) return 0ull;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    if (module_idx >= static_cast<int>(c->module_uuids.size())) return 0ull;
    return c->module_uuids[static_cast<size_t>(module_idx)];
}

extern "C" int gp_canvas_set_module_frame_port_uuid(GP_CanvasContext* ctx, int module_idx, int is_send, int col, int led_idx, uint64_t port_uuid) {
    if (!ctx || module_idx < 0) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    if (module_idx >= static_cast<int>(c->module_frame_links.size())) return 0;
    if (led_idx < 0 || led_idx >= kModuleExtraLedCount) return 0;
    if (col < 0 || col > 1) return 0;
    int row = is_send ? (col == 0 ? 0 : 1) : (col == 0 ? 2 : 3);
    auto &links = c->module_frame_links[module_idx];
    // remove old mapping if present
    uint64_t old = links.port_uuids[static_cast<size_t>(row)][static_cast<size_t>(led_idx)];
    if (old != 0ull) c->module_port_uuid_map.erase(old);
    links.port_uuids[static_cast<size_t>(row)][static_cast<size_t>(led_idx)] = port_uuid;
    if (port_uuid != 0ull) c->module_port_uuid_map[port_uuid] = {static_cast<int>(module_idx), row, led_idx};
    return 1;
}

extern "C" uint64_t gp_canvas_get_module_frame_port_uuid(GP_CanvasContext* ctx, int module_idx, int is_send, int col, int led_idx) {
    if (!ctx || module_idx < 0) return 0ull;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    if (module_idx >= static_cast<int>(c->module_frame_links.size())) return 0ull;
    if (led_idx < 0 || led_idx >= kModuleExtraLedCount) return 0ull;
    if (col < 0 || col > 1) return 0ull;
    int row = is_send ? (col == 0 ? 0 : 1) : (col == 0 ? 2 : 3);
    return c->module_frame_links[module_idx].port_uuids[static_cast<size_t>(row)][static_cast<size_t>(led_idx)];
}



extern "C" int gp_canvas_register_table_module_uuid(GP_CanvasContext* ctx, GP_TableContext* table, uint64_t module_uuid) {
    if (!ctx || !table) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    for (size_t i = 0; i < c->module_tables.size(); ++i) {
        if (c->module_tables[i] == table) {
            return gp_canvas_set_module_uuid(ctx, static_cast<int>(i), module_uuid);
        }
    }
    return 0;
}

extern "C" int gp_canvas_register_table_frame_port_uuid(GP_CanvasContext* ctx, GP_TableContext* table, int row, int col_idx, uint64_t port_uuid) {
    if (!ctx || !table) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    for (size_t i = 0; i < c->module_tables.size(); ++i) {
        if (c->module_tables[i] == table) {
            int module_idx = static_cast<int>(i);
            if (row < 0 || row >= kModuleExtraLedRows) return 0;
            if (col_idx < 0 || col_idx >= kModuleExtraLedCount) return 0;
            uint64_t old = c->module_frame_links[static_cast<size_t>(module_idx)].port_uuids[static_cast<size_t>(row)][static_cast<size_t>(col_idx)];
            if (old != 0ull) c->module_port_uuid_map.erase(old);
            c->module_frame_links[static_cast<size_t>(module_idx)].port_uuids[static_cast<size_t>(row)][static_cast<size_t>(col_idx)] = port_uuid;
            if (port_uuid != 0ull) c->module_port_uuid_map[port_uuid] = {module_idx, row, col_idx};
            return 1;
        }
    }
    return 0;
}

static void canvas_update_subgroup_palette(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    constexpr float kApproach = 0.18f;
    for (int i = 0; i < kSubgroupBinCount; ++i) {
        const uint32_t packed = ctx->subgroup_target_rgba[static_cast<size_t>(i)].load(std::memory_order_acquire);
        const Color target = unpack_rgba(packed);
        auto &base = ctx->subgroup_base_rgba[static_cast<size_t>(i)];
        const float target_vals[4] = {float(target.r), float(target.g), float(target.b), float(target.a)};
        for (int c = 0; c < 4; ++c) {
            float delta = target_vals[c] - base[static_cast<size_t>(c)];
            if (std::fabs(delta) < 0.5f) {
                base[static_cast<size_t>(c)] = target_vals[c];
            } else {
                base[static_cast<size_t>(c)] += delta * kApproach;
            }
        }
        ctx->subgroup_base_colors[static_cast<size_t>(i)] = Color{
            static_cast<uint8_t>(std::lround(std::clamp(base[0], 0.0f, 255.0f))),
            static_cast<uint8_t>(std::lround(std::clamp(base[1], 0.0f, 255.0f))),
            static_cast<uint8_t>(std::lround(std::clamp(base[2], 0.0f, 255.0f))),
            static_cast<uint8_t>(std::lround(std::clamp(base[3], 0.0f, 255.0f)))
        };
    }
    canvas_recompute_subgroup_palette(ctx);
}

static float subgroup_flags_to_hue(const GP_CanvasContextImpl* ctx, uint32_t flags) {
    if (!ctx) return 0.0f;
    uint32_t idx = subgroup_palette_index(flags);
    return ctx->subgroup_palette_hue[static_cast<size_t>(idx)];
}

static Color subgroup_flags_to_color(const GP_CanvasContextImpl* ctx, uint32_t flags, uint8_t alpha=255) {
    if (!ctx) return Color{120, 120, 130, alpha};
    uint32_t idx = subgroup_palette_index(flags);
    Color out = ctx->subgroup_palette[static_cast<size_t>(idx)];
    out.a = alpha;
    return out;
}

// forward declaration: update scroll/clamp state (defined later in this file)
static CanvasBounds update_canvas_scroll_state(GP_CanvasContextImpl* ctx, bool pull_from_container);

// Forward declarations for symbols defined later but referenced earlier.
struct KeyRecorderState;
static void canvas_dispatch_event_to_bound_ports(GP_CanvasContextImpl* c, int32_t action_id, int x, int y, bool down, bool up);

// forward declaration: write the tail values into a module's stack snapshot
static void module_stack_tail_write(GP_CanvasContextImpl* ctx, int module_idx, const float* values, int count);

// forward declarations: root sim helpers (defined later)
static RopeSim* canvas_root_sim(GP_CanvasContextImpl* ctx);
static RopeSim* canvas_require_root_sim(GP_CanvasContextImpl* ctx);

const std::vector<ModuleIORow>* canvas_get_module_io_rows(int module_idx) {
    if (!g_canvas_context_singleton) return nullptr;
    if (module_idx < 0 || module_idx >= static_cast<int>(g_canvas_context_singleton->module_io_rows.size())) return nullptr;
    return &g_canvas_context_singleton->module_io_rows[module_idx];
}

// Return the live plugin instance (ITool*) for the given module row, or nullptr.
ITool* canvas_get_plugin_instance(int module_idx, int row_idx) {
    if (!g_canvas_context_singleton) return nullptr;
    auto *ctx = g_canvas_context_singleton;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_plugin_instances.size())) return nullptr;
    const auto &vec = ctx->module_plugin_instances[module_idx];
    if (row_idx < 0 || row_idx >= static_cast<int>(vec.size())) return nullptr;
    auto &p = vec[static_cast<size_t>(row_idx)];
    return p ? p.get() : nullptr;
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

void canvas_set_module_stack_tail(int module_idx, const float* values, int count) {
    if (!g_canvas_context_singleton) return;
    module_stack_tail_write(g_canvas_context_singleton, module_idx, values, count);
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
            rows.push_back({ModuleRowKind::Tool, -1, tool, 0, ModuleToolOrigin::Builtin, std::string()});
        }
    }
    for (int count : output_rows) {
        rows.push_back({ModuleRowKind::Output, 0, ModuleToolKind::None, std::clamp(count, 1, 32)});
    }
}

static void canvas_setup_stage_table(GP_TableContext* t, int w_px);
static void canvas_setup_stage_defaults(GP_StageContext* st, int w_px, int h_px);
static void stage_bg_callback(void* user, int module_idx, int width, int height, uint8_t* out_rgba, int32_t out_pitch);
static void sync_module_table_io_layout(GP_CanvasContextImpl* ctx, int module_idx);

static void canvas_apply_io_rows(GP_CanvasContextImpl* ctx, int module_idx, const std::vector<ModuleIORow>& rows) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) ctx->module_io_rows.resize(module_idx + 1);
    if (module_idx >= static_cast<int>(ctx->module_io_input_rows.size())) ctx->module_io_input_rows.resize(module_idx + 1);
    if (module_idx >= static_cast<int>(ctx->module_io_output_rows.size())) ctx->module_io_output_rows.resize(module_idx + 1);
    if (module_idx >= static_cast<int>(ctx->module_tool_stack.size())) ctx->module_tool_stack.resize(module_idx + 1);

    ctx->module_io_rows[module_idx].clear();
    ctx->module_io_input_rows[module_idx].clear();
    ctx->module_io_output_rows[module_idx].clear();
    ctx->module_tool_stack[module_idx].clear();

    int in_total = 0;
    for (const auto &row : rows) {
        if (row.kind == ModuleRowKind::Input) {
            in_total += std::clamp(row.attachment_count, 1, 32);
        }
    }
    int in_offset = 0;
    int out_offset = 0;
    for (const auto &row : rows) {
        if (row.kind == ModuleRowKind::Tool) {
            ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Tool, -1, row.tool, row.attachment_count, row.tool_origin, row.plugin_id});
            ctx->module_tool_stack[module_idx].push_back(row.tool);
            continue;
        }
        int count = std::clamp(row.attachment_count, 1, 32);
        if (row.kind == ModuleRowKind::Input) {
            ctx->module_io_input_rows[module_idx].push_back(count);
            ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Input, in_offset, ModuleToolKind::None, count});
            in_offset += count;
        } else if (row.kind == ModuleRowKind::Output) {
            ctx->module_io_output_rows[module_idx].push_back(count);
            ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Output, in_total + out_offset, ModuleToolKind::None, count});
            out_offset += count;
        }
    }
    if (module_idx >= static_cast<int>(ctx->module_io_in_count.size())) ctx->module_io_in_count.resize(module_idx + 1, 0);
    if (module_idx >= static_cast<int>(ctx->module_io_out_count.size())) ctx->module_io_out_count.resize(module_idx + 1, 0);
    ctx->module_io_in_count[module_idx] = in_offset;
    ctx->module_io_out_count[module_idx] = out_offset;
    ensure_module_row_order(ctx, module_idx);
    // synchronize plugin instance vector for this module
    if (module_idx >= static_cast<int>(ctx->module_plugin_instances.size())) ctx->module_plugin_instances.resize(module_idx + 1);
    ctx->module_plugin_instances[module_idx].clear();
    ctx->module_plugin_instances[module_idx].resize(ctx->module_io_rows[module_idx].size());
    // attempt to instantiate plugin-origin rows immediately if registry has them
    for (size_t ri = 0; ri < ctx->module_io_rows[module_idx].size(); ++ri) {
        const auto &r = ctx->module_io_rows[module_idx][ri];
        if (r.kind == ModuleRowKind::Tool && r.tool_origin == ModuleToolOrigin::Plugin && !r.plugin_id.empty()) {
            try {
                auto inst = tool_registry_global().create(r.plugin_id);
                if (inst) ctx->module_plugin_instances[module_idx][ri] = std::move(inst);
            } catch (...) {}
        }
    }
}

static void canvas_configure_stage_module(GP_CanvasContextImpl* ctx, int module_idx, int w_px, int h_px) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx < static_cast<int>(ctx->module_is_stage.size())) ctx->module_is_stage[module_idx] = 1;
    if (module_idx < static_cast<int>(ctx->module_stages.size())) {
        GP_StageContext* st = gp_stage_create(w_px, h_px, 2);
        ctx->module_stages[module_idx] = st;
        if (module_idx < static_cast<int>(ctx->module_stage_owned.size())) ctx->module_stage_owned[module_idx] = 1;
        canvas_setup_stage_defaults(st, w_px, h_px);
    }
    if (module_idx < static_cast<int>(ctx->module_io_in_count.size())) ctx->module_io_in_count[module_idx] = 1;
    if (module_idx < static_cast<int>(ctx->module_io_out_count.size())) ctx->module_io_out_count[module_idx] = 1;
    if (module_idx < static_cast<int>(ctx->module_bg.size())) {
        ctx->module_bg[module_idx].cb = stage_bg_callback;
        ctx->module_bg[module_idx].user = ctx;
        ctx->module_bg[module_idx].table_alpha = 192;
        ctx->module_bg[module_idx].table_alpha_ray = 192;
        ctx->module_bg[module_idx].header_margin_px = 0;
    }
    if (module_idx < static_cast<int>(ctx->module_tables.size()) && ctx->module_tables[module_idx]) {
        canvas_setup_stage_table(ctx->module_tables[module_idx], w_px);
        sync_module_table_io_layout(ctx, module_idx);
    }
}

static void canvas_clear_workspace(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    for (size_t i = 0; i < ctx->module_tables.size(); ++i) {
        if (ctx->module_tables[i] && i < ctx->module_table_owned.size() && ctx->module_table_owned[i]) {
            gp_table_destroy(ctx->module_tables[i]);
        }
        if (i < ctx->module_key_recorder_state.size()) {
            void* s = ctx->module_key_recorder_state[i];
            if (s) delete reinterpret_cast<KeyRecorderState*>(s);
        }
    }
    for (size_t i = 0; i < ctx->module_stages.size(); ++i) {
        if (ctx->module_stages[i] && i < ctx->module_stage_owned.size() && ctx->module_stage_owned[i]) {
            gp_stage_destroy(ctx->module_stages[i]);
        }
    }
    if (ctx->container_table) {
        gp_table_clear_edges(ctx->container_table);
    }
    ctx->modules.clear();
    ctx->module_tables.clear();
    ctx->module_table_owned.clear();
    ctx->module_stages.clear();
    ctx->module_stage_owned.clear();
    ctx->module_is_stage.clear();
    ctx->module_stage_images.clear();
    ctx->module_stage_cache_rgba.clear();
    ctx->module_stage_cache_w.clear();
    ctx->module_stage_cache_h.clear();
    ctx->module_stage_cache_pitch.clear();
    ctx->module_stage_cache_mu.clear();
    ctx->module_stage_integrator_mode.clear();
    ctx->module_stage_integrator_accum.clear();
    ctx->module_bg.clear();
    ctx->module_io_in_count.clear();
    ctx->module_io_out_count.clear();
    ctx->module_io_input_rows.clear();
    ctx->module_io_output_rows.clear();
    ctx->module_input_layout.clear();
    ctx->module_output_layout.clear();
    ctx->module_io_rows.clear();
    ctx->module_table_rows.clear();
    ctx->module_frame_leds.clear();
    ctx->module_frame_links.clear();
    ctx->module_preview_buffers.clear();
    ctx->module_stack_tail.clear();
    ctx->module_stack_snapshots.clear();
    ctx->module_tool_stack.clear();
    ctx->module_key_recorder_state.clear();
    ctx->module_input_state.clear();
    ctx->module_chat_text.clear();
    ctx->module_chat_color.clear();
    ctx->module_chat_ttl.clear();
    ctx->module_plugin_instances.clear();
    ctx->edges.clear();
    ctx->nodes.clear();
    ctx->module_node_id.clear();
    ctx->selected = {};
    ctx->dispatch_module_idx = -1;
    ctx->focused_module = -1;
    ctx->prospective_rope_idx = -1;
    ctx->tool_menu_open = false;
    int max_window_node = 0;
    for (const auto &entry : ctx->window_node_ids) {
        max_window_node = std::max(max_window_node, entry.second);
    }
    ctx->next_node_id = std::max(1, max_window_node + 1);
}

static void canvas_clear_module_table(GP_CanvasContextImpl* ctx, int module_idx) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx < static_cast<int>(ctx->module_tables.size()) && ctx->module_tables[module_idx]) {
        gp_table_clear_edges(ctx->module_tables[module_idx]);
        gp_table_clear_led_glow(ctx->module_tables[module_idx]);
    }
    if (module_idx < static_cast<int>(ctx->module_stack_snapshots.size())) {
        ctx->module_stack_snapshots[module_idx].clear();
    }
    if (module_idx < static_cast<int>(ctx->module_stack_tail.size())) {
        ModuleStackTail &tail = ctx->module_stack_tail[module_idx];
        tail.count.store(0);
        tail.seq.store(tail.seq.load() + 1);
    }
    if (module_idx < static_cast<int>(ctx->module_chat_text.size())) ctx->module_chat_text[module_idx].clear();
    if (module_idx < static_cast<int>(ctx->module_chat_ttl.size())) ctx->module_chat_ttl[module_idx] = 0;
}

static int canvas_clone_module(GP_CanvasContextImpl* ctx, int module_idx) {
    if (!ctx) return -1;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return -1;
    GP_CanvasModuleDesc d = ctx->modules[module_idx];
    d.x += 24;
    d.y += 24;
    int new_idx = gp_canvas_add_module(reinterpret_cast<GP_CanvasContext*>(ctx), &d);
    if (new_idx < 0) return -1;
    bool is_stage = (module_idx < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[module_idx]);
    if (is_stage) {
        canvas_configure_stage_module(ctx, new_idx, d.w, d.h);
    } else if (module_idx < static_cast<int>(ctx->module_bg.size()) && new_idx < static_cast<int>(ctx->module_bg.size())) {
        const auto &src = ctx->module_bg[module_idx];
        auto &dst = ctx->module_bg[new_idx];
        dst.mode = src.mode;
        dst.rays = src.rays;
        dst.reflections = src.reflections;
        dst.blur_sigma = src.blur_sigma;
        dst.oversample = src.oversample;
        dst.temporal_decay = src.temporal_decay;
        dst.temporal_max = src.temporal_max;
        dst.ray_exposure = src.ray_exposure;
        dst.ray_bounce_decay = src.ray_bounce_decay;
        dst.ray_air_decay = src.ray_air_decay;
        dst.table_alpha = src.table_alpha;
        dst.table_alpha_ray = src.table_alpha_ray;
        dst.header_margin_px = src.header_margin_px;
    }
    const std::vector<ModuleIORow>* rows_ptr = nullptr;
    if (module_idx < static_cast<int>(ctx->module_table_rows.size()) &&
        !ctx->module_table_rows[module_idx].empty()) {
        rows_ptr = &ctx->module_table_rows[module_idx];
    } else if (module_idx < static_cast<int>(ctx->module_io_rows.size())) {
        rows_ptr = &ctx->module_io_rows[module_idx];
    }
    if (rows_ptr) {
        canvas_apply_io_rows(ctx, new_idx, *rows_ptr);
        sync_module_table_io_layout(ctx, new_idx);
    }
    return new_idx;
}

static void canvas_destroy_module(GP_CanvasContextImpl* ctx, int module_idx) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx < static_cast<int>(ctx->module_tables.size()) &&
        module_idx < static_cast<int>(ctx->module_table_owned.size()) &&
        ctx->module_tables[module_idx] && ctx->module_table_owned[module_idx]) {
        gp_table_destroy(ctx->module_tables[module_idx]);
    }
    if (module_idx < static_cast<int>(ctx->module_stages.size()) &&
        module_idx < static_cast<int>(ctx->module_stage_owned.size()) &&
        ctx->module_stages[module_idx] && ctx->module_stage_owned[module_idx]) {
        gp_stage_destroy(ctx->module_stages[module_idx]);
    }
    if (module_idx < static_cast<int>(ctx->module_key_recorder_state.size())) {
        void* s = ctx->module_key_recorder_state[module_idx];
        if (s) delete reinterpret_cast<KeyRecorderState*>(s);
    }
    if (module_idx < static_cast<int>(ctx->module_bg.size()) && ctx->module_bg[module_idx].ray) {
        raytrace2d_destroy(ctx->module_bg[module_idx].ray);
        ctx->module_bg[module_idx].ray = nullptr;
    }
    for (size_t i = 0; i < ctx->edges.size();) {
        auto &edge = ctx->edges[i];
        if (edge.desc.a_module == module_idx || edge.desc.b_module == module_idx) {
            ctx->edges.erase(ctx->edges.begin() + static_cast<long>(i));
            continue;
        }
        if (edge.desc.a_module > module_idx) --edge.desc.a_module;
        if (edge.desc.b_module > module_idx) --edge.desc.b_module;
        ++i;
    }
    for (auto it = ctx->nodes.begin(); it != ctx->nodes.end();) {
        if (it->module_idx == module_idx) {
            it = ctx->nodes.erase(it);
            continue;
        }
        if (it->module_idx > module_idx) --it->module_idx;
        ++it;
    }
    // Remove any module UUID / module-port UUID mappings for this module
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_uuids.size())) {
        uint64_t mu = ctx->module_uuids[static_cast<size_t>(module_idx)];
        if (mu != 0ull) ctx->module_uuid_map.erase(mu);
    }
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_frame_links.size())) {
        const auto &links = ctx->module_frame_links[static_cast<size_t>(module_idx)];
        for (int row = 0; row < kModuleExtraLedRows; ++row) {
            for (int li = 0; li < kModuleExtraLedCount; ++li) {
                uint64_t pu = links.port_uuids[static_cast<size_t>(row)][static_cast<size_t>(li)];
                if (pu != 0ull) ctx->module_port_uuid_map.erase(pu);
            }
        }
    }
    auto erase_at = [&](auto &vec) {
        if (module_idx >= 0 && module_idx < static_cast<int>(vec.size())) {
            vec.erase(vec.begin() + module_idx);
        }
    };
    erase_at(ctx->modules);
    erase_at(ctx->module_tables);
    erase_at(ctx->module_table_owned);
    erase_at(ctx->module_stages);
    erase_at(ctx->module_stage_owned);
    erase_at(ctx->module_is_stage);
    erase_at(ctx->module_stage_images);
    erase_at(ctx->module_stage_cache_rgba);
    erase_at(ctx->module_stage_cache_w);
    erase_at(ctx->module_stage_cache_h);
    erase_at(ctx->module_stage_cache_pitch);
    erase_at(ctx->module_stage_cache_mu);
    erase_at(ctx->module_stage_integrator_mode);
    erase_at(ctx->module_stage_integrator_accum);
    erase_at(ctx->module_bg);
    erase_at(ctx->module_io_in_count);
    erase_at(ctx->module_io_out_count);
    erase_at(ctx->module_io_input_rows);
    erase_at(ctx->module_io_output_rows);
    erase_at(ctx->module_input_layout);
    erase_at(ctx->module_output_layout);
    erase_at(ctx->module_io_rows);
    erase_at(ctx->module_table_rows);
    erase_at(ctx->module_frame_leds);
    erase_at(ctx->module_frame_links);
    erase_at(ctx->module_preview_buffers);
    erase_at(ctx->module_stack_tail);
    erase_at(ctx->module_stack_snapshots);
    erase_at(ctx->module_tool_stack);
    erase_at(ctx->module_key_recorder_state);
    erase_at(ctx->module_input_state);
    erase_at(ctx->module_chat_text);
    erase_at(ctx->module_chat_color);
    erase_at(ctx->module_chat_ttl);
    erase_at(ctx->module_uuids);
    erase_at(ctx->module_node_id);
    if (ctx->focused_module == module_idx) ctx->focused_module = -1;
    else if (ctx->focused_module > module_idx) --ctx->focused_module;
    if (ctx->selected.module == module_idx) {
        ctx->selected = {};
        ctx->prospective_rope_idx = -1;
    } else if (ctx->selected.module > module_idx) {
        --ctx->selected.module;
    }
    if (ctx->dispatch_module_idx == module_idx) ctx->dispatch_module_idx = -1;
    else if (ctx->dispatch_module_idx > module_idx) --ctx->dispatch_module_idx;
    update_canvas_scroll_state(ctx, /*pull_from_container=*/false);
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

// forward declarations for helpers used by interactive and deserialization paths
static int canvas_create_overlay_and_attach(GP_CanvasContextImpl* c, GP_TableContext* t, float ox1, float oy1, float ox2, float oy2, unsigned long long* out_key_a, unsigned long long* out_key_b);
static void canvas_create_and_register_ring(GP_CanvasContextImpl* c, GP_TableContext* tbl, GP_MetaGroup* mg, int first_rope_for_ring_local, int first_vid_for_ring_local, float saved_u_override);
static void canvas_finalize_lasso_meta_group(GP_CanvasContextImpl* ctx, GP_TableContext* tbl, GP_MetaGroup* mg, int first_rope_for_ring_local, int first_vid_for_ring_local, float saved_u_override);

static void canvas_record_mouse_input(GP_CanvasContextImpl* ctx, int x, int y, bool down, bool up) {
    if (!ctx) return;
    // compute canvas/world coords for events
    float fx = static_cast<float>(x + ctx->offset_x);
    float fy = static_cast<float>(y + ctx->offset_y);

    // Click-drag tracking: independent of tool state. Dispatch start/move/end
    // events to any registered callback so other systems can subscribe.
    if (down) {
        ctx->click_drag_active = true;
        ctx->click_drag_start_x = fx;
        ctx->click_drag_start_y = fy;
        if (ctx->click_drag_cb) {
            ctx->click_drag_cb(ctx->click_drag_cb_user, 1, ctx->click_drag_start_x, ctx->click_drag_start_y, fx, fy);
        }
        printf("gp_canvas_click_drag: start at %.2f,%.2f (ctx=%p)\n", fx, fy, (void*)ctx);
    } else if (up) {
        if (ctx->click_drag_active) {
            if (ctx->click_drag_cb) {
                ctx->click_drag_cb(ctx->click_drag_cb_user, 3, ctx->click_drag_start_x, ctx->click_drag_start_y, fx, fy);
            }
            printf("gp_canvas_click_drag: end at %.2f,%.2f (start %.2f,%.2f)\n", fx, fy, ctx->click_drag_start_x, ctx->click_drag_start_y);
            ctx->click_drag_active = false;
        }
    } else {
        if (ctx->click_drag_active) {
            if (ctx->click_drag_cb) ctx->click_drag_cb(ctx->click_drag_cb_user, 2, ctx->click_drag_start_x, ctx->click_drag_start_y, fx, fy);
        }
    }

    // Lasso mode handling: collect points while dragging and on mouse-up
    // resolve intersections with attached table ropes to create a
    // meta-group. This keeps the toolbar button lightweight and avoids
    // adding an additional active tool registration.
    if (ctx->lasso_mode) {
        (void)0; // use fx/fy computed above
        if (down) {
            ctx->lasso_points.clear();
            ctx->lasso_points.emplace_back(fx, fy);
            /* lasso start log removed */
            if (ctx->lasso_cb) {
                float pts[2] = { fx, fy };
                ctx->lasso_cb(ctx->lasso_cb_user, 1, pts, 1);
            }
            // Create a prospective rope in the table's sim so the UI shows
            // a temporary rope while the user is drawing. Use table-local
            // coords so projection/rendering aligns.
            GP_TableContext* tbl_start = ctx->container_table;
            if (!tbl_start && ctx->root_module_idx >= 0 && ctx->root_module_idx < static_cast<int>(ctx->module_tables.size())) {
                tbl_start = ctx->module_tables[ctx->root_module_idx];
            }
            if (tbl_start) {
                RopeSim* sim = gp_table_get_rope_sim(tbl_start);
                if (!sim) {
                    RopeSim* rootsim = canvas_root_sim(ctx);
                    if (!rootsim) rootsim = canvas_require_root_sim(ctx);
                    if (rootsim) { gp_table_attach_rope_sim(tbl_start, rootsim, 0); sim = gp_table_get_rope_sim(tbl_start); }
                }
                if (sim) {
                    // compute table-local offset for host module if present
                    float table_off_x = 0.0f, table_off_y = 0.0f;
                    int host_mod = -1;
                    for (int mi = 0; mi < static_cast<int>(ctx->module_tables.size()); ++mi) {
                        if (ctx->module_tables[mi] == tbl_start) { host_mod = mi; break; }
                    }
                    if (host_mod >= 0) {
                        const auto &m = ctx->modules[host_mod];
                        int top_h = std::min(m.h, kModuleTopUiHeight);
                        table_off_x = static_cast<float>(m.x);
                        table_off_y = static_cast<float>(m.y + top_h);
                    }
                    float sx_local = fx - table_off_x;
                    float sy_local = fy - table_off_y;
                    float plug_z = -10.0f;
                    int segs = 2;
                    int rope_idx = rope_sim_add_rope3(sim, sx_local, sy_local, plug_z, sx_local, sy_local, plug_z, segs, 0.0f);
                    if (rope_idx >= 0) {
                        ctx->prospective_rope_idx = rope_idx;
                        gp_table_set_prospective_rope_index(tbl_start, rope_idx);
                        /* prospective rope creation log removed */
                    }
                }
            }
        } else if (up) {
            // finalize lasso: if we have any points, process intersections
            if (!ctx->lasso_points.empty()) {
                // append last point (only if moved enough since last sample)
                {
                    const float min_dist_sq = 4.0f; // ~2 pixels
                    if (!ctx->lasso_points.empty()) {
                        float dx = fx - ctx->lasso_points.back().first;
                        float dy = fy - ctx->lasso_points.back().second;
                        if (dx*dx + dy*dy >= min_dist_sq) ctx->lasso_points.emplace_back(fx, fy);
                    } else {
                        ctx->lasso_points.emplace_back(fx, fy);
                    }
                }
                // Attempt to resolve against the container table if present
                GP_TableContext* tbl = ctx->container_table;
                int first_rope_for_ring = -1;
                int first_vid_for_ring = -1;
                if (!tbl && ctx->root_module_idx >= 0 && ctx->root_module_idx < static_cast<int>(ctx->module_tables.size())) {
                    tbl = ctx->module_tables[ctx->root_module_idx];
                }
                if (tbl) {
                    // create or reuse a meta group and bind any intersected rope vertices
                    GP_MetaGroup* mg = nullptr;
                    // If an overlay already has an authoritative meta-group bound
                    // and its rectangle intersects the lasso, prefer reusing it
                    // instead of creating a new meta-group (prevents empty mg churn).
                    
                    // don't create a meta-group until we actually need one
                    // (we may have found an overlay-bound mg above and set `mg`),
                    // otherwise create lazily when adding the first vertex.
                        // helper: segment-segment intersection
                        auto seg_intersect = [](float x1,float y1,float x2,float y2,float x3,float y3,float x4,float y4, float* ix, float* iy)->bool{
                            float den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
                            if (std::fabs(den) < 1e-6f) return false;
                            float t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den;
                            float u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / den;
                            if (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f) {
                                if (ix) *ix = x1 + t * (x2 - x1);
                                if (iy) *iy = y1 + t * (y2 - y1);
                                return true;
                            }
                            return false;
                        };

                        // iterate ropes known to the table via public edge->rope mapping
                        // Ensure the table has a RopeSim attached so we can
                        // query vertices. If it doesn't, attach the canvas
                        // root sim so coordinates match world space and
                        // meta-groups register correctly.
                        RopeSim* sim = gp_table_get_rope_sim(tbl);
                        if (!sim) {
                            RopeSim* rootsim = canvas_root_sim(ctx);
                            if (!rootsim) rootsim = canvas_require_root_sim(ctx);
                            if (rootsim) {
                                gp_table_attach_rope_sim(tbl, rootsim, 0);
                                sim = gp_table_get_rope_sim(tbl);
                            }
                        }
                        if (sim) {
                            // Build table-local lasso points. If this table is hosted
                            // inside a module, convert world coords -> table-local so
                            // intersections use the same space as projected verts.
                            float table_off_x = 0.0f;
                            float table_off_y = 0.0f;
                            int host_mod = -1;
                            for (int mi = 0; mi < static_cast<int>(ctx->module_tables.size()); ++mi) {
                                if (ctx->module_tables[mi] == tbl) { host_mod = mi; break; }
                            }
                            if (host_mod >= 0) {
                                const auto& m = ctx->modules[host_mod];
                                int top_h = std::min(m.h, kModuleTopUiHeight);
                                table_off_x = static_cast<float>(m.x);
                                table_off_y = static_cast<float>(m.y + top_h);
                            }

                            std::vector<std::pair<float,float>> lasso_local;
                            lasso_local.reserve(ctx->lasso_points.size());
                            for (const auto &p : ctx->lasso_points) {
                                lasso_local.emplace_back(p.first - table_off_x, p.second - table_off_y);
                            }

                            // Collect rope indices to test — id-first.
                            // Only consider rope indices that have a canonical persistent id
                            // (local table -> container/root -> BFS) so we never test
                            // attached-only ropes without canonical ids.
                            std::unordered_set<int> rope_set;
                            int32_t edge_count = gp_table_get_edge_count(tbl);
                            for (int32_t ei = 0; ei < edge_count; ++ei) {
                                int32_t rope_idx = -1;
                                if (!gp_table_get_edge_rope_index(tbl, ei, &rope_idx)) continue;
                                if (rope_idx < 0) continue;
                                uint64_t pid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, rope_idx);
                                if (pid != 0ull) rope_set.insert(rope_idx);
                            }

                            // Also include ropes referenced in the canvas registry that share
                            // the same RopeSim as the current table (allows top-level ropes
                            // hosted on root/container tables to be discovered).
                            for (const auto &kv : ctx->rope_id_map) {
                                GP_TableContext* rt = kv.second.table;
                                if (!rt) continue;
                                RopeSim* rt_sim = gp_table_get_rope_sim(rt);
                                if (!rt_sim || rt_sim != sim) continue;
                                int rsi = gp_table_resolve_rope_id_to_sim_index(rt, kv.second.rope_id);
                                if (rsi >= 0) rope_set.insert(rsi);
                            }

                            // Prefer any explicit per-table rope id list for this table
                            // (adds indices which have non-zero per-table ids).
                            int local_cnt = gp_table_get_rope_id_count(tbl);
                            if (local_cnt > 0) {
                                std::vector<uint64_t> tmp(static_cast<size_t>(local_cnt));
                                int got = gp_table_get_rope_ids(tbl, tmp.data(), local_cnt);
                                if (got > 0) {
                                    for (int ri = 0; ri < got; ++ri) {
                                        if (tmp[static_cast<size_t>(ri)] != 0ull) rope_set.insert(ri);
                                    }
                                }
                            }

                            // Scan the attached RopeSim for active ropes and keep only those
                            // that resolve to canonical ids via the standard lookup.
                            {
                                const int kScanMax = 128;
                                for (int ri = 0; ri < kScanMax; ++ri) {
                                    int vc = rope_sim_get_vertex_count(sim, ri);
                                    if (vc <= 0) continue;
                                    uint64_t pid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, ri);
                                    if (pid != 0ull) rope_set.insert(ri);
                                }
                            }

                            printf("lasso: finalize points=%zu rope_set=%zu tbl=%p sim=%p\n",
                                ctx->lasso_points.size(), rope_set.size(), (void*)tbl, (void*)sim);

                            int first_rope_for_ring = -1;
                            int first_vid_for_ring = -1;
                            for (int rope_idx : rope_set) {
                                int vc = rope_sim_get_vertex_count(sim, rope_idx);
                                if (vc <= 1) continue;
                                std::vector<float> proj_xy(static_cast<size_t>(vc * 2));
                                int got = gp_table_get_projected_rope_vertices(tbl, rope_idx, proj_xy.data(), static_cast<int>(proj_xy.size()));
                                if (got != vc) continue;
                                // for each straight physics segment, test against lasso segments
                                for (int ri = 0; ri < vc - 1; ++ri) {
                                    float rx0 = proj_xy[2*ri+0]; float ry0 = proj_xy[2*ri+1];
                                    float rx1 = proj_xy[2*(ri+1)+0]; float ry1 = proj_xy[2*(ri+1)+1];
                                    for (size_t li = 0; li + 1 < lasso_local.size(); ++li) {
                                        float lx0 = lasso_local[li].first; float ly0 = lasso_local[li].second;
                                        float lx1 = lasso_local[li+1].first; float ly1 = lasso_local[li+1].second;
                                        float ix=0, iy=0;
                                        if (seg_intersect(rx0,ry0,rx1,ry1,lx0,ly0,lx1,ly1,&ix,&iy)) {
                                                float d0 = (ix-rx0)*(ix-rx0)+(iy-ry0)*(iy-ry0);
                                                float d1 = (ix-rx1)*(ix-rx1)+(iy-ry1)*(iy-ry1);
                                                // compute fractional parameter along projected segment
                                                float sx = rx1 - rx0; float sy = ry1 - ry0;
                                                float denom = sx*sx + sy*sy;
                                                float t = 0.0f;
                                                if (denom > 1e-8f) t = ((ix - rx0) * sx + (iy - ry0) * sy) / denom;
                                                if (t < 0.0f) t = 0.0f; if (t > 1.0f) t = 1.0f;
                                                int new_vid = gp_table_rope_insert_vertex(tbl, static_cast<int32_t>(rope_idx), ri, t);
                                                // make meta-group lazily when first vertex is about to be added
                                                if (!mg) mg = gp_table_meta_create(tbl);
                                                if (new_vid < 0) {
                                                    int pick = (d0 <= d1) ? ri : (ri+1);
                                                    // Use canonical canvas lookup for persistent id (local -> up -> BFS)
                                                    uint64_t persistent_id = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, rope_idx);
                                                    if (persistent_id == 0ull) {
                                                        printf("lasso: ERROR: intersected rope has no persistent id; aborting lasso (tbl=%p rope_idx=%d)\n", (void*)tbl, rope_idx);
                                                        if (mg) { gp_table_meta_destroy(tbl, mg); mg = nullptr; }
                                                        goto lasso_abort;
                                                    }
                                                    gp_table_meta_add_vertex(tbl, mg, static_cast<int32_t>(rope_idx), static_cast<int32_t>(pick));
                                                    printf("lasso: added existing vertex rope=%d vid=%d pid=%llu to mg=%p at ix=%.2f,iy=%.2f (tbl_mod=%d)\n", rope_idx, pick, (unsigned long long)persistent_id, (void*)mg, ix, iy, host_mod);
                                                    if (first_rope_for_ring < 0) { first_rope_for_ring = rope_idx; first_vid_for_ring = pick; }
                                                } else {
                                                    // Use canonical canvas lookup for persistent id (local -> up -> BFS)
                                                    uint64_t persistent_id = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, rope_idx);
                                                    if (persistent_id == 0ull) {
                                                        printf("lasso: ERROR: intersected rope has no persistent id; aborting lasso (tbl=%p rope_idx=%d)\n", (void*)tbl, rope_idx);
                                                        if (mg) { gp_table_meta_destroy(tbl, mg); mg = nullptr; }
                                                        goto lasso_abort;
                                                    }
                                                    gp_table_meta_add_vertex(tbl, mg, static_cast<int32_t>(rope_idx), static_cast<int32_t>(new_vid));
                                                    printf("lasso: inserted vertex rope=%d vid=%d pid=%llu to mg=%p at t=%.3f ix=%.2f,iy=%.2f (tbl_mod=%d)\n", rope_idx, new_vid, (unsigned long long)persistent_id, (void*)mg, t, ix, iy, host_mod);
                                                    if (first_rope_for_ring < 0) { first_rope_for_ring = rope_idx; first_vid_for_ring = new_vid; }
                                                }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
            lasso_abort:
                            if (mg) {
                                // If no vertices were found, destroy the empty meta-group
                                int mg_vcount = gp_table_meta_get_vertex_count(tbl, mg);
                                // Diagnostic dump: verify mg resides in ctx and list vertices
                                int sim_idx = -999;
                                gp_table_meta_get_sim_group_index(tbl, mg, &sim_idx);
                                printf("lasso: pre-destroy-check mg=%p vcount=%d sim_group_idx=%d tbl=%p\n", (void*)mg, mg_vcount, sim_idx, (void*)tbl);
                                if (mg_vcount > 0) {
                                    for (int vi = 0; vi < mg_vcount; ++vi) {
                                        int r = -1, v = -1;
                                        if (gp_table_meta_get_vertex(tbl, mg, vi, &r, &v)) {
                                            printf("  mg vertex[%d] = rope=%d vert=%d\n", vi, r, v);
                                        } else {
                                            printf("  mg vertex[%d] = <failed to read>\n", vi);
                                        }
                                    }
                                }
                                // Recompute vcount directly from internal storage as a cross-check
                                int direct_vcount = 0;
                                if (tbl) {
                                    // find mg in ctx meta_groups and inspect directly
                                    bool found = false;
                                    for (int mi = 0; mi < gp_table_get_meta_group_count(tbl); ++mi) {
                                        GP_MetaGroup* mg2 = gp_table_get_meta_group(tbl, mi);
                                        if (mg2 == mg) { found = true; break; }
                                    }
                                    printf("  mg present_in_ctx=%d\n", found ? 1 : 0);
                                }
                                // If there are truly no vertices, destroy the meta-group
                                if (mg_vcount == 0) {
                                    gp_table_meta_destroy(tbl, mg);
                                    printf("lasso: destroyed empty meta-group ptr=%p\n", (void*)mg);
                                }
                                else {
                                    // If we detected a preferred first intersection, record it
                                    // on the meta-group so widget creation will use the correct anchor.
                                    if (first_rope_for_ring >= 0 && first_vid_for_ring >= 0) {
                                        gp_table_meta_set_anchor(tbl, mg, first_rope_for_ring, first_vid_for_ring);
                                    }
                                    // Ensure the meta-group carries lasso config so it
                                    // serializes/restores properly. Use the canvas' current
                                    // subgroup flags as the lasso flags and pick a sensible
                                    // non-zero default widget type so dangling widgets
                                    // get created with a visible widget.
                                    int default_widget_type = 1; // non-zero default
                                    gp_table_meta_set_lasso_fields(tbl, mg, ctx->selected_tool_subgroup_flags, default_widget_type);
                                    // Create a dangling widget attached to the meta-group (conceptual object).
                                    // Do this before enabling edge-springs so the T-off center is
                                    // a genuine member when the sim builds edge-springs.
                                    if (gp_table_meta_create_widget(tbl, mg)) {
                                        printf("lasso: created dangling widget for mg=%p\n", (void*)mg);
                                    }
                                    // Enable edge-springs connecting the found vertices so
                                    // the set compresses over time. Use conservative
                                    // defaults: minimum rest length and a linear reduction
                                    // rate (units per second).
                                    float min_rest = 2.0f;
                                    float reduce_rate = 50.0f;
                                    if (gp_table_meta_enable_edge_springs(tbl, mg, min_rest, reduce_rate)) {
                                        printf("lasso: enabled edge-springs for mg=%p min_rest=%.2f rate=%.2f\n", (void*)mg, min_rest, reduce_rate);
                                    }
                                    // Create a ring at the first added rope/vertex so the renderer will show a small sampled ring.
                                    // Prefer registering the ring on the root table when both tables share the same RopeSim
                                    int first_rope_for_ring_local = first_rope_for_ring;
                                    int first_vid_for_ring_local = first_vid_for_ring;
                                    if (first_rope_for_ring_local >= 0 && first_vid_for_ring_local >= 0) {
                                        canvas_finalize_lasso_meta_group(ctx, tbl, mg, first_rope_for_ring_local, first_vid_for_ring_local, 0.0f);
                                    }
                                }
                                // dispatch lasso-end event with collected points
                                if (ctx->lasso_cb && !ctx->lasso_points.empty()) {
                                    int n = static_cast<int>(ctx->lasso_points.size());
                                    std::vector<float> pts(static_cast<size_t>(n) * 2);
                                    for (int i = 0; i < n; ++i) { pts[2*i+0] = ctx->lasso_points[static_cast<size_t>(i)].first; pts[2*i+1] = ctx->lasso_points[static_cast<size_t>(i)].second; }
                                    ctx->lasso_cb(ctx->lasso_cb_user, 3, pts.data(), n);
                                }
                    }
                }
            }
            // Clear any prospective rope shown on the table
            GP_TableContext* tbl_clear = ctx->container_table;
            if (!tbl_clear && ctx->root_module_idx >= 0 && ctx->root_module_idx < static_cast<int>(ctx->module_tables.size())) {
                tbl_clear = ctx->module_tables[ctx->root_module_idx];
            }
            if (tbl_clear && ctx->prospective_rope_idx >= 0) {
                gp_table_set_prospective_rope_index(tbl_clear, -1);
                /* cleared prospective rope log removed */
                ctx->prospective_rope_idx = -1;
            }
            ctx->lasso_points.clear();
        } else {
            // mouse move while lasso mode enabled and not yet released: sample
            // only when movement exceeds threshold to avoid excessive points.
            const float min_dist_sq = 4.0f; // ~2 pixels
            if (!ctx->lasso_points.empty()) {
                float dx = fx - ctx->lasso_points.back().first;
                float dy = fy - ctx->lasso_points.back().second;
                if (dx*dx + dy*dy >= min_dist_sq) {
                    ctx->lasso_points.emplace_back(fx, fy);
                    /* lasso sample log removed */
                    if (ctx->lasso_cb) { float pts[2] = { fx, fy }; ctx->lasso_cb(ctx->lasso_cb_user, 2, pts, 1); }
                    // update prospective rope endpoints if present
                    if (ctx->prospective_rope_idx >= 0) {
                        GP_TableContext* tbl_upd = ctx->container_table;
                        if (!tbl_upd && ctx->root_module_idx >= 0 && ctx->root_module_idx < static_cast<int>(ctx->module_tables.size())) {
                            tbl_upd = ctx->module_tables[ctx->root_module_idx];
                        }
                        if (tbl_upd) {
                            RopeSim* sim = gp_table_get_rope_sim(tbl_upd);
                            if (sim) {
                                // compute table-local offsets same as creation
                                float table_off_x = 0.0f, table_off_y = 0.0f;
                                int host_mod = -1;
                                for (int mi = 0; mi < static_cast<int>(ctx->module_tables.size()); ++mi) {
                                    if (ctx->module_tables[mi] == tbl_upd) { host_mod = mi; break; }
                                }
                                if (host_mod >= 0) {
                                    const auto &m = ctx->modules[host_mod];
                                    int top_h = std::min(m.h, kModuleTopUiHeight);
                                    table_off_x = static_cast<float>(m.x);
                                    table_off_y = static_cast<float>(m.y + top_h);
                                }
                                float sx_local = ctx->lasso_points.front().first - table_off_x;
                                float sy_local = ctx->lasso_points.front().second - table_off_y;
                                float fx_local = fx - table_off_x;
                                float fy_local = fy - table_off_y;
                                float plug_z = -10.0f;
                                rope_sim_move_endpoints3(sim, ctx->prospective_rope_idx, sx_local, sy_local, plug_z, fx_local, fy_local, plug_z);
                            }
                        }
                    }
                }
            } else {
                ctx->lasso_points.emplace_back(fx, fy);
                /* lasso sample log removed */
                if (ctx->lasso_cb) { float pts[2] = { fx, fy }; ctx->lasso_cb(ctx->lasso_cb_user, 2, pts, 1); }
            }
        }
        // still forward events to module listeners below
    }
    // Forward events to module listeners: keep `module_input_state` updated
    // so built-in MouseListener tool instances receive live mouse values.
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

// Finalize a meta-group after vertices have been added: set anchor, create
// dangling widget, enable edge-springs, and create/register a ring using the
// same logic as the interactive lasso. This does NOT dispatch lasso callbacks.
static void canvas_finalize_lasso_meta_group(GP_CanvasContextImpl* ctx, GP_TableContext* tbl, GP_MetaGroup* mg, int first_rope_for_ring_local, int first_vid_for_ring_local, float saved_u_override) {
    if (!ctx || !tbl || !mg) return;
    int mg_vcount = gp_table_meta_get_vertex_count(tbl, mg);
    int sim_idx = -999; gp_table_meta_get_sim_group_index(tbl, mg, &sim_idx);
    unsigned long long mgid = 0ull; gp_table_meta_get_id(tbl, mg, &mgid);
    printf("canvas_finalize_lasso_meta_group: mg=%p id=%llu vcount=%d sim_group_idx=%d tbl=%p\n", (void*)mg, (unsigned long long)mgid, mg_vcount, sim_idx, (void*)tbl);
    if (sim_idx >= 0 && mgid != 0ull) {
        ctx->lasso_id_map[mgid] = sim_idx;
        printf("canvas_finalize_lasso_meta_group: registered lasso_id=%llu -> sim_idx=%d\n", (unsigned long long)mgid, sim_idx);
    }
    if (mg_vcount == 0) {
        gp_table_meta_destroy(tbl, mg);
        printf("canvas_finalize_lasso_meta_group: destroyed empty meta-group ptr=%p\n", (void*)mg);
        return;
    }
    // Log vertices for diagnostics
    for (int vi = 0; vi < mg_vcount; ++vi) {
        int r = -1, v = -1;
        if (gp_table_meta_get_vertex(tbl, mg, vi, &r, &v)) {
            printf("  mg vertex[%d] = rope=%d vert=%d\n", vi, r, v);
        } else {
            printf("  mg vertex[%d] = <failed to read>\n", vi);
        }
    }
    // Dump lasso/meta-group full state
    gp_table_debug_dump_meta_group(tbl, mg, "finalize");
    // If we detected a preferred first intersection, record it as anchor
    if (first_rope_for_ring_local >= 0 && first_vid_for_ring_local >= 0) {
        gp_table_meta_set_anchor(tbl, mg, first_rope_for_ring_local, first_vid_for_ring_local);
    }
    // Create dangling widget attached to the meta-group
    if (gp_table_meta_create_widget(tbl, mg)) {
        printf("canvas_finalize_lasso_meta_group: created dangling widget for mg=%p\n", (void*)mg);
    }
    // Enable edge-springs
    float min_rest = 2.0f;
    float reduce_rate = 50.0f;
    // store spring params on the meta-group so they persist through save/load
    gp_table_meta_set_edge_spring_params(tbl, mg, min_rest, reduce_rate, 0);
    if (gp_table_meta_enable_edge_springs(tbl, mg, min_rest, reduce_rate)) {
        printf("canvas_finalize_lasso_meta_group: enabled edge-springs for mg=%p min_rest=%.2f rate=%.2f\n", (void*)mg, min_rest, reduce_rate);
    }
    // Create/register ring if we have a rope+vertex
    if (first_rope_for_ring_local >= 0 && first_vid_for_ring_local >= 0) {
        canvas_create_and_register_ring(ctx, tbl, mg, first_rope_for_ring_local, first_vid_for_ring_local, saved_u_override);
    }
}

// Helper: create an overlay at given canvas coords, ensure the table has a
// RopeSim, add a rope in that sim at the same endpoints, attach the rope to
// the overlay via the canvas API, and return the rope index (or -1).
static int canvas_create_overlay_and_attach(GP_CanvasContextImpl* c, GP_TableContext* t, float ox1, float oy1, float ox2, float oy2, unsigned long long* out_key_a, unsigned long long* out_key_b) {
    if (!c || !t) return -1;
    RopeSim* sim = gp_table_get_rope_sim(t);
    if (!sim) {
        RopeSim* rootsim = canvas_root_sim(c);
        if (!rootsim) rootsim = canvas_require_root_sim(c);
        if (rootsim) gp_table_attach_rope_sim(t, rootsim, 0);
        sim = gp_table_get_rope_sim(t);
    }
    int rope_idx = -1;
    if (sim) {
        float table_off_x = 0.0f, table_off_y = 0.0f;
        int host_mod = -1;
        for (int mi = 0; mi < static_cast<int>(c->module_tables.size()); ++mi) {
            if (c->module_tables[mi] == t) { host_mod = mi; break; }
        }
        if (host_mod >= 0) {
            const auto &m = c->modules[host_mod];
            int top_h = std::min(m.h, kModuleTopUiHeight);
            table_off_x = static_cast<float>(m.x);
            table_off_y = static_cast<float>(m.y + top_h);
        }
        float sx_local = ox1 - table_off_x;
        float sy_local = oy1 - table_off_y;
        float fx_local = ox2 - table_off_x;
        float fy_local = oy2 - table_off_y;
        float plug_z = -10.0f;
        int segs = 2;
        rope_idx = rope_sim_add_rope3(sim, sx_local, sy_local, plug_z, fx_local, fy_local, plug_z, segs, 0.0f);
    }
    unsigned long long ka = 0ull, kb = 0ull;
    if (gp_canvas_create_overlay_with_leds(reinterpret_cast<GP_CanvasContext*>(c), ox1, oy1, ox2, oy2, &ka, &kb)) {
        if (out_key_a) *out_key_a = ka;
        if (out_key_b) *out_key_b = kb;
        if (rope_idx >= 0) gp_canvas_attach_rope_to_overlay(reinterpret_cast<GP_CanvasContext*>(c), ka, kb, rope_idx);
    }
    return rope_idx;
}

// Helper: from first rope+vertex metadata, create a ring using the same
// logic as the interactive lasso: prefer registering on the root table if
// it shares the same RopeSim, otherwise register on the provided table.
static void canvas_create_and_register_ring(GP_CanvasContextImpl* c, GP_TableContext* tbl, GP_MetaGroup* mg, int first_rope_for_ring_local, int first_vid_for_ring_local, float saved_u_override) {
    if (!c || !tbl || !mg) return;
    if (first_rope_for_ring_local < 0 || first_vid_for_ring_local < 0) return;
    GP_TableContext* root_tbl = c->container_table;
    GP_TableContext* reg_tbl = nullptr;
    RopeSim* tbl_sim = gp_table_get_rope_sim(tbl);
    RopeSim* root_sim = root_tbl ? gp_table_get_rope_sim(root_tbl) : nullptr;
    if (root_tbl && root_sim && tbl_sim && root_sim == tbl_sim) reg_tbl = root_tbl;
    else reg_tbl = tbl;
    if (!reg_tbl) return;
    RopeSim* reg_sim = gp_table_get_rope_sim(reg_tbl);
    int vc = rope_sim_get_vertex_count(reg_sim, first_rope_for_ring_local);
    if (vc > 1) {
        float u = saved_u_override;
        if (!(u > 0.0f && u <= 1.0f)) {
            // Avoid placing the ring exactly on a discrete vertex index.
            // Use the midpoint between the vertex and the next vertex so the
            // ring sits on the segment, not coincident with the vertex.
            float fidx = static_cast<float>(first_vid_for_ring_local) + 0.5f;
            u = fidx / static_cast<float>(vc - 1);
            if (u <= 0.0f) u = 0.001f;
            if (u >= 1.0f) u = 0.999f;
        }
        int ring_id = gp_table_create_ring(reg_tbl, first_rope_for_ring_local, u);
        if (ring_id >= 0) {
            unsigned long long mgid = 0ull;
            if (!gp_table_meta_get_id(reg_tbl, mg, &mgid)) mgid = 0ull;
            gp_table_register_ring_edge(reg_tbl, ring_id, mgid);
            printf("canvas_create_and_register_ring: created ring id=%d on rope=%d u=%.3f registered mgid=%llu\n", ring_id, first_rope_for_ring_local, u, (unsigned long long)mgid);
        }
    }
}
static inline int table_hit_contact_index(const GP_TableHitBox& hb) {
    if (hb.row_idx >= 0) return hb.row_idx;
    if (hb.part == GP_TABLE_HIT_LED_TABLE) return hb.aux1;
    return hb.aux0;
}

static int resolve_contact_index(const GP_CanvasContextImpl* ctx, int module_idx, const GP_TableHitBox& hb) {
    if (!ctx) return table_hit_contact_index(hb);
    const std::vector<ModuleIORow>* rows_ptr = nullptr;
    bool using_table_rows = false;
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_table_rows.size()) &&
        !ctx->module_table_rows[module_idx].empty()) {
        rows_ptr = &ctx->module_table_rows[module_idx];
        using_table_rows = true;
    } else if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_io_rows.size())) {
        rows_ptr = &ctx->module_io_rows[module_idx];
    }
    if (rows_ptr) {
        const auto &rows = *rows_ptr;
        if (hb.row_idx >= 0 && hb.row_idx < static_cast<int>(rows.size())) {
            const ModuleIORow &meta = rows[hb.row_idx];
            if (meta.kind == ModuleRowKind::Tool) return -1;
            if (using_table_rows) {
                bool is_left = (hb.col_idx == kModuleColLeftLed);
                bool is_right = (hb.col_idx == kModuleColRightLed);
                if (meta.kind == ModuleRowKind::Input && !is_left) return -1;
                if (meta.kind == ModuleRowKind::Output && !is_right) return -1;
                if (!is_left && !is_right) return -1;
            }
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

struct ModuleLayout {
    int top_h = 0;
    int table_y = 0;
    int table_clip_h = 0;
    int preview_h = 0;
    int preview_y = 0;
    int drag_y = 0;
    int drag_h = 0;
};

static ModuleLayout module_layout_for(const GP_CanvasModuleDesc& m) {
    ModuleLayout layout{};
    if (m.w <= 0 || m.h <= 0) return layout;
    layout.top_h = std::min(m.h, kModuleTopUiHeight);
    layout.table_y = layout.top_h;
    int available_h = std::max(0, m.h - layout.top_h);
    int preview_h = std::min(m.w, available_h);
    if ((available_h - preview_h) < kModulePreviewMinTableHeight) {
        preview_h = std::max(0, available_h - kModulePreviewMinTableHeight);
        preview_h = std::min(preview_h, m.w);
    }
    layout.preview_h = std::max(0, preview_h);
    layout.preview_y = std::max(0, m.h - layout.preview_h);
    layout.table_clip_h = std::max(1, layout.preview_y - layout.table_y);
    layout.drag_y = std::min(layout.top_h, kModuleTopPadding + kModuleTitleRowH);
    layout.drag_h = std::max(0, std::min(kModuleThumbRowH, layout.top_h - layout.drag_y));
    return layout;
}

static ModuleLayout module_layout_for(const GP_CanvasContextImpl* ctx, int module_idx, const GP_CanvasModuleDesc& m) {
    ModuleLayout layout = module_layout_for(m);
    bool is_stage = ctx && module_idx >= 0 && module_idx < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[module_idx];
    if (is_stage) {
        layout.top_h = 0;
        layout.table_y = 0;
        layout.preview_h = 0;
        layout.preview_y = std::max(0, m.h);
        layout.table_clip_h = std::max(1, m.h);
        layout.drag_y = 0;
        layout.drag_h = std::min(kModuleThumbRowH, std::max(0, m.h));
    }
    return layout;
}

static GP_TableCell* module_frame_led_cell(GP_CanvasContextImpl* ctx, int module_idx, int row, int idx) {
    if (!ctx) return nullptr;
    if (row < 0 || row >= kModuleExtraLedRows) return nullptr;
    if (idx < 0 || idx >= kModuleExtraLedCount) return nullptr;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_frame_leds.size())) return nullptr;
    auto &group = ctx->module_frame_leds[module_idx];
    return &group.cells[static_cast<size_t>(row)][static_cast<size_t>(idx)];
}

static void module_stack_tail_write(GP_CanvasContextImpl* ctx, int module_idx, const float* values, int count) {
    if (!ctx || module_idx < 0) return;
    if (module_idx >= static_cast<int>(ctx->module_stack_tail.size())) return;
    ModuleStackTail &tail = ctx->module_stack_tail[module_idx];
    uint32_t seq = tail.seq.load(std::memory_order_relaxed);
    tail.seq.store(seq + 1, std::memory_order_release);
    int to_copy = std::clamp(count, 0, static_cast<int>(tail.values.size()));
    for (int i = 0; i < to_copy; ++i) {
        tail.values[static_cast<size_t>(i)] = values ? values[i] : 0.0f;
    }
    tail.count.store(to_copy, std::memory_order_release);
    tail.seq.store(seq + 2, std::memory_order_release);
}

static void module_stack_tail_read(const GP_CanvasContextImpl* ctx, int module_idx, float* out_vals, int* out_count) {
    if (out_count) *out_count = 0;
    if (!ctx || !out_vals || !out_count) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_stack_tail.size())) return;
    const ModuleStackTail &tail = ctx->module_stack_tail[module_idx];
    for (int attempt = 0; attempt < 3; ++attempt) {
        uint32_t start = tail.seq.load(std::memory_order_acquire);
        if (start & 1u) continue;
        int count = tail.count.load(std::memory_order_acquire);
        int to_copy = std::clamp(count, 0, static_cast<int>(tail.values.size()));
        for (int i = 0; i < to_copy; ++i) {
            out_vals[i] = tail.values[static_cast<size_t>(i)];
        }
        uint32_t end = tail.seq.load(std::memory_order_acquire);
        if (start == end && !(end & 1u)) {
            *out_count = to_copy;
            return;
        }
    }
}

static void build_module_preview_input(const GP_CanvasContextImpl* ctx, int module_idx, GP_ModulePreviewInput &out) {
    std::memset(&out, 0, sizeof(out));
    if (!ctx) return;
    out.rows = 0;
    out.cols = 0;
    out.layout_strategy = 0;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_frame_links.size())) return;
    const auto &links = ctx->module_frame_links[module_idx];
    // populate preview with the first send/receive logical rows (left column)
    for (int i = 0; i < kModuleExtraLedCount; ++i) {
        out.send_ptrs[i] = links.ptrs[0][static_cast<size_t>(i)];
        out.receive_ptrs[i] = links.ptrs[2][static_cast<size_t>(i)];
    }
    module_stack_tail_read(ctx, module_idx, out.stack_tail, &out.stack_tail_count);
}

static void draw_module_top_ui(GP_CanvasContextImpl* ctx, int module_idx, const GP_CanvasModuleDesc& m, uint8_t* out_rgba, int w, int h, int pitch) {
    if (!ctx || !out_rgba) return;
    ModuleLayout layout = module_layout_for(ctx, module_idx, m);
    if (layout.top_h <= 0) return;
    int sx = m.x - ctx->offset_x;
    int sy = m.y - ctx->offset_y;
    int top_h = layout.top_h;
    memset_rect(out_rgba, w, h, pitch, sx, sy, m.w, top_h, Color{34,34,46,235});
    memset_rect(out_rgba, w, h, pitch, sx, sy + top_h - 1, m.w, 1, Color{18,18,26,255});

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

    int cursor_y = sy + kModuleTopPadding;
    const char* title = (m.label[0] != '\0') ? m.label : "Module";
    blit_text(title, sx + kModuleTopPadding, cursor_y, 1.05f, Color{220,220,230,255});
    cursor_y += kModuleTitleRowH;

    int thumb_y = cursor_y;
    int thumb_x = sx + kModuleTopPadding;
    int thumb_w = std::max(1, m.w - kModuleTopPadding * 2);
    memset_rect(out_rgba, w, h, pitch, thumb_x, thumb_y, thumb_w, kModuleThumbRowH, Color{40,40,52,255});
    for (int i = 0; i < 3; ++i) {
        int bar_w = thumb_w / 4;
        int bar_x = thumb_x + (thumb_w - bar_w) / 2;
        int bar_y = thumb_y + 3 + i * 4;
        memset_rect(out_rgba, w, h, pitch, bar_x, bar_y, bar_w, 2, Color{70,70,88,255});
    }
    cursor_y += kModuleThumbRowH + kModuleTopGap;

    int control_y = cursor_y;
    int control_h = std::max(1, kModuleControlRowH - 2);
    int gap = 6;
    auto draw_button = [&](int bx, int by, int bw, int bh, Color fill, const char* label, float scale) {
        memset_rect(out_rgba, w, h, pitch, bx, by, bw, bh, fill);
        memset_rect(out_rgba, w, h, pitch, bx, by, bw, 1, Color{20,20,28,255});
        memset_rect(out_rgba, w, h, pitch, bx, by + bh - 1, bw, 1, Color{12,12,18,255});
        if (label && label[0] != '\0') {
            auto bm = render_text_to_rgba(label, scale, {230,230,235,255});
            if (!bm.pixels.empty()) {
                int tx = bx + (bw - bm.width) / 2;
                int ty = by + (bh - bm.height) / 2;
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
    };

    int group_x = sx + kModuleTopPadding;
    int nbw = control_h;
    int num_w = std::max(40, control_h * 2);
    int btn_y = control_y + 1;
    draw_button(group_x, btn_y, nbw, control_h, Color{52,52,64,255}, LABEL_IO_MINUS, 1.0f);
    draw_button(group_x + nbw + gap, btn_y, num_w, control_h, Color{36,36,46,255}, "0", 0.9f);
    draw_button(group_x + nbw + gap + num_w + gap, btn_y, nbw, control_h, Color{52,52,64,255}, LABEL_IO_PLUS, 1.0f);

    int right_x = sx + m.w - kModuleTopPadding;
    int menu_w = std::max(30, control_h);
    int pause_w = std::max(42, control_h * 2);
    int menu_x = right_x - menu_w;
    int pause_x = menu_x - gap - pause_w;
    // Per-module play/pause: this button toggles the entire-module skip flag.
    bool module_paused = false;
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_skip.size())) module_paused = (ctx->module_skip[module_idx] != 0);
    else module_paused = ctx->thread_mgr_paused;
    const char* pause_label = module_paused ? LABEL_THREAD_PLAY_SHORT : LABEL_THREAD_PAUSE_SHORT;
    // Split the pause/play button: left half = module SIM toggle, right half = module play/pause
    int half_play_w = std::max(8, pause_w / 2);
    int sim_x = pause_x;
    int sim_w = half_play_w;
    int play_x = pause_x + half_play_w;
    int play_w = pause_w - half_play_w;
    bool module_sim = false;
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_sim_enabled.size())) module_sim = (ctx->module_sim_enabled[module_idx] != 0);
    const char* sim_label = module_sim ? "SIM" : "sim";
    // draw left SIM half
    draw_button(sim_x, btn_y, sim_w, control_h, Color{46,46,56,255}, sim_label, 0.8f);
    // draw right play/pause half
    draw_button(play_x, btn_y, play_w, control_h, Color{44,52,60,255}, pause_label, 1.0f);
    draw_button(menu_x, btn_y, menu_w, control_h, Color{50,50,62,255}, LABEL_MODULE_MENU, 0.85f);

    // Module action buttons: Clone / Clear / Destroy / Commit / Export (right-aligned)
    int module_btn_count = 5;
    int module_btn_w = nbw;
    int module_gap = 6;
    int btns_total_w = module_btn_count * (module_btn_w + module_gap) - module_gap;
        int btns_right = pause_x - module_gap;
        int btns_left = btns_right - btns_total_w;
        int bx_btn = btns_left;
    draw_button(bx_btn, btn_y, module_btn_w, control_h, Color{52,52,64,255}, LABEL_MODULE_CLONE_SHORT, 0.9f); bx_btn += module_btn_w + module_gap;
    draw_button(bx_btn, btn_y, module_btn_w, control_h, Color{52,52,64,255}, LABEL_MODULE_CLEAR_SHORT, 0.9f); bx_btn += module_btn_w + module_gap;
    draw_button(bx_btn, btn_y, module_btn_w, control_h, Color{52,52,64,255}, LABEL_MODULE_DESTROY_SHORT, 0.9f); bx_btn += module_btn_w + module_gap;
    draw_button(bx_btn, btn_y, module_btn_w, control_h, Color{48,48,56,255}, "CMT", 0.9f); bx_btn += module_btn_w + module_gap;
    draw_button(bx_btn, btn_y, module_btn_w, control_h, Color{44,60,48,255}, "EXP", 0.9f);

    cursor_y += kModuleControlRowH + kModuleTopGap;
    constexpr float kLedLabelScale = 0.7f;
    constexpr int kLedLabelGap = 6;
    // Four logical LED sets arranged as 2 columns x 2 rows: top row = "send", bottom row = "receive"
    const int full_cols = 2; // left/right
    int visible_total = ctx ? std::clamp(ctx->module_frame_pair_count, 1, kModuleExtraLedCount * full_cols) : kModuleExtraLedCount * full_cols; // total pair-columns across both sides
    int pair_count = (kModuleExtraLedRows + full_cols - 1) / full_cols; // logical label rows (send/receive)
    std::vector<const char*> led_labels_vec;
    led_labels_vec.reserve(static_cast<size_t>(pair_count));
    for (int gi = 0; gi < pair_count; ++gi) {
        // row 0 = send, row 1 = receive
        led_labels_vec.push_back((gi == 0) ? "send" : "receive");
    }
    std::vector<TextBitmap> label_bitmaps(static_cast<size_t>(pair_count));
    int label_w = 0;
    for (int i = 0; i < pair_count; ++i) {
        label_bitmaps[static_cast<size_t>(i)] = render_text_to_rgba(led_labels_vec[static_cast<size_t>(i)], kLedLabelScale, {190,190,205,255});
        label_w = std::max(label_w, label_bitmaps[static_cast<size_t>(i)].width);
    }
    const int label_pad = (label_w > 0) ? kLedLabelGap : 0;
    int led_row_h = kModuleLedRowH;
    const int grid_rows = pair_count;
    // reserve space for the numeric control on the left, then divide remaining width
    int ctrl_nbw = nbw;
    int ctrl_num_w = num_w;
    int ctrl_gap = gap;
    int ctrl_total_w = ctrl_nbw + ctrl_gap + ctrl_num_w + ctrl_gap + ctrl_nbw;
    int remaining_w = std::max(1, m.w - kModuleTopPadding * 2 - ctrl_total_w);
    int col_total_w = std::max(1, remaining_w / full_cols);
    int led_gap = 4;
    // visible per-side counts derived from total
    int visible_on_left = std::clamp(visible_total, 0, kModuleExtraLedCount);
    int visible_on_right = std::clamp(visible_total - kModuleExtraLedCount, 0, kModuleExtraLedCount);
    int max_visible_side = std::max(visible_on_left, visible_on_right);
    int led_area_inner_w = std::max(1, col_total_w - label_w - label_pad - kModuleTopPadding);
    int led_size = (led_area_inner_w - led_gap * std::max(0, max_visible_side - 1)) / std::max(1, max_visible_side);
    if (led_size < 8) {
        led_gap = 2;
        led_size = (led_area_inner_w - led_gap * std::max(0, max_visible_side - 1)) / std::max(1, max_visible_side);
    }
    led_size = std::max(4, std::min(led_size, led_row_h - 4));
    Color led_on{255,210,90,255};
    Color led_off{70,70,80,255};
    Color led_edge{20,20,28,255};
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_tables.size()) && ctx->module_tables[module_idx]) {
        GP_TableStyle st{};
        gp_table_get_style(ctx->module_tables[module_idx], &st);
        led_on = Color{st.led_on_rgba[0], st.led_on_rgba[1], st.led_on_rgba[2], st.led_on_rgba[3]};
        led_off = Color{st.led_off_rgba[0], st.led_off_rgba[1], st.led_off_rgba[2], st.led_off_rgba[3]};
        led_edge = Color{st.led_edge_rgba[0], st.led_edge_rgba[1], st.led_edge_rgba[2], st.led_edge_rgba[3]};
    }
    int total_rows = pair_count * full_cols;
    // draw single pair-count controls (left of left column) once
    {
        int group_x = sx + kModuleTopPadding;
        int nbw = control_h;
        int num_w = std::max(24, nbw * 2);
        int btn_y = cursor_y + (led_row_h - control_h) / 2;
        draw_button(group_x, btn_y, nbw, control_h, Color{52,52,64,255}, "-", 1.0f);
        // render count label
        auto bm_count = render_text_to_rgba(std::to_string(visible_total).c_str(), 0.9f, {230,230,235,255});
        if (!bm_count.pixels.empty()) {
            int tx = group_x + nbw + gap + (num_w - bm_count.width) / 2;
            int ty = btn_y + (control_h - bm_count.height) / 2;
            for (int yy = 0; yy < bm_count.height; ++yy) {
                int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < bm_count.width; ++xx) {
                    int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &bm_count.pixels[(yy * bm_count.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
        draw_button(group_x + nbw + gap + num_w + gap, btn_y, nbw, control_h, Color{52,52,64,255}, "+", 1.0f);
    }

    for (int row = 0; row < total_rows; ++row) {
        int grid_row = row / full_cols;
        int grid_col = row % full_cols;
        int led_row_y = cursor_y + grid_row * (kModuleLedRowH + kModuleTopGap);
        // draw label only for left column (grid_col == 0) and reuse per-grid_row bitmap
        if (grid_col == 0) {
            const auto &bm = label_bitmaps[static_cast<size_t>(grid_row)];
            int label_x = sx + kModuleTopPadding + ctrl_total_w + grid_col * col_total_w;
            if (!bm.pixels.empty()) {
                int label_y = led_row_y + (led_row_h - bm.height) / 2;
                for (int yy = 0; yy < bm.height; ++yy) {
                    int dst_y = label_y + yy;
                    if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < bm.width; ++xx) {
                        int dst_x = label_x + xx;
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
        // left-aligned groups: compute left base then place right group immediately after left group's LEDs
        int left_base_x = sx + kModuleTopPadding + ctrl_total_w + /*label space*/ (label_w + label_pad);
        int left_led_total = led_size * visible_on_left + led_gap * std::max(0, visible_on_left - 1);
        int visible_here = (grid_col == 0) ? visible_on_left : visible_on_right;
        int led_start_x = (grid_col == 0) ? left_base_x : (left_base_x + left_led_total);
        int led_y = led_row_y + (led_row_h - led_size) / 2;
        for (int i = 0; i < visible_here; ++i) {
            int lx = led_start_x + i * (led_size + led_gap);
            const GP_TableCell* cell = module_frame_led_cell(ctx, module_idx, row, i);
            bool on = cell && ((cell->flags & 0x1u) != 0u);
            bool active = cell && ((static_cast<uint32_t>(cell->reserved0) & 0x1u) != 0u);
            Color fill = (on || active) ? led_on : led_off;
            int cx = lx + led_size / 2;
            int cy = led_y + led_size / 2;
            int radius = std::max(2, led_size / 2 - 1);
            draw_circle(out_rgba, w, h, pitch, cx, cy, radius, led_edge);
            draw_circle(out_rgba, w, h, pitch, cx, cy, std::max(1, radius - 1), fill);
        }
    }
}

template <typename Fn>
static void for_each_module_frame_led(const GP_CanvasContextImpl* ctx, int module_idx, const GP_CanvasModuleDesc& m, Fn&& fn) {
    if (!ctx) return;
    ModuleLayout layout = module_layout_for(ctx, module_idx, m);
    if (layout.top_h <= 0) return;
    int cursor_y = kModuleTopPadding;
    cursor_y += kModuleTitleRowH;
    cursor_y += kModuleThumbRowH + kModuleTopGap;
    cursor_y += kModuleControlRowH + kModuleTopGap;
    constexpr float kLedLabelScale = 0.7f;
    constexpr int kLedLabelGap = 6;
    // prepare label bitmaps for logical rows (send/receive)
    const int full_cols = 2; // left/right columns
    int visible_total = ctx ? std::clamp(ctx->module_frame_pair_count, 1, kModuleExtraLedCount * full_cols) : kModuleExtraLedCount * full_cols;
    int pair_count = (kModuleExtraLedRows + full_cols - 1) / full_cols; // logical label rows (send/receive)
    std::vector<TextBitmap> label_bitmaps(static_cast<size_t>(pair_count));
    int label_w = 0;
    for (int i = 0; i < pair_count; ++i) {
        const char* lbl = (i == 0) ? "send" : "receive";
        label_bitmaps[static_cast<size_t>(i)] = render_text_to_rgba(lbl, kLedLabelScale, {190,190,205,255});
        label_w = std::max(label_w, label_bitmaps[static_cast<size_t>(i)].width);
    }
    const int label_pad = (label_w > 0) ? kLedLabelGap : 0;
    int led_row_h = kModuleLedRowH;
    // estimate control width same as main raster so hitboxes align with drawn controls
    int nbw = std::max(1, kModuleControlRowH - 2);
    int num_w = std::max(24, nbw * 2);
    int gap2 = 6;
    int ctrl_total_w = nbw + gap2 + num_w + gap2 + nbw;
    int remaining_w2 = std::max(1, m.w - kModuleTopPadding * 2 - ctrl_total_w);
    int col_total_w = std::max(1, remaining_w2 / full_cols);
    int led_gap = 4;
    int visible_on_left = std::clamp(visible_total, 0, kModuleExtraLedCount);
    int visible_on_right = std::clamp(visible_total - kModuleExtraLedCount, 0, kModuleExtraLedCount);
    int max_visible_side = std::max(visible_on_left, visible_on_right);
    int led_area_inner_w = std::max(1, col_total_w - label_w - label_pad - kModuleTopPadding);
    int led_size = (led_area_inner_w - led_gap * std::max(0, max_visible_side - 1)) / std::max(1, max_visible_side);
    if (led_size < 8) {
        led_gap = 2;
        led_size = (led_area_inner_w - led_gap * std::max(0, max_visible_side - 1)) / std::max(1, max_visible_side);
    }
    led_size = std::max(4, std::min(led_size, led_row_h - 4));
    int total_rows = pair_count * full_cols;
    for (int row = 0; row < total_rows; ++row) {
        int grid_row = row / full_cols;
        int grid_col = row % full_cols;
        int led_row_y = cursor_y + grid_row * (kModuleLedRowH + kModuleTopGap);
        // left-aligned groups for hitbox enumeration: match raster layout
        int left_base_x = kModuleTopPadding + ctrl_total_w + (label_w + label_pad);
        int left_led_total = led_size * visible_on_left + led_gap * std::max(0, visible_on_left - 1);
        int visible_here = (grid_col == 0) ? visible_on_left : visible_on_right;
        int led_start_x = (grid_col == 0) ? left_base_x : (left_base_x + left_led_total);
        int led_y = led_row_y + (led_row_h - led_size) / 2;
        for (int i = 0; i < visible_here; ++i) {
            int lx = led_start_x + i * (led_size + led_gap);
            fn(row, i, lx, led_y, lx + led_size, led_y + led_size);
        }
    }
}

static GP_TableHitBox make_module_frame_led_hitbox(int row, int idx, int x0, int y0, int x1, int y1) {
    GP_TableHitBox hb{};
    hb.x0 = x0;
    hb.y0 = y0;
    hb.x1 = x1;
    hb.y1 = y1;
    hb.cell_kind = GP_TABLE_CELL_LEDS;
    hb.part = GP_TABLE_HIT_LED;
    const int full_cols = 2;
    int grid_row = row / full_cols;
    int grid_col = row % full_cols;
    hb.row_idx = (grid_row == 0) ? kModuleFrameRowSend : kModuleFrameRowReceive;
    hb.col_idx = (grid_col == 0) ? kModuleColLeftLed : kModuleColRightLed;
    int base = kModuleFrameContactBase + row * kModuleExtraLedCount;
    hb.aux0 = base + idx;
    hb.aux1 = 0;
    return hb;
}

static bool find_module_frame_led_hit(const GP_CanvasContextImpl* ctx, int module_idx, const GP_CanvasModuleDesc& m, int lx, int ly, GP_TableHitBox* out_hit) {
    if (!out_hit) return false;
    bool hit = false;
    for_each_module_frame_led(ctx, module_idx, m, [&](int row, int idx, int x0, int y0, int x1, int y1) {
        if (hit) return;
        if (lx >= x0 && lx < x1 && ly >= y0 && ly < y1) {
            *out_hit = make_module_frame_led_hitbox(row, idx, x0, y0, x1, y1);
            hit = true;
        }
    });
    return hit;
}

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
    CANVAS_ACT_SAVE = 2004,
    CANVAS_ACT_CLEAR = 2005,
    CANVAS_ACT_TOOL_CANVAS_0 = 2010,
    CANVAS_ACT_TOOL_CANVAS_1 = 2011,
    CANVAS_ACT_TOOL_CANVAS_2 = 2012,
    CANVAS_ACT_TOOL_CANVAS_3 = 2013,
    CANVAS_ACT_TOOL_EDGE_0 = 2014,
    CANVAS_ACT_TOOL_EDGE_1 = 2015,
    CANVAS_ACT_TOOL_EDGE_2 = 2016,
    CANVAS_ACT_TOOL_EDGE_3 = 2017,
    CANVAS_ACT_TOOL_EDGE_4 = 2018, // Meta-Edge Lasso (toolbar edge-group button)
    CANVAS_ACT_EDGE_ORDER_DEC = 2019,
    CANVAS_ACT_EDGE_ORDER_INC = 2020,
    CANVAS_ACT_EDGE_ORDER_TOOL = 2021,
    CANVAS_ACT_TOOL_TABLE_0 = 2030,
    CANVAS_ACT_TOOL_TABLE_1 = 2031,
    CANVAS_ACT_TOOL_TABLE_2 = 2032,
    CANVAS_ACT_IO_COUNT_DEC = 2040,
    CANVAS_ACT_IO_COUNT_INC = 2041,
    CANVAS_ACT_IO_CONSUMER_ADD = 2042,
    CANVAS_ACT_IO_PRODUCER_ADD = 2043,
    CANVAS_ACT_TABLE_TOOL_NUM_DEC = 2044,
    CANVAS_ACT_TABLE_TOOL_NUM_INC = 2045,
    CANVAS_ACT_META_CHAN_DEC = 2500,
    CANVAS_ACT_META_CHAN_INC = 2501,
    CANVAS_ACT_MODULE_LED = 2050,
    CANVAS_ACT_FRAME_PAIRS_DEC = 2200,
    CANVAS_ACT_FRAME_PAIRS_INC = 2201,
    CANVAS_ACT_TOOL_KPN_0 = 2060,
    CANVAS_ACT_TOOL_KPN_1 = 2061,
    CANVAS_ACT_TOOL_KPN_2 = 2062,
    CANVAS_ACT_THREAD_TOGGLE = 2070,
    CANVAS_ACT_KPN_GLOBAL_TOGGLE = 2079,
    CANVAS_ACT_THREAD_SIM_TOGGLE = 2078,
    CANVAS_ACT_CANVAS_ROOT_SIM_TOGGLE = 2088,
    CANVAS_ACT_THREAD_DELAY_DEC = 2071,
    CANVAS_ACT_THREAD_DELAY_INC = 2072,
    CANVAS_ACT_MODULE_CLONE = 2073,
    CANVAS_ACT_MODULE_CLEAR = 2074,
    CANVAS_ACT_MODULE_DESTROY = 2075,
    CANVAS_ACT_MODULE_EXPORT = 2076,
    CANVAS_ACT_MODULE_COMMIT = 2077,
    CANVAS_ACT_TOOL_SUBGROUP_0 = 2080,
    CANVAS_ACT_TOOL_SUBGROUP_1 = 2081,
    CANVAS_ACT_TOOL_SUBGROUP_2 = 2082,
    CANVAS_ACT_TOOL_SUBGROUP_3 = 2083,
    CANVAS_ACT_TOOL_SUBGROUP_4 = 2084,
    CANVAS_ACT_TOOL_SUBGROUP_5 = 2085,
    CANVAS_ACT_TOOL_SUBGROUP_6 = 2086,
    CANVAS_ACT_TOOL_SUBGROUP_7 = 2087,
    CANVAS_ACT_SPAWN_ROOT = 2090, // explicit spawn-root toolbar button
    CANVAS_ACT_MENU_TOOL_ADD = 2101,
    CANVAS_ACT_MENU_TOOL_SUB = 2102,
    CANVAS_ACT_MENU_TOOL_MUL = 2103,
    CANVAS_ACT_MENU_TOOL_DIV = 2104,
    CANVAS_ACT_MENU_TOOL_MOD = 2105,
    CANVAS_ACT_MENU_TOOL_KEYBOARD = 2106,
    CANVAS_ACT_MENU_TOOL_MOUSE = 2107,
    CANVAS_ACT_MENU_TOOL_STACK = 2108,
    CANVAS_ACT_MENU_TOOL_CLONE = 2109,
    CANVAS_ACT_MENU_TOOL_RECT = 2110,
    CANVAS_ACT_MENU_TOOL_NUMBER = 2111,
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
        case ModuleToolKind::Clone: return LABEL_TOOL_CLONE;
        case ModuleToolKind::RectRgba: return LABEL_TOOL_RECT;
        case ModuleToolKind::TableNumber: return LABEL_TOOL_NUMBER;
        case ModuleToolKind::None:
        default:
            return "";
    }
}

static std::string shared_library_extension() {
#if defined(_WIN32)
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
}

static bool shared_library_matches_tool(const std::string& filename, const std::string& tool_id) {
    const std::string ext = shared_library_extension();
    if (filename == tool_id + ext) return true;
    if (filename == "lib" + tool_id + ext) return true;
    if (ext == ".so") {
        const std::string prefix = "lib" + tool_id + ".so";
        if (filename.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

static bool parse_tool_kind_from_id(const std::string& tool_id, ModuleToolKind& out_kind) {
    constexpr const char* kPrefix = "tool_";
    if (tool_id.rfind(kPrefix, 0) != 0) return false;
    std::string digits = tool_id.substr(std::strlen(kPrefix));
    if (digits.empty()) return false;
    int value = 0;
    for (char ch : digits) {
        if (ch < '0' || ch > '9') return false;
        value = value * 10 + (ch - '0');
    }
    const int max_kind = static_cast<int>(ModuleToolKind::Clone);
    if (value <= 0 || value > max_kind) return false;
    out_kind = static_cast<ModuleToolKind>(value);
    return true;
}

static std::vector<ModuleToolKind> discover_compiled_plugin_tools() {
    namespace fs = std::filesystem;
    std::vector<ModuleToolKind> out;
    fs::path root = gp_module_library_default_root();
    fs::path tool_dir = root / "source" / "tools";
    if (!fs::exists(tool_dir)) return out;

    std::vector<std::string> tool_ids;
    for (const auto& entry : fs::directory_iterator(tool_dir, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) continue;
        const fs::path& path = entry.path();
        if (path.extension() != ".cpp") continue;
        tool_ids.push_back(path.stem().string());
    }
    std::cerr << "DEBUG: discover_compiled_plugin_tools: found tool source ids: ";
    for (const auto &tid : tool_ids) std::cerr << tid << ",";
    std::cerr << "\n";
    if (tool_ids.empty()) return out;

    std::vector<std::string> shared_libs;
    if (fs::exists(root)) {
        for (const auto& entry : fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) continue;
            shared_libs.push_back(entry.path().filename().string());
        }
    }

    for (const auto& tool_id : tool_ids) {
        ModuleToolKind kind = ModuleToolKind::None;
        if (!parse_tool_kind_from_id(tool_id, kind)) continue;
        bool compiled = false;
        for (const auto& filename : shared_libs) {
            if (shared_library_matches_tool(filename, tool_id)) {
                compiled = true;
                break;
            }
        }
            if (compiled) std::cerr << "DEBUG: discover_compiled_plugin_tools: tool_id='" << tool_id << "' matched on-disk filename\n";
        if (compiled) out.push_back(kind);
    }

    std::sort(out.begin(), out.end(), [](ModuleToolKind a, ModuleToolKind b) {
        return static_cast<int>(a) < static_cast<int>(b);
    });
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
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
    { CANVAS_ACT_MENU_TOOL_CLONE, LABEL_TOOL_CLONE, ModuleToolKind::Clone },
    { CANVAS_ACT_MENU_TOOL_RECT, LABEL_TOOL_RECT, ModuleToolKind::RectRgba },
    { CANVAS_ACT_MENU_TOOL_NUMBER, LABEL_TOOL_NUMBER, ModuleToolKind::TableNumber },
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
    int table_section_y = 0;
    int table_control_y = 0;
    int table_control_h = 0;
};

struct PluginMenuLayout {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    int header_h = 0;
    int row_h = 0;
    int item_start_y = 0;
    int item_count = 0;
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
    h += padding + header_h + row_h;
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
    int table_start_y = layout.item_start_y + static_cast<int>(std::size(kToolMenuItems)) * row_h + padding;
    if (stack_count > 0) {
        table_start_y = layout.stack_start_y + stack_count * row_h + padding;
    }
    layout.table_section_y = table_start_y;
    layout.table_control_y = table_start_y + header_h;
    layout.table_control_h = row_h;
    return layout;
}

static PluginMenuLayout compute_plugin_menu_layout(const GP_CanvasContextImpl* ctx, int module_idx, int item_count) {
    PluginMenuLayout layout{};
    if (!ctx || module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return layout;
    const auto& m = ctx->modules[module_idx];
    ModuleLayout mod_layout = module_layout_for(ctx, module_idx, m);
    const int padding = 8;
    const int margin = 12;
    const int header_h = 18;
    const int row_h = 22;
    const int w = 200;
    const int rows = std::max(1, item_count);
    int h = padding * 2 + header_h + rows * row_h;
    int sx = m.x - ctx->offset_x;
    int sy = m.y - ctx->offset_y;
    int x = sx + m.w - w - margin;
    x = std::clamp(x, margin, std::max(margin, ctx->width - w - margin));
    int y = sy + mod_layout.table_y + margin;
    y = std::clamp(y, margin, std::max(margin, ctx->height - h - margin));
    layout.x = x;
    layout.y = y;
    layout.w = w;
    layout.h = h;
    layout.header_h = header_h;
    layout.row_h = row_h;
    layout.item_start_y = y + padding + header_h;
    layout.item_count = rows;
    return layout;
}

static void canvas_refresh_plugin_tools(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    // Discover compiled tool source stems and on-disk shared objects.
    namespace fs = std::filesystem;
    ctx->plugin_tool_kinds.clear();
    ctx->plugin_tool_ids.clear();
    ctx->plugin_tool_labels.clear();

    fs::path root = gp_module_library_default_root();
    fs::path tool_dir = root / "source" / "tools";
    if (!fs::exists(tool_dir)) return;

    std::vector<std::string> tool_ids;
    for (const auto& entry : fs::directory_iterator(tool_dir, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) continue;
        const fs::path& path = entry.path();
        if (path.extension() != ".cpp") continue;
        tool_ids.push_back(path.stem().string());
    }

    // gather on-disk file names under module library root so we can detect compiled artifacts
    std::vector<std::string> shared_libs;
    if (fs::exists(root)) {
        for (const auto& entry : fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) continue;
            shared_libs.push_back(entry.path().filename().string());
        }
    }

    for (const auto& tool_id : tool_ids) {
        ModuleToolKind kind = ModuleToolKind::None;
        if (parse_tool_kind_from_id(tool_id, kind)) {
            // tool id encodes a builtin kind like tool_1 etc.
            bool compiled = false;
            for (const auto& filename : shared_libs) {
                if (shared_library_matches_tool(filename, tool_id)) { compiled = true; break; }
            }
            if (compiled) {
                ctx->plugin_tool_kinds.push_back(kind);
                ctx->plugin_tool_ids.push_back(tool_id);
                ctx->plugin_tool_labels.push_back(tool_id);
            }
        } else {
            // Non-numeric generated tool names (e.g. tool_module_0_ver_...) -- still surface them
            bool compiled = false;
            for (const auto& filename : shared_libs) {
                if (shared_library_matches_tool(filename, tool_id)) { compiled = true; break; }
            }
            if (compiled) {
                ctx->plugin_tool_kinds.push_back(ModuleToolKind::None);
                ctx->plugin_tool_ids.push_back(tool_id);
                ctx->plugin_tool_labels.push_back(tool_id);
            }
        }
    }

    // deduplicate and keep order
    std::vector<std::string> ids_unique;
    std::vector<std::string> labels_unique;
    std::vector<ModuleToolKind> kinds_unique;
    for (size_t i = 0; i < ctx->plugin_tool_ids.size(); ++i) {
        if (std::find(ids_unique.begin(), ids_unique.end(), ctx->plugin_tool_ids[i]) == ids_unique.end()) {
            ids_unique.push_back(ctx->plugin_tool_ids[i]);
            labels_unique.push_back(ctx->plugin_tool_labels[i]);
            kinds_unique.push_back(ctx->plugin_tool_kinds[i]);
        }
    }
    ctx->plugin_tool_ids.swap(ids_unique);
    ctx->plugin_tool_labels.swap(labels_unique);
    ctx->plugin_tool_kinds.swap(kinds_unique);
}

struct ToolMenuCounterLayout {
    int bx_minus = 0;
    int bx_num = 0;
    int bx_plus = 0;
    int by = 0;
    int nbw = 0;
    int num_w = 0;
    int h = 0;
};

static ToolMenuCounterLayout compute_tool_menu_counter_layout(const ToolMenuLayout& layout) {
    ToolMenuCounterLayout out{};
    int control_x = layout.x + 8;
    int control_w = layout.w - 16;
    int control_h = std::max(18, layout.table_control_h - 2);
    int gap = 8;
    int nbw = control_h;
    int num_w = std::max(32, control_w - nbw * 2 - gap * 2);
    int total_w = nbw + gap + num_w + gap + nbw;
    int bx_minus = control_x + (control_w - total_w) / 2;
    out.bx_minus = bx_minus;
    out.bx_num = bx_minus + nbw + gap;
    out.bx_plus = out.bx_num + num_w + gap;
    out.by = layout.table_control_y + 1;
    out.nbw = nbw;
    out.num_w = num_w;
    out.h = control_h;
    return out;
}

static const GP_TableAction kCanvasRootActions[] = {
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SIM_SEGS_DEC, CANVAS_ACT_SIM_SEGS_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SIM_SEGS_INC, CANVAS_ACT_SIM_SEGS_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SIM_SLACK_DEC, CANVAS_ACT_SIM_SLACK_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SIM_SLACK_INC, CANVAS_ACT_SIM_SLACK_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SAVE, CANVAS_ACT_SAVE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_CLEAR, CANVAS_ACT_CLEAR },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_CANVAS_0, CANVAS_ACT_TOOL_CANVAS_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_CANVAS_1, CANVAS_ACT_TOOL_CANVAS_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_CANVAS_2, CANVAS_ACT_TOOL_CANVAS_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_CANVAS_3, CANVAS_ACT_TOOL_CANVAS_3 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_0, CANVAS_ACT_TOOL_EDGE_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_1, CANVAS_ACT_TOOL_EDGE_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_2, CANVAS_ACT_TOOL_EDGE_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_3, CANVAS_ACT_TOOL_EDGE_3 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_4, CANVAS_ACT_TOOL_EDGE_4 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EDGE_ORDER_DEC, CANVAS_ACT_EDGE_ORDER_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EDGE_ORDER_INC, CANVAS_ACT_EDGE_ORDER_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EDGE_ORDER_TOOL, CANVAS_ACT_EDGE_ORDER_TOOL },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_TABLE_0, CANVAS_ACT_TOOL_TABLE_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_TABLE_1, CANVAS_ACT_TOOL_TABLE_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_TABLE_2, CANVAS_ACT_TOOL_TABLE_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_FRAME_PAIRS_DEC, CANVAS_ACT_FRAME_PAIRS_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_FRAME_PAIRS_INC, CANVAS_ACT_FRAME_PAIRS_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_IO_COUNT_DEC, CANVAS_ACT_IO_COUNT_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_IO_COUNT_INC, CANVAS_ACT_IO_COUNT_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_IO_CONSUMER_ADD, CANVAS_ACT_IO_CONSUMER_ADD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_IO_PRODUCER_ADD, CANVAS_ACT_IO_PRODUCER_ADD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TABLE_TOOL_NUM_DEC, CANVAS_ACT_TABLE_TOOL_NUM_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TABLE_TOOL_NUM_INC, CANVAS_ACT_TABLE_TOOL_NUM_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_META_CHAN_DEC, CANVAS_ACT_META_CHAN_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_META_CHAN_INC, CANVAS_ACT_META_CHAN_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_KPN_0, CANVAS_ACT_TOOL_KPN_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_KPN_1, CANVAS_ACT_TOOL_KPN_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_KPN_2, CANVAS_ACT_TOOL_KPN_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_THREAD_TOGGLE, CANVAS_ACT_THREAD_TOGGLE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_THREAD_SIM_TOGGLE, CANVAS_ACT_THREAD_SIM_TOGGLE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_CANVAS_ROOT_SIM_TOGGLE, CANVAS_ACT_CANVAS_ROOT_SIM_TOGGLE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_KPN_GLOBAL_TOGGLE, CANVAS_ACT_KPN_GLOBAL_TOGGLE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_THREAD_DELAY_DEC, CANVAS_ACT_THREAD_DELAY_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_THREAD_DELAY_INC, CANVAS_ACT_THREAD_DELAY_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MODULE_CLONE, CANVAS_ACT_MODULE_CLONE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MODULE_CLEAR, CANVAS_ACT_MODULE_CLEAR },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MODULE_DESTROY, CANVAS_ACT_MODULE_DESTROY },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MODULE_COMMIT, CANVAS_ACT_MODULE_COMMIT },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MODULE_EXPORT, CANVAS_ACT_MODULE_EXPORT },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_0, CANVAS_ACT_TOOL_SUBGROUP_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_1, CANVAS_ACT_TOOL_SUBGROUP_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_2, CANVAS_ACT_TOOL_SUBGROUP_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_3, CANVAS_ACT_TOOL_SUBGROUP_3 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_4, CANVAS_ACT_TOOL_SUBGROUP_4 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_5, CANVAS_ACT_TOOL_SUBGROUP_5 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_6, CANVAS_ACT_TOOL_SUBGROUP_6 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_7, CANVAS_ACT_TOOL_SUBGROUP_7 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_ADD, CANVAS_ACT_MENU_TOOL_ADD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_SUB, CANVAS_ACT_MENU_TOOL_SUB },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_MUL, CANVAS_ACT_MENU_TOOL_MUL },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_DIV, CANVAS_ACT_MENU_TOOL_DIV },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_MOD, CANVAS_ACT_MENU_TOOL_MOD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_KEYBOARD, CANVAS_ACT_MENU_TOOL_KEYBOARD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_MOUSE, CANVAS_ACT_MENU_TOOL_MOUSE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_STACK, CANVAS_ACT_MENU_TOOL_STACK },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_CLONE, CANVAS_ACT_MENU_TOOL_CLONE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_RECT, CANVAS_ACT_MENU_TOOL_RECT },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_NUMBER, CANVAS_ACT_MENU_TOOL_NUMBER },
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
static void canvas_push_tool_to_focused(GP_CanvasContextImpl* ctx, ModuleToolKind tool, ModuleToolOrigin origin);

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
        // Click-listen mode: capture the action intent instead of dispatching.
        if (c->click_listen_mode) {
            if (c->pending_action) {
                // Already have a pending action; overwrite (caller intent keeps most recent)
                delete c->pending_action;
                c->pending_action = nullptr;
            }
            c->pending_action = new GP_CanvasContextImpl::PendingAction();
            c->pending_action->action_id = action_id;
            c->pending_action->hit = *hit;
            printf("gp_canvas_on_click: captured action_id=%d in click-listen mode\n", action_id);
            return;
        }
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
            case CANVAS_ACT_SAVE:
                if (!c->autosave_path.empty()) {
                    gp_canvas_save_to_file(reinterpret_cast<GP_CanvasContext*>(c), c->autosave_path.c_str());
                } else {
                    gp_canvas_save_to_file(reinterpret_cast<GP_CanvasContext*>(c), kCanvasDefaultWorkspacePath);
                }
                break;
            case CANVAS_ACT_CLEAR:
                // Clear overlays, meta-groups, and modules via unified API
                gp_canvas_clear_meta_and_overlays(reinterpret_cast<GP_CanvasContext*>(c));
                update_canvas_scroll_state(c, /*pull_from_container=*/false);
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
                // Selecting a regular edge tool clears lasso mode
                c->lasso_mode = false;
                int tool = static_cast<int>(action_id - CANVAS_ACT_TOOL_EDGE_0);
                if (c->selected_tool_edge == tool) c->selected_tool_edge = -1; else c->selected_tool_edge = tool;
                printf("gp_canvas_on_click: edge tool %d toggled -> selected_tool_edge=%d\n", tool, c->selected_tool_edge);
                break;
            }
            case CANVAS_ACT_TOOL_EDGE_4: {
                // Lasso button: toggle lasso_mode and ensure other edge tools are deselected
                c->selected_tool_edge = -1;
                c->lasso_mode = !c->lasso_mode;
                printf("gp_canvas_on_click: lasso_mode toggled -> %d\n", c->lasso_mode);
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
                c->plugin_menu_open = false;
                c->plugin_menu_module_idx = -1;
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
            case CANVAS_ACT_FRAME_PAIRS_DEC:
            case CANVAS_ACT_FRAME_PAIRS_INC: {
                int delta = (action_id == CANVAS_ACT_FRAME_PAIRS_INC) ? 1 : -1;
                const int max_pairs = kModuleExtraLedCount * 2; // allow across both left+right
                c->module_frame_pair_count = std::clamp(c->module_frame_pair_count + delta, 1, max_pairs);
                printf("gp_canvas_on_click: module_frame_pair_count -> %d\n", c->module_frame_pair_count);
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
            case CANVAS_ACT_TABLE_TOOL_NUM_DEC:
            case CANVAS_ACT_TABLE_TOOL_NUM_INC: {
                int delta = (action_id == CANVAS_ACT_TABLE_TOOL_NUM_INC) ? 1 : -1;
                c->table_tool_number = std::clamp(c->table_tool_number + delta, 0, 99);
                printf("gp_canvas_on_click: table_tool_number -> %d\n", c->table_tool_number);
                break;
            }
            case CANVAS_ACT_META_CHAN_DEC:
            case CANVAS_ACT_META_CHAN_INC: {
                int delta = (action_id == CANVAS_ACT_META_CHAN_INC) ? 1 : -1;
                // If aux0 encodes an overlay id (ov.id), use the overlay's
                // authoritative meta binding directly to mutate the meta-group.
                int overlay_id = hit->aux0;
                if (overlay_id > 0) {
                    auto oit = c->overlays.find(overlay_id);
                    if (oit != c->overlays.end()) {
                        auto &ov = oit->second;
                        if (ov.meta_table && ov.meta_mg) {
                            int cur = 0; gp_table_meta_get_channel_group(ov.meta_table, ov.meta_mg, &cur);
                            cur = std::clamp(cur + delta, -32768, 32767);
                            gp_table_meta_set_channel_group(ov.meta_table, ov.meta_mg, cur);
                            update_canvas_scroll_state(c, /*pull_from_container=*/false);
                            printf("gp_canvas_on_click: meta mg=%p ov=%d new_group=%d\n", (void*)ov.meta_mg, overlay_id, cur);
                            break;
                        }
                    }
                }
                // Fallback: hit->aux0 carried a rope_idx; search tables for matching dangling rope
                int rope_idx = hit->aux0;
                auto try_handle = [&](GP_TableContext* t)->bool{
                    if (!t) return false;
                    int mgcount = gp_table_get_meta_group_count(t);
                    for (int mgi = 0; mgi < mgcount; ++mgi) {
                        GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                        if (!mg) continue;
                        int _dr=-1,_dv=-1; gp_table_meta_get_dangling_rope_info(t, mg, &_dr, &_dv);
                        if (_dr != rope_idx) continue;
                        int cur = 0; gp_table_meta_get_channel_group(t, mg, &cur);
                        cur = std::clamp(cur + delta, -32768, 32767);
                        gp_table_meta_set_channel_group(t, mg, cur);
                        update_canvas_scroll_state(c, /*pull_from_container=*/false);
                        printf("gp_canvas_on_click: meta mg=%p rope=%d new_group=%d\n", (void*)mg, rope_idx, cur);
                        return true;
                    }
                    return false;
                };
                if (try_handle(c->container_table)) break;
                for (size_t mi = 0; mi < c->module_tables.size(); ++mi) if (try_handle(c->module_tables[mi])) break;
                break;
            }
            case CANVAS_ACT_TOOL_KPN_0:
            case CANVAS_ACT_TOOL_KPN_1:
            case CANVAS_ACT_TOOL_KPN_2: {
                int tool = static_cast<int>(action_id - CANVAS_ACT_TOOL_KPN_0);
                if (c->selected_tool_kpn == tool) c->selected_tool_kpn = -1; else c->selected_tool_kpn = tool;
                printf("gp_canvas_on_click: kpn tool %d toggled -> selected_tool_kpn=%d\n", tool, c->selected_tool_kpn);
                break;
            }
            case CANVAS_ACT_TOOL_SUBGROUP_0:
            case CANVAS_ACT_TOOL_SUBGROUP_1:
            case CANVAS_ACT_TOOL_SUBGROUP_2:
            case CANVAS_ACT_TOOL_SUBGROUP_3:
            case CANVAS_ACT_TOOL_SUBGROUP_4:
            case CANVAS_ACT_TOOL_SUBGROUP_5:
            case CANVAS_ACT_TOOL_SUBGROUP_6:
            case CANVAS_ACT_TOOL_SUBGROUP_7: {
                int tool = static_cast<int>(action_id - CANVAS_ACT_TOOL_SUBGROUP_0);
                if (tool >= kSubgroupBinCount) break;
                uint32_t bit = subgroup_mask_for_index(tool);
                if (c->selected_tool_subgroup_flags & bit) c->selected_tool_subgroup_flags &= ~bit;
                else c->selected_tool_subgroup_flags |= bit;
                printf("gp_canvas_on_click: subgroup tool %d toggled -> selected_tool_subgroup_flags=0x%08x\n", tool, c->selected_tool_subgroup_flags);
                break;
            }
            case CANVAS_ACT_THREAD_TOGGLE: {
                // If a module is focused, toggle that module's sim flag (per-module play/pause).
                // If a module-specific action is in-flight (dispatch_module_idx), prefer it.
                int target_module = -1;
                if (c->dispatch_module_idx >= 0) target_module = c->dispatch_module_idx;
                else if (c->focused_module >= 0) target_module = c->focused_module;
                if (target_module >= 0 && target_module < static_cast<int>(c->modules.size())) {
                    int mi = target_module;
                    if (mi >= static_cast<int>(c->module_skip.size())) c->module_skip.resize(mi + 1, 0);
                    c->module_skip[mi] = c->module_skip[mi] ? 0 : 1;
                    printf("gp_canvas_on_click: module %d module_skip -> %d\n", mi, c->module_skip[mi]);
                } else {
                    // No module target: fall back to toggling global thread manager pause
                    // keep CANVAS_ACT_THREAD_TOGGLE module-scoped; use GLOBAL_TOGGLE for global toolbar
                    printf("gp_canvas_on_click: THREAD_TOGGLE invoked with no module target - ignoring global toggle\n");
                }
                break;
            }
            case CANVAS_ACT_KPN_GLOBAL_TOGGLE: {
                c->thread_mgr_paused = !c->thread_mgr_paused;
                if (c->thread_mgr_paused) {
                    c->thread_mgr_delay_accum_s = 0.0;
                } else {
                    c->thread_mgr_delay_accum_s = static_cast<double>(std::max(0, c->thread_mgr_delay_ms)) / 1000.0;
                }
                printf("gp_canvas_on_click: GLOBAL kpn_paused -> %d\n", c->thread_mgr_paused ? 1 : 0);
                break;
            }
            case CANVAS_ACT_THREAD_SIM_TOGGLE: {
                int target_module = -1;
                if (c->dispatch_module_idx >= 0) target_module = c->dispatch_module_idx;
                else if (c->focused_module >= 0) target_module = c->focused_module;
                if (target_module >= 0 && target_module < static_cast<int>(c->modules.size())) {
                    int mi = target_module;
                    if (mi >= static_cast<int>(c->module_sim_enabled.size())) c->module_sim_enabled.resize(mi + 1, 1);
                    c->module_sim_enabled[mi] = c->module_sim_enabled[mi] ? 0 : 1;
                    // apply to attached table if present
                    if (mi >= 0 && mi < static_cast<int>(c->module_tables.size()) && c->module_tables[mi]) {
                        gp_table_set_sim_enabled(c->module_tables[mi], c->module_sim_enabled[mi]);
                    }
                    printf("gp_canvas_on_click: module %d sim_enabled -> %d\n", mi, c->module_sim_enabled[mi]);
                } else {
                    // No module target — treat as canvas-root simulator toggle (toolbar SIM)
                    // This handler path is only reached for module-targeted SIM toggles.
                    // Toolbar-level SIM now uses CANVAS_ACT_CANVAS_ROOT_SIM_TOGGLE and
                    // is handled separately.
                }
                break;
            }
            case CANVAS_ACT_CANVAS_ROOT_SIM_TOGGLE: {
                GP_TableContext* root = canvas_ensure_root_table(c);
                if (root) {
                    int cur = 0;
                    gp_table_get_sim_enabled(root, &cur);
                    int next = cur ? 0 : 1;
                    // Toggle per-table sim enable as before
                    gp_table_set_sim_enabled(root, next);
                    // Also toggle a toolbar-level global pause so the canvas root sim
                    // (and all attached table sims) defer to the global skip decision.
                    c->sim_root_paused = next ? 0 : 1;
                    if (c->sim_root_paused) {
                        // set an effectively infinite skip so gp_table_should_step_sim
                        // will return false for all tables until unpaused
                        gp_table_set_global_sim_frame_skip_count(INT32_MAX / 4);
                    } else {
                        // restore canvas-requested skip count
                        gp_table_set_global_sim_frame_skip_count(c->sim_frame_skip_count);
                    }
                    printf("gp_canvas_on_click: canvas-root sim_enabled -> %d sim_root_paused=%d\n", next, c->sim_root_paused);
                } else {
                    printf("gp_canvas_on_click: canvas-root table unavailable for SIM toggle\n");
                }
                break;
            }
            
            case CANVAS_ACT_THREAD_DELAY_DEC:
            case CANVAS_ACT_THREAD_DELAY_INC: {
                int delta = (action_id == CANVAS_ACT_THREAD_DELAY_INC) ? 10 : -10;
                c->thread_mgr_delay_ms = std::clamp(c->thread_mgr_delay_ms + delta, 0, 2000);
                printf("gp_canvas_on_click: thread_mgr_delay_ms -> %d\n", c->thread_mgr_delay_ms);
                break;
            }
            case CANVAS_ACT_MODULE_CLONE: {
                int focused = c->focused_module;
                if (focused >= 0 && focused < static_cast<int>(c->modules.size())) {
                    int new_idx = canvas_clone_module(c, focused);
                    if (new_idx >= 0) c->focused_module = new_idx;
                }
                break;
            }
            case CANVAS_ACT_MODULE_CLEAR: {
                int focused = c->focused_module;
                if (focused >= 0 && focused < static_cast<int>(c->modules.size())) {
                    canvas_clear_module_table(c, focused);
                }
                break;
            }
            case CANVAS_ACT_MODULE_DESTROY: {
                int focused = c->focused_module;
                if (focused >= 0 && focused < static_cast<int>(c->modules.size())) {
                    canvas_destroy_module(c, focused);
                }
                break;
            }
            case CANVAS_ACT_MODULE_COMMIT: {
                int focused = c->focused_module;
                if (focused >= 0 && focused < static_cast<int>(c->modules.size())) {
                    // Export module sources to the module library root, actualize it as a tool, then perform a per-module scratch build
                    gp_canvas_export_module_to_root(reinterpret_cast<GP_CanvasContext*>(c), focused, nullptr);
                    // Ensure the exported module is actualized as a tool (generate full tool source including create_tool)
                    GP_ModuleLibrary lib{};
                    lib.root_dir = gp_module_library_default_root();
                    GP_ModuleLibraryModule mod{};
                    mod.module_idx = focused;
                    mod.id = gp_module_library_module_id(focused);
                    mod.source_path = gp_module_library_module_source_path(std::string(), mod.id);
                    mod.convert_to_tool = true;
                    lib.modules.push_back(std::move(mod));
                    gp_module_library_actualize_sources(lib, lib.root_dir.c_str());

                    std::string module_id = gp_module_library_module_id(focused);
                    std::string dest = gp_module_library_default_root();
                    // find the newest generated versioned tool source for this module in the tools dir
                    std::filesystem::path tools_dir = std::filesystem::path(dest) / "source" / "tools";
                    std::string prefix = ("tool_" + module_id) + std::string("_ver_");
                    std::filesystem::path chosen;
                    std::filesystem::file_time_type latest;
                    if (std::filesystem::exists(tools_dir)) {
                        for (auto &ent : std::filesystem::directory_iterator(tools_dir)) {
                            if (!ent.is_regular_file()) continue;
                            std::string name = ent.path().filename().string();
                            if (name.rfind(prefix, 0) != 0) continue;
                            auto ftime = std::filesystem::last_write_time(ent.path());
                            if (chosen.empty() || ftime > latest) {
                                latest = ftime;
                                chosen = ent.path();
                            }
                        }
                    }
                    std::string module_src;
                    if (!chosen.empty()) module_src = chosen.generic_string();
                    else module_src = std::filesystem::path(gp_module_library_module_source_path(std::string(), module_id)).generic_string();
                    char out_id[256];
                    int ok = gp_plugin_build_module_and_load(module_src.c_str(), ".", dest.c_str(), nullptr, out_id, static_cast<int>(sizeof(out_id)));
                    if (ok) {
                        printf("module commit: scratch-built+loaded id=%s (module=%s src=%s)\n", out_id, module_id.c_str(), module_src.c_str());
                        // Refresh the in-memory plugin tool list so the UI reflects the newly loaded plugin
                        canvas_refresh_plugin_tools(c);
                    } else {
                        printf("module commit: build+load failed for module=%s\n", module_id.c_str());
                    }
                }
                break;
            }

            case CANVAS_ACT_MODULE_EXPORT: {
                int focused = c->focused_module;
                if (focused >= 0 && focused < static_cast<int>(c->modules.size())) {
                    gp_canvas_export_module_to_root(reinterpret_cast<GP_CanvasContext*>(c), focused, nullptr);
                }
                break;
            }
            case CANVAS_ACT_MODULE_LED:
                if (c->dispatch_module_idx >= 0) {
                    canvas_handle_module_led_hit(c, c->dispatch_module_idx, *hit);
                }
                break;
            case CANVAS_ACT_MENU_TOOL_ADD:
                canvas_push_tool_to_focused(c, ModuleToolKind::Add, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_SUB:
                canvas_push_tool_to_focused(c, ModuleToolKind::Subtract, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_MUL:
                canvas_push_tool_to_focused(c, ModuleToolKind::Multiply, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_DIV:
                canvas_push_tool_to_focused(c, ModuleToolKind::Divide, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_MOD:
                canvas_push_tool_to_focused(c, ModuleToolKind::Modulo, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_KEYBOARD:
                canvas_push_tool_to_focused(c, ModuleToolKind::KeyboardListener, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_MOUSE:
                canvas_push_tool_to_focused(c, ModuleToolKind::MouseListener, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_STACK:
                canvas_push_tool_to_focused(c, ModuleToolKind::StackDisplay, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_CLONE:
                canvas_push_tool_to_focused(c, ModuleToolKind::Clone, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_RECT:
                canvas_push_tool_to_focused(c, ModuleToolKind::RectRgba, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_NUMBER:
                canvas_push_tool_to_focused(c, ModuleToolKind::TableNumber, ModuleToolOrigin::Builtin);
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
    // Ensure a root RopeSim exists immediately so edges created later
    // on the root/container table get real rope indices and persistent uids.
    if (ctx->container_table && !gp_table_get_rope_sim(ctx->container_table)) {
        RopeSim* sim = rope_sim_create(1024, 64);
        gp_table_attach_rope_sim(ctx->container_table, sim, 1);
        // Attach any existing module tables to the new root sim so their
        // edges also get real ropes/uids rather than deferring until later.
        for (size_t i = 0; i < ctx->module_tables.size(); ++i) {
            if (ctx->module_tables[i]) gp_table_attach_rope_sim(ctx->module_tables[i], sim, 0);
        }
        printf("canvas_ensure_root_table: created and attached root RopeSim %p\n", (void*)sim);
    }
    return ctx->container_table;
}

static RopeSim* canvas_root_sim(GP_CanvasContextImpl* ctx) {
    if (!ctx || !ctx->container_table) return nullptr;
    return gp_table_get_rope_sim(ctx->container_table);
}

// Ensure a synthetic canvas module that mirrors the root/container table exists.
static int canvas_ensure_root_module(GP_CanvasContextImpl* ctx) {
    if (!ctx) return -1;
    if (ctx->root_module_idx >= 0 && ctx->root_module_idx < static_cast<int>(ctx->modules.size())) return ctx->root_module_idx;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return -1;
    GP_CanvasModuleDesc d{};
    d.x = 16; d.y = 64; d.w = std::max(160, ctx->width - 32); d.h = std::max(120, ctx->height / 3);
    std::strncpy(d.label, "ROOT", sizeof(d.label) - 1);
    d.label[sizeof(d.label) - 1] = '\0';
    int new_idx = gp_canvas_add_module(reinterpret_cast<GP_CanvasContext*>(ctx), &d);
    if (new_idx < 0) return -1;
    // gp_canvas_add_module created a canvas-owned table for this module; replace it with the real root table
    if (new_idx < static_cast<int>(ctx->module_tables.size())) {
        if (ctx->module_tables[new_idx] && ctx->module_table_owned[new_idx]) {
            gp_table_destroy(ctx->module_tables[new_idx]);
        }
        ctx->module_tables[new_idx] = root;
        ctx->module_table_owned[new_idx] = 0;
    }
    // sync layout to reflect root table IO
    // Ensure the sync runs even if the canvas isn't focused on this module by
    // temporarily setting focus so table-driven IO rows are created.
    int prev_focused = ctx->focused_module;
    ctx->focused_module = new_idx;
    sync_module_table_io_layout(ctx, new_idx);
    ctx->focused_module = prev_focused;
    ctx->root_module_idx = new_idx;
    return new_idx;
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
    // fast-path: explicit spawn-root action should create the synthetic
    // root module immediately so it's visible and clickable.
    if (action_id == CANVAS_ACT_SPAWN_ROOT) {
        canvas_ensure_root_module(ctx);
        return 1;
    }
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

static void canvas_set_subgroup_flags_for_edge(GP_CanvasContextImpl* ctx, const GP_CanvasEdgeDesc &desc, uint32_t flags) {
    if (!ctx) return;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return;
    uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
    uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
    int edge_idx = -1;
    if (gp_table_edge_index_for_pair(root, ka, kb, &edge_idx)) {
        gp_table_edge_set_subgroup_flags(root, edge_idx, flags);
    } else if (gp_table_edge_index_for_pair(root, kb, ka, &edge_idx)) {
        gp_table_edge_set_subgroup_flags(root, edge_idx, flags);
    }
}

static void canvas_apply_subgroup_to_edge_idx(GP_CanvasContextImpl* ctx, int edge_idx, uint32_t flags) {
    if (!ctx || edge_idx < 0 || edge_idx >= static_cast<int>(ctx->edges.size())) return;
    ctx->edges[static_cast<size_t>(edge_idx)].subgroup_flags = flags;
    canvas_set_subgroup_flags_for_edge(ctx, ctx->edges[static_cast<size_t>(edge_idx)].desc, flags);
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
    // keep module_plugin_instances aligned with module_io_rows for this module
    if (module_idx >= static_cast<int>(ctx->module_plugin_instances.size())) ctx->module_plugin_instances.resize(module_idx + 1);
    if (module_idx < static_cast<int>(ctx->module_io_rows.size())) {
        const size_t want = ctx->module_io_rows[module_idx].size();
        if (ctx->module_plugin_instances[module_idx].size() < want) ctx->module_plugin_instances[module_idx].resize(want);
    }
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
        if (module_idx >= static_cast<int>(ctx->module_plugin_instances.size())) ctx->module_plugin_instances.resize(module_idx + 1);
        ctx->module_plugin_instances[module_idx].clear();
        ctx->module_plugin_instances[module_idx].resize(ctx->module_io_rows[module_idx].size());
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

    GP_TableStyle st{};
    gp_table_get_style(t, &st);
    const auto &mod = ctx->modules[module_idx];
    st.width_px = std::max(1, mod.w);
    st.name_w_px = 0;
    if (st.row_h_px <= 0) st.row_h_px = 22;
    gp_table_set_style(t, &st);

    int content_w = std::max(1, st.width_px);
    int led_w = std::max(30, std::min(60, content_w / 3));
    int text_w = std::max(1, content_w - 2 * led_w);
    GP_TableColumn cols[kModuleColCount];
    cols[kModuleColLeftLed] = { GP_TABLE_CELL_LEDS_ARG, led_w, 0 };
    cols[kModuleColText] = { GP_TABLE_CELL_TEXT, text_w, 0 };
    cols[kModuleColRightLed] = { GP_TABLE_CELL_LEDS_ARG, led_w, 0 };
    gp_table_set_columns(t, cols, kModuleColCount);

    int base_row_h = st.row_h_px > 0 ? st.row_h_px : 22;
    int tool_row_h = std::max(14, base_row_h - 6);
    int stack_value_row_h = std::max(12, base_row_h - 8);

    int total_rows = static_cast<int>(ordered_rows.size());
    if (total_rows <= 0) {
        GP_TableRow prow{}; memset(&prow, 0, sizeof(prow));
        prow.kind = GP_TABLE_ROW_HEADER; prow.depth = 0; prow.expanded = 1; prow.selected = 0;
        prow.cell_count = kModuleColCount;
        prow.reserved0 = tool_row_h;
        prow.cells[kModuleColLeftLed].kind = GP_TABLE_CELL_TEXT;
        prow.cells[kModuleColText].kind = GP_TABLE_CELL_TEXT;
        prow.cells[kModuleColRightLed].kind = GP_TABLE_CELL_TEXT;
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

    auto fill_counter_cell = [](GP_TableCell &cell, int value, const char* label) {
        cell.kind = GP_TABLE_CELL_COUNTER;
        cell.value = static_cast<float>(std::max(0, value));
        cell.flags = 0;
        if (label && label[0] != '\0') {
            size_t len = std::min<std::size_t>(std::strlen(label), sizeof(cell.text) - 1);
            std::memcpy(cell.text, label, len);
            cell.text[len] = '\0';
        } else {
            cell.text[0] = '\0';
        }
    };

    auto fill_led_cell = [](GP_TableCell &cell, int count, uint32_t on_mask, uint32_t active_mask) {
        cell.kind = GP_TABLE_CELL_LEDS_ARG;
        int clamped = std::clamp(count, 1, 32);
        cell.value = static_cast<float>(clamped);
        cell.flags = on_mask;
        cell.reserved0 = static_cast<int32_t>(active_mask);
    };

    std::vector<GP_TableRow> rows;
    rows.reserve(static_cast<size_t>(total_rows));
    std::vector<ModuleIORow> row_meta;
    row_meta.reserve(static_cast<size_t>(total_rows));

    auto append_row = [&](const char* label, ModuleRowKind kind, int contact_idx, int attachment_count, int tool_value, ModuleToolKind tool_kind, ModuleToolOrigin tool_origin, int row_h, bool left_active, bool right_active) {
        GP_TableRow r{};
        memset(&r, 0, sizeof(r));
        if (kind == ModuleRowKind::Tool) {
            r.kind = (tool_origin == ModuleToolOrigin::Plugin) ? GP_TABLE_ROW_DEVICE : GP_TABLE_ROW_HEADER;
        } else {
            r.kind = GP_TABLE_ROW_DEVICE;
        }
        r.depth = 0;
        r.expanded = 1;
        r.selected = 0;
        r.cell_count = kModuleColCount;
        r.reserved0 = row_h;
        uint32_t mask = (attachment_count >= 32) ? 0xFFFFFFFFu : ((1u << std::max(1, attachment_count)) - 1u);
        uint32_t left_mask = left_active ? mask : 0u;
        uint32_t right_mask = right_active ? mask : 0u;
        if (kind == ModuleRowKind::Tool) {
            fill_text_cell(r.cells[kModuleColLeftLed], "");
            fill_text_cell(r.cells[kModuleColRightLed], "");
        } else {
            if (left_active) {
                fill_led_cell(r.cells[kModuleColLeftLed], std::max(1, attachment_count), left_mask, left_mask);
            } else {
                fill_text_cell(r.cells[kModuleColLeftLed], "");
            }
            if (right_active) {
                fill_led_cell(r.cells[kModuleColRightLed], std::max(1, attachment_count), right_mask, right_mask);
            } else {
                fill_text_cell(r.cells[kModuleColRightLed], "");
            }
        }
        if (kind == ModuleRowKind::Tool && tool_kind == ModuleToolKind::TableNumber) {
            fill_counter_cell(r.cells[kModuleColText], tool_value, label);
        } else {
            fill_text_cell(r.cells[kModuleColText], label);
        }
        rows.push_back(r);
        row_meta.push_back({kind, contact_idx, tool_kind, attachment_count, tool_origin});
    };

    auto append_stack_display_rows = [&](int logical_row_idx, ModuleToolOrigin tool_origin) {
        GP_TableRow header{};
        memset(&header, 0, sizeof(header));
        header.kind = (tool_origin == ModuleToolOrigin::Plugin) ? GP_TABLE_ROW_DEVICE : GP_TABLE_ROW_HEADER;
        header.depth = 0;
        header.expanded = 1;
        header.selected = 0;
        header.cell_count = kModuleColCount;
        header.reserved0 = tool_row_h;
        fill_text_cell(header.cells[kModuleColLeftLed], "");
        fill_text_cell(header.cells[kModuleColText], LABEL_TOOL_STACK);
        fill_text_cell(header.cells[kModuleColRightLed], "");
        rows.push_back(header);
        row_meta.push_back({ModuleRowKind::Tool, logical_row_idx, ModuleToolKind::StackDisplay, 0, tool_origin, std::string()});

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
            vr.cell_count = kModuleColCount;
            vr.reserved0 = stack_value_row_h;
            char buf[96];
            std::snprintf(buf, sizeof(buf), "%d: %.6g", display_idx, *it);
            fill_text_cell(vr.cells[kModuleColLeftLed], "");
            fill_text_cell(vr.cells[kModuleColText], buf);
            fill_text_cell(vr.cells[kModuleColRightLed], "");
            rows.push_back(vr);
            row_meta.push_back({ModuleRowKind::Tool, -1, ModuleToolKind::StackDisplay, 0, tool_origin, std::string()});
        }
    };

    int input_contact = 0;
    int output_contact = 0;
    for (size_t row_idx = 0; row_idx < ordered_rows.size(); ++row_idx) {
        const auto &row = ordered_rows[row_idx];
        if (row.kind == ModuleRowKind::Tool) {
            if (row.tool == ModuleToolKind::StackDisplay) {
                append_stack_display_rows(static_cast<int>(row_idx), row.tool_origin);
            } else {
                int tool_value = (row.tool == ModuleToolKind::TableNumber) ? std::max(0, row.attachment_count) : kModuleLedPerSide;
                std::string label_str;
                if (row.tool_origin == ModuleToolOrigin::Plugin) {
                    // prefer live instance name if present
                    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_plugin_instances.size())) {
                        const auto &insts = ctx->module_plugin_instances[module_idx];
                        if (row_idx < insts.size() && insts[row_idx]) {
                            try { label_str = insts[row_idx]->name(); } catch(...) { label_str = row.plugin_id; }
                        }
                    }
                    if (label_str.empty()) {
                        // fallback: use registry entry name if available
                        auto *entry = tool_registry_global().find(row.plugin_id);
                        if (entry) label_str = entry->name;
                        else label_str = row.plugin_id.empty() ? tool_label(row.tool) : row.plugin_id;
                    }
                } else {
                    label_str = tool_label(row.tool);
                }
                append_row(label_str.c_str(), ModuleRowKind::Tool, static_cast<int>(row_idx), kModuleLedPerSide, tool_value, row.tool, row.tool_origin, tool_row_h, false, false);
            }
            continue;
        }
        if (row.kind == ModuleRowKind::Input) {
            int remaining = std::clamp(row.attachment_count, 1, 32);
            while (remaining > 0) {
                int attachments = std::min(kModuleLedPerSide, remaining);
                append_row(LABEL_IO_INPUT, ModuleRowKind::Input, input_contact, attachments, attachments, ModuleToolKind::None, ModuleToolOrigin::Builtin, base_row_h, true, false);
                input_contact += attachments;
                remaining -= attachments;
            }
        } else if (row.kind == ModuleRowKind::Output) {
            int remaining = std::clamp(row.attachment_count, 1, 32);
            while (remaining > 0) {
                int attachments = std::min(kModuleLedPerSide, remaining);
                append_row(LABEL_IO_OUTPUT, ModuleRowKind::Output, in_count + output_contact, attachments, attachments, ModuleToolKind::None, ModuleToolOrigin::Builtin, base_row_h, false, true);
                output_contact += attachments;
                remaining -= attachments;
            }
        }
    }

    auto row_height_for = [&](const GP_TableRow& row) {
        int h = row.reserved0 > 0 ? row.reserved0 : base_row_h;
        return std::max(1, h);
    };
    int table_content_h = 0;
    for (const auto &r : rows) table_content_h += row_height_for(r);
    bool is_stage = (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[module_idx]);
    if (!is_stage) {
        int preview_target = std::max(1, mod.w);
        int target_h = kModuleTopUiHeight + table_content_h + preview_target;
        if (mod.h < target_h && module_idx >= 0 && module_idx < static_cast<int>(ctx->modules.size())) {
            ctx->modules[module_idx].h = target_h;
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
        int col_idx = is_input ? kModuleColLeftLed : kModuleColRightLed;
        if (col_idx >= rows[ri].cell_count) continue;
        const GP_TableCell &cell = rows[ri].cells[col_idx];
        int led_count = std::max(1, row_meta[ri].attachment_count);
        if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
            int c = std::max(0, std::min(32, static_cast<int>(cell.value)));
            if (c == 0) c = static_cast<int>(cell.flags & 0xFFu);
            if (c == 0) c = 12;
            led_count = std::min(led_count, c);
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

static void canvas_push_tool_to_focused(GP_CanvasContextImpl* ctx, ModuleToolKind tool, ModuleToolOrigin origin) {
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
        row.attachment_count = (tool == ModuleToolKind::TableNumber) ? 1 : 0;
        row.tool_origin = origin;
        rows.insert(rows.begin() + static_cast<ptrdiff_t>(idx), row);
    };
    if (tool == ModuleToolKind::KeyboardListener) {
        auto it = std::find_if(rows.begin(), rows.end(), [](const ModuleIORow &r) {
            return r.kind == ModuleRowKind::Tool && r.tool == ModuleToolKind::MouseListener;
        });
        if (it != rows.end()) {
            insert_tool_row(static_cast<size_t>(std::distance(rows.begin(), it) + 1));
        } else {
            rows.push_back({ModuleRowKind::Tool, -1, tool, (tool == ModuleToolKind::TableNumber) ? 1 : 0, origin, std::string()});
        }
    } else if (tool == ModuleToolKind::MouseListener) {
        auto it = std::find_if(rows.begin(), rows.end(), [](const ModuleIORow &r) {
            return r.kind == ModuleRowKind::Tool && r.tool == ModuleToolKind::KeyboardListener;
        });
        if (it != rows.end()) {
            insert_tool_row(static_cast<size_t>(std::distance(rows.begin(), it)));
        } else {
            rows.push_back({ModuleRowKind::Tool, -1, tool, (tool == ModuleToolKind::TableNumber) ? 1 : 0, origin, std::string()});
        }
    } else {
        rows.push_back({ModuleRowKind::Tool, -1, tool, (tool == ModuleToolKind::TableNumber) ? 1 : 0, origin, std::string()});
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
    // If the mouse tool was enabled for this module, programmatically bind
    // a small set of receive ports so the module's frame LEDs reflect that
    // the tool intends to receive mouse events. We bind four slots (up,
    // down, y, x) into the left receive column by default.
    if (tool == ModuleToolKind::MouseListener) {
        // Bind led indices 0..3 in the receive (is_send=0), left column (col=0)
        for (int li = 0; li < 4; ++li) {
            (void)gp_canvas_bind_action_enum_to_module_port(reinterpret_cast<GP_CanvasContext*>(ctx), focused, 0 /*receive*/, 0 /*left*/, li, CANVAS_ACT_MENU_TOOL_MOUSE);
        }
    }
}

static void canvas_push_plugin_tool_to_focused(GP_CanvasContextImpl* ctx, const std::string &plugin_id) {
    if (!ctx) return;
    int focused = ctx->focused_module;
    if (focused < 0 || focused >= static_cast<int>(ctx->modules.size())) return;
    if (focused >= static_cast<int>(ctx->module_tool_stack.size())) ctx->module_tool_stack.resize(focused + 1);
    ensure_module_row_order(ctx, focused);
    if (focused >= static_cast<int>(ctx->module_io_rows.size())) return;
    auto &rows = ctx->module_io_rows[focused];
    ModuleIORow row;
    row.kind = ModuleRowKind::Tool;
    row.contact_idx = -1;
    row.tool = ModuleToolKind::None;
    row.attachment_count = 0;
    row.tool_origin = ModuleToolOrigin::Plugin;
    row.plugin_id = plugin_id;
    rows.push_back(row);
    // ensure plugin instances vector aligns with rows
    if (focused >= static_cast<int>(ctx->module_plugin_instances.size())) ctx->module_plugin_instances.resize(focused + 1);
    if (ctx->module_plugin_instances[focused].size() < rows.size()) ctx->module_plugin_instances[focused].resize(rows.size());

    // try to create an instance for the plugin id via the ToolRegistry
    try {
        auto inst = tool_registry_global().create(plugin_id);
        if (inst) {
            // store instance in parallel vector at same index as the pushed row
            ctx->module_plugin_instances[focused][rows.size() - 1] = std::move(inst);
        }
    } catch (...) {
        // creation failed; leave null instance
    }

    auto &tools = ctx->module_tool_stack[focused];
    tools.clear();
}

static int canvas_handle_module_counter_hit(GP_CanvasContextImpl* ctx, int module_idx, const GP_TableHitBox& found) {
    if (!ctx) return 0;
    if (found.part != GP_TABLE_HIT_COUNTER_DEC && found.part != GP_TABLE_HIT_COUNTER_INC) return 0;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_table_rows.size())) return 0;
    const auto &rows = ctx->module_table_rows[module_idx];
    if (found.row_idx < 0 || found.row_idx >= static_cast<int>(rows.size())) return 0;
    const auto &meta = rows[found.row_idx];
    if (meta.kind != ModuleRowKind::Tool || meta.tool != ModuleToolKind::TableNumber) return 0;
    if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) return 0;
    int tool_row_idx = meta.contact_idx;
    if (tool_row_idx < 0 || tool_row_idx >= static_cast<int>(ctx->module_io_rows[module_idx].size())) return 0;
    auto &row = ctx->module_io_rows[module_idx][tool_row_idx];
    if (row.kind != ModuleRowKind::Tool || row.tool != ModuleToolKind::TableNumber) return 0;
    int delta = (found.part == GP_TABLE_HIT_COUNTER_INC) ? 1 : -1;
    row.attachment_count = std::clamp(row.attachment_count + delta, 0, 99);
    sync_module_table_io_layout(ctx, module_idx);
    return 1;
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
    if (ctx->selected_tool_subgroup_flags != 0u) {
        std::vector<int> edge_indices;
        canvas_collect_edges_for_contact(ctx, module_idx, contact_idx, edge_indices);
        if (!edge_indices.empty()) {
            uint32_t flags = ctx->selected_tool_subgroup_flags;
            for (int idx : edge_indices) {
                canvas_apply_subgroup_to_edge_idx(ctx, idx, flags);
            }
            ctx->selected.module = -1; ctx->selected.contact_idx = -1; ctx->selected.left = -1; ctx->selected.anchor_x = -1; ctx->selected.anchor_y = -1;
            if (ctx->prospective_rope_idx >= 0) ctx->prospective_rope_idx = -1;
            return 1;
        }
    }
    // Determine producer/consumer using row metadata when available.
    bool is_producer = false;
    bool resolved_role = false;
    if (found.row_idx == kModuleFrameRowSend || found.row_idx == kModuleFrameRowReceive) {
        is_producer = (found.row_idx == kModuleFrameRowSend);
        resolved_role = true;
    }
    const std::vector<ModuleIORow>* rows_ptr = nullptr;
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_table_rows.size()) &&
        !ctx->module_table_rows[module_idx].empty()) {
        rows_ptr = &ctx->module_table_rows[module_idx];
    } else if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_io_rows.size())) {
        rows_ptr = &ctx->module_io_rows[module_idx];
    }
    if (!resolved_role && rows_ptr) {
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
        printf("gp_canvas_on_click: adding edge sel %d.%d->%d.%d prospective_rope=%d\n",
            ctx->selected.module, ctx->selected.contact_idx, module_idx, contact_idx, ctx->prospective_rope_idx);
        int ei = gp_canvas_add_edge_with_type(reinterpret_cast<GP_CanvasContext*>(ctx), &e, /*type_id=*/0);
        printf("gp_canvas_on_click: gp_canvas_add_edge returned %d\n", ei);
        if (ei >= 0 && ctx->prospective_rope_idx >= 0) {
            ctx->edges[ei].rope_idx = ctx->prospective_rope_idx;
            printf("gp_canvas_on_click: attached rope %d to edge %d\n", ctx->prospective_rope_idx, ei);
            ctx->prospective_rope_idx = -1;
        } else {
            printf("gp_canvas_on_click: no rope attached (prospective=%d)\n", ctx->prospective_rope_idx);
        }
        ctx->selected.module = -1; ctx->selected.contact_idx = -1; ctx->selected.left = -1; ctx->selected.anchor_x = -1; ctx->selected.anchor_y = -1;
        return 1;
    } else if (!sel_producer && is_producer) {
        GP_CanvasEdgeDesc e; e.a_module = module_idx; e.a_contact_idx = contact_idx; e.b_module = ctx->selected.module; e.b_contact_idx = ctx->selected.contact_idx;
        printf("gp_canvas_on_click: adding edge sel %d.%d->%d.%d prospective_rope=%d\n",
            module_idx, contact_idx, ctx->selected.module, ctx->selected.contact_idx, ctx->prospective_rope_idx);
        int ei = gp_canvas_add_edge_with_type(reinterpret_cast<GP_CanvasContext*>(ctx), &e, /*type_id=*/0);
        printf("gp_canvas_on_click: gp_canvas_add_edge returned %d\n", ei);
        if (ei >= 0 && ctx->prospective_rope_idx >= 0) {
            ctx->edges[ei].rope_idx = ctx->prospective_rope_idx;
            printf("gp_canvas_on_click: attached rope %d to edge %d\n", ctx->prospective_rope_idx, ei);
            ctx->prospective_rope_idx = -1;
        } else {
            printf("gp_canvas_on_click: no rope attached (prospective=%d)\n", ctx->prospective_rope_idx);
        }
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
    canvas_init_subgroup_palette(c);
    canvas_ensure_root_table(c);
    // ensure a persistent canvas-root-reflection module is present
    canvas_ensure_root_module(c);
    c->thread_mgr = std::make_unique<ThreadManager>();
    c->thread_mgr->set_mode(ThreadManager::Mode::Scheduled);
    c->thread_mgr->start();
    // expose as global for table-layer integration
    ThreadManager::set_global(c->thread_mgr.get());
    c->autosave_path = kCanvasDefaultWorkspacePath;
    c->autosave_interval_s = 2.0;
    c->autosave_accum_s = 0.0;
    std::ifstream ifs(kCanvasDefaultWorkspacePath);
    if (ifs.good()) {
        gp_canvas_load_from_file(reinterpret_cast<GP_CanvasContext*>(c), kCanvasDefaultWorkspacePath);
    }
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
    c->module_frame_leds.push_back(make_module_frame_led_group());
    c->module_frame_links.emplace_back();
    c->module_uuids.push_back(0ull);
    c->module_preview_buffers.emplace_back();
    c->module_stack_tail.emplace_back();
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
    // Dispatch mouse-up event to any bound ports for CANVAS_ACT_MOUSE_UP
    canvas_dispatch_event_to_bound_ports(c, CANVAS_ACT_MOUSE_UP, world_x, world_y, false, true);
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
        const int edge_btn_count = 5;
        const int save_btn_count = 2;
        const int table_btn_count = 3;
        const int kpn_btn_count = 3;
        const int spacing = 8;
        int inner_h = std::max(0, c->control_bar_h - 8);
        int row_gap = 4;
        int row_h = std::max(4, (inner_h - row_gap) / 2);
        int by0 = c->rope_bar_h + 4;
        int by1 = by0 + row_h + row_gap;
        int bh = row_h;
        int bw = bh; // square buttons
        // left canvas group
        int bx = 8;
        int canvas_group_w = canvas_btn_count * (bw + spacing) - spacing;
        for (int bi = 0; bi < canvas_btn_count; ++bi) {
            int bx_i = bx + bi * (bw + spacing);
            if (view_x >= bx_i && view_x < bx_i + bw && view_y >= by0 && view_y < by0 + bh) {
                int action_id = CANVAS_ACT_TOOL_CANVAS_0 + bi;
                if (canvas_dispatch_root_action(c, action_id)) return 1;
            }
        }
        int bx_edge = bx + canvas_group_w + spacing * 2;
        for (int bi = 0; bi < edge_btn_count; ++bi) {
            int bx_i = bx_edge + bi * (bw + spacing);
            if (view_x >= bx_i && view_x < bx_i + bw && view_y >= by0 && view_y < by0 + bh) {
                int action_id = CANVAS_ACT_TOOL_EDGE_0 + bi;
                if (canvas_dispatch_root_action(c, action_id)) return 1;
            }
        }
        int group_width = table_btn_count * (bw + spacing) - spacing;
        int bx_r = std::max(8, c->width - 8 - group_width);
        int save_group_w = save_btn_count * (bw + spacing) - spacing;
        int bx_save = bx_r - save_group_w - spacing * 2;
        for (int bi = 0; bi < save_btn_count; ++bi) {
            int bx_i = bx_save + bi * (bw + spacing);
            if (view_x >= bx_i && view_x < bx_i + bw && view_y >= by0 && view_y < by0 + bh) {
                int action_id = (bi == 0) ? CANVAS_ACT_SAVE : CANVAS_ACT_CLEAR;
                if (canvas_dispatch_root_action(c, action_id)) return 1;
            }
        }
        // right table group
        for (int bi = 0; bi < table_btn_count; ++bi) {
            int bx_i = bx_r + bi * (bw + spacing);
            if (view_x >= bx_i && view_x < bx_i + bw && view_y >= by0 && view_y < by0 + bh) {
                int action_id = CANVAS_ACT_TOOL_TABLE_0 + bi;
                if (canvas_dispatch_root_action(c, action_id)) return 1;
            }
        }
        // IO controls to the left of table buttons: compute positions matching raster
        int io_base_x = bx_save - spacing * 2;
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
        if (view_y >= by0 && view_y < by0 + bh) {
            if (view_x >= bx_consumer && view_x < bx_consumer + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_IO_CONSUMER_ADD)) return 1;
            }
            if (view_x >= bx_producer && view_x < bx_producer + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_IO_PRODUCER_ADD)) return 1;
            }
        }
        if (view_y >= by0 && view_y < by0 + bh) {
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
        if (view_y >= by0 && view_y < by0 + bh) {
            if (view_x >= bx_minus && view_x < bx_minus + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_IO_COUNT_DEC)) return 1;
            }
            if (view_x >= bx_plus && view_x < bx_plus + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_IO_COUNT_INC)) return 1;
            }
        }
        // second row: kpn tools and thread manager controls
        int bx_kpn = 8;
        for (int bi = 0; bi < kpn_btn_count; ++bi) {
            int bx_i = bx_kpn + bi * (bw + spacing);
            if (view_x >= bx_i && view_x < bx_i + bw && view_y >= by1 && view_y < by1 + bh) {
                int action_id = CANVAS_ACT_TOOL_KPN_0 + bi;
                if (canvas_dispatch_root_action(c, action_id)) return 1;
            }
        }
        int delay_num_w = std::max(24, nbw * 2);
        int delay_gap = 10;
        int delay_total_w = nbw + delay_gap + delay_num_w + delay_gap + nbw;
        int play_gap = 12;
        int bx_play = c->width - 8 - bw;
        int action_btn_gap = 6;
        int action_btn_count = 3;
        int action_group_w = action_btn_count * bw + action_btn_gap * (action_btn_count - 1);
        int bx_action_right = bx_play - play_gap;
        int bx_action_left = bx_action_right - action_group_w;
        int bx_delay_plus = bx_action_left - play_gap;
        int bx_delay_minus = bx_delay_plus - delay_total_w;
        int kpn_group_w = kpn_btn_count * (bw + spacing) - spacing;
        int kpn_right = bx_kpn + kpn_group_w;
        int subgroup_btn_count = kSubgroupBinCount;
        int subgroup_group_w = subgroup_btn_count * (bw + spacing) - spacing;
        int subgroup_left = kpn_right + spacing * 2;
        int delay_left = bx_delay_minus;
        int available = delay_left - subgroup_left;
        if (available < subgroup_group_w) {
            subgroup_left = std::max(kpn_right + spacing, delay_left - subgroup_group_w);
        }
        for (int bi = 0; bi < subgroup_btn_count; ++bi) {
            int bx_i = subgroup_left + bi * (bw + spacing);
            if (view_x >= bx_i && view_x < bx_i + bw && view_y >= by1 && view_y < by1 + bh) {
                int action_id = CANVAS_ACT_TOOL_SUBGROUP_0 + bi;
                if (canvas_dispatch_root_action(c, action_id)) return 1;
            }
        }
        // spawn-root hit handling (to the right of subgroup colors)
        int bx_spawn = subgroup_left + subgroup_group_w + spacing;
        if (view_y >= by1 && view_y < by1 + bh) {
            if (bx_spawn + bw < bx_delay_plus) {
                if (view_x >= bx_spawn && view_x < bx_spawn + bw) {
                    if (canvas_dispatch_root_action(c, CANVAS_ACT_SPAWN_ROOT)) return 1;
                }
            }
        }
        // Check toolbar LED hitboxes (render-time populated) and treat them
        // as module LED hits on the canvas-root-reflection module so they
        // can act as rope anchors.
        if (!c->toolbar_leds.empty()) {
            int root_mod = canvas_ensure_root_module(c);
            if (root_mod >= 0) {
                for (const auto &tb : c->toolbar_leds) {
                    if (view_x >= tb.x0 && view_x < tb.x1 && view_y >= tb.y0 && view_y < tb.y1) {
                        GP_TableHitBox hb{};
                        // convert world coords into module-local coords
                        hb.x0 = tb.wx0 - c->modules[root_mod].x;
                        hb.x1 = tb.wx1 - c->modules[root_mod].x;
                        hb.y0 = tb.wy0 - c->modules[root_mod].y;
                        hb.y1 = tb.wy1 - c->modules[root_mod].y;
                        hb.row_idx = -1; hb.col_idx = -1;
                        // Convert toolbar LED hit into a distinct module-frame-style
                        // LED that does not collide with regular frame LEDs. Use
                        // a separate toolbar contact base so connectors are unique.
                        int toolbar_base = kModuleFrameContactBase + kModuleExtraLedCount * 2;
                        // mark as frame receive (consumer)
                        hb.row_idx = kModuleFrameRowReceive;
                        hb.col_idx = 0;
                        hb.part = GP_TABLE_HIT_LED;
                        hb.aux0 = toolbar_base + tb.subgroup_idx;
                        hb.aux1 = 0;
                        if (canvas_handle_module_led_hit(c, root_mod, hb)) return 1;
                    }
                }
            }
        }
        if (view_y >= by1 && view_y < by1 + bh) {
            // play button was split: left half = SIM, right half = play (global KPN)
            int half_play_w = std::max(8, bw / 2);
            int sim_x = bx_play;
            int sim_w = half_play_w;
            int play_x = bx_play + half_play_w;
            int play_w = bw - half_play_w;
            // SIM area (left half) - this is the toolbar SIM: toggle canvas-root sim
            if (view_x >= sim_x && view_x < sim_x + sim_w) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_CANVAS_ROOT_SIM_TOGGLE)) return 1;
            }
            // play area (right half) - global KPN pause/resume
            if (view_x >= play_x && view_x < play_x + play_w) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_KPN_GLOBAL_TOGGLE)) return 1;
            }
            // Module-level clone/clear/destroy buttons moved into module top UI.
            if (view_x >= bx_delay_minus && view_x < bx_delay_minus + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_THREAD_DELAY_DEC)) return 1;
            }
            if (view_x >= bx_delay_plus && view_x < bx_delay_plus + nbw) {
                if (canvas_dispatch_root_action(c, CANVAS_ACT_THREAD_DELAY_INC)) return 1;
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
            ToolMenuCounterLayout counter = compute_tool_menu_counter_layout(layout);
            if (view_y >= counter.by && view_y < counter.by + counter.h) {
                if (view_x >= counter.bx_minus && view_x < counter.bx_minus + counter.nbw) {
                    if (canvas_dispatch_root_action(c, CANVAS_ACT_TABLE_TOOL_NUM_DEC)) return 1;
                }
                if (view_x >= counter.bx_plus && view_x < counter.bx_plus + counter.nbw) {
                    if (canvas_dispatch_root_action(c, CANVAS_ACT_TABLE_TOOL_NUM_INC)) return 1;
                }
            }
            return 1;
        }
        c->tool_menu_open = false;
        c->selected_tool_table = 0;
    }
    if (c->plugin_menu_open) {
        PluginMenuLayout layout = compute_plugin_menu_layout(c, c->plugin_menu_module_idx, static_cast<int>(c->plugin_tool_ids.size()));
        if (view_x >= layout.x && view_x < layout.x + layout.w && view_y >= layout.y && view_y < layout.y + layout.h) {
            int idx = (view_y - layout.item_start_y) / std::max(1, layout.row_h);
                if (!c->plugin_tool_ids.empty() && idx >= 0 && idx < static_cast<int>(c->plugin_tool_ids.size())) {
                    // push plugin tool by id
                    canvas_push_plugin_tool_to_focused(c, c->plugin_tool_ids[static_cast<size_t>(idx)]);
            }
            c->plugin_menu_open = false;
            c->plugin_menu_module_idx = -1;
            return 1;
        }
        c->plugin_menu_open = false;
        c->plugin_menu_module_idx = -1;
    }
    for (int mi = 0; mi < static_cast<int>(c->modules.size()); ++mi) {
        const auto &m = c->modules[mi];
        ModuleLayout layout = module_layout_for(c, mi, m);
        if (world_x >= m.x && world_x < m.x + m.w && world_y >= m.y && world_y < m.y + m.h) {
            int lx = world_x - m.x;
            int ly = world_y - m.y;
            if (ly >= 0 && ly < layout.top_h) {
                // Check for clicks on the module frame pair-count control (single minus/plus)
                // Compute local positions used by the renderer so hit area matches the buttons
                const int control_h = std::max(1, kModuleControlRowH - 2);
                const int gap = 6;
                const int nbw = control_h;
                const int num_w = std::max(24, nbw * 2);
                // led rows start after title, thumb, control rows and gaps
                const int led_rows_y = kModuleTopPadding + kModuleTitleRowH + kModuleThumbRowH + kModuleTopGap + kModuleControlRowH + kModuleTopGap;
                const int led_row_h = kModuleLedRowH;
                const int btn_y = led_rows_y + (led_row_h - control_h) / 2;
                const int group_x = kModuleTopPadding;
                // minus button
                if (lx >= group_x && lx < group_x + nbw && ly >= btn_y && ly < btn_y + control_h) {
                    if (canvas_dispatch_root_action(c, CANVAS_ACT_FRAME_PAIRS_DEC)) return 1;
                }
                // plus button
                int plus_x = group_x + nbw + gap + num_w + gap;
                if (lx >= plus_x && lx < plus_x + nbw && ly >= btn_y && ly < btn_y + control_h) {
                    if (canvas_dispatch_root_action(c, CANVAS_ACT_FRAME_PAIRS_INC)) return 1;
                }

                GP_TableHitBox frame_hit{};
                if (find_module_frame_led_hit(c, mi, m, lx, ly, &frame_hit)) {
                    if (canvas_handle_module_led_hit(c, mi, frame_hit)) return 1;
                }
            }
        }
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
                ModuleLayout layout = module_layout_for(c, mi, m);
                int table_clip_h = std::max(1, layout.table_clip_h);
                int ly_table = ly - layout.table_y;
                if (ly_table < 0 || ly_table >= table_clip_h) continue;
                GP_TableGeom geom{};
                gp_table_get_geom(t, &geom);
                geom.width_px = tw;
                int table_render_h = std::max(1, table_clip_h);
                int th = std::max(1, table_render_h);
                std::vector<uint8_t> tmp(static_cast<size_t>(tw) * static_cast<size_t>(th) * 4);
                geom.height_px = th;
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
                        if (hb.y0 >= table_clip_h) continue;
                        if (lx >= hb.x0 && lx < hb.x1 && ly_table >= hb.y0 && ly_table < hb.y1) { found = hb; found_any = true; break; }
                    }
                    if (!found_any) {
                        printf("gp_canvas_on_click: module=%d table_hits_present=%d but none contain (%d,%d) local\n", mi, hits_written, lx, ly);
                        for (int hi = 0; hi < hits_written; ++hi) {
                            const GP_TableHitBox &hb = hits[hi];
                            printf("  hit[%d]=part=%d row=%d col=%d aux0=%d aux1=%d rect=%d,%d-%d,%d flags=0x%x\n", hi, hb.part, hb.row_idx, hb.col_idx, hb.aux0, hb.aux1, hb.x0, hb.y0, hb.x1, hb.y1, hb.flags);
                        }
                    }
                    if (found_any) {
                        GP_TableHitBox adjusted = found;
                        adjusted.y0 += layout.table_y;
                        adjusted.y1 += layout.table_y;
                        c->dispatch_module_idx = mi;
                        if (canvas_handle_module_counter_hit(c, mi, found)) {
                            c->dispatch_module_idx = -1;
                            return 1;
                        }
                        int handled = canvas_dispatch_root_hit(c, adjusted);
                        c->dispatch_module_idx = -1;
                        if (handled) return 1;
                        // If LED hit, handle canvas-level connection flow. In
                        // edge-drawing mode we avoid calling into the table so
                        // we don't toggle its internal selection state.
                        if (found.part == GP_TABLE_HIT_LED || found.part == GP_TABLE_HIT_LED_ARG || found.part == GP_TABLE_HIT_LED_TABLE) {
                            if (canvas_handle_module_led_hit(c, mi, adjusted)) return 1;
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
    if (c->selected_tool_subgroup_flags != 0u) {
        int edge_idx = canvas_pick_edge_by_rope(c, world_x, world_y, static_cast<float>(pick_r));
        if (edge_idx >= 0 && edge_idx < static_cast<int>(c->edges.size())) {
            uint32_t flags = c->selected_tool_subgroup_flags;
            canvas_apply_subgroup_to_edge_idx(c, edge_idx, flags);
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
        d.x = nx; d.y = ny; d.w = kModuleDefaultWidth; d.h = kModuleDefaultHeight;
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
        d.x = nx; d.y = ny; d.w = kModuleDefaultWidth; d.h = kModuleDefaultHeight;
        std::memset(d.label, 0, sizeof(d.label));
        std::memcpy(d.label, LABEL_MODULE_STAGE, std::min<size_t>(std::strlen(LABEL_MODULE_STAGE), sizeof(d.label) - 1));
        int new_idx = gp_canvas_add_module(ctx_, &d);
        if (new_idx >= 0) {
            // Mark as stage module and set up stage + frame table.
            canvas_configure_stage_module(c, new_idx, d.w, d.h);
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
            // Detect clicks on module-top action buttons (Clone/Clear/Destroy/Export)
            {
                ModuleLayout layout = module_layout_for(c, mi, m);
                if (layout.top_h > 0) {
                    int sx = m.x;
                    int sy = m.y;
                    int cursor_y = sy + kModuleTopPadding;
                    cursor_y += kModuleTitleRowH;
                    cursor_y += kModuleThumbRowH + kModuleTopGap;
                    int control_y = cursor_y;
                    int control_h = std::max(1, kModuleControlRowH - 2);
                    int btn_y = control_y + 1;
                    int nbw = control_h;
                    int gap = 6;
                    int right_x = sx + m.w - kModuleTopPadding;
                    int menu_w = std::max(30, control_h);
                    int pause_w = std::max(42, control_h * 2);
                    int menu_x = right_x - menu_w;
                    int pause_x = menu_x - gap - pause_w;
                    int module_btn_count = 5;
                    int module_btn_w = nbw;
                    int module_gap = 6;
                    int btns_total_w = module_btn_count * (module_btn_w + module_gap) - module_gap;
                    int btns_right = pause_x - module_gap;
                    int btns_left = btns_right - btns_total_w;
                    int bx = btns_left;
                    if (world_y >= btn_y && world_y < btn_y + control_h) {
                        // Clone
                        if (world_x >= bx && world_x < bx + module_btn_w) {
                            if (canvas_dispatch_root_action(c, CANVAS_ACT_MODULE_CLONE)) return 1;
                        }
                        bx += module_btn_w + module_gap;
                        // Clear
                        if (world_x >= bx && world_x < bx + module_btn_w) {
                            if (canvas_dispatch_root_action(c, CANVAS_ACT_MODULE_CLEAR)) return 1;
                        }
                        bx += module_btn_w + module_gap;
                        // Destroy
                        if (world_x >= bx && world_x < bx + module_btn_w) {
                            if (canvas_dispatch_root_action(c, CANVAS_ACT_MODULE_DESTROY)) return 1;
                        }
                        bx += module_btn_w + module_gap;
                        // Commit (build+load)
                        if (world_x >= bx && world_x < bx + module_btn_w) {
                            if (canvas_dispatch_root_action(c, CANVAS_ACT_MODULE_COMMIT)) return 1;
                        }
                        bx += module_btn_w + module_gap;
                        // Export
                        if (world_x >= bx && world_x < bx + module_btn_w) {
                            if (canvas_dispatch_root_action(c, CANVAS_ACT_MODULE_EXPORT)) return 1;
                        }
                        if (world_x >= menu_x && world_x < menu_x + menu_w) {
                            bool reopen = !(c->plugin_menu_open && c->plugin_menu_module_idx == mi);
                            c->plugin_menu_open = reopen;
                            c->plugin_menu_module_idx = reopen ? mi : -1;
                            if (reopen) {
                                canvas_refresh_plugin_tools(c);
                                c->tool_menu_open = false;
                                c->selected_tool_table = 0;
                            }
                            return 1;
                        }
                        // Per-module pause/play and sim-toggle buttons (drawn in top UI)
                        // Pause/play is split: left half = module SIM toggle, right half = module play/pause
                        int half_play_w = std::max(8, pause_w / 2);
                        int sim_x = pause_x;
                        int sim_w = half_play_w;
                        int play_x = pause_x + half_play_w;
                        int play_w = pause_w - half_play_w;
                        // Module SIM (left half)
                        if (world_x >= sim_x && world_x < sim_x + sim_w) {
                            c->dispatch_module_idx = mi;
                            bool handled = canvas_dispatch_root_action(c, CANVAS_ACT_THREAD_SIM_TOGGLE);
                            c->dispatch_module_idx = -1;
                            if (handled) return 1;
                        }
                        // Module play/pause (right half)
                        if (world_x >= play_x && world_x < play_x + play_w) {
                            c->dispatch_module_idx = mi;
                            bool handled = canvas_dispatch_root_action(c, CANVAS_ACT_THREAD_TOGGLE);
                            c->dispatch_module_idx = -1;
                            if (handled) return 1;
                        }
                    }
                }
            }
            if (mi < static_cast<int>(c->module_tables.size()) && c->module_tables[mi] && c->selected_tool_canvas == 0) {
                // local coords; only forward clicks into attached tables when
                // canvas is in select/interaction mode (tool 0). In edge-mode
                // we performed non-mutating hit tests earlier and should avoid
                // letting the table change its own selection state here.
                int lx = world_x - m.x;
                int ly = world_y - m.y;
                ModuleLayout layout = module_layout_for(c, mi, m);
                if (ly < layout.table_y || ly >= layout.table_y + layout.table_clip_h) break;
                GP_TableHitBox hb{};
                int ok = gp_table_on_click(c->module_tables[mi], lx, ly - layout.table_y, &hb);
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
        ModuleLayout layout = module_layout_for(c, mi, m);
        int drag_y = layout.drag_y;
        int drag_h = std::max(0, layout.drag_h);
        int drag_top = m.y + drag_y;
        int drag_bottom = drag_top + drag_h;
        if (world_x >= m.x && world_x < m.x + m.w && world_y >= drag_top && world_y < drag_bottom) {
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
    // Check overlays for direct drag (any click inside overlay should start moving it)
    for (auto &kv : c->overlays) {
        auto &ov = kv.second;
        // First: hit-test the renderer-cached control rect (screen coords)
        // even if the click is outside the overlay bbox. This ensures the
        // numeric control is clickable under all circumstances.
        if (ov.ctrl_w > 0) {
            if (x >= ov.ctrl_x && x <= ov.ctrl_x + ov.ctrl_w && y >= ov.ctrl_y && y <= ov.ctrl_y + ov.ctrl_h) {
                // If overlay carries an authoritative meta binding, dispatch
                // directly to that meta-group without rescanning ropes/tables.
                if (ov.meta_table && ov.meta_mg) {
                    int btn_w2 = ov.ctrl_h; int gap = 6; int bx_minus = ov.ctrl_x + 2; int bx_num = bx_minus + btn_w2 + gap; int num_w = ov.ctrl_w - (btn_w2*2 + gap*2) - 4; int bx_plus = bx_num + num_w + gap;
                    if (x >= bx_minus && x < bx_minus + btn_w2) {
                        void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_DEC);
                        if (pa) { auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa); p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1; p->hit.aux0 = ov.id; gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); }
                        return 1;
                    }
                    if (x >= bx_plus && x < bx_plus + btn_w2) {
                        void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_INC);
                        if (pa) { auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa); p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1; p->hit.aux0 = ov.id; gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); }
                        return 1;
                    }
                    // mid-area mapping
                    if (x >= ov.ctrl_x && x < ov.ctrl_x + ov.ctrl_w) {
                        int mid = ov.ctrl_x + ov.ctrl_w/2;
                        if (x < mid) {
                            void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_DEC);
                            if (pa) { auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa); p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1; p->hit.aux0 = ov.id; gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); }
                            return 1;
                        } else {
                            void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_INC);
                            if (pa) { auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa); p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1; p->hit.aux0 = ov.id; gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); }
                            return 1;
                        }
                    }
                }
                // Fallback: legacy behavior (find rope index and map to meta via dangling_rope)
                int rope_idx = -1;
                for (size_t ei = 0; ei < c->edges.size(); ++ei) {
                    const auto &e = c->edges[ei];
                    if (e.overlay_key_a == ov.key_a || e.overlay_key_b == ov.key_b) { rope_idx = e.rope_idx; break; }
                }
                if (rope_idx >= 0) {
                    // dump meta-group dangling rope indices at this higher level
                    auto dump_mgs = [&](GP_TableContext* t){
                        if (!t) return;
                        int mgcount = gp_table_get_meta_group_count(t);
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            int dr=-1,dv=-1; gp_table_meta_get_dangling_rope_info(t, mg, &dr, &dv);
                            printf("  early-dump table=%p mg[%d]=%p dangling_rope=%d vid=%d\n", (void*)t, mgi, (void*)mg, dr, dv);
                        }
                    };
                    dump_mgs(c->container_table);
                    for (size_t _mi = 0; _mi < c->module_tables.size(); ++_mi) dump_mgs(c->module_tables[_mi]);
                    auto try_handle_numeric = [&](GP_TableContext* t)->bool{
                        if (!t) return false;
                        int mgcount = gp_table_get_meta_group_count(t);
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            if (!mg) continue;
                            int _dr=-1,_dv=-1; gp_table_meta_get_dangling_rope_info(t, mg, &_dr, &_dv);
                            if (_dr != rope_idx) continue;
                            int btn_w2 = ov.ctrl_h; int gap = 6; int bx_minus = ov.ctrl_x + 2; int bx_num = bx_minus + btn_w2 + gap; int num_w = ov.ctrl_w - (btn_w2*2 + gap*2) - 4; int bx_plus = bx_num + num_w + gap;
                            if (x >= bx_minus && x < bx_minus + btn_w2) {
                                void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_DEC);
                                if (pa) { auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa); p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1; p->hit.aux0 = rope_idx; p->aux_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, rope_idx); gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); }
                                return true;
                            }
                            if (x >= bx_plus && x < bx_plus + btn_w2) {
                                void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_INC);
                                if (pa) { auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa); p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1; p->hit.aux0 = rope_idx; p->aux_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, rope_idx); gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); }
                                return true;
                            }
                            if (x >= ov.ctrl_x && x < ov.ctrl_x + ov.ctrl_w) {
                                int mid = ov.ctrl_x + ov.ctrl_w/2;
                                if (x < mid) {
                                    void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_DEC);
                                    if (pa) { auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa); p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1; p->hit.aux0 = rope_idx; p->aux_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, rope_idx); gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); }
                                    return true;
                                } else {
                                    void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_INC);
                                    if (pa) { auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa); p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1; p->hit.aux0 = rope_idx; p->aux_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, rope_idx); gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); }
                                    return true;
                                }
                            }
                        }
                        return false;
                    };
                    if (try_handle_numeric(c->container_table)) return 1;
                    for (size_t mi = 0; mi < c->module_tables.size(); ++mi) if (try_handle_numeric(c->module_tables[mi])) return 1;
                }
            }
        }
        int rx0 = static_cast<int>(std::floor(std::min(ov.x1, ov.x2)));
        int ry0 = static_cast<int>(std::floor(std::min(ov.y1, ov.y2)));
        int rx1 = static_cast<int>(std::ceil(std::max(ov.x1, ov.x2)));
        int ry1 = static_cast<int>(std::ceil(std::max(ov.y1, ov.y2)));
        // expand hitbox downward to include the numeric control drawn below the
        // tiny mode button so users can click the visible plus/minus area.
        const int extra_down = 48; // pixels
        ry1 += extra_down;
        if (world_x >= rx0 && world_x <= rx1 && world_y >= ry0 && world_y <= ry1) {
            c->drag.dragging = 1;
            c->drag.module = -1; // not module drag
            c->drag.overlay_id = kv.first;
            c->drag.overlay_offx = world_x - rx0;
            c->drag.overlay_offy = world_y - ry0;
            printf("gp_canvas_on_mouse_down: start overlay-drag canvas=%p overlay=%d off=%d,%d\n", (void*)c, kv.first, c->drag.overlay_offx, c->drag.overlay_offy);
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
    // Dispatch mouse-down event to any bound ports for CANVAS_ACT_MOUSE_DOWN
    canvas_dispatch_event_to_bound_ports(c, CANVAS_ACT_MOUSE_DOWN, world_x, world_y, true, false);
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
    if (ds.dragging) {
        if (ds.module >= 0) {
            int nx = world_x - ds.offx;
            int ny = world_y - ds.offy;
            c->modules[ds.module].x = nx;
            c->modules[ds.module].y = ny;
            //printf("gp_canvas_on_mouse_move: canvas=%p module=%d -> %d,%d\n", (void*)c, ds.module, nx, ny);
            handled = true;
        } else if (ds.overlay_id >= 0) {
            auto it = c->overlays.find(ds.overlay_id);
            if (it != c->overlays.end()) {
                auto &ov = it->second;
                int rx0 = static_cast<int>(std::floor(std::min(ov.x1, ov.x2)));
                int ry0 = static_cast<int>(std::floor(std::min(ov.y1, ov.y2)));
                int rx1 = static_cast<int>(std::ceil(std::max(ov.x1, ov.x2)));
                int ry1 = static_cast<int>(std::ceil(std::max(ov.y1, ov.y2)));
                int w = rx1 - rx0;
                int h = ry1 - ry0;
                int nx = world_x - ds.overlay_offx;
                int ny = world_y - ds.overlay_offy;
                // update overlay coordinates preserving size
                ov.x1 = static_cast<float>(nx);
                ov.x2 = static_cast<float>(nx + w);
                ov.y1 = static_cast<float>(ny);
                ov.y2 = static_cast<float>(ny + h);
                //printf("gp_canvas_on_mouse_move: moved overlay %d -> %d,%d\n", ds.overlay_id, nx, ny);
                handled = true;
            }
        }
    }
    if (handled) {
        update_canvas_scroll_state(c, /*pull_from_container=*/false);
        return 1;
    }
    // Dispatch mouse-move event to any bound ports for CANVAS_ACT_MOUSE_MOVE
    canvas_dispatch_event_to_bound_ports(c, CANVAS_ACT_MOUSE_MOVE, world_x, world_y, false, false);
    return 0;
}

extern "C" int gp_canvas_on_mouse_up(GP_CanvasContext* ctx_, int x, int y) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    printf("gp_canvas_on_mouse_up: called canvas=%p x=%d y=%d\n", (void*)c, x, y);
    canvas_record_mouse_input(c, x, y, /*down=*/false, /*up=*/true);
    (void)c->overlays.size();
    (void)c;
    // Check for overlay mode-button clicks even when not dragging
    int world_x = x + c->offset_x;
    int world_y = y + c->offset_y;
    for (const auto &kv : c->overlays) {
        const auto &ov = kv.second;
        int rx0 = static_cast<int>(std::floor(std::min(ov.x1, ov.x2)));
        int ry0 = static_cast<int>(std::floor(std::min(ov.y1, ov.y2)));
        int rx1 = static_cast<int>(std::ceil(std::max(ov.x1, ov.x2)));
        int ry1 = static_cast<int>(std::ceil(std::max(ov.y1, ov.y2)));
        if (world_x >= rx0 && world_x <= rx1 && world_y >= ry0 && world_y <= ry1) {
            // allow clicking near overlay endpoints (LED dots) to behave like
            // clicking the canonical root-module LED contacts so colors/edges
            // can be toggled/created. If click is close to either endpoint,
            // synthesize a module-LED hit routed to the root module.
            const int endpoint_radius = 12;
            auto dist2 = [](int ax,int ay,int bx,int by)->int { int dx = ax-bx; int dy = ay-by; return dx*dx + dy*dy; };
            int ex1 = static_cast<int>(std::lround(ov.x1));
            int ey1 = static_cast<int>(std::lround(ov.y1));
            int ex2 = static_cast<int>(std::lround(ov.x2));
            int ey2 = static_cast<int>(std::lround(ov.y2));
            int d1 = dist2(world_x, world_y, ex1, ey1);
            int d2 = dist2(world_x, world_y, ex2, ey2);
            if (d1 <= endpoint_radius * endpoint_radius || d2 <= endpoint_radius * endpoint_radius) {
                int rope_idx = -1;
                for (size_t ei = 0; ei < c->edges.size(); ++ei) {
                    const auto &e = c->edges[ei];
                    if (e.overlay_key_a == ov.key_a || e.overlay_key_b == ov.key_b) { rope_idx = e.rope_idx; break; }
                }
                // Map to canonical root contact indices: overlay attachments use
                // canonical root contacts 0 and 1 (gp_canvas_attach_rope_to_overlay)
                int root_mod = canvas_ensure_root_module(c);
                if (root_mod >= 0) {
                    GP_TableHitBox hb{};
                    hb.part = GP_TABLE_HIT_LED;
                    hb.cell_kind = GP_TABLE_CELL_LEDS;
                    hb.row_idx = -1; hb.col_idx = -1;
                    // choose contact 0 for endpoint A, 1 for endpoint B
                    hb.aux0 = (d1 <= d2) ? 0 : 1;
                    // dispatch as if clicked on module frame LED
                    (void)canvas_handle_module_led_hit(c, root_mod, hb);
                    // Ensure any overlay drag state is cleared so widgets don't stick
                    if (c->drag.dragging) {
                        c->drag.dragging = 0;
                        c->drag.panning = 0;
                        c->drag.module = -1;
                        c->drag.overlay_id = -1;
                        c->drag.overlay_offx = 0;
                        c->drag.overlay_offy = 0;
                    }
                    update_canvas_scroll_state(c, /*pull_from_container=*/false);
                    return 1;
                }
            }

                // Prefer hit-testing against the renderer-cached control rect (screen coords).
                if (ov.ctrl_w > 0) {
                    (void)ov.ctrl_x; (void)ov.ctrl_y; (void)ov.ctrl_w; (void)ov.ctrl_h; (void)x; (void)y;
                    if (x >= ov.ctrl_x && x <= ov.ctrl_x + ov.ctrl_w && y >= ov.ctrl_y && y <= ov.ctrl_y + ov.ctrl_h) {
                        int rope_idx = -1;
                        for (size_t ei = 0; ei < c->edges.size(); ++ei) {
                            const auto &e = c->edges[ei];
                            if (e.overlay_key_a == ov.key_a || e.overlay_key_b == ov.key_b) { rope_idx = e.rope_idx; break; }
                        }
                        if (rope_idx >= 0) {
                            // If this overlay carries an authoritative meta binding, handle numeric clicks
                            // directly via the bound meta-group to avoid brittle rope-index scans.
                            if (ov.meta_table && ov.meta_mg) {
                                int btn_w2 = ov.ctrl_h; int gap = 6; int bx_minus = ov.ctrl_x + 2; int bx_num = bx_minus + btn_w2 + gap; int num_w = ov.ctrl_w - (btn_w2*2 + gap*2) - 4; int bx_plus = bx_num + num_w + gap;
                                int cur = 0; gp_table_meta_get_channel_group(ov.meta_table, ov.meta_mg, &cur);
                                if (x >= bx_minus && x < bx_minus + btn_w2) {
                                    int nxt = std::clamp(cur - 1, -32768, 32767);
                                    gp_table_meta_set_channel_group(ov.meta_table, ov.meta_mg, nxt);
                                    update_canvas_scroll_state(c, /*pull_from_container=*/false);
                                    (void)ov.meta_mg; (void)ov.id; (void)nxt;
                                    return 1;
                                }
                                if (x >= bx_plus && x < bx_plus + btn_w2) {
                                    int nxt = std::clamp(cur + 1, -32768, 32767);
                                    gp_table_meta_set_channel_group(ov.meta_table, ov.meta_mg, nxt);
                                    update_canvas_scroll_state(c, /*pull_from_container=*/false);
                                    (void)ov.meta_mg; (void)ov.id; (void)nxt;
                                    return 1;
                                }
                            }
                            (void)rope_idx; (void)c;
                            // enumerate meta-groups in container and module tables to compare dangling rope indices
                            auto dump_mgs = [&](GP_TableContext* t){
                                if (!t) return;
                                int mgcount = gp_table_get_meta_group_count(t);
                                for (int mgi = 0; mgi < mgcount; ++mgi) {
                                    GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                                    int dr=-1,dv=-1; gp_table_meta_get_dangling_rope_info(t, mg, &dr, &dv);
                                    (void)t; (void)mgi; (void)mg; (void)dr; (void)dv;
                                }
                            };
                            dump_mgs(c->container_table);
                            for (size_t _mi = 0; _mi < c->module_tables.size(); ++_mi) dump_mgs(c->module_tables[_mi]);
                            auto try_handle_numeric = [&](GP_TableContext* t)->bool{
                                if (!t) return false;
                                int mgcount = gp_table_get_meta_group_count(t);
                                for (int mgi = 0; mgi < mgcount; ++mgi) {
                                    GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                                    if (!mg) continue;
                                    int _dr=-1,_dv=-1; gp_table_meta_get_dangling_rope_info(t, mg, &_dr, &_dv);
                                    if (_dr != rope_idx) continue;
                                    int btn_w2 = ov.ctrl_h; int gap = 6; int bx_minus = ov.ctrl_x + 2; int bx_num = bx_minus + btn_w2 + gap; int num_w = ov.ctrl_w - (btn_w2*2 + gap*2) - 4; int bx_plus = bx_num + num_w + gap;
                                    int cur = 0; gp_table_meta_get_channel_group(t, mg, &cur);
                                    // decide inc/dec based on screen X
                                    if (x >= bx_minus && x < bx_minus + btn_w2) {
                                        void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_DEC);
                                        if (pa) { auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa); p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1; p->hit.aux0 = rope_idx; p->aux_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, rope_idx); gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); }
                                        return true;
                                    }
                                    if (x >= bx_plus && x < bx_plus + btn_w2) {
                                        void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_INC);
                                        if (pa) { auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa); p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1; p->hit.aux0 = rope_idx; p->aux_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, rope_idx); gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa); }
                                        return true;
                                    }
                                }
                                return false;
                            };
                            if (try_handle_numeric(c->container_table)) return 1;
                            for (size_t mi = 0; mi < c->module_tables.size(); ++mi) if (try_handle_numeric(c->module_tables[mi])) return 1;
                        }
                    }
                } else {
                    // fallback: previous heuristic using overlay bbox/mode-button position
                    int btn_w = 20; int btn_h = 18; int margin = 6;
                    int bx = rx1 - margin - btn_w;
                    int by = ry0 + margin;
                    int ctrl_w = 88; int ctrl_h = 18;
                    int ctrl_x = bx + (btn_w/2) - (ctrl_w/2);
                    int ctrl_y = by + btn_h + 6;
                    (void)ctrl_x; (void)ctrl_y; (void)ctrl_w; (void)ctrl_h; (void)world_x; (void)world_y;
                    if (world_x >= ctrl_x && world_x <= ctrl_x + ctrl_w && world_y >= ctrl_y && world_y <= ctrl_y + ctrl_h) {
                        // If this overlay carries an authoritative meta binding, handle numeric clicks
                        // directly via the bound meta-group (world coords case).
                        if (ov.meta_table && ov.meta_mg) {
                            int btn_w2 = ctrl_h; int gap = 6; int bx_minus = ctrl_x + 2; int bx_num = bx_minus + btn_w2 + gap; int num_w = ctrl_w - (btn_w2*2 + gap*2) - 4; int bx_plus = bx_num + num_w + gap;
                            int cur = 0; gp_table_meta_get_channel_group(ov.meta_table, ov.meta_mg, &cur);
                            if (world_x >= bx_minus && world_x < bx_minus + btn_w2) {
                                int nxt = std::clamp(cur - 1, -32768, 32767);
                                gp_table_meta_set_channel_group(ov.meta_table, ov.meta_mg, nxt);
                                printf("gp_canvas_on_mouse_up: direct meta_set mg=%p ov=%d new_group=%d\n", (void*)ov.meta_mg, ov.id, nxt);
                                update_canvas_scroll_state(c, /*pull_from_container=*/false);
                                fflush(stdout);
                                return 1;
                            }
                            if (world_x >= bx_plus && world_x < bx_plus + btn_w2) {
                                int nxt = std::clamp(cur + 1, -32768, 32767);
                                gp_table_meta_set_channel_group(ov.meta_table, ov.meta_mg, nxt);
                                printf("gp_canvas_on_mouse_up: direct meta_set mg=%p ov=%d new_group=%d\n", (void*)ov.meta_mg, ov.id, nxt);
                                update_canvas_scroll_state(c, /*pull_from_container=*/false);
                                fflush(stdout);
                                return 1;
                            }
                        }
                        int rope_idx = -1;
                        for (size_t ei = 0; ei < c->edges.size(); ++ei) {
                            const auto &e = c->edges[ei];
                            if (e.overlay_key_a == ov.key_a || e.overlay_key_b == ov.key_b) { rope_idx = e.rope_idx; break; }
                        }
                        if (rope_idx >= 0) {
                            auto try_handle_numeric = [&](GP_TableContext* t)->bool{
                                if (!t) return false;
                                int mgcount = gp_table_get_meta_group_count(t);
                                for (int mgi = 0; mgi < mgcount; ++mgi) {
                                    GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                                    if (!mg) continue;
                                    int _dr=-1,_dv=-1; gp_table_meta_get_dangling_rope_info(t, mg, &_dr, &_dv);
                                    if (_dr != rope_idx) continue;
                                    int btn_w2 = ctrl_h; int gap = 6; int bx_minus = ctrl_x + 2; int bx_num = bx_minus + btn_w2 + gap; int num_w = ctrl_w - (btn_w2*2 + gap*2) - 4; int bx_plus = bx_num + num_w + gap;
                                    int cur = 0; gp_table_meta_get_channel_group(t, mg, &cur);
                                    if (world_x >= bx_minus && world_x < bx_minus + btn_w2) {
                                        void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_DEC);
                                        if (pa) {
                                            auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa);
                                            p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1;
                                            p->hit.aux0 = rope_idx;
                                            p->aux_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, rope_idx);
                                            gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa);
                                            gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa);
                                        }
                                        return true;
                                    }
                                    if (world_x >= bx_plus && world_x < bx_plus + btn_w2) {
                                                void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_INC);
                                                if (pa) {
                                                    auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa);
                                                    p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1;
                                                    p->hit.aux0 = rope_idx;
                                                    p->aux_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, rope_idx);
                                                    gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa);
                                                    gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa);
                                                }
                                                return true;
                                            }
                                }
                                return false;
                            };
                            if (try_handle_numeric(c->container_table)) return 1;
                            for (size_t mi = 0; mi < c->module_tables.size(); ++mi) if (try_handle_numeric(c->module_tables[mi])) return 1;
                        }
                    }
                }

            // check if click is within the small top-right mode button
            int btn_w = 20; int btn_h = 18; int margin = 6;
            int bx = rx1 - margin - btn_w;
            int by = ry0 + margin;
            if (world_x >= bx && world_x <= bx + btn_w && world_y >= by && world_y <= by + btn_h) {
                // Ensure any active drag state is cleared when clicking overlay buttons
                c->drag.dragging = 0;
                c->drag.panning = 0;
                c->drag.module = -1;
                c->drag.overlay_id = -1;
                c->drag.overlay_offx = 0;
                c->drag.overlay_offy = 0;
                // find rope idx associated with this overlay via edges
                int rope_idx = -1;
                for (size_t ei = 0; ei < c->edges.size(); ++ei) {
                    const auto &e = c->edges[ei];
                    if (e.overlay_key_a == ov.key_a || e.overlay_key_b == ov.key_b) { rope_idx = e.rope_idx; break; }
                }
                (void)ov.key_a; (void)ov.key_b; (void)rope_idx; (void)c;
                if (rope_idx >= 0) {
                    // Diagnostic: print canvas table pointers and module tables overview
                    (void)c;
                    for (size_t _mi = 0; _mi < c->module_tables.size(); ++_mi) { (void)c->module_tables[_mi]; }
                    auto try_toggle_on_table = [&](GP_TableContext* t)->bool{
                        (void)t;
                        if (!t) return false;
                        int mgcount = gp_table_get_meta_group_count(t);
                        (void)mgcount;
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            // Ensure this meta-group is associated with the overlay's rope
                            // so clicks only affect the same mg that rendering displays.
                            int _dr = -1, _dv = -1;
                            gp_table_meta_get_dangling_rope_info(t, mg, &_dr, &_dv);
                            if (_dr != rope_idx) continue;
                            if (!mg) continue;
                            // Check for clicks on the widget's numeric channel-group control.
                            int wwid = -1;
                            if (gp_table_meta_get_dangling_widget_id(t, mg, &wwid) && wwid >= 0) {
                                float wpos_local[3] = {0.0f,0.0f,0.0f};
                                bool have_pos = false;
                                int wcx = 0, wcy = 0;
                                if (gp_table_get_widget_position(t, wwid, wpos_local)) {
                                    wcx = static_cast<int>(std::lround(wpos_local[0]));
                                    wcy = static_cast<int>(std::lround(wpos_local[1]));
                                    have_pos = true;
                                } else {
                                    // Fallback: resolve overlay pixel coords (overlay created at widget-creation)
                                    int oxi=0, oyi=0;
                                    unsigned long long trykey = ov.key_a ? ov.key_a : ov.key_b;
                                    if (trykey && gp_canvas_resolve_overlay_key(reinterpret_cast<GP_CanvasContext*>(c), trykey, &oxi, &oyi)) {
                                        // convert overlay pixel coords to world coords for hit testing
                                        wcx = oxi + c->offset_x;
                                        wcy = oyi + c->offset_y;
                                        have_pos = true;
                                    }
                                }
                                if (have_pos) {
                                    int ctrl_w = 88; int ctrl_h = 18;
                                    // debug: report overlay/button and widget positions for hit-testing
                                    (void)mg; (void)wwid; (void)wcx; (void)wcy; (void)bx; (void)by; (void)btn_w; (void)btn_h; (void)world_x; (void)world_y;
                                    // compute control rect in world coords relative to overlay mode/button
                                    int ctrl_x = bx + (btn_w/2) - (ctrl_w/2);
                                    int ctrl_y = by + btn_h + 6;
                                    // world coords: world_x/world_y already in world space
                                    if (world_x >= ctrl_x && world_x <= ctrl_x + ctrl_w && world_y >= ctrl_y && world_y <= ctrl_y + ctrl_h) {
                                        int btn_w2 = ctrl_h; int gap = 6; int bx_minus = ctrl_x + 2; int bx_num = bx_minus + btn_w2 + gap;
                                        int num_w = ctrl_w - (btn_w2*2 + gap*2) - 4; int bx_plus = bx_num + num_w + gap;
                                        int cur = 0; gp_table_meta_get_channel_group(t, mg, &cur);
                                        if (world_x >= bx_minus && world_x < bx_minus + btn_w2) {
                                            // dispatch via canvas action path so behavior matches other numeric controls
                                            void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_DEC);
                                            if (pa) {
                                                auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa);
                                                p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1;
                                                p->hit.aux0 = rope_idx; // carry rope idx so root handler can find mg
                                                p->aux_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, rope_idx);
                                                (void)gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa);
                                                (void)gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa);
                                            }
                                            return true;
                                        }
                                        if (world_x >= bx_plus && world_x < bx_plus + btn_w2) {
                                            void* pa = gp_canvas_create_action_from_enum(reinterpret_cast<GP_CanvasContext*>(c), CANVAS_ACT_META_CHAN_INC);
                                            if (pa) {
                                                auto *p = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pa);
                                                p->hit.x0 = world_x; p->hit.y0 = world_y; p->hit.x1 = world_x+1; p->hit.y1 = world_y+1;
                                                p->hit.aux0 = rope_idx;
                                                p->aux_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, rope_idx);
                                                (void)gp_canvas_invoke_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa);
                                                (void)gp_canvas_free_pending_action(reinterpret_cast<GP_CanvasContext*>(c), pa);
                                            }
                                            return true;
                                        }
                                    }
                                }
                            }
                            int ar = -1, av = -1;
                            if (!gp_table_meta_get_anchor(t, mg, &ar, &av)) continue;
                            if (ar == rope_idx) {
                                int32_t cur = 0;
                                gp_table_meta_get_ring_mode(t, mg, &cur);
                                int32_t nxt = (cur + 1) % 3;
                                printf("gp_canvas_on_mouse_up: toggling mg=%p on table=%p cur=%d nxt=%d\n", (void*)mg, (void*)t, cur, nxt);
                                // If a host registered an overlay button handler, invoke it first.
                                if (c->overlay_button_cb) {
                                    int handled = c->overlay_button_cb(c->overlay_button_user, ov.key_a, ov.key_b, rope_idx, 0);
                                    if (handled) return true;
                                }
                                // Direct sim-only operation: find/create sim meta-group and toggle mode
                                {
                                    RopeSim* sim = gp_table_get_rope_sim(t);
                                    if (sim) {
                                        int mg_idx = rope_sim_find_meta_group_with_rope(sim, rope_idx);
                                        if (mg_idx < 0) {
                                            int vc = rope_sim_get_vertex_count(sim, static_cast<int>(rope_idx));
                                            if (vc > 1) {
                                                int sg = rope_sim_create_meta_group(sim, 1.0f);
                                                if (sg >= 0) {
                                                    rope_sim_meta_group_add(sim, sg, static_cast<int>(rope_idx), 0);
                                                    rope_sim_meta_group_add(sim, sg, static_cast<int>(rope_idx), vc - 1);
                                                    rope_sim_meta_group_set_mode(sim, sg, 1);
                                                    rope_sim_meta_group_disable_edge_springs(sim, sg);
                                                    rope_sim_meta_group_enable_edge_springs(sim, sg, 2.0f, 50.0f);
                                                    printf("gp_canvas_on_mouse_up: created sim-mg=%d for rope=%d\n", sg, rope_idx);
                                                    return true;
                                                }
                                            }
                                        } else {
                                            int cur = 0;
                                            if (rope_sim_meta_group_get_mode(sim, mg_idx, &cur)) {
                                                int nxt = (cur + 1) % 3;
                                                rope_sim_meta_group_set_mode(sim, mg_idx, nxt);
                                                rope_sim_meta_group_disable_edge_springs(sim, mg_idx);
                                                rope_sim_meta_group_enable_edge_springs(sim, mg_idx, 2.0f, 50.0f);
                                                printf("gp_canvas_on_mouse_up: toggled sim-mg=%d rope=%d cur=%d nxt=%d\n", mg_idx, rope_idx, cur, nxt);
                                                return true;
                                            }
                                        }
                                    }
                                }
                                return true;
                            }
                        }
                        // If no explicit anchor matched, try finding any member rope
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            if (!mg) continue;
                            int vcount = gp_table_meta_get_vertex_count(t, mg);
                            for (int vi = 0; vi < vcount; ++vi) {
                                int vr = -1, vv = -1;
                                if (!gp_table_meta_get_vertex(t, mg, vi, &vr, &vv)) continue;
                                if (vr == rope_idx) {
                                    int32_t cur = 0;
                                    gp_table_meta_get_ring_mode(t, mg, &cur);
                                    int32_t nxt = (cur + 1) % 3;
                                    printf("gp_canvas_on_mouse_up: toggling mg=%p on table=%p via member rope cur=%d nxt=%d (member_idx=%d)\n", (void*)mg, (void*)t, cur, nxt, vi);
                                    // Direct sim-only operation: find/create sim meta-group and toggle mode
                                    {
                                        RopeSim* sim = gp_table_get_rope_sim(t);
                                        if (sim) {
                                            int mg_idx = rope_sim_find_meta_group_with_rope(sim, rope_idx);
                                            if (mg_idx < 0) {
                                                int vc = rope_sim_get_vertex_count(sim, static_cast<int>(rope_idx));
                                                if (vc > 1) {
                                                    int sg = rope_sim_create_meta_group(sim, 1.0f);
                                                    if (sg >= 0) {
                                                        rope_sim_meta_group_add(sim, sg, static_cast<int>(rope_idx), 0);
                                                        rope_sim_meta_group_add(sim, sg, static_cast<int>(rope_idx), vc - 1);
                                                        rope_sim_meta_group_set_mode(sim, sg, 1);
                                                        rope_sim_meta_group_disable_edge_springs(sim, sg);
                                                        rope_sim_meta_group_enable_edge_springs(sim, sg, 2.0f, 50.0f);
                                                        printf("gp_canvas_on_mouse_up: created sim-mg=%d for rope=%d\n", sg, rope_idx);
                                                        return true;
                                                    }
                                                }
                                            } else {
                                                int cur = 0;
                                                if (rope_sim_meta_group_get_mode(sim, mg_idx, &cur)) {
                                                    int nxt = (cur + 1) % 3;
                                                    rope_sim_meta_group_set_mode(sim, mg_idx, nxt);
                                                    rope_sim_meta_group_disable_edge_springs(sim, mg_idx);
                                                    rope_sim_meta_group_enable_edge_springs(sim, mg_idx, 2.0f, 50.0f);
                                                    printf("gp_canvas_on_mouse_up: toggled sim-mg=%d rope=%d cur=%d nxt=%d\n", mg_idx, rope_idx, cur, nxt);
                                                    return true;
                                                }
                                            }
                                        }
                                    }
                                    return true;
                                }
                            }
                        }
                        return false;
                    };
                    if (try_toggle_on_table(c->container_table)) return 1;
                    for (size_t mi = 0; mi < c->module_tables.size(); ++mi) {
                        if (try_toggle_on_table(c->module_tables[mi])) return 1;
                    }
                    // No table-side meta-group found; attempt sim-only fallback
                    GP_TableContext* prefer = c->container_table;
                    if (!prefer) {
                        for (size_t mi = 0; mi < c->module_tables.size(); ++mi) { if (c->module_tables[mi]) { prefer = c->module_tables[mi]; break; } }
                    }
                    if (prefer) {
                        if (c->overlay_button_cb) {
                            int handled = c->overlay_button_cb(c->overlay_button_user, ov.key_a, ov.key_b, rope_idx, 0);
                            if (handled) return 1;
                        }
                        int simres = gp_table_sim_add_meta_group_for_rope(prefer, rope_idx);
                        printf("gp_canvas_on_mouse_up: sim-only fallback for rope=%d res=%d\n", rope_idx, simres);
                        if (simres) return 1;
                    }
                }
                else {
                    // debug: dump any edges that carry overlay keys to understand mapping
                    printf("gp_canvas_on_mouse_up: no edge matched overlay; edges_count=%zu\n", c->edges.size());
                    for (size_t ei = 0; ei < c->edges.size(); ++ei) {
                        const auto &e = c->edges[ei];
                        if (e.overlay_key_a || e.overlay_key_b) {
                            printf("  edge[%zu] overlay_a=%llu overlay_b=%llu rope_idx=%d\n", ei, (unsigned long long)e.overlay_key_a, (unsigned long long)e.overlay_key_b, e.rope_idx);
                        }
                    }
                }
            }
        }
    }
    if (!c->drag.dragging) return 0;
    c->drag.dragging = 0;
    c->drag.panning = 0;
    printf("gp_canvas_on_mouse_up: canvas=%p module=%d\n", (void*)c, c->drag.module);
    c->drag.module = -1;
    c->drag.overlay_id = -1;
    c->drag.overlay_offx = 0;
    c->drag.overlay_offy = 0;
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
        GP_TableContext* root_tbl = canvas_ensure_root_table(c);
        // publish global skip count and advance global tick so tables make a
        // consistent decision within this frame
        gp_table_set_global_sim_frame_skip_count(c->sim_frame_skip_count);
        gp_table_advance_global_sim_tick();
        // Only step the root rope sim if the table-level sim is enabled and
        // the global sim-skip cadence allows stepping this frame.
        if (root_tbl && gp_table_should_step_sim(root_tbl)) {
            rope_sim_step(sim, dt, c->sim_maxforce, c->sim_iters, c->sim_damping);
        }
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
    if (c->thread_mgr && !c->thread_mgr_paused) {
        const double delay_s = static_cast<double>(std::max(0, c->thread_mgr_delay_ms)) / 1000.0;
        bool should_submit = true;
        if (delay_s <= 0.0) {
            c->thread_mgr_delay_accum_s = 0.0;
        } else {
            c->thread_mgr_delay_accum_s += static_cast<double>(dt);
            if (c->thread_mgr_delay_accum_s + 1e-9 < delay_s) {
                should_submit = false;
            } else {
                c->thread_mgr_delay_accum_s = std::max(0.0, c->thread_mgr_delay_accum_s - delay_s);
            }
        }
        if (should_submit) {
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
                        // propagate per-module sim-enabled flag (default to enabled)
                        if (mi >= 0 && mi < static_cast<int>(c->module_sim_enabled.size())) mod.sim_enabled = c->module_sim_enabled[mi];
                        else mod.sim_enabled = 1;
                        // propagate per-module sim-enabled flag (default to enabled)
                        // propagate per-module full-skip flag (default to not skipped)
                        if (mi >= 0 && mi < static_cast<int>(c->module_skip.size())) mod.module_skip = c->module_skip[mi];
                        else mod.module_skip = 0;
                    // propagate per-module execution cadence (skip count). 0 => run every frame.
                    if (mi >= 0 && mi < static_cast<int>(c->module_exec_skip_count.size())) mod.exec_skip_count = c->module_exec_skip_count[mi];
                    else mod.exec_skip_count = 0;
                req.modules.push_back(mod);
            }
            // Update global sim frame-skip count so table stepping decisions
            // use the current canvas-level setting.
            gp_table_set_global_sim_frame_skip_count(c->sim_frame_skip_count);
            // Advance the global sim tick once per canvas frame.
            gp_table_advance_global_sim_tick();
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
                        uint8_t* row = tmp.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * 4u;
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
        // Bind any pre-existing meta-group overlay keys to canvas overlays
        // (covers case where table was deserialized earlier before canvas existed)
        int mgc = gp_table_get_meta_group_count(table);
        for (int m = 0; m < mgc; ++m) {
            GP_MetaGroup* mg = gp_table_get_meta_group(table, m);
            if (!mg) continue;
            unsigned long long ka = 0ull, kb = 0ull;
            gp_table_meta_get_overlay_keys(table, mg, &ka, &kb);
            if ((ka != 0ull || kb != 0ull)) {
                gp_canvas_set_overlay_meta(ctx_, ka, kb, table, reinterpret_cast<void*>(mg));
            }
        }
        // Attempt to load module serialized data from module_library (module_library/serialized/module_X.gpmod)
        if (module_idx >= 0) {
            std::string mid = gp_module_library_module_id(module_idx);
            std::string rel = gp_module_library_module_serialized_path(std::string(), mid);
            std::string root = gp_module_library_default_root();
            std::filesystem::path full = std::filesystem::path(root) / rel;
            try {
                if (std::filesystem::exists(full) && std::filesystem::file_size(full) > 0) {
                    std::ifstream ifs(full, std::ios::binary);
                    if (ifs.good()) {
                        std::vector<char> buf((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                        ifs.close();
                        if (!buf.empty()) {
                            printf("gp_canvas_attach_table: loading serialized module %s from %s (size=%zu)\n", mid.c_str(), full.string().c_str(), buf.size());
                            gp_table_deserialize(table, buf.data(), static_cast<int32_t>(buf.size()));
                        }
                    }
                }
            } catch (...) {
                // ignore errors reading module serialized file
            }
        }
    }
    return 1;
}

extern "C" int gp_canvas_set_module_frame_ptr(GP_CanvasContext* ctx_, int module_idx, int is_send, int led_idx, void* ptr) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->module_frame_links.size())) return 0;
    if (led_idx < 0 || led_idx >= kModuleExtraLedCount) return 0;
    auto &links = c->module_frame_links[module_idx];
    // Before overwriting, remove any existing action_port_bindings associated
    // with the links we'll clear so the registry stays consistent.
    for (int row = is_send ? 0 : 2; row < (is_send ? 2 : 4); ++row) {
        void* oldp = links.ptrs[static_cast<size_t>(row)][static_cast<size_t>(led_idx)];
        if (oldp) {
            auto *oldpa = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(oldp);
            if (oldpa) {
                int32_t aid = oldpa->action_id;
                std::lock_guard<std::mutex> lk(c->action_subscribers_mu);
                auto it = c->action_port_bindings.find(aid);
                if (it != c->action_port_bindings.end()) {
                    auto &vec = it->second;
                    for (auto vit = vec.begin(); vit != vec.end(); ) {
                        if (vit->module_idx == module_idx && vit->row == row && vit->led_idx == led_idx && vit->pending_ptr == oldp) {
                            vit = vec.erase(vit);
                        } else ++vit;
                    }
                    if (vec.empty()) c->action_port_bindings.erase(it);
                }
            }
        }
    }
    // map legacy single send/receive API to the 4 logical rows by duplicating
    // into both left and right columns for the appropriate send/receive rows.
    if (is_send) {
        links.ptrs[0][static_cast<size_t>(led_idx)] = ptr; // send-left
        links.ptrs[1][static_cast<size_t>(led_idx)] = ptr; // send-right (duplicate)
    } else {
        links.ptrs[2][static_cast<size_t>(led_idx)] = ptr; // receive-left
        links.ptrs[3][static_cast<size_t>(led_idx)] = ptr; // receive-right (duplicate)
    }
    // Update the visual LED cell state so bound ports appear bright and
    // unbound ports appear dim. We use the cell's reserved0 as the "active"
    // indicator (non-zero => bright). Keep this consistent with
    // gp_canvas_bind_pending_action_to_module which sets reserved0=1.
    GP_TableCell* cell = nullptr;
    if (is_send) {
        cell = module_frame_led_cell(c, module_idx, 0, led_idx);
        if (cell) cell->reserved0 = ptr ? 1 : 0;
        cell = module_frame_led_cell(c, module_idx, 1, led_idx);
        if (cell) cell->reserved0 = ptr ? 1 : 0;
    } else {
        cell = module_frame_led_cell(c, module_idx, 2, led_idx);
        if (cell) cell->reserved0 = ptr ? 1 : 0;
        cell = module_frame_led_cell(c, module_idx, 3, led_idx);
        if (cell) cell->reserved0 = ptr ? 1 : 0;
    }
    return 1;
}

extern "C" void* gp_canvas_create_action_from_enum(GP_CanvasContext* ctx_, int32_t action_id) {
    if (!ctx_) return nullptr;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    auto *pa = new GP_CanvasContextImpl::PendingAction();
    pa->action_id = action_id;
    std::memset(&pa->hit, 0, sizeof(pa->hit));
    return reinterpret_cast<void*>(pa);
}

extern "C" int gp_canvas_bind_action_ptr_to_module_port(GP_CanvasContext* ctx_, int module_idx, int is_send, int col, int led_idx, void* pending_ptr) {
    if (!ctx_ || module_idx < 0 || !pending_ptr) return 0;
    if (led_idx < 0 || led_idx >= kModuleExtraLedCount) return 0;
    if (col < 0 || col > 1) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->module_frame_links.size())) return 0;
    // map to logical row: send-left=0, send-right=1, receive-left=2, receive-right=3
    int row = is_send ? (col == 0 ? 0 : 1) : (col == 0 ? 2 : 3);
    c->module_frame_links[module_idx].ptrs[static_cast<size_t>(row)][static_cast<size_t>(led_idx)] = pending_ptr;
    // update the canvas-side LED cell reserved0 so the canvas renderer shows it bright
    GP_TableCell* cell = module_frame_led_cell(c, module_idx, row, led_idx);
    if (cell) cell->reserved0 = 1;

    // Register this bound pending action in the action->port binding map so
    // events can be delivered to bound ports by action id.
    auto *pa = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pending_ptr);
    if (pa) {
        int32_t aid = pa->action_id;
        std::lock_guard<std::mutex> lk(c->action_subscribers_mu);
        GP_CanvasContextImpl::ActionBinding ab{};
        ab.module_idx = module_idx; ab.row = row; ab.col = col; ab.led_idx = led_idx; ab.pending_ptr = pending_ptr;
        // create root edge and writer_key if possible (stored on binding)
        // Try to create/ensure a root-table edge for this binding so tools can
        // observe and re-route the event FIFO via subgroup flags. We compute a
        // canonical writer key based on module/contact and create a paired
        // root edge if possible.
        GP_TableContext* root = canvas_ensure_root_table(c);
        if (root) {
            int contact_idx = kModuleFrameContactBase + row * kModuleExtraLedCount + led_idx;
            unsigned long long ka = (static_cast<unsigned long long>(static_cast<uint32_t>(module_idx)) << 32) |
                                    (static_cast<unsigned long long>(static_cast<uint32_t>(contact_idx)) << 16) |
                                    static_cast<unsigned long long>(0);
            unsigned long long kb = ka + 1ull; // complementary endpoint
            gp_table_add_edge(root, ka, kb);
            int idx = -1;
            if (gp_table_edge_index_for_key(root, ka, &idx) && idx >= 0) {
                ab.root_edge_idx = idx;
                ab.writer_key = ka;
            }
        }
        c->action_port_bindings[aid].push_back(std::move(ab));
    }

    // Also attempt to set per-table LED glow on the attached module table so
    // the bound port is visibly highlighted inside the module's table.
    if (module_idx < static_cast<int>(c->module_tables.size())) {
        GP_TableContext* t = c->module_tables[module_idx];
        if (t && module_idx < static_cast<int>(c->module_table_rows.size()) && !c->module_table_rows[module_idx].empty()) {
            const auto &rows = c->module_table_rows[module_idx];
            // For visual mapping, highlight the input column for receives and
            // the output column for sends. Iterate rows and set glow for any
            // row that contains at least `led_idx+1` attachments.
            int target_col = is_send ? kModuleColRightLed : kModuleColLeftLed;
            for (int ri = 0; ri < static_cast<int>(rows.size()); ++ri) {
                const auto &meta = rows[ri];
                if ((is_send && meta.kind != ModuleRowKind::Output) || (!is_send && meta.kind != ModuleRowKind::Input)) continue;
                if (meta.attachment_count <= led_idx) continue;
                // local led index inside this row is the same `led_idx` position
                (void)gp_table_set_led_glow(t, ri, target_col, led_idx, 1.0f);
                (void)gp_table_set_led_selected(t, ri, target_col, led_idx, 1);
            }
        }
    }
    return 1;
}

extern "C" int gp_canvas_bind_action_enum_to_module_port(GP_CanvasContext* ctx_, int module_idx, int is_send, int col, int led_idx, int32_t action_id) {
    if (!ctx_) return 0;
    void* pa = gp_canvas_create_action_from_enum(ctx_, action_id);
    if (!pa) return 0;
    if (!gp_canvas_bind_action_ptr_to_module_port(ctx_, module_idx, is_send, col, led_idx, pa)) {
        // cleanup on failure
        gp_canvas_free_pending_action(ctx_, pa);
        return 0;
    }
    return 1;
}

extern "C" int gp_canvas_invoke_pending_action(GP_CanvasContext* ctx_, void* pending_ptr) {
    if (!ctx_ || !pending_ptr) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    auto *pa = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pending_ptr);
    if (!pa) return 0;
    GP_TableContext* root = canvas_ensure_root_table(c);
    if (!root) return 0;
    // Temporarily disable click-listen so dispatch executes action semantics
    bool old_listen = c->click_listen_mode;
    c->click_listen_mode = false;
    // First, consult any registered subscribers for this action id. If any
    // subscriber reports the event as handled (returns non-zero), skip the
    // default canvas dispatch. Otherwise fall back to gp_table_dispatch_hit.
    bool handled = false;
    {
        std::lock_guard<std::mutex> lk(c->action_subscribers_mu);
        auto it = c->action_subscribers.find(pa->action_id);
        if (it != c->action_subscribers.end()) {
            for (const auto &p : it->second) {
                GP_CanvasActionSubscriberFn cb = p.first;
                void* user = p.second;
                if (!cb) continue;
                int r = 0;
                try { r = cb(user, reinterpret_cast<GP_CanvasContext*>(c), pa->action_id, &pa->hit); } catch (...) { r = 0; }
                if (r) { handled = true; break; }
            }
        }
    }
    if (!handled) {
        // Short-circuit: if this pending action targets meta channel inc/dec and
        // aux0 encodes an overlay id, prefer the overlay's authoritative binding
        // so we can update meta channel group directly without relying on table
        // hitbox dispatch (which may fail in some overlay->rope mismatch cases).
        if ((pa->action_id == CANVAS_ACT_META_CHAN_DEC || pa->action_id == CANVAS_ACT_META_CHAN_INC) && pa->hit.aux0 > 0) {
            int overlay_id = pa->hit.aux0;
            auto oit = c->overlays.find(overlay_id);
            if (oit != c->overlays.end()) {
                auto &ov = oit->second;
                if (ov.meta_table && ov.meta_mg) {
                    int delta = (pa->action_id == CANVAS_ACT_META_CHAN_INC) ? 1 : -1;
                    int cur = 0; gp_table_meta_get_channel_group(ov.meta_table, ov.meta_mg, &cur);
                    cur = std::clamp(cur + delta, -32768, 32767);
                    gp_table_meta_set_channel_group(ov.meta_table, ov.meta_mg, cur);
                    update_canvas_scroll_state(c, /*pull_from_container=*/false);
                    (void)ov.meta_mg; (void)overlay_id; (void)cur; (void)pa;
                    handled = true;
                }
            }
        }
        if (!handled) {
            int dispatched = gp_table_dispatch_hit(root, &pa->hit);
            (void)dispatched; (void)pa;
        }
    }
    c->click_listen_mode = old_listen;
    return 1;
}

// Note: internal queued pending-action API removed. Events are published
// into root table FIFO edges via gp_edge_publish_ptr and consumed by the
// manager using normal FIFO pointer-mode semantics.

// Deliver a synthesized event to all ports bound to `action_id`.
static void canvas_dispatch_event_to_bound_ports(GP_CanvasContextImpl* c, int32_t action_id, int x, int y, bool down, bool up) {
    if (!c) return;
    std::vector<GP_CanvasContextImpl::ActionBinding> copy;
    {
        std::lock_guard<std::mutex> lk(c->action_subscribers_mu);
        auto it = c->action_port_bindings.find(action_id);
        if (it == c->action_port_bindings.end()) return;
        copy = it->second; // shallow copy to avoid holding lock while invoking
    }
    GP_TableContext* root = canvas_ensure_root_table(c);
    for (const auto &b : copy) {
        if (!b.pending_ptr) continue;
        auto *orig = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(b.pending_ptr);
        if (!orig) continue;
        auto *copy_pa = new GP_CanvasContextImpl::PendingAction();
        copy_pa->action_id = orig->action_id;
        GP_TableHitBox h{};
        h.x0 = x; h.y0 = y; h.x1 = x+1; h.y1 = y+1;
        h.row_idx = -1; h.col_idx = -1; h.cell_kind = GP_TABLE_CELL_LEDS;
        h.part = GP_TABLE_HIT_LED; h.aux0 = b.led_idx; h.aux1 = 0; h.flags = 0;
        copy_pa->hit = h;
        copy_pa->aux_uid = 0ull;
        // Publish a pointer-mode EventPayload into the root table FIFO for
        // this binding's edge so external tools can observe and route the event.
        if (root && b.root_edge_idx >= 0) {
            EventPayload* ep = new EventPayload{reinterpret_cast<void*>(copy_pa), b.module_idx, b.led_idx};
            int dropped = 0;
            (void)gp_edge_publish_ptr(root, b.root_edge_idx, b.writer_key, reinterpret_cast<void*>(ep), &dropped);
        } else {
            // If no root FIFO exists for this binding, drop the copy to avoid leaks.
            delete copy_pa;
        }
    }
}

extern "C" int gp_canvas_subscribe_action(GP_CanvasContext* ctx_, int32_t action_id, GP_CanvasActionSubscriberFn cb, void* user) {
    if (!ctx_ || !cb) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    std::lock_guard<std::mutex> lk(c->action_subscribers_mu);
    auto &vec = c->action_subscribers[action_id];
    // avoid duplicate identical subscriptions
    for (const auto &p : vec) if (p.first == cb && p.second == user) return 1;
    vec.emplace_back(cb, user);
    return 1;
}

extern "C" int gp_canvas_unsubscribe_action(GP_CanvasContext* ctx_, int32_t action_id, GP_CanvasActionSubscriberFn cb, void* user) {
    if (!ctx_ || !cb) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    std::lock_guard<std::mutex> lk(c->action_subscribers_mu);
    auto it = c->action_subscribers.find(action_id);
    if (it == c->action_subscribers.end()) return 0;
    auto &vec = it->second;
    for (auto vit = vec.begin(); vit != vec.end(); ++vit) {
        if (vit->first == cb && vit->second == user) {
            vec.erase(vit);
            if (vec.empty()) c->action_subscribers.erase(it);
            return 1;
        }
    }
    return 0;
}

extern "C" int gp_canvas_free_pending_action(GP_CanvasContext* ctx_, void* pending_ptr) {
    if (!ctx_ || !pending_ptr) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    auto *pa = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(pending_ptr);
    if (!pa) return 0;
    // Only delete if the pointer is not the currently held pending_action
    // (if it is, transfer ownership back to canvas and free safely).
    if (c->pending_action == pa) {
        delete c->pending_action;
        c->pending_action = nullptr;
        return 1;
    }
    delete pa;
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
    // Ensure a single authoritative root module that references the container table.
    // This keeps `root_module_idx` and `container_table` consistent and owned by the canvas.
    canvas_ensure_root_module(c);
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

extern "C" int gp_canvas_set_table_tool_number(GP_CanvasContext* ctx_, int value) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->table_tool_number = std::clamp(value, 0, 99);
    return 1;
}

extern "C" int gp_canvas_get_table_tool_number(GP_CanvasContext* ctx_, int* out_value) {
    if (!ctx_ || !out_value) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    *out_value = c->table_tool_number;
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

extern "C" int gp_canvas_set_lasso_callback(GP_CanvasContext* ctx_, GP_CanvasLassoFn cb, void* user) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->lasso_cb = cb;
    c->lasso_cb_user = user;
    return 1;
}

extern "C" int gp_canvas_set_click_drag_callback(GP_CanvasContext* ctx_, GP_CanvasClickDragFn cb, void* user) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->click_drag_cb = cb;
    c->click_drag_cb_user = user;
    return 1;
}

extern "C" int gp_canvas_set_overlay_button_callback(GP_CanvasContext* ctx_, GP_CanvasOverlayButtonFn cb, void* user) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->overlay_button_cb = cb;
    c->overlay_button_user = user;
    return 1;
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

    GP_CanvasContextImpl::EdgeInfo ei{};
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
        // Diagnostic: report node-group assignments for these keys (helps debug handshake/group mismatches)
        int gka = 0, gkb = 0;
        if (!gp_table_node_group_get(root, ka, &gka)) gka = -1;
        if (!gp_table_node_group_get(root, kb, &gkb)) gkb = -1;
        printf("gp_canvas_add_edge_with_type: root node groups ka_group=%d kb_group=%d\n", gka, gkb);
        if (gka == -1 || gkb == -1) {
            int ma_id = (desc->a_module < static_cast<int>(c->module_node_id.size())) ? c->module_node_id[desc->a_module] : -1;
            int mb_id = (desc->b_module < static_cast<int>(c->module_node_id.size())) ? c->module_node_id[desc->b_module] : -1;
            printf("gp_canvas_add_edge_with_type: module_node_id a_module=%d->nid=%d b_module=%d->nid=%d\n", desc->a_module, ma_id, desc->b_module, mb_id);
            // Print node contracts for these node ids if present
            for (const auto &n : c->nodes) {
                if (n.node_id == ma_id || n.node_id == mb_id) {
                    printf("gp_canvas_add_edge_with_type: node nid=%d module=%d inputs=%zu outputs=%zu\n", n.node_id, n.module_idx, n.input_types.size(), n.output_types.size());
                }
            }
        }
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
// Format (V4):
// CANVAS V4
// WIDTH HEIGHT CONTROL_BAR_H
// OFFSET offx offy
// MODULE x y w h in_count out_count is_stage bg_mode label
// ROWS module_idx row_count [kind tool attachment_count tool_origin]...
// EDGE a_module a_contact_idx b_module b_contact_idx type_id subgroup_flags
// FRAMELED module_idx row_idx led_idx flags reserved0
// NODE node_id module_idx input_count [inputs...] output_count [outputs...]
// Lines may appear in any order; loader will reconstruct internal vectors.
extern "C" int gp_canvas_save_to_file(GP_CanvasContext* ctx_, const char* path) {
    if (!ctx_ || !path) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    std::ofstream ofs(path);
    if (!ofs.good()) return 0;
    // Ensure the canvas root/container table exists so we can detect
    // meta-groups stored there and register its persistent rope_ids.
    GP_TableContext* root_tbl = canvas_ensure_root_table(c);
    printf("gp_canvas_save_to_file: ensured container_table=%p\n", (void*)root_tbl);
    // Flush any pending ops that may have enqueued edge additions so
    // the root/table rope_ids vectors are populated before serialization.
    if (root_tbl) {
        gp_table_apply_pending_ops(root_tbl);
    }
    for (size_t ti = 0; ti < c->module_tables.size(); ++ti) {
        GP_TableContext* mt = c->module_tables[ti];
        if (!mt) continue;
        gp_table_apply_pending_ops(mt);
    }
    if (root_tbl) {
        int root_cnt = gp_table_get_rope_id_count(root_tbl);
        printf("gp_canvas_save_to_file: container_table rope_id_count=%d\n", root_cnt);
        if (root_cnt > 0) {
            std::vector<uint64_t> tmp; tmp.resize(static_cast<size_t>(root_cnt));
            int got = gp_table_get_rope_ids(root_tbl, tmp.data(), root_cnt);
            if (got > 0) gp_canvas_register_table_rope_ids_from_array(reinterpret_cast<GP_CanvasContext*>(c), root_tbl, tmp.data(), got);
        }
    }
    ofs << "CANVAS V4\n";
    ofs << c->width << " " << c->height << " " << c->control_bar_h << "\n";
    ofs << "OFFSET " << c->offset_x << " " << c->offset_y << "\n";
    // modules
    for (size_t i = 0; i < c->modules.size(); ++i) {
        const auto &m = c->modules[i];
        int in_count = (i < c->module_io_in_count.size()) ? c->module_io_in_count[i] : 0;
        int out_count = (i < c->module_io_out_count.size()) ? c->module_io_out_count[i] : 0;
        int is_stage = (i < c->module_is_stage.size() && c->module_is_stage[i]) ? 1 : 0;
        int bg_mode = (i < c->module_bg.size()) ? c->module_bg[i].mode : 0;
        // label may contain spaces; write as remainder of line
        ofs << "MODULE " << m.x << " " << m.y << " " << m.w << " " << m.h << " "
            << in_count << " " << out_count << " " << is_stage << " " << bg_mode << " ";
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
    // emit module UUIDs
    for (size_t i = 0; i < c->module_uuids.size(); ++i) {
        uint64_t mid = c->module_uuids[i];
        if (mid != 0ull) ofs << "MODULE_UUID " << static_cast<int>(i) << " " << mid << "\n";
    }
    // Export per-module serialized table blobs so canvas loads can restore meta-groups
    // Writes to module_library/serialized/module_X.gpmod via gp_canvas_export_module_to_root
    for (size_t i = 0; i < c->modules.size(); ++i) {
        // attempt to write module serialized state; ignore failures
        try {
            gp_canvas_export_module_to_root(reinterpret_cast<GP_CanvasContext*>(c), static_cast<int>(i), nullptr);
        } catch (...) { }
    }
    // Before exporting per-module blobs, inject any canvas-registered rope ids
    // into the corresponding table contexts so their serialized blobs include
    // the canonical persistent rope ids recorded in `rope_id_map`.
    {
        std::unordered_map<GP_TableContext*, std::vector<uint64_t>> table_to_ids;
        for (const auto &kv : c->rope_id_map) {
            uint64_t id = kv.second.rope_id;
            GP_TableContext* t = kv.second.table;
            if (!t) continue;
            table_to_ids[t].push_back(id);
        }
        for (const auto &ti : table_to_ids) {
            GP_TableContext* t = ti.first;
            const std::vector<uint64_t> &vec = ti.second;
            gp_table_set_rope_ids_from_array(t, vec.data(), static_cast<int>(vec.size()));
        }
    }
    for (size_t i = 0; i < c->modules.size(); ++i) {
        GP_TableContext* t = (i < c->module_tables.size()) ? c->module_tables[i] : nullptr;
        if (!t) {
            // still create placeholder file so loader sees an empty file
            gp_canvas_export_module_to_root(ctx_, static_cast<int>(i), nullptr);
        } else {
            gp_canvas_export_module_to_root(ctx_, static_cast<int>(i), nullptr);
        }
    }
    // Persist the root/container table as a standalone blob so canvas-level
    // lasso ropes and overlay bindings survive save/load cycles even when
    // module tables don't own those ropes. Future steps will mirror the
    // contained data back into modules and dedupe on load.
    {
        std::string root = gp_module_library_default_root();
        std::string rel = gp_module_library_module_serialized_path(std::string(), "container");
        std::filesystem::path full = std::filesystem::path(root) / rel;
        try {
            std::filesystem::create_directories(full.parent_path());
        } catch (...) {
            // ignore failures to keep save best-effort
        }
        if (root_tbl) {
            int need = gp_table_serialize(root_tbl, nullptr, 0);
            if (need > 0) {
                std::vector<char> buf(static_cast<size_t>(need));
                int wrote = gp_table_serialize(root_tbl, buf.data(), need);
                if (wrote == need) {
                    std::ofstream rofs(full, std::ios::binary | std::ios::trunc);
                    if (rofs.good()) {
                        rofs.write(buf.data(), static_cast<std::streamsize>(buf.size()));
                        rofs.close();
                    }
                }
            }
        }
        // TODO(step2): push root-contained ropes/edges down into owning module tables before serialization.
    }
    // Persist canvas-level meta-groups explicitly so meta-edges survive
    // canvas save/load cycles. For each module table enumerate its meta-groups
    // and emit a small, textual block describing vertices and restored fields.
    for (size_t i = 0; i < c->modules.size(); ++i) {
        GP_TableContext* t = (i < c->module_tables.size()) ? c->module_tables[i] : nullptr;
        if (!t) continue;
        int mgcount = gp_table_get_meta_group_count(t);
        for (int mgi = 0; mgi < mgcount; ++mgi) {
            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
            if (!mg) continue;
            int channel_group = 0; gp_table_meta_get_channel_group(t, mg, &channel_group);
            int anchor_r = -1, anchor_v = -1; gp_table_meta_get_anchor(t, mg, &anchor_r, &anchor_v);
            uint64_t anchor_uid = 0ull;
            if (anchor_r >= 0) {
                anchor_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, anchor_r);
            }
            uint32_t sgflags = 0; gp_table_meta_get_subgroup_flags(t, mg, &sgflags);
            int ring_mode = 0; gp_table_meta_get_ring_mode(t, mg, &ring_mode);
            float ring_u = 0.0f;
            // attempt to find a registered ring associated with this meta-group
            int ring_count = gp_table_get_ring_edge_count(t);
            for (int rei = 0; rei < ring_count; ++rei) {
                int ring_id = -1; unsigned long long ring_key = 0ull;
                if (!gp_table_get_ring_edge(t, rei, &ring_id, &ring_key)) continue;
                if (ring_key == 0ull) continue;
                unsigned long long _mgid_local = 0ull; gp_table_meta_get_id(t, mg, &_mgid_local);
                if (ring_key == _mgid_local) {
                    gp_table_get_ring_u(t, ring_id, &ring_u);
                    break;
                }
            }
            unsigned long long oka = 0ull, okb = 0ull; gp_table_meta_get_overlay_keys(t, mg, &oka, &okb);
            unsigned int lflags = 0; int32_t lwt = 0; gp_table_meta_get_lasso_fields(t, mg, &lflags, &lwt);
            float conf = 0.0f; gp_table_meta_get_confinement(t, mg, &conf);
            unsigned long long mgid = 0ull; gp_table_meta_get_id(t, mg, &mgid);
            float dang_len = 0.0f; gp_table_meta_get_dangling_hang_len(t, mg, &dang_len);
            int vcount = gp_table_meta_get_vertex_count(t, mg);
            ofs << "META_GROUP " << static_cast<int>(i) << " " << mgi << " " << channel_group << " " << anchor_uid << " " << anchor_v << " " << sgflags << " " << ring_mode << " " << ring_u << " " << oka << " " << okb << " " << conf << " " << mgid << " " << dang_len << "\n";
            printf("gp_canvas_save_to_file: META_LASSO flags=%u widget=%d (table=%p mg=%p)\n", lflags, lwt, (void*)t, (void*)mg);
            ofs << "META_LASSO " << lflags << " " << lwt << "\n";
            ofs << "META_VERTS " << vcount;
            for (int vi = 0; vi < vcount; ++vi) {
                int rr = 0, vv = 0; gp_table_meta_get_vertex(t, mg, vi, &rr, &vv);
                uint64_t rope_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, rr);
                ofs << " " << rope_uid << " " << vv;
            }
            ofs << "\n";
        }
    }
    // Also persist any canvas overlays that have meta bindings (canvas-level)
    for (const auto &p : c->overlays) {
        const auto &ov = p.second;
        if (!ov.meta_mg) continue;
        GP_MetaGroup* mg = ov.meta_mg;
        GP_TableContext* t = ov.meta_table;
        if (!t) continue; // need table context for ABI getters
        // Debug: report module_tables pointers and meta counts to aid lookup
        printf("gp_canvas_save_to_file: module_tables.size=%zu\n", c->module_tables.size());
        for (size_t ti = 0; ti < c->module_tables.size(); ++ti) {
            GP_TableContext* tt = c->module_tables[ti];
            int mgcount = tt ? gp_table_get_meta_group_count(tt) : -1;
            printf("  module_table[%zu]=%p mgcount=%d\n", ti, (void*)tt, mgcount);
        }
        printf("  container_table=%p\n", (void*)c->container_table);
        if (c->container_table) printf("  container mgcount=%d\n", gp_table_get_meta_group_count(c->container_table));
        printf("  overlay id=%d meta_table=%p meta_mg=%p\n", p.first, (void*)t, (void*)mg);
        printf("  overlay.meta_table==container_table? %d\n", (int)(t == c->container_table));
        fflush(stdout);
        // log that we're emitting a canvas-level meta-group
        printf("gp_canvas_save_to_file: emitting canvas META_GROUP overlay_idx=%d table=%p mg=%p\n", p.first, (void*)t, (void*)mg);
        fflush(stdout);
        int channel_group = 0; gp_table_meta_get_channel_group(t, mg, &channel_group);
        int anchor_r = -1, anchor_v = -1; gp_table_meta_get_anchor(t, mg, &anchor_r, &anchor_v);
        uint64_t anchor_uid = 0ull;
        if (anchor_r >= 0) {
            anchor_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, anchor_r);
        }
        uint32_t sgflags = 0; gp_table_meta_get_subgroup_flags(t, mg, &sgflags);
        int ring_mode = 0; gp_table_meta_get_ring_mode(t, mg, &ring_mode);
        float ring_u = 0.0f;
        int ring_count = gp_table_get_ring_edge_count(t);
        unsigned long long mgid = 0ull; gp_table_meta_get_id(t, mg, &mgid);
        for (int rei = 0; rei < ring_count; ++rei) {
            int ring_id = -1; unsigned long long ring_key = 0ull;
            if (!gp_table_get_ring_edge(t, rei, &ring_id, &ring_key)) continue;
            if (ring_key == mgid) { gp_table_get_ring_u(t, ring_id, &ring_u); break; }
        }
        unsigned long long oka = 0ull, okb = 0ull; gp_table_meta_get_overlay_keys(t, mg, &oka, &okb);
        unsigned int lflags = 0; int32_t lwt = 0; gp_table_meta_get_lasso_fields(t, mg, &lflags, &lwt);
        float conf = 0.0f; gp_table_meta_get_confinement(t, mg, &conf);
        
        float dang_len = 0.0f; gp_table_meta_get_dangling_hang_len(t, mg, &dang_len);
        int vcount = gp_table_meta_get_vertex_count(t, mg);
        // Prefer emitting the backing module index + meta-index so loader
        // can restore the meta-group into the correct table even if
        // overlays haven't been created yet. Fall back to module_idx=-1
        // and overlay index if we can't resolve the backing table/index.
        int backing_module_idx = -1;
        int backing_mgi = -1;
        // Implement: for each module table do a per-table search, then check
        // the canvas root (container) before moving to the next module.
        // Also check the provided backing table `t` first as a fast-path.
        bool located = false;
        if (t) {
            int mgcount = gp_table_get_meta_group_count(t);
            for (int mgi = 0; mgi < mgcount; ++mgi) {
                GP_MetaGroup* mg2 = gp_table_get_meta_group(t, mgi);
                if (mg2 == mg) {
                    if (t == c->container_table) { backing_module_idx = -1; backing_mgi = mgi; }
                    else {
                        for (size_t ti = 0; ti < c->module_tables.size(); ++ti) {
                            if (c->module_tables[ti] == t) { backing_module_idx = static_cast<int>(ti); backing_mgi = mgi; break; }
                        }
                    }
                    located = true;
                    break;
                }
            }
        }
        if (!located) {
            // Stage 2: check the canvas root/container table
            if (c->container_table) {
                int root_mgcount = gp_table_get_meta_group_count(c->container_table);
                for (int mgi = 0; mgi < root_mgcount; ++mgi) {
                    GP_MetaGroup* mg2 = gp_table_get_meta_group(c->container_table, mgi);
                    if (mg2 == mg) { backing_module_idx = -1; backing_mgi = mgi; located = true; printf("gp_canvas_save_to_file: found meta-group in container_table mg=%p mgi=%d\n", (void*)mg, mgi); break; }
                }
            }
        }
        if (!located) {
            // Stage 3: breadth-first search across module tables; short-circuit on first match
            size_t nmods = c->module_tables.size();
            if (nmods > 0) {
                std::vector<char> visited(nmods, 0);
                std::deque<size_t> q;
                // seed BFS from the module that owns `t`, else root module, else 0
                std::vector<size_t> starts;
                for (size_t mi = 0; mi < nmods; ++mi) {
                    if (c->module_tables[mi] == t) { starts.push_back(mi); break; }
                }
                if (starts.empty()) {
                    if (c->root_module_idx >= 0 && c->root_module_idx < static_cast<int>(nmods)) {
                        starts.push_back(static_cast<size_t>(c->root_module_idx));
                    } else {
                        starts.push_back(0);
                    }
                }
                for (size_t s : starts) {
                    if (s < nmods && !visited[s]) { visited[s] = 1; q.push_back(s); }
                }

                while (!q.empty() && !located) {
                    size_t idx = q.front(); q.pop_front();
                    GP_TableContext* tt = c->module_tables[idx];
                    if (tt && tt != t) {
                        int mgcount = gp_table_get_meta_group_count(tt);
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg2 = gp_table_get_meta_group(tt, mgi);
                            if (mg2 == mg) { backing_module_idx = static_cast<int>(idx); backing_mgi = mgi; located = true; break; }
                        }
                    }
                    if (located) break;
                    // enqueue neighbors from canvas edges graph
                    for (const auto &ei : c->edges) {
                        int a = ei.desc.a_module;
                        int b = ei.desc.b_module;
                        if (a == static_cast<int>(idx) && b >= 0 && b < static_cast<int>(nmods) && !visited[static_cast<size_t>(b)]) {
                            visited[static_cast<size_t>(b)] = 1; q.push_back(static_cast<size_t>(b));
                        } else if (b == static_cast<int>(idx) && a >= 0 && a < static_cast<int>(nmods) && !visited[static_cast<size_t>(a)]) {
                            visited[static_cast<size_t>(a)] = 1; q.push_back(static_cast<size_t>(a));
                        }
                    }
                }
            }
        }
            if (backing_module_idx >= 0 && backing_mgi >= 0) {
                // try to find an overlay entry bound to this meta-group so we can emit its rect
                float ox1 = 0.0f, oy1 = 0.0f, ox2 = 0.0f, oy2 = 0.0f;
                for (const auto &pp : c->overlays) {
                    const auto &ov = pp.second;
                    if (ov.meta_table == t && ov.meta_mg == mg) { ox1 = ov.x1; oy1 = ov.y1; ox2 = ov.x2; oy2 = ov.y2; break; }
                }
                ofs << "META_GROUP " << backing_module_idx << " " << backing_mgi << " " << channel_group << " " << anchor_uid << " " << anchor_v << " " << sgflags << " " << ring_mode << " " << ring_u << " " << oka << " " << okb << " " << conf << " " << mgid << " " << dang_len << " " << ox1 << " " << oy1 << " " << ox2 << " " << oy2 << "\n";
            } else {
                // fallback: emit module_idx = -1 and overlay index in second column
                const auto &ov = p.second;
                ofs << "META_GROUP " << -1 << " " << static_cast<int>(p.first) << " " << channel_group << " " << anchor_uid << " " << anchor_v << " " << sgflags << " " << ring_mode << " " << ring_u << " " << oka << " " << okb << " " << conf << " " << mgid << " " << dang_len << " " << ov.x1 << " " << ov.y1 << " " << ov.x2 << " " << ov.y2 << "\n";
            }
        printf("gp_canvas_save_to_file: META_LASSO flags=%u widget=%d overlay_idx=%d table=%p mg=%p\n", lflags, lwt, p.first, (void*)t, (void*)mg);
        ofs << "META_LASSO " << lflags << " " << lwt << "\n";
        ofs << "META_VERTS " << vcount;
        for (int vi = 0; vi < vcount; ++vi) {
            int rr = 0, vv = 0; gp_table_meta_get_vertex(t, mg, vi, &rr, &vv);
            uint64_t rope_uid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(c), t, rr);
            ofs << " " << rope_uid << " " << vv;
        }
        ofs << "\n";
    }
    // module IO row layout
    for (size_t i = 0; i < c->module_io_rows.size(); ++i) {
        const auto &rows = c->module_io_rows[i];
        if (rows.empty()) continue;
        ofs << "ROWS " << i << " " << rows.size();
        for (const auto &row : rows) {
            ofs << " " << static_cast<int>(row.kind) << " " << static_cast<int>(row.tool) << " "
                << row.attachment_count << " " << static_cast<int>(row.tool_origin);
            if (row.tool_origin == ModuleToolOrigin::Plugin && !row.plugin_id.empty()) {
                ofs << " " << row.plugin_id;
            }
        }
        ofs << "\n";
    }
    // edges
    for (const auto &ei : c->edges) {
        const auto &e = ei.desc;
        ofs << "EDGE " << e.a_module << " " << e.a_contact_idx << " " << e.b_module << " " << e.b_contact_idx << " "
            << ei.type_id << " " << ei.subgroup_flags << "\n";
    }
    for (size_t mi = 0; mi < c->module_frame_leds.size(); ++mi) {
        const auto &group = c->module_frame_leds[mi];
        for (int row = 0; row < kModuleExtraLedRows; ++row) {
            for (int idx = 0; idx < kModuleExtraLedCount; ++idx) {
                const GP_TableCell &cell = group.cells[static_cast<size_t>(row)][static_cast<size_t>(idx)];
                if (cell.flags == 0u && cell.reserved0 == 0) continue;
                ofs << "FRAMELED " << mi << " " << row << " " << idx << " " << cell.flags << " " << cell.reserved0 << "\n";
            }
        }
    }
    // persist module-frame port UUIDs
    for (size_t mi = 0; mi < c->module_frame_links.size(); ++mi) {
        const auto &links = c->module_frame_links[mi];
        for (int row = 0; row < kModuleExtraLedRows; ++row) {
            for (int idx = 0; idx < kModuleExtraLedCount; ++idx) {
                uint64_t pu = links.port_uuids[static_cast<size_t>(row)][static_cast<size_t>(idx)];
                if (pu != 0ull) ofs << "FRAMEPORTUUID " << static_cast<int>(mi) << " " << row << " " << idx << " " << pu << "\n";
            }
        }
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
    struct ModuleSnapshot {
        GP_CanvasModuleDesc desc{};
        int in_count = 0;
        int out_count = 0;
        int is_stage = 0;
        int bg_mode = 0;
        uint64_t module_uuid = 0ull;
    };
    struct EdgeSnapshot {
        GP_CanvasEdgeDesc desc{};
        int type_id = 0;
        uint32_t subgroup_flags = 0u;
    };
    struct RowSnapshot {
        int module_idx = -1;
        std::vector<ModuleIORow> rows;
    };
    struct FrameLedSnapshot {
        int module_idx = -1;
        int row = 0;
        int idx = 0;
        uint32_t flags = 0u;
        int reserved0 = 0;
    };

    

    int version = 1;
    int file_w = c->width;
    int file_h = c->height;
    int file_cbh = c->control_bar_h;
    int file_offx = 0;
    int file_offy = 0;
    std::vector<ModuleSnapshot> modules;
    std::vector<EdgeSnapshot> edges;
    std::vector<RowSnapshot> row_sets;
    std::vector<FrameLedSnapshot> frame_leds;
    struct FramePortSnapshot { int module_idx = -1; int row = 0; int idx = 0; uint64_t port_uuid = 0ull; };
    std::vector<FramePortSnapshot> frame_port_uuids;
    std::vector<GP_CanvasContextImpl::NodeContract> nodes;
    std::unordered_map<int, std::vector<char>> module_blobs;
    std::vector<CanvasMetaSnapshot> meta_group_snapshots;

    std::string line;
    bool has_header = false;
    if (std::getline(ifs, line)) {
        if (line.rfind("CANVAS", 0) == 0) {
            has_header = true;
            std::istringstream header(line);
            std::string canvas_token;
            std::string version_token;
            header >> canvas_token >> version_token;
            if (version_token == "V2") version = 2;
            else if (version_token == "V3") version = 3;
            else if (version_token == "V4") version = 4;
            if (!std::getline(ifs, line)) return 0;
            std::istringstream sh(line);
            sh >> file_w >> file_h >> file_cbh;
        } else {
            ifs.clear();
            ifs.seekg(0);
        }
    }
    if (!has_header) {
        version = 1;
    }

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tag; ss >> tag;
        if (tag == "MODULE") {
            ModuleSnapshot snap{};
            ss >> snap.desc.x >> snap.desc.y >> snap.desc.w >> snap.desc.h;
            if (version >= 2) {
                ss >> snap.in_count >> snap.out_count >> snap.is_stage >> snap.bg_mode;
            }
            std::string rest;
            std::getline(ss, rest);
            if (!rest.empty() && rest[0] == ' ') rest.erase(0,1);
            std::string lbl; lbl.reserve(rest.size());
            for (size_t i = 0; i < rest.size(); ++i) {
                char ch = rest[i];
                if (ch == '\\' && i + 1 < rest.size()) {
                    char nx = rest[i+1];
                    if (nx == 'n') { lbl.push_back('\n'); ++i; }
                    else { lbl.push_back(nx); ++i; }
                } else lbl.push_back(ch);
            }
            std::memset(snap.desc.label, 0, sizeof(snap.desc.label));
            std::memcpy(snap.desc.label, lbl.c_str(), std::min<size_t>(lbl.size(), sizeof(snap.desc.label)-1));
            modules.push_back(std::move(snap));
        } else if (tag == "MODULE_UUID") {
            int midx = -1; uint64_t mu = 0ull; ss >> midx >> mu;
            if (midx >= 0) {
                if (midx < static_cast<int>(modules.size())) modules[midx].module_uuid = mu;
                else {
                    // ensure vector size
                    modules.resize(static_cast<size_t>(midx+1));
                    modules[midx].module_uuid = mu;
                }
            }
        } else if (tag == "ROWS") {
            RowSnapshot rs{};
            int row_count = 0;
            ss >> rs.module_idx >> row_count;
            for (int i = 0; i < row_count; ++i) {
                int kind_int = 0;
                int tool_int = 0;
                int attachment_count = 0;
                int origin_int = 0;
                ss >> kind_int >> tool_int >> attachment_count;
                if (version >= 4) {
                    if (!(ss >> origin_int)) origin_int = 0;
                }
                ModuleIORow row{};
                row.kind = static_cast<ModuleRowKind>(kind_int);
                row.tool = static_cast<ModuleToolKind>(tool_int);
                row.contact_idx = 0;
                row.attachment_count = attachment_count;
                row.tool_origin = static_cast<ModuleToolOrigin>(origin_int);
                if (row.tool_origin == ModuleToolOrigin::Plugin) {
                    std::string pid;
                    if (ss >> pid) row.plugin_id = pid;
                }
                rs.rows.push_back(row);
            }
            row_sets.push_back(std::move(rs));
        } else if (tag == "EDGE") {
            EdgeSnapshot es{};
            ss >> es.desc.a_module >> es.desc.a_contact_idx >> es.desc.b_module >> es.desc.b_contact_idx >> es.type_id;
            if (version >= 3) {
                ss >> es.subgroup_flags;
            } else if (ss.good()) {
                uint32_t maybe_flags = 0u;
                if (ss >> maybe_flags) es.subgroup_flags = maybe_flags;
            }
            edges.push_back(std::move(es));
        } else if (tag == "FRAMELED") {
            FrameLedSnapshot fs{};
            ss >> fs.module_idx >> fs.row >> fs.idx >> fs.flags >> fs.reserved0;
            frame_leds.push_back(std::move(fs));
        } else if (tag == "FRAMEPORTUUID") {
            FramePortSnapshot fps{}; ss >> fps.module_idx >> fps.row >> fps.idx >> fps.port_uuid; frame_port_uuids.push_back(std::move(fps));
        } else if (tag == "NODE") {
            GP_CanvasContextImpl::NodeContract nc{};
            ss >> nc.node_id >> nc.module_idx;
            int in_count = 0; ss >> in_count;
            for (int i = 0; i < in_count; ++i) { int t; ss >> t; nc.input_types.push_back(t); }
            int out_count = 0; ss >> out_count;
            for (int i = 0; i < out_count; ++i) { int t; ss >> t; nc.output_types.push_back(t); }
            nodes.push_back(std::move(nc));
        } else if (tag == "OFFSET") {
            ss >> file_offx >> file_offy;
        } else if (tag == "META_GROUP") {
            // META_GROUP header: module_idx mgi channel_group anchor_uid anchor_v subgroup_flags ring_mode ring_u oka okb confinement mgid dangling_len overlay_x1 overlay_y1 overlay_x2 overlay_y2
            int module_idx = -1; int mgi = -1; int channel_group = 0; uint64_t anchor_uid = 0ull; int anchor_v = -1; unsigned int sgflags = 0; int ring_mode = 0; float ring_u = 0.0f; unsigned long long oka = 0ull, okb = 0ull; float conf = 0.0f; unsigned long long mgid = 0ull; float dang_len = 0.0f; float ox1 = 0.0f, oy1 = 0.0f, ox2 = 0.0f, oy2 = 0.0f;
            ss >> module_idx >> mgi >> channel_group >> anchor_uid >> anchor_v >> sgflags >> ring_mode >> ring_u >> oka >> okb >> conf >> mgid >> dang_len >> ox1 >> oy1 >> ox2 >> oy2;
            // read lasso line
            std::string lasso_line;
            if (!std::getline(ifs, lasso_line)) break;
            std::istringstream ls(lasso_line);
            std::string ltag; ls >> ltag;
            unsigned int lflags = 0; int32_t lwt = 0;
            if (ltag == "META_LASSO") { ls >> lflags >> lwt; }
            // read verts line
            std::string verts_line;
            if (!std::getline(ifs, verts_line)) break;
            std::istringstream vs(verts_line);
            std::string vtag; vs >> vtag;
            std::vector<CanvasMetaVert> verts;
            if (vtag == "META_VERTS") {
                int vcount = 0; vs >> vcount;
                for (int vi = 0; vi < vcount; ++vi) {
                    uint64_t rope_uid = 0ull; int vv = 0;
                    vs >> rope_uid >> vv;
                    verts.push_back(CanvasMetaVert{rope_uid, vv});
                }
            }
            // store snapshot for creation after modules are added
            CanvasMetaSnapshot snap;
            snap.module_idx = module_idx;
            snap.meta_slot = mgi;
            snap.verts = std::move(verts);
            snap.channel_group = channel_group;
            snap.anchor_rope_id = anchor_uid;
            snap.anchor_v = anchor_v;
            snap.subgroup_flags = sgflags;
            snap.ring_mode = ring_mode;
            snap.ring_u = ring_u;
            snap.overlay_a = oka;
            snap.overlay_b = okb;
            snap.overlay_x1 = ox1;
            snap.overlay_y1 = oy1;
            snap.overlay_x2 = ox2;
            snap.overlay_y2 = oy2;
            snap.confinement = conf;
            snap.mgid = mgid;
            snap.dangling_len = dang_len;
            snap.lasso_flags = lflags;
            snap.lasso_widget_type = lwt;
            meta_group_snapshots.push_back(std::move(snap));
        }
    }

    canvas_clear_workspace(c);
    c->width = file_w;
    c->height = file_h;
    c->control_bar_h = std::max(file_cbh, 56);
    c->offset_x = file_offx;
    c->offset_y = file_offy;

    std::unordered_map<int, std::vector<ModuleIORow>> row_map;
    for (const auto &rs : row_sets) {
        row_map[rs.module_idx] = rs.rows;
    }

    for (size_t i = 0; i < modules.size(); ++i) {
        const auto &snap = modules[i];
        int new_idx = gp_canvas_add_module(ctx_, &snap.desc);
        if (new_idx < 0) continue;
        // Attempt to restore a per-table template by module label (non-module workflows)
        try {
            std::string lbl(snap.desc.label, snap.desc.label + sizeof(snap.desc.label));
            size_t zpos = lbl.find('\0'); if (zpos != std::string::npos) lbl.resize(zpos);
            GP_TableContext* t = (new_idx >= 0 && new_idx < static_cast<int>(c->module_tables.size())) ? c->module_tables[new_idx] : nullptr;
            if (t && !lbl.empty()) {
                // load from template library (uses gp_table_load_template -> gp_table_deserialize)
                if (gp_table_load_template(t, nullptr, lbl.c_str())) {
                    printf("gp_canvas_load_from_file: restored template '%s' for module=%d\n", lbl.c_str(), static_cast<int>(i));
                }
            }
        } catch (...) { }
        if (snap.is_stage) {
            canvas_configure_stage_module(c, new_idx, snap.desc.w, snap.desc.h);
        } else {
            if (new_idx < static_cast<int>(c->module_bg.size())) {
                c->module_bg[new_idx].mode = snap.bg_mode;
            }
        // Attempt to restore per-module serialized table (.gpmod) from module_library
        try {
            std::string mid = gp_module_library_module_id(static_cast<int>(i));
            std::string rel = gp_module_library_module_serialized_path(std::string(), mid);
            std::string root = gp_module_library_default_root();
            std::filesystem::path full = std::filesystem::path(root) / rel;
            if (std::filesystem::exists(full) && std::filesystem::file_size(full) > 0) {
                std::ifstream ifs(full, std::ios::binary);
                if (ifs.good()) {
                    std::vector<char> buf((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                    ifs.close();
                    if (!buf.empty()) {
                        GP_TableContext* t = (new_idx >= 0 && new_idx < static_cast<int>(c->module_tables.size())) ? c->module_tables[new_idx] : nullptr;
                        if (t) {
                            printf("gp_canvas_load_from_file: restoring table for module=%d from %s size=%zu\n", static_cast<int>(i), full.string().c_str(), buf.size());
                            gp_table_deserialize(t, buf.data(), static_cast<int32_t>(buf.size()));
                        }
                    }
                }
            }
        } catch (...) {
            // ignore errors
        }
        // embedded module blobs are no longer used; explicit META_GROUP entries
        // will be restored below after modules/tables are created.
            if (new_idx < static_cast<int>(c->module_io_in_count.size())) c->module_io_in_count[new_idx] = std::max(0, snap.in_count);
            if (new_idx < static_cast<int>(c->module_io_out_count.size())) c->module_io_out_count[new_idx] = std::max(0, snap.out_count);
            // restore module UUID if present
            if (snap.module_uuid != 0ull) {
                gp_canvas_set_module_uuid(reinterpret_cast<GP_CanvasContext*>(c), new_idx, snap.module_uuid);
            }
            auto it = row_map.find(static_cast<int>(i));
            if (it != row_map.end()) {
                canvas_apply_io_rows(c, new_idx, it->second);
            } else {
                std::vector<ModuleIORow> default_rows;
                if (snap.in_count > 0) default_rows.push_back({ModuleRowKind::Input, 0, ModuleToolKind::None, std::clamp(snap.in_count, 1, 32)});
                if (snap.out_count > 0) default_rows.push_back({ModuleRowKind::Output, 0, ModuleToolKind::None, std::clamp(snap.out_count, 1, 32)});
                if (!default_rows.empty()) {
                    canvas_apply_io_rows(c, new_idx, default_rows);
                } else {
                    ensure_module_row_order(c, new_idx);
                }
            }
        }
    }

    // Load the root/container table blob so canvas-level ropes/overlays regain
    // their rope_id bindings before overlays and meta-groups are recreated.
    {
        std::string root = gp_module_library_default_root();
        std::string rel = gp_module_library_module_serialized_path(std::string(), "container");
        std::filesystem::path full = std::filesystem::path(root) / rel;
        if (std::filesystem::exists(full) && std::filesystem::file_size(full) > 0) {
            try {
                std::ifstream rifs(full, std::ios::binary);
                if (rifs.good()) {
                    std::vector<char> buf((std::istreambuf_iterator<char>(rifs)), std::istreambuf_iterator<char>());
                    rifs.close();
                    if (!buf.empty()) {
                        GP_TableContext* rt = canvas_ensure_root_table(c);
                        if (rt) {
                            printf("gp_canvas_load_from_file: restoring root table from %s size=%zu\n", full.string().c_str(), buf.size());
                            gp_table_deserialize(rt, buf.data(), static_cast<int32_t>(buf.size()));
                        }
                    }
                }
            } catch (...) {
                // keep load resilient; missing root blob is acceptable
            }
        }
        // TODO(step3): verify root/module rope parity on load and dedupe shared rope_ids back into the canvas map.
    }

    // NOTE: canvas-level META_GROUP snapshots are restored after edges are
    // recreated (moved below) so rope indices referenced by meta-group
    // vertices refer to valid ropes. Snapshots deferred to overlays are
    // collected in `c->pending_meta_snapshots` and applied when overlays are
    // created.

    for (const auto &fs : frame_leds) {
        if (fs.module_idx < 0 || fs.module_idx >= static_cast<int>(c->modules.size())) continue;
        if (fs.row < 0 || fs.row >= kModuleExtraLedRows) continue;
        if (fs.idx < 0 || fs.idx >= kModuleExtraLedCount) continue;
        GP_TableCell* cell = module_frame_led_cell(c, fs.module_idx, fs.row, fs.idx);
        if (!cell) continue;
        cell->flags = fs.flags;
        cell->reserved0 = fs.reserved0;
    }
    // restore module-frame port UUIDs
    for (const auto &fps : frame_port_uuids) {
        if (fps.module_idx < 0 || fps.module_idx >= static_cast<int>(c->module_frame_links.size())) continue;
        if (fps.row < 0 || fps.row >= kModuleExtraLedRows) continue;
        if (fps.idx < 0 || fps.idx >= kModuleExtraLedCount) continue;
        c->module_frame_links[static_cast<size_t>(fps.module_idx)].port_uuids[static_cast<size_t>(fps.row)][static_cast<size_t>(fps.idx)] = fps.port_uuid;
        if (fps.port_uuid != 0ull) c->module_port_uuid_map[fps.port_uuid] = {fps.module_idx, fps.row, fps.idx};
    }

    c->nodes.clear();
    c->module_node_id.assign(c->modules.size(), -1);
    int max_node_id = 0;
    for (auto &nc : nodes) {
        if (nc.module_idx >= 0 && nc.module_idx < static_cast<int>(c->modules.size())) {
            c->module_node_id[nc.module_idx] = nc.node_id;
            max_node_id = std::max(max_node_id, nc.node_id);
            c->nodes.push_back(std::move(nc));
        }
    }
    for (size_t i = 0; i < c->modules.size(); ++i) {
        if (c->module_node_id[i] >= 0) continue;
        int nid = ++max_node_id;
        c->module_node_id[i] = nid;
        GP_CanvasContextImpl::NodeContract nc{};
        nc.node_id = nid;
        nc.module_idx = static_cast<int>(i);
        c->nodes.push_back(std::move(nc));
    }
    int max_window_node = 0;
    for (const auto &entry : c->window_node_ids) {
        max_window_node = std::max(max_window_node, entry.second);
    }
    c->next_node_id = std::max(max_node_id + 1, max_window_node + 1);

    for (const auto &es : edges) {
        int edge_idx = gp_canvas_add_edge_with_type(ctx_, &es.desc, es.type_id);
        if (edge_idx >= 0 && es.subgroup_flags != 0u) {
            canvas_apply_subgroup_to_edge_idx(c, edge_idx, es.subgroup_flags);
        }
    }

    // Register per-table rope UIDs with the canvas so overlay attachments
    // can resolve persisted stable rope IDs to runtime indices. Doing this
    // here (after tables/edges are restored but before META_GROUP snapshots)
    // avoids a race where attachment code runs before tables have registered
    // their saved `rope_ids`.
    for (size_t ti = 0; ti < c->module_tables.size(); ++ti) {
        GP_TableContext* tt = c->module_tables[ti];
        if (!tt) continue;
        int cnt = gp_table_get_rope_id_count(tt);
        if (cnt <= 0) continue;
        std::vector<uint64_t> tmp; tmp.resize(static_cast<size_t>(cnt));
        int got = gp_table_get_rope_ids(tt, tmp.data(), cnt);
        if (got > 0) {
            gp_canvas_register_table_rope_ids_from_array(reinterpret_cast<GP_CanvasContext*>(c), tt, tmp.data(), got);
        }
    }
    // Also attempt for root table
    GP_TableContext* rt = canvas_ensure_root_table(c);
    if (rt) {
        int cnt = gp_table_get_rope_id_count(rt);
        if (cnt > 0) {
            std::vector<uint64_t> tmp; tmp.resize(static_cast<size_t>(cnt));
            int got = gp_table_get_rope_ids(rt, tmp.data(), cnt);
            if (got > 0) gp_canvas_register_table_rope_ids_from_array(reinterpret_cast<GP_CanvasContext*>(c), rt, tmp.data(), got);
        }
    }
    // Re-resolve rope ids to sim indices if any table attachment or rope id
    // updates marked the rope map dirty before meta-group restore.
    canvas_refresh_rope_map(c);
    printf("gp_canvas_load_from_file: restoring %zu META_GROUP snapshots, overlays.size=%zu\n", meta_group_snapshots.size(), c->overlays.size());
    for (const auto &ms : meta_group_snapshots) {
        GP_TableContext* t = nullptr;
        if (ms.module_idx >= 0 && ms.module_idx < static_cast<int>(c->module_tables.size())) {
            t = c->module_tables[static_cast<size_t>(ms.module_idx)];
        } else if (ms.module_idx < 0) {
            int overlay_idx = ms.meta_slot;
            auto it = c->overlays.find(overlay_idx);
            if (it != c->overlays.end()) {
                t = it->second.meta_table;
                printf("gp_canvas_load_from_file: snapshot overlay_idx=%d found overlay.meta_table=%p\n", overlay_idx, (void*)t);
            } else {
                // Try to auto-create an overlay entry so the saved META_GROUP can
                // bind immediately on load (avoids relying on later UI-created
                // overlays). Prefer the saved overlay slot, then fall back to
                // keys embedded in the snapshot.
                int desired_idx = overlay_idx;
                if (desired_idx < 0) {
                    if (ms.overlay_a != 0ull) desired_idx = static_cast<int>((ms.overlay_a >> 16) & 0xFFFFu);
                    else if (ms.overlay_b != 0ull) desired_idx = static_cast<int>((ms.overlay_b >> 16) & 0xFFFFu);
                }
                if (desired_idx >= 0) {
                    // create a rope and overlay using the common canvas APIs
                    float ox1 = ms.overlay_x1 != 0.0f ? ms.overlay_x1 : 10.0f;
                    float oy1 = ms.overlay_y1 != 0.0f ? ms.overlay_y1 : 10.0f;
                    float ox2 = ms.overlay_x2 != 0.0f ? ms.overlay_x2 : 110.0f;
                    float oy2 = ms.overlay_y2 != 0.0f ? ms.overlay_y2 : 60.0f;
                    // Attempt to locate backing table by mgid first
                    for (size_t ti = 0; ti < c->module_tables.size(); ++ti) {
                        GP_TableContext* tt = c->module_tables[ti];
                        if (!tt) continue;
                        int mgcount = gp_table_get_meta_group_count(tt);
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg2 = gp_table_get_meta_group(tt, mgi);
                            unsigned long long curid = 0ull; gp_table_meta_get_id(tt, mg2, &curid);
                            if (ms.mgid != 0ull && curid == ms.mgid) { t = tt; break; }
                        }
                        if (t) break;
                    }
                    if (!t) t = canvas_ensure_root_table(c);
                    if (t) {
                        RopeSim* sim = gp_table_get_rope_sim(t);
                        if (!sim) {
                            RopeSim* rootsim = canvas_root_sim(c);
                            if (!rootsim) rootsim = canvas_require_root_sim(c);
                            if (rootsim) gp_table_attach_rope_sim(t, rootsim, 0);
                            sim = gp_table_get_rope_sim(t);
                        }
                        int rope_idx = -1;
                        if (sim) {
                            float table_off_x = 0.0f, table_off_y = 0.0f;
                            int host_mod = -1;
                            for (int mi = 0; mi < static_cast<int>(c->module_tables.size()); ++mi) {
                                if (c->module_tables[mi] == t) { host_mod = mi; break; }
                            }
                            if (host_mod >= 0) {
                                const auto &m = c->modules[host_mod];
                                int top_h = std::min(m.h, kModuleTopUiHeight);
                                table_off_x = static_cast<float>(m.x);
                                table_off_y = static_cast<float>(m.y + top_h);
                            }
                            float sx_local = ox1 - table_off_x;
                            float sy_local = oy1 - table_off_y;
                            float fx_local = ox2 - table_off_x;
                            float fy_local = oy2 - table_off_y;
                            float plug_z = -10.0f;
                            int segs = 2;
                            rope_idx = rope_sim_add_rope3(sim, sx_local, sy_local, plug_z, fx_local, fy_local, plug_z, segs, 0.0f);
                        }
                        unsigned long long ka=0ull,kb=0ull;
                        // If saved overlay keys are present, prefer reusing/creating an
                        // overlay entry with those keys so we don't synthesize new
                        // keys for the same logical overlay and end up with
                        // duplicates.
                        if (ms.overlay_a != 0ull || ms.overlay_b != 0ull) {
                            bool found = false;
                            for (auto &pp : c->overlays) {
                                auto &ov = pp.second;
                                if (ov.key_a == ms.overlay_a || ov.key_b == ms.overlay_a || ov.key_a == ms.overlay_b || ov.key_b == ms.overlay_b) {
                                    ka = ov.key_a; kb = ov.key_b; found = true; break;
                                }
                            }
                            if (!found) {
                                // Create an overlay entry using the saved keys and
                                // a deterministic id (prefer the saved slot if
                                // available).
                                int new_oid = -1;
                                if (ms.meta_slot >= 0) {
                                    // don't clobber an existing entry
                                    if (c->overlays.find(ms.meta_slot) == c->overlays.end()) new_oid = ms.meta_slot;
                                }
                                if (new_oid < 0) new_oid = c->next_overlay_id++;
                                GP_CanvasContextImpl::OverlayEntry ov{};
                                ov.id = new_oid;
                                ov.x1 = ox1; ov.y1 = oy1; ov.x2 = ox2; ov.y2 = oy2;
                                ov.key_a = ms.overlay_a ? ms.overlay_a : ((static_cast<unsigned long long>(0xFFFFFFFFu) << 32) | (static_cast<unsigned long long>(static_cast<uint32_t>(ov.id)) << 16) | 0ull);
                                ov.key_b = ms.overlay_b ? ms.overlay_b : ((static_cast<unsigned long long>(0xFFFFFFFFu) << 32) | (static_cast<unsigned long long>(static_cast<uint32_t>(ov.id)) << 16) | 1ull);
                                c->overlays[ov.id] = ov;
                                c->overlay_key_map[ov.key_a] = ov.id;
                                c->overlay_key_map[ov.key_b] = ov.id;
                                ka = ov.key_a; kb = ov.key_b;
                                printf("gp_canvas_load_from_file: recreated overlay entry from saved keys for META_GROUP snapshot (keys=%llu,%llu) id=%d\n", ka, kb, ov.id);
                            }
                        }
                        if (ka == 0ull && kb == 0ull) {
                            if (gp_canvas_create_overlay_with_leds(reinterpret_cast<GP_CanvasContext*>(c), ox1, oy1, ox2, oy2, &ka, &kb)) {
                                if (rope_idx >= 0) gp_canvas_attach_rope_to_overlay(reinterpret_cast<GP_CanvasContext*>(c), ka, kb, rope_idx);
                                printf("gp_canvas_load_from_file: auto-created overlay via API for META_GROUP snapshot (keys=%llu,%llu)\n", ka, kb);
                            }
                        } else {
                            if (rope_idx >= 0) gp_canvas_attach_rope_to_overlay(reinterpret_cast<GP_CanvasContext*>(c), ka, kb, rope_idx);
                        }
                    } else {
                        printf("gp_canvas_load_from_file: unable to attach overlay to any table; deferring\n");
                    }
                } else {
                    // No desired index; create overlay via canvas API and attach to root
                    float ox1 = ms.overlay_x1 != 0.0f ? ms.overlay_x1 : 10.0f;
                    float oy1 = ms.overlay_y1 != 0.0f ? ms.overlay_y1 : 10.0f;
                    float ox2 = ms.overlay_x2 != 0.0f ? ms.overlay_x2 : 110.0f;
                    float oy2 = ms.overlay_y2 != 0.0f ? ms.overlay_y2 : 60.0f;
                    GP_TableContext* rt = canvas_ensure_root_table(c);
                    if (rt) {
                        RopeSim* sim = gp_table_get_rope_sim(rt);
                        if (!sim) {
                            RopeSim* rootsim = canvas_root_sim(c);
                            if (!rootsim) rootsim = canvas_require_root_sim(c);
                            if (rootsim) gp_table_attach_rope_sim(rt, rootsim, 0);
                            sim = gp_table_get_rope_sim(rt);
                        }
                        int rope_idx = -1;
                        if (sim) {
                            float table_off_x = 0.0f, table_off_y = 0.0f;
                            int host_mod = -1;
                            for (int mi = 0; mi < static_cast<int>(c->module_tables.size()); ++mi) {
                                if (c->module_tables[mi] == rt) { host_mod = mi; break; }
                            }
                            if (host_mod >= 0) {
                                const auto &m = c->modules[host_mod];
                                int top_h = std::min(m.h, kModuleTopUiHeight);
                                table_off_x = static_cast<float>(m.x);
                                table_off_y = static_cast<float>(m.y + top_h);
                            }
                            float sx_local = ox1 - table_off_x;
                            float sy_local = oy1 - table_off_y;
                            float fx_local = ox2 - table_off_x;
                            float fy_local = oy2 - table_off_y;
                            float plug_z = -10.0f;
                            int segs = 2;
                            rope_idx = rope_sim_add_rope3(sim, sx_local, sy_local, plug_z, fx_local, fy_local, plug_z, segs, 0.0f);
                        }
                        unsigned long long ka=0ull,kb=0ull;
                        if (ms.overlay_a != 0ull || ms.overlay_b != 0ull) {
                            bool found = false;
                            for (auto &pp : c->overlays) {
                                auto &ov = pp.second;
                                if (ov.key_a == ms.overlay_a || ov.key_b == ms.overlay_a || ov.key_a == ms.overlay_b || ov.key_b == ms.overlay_b) {
                                    ka = ov.key_a; kb = ov.key_b; found = true; break;
                                }
                            }
                            if (!found) {
                                int new_oid = -1;
                                if (ms.meta_slot >= 0) {
                                    if (c->overlays.find(ms.meta_slot) == c->overlays.end()) new_oid = ms.meta_slot;
                                }
                                if (new_oid < 0) new_oid = c->next_overlay_id++;
                                GP_CanvasContextImpl::OverlayEntry ov{};
                                ov.id = new_oid;
                                ov.x1 = ox1; ov.y1 = oy1; ov.x2 = ox2; ov.y2 = oy2;
                                ov.key_a = ms.overlay_a ? ms.overlay_a : ((static_cast<unsigned long long>(0xFFFFFFFFu) << 32) | (static_cast<unsigned long long>(static_cast<uint32_t>(ov.id)) << 16) | 0ull);
                                ov.key_b = ms.overlay_b ? ms.overlay_b : ((static_cast<unsigned long long>(0xFFFFFFFFu) << 32) | (static_cast<unsigned long long>(static_cast<uint32_t>(ov.id)) << 16) | 1ull);
                                c->overlays[ov.id] = ov;
                                c->overlay_key_map[ov.key_a] = ov.id;
                                c->overlay_key_map[ov.key_b] = ov.id;
                                ka = ov.key_a; kb = ov.key_b;
                                printf("gp_canvas_load_from_file: recreated overlay entry from saved keys for META_GROUP snapshot (keys=%llu,%llu) id=%d\n", ka, kb, ov.id);
                            }
                        }
                        if (ka == 0ull && kb == 0ull) {
                            if (gp_canvas_create_overlay_with_leds(reinterpret_cast<GP_CanvasContext*>(c), ox1, oy1, ox2, oy2, &ka, &kb)) {
                                if (rope_idx >= 0) gp_canvas_attach_rope_to_overlay(reinterpret_cast<GP_CanvasContext*>(c), ka, kb, rope_idx);
                                printf("gp_canvas_load_from_file: synthesized overlay created via API for META_GROUP snapshot (keys=%llu,%llu)\n", ka, kb);
                            }
                        } else {
                            if (rope_idx >= 0) gp_canvas_attach_rope_to_overlay(reinterpret_cast<GP_CanvasContext*>(c), ka, kb, rope_idx);
                        }
                        t = rt;
                    }
                }
            }
        }
        if (!t) continue;
        GP_MetaGroup* mg = gp_table_meta_create(t);
        if (!mg) continue;
        printf("gp_canvas_load_from_file: restoring canvas META_GROUP to table=%p mg=%p (module_idx=%d overlay_idx=%d)\n", (void*)t, (void*)mg, ms.module_idx, ms.meta_slot);
        fflush(stdout);
        gp_table_meta_set_confinement(t, mg, ms.confinement);
        gp_table_meta_set_id(t, mg, ms.mgid);
        gp_table_meta_set_lasso_fields(t, mg, ms.lasso_flags, ms.lasso_widget_type);
        // Add vertices: prefer saved rope ids, but resolve when
        // ropes have changed. Emulate lasso behavior: ensure a RopeSim is
        // attached, map overlay->edge rope indices, insert vertices when
        // needed, and fall back to root table ropes so the META_GROUP is
        // usable even if the original backing table layout changed.
        {
            RopeSim* sim = gp_table_get_rope_sim(t);
            if (!sim) {
                RopeSim* rootsim = canvas_root_sim(c);
                if (!rootsim) rootsim = canvas_require_root_sim(c);
                if (rootsim) gp_table_attach_rope_sim(t, rootsim, 0);
                sim = gp_table_get_rope_sim(t);
            }
            if (ms.anchor_rope_id != 0ull) {
                int anchor_idx = gp_table_resolve_rope_id_to_sim_index(t, ms.anchor_rope_id);
                if (anchor_idx >= 0) {
                    gp_table_meta_set_anchor(t, mg, anchor_idx, ms.anchor_v);
                } else {
                    printf("gp_canvas_load_from_file: unable to resolve anchor rope_id=%llu for mg=%p\n", (unsigned long long)ms.anchor_rope_id, (void*)mg);
                }
            }
            for (const auto &vp : ms.verts) {
                uint64_t rope_uid = vp.rope_id;
                int saved_vid = vp.vertex_idx;
                bool added = false;
                if (sim && rope_uid != 0ull) {
                    int resolved_rope = gp_table_resolve_rope_id_to_sim_index(t, rope_uid);
                    if (resolved_rope >= 0) {
                        int vc = rope_sim_get_vertex_count(sim, resolved_rope);
                        if (vc > 0 && saved_vid >= 0 && saved_vid < vc) {
                            gp_table_meta_add_vertex(t, mg, resolved_rope, saved_vid);
                            added = true;
                        }
                    }
                }
                if (!added && sim) {
                    // Try to map using overlay keys -> canvas edges -> rope_idx
                    int mapped_rope = -1;
                    if (ms.overlay_a != 0ull || ms.overlay_b != 0ull) {
                        for (size_t ei = 0; ei < c->edges.size(); ++ei) {
                            const auto &e = c->edges[ei];
                            if (e.overlay_key_a && (e.overlay_key_a == ms.overlay_a || e.overlay_key_a == ms.overlay_b)) { mapped_rope = e.rope_idx; break; }
                            if (e.overlay_key_b && (e.overlay_key_b == ms.overlay_a || e.overlay_key_b == ms.overlay_b)) { mapped_rope = e.rope_idx; break; }
                        }
                    }
                    // If mapping found, try to add/clamp/insert a vertex there
                    if (mapped_rope >= 0) {
                        int mvc = rope_sim_get_vertex_count(sim, mapped_rope);
                        if (mvc > 1) {
                            int seg = std::clamp(saved_vid, 0, mvc - 2);
                            float tparam = 0.5f;
                            int newv = gp_table_rope_insert_vertex(t, mapped_rope, seg, tparam);
                            if (newv >= 0) {
                                gp_table_meta_add_vertex(t, mg, mapped_rope, newv);
                            } else {
                                int use_vid = std::clamp(saved_vid, 0, mvc - 1);
                                gp_table_meta_add_vertex(t, mg, mapped_rope, use_vid);
                            }
                            added = true;
                        } else if (mvc == 1) {
                            // only one vertex, cannot insert — attach existing
                            gp_table_meta_add_vertex(t, mg, mapped_rope, 0);
                            added = true;
                        } else {
                            // no vertices present — attempt insertion at seg 0
                            int newv = gp_table_rope_insert_vertex(t, mapped_rope, 0, 0.5f);
                            if (newv >= 0) {
                                gp_table_meta_add_vertex(t, mg, mapped_rope, newv);
                                added = true;
                            }
                        }
                    }
                }
                if (!added && sim) {
                    // Fallback: find any rope in this table with vertices
                    int found_rope = -1;
                    for (int ri = 0; ri < 1024; ++ri) {
                        int vc = rope_sim_get_vertex_count(sim, ri);
                        if (vc > 0) { found_rope = ri; break; }
                    }
                    if (found_rope >= 0) {
                        int vc = rope_sim_get_vertex_count(sim, found_rope);
                        if (vc > 1) {
                            int seg = std::clamp(saved_vid, 0, vc - 2);
                            int newv = gp_table_rope_insert_vertex(t, found_rope, seg, 0.5f);
                            if (newv >= 0) gp_table_meta_add_vertex(t, mg, found_rope, newv);
                            else gp_table_meta_add_vertex(t, mg, found_rope, std::max(0, std::min(vc - 1, saved_vid)));
                        } else if (vc == 1) {
                            gp_table_meta_add_vertex(t, mg, found_rope, 0);
                        }
                        added = true;
                    }
                }
                if (!added) {
                    // Last resort: create a vertex-less placeholder (no-op)
                    printf("gp_canvas_load_from_file: unable to map meta vertex (rope=%llu,vid=%d) for mg=%p\n", (unsigned long long)rope_uid, saved_vid, (void*)mg);
                }
            }
        }
        gp_table_meta_set_channel_group(t, mg, ms.channel_group);
        gp_table_meta_set_overlay_keys(t, mg, ms.overlay_a, ms.overlay_b);
        gp_table_meta_set_dangling_hang_len(t, mg, ms.dangling_len);
        gp_table_meta_set_ring_mode(t, mg, ms.ring_mode);
        // Defer deterministic finalize until after full file load so ropes/edges exist
        c->post_load_meta_pending.push_back({ t, mg, ms.ring_u, ms.ring_mode, ms.overlay_a, ms.overlay_b });
        if ((ms.overlay_a != 0ull || ms.overlay_b != 0ull) && c) {
            // Ensure overlay rope exists and is attached before binding meta
            GP_CanvasContextImpl::OverlayEntry* pov = nullptr;
            for (auto &pp : c->overlays) {
                auto &ov = pp.second;
                if (ov.key_a == ms.overlay_a || ov.key_b == ms.overlay_a || ov.key_a == ms.overlay_b || ov.key_b == ms.overlay_b) { pov = &ov; break; }
            }
            if (!pov && ms.meta_slot >= 0) {
                auto itov = c->overlays.find(ms.meta_slot);
                if (itov != c->overlays.end()) pov = &itov->second;
            }
            if (pov) {
                RopeSim* sim = gp_table_get_rope_sim(t);
                if (!sim) {
                    RopeSim* rootsim = canvas_root_sim(c);
                    if (!rootsim) rootsim = canvas_require_root_sim(c);
                    if (rootsim) gp_table_attach_rope_sim(t, rootsim, 0);
                    sim = gp_table_get_rope_sim(t);
                }
                if (sim) {
                    float table_off_x = 0.0f, table_off_y = 0.0f;
                    int host_mod = -1;
                    for (int mi = 0; mi < static_cast<int>(c->module_tables.size()); ++mi) {
                        if (c->module_tables[mi] == t) { host_mod = mi; break; }
                    }
                    if (host_mod >= 0) {
                        const auto &m = c->modules[host_mod];
                        int top_h = std::min(m.h, kModuleTopUiHeight);
                        table_off_x = static_cast<float>(m.x);
                        table_off_y = static_cast<float>(m.y + top_h);
                    }
                    float sx_local = pov->x1 - table_off_x;
                    float sy_local = pov->y1 - table_off_y;
                    float fx_local = pov->x2 - table_off_x;
                    float fy_local = pov->y2 - table_off_y;
                    float plug_z = -10.0f;
                    int segs = 2;
                    int rope_idx = rope_sim_add_rope3(sim, sx_local, sy_local, plug_z, fx_local, fy_local, plug_z, segs, 0.0f);
                    if (rope_idx >= 0) {
                        gp_canvas_attach_rope_to_overlay(reinterpret_cast<GP_CanvasContext*>(c), pov->key_a, pov->key_b, rope_idx);
                    }
                }
            }
            gp_canvas_set_overlay_meta(reinterpret_cast<GP_CanvasContext*>(c), ms.overlay_a, ms.overlay_b, t, reinterpret_cast<void*>(mg));
        }
    }

    // Deterministic post-load pass: attempt to finalize any meta-groups that
    // were created during load but couldn't be fully registered earlier.
    for (auto &pend : c->post_load_meta_pending) {
        GP_TableContext* t = pend.table;
        GP_MetaGroup* mg = pend.mg;
        float saved_u = pend.ring_u;
        if (!t || !mg) { printf("canvas: post-load finalize skipped: missing table or meta-group\n"); continue; }

        // Skip if ring already registered for this meta-group
        bool ring_registered = false;
        int rc = gp_table_get_ring_edge_count(t);
        for (int rei = 0; rei < rc; ++rei) {
            int ring_id = -1; unsigned long long ring_key = 0ull;
            if (!gp_table_get_ring_edge(t, rei, &ring_id, &ring_key)) continue;
            if (ring_key == 0ull) continue;
            unsigned long long myid = 0ull; gp_table_meta_get_id(t, mg, &myid);
            if (ring_key == myid) { ring_registered = true; break; }
        }
        if (ring_registered) continue;

        // Ensure RopeSim exists
        RopeSim* sim = gp_table_get_rope_sim(t);
        if (!sim) {
            RopeSim* rootsim = canvas_root_sim(c);
            if (!rootsim) rootsim = canvas_require_root_sim(c);
            if (rootsim) gp_table_attach_rope_sim(t, rootsim, 0);
            sim = gp_table_get_rope_sim(t);
        }
        if (!sim) { printf("canvas: post-load finalize skipped: no RopeSim for table\n"); continue; }

        // Find backing rope + vertex: prefer anchor then first vertex (rope index + vid)
        int first_rope = -1; int first_vid = -1;
        // gp_table_meta_get_anchor always returns success; inspect anchor values
        gp_table_meta_get_anchor(t, mg, &first_rope, &first_vid);
        if (first_rope < 0 || first_vid < 0) {
            int vcount = gp_table_meta_get_vertex_count(t, mg);
            printf("canvas: post-load finalize: mg=%p vcount=%d (table=%p)\n", (void*)mg, vcount, (void*)t);
            for (int vi = 0; vi < vcount; ++vi) {
                int rr = -1, rv = -1; gp_table_meta_get_vertex(t, mg, vi, &rr, &rv);
                int rc = rope_sim_get_vertex_count(sim, rr);
                printf("  vertex[%d] = rope=%d vert=%d rope_vcount=%d\n", vi, rr, rv, rc);
            }
            if (vcount > 0) gp_table_meta_get_vertex(t, mg, 0, &first_rope, &first_vid);
        }
        if (first_rope < 0 || first_vid < 0) {
            printf("canvas: post-load finalize skipped: no backing rope/vertex for meta-group mg=%p table=%p\n", (void*)mg, (void*)t);
            continue;
        }

        canvas_finalize_lasso_meta_group(c, t, mg, first_rope, first_vid, saved_u);
        // Ensure RopeSim meta-group contains all table-side vertices. Some
        // vertices may have been added before a RopeSim was attached; replay
        // any missing members into the sim meta-group now that a root sim
        // should be present. `sim` was retrieved above.
        if (sim) {
            int sim_group_idx = -1;
            gp_table_meta_get_sim_group_index(t, mg, &sim_group_idx);
            // If sim_group_idx not present on the meta-group, try resolving
            // by the persistent meta-group id via canvas mapping.
            if (sim_group_idx < 0) {
                unsigned long long mgid = 0ull;
                gp_table_meta_get_id(t, mg, &mgid);
                if (mgid != 0ull) {
                    auto it = c->lasso_id_map.find(mgid);
                    if (it != c->lasso_id_map.end()) sim_group_idx = it->second;
                }
            }
            if (sim_group_idx >= 0) {
                int vcount = gp_table_meta_get_vertex_count(t, mg);
                for (int vi = 0; vi < vcount; ++vi) {
                    int rr = -1, rv = -1;
                    if (!gp_table_meta_get_vertex(t, mg, vi, &rr, &rv)) continue;
                    if (rr < 0 || rv < 0) continue;
                    if (!rope_sim_meta_group_has_member(sim, sim_group_idx, rr, rv)) {
                        unsigned long long mgid_dbg = 0ull; gp_table_meta_get_id(t, mg, &mgid_dbg);
                        printf("canvas: replaying sim add for mg=%p id=%llu sim=%p group=%d member=(%d,%d)\n", (void*)mg, (unsigned long long)mgid_dbg, (void*)sim, sim_group_idx, rr, rv);
                        rope_sim_meta_group_add(sim, sim_group_idx, rr, rv);
                    }
                }
            }
        }
    }
    c->post_load_meta_pending.clear();

    update_canvas_scroll_state(c, /*pull_from_container=*/false);
    return 1;
}

static std::string canvas_module_label(const GP_CanvasModuleDesc& desc) {
    std::string label(desc.label, desc.label + sizeof(desc.label));
    size_t null_pos = label.find('\0');
    if (null_pos != std::string::npos) label.resize(null_pos);
    return label;
}

extern "C" int gp_canvas_export_module_library(GP_CanvasContext* ctx_, const char* path) {
    if (!ctx_ || !path) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);

    GP_ModuleLibrary library{};
    library.root_dir = gp_module_library_default_root();

    std::map<int, GP_ModuleLibraryTool> tool_registry_by_kind;

    for (size_t i = 0; i < c->modules.size(); ++i) {
        const auto &mod = c->modules[i];
        GP_ModuleLibraryModule module{};
        module.module_idx = static_cast<int>(i);
        module.id = gp_module_library_module_id(module.module_idx);
        module.label = canvas_module_label(mod);
        // Use relative paths (no root) so actualizer will join with the chosen root
        module.serialized_path = gp_module_library_module_serialized_path(std::string(), module.id);
        module.source_path = gp_module_library_module_source_path(std::string(), module.id);

        if (i < c->module_io_rows.size()) {
            const auto &rows = c->module_io_rows[i];
            for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
                const auto &row = rows[row_idx];
                if (row.kind != ModuleRowKind::Tool) continue;
                ModuleToolKind tool_kind = row.tool;
                if (tool_kind == ModuleToolKind::None) continue;
                auto tool_it = tool_registry_by_kind.find(static_cast<int>(tool_kind));
                if (tool_it == tool_registry_by_kind.end()) {
                    GP_ModuleLibraryTool tool{};
                    tool.kind = tool_kind;
                    tool.id = gp_module_library_tool_id(tool_kind);
                    tool.name = gp_module_tool_kind_name(tool_kind);
                    // Use relative tool source path
                    tool.source_path = gp_module_library_tool_source_path(std::string(), tool.id);
                    tool_registry_by_kind.emplace(static_cast<int>(tool_kind), std::move(tool));
                }
                GP_ModuleToolInstance instance{};
                instance.row_idx = static_cast<int>(row_idx);
                instance.attachment_count = row.attachment_count;
                instance.tool_id = gp_module_library_tool_id(tool_kind);
                module.tool_instances.push_back(std::move(instance));
            }
        }
        library.modules.push_back(std::move(module));
    }

    for (auto &entry : tool_registry_by_kind) {
        library.tool_registry.push_back(std::move(entry.second));
    }

    return gp_module_library_write_to_file(library, path);
}

extern "C" int gp_canvas_actualize_to_root(GP_CanvasContext* ctx_, const char* output_root) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);

    GP_ModuleLibrary library{};
    library.root_dir = gp_module_library_default_root();
    std::map<int, GP_ModuleLibraryTool> tool_registry_by_kind;

    for (size_t i = 0; i < c->modules.size(); ++i) {
        const auto &mod = c->modules[i];
        GP_ModuleLibraryModule module{};
        module.module_idx = static_cast<int>(i);
        module.id = gp_module_library_module_id(module.module_idx);
        module.label = canvas_module_label(mod);
        // Use relative paths (no root) so actualizer will join with the chosen root
        module.serialized_path = gp_module_library_module_serialized_path(std::string(), module.id);
        module.source_path = gp_module_library_module_source_path(std::string(), module.id);

        if (i < c->module_io_rows.size()) {
            const auto &rows = c->module_io_rows[i];
            for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
                const auto &row = rows[row_idx];
                if (row.kind != ModuleRowKind::Tool) continue;
                ModuleToolKind tool_kind = row.tool;
                if (tool_kind == ModuleToolKind::None) continue;
                auto tool_it = tool_registry_by_kind.find(static_cast<int>(tool_kind));
                if (tool_it == tool_registry_by_kind.end()) {
                    GP_ModuleLibraryTool tool{};
                    tool.kind = tool_kind;
                    tool.id = gp_module_library_tool_id(tool_kind);
                    tool.name = gp_module_tool_kind_name(tool_kind);
                    // Use relative tool source path
                    tool.source_path = gp_module_library_tool_source_path(std::string(), tool.id);
                    tool_registry_by_kind.emplace(static_cast<int>(tool_kind), std::move(tool));
                }
                GP_ModuleToolInstance instance{};
                instance.row_idx = static_cast<int>(row_idx);
                instance.attachment_count = row.attachment_count;
                instance.tool_id = gp_module_library_tool_id(tool_kind);
                module.tool_instances.push_back(std::move(instance));
            }
        }
        library.modules.push_back(std::move(module));
    }

    for (auto &entry : tool_registry_by_kind) {
        library.tool_registry.push_back(std::move(entry.second));
    }

    const char* root = output_root ? output_root : nullptr;
    // If output_root is null, let actualizer use library.root_dir (which defaults to "module_library")
    std::string root_str = root ? std::string(root) : std::string();
    return gp_module_library_actualize_sources(library, root_str.empty() ? nullptr : root_str.c_str());
}

extern "C" int gp_canvas_export_module_to_root(GP_CanvasContext* ctx_, int module_idx, const char* output_root) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->modules.size())) return 0;

    std::string root = output_root ? std::string(output_root) : gp_module_library_default_root();
    std::string module_id = gp_module_library_module_id(module_idx);
    std::string rel = gp_module_library_module_serialized_path(std::string(), module_id);
    std::filesystem::path full = std::filesystem::path(root) / rel;

    try {
        std::filesystem::create_directories(full.parent_path());
    } catch (...) {
        // ignore
    }

    GP_TableContext* t = (module_idx >= 0 && module_idx < static_cast<int>(c->module_tables.size())) ? c->module_tables[module_idx] : nullptr;
    if (!t) {
        // create an empty placeholder file
        std::ofstream ofs(full, std::ios::binary | std::ios::trunc);
        if (!ofs.good()) return 0;
        ofs.close();
        return 1;
    }

    // Ensure this table has registered its persistent rope_ids with the canvas
    // so any saved META_GROUPs referencing rope ids can be resolved.
    if (t) {
        int cnt = gp_table_get_rope_id_count(t);
        if (cnt > 0) {
            std::vector<uint64_t> tmp; tmp.resize(static_cast<size_t>(cnt));
            int got = gp_table_get_rope_ids(t, tmp.data(), cnt);
            if (got > 0) gp_canvas_register_table_rope_ids_from_array(reinterpret_cast<GP_CanvasContext*>(c), t, tmp.data(), got);
        }
    }

    // Diagnostic: log canvas-level rope id registry so we can compare
    // what the canvas knows vs what the table serializer will see.
    printf("gp_canvas_export_module_to_root: canvas rope_id_map entries:\n");
    for (const auto &kv : c->rope_id_map) {
        uint64_t rid = kv.second.rope_id;
        GP_TableContext* rt = kv.second.table;
        int rsi = canvas_resolve_rope_id_to_sim_index(rt, rid);
        int midx = -2; // -2 = unknown, -1 = container/root
        if (rt) {
            if (rt == c->container_table) midx = -1;
            else {
                for (size_t mi = 0; mi < c->module_tables.size(); ++mi) {
                    if (c->module_tables[mi] == rt) { midx = static_cast<int>(mi); break; }
                }
            }
        }
        printf("  rope_id=%llu table=%p module_idx=%d sim_idx=%d\n", (unsigned long long)rid, (void*)rt, midx, rsi);
    }
    int32_t need = gp_table_serialize(t, nullptr, 0);
    if (need <= 0) {
        // create empty file
        std::ofstream ofs(full, std::ios::binary | std::ios::trunc);
        if (!ofs.good()) return 0;
        ofs.close();
        return 1;
    }
    std::vector<char> buf(static_cast<size_t>(need));
    int32_t written = gp_table_serialize(t, buf.data(), need);
    if (written != need) return 0;
    std::ofstream ofs(full, std::ios::binary | std::ios::trunc);
    if (!ofs.good()) return 0;
    ofs.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    ofs.close();
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

extern "C" int gp_canvas_set_subgroup_toolbar_rgba(GP_CanvasContext* ctx_, const float* rgba, int value_count) {
    if (!ctx_) return 0;
    if (!rgba || value_count <= 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    int color_count = std::min(kSubgroupBinCount, value_count / 4);
    for (int i = 0; i < color_count; ++i) {
        const Color col = rgba_from_floats(rgba + i * 4);
        c->subgroup_target_rgba[static_cast<size_t>(i)].store(pack_rgba(col), std::memory_order_release);
    }
    return 1;
}

extern "C" int gp_canvas_set_subgroup_toolbar_rgba_at(GP_CanvasContext* ctx_, int idx, const float* rgba) {
    if (!ctx_ || !rgba) return 0;
    if (idx < 0 || idx >= kSubgroupBinCount) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    const Color col = rgba_from_floats(rgba);
    c->subgroup_target_rgba[static_cast<size_t>(idx)].store(pack_rgba(col), std::memory_order_release);
    return 1;
}

extern "C" GP_CanvasContext* gp_canvas_get_singleton() {
    return reinterpret_cast<GP_CanvasContext*>(g_canvas_context_singleton);
}

extern "C" int gp_canvas_create_overlay_with_leds(GP_CanvasContext* ctx_, float x1, float y1, float x2, float y2, unsigned long long* out_key_a, unsigned long long* out_key_b) {
    if (!ctx_ || !out_key_a || !out_key_b) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // Always create a new overlay entry for explicit API calls.
    // Do NOT use spatial heuristics for reuse; callers must rely on
    // saved overlay keys/ids for canonical restoration.
    int oid = c->next_overlay_id++;
    GP_CanvasContextImpl::OverlayEntry ov{};
    ov.id = oid;
    ov.x1 = x1; ov.y1 = y1; ov.x2 = x2; ov.y2 = y2;
    // encode keys: sentinel module id 0xFFFFFFFF, contact = overlay id, led index in low 16
    unsigned long long base = (static_cast<unsigned long long>(0xFFFFFFFFu) << 32) | (static_cast<unsigned long long>(static_cast<uint32_t>(oid)) << 16);
    ov.key_a = base | static_cast<unsigned long long>(0);
    ov.key_b = base | static_cast<unsigned long long>(1);
        // assign stable per-port UUIDs
        ov.port_uuid_a = gp_canvas_generate_id(ctx_, 0ull);
        ov.port_uuid_b = gp_canvas_generate_id(ctx_, 0ull);
    c->overlays[oid] = ov;
    *out_key_a = ov.key_a;
    *out_key_b = ov.key_b;
    c->overlay_key_map[ov.key_a] = ov.id;
    c->overlay_key_map[ov.key_b] = ov.id;
        // register port UUIDs
        if (ov.port_uuid_a) c->overlay_port_uuid_map[ov.port_uuid_a] = std::make_pair(ov.id, 0);
        if (ov.port_uuid_b) c->overlay_port_uuid_map[ov.port_uuid_b] = std::make_pair(ov.id, 1);
    // If any pending canvas meta-group snapshots refer to this overlay id,
    // apply them now so meta-groups restored from file become visible.
    for (auto it = c->pending_meta_snapshots.begin(); it != c->pending_meta_snapshots.end(); ) {
        const CanvasMetaSnapshot &ms = *it;
        bool matches = false;
        if (ms.module_idx < 0 && ms.meta_slot == oid) matches = true;
        // also match by overlay keys if present
        if (!matches && (ms.overlay_a != 0ull || ms.overlay_b != 0ull)) {
            if (ms.overlay_a == ov.key_a || ms.overlay_a == ov.key_b || ms.overlay_b == ov.key_a || ms.overlay_b == ov.key_b) matches = true;
        }
        if (matches) {
            // attempt to resolve backing table: prefer meta_table cached in overlay (none yet)
            GP_TableContext* t = nullptr;
            if (ms.module_idx >= 0 && ms.module_idx < static_cast<int>(c->module_tables.size())) t = c->module_tables[ms.module_idx];
            if (!t) {
                // try to find table by scanning module_tables for a meta-group with matching id
                for (size_t ti = 0; ti < c->module_tables.size(); ++ti) {
                    GP_TableContext* tt = c->module_tables[ti];
                    if (!tt) continue;
                    int mgcount = gp_table_get_meta_group_count(tt);
                    for (int mgi = 0; mgi < mgcount; ++mgi) {
                        GP_MetaGroup* mg2 = gp_table_get_meta_group(tt, mgi);
                        unsigned long long curid = 0ull; gp_table_meta_get_id(tt, mg2, &curid);
                        if (ms.mgid != 0ull && curid == ms.mgid) { t = tt; break; }
                    }
                    if (t) break;
                }
            }
            if (!t) {
                printf("pending_meta_snapshot: no backing table found for snapshot module_idx=%d meta_slot=%d mgid=%llu overlay_a=%llu overlay_b=%llu\n",
                    ms.module_idx, ms.meta_slot, (unsigned long long)ms.mgid, (unsigned long long)ms.overlay_a, (unsigned long long)ms.overlay_b);
                fflush(stdout);
            }
            if (t) {
                // Check if a meta-group with the same persistent id already
                // exists in the table to avoid creating duplicates during
                // deserialize paths that may run more than once for the same
                // saved snapshot (canvas-level loader + pending snapshot apply).
                GP_MetaGroup* mg = nullptr;
                if (ms.mgid != 0ull) {
                    int mgcount = gp_table_get_meta_group_count(t);
                    for (int mgi = 0; mgi < mgcount; ++mgi) {
                        GP_MetaGroup* cand = gp_table_get_meta_group(t, mgi);
                        if (!cand) continue;
                        unsigned long long curid = 0ull; gp_table_meta_get_id(t, cand, &curid);
                        if (curid == ms.mgid) { mg = cand; break; }
                    }
                }
                if (!mg) {
                    mg = gp_table_meta_create(t);
                    if (!mg) {
                        printf("pending_meta_snapshot: gp_table_meta_create failed for table=%p mgid=%llu\n", (void*)t, (unsigned long long)ms.mgid);
                        fflush(stdout);
                    }
                    if (mg) {
                        gp_table_meta_set_confinement(t, mg, ms.confinement);
                        gp_table_meta_set_id(t, mg, ms.mgid);
                        gp_table_meta_set_lasso_fields(t, mg, ms.lasso_flags, ms.lasso_widget_type);
                        RopeSim* sim = gp_table_get_rope_sim(t);
                        if (!sim) {
                            RopeSim* rootsim = canvas_root_sim(c);
                            if (!rootsim) rootsim = canvas_require_root_sim(c);
                            if (rootsim) gp_table_attach_rope_sim(t, rootsim, 0);
                            sim = gp_table_get_rope_sim(t);
                        }
                        if (!sim) {
                            printf("pending_meta_snapshot: no RopeSim available for table=%p while restoring mgid=%llu\n", (void*)t, (unsigned long long)ms.mgid);
                            fflush(stdout);
                        } else {
                            if (ms.anchor_rope_id != 0ull) {
                                int anchor_idx = gp_table_resolve_rope_id_to_sim_index(t, ms.anchor_rope_id);
                                if (anchor_idx >= 0) gp_table_meta_set_anchor(t, mg, anchor_idx, ms.anchor_v);
                            }
                            for (const auto &vp : ms.verts) {
                                if (vp.rope_id == 0ull) continue;
                                int resolved = gp_table_resolve_rope_id_to_sim_index(t, vp.rope_id);
                                if (resolved >= 0) gp_table_meta_add_vertex(t, mg, resolved, vp.vertex_idx);
                            }
                        }
                        gp_table_meta_set_channel_group(t, mg, ms.channel_group);
                        gp_table_meta_set_overlay_keys(t, mg, ms.overlay_a, ms.overlay_b);
                        gp_table_meta_set_dangling_hang_len(t, mg, ms.dangling_len);
                        gp_table_meta_set_ring_mode(t, mg, ms.ring_mode);
                    }
                } else {
                    // existing meta-group found — ensure its overlay keys are up-to-date
                    gp_table_meta_set_overlay_keys(t, mg, ms.overlay_a, ms.overlay_b);
                }
                if (mg) {
                    // Ensure an overlay rope exists and is attached before binding
                    if ((ms.overlay_a != 0ull || ms.overlay_b != 0ull)) {
                        // find overlay entry by keys
                        int oid = -1; GP_CanvasContextImpl::OverlayEntry *pov = nullptr;
                        for (auto &pp : c->overlays) {
                            auto &ov = pp.second;
                            if (ov.key_a == ms.overlay_a || ov.key_b == ms.overlay_a || ov.key_a == ms.overlay_b || ov.key_b == ms.overlay_b) { oid = ov.id; pov = &ov; break; }
                        }
                        if (!pov) {
                            // if overlay slot provided, try that
                            if (ms.meta_slot >= 0) {
                                auto itov = c->overlays.find(ms.meta_slot);
                                if (itov != c->overlays.end()) { oid = itov->first; pov = &itov->second; }
                            }
                            if (!pov) {
                                printf("pending_meta_snapshot: no overlay entry found for keys %llu,%llu and meta_slot=%d (mgid=%llu)\n",
                                    (unsigned long long)ms.overlay_a, (unsigned long long)ms.overlay_b, ms.meta_slot, (unsigned long long)ms.mgid);
                                fflush(stdout);
                            }
                        }
                        if (pov) {
                            // create rope in table's sim using the same placement logic as lasso
                            RopeSim* sim = gp_table_get_rope_sim(t);
                            if (!sim) {
                                RopeSim* rootsim = canvas_root_sim(c);
                                if (!rootsim) rootsim = canvas_require_root_sim(c);
                                if (rootsim) gp_table_attach_rope_sim(t, rootsim, 0);
                                sim = gp_table_get_rope_sim(t);
                            }
                            if (!sim) {
                                printf("pending_meta_snapshot: no RopeSim available for table=%p overlay_id=%d mgid=%llu\n", (void*)t, pov->id, (unsigned long long)ms.mgid);
                                fflush(stdout);
                            }
                            if (sim) {
                                // compute table-local offset for host module if present
                                float table_off_x = 0.0f, table_off_y = 0.0f;
                                int host_mod = -1;
                                for (int mi = 0; mi < static_cast<int>(c->module_tables.size()); ++mi) {
                                    if (c->module_tables[mi] == t) { host_mod = mi; break; }
                                }
                                if (host_mod >= 0) {
                                    const auto &m = c->modules[host_mod];
                                    int top_h = std::min(m.h, kModuleTopUiHeight);
                                    table_off_x = static_cast<float>(m.x);
                                    table_off_y = static_cast<float>(m.y + top_h);
                                }
                                float sx_local = pov->x1 - table_off_x;
                                float sy_local = pov->y1 - table_off_y;
                                float fx_local = pov->x2 - table_off_x;
                                float fy_local = pov->y2 - table_off_y;
                                float plug_z = -10.0f;
                                int segs = 2;
                                int rope_idx = rope_sim_add_rope3(sim, sx_local, sy_local, plug_z, fx_local, fy_local, plug_z, segs, 0.0f);
                                if (rope_idx >= 0) {
                                    // attach root/canonical edge and map overlay keys
                                    gp_canvas_attach_rope_to_overlay(reinterpret_cast<GP_CanvasContext*>(c), pov->key_a, pov->key_b, rope_idx);
                                }
                            }
                        }
                    }
                    // bind overlay
                    gp_canvas_set_overlay_meta(reinterpret_cast<GP_CanvasContext*>(c), ms.overlay_a, ms.overlay_b, t, reinterpret_cast<void*>(mg));
                    // After binding, finalize meta-group (widget, springs, ring)
                    if (ms.ring_mode != 0) {
                        int first_rope = -1;
                        int first_vid = -1;
                        if (ms.anchor_rope_id != 0ull) {
                            int resolved = gp_table_resolve_rope_id_to_sim_index(t, ms.anchor_rope_id);
                            if (resolved >= 0) { first_rope = resolved; first_vid = ms.anchor_v; }
                        }
                        if (first_rope < 0 && !ms.verts.empty()) {
                            for (const auto &vp : ms.verts) {
                                if (vp.rope_id == 0ull) continue;
                                int resolved = gp_table_resolve_rope_id_to_sim_index(t, vp.rope_id);
                                if (resolved >= 0) {
                                    first_rope = resolved;
                                    first_vid = vp.vertex_idx;
                                    break;
                                }
                            }
                        }
                        if (first_rope < 0 && (ms.overlay_a != 0ull || ms.overlay_b != 0ull)) {
                            for (size_t ei = 0; ei < c->edges.size(); ++ei) {
                                const auto &e = c->edges[ei];
                                if ((e.overlay_key_a && (e.overlay_key_a == ms.overlay_a || e.overlay_key_a == ms.overlay_b)) ||
                                    (e.overlay_key_b && (e.overlay_key_b == ms.overlay_a || e.overlay_key_b == ms.overlay_b))) {
                                    if (e.rope_idx >= 0) { first_rope = e.rope_idx; break; }
                                }
                            }
                        }
                        if (first_rope >= 0 && first_vid >= 0) {
                            canvas_finalize_lasso_meta_group(c, t, mg, first_rope, first_vid, ms.ring_u);
                        } else {
                            // couldn't find a rope/vertex to finalize ring registration
                            printf("pending_meta_snapshot: cannot finalize ring for mg=%p ring_mode=%d first_rope=%d first_vid=%d overlay_a=%llu overlay_b=%llu mgid=%llu\n",
                                (void*)mg, ms.ring_mode, first_rope, first_vid, (unsigned long long)ms.overlay_a, (unsigned long long)ms.overlay_b, (unsigned long long)ms.mgid);
                            fflush(stdout);
                        }
                        // Deliver saved lasso points (reconstructed from saved vertex refs)
                        if (c->lasso_cb && !ms.verts.empty()) {
                            // compute table-local offset for host module if present
                            float table_off_x = 0.0f;
                            float table_off_y = 0.0f;
                            int host_mod = -1;
                            for (int mi = 0; mi < static_cast<int>(c->module_tables.size()); ++mi) {
                                if (c->module_tables[mi] == t) { host_mod = mi; break; }
                            }
                            if (host_mod >= 0) {
                                const auto &m = c->modules[host_mod];
                                int top_h = std::min(m.h, kModuleTopUiHeight);
                                table_off_x = static_cast<float>(m.x);
                                table_off_y = static_cast<float>(m.y + top_h);
                            }
                            RopeSim* sim = gp_table_get_rope_sim(t);
                            std::unordered_map<int, std::vector<float>> proj_cache;
                            std::vector<float> pts;
                            pts.reserve(ms.verts.size() * 2);
                            for (const auto &vp : ms.verts) {
                                if (vp.rope_id == 0ull) continue;
                                int rope_idx = gp_table_resolve_rope_id_to_sim_index(t, vp.rope_id);
                                int vid = vp.vertex_idx;
                                if (rope_idx < 0 || vid < 0) continue;
                                auto itp = proj_cache.find(rope_idx);
                                if (itp == proj_cache.end()) {
                                    int vc = 0;
                                    if (sim) vc = rope_sim_get_vertex_count(sim, rope_idx);
                                    if (vc > 0) {
                                        std::vector<float> proj(static_cast<size_t>(vc * 2));
                                        int got = gp_table_get_projected_rope_vertices(t, rope_idx, proj.data(), static_cast<int>(proj.size()));
                                        if (got == vc) proj_cache[rope_idx] = std::move(proj);
                                        else proj_cache[rope_idx] = std::vector<float>();
                                    } else {
                                        proj_cache[rope_idx] = std::vector<float>();
                                    }
                                    itp = proj_cache.find(rope_idx);
                                }
                                const auto &proj = itp->second;
                                if (proj.empty()) continue;
                                if (vid >= 0 && vid < static_cast<int>(proj.size() / 2)) {
                                    float px = proj[static_cast<size_t>(vid) * 2 + 0] + table_off_x;
                                    float py = proj[static_cast<size_t>(vid) * 2 + 1] + table_off_y;
                                    pts.push_back(px);
                                    pts.push_back(py);
                                }
                            }
                            if (!pts.empty()) {
                                int n = static_cast<int>(pts.size() / 2);
                                c->lasso_cb(c->lasso_cb_user, 3, pts.data(), n);
                            }
                        }
                    }
                }
            }
            it = c->pending_meta_snapshots.erase(it);
        } else ++it;
    }
    return 1;
}

// Register an overlay using persisted overlay keys during table deserialization.
// This ensures a single canonical overlay instance exists for the saved keys
// and avoids creating duplicate overlays via heuristic code paths.
extern "C" int gp_canvas_register_table_overlay(GP_CanvasContext* ctx_, unsigned long long key_a, unsigned long long key_b, uint64_t port_uuid_a, uint64_t port_uuid_b) {
    if (!ctx_ || (key_a == 0ull && key_b == 0ull)) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // If either key already maps to an overlay, nothing to do.
    if (key_a && c->overlay_key_map.find(key_a) != c->overlay_key_map.end()) return 1;
    if (key_b && c->overlay_key_map.find(key_b) != c->overlay_key_map.end()) return 1;
    // Create a single overlay entry using the supplied keys and optional port UUIDs.
    int oid = c->next_overlay_id++;
    GP_CanvasContextImpl::OverlayEntry ov{};
    ov.id = oid;
    ov.key_a = key_a;
    ov.key_b = key_b;
    ov.x1 = ov.y1 = ov.x2 = ov.y2 = 0.0f;
    // Use supplied port UUIDs when non-zero, otherwise assign deterministic ones.
    if (port_uuid_a != 0ull) ov.port_uuid_a = port_uuid_a; else ov.port_uuid_a = gp_canvas_generate_id(reinterpret_cast<GP_CanvasContext*>(c), 0ull);
    if (port_uuid_b != 0ull) ov.port_uuid_b = port_uuid_b; else ov.port_uuid_b = gp_canvas_generate_id(reinterpret_cast<GP_CanvasContext*>(c), 0ull);
    c->overlays[static_cast<size_t>(oid)] = ov;
    if (ov.key_a) c->overlay_key_map[ov.key_a] = ov.id;
    if (ov.key_b) c->overlay_key_map[ov.key_b] = ov.id;
    if (ov.port_uuid_a) c->overlay_port_uuid_map[ov.port_uuid_a] = std::make_pair(ov.id, 0);
    if (ov.port_uuid_b) c->overlay_port_uuid_map[ov.port_uuid_b] = std::make_pair(ov.id, 1);
    return 1;
}

// Retrieve overlay port UUIDs for the given overlay keys.
extern "C" int gp_canvas_get_overlay_port_uuids(GP_CanvasContext* ctx_, unsigned long long key_a, unsigned long long key_b, uint64_t* out_port_a, uint64_t* out_port_b) {
    if (!ctx_ || (key_a == 0ull && key_b == 0ull)) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    int oid = -1;
    if (key_a) {
        auto it = c->overlay_key_map.find(key_a);
        if (it != c->overlay_key_map.end()) oid = it->second;
    }
    if (oid < 0 && key_b) {
        auto it = c->overlay_key_map.find(key_b);
        if (it != c->overlay_key_map.end()) oid = it->second;
    }
    if (oid < 0) return 0;
    if (static_cast<size_t>(oid) >= c->overlays.size()) return 0;
    const auto &ov = c->overlays[static_cast<size_t>(oid)];
    if (out_port_a) *out_port_a = ov.port_uuid_a;
    if (out_port_b) *out_port_b = ov.port_uuid_b;
    return 1;
}

extern "C" int gp_canvas_attach_rope_to_overlay(GP_CanvasContext* ctx_, unsigned long long key_a, unsigned long long key_b, int rope_idx) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // Ensure a root module exists to host canonical LED contacts for this overlay
    int root_mod = canvas_ensure_root_module(c);
    if (root_mod < 0) return 0;
    // If the canvas has not yet registered any rope ids (common during
    // initial restore), register rope_ids from existing module/root tables
    // so attachments can resolve persisted rope ids to runtime indices.
    if (c->rope_id_map.empty()) {
        // Use table API accessors (no internal struct access from canvas unit)
        for (size_t ti = 0; ti < c->module_tables.size(); ++ti) {
            GP_TableContext* tt = c->module_tables[ti];
            if (!tt) continue;
            int cnt = gp_table_get_rope_id_count(tt);
            if (cnt > 0) {
                std::vector<uint64_t> tmp; tmp.resize(static_cast<size_t>(cnt));
                int got = gp_table_get_rope_ids(tt, tmp.data(), cnt);
                if (got > 0) gp_canvas_register_table_rope_ids_from_array(reinterpret_cast<GP_CanvasContext*>(c), tt, tmp.data(), got);
            }
        }
        // Also register root/container table if present
        GP_TableContext* rt = canvas_ensure_root_table(c);
        if (rt) {
            int cnt = gp_table_get_rope_id_count(rt);
            if (cnt > 0) {
                std::vector<uint64_t> tmp; tmp.resize(static_cast<size_t>(cnt));
                int got = gp_table_get_rope_ids(rt, tmp.data(), cnt);
                if (got > 0) gp_canvas_register_table_rope_ids_from_array(reinterpret_cast<GP_CanvasContext*>(c), rt, tmp.data(), got);
            }
        }
    }
    // Choose two contact indices for the overlay endpoints. Use 0 and 1 by default.
    GP_CanvasEdgeDesc desc{};
    desc.a_module = root_mod; desc.a_contact_idx = 0;
    desc.b_module = root_mod; desc.b_contact_idx = 1;
    // Add a canvas edge which will also enqueue a root-table edge and setup FIFOs.
    int edge_idx = gp_canvas_add_edge_with_type(ctx_, &desc, /*type_id=*/0);
    if (edge_idx < 0) return 0;
    // Now embellish the created edge with overlay keys and rope mapping so
    // rendering and interaction resolve to the overlay positions while the
    // root table holds the canonical keys for network/connectivity.
    if (edge_idx >= static_cast<int>(c->edges.size())) return 0;
    auto &created = c->edges[static_cast<size_t>(edge_idx)];
    printf("gp_canvas_attach_rope_to_overlay: called key_a=%llu key_b=%llu rope_idx=%d edge_idx=%d table_root_mod=%d\n", (unsigned long long)key_a, (unsigned long long)key_b, rope_idx, edge_idx, root_mod);
    created.overlay_key_a = key_a;
    created.overlay_key_b = key_b;
    created.rope_idx = rope_idx;
    printf("  -> attached: edges.size=%zu overlays.size=%zu rope_id_map_size=%zu\n", c->edges.size(), c->overlays.size(), c->rope_id_map.size());
    // compute canonical root keys used by gp_canvas_add_edge_with_type above
    uint64_t rka = (static_cast<uint64_t>(static_cast<uint32_t>(root_mod)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(0)) << 16) | static_cast<uint64_t>(0);
    uint64_t rkb = (static_cast<uint64_t>(static_cast<uint32_t>(root_mod)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(1)) << 16) | static_cast<uint64_t>(0);
    c->canonical_to_overlay[rka] = key_a;
    c->canonical_to_overlay[rkb] = key_b;
    return edge_idx;
}

extern "C" int gp_canvas_attach_rope_to_overlay_with_module(GP_CanvasContext* ctx_, unsigned long long key_a, unsigned long long key_b, int rope_idx, int module_idx, int a_contact_idx, int b_contact_idx) {
    if (!ctx_) return -1;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // validate module indices
    if (module_idx < 0 || module_idx >= static_cast<int>(c->modules.size())) return -1;
    GP_CanvasEdgeDesc desc{};
    desc.a_module = module_idx; desc.a_contact_idx = a_contact_idx;
    desc.b_module = module_idx; desc.b_contact_idx = b_contact_idx;
    int edge_idx = gp_canvas_add_edge_with_type(ctx_, &desc, /*type_id=*/0);
    if (edge_idx < 0) return -1;
    if (edge_idx >= static_cast<int>(c->edges.size())) return -1;
    auto &created = c->edges[static_cast<size_t>(edge_idx)];
    created.overlay_key_a = key_a;
    created.overlay_key_b = key_b;
    created.rope_idx = rope_idx;
    uint64_t rka = (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(a_contact_idx)) << 16) | static_cast<uint64_t>(0);
    uint64_t rkb = (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(b_contact_idx)) << 16) | static_cast<uint64_t>(0);
    c->canonical_to_overlay[rka] = key_a;
    c->canonical_to_overlay[rkb] = key_b;
    return edge_idx;
}

// Resolve a canonical root key to overlay pixel coords if it maps to an overlay
extern "C" int gp_canvas_resolve_canonical_key(GP_CanvasContext* ctx_, unsigned long long key, int* out_x, int* out_y) {
    if (!ctx_ || !out_x || !out_y) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    auto it = c->canonical_to_overlay.find(key);
    if (it == c->canonical_to_overlay.end()) return 0;
    unsigned long long overlay_key = it->second;
    return gp_canvas_resolve_overlay_key(ctx_, overlay_key, out_x, out_y);
}

extern "C" int gp_canvas_resolve_overlay_key(GP_CanvasContext* ctx_, unsigned long long key, int* out_x, int* out_y) {
    if (!ctx_ || !out_x || !out_y) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    uint32_t r_orig = static_cast<uint32_t>(key >> 32);
    if (r_orig != 0xFFFFFFFFu) return 0;
    uint32_t c_idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
    uint32_t led = static_cast<uint32_t>(key & 0xFFFFu);
    auto it = c->overlays.find(static_cast<int>(c_idx));
    if (it == c->overlays.end()) return 0;
    const auto &ov = it->second;
    if (led == 0u) { *out_x = static_cast<int>(std::lround(ov.x1)); *out_y = static_cast<int>(std::lround(ov.y1)); }
    else { *out_x = static_cast<int>(std::lround(ov.x2)); *out_y = static_cast<int>(std::lround(ov.y2)); }
    return 1;
}

// Set an authoritative meta binding on an overlay (table + meta-group)
extern "C" int gp_canvas_set_overlay_meta(GP_CanvasContext* ctx_, unsigned long long key_a, unsigned long long key_b, GP_TableContext* table, void* meta_mg) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    auto set_for_key = [&](unsigned long long key)->int{
        if (!key) return 0;
        // Prefer direct key->overlay id mapping to avoid relying on encoded
        // id fields which may have been synthesized differently across
        // restore paths. This ensures we always bind to the unique overlay
        // instance owning these keys when present.
        auto itmap = c->overlay_key_map.find(key);
        GP_CanvasContextImpl::OverlayEntry* pov = nullptr;
        if (itmap != c->overlay_key_map.end()) {
            int found_id = itmap->second;
            auto it2 = c->overlays.find(found_id);
            if (it2 != c->overlays.end()) pov = &it2->second;
        }
        if (!pov) {
            uint32_t sentinel = static_cast<uint32_t>(key >> 32);
            if (sentinel != 0xFFFFFFFFu) return 0;
            uint32_t idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
            auto it = c->overlays.find(static_cast<int>(idx));
            if (it == c->overlays.end()) return 0;
            pov = &it->second;
        }
        auto &ov = *pov;
        // If overlay is already bound to the same table+meta-group, no-op
        if (ov.meta_table == table && ov.meta_mg == reinterpret_cast<GP_MetaGroup*>(meta_mg)) return 1;
        // Prevent rebinding an overlay to a different meta-group: prefer the
        // existing authoritative binding. If both meta-groups expose a
        // persistent mgid and they match, allow binding; otherwise refuse.
        if (ov.meta_mg && ov.meta_mg != reinterpret_cast<GP_MetaGroup*>(meta_mg)) {
            unsigned long long existing_mgid = 0ull;
            unsigned long long new_mgid = 0ull;
            if (ov.meta_table && ov.meta_mg) gp_table_meta_get_id(ov.meta_table, ov.meta_mg, &existing_mgid);
            if (table && meta_mg) gp_table_meta_get_id(table, reinterpret_cast<GP_MetaGroup*>(meta_mg), &new_mgid);
            if (existing_mgid != 0ull || new_mgid != 0ull) {
                if (existing_mgid == new_mgid) return 1; // same persistent mgid, treat as idempotent
            }
            printf("gp_canvas_set_overlay_meta: overlay id=%d already bound mg=%p (mgid=%llu) refusing rebinding to mg=%p (mgid=%llu)\n",
                ov.id, (void*)ov.meta_mg, (unsigned long long)existing_mgid, (void*)meta_mg, (unsigned long long)new_mgid);
            fflush(stdout);
            return 0;
        }
        printf("gp_canvas_set_overlay_meta: binding overlay id=%d to table=%p mg=%p\n", ov.id, (void*)table, (void*)meta_mg);
        fflush(stdout);
        // Hard-fail if the meta-group has no lasso membership or sim group
        if (meta_mg && table) {
            GP_MetaGroup* mgptr = reinterpret_cast<GP_MetaGroup*>(meta_mg);
            int vcount = gp_table_meta_get_vertex_count(table, mgptr);
            int sim_idx = -1; gp_table_meta_get_sim_group_index(table, mgptr, &sim_idx);
            if (vcount <= 0) {
                printf("FATAL: gp_canvas_set_overlay_meta: overlay id=%d binding to empty meta-group mg=%p (vcount=0) - aborting\n", ov.id, (void*)mgptr);
                fflush(stdout);
                std::abort();
            }
            if (sim_idx < 0) {
                printf("FATAL: gp_canvas_set_overlay_meta: overlay id=%d binding to meta-group mg=%p without sim_group (sim_idx=%d) - aborting\n", ov.id, (void*)mgptr, sim_idx);
                fflush(stdout);
                std::abort();
            }
        }
        ov.meta_table = table;
        ov.meta_mg = reinterpret_cast<GP_MetaGroup*>(meta_mg);
        // Also record overlay keys on the meta-group via table API so
        // serialization can persist the association without needing the
        // meta-group definition in this translation unit.
        if (meta_mg && table) {
            gp_table_meta_set_overlay_keys(table, reinterpret_cast<GP_MetaGroup*>(meta_mg), ov.key_a, ov.key_b);
        }
        return 1;
    };
    int a = set_for_key(key_a);
    int b = set_for_key(key_b);
    return (a || b) ? 1 : 0;
}

// Clear and remove all overlays and table meta-groups, then clear modules.
extern "C" int gp_canvas_clear_meta_and_overlays(GP_CanvasContext* ctx_) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);

    // 1) Destroy all meta-groups in the container table (if any)
    if (c->container_table) {
        GP_TableContext* t = c->container_table;
        int mgc = gp_table_get_meta_group_count(t);
        for (int i = mgc - 1; i >= 0; --i) {
            GP_MetaGroup* mg = gp_table_get_meta_group(t, i);
            if (mg) gp_table_meta_destroy(t, mg);
        }
    }

    // 2) Destroy all meta-groups in each module table
    for (size_t mi = 0; mi < c->module_tables.size(); ++mi) {
        GP_TableContext* t = c->module_tables[mi];
        if (!t) continue;
        int mgc = gp_table_get_meta_group_count(t);
        for (int i = mgc - 1; i >= 0; --i) {
            GP_MetaGroup* mg = gp_table_get_meta_group(t, i);
            if (mg) gp_table_meta_destroy(t, mg);
        }
    }

    // 3) Remove any canvas edges that reference overlays (erase from root table)
    for (int ei = static_cast<int>(c->edges.size()) - 1; ei >= 0; --ei) {
        const auto &e = c->edges[static_cast<size_t>(ei)];
        if (e.overlay_key_a || e.overlay_key_b) {
            canvas_remove_edge_at(c, ei);
        }
    }

    // 4) Clear overlay bindings on the canvas and inform tables (best-effort)
    for (auto &kv : c->overlays) {
        auto &ov = kv.second;
        if (ov.meta_table && ov.meta_mg) {
            // clear overlay keys recorded on the meta-group (if still present)
            gp_table_meta_set_overlay_keys(ov.meta_table, ov.meta_mg, 0ull, 0ull);
        }
    }
    c->overlays.clear();
    c->canonical_to_overlay.clear();
    c->overlay_key_map.clear();

    // 5) Clear pending snapshot queues
    c->pending_meta_snapshots.clear();
    c->post_load_meta_pending.clear();

    // 6) Clear modules/workspace (destroys tables/stages and remaining edges)
    canvas_clear_workspace(c);

    // reset overlay id generator
    c->next_overlay_id = 1;

    return 1;
}

extern "C" int gp_canvas_get_root_module_idx() {
    if (!g_canvas_context_singleton) return -1;
    return g_canvas_context_singleton->root_module_idx;
}

// Click-listen mode controls
extern "C" int gp_canvas_set_click_listen_mode(GP_CanvasContext* ctx_, int enable) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->click_listen_mode = (enable != 0);
    return 1;
}

extern "C" int gp_canvas_get_click_listen_mode(GP_CanvasContext* ctx_, int* out_enabled) {
    if (!ctx_ || !out_enabled) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    *out_enabled = c->click_listen_mode ? 1 : 0;
    return 1;
}

// Bind the current pending action pointer into the most-left unused
// receive frame ptr for the module. Transfers ownership of the pending
// action into the module frame (does not copy).
extern "C" int gp_canvas_bind_pending_action_to_module(GP_CanvasContext* ctx_, int module_idx) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!c->pending_action) return 0;
    if (module_idx >= static_cast<int>(c->module_frame_links.size())) return 0;
    auto &links = c->module_frame_links[module_idx];
    // Search leftmost LED indices first. For each LED index, bind into the
    // full send/receive pair atomically. Prefer the left column pair
    // (send-left,row=0 and receive-left,row=2) then the right column pair
    // (send-right,row=1 and receive-right,row=3).
    for (int i = 0; i < kModuleExtraLedCount; ++i) {
        // left column pair
        bool left_free = (links.ptrs[0][static_cast<size_t>(i)] == nullptr) && (links.ptrs[2][static_cast<size_t>(i)] == nullptr);
        if (left_free) {
            void* p = reinterpret_cast<void*>(c->pending_action);
            links.ptrs[0][static_cast<size_t>(i)] = p; // send-left
            links.ptrs[2][static_cast<size_t>(i)] = p; // receive-left
            GP_TableCell* cell_send = module_frame_led_cell(c, module_idx, 0, i);
            GP_TableCell* cell_recv = module_frame_led_cell(c, module_idx, 2, i);
            if (cell_send) cell_send->reserved0 = 1;
            if (cell_recv) cell_recv->reserved0 = 1;
            c->pending_action = nullptr;
            printf("gp_canvas_bind_pending_action_to_module: bound pending action to module=%d left-pair idx=%d\n", module_idx, i);
            return 1;
        }
        // right column pair
        bool right_free = (links.ptrs[1][static_cast<size_t>(i)] == nullptr) && (links.ptrs[3][static_cast<size_t>(i)] == nullptr);
        if (right_free) {
            void* p = reinterpret_cast<void*>(c->pending_action);
            links.ptrs[1][static_cast<size_t>(i)] = p; // send-right
            links.ptrs[3][static_cast<size_t>(i)] = p; // receive-right
            GP_TableCell* cell_send = module_frame_led_cell(c, module_idx, 1, i);
            GP_TableCell* cell_recv = module_frame_led_cell(c, module_idx, 3, i);
            if (cell_send) cell_send->reserved0 = 1;
            if (cell_recv) cell_recv->reserved0 = 1;
            c->pending_action = nullptr;
            printf("gp_canvas_bind_pending_action_to_module: bound pending action to module=%d right-pair idx=%d\n", module_idx, i);
            return 1;
        }
    }
    return 0;
}

extern "C" void* gp_canvas_get_module_frame_ptr(GP_CanvasContext* ctx_, int module_idx, int is_send, int led_idx) {
    if (!ctx_) return nullptr;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->module_frame_links.size())) return nullptr;
    if (led_idx < 0 || led_idx >= kModuleExtraLedCount) return nullptr;
    int row = is_send ? 0 : 2; // map send->send-left, receive->receive-left
    return c->module_frame_links[module_idx].ptrs[static_cast<size_t>(row)][static_cast<size_t>(led_idx)];
}

// Convenience: get module frame pointer by contact index (contact space uses
// kModuleFrameContactBase + row * kModuleExtraLedCount + idx). This maps the
// contact index into the appropriate send/receive and left/right row index.
extern "C" void* gp_canvas_get_module_frame_ptr_for_contact(GP_CanvasContext* ctx_, int module_idx, int contact_idx) {
    if (!ctx_) return nullptr;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->module_frame_links.size())) return nullptr;
    int base = kModuleFrameContactBase;
    if (contact_idx < base) return nullptr;
    int local = contact_idx - base;
    const int full_cols = 2;
    int total = kModuleExtraLedCount * kModuleExtraLedRows;
    if (local < 0 || local >= total) return nullptr;
    int row = local / kModuleExtraLedCount; // logical row index 0..kModuleExtraLedRows-1
    int led_idx = local % kModuleExtraLedCount;
    int grid_row = row / full_cols; // 0 => send, 1 => receive
    int grid_col = row % full_cols; // 0 => left, 1 => right
    int row_index = 0;
    if (grid_row == 0) { // send
        row_index = (grid_col == 0) ? 0 : 1;
    } else { // receive
        row_index = (grid_col == 0) ? 2 : 3;
    }
    return c->module_frame_links[static_cast<size_t>(module_idx)].ptrs[static_cast<size_t>(row_index)][static_cast<size_t>(led_idx)];
}

extern "C" int gp_canvas_raster_rgba(GP_CanvasContext* ctx_, uint8_t* out_rgba, int32_t out_len_bytes) {
    if (!ctx_ || !out_rgba) return 0;
    auto *ctx = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    int w = ctx->width;
    int h = ctx->height;
    int pitch = w * 4;
    if (out_len_bytes < w * h * 4) return 0;

    update_canvas_scroll_state(ctx, /*pull_from_container=*/true);
    canvas_update_subgroup_palette(ctx);

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
        const int edge_btn_count = 5;
        const int save_btn_count = 2;
        const int table_btn_count = 3;
        const int kpn_btn_count = 3;
        const int spacing = 8;
        int inner_h = std::max(0, cbh - 8);
        int row_gap = 4;
        int row_h = std::max(4, (inner_h - row_gap) / 2);
        int by0 = rb + 4;
        int by1 = by0 + row_h + row_gap;
        int bh = row_h;
        int bw = bh; // square buttons
        // left (canvas) group
        int bx = 8; int by = by0;
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
            bool selected = (bi < 4) ? (ctx->selected_tool_edge == bi) : ctx->lasso_mode;
            Color fill;
            if (bi < 4) fill = selected ? Color{90,80,70,255} : Color{60,54,48,255};
            else fill = selected ? Color{110,70,80,255} : Color{70,58,52,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, by, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
            const char* edge_labels[5] = {
                LABEL_EDGE_TOOL_CREATE,
                LABEL_EDGE_TOOL_DESTROY,
                LABEL_EDGE_TOOL_ON_CHANGE,
                LABEL_EDGE_TOOL_CONTINUOUS,
                LABEL_EDGE_TOOL_LASSO
            };
            auto lbm = render_text_to_rgba(edge_labels[bi], 0.85f, {235,228,218,255});
            const char* edge_short[5] = {
                LABEL_EDGE_TOOL_SHORT_CREATE,
                LABEL_EDGE_TOOL_SHORT_DESTROY,
                LABEL_EDGE_TOOL_SHORT_ON_CHANGE,
                LABEL_EDGE_TOOL_SHORT_CONTINUOUS,
                LABEL_EDGE_TOOL_SHORT_LASSO
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
        int group_width = table_btn_count * (bw + spacing) - spacing;
        int bx_r = std::max(8, w - 8 - group_width);
        int save_group_w = save_btn_count * (bw + spacing) - spacing;
        int bx_save = bx_r - save_group_w - spacing * 2;
        for (int bi = 0; bi < save_btn_count; ++bi) {
            int bx_i = bx_save + bi * (bw + spacing);
            Color fill = Color{50,62,54,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, by, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=36; pleft[1]=36; pleft[2]=40; pleft[3]=255;
                pright[0]=36; pright[1]=36; pright[2]=40; pright[3]=255;
            }
            const char* save_labels[2] = { LABEL_CANVAS_SAVE, LABEL_CANVAS_CLEAR };
            const char* save_short[2] = { LABEL_CANVAS_SAVE_SHORT, LABEL_CANVAS_CLEAR_SHORT };
            auto lbm = render_text_to_rgba(save_labels[bi], 0.85f, {225,235,228,255});
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
            auto small = render_text_to_rgba(save_short[bi], 1.0f, {225,235,228,255});
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
        auto draw_action_button = [&](int bx, int byy, const char* label, Color fill) {
            memset_rect(out_rgba, w, h, pitch, bx, byy, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = byy + oy; if (y < 0 || y >= h) continue;
                int left_x = bx; int right_x = bx + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
            auto text = render_text_to_rgba(label, 0.8f, {230,230,230,255});
            if (!text.pixels.empty()) {
                int tx = bx + (bw - text.width) / 2;
                int ty = byy + (bh - text.height) / 2;
                for (int yy = 0; yy < text.height; ++yy) {
                    int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < text.width; ++xx) {
                        int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &text.pixels[(yy * text.width + xx) * 4];
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
        int io_base_x = bx_save - spacing * 2; // place IO groups to the left of save/clear buttons
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

        // second row: KPN tools + thread manager controls
        int kpn_by = by1;
        int bx_kpn = 8;
        for (int bi = 0; bi < kpn_btn_count; ++bi) {
            int bx_i = bx_kpn + bi * (bw + spacing);
            bool selected = (ctx->selected_tool_kpn == bi);
            Color fill = selected ? Color{80,90,110,255} : Color{52,58,70,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, kpn_by, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = kpn_by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
            const char* kpn_labels[3] = {
                LABEL_KPN_TOOL_K,
                LABEL_KPN_TOOL_P,
                LABEL_KPN_TOOL_N
            };
            const char* kpn_short[3] = {
                LABEL_KPN_TOOL_SHORT_K,
                LABEL_KPN_TOOL_SHORT_P,
                LABEL_KPN_TOOL_SHORT_N
            };
            auto lbm = render_text_to_rgba(kpn_labels[bi], 0.8f, {220,228,240,255});
            if (!lbm.pixels.empty()) {
                int tx = bx_i + (bw - lbm.width) / 2;
                int ty = kpn_by + (bh - lbm.height) / 2;
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
            auto small = render_text_to_rgba(kpn_short[bi], 1.0f, {235,238,245,255});
            if (!small.pixels.empty()) {
                int txs = bx_i + (bw - small.width) / 2;
                int tys = kpn_by + (bh - small.height) / 2;
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
        int kpn_group_w = kpn_btn_count * (bw + spacing) - spacing;
        int kpn_right = bx_kpn + kpn_group_w;
        int delay_num_w = std::max(24, nbw * 2);
        int delay_gap = 10;
        int delay_total_w = nbw + delay_gap + delay_num_w + delay_gap + nbw;
        int play_gap = 12;
        int bx_play = w - 8 - bw;
        int action_btn_gap = 6;
        int action_btn_count = 3;
        int action_group_w = action_btn_count * bw + action_btn_gap * (action_btn_count - 1);
        int bx_action_right = bx_play - play_gap;
        int bx_action_left = bx_action_right - action_group_w;
        int bx_delay_plus = bx_action_left - play_gap;
        int delay_left = bx_delay_plus - delay_total_w;
        int subgroup_btn_count = kSubgroupBinCount;
        int subgroup_group_w = subgroup_btn_count * (bw + spacing) - spacing;
        int subgroup_left = kpn_right + spacing * 2;
        int available = delay_left - subgroup_left;
        if (available < subgroup_group_w) {
            subgroup_left = std::max(kpn_right + spacing, delay_left - subgroup_group_w);
        }
        // reset toolbar LED hit list for this frame
        ctx->toolbar_leds.clear();
        for (int bi = 0; bi < subgroup_btn_count; ++bi) {
            int bx_i = subgroup_left + bi * (bw + spacing);
            uint32_t flags = subgroup_mask_for_index(bi);
            Color fill = subgroup_flags_to_color(ctx, flags, 255);
            bool selected = (ctx->selected_tool_subgroup_flags & subgroup_mask_for_index(bi)) != 0u;
            if (!selected) {
                fill.r = static_cast<uint8_t>(std::lround(fill.r * 0.75f));
                fill.g = static_cast<uint8_t>(std::lround(fill.g * 0.75f));
                fill.b = static_cast<uint8_t>(std::lround(fill.b * 0.75f));
            }
            memset_rect(out_rgba, w, h, pitch, bx_i, kpn_by, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = kpn_by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=selected ? 240 : 40; pleft[1]=selected ? 240 : 40; pleft[2]=selected ? 240 : 44; pleft[3]=255;
                pright[0]=selected ? 240 : 40; pright[1]=selected ? 240 : 40; pright[2]=selected ? 240 : 44; pright[3]=255;
            }
            Color led_target = unpack_rgba(ctx->subgroup_target_rgba[static_cast<size_t>(bi)].load(std::memory_order_acquire));
            int led_r = std::max(2, bw / 6);
            // shift LED right by roughly its own radius so it sits between buttons
            int led_cx = bx_i + bw + led_r;
            int led_cy = kpn_by + bh / 2;
            draw_circle(out_rgba, w, h, pitch, led_cx, led_cy, led_r + 1, Color{20,20,28,255});
            draw_circle(out_rgba, w, h, pitch, led_cx, led_cy, led_r, led_target);
            if (selected) {
                int y_top = kpn_by;
                int y_bot = kpn_by + bh - 1;
                for (int xx = 0; xx < bw; ++xx) {
                    int xh = bx_i + xx;
                    if (xh < 0 || xh >= w) continue;
                    if (y_top >= 0 && y_top < h) {
                        uint8_t* pt = out_rgba + y_top * pitch + xh * 4;
                        pt[0]=240; pt[1]=240; pt[2]=240; pt[3]=255;
                    }
                    if (y_bot >= 0 && y_bot < h) {
                        uint8_t* pb = out_rgba + y_bot * pitch + xh * 4;
                        pb[0]=240; pb[1]=240; pb[2]=240; pb[3]=255;
                    }
                }
            }
            // record toolbar LED hit rect (view coords and world coords)
            {
                GP_CanvasContextImpl::ToolbarLedBox tb;
                tb.x0 = led_cx - (led_r + 2);
                tb.y0 = led_cy - (led_r + 2);
                tb.x1 = led_cx + (led_r + 2);
                tb.y1 = led_cy + (led_r + 2);
                tb.wx0 = tb.x0 + ctx->offset_x; tb.wy0 = tb.y0 + ctx->offset_y;
                tb.wx1 = tb.x1 + ctx->offset_x; tb.wy1 = tb.y1 + ctx->offset_y;
                tb.subgroup_idx = bi;
                ctx->toolbar_leds.push_back(tb);
            }
        }
        // Spawn-root button: place to the right of subgroup colors/LEDs
        int bx_spawn = subgroup_left + subgroup_group_w + spacing;
        // ensure we don't draw off-screen; clamp to reasonable area near action group
        if (bx_spawn + bw < bx_delay_plus) {
            draw_action_button(bx_spawn, kpn_by, "ROOT", Color{70,70,90,255});
        }
        draw_io_group(bx_delay_plus, kpn_by, std::max(0, ctx->thread_mgr_delay_ms), LABEL_THREAD_DELAY_SHORT);

        int bx_clone = bx_action_left;
        int bx_clear = bx_clone + bw + action_btn_gap;
        int bx_destroy = bx_clear + bw + action_btn_gap;
        draw_action_button(bx_clone, kpn_by, LABEL_MODULE_CLONE_SHORT, Color{58,58,70,255});
        draw_action_button(bx_clear, kpn_by, LABEL_MODULE_CLEAR_SHORT, Color{64,56,52,255});
        draw_action_button(bx_destroy, kpn_by, LABEL_MODULE_DESTROY_SHORT, Color{70,52,52,255});

        // Split the former play button into a small SIM button (left half)
        // and a reduced play button (right half).
        int half_play_w = std::max(8, bw / 2);
        int sim_btn_w = half_play_w;
        int play_btn_w = bw - sim_btn_w;
        int sim_x = bx_play;
        int play_x = bx_play + sim_btn_w;
        Color sim_fill = Color{46,46,56,255};
        memset_rect(out_rgba, w, h, pitch, sim_x, kpn_by, sim_btn_w, bh, sim_fill);
        Color play_fill = ctx->thread_mgr_paused ? Color{70,60,70,255} : Color{60,80,60,255};
        memset_rect(out_rgba, w, h, pitch, play_x, kpn_by, play_btn_w, bh, play_fill);
        for (int oy = 0; oy < bh; ++oy) {
            int y = kpn_by + oy; if (y < 0 || y >= h) continue;
            int left_x = play_x; int right_x = play_x + play_btn_w - 1;
            uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
            uint8_t* pright = out_rgba + y * pitch + right_x * 4;
            pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
            pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
        }
        const char* play_label = ctx->thread_mgr_paused ? LABEL_THREAD_PLAY : LABEL_THREAD_PAUSE;
        const char* play_short = ctx->thread_mgr_paused ? LABEL_THREAD_PLAY_SHORT : LABEL_THREAD_PAUSE_SHORT;
        auto play_text = render_text_to_rgba(play_label, 0.75f, {230,230,230,255});
        if (!play_text.pixels.empty()) {
            int tx = play_x + (play_btn_w - play_text.width) / 2;
            int ty = kpn_by + (bh - play_text.height) / 2;
            for (int yy = 0; yy < play_text.height; ++yy) {
                int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < play_text.width; ++xx) {
                    int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &play_text.pixels[(yy * play_text.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
        auto play_short_text = render_text_to_rgba(play_short, 1.0f, {240,240,240,255});
        if (!play_short_text.pixels.empty()) {
            int txs = play_x + (play_btn_w - play_short_text.width) / 2;
            int tys = kpn_by + (bh - play_short_text.height) / 2;
            for (int yy = 0; yy < play_short_text.height; ++yy) {
                int dst_y = tys + yy; if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < play_short_text.width; ++xx) {
                    int dst_x = txs + xx; if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &play_short_text.pixels[(yy * play_short_text.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
        // Draw SIM label into sim button (left half)
        const char* sim_label = "sim";
        auto sim_text = render_text_to_rgba(sim_label, 0.8f, {220,220,235,255});
        if (!sim_text.pixels.empty()) {
            int txs2 = sim_x + (sim_btn_w - sim_text.width) / 2;
            int tys2 = kpn_by + (bh - sim_text.height) / 2;
            for (int yy = 0; yy < sim_text.height; ++yy) {
                int dst_y = tys2 + yy; if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < sim_text.width; ++xx) {
                    int dst_x = txs2 + xx; if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &sim_text.pixels[(yy * sim_text.width + xx) * 4];
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

    // Prepare storage for per-module table hitboxes discovered during table rendering.
    std::vector<std::vector<GP_TableHitBox>> module_hitboxes(ctx->modules.size());
    std::vector<std::unordered_map<int, ContactLight>> module_contact_lights(ctx->modules.size());
    std::vector<std::unordered_map<int, bool>> module_frame_roles(ctx->modules.size());
    std::vector<std::vector<uint8_t>> module_tables(ctx->modules.size());
    std::vector<std::unordered_map<int, uint32_t>> module_contact_subgroups(ctx->modules.size());

    for (const auto &edge : ctx->edges) {
        if (edge.subgroup_flags == 0u) continue;
        if (edge.desc.a_module >= 0 && edge.desc.a_module < static_cast<int>(module_contact_subgroups.size())) {
            module_contact_subgroups[edge.desc.a_module][edge.desc.a_contact_idx] |= edge.subgroup_flags;
        }
        if (edge.desc.b_module >= 0 && edge.desc.b_module < static_cast<int>(module_contact_subgroups.size())) {
            module_contact_subgroups[edge.desc.b_module][edge.desc.b_contact_idx] |= edge.subgroup_flags;
        }
    }

    // First pass: render tables and gather hitboxes/light info (no drawing yet).
    for (int mi = 0; mi < static_cast<int>(ctx->modules.size()); ++mi) {
        const auto &m = ctx->modules[mi];
        module_hitboxes[mi].clear();
        module_contact_lights[mi].clear();
        module_frame_roles[mi].clear();
        module_tables[mi].clear();
        GP_TableContext* t = (mi >= 0 && mi < static_cast<int>(ctx->module_tables.size())) ? ctx->module_tables[mi] : nullptr;
        if (t) {
            // Ensure per-module table layout is up-to-date before rendering it.
            sync_module_table_io_layout(ctx, mi);
            bool is_stage = (mi >= 0 && mi < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[mi]);
            int tw = std::max(1, m.w);
            int th = std::max(1, m.h);
            ModuleLayout layout = module_layout_for(ctx, mi, m);
            int table_clip_h = std::max(1, layout.table_clip_h);
            int table_offset_y = layout.table_y;
            GP_TableStyle st{};
            gp_table_get_style(t, &st);
            Color table_bg{st.bg_rgba[0], st.bg_rgba[1], st.bg_rgba[2], st.bg_rgba[3]};
            GP_TableGeom geom{};
            gp_table_get_geom(t, &geom);
            geom.width_px = tw;
            int table_render_h = std::max(1, layout.table_clip_h);
            module_tables[mi].assign(static_cast<size_t>(tw) * static_cast<size_t>(th) * 4u, 0);
            if (!is_stage) {
                memset_rect(module_tables[mi].data(), tw, th, tw * 4, 0, table_offset_y, tw, table_clip_h, table_bg);
            }
            const int hitcap = 4096;
            std::vector<GP_TableHitBox> hits(hitcap);
            int hits_written = 0;
            if (mi < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[mi]) {
                // For stage modules we want the stage image to act as the full module background.
                geom.height_px = th;
                if (!ctx->module_stage_images[mi].rgba) {
                    gp_table_render_rgba_with_state(t, nullptr, module_tables[mi].data(), static_cast<int32_t>(module_tables[mi].size()), &geom, hits.data(), hitcap, &hits_written);
                } else {
                    const uint8_t* src_stage = ctx->module_stage_images[mi].rgba;
                    size_t bytes = static_cast<size_t>(tw) * static_cast<size_t>(th) * 4u;
                    std::memcpy(module_tables[mi].data(), src_stage, bytes);
                    std::vector<uint8_t> tmp_buf(module_tables[mi].size());
                    int ok = gp_table_render_rgba_with_state(t, nullptr, tmp_buf.data(), static_cast<int32_t>(tmp_buf.size()), &geom, hits.data(), hitcap, &hits_written);
                    if (ok) {
                        size_t pixels = static_cast<size_t>(tw) * static_cast<size_t>(th);
                        uint8_t* dst = module_tables[mi].data();
                        const uint8_t* src = tmp_buf.data();
                        for (size_t pi = 0; pi < pixels; ++pi) {
                            const uint8_t sa = src[pi * 4 + 3];
                            if (sa == 0) continue;
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
                }
            } else {
                geom.height_px = table_render_h;
                std::vector<uint8_t> tmp_buf(static_cast<size_t>(tw) * static_cast<size_t>(table_render_h) * 4u);
                int ok = gp_table_render_rgba_with_state(t, nullptr, tmp_buf.data(), static_cast<int32_t>(tmp_buf.size()), &geom, hits.data(), hitcap, &hits_written);
                if (ok) {
                    int copy_h = std::min(table_clip_h, table_render_h);
                    for (int yy = 0; yy < copy_h; ++yy) {
                        const uint8_t* src = tmp_buf.data() + static_cast<size_t>(yy) * static_cast<size_t>(tw) * 4u;
                        uint8_t* dst = module_tables[mi].data() + static_cast<size_t>(yy + table_offset_y) * static_cast<size_t>(tw) * 4u;
                        std::memcpy(dst, src, static_cast<size_t>(tw) * 4u);
                    }
                }
            }
            if (hits_written > 0) {
                for (int hi = 0; hi < hits_written; ++hi) {
                    const auto &hb = hits[hi];
                    if (!is_stage && hb.y0 >= table_clip_h) continue;
                    GP_TableHitBox adjusted = hb;
                    adjusted.y0 += table_offset_y;
                    adjusted.y1 += table_offset_y;
                    module_hitboxes[mi].push_back(adjusted);
                }
            }
            auto &light_map = module_contact_lights[mi];
            light_map.clear();
            Color led_on{st.led_on_rgba[0], st.led_on_rgba[1], st.led_on_rgba[2], st.led_on_rgba[3]};
            int in_count = (mi >= 0 && mi < static_cast<int>(ctx->module_io_in_count.size())) ? ctx->module_io_in_count[mi] : 0;
            int out_count = (mi >= 0 && mi < static_cast<int>(ctx->module_io_out_count.size())) ? ctx->module_io_out_count[mi] : 0;
            const auto &hits_filtered = module_hitboxes[mi];
            for (size_t hi = 0; hi < hits_filtered.size(); ++hi) {
                const auto &hb = hits_filtered[hi];
                if (hb.part != GP_TABLE_HIT_LED && hb.part != GP_TABLE_HIT_LED_ARG && hb.part != GP_TABLE_HIT_LED_TABLE) continue;
                if (hb.row_idx < 0) continue;
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
                    uint32_t subgroup_flags = 0u;
                    if (mi < static_cast<int>(module_contact_subgroups.size())) {
                        auto it = module_contact_subgroups[mi].find(contact_idx);
                        if (it != module_contact_subgroups[mi].end()) subgroup_flags = it->second;
                    }
                    cl.col = (subgroup_flags != 0u) ? subgroup_flags_to_color(ctx, subgroup_flags, led_on.a) : led_on;
                    cl.intensity = glow;
                    cl.valid = true;
                    light_map[contact_idx] = cl;
                }
            }
        }
            for_each_module_frame_led(ctx, mi, m, [&](int row, int idx, int x0, int y0, int x1, int y1) {
                GP_TableHitBox hb = make_module_frame_led_hitbox(row, idx, x0, y0, x1, y1);
                module_hitboxes[mi].push_back(hb);
                const int full_cols = 2;
                int grid_row = row / full_cols;
                module_frame_roles[mi][hb.aux0] = (grid_row != 0);
            });
            
        // If this is the synthetic root module, also expose toolbar LED
        // hitboxes as module-local frame LEDs using the distinct toolbar
        // contact base so ropes can resolve to the toolbar positions.
        if (mi == ctx->root_module_idx && !ctx->toolbar_leds.empty()) {
            int toolbar_base = kModuleFrameContactBase + kModuleExtraLedCount * kModuleExtraLedRows;
            for (const auto &tb : ctx->toolbar_leds) {
                GP_TableHitBox thb{};
                thb.x0 = tb.wx0 - ctx->modules[mi].x;
                thb.x1 = tb.wx1 - ctx->modules[mi].x;
                thb.y0 = tb.wy0 - ctx->modules[mi].y;
                thb.y1 = tb.wy1 - ctx->modules[mi].y;
                thb.cell_kind = GP_TABLE_CELL_LEDS;
                thb.part = GP_TABLE_HIT_LED;
                thb.row_idx = kModuleFrameRowReceive;
                thb.col_idx = 0;
                thb.aux0 = toolbar_base + tb.subgroup_idx;
                thb.aux1 = 0;
                module_hitboxes[mi].push_back(thb);
                module_frame_roles[mi][thb.aux0] = true; // receive role
            }
        }
        Color frame_led_on{255,210,90,255};
        if (t) {
            GP_TableStyle st{};
            gp_table_get_style(t, &st);
            frame_led_on = Color{st.led_on_rgba[0], st.led_on_rgba[1], st.led_on_rgba[2], st.led_on_rgba[3]};
        }
        auto &frame_light_map = module_contact_lights[mi];
        const int full_cols = 2;
        int pair_count = (kModuleExtraLedRows + full_cols - 1) / full_cols;
        int total_rows = pair_count * full_cols;
        for (int row = 0; row < total_rows; ++row) {
            int base = kModuleFrameContactBase + row * kModuleExtraLedCount;
            for (int idx = 0; idx < kModuleExtraLedCount; ++idx) {
                const GP_TableCell* cell = module_frame_led_cell(ctx, mi, row, idx);
                if (!cell) continue;
                bool on = (cell->flags & 0x1u) != 0u;
                bool active = (static_cast<uint32_t>(cell->reserved0) & 0x1u) != 0u;
                if (!on && !active) continue;
                ContactLight cl;
                cl.col = frame_led_on;
                cl.intensity = on ? 0.6f : 0.35f;
                cl.valid = true;
                frame_light_map[base + idx] = cl;
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
        if (module_idx >= 0 && module_idx < static_cast<int>(module_frame_roles.size())) {
            const auto &roles = module_frame_roles[module_idx];
            auto it = roles.find(contact_idx);
            if (it != roles.end()) return it->second;
        }
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
        bool resolvedA = false, resolvedB = false;
        // If this edge is an overlay (custom overlay keys set), prefer
        // resolving the overlay LED positions to determine rope endpoints
        // so overlay-attached ropes remain stable and don't get cleared.
        auto &edgeinfo = ctx->edges[ei];
        if (edgeinfo.overlay_key_a || edgeinfo.overlay_key_b) {
            int ox = 0, oy = 0, ox2 = 0, oy2 = 0;
            if (gp_canvas_resolve_overlay_key(reinterpret_cast<GP_CanvasContext*>(ctx), edgeinfo.overlay_key_a, &ox, &oy) &&
                gp_canvas_resolve_overlay_key(reinterpret_cast<GP_CanvasContext*>(ctx), edgeinfo.overlay_key_b, &ox2, &oy2)) {
                ax = ox; ay = oy; bx = ox2; by = oy2;
                resolvedA = true; resolvedB = true;
            }
        }
        // compute contact positions. If the endpoint module has an attached table
        // and we captured hitboxes during rendering, prefer the table-provided
        // hitbox center for exact LED anchor coordinates. Fall back to legacy
        // computed positions otherwise.
        const GP_CanvasModuleDesc &ma = ctx->modules[edge.a_module];
        const GP_CanvasModuleDesc &mb = ctx->modules[edge.b_module];
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

    // step sim if allowed by global/table sim cadence
    if (sim) {
        GP_TableContext* root_tbl = canvas_ensure_root_table(ctx);
        if (root_tbl && gp_table_should_step_sim(root_tbl)) {
            rope_sim_step(sim, 1.0f/60.0f, ctx->sim_maxforce, ctx->sim_iters, ctx->sim_damping);
        }
    }

    // Second pass: draw module backgrounds (raytrace/callback) and blit tables.
    for (int mi = 0; mi < static_cast<int>(ctx->modules.size()); ++mi) {
        const auto &m = ctx->modules[mi];
        int sx = m.x - ctx->offset_x;
        int sy = m.y - ctx->offset_y;
        std::vector<InputRayLight> inputs;
        bool is_stage = (mi >= 0 && mi < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[mi]);
        Color bgc{40,40,50,255};
        if (!is_stage) {
            memset_rect(out_rgba, w, h, pitch, sx, sy, m.w, m.h, bgc);
        }
        ModuleLayout layout = module_layout_for(ctx, mi, m);
        if (mi >= 0 && mi < static_cast<int>(ctx->module_bg.size())) {
            auto &bg = ctx->module_bg[mi];
            if (bg.cb) {
                if (is_stage) {
                    ensure_module_bg_storage(bg, m.w, m.h, /*oversample=*/1);
                    bg.cb(bg.user, mi, m.w, m.h, bg.scratch.data(), m.w * 4);
                } else {
                    int preview_h = layout.preview_h;
                    if (preview_h > 0) {
                        ensure_module_bg_storage(bg, m.w, preview_h, /*oversample=*/1);
                        bg.cb(bg.user, mi, m.w, preview_h, bg.scratch.data(), m.w * 4);
                        blit_module_buffer(out_rgba, w, h, pitch, sx, sy + layout.preview_y, m.w, preview_h, bg.scratch);
                    }
                }
                if (is_stage) {
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
                            if (img.rgba) {
                                size_t bytes = static_cast<size_t>(m.w) * static_cast<size_t>(m.h) * 4u;
                                if (module_tables[mi].size() >= bytes) {
                                    std::memcpy(module_tables[mi].data(), img.rgba, bytes);
                                }
                            }
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
            } else if (bg.mode == 1 && !is_stage) {
                int preview_h = layout.preview_h;
                if (preview_h > 0) {
                    render_module_raytrace_bg(bg, m.w, preview_h, inputs);
                    blit_module_buffer(out_rgba, w, h, pitch, sx, sy + layout.preview_y, m.w, preview_h, bg.scratch);
                }
            } else if (bg.mode == 1) {
                render_module_raytrace_bg(bg, m.w, m.h, inputs);
                blit_module_buffer(out_rgba, w, h, pitch, sx, sy, m.w, m.h, bg.scratch);
            }
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
            } else {
                blit_module_buffer_alpha(out_rgba, w, h, pitch, sx, sy, m.w, m.h, module_tables[mi], table_alpha);
            }
        }
        draw_module_top_ui(ctx, mi, m, out_rgba, w, h, pitch);
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

        // Draw module preview on top of everything in the module frame area
        if (!is_stage && layout.preview_h > 0 && mi >= 0 && mi < static_cast<int>(ctx->module_preview_buffers.size())) {
            ModulePreviewBuffer &preview = ctx->module_preview_buffers[mi];
            int preview_w = std::max(1, m.w);
            int preview_h = layout.preview_h;
            int preview_pitch = preview_w * 4;
            if (preview.width_px != preview_w || preview.height_px != preview_h || preview.pitch_bytes != preview_pitch) {
                preview.width_px = preview_w;
                preview.height_px = preview_h;
                preview.pitch_bytes = preview_pitch;
                preview.rgba.assign(static_cast<size_t>(preview_w) * static_cast<size_t>(preview_h) * 4u, 0);
            }
            if (preview.hitboxes.size() < 1024) preview.hitboxes.resize(1024);
            GP_ModulePreviewInput input{};
            build_module_preview_input(ctx, mi, input);
            GP_ModulePreviewOutput output{};
            output.rgba = preview.rgba.data();
            output.width_px = preview.width_px;
            output.height_px = preview.height_px;
            output.pitch_bytes = preview.pitch_bytes;
            output.hitboxes = preview.hitboxes.data();
            output.hitbox_capacity = static_cast<int32_t>(preview.hitboxes.size());
            output.hitbox_count = 0;
            gp_module_preview_build(&input, &output);
            if (!preview.rgba.empty()) {
                blit_module_buffer_srcalpha(out_rgba, w, h, pitch, sx, sy + layout.preview_y, preview_w, preview_h, preview.rgba);
            }
        }
    }

    // Render custom overlays (behind ropes): simple translucent rectangles
    // defined in world coords and transformed into canvas pixels.
    if (ctx->overlays.size() > 0) {
        for (auto &kv : ctx->overlays) {
            auto &ov = kv.second;
            // clear any previously cached control rect; renderer will set when drawing
            ov.ctrl_x = -1; ov.ctrl_y = -1; ov.ctrl_w = 0; ov.ctrl_h = 0;
            int rx0 = static_cast<int>(std::floor(std::min(ov.x1, ov.x2) - static_cast<float>(ctx->offset_x)));
            int ry0 = static_cast<int>(std::floor(std::min(ov.y1, ov.y2) - static_cast<float>(ctx->offset_y)));
            int rx1 = static_cast<int>(std::ceil(std::max(ov.x1, ov.x2) - static_cast<float>(ctx->offset_x)));
            int ry1 = static_cast<int>(std::ceil(std::max(ov.y1, ov.y2) - static_cast<float>(ctx->offset_y)));
            int rw = rx1 - rx0; int rh = ry1 - ry0;
            if (rw > 0 && rh > 0) {
                Color bg{24, 28, 32, 160};
                memset_rect(out_rgba, w, h, pitch, rx0, ry0, rw, rh, bg);
                // draw anchor LEDs at overlay endpoints
                int led_ax = static_cast<int>(std::lround(ov.x1 - static_cast<float>(ctx->offset_x)));
                int led_ay = static_cast<int>(std::lround(ov.y1 - static_cast<float>(ctx->offset_y)));
                int led_bx = static_cast<int>(std::lround(ov.x2 - static_cast<float>(ctx->offset_x)));
                int led_by = static_cast<int>(std::lround(ov.y2 - static_cast<float>(ctx->offset_y)));
                draw_circle(out_rgba, w, h, pitch, led_ax, led_ay, 6, Color{220,220,200,220});
                draw_circle(out_rgba, w, h, pitch, led_bx, led_by, 6, Color{220,220,200,220});
                draw_circle(out_rgba, w, h, pitch, led_ax, led_ay, 3, Color{40,40,48,220});
                draw_circle(out_rgba, w, h, pitch, led_bx, led_by, 3, Color{40,40,48,220});
                // (removed decorative LED placeholders) rely on overlay endpoints drawn above
                // draw small mode-toggle button in top-right of overlay (uses table API)
                int btn_w = 20; int btn_h = 18; int margin = 6;
                // compute button position in world coords (ov.* are world coords)
                float btn_world_x = std::max(ov.x1, ov.x2) - static_cast<float>(margin) - static_cast<float>(btn_w);
                float btn_world_y = std::min(ov.y1, ov.y2) + static_cast<float>(margin);
                int bx = static_cast<int>(std::lround(btn_world_x - static_cast<float>(ctx->offset_x)));
                int by = static_cast<int>(std::lround(btn_world_y - static_cast<float>(ctx->offset_y)));
                // determine associated rope index (if any) by scanning edges
                int overlay_rope_idx = -1;
                for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
                    const auto &e = ctx->edges[ei];
                    if (e.overlay_key_a == ov.key_a || e.overlay_key_b == ov.key_b) { overlay_rope_idx = e.rope_idx; break; }
                }
                // precompute a default control rect in screen coords so input can
                // hit-test it even if we don't successfully look up the widget
                // position in the table. Always cache the default rect based on
                // the overlay button position so hit-testing is consistent.
                {
                    int ctrl_w_def = 88; int ctrl_h_def = 18;
                    int ctrl_x_def = bx + (btn_w/2) - (ctrl_w_def/2);
                    int ctrl_y_def = by + btn_h + 6;
                    ov.ctrl_x = ctrl_x_def; ov.ctrl_y = ctrl_y_def; ov.ctrl_w = ctrl_w_def; ov.ctrl_h = ctrl_h_def;
                }
                // If this overlay is attached to a dangling-widget rope for a meta-group,
                // draw the numeric control at the widget position so it appears where
                // the existing binding/mode button is located.
                if (overlay_rope_idx >= 0) {
                    int display_ch = 0;
                    float wpos[3] = {0.0f,0.0f,0.0f};
                    bool drew = false;
                    auto try_draw_for_table = [&](GP_TableContext* t)->bool {
                        if (!t) return false;
                        int mgcount = gp_table_get_meta_group_count(t);
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            if (!mg) continue;
                            int dr = -1, dv = -1;
                            gp_table_meta_get_dangling_rope_info(t, mg, &dr, &dv);
                            int ch_tmp = 0; gp_table_meta_get_channel_group(t, mg, &ch_tmp);
                            (void)ch_tmp;
                            if (dr != overlay_rope_idx) continue;
                            // found a meta-group associated with this overlay rope
                            gp_table_meta_get_channel_group(t, mg, &display_ch);
                            int wwid = -1;
                            gp_table_meta_get_dangling_widget_id(t, mg, &wwid);
                            if (wwid >= 0 && gp_table_get_widget_position(t, wwid, wpos)) {
                                int wcx = static_cast<int>(std::lround(wpos[0] - static_cast<float>(ctx->offset_x)));
                                int wcy = static_cast<int>(std::lround(wpos[1] - static_cast<float>(ctx->offset_y)));
                                int ctrl_w = 88; int ctrl_h = 18;
                                // place numeric control centered below the mode/button
                                int ctrl_x = bx + (btn_w/2) - (ctrl_w/2);
                                int ctrl_y = by + btn_h + 6;
                                // cache control rect in overlay (screen/view coords)
                                ov.ctrl_x = ctrl_x; ov.ctrl_y = ctrl_y; ov.ctrl_w = ctrl_w; ov.ctrl_h = ctrl_h;
                                (void)wwid; (void)wcx; (void)wcy;
                                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, ctrl_h, Color{40,40,50,220});
                                // outline (1px) in bright green so numeric control is obvious
                                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, 1, Color{80,200,120,220});
                                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y + ctrl_h - 1, ctrl_w, 1, Color{80,200,120,220});
                                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, 1, ctrl_h, Color{80,200,120,220});
                                memset_rect(out_rgba, w, h, pitch, ctrl_x + ctrl_w - 1, ctrl_y, 1, ctrl_h, Color{80,200,120,220});
                                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y + ctrl_h - 1, ctrl_w, 1, Color{16,16,20,255});
                                int btn_w = ctrl_h; int gap = 6;
                                int bx_minus = ctrl_x + 2;
                                int bx_num = bx_minus + btn_w + gap;
                                int num_w = ctrl_w - (btn_w*2 + gap*2) - 4;
                                int bx_plus = bx_num + num_w + gap;
                                memset_rect(out_rgba, w, h, pitch, bx_minus, ctrl_y + 1, btn_w, ctrl_h - 2, Color{60,60,70,255});
                                memset_rect(out_rgba, w, h, pitch, bx_num, ctrl_y + 1, num_w, ctrl_h - 2, Color{36,36,46,255});
                                memset_rect(out_rgba, w, h, pitch, bx_plus, ctrl_y + 1, btn_w, ctrl_h - 2, Color{60,60,70,255});
                                // render minus/plus/number labels
                                auto bm_minus = render_text_to_rgba("-", 1.0f, {220,220,220,255});
                                if (!bm_minus.pixels.empty()) {
                                    int tx = bx_minus + (btn_w - bm_minus.width) / 2;
                                    int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_minus.height) / 2;
                                    for (int yy = 0; yy < bm_minus.height; ++yy) {
                                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                                        for (int xx = 0; xx < bm_minus.width; ++xx) {
                                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                            const unsigned char* src = &bm_minus.pixels[(yy * bm_minus.width + xx) * 4];
                                            float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                            else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                                        }
                                    }
                                }
                                auto bm_plus = render_text_to_rgba("+", 1.0f, {220,220,220,255});
                                if (!bm_plus.pixels.empty()) {
                                    int tx = bx_plus + (btn_w - bm_plus.width) / 2;
                                    int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_plus.height) / 2;
                                    for (int yy = 0; yy < bm_plus.height; ++yy) {
                                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                                        for (int xx = 0; xx < bm_plus.width; ++xx) {
                                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                            const unsigned char* src = &bm_plus.pixels[(yy * bm_plus.width + xx) * 4];
                                            float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                            else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                                        }
                                    }
                                }
                                std::string s = std::to_string(display_ch);
                                auto bm_num = render_text_to_rgba(s, 0.9f, {230,230,235,255});
                                if (!bm_num.pixels.empty()) {
                                    int tx = bx_num + (num_w - bm_num.width) / 2;
                                    int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_num.height) / 2;
                                    for (int yy = 0; yy < bm_num.height; ++yy) {
                                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                                        for (int xx = 0; xx < bm_num.width; ++xx) {
                                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                            const unsigned char* src = &bm_num.pixels[(yy * bm_num.width + xx) * 4];
                                            float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                            else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                                        }
                                    }
                                }
                                return true;
                            }
                        }
                        return false;
                    };
                    if (try_draw_for_table(ctx->container_table)) drew = true;
                    if (!drew) {
                        for (size_t mi = 0; mi < ctx->module_tables.size(); ++mi) {
                            if (try_draw_for_table(ctx->module_tables[mi])) { drew = true; break; }
                        }
                    }
                }
                  // Diagnostic: overlay rect/button logging removed to reduce runtime noise
                  // printf("canvas_render: overlay id=%d rect=[%d,%d,%d,%d] btn=[%d,%d,%d,%d] overlay_rope_idx=%d\n",
                  //        kv.first, rx0, ry0, rx1, ry1, bx, by, btn_w, btn_h, overlay_rope_idx);
                int mode = 0; // default
                if (overlay_rope_idx >= 0) {
                    auto check_table_for_mode = [&](GP_TableContext* t)->bool{
                        if (!t) return false;
                        int mgcount = gp_table_get_meta_group_count(t);
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            if (!mg) continue;
                            int ar = -1, av = -1;
                            if (!gp_table_meta_get_anchor(t, mg, &ar, &av)) continue;
                            if (ar == overlay_rope_idx) {
                                int32_t got = 0;
                                if (gp_table_meta_get_ring_mode(t, mg, &got)) mode = got;
                                return true;
                            }
                        }
                        return false;
                    };
                    if (!check_table_for_mode(ctx->container_table)) {
                        for (size_t mi = 0; mi < ctx->module_tables.size(); ++mi) {
                            if (check_table_for_mode(ctx->module_tables[mi])) break;
                        }
                    }
                }
                Color btn_col;
                const char* lbl = "R";
                if (mode == 0) { btn_col = Color{100,140,120,220}; lbl = "R"; }
                else if (mode == 1) { btn_col = Color{120,100,140,220}; lbl = "C"; }
                else { btn_col = Color{140,120,100,220}; lbl = "D"; }
                // Also draw the numeric channel-group control co-located with
                // the overlay's binding/mode button by resolving the dangling
                // widget for any meta-group anchored to this overlay rope.
                {
                    int display_ch = 0;
                    float wpos_local[3] = {0.0f,0.0f,0.0f};
                    bool have_wpos = false;
                    auto find_widget_pos = [&](GP_TableContext* t)->bool {
                        if (!t) return false;
                        int mgcount = gp_table_get_meta_group_count(t);
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            if (!mg) continue;
                            int ar = -1, av = -1;
                            if (!gp_table_meta_get_anchor(t, mg, &ar, &av)) continue;
                            if (ar != overlay_rope_idx) continue;
                            gp_table_meta_get_channel_group(t, mg, &display_ch);
                            int wwid = -1;
                            if (gp_table_meta_get_dangling_widget_id(t, mg, &wwid) && wwid >= 0) {
                                if (gp_table_get_widget_position(t, wwid, wpos_local)) { have_wpos = true; return true; }
                            }
                        }
                        return false;
                    };
                    if (!find_widget_pos(ctx->container_table)) {
                        for (size_t mi = 0; mi < ctx->module_tables.size(); ++mi) {
                            if (find_widget_pos(ctx->module_tables[mi])) break;
                        }
                    }
                    if (have_wpos) {
                        int wcx = static_cast<int>(std::lround(wpos_local[0] - static_cast<float>(ctx->offset_x)));
                        int wcy = static_cast<int>(std::lround(wpos_local[1] - static_cast<float>(ctx->offset_y)));
                        int ctrl_w = 88; int ctrl_h = 18;
                        // place numeric control centered below the mode/button
                        int ctrl_x = bx + (btn_w/2) - (ctrl_w/2);
                        int ctrl_y = by + btn_h + 6;
                        printf("drawnum_overlay_btn: overlay_rope_idx=%d bx=%d by=%d ctrl_w=%d ctrl_h=%d\n", overlay_rope_idx, bx, by, ctrl_w, ctrl_h);
                        fflush(stdout);
                        memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, ctrl_h, Color{40,40,50,220});
                        // outline (1px) in bright green so numeric control is obvious
                        memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, 1, Color{80,200,120,220});
                        memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y + ctrl_h - 1, ctrl_w, 1, Color{16,16,20,255});
                        memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, 1, ctrl_h, Color{80,200,120,220});
                        memset_rect(out_rgba, w, h, pitch, ctrl_x + ctrl_w - 1, ctrl_y, 1, ctrl_h, Color{80,200,120,220});
                        int btn_w = ctrl_h; int gap = 6;
                        int bx_minus = ctrl_x + 2;
                        int bx_num = bx_minus + btn_w + gap;
                        int num_w = ctrl_w - (btn_w*2 + gap*2) - 4;
                        int bx_plus = bx_num + num_w + gap;
                        memset_rect(out_rgba, w, h, pitch, bx_minus, ctrl_y + 1, btn_w, ctrl_h - 2, Color{60,60,70,255});
                        memset_rect(out_rgba, w, h, pitch, bx_num, ctrl_y + 1, num_w, ctrl_h - 2, Color{36,36,46,255});
                        memset_rect(out_rgba, w, h, pitch, bx_plus, ctrl_y + 1, btn_w, ctrl_h - 2, Color{60,60,70,255});
                        auto bm_minus = render_text_to_rgba("-", 1.0f, {220,220,220,255});
                        if (!bm_minus.pixels.empty()) {
                            int tx = bx_minus + (btn_w - bm_minus.width) / 2;
                            int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_minus.height) / 2;
                            for (int yy = 0; yy < bm_minus.height; ++yy) {
                                int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                                for (int xx = 0; xx < bm_minus.width; ++xx) {
                                    int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                    const unsigned char* src = &bm_minus.pixels[(yy * bm_minus.width + xx) * 4];
                                    float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                    else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                                }
                            }
                        }
                        auto bm_plus = render_text_to_rgba("+", 1.0f, {220,220,220,255});
                        if (!bm_plus.pixels.empty()) {
                            int tx = bx_plus + (btn_w - bm_plus.width) / 2;
                            int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_plus.height) / 2;
                            for (int yy = 0; yy < bm_plus.height; ++yy) {
                                int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                                for (int xx = 0; xx < bm_plus.width; ++xx) {
                                    int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                    const unsigned char* src = &bm_plus.pixels[(yy * bm_plus.width + xx) * 4];
                                    float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                    else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                                }
                            }
                        }
                        std::string s = std::to_string(display_ch);
                        auto bm_num = render_text_to_rgba(s, 0.9f, {230,230,235,255});
                        if (!bm_num.pixels.empty()) {
                            int tx = bx_num + (num_w - bm_num.width) / 2;
                            int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_num.height) / 2;
                            for (int yy = 0; yy < bm_num.height; ++yy) {
                                int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                                for (int xx = 0; xx < bm_num.width; ++xx) {
                                    int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                    const unsigned char* src = &bm_num.pixels[(yy * bm_num.width + xx) * 4];
                                    float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                    else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                                }
                            }
                        }
                    }
                }
                // TEMP VISUAL MARKER: draw a translucent red box slightly
                // larger than the mode button so it's obvious if this path
                // is executed. Remove after verification.
                memset_rect(out_rgba, w, h, pitch, bx - 4, by - 4, btn_w + 8, btn_h + 8, Color{220,40,40,140});
                memset_rect(out_rgba, w, h, pitch, bx, by, btn_w, btn_h, btn_col);
                // Force-draw numeric control centered below the mode/button (always)
                {
                    int ctrl_w = 88; int ctrl_h = 18;
                    // center horizontally on the button, place below with padding
                    int ctrl_x = bx + (btn_w/2) - (ctrl_w/2);
                    int ctrl_y = by + btn_h + 6;
                    int display_ch = 0;
                    // Prefer to find a meta-group associated with this overlay rope
                    // by checking dangling-rope info first (widget-attached), then
                    // falling back to anchor/members. This ensures the same meta-group
                    // is used for both hit-testing (which prefers widget/dangling info)
                    // and rendering so the displayed number updates immediately.
                    auto find_ch = [&](GP_TableContext* t)->bool {
                        if (!t) return false;
                        // If the overlay has an authoritative binding to a table/meta-group,
                        // prefer that immediately (this avoids rescanning ropes and ensures
                        // the displayed number matches direct overlay bindings).
                        if (ov.meta_table && ov.meta_mg) {
                            if (ov.meta_table == t) {
                                gp_table_meta_get_channel_group(t, ov.meta_mg, &display_ch);
                                return true;
                            }
                        }
                        int mgcount = gp_table_get_meta_group_count(t);
                        // first pass: match dangling-rope info
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            if (!mg) continue;
                            int dr=-1,dv=-1; gp_table_meta_get_dangling_rope_info(t, mg, &dr, &dv);
                            if (dr == overlay_rope_idx) { gp_table_meta_get_channel_group(t, mg, &display_ch); return true; }
                        }
                        // second pass: match anchor
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            if (!mg) continue;
                            int ar=-1,av=-1; if (!gp_table_meta_get_anchor(t, mg, &ar, &av)) continue;
                            if (ar != overlay_rope_idx) continue;
                            gp_table_meta_get_channel_group(t, mg, &display_ch);
                            return true;
                        }
                        return false;
                    };
                    if (!find_ch(ctx->container_table)) for (size_t mi=0; mi<ctx->module_tables.size(); ++mi) if (find_ch(ctx->module_tables[mi])) break;
                    memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, ctrl_h, Color{24,36,44,220});
                    // cyan outline
                    memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, 1, Color{80,220,220,220});
                    memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y + ctrl_h - 1, ctrl_w, 1, Color{16,16,20,255});
                    memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, 1, ctrl_h, Color{80,220,220,220});
                    memset_rect(out_rgba, w, h, pitch, ctrl_x + ctrl_w - 1, ctrl_y, 1, ctrl_h, Color{80,220,220,220});
                    int btn_w2 = ctrl_h; int gap = 6;
                    int bx_minus = ctrl_x + 2;
                    int bx_num = bx_minus + btn_w2 + gap;
                    int num_w = ctrl_w - (btn_w2*2 + gap*2) - 4;
                    int bx_plus = bx_num + num_w + gap;
                    memset_rect(out_rgba, w, h, pitch, bx_minus, ctrl_y + 1, btn_w2, ctrl_h - 2, Color{60,60,70,255});
                    memset_rect(out_rgba, w, h, pitch, bx_num, ctrl_y + 1, num_w, ctrl_h - 2, Color{36,36,46,255});
                    memset_rect(out_rgba, w, h, pitch, bx_plus, ctrl_y + 1, btn_w2, ctrl_h - 2, Color{60,60,70,255});
                    std::string s = std::to_string(display_ch);
                    auto bm_num = render_text_to_rgba(s, 0.9f, {230,230,235,255});
                    if (!bm_num.pixels.empty()) {
                        int tx = bx_num + (num_w - bm_num.width) / 2;
                        int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_num.height) / 2;
                        for (int yy = 0; yy < bm_num.height; ++yy) {
                            int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                            for (int xx = 0; xx < bm_num.width; ++xx) {
                                int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                const unsigned char* src = &bm_num.pixels[(yy * bm_num.width + xx) * 4];
                                float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                            }
                        }
                    }
                }
                auto tb = render_text_to_rgba(std::string(lbl), 1.1f, {240,240,240,255});
                if (!tb.pixels.empty()) {
                    int tx = bx + (btn_w - tb.width) / 2;
                    int ty = by + (btn_h - tb.height) / 2;
                    for (int yy = 0; yy < tb.height; ++yy) {
                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < tb.width; ++xx) {
                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
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
            uint32_t subgroup_flags = ctx->edges[ei].subgroup_flags;
            if (subgroup_flags != 0u) {
                float hue = subgroup_flags_to_hue(ctx, subgroup_flags);
                float hue_vals[1] = { hue };
                table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, verts_view.data(), got, jacket_px, jacket_border, hue_vals, 1, 3, 0.65f);
            } else {
                table_draw_rope_curve_blend(out_rgba, w, h, pitch, verts_view.data(), got, jacket_px, jacket_border, 200, 200, 200, 180, 3);
            }
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
                int samples_per_segment = 3;
                if (ctx->selected_tool_subgroup_flags != 0u) {
                    uint32_t flags = ctx->selected_tool_subgroup_flags;
                    float hue = subgroup_flags_to_hue(ctx, flags);
                    float hue_vals[1] = { hue };
                    table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, verts_view.data(), got, jacket_px, jacket_border, hue_vals, 1, samples_per_segment, 0.55f);
                } else {
                    table_draw_rope_curve_blend(out_rgba, w, h, pitch, verts_view.data(), got, jacket_px, jacket_border, 200, 200, 200, 180, samples_per_segment);
                }
            }
        }
    }

    // Render ring entries (short sampled segments around ring u) from all known tables.
    // Also collect ring positions grouped by ring_key and draw continuous splines
    // connecting groups of rings (uses the same spline drawer as rope rendering).
    {
        std::vector<GP_TableContext*> draw_tables;
        if (ctx->container_table) draw_tables.push_back(ctx->container_table);
        for (GP_TableContext* mt : ctx->module_tables) if (mt) draw_tables.push_back(mt);

        // collect per-key ring world positions (canvas coords)
        std::unordered_map<unsigned long long, std::vector<std::pair<float,float>>> ring_groups;

        for (GP_TableContext* rt : draw_tables) {
            if (!rt) continue;
            RopeSim* tsim = gp_table_get_rope_sim(rt);
            if (!tsim) continue;
            int ring_count = gp_table_get_ring_edge_count(rt);
            for (int rei = 0; rei < ring_count; ++rei) {
                int ring_id = -1; unsigned long long ring_key = 0ull;
                if (!gp_table_get_ring_edge(rt, rei, &ring_id, &ring_key)) continue;
                if (ring_id < 0) continue;
                int parent_ridx = -1;
                if (!rope_sim_get_ring_rope_index(tsim, ring_id, &parent_ridx) || parent_ridx < 0) continue;
                int vc = rope_sim_get_vertex_count(tsim, parent_ridx);
                if (vc < 2) continue;
                std::vector<float> verts(static_cast<size_t>(vc) * 2);
                int got = rope_sim_get_vertices(tsim, parent_ridx, verts.data(), static_cast<int>(verts.size()));
                if (got <= 0) continue;
                float u = 0.0f;
                rope_sim_get_ring_u(tsim, ring_id, &u);

                // compute a central world point for this ring (linear interp using u)
                float idxf = u * static_cast<float>(got - 1);
                int lo = static_cast<int>(std::floor(idxf));
                int hi = std::min(got - 1, lo + 1);
                float frac = idxf - static_cast<float>(lo);
                float wx = (1.0f - frac) * verts[2*lo+0] + frac * verts[2*hi+0];
                float wy = (1.0f - frac) * verts[2*lo+1] + frac * verts[2*hi+1];
                // transform to canvas coords and collect
                float cx = wx - static_cast<float>(ctx->offset_x);
                float cy = wy - static_cast<float>(ctx->offset_y);
                ring_groups[ring_key].emplace_back(cx, cy);

                // draw a short sampled segment centered at u (as before)
                int center_idx = static_cast<int>(idxf + 0.5f);
                int half = 2;
                int start = std::max(0, center_idx - half);
                int end = std::min(got - 1, center_idx + half);
                int use_count = end - start + 1;
                if (use_count < 2) continue;
                std::vector<float> verts_view(static_cast<size_t>(use_count) * 2);
                for (int vi = start; vi <= end; ++vi) {
                    int idx = vi - start;
                    verts_view[idx * 2 + 0] = verts[vi * 2 + 0] - static_cast<float>(ctx->offset_x);
                    verts_view[idx * 2 + 1] = verts[vi * 2 + 1] - static_cast<float>(ctx->offset_y);
                }
                int jacket_px = ctx->jacket_px;
                int jacket_border = ctx->jacket_border;
                int samples_per_segment = 3;
                uint32_t flags = 0;
                gp_table_ring_get_subgroup_flags(rt, rei, &flags);
                if (flags != 0u) {
                    float hue = subgroup_flags_to_hue(ctx, flags);
                    float hue_vals[1] = { hue };
                    table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, verts_view.data(), use_count, jacket_px, jacket_border, hue_vals, 1, samples_per_segment, 0.65f);
                } else {
                    table_draw_rope_curve_blend(out_rgba, w, h, pitch, verts_view.data(), use_count, jacket_px, jacket_border, 220, 160, 80, 220, samples_per_segment);
                }
            }
        }

        // Now draw connecting splines for each group of rings sharing the same key
        for (auto &kv : ring_groups) {
            auto &pts = kv.second;
            if (pts.size() < 2) continue;
            // build simple polyline in collected order
            std::vector<float> poly(static_cast<size_t>(pts.size() * 2));
            for (size_t i = 0; i < pts.size(); ++i) { poly[2*i+0] = pts[i].first; poly[2*i+1] = pts[i].second; }
            int jacket_px = ctx->jacket_px;
            int jacket_border = ctx->jacket_border;
            // neutral colored connecting rope
            table_draw_rope_curve_blend(out_rgba, w, h, pitch, poly.data(), static_cast<int>(pts.size()), jacket_px, jacket_border, 180, 140, 100, 220, 3);
        }
    }

    // Render meta-group edges (treat meta edges as rope-like splines connecting stored vertices)
    {
        std::vector<GP_TableContext*> draw_tables;
        if (ctx->container_table) draw_tables.push_back(ctx->container_table);
        for (GP_TableContext* mt : ctx->module_tables) if (mt) draw_tables.push_back(mt);
        for (GP_TableContext* rt : draw_tables) {
            if (!rt) continue;
            RopeSim* tsim = gp_table_get_rope_sim(rt);
            int mg_count = gp_table_get_meta_group_count(rt);
            for (int mgi = 0; mgi < mg_count; ++mgi) {
                GP_MetaGroup* mg = gp_table_get_meta_group(rt, mgi);
                if (!mg) continue;
                int dr=-1,dv=-1; gp_table_meta_get_dangling_rope_info(rt, mg, &dr, &dv);
                int ch_tmp = 0; gp_table_meta_get_channel_group(rt, mg, &ch_tmp);
                (void)ch_tmp;
                int vcount = gp_table_meta_get_vertex_count(rt, mg);
                if (vcount <= 0) continue;
                // For each vertex, prefer sim-sampled world position when available,
                // otherwise fall back to the table-projected vertex index lookup.
                std::vector<float> poly(static_cast<size_t>(vcount * 2));
                bool got_all = true;
                for (int vi = 0; vi < vcount; ++vi) {
                    float wx = 0.0f, wy = 0.0f;
                    bool used_sim = false;
                    if (tsim) {
                        int32_t sim_idx = -1;
                        if (gp_table_meta_get_sim_group_index(rt, mg, &sim_idx) && sim_idx >= 0) {
                            float wpos[3];
                            if (rope_sim_meta_group_get_member_world_pos(tsim, sim_idx, vi, wpos)) {
                                wx = wpos[0]; wy = wpos[1]; used_sim = true;
                            }
                        }
                    }
                    if (!used_sim) {
                        int rope_idx = -1, vert_idx = -1;
                        if (!gp_table_meta_get_vertex(rt, mg, vi, &rope_idx, &vert_idx)) { got_all = false; break; }
                        if (rope_idx < 0 || vert_idx < 0) { got_all = false; break; }
                        // get projected vertices of that rope from the table (table coordinates)
                        int maxv = 256;
                        std::vector<float> proj(static_cast<size_t>(maxv * 2));
                        int got = gp_table_get_projected_rope_vertices(rt, rope_idx, proj.data(), static_cast<int>(proj.size()));
                        if (got <= vert_idx) { got_all = false; break; }
                        wx = proj[vert_idx * 2 + 0];
                        wy = proj[vert_idx * 2 + 1];
                    }
                    // convert to canvas coords
                    poly[vi * 2 + 0] = wx - static_cast<float>(ctx->offset_x);
                    poly[vi * 2 + 1] = wy - static_cast<float>(ctx->offset_y);
                }
                if (!got_all) continue;
                int jacket_px = ctx->jacket_px;
                int jacket_border = ctx->jacket_border;
                // draw meta-group connector spline using subgroup color when present
                uint32_t mg_flags = 0u;
                gp_table_meta_get_subgroup_flags(rt, mg, &mg_flags);
                if (mg_flags != 0u) {
                    float hue = subgroup_flags_to_hue(ctx, mg_flags);
                    float hue_vals[1] = { hue };
                    table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, poly.data(), vcount, jacket_px, jacket_border, hue_vals, 1, 3, 0.6f);
                } else {
                    table_draw_rope_curve_blend(out_rgba, w, h, pitch, poly.data(), vcount, jacket_px, jacket_border, 160, 200, 210, 200, 3);
                }

                // Debug overlay: draw a small blob at each stored vertex and always log mapping
                // when environment variable NODUS_DEBUG_META is set. Bright red marks indicate
                // failed projected lookups.
                if (getenv("NODUS_DEBUG_META") != nullptr) {
                    bool any_bad = false;
                    std::string mapstr;
                    for (int vi = 0; vi < vcount; ++vi) {
                        int rope_idx = -1, vert_idx = -1;
                        gp_table_meta_get_vertex(rt, mg, vi, &rope_idx, &vert_idx);
                        int cx = static_cast<int>(poly[vi * 2 + 0] + 0.5f);
                        int cy = static_cast<int>(poly[vi * 2 + 1] + 0.5f);
                        Color bc{ static_cast<uint8_t>((rope_idx * 97) & 0xFF), static_cast<uint8_t>((rope_idx * 223) & 0xFF), static_cast<uint8_t>((vert_idx * 61) & 0xFF), 220 };
                        draw_blob_blend(out_rgba, w, h, pitch, cx, cy, std::max(2, jacket_px / 2), bc);

                        int maxv_dbg = 256;
                        std::vector<float> proj_dbg(static_cast<size_t>(maxv_dbg * 2));
                        int got_dbg = gp_table_get_projected_rope_vertices(rt, rope_idx, proj_dbg.data(), static_cast<int>(proj_dbg.size()));
                        if (got_dbg <= 0 || vert_idx < 0 || vert_idx >= got_dbg) {
                            any_bad = true;
                            Color bad{255,40,40,200};
                            draw_blob_blend(out_rgba, w, h, pitch, cx, cy, std::max(3, jacket_px), bad);
                            (void)got_dbg; (void)vi; (void)rope_idx; (void)vert_idx;
                        }

                        char buf[64]; std::snprintf(buf, sizeof(buf), "%d:%d", rope_idx, vert_idx);
                        if (!mapstr.empty()) mapstr += ",";
                        mapstr += buf;
                    }
                    (void)mapstr; (void)any_bad;
                }
            }
        }
    }

    // Render dangling widgets (blended blob + short sampled rope segment)
    {
        std::vector<GP_TableContext*> draw_tables;
        if (ctx->container_table) draw_tables.push_back(ctx->container_table);
        for (GP_TableContext* mt : ctx->module_tables) if (mt) draw_tables.push_back(mt);
        for (GP_TableContext* rt : draw_tables) {
            if (!rt) continue;
            RopeSim* tsim = gp_table_get_rope_sim(rt);
            if (!tsim) continue;
            int mg_count = gp_table_get_meta_group_count(rt);
            for (int mgi = 0; mgi < mg_count; ++mgi) {
                GP_MetaGroup* mg = gp_table_get_meta_group(rt, mgi);
                if (!mg) continue;
                int wid = -1;
                if (!gp_table_meta_get_dangling_widget_id(rt, mg, &wid)) continue;
                if (wid < 0) continue;
                // get widget world pos
                float wpos[3] = {0.0f,0.0f,0.0f};
                if (!gp_table_get_widget_position(rt, wid, wpos)) continue;
                int cx = static_cast<int>(std::lround(wpos[0] - static_cast<float>(ctx->offset_x)));
                int cy = static_cast<int>(std::lround(wpos[1] - static_cast<float>(ctx->offset_y)));
                // draw blob
                draw_blob_blend(out_rgba, w, h, pitch, cx, cy, 6, Color{200,60,60,200});

                // Draw a small numeric control above the widget to set channel group.
                // Layout: [ - ] [  NUM  ] [ + ] centered horizontally above the blob.
                int ctrl_w = 88;
                int ctrl_h = 18;
                int ctrl_x = cx - ctrl_w / 2;
                int ctrl_y = cy + 12; // place below the blob
                // background
                // debug: drawing meta control logging removed
                // printf("canvas_render: drawing meta control mg=%p wid=%d cx=%d cy=%d ctrl_x=%d ctrl_y=%d ctrl_w=%d ctrl_h=%d\n",
                //     (void*)mg, wid, cx, cy, ctrl_x, ctrl_y, ctrl_w, ctrl_h);
                // fflush(stdout);
                // Remove per-table dangling draw to avoid duplication; use the
                // unconditional overlay-forced draw so rendering & hit-tests
                // are driven from the same code path and update reliably.
                // (leave the debug print for diagnostics)
                (void)mg; (void)wid; (void)cx; (void)cy; (void)ctrl_w; (void)ctrl_h;
                // memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, ctrl_h, Color{40,40,50,220});
                // outline (1px) in bright green so numeric control is obvious
                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, 1, Color{80,200,120,220});
                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y + ctrl_h - 1, ctrl_w, 1, Color{16,16,20,255});
                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, 1, ctrl_h, Color{80,200,120,220});
                memset_rect(out_rgba, w, h, pitch, ctrl_x + ctrl_w - 1, ctrl_y, 1, ctrl_h, Color{80,200,120,220});
                int btn_w = ctrl_h; int gap = 6;
                int bx_minus = ctrl_x + 2;
                int bx_num = bx_minus + btn_w + gap;
                int num_w = ctrl_w - (btn_w*2 + gap*2) - 4;
                int bx_plus = bx_num + num_w + gap;
                // Diagnostic: print control layout once per frame for visible widgets
                (void)mg; (void)wid; (void)cx; (void)cy; (void)ctrl_x; (void)ctrl_y; (void)num_w;
                // minus button
                // debug: meta control buttons logging removed
                // printf("canvas_render: meta control buttons mg=%p minus=[%d,%d,%d,%d] num=[%d,%d,%d,%d] plus=[%d,%d,%d,%d]\n",
                //     (void*)mg, bx_minus, ctrl_y + 1, btn_w, ctrl_h - 2, bx_num, ctrl_y + 1, num_w, ctrl_h - 2, bx_plus, ctrl_y + 1, btn_w, ctrl_h - 2);
                // fflush(stdout);
                memset_rect(out_rgba, w, h, pitch, bx_minus, ctrl_y + 1, btn_w, ctrl_h - 2, Color{60,60,70,255});
                // number background
                memset_rect(out_rgba, w, h, pitch, bx_num, ctrl_y + 1, num_w, ctrl_h - 2, Color{36,36,46,255});
                // plus button
                memset_rect(out_rgba, w, h, pitch, bx_plus, ctrl_y + 1, btn_w, ctrl_h - 2, Color{60,60,70,255});
                // render labels
                int ch_val = 0; gp_table_meta_get_channel_group(rt, mg, &ch_val);
                auto bm_minus = render_text_to_rgba("-", 1.0f, {220,220,220,255});
                if (!bm_minus.pixels.empty()) {
                    int tx = bx_minus + (btn_w - bm_minus.width) / 2;
                    int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_minus.height) / 2;
                    for (int yy = 0; yy < bm_minus.height; ++yy) {
                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < bm_minus.width; ++xx) {
                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &bm_minus.pixels[(yy * bm_minus.width + xx) * 4];
                            float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                        }
                    }
                }
                auto bm_plus = render_text_to_rgba("+", 1.0f, {220,220,220,255});
                if (!bm_plus.pixels.empty()) {
                    int tx = bx_plus + (btn_w - bm_plus.width) / 2;
                    int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_plus.height) / 2;
                    for (int yy = 0; yy < bm_plus.height; ++yy) {
                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < bm_plus.width; ++xx) {
                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &bm_plus.pixels[(yy * bm_plus.width + xx) * 4];
                            float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                        }
                    }
                }
                // number
                std::string s = std::to_string(ch_val);
                auto bm_num = render_text_to_rgba(s, 0.9f, {230,230,235,255});
                if (!bm_num.pixels.empty()) {
                    int tx = bx_num + (num_w - bm_num.width) / 2;
                    int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_num.height) / 2;
                    for (int yy = 0; yy < bm_num.height; ++yy) {
                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < bm_num.width; ++xx) {
                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &bm_num.pixels[(yy * bm_num.width + xx) * 4];
                            float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                        }
                    }
                }

                // If the meta-group created a short dangling rope for the widget,
                // draw that rope so the widget appears connected. Prefer the
                // meta-group's subgroup flags for coloring.
                int wrope = -1, wvid = -1;
                gp_table_meta_get_dangling_rope_info(rt, mg, &wrope, &wvid);
                if (wrope >= 0) {
                    int maxv_w = 512;
                    std::vector<float> wproj(static_cast<size_t>(maxv_w * 2));
                    int wgot = gp_table_get_projected_rope_vertices(rt, wrope, wproj.data(), static_cast<int>(wproj.size()));
                    if (wgot > 1) {
                        std::vector<float> wseg(static_cast<size_t>(wgot * 2));
                        for (int wi = 0; wi < wgot; ++wi) { wseg[wi*2+0] = wproj[wi*2+0] - static_cast<float>(ctx->offset_x); wseg[wi*2+1] = wproj[wi*2+1] - static_cast<float>(ctx->offset_y); }
                        int jacket_px_w = ctx->jacket_px;
                        int jacket_border_w = ctx->jacket_border;
                        uint32_t mf = 0u;
                        gp_table_meta_get_subgroup_flags(rt, mg, &mf);
                        if (mf != 0u) {
                            float hue = subgroup_flags_to_hue(ctx, mf);
                            float hue_vals[1] = { hue };
                            table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, wseg.data(), wgot, jacket_px_w, jacket_border_w, hue_vals, 1, 3, 0.6f);
                        } else {
                            table_draw_rope_curve_blend(out_rgba, w, h, pitch, wseg.data(), wgot, jacket_px_w, jacket_border_w, 160, 200, 210, 200, 3);
                        }
                    }
                }

                // draw short sampled rope segment around anchor vertex (first vertex)
                if (gp_table_meta_get_vertex_count(rt, mg) > 0) {
                    int rope_idx = -1, vert_idx = -1;
                    if (gp_table_meta_get_vertex(rt, mg, 0, &rope_idx, &vert_idx)) {
                        if (rope_idx >= 0 && vert_idx >= 0) {
                            // fetch projected rope verts from table
                            int maxv = 512;
                            std::vector<float> proj(static_cast<size_t>(maxv * 2));
                            int got = gp_table_get_projected_rope_vertices(rt, rope_idx, proj.data(), static_cast<int>(proj.size()));
                            if (got > 1 && vert_idx < got) {
                                int half = 3;
                                int start = std::max(0, vert_idx - half);
                                int end = std::min(got - 1, vert_idx + half);
                                int use_count = end - start + 1;
                                if (use_count >= 2) {
                                    std::vector<float> seg(static_cast<size_t>(use_count * 2));
                                    for (int vi = start; vi <= end; ++vi) {
                                        int idx = vi - start;
                                        seg[idx*2+0] = proj[vi*2+0] - static_cast<float>(ctx->offset_x);
                                        seg[idx*2+1] = proj[vi*2+1] - static_cast<float>(ctx->offset_y);
                                    }
                                    int jacket_px = std::max(1, ctx->jacket_px / 2);
                                    int jacket_border = std::max(1, ctx->jacket_border / 2);
                                    // Try to draw the widget's short rope using the same subgroup color
                                    // as any edge that maps to this rope so we don't obscure it.
                                    // prefer meta-group's own subgroup flags (set via lasso config)
                                    uint32_t mf = 0u;
                                    gp_table_meta_get_subgroup_flags(rt, mg, &mf);
                                    if (mf != 0u) {
                                        float hue = subgroup_flags_to_hue(ctx, mf);
                                        float hue_vals[1] = { hue };
                                        table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, seg.data(), use_count, jacket_px, jacket_border, hue_vals, 1, 3, 0.45f);
                                    } else {
                                        int mapped_edge = -1;
                                        for (size_t eii = 0; eii < ctx->edges.size(); ++eii) {
                                            if (ctx->edges[eii].rope_idx == rope_idx) { mapped_edge = static_cast<int>(eii); break; }
                                        }
                                        if (mapped_edge >= 0 && mapped_edge < static_cast<int>(ctx->edges.size()) && ctx->edges[mapped_edge].subgroup_flags != 0u) {
                                            float hue = subgroup_flags_to_hue(ctx, ctx->edges[mapped_edge].subgroup_flags);
                                            float hue_vals[1] = { hue };
                                            table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, seg.data(), use_count, jacket_px, jacket_border, hue_vals, 1, 3, 0.45f);
                                        } else {
                                            // subtle neutral tint so widget rope doesn't fully overlay the real rope
                                            table_draw_rope_curve_blend(out_rgba, w, h, pitch, seg.data(), use_count, jacket_px, jacket_border, 200, 160, 160, 120, 2);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
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
        int table_title_y = layout.table_section_y + 2;
        blit_text(LABEL_MENU_TABLE_NUMBER, layout.x + 8, table_title_y, 0.9f, Color{200,200,210,255});
        ToolMenuCounterLayout counter = compute_tool_menu_counter_layout(layout);
        memset_rect(out_rgba, w, h, pitch, counter.bx_minus, counter.by, counter.nbw, counter.h, Color{50,50,56,255});
        memset_rect(out_rgba, w, h, pitch, counter.bx_num, counter.by, counter.num_w, counter.h, Color{36,36,42,255});
        memset_rect(out_rgba, w, h, pitch, counter.bx_plus, counter.by, counter.nbw, counter.h, Color{50,50,56,255});
        auto minus = render_text_to_rgba(LABEL_IO_MINUS, 1.0f, {230,230,235,255});
        if (!minus.pixels.empty()) {
            int tx = counter.bx_minus + (counter.nbw - minus.width) / 2;
            int ty = counter.by + (counter.h - minus.height) / 2;
            for (int yy = 0; yy < minus.height; ++yy) {
                int dst_y = ty + yy;
                if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < minus.width; ++xx) {
                    int dst_x = tx + xx;
                    if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &minus.pixels[(yy * minus.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
        auto plus = render_text_to_rgba(LABEL_IO_PLUS, 1.0f, {230,230,235,255});
        if (!plus.pixels.empty()) {
            int tx = counter.bx_plus + (counter.nbw - plus.width) / 2;
            int ty = counter.by + (counter.h - plus.height) / 2;
            for (int yy = 0; yy < plus.height; ++yy) {
                int dst_y = ty + yy;
                if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < plus.width; ++xx) {
                    int dst_x = tx + xx;
                    if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &plus.pixels[(yy * plus.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
        std::string s = std::to_string(std::max(0, ctx->table_tool_number));
        auto bm = render_text_to_rgba(s, 1.0f, {255,255,255,255});
        if (!bm.pixels.empty()) {
            int tx = counter.bx_num + (counter.num_w - bm.width) / 2;
            int ty = counter.by + (counter.h - bm.height) / 2;
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
        auto lb = render_text_to_rgba(LABEL_MENU_TABLE_NUMBER_SHORT, 0.75f, {200,200,200,255});
        if (!lb.pixels.empty()) {
            int tx = counter.bx_num + (counter.num_w - lb.width) / 2;
            int ty = counter.by - lb.height - 2;
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

    if (ctx->plugin_menu_open) {
        PluginMenuLayout layout = compute_plugin_menu_layout(ctx, ctx->plugin_menu_module_idx, static_cast<int>(ctx->plugin_tool_ids.size()));
        if (layout.w <= 0 || layout.h <= 0) return 1;
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y, layout.w, layout.h, Color{28,32,44,235});
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y, layout.w, 1, Color{70,80,110,255});
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y + layout.h - 1, layout.w, 1, Color{12,12,18,255});
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
        blit_text(LABEL_PLUGIN_MENU_TITLE, title_x, title_y, 0.95f, Color{220,230,245,255});
        int item_y = layout.item_start_y;
        if (ctx->plugin_tool_ids.empty()) {
            int y0 = item_y;
            memset_rect(out_rgba, w, h, pitch, layout.x + 4, y0, layout.w - 8, layout.row_h - 2, Color{34,38,52,235});
            blit_text(LABEL_PLUGIN_MENU_EMPTY, layout.x + 12, y0 + 3, 0.9f, Color{170,180,200,255});
        } else {
            for (size_t i = 0; i < ctx->plugin_tool_ids.size(); ++i) {
                int y0 = item_y + static_cast<int>(i) * layout.row_h;
                memset_rect(out_rgba, w, h, pitch, layout.x + 4, y0, layout.w - 8, layout.row_h - 2, Color{42,52,70,240});
                const std::string &lbl = ctx->plugin_tool_labels[static_cast<size_t>(i)];
                blit_text(lbl.c_str(), layout.x + 12, y0 + 3, 0.9f, Color{240,245,255,255});
            }
        }
    }

    return 1;
}
