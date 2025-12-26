#include "table_abi.h"
#include "canvas_abi.h"
#include <cstdlib>
#include "menu_waveform_abi.h"
// Local copy of LassoConfig layout (kept here to avoid cross-translation-unit
// include path issues). The public header forward-declares `LassoConfig`.
typedef struct LassoConfig {
    uint32_t flags;
    uint8_t widget_type;
    uint8_t reserved[3];
    // Edge-spring parameters recorded when a lasso/meta-group enables springs
    float spring_min_rest;    // conservative rest length added when enabling springs
    float spring_reduce_rate; // rate used to reduce rest back to normal
    uint8_t spring_mode;      // reserved mode field for future behaviors
    uint8_t spring_reserved[3];
} LassoConfig;

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <cstdio>
#include <memory>
#include <fstream>
#include <iterator>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <filesystem>
#include <limits>
#include <mutex>
#include <condition_variable>
#include <tuple>
#include "text_render_helper.h"
#include "thread_manager.h"
#include "table_node_groups.h"
#include "rope_sim.h"

namespace {

// Forward declarations for static helpers used earlier in the file.


struct Color {
    uint8_t r = 0, g = 0, b = 0, a = 255;
};

constexpr float kPi = 3.14159265358979323846f;

// Edge sample type and consumer policy flags
// These provide a small, discoverable place to extend the per-edge
// metadata (e.g. `GP_TableContext::edge_subgroup_flags`) with a
// contract between producers and consumers. Values here are intentionally
// simple so we can iterate on a negotiator later.
enum EdgeSampleType : uint32_t {
    EDGE_SAMPLE_FLOAT = 0u,
    EDGE_SAMPLE_POINTER = 1u,
};

enum EdgeConsumePolicy : uint32_t {
    EDGE_POLICY_PEEK = 0u,      // consumer may peek without consuming
    EDGE_POLICY_CONSUME = 1u,   // consumer consumes and advances
    EDGE_POLICY_CONDITIONAL = 2u// consumer decides based on additional metadata
};

// Subgroup / color flags stored per-edge in `GP_TableContext::edge_subgroup_flags`.
// Reserve a few low bits for up to 5 color hues and one BYREF marker.
static constexpr uint32_t EDGE_SUBGROUP_COLOR0 = (1u << 0);
static constexpr uint32_t EDGE_SUBGROUP_COLOR1 = (1u << 1);
static constexpr uint32_t EDGE_SUBGROUP_COLOR2 = (1u << 2);
static constexpr uint32_t EDGE_SUBGROUP_COLOR3 = (1u << 3);
static constexpr uint32_t EDGE_SUBGROUP_COLOR4 = (1u << 4);
// Per-edge BYREF flag (separate from subgroup colors). Keep this distinct
// so users can either set the BYREF bit directly or map subgroup colors to
// imply BYREF behavior via `gp_table_set_subgroup_color_mapping`.
static constexpr uint32_t EDGE_SUBGROUP_BYREF  = (1u << 8); // by-ref flag forces pointer semantics

// Configurable mapping from subgroup color index (0..4) to FIFO flags.
// Users can call `gp_table_set_subgroup_color_mapping` to change these at
// runtime before applying subgroup colors to edges.
static uint32_t g_subgroup_color_to_fifo_flags[5] = {0,0,0,0,0};

// Helper to determine whether a given subgroup `flags` value implies BYREF
// semantics, either directly (BYREF bit) or via a mapped color->flag entry.
static bool flags_imply_byref(uint32_t flags) {
    if (flags & EDGE_SUBGROUP_BYREF) return true;
    // check color bits (0..4)
    for (int i = 0; i < 5; ++i) {
        uint32_t color_bit = (1u << i);
        if ((flags & color_bit) != 0) {
            if ((g_subgroup_color_to_fifo_flags[i] & EDGE_SUBGROUP_BYREF) != 0) return true;
        }
    }
    return false;
}

// Box used to carry non-pointer samples by-reference when an edge is marked BYREF.
struct BoxedSample {
    uint32_t magic; // simple tag to recognize boxed samples
    int32_t sample_len; // number of floats
    // followed by sample_len * sizeof(float) bytes
};
static constexpr uint32_t BOXED_SAMPLE_MAGIC = 0xB0B5A55Au; // arbitrary unique tag

// Global template library directory (can be set by gp_table_set_library_dir).
static std::string g_template_library_dir;

static std::string make_template_filename(const char* dir, const char* name) {
    std::string d;
    if (dir && dir[0]) d = std::string(dir);
    else d = g_template_library_dir;
    if (d.empty()) d = ".";
    std::filesystem::path p(d);
    std::string fname = name ? std::string(name) : std::string("untitled");
    p /= (fname + std::string(".gptbl"));
    return p.string();
}

extern "C" int32_t gp_table_set_library_dir(const char* dir) {
    if (!dir || dir[0] == '\0') { g_template_library_dir.clear(); return 1; }
    try { g_template_library_dir = std::string(dir); return 1; } catch (...) { return 0; }
}

extern "C" int32_t gp_table_get_library_dir(char* out_buf, int32_t out_len) {
    if (!out_buf && out_len != 0) return 0;
    const std::string &d = g_template_library_dir;
    if (!out_buf) return static_cast<int32_t>(d.size());
    int32_t to_write = std::min<int32_t>(static_cast<int32_t>(d.size()), out_len);
    if (to_write > 0) memcpy(out_buf, d.data(), static_cast<size_t>(to_write));
    return to_write;
}

int32_t gp_table_save_template(GP_TableContext* ctx, const char* dir, const char* name) {
    if (!ctx || !name) return 0;
    int32_t need = gp_table_serialize(ctx, nullptr, 0);
    if (need <= 0) return 0;
    std::vector<char> buf(static_cast<size_t>(need));
    int32_t wrote = gp_table_serialize(ctx, buf.data(), need);
    if (wrote != need) return 0;
    std::string path = make_template_filename(dir, name);
    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        if (!ofs.good()) return 0;
        ofs.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        ofs.close();
    } catch (...) {
        return 0;
    }
    return 1;
}

extern "C" int32_t gp_table_set_subgroup_color_mapping(int32_t color_idx, uint32_t fifo_flags) {
    if (color_idx < 0 || color_idx >= 5) return 0;
    g_subgroup_color_to_fifo_flags[static_cast<size_t>(color_idx)] = fifo_flags;
    return 1;
}

extern "C" int32_t gp_table_load_template(GP_TableContext* ctx, const char* dir, const char* name) {
    if (!ctx || !name) return 0;
    std::string path = make_template_filename(dir, name);
    try {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.good()) return 0;
        std::vector<char> buf((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        ifs.close();
        if (buf.empty()) return 0;
        return gp_table_deserialize(ctx, buf.data(), static_cast<int32_t>(buf.size()));
    } catch (...) {
        return 0;
    }
}

extern "C" int32_t gp_table_list_templates(const char* dir, char* out_buf, int32_t out_len) {
    std::string d;
    if (dir && dir[0]) d = std::string(dir);
    else d = g_template_library_dir;
    if (d.empty()) d = ".";
    std::string listing;
    try {
        for (auto &entry : std::filesystem::directory_iterator(d)) {
            if (!entry.is_regular_file()) continue;
            auto p = entry.path();
            if (p.extension() == ".gptbl") {
                listing += p.stem().string();
                listing += '\n';
            }
        }
    } catch (...) {
        return 0;
    }
    if (!out_buf) return static_cast<int32_t>(listing.size());
    int32_t to_write = std::min<int32_t>(static_cast<int32_t>(listing.size()), out_len);
    if (to_write > 0) memcpy(out_buf, listing.data(), static_cast<size_t>(to_write));
    return to_write;
}

// Simple default-table spec used by the C++ table implementation.
// Keeps a single place to tune the "charcoal" default appearance.
struct DefaultTableSpec {
    int width_px = 640;
    int row_h_px = 20;
    uint8_t bg_rgba[4] = {40, 40, 50, 255}; // charcoal
};
static const DefaultTableSpec kDefaultTableSpec;

static void memset_rect(uint8_t* img, int w, int h, int pitch, int x0, int y0, int rw, int rh, Color c);

static inline Color to_color(const uint8_t rgba[4]) {
    return Color{rgba[0], rgba[1], rgba[2], rgba[3]};
}
static void draw_calib_strip(
    uint8_t* img,
    int w,
    int h,
    int pitch,
    int x0,
    int y0,
    int cw,
    int ch,
    bool inv_on,
    int mode,
    Color dim,
    Color active,
    Color inv_col) {
    if (!img || cw <= 0 || ch <= 0) return;
    const int slots = 5; // INV, TRIM, CAP, DED, RST
    int gap = 2;
    int slot_w = std::max(4, (cw - gap * (slots - 1)) / slots);
    int slot_h = std::max(4, ch);

    // Calibration slot label texts and colors (text color still used; background will be transparent)
    static const char* calib_labels_lower[5] = { "inv", "trim", "cap", "ded", "rst" };
    static const char* calib_labels_upper[5] = { "INV", "TRIM", "CAP", "DED", "RST" };
    static const uint8_t calib_label_color[5][4] = {
        {160,160,160,255}, // grey (inv)
        {255,255,255,255}, // white (trim)
        {64,220,64,255},   // green (cap)
        {220,64,64,255},   // red (ded)
        {255,255,255,255}  // white (rst)
    };

    // Pre-render labels to determine required slot height
    std::array<TextBitmap,5> bms;
    int max_bh = 0;
    for (int i = 0; i < slots; ++i) {
        const char* txt = (i == 2 || i == 3) ? calib_labels_upper[i] : calib_labels_lower[i];
        std::array<unsigned char,4> col = { calib_label_color[i][0], calib_label_color[i][1], calib_label_color[i][2], calib_label_color[i][3] };
        bms[i] = render_text_to_rgba(std::string(txt), 1.0f, col);
        if (!bms[i].pixels.empty()) max_bh = std::max<int>(max_bh, bms[i].height);
    }

    const int vpad = 6; // vertical padding inside slot
    slot_h = std::max(slot_h, max_bh + vpad);
    int y = y0 + (ch - slot_h) / 2;

    // Draw only outlines for simple text buttons (no colored fills)
    for (int i = 0; i < slots; ++i) {
        int x = x0 + i * (slot_w + gap);
        // thin outline for button
        memset_rect(img, w, h, pitch, x, y, slot_w, 1, Color{255,255,255,32});
        memset_rect(img, w, h, pitch, x, y + slot_h - 1, slot_w, 1, Color{0,0,0,64});
        memset_rect(img, w, h, pitch, x, y, 1, slot_h, Color{255,255,255,32});
        memset_rect(img, w, h, pitch, x + slot_w - 1, y, 1, slot_h, Color{0,0,0,64});
    }

    // Render labels centered in each slot. Bold simulated by overdrawing with 1px offset.
    for (int i = 0; i < slots; ++i) {
        int sx = x0 + i * (slot_w + gap);
        int sy = y;
        const TextBitmap &bm = bms[i];
        if (bm.pixels.empty()) continue;
        int bx = sx + (slot_w - bm.width) / 2;
        int by = sy + (slot_h - bm.height) / 2;
        for (int yy = 0; yy < bm.height; ++yy) {
            int dst_y = by + yy;
            if (dst_y < 0 || dst_y >= h) continue;
            for (int xx = 0; xx < bm.width; ++xx) {
                int dst_x = bx + xx;
                if (dst_x < 0 || dst_x >= w) continue;
                unsigned char* dst = img + dst_y * pitch + dst_x * 4;
                const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                float sa = src[3] / 255.0f;
                if (sa >= 0.999f) {
                    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
                } else if (sa > 0.001f) {
                    for (int cch = 0; cch < 3; ++cch) {
                        float s = src[cch] / 255.0f;
                        float d = dst[cch] / 255.0f;
                        float out = s * sa + d * (1.0f - sa);
                        dst[cch] = static_cast<uint8_t>(std::lround(out * 255.0f));
                    }
                    float da = dst[3] / 255.0f;
                    float outa = sa + da * (1.0f - sa);
                    dst[3] = static_cast<uint8_t>(std::lround(outa * 255.0f));
                }
            }
        }
        // bold: overdraw with 1px offset
        for (int yy = 0; yy < bm.height; ++yy) {
            int dst_y = by + yy + 1;
            if (dst_y < 0 || dst_y >= h) continue;
            for (int xx = 0; xx < bm.width; ++xx) {
                int dst_x = bx + xx + 1;
                if (dst_x < 0 || dst_x >= w) continue;
                unsigned char* dst = img + dst_y * pitch + dst_x * 4;
                const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                float sa = src[3] / 255.0f;
                if (sa >= 0.999f) {
                    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
                } else if (sa > 0.001f) {
                    for (int cch = 0; cch < 3; ++cch) {
                        float s = src[cch] / 255.0f;
                        float d = dst[cch] / 255.0f;
                        float out = s * sa + d * (1.0f - sa);
                        dst[cch] = static_cast<uint8_t>(std::lround(out * 255.0f));
                    }
                    float da = dst[3] / 255.0f;
                    float outa = sa + da * (1.0f - sa);
                    dst[3] = static_cast<uint8_t>(std::lround(outa * 255.0f));
                }
            }
        }
    }
}


struct Style {
    int w = 640;
    int row_h = 20;
    int indent = 14;
    int expand_w = 12;
    int name_w = 160;

    Color bg{8, 8, 12, 255};
    Color bg_sel{18, 18, 26, 255};
    Color hdr{20, 20, 28, 255};
    Color text{240, 240, 245, 255};
    Color text_hdr{255, 255, 255, 255};
    Color led_on{255, 210, 90, 255};
    Color led_off{70, 70, 80, 255};
    Color led_edge{255, 255, 255, 255};
    Color axis_bg{18, 18, 22, 255};
    Color axis_tick{200, 200, 200, 255};
    Color axis_val{255, 255, 140, 255};
    Color timer{110, 180, 255, 255};
    Color wave_bg{6, 6, 8, 255};
    Color wave_fg{255, 255, 255, 255};

    uint32_t led_mask[9] = {1u << 0, 1u << 1, 1u << 2, 1u << 3, 1u << 4, 1u << 5, 1u << 6, 1u << 7, 1u << 8};
    uint32_t led_edge_mask = (1u << 1) | (1u << 2) | (1u << 4) | (1u << 7) | (1u << 8);
    // Cable / edge render parameters
    int cable_segments = 24;         // samples along cable
    int cable_jacket_px = 5;        // outer jacket radius in pixels
    int cable_jacket_border = 2;    // inner border thickness (core = jacket - border)
    float cable_core_alpha = 0.92f; // core translucency (0..1)
    float cable_sag = 0.10f;        // sag factor relative to distance (0..1)
    float cable_plug_depth = 10.0f; // depth to sink plugs behind the UI plane
    float cable_tilt_x = 0.06f;     // orthographic tilt factor in X for depth projection
    float cable_tilt_y = -0.08f;    // orthographic tilt factor in Y for depth projection
    float cable_depth_fade = 0.55f; // how much depth mutes fiber glow
    // increase default fiber gain to make transferred glow/hue stronger
    float cable_fiber_gain = 1.45f; // multiplier for fiber optic overlay
    float cable_fiber_radius_scale = 0.55f; // overlay radius relative to jacket
    int cable_fifo_light_mode = 1;
    float cable_fifo_friction_half_life = 0.35f;
    float cable_fifo_friction_gain = 0.85f;
    int cable_fifo_friction_regions = 0;
    float cable_fifo_friction_tint = 0.12f;
};

static Style load_style(const GP_TableStyle* s) {
    Style out;
    if (!s) return out;
    out.w = std::max(64, int(s->width_px));
    out.row_h = std::max(8, int(s->row_h_px));
    out.indent = std::max(0, int(s->indent_px));
    out.expand_w = std::max(8, int(s->expand_w_px));
    out.name_w = std::max(20, int(s->name_w_px));
    out.bg = to_color(s->bg_rgba);
    out.bg_sel = to_color(s->bg_sel_rgba);
    out.hdr = to_color(s->hdr_rgba);
    out.text = to_color(s->text_rgba);
    out.text_hdr = to_color(s->text_hdr_rgba);
    out.led_on = to_color(s->led_on_rgba);
    out.led_off = to_color(s->led_off_rgba);
    out.led_edge = to_color(s->led_edge_rgba);
    out.axis_bg = to_color(s->axis_bg_rgba);
    out.axis_tick = to_color(s->axis_tick_rgba);
    out.axis_val = to_color(s->axis_val_rgba);
    out.timer = to_color(s->timer_rgba);
    out.wave_bg = to_color(s->wave_bg_rgba);
    out.wave_fg = to_color(s->wave_fg_rgba);
    for (int i = 0; i < 9; ++i) out.led_mask[i] = s->led_mask[i];
    out.led_edge_mask = s->led_edge_mask;
    out.cable_fifo_light_mode = s->cable_fifo_light_mode != 0;
    if (s->cable_fifo_friction_half_life > 0.0f) out.cable_fifo_friction_half_life = s->cable_fifo_friction_half_life;
    if (s->cable_fifo_friction_gain > 0.0f) out.cable_fifo_friction_gain = s->cable_fifo_friction_gain;
    if (s->cable_fifo_friction_regions > 0) out.cable_fifo_friction_regions = s->cable_fifo_friction_regions;
    if (s->cable_fifo_friction_tint > 0.0f) out.cable_fifo_friction_tint = s->cable_fifo_friction_tint;
    if (out.cable_fifo_friction_regions <= 0) out.cable_fifo_friction_regions = std::max(1, out.cable_segments);
    return out;
}

static GP_TableStyle make_default_style() {
    GP_TableStyle s{};
    s.width_px = kDefaultTableSpec.width_px;
    s.row_h_px = kDefaultTableSpec.row_h_px;
    s.indent_px = 14;
    s.expand_w_px = 12;
    s.name_w_px = 160;
    auto set = [](uint8_t dst[4], uint8_t r, uint8_t g, uint8_t b, uint8_t a) { dst[0] = r; dst[1] = g; dst[2] = b; dst[3] = a; };
    set(s.bg_rgba, kDefaultTableSpec.bg_rgba[0], kDefaultTableSpec.bg_rgba[1], kDefaultTableSpec.bg_rgba[2], kDefaultTableSpec.bg_rgba[3]);
    set(s.bg_sel_rgba, 18, 18, 26, 255);
    set(s.hdr_rgba, 20, 20, 28, 255);
    set(s.text_rgba, 240, 240, 245, 255);
    set(s.text_hdr_rgba, 255, 255, 255, 255);
    set(s.led_on_rgba, 255, 210, 90, 255);
    set(s.led_off_rgba, 70, 70, 80, 255);
    set(s.led_edge_rgba, 255, 255, 255, 255);
    set(s.axis_bg_rgba, 18, 18, 22, 255);
    set(s.axis_tick_rgba, 200, 200, 200, 255);
    set(s.axis_val_rgba, 255, 255, 140, 255);
    set(s.timer_rgba, 110, 180, 255, 255);
    set(s.wave_bg_rgba, 6, 6, 8, 255);
    set(s.wave_fg_rgba, 255, 255, 255, 255);
    uint32_t default_bits[9] = {1u << 0, 1u << 1, 1u << 2, 1u << 3, 1u << 4, 1u << 5, 1u << 6, 1u << 7, 1u << 8};
    for (int i = 0; i < 9; ++i) s.led_mask[i] = default_bits[i];
    s.led_edge_mask = (1u << 1) | (1u << 2) | (1u << 4) | (1u << 7) | (1u << 8);
    s.cable_fifo_light_mode = 1;
    s.cable_fifo_friction_half_life = 0.35f;
    s.cable_fifo_friction_gain = 0.85f;
    s.cable_fifo_friction_regions = 0;
    s.cable_fifo_friction_tint = 0.12f;
    return s;
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
            row[0] = c.r;
            row[1] = c.g;
            row[2] = c.b;
            row[3] = c.a;
            row += 4;
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
            if (dx * dx + dy * dy > r2) continue;
            uint8_t* p = img + y * pitch + x * 4;
            p[0] = c.r;
            p[1] = c.g;
            p[2] = c.b;
            p[3] = c.a;
        }
    }
}

static void draw_line(uint8_t* img, int w, int h, int pitch, int x0, int y0, int x1, int y1, int thick, Color c) {
    if (!img) return;
    const int dx = std::abs(x1 - x0);
    const int dy = -std::abs(y1 - y0);
    const int sx = x0 < x1 ? 1 : -1;
    const int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int x = x0;
    int y = y0;
    while (true) {
        memset_rect(img, w, h, pitch, x - thick / 2, y - thick / 2, thick, thick, c);
        if (x == x1 && y == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y += sy;
        }
    }
}

static void draw_axis_bar_ext(
    uint8_t* img,
    int w,
    int h,
    int pitch,
    int x0,
    int y0,
    int bar_w,
    int bar_h,
    float v_cur,
    float v_min,
    float v_max,
    float cap_min,
    float cap_max,
    float trim,
    float deadzone,
    bool seen_min,
    bool seen_max,
    Color bg,
    Color tick,
    Color val) {
    if (!img || bar_w <= 0 || bar_h <= 0) return;

    auto clamp_unit = [](float v) { return std::max(-1.0f, std::min(1.0f, v)); };
    auto x_at = [&](float vv) {
        float v01 = 0.5f * (clamp_unit(vv) + 1.0f);
        return x0 + int(std::lround(v01 * float(std::max(1, bar_w - 1))));
    };

    // Background
    memset_rect(img, w, h, pitch, x0, y0, bar_w, bar_h, bg);

    // End ticks: red if never reached, white otherwise.
    Color col_end_ok{255, 255, 255, 255};
    Color col_end_bad{255, 80, 80, 255};
    draw_line(img, w, h, pitch, x_at(-1.0f), y0 - 2, x_at(-1.0f), y0 + bar_h + 2, 1, seen_min ? col_end_ok : col_end_bad);
    draw_line(img, w, h, pitch, x_at(+1.0f), y0 - 2, x_at(+1.0f), y0 + bar_h + 2, 1, seen_max ? col_end_ok : col_end_bad);

    // Center tick
    draw_line(img, w, h, pitch, x_at(0.0f), y0 - 2, x_at(0.0f), y0 + bar_h + 2, 1, col_end_ok);

    // Deadzone band around trim point
    float cap_lo = cap_min;
    float cap_hi = cap_max;
    if (!(cap_hi > cap_lo + 1e-6f)) {
        cap_lo = -1.0f;
        cap_hi = 1.0f;
    }
    float mid = 0.5f * (cap_lo + cap_hi);
    float span = 0.5f * (cap_hi - cap_lo);
    if (span < 1e-6f) span = 1e-6f;

    // Trim tick (orange) and deadzone bounds (blue) if provided.
    Color col_trim{255, 140, 80, 255};
    Color col_dz{64, 140, 255, 255};
    float trim_raw = mid + trim * span;
    draw_line(img, w, h, pitch, x_at(trim_raw), y0 - 2, x_at(trim_raw), y0 + bar_h + 2, 1, col_trim);

    if (deadzone > 0.0f) {
        float dz_raw = deadzone * span;
        int x0_dz = x_at(trim_raw - dz_raw);
        int x1_dz = x_at(trim_raw + dz_raw);
        int xa = std::min(x0_dz, x1_dz);
        int xb = std::max(x0_dz, x1_dz);
        memset_rect(img, w, h, pitch, xa, y0, std::max(1, xb - xa + 1), bar_h, Color{32, 32, 32, 255});
        draw_line(img, w, h, pitch, x0_dz, y0 - 2, x0_dz, y0 + bar_h + 2, 1, col_dz);
        draw_line(img, w, h, pitch, x1_dz, y0 - 2, x1_dz, y0 + bar_h + 2, 1, col_dz);
    }

    // Observed min/max ticks (green)
    Color col_seen{64, 255, 64, 255};
    draw_line(img, w, h, pitch, x_at(v_min), y0 - 1, x_at(v_min), y0 + bar_h + 1, 1, col_seen);
    draw_line(img, w, h, pitch, x_at(v_max), y0 - 1, x_at(v_max), y0 + bar_h + 1, 1, col_seen);

    // Current value (yellow)
    draw_line(img, w, h, pitch, x_at(v_cur), y0 - 3, x_at(v_cur), y0 + bar_h + 3, 2, val);
}

static void draw_timers(uint8_t* img, int w, int h, int pitch, int x0, int y0, int w_px, int h_px, float hold_s, float last_s, Color c) {
    if (!img || w_px <= 0 || h_px <= 0) return;
    const float clamp_hold = std::min(1.0f, std::max(0.0f, hold_s));
    const float clamp_last = std::min(1.0f, std::max(0.0f, last_s));
    int hold_w = int(std::lround(clamp_hold * float(w_px)));
    int last_w = int(std::lround(clamp_last * float(w_px)));
    int mid = y0 + h_px / 2;
    memset_rect(img, w, h, pitch, x0, mid - h_px / 2, hold_w, h_px / 2, c);
    memset_rect(img, w, h, pitch, x0, mid, last_w, h_px / 2, c);
}

static void draw_waveform(uint8_t* img, int w, int h, int pitch, int x0, int y0, int ww, int wh, const GP_MenuWaveform* wf, Color bg, Color fg) {
    if (!img || ww <= 0 || wh <= 0) return;
    memset_rect(img, w, h, pitch, x0, y0, ww, wh, bg);
    if (!wf) return;
    const int need = ww * wh * 4;
    std::vector<uint8_t> tmp;
    tmp.resize(static_cast<std::size_t>(need));
    if (!gp_menu_waveform_raster_rgba(const_cast<GP_MenuWaveform*>(wf), tmp.data(), int32_t(tmp.size()))) {
        return;
    }
    for (int yy = 0; yy < wh; ++yy) {
        for (int xx = 0; xx < ww; ++xx) {
            int tx = x0 + xx;
            int ty = y0 + yy;
            if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
            const uint8_t* src = &tmp[(yy * ww + xx) * 4];
            uint8_t* dst = img + ty * pitch + tx * 4;
            dst[0] = uint8_t((int(src[0]) * int(fg.r)) / 255);
            dst[1] = uint8_t((int(src[1]) * int(fg.g)) / 255);
            dst[2] = uint8_t((int(src[2]) * int(fg.b)) / 255);
            dst[3] = src[3];
        }
    }
}

// Simple nearest-neighbor image blit scaled into a cell rectangle.
static void draw_image_cell(uint8_t* img, int w, int h, int pitch, int x0, int y0, int cw, int ch, const GP_TableImage* src) {
    if (!img || !src || !src->rgba || cw <= 0 || ch <= 0 || src->width_px <= 0 || src->height_px <= 0) return;
    int sw = src->width_px;
    int sh = src->height_px;
    int spitch = src->pitch_bytes;
    // Many producers treat the buffer as opaque but leave alpha uninitialized/zero.
    // Detect the common "alpha is all zero" case and treat it as fully opaque.
    bool alpha_all_zero = true;
    {
        const int sx_samples[4] = {0, std::max(0, sw / 3), std::max(0, (2 * sw) / 3), std::max(0, sw - 1)};
        const int sy_samples[4] = {0, std::max(0, sh / 3), std::max(0, (2 * sh) / 3), std::max(0, sh - 1)};
        for (int syi = 0; syi < 4 && alpha_all_zero; ++syi) {
            int sy = std::clamp(sy_samples[syi], 0, sh - 1);
            const uint8_t* row = src->rgba + sy * spitch;
            for (int sxi = 0; sxi < 4; ++sxi) {
                int sx = std::clamp(sx_samples[sxi], 0, sw - 1);
                const uint8_t a = row[sx * 4 + 3];
                if (a != 0) { alpha_all_zero = false; break; }
            }
        }
    }

    // Preserve aspect ratio: fit source into destination rect and center.
    float sx = float(cw) / float(std::max(1, sw));
    float sy = float(ch) / float(std::max(1, sh));
    float s = std::min(sx, sy);
    int dw = std::max(1, int(std::lround(float(sw) * s)));
    int dh = std::max(1, int(std::lround(float(sh) * s)));
    dw = std::min(dw, cw);
    dh = std::min(dh, ch);
    int ox = x0 + (cw - dw) / 2;
    int oy = y0 + (ch - dh) / 2;

    for (int yy = 0; yy < dh; ++yy) {
        int dy = oy + yy;
        if (dy < 0 || dy >= h) continue;
        float syf = (dh > 1) ? (float(yy) / float(dh - 1)) * float(sh - 1) : 0.0f;
        int syi = std::clamp(int(std::round(syf)), 0, sh - 1);
        const uint8_t* src_row = src->rgba + syi * spitch;
        for (int xx = 0; xx < dw; ++xx) {
            int dx = ox + xx;
            if (dx < 0 || dx >= w) continue;
            float sxf = (dw > 1) ? (float(xx) / float(dw - 1)) * float(sw - 1) : 0.0f;
            int sxi = std::clamp(int(std::round(sxf)), 0, sw - 1);
            const uint8_t* sp = src_row + sxi * 4;
            uint8_t* dp = img + dy * pitch + dx * 4;
            float sa = alpha_all_zero ? 1.0f : (sp[3] / 255.0f);
            if (sa <= 0.0f) continue;
            float inv = 1.0f - sa;
            for (int c = 0; c < 3; ++c) {
                float s0 = sp[c] / 255.0f;
                float d0 = dp[c] / 255.0f;
                dp[c] = static_cast<uint8_t>(std::lround((s0 * sa + d0 * inv) * 255.0f));
            }
            float da = dp[3] / 255.0f;
            dp[3] = alpha_all_zero ? 255 : static_cast<uint8_t>(std::lround((sa + da * inv) * 255.0f));
        }
    }
}

static void draw_expand_box(uint8_t* img, int w, int h, int pitch, int x0, int y0, int size, bool expanded, Color c) {
    int s = std::max(6, size);
    int y_mid = y0 + s / 2;
    int x_mid = x0 + s / 2;
    // box outline
    memset_rect(img, w, h, pitch, x0, y0, s, 1, c);
    memset_rect(img, w, h, pitch, x0, y0 + s - 1, s, 1, c);
    memset_rect(img, w, h, pitch, x0, y0, 1, s, c);
    memset_rect(img, w, h, pitch, x0 + s - 1, y0, 1, s, c);
    // plus/minus
    draw_line(img, w, h, pitch, x0 + 2, y_mid, x0 + s - 3, y_mid, 1, c);
    if (!expanded) {
        draw_line(img, w, h, pitch, x_mid, y0 + 2, x_mid, y0 + s - 3, 1, c);
    }
}

static void draw_led_strip(
    uint8_t* img,
    int w,
    int h,
    int pitch,
    int x0,
    int cy,
    int cw,
    int count,
    uint32_t on_mask,
    uint32_t edge_mask,
    uint32_t active_mask,
    int radius,
    Color on,
    Color off,
    Color edge,
    Color active_only) {
    if (!img || cw <= 0 || count <= 0) return;
    int led_spacing = std::max(radius * 2 + 2, cw / std::max(1, count + 1));
    int cx0 = x0 + led_spacing;
    for (int li = 0; li < count; ++li) {
        int cx = cx0 + li * led_spacing;
        uint32_t bit = 1u << li;
        bool on_b = (on_mask & bit) != 0;
        bool edge_b = (edge_mask & bit) != 0;
        bool active_but_off = !on_b && (active_mask & bit);
        Color lc = on_b ? on : (active_but_off ? active_only : off);
        if (edge_b) lc = edge;
        draw_circle(img, w, h, pitch, cx, cy, radius, lc);
    }
}

static void draw_scrollbar(
    uint8_t* img,
    int w,
    int h,
    int pitch,
    int x0,
    int y0,
    int cw,
    int ch,
    float value01,
    int total_rows,
    int visible_rows,
    bool arrow_up_pressed,
    bool arrow_dn_pressed,
    Color track,
    Color thumb,
    Color arrow_up,
    Color arrow_dn,
    Color stripe_even,
    Color stripe_odd) {
    if (!img || cw <= 4 || ch <= 4) return;
    int bar_x = x0;
    int bar_w = std::max(6, cw);
    int bar_h = ch;
    int arrow_h = std::max(8, bar_h / 6);
    // Track
    memset_rect(img, w, h, pitch, bar_x, y0, bar_w, bar_h, track);
    // Arrows
    memset_rect(img, w, h, pitch, bar_x, y0, bar_w, arrow_h, arrow_up_pressed ? arrow_dn : arrow_up);
    memset_rect(img, w, h, pitch, bar_x, y0 + bar_h - arrow_h, bar_w, arrow_h, arrow_dn_pressed ? arrow_up : arrow_dn);

    int track_y0 = y0 + arrow_h;
    int track_h = bar_h - 2 * arrow_h;
    if (track_h <= 0) return;

    // Alternating stripes to suggest rows.
    int rows = std::max(1, total_rows);
    float stripe_h = float(track_h) / float(rows);
    for (int r = 0; r < rows; ++r) {
        int sy0 = track_y0 + int(std::floor(stripe_h * r));
        int sy1 = track_y0 + int(std::floor(stripe_h * (r + 1)));
        int sh = std::max(1, sy1 - sy0);
        Color sc = (r % 2 == 0) ? stripe_even : stripe_odd;
        memset_rect(img, w, h, pitch, bar_x, sy0, bar_w, sh, sc);
    }

    // Thumb size ~ visible/total.
    float vis_frac = std::clamp(float(visible_rows) / float(std::max(visible_rows, total_rows)), 0.05f, 1.0f);
    int thumb_h = std::max(6, int(vis_frac * track_h));
    float clamped_v = std::clamp(value01, 0.0f, 1.0f);
    int thumb_y = track_y0 + int((track_h - thumb_h) * clamped_v);
    memset_rect(img, w, h, pitch, bar_x, thumb_y, bar_w, thumb_h, thumb);
    // Outline
    memset_rect(img, w, h, pitch, bar_x, thumb_y, bar_w, 1, Color{255, 255, 255, 48});
    memset_rect(img, w, h, pitch, bar_x, thumb_y + thumb_h - 1, bar_w, 1, Color{0, 0, 0, 96});
}

static void blit_text_bitmap(uint8_t* img, int w, int h, int pitch, const TextBitmap& bm, int x, int y, int clip_x0, int clip_y0, int clip_x1, int clip_y1) {
    if (!img || bm.pixels.empty()) return;
    for (int yy = 0; yy < bm.height; ++yy) {
        int dst_y = y + yy;
        if (dst_y < clip_y0 || dst_y >= clip_y1 || dst_y < 0 || dst_y >= h) continue;
        for (int xx = 0; xx < bm.width; ++xx) {
            int dst_x = x + xx;
            if (dst_x < clip_x0 || dst_x >= clip_x1 || dst_x < 0 || dst_x >= w) continue;
            uint8_t* dst = img + dst_y * pitch + dst_x * 4;
            const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
            float sa = src[3] / 255.0f;
            if (sa >= 0.999f) {
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
            } else if (sa > 0.001f) {
                for (int cch = 0; cch < 3; ++cch) {
                    float s = src[cch] / 255.0f;
                    float d = dst[cch] / 255.0f;
                    float out = s * sa + d * (1.0f - sa);
                    dst[cch] = static_cast<uint8_t>(std::lround(out * 255.0f));
                }
                float da = dst[3] / 255.0f;
                float outa = sa + da * (1.0f - sa);
                dst[3] = static_cast<uint8_t>(std::lround(outa * 255.0f));
            }
        }
    }
}

static void compute_columns(const GP_TableColumn* cols, int col_count, int table_w, int name_w, int* out_x0, int* out_w) {
    int x = name_w;
    for (int i = 0; i < col_count && i < 8; ++i) {
        int cw = std::max(1, int(cols[i].width_px));
        out_x0[i] = x;
        out_w[i] = cw;
        x += cw;
    }
    for (int i = col_count; i < 8; ++i) {
        out_x0[i] = x;
        out_w[i] = 0;
    }
}

static int row_base_height_px(const Style& st, const GP_TableRow& row) {
    if (row.reserved0 > 0) return std::max(1, int(row.reserved0));
    return std::max(1, st.row_h);
}

static bool row_find_image_cell(const GP_TableRow& row, const GP_TableCell** out_cell) {
    if (out_cell) *out_cell = nullptr;
    for (int c = 0; c < row.cell_count && c < 8; ++c) {
        if (row.cells[c].kind == GP_TABLE_CELL_IMAGE) {
            if (out_cell) *out_cell = &row.cells[c];
            return true;
        }
    }
    return false;
}

static int row_image_header_h(const Style& st, const GP_TableRow& row) {
    (void)row;
    return std::max(1, st.row_h);
}

static int row_desired_height_px(const Style& st, const GP_TableRow& row) {
    // Fixed row height always wins.
    if (row.reserved0 > 0) return std::max(1, int(row.reserved0));

    const GP_TableCell* img_cell = nullptr;
    if (!row_find_image_cell(row, &img_cell)) return std::max(1, st.row_h);

    // Image rows: a header strip plus optional expanded image area.
    int header_h = row_image_header_h(st, row);
    if (!img_cell || row.expanded == 0) return header_h;
    if (!img_cell->image || !img_cell->image->rgba || img_cell->image->width_px <= 0 || img_cell->image->height_px <= 0) return header_h;

    const int max_img_h_default = 240;
    int max_img_h = (img_cell->reserved0 > 0) ? std::max(1, img_cell->reserved0) : max_img_h_default;

    // Default behavior: image content spans the full content width (excluding the label gutter).
    int content_w = std::max(1, st.w - st.name_w - 4);
    float aspect = float(img_cell->image->height_px) / float(std::max(1, img_cell->image->width_px));
    
    int desired_img_h = std::max(1, int(std::lround(float(content_w) * aspect)));
    int img_h = std::clamp(desired_img_h, 0, max_img_h);
    return header_h + img_h;
}

// Row layout policy:
// - row.reserved0 > 0 => fixed height in pixels
// - row.reserved0 == 0 => default height (style row height)
// - row.reserved0 < 0 => flex row (eligible to absorb extra height)
static void compute_row_layout(const GP_TableRow* rows, int row_count, const Style& st, int target_h,
    std::vector<int>& out_y0, std::vector<int>& out_h) {
    out_y0.assign(static_cast<size_t>(std::max(0, row_count)), 0);
    out_h.assign(static_cast<size_t>(std::max(0, row_count)), 0);
    if (!rows || row_count <= 0) return;

    std::vector<int> h0(static_cast<size_t>(row_count), 0);
    std::vector<int> hmin(static_cast<size_t>(row_count), 0);
    std::vector<uint8_t> is_img_row(static_cast<size_t>(row_count), 0);
    int sum = 0;
    int flex_count = 0;
    for (int i = 0; i < row_count; ++i) {
        const GP_TableRow& r = rows[i];
        const bool has_img = row_find_image_cell(r, nullptr);
        is_img_row[static_cast<size_t>(i)] = has_img ? 1u : 0u;

        int rh_min = row_base_height_px(st, r);
        if (has_img && r.reserved0 <= 0) rh_min = row_image_header_h(st, r);
        rh_min = std::max(1, rh_min);
        hmin[static_cast<size_t>(i)] = rh_min;

        int rh = has_img ? row_desired_height_px(st, r) : rh_min;
        rh = std::max(rh_min, rh);
        h0[static_cast<size_t>(i)] = rh;
        sum += rh;
        if (rows[i].reserved0 < 0) ++flex_count;
    }

    if (target_h > 0 && target_h != sum) {
        if (target_h > sum) {
            int extra = target_h - sum;
            if (flex_count > 0) {
                int per = extra / flex_count;
                int rem = extra % flex_count;
                for (int i = 0; i < row_count; ++i) {
                    if (rows[i].reserved0 < 0) {
                        h0[static_cast<size_t>(i)] += per;
                        if (rem > 0) { h0[static_cast<size_t>(i)] += 1; --rem; }
                    }
                }
            } else {
                h0.back() += extra;
            }
        } else {
            // Shrink to fit; prefer shrinking expanded image rows down to their header height first.
            int over = sum - target_h;
            for (int i = 0; i < row_count && over > 0; ++i) {
                if (!is_img_row[static_cast<size_t>(i)]) continue;
                int can = h0[static_cast<size_t>(i)] - hmin[static_cast<size_t>(i)];
                if (can <= 0) continue;
                int take = std::min(over, can);
                h0[static_cast<size_t>(i)] -= take;
                over -= take;
            }

            if (over > 0) {
                // Scale down remaining overflow across all rows, bounded by hmin.
                int cur_sum = 0;
                for (int i = 0; i < row_count; ++i) cur_sum += h0[static_cast<size_t>(i)];
                double scale = (cur_sum > 0) ? (double(target_h) / double(cur_sum)) : 1.0;
                int new_sum = 0;
                for (int i = 0; i < row_count; ++i) {
                    int rh = int(std::floor(double(h0[static_cast<size_t>(i)]) * scale));
                    rh = std::max(hmin[static_cast<size_t>(i)], std::max(1, rh));
                    h0[static_cast<size_t>(i)] = rh;
                    new_sum += rh;
                }
                int diff = target_h - new_sum;
                if (diff > 0) {
                    for (int i = 0; i < row_count && diff > 0; ++i) { h0[static_cast<size_t>(i)] += 1; --diff; }
                } else if (diff < 0) {
                    for (int i = row_count - 1; i >= 0 && diff < 0; --i) {
                        int minv = std::max(1, hmin[static_cast<size_t>(i)]);
                        if (h0[static_cast<size_t>(i)] > minv) { h0[static_cast<size_t>(i)] -= 1; ++diff; }
                    }
                }
            }
        }
    }

    int y = 0;
    for (int i = 0; i < row_count; ++i) {
        out_y0[static_cast<size_t>(i)] = y;
        out_h[static_cast<size_t>(i)] = h0[static_cast<size_t>(i)];
        y += h0[static_cast<size_t>(i)];
    }
}

} // namespace

extern "C" {


int32_t gp_table_calc_size(const GP_TableStyle* style, int32_t row_count, int32_t col_count, GP_TableGeom* out_geom) {
    if (!out_geom || row_count < 0 || col_count < 0) return 0;
    Style st = load_style(style);
    out_geom->width_px = st.w;
    out_geom->height_px = std::max(1, int(row_count) * st.row_h);
    compute_columns(nullptr, 0, st.w, st.name_w, out_geom->col_x0, out_geom->col_w);
    return 1;
}

int32_t gp_table_raster_rgba_with_hits(
    const GP_TableRow* rows,
    int32_t row_count,
    const GP_TableColumn* cols,
    int32_t col_count,
    const GP_TableStyle* style,
    const GP_TableRenderState* render_state,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom,
    GP_TableHitBox* hitboxes_out,
    int32_t hitboxes_cap,
    int32_t* hitboxes_written) {
    if (!rows || !cols || !out_rgba || row_count <= 0 || col_count <= 0) return 0;
    Style st = load_style(style);
    int w = st.w;
    int h = std::max(1, int(row_count) * st.row_h);
    // If caller provided an output geometry (e.g. canvas wants to render into
    // a specific module rect), honor its width/height as the target buffer
    // size. This allows embedding tables into arbitrary-sized module boxes
    // and ensures hitbox coordinates align with the provided buffer.
    if (out_geom && out_geom->width_px > 0 && out_geom->height_px > 0) {
        w = out_geom->width_px;
        h = out_geom->height_px;
    }
    // Ensure any content-driven sizing (e.g., image aspect) uses the actual target width.
    st.w = std::max(1, w);
    const int need = w * h * 4;
    if (out_len_bytes < need) return 0;

    const bool want_hits = hitboxes_out != nullptr && hitboxes_cap > 0;
    int hit_count = 0;
    auto push_hit = [&](int x0, int y0, int x1, int y1, int row_idx, int col_idx, GP_TableCellKind ck, GP_TableHitPart part, int aux0, int aux1, uint32_t flags) {
        if (!want_hits) return;
        if (hit_count >= hitboxes_cap) return;
        GP_TableHitBox hb{};
        hb.x0 = x0;
        hb.y0 = y0;
        hb.x1 = x1;
        hb.y1 = y1;
        hb.row_idx = row_idx;
        hb.col_idx = col_idx;
        hb.cell_kind = static_cast<int32_t>(ck);
        hb.part = static_cast<int32_t>(part);
        hb.aux0 = aux0;
        hb.aux1 = aux1;
        hb.flags = flags;
        hitboxes_out[hit_count++] = hb;
    };

    // Clear
    memset(out_rgba, 0, static_cast<std::size_t>(need));
    memset_rect(out_rgba, w, h, w * 4, 0, 0, w, h, st.bg);

    int col_x0[8] = {0};
    int col_w[8] = {0};
    compute_columns(cols, col_count, w, st.name_w, col_x0, col_w);
    std::vector<int> row_y0;
    std::vector<int> row_h;
    compute_row_layout(rows, int(row_count), st, h, row_y0, row_h);
    if (out_geom) {
        out_geom->width_px = w;
        out_geom->height_px = h;
        for (int i = 0; i < 8; ++i) {
            out_geom->col_x0[i] = col_x0[i];
            out_geom->col_w[i] = col_w[i];
        }
    }

    const int led_count = 9;
    for (int i = 0; i < row_count; ++i) {
        const GP_TableRow& r = rows[i];
        const GP_TableCell* img_cell = nullptr;
        const bool is_image_row = row_find_image_cell(r, &img_cell);
        const int header_h = is_image_row ? row_image_header_h(st, r) : 0;
        int y0 = row_y0[static_cast<size_t>(i)];
        int rh = row_h[static_cast<size_t>(i)];
        Color bg = (r.kind == GP_TABLE_ROW_HEADER) ? st.hdr : (r.selected ? st.bg_sel : st.bg);
        memset_rect(out_rgba, w, h, w * 4, 0, y0, w, rh, bg);

        // Expand box only on header/device rows to avoid clutter on leaf rows.
        int indent_px = st.indent * std::max(0, r.depth);
        if (r.kind == GP_TABLE_ROW_HEADER || r.kind == GP_TABLE_ROW_DEVICE) {
            int exp_x = 4 + indent_px;
            int band_h = is_image_row ? std::min(rh, header_h) : rh;
            int exp_y = y0 + (band_h - st.expand_w) / 2;
            draw_expand_box(out_rgba, w, h, w * 4, exp_x, exp_y, st.expand_w, r.expanded != 0, st.text);
            push_hit(exp_x, exp_y, exp_x + st.expand_w, exp_y + st.expand_w, i, -1, GP_TABLE_CELL_TEXT, GP_TABLE_HIT_EXPAND, 0, 0, 0);
        }
        // Label gutter still reserves name_w; callers overlay text as needed.

        // Cells
        int cell_band_h = is_image_row ? std::min(rh, header_h) : rh;
        int cy = y0 + cell_band_h / 2;

        // Image rows: render image under the header band, spanning the full content width.
        if (is_image_row && img_cell && img_cell->image && img_cell->image->rgba && r.expanded != 0 && rh > header_h) {
            const int max_img_h_default = 240;
            int max_img_h = (img_cell->reserved0 > 0) ? std::max(1, img_cell->reserved0) : max_img_h_default;
            int img_x0 = st.name_w + 2;
            int img_y0 = y0 + header_h;
            int img_w = std::max(1, w - st.name_w - 4);
            int img_h = std::max(0, std::min(rh - header_h, max_img_h));
            draw_image_cell(out_rgba, w, h, w * 4, img_x0, img_y0, img_w, img_h, img_cell->image);
        }
        for (int c = 0; c < r.cell_count && c < col_count && c < 8; ++c) {
            const GP_TableCell& cell = r.cells[c];
            int x0 = col_x0[c];
            int cw = col_w[c];
            if (cw <= 0) continue;
            int align = cols[c].align;

            // Whole-cell hitbox (moved to emit AFTER per-part hitboxes)
            // (previously emitted here which caused coarse hits to shadow per-LED hits)

            switch (cell.kind) {
                case GP_TABLE_CELL_LEDS: {
                    uint32_t on_mask = cell.flags;
                    uint32_t edge_mask = st.led_edge_mask ? (cell.flags & st.led_edge_mask) : 0u;
                    int eff_w = std::max(1, cw - 4);
                    int led_spacing = std::max(4, eff_w / std::max(1, led_count + 1));
                    int cx0 = x0 + 2 + led_spacing;
                    for (int li = 0; li < led_count; ++li) {
                        int cx = cx0 + li * led_spacing;
                        push_hit(cx - 6, cy - 6, cx + 6, cy + 6, i, c, GP_TABLE_CELL_LEDS, GP_TABLE_HIT_LED, li, 0, 0);
                    }
                    draw_led_strip(out_rgba, w, h, w * 4, x0 + 2, cy, eff_w, led_count, on_mask, edge_mask, on_mask, 4, st.led_on, st.led_off, st.led_edge, st.led_off);
                    break;
                }
                case GP_TABLE_CELL_LEDS_ARG: {
                    int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                    if (count == 0) count = (cell.flags & 0xFF);
                    if (count == 0) count = 12;
                    uint32_t linked_mask = cell.flags;
                    uint32_t required_mask = static_cast<uint32_t>(cell.reserved0);
                    if (required_mask == 0 && count > 0) required_mask = (count >= 32) ? 0xFFFFFFFFu : ((1u << count) - 1u);
                    int eff_w = std::max(1, cw - 4);
                    int led_spacing = std::max(4, eff_w / std::max(1, count + 1));
                    int cx0 = x0 + 2 + led_spacing;
                    for (int li = 0; li < count; ++li) {
                        int cx = cx0 + li * led_spacing;
                        uint32_t bit = 1u << li;
                        uint32_t flags = (required_mask & bit) ? 1u : 0u;
                        push_hit(cx - 6, cy - 6, cx + 6, cy + 6, i, c, GP_TABLE_CELL_LEDS_ARG, GP_TABLE_HIT_LED_ARG, li, 0, flags);
                    }
                    draw_led_strip(out_rgba, w, h, w * 4, x0 + 2, cy, eff_w, count, linked_mask, 0u, required_mask, 4, st.led_on, st.led_off, st.led_edge, st.axis_tick);
                    break;
                }
                case GP_TABLE_CELL_LEDS_TABLE: {
                    struct PackedStrip {
                        uint8_t count;
                        uint8_t reserved[3];
                        uint32_t on_mask;
                        uint32_t edge_mask;
                        uint32_t active_mask;
                    };
                    struct PackedTable {
                        uint8_t n;
                        uint8_t pad[3];
                        PackedStrip strips[3];
                    } pt{};
                    std::memcpy(&pt, cell.text, std::min<std::size_t>(sizeof(pt), sizeof(cell.text)));
                    int n = std::max(0, std::min<int>(pt.n, 3));
                    if (n <= 0) break;
                    int eff_w = std::max(1, cw - 4);
                    int slot_h = std::max(6, rh / std::max(1, n + 1));
                    for (int si = 0; si < n; ++si) {
                        const PackedStrip& ps = pt.strips[si];
                        int cy_slot = y0 + (rh * (si + 1)) / (n + 1);
                        int led_spacing = std::max(3 * 2 + 2, eff_w / std::max(1, int(ps.count) + 1));
                        int cx0 = x0 + 2 + led_spacing;
                        for (int li = 0; li < ps.count; ++li) {
                            int cx = cx0 + li * led_spacing;
                            push_hit(cx - 5, cy_slot - 5, cx + 5, cy_slot + 5, i, c, GP_TABLE_CELL_LEDS_TABLE, GP_TABLE_HIT_LED_TABLE, si, li, 0);
                        }
                    draw_led_strip(out_rgba, w, h, w * 4, x0 + 2, cy_slot, eff_w, std::max<int>(0, ps.count), ps.on_mask, ps.edge_mask, ps.active_mask, 3, st.led_on, st.led_off, st.led_edge, st.axis_tick);
                    }
                    break;
                }
                case GP_TABLE_CELL_IMAGE: {
                    // Image rows are rendered as full-row content under the header band (above).
                    if (!is_image_row) {
                        int eff_w = std::max(1, cw - 4);
                        int eff_h = std::max(1, rh - 2);
                        draw_image_cell(out_rgba, w, h, w * 4, x0 + 2, y0 + 1, eff_w, eff_h, cell.image);
                    }
                    break;
                }
                case GP_TABLE_CELL_COUNTER: {
                    int pad = 2;
                    int inner_w = std::max(1, cw - pad * 2);
                    int box_h = std::max(10, rh - 4);
                    int by = y0 + (rh - box_h) / 2;
                    int gap = std::max(4, box_h / 5);
                    int nbw = std::max(10, std::min(box_h, (inner_w - gap * 2) / 4));
                    int num_w = std::max(20, inner_w - nbw * 2 - gap * 2);
                    int total_w = nbw + gap + num_w + gap + nbw;
                    int bx_minus = x0 + pad + (inner_w - total_w) / 2;
                    int bx_num = bx_minus + nbw + gap;
                    int bx_plus = bx_num + num_w + gap;
                    memset_rect(out_rgba, w, h, w * 4, bx_minus, by, nbw, box_h, st.hdr);
                    memset_rect(out_rgba, w, h, w * 4, bx_num, by, num_w, box_h, st.axis_bg);
                    memset_rect(out_rgba, w, h, w * 4, bx_plus, by, nbw, box_h, st.hdr);
                    auto draw_centered = [&](const char* text, int x, int y, int bw, int bh, float scale, Color col) {
                        TextBitmap bm = render_text_to_rgba(text, scale, {col.r, col.g, col.b, col.a});
                        if (bm.pixels.empty()) return;
                        int tx = x + (bw - bm.width) / 2;
                        int ty = y + (bh - bm.height) / 2;
                        int clip_x0 = x;
                        int clip_y0 = y;
                        int clip_x1 = x + bw;
                        int clip_y1 = y + bh;
                        blit_text_bitmap(out_rgba, w, h, w * 4, bm, tx, ty, clip_x0, clip_y0, clip_x1, clip_y1);
                    };
                    draw_centered("-", bx_minus, by, nbw, box_h, 1.0f, st.text_hdr);
                    draw_centered("+", bx_plus, by, nbw, box_h, 1.0f, st.text_hdr);
                    int val = std::max(0, static_cast<int>(std::lround(cell.value)));
                    std::string num = std::to_string(val);
                    draw_centered(num.c_str(), bx_num, by, num_w, box_h, 1.0f, st.text);
                    if (cell.text[0] != '\0') {
                        TextBitmap lb = render_text_to_rgba(cell.text, 0.7f, {st.text_hdr.r, st.text_hdr.g, st.text_hdr.b, st.text_hdr.a});
                        if (!lb.pixels.empty()) {
                            int tx = x0 + (cw - lb.width) / 2;
                            int ty = std::max(y0 + 1, by - lb.height - 2);
                            int clip_x0 = x0 + 1;
                            int clip_y0 = y0 + 1;
                            int clip_x1 = x0 + cw - 1;
                            int clip_y1 = y0 + rh - 1;
                            blit_text_bitmap(out_rgba, w, h, w * 4, lb, tx, ty, clip_x0, clip_y0, clip_x1, clip_y1);
                        }
                    }
                    push_hit(bx_minus, by, bx_minus + nbw, by + box_h, i, c, GP_TABLE_CELL_COUNTER, GP_TABLE_HIT_COUNTER_DEC, 0, 0, 0);
                    push_hit(bx_plus, by, bx_plus + nbw, by + box_h, i, c, GP_TABLE_CELL_COUNTER, GP_TABLE_HIT_COUNTER_INC, 0, 0, 0);
                    break;
                }
                case GP_TABLE_CELL_SCROLL: {
                    float v = cell.value;            // 0..1 scroll fraction
                    int total = std::max(1, int(cell.hold_s));    // total rows/items
                    int visible = std::max(1, int(cell.last_s));  // visible rows/items
                    bool up_press = (cell.flags & 0x1u) != 0;
                    bool dn_press = (cell.flags & 0x2u) != 0;
                    int bar_w = std::max(1, cw - 2);
                    int arrow_h = std::max(8, (rh - 2) / 6);
                    draw_scrollbar(out_rgba, w, h, w * 4, x0 + 1, y0 + 1, bar_w, rh - 2, v, total, visible, up_press, dn_press, st.axis_bg, st.axis_val, st.text, st.text_hdr, st.bg_sel, st.bg);
                    // Hitboxes: arrows + thumb
                    push_hit(x0 + 1, y0 + 1, x0 + 1 + bar_w, y0 + 1 + arrow_h, i, c, GP_TABLE_CELL_SCROLL, GP_TABLE_HIT_SCROLL_UP, 0, 0, 0);
                    push_hit(x0 + 1, y0 + rh - 1 - arrow_h, x0 + 1 + bar_w, y0 + rh - 1, i, c, GP_TABLE_CELL_SCROLL, GP_TABLE_HIT_SCROLL_DOWN, 0, 0, 0);
                    int track_y0 = y0 + 1 + arrow_h;
                    int track_h = (rh - 2) - 2 * arrow_h;
                    if (track_h > 0) {
                        float vis_frac = std::clamp(float(visible) / float(std::max(visible, total)), 0.05f, 1.0f);
                        int thumb_h = std::max(6, int(vis_frac * track_h));
                        float clamped_v = std::clamp(v, 0.0f, 1.0f);
                        int thumb_y = track_y0 + int((track_h - thumb_h) * clamped_v);
                        push_hit(x0 + 1, thumb_y, x0 + 1 + bar_w, thumb_y + thumb_h, i, c, GP_TABLE_CELL_SCROLL, GP_TABLE_HIT_SCROLL_THUMB, 0, 0, 0);
                    }
                    break;
                }
                case GP_TABLE_CELL_AXIS: {
                    int bar_h = std::max(6, rh / 3);
                    int bar_y = y0 + (rh - bar_h) / 2;

                    // Optional extras: hold_s/min, last_s/max, calib params encoded as text.
                    float v_min = cell.hold_s;
                    float v_max = cell.last_s;
                    float cap_min = -1.0f;
                    float cap_max = 1.0f;
                    float trim = 0.0f;
                    float deadzone = 0.10f;
                    bool seen_min = (cell.flags & (1u << 0)) != 0;
                    bool seen_max = (cell.flags & (1u << 1)) != 0;
                    if ((cell.flags & 0x3u) == 0) {
                        // Default to "seen" so legacy callers render white end ticks.
                        seen_min = true;
                        seen_max = true;
                    }

                    if (cell.text[0] != '\0') {
                        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
                        if (std::sscanf(cell.text, "%f %f %f %f", &t0, &t1, &t2, &t3) == 4) {
                            cap_min = t0;
                            cap_max = t1;
                            trim = t2;
                            deadzone = std::max(0.0f, t3);
                        }
                    }

                    int bar_w = std::max(1, cw - 4);
                    draw_axis_bar_ext(
                        out_rgba,
                        w,
                        h,
                        w * 4,
                        x0 + 2,
                        bar_y,
                        bar_w,
                        bar_h,
                        cell.value,
                        v_min,
                        v_max,
                        cap_min,
                        cap_max,
                        trim,
                        deadzone,
                        seen_min,
                        seen_max,
                        st.axis_bg,
                        st.axis_tick,
                        st.axis_val);
                    push_hit(x0 + 2, bar_y, x0 + 2 + bar_w, bar_y + bar_h, i, c, GP_TABLE_CELL_AXIS, GP_TABLE_HIT_AXIS, 0, 0, 0);
                    break;
                }
                case GP_TABLE_CELL_CALIB: {
                    bool inv_on = (cell.flags & 0x1u) != 0;
                    int mode = int((cell.flags >> 1) & 0x3u); // 0:none,1:trim,2:cap,3:ded
                    int strip_h = std::max(6, rh / 3);
                    int strip_y = y0 + (rh - strip_h) / 2;
                    int eff_w = std::max(1, cw - 4);
                    draw_calib_strip(out_rgba, w, h, w * 4, x0 + 2, strip_y, eff_w, strip_h, inv_on, mode, st.axis_tick, st.timer, st.led_on);
                    // Slots: 0 INV, 1 TRIM, 2 CAP, 3 DED, 4 RST
                    int slots = 5;
                    int gap = 2;
                    int slot_w = std::max(4, (eff_w - gap * (slots - 1)) / slots);
                    int slot_y = strip_y;
                    int slot_h = strip_h;
                    for (int si = 0; si < slots; ++si) {
                        int sx = x0 + 2 + si * (slot_w + gap);
                        push_hit(sx, slot_y, sx + slot_w, slot_y + slot_h, i, c, GP_TABLE_CELL_CALIB, GP_TABLE_HIT_CALIB_SLOT, si, 0, 0);
                    }
                    break;
                }
                case GP_TABLE_CELL_TIMERS: {
                    int t_h = std::max(2, rh / 4);
                    int t_y = y0 + (rh - t_h) / 2;
                    draw_timers(out_rgba, w, h, w * 4, x0 + 2, t_y, std::max(1, cw - 4), t_h, cell.hold_s, cell.last_s, st.timer);
                    break;
                }
                case GP_TABLE_CELL_WAVE: {
                    int wh = std::min(rh - 4, cw);
                    int wy = y0 + (rh - wh) / 2;
                    draw_waveform(out_rgba, w, h, w * 4, x0 + 2, wy, std::max(1, cw - 4), wh, cell.wave, st.wave_bg, st.wave_fg);
                    break;
                }
                case GP_TABLE_CELL_TEXT:
                default:
                    if (cell.text[0] != '\0') {
                        Color tc = (r.kind == GP_TABLE_ROW_HEADER) ? st.text_hdr : st.text;
                        float scale = 0.9f;
                        if (rh <= 16) scale = 0.75f;
                        if (rh <= 12) scale = 0.65f;
                        TextBitmap bm = render_text_to_rgba(cell.text, scale, {tc.r, tc.g, tc.b, tc.a});
                        if (!bm.pixels.empty()) {
                            const int pad = 4;
                            int tx = x0 + pad;
                            if (align == 1) {
                                tx = x0 + (cw - bm.width) / 2;
                            } else if (align == 2) {
                                tx = x0 + cw - bm.width - pad;
                            }
                            int ty = y0 + (rh - bm.height) / 2;
                            int clip_x0 = x0 + 1;
                            int clip_x1 = x0 + cw - 1;
                            int clip_y0 = y0 + 1;
                            int clip_y1 = y0 + rh - 1;
                            blit_text_bitmap(out_rgba, w, h, w * 4, bm, tx, ty, clip_x0, clip_y0, clip_x1, clip_y1);
                        }
                    }
                    break;
            }
            // Emit whole-cell hitbox after all per-part hitboxes so small parts (LEDs)
            // are found before the coarse whole-cell region by hit iteration order.
            push_hit(x0, y0, x0 + cw, y0 + rh, i, c, static_cast<GP_TableCellKind>(cell.kind), GP_TABLE_HIT_CELL, 0, 0, 0);
        }
    }

    if (hitboxes_written) *hitboxes_written = hit_count;

    // Apply transient render-state overlays (highlights) if requested.
    if (render_state && out_rgba) {
        // Load style for layout math
        Style st_local = load_style(style);
        int w_local = st_local.w;
        int h_local = std::max(1, int(row_count) * st_local.row_h);
        int row_h_local = std::max(1, st_local.row_h);
        if (row_count > 0) row_h_local = std::max(1, h_local / row_count);
        int pitch_local = w_local * 4;

        // Determine highlight color
        Color hcol{255, 210, 90, 200};
        bool have_custom = false;
        if (render_state->highlight_color[0] || render_state->highlight_color[1] || render_state->highlight_color[2] || render_state->highlight_color[3]) {
            hcol = Color{render_state->highlight_color[0], render_state->highlight_color[1], render_state->highlight_color[2], render_state->highlight_color[3]};
            have_custom = true;
        }

        // helper to draw an outline rect
        auto draw_outline = [&](int rx0, int ry0, int rw, int rh, Color col){
            if (rw <= 0 || rh <= 0) return;
            memset_rect(out_rgba, w_local, h_local, pitch_local, rx0, ry0, rw, 1, col);
            memset_rect(out_rgba, w_local, h_local, pitch_local, rx0, ry0 + rh - 1, rw, 1, col);
            memset_rect(out_rgba, w_local, h_local, pitch_local, rx0, ry0, 1, rh, col);
            memset_rect(out_rgba, w_local, h_local, pitch_local, rx0 + rw - 1, ry0, 1, rh, col);
        };

        int col_x0[8] = {0};
        int col_w[8] = {0};
        compute_columns(cols, col_count, st_local.w, st_local.name_w, col_x0, col_w);

        if (render_state->highlight_row >= 0) {
            int r = render_state->highlight_row;
            if (r >= 0 && r < row_count) {
                int y0 = r * row_h_local;
                // whole-cell highlight if col not provided
                if (render_state->highlight_col < 0) {
                    draw_outline(0, y0, w_local, row_h_local, hcol);
                } else {
                    int c = render_state->highlight_col;
                    if (c >= 0 && c < col_count) {
                        int x0 = col_x0[c];
                        int cw = col_w[c];
                        // Decide based on highlighted part
                        int part = render_state->highlight_part;
                        if (part == GP_TABLE_HIT_LED || part == GP_TABLE_HIT_LED_ARG || part == GP_TABLE_HIT_LED_TABLE) {
                            // compute LED positions similar to raster
                            const GP_TableRow &row = rows[r];
                            const GP_TableCell &cell = row.cells[c];
                            int led_count = 9;
                            if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
                                int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                                if (count == 0) count = (cell.flags & 0xFF);
                                if (count == 0) count = 12;
                                led_count = count;
                            } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
                                // best-effort: use 8 as fallback
                                led_count = 8;
                            }
                            int eff_w = std::max(1, cw - 4);
                            int radius = 4;
                            int led_spacing = std::max(radius * 2 + 2, eff_w / std::max(1, led_count + 1));
                            int cx0 = x0 + 2 + led_spacing;
                            int li = render_state->highlight_aux0;
                            if (li >= 0 && li < led_count) {
                                int cx = cx0 + li * led_spacing;
                                int cy = y0 + row_h_local / 2;
                                // slightly larger ring for highlight
                                draw_circle(out_rgba, w_local, h_local, pitch_local, cx, cy, radius + 2, hcol);
                            }
                        } else if (part == GP_TABLE_HIT_CALIB_SLOT) {
                            // compute calib slot rects similar to draw_calib_strip assumptions
                            int slots = 5;
                            int gap = 2;
                            int eff_w = std::max(1, cw - 4);
                            int slot_w = std::max(4, (eff_w - gap * (slots - 1)) / slots);
                            int sy = y0 + (row_h_local - slot_w) / 2; // approximate
                            int si = render_state->highlight_aux0;
                            if (si >= 0 && si < slots) {
                                int sx = x0 + 2 + si * (slot_w + gap);
                                draw_outline(sx, y0, slot_w, row_h_local, hcol);
                            }
                        } else if (part == GP_TABLE_HIT_SCROLL_THUMB) {
                            draw_outline(x0, y0, cw, row_h_local, hcol);
                        } else {
                            // default: whole-cell outline
                            draw_outline(x0, y0, cw, st_local.row_h, hcol);
                        }
                    }
                }
            }
        } else if (render_state->mouse_x >= 0 && render_state->mouse_y >= 0) {
            // mouse-based highlight: find hitbox under mouse if hitboxes were emitted
            if (hitboxes_out && hitboxes_cap > 0) {
                int mx = render_state->mouse_x;
                int my = render_state->mouse_y;
                for (int hi = 0; hi < hit_count; ++hi) {
                    const GP_TableHitBox &hb = hitboxes_out[hi];
                    if (mx >= hb.x0 && mx < hb.x1 && my >= hb.y0 && my < hb.y1) {
                        draw_outline(hb.x0, hb.y0, hb.x1 - hb.x0, hb.y1 - hb.y0, hcol);
                        break;
                    }
                }
            }
        }
    }

    return 1;
}

int32_t gp_table_raster_rgba(
    const GP_TableRow* rows,
    int32_t row_count,
    const GP_TableColumn* cols,
    int32_t col_count,
    const GP_TableStyle* style,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom) {
    return gp_table_raster_rgba_with_hits(rows, row_count, cols, col_count, style, /*render_state=*/nullptr, out_rgba, out_len_bytes, out_geom, nullptr, 0, nullptr);
}

struct StagePortBinding {
    GP_StageContext* stage = nullptr;
    int is_output = 0; // 1=output port, 0=input port
    int channel = 0;   // optional stage-local channel index
};

struct EdgeTensorFifo {
    static constexpr size_t kMaxReaders = 64;
    static constexpr int32_t kMaxOrder = 4;

    struct ReaderEntry {
        std::atomic<uint64_t> key{0};
        std::atomic<uint64_t> seq{0}; // next sequence number to read
    };

    struct Impl {
        std::atomic<uint64_t> write_seq{0}; // next sequence to write
        std::atomic<uint64_t> writer{0};    // bound writer key (0 => unbound)
        std::unique_ptr<std::atomic<uint64_t>[]> slot_seq; // published tag per slot (seq+1), 0 => empty
        std::unique_ptr<uint8_t[]> storage_bytes;          // raw bytes: slots * stride * sizeof(float)
        float* storage_f = nullptr;                        // typed view for float samples
        std::unique_ptr<ReaderEntry[]> readers;
        size_t stride = 1;
        size_t slots = 1;
        size_t top_k = 0;
        std::atomic<float> write_friction{0.0f};
        std::atomic<float> read_friction{0.0f};
        std::atomic<uint64_t> last_write_seq{0};
        std::atomic<uint64_t> last_read_seq{0};
        std::atomic<int32_t> last_write_region{-1};
        std::atomic<int32_t> last_read_region{-1};
        std::atomic<float> write_phase{0.0f};
        std::atomic<float> read_phase{0.0f};
        std::atomic<int32_t> friction_regions{8};
        std::mutex cv_mu;
        std::condition_variable cv;
        bool configured = false;

        Impl() : readers(new ReaderEntry[kMaxReaders]) {}
    };

    std::vector<int32_t> shape;
    std::unique_ptr<Impl> impl;
    std::vector<float> last_sample;
    bool last_sample_valid = false;
    bool delta_mode = false;
    std::vector<float> order_history;
    std::vector<float> order_integrator;
    std::vector<float> scratch;
    int32_t order_mode = 0;
    int32_t order_history_count = 0;
    int32_t order_history_cursor = 0;

    EdgeTensorFifo() : impl(new Impl()) {}
    EdgeTensorFifo(EdgeTensorFifo&&) noexcept = default;
    EdgeTensorFifo& operator=(EdgeTensorFifo&&) noexcept = default;
    EdgeTensorFifo(const EdgeTensorFifo&) = delete;
    EdgeTensorFifo& operator=(const EdgeTensorFifo&) = delete;

    void configure_default() { configure(std::vector<int32_t>{1}, 16, 0); }

    void configure(const std::vector<int32_t>& dims, size_t slot_count, size_t topk) {
        if (!impl) impl.reset(new Impl());
        shape = dims;
        if (shape.empty()) shape.push_back(1);
        size_t stride_local = 1;
        for (int32_t d : shape) {
            stride_local *= static_cast<size_t>(std::max<int32_t>(1, d));
        }
        impl->stride = std::max<size_t>(1, stride_local);
        impl->slots = std::max<size_t>(1, slot_count);
        impl->top_k = topk;

        last_sample.assign(impl->stride, 0.0f);
        last_sample_valid = false;
        order_history.assign(static_cast<size_t>(kMaxOrder) * impl->stride, 0.0f);
        order_integrator.assign(static_cast<size_t>(kMaxOrder) * impl->stride, 0.0f);
        scratch.assign(impl->stride, 0.0f);
        order_mode = 0;
        order_history_count = 0;
        order_history_cursor = 0;

        impl->storage_bytes.reset(new uint8_t[impl->stride * impl->slots * sizeof(float)]);
        impl->storage_f = reinterpret_cast<float*>(impl->storage_bytes.get());
        impl->slot_seq.reset(new std::atomic<uint64_t>[impl->slots]);
        std::memset(impl->storage_bytes.get(), 0, static_cast<size_t>(impl->stride * impl->slots * sizeof(float)));
        for (size_t i = 0; i < impl->slots; ++i) impl->slot_seq[i].store(0, std::memory_order_relaxed);
        impl->write_seq.store(0, std::memory_order_relaxed);
        impl->writer.store(0, std::memory_order_relaxed);
        impl->write_friction.store(0.0f, std::memory_order_relaxed);
        impl->read_friction.store(0.0f, std::memory_order_relaxed);
        impl->last_write_seq.store(0, std::memory_order_relaxed);
        impl->last_read_seq.store(0, std::memory_order_relaxed);
        impl->last_write_region.store(-1, std::memory_order_relaxed);
        impl->last_read_region.store(-1, std::memory_order_relaxed);
        impl->write_phase.store(0.0f, std::memory_order_relaxed);
        impl->read_phase.store(0.0f, std::memory_order_relaxed);
        for (size_t i = 0; i < kMaxReaders; ++i) {
            impl->readers[i].key.store(0, std::memory_order_relaxed);
            impl->readers[i].seq.store(0, std::memory_order_relaxed);
        }
        impl->configured = true;
    }

    size_t elem_count() const { return impl ? impl->stride : 0; }

    void set_delta_mode(bool enabled) { delta_mode = enabled; }
    void set_order_mode(int32_t mode) {
        order_mode = std::clamp(mode, -kMaxOrder, kMaxOrder);
        order_history_count = 0;
        order_history_cursor = 0;
        std::fill(order_history.begin(), order_history.end(), 0.0f);
        std::fill(order_integrator.begin(), order_integrator.end(), 0.0f);
        last_sample_valid = false;
    }

    void maybe_claim_writer(uint64_t key) {
        if (!impl) return;
        uint64_t prev = impl->writer.load(std::memory_order_relaxed);
        if (prev == 0 || prev == key) {
            (void)impl->writer.compare_exchange_strong(prev, key, std::memory_order_relaxed);
        }
    }

    ReaderEntry* find_reader(uint64_t key) {
        if (!impl) return nullptr;
        for (size_t i = 0; i < kMaxReaders; ++i) {
            if (impl->readers[i].key.load(std::memory_order_relaxed) == key) return &impl->readers[i];
        }
        return nullptr;
    }

    void set_friction_regions(int32_t regions) {
        if (!impl) return;
        int32_t clamped = std::max(1, regions);
        impl->friction_regions.store(clamped, std::memory_order_relaxed);
    }

    bool subscribe(uint64_t key, bool start_at_head) {
        if (!impl) return false;
        if (key == 0) return false;
        if (find_reader(key)) return true;
        for (size_t i = 0; i < kMaxReaders; ++i) {
            uint64_t expected = 0;
            if (!impl->readers[i].key.compare_exchange_strong(expected, key, std::memory_order_acq_rel)) continue;
            uint64_t head = impl->write_seq.load(std::memory_order_acquire);
            uint64_t start = head;
            if (!start_at_head) {
                uint64_t cap = static_cast<uint64_t>(impl->slots);
                start = (head > cap) ? (head - cap) : 0;
            }
            impl->readers[i].seq.store(start, std::memory_order_release);
            uint64_t prev = impl->last_read_seq.load(std::memory_order_relaxed);
            if (prev == 0) {
                impl->last_read_seq.store(start, std::memory_order_relaxed);
                if (impl->slots > 0) {
                    float phase = static_cast<float>(start % static_cast<uint64_t>(impl->slots)) / static_cast<float>(impl->slots);
                    impl->read_phase.store(phase, std::memory_order_relaxed);
                }
            }
            impl->cv.notify_all();
            return true;
        }
        return false;
    }

    void unsubscribe(uint64_t key) {
        if (!impl || key == 0) return;
        for (size_t i = 0; i < kMaxReaders; ++i) {
            if (impl->readers[i].key.load(std::memory_order_relaxed) != key) continue;
            impl->readers[i].key.store(0, std::memory_order_release);
            impl->cv.notify_all();
            return;
        }
    }

    uint64_t min_reader_seq(uint64_t head) const {
        if (!impl) return head;
        uint64_t m = head;
        bool any = false;
        for (size_t i = 0; i < kMaxReaders; ++i) {
            if (impl->readers[i].key.load(std::memory_order_relaxed) == 0) continue;
            uint64_t s = impl->readers[i].seq.load(std::memory_order_relaxed);
            m = std::min<uint64_t>(m, s);
            any = true;
        }
        return any ? m : head;
    }

    static float atomic_add(std::atomic<float>& v, float delta) {
        float cur = v.load(std::memory_order_relaxed);
        while (!v.compare_exchange_weak(cur, cur + delta, std::memory_order_relaxed)) {}
        return cur + delta;
    }

    static void atomic_scale(std::atomic<float>& v, float factor) {
        float cur = v.load(std::memory_order_relaxed);
        while (!v.compare_exchange_weak(cur, cur * factor, std::memory_order_relaxed)) {}
    }

    void note_activity(uint64_t seq,
                       std::atomic<uint64_t>& last_seq,
                       std::atomic<int32_t>& last_region,
                       std::atomic<float>& friction,
                       std::atomic<float>& phase) {
        if (!impl || impl->slots == 0) return;
        uint64_t prev = last_seq.exchange(seq, std::memory_order_relaxed);
        if (seq <= prev) return;
        uint64_t delta = seq - prev;
        float inc = static_cast<float>(delta) / static_cast<float>(impl->slots);
        uint64_t slot = seq % static_cast<uint64_t>(impl->slots);
        int32_t regions = impl->friction_regions.load(std::memory_order_relaxed);
        if (regions > 0) {
            int32_t region = static_cast<int32_t>((slot * static_cast<uint64_t>(regions)) / static_cast<uint64_t>(impl->slots));
            int32_t prev_region = last_region.exchange(region, std::memory_order_relaxed);
            if (prev_region >= 0 && prev_region != region) {
                int32_t diff = std::abs(region - prev_region);
                if (diff > regions / 2) diff = regions - diff;
                inc += 2.0f * static_cast<float>(std::max(1, diff));
            }
        }
        float phase_val = static_cast<float>(slot) / static_cast<float>(impl->slots);
        phase.store(phase_val, std::memory_order_relaxed);
        atomic_add(friction, inc);
    }

    void tick_friction(float dt, float half_life) {
        if (!impl) return;
        if (half_life <= 0.0f || dt <= 0.0f) return;
        float factor = std::pow(0.5f, dt / half_life);
        atomic_scale(impl->write_friction, factor);
        atomic_scale(impl->read_friction, factor);
    }

    float write_friction() const { return impl ? impl->write_friction.load(std::memory_order_relaxed) : 0.0f; }
    float read_friction() const { return impl ? impl->read_friction.load(std::memory_order_relaxed) : 0.0f; }
    float write_phase() const { return impl ? impl->write_phase.load(std::memory_order_relaxed) : 0.0f; }
    float read_phase() const { return impl ? impl->read_phase.load(std::memory_order_relaxed) : 0.0f; }
    uint64_t write_seq() const { return impl ? impl->write_seq.load(std::memory_order_relaxed) : 0; }

    float phase_delta() const {
        if (!impl) return 0.0f;
        float w = impl->write_phase.load(std::memory_order_relaxed);
        float r = impl->read_phase.load(std::memory_order_relaxed);
        float delta = w - r;
        if (delta > 0.5f) delta -= 1.0f;
        if (delta < -0.5f) delta += 1.0f;
        return delta;
    }

    bool fill_state(uint64_t edge_id, float& out_fill, float& out_head_phase, float& out_tail_phase) const {
        out_fill = 0.0f;
        out_head_phase = 0.0f;
        out_tail_phase = 0.0f;
        if (!impl || impl->slots == 0) return false;
        uint64_t head = impl->write_seq.load(std::memory_order_acquire);
        uint64_t min_seq = min_reader_seq(head);
        if (edge_id != 0) {
            ThreadManager* tm = ThreadManager::global();
            if (tm) {
                uint64_t mgr_min = tm->min_reader_seq_for_edge(edge_id);
                if (mgr_min != UINT64_MAX) min_seq = mgr_min;
            }
        }
        uint64_t used = (head >= min_seq) ? (head - min_seq) : 0;
        float fill = static_cast<float>(used) / static_cast<float>(impl->slots);
        out_fill = std::clamp(fill, 0.0f, 1.0f);
        out_head_phase = static_cast<float>(head % static_cast<uint64_t>(impl->slots)) / static_cast<float>(impl->slots);
        out_tail_phase = static_cast<float>(min_seq % static_cast<uint64_t>(impl->slots)) / static_cast<float>(impl->slots);
        return true;
    }

    bool is_full(uint64_t seq, uint64_t edge_id = 0) const {
        if (!impl) return false;
        uint64_t head = impl->write_seq.load(std::memory_order_acquire);
        uint64_t min_seq = min_reader_seq(head);
        if (edge_id != 0) {
            ThreadManager* tm = ThreadManager::global();
            if (tm) {
                uint64_t mgr_min = tm->min_reader_seq_for_edge(edge_id);
                if (mgr_min != UINT64_MAX) min_seq = mgr_min;
            }
        }
        uint64_t used = (seq >= min_seq) ? (seq - min_seq) : 0;
        return used >= static_cast<uint64_t>(impl->slots);
    }

    bool ensure_space_for_write(uint64_t seq, bool* out_dropped, uint64_t edge_id = 0) {
        if (!impl) return false;
        if (out_dropped) *out_dropped = false;
        uint64_t head = impl->write_seq.load(std::memory_order_acquire);
        uint64_t min_seq = min_reader_seq(head);
        // If a ThreadManager is present and an edge id is supplied, consult
        // its view of the minimum reader sequence for this edge. Fall back
        // to the local reader table if the manager has no info.
        if (edge_id != 0) {
            ThreadManager* tm = ThreadManager::global();
            if (tm) {
                uint64_t mgr_min = tm->min_reader_seq_for_edge(edge_id);
                if (mgr_min != UINT64_MAX) min_seq = mgr_min;
            }
        }
        uint64_t used = (seq >= min_seq) ? (seq - min_seq) : 0;
        if (used < static_cast<uint64_t>(impl->slots)) return true;
        if (impl->top_k == 0) return false;
        uint64_t window = std::min<uint64_t>(static_cast<uint64_t>(impl->top_k), static_cast<uint64_t>(impl->slots));
        uint64_t target_min = (window > 0 && seq >= (window - 1)) ? (seq - (window - 1)) : 0;
        if (target_min <= min_seq) return false;
        for (size_t i = 0; i < kMaxReaders; ++i) {
            if (impl->readers[i].key.load(std::memory_order_relaxed) == 0) continue;
            uint64_t s = impl->readers[i].seq.load(std::memory_order_relaxed);
            if (s < target_min) impl->readers[i].seq.store(target_min, std::memory_order_relaxed);
        }
        if (out_dropped) *out_dropped = true;
        min_seq = min_reader_seq(head);
        used = (seq >= min_seq) ? (seq - min_seq) : 0;
        return used < static_cast<uint64_t>(impl->slots);
    }

    bool push(uint64_t edge_id, uint64_t writer_id, const float* sample, size_t sample_len, bool* out_dropped) {
        if (!impl || !impl->configured) return false;
        if (!sample) return false;
        if (sample_len != impl->stride) return false;
        uint64_t bound = impl->writer.load(std::memory_order_relaxed);
        if (bound == 0) {
            (void)impl->writer.compare_exchange_strong(bound, writer_id, std::memory_order_relaxed);
        }
        bound = impl->writer.load(std::memory_order_relaxed);
        if (bound != 0 && bound != writer_id) return false;

        const float* effective_sample = sample;
        if (order_mode != 0 && scratch.size() == impl->stride) {
            int32_t mode = std::clamp(order_mode, -kMaxOrder, kMaxOrder);
            if (mode > 0) {
                for (size_t i = 0; i < impl->stride; ++i) {
                    float acc = sample[i];
                    order_integrator[i] += acc;
                    for (int32_t level = 1; level < mode; ++level) {
                        size_t idx = static_cast<size_t>(level) * impl->stride + i;
                        size_t prev = static_cast<size_t>(level - 1) * impl->stride + i;
                        order_integrator[idx] += order_integrator[prev];
                    }
                    scratch[i] = order_integrator[static_cast<size_t>(mode - 1) * impl->stride + i];
                }
            } else {
                int32_t order = -mode;
                if (order_history_count >= order) {
                    for (size_t i = 0; i < impl->stride; ++i) {
                        float sum = sample[i];
                        int32_t coef = 1;
                        for (int32_t k = 1; k <= order; ++k) {
                            coef = (coef * (order - (k - 1))) / k;
                            int32_t idx = order_history_cursor - k;
                            if (idx < 0) idx += kMaxOrder;
                            float prev = order_history[static_cast<size_t>(idx) * impl->stride + i];
                            float sign = (k % 2 == 0) ? 1.0f : -1.0f;
                            sum += sign * static_cast<float>(coef) * prev;
                        }
                        scratch[i] = sum;
                    }
                } else {
                    std::memcpy(scratch.data(), sample, impl->stride * sizeof(float));
                }
            }
            effective_sample = scratch.data();
        }

        if (delta_mode && last_sample_valid && last_sample.size() == impl->stride) {
            if (std::memcmp(last_sample.data(), effective_sample, impl->stride * sizeof(float)) == 0) {
                if (out_dropped) *out_dropped = 0;
                return true;
            }
        }

        uint64_t seq = impl->write_seq.load(std::memory_order_relaxed);
        bool dropped = false;
        if (!ensure_space_for_write(seq, &dropped, edge_id)) {
            if (out_dropped) *out_dropped = 1;
            return false;
        }

        size_t slot = static_cast<size_t>(seq % static_cast<uint64_t>(impl->slots));
        float* dst = impl->storage_f + slot * impl->stride;
        std::memcpy(dst, effective_sample, impl->stride * sizeof(float));
        impl->slot_seq[slot].store(seq + 1, std::memory_order_release);
        impl->write_seq.store(seq + 1, std::memory_order_release);
        note_activity(seq + 1, impl->last_write_seq, impl->last_write_region, impl->write_friction, impl->write_phase);
        if (!order_history.empty()) {
            size_t base = static_cast<size_t>(order_history_cursor) * impl->stride;
            std::memcpy(order_history.data() + base, sample, impl->stride * sizeof(float));
            order_history_cursor = (order_history_cursor + 1) % kMaxOrder;
            order_history_count = std::min(order_history_count + 1, kMaxOrder);
        }
        if (last_sample.size() == impl->stride) {
            std::memcpy(last_sample.data(), effective_sample, impl->stride * sizeof(float));
            last_sample_valid = true;
        }
        impl->cv.notify_all();
        if (out_dropped) *out_dropped = dropped ? 1 : 0;
        return true;
    }

    // Pointer-mode push: publish opaque pointer values into FIFO slots.
    bool push_ptr(uint64_t edge_id, uint64_t writer_id, void* ptr, bool* out_dropped) {
        if (!impl || !impl->configured) return false;
        uint64_t bound = impl->writer.load(std::memory_order_relaxed);
        if (bound == 0) {
            (void)impl->writer.compare_exchange_strong(bound, writer_id, std::memory_order_relaxed);
        }
        bound = impl->writer.load(std::memory_order_relaxed);
        if (bound != 0 && bound != writer_id) return false;

        uint64_t seq = impl->write_seq.load(std::memory_order_relaxed);
        bool dropped = false;
        if (!ensure_space_for_write(seq, &dropped, edge_id)) {
            if (out_dropped) *out_dropped = 1;
            return false;
        }

        size_t slot = static_cast<size_t>(seq % static_cast<uint64_t>(impl->slots));
        // Write the pointer bytes into the slot's byte storage so pointers travel
        // through the same stride-based FIFO storage as float samples. The
        // consumer is expected to interpret the slot according to the edge
        // metadata (pointer vs float).
        uint8_t* base = impl->storage_bytes.get() + slot * impl->stride * sizeof(float);
        // Zero the slot first to avoid leaving stale bytes in trailing area
        std::memset(base, 0, impl->stride * sizeof(float));
        std::memcpy(base, &ptr, std::min<size_t>(sizeof(void*), impl->stride * sizeof(float)));
        impl->slot_seq[slot].store(seq + 1, std::memory_order_release);
        impl->write_seq.store(seq + 1, std::memory_order_release);
        note_activity(seq + 1, impl->last_write_seq, impl->last_write_region, impl->write_friction, impl->write_phase);
        impl->cv.notify_all();
        if (out_dropped) *out_dropped = dropped ? 1 : 0;
        return true;
    }

    bool pop_ptr(uint64_t reader_id, void** out_ptr) {
        if (!impl || !impl->configured) return false;
        if (!out_ptr) return false;
        ReaderEntry* r = find_reader(reader_id);
        if (!r) return false;

        uint64_t rseq = r->seq.load(std::memory_order_relaxed);
        size_t slot = static_cast<size_t>(rseq % static_cast<uint64_t>(impl->slots));
        uint64_t observed = impl->slot_seq[slot].load(std::memory_order_acquire);
        if (observed != (rseq + 1)) return false;

        // Read pointer bytes from the slot's storage and return as opaque pointer.
        uint8_t* base = impl->storage_bytes.get() + slot * impl->stride * sizeof(float);
        void* p = nullptr;
        std::memcpy(&p, base, std::min<size_t>(sizeof(void*), impl->stride * sizeof(float)));
        *out_ptr = p;
        r->seq.store(rseq + 1, std::memory_order_relaxed);
        note_activity(rseq + 1, impl->last_read_seq, impl->last_read_region, impl->read_friction, impl->read_phase);
        impl->cv.notify_all();
        return true;
    }

    // Non-destructive peek for pointer slots: read opaque pointer bytes
    // without advancing the reader sequence. Returns true if a pointer
    // was available and copied into out_ptr.
    bool peek_ptr(uint64_t reader_id, void** out_ptr) {
        if (!impl || !impl->configured) return false;
        if (!out_ptr) return false;
        ReaderEntry* r = find_reader(reader_id);
        if (!r) return false;

        uint64_t rseq = r->seq.load(std::memory_order_relaxed);
        size_t slot = static_cast<size_t>(rseq % static_cast<uint64_t>(impl->slots));
        uint64_t observed = impl->slot_seq[slot].load(std::memory_order_acquire);
        if (observed != (rseq + 1)) return false;

        uint8_t* base = impl->storage_bytes.get() + slot * impl->stride * sizeof(float);
        void* p = nullptr;
        std::memcpy(&p, base, std::min<size_t>(sizeof(void*), impl->stride * sizeof(float)));
        *out_ptr = p;
        return true;
    }

    // Blocking pointer pop: wait until a pointer entry is available then
    // read and advance the reader. Returns true on success.
    bool pop_ptr_blocking(uint64_t reader_id, void** out_ptr, int timeout_ms) {
        if (!impl || !impl->configured) return false;
        if (!find_reader(reader_id)) return false;
        if (!out_ptr) return false;
        if (timeout_ms == 0) return pop_ptr(reader_id, out_ptr);
        using clock = std::chrono::steady_clock;
        auto deadline = (timeout_ms < 0) ? clock::time_point::max() : (clock::now() + std::chrono::milliseconds(timeout_ms));
        std::unique_lock<std::mutex> lk(impl->cv_mu);
        while (true) {
            lk.unlock();
            bool ok = pop_ptr(reader_id, out_ptr);
            lk.lock();
            if (ok) return true;
            if (timeout_ms < 0) {
                impl->cv.wait(lk);
            } else {
                if (impl->cv.wait_until(lk, deadline) == std::cv_status::timeout) return false;
            }
        }
    }

    bool pop(uint64_t reader_id, float* out_sample, size_t out_cap, size_t& out_written) {
        out_written = 0;
        if (!impl || !impl->configured) return false;
        ReaderEntry* r = find_reader(reader_id);
        if (!r) return false;
        if (!out_sample || out_cap < impl->stride) return false;

        uint64_t rseq = r->seq.load(std::memory_order_relaxed);
        size_t slot = static_cast<size_t>(rseq % static_cast<uint64_t>(impl->slots));
        uint64_t observed = impl->slot_seq[slot].load(std::memory_order_acquire);
        if (observed != (rseq + 1)) return false;

        const float* src = impl->storage_f + slot * impl->stride;
        std::memcpy(out_sample, src, impl->stride * sizeof(float));
        r->seq.store(rseq + 1, std::memory_order_relaxed);
        note_activity(rseq + 1, impl->last_read_seq, impl->last_read_region, impl->read_friction, impl->read_phase);
        impl->cv.notify_all();
        out_written = impl->stride;
        return true;
    }

    // Non-destructive peek: copy the next available sample for `reader_id`
    // into `out_sample` without advancing the reader sequence. Returns true
    // if a sample was available and copied. Contract matches `pop` with the
    // requirement that out_cap >= impl->stride.
    bool peek(uint64_t reader_id, float* out_sample, size_t out_cap, size_t& out_written) {
        out_written = 0;
        if (!impl || !impl->configured) return false;
        ReaderEntry* r = find_reader(reader_id);
        if (!r) return false;
        if (!out_sample || out_cap < impl->stride) return false;

        uint64_t rseq = r->seq.load(std::memory_order_relaxed);
        size_t slot = static_cast<size_t>(rseq % static_cast<uint64_t>(impl->slots));
        uint64_t observed = impl->slot_seq[slot].load(std::memory_order_acquire);
        if (observed != (rseq + 1)) return false;

        const float* src = impl->storage_f + slot * impl->stride;
        std::memcpy(out_sample, src, impl->stride * sizeof(float));
        // Note: do NOT advance r->seq and do NOT call note_activity / notify.
        out_written = impl->stride;
        return true;
    }

    bool push_blocking(uint64_t edge_id, uint64_t writer_id, const float* sample, size_t sample_len, bool* out_dropped, int timeout_ms) {
        if (!impl || !impl->configured) return false;
        if (!sample || sample_len != impl->stride) return false;
        uint64_t bound = impl->writer.load(std::memory_order_relaxed);
        if (bound != 0 && bound != writer_id) return false;
        if (timeout_ms == 0) return push(edge_id, writer_id, sample, sample_len, out_dropped);
        using clock = std::chrono::steady_clock;
        auto deadline = (timeout_ms < 0) ? clock::time_point::max() : (clock::now() + std::chrono::milliseconds(timeout_ms));
        std::unique_lock<std::mutex> lk(impl->cv_mu);
        while (true) {
            lk.unlock();
            bool ok = push(edge_id, writer_id, sample, sample_len, out_dropped);
            lk.lock();
            if (ok) return true;
            uint64_t seq = impl->write_seq.load(std::memory_order_relaxed);
            if (!is_full(seq, edge_id)) return false;
            if (timeout_ms < 0) {
                impl->cv.wait(lk);
            } else {
                if (impl->cv.wait_until(lk, deadline) == std::cv_status::timeout) return false;
            }
        }
    }

    // Ensure the FIFO stride (in bytes) can accomodate at least `min_bytes` per-slot.
    // If the current stride is too small we grow the stride and re-layout existing
    // per-slot contents into the new stride. This attempts to preserve outstanding
    // samples so FIFOs can be reconfigured (e.g. when an edge becomes by-ref)
    // without destroying the FIFO object.
    bool ensure_stride_for_bytes(size_t min_bytes) {
        if (!impl || !impl->configured) return false;
        size_t cur_bytes = impl->stride * sizeof(float);
        if (cur_bytes >= min_bytes) return true;
        size_t needed_floats = (min_bytes + sizeof(float) - 1) / sizeof(float);
        size_t new_stride = std::max(impl->stride, needed_floats);

        // allocate new storage
        std::unique_ptr<uint8_t[]> new_bytes(new uint8_t[new_stride * impl->slots * sizeof(float)]);
        std::memset(new_bytes.get(), 0, new_stride * impl->slots * sizeof(float));

        // copy per-slot existing float bytes into the new layout (preserve min region)
        for (size_t s = 0; s < impl->slots; ++s) {
            uint8_t* src = impl->storage_bytes.get() + s * impl->stride * sizeof(float);
            uint8_t* dst = new_bytes.get() + s * new_stride * sizeof(float);
            size_t copy_bytes = impl->stride * sizeof(float);
            std::memcpy(dst, src, copy_bytes);
        }

        // swap in new storage and update typed view
        impl->storage_bytes.swap(new_bytes);
        impl->storage_f = reinterpret_cast<float*>(impl->storage_bytes.get());
        impl->stride = new_stride;

        // update our cached scratch/last_sample sizes to match new stride
        last_sample.assign(impl->stride, 0.0f);
        scratch.assign(impl->stride, 0.0f);
        order_history.assign(static_cast<size_t>(kMaxOrder) * impl->stride, 0.0f);
        order_integrator.assign(static_cast<size_t>(kMaxOrder) * impl->stride, 0.0f);
        return true;
    }

    bool pop_blocking(uint64_t reader_id, float* out_sample, size_t out_cap, size_t& out_written, int timeout_ms) {
        if (!impl || !impl->configured) return false;
        if (!find_reader(reader_id)) return false;
        if (!out_sample || out_cap < impl->stride) return false;
        if (timeout_ms == 0) return pop(reader_id, out_sample, out_cap, out_written);
        using clock = std::chrono::steady_clock;
        auto deadline = (timeout_ms < 0) ? clock::time_point::max() : (clock::now() + std::chrono::milliseconds(timeout_ms));
        std::unique_lock<std::mutex> lk(impl->cv_mu);
        while (true) {
            lk.unlock();
            bool ok = pop(reader_id, out_sample, out_cap, out_written);
            lk.lock();
            if (ok) return true;
            if (timeout_ms < 0) {
                impl->cv.wait(lk);
            } else {
                if (impl->cv.wait_until(lk, deadline) == std::cv_status::timeout) return false;
            }
        }
    }

    uint64_t unread(uint64_t reader_id) const {
        if (!impl || !impl->configured) return 0;
        const ReaderEntry* r = nullptr;
        for (size_t i = 0; i < kMaxReaders; ++i) {
            if (impl->readers[i].key.load(std::memory_order_relaxed) == reader_id) { r = &impl->readers[i]; break; }
        }
        if (!r) return 0;
        uint64_t head = impl->write_seq.load(std::memory_order_acquire);
        uint64_t tail = r->seq.load(std::memory_order_relaxed);
        return (head >= tail) ? (head - tail) : 0;
    }

    GP_TableEdgeTensorSpec to_spec() const {
        GP_TableEdgeTensorSpec s{};
        s.dim_count = static_cast<int32_t>(std::min<size_t>(shape.size(), sizeof(s.dims) / sizeof(s.dims[0])));
        for (int32_t i = 0; i < s.dim_count; ++i) s.dims[i] = shape[static_cast<size_t>(i)];
        s.slots = impl ? static_cast<int32_t>(impl->slots) : 0;
        s.top_k = impl ? static_cast<int32_t>(impl->top_k) : 0;
        return s;
    }
};

// Stateful context ----------------------------------------------------------

struct GP_TableContext {
    GP_TableStyle style_raw{};
    Style st{};
    std::vector<GP_TableRow> rows;
    std::vector<GP_TableColumn> cols;
    GP_TableGeom geom{};
    // Scroll state: row offset (first visible row index) and fraction (0..1)
    int32_t scroll_row_offset = 0;
    float scroll_frac = 0.0f;
    float scroll_frac_x = 0.0f;
    // selected LED keys: packed (row<<32) | (col<<16) | led_index
    std::unordered_set<uint64_t> selected_leds;
    std::unordered_map<uint64_t, float> led_glow_strength;
    std::unordered_map<uint64_t, float> led_flow_glow_strength;
    std::vector<std::pair<uint64_t,uint64_t>> edges;
    std::vector<EdgeTensorFifo> edge_fifos; // companion FIFO per rope edge
    std::vector<uint64_t> edge_last_write_seq;
    std::vector<float> edge_flow_phase;
    std::vector<GP_TableEdgeBatchMetadata> edge_batch_metadata;
    std::vector<uint32_t> edge_subgroup_flags;
    // per-edge persistent unique id used for ThreadManager registration
    std::vector<uint64_t> edge_ids;
    uint64_t next_edge_id = 1;
    // per-edge subscriber_key -> reader_slot registered with ThreadManager
    std::vector<std::unordered_map<uint64_t,int>> edge_subscriber_slots;
    std::unordered_map<uint64_t, StagePortBinding> stage_ports; // LED key -> stage port binding
    // Relaxation state (per-edge values/velocities are maintained in parallel to edges)
    int32_t relax_mode = GP_TABLE_RELAX_OFF;
    float relax_stiffness = 10.0f;
    float relax_damping = 2.0f;
    float relax_threshold = 1e-3f;
    int32_t relax_max_iters = 200;
    double relax_last_time = 0.0; // seconds since epoch
    std::vector<float> relax_value; // 0..1 value per edge (1.0 = relaxed)
    std::vector<float> relax_vel;   // velocity per edge
    // rope simulator instance and per-rope sim index mapping
    RopeSim* rope_sim = nullptr;
    int rope_sim_owned = 0; // 1 if this context owns and should destroy the sim
    // mapping from persistent rope id -> rope_sim index. Using IDs
    // avoids relying on edge/vector ordering and is resilient to reorders.
    std::unordered_map<uint64_t,int> rope_id_to_sim_idx;
    // per-rope persistent unique ids (indexed by created rope order). Used
    // to map serialized vertices to runtime rope indices deterministically.
    std::vector<uint64_t> rope_ids;
    uint64_t next_rope_id = 1;
    // flag set during gp_table_deserialize to indicate we're restoring
    int restoring = 0;
    // Optional module UUID embedded into serialized blobs (0 == unset)
    uint64_t module_uuid = 0ull;
    // Optional module-frame port UUIDs to serialize: entries are (row, idx, uuid)
    struct FramePortEntry { int32_t row; int32_t idx; uint64_t uuid; };
    std::vector<FramePortEntry> frame_port_uuids;
    // Prospective live-edge state (used when one node selected and mode enabled)
    int32_t prospective_mode = 0; // 0=off,1=on
    bool prospective_initialized = false;
    float prospective_x = 0.0f;
    float prospective_y = 0.0f;
    float prospective_vx = 0.0f;
    float prospective_vy = 0.0f;
    // queue of recent mouse targets (front = oldest to be cleared next)
    std::vector<std::pair<float,float>> prospective_targets;
    int32_t prospective_max_history = 8;
    float prospective_slack = 4.0f;
    float prospective_rope_length = 0.0f;
    int prospective_rope_idx = -1; // temporary rope index for live prospective cable
    // editable flag (editor potential). Default: editable (1).
    int32_t editable = 1;
    // per-key type hints and io sets
    std::unordered_map<uint64_t,int32_t> key_type_hint; // key -> type_id
    std::unordered_set<uint64_t> key_is_input; // keys marked input
    std::unordered_set<uint64_t> key_is_output; // keys marked output
    // reading direction for sides (0=input,1=output). 0=LTR,1=TTB,2=RTL,3=BTB
    int32_t side_reading_dir[2] = {0, 0};
    // LED grid preference (cols, rows, aspect)
    int32_t led_pref_cols = 0;
    int32_t led_pref_rows = 0;
    float led_pref_aspect = 0.0f;
    // Optional step callback for node/table shims
    GP_TableStepFn step_callback = nullptr;
    void* step_user = nullptr;
    // Whether table-side simulator stepping is enabled (1) or disabled (0).
    // KPN or other managers can toggle this to pause heavy sim work per table.
    int32_t sim_enabled = 1;
    // (No per-table frame-skip; global frame-skip handled by table_abi global state)
    // Optional click-action dispatch
    std::vector<GP_TableAction> actions;
    GP_TableActionFn action_callback = nullptr;
    void* action_user = nullptr;
    // Optional keyboard callback
    GP_TableKeyFn key_callback = nullptr;
    void* key_user = nullptr;
    // meta-groups created by tools (opaque internal storage)
    struct RingEntry {
        int ring_id = -1; // rope_sim ring id
        uint64_t key = 0; // caller-provided key for identification
        EdgeTensorFifo fifo;
        GP_TableEdgeBatchMetadata batch_metadata{};
        uint32_t subgroup_flags = 0;
        uint64_t uid = 0;
    };
    std::vector<RingEntry> rings;
    uint64_t next_ring_uid = 1;
    std::vector<std::unordered_map<uint64_t,int>> ring_subscriber_slots; // per-ring subscriber -> slot
    struct GP_MetaGroupInternal;
    std::vector<std::unique_ptr<GP_MetaGroupInternal>> meta_groups;
    // Pending operations enqueued by UI threads to be applied by the manager.
    enum PendingOpType {
        PENDING_OP_ADD_EDGE = 1,
        PENDING_OP_CLEAR_EDGES = 2,
        PENDING_OP_SUBSCRIBE_EDGE = 3,
        PENDING_OP_UNSUBSCRIBE_EDGE = 4,
        PENDING_OP_BIND_STAGE = 5,
        PENDING_OP_UNBIND_STAGE = 6,
    };
    struct PendingOp {
        PendingOpType type;
        // fields used by various ops
        uint64_t a_key = 0;
        uint64_t b_key = 0;
        int32_t edge_idx = -1;
        uint64_t sub_key = 0;
        int32_t start_at_head = 1;
        GP_StageContext* stage = nullptr;
        int32_t is_output = 0;
        int32_t channel = 0;
    };
    std::vector<PendingOp> pending_ops;
    std::mutex pending_ops_mu;
};

// Global sim frame-skip state (shared across all tables)
static int g_global_sim_frame_skip_count = 0; // number of frames to skip between steps (0 = every frame)
static uint64_t g_global_sim_frame_tick = 0; // incremented once per canvas frame

static inline bool table_should_step_sim(GP_TableContext* ctx) {
    if (!ctx) return false;
    if (!ctx->sim_enabled) return false;
    int count = std::max(0, g_global_sim_frame_skip_count);
    if (count <= 0) return true;
    // step once every (count+1) frames
    return (g_global_sim_frame_tick % static_cast<uint64_t>(count + 1)) == 0ull;
}

int32_t gp_table_set_global_sim_frame_skip_count(int32_t count) {
    g_global_sim_frame_skip_count = std::max(0, count);
    return 1;
}

int32_t gp_table_get_global_sim_frame_skip_count(int32_t* out_count) {
    if (!out_count) return 0;
    *out_count = g_global_sim_frame_skip_count;
    return 1;
}

void gp_table_advance_global_sim_tick() {
    g_global_sim_frame_tick = (g_global_sim_frame_tick + 1) % 0xFFFFFFFFFFFFu;
}

int32_t gp_table_should_step_sim(GP_TableContext* ctx) {
    return table_should_step_sim(ctx) ? 1 : 0;
}

// Internal representation of a meta-group. Exposed to C callers as an
// opaque `GP_MetaGroup*` pointer (allocated here and stored in the
// table's `meta_groups` vector to keep lifetime management consistent).
struct GP_MetaGroup {
    std::vector<std::pair<int,int>> vertices; // (rope_idx, vertex_idx)
    float confinement = 1.0f; // tightness/pressure
    int sim_group_idx = -1; // index into RopeSim meta_groups if registered
    uint64_t id = 0; // debug id
    LassoConfig lasso_config; // configuration flags and widget type for this meta-group
    int dangling_widget_id = -1; // RopeSim widget id if created
    // optional anchor override used by helpers (e.g., dangling widget attach)
    int anchor_rope = -1;
    int anchor_vert = -1;
    // optional overlay keys created by canvas-level helpers (0 == none)
    unsigned long long overlay_key_a = 0ull;
    unsigned long long overlay_key_b = 0ull;
    // optional FIFO and subgroup flags so meta-groups can behave like edges
    EdgeTensorFifo fifo;
    uint32_t subgroup_flags = 0;
    // channel group id (user-tunable integer controlling grouping of FIFOs/edges)
    int channel_group = 0;
    // if a dangling short-rope + widget was created, remember rope id and vertex
    int dangling_widget_rope = -1;
    int dangling_widget_rope_vid = -1;
    float dangling_hang_len = 0.0f;
    // ring topology mode: 0=ribbon (chain), 1=closed loop, 2=dense cross-links
    int ring_mode = 0;
};

// Thin adaptor type used to store meta-groups in the context vector.
struct GP_TableContext::GP_MetaGroupInternal : public GP_MetaGroup {};

extern "C" GP_MetaGroup* gp_table_meta_create(GP_TableContext* ctx) {
    if (!ctx) return nullptr;
    auto mg = std::make_unique<GP_TableContext::GP_MetaGroupInternal>();
    static uint64_t next_mg_id = 1;
    mg->id = next_mg_id++;
    // Initialize lasso_config to avoid uninitialized reads in canvas logic
    mg->lasso_config.flags = 0;
    mg->lasso_config.widget_type = 0;
    mg->lasso_config.reserved[0] = 0;
    mg->lasso_config.reserved[1] = 0;
    mg->lasso_config.reserved[2] = 0;
    mg->lasso_config.spring_min_rest = 0.0f;
    mg->lasso_config.spring_reduce_rate = 0.0f;
    mg->lasso_config.spring_mode = 0;
    mg->lasso_config.spring_reserved[0] = 0;
    mg->lasso_config.spring_reserved[1] = 0;
    mg->lasso_config.spring_reserved[2] = 0;
    GP_MetaGroup* ptr = mg.get();
    ctx->meta_groups.push_back(std::move(mg));
    printf("gp_table_meta_create: created mg=%p id=%llu on ctx=%p\n", (void*)ptr, (unsigned long long)ptr->id, (void*)ctx);
    return ptr;
}

// Debug helper: print a meta-group's vertices and lasso config for diagnostics.
extern "C" int32_t gp_table_debug_dump_meta_group(GP_TableContext* ctx, GP_MetaGroup* mg, const char* tag) {
    if (!ctx || !mg) return 0;
    if (!tag) tag = "dump";
    printf("gp_table_debug_dump_meta_group: [%s] mg=%p id=%llu sim_group_idx=%d confinement=%.3f ring_mode=%d dangling_rope=%d dangling_vid=%d\n",
           tag, (void*)mg, (unsigned long long)mg->id, mg->sim_group_idx, mg->confinement, mg->ring_mode, mg->dangling_widget_rope, mg->dangling_widget_rope_vid);
    printf("  lasso_config: flags=0x%08x widget=%u spring_min_rest=%.2f spring_reduce_rate=%.2f spring_mode=%u\n",
           mg->lasso_config.flags, static_cast<unsigned int>(mg->lasso_config.widget_type), mg->lasso_config.spring_min_rest, mg->lasso_config.spring_reduce_rate, static_cast<unsigned int>(mg->lasso_config.spring_mode));
    int vcount = static_cast<int>(mg->vertices.size());
    printf("  vertices.count=%d\n", vcount);
    for (int vi = 0; vi < vcount; ++vi) {
        int r = mg->vertices[static_cast<size_t>(vi)].first;
        int v = mg->vertices[static_cast<size_t>(vi)].second;
        uint64_t ru = 0ull;
        if (r >= 0) {
            // rope indices are transient sim indices; find the persistent UID that maps to this sim index
            for (const auto &kv : ctx->rope_id_to_sim_idx) {
                if (kv.second == r) { ru = kv.first; break; }
            }
        }
        printf("    [%d] rope_idx=%d vert_idx=%d rope_id=%llu\n", vi, r, v, (unsigned long long)ru);
    }
    fflush(stdout);
    return 1;
}

// Removed: gp_table_sim_toggle_meta_group_mode_for_rope

extern "C" int32_t gp_table_meta_get_ring_mode(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_mode) {
    if (!ctx || !mg || !out_mode) return 0;
    *out_mode = mg->ring_mode;
    return 1;
}

extern "C" int32_t gp_table_meta_get_confinement(const GP_TableContext* ctx, GP_MetaGroup* mg, float* out_conf) {
    if (!ctx || !mg || !out_conf) return 0;
    *out_conf = mg->confinement;
    return 1;
}
extern "C" int32_t gp_table_meta_set_confinement(GP_TableContext* ctx, GP_MetaGroup* mg, float conf) {
    if (!ctx || !mg) return 0;
    mg->confinement = conf;
    RopeSim* sim = ctx->rope_sim;
    if (mg->sim_group_idx >= 0 && sim) {
        rope_sim_meta_group_set_pressure(sim, mg->sim_group_idx, conf);
    }
    return 1;
}
extern "C" int32_t gp_table_meta_get_id(const GP_TableContext* ctx, GP_MetaGroup* mg, unsigned long long* out_id) {
    if (!ctx || !mg || !out_id) return 0;
    *out_id = mg->id;
    return 1;
}
extern "C" int32_t gp_table_meta_set_id(GP_TableContext* ctx, GP_MetaGroup* mg, unsigned long long id) {
    if (!ctx || !mg) return 0;
    mg->id = id;
    return 1;
}
extern "C" int32_t gp_table_meta_get_dangling_hang_len(const GP_TableContext* ctx, GP_MetaGroup* mg, float* out_len) {
    if (!ctx || !mg || !out_len) return 0;
    *out_len = mg->dangling_hang_len;
    return 1;
}
extern "C" int32_t gp_table_meta_set_dangling_hang_len(GP_TableContext* ctx, GP_MetaGroup* mg, float len) {
    if (!ctx || !mg) return 0;
    mg->dangling_hang_len = len;
    return 1;
}
extern "C" int32_t gp_table_meta_get_lasso_fields(const GP_TableContext* ctx, GP_MetaGroup* mg, unsigned int* out_flags, int32_t* out_widget_type) {
    if (!ctx || !mg || !out_flags || !out_widget_type) return 0;
    *out_flags = mg->lasso_config.flags;
    *out_widget_type = static_cast<int32_t>(mg->lasso_config.widget_type);
    return 1;
}
extern "C" int32_t gp_table_meta_set_lasso_fields(GP_TableContext* ctx, GP_MetaGroup* mg, unsigned int flags, int32_t widget_type) {
    if (!ctx || !mg) return 0;
    mg->lasso_config.flags = flags;
    mg->lasso_config.widget_type = static_cast<uint8_t>(widget_type & 0xFF);
    return 1;
}
extern "C" int32_t gp_table_meta_set_ring_mode(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t mode) {
    if (!ctx || !mg) return 0;
    mg->ring_mode = mode;
    return 1;
}

extern "C" int32_t gp_table_meta_destroy(GP_TableContext* ctx, GP_MetaGroup* mg) {
    if (!ctx || !mg) return 0;
    for (size_t i = 0; i < ctx->meta_groups.size(); ++i) {
        if (ctx->meta_groups[i].get() == mg) {
            // If the table has a sim attached and the meta-group registered
            // a sim_group_idx, ensure sim-side resources (springs, meta-group)
            // are cleaned up before removing the meta-group record.
            RopeSim* sim = ctx->rope_sim;
            int sim_idx = mg->sim_group_idx;
            if (sim && sim_idx >= 0) {
                // disable any edge-springs associated with this meta-group
                rope_sim_meta_group_disable_edge_springs(sim, sim_idx);
                rope_sim_destroy_meta_group(sim, sim_idx);
            }
            ctx->meta_groups.erase(ctx->meta_groups.begin() + static_cast<ptrdiff_t>(i));
            return 1;
        }
    }
    return 0;
}

extern "C" int32_t gp_table_meta_enable_edge_springs(GP_TableContext* ctx, GP_MetaGroup* mg, float min_rest, float reduce_rate) {
    if (!ctx || !mg) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim || mg->sim_group_idx < 0) return 0;
    int res = rope_sim_meta_group_enable_edge_springs(sim, mg->sim_group_idx, min_rest, reduce_rate);
    if (!res) return 0;
    // Additionally, schedule per-rope rest-length adjustments so ropes are
    // briefly let out and then gradually shortened to simulate trimming/tensioning.
    float extend_amount = 5.0f; // increase rest length immediately
    float shorten_factor = 0.6f; // target = current * factor
    float shorten_rate = 1.0f; // units per second
    float shorten_delay = 0.5f; // seconds before shortening begins
    std::unordered_set<int> handled;
    for (const auto &p : mg->vertices) {
        int r = p.first;
        if (handled.find(r) != handled.end()) continue;
        handled.insert(r);
        rope_sim_modify_rest_length(sim, r, extend_amount);
        float cur = 0.0f;
        if (rope_sim_get_rope_rest_length(sim, r, &cur)) {
            float target = std::max(0.0001f, cur * shorten_factor);
            rope_sim_set_rope_rest_target(sim, r, target, shorten_rate, shorten_delay);
            printf("gp_table_meta_enable_edge_springs: rope %d rest increased by %.2f then scheduled target %.2f (delay=%.2f rate=%.2f)\n", r, extend_amount, target, shorten_delay, shorten_rate);
        }
    }
    return 1;
}

extern "C" int32_t gp_table_meta_disable_edge_springs(GP_TableContext* ctx, GP_MetaGroup* mg) {
    if (!ctx || !mg) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim || mg->sim_group_idx < 0) return 0;
    return rope_sim_meta_group_disable_edge_springs(sim, mg->sim_group_idx);
}

extern "C" int32_t gp_table_rope_modify_rest_length(GP_TableContext* ctx, int32_t rope_idx, float delta) {
    if (!ctx) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_modify_rest_length(sim, static_cast<int>(rope_idx), delta);
}

extern "C" int32_t gp_table_rope_set_rest_target(GP_TableContext* ctx, int32_t rope_idx, float target_rest, float rate, float delay) {
    if (!ctx) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_set_rope_rest_target(sim, static_cast<int>(rope_idx), target_rest, rate, delay);
}

extern "C" int32_t gp_table_rope_get_rest_length(GP_TableContext* ctx, int32_t rope_idx, float* out_rest) {
    if (!ctx || !out_rest) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_get_rope_rest_length(sim, static_cast<int>(rope_idx), out_rest);
}

extern "C" int32_t gp_table_rope_set_radius(GP_TableContext* ctx, int32_t rope_idx, float radius) {
    if (!ctx) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_set_rope_radius(sim, static_cast<int>(rope_idx), radius);
}

extern "C" int32_t gp_table_rope_get_radius(GP_TableContext* ctx, int32_t rope_idx, float* out_radius) {
    if (!ctx || !out_radius) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_get_rope_radius(sim, static_cast<int>(rope_idx), out_radius);
}

extern "C" int32_t gp_table_rope_insert_vertex(GP_TableContext* ctx, int32_t rope_idx, int32_t seg_index, float t) {
    if (!ctx) return -1;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return -1;
    return rope_sim_insert_vertex(sim, static_cast<int>(rope_idx), static_cast<int>(seg_index), t);
}
extern "C" int32_t gp_table_sim_add_meta_group_for_rope(GP_TableContext* ctx, int32_t rope_idx) {
    if (!ctx) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    int vc = rope_sim_get_vertex_count(sim, static_cast<int>(rope_idx));
    if (vc <= 1) return 0;
    int sg = rope_sim_create_meta_group(sim, 1.0f);
    if (sg < 0) return 0;
    // add first and last vertex as members so edge-springs can be created
    rope_sim_meta_group_add(sim, sg, static_cast<int>(rope_idx), 0);
    rope_sim_meta_group_add(sim, sg, static_cast<int>(rope_idx), vc - 1);
    rope_sim_meta_group_set_mode(sim, sg, 0);
    int res = rope_sim_meta_group_enable_edge_springs(sim, sg, 2.0f, 50.0f);
    printf("gp_table_sim_add_meta_group_for_rope: created sg=%d for rope=%d res=%d\n", sg, rope_idx, res);
    return res ? 1 : 0;
}

extern "C" int32_t gp_table_create_ring(GP_TableContext* ctx, int32_t rope_idx, float u) {
    if (!ctx) return -1;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return -1;
    // Create a ring at parameter u (no insertion into the main rope). Then
    // create a short T-off rope centered at the ring's world position and
    // attach a heavy widget to its center. The T-off center is joined into
    // the meta-group next to the existing anchor vertex so forces transmit.
    int vc = rope_sim_get_vertex_count(sim, static_cast<int>(rope_idx));
    if (vc <= 1) {
        return rope_sim_create_ring(sim, static_cast<int>(rope_idx), u);
    }
    // sample world position along rope at param u
    std::vector<float> verts3(static_cast<size_t>(vc) * 3);
    int got = rope_sim_get_vertices3(sim, static_cast<int>(rope_idx), verts3.data(), static_cast<int>(verts3.size()));
    if (got <= 1) return rope_sim_create_ring(sim, static_cast<int>(rope_idx), u);
    float fidx = u * static_cast<float>(vc - 1);
    int i0 = static_cast<int>(std::floor(fidx));
    if (i0 < 0) i0 = 0; if (i0 >= vc-1) i0 = vc-2;
    int i1 = i0 + 1;
    float local_t = fidx - static_cast<float>(i0);
    float ax = verts3[3*i0+0]; float ay = verts3[3*i0+1]; float az = verts3[3*i0+2];
    float bx = verts3[3*i1+0]; float by = verts3[3*i1+1]; float bz = verts3[3*i1+2];
    float px = ax + (bx - ax) * local_t;
    float py = ay + (by - ay) * local_t;
    float pz = az + (bz - az) * local_t;
    // create ring at u
    int ring_id = rope_sim_create_ring(sim, static_cast<int>(rope_idx), u);

    // Build T-off: compute perpendicular to tangent (using local segment)
    float tx = bx - ax; float ty = by - ay; float tz = bz - az;
    float tlen = std::sqrt(tx*tx + ty*ty + tz*tz);
    float pxp = 0.0f, pyp = 1.0f; // default perp
    if (tlen > 1e-6f) {
        tx /= tlen; ty /= tlen; tz /= tlen;
        // 2D perp in XY plane
        pxp = -ty; pyp = tx;
    }
    float half = 12.0f; // half-length of T-off
    float e1x = px + pxp * half;
    float e1y = py + pyp * half;
    float e1z = pz;
    float e2x = px - pxp * half;
    float e2y = py - pyp * half;
    float e2z = pz;
    int to_segs = 2;
    int to_rope = rope_sim_add_rope3(sim, e1x, e1y, e1z, e2x, e2y, e2z, to_segs, 0.0f);
    if (to_rope < 0) return ring_id;

    // Create a custom canvas overlay rectangle for the short T-off endpoints
    // and register two overlay LED keys. Also attach the created rope to the
    // overlay so canvas-level rendering and interaction can bind to it.
    GP_CanvasContext* cvs = gp_canvas_get_singleton();
    unsigned long long t_off_overlay_a = 0ull, t_off_overlay_b = 0ull;
    if (cvs) {
        // If the meta-group already has persisted overlay keys, prefer them
        // and attach the created T-off rope to the canonical overlay instead
        // of creating a new overlay instance. This avoids duplicate overlays
        // when restoring from persisted keys.
        // Note: we'll attach per-meta-group below when iterating meta-groups.
    }

    // For every meta-group anchored to this rope, insert the T-off center vertex
    // into the group's ordering adjacent to the anchor so the T-off is joined
    // to the existing ring bindings (transmits forces). Attach a heavy widget
    // to the T-off center (vertex index 1) and increase its mass.
    for (size_t mgi = 0; mgi < ctx->meta_groups.size(); ++mgi) {
        auto &mgptr = ctx->meta_groups[mgi];
        if (!mgptr) continue;
        GP_MetaGroup* mg = mgptr.get();
        if (!mg) continue;
        if (mg->anchor_rope != rope_idx) continue;
        if (mg->sim_group_idx < 0) {
            int sg = rope_sim_create_meta_group(sim, mg->confinement);
            if (sg >= 0) mg->sim_group_idx = sg;
        }
            if (mg->sim_group_idx >= 0) {
            // add T-off center to the table-level meta-group record so it's a
            // genuine member visible to APIs and rendering
            gp_table_meta_add_vertex(ctx, mg, static_cast<int32_t>(to_rope), 1);
            // insert T-off center (vertex 1) after the anchor vertex in sim ordering
            rope_sim_meta_group_insert(sim, mg->sim_group_idx, mg->anchor_rope, mg->anchor_vert, to_rope, 1);
            // if there is a dangling widget rope, insert it after the T-off so it's connected
            if (mg->dangling_widget_rope >= 0) {
                rope_sim_meta_group_insert(sim, mg->sim_group_idx, to_rope, 1, mg->dangling_widget_rope, 0);
            }
            rope_sim_meta_group_set_pressure(sim, mg->sim_group_idx, mg->confinement);
            // create heavy widget on center of T-off
            int wid = rope_sim_create_dangling_widget(sim, to_rope, 1, static_cast<unsigned int>(mg->lasso_config.widget_type));
            if (wid >= 0) {
                rope_sim_set_widget_mass(sim, wid, 50.0f);
            }
            // record dangling widget/rope on meta-group so rendering and
            // queries will reference the real hanging rope and widget.
            if (wid >= 0) {
                mg->dangling_widget_id = wid;
                mg->dangling_widget_rope = to_rope;
                mg->dangling_widget_rope_vid = 1;
                mg->dangling_hang_len = half;
                // inform RopeSim about the dangling/widget rope so it can
                // give special rest-length and stiffness behavior on edges.
                rope_sim_meta_group_set_dangling_rope(sim, mg->sim_group_idx, to_rope);
            }
            // If the meta-group already had persisted overlay keys, attach
            // the created T-off rope to that canonical overlay. Otherwise,
            // create a new overlay for the T-off endpoints as before.
            if (cvs) {
                if (mg->overlay_key_a != 0ull || mg->overlay_key_b != 0ull) {
                    // ensure canonical overlay exists and attach rope
                    gp_canvas_register_table_overlay(cvs, mg->overlay_key_a, mg->overlay_key_b, 0ull, 0ull);
                    gp_canvas_attach_rope_to_overlay(cvs, mg->overlay_key_a, mg->overlay_key_b, to_rope);
                    gp_canvas_set_overlay_meta(cvs, mg->overlay_key_a, mg->overlay_key_b, ctx, reinterpret_cast<void*>(mg));
                } else {
                    unsigned long long key_a = 0ull, key_b = 0ull;
                    if (gp_canvas_create_overlay_with_leds(cvs, e1x, e1y, e2x, e2y, &key_a, &key_b)) {
                        int edge_idx = gp_canvas_attach_rope_to_overlay(cvs, key_a, key_b, to_rope);
                        printf("gp_table: attached rope %d to overlay keys (%llu,%llu) edge_idx=%d\n", to_rope, (unsigned long long)key_a, (unsigned long long)key_b, edge_idx);
                        t_off_overlay_a = key_a;
                        t_off_overlay_b = key_b;
                        mg->overlay_key_a = t_off_overlay_a;
                        mg->overlay_key_b = t_off_overlay_b;
                        gp_canvas_set_overlay_meta(cvs, t_off_overlay_a, t_off_overlay_b, ctx, reinterpret_cast<void*>(mg));
                    }
                }
            }
        }
    }

    return ring_id;
}

// Create ring by persisted rope id. Resolve id to runtime rope index using
// canvas mapping then table-local fallback, and call gp_table_create_ring.
extern "C" int32_t gp_table_create_ring_by_id(GP_TableContext* ctx, uint64_t rope_id, float u) {
    if (!ctx || rope_id == 0ull) return -1;
    GP_CanvasContext* cvs = gp_canvas_get_singleton();
    int resolved = -1;
    if (cvs) resolved = gp_canvas_resolve_rope_id_to_index(cvs, ctx, rope_id);
    if (resolved < 0) {
        // fallback to table-local id->sim mapping
        for (size_t ri = 0; ri < ctx->rope_ids.size(); ++ri) {
            if (ctx->rope_ids[ri] == rope_id) {
                auto it = ctx->rope_id_to_sim_idx.find(rope_id);
                if (it != ctx->rope_id_to_sim_idx.end()) resolved = it->second;
                else resolved = -1;
                break;
            }
        }
    }
    if (resolved < 0) return -2; // distinct code for id->index resolution failure
    return gp_table_create_ring(ctx, resolved, u);
}

extern "C" int32_t gp_table_destroy_ring(GP_TableContext* ctx, int32_t ring_id) {
    if (!ctx) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_destroy_ring(sim, static_cast<int>(ring_id));
}

extern "C" int32_t gp_table_set_ring_target(GP_TableContext* ctx, int32_t ring_id, float target_u, float speed) {
    if (!ctx) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_set_ring_target(sim, static_cast<int>(ring_id), target_u, speed);
}

extern "C" int32_t gp_table_get_ring_u(GP_TableContext* ctx, int32_t ring_id, float* out_u) {
    if (!ctx || !out_u) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_get_ring_u(sim, static_cast<int>(ring_id), out_u);
}

extern "C" int32_t gp_table_meta_add_vertex(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t rope_idx, int32_t vertex_idx) {
    if (!ctx || !mg) return 0;
    mg->vertices.emplace_back(static_cast<int>(rope_idx), static_cast<int>(vertex_idx));
    RopeSim* sim = ctx->rope_sim;
    printf("gp_table_meta_add_vertex: mg=%p add (rope=%d,vert=%d) sim=%p sim_group_idx=%d\n", (void*)mg, rope_idx, vertex_idx, (void*)sim, mg->sim_group_idx);
    // If the table owns or is attached to a RopeSim, ensure a sim-level
    // meta group exists and register the vertex there so confinement
    // forces are applied during simulation.
    if (sim) {
        if (mg->sim_group_idx < 0) {
            int sg = rope_sim_create_meta_group(sim, mg->confinement);
            if (sg >= 0) mg->sim_group_idx = sg;
        }
        if (mg->sim_group_idx >= 0) {
            rope_sim_meta_group_add(sim, mg->sim_group_idx, static_cast<int>(rope_idx), static_cast<int>(vertex_idx));
            // update pressure in sim if different
            rope_sim_meta_group_set_pressure(sim, mg->sim_group_idx, mg->confinement);
        }
    }
    return 1;
}

extern "C" int32_t gp_table_meta_set_anchor(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t rope_idx, int32_t vertex_idx) {
    if (!ctx || !mg) return 0;
    mg->anchor_rope = rope_idx;
    mg->anchor_vert = vertex_idx;
    return 1;
}

extern "C" int32_t gp_table_meta_get_anchor(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_rope_idx, int32_t* out_vertex_idx) {
    if (!ctx || !mg) return 0;
    if (out_rope_idx) *out_rope_idx = mg->anchor_rope;
    if (out_vertex_idx) *out_vertex_idx = mg->anchor_vert;
    return 1;
}

extern "C" int32_t gp_table_meta_get_vertex_count(GP_TableContext* ctx, GP_MetaGroup* mg) {
    if (!ctx || !mg) return 0;
    return static_cast<int32_t>(mg->vertices.size());
}

extern "C" int32_t gp_table_get_meta_group_count(const GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int32_t>(ctx->meta_groups.size());
}

extern "C" GP_MetaGroup* gp_table_get_meta_group(GP_TableContext* ctx, int32_t idx) {
    if (!ctx) return nullptr;
    if (idx < 0 || static_cast<size_t>(idx) >= ctx->meta_groups.size()) return nullptr;
    return ctx->meta_groups[static_cast<size_t>(idx)].get();
}

extern "C" int32_t gp_table_meta_get_vertex(const GP_TableContext* ctx, GP_MetaGroup* mg, int32_t idx, int32_t* out_rope_idx, int32_t* out_vertex_idx) {
    if (!ctx || !mg) return 0;
    if (idx < 0 || static_cast<size_t>(idx) >= mg->vertices.size()) return 0;
    auto &p = mg->vertices[static_cast<size_t>(idx)];
    if (out_rope_idx) *out_rope_idx = p.first;
    if (out_vertex_idx) *out_vertex_idx = p.second;
    return 1;
}

extern "C" int32_t gp_table_get_widget_position(GP_TableContext* ctx, int32_t widget_id, float* out_xyz) {
    if (!ctx || !out_xyz) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_get_widget_position(sim, static_cast<int>(widget_id), out_xyz);
}

extern "C" int32_t gp_table_meta_get_dangling_widget_id(const GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_widget_id) {
    if (!ctx || !mg || !out_widget_id) return 0;
    *out_widget_id = mg->dangling_widget_id;
    return 1;
}

extern "C" int32_t gp_table_meta_set_lasso_config(GP_TableContext* ctx, GP_MetaGroup* mg, const LassoConfig* cfg) {
    if (!ctx || !mg) return 0;
    if (cfg) {
        mg->lasso_config.flags = cfg->flags;
        mg->lasso_config.widget_type = cfg->widget_type;
        mg->lasso_config.spring_min_rest = cfg->spring_min_rest;
        mg->lasso_config.spring_reduce_rate = cfg->spring_reduce_rate;
        mg->lasso_config.spring_mode = cfg->spring_mode;
    } else {
        mg->lasso_config.flags = 0;
        mg->lasso_config.widget_type = 0;
    }
    return 1;
}

extern "C" int32_t gp_table_meta_get_lasso_config(GP_TableContext* ctx, GP_MetaGroup* mg, LassoConfig* out_cfg) {
    if (!ctx || !mg || !out_cfg) return 0;
    out_cfg->flags = mg->lasso_config.flags;
    out_cfg->widget_type = mg->lasso_config.widget_type;
    out_cfg->reserved[0] = mg->lasso_config.reserved[0];
    out_cfg->reserved[1] = mg->lasso_config.reserved[1];
    out_cfg->reserved[2] = mg->lasso_config.reserved[2];
    out_cfg->spring_min_rest = mg->lasso_config.spring_min_rest;
    out_cfg->spring_reduce_rate = mg->lasso_config.spring_reduce_rate;
    out_cfg->spring_mode = mg->lasso_config.spring_mode;
    out_cfg->spring_reserved[0] = mg->lasso_config.spring_reserved[0];
    out_cfg->spring_reserved[1] = mg->lasso_config.spring_reserved[1];
    out_cfg->spring_reserved[2] = mg->lasso_config.spring_reserved[2];
    return 1;
}

extern "C" int32_t gp_table_meta_set_edge_spring_params(GP_TableContext* ctx, GP_MetaGroup* mg, float min_rest, float reduce_rate, int32_t mode) {
    if (!ctx || !mg) return 0;
    mg->lasso_config.spring_min_rest = min_rest;
    mg->lasso_config.spring_reduce_rate = reduce_rate;
    mg->lasso_config.spring_mode = static_cast<uint8_t>(mode & 0xFF);
    return 1;
}

extern "C" int32_t gp_table_meta_get_subgroup_flags(GP_TableContext* ctx, GP_MetaGroup* mg, uint32_t* out_flags) {
    if (!ctx || !mg || !out_flags) return 0;
    // Prefer explicit subgroup_flags stored on the meta-group, otherwise fall
    // back to any lasso-config flags the creator supplied.
    if (mg->subgroup_flags != 0u) {
        *out_flags = mg->subgroup_flags;
        return 1;
    }
    *out_flags = mg->lasso_config.flags;
    return 1;
}

extern "C" int32_t gp_table_meta_get_dangling_rope_info(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_rope_idx, int32_t* out_vertex_idx) {
    if (!ctx || !mg || !out_rope_idx || !out_vertex_idx) return 0;
    *out_rope_idx = mg->dangling_widget_rope;
    *out_vertex_idx = mg->dangling_widget_rope_vid;
    return 1;
}

extern "C" int32_t gp_table_meta_set_channel_group(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t channel_group) {
    if (!ctx || !mg) return 0;
    mg->channel_group = channel_group;
    (void)ctx; (void)mg; (void)channel_group;
    return 1;
}

extern "C" int32_t gp_table_meta_get_channel_group(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_channel_group) {
    if (!ctx || !mg || !out_channel_group) return 0;
    *out_channel_group = mg->channel_group;
    (void)ctx; (void)mg; (void)out_channel_group;
    return 1;
}

extern "C" int32_t gp_table_meta_get_fifo_snapshot(GP_TableContext* ctx, GP_MetaGroup* mg, float* out_buf, int32_t out_len) {
    if (!ctx || !mg || !out_buf || out_len <= 0) return 0;
    // Collect a simple float-based snapshot of key fifo fields.
    int32_t pos = 0;
    auto pushf = [&](float v)->bool { if (pos >= out_len) return false; out_buf[pos++] = v; return true; };
    // id, vertex count, channel_group
    pushf(static_cast<float>(static_cast<double>(mg->id)));
    pushf(static_cast<float>(static_cast<int>(mg->vertices.size())));
    pushf(static_cast<float>(mg->channel_group));
    // FIFO internals (best-effort atomic reads)
    if (!mg) return pos;
    EdgeTensorFifo &ef = mg->fifo;
    size_t stride = ef.impl ? ef.impl->stride : 0;
    size_t slots = ef.impl ? ef.impl->slots : 0;
    size_t topk = ef.impl ? ef.impl->top_k : 0;
    uint64_t write_seq = ef.impl ? ef.impl->write_seq.load(std::memory_order_relaxed) : 0ull;
    uint64_t writer = ef.impl ? ef.impl->writer.load(std::memory_order_relaxed) : 0ull;
    uint64_t last_write_seq = ef.impl ? ef.impl->last_write_seq.load(std::memory_order_relaxed) : 0ull;
    uint64_t last_read_seq = ef.impl ? ef.impl->last_read_seq.load(std::memory_order_relaxed) : 0ull;
    float write_friction = ef.impl ? ef.impl->write_friction.load(std::memory_order_relaxed) : 0.0f;
    float read_friction = ef.impl ? ef.impl->read_friction.load(std::memory_order_relaxed) : 0.0f;
    int32_t last_write_region = ef.impl ? ef.impl->last_write_region.load(std::memory_order_relaxed) : -1;
    int32_t last_read_region = ef.impl ? ef.impl->last_read_region.load(std::memory_order_relaxed) : -1;
    float write_phase = ef.impl ? ef.impl->write_phase.load(std::memory_order_relaxed) : 0.0f;
    float read_phase = ef.impl ? ef.impl->read_phase.load(std::memory_order_relaxed) : 0.0f;
    int32_t friction_regions = ef.impl ? ef.impl->friction_regions.load(std::memory_order_relaxed) : 0;
    bool configured = ef.impl ? ef.impl->configured : false;
    pushf(static_cast<float>(stride));
    pushf(static_cast<float>(slots));
    pushf(static_cast<float>(topk));
    // sequence numbers may exceed float precision; low 32 bits are kept.
    pushf(static_cast<float>(static_cast<uint32_t>(write_seq & 0xFFFFFFFFu)));
    pushf(static_cast<float>(static_cast<uint32_t>(writer & 0xFFFFFFFFu)));
    pushf(static_cast<float>(static_cast<uint32_t>(last_write_seq & 0xFFFFFFFFu)));
    pushf(static_cast<float>(static_cast<uint32_t>(last_read_seq & 0xFFFFFFFFu)));
    pushf(write_friction);
    pushf(read_friction);
    pushf(static_cast<float>(last_write_region));
    pushf(static_cast<float>(last_read_region));
    pushf(write_phase);
    pushf(read_phase);
    pushf(static_cast<float>(friction_regions));
    pushf(configured ? 1.0f : 0.0f);
    // shape
    pushf(static_cast<float>(static_cast<int>(ef.shape.size())));
    for (size_t i = 0; i < ef.shape.size(); ++i) {
        pushf(static_cast<float>(ef.shape[i]));
    }
    return pos;
}

int32_t gp_table_meta_get_sim_group_index(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_sim_idx) {
    if (!ctx || !mg || !out_sim_idx) return 0;
    *out_sim_idx = mg->sim_group_idx;
    return 1;
}

extern "C" int32_t gp_table_meta_create_widget(GP_TableContext* ctx, GP_MetaGroup* mg) {
    if (!ctx || !mg) return 0;
    // Quick runtime toggle to disable creating the hanging widget/rope for testing.
    const char* disable_env = std::getenv("NODUS_DISABLE_WIDGET_HANG");
    if (disable_env && disable_env[0] != '\0') {
        printf("gp_table_meta_create_widget: disabled via NODUS_DISABLE_WIDGET_HANG\n");
        return 0;
    }
    if (mg->dangling_widget_id >= 0) return 1; // already created
    if (mg->vertices.empty()) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    // Prefer creating a short hanging rope anchored to the meta-group, then
    // attach the widget to the rope's lower vertex so it can hang freely.
    // prefer explicit anchor if set on the meta-group
    int anchor_rope = mg->anchor_rope >= 0 ? mg->anchor_rope : mg->vertices[0].first;
    int anchor_vert = mg->anchor_vert >= 0 ? mg->anchor_vert : mg->vertices[0].second;
    // Query 3D vertex positions for anchor
    int vc = rope_sim_get_vertex_count(sim, anchor_rope);
    if (vc <= 0) return 0;
    std::vector<float> verts3(static_cast<size_t>(vc) * 3);
    int got = rope_sim_get_vertices3(sim, anchor_rope, verts3.data(), static_cast<int>(verts3.size()));
    if (got <= anchor_vert) {
        // fallback: attach widget directly to the existing vertex
        int wid = rope_sim_create_dangling_widget(sim, anchor_rope, anchor_vert, static_cast<unsigned int>(mg->lasso_config.widget_type));
        if (wid < 0) return 0;
        mg->dangling_widget_id = wid;
        return 1;
    }
    float ax = verts3[anchor_vert * 3 + 0];
    float ay = verts3[anchor_vert * 3 + 1];
    float az = verts3[anchor_vert * 3 + 2];
    // Create a short T-off rope anchored near the anchor vertex so the
    // hanging endpoints are genuine rope vertices (this lets rings/connectors
    // slide and bind to the rope correctly). We create a small horizontal
    // T-off centered at the anchor, create an overlay with two LED keys at
    // the endpoints, attach the rope to that overlay, and create a heavy
    // dangling widget attached to the center vertex (vid=1).
    float hang_len = 48.0f;
    // choose neighbor to compute tangent
    int neighbor = (anchor_vert + 1 < got) ? (anchor_vert + 1) : (anchor_vert - 1);
    if (neighbor < 0) neighbor = anchor_vert;
    float bx = verts3[neighbor * 3 + 0];
    float by = verts3[neighbor * 3 + 1];
    float bz = verts3[neighbor * 3 + 2];
    float tx = bx - ax; float ty = by - ay; float tz = bz - az;
    float tlen = std::sqrt(tx*tx + ty*ty + tz*tz);
    if (tlen > 1e-6f) { tx /= tlen; ty /= tlen; tz /= tlen; }
    float pxp = -ty; float pyp = tx; // 2D perp
    float half = hang_len * 0.5f;
    float e1x = ax + pxp * half;
    float e1y = ay + pyp * half;
    float e1z = az;
    float e2x = ax - pxp * half;
    float e2y = ay - pyp * half;
    float e2z = az;
    int to_segs = 2;
    int to_rope = rope_sim_add_rope3(sim, e1x, e1y, e1z, e2x, e2y, e2z, to_segs, 0.0f);
    if (to_rope < 0) {
        // fallback: create widget attached to anchor vertex
        int wid_fb = rope_sim_create_dangling_widget(sim, anchor_rope, anchor_vert, static_cast<unsigned int>(mg->lasso_config.widget_type));
        if (wid_fb < 0) return 0;
        mg->dangling_widget_id = wid_fb;
        mg->dangling_widget_rope = anchor_rope;
        mg->dangling_widget_rope_vid = anchor_vert;
        mg->dangling_hang_len = hang_len;
        return 1;
    }

    // create overlay for the T-off endpoints and attach the rope to the overlay
    GP_CanvasContext* cvs = gp_canvas_get_singleton();
    if (cvs) {
        unsigned long long key_a = 0ull, key_b = 0ull;
        if (gp_canvas_create_overlay_with_leds(cvs, e1x, e1y, e2x, e2y, &key_a, &key_b)) {
            gp_canvas_attach_rope_to_overlay(cvs, key_a, key_b, to_rope);
            // ensure this initial overlay is also bound to the meta-group
            gp_canvas_set_overlay_meta(cvs, key_a, key_b, ctx, mg);
            mg->overlay_key_a = key_a;
            mg->overlay_key_b = key_b;
        }
    }

    // ensure meta-group sim registration
    if (mg->sim_group_idx < 0) {
        int sg = rope_sim_create_meta_group(sim, mg->confinement);
        if (sg >= 0) mg->sim_group_idx = sg;
    }
    try { mg->fifo.configure_default(); } catch(...) {}
    mg->subgroup_flags = mg->lasso_config.flags;

    if (mg->sim_group_idx >= 0) {
        // add the T-off center (vid=1) into the meta-group so forces transmit
        gp_table_meta_add_vertex(ctx, mg, static_cast<int32_t>(to_rope), 1);
        rope_sim_meta_group_insert(sim, mg->sim_group_idx, mg->anchor_rope, mg->anchor_vert, to_rope, 1);
        if (mg->dangling_widget_rope >= 0) {
            rope_sim_meta_group_insert(sim, mg->sim_group_idx, to_rope, 1, mg->dangling_widget_rope, 0);
        }
        rope_sim_meta_group_set_pressure(sim, mg->sim_group_idx, mg->confinement);
    }

    // create heavy widget attached to T-off center
    int wid = rope_sim_create_dangling_widget(sim, to_rope, 1, static_cast<unsigned int>(mg->lasso_config.widget_type));
    if (wid >= 0) rope_sim_set_widget_mass(sim, wid, 8.0f);
    if (wid < 0) {
        // fallback to attach to anchor
        int fallback_wid = rope_sim_create_dangling_widget(sim, anchor_rope, anchor_vert, static_cast<unsigned int>(mg->lasso_config.widget_type));
        if (fallback_wid < 0) return 0;
        mg->dangling_widget_id = fallback_wid;
        mg->dangling_widget_rope = anchor_rope;
        mg->dangling_widget_rope_vid = anchor_vert;
        mg->dangling_hang_len = hang_len;
        return 1;
    }

    mg->dangling_widget_id = wid;
    mg->dangling_widget_rope = to_rope;
    mg->dangling_widget_rope_vid = 1;
    mg->dangling_hang_len = hang_len;
    // Create three small overlays (minus/number/plus) centered on the widget
    // so control regions are allocated at creation time and attached to the
    // widget rope. These overlays provide clickable regions the canvas will
    // route through the existing overlay click handling logic.
    try {
        GP_CanvasContext* cvs = gp_canvas_get_singleton();
        if (cvs) {
            float wpos[3] = {0.0f,0.0f,0.0f};
            // Try to obtain the widget world position; if unavailable (sim
            // provides stubs), fall back to the T-off center computed earlier.
            if (!gp_table_get_widget_position(ctx, mg->dangling_widget_id, wpos)) {
                wpos[0] = (e1x + e2x) * 0.5f;
                wpos[1] = (e1y + e2y) * 0.5f;
                wpos[2] = 0.0f;
            }
            float ox = wpos[0]; float oy = wpos[1];
            unsigned long long ka=0ull,kb=0ull;
            if (gp_canvas_create_overlay_with_leds(cvs, ox - 18.0f, oy - 10.0f, ox + 18.0f, oy + 10.0f, &ka, &kb)) {
                gp_canvas_attach_rope_to_overlay(cvs, ka, kb, mg->dangling_widget_rope);
                printf("gp_table_meta_create_widget: created center overlays for mg=%p wid=%d rope=%d ox=%.1f oy=%.1f ka=%llu kb=%llu\n",
                    (void*)mg, mg->dangling_widget_id, mg->dangling_widget_rope, ox, oy, ka, kb);
                {
                    int oxa=0, oya=0, oxb=0, oyb=0;
                    if (gp_canvas_resolve_overlay_key(cvs, ka, &oxa, &oya)) {
                        printf("  resolved ka -> %d,%d\n", oxa, oya);
                    }
                    if (gp_canvas_resolve_overlay_key(cvs, kb, &oxb, &oyb)) {
                        printf("  resolved kb -> %d,%d\n", oxb, oyb);
                    }
                }
            } else {
                printf("gp_table_meta_create_widget: failed to create center overlay for mg=%p wid=%d ox=%.1f oy=%.1f\n",
                    (void*)mg, mg->dangling_widget_id, ox, oy);
            }
                // populate overlay meta binding so the canvas can dispatch clicks
                if (ka || kb) {
                    gp_canvas_set_overlay_meta(cvs, ka, kb, ctx, mg);
                    mg->overlay_key_a = ka;
                    mg->overlay_key_b = kb;
                }
            unsigned long long la=0ull,lb=0ull;
            if (gp_canvas_create_overlay_with_leds(cvs, ox - 52.0f, oy - 10.0f, ox - 22.0f, oy + 10.0f, &la, &lb)) {
                gp_canvas_attach_rope_to_overlay(cvs, la, lb, mg->dangling_widget_rope);
                printf("gp_table_meta_create_widget: created left overlay for mg=%p wid=%d rope=%d la=%llu lb=%llu\n",
                    (void*)mg, mg->dangling_widget_id, mg->dangling_widget_rope, la, lb);
                {
                    int lxa=0, lya=0, lxb=0, lyb=0;
                    if (gp_canvas_resolve_overlay_key(cvs, la, &lxa, &lya)) printf("  resolved la -> %d,%d\n", lxa, lya);
                    if (gp_canvas_resolve_overlay_key(cvs, lb, &lxb, &lyb)) printf("  resolved lb -> %d,%d\n", lxb, lyb);
                }
            } else {
                printf("gp_table_meta_create_widget: failed to create left overlay for mg=%p wid=%d\n", (void*)mg, mg->dangling_widget_id);
            }
                if (la || lb) {
                    gp_canvas_set_overlay_meta(cvs, la, lb, ctx, mg);
                    if (mg->overlay_key_a == 0ull && mg->overlay_key_b == 0ull) {
                        mg->overlay_key_a = la;
                        mg->overlay_key_b = lb;
                    }
                }
            unsigned long long ra=0ull,rb=0ull;
            if (gp_canvas_create_overlay_with_leds(cvs, ox + 22.0f, oy - 10.0f, ox + 52.0f, oy + 10.0f, &ra, &rb)) {
                gp_canvas_attach_rope_to_overlay(cvs, ra, rb, mg->dangling_widget_rope);
                printf("gp_table_meta_create_widget: created right overlay for mg=%p wid=%d rope=%d ra=%llu rb=%llu\n",
                    (void*)mg, mg->dangling_widget_id, mg->dangling_widget_rope, ra, rb);
                {
                    int rxa=0, rya=0, rxb=0, ryb=0;
                    if (gp_canvas_resolve_overlay_key(cvs, ra, &rxa, &rya)) printf("  resolved ra -> %d,%d\n", rxa, rya);
                    if (gp_canvas_resolve_overlay_key(cvs, rb, &rxb, &ryb)) printf("  resolved rb -> %d,%d\n", rxb, ryb);
                }
            } else {
                printf("gp_table_meta_create_widget: failed to create right overlay for mg=%p wid=%d\n", (void*)mg, mg->dangling_widget_id);
            }
                if (ra || rb) {
                    gp_canvas_set_overlay_meta(cvs, ra, rb, ctx, mg);
                    if (mg->overlay_key_a == 0ull && mg->overlay_key_b == 0ull) {
                        mg->overlay_key_a = ra;
                        mg->overlay_key_b = rb;
                    }
                }
        }
    } catch(...) {}
    return 1;
}

extern "C" int32_t gp_table_meta_destroy_widget(GP_TableContext* ctx, GP_MetaGroup* mg) {
    if (!ctx || !mg) return 0;
    if (mg->dangling_widget_id < 0) return 1;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    int res = rope_sim_destroy_dangling_widget(sim, mg->dangling_widget_id);
    mg->dangling_widget_id = -1;
    return res;
}

extern "C" int32_t gp_table_meta_set_overlay_keys(GP_TableContext* ctx, GP_MetaGroup* mg, unsigned long long key_a, unsigned long long key_b) {
    if (!ctx || !mg) return 0;
    mg->overlay_key_a = key_a;
    mg->overlay_key_b = key_b;
    return 1;
}

extern "C" int32_t gp_table_meta_get_overlay_keys(GP_TableContext* ctx, GP_MetaGroup* mg, unsigned long long* out_key_a, unsigned long long* out_key_b) {
    if (!ctx || !mg || (!out_key_a && !out_key_b)) return 0;
    if (out_key_a) *out_key_a = mg->overlay_key_a;
    if (out_key_b) *out_key_b = mg->overlay_key_b;
    return 1;
}

// Attach/detach an external RopeSim instance to the table context.
// If `sim` is non-null the table will use that simulator for all rope
// allocations and updates. `take_ownership` indicates whether the table
// should destroy the provided simulator when the table is destroyed
// (1 = table destroys the sim, 0 = caller retains ownership). Passing
// `sim == NULL` detaches any external simulator; the table may create
// its own simulator later on demand. Returns 1 on success.
int32_t gp_table_attach_rope_sim(GP_TableContext* ctx, RopeSim* sim, int32_t take_ownership) {
    if (!ctx) return 0;
    // If we currently own a sim, destroy it first
    if (ctx->rope_sim && ctx->rope_sim_owned) {
        rope_sim_destroy(ctx->rope_sim);
    }
    ctx->rope_sim = sim;
    ctx->rope_sim_owned = (sim != nullptr) ? (take_ownership ? 1 : 0) : 0;
    // reset mapping so edges will create ropes in the new sim when next rendered
    ctx->rope_id_to_sim_idx.clear();
    // If attaching a simulator, create rope entries for any existing edges
    // that lack sim indices or persistent uids so we have stable mapping.
    if (ctx->rope_sim) {
        // Prepare row layout helpers
        std::vector<int> row_y0;
        std::vector<int> row_h;
        compute_row_layout(ctx->rows.data(), static_cast<int>(ctx->rows.size()), ctx->st, ctx->geom.height_px, row_y0, row_h);
        auto compute_center_local = [&](uint64_t key, int &outx, int &outy) {
            outx = -1; outy = -1;
            uint32_t r_orig = static_cast<uint32_t>(key >> 32);
            uint32_t c_idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
            uint32_t led = static_cast<uint32_t>(key & 0xFFFFu);
            if (r_orig >= ctx->rows.size()) return;
            const GP_TableRow &row = ctx->rows[static_cast<size_t>(r_orig)];
            if (static_cast<int>(c_idx) < 0 || static_cast<int>(c_idx) >= row.cell_count) return;
            int col_x0[8] = {0}; int col_w[8] = {0};
            compute_columns(ctx->cols.data(), static_cast<int>(ctx->cols.size()), ctx->st.w, ctx->st.name_w, col_x0, col_w);
            const GP_TableCell &cell = row.cells[static_cast<int>(c_idx)];
            int x0 = col_x0[static_cast<int>(c_idx)];
            int cw = col_w[static_cast<int>(c_idx)];
            int y0 = (r_orig < row_y0.size()) ? row_y0[static_cast<size_t>(r_orig)] : (static_cast<int>(r_orig) * ctx->st.row_h);
            int rh = (r_orig < row_h.size()) ? row_h[static_cast<size_t>(r_orig)] : ctx->st.row_h;
            int led_count = 9;
            if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
                int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                if (count == 0) count = (cell.flags & 0xFF);
                if (count == 0) count = 12;
                led_count = count;
            } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
                led_count = 8;
            }
            int eff_w = std::max(1, cw - 4);
            int radius = 4;
            int led_spacing = std::max(radius * 2 + 2, eff_w / std::max(1, led_count + 1));
            int cx0 = x0 + 2 + led_spacing;
            if (static_cast<int>(led) >= 0 && static_cast<int>(led) < led_count) {
                outx = cx0 + static_cast<int>(led) * led_spacing;
                const bool is_image_row = row_find_image_cell(row, nullptr);
                int band_h = is_image_row ? std::min(rh, row_image_header_h(ctx->st, row)) : rh;
                outy = y0 + band_h / 2;
            }
        };

        // Ensure rope_ids vector matches edge count
        if (ctx->rope_ids.size() < ctx->edges.size()) ctx->rope_ids.resize(ctx->edges.size(), 0ull);

        for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
            uint64_t existing_id = (ei < ctx->rope_ids.size()) ? ctx->rope_ids[ei] : 0ull;
            if (existing_id != 0ull && ctx->rope_id_to_sim_idx.find(existing_id) != ctx->rope_id_to_sim_idx.end()) continue; // already has rope mapped
            unsigned long long a = ctx->edges[ei].first;
            unsigned long long b = ctx->edges[ei].second;
            int ax = 0, ay = 0, bx = 0, by = 0;
            compute_center_local(a, ax, ay);
            compute_center_local(b, bx, by);
            int segs = std::max(4, ctx->st.cable_segments);
            float slack = 0.0f;
            float plug_z = -ctx->st.cable_plug_depth;
            int idx = rope_sim_add_rope3(ctx->rope_sim, static_cast<float>(ax), static_cast<float>(ay), plug_z, static_cast<float>(bx), static_cast<float>(by), plug_z, segs, slack);
            uint64_t rid = 0ull;
            if (existing_id != 0ull) {
                rid = existing_id;
            } else {
                rid = gp_canvas_generate_id(gp_canvas_get_singleton(), 0ull);
            }
            ctx->rope_ids[ei] = rid;
            ctx->rope_id_to_sim_idx[rid] = idx;
            printf("gp_table_attach_rope_sim: ctx=%p backfilled rope idx=%d id=%llu\n", (void*)ctx, idx, (unsigned long long)rid);
            {
                GP_CanvasContext* cvs2 = gp_canvas_get_singleton();
                if (cvs2) {
                    uint64_t tmp2 = rid;
                    gp_canvas_register_table_rope_ids_from_array(cvs2, ctx, &tmp2, 1);
                }
            }
        }
    }
    if (GP_CanvasContext* cvs = gp_canvas_get_singleton()) {
        gp_canvas_mark_rope_map_dirty(cvs);
    }
    return 1;
}

// Enqueue operations -------------------------------------------------------
int32_t gp_table_enqueue_add_edge(GP_TableContext* ctx, unsigned long long a, unsigned long long b) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
    GP_TableContext::PendingOp op;
    op.type = GP_TableContext::PENDING_OP_ADD_EDGE;
    op.a_key = a;
    op.b_key = b;
    ctx->pending_ops.push_back(op);
    return 1;
}

int32_t gp_table_enqueue_clear_edges(GP_TableContext* ctx) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
    GP_TableContext::PendingOp op;
    op.type = GP_TableContext::PENDING_OP_CLEAR_EDGES;
    ctx->pending_ops.push_back(op);
    return 1;
}

int32_t gp_table_enqueue_edge_subscribe_ex(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, int32_t start_at_head) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
    GP_TableContext::PendingOp op;
    op.type = GP_TableContext::PENDING_OP_SUBSCRIBE_EDGE;
    op.edge_idx = edge_idx;
    op.sub_key = subscriber_key;
    op.start_at_head = start_at_head;
    ctx->pending_ops.push_back(op);
    return 1;
}

int32_t gp_table_enqueue_edge_unsubscribe(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
    GP_TableContext::PendingOp op;
    op.type = GP_TableContext::PENDING_OP_UNSUBSCRIBE_EDGE;
    op.edge_idx = edge_idx;
    op.sub_key = subscriber_key;
    ctx->pending_ops.push_back(op);
    return 1;
}

int32_t gp_table_enqueue_bind_stage_port(GP_TableContext* ctx, unsigned long long led_key, GP_StageContext* stage, int32_t is_output, int32_t channel) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
    GP_TableContext::PendingOp op;
    op.type = GP_TableContext::PENDING_OP_BIND_STAGE;
    op.a_key = led_key;
    op.stage = stage;
    op.is_output = is_output;
    op.channel = channel;
    ctx->pending_ops.push_back(op);
    return 1;
}

// Set/Get per-edge subgroup flags (color hues, BYREF bit, etc.). Caller should
// call `gp_table_apply_pending_ops` or similar manager-side sync to have
// reconfiguration take effect; this function updates the context and triggers
// an immediate FIFO sync for that edge.
// Forward declare file-scope sync helper so earlier callers can invoke it.
static void sync_edge_tensor_for_idx(GP_TableContext* ctx, size_t ei);
int32_t gp_table_set_edge_subgroup_flags(GP_TableContext* ctx, int32_t edge_idx, uint32_t flags) {
    if (!ctx) return 0;
    if (edge_idx < 0 || static_cast<size_t>(edge_idx) >= ctx->edges.size()) return 0;
    ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)] = flags;
    // forward declaration may be below; ensure symbol visible by declaring prototype above in file
    sync_edge_tensor_for_idx(ctx, static_cast<size_t>(edge_idx));
    return 1;
}

int32_t gp_table_get_edge_subgroup_flags(GP_TableContext* ctx, int32_t edge_idx, uint32_t* out_flags) {
    if (!ctx || !out_flags) return 0;
    if (edge_idx < 0 || static_cast<size_t>(edge_idx) >= ctx->edges.size()) return 0;
    *out_flags = ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)];
    return 1;
}

int32_t gp_table_enqueue_unbind_stage_port(GP_TableContext* ctx, unsigned long long led_key) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
    GP_TableContext::PendingOp op;
    op.type = GP_TableContext::PENDING_OP_UNBIND_STAGE;
    op.a_key = led_key;
    ctx->pending_ops.push_back(op);
    return 1;
}

int32_t gp_table_set_sim_enabled(GP_TableContext* ctx, int32_t enabled) {
    if (!ctx) return 0;
    ctx->sim_enabled = (enabled ? 1 : 0);
    return 1;
}

int32_t gp_table_get_sim_enabled(GP_TableContext* ctx, int32_t* out_enabled) {
    if (!ctx || !out_enabled) return 0;
    *out_enabled = ctx->sim_enabled;
    return 1;
}

// Apply pending ops (manager thread should call this before scheduling)
int32_t gp_table_apply_pending_ops(GP_TableContext* ctx) {
    if (!ctx) return 0;
    std::vector<GP_TableContext::PendingOp> ops;
    {
        std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
        if (ctx->pending_ops.empty()) return 1;
        ops.swap(ctx->pending_ops);
    }
    for (const auto &op : ops) {
        switch (op.type) {
        case GP_TableContext::PENDING_OP_ADD_EDGE:
            gp_table_add_edge(ctx, op.a_key, op.b_key);
            break;
        case GP_TableContext::PENDING_OP_CLEAR_EDGES:
            gp_table_clear_edges(ctx);
            break;
        case GP_TableContext::PENDING_OP_SUBSCRIBE_EDGE:
            gp_table_edge_subscribe_ex(ctx, op.edge_idx, op.sub_key, op.start_at_head);
            break;
        case GP_TableContext::PENDING_OP_UNSUBSCRIBE_EDGE:
            gp_table_edge_unsubscribe(ctx, op.edge_idx, op.sub_key);
            break;
        case GP_TableContext::PENDING_OP_BIND_STAGE:
            gp_table_bind_stage_port(ctx, op.a_key, op.stage, op.is_output, op.channel);
            break;
        case GP_TableContext::PENDING_OP_UNBIND_STAGE:
            gp_table_unbind_stage_port(ctx, op.a_key);
            break;
        default:
            break;
        }
    }
    return 1;
}

// Network-only snapshot implementation (no locking; caller must be manager thread)
int32_t gp_table_snapshot_network_size(GP_TableContext* ctx, int32_t* out_node_count, int32_t* out_edge_count, uint64_t* out_stamp) {
    if (!ctx || !out_node_count || !out_edge_count || !out_stamp) return 0;
    // Count unique endpoint keys
    std::unordered_set<uint64_t> keys;
    keys.reserve(ctx->edges.size() * 2 + 1);
    for (const auto &e : ctx->edges) {
        keys.insert(e.first);
        keys.insert(e.second);
    }
    *out_node_count = static_cast<int32_t>(keys.size());
    *out_edge_count = static_cast<int32_t>(ctx->edges.size());
    // Stamp: simple generation combining edge count and next_edge_id to detect changes
    uint64_t stamp = (static_cast<uint64_t>(ctx->edges.size()) << 32) ^ (ctx->next_edge_id & 0xffffffffull);
    *out_stamp = stamp;
    return 1;
}

int32_t gp_table_snapshot_network_fill(GP_TableContext* ctx, uint64_t* node_buf, int32_t node_buf_len, GP_TableEdgeSnapshot* edge_buf, int32_t edge_buf_len, uint64_t expected_stamp) {
    if (!ctx || !node_buf || !edge_buf) return 0;
    // Recompute stamp
    uint64_t stamp = (static_cast<uint64_t>(ctx->edges.size()) << 32) ^ (ctx->next_edge_id & 0xffffffffull);
    if (expected_stamp != stamp) return 0; // caller should retry size/fill

    // Build a node index map in caller-visible order: insert as discovered while scanning edges
    std::unordered_map<uint64_t,uint32_t> idx;
    idx.reserve(ctx->edges.size() * 2 + 1);
    uint32_t next_idx = 0;
    for (const auto &e : ctx->edges) {
        if (idx.find(e.first) == idx.end()) {
            if (next_idx >= static_cast<uint32_t>(node_buf_len)) return 0;
            idx[e.first] = next_idx;
            node_buf[next_idx] = e.first;
            ++next_idx;
        }
        if (idx.find(e.second) == idx.end()) {
            if (next_idx >= static_cast<uint32_t>(node_buf_len)) return 0;
            idx[e.second] = next_idx;
            node_buf[next_idx] = e.second;
            ++next_idx;
        }
    }
    if (next_idx > static_cast<uint32_t>(node_buf_len)) return 0;
    // Fill edges
    if (static_cast<int32_t>(ctx->edges.size()) > edge_buf_len) return 0;
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto &e = ctx->edges[i];
        auto it_a = idx.find(e.first);
        auto it_b = idx.find(e.second);
        if (it_a == idx.end() || it_b == idx.end()) return 0; // should not happen
        edge_buf[i].a_idx = it_a->second;
        edge_buf[i].b_idx = it_b->second;
        edge_buf[i].edge_uid = (i < ctx->edge_ids.size()) ? ctx->edge_ids[i] : 0ull;
    }
    return 1;
}

RopeSim* gp_table_get_rope_sim(GP_TableContext* ctx) {
    if (!ctx) return nullptr;
    return ctx->rope_sim;
}

int32_t gp_table_set_step_callback(GP_TableContext* ctx, GP_TableStepFn cb, void* user) {
    if (!ctx) return 0;
    ctx->step_callback = cb;
    ctx->step_user = user;
    return 1;
}

int32_t gp_table_clear_step_callback(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ctx->step_callback = nullptr;
    ctx->step_user = nullptr;
    return 1;
}

int32_t gp_table_step(GP_TableContext* ctx, const float* inputs, int32_t in_count, float* outputs, int32_t out_count, double dt) {
    if (!ctx) return 0;
    if (!ctx->step_callback) return 0;
    // Defensive: allow null arrays as zero-length
    if ((in_count > 0 && !inputs) || (out_count > 0 && !outputs)) return 0;
    try {
        ctx->step_callback(ctx->step_user, inputs, in_count, outputs, out_count, dt);
        return 1;
    } catch (...) {
        return 0;
    }
}

int32_t gp_table_set_actions(GP_TableContext* ctx, const GP_TableAction* actions, int32_t count) {
    if (!ctx) return 0;
    if (count < 0) return 0;
    if (count > 0 && !actions) return 0;
    if (count == 0) {
        ctx->actions.clear();
        return 1;
    }
    ctx->actions.assign(actions, actions + count);
    return 1;
}

int32_t gp_table_set_action_callback(GP_TableContext* ctx, GP_TableActionFn cb, void* user) {
    if (!ctx) return 0;
    ctx->action_callback = cb;
    ctx->action_user = user;
    return 1;
}

int32_t gp_table_clear_action_callback(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ctx->action_callback = nullptr;
    ctx->action_user = nullptr;
    return 1;
}

int32_t gp_table_set_key_callback(GP_TableContext* ctx, GP_TableKeyFn cb, void* user) {
    if (!ctx) return 0;
    ctx->key_callback = cb;
    ctx->key_user = user;
    return 1;
}

int32_t gp_table_clear_key_callback(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ctx->key_callback = nullptr;
    ctx->key_user = nullptr;
    return 1;
}

int32_t gp_table_on_key(GP_TableContext* ctx, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
    if (!ctx) return 0;
    if (!ctx->key_callback) return 0;
    try {
        ctx->key_callback(ctx->key_user, key, scancode, action, mods);
        return 1;
    } catch (...) {
        return 0;
    }
}

static void recompute_geom(GP_TableContext* ctx) {
    if (!ctx) return;
    ctx->geom.width_px = ctx->st.w;
    int h_sum = 0;
    for (const auto& r : ctx->rows) {
        int rh = row_desired_height_px(ctx->st, r);
        h_sum += std::max(1, rh);
    }
    ctx->geom.height_px = std::max<int32_t>(1, static_cast<int32_t>(h_sum));
    compute_columns(ctx->cols.data(), static_cast<int>(ctx->cols.size()), ctx->st.w, ctx->st.name_w, ctx->geom.col_x0, ctx->geom.col_w);
}

// Ensure the edge FIFO list matches the edge list length.
static void ensure_edge_fifos(GP_TableContext* ctx) {
    if (!ctx) return;
    while (ctx->edge_fifos.size() < ctx->edges.size()) {
        EdgeTensorFifo fifo;
        fifo.configure_default();
        fifo.set_friction_regions(ctx->st.cable_fifo_friction_regions);
        ctx->edge_fifos.push_back(std::move(fifo));
    }
    if (ctx->edge_fifos.size() > ctx->edges.size()) {
        ctx->edge_fifos.resize(ctx->edges.size());
    }
    for (auto &fifo : ctx->edge_fifos) {
        fifo.set_friction_regions(ctx->st.cable_fifo_friction_regions);
    }
    while (ctx->edge_batch_metadata.size() < ctx->edges.size()) {
        ctx->edge_batch_metadata.emplace_back(GP_TableEdgeBatchMetadata{});
    }
    if (ctx->edge_batch_metadata.size() > ctx->edges.size()) {
        ctx->edge_batch_metadata.resize(ctx->edges.size());
    }
    while (ctx->edge_subgroup_flags.size() < ctx->edges.size()) {
        ctx->edge_subgroup_flags.push_back(0u);
    }
    if (ctx->edge_subgroup_flags.size() > ctx->edges.size()) {
        ctx->edge_subgroup_flags.resize(ctx->edges.size());
    }
    // keep subscriber slot maps in sync with edges
    while (ctx->edge_subscriber_slots.size() < ctx->edges.size()) ctx->edge_subscriber_slots.emplace_back();
    if (ctx->edge_subscriber_slots.size() > ctx->edges.size()) ctx->edge_subscriber_slots.resize(ctx->edges.size());
}

// Apply stage port bindings to an edge's FIFO: output keys claim writer, input keys subscribe.
static void sync_edge_tensor_for_idx(GP_TableContext* ctx, size_t ei) {
    if (!ctx) return;
    ensure_edge_fifos(ctx);
    if (ei >= ctx->edges.size() || ei >= ctx->edge_fifos.size()) return;
    const auto &edge = ctx->edges[ei];
    EdgeTensorFifo &fifo = ctx->edge_fifos[ei];
    auto bind_one = [&](uint64_t key) {
        auto it = ctx->stage_ports.find(key);
        if (it == ctx->stage_ports.end()) return;
        const StagePortBinding &b = it->second;
        if (b.is_output) {
            fifo.maybe_claim_writer(key);
        } else {
            fifo.subscribe(key, /*start_at_head=*/true);
        }
    };
    bind_one(edge.first);
    bind_one(edge.second);

    // If edge has BYREF subgroup flag, ensure FIFO stride can carry pointer-sized
    // payloads. We do this in-place so the FIFO object does not need to be
    // destroyed and re-created; existing samples are preserved where possible.
    if (ei < ctx->edge_subgroup_flags.size()) {
        uint32_t f = ctx->edge_subgroup_flags[ei];
        if (flags_imply_byref(f)) {
            // ensure stride can hold a pointer
            EdgeTensorFifo &ef = ctx->edge_fifos[ei];
            ef.ensure_stride_for_bytes(sizeof(void*));
        }
    }
}

static void sync_edge_tensors_for_key(GP_TableContext* ctx, uint64_t key) {
    if (!ctx) return;
    ensure_edge_fifos(ctx);
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto &e = ctx->edges[i];
        if (e.first == key || e.second == key) {
            sync_edge_tensor_for_idx(ctx, i);
        }
    }
}

static void draw_circle_outline(uint8_t* img, int w, int h, int pitch, int cx, int cy, int r, int thickness, Color c) {
    if (!img || r <= 0) return;
    const int r2 = r * r;
    const int outer = r + thickness;
    const int outer2 = outer * outer;
    for (int dy = -outer; dy <= outer; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= h) continue;
        for (int dx = -outer; dx <= outer; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= w) continue;
            int d2 = dx * dx + dy * dy;
            if (d2 >= r2 && d2 <= outer2) {
                uint8_t* p = img + y * pitch + x * 4;
                p[0] = c.r; p[1] = c.g; p[2] = c.b; p[3] = c.a;
            }
        }
    }
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

// Draw a filled circle with blending (soft core) used for cable sampling points
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
            // simple linear falloff alpha across radius
            float t = 1.0f - (std::sqrt((float)d2) / (float)r);
            uint8_t sa = static_cast<uint8_t>(std::lround(c.a * t));
            uint8_t sr = c.r;
            uint8_t sg = c.g;
            uint8_t sb = c.b;
            uint8_t* dst = img + y * pitch + x * 4;
            blend_pixel(dst, sr, sg, sb, sa);
        }
    }
}

// Draw a smeared segment using the segment tangent as the cross-section normal.
// This paints a continuous tube between (x1,y1) and (x2,y2) with radius and soft falloff.
static void draw_segment_smear(uint8_t* img, int w, int h, int pitch, float x1, float y1, float x2, float y2, int radius, Color c, float alpha_scale = 1.0f) {
    if (!img || radius <= 0) return;
    float dx = x2 - x1;
    float dy = y2 - y1;
    float len2 = dx*dx + dy*dy;
    if (len2 <= 1e-6f) {
        // fallback to blob
        draw_blob_blend(img, w, h, pitch, int(std::lround(x1)), int(std::lround(y1)), radius, c);
        return;
    }
    float len = std::sqrt(len2);
    // normal pointing to the left of the segment (perpendicular)
    float nx = -dy / len;
    float ny = dx / len;
    float r = float(radius);
    float rplus = r + 1.0f;
    int minx = static_cast<int>(std::floor(std::min(x1,x2) - rplus));
    int maxx = static_cast<int>(std::ceil(std::max(x1,x2) + rplus));
    int miny = static_cast<int>(std::floor(std::min(y1,y2) - rplus));
    int maxy = static_cast<int>(std::ceil(std::max(y1,y2) + rplus));
    minx = std::max(minx, 0);
    miny = std::max(miny, 0);
    maxx = std::min(maxx, w - 1);
    maxy = std::min(maxy, h - 1);
    for (int y = miny; y <= maxy; ++y) {
        for (int x = minx; x <= maxx; ++x) {
            // compute projection along segment
            float vx = float(x) - x1;
            float vy = float(y) - y1;
            float proj = (vx * dx + vy * dy) / len2;
            float t = std::clamp(proj, 0.0f, 1.0f);
            float cxp = x1 + dx * t;
            float cyp = y1 + dy * t;
            // lateral distance to the segment centerline
            float lx = float(x) - cxp;
            float ly = float(y) - cyp;
            float lateral = std::abs(lx * nx + ly * ny);
            if (lateral > r) continue;
            // simple linear falloff by lateral distance
            float fall = 1.0f - (lateral / r);
            uint8_t sa = static_cast<uint8_t>(std::lround(float(c.a) * fall * alpha_scale));
            uint8_t* dst = img + y * pitch + x * 4;
            blend_pixel(dst, c.r, c.g, c.b, sa);
        }
    }
}

// High-quality parametric SDF segment: uses endpoint tangents to interpolate a cross-section
// and computes a smooth Gaussian falloff in cross-section distance.
static void draw_segment_parametric_sdf(uint8_t* img, int w, int h, int pitch,
    float x1, float y1, float x2, float y2,
    float tx1, float ty1, float tx2, float ty2,
    int radius, Color c, float alpha_scale = 1.0f, float tint_strength = 0.0f) {
    if (!img || radius <= 0) return;
    float dx = x2 - x1;
    float dy = y2 - y1;
    float len2 = dx*dx + dy*dy;
    if (len2 <= 1e-6f) {
        draw_blob_blend(img, w, h, pitch, int(std::lround(x1)), int(std::lround(y1)), radius, c);
        return;
    }
    float r = float(radius);
    float rplus = r + 2.0f;
    int minx = static_cast<int>(std::floor(std::min(x1,x2) - rplus));
    int maxx = static_cast<int>(std::ceil(std::max(x1,x2) + rplus));
    int miny = static_cast<int>(std::floor(std::min(y1,y2) - rplus));
    int maxy = static_cast<int>(std::ceil(std::max(y1,y2) + rplus));
    minx = std::max(minx, 0);
    miny = std::max(miny, 0);
    maxx = std::min(maxx, w - 1);
    maxy = std::min(maxy, h - 1);

    // pre-normalize endpoint tangents
    float t1len = std::sqrt(tx1*tx1 + ty1*ty1);
    float t2len = std::sqrt(tx2*tx2 + ty2*ty2);
    if (t1len <= 1e-6f) { tx1 = dx; ty1 = dy; t1len = std::sqrt(dx*dx+dy*dy); }
    if (t2len <= 1e-6f) { tx2 = dx; ty2 = dy; t2len = std::sqrt(dx*dx+dy*dy); }
    tx1 /= t1len; ty1 /= t1len; tx2 /= t2len; ty2 /= t2len;

    // Gaussian width parameter (so radius maps to ~3-sigma). Use smoother falloff.
    float sigma = r / 2.0f;
    float inv2sig2 = 1.0f / (2.0f * sigma * sigma);

    for (int y = miny; y <= maxy; ++y) {
        for (int x = minx; x <= maxx; ++x) {
            // project point onto the segment (param t in [0,1])
            float vx = float(x) - x1;
            float vy = float(y) - y1;
            float proj = (vx * dx + vy * dy) / len2;
            float t = std::clamp(proj, 0.0f, 1.0f);
            float cxp = x1 + dx * t;
            float cyp = y1 + dy * t;

            // interpolate tangent
            float tx = (1.0f - t) * tx1 + t * tx2;
            float ty = (1.0f - t) * ty1 + t * ty2;
            float tlen = std::sqrt(tx*tx + ty*ty);
            if (tlen <= 1e-6f) continue;
            tx /= tlen; ty /= tlen;

            // normal (perp)
            float nx = -ty;
            float ny = tx;

            float lx = float(x) - cxp;
            float ly = float(y) - cyp;
            float lateral = std::abs(lx * nx + ly * ny);
            if (lateral > r) continue;

            // Gaussian falloff based on lateral distance
            float gauss = std::exp(- (lateral * lateral) * inv2sig2);

            // compute tint factor based on radial proximity (1 at center, 0 at radius)
            float radial = std::clamp(1.0f - (lateral / r), 0.0f, 1.0f);
            float tint_amt = std::clamp(tint_strength * radial, 0.0f, 1.0f);

            // blend between neutral jacket (light grey) and provided color by tint_amt
            float base_r = 200.0f;
            float base_g = 200.0f;
            float base_b = 200.0f;
            float pr = base_r * (1.0f - tint_amt) + float(c.r) * tint_amt;
            float pg = base_g * (1.0f - tint_amt) + float(c.g) * tint_amt;
            float pb = base_b * (1.0f - tint_amt) + float(c.b) * tint_amt;

            uint8_t sa = static_cast<uint8_t>(std::lround(float(c.a) * gauss * alpha_scale));
            uint8_t* dst = img + y * pitch + x * 4;
            blend_pixel(dst, static_cast<uint8_t>(std::lround(pr)), static_cast<uint8_t>(std::lround(pg)), static_cast<uint8_t>(std::lround(pb)), sa);
        }
    }
}

static void draw_rope_curve_blend_rgb(uint8_t* img, int w, int h, int pitch, const float* verts, int count,
    int jacket_px, int jacket_border, Color col_a, Color col_b, float intensity) {
    if (!img || !verts || count < 2) return;

    auto get = [&](int idx) {
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts[2*idx+0], verts[2*idx+1]);
    };
    auto catmull = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        return std::pair<float,float>(x,y);
    };
    auto catmull_deriv = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float dx = 0.5f * ((-p0.first + p2.first) + 2.0f * (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t + 3.0f * (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t2);
        float dy = 0.5f * ((-p0.second + p2.second) + 2.0f * (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t + 3.0f * (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t2);
        return std::pair<float,float>(dx, dy);
    };

    std::vector<std::pair<float,float>> samples;
    std::vector<std::pair<float,float>> tangents;
    samples.reserve((count - 1) * 8);
    tangents.reserve((count - 1) * 8);
    for (int i = 0; i < count - 1; ++i) {
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        int s_start = (i == 0) ? 0 : 1; // avoid duplicate samples at segment boundaries
        for (int s = s_start; s <= n; ++s) {
            float t = float(s) / float(n);
            auto p = catmull(i, t);
            auto d = catmull_deriv(i, t);
            samples.emplace_back(p.first, p.second);
            tangents.emplace_back(d.first, d.second);
        }
    }
    if (samples.empty()) return;

    auto lerp_color = [&](const Color &a, const Color &b, float t) {
        float tt = std::clamp(t, 0.0f, 1.0f);
        Color out;
        out.r = static_cast<uint8_t>(std::lround(float(a.r) * (1.0f - tt) + float(b.r) * tt));
        out.g = static_cast<uint8_t>(std::lround(float(a.g) * (1.0f - tt) + float(b.g) * tt));
        out.b = static_cast<uint8_t>(std::lround(float(a.b) * (1.0f - tt) + float(b.b) * tt));
        out.a = static_cast<uint8_t>(std::lround(float(a.a) * (1.0f - tt) + float(b.a) * tt));
        return out;
    };

    int eff_jacket = std::max(1, jacket_px - 1);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float u = float(i) / float(std::max<size_t>(1, samples.size() - 1));
        Color jacket_col = lerp_color(col_a, col_b, u);
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, eff_jacket, jacket_col, 1.0f, intensity);
    }

    int core_r = std::max(1, jacket_px - jacket_border - 0);
    uint8_t core_alpha = static_cast<uint8_t>(std::lround(255.0f * std::clamp(intensity, 0.0f, 1.0f) * 0.18f));
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float u = float(i) / float(std::max<size_t>(1, samples.size() - 1));
        Color core_col = lerp_color(col_a, col_b, u);
        core_col.a = core_alpha;
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, core_r, core_col, 1.0f, intensity);
    }
}

static void draw_rope_curve_blend_rgb_falloff(uint8_t* img, int w, int h, int pitch, const float* verts, int count,
    int jacket_px, int jacket_border, Color col_a, Color col_b, float glow_a, float glow_b, float decay) {
    if (!img || !verts || count < 2) return;

    auto get = [&](int idx) {
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts[2*idx+0], verts[2*idx+1]);
    };
    auto catmull = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        return std::pair<float,float>(x,y);
    };
    auto catmull_deriv = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float dx = 0.5f * ((-p0.first + p2.first) + 2.0f * (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t + 3.0f * (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t2);
        float dy = 0.5f * ((-p0.second + p2.second) + 2.0f * (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t + 3.0f * (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t2);
        return std::pair<float,float>(dx, dy);
    };

    std::vector<std::pair<float,float>> samples;
    std::vector<std::pair<float,float>> tangents;
    samples.reserve((count - 1) * 8);
    tangents.reserve((count - 1) * 8);
    for (int i = 0; i < count - 1; ++i) {
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        int s_start = (i == 0) ? 0 : 1; // avoid duplicate samples at segment boundaries
        for (int s = s_start; s <= n; ++s) {
            float t = float(s) / float(n);
            auto p = catmull(i, t);
            auto d = catmull_deriv(i, t);
            samples.emplace_back(p.first, p.second);
            tangents.emplace_back(d.first, d.second);
        }
    }
    if (samples.empty()) return;

    auto mix = [&](const Color &a, const Color &b, float t) {
        float tt = std::clamp(t, 0.0f, 1.0f);
        Color out;
        out.r = static_cast<uint8_t>(std::lround(float(a.r) * (1.0f - tt) + float(b.r) * tt));
        out.g = static_cast<uint8_t>(std::lround(float(a.g) * (1.0f - tt) + float(b.g) * tt));
        out.b = static_cast<uint8_t>(std::lround(float(a.b) * (1.0f - tt) + float(b.b) * tt));
        out.a = static_cast<uint8_t>(std::lround(float(a.a) * (1.0f - tt) + float(b.a) * tt));
        return out;
    };

    int eff_jacket = std::max(1, jacket_px - 1);
    int core_r = std::max(1, jacket_px - jacket_border - 0);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float u = float(i) / float(std::max<size_t>(1, samples.size() - 1));
        float ia = glow_a * std::exp(-decay * u);
        float ib = glow_b * std::exp(-decay * (1.0f - u));
        float total = (ia + ib) * 1.6f;
        if (total <= 1e-4f) continue;
        float t = (total > 0.0f) ? (ib / total) : 0.0f;
        Color lit = mix(col_a, col_b, t);
        float tint = std::clamp(total, 0.0f, 1.0f);

        Color jacket_col = lit;
        jacket_col.a = 26;
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, eff_jacket, jacket_col, tint, tint);

        Color core_col = lit;
        core_col.a = static_cast<uint8_t>(std::lround(255.0f * 0.35f * tint));
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, core_r, core_col, 1.0f, tint);
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

static void draw_glow_blob(uint8_t* img, int w, int h, int pitch, int cx, int cy, int radius, float strength01, Color c) {
    if (!img) return;
    float s = std::max(0.0f, std::min(1.0f, strength01));
    if (s <= 0.0f) return;
    // Slightly larger falloff radius than the LED core to create diffusion.
    int glow_r = std::max(radius + 2, static_cast<int>(std::lround(float(radius) * (2.2f + 0.8f * s))));
    Color halo = c;
    halo.a = static_cast<uint8_t>(std::lround(float(c.a) * (0.55f + 0.45f * s)));
    draw_blob_blend(img, w, h, pitch, cx, cy, glow_r, halo);
    // Soft core to keep the center lively.
    Color core = c;
    core.a = static_cast<uint8_t>(std::lround(float(c.a) * std::min(1.0f, s * 1.4f)));
    draw_blob_blend(img, w, h, pitch, cx, cy, std::max(radius, glow_r / 3), core);
}

// Draw a droopy blended cable between two points.
static void draw_cable_blend(uint8_t* img, int w, int h, int pitch, int ax, int ay, int bx, int by, int jacket_px, int jacket_border, Color core_col, int segments, float sag_factor, float relax_v) {
    if (!img) return;
    if (segments < 4) segments = 4;
    float dx = float(bx - ax);
    float dy = float(by - ay);
    float dist = std::sqrt(dx*dx + dy*dy);
    float sag = dist * sag_factor;
    // core color modulated by relax value but keep a minimum visibility so cables never fully disappear
    Color core = core_col;
    float rv = std::min(1.0f, std::max(0.0f, relax_v));
    const float min_vis = 0.18f;
    float vis = std::max(min_vis, rv);
    core.a = static_cast<uint8_t>(std::lround(core.a * vis));
    // jacket color: use faint neutral grey to avoid dark edges
    Color jacket{200,200,200,13};
    Color jacket_edge{200,200,200,20};
    // choose samples so blob spacing is <= ~0.6 * jacket_px to avoid visible gaps
    int min_seg_for_spacing = 1;
    if (jacket_px > 0) min_seg_for_spacing = static_cast<int>(std::ceil(dist / (std::max(1.0f, float(jacket_px) * 0.6f))));
    int use_segments = std::max(segments, std::max(4, min_seg_for_spacing));
    // sample points and draw overlapping blobs for a continuous tube
    for (int si = 0; si <= use_segments; ++si) {
        float t = float(si) / float(use_segments);
        float px = float(ax) + dx * t;
        float py = float(ay) + dy * t + sag * std::sin(3.14159265f * t);
        int ipx = static_cast<int>(std::lround(px));
        int ipy = static_cast<int>(std::lround(py));
        // smear segment from previous sample to this sample (avoids per-sample caps)
        if (si > 0) {
            float px0 = float(ax) + dx * float(si - 1) / float(use_segments);
            float py0 = float(ay) + dy * float(si - 1) / float(use_segments) + sag * std::sin(3.14159265f * (float(si - 1) / float(use_segments)));
            // outer jacket smear
            draw_segment_smear(img, w, h, pitch, px0, py0, px, py, jacket_px, jacket);
            // thin jacket edge smear
            draw_segment_smear(img, w, h, pitch, px0, py0, px, py, std::max(1, jacket_px - 1), jacket_edge);
            // core smear
            int core_r = std::max(1, jacket_px - jacket_border);
            Color corec = core;
            corec.a = static_cast<uint8_t>(std::lround(corec.a * 1.0f));
            draw_segment_smear(img, w, h, pitch, px0, py0, px, py, core_r, corec);
        }
    }
    // end plugs: blended caps so endpoints remain joined to smears
    draw_blob_blend(img, w, h, pitch, ax, ay, jacket_px, Color{200,200,200,13});
    draw_blob_blend(img, w, h, pitch, bx, by, jacket_px, Color{200,200,200,13});
    // inner core caps (blended)
    Color corecap = core_col;
    corecap.a = static_cast<uint8_t>(std::lround(corecap.a * 0.2f));
    draw_blob_blend(img, w, h, pitch, ax, ay, std::max(1, jacket_px - jacket_border), corecap);
    draw_blob_blend(img, w, h, pitch, bx, by, std::max(1, jacket_px - jacket_border), corecap);
}

// Helper: draw a blended thick segment with optional alpha scale.
static void draw_segment_blend(uint8_t* img, int w, int h, int pitch, float x1, float y1, float x2, float y2, int radius, Color col, float alpha_scale = 1.0f) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float len2 = dx*dx + dy*dy;
    float rplus = float(radius) + 1.0f; // allow soft edge
    int minx = static_cast<int>(std::floor(std::min(x1,x2) - rplus));
    int maxx = static_cast<int>(std::ceil(std::max(x1,x2) + rplus));
    int miny = static_cast<int>(std::floor(std::min(y1,y2) - rplus));
    int maxy = static_cast<int>(std::ceil(std::max(y1,y2) + rplus));
    minx = std::max(minx, 0);
    miny = std::max(miny, 0);
    maxx = std::min(maxx, w - 1);
    maxy = std::min(maxy, h - 1);
    for (int py = miny; py <= maxy; ++py) {
        for (int px = minx; px <= maxx; ++px) {
            float cx = px + 0.5f;
            float cy = py + 0.5f;
            float t = 0.0f;
            if (len2 > 1e-6f) {
                t = ((cx - x1) * dx + (cy - y1) * dy) / len2;
                if (t < 0.0f) t = 0.0f;
                else if (t > 1.0f) t = 1.0f;
            }
            float closestX = x1 + dx * t;
            float closestY = y1 + dy * t;
            float ddx = cx - closestX;
            float ddy = cy - closestY;
            float dist = std::sqrt(ddx*ddx + ddy*ddy);
            if (dist <= rplus) {
                float coverage = 1.0f - (dist / rplus);
                float a_scaled = float(col.a) * coverage * alpha_scale;
                if (a_scaled <= 0.0f) continue;
                Color cc = col;
                cc.a = static_cast<uint8_t>(std::lround(std::min(255.0f, a_scaled)));
                uint8_t* dstp = img + py * pitch + px * 4;
                blend_pixel(dstp, cc.r, cc.g, cc.b, cc.a);
            }
        }
    }
}

// Orthographic projection of 3D rope verts into 2D table space with a subtle tilt.
static void project_rope_vertices_ortho(const float* verts3, int count, float tilt_x, float tilt_y, std::vector<float>& out_xy, std::vector<float>& out_z, float &out_min_z, float &out_max_z) {
    out_xy.resize(static_cast<std::size_t>(count) * 2);
    out_z.resize(static_cast<std::size_t>(count));
    out_min_z = std::numeric_limits<float>::max();
    out_max_z = std::numeric_limits<float>::lowest();
    for (int i = 0; i < count; ++i) {
        float x = verts3[3*i+0];
        float y = verts3[3*i+1];
        float z = verts3[3*i+2];
        out_z[static_cast<std::size_t>(i)] = z;
        out_xy[2*i+0] = x + tilt_x * z;
        out_xy[2*i+1] = y + tilt_y * z;
        out_min_z = std::min(out_min_z, z);
        out_max_z = std::max(out_max_z, z);
    }
    if (out_min_z > out_max_z) {
        out_min_z = out_max_z = 0.0f;
    }
}

// C API wrapper: return projected 2D verts for a rope known to this table.
extern "C" int32_t gp_table_get_projected_rope_vertices(GP_TableContext* ctx, int32_t rope_idx, float* out_xy, int32_t max_count) {
    if (!ctx || !out_xy) return 0;
    RopeSim* sim = gp_table_get_rope_sim(ctx);
    if (!sim) return 0;
    if (rope_idx < 0) return 0;
    int vc = rope_sim_get_vertex_count(sim, rope_idx);
    if (vc <= 0) return 0;
    if (max_count < 2 * vc) return 0;

    std::vector<float> verts3(static_cast<size_t>(vc) * 3);
    int got = rope_sim_get_vertices3(sim, rope_idx, verts3.data(), static_cast<int>(verts3.size()));
    if (got != vc) return 0;

    std::vector<float> proj_xy;
    std::vector<float> proj_z;
    float min_z = 0.0f, max_z = 0.0f;
    project_rope_vertices_ortho(verts3.data(), got, ctx->st.cable_tilt_x, ctx->st.cable_tilt_y, proj_xy, proj_z, min_z, max_z);

    // copy into out_xy interleaved
    for (int i = 0; i < vc; ++i) {
        out_xy[2*i+0] = proj_xy[2*i+0];
        out_xy[2*i+1] = proj_xy[2*i+1];
    }
    return vc;
}

extern "C" int32_t gp_table_set_prospective_rope_index(GP_TableContext* ctx, int32_t rope_idx) {
    if (!ctx) return 0;
    ctx->prospective_rope_idx = rope_idx;
    return 1;
}

// Build Catmull-Rom samples for a rope and optionally depth samples aligned with them.
static void build_rope_samples_with_depth(const float* verts2d, const float* depth_per_vert, int count, int jacket_px, std::vector<std::pair<float,float>>& samples, std::vector<float>* depth_samples) {
    samples.clear();
    if (depth_samples) depth_samples->clear();
    if (!verts2d || count < 2) return;

    auto get = [&](int idx) {
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts2d[2*idx+0], verts2d[2*idx+1]);
    };
    auto get_depth = [&](int idx) {
        if (!depth_per_vert) return 0.0f;
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return depth_per_vert[idx];
    };

    auto catmull = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        float dz = 0.0f;
        if (depth_per_vert) {
            float z0 = get_depth(i-1);
            float z1 = get_depth(i+0);
            float z2 = get_depth(i+1);
            float z3 = get_depth(i+2);
            dz = 0.5f * ((2.0f * z1) + (-z0 + z2) * t + (2.0f*z0 - 5.0f*z1 + 4.0f*z2 - z3) * t2 + (-z0 + 3.0f*z1 - 3.0f*z2 + z3) * t3);
        }
        return std::tuple<float,float,float>(x,y,dz);
    };

    samples.reserve(static_cast<std::size_t>((count - 1) * 8));
    if (depth_samples) depth_samples->reserve(static_cast<std::size_t>((count - 1) * 8));
    for (int i = 0; i < count - 1; ++i) {
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        // sample spacing: ~0.6*jacket_px to avoid visible gaps while following spline
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        int s_start = (i == 0) ? 0 : 1; // avoid duplicate samples at segment boundaries
        for (int s = s_start; s <= n; ++s) {
            float t = float(s) / float(n);
            auto [sx, sy, sz] = catmull(i, t);
            samples.emplace_back(sx, sy);
            if (depth_samples) depth_samples->push_back(sz);
        }
    }
}

static void draw_rope_fiber_overlay(uint8_t* img, int w, int h, int pitch, const std::vector<std::pair<float,float>>& samples, const std::vector<float>& depth_samples, int radius, Color glow_col, float depth_fade, float gain, float min_z, float max_z) {
    if (!img || samples.size() < 2) return;
    float depth_span = std::max(1e-3f, max_z - min_z);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        auto &a = samples[i];
        auto &b = samples[i+1];
        float fade = 1.0f;
        if (depth_samples.size() == samples.size()) {
            float z_avg = 0.5f * (depth_samples[i] + depth_samples[i+1]);
            float norm = (z_avg - min_z) / depth_span;
            norm = std::clamp(norm, 0.0f, 1.0f);
            fade = 1.0f - depth_fade * norm;
        }
        float alpha_scale = std::max(0.0f, gain * fade);
        draw_segment_blend(img, w, h, pitch, a.first, a.second, b.first, b.second, radius, glow_col, alpha_scale);
    }
}

// forward declaration: color interpolation used by several rope/rope-glow helpers
static inline Color lerp_color(Color a, Color b, float t);

static float draw_rope_diffused_glow(
    uint8_t* img,
    int w,
    int h,
    int pitch,
    const std::vector<std::pair<float,float>>& samples,
    const std::vector<float>& depth_samples,
    bool start_is_front,
    float start_glow,
    float depth_fade,
    float gain,
    Color glow_col,
    const std::vector<Color>* jacket_colors,
    float min_z,
    float max_z,
    int radius) {
    if (samples.size() < 2) return 0.0f;
    float clamped_glow = std::max(0.0f, start_glow);
    if (clamped_glow <= 0.0f) return 0.0f;
    float depth_span = std::max(1e-3f, max_z - min_z);

    std::vector<float> prefix(samples.size(), 0.0f);
    for (size_t i = 1; i < samples.size(); ++i) {
        float dx = samples[i].first - samples[i-1].first;
        float dy = samples[i].second - samples[i-1].second;
        prefix[i] = prefix[i-1] + std::sqrt(dx*dx + dy*dy);
    }
    float total_len = prefix.back();
    if (total_len <= 1e-5f) return 0.0f;

    auto depth_scale_at = [&](size_t idx) {
        if (depth_samples.size() != samples.size()) return 1.0f;
        float norm = (depth_samples[idx] - min_z) / depth_span;
        norm = std::clamp(norm, 0.0f, 1.0f);
        return 1.0f - depth_fade * norm;
    };
    auto dist_from_start = [&](size_t idx) {
        float d = prefix[idx];
        return start_is_front ? d : (total_len - d);
    };

    // modest diffusion: higher decay keeps glow tighter along cable
    const float decay = 1.1f;
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float da = dist_from_start(i);
        float db = dist_from_start(i + 1);
        float dist_mid = 0.5f * (da + db);
        float along = (total_len > 1e-4f) ? (dist_mid / total_len) : 0.0f;
        float atten = std::exp(-decay * along);
        float depth_scale = 0.5f * (depth_scale_at(i) + depth_scale_at(i + 1));
        float alpha_scale = gain * clamped_glow * atten * depth_scale;
        if (img && alpha_scale > 0.0f) {
            Color seg_glow = glow_col;
            if (jacket_colors && !jacket_colors->empty()) {
                float u = (total_len > 1e-4f) ? ((prefix[i] + prefix[i + 1]) * 0.5f / total_len) : 0.0f;
                int seg_idx = static_cast<int>(std::floor(u * static_cast<float>(jacket_colors->size())));
                seg_idx = std::clamp(seg_idx, 0, static_cast<int>(jacket_colors->size()) - 1);
                Color jacket_col = (*jacket_colors)[static_cast<size_t>(seg_idx)];
                if (jacket_col.a > 0) {
                    float jacket_weight = std::clamp(static_cast<float>(jacket_col.a) / 255.0f, 0.0f, 1.0f) * 0.4f;
                    seg_glow = lerp_color(seg_glow, jacket_col, jacket_weight);
                    seg_glow.a = glow_col.a;
                }
            }
            draw_segment_kernel_glow(img, w, h, pitch, samples[i].first, samples[i].second, samples[i+1].first, samples[i+1].second, float(radius), seg_glow, alpha_scale);
        }
    }

    size_t dest_idx = start_is_front ? samples.size() - 1 : 0;
    float transmitted = clamped_glow * std::exp(-decay) * depth_scale_at(dest_idx);
    return transmitted;
}

// Draw a smooth blended rope/tube along given interleaved vertices using Catmull-Rom
// verts: float array [x0,y0, x1,y1, ...], count = number of vertices
static void draw_rope_curve_blend(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, Color core_col, int samples_per_segment) {
    if (!img || !verts || count < 2) return;
    if (samples_per_segment < 2) samples_per_segment = 2;

    auto get = [&](int idx) {
        // clamp
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts[2*idx+0], verts[2*idx+1]);
    };

    auto catmull = [&](int i, float t) {
        // control points p0..p3 for segment between i and i+1
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        // Catmull-Rom with tension 0.5
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        return std::pair<float,float>(x,y);
    };

    auto catmull_deriv = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float dx = 0.5f * ((-p0.first + p2.first) + 2.0f * (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t + 3.0f * (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t2);
        float dy = 0.5f * ((-p0.second + p2.second) + 2.0f * (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t + 3.0f * (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t2);
        return std::pair<float,float>(dx, dy);
    };

    // analytic Catmull-Rom derivative (tangent) for parametric cross-sections
    

    // Build dense samples along the spline then rasterize as thick blended segments
    std::vector<std::pair<float,float>> samples;
    std::vector<std::pair<float,float>> tangents;
    samples.reserve((count - 1) * 8);
    tangents.reserve((count - 1) * 8);
    for (int i = 0; i < count - 1; ++i) {
        // estimate chord length for this segment (p1-p2)
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        // sample spacing: ~0.6*jacket_px to avoid visible gaps while following spline
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        int s_start = (i == 0) ? 0 : 1; // avoid duplicate samples at segment boundaries
        for (int s = s_start; s <= n; ++s) {
            float t = float(s) / float(n);
            auto p = catmull(i, t);
            auto d = catmull_deriv(i, t);
            samples.emplace_back(p.first, p.second);
            tangents.emplace_back(d.first, d.second);
        }
    }

    if (samples.empty()) return;

    // endpoint caps removed to avoid double-painted ends

    // draw jacket (slimmer and translucent so background shows through)
    int eff_jacket = std::max(1, jacket_px - 1);
    // use neutral grey jacket as requested
    Color jacket_col = Color{200,200,200, static_cast<uint8_t>(13)};
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, eff_jacket, jacket_col, 1.0f);
    }

    // draw core (thinner, faint tinted core so cable looks clear)
    int core_r = std::max(1, jacket_px - jacket_border - 0);
    Color corec = core_col;
    // keep inner core at requested alpha so producer color remains visible
    corec.a = static_cast<uint8_t>(std::clamp<int>(corec.a, 0, 255));
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, core_r, corec, 1.0f);
    }
}

static void build_rope_samples_with_tangents(const float* verts, int count, int jacket_px, std::vector<std::pair<float,float>>& samples, std::vector<std::pair<float,float>>& tangents) {
    samples.clear();
    tangents.clear();
    if (!verts || count < 2) return;
    auto get = [&](int idx) {
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts[2*idx+0], verts[2*idx+1]);
    };
    auto catmull = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        return std::pair<float,float>(x,y);
    };
    auto catmull_deriv = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float dx = 0.5f * ((-p0.first + p2.first) + 2.0f * (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t + 3.0f * (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t2);
        float dy = 0.5f * ((-p0.second + p2.second) + 2.0f * (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t + 3.0f * (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t2);
        return std::pair<float,float>(dx, dy);
    };
    samples.reserve((count - 1) * 8);
    tangents.reserve((count - 1) * 8);
    for (int i = 0; i < count - 1; ++i) {
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        int s_start = (i == 0) ? 0 : 1;
        for (int s = s_start; s <= n; ++s) {
            float t = float(s) / float(n);
            auto p = catmull(i, t);
            auto d = catmull_deriv(i, t);
            samples.emplace_back(p.first, p.second);
            tangents.emplace_back(d.first, d.second);
        }
    }
}

static void draw_rope_jacket_overlay(uint8_t* img, int w, int h, int pitch,
                                     const std::vector<std::pair<float,float>>& samples,
                                     const std::vector<std::pair<float,float>>& tangents,
                                     int jacket_px,
                                     const std::vector<Color>& jacket_colors) {
    if (!img || samples.size() < 2 || samples.size() != tangents.size() || jacket_colors.empty()) return;
    int eff_jacket = std::max(1, jacket_px - 1);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float u = (samples.size() > 1) ? (static_cast<float>(i) + 0.5f) / static_cast<float>(samples.size() - 1) : 0.0f;
        int seg_idx = static_cast<int>(std::floor(u * static_cast<float>(jacket_colors.size())));
        seg_idx = std::clamp(seg_idx, 0, static_cast<int>(jacket_colors.size()) - 1);
        Color col = jacket_colors[static_cast<size_t>(seg_idx)];
        if (col.a == 0) continue;
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, eff_jacket, col, 1.0f);
    }
}

static void draw_rope_core_gradient(uint8_t* img, int w, int h, int pitch,
                                    const std::vector<std::pair<float,float>>& samples,
                                    const std::vector<std::pair<float,float>>& tangents,
                                    int jacket_px,
                                    int jacket_border,
                                    Color col_a,
                                    Color col_b,
                                    float intensity) {
    if (!img || samples.size() < 2 || samples.size() != tangents.size()) return;
    int core_r = std::max(1, jacket_px - jacket_border - 0);
    float base_alpha = 255.0f * std::clamp(intensity, 0.0f, 1.0f) * 0.18f;
    auto lerp_color = [&](const Color &a, const Color &b, float t) {
        float tt = std::clamp(t, 0.0f, 1.0f);
        Color out;
        out.r = static_cast<uint8_t>(std::lround(float(a.r) * (1.0f - tt) + float(b.r) * tt));
        out.g = static_cast<uint8_t>(std::lround(float(a.g) * (1.0f - tt) + float(b.g) * tt));
        out.b = static_cast<uint8_t>(std::lround(float(a.b) * (1.0f - tt) + float(b.b) * tt));
        out.a = 0;
        return out;
    };
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float u = (samples.size() > 1) ? static_cast<float>(i) / static_cast<float>(samples.size() - 1) : 0.0f;
        float alpha_mix = (1.0f - u) * float(col_a.a) + u * float(col_b.a);
        uint8_t alpha = static_cast<uint8_t>(std::lround(base_alpha * std::clamp(alpha_mix / 255.0f, 0.0f, 1.0f)));
        if (alpha == 0) continue;
        Color core_col = lerp_color(col_a, col_b, u);
        core_col.a = alpha;
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, core_r, core_col, 1.0f);
    }
}

// Convert HSV (h in 0..1, s 0..1, v 0..1) to Color (alpha=255)
static inline Color hsv_to_color(float h, float s, float v, uint8_t a=255) {
    h = h - std::floor(h);
    float hh = h * 6.0f;
    int i = static_cast<int>(std::floor(hh));
    float f = hh - float(i);
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    float r=0,g=0,b=0;
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }
    return Color{ static_cast<uint8_t>(std::lround(r * 255.0f)), static_cast<uint8_t>(std::lround(g * 255.0f)), static_cast<uint8_t>(std::lround(b * 255.0f)), a };
}

// Convert RGB Color to hue in [0..1]
static inline float rgb_to_hue(const Color &c) {
    float r = c.r / 255.0f;
    float g = c.g / 255.0f;
    float b = c.b / 255.0f;
    float mx = std::max(r, std::max(g, b));
    float mn = std::min(r, std::min(g, b));
    float d = mx - mn;
    if (d <= 1e-6f) return 0.0f;
    float h = 0.0f;
    if (mx == r) h = (g - b) / d + (g < b ? 6.0f : 0.0f);
    else if (mx == g) h = (b - r) / d + 2.0f;
    else h = (r - g) / d + 4.0f;
    h /= 6.0f;
    return h - std::floor(h);
}

static inline Color lerp_color(Color a, Color b, float t) {
    float tt = std::clamp(t, 0.0f, 1.0f);
    auto mix = [&](uint8_t ca, uint8_t cb) -> uint8_t {
        return static_cast<uint8_t>(std::lround(float(ca) * (1.0f - tt) + float(cb) * tt));
    };
    Color out;
    out.r = mix(a.r, b.r);
    out.g = mix(a.g, b.g);
    out.b = mix(a.b, b.b);
    out.a = mix(a.a, b.a);
    return out;
}

static inline Color tint_color_hue(Color base, float hue_shift, float tint_strength) {
    float hue = rgb_to_hue(base);
    Color shifted = hsv_to_color(hue + hue_shift, 1.0f, 1.0f, base.a);
    return lerp_color(base, shifted, tint_strength);
}

// Colored variant: `hues` is an optional array of per-vertex hue values in [0..1]. If null, falls back to neutral drawing.
static void draw_rope_curve_blend_colored(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, const float* hues, int hue_count, int samples_per_segment, float hue_intensity) {
    if (!hues || hue_count <= 0) {
        // fallback
        Color neutral{200,200,200,200};
        draw_rope_curve_blend(img, w, h, pitch, verts, count, jacket_px, jacket_border, neutral, samples_per_segment);
        return;
    }
    if (!img || !verts || count < 2) return;

    // Build samples along spline (same as non-colored variant)
    auto get = [&](int idx) {
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts[2*idx+0], verts[2*idx+1]);
    };
    auto catmull = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        return std::pair<float,float>(x,y);
    };

    // analytic Catmull-Rom derivative (tension 0.5)
    auto catmull_deriv = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float dx = 0.5f * ((-p0.first + p2.first) + 2.0f * (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t + 3.0f * (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t2);
        float dy = 0.5f * ((-p0.second + p2.second) + 2.0f * (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t + 3.0f * (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t2);
        return std::pair<float,float>(dx, dy);
    };

    std::vector<std::pair<float,float>> samples;
    std::vector<std::pair<float,float>> tangents;
    samples.reserve((count - 1) * 8);
    tangents.reserve((count - 1) * 8);
    for (int i = 0; i < count - 1; ++i) {
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        // build hue samples spacing: ~0.6*jacket_px to align hue samples with visual samples
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        for (int s = 0; s <= n; ++s) {
            float t = float(s) / float(n);
            auto p = catmull(i, t);
            auto d = catmull_deriv(i, t);
            samples.emplace_back(p.first, p.second);
            tangents.emplace_back(d.first, d.second);
        }
    }
    if (samples.empty()) return;

    // build hue samples aligned with `samples` by interpolating provided `hues` across the rope
    std::vector<float> hue_samples;
    hue_samples.reserve(samples.size());
    for (size_t si = 0; si < samples.size(); ++si) {
        float u = float(si) / float(std::max<size_t>(1, samples.size() - 1));
        if (hue_count <= 1) {
            hue_samples.push_back(hues[0]);
            continue;
        }
        float v = u * float(hue_count - 1);
        int idx = static_cast<int>(std::floor(v));
        idx = std::clamp(idx, 0, hue_count - 2);
        float ft = v - float(idx);
        float h0 = hues[idx];
        float h1 = hues[idx + 1];
        hue_samples.push_back(h0 * (1.0f - ft) + h1 * ft);
    }

    // helper to draw segments (reuse draw_segment_blend lambda from earlier by reimplementing minimal inline)
    auto draw_segment_blend_local = [&](float x1, float y1, float x2, float y2, int radius, Color col) {
        float dx = x2 - x1;
        float dy = y2 - y1;
        float len2 = dx*dx + dy*dy;
        float rplus = float(radius) + 1.0f;
        int minx = static_cast<int>(std::floor(std::min(x1,x2) - rplus));
        int maxx = static_cast<int>(std::ceil(std::max(x1,x2) + rplus));
        int miny = static_cast<int>(std::floor(std::min(y1,y2) - rplus));
        int maxy = static_cast<int>(std::ceil(std::max(y1,y2) + rplus));
        minx = std::max(minx, 0);
        miny = std::max(miny, 0);
        maxx = std::min(maxx, w - 1);
        maxy = std::min(maxy, h - 1);
        for (int py = miny; py <= maxy; ++py) {
            for (int px = minx; px <= maxx; ++px) {
                float cx = px + 0.5f;
                float cy = py + 0.5f;
                float t = 0.0f;
                if (len2 > 1e-6f) {
                    t = ((cx - x1) * dx + (cy - y1) * dy) / len2;
                    if (t < 0.0f) t = 0.0f;
                    else if (t > 1.0f) t = 1.0f;
                }
                float closestX = x1 + dx * t;
                float closestY = y1 + dy * t;
                float ddx = cx - closestX;
                float ddy = cy - closestY;
                float dist = std::sqrt(ddx*ddx + ddy*ddy);
                if (dist <= rplus) {
                    float coverage = 1.0f - (dist / rplus);
                    uint8_t a = static_cast<uint8_t>(std::lround(float(col.a) * coverage));
                    if (a == 0) continue;
                    Color cc = col;
                    cc.a = a;
                    uint8_t* dstp = img + py * pitch + px * 4;
                    blend_pixel(dstp, cc.r, cc.g, cc.b, cc.a);
                }
            }
        }
    };

    // Draw jacket first (grey base tinted by LED hue via tint_strength)
    int eff_jacket = std::max(1, jacket_px - 1);
    uint8_t jacket_alpha = static_cast<uint8_t>(13);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float hue_mid = 0.5f * (hue_samples[i] + hue_samples[i+1]);
        Color jacket_hue = hsv_to_color(hue_mid, 1.0f, 1.0f, jacket_alpha);
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, eff_jacket, jacket_hue, 1.0f, hue_intensity);
    }

    // Draw colored core using hue_samples but keep core faint so cable looks clear
    int core_r = std::max(1, jacket_px - jacket_border - 0);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float hue_mid = 0.5f * (hue_samples[i] + hue_samples[i+1]);
        // keep colored core very faint so the rope interior remains mostly clear (≈5%)
        uint8_t alpha = static_cast<uint8_t>(std::lround(255.0f * std::clamp(hue_intensity, 0.0f, 1.0f) * 0.18f));
        Color hc = hsv_to_color(hue_mid, 1.0f, 1.0f, alpha);
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, core_r, hc, 1.0f, hue_intensity);
    }
    // note: endpoint rims already drawn by caller if desired
}


GP_TableContext* gp_table_create(const GP_TableStyle* style) {
    std::unique_ptr<GP_TableContext> ctx(new GP_TableContext());
    if (style) {
        ctx->style_raw = *style;
    } else {
        ctx->style_raw = make_default_style();
    }
    ctx->st = load_style(&ctx->style_raw);
    ctx->cols.resize(0);
    ctx->rows.resize(0);
    recompute_geom(ctx.get());
    // create rope simulator with reasonable capacities (owned by default)
    int max_ropes = 1024;
    int max_segs = std::max(4, ctx->st.cable_segments);
    ctx->rope_sim = rope_sim_create(max_ropes, max_segs);
    ctx->rope_sim_owned = 1;
    ctx->rope_id_to_sim_idx.clear();
    return ctx.release();
}

void gp_table_destroy(GP_TableContext* ctx) {
    if (!ctx) return;
    if (ctx->rope_sim && ctx->rope_sim_owned) {
        rope_sim_destroy(ctx->rope_sim);
        ctx->rope_sim = nullptr;
    }
    delete ctx;
}

int32_t gp_table_set_style(GP_TableContext* ctx, const GP_TableStyle* style) {
    if (!ctx) return 0;
    if (style) {
        ctx->style_raw = *style;
    } else {
        ctx->style_raw = make_default_style();
    }
    ctx->st = load_style(&ctx->style_raw);
    recompute_geom(ctx);
    for (auto &fifo : ctx->edge_fifos) {
        fifo.set_friction_regions(ctx->st.cable_fifo_friction_regions);
    }
    return 1;
}

int32_t gp_table_set_columns(GP_TableContext* ctx, const GP_TableColumn* cols, int32_t col_count) {
    if (!ctx) return 0;
    if (col_count < 0 || col_count > 8) return 0;
    if (col_count > 0 && !cols) return 0;
    if (col_count == 0) {
        ctx->cols.clear();
    } else {
        ctx->cols.assign(cols, cols + col_count);
    }
    recompute_geom(ctx);
    return 1;
}

int32_t gp_table_set_rows(GP_TableContext* ctx, const GP_TableRow* rows, int32_t row_count) {
    if (!ctx) return 0;
    if (row_count < 0) return 0;
    if (row_count > 0 && !rows) return 0;
    if (row_count == 0) {
        ctx->rows.clear();
    } else {
        ctx->rows.assign(rows, rows + row_count);
    }
    recompute_geom(ctx);
    return 1;
}

int32_t gp_table_apply_object_def(GP_TableContext* ctx, const GP_TableObjectDef* def) {
    if (!ctx || !def) return 0;
    if (!gp_table_set_columns(ctx, def->cols, def->col_count)) return 0;
    if (!gp_table_set_rows(ctx, def->rows, def->row_count)) return 0;
    if (!gp_table_set_actions(ctx, def->actions, def->action_count)) return 0;
    return 1;
}

int32_t gp_table_get_geom(const GP_TableContext* ctx, GP_TableGeom* out_geom) {
    if (!ctx || !out_geom) return 0;
    *out_geom = ctx->geom;
    return 1;
}

static bool action_matches(const GP_TableAction& action, const GP_TableHitBox& hb) {
    if (action.row_idx != GP_TABLE_ACTION_ANY && action.row_idx != hb.row_idx) return false;
    if (action.col_idx != GP_TABLE_ACTION_ANY && action.col_idx != hb.col_idx) return false;
    if (action.part != GP_TABLE_ACTION_ANY && action.part != hb.part) return false;
    if (action.aux0 != GP_TABLE_ACTION_ANY && action.aux0 != hb.aux0) return false;
    return true;
}

static bool dispatch_action_list(GP_TableContext* ctx, const GP_TableHitBox& hb, const GP_TableAction* actions, int count) {
    if (!ctx || !ctx->action_callback || !actions || count <= 0) return false;
    bool matched = false;
    for (int i = 0; i < count; ++i) {
        if (action_matches(actions[i], hb)) {
            ctx->action_callback(ctx->action_user, actions[i].action_id, &hb);
            matched = true;
        }
    }
    return matched;
}

static void dispatch_actions(GP_TableContext* ctx, const GP_TableHitBox& hb) {
    if (!ctx || !ctx->action_callback) return;
    if (!ctx->actions.empty()) {
        dispatch_action_list(ctx, hb, ctx->actions.data(), static_cast<int>(ctx->actions.size()));
        return;
    }
    static const GP_TableAction kDefaultActions[] = {
        { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_EXPAND, GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_EXPAND_TOGGLE },
        { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_SCROLL_UP, GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_SCROLL_UP },
        { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_SCROLL_DOWN, GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_SCROLL_DOWN },
        { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_LED, GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_LED_TOGGLE },
        { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_LED_ARG, GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_LED_TOGGLE },
        { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_LED_TABLE, GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_LED_TOGGLE }
    };
    dispatch_action_list(ctx, hb, kDefaultActions, static_cast<int>(sizeof(kDefaultActions) / sizeof(kDefaultActions[0])));
}

int32_t gp_table_dispatch_hit(GP_TableContext* ctx, const GP_TableHitBox* hit) {
    if (!ctx || !hit) return 0;
    if (ctx->actions.empty()) return 0;
    return dispatch_action_list(ctx, *hit, ctx->actions.data(), static_cast<int>(ctx->actions.size())) ? 1 : 0;
}

int32_t gp_table_on_click(GP_TableContext* ctx, int32_t x, int32_t y, GP_TableHitBox* out_hit) {
    if (!ctx) return 0;
    // Render into a temporary buffer to collect hitboxes
    GP_TableGeom geom{};
    gp_table_get_geom(ctx, &geom);
    const int w = geom.width_px;
    const int h = geom.height_px;
    if (w <= 0 || h <= 0) return 0;
    std::vector<uint8_t> tmp;
    tmp.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
    const int hitcap = 4096;
    std::vector<GP_TableHitBox> hits(hitcap);
    int written = 0;
    int ok = gp_table_render_rgba_with_state(ctx, nullptr, tmp.data(), static_cast<int32_t>(tmp.size()), &geom, hits.data(), hitcap, &written);
    if (!ok) return 0;
    // find first hit that contains (x,y)
    for (int i = 0; i < written; ++i) {
        const GP_TableHitBox &hb = hits[i];
        if (x >= hb.x0 && x < hb.x1 && y >= hb.y0 && y < hb.y1) {
            // process default actions
            if (out_hit) *out_hit = hb;
            dispatch_actions(ctx, hb);
            // expand toggle
            if (hb.part == GP_TABLE_HIT_EXPAND && hb.row_idx >= 0 && hb.row_idx < static_cast<int>(ctx->rows.size())) {
                ctx->rows[hb.row_idx].expanded = ctx->rows[hb.row_idx].expanded ? 0 : 1;
                recompute_geom(ctx);
                return 1;
            }
            // scroll up/down
            if (hb.part == GP_TABLE_HIT_SCROLL_UP) {
                ctx->scroll_row_offset = std::max(0, ctx->scroll_row_offset - 1);
                return 1;
            }
            if (hb.part == GP_TABLE_HIT_SCROLL_DOWN) {
                ctx->scroll_row_offset = std::min<int>(std::max(0, static_cast<int>(ctx->rows.size()) - 1), ctx->scroll_row_offset + 1);
                return 1;
            }
            // LED click: toggle selection (separate from on/off state)
            if ((hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG || hb.part == GP_TABLE_HIT_LED_TABLE) && hb.row_idx >= 0 && hb.col_idx >= 0) {
                int r_idx = hb.row_idx;
                int c_idx = hb.col_idx;
                int led = hb.aux0;
                if (led >= 0) {
                    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(r_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(c_idx)) << 16) | static_cast<uint64_t>(static_cast<uint32_t>(led));
                    auto it = ctx->selected_leds.find(key);
                    if (it == ctx->selected_leds.end()) ctx->selected_leds.insert(key);
                    else ctx->selected_leds.erase(it);
                            // If exactly one other LED was selected prior to this click, form an edge.
                            if (ctx->selected_leds.size() == 2) {
                                // grab the two keys
                                auto it2 = ctx->selected_leds.begin();
                                uint64_t k0 = *it2; ++it2; uint64_t k1 = *it2;
                                // add edge if allowed by node-group rules (directional: k0 -> k1)
                                int allowed = gp_table_node_group_is_edge_allowed(ctx, k0, k1);
                                if (allowed == 1) {
                                    // use API helper so relax arrays are kept in sync
                                    gp_table_add_edge(ctx, k0, k1);
                                }
                                // clear selections after attempting to form edge
                                ctx->selected_leds.clear();
                            }
                            return 1;
                }
            }
            // Other parts: no default action, but return hit
            return 1;
        }
    }
    return 0;
}

int32_t gp_table_set_scroll_fraction(GP_TableContext* ctx, float frac) {
    if (!ctx) return 0;
    ctx->scroll_frac = std::clamp(frac, 0.0f, 1.0f);
    // compute row offset based on fraction and available rows
    int visible_rows = std::max(1, ctx->geom.height_px / ctx->st.row_h);
    int total = static_cast<int>(ctx->rows.size());
    int max_off = std::max(0, total - visible_rows);
    ctx->scroll_row_offset = static_cast<int>(std::lround(ctx->scroll_frac * float(max_off)));
    return 1;
}

int32_t gp_table_set_scroll_fraction_xy(GP_TableContext* ctx, float frac_x, float frac_y) {
    if (!ctx) return 0;
    ctx->scroll_frac_x = std::clamp(frac_x, 0.0f, 1.0f);
    return gp_table_set_scroll_fraction(ctx, frac_y);
}

// Edge list helpers
int32_t gp_table_add_edge(GP_TableContext* ctx, unsigned long long a, unsigned long long b) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    ctx->edges.emplace_back(a, b);
    // create and record a persistent unique id for this edge
    uint64_t uid = 0ull;
    uid = gp_canvas_generate_id(gp_canvas_get_singleton(), 0ull);
    ctx->edge_ids.push_back(uid);
    // ensure relax arrays stay in sync
    // start unrelaxed so the cable animates into place
    ctx->relax_value.push_back(0.0f);
    ctx->relax_vel.push_back(0.0f);
    // companion tensor FIFO for this edge
    EdgeTensorFifo fifo;
    fifo.configure_default();
    ctx->edge_fifos.push_back(std::move(fifo));
    // reset prospective state when a real edge is added
    ctx->prospective_initialized = false;
    // create a rope entry in the simulator (if available)
    if (ctx->rope_sim) {
        // compute approximate endpoints in table-local coords
        int ax = 0, ay = 0, bx = 0, by = 0;
        std::vector<int> row_y0;
        std::vector<int> row_h;
        compute_row_layout(ctx->rows.data(), static_cast<int>(ctx->rows.size()), ctx->st, ctx->geom.height_px, row_y0, row_h);
        auto compute_center_local = [&](uint64_t key, int &outx, int &outy) {
            outx = -1; outy = -1;
            uint32_t r_orig = static_cast<uint32_t>(key >> 32);
            uint32_t c_idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
            uint32_t led = static_cast<uint32_t>(key & 0xFFFFu);
            if (r_orig >= ctx->rows.size()) return;
            const GP_TableRow &row = ctx->rows[static_cast<size_t>(r_orig)];
            if (static_cast<int>(c_idx) < 0 || static_cast<int>(c_idx) >= row.cell_count) return;
            int col_x0[8] = {0}; int col_w[8] = {0};
            compute_columns(ctx->cols.data(), static_cast<int>(ctx->cols.size()), ctx->st.w, ctx->st.name_w, col_x0, col_w);
            const GP_TableCell &cell = row.cells[static_cast<int>(c_idx)];
            int x0 = col_x0[static_cast<int>(c_idx)];
            int cw = col_w[static_cast<int>(c_idx)];
            int y0 = (r_orig < row_y0.size()) ? row_y0[static_cast<size_t>(r_orig)] : (static_cast<int>(r_orig) * ctx->st.row_h);
            int rh = (r_orig < row_h.size()) ? row_h[static_cast<size_t>(r_orig)] : ctx->st.row_h;
            int led_count = 9;
            if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
                int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                if (count == 0) count = (cell.flags & 0xFF);
                if (count == 0) count = 12;
                led_count = count;
            } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
                led_count = 8;
            }
            int eff_w = std::max(1, cw - 4);
            int radius = 4;
            int led_spacing = std::max(radius * 2 + 2, eff_w / std::max(1, led_count + 1));
            int cx0 = x0 + 2 + led_spacing;
            if (static_cast<int>(led) >= 0 && static_cast<int>(led) < led_count) {
                outx = cx0 + static_cast<int>(led) * led_spacing;
                const bool is_image_row = row_find_image_cell(row, nullptr);
                int band_h = is_image_row ? std::min(rh, row_image_header_h(ctx->st, row)) : rh;
                outy = y0 + band_h / 2;
            }
        };
        compute_center_local(a, ax, ay);
        compute_center_local(b, bx, by);
        int segs = std::max(4, ctx->st.cable_segments);
        float slack = 0.0f;
        float plug_z = -ctx->st.cable_plug_depth;
        int idx = rope_sim_add_rope3(ctx->rope_sim, static_cast<float>(ax), static_cast<float>(ay), plug_z, static_cast<float>(bx), static_cast<float>(by), plug_z, segs, slack);
        uint64_t rid = gp_canvas_generate_id(gp_canvas_get_singleton(), 0ull);
        ctx->rope_ids.push_back(rid);
        printf("gp_table_add_edge: ctx=%p rope_ids_count=%zu after push id=%llu\n", (void*)ctx, ctx->rope_ids.size(), (unsigned long long)rid);
        ctx->rope_id_to_sim_idx[rid] = idx;
        printf("gp_table_add_edge: ctx=%p added rope idx=%d id=%llu\n", (void*)ctx, idx, (unsigned long long)rid);
        {
            GP_CanvasContext* cvs = gp_canvas_get_singleton();
            if (cvs) {
                uint64_t tmp = rid;
                gp_canvas_register_table_rope_ids_from_array(cvs, ctx, &tmp, 1);
            }
        }
    } else {
        uint64_t rid = gp_canvas_generate_id(gp_canvas_get_singleton(), 0ull);
        ctx->rope_ids.push_back(rid);
        printf("gp_table_add_edge: ctx=%p queued edge id=%llu (no sim)\n", (void*)ctx, (unsigned long long)rid);
        {
            GP_CanvasContext* cvs = gp_canvas_get_singleton();
            if (cvs) {
                uint64_t tmp = rid;
                gp_canvas_register_table_rope_ids_from_array(cvs, ctx, &tmp, 1);
            }
        }
    }
    sync_edge_tensor_for_idx(ctx, ctx->edges.size() - 1);
    int32_t edge_idx = static_cast<int32_t>(ctx->edges.size() - 1);
    return 1;
}

int32_t gp_table_clear_edges(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ctx->edges.clear();
    ctx->rings.clear();
    ctx->edge_ids.clear();
    ctx->next_edge_id = 1;
    ctx->relax_value.clear();
    ctx->relax_vel.clear();
    ctx->edge_fifos.clear();
    ctx->edge_batch_metadata.clear();
    ctx->edge_subgroup_flags.clear();
    ctx->edge_subscriber_slots.clear();
    // reset rope simulator mapping and recreate sim to free resources
    ctx->rope_id_to_sim_idx.clear();
    if (!ctx->rope_ids.empty()) printf("gp_table_clear_edges: ctx=%p clearing %zu rope_ids\n", (void*)ctx, ctx->rope_ids.size());
    ctx->rope_ids.clear();
    ctx->next_rope_id = 1;
    if (ctx->rope_sim && ctx->rope_sim_owned) {
        rope_sim_destroy(ctx->rope_sim);
        int max_ropes = 1024;
        int max_segs = std::max(4, ctx->st.cable_segments);
        ctx->rope_sim = rope_sim_create(max_ropes, max_segs);
    } else {
        // if rope_sim is external or null, leave it alone; indices already cleared
    }
    return 1;
}

extern "C" int32_t gp_table_register_ring_edge(GP_TableContext* ctx, int32_t ring_id, unsigned long long ring_key) {
    if (!ctx) return -1;
    GP_TableContext::RingEntry re;
    re.ring_id = ring_id;
    re.key = ring_key;
    re.fifo.configure_default();
    re.uid = gp_canvas_generate_id(gp_canvas_get_singleton(), 0ull);
    ctx->rings.push_back(std::move(re));
    return static_cast<int32_t>(ctx->rings.size() - 1);
}

extern "C" int32_t gp_table_unregister_ring_edge(GP_TableContext* ctx, int32_t ring_entry_idx) {
    if (!ctx) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    ctx->rings.erase(ctx->rings.begin() + ring_entry_idx);
    return 1;
}

extern "C" int32_t gp_table_get_ring_edge_count(const GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int32_t>(ctx->rings.size());
}

extern "C" int32_t gp_table_get_ring_edge(const GP_TableContext* ctx, int32_t idx, int32_t* out_ring_id, unsigned long long* out_key) {
    if (!ctx || !out_ring_id || !out_key) return 0;
    if (idx < 0 || idx >= static_cast<int>(ctx->rings.size())) return 0;
    const auto &re = ctx->rings[static_cast<size_t>(idx)];
    *out_ring_id = re.ring_id;
    *out_key = re.key;
    return 1;
}

extern "C" int32_t gp_table_ring_set_subgroup_flags(GP_TableContext* ctx, int32_t ring_entry_idx, uint32_t flags) {
    if (!ctx) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    ctx->rings[static_cast<size_t>(ring_entry_idx)].subgroup_flags = flags;
    return 1;
}

extern "C" int32_t gp_table_ring_get_subgroup_flags(GP_TableContext* ctx, int32_t ring_entry_idx, uint32_t* out_flags) {
    if (!ctx || !out_flags) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    *out_flags = ctx->rings[static_cast<size_t>(ring_entry_idx)].subgroup_flags;
    return 1;
}

extern "C" int32_t gp_table_ring_set_tensor_spec(GP_TableContext* ctx, int32_t ring_entry_idx, const GP_TableEdgeTensorSpec* spec) {
    if (!ctx || !spec) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    auto &re = ctx->rings[static_cast<size_t>(ring_entry_idx)];
    // Mirror gp_table_edge_set_tensor_spec behavior: disallow reconfigure after writes
    if (re.fifo.impl && re.fifo.impl->write_seq.load(std::memory_order_relaxed) != 0) return 0;
    std::vector<int32_t> dims;
    int dc = std::max(0, std::min(8, spec->dim_count));
    dims.reserve(static_cast<size_t>(dc));
    for (int i = 0; i < dc; ++i) {
        int32_t d = spec->dims[i];
        if (d < 1) d = 1;
        dims.push_back(d);
    }
    size_t slots = spec->slots > 0 ? static_cast<size_t>(spec->slots) : size_t(1);
    size_t topk = spec->top_k > 0 ? static_cast<size_t>(spec->top_k) : size_t(0);
    re.fifo.configure(dims, slots, topk);
    re.batch_metadata = GP_TableEdgeBatchMetadata();
    return 1;
}

extern "C" int32_t gp_table_ring_get_tensor_spec(GP_TableContext* ctx, int32_t ring_entry_idx, GP_TableEdgeTensorSpec* out_spec) {
    if (!ctx || !out_spec) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    *out_spec = ctx->rings[static_cast<size_t>(ring_entry_idx)].fifo.to_spec();
    return 1;
}

extern "C" int32_t gp_table_ring_subscribe(GP_TableContext* ctx, int32_t ring_entry_idx, unsigned long long subscriber_key) {
    if (!ctx) return 0;
    return gp_table_ring_subscribe_ex(ctx, ring_entry_idx, subscriber_key, /*start_at_head=*/1);
}

extern "C" int32_t gp_table_ring_subscribe_ex(GP_TableContext* ctx, int32_t ring_entry_idx, unsigned long long subscriber_key, int32_t start_at_head) {
    if (!ctx) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    auto &re = ctx->rings[static_cast<size_t>(ring_entry_idx)];
    bool ok = re.fifo.subscribe(subscriber_key, start_at_head != 0);
    if (!ok) return 0;
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        if (static_cast<size_t>(ring_entry_idx) >= ctx->ring_subscriber_slots.size()) ctx->ring_subscriber_slots.resize(ctx->rings.size());
        auto &map = ctx->ring_subscriber_slots[static_cast<size_t>(ring_entry_idx)];
        if (map.find(subscriber_key) == map.end()) {
            uint64_t rid = re.uid;
            int slot = tm->register_reader_for_edge(rid);
            if (slot > 0) map[subscriber_key] = slot;
            auto *r = re.fifo.find_reader(subscriber_key);
            if (r && slot > 0) {
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return 1;
}

extern "C" int32_t gp_table_ring_unsubscribe(GP_TableContext* ctx, int32_t ring_entry_idx, unsigned long long subscriber_key) {
    if (!ctx) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    auto &re = ctx->rings[static_cast<size_t>(ring_entry_idx)];
    re.fifo.unsubscribe(subscriber_key);
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        if (static_cast<size_t>(ring_entry_idx) < ctx->ring_subscriber_slots.size()) {
            auto &m = ctx->ring_subscriber_slots[static_cast<size_t>(ring_entry_idx)];
            auto it = m.find(subscriber_key);
            if (it != m.end()) {
                int slot = it->second;
                if (slot > 0) tm->unregister_reader_slot(slot);
                m.erase(it);
            }
        }
    }
    return 1;
}

extern "C" int32_t gp_table_ring_publish(GP_TableContext* ctx, int32_t ring_entry_idx, unsigned long long writer_key, const float* sample, int32_t sample_len, int32_t* out_dropped) {
    if (out_dropped) *out_dropped = 0;
    if (!ctx || !sample || sample_len < 0) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    auto &re = ctx->rings[static_cast<size_t>(ring_entry_idx)];
    bool dropped = false;
    uint64_t rid = re.uid;
    uint32_t flags = re.subgroup_flags;
    bool ok = false;
    if (flags_imply_byref(flags)) {
        size_t sample_bytes = static_cast<size_t>(sample_len) * sizeof(float);
        size_t alloc_sz = sizeof(BoxedSample) + sample_bytes;
        uint8_t* buf = static_cast<uint8_t*>(std::malloc(alloc_sz));
        if (!buf) { if (out_dropped) *out_dropped = 1; return 0; }
        BoxedSample* box = reinterpret_cast<BoxedSample*>(buf);
        box->magic = BOXED_SAMPLE_MAGIC;
        box->sample_len = sample_len;
        uint8_t* payload = buf + sizeof(BoxedSample);
        std::memcpy(payload, sample, sample_bytes);
        ok = re.fifo.push_ptr(rid, writer_key, static_cast<void*>(box), &dropped);
    } else {
        ok = re.fifo.push(rid, writer_key, sample, static_cast<size_t>(sample_len), &dropped);
    }
    if (out_dropped && dropped) *out_dropped = 1;
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        if (static_cast<size_t>(ring_entry_idx) < ctx->ring_subscriber_slots.size()) {
            auto &m = ctx->ring_subscriber_slots[static_cast<size_t>(ring_entry_idx)];
            std::vector<std::pair<uint64_t,int>> subs; subs.reserve(m.size());
            for (const auto &kv : m) subs.emplace_back(kv.first, kv.second);
            for (const auto &kv : subs) {
                uint64_t subscriber_key = kv.first; int slot = kv.second;
                if (slot <= 0) continue;
                auto *r = re.fifo.find_reader(subscriber_key);
                if (!r) continue;
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return ok ? 1 : 0;
}

int32_t gp_table_get_edge_count(const GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int32_t>(ctx->edges.size());
}

int32_t gp_table_get_edge(const GP_TableContext* ctx, int32_t idx, unsigned long long* out_a, unsigned long long* out_b) {
    if (!ctx || !out_a || !out_b) return 0;
    if (idx < 0 || idx >= static_cast<int32_t>(ctx->edges.size())) return 0;
    *out_a = ctx->edges[static_cast<size_t>(idx)].first;
    *out_b = ctx->edges[static_cast<size_t>(idx)].second;
    return 1;
}

int32_t gp_table_edge_set_tensor_spec(GP_TableContext* ctx, int32_t edge_idx, const GP_TableEdgeTensorSpec* spec) {
    if (!ctx || !spec) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    // Spec immutability: disallow reconfigure after any writes have occurred.
    // (Caller may reconfigure only while the edge is quiescent.)
    {
        auto& fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
        if (fifo.impl && fifo.impl->write_seq.load(std::memory_order_relaxed) != 0) {
            return 0;
        }
    }
    std::vector<int32_t> dims;
    int dc = std::max(0, std::min(8, spec->dim_count));
    dims.reserve(static_cast<size_t>(dc));
    for (int i = 0; i < dc; ++i) {
        int32_t d = spec->dims[i];
        if (d < 1) d = 1;
        dims.push_back(d);
    }
    size_t slots = spec->slots > 0 ? static_cast<size_t>(spec->slots) : size_t(1);
    size_t topk = spec->top_k > 0 ? static_cast<size_t>(spec->top_k) : size_t(0);
    ctx->edge_fifos[static_cast<size_t>(edge_idx)].configure(dims, slots, topk);
    sync_edge_tensor_for_idx(ctx, static_cast<size_t>(edge_idx));
    // If a ThreadManager is present, update registered reader slots with
    // the freshly-initialized sequence (usually zero) so manager state
    // remains consistent after reconfigure.
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        auto &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
        for (const auto &kv : m) {
            uint64_t subscriber_key = kv.first;
            int slot = kv.second;
            if (slot <= 0) continue;
            auto *r = fifo.find_reader(subscriber_key);
            if (!r) continue;
            uint64_t seq = r->seq.load(std::memory_order_relaxed);
            tm->update_reader_seq(slot, seq);
        }
    }
    return 1;
}

int32_t gp_table_edge_get_tensor_spec(GP_TableContext* ctx, int32_t edge_idx, GP_TableEdgeTensorSpec* out_spec) {
    if (!ctx || !out_spec) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    *out_spec = ctx->edge_fifos[static_cast<size_t>(edge_idx)].to_spec();
    return 1;
}

int32_t gp_table_edge_subscribe(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key) {
    if (!ctx) return 0;
    return gp_table_edge_subscribe_ex(ctx, edge_idx, subscriber_key, /*start_at_head=*/1);
}

int32_t gp_table_edge_subscribe_ex(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, int32_t start_at_head) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    bool ok = ctx->edge_fifos[static_cast<size_t>(edge_idx)].subscribe(subscriber_key, start_at_head != 0);
    if (!ok) return 0;
    // Register reader slot with ThreadManager global (if available).
    // Avoid re-registering if this subscriber already has a slot mapping.
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &map = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        if (map.find(subscriber_key) == map.end()) {
            uint64_t edge_id = ctx->edge_ids[static_cast<size_t>(edge_idx)];
            int slot = tm->register_reader_for_edge(edge_id);
            if (slot > 0) map[subscriber_key] = slot;
            // Notify manager of the starting sequence for this reader (if available).
            auto *r = ctx->edge_fifos[static_cast<size_t>(edge_idx)].find_reader(subscriber_key);
            if (r && slot > 0) {
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return 1;
}

int32_t gp_table_edge_unsubscribe(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    ctx->edge_fifos[static_cast<size_t>(edge_idx)].unsubscribe(subscriber_key);
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        auto it = m.find(subscriber_key);
        if (it != m.end()) {
            int slot = it->second;
            if (slot > 0) tm->unregister_reader_slot(slot);
            m.erase(it);
        }
    }
    return 1;
}

int32_t gp_table_edge_publish(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const float* sample, int32_t sample_len, int32_t* out_dropped) {
    if (out_dropped) *out_dropped = 0;
    if (!ctx || !sample || sample_len < 0) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    bool dropped = false;
    uint64_t edge_id = ctx->edge_ids[static_cast<size_t>(edge_idx)];

    // Inspect edge subgroup flags to determine publish behavior. We use a
    // simple switch so future policies can be tacked on easily.
    uint32_t flags = 0;
    if (static_cast<size_t>(edge_idx) < ctx->edge_subgroup_flags.size()) flags = ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)];
    bool ok = false;
    if (flags_imply_byref(flags)) {
        // BYREF: box the float sample and publish its pointer instead so
        // consumers receive by-reference payloads. Box format: [BoxedSample][float data]
        size_t sample_bytes = static_cast<size_t>(sample_len) * sizeof(float);
        size_t alloc_sz = sizeof(BoxedSample) + sample_bytes;
        uint8_t* buf = static_cast<uint8_t*>(std::malloc(alloc_sz));
        if (!buf) { if (out_dropped) *out_dropped = 1; return 0; }
        BoxedSample* box = reinterpret_cast<BoxedSample*>(buf);
        box->magic = BOXED_SAMPLE_MAGIC;
        box->sample_len = sample_len;
        uint8_t* payload = buf + sizeof(BoxedSample);
        std::memcpy(payload, sample, sample_bytes);
        ok = fifo.push_ptr(edge_id, writer_key, static_cast<void*>(box), &dropped);
    } else {
        // Normal float path
        ok = fifo.push(edge_id, writer_key, sample, static_cast<size_t>(sample_len), &dropped);
    }
    if (out_dropped && dropped) *out_dropped = 1;
    // After a publish, the FIFO implementation may have advanced reader sequences
    // (e.g., during top-k trimming). Ensure the ThreadManager has up-to-date
    // per-slot sequences for all registered subscribers on this edge.
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        // Snapshot subscriber map to avoid iterator invalidation if
        // concurrent subscribe/unsubscribe mutates the map.
        std::vector<std::pair<uint64_t,int>> subs;
        subs.reserve(m.size());
        for (const auto &kv : m) subs.emplace_back(kv.first, kv.second);
        for (const auto &kv : subs) {
            uint64_t subscriber_key = kv.first;
            int slot = kv.second;
            if (slot <= 0) continue;
            auto *r = fifo.find_reader(subscriber_key);
            if (!r) continue;
            uint64_t seq = r->seq.load(std::memory_order_relaxed);
            tm->update_reader_seq(slot, seq);
        }
    }
    return ok ? 1 : 0;
}

int32_t gp_table_edge_publish_blocking(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const float* sample, int32_t sample_len, int32_t* out_dropped, int32_t timeout_ms) {
    if (out_dropped) *out_dropped = 0;
    if (!ctx || !sample || sample_len < 0) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    bool dropped = false;
    uint64_t edge_id = ctx->edge_ids[static_cast<size_t>(edge_idx)];
    bool ok = fifo.push_blocking(edge_id, writer_key, sample, static_cast<size_t>(sample_len), &dropped, timeout_ms);
    if (out_dropped && dropped) *out_dropped = 1;
    if (!ok) return 0;
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        std::vector<std::pair<uint64_t,int>> subs;
        subs.reserve(m.size());
        for (const auto &kv : m) subs.emplace_back(kv.first, kv.second);
        for (const auto &kv : subs) {
            uint64_t subscriber_key = kv.first;
            int slot = kv.second;
            if (slot <= 0) continue;
            auto *r = fifo.find_reader(subscriber_key);
            if (!r) continue;
            uint64_t seq = r->seq.load(std::memory_order_relaxed);
            tm->update_reader_seq(slot, seq);
        }
    }
    return 1;
}

// Pointer-oriented publish: publish an opaque pointer into the edge FIFO.
int32_t gp_table_edge_publish_ptr(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, void* ptr, int32_t* out_dropped) {
    if (out_dropped) *out_dropped = 0;
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    bool dropped = false;
    uint64_t edge_id = ctx->edge_ids[static_cast<size_t>(edge_idx)];
    bool ok = fifo.push_ptr(edge_id, writer_key, ptr, &dropped);
    if (out_dropped && dropped) *out_dropped = 1;
    // Sync ThreadManager reader sequences similar to float publish.
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        std::vector<std::pair<uint64_t,int>> subs;
        subs.reserve(m.size());
        for (const auto &kv : m) subs.emplace_back(kv.first, kv.second);
        for (const auto &kv : subs) {
            uint64_t subscriber_key = kv.first;
            int slot = kv.second;
            if (slot <= 0) continue;
            auto *r = fifo.find_reader(subscriber_key);
            if (!r) continue;
            uint64_t seq = r->seq.load(std::memory_order_relaxed);
            tm->update_reader_seq(slot, seq);
        }
    }
    return ok ? 1 : 0;
}

// Generic wrappers to provide a neutral edge API surface. These forward to
// the table-specific implementations so callers outside the table system can
// use a stable `gp_edge_*` API while we evolve internals.
int32_t gp_edge_publish(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const float* sample, int32_t sample_len, int32_t* out_dropped) {
    return gp_table_edge_publish(ctx, edge_idx, writer_key, sample, sample_len, out_dropped);
}
int32_t gp_edge_publish_blocking(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const float* sample, int32_t sample_len, int32_t* out_dropped, int32_t timeout_ms) {
    return gp_table_edge_publish_blocking(ctx, edge_idx, writer_key, sample, sample_len, out_dropped, timeout_ms);
}
int32_t gp_edge_publish_ptr(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, void* ptr, int32_t* out_dropped) {
    return gp_table_edge_publish_ptr(ctx, edge_idx, writer_key, ptr, out_dropped);
}
int32_t gp_edge_consume_ptr(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void** out_ptr) {
    return gp_table_edge_consume_ptr(ctx, edge_idx, subscriber_key, out_ptr);
}

int32_t gp_table_edge_consume_ptr(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void** out_ptr) {
    if (out_ptr) *out_ptr = nullptr;
    if (!ctx || !out_ptr) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    void* p = nullptr;
    bool ok = fifo.pop_ptr(subscriber_key, &p);
    if (!ok) return 0;
    if (out_ptr) *out_ptr = p;
    // Notify ThreadManager of read advancement
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        auto it = m.find(subscriber_key);
        if (it != m.end()) {
            int slot = it->second;
            auto *r = fifo.find_reader(subscriber_key);
            if (r && slot > 0) {
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return 1;
}

int32_t gp_table_edge_consume(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, float* out_sample, int32_t out_len, int32_t* out_written) {
    if (out_written) *out_written = 0;
    if (!ctx || !out_sample || out_len < 0) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    // If the edge is configured BYREF, the FIFO carries pointers to boxed
    // float samples. In that case, consume via pop_ptr and transparently
    // unbox into the caller-provided float buffer and free the boxed memory.
    size_t wrote = 0;
    uint32_t flags = 0;
    if (static_cast<size_t>(edge_idx) < ctx->edge_subgroup_flags.size()) flags = ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)];
    bool ok = false;
    if (flags_imply_byref(flags)) {
        // Peek first: if there's a pointer and it's a boxed sample, then
        // consume it; if it's a non-boxed pointer (e.g., EventPayload)
        // do not advance the reader here — let pointer-oriented APIs
        // (`gp_table_edge_consume_ptr`) handle it.
        void* ppeek = nullptr;
        if (fifo.peek_ptr(subscriber_key, &ppeek)) {
            if (!ppeek) return 0;
            BoxedSample* boxpeek = reinterpret_cast<BoxedSample*>(ppeek);
            if (boxpeek->magic == BOXED_SAMPLE_MAGIC) {
                // Now actually pop the pointer and unbox.
                void* p = nullptr;
                if (!fifo.pop_ptr(subscriber_key, &p)) return 0;
                if (!p) return 0;
                BoxedSample* box = reinterpret_cast<BoxedSample*>(p);
                int need = box->sample_len;
                if (out_len < need) {
                    std::free(box);
                    return 0;
                }
                float* payload = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(box) + sizeof(BoxedSample));
                std::memcpy(out_sample, payload, static_cast<size_t>(need) * sizeof(float));
                std::free(box);
                wrote = static_cast<size_t>(need);
                ok = true;
            } else {
                // A raw pointer was delivered; do not consume here.
                return 0;
            }
        } else {
            // No pointer available — try float path.
            ok = fifo.pop(subscriber_key, out_sample, static_cast<size_t>(out_len), wrote);
        }
    } else {
        ok = fifo.pop(subscriber_key, out_sample, static_cast<size_t>(out_len), wrote);
    }
    if (out_written) *out_written = static_cast<int32_t>(wrote);
    if (!ok) return 0;
    // Notify ThreadManager of reader advancement, if registered.
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        auto it = m.find(subscriber_key);
        if (it != m.end()) {
            int slot = it->second;
            auto *r = fifo.find_reader(subscriber_key);
            if (r && slot > 0) {
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return 1;
}

int32_t gp_table_edge_peek(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, float* out_sample, int32_t out_len, int32_t* out_written) {
    if (out_written) *out_written = 0;
    if (!ctx || !out_sample || out_len < 0) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    size_t wrote = 0;
    uint32_t flags = 0;
    if (static_cast<size_t>(edge_idx) < ctx->edge_subgroup_flags.size()) flags = ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)];
    bool ok = false;
    if (flags_imply_byref(flags)) {
        void* p = nullptr;
        if (!fifo.peek_ptr(subscriber_key, &p)) return 0;
        if (!p) return 0;
        BoxedSample* box = reinterpret_cast<BoxedSample*>(p);
        if (box->magic != BOXED_SAMPLE_MAGIC) return 0;
        int need = box->sample_len;
        if (out_len < need) return 0;
        float* payload = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(box) + sizeof(BoxedSample));
        std::memcpy(out_sample, payload, static_cast<size_t>(need) * sizeof(float));
        wrote = static_cast<size_t>(need);
        ok = true;
    } else {
        ok = fifo.peek(subscriber_key, out_sample, static_cast<size_t>(out_len), wrote);
    }
    if (out_written) *out_written = static_cast<int32_t>(wrote);
    if (!ok) return 0;
    // Do NOT notify ThreadManager because we didn't advance the reader.
    return 1;
}

int32_t gp_table_edge_consume_blocking(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, float* out_sample, int32_t out_len, int32_t* out_written, int32_t timeout_ms) {
    if (out_written) *out_written = 0;
    if (!ctx || !out_sample || out_len < 0) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    size_t wrote = 0;
    uint32_t flags = 0;
    if (static_cast<size_t>(edge_idx) < ctx->edge_subgroup_flags.size()) flags = ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)];
    bool ok = false;
    if (flags_imply_byref(flags)) {
        // Wait for either a boxed pointer or a float sample. We'll loop until
        // the timeout expires. To avoid busy-waiting we sleep in short
        // intervals while checking for availability.
        using clock = std::chrono::steady_clock;
        auto start = clock::now();
        auto deadline = (timeout_ms < 0) ? clock::time_point::max() : (start + std::chrono::milliseconds(timeout_ms));
        while (true) {
            void* ppeek = nullptr;
            if (fifo.peek_ptr(subscriber_key, &ppeek)) {
                if (!ppeek) return 0;
                BoxedSample* boxpeek = reinterpret_cast<BoxedSample*>(ppeek);
                if (boxpeek->magic == BOXED_SAMPLE_MAGIC) {
                    // consume boxed sample
                    void* p = nullptr;
                    if (!fifo.pop_ptr(subscriber_key, &p)) return 0;
                    if (!p) return 0;
                    BoxedSample* box = reinterpret_cast<BoxedSample*>(p);
                    int need = box->sample_len;
                    if (out_len < need) { std::free(box); return 0; }
                    float* payload = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(box) + sizeof(BoxedSample));
                    std::memcpy(out_sample, payload, static_cast<size_t>(need) * sizeof(float));
                    std::free(box);
                    wrote = static_cast<size_t>(need);
                    ok = true;
                    break;
                } else {
                    // raw pointer delivered; don't consume here
                    return 0;
                }
            }
            // Try to pop a float sample with the remaining timeout
            if (timeout_ms == 0) break;
            auto now = clock::now();
            if (now >= deadline) break;
            int remaining_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
            if (remaining_ms <= 0) break;
            if (fifo.pop_blocking(subscriber_key, out_sample, static_cast<size_t>(out_len), wrote, remaining_ms)) {
                ok = true;
                break;
            }
            // brief sleep to yield CPU before re-checking
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } else {
        ok = fifo.pop_blocking(subscriber_key, out_sample, static_cast<size_t>(out_len), wrote, timeout_ms);
    }
    if (out_written) *out_written = static_cast<int32_t>(wrote);
    if (!ok) return 0;
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        auto it = m.find(subscriber_key);
        if (it != m.end()) {
            int slot = it->second;
            auto *r = fifo.find_reader(subscriber_key);
            if (r && slot > 0) {
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return 1;
}

int32_t gp_table_edge_unread(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, int32_t* out_count) {
    if (!ctx || !out_count) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    size_t cnt = ctx->edge_fifos[static_cast<size_t>(edge_idx)].unread(subscriber_key);
    *out_count = static_cast<int32_t>(std::min<size_t>(cnt, static_cast<size_t>(std::numeric_limits<int32_t>::max())));
    return 1;
}

int32_t gp_table_edge_set_batch_metadata(GP_TableContext* ctx, int32_t edge_idx, const GP_TableEdgeBatchMetadata* metadata) {
    if (!ctx || !metadata) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_batch_metadata.size())) return 0;
    ctx->edge_batch_metadata[static_cast<size_t>(edge_idx)] = *metadata;
    return 1;
}

int32_t gp_table_edge_get_batch_metadata(GP_TableContext* ctx, int32_t edge_idx, GP_TableEdgeBatchMetadata* out_metadata) {
    if (!ctx || !out_metadata) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_batch_metadata.size())) return 0;
    *out_metadata = ctx->edge_batch_metadata[static_cast<size_t>(edge_idx)];
    return 1;
}

int32_t gp_table_edge_set_subgroup_flags(GP_TableContext* ctx, int32_t edge_idx, uint32_t flags) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_subgroup_flags.size())) return 0;
    ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)] = flags;
    return 1;
}

extern "C" int32_t gp_table_get_edge_rope_index(const GP_TableContext* ctx, int32_t edge_idx, int32_t* out_rope_idx) {
    if (!ctx || !out_rope_idx) return 0;
    if (edge_idx < 0 || edge_idx >= static_cast<int>(ctx->rope_ids.size())) { *out_rope_idx = -1; return 0; }
    uint64_t id = ctx->rope_ids[static_cast<size_t>(edge_idx)];
    if (id == 0ull) { *out_rope_idx = -1; return 1; }
    auto it = ctx->rope_id_to_sim_idx.find(id);
    if (it == ctx->rope_id_to_sim_idx.end()) { *out_rope_idx = -1; return 1; }
    *out_rope_idx = it->second;
    return 1;
}

int32_t gp_table_edge_get_subgroup_flags(GP_TableContext* ctx, int32_t edge_idx, uint32_t* out_flags) {
    if (!ctx || !out_flags) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_subgroup_flags.size())) return 0;
    *out_flags = ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)];
    return 1;
}

int32_t gp_table_edge_index_for_key(GP_TableContext* ctx, unsigned long long led_key, int32_t* out_edge_idx) {
    if (!ctx || !out_edge_idx) return 0;
    ensure_edge_fifos(ctx);
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto& e = ctx->edges[i];
        if (e.first == led_key || e.second == led_key) {
            *out_edge_idx = static_cast<int32_t>(i);
            return 1;
        }
    }
    return 0;
}

// Resolve a persistent rope id to the attached RopeSim index for this table.
// Returns -1 if not present.
extern "C" int gp_table_resolve_rope_id_to_sim_index(GP_TableContext* ctx, uint64_t id) {
    if (!ctx) return -1;
    if (id == 0ull) return -1;
    auto it = ctx->rope_id_to_sim_idx.find(id);
    if (it == ctx->rope_id_to_sim_idx.end()) return -1;
    return it->second;
}
    // Allow external code (canvas) to set the table's persistent rope id list
    // prior to serialization so exported blobs include canonical ids.
extern "C" int gp_table_set_rope_ids_from_array(GP_TableContext* ctx, const uint64_t* ids, int count) {
    if (!ctx) return 0;
    if (!ids || count <= 0) {
        ctx->rope_ids.clear();
        if (GP_CanvasContext* cvs = gp_canvas_get_singleton()) {
            gp_canvas_mark_rope_map_dirty(cvs);
        }
        return 1;
    }
    ctx->rope_ids.assign(ids, ids + count);
    // rebuild the lookup map so callers can resolve sim indices from the
    // provided persistent ids.
    ctx->rope_id_to_sim_idx.clear();
    for (int i = 0; i < count; ++i) {
        uint64_t uid = ids[static_cast<size_t>(i)];
        if (uid != 0ull) ctx->rope_id_to_sim_idx[uid] = i;
    }
    // set next_rope_id to one past maximum to avoid collisions
    uint64_t mx = 1;
    for (auto v : ctx->rope_ids) if (v >= mx) mx = v + 1;
    ctx->next_rope_id = mx;
    if (GP_CanvasContext* cvs = gp_canvas_get_singleton()) {
        gp_canvas_mark_rope_map_dirty(cvs);
    }
    return 1;
    }

int32_t gp_table_edge_index_for_pair(GP_TableContext* ctx, unsigned long long a, unsigned long long b, int32_t* out_edge_idx) {
    if (!ctx || !out_edge_idx) return 0;
    ensure_edge_fifos(ctx);
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto& e = ctx->edges[i];
        if (e.first == a && e.second == b) {
            *out_edge_idx = static_cast<int32_t>(i);
            return 1;
        }
    }
    return 0;
}

int32_t gp_table_edge_set_delta_mode(GP_TableContext* ctx, int32_t edge_idx, int32_t delta_mode) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    ctx->edge_fifos[static_cast<size_t>(edge_idx)].set_delta_mode(delta_mode != 0);
    return 1;
}

int32_t gp_table_edge_set_order_mode(GP_TableContext* ctx, int32_t edge_idx, int32_t order_mode) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    ctx->edge_fifos[static_cast<size_t>(edge_idx)].set_order_mode(order_mode);
    return 1;
}

int32_t gp_table_remove_edge(GP_TableContext* ctx, int32_t edge_idx) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edges.size())) return 0;
    ThreadManager* tm = ThreadManager::global();
    if (tm && edge_idx < static_cast<int32_t>(ctx->edge_subscriber_slots.size())) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        for (const auto &kv : m) {
            int slot = kv.second;
            if (slot > 0) tm->unregister_reader_slot(slot);
        }
        m.clear();
    }
    auto erase_at = [&](auto &vec) {
        if (edge_idx >= 0 && edge_idx < static_cast<int32_t>(vec.size())) {
            vec.erase(vec.begin() + edge_idx);
        }
    };
    // capture id for removed edge so we can remove mapping entries
    uint64_t removed_id = 0ull;
    if (edge_idx >= 0 && edge_idx < static_cast<int32_t>(ctx->rope_ids.size())) removed_id = ctx->rope_ids[static_cast<size_t>(edge_idx)];
    erase_at(ctx->edges);
    erase_at(ctx->edge_fifos);
    erase_at(ctx->edge_batch_metadata);
    erase_at(ctx->edge_subgroup_flags);
    erase_at(ctx->edge_ids);
    erase_at(ctx->edge_subscriber_slots);
    erase_at(ctx->relax_value);
    erase_at(ctx->relax_vel);
    // remove any id->sim mapping for the removed edge id
    if (removed_id != 0ull) ctx->rope_id_to_sim_idx.erase(removed_id);
    return 1;
}

int32_t gp_table_remove_edge_pair(GP_TableContext* ctx, unsigned long long a, unsigned long long b) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto &e = ctx->edges[i];
        if ((e.first == a && e.second == b) || (e.first == b && e.second == a)) {
            return gp_table_remove_edge(ctx, static_cast<int32_t>(i));
        }
    }
    return 0;
}

int32_t gp_table_bind_stage_port(GP_TableContext* ctx, unsigned long long led_key, GP_StageContext* stage, int32_t is_output, int32_t channel) {
    if (!ctx || !stage) return 0;
    StagePortBinding b;
    b.stage = stage;
    b.is_output = is_output ? 1 : 0;
    b.channel = channel;
    ctx->stage_ports[led_key] = b;
    sync_edge_tensors_for_key(ctx, led_key);
    return 1;
}

int32_t gp_table_unbind_stage_port(GP_TableContext* ctx, unsigned long long led_key) {
    if (!ctx) return 0;
    ctx->stage_ports.erase(led_key);
    ensure_edge_fifos(ctx);
    for (size_t i = 0; i < ctx->edges.size() && i < ctx->edge_fifos.size(); ++i) {
        auto &fifo = ctx->edge_fifos[i];
        if (fifo.impl) {
            uint64_t cur = fifo.impl->writer.load(std::memory_order_relaxed);
            if (cur == led_key) fifo.impl->writer.store(0, std::memory_order_relaxed);
        }
        fifo.unsubscribe(led_key);
    }
    sync_edge_tensors_for_key(ctx, led_key);
    return 1;
}

int32_t gp_table_get_stage_port(GP_TableContext* ctx, unsigned long long led_key, GP_StageContext** out_stage, int32_t* out_is_output, int32_t* out_channel) {
    if (!ctx) return 0;
    auto it = ctx->stage_ports.find(led_key);
    if (it == ctx->stage_ports.end()) return 0;
    if (out_stage) *out_stage = it->second.stage;
    if (out_is_output) *out_is_output = it->second.is_output;
    if (out_channel) *out_channel = it->second.channel;
    return 1;
}

int32_t gp_table_get_selected_count(const GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int32_t>(ctx->selected_leds.size());
}

int32_t gp_table_get_selected_key(const GP_TableContext* ctx, int32_t idx, unsigned long long* out_key) {
    if (!ctx || !out_key) return 0;
    if (idx < 0 || idx >= static_cast<int>(ctx->selected_leds.size())) return 0;
    auto it = ctx->selected_leds.begin();
    std::advance(it, idx);
    *out_key = *it;
    return 1;
}

int32_t gp_table_set_prospective_mode(GP_TableContext* ctx, int32_t enabled) {
    if (!ctx) return 0;
    ctx->prospective_mode = enabled ? 1 : 0;
    if (!ctx->prospective_mode) ctx->prospective_initialized = false;
    return 1;
}

int32_t gp_table_get_prospective_mode(GP_TableContext* ctx, int32_t* out_enabled) {
    if (!ctx || !out_enabled) return 0;
    *out_enabled = ctx->prospective_mode;
    return 1;
}

// IO / type hint APIs
int32_t gp_table_set_key_type_hint(GP_TableContext* ctx, unsigned long long key, int32_t type_id, int32_t is_input, int32_t is_output) {
    if (!ctx) return 0;
    ctx->key_type_hint[key] = type_id;
    if (is_input) ctx->key_is_input.insert(key); else ctx->key_is_input.erase(key);
    if (is_output) ctx->key_is_output.insert(key); else ctx->key_is_output.erase(key);
    return 1;
}

int32_t gp_table_get_key_type_hint(GP_TableContext* ctx, unsigned long long key, int32_t* out_type_id, int32_t* out_is_input, int32_t* out_is_output) {
    if (!ctx) return 0;
    auto it = ctx->key_type_hint.find(key);
    if (out_type_id) *out_type_id = (it != ctx->key_type_hint.end()) ? it->second : -1;
    if (out_is_input) *out_is_input = ctx->key_is_input.count(key) ? 1 : 0;
    if (out_is_output) *out_is_output = ctx->key_is_output.count(key) ? 1 : 0;
    return 1;
}

int32_t gp_table_has_io_sections(GP_TableContext* ctx) {
    if (!ctx) return 0;
    return (ctx->key_is_input.size() > 0 && ctx->key_is_output.size() > 0) ? 1 : 0;
}

int32_t gp_table_enumerate_io_keys(GP_TableContext* ctx, int32_t direction, unsigned long long* out_keys, int32_t cap) {
    if (!ctx || !out_keys || cap <= 0) return 0;
    int written = 0;
    if (direction == 0) {
        for (auto k : ctx->key_is_input) {
            if (written >= cap) break;
            out_keys[written++] = k;
        }
    } else {
        for (auto k : ctx->key_is_output) {
            if (written >= cap) break;
            out_keys[written++] = k;
        }
    }
    return written;
}

int32_t gp_table_set_side_reading_direction(GP_TableContext* ctx, int32_t side, int32_t dir) {
    if (!ctx) return 0;
    if (side < 0 || side > 1) return 0;
    if (dir < 0 || dir > 3) return 0;
    ctx->side_reading_dir[side] = dir;
    return 1;
}

int32_t gp_table_get_side_reading_direction(GP_TableContext* ctx, int32_t side, int32_t* out_dir) {
    if (!ctx || !out_dir) return 0;
    if (side < 0 || side > 1) return 0;
    *out_dir = ctx->side_reading_dir[side];
    return 1;
}

int32_t gp_table_set_led_grid_preference(GP_TableContext* ctx, int32_t pref_cols, int32_t pref_rows, float pref_aspect) {
    if (!ctx) return 0;
    ctx->led_pref_cols = std::max(0, pref_cols);
    ctx->led_pref_rows = std::max(0, pref_rows);
    ctx->led_pref_aspect = pref_aspect;
    return 1;
}

int32_t gp_table_get_led_grid_preference(GP_TableContext* ctx, int32_t* out_pref_cols, int32_t* out_pref_rows, float* out_pref_aspect) {
    if (!ctx) return 0;
    if (out_pref_cols) *out_pref_cols = ctx->led_pref_cols;
    if (out_pref_rows) *out_pref_rows = ctx->led_pref_rows;
    if (out_pref_aspect) *out_pref_aspect = ctx->led_pref_aspect;
    return 1;
}

int32_t gp_table_prospective_set_params(GP_TableContext* ctx, int32_t max_history, float slack, float rope_length) {
    if (!ctx) return 0;
    ctx->prospective_max_history = std::max(1, max_history);
    ctx->prospective_slack = slack;
    ctx->prospective_rope_length = rope_length;
    // trim existing queue if needed
    if (static_cast<int>(ctx->prospective_targets.size()) > ctx->prospective_max_history) {
        ctx->prospective_targets.erase(ctx->prospective_targets.begin(), ctx->prospective_targets.begin() + (ctx->prospective_targets.size() - ctx->prospective_max_history));
    }
    return 1;
}

int32_t gp_table_prospective_get_params(GP_TableContext* ctx, int32_t* out_max_history, float* out_slack, float* out_rope_length) {
    if (!ctx) return 0;
    if (out_max_history) *out_max_history = ctx->prospective_max_history;
    if (out_slack) *out_slack = ctx->prospective_slack;
    if (out_rope_length) *out_rope_length = ctx->prospective_rope_length;
    return 1;
}

// Editable flag helpers
int32_t gp_table_set_editable(GP_TableContext* ctx, int32_t editable) {
    if (!ctx) return 0;
    ctx->editable = editable ? 1 : 0;
    return 1;
}

int32_t gp_table_get_editable(GP_TableContext* ctx, int32_t* out_editable) {
    if (!ctx || !out_editable) return 0;
    *out_editable = ctx->editable;
    return 1;
}

// Serialization format (simple binary):
// [8 bytes magic 'GPTBL001'][uint32_t version]
// version 1: GP_TableStyle, cols, rows, edges, selected keys
// version 2: same as v1, then int32 meta_group_count, followed by per-meta-group blob
// Per-meta-group blob: int32 vertex_count, (int32 rope_idx,int32 vert_idx)*N, float confinement, int32 sim_group_idx,
// uint64 id, LassoConfig (8 bytes), uint64 anchor_rope_uid, int32 anchor_vert, uint32 subgroup_flags, int32 channel_group,
// int32 dangling_widget_rope, int32 dangling_widget_rope_vid, float dangling_hang_len, int32 ring_mode,
// uint64 overlay_key_a, uint64 overlay_key_b
int32_t gp_table_serialize(GP_TableContext* ctx, char* out_buf, int32_t out_len) {
    if (!ctx) return 0;
    const uint32_t version = 3;
    const char magic[8] = {'G','P','T','B','L','0','0','1'};
    // Build a canonical UUID atlas covering all UUIDs referenced by this
    // table blob (edges, ropes, overlays, module/frame ports, rings,
    // meta-group ids). The atlas will be written immediately after the
    // version so loaders can read it first and reserve/register any
    // required identities before reconstructing dependent structures.
    std::unordered_set<uint64_t> _atlas_set;
    if (ctx->module_uuid != 0ull) _atlas_set.insert(ctx->module_uuid);
    for (const auto &fpe : ctx->frame_port_uuids) if (fpe.uuid != 0ull) _atlas_set.insert(fpe.uuid);
    for (auto ru : ctx->rope_ids) if (ru != 0ull) _atlas_set.insert(ru);
    for (const auto &e : ctx->edges) { if (e.first != 0ull) _atlas_set.insert(e.first); if (e.second != 0ull) _atlas_set.insert(e.second); }
    for (const auto &re : ctx->rings) if (re.key != 0ull) _atlas_set.insert(re.key);
    for (size_t mi = 0; mi < ctx->meta_groups.size(); ++mi) {
        GP_MetaGroup* mg = ctx->meta_groups[mi].get();
        if (!mg) continue;
        if (mg->overlay_key_a != 0ull) _atlas_set.insert(mg->overlay_key_a);
        if (mg->overlay_key_b != 0ull) _atlas_set.insert(mg->overlay_key_b);
        if (mg->id != 0ull) _atlas_set.insert(mg->id);
        if (mg->anchor_rope >= 0 && static_cast<size_t>(mg->anchor_rope) < ctx->rope_ids.size()) _atlas_set.insert(ctx->rope_ids[static_cast<size_t>(mg->anchor_rope)]);
    }
    std::vector<uint64_t> atlas;
    atlas.reserve(_atlas_set.size());
    for (auto v : _atlas_set) atlas.push_back(v);
    int32_t col_count = static_cast<int32_t>(ctx->cols.size());
    int32_t row_count = static_cast<int32_t>(ctx->rows.size());
    int32_t edge_count = static_cast<int32_t>(ctx->edges.size());
    int32_t sel_count = static_cast<int32_t>(ctx->selected_leds.size());
    int32_t need = 0;
    need += 8; // magic
    need += 4; // version
    // atlas count + entries (written immediately after version)
    need += 4; // atlas_count
    need += static_cast<int32_t>(atlas.size()) * static_cast<int32_t>(sizeof(uint64_t));
    need += static_cast<int32_t>(sizeof(GP_TableStyle));
    need += 4; // col_count
    need += col_count * static_cast<int32_t>(sizeof(GP_TableColumn));
    need += 4; // row_count
    need += row_count * static_cast<int32_t>(sizeof(GP_TableRow));
    need += 4; // edge_count
    need += edge_count * static_cast<int32_t>(sizeof(uint64_t) * 2);
    // per-rope persistent uids (one per edge/rope created)
    need += 4; // rope_uid_count
    need += edge_count * static_cast<int32_t>(sizeof(uint64_t));
    need += 4; // sel_count
    need += sel_count * static_cast<int32_t>(sizeof(uint64_t));
    // module uuid + frame-port entries (version 3)
    need += 8; // module_uuid
    // frame port count + entries
    need += 4; // frame_port_count
    // each entry: row(int32), idx(int32), uuid(uint64)
    need += static_cast<int32_t>(ctx->frame_port_uuids.size()) * (4 + 4 + 8);
    // meta-groups (version 2)
    int32_t mg_count = static_cast<int32_t>(ctx->meta_groups.size());
    need += 4; // mg_count
    for (int i = 0; i < mg_count; ++i) {
        GP_MetaGroup* mg = ctx->meta_groups[static_cast<size_t>(i)].get();
        int32_t vcount = static_cast<int32_t>(mg ? mg->vertices.size() : 0);
        need += 4; // vcount
        need += vcount * (8 + 4); // rope_uid,uint64 + vertex_idx,int32
        need += 4; // confinement (stored as float)
        need += 4; // sim_group_idx
        need += 8; // id (uint64)
        need += static_cast<int32_t>(sizeof(LassoConfig)); // lasso_config
        need += 8; // anchor_rope_uid (uint64)
        need += 4; // anchor_vert
        need += 4; // subgroup_flags (uint32)
        need += 4; // channel_group
        need += 4; // dangling_widget_rope
        need += 4; // dangling_widget_rope_vid
        need += 4; // dangling_hang_len (float)
        need += 4; // ring_mode
        need += 4; // ring_u (float)
        need += 8; // overlay_key_a
        need += 8; // overlay_key_b
        need += 8; // overlay port_uuid_a
        need += 8; // overlay port_uuid_b
    }

    // Log meta-group count and total size needed when serializing (helps
    // troubleshooting when saved modules appear to lack meta-groups).
    printf("gp_table_serialize: ctx=%p mg_count=%d need=%d\n", (void*)ctx, mg_count, need);
    if (!ctx->rope_ids.empty()) {
        printf("gp_table_serialize: rope_ids:");
        for (size_t ri = 0; ri < ctx->rope_ids.size(); ++ri) printf(" %llu", (unsigned long long)ctx->rope_ids[ri]);
        printf("\n");
    } else {
        printf("gp_table_serialize: rope_ids: <empty>\n");
    }

    if (!out_buf) return need;
    if (out_len < need) return 0;

    char* p = out_buf;
    // write magic
    memcpy(p, magic, 8); p += 8;
    // write version
    memcpy(p, &version, 4); p += 4;
    // write atlas: count + entries
    int32_t atlas_count = static_cast<int32_t>(atlas.size());
    memcpy(p, &atlas_count, 4); p += 4;
    for (int ai = 0; ai < atlas_count; ++ai) { uint64_t u = atlas[static_cast<size_t>(ai)]; memcpy(p, &u, sizeof(uint64_t)); p += sizeof(uint64_t); }
    // write style_raw
    memcpy(p, &ctx->style_raw, sizeof(GP_TableStyle)); p += sizeof(GP_TableStyle);
    // write cols
    memcpy(p, &col_count, 4); p += 4;
    if (col_count > 0) {
        memcpy(p, ctx->cols.data(), static_cast<size_t>(col_count) * sizeof(GP_TableColumn));
        p += col_count * static_cast<int32_t>(sizeof(GP_TableColumn));
    }
    // write rows (we zero any waveform pointers)
    memcpy(p, &row_count, 4); p += 4;
    for (int i = 0; i < row_count; ++i) {
        GP_TableRow row = ctx->rows[static_cast<size_t>(i)];
        // clear wave pointers inside cells to avoid serializing pointers
        for (int c = 0; c < row.cell_count && c < 8; ++c) {
            row.cells[c].wave = nullptr;
            row.cells[c].image = nullptr;
        }
        memcpy(p, &row, sizeof(GP_TableRow)); p += sizeof(GP_TableRow);
    }
    // write edges
    memcpy(p, &edge_count, 4); p += 4;
    for (int i = 0; i < edge_count; ++i) {
        uint64_t a = ctx->edges[static_cast<size_t>(i)].first;
        uint64_t b = ctx->edges[static_cast<size_t>(i)].second;
        memcpy(p, &a, sizeof(uint64_t)); p += sizeof(uint64_t);
        memcpy(p, &b, sizeof(uint64_t)); p += sizeof(uint64_t);
    }
    // write per-rope ids (one per edge/rope)
    int32_t rope_id_count = static_cast<int32_t>(ctx->rope_ids.size());
    memcpy(p, &rope_id_count, 4); p += 4;
    for (int i = 0; i < rope_id_count; ++i) {
        uint64_t ru = ctx->rope_ids[static_cast<size_t>(i)];
        memcpy(p, &ru, sizeof(uint64_t)); p += sizeof(uint64_t);
    }
    // write selected keys
    memcpy(p, &sel_count, 4); p += 4;
    for (auto k : ctx->selected_leds) {
        uint64_t key = k;
        memcpy(p, &key, sizeof(uint64_t)); p += sizeof(uint64_t);
    }

    // write module uuid and frame-port entries (version 3)
    uint64_t module_uuid = ctx->module_uuid;
    memcpy(p, &module_uuid, 8); p += 8;
    int32_t fpcount = static_cast<int32_t>(ctx->frame_port_uuids.size());
    memcpy(p, &fpcount, 4); p += 4;
    for (int i = 0; i < fpcount; ++i) {
        int32_t r = ctx->frame_port_uuids[static_cast<size_t>(i)].row;
        int32_t idx = ctx->frame_port_uuids[static_cast<size_t>(i)].idx;
        uint64_t pu = ctx->frame_port_uuids[static_cast<size_t>(i)].uuid;
        memcpy(p, &r, 4); p += 4;
        memcpy(p, &idx, 4); p += 4;
        memcpy(p, &pu, 8); p += 8;
    }

    // write meta-groups (version 3)
    memcpy(p, &mg_count, 4); p += 4;
    for (int i = 0; i < mg_count; ++i) {
        GP_MetaGroup* mg = ctx->meta_groups[static_cast<size_t>(i)].get();
        if (mg) {
            LassoConfig lc = mg->lasso_config;
            printf("gp_table_serialize: mg id=%llu vcount=%d sim_idx=%d lasso.spring_min_rest=%.2f spring_reduce_rate=%.2f spring_mode=%u\n",
                   (unsigned long long)mg->id, (int)mg->vertices.size(), mg->sim_group_idx, lc.spring_min_rest, lc.spring_reduce_rate, (unsigned)lc.spring_mode);
            // Dump full meta-group for diagnostics
            gp_table_debug_dump_meta_group(ctx, mg, "serialize");
        } else {
            printf("gp_table_serialize: mg <null>\n");
        }
        int32_t vcount = static_cast<int32_t>(mg ? mg->vertices.size() : 0);
        memcpy(p, &vcount, 4); p += 4;
        for (int vi = 0; vi < vcount; ++vi) {
            int32_t rope_idx = mg->vertices[static_cast<size_t>(vi)].first;
            int32_t vert_idx = mg->vertices[static_cast<size_t>(vi)].second;
            uint64_t rope_id = 0ull;
            if (rope_idx >= 0 && static_cast<size_t>(rope_idx) < ctx->rope_ids.size()) rope_id = ctx->rope_ids[static_cast<size_t>(rope_idx)];
            memcpy(p, &rope_id, 8); p += 8;
            memcpy(p, &vert_idx, 4); p += 4;
        }
        float conf = mg ? mg->confinement : 1.0f;
        memcpy(p, &conf, 4); p += 4;
        int32_t sim_idx = mg ? mg->sim_group_idx : -1;
        memcpy(p, &sim_idx, 4); p += 4;
        uint64_t mgid = mg ? mg->id : 0ull;
        memcpy(p, &mgid, 8); p += 8;
        LassoConfig lc{};
        if (mg) lc = mg->lasso_config;
        memcpy(p, &lc, sizeof(LassoConfig)); p += sizeof(LassoConfig);
        // write anchor as a persistent rope id (0 == none)
        uint64_t anchor_rope_id = 0ull;
        int32_t anchor_v = -1;
        if (mg) {
            anchor_v = mg->anchor_vert;
            if (mg->anchor_rope >= 0 && static_cast<size_t>(mg->anchor_rope) < ctx->rope_ids.size()) anchor_rope_id = ctx->rope_ids[static_cast<size_t>(mg->anchor_rope)];
        }
        memcpy(p, &anchor_rope_id, 8); p += 8;
        memcpy(p, &anchor_v, 4); p += 4;
        uint32_t sgflags = mg ? mg->subgroup_flags : 0u;
        memcpy(p, &sgflags, 4); p += 4;
        int32_t channel_group = mg ? mg->channel_group : 0;
        memcpy(p, &channel_group, 4); p += 4;
        int32_t dang_rope = mg ? mg->dangling_widget_rope : -1;
        int32_t dang_vid = mg ? mg->dangling_widget_rope_vid : -1;
        memcpy(p, &dang_rope, 4); p += 4;
        memcpy(p, &dang_vid, 4); p += 4;
        float dang_len = mg ? mg->dangling_hang_len : 0.0f;
        memcpy(p, &dang_len, 4); p += 4;
        int32_t ring_mode = mg ? mg->ring_mode : 0;
        memcpy(p, &ring_mode, 4); p += 4;
        float ring_u = 0.0f;
        if (mg) {
            // attempt to find a ring registered for this meta-group by id
            for (size_t ri = 0; ri < ctx->rings.size(); ++ri) {
                const auto &re = ctx->rings[ri];
                if (re.key == mg->id) {
                    int ring_id = re.ring_id;
                    gp_table_get_ring_u(ctx, ring_id, &ring_u);
                    break;
                }
            }
        }
        memcpy(p, &ring_u, 4); p += 4;
        uint64_t oka = mg ? mg->overlay_key_a : 0ull;
        uint64_t okb = mg ? mg->overlay_key_b : 0ull;
        memcpy(p, &oka, 8); p += 8;
        memcpy(p, &okb, 8); p += 8;
        // persist overlay per-port UUIDs so restore can re-establish exact port identities
        uint64_t pu_a = 0ull, pu_b = 0ull;
        GP_CanvasContext* cvs = gp_canvas_get_singleton();
        if (cvs && (oka || okb)) {
            gp_canvas_get_overlay_port_uuids(cvs, oka, okb, &pu_a, &pu_b);
        }
        memcpy(p, &pu_a, 8); p += 8;
        memcpy(p, &pu_b, 8); p += 8;
    }

    return need;
}

int32_t gp_table_deserialize(GP_TableContext* ctx, const char* in_buf, int32_t in_len) {
    if (!ctx || !in_buf || in_len <= 0) return 0;
    const char expect_magic[8] = {'G','P','T','B','L','0','0','1'};
    if (in_len < 8 + 4 + static_cast<int>(sizeof(GP_TableStyle))) return 0;
    const char* p = in_buf;
    if (memcmp(p, expect_magic, 8) != 0) return 0;
    p += 8;
    uint32_t version = 0;
    memcpy(&version, p, 4); p += 4;
    // Read UUID atlas (written immediately after version).
    std::vector<uint64_t> atlas;
    if (p + 4 > in_buf + in_len) return 0;
    int32_t atlas_count = 0;
    memcpy(&atlas_count, p, 4); p += 4;
    if (atlas_count < 0) return 0;
    atlas.reserve(static_cast<size_t>(atlas_count));
    for (int i = 0; i < atlas_count; ++i) {
        if (p + 8 > in_buf + in_len) return 0;
        uint64_t u = 0ull; memcpy(&u, p, 8); p += 8;
        atlas.push_back(u);
    }
    printf("gp_table_deserialize: read UUID atlas count=%d\n", atlas_count);
    printf("gp_table_deserialize: canvas singleton=%p\n", (void*)gp_canvas_get_singleton());
    printf("gp_table_deserialize: read UUID atlas count=%d\n", atlas_count);
    // trace: report deserialize invocation
    printf("gp_table_deserialize: ctx=%p len=%d version=%u\n", (void*)ctx, in_len, (unsigned)version);
    // Canvas singleton (may be null). Declare early so deserialization can
    // register persisted rope/module UUIDs with the canvas if available.
    GP_CanvasContext* cvs = gp_canvas_get_singleton();
    if (version < 2 || version > 3) return 0;
    // read style
    GP_TableStyle style{};
    memcpy(&style, p, sizeof(GP_TableStyle)); p += sizeof(GP_TableStyle);
    // columns
    int32_t col_count = 0;
    memcpy(&col_count, p, 4); p += 4;
    if (col_count < 0 || col_count > 8) return 0;
    std::vector<GP_TableColumn> cols;
    if (col_count > 0) {
        cols.resize(static_cast<size_t>(col_count));
        memcpy(cols.data(), p, static_cast<size_t>(col_count) * sizeof(GP_TableColumn));
        p += col_count * static_cast<int>(sizeof(GP_TableColumn));
    }
    // rows
    int32_t row_count = 0;
    memcpy(&row_count, p, 4); p += 4;
    if (row_count < 0) return 0;
    std::vector<GP_TableRow> rows;
    if (row_count > 0) {
        rows.resize(static_cast<size_t>(row_count));
        for (int i = 0; i < row_count; ++i) {
            memcpy(&rows[static_cast<size_t>(i)], p, sizeof(GP_TableRow)); p += sizeof(GP_TableRow);
            // ensure wave pointers are null for safety
            for (int c = 0; c < rows[static_cast<size_t>(i)].cell_count && c < 8; ++c) rows[static_cast<size_t>(i)].cells[c].wave = nullptr;
        }
    }
    // edges
    int32_t edge_count = 0;
    memcpy(&edge_count, p, 4); p += 4;
    if (edge_count < 0) return 0;
    std::vector<std::pair<uint64_t,uint64_t>> edges;
    edges.reserve(static_cast<size_t>(edge_count));
    for (int i = 0; i < edge_count; ++i) {
        uint64_t a=0,b=0;
        memcpy(&a, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        memcpy(&b, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        edges.emplace_back(a,b);
    }
    // read per-rope uids
    int32_t rope_uid_count = 0;
    if (p + 4 > in_buf + in_len) return 0;
    memcpy(&rope_uid_count, p, 4); p += 4;
    std::vector<uint64_t> rope_ids_temp;
    rope_ids_temp.reserve(static_cast<size_t>(std::max(0, rope_uid_count)));
    for (int i = 0; i < rope_uid_count; ++i) {
        uint64_t ru = 0ull;
        if (p + 8 > in_buf + in_len) return 0;
        memcpy(&ru, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        rope_ids_temp.push_back(ru);
    }
    printf("gp_table_deserialize: rope_uid_count=%d\n", rope_uid_count);
    if (!rope_ids_temp.empty()) {
        printf("gp_table_deserialize: rope_ids:");
        for (size_t ri = 0; ri < rope_ids_temp.size(); ++ri) printf(" %llu", (unsigned long long)rope_ids_temp[ri]);
        printf("\n");
    }
    // selected keys
    int32_t sel_count = 0;
    memcpy(&sel_count, p, 4); p += 4;
    if (sel_count < 0) return 0;
    std::unordered_set<uint64_t> selset;
    for (int i = 0; i < sel_count; ++i) {
        uint64_t key = 0;
        memcpy(&key, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        selset.insert(key);
    }

    // version 3: read module UUID and frame-port entries
    uint64_t module_uuid = 0ull;
    if (version >= 3) {
        if (p + 8 > in_buf + in_len) return 0;
        memcpy(&module_uuid, p, 8); p += 8;
        int32_t fpcount = 0;
        if (p + 4 > in_buf + in_len) return 0;
        memcpy(&fpcount, p, 4); p += 4;
        if (fpcount < 0) return 0;
        std::vector<std::tuple<int32_t,int32_t,uint64_t>> fp_entries;
        fp_entries.reserve(static_cast<size_t>(fpcount));
        for (int i = 0; i < fpcount; ++i) {
            if (p + 4 + 4 + 8 > in_buf + in_len) return 0;
            int32_t r=0, idx=0; uint64_t pu=0ull;
            memcpy(&r, p, 4); p += 4;
            memcpy(&idx, p, 4); p += 4;
            memcpy(&pu, p, 8); p += 8;
            fp_entries.emplace_back(r, idx, pu);
        }
        // attach module_uuid and frame-port entries to ctx so callers (canvas)
        // can read them when needed. We store into ctx fields for later use.
        ctx->module_uuid = module_uuid;
        ctx->frame_port_uuids.clear();
        for (auto &ent : fp_entries) {
            int32_t r = std::get<0>(ent);
            int32_t idx = std::get<1>(ent);
            uint64_t pu = std::get<2>(ent);
            GP_TableContext::FramePortEntry fpe{}; fpe.row = r; fpe.idx = idx; fpe.uuid = pu;
            ctx->frame_port_uuids.push_back(fpe);
        }
    }

    // Commit to ctx: replace style, cols, rows, edges, selected set
    ctx->style_raw = style;
    ctx->st = load_style(&ctx->style_raw);
    ctx->cols = std::move(cols);
    ctx->rows = std::move(rows);
    // clear existing edges via API to keep rope_sim indices consistent
    gp_table_clear_edges(ctx);
    for (auto &e : edges) gp_table_add_edge(ctx, e.first, e.second);
    // if serialized per-rope uids were present, adopt them so runtime rope
    // indices map to the saved stable ids (this preserves stable mapping
    // for meta-group vertices which reference ropes by uid).
    if (!rope_ids_temp.empty()) {
        ctx->rope_ids = rope_ids_temp;
        // rebuild rope_id_to_sim_idx so canvas lookups resolve to the current
        // rope indices (edge order) using the persisted ids.
        ctx->rope_id_to_sim_idx.clear();
        for (size_t i = 0; i < ctx->rope_ids.size(); ++i) {
            uint64_t uid = ctx->rope_ids[i];
            if (uid != 0ull) ctx->rope_id_to_sim_idx[uid] = static_cast<int>(i);
        }
        // set next_rope_id to one past the max saved id
        uint64_t mx = 1;
        for (auto ru : rope_ids_temp) if (ru >= mx) mx = ru + 1;
        ctx->next_rope_id = mx;
        // inform canvas of this table's persisted rope IDs so the canvas
        // can build a canonical mapping for deterministic restoration.
        if (cvs) {
            gp_canvas_register_table_rope_ids_from_array(cvs, ctx, ctx->rope_ids.data(), static_cast<int>(ctx->rope_ids.size()));
        }
    }
    ctx->selected_leds = std::move(selset);
    recompute_geom(ctx);
    // Parse and restore meta-groups (version 2)
    // ensure there's enough data remaining
    if (p + 4 > in_buf + in_len) return 1; // nothing more
    int32_t mg_count = 0;
    memcpy(&mg_count, p, 4); p += 4;
    printf("gp_table_deserialize: meta-group count=%d\n", mg_count);
    if (mg_count < 0) return 1;
    struct MGData { std::vector<std::pair<uint64_t,int>> verts; float confinement; int32_t sim_idx; uint64_t id; LassoConfig lc; uint64_t anchor_uid; int32_t anchor_v; uint32_t subgroup_flags; int32_t channel_group; int32_t dang_rope; int32_t dang_vid; float dang_len; int32_t ring_mode; float ring_u; uint64_t oka; uint64_t okb; uint64_t port_a; uint64_t port_b; };
    std::vector<MGData> mgds;
    mgds.reserve(static_cast<size_t>(mg_count));
    for (int m = 0; m < mg_count; ++m) {
        if (p + 4 > in_buf + in_len) return 1;
        int32_t vcount = 0;
        memcpy(&vcount, p, 4); p += 4;
        MGData d; d.verts.reserve(static_cast<size_t>(std::max(0, vcount)));
        for (int vi = 0; vi < vcount; ++vi) {
            uint64_t ru = 0ull; int32_t vid = 0;
            if (p + 8 > in_buf + in_len) return 1;
            memcpy(&ru, p, 8); p += 8;
            memcpy(&vid, p, 4); p += 4;
            d.verts.emplace_back(ru, vid);
        }
        memcpy(&d.confinement, p, 4); p += 4;
        memcpy(&d.sim_idx, p, 4); p += 4;
        memcpy(&d.id, p, 8); p += 8;
        memcpy(&d.lc, p, sizeof(LassoConfig)); p += sizeof(LassoConfig);
        uint64_t anchor_uid = 0ull;
        memcpy(&anchor_uid, p, 8); p += 8;
        memcpy(&d.anchor_v, p, 4); p += 4;
        d.anchor_uid = anchor_uid;
        memcpy(&d.subgroup_flags, p, 4); p += 4;
        memcpy(&d.channel_group, p, 4); p += 4;
        memcpy(&d.dang_rope, p, 4); p += 4;
        memcpy(&d.dang_vid, p, 4); p += 4;
        memcpy(&d.dang_len, p, 4); p += 4;
        memcpy(&d.ring_mode, p, 4); p += 4;
        memcpy(&d.ring_u, p, 4); p += 4;
        memcpy(&d.oka, p, 8); p += 8;
        memcpy(&d.okb, p, 8); p += 8;
        // read persisted per-port UUIDs (may be zero)
        memcpy(&d.port_a, p, 8); p += 8;
        memcpy(&d.port_b, p, 8); p += 8;
        mgds.push_back(std::move(d));
    }

    // Recreate meta-groups in the context
    for (auto &d : mgds) {
        GP_MetaGroup* mg = gp_table_meta_create(ctx);
        if (!mg) continue;
        printf("gp_table_deserialize: creating meta-group ctx=%p mg=%p id=%llu oka=%llu okb=%llu verts=%zu sim_idx=%d\n",
               (void*)ctx, (void*)mg, static_cast<unsigned long long>(d.id), static_cast<unsigned long long>(d.oka), static_cast<unsigned long long>(d.okb), d.verts.size(), d.sim_idx);
        // restore simple fields
        mg->confinement = d.confinement;
        // Do NOT blindly restore the serialized sim_group_idx value: rope-sim
        // meta-group indices are not stable across process runs or when the
        // RopeSim instance is (re)attached. Leave sim_group_idx unset so
        // that `gp_table_meta_add_vertex` will create a fresh sim meta-group
        // in the current simulator and register members deterministically.
        mg->sim_group_idx = -1;
        mg->id = d.id;
        mg->lasso_config = d.lc;
        // resolve persisted anchor UID to runtime rope index (if present)
        mg->anchor_rope = -1;
        if (d.anchor_uid != 0ull) {
            int resolved = -1;
            if (cvs) resolved = gp_canvas_resolve_rope_id_to_index(cvs, ctx, d.anchor_uid);
            if (resolved < 0) {
                for (size_t ri = 0; ri < ctx->rope_ids.size(); ++ri) {
                    if (ctx->rope_ids[ri] == d.anchor_uid) { resolved = static_cast<int>(ri); break; }
                }
            }
            if (resolved >= 0) {
                mg->anchor_rope = resolved;
            } else {
                printf("gp_table_deserialize: ERROR - could not resolve anchor rope id=%llu\n", (unsigned long long)d.anchor_uid);
                return 2; // hard-fail: unresolved persisted anchor UID
            }
        }
        mg->anchor_vert = d.anchor_v;
        mg->subgroup_flags = d.subgroup_flags;
        mg->channel_group = d.channel_group;
        mg->dangling_widget_rope = d.dang_rope;
        mg->dangling_widget_rope_vid = d.dang_vid;
        mg->dangling_hang_len = d.dang_len;
        mg->ring_mode = d.ring_mode;
        // Register canonical overlay early so ring/T-off creation can attach
        // to the persisted overlay keys instead of creating new overlays.
        if (cvs && (d.oka != 0ull || d.okb != 0ull)) {
            gp_canvas_register_table_overlay(cvs, d.oka, d.okb, d.port_a, d.port_b);
            mg->overlay_key_a = d.oka;
            mg->overlay_key_b = d.okb;
        }
        // If a ring parameter was serialized, prefer to defer ring creation
        // to MetaCloud re-application when available. For now, recreate ring
        // here the old way to preserve prior behavior.
        if (d.ring_mode != 0) {
            float ru = d.ring_u;
            uint64_t target_uid = 0ull;
            if (d.anchor_uid != 0ull) target_uid = d.anchor_uid;
            else if (!d.verts.empty()) target_uid = d.verts[0].first;
            if (target_uid != 0ull) {
                int ring_id = gp_table_create_ring_by_id(ctx, target_uid, ru);
                if (ring_id == -2) {
                    printf("gp_table_deserialize: ERROR - could not resolve rope id=%llu for ring creation\n", (unsigned long long)target_uid);
                    return 3; // hard-fail: unresolved rope id for ring
                }
                if (ring_id >= 0) {
                    gp_table_register_ring_edge(ctx, ring_id, mg->id);
                    printf("gp_table_deserialize: recreated ring id=%d u=%.3f for mg=%p id=%llu\n", ring_id, ru, (void*)mg, (unsigned long long)mg->id);
                }
            }
        }
        mg->overlay_key_a = d.oka;
        mg->overlay_key_b = d.okb;
        // If a canvas is present, prefer attaching the canvas root RopeSim
        // so meta-group membership is registered against the shared root sim
        // instead of creating a local simulator. Do not synthesize a local
        // sim here; defer to the canvas to create one if needed.
        if (cvs && !ctx->rope_sim) {
            void* root_sim_void = gp_canvas_get_rope_sim(cvs);
            if (root_sim_void) {
                RopeSim* rootsim = reinterpret_cast<RopeSim*>(root_sim_void);
                gp_table_attach_rope_sim(ctx, rootsim, 0);
                printf("gp_table_deserialize: attached canvas root RopeSim %p to table %p\n", (void*)rootsim, (void*)ctx);
            } else {
                printf("gp_table_deserialize: canvas has no root RopeSim available; deferring sim attachment for table %p\n", (void*)ctx);
            }
        }

        // add vertices (resolve via canvas mapping by persistent rope UID)
        printf("gp_table_deserialize: mg id=%llu sim_idx=%d overlay_oka=%llu okb=%llu lasso_flags=%u widget=%u spring_min_rest=%.2f spring_reduce_rate=%.2f spring_mode=%u verts=%zu\n",
               (unsigned long long)mg->id, d.sim_idx, (unsigned long long)d.oka, (unsigned long long)d.okb,
               d.lc.flags, static_cast<unsigned int>(d.lc.widget_type), d.lc.spring_min_rest, d.lc.spring_reduce_rate, static_cast<unsigned int>(d.lc.spring_mode), d.verts.size());

        for (auto &vp : d.verts) {
            uint64_t want_uid = vp.first;
            if (want_uid == 0ull) {
                printf("gp_table_deserialize: warning, vertex has zero rope_uid, skipping\n");
                continue;
            }
            if (cvs) {
                int resolved = gp_canvas_resolve_rope_id_to_index(cvs, ctx, want_uid);
                if (resolved < 0) {
                    // fallback: try table-local mapping
                    for (size_t ri = 0; ri < ctx->rope_ids.size(); ++ri) {
                        if (ctx->rope_ids[ri] == want_uid) { resolved = static_cast<int>(ri); break; }
                    }
                }
                printf("gp_table_deserialize: mapping rope_id=%llu -> resolved_idx=%d vert=%d\n", (unsigned long long)want_uid, resolved, vp.second);
                int ok = gp_canvas_table_meta_add_vertex_by_id(cvs, ctx, reinterpret_cast<void*>(mg), want_uid, vp.second);
                if (!ok) {
                    printf("gp_table_deserialize: ERROR - canvas failed to add meta vertex for rope_id=%llu\n", (unsigned long long)want_uid);
                    return 4; // hard-fail: could not add meta vertex by id
                }
            } else {
                // fallback: try table-local mapping (deprecated)
                int mapped_idx = -1;
                for (size_t ri = 0; ri < ctx->rope_ids.size(); ++ri) {
                    if (ctx->rope_ids[ri] == want_uid) { mapped_idx = static_cast<int>(ri); break; }
                }
                if (mapped_idx < 0) {
                    printf("gp_table_deserialize: ERROR - could not find rope id=%llu for meta-group vertex\n", (unsigned long long)want_uid);
                    return 5; // hard-fail: could not resolve vertex rope UID
                }
                gp_table_meta_add_vertex(ctx, mg, mapped_idx, vp.second);
                printf("gp_table_deserialize: added vertex (fallback) mapped_idx=%d vert=%d\n", mapped_idx, vp.second);
            }
            // If we registered an overlay earlier, attach the ropes referenced
            // by this meta-group deterministically to that canonical overlay.
            if ((mg->overlay_key_a != 0ull || mg->overlay_key_b != 0ull) && cvs) {
                std::unordered_set<uint64_t> seen_rope_ids;
                for (const auto &vp : d.verts) {
                    uint64_t ru = vp.first;
                    if (ru == 0ull) continue;
                    if (seen_rope_ids.find(ru) != seen_rope_ids.end()) continue;
                    seen_rope_ids.insert(ru);
                    int resolved = gp_canvas_resolve_rope_id_to_index(cvs, ctx, ru);
                    if (resolved < 0) {
                        for (size_t ri = 0; ri < ctx->rope_ids.size(); ++ri) {
                            if (ctx->rope_ids[ri] == ru) { resolved = static_cast<int>(ri); break; }
                        }
                    }
                    if (resolved >= 0) {
                        gp_canvas_attach_rope_to_overlay(cvs, mg->overlay_key_a, mg->overlay_key_b, resolved);
                    }
                }
                gp_canvas_set_overlay_meta(cvs, mg->overlay_key_a, mg->overlay_key_b, ctx, reinterpret_cast<void*>(mg));
            }
        }
        // If the serialized blob indicated a sim meta-group existed previously
        // (d.sim_idx >= 0) then enable edge-springs on the reconstituted
        // meta-group now that vertices have been registered into the current
        // RopeSim. This re-enables the short T-off / spring binding behavior
        // that was active when the snapshot was taken.
        if (d.sim_idx >= 0) {
            // prefer saved spring params from the serialized LassoConfig; fall
            // back to conservative defaults if they are zero/unset.
            float use_min_rest = (d.lc.spring_min_rest > 0.0f) ? d.lc.spring_min_rest : 2.0f;
            float use_reduce_rate = (d.lc.spring_reduce_rate > 0.0f) ? d.lc.spring_reduce_rate : 50.0f;
            gp_table_meta_set_edge_spring_params(ctx, mg, use_min_rest, use_reduce_rate, static_cast<int32_t>(d.lc.spring_mode));
            gp_table_meta_enable_edge_springs(ctx, mg, use_min_rest, use_reduce_rate);
        }
        // log sim installation if present
        {
            RopeSim* sim = ctx->rope_sim;
            if (sim && mg->sim_group_idx >= 0) {
                printf("gp_table_deserialize: installed mg=%p id=%llu sim=%p sim_group_idx=%d vertices=%zu\n",
                       (void*)mg, (unsigned long long)mg->id, (void*)sim, mg->sim_group_idx, mg->vertices.size());
                fflush(stdout);
            }
        }
        // set lasso config via API
        gp_table_meta_set_lasso_config(ctx, mg, &d.lc);
        // set anchor if present (resolved earlier)
        if (mg->anchor_rope >= 0) {
            gp_table_meta_set_anchor(ctx, mg, mg->anchor_rope, mg->anchor_vert);
        }
        // set channel/group
        gp_table_meta_set_channel_group(ctx, mg, d.channel_group);
        // Dump meta-group state after vertices and attachments for diagnostics
        gp_table_debug_dump_meta_group(ctx, mg, "deserialize_post_add");
        
    }

    // If this table blob contained module UUID or frame-port UUIDs, attempt
    // to register them with the canvas so bindings can be reconstructed
    // deterministically. Find the canvas module index for this table.
    if (ctx->module_uuid != 0ull && cvs) {
        // register module UUID with canvas using table pointer (canvas will
        // map table->module_idx internally)
        gp_canvas_register_table_module_uuid(cvs, ctx, ctx->module_uuid);
        // register frame-port UUIDs
        for (const auto &fpe : ctx->frame_port_uuids) {
            gp_canvas_register_table_frame_port_uuid(cvs, ctx, fpe.row, fpe.idx, fpe.uuid);
        }
    }
    return 1;
}

// Relaxation helper: ensure internal arrays match edges size (called before stepping)
static void ensure_relax_vectors(GP_TableContext* ctx) {
    if (!ctx) return;
    size_t n = ctx->edges.size();
    if (ctx->relax_value.size() < n) {
        ctx->relax_value.resize(n, 1.0f);
        ctx->relax_vel.resize(n, 0.0f);
    } else if (ctx->relax_value.size() > n) {
        ctx->relax_value.resize(n);
        ctx->relax_vel.resize(n);
    }
}

// Relaxation control
int32_t gp_table_relax_set_mode(GP_TableContext* ctx, int32_t mode) {
    if (!ctx) return 0;
    ctx->relax_mode = mode;
    if (mode == GP_TABLE_RELAX_WALL_TIME) {
        // set last_time to now
        using namespace std::chrono;
        ctx->relax_last_time = duration<double>(high_resolution_clock::now().time_since_epoch()).count();
    }
    return 1;
}

int32_t gp_table_relax_get_mode(GP_TableContext* ctx, int32_t* out_mode) {
    if (!ctx || !out_mode) return 0;
    *out_mode = ctx->relax_mode;
    return 1;
}

int32_t gp_table_relax_set_params(GP_TableContext* ctx, float stiffness, float damping, float threshold, int32_t max_iters) {
    if (!ctx) return 0;
    ctx->relax_stiffness = stiffness;
    ctx->relax_damping = damping;
    ctx->relax_threshold = threshold;
    ctx->relax_max_iters = max_iters > 0 ? max_iters : 1;
    return 1;
}

int32_t gp_table_relax_step(GP_TableContext* ctx, float dt) {
    if (!ctx) return 0;
    if (dt <= 0.0f) return 0;
    ensure_relax_vectors(ctx);
    float max_change = 0.0f;
    const float stiffness = ctx->relax_stiffness;
    const float damping = ctx->relax_damping;
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        float &v = ctx->relax_value[i];
        float &vel = ctx->relax_vel[i];
        const float target = 1.0f;
        float acc = stiffness * (target - v) - damping * vel;
        vel += acc * dt;
        float dv = vel * dt;
        v += dv;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        max_change = std::max(max_change, std::abs(dv));
    }
    return 1;
}

int32_t gp_table_relax_update(GP_TableContext* ctx) {
    if (!ctx) return 0;
    using namespace std::chrono;
    double now = duration<double>(high_resolution_clock::now().time_since_epoch()).count();
    double last = ctx->relax_last_time;
    if (last <= 0.0) last = now;
    double dt = now - last;
    ctx->relax_last_time = now;
    if (dt <= 0.0) return 0;
    return gp_table_relax_step(ctx, static_cast<float>(dt));
}

int32_t gp_table_relax_run_until_stable(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ensure_relax_vectors(ctx);
    const float threshold = ctx->relax_threshold;
    const int max_iters = ctx->relax_max_iters;
    const float dt = 0.016f; // fixed small step (60Hz)
    for (int it = 0; it < max_iters; ++it) {
        // step and compute max change
        float max_change = 0.0f;
        const float stiffness = ctx->relax_stiffness;
        const float damping = ctx->relax_damping;
        for (size_t i = 0; i < ctx->edges.size(); ++i) {
            float &v = ctx->relax_value[i];
            float &vel = ctx->relax_vel[i];
            const float target = 1.0f;
            float acc = stiffness * (target - v) - damping * vel;
            vel += acc * dt;
            float dv = vel * dt;
            v += dv;
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            max_change = std::max(max_change, std::abs(dv));
        }
        if (max_change <= threshold) return 1;
    }
    return 1;
}

int32_t gp_table_get_scroll_fraction(GP_TableContext* ctx, float* out_frac) {
    if (!ctx || !out_frac) return 0;
    *out_frac = ctx->scroll_frac;
    return 1;
}

int32_t gp_table_get_scroll_fraction_xy(GP_TableContext* ctx, float* out_frac_x, float* out_frac_y) {
    if (!ctx) return 0;
    if (out_frac_x) *out_frac_x = ctx->scroll_frac_x;
    if (out_frac_y) *out_frac_y = ctx->scroll_frac;
    return 1;
}

int32_t gp_table_get_row_count(const GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int32_t>(ctx->rows.size());
}

int32_t gp_table_get_row(const GP_TableContext* ctx, int32_t idx, GP_TableRow* out_row) {
    if (!ctx || !out_row) return 0;
    if (idx < 0 || idx >= static_cast<int32_t>(ctx->rows.size())) return 0;
    *out_row = ctx->rows[static_cast<size_t>(idx)];
    return 1;
}

extern "C" int gp_table_get_rope_id_count(GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int>(ctx->rope_ids.size());
}

extern "C" int gp_table_get_rope_ids(GP_TableContext* ctx, uint64_t* out_ids, int cap) {
    if (!ctx || !out_ids || cap <= 0) return 0;
    int n = std::min(cap, static_cast<int>(ctx->rope_ids.size()));
    for (int i = 0; i < n; ++i) out_ids[i] = ctx->rope_ids[static_cast<size_t>(i)];
    return n;
}

int32_t gp_table_set_led_selected(GP_TableContext* ctx, int32_t row_idx, int32_t col_idx, int32_t led_index, int32_t selected) {
    if (!ctx) return 0;
    if (row_idx < 0 || row_idx >= static_cast<int>(ctx->rows.size())) return 0;
    if (col_idx < 0 || col_idx >= static_cast<int>(ctx->cols.size())) return 0;
    if (led_index < 0 || led_index > 65535) return 0;
    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(row_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(col_idx)) << 16) | static_cast<uint64_t>(static_cast<uint32_t>(led_index));
    if (selected) ctx->selected_leds.insert(key);
    else ctx->selected_leds.erase(key);
    return 1;
}

int32_t gp_table_get_led_selected(const GP_TableContext* ctx, int32_t row_idx, int32_t col_idx, int32_t led_index) {
    if (!ctx) return 0;
    if (row_idx < 0 || row_idx >= static_cast<int>(ctx->rows.size())) return 0;
    if (col_idx < 0 || col_idx >= static_cast<int>(ctx->cols.size())) return 0;
    if (led_index < 0 || led_index > 65535) return 0;
    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(row_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(col_idx)) << 16) | static_cast<uint64_t>(static_cast<uint32_t>(led_index));
    return ctx->selected_leds.find(key) != ctx->selected_leds.end() ? 1 : 0;
}

int32_t gp_table_set_led_glow(GP_TableContext* ctx, int32_t row_idx, int32_t col_idx, int32_t led_index, float glow01) {
    if (!ctx) return 0;
    if (row_idx < 0 || row_idx >= static_cast<int>(ctx->rows.size())) return 0;
    if (col_idx < 0 || col_idx >= static_cast<int>(ctx->cols.size())) return 0;
    if (led_index < 0 || led_index > 65535) return 0;
    float g = std::max(0.0f, std::min(1.0f, glow01));
    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(row_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(col_idx)) << 16) | static_cast<uint64_t>(static_cast<uint32_t>(led_index));
    if (g <= 0.0f) ctx->led_glow_strength.erase(key);
    else ctx->led_glow_strength[key] = g;
    return 1;
}

int32_t gp_table_get_led_glow(const GP_TableContext* ctx, int32_t row_idx, int32_t col_idx, int32_t led_index, float* out_glow01) {
    if (!ctx || !out_glow01) return 0;
    if (row_idx < 0 || row_idx >= static_cast<int>(ctx->rows.size())) return 0;
    if (col_idx < 0 || col_idx >= static_cast<int>(ctx->cols.size())) return 0;
    if (led_index < 0 || led_index > 65535) return 0;
    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(row_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(col_idx)) << 16) | static_cast<uint64_t>(static_cast<uint32_t>(led_index));
    auto it = ctx->led_glow_strength.find(key);
    *out_glow01 = (it != ctx->led_glow_strength.end()) ? it->second : 0.0f;
    return 1;
}

int32_t gp_table_clear_led_glow(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ctx->led_glow_strength.clear();
    return 1;
}

int32_t gp_table_get_led_info(const GP_TableContext* ctx, int32_t row_idx, int32_t col_idx, int32_t led_index,
    int32_t* out_on, int32_t* out_active, int32_t* out_is_input, int32_t* out_is_output) {
    if (!ctx) return 0;
    if (row_idx < 0 || row_idx >= static_cast<int32_t>(ctx->rows.size())) return 0;
    const GP_TableRow &row = ctx->rows[static_cast<size_t>(row_idx)];
    if (col_idx < 0 || col_idx >= row.cell_count) return 0;
    const GP_TableCell &cell = row.cells[col_idx];
    if (cell.kind != GP_TABLE_CELL_LEDS && cell.kind != GP_TABLE_CELL_LEDS_ARG && cell.kind != GP_TABLE_CELL_LEDS_TABLE) return 0;
    int led_count = 9;
    uint32_t on_mask = cell.flags;
    uint32_t active_mask = cell.flags;
    if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
        int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
        if (count == 0) count = (cell.flags & 0xFF);
        if (count == 0) count = 12;
        led_count = count;
        on_mask = cell.flags;
        active_mask = static_cast<uint32_t>(cell.reserved0);
        if (active_mask == 0 && count > 0) active_mask = (count >= 32) ? 0xFFFFFFFFu : ((1u << count) - 1u);
    } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
        led_count = 8;
    }
    if (led_index < 0 || led_index >= led_count) return 0;

    uint32_t bit = 1u << static_cast<uint32_t>(led_index);
    if (out_on) *out_on = (on_mask & bit) != 0;
    if (out_active) *out_active = (active_mask & bit) != 0;

    if (out_is_input || out_is_output) {
        uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(row_idx)) << 32)
            | (static_cast<uint64_t>(static_cast<uint32_t>(col_idx)) << 16)
            | static_cast<uint64_t>(static_cast<uint32_t>(led_index));
        if (out_is_input) *out_is_input = ctx->key_is_input.count(key) != 0;
        if (out_is_output) *out_is_output = ctx->key_is_output.count(key) != 0;
    }

    return 1;
}

int32_t gp_table_get_style(const GP_TableContext* ctx, GP_TableStyle* out_style) {
    if (!ctx || !out_style) return 0;
    *out_style = ctx->style_raw;
    return 1;
}

int32_t gp_table_render_rgba_with_hits(
    GP_TableContext* ctx,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom,
    GP_TableHitBox* hitboxes_out,
    int32_t hitboxes_cap,
    int32_t* hitboxes_written) {
    if (!ctx) return 0;
    return gp_table_raster_rgba_with_hits(
        ctx->rows.data(), static_cast<int32_t>(ctx->rows.size()),
        ctx->cols.data(), static_cast<int32_t>(ctx->cols.size()),
        &ctx->style_raw,
        /*render_state=*/nullptr,
        out_rgba,
        out_len_bytes,
        out_geom,
        hitboxes_out,
        hitboxes_cap,
        hitboxes_written);
}

static void update_led_flow_glow(GP_TableContext* ctx, float dt_frame) {
    if (!ctx) return;
    const size_t edge_count = ctx->edges.size();
    if (ctx->edge_last_write_seq.size() != edge_count) {
        ctx->edge_last_write_seq.assign(edge_count, 0);
        ctx->edge_flow_phase.assign(edge_count, 0.0f);
    }
    ctx->led_flow_glow_strength.clear();
    if (edge_count == 0 || ctx->edge_fifos.empty()) return;
    if (dt_frame <= 0.0f) dt_frame = 1.0f / 60.0f;
    for (size_t ei = 0; ei < edge_count; ++ei) {
        if (ei >= ctx->edge_fifos.size()) break;
        uint64_t seq = ctx->edge_fifos[ei].write_seq();
        uint64_t prev = ctx->edge_last_write_seq[ei];
        ctx->edge_last_write_seq[ei] = seq;
        if (prev == 0 || seq <= prev) continue;
        float rate = static_cast<float>(seq - prev) / dt_frame;
        float freq = std::min(30.0f, rate);
        float amp = std::clamp(rate / 30.0f, 0.0f, 1.0f);
        float phase = ctx->edge_flow_phase[ei];
        phase = std::fmod(phase + 2.0f * kPi * freq * dt_frame, 2.0f * kPi);
        if (phase < 0.0f) phase += 2.0f * kPi;
        ctx->edge_flow_phase[ei] = phase;
        float glow = 0.5f * (std::sin(phase) + 1.0f) * amp;
        if (glow <= 0.0f) continue;
        auto update_glow = [&](uint64_t key) {
            auto it = ctx->led_flow_glow_strength.find(key);
            if (it == ctx->led_flow_glow_strength.end()) ctx->led_flow_glow_strength[key] = glow;
            else it->second = std::max(it->second, glow);
        };
        update_glow(ctx->edges[ei].first);
        update_glow(ctx->edges[ei].second);
    }
}

int32_t gp_table_render_rgba_with_state(
    GP_TableContext* ctx,
    const GP_TableRenderState* render_state,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom,
    GP_TableHitBox* hitboxes_out,
    int32_t hitboxes_cap,
    int32_t* hitboxes_written) {
    if (!ctx) return 0;

    // Build filtered visible rows according to expanded flags and create mapping.
    std::vector<GP_TableRow> vis_rows;
    std::vector<int> map_vis_to_orig; // vis index -> original index
    vis_rows.reserve(ctx->rows.size());
    map_vis_to_orig.reserve(ctx->rows.size());
    const int max_depth = 64;
    std::vector<bool> expanded_at_depth(max_depth, true); // expanded_at_depth[d] controls visibility of depth d+1 children
    for (size_t i = 0; i < ctx->rows.size(); ++i) {
        const GP_TableRow &r = ctx->rows[i];
        int d = std::max(0, r.depth);
        if (d >= max_depth) d = max_depth - 1;
        bool visible = true;
        for (int dd = 0; dd < d; ++dd) {
            if (!expanded_at_depth[dd]) { visible = false; break; }
        }
        if (visible) {
            vis_rows.push_back(r);
            map_vis_to_orig.push_back(static_cast<int>(i));
        }
        // This row controls whether its children (depth d+1) are visible.
        expanded_at_depth[d] = (r.expanded != 0);
        for (int dd = d + 1; dd < max_depth; ++dd) expanded_at_depth[dd] = true;
    }

    // Prepare a local render state with translated highlight_row if present
    GP_TableRenderState local_rs{};
    if (render_state) local_rs = *render_state;
    if (render_state && render_state->highlight_row >= 0) {
        // find mapped index
        int orig = render_state->highlight_row;
        int mapped = -1;
        for (size_t vi = 0; vi < map_vis_to_orig.size(); ++vi) if (map_vis_to_orig[vi] == orig) { mapped = static_cast<int>(vi); break; }
        local_rs.highlight_row = mapped;
    }

    // Call the raster on the visible rows. Honor caller-provided geometry so
    // buffers sized to a clipped viewport still render correctly.
    GP_TableGeom local_geom{};
    GP_TableGeom* geom_ptr = out_geom ? out_geom : &local_geom;
    int ok = gp_table_raster_rgba_with_hits(
        vis_rows.data(), static_cast<int32_t>(vis_rows.size()),
        ctx->cols.data(), static_cast<int32_t>(ctx->cols.size()),
        &ctx->style_raw,
        render_state ? &local_rs : nullptr,
        out_rgba,
        out_len_bytes,
        geom_ptr,
        hitboxes_out,
        hitboxes_cap,
        hitboxes_written);

    if (!ok) return 0;

    // Translate hitboxes back to original row indices so clicks/actions operate on ctx->rows.
    if (hitboxes_out && hitboxes_written && *hitboxes_written > 0) {
        int n = *hitboxes_written;
        for (int i = 0; i < n; ++i) {
            int vr = hitboxes_out[i].row_idx;
            if (vr >= 0 && vr < static_cast<int>(map_vis_to_orig.size())) hitboxes_out[i].row_idx = map_vis_to_orig[static_cast<size_t>(vr)];
        }
    }

    // Prepare column geometry for overlay placement.
    int col_x0[8] = {0};
    int col_w[8] = {0};
    compute_columns(ctx->cols.data(), static_cast<int>(ctx->cols.size()), geom_ptr->width_px, ctx->st.name_w, col_x0, col_w);

    // Cache row layout for the visible row set (for post passes that compute LED centers).
    Style st_local = load_style(&ctx->style_raw);
    st_local.w = std::max(1, geom_ptr->width_px);
    std::vector<int> vis_row_y0;
    std::vector<int> vis_row_h;
    compute_row_layout(vis_rows.data(), static_cast<int>(vis_rows.size()), st_local, geom_ptr->height_px, vis_row_y0, vis_row_h);

    auto find_visible_row = [&](uint32_t orig_idx) -> int {
        for (size_t vi = 0; vi < map_vis_to_orig.size(); ++vi) {
            if (map_vis_to_orig[vi] == static_cast<int>(orig_idx)) return static_cast<int>(vi);
        }
        return -1;
    };

    double dt_frame = 1.0 / 60.0;
    if (out_rgba) {
        using namespace std::chrono;
        double now_frame = duration<double>(high_resolution_clock::now().time_since_epoch()).count();
        double last_frame = ctx->relax_last_time;
        if (last_frame <= 0.0) last_frame = now_frame;
        dt_frame = now_frame - last_frame;
        if (dt_frame <= 0.0) dt_frame = 1.0 / 60.0;
        ctx->relax_last_time = now_frame;
        // GUI-thread flow visualization: normalize flow rate to a sub-30Hz oscillation.
        update_led_flow_glow(ctx, static_cast<float>(dt_frame));
    }

    // Optional LED glow pass driven by per-key strengths and input/output hints.
    if (out_rgba && (!ctx->led_glow_strength.empty() || !ctx->led_flow_glow_strength.empty() || !ctx->key_is_input.empty() || !ctx->key_is_output.empty())) {
        int w_local = geom_ptr->width_px;
        int h_local = geom_ptr->height_px;
        int pitch_local = w_local * 4;

        // Track processed keys so we do not double-apply when a key is both input and output.
        std::unordered_set<uint64_t> seen_keys;
        auto process_key = [&](uint64_t key, float strength){
            if (seen_keys.count(key)) return;
            seen_keys.insert(key);

            uint32_t r_orig = static_cast<uint32_t>(key >> 32);
            uint32_t c_idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
            uint32_t led = static_cast<uint32_t>(key & 0xFFFFu);
            int vis_idx = find_visible_row(r_orig);
            if (vis_idx < 0) return;
            if (c_idx >= ctx->cols.size() || c_idx >= 8) return;
            if (vis_idx >= static_cast<int>(vis_rows.size())) return;
            const GP_TableRow &row = vis_rows[vis_idx];
            if (c_idx >= static_cast<uint32_t>(row.cell_count)) return;
            const GP_TableCell &cell = row.cells[c_idx];
            if (cell.kind != GP_TABLE_CELL_LEDS && cell.kind != GP_TABLE_CELL_LEDS_ARG) return; // stacked strips omitted for now

            int x0 = col_x0[static_cast<int>(c_idx)];
            int cw = col_w[static_cast<int>(c_idx)];
            int y0 = (vis_idx >= 0 && vis_idx < static_cast<int>(vis_row_y0.size())) ? vis_row_y0[static_cast<size_t>(vis_idx)] : (vis_idx * ctx->st.row_h);
            int rh = (vis_idx >= 0 && vis_idx < static_cast<int>(vis_row_h.size())) ? vis_row_h[static_cast<size_t>(vis_idx)] : ctx->st.row_h;
            const bool is_image_row = row_find_image_cell(row, nullptr);
            int band_h = is_image_row ? std::min(rh, row_image_header_h(st_local, row)) : rh;
            int cy = y0 + band_h / 2;
            int count = 9;
            int radius = 4;
            uint32_t on_mask = cell.flags;
            uint32_t active_mask = cell.flags;
            if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
                count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                if (count == 0) count = (cell.flags & 0xFF);
                if (count == 0) count = 12;
                on_mask = cell.flags;
                active_mask = static_cast<uint32_t>(cell.reserved0);
                if (active_mask == 0 && count > 0) active_mask = (count >= 32) ? 0xFFFFFFFFu : ((1u << count) - 1u);
            }
            if (led >= static_cast<uint32_t>(count)) return;

            int eff_w = std::max(1, cw - 4);
            int led_spacing = std::max(radius * 2 + 2, eff_w / std::max(1, count + 1));
            int cx0 = x0 + 2 + led_spacing;
            int cx = cx0 + static_cast<int>(led) * led_spacing;

            uint32_t bit = 1u << led;
            bool on_b = (on_mask & bit) != 0;
            bool active_but_off = !on_b && (active_mask & bit);
            bool is_input = ctx->key_is_input.count(key) != 0;
            bool is_output = ctx->key_is_output.count(key) != 0;

            float glow = std::max(0.0f, std::min(1.0f, strength));
            if (glow <= 0.0f && is_output && on_b) {
                // Outputs light up even without an explicit glow value.
                glow = 0.35f;
            }
            if (!on_b) glow *= 0.3f; // off LEDs only faintly glow unless explicitly boosted
            if (is_input) glow *= 0.35f; // inputs read as dim glass bulbs
            if (is_output && on_b) glow = std::min(1.0f, glow * 1.25f + 0.15f); // outputs pop when on

            // Dead-glass tint for inputs that are off or inactive.
            if (is_input && (!on_b || active_but_off)) {
                Color tint = ctx->st.led_off;
                tint.a = static_cast<uint8_t>(std::lround(180.0f));
                draw_blob_blend(out_rgba, w_local, h_local, pitch_local, cx, cy, radius + 1, tint);
            }

            if (glow > 0.0f) {
                Color glow_col = ctx->st.led_on;
                glow_col.a = std::min<uint8_t>(255, static_cast<uint8_t>(std::lround(220.0f * glow)));
                draw_glow_blob(out_rgba, w_local, h_local, pitch_local, cx, cy, radius, glow, glow_col);
            }
        };

        std::unordered_map<uint64_t, float> combined_glow = ctx->led_glow_strength;
        for (const auto &kv : ctx->led_flow_glow_strength) {
            auto it = combined_glow.find(kv.first);
            if (it == combined_glow.end()) combined_glow[kv.first] = kv.second;
            else it->second = std::max(it->second, kv.second);
        }
        for (const auto &kv : combined_glow) {
            process_key(kv.first, kv.second);
        }
        for (uint64_t key : ctx->key_is_input) {
            process_key(key, 0.0f);
        }
        for (uint64_t key : ctx->key_is_output) {
            process_key(key, 0.0f);
        }
    }

    // Draw selection overlays for any selected LEDs kept in ctx.
    // Selected LEDs are stored as packed keys: (row<<32)|(col<<16)|led
    if (out_rgba && !ctx->selected_leds.empty()) {
        int w_local = local_geom.width_px;
        int h_local = local_geom.height_px;
        int pitch_local = w_local * 4;
        Color sel_col{80, 160, 255, 192};
        // For each selected LED, compute its position in visible rows and draw outline.
        for (uint64_t key : ctx->selected_leds) {
            uint32_t r_orig = static_cast<uint32_t>(key >> 32);
            uint32_t c_idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
            uint32_t led = static_cast<uint32_t>(key & 0xFFFFu);
            // find visible index
            int vis_idx = -1;
            for (size_t vi = 0; vi < map_vis_to_orig.size(); ++vi) if (map_vis_to_orig[vi] == static_cast<int>(r_orig)) { vis_idx = static_cast<int>(vi); break; }
            if (vis_idx < 0) continue;
            const GP_TableRow &row = vis_rows[vis_idx];
            if (c_idx < 0 || c_idx >= static_cast<uint32_t>(row.cell_count)) continue;
            const GP_TableCell &cell = row.cells[c_idx];
            int x0 = col_x0[static_cast<int>(c_idx)];
            int cw = col_w[static_cast<int>(c_idx)];
            int y0 = (vis_idx >= 0 && vis_idx < static_cast<int>(vis_row_y0.size())) ? vis_row_y0[static_cast<size_t>(vis_idx)] : (vis_idx * ctx->st.row_h);
            int rh = (vis_idx >= 0 && vis_idx < static_cast<int>(vis_row_h.size())) ? vis_row_h[static_cast<size_t>(vis_idx)] : ctx->st.row_h;
            const bool is_image_row = row_find_image_cell(row, nullptr);
            int band_h = is_image_row ? std::min(rh, row_image_header_h(st_local, row)) : rh;
            // approximate LED layout similar to raster
            int led_count = 9;
            if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
                int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                if (count == 0) count = (cell.flags & 0xFF);
                if (count == 0) count = 12;
                led_count = count;
            } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
                led_count = 8;
            }
            int eff_w = std::max(1, cw - 4);
            int radius = 4;
            int led_spacing = std::max(radius * 2 + 2, eff_w / std::max(1, led_count + 1));
            int cx0 = x0 + 2 + led_spacing;
            if (static_cast<int>(led) >= 0 && static_cast<int>(led) < led_count) {
                int cx = cx0 + static_cast<int>(led) * led_spacing;
                int cy = y0 + band_h / 2;
                draw_circle_outline(out_rgba, w_local, h_local, pitch_local, cx, cy, radius + 2, 2, sel_col);
            }
        }
    }

    struct LedContactInfo {
        int x = -1;
        int y = -1;
        bool on = false;
        bool active = false;
        bool is_input = false;
        bool is_output = false;
    };

    auto fetch_led_contact = [&](uint64_t key, LedContactInfo& out) -> bool {
        out = {};
        out.is_input = ctx->key_is_input.count(key) != 0;
        out.is_output = ctx->key_is_output.count(key) != 0;
        uint32_t r_orig = static_cast<uint32_t>(key >> 32);
        uint32_t c_idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
        uint32_t led = static_cast<uint32_t>(key & 0xFFFFu);
        // First: allow canonical root keys mapped to overlays to resolve to overlay coords
        GP_CanvasContext* canvas = gp_canvas_get_singleton();
        if (canvas) {
            int ox = -1, oy = -1;
            if (gp_canvas_resolve_canonical_key(canvas, key, &ox, &oy)) {
                out.x = ox; out.y = oy; out.is_input = out.is_output = false;
                return true;
            }
        }
        // Overlay sentinel keys: module id 0xFFFFFFFF indicates a canvas overlay
        if (r_orig == 0xFFFFFFFFu) {
            int ox = -1, oy = -1;
            if (!canvas) return false;
            if (!gp_canvas_resolve_overlay_key(canvas, key, &ox, &oy)) return false;
            out.x = ox; out.y = oy; out.is_input = out.is_output = false;
            return true;
        }
        int vis_idx = find_visible_row(r_orig);
        if (vis_idx < 0 || vis_idx >= static_cast<int>(vis_rows.size())) return false;
        if (c_idx >= ctx->cols.size() || c_idx >= 8) return false;
        const GP_TableRow &row = vis_rows[vis_idx];
        if (c_idx >= static_cast<uint32_t>(row.cell_count)) return false;
        const GP_TableCell &cell = row.cells[c_idx];
        int x0 = col_x0[static_cast<int>(c_idx)];
        int cw = col_w[static_cast<int>(c_idx)];
        int y0 = (vis_idx >= 0 && vis_idx < static_cast<int>(vis_row_y0.size())) ? vis_row_y0[static_cast<size_t>(vis_idx)] : (vis_idx * ctx->st.row_h);
        int rh = (vis_idx >= 0 && vis_idx < static_cast<int>(vis_row_h.size())) ? vis_row_h[static_cast<size_t>(vis_idx)] : ctx->st.row_h;
        const bool is_image_row = row_find_image_cell(row, nullptr);
        int band_h = is_image_row ? std::min(rh, row_image_header_h(st_local, row)) : rh;
        int led_count = 9;
        uint32_t on_mask = cell.flags;
        uint32_t active_mask = cell.flags;
        if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
            int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
            if (count == 0) count = (cell.flags & 0xFF);
            if (count == 0) count = 12;
            led_count = count;
            on_mask = cell.flags;
            active_mask = static_cast<uint32_t>(cell.reserved0);
            if (active_mask == 0 && count > 0) active_mask = (count >= 32) ? 0xFFFFFFFFu : ((1u << count) - 1u);
        } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
            led_count = 8;
        }
        int eff_w = std::max(1, cw - 4);
        int radius = 4;
        int led_spacing = std::max(radius * 2 + 2, eff_w / std::max(1, led_count + 1));
        int cx0 = x0 + 2 + led_spacing;
        if (led >= static_cast<uint32_t>(led_count)) return false;
        out.x = cx0 + static_cast<int>(led) * led_spacing;
        out.y = y0 + band_h / 2;
        uint32_t bit = 1u << led;
        out.on = (on_mask & bit) != 0;
        out.active = (active_mask & bit) != 0;
        return true;
    };

    auto compute_led_glow = [&](const LedContactInfo& info, uint64_t key) -> float {
        bool lit = info.on || info.active;
        float glow = 0.0f;
        auto it = ctx->led_glow_strength.find(key);
        if (it != ctx->led_glow_strength.end()) glow = std::max(glow, std::clamp(it->second, 0.0f, 1.0f));
        auto flow_it = ctx->led_flow_glow_strength.find(key);
        if (flow_it != ctx->led_flow_glow_strength.end()) glow = std::max(glow, std::clamp(flow_it->second, 0.0f, 1.0f));
        if (glow <= 0.0f && info.is_output && lit) {
            glow = 0.35f;
        }
        if (!lit) glow *= 0.3f;
        if (info.is_input) glow *= 0.35f;
        if (info.is_output && lit) glow = std::min(1.0f, glow * 1.25f + 0.15f);
        return glow;
    };


    // Draw edges between LED centers
    // Prospective live-edge: if enabled and exactly one LED selected, we will
    // relax a free end of the cord toward the supplied mouse position and
    // render a live prospective edge. This updates small state in the context
    // so behavior is smooth across frames.
    if (out_rgba) {
        // step relax values by the frame dt (keeps animation in render loop)
        gp_table_relax_step(ctx, static_cast<float>(dt_frame));
        // attempt prospective update/draw if enabled
        if (ctx->prospective_mode && ctx->selected_leds.size() == 1) {
            if (render_state && render_state->mouse_x >= 0 && render_state->mouse_y >= 0) {
                // compute selected node center
                uint64_t sel_key = *ctx->selected_leds.begin();
                LedContactInfo sel_info;
                bool have_sel = fetch_led_contact(sel_key, sel_info);
                int sx = have_sel ? sel_info.x : -1;
                int sy = have_sel ? sel_info.y : -1;
                float sel_glow = have_sel ? compute_led_glow(sel_info, sel_key) : 0.0f;
                if (sx >= 0 && sy >= 0) {
                    // compute local image dims/pitch for drawing (use distinct names)
                    int p_w = local_geom.width_px;
                    int p_h = local_geom.height_px;
                    int p_pitch = p_w * 4;
                        // target is mouse in table-local coords; push into queue
                        float tx = static_cast<float>(render_state->mouse_x);
                        float ty = static_cast<float>(render_state->mouse_y);
                        ctx->prospective_targets.emplace_back(tx, ty);
                        // drop oldest if history exceeds max
                        if (static_cast<int>(ctx->prospective_targets.size()) > ctx->prospective_max_history) {
                            int drop = static_cast<int>(ctx->prospective_targets.size()) - ctx->prospective_max_history;
                            ctx->prospective_targets.erase(ctx->prospective_targets.begin(), ctx->prospective_targets.begin() + drop);
                        }
                        // effective target is newest queued target (latest mouse position)
                        float eff_tx = tx;
                        float eff_ty = ty;
                        if (!ctx->prospective_targets.empty()) {
                            eff_tx = ctx->prospective_targets.back().first;
                            eff_ty = ctx->prospective_targets.back().second;
                        }
                    // initialize prospect pos if needed
                    if (!ctx->prospective_initialized) {
                        ctx->prospective_x = static_cast<float>(sx);
                        ctx->prospective_y = static_cast<float>(sy);
                        ctx->prospective_vx = 0.0f;
                        ctx->prospective_vy = 0.0f;
                        ctx->prospective_initialized = true;
                        // reset time base
                        using namespace std::chrono;
                        ctx->relax_last_time = duration<double>(high_resolution_clock::now().time_since_epoch()).count();
                    }
                    // Immediate snap: follow newest mouse position exactly for responsiveness
                    float &px = ctx->prospective_x;
                    float &py = ctx->prospective_y;
                    float &vx = ctx->prospective_vx;
                    float &vy = ctx->prospective_vy;
                    px = eff_tx;
                    py = eff_ty;
                    // zero motion state so future relax/springs start from rest
                    vx = 0.0f;
                    vy = 0.0f;
                    // update prospective rope endpoints in simulator so last vertex equals mouse immediately
                    int ipx = static_cast<int>(std::lround(px));
                    int ipy = static_cast<int>(std::lround(py));
                    if (!ctx->rope_sim) {
                        int max_ropes = 16;
                        int max_segs = std::max(4, ctx->st.cable_segments);
                        ctx->rope_sim = rope_sim_create(max_ropes, max_segs);
                        ctx->rope_id_to_sim_idx.clear();
                    }
                    float plug_z = -ctx->st.cable_plug_depth;
                    // create a single persistent prospective rope if not present
                    if (ctx->prospective_rope_idx < 0) {
                        int segs = std::max(4, ctx->st.cable_segments);
                        float slack = ctx->prospective_rope_length > 0.0f ? ctx->prospective_rope_length : 0.0f;
                        ctx->prospective_rope_idx = rope_sim_add_rope3(ctx->rope_sim, static_cast<float>(sx), static_cast<float>(sy), plug_z, static_cast<float>(ipx), static_cast<float>(ipy), plug_z, segs, slack);
                    } else {
                        // move endpoints preserving previous positions so integrator receives velocity impulse
                        rope_sim_move_endpoints3(ctx->rope_sim, ctx->prospective_rope_idx, static_cast<float>(sx), static_cast<float>(sy), plug_z, static_cast<float>(ipx), static_cast<float>(ipy), plug_z);
                    }
                    // consume the newest queued target when close enough
                    float ddx = eff_tx - px;
                    float ddy = eff_ty - py;
                    float d2 = ddx*ddx + ddy*ddy;
                    if (d2 <= ctx->prospective_slack * ctx->prospective_slack) ctx->prospective_targets.clear();
                    // if no permanent edges will step the sim later, step now so prospective rope animates
                    if (ctx->edges.empty() && table_should_step_sim(ctx)) {
                        // freer whipping (lower damping) but stronger constraint solve so it settles quickly
                        float gravity = 800.0f;
                        int constraint_iters = 8;
                        float damping = 0.86f;
                        rope_sim_step(ctx->rope_sim, static_cast<float>(dt_frame), gravity, constraint_iters, damping);
                    }
                    // draw prospective rope from sim vertices
                    if (ctx->prospective_rope_idx >= 0) {
                        int vc = rope_sim_get_vertex_count(ctx->rope_sim, ctx->prospective_rope_idx);
                        if (vc >= 2) {
                            std::vector<float> verts3(static_cast<size_t>(vc) * 3);
                            int got = rope_sim_get_vertices3(ctx->rope_sim, ctx->prospective_rope_idx, verts3.data(), static_cast<int>(verts3.size()));
                            if (got > 0) {
                                std::vector<float> proj_xy;
                                std::vector<float> proj_z;
                                float min_z = 0.0f, max_z = 0.0f;
                                project_rope_vertices_ortho(verts3.data(), got, ctx->st.cable_tilt_x, ctx->st.cable_tilt_y, proj_xy, proj_z, min_z, max_z);

                                // choose colored or neutral rope draw depending on selected LED output
                                int samples_per_segment = std::max(2, ctx->st.cable_segments / std::max(1, got - 1));
                                if (sel_info.is_output && sel_glow > 0.0f) {
                                    float hue = rgb_to_hue(ctx->st.led_on);
                                    std::vector<float> hues_local = { hue };
                                    draw_rope_curve_blend_colored(out_rgba, p_w, p_h, p_pitch, proj_xy.data(), got, ctx->st.cable_jacket_px, ctx->st.cable_jacket_border, hues_local.data(), static_cast<int>(hues_local.size()), samples_per_segment, sel_glow);
                                } else {
                                    Color pcol{200,200,200, static_cast<uint8_t>(std::lround(180.0f))};
                                    draw_rope_curve_blend(out_rgba, p_w, p_h, p_pitch, proj_xy.data(), got, ctx->st.cable_jacket_px, ctx->st.cable_jacket_border, pcol, samples_per_segment);
                                }

                                std::vector<std::pair<float,float>> fiber_samples;
                                std::vector<float> depth_samples;
                                build_rope_samples_with_depth(proj_xy.data(), proj_z.data(), got, ctx->st.cable_jacket_px, fiber_samples, &depth_samples);
                                if (!fiber_samples.empty()) {
                                    int fiber_r = std::max(1, static_cast<int>(std::lround(float(ctx->st.cable_jacket_px) * ctx->st.cable_fiber_radius_scale)));
                                    // fiber pass removed: jacket/core will be tinted directly by SDF
                                        // edge sliver drawing removed — rely on parametric SDF jacket/core and glow
                                    if (sel_info.is_output && sel_glow > 0.0f) {
                                        Color glow_col = ctx->st.led_on;
                                        // moderate glow source alpha so tube remains translucent
                                        glow_col.a = static_cast<uint8_t>(std::lround(float(glow_col.a) * 0.85f));
                                        int glow_r = fiber_r;
                                        // increase multiplier so more of the source glow transmits
                                        draw_rope_diffused_glow(out_rgba, p_w, p_h, p_pitch, fiber_samples, depth_samples, true, sel_glow, ctx->st.cable_depth_fade, ctx->st.cable_fiber_gain * 0.8f, glow_col, nullptr, min_z, max_z, glow_r);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (out_rgba && !ctx->edges.empty()) {
        ensure_edge_fifos(ctx);
        int w_local = local_geom.width_px;
        int h_local = local_geom.height_px;
        int pitch_local = w_local * 4;
        std::vector<LedContactInfo> edge_contact_a(ctx->edges.size());
        std::vector<LedContactInfo> edge_contact_b(ctx->edges.size());
        std::vector<float> edge_glow_a(ctx->edges.size(), 0.0f);
        std::vector<float> edge_glow_b(ctx->edges.size(), 0.0f);

        // ensure rope simulator exists
        if (!ctx->rope_sim) {
            int max_ropes = std::max<int>(1024, static_cast<int>(ctx->edges.size()) + 16);
            int max_segs = std::max(4, ctx->st.cable_segments);
            ctx->rope_sim = rope_sim_create(max_ropes, max_segs);
            ctx->rope_id_to_sim_idx.clear();
        }

        // ensure uid->sim mapping is present for any existing mapped ropes
        // update endpoints in sim (and create ropes if missing)
        for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
            const auto &e = ctx->edges[ei];
            uint64_t ka = e.first;
            uint64_t kb = e.second;
            LedContactInfo info_a;
            LedContactInfo info_b;
            if (!fetch_led_contact(ka, info_a) || !fetch_led_contact(kb, info_b)) continue;
            int ax = info_a.x;
            int ay = info_a.y;
            int bx = info_b.x;
            int by = info_b.y;
            float glow_a = compute_led_glow(info_a, ka);
            float glow_b = compute_led_glow(info_b, kb);
            edge_contact_a[ei] = info_a;
            edge_contact_b[ei] = info_b;
            edge_glow_a[ei] = glow_a;
            edge_glow_b[ei] = glow_b;

            // resolve sim index using persistent id mapping
            uint64_t uid = (ei < ctx->rope_ids.size()) ? ctx->rope_ids[ei] : 0ull;
            int rope_idx = -1;
            if (uid != 0ull) {
                auto it = ctx->rope_id_to_sim_idx.find(uid);
                if (it != ctx->rope_id_to_sim_idx.end()) rope_idx = it->second;
            }
            if (rope_idx < 0) {
                int segs = std::max(4, ctx->st.cable_segments);
                float slack = 0.0f;
                float plug_z = -ctx->st.cable_plug_depth;
                int new_idx = rope_sim_add_rope3(ctx->rope_sim, static_cast<float>(ax), static_cast<float>(ay), plug_z, static_cast<float>(bx), static_cast<float>(by), plug_z, segs, slack);
                if (uid == 0ull) {
                        // assign persistent id for this rope if not already present
                        uid = gp_canvas_generate_id(gp_canvas_get_singleton(), 0ull);
                        if (ei < ctx->rope_ids.size()) ctx->rope_ids[ei] = uid;
                        else ctx->rope_ids.push_back(uid);
                    }
                ctx->rope_id_to_sim_idx[uid] = new_idx;
                rope_idx = new_idx;
            } else {
                float plug_z = -ctx->st.cable_plug_depth;
                rope_sim_move_endpoints3(ctx->rope_sim, rope_idx, static_cast<float>(ax), static_cast<float>(ay), plug_z, static_cast<float>(bx), static_cast<float>(by), plug_z);
            }
        }

        // no-op here: dangling ropes should be integrated into meta-group
        // ordering so the sim connects them via edge-springs. Avoid moving
        // endpoints at render-time; use insertion into the sim meta-group
        // to make the short rope participate in ring physics.

        // Step the simulator for this frame: allow more whip (lower damping) but more constraint iterations to settle
        // default sim step (used by FIFO timing even when sim disabled)
        float sim_dt = 1.0f / 60.0f;
        if (table_should_step_sim(ctx)) {
            float gravity = 800.0f;
            int constraint_iters = 8;
            float damping = 0.86f;
            // use sim_dt (could be overridden in future if frame dt available)
            rope_sim_step(ctx->rope_sim, sim_dt, gravity, constraint_iters, damping);
        }

        // Now render ropes from simulator vertices
        for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
            const LedContactInfo &info_a = edge_contact_a[ei];
            const LedContactInfo &info_b = edge_contact_b[ei];
            if (info_a.x < 0 || info_b.x < 0) continue;
            float glow_a = edge_glow_a[ei];
            float glow_b = edge_glow_b[ei];
            float ev = 1.0f;
            if (ei < ctx->relax_value.size()) ev = ctx->relax_value[ei];
            Color rope_light_col = ctx->st.led_on;
            if (ctx->st.cable_fifo_light_mode) {
                rope_light_col = ctx->st.led_edge;
                rope_light_col.a = 255;
            }
            float glow_a_eff = ctx->st.cable_fifo_light_mode ? 0.0f : (glow_a * ev);
            float glow_b_eff = ctx->st.cable_fifo_light_mode ? 0.0f : (glow_b * ev);
            float fifo_write_glow = 0.0f;
            float fifo_read_glow = 0.0f;
            float fifo_tint = 0.0f;
            float fifo_phase_delta = 0.0f;
            float fifo_fill = 0.0f;
            float fifo_head_phase = 0.0f;
            float fifo_tail_phase = 0.0f;
            float fifo_write_phase = 0.0f;
            float fifo_read_phase = 0.0f;
            bool fifo_has_state = false;
            float fifo_core_intensity = 0.0f;
            if (ctx->st.cable_fifo_light_mode && ei < ctx->edge_fifos.size()) {
                auto &fifo = ctx->edge_fifos[ei];
                fifo.tick_friction(sim_dt, ctx->st.cable_fifo_friction_half_life);
                float wf = fifo.write_friction();
                float rf = fifo.read_friction();
                float gain = ctx->st.cable_fifo_friction_gain;
                float mag = std::sqrt(wf * wf + rf * rf);
                fifo_write_glow = std::min(1.0f, wf * gain);
                fifo_read_glow = std::min(1.0f, rf * gain);
                float fifo_write_glow_eff = std::clamp(fifo_write_glow * ev, 0.0f, 1.0f);
                float fifo_read_glow_eff = std::clamp(fifo_read_glow * ev, 0.0f, 1.0f);
                float fifo_mag = std::min(1.0f, mag * gain);
                fifo_phase_delta = fifo.phase_delta();
                float theta = std::abs(fifo_phase_delta) * 2.0f * kPi;
                fifo_tint = std::clamp(theta / kPi, 0.0f, 1.0f) * ctx->st.cable_fifo_friction_tint;
                fifo_write_phase = fifo.write_phase();
                fifo_read_phase = fifo.read_phase();
                uint64_t edge_id = (ei < ctx->edge_ids.size()) ? ctx->edge_ids[ei] : 0ull;
                fifo_has_state = fifo.fill_state(edge_id, fifo_fill, fifo_head_phase, fifo_tail_phase);
                float fifo_fill_glow = fifo_has_state ? std::clamp(fifo_fill * ev, 0.0f, 1.0f) : 0.0f;
                fifo_core_intensity = std::max({fifo_fill_glow, fifo_write_glow_eff, fifo_read_glow_eff});
                if (fifo_mag > 0.0f) {
                    if (info_a.is_output && info_b.is_input) {
                        glow_a_eff = std::max(glow_a_eff, fifo_write_glow_eff);
                        glow_b_eff = std::max(glow_b_eff, fifo_read_glow_eff);
                    } else if (info_b.is_output && info_a.is_input) {
                        glow_b_eff = std::max(glow_b_eff, fifo_write_glow_eff);
                        glow_a_eff = std::max(glow_a_eff, fifo_read_glow_eff);
                    } else {
                        glow_a_eff = std::max(glow_a_eff, fifo_mag * ev);
                        glow_b_eff = std::max(glow_b_eff, fifo_mag * ev);
                    }
                }
                if (fifo_fill_glow > 0.0f) {
                    glow_a_eff = std::max(glow_a_eff, fifo_fill_glow);
                    glow_b_eff = std::max(glow_b_eff, fifo_fill_glow);
                }
            }
            bool reverse_phases = info_b.is_output && !info_a.is_output;
            // resolve sim index from persistent rope id mapping
            uint64_t rope_id = (ei < ctx->rope_ids.size()) ? ctx->rope_ids[ei] : 0ull;
            int rope_idx = -1;
            if (rope_id != 0ull) {
                auto it = ctx->rope_id_to_sim_idx.find(rope_id);
                if (it != ctx->rope_id_to_sim_idx.end()) rope_idx = it->second;
            }
            if (rope_idx < 0) continue;
            int vc = rope_sim_get_vertex_count(ctx->rope_sim, rope_idx);
            if (getenv("NODUS_DEBUG_EDGE") != nullptr) {
                uint32_t flags = 0u;
                if (ei < ctx->edge_subgroup_flags.size()) flags = ctx->edge_subgroup_flags[ei];
                int colored = (flags != 0u) ? 1 : 0;
                printf("[EDGE_DEBUG] ei=%llu rope=%d vc=%d subgroup=0x%08x colored=%d fifo_has_state=%d fifo_core=%.2f\n",
                    static_cast<unsigned long long>(ei), rope_idx, vc, flags, colored, fifo_has_state ? 1 : 0, fifo_core_intensity);
                fflush(stdout);
            }
            if (vc < 2) continue;
            std::vector<float> verts3(static_cast<size_t>(vc) * 3);
            int got = rope_sim_get_vertices3(ctx->rope_sim, rope_idx, verts3.data(), static_cast<int>(verts3.size()));
            if (got <= 0) continue;
            std::vector<float> proj_xy;
            std::vector<float> proj_z;
            float min_z = 0.0f, max_z = 0.0f;
            project_rope_vertices_ortho(verts3.data(), got, ctx->st.cable_tilt_x, ctx->st.cable_tilt_y, proj_xy, proj_z, min_z, max_z);
            // draw smooth spline curve along simulator vertices; overlay LED light falloff
            float dominant_glow = std::max(glow_a_eff, glow_b_eff);
            std::vector<std::pair<float,float>> jacket_samples;
            std::vector<std::pair<float,float>> jacket_tangents;
            build_rope_samples_with_tangents(proj_xy.data(), got, ctx->st.cable_jacket_px, jacket_samples, jacket_tangents);
            int jacket_segments = std::max(1, got - 1);
            std::vector<Color> base_jacket_colors(static_cast<size_t>(jacket_segments), Color{200, 200, 200, 13});
            std::vector<Color> fifo_jacket_colors;
            bool has_fifo_jacket = false;
            if (ctx->st.cable_fifo_light_mode && fifo_has_state) {
                int fifo_segments = jacket_segments;
                fifo_jacket_colors.assign(static_cast<size_t>(fifo_segments), Color{0, 0, 0, 0});
                auto normalize_phase = [&](float p) {
                    float v = p - std::floor(p);
                    if (v < 0.0f) v += 1.0f;
                    return v;
                };
                auto orient_phase = [&](float p) {
                    float v = normalize_phase(p);
                    if (reverse_phases) {
                        v = 1.0f - v;
                        if (v >= 1.0f) v = 0.0f;
                    }
                    return v;
                };
                auto phase_in_arc = [&](float phase, float tail, float head) {
                    if (head == tail) return false;
                    if (head > tail) return (phase >= tail && phase < head);
                    return (phase >= tail || phase < head);
                };
                auto ring_dist = [&](int a, int b, int n) {
                    int d = std::abs(a - b);
                    return std::min(d, n - d);
                };
                float fill_strength = std::clamp(fifo_fill * ev, 0.0f, 1.0f);
                float write_strength = std::clamp(fifo_write_glow * ev, 0.0f, 1.0f);
                float read_strength = std::clamp(fifo_read_glow * ev, 0.0f, 1.0f);
                float head_phase = orient_phase(fifo_head_phase);
                float tail_phase = orient_phase(fifo_tail_phase);
                float write_phase = orient_phase(fifo_write_phase);
                float read_phase = orient_phase(fifo_read_phase);
                int write_idx = std::clamp(static_cast<int>(std::floor(write_phase * fifo_segments)), 0, fifo_segments - 1);
                int read_idx = std::clamp(static_cast<int>(std::floor(read_phase * fifo_segments)), 0, fifo_segments - 1);
                Color fill_col{40, 120, 255, 255};
                Color write_col{255, 80, 80, 255};
                Color read_col{80, 255, 120, 255};
                for (int si = 0; si < fifo_segments; ++si) {
                    float seg_phase = (static_cast<float>(si) + 0.5f) / static_cast<float>(fifo_segments);
                    float fill_local = (fill_strength > 0.0f && phase_in_arc(seg_phase, tail_phase, head_phase)) ? fill_strength : 0.0f;
                    float write_local = 0.0f;
                    float read_local = 0.0f;
                    if (write_strength > 0.0f) {
                        int dist = ring_dist(si, write_idx, fifo_segments);
                        write_local = write_strength * std::exp(-0.9f * static_cast<float>(dist));
                    }
                    if (read_strength > 0.0f) {
                        int dist = ring_dist(si, read_idx, fifo_segments);
                        read_local = read_strength * std::exp(-0.9f * static_cast<float>(dist));
                    }
                    Color out{0, 0, 0, 0};
                    auto apply_tint = [&](Color tint, float strength, uint8_t alpha_max) {
                        if (strength <= 0.0f) return;
                        Color t = tint;
                        t.a = static_cast<uint8_t>(std::lround(alpha_max * std::clamp(strength, 0.0f, 1.0f)));
                        if (out.a == 0) {
                            out = t;
                            return;
                        }
                        out = lerp_color(out, t, strength);
                        out.a = std::max(out.a, t.a);
                    };
                    apply_tint(fill_col, fill_local, 170);
                    apply_tint(write_col, write_local, 200);
                    apply_tint(read_col, read_local, 200);
                    fifo_jacket_colors[static_cast<size_t>(si)] = out;
                }
                has_fifo_jacket = true;
            }
            if (dominant_glow > 0.0f) {
                bool lit_a = info_a.on || info_a.active;
                bool lit_b = info_b.on || info_b.active;
                Color led_a = lit_a ? ctx->st.led_on : ctx->st.led_off;
                Color led_b = lit_b ? ctx->st.led_on : ctx->st.led_off;
                if (ctx->st.cable_fifo_light_mode) {
                    led_a = rope_light_col;
                    led_b = rope_light_col;
                }
                if (ctx->st.cable_fifo_light_mode && fifo_tint > 0.0f) {
                    float hue_shift = (fifo_phase_delta >= 0.0f) ? fifo_tint : -fifo_tint;
                    led_a = tint_color_hue(led_a, hue_shift, fifo_tint);
                    led_b = tint_color_hue(led_b, hue_shift, fifo_tint);
                }
                float decay = 2.0f;
                draw_rope_curve_blend_rgb_falloff(out_rgba, w_local, h_local, pitch_local, proj_xy.data(), got, ctx->st.cable_jacket_px, ctx->st.cable_jacket_border, led_a, led_b, glow_a_eff, glow_b_eff, decay);
            }

            // Fiber-optic overlay using LED on color as tint, masked by rope texture.
            std::vector<std::pair<float,float>> fiber_samples;
            std::vector<float> depth_samples;
            build_rope_samples_with_depth(proj_xy.data(), proj_z.data(), got, ctx->st.cable_jacket_px, fiber_samples, &depth_samples);
                if (!fiber_samples.empty()) {
                int fiber_r = std::max(1, static_cast<int>(std::lround(float(ctx->st.cable_jacket_px) * ctx->st.cable_fiber_radius_scale * 0.95f)));
                // fiber pass removed: jacket/core will be tinted directly by SDF; keep fiber_r for glow radius
                bool lit_a = info_a.on || info_a.active;
                bool lit_b = info_b.on || info_b.active;
                Color glow_col_a = lit_a ? ctx->st.led_on : ctx->st.led_off;
                Color glow_col_b = lit_b ? ctx->st.led_on : ctx->st.led_off;
                glow_col_a.a = static_cast<uint8_t>(std::lround(float(glow_col_a.a) * 0.6f));
                glow_col_b.a = static_cast<uint8_t>(std::lround(float(glow_col_b.a) * 0.6f));
                // keep glow multiplier subtle so it doesn't overpower jacket/core
                float glow_gain = ctx->st.cable_fiber_gain * 0.85f;
                std::vector<Color> glow_jacket_colors = base_jacket_colors;
                if (has_fifo_jacket && fifo_jacket_colors.size() == glow_jacket_colors.size()) {
                    for (size_t si = 0; si < glow_jacket_colors.size(); ++si) {
                        if (fifo_jacket_colors[si].a == 0) continue;
                        glow_jacket_colors[si] = lerp_color(glow_jacket_colors[si], fifo_jacket_colors[si], 0.7f);
                        glow_jacket_colors[si].a = std::max(glow_jacket_colors[si].a, fifo_jacket_colors[si].a);
                    }
                }
                if (info_a.is_output && glow_a_eff > 0.0f) {
                    float transmitted = draw_rope_diffused_glow(out_rgba, w_local, h_local, pitch_local, fiber_samples, depth_samples, true, glow_a_eff, ctx->st.cable_depth_fade, glow_gain, glow_col_a, &glow_jacket_colors, min_z, max_z, fiber_r);
                    if (info_b.is_input && transmitted > 0.0f) {
                        Color input_col = glow_col_a;
                        float boost = std::min(1.0f, transmitted * 0.6f);
                        input_col.a = static_cast<uint8_t>(std::lround(float(input_col.a) * (0.35f + 0.4f * boost)));
                        float gx0 = static_cast<float>(info_b.x) - 0.5f;
                        float gy0 = static_cast<float>(info_b.y);
                        float gx1 = static_cast<float>(info_b.x) + 0.5f;
                        float gy1 = static_cast<float>(info_b.y);
                        draw_segment_kernel_glow(out_rgba, w_local, h_local, pitch_local, gx0, gy0, gx1, gy1, 5.0f, input_col, boost);
                    }
                }
                if (info_b.is_output && glow_b_eff > 0.0f) {
                    float transmitted = draw_rope_diffused_glow(out_rgba, w_local, h_local, pitch_local, fiber_samples, depth_samples, false, glow_b_eff, ctx->st.cable_depth_fade, glow_gain, glow_col_b, &glow_jacket_colors, min_z, max_z, fiber_r);
                    if (info_a.is_input && transmitted > 0.0f) {
                        Color input_col = glow_col_b;
                        float boost = std::min(1.0f, transmitted * 0.6f);
                        input_col.a = static_cast<uint8_t>(std::lround(float(input_col.a) * (0.35f + 0.4f * boost)));
                        float gx0 = static_cast<float>(info_a.x) - 0.5f;
                        float gy0 = static_cast<float>(info_a.y);
                        float gx1 = static_cast<float>(info_a.x) + 0.5f;
                        float gy1 = static_cast<float>(info_a.y);
                        draw_segment_kernel_glow(out_rgba, w_local, h_local, pitch_local, gx0, gy0, gx1, gy1, 5.0f, input_col, boost);
                    }
                        // edge sliver removed - parametric SDF and glow provide saturation
                }
            }
            draw_rope_jacket_overlay(out_rgba, w_local, h_local, pitch_local, jacket_samples, jacket_tangents, ctx->st.cable_jacket_px, base_jacket_colors);
            auto make_led_color = [&](const LedContactInfo &info, float glow) {
                Color c = (info.on || info.active) ? ctx->st.led_on : ctx->st.led_off;
                c.a = static_cast<uint8_t>(std::lround(255.0f * std::clamp(glow, 0.0f, 1.0f)));
                return c;
            };
            Color core_a = make_led_color(info_a, glow_a_eff);
            Color core_b = make_led_color(info_b, glow_b_eff);
            float core_intensity = (ctx->st.cable_fifo_light_mode ? std::clamp(fifo_core_intensity, 0.0f, 1.0f) : ev)
                                   * std::clamp(ctx->st.cable_core_alpha, 0.0f, 1.0f);
            draw_rope_core_gradient(out_rgba, w_local, h_local, pitch_local, jacket_samples, jacket_tangents, ctx->st.cable_jacket_px, ctx->st.cable_jacket_border, core_a, core_b, core_intensity);
            if (has_fifo_jacket) {
                draw_rope_jacket_overlay(out_rgba, w_local, h_local, pitch_local, jacket_samples, jacket_tangents, ctx->st.cable_jacket_px, fifo_jacket_colors);
            }
        }
    }
    // Remap hitboxes' row indices from visible index -> original index
    if (hitboxes_out && hitboxes_cap > 0 && map_vis_to_orig.size() > 0 && hitboxes_written && *hitboxes_written > 0) {
        for (int hi = 0; hi < *hitboxes_written; ++hi) {
            int vis_idx = hitboxes_out[hi].row_idx;
            if (vis_idx >= 0 && vis_idx < static_cast<int>(map_vis_to_orig.size())) {
                hitboxes_out[hi].row_idx = map_vis_to_orig[vis_idx];
            }
        }
    }

    // Populate out_geom in original coordinate system (width/height same)
    if (out_geom) {
        *out_geom = local_geom;
    }

    return 1;
}

} // extern "C"

#ifdef __cplusplus
// C++ container helpers for table_abi (no new files). These mirror the C structs but provide
// builder-style APIs. They do not change raster behavior.
namespace gp {

struct RenderedTable {
    int width_px = 0;
    int height_px = 0;
    GP_TableGeom geom{};
    std::vector<uint8_t> rgba; // width*height*4
    bool ok() const { return width_px > 0 && height_px > 0 && rgba.size() == static_cast<std::size_t>(width_px * height_px * 4); }
};

struct TableStyle {
    GP_TableStyle style{};
    static TableStyle defaults(int width_px = 640, int row_h_px = 20) {
        TableStyle s;
        s.style = GP_TableStyle{};
        s.style.width_px = width_px;
        s.style.row_h_px = row_h_px;
        s.style.indent_px = 14;
        s.style.expand_w_px = 12;
        s.style.name_w_px = 160;
        auto set = [&](uint8_t dst[4], uint8_t r, uint8_t g, uint8_t b, uint8_t a) { dst[0] = r; dst[1] = g; dst[2] = b; dst[3] = a; };
        set(s.style.bg_rgba, 8, 8, 12, 255);
        set(s.style.bg_sel_rgba, 18, 18, 26, 255);
        set(s.style.hdr_rgba, 20, 20, 28, 255);
        set(s.style.text_rgba, 240, 240, 245, 255);
        set(s.style.text_hdr_rgba, 255, 255, 255, 255);
        set(s.style.led_on_rgba, 255, 210, 90, 255);
        set(s.style.led_off_rgba, 70, 70, 80, 255);
        set(s.style.led_edge_rgba, 255, 255, 255, 255);
        set(s.style.axis_bg_rgba, 18, 18, 22, 255);
        set(s.style.axis_tick_rgba, 200, 200, 200, 255);
        set(s.style.axis_val_rgba, 255, 255, 140, 255);
        set(s.style.timer_rgba, 110, 180, 255, 255);
        set(s.style.wave_bg_rgba, 6, 6, 8, 255);
        set(s.style.wave_fg_rgba, 255, 255, 255, 255);
        uint32_t default_bits[9] = {1u << 0, 1u << 1, 1u << 2, 1u << 3, 1u << 4, 1u << 5, 1u << 6, 1u << 7, 1u << 8};
        for (int i = 0; i < 9; ++i) s.style.led_mask[i] = default_bits[i];
        s.style.led_edge_mask = (1u << 1) | (1u << 2) | (1u << 4) | (1u << 7) | (1u << 8);
        return s;
    }
};

struct TableCell {
    GP_TableCell cell{};
    TableCell() { cell.kind = GP_TABLE_CELL_TEXT; }

    static TableCell text(std::string_view t, int align = 0) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_TEXT;
        auto len = std::min<std::size_t>(t.size(), sizeof(c.cell.text) - 1);
        std::memcpy(c.cell.text, t.data(), len);
        c.cell.text[len] = '\0';
        c.cell.flags = static_cast<uint32_t>(align);
        return c;
    }

    static TableCell leds(uint32_t flags) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_LEDS;
        c.cell.flags = flags;
        return c;
    }

    static TableCell axis(float value, bool seen_min, bool seen_max) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_AXIS;
        c.cell.value = value;
        c.cell.flags = (seen_min ? 1u : 0u) | (seen_max ? 2u : 0u);
        return c;
    }

    static TableCell axis_calib(float value, float v_min, float v_max, float cap_min, float cap_max, float trim, float deadzone, bool seen_min, bool seen_max) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_AXIS;
        c.cell.value = value;
        c.cell.hold_s = v_min;
        c.cell.last_s = v_max;
        c.cell.flags = (seen_min ? 1u : 0u) | (seen_max ? 2u : 0u);
        std::snprintf(c.cell.text, sizeof(c.cell.text), "%0.3f %0.3f %0.3f %0.3f", double(cap_min), double(cap_max), double(trim), double(deadzone));
        return c;
    }

    static TableCell timers(float hold_s, float last_s) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_TIMERS;
        c.cell.hold_s = hold_s;
        c.cell.last_s = last_s;
        return c;
    }

    static TableCell wave(const GP_MenuWaveform* wf) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_WAVE;
        c.cell.wave = wf;
        return c;
    }

    // Signal-list helpers (right-pane signals: name, args LEDs, history, op, channel).
    // These are encoded into TEXT/WAVE cells so the existing raster can carry them; UI overlays
    // can decode the packed flags if they want richer markers (arg counts, expansion state).
    static TableCell signal_args(int req, int linked, int max_leds = 12, bool expandable = false) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_TEXT;
        const int r = std::max(0, req);
        const int l = std::max(0, linked);
        const int m = std::max(0, max_leds);
        // Pack small integers so overlays can render LED dots without extra sidecar data.
        c.cell.flags = (static_cast<uint32_t>(r) & 0xFFu) |
                       ((static_cast<uint32_t>(l) & 0xFFu) << 8) |
                       ((static_cast<uint32_t>(m) & 0xFFu) << 16) |
                       (expandable ? (1u << 24) : 0u);
        std::snprintf(c.cell.text, sizeof(c.cell.text), "args %d/%d/%d%s", l, r, m, expandable ? "*" : "");
        return c;
    }

    static TableCell signal_hist(const GP_MenuWaveform* wf) {
        // History strip is carried as a waveform handle; caller feeds samples externally.
        return wave(wf);
    }

    static TableCell signal_op(std::string_view op_lbl) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_TEXT;
        auto len = std::min<std::size_t>(op_lbl.size(), sizeof(c.cell.text) - 1);
        std::memcpy(c.cell.text, op_lbl.data(), len);
        c.cell.text[len] = '\0';
        return c;
    }

    static TableCell signal_channel(int ch) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_TEXT;
        std::string tmp = (ch < 0) ? std::string("-") : std::to_string(ch);
        std::snprintf(c.cell.text, sizeof(c.cell.text), "ch:%s", tmp.c_str());
        return c;
    }

    static TableCell calib_strip(bool invert_on, int mode /*0:none,1:trim,2:cap,3:ded*/) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_CALIB;
        uint32_t flags = 0;
        if (invert_on) flags |= 1u;
        flags |= (static_cast<uint32_t>(mode) & 0x3u) << 1;
        c.cell.flags = flags;
        return c;
    }

    static TableCell leds_arg(int count, uint32_t linked_mask, uint32_t required_mask) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_LEDS_ARG;
        c.cell.value = static_cast<float>(count);
        c.cell.flags = linked_mask;
        c.cell.reserved0 = static_cast<int32_t>(required_mask);
        return c;
    }

    struct PackedStrip {
        uint8_t count = 0;
        uint8_t reserved[3]{};
        uint32_t on_mask = 0;
        uint32_t edge_mask = 0;
        uint32_t active_mask = 0;
    };

    struct PackedTable {
        uint8_t n = 0;
        uint8_t pad[3]{};
        PackedStrip strips[3]{};
    };

    static TableCell leds_table(const std::vector<PackedStrip>& strips) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_LEDS_TABLE;
        PackedTable pt{};
        pt.n = static_cast<uint8_t>(std::min<std::size_t>(strips.size(), 3));
        for (std::size_t i = 0; i < strips.size() && i < 3; ++i) pt.strips[i] = strips[i];
        std::memcpy(c.cell.text, &pt, std::min<std::size_t>(sizeof(pt), sizeof(c.cell.text)));
        return c;
    }

    static TableCell scrollbar(float value01, int total_rows, int visible_rows, bool arrow_up_pressed, bool arrow_dn_pressed) {
        TableCell c;
        c.cell.kind = GP_TABLE_CELL_SCROLL;
        c.cell.value = value01;
        c.cell.hold_s = static_cast<float>(total_rows);
        c.cell.last_s = static_cast<float>(visible_rows);
        uint32_t flags = 0;
        if (arrow_up_pressed) flags |= 1u;
        if (arrow_dn_pressed) flags |= 1u << 1;
        c.cell.flags = flags;
        return c;
    }
};

struct TableRow {
    GP_TableRow row{};
    std::vector<TableCell> cells;

    TableRow(GP_TableRowKind kind, std::string_view label, int depth = 0, bool expanded = false, bool selected = false) {
        row.kind = static_cast<int32_t>(kind);
        row.depth = depth;
        row.expanded = expanded ? 1 : 0;
        row.selected = selected ? 1 : 0;
        auto len = std::min<std::size_t>(label.size(), sizeof(row.label) - 1);
        std::memcpy(row.label, label.data(), len);
        row.label[len] = '\0';
    }

    TableRow& add_cell(const TableCell& c) {
        if (cells.size() < 8) cells.push_back(c);
        return *this;
    }

    GP_TableRow finalize() const {
        GP_TableRow out = row;
        const int n = static_cast<int>(std::min<std::size_t>(cells.size(), 8));
        for (int i = 0; i < n; ++i) out.cells[i] = cells[static_cast<std::size_t>(i)].cell;
        out.cell_count = n;
        return out;
    }
};

// Convenience builder for right-pane signal rows (name + args + history + op + channel).
struct SignalRow {
    TableRow row;

    explicit SignalRow(std::string_view label, bool selected = false)
        : row(GP_TABLE_ROW_NOTE, label, /*depth=*/0, /*expanded=*/false, selected) {}

    SignalRow& args(int req, int linked, int max_leds = 12, bool expandable = false) {
        row.add_cell(TableCell::signal_args(req, linked, max_leds, expandable));
        return *this;
    }

    SignalRow& history(const GP_MenuWaveform* wf) {
        row.add_cell(TableCell::signal_hist(wf));
        return *this;
    }

    SignalRow& op(std::string_view op_lbl) {
        row.add_cell(TableCell::signal_op(op_lbl));
        return *this;
    }

    SignalRow& channel(int ch) {
        row.add_cell(TableCell::signal_channel(ch));
        return *this;
    }

    GP_TableRow finalize() const { return row.finalize(); }
};

class TableTexture {
public:
    TableTexture() : style_(TableStyle::defaults()) {}

    TableTexture& set_style(const GP_TableStyle& s) {
        style_.style = s;
        return *this;
    }

    TableTexture& add_column(GP_TableCellKind kind, int width_px, int align = 0) {
        GP_TableColumn c{};
        c.kind = static_cast<int32_t>(kind);
        c.width_px = width_px;
        c.align = align;
        cols_.push_back(c);
        return *this;
    }

    TableRow& add_row(GP_TableRowKind kind, std::string_view label, int depth = 0, bool expanded = false, bool selected = false) {
        rows_.emplace_back(kind, label, depth, expanded, selected);
        return rows_.back();
    }

    TableTexture& add_signal_row(const SignalRow& s) {
        rows_.push_back(s.row);
        return *this;
    }

    RenderedTable render() const {
        RenderedTable out;
        if (rows_.empty() || cols_.empty()) return out;
        std::vector<GP_TableRow> packed;
        packed.reserve(rows_.size());
        for (const auto& r : rows_) packed.push_back(r.finalize());

        GP_TableGeom geom{};
        const int row_count = static_cast<int>(packed.size());
        const int col_count = static_cast<int>(cols_.size());
        GP_TableStyle st = style_.style;
        if (!gp_table_calc_size(&st, row_count, col_count, &geom)) return out;
        const int need = geom.width_px * geom.height_px * 4;
        if (need <= 0) return out;
        out.rgba.resize(static_cast<std::size_t>(need));
        if (!gp_table_raster_rgba(packed.data(), row_count, cols_.data(), col_count, &st, out.rgba.data(), static_cast<int32_t>(out.rgba.size()), &geom)) {
            out.rgba.clear();
            return out;
        }
        out.width_px = geom.width_px;
        out.height_px = geom.height_px;
        out.geom = geom;
        return out;
    }

private:
    TableStyle style_;
    std::vector<GP_TableColumn> cols_;
    std::vector<TableRow> rows_;
};

} // namespace gp
#endif // __cplusplus

// C-linkage wrappers so other translation units can use the exact spline drawer
extern "C" void table_draw_rope_curve_blend(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, uint8_t cr, uint8_t cg, uint8_t cb, uint8_t ca, int samples_per_segment) {
    Color core_col{cr, cg, cb, ca};
    draw_rope_curve_blend(img, w, h, pitch, verts, count, jacket_px, jacket_border, core_col, samples_per_segment);
}

extern "C" void table_draw_rope_curve_blend_colored(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, const float* hues, int hue_count, int samples_per_segment, float hue_intensity) {
    draw_rope_curve_blend_colored(img, w, h, pitch, verts, count, jacket_px, jacket_border, hues, hue_count, samples_per_segment, hue_intensity);
}
