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
