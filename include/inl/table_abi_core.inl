extern "C" {
#include "mem_backend.h"
#include "value_types.h"


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
        // Storage is now owned via a backend buffer handle. Backends may be
        // host-backed or device-backed; operations should map or use vtable
        // copy hooks when accessing data.
        gp_mem_backend_handle_t storage_handle{nullptr};
        // Typed view is not persisted; map on demand. Backends should be
        // accessed via map/copy helpers — no cached typed pointer is kept.
        size_t elem_size = sizeof(float);                  // bytes per element
        int32_t type_id = -1;                              // schema id
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

        // Optional memory backend handle attached to this FIFO
        gp_mem_backend_handle_t backend{nullptr};

        Impl() : readers(new ReaderEntry[kMaxReaders]) {}
    };

    std::vector<int32_t> shape;
    std::unique_ptr<Impl> impl;
    std::vector<float> last_sample;
    std::vector<uint8_t> last_sample_bytes;
    bool last_sample_valid = false;
    bool last_sample_bytes_valid = false;
    bool delta_mode = false;
    std::vector<float> order_history;
    std::vector<float> order_integrator;
    std::vector<float> scratch;
    int32_t order_mode = 0;
    int32_t order_history_count = 0;
    int32_t order_history_cursor = 0;
    // Dirty-grid state for delta-aware buffers (tile mask for recent write).
    int32_t dirty_grid_x_req = 0;
    int32_t dirty_grid_y_req = 0;
    int32_t dirty_grid_x = 0;
    int32_t dirty_grid_y = 0;
    float dirty_threshold = 0.0f;
    std::vector<uint8_t> dirty_mask;
    std::atomic<uint64_t> dirty_seq{0};
    std::mutex dirty_mu;

    EdgeTensorFifo() : impl(new Impl()) {}
    EdgeTensorFifo(EdgeTensorFifo&&) noexcept = default;
    EdgeTensorFifo& operator=(EdgeTensorFifo&&) noexcept = default;
    EdgeTensorFifo(const EdgeTensorFifo&) = delete;
    EdgeTensorFifo& operator=(const EdgeTensorFifo&) = delete;

    void configure_default() { configure(std::vector<int32_t>{1}, 16, 0); }

    void configure(const std::vector<int32_t>& dims, size_t slot_count, size_t topk, size_t elem_size_bytes = sizeof(float), int32_t type_id_in = -1) {
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
        impl->elem_size = std::max<size_t>(1, elem_size_bytes);
        impl->type_id = type_id_in;

        last_sample.assign(impl->stride, 0.0f);
        last_sample_valid = false;
        last_sample_bytes.assign(impl->stride * impl->elem_size, 0u);
        last_sample_bytes_valid = false;
        order_history.assign(static_cast<size_t>(kMaxOrder) * impl->stride, 0.0f);
        order_integrator.assign(static_cast<size_t>(kMaxOrder) * impl->stride, 0.0f);
        scratch.assign(impl->stride, 0.0f);
        order_mode = 0;
        order_history_count = 0;
        order_history_cursor = 0;
        refresh_dirty_grid();

        size_t total_bytes = impl->stride * impl->slots * impl->elem_size;
        // Allocate a host-backed storage buffer by default. If a backend was
        // previously attached via set_backend(), use that backend to host
        // the allocation if the backend exposes alloc, otherwise fall back
        // to creating a host buffer handle.
        if (impl->backend) {
            const gp_mem_backend_vtable_t* bvt = gp_mem_backend_get_vtable(impl->backend);
            if (bvt && bvt->alloc) {
                impl->storage_handle = bvt->alloc(impl->backend, total_bytes, /*alignment=*/0);
            } else {
                // backend doesn't provide alloc: create a generic host buffer
                impl->storage_handle = gp_mem_backend_create_host(total_bytes);
            }
        } else {
            impl->storage_handle = gp_mem_backend_create_host(total_bytes);
        }
        impl->slot_seq.reset(new std::atomic<uint64_t>[impl->slots]);
        if (impl->storage_handle) {
            const gp_mem_backend_vtable_t* sh_vt = gp_mem_backend_get_vtable(impl->storage_handle);
            if (sh_vt && sh_vt->copy_to_backend) {
                std::vector<uint8_t> zeros;
                try { zeros.resize(total_bytes); } catch(...) { }
                if (zeros.size() == total_bytes) sh_vt->copy_to_backend(impl->storage_handle, 0, zeros.data(), total_bytes);
            } else {
                void* m = gp_mem_backend_map_or_null(impl->storage_handle);
                if (m) {
                    std::memset(m, 0, total_bytes);
                    gp_mem_backend_unmap(impl->storage_handle);
                }
            }
        }
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

    // Attach a memory backend to this FIFO. The FIFO will consult backend
    // capabilities (e.g. BYREF safety) when deciding how to expose pointer
    // semantics to consumers.
    void set_backend(gp_mem_backend_handle_t h) {
        if (!impl) return;
        impl->backend = h;
    }
    gp_mem_backend_handle_t get_backend() const {
        if (!impl) return nullptr;
        return impl->backend;
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
    void set_dirty_grid(int32_t grid_x, int32_t grid_y, float threshold) {
        dirty_grid_x_req = grid_x;
        dirty_grid_y_req = grid_y;
        dirty_threshold = std::max(0.0f, threshold);
        refresh_dirty_grid();
    }
    int32_t dirty_mask_copy(uint8_t* out_mask, int32_t out_len, int32_t* out_grid_x, int32_t* out_grid_y, uint64_t* out_seq) {
        if (out_grid_x) *out_grid_x = dirty_grid_x;
        if (out_grid_y) *out_grid_y = dirty_grid_y;
        if (out_seq) *out_seq = dirty_seq.load(std::memory_order_relaxed);
        if (!out_mask || out_len <= 0) return 0;
        std::lock_guard<std::mutex> lock(dirty_mu);
        int32_t need = static_cast<int32_t>(dirty_mask.size());
        int32_t copy_len = std::min(out_len, need);
        if (copy_len > 0) {
            std::memcpy(out_mask, dirty_mask.data(), static_cast<size_t>(copy_len));
        }
        return copy_len;
    }

private:
    void refresh_dirty_grid() {
        int32_t gx = dirty_grid_x_req;
        int32_t gy = dirty_grid_y_req;
        if (gx <= 0 || gy <= 0 || !impl) {
            std::lock_guard<std::mutex> lock(dirty_mu);
            dirty_grid_x = 0;
            dirty_grid_y = 0;
            dirty_mask.clear();
            dirty_seq.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        size_t width = 0;
        if (!shape.empty()) width = static_cast<size_t>(std::max<int32_t>(1, shape.back()));
        if (width == 0) width = impl->stride;
        size_t height = (width > 0) ? ((impl->stride + width - 1) / width) : 0;
        int32_t max_x = static_cast<int32_t>(std::max<size_t>(1, width));
        int32_t max_y = static_cast<int32_t>(std::max<size_t>(1, height));
        int32_t min_x = (max_x >= 2) ? 2 : 1;
        int32_t min_y = (max_y >= 2) ? 2 : 1;
        gx = std::clamp(gx, min_x, max_x);
        gy = std::clamp(gy, min_y, max_y);
        std::lock_guard<std::mutex> lock(dirty_mu);
        dirty_grid_x = gx;
        dirty_grid_y = gy;
        dirty_mask.assign(static_cast<size_t>(gx) * static_cast<size_t>(gy), 0u);
        dirty_seq.fetch_add(1, std::memory_order_relaxed);
    }

    void update_dirty_mask(const void* current_sample, size_t bytes) {
        if (!impl) return;
        if (dirty_grid_x <= 0 || dirty_grid_y <= 0 || !current_sample) return;
        size_t width = 0;
        if (!shape.empty()) width = static_cast<size_t>(std::max<int32_t>(1, shape.back()));
        if (width == 0) width = impl->stride;
        size_t height = (width > 0) ? ((impl->stride + width - 1) / width) : 0;
        if (width == 0 || height == 0) return;
        int32_t gx = dirty_grid_x;
        int32_t gy = dirty_grid_y;
        std::lock_guard<std::mutex> lock(dirty_mu);
        if (dirty_mask.size() != static_cast<size_t>(gx) * static_cast<size_t>(gy)) {
            dirty_mask.assign(static_cast<size_t>(gx) * static_cast<size_t>(gy), 0u);
        }
        if (!last_sample_bytes_valid || last_sample_bytes.size() != bytes) {
            std::fill(dirty_mask.begin(), dirty_mask.end(), static_cast<uint8_t>(1));
            dirty_seq.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const size_t elem_size = impl->elem_size;
        const bool use_float = (elem_size == sizeof(float));
        const float* cur_f = reinterpret_cast<const float*>(current_sample);
        const float* prev_f = nullptr;
        const uint8_t* cur_b = reinterpret_cast<const uint8_t*>(current_sample);
        const uint8_t* prev_b = reinterpret_cast<const uint8_t*>(last_sample_bytes.data());
        if (use_float) {
            if (!last_sample_valid || last_sample.size() != impl->stride) {
                std::fill(dirty_mask.begin(), dirty_mask.end(), static_cast<uint8_t>(1));
                dirty_seq.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            prev_f = last_sample.data();
        }
        auto seg_bounds = [](int32_t idx, int32_t count, size_t size, size_t* out_start, size_t* out_end) {
            if (!out_start || !out_end) return;
            if (count <= 0 || size == 0) {
                *out_start = 0;
                *out_end = 0;
                return;
            }
            size_t base = size / static_cast<size_t>(count);
            size_t rem = size % static_cast<size_t>(count);
            size_t start = static_cast<size_t>(idx) * base + static_cast<size_t>(std::min<int32_t>(idx, static_cast<int32_t>(rem)));
            size_t len = base + ((static_cast<size_t>(idx) < rem) ? 1u : 0u);
            *out_start = start;
            *out_end = start + len;
        };
        float threshold = std::max(0.0f, dirty_threshold);
        size_t stride_elems = impl->stride;
        for (int32_t sy = 0; sy < gy; ++sy) {
            size_t y0 = 0, y1 = 0;
            seg_bounds(sy, gy, height, &y0, &y1);
            for (int32_t sx = 0; sx < gx; ++sx) {
                size_t x0 = 0, x1 = 0;
                seg_bounds(sx, gx, width, &x0, &x1);
                bool dirty = false;
                for (size_t y = y0; y < y1 && !dirty; ++y) {
                    size_t row = y * width;
                    for (size_t x = x0; x < x1; ++x) {
                        size_t idx = row + x;
                        if (idx >= stride_elems) { dirty = false; break; }
                        if (use_float) {
                            float dv = std::fabs(cur_f[idx] - prev_f[idx]);
                            if (dv > threshold) { dirty = true; break; }
                        } else {
                            size_t off = idx * elem_size;
                            bool diff = false;
                            for (size_t b = 0; b < elem_size; ++b) {
                                if (cur_b[off + b] != prev_b[off + b]) { diff = true; break; }
                            }
                            if (diff) { dirty = true; break; }
                        }
                    }
                }
                dirty_mask[static_cast<size_t>(sy) * static_cast<size_t>(gx) + static_cast<size_t>(sx)] = dirty ? 1u : 0u;
            }
        }
        dirty_seq.fetch_add(1, std::memory_order_relaxed);
    }

    void mark_dirty_all() {
        if (dirty_grid_x <= 0 || dirty_grid_y <= 0) return;
        std::lock_guard<std::mutex> lock(dirty_mu);
        if (!dirty_mask.empty()) {
            std::fill(dirty_mask.begin(), dirty_mask.end(), static_cast<uint8_t>(1));
            dirty_seq.fetch_add(1, std::memory_order_relaxed);
        }
    }
public:

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

    bool push(uint64_t edge_id, uint64_t writer_id, const void* sample_bytes, size_t sample_len_bytes, bool* out_dropped) {
        if (!impl || !impl->configured) return false;
        if (!sample_bytes) return false;
        if (sample_len_bytes != impl->stride * impl->elem_size) return false;
        uint64_t bound = impl->writer.load(std::memory_order_relaxed);
        if (bound == 0) {
            (void)impl->writer.compare_exchange_strong(bound, writer_id, std::memory_order_relaxed);
        }
        bound = impl->writer.load(std::memory_order_relaxed);
        if (bound != 0 && bound != writer_id) return false;

        const uint8_t* sample_bytes_u = reinterpret_cast<const uint8_t*>(sample_bytes);
        const float* effective_sample_f = nullptr;
        std::vector<float> temp_scratch;
        if (impl->elem_size == sizeof(float) && order_mode != 0 && scratch.size() == impl->stride) {
            const float* sample_f = reinterpret_cast<const float*>(sample_bytes);
            int32_t mode = std::clamp(order_mode, -kMaxOrder, kMaxOrder);
            if (mode > 0) {
                for (size_t i = 0; i < impl->stride; ++i) {
                    float acc = sample_f[i];
                    order_integrator[i] += acc;
                    for (int32_t level = 1; level < mode; ++level) {
                        size_t idx = static_cast<size_t>(level) * impl->stride + i;
                        size_t prev = static_cast<size_t>(level - 1) * impl->stride + i;
                        order_integrator[idx] += order_integrator[prev];
                    }
                    scratch[i] = order_integrator[static_cast<size_t>(mode - 1) * impl->stride + i];
                }
                effective_sample_f = scratch.data();
            } else {
                int32_t order = -mode;
                if (order_history_count >= order) {
                    for (size_t i = 0; i < impl->stride; ++i) {
                        float sum = sample_f[i];
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
                    effective_sample_f = scratch.data();
                } else {
                    temp_scratch.resize(impl->stride);
                    std::memcpy(temp_scratch.data(), sample_bytes, impl->stride * impl->elem_size);
                    effective_sample_f = reinterpret_cast<const float*>(temp_scratch.data());
                }
            }
        }

        if (delta_mode && last_sample_valid) {
            size_t byte_count = impl->stride * impl->elem_size;
            if (last_sample.size() * sizeof(float) == byte_count) {
                if (std::memcmp(last_sample.data(), sample_bytes, byte_count) == 0) {
                    if (out_dropped) *out_dropped = 0;
                    return true;
                }
            }
        }

        uint64_t seq = impl->write_seq.load(std::memory_order_relaxed);
        bool dropped = false;
        if (!ensure_space_for_write(seq, &dropped, edge_id)) {
            if (out_dropped) *out_dropped = 1;
            return false;
        }

        size_t slot = static_cast<size_t>(seq % static_cast<uint64_t>(impl->slots));
        size_t bytes = impl->stride * impl->elem_size;
        const void* effective_sample_ptr = (effective_sample_f ? reinterpret_cast<const void*>(effective_sample_f) : reinterpret_cast<const void*>(sample_bytes_u));
        update_dirty_mask(effective_sample_ptr, bytes);
        // Write into the FIFO's backend-owned storage buffer.
        gp_mem_backend_handle_t sh = impl->storage_handle;
        if (!sh) {
            if (out_dropped) *out_dropped = 1;
            return false;
        }
        const gp_mem_backend_vtable_t* sh_vt = gp_mem_backend_get_vtable(sh);
        size_t offset = slot * bytes;
        if (sh_vt && sh_vt->copy_to_backend) {
            if (!sh_vt->copy_to_backend(sh, offset, effective_sample_ptr, bytes)) {
                if (out_dropped) *out_dropped = 1;
                return false;
            }
        } else {
            void* m = gp_mem_backend_map_or_null(sh);
            if (!m) {
                if (out_dropped) *out_dropped = 1;
                return false;
            }
            std::memcpy(reinterpret_cast<uint8_t*>(m) + offset, effective_sample_ptr, bytes);
            gp_mem_backend_unmap(sh);
        }
        impl->slot_seq[slot].store(seq + 1, std::memory_order_release);
        impl->write_seq.store(seq + 1, std::memory_order_release);
        note_activity(seq + 1, impl->last_write_seq, impl->last_write_region, impl->write_friction, impl->write_phase);
        if (!order_history.empty()) {
            size_t base = static_cast<size_t>(order_history_cursor) * impl->stride;
            if (impl->elem_size == sizeof(float)) std::memcpy(order_history.data() + base, sample_bytes, impl->stride * impl->elem_size);
            order_history_cursor = (order_history_cursor + 1) % kMaxOrder;
            order_history_count = std::min(order_history_count + 1, kMaxOrder);
        }
        if (last_sample.size() == impl->stride) {
            if (impl->elem_size == sizeof(float)) {
                std::memcpy(last_sample.data(), (effective_sample_f ? effective_sample_f : reinterpret_cast<const float*>(sample_bytes)), impl->stride * sizeof(float));
                last_sample_valid = true;
            } else {
                // For non-float element sizes we keep last_sample invalid (or zeroed)
                last_sample_valid = false;
            }
        }
        if (last_sample_bytes.size() == impl->stride * impl->elem_size) {
            std::memcpy(last_sample_bytes.data(), effective_sample_ptr, impl->stride * impl->elem_size);
            last_sample_bytes_valid = true;
        } else {
            last_sample_bytes_valid = false;
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
        size_t bytes = impl->stride * impl->elem_size;
        // Write pointer bytes into storage buffer
        gp_mem_backend_handle_t sh_ptr = impl->storage_handle;
        if (!sh_ptr) { if (out_dropped) *out_dropped = 1; return false; }
        const gp_mem_backend_vtable_t* shp_vt = gp_mem_backend_get_vtable(sh_ptr);
        size_t poff = slot * bytes;
        size_t pbytes = std::min<size_t>(sizeof(void*), bytes);
        if (shp_vt && shp_vt->copy_to_backend) {
            if (!shp_vt->copy_to_backend(sh_ptr, poff, &ptr, pbytes)) { if (out_dropped) *out_dropped = 1; return false; }
        } else {
            void* m = gp_mem_backend_map_or_null(sh_ptr);
            if (!m) { if (out_dropped) *out_dropped = 1; return false; }
            std::memset(reinterpret_cast<uint8_t*>(m) + poff, 0, bytes);
            std::memcpy(reinterpret_cast<uint8_t*>(m) + poff, &ptr, pbytes);
            gp_mem_backend_unmap(sh_ptr);
        }
        impl->slot_seq[slot].store(seq + 1, std::memory_order_release);
        impl->write_seq.store(seq + 1, std::memory_order_release);
        note_activity(seq + 1, impl->last_write_seq, impl->last_write_region, impl->write_friction, impl->write_phase);
        impl->cv.notify_all();
        if (out_dropped) *out_dropped = dropped ? 1 : 0;
        return true;
    }

    // Push a single typed element into the FIFO by popping it from a
    // RawStackFrame source. This attempts an optimized backend-to-backend
    // transfer when both source frame and FIFO storage expose backends and
    // fallbacks to a host-mediated copy otherwise. On success the source
    // frame has the element removed (popped). Returns true on success.
    bool push_from_frame(uint64_t edge_id, uint64_t writer_id, RawStackFrame* src_frame, int32_t type_id, bool* out_dropped) {
        if (!impl || !impl->configured || !src_frame) return false;
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
        size_t bytes = impl->stride * impl->elem_size;
        size_t dst_offset = slot * bytes;

        // Try direct backend-to-backend via raw_stack helper which knows how
        // to pop from the frame and write into a destination backend.
        if (src_frame->backend && impl->storage_handle) {
            if (gp_raw_stack_frame_pop_into_backend(src_frame, impl->storage_handle, dst_offset, type_id)) {
                impl->slot_seq[slot].store(seq + 1, std::memory_order_release);
                impl->write_seq.store(seq + 1, std::memory_order_release);
                note_activity(seq + 1, impl->last_write_seq, impl->last_write_region, impl->write_friction, impl->write_phase);
                impl->cv.notify_all();
                mark_dirty_all();
                last_sample_valid = false;
                last_sample_bytes_valid = false;
                if (out_dropped) *out_dropped = dropped ? 1 : 0;
                return true;
            }
        }

        // Fallback host-mediated path: read element bytes into host tmp,
        // push them into FIFO storage via existing push(), and then pop the
        // element from the source frame. This duplicates a transfer but
        // keeps correctness when optimized path unavailable.
        const ValueType* vt = ValueTypeRegistry::global().get(type_id);
        if (!vt || vt->size == 0) return false;
        size_t need = vt->size;
        std::vector<uint8_t> tmp;
        try { tmp.resize(bytes); } catch(...) { return false; }
        std::memset(tmp.data(), 0, bytes);

        // Read source element into tmp_head without popping yet.
        bool read_ok = false;
        if (src_frame->backend) {
            const gp_mem_backend_vtable_t* src_vt = gp_mem_backend_get_vtable(src_frame->backend);
            size_t start = (src_frame->byte_count >= need) ? (src_frame->byte_count - need) : 0;
            if (src_vt && src_vt->copy_from_backend) {
                if (src_vt->copy_from_backend(src_frame->backend, start, tmp.data(), need)) read_ok = true;
            }
            if (!read_ok) {
                void* m = gp_mem_backend_map_or_null(src_frame->backend);
                if (m) {
                    std::memcpy(tmp.data(), reinterpret_cast<uint8_t*>(m) + start, need);
                    gp_mem_backend_unmap(src_frame->backend);
                    read_ok = true;
                }
            }
        }
        if (!read_ok) return false;

        // Use existing push() to write tmp into FIFO storage (will copy-to-backend)
        if (!push(edge_id, writer_id, tmp.data(), bytes, &dropped)) return false;

        // Now remove the element from source frame (pop) to reflect transfer.
        // Pop into a throwaway buffer.
        std::vector<uint8_t> throwaway;
        try { throwaway.resize(need); } catch(...) { return false; }
        if (!raw_stack_pop_block(*src_frame, throwaway.data(), 1, type_id)) return false;

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
        size_t bytes = impl->stride * impl->elem_size;
        void* p = nullptr;
        gp_mem_backend_handle_t shr = impl->storage_handle;
        if (!shr) return false;
        const gp_mem_backend_vtable_t* shr_vt = gp_mem_backend_get_vtable(shr);
        size_t rbytes = std::min<size_t>(sizeof(void*), bytes);
        if (shr_vt && shr_vt->copy_from_backend) {
            size_t roff = slot * bytes;
            if (!shr_vt->copy_from_backend(shr, roff, &p, rbytes)) return false;
        } else {
            void* m = gp_mem_backend_map_or_null(shr);
            if (!m) return false;
            std::memcpy(&p, reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size, rbytes);
            gp_mem_backend_unmap(shr);
        }
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

        size_t bytes = impl->stride * impl->elem_size;
        void* p = nullptr;
        if (impl->backend) {
            const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(impl->backend);
            if (vt && vt->copy_from_backend) {
                size_t offset = slot * bytes;
                void* tmp = nullptr;
                if (!vt->copy_from_backend(impl->backend, offset, &tmp, std::min<size_t>(sizeof(void*), bytes))) return false;
                p = tmp;
            } else {
                void* m = gp_mem_backend_map_or_null(impl->storage_handle);
                if (!m) return false;
                std::memcpy(&p, reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size, std::min<size_t>(sizeof(void*), bytes));
                gp_mem_backend_unmap(impl->storage_handle);
            }
            } else {
                void* m = gp_mem_backend_map_or_null(impl->storage_handle);
                if (!m) return false;
                std::memcpy(&p, reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size, std::min<size_t>(sizeof(void*), bytes));
                gp_mem_backend_unmap(impl->storage_handle);
            }
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

    bool pop(uint64_t reader_id, void* out_sample_bytes, size_t out_cap_bytes, size_t& out_written_bytes) {
        out_written_bytes = 0;
        if (!impl || !impl->configured) return false;
        ReaderEntry* r = find_reader(reader_id);
        if (!r) return false;
        if (!out_sample_bytes || out_cap_bytes < impl->stride * impl->elem_size) return false;

        uint64_t rseq = r->seq.load(std::memory_order_relaxed);
        size_t slot = static_cast<size_t>(rseq % static_cast<uint64_t>(impl->slots));
        uint64_t observed = impl->slot_seq[slot].load(std::memory_order_acquire);
        if (observed != (rseq + 1)) return false;

        size_t bytes = impl->stride * impl->elem_size;
        if (impl->backend) {
            const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(impl->backend);
            if (vt && vt->copy_from_backend) {
                size_t offset = slot * bytes;
                if (!vt->copy_from_backend(impl->backend, offset, out_sample_bytes, bytes)) return false;
            } else {
                void* m = gp_mem_backend_map_or_null(impl->storage_handle);
                if (!m) return false;
                uint8_t* src = reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size;
                std::memcpy(out_sample_bytes, src, bytes);
                gp_mem_backend_unmap(impl->storage_handle);
            }
        } else {
            void* m = gp_mem_backend_map_or_null(impl->storage_handle);
            if (!m) return false;
            uint8_t* src = reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size;
            std::memcpy(out_sample_bytes, src, bytes);
            gp_mem_backend_unmap(impl->storage_handle);
        }
        r->seq.store(rseq + 1, std::memory_order_relaxed);
        note_activity(rseq + 1, impl->last_read_seq, impl->last_read_region, impl->read_friction, impl->read_phase);
        impl->cv.notify_all();
        out_written_bytes = impl->stride * impl->elem_size;
        return true;
    }

    // Non-destructive peek: copy the next available sample for `reader_id`
    // into `out_sample` without advancing the reader sequence. Returns true
    // if a sample was available and copied. Contract matches `pop` with the
    // requirement that out_cap >= impl->stride.
    bool peek(uint64_t reader_id, void* out_sample_bytes, size_t out_cap_bytes, size_t& out_written_bytes) {
        out_written_bytes = 0;
        if (!impl || !impl->configured) return false;
        ReaderEntry* r = find_reader(reader_id);
        if (!r) return false;
        if (!out_sample_bytes || out_cap_bytes < impl->stride * impl->elem_size) return false;

        uint64_t rseq = r->seq.load(std::memory_order_relaxed);
        size_t slot = static_cast<size_t>(rseq % static_cast<uint64_t>(impl->slots));
        uint64_t observed = impl->slot_seq[slot].load(std::memory_order_acquire);
        if (observed != (rseq + 1)) return false;

        size_t bytes = impl->stride * impl->elem_size;
        if (impl->backend) {
            const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(impl->backend);
            if (vt && vt->copy_from_backend) {
                size_t offset = slot * bytes;
                if (!vt->copy_from_backend(impl->backend, offset, out_sample_bytes, bytes)) return false;
            } else {
                void* m = gp_mem_backend_map_or_null(impl->storage_handle);
                if (!m) return false;
                uint8_t* src = reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size;
                std::memcpy(out_sample_bytes, src, bytes);
                gp_mem_backend_unmap(impl->storage_handle);
            }
        } else {
            void* m = gp_mem_backend_map_or_null(impl->storage_handle);
            if (!m) return false;
            uint8_t* src = reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size;
            std::memcpy(out_sample_bytes, src, bytes);
            gp_mem_backend_unmap(impl->storage_handle);
        }
        // Note: do NOT advance r->seq and do NOT call note_activity / notify.
        out_written_bytes = impl->stride * impl->elem_size;
        return true;
    }

    bool push_blocking(uint64_t edge_id, uint64_t writer_id, const void* sample_bytes, size_t sample_len_bytes, bool* out_dropped, int timeout_ms) {
        if (!impl || !impl->configured) return false;
        if (!sample_bytes || sample_len_bytes != impl->stride * impl->elem_size) return false;
        uint64_t bound = impl->writer.load(std::memory_order_relaxed);
        if (bound != 0 && bound != writer_id) return false;
        if (timeout_ms == 0) return push(edge_id, writer_id, sample_bytes, sample_len_bytes, out_dropped);
        using clock = std::chrono::steady_clock;
        auto deadline = (timeout_ms < 0) ? clock::time_point::max() : (clock::now() + std::chrono::milliseconds(timeout_ms));
        std::unique_lock<std::mutex> lk(impl->cv_mu);
        while (true) {
            lk.unlock();
            bool ok = push(edge_id, writer_id, sample_bytes, sample_len_bytes, out_dropped);
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
        size_t cur_bytes = impl->stride * impl->elem_size;
        if (cur_bytes >= min_bytes) return true;
        size_t needed_elems = (min_bytes + impl->elem_size - 1) / impl->elem_size;
        size_t new_stride = std::max(impl->stride, needed_elems);

        // allocate a new backend buffer for the resized layout and copy
        size_t new_total = new_stride * impl->slots * impl->elem_size;
        gp_mem_backend_handle_t old_h = impl->storage_handle;
        gp_mem_backend_handle_t new_h = nullptr;
        if (impl->backend) {
            const gp_mem_backend_vtable_t* bvt = gp_mem_backend_get_vtable(impl->backend);
            if (bvt && bvt->alloc) new_h = bvt->alloc(impl->backend, new_total, /*alignment=*/0);
            else new_h = gp_mem_backend_create_host(new_total);
        } else {
            new_h = gp_mem_backend_create_host(new_total);
        }
        if (!new_h) return false;
        // zero initialize new buffer
        const gp_mem_backend_vtable_t* new_vt = gp_mem_backend_get_vtable(new_h);
        if (new_vt && new_vt->copy_to_backend) {
            std::vector<uint8_t> zeros;
            try { zeros.resize(new_total); } catch(...) { }
            if (zeros.size() == new_total) new_vt->copy_to_backend(new_h, 0, zeros.data(), new_total);
        } else {
            void* nm = gp_mem_backend_map_or_null(new_h);
            if (nm) { std::memset(nm, 0, new_total); gp_mem_backend_unmap(new_h); }
        }

        // copy per-slot from old_h -> new_h
        const gp_mem_backend_vtable_t* old_vt = gp_mem_backend_get_vtable(old_h);
        for (size_t s = 0; s < impl->slots; ++s) {
            size_t copy_bytes = impl->stride * impl->elem_size;
            size_t old_off = s * impl->stride * impl->elem_size;
            size_t new_off = s * new_stride * impl->elem_size;
            // attempt optimized transfer between buffers
            if (old_vt && new_vt && old_vt->copy_between_backends) {
                if (!old_vt->copy_between_backends(old_h, new_h, old_off, new_off, copy_bytes)) {
                    // fallback to staged per-slot copy
                    std::vector<uint8_t> tmp;
                    try { tmp.resize(copy_bytes); } catch(...) { gp_mem_backend_release(new_h); return false; }
                    if (old_vt && old_vt->copy_from_backend) {
                        if (!old_vt->copy_from_backend(old_h, old_off, tmp.data(), copy_bytes)) { gp_mem_backend_release(new_h); return false; }
                    } else {
                        void* om = gp_mem_backend_map_or_null(old_h);
                        if (!om) { gp_mem_backend_release(new_h); return false; }
                        std::memcpy(tmp.data(), reinterpret_cast<uint8_t*>(om) + old_off, copy_bytes);
                        gp_mem_backend_unmap(old_h);
                    }
                    if (new_vt && new_vt->copy_to_backend) {
                        if (!new_vt->copy_to_backend(new_h, new_off, tmp.data(), copy_bytes)) { gp_mem_backend_release(new_h); return false; }
                    } else {
                        void* nm = gp_mem_backend_map_or_null(new_h);
                        if (!nm) { gp_mem_backend_release(new_h); return false; }
                        std::memcpy(reinterpret_cast<uint8_t*>(nm) + new_off, tmp.data(), copy_bytes);
                        gp_mem_backend_unmap(new_h);
                    }
                }
            } else {
                // staged copy path
                std::vector<uint8_t> tmp;
                try { tmp.resize(copy_bytes); } catch(...) { gp_mem_backend_release(new_h); return false; }
                if (old_vt && old_vt->copy_from_backend) {
                    if (!old_vt->copy_from_backend(old_h, old_off, tmp.data(), copy_bytes)) { gp_mem_backend_release(new_h); return false; }
                } else {
                    void* om = gp_mem_backend_map_or_null(old_h);
                    if (!om) { gp_mem_backend_release(new_h); return false; }
                    std::memcpy(tmp.data(), reinterpret_cast<uint8_t*>(om) + old_off, copy_bytes);
                    gp_mem_backend_unmap(old_h);
                }
                if (new_vt && new_vt->copy_to_backend) {
                    if (!new_vt->copy_to_backend(new_h, new_off, tmp.data(), copy_bytes)) { gp_mem_backend_release(new_h); return false; }
                } else {
                    void* nm = gp_mem_backend_map_or_null(new_h);
                    if (!nm) { gp_mem_backend_release(new_h); return false; }
                    std::memcpy(reinterpret_cast<uint8_t*>(nm) + new_off, tmp.data(), copy_bytes);
                    gp_mem_backend_unmap(new_h);
                }
            }
        }

        // free old storage handle and swap in new one
        if (old_vt && old_vt->free) old_vt->free(old_h); else gp_mem_backend_release(old_h);
        impl->storage_handle = new_h;
        impl->stride = new_stride;

        // update our cached scratch/last_sample sizes to match new stride
        last_sample.assign(impl->stride, 0.0f);
        scratch.assign(impl->stride, 0.0f);
        order_history.assign(static_cast<size_t>(kMaxOrder) * impl->stride, 0.0f);
        order_integrator.assign(static_cast<size_t>(kMaxOrder) * impl->stride, 0.0f);
        return true;
    }

    bool pop_blocking(uint64_t reader_id, void* out_sample_bytes, size_t out_cap_bytes, size_t& out_written, int timeout_ms) {
        if (!impl || !impl->configured) return false;
        if (!find_reader(reader_id)) return false;
        if (!out_sample_bytes || out_cap_bytes < impl->stride * impl->elem_size) return false;
        if (timeout_ms == 0) return pop(reader_id, out_sample_bytes, out_cap_bytes, out_written);
        using clock = std::chrono::steady_clock;
        auto deadline = (timeout_ms < 0) ? clock::time_point::max() : (clock::now() + std::chrono::milliseconds(timeout_ms));
        std::unique_lock<std::mutex> lk(impl->cv_mu);
        while (true) {
            lk.unlock();
            bool ok = pop(reader_id, out_sample_bytes, out_cap_bytes, out_written);
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

    GP_TableEdgeTensorSpecTyped to_spec() const {
        GP_TableEdgeTensorSpecTyped s{};
        s.dim_count = static_cast<int32_t>(std::min<size_t>(shape.size(), sizeof(s.dims) / sizeof(s.dims[0])));
        for (int32_t i = 0; i < s.dim_count; ++i) s.dims[i] = shape[static_cast<size_t>(i)];
        s.slots = impl ? static_cast<int32_t>(impl->slots) : 0;
        s.top_k = impl ? static_cast<int32_t>(impl->top_k) : 0;
        s.elem_size = impl ? static_cast<int32_t>(impl->elem_size) : static_cast<int32_t>(sizeof(float));
        s.type_id = impl ? impl->type_id : -1;
        return s;
    }
};

