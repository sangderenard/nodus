#include "common/tensors/abstraction/tensor_types.h"
#include "common/tensors/abstraction/coo_matrix.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/tensor_backend.h"

#include <unordered_map>
#include <cmath>
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

#include "edge_tensor_fifo.h"

static GP_TableEdgeTensorSpecTyped edge_fifo_to_table_spec(const EdgeTensorFifo& fifo) {
    GP_TableEdgeTensorSpecTyped spec{};
    spec.dim_count = static_cast<int32_t>(
        std::min<size_t>(fifo.shape.size(), sizeof(spec.dims) / sizeof(spec.dims[0]))
    );
    for (int32_t i = 0; i < spec.dim_count; ++i) {
        spec.dims[i] = fifo.shape[static_cast<size_t>(i)];
    }
    spec.slots = fifo.impl ? static_cast<int32_t>(fifo.impl->slots) : 0;
    spec.top_k = fifo.impl ? static_cast<int32_t>(fifo.impl->top_k) : 0;
    spec.elem_size = fifo.impl
        ? static_cast<int32_t>(fifo.impl->elem_size)
        : static_cast<int32_t>(sizeof(float));
    spec.type_id = fifo.impl ? fifo.impl->type_id : -1;
    spec.layout = static_cast<int32_t>(fifo.layout);
    spec.dtype = static_cast<int32_t>(fifo.dtype);
    return spec;
}

