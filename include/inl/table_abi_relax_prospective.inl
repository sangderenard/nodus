// Relaxation control
#include "console_logger.h"
#ifndef printf
#define printf(...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
#ifndef fprintf
#define fprintf(file, ...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
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
                        int max_segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : std::max(2, ctx->st.cable_segments);
                        ctx->rope_sim = rope_sim_create(max_ropes, max_segs);
                        ctx->rope_id_to_sim_idx.clear();
                    }
                    float plug_z = -ctx->st.cable_plug_depth;
                    // create a single persistent prospective rope if not present
                    if (ctx->prospective_rope_idx < 0) {
                        int segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : std::max(2, ctx->st.cable_segments);
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
                        uint32_t dbg = ctx->debug_flags;
                        bool disable_sim = (dbg & GP_CANVAS_DEBUG_NO_SPRINGS) != 0u || (dbg & GP_CANVAS_DEBUG_RING_STATIC) != 0u;
                        float gravity = (dbg & GP_CANVAS_DEBUG_NO_GRAVITY) ? 0.0f : 800.0f;
                        int constraint_iters = 8;
                        float damping = 0.86f;
                        if (!disable_sim) {
                            rope_sim_step(ctx->rope_sim, static_cast<float>(dt_frame), gravity, constraint_iters, damping);
                        }
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
            int max_segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : std::max(2, ctx->st.cable_segments);
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
                int segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : std::max(2, ctx->st.cable_segments);
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
            uint32_t dbg = ctx->debug_flags;
            bool disable_sim = (dbg & GP_CANVAS_DEBUG_NO_SPRINGS) != 0u || (dbg & GP_CANVAS_DEBUG_RING_STATIC) != 0u;
            float gravity = (dbg & GP_CANVAS_DEBUG_NO_GRAVITY) ? 0.0f : 800.0f;
            int constraint_iters = 8;
            float damping = 0.86f;
            // use sim_dt (could be overridden in future if frame dt available)
            if (!disable_sim) {
                rope_sim_step(ctx->rope_sim, sim_dt, gravity, constraint_iters, damping);
            }
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
            // If table sim is disabled, draw a unified fallback curve (default: 3-point sag)
            int sim_on = 1; gp_table_get_sim_enabled(ctx, &sim_on);
            if (!sim_on) {
                int ax = info_a.x;
                int ay = info_a.y;
                int bx = info_b.x;
                int by = info_b.y;
                uint32_t flags = 0u;
                if (ei < ctx->edge_subgroup_flags.size()) flags = ctx->edge_subgroup_flags[ei];
                // Table code does not have canvas color helpers; pick a neutral tint when subgroup present.
                Color pcol = (flags != 0u) ? Color{200,160,80,220} : Color{200,200,200, static_cast<uint8_t>(std::lround(180.0f))};
                draw_fallback_rope(out_rgba, w_local, h_local, pitch_local, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by), ctx->st.cable_jacket_px, ctx->st.cable_jacket_border, pcol, 1);
                continue;
            }

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
