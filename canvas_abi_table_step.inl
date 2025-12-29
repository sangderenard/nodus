extern "C" int gp_canvas_create_table(GP_CanvasContext* ctx_, int module_idx) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->modules.size())) return 0;
    if (c->module_tables[module_idx]) return 0; // already has a table
    GP_TableContext* t = gp_table_create(nullptr);
    if (!t) return 0;
    c->module_tables[module_idx] = t;
    c->module_table_owned[module_idx] = 1;
    gp_table_set_debug_flags(t, c->debug_flags);
    {
        int segs = (c->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : std::max(2, c->sim_segs);
        gp_table_set_cable_segments(t, segs);
    }
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
        uint32_t dbg = c->debug_flags;
        bool disable_sim = (dbg & GP_CANVAS_DEBUG_NO_SPRINGS) != 0u || (dbg & GP_CANVAS_DEBUG_RING_STATIC) != 0u;
        float gravity = (dbg & GP_CANVAS_DEBUG_NO_GRAVITY) ? 0.0f : c->sim_maxforce;
        if (!disable_sim && root_tbl && gp_table_should_step_sim(root_tbl)) {
            rope_sim_step(sim, dt, gravity, c->sim_iters, c->sim_damping);
        }
    }
    // Process any pending module commits queued during input callbacks (e.g., mouse click)
    if (!c->pending_module_commits.empty()) {
        std::vector<int> commits;
        commits.swap(c->pending_module_commits);
        for (int mid : commits) {
            if (mid < 0 || mid >= static_cast<int>(c->modules.size())) continue;
            gp_canvas_export_module_to_root(ctx_, mid, nullptr);
            (void)canvas_write_module_library_manifest(c, mid);
            GP_ModuleLibrary lib{};
            if (canvas_build_module_library_for_module(c, mid, true, &lib)) {
                gp_module_library_actualize_sources(lib, lib.root_dir.c_str());
            }
            std::string module_id = gp_module_library_module_id(mid);
            std::string dest = gp_module_library_default_root();
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
                canvas_refresh_plugin_tools(c);
            } else {
                printf("module commit: build+load failed for module=%s\n", module_id.c_str());
            }
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
                    // Determine exec_mode: module-local override supersedes global override.
                    // Canvas values: -1 = no selection, 0=Sequential,1=Pooled,2=Slip,3=Free
                    int chosen_mode = -1;
                    if (mi >= 0 && mi < static_cast<int>(c->module_exec_mode.size())) chosen_mode = c->module_exec_mode[mi];
                    if (chosen_mode == -1 && c->thread_mgr_global_exec_mode != -1) chosen_mode = c->thread_mgr_global_exec_mode;
                    if (chosen_mode == 0) mod.exec_mode = ThreadManager::ExecMode::Sequential;
                    else if (chosen_mode == 1) mod.exec_mode = ThreadManager::ExecMode::Pooled;
                    else if (chosen_mode == 2) mod.exec_mode = ThreadManager::ExecMode::Slip;
                    // value 3 (Free) is a canvas-only toggle handled elsewhere (manager mode),
                    // so leave ModuleContract::exec_mode as the default for that value.
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
    GP_TableEdgeTensorSpecTyped spec{};
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
                if (!gp_table_edge_consume(table, edge_idx, kStageInputLedKey, sample.data(), static_cast<int>(stride * sizeof(float)), &written) || written <= 0) {
                    break;
                }
                size_t plen = static_cast<size_t>(written) / sizeof(float);
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

