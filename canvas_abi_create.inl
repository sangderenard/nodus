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
