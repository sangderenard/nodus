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
        // Enqueue edge addition to be applied by the manager thread unless
        // the restored root table already contains this edge.
        int existing_edge_idx = -1;
        if (!gp_table_edge_index_for_pair(root, ka, kb, &existing_edge_idx)) {
            gp_table_enqueue_add_edge(root, ka, kb);
        } else {
            printf("gp_canvas_add_edge_with_type: root edge already present (edge_idx=%d)\n", existing_edge_idx);
        }
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
