// Deferred console logging
#include "console_logger.h"
#ifndef printf
#define printf(...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
#ifndef fprintf
#define fprintf(file, ...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
#include <thread>
#include <chrono>
#include <atomic>

static void autosave_worker_loop(GP_CanvasContextImpl* c) {
    using namespace std::chrono;
    while (c->autosave_thread_stop.load(std::memory_order_acquire) == 0) {
        if (c->autosave_save_requested.load(std::memory_order_acquire) != 0) {
            long long when_ms = c->autosave_save_request_when_ms.load(std::memory_order_acquire);
            long long now_ms = duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
            if (now_ms >= when_ms) {
                long long last_save = c->autosave_last_save_ts_ms.load(std::memory_order_acquire);
                long long cooldown_ms = static_cast<long long>(c->autosave_action_cooldown_s * 1000.0);
                if ((now_ms - last_save) >= cooldown_ms) {
                    int expected = 0;
                    if (c->autosave_save_in_progress.compare_exchange_strong(expected, 1)) {
                        // perform save in background
                        std::string path = c->autosave_path;
                        if (!path.empty()) {
                            gp_canvas_save_to_file(reinterpret_cast<GP_CanvasContext*>(c), path.c_str());
                            c->autosave_last_save_ts_ms.store(duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count(), std::memory_order_release);
                        }
                        c->autosave_save_in_progress.store(0, std::memory_order_release);
                        c->autosave_save_requested.store(0, std::memory_order_release);
                    }
                }
            }
        }
        std::this_thread::sleep_for(milliseconds(40));
    }
}

static void ensure_autosave_thread_started(GP_CanvasContextImpl* c) {
    int stop = c->autosave_thread_stop.load(std::memory_order_acquire);
    if (stop == 0 && c->autosave_thread && c->autosave_thread->joinable()) return;
    c->autosave_thread_stop.store(0, std::memory_order_release);
    c->autosave_thread = std::make_unique<std::thread>([c]() { autosave_worker_loop(c); });
}

// Lazily create/retrieve a stable tool-row id for a specific module/contact.
// Stored in `module_binding_id_map` keyed by (module_idx<<32)|contact_idx.
extern "C" uint64_t gp_canvas_get_tool_row_id(GP_CanvasContext* ctx_, int module_idx, int contact_idx) {
    if (!ctx_ || module_idx < 0 || contact_idx < 0) return 0ull;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    uint64_t map_key = (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(contact_idx));
    auto it = c->module_binding_id_map.find(map_key);
    if (it != c->module_binding_id_map.end()) return it->second;
    uint64_t nid = gp_canvas_generate_id(ctx_, 0ull);
    if (nid == 0ull) return 0ull;
    c->module_binding_id_map[map_key] = nid;
    return nid;
}

static void stop_autosave_thread(GP_CanvasContextImpl* c) {
    c->autosave_thread_stop.store(1, std::memory_order_release);
    if (c->autosave_thread && c->autosave_thread->joinable()) {
        c->autosave_thread->join();
    }
    c->autosave_thread.reset();
}

extern "C" void* gp_canvas_create_action_from_enum(GP_CanvasContext* ctx_, int32_t action_id) {
    if (!ctx_) return nullptr;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    auto *pa = new GP_CanvasContextImpl::PendingAction();
    pa->action_id = action_id;
    std::memset(&pa->hit, 0, sizeof(pa->hit));
    pa->dx = 0.0f;
    pa->dy = 0.0f;
    return reinterpret_cast<void*>(pa);
}

extern "C" int gp_canvas_bind_action_ptr_to_module_port(GP_CanvasContext* ctx_, int module_idx, int is_send, int col, int led_idx, void* pending_ptr) {
    if (!ctx_ || module_idx < 0 || !pending_ptr) return 0;
    if (led_idx < 0 || led_idx >= kModuleExtraLedCount) return 0;
    if (col < 0 || col > 1) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->module_frame_links.size())) return 0;
    // Enforce single-writer rule: refuse to bind additional send ports
    // if a send binding already exists for this LED index.
    if (is_send) {
        if (c->module_frame_links[module_idx].ptrs[0][static_cast<size_t>(led_idx)] || c->module_frame_links[module_idx].ptrs[1][static_cast<size_t>(led_idx)]) {
            return 0;
        }
    }
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
            // Generate/retrieve a stable tool-row id using the canvas common id generator.
            // Do NOT fall back to any canonical composed key; if a tool-row id
            // could not be generated, we do not create a root FIFO edge for
            // this binding. This enforces the new-only id policy.
            uint64_t binding_id = gp_canvas_get_tool_row_id(reinterpret_cast<GP_CanvasContext*>(c), module_idx, contact_idx);
            if (binding_id) {
                unsigned long long ka = binding_id;
                unsigned long long kb = ka + 1ull;
                gp_table_add_edge(root, ka, kb);
                int idx = -1;
                if (gp_table_edge_index_for_key(root, ka, &idx) && idx >= 0) {
                    ab.root_edge_idx = idx;
                    ab.writer_key = ka;
                    // record binding id mapping so manager can discover it without touching port UUIDs
                    uint64_t map_key = (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(contact_idx));
                    c->module_binding_id_map[map_key] = binding_id;
                    fprintf(stderr, "[DBG] bind module=%d row=%d led=%d action=%d root_edge=%d writer_key=%llu\n", module_idx, row, led_idx, aid, ab.root_edge_idx, (unsigned long long)ab.writer_key);
                }
            } else {
                // No tool-row id available: leave root_edge_idx == -1 and writer_key == 0
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
    // Backwards-friendly convenience: if caller requests binding for any of
    // the mouse action enums, bind the whole mouse action group (down/up/move/scroll up/scroll down)
    // into consecutive LED slots starting at `led_idx`. This ensures frontends
    // that expect separate ports per mouse action get them automatically.
    if (action_id >= CANVAS_ACT_MOUSE_DOWN && action_id <= CANVAS_ACT_MOUSE_SCROLL_DOWN) {
        const int32_t mouse_actions[] = { CANVAS_ACT_MOUSE_DOWN, CANVAS_ACT_MOUSE_UP, CANVAS_ACT_MOUSE_MOVE, CANVAS_ACT_MOUSE_SCROLL_UP, CANVAS_ACT_MOUSE_SCROLL_DOWN };
        const int count = static_cast<int>(sizeof(mouse_actions)/sizeof(mouse_actions[0]));
        // Ensure led range fits
        for (int i = 0; i < count; ++i) {
            int li = led_idx + i;
            if (li < 0 || li >= kModuleExtraLedCount) return 0;
        }
        // Create and bind each pending action; if any fail, rollback previous binds
        std::vector<void*> created;
        created.reserve(count);
        for (int i = 0; i < count; ++i) {
            int32_t aid = mouse_actions[i];
            void* pa = gp_canvas_create_action_from_enum(ctx_, aid);
            if (!pa) { // rollback
                for (void* p : created) gp_canvas_free_pending_action(ctx_, p);
                return 0;
            }
            int li = led_idx + i;
            if (!gp_canvas_bind_action_ptr_to_module_port(ctx_, module_idx, is_send, col, li, pa)) {
                // cleanup on failure
                gp_canvas_free_pending_action(ctx_, pa);
                for (void* p : created) gp_canvas_free_pending_action(ctx_, p);
                return 0;
            }
            created.push_back(pa);
        }
        return 1;
    }

    // Default: bind single requested action
    void* pa = gp_canvas_create_action_from_enum(ctx_, action_id);
    if (!pa) return 0;
    if (!gp_canvas_bind_action_ptr_to_module_port(ctx_, module_idx, is_send, col, led_idx, pa)) {
        // cleanup on failure
        gp_canvas_free_pending_action(ctx_, pa);
        return 0;
    }
    return 1;
}

    extern "C" int gp_canvas_unbind_action_from_module_port(GP_CanvasContext* ctx_, int module_idx, int is_send, int col, int led_idx) {
        if (!ctx_) return 0;
        auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
        if (module_idx < 0 || module_idx >= static_cast<int>(c->module_frame_links.size())) return 0;
        if (led_idx < 0 || led_idx >= kModuleExtraLedCount) return 0;
        if (col < 0 || col > 1) return 0;
        int row = is_send ? (col == 0 ? 0 : 1) : (col == 0 ? 2 : 3);

        void* ptr = c->module_frame_links[module_idx].ptrs[static_cast<size_t>(row)][static_cast<size_t>(led_idx)];
        if (!ptr) return 0;

        // Remove any action_port_bindings entries that reference this ptr
        auto *pa = reinterpret_cast<GP_CanvasContextImpl::PendingAction*>(ptr);
        if (pa) {
            int32_t aid = pa->action_id;
            std::lock_guard<std::mutex> lk(c->action_subscribers_mu);
            auto it = c->action_port_bindings.find(aid);
            if (it != c->action_port_bindings.end()) {
                auto &vec = it->second;
                for (auto vit = vec.begin(); vit != vec.end();) {
                    if (vit->module_idx == module_idx && vit->row == row && vit->led_idx == led_idx && vit->pending_ptr == ptr) {
                        vit = vec.erase(vit);
                    } else ++vit;
                }
                if (vec.empty()) c->action_port_bindings.erase(it);
            }
        }

        // Clear the frame ptr and visual markers
        c->module_frame_links[module_idx].ptrs[static_cast<size_t>(row)][static_cast<size_t>(led_idx)] = nullptr;
        GP_TableCell* cell = module_frame_led_cell(c, module_idx, row, led_idx);
        if (cell) cell->reserved0 = 0;

        if (module_idx < static_cast<int>(c->module_tables.size())) {
            GP_TableContext* t = c->module_tables[module_idx];
            if (t && module_idx < static_cast<int>(c->module_table_rows.size()) && !c->module_table_rows[module_idx].empty()) {
                const auto &rows = c->module_table_rows[module_idx];
                int target_col = is_send ? kModuleColRightLed : kModuleColLeftLed;
                for (int ri = 0; ri < static_cast<int>(rows.size()); ++ri) {
                    const auto &meta = rows[ri];
                    if ((is_send && meta.kind != ModuleRowKind::Output) || (!is_send && meta.kind != ModuleRowKind::Input)) continue;
                    if (meta.attachment_count <= led_idx) continue;
                    (void)gp_table_set_led_glow(t, ri, target_col, led_idx, 0.0f);
                    (void)gp_table_set_led_selected(t, ri, target_col, led_idx, 0);
                }
            }
        }

        // Free the pending action pointer
        gp_canvas_free_pending_action(ctx_, ptr);
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
    // Schedule autosave after actions if policy==ACTION and action is whitelisted
    if (c->autosave_policy == GP_CanvasContextImpl::AUTOSAVE_ACTION && !c->autosave_path.empty()) {
        bool allowed = false;
        if (c->autosave_action_whitelist.empty()) allowed = true;
        else {
            for (int aid : c->autosave_action_whitelist) if (aid == pa->action_id) { allowed = true; break; }
        }
        if (allowed) {
            using namespace std::chrono;
            auto now = duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
            c->autosave_last_action_ts_ms.store(static_cast<long long>(now), std::memory_order_release);
            long long when = static_cast<long long>(now + static_cast<long long>(c->autosave_action_delay_s * 1000.0));
            c->autosave_save_request_when_ms.store(when, std::memory_order_release);
            c->autosave_save_requested.store(1, std::memory_order_release);
        }
    }
    return 1;
}

// Note: internal queued pending-action API removed. Events are published
// into root table FIFO edges via gp_edge_publish_ptr and consumed by the
// manager using normal FIFO pointer-mode semantics.

// Deliver a synthesized event to all ports bound to `action_id`.
static void canvas_dispatch_event_to_bound_ports(GP_CanvasContextImpl* c, int32_t action_id, int x, int y, bool down, bool up, float dx, float dy, int button, float scroll) {
    if (!c) return;
    auto is_valid_pending_ptr = [](void* p) -> bool {
        uintptr_t v = reinterpret_cast<uintptr_t>(p);
        if (v == 0 || v == static_cast<uintptr_t>(~0ULL)) return false;
        return (v & (alignof(void*) - 1)) == 0;
    };
    std::vector<GP_CanvasContextImpl::ActionBinding> copy;
    {
        std::lock_guard<std::mutex> lk(c->action_subscribers_mu);
        auto it = c->action_port_bindings.find(action_id);
        if (it == c->action_port_bindings.end()) return;
        copy = it->second; // shallow copy to avoid holding lock while invoking
    }
    GP_TableContext* root = canvas_ensure_root_table(c);
    for (const auto &b : copy) {
        if (!b.pending_ptr || !is_valid_pending_ptr(b.pending_ptr)) continue;
        if (b.module_idx < 0 || b.module_idx >= static_cast<int>(c->module_frame_links.size())) continue;
        if (b.led_idx < 0 || b.led_idx >= kModuleExtraLedCount) continue;
        if (b.row < 0 || b.row >= kModuleExtraLedRows) continue;
        void* live_ptr = c->module_frame_links[static_cast<size_t>(b.module_idx)].ptrs[static_cast<size_t>(b.row)][static_cast<size_t>(b.led_idx)];
        if (live_ptr != b.pending_ptr) continue; // stale binding; skip
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
        // carry motion deltas, button and scroll into pending action so manager can expose them
        copy_pa->dx = dx;
        copy_pa->dy = dy;
        copy_pa->button = button;
        copy_pa->scroll = scroll;
        // synthesize per-button masks and default device id if caller provided a single button index
        copy_pa->button_mask_down = down ? (button > 0 ? (1u << static_cast<uint32_t>(button)) : 0u) : 0u;
        copy_pa->button_mask_up = up ? (button > 0 ? (1u << static_cast<uint32_t>(button)) : 0u) : 0u;
        copy_pa->device_id = 0; // frontend may populate this later if device awareness is added
        // Publish a pointer-mode EventPayload into the root table FIFO for
        // this binding's edge so external tools can observe and route the event.
        if (root && b.root_edge_idx >= 0) {
            // Debug: report publish target
            fprintf(stderr, "[DBG] publish action=%d -> target_mod=%d row=%d led=%d root_edge=%d writer_key=%llu\n", action_id, b.module_idx, b.row, b.led_idx, b.root_edge_idx, (unsigned long long)b.writer_key);
            EventPayload* ep = new EventPayload{reinterpret_cast<void*>(copy_pa), b.module_idx, b.led_idx};
            int dropped = 0;
            (void)gp_edge_publish_ptr(root, b.root_edge_idx, b.writer_key, reinterpret_cast<void*>(ep), &dropped);
            if (dropped) fprintf(stderr, "[DBG] publish dropped=%d action=%d root_edge=%d\n", dropped, action_id, b.root_edge_idx);
        } else {
            // If no root FIFO exists for this binding, drop the copy to avoid leaks.
            fprintf(stderr, "[DBG] publish no-root action=%d -> mod=%d row=%d led=%d\n", action_id, b.module_idx, b.row, b.led_idx);
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

extern "C" void* gp_canvas_pop_managed_event_payload(GP_CanvasContext* ctx_, int module_idx, int led_idx) {
    if (!ctx_) return nullptr;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(led_idx));
    std::lock_guard<std::mutex> lk(c->action_subscribers_mu);
    auto it = c->managed_event_payloads.find(key);
    if (it == c->managed_event_payloads.end()) return nullptr;
    void* p = it->second;
    c->managed_event_payloads.erase(it);
    return p;
}

extern "C" int gp_canvas_stash_managed_event_payload(GP_CanvasContext* ctx_, int module_idx, int led_idx, void* payload) {
    if (!ctx_ || !payload) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(led_idx));
    std::lock_guard<std::mutex> lk(c->action_subscribers_mu);
    c->managed_event_payloads[key] = payload;
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
        c->autosave_path.clear(); c->autosave_interval_s = 0.0; c->autosave_accum_s = 0.0; c->autosave_policy = GP_CanvasContextImpl::AUTOSAVE_DISABLED; 
        stop_autosave_thread(c);
        return 1;
    }
    c->autosave_path = std::string(path);
    c->autosave_interval_s = interval_s;
    c->autosave_accum_s = 0.0;
    c->autosave_policy = GP_CanvasContextImpl::AUTOSAVE_TIMER;
    // initialize last-save signature after enabling autosave (best-effort)
    try { gp_canvas_update_last_save_signature(reinterpret_cast<GP_CanvasContext*>(c)); } catch (...) { }
    // ensure background worker is running for asynchronous saves
    ensure_autosave_thread_started(c);
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

extern "C" int gp_canvas_set_autosave_policy(GP_CanvasContext* ctx_, int policy) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (policy < GP_CanvasContextImpl::AUTOSAVE_DISABLED || policy > GP_CanvasContextImpl::AUTOSAVE_ACTION) return 0;
    c->autosave_policy = policy;
    if (policy == GP_CanvasContextImpl::AUTOSAVE_ACTION && !c->autosave_path.empty()) ensure_autosave_thread_started(c);
    if (policy == GP_CanvasContextImpl::AUTOSAVE_DISABLED) stop_autosave_thread(c);
    return 1;
}

extern "C" int gp_canvas_set_autosave_action_whitelist(GP_CanvasContext* ctx_, const int* action_ids, int count) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->autosave_action_whitelist.clear();
    if (!action_ids || count <= 0) return 1;
    for (int i = 0; i < count; ++i) c->autosave_action_whitelist.push_back(action_ids[i]);
    return 1;
}

extern "C" int gp_canvas_set_autosave_action_timing(GP_CanvasContext* ctx_, double delay_s, double cooldown_s) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->autosave_action_delay_s = std::max(0.0, delay_s);
    c->autosave_action_cooldown_s = std::max(0.0, cooldown_s);
    return 1;
}

extern "C" int gp_canvas_get_autosave_policy(GP_CanvasContext* ctx_, int* out_policy) {
    if (!ctx_ || !out_policy) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    *out_policy = c->autosave_policy;
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

extern "C" int gp_canvas_get_thread_manager_paused(GP_CanvasContext* ctx_, int* out_paused) {
    if (!ctx_ || !out_paused) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    *out_paused = c->thread_mgr_paused ? 1 : 0;
    return 1;
}

extern "C" int gp_canvas_set_thread_manager_paused(GP_CanvasContext* ctx_, int paused) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->thread_mgr_paused = (paused != 0);
    c->thread_mgr_delay_accum_s = c->thread_mgr_paused ? 0.0 : static_cast<double>(std::max(0, c->thread_mgr_delay_ms)) / 1000.0;
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
        GraphRuntime::NodeContract* nodeA = nullptr;
        GraphRuntime::NodeContract* nodeB = nullptr;
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
