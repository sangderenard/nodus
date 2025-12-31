static std::string gp_canvas_compute_signature(GP_CanvasContextImpl* cc) {
    std::ostringstream ss;
    ss << cc->modules.size() << ":";
    for (size_t i = 0; i < cc->modules.size(); ++i) {
        const auto &m = cc->modules[i];
        ss << m.x << ',' << m.y << ',' << m.w << ',' << m.h << ',';
        std::string lbl(m.label, m.label + sizeof(m.label));
        size_t z = lbl.find('\0'); if (z != std::string::npos) lbl.resize(z);
        ss << lbl.size() << ',';
    }
    ss << "|E:" << cc->edges.size() << ":";
    for (const auto &e : cc->edges) {
        ss << e.desc.a_module << ',' << e.desc.a_contact_idx << ',' << e.desc.b_module << ',' << e.desc.b_contact_idx << ',' << e.type_id << ',' << e.subgroup_flags << ';';
    }
    ss << "|O:" << cc->overlays.size();
    return ss.str();
}

extern "C" int gp_canvas_save_to_file(GP_CanvasContext* ctx_, const char* path) {
    #include "console_logger.h"
    #ifndef printf
    #define printf(...) CONSOLE_PRINTF(__VA_ARGS__)
    #endif
    #ifndef fprintf
    #define fprintf(file, ...) CONSOLE_PRINTF(__VA_ARGS__)
    #endif
    #include <sstream>
    if (!ctx_ || !path) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // quick-change signature to avoid serializing -> disk when nothing changed
    std::string sig = gp_canvas_compute_signature(c);
    // compare against active buffer slot
    int cur_idx = c->last_save_idx.load(std::memory_order_acquire);
    if (!c->last_save_buf[cur_idx].empty() && c->last_save_buf[cur_idx] == sig) {
        try { if (std::filesystem::exists(path)) return 1; } catch (...) { /* fallthrough */ }
    }
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
    // Update in-memory signature to reflect this successful save using atomic flip
    try {
        int other = (c->last_save_idx.load(std::memory_order_acquire) ^ 1);
        c->last_save_buf[other] = sig;
        c->last_save_idx.store(other, std::memory_order_release);
    } catch (...) { }
    return 1;
}

extern "C" int gp_canvas_update_last_save_signature(GP_CanvasContext* ctx_) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    std::string sig = gp_canvas_compute_signature(c);
    int other = (c->last_save_idx.load(std::memory_order_acquire) ^ 1);
    c->last_save_buf[other] = sig;
    c->last_save_idx.store(other, std::memory_order_release);
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
        auto find_meta_group_by_id = [&](GP_TableContext* tt, unsigned long long mgid) -> GP_MetaGroup* {
            if (!tt || mgid == 0ull) return nullptr;
            int mgcount = gp_table_get_meta_group_count(tt);
            for (int mgi = 0; mgi < mgcount; ++mgi) {
                GP_MetaGroup* mg2 = gp_table_get_meta_group(tt, mgi);
                unsigned long long curid = 0ull;
                gp_table_meta_get_id(tt, mg2, &curid);
                if (curid == mgid) return mg2;
            }
            return nullptr;
        };
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
                        bool allow_overlay_rope = (find_meta_group_by_id(t, ms.mgid) == nullptr);
                        if (t && gp_table_get_rope_id_count(t) > 0) {
                            allow_overlay_rope = false;
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
                            float sx_local = ox1 - table_off_x;
                            float sy_local = oy1 - table_off_y;
                            float fx_local = ox2 - table_off_x;
                            float fy_local = oy2 - table_off_y;
                            float plug_z = -10.0f;
                            int segs = (c->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : 2;
                            if (allow_overlay_rope) {
                                rope_idx = rope_sim_add_rope3(sim, sx_local, sy_local, plug_z, fx_local, fy_local, plug_z, segs, 0.0f);
                            }
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
                        bool allow_overlay_rope = (find_meta_group_by_id(rt, ms.mgid) == nullptr);
                        if (rt && gp_table_get_rope_id_count(rt) > 0) {
                            allow_overlay_rope = false;
                        }
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
                            int segs = (c->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : 2;
                            if (allow_overlay_rope) {
                                rope_idx = rope_sim_add_rope3(sim, sx_local, sy_local, plug_z, fx_local, fy_local, plug_z, segs, 0.0f);
                            }
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
        bool mg_exists = false;
        GP_MetaGroup* mg = find_meta_group_by_id(t, ms.mgid);
        if (mg) {
            mg_exists = true;
        } else {
            mg = gp_table_meta_create(t);
            if (!mg) continue;
            printf("gp_canvas_load_from_file: restoring canvas META_GROUP to table=%p mg=%p (module_idx=%d overlay_idx=%d)\n", (void*)t, (void*)mg, ms.module_idx, ms.meta_slot);
            fflush(stdout);
            gp_table_meta_set_confinement(t, mg, ms.confinement);
            gp_table_meta_set_id(t, mg, ms.mgid);
            gp_table_meta_set_lasso_fields(t, mg, ms.lasso_flags, ms.lasso_widget_type);
        }
        if (mg_exists) {
            gp_table_meta_set_overlay_keys(t, mg, ms.overlay_a, ms.overlay_b);
            int overlay_rope = -1;
            int overlay_vid = -1;
            gp_table_meta_get_dangling_rope_info(t, mg, &overlay_rope, &overlay_vid);
            if (overlay_rope >= 0 && (ms.overlay_a != 0ull || ms.overlay_b != 0ull)) {
                gp_canvas_attach_rope_to_overlay(reinterpret_cast<GP_CanvasContext*>(c), ms.overlay_a, ms.overlay_b, overlay_rope);
            }
            gp_canvas_set_overlay_meta(reinterpret_cast<GP_CanvasContext*>(c), ms.overlay_a, ms.overlay_b, t, reinterpret_cast<void*>(mg));
            continue;
        }
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
                int existing_rope = -1;
                int existing_vid = -1;
                gp_table_meta_get_dangling_rope_info(t, mg, &existing_rope, &existing_vid);
                if (existing_rope >= 0) {
                    gp_canvas_attach_rope_to_overlay(reinterpret_cast<GP_CanvasContext*>(c), pov->key_a, pov->key_b, existing_rope);
                } else if (sim && gp_table_get_rope_id_count(t) == 0) {
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
                    int segs = (c->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : 2;
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

        float spawn_x = 0.0f;
        float spawn_y = 0.0f;
        if (sim && first_rope >= 0 && first_vid >= 0) {
            int vc = rope_sim_get_vertex_count(sim, first_rope);
            if (vc > 0 && first_vid < vc) {
                std::vector<float> verts3(static_cast<size_t>(vc * 3));
                rope_sim_get_vertices3(sim, first_rope, verts3.data(), static_cast<int>(verts3.size()));
                spawn_x = verts3[static_cast<size_t>(first_vid) * 3 + 0];
                spawn_y = verts3[static_cast<size_t>(first_vid) * 3 + 1];
            }
        }
        canvas_finalize_lasso_meta_group(c, t, mg, first_rope, first_vid, saved_u, first_rope, spawn_x, spawn_y);
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

extern "C" int gp_canvas_export_module_library(GP_CanvasContext* ctx_, const char* path) {
    if (!ctx_ || !path) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);

    GP_ModuleLibrary library{};
    library.root_dir = gp_module_library_default_root();

    std::map<int, GP_ModuleLibraryTool> tool_registry_by_kind;
    std::map<std::string, GP_ModuleLibraryTool> plugin_registry_by_id;
    auto normalize_source = [&](const std::string& source_path) -> std::string {
        if (source_path.empty()) return {};
        namespace fs = std::filesystem;
        try {
            fs::path src = fs::path(source_path);
            if (!src.is_absolute()) return src.generic_string();
            fs::path root = fs::absolute(fs::path(library.root_dir));
            fs::path abs_src = fs::absolute(src);
            std::error_code ec;
            fs::path rel = fs::relative(abs_src, root, ec);
            if (!ec) {
                std::string rel_str = rel.generic_string();
                if (!rel_str.empty() && rel_str.rfind("..", 0) != 0) {
                    return rel_str;
                }
            }
            return abs_src.generic_string();
        } catch (...) {
            return source_path;
        }
    };

    for (size_t i = 0; i < c->modules.size(); ++i) {
        const auto &mod = c->modules[i];
        GP_ModuleLibraryModule module{};
        module.module_idx = static_cast<int>(i);
        module.id = gp_module_library_module_id(module.module_idx);
        module.label = canvas_module_label(mod);
        // Use relative paths (no root) so actualizer will join with the chosen root
        module.serialized_path = gp_module_library_module_serialized_path(std::string(), module.id);
        module.source_path = gp_module_library_module_source_path(std::string(), module.id);
        module.input_count = 0;
        module.output_count = 0;

        if (i < c->module_io_rows.size()) {
            const auto &rows = c->module_io_rows[i];
            for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
                const auto &row = rows[row_idx];
                if (row.kind == ModuleRowKind::Input) {
                    module.input_count += std::clamp(row.attachment_count, 1, 32);
                    continue;
                }
                if (row.kind == ModuleRowKind::Output) {
                    module.output_count += std::clamp(row.attachment_count, 1, 32);
                    continue;
                }
                if (row.kind != ModuleRowKind::Tool) continue;
                if (row.tool_origin == ModuleToolOrigin::Builtin) {
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
                } else if (row.tool_origin == ModuleToolOrigin::Plugin && !row.plugin_id.empty()) {
                    if (plugin_registry_by_id.find(row.plugin_id) == plugin_registry_by_id.end()) {
                        GP_ModuleLibraryTool tool{};
                        tool.kind = ModuleToolKind::None;
                        tool.id = row.plugin_id;
                        const auto *entry = tool_registry_global().find(row.plugin_id);
                        tool.name = entry ? entry->name : row.plugin_id;
                        if (entry) tool.source_path = normalize_source(entry->source_path);
                        plugin_registry_by_id.emplace(tool.id, std::move(tool));
                    }
                    GP_ModuleToolInstance instance{};
                    instance.row_idx = static_cast<int>(row_idx);
                    instance.attachment_count = row.attachment_count;
                    instance.tool_id = row.plugin_id;
                    module.tool_instances.push_back(std::move(instance));
                }
            }
        }
        library.modules.push_back(std::move(module));
    }

    for (auto &entry : tool_registry_by_kind) {
        library.tool_registry.push_back(std::move(entry.second));
    }
    for (auto &entry : plugin_registry_by_id) {
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
    std::map<std::string, GP_ModuleLibraryTool> plugin_registry_by_id;
    auto normalize_source = [&](const std::string& source_path) -> std::string {
        if (source_path.empty()) return {};
        namespace fs = std::filesystem;
        try {
            fs::path src = fs::path(source_path);
            if (!src.is_absolute()) return src.generic_string();
            fs::path root = fs::absolute(fs::path(library.root_dir));
            fs::path abs_src = fs::absolute(src);
            std::error_code ec;
            fs::path rel = fs::relative(abs_src, root, ec);
            if (!ec) {
                std::string rel_str = rel.generic_string();
                if (!rel_str.empty() && rel_str.rfind("..", 0) != 0) {
                    return rel_str;
                }
            }
            return abs_src.generic_string();
        } catch (...) {
            return source_path;
        }
    };

    for (size_t i = 0; i < c->modules.size(); ++i) {
        const auto &mod = c->modules[i];
        GP_ModuleLibraryModule module{};
        module.module_idx = static_cast<int>(i);
        module.id = gp_module_library_module_id(module.module_idx);
        module.label = canvas_module_label(mod);
        // Use relative paths (no root) so actualizer will join with the chosen root
        module.serialized_path = gp_module_library_module_serialized_path(std::string(), module.id);
        module.source_path = gp_module_library_module_source_path(std::string(), module.id);
        module.input_count = 0;
        module.output_count = 0;

        if (i < c->module_io_rows.size()) {
            const auto &rows = c->module_io_rows[i];
            for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
                const auto &row = rows[row_idx];
                if (row.kind == ModuleRowKind::Input) {
                    module.input_count += std::clamp(row.attachment_count, 1, 32);
                    continue;
                }
                if (row.kind == ModuleRowKind::Output) {
                    module.output_count += std::clamp(row.attachment_count, 1, 32);
                    continue;
                }
                if (row.kind != ModuleRowKind::Tool) continue;
                if (row.tool_origin == ModuleToolOrigin::Builtin) {
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
                } else if (row.tool_origin == ModuleToolOrigin::Plugin && !row.plugin_id.empty()) {
                    if (plugin_registry_by_id.find(row.plugin_id) == plugin_registry_by_id.end()) {
                        GP_ModuleLibraryTool tool{};
                        tool.kind = ModuleToolKind::None;
                        tool.id = row.plugin_id;
                        const auto *entry = tool_registry_global().find(row.plugin_id);
                        tool.name = entry ? entry->name : row.plugin_id;
                        if (entry) tool.source_path = normalize_source(entry->source_path);
                        plugin_registry_by_id.emplace(tool.id, std::move(tool));
                    }
                    GP_ModuleToolInstance instance{};
                    instance.row_idx = static_cast<int>(row_idx);
                    instance.attachment_count = row.attachment_count;
                    instance.tool_id = row.plugin_id;
                    module.tool_instances.push_back(std::move(instance));
                }
            }
        }
        library.modules.push_back(std::move(module));
    }

    for (auto &entry : tool_registry_by_kind) {
        library.tool_registry.push_back(std::move(entry.second));
    }
    for (auto &entry : plugin_registry_by_id) {
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

extern "C" int gp_canvas_set_debug_flags(GP_CanvasContext* ctx_, uint32_t flags) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->debug_flags = flags;
    if (c->container_table) gp_table_set_debug_flags(c->container_table, flags);
    for (GP_TableContext* t : c->module_tables) {
        if (t) gp_table_set_debug_flags(t, flags);
    }
    return 1;
}

extern "C" int gp_canvas_get_debug_flags(GP_CanvasContext* ctx_, uint32_t* out_flags) {
    if (!ctx_ || !out_flags) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    *out_flags = c->debug_flags;
    return 1;
}

extern "C" GP_CanvasContext* gp_canvas_get_singleton() {
    return reinterpret_cast<GP_CanvasContext*>(g_canvas_context_singleton);
}

