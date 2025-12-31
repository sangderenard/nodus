// Deferred console logging
#include "console_logger.h"
#ifndef printf
#define printf(...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
#ifndef fprintf
#define fprintf(file, ...) CONSOLE_PRINTF(__VA_ARGS__)
#endif

extern "C" int gp_canvas_create_overlay_with_leds(GP_CanvasContext* ctx_, float x1, float y1, float x2, float y2, unsigned long long* out_key_a, unsigned long long* out_key_b) {
    if (!ctx_ || !out_key_a || !out_key_b) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // Always create a new overlay entry for explicit API calls.
    // Do NOT use spatial heuristics for reuse; callers must rely on
    // saved overlay keys/ids for canonical restoration.
    int oid = c->next_overlay_id++;
    GP_CanvasContextImpl::OverlayEntry ov{};
    ov.id = oid;
    ov.x1 = x1; ov.y1 = y1; ov.x2 = x2; ov.y2 = y2;
    // encode keys: sentinel module id 0xFFFFFFFF, contact = overlay id, led index in low 16
    unsigned long long base = (static_cast<unsigned long long>(0xFFFFFFFFu) << 32) | (static_cast<unsigned long long>(static_cast<uint32_t>(oid)) << 16);
    ov.key_a = base | static_cast<unsigned long long>(0);
    ov.key_b = base | static_cast<unsigned long long>(1);
        // assign stable per-port UUIDs
        ov.port_uuid_a = gp_canvas_generate_id(ctx_, 0ull);
        ov.port_uuid_b = gp_canvas_generate_id(ctx_, 0ull);
    c->overlays[oid] = ov;
    *out_key_a = ov.key_a;
    *out_key_b = ov.key_b;
    c->overlay_key_map[ov.key_a] = ov.id;
    c->overlay_key_map[ov.key_b] = ov.id;
        // register port UUIDs
        if (ov.port_uuid_a) c->overlay_port_uuid_map[ov.port_uuid_a] = std::make_pair(ov.id, 0);
        if (ov.port_uuid_b) c->overlay_port_uuid_map[ov.port_uuid_b] = std::make_pair(ov.id, 1);
    // If any pending canvas meta-group snapshots refer to this overlay id,
    // apply them now so meta-groups restored from file become visible.
    for (auto it = c->pending_meta_snapshots.begin(); it != c->pending_meta_snapshots.end(); ) {
        const CanvasMetaSnapshot &ms = *it;
        bool matches = false;
        if (ms.module_idx < 0 && ms.meta_slot == oid) matches = true;
        // also match by overlay keys if present
        if (!matches && (ms.overlay_a != 0ull || ms.overlay_b != 0ull)) {
            if (ms.overlay_a == ov.key_a || ms.overlay_a == ov.key_b || ms.overlay_b == ov.key_a || ms.overlay_b == ov.key_b) matches = true;
        }
        if (matches) {
            // attempt to resolve backing table: prefer meta_table cached in overlay (none yet)
            GP_TableContext* t = nullptr;
            if (ms.module_idx >= 0 && ms.module_idx < static_cast<int>(c->module_tables.size())) t = c->module_tables[ms.module_idx];
            if (!t) {
                // try to find table by scanning module_tables for a meta-group with matching id
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
            }
            if (!t) {
                printf("pending_meta_snapshot: no backing table found for snapshot module_idx=%d meta_slot=%d mgid=%llu overlay_a=%llu overlay_b=%llu\n",
                    ms.module_idx, ms.meta_slot, (unsigned long long)ms.mgid, (unsigned long long)ms.overlay_a, (unsigned long long)ms.overlay_b);
                fflush(stdout);
            }
            if (t) {
                // Check if a meta-group with the same persistent id already
                // exists in the table to avoid creating duplicates during
                // deserialize paths that may run more than once for the same
                // saved snapshot (canvas-level loader + pending snapshot apply).
                GP_MetaGroup* mg = nullptr;
                if (ms.mgid != 0ull) {
                    int mgcount = gp_table_get_meta_group_count(t);
                    for (int mgi = 0; mgi < mgcount; ++mgi) {
                        GP_MetaGroup* cand = gp_table_get_meta_group(t, mgi);
                        if (!cand) continue;
                        unsigned long long curid = 0ull; gp_table_meta_get_id(t, cand, &curid);
                        if (curid == ms.mgid) { mg = cand; break; }
                    }
                }
                if (!mg) {
                    mg = gp_table_meta_create(t);
                    if (!mg) {
                        printf("pending_meta_snapshot: gp_table_meta_create failed for table=%p mgid=%llu\n", (void*)t, (unsigned long long)ms.mgid);
                        fflush(stdout);
                    }
                    if (mg) {
                        gp_table_meta_set_confinement(t, mg, ms.confinement);
                        gp_table_meta_set_id(t, mg, ms.mgid);
                        gp_table_meta_set_lasso_fields(t, mg, ms.lasso_flags, ms.lasso_widget_type);
                        RopeSim* sim = gp_table_get_rope_sim(t);
                        if (!sim) {
                            RopeSim* rootsim = canvas_root_sim(c);
                            if (!rootsim) rootsim = canvas_require_root_sim(c);
                            if (rootsim) gp_table_attach_rope_sim(t, rootsim, 0);
                            sim = gp_table_get_rope_sim(t);
                        }
                        if (!sim) {
                            printf("pending_meta_snapshot: no RopeSim available for table=%p while restoring mgid=%llu\n", (void*)t, (unsigned long long)ms.mgid);
                            fflush(stdout);
                        } else {
                            if (ms.anchor_rope_id != 0ull) {
                                int anchor_idx = gp_table_resolve_rope_id_to_sim_index(t, ms.anchor_rope_id);
                                if (anchor_idx >= 0) gp_table_meta_set_anchor(t, mg, anchor_idx, ms.anchor_v);
                            }
                            for (const auto &vp : ms.verts) {
                                if (vp.rope_id == 0ull) continue;
                                int resolved = gp_table_resolve_rope_id_to_sim_index(t, vp.rope_id);
                                if (resolved >= 0) gp_table_meta_add_vertex(t, mg, resolved, vp.vertex_idx);
                            }
                        }
                        gp_table_meta_set_channel_group(t, mg, ms.channel_group);
                        gp_table_meta_set_overlay_keys(t, mg, ms.overlay_a, ms.overlay_b);
                        gp_table_meta_set_dangling_hang_len(t, mg, ms.dangling_len);
                        gp_table_meta_set_ring_mode(t, mg, ms.ring_mode);
                    }
                } else {
                    // existing meta-group found — ensure its overlay keys are up-to-date
                    gp_table_meta_set_overlay_keys(t, mg, ms.overlay_a, ms.overlay_b);
                }
                if (mg) {
                    // Ensure an overlay rope exists and is attached before binding
                    if ((ms.overlay_a != 0ull || ms.overlay_b != 0ull)) {
                        // find overlay entry by keys
                        int oid = -1; GP_CanvasContextImpl::OverlayEntry *pov = nullptr;
                        for (auto &pp : c->overlays) {
                            auto &ov = pp.second;
                            if (ov.key_a == ms.overlay_a || ov.key_b == ms.overlay_a || ov.key_a == ms.overlay_b || ov.key_b == ms.overlay_b) { oid = ov.id; pov = &ov; break; }
                        }
                        if (!pov) {
                            // if overlay slot provided, try that
                            if (ms.meta_slot >= 0) {
                                auto itov = c->overlays.find(ms.meta_slot);
                                if (itov != c->overlays.end()) { oid = itov->first; pov = &itov->second; }
                            }
                            if (!pov) {
                                printf("pending_meta_snapshot: no overlay entry found for keys %llu,%llu and meta_slot=%d (mgid=%llu)\n",
                                    (unsigned long long)ms.overlay_a, (unsigned long long)ms.overlay_b, ms.meta_slot, (unsigned long long)ms.mgid);
                                fflush(stdout);
                            }
                        }
                        if (pov) {
                            // create rope in table's sim using the same placement logic as lasso
                            RopeSim* sim = gp_table_get_rope_sim(t);
                            if (!sim) {
                                RopeSim* rootsim = canvas_root_sim(c);
                                if (!rootsim) rootsim = canvas_require_root_sim(c);
                                if (rootsim) gp_table_attach_rope_sim(t, rootsim, 0);
                                sim = gp_table_get_rope_sim(t);
                            }
                            if (!sim) {
                                printf("pending_meta_snapshot: no RopeSim available for table=%p overlay_id=%d mgid=%llu\n", (void*)t, pov->id, (unsigned long long)ms.mgid);
                                fflush(stdout);
                            }
                            if (sim) {
                                // compute table-local offset for host module if present
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
                                    // attach root/canonical edge and map overlay keys
                                    gp_canvas_attach_rope_to_overlay(reinterpret_cast<GP_CanvasContext*>(c), pov->key_a, pov->key_b, rope_idx);
                                }
                            }
                        }
                    }
                    // bind overlay
                    gp_canvas_set_overlay_meta(reinterpret_cast<GP_CanvasContext*>(c), ms.overlay_a, ms.overlay_b, t, reinterpret_cast<void*>(mg));
                    // After binding, finalize meta-group (widget, springs, ring)
                    if (ms.ring_mode != 0) {
                        int first_rope = -1;
                        int first_vid = -1;
                        if (ms.anchor_rope_id != 0ull) {
                            int resolved = gp_table_resolve_rope_id_to_sim_index(t, ms.anchor_rope_id);
                            if (resolved >= 0) { first_rope = resolved; first_vid = ms.anchor_v; }
                        }
                        if (first_rope < 0 && !ms.verts.empty()) {
                            for (const auto &vp : ms.verts) {
                                if (vp.rope_id == 0ull) continue;
                                int resolved = gp_table_resolve_rope_id_to_sim_index(t, vp.rope_id);
                                if (resolved >= 0) {
                                    first_rope = resolved;
                                    first_vid = vp.vertex_idx;
                                    break;
                                }
                            }
                        }
                        if (first_rope < 0 && (ms.overlay_a != 0ull || ms.overlay_b != 0ull)) {
                            for (size_t ei = 0; ei < c->edges.size(); ++ei) {
                                const auto &e = c->edges[ei];
                                if ((e.overlay_key_a && (e.overlay_key_a == ms.overlay_a || e.overlay_key_a == ms.overlay_b)) ||
                                    (e.overlay_key_b && (e.overlay_key_b == ms.overlay_a || e.overlay_key_b == ms.overlay_b))) {
                                    if (e.rope_idx >= 0) { first_rope = e.rope_idx; break; }
                                }
                            }
                        }
                        if (first_rope >= 0 && first_vid >= 0) {
                            RopeSim* sim = gp_table_get_rope_sim(t);
                            float spawn_x = 0.0f;
                            float spawn_y = 0.0f;
                            if (sim) {
                                int vc = rope_sim_get_vertex_count(sim, first_rope);
                                if (vc > 0 && first_vid < vc) {
                                    std::vector<float> verts3(static_cast<size_t>(vc * 3));
                                    rope_sim_get_vertices3(sim, first_rope, verts3.data(), static_cast<int>(verts3.size()));
                                    spawn_x = verts3[static_cast<size_t>(first_vid) * 3 + 0];
                                    spawn_y = verts3[static_cast<size_t>(first_vid) * 3 + 1];
                                }
                            }
                            canvas_finalize_lasso_meta_group(c, t, mg, first_rope, first_vid, ms.ring_u, first_rope, spawn_x, spawn_y);
                        } else {
                            // couldn't find a rope/vertex to finalize ring registration
                            printf("pending_meta_snapshot: cannot finalize ring for mg=%p ring_mode=%d first_rope=%d first_vid=%d overlay_a=%llu overlay_b=%llu mgid=%llu\n",
                                (void*)mg, ms.ring_mode, first_rope, first_vid, (unsigned long long)ms.overlay_a, (unsigned long long)ms.overlay_b, (unsigned long long)ms.mgid);
                            fflush(stdout);
                        }
                        // Deliver saved lasso points (reconstructed from saved vertex refs)
                        if (c->lasso_cb && !ms.verts.empty()) {
                            // compute table-local offset for host module if present
                            float table_off_x = 0.0f;
                            float table_off_y = 0.0f;
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
                            RopeSim* sim = gp_table_get_rope_sim(t);
                            std::unordered_map<int, std::vector<float>> proj_cache;
                            std::vector<float> pts;
                            pts.reserve(ms.verts.size() * 2);
                            for (const auto &vp : ms.verts) {
                                if (vp.rope_id == 0ull) continue;
                                int rope_idx = gp_table_resolve_rope_id_to_sim_index(t, vp.rope_id);
                                int vid = vp.vertex_idx;
                                if (rope_idx < 0 || vid < 0) continue;
                                auto itp = proj_cache.find(rope_idx);
                                if (itp == proj_cache.end()) {
                                    int vc = 0;
                                    if (sim) vc = rope_sim_get_vertex_count(sim, rope_idx);
                                    if (vc > 0) {
                                        std::vector<float> proj(static_cast<size_t>(vc * 2));
                                        int got = gp_table_get_projected_rope_vertices(t, rope_idx, proj.data(), static_cast<int>(proj.size()));
                                        if (got == vc) proj_cache[rope_idx] = std::move(proj);
                                        else proj_cache[rope_idx] = std::vector<float>();
                                    } else {
                                        proj_cache[rope_idx] = std::vector<float>();
                                    }
                                    itp = proj_cache.find(rope_idx);
                                }
                                const auto &proj = itp->second;
                                if (proj.empty()) continue;
                                if (vid >= 0 && vid < static_cast<int>(proj.size() / 2)) {
                                    float px = proj[static_cast<size_t>(vid) * 2 + 0] + table_off_x;
                                    float py = proj[static_cast<size_t>(vid) * 2 + 1] + table_off_y;
                                    pts.push_back(px);
                                    pts.push_back(py);
                                }
                            }
                            if (!pts.empty()) {
                                int n = static_cast<int>(pts.size() / 2);
                                c->lasso_cb(c->lasso_cb_user, 3, pts.data(), n);
                            }
                        }
                    }
                }
            }
            it = c->pending_meta_snapshots.erase(it);
        } else ++it;
    }
    return 1;
}

// Register an overlay using persisted overlay keys during table deserialization.
// This ensures a single canonical overlay instance exists for the saved keys
// and avoids creating duplicate overlays via heuristic code paths.
extern "C" int gp_canvas_register_table_overlay(GP_CanvasContext* ctx_, unsigned long long key_a, unsigned long long key_b, uint64_t port_uuid_a, uint64_t port_uuid_b) {
    if (!ctx_ || (key_a == 0ull && key_b == 0ull)) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // If either key already maps to an overlay, nothing to do.
    if (key_a && c->overlay_key_map.find(key_a) != c->overlay_key_map.end()) return 1;
    if (key_b && c->overlay_key_map.find(key_b) != c->overlay_key_map.end()) return 1;
    // Create a single overlay entry using the supplied keys and optional port UUIDs.
    int oid = c->next_overlay_id++;
    GP_CanvasContextImpl::OverlayEntry ov{};
    ov.id = oid;
    ov.key_a = key_a;
    ov.key_b = key_b;
    ov.x1 = ov.y1 = ov.x2 = ov.y2 = 0.0f;
    // Use supplied port UUIDs when non-zero, otherwise assign deterministic ones.
    if (port_uuid_a != 0ull) ov.port_uuid_a = port_uuid_a; else ov.port_uuid_a = gp_canvas_generate_id(reinterpret_cast<GP_CanvasContext*>(c), 0ull);
    if (port_uuid_b != 0ull) ov.port_uuid_b = port_uuid_b; else ov.port_uuid_b = gp_canvas_generate_id(reinterpret_cast<GP_CanvasContext*>(c), 0ull);
    c->overlays[static_cast<size_t>(oid)] = ov;
    if (ov.key_a) c->overlay_key_map[ov.key_a] = ov.id;
    if (ov.key_b) c->overlay_key_map[ov.key_b] = ov.id;
    if (ov.port_uuid_a) c->overlay_port_uuid_map[ov.port_uuid_a] = std::make_pair(ov.id, 0);
    if (ov.port_uuid_b) c->overlay_port_uuid_map[ov.port_uuid_b] = std::make_pair(ov.id, 1);
    return 1;
}

// Retrieve overlay port UUIDs for the given overlay keys.
extern "C" int gp_canvas_get_overlay_port_uuids(GP_CanvasContext* ctx_, unsigned long long key_a, unsigned long long key_b, uint64_t* out_port_a, uint64_t* out_port_b) {
    if (!ctx_ || (key_a == 0ull && key_b == 0ull)) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    int oid = -1;
    if (key_a) {
        auto it = c->overlay_key_map.find(key_a);
        if (it != c->overlay_key_map.end()) oid = it->second;
    }
    if (oid < 0 && key_b) {
        auto it = c->overlay_key_map.find(key_b);
        if (it != c->overlay_key_map.end()) oid = it->second;
    }
    if (oid < 0) return 0;
    if (static_cast<size_t>(oid) >= c->overlays.size()) return 0;
    const auto &ov = c->overlays[static_cast<size_t>(oid)];
    if (out_port_a) *out_port_a = ov.port_uuid_a;
    if (out_port_b) *out_port_b = ov.port_uuid_b;
    return 1;
}

extern "C" int gp_canvas_attach_rope_to_overlay(GP_CanvasContext* ctx_, unsigned long long key_a, unsigned long long key_b, int rope_idx) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // If an edge is already bound to this overlay, reuse it.
    for (size_t ei = 0; ei < c->edges.size(); ++ei) {
        auto &edge = c->edges[ei];
        bool match_a = (key_a != 0ull && (edge.overlay_key_a == key_a || edge.overlay_key_b == key_a));
        bool match_b = (key_b != 0ull && (edge.overlay_key_a == key_b || edge.overlay_key_b == key_b));
        if (match_a || match_b) {
            edge.overlay_key_a = key_a;
            edge.overlay_key_b = key_b;
            edge.rope_idx = rope_idx;
            printf("gp_canvas_attach_rope_to_overlay: reusing edge_idx=%zu for keys=%llu/%llu rope_idx=%d\n",
                   ei, (unsigned long long)key_a, (unsigned long long)key_b, rope_idx);
            return static_cast<int>(ei);
        }
    }
    // Ensure a root module exists to host canonical LED contacts for this overlay
    int root_mod = canvas_ensure_root_module(c);
    if (root_mod < 0) return 0;
    // If the canvas has not yet registered any rope ids (common during
    // initial restore), register rope_ids from existing module/root tables
    // so attachments can resolve persisted rope ids to runtime indices.
    if (c->rope_id_map.empty()) {
        // Use table API accessors (no internal struct access from canvas unit)
        for (size_t ti = 0; ti < c->module_tables.size(); ++ti) {
            GP_TableContext* tt = c->module_tables[ti];
            if (!tt) continue;
            int cnt = gp_table_get_rope_id_count(tt);
            if (cnt > 0) {
                std::vector<uint64_t> tmp; tmp.resize(static_cast<size_t>(cnt));
                int got = gp_table_get_rope_ids(tt, tmp.data(), cnt);
                if (got > 0) gp_canvas_register_table_rope_ids_from_array(reinterpret_cast<GP_CanvasContext*>(c), tt, tmp.data(), got);
            }
        }
        // Also register root/container table if present
        GP_TableContext* rt = canvas_ensure_root_table(c);
        if (rt) {
            int cnt = gp_table_get_rope_id_count(rt);
            if (cnt > 0) {
                std::vector<uint64_t> tmp; tmp.resize(static_cast<size_t>(cnt));
                int got = gp_table_get_rope_ids(rt, tmp.data(), cnt);
                if (got > 0) gp_canvas_register_table_rope_ids_from_array(reinterpret_cast<GP_CanvasContext*>(c), rt, tmp.data(), got);
            }
        }
    }
    // Choose two contact indices for the overlay endpoints. Use 0 and 1 by default.
    GP_CanvasEdgeDesc desc{};
    desc.a_module = root_mod; desc.a_contact_idx = 0;
    desc.b_module = root_mod; desc.b_contact_idx = 1;
    // Add a canvas edge which will also enqueue a root-table edge and setup FIFOs.
    int edge_idx = gp_canvas_add_edge_with_type(ctx_, &desc, /*type_id=*/0);
    if (edge_idx < 0) return 0;
    // Now embellish the created edge with overlay keys and rope mapping so
    // rendering and interaction resolve to the overlay positions while the
    // root table holds the canonical keys for network/connectivity.
    if (edge_idx >= static_cast<int>(c->edges.size())) return 0;
    auto &created = c->edges[static_cast<size_t>(edge_idx)];
    printf("gp_canvas_attach_rope_to_overlay: called key_a=%llu key_b=%llu rope_idx=%d edge_idx=%d table_root_mod=%d\n", (unsigned long long)key_a, (unsigned long long)key_b, rope_idx, edge_idx, root_mod);
    created.overlay_key_a = key_a;
    created.overlay_key_b = key_b;
    created.rope_idx = rope_idx;
    printf("  -> attached: edges.size=%zu overlays.size=%zu rope_id_map_size=%zu\n", c->edges.size(), c->overlays.size(), c->rope_id_map.size());
    // compute canonical root keys used by gp_canvas_add_edge_with_type above
    uint64_t rka = (static_cast<uint64_t>(static_cast<uint32_t>(root_mod)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(0)) << 16) | static_cast<uint64_t>(0);
    uint64_t rkb = (static_cast<uint64_t>(static_cast<uint32_t>(root_mod)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(1)) << 16) | static_cast<uint64_t>(0);
    c->canonical_to_overlay[rka] = key_a;
    c->canonical_to_overlay[rkb] = key_b;
    return edge_idx;
}

extern "C" int gp_canvas_attach_rope_to_overlay_with_module(GP_CanvasContext* ctx_, unsigned long long key_a, unsigned long long key_b, int rope_idx, int module_idx, int a_contact_idx, int b_contact_idx) {
    if (!ctx_) return -1;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // validate module indices
    if (module_idx < 0 || module_idx >= static_cast<int>(c->modules.size())) return -1;
    GP_CanvasEdgeDesc desc{};
    desc.a_module = module_idx; desc.a_contact_idx = a_contact_idx;
    desc.b_module = module_idx; desc.b_contact_idx = b_contact_idx;
    int edge_idx = gp_canvas_add_edge_with_type(ctx_, &desc, /*type_id=*/0);
    if (edge_idx < 0) return -1;
    if (edge_idx >= static_cast<int>(c->edges.size())) return -1;
    auto &created = c->edges[static_cast<size_t>(edge_idx)];
    created.overlay_key_a = key_a;
    created.overlay_key_b = key_b;
    created.rope_idx = rope_idx;
    uint64_t rka = (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(a_contact_idx)) << 16) | static_cast<uint64_t>(0);
    uint64_t rkb = (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(b_contact_idx)) << 16) | static_cast<uint64_t>(0);
    c->canonical_to_overlay[rka] = key_a;
    c->canonical_to_overlay[rkb] = key_b;
    return edge_idx;
}

// Resolve a canonical root key to overlay pixel coords if it maps to an overlay
extern "C" int gp_canvas_resolve_canonical_key(GP_CanvasContext* ctx_, unsigned long long key, int* out_x, int* out_y) {
    if (!ctx_ || !out_x || !out_y) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    auto it = c->canonical_to_overlay.find(key);
    if (it == c->canonical_to_overlay.end()) return 0;
    unsigned long long overlay_key = it->second;
    return gp_canvas_resolve_overlay_key(ctx_, overlay_key, out_x, out_y);
}

extern "C" int gp_canvas_resolve_overlay_key(GP_CanvasContext* ctx_, unsigned long long key, int* out_x, int* out_y) {
    if (!ctx_ || !out_x || !out_y) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    uint32_t r_orig = static_cast<uint32_t>(key >> 32);
    if (r_orig != 0xFFFFFFFFu) return 0;
    uint32_t c_idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
    uint32_t led = static_cast<uint32_t>(key & 0xFFFFu);
    auto it = c->overlays.find(static_cast<int>(c_idx));
    if (it == c->overlays.end()) return 0;
    const auto &ov = it->second;
    if (led == 0u) { *out_x = static_cast<int>(std::lround(ov.x1)); *out_y = static_cast<int>(std::lround(ov.y1)); }
    else { *out_x = static_cast<int>(std::lround(ov.x2)); *out_y = static_cast<int>(std::lround(ov.y2)); }
    return 1;
}

// Set an authoritative meta binding on an overlay (table + meta-group)
extern "C" int gp_canvas_set_overlay_meta(GP_CanvasContext* ctx_, unsigned long long key_a, unsigned long long key_b, GP_TableContext* table, void* meta_mg) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    auto set_for_key = [&](unsigned long long key)->int{
        if (!key) return 0;
        // Prefer direct key->overlay id mapping to avoid relying on encoded
        // id fields which may have been synthesized differently across
        // restore paths. This ensures we always bind to the unique overlay
        // instance owning these keys when present.
        auto itmap = c->overlay_key_map.find(key);
        GP_CanvasContextImpl::OverlayEntry* pov = nullptr;
        if (itmap != c->overlay_key_map.end()) {
            int found_id = itmap->second;
            auto it2 = c->overlays.find(found_id);
            if (it2 != c->overlays.end()) pov = &it2->second;
        }
        if (!pov) {
            uint32_t sentinel = static_cast<uint32_t>(key >> 32);
            if (sentinel != 0xFFFFFFFFu) return 0;
            uint32_t idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
            auto it = c->overlays.find(static_cast<int>(idx));
            if (it == c->overlays.end()) return 0;
            pov = &it->second;
        }
        auto &ov = *pov;
        // If overlay is already bound to the same table+meta-group, no-op
        if (ov.meta_table == table && ov.meta_mg == reinterpret_cast<GP_MetaGroup*>(meta_mg)) return 1;
        // Prevent rebinding an overlay to a different meta-group: prefer the
        // existing authoritative binding. If both meta-groups expose a
        // persistent mgid and they match, allow binding; otherwise refuse.
        if (ov.meta_mg && ov.meta_mg != reinterpret_cast<GP_MetaGroup*>(meta_mg)) {
            unsigned long long existing_mgid = 0ull;
            unsigned long long new_mgid = 0ull;
            if (ov.meta_table && ov.meta_mg) gp_table_meta_get_id(ov.meta_table, ov.meta_mg, &existing_mgid);
            if (table && meta_mg) gp_table_meta_get_id(table, reinterpret_cast<GP_MetaGroup*>(meta_mg), &new_mgid);
            if (existing_mgid != 0ull || new_mgid != 0ull) {
                if (existing_mgid == new_mgid) return 1; // same persistent mgid, treat as idempotent
            }
            printf("gp_canvas_set_overlay_meta: overlay id=%d already bound mg=%p (mgid=%llu) refusing rebinding to mg=%p (mgid=%llu)\n",
                ov.id, (void*)ov.meta_mg, (unsigned long long)existing_mgid, (void*)meta_mg, (unsigned long long)new_mgid);
            fflush(stdout);
            return 0;
        }
        printf("gp_canvas_set_overlay_meta: binding overlay id=%d to table=%p mg=%p\n", ov.id, (void*)table, (void*)meta_mg);
        fflush(stdout);
        // Hard-fail if the meta-group has no lasso membership or sim group
        if (meta_mg && table) {
            GP_MetaGroup* mgptr = reinterpret_cast<GP_MetaGroup*>(meta_mg);
            int vcount = gp_table_meta_get_vertex_count(table, mgptr);
            int sim_idx = -1; gp_table_meta_get_sim_group_index(table, mgptr, &sim_idx);
            if (vcount <= 0) {
                printf("FATAL: gp_canvas_set_overlay_meta: overlay id=%d binding to empty meta-group mg=%p (vcount=0) - aborting\n", ov.id, (void*)mgptr);
                fflush(stdout);
                std::abort();
            }
            if (sim_idx < 0) {
                printf("FATAL: gp_canvas_set_overlay_meta: overlay id=%d binding to meta-group mg=%p without sim_group (sim_idx=%d) - aborting\n", ov.id, (void*)mgptr, sim_idx);
                fflush(stdout);
                std::abort();
            }
        }
        ov.meta_table = table;
        ov.meta_mg = reinterpret_cast<GP_MetaGroup*>(meta_mg);
        // Also record overlay keys on the meta-group via table API so
        // serialization can persist the association without needing the
        // meta-group definition in this translation unit.
        if (meta_mg && table) {
            gp_table_meta_set_overlay_keys(table, reinterpret_cast<GP_MetaGroup*>(meta_mg), ov.key_a, ov.key_b);
        }
        return 1;
    };
    int a = set_for_key(key_a);
    int b = set_for_key(key_b);
    return (a || b) ? 1 : 0;
}

// Clear and remove all overlays and table meta-groups, then clear modules.
extern "C" int gp_canvas_clear_meta_and_overlays(GP_CanvasContext* ctx_) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);

    // 1) Destroy all meta-groups in the container table (if any)
    if (c->container_table) {
        GP_TableContext* t = c->container_table;
        int mgc = gp_table_get_meta_group_count(t);
        for (int i = mgc - 1; i >= 0; --i) {
            GP_MetaGroup* mg = gp_table_get_meta_group(t, i);
            if (mg) gp_table_meta_destroy(t, mg);
        }
    }

    // 2) Destroy all meta-groups in each module table
    for (size_t mi = 0; mi < c->module_tables.size(); ++mi) {
        GP_TableContext* t = c->module_tables[mi];
        if (!t) continue;
        int mgc = gp_table_get_meta_group_count(t);
        for (int i = mgc - 1; i >= 0; --i) {
            GP_MetaGroup* mg = gp_table_get_meta_group(t, i);
            if (mg) gp_table_meta_destroy(t, mg);
        }
    }

    // 3) Remove any canvas edges that reference overlays (erase from root table)
    for (int ei = static_cast<int>(c->edges.size()) - 1; ei >= 0; --ei) {
        const auto &e = c->edges[static_cast<size_t>(ei)];
        if (e.overlay_key_a || e.overlay_key_b) {
            canvas_remove_edge_at(c, ei);
        }
    }

    // 4) Clear overlay bindings on the canvas and inform tables (best-effort)
    for (auto &kv : c->overlays) {
        auto &ov = kv.second;
        if (ov.meta_table && ov.meta_mg) {
            // clear overlay keys recorded on the meta-group (if still present)
            gp_table_meta_set_overlay_keys(ov.meta_table, ov.meta_mg, 0ull, 0ull);
        }
    }
    c->overlays.clear();
    c->canonical_to_overlay.clear();
    c->overlay_key_map.clear();

    // 5) Clear pending snapshot queues
    c->pending_meta_snapshots.clear();
    c->post_load_meta_pending.clear();

    // 6) Clear modules/workspace (destroys tables/stages and remaining edges)
    canvas_clear_workspace(c);

    // reset overlay id generator
    c->next_overlay_id = 1;

    return 1;
}

extern "C" int gp_canvas_get_root_module_idx() {
    if (!g_canvas_context_singleton) return -1;
    return g_canvas_context_singleton->root_module_idx;
}

// Click-listen mode controls
extern "C" int gp_canvas_set_click_listen_mode(GP_CanvasContext* ctx_, int enable) {
    if (!ctx_) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    c->click_listen_mode = (enable != 0);
    return 1;
}

extern "C" int gp_canvas_get_click_listen_mode(GP_CanvasContext* ctx_, int* out_enabled) {
    if (!ctx_ || !out_enabled) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    *out_enabled = c->click_listen_mode ? 1 : 0;
    return 1;
}

// Bind the current pending action pointer into the most-left unused
// receive frame ptr for the module. Transfers ownership of the pending
// action into the module frame (does not copy).
extern "C" int gp_canvas_bind_pending_action_to_module(GP_CanvasContext* ctx_, int module_idx) {
    if (!ctx_ || module_idx < 0) return 0;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!c->pending_action) return 0;
    if (module_idx >= static_cast<int>(c->module_frame_links.size())) return 0;
    auto &links = c->module_frame_links[module_idx];
    // Search leftmost LED indices first. For each LED index, bind into the
    // full send/receive pair atomically. Prefer the left column pair
    // (send-left,row=0 and receive-left,row=2) then the right column pair
    // (send-right,row=1 and receive-right,row=3).
    for (int i = 0; i < kModuleExtraLedCount; ++i) {
        // left column pair
        bool left_free = (links.ptrs[0][static_cast<size_t>(i)] == nullptr) && (links.ptrs[2][static_cast<size_t>(i)] == nullptr);
        if (left_free) {
            void* p = reinterpret_cast<void*>(c->pending_action);
            links.ptrs[0][static_cast<size_t>(i)] = p; // send-left
            links.ptrs[2][static_cast<size_t>(i)] = p; // receive-left
            GP_TableCell* cell_send = module_frame_led_cell(c, module_idx, 0, i);
            GP_TableCell* cell_recv = module_frame_led_cell(c, module_idx, 2, i);
            if (cell_send) cell_send->reserved0 = 1;
            if (cell_recv) cell_recv->reserved0 = 1;
            c->pending_action = nullptr;
            printf("gp_canvas_bind_pending_action_to_module: bound pending action to module=%d left-pair idx=%d\n", module_idx, i);
            return 1;
        }
        // right column pair
        bool right_free = (links.ptrs[1][static_cast<size_t>(i)] == nullptr) && (links.ptrs[3][static_cast<size_t>(i)] == nullptr);
        if (right_free) {
            void* p = reinterpret_cast<void*>(c->pending_action);
            links.ptrs[1][static_cast<size_t>(i)] = p; // send-right
            links.ptrs[3][static_cast<size_t>(i)] = p; // receive-right
            GP_TableCell* cell_send = module_frame_led_cell(c, module_idx, 1, i);
            GP_TableCell* cell_recv = module_frame_led_cell(c, module_idx, 3, i);
            if (cell_send) cell_send->reserved0 = 1;
            if (cell_recv) cell_recv->reserved0 = 1;
            c->pending_action = nullptr;
            printf("gp_canvas_bind_pending_action_to_module: bound pending action to module=%d right-pair idx=%d\n", module_idx, i);
            return 1;
        }
    }
    return 0;
}

extern "C" void* gp_canvas_get_module_frame_ptr(GP_CanvasContext* ctx_, int module_idx, int is_send, int led_idx) {
    if (!ctx_) return nullptr;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->module_frame_links.size())) return nullptr;
    if (led_idx < 0 || led_idx >= kModuleExtraLedCount) return nullptr;
    int row = is_send ? 0 : 2; // map send->send-left, receive->receive-left
    return c->module_frame_links[module_idx].ptrs[static_cast<size_t>(row)][static_cast<size_t>(led_idx)];
}

// Convenience: get module frame pointer by contact index (contact space uses
// kModuleFrameContactBase + row * kModuleExtraLedCount + idx). This maps the
// contact index into the appropriate send/receive and left/right row index.
extern "C" void* gp_canvas_get_module_frame_ptr_for_contact(GP_CanvasContext* ctx_, int module_idx, int contact_idx) {
    if (!ctx_) return nullptr;
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx < 0 || module_idx >= static_cast<int>(c->module_frame_links.size())) return nullptr;
    int base = kModuleFrameContactBase;
    if (contact_idx < base) return nullptr;
    int local = contact_idx - base;
    const int full_cols = 2;
    int total = kModuleExtraLedCount * kModuleExtraLedRows;
    if (local < 0 || local >= total) return nullptr;
    int row = local / kModuleExtraLedCount; // logical row index 0..kModuleExtraLedRows-1
    int led_idx = local % kModuleExtraLedCount;
    int grid_row = row / full_cols; // 0 => send, 1 => receive
    int grid_col = row % full_cols; // 0 => left, 1 => right
    int row_index = 0;
    if (grid_row == 0) { // send
        row_index = (grid_col == 0) ? 0 : 1;
    } else { // receive
        row_index = (grid_col == 0) ? 2 : 3;
    }
    return c->module_frame_links[static_cast<size_t>(module_idx)].ptrs[static_cast<size_t>(row_index)][static_cast<size_t>(led_idx)];
}

extern "C" int gp_canvas_raster_rgba(GP_CanvasContext* ctx_, uint8_t* out_rgba, int32_t out_len_bytes) {
    if (!ctx_ || !out_rgba) return 0;
    auto *ctx = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    int w = ctx->width;
    int h = ctx->height;
    int pitch = w * 4;
    if (out_len_bytes < w * h * 4) return 0;

    update_canvas_scroll_state(ctx, /*pull_from_container=*/true);
    canvas_update_subgroup_palette(ctx);
    const uint32_t debug_flags = ctx->debug_flags;
    const bool simple_render = (debug_flags & GP_CANVAS_DEBUG_SIMPLE_RENDER) != 0u;
    const bool no_lighting = (debug_flags & GP_CANVAS_DEBUG_NO_LIGHTING) != 0u;
    const int simple_line_radius = 2;
    const int simple_dot_radius = std::max(3, simple_line_radius + 2);

    // clear
    memset(out_rgba, 0, static_cast<size_t>(w) * h * 4);
    // draw control bar at top with toggle tool buttons
    int rb = ctx->rope_bar_h;
    if (rb > 0) {
        // draw rope sim bar at very top
        memset_rect(out_rgba, w, h, pitch, 0, 0, w, rb, Color{22,22,28,255});
        // draw simple controls: segs +/- at left, slack +/- at right, and display values
        int bw = std::max(4, rb - 8);
        int spacing = 8;
        int byy = 4;
        int seg_minus_x = 8;
        int seg_plus_x = seg_minus_x + bw + spacing;
        int menu_w = std::max(bw, 36);
        int menu_x = seg_plus_x + bw + spacing;
        memset_rect(out_rgba, w, h, pitch, seg_minus_x, byy, bw, rb - 8, Color{60,60,72,255});
        memset_rect(out_rgba, w, h, pitch, seg_plus_x, byy, bw, rb - 8, Color{60,60,72,255});
        Color menu_fill = ctx->rope_menu_open ? Color{70,70,84,255} : Color{60,60,72,255};
        memset_rect(out_rgba, w, h, pitch, menu_x, byy, menu_w, rb - 8, menu_fill);
        // slack buttons on right
        int bx2 = w - 8 - bw*2 - spacing; memset_rect(out_rgba, w, h, pitch, bx2, byy, bw, rb - 8, Color{60,60,72,255}); bx2 += bw + spacing; memset_rect(out_rgba, w, h, pitch, bx2, byy, bw, rb - 8, Color{60,60,72,255});
        // value text (render_text_to_rgba is available)
        {
            std::string s = std::string(LABEL_ROPE_SEGS_PREFIX) + std::to_string(ctx->sim_segs) + LABEL_ROPE_SLACK_PREFIX + std::to_string(ctx->sim_slack);
            auto bm = render_text_to_rgba(s, 1.0f, {220,220,220,255});
            if (!bm.pixels.empty()) {
                int tx = (w - bm.width) / 2;
                int ty = (rb - bm.height) / 2;
                for (int yy = 0; yy < bm.height; ++yy) {
                    int dst_y = ty + yy;
                    if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < bm.width; ++xx) {
                        int dst_x = tx + xx;
                        if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
        }
        // render labels for the small rope-bar buttons (Seg -, Seg +, Slack -, Slack +)
        {
            int bw = std::max(4, rb - 8);
            int spacing = 8;
            int bx = 8;
            int byy = 4;
            std::vector<std::string> lbls = {LABEL_ROPE_SEGS_DEC, LABEL_ROPE_SEGS_INC};
            for (int i = 0; i < 2; ++i) {
                int bx_i = bx + i * (bw + spacing);
                auto tb = render_text_to_rgba(lbls[i], 0.9f, {230,230,230,255});
                if (!tb.pixels.empty()) {
                    int tx = bx_i + (bw - tb.width) / 2;
                    int ty = byy + (rb - 8 - tb.height) / 2;
                    for (int yy = 0; yy < tb.height; ++yy) {
                        int dst_y = ty + yy;
                        if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < tb.width; ++xx) {
                            int dst_x = tx + xx;
                            if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &tb.pixels[(yy * tb.width + xx) * 4];
                            float sa = src[3] / 255.0f;
                            if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) {
                                for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                dst[3] = 255;
                            }
                        }
                    }
                }
            }
            // rope sim menu button label
            {
                auto tb = render_text_to_rgba(LABEL_ROPE_MENU_BUTTON, 0.85f, {230,230,230,255});
                if (!tb.pixels.empty()) {
                    int tx = menu_x + (menu_w - tb.width) / 2;
                    int ty = byy + (rb - 8 - tb.height) / 2;
                    for (int yy = 0; yy < tb.height; ++yy) {
                        int dst_y = ty + yy;
                        if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < tb.width; ++xx) {
                            int dst_x = tx + xx;
                            if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &tb.pixels[(yy * tb.width + xx) * 4];
                            float sa = src[3] / 255.0f;
                            if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) {
                                for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                dst[3] = 255;
                            }
                        }
                    }
                }
            }
            int bx2 = w - 8 - bw*2 - spacing;
            std::vector<std::string> lbls2 = {LABEL_ROPE_SLACK_DEC, LABEL_ROPE_SLACK_INC};
            for (int i = 0; i < 2; ++i) {
                int bx_i = bx2 + i * (bw + spacing);
                auto tb = render_text_to_rgba(lbls2[i], 0.9f, {230,230,230,255});
                if (!tb.pixels.empty()) {
                    int tx = bx_i + (bw - tb.width) / 2;
                    int ty = byy + (rb - 8 - tb.height) / 2;
                    for (int yy = 0; yy < tb.height; ++yy) {
                        int dst_y = ty + yy;
                        if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < tb.width; ++xx) {
                            int dst_x = tx + xx;
                            if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &tb.pixels[(yy * tb.width + xx) * 4];
                            float sa = src[3] / 255.0f;
                            if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) {
                                for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                dst[3] = 255;
                            }
                        }
                    }
                }
            }
        }
    }
    int cbh = ctx->control_bar_h;
    if (cbh > 0) {
        memset_rect(out_rgba, w, h, pitch, 0, rb, w, cbh, Color{28,28,34,255});
        // two tool groups: canvas (left) and table (right)
        const int canvas_btn_count = 4;
        const int edge_btn_count = 5;
        const int save_btn_count = 2;
        const int table_btn_count = 3;
        const int kpn_btn_count = 3;
        const int spacing = 8;
        int inner_h = std::max(0, cbh - 8);
        int row_gap = 4;
        int row_h = std::max(4, (inner_h - row_gap) / 2);
        int by0 = rb + 4;
        int by1 = by0 + row_h + row_gap;
        int bh = row_h;
        int bw = bh; // square buttons
        // left (canvas) group
        int bx = 8; int by = by0;
        int canvas_group_w = canvas_btn_count * (bw + spacing) - spacing;
        for (int bi = 0; bi < canvas_btn_count; ++bi) {
            int bx_i = bx + bi * (bw + spacing);
            bool selected = (ctx->selected_tool_canvas == bi);
            Color fill = selected ? Color{90,90,110,255} : Color{60,60,72,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, by, bw, bh, fill);
            // left/right border
            for (int oy = 0; oy < bh; ++oy) {
                int y = by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
            // render canvas tool labels (full text) and a short letter inside the button
            const char* canvas_labels[4] = {
                LABEL_CANVAS_TOOL_SELECT,
                LABEL_CANVAS_TOOL_NEW_TABLE,
                LABEL_CANVAS_TOOL_EDGE_MODE,
                LABEL_CANVAS_TOOL_NEW_STAGE
            };
            auto lbm = render_text_to_rgba(canvas_labels[bi], 0.95f, {240,240,240,255});
            const char* canvas_short[4] = {
                LABEL_CANVAS_TOOL_SHORT_SELECT,
                LABEL_CANVAS_TOOL_SHORT_TABLE,
                LABEL_CANVAS_TOOL_SHORT_EDGE,
                LABEL_CANVAS_TOOL_SHORT_STAGE
            };
            if (!lbm.pixels.empty()) {
                int tx = bx_i + (bw - lbm.width) / 2;
                int ty = by + (bh - lbm.height) / 2;
                for (int yy = 0; yy < lbm.height; ++yy) {
                    int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < lbm.width; ++xx) {
                        int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &lbm.pixels[(yy * lbm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
            // small centered letter inside the square button for quick ID
            auto small = render_text_to_rgba(canvas_short[bi], 1.1f, {240,240,240,255});
            if (!small.pixels.empty()) {
                int txs = bx_i + (bw - small.width) / 2;
                int tys = by + (bh - small.height) / 2;
                for (int yy = 0; yy < small.height; ++yy) {
                    int dst_y = tys + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < small.width; ++xx) {
                        int dst_x = txs + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &small.pixels[(yy * small.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
        }
        // middle (edge) group
        int bx_edge = bx + canvas_group_w + spacing * 2;
        for (int bi = 0; bi < edge_btn_count; ++bi) {
            int bx_i = bx_edge + bi * (bw + spacing);
            bool selected = (bi < 4) ? (ctx->selected_tool_edge == bi) : ctx->lasso_mode;
            Color fill;
            if (bi < 4) fill = selected ? Color{90,80,70,255} : Color{60,54,48,255};
            else fill = selected ? Color{110,70,80,255} : Color{70,58,52,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, by, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
            const char* edge_labels[5] = {
                LABEL_EDGE_TOOL_CREATE,
                LABEL_EDGE_TOOL_DESTROY,
                LABEL_EDGE_TOOL_ON_CHANGE,
                LABEL_EDGE_TOOL_CONTINUOUS,
                LABEL_EDGE_TOOL_LASSO
            };
            auto lbm = render_text_to_rgba(edge_labels[bi], 0.85f, {235,228,218,255});
            const char* edge_short[5] = {
                LABEL_EDGE_TOOL_SHORT_CREATE,
                LABEL_EDGE_TOOL_SHORT_DESTROY,
                LABEL_EDGE_TOOL_SHORT_ON_CHANGE,
                LABEL_EDGE_TOOL_SHORT_CONTINUOUS,
                LABEL_EDGE_TOOL_SHORT_LASSO
            };
            if (!lbm.pixels.empty()) {
                int tx = bx_i + (bw - lbm.width) / 2;
                int ty = by + (bh - lbm.height) / 2;
                for (int yy = 0; yy < lbm.height; ++yy) {
                    int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < lbm.width; ++xx) {
                        int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &lbm.pixels[(yy * lbm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
            auto small = render_text_to_rgba(edge_short[bi], 1.05f, {235,228,218,255});
            if (!small.pixels.empty()) {
                int txs = bx_i + (bw - small.width) / 2;
                int tys = by + (bh - small.height) / 2;
                for (int yy = 0; yy < small.height; ++yy) {
                    int dst_y = tys + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < small.width; ++xx) {
                        int dst_x = txs + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &small.pixels[(yy * small.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
        }
        int group_width = table_btn_count * (bw + spacing) - spacing;
        int bx_r = std::max(8, w - 8 - group_width);
        int save_group_w = save_btn_count * (bw + spacing) - spacing;
        int bx_save = bx_r - save_group_w - spacing * 2;
        for (int bi = 0; bi < save_btn_count; ++bi) {
            int bx_i = bx_save + bi * (bw + spacing);
            Color fill = Color{50,62,54,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, by, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=36; pleft[1]=36; pleft[2]=40; pleft[3]=255;
                pright[0]=36; pright[1]=36; pright[2]=40; pright[3]=255;
            }
            const char* save_labels[2] = { LABEL_CANVAS_SAVE, LABEL_CANVAS_CLEAR };
            const char* save_short[2] = { LABEL_CANVAS_SAVE_SHORT, LABEL_CANVAS_CLEAR_SHORT };
            auto lbm = render_text_to_rgba(save_labels[bi], 0.85f, {225,235,228,255});
            if (!lbm.pixels.empty()) {
                int tx = bx_i + (bw - lbm.width) / 2;
                int ty = by + (bh - lbm.height) / 2;
                for (int yy = 0; yy < lbm.height; ++yy) {
                    int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < lbm.width; ++xx) {
                        int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &lbm.pixels[(yy * lbm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
            auto small = render_text_to_rgba(save_short[bi], 1.0f, {225,235,228,255});
            if (!small.pixels.empty()) {
                int txs = bx_i + (bw - small.width) / 2;
                int tys = by + (bh - small.height) / 2;
                for (int yy = 0; yy < small.height; ++yy) {
                    int dst_y = tys + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < small.width; ++xx) {
                        int dst_x = txs + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &small.pixels[(yy * small.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
        }
        // right (table) group
        for (int bi = 0; bi < table_btn_count; ++bi) {
            int bx_i = bx_r + bi * (bw + spacing);
            Color fill = (ctx->selected_tool_table == bi) ? Color{90,70,90,255} : Color{60,50,60,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, by, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
            // render table tool labels (full text) and a short letter inside the button
            const char* table_labels[3] = {
                LABEL_TABLE_TOOL_SELECT,
                LABEL_TABLE_TOOL_EDIT,
                LABEL_TABLE_TOOL_MORE
            };
            auto lbm2 = render_text_to_rgba(table_labels[bi], 0.85f, {230,220,240,255});
            const char* table_short[3] = {
                LABEL_TABLE_TOOL_SHORT_SELECT,
                LABEL_TABLE_TOOL_SHORT_EDIT,
                LABEL_TABLE_TOOL_SHORT_MORE
            };
            if (!lbm2.pixels.empty()) {
                int tx = bx_i + (bw - lbm2.width) / 2;
                int ty = by + (bh - lbm2.height) / 2;
                for (int yy = 0; yy < lbm2.height; ++yy) {
                    int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < lbm2.width; ++xx) {
                        int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &lbm2.pixels[(yy * lbm2.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
                    // small centered letter inside the square button for quick ID
                    auto small2 = render_text_to_rgba(table_short[bi], 1.0f, {230,220,240,255});
                    if (!small2.pixels.empty()) {
                        int txs = bx_i + (bw - small2.width) / 2;
                        int tys = by + (bh - small2.height) / 2;
                        for (int yy = 0; yy < small2.height; ++yy) {
                            int dst_y = tys + yy; if (dst_y < 0 || dst_y >= h) continue;
                            for (int xx = 0; xx < small2.width; ++xx) {
                                int dst_x = txs + xx; if (dst_x < 0 || dst_x >= w) continue;
                                uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                const unsigned char* src = &small2.pixels[(yy * small2.width + xx) * 4];
                                float sa = src[3] / 255.0f;
                                if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                else if (sa > 0.001f) {
                                    for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                    dst[3] = 255;
                                }
                            }
                        }
                    }
            }
        }
        // IO counter shown to the left of the table buttons
        auto draw_io_group = [&](int base_x, int byy, int in_count, const char* label) {
            // minus box, number area, plus box
            int nbw = bw;
            int num_w = std::max(24, nbw * 2);
            int gap = 10;
            int bx_minus = base_x - (nbw + gap + num_w + gap + nbw);
            int bx_num = bx_minus + nbw + gap;
            int bx_plus = bx_num + num_w + gap;
            // minus
            memset_rect(out_rgba, w, h, pitch, bx_minus, byy, nbw, bh, Color{50,50,56,255});
            // number background
            memset_rect(out_rgba, w, h, pitch, bx_num, byy, num_w, bh, Color{36,36,42,255});
            // plus
            memset_rect(out_rgba, w, h, pitch, bx_plus, byy, nbw, bh, Color{50,50,56,255});
            // render number text centered
            std::string s = std::to_string(in_count);
            auto bm = render_text_to_rgba(s, 1.0f, {255,255,255,255});
            if (!bm.pixels.empty()) {
                int tx = bx_num + (num_w - bm.width) / 2;
                int ty = byy + (bh - bm.height) / 2;
                for (int yy = 0; yy < bm.height; ++yy) {
                    int dst_y = ty + yy;
                    if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < bm.width; ++xx) {
                        int dst_x = tx + xx;
                        if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
            // tiny label: render above number area
            if (label && label[0] != '\0') {
                auto lb = render_text_to_rgba(label, 0.75f, {200,200,200,255});
                if (!lb.pixels.empty()) {
                    int tx = bx_num + (num_w - lb.width) / 2;
                    int ty = byy - lb.height - 2; // place slightly above number
                    for (int yy = 0; yy < lb.height; ++yy) {
                        int dst_y = ty + yy;
                        if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < lb.width; ++xx) {
                            int dst_x = tx + xx;
                            if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &lb.pixels[(yy * lb.width + xx) * 4];
                            float sa = src[3] / 255.0f;
                            if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) {
                                for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                dst[3] = 255;
                            }
                        }
                    }
                }
            }
        };
        auto draw_io_action_pair = [&](int left_x, int byy, const char* left_label, const char* right_label) {
            int nbw = bw;
            int gap = 10;
            int bx_left = left_x;
            int bx_right = bx_left + nbw + gap;
            memset_rect(out_rgba, w, h, pitch, bx_left, byy, nbw, bh, Color{50,50,56,255});
            memset_rect(out_rgba, w, h, pitch, bx_right, byy, nbw, bh, Color{50,50,56,255});
            auto draw_label = [&](int bx, const char* label) {
                auto bm = render_text_to_rgba(label, 0.9f, {230,230,235,255});
                if (!bm.pixels.empty()) {
                    int tx = bx + (nbw - bm.width) / 2;
                    int ty = byy + (bh - bm.height) / 2;
                    for (int yy = 0; yy < bm.height; ++yy) {
                        int dst_y = ty + yy;
                        if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < bm.width; ++xx) {
                            int dst_x = tx + xx;
                            if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                            float sa = src[3] / 255.0f;
                            if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) {
                                for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                dst[3] = 255;
                            }
                        }
                    }
                }
            };
            if (left_label && left_label[0] != '\0') draw_label(bx_left, left_label);
            if (right_label && right_label[0] != '\0') draw_label(bx_right, right_label);
        };
        auto draw_edge_order_group = [&](int left_x, int byy, int value, bool tool_active) {
            int nbw = bw;
            int num_w = std::max(28, nbw * 2);
            int gap = 10;
            int tool_gap = 10;
            int bx_down = left_x;
            int bx_num = bx_down + nbw + gap;
            int bx_up = bx_num + num_w + gap;
            int bx_tool = bx_up + nbw + tool_gap;
            memset_rect(out_rgba, w, h, pitch, bx_down, byy, nbw, bh, Color{50,50,56,255});
            memset_rect(out_rgba, w, h, pitch, bx_num, byy, num_w, bh, Color{36,36,42,255});
            memset_rect(out_rgba, w, h, pitch, bx_up, byy, nbw, bh, Color{50,50,56,255});
            Color tool_fill = tool_active ? Color{90,80,70,255} : Color{60,54,48,255};
            memset_rect(out_rgba, w, h, pitch, bx_tool, byy, nbw, bh, tool_fill);
            auto draw_label = [&](int bx, const char* label, float scale, Color col) {
                auto bm = render_text_to_rgba(label, scale, {col.r, col.g, col.b, col.a});
                if (!bm.pixels.empty()) {
                    int tx = bx + (nbw - bm.width) / 2;
                    int ty = byy + (bh - bm.height) / 2;
                    for (int yy = 0; yy < bm.height; ++yy) {
                        int dst_y = ty + yy;
                        if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < bm.width; ++xx) {
                            int dst_x = tx + xx;
                            if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                            float sa = src[3] / 255.0f;
                            if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) {
                                for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                dst[3] = 255;
                            }
                        }
                    }
                }
            };
            draw_label(bx_down, LABEL_IO_MINUS, 1.0f, {230,230,235,255});
            draw_label(bx_up, LABEL_IO_PLUS, 1.0f, {230,230,235,255});
            draw_label(bx_tool, LABEL_EDGE_ORDER_TOOL, 0.8f, {235,228,218,255});
            std::string s = (value >= 0) ? ("+" + std::to_string(value)) : std::to_string(value);
            auto bm = render_text_to_rgba(s, 0.95f, {255,255,255,255});
            if (!bm.pixels.empty()) {
                int tx = bx_num + (num_w - bm.width) / 2;
                int ty = byy + (bh - bm.height) / 2;
                for (int yy = 0; yy < bm.height; ++yy) {
                    int dst_y = ty + yy;
                    if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < bm.width; ++xx) {
                        int dst_x = tx + xx;
                        if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
        };
        auto draw_action_button = [&](int bx, int byy, const char* label, Color fill) {
            memset_rect(out_rgba, w, h, pitch, bx, byy, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = byy + oy; if (y < 0 || y >= h) continue;
                int left_x = bx; int right_x = bx + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
            auto text = render_text_to_rgba(label, 0.8f, {230,230,230,255});
            if (!text.pixels.empty()) {
                int tx = bx + (bw - text.width) / 2;
                int ty = byy + (bh - text.height) / 2;
                for (int yy = 0; yy < text.height; ++yy) {
                    int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < text.width; ++xx) {
                        int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &text.pixels[(yy * text.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
        };
        // compute left of table buttons start for groups placement
        int io_base_x = bx_save - spacing * 2; // place IO groups to the left of save/clear buttons
        int io_by = by;
        int counter_value = std::max(1, ctx->io_attachment_count);
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
        draw_edge_order_group(order_left_x, io_by, ctx->edge_order_value, ctx->edge_order_tool_active != 0);
        draw_io_action_pair(pair_left_x, io_by, LABEL_IO_CONSUMER_SHORT, LABEL_IO_PRODUCER_SHORT);
        draw_io_group(io_base_x, io_by, counter_value, LABEL_IO_COUNT_SHORT);

        // second row: KPN tools + thread manager controls
        int kpn_by = by1;
        int bx_kpn = 8;
        for (int bi = 0; bi < kpn_btn_count; ++bi) {
            int bx_i = bx_kpn + bi * (bw + spacing);
            bool selected = (ctx->selected_tool_kpn == bi);
            Color fill = selected ? Color{80,90,110,255} : Color{52,58,70,255};
            memset_rect(out_rgba, w, h, pitch, bx_i, kpn_by, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = kpn_by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
                pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
            }
            const char* kpn_labels[3] = {
                LABEL_KPN_TOOL_K,
                LABEL_KPN_TOOL_P,
                LABEL_KPN_TOOL_N
            };
            const char* kpn_short[3] = {
                LABEL_KPN_TOOL_SHORT_K,
                LABEL_KPN_TOOL_SHORT_P,
                LABEL_KPN_TOOL_SHORT_N
            };
            auto lbm = render_text_to_rgba(kpn_labels[bi], 0.8f, {220,228,240,255});
            if (!lbm.pixels.empty()) {
                int tx = bx_i + (bw - lbm.width) / 2;
                int ty = kpn_by + (bh - lbm.height) / 2;
                for (int yy = 0; yy < lbm.height; ++yy) {
                    int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < lbm.width; ++xx) {
                        int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &lbm.pixels[(yy * lbm.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
            auto small = render_text_to_rgba(kpn_short[bi], 1.0f, {235,238,245,255});
            if (!small.pixels.empty()) {
                int txs = bx_i + (bw - small.width) / 2;
                int tys = kpn_by + (bh - small.height) / 2;
                for (int yy = 0; yy < small.height; ++yy) {
                    int dst_y = tys + yy; if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < small.width; ++xx) {
                        int dst_x = txs + xx; if (dst_x < 0 || dst_x >= w) continue;
                        uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                        const unsigned char* src = &small.pixels[(yy * small.width + xx) * 4];
                        float sa = src[3] / 255.0f;
                        if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                        else if (sa > 0.001f) {
                            for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                            dst[3] = 255;
                        }
                    }
                }
            }
        }
        int kpn_group_w = kpn_btn_count * (bw + spacing) - spacing;
        int kpn_right = bx_kpn + kpn_group_w;
        int delay_num_w = std::max(24, nbw * 2);
        int delay_gap = 10;
        int delay_total_w = nbw + delay_gap + delay_num_w + delay_gap + nbw;
        int play_gap = 12;
        int bx_play = w - 8 - bw;
        int action_btn_gap = 6;
        int action_btn_count = 3;
        int action_group_w = action_btn_count * bw + action_btn_gap * (action_btn_count - 1);
        int bx_action_right = bx_play - play_gap;
        int bx_action_left = bx_action_right - action_group_w;
        int bx_delay_plus = bx_action_left - play_gap;
        int delay_left = bx_delay_plus - delay_total_w;
        int subgroup_btn_count = kSubgroupBinCount;
        int subgroup_group_w = subgroup_btn_count * (bw + spacing) - spacing;
        int subgroup_left = kpn_right + spacing * 2;
        int available = delay_left - subgroup_left;
        if (available < subgroup_group_w) {
            subgroup_left = std::max(kpn_right + spacing, delay_left - subgroup_group_w);
        }
        // reset toolbar LED hit list for this frame
        ctx->toolbar_leds.clear();
        for (int bi = 0; bi < subgroup_btn_count; ++bi) {
            int bx_i = subgroup_left + bi * (bw + spacing);
            uint32_t flags = subgroup_mask_for_index(bi);
            Color fill = subgroup_flags_to_color(ctx, flags, 255);
            bool selected = (ctx->selected_tool_subgroup_flags & subgroup_mask_for_index(bi)) != 0u;
            if (!selected) {
                fill.r = static_cast<uint8_t>(std::lround(fill.r * 0.75f));
                fill.g = static_cast<uint8_t>(std::lround(fill.g * 0.75f));
                fill.b = static_cast<uint8_t>(std::lround(fill.b * 0.75f));
            }
            memset_rect(out_rgba, w, h, pitch, bx_i, kpn_by, bw, bh, fill);
            for (int oy = 0; oy < bh; ++oy) {
                int y = kpn_by + oy; if (y < 0 || y >= h) continue;
                int left_x = bx_i; int right_x = bx_i + bw - 1;
                uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
                uint8_t* pright = out_rgba + y * pitch + right_x * 4;
                pleft[0]=selected ? 240 : 40; pleft[1]=selected ? 240 : 40; pleft[2]=selected ? 240 : 44; pleft[3]=255;
                pright[0]=selected ? 240 : 40; pright[1]=selected ? 240 : 40; pright[2]=selected ? 240 : 44; pright[3]=255;
            }
            Color led_target = unpack_rgba(ctx->subgroup_target_rgba[static_cast<size_t>(bi)].load(std::memory_order_acquire));
            int led_r = std::max(2, bw / 6);
            // shift LED right by roughly its own radius so it sits between buttons
            int led_cx = bx_i + bw + led_r;
            int led_cy = kpn_by + bh / 2;
            draw_circle(out_rgba, w, h, pitch, led_cx, led_cy, led_r + 1, Color{20,20,28,255});
            draw_circle(out_rgba, w, h, pitch, led_cx, led_cy, led_r, led_target);
            if (selected) {
                int y_top = kpn_by;
                int y_bot = kpn_by + bh - 1;
                for (int xx = 0; xx < bw; ++xx) {
                    int xh = bx_i + xx;
                    if (xh < 0 || xh >= w) continue;
                    if (y_top >= 0 && y_top < h) {
                        uint8_t* pt = out_rgba + y_top * pitch + xh * 4;
                        pt[0]=240; pt[1]=240; pt[2]=240; pt[3]=255;
                    }
                    if (y_bot >= 0 && y_bot < h) {
                        uint8_t* pb = out_rgba + y_bot * pitch + xh * 4;
                        pb[0]=240; pb[1]=240; pb[2]=240; pb[3]=255;
                    }
                }
            }
            // record toolbar LED hit rect (view coords and world coords)
            {
                GP_CanvasContextImpl::ToolbarLedBox tb;
                tb.x0 = led_cx - (led_r + 2);
                tb.y0 = led_cy - (led_r + 2);
                tb.x1 = led_cx + (led_r + 2);
                tb.y1 = led_cy + (led_r + 2);
                tb.wx0 = tb.x0 + ctx->offset_x; tb.wy0 = tb.y0 + ctx->offset_y;
                tb.wx1 = tb.x1 + ctx->offset_x; tb.wy1 = tb.y1 + ctx->offset_y;
                tb.subgroup_idx = bi;
                ctx->toolbar_leds.push_back(tb);
            }
        }
        // Spawn-root button and delay numeric control will be positioned
        // after the exec-mode/action group layout so they sit to the left
        // of the new buttons and avoid overlap.
        int bx_spawn = subgroup_left + subgroup_group_w + spacing;

        int bx_clone = bx_action_left;
        int bx_clear = bx_clone + bw + action_btn_gap;
        int bx_destroy = bx_clear + bw + action_btn_gap;
        draw_action_button(bx_clone, kpn_by, LABEL_MODULE_CLONE_SHORT, Color{58,58,70,255});
        draw_action_button(bx_clear, kpn_by, LABEL_MODULE_CLEAR_SHORT, Color{64,56,52,255});
        draw_action_button(bx_destroy, kpn_by, LABEL_MODULE_DESTROY_SHORT, Color{70,52,52,255});

        // Execution mode cluster (4 small buttons) to the left of the action group.
        // Use toolbar LED slots mapped into subgroup indices to receive clicks.
        {
            int mode_count = 4;
            int mode_w = bw;
            int mode_gap = 4;
            int total_w = mode_count * mode_w + (mode_count - 1) * mode_gap;
            int bx_mode_left = bx_action_left - total_w - action_btn_gap;
            // Draw global timing toggle button immediately left of the exec-mode cluster
            int bx_clock = bx_mode_left - mode_gap - mode_w;
            bool timing_on_global = false;
            if (ThreadManager::global()) timing_on_global = ThreadManager::global()->timing_enabled();
            Color clock_fill = timing_on_global ? Color{64,120,60,255} : Color{48,48,62,255};
            draw_action_button(bx_clock, kpn_by, "TIME", clock_fill);
            const char* mode_labels[4] = {"SEQ","POOL","SLIP","FREE"};
            for (int mi = 0; mi < mode_count; ++mi) {
                int bx_i = bx_mode_left + mi * (mode_w + mode_gap);
                bool selected = (ctx->thread_mgr_global_exec_mode == mi);
                Color fill = selected ? Color{80,90,110,255} : Color{48,48,62,255};
                draw_action_button(bx_i, kpn_by, mode_labels[mi], fill);
                if (selected) {
                    int y_top = kpn_by;
                    int y_bot = kpn_by + bh - 1;
                    for (int xx = 0; xx < mode_w; ++xx) {
                        int xh = bx_i + xx;
                        if (xh < 0 || xh >= w) continue;
                        if (y_top >= 0 && y_top < h) {
                            uint8_t* pt = out_rgba + y_top * pitch + xh * 4;
                            pt[0]=240; pt[1]=240; pt[2]=240; pt[3]=255;
                        }
                        if (y_bot >= 0 && y_bot < h) {
                            uint8_t* pb = out_rgba + y_bot * pitch + xh * 4;
                            pb[0]=240; pb[1]=240; pb[2]=240; pb[3]=255;
                        }
                    }
                }
                // register small toolbar LED hitbox so root-module hitboxes include these
                GP_CanvasContextImpl::ToolbarLedBox tb;
                int led_r = std::max(2, bw / 6);
                int led_cx = bx_i + mode_w / 2;
                int led_cy = kpn_by + bh / 2;
                tb.x0 = led_cx - (led_r + 2);
                tb.y0 = led_cy - (led_r + 2);
                tb.x1 = led_cx + (led_r + 2);
                tb.y1 = led_cy + (led_r + 2);
                tb.wx0 = tb.x0 + ctx->offset_x; tb.wy0 = tb.y0 + ctx->offset_y;
                tb.wx1 = tb.x1 + ctx->offset_x; tb.wy1 = tb.y1 + ctx->offset_y;
                // map these exec-mode buttons into toolbar subgroup slots starting at 8
                tb.subgroup_idx = 8 + mi;
                ctx->toolbar_leds.push_back(tb);
            }
        }

        // Now that the exec-mode/action group position is known, place the
        // numeric delay control to the left of the exec-mode cluster so it
        // does not get visually overlapped by the new buttons.
        {
            int mode_count = 4;
            int mode_w = bw;
            int mode_gap = 4;
            int total_w = mode_count * mode_w + (mode_count - 1) * mode_gap;
            int bx_mode_left = bx_action_left - total_w - action_btn_gap;
            int bx_delay_new = bx_mode_left - delay_total_w - action_btn_gap;
            // draw numeric delay control at new location
            draw_io_group(bx_delay_new, kpn_by, std::max(0, ctx->thread_mgr_delay_ms), LABEL_THREAD_DELAY_SHORT);
            // spawn root button sits to the right of subgroup colors but left of numeric control
            if (bx_spawn + bw < bx_delay_new) {
                draw_action_button(bx_spawn, kpn_by, "ROOT", Color{70,70,90,255});
            }
        }

        // Split the former play button into a small SIM button (left half)
        // and a reduced play button (right half).
        int half_play_w = std::max(8, bw / 2);
        int sim_btn_w = half_play_w;
        int play_btn_w = bw - sim_btn_w;
        int sim_x = bx_play;
        int play_x = bx_play + sim_btn_w;
        Color sim_fill = Color{46,46,56,255};
        memset_rect(out_rgba, w, h, pitch, sim_x, kpn_by, sim_btn_w, bh, sim_fill);
        Color play_fill = ctx->thread_mgr_paused ? Color{70,60,70,255} : Color{60,80,60,255};
        memset_rect(out_rgba, w, h, pitch, play_x, kpn_by, play_btn_w, bh, play_fill);
        for (int oy = 0; oy < bh; ++oy) {
            int y = kpn_by + oy; if (y < 0 || y >= h) continue;
            int left_x = play_x; int right_x = play_x + play_btn_w - 1;
            uint8_t* pleft = out_rgba + y * pitch + left_x * 4;
            uint8_t* pright = out_rgba + y * pitch + right_x * 4;
            pleft[0]=40; pleft[1]=40; pleft[2]=44; pleft[3]=255;
            pright[0]=40; pright[1]=40; pright[2]=44; pright[3]=255;
        }
        const char* play_label = ctx->thread_mgr_paused ? LABEL_THREAD_PLAY : LABEL_THREAD_PAUSE;
        const char* play_short = ctx->thread_mgr_paused ? LABEL_THREAD_PLAY_SHORT : LABEL_THREAD_PAUSE_SHORT;
        auto play_text = render_text_to_rgba(play_label, 0.75f, {230,230,230,255});
        if (!play_text.pixels.empty()) {
            int tx = play_x + (play_btn_w - play_text.width) / 2;
            int ty = kpn_by + (bh - play_text.height) / 2;
            for (int yy = 0; yy < play_text.height; ++yy) {
                int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < play_text.width; ++xx) {
                    int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &play_text.pixels[(yy * play_text.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
        auto play_short_text = render_text_to_rgba(play_short, 1.0f, {240,240,240,255});
        if (!play_short_text.pixels.empty()) {
            int txs = play_x + (play_btn_w - play_short_text.width) / 2;
            int tys = kpn_by + (bh - play_short_text.height) / 2;
            for (int yy = 0; yy < play_short_text.height; ++yy) {
                int dst_y = tys + yy; if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < play_short_text.width; ++xx) {
                    int dst_x = txs + xx; if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &play_short_text.pixels[(yy * play_short_text.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
        // Draw SIM label into sim button (left half)
        const char* sim_label = "sim";
        auto sim_text = render_text_to_rgba(sim_label, 0.8f, {220,220,235,255});
        if (!sim_text.pixels.empty()) {
            int txs2 = sim_x + (sim_btn_w - sim_text.width) / 2;
            int tys2 = kpn_by + (bh - sim_text.height) / 2;
            for (int yy = 0; yy < sim_text.height; ++yy) {
                int dst_y = tys2 + yy; if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < sim_text.width; ++xx) {
                    int dst_x = txs2 + xx; if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &sim_text.pixels[(yy * sim_text.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
    }

    // Prepare storage for per-module table hitboxes discovered during table rendering.
    std::vector<std::vector<GP_TableHitBox>> module_hitboxes(ctx->modules.size());
    std::vector<std::unordered_map<int, ContactLight>> module_contact_lights(ctx->modules.size());
    std::vector<std::unordered_map<int, bool>> module_frame_roles(ctx->modules.size());
    std::vector<std::vector<uint8_t>> module_tables(ctx->modules.size());
    std::vector<std::unordered_map<int, uint32_t>> module_contact_subgroups(ctx->modules.size());

    for (const auto &edge : ctx->edges) {
        if (edge.subgroup_flags == 0u) continue;
        if (edge.desc.a_module >= 0 && edge.desc.a_module < static_cast<int>(module_contact_subgroups.size())) {
            module_contact_subgroups[edge.desc.a_module][edge.desc.a_contact_idx] |= edge.subgroup_flags;
        }
        if (edge.desc.b_module >= 0 && edge.desc.b_module < static_cast<int>(module_contact_subgroups.size())) {
            module_contact_subgroups[edge.desc.b_module][edge.desc.b_contact_idx] |= edge.subgroup_flags;
        }
    }

    // First pass: render tables and gather hitboxes/light info (no drawing yet).
    for (int mi = 0; mi < static_cast<int>(ctx->modules.size()); ++mi) {
        const auto &m = ctx->modules[mi];
        module_hitboxes[mi].clear();
        module_contact_lights[mi].clear();
        module_frame_roles[mi].clear();
        module_tables[mi].clear();
        GP_TableContext* t = (mi >= 0 && mi < static_cast<int>(ctx->module_tables.size())) ? ctx->module_tables[mi] : nullptr;
        if (t) {
            // Ensure per-module table layout is up-to-date before rendering it.
            sync_module_table_io_layout(ctx, mi);
            bool is_stage = (mi >= 0 && mi < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[mi]);
            int tw = std::max(1, m.w);
            int th = std::max(1, m.h);
            ModuleLayout layout = module_layout_for(ctx, mi, m);
            int table_clip_h = std::max(1, layout.table_clip_h);
            int table_offset_y = layout.table_y;
            GP_TableStyle st{};
            gp_table_get_style(t, &st);
            Color table_bg{st.bg_rgba[0], st.bg_rgba[1], st.bg_rgba[2], st.bg_rgba[3]};
            GP_TableGeom geom{};
            gp_table_get_geom(t, &geom);
            geom.width_px = tw;
            int table_render_h = std::max(1, layout.table_clip_h);
            module_tables[mi].assign(static_cast<size_t>(tw) * static_cast<size_t>(th) * 4u, 0);
            if (!is_stage) {
                memset_rect(module_tables[mi].data(), tw, th, tw * 4, 0, table_offset_y, tw, table_clip_h, table_bg);
            }
            const int hitcap = 4096;
            std::vector<GP_TableHitBox> hits(hitcap);
            int hits_written = 0;
            if (mi < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[mi]) {
                // For stage modules we want the stage image to act as the full module background.
                geom.height_px = th;
                if (!ctx->module_stage_images[mi].rgba) {
                    gp_table_render_rgba_with_state(t, nullptr, module_tables[mi].data(), static_cast<int32_t>(module_tables[mi].size()), &geom, hits.data(), hitcap, &hits_written);
                } else {
                    const uint8_t* src_stage = ctx->module_stage_images[mi].rgba;
                    size_t bytes = static_cast<size_t>(tw) * static_cast<size_t>(th) * 4u;
                    std::memcpy(module_tables[mi].data(), src_stage, bytes);
                    std::vector<uint8_t> tmp_buf(module_tables[mi].size());
                    int ok = gp_table_render_rgba_with_state(t, nullptr, tmp_buf.data(), static_cast<int32_t>(tmp_buf.size()), &geom, hits.data(), hitcap, &hits_written);
                    if (ok) {
                        size_t pixels = static_cast<size_t>(tw) * static_cast<size_t>(th);
                        uint8_t* dst = module_tables[mi].data();
                        const uint8_t* src = tmp_buf.data();
                        for (size_t pi = 0; pi < pixels; ++pi) {
                            const uint8_t sa = src[pi * 4 + 3];
                            if (sa == 0) continue;
                            if (sa >= 255) {
                                dst[pi * 4 + 0] = src[pi * 4 + 0];
                                dst[pi * 4 + 1] = src[pi * 4 + 1];
                                dst[pi * 4 + 2] = src[pi * 4 + 2];
                                dst[pi * 4 + 3] = 255;
                            } else {
                                float a = sa / 255.0f;
                                for (int cc = 0; cc < 3; ++cc) {
                                    dst[pi * 4 + cc] = static_cast<uint8_t>(std::lround(src[pi * 4 + cc] * a + dst[pi * 4 + cc] * (1.0f - a)));
                                }
                                dst[pi * 4 + 3] = 255;
                            }
                        }
                    }
                }
            } else {
                geom.height_px = table_render_h;
                std::vector<uint8_t> tmp_buf(static_cast<size_t>(tw) * static_cast<size_t>(table_render_h) * 4u);
                int ok = gp_table_render_rgba_with_state(t, nullptr, tmp_buf.data(), static_cast<int32_t>(tmp_buf.size()), &geom, hits.data(), hitcap, &hits_written);
                if (ok) {
                    int copy_h = std::min(table_clip_h, table_render_h);
                    for (int yy = 0; yy < copy_h; ++yy) {
                        const uint8_t* src = tmp_buf.data() + static_cast<size_t>(yy) * static_cast<size_t>(tw) * 4u;
                        uint8_t* dst = module_tables[mi].data() + static_cast<size_t>(yy + table_offset_y) * static_cast<size_t>(tw) * 4u;
                        std::memcpy(dst, src, static_cast<size_t>(tw) * 4u);
                    }
                }
            }
            if (hits_written > 0) {
                for (int hi = 0; hi < hits_written; ++hi) {
                    const auto &hb = hits[hi];
                    if (!is_stage && hb.y0 >= table_clip_h) continue;
                    GP_TableHitBox adjusted = hb;
                    adjusted.y0 += table_offset_y;
                    adjusted.y1 += table_offset_y;
                    module_hitboxes[mi].push_back(adjusted);
                }
            }
            auto &light_map = module_contact_lights[mi];
            light_map.clear();
            Color led_on{st.led_on_rgba[0], st.led_on_rgba[1], st.led_on_rgba[2], st.led_on_rgba[3]};
            int in_count = (mi >= 0 && mi < static_cast<int>(ctx->module_io_in_count.size())) ? ctx->module_io_in_count[mi] : 0;
            int out_count = (mi >= 0 && mi < static_cast<int>(ctx->module_io_out_count.size())) ? ctx->module_io_out_count[mi] : 0;
            const auto &hits_filtered = module_hitboxes[mi];
            for (size_t hi = 0; hi < hits_filtered.size(); ++hi) {
                const auto &hb = hits_filtered[hi];
                if (hb.part != GP_TABLE_HIT_LED && hb.part != GP_TABLE_HIT_LED_ARG && hb.part != GP_TABLE_HIT_LED_TABLE) continue;
                if (hb.row_idx < 0) continue;
                int contact_idx = resolve_contact_index(ctx, mi, hb);
                if (contact_idx < 0) continue;
                int on = 0, active = 0;
                if (!gp_table_get_led_info(t, hb.row_idx, hb.col_idx, hb.aux0, &on, &active, nullptr, nullptr)) continue;
                float glow = 0.0f;
                gp_table_get_led_glow(t, hb.row_idx, hb.col_idx, hb.aux0, &glow);
                bool lit = (on != 0) || (active != 0);
                bool is_input = (contact_idx >= 0 && contact_idx < in_count);
                bool is_output = (contact_idx >= in_count && contact_idx >= 0 && contact_idx < (in_count + out_count));
                if (glow <= 0.0f && is_output && lit) glow = 0.35f;
                if (!lit) glow *= 0.3f;
                if (is_input) glow *= 0.35f;
                if (is_output && lit) glow = std::min(1.0f, glow * 1.25f + 0.15f);
                if (glow > 0.0f) {
                    ContactLight cl;
                    uint32_t subgroup_flags = 0u;
                    if (mi < static_cast<int>(module_contact_subgroups.size())) {
                        auto it = module_contact_subgroups[mi].find(contact_idx);
                        if (it != module_contact_subgroups[mi].end()) subgroup_flags = it->second;
                    }
                    cl.col = (subgroup_flags != 0u) ? subgroup_flags_to_color(ctx, subgroup_flags, led_on.a) : led_on;
                    cl.intensity = glow;
                    cl.valid = true;
                    light_map[contact_idx] = cl;
                }
            }
        }
            for_each_module_frame_led(ctx, mi, m, [&](int row, int idx, int x0, int y0, int x1, int y1) {
                GP_TableHitBox hb = make_module_frame_led_hitbox(row, idx, x0, y0, x1, y1);
                module_hitboxes[mi].push_back(hb);
                const int full_cols = 2;
                int grid_row = row / full_cols;
                module_frame_roles[mi][hb.aux0] = (grid_row != 0);
            });
            
        // If this is the synthetic root module, also expose toolbar LED
        // hitboxes as module-local frame LEDs using the distinct toolbar
        // contact base so ropes can resolve to the toolbar positions.
        if (mi == ctx->root_module_idx && !ctx->toolbar_leds.empty()) {
            int toolbar_base = kModuleFrameContactBase + kModuleExtraLedCount * kModuleExtraLedRows;
            for (const auto &tb : ctx->toolbar_leds) {
                GP_TableHitBox thb{};
                thb.x0 = tb.wx0 - ctx->modules[mi].x;
                thb.x1 = tb.wx1 - ctx->modules[mi].x;
                thb.y0 = tb.wy0 - ctx->modules[mi].y;
                thb.y1 = tb.wy1 - ctx->modules[mi].y;
                thb.cell_kind = GP_TABLE_CELL_LEDS;
                thb.part = GP_TABLE_HIT_LED;
                thb.row_idx = kModuleFrameRowReceive;
                thb.col_idx = 0;
                thb.aux0 = toolbar_base + tb.subgroup_idx;
                thb.aux1 = 0;
                module_hitboxes[mi].push_back(thb);
                module_frame_roles[mi][thb.aux0] = true; // receive role
            }
        }
        Color frame_led_on{255,210,90,255};
        if (t) {
            GP_TableStyle st{};
            gp_table_get_style(t, &st);
            frame_led_on = Color{st.led_on_rgba[0], st.led_on_rgba[1], st.led_on_rgba[2], st.led_on_rgba[3]};
        }
        auto &frame_light_map = module_contact_lights[mi];
        const int full_cols = 2;
        int pair_count = (kModuleExtraLedRows + full_cols - 1) / full_cols;
        int total_rows = pair_count * full_cols;
        for (int row = 0; row < total_rows; ++row) {
            int base = kModuleFrameContactBase + row * kModuleExtraLedCount;
            for (int idx = 0; idx < kModuleExtraLedCount; ++idx) {
                const GP_TableCell* cell = module_frame_led_cell(ctx, mi, row, idx);
                if (!cell) continue;
                bool on = (cell->flags & 0x1u) != 0u;
                bool active = (static_cast<uint32_t>(cell->reserved0) & 0x1u) != 0u;
                if (!on && !active) continue;
                ContactLight cl;
                cl.col = frame_led_on;
                cl.intensity = on ? 0.6f : 0.35f;
                cl.valid = true;
                frame_light_map[base + idx] = cl;
            }
        }
    }

    // ensure root rope sim exists
    RopeSim* sim = canvas_require_root_sim(ctx);

    std::vector<ContactLight> edge_light_a(ctx->edges.size());
    std::vector<ContactLight> edge_light_b(ctx->edges.size());
    const float rope_decay = 0.35f;
    const float rope_end_gain = std::exp(-rope_decay);
    auto resolve_edge_light_for_rope = [&](int rope_idx, ContactLight* out_a, ContactLight* out_b) -> bool {
        if (rope_idx < 0) return false;
        for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
            if (ctx->edges[ei].rope_idx != rope_idx) continue;
            if (out_a) *out_a = edge_light_a[ei];
            if (out_b) *out_b = edge_light_b[ei];
            return true;
        }
        return false;
    };
    auto is_input_contact = [&](int module_idx, int contact_idx) -> bool {
        if (module_idx >= 0 && module_idx < static_cast<int>(module_frame_roles.size())) {
            const auto &roles = module_frame_roles[module_idx];
            auto it = roles.find(contact_idx);
            if (it != roles.end()) return it->second;
        }
        if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_io_in_count.size())) {
            int in_count = ctx->module_io_in_count[module_idx];
            return (contact_idx >= 0 && contact_idx < in_count);
        }
        return false;
    };

    // update/create ropes for edges
    for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
        const auto &edge = ctx->edges[ei].desc;
        if (edge.a_module < 0 || edge.a_module >= static_cast<int>(ctx->modules.size())) continue;
        if (edge.b_module < 0 || edge.b_module >= static_cast<int>(ctx->modules.size())) continue;
        int ax, ay, bx, by;
        bool resolvedA = false, resolvedB = false;
        // If this edge is an overlay (custom overlay keys set), prefer
        // resolving the overlay LED positions to determine rope endpoints
        // so overlay-attached ropes remain stable and don't get cleared.
        auto &edgeinfo = ctx->edges[ei];
        if (edgeinfo.overlay_key_a || edgeinfo.overlay_key_b) {
            int ox = 0, oy = 0, ox2 = 0, oy2 = 0;
            if (gp_canvas_resolve_overlay_key(reinterpret_cast<GP_CanvasContext*>(ctx), edgeinfo.overlay_key_a, &ox, &oy) &&
                gp_canvas_resolve_overlay_key(reinterpret_cast<GP_CanvasContext*>(ctx), edgeinfo.overlay_key_b, &ox2, &oy2)) {
                ax = ox; ay = oy; bx = ox2; by = oy2;
                resolvedA = true; resolvedB = true;
            }
        }
        // compute contact positions. If the endpoint module has an attached table
        // and we captured hitboxes during rendering, prefer the table-provided
        // hitbox center for exact LED anchor coordinates. Fall back to legacy
        // computed positions otherwise.
        const GP_CanvasModuleDesc &ma = ctx->modules[edge.a_module];
        const GP_CanvasModuleDesc &mb = ctx->modules[edge.b_module];
        if (edge.a_module < static_cast<int>(module_hitboxes.size()) && !module_hitboxes[edge.a_module].empty()) {
            for (const auto &hb : module_hitboxes[edge.a_module]) {
                if ((hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG || hb.part == GP_TABLE_HIT_LED_TABLE) && resolve_contact_index(ctx, edge.a_module, hb) == edge.a_contact_idx) {
                    int local_x = (hb.x0 + hb.x1) / 2;
                    int local_y = (hb.y0 + hb.y1) / 2;
                    ax = ctx->modules[edge.a_module].x + local_x;
                    ay = ctx->modules[edge.a_module].y + local_y;
                    if (edge.a_module < static_cast<int>(module_contact_lights.size())) {
                        auto &map = module_contact_lights[edge.a_module];
                        auto it = map.find(edge.a_contact_idx);
                        if (it != map.end()) edge_light_a[ei] = it->second;
                    }
                    resolvedA = true; break;
                }
            }
        }
        if (edge.b_module < static_cast<int>(module_hitboxes.size()) && !module_hitboxes[edge.b_module].empty()) {
            for (const auto &hb : module_hitboxes[edge.b_module]) {
                if ((hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG || hb.part == GP_TABLE_HIT_LED_TABLE) && resolve_contact_index(ctx, edge.b_module, hb) == edge.b_contact_idx) {
                    int local_x = (hb.x0 + hb.x1) / 2;
                    int local_y = (hb.y0 + hb.y1) / 2;
                    bx = ctx->modules[edge.b_module].x + local_x;
                    by = ctx->modules[edge.b_module].y + local_y;
                    if (edge.b_module < static_cast<int>(module_contact_lights.size())) {
                        auto &map = module_contact_lights[edge.b_module];
                        auto it = map.find(edge.b_contact_idx);
                        if (it != map.end()) edge_light_b[ei] = it->second;
                    }
                    resolvedB = true; break;
                }
            }
        }
        if (resolvedA && resolvedB) {
            bool a_input = is_input_contact(edge.a_module, edge.a_contact_idx);
            bool b_input = is_input_contact(edge.b_module, edge.b_contact_idx);
            if (!a_input && b_input) {
                float src = edge_light_a[ei].valid ? edge_light_a[ei].intensity : 0.0f;
                float delivered = src * rope_end_gain;
                if (delivered > 0.0f) {
                    if (!edge_light_b[ei].valid) {
                        edge_light_b[ei].col = edge_light_a[ei].col;
                        edge_light_b[ei].valid = true;
                        edge_light_b[ei].intensity = delivered;
                    } else {
                        edge_light_b[ei].intensity = std::max(edge_light_b[ei].intensity, delivered);
                    }
                }
            }
        }
        // If either endpoint couldn't be resolved from table hitboxes, skip
        // rope creation for this edge - legacy side-based geometry removed.
        if (!resolvedA || !resolvedB) {
            ctx->edges[ei].rope_idx = -1;
            continue;
        }
        int ridx = ctx->edges[ei].rope_idx;
            if (ridx < 0) {
            int segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : ctx->sim_segs;
            float slack = ctx->sim_slack;
            int newr = sim ? rope_sim_add_rope(sim, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by), segs, slack) : -1;
            ctx->edges[ei].rope_idx = newr;
            if (ei < 6) {
#if defined(GP_CANVAS_DEBUG_PRINTF)
                printf("canvas_raster: edge=%zu created rope sim=%p ridx=%d ax=%.2f,%.2f bx=%.2f,%.2f segs=%d slack=%.3f\n", ei, (void*)sim, newr, (float)ax, (float)ay, (float)bx, (float)by, segs, slack);
                if (sim && newr >= 0) {
                    int vc = rope_sim_get_vertex_count(sim, newr);
                    std::vector<float> verts(static_cast<size_t>(std::max(0, vc) * 2));
                    int got = rope_sim_get_vertices(sim, newr, verts.data(), static_cast<int>(verts.size()));
                    if (got > 0) printf("canvas_raster: edge=%zu rope=%d verts=%d first=(%.2f,%.2f)\n", ei, newr, got, verts[0], verts[1]);
                }
#endif
            }
        } else {
            if (sim) {
                rope_sim_move_endpoints(sim, ridx, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by));
                if (ei < 6) {
#if defined(GP_CANVAS_DEBUG_PRINTF)
                    int vc = rope_sim_get_vertex_count(sim, ridx);
                    std::vector<float> verts(static_cast<size_t>(std::max(0, vc) * 2));
                    int got = rope_sim_get_vertices(sim, ridx, verts.data(), static_cast<int>(verts.size()));
                    printf("canvas_raster: edge=%zu moved rope sim=%p ridx=%d verts=%d\n", ei, (void*)sim, ridx, got);
                    if (got > 0) printf("canvas_raster: edge=%zu rope=%d first=(%.2f,%.2f)\n", ei, ridx, verts[0], verts[1]);
#endif
                }
            }
        }
    }

    // step sim if allowed by global/table sim cadence
    if (sim) {
        GP_TableContext* root_tbl = canvas_ensure_root_table(ctx);
        if (root_tbl && gp_table_should_step_sim(root_tbl)) {
            uint32_t dbg = ctx->debug_flags;
            bool disable_sim = (dbg & GP_CANVAS_DEBUG_NO_SPRINGS) != 0u || (dbg & GP_CANVAS_DEBUG_RING_STATIC) != 0u;
            float gravity = (dbg & GP_CANVAS_DEBUG_NO_GRAVITY) ? 0.0f : ctx->sim_maxforce;
            if (!disable_sim) {
                rope_sim_step(sim, 1.0f/60.0f, gravity, ctx->sim_iters, ctx->sim_damping);
            }
        }
    }

    // Second pass: draw module backgrounds (raytrace/callback) and blit tables.
    for (int mi = 0; mi < static_cast<int>(ctx->modules.size()); ++mi) {
        const auto &m = ctx->modules[mi];
        int sx = m.x - ctx->offset_x;
        int sy = m.y - ctx->offset_y;
        std::vector<InputRayLight> inputs;
        bool is_stage = (mi >= 0 && mi < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[mi]);
        Color bgc{40,40,50,255};
        if (!is_stage) {
            memset_rect(out_rgba, w, h, pitch, sx, sy, m.w, m.h, bgc);
        }
        ModuleLayout layout = module_layout_for(ctx, mi, m);
        if (mi >= 0 && mi < static_cast<int>(ctx->module_bg.size())) {
            auto &bg = ctx->module_bg[mi];
            if (bg.cb) {
                if (is_stage) {
                    ensure_module_bg_storage(bg, m.w, m.h, /*oversample=*/1);
                    bg.cb(bg.user, mi, m.w, m.h, bg.scratch.data(), m.w * 4);
                } else {
                    int preview_h = layout.preview_h;
                    if (preview_h > 0) {
                        ensure_module_bg_storage(bg, m.w, preview_h, /*oversample=*/1);
                        bg.cb(bg.user, mi, m.w, preview_h, bg.scratch.data(), m.w * 4);
                        blit_module_buffer(out_rgba, w, h, pitch, sx, sy + layout.preview_y, m.w, preview_h, bg.scratch);
                    }
                }
                if (is_stage) {
                    // Stage modules: embed the stage scratch buffer inside the table's content
                    // image cell and re-render the table now that the stage buffer is current.
                    if (mi >= static_cast<int>(ctx->module_stage_images.size())) ctx->module_stage_images.resize(mi + 1);
                    GP_TableImage img{};
                    if (!bg.scratch.empty()) {
                        img.rgba = bg.scratch.data();
                        img.width_px = m.w;
                        img.height_px = m.h;
                        img.pitch_bytes = m.w * 4;
                    }
                    ctx->module_stage_images[mi] = img;
                    if (mi >= 0 && mi < static_cast<int>(ctx->module_tables.size()) && ctx->module_tables[mi]) {
                        GP_TableContext* t = ctx->module_tables[mi];
                        int rc = gp_table_get_row_count(t);
                        if (rc > 0) {
                            std::vector<GP_TableRow> rows(static_cast<size_t>(rc));
                            for (int ri = 0; ri < rc; ++ri) gp_table_get_row(t, ri, &rows[static_cast<size_t>(ri)]);
                            bool changed = false;
                            for (auto &r : rows) {
                                for (int ci = 0; ci < r.cell_count && ci < 8; ++ci) {
                                    if (r.cells[ci].kind == GP_TABLE_CELL_IMAGE) {
                                        r.cells[ci].image = img.rgba ? &ctx->module_stage_images[mi] : nullptr;
                                        // Allow the table to allocate the full module height for
                                        // the image content by setting the cell's reserved0
                                        // to the module height. Also set the row's
                                        // reserved0 to the module height so the row
                                        // requests the full module area (header + image)
                                        // instead of collapsing to the header band.
                                        if (img.rgba) {
                                            r.cells[ci].reserved0 = m.h;
                                            r.reserved0 = m.h;
                                        }
                                        changed = true;
                                    }
                                }
                            }
                            if (changed) gp_table_set_rows(t, rows.data(), rc);
                        }
                        if (mi >= 0 && mi < static_cast<int>(module_tables.size()) && !module_tables[mi].empty()) {
                            if (img.rgba) {
                                size_t bytes = static_cast<size_t>(m.w) * static_cast<size_t>(m.h) * 4u;
                                if (module_tables[mi].size() >= bytes) {
                                    std::memcpy(module_tables[mi].data(), img.rgba, bytes);
                                }
                            }
                            GP_TableGeom geom{};
                            geom.width_px = std::max(1, m.w);
                            geom.height_px = std::max(1, m.h);
                            // Debug: report computed table geom and row/cell reserved values
                            GP_TableGeom gtmp{};
                            if (gp_table_get_geom(t, &gtmp)) {
                                printf("DEBUG: before render: table rows=%d ctx_geom=%dx%d target_geom=%dx%d\n", rc, gtmp.width_px, gtmp.height_px, geom.width_px, geom.height_px);
                            } else {
                                printf("DEBUG: before render: table rows=%d target_geom=%dx%d (no ctx geom)\n", rc, geom.width_px, geom.height_px);
                            }
                            for (int ri = 0; ri < rc; ++ri) {
                                GP_TableRow r{};
                                gp_table_get_row(t, ri, &r);
                                printf("DEBUG: row %d kind=%d expanded=%d reserved0=%d cell_count=%d\n", ri, r.kind, r.expanded, r.reserved0, r.cell_count);
                                for (int ci = 0; ci < r.cell_count && ci < 8; ++ci) {
                                    const GP_TableCell &c = r.cells[ci];
                                    if (c.kind == GP_TABLE_CELL_IMAGE) printf("DEBUG:  cell %d is IMAGE reserved0=%d image=%p\n", ci, c.reserved0, (void*)c.image);
                                }
                            }

                            // Render table into a temp buffer and composite over the
                            // existing module buffer so the stage background (which
                            // lives in module_tables[mi]) remains visible where the
                            // table is transparent.
                            std::vector<uint8_t> tmp_buf(module_tables[mi].size());
                            int ok = gp_table_render_rgba_with_state(t, nullptr, tmp_buf.data(), static_cast<int32_t>(tmp_buf.size()), &geom, nullptr, 0, nullptr);
                            if (ok) {
                                int tw_local = std::max(1, m.w);
                                int th_local = std::max(1, m.h);
                                size_t count_nonzero = 0;
                                size_t pixels = static_cast<size_t>(tw_local) * static_cast<size_t>(th_local);
                                uint8_t* dst = module_tables[mi].data();
                                const uint8_t* src = tmp_buf.data();
                                for (size_t pi = 0; pi < pixels; ++pi) {
                                    const uint8_t sa = src[pi * 4 + 3];
                                    if (sa == 0) {
                                        // nothing drawn here; keep existing base
                                        if (dst[pi * 4 + 3] != 0) ++count_nonzero;
                                        continue;
                                    }
                                    if (sa >= 255) {
                                        dst[pi * 4 + 0] = src[pi * 4 + 0];
                                        dst[pi * 4 + 1] = src[pi * 4 + 1];
                                        dst[pi * 4 + 2] = src[pi * 4 + 2];
                                        dst[pi * 4 + 3] = 255;
                                        ++count_nonzero;
                                    } else {
                                        float a = sa / 255.0f;
                                        for (int cc = 0; cc < 3; ++cc) {
                                            dst[pi * 4 + cc] = static_cast<uint8_t>(std::lround(src[pi * 4 + cc] * a + dst[pi * 4 + cc] * (1.0f - a)));
                                        }
                                        dst[pi * 4 + 3] = 255;
                                        ++count_nonzero;
                                    }
                                }
                                printf("DEBUG: after render: nonzero_alpha_pixels=%zu of %zu\n", count_nonzero, pixels);
                            }
                        }
                    }
                }
            } else if (bg.mode == 1 && !is_stage) {
                int preview_h = layout.preview_h;
                if (preview_h > 0) {
                    render_module_raytrace_bg(bg, m.w, preview_h, inputs);
                    blit_module_buffer(out_rgba, w, h, pitch, sx, sy + layout.preview_y, m.w, preview_h, bg.scratch);
                }
            } else if (bg.mode == 1) {
                render_module_raytrace_bg(bg, m.w, m.h, inputs);
                blit_module_buffer(out_rgba, w, h, pitch, sx, sy, m.w, m.h, bg.scratch);
            }
        }
        
        if (mi >= 0 && mi < static_cast<int>(module_tables.size()) && !module_tables[mi].empty()) {
            uint8_t table_alpha = 255;
            if (mi >= 0 && mi < static_cast<int>(ctx->module_bg.size())) {
                const auto &bg = ctx->module_bg[mi];
                table_alpha = bg.table_alpha;
                if (bg.mode == 1) {
                    table_alpha = std::min(table_alpha, bg.table_alpha_ray);
                }
            }
            if (is_stage) {
                // Stage imagery is embedded into the table buffer; blend using the table's per-pixel alpha.
                blit_module_buffer_srcalpha(out_rgba, w, h, pitch, sx, sy, m.w, m.h, module_tables[mi]);
            } else {
                blit_module_buffer_alpha(out_rgba, w, h, pitch, sx, sy, m.w, m.h, module_tables[mi], table_alpha);
            }
        }
        draw_module_top_ui(ctx, mi, m, out_rgba, w, h, pitch);
        // Add hitbox for the timing toggle button so clicks get dispatched.
        // Recompute the control geometry used by draw_module_top_ui to derive the timing button rect.
        {
            int sx = m.x - ctx->offset_x;
            int sy = m.y - ctx->offset_y;
            int cursor_y = sy + kModuleTopPadding;
            cursor_y += kModuleTitleRowH;
            cursor_y += kModuleThumbRowH + kModuleTopGap;
            int control_y = cursor_y;
            int control_h = std::max(1, kModuleControlRowH - 2);
            int gap = 6;
            int nbw = control_h;
            int btn_y = control_y + 1;
            int right_x = sx + m.w - kModuleTopPadding;
            int menu_w = std::max(30, control_h);
            int lib_w = menu_w;
            int pause_w = std::max(42, control_h * 2);
            int menu_x = right_x - menu_w;
            int lib_x = menu_x - gap - lib_w;
            int pause_x = lib_x - gap - pause_w;
            int module_btn_count = 5;
            int module_btn_w = nbw;
            int module_gap = 6;
            int btns_total_w = module_btn_count * (module_btn_w + module_gap) - module_gap;
            int btns_right = pause_x - module_gap;
            int btns_left = btns_right - btns_total_w;
            int timing_btn_w = nbw;
            int timing_x = btns_left - module_gap - timing_btn_w;
            GP_TableHitBox thb{};
            thb.x0 = timing_x - sx;
            thb.x1 = thb.x0 + timing_btn_w;
            thb.y0 = btn_y - sy;
            thb.y1 = thb.y0 + control_h;
            thb.cell_kind = GP_TABLE_CELL_TEXT;
            thb.part = GP_TABLE_HIT_CELL;
            thb.row_idx = kModuleFrameRowSend;
            thb.col_idx = kModuleColText;
            thb.aux0 = 0;
            module_hitboxes[mi].push_back(thb);
        }
        if (mi == ctx->focused_module) {
            Color fb{60,120,220,255};
            int t = 2;
            for (int dy = 1; dy <= t; ++dy) {
                int ytop = sy - dy;
                int ybot = sy + m.h - 1 + dy;
                if (ytop >= 0 && ytop < h) memset_rect(out_rgba, w, h, pitch, std::max(0, sx - dy), ytop, std::min(w, m.w + 2*dy), 1, fb);
                if (ybot >= 0 && ybot < h) memset_rect(out_rgba, w, h, pitch, std::max(0, sx - dy), ybot, std::min(w, m.w + 2*dy), 1, fb);
            }
            for (int dx = 1; dx <= t; ++dx) {
                int lx = sx - dx;
                int rx = sx + m.w - 1 + dx;
                if (lx >= 0 && lx < w) memset_rect(out_rgba, w, h, pitch, lx, std::max(0, sy - t), 1, std::min(h, m.h + 2*t), fb);
                if (rx >= 0 && rx < w) memset_rect(out_rgba, w, h, pitch, rx, std::max(0, sy - t), 1, std::min(h, m.h + 2*t), fb);
            }
        }

        // Draw module preview on top of everything in the module frame area
        if (!is_stage && layout.preview_h > 0 && mi >= 0 && mi < static_cast<int>(ctx->module_preview_buffers.size())) {
            ModulePreviewBuffer &preview = ctx->module_preview_buffers[mi];
            int preview_w = std::max(1, m.w);
            int preview_h = layout.preview_h;
            int preview_pitch = preview_w * 4;
            if (preview.width_px != preview_w || preview.height_px != preview_h || preview.pitch_bytes != preview_pitch) {
                preview.width_px = preview_w;
                preview.height_px = preview_h;
                preview.pitch_bytes = preview_pitch;
                preview.rgba.assign(static_cast<size_t>(preview_w) * static_cast<size_t>(preview_h) * 4u, 0);
            }
            if (preview.hitboxes.size() < 1024) preview.hitboxes.resize(1024);
            GP_ModulePreviewInput input{};
            build_module_preview_input(ctx, mi, input);
            GP_ModulePreviewOutput output{};
            output.rgba = preview.rgba.data();
            output.width_px = preview.width_px;
            output.height_px = preview.height_px;
            output.pitch_bytes = preview.pitch_bytes;
            output.hitboxes = preview.hitboxes.data();
            output.hitbox_capacity = static_cast<int32_t>(preview.hitboxes.size());
            output.hitbox_count = 0;
            gp_module_preview_build(&input, &output);
            if (!preview.rgba.empty()) {
                blit_module_buffer_srcalpha(out_rgba, w, h, pitch, sx, sy + layout.preview_y, preview_w, preview_h, preview.rgba);
            }
        }
    }

    // Render custom overlays (behind ropes): simple translucent rectangles
    // defined in world coords and transformed into canvas pixels.
    if (ctx->overlays.size() > 0) {
        for (auto &kv : ctx->overlays) {
            auto &ov = kv.second;
            // clear any previously cached control rect; renderer will set when drawing
            ov.ctrl_x = -1; ov.ctrl_y = -1; ov.ctrl_w = 0; ov.ctrl_h = 0;
            int rx0 = static_cast<int>(std::floor(std::min(ov.x1, ov.x2) - static_cast<float>(ctx->offset_x)));
            int ry0 = static_cast<int>(std::floor(std::min(ov.y1, ov.y2) - static_cast<float>(ctx->offset_y)));
            int rx1 = static_cast<int>(std::ceil(std::max(ov.x1, ov.x2) - static_cast<float>(ctx->offset_x)));
            int ry1 = static_cast<int>(std::ceil(std::max(ov.y1, ov.y2) - static_cast<float>(ctx->offset_y)));
            int rw = rx1 - rx0; int rh = ry1 - ry0;
            if (rw > 0 && rh > 0) {
                Color bg{24, 28, 32, 160};
                memset_rect(out_rgba, w, h, pitch, rx0, ry0, rw, rh, bg);
                // draw anchor LEDs at overlay endpoints
                int led_ax = static_cast<int>(std::lround(ov.x1 - static_cast<float>(ctx->offset_x)));
                int led_ay = static_cast<int>(std::lround(ov.y1 - static_cast<float>(ctx->offset_y)));
                int led_bx = static_cast<int>(std::lround(ov.x2 - static_cast<float>(ctx->offset_x)));
                int led_by = static_cast<int>(std::lround(ov.y2 - static_cast<float>(ctx->offset_y)));
                draw_circle(out_rgba, w, h, pitch, led_ax, led_ay, 6, Color{220,220,200,220});
                draw_circle(out_rgba, w, h, pitch, led_bx, led_by, 6, Color{220,220,200,220});
                draw_circle(out_rgba, w, h, pitch, led_ax, led_ay, 3, Color{40,40,48,220});
                draw_circle(out_rgba, w, h, pitch, led_bx, led_by, 3, Color{40,40,48,220});
                // (removed decorative LED placeholders) rely on overlay endpoints drawn above
                // draw small mode-toggle button in top-right of overlay (uses table API)
                int btn_w = 20; int btn_h = 18; int margin = 6;
                // compute button position in world coords (ov.* are world coords)
                float btn_world_x = std::max(ov.x1, ov.x2) - static_cast<float>(margin) - static_cast<float>(btn_w);
                float btn_world_y = std::min(ov.y1, ov.y2) + static_cast<float>(margin);
                int bx = static_cast<int>(std::lround(btn_world_x - static_cast<float>(ctx->offset_x)));
                int by = static_cast<int>(std::lround(btn_world_y - static_cast<float>(ctx->offset_y)));
                // determine associated rope index (if any) by scanning edges
                int overlay_rope_idx = -1;
                for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
                    const auto &e = ctx->edges[ei];
                    if (e.overlay_key_a == ov.key_a || e.overlay_key_b == ov.key_b) { overlay_rope_idx = e.rope_idx; break; }
                }
                // precompute a default control rect in screen coords so input can
                // hit-test it even if we don't successfully look up the widget
                // position in the table. Always cache the default rect based on
                // the overlay button position so hit-testing is consistent.
                {
                    int ctrl_w_def = 88; int ctrl_h_def = 18;
                    int ctrl_x_def = bx + (btn_w/2) - (ctrl_w_def/2);
                    int ctrl_y_def = by + btn_h + 6;
                    ov.ctrl_x = ctrl_x_def; ov.ctrl_y = ctrl_y_def; ov.ctrl_w = ctrl_w_def; ov.ctrl_h = ctrl_h_def;
                }
                // If this overlay is attached to a dangling-widget rope for a meta-group,
                // draw the numeric control at the widget position so it appears where
                // the existing binding/mode button is located.
                if (overlay_rope_idx >= 0) {
                    int display_ch = 0;
                    float wpos[3] = {0.0f,0.0f,0.0f};
                    bool drew = false;
                    auto try_draw_for_table = [&](GP_TableContext* t)->bool {
                        if (!t) return false;
                        int mgcount = gp_table_get_meta_group_count(t);
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            if (!mg) continue;
                            int dr = -1, dv = -1;
                            gp_table_meta_get_dangling_rope_info(t, mg, &dr, &dv);
                            int ch_tmp = 0; gp_table_meta_get_channel_group(t, mg, &ch_tmp);
                            (void)ch_tmp;
                            if (dr != overlay_rope_idx) continue;
                            // found a meta-group associated with this overlay rope
                            gp_table_meta_get_channel_group(t, mg, &display_ch);
                            int wwid = -1;
                            gp_table_meta_get_dangling_widget_id(t, mg, &wwid);
                            if (wwid >= 0 && gp_table_get_widget_position(t, wwid, wpos)) {
                                int wcx = static_cast<int>(std::lround(wpos[0] - static_cast<float>(ctx->offset_x)));
                                int wcy = static_cast<int>(std::lround(wpos[1] - static_cast<float>(ctx->offset_y)));
                                int ctrl_w = 88; int ctrl_h = 18;
                                // place numeric control centered below the mode/button
                                int ctrl_x = bx + (btn_w/2) - (ctrl_w/2);
                                int ctrl_y = by + btn_h + 6;
                                // cache control rect in overlay (screen/view coords)
                                ov.ctrl_x = ctrl_x; ov.ctrl_y = ctrl_y; ov.ctrl_w = ctrl_w; ov.ctrl_h = ctrl_h;
                                (void)wwid; (void)wcx; (void)wcy;
                                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, ctrl_h, Color{40,40,50,220});
                                // outline (1px) in bright green so numeric control is obvious
                                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, 1, Color{80,200,120,220});
                                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y + ctrl_h - 1, ctrl_w, 1, Color{80,200,120,220});
                                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, 1, ctrl_h, Color{80,200,120,220});
                                memset_rect(out_rgba, w, h, pitch, ctrl_x + ctrl_w - 1, ctrl_y, 1, ctrl_h, Color{80,200,120,220});
                                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y + ctrl_h - 1, ctrl_w, 1, Color{16,16,20,255});
                                int btn_w = ctrl_h; int gap = 6;
                                int bx_minus = ctrl_x + 2;
                                int bx_num = bx_minus + btn_w + gap;
                                int num_w = ctrl_w - (btn_w*2 + gap*2) - 4;
                                int bx_plus = bx_num + num_w + gap;
                                memset_rect(out_rgba, w, h, pitch, bx_minus, ctrl_y + 1, btn_w, ctrl_h - 2, Color{60,60,70,255});
                                memset_rect(out_rgba, w, h, pitch, bx_num, ctrl_y + 1, num_w, ctrl_h - 2, Color{36,36,46,255});
                                memset_rect(out_rgba, w, h, pitch, bx_plus, ctrl_y + 1, btn_w, ctrl_h - 2, Color{60,60,70,255});
                                // render minus/plus/number labels
                                auto bm_minus = render_text_to_rgba("-", 1.0f, {220,220,220,255});
                                if (!bm_minus.pixels.empty()) {
                                    int tx = bx_minus + (btn_w - bm_minus.width) / 2;
                                    int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_minus.height) / 2;
                                    for (int yy = 0; yy < bm_minus.height; ++yy) {
                                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                                        for (int xx = 0; xx < bm_minus.width; ++xx) {
                                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                            const unsigned char* src = &bm_minus.pixels[(yy * bm_minus.width + xx) * 4];
                                            float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                            else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                                        }
                                    }
                                }
                                auto bm_plus = render_text_to_rgba("+", 1.0f, {220,220,220,255});
                                if (!bm_plus.pixels.empty()) {
                                    int tx = bx_plus + (btn_w - bm_plus.width) / 2;
                                    int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_plus.height) / 2;
                                    for (int yy = 0; yy < bm_plus.height; ++yy) {
                                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                                        for (int xx = 0; xx < bm_plus.width; ++xx) {
                                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                            const unsigned char* src = &bm_plus.pixels[(yy * bm_plus.width + xx) * 4];
                                            float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                            else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                                        }
                                    }
                                }
                                std::string s = std::to_string(display_ch);
                                auto bm_num = render_text_to_rgba(s, 0.9f, {230,230,235,255});
                                if (!bm_num.pixels.empty()) {
                                    int tx = bx_num + (num_w - bm_num.width) / 2;
                                    int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_num.height) / 2;
                                    for (int yy = 0; yy < bm_num.height; ++yy) {
                                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                                        for (int xx = 0; xx < bm_num.width; ++xx) {
                                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                            const unsigned char* src = &bm_num.pixels[(yy * bm_num.width + xx) * 4];
                                            float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                            else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                                        }
                                    }
                                }
                                return true;
                            }
                        }
                        return false;
                    };
                    if (try_draw_for_table(ctx->container_table)) drew = true;
                    if (!drew) {
                        for (size_t mi = 0; mi < ctx->module_tables.size(); ++mi) {
                            if (try_draw_for_table(ctx->module_tables[mi])) { drew = true; break; }
                        }
                    }
                }
                  // Diagnostic: overlay rect/button logging removed to reduce runtime noise
                  // printf("canvas_render: overlay id=%d rect=[%d,%d,%d,%d] btn=[%d,%d,%d,%d] overlay_rope_idx=%d\n",
                  //        kv.first, rx0, ry0, rx1, ry1, bx, by, btn_w, btn_h, overlay_rope_idx);
                int mode = 0; // default
                if (overlay_rope_idx >= 0) {
                    auto check_table_for_mode = [&](GP_TableContext* t)->bool{
                        if (!t) return false;
                        int mgcount = gp_table_get_meta_group_count(t);
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            if (!mg) continue;
                            int ar = -1, av = -1;
                            if (!gp_table_meta_get_anchor(t, mg, &ar, &av)) continue;
                            if (ar == overlay_rope_idx) {
                                int32_t got = 0;
                                if (gp_table_meta_get_ring_mode(t, mg, &got)) mode = got;
                                return true;
                            }
                        }
                        return false;
                    };
                    if (!check_table_for_mode(ctx->container_table)) {
                        for (size_t mi = 0; mi < ctx->module_tables.size(); ++mi) {
                            if (check_table_for_mode(ctx->module_tables[mi])) break;
                        }
                    }
                }
                Color btn_col;
                const char* lbl = "R";
                if (mode == 0) { btn_col = Color{100,140,120,220}; lbl = "R"; }
                else if (mode == 1) { btn_col = Color{120,100,140,220}; lbl = "C"; }
                else { btn_col = Color{140,120,100,220}; lbl = "D"; }
                // Also draw the numeric channel-group control co-located with
                // the overlay's binding/mode button by resolving the dangling
                // widget for any meta-group anchored to this overlay rope.
                {
                    int display_ch = 0;
                    float wpos_local[3] = {0.0f,0.0f,0.0f};
                    bool have_wpos = false;
                    auto find_widget_pos = [&](GP_TableContext* t)->bool {
                        if (!t) return false;
                        int mgcount = gp_table_get_meta_group_count(t);
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            if (!mg) continue;
                            int ar = -1, av = -1;
                            if (!gp_table_meta_get_anchor(t, mg, &ar, &av)) continue;
                            if (ar != overlay_rope_idx) continue;
                            gp_table_meta_get_channel_group(t, mg, &display_ch);
                            int wwid = -1;
                            if (gp_table_meta_get_dangling_widget_id(t, mg, &wwid) && wwid >= 0) {
                                if (gp_table_get_widget_position(t, wwid, wpos_local)) { have_wpos = true; return true; }
                            }
                        }
                        return false;
                    };
                    if (!find_widget_pos(ctx->container_table)) {
                        for (size_t mi = 0; mi < ctx->module_tables.size(); ++mi) {
                            if (find_widget_pos(ctx->module_tables[mi])) break;
                        }
                    }
                    if (have_wpos) {
                        int wcx = static_cast<int>(std::lround(wpos_local[0] - static_cast<float>(ctx->offset_x)));
                        int wcy = static_cast<int>(std::lround(wpos_local[1] - static_cast<float>(ctx->offset_y)));
                        int ctrl_w = 88; int ctrl_h = 18;
                        // place numeric control centered below the mode/button
                        int ctrl_x = bx + (btn_w/2) - (ctrl_w/2);
                        int ctrl_y = by + btn_h + 6;
                        printf("drawnum_overlay_btn: overlay_rope_idx=%d bx=%d by=%d ctrl_w=%d ctrl_h=%d\n", overlay_rope_idx, bx, by, ctrl_w, ctrl_h);
                        fflush(stdout);
                        memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, ctrl_h, Color{40,40,50,220});
                        // outline (1px) in bright green so numeric control is obvious
                        memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, 1, Color{80,200,120,220});
                        memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y + ctrl_h - 1, ctrl_w, 1, Color{16,16,20,255});
                        memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, 1, ctrl_h, Color{80,200,120,220});
                        memset_rect(out_rgba, w, h, pitch, ctrl_x + ctrl_w - 1, ctrl_y, 1, ctrl_h, Color{80,200,120,220});
                        int btn_w = ctrl_h; int gap = 6;
                        int bx_minus = ctrl_x + 2;
                        int bx_num = bx_minus + btn_w + gap;
                        int num_w = ctrl_w - (btn_w*2 + gap*2) - 4;
                        int bx_plus = bx_num + num_w + gap;
                        memset_rect(out_rgba, w, h, pitch, bx_minus, ctrl_y + 1, btn_w, ctrl_h - 2, Color{60,60,70,255});
                        memset_rect(out_rgba, w, h, pitch, bx_num, ctrl_y + 1, num_w, ctrl_h - 2, Color{36,36,46,255});
                        memset_rect(out_rgba, w, h, pitch, bx_plus, ctrl_y + 1, btn_w, ctrl_h - 2, Color{60,60,70,255});
                        auto bm_minus = render_text_to_rgba("-", 1.0f, {220,220,220,255});
                        if (!bm_minus.pixels.empty()) {
                            int tx = bx_minus + (btn_w - bm_minus.width) / 2;
                            int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_minus.height) / 2;
                            for (int yy = 0; yy < bm_minus.height; ++yy) {
                                int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                                for (int xx = 0; xx < bm_minus.width; ++xx) {
                                    int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                    const unsigned char* src = &bm_minus.pixels[(yy * bm_minus.width + xx) * 4];
                                    float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                    else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                                }
                            }
                        }
                        auto bm_plus = render_text_to_rgba("+", 1.0f, {220,220,220,255});
                        if (!bm_plus.pixels.empty()) {
                            int tx = bx_plus + (btn_w - bm_plus.width) / 2;
                            int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_plus.height) / 2;
                            for (int yy = 0; yy < bm_plus.height; ++yy) {
                                int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                                for (int xx = 0; xx < bm_plus.width; ++xx) {
                                    int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                    const unsigned char* src = &bm_plus.pixels[(yy * bm_plus.width + xx) * 4];
                                    float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                    else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                                }
                            }
                        }
                        std::string s = std::to_string(display_ch);
                        auto bm_num = render_text_to_rgba(s, 0.9f, {230,230,235,255});
                        if (!bm_num.pixels.empty()) {
                            int tx = bx_num + (num_w - bm_num.width) / 2;
                            int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_num.height) / 2;
                            for (int yy = 0; yy < bm_num.height; ++yy) {
                                int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                                for (int xx = 0; xx < bm_num.width; ++xx) {
                                    int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                    const unsigned char* src = &bm_num.pixels[(yy * bm_num.width + xx) * 4];
                                    float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                    else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                                }
                            }
                        }
                    }
                }
                // TEMP VISUAL MARKER: draw a translucent red box slightly
                // larger than the mode button so it's obvious if this path
                // is executed. Remove after verification.
                memset_rect(out_rgba, w, h, pitch, bx - 4, by - 4, btn_w + 8, btn_h + 8, Color{220,40,40,140});
                memset_rect(out_rgba, w, h, pitch, bx, by, btn_w, btn_h, btn_col);
                // Force-draw numeric control centered below the mode/button (always)
                {
                    int ctrl_w = 88; int ctrl_h = 18;
                    // center horizontally on the button, place below with padding
                    int ctrl_x = bx + (btn_w/2) - (ctrl_w/2);
                    int ctrl_y = by + btn_h + 6;
                    int display_ch = 0;
                    // Prefer to find a meta-group associated with this overlay rope
                    // by checking dangling-rope info first (widget-attached), then
                    // falling back to anchor/members. This ensures the same meta-group
                    // is used for both hit-testing (which prefers widget/dangling info)
                    // and rendering so the displayed number updates immediately.
                    auto find_ch = [&](GP_TableContext* t)->bool {
                        if (!t) return false;
                        // If the overlay has an authoritative binding to a table/meta-group,
                        // prefer that immediately (this avoids rescanning ropes and ensures
                        // the displayed number matches direct overlay bindings).
                        if (ov.meta_table && ov.meta_mg) {
                            if (ov.meta_table == t) {
                                gp_table_meta_get_channel_group(t, ov.meta_mg, &display_ch);
                                return true;
                            }
                        }
                        int mgcount = gp_table_get_meta_group_count(t);
                        // first pass: match dangling-rope info
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            if (!mg) continue;
                            int dr=-1,dv=-1; gp_table_meta_get_dangling_rope_info(t, mg, &dr, &dv);
                            if (dr == overlay_rope_idx) { gp_table_meta_get_channel_group(t, mg, &display_ch); return true; }
                        }
                        // second pass: match anchor
                        for (int mgi = 0; mgi < mgcount; ++mgi) {
                            GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                            if (!mg) continue;
                            int ar=-1,av=-1; if (!gp_table_meta_get_anchor(t, mg, &ar, &av)) continue;
                            if (ar != overlay_rope_idx) continue;
                            gp_table_meta_get_channel_group(t, mg, &display_ch);
                            return true;
                        }
                        return false;
                    };
                    if (!find_ch(ctx->container_table)) for (size_t mi=0; mi<ctx->module_tables.size(); ++mi) if (find_ch(ctx->module_tables[mi])) break;
                    memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, ctrl_h, Color{24,36,44,220});
                    // cyan outline
                    memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, 1, Color{80,220,220,220});
                    memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y + ctrl_h - 1, ctrl_w, 1, Color{16,16,20,255});
                    memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, 1, ctrl_h, Color{80,220,220,220});
                    memset_rect(out_rgba, w, h, pitch, ctrl_x + ctrl_w - 1, ctrl_y, 1, ctrl_h, Color{80,220,220,220});
                    int btn_w2 = ctrl_h; int gap = 6;
                    int bx_minus = ctrl_x + 2;
                    int bx_num = bx_minus + btn_w2 + gap;
                    int num_w = ctrl_w - (btn_w2*2 + gap*2) - 4;
                    int bx_plus = bx_num + num_w + gap;
                    memset_rect(out_rgba, w, h, pitch, bx_minus, ctrl_y + 1, btn_w2, ctrl_h - 2, Color{60,60,70,255});
                    memset_rect(out_rgba, w, h, pitch, bx_num, ctrl_y + 1, num_w, ctrl_h - 2, Color{36,36,46,255});
                    memset_rect(out_rgba, w, h, pitch, bx_plus, ctrl_y + 1, btn_w2, ctrl_h - 2, Color{60,60,70,255});
                    std::string s = std::to_string(display_ch);
                    auto bm_num = render_text_to_rgba(s, 0.9f, {230,230,235,255});
                    if (!bm_num.pixels.empty()) {
                        int tx = bx_num + (num_w - bm_num.width) / 2;
                        int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_num.height) / 2;
                        for (int yy = 0; yy < bm_num.height; ++yy) {
                            int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                            for (int xx = 0; xx < bm_num.width; ++xx) {
                                int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                                uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                                const unsigned char* src = &bm_num.pixels[(yy * bm_num.width + xx) * 4];
                                float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                                else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                            }
                        }
                    }
                }
                auto tb = render_text_to_rgba(std::string(lbl), 1.1f, {240,240,240,255});
                if (!tb.pixels.empty()) {
                    int tx = bx + (btn_w - tb.width) / 2;
                    int ty = by + (btn_h - tb.height) / 2;
                    for (int yy = 0; yy < tb.height; ++yy) {
                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < tb.width; ++xx) {
                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &tb.pixels[(yy * tb.width + xx) * 4];
                            float sa = src[3] / 255.0f;
                            if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) {
                                for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                                dst[3] = 255;
                            }
                        }
                    }
                }
            }
        }
    }

    auto draw_polyline = [&](const float* verts, int count, Color col) {
        if (!verts || count < 2) return;
        table_draw_rope_polyline(out_rgba, w, h, pitch, verts, count, simple_line_radius, col.r, col.g, col.b, col.a);
    };
    auto draw_vertices = [&](const float* verts, int count, Color col) {
        if (!verts || count <= 0) return;
        for (int vi = 0; vi < count; ++vi) {
            int cx = static_cast<int>(std::lround(verts[vi * 2 + 0]));
            int cy = static_cast<int>(std::lround(verts[vi * 2 + 1]));
            draw_blob_blend(out_rgba, w, h, pitch, cx, cy, simple_dot_radius, col);
        }
    };

    // Prefer manager-supplied immutable network snapshot for rendering the connection graph.
    // The snapshot contains only endpoint keys and edges (no UI data). Map endpoint keys
    // to module centers for visual placement (canvas module positions are UI-owned).
    GP_TableContext* root_table = canvas_ensure_root_table(ctx);
    std::shared_ptr<ThreadManager::NetworkSnapshot> snap;
    ThreadManager* tm = ThreadManager::global();
    if (tm && root_table) snap = tm->get_table_snapshot(root_table);
    if (snap && !snap->edges.empty() && !simple_render) {
        // Draw simple straight lines between module centers derived from endpoint keys.
        for (size_t ei = 0; ei < snap->edges.size(); ++ei) {
            const auto &se = snap->edges[ei];
            // Resolve endpoints using the canonical indices from the snapshot.
            if (se.a_idx >= snap->nodes.size() || se.b_idx >= snap->nodes.size()) continue;
            uint64_t a_key = snap->nodes[se.a_idx];
            uint64_t b_key = snap->nodes[se.b_idx];
            int a_module = static_cast<int>((a_key >> 32) & 0xFFFFFFFFu);
            int b_module = static_cast<int>((b_key >> 32) & 0xFFFFFFFFu);
            if (a_module < 0 || a_module >= static_cast<int>(ctx->modules.size())) continue;
            if (b_module < 0 || b_module >= static_cast<int>(ctx->modules.size())) continue;
            int ax = ctx->modules[static_cast<size_t>(a_module)].x + ctx->modules[static_cast<size_t>(a_module)].w / 2 - ctx->offset_x;
            int ay = ctx->modules[static_cast<size_t>(a_module)].y + ctx->modules[static_cast<size_t>(a_module)].h / 2 - ctx->offset_y;
            int bx = ctx->modules[static_cast<size_t>(b_module)].x + ctx->modules[static_cast<size_t>(b_module)].w / 2 - ctx->offset_x;
            int by = ctx->modules[static_cast<size_t>(b_module)].y + ctx->modules[static_cast<size_t>(b_module)].h / 2 - ctx->offset_y;
            // draw a simple line (Bresenham-ish) with light gray color
            Color col{200,200,200,180};
            int dx = std::abs(bx - ax), sx = ax < bx ? 1 : -1;
            int dy = -std::abs(by - ay), sy = ay < by ? 1 : -1;
            int err = dx + dy;
            int x0 = ax, y0 = ay;
            while (true) {
                if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h) {
                    uint8_t* px = out_rgba + (y0 * pitch) + (x0 * 4);
                    blend_pixel(px, col.r, col.g, col.b, col.a);
                }
                if (x0 == bx && y0 == by) break;
                int e2 = 2 * err;
                if (e2 >= dy) { err += dy; x0 += sx; }
                if (e2 <= dx) { err += dx; y0 += sy; }
            }
        }
    } else {
        // Fallback: legacy per-edge rope rendering using canvas-local rope sim/indices
        // forward-declare shared fallback (implemented in table_abi_rope_draw.inl)
        extern void draw_fallback_rope(uint8_t* out_rgba, int w, int h, int pitch, float ax, float ay, float bx, float by, int jacket_px, int jacket_border, Color rope_col, int fallback_mode);
        for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
            int ridx = ctx->edges[ei].rope_idx;
            if (ridx < 0) continue;
            int vc = sim ? rope_sim_get_vertex_count(sim, ridx) : 0;
            if (vc < 2) {
                // Simulator missing or too few verts: compute endpoints and draw unified fallback.
                const auto &edge = ctx->edges[ei].desc;
                // Resolve endpoints from module hitboxes (same logic as earlier)
                int ax = 0, ay = 0, bx = 0, by = 0;
                bool resolvedA = false, resolvedB = false;
                if (edge.a_module >= 0 && edge.a_module < static_cast<int>(module_hitboxes.size()) && !module_hitboxes[edge.a_module].empty()) {
                    for (const auto &hb : module_hitboxes[edge.a_module]) {
                        if ((hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG || hb.part == GP_TABLE_HIT_LED_TABLE) && resolve_contact_index(ctx, edge.a_module, hb) == edge.a_contact_idx) {
                            int local_x = (hb.x0 + hb.x1) / 2;
                            int local_y = (hb.y0 + hb.y1) / 2;
                            ax = ctx->modules[edge.a_module].x + local_x - ctx->offset_x;
                            ay = ctx->modules[edge.a_module].y + local_y - ctx->offset_y;
                            resolvedA = true; break;
                        }
                    }
                }
                if (edge.b_module >= 0 && edge.b_module < static_cast<int>(module_hitboxes.size()) && !module_hitboxes[edge.b_module].empty()) {
                    for (const auto &hb : module_hitboxes[edge.b_module]) {
                        if ((hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG || hb.part == GP_TABLE_HIT_LED_TABLE) && resolve_contact_index(ctx, edge.b_module, hb) == edge.b_contact_idx) {
                            int local_x = (hb.x0 + hb.x1) / 2;
                            int local_y = (hb.y0 + hb.y1) / 2;
                            bx = ctx->modules[edge.b_module].x + local_x - ctx->offset_x;
                            by = ctx->modules[edge.b_module].y + local_y - ctx->offset_y;
                            resolvedB = true; break;
                        }
                    }
                }
                if (!resolvedA || !resolvedB) continue;
                uint32_t flags = 0u;
                if (ei < ctx->edges.size()) flags = ctx->edges[ei].subgroup_flags;
                Color rope_col = (flags != 0u) ? subgroup_flags_to_color(ctx, flags, 220) : Color{200,200,200,220};
                draw_fallback_rope(out_rgba, w, h, pitch, static_cast<float>(ax), static_cast<float>(ay), static_cast<float>(bx), static_cast<float>(by), ctx->jacket_px, ctx->jacket_border, rope_col, 1);
                continue;
            }
            std::vector<float> verts(static_cast<size_t>(vc) * 2);
            int got = sim ? rope_sim_get_vertices(sim, ridx, verts.data(), static_cast<int>(verts.size())) : 0;
            if (got <= 0) continue;
            std::vector<float> verts_view(static_cast<size_t>(got) * 2);
            for (int vi = 0; vi < got; ++vi) {
                verts_view[vi * 2 + 0] = verts[vi * 2 + 0] - static_cast<float>(ctx->offset_x);
                verts_view[vi * 2 + 1] = verts[vi * 2 + 1] - static_cast<float>(ctx->offset_y);
            }
            int jacket_px = ctx->jacket_px;
            int jacket_border = ctx->jacket_border;
            uint32_t subgroup_flags = ctx->edges[ei].subgroup_flags;
            Color rope_col = (subgroup_flags != 0u) ? subgroup_flags_to_color(ctx, subgroup_flags, 220) : Color{200,200,200,220};
            if (simple_render) {
                draw_polyline(verts_view.data(), got, rope_col);
                draw_vertices(verts_view.data(), got, rope_col);
            } else {
                if (subgroup_flags != 0u) {
                    float hue = subgroup_flags_to_hue(ctx, subgroup_flags);
                    float hue_vals[1] = { hue };
                    table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, verts_view.data(), got, jacket_px, jacket_border, hue_vals, 1, 3, 0.65f);
                } else {
                    table_draw_rope_curve_blend(out_rgba, w, h, pitch, verts_view.data(), got, jacket_px, jacket_border, 200, 200, 200, 180, 3);
                }
                if (!no_lighting) {
                    int glow_r = std::max(2, jacket_px * 2);
                    draw_rope_light_falloff(out_rgba, w, h, pitch, verts_view.data(), got, edge_light_a[ei], edge_light_b[ei], rope_decay, glow_r);
                }
            }
        }
    }

    // render provisional prospective rope (follows mouse) if present
    if (ctx->prospective_rope_idx >= 0) {
        int ridx = ctx->prospective_rope_idx;
        int vc = sim ? rope_sim_get_vertex_count(sim, ridx) : 0;
        if (vc >= 2) {
            std::vector<float> verts(static_cast<size_t>(vc) * 2);
            int got = sim ? rope_sim_get_vertices(sim, ridx, verts.data(), static_cast<int>(verts.size())) : 0;
            if (got > 0) {
                std::vector<float> verts_view(static_cast<size_t>(got) * 2);
                for (int vi = 0; vi < got; ++vi) {
                    verts_view[vi * 2 + 0] = verts[vi * 2 + 0] - static_cast<float>(ctx->offset_x);
                    verts_view[vi * 2 + 1] = verts[vi * 2 + 1] - static_cast<float>(ctx->offset_y);
                }
                int jacket_px = ctx->jacket_px;
                int jacket_border = ctx->jacket_border;
                int samples_per_segment = 3;
                uint32_t flags = ctx->selected_tool_subgroup_flags;
                Color rope_col = (flags != 0u) ? subgroup_flags_to_color(ctx, flags, 220) : Color{200,200,200,220};
                if (simple_render) {
                    draw_polyline(verts_view.data(), got, rope_col);
                    draw_vertices(verts_view.data(), got, rope_col);
                } else {
                    if (flags != 0u) {
                        float hue = subgroup_flags_to_hue(ctx, flags);
                        float hue_vals[1] = { hue };
                        table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, verts_view.data(), got, jacket_px, jacket_border, hue_vals, 1, samples_per_segment, 0.55f);
                    } else {
                        table_draw_rope_curve_blend(out_rgba, w, h, pitch, verts_view.data(), got, jacket_px, jacket_border, 200, 200, 200, 180, samples_per_segment);
                    }
                }
            }
        }
    }

    // Render ring entries (short sampled segments around ring u) from all known tables.
    // Also collect ring positions grouped by ring_key and draw continuous splines
    // connecting groups of rings (uses the same spline drawer as rope rendering).
    {
        std::vector<GP_TableContext*> draw_tables;
        if (ctx->container_table) draw_tables.push_back(ctx->container_table);
        for (GP_TableContext* mt : ctx->module_tables) if (mt) draw_tables.push_back(mt);

        std::unordered_map<unsigned long long, int> ring_mode_by_key;
        for (GP_TableContext* rt : draw_tables) {
            if (!rt) continue;
            int mg_count = gp_table_get_meta_group_count(rt);
            for (int mgi = 0; mgi < mg_count; ++mgi) {
                GP_MetaGroup* mg = gp_table_get_meta_group(rt, mgi);
                if (!mg) continue;
                unsigned long long mgid = 0ull;
                gp_table_meta_get_id(rt, mg, &mgid);
                if (mgid == 0ull) continue;
                int mode = 0;
                gp_table_meta_get_ring_mode(rt, mg, &mode);
                ring_mode_by_key[mgid] = mode;
            }
        }

        struct RingGroupData {
            std::vector<std::pair<float,float>> pts;
            int rope_idx = -1;
        };
        // collect per-key ring world positions (canvas coords)
        std::unordered_map<unsigned long long, RingGroupData> ring_groups;

        for (GP_TableContext* rt : draw_tables) {
            if (!rt) continue;
            RopeSim* tsim = gp_table_get_rope_sim(rt);
            if (!tsim) continue;
            int ring_count = gp_table_get_ring_edge_count(rt);
            for (int rei = 0; rei < ring_count; ++rei) {
                int ring_id = -1; unsigned long long ring_key = 0ull;
                if (!gp_table_get_ring_edge(rt, rei, &ring_id, &ring_key)) continue;
                if (ring_id < 0) continue;
                int parent_ridx = -1;
                if (!rope_sim_get_ring_rope_index(tsim, ring_id, &parent_ridx) || parent_ridx < 0) continue;
                int vc = rope_sim_get_vertex_count(tsim, parent_ridx);
                if (vc < 2) continue;
                std::vector<float> verts(static_cast<size_t>(vc) * 2);
                int got = rope_sim_get_vertices(tsim, parent_ridx, verts.data(), static_cast<int>(verts.size()));
                if (got <= 0) continue;
                float u = 0.0f;
                rope_sim_get_ring_u(tsim, ring_id, &u);

                // compute a central world point for this ring (linear interp using u)
                float idxf = u * static_cast<float>(got - 1);
                int lo = static_cast<int>(std::floor(idxf));
                int hi = std::min(got - 1, lo + 1);
                float frac = idxf - static_cast<float>(lo);
                float wx = (1.0f - frac) * verts[2*lo+0] + frac * verts[2*hi+0];
                float wy = (1.0f - frac) * verts[2*lo+1] + frac * verts[2*hi+1];
                // transform to canvas coords and collect
                float cx = wx - static_cast<float>(ctx->offset_x);
                float cy = wy - static_cast<float>(ctx->offset_y);
                auto &group = ring_groups[ring_key];
                if (group.rope_idx < 0) group.rope_idx = parent_ridx;
                group.pts.emplace_back(cx, cy);
                if (simple_render) {
                    uint32_t flags = 0;
                    gp_table_ring_get_subgroup_flags(rt, rei, &flags);
                    Color ring_col = (flags != 0u) ? subgroup_flags_to_color(ctx, flags, 230) : Color{220,160,80,230};
                    draw_blob_blend(out_rgba, w, h, pitch, static_cast<int>(std::lround(cx)), static_cast<int>(std::lround(cy)), simple_dot_radius, ring_col);
                    continue;
                }

                // draw a short sampled segment centered at u (as before)
                int center_idx = static_cast<int>(idxf + 0.5f);
                int half = 2;
                int start = std::max(0, center_idx - half);
                int end = std::min(got - 1, center_idx + half);
                int use_count = end - start + 1;
                if (use_count < 2) continue;
                std::vector<float> verts_view(static_cast<size_t>(use_count) * 2);
                for (int vi = start; vi <= end; ++vi) {
                    int idx = vi - start;
                    verts_view[idx * 2 + 0] = verts[vi * 2 + 0] - static_cast<float>(ctx->offset_x);
                    verts_view[idx * 2 + 1] = verts[vi * 2 + 1] - static_cast<float>(ctx->offset_y);
                }
                int jacket_px = ctx->jacket_px;
                int jacket_border = ctx->jacket_border;
                int samples_per_segment = 3;
                uint32_t flags = 0;
                gp_table_ring_get_subgroup_flags(rt, rei, &flags);
                if (flags != 0u) {
                    float hue = subgroup_flags_to_hue(ctx, flags);
                    float hue_vals[1] = { hue };
                    table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, verts_view.data(), use_count, jacket_px, jacket_border, hue_vals, 1, samples_per_segment, 0.65f);
                } else {
                    table_draw_rope_curve_blend(out_rgba, w, h, pitch, verts_view.data(), use_count, jacket_px, jacket_border, 220, 160, 80, 220, samples_per_segment);
                }
                if (!no_lighting) {
                    ContactLight la{}, lb{};
                    if (resolve_edge_light_for_rope(parent_ridx, &la, &lb)) {
                        int glow_r = std::max(2, jacket_px * 2);
                        draw_rope_light_falloff(out_rgba, w, h, pitch, verts_view.data(), use_count, la, lb, rope_decay, glow_r);
                    }
                }
            }
        }

        // Now draw connecting splines for each group of rings sharing the same key
        for (auto &kv : ring_groups) {
            if (simple_render) continue;
            auto it_mode = ring_mode_by_key.find(kv.first);
            if (it_mode != ring_mode_by_key.end() && it_mode->second == 2) {
                continue;
            }
            auto &group = kv.second;
            auto &pts = group.pts;
            if (pts.size() < 2) continue;
            // build simple polyline in collected order
            std::vector<float> poly(static_cast<size_t>(pts.size() * 2));
            for (size_t i = 0; i < pts.size(); ++i) { poly[2*i+0] = pts[i].first; poly[2*i+1] = pts[i].second; }
            int jacket_px = ctx->jacket_px;
            int jacket_border = ctx->jacket_border;
            // neutral colored connecting rope
            table_draw_rope_curve_blend(out_rgba, w, h, pitch, poly.data(), static_cast<int>(pts.size()), jacket_px, jacket_border, 180, 140, 100, 220, 3);
            if (!no_lighting) {
                ContactLight la{}, lb{};
                if (resolve_edge_light_for_rope(group.rope_idx, &la, &lb)) {
                    int glow_r = std::max(2, jacket_px * 2);
                    draw_rope_light_falloff(out_rgba, w, h, pitch, poly.data(), static_cast<int>(pts.size()), la, lb, rope_decay, glow_r);
                }
            }
        }
    }

    // Render meta-group edges (treat meta edges as rope-like splines connecting stored vertices)
    {
        std::vector<GP_TableContext*> draw_tables;
        if (ctx->container_table) draw_tables.push_back(ctx->container_table);
        for (GP_TableContext* mt : ctx->module_tables) if (mt) draw_tables.push_back(mt);
        for (GP_TableContext* rt : draw_tables) {
            if (!rt) continue;
            RopeSim* tsim = gp_table_get_rope_sim(rt);
            int mg_count = gp_table_get_meta_group_count(rt);
            for (int mgi = 0; mgi < mg_count; ++mgi) {
                GP_MetaGroup* mg = gp_table_get_meta_group(rt, mgi);
                if (!mg) continue;
                int dr=-1,dv=-1; gp_table_meta_get_dangling_rope_info(rt, mg, &dr, &dv);
                int ch_tmp = 0; gp_table_meta_get_channel_group(rt, mg, &ch_tmp);
                (void)ch_tmp;
                int vcount = gp_table_meta_get_vertex_count(rt, mg);
                if (vcount <= 0) continue;
                // For each vertex, prefer sim-sampled world position when available,
                // otherwise fall back to the table-projected vertex index lookup.
                std::vector<float> poly(static_cast<size_t>(vcount * 2));
                bool got_all = true;
                for (int vi = 0; vi < vcount; ++vi) {
                    float wx = 0.0f, wy = 0.0f;
                    bool used_sim = false;
                    if (tsim) {
                        int32_t sim_idx = -1;
                        if (gp_table_meta_get_sim_group_index(rt, mg, &sim_idx) && sim_idx >= 0) {
                            float wpos[3];
                            if (rope_sim_meta_group_get_member_world_pos(tsim, sim_idx, vi, wpos)) {
                                wx = wpos[0]; wy = wpos[1]; used_sim = true;
                            }
                        }
                    }
                    if (!used_sim) {
                        int rope_idx = -1, vert_idx = -1;
                        if (!gp_table_meta_get_vertex(rt, mg, vi, &rope_idx, &vert_idx)) { got_all = false; break; }
                        if (rope_idx < 0 || vert_idx < 0) { got_all = false; break; }
                        // get projected vertices of that rope from the table (table coordinates)
                        int maxv = 256;
                        std::vector<float> proj(static_cast<size_t>(maxv * 2));
                        int got = gp_table_get_projected_rope_vertices(rt, rope_idx, proj.data(), static_cast<int>(proj.size()));
                        if (got <= vert_idx) { got_all = false; break; }
                        wx = proj[vert_idx * 2 + 0];
                        wy = proj[vert_idx * 2 + 1];
                    }
                    // convert to canvas coords
                    poly[vi * 2 + 0] = wx - static_cast<float>(ctx->offset_x);
                    poly[vi * 2 + 1] = wy - static_cast<float>(ctx->offset_y);
                }
                if (!got_all) continue;
                int jacket_px = ctx->jacket_px;
                int jacket_border = ctx->jacket_border;
                int ring_mode = 0;
                gp_table_meta_get_ring_mode(rt, mg, &ring_mode);
                int light_rope_idx = -1;
                int light_vert_idx = -1;
                gp_table_meta_get_anchor(rt, mg, &light_rope_idx, &light_vert_idx);
                if (light_rope_idx < 0) {
                    gp_table_meta_get_vertex(rt, mg, 0, &light_rope_idx, &light_vert_idx);
                }
                // draw meta-group connector spline using subgroup color when present
                uint32_t mg_flags = 0u;
                gp_table_meta_get_subgroup_flags(rt, mg, &mg_flags);
                if (ring_mode == 2) {
                    RopeSim* tsim = gp_table_get_rope_sim(rt);
                    int center_rope = -1;
                    float cx = 0.0f;
                    float cy = 0.0f;
                    int center_members = 0;
                    for (int vi = 0; vi < vcount; ++vi) {
                        int r = -1, v = -1;
                        if (!gp_table_meta_get_vertex(rt, mg, vi, &r, &v)) continue;
                        if (tsim && rope_sim_is_meta_rope(tsim, r)) {
                            center_rope = r;
                            continue;
                        }
                        float wx = poly[vi * 2 + 0] + static_cast<float>(ctx->offset_x);
                        float wy = poly[vi * 2 + 1] + static_cast<float>(ctx->offset_y);
                        cx += wx;
                        cy += wy;
                        ++center_members;
                    }
                    if (center_members > 0) {
                        cx /= static_cast<float>(center_members);
                        cy /= static_cast<float>(center_members);
                    }
                    int stem_idx = -1;
                    float stem_dist = 1e9f;
                    if (center_rope >= 0) {
                        int stem_vc = rope_sim_get_vertex_count(tsim, center_rope);
                        for (int vi = 0; vi < vcount; ++vi) {
                            int r = -1, v = -1;
                            if (!gp_table_meta_get_vertex(rt, mg, vi, &r, &v)) continue;
                            if (r != center_rope) continue;
                            float u = (stem_vc > 1) ? (static_cast<float>(v) / static_cast<float>(stem_vc - 1)) : 0.5f;
                            float du = std::fabs(u - 0.5f);
                            if (du < stem_dist) { stem_dist = du; stem_idx = vi; }
                        }
                    }
                    auto draw_star_edge = [&](float x0, float y0, float x1, float y1) {
                        float seg[4] = { x0 - ctx->offset_x, y0 - ctx->offset_y, x1 - ctx->offset_x, y1 - ctx->offset_y };
                        Color edge_col = (mg_flags != 0u) ? subgroup_flags_to_color(ctx, mg_flags, 220) : Color{160,200,210,220};
                        if (simple_render) {
                            draw_polyline(seg, 2, edge_col);
                            draw_vertices(seg, 2, edge_col);
                        } else {
                            if (mg_flags != 0u) {
                                float hue = subgroup_flags_to_hue(ctx, mg_flags);
                                float hue_vals[1] = { hue };
                                table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, seg, 2, jacket_px, jacket_border, hue_vals, 1, 3, 0.55f);
                            } else {
                                table_draw_rope_curve_blend(out_rgba, w, h, pitch, seg, 2, jacket_px, jacket_border, 160, 200, 210, 200, 3);
                            }
                            if (!no_lighting) {
                                ContactLight la{}, lb{};
                                if (resolve_edge_light_for_rope(center_rope, &la, &lb)) {
                                    int glow_r = std::max(2, jacket_px * 2);
                                    draw_rope_light_falloff(out_rgba, w, h, pitch, seg, 2, la, lb, rope_decay, glow_r);
                                }
                            }
                        }
                    };
                    if (center_members > 0) {
                        for (int vi = 0; vi < vcount; ++vi) {
                            int r = -1, v = -1;
                            if (!gp_table_meta_get_vertex(rt, mg, vi, &r, &v)) continue;
                            if (r == center_rope) continue;
                            float wx = poly[vi * 2 + 0] + static_cast<float>(ctx->offset_x);
                            float wy = poly[vi * 2 + 1] + static_cast<float>(ctx->offset_y);
                            draw_star_edge(cx, cy, wx, wy);
                        }
                        if (stem_idx >= 0) {
                            float sx = poly[stem_idx * 2 + 0] + static_cast<float>(ctx->offset_x);
                            float sy = poly[stem_idx * 2 + 1] + static_cast<float>(ctx->offset_y);
                            draw_star_edge(cx, cy, sx, sy);
                        }
                    }
                    continue;
                }
                if (mg_flags != 0u) {
                    if (simple_render) {
                        Color edge_col = subgroup_flags_to_color(ctx, mg_flags, 220);
                        draw_polyline(poly.data(), vcount, edge_col);
                        draw_vertices(poly.data(), vcount, edge_col);
                    } else {
                        float hue = subgroup_flags_to_hue(ctx, mg_flags);
                        float hue_vals[1] = { hue };
                        table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, poly.data(), vcount, jacket_px, jacket_border, hue_vals, 1, 3, 0.6f);
                    }
                } else {
                    if (simple_render) {
                        Color edge_col{160,200,210,220};
                        draw_polyline(poly.data(), vcount, edge_col);
                        draw_vertices(poly.data(), vcount, edge_col);
                    } else {
                        table_draw_rope_curve_blend(out_rgba, w, h, pitch, poly.data(), vcount, jacket_px, jacket_border, 160, 200, 210, 200, 3);
                    }
                }
                if (!simple_render && !no_lighting) {
                    ContactLight la{}, lb{};
                    if (resolve_edge_light_for_rope(light_rope_idx, &la, &lb)) {
                        int glow_r = std::max(2, jacket_px * 2);
                        draw_rope_light_falloff(out_rgba, w, h, pitch, poly.data(), vcount, la, lb, rope_decay, glow_r);
                    }
                }

                // Debug overlay: draw a small blob at each stored vertex and always log mapping
                // when environment variable NODUS_DEBUG_META is set. Bright red marks indicate
                // failed projected lookups.
                if (getenv("NODUS_DEBUG_META") != nullptr) {
                    bool any_bad = false;
                    std::string mapstr;
                    for (int vi = 0; vi < vcount; ++vi) {
                        int rope_idx = -1, vert_idx = -1;
                        gp_table_meta_get_vertex(rt, mg, vi, &rope_idx, &vert_idx);
                        int cx = static_cast<int>(poly[vi * 2 + 0] + 0.5f);
                        int cy = static_cast<int>(poly[vi * 2 + 1] + 0.5f);
                        Color bc{ static_cast<uint8_t>((rope_idx * 97) & 0xFF), static_cast<uint8_t>((rope_idx * 223) & 0xFF), static_cast<uint8_t>((vert_idx * 61) & 0xFF), 220 };
                        draw_blob_blend(out_rgba, w, h, pitch, cx, cy, std::max(2, jacket_px / 2), bc);

                        int maxv_dbg = 256;
                        std::vector<float> proj_dbg(static_cast<size_t>(maxv_dbg * 2));
                        int got_dbg = gp_table_get_projected_rope_vertices(rt, rope_idx, proj_dbg.data(), static_cast<int>(proj_dbg.size()));
                        if (got_dbg <= 0 || vert_idx < 0 || vert_idx >= got_dbg) {
                            any_bad = true;
                            Color bad{255,40,40,200};
                            draw_blob_blend(out_rgba, w, h, pitch, cx, cy, std::max(3, jacket_px), bad);
                            (void)got_dbg; (void)vi; (void)rope_idx; (void)vert_idx;
                        }

                        char buf[64]; std::snprintf(buf, sizeof(buf), "%d:%d", rope_idx, vert_idx);
                        if (!mapstr.empty()) mapstr += ",";
                        mapstr += buf;
                    }
                    (void)mapstr; (void)any_bad;
                }
            }
        }
    }

    // Render dangling widgets (blended blob + short sampled rope segment)
    {
        std::vector<GP_TableContext*> draw_tables;
        if (ctx->container_table) draw_tables.push_back(ctx->container_table);
        for (GP_TableContext* mt : ctx->module_tables) if (mt) draw_tables.push_back(mt);
        for (GP_TableContext* rt : draw_tables) {
            if (!rt) continue;
            RopeSim* tsim = gp_table_get_rope_sim(rt);
            if (!tsim) continue;
            int mg_count = gp_table_get_meta_group_count(rt);
            for (int mgi = 0; mgi < mg_count; ++mgi) {
                GP_MetaGroup* mg = gp_table_get_meta_group(rt, mgi);
                if (!mg) continue;
                int wid = -1;
                if (!gp_table_meta_get_dangling_widget_id(rt, mg, &wid)) continue;
                if (wid < 0) continue;
                // get widget world pos
                float wpos[3] = {0.0f,0.0f,0.0f};
                if (!gp_table_get_widget_position(rt, wid, wpos)) continue;
                int cx = static_cast<int>(std::lround(wpos[0] - static_cast<float>(ctx->offset_x)));
                int cy = static_cast<int>(std::lround(wpos[1] - static_cast<float>(ctx->offset_y)));
                // draw blob
                draw_blob_blend(out_rgba, w, h, pitch, cx, cy, 6, Color{200,60,60,200});

                // Draw a small numeric control above the widget to set channel group.
                // Layout: [ - ] [  NUM  ] [ + ] centered horizontally above the blob.
                int ctrl_w = 88;
                int ctrl_h = 18;
                int ctrl_x = cx - ctrl_w / 2;
                int ctrl_y = cy + 12; // place below the blob
                // background
                // debug: drawing meta control logging removed
                // printf("canvas_render: drawing meta control mg=%p wid=%d cx=%d cy=%d ctrl_x=%d ctrl_y=%d ctrl_w=%d ctrl_h=%d\n",
                //     (void*)mg, wid, cx, cy, ctrl_x, ctrl_y, ctrl_w, ctrl_h);
                // fflush(stdout);
                // Remove per-table dangling draw to avoid duplication; use the
                // unconditional overlay-forced draw so rendering & hit-tests
                // are driven from the same code path and update reliably.
                // (leave the debug print for diagnostics)
                (void)mg; (void)wid; (void)cx; (void)cy; (void)ctrl_w; (void)ctrl_h;
                // memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, ctrl_h, Color{40,40,50,220});
                // outline (1px) in bright green so numeric control is obvious
                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, ctrl_w, 1, Color{80,200,120,220});
                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y + ctrl_h - 1, ctrl_w, 1, Color{16,16,20,255});
                memset_rect(out_rgba, w, h, pitch, ctrl_x, ctrl_y, 1, ctrl_h, Color{80,200,120,220});
                memset_rect(out_rgba, w, h, pitch, ctrl_x + ctrl_w - 1, ctrl_y, 1, ctrl_h, Color{80,200,120,220});
                int btn_w = ctrl_h; int gap = 6;
                int bx_minus = ctrl_x + 2;
                int bx_num = bx_minus + btn_w + gap;
                int num_w = ctrl_w - (btn_w*2 + gap*2) - 4;
                int bx_plus = bx_num + num_w + gap;
                // Diagnostic: print control layout once per frame for visible widgets
                (void)mg; (void)wid; (void)cx; (void)cy; (void)ctrl_x; (void)ctrl_y; (void)num_w;
                // minus button
                // debug: meta control buttons logging removed
                // printf("canvas_render: meta control buttons mg=%p minus=[%d,%d,%d,%d] num=[%d,%d,%d,%d] plus=[%d,%d,%d,%d]\n",
                //     (void*)mg, bx_minus, ctrl_y + 1, btn_w, ctrl_h - 2, bx_num, ctrl_y + 1, num_w, ctrl_h - 2, bx_plus, ctrl_y + 1, btn_w, ctrl_h - 2);
                // fflush(stdout);
                memset_rect(out_rgba, w, h, pitch, bx_minus, ctrl_y + 1, btn_w, ctrl_h - 2, Color{60,60,70,255});
                // number background
                memset_rect(out_rgba, w, h, pitch, bx_num, ctrl_y + 1, num_w, ctrl_h - 2, Color{36,36,46,255});
                // plus button
                memset_rect(out_rgba, w, h, pitch, bx_plus, ctrl_y + 1, btn_w, ctrl_h - 2, Color{60,60,70,255});
                // render labels
                int ch_val = 0; gp_table_meta_get_channel_group(rt, mg, &ch_val);
                auto bm_minus = render_text_to_rgba("-", 1.0f, {220,220,220,255});
                if (!bm_minus.pixels.empty()) {
                    int tx = bx_minus + (btn_w - bm_minus.width) / 2;
                    int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_minus.height) / 2;
                    for (int yy = 0; yy < bm_minus.height; ++yy) {
                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < bm_minus.width; ++xx) {
                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &bm_minus.pixels[(yy * bm_minus.width + xx) * 4];
                            float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                        }
                    }
                }
                auto bm_plus = render_text_to_rgba("+", 1.0f, {220,220,220,255});
                if (!bm_plus.pixels.empty()) {
                    int tx = bx_plus + (btn_w - bm_plus.width) / 2;
                    int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_plus.height) / 2;
                    for (int yy = 0; yy < bm_plus.height; ++yy) {
                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < bm_plus.width; ++xx) {
                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &bm_plus.pixels[(yy * bm_plus.width + xx) * 4];
                            float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                        }
                    }
                }
                // number
                std::string s = std::to_string(ch_val);
                auto bm_num = render_text_to_rgba(s, 0.9f, {230,230,235,255});
                if (!bm_num.pixels.empty()) {
                    int tx = bx_num + (num_w - bm_num.width) / 2;
                    int ty = ctrl_y + 1 + ((ctrl_h - 2) - bm_num.height) / 2;
                    for (int yy = 0; yy < bm_num.height; ++yy) {
                        int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                        for (int xx = 0; xx < bm_num.width; ++xx) {
                            int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                            uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                            const unsigned char* src = &bm_num.pixels[(yy * bm_num.width + xx) * 4];
                            float sa = src[3] / 255.0f; if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                            else if (sa > 0.001f) { for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f)); dst[3] = 255; }
                        }
                    }
                }

                // If the meta-group created a short dangling rope for the widget,
                // draw that rope so the widget appears connected. Prefer the
                // meta-group's subgroup flags for coloring.
                int wrope = -1, wvid = -1;
                gp_table_meta_get_dangling_rope_info(rt, mg, &wrope, &wvid);
                if (wrope >= 0) {
                    int maxv_w = 512;
                    std::vector<float> wproj(static_cast<size_t>(maxv_w * 2));
                    int wgot = gp_table_get_projected_rope_vertices(rt, wrope, wproj.data(), static_cast<int>(wproj.size()));
                    if (wgot > 1) {
                        std::vector<float> wseg(static_cast<size_t>(wgot * 2));
                        for (int wi = 0; wi < wgot; ++wi) { wseg[wi*2+0] = wproj[wi*2+0] - static_cast<float>(ctx->offset_x); wseg[wi*2+1] = wproj[wi*2+1] - static_cast<float>(ctx->offset_y); }
                        int jacket_px_w = ctx->jacket_px;
                        int jacket_border_w = ctx->jacket_border;
                        uint32_t mf = 0u;
                        gp_table_meta_get_subgroup_flags(rt, mg, &mf);
                        Color rope_col = (mf != 0u) ? subgroup_flags_to_color(ctx, mf, 220) : Color{160,200,210,220};
                        if (simple_render) {
                            draw_polyline(wseg.data(), wgot, rope_col);
                            draw_vertices(wseg.data(), wgot, rope_col);
                        } else {
                            if (mf != 0u) {
                                float hue = subgroup_flags_to_hue(ctx, mf);
                                float hue_vals[1] = { hue };
                                table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, wseg.data(), wgot, jacket_px_w, jacket_border_w, hue_vals, 1, 3, 0.6f);
                            } else {
                                table_draw_rope_curve_blend(out_rgba, w, h, pitch, wseg.data(), wgot, jacket_px_w, jacket_border_w, 160, 200, 210, 200, 3);
                            }
                            if (!no_lighting) {
                                ContactLight la{}, lb{};
                                if (resolve_edge_light_for_rope(wrope, &la, &lb)) {
                                    int glow_r = std::max(2, jacket_px_w * 2);
                                    draw_rope_light_falloff(out_rgba, w, h, pitch, wseg.data(), wgot, la, lb, rope_decay, glow_r);
                                }
                            }
                        }
                    }
                }

                // draw short sampled rope segment around anchor vertex (first vertex)
                if (gp_table_meta_get_vertex_count(rt, mg) > 0) {
                    int rope_idx = -1, vert_idx = -1;
                    if (gp_table_meta_get_vertex(rt, mg, 0, &rope_idx, &vert_idx)) {
                        if (rope_idx >= 0 && vert_idx >= 0) {
                            // fetch projected rope verts from table
                            int maxv = 512;
                            std::vector<float> proj(static_cast<size_t>(maxv * 2));
                            int got = gp_table_get_projected_rope_vertices(rt, rope_idx, proj.data(), static_cast<int>(proj.size()));
                            if (got > 1 && vert_idx < got) {
                                int half = 3;
                                int start = std::max(0, vert_idx - half);
                                int end = std::min(got - 1, vert_idx + half);
                                int use_count = end - start + 1;
                                if (use_count >= 2) {
                                    std::vector<float> seg(static_cast<size_t>(use_count * 2));
                                    for (int vi = start; vi <= end; ++vi) {
                                        int idx = vi - start;
                                        seg[idx*2+0] = proj[vi*2+0] - static_cast<float>(ctx->offset_x);
                                        seg[idx*2+1] = proj[vi*2+1] - static_cast<float>(ctx->offset_y);
                                    }
                                    int jacket_px = std::max(1, ctx->jacket_px / 2);
                                    int jacket_border = std::max(1, ctx->jacket_border / 2);
                                    // Try to draw the widget's short rope using the same subgroup color
                                    // as any edge that maps to this rope so we don't obscure it.
                                    // prefer meta-group's own subgroup flags (set via lasso config)
                                    uint32_t mf = 0u;
                                    gp_table_meta_get_subgroup_flags(rt, mg, &mf);
                                    Color rope_col = (mf != 0u) ? subgroup_flags_to_color(ctx, mf, 200) : Color{200,160,160,200};
                                    if (simple_render) {
                                        draw_polyline(seg.data(), use_count, rope_col);
                                        draw_vertices(seg.data(), use_count, rope_col);
                                    } else {
                                        if (mf != 0u) {
                                            float hue = subgroup_flags_to_hue(ctx, mf);
                                            float hue_vals[1] = { hue };
                                            table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, seg.data(), use_count, jacket_px, jacket_border, hue_vals, 1, 3, 0.45f);
                                        } else {
                                            int mapped_edge = -1;
                                            for (size_t eii = 0; eii < ctx->edges.size(); ++eii) {
                                                if (ctx->edges[eii].rope_idx == rope_idx) { mapped_edge = static_cast<int>(eii); break; }
                                            }
                                            if (mapped_edge >= 0 && mapped_edge < static_cast<int>(ctx->edges.size()) && ctx->edges[mapped_edge].subgroup_flags != 0u) {
                                                float hue = subgroup_flags_to_hue(ctx, ctx->edges[mapped_edge].subgroup_flags);
                                                float hue_vals[1] = { hue };
                                                table_draw_rope_curve_blend_colored(out_rgba, w, h, pitch, seg.data(), use_count, jacket_px, jacket_border, hue_vals, 1, 3, 0.45f);
                                            } else {
                                                // subtle neutral tint so widget rope doesn't fully overlay the real rope
                                                table_draw_rope_curve_blend(out_rgba, w, h, pitch, seg.data(), use_count, jacket_px, jacket_border, 200, 160, 160, 120, 2);
                                            }
                                        }
                                        if (!no_lighting) {
                                            ContactLight la{}, lb{};
                                            if (resolve_edge_light_for_rope(rope_idx, &la, &lb)) {
                                                int glow_r = std::max(2, jacket_px * 2);
                                                draw_rope_light_falloff(out_rgba, w, h, pitch, seg.data(), use_count, la, lb, rope_decay, glow_r);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (ctx->rope_menu_open) {
        RopeMenuLayout layout = compute_rope_menu_layout(ctx);
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y, layout.w, layout.h, Color{32,32,40,230});
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y, layout.w, 1, Color{90,90,110,255});
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y + layout.h - 1, layout.w, 1, Color{10,10,14,255});
        auto blit_text = [&](const char* text, int tx, int ty, float scale, Color col) {
            if (!text || text[0] == '\0') return;
            auto bm = render_text_to_rgba(text, scale, {col.r, col.g, col.b, col.a});
            if (bm.pixels.empty()) return;
            for (int yy = 0; yy < bm.height; ++yy) {
                int dst_y = ty + yy;
                if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < bm.width; ++xx) {
                    int dst_x = tx + xx;
                    if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        };
        int title_x = layout.x + 8;
        int title_y = layout.y + 6;
        blit_text(LABEL_ROPE_MENU_TITLE, title_x, title_y, 0.95f, Color{220,220,230,255});
        bool simple_mode = (ctx->debug_flags & GP_CANVAS_DEBUG_NORENDER_MODE) == GP_CANVAS_DEBUG_NORENDER_MODE;
        for (int row = 0; row < 4; ++row) {
            int y0 = layout.item_start_y + row * layout.row_h;
            Color row_col{32,32,40,230};
            if ((row == 0 && simple_mode) || (row == 1 && !simple_mode)) {
                row_col = Color{46,46,58,255};
            }
            memset_rect(out_rgba, w, h, pitch, layout.x + 1, y0, layout.w - 2, layout.row_h, row_col);
            if (row == 0) {
                blit_text(LABEL_ROPE_MODE_SIMPLE, layout.x + 12, y0 + 3, 0.9f, Color{240,240,240,255});
            } else if (row == 1) {
                blit_text(LABEL_ROPE_MODE_FULL, layout.x + 12, y0 + 3, 0.9f, Color{240,240,240,255});
            } else {
                RopeMenuCounterLayout counter = compute_rope_menu_counter_layout(layout, row);
                const char* lbl = (row == 2) ? "Segments" : "Slack";
                blit_text(lbl, counter.label_x, y0 + 3, 0.85f, Color{200,200,210,255});
                memset_rect(out_rgba, w, h, pitch, counter.bx_minus, counter.by, counter.nbw, counter.h, Color{60,60,70,255});
                memset_rect(out_rgba, w, h, pitch, counter.bx_num, counter.by, counter.num_w, counter.h, Color{36,36,46,255});
                memset_rect(out_rgba, w, h, pitch, counter.bx_plus, counter.by, counter.nbw, counter.h, Color{60,60,70,255});
                blit_text("-", counter.bx_minus + (counter.nbw / 2) - 4, counter.by + 1, 0.9f, Color{220,220,220,255});
                blit_text("+", counter.bx_plus + (counter.nbw / 2) - 4, counter.by + 1, 0.9f, Color{220,220,220,255});
                char buf[32];
                if (row == 2) {
                    std::snprintf(buf, sizeof(buf), "%d", ctx->sim_segs);
                } else {
                    std::snprintf(buf, sizeof(buf), "%.2f", ctx->sim_slack);
                }
                blit_text(buf, counter.bx_num + 6, counter.by + 2, 0.85f, Color{230,230,235,255});
            }
        }
    }

    if (ctx->tool_menu_open) {
        ToolMenuLayout layout = compute_tool_menu_layout(ctx);
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y, layout.w, layout.h, Color{32,32,40,230});
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y, layout.w, 1, Color{90,90,110,255});
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y + layout.h - 1, layout.w, 1, Color{10,10,14,255});
        auto blit_text = [&](const char* text, int tx, int ty, float scale, Color col) {
            if (!text || text[0] == '\0') return;
            auto bm = render_text_to_rgba(text, scale, {col.r, col.g, col.b, col.a});
            if (bm.pixels.empty()) return;
            for (int yy = 0; yy < bm.height; ++yy) {
                int dst_y = ty + yy;
                if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < bm.width; ++xx) {
                    int dst_x = tx + xx;
                    if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        };
        int title_x = layout.x + 8;
        int title_y = layout.y + 6;
        blit_text(LABEL_MENU_TITLE, title_x, title_y, 0.95f, Color{220,220,230,255});
        int item_y = layout.item_start_y;
        for (size_t i = 0; i < std::size(kToolMenuItems); ++i) {
            int y0 = item_y + static_cast<int>(i) * layout.row_h;
            memset_rect(out_rgba, w, h, pitch, layout.x + 4, y0, layout.w - 8, layout.row_h - 2, Color{45,45,58,240});
            blit_text(kToolMenuItems[i].label, layout.x + 12, y0 + 3, 0.9f, Color{240,240,240,255});
        }
        if (layout.stack_count > 0) {
            int stack_title_y = layout.stack_start_y - layout.header_h + 2;
            blit_text(LABEL_MENU_STACK_TITLE, layout.x + 8, stack_title_y, 0.9f, Color{200,200,210,255});
            int stack_y = layout.stack_start_y;
            if (ctx->focused_module >= 0 && ctx->focused_module < static_cast<int>(ctx->module_io_rows.size())) {
                const auto &stack = ctx->module_io_rows[ctx->focused_module];
                for (size_t i = 0; i < stack.size(); ++i) {
                    int y0 = stack_y + static_cast<int>(i) * layout.row_h;
                    memset_rect(out_rgba, w, h, pitch, layout.x + 4, y0, layout.w - 8, layout.row_h - 2, Color{36,36,48,240});
                    char label[96];
                    label[0] = '\0';
                    const auto &row = stack[i];
                    if (row.kind == ModuleRowKind::Input) {
                        if (row.attachment_count > 1) {
                            std::snprintf(label, sizeof(label), "%s x%d", LABEL_IO_CONSUMER, row.attachment_count);
                        } else {
                            std::snprintf(label, sizeof(label), "%s", LABEL_IO_CONSUMER);
                        }
                    } else if (row.kind == ModuleRowKind::Output) {
                        if (row.attachment_count > 1) {
                            std::snprintf(label, sizeof(label), "%s x%d", LABEL_IO_PRODUCER, row.attachment_count);
                        } else {
                            std::snprintf(label, sizeof(label), "%s", LABEL_IO_PRODUCER);
                        }
                    } else {
                        std::snprintf(label, sizeof(label), "%s", tool_label(row.tool));
                    }
                    blit_text(label, layout.x + 12, y0 + 3, 0.85f, Color{210,210,220,255});
                }
            }
        }
        int table_title_y = layout.table_section_y + 2;
        blit_text(LABEL_MENU_TABLE_NUMBER, layout.x + 8, table_title_y, 0.9f, Color{200,200,210,255});
        ToolMenuCounterLayout counter = compute_tool_menu_counter_layout(layout);
        memset_rect(out_rgba, w, h, pitch, counter.bx_minus, counter.by, counter.nbw, counter.h, Color{50,50,56,255});
        memset_rect(out_rgba, w, h, pitch, counter.bx_num, counter.by, counter.num_w, counter.h, Color{36,36,42,255});
        memset_rect(out_rgba, w, h, pitch, counter.bx_plus, counter.by, counter.nbw, counter.h, Color{50,50,56,255});
        auto minus = render_text_to_rgba(LABEL_IO_MINUS, 1.0f, {230,230,235,255});
        if (!minus.pixels.empty()) {
            int tx = counter.bx_minus + (counter.nbw - minus.width) / 2;
            int ty = counter.by + (counter.h - minus.height) / 2;
            for (int yy = 0; yy < minus.height; ++yy) {
                int dst_y = ty + yy;
                if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < minus.width; ++xx) {
                    int dst_x = tx + xx;
                    if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &minus.pixels[(yy * minus.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
        auto plus = render_text_to_rgba(LABEL_IO_PLUS, 1.0f, {230,230,235,255});
        if (!plus.pixels.empty()) {
            int tx = counter.bx_plus + (counter.nbw - plus.width) / 2;
            int ty = counter.by + (counter.h - plus.height) / 2;
            for (int yy = 0; yy < plus.height; ++yy) {
                int dst_y = ty + yy;
                if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < plus.width; ++xx) {
                    int dst_x = tx + xx;
                    if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &plus.pixels[(yy * plus.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
        std::string s = std::to_string(std::max(0, ctx->table_tool_number));
        auto bm = render_text_to_rgba(s, 1.0f, {255,255,255,255});
        if (!bm.pixels.empty()) {
            int tx = counter.bx_num + (counter.num_w - bm.width) / 2;
            int ty = counter.by + (counter.h - bm.height) / 2;
            for (int yy = 0; yy < bm.height; ++yy) {
                int dst_y = ty + yy;
                if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < bm.width; ++xx) {
                    int dst_x = tx + xx;
                    if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
        auto lb = render_text_to_rgba(LABEL_MENU_TABLE_NUMBER_SHORT, 0.75f, {200,200,200,255});
        if (!lb.pixels.empty()) {
            int tx = counter.bx_num + (counter.num_w - lb.width) / 2;
            int ty = counter.by - lb.height - 2;
            for (int yy = 0; yy < lb.height; ++yy) {
                int dst_y = ty + yy;
                if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < lb.width; ++xx) {
                    int dst_x = tx + xx;
                    if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &lb.pixels[(yy * lb.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
    }

    if (ctx->module_menu_open) {
        ModuleMenuLayout layout = compute_module_menu_layout(ctx, ctx->module_menu_module_idx, static_cast<int>(ctx->module_library_labels.size()));
        if (layout.w <= 0 || layout.h <= 0) return 1;
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y, layout.w, layout.h, Color{30,36,32,235});
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y, layout.w, 1, Color{70,90,80,255});
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y + layout.h - 1, layout.w, 1, Color{12,18,12,255});
        auto blit_text = [&](const char* text, int tx, int ty, float scale, Color col) {
            if (!text || text[0] == '\0') return;
            auto bm = render_text_to_rgba(text, scale, {col.r, col.g, col.b, col.a});
            if (bm.pixels.empty()) return;
            for (int yy = 0; yy < bm.height; ++yy) {
                int dst_y = ty + yy;
                if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < bm.width; ++xx) {
                    int dst_x = tx + xx;
                    if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        };
        int title_x = layout.x + 8;
        int title_y = layout.y + 6;
        blit_text(LABEL_MODULE_MENU_TITLE, title_x, title_y, 0.95f, Color{220,235,220,255});
        int item_y = layout.item_start_y;
        if (ctx->module_library_labels.empty()) {
            int y0 = item_y;
            memset_rect(out_rgba, w, h, pitch, layout.x + 4, y0, layout.w - 8, layout.row_h - 2, Color{34,42,36,235});
            blit_text(LABEL_MODULE_MENU_EMPTY, layout.x + 12, y0 + 3, 0.9f, Color{170,190,175,255});
        } else {
            for (size_t i = 0; i < ctx->module_library_labels.size(); ++i) {
                int y0 = item_y + static_cast<int>(i) * layout.row_h;
                memset_rect(out_rgba, w, h, pitch, layout.x + 4, y0, layout.w - 8, layout.row_h - 2, Color{40,50,42,240});
                const std::string &lbl = ctx->module_library_labels[static_cast<size_t>(i)];
                blit_text(lbl.c_str(), layout.x + 12, y0 + 3, 0.9f, Color{235,245,235,255});
            }
        }
    }

    if (ctx->plugin_menu_open) {
        PluginMenuLayout layout = compute_plugin_menu_layout(ctx, ctx->plugin_menu_module_idx, static_cast<int>(ctx->plugin_tool_ids.size()));
        if (layout.w <= 0 || layout.h <= 0) return 1;
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y, layout.w, layout.h, Color{28,32,44,235});
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y, layout.w, 1, Color{70,80,110,255});
        memset_rect(out_rgba, w, h, pitch, layout.x, layout.y + layout.h - 1, layout.w, 1, Color{12,12,18,255});
        auto blit_text = [&](const char* text, int tx, int ty, float scale, Color col) {
            if (!text || text[0] == '\0') return;
            auto bm = render_text_to_rgba(text, scale, {col.r, col.g, col.b, col.a});
            if (bm.pixels.empty()) return;
            for (int yy = 0; yy < bm.height; ++yy) {
                int dst_y = ty + yy;
                if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < bm.width; ++xx) {
                    int dst_x = tx + xx;
                    if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &bm.pixels[(yy * bm.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        };
        int title_x = layout.x + 8;
        int title_y = layout.y + 6;
        blit_text(LABEL_PLUGIN_MENU_TITLE, title_x, title_y, 0.95f, Color{220,230,245,255});
        int item_y = layout.item_start_y;
        if (ctx->plugin_tool_ids.empty()) {
            int y0 = item_y;
            memset_rect(out_rgba, w, h, pitch, layout.x + 4, y0, layout.w - 8, layout.row_h - 2, Color{34,38,52,235});
            blit_text(LABEL_PLUGIN_MENU_EMPTY, layout.x + 12, y0 + 3, 0.9f, Color{170,180,200,255});
        } else {
            for (size_t i = 0; i < ctx->plugin_tool_ids.size(); ++i) {
                int y0 = item_y + static_cast<int>(i) * layout.row_h;
                memset_rect(out_rgba, w, h, pitch, layout.x + 4, y0, layout.w - 8, layout.row_h - 2, Color{42,52,70,240});
                const std::string &lbl = ctx->plugin_tool_labels[static_cast<size_t>(i)];
                blit_text(lbl.c_str(), layout.x + 12, y0 + 3, 0.9f, Color{240,245,255,255});
            }
        }
    }

    return 1;
}
