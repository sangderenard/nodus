// Serialization format (simple binary):
// [8 bytes magic 'GPTBL001'][uint32_t version]
// version 1: GP_TableStyle, cols, rows, edges, selected keys
// version 2: same as v1, then int32 meta_group_count, followed by per-meta-group blob
// version 4: adds RopeSim blob (int32 size + bytes) before meta-groups
// Per-meta-group blob: int32 vertex_count, (uint64 rope_uid,int32 vert_idx)*N, float confinement, int32 sim_group_idx,
// uint64 id, LassoConfig (8 bytes), uint64 anchor_rope_uid, int32 anchor_vert, uint32 subgroup_flags, int32 channel_group,
// int32 dangling_widget_rope, int32 dangling_widget_rope_vid, float dangling_hang_len, int32 ring_mode,
// uint64 overlay_key_a, uint64 overlay_key_b
int32_t gp_table_serialize(GP_TableContext* ctx, char* out_buf, int32_t out_len) {
    if (!ctx) return 0;
    const uint32_t version = 4;
    const char magic[8] = {'G','P','T','B','L','0','0','1'};
    // Build a canonical UUID atlas covering all UUIDs referenced by this
    // table blob (edges, ropes, overlays, module/frame ports, rings,
    // meta-group ids). The atlas will be written immediately after the
    // version so loaders can read it first and reserve/register any
    // required identities before reconstructing dependent structures.
    std::unordered_set<uint64_t> _atlas_set;
    if (ctx->module_uuid != 0ull) _atlas_set.insert(ctx->module_uuid);
    for (const auto &fpe : ctx->frame_port_uuids) if (fpe.uuid != 0ull) _atlas_set.insert(fpe.uuid);
    for (auto ru : ctx->rope_ids) if (ru != 0ull) _atlas_set.insert(ru);
    for (const auto &e : ctx->edges) { if (e.first != 0ull) _atlas_set.insert(e.first); if (e.second != 0ull) _atlas_set.insert(e.second); }
    for (const auto &re : ctx->rings) if (re.key != 0ull) _atlas_set.insert(re.key);
    for (size_t mi = 0; mi < ctx->meta_groups.size(); ++mi) {
        GP_MetaGroup* mg = ctx->meta_groups[mi].get();
        if (!mg) continue;
        if (mg->overlay_key_a != 0ull) _atlas_set.insert(mg->overlay_key_a);
        if (mg->overlay_key_b != 0ull) _atlas_set.insert(mg->overlay_key_b);
        if (mg->id != 0ull) _atlas_set.insert(mg->id);
        if (mg->anchor_rope >= 0 && static_cast<size_t>(mg->anchor_rope) < ctx->rope_ids.size()) _atlas_set.insert(ctx->rope_ids[static_cast<size_t>(mg->anchor_rope)]);
    }
    std::vector<uint64_t> atlas;
    atlas.reserve(_atlas_set.size());
    for (auto v : _atlas_set) atlas.push_back(v);
    int32_t col_count = static_cast<int32_t>(ctx->cols.size());
    int32_t row_count = static_cast<int32_t>(ctx->rows.size());
    int32_t edge_count = static_cast<int32_t>(ctx->edges.size());
    int32_t rope_id_count = static_cast<int32_t>(ctx->rope_ids.size());
    int32_t sel_count = static_cast<int32_t>(ctx->selected_leds.size());
    int32_t rope_blob_len = 0;
    if (ctx->rope_sim) {
        rope_blob_len = rope_sim_serialized_size(ctx->rope_sim);
    }
    int32_t need = 0;
    need += 8; // magic
    need += 4; // version
    // atlas count + entries (written immediately after version)
    need += 4; // atlas_count
    need += static_cast<int32_t>(atlas.size()) * static_cast<int32_t>(sizeof(uint64_t));
    need += static_cast<int32_t>(sizeof(GP_TableStyle));
    need += 4; // col_count
    need += col_count * static_cast<int32_t>(sizeof(GP_TableColumn));
    need += 4; // row_count
    need += row_count * static_cast<int32_t>(sizeof(GP_TableRow));
    need += 4; // edge_count
    need += edge_count * static_cast<int32_t>(sizeof(uint64_t) * 2);
    // per-rope persistent uids (one per rope created)
    need += 4; // rope_uid_count
    need += rope_id_count * static_cast<int32_t>(sizeof(uint64_t));
    need += 4; // sel_count
    need += sel_count * static_cast<int32_t>(sizeof(uint64_t));
    // module uuid + frame-port entries (version 3)
    need += 8; // module_uuid
    // frame port count + entries
    need += 4; // frame_port_count
    // each entry: row(int32), idx(int32), uuid(uint64)
    need += static_cast<int32_t>(ctx->frame_port_uuids.size()) * (4 + 4 + 8);
    // RopeSim blob (version 4)
    need += 4; // rope_blob_len
    need += rope_blob_len;
    // meta-groups (version 2)
    int32_t mg_count = static_cast<int32_t>(ctx->meta_groups.size());
    need += 4; // mg_count
    for (int i = 0; i < mg_count; ++i) {
        GP_MetaGroup* mg = ctx->meta_groups[static_cast<size_t>(i)].get();
        int32_t vcount = static_cast<int32_t>(mg ? mg->vertices.size() : 0);
        need += 4; // vcount
        need += vcount * (8 + 4); // rope_uid,uint64 + vertex_idx,int32
        need += 4; // confinement (stored as float)
        need += 4; // sim_group_idx
        need += 8; // id (uint64)
        need += static_cast<int32_t>(sizeof(LassoConfig)); // lasso_config
        need += 8; // anchor_rope_uid (uint64)
        need += 4; // anchor_vert
        need += 4; // subgroup_flags (uint32)
        need += 4; // channel_group
        need += 4; // dangling_widget_rope
        need += 4; // dangling_widget_rope_vid
        need += 4; // dangling_hang_len (float)
        need += 4; // ring_mode
        need += 4; // ring_u (float)
        need += 8; // overlay_key_a
        need += 8; // overlay_key_b
        need += 8; // overlay port_uuid_a
        need += 8; // overlay port_uuid_b
    }

    // Log meta-group count and total size needed when serializing (helps
    // troubleshooting when saved modules appear to lack meta-groups).
    printf("gp_table_serialize: ctx=%p mg_count=%d need=%d\n", (void*)ctx, mg_count, need);
    if (!ctx->rope_ids.empty()) {
        printf("gp_table_serialize: rope_ids:");
        for (size_t ri = 0; ri < ctx->rope_ids.size(); ++ri) printf(" %llu", (unsigned long long)ctx->rope_ids[ri]);
        printf("\n");
    } else {
        printf("gp_table_serialize: rope_ids: <empty>\n");
    }

    if (!out_buf) return need;
    if (out_len < need) return 0;

    char* p = out_buf;
    // write magic
    memcpy(p, magic, 8); p += 8;
    // write version
    memcpy(p, &version, 4); p += 4;
    // write atlas: count + entries
    int32_t atlas_count = static_cast<int32_t>(atlas.size());
    memcpy(p, &atlas_count, 4); p += 4;
    for (int ai = 0; ai < atlas_count; ++ai) { uint64_t u = atlas[static_cast<size_t>(ai)]; memcpy(p, &u, sizeof(uint64_t)); p += sizeof(uint64_t); }
    // write style_raw
    memcpy(p, &ctx->style_raw, sizeof(GP_TableStyle)); p += sizeof(GP_TableStyle);
    // write cols
    memcpy(p, &col_count, 4); p += 4;
    if (col_count > 0) {
        memcpy(p, ctx->cols.data(), static_cast<size_t>(col_count) * sizeof(GP_TableColumn));
        p += col_count * static_cast<int32_t>(sizeof(GP_TableColumn));
    }
    // write rows (we zero any waveform pointers)
    memcpy(p, &row_count, 4); p += 4;
    for (int i = 0; i < row_count; ++i) {
        GP_TableRow row = ctx->rows[static_cast<size_t>(i)];
        // clear wave pointers inside cells to avoid serializing pointers
        for (int c = 0; c < row.cell_count && c < 8; ++c) {
            row.cells[c].wave = nullptr;
            row.cells[c].image = nullptr;
        }
        memcpy(p, &row, sizeof(GP_TableRow)); p += sizeof(GP_TableRow);
    }
    // write edges
    memcpy(p, &edge_count, 4); p += 4;
    for (int i = 0; i < edge_count; ++i) {
        uint64_t a = ctx->edges[static_cast<size_t>(i)].first;
        uint64_t b = ctx->edges[static_cast<size_t>(i)].second;
        memcpy(p, &a, sizeof(uint64_t)); p += sizeof(uint64_t);
        memcpy(p, &b, sizeof(uint64_t)); p += sizeof(uint64_t);
    }
    // write per-rope ids (one per edge/rope)
    memcpy(p, &rope_id_count, 4); p += 4;
    for (int i = 0; i < rope_id_count; ++i) {
        uint64_t ru = ctx->rope_ids[static_cast<size_t>(i)];
        memcpy(p, &ru, sizeof(uint64_t)); p += sizeof(uint64_t);
    }
    // write selected keys
    memcpy(p, &sel_count, 4); p += 4;
    for (auto k : ctx->selected_leds) {
        uint64_t key = k;
        memcpy(p, &key, sizeof(uint64_t)); p += sizeof(uint64_t);
    }

    // write module uuid and frame-port entries (version 3)
    uint64_t module_uuid = ctx->module_uuid;
    memcpy(p, &module_uuid, 8); p += 8;
    int32_t fpcount = static_cast<int32_t>(ctx->frame_port_uuids.size());
    memcpy(p, &fpcount, 4); p += 4;
    for (int i = 0; i < fpcount; ++i) {
        int32_t r = ctx->frame_port_uuids[static_cast<size_t>(i)].row;
        int32_t idx = ctx->frame_port_uuids[static_cast<size_t>(i)].idx;
        uint64_t pu = ctx->frame_port_uuids[static_cast<size_t>(i)].uuid;
        memcpy(p, &r, 4); p += 4;
        memcpy(p, &idx, 4); p += 4;
        memcpy(p, &pu, 8); p += 8;
    }

    // write RopeSim blob (version 4)
    memcpy(p, &rope_blob_len, 4); p += 4;
    if (rope_blob_len > 0) {
        int wrote = rope_sim_serialize(ctx->rope_sim, p, rope_blob_len);
        if (wrote != rope_blob_len) return 0;
        p += rope_blob_len;
    }

    // write meta-groups (version 3)
    memcpy(p, &mg_count, 4); p += 4;
    for (int i = 0; i < mg_count; ++i) {
        GP_MetaGroup* mg = ctx->meta_groups[static_cast<size_t>(i)].get();
        if (mg) {
            LassoConfig lc = mg->lasso_config;
            printf("gp_table_serialize: mg id=%llu vcount=%d sim_idx=%d lasso.spring_min_rest=%.2f spring_reduce_rate=%.2f spring_mode=%u\n",
                   (unsigned long long)mg->id, (int)mg->vertices.size(), mg->sim_group_idx, lc.spring_min_rest, lc.spring_reduce_rate, (unsigned)lc.spring_mode);
            // Dump full meta-group for diagnostics
            gp_table_debug_dump_meta_group(ctx, mg, "serialize");
        } else {
            printf("gp_table_serialize: mg <null>\n");
        }
        int32_t vcount = static_cast<int32_t>(mg ? mg->vertices.size() : 0);
        memcpy(p, &vcount, 4); p += 4;
        for (int vi = 0; vi < vcount; ++vi) {
            const auto &mv = mg->vertices[static_cast<size_t>(vi)];
            uint64_t rope_id = mv.rope_id;
            int32_t vert_idx = mv.vertex_idx;
            memcpy(p, &rope_id, 8); p += 8;
            memcpy(p, &vert_idx, 4); p += 4;
        }
        float conf = mg ? mg->confinement : 1.0f;
        memcpy(p, &conf, 4); p += 4;
        int32_t sim_idx = mg ? mg->sim_group_idx : -1;
        memcpy(p, &sim_idx, 4); p += 4;
        uint64_t mgid = mg ? mg->id : 0ull;
        memcpy(p, &mgid, 8); p += 8;
        LassoConfig lc{};
        if (mg) lc = mg->lasso_config;
        memcpy(p, &lc, sizeof(LassoConfig)); p += sizeof(LassoConfig);
        // write anchor as a persistent rope id (0 == none)
        uint64_t anchor_rope_id = 0ull;
        int32_t anchor_v = -1;
        if (mg) {
            anchor_v = mg->anchor_vert;
            if (mg->anchor_rope >= 0 && static_cast<size_t>(mg->anchor_rope) < ctx->rope_ids.size()) anchor_rope_id = ctx->rope_ids[static_cast<size_t>(mg->anchor_rope)];
        }
        memcpy(p, &anchor_rope_id, 8); p += 8;
        memcpy(p, &anchor_v, 4); p += 4;
        uint32_t sgflags = mg ? mg->subgroup_flags : 0u;
        memcpy(p, &sgflags, 4); p += 4;
        int32_t channel_group = mg ? mg->channel_group : 0;
        memcpy(p, &channel_group, 4); p += 4;
        int32_t dang_rope = mg ? mg->dangling_widget_rope : -1;
        int32_t dang_vid = mg ? mg->dangling_widget_rope_vid : -1;
        memcpy(p, &dang_rope, 4); p += 4;
        memcpy(p, &dang_vid, 4); p += 4;
        float dang_len = mg ? mg->dangling_hang_len : 0.0f;
        memcpy(p, &dang_len, 4); p += 4;
        int32_t ring_mode = mg ? mg->ring_mode : 0;
        memcpy(p, &ring_mode, 4); p += 4;
        float ring_u = 0.0f;
        if (mg) {
            // attempt to find a ring registered for this meta-group by id
            for (size_t ri = 0; ri < ctx->rings.size(); ++ri) {
                const auto &re = ctx->rings[ri];
                if (re.key == mg->id) {
                    int ring_id = re.ring_id;
                    gp_table_get_ring_u(ctx, ring_id, &ring_u);
                    break;
                }
            }
        }
        memcpy(p, &ring_u, 4); p += 4;
        uint64_t oka = mg ? mg->overlay_key_a : 0ull;
        uint64_t okb = mg ? mg->overlay_key_b : 0ull;
        memcpy(p, &oka, 8); p += 8;
        memcpy(p, &okb, 8); p += 8;
        // persist overlay per-port UUIDs so restore can re-establish exact port identities
        uint64_t pu_a = 0ull, pu_b = 0ull;
        GP_CanvasContext* cvs = gp_canvas_get_singleton();
        if (cvs && (oka || okb)) {
            gp_canvas_get_overlay_port_uuids(cvs, oka, okb, &pu_a, &pu_b);
        }
        memcpy(p, &pu_a, 8); p += 8;
        memcpy(p, &pu_b, 8); p += 8;
    }

    return need;
}

int32_t gp_table_deserialize(GP_TableContext* ctx, const char* in_buf, int32_t in_len) {
    if (!ctx || !in_buf || in_len <= 0) return 0;
    const char expect_magic[8] = {'G','P','T','B','L','0','0','1'};
    if (in_len < 8 + 4 + static_cast<int>(sizeof(GP_TableStyle))) return 0;
    const char* p = in_buf;
    if (memcmp(p, expect_magic, 8) != 0) return 0;
    p += 8;
    uint32_t version = 0;
    memcpy(&version, p, 4); p += 4;
    // Read UUID atlas (written immediately after version).
    std::vector<uint64_t> atlas;
    if (p + 4 > in_buf + in_len) return 0;
    int32_t atlas_count = 0;
    memcpy(&atlas_count, p, 4); p += 4;
    if (atlas_count < 0) return 0;
    atlas.reserve(static_cast<size_t>(atlas_count));
    for (int i = 0; i < atlas_count; ++i) {
        if (p + 8 > in_buf + in_len) return 0;
        uint64_t u = 0ull; memcpy(&u, p, 8); p += 8;
        atlas.push_back(u);
    }
    printf("gp_table_deserialize: read UUID atlas count=%d\n", atlas_count);
    printf("gp_table_deserialize: canvas singleton=%p\n", (void*)gp_canvas_get_singleton());
    printf("gp_table_deserialize: read UUID atlas count=%d\n", atlas_count);
    // trace: report deserialize invocation
    printf("gp_table_deserialize: ctx=%p len=%d version=%u\n", (void*)ctx, in_len, (unsigned)version);
    // Canvas singleton (may be null). Declare early so deserialization can
    // register persisted rope/module UUIDs with the canvas if available.
    GP_CanvasContext* cvs = gp_canvas_get_singleton();
    const uint32_t expected_version = 4u;
    if (version != expected_version) {
        // No backward-compatibility: if save version mismatches current
        // format, abandon any attempt to negotiate and wipe the context.
        printf("gp_table_deserialize: version mismatch (have=%u want=%u) - wiping ctx=%p\n", (unsigned)version, (unsigned)expected_version, (void*)ctx);
        if (ctx) {
            // Clear visible data and reset state to a clean default.
            ctx->rows.clear();
            ctx->cols.clear();
            ctx->selected_leds.clear();
            ctx->led_glow_strength.clear();
            ctx->led_flow_glow_strength.clear();
            ctx->frame_port_uuids.clear();
            ctx->module_uuid = 0ull;
            ctx->meta_groups.clear();
            ctx->pending_meta_vertices.clear();
            ctx->pending_ops.clear();
            ctx->actions.clear();
            ctx->stage_ports.clear();
            ctx->key_type_hint.clear();
            ctx->key_is_input.clear();
            ctx->key_is_output.clear();
            ctx->rings.clear();
            ctx->ring_subscriber_slots.clear();
            ctx->ring_subscriber_slots.shrink_to_fit();
            // Clear edges and associated rope state (this will destroy owned RopeSim)
            gp_table_clear_edges(ctx);
            // Reset style to defaults and recompute geometry
            ctx->style_raw = GP_TableStyle();
            ctx->st = load_style(&ctx->style_raw);
            recompute_geom(ctx);
        }
        return 1; // indicate success (wiped/ignored)
    }
    // read style
    GP_TableStyle style{};
    memcpy(&style, p, sizeof(GP_TableStyle)); p += sizeof(GP_TableStyle);
    // columns
    int32_t col_count = 0;
    memcpy(&col_count, p, 4); p += 4;
    if (col_count < 0 || col_count > 8) return 0;
    std::vector<GP_TableColumn> cols;
    if (col_count > 0) {
        cols.resize(static_cast<size_t>(col_count));
        memcpy(cols.data(), p, static_cast<size_t>(col_count) * sizeof(GP_TableColumn));
        p += col_count * static_cast<int>(sizeof(GP_TableColumn));
    }
    // rows
    int32_t row_count = 0;
    memcpy(&row_count, p, 4); p += 4;
    if (row_count < 0) return 0;
    std::vector<GP_TableRow> rows;
    if (row_count > 0) {
        rows.resize(static_cast<size_t>(row_count));
        for (int i = 0; i < row_count; ++i) {
            memcpy(&rows[static_cast<size_t>(i)], p, sizeof(GP_TableRow)); p += sizeof(GP_TableRow);
            // ensure wave pointers are null for safety
            for (int c = 0; c < rows[static_cast<size_t>(i)].cell_count && c < 8; ++c) rows[static_cast<size_t>(i)].cells[c].wave = nullptr;
        }
    }
    // edges
    int32_t edge_count = 0;
    memcpy(&edge_count, p, 4); p += 4;
    if (edge_count < 0) return 0;
    std::vector<std::pair<uint64_t,uint64_t>> edges;
    edges.reserve(static_cast<size_t>(edge_count));
    for (int i = 0; i < edge_count; ++i) {
        uint64_t a=0,b=0;
        memcpy(&a, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        memcpy(&b, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        edges.emplace_back(a,b);
    }
    // read per-rope uids
    int32_t rope_uid_count = 0;
    if (p + 4 > in_buf + in_len) return 0;
    memcpy(&rope_uid_count, p, 4); p += 4;
    std::vector<uint64_t> rope_ids_temp;
    rope_ids_temp.reserve(static_cast<size_t>(std::max(0, rope_uid_count)));
    for (int i = 0; i < rope_uid_count; ++i) {
        uint64_t ru = 0ull;
        if (p + 8 > in_buf + in_len) return 0;
        memcpy(&ru, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        rope_ids_temp.push_back(ru);
    }
    printf("gp_table_deserialize: rope_uid_count=%d\n", rope_uid_count);
    if (!rope_ids_temp.empty()) {
        printf("gp_table_deserialize: rope_ids:");
        for (size_t ri = 0; ri < rope_ids_temp.size(); ++ri) printf(" %llu", (unsigned long long)rope_ids_temp[ri]);
        printf("\n");
    }
    // selected keys
    int32_t sel_count = 0;
    memcpy(&sel_count, p, 4); p += 4;
    if (sel_count < 0) return 0;
    std::unordered_set<uint64_t> selset;
    for (int i = 0; i < sel_count; ++i) {
        uint64_t key = 0;
        memcpy(&key, p, sizeof(uint64_t)); p += sizeof(uint64_t);
        selset.insert(key);
    }

    // version 3: read module UUID and frame-port entries
    uint64_t module_uuid = 0ull;
    RopeSim* restored_sim = nullptr;
    if (version >= 3) {
        if (p + 8 > in_buf + in_len) return 0;
        memcpy(&module_uuid, p, 8); p += 8;
        int32_t fpcount = 0;
        if (p + 4 > in_buf + in_len) return 0;
        memcpy(&fpcount, p, 4); p += 4;
        if (fpcount < 0) return 0;
        std::vector<std::tuple<int32_t,int32_t,uint64_t>> fp_entries;
        fp_entries.reserve(static_cast<size_t>(fpcount));
        for (int i = 0; i < fpcount; ++i) {
            if (p + 4 + 4 + 8 > in_buf + in_len) return 0;
            int32_t r=0, idx=0; uint64_t pu=0ull;
            memcpy(&r, p, 4); p += 4;
            memcpy(&idx, p, 4); p += 4;
            memcpy(&pu, p, 8); p += 8;
            fp_entries.emplace_back(r, idx, pu);
        }
        // attach module_uuid and frame-port entries to ctx so callers (canvas)
        // can read them when needed. We store into ctx fields for later use.
        ctx->module_uuid = module_uuid;
        ctx->frame_port_uuids.clear();
        for (auto &ent : fp_entries) {
            int32_t r = std::get<0>(ent);
            int32_t idx = std::get<1>(ent);
            uint64_t pu = std::get<2>(ent);
            GP_TableContext::FramePortEntry fpe{}; fpe.row = r; fpe.idx = idx; fpe.uuid = pu;
            ctx->frame_port_uuids.push_back(fpe);
        }
    }

    // version 4: read RopeSim blob
    if (version >= 4) {
        if (p + 4 > in_buf + in_len) return 0;
        int32_t rope_blob_len = 0;
        memcpy(&rope_blob_len, p, 4); p += 4;
        if (rope_blob_len < 0 || p + rope_blob_len > in_buf + in_len) return 0;
        if (rope_blob_len > 0) {
            restored_sim = rope_sim_deserialize(p, rope_blob_len);
            if (!restored_sim) {
                printf("gp_table_deserialize: failed to deserialize RopeSim blob\n");
            }
            p += rope_blob_len;
        }
    }

    // Commit to ctx: replace style, cols, rows, edges, selected set
    ctx->style_raw = style;
    ctx->st = load_style(&ctx->style_raw);
    ctx->cols = std::move(cols);
    ctx->rows = std::move(rows);
    // clear existing edges via API to keep rope_sim indices consistent
    gp_table_clear_edges(ctx);
    auto add_edge_without_rope = [&](uint64_t a, uint64_t b) {
        ctx->edges.emplace_back(a, b);
        uint64_t uid = gp_canvas_generate_id(gp_canvas_get_singleton(), 0ull);
        ctx->edge_ids.push_back(uid);
        ctx->relax_value.push_back(0.0f);
        ctx->relax_vel.push_back(0.0f);
        ctx->prospective_initialized = false;
        ensure_edge_fifos(ctx);
        sync_edge_tensor_for_idx(ctx, ctx->edges.size() - 1);
    };
    bool suppress_edge_rope_creation = (restored_sim != nullptr) || !rope_ids_temp.empty();
    for (auto &e : edges) {
        if (suppress_edge_rope_creation) add_edge_without_rope(e.first, e.second);
        else gp_table_add_edge(ctx, e.first, e.second);
    }
    // if serialized per-rope uids were present, adopt them so runtime rope
    // indices map to the saved stable ids (this preserves stable mapping
    // for meta-group vertices which reference ropes by uid).
    if (!rope_ids_temp.empty()) {
        if (rope_ids_temp.size() < ctx->edges.size()) {
            rope_ids_temp.resize(ctx->edges.size(), 0ull);
        }
        ctx->rope_ids = rope_ids_temp;
        // rebuild rope_id_to_sim_idx so canvas lookups resolve to the current
        // rope indices (edge order) using the persisted ids.
        ctx->rope_id_to_sim_idx.clear();
        for (size_t i = 0; i < ctx->rope_ids.size(); ++i) {
            uint64_t uid = ctx->rope_ids[i];
            if (uid != 0ull) ctx->rope_id_to_sim_idx[uid] = static_cast<int>(i);
        }
        // set next_rope_id to one past the max saved id
        uint64_t mx = 1;
        for (auto ru : rope_ids_temp) if (ru >= mx) mx = ru + 1;
        ctx->next_rope_id = mx;
        // inform canvas of this table's persisted rope IDs so the canvas
        // can build a canonical mapping for deterministic restoration.
        if (cvs) {
            gp_canvas_register_table_rope_ids_from_array(cvs, ctx, ctx->rope_ids.data(), static_cast<int>(ctx->rope_ids.size()));
        }
    }
    ctx->selected_leds = std::move(selset);
    recompute_geom(ctx);
    const bool restored_sim_attached = (restored_sim != nullptr);
    if (restored_sim) {
        if (ctx->rope_sim && ctx->rope_sim_owned) {
            rope_sim_destroy(ctx->rope_sim);
        }
        ctx->rope_sim = restored_sim;
        ctx->rope_sim_owned = 1;
    }
    // Parse and restore meta-groups (version 2)
    // ensure there's enough data remaining
    if (p + 4 > in_buf + in_len) return 1; // nothing more
    int32_t mg_count = 0;
    memcpy(&mg_count, p, 4); p += 4;
    printf("gp_table_deserialize: meta-group count=%d\n", mg_count);
    if (mg_count < 0) return 1;
    struct MGData { std::vector<std::pair<uint64_t,int>> verts; float confinement; int32_t sim_idx; uint64_t id; LassoConfig lc; uint64_t anchor_uid; int32_t anchor_v; uint32_t subgroup_flags; int32_t channel_group; int32_t dang_rope; int32_t dang_vid; float dang_len; int32_t ring_mode; float ring_u; uint64_t oka; uint64_t okb; uint64_t port_a; uint64_t port_b; };
    std::vector<MGData> mgds;
    mgds.reserve(static_cast<size_t>(mg_count));
    for (int m = 0; m < mg_count; ++m) {
        if (p + 4 > in_buf + in_len) return 1;
        int32_t vcount = 0;
        memcpy(&vcount, p, 4); p += 4;
        MGData d; d.verts.reserve(static_cast<size_t>(std::max(0, vcount)));
        for (int vi = 0; vi < vcount; ++vi) {
            uint64_t ru = 0ull; int32_t vid = 0;
            if (p + 8 > in_buf + in_len) return 1;
            memcpy(&ru, p, 8); p += 8;
            memcpy(&vid, p, 4); p += 4;
            d.verts.emplace_back(ru, vid);
        }
        memcpy(&d.confinement, p, 4); p += 4;
        memcpy(&d.sim_idx, p, 4); p += 4;
        memcpy(&d.id, p, 8); p += 8;
        memcpy(&d.lc, p, sizeof(LassoConfig)); p += sizeof(LassoConfig);
        uint64_t anchor_uid = 0ull;
        memcpy(&anchor_uid, p, 8); p += 8;
        memcpy(&d.anchor_v, p, 4); p += 4;
        d.anchor_uid = anchor_uid;
        memcpy(&d.subgroup_flags, p, 4); p += 4;
        memcpy(&d.channel_group, p, 4); p += 4;
        memcpy(&d.dang_rope, p, 4); p += 4;
        memcpy(&d.dang_vid, p, 4); p += 4;
        memcpy(&d.dang_len, p, 4); p += 4;
        memcpy(&d.ring_mode, p, 4); p += 4;
        memcpy(&d.ring_u, p, 4); p += 4;
        memcpy(&d.oka, p, 8); p += 8;
        memcpy(&d.okb, p, 8); p += 8;
        // read persisted per-port UUIDs (may be zero)
        memcpy(&d.port_a, p, 8); p += 8;
        memcpy(&d.port_b, p, 8); p += 8;
        mgds.push_back(std::move(d));
    }

    // Recreate meta-groups in the context
    for (auto &d : mgds) {
        GP_MetaGroup* mg = gp_table_meta_create(ctx);
        if (!mg) continue;
        printf("gp_table_deserialize: creating meta-group ctx=%p mg=%p id=%llu oka=%llu okb=%llu verts=%zu sim_idx=%d\n",
               (void*)ctx, (void*)mg, static_cast<unsigned long long>(d.id), static_cast<unsigned long long>(d.oka), static_cast<unsigned long long>(d.okb), d.verts.size(), d.sim_idx);
        // restore simple fields
        mg->confinement = d.confinement;
        // If we restored a RopeSim blob, preserve its meta-group index so we
        // can reuse the saved meta-group state without duplicating members.
        // Otherwise, leave sim_group_idx unset so we create a fresh sim group.
        mg->sim_group_idx = restored_sim_attached ? d.sim_idx : -1;
        mg->id = d.id;
        mg->lasso_config = d.lc;
        // resolve persisted anchor UID to runtime rope index (if present)
        mg->anchor_rope = -1;
        if (d.anchor_uid != 0ull) {
            int resolved = -1;
            if (cvs) resolved = gp_canvas_resolve_rope_id_to_index(cvs, ctx, d.anchor_uid);
            if (resolved < 0) {
                for (size_t ri = 0; ri < ctx->rope_ids.size(); ++ri) {
                    if (ctx->rope_ids[ri] == d.anchor_uid) { resolved = static_cast<int>(ri); break; }
                }
            }
            if (resolved >= 0) {
                mg->anchor_rope = resolved;
            } else {
                printf("gp_table_deserialize: ERROR - could not resolve anchor rope id=%llu\n", (unsigned long long)d.anchor_uid);
                return 2; // hard-fail: unresolved persisted anchor UID
            }
        }
        mg->anchor_vert = d.anchor_v;
        mg->subgroup_flags = d.subgroup_flags;
        mg->channel_group = d.channel_group;
        mg->dangling_widget_rope = d.dang_rope;
        mg->dangling_widget_rope_vid = d.dang_vid;
        mg->dangling_hang_len = d.dang_len;
        mg->ring_mode = d.ring_mode;
        // Register canonical overlay early so ring/T-off creation can attach
        // to the persisted overlay keys instead of creating new overlays.
        if (cvs && (d.oka != 0ull || d.okb != 0ull)) {
            gp_canvas_register_table_overlay(cvs, d.oka, d.okb, d.port_a, d.port_b);
            mg->overlay_key_a = d.oka;
            mg->overlay_key_b = d.okb;
        }
        // If a ring parameter was serialized, prefer to reuse an existing
        // ring from the restored RopeSim blob. Fall back to recreating a ring
        // only when we cannot find a matching ring to bind.
        if (d.ring_mode != 0) {
            float ru = d.ring_u;
            uint64_t target_uid = 0ull;
            if (d.anchor_uid != 0ull) target_uid = d.anchor_uid;
            else if (!d.verts.empty()) target_uid = d.verts[0].first;
            if (target_uid != 0ull) {
                int ring_id = -1;
                RopeSim* sim = ctx->rope_sim;
                if (sim && rope_sim_get_ring_count(sim) > 0) {
                    int resolved = -1;
                    if (cvs) resolved = gp_canvas_resolve_rope_id_to_index(cvs, ctx, target_uid);
                    if (resolved < 0) resolved = gp_table_resolve_rope_id_to_sim_index(ctx, target_uid);
                    if (resolved >= 0) {
                        int ring_count = rope_sim_get_ring_count(sim);
                        float best_dist = std::numeric_limits<float>::infinity();
                        for (int ri = 0; ri < ring_count; ++ri) {
                            int rope_idx = -1;
                            float ring_u = 0.0f;
                            if (!rope_sim_get_ring_rope_index(sim, ri, &rope_idx)) continue;
                            if (rope_idx != resolved) continue;
                            if (!rope_sim_get_ring_u(sim, ri, &ring_u)) continue;
                            float dist = std::fabs(ring_u - ru);
                            if (dist < best_dist) {
                                best_dist = dist;
                                ring_id = ri;
                            }
                        }
                    }
                }
                if (ring_id < 0) {
                    ring_id = gp_table_create_ring_by_id(ctx, target_uid, ru);
                    if (ring_id == -2) {
                        printf("gp_table_deserialize: ERROR - could not resolve rope id=%llu for ring creation\n", (unsigned long long)target_uid);
                        return 3; // hard-fail: unresolved rope id for ring
                    }
                    if (ring_id >= 0) {
                        printf("gp_table_deserialize: recreated ring id=%d u=%.3f for mg=%p id=%llu\n", ring_id, ru, (void*)mg, (unsigned long long)mg->id);
                    }
                }
                if (ring_id >= 0) {
                    gp_table_register_ring_edge(ctx, ring_id, mg->id);
                }
            }
        }
        mg->overlay_key_a = d.oka;
        mg->overlay_key_b = d.okb;
        // If a canvas is present, prefer attaching the canvas root RopeSim
        // so meta-group membership is registered against the shared root sim
        // instead of creating a local simulator. Do not synthesize a local
        // sim here; defer to the canvas to create one if needed.
        if (cvs && !ctx->rope_sim) {
            void* root_sim_void = gp_canvas_get_rope_sim(cvs);
            if (root_sim_void) {
                RopeSim* rootsim = reinterpret_cast<RopeSim*>(root_sim_void);
                gp_table_attach_rope_sim(ctx, rootsim, 0);
                printf("gp_table_deserialize: attached canvas root RopeSim %p to table %p\n", (void*)rootsim, (void*)ctx);
            } else {
                printf("gp_table_deserialize: canvas has no root RopeSim available; deferring sim attachment for table %p\n", (void*)ctx);
            }
        }

        // add vertices (resolve via canvas mapping by persistent rope UID)
        printf("gp_table_deserialize: mg id=%llu sim_idx=%d overlay_oka=%llu okb=%llu lasso_flags=%u widget=%u spring_min_rest=%.2f spring_reduce_rate=%.2f spring_mode=%u verts=%zu\n",
               (unsigned long long)mg->id, d.sim_idx, (unsigned long long)d.oka, (unsigned long long)d.okb,
               d.lc.flags, static_cast<unsigned int>(d.lc.widget_type), d.lc.spring_min_rest, d.lc.spring_reduce_rate, static_cast<unsigned int>(d.lc.spring_mode), d.verts.size());

        for (auto &vp : d.verts) {
            uint64_t want_uid = vp.first;
            if (want_uid == 0ull) {
                printf("gp_table_deserialize: warning, vertex has zero rope_uid, skipping\n");
                continue;
            }
            if (!ctx->rope_sim) {
                printf("gp_table_deserialize: deferring meta vertex rope_id=%llu (no RopeSim attached yet)\n", (unsigned long long)want_uid);
                gp_table_queue_pending_meta_vertex(ctx, mg, want_uid, vp.second);
                continue;
            }
            int resolved = -1;
            if (cvs) resolved = gp_canvas_resolve_rope_id_to_index(cvs, ctx, want_uid);
            if (resolved < 0) resolved = gp_table_resolve_rope_id_to_sim_index(ctx, want_uid);
            printf("gp_table_deserialize: mapping rope_id=%llu -> resolved_idx=%d vert=%d\n", (unsigned long long)want_uid, resolved, vp.second);
            if (resolved < 0) {
                printf("gp_table_deserialize: deferring unresolved rope id=%llu for meta-group vertex\n", (unsigned long long)want_uid);
                gp_table_queue_pending_meta_vertex(ctx, mg, want_uid, vp.second);
                continue;
            }
            // If we restored a RopeSim blob, preserve its meta-group membership
            // (including saved member_u) rather than re-adding members.
            const bool update_sim = !restored_sim_attached;
            gp_table_meta_add_vertex_with_id_internal(ctx, mg, want_uid, resolved, vp.second, update_sim);
        }
        // If we registered an overlay earlier, attach a single rope (prefer
        // the dangling widget rope) to that canonical overlay.
        if ((mg->overlay_key_a != 0ull || mg->overlay_key_b != 0ull) && cvs) {
            int overlay_rope_idx = -1;
            if (d.dang_rope >= 0) {
                overlay_rope_idx = d.dang_rope;
            } else if (d.anchor_uid != 0ull) {
                overlay_rope_idx = gp_canvas_resolve_rope_id_to_index(cvs, ctx, d.anchor_uid);
                if (overlay_rope_idx < 0) {
                    for (size_t ri = 0; ri < ctx->rope_ids.size(); ++ri) {
                        if (ctx->rope_ids[ri] == d.anchor_uid) { overlay_rope_idx = static_cast<int>(ri); break; }
                    }
                }
            } else if (!d.verts.empty()) {
                uint64_t ru = d.verts[0].first;
                if (ru != 0ull) {
                    overlay_rope_idx = gp_canvas_resolve_rope_id_to_index(cvs, ctx, ru);
                    if (overlay_rope_idx < 0) {
                        for (size_t ri = 0; ri < ctx->rope_ids.size(); ++ri) {
                            if (ctx->rope_ids[ri] == ru) { overlay_rope_idx = static_cast<int>(ri); break; }
                        }
                    }
                }
            }
            if (overlay_rope_idx >= 0) {
                gp_canvas_attach_rope_to_overlay(cvs, mg->overlay_key_a, mg->overlay_key_b, overlay_rope_idx);
            }
            gp_canvas_set_overlay_meta(cvs, mg->overlay_key_a, mg->overlay_key_b, ctx, reinterpret_cast<void*>(mg));
        }
        // If the serialized blob indicated a sim meta-group existed previously
        // (d.sim_idx >= 0) then enable edge-springs on the reconstituted
        // meta-group now that vertices have been registered into the current
        // RopeSim. This re-enables the short T-off / spring binding behavior
        // that was active when the snapshot was taken.
        if (d.sim_idx >= 0 && !restored_sim_attached) {
            // prefer saved spring params from the serialized LassoConfig; fall
            // back to conservative defaults if they are zero/unset.
            float use_min_rest = (d.lc.spring_min_rest > 0.0f) ? d.lc.spring_min_rest : 2.0f;
            float use_reduce_rate = (d.lc.spring_reduce_rate > 0.0f) ? d.lc.spring_reduce_rate : 50.0f;
            gp_table_meta_set_edge_spring_params(ctx, mg, use_min_rest, use_reduce_rate, static_cast<int32_t>(d.lc.spring_mode));
            gp_table_meta_enable_edge_springs(ctx, mg, use_min_rest, use_reduce_rate);
        }
        // log sim installation if present
        {
            RopeSim* sim = ctx->rope_sim;
            if (sim && mg->sim_group_idx >= 0) {
                printf("gp_table_deserialize: installed mg=%p id=%llu sim=%p sim_group_idx=%d vertices=%zu\n",
                       (void*)mg, (unsigned long long)mg->id, (void*)sim, mg->sim_group_idx, mg->vertices.size());
                fflush(stdout);
            }
        }
        // set lasso config via API
        gp_table_meta_set_lasso_config(ctx, mg, &d.lc);
        // set anchor if present (resolved earlier)
        if (mg->anchor_rope >= 0) {
            gp_table_meta_set_anchor(ctx, mg, mg->anchor_rope, mg->anchor_vert);
        }
        // set channel/group
        gp_table_meta_set_channel_group(ctx, mg, d.channel_group);
        // Dump meta-group state after vertices and attachments for diagnostics
        gp_table_debug_dump_meta_group(ctx, mg, "deserialize_post_add");
        
    }

    gp_table_apply_pending_meta_vertices(ctx);

    // If this table blob contained module UUID or frame-port UUIDs, attempt
    // to register them with the canvas so bindings can be reconstructed
    // deterministically. Find the canvas module index for this table.
    if (ctx->module_uuid != 0ull && cvs) {
        // register module UUID with canvas using table pointer (canvas will
        // map table->module_idx internally)
        gp_canvas_register_table_module_uuid(cvs, ctx, ctx->module_uuid);
        // register frame-port UUIDs
        for (const auto &fpe : ctx->frame_port_uuids) {
            gp_canvas_register_table_frame_port_uuid(cvs, ctx, fpe.row, fpe.idx, fpe.uuid);
        }
    }
    return 1;
}

// Relaxation helper: ensure internal arrays match edges size (called before stepping)
static void ensure_relax_vectors(GP_TableContext* ctx) {
    if (!ctx) return;
    size_t n = ctx->edges.size();
    if (ctx->relax_value.size() < n) {
        ctx->relax_value.resize(n, 1.0f);
        ctx->relax_vel.resize(n, 0.0f);
    } else if (ctx->relax_value.size() > n) {
        ctx->relax_value.resize(n);
        ctx->relax_vel.resize(n);
    }
}

