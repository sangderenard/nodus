// Enqueue operations -------------------------------------------------------
int32_t gp_table_enqueue_add_edge(GP_TableContext* ctx, unsigned long long a, unsigned long long b) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
    GP_TableContext::PendingOp op;
    op.type = GP_TableContext::PENDING_OP_ADD_EDGE;
    op.a_key = a;
    op.b_key = b;
    ctx->pending_ops.push_back(op);
    return 1;
}

int32_t gp_table_enqueue_clear_edges(GP_TableContext* ctx) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
    GP_TableContext::PendingOp op;
    op.type = GP_TableContext::PENDING_OP_CLEAR_EDGES;
    ctx->pending_ops.push_back(op);
    return 1;
}

int32_t gp_table_enqueue_edge_subscribe_ex(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, int32_t start_at_head) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
    GP_TableContext::PendingOp op;
    op.type = GP_TableContext::PENDING_OP_SUBSCRIBE_EDGE;
    op.edge_idx = edge_idx;
    op.sub_key = subscriber_key;
    op.start_at_head = start_at_head;
    ctx->pending_ops.push_back(op);
    return 1;
}

int32_t gp_table_enqueue_edge_unsubscribe(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
    GP_TableContext::PendingOp op;
    op.type = GP_TableContext::PENDING_OP_UNSUBSCRIBE_EDGE;
    op.edge_idx = edge_idx;
    op.sub_key = subscriber_key;
    ctx->pending_ops.push_back(op);
    return 1;
}

int32_t gp_table_enqueue_bind_stage_port(GP_TableContext* ctx, unsigned long long led_key, GP_StageContext* stage, int32_t is_output, int32_t channel) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
    GP_TableContext::PendingOp op;
    op.type = GP_TableContext::PENDING_OP_BIND_STAGE;
    op.a_key = led_key;
    op.stage = stage;
    op.is_output = is_output;
    op.channel = channel;
    ctx->pending_ops.push_back(op);
    return 1;
}

// Set/Get per-edge subgroup flags (color hues, BYREF bit, etc.). Caller should
// call `gp_table_apply_pending_ops` or similar manager-side sync to have
// reconfiguration take effect; this function updates the context and triggers
// an immediate FIFO sync for that edge.
// Forward declare file-scope sync helper so earlier callers can invoke it.
static void sync_edge_tensor_for_idx(GP_TableContext* ctx, size_t ei);
int32_t gp_table_set_edge_subgroup_flags(GP_TableContext* ctx, int32_t edge_idx, uint32_t flags) {
    if (!ctx) return 0;
    if (edge_idx < 0 || static_cast<size_t>(edge_idx) >= ctx->edges.size()) return 0;
    ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)] = flags;
    // forward declaration may be below; ensure symbol visible by declaring prototype above in file
    sync_edge_tensor_for_idx(ctx, static_cast<size_t>(edge_idx));
    return 1;
}

int32_t gp_table_get_edge_subgroup_flags(GP_TableContext* ctx, int32_t edge_idx, uint32_t* out_flags) {
    if (!ctx || !out_flags) return 0;
    if (edge_idx < 0 || static_cast<size_t>(edge_idx) >= ctx->edges.size()) return 0;
    *out_flags = ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)];
    return 1;
}

int32_t gp_table_enqueue_unbind_stage_port(GP_TableContext* ctx, unsigned long long led_key) {
    if (!ctx) return 0;
    std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
    GP_TableContext::PendingOp op;
    op.type = GP_TableContext::PENDING_OP_UNBIND_STAGE;
    op.a_key = led_key;
    ctx->pending_ops.push_back(op);
    return 1;
}

int32_t gp_table_set_sim_enabled(GP_TableContext* ctx, int32_t enabled) {
    if (!ctx) return 0;
    ctx->sim_enabled = (enabled ? 1 : 0);
    return 1;
}

int32_t gp_table_get_sim_enabled(GP_TableContext* ctx, int32_t* out_enabled) {
    if (!ctx || !out_enabled) return 0;
    *out_enabled = ctx->sim_enabled;
    return 1;
}

// Apply pending ops (manager thread should call this before scheduling)
int32_t gp_table_apply_pending_ops(GP_TableContext* ctx) {
    if (!ctx) return 0;
    std::vector<GP_TableContext::PendingOp> ops;
    {
        std::lock_guard<std::mutex> lk(ctx->pending_ops_mu);
        if (ctx->pending_ops.empty()) return 1;
        ops.swap(ctx->pending_ops);
    }
    for (const auto &op : ops) {
        switch (op.type) {
        case GP_TableContext::PENDING_OP_ADD_EDGE:
            gp_table_add_edge(ctx, op.a_key, op.b_key);
            break;
        case GP_TableContext::PENDING_OP_CLEAR_EDGES:
            gp_table_clear_edges(ctx);
            break;
        case GP_TableContext::PENDING_OP_SUBSCRIBE_EDGE:
            gp_table_edge_subscribe_ex(ctx, op.edge_idx, op.sub_key, op.start_at_head);
            break;
        case GP_TableContext::PENDING_OP_UNSUBSCRIBE_EDGE:
            gp_table_edge_unsubscribe(ctx, op.edge_idx, op.sub_key);
            break;
        case GP_TableContext::PENDING_OP_BIND_STAGE:
            gp_table_bind_stage_port(ctx, op.a_key, op.stage, op.is_output, op.channel);
            break;
        case GP_TableContext::PENDING_OP_UNBIND_STAGE:
            gp_table_unbind_stage_port(ctx, op.a_key);
            break;
        default:
            break;
        }
    }
    return 1;
}

// Network-only snapshot implementation (no locking; caller must be manager thread)
int32_t gp_table_snapshot_network_size(GP_TableContext* ctx, int32_t* out_node_count, int32_t* out_edge_count, uint64_t* out_stamp) {
    if (!ctx || !out_node_count || !out_edge_count || !out_stamp) return 0;
    // Count unique endpoint keys
    std::unordered_set<uint64_t> keys;
    keys.reserve(ctx->edges.size() * 2 + 1);
    for (const auto &e : ctx->edges) {
        keys.insert(e.first);
        keys.insert(e.second);
    }
    *out_node_count = static_cast<int32_t>(keys.size());
    *out_edge_count = static_cast<int32_t>(ctx->edges.size());
    // Stamp: simple generation combining edge count and next_edge_id to detect changes
    uint64_t stamp = (static_cast<uint64_t>(ctx->edges.size()) << 32) ^ (ctx->next_edge_id & 0xffffffffull);
    *out_stamp = stamp;
    return 1;
}

int32_t gp_table_snapshot_network_fill(GP_TableContext* ctx, uint64_t* node_buf, int32_t node_buf_len, GP_TableEdgeSnapshot* edge_buf, int32_t edge_buf_len, uint64_t expected_stamp) {
    if (!ctx || !node_buf || !edge_buf) return 0;
    // Recompute stamp
    uint64_t stamp = (static_cast<uint64_t>(ctx->edges.size()) << 32) ^ (ctx->next_edge_id & 0xffffffffull);
    if (expected_stamp != stamp) return 0; // caller should retry size/fill

    // Build a node index map in caller-visible order: insert as discovered while scanning edges
    std::unordered_map<uint64_t,uint32_t> idx;
    idx.reserve(ctx->edges.size() * 2 + 1);
    uint32_t next_idx = 0;
    for (const auto &e : ctx->edges) {
        if (idx.find(e.first) == idx.end()) {
            if (next_idx >= static_cast<uint32_t>(node_buf_len)) return 0;
            idx[e.first] = next_idx;
            node_buf[next_idx] = e.first;
            ++next_idx;
        }
        if (idx.find(e.second) == idx.end()) {
            if (next_idx >= static_cast<uint32_t>(node_buf_len)) return 0;
            idx[e.second] = next_idx;
            node_buf[next_idx] = e.second;
            ++next_idx;
        }
    }
    if (next_idx > static_cast<uint32_t>(node_buf_len)) return 0;
    // Fill edges
    if (static_cast<int32_t>(ctx->edges.size()) > edge_buf_len) return 0;
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto &e = ctx->edges[i];
        auto it_a = idx.find(e.first);
        auto it_b = idx.find(e.second);
        if (it_a == idx.end() || it_b == idx.end()) return 0; // should not happen
        edge_buf[i].a_idx = it_a->second;
        edge_buf[i].b_idx = it_b->second;
        edge_buf[i].edge_uid = (i < ctx->edge_ids.size()) ? ctx->edge_ids[i] : 0ull;
    }
    return 1;
}

RopeSim* gp_table_get_rope_sim(GP_TableContext* ctx) {
    if (!ctx) return nullptr;
    return ctx->rope_sim;
}

int32_t gp_table_set_step_callback(GP_TableContext* ctx, GP_TableStepFn cb, void* user) {
    if (!ctx) return 0;
    ctx->step_callback = cb;
    ctx->step_user = user;
    return 1;
}

int32_t gp_table_clear_step_callback(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ctx->step_callback = nullptr;
    ctx->step_user = nullptr;
    return 1;
}

int32_t gp_table_step(GP_TableContext* ctx, const float* inputs, int32_t in_count, float* outputs, int32_t out_count, double dt) {
    if (!ctx) return 0;
    if (!ctx->step_callback) return 0;
    // Defensive: allow null arrays as zero-length
    if ((in_count > 0 && !inputs) || (out_count > 0 && !outputs)) return 0;
    try {
        ctx->step_callback(ctx->step_user, inputs, in_count, outputs, out_count, dt);
        return 1;
    } catch (...) {
        return 0;
    }
}

int32_t gp_table_set_actions(GP_TableContext* ctx, const GP_TableAction* actions, int32_t count) {
    if (!ctx) return 0;
    if (count < 0) return 0;
    if (count > 0 && !actions) return 0;
    if (count == 0) {
        ctx->actions.clear();
        return 1;
    }
    ctx->actions.assign(actions, actions + count);
    return 1;
}

int32_t gp_table_set_action_callback(GP_TableContext* ctx, GP_TableActionFn cb, void* user) {
    if (!ctx) return 0;
    ctx->action_callback = cb;
    ctx->action_user = user;
    return 1;
}

int32_t gp_table_clear_action_callback(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ctx->action_callback = nullptr;
    ctx->action_user = nullptr;
    return 1;
}

int32_t gp_table_set_key_callback(GP_TableContext* ctx, GP_TableKeyFn cb, void* user) {
    if (!ctx) return 0;
    ctx->key_callback = cb;
    ctx->key_user = user;
    return 1;
}

int32_t gp_table_clear_key_callback(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ctx->key_callback = nullptr;
    ctx->key_user = nullptr;
    return 1;
}

int32_t gp_table_on_key(GP_TableContext* ctx, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
    if (!ctx) return 0;
    if (!ctx->key_callback) return 0;
    try {
        ctx->key_callback(ctx->key_user, key, scancode, action, mods);
        return 1;
    } catch (...) {
        return 0;
    }
}

static void recompute_geom(GP_TableContext* ctx) {
    if (!ctx) return;
    ctx->geom.width_px = ctx->st.w;
    int h_sum = 0;
    for (const auto& r : ctx->rows) {
        int rh = row_desired_height_px(ctx->st, r);
        h_sum += std::max(1, rh);
    }
    ctx->geom.height_px = std::max<int32_t>(1, static_cast<int32_t>(h_sum));
    compute_columns(ctx->cols.data(), static_cast<int>(ctx->cols.size()), ctx->st.w, ctx->st.name_w, ctx->geom.col_x0, ctx->geom.col_w);
}

// Ensure the edge FIFO list matches the edge list length.
static void ensure_edge_fifos(GP_TableContext* ctx) {
    if (!ctx) return;
    while (ctx->edge_fifos.size() < ctx->edges.size()) {
        EdgeTensorFifo fifo;
        fifo.configure_default();
        fifo.set_friction_regions(ctx->st.cable_fifo_friction_regions);
        ctx->edge_fifos.push_back(std::move(fifo));
    }
    if (ctx->edge_fifos.size() > ctx->edges.size()) {
        ctx->edge_fifos.resize(ctx->edges.size());
    }
    for (auto &fifo : ctx->edge_fifos) {
        fifo.set_friction_regions(ctx->st.cable_fifo_friction_regions);
    }
    while (ctx->edge_batch_metadata.size() < ctx->edges.size()) {
        ctx->edge_batch_metadata.emplace_back(GP_TableEdgeBatchMetadata{});
    }
    if (ctx->edge_batch_metadata.size() > ctx->edges.size()) {
        ctx->edge_batch_metadata.resize(ctx->edges.size());
    }
    while (ctx->edge_subgroup_flags.size() < ctx->edges.size()) {
        ctx->edge_subgroup_flags.push_back(0u);
    }
    if (ctx->edge_subgroup_flags.size() > ctx->edges.size()) {
        ctx->edge_subgroup_flags.resize(ctx->edges.size());
    }
    // keep subscriber slot maps in sync with edges
    while (ctx->edge_subscriber_slots.size() < ctx->edges.size()) ctx->edge_subscriber_slots.emplace_back();
    if (ctx->edge_subscriber_slots.size() > ctx->edges.size()) ctx->edge_subscriber_slots.resize(ctx->edges.size());
}

// Apply stage port bindings to an edge's FIFO: output keys claim writer, input keys subscribe.
static void sync_edge_tensor_for_idx(GP_TableContext* ctx, size_t ei) {
    if (!ctx) return;
    ensure_edge_fifos(ctx);
    if (ei >= ctx->edges.size() || ei >= ctx->edge_fifos.size()) return;
    const auto &edge = ctx->edges[ei];
    EdgeTensorFifo &fifo = ctx->edge_fifos[ei];
    auto bind_one = [&](uint64_t key) {
        auto it = ctx->stage_ports.find(key);
        if (it == ctx->stage_ports.end()) return;
        const StagePortBinding &b = it->second;
        if (b.is_output) {
            fifo.maybe_claim_writer(key);
        } else {
            fifo.subscribe(key, /*start_at_head=*/true);
        }
    };
    bind_one(edge.first);
    bind_one(edge.second);

    // If edge has BYREF subgroup flag, ensure FIFO stride can carry pointer-sized
    // payloads. We do this in-place so the FIFO object does not need to be
    // destroyed and re-created; existing samples are preserved where possible.
    if (ei < ctx->edge_subgroup_flags.size()) {
        uint32_t f = ctx->edge_subgroup_flags[ei];
        if (flags_imply_byref(f)) {
            // ensure stride can hold a pointer
            EdgeTensorFifo &ef = ctx->edge_fifos[ei];
            ef.ensure_stride_for_bytes(sizeof(void*));
        }
    }
}

static void sync_edge_tensors_for_key(GP_TableContext* ctx, uint64_t key) {
    if (!ctx) return;
    ensure_edge_fifos(ctx);
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto &e = ctx->edges[i];
        if (e.first == key || e.second == key) {
            sync_edge_tensor_for_idx(ctx, i);
        }
    }
}

static void draw_circle_outline(uint8_t* img, int w, int h, int pitch, int cx, int cy, int r, int thickness, Color c) {
    if (!img || r <= 0) return;
    const int r2 = r * r;
    const int outer = r + thickness;
    const int outer2 = outer * outer;
    for (int dy = -outer; dy <= outer; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= h) continue;
        for (int dx = -outer; dx <= outer; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= w) continue;
            int d2 = dx * dx + dy * dy;
            if (d2 >= r2 && d2 <= outer2) {
                uint8_t* p = img + y * pitch + x * 4;
                p[0] = c.r; p[1] = c.g; p[2] = c.b; p[3] = c.a;
            }
        }
    }
}

// Blend src color (with alpha 0..255) over destination pixel in-place
static inline void blend_pixel(uint8_t* dst, uint8_t sr, uint8_t sg, uint8_t sb, uint8_t sa) {
    if (!dst) return;
    float a = sa / 255.0f;
    if (a <= 0.0f) return;
    float inv = 1.0f - a;
    float dr = dst[0] / 255.0f;
    float dg = dst[1] / 255.0f;
    float db = dst[2] / 255.0f;
    float da = dst[3] / 255.0f;
    float srf = sr / 255.0f;
    float sgf = sg / 255.0f;
    float sbf = sb / 255.0f;
    float outa = a + da * inv;
    if (outa <= 0.0f) {
        dst[0] = dst[1] = dst[2] = dst[3] = 0;
        return;
    }
    float out_r = (srf * a + dr * da * inv) / outa;
    float out_g = (sgf * a + dg * da * inv) / outa;
    float out_b = (sbf * a + db * da * inv) / outa;
    dst[0] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_r)) * 255.0f));
    dst[1] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_g)) * 255.0f));
    dst[2] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, out_b)) * 255.0f));
    dst[3] = static_cast<uint8_t>(std::lround(std::max(0.0f, std::min(1.0f, outa)) * 255.0f));
}

// Draw a filled circle with blending (soft core) used for cable sampling points
static void draw_blob_blend(uint8_t* img, int w, int h, int pitch, int cx, int cy, int radius, Color c) {
    if (!img || radius <= 0) return;
    int r = radius;
    int r2 = r * r;
    int y0 = std::max(0, cy - r);
    int y1 = std::min(h - 1, cy + r);
    for (int y = y0; y <= y1; ++y) {
        int dy = y - cy;
        int dx_limit = static_cast<int>(std::floor(std::sqrt((double)r2 - double(dy * dy))));
        int x0 = std::max(0, cx - dx_limit);
        int x1 = std::min(w - 1, cx + dx_limit);
        for (int x = x0; x <= x1; ++x) {
            int dx = x - cx;
            int d2 = dx * dx + dy * dy;
            if (d2 > r2) continue;
            // simple linear falloff alpha across radius
            float t = 1.0f - (std::sqrt((float)d2) / (float)r);
            uint8_t sa = static_cast<uint8_t>(std::lround(c.a * t));
            uint8_t sr = c.r;
            uint8_t sg = c.g;
            uint8_t sb = c.b;
            uint8_t* dst = img + y * pitch + x * 4;
            blend_pixel(dst, sr, sg, sb, sa);
        }
    }
}

// Draw a smeared segment using the segment tangent as the cross-section normal.
// This paints a continuous tube between (x1,y1) and (x2,y2) with radius and soft falloff.
static void draw_segment_smear(uint8_t* img, int w, int h, int pitch, float x1, float y1, float x2, float y2, int radius, Color c, float alpha_scale = 1.0f) {
    if (!img || radius <= 0) return;
    float dx = x2 - x1;
    float dy = y2 - y1;
    float len2 = dx*dx + dy*dy;
    if (len2 <= 1e-6f) {
        // fallback to blob
        draw_blob_blend(img, w, h, pitch, int(std::lround(x1)), int(std::lround(y1)), radius, c);
        return;
    }
    float len = std::sqrt(len2);
    // normal pointing to the left of the segment (perpendicular)
    float nx = -dy / len;
    float ny = dx / len;
    float r = float(radius);
    float rplus = r + 1.0f;
    int minx = static_cast<int>(std::floor(std::min(x1,x2) - rplus));
    int maxx = static_cast<int>(std::ceil(std::max(x1,x2) + rplus));
    int miny = static_cast<int>(std::floor(std::min(y1,y2) - rplus));
    int maxy = static_cast<int>(std::ceil(std::max(y1,y2) + rplus));
    minx = std::max(minx, 0);
    miny = std::max(miny, 0);
    maxx = std::min(maxx, w - 1);
    maxy = std::min(maxy, h - 1);
    for (int y = miny; y <= maxy; ++y) {
        for (int x = minx; x <= maxx; ++x) {
            // compute projection along segment
            float vx = float(x) - x1;
            float vy = float(y) - y1;
            float proj = (vx * dx + vy * dy) / len2;
            float t = std::clamp(proj, 0.0f, 1.0f);
            float cxp = x1 + dx * t;
            float cyp = y1 + dy * t;
            // lateral distance to the segment centerline
            float lx = float(x) - cxp;
            float ly = float(y) - cyp;
            float lateral = std::abs(lx * nx + ly * ny);
            if (lateral > r) continue;
            // simple linear falloff by lateral distance
            float fall = 1.0f - (lateral / r);
            uint8_t sa = static_cast<uint8_t>(std::lround(float(c.a) * fall * alpha_scale));
            uint8_t* dst = img + y * pitch + x * 4;
            blend_pixel(dst, c.r, c.g, c.b, sa);
        }
    }
}

// High-quality parametric SDF segment: uses endpoint tangents to interpolate a cross-section
// and computes a smooth Gaussian falloff in cross-section distance.
static void draw_segment_parametric_sdf(uint8_t* img, int w, int h, int pitch,
    float x1, float y1, float x2, float y2,
    float tx1, float ty1, float tx2, float ty2,
    int radius, Color c, float alpha_scale = 1.0f, float tint_strength = 0.0f) {
    if (!img || radius <= 0) return;
    float dx = x2 - x1;
    float dy = y2 - y1;
    float len2 = dx*dx + dy*dy;
    if (len2 <= 1e-6f) {
        draw_blob_blend(img, w, h, pitch, int(std::lround(x1)), int(std::lround(y1)), radius, c);
        return;
    }
    float r = float(radius);
    float rplus = r + 2.0f;
    int minx = static_cast<int>(std::floor(std::min(x1,x2) - rplus));
    int maxx = static_cast<int>(std::ceil(std::max(x1,x2) + rplus));
    int miny = static_cast<int>(std::floor(std::min(y1,y2) - rplus));
    int maxy = static_cast<int>(std::ceil(std::max(y1,y2) + rplus));
    minx = std::max(minx, 0);
    miny = std::max(miny, 0);
    maxx = std::min(maxx, w - 1);
    maxy = std::min(maxy, h - 1);

    // pre-normalize endpoint tangents
    float t1len = std::sqrt(tx1*tx1 + ty1*ty1);
    float t2len = std::sqrt(tx2*tx2 + ty2*ty2);
    if (t1len <= 1e-6f) { tx1 = dx; ty1 = dy; t1len = std::sqrt(dx*dx+dy*dy); }
    if (t2len <= 1e-6f) { tx2 = dx; ty2 = dy; t2len = std::sqrt(dx*dx+dy*dy); }
    tx1 /= t1len; ty1 /= t1len; tx2 /= t2len; ty2 /= t2len;

    // Gaussian width parameter (so radius maps to ~3-sigma). Use smoother falloff.
    float sigma = r / 2.0f;
    float inv2sig2 = 1.0f / (2.0f * sigma * sigma);

    for (int y = miny; y <= maxy; ++y) {
        for (int x = minx; x <= maxx; ++x) {
            // project point onto the segment (param t in [0,1])
            float vx = float(x) - x1;
            float vy = float(y) - y1;
            float proj = (vx * dx + vy * dy) / len2;
            float t = std::clamp(proj, 0.0f, 1.0f);
            float cxp = x1 + dx * t;
            float cyp = y1 + dy * t;

            // interpolate tangent
            float tx = (1.0f - t) * tx1 + t * tx2;
            float ty = (1.0f - t) * ty1 + t * ty2;
            float tlen = std::sqrt(tx*tx + ty*ty);
            if (tlen <= 1e-6f) continue;
            tx /= tlen; ty /= tlen;

            // normal (perp)
            float nx = -ty;
            float ny = tx;

            float lx = float(x) - cxp;
            float ly = float(y) - cyp;
            float lateral = std::abs(lx * nx + ly * ny);
            if (lateral > r) continue;

            // Gaussian falloff based on lateral distance
            float gauss = std::exp(- (lateral * lateral) * inv2sig2);

            // compute tint factor based on radial proximity (1 at center, 0 at radius)
            float radial = std::clamp(1.0f - (lateral / r), 0.0f, 1.0f);
            float tint_amt = std::clamp(tint_strength * radial, 0.0f, 1.0f);

            // blend between neutral jacket (light grey) and provided color by tint_amt
            float base_r = 200.0f;
            float base_g = 200.0f;
            float base_b = 200.0f;
            float pr = base_r * (1.0f - tint_amt) + float(c.r) * tint_amt;
            float pg = base_g * (1.0f - tint_amt) + float(c.g) * tint_amt;
            float pb = base_b * (1.0f - tint_amt) + float(c.b) * tint_amt;

            uint8_t sa = static_cast<uint8_t>(std::lround(float(c.a) * gauss * alpha_scale));
            uint8_t* dst = img + y * pitch + x * 4;
            blend_pixel(dst, static_cast<uint8_t>(std::lround(pr)), static_cast<uint8_t>(std::lround(pg)), static_cast<uint8_t>(std::lround(pb)), sa);
        }
    }
}

static void draw_rope_curve_blend_rgb(uint8_t* img, int w, int h, int pitch, const float* verts, int count,
    int jacket_px, int jacket_border, Color col_a, Color col_b, float intensity) {
    if (!img || !verts || count < 2) return;

    auto get = [&](int idx) {
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts[2*idx+0], verts[2*idx+1]);
    };
    auto catmull = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        return std::pair<float,float>(x,y);
    };
    auto catmull_deriv = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float dx = 0.5f * ((-p0.first + p2.first) + 2.0f * (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t + 3.0f * (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t2);
        float dy = 0.5f * ((-p0.second + p2.second) + 2.0f * (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t + 3.0f * (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t2);
        return std::pair<float,float>(dx, dy);
    };

    std::vector<std::pair<float,float>> samples;
    std::vector<std::pair<float,float>> tangents;
    samples.reserve((count - 1) * 8);
    tangents.reserve((count - 1) * 8);
    for (int i = 0; i < count - 1; ++i) {
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        int s_start = (i == 0) ? 0 : 1; // avoid duplicate samples at segment boundaries
        for (int s = s_start; s <= n; ++s) {
            float t = float(s) / float(n);
            auto p = catmull(i, t);
            auto d = catmull_deriv(i, t);
            samples.emplace_back(p.first, p.second);
            tangents.emplace_back(d.first, d.second);
        }
    }
    if (samples.empty()) return;

    auto lerp_color = [&](const Color &a, const Color &b, float t) {
        float tt = std::clamp(t, 0.0f, 1.0f);
        Color out;
        out.r = static_cast<uint8_t>(std::lround(float(a.r) * (1.0f - tt) + float(b.r) * tt));
        out.g = static_cast<uint8_t>(std::lround(float(a.g) * (1.0f - tt) + float(b.g) * tt));
        out.b = static_cast<uint8_t>(std::lround(float(a.b) * (1.0f - tt) + float(b.b) * tt));
        out.a = static_cast<uint8_t>(std::lround(float(a.a) * (1.0f - tt) + float(b.a) * tt));
        return out;
    };

    int eff_jacket = std::max(1, jacket_px - 1);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float u = float(i) / float(std::max<size_t>(1, samples.size() - 1));
        Color jacket_col = lerp_color(col_a, col_b, u);
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, eff_jacket, jacket_col, 1.0f, intensity);
    }

    int core_r = std::max(1, jacket_px - jacket_border - 0);
    uint8_t core_alpha = static_cast<uint8_t>(std::lround(255.0f * std::clamp(intensity, 0.0f, 1.0f) * 0.18f));
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float u = float(i) / float(std::max<size_t>(1, samples.size() - 1));
        Color core_col = lerp_color(col_a, col_b, u);
        core_col.a = core_alpha;
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, core_r, core_col, 1.0f, intensity);
    }
}

static void draw_rope_curve_blend_rgb_falloff(uint8_t* img, int w, int h, int pitch, const float* verts, int count,
    int jacket_px, int jacket_border, Color col_a, Color col_b, float glow_a, float glow_b, float decay) {
    if (!img || !verts || count < 2) return;

    auto get = [&](int idx) {
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts[2*idx+0], verts[2*idx+1]);
    };
    auto catmull = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        return std::pair<float,float>(x,y);
    };
    auto catmull_deriv = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float dx = 0.5f * ((-p0.first + p2.first) + 2.0f * (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t + 3.0f * (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t2);
        float dy = 0.5f * ((-p0.second + p2.second) + 2.0f * (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t + 3.0f * (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t2);
        return std::pair<float,float>(dx, dy);
    };

    std::vector<std::pair<float,float>> samples;
    std::vector<std::pair<float,float>> tangents;
    samples.reserve((count - 1) * 8);
    tangents.reserve((count - 1) * 8);
    for (int i = 0; i < count - 1; ++i) {
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        int s_start = (i == 0) ? 0 : 1; // avoid duplicate samples at segment boundaries
        for (int s = s_start; s <= n; ++s) {
            float t = float(s) / float(n);
            auto p = catmull(i, t);
            auto d = catmull_deriv(i, t);
            samples.emplace_back(p.first, p.second);
            tangents.emplace_back(d.first, d.second);
        }
    }
    if (samples.empty()) return;

    auto mix = [&](const Color &a, const Color &b, float t) {
        float tt = std::clamp(t, 0.0f, 1.0f);
        Color out;
        out.r = static_cast<uint8_t>(std::lround(float(a.r) * (1.0f - tt) + float(b.r) * tt));
        out.g = static_cast<uint8_t>(std::lround(float(a.g) * (1.0f - tt) + float(b.g) * tt));
        out.b = static_cast<uint8_t>(std::lround(float(a.b) * (1.0f - tt) + float(b.b) * tt));
        out.a = static_cast<uint8_t>(std::lround(float(a.a) * (1.0f - tt) + float(b.a) * tt));
        return out;
    };

    int eff_jacket = std::max(1, jacket_px - 1);
    int core_r = std::max(1, jacket_px - jacket_border - 0);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float u = float(i) / float(std::max<size_t>(1, samples.size() - 1));
        float ia = glow_a * std::exp(-decay * u);
        float ib = glow_b * std::exp(-decay * (1.0f - u));
        float total = (ia + ib) * 1.6f;
        if (total <= 1e-4f) continue;
        float t = (total > 0.0f) ? (ib / total) : 0.0f;
        Color lit = mix(col_a, col_b, t);
        float tint = std::clamp(total, 0.0f, 1.0f);

        Color jacket_col = lit;
        jacket_col.a = 26;
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, eff_jacket, jacket_col, tint, tint);

        Color core_col = lit;
        core_col.a = static_cast<uint8_t>(std::lround(255.0f * 0.35f * tint));
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, core_r, core_col, 1.0f, tint);
    }
}

static void draw_segment_kernel_glow(uint8_t* img, int w, int h, int pitch,
    float x1, float y1, float x2, float y2, float radius, Color c, float alpha_scale = 1.0f) {
    if (!img || radius <= 0.0f) return;
    float dx = x2 - x1;
    float dy = y2 - y1;
    float len2 = dx * dx + dy * dy;
    float r = radius;
    float rplus = r + 2.0f;
    int minx = static_cast<int>(std::floor(std::min(x1, x2) - rplus));
    int maxx = static_cast<int>(std::ceil(std::max(x1, x2) + rplus));
    int miny = static_cast<int>(std::floor(std::min(y1, y2) - rplus));
    int maxy = static_cast<int>(std::ceil(std::max(y1, y2) + rplus));
    minx = std::max(minx, 0);
    miny = std::max(miny, 0);
    maxx = std::min(maxx, w - 1);
    maxy = std::min(maxy, h - 1);

    float sigma = r * 0.5f;
    float inv2sig2 = 1.0f / (2.0f * sigma * sigma);
    for (int y = miny; y <= maxy; ++y) {
        for (int x = minx; x <= maxx; ++x) {
            float t = 0.0f;
            if (len2 > 1e-6f) {
                float vx = float(x) - x1;
                float vy = float(y) - y1;
                float proj = (vx * dx + vy * dy) / len2;
                t = std::clamp(proj, 0.0f, 1.0f);
            }
            float cxp = x1 + dx * t;
            float cyp = y1 + dy * t;
            float lx = float(x) - cxp;
            float ly = float(y) - cyp;
            float dist2 = lx * lx + ly * ly;
            if (dist2 > r * r) continue;
            float gauss = std::exp(-dist2 * inv2sig2);
            uint8_t sa = static_cast<uint8_t>(std::lround(float(c.a) * gauss * alpha_scale));
            if (sa == 0) continue;
            uint8_t* dst = img + y * pitch + x * 4;
            blend_pixel(dst, c.r, c.g, c.b, sa);
        }
    }
}

static void draw_glow_blob(uint8_t* img, int w, int h, int pitch, int cx, int cy, int radius, float strength01, Color c) {
    if (!img) return;
    float s = std::max(0.0f, std::min(1.0f, strength01));
    if (s <= 0.0f) return;
    // Slightly larger falloff radius than the LED core to create diffusion.
    int glow_r = std::max(radius + 2, static_cast<int>(std::lround(float(radius) * (2.2f + 0.8f * s))));
    Color halo = c;
    halo.a = static_cast<uint8_t>(std::lround(float(c.a) * (0.55f + 0.45f * s)));
    draw_blob_blend(img, w, h, pitch, cx, cy, glow_r, halo);
    // Soft core to keep the center lively.
    Color core = c;
    core.a = static_cast<uint8_t>(std::lround(float(c.a) * std::min(1.0f, s * 1.4f)));
    draw_blob_blend(img, w, h, pitch, cx, cy, std::max(radius, glow_r / 3), core);
}

