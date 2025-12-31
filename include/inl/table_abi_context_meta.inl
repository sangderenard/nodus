// Stateful context ----------------------------------------------------------
#include "console_logger.h"
#ifndef printf
#define printf(...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
#ifndef fprintf
#define fprintf(file, ...) CONSOLE_PRINTF(__VA_ARGS__)
#endif

#include <atomic>
#include <memory>

struct GP_TableContext {
    GP_TableStyle style_raw{};
    Style st{};
    std::vector<GP_TableRow> rows;
    std::vector<GP_TableColumn> cols;
    GP_TableGeom geom{};
    // Scroll state: row offset (first visible row index) and fraction (0..1)
    int32_t scroll_row_offset = 0;
    float scroll_frac = 0.0f;
    float scroll_frac_x = 0.0f;
    // selected LED keys: packed (row<<32) | (col<<16) | led_index
    std::unordered_set<uint64_t> selected_leds;
    std::unordered_map<uint64_t, float> led_glow_strength;
    std::unordered_map<uint64_t, float> led_flow_glow_strength;
    std::vector<std::pair<uint64_t,uint64_t>> edges;
    std::vector<EdgeTensorFifo> edge_fifos; // companion FIFO per rope edge
    std::vector<uint64_t> edge_last_write_seq;
    std::vector<float> edge_flow_phase;
    std::vector<GP_TableEdgeBatchMetadata> edge_batch_metadata;
    std::vector<uint32_t> edge_subgroup_flags;
    // per-edge persistent unique id used for ThreadManager registration
    std::vector<uint64_t> edge_ids;
    uint64_t next_edge_id = 1;
    // per-edge subscriber_key -> reader_slot registered with ThreadManager
    std::vector<std::unordered_map<uint64_t,int>> edge_subscriber_slots;
    std::unordered_map<uint64_t, StagePortBinding> stage_ports; // LED key -> stage port binding
    // Relaxation state (per-edge values/velocities are maintained in parallel to edges)
    int32_t relax_mode = GP_TABLE_RELAX_OFF;
    float relax_stiffness = 10.0f;
    float relax_damping = 2.0f;
    float relax_threshold = 1e-3f;
    int32_t relax_max_iters = 200;
    double relax_last_time = 0.0; // seconds since epoch
    std::vector<float> relax_value; // 0..1 value per edge (1.0 = relaxed)
    std::vector<float> relax_vel;   // velocity per edge
    // rope simulator instance and per-rope sim index mapping
    RopeSim* rope_sim = nullptr;
    int rope_sim_owned = 0; // 1 if this context owns and should destroy the sim
    // mapping from persistent rope id -> rope_sim index. Using IDs
    // avoids relying on edge/vector ordering and is resilient to reorders.
    std::unordered_map<uint64_t,int> rope_id_to_sim_idx;
    // per-rope persistent unique ids (indexed by created rope order). Used
    // to map serialized vertices to runtime rope indices deterministically.
    std::vector<uint64_t> rope_ids;
    uint64_t next_rope_id = 1;
    // Deferred meta-group vertices awaiting rope-id resolution after sim attach.
    struct PendingMetaVertex {
        GP_MetaGroup* mg = nullptr;
        uint64_t rope_id = 0ull;
        int vertex_idx = -1;
    };
    std::vector<PendingMetaVertex> pending_meta_vertices;
    std::mutex pending_meta_mu;
    // flag set during gp_table_deserialize to indicate we're restoring
    int restoring = 0;
    // Optional module UUID embedded into serialized blobs (0 == unset)
    uint64_t module_uuid = 0ull;
    // Optional module-frame port UUIDs to serialize: entries are (row, idx, uuid)
    struct FramePortEntry { int32_t row; int32_t idx; uint64_t uuid; };
    std::vector<FramePortEntry> frame_port_uuids;
    // Prospective live-edge state (used when one node selected and mode enabled)
    int32_t prospective_mode = 0; // 0=off,1=on
    bool prospective_initialized = false;
    float prospective_x = 0.0f;
    float prospective_y = 0.0f;
    float prospective_vx = 0.0f;
    float prospective_vy = 0.0f;
    // queue of recent mouse targets (front = oldest to be cleared next)
    std::vector<std::pair<float,float>> prospective_targets;
    int32_t prospective_max_history = 8;
    float prospective_slack = 4.0f;
    float prospective_rope_length = 0.0f;
    int prospective_rope_idx = -1; // temporary rope index for live prospective cable
    // editable flag (editor potential). Default: editable (1).
    int32_t editable = 1;
    // per-key type hints and io sets
    std::unordered_map<uint64_t,int32_t> key_type_hint; // key -> type_id
    std::unordered_set<uint64_t> key_is_input; // keys marked input
    std::unordered_set<uint64_t> key_is_output; // keys marked output
    // Lock-free snapshot of detected type ids (diagnostic reader use).
    // Use a plain shared_ptr and the free-function atomic_load/atomic_store
    // overloads for `std::shared_ptr` to remain portable across MSVC.
    std::shared_ptr<std::vector<int32_t>> type_ids_snapshot{nullptr};
    // reading direction for sides (0=input,1=output). 0=LTR,1=TTB,2=RTL,3=BTB
    int32_t side_reading_dir[2] = {0, 0};
    // LED grid preference (cols, rows, aspect)
    int32_t led_pref_cols = 0;
    int32_t led_pref_rows = 0;
    float led_pref_aspect = 0.0f;
    // Optional step callback for node/table shims
    GP_TableStepFn step_callback = nullptr;
    void* step_user = nullptr;
    // Whether table-side simulator stepping is enabled (1) or disabled (0).
    // KPN or other managers can toggle this to pause heavy sim work per table.
    int32_t sim_enabled = 1;
    uint32_t debug_flags = 0u; // debug render/sim overrides propagated from canvas
    // (No per-table frame-skip; global frame-skip handled by table_abi global state)
    // Optional click-action dispatch
    std::vector<GP_TableAction> actions;
    GP_TableActionFn action_callback = nullptr;
    void* action_user = nullptr;
    // Optional keyboard callback
    GP_TableKeyFn key_callback = nullptr;
    void* key_user = nullptr;
    // meta-groups created by tools (opaque internal storage)
    struct RingEntry {
        int ring_id = -1; // rope_sim ring id
        uint64_t key = 0; // caller-provided key for identification
        EdgeTensorFifo fifo;
        GP_TableEdgeBatchMetadata batch_metadata{};
        uint32_t subgroup_flags = 0;
        uint64_t uid = 0;
    };
    std::vector<RingEntry> rings;
    uint64_t next_ring_uid = 1;
    std::vector<std::unordered_map<uint64_t,int>> ring_subscriber_slots; // per-ring subscriber -> slot
    struct GP_MetaGroupInternal;
    std::vector<std::unique_ptr<GP_MetaGroupInternal>> meta_groups;
    // Pending operations enqueued by UI threads to be applied by the manager.
    enum PendingOpType {
        PENDING_OP_ADD_EDGE = 1,
        PENDING_OP_CLEAR_EDGES = 2,
        PENDING_OP_SUBSCRIBE_EDGE = 3,
        PENDING_OP_UNSUBSCRIBE_EDGE = 4,
        PENDING_OP_BIND_STAGE = 5,
        PENDING_OP_UNBIND_STAGE = 6,
    };
    struct PendingOp {
        PendingOpType type;
        // fields used by various ops
        uint64_t a_key = 0;
        uint64_t b_key = 0;
        int32_t edge_idx = -1;
        uint64_t sub_key = 0;
        int32_t start_at_head = 1;
        GP_StageContext* stage = nullptr;
        int32_t is_output = 0;
        int32_t channel = 0;
    };
    std::vector<PendingOp> pending_ops;
    std::mutex pending_ops_mu;
};

// Global sim frame-skip state (shared across all tables)
static int g_global_sim_frame_skip_count = 0; // number of frames to skip between steps (0 = every frame)
static uint64_t g_global_sim_frame_tick = 0; // incremented once per canvas frame

static inline bool table_should_step_sim(GP_TableContext* ctx) {
    if (!ctx) return false;
    if (!ctx->sim_enabled) return false;
    int count = std::max(0, g_global_sim_frame_skip_count);
    if (count <= 0) return true;
    // step once every (count+1) frames
    return (g_global_sim_frame_tick % static_cast<uint64_t>(count + 1)) == 0ull;
}

int32_t gp_table_set_global_sim_frame_skip_count(int32_t count) {
    g_global_sim_frame_skip_count = std::max(0, count);
    return 1;
}

int32_t gp_table_get_global_sim_frame_skip_count(int32_t* out_count) {
    if (!out_count) return 0;
    *out_count = g_global_sim_frame_skip_count;
    return 1;
}

void gp_table_advance_global_sim_tick() {
    g_global_sim_frame_tick = (g_global_sim_frame_tick + 1) % 0xFFFFFFFFFFFFu;
}

int32_t gp_table_should_step_sim(GP_TableContext* ctx) {
    return table_should_step_sim(ctx) ? 1 : 0;
}

extern "C" int32_t gp_table_set_debug_flags(GP_TableContext* ctx, uint32_t flags) {
    if (!ctx) return 0;
    ctx->debug_flags = flags;
    return 1;
}

extern "C" int32_t gp_table_get_debug_flags(const GP_TableContext* ctx, uint32_t* out_flags) {
    if (!ctx || !out_flags) return 0;
    *out_flags = ctx->debug_flags;
    return 1;
}

extern "C" int32_t gp_table_set_cable_segments(GP_TableContext* ctx, int32_t segments) {
    if (!ctx) return 0;
    int segs = std::clamp(segments, 1, 64);
    ctx->st.cable_segments = segs;
    if (ctx->st.cable_fifo_friction_regions <= 0) {
        ctx->st.cable_fifo_friction_regions = std::max(1, segs);
    }
    return 1;
}

// Internal representation of a meta-group. Exposed to C callers as an
// opaque `GP_MetaGroup*` pointer (allocated here and stored in the
// table's `meta_groups` vector to keep lifetime management consistent).
struct GP_MetaVertex {
    uint64_t rope_id = 0ull;
    int rope_idx = -1;
    int vertex_idx = -1;
    float u = -1.0f;
};

struct GP_MetaGroup {
    std::vector<GP_MetaVertex> vertices; // rope identity + runtime index
    float confinement = 1.0f; // tightness/pressure
    int sim_group_idx = -1; // index into RopeSim meta_groups if registered
    uint64_t id = 0; // debug id
    LassoConfig lasso_config; // configuration flags and widget type for this meta-group
    int dangling_widget_id = -1; // RopeSim widget id if created
    // optional anchor override used by helpers (e.g., dangling widget attach)
    int anchor_rope = -1;
    int anchor_vert = -1;
    // optional overlay keys created by canvas-level helpers (0 == none)
    unsigned long long overlay_key_a = 0ull;
    unsigned long long overlay_key_b = 0ull;
    // optional FIFO and subgroup flags so meta-groups can behave like edges
    EdgeTensorFifo fifo;
    uint32_t subgroup_flags = 0;
    // channel group id (user-tunable integer controlling grouping of FIFOs/edges)
    int channel_group = 0;
    // if a dangling short-rope + widget was created, remember rope id and vertex
    int dangling_widget_rope = -1;
    int dangling_widget_rope_vid = -1;
    float dangling_hang_len = 0.0f;
    // ring topology mode: 0=ribbon (chain), 1=closed loop, 2=dense cross-links
    int ring_mode = 0;
};

// Thin adaptor type used to store meta-groups in the context vector.
struct GP_TableContext::GP_MetaGroupInternal : public GP_MetaGroup {};

extern "C" GP_MetaGroup* gp_table_meta_create(GP_TableContext* ctx) {
    if (!ctx) return nullptr;
    auto mg = std::make_unique<GP_TableContext::GP_MetaGroupInternal>();
    static uint64_t next_mg_id = 1;
    mg->id = next_mg_id++;
    // Initialize lasso_config to avoid uninitialized reads in canvas logic
    mg->lasso_config.flags = 0;
    mg->lasso_config.widget_type = 0;
    mg->lasso_config.reserved[0] = 0;
    mg->lasso_config.reserved[1] = 0;
    mg->lasso_config.reserved[2] = 0;
    mg->lasso_config.spring_min_rest = 0.0f;
    mg->lasso_config.spring_reduce_rate = 0.0f;
    mg->lasso_config.spring_mode = 0;
    mg->lasso_config.spring_reserved[0] = 0;
    mg->lasso_config.spring_reserved[1] = 0;
    mg->lasso_config.spring_reserved[2] = 0;
    GP_MetaGroup* ptr = mg.get();
    ctx->meta_groups.push_back(std::move(mg));
    printf("gp_table_meta_create: created mg=%p id=%llu on ctx=%p\n", (void*)ptr, (unsigned long long)ptr->id, (void*)ctx);
    return ptr;
}

// Debug helper: print a meta-group's vertices and lasso config for diagnostics.
extern "C" int32_t gp_table_debug_dump_meta_group(GP_TableContext* ctx, GP_MetaGroup* mg, const char* tag) {
    if (!ctx || !mg) return 0;
    if (!tag) tag = "dump";
    printf("gp_table_debug_dump_meta_group: [%s] mg=%p id=%llu sim_group_idx=%d confinement=%.3f ring_mode=%d dangling_rope=%d dangling_vid=%d\n",
           tag, (void*)mg, (unsigned long long)mg->id, mg->sim_group_idx, mg->confinement, mg->ring_mode, mg->dangling_widget_rope, mg->dangling_widget_rope_vid);
    printf("  lasso_config: flags=0x%08x widget=%u spring_min_rest=%.2f spring_reduce_rate=%.2f spring_mode=%u\n",
           mg->lasso_config.flags, static_cast<unsigned int>(mg->lasso_config.widget_type), mg->lasso_config.spring_min_rest, mg->lasso_config.spring_reduce_rate, static_cast<unsigned int>(mg->lasso_config.spring_mode));
    int vcount = static_cast<int>(mg->vertices.size());
    printf("  vertices.count=%d\n", vcount);
    for (int vi = 0; vi < vcount; ++vi) {
        const auto &mv = mg->vertices[static_cast<size_t>(vi)];
    printf("    [%d] rope_idx=%d vert_idx=%d rope_id=%llu u=%.6f\n",
               vi, mv.rope_idx, mv.vertex_idx, (unsigned long long)mv.rope_id, mv.u);
    }
    fflush(stdout);
    return 1;
}

static uint64_t gp_table_lookup_rope_id_for_index(GP_TableContext* ctx, int rope_idx) {
    if (!ctx || rope_idx < 0) return 0ull;
    if (static_cast<size_t>(rope_idx) < ctx->rope_ids.size()) {
        uint64_t rid = ctx->rope_ids[static_cast<size_t>(rope_idx)];
        if (rid != 0ull) return rid;
    }
    for (const auto &kv : ctx->rope_id_to_sim_idx) {
        if (kv.second == rope_idx) return kv.first;
    }
    return 0ull;
}

static uint64_t gp_table_ensure_rope_id_for_index(GP_TableContext* ctx, int rope_idx) {
    if (!ctx || rope_idx < 0) return 0ull;
    uint64_t rid = gp_table_lookup_rope_id_for_index(ctx, rope_idx);
    if (rid != 0ull) return rid;
    GP_CanvasContext* cvs = gp_canvas_get_singleton();
    rid = gp_canvas_generate_id(cvs, 0ull);
    if (rid != 0ull && rope_idx >= 0 && ctx) {
        ctx->rope_id_to_sim_idx[rid] = rope_idx;
    }
    return rid;
}

static bool gp_table_meta_vertex_exists(const GP_MetaGroup* mg, uint64_t rope_id, int rope_idx, int vertex_idx) {
    if (!mg) return false;
    for (const auto &mv : mg->vertices) {
        if (rope_id != 0ull) {
            if (mv.rope_id == rope_id && mv.vertex_idx == vertex_idx) return true;
        } else if (rope_idx >= 0) {
            if (mv.rope_idx == rope_idx && mv.vertex_idx == vertex_idx) return true;
        }
    }
    return false;
}

static int32_t gp_table_meta_add_vertex_with_id_internal(GP_TableContext* ctx, GP_MetaGroup* mg, uint64_t rope_id, int32_t rope_idx, int32_t vertex_idx, bool update_sim) {
    if (!ctx || !mg) return 0;
    if (gp_table_meta_vertex_exists(mg, rope_id, rope_idx, vertex_idx)) return 1;
    if (rope_id != 0ull && rope_idx >= 0) {
        ctx->rope_id_to_sim_idx[rope_id] = rope_idx;
        if (static_cast<size_t>(rope_idx) >= ctx->rope_ids.size()) {
            ctx->rope_ids.resize(static_cast<size_t>(rope_idx) + 1, 0ull);
        }
        if (ctx->rope_ids[static_cast<size_t>(rope_idx)] == 0ull) {
            ctx->rope_ids[static_cast<size_t>(rope_idx)] = rope_id;
        }
    }
    GP_MetaVertex mv{};
    mv.rope_id = rope_id;
    mv.rope_idx = static_cast<int>(rope_idx);
    mv.vertex_idx = static_cast<int>(vertex_idx);
    mv.u = -1.0f;
    mg->vertices.push_back(mv);
    RopeSim* sim = ctx->rope_sim;
    printf("gp_table_meta_add_vertex: mg=%p add (rope=%d,vert=%d) sim=%p sim_group_idx=%d\n", (void*)mg, rope_idx, vertex_idx, (void*)sim, mg->sim_group_idx);
    // If the table owns or is attached to a RopeSim, ensure a sim-level
    // meta group exists and register the vertex there so confinement
    // forces are applied during simulation.
    if (update_sim && sim && rope_idx >= 0) {
        if (mg->sim_group_idx < 0) {
            int sg = rope_sim_create_meta_group(sim, mg->confinement);
            if (sg >= 0) mg->sim_group_idx = sg;
        }
        if (mg->sim_group_idx >= 0) {
            if (!rope_sim_meta_group_has_member(sim, mg->sim_group_idx, static_cast<int>(rope_idx), static_cast<int>(vertex_idx))) {
                rope_sim_meta_group_add(sim, mg->sim_group_idx, static_cast<int>(rope_idx), static_cast<int>(vertex_idx));
            }
            int vc = rope_sim_get_vertex_count(sim, static_cast<int>(rope_idx));
            if (vc > 1) {
                float u = static_cast<float>(vertex_idx) / static_cast<float>(vc - 1);
                if (u < 0.0f) u = 0.0f;
                if (u > 1.0f) u = 1.0f;
                mg->vertices.back().u = u;
                rope_sim_meta_group_set_member_u(sim, mg->sim_group_idx, static_cast<int>(rope_idx), static_cast<int>(vertex_idx), u);
            }
            // update pressure in sim if different
            rope_sim_meta_group_set_pressure(sim, mg->sim_group_idx, mg->confinement);
        }
    }
    return 1;
}

static int32_t gp_table_meta_add_vertex_with_id(GP_TableContext* ctx, GP_MetaGroup* mg, uint64_t rope_id, int32_t rope_idx, int32_t vertex_idx) {
    return gp_table_meta_add_vertex_with_id_internal(ctx, mg, rope_id, rope_idx, vertex_idx, true);
}

static void gp_table_queue_pending_meta_vertex(GP_TableContext* ctx, GP_MetaGroup* mg, uint64_t rope_id, int vertex_idx) {
    if (!ctx || !mg || rope_id == 0ull) return;
    std::lock_guard<std::mutex> lk(ctx->pending_meta_mu);
    ctx->pending_meta_vertices.push_back({mg, rope_id, vertex_idx});
}

static void gp_table_apply_pending_meta_vertices(GP_TableContext* ctx) {
    if (!ctx || !ctx->rope_sim) return;
    std::vector<GP_TableContext::PendingMetaVertex> pending;
    {
        std::lock_guard<std::mutex> lk(ctx->pending_meta_mu);
        if (ctx->pending_meta_vertices.empty()) return;
        pending.swap(ctx->pending_meta_vertices);
    }
    GP_CanvasContext* cvs = gp_canvas_get_singleton();
    for (const auto &entry : pending) {
        if (!entry.mg || entry.rope_id == 0ull) continue;
        int resolved = -1;
        if (cvs) resolved = gp_canvas_resolve_rope_id_to_index(cvs, ctx, entry.rope_id);
        if (resolved < 0) resolved = gp_table_resolve_rope_id_to_sim_index(ctx, entry.rope_id);
        if (resolved < 0) {
            gp_table_queue_pending_meta_vertex(ctx, entry.mg, entry.rope_id, entry.vertex_idx);
            continue;
        }
        gp_table_meta_add_vertex_with_id(ctx, entry.mg, entry.rope_id, resolved, entry.vertex_idx);
    }
}

// Removed: gp_table_sim_toggle_meta_group_mode_for_rope

extern "C" int32_t gp_table_meta_get_ring_mode(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_mode) {
    if (!ctx || !mg || !out_mode) return 0;
    *out_mode = mg->ring_mode;
    return 1;
}

extern "C" int32_t gp_table_meta_get_confinement(const GP_TableContext* ctx, GP_MetaGroup* mg, float* out_conf) {
    if (!ctx || !mg || !out_conf) return 0;
    *out_conf = mg->confinement;
    return 1;
}
extern "C" int32_t gp_table_meta_set_confinement(GP_TableContext* ctx, GP_MetaGroup* mg, float conf) {
    if (!ctx || !mg) return 0;
    mg->confinement = conf;
    RopeSim* sim = ctx->rope_sim;
    if (mg->sim_group_idx >= 0 && sim) {
        rope_sim_meta_group_set_pressure(sim, mg->sim_group_idx, conf);
    }
    return 1;
}
extern "C" int32_t gp_table_meta_get_id(const GP_TableContext* ctx, GP_MetaGroup* mg, unsigned long long* out_id) {
    if (!ctx || !mg || !out_id) return 0;
    *out_id = mg->id;
    return 1;
}
extern "C" int32_t gp_table_meta_set_id(GP_TableContext* ctx, GP_MetaGroup* mg, unsigned long long id) {
    if (!ctx || !mg) return 0;
    mg->id = id;
    return 1;
}
extern "C" int32_t gp_table_meta_get_dangling_hang_len(const GP_TableContext* ctx, GP_MetaGroup* mg, float* out_len) {
    if (!ctx || !mg || !out_len) return 0;
    *out_len = mg->dangling_hang_len;
    return 1;
}
extern "C" int32_t gp_table_meta_set_dangling_hang_len(GP_TableContext* ctx, GP_MetaGroup* mg, float len) {
    if (!ctx || !mg) return 0;
    mg->dangling_hang_len = len;
    return 1;
}
extern "C" int32_t gp_table_meta_get_lasso_fields(const GP_TableContext* ctx, GP_MetaGroup* mg, unsigned int* out_flags, int32_t* out_widget_type) {
    if (!ctx || !mg || !out_flags || !out_widget_type) return 0;
    *out_flags = mg->lasso_config.flags;
    *out_widget_type = static_cast<int32_t>(mg->lasso_config.widget_type);
    return 1;
}
extern "C" int32_t gp_table_meta_set_lasso_fields(GP_TableContext* ctx, GP_MetaGroup* mg, unsigned int flags, int32_t widget_type) {
    if (!ctx || !mg) return 0;
    mg->lasso_config.flags = flags;
    mg->lasso_config.widget_type = static_cast<uint8_t>(widget_type & 0xFF);
    return 1;
}
extern "C" int32_t gp_table_meta_set_ring_mode(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t mode) {
    if (!ctx || !mg) return 0;
    mg->ring_mode = mode;
    return 1;
}

extern "C" int32_t gp_table_meta_destroy(GP_TableContext* ctx, GP_MetaGroup* mg) {
    if (!ctx || !mg) return 0;
    for (size_t i = 0; i < ctx->meta_groups.size(); ++i) {
        if (ctx->meta_groups[i].get() == mg) {
            // If the table has a sim attached and the meta-group registered
            // a sim_group_idx, ensure sim-side resources (springs, meta-group)
            // are cleaned up before removing the meta-group record.
            RopeSim* sim = ctx->rope_sim;
            int sim_idx = mg->sim_group_idx;
            if (sim && sim_idx >= 0) {
                // disable any edge-springs associated with this meta-group
                rope_sim_meta_group_disable_edge_springs(sim, sim_idx);
                rope_sim_destroy_meta_group(sim, sim_idx);
            }
            ctx->meta_groups.erase(ctx->meta_groups.begin() + static_cast<ptrdiff_t>(i));
            return 1;
        }
    }
    return 0;
}

extern "C" int32_t gp_table_meta_enable_edge_springs(GP_TableContext* ctx, GP_MetaGroup* mg, float min_rest, float reduce_rate) {
    if (!ctx || !mg) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim || mg->sim_group_idx < 0) return 0;
    // respect the meta-group's requested topology before wiring springs
    rope_sim_meta_group_set_mode(sim, mg->sim_group_idx, mg->ring_mode);
    int res = rope_sim_meta_group_enable_edge_springs(sim, mg->sim_group_idx, min_rest, reduce_rate);
    if (!res) return 0;
    // For lasso-style dense networks, leave rope rest lengths alone; the
    // meta-group springs handle contraction without collapsing the ropes.
    if (mg->ring_mode != 2) {
        float extend_amount = 5.0f; // increase rest length immediately
        float shorten_factor = 0.6f; // target = current * factor
        float shorten_rate = 1.0f; // units per second
        float shorten_delay = 0.5f; // seconds before shortening begins
        std::unordered_set<int> handled;
        for (const auto &p : mg->vertices) {
            int r = p.rope_idx;
            if (handled.find(r) != handled.end()) continue;
            handled.insert(r);
            rope_sim_modify_rest_length(sim, r, extend_amount);
            float cur = 0.0f;
            if (rope_sim_get_rope_rest_length(sim, r, &cur)) {
                float target = std::max(0.0001f, cur * shorten_factor);
                rope_sim_set_rope_rest_target(sim, r, target, shorten_rate, shorten_delay);
                printf("gp_table_meta_enable_edge_springs: rope %d rest increased by %.2f then scheduled target %.2f (delay=%.2f rate=%.2f)\n", r, extend_amount, target, shorten_delay, shorten_rate);
            }
        }
    } else if (std::getenv("NODUS_DEBUG_META")) {
        printf("gp_table_meta_enable_edge_springs: ring_mode=2 skipping rope rest-length adjustment\n");
    }
    return 1;
}

extern "C" int32_t gp_table_meta_disable_edge_springs(GP_TableContext* ctx, GP_MetaGroup* mg) {
    if (!ctx || !mg) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim || mg->sim_group_idx < 0) return 0;
    return rope_sim_meta_group_disable_edge_springs(sim, mg->sim_group_idx);
}

extern "C" int32_t gp_table_rope_modify_rest_length(GP_TableContext* ctx, int32_t rope_idx, float delta) {
    if (!ctx) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_modify_rest_length(sim, static_cast<int>(rope_idx), delta);
}

extern "C" int32_t gp_table_rope_set_rest_target(GP_TableContext* ctx, int32_t rope_idx, float target_rest, float rate, float delay) {
    if (!ctx) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_set_rope_rest_target(sim, static_cast<int>(rope_idx), target_rest, rate, delay);
}

extern "C" int32_t gp_table_rope_get_rest_length(GP_TableContext* ctx, int32_t rope_idx, float* out_rest) {
    if (!ctx || !out_rest) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_get_rope_rest_length(sim, static_cast<int>(rope_idx), out_rest);
}

extern "C" int32_t gp_table_rope_set_radius(GP_TableContext* ctx, int32_t rope_idx, float radius) {
    if (!ctx) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_set_rope_radius(sim, static_cast<int>(rope_idx), radius);
}

extern "C" int32_t gp_table_rope_get_radius(GP_TableContext* ctx, int32_t rope_idx, float* out_radius) {
    if (!ctx || !out_radius) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_get_rope_radius(sim, static_cast<int>(rope_idx), out_radius);
}

extern "C" int32_t gp_table_rope_insert_vertex(GP_TableContext* ctx, int32_t rope_idx, int32_t seg_index, float t) {
    if (!ctx) return -1;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return -1;
    return rope_sim_insert_vertex(sim, static_cast<int>(rope_idx), static_cast<int>(seg_index), t);
}
extern "C" int32_t gp_table_sim_add_meta_group_for_rope(GP_TableContext* ctx, int32_t rope_idx) {
    if (!ctx) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    int vc = rope_sim_get_vertex_count(sim, static_cast<int>(rope_idx));
    if (vc <= 1) return 0;
    int sg = rope_sim_create_meta_group(sim, 1.0f);
    if (sg < 0) return 0;
    // add first and last vertex as members so edge-springs can be created
    rope_sim_meta_group_add(sim, sg, static_cast<int>(rope_idx), 0);
    rope_sim_meta_group_add(sim, sg, static_cast<int>(rope_idx), vc - 1);
    rope_sim_meta_group_set_mode(sim, sg, 0);
    int res = rope_sim_meta_group_enable_edge_springs(sim, sg, 2.0f, 50.0f);
    printf("gp_table_sim_add_meta_group_for_rope: created sg=%d for rope=%d res=%d\n", sg, rope_idx, res);
    return res ? 1 : 0;
}

extern "C" int32_t gp_table_create_ring(GP_TableContext* ctx, int32_t rope_idx, float u) {
    if (!ctx) return -1;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return -1;
    // Create a lightweight ring at parameter u; no overlays or dangling widgets.
    return rope_sim_create_ring(sim, static_cast<int>(rope_idx), u);
}

// Create ring by persisted rope id. Resolve id to runtime rope index using
// canvas mapping then table-local fallback, and call gp_table_create_ring.
extern "C" int32_t gp_table_create_ring_by_id(GP_TableContext* ctx, uint64_t rope_id, float u) {
    if (!ctx || rope_id == 0ull) return -1;
    GP_CanvasContext* cvs = gp_canvas_get_singleton();
    int resolved = -1;
    if (cvs) resolved = gp_canvas_resolve_rope_id_to_index(cvs, ctx, rope_id);
    if (resolved < 0) {
        // fallback to table-local id->sim mapping
        for (size_t ri = 0; ri < ctx->rope_ids.size(); ++ri) {
            if (ctx->rope_ids[ri] == rope_id) {
                auto it = ctx->rope_id_to_sim_idx.find(rope_id);
                if (it != ctx->rope_id_to_sim_idx.end()) resolved = it->second;
                else resolved = -1;
                break;
            }
        }
    }
    if (resolved < 0) return -2; // distinct code for id->index resolution failure
    return gp_table_create_ring(ctx, resolved, u);
}

extern "C" int32_t gp_table_destroy_ring(GP_TableContext* ctx, int32_t ring_id) {
    if (!ctx) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_destroy_ring(sim, static_cast<int>(ring_id));
}

extern "C" int32_t gp_table_set_ring_target(GP_TableContext* ctx, int32_t ring_id, float target_u, float speed) {
    if (!ctx) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_set_ring_target(sim, static_cast<int>(ring_id), target_u, speed);
}

extern "C" int32_t gp_table_get_ring_u(GP_TableContext* ctx, int32_t ring_id, float* out_u) {
    if (!ctx || !out_u) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_get_ring_u(sim, static_cast<int>(ring_id), out_u);
}

extern "C" int32_t gp_table_meta_add_vertex(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t rope_idx, int32_t vertex_idx) {
    if (!ctx || !mg) return 0;
    uint64_t rope_id = gp_table_ensure_rope_id_for_index(ctx, rope_idx);
    return gp_table_meta_add_vertex_with_id(ctx, mg, rope_id, rope_idx, vertex_idx);
}

extern "C" int32_t gp_table_meta_set_anchor(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t rope_idx, int32_t vertex_idx) {
    if (!ctx || !mg) return 0;
    mg->anchor_rope = rope_idx;
    mg->anchor_vert = vertex_idx;
    return 1;
}

extern "C" int32_t gp_table_meta_get_anchor(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_rope_idx, int32_t* out_vertex_idx) {
    if (!ctx || !mg) return 0;
    if (out_rope_idx) *out_rope_idx = mg->anchor_rope;
    if (out_vertex_idx) *out_vertex_idx = mg->anchor_vert;
    return 1;
}

extern "C" int32_t gp_table_meta_get_vertex_count(GP_TableContext* ctx, GP_MetaGroup* mg) {
    if (!ctx || !mg) return 0;
    return static_cast<int32_t>(mg->vertices.size());
}

extern "C" int32_t gp_table_get_meta_group_count(const GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int32_t>(ctx->meta_groups.size());
}

extern "C" GP_MetaGroup* gp_table_get_meta_group(GP_TableContext* ctx, int32_t idx) {
    if (!ctx) return nullptr;
    if (idx < 0 || static_cast<size_t>(idx) >= ctx->meta_groups.size()) return nullptr;
    return ctx->meta_groups[static_cast<size_t>(idx)].get();
}

extern "C" int32_t gp_table_meta_get_vertex(const GP_TableContext* ctx, GP_MetaGroup* mg, int32_t idx, int32_t* out_rope_idx, int32_t* out_vertex_idx) {
    if (!ctx || !mg) return 0;
    if (idx < 0 || static_cast<size_t>(idx) >= mg->vertices.size()) return 0;
    auto &mv = mg->vertices[static_cast<size_t>(idx)];
    if (out_rope_idx) *out_rope_idx = mv.rope_idx;
    if (out_vertex_idx) *out_vertex_idx = mv.vertex_idx;
    return 1;
}

extern "C" int32_t gp_table_meta_get_vertex_u(const GP_TableContext* ctx, GP_MetaGroup* mg, int32_t idx, float* out_u) {
    if (!ctx || !mg || !out_u) return 0;
    if (idx < 0 || static_cast<size_t>(idx) >= mg->vertices.size()) return 0;
    *out_u = mg->vertices[static_cast<size_t>(idx)].u;
    return 1;
}

extern "C" int32_t gp_table_meta_set_vertex_u(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t rope_idx, int32_t vertex_idx, float u) {
    if (!ctx || !mg) return 0;
    if (u < 0.0f) u = 0.0f;
    if (u > 1.0f) u = 1.0f;
    for (auto &mv : mg->vertices) {
        if (mv.rope_idx == rope_idx && mv.vertex_idx == vertex_idx) {
            mv.u = u;
            RopeSim* sim = ctx->rope_sim;
            if (sim && mg->sim_group_idx >= 0) {
                rope_sim_meta_group_set_member_u(sim, mg->sim_group_idx, rope_idx, vertex_idx, u);
            }
            return 1;
        }
    }
    return 0;
}

extern "C" int32_t gp_table_get_widget_position(GP_TableContext* ctx, int32_t widget_id, float* out_xyz) {
    if (!ctx || !out_xyz) return 0;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    return rope_sim_get_widget_position(sim, static_cast<int>(widget_id), out_xyz);
}

extern "C" int32_t gp_table_meta_get_dangling_widget_id(const GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_widget_id) {
    if (!ctx || !mg || !out_widget_id) return 0;
    *out_widget_id = mg->dangling_widget_id;
    return 1;
}

extern "C" int32_t gp_table_meta_set_lasso_config(GP_TableContext* ctx, GP_MetaGroup* mg, const LassoConfig* cfg) {
    if (!ctx || !mg) return 0;
    if (cfg) {
        mg->lasso_config.flags = cfg->flags;
        mg->lasso_config.widget_type = cfg->widget_type;
        mg->lasso_config.spring_min_rest = cfg->spring_min_rest;
        mg->lasso_config.spring_reduce_rate = cfg->spring_reduce_rate;
        mg->lasso_config.spring_mode = cfg->spring_mode;
    } else {
        mg->lasso_config.flags = 0;
        mg->lasso_config.widget_type = 0;
    }
    return 1;
}

extern "C" int32_t gp_table_meta_get_lasso_config(GP_TableContext* ctx, GP_MetaGroup* mg, LassoConfig* out_cfg) {
    if (!ctx || !mg || !out_cfg) return 0;
    out_cfg->flags = mg->lasso_config.flags;
    out_cfg->widget_type = mg->lasso_config.widget_type;
    out_cfg->reserved[0] = mg->lasso_config.reserved[0];
    out_cfg->reserved[1] = mg->lasso_config.reserved[1];
    out_cfg->reserved[2] = mg->lasso_config.reserved[2];
    out_cfg->spring_min_rest = mg->lasso_config.spring_min_rest;
    out_cfg->spring_reduce_rate = mg->lasso_config.spring_reduce_rate;
    out_cfg->spring_mode = mg->lasso_config.spring_mode;
    out_cfg->spring_reserved[0] = mg->lasso_config.spring_reserved[0];
    out_cfg->spring_reserved[1] = mg->lasso_config.spring_reserved[1];
    out_cfg->spring_reserved[2] = mg->lasso_config.spring_reserved[2];
    return 1;
}

extern "C" int32_t gp_table_meta_set_edge_spring_params(GP_TableContext* ctx, GP_MetaGroup* mg, float min_rest, float reduce_rate, int32_t mode) {
    if (!ctx || !mg) return 0;
    mg->lasso_config.spring_min_rest = min_rest;
    mg->lasso_config.spring_reduce_rate = reduce_rate;
    mg->lasso_config.spring_mode = static_cast<uint8_t>(mode & 0xFF);
    return 1;
}

extern "C" int32_t gp_table_meta_get_subgroup_flags(GP_TableContext* ctx, GP_MetaGroup* mg, uint32_t* out_flags) {
    if (!ctx || !mg || !out_flags) return 0;
    // Prefer explicit subgroup_flags stored on the meta-group, otherwise fall
    // back to any lasso-config flags the creator supplied.
    if (mg->subgroup_flags != 0u) {
        *out_flags = mg->subgroup_flags;
        return 1;
    }
    *out_flags = mg->lasso_config.flags;
    return 1;
}

extern "C" int32_t gp_table_meta_get_dangling_rope_info(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_rope_idx, int32_t* out_vertex_idx) {
    if (!ctx || !mg || !out_rope_idx || !out_vertex_idx) return 0;
    *out_rope_idx = mg->dangling_widget_rope;
    *out_vertex_idx = mg->dangling_widget_rope_vid;
    return 1;
}

extern "C" int32_t gp_table_meta_set_channel_group(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t channel_group) {
    if (!ctx || !mg) return 0;
    mg->channel_group = channel_group;
    (void)ctx; (void)mg; (void)channel_group;
    return 1;
}

extern "C" int32_t gp_table_meta_get_channel_group(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_channel_group) {
    if (!ctx || !mg || !out_channel_group) return 0;
    *out_channel_group = mg->channel_group;
    (void)ctx; (void)mg; (void)out_channel_group;
    return 1;
}

extern "C" int32_t gp_table_meta_get_fifo_snapshot(GP_TableContext* ctx, GP_MetaGroup* mg, float* out_buf, int32_t out_len) {
    if (!ctx || !mg || !out_buf || out_len <= 0) return 0;
    // Collect a simple float-based snapshot of key fifo fields.
    int32_t pos = 0;
    auto pushf = [&](float v)->bool { if (pos >= out_len) return false; out_buf[pos++] = v; return true; };
    // id, vertex count, channel_group
    pushf(static_cast<float>(static_cast<double>(mg->id)));
    pushf(static_cast<float>(static_cast<int>(mg->vertices.size())));
    pushf(static_cast<float>(mg->channel_group));
    // FIFO internals (best-effort atomic reads)
    if (!mg) return pos;
    EdgeTensorFifo &ef = mg->fifo;
    size_t stride = ef.impl ? ef.impl->stride : 0;
    size_t slots = ef.impl ? ef.impl->slots : 0;
    size_t topk = ef.impl ? ef.impl->top_k : 0;
    uint64_t write_seq = ef.impl ? ef.impl->write_seq.load(std::memory_order_relaxed) : 0ull;
    uint64_t writer = ef.impl ? ef.impl->writer.load(std::memory_order_relaxed) : 0ull;
    uint64_t last_write_seq = ef.impl ? ef.impl->last_write_seq.load(std::memory_order_relaxed) : 0ull;
    uint64_t last_read_seq = ef.impl ? ef.impl->last_read_seq.load(std::memory_order_relaxed) : 0ull;
    float write_friction = ef.impl ? ef.impl->write_friction.load(std::memory_order_relaxed) : 0.0f;
    float read_friction = ef.impl ? ef.impl->read_friction.load(std::memory_order_relaxed) : 0.0f;
    int32_t last_write_region = ef.impl ? ef.impl->last_write_region.load(std::memory_order_relaxed) : -1;
    int32_t last_read_region = ef.impl ? ef.impl->last_read_region.load(std::memory_order_relaxed) : -1;
    float write_phase = ef.impl ? ef.impl->write_phase.load(std::memory_order_relaxed) : 0.0f;
    float read_phase = ef.impl ? ef.impl->read_phase.load(std::memory_order_relaxed) : 0.0f;
    int32_t friction_regions = ef.impl ? ef.impl->friction_regions.load(std::memory_order_relaxed) : 0;
    bool configured = ef.impl ? ef.impl->configured : false;
    pushf(static_cast<float>(stride));
    pushf(static_cast<float>(slots));
    pushf(static_cast<float>(topk));
    // sequence numbers may exceed float precision; low 32 bits are kept.
    pushf(static_cast<float>(static_cast<uint32_t>(write_seq & 0xFFFFFFFFu)));
    pushf(static_cast<float>(static_cast<uint32_t>(writer & 0xFFFFFFFFu)));
    pushf(static_cast<float>(static_cast<uint32_t>(last_write_seq & 0xFFFFFFFFu)));
    pushf(static_cast<float>(static_cast<uint32_t>(last_read_seq & 0xFFFFFFFFu)));
    pushf(write_friction);
    pushf(read_friction);
    pushf(static_cast<float>(last_write_region));
    pushf(static_cast<float>(last_read_region));
    pushf(write_phase);
    pushf(read_phase);
    pushf(static_cast<float>(friction_regions));
    pushf(configured ? 1.0f : 0.0f);
    // shape
    pushf(static_cast<float>(static_cast<int>(ef.shape.size())));
    for (size_t i = 0; i < ef.shape.size(); ++i) {
        pushf(static_cast<float>(ef.shape[i]));
    }
    return pos;
}

int32_t gp_table_meta_get_sim_group_index(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_sim_idx) {
    if (!ctx || !mg || !out_sim_idx) return 0;
    *out_sim_idx = mg->sim_group_idx;
    return 1;
}

extern "C" int32_t gp_table_meta_create_widget(GP_TableContext* ctx, GP_MetaGroup* mg) {
    if (!ctx || !mg) return 0;
    // Strip down: no dangling widget or overlays. Clear prior hooks and signal success.
    mg->dangling_widget_id = -1;
    mg->dangling_widget_rope = -1;
    mg->dangling_widget_rope_vid = -1;
    mg->dangling_hang_len = 0.0f;
    mg->overlay_key_a = 0ull;
    mg->overlay_key_b = 0ull;
    try { mg->fifo.configure_default(); } catch(...) {}
    mg->subgroup_flags = mg->lasso_config.flags;
    return 1;
}

extern "C" int32_t gp_table_meta_destroy_widget(GP_TableContext* ctx, GP_MetaGroup* mg) {
    if (!ctx || !mg) return 0;
    if (mg->dangling_widget_id < 0) return 1;
    RopeSim* sim = ctx->rope_sim;
    if (!sim) return 0;
    int res = rope_sim_destroy_dangling_widget(sim, mg->dangling_widget_id);
    mg->dangling_widget_id = -1;
    return res;
}

extern "C" int32_t gp_table_meta_set_overlay_keys(GP_TableContext* ctx, GP_MetaGroup* mg, unsigned long long key_a, unsigned long long key_b) {
    if (!ctx || !mg) return 0;
    mg->overlay_key_a = key_a;
    mg->overlay_key_b = key_b;
    return 1;
}

extern "C" int32_t gp_table_meta_get_overlay_keys(GP_TableContext* ctx, GP_MetaGroup* mg, unsigned long long* out_key_a, unsigned long long* out_key_b) {
    if (!ctx || !mg || (!out_key_a && !out_key_b)) return 0;
    if (out_key_a) *out_key_a = mg->overlay_key_a;
    if (out_key_b) *out_key_b = mg->overlay_key_b;
    return 1;
}

// Attach/detach an external RopeSim instance to the table context.
// If `sim` is non-null the table will use that simulator for all rope
// allocations and updates. `take_ownership` indicates whether the table
// should destroy the provided simulator when the table is destroyed
// (1 = table destroys the sim, 0 = caller retains ownership). Passing
// `sim == NULL` detaches any external simulator; the table may create
// its own simulator later on demand. Returns 1 on success.
int32_t gp_table_attach_rope_sim(GP_TableContext* ctx, RopeSim* sim, int32_t take_ownership) {
    if (!ctx) return 0;
    // If the table already owns a restored simulator, keep it instead of
    // overwriting with a shared/root simulator attachment.
    if (ctx->rope_sim && ctx->rope_sim_owned && sim && !take_ownership) {
        return 1;
    }
    // If we currently own a sim, destroy it first
    if (ctx->rope_sim && ctx->rope_sim_owned) {
        rope_sim_destroy(ctx->rope_sim);
    }
    ctx->rope_sim = sim;
    ctx->rope_sim_owned = (sim != nullptr) ? (take_ownership ? 1 : 0) : 0;
    // reset mapping so edges will create ropes in the new sim when next rendered
    ctx->rope_id_to_sim_idx.clear();
    // If attaching a simulator, create rope entries for any existing edges
    // that lack sim indices or persistent uids so we have stable mapping.
    if (ctx->rope_sim) {
        // Prepare row layout helpers
        std::vector<int> row_y0;
        std::vector<int> row_h;
        compute_row_layout(ctx->rows.data(), static_cast<int>(ctx->rows.size()), ctx->st, ctx->geom.height_px, row_y0, row_h);
        auto compute_center_local = [&](uint64_t key, int &outx, int &outy) {
            outx = -1; outy = -1;
            uint32_t r_orig = static_cast<uint32_t>(key >> 32);
            uint32_t c_idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
            uint32_t led = static_cast<uint32_t>(key & 0xFFFFu);
            if (r_orig >= ctx->rows.size()) return;
            const GP_TableRow &row = ctx->rows[static_cast<size_t>(r_orig)];
            if (static_cast<int>(c_idx) < 0 || static_cast<int>(c_idx) >= row.cell_count) return;
            int col_x0[8] = {0}; int col_w[8] = {0};
            compute_columns(ctx->cols.data(), static_cast<int>(ctx->cols.size()), ctx->st.w, ctx->st.name_w, col_x0, col_w);
            const GP_TableCell &cell = row.cells[static_cast<int>(c_idx)];
            int x0 = col_x0[static_cast<int>(c_idx)];
            int cw = col_w[static_cast<int>(c_idx)];
            int y0 = (r_orig < row_y0.size()) ? row_y0[static_cast<size_t>(r_orig)] : (static_cast<int>(r_orig) * ctx->st.row_h);
            int rh = (r_orig < row_h.size()) ? row_h[static_cast<size_t>(r_orig)] : ctx->st.row_h;
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
                outx = cx0 + static_cast<int>(led) * led_spacing;
                const bool is_image_row = row_find_image_cell(row, nullptr);
                int band_h = is_image_row ? std::min(rh, row_image_header_h(ctx->st, row)) : rh;
                outy = y0 + band_h / 2;
            }
        };

        // Ensure rope_ids vector matches edge count
        if (ctx->rope_ids.size() < ctx->edges.size()) ctx->rope_ids.resize(ctx->edges.size(), 0ull);

        for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
            uint64_t existing_id = (ei < ctx->rope_ids.size()) ? ctx->rope_ids[ei] : 0ull;
            if (existing_id != 0ull && ctx->rope_id_to_sim_idx.find(existing_id) != ctx->rope_id_to_sim_idx.end()) continue; // already has rope mapped
            unsigned long long a = ctx->edges[ei].first;
            unsigned long long b = ctx->edges[ei].second;
            int ax = 0, ay = 0, bx = 0, by = 0;
            compute_center_local(a, ax, ay);
            compute_center_local(b, bx, by);
            if (ax == bx && ay == by) {
                // Guard against degenerate endpoints: derive a simple offset from contact ids.
                auto decode = [](uint64_t key, int &row, int &col, int &led) {
                    row = static_cast<int>(static_cast<uint32_t>(key >> 32));
                    col = static_cast<int>((static_cast<uint32_t>(key >> 16)) & 0xFFFFu);
                    led = static_cast<int>(static_cast<uint32_t>(key & 0xFFFFu));
                };
                int ra = 0, ca = 0, la = 0;
                int rb = 0, cb = 0, lb = 0;
                decode(a, ra, ca, la);
                decode(b, rb, cb, lb);
                int row_h = ctx->st.row_h > 0 ? ctx->st.row_h : 22;
                int dx = (cb - ca) * 16 + (lb - la) * 8;
                int dy = (rb - ra) * std::max(10, row_h / 2);
                if (dx == 0 && dy == 0) dx = 24; // last resort nudge
                bx += dx;
                by += dy;
                printf("gp_table_attach_rope_sim: adjusted degenerate endpoints a_row=%d b_row=%d dx=%d dy=%d -> (%d,%d)->(%d,%d)\n",
                    ra, rb, dx, dy, ax, ay, bx, by);
            }
            int segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : std::max(2, ctx->st.cable_segments);
            float slack = 0.0f;
            float plug_z = -ctx->st.cable_plug_depth;
            int idx = rope_sim_add_rope3(ctx->rope_sim, static_cast<float>(ax), static_cast<float>(ay), plug_z, static_cast<float>(bx), static_cast<float>(by), plug_z, segs, slack);
            uint64_t rid = 0ull;
            if (existing_id != 0ull) {
                rid = existing_id;
            } else {
                rid = gp_canvas_generate_id(gp_canvas_get_singleton(), 0ull);
            }
            ctx->rope_ids[ei] = rid;
            ctx->rope_id_to_sim_idx[rid] = idx;
            printf("gp_table_attach_rope_sim: ctx=%p backfilled rope idx=%d id=%llu\n", (void*)ctx, idx, (unsigned long long)rid);
            {
                GP_CanvasContext* cvs2 = gp_canvas_get_singleton();
                if (cvs2) {
                    uint64_t tmp2 = rid;
                    gp_canvas_register_table_rope_ids_from_array(cvs2, ctx, &tmp2, 1);
                }
            }
        }
    }
    if (GP_CanvasContext* cvs = gp_canvas_get_singleton()) {
        gp_canvas_mark_rope_map_dirty(cvs);
    }
    gp_table_apply_pending_meta_vertices(ctx);
    return 1;
}

