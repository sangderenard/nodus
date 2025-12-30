// Forward declare meta-group type so helpers can accept pointers without needing the full definition here.
#include "console_logger.h"
#ifndef printf
#define printf(...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
#ifndef fprintf
#define fprintf(file, ...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
struct GP_MetaGroup;
// Local copy of LassoConfig layout (header only forward-declares it).
typedef struct LassoConfig {
    uint32_t flags;
    uint8_t widget_type;
    uint8_t reserved[3];
    float spring_min_rest;
    float spring_reduce_rate;
    uint8_t spring_mode;
    uint8_t spring_reserved[3];
} LassoConfig;

// Forward declarations for helpers defined later.
static GP_TableContext* canvas_ensure_root_table(GP_CanvasContextImpl* ctx);
static RopeSim* canvas_root_sim(GP_CanvasContextImpl* ctx);
static RopeSim* canvas_require_root_sim(GP_CanvasContextImpl* ctx);
static uint64_t canvas_root_key_for_contact(int module_idx, int contact_idx);

// Canvas action IDs moved up so autobind helpers can reference them immediately.
enum CanvasActionId {
    CANVAS_ACT_SIM_SEGS_DEC = 2000,
    CANVAS_ACT_SIM_SEGS_INC = 2001,
    CANVAS_ACT_SIM_SLACK_DEC = 2002,
    CANVAS_ACT_SIM_SLACK_INC = 2003,
    CANVAS_ACT_SAVE = 2004,
    CANVAS_ACT_CLEAR = 2005,
    CANVAS_ACT_TOOL_CANVAS_0 = 2010,
    CANVAS_ACT_TOOL_CANVAS_1 = 2011,
    CANVAS_ACT_TOOL_CANVAS_2 = 2012,
    CANVAS_ACT_TOOL_CANVAS_3 = 2013,
    CANVAS_ACT_TOOL_EDGE_0 = 2014,
    CANVAS_ACT_TOOL_EDGE_1 = 2015,
    CANVAS_ACT_TOOL_EDGE_2 = 2016,
    CANVAS_ACT_TOOL_EDGE_3 = 2017,
    CANVAS_ACT_TOOL_EDGE_4 = 2018, // Meta-Edge Lasso (toolbar edge-group button)
    CANVAS_ACT_EDGE_ORDER_DEC = 2019,
    CANVAS_ACT_EDGE_ORDER_INC = 2020,
    CANVAS_ACT_EDGE_ORDER_TOOL = 2021,
    CANVAS_ACT_TOOL_TABLE_0 = 2030,
    CANVAS_ACT_TOOL_TABLE_1 = 2031,
    CANVAS_ACT_TOOL_TABLE_2 = 2032,
    CANVAS_ACT_IO_COUNT_DEC = 2040,
    CANVAS_ACT_IO_COUNT_INC = 2041,
    CANVAS_ACT_IO_CONSUMER_ADD = 2042,
    CANVAS_ACT_IO_PRODUCER_ADD = 2043,
    CANVAS_ACT_TABLE_TOOL_NUM_DEC = 2044,
    CANVAS_ACT_TABLE_TOOL_NUM_INC = 2045,
    CANVAS_ACT_META_CHAN_DEC = 2500,
    CANVAS_ACT_META_CHAN_INC = 2501,
    CANVAS_ACT_MODULE_LED = 2050,
    CANVAS_ACT_FRAME_PAIRS_DEC = 2200,
    CANVAS_ACT_FRAME_PAIRS_INC = 2201,
    CANVAS_ACT_TOOL_KPN_0 = 2060,
    CANVAS_ACT_TOOL_KPN_1 = 2061,
    CANVAS_ACT_TOOL_KPN_2 = 2062,
    CANVAS_ACT_THREAD_TOGGLE = 2070,
    CANVAS_ACT_TIMING_TOGGLE = 2089,
    CANVAS_ACT_KPN_GLOBAL_TOGGLE = 2079,
    CANVAS_ACT_THREAD_SIM_TOGGLE = 2078,
    CANVAS_ACT_CANVAS_ROOT_SIM_TOGGLE = 2088,
    CANVAS_ACT_THREAD_DELAY_DEC = 2071,
    CANVAS_ACT_THREAD_DELAY_INC = 2072,
    CANVAS_ACT_MODULE_CLONE = 2073,
    CANVAS_ACT_MODULE_CLEAR = 2074,
    CANVAS_ACT_MODULE_DESTROY = 2075,
    CANVAS_ACT_MODULE_EXPORT = 2076,
    CANVAS_ACT_MODULE_COMMIT = 2077,
    CANVAS_ACT_TOOL_SUBGROUP_0 = 2080,
    CANVAS_ACT_TOOL_SUBGROUP_1 = 2081,
    CANVAS_ACT_TOOL_SUBGROUP_2 = 2082,
    CANVAS_ACT_TOOL_SUBGROUP_3 = 2083,
    CANVAS_ACT_TOOL_SUBGROUP_4 = 2084,
    CANVAS_ACT_TOOL_SUBGROUP_5 = 2085,
    CANVAS_ACT_TOOL_SUBGROUP_6 = 2086,
    CANVAS_ACT_TOOL_SUBGROUP_7 = 2087,
    CANVAS_ACT_SPAWN_ROOT = 2090, // explicit spawn-root toolbar button
    // Execution mode toolbar actions (per-module or global override)
    CANVAS_ACT_EXEC_MODE_SEQ = 2091,
    CANVAS_ACT_EXEC_MODE_POOLED = 2092,
    CANVAS_ACT_EXEC_MODE_SLIP = 2093,
    CANVAS_ACT_EXEC_MODE_FREE = 2094,
    // Per-module execution mode actions (module-local buttons)
    CANVAS_ACT_EXEC_MODE_MODULE_SEQ = 2095,
    CANVAS_ACT_EXEC_MODE_MODULE_POOLED = 2096,
    CANVAS_ACT_EXEC_MODE_MODULE_SLIP = 2097,
    CANVAS_ACT_EXEC_MODE_MODULE_FREE = 2098,
    CANVAS_ACT_MENU_TOOL_ADD = 2101,
    CANVAS_ACT_MENU_TOOL_SUB = 2102,
    CANVAS_ACT_MENU_TOOL_MUL = 2103,
    CANVAS_ACT_MENU_TOOL_DIV = 2104,
    CANVAS_ACT_MENU_TOOL_MOD = 2105,
    CANVAS_ACT_MENU_TOOL_KEYBOARD = 2106,
    CANVAS_ACT_MENU_TOOL_MOUSE = 2107,
    CANVAS_ACT_MENU_TOOL_STACK = 2108,
    CANVAS_ACT_MENU_TOOL_CLONE = 2109,
    CANVAS_ACT_MENU_TOOL_RECT = 2110,
    CANVAS_ACT_MENU_TOOL_NUMBER = 2111,
    CANVAS_ACT_MENU_TOOL_ALLOCATOR = 2115,
    CANVAS_ACT_ROPE_MENU_TOGGLE = 2112,
    CANVAS_ACT_ROPE_MODE_SIMPLE = 2113,
    CANVAS_ACT_ROPE_MODE_FULL = 2114,
};

// Bind up to `max_ports` currently-empty receive frame ports on `module_idx`.
static int canvas_autobind_action_ports(GP_CanvasContext* ctx_, int module_idx, int action_id, int max_ports) {
    if (!ctx_ || module_idx < 0 || max_ports <= 0) return 0;
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (module_idx >= static_cast<int>(c->module_frame_links.size())) return 0;
    int bound = 0;
    const int rows_to_try[2] = {2, 3};
    // If caller requested the menu-tool for mouse, bind the concrete mouse
    // action group (down/up/move/scroll up/scroll down) so each event type
    // has its own distinct module-frame port.
    int bind_action_id = action_id;
    if (action_id == CANVAS_ACT_MENU_TOOL_MOUSE) bind_action_id = CANVAS_ACT_MOUSE_DOWN;
    for (int row_idx : rows_to_try) {
        for (int li = 0; li < kModuleExtraLedCount && bound < max_ports; ++li) {
            void* existing = c->module_frame_links[static_cast<size_t>(module_idx)].ptrs[static_cast<size_t>(row_idx)][static_cast<size_t>(li)];
            if (existing) continue;
            int col = (row_idx == 2) ? 0 : 1;
            if (gp_canvas_bind_action_enum_to_module_port(ctx_, module_idx, /*is_send=*/0, col, li, bind_action_id)) {
                void* now_bound = c->module_frame_links[static_cast<size_t>(module_idx)].ptrs[static_cast<size_t>(row_idx)][static_cast<size_t>(li)];
                if (now_bound) {
                    // If the bind created a group (e.g., mouse group), we may have
                    // consumed multiple LED slots; count how many new non-null
                    // pointers exist at this column starting at `li`.
                    int newly = 0;
                    for (int k = li; k < kModuleExtraLedCount && newly + bound < max_ports; ++k) {
                        void* check = c->module_frame_links[static_cast<size_t>(module_idx)].ptrs[static_cast<size_t>(row_idx)][static_cast<size_t>(k)];
                        if (check) ++newly; else break;
                    }
                    bound += newly;
                    // Advance li past the newly bound slots so we don't rebind them
                    li += std::max(0, newly - 1);
                }
            }
        }
        if (bound >= max_ports) break;
    }
    return bound;
}

extern "C" int gp_canvas_autobind_actions(GP_CanvasContext* ctx, int module_idx, const int32_t* action_ids, int action_count, int max_ports) {
    if (!ctx || !action_ids || action_count <= 0) return 0;
    int total_bound = 0;
    for (int i = 0; i < action_count && total_bound < max_ports; ++i) {
        int remaining = max_ports - total_bound;
        total_bound += canvas_autobind_action_ports(ctx, module_idx, action_ids[i], remaining);
    }
    return total_bound;
}

extern "C" int gp_canvas_register_table_rope_ids_from_array(GP_CanvasContext* ctx, GP_TableContext* table, const uint64_t* ids, int count) {
    if (!ctx || !ids || count <= 0) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    printf("gp_canvas_register_table_rope_ids_from_array: registering %d ids for table %p\n", count, (void*)table);
    for (int i = 0; i < count; ++i) {
        uint64_t id = ids[static_cast<size_t>(i)];
        if (id == 0ull) continue;
        c->rope_id_map[id] = RopeIdEntry{table, id};
        printf("  registered rope_id=%llu -> table=%p\n", (unsigned long long)id, (void*)table);
    }
    // Try to attach registered IDs to any canvas edges that map to this table.
    for (size_t ei = 0; ei < c->edges.size(); ++ei) {
        const auto &e = c->edges[ei];
        uint64_t ka = (static_cast<uint64_t>(static_cast<uint32_t>(e.desc.a_module)) << 32) |
                      (static_cast<uint64_t>(static_cast<uint32_t>(e.desc.a_contact_idx)) << 16) |
                      static_cast<uint64_t>(0);
        uint64_t kb = (static_cast<uint64_t>(static_cast<uint32_t>(e.desc.b_module)) << 32) |
                      (static_cast<uint64_t>(static_cast<uint32_t>(e.desc.b_contact_idx)) << 16) |
                      static_cast<uint64_t>(0);
        int edge_idx = -1;
        if (!gp_table_edge_index_for_pair(table, ka, kb, &edge_idx)) continue;
        if (edge_idx < 0 || edge_idx >= count) continue;
        uint64_t mapped_id = ids[static_cast<size_t>(edge_idx)];
        if (mapped_id == 0ull) continue;
        c->edges[ei].rope_uid = mapped_id; // keep existing edge field name
        // refresh rope map entry to ensure table association is recorded
        c->rope_id_map[mapped_id] = RopeIdEntry{table, mapped_id};
        printf("  bound canvas.edge[%zu] -> rope_id=%llu (table_edge=%d)\n", ei, (unsigned long long)mapped_id, edge_idx);
    }
    return 1;
}

// Central ID generation API. Callers should use this instead of
// incrementing per-field counters so the canvas can provide a uniform
// ID scheme via `gp_canvas_set_id_hook`.
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

extern "C" uint64_t gp_canvas_generate_id(GP_CanvasContext* ctx, uint64_t hint) {
    // Accept null ctx: use a global fallback counter so callers don't need
    // to special-case canvas availability. Also honor a pluggable id_hook
    // when a canvas is present.
    static std::atomic<uint64_t> s_fallback_counter{1};
    if (ctx) {
        GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
        if (c->id_hook) return c->id_hook(ctx, hint);
        uint64_t count = c->next_id_counter.fetch_add(1);
        uint64_t seed = (hint * 0x9e3779b97f4a7c15ULL) ^ count ^ reinterpret_cast<uint64_t>(ctx);
        uint64_t out = splitmix64(seed);
        if (out == 0ull) out = splitmix64(seed ^ 0xdeadbeefcafebabeULL);
        return out;
    } else {
        uint64_t count = s_fallback_counter.fetch_add(1);
        uint64_t seed = (hint * 0x9e3779b97f4a7c15ULL) ^ count;
        uint64_t out = splitmix64(seed);
        if (out == 0ull) out = splitmix64(seed ^ 0xdeadbeefcafebabeULL);
        return out;
    }
}

extern "C" GP_CanvasIdHookFn gp_canvas_set_id_hook(GP_CanvasContext* ctx, GP_CanvasIdHookFn hook) {
    if (!ctx) return nullptr;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    GP_CanvasIdHookFn prev = c->id_hook;
    c->id_hook = hook;
    return prev;
}

// Return edges whose both endpoints are module indices that map to `table`.
// For each matching canvas edge, optionally output the edge desc and the
// persistent rope id (if known). If `out_edges` or `out_rope_ids` are
// NULL, the function returns the required count. `max_entries` bounds the
// number of entries written when output buffers are provided.
extern "C" int gp_canvas_get_table_local_edges_and_rope_ids(GP_CanvasContext* ctx_, GP_TableContext* table, GP_CanvasEdgeDesc* out_edges, uint64_t* out_rope_ids, int max_entries) {
    if (!ctx_ || !table) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    // determine module indices that point to the provided table
    std::vector<int> module_idxs;
    for (size_t mi = 0; mi < c->module_tables.size(); ++mi) {
        if (c->module_tables[mi] == table) module_idxs.push_back(static_cast<int>(mi));
    }
    // also treat the container_table as a special module index marker (-1)
    bool is_container = (c->container_table == table);
    auto module_belongs = [&](int mid) -> bool {
        if (mid < 0 && is_container) return true;
        for (int x : module_idxs) if (x == mid) return true;
        return false;
    };

    // first pass: count matching edges
    int count = 0;
    for (const auto &ei : c->edges) {
        int a = ei.desc.a_module;
        int b = ei.desc.b_module;
        if (module_belongs(a) && module_belongs(b)) ++count;
    }
    if (!out_edges && !out_rope_ids) return count;

    int written = 0;
    for (const auto &ei : c->edges) {
        if (written >= max_entries) break;
        int a = ei.desc.a_module;
        int b = ei.desc.b_module;
        if (!(module_belongs(a) && module_belongs(b))) continue;
        if (out_edges) out_edges[written] = ei.desc;
        uint64_t rid = 0ull;
        // prefer explicit edge rope uid if present; otherwise search rope_id_map
        if (ei.rope_uid != 0ull) rid = ei.rope_uid;
        else {
            rid = gp_canvas_find_persistent_rope_id(ctx_, table, ei.rope_idx);
        }
        if (out_rope_ids) out_rope_ids[written] = rid;
        ++written;
    }
    return written;
}

extern "C" int gp_canvas_table_meta_add_vertex_by_id(GP_CanvasContext* ctx, GP_TableContext* table, void* meta_mg, uint64_t rope_id, int vertex_idx) {
    if (!ctx || !table || !meta_mg || rope_id == 0ull) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    auto it = c->rope_id_map.find(rope_id);
    if (it == c->rope_id_map.end()) {
        printf("gp_canvas_table_meta_add_vertex_by_id: could not find mapping for rope_id=%llu\n", (unsigned long long)rope_id);
        return 0;
    }
    GP_TableContext* mapped_table = it->second.table;
    if (mapped_table != table) {
        printf("gp_canvas_table_meta_add_vertex_by_id: rope_id=%llu maps to a different table (%p) than target (%p)\n", (unsigned long long)rope_id, (void*)mapped_table, (void*)table);
        return 0;
    }
    if (!gp_table_get_rope_sim(table)) {
        printf("gp_canvas_table_meta_add_vertex_by_id: rope_id=%llu table=%p has no RopeSim attached yet; deferring\n", (unsigned long long)rope_id, (void*)table);
        return 0;
    }
    int mapped_idx = gp_table_resolve_rope_id_to_sim_index(table, rope_id);
    if (mapped_idx < 0) {
        printf("gp_canvas_table_meta_add_vertex_by_id: rope_id=%llu could not resolve sim index for table=%p\n", (unsigned long long)rope_id, (void*)table);
        return 0;
    }
    return gp_table_meta_add_vertex(table, reinterpret_cast<GP_MetaGroup*>(meta_mg), mapped_idx, vertex_idx);
}

extern "C" int gp_canvas_resolve_rope_id_to_index(GP_CanvasContext* ctx, GP_TableContext* table, uint64_t rope_id) {
    if (!ctx || !table || rope_id == 0ull) return -1;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    auto it = c->rope_id_map.find(rope_id);
    if (it == c->rope_id_map.end()) return -1;
    if (it->second.table != table) return -1;
    return canvas_resolve_rope_id_to_sim_index(table, rope_id);
}

extern "C" void gp_canvas_mark_rope_map_dirty(GP_CanvasContext* ctx) {
    if (!ctx) return;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    c->rope_map_dirty = true;
}

// Canonical search for persistent rope id for a given table+sim index.
extern "C" uint64_t gp_canvas_find_persistent_rope_id(GP_CanvasContext* ctx_, GP_TableContext* table, int sim_idx) {
    if (!ctx_ || !table || sim_idx < 0) return 0ull;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);

    // Helper: check a specific table for a mapping that refers to the same RopeSim
    auto check_table_for_sim = [&](GP_TableContext* other) -> uint64_t {
        if (!other) return 0ull;
        RopeSim* s1 = gp_table_get_rope_sim(table);
        RopeSim* s2 = gp_table_get_rope_sim(other);
        if (s1 == nullptr || s2 == nullptr) return 0ull;
        if (s1 != s2) return 0ull; // only meaningful if same sim attached

        // Prefer explicit per-table rope ids
        int cur_cnt = gp_table_get_rope_id_count(other);
        if (cur_cnt > 0 && sim_idx >= 0 && sim_idx < cur_cnt) {
            std::vector<uint64_t> tmp(static_cast<size_t>(cur_cnt));
            int got = gp_table_get_rope_ids(other, tmp.data(), cur_cnt);
            if (got > sim_idx && tmp[static_cast<size_t>(sim_idx)] != 0ull) return tmp[static_cast<size_t>(sim_idx)];
        }

        // Fall back to canvas registry entries that point to this table
        return canvas_find_rope_id_for_sim(c, other, sim_idx);
    };

    // 1) Local table
    // Prefer table-local persisted ids if available
    int local_cnt = gp_table_get_rope_id_count(table);
    if (local_cnt > 0 && sim_idx >= 0 && sim_idx < local_cnt) {
        std::vector<uint64_t> tmp(static_cast<size_t>(local_cnt));
        int got = gp_table_get_rope_ids(table, tmp.data(), local_cnt);
        if (got > sim_idx && tmp[static_cast<size_t>(sim_idx)] != 0ull) return tmp[static_cast<size_t>(sim_idx)];
    }

    // also consult canvas registry directly for exact (table,sim_idx) mapping
    uint64_t by_map = canvas_find_rope_id_for_sim(c, table, sim_idx);
    if (by_map != 0ull) return by_map;

    // 2) Up the chain: container table and root module table
    uint64_t found = 0ull;
    found = check_table_for_sim(c->container_table);
    if (found != 0ull) return found;
    if (c->root_module_idx >= 0 && c->root_module_idx < static_cast<int>(c->module_tables.size())) {
        found = check_table_for_sim(c->module_tables[c->root_module_idx]);
        if (found != 0ull) return found;
    }

    // 3) Breadth-first search across module graph
    int mod_count = static_cast<int>(c->modules.size());
    if (mod_count <= 0) return 0ull;
    // find starting modules that host `table`
    std::vector<int> start_mods;
    for (int mi = 0; mi < static_cast<int>(c->module_tables.size()); ++mi) {
        if (c->module_tables[mi] == table) start_mods.push_back(mi);
    }
    if (start_mods.empty()) {
        // fallback to root module if none found
        if (c->root_module_idx >= 0 && c->root_module_idx < static_cast<int>(c->module_tables.size())) start_mods.push_back(c->root_module_idx);
        else start_mods.push_back(0);
    }

    std::vector<char> visited(static_cast<size_t>(mod_count));
    std::deque<int> q;
    for (int s : start_mods) { if (s >= 0 && s < mod_count) { visited[static_cast<size_t>(s)] = 1; q.push_back(s); } }

    while (!q.empty()) {
        int cur = q.front(); q.pop_front();
        GP_TableContext* t = nullptr;
        if (cur >= 0 && cur < static_cast<int>(c->module_tables.size())) t = c->module_tables[cur];
        if (t) {
            uint64_t id = check_table_for_sim(t);
            if (id != 0ull) return id;
        }

        // enqueue neighbors from canvas edges graph
        for (const auto &ei : c->edges) {
            int a = ei.desc.a_module;
            int b = ei.desc.b_module;
            if (a == cur && b >= 0 && b < mod_count && !visited[static_cast<size_t>(b)]) { visited[static_cast<size_t>(b)] = 1; q.push_back(b); }
            else if (b == cur && a >= 0 && a < mod_count && !visited[static_cast<size_t>(a)]) { visited[static_cast<size_t>(a)] = 1; q.push_back(a); }
        }
    }

    return 0ull;
}

extern "C" uint64_t gp_canvas_generate_port_uuid(GP_CanvasContext* ctx) {
    if (!ctx) return 0ull;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    // Prefer central generator so ID scheme is uniform and pluggable.
    return gp_canvas_generate_id(ctx, 0ull);
}

extern "C" uint64_t gp_canvas_generate_module_uuid(GP_CanvasContext* ctx) {
    if (!ctx) return 0ull;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    return gp_canvas_generate_id(ctx, 0ull);
}

// Rope metadata exposure: rope-owned dense network and membership info.
static RopeSim* canvas_table_sim(GP_CanvasContextImpl* c, GP_TableContext* t) {
    if (!t && c) t = c->container_table;
    return t ? gp_table_get_rope_sim(t) : nullptr;
}

extern "C" int gp_canvas_get_rope_meta_owner(GP_CanvasContext* ctx_, GP_TableContext* table, int rope_idx) {
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    RopeSim* sim = canvas_table_sim(c, table);
    if (!sim) return -1;
    int owner = -1;
    if (rope_sim_get_rope_meta_owner(sim, rope_idx, &owner)) return owner;
    return -1;
}

extern "C" int gp_canvas_get_rope_rings(GP_CanvasContext* ctx_, GP_TableContext* table, int rope_idx, int* out_ids, int max_count) {
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    RopeSim* sim = canvas_table_sim(c, table);
    if (!sim) return 0;
    return rope_sim_get_rope_rings(sim, rope_idx, out_ids, max_count);
}

extern "C" int gp_canvas_get_rope_meta_groups(GP_CanvasContext* ctx_, GP_TableContext* table, int rope_idx, int* out_groups, int max_count) {
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    RopeSim* sim = canvas_table_sim(c, table);
    if (!sim) return 0;
    return rope_sim_get_rope_meta_groups(sim, rope_idx, out_groups, max_count);
}

extern "C" int gp_canvas_is_meta_rope(GP_CanvasContext* ctx_, GP_TableContext* table, int rope_idx) {
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    RopeSim* sim = canvas_table_sim(c, table);
    if (!sim) return 0;
    return rope_sim_is_meta_rope(sim, rope_idx);
}

extern "C" int gp_canvas_get_meta_vertices(GP_CanvasContext* ctx_, GP_TableContext* table, int rope_idx, int* out_vertices, int max_count) {
    auto *c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    RopeSim* sim = canvas_table_sim(c, table);
    if (!sim) return 0;
    return rope_sim_get_meta_vertices(sim, rope_idx, out_vertices, max_count);
}

extern "C" int gp_canvas_set_module_uuid(GP_CanvasContext* ctx, int module_idx, uint64_t module_uuid) {
    if (!ctx || module_idx < 0) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    if (module_idx >= static_cast<int>(c->modules.size())) return 0;
    if (static_cast<int>(c->module_uuids.size()) <= module_idx) c->module_uuids.resize(c->modules.size());
    // erase previous mapping if present
    uint64_t prev = c->module_uuids[static_cast<size_t>(module_idx)];
    if (prev != 0ull) c->module_uuid_map.erase(prev);
    c->module_uuids[static_cast<size_t>(module_idx)] = module_uuid;
    if (module_uuid != 0ull) c->module_uuid_map[module_uuid] = module_idx;
    return 1;
}

extern "C" uint64_t gp_canvas_get_module_uuid(GP_CanvasContext* ctx, int module_idx) {
    if (!ctx || module_idx < 0) return 0ull;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    if (module_idx >= static_cast<int>(c->module_uuids.size())) return 0ull;
    return c->module_uuids[static_cast<size_t>(module_idx)];
}

extern "C" int gp_canvas_set_module_frame_port_uuid(GP_CanvasContext* ctx, int module_idx, int is_send, int col, int led_idx, uint64_t port_uuid) {
    if (!ctx || module_idx < 0) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    if (module_idx >= static_cast<int>(c->module_frame_links.size())) return 0;
    if (led_idx < 0 || led_idx >= kModuleExtraLedCount) return 0;
    if (col < 0 || col > 1) return 0;
    int row = is_send ? (col == 0 ? 0 : 1) : (col == 0 ? 2 : 3);
    auto &links = c->module_frame_links[module_idx];
    // remove old mapping if present
    uint64_t old = links.port_uuids[static_cast<size_t>(row)][static_cast<size_t>(led_idx)];
    if (old != 0ull) c->module_port_uuid_map.erase(old);
    links.port_uuids[static_cast<size_t>(row)][static_cast<size_t>(led_idx)] = port_uuid;
    if (port_uuid != 0ull) c->module_port_uuid_map[port_uuid] = {static_cast<int>(module_idx), row, led_idx};
    return 1;
}

extern "C" uint64_t gp_canvas_get_module_frame_port_uuid(GP_CanvasContext* ctx, int module_idx, int is_send, int col, int led_idx) {
    if (!ctx || module_idx < 0) return 0ull;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    if (module_idx >= static_cast<int>(c->module_frame_links.size())) return 0ull;
    if (led_idx < 0 || led_idx >= kModuleExtraLedCount) return 0ull;
    if (col < 0 || col > 1) return 0ull;
    int row = is_send ? (col == 0 ? 0 : 1) : (col == 0 ? 2 : 3);
    return c->module_frame_links[module_idx].port_uuids[static_cast<size_t>(row)][static_cast<size_t>(led_idx)];
}



extern "C" int gp_canvas_register_table_module_uuid(GP_CanvasContext* ctx, GP_TableContext* table, uint64_t module_uuid) {
    if (!ctx || !table) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    for (size_t i = 0; i < c->module_tables.size(); ++i) {
        if (c->module_tables[i] == table) {
            return gp_canvas_set_module_uuid(ctx, static_cast<int>(i), module_uuid);
        }
    }
    return 0;
}

extern "C" int gp_canvas_register_table_frame_port_uuid(GP_CanvasContext* ctx, GP_TableContext* table, int row, int col_idx, uint64_t port_uuid) {
    if (!ctx || !table) return 0;
    GP_CanvasContextImpl* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    for (size_t i = 0; i < c->module_tables.size(); ++i) {
        if (c->module_tables[i] == table) {
            int module_idx = static_cast<int>(i);
            if (row < 0 || row >= kModuleExtraLedRows) return 0;
            if (col_idx < 0 || col_idx >= kModuleExtraLedCount) return 0;
            uint64_t old = c->module_frame_links[static_cast<size_t>(module_idx)].port_uuids[static_cast<size_t>(row)][static_cast<size_t>(col_idx)];
            if (old != 0ull) c->module_port_uuid_map.erase(old);
            c->module_frame_links[static_cast<size_t>(module_idx)].port_uuids[static_cast<size_t>(row)][static_cast<size_t>(col_idx)] = port_uuid;
            if (port_uuid != 0ull) c->module_port_uuid_map[port_uuid] = {module_idx, row, col_idx};
            return 1;
        }
    }
    return 0;
}

extern "C" int gp_canvas_autobind_mouse_ports(GP_CanvasContext* ctx, int module_idx, int max_ports) {
    return canvas_autobind_action_ports(ctx, module_idx, CANVAS_ACT_MENU_TOOL_MOUSE, std::max(0, max_ports));
}

extern "C" int gp_canvas_autobind_keyboard_ports(GP_CanvasContext* ctx, int module_idx, int max_ports) {
    return canvas_autobind_action_ports(ctx, module_idx, CANVAS_ACT_MENU_TOOL_KEYBOARD, std::max(0, max_ports));
}

static void canvas_update_subgroup_palette(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    constexpr float kApproach = 0.18f;
    for (int i = 0; i < kSubgroupBinCount; ++i) {
        const uint32_t packed = ctx->subgroup_target_rgba[static_cast<size_t>(i)].load(std::memory_order_acquire);
        const Color target = unpack_rgba(packed);
        auto &base = ctx->subgroup_base_rgba[static_cast<size_t>(i)];
        const float target_vals[4] = {float(target.r), float(target.g), float(target.b), float(target.a)};
        for (int c = 0; c < 4; ++c) {
            float delta = target_vals[c] - base[static_cast<size_t>(c)];
            if (std::fabs(delta) < 0.5f) {
                base[static_cast<size_t>(c)] = target_vals[c];
            } else {
                base[static_cast<size_t>(c)] += delta * kApproach;
            }
        }
        ctx->subgroup_base_colors[static_cast<size_t>(i)] = Color{
            static_cast<uint8_t>(std::lround(std::clamp(base[0], 0.0f, 255.0f))),
            static_cast<uint8_t>(std::lround(std::clamp(base[1], 0.0f, 255.0f))),
            static_cast<uint8_t>(std::lround(std::clamp(base[2], 0.0f, 255.0f))),
            static_cast<uint8_t>(std::lround(std::clamp(base[3], 0.0f, 255.0f)))
        };
    }
    canvas_recompute_subgroup_palette(ctx);
}

static float subgroup_flags_to_hue(const GP_CanvasContextImpl* ctx, uint32_t flags) {
    if (!ctx) return 0.0f;
    uint32_t idx = subgroup_palette_index(flags);
    return ctx->subgroup_palette_hue[static_cast<size_t>(idx)];
}

static Color subgroup_flags_to_color(const GP_CanvasContextImpl* ctx, uint32_t flags, uint8_t alpha=255) {
    if (!ctx) return Color{120, 120, 130, alpha};
    uint32_t idx = subgroup_palette_index(flags);
    Color out = ctx->subgroup_palette[static_cast<size_t>(idx)];
    out.a = alpha;
    return out;
}

// forward declaration: update scroll/clamp state (defined later in this file)
static CanvasBounds update_canvas_scroll_state(GP_CanvasContextImpl* ctx, bool pull_from_container);

// Forward declarations for symbols defined later but referenced earlier.
struct KeyRecorderState;
static void canvas_dispatch_event_to_bound_ports(GP_CanvasContextImpl* c, int32_t action_id, int x, int y, bool down, bool up, float dx, float dy, int button, float scroll);

// forward declaration: write the tail values into a module's stack snapshot
static void module_stack_tail_write(GP_CanvasContextImpl* ctx, int module_idx, const float* values, int count);

// forward declarations: root sim helpers (defined later)
static RopeSim* canvas_root_sim(GP_CanvasContextImpl* ctx);
static RopeSim* canvas_require_root_sim(GP_CanvasContextImpl* ctx);

const std::vector<ModuleIORow>* canvas_get_module_io_rows(int module_idx) {
    if (!g_canvas_context_singleton) return nullptr;
    if (module_idx < 0 || module_idx >= static_cast<int>(g_canvas_context_singleton->module_io_rows.size())) return nullptr;
    return &g_canvas_context_singleton->module_io_rows[module_idx];
}

// Return the live plugin instance (ITool*) for the given module row, or nullptr.
ITool* canvas_get_plugin_instance(int module_idx, int row_idx) {
    if (!g_canvas_context_singleton) return nullptr;
    auto *ctx = g_canvas_context_singleton;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_plugin_instances.size())) return nullptr;
    const auto &vec = ctx->module_plugin_instances[module_idx];
    if (row_idx < 0 || row_idx >= static_cast<int>(vec.size())) return nullptr;
    auto &p = vec[static_cast<size_t>(row_idx)];
    return p ? p.get() : nullptr;
}

bool canvas_get_module_input_state(int module_idx, ModuleInputState* out_state) {
    if (!g_canvas_context_singleton || !out_state) return false;
    if (module_idx < 0 || module_idx >= static_cast<int>(g_canvas_context_singleton->module_input_state.size())) return false;
    *out_state = g_canvas_context_singleton->module_input_state[module_idx];
    return true;
}

void canvas_clear_module_input_pulses(int module_idx) {
    if (!g_canvas_context_singleton) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(g_canvas_context_singleton->module_input_state.size())) return;
    auto &state = g_canvas_context_singleton->module_input_state[module_idx];
    state.mouse_down = 0;
    state.mouse_up = 0;
    state.key_event = 0;
}

void canvas_set_module_stack_snapshot(int module_idx, int row_idx, const float* values, int count) {
    if (!g_canvas_context_singleton) return;
    if (module_idx < 0) return;
    if (module_idx >= static_cast<int>(g_canvas_context_singleton->module_stack_snapshots.size())) {
        g_canvas_context_singleton->module_stack_snapshots.resize(module_idx + 1);
    }
    auto &snapshots = g_canvas_context_singleton->module_stack_snapshots[module_idx];
    std::vector<float> data;
    if (values && count > 0) {
        data.assign(values, values + count);
    }
    snapshots[row_idx] = std::move(data);
}

void canvas_set_module_stack_tail(int module_idx, const float* values, int count) {
    if (!g_canvas_context_singleton) return;
    module_stack_tail_write(g_canvas_context_singleton, module_idx, values, count);
}

static bool module_has_tool(const GP_CanvasContextImpl* ctx, int module_idx, ModuleToolKind tool) {
    if (!ctx) return false;
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_io_rows.size())) {
        const auto &rows = ctx->module_io_rows[module_idx];
        for (const auto &row : rows) {
            if (row.kind == ModuleRowKind::Tool && row.tool == tool) return true;
        }
    }
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_tool_stack.size())) {
        const auto &tools = ctx->module_tool_stack[module_idx];
        return std::find(tools.begin(), tools.end(), tool) != tools.end();
    }
    return false;
}

static void ensure_module_row_order(GP_CanvasContextImpl* ctx, int module_idx) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) ctx->module_io_rows.resize(module_idx + 1);
    auto &rows = ctx->module_io_rows[module_idx];
    if (!rows.empty()) return;
    if (module_idx >= static_cast<int>(ctx->module_io_input_rows.size())) ctx->module_io_input_rows.resize(module_idx + 1);
    if (module_idx >= static_cast<int>(ctx->module_io_output_rows.size())) ctx->module_io_output_rows.resize(module_idx + 1);
    const auto &input_rows = ctx->module_io_input_rows[module_idx];
    const auto &output_rows = ctx->module_io_output_rows[module_idx];
    for (int count : input_rows) {
        rows.push_back({ModuleRowKind::Input, 0, ModuleToolKind::None, std::clamp(count, 1, 32)});
    }
    if (module_idx < static_cast<int>(ctx->module_tool_stack.size())) {
        for (ModuleToolKind tool : ctx->module_tool_stack[module_idx]) {
            rows.push_back({ModuleRowKind::Tool, -1, tool, 0, ModuleToolOrigin::Builtin, std::string()});
        }
    }
    for (int count : output_rows) {
        rows.push_back({ModuleRowKind::Output, 0, ModuleToolKind::None, std::clamp(count, 1, 32)});
    }
}

static void canvas_setup_stage_table(GP_TableContext* t, int w_px);
static void canvas_setup_stage_defaults(GP_StageContext* st, int w_px, int h_px);
static void stage_bg_callback(void* user, int module_idx, int width, int height, uint8_t* out_rgba, int32_t out_pitch);
static void sync_module_table_io_layout(GP_CanvasContextImpl* ctx, int module_idx);

static void canvas_apply_io_rows(GP_CanvasContextImpl* ctx, int module_idx, const std::vector<ModuleIORow>& rows) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) ctx->module_io_rows.resize(module_idx + 1);
    if (module_idx >= static_cast<int>(ctx->module_io_input_rows.size())) ctx->module_io_input_rows.resize(module_idx + 1);
    if (module_idx >= static_cast<int>(ctx->module_io_output_rows.size())) ctx->module_io_output_rows.resize(module_idx + 1);
    if (module_idx >= static_cast<int>(ctx->module_tool_stack.size())) ctx->module_tool_stack.resize(module_idx + 1);

    ctx->module_io_rows[module_idx].clear();
    ctx->module_io_input_rows[module_idx].clear();
    ctx->module_io_output_rows[module_idx].clear();
    ctx->module_tool_stack[module_idx].clear();

    int in_total = 0;
    for (const auto &row : rows) {
        if (row.kind == ModuleRowKind::Input) {
            in_total += std::clamp(row.attachment_count, 1, 32);
        }
    }
    int in_offset = 0;
    int out_offset = 0;
    for (const auto &row : rows) {
        if (row.kind == ModuleRowKind::Tool) {
            ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Tool, -1, row.tool, row.attachment_count, row.tool_origin, row.plugin_id});
            ctx->module_tool_stack[module_idx].push_back(row.tool);
            continue;
        }
        int count = std::clamp(row.attachment_count, 1, 32);
        if (row.kind == ModuleRowKind::Input) {
            ctx->module_io_input_rows[module_idx].push_back(count);
            ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Input, in_offset, ModuleToolKind::None, count});
            in_offset += count;
        } else if (row.kind == ModuleRowKind::Output) {
            ctx->module_io_output_rows[module_idx].push_back(count);
            ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Output, in_total + out_offset, ModuleToolKind::None, count});
            out_offset += count;
        }
    }
    if (module_idx >= static_cast<int>(ctx->module_io_in_count.size())) ctx->module_io_in_count.resize(module_idx + 1, 0);
    if (module_idx >= static_cast<int>(ctx->module_io_out_count.size())) ctx->module_io_out_count.resize(module_idx + 1, 0);
    ctx->module_io_in_count[module_idx] = in_offset;
    ctx->module_io_out_count[module_idx] = out_offset;
    ensure_module_row_order(ctx, module_idx);
    // synchronize plugin instance vector for this module
    if (module_idx >= static_cast<int>(ctx->module_plugin_instances.size())) ctx->module_plugin_instances.resize(module_idx + 1);
    ctx->module_plugin_instances[module_idx].clear();
    ctx->module_plugin_instances[module_idx].resize(ctx->module_io_rows[module_idx].size());
    // attempt to instantiate plugin-origin rows immediately if registry has them
    for (size_t ri = 0; ri < ctx->module_io_rows[module_idx].size(); ++ri) {
        const auto &r = ctx->module_io_rows[module_idx][ri];
        if (r.kind == ModuleRowKind::Tool && r.tool_origin == ModuleToolOrigin::Plugin && !r.plugin_id.empty()) {
            try {
                auto inst = tool_registry_global().create(r.plugin_id);
                if (inst) {
                    ToolInitContext tctx{};
                    tctx.user = reinterpret_cast<void*>(static_cast<intptr_t>(module_idx));
                    try { inst->initialize(tctx); } catch (...) {}
                    ctx->module_plugin_instances[module_idx][ri] = std::move(inst);
                }
            } catch (...) {}
        }
    }
}

static void canvas_configure_stage_module(GP_CanvasContextImpl* ctx, int module_idx, int w_px, int h_px) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx < static_cast<int>(ctx->module_is_stage.size())) ctx->module_is_stage[module_idx] = 1;
    if (module_idx < static_cast<int>(ctx->module_stages.size())) {
        GP_StageContext* st = gp_stage_create(w_px, h_px, 2);
        ctx->module_stages[module_idx] = st;
        if (module_idx < static_cast<int>(ctx->module_stage_owned.size())) ctx->module_stage_owned[module_idx] = 1;
        canvas_setup_stage_defaults(st, w_px, h_px);
    }
    if (module_idx < static_cast<int>(ctx->module_io_in_count.size())) ctx->module_io_in_count[module_idx] = 1;
    if (module_idx < static_cast<int>(ctx->module_io_out_count.size())) ctx->module_io_out_count[module_idx] = 1;
    if (module_idx < static_cast<int>(ctx->module_bg.size())) {
        ctx->module_bg[module_idx].cb = stage_bg_callback;
        ctx->module_bg[module_idx].user = ctx;
        ctx->module_bg[module_idx].table_alpha = 192;
        ctx->module_bg[module_idx].table_alpha_ray = 192;
        ctx->module_bg[module_idx].header_margin_px = 0;
    }
    if (module_idx < static_cast<int>(ctx->module_tables.size()) && ctx->module_tables[module_idx]) {
        canvas_setup_stage_table(ctx->module_tables[module_idx], w_px);
        sync_module_table_io_layout(ctx, module_idx);
    }
}

static void canvas_clear_workspace(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    for (size_t i = 0; i < ctx->module_tables.size(); ++i) {
        if (ctx->module_tables[i] && i < ctx->module_table_owned.size() && ctx->module_table_owned[i]) {
            gp_table_destroy(ctx->module_tables[i]);
        }
        if (i < ctx->module_key_recorder_state.size()) {
            void* s = ctx->module_key_recorder_state[i];
            if (s) delete reinterpret_cast<KeyRecorderState*>(s);
        }
    }
    for (size_t i = 0; i < ctx->module_stages.size(); ++i) {
        if (ctx->module_stages[i] && i < ctx->module_stage_owned.size() && ctx->module_stage_owned[i]) {
            gp_stage_destroy(ctx->module_stages[i]);
        }
    }
    if (ctx->container_table) {
        gp_table_clear_edges(ctx->container_table);
    }
    ctx->modules.clear();
    ctx->module_tables.clear();
    ctx->module_table_owned.clear();
    ctx->module_stages.clear();
    ctx->module_stage_owned.clear();
    ctx->module_is_stage.clear();
    ctx->module_stage_images.clear();
    ctx->module_stage_cache_rgba.clear();
    ctx->module_stage_cache_w.clear();
    ctx->module_stage_cache_h.clear();
    ctx->module_stage_cache_pitch.clear();
    ctx->module_stage_cache_mu.clear();
    ctx->module_stage_integrator_mode.clear();
    ctx->module_stage_integrator_accum.clear();
    ctx->module_bg.clear();
    ctx->module_io_in_count.clear();
    ctx->module_io_out_count.clear();
    ctx->module_io_input_rows.clear();
    ctx->module_io_output_rows.clear();
    ctx->module_input_layout.clear();
    ctx->module_output_layout.clear();
    ctx->module_io_rows.clear();
    ctx->module_table_rows.clear();
    ctx->module_frame_leds.clear();
    ctx->module_frame_links.clear();
    ctx->module_preview_buffers.clear();
    ctx->module_stack_tail.clear();
    ctx->module_stack_snapshots.clear();
    ctx->module_tool_stack.clear();
    ctx->module_key_recorder_state.clear();
    ctx->module_input_state.clear();
    ctx->module_chat_text.clear();
    ctx->module_chat_color.clear();
    ctx->module_chat_ttl.clear();
    ctx->module_plugin_instances.clear();
    ctx->edges.clear();
    ctx->overlays.clear();
    ctx->overlay_port_uuid_map.clear();
    ctx->overlay_key_map.clear();
    ctx->pending_meta_snapshots.clear();
    ctx->post_load_meta_pending.clear();
    ctx->canonical_to_overlay.clear();
    ctx->nodes.clear();
    ctx->module_node_id.clear();
    ctx->module_uuid_map.clear();
    ctx->module_port_uuid_map.clear();
    ctx->selected = {};
    ctx->dispatch_module_idx = -1;
    ctx->focused_module = -1;
    ctx->prospective_rope_idx = -1;
    ctx->rope_id_map.clear();
    ctx->rope_map_dirty = false;
    ctx->lasso_id_map.clear();
    ctx->next_overlay_id = 1;
    ctx->tool_menu_open = false;
    ctx->rope_menu_open = false;
    ctx->plugin_menu_open = false;
    ctx->plugin_menu_module_idx = -1;
    ctx->module_menu_open = false;
    ctx->module_menu_module_idx = -1;
    int max_window_node = 0;
    for (const auto &entry : ctx->window_node_ids) {
        max_window_node = std::max(max_window_node, entry.second);
    }
    ctx->next_node_id = std::max(1, max_window_node + 1);
}

static void canvas_clear_module_table(GP_CanvasContextImpl* ctx, int module_idx) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx < static_cast<int>(ctx->module_tables.size()) && ctx->module_tables[module_idx]) {
        gp_table_clear_edges(ctx->module_tables[module_idx]);
        gp_table_clear_led_glow(ctx->module_tables[module_idx]);
    }
    if (module_idx < static_cast<int>(ctx->module_stack_snapshots.size())) {
        ctx->module_stack_snapshots[module_idx].clear();
    }
    if (module_idx < static_cast<int>(ctx->module_stack_tail.size())) {
        ModuleStackTail &tail = ctx->module_stack_tail[module_idx];
        tail.count.store(0);
        tail.seq.store(tail.seq.load() + 1);
    }
    if (module_idx < static_cast<int>(ctx->module_chat_text.size())) ctx->module_chat_text[module_idx].clear();
    if (module_idx < static_cast<int>(ctx->module_chat_ttl.size())) ctx->module_chat_ttl[module_idx] = 0;
}

static int canvas_clone_module(GP_CanvasContextImpl* ctx, int module_idx) {
    if (!ctx) return -1;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return -1;
    GP_CanvasModuleDesc d = ctx->modules[module_idx];
    d.x += 24;
    d.y += 24;
    int new_idx = gp_canvas_add_module(reinterpret_cast<GP_CanvasContext*>(ctx), &d);
    if (new_idx < 0) return -1;
    bool is_stage = (module_idx < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[module_idx]);
    if (is_stage) {
        canvas_configure_stage_module(ctx, new_idx, d.w, d.h);
    } else if (module_idx < static_cast<int>(ctx->module_bg.size()) && new_idx < static_cast<int>(ctx->module_bg.size())) {
        const auto &src = ctx->module_bg[module_idx];
        auto &dst = ctx->module_bg[new_idx];
        dst.mode = src.mode;
        dst.rays = src.rays;
        dst.reflections = src.reflections;
        dst.blur_sigma = src.blur_sigma;
        dst.oversample = src.oversample;
        dst.temporal_decay = src.temporal_decay;
        dst.temporal_max = src.temporal_max;
        dst.ray_exposure = src.ray_exposure;
        dst.ray_bounce_decay = src.ray_bounce_decay;
        dst.ray_air_decay = src.ray_air_decay;
        dst.table_alpha = src.table_alpha;
        dst.table_alpha_ray = src.table_alpha_ray;
        dst.header_margin_px = src.header_margin_px;
    }
    const std::vector<ModuleIORow>* rows_ptr = nullptr;
    if (module_idx < static_cast<int>(ctx->module_table_rows.size()) &&
        !ctx->module_table_rows[module_idx].empty()) {
        rows_ptr = &ctx->module_table_rows[module_idx];
    } else if (module_idx < static_cast<int>(ctx->module_io_rows.size())) {
        rows_ptr = &ctx->module_io_rows[module_idx];
    }
    if (rows_ptr) {
        canvas_apply_io_rows(ctx, new_idx, *rows_ptr);
        sync_module_table_io_layout(ctx, new_idx);
    }
    return new_idx;
}

static void canvas_destroy_module(GP_CanvasContextImpl* ctx, int module_idx) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx < static_cast<int>(ctx->module_tables.size()) &&
        module_idx < static_cast<int>(ctx->module_table_owned.size()) &&
        ctx->module_tables[module_idx] && ctx->module_table_owned[module_idx]) {
        gp_table_destroy(ctx->module_tables[module_idx]);
    }
    if (module_idx < static_cast<int>(ctx->module_stages.size()) &&
        module_idx < static_cast<int>(ctx->module_stage_owned.size()) &&
        ctx->module_stages[module_idx] && ctx->module_stage_owned[module_idx]) {
        gp_stage_destroy(ctx->module_stages[module_idx]);
    }
    if (module_idx < static_cast<int>(ctx->module_key_recorder_state.size())) {
        void* s = ctx->module_key_recorder_state[module_idx];
        if (s) delete reinterpret_cast<KeyRecorderState*>(s);
    }
    if (module_idx < static_cast<int>(ctx->module_bg.size()) && ctx->module_bg[module_idx].ray) {
        raytrace2d_destroy(ctx->module_bg[module_idx].ray);
        ctx->module_bg[module_idx].ray = nullptr;
    }
    for (size_t i = 0; i < ctx->edges.size();) {
        auto &edge = ctx->edges[i];
        if (edge.desc.a_module == module_idx || edge.desc.b_module == module_idx) {
            ctx->edges.erase(ctx->edges.begin() + static_cast<long>(i));
            continue;
        }
        if (edge.desc.a_module > module_idx) --edge.desc.a_module;
        if (edge.desc.b_module > module_idx) --edge.desc.b_module;
        ++i;
    }
    for (auto it = ctx->nodes.begin(); it != ctx->nodes.end();) {
        if (it->module_idx == module_idx) {
            it = ctx->nodes.erase(it);
            continue;
        }
        if (it->module_idx > module_idx) --it->module_idx;
        ++it;
    }
    // Remove any module UUID / module-port UUID mappings for this module
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_uuids.size())) {
        uint64_t mu = ctx->module_uuids[static_cast<size_t>(module_idx)];
        if (mu != 0ull) ctx->module_uuid_map.erase(mu);
    }
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_frame_links.size())) {
        const auto &links = ctx->module_frame_links[static_cast<size_t>(module_idx)];
        for (int row = 0; row < kModuleExtraLedRows; ++row) {
            for (int li = 0; li < kModuleExtraLedCount; ++li) {
                uint64_t pu = links.port_uuids[static_cast<size_t>(row)][static_cast<size_t>(li)];
                if (pu != 0ull) ctx->module_port_uuid_map.erase(pu);
            }
        }
    }
    // Drop any action->port bindings that referenced this module so event dispatch
    // can't hit stale PendingAction pointers after the module is destroyed or moved.
    {
        std::lock_guard<std::mutex> lk(ctx->action_subscribers_mu);
        for (auto it = ctx->action_port_bindings.begin(); it != ctx->action_port_bindings.end(); ) {
            auto &vec = it->second;
            for (auto vit = vec.begin(); vit != vec.end(); ) {
                if (vit->module_idx == module_idx) {
                    vit = vec.erase(vit);
                } else {
                    // Fix up module indices greater than the removed one.
                    if (vit->module_idx > module_idx) --vit->module_idx;
                    ++vit;
                }
            }
            if (vec.empty()) {
                it = ctx->action_port_bindings.erase(it);
            } else {
                ++it;
            }
        }
    }
    auto erase_at = [&](auto &vec) {
        if (module_idx >= 0 && module_idx < static_cast<int>(vec.size())) {
            vec.erase(vec.begin() + module_idx);
        }
    };
    erase_at(ctx->modules);
    erase_at(ctx->module_tables);
    erase_at(ctx->module_table_owned);
    erase_at(ctx->module_stages);
    erase_at(ctx->module_stage_owned);
    erase_at(ctx->module_is_stage);
    erase_at(ctx->module_stage_images);
    erase_at(ctx->module_stage_cache_rgba);
    erase_at(ctx->module_stage_cache_w);
    erase_at(ctx->module_stage_cache_h);
    erase_at(ctx->module_stage_cache_pitch);
    erase_at(ctx->module_stage_cache_mu);
    erase_at(ctx->module_stage_integrator_mode);
    erase_at(ctx->module_stage_integrator_accum);
    erase_at(ctx->module_bg);
    erase_at(ctx->module_io_in_count);
    erase_at(ctx->module_io_out_count);
    erase_at(ctx->module_io_input_rows);
    erase_at(ctx->module_io_output_rows);
    erase_at(ctx->module_input_layout);
    erase_at(ctx->module_output_layout);
    erase_at(ctx->module_io_rows);
    erase_at(ctx->module_table_rows);
    erase_at(ctx->module_frame_leds);
    erase_at(ctx->module_frame_links);
    erase_at(ctx->module_preview_buffers);
    erase_at(ctx->module_stack_tail);
    erase_at(ctx->module_stack_snapshots);
    erase_at(ctx->module_tool_stack);
    erase_at(ctx->module_key_recorder_state);
    erase_at(ctx->module_input_state);
    erase_at(ctx->module_chat_text);
    erase_at(ctx->module_chat_color);
    erase_at(ctx->module_chat_ttl);
    erase_at(ctx->module_uuids);
    erase_at(ctx->module_node_id);
    if (ctx->focused_module == module_idx) ctx->focused_module = -1;
    else if (ctx->focused_module > module_idx) --ctx->focused_module;
    if (ctx->selected.module == module_idx) {
        ctx->selected = {};
        ctx->prospective_rope_idx = -1;
    } else if (ctx->selected.module > module_idx) {
        --ctx->selected.module;
    }
    if (ctx->dispatch_module_idx == module_idx) ctx->dispatch_module_idx = -1;
    else if (ctx->dispatch_module_idx > module_idx) --ctx->dispatch_module_idx;
    update_canvas_scroll_state(ctx, /*pull_from_container=*/false);
}

static void canvas_record_key_input(GP_CanvasContextImpl* ctx, int key, int action) {
    if (!ctx || action == 0) return;
    for (int mi = 0; mi < static_cast<int>(ctx->module_input_state.size()); ++mi) {
        if (!module_has_tool(ctx, mi, ModuleToolKind::KeyboardListener)) continue;
        if (mi < 0 || mi >= static_cast<int>(ctx->module_input_state.size())) continue;
        auto &state = ctx->module_input_state[mi];
        state.key = key;
        state.key_event = 1;
    }
}

// forward declarations for helpers used by interactive and deserialization paths
static int canvas_create_overlay_and_attach(GP_CanvasContextImpl* c, GP_TableContext* t, float ox1, float oy1, float ox2, float oy2, unsigned long long* out_key_a, unsigned long long* out_key_b);
static void canvas_create_and_register_ring(GP_CanvasContextImpl* c, GP_TableContext* tbl, GP_MetaGroup* mg, int first_rope_for_ring_local, int first_vid_for_ring_local, float saved_u_override);
static bool canvas_spawn_lasso_anchor_module(GP_CanvasContextImpl* ctx, GP_TableContext* tbl, GP_MetaGroup* mg, int anchor_rope_idx, uint64_t anchor_rope_uid);
static void canvas_finalize_lasso_meta_group(GP_CanvasContextImpl* ctx, GP_TableContext* tbl, GP_MetaGroup* mg, int first_rope_for_ring_local, int first_vid_for_ring_local, float saved_u_override, int lasso_rope_idx, float spawn_x, float spawn_y);
static GP_TableContext* canvas_ensure_root_table(GP_CanvasContextImpl* ctx);
static int canvas_spawn_meta_rope_module(GP_CanvasContextImpl* ctx, GP_TableContext* tbl, GP_MetaGroup* mg, int lasso_rope_idx, float cx, float cy);
static int canvas_create_lasso_meta_rope(GP_CanvasContextImpl* ctx, GP_TableContext* tbl, float cx, float cy);

// Sync a root-table rope's endpoints to explicit world coordinates. This ensures
// the canonical RopeSim used for lasso/meta operations matches the on-screen
// cable geometry.
static void canvas_sync_root_rope_endpoints(GP_CanvasContextImpl* ctx, const GP_CanvasEdgeDesc& desc, float ax, float ay, float bx, float by) {
    if (!ctx) return;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    RopeSim* sim = canvas_root_sim(ctx);
    if (!root || !sim) return;
    gp_table_apply_pending_ops(root);
    uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
    uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
    int edge_idx = -1;
    if (!gp_table_edge_index_for_pair(root, ka, kb, &edge_idx)) return;
    int id_count = gp_table_get_rope_id_count(root);
    if (edge_idx < 0 || edge_idx >= id_count) return;
    std::vector<uint64_t> ids(static_cast<size_t>(id_count));
    int got = gp_table_get_rope_ids(root, ids.data(), id_count);
    if (got <= edge_idx) return;
    uint64_t rid = ids[static_cast<size_t>(edge_idx)];
    int rsi = gp_table_resolve_rope_id_to_sim_index(root, rid);
    if (rsi < 0) return;
    float plug_z = -10.0f;
    rope_sim_move_endpoints3(sim, rsi, ax, ay, plug_z, bx, by, plug_z);
}

static void canvas_record_mouse_input(GP_CanvasContextImpl* ctx, float x, float y, float dx, float dy, bool down, bool up, int button, float scroll) {
    if (!ctx) return;
    // compute canvas/world coords for events (floating point)
    float fx = x + static_cast<float>(ctx->offset_x);
    float fy = y + static_cast<float>(ctx->offset_y);

    // Click-drag tracking: independent of tool state. Dispatch start/move/end
    // events to any registered callback so other systems can subscribe.
    if (down) {
        ctx->click_drag_active = true;
        ctx->click_drag_start_x = fx;
        ctx->click_drag_start_y = fy;
        if (ctx->click_drag_cb) {
            ctx->click_drag_cb(ctx->click_drag_cb_user, 1, ctx->click_drag_start_x, ctx->click_drag_start_y, fx, fy);
        }
        printf("gp_canvas_click_drag: start at %.2f,%.2f (ctx=%p)\n", fx, fy, (void*)ctx);
    } else if (up) {
        if (ctx->click_drag_active) {
            if (ctx->click_drag_cb) {
                ctx->click_drag_cb(ctx->click_drag_cb_user, 3, ctx->click_drag_start_x, ctx->click_drag_start_y, fx, fy);
            }
            printf("gp_canvas_click_drag: end at %.2f,%.2f (start %.2f,%.2f)\n", fx, fy, ctx->click_drag_start_x, ctx->click_drag_start_y);
            ctx->click_drag_active = false;
        }
    } else {
        if (ctx->click_drag_active) {
            if (ctx->click_drag_cb) ctx->click_drag_cb(ctx->click_drag_cb_user, 2, ctx->click_drag_start_x, ctx->click_drag_start_y, fx, fy);
        }
    }

    // Lasso mode handling: collect points while dragging and on mouse-up
    // resolve intersections with attached table ropes to create a
    // meta-group. This keeps the toolbar button lightweight and avoids
    // adding an additional active tool registration.
    if (ctx->lasso_mode) {
        (void)0; // use fx/fy computed above
        if (down) {
            ctx->lasso_points.clear();
            ctx->lasso_points.emplace_back(fx, fy);
            /* lasso start log removed */
            if (ctx->lasso_cb) {
                float pts[2] = { fx, fy };
                ctx->lasso_cb(ctx->lasso_cb_user, 1, pts, 1);
            }
            // Defer module spawn until mouse-up so it only appears after intersections.
            ctx->lasso_module_idx = -1;
            ctx->lasso_rope_idx = -1;
        } else if (up) {
            // finalize lasso: if we have any points, process intersections
            if (!ctx->lasso_points.empty()) {
                // append last point (only if moved enough since last sample)
                {
                    const float min_dist_sq = 4.0f; // ~2 pixels
                    if (!ctx->lasso_points.empty()) {
                        float dx = fx - ctx->lasso_points.back().first;
                        float dy = fy - ctx->lasso_points.back().second;
                        if (dx*dx + dy*dy >= min_dist_sq) ctx->lasso_points.emplace_back(fx, fy);
                    } else {
                        ctx->lasso_points.emplace_back(fx, fy);
                    }
                }
                // Attempt to resolve against the container table if present
                GP_TableContext* tbl = ctx->container_table;
                int first_rope_for_ring = -1;
                int first_vid_for_ring = -1;
                float first_u_for_ring = -1.0f;
                if (!tbl && ctx->root_module_idx >= 0 && ctx->root_module_idx < static_cast<int>(ctx->module_tables.size())) {
                    tbl = ctx->module_tables[ctx->root_module_idx];
                }
                if (tbl) {
                    // create or reuse a meta group and bind any intersected rope vertices
                    GP_MetaGroup* mg = nullptr;
                    // If an overlay already has an authoritative meta-group bound
                    // and its rectangle intersects the lasso, prefer reusing it
                    // instead of creating a new meta-group (prevents empty mg churn).
                    
                    // don't create a meta-group until we actually need one
                    // (we may have found an overlay-bound mg above and set `mg`),
                    // otherwise create lazily when adding the first vertex.
                        // helper: segment-segment intersection
                        auto seg_intersect = [](float x1,float y1,float x2,float y2,float x3,float y3,float x4,float y4, float* ix, float* iy)->bool{
                            float den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
                            if (std::fabs(den) < 1e-6f) return false;
                            float t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den;
                            float u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / den;
                            if (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f) {
                                if (ix) *ix = x1 + t * (x2 - x1);
                                if (iy) *iy = y1 + t * (y2 - y1);
                                return true;
                            }
                            return false;
                        };

                        // iterate ropes known to the table via public edge->rope mapping
                        // Ensure the table has a RopeSim attached so we can
                        // query vertices. If it doesn't, attach the canvas
                        // root sim so coordinates match world space and
                        // meta-groups register correctly.
                        RopeSim* sim = gp_table_get_rope_sim(tbl);
                        if (!sim) {
                            RopeSim* rootsim = canvas_root_sim(ctx);
                            if (!rootsim) rootsim = canvas_require_root_sim(ctx);
                            if (rootsim) {
                                gp_table_attach_rope_sim(tbl, rootsim, 0);
                                sim = gp_table_get_rope_sim(tbl);
                            }
                        }
                        if (sim) {
                            GP_TableGeom g{};
                            gp_table_get_geom(tbl, &g);
                            // offsets computed below; log geom only
                            printf("lasso: table geom w=%d h=%d\n", g.width_px, g.height_px);
                            // Build table-local lasso points. If this table is hosted
                            // inside a module, convert world coords -> table-local so
                            // intersections use the same space as projected verts.
                            float table_off_x = 0.0f;
                            float table_off_y = 0.0f;
                            int host_mod = -1;
                            bool is_container = (tbl == ctx->container_table);
                            if (!is_container) {
                                for (int mi = 0; mi < static_cast<int>(ctx->module_tables.size()); ++mi) {
                                    if (ctx->module_tables[mi] == tbl) { host_mod = mi; break; }
                                }
                                if (host_mod >= 0) {
                                    const auto& m = ctx->modules[host_mod];
                                    int top_h = std::min(m.h, kModuleTopUiHeight);
                                    table_off_x = static_cast<float>(m.x);
                                    table_off_y = static_cast<float>(m.y + top_h);
                                }
                            }
                            printf("lasso: table_off=(%.2f,%.2f) host_mod=%d\n", table_off_x, table_off_y, host_mod);

                            std::vector<std::pair<float,float>> lasso_local;
                            lasso_local.reserve(ctx->lasso_points.size());
                            for (const auto &p : ctx->lasso_points) {
                                lasso_local.emplace_back(p.first - table_off_x, p.second - table_off_y);
                            }
                            // Log lasso segments for diagnostics
                            printf("lasso: lasso_local pts=%zu\n", lasso_local.size());
                            for (size_t li = 0; li + 1 < lasso_local.size(); ++li) {
                                printf("  lasso seg[%zu]=(%.2f,%.2f)->(%.2f,%.2f)\n",
                                    li, lasso_local[li].first, lasso_local[li].second,
                                    lasso_local[li+1].first, lasso_local[li+1].second);
                            }

                            // Collect rope indices to test.
                            std::unordered_set<int> rope_set;
                            if (is_container) {
                                RopeSim* rootsim = canvas_root_sim(ctx);
                                if (rootsim) {
                                    for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
                                        int rsi = ctx->edges[ei].rope_idx;
                                        if (rsi < 0) continue;
                                        int vc = rope_sim_get_vertex_count(rootsim, rsi);
                                        if (vc <= 1) continue;
                                        rope_set.insert(rsi);
                                        uint64_t rid = ctx->edges[ei].rope_uid;
                                        if (rid == 0ull) rid = gp_canvas_generate_id(reinterpret_cast<GP_CanvasContext*>(ctx), 0ull);
                                        if (rid != 0ull) {
                                            gp_table_bind_rope_id_to_sim_index(tbl, rid, rsi);
                                            ctx->rope_id_map[rid] = RopeIdEntry{tbl, rid};
                                            ctx->edges[ei].rope_uid = rid;
                                        }
                                    }
                                }
                            } else {
                                int32_t edge_count = gp_table_get_edge_count(tbl);
                                for (int32_t ei = 0; ei < edge_count; ++ei) {
                                    int32_t rope_idx = -1;
                                    if (!gp_table_get_edge_rope_index(tbl, ei, &rope_idx)) continue;
                                    if (rope_idx < 0) continue;
                                    uint64_t pid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, rope_idx);
                                    if (pid != 0ull) rope_set.insert(rope_idx);
                                }

                                for (const auto &kv : ctx->rope_id_map) {
                                    GP_TableContext* rt = kv.second.table;
                                    if (!rt) continue;
                                    RopeSim* rt_sim = gp_table_get_rope_sim(rt);
                                    if (!rt_sim || rt_sim != sim) continue;
                                    int rsi = gp_table_resolve_rope_id_to_sim_index(rt, kv.second.rope_id);
                                    if (rsi >= 0) rope_set.insert(rsi);
                                }

                                int local_cnt = gp_table_get_rope_id_count(tbl);
                                if (local_cnt > 0) {
                                    std::vector<uint64_t> tmp(static_cast<size_t>(local_cnt));
                                    int got = gp_table_get_rope_ids(tbl, tmp.data(), local_cnt);
                                    if (got > 0) {
                                        for (int ri = 0; ri < got; ++ri) {
                                            if (tmp[static_cast<size_t>(ri)] != 0ull) rope_set.insert(ri);
                                        }
                                    }
                                }

                                const int kScanMax = 128;
                                for (int ri = 0; ri < kScanMax; ++ri) {
                                    int vc = rope_sim_get_vertex_count(sim, ri);
                                    if (vc <= 0) continue;
                                    uint64_t pid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, ri);
                                    if (pid != 0ull) rope_set.insert(ri);
                                }
                            }

                            printf("lasso: finalize points=%zu rope_set=%zu tbl=%p sim=%p\n",
                                ctx->lasso_points.size(), rope_set.size(), (void*)tbl, (void*)sim);

                            // track first intersected rope for anchor ring
                            std::set<int> hit_ropes;
                            for (int rope_idx : rope_set) {
                                if (hit_ropes.find(rope_idx) != hit_ropes.end()) continue;
                                int vc = rope_sim_get_vertex_count(sim, rope_idx);
                                if (vc <= 1) continue;
                                std::vector<float> proj_xy(static_cast<size_t>(vc * 2));
                                int got = gp_table_get_projected_rope_vertices(tbl, rope_idx, proj_xy.data(), static_cast<int>(proj_xy.size()));
                                if (got != vc) continue;
                                bool rope_hit = false;
                                float min_dist = 1e9f;
                                // Also log raw sim vertices for comparison
                                std::vector<float> verts3(static_cast<size_t>(vc * 3));
                                rope_sim_get_vertices3(sim, rope_idx, verts3.data(), static_cast<int>(verts3.size()));
                                printf("lasso: testing rope=%d vc=%d raw_xyz:", rope_idx, vc);
                                for (int vi = 0; vi < vc; ++vi) {
                                    printf(" (%.2f,%.2f,%.2f)", verts3[3*vi+0], verts3[3*vi+1], verts3[3*vi+2]);
                                }
                                printf("\n");
                                // Build segment list (projected).
                                struct Seg { float x0,y0,x1,y1; };
                                std::vector<Seg> segs;
                                for (int ri = 0; ri < vc - 1; ++ri) {
                                    float rx0 = proj_xy[2*ri+0]; float ry0 = proj_xy[2*ri+1];
                                    float rx1 = proj_xy[2*(ri+1)+0]; float ry1 = proj_xy[2*(ri+1)+1];
                                    segs.push_back({rx0, ry0, rx1, ry1});
                                }
                                // Also print projected verts for diagnostic purposes
                                printf("lasso: projected rope=%d vc=%d segs=%zu\n", rope_idx, vc, segs.size());
                                printf("lasso: proj_xy:");
                                for (int vi = 0; vi < vc; ++vi) {
                                    printf(" (%.2f,%.2f)", proj_xy[2*vi+0], proj_xy[2*vi+1]);
                                }
                                printf("\n");
                                // If the rope appears to be a closed loop (last->first non-negligible), include wrap-around segment
                                if (vc > 1) {
                                    float lx0 = proj_xy[0]; float ly0 = proj_xy[1];
                                    float lx1 = proj_xy[2*(vc-1)+0]; float ly1 = proj_xy[2*(vc-1)+1];
                                    float dx = lx0 - lx1; float dy = ly0 - ly1;
                                    if ((dx*dx + dy*dy) > 1e-6f) {
                                        // include last->first as an additional segment for diagnostics
                                        segs.push_back({lx1, ly1, lx0, ly0});
                                    }
                                }
                                for (size_t si = 0; si < segs.size() && !rope_hit; ++si) {
                                    auto s = segs[si];
                                    printf("  rope seg[%zu]=(%.2f,%.2f)->(%.2f,%.2f)\n", si, s.x0, s.y0, s.x1, s.y1);
                                    for (size_t li = 0; li + 1 < lasso_local.size(); ++li) {
                                        float lx0 = lasso_local[li].first; float ly0 = lasso_local[li].second;
                                        float lx1 = lasso_local[li+1].first; float ly1 = lasso_local[li+1].second;
                                        float ix=0, iy=0;
                                        if (seg_intersect(s.x0,s.y0,s.x1,s.y1,lx0,ly0,lx1,ly1,&ix,&iy)) {
                                                rope_hit = true;
                                                float d0 = (ix-s.x0)*(ix-s.x0)+(iy-s.y0)*(iy-s.y0);
                                                float d1 = (ix-s.x1)*(ix-s.x1)+(iy-s.y1)*(iy-s.y1);
                                                // compute fractional parameter along projected segment
                                                float sx = s.x1 - s.x0; float sy = s.y1 - s.y0;
                                                float denom = sx*sx + sy*sy;
                                                float t = 0.0f;
                                                if (denom > 1e-8f) t = ((ix - s.x0) * sx + (iy - s.y0) * sy) / denom;
                                                if (t < 0.0f) t = 0.0f; if (t > 1.0f) t = 1.0f;
                                                float u_param = 0.0f;
                                                if (vc > 1) {
                                                    u_param = (static_cast<float>(si) + t) / static_cast<float>(vc - 1);
                                                }
                                                int new_vid = gp_table_rope_insert_vertex(tbl, static_cast<int32_t>(rope_idx), static_cast<int>(si), t);
                                                // make meta-group lazily when first vertex is about to be added
                                                if (!mg) mg = gp_table_meta_create(tbl);
                                                if (new_vid < 0) {
                                                    int pick = (d0 <= d1) ? static_cast<int>(si) : static_cast<int>(si+1);
                                                    // Use canonical canvas lookup for persistent id (local -> up -> BFS)
                                                    uint64_t persistent_id = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, rope_idx);
                                                    if (persistent_id == 0ull) {
                                                        printf("lasso: ERROR: intersected rope has no persistent id; aborting lasso (tbl=%p rope_idx=%d)\n", (void*)tbl, rope_idx);
                                                        if (mg) { gp_table_meta_destroy(tbl, mg); mg = nullptr; }
                                                        goto lasso_abort;
                                                    }
                                                    gp_table_meta_add_vertex(tbl, mg, static_cast<int32_t>(rope_idx), static_cast<int32_t>(pick));
                                                    printf("lasso: added existing vertex rope=%d vid=%d pid=%llu to mg=%p at ix=%.2f,iy=%.2f (tbl_mod=%d)\n", rope_idx, pick, (unsigned long long)persistent_id, (void*)mg, ix, iy, host_mod);
                                                    gp_table_meta_set_vertex_u(tbl, mg, rope_idx, pick, u_param);
                                                    int sim_idx = -1;
                                                    if (gp_table_meta_get_sim_group_index(tbl, mg, &sim_idx) && sim_idx >= 0) {
                                                        rope_sim_meta_group_set_member_u(sim, sim_idx, rope_idx, pick, u_param);
                                                    }
                                                    if (first_rope_for_ring < 0) { first_rope_for_ring = rope_idx; first_vid_for_ring = pick; first_u_for_ring = u_param; }
                                                } else {
                                                    // Use canonical canvas lookup for persistent id (local -> up -> BFS)
                                                    uint64_t persistent_id = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, rope_idx);
                                                    if (persistent_id == 0ull) {
                                                        printf("lasso: ERROR: intersected rope has no persistent id; aborting lasso (tbl=%p rope_idx=%d)\n", (void*)tbl, rope_idx);
                                                        if (mg) { gp_table_meta_destroy(tbl, mg); mg = nullptr; }
                                                        goto lasso_abort;
                                                    }
                                                    gp_table_meta_add_vertex(tbl, mg, static_cast<int32_t>(rope_idx), static_cast<int32_t>(new_vid));
                                                    printf("lasso: inserted vertex rope=%d vid=%d pid=%llu to mg=%p at t=%.3f ix=%.2f,iy=%.2f (tbl_mod=%d)\n", rope_idx, new_vid, (unsigned long long)persistent_id, (void*)mg, t, ix, iy, host_mod);
                                                    gp_table_meta_set_vertex_u(tbl, mg, rope_idx, new_vid, u_param);
                                                    int sim_idx = -1;
                                                    if (gp_table_meta_get_sim_group_index(tbl, mg, &sim_idx) && sim_idx >= 0) {
                                                        rope_sim_meta_group_set_member_u(sim, sim_idx, rope_idx, new_vid, u_param);
                                                    }
                                                    if (first_rope_for_ring < 0) { first_rope_for_ring = rope_idx; first_vid_for_ring = new_vid; first_u_for_ring = u_param; }
                                                }
                                            rope_hit = true;
                                            hit_ropes.insert(rope_idx);
                                            break;
                                        }
                                        // track nearest approach for diagnostics
                                        float vx = lx1 - lx0; float vy = ly1 - ly0;
                                        float wx = s.x0 - lx0; float wy = s.y0 - ly0;
                                        float seg_len = vx*vx + vy*vy;
                                        float tproj = (seg_len > 1e-6f) ? (wx*vx + wy*vy) / seg_len : 0.0f;
                                        tproj = std::clamp(tproj, 0.0f, 1.0f);
                                        float px = lx0 + tproj * vx;
                                        float py = ly0 + tproj * vy;
                                        float dd = (s.x0 - px)*(s.x0 - px) + (s.y0 - py)*(s.y0 - py);
                                        if (dd < min_dist) min_dist = dd;
                                    }
                                }
                                if (!rope_hit) {
                                    printf("lasso: no intersection for rope=%d (vc=%d) against %zu lasso segments (min_dist=%.2f)\n", rope_idx, vc, lasso_local.size() > 0 ? lasso_local.size() - 1 : 0, std::sqrt(min_dist));
                                }
                            }
                        }
            lasso_abort:
                            if (mg) {
                                // If no vertices were found, destroy the empty meta-group
                                int mg_vcount = gp_table_meta_get_vertex_count(tbl, mg);
                                // Diagnostic dump: verify mg resides in ctx and list vertices
                                int sim_idx = -999;
                                gp_table_meta_get_sim_group_index(tbl, mg, &sim_idx);
                                printf("lasso: pre-destroy-check mg=%p vcount=%d sim_group_idx=%d tbl=%p\n", (void*)mg, mg_vcount, sim_idx, (void*)tbl);
                                if (mg_vcount > 0) {
                                    for (int vi = 0; vi < mg_vcount; ++vi) {
                                        int r = -1, v = -1;
                                        if (gp_table_meta_get_vertex(tbl, mg, vi, &r, &v)) {
                                            printf("  mg vertex[%d] = rope=%d vert=%d\n", vi, r, v);
                                        } else {
                                            printf("  mg vertex[%d] = <failed to read>\n", vi);
                                        }
                                    }
                                }
                                // Recompute vcount directly from internal storage as a cross-check
                                if (tbl) {
                                    // find mg in ctx meta_groups and inspect directly
                                    bool found = false;
                                    for (int mi = 0; mi < gp_table_get_meta_group_count(tbl); ++mi) {
                                        GP_MetaGroup* mg2 = gp_table_get_meta_group(tbl, mi);
                                        if (mg2 == mg) { found = true; break; }
                                    }
                                    printf("  mg present_in_ctx=%d\n", found ? 1 : 0);
                                }
                                // If there are truly no vertices, destroy the meta-group
                                if (mg_vcount == 0) {
                                    gp_table_meta_destroy(tbl, mg);
                                    printf("lasso: destroyed empty meta-group ptr=%p\n", (void*)mg);
                                }
                                else {
                                    // If we detected a preferred first intersection, record it
                                    // on the meta-group so widget creation will use the correct anchor.
                                    if (first_rope_for_ring >= 0 && first_vid_for_ring >= 0) {
                                        gp_table_meta_set_anchor(tbl, mg, first_rope_for_ring, first_vid_for_ring);
                                    }
                                    // Ensure the meta-group carries lasso config so it
                                    // serializes/restores properly. Use the canvas' current
                                    // subgroup flags as the lasso flags and pick a sensible
                                    // non-zero default widget type so dangling widgets
                                    // get created with a visible widget.
                                    int default_widget_type = 1; // non-zero default
                                    gp_table_meta_set_lasso_fields(tbl, mg, ctx->selected_tool_subgroup_flags, default_widget_type);
                                    // Create a ring at the first added rope/vertex so the renderer will show a small sampled ring.
                                    // Prefer registering the ring on the root table when both tables share the same RopeSim
                                    int first_rope_for_ring_local = first_rope_for_ring;
                                    int first_vid_for_ring_local = first_vid_for_ring;
                                    printf("lasso: before finalize mg=%p vcount=%d first_rope=%d first_vid=%d\n",
                                        (void*)mg, gp_table_meta_get_vertex_count(tbl, mg), first_rope_for_ring_local, first_vid_for_ring_local);
                                    canvas_finalize_lasso_meta_group(ctx, tbl, mg, first_rope_for_ring_local, first_vid_for_ring_local, first_u_for_ring, ctx->lasso_rope_idx, fx, fy);
                                    printf("lasso: finalized mg=%p on tbl=%p vcount=%d mg_total=%d\n",
                                        (void*)mg, (void*)tbl,
                                        gp_table_meta_get_vertex_count(tbl, mg),
                                        gp_table_get_meta_group_count(tbl));
                                }
                                // dispatch lasso-end event with collected points
                                if (ctx->lasso_cb && !ctx->lasso_points.empty()) {
                                    int n = static_cast<int>(ctx->lasso_points.size());
                                    std::vector<float> pts(static_cast<size_t>(n) * 2);
                                    for (int i = 0; i < n; ++i) { pts[2*i+0] = ctx->lasso_points[static_cast<size_t>(i)].first; pts[2*i+1] = ctx->lasso_points[static_cast<size_t>(i)].second; }
                                    ctx->lasso_cb(ctx->lasso_cb_user, 3, pts.data(), n);
                                }
                    } else {
                        printf("lasso: finalize aborted with mg=null (no intersections)\n");
                    }
                }
            }
            // Clear any prospective rope shown on the table
            GP_TableContext* tbl_clear = ctx->container_table;
            if (!tbl_clear && ctx->root_module_idx >= 0 && ctx->root_module_idx < static_cast<int>(ctx->module_tables.size())) {
                tbl_clear = ctx->module_tables[ctx->root_module_idx];
            }
            if (tbl_clear && ctx->prospective_rope_idx >= 0 && ctx->prospective_rope_idx != ctx->lasso_rope_idx) {
                gp_table_set_prospective_rope_index(tbl_clear, -1);
                /* cleared prospective rope log removed */
                ctx->prospective_rope_idx = -1;
            }
            ctx->lasso_points.clear();
        } else {
            // mouse move while lasso mode enabled and not yet released: sample
            // only when movement exceeds threshold to avoid excessive points.
            const float min_dist_sq = 4.0f; // ~2 pixels
            if (!ctx->lasso_points.empty()) {
                float dx = fx - ctx->lasso_points.back().first;
                float dy = fy - ctx->lasso_points.back().second;
                if (dx*dx + dy*dy >= min_dist_sq) {
                    ctx->lasso_points.emplace_back(fx, fy);
                    /* lasso sample log removed */
                    if (ctx->lasso_cb) { float pts[2] = { fx, fy }; ctx->lasso_cb(ctx->lasso_cb_user, 2, pts, 1); }
                    // update prospective rope endpoints if present and not the meta rope
                    if (ctx->prospective_rope_idx >= 0 && ctx->prospective_rope_idx != ctx->lasso_rope_idx) {
                        GP_TableContext* tbl_upd = ctx->container_table;
                        if (!tbl_upd && ctx->root_module_idx >= 0 && ctx->root_module_idx < static_cast<int>(ctx->module_tables.size())) {
                            tbl_upd = ctx->module_tables[ctx->root_module_idx];
                        }
                        if (tbl_upd) {
                            RopeSim* sim = gp_table_get_rope_sim(tbl_upd);
                            if (sim) {
                                // compute table-local offsets same as creation
                                float table_off_x = 0.0f, table_off_y = 0.0f;
                                int host_mod = -1;
                                for (int mi = 0; mi < static_cast<int>(ctx->module_tables.size()); ++mi) {
                                    if (ctx->module_tables[mi] == tbl_upd) { host_mod = mi; break; }
                                }
                                if (host_mod >= 0) {
                                    const auto &m = ctx->modules[host_mod];
                                    int top_h = std::min(m.h, kModuleTopUiHeight);
                                    table_off_x = static_cast<float>(m.x);
                                    table_off_y = static_cast<float>(m.y + top_h);
                                }
                                float sx_local = ctx->lasso_points.front().first - table_off_x;
                                float sy_local = ctx->lasso_points.front().second - table_off_y;
                                float fx_local = fx - table_off_x;
                                float fy_local = fy - table_off_y;
                                float plug_z = -10.0f;
                                rope_sim_move_endpoints3(sim, ctx->prospective_rope_idx, sx_local, sy_local, plug_z, fx_local, fy_local, plug_z);
                            }
                        }
                    }
                }
            } else {
                ctx->lasso_points.emplace_back(fx, fy);
                /* lasso sample log removed */
                if (ctx->lasso_cb) { float pts[2] = { fx, fy }; ctx->lasso_cb(ctx->lasso_cb_user, 2, pts, 1); }
            }
        }
        // still forward events to module listeners below
    }
    // Forward events to module listeners: keep `module_input_state` updated
    // so built-in MouseListener tool instances receive live mouse values.
    for (int mi = 0; mi < static_cast<int>(ctx->module_input_state.size()); ++mi) {
        if (!module_has_tool(ctx, mi, ModuleToolKind::MouseListener)) continue;
        if (mi < 0 || mi >= static_cast<int>(ctx->module_input_state.size())) continue;
        auto &state = ctx->module_input_state[mi];
        state.mouse_x = static_cast<float>(x);
        state.mouse_y = static_cast<float>(y);
        if (down) state.mouse_down = 1;
        if (up) state.mouse_up = 1;
        state.mouse_dx = dx;
        state.mouse_dy = dy;
        state.mouse_button = button;
        state.mouse_scroll = scroll;
        state.mouse_button_mask_down = down ? (button > 0 ? (1u << static_cast<uint32_t>(button)) : 0u) : 0u;
        state.mouse_button_mask_up = up ? (button > 0 ? (1u << static_cast<uint32_t>(button)) : 0u) : 0u;
        state.mouse_device_id = 0;
    }
}

// Finalize a meta-group after vertices have been added: set anchor, enforce
// dense ring topology, enable edge-springs, and create/register rings. This
static int canvas_find_module_loopback_edge(const GP_CanvasContextImpl* ctx, int module_idx) {
    if (!ctx || module_idx < 0) return -1;
    for (int ei = static_cast<int>(ctx->edges.size()) - 1; ei >= 0; --ei) {
        const auto &e = ctx->edges[static_cast<size_t>(ei)].desc;
        if (e.a_module == module_idx && e.b_module == module_idx && e.a_contact_idx == 1 && e.b_contact_idx == 0) return ei;
    }
    return -1;
}

static int canvas_create_lasso_rope_for_module(GP_CanvasContextImpl* ctx, GP_TableContext* tbl, int module_idx) {
    if (!ctx || !tbl || module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return -1;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return -1;
    RopeSim* sim = gp_table_get_rope_sim(root);
    if (!sim) sim = canvas_require_root_sim(ctx);
    if (sim && gp_table_get_rope_sim(tbl) != sim) {
        gp_table_attach_rope_sim(tbl, sim, 0);
    }
    if (!sim) return -1;
    gp_table_apply_pending_ops(root);
    uint64_t ka = (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) |
                  (static_cast<uint64_t>(1u) << 16);
    uint64_t kb = (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) |
                  (static_cast<uint64_t>(0u) << 16);
    int edge_idx = -1;
    if (!gp_table_edge_index_for_pair(root, ka, kb, &edge_idx)) {
        if (!gp_table_edge_index_for_pair(root, kb, ka, &edge_idx)) {
            printf("canvas_create_lasso_rope_for_module: no loopback edge for module=%d\n", module_idx);
            return -1;
        }
    }
    int rope_idx = -1;
    if (!gp_table_get_edge_rope_index(root, edge_idx, &rope_idx) || rope_idx < 0) {
        printf("canvas_create_lasso_rope_for_module: missing rope for edge_idx=%d module=%d\n", edge_idx, module_idx);
        return -1;
    }
    uint64_t rid = 0ull;
    int id_count = gp_table_get_rope_id_count(root);
    if (edge_idx >= 0 && edge_idx < id_count) {
        std::vector<uint64_t> ids(static_cast<size_t>(id_count));
        int got = gp_table_get_rope_ids(root, ids.data(), id_count);
        if (edge_idx < got) rid = ids[static_cast<size_t>(edge_idx)];
    }
    if (rid == 0ull) {
        rid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, rope_idx);
    }
    if (rid != 0ull) {
        gp_table_bind_rope_id_to_sim_index(tbl, rid, rope_idx);
        ctx->rope_id_map[rid] = RopeIdEntry{tbl, rid};
    }
    ctx->lasso_rope_idx = rope_idx;
    ctx->prospective_rope_idx = rope_idx;
    int eidx = canvas_find_module_loopback_edge(ctx, module_idx);
    if (eidx >= 0 && eidx < static_cast<int>(ctx->edges.size())) {
        ctx->edges[static_cast<size_t>(eidx)].rope_idx = rope_idx;
        if (rid != 0ull) ctx->edges[static_cast<size_t>(eidx)].rope_uid = rid;
        // Align the root rope geometry with the new module's port positions.
        const auto &m = ctx->modules[static_cast<size_t>(module_idx)];
        float margin_y = 12.0f;
        float span = std::max(1.0f, float(m.h) - 2.0f * margin_y);
        float in_y = float(m.y) + margin_y + 0.5f * span;
        float out_y = in_y;
        float in_x = float(m.x) + 10.0f;
        float out_x = float(m.x + m.w) - 10.0f;
        float plug_z = -10.0f;
        // Use a hard set here so the rope spawns at rest (no endpoint velocity).
        rope_sim_set_endpoints3(sim, rope_idx, in_x, in_y, plug_z, out_x, out_y, plug_z);
    }
    return rope_idx;
}

// does NOT dispatch lasso callbacks.
static void canvas_finalize_lasso_meta_group(GP_CanvasContextImpl* ctx, GP_TableContext* tbl, GP_MetaGroup* mg, int first_rope_for_ring_local, int first_vid_for_ring_local, float saved_u_override, int lasso_rope_idx, float spawn_x, float spawn_y) {
    if (!ctx || !tbl || !mg) return;
    int mg_vcount = gp_table_meta_get_vertex_count(tbl, mg);
    int sim_idx = -999; gp_table_meta_get_sim_group_index(tbl, mg, &sim_idx);
    unsigned long long mgid = 0ull; gp_table_meta_get_id(tbl, mg, &mgid);
    printf("canvas_finalize_lasso_meta_group: mg=%p id=%llu vcount=%d sim_group_idx=%d tbl=%p first_rope=%d first_vid=%d\n", (void*)mg, (unsigned long long)mgid, mg_vcount, sim_idx, (void*)tbl, first_rope_for_ring_local, first_vid_for_ring_local);
    if (sim_idx >= 0 && mgid != 0ull) {
        ctx->lasso_id_map[mgid] = sim_idx;
        printf("canvas_finalize_lasso_meta_group: registered lasso_id=%llu -> sim_idx=%d\n", (unsigned long long)mgid, sim_idx);
    }
    if (mg_vcount == 0) {
        gp_table_meta_destroy(tbl, mg);
        printf("canvas_finalize_lasso_meta_group: destroyed empty meta-group ptr=%p\n", (void*)mg);
        return;
    }
    // Log vertices for diagnostics
    for (int vi = 0; vi < mg_vcount; ++vi) {
        int r = -1, v = -1;
        if (gp_table_meta_get_vertex(tbl, mg, vi, &r, &v)) {
            printf("  mg vertex[%d] = rope=%d vert=%d\n", vi, r, v);
        } else {
            printf("  mg vertex[%d] = <failed to read>\n", vi);
        }
    }
    // Dump lasso/meta-group full state
    gp_table_debug_dump_meta_group(tbl, mg, "finalize");
    // If we detected a preferred first intersection, record it as anchor
    if (first_rope_for_ring_local >= 0 && first_vid_for_ring_local >= 0) {
        gp_table_meta_set_anchor(tbl, mg, first_rope_for_ring_local, first_vid_for_ring_local);
    }
    printf("canvas_finalize_lasso_meta_group: before anchor rope count=%d mg_count_on_tbl=%d\n",
           gp_table_get_edge_count(tbl),
           gp_table_get_meta_group_count(tbl));
    RopeSim* sim = gp_table_get_rope_sim(tbl);
    if (!sim || sim_idx < 0) {
        printf("canvas_finalize_lasso_meta_group: missing sim (sim=%p sim_idx=%d)\n",
               (void*)sim, sim_idx);
        return;
    }
    // Spawn the module at mouse-up before proceeding with meta construction.
    if (ctx->lasso_module_idx < 0) {
        float cx_spawn = spawn_x + kModuleDefaultWidth * 0.5f;
        float cy_spawn = spawn_y + kModuleDefaultHeight * 0.5f;
        int mod = canvas_spawn_meta_rope_module(ctx, tbl, mg, -1, cx_spawn, cy_spawn);
        ctx->lasso_module_idx = mod;
    }
    if (ctx->lasso_module_idx < 0) {
        printf("canvas_finalize_lasso_meta_group: failed to spawn module; aborting\n");
        return;
    }
    if (ctx->lasso_rope_idx < 0) {
        canvas_create_lasso_rope_for_module(ctx, tbl, ctx->lasso_module_idx);
    }
    if (ctx->lasso_rope_idx < 0) {
        printf("canvas_finalize_lasso_meta_group: failed to create lasso rope; aborting\n");
        return;
    }
    lasso_rope_idx = ctx->lasso_rope_idx;
    float table_off_x = 0.0f, table_off_y = 0.0f;
    if (tbl != ctx->container_table) {
        int host_mod = -1;
        for (int mi = 0; mi < static_cast<int>(ctx->module_tables.size()); ++mi) {
            if (ctx->module_tables[mi] == tbl) { host_mod = mi; break; }
        }
        if (host_mod >= 0) {
            const auto &m = ctx->modules[host_mod];
            int top_h = std::min(m.h, kModuleTopUiHeight);
            table_off_x = static_cast<float>(m.x);
            table_off_y = static_cast<float>(m.y + top_h);
        }
    }
    // Compute centroid from non-lasso members for star center placement.
    float cx = 0.0f, cy = 0.0f;
    int samples = 0;
    for (int vi = 0; vi < mg_vcount; ++vi) {
        int r = -1, v = -1;
        if (!gp_table_meta_get_vertex(tbl, mg, vi, &r, &v)) continue;
        if (r == lasso_rope_idx) continue;
        float pos[3] = {0.0f, 0.0f, 0.0f};
        if (rope_sim_meta_group_get_member_world_pos(sim, sim_idx, vi, pos)) {
            // Member positions are table-local; convert to canvas coords for the meta rope.
            cx += pos[0] + table_off_x;
            cy += pos[1] + table_off_y;
            ++samples;
        }
    }
    if (samples > 0) {
        cx /= static_cast<float>(samples);
        cy /= static_cast<float>(samples);
    } else {
        // fallback to lasso rope midpoint
        int vc = rope_sim_get_vertex_count(sim, lasso_rope_idx);
        if (vc > 1) {
            std::vector<float> verts3(static_cast<size_t>(vc * 3));
            rope_sim_get_vertices3(sim, lasso_rope_idx, verts3.data(), static_cast<int>(verts3.size()));
            int mid = vc / 2;
            cx = verts3[static_cast<size_t>(mid) * 3 + 0];
            cy = verts3[static_cast<size_t>(mid) * 3 + 1];
        }
    }
    // Force dense ring topology for lasso-generated groups.
    gp_table_meta_set_ring_mode(tbl, mg, 2);
    rope_sim_meta_group_set_mode(sim, sim_idx, 2);
    rope_sim_set_rope_meta_owner(sim, lasso_rope_idx, sim_idx);
    // Keep the lasso rope geometry minimal; insert a dedicated centroid vertex
    // and pick a separate stem vertex closest to u=0.5.
    int vc = rope_sim_get_vertex_count(sim, lasso_rope_idx);
    if (vc < 2) {
        rope_sim_insert_vertex(sim, lasso_rope_idx, 0, 0.5f);
        vc = rope_sim_get_vertex_count(sim, lasso_rope_idx);
    }
    int segs = std::max(1, vc - 1);
    int mid_seg = std::clamp(segs / 2, 0, segs - 1);
    int centroid_idx = rope_sim_insert_vertex(sim, lasso_rope_idx, mid_seg, 0.5f);
    vc = rope_sim_get_vertex_count(sim, lasso_rope_idx);
    if (centroid_idx < 0) centroid_idx = std::clamp(mid_seg, 0, std::max(0, vc - 1));
    int stem_idx = -1;
    float best_du = 1e9f;
    for (int vi = 0; vi < vc; ++vi) {
        if (vi == centroid_idx) continue;
        float u = (vc > 1) ? (static_cast<float>(vi) / static_cast<float>(vc - 1)) : 0.5f;
        float du = std::fabs(u - 0.5f);
        if (du < best_du) { best_du = du; stem_idx = vi; }
    }
    if (stem_idx < 0) stem_idx = std::clamp(centroid_idx, 0, std::max(0, vc - 1));
    rope_sim_set_vertex_position(sim, lasso_rope_idx, centroid_idx, cx, cy, -10.0f);
    bool has_centroid = false;
    bool has_stem = false;
    for (int vi = 0; vi < mg_vcount; ++vi) {
        int r = -1, v = -1;
        if (!gp_table_meta_get_vertex(tbl, mg, vi, &r, &v)) continue;
        if (r != lasso_rope_idx) continue;
        if (v == centroid_idx) has_centroid = true;
        if (v == stem_idx) has_stem = true;
    }
    if (!has_centroid) {
        gp_table_meta_add_vertex(tbl, mg, lasso_rope_idx, centroid_idx);
        float cu = (vc > 1) ? (static_cast<float>(centroid_idx) / static_cast<float>(vc - 1)) : 0.5f;
        gp_table_meta_set_vertex_u(tbl, mg, lasso_rope_idx, centroid_idx, cu);
        mg_vcount = gp_table_meta_get_vertex_count(tbl, mg);
    }
    if (!has_stem) {
        gp_table_meta_add_vertex(tbl, mg, lasso_rope_idx, stem_idx);
        gp_table_meta_set_vertex_u(tbl, mg, lasso_rope_idx, stem_idx, 0.5f);
        mg_vcount = gp_table_meta_get_vertex_count(tbl, mg);
    }
    int stem_member_idx = -1;
    for (int vi = 0; vi < mg_vcount; ++vi) {
        int r = -1, v = -1;
        if (!gp_table_meta_get_vertex(tbl, mg, vi, &r, &v)) continue;
        if (r == lasso_rope_idx && v == stem_idx) { stem_member_idx = vi; break; }
    }
    rope_sim_meta_group_set_star_center(sim, sim_idx, lasso_rope_idx, centroid_idx);
    // Mark lasso rope as meta rope and store all its vertices.
    int vc_all = rope_sim_get_vertex_count(sim, lasso_rope_idx);
    std::vector<int> meta_vertices;
    meta_vertices.reserve(static_cast<size_t>(vc_all));
    for (int vi = 0; vi < vc_all; ++vi) meta_vertices.push_back(vi);
    rope_sim_mark_meta_rope(sim, lasso_rope_idx, meta_vertices.data(), static_cast<int>(meta_vertices.size()));
    // Ensure the meta rope has a persistent id bound to this table for endpoint registration.
    uint64_t rid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, lasso_rope_idx);
    if (rid == 0ull) {
        rid = gp_canvas_generate_id(reinterpret_cast<GP_CanvasContext*>(ctx), 0ull);
        gp_table_bind_rope_id_to_sim_index(tbl, rid, lasso_rope_idx);
        ctx->rope_id_map[rid] = RopeIdEntry{tbl, rid};
    }
    // Enable edge-springs (use lasso-config values if present)
    LassoConfig lc{};
    gp_table_meta_get_lasso_config(tbl, mg, &lc);
    float min_rest = (lc.spring_min_rest > 0.0f) ? lc.spring_min_rest : 8.0f;
    float reduce_rate = (lc.spring_reduce_rate > 0.0f) ? lc.spring_reduce_rate : 20.0f;
    gp_table_meta_set_edge_spring_params(tbl, mg, min_rest, reduce_rate, 0);
    if (gp_table_meta_enable_edge_springs(tbl, mg, min_rest, reduce_rate)) {
        printf("canvas_finalize_lasso_meta_group: enabled edge-springs for mg=%p min_rest=%.2f rate=%.2f\n", (void*)mg, min_rest, reduce_rate);
    }
    // Create rings for all non-lasso members, plus the stem ring at u=0.5.
    if (mgid != 0ull) {
        uint32_t mg_flags = 0u;
        gp_table_meta_get_subgroup_flags(tbl, mg, &mg_flags);
        std::set<std::pair<int,int>> seen;
        for (int vi = 0; vi < mg_vcount; ++vi) {
            int r = -1, v = -1;
            if (!gp_table_meta_get_vertex(tbl, mg, vi, &r, &v)) continue;
            if (r == lasso_rope_idx) continue;
            auto key = std::make_pair(r, v);
            if (seen.insert(key).second) {
                int vcount = rope_sim_get_vertex_count(sim, r);
                float u = -1.0f;
                gp_table_meta_get_vertex_u(tbl, mg, vi, &u);
                if (!(u >= 0.0f && u <= 1.0f)) {
                    u = (vcount > 1) ? (static_cast<float>(v) / static_cast<float>(vcount - 1)) : 0.5f;
                }
                int ring_id = gp_table_create_ring(tbl, r, u);
                if (ring_id >= 0) {
                    rope_sim_bind_ring_to_member(sim, ring_id, sim_idx, vi);
                    int ring_entry = gp_table_register_ring_edge(tbl, ring_id, mgid);
                    if (ring_entry >= 0 && mg_flags != 0u) gp_table_ring_set_subgroup_flags(tbl, ring_entry, mg_flags);
                }
            }
        }
        int stem_ring = gp_table_create_ring(tbl, lasso_rope_idx, 0.5f);
        if (stem_ring >= 0) {
            if (stem_member_idx >= 0) {
                rope_sim_bind_ring_to_member(sim, stem_ring, sim_idx, stem_member_idx);
            }
            int ring_entry = gp_table_register_ring_edge(tbl, stem_ring, mgid);
            if (ring_entry >= 0 && mg_flags != 0u) gp_table_ring_set_subgroup_flags(tbl, ring_entry, mg_flags);
        }
    }
    printf("canvas_finalize_lasso_meta_group: complete mg=%p now vcount=%d sim_idx=%d table_mg_count=%d\n",
           (void*)mg,
           gp_table_meta_get_vertex_count(tbl, mg),
           sim_idx,
           gp_table_get_meta_group_count(tbl));
}

// Helper: create an overlay at given canvas coords, ensure the table has a
// RopeSim, add a rope in that sim at the same endpoints, attach the rope to
// the overlay via the canvas API, and return the rope index (or -1).
static int canvas_create_overlay_and_attach(GP_CanvasContextImpl* c, GP_TableContext* t, float ox1, float oy1, float ox2, float oy2, unsigned long long* out_key_a, unsigned long long* out_key_b) {
    if (!c || !t) return -1;
    RopeSim* sim = gp_table_get_rope_sim(t);
    if (!sim) {
        RopeSim* rootsim = canvas_root_sim(c);
        if (!rootsim) rootsim = canvas_require_root_sim(c);
        if (rootsim) gp_table_attach_rope_sim(t, rootsim, 0);
        sim = gp_table_get_rope_sim(t);
    }
    int rope_idx = -1;
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
        rope_idx = rope_sim_add_rope3(sim, sx_local, sy_local, plug_z, fx_local, fy_local, plug_z, segs, 0.0f);
    }
    unsigned long long ka = 0ull, kb = 0ull;
    if (gp_canvas_create_overlay_with_leds(reinterpret_cast<GP_CanvasContext*>(c), ox1, oy1, ox2, oy2, &ka, &kb)) {
        if (out_key_a) *out_key_a = ka;
        if (out_key_b) *out_key_b = kb;
        if (rope_idx >= 0) gp_canvas_attach_rope_to_overlay(reinterpret_cast<GP_CanvasContext*>(c), ka, kb, rope_idx);
    }
    return rope_idx;
}

// Create a normal module and a lasso/meta rope bound to its first in/out ports.
static int canvas_create_lasso_meta_rope(GP_CanvasContextImpl* ctx, GP_TableContext* tbl, float cx, float cy) {
    if (!ctx || !tbl) return -1;
    RopeSim* sim = gp_table_get_rope_sim(tbl);
    if (!sim) {
        RopeSim* rootsim = canvas_root_sim(ctx);
        if (!rootsim) rootsim = canvas_require_root_sim(ctx);
        if (rootsim) gp_table_attach_rope_sim(tbl, rootsim, 0);
        sim = gp_table_get_rope_sim(tbl);
    }
    if (!sim) return -1;
    int mod = canvas_spawn_meta_rope_module(ctx, tbl, nullptr, -1, cx, cy);
    if (mod < 0) return -1;
    ctx->lasso_module_idx = mod;
    // Compute approximate port positions on the module.
    const auto &m = ctx->modules[static_cast<size_t>(mod)];
    float margin_y = 12.0f;
    float span = std::max(1.0f, float(m.h) - 2.0f * margin_y);
    float step = span;
    float in_y = float(m.y) + margin_y + 0.5f * step;
    float out_y = in_y;
    float in_x = float(m.x) + 10.0f;
    float out_x = float(m.x + m.w) - 10.0f;
    float plug_z = -10.0f;
    int segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : 2;
    int rope_idx = rope_sim_add_rope3(sim, in_x, in_y, plug_z, out_x, out_y, plug_z, segs, 0.0f);
    if (rope_idx < 0) return -1;
    ctx->lasso_rope_idx = rope_idx;
    ctx->prospective_rope_idx = rope_idx;
    // Bind rope to the module edge (output->input).
    if (!ctx->edges.empty()) {
        int eidx = static_cast<int>(ctx->edges.size() - 1);
        if (eidx >= 0 && eidx < static_cast<int>(ctx->edges.size())) {
            uint64_t rid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, rope_idx);
            if (rid == 0ull) {
                rid = gp_canvas_generate_id(reinterpret_cast<GP_CanvasContext*>(ctx), 0ull);
                gp_table_bind_rope_id_to_sim_index(tbl, rid, rope_idx);
                ctx->rope_id_map[rid] = RopeIdEntry{tbl, rid};
            }
            ctx->edges[static_cast<size_t>(eidx)].rope_idx = rope_idx;
            ctx->edges[static_cast<size_t>(eidx)].rope_uid = rid;
        }
    }
    return rope_idx;
}

// Spawn a normal module and bind the lasso/meta rope to its first in/out ports.
static int canvas_spawn_meta_rope_module(GP_CanvasContextImpl* ctx, GP_TableContext* tbl, GP_MetaGroup* mg, int lasso_rope_idx, float cx, float cy) {
    if (!ctx || !tbl) return -1;
    GP_CanvasModuleDesc md{};
    md.w = kModuleDefaultWidth; md.h = kModuleDefaultHeight;
    md.x = static_cast<int>(std::round(cx - md.w * 0.5f));
    md.y = static_cast<int>(std::round(cy - md.h * 0.5f));
    std::memset(md.label, 0, sizeof(md.label));
    std::memcpy(md.label, LABEL_MODULE_TABLE, std::min<size_t>(std::strlen(LABEL_MODULE_TABLE), sizeof(md.label) - 1));
    int mod = gp_canvas_add_module(reinterpret_cast<GP_CanvasContext*>(ctx), &md);
    if (mod < 0) return -1;
    gp_canvas_create_table(reinterpret_cast<GP_CanvasContext*>(ctx), mod);
    if (mod < static_cast<int>(ctx->module_io_in_count.size())) ctx->module_io_in_count[mod] = 1;
    if (mod < static_cast<int>(ctx->module_io_out_count.size())) ctx->module_io_out_count[mod] = 1;
    // Force IO layout so ports exist immediately.
    int prev_focus = ctx->focused_module;
    ctx->focused_module = mod;
    sync_module_table_io_layout(ctx, mod);
    ctx->focused_module = prev_focus;

    // Create edge between output->input and bind lasso rope to it.
    GP_CanvasEdgeDesc ed{};
    ed.a_module = mod; ed.a_contact_idx = 1; // output
    ed.b_module = mod; ed.b_contact_idx = 0; // input
    int eidx = gp_canvas_add_edge(reinterpret_cast<GP_CanvasContext*>(ctx), &ed);
    if (eidx >= 0 && eidx < static_cast<int>(ctx->edges.size())) {
        uint64_t rid = 0ull;
        if (lasso_rope_idx >= 0) {
            rid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, lasso_rope_idx);
        }
        if (lasso_rope_idx >= 0) {
            if (rid == 0ull) {
                rid = gp_canvas_generate_id(reinterpret_cast<GP_CanvasContext*>(ctx), 0ull);
                gp_table_bind_rope_id_to_sim_index(tbl, rid, lasso_rope_idx);
                ctx->rope_id_map[rid] = RopeIdEntry{tbl, rid};
            }
            ctx->edges[static_cast<size_t>(eidx)].rope_idx = lasso_rope_idx;
            ctx->edges[static_cast<size_t>(eidx)].rope_uid = rid;
        }
        uint32_t sflags = 0;
        if (mg && gp_table_meta_get_subgroup_flags(tbl, mg, &sflags)) {
            ctx->edges[static_cast<size_t>(eidx)].subgroup_flags = sflags;
        }
    }
    return mod;
}

// Helper: from first rope+vertex metadata, create a ring using the same
// logic as the interactive lasso: prefer registering on the root table if
// it shares the same RopeSim, otherwise register on the provided table.
static void canvas_create_and_register_ring(GP_CanvasContextImpl* c, GP_TableContext* tbl, GP_MetaGroup* mg, int first_rope_for_ring_local, int first_vid_for_ring_local, float saved_u_override) {
    if (!c || !tbl || !mg) return;
    if (first_rope_for_ring_local < 0 || first_vid_for_ring_local < 0) return;
    GP_TableContext* root_tbl = c->container_table;
    GP_TableContext* reg_tbl = nullptr;
    RopeSim* tbl_sim = gp_table_get_rope_sim(tbl);
    RopeSim* root_sim = root_tbl ? gp_table_get_rope_sim(root_tbl) : nullptr;
    if (root_tbl && root_sim && tbl_sim && root_sim == tbl_sim) reg_tbl = root_tbl;
    else reg_tbl = tbl;
    if (!reg_tbl) return;
    RopeSim* reg_sim = gp_table_get_rope_sim(reg_tbl);
    int vc = rope_sim_get_vertex_count(reg_sim, first_rope_for_ring_local);
    if (vc > 1) {
        float u = saved_u_override;
        if (!(u > 0.0f && u <= 1.0f)) {
            // Avoid placing the ring exactly on a discrete vertex index.
            // Use the midpoint between the vertex and the next vertex so the
            // ring sits on the segment, not coincident with the vertex.
            float fidx = static_cast<float>(first_vid_for_ring_local) + 0.5f;
            u = fidx / static_cast<float>(vc - 1);
            if (u <= 0.0f) u = 0.001f;
            if (u >= 1.0f) u = 0.999f;
        }
        int ring_id = gp_table_create_ring(reg_tbl, first_rope_for_ring_local, u);
        if (ring_id >= 0) {
            unsigned long long mgid = 0ull;
            if (!gp_table_meta_get_id(reg_tbl, mg, &mgid)) mgid = 0ull;
            int ring_entry = gp_table_register_ring_edge(reg_tbl, ring_id, mgid);
            uint32_t flags = 0u;
            gp_table_meta_get_subgroup_flags(reg_tbl, mg, &flags);
            if (ring_entry >= 0 && flags != 0u) gp_table_ring_set_subgroup_flags(reg_tbl, ring_entry, flags);
            printf("canvas_create_and_register_ring: created ring id=%d on rope=%d u=%.3f registered mgid=%llu\n", ring_id, first_rope_for_ring_local, u, (unsigned long long)mgid);
        }
    }
}

// Create a minimal anchor module at the centroid of the meta-group members,
// add a single input/output rope between its first contacts, and join that
// rope into the provided meta-group with a ring for visibility.
static bool canvas_spawn_lasso_anchor_module(GP_CanvasContextImpl* ctx, GP_TableContext* tbl, GP_MetaGroup* mg, int anchor_rope_idx, uint64_t anchor_rope_uid) {
    if (!ctx || !tbl || !mg) return false;
    RopeSim* sim = gp_table_get_rope_sim(tbl);
    if (!sim) {
        RopeSim* rootsim = canvas_root_sim(ctx);
        if (!rootsim) rootsim = canvas_require_root_sim(ctx);
        if (rootsim) {
            gp_table_attach_rope_sim(tbl, rootsim, 0);
            sim = gp_table_get_rope_sim(tbl);
        }
    }
    int sim_idx = -1;
    gp_table_meta_get_sim_group_index(tbl, mg, &sim_idx);
    if (!sim || sim_idx < 0) {
        printf("lasso_anchor: no sim or sim_idx for mg=%p tbl=%p sim=%p sim_idx=%d\n", (void*)mg, (void*)tbl, (void*)sim, sim_idx);
        return false;
    }
    int member_count = gp_table_meta_get_vertex_count(tbl, mg);
    if (member_count <= 0) {
        printf("lasso_anchor: mg=%p has no members; skipping anchor\n", (void*)mg);
        return false;
    }

    float table_off_x = 0.0f, table_off_y = 0.0f;
    if (tbl != ctx->container_table) {
        int host_mod = -1;
        for (int mi = 0; mi < static_cast<int>(ctx->module_tables.size()); ++mi) {
            if (ctx->module_tables[mi] == tbl) { host_mod = mi; break; }
        }
        if (host_mod >= 0) {
            const auto &m = ctx->modules[host_mod];
            int top_h = std::min(m.h, kModuleTopUiHeight);
            table_off_x = static_cast<float>(m.x);
            table_off_y = static_cast<float>(m.y + top_h);
        }
    }

    float cx = 0.0f, cy = 0.0f;
    int samples = 0;
    for (int i = 0; i < member_count; ++i) {
        float pos[3] = {0.0f, 0.0f, 0.0f};
        if (rope_sim_meta_group_get_member_world_pos(sim, sim_idx, i, pos)) {
            // Member positions are table-local; convert to canvas coords.
            cx += pos[0] + table_off_x;
            cy += pos[1] + table_off_y;
            ++samples;
        } else {
            printf("lasso_anchor: failed to sample member %d\n", i);
        }
    }
    if (samples <= 0) {
        printf("lasso_anchor: could not sample member positions for mg=%p\n", (void*)mg);
        return false;
    }
    cx /= static_cast<float>(samples);
    cy /= static_cast<float>(samples);

    // Convert centroid to table-local coords (if table is hosted in a module)
    float lx = cx - table_off_x;
    float ly = cy - table_off_y;
    float plug_z = -10.0f;
    int rope_idx = anchor_rope_idx;
    if (rope_idx < 0) {
        printf("lasso_anchor: no anchor rope provided; skipping module spawn\n");
        return false;
    }
    uint64_t anchor_rid = anchor_rope_uid;
    if (anchor_rid == 0ull) {
        anchor_rid = gp_canvas_find_persistent_rope_id(reinterpret_cast<GP_CanvasContext*>(ctx), tbl, rope_idx);
        if (anchor_rid == 0ull) anchor_rid = gp_canvas_generate_id(reinterpret_cast<GP_CanvasContext*>(ctx), 0ull);
    }
    gp_table_bind_rope_id_to_sim_index(tbl, anchor_rid, rope_idx);
    ctx->rope_id_map[anchor_rid] = RopeIdEntry{tbl, anchor_rid};

    // Use the existing rope as anchor: add its midpoint as meta vertex/anchor.
    int vc = rope_sim_get_vertex_count(sim, rope_idx);
    int anchor_vid = (vc > 1) ? std::max(0, std::min(vc - 1, vc / 2)) : 0;
    gp_table_meta_add_vertex(tbl, mg, rope_idx, anchor_vid);
    gp_table_meta_set_anchor(tbl, mg, rope_idx, anchor_vid);

    // Spawn a small canvas module at the centroid with one input and one output,
    // and attach the anchor rope to its first edge for data endpoints.
    GP_CanvasModuleDesc md{};
    md.w = kModuleDefaultWidth; md.h = kModuleDefaultHeight;
    md.x = static_cast<int>(std::round(cx - md.w * 0.5f));
    md.y = static_cast<int>(std::round(cy - md.h * 0.5f));
    std::memset(md.label, 0, sizeof(md.label));
    std::memcpy(md.label, LABEL_MODULE_TABLE, std::min<size_t>(std::strlen(LABEL_MODULE_TABLE), sizeof(md.label) - 1));
    int anchor_mod = gp_canvas_add_module(reinterpret_cast<GP_CanvasContext*>(ctx), &md);
    if (anchor_mod >= 0) {
        gp_canvas_create_table(reinterpret_cast<GP_CanvasContext*>(ctx), anchor_mod);
    }
    if (anchor_mod >= 0 && anchor_mod < static_cast<int>(ctx->module_io_in_count.size())) {
        ctx->module_io_in_count[anchor_mod] = 1;
    }
    if (anchor_mod >= 0 && anchor_mod < static_cast<int>(ctx->module_io_out_count.size())) {
        ctx->module_io_out_count[anchor_mod] = 1;
    }
    if (anchor_mod >= 0) {
        // Force-build IO layout so hitboxes exist even if this module isn't focused.
        int prev_focus = ctx->focused_module;
        ctx->focused_module = anchor_mod;
        sync_module_table_io_layout(ctx, anchor_mod);
        ctx->focused_module = prev_focus;
        GP_TableContext* mt = (anchor_mod < static_cast<int>(ctx->module_tables.size())) ? ctx->module_tables[anchor_mod] : nullptr;
        if (mt && gp_table_get_rope_sim(mt) != sim) gp_table_attach_rope_sim(mt, sim, 0);
        if (mt) {
            gp_table_bind_rope_id_to_sim_index(mt, anchor_rid, rope_idx);
            ctx->rope_id_map[anchor_rid] = RopeIdEntry{mt, anchor_rid};
        }
        GP_CanvasEdgeDesc ed{};
        ed.a_module = anchor_mod; ed.a_contact_idx = 1; // output
        ed.b_module = anchor_mod; ed.b_contact_idx = 0; // input
        int eidx = gp_canvas_add_edge(reinterpret_cast<GP_CanvasContext*>(ctx), &ed);
        if (eidx >= 0 && eidx < static_cast<int>(ctx->edges.size())) {
            ctx->edges[static_cast<size_t>(eidx)].rope_idx = rope_idx;
            ctx->edges[static_cast<size_t>(eidx)].rope_uid = anchor_rid;
            uint32_t sflags = 0;
            if (gp_table_meta_get_subgroup_flags(tbl, mg, &sflags)) {
                ctx->edges[static_cast<size_t>(eidx)].subgroup_flags = sflags;
            }
        }
        // Snap the anchor rope endpoints roughly to the first input/output contact centers.
        const auto &m = ctx->modules[static_cast<size_t>(anchor_mod)];
        float margin_y = 12.0f;
        float span = std::max(1.0f, float(m.h) - 2.0f * margin_y);
        float step = span; // only one contact each side
        float in_y = float(m.y) + margin_y + 0.5f * step;
        float out_y = in_y;
        float in_x = float(m.x) + 10.0f;
        float out_x = float(m.x + m.w) - 10.0f;
        rope_sim_set_endpoints3(sim, rope_idx, in_x, in_y, plug_z, out_x, out_y, plug_z);
    }
    return true;
}
static inline int table_hit_contact_index(const GP_TableHitBox& hb) {
    if (hb.row_idx >= 0) return hb.row_idx;
    if (hb.part == GP_TABLE_HIT_LED_TABLE) return hb.aux1;
    return hb.aux0;
}

static int resolve_contact_index(const GP_CanvasContextImpl* ctx, int module_idx, const GP_TableHitBox& hb) {
    if (!ctx) return table_hit_contact_index(hb);
    const std::vector<ModuleIORow>* rows_ptr = nullptr;
    bool using_table_rows = false;
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_table_rows.size()) &&
        !ctx->module_table_rows[module_idx].empty()) {
        rows_ptr = &ctx->module_table_rows[module_idx];
        using_table_rows = true;
    } else if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_io_rows.size())) {
        rows_ptr = &ctx->module_io_rows[module_idx];
    }
    if (rows_ptr) {
        const auto &rows = *rows_ptr;
        if (hb.row_idx >= 0 && hb.row_idx < static_cast<int>(rows.size())) {
            const ModuleIORow &meta = rows[hb.row_idx];
            if (meta.kind == ModuleRowKind::Tool) return -1;
            if (using_table_rows) {
                bool is_left = (hb.col_idx == kModuleColLeftLed);
                bool is_right = (hb.col_idx == kModuleColRightLed);
                if (meta.kind == ModuleRowKind::Input && !is_left) return -1;
                if (meta.kind == ModuleRowKind::Output && !is_right) return -1;
                if (!is_left && !is_right) return -1;
            }
            int attachment_count = std::max(1, meta.attachment_count);
            int led_idx = 0;
            if (hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG) {
                led_idx = std::clamp(hb.aux0, 0, attachment_count - 1);
            } else if (hb.part == GP_TABLE_HIT_LED_TABLE) {
                led_idx = std::clamp(hb.aux1, 0, attachment_count - 1);
            }
            return meta.contact_idx + led_idx;
        }
    }
    return table_hit_contact_index(hb);
}

static int resolve_side_contact_index(const GP_CanvasContextImpl* ctx, int module_idx, bool is_input, int contact_idx) {
    if (is_input) return contact_idx;
    if (!ctx) return contact_idx;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_io_in_count.size())) return contact_idx;
    int offset = std::max(0, ctx->module_io_in_count[module_idx]);
    return contact_idx - offset;
}

struct ModuleLayout {
    int top_h = 0;
    int table_y = 0;
    int table_clip_h = 0;
    int preview_h = 0;
    int preview_y = 0;
    int drag_y = 0;
    int drag_h = 0;
};

static ModuleLayout module_layout_for(const GP_CanvasModuleDesc& m) {
    ModuleLayout layout{};
    if (m.w <= 0 || m.h <= 0) return layout;
    layout.top_h = std::min(m.h, kModuleTopUiHeight);
    layout.table_y = layout.top_h;
    int available_h = std::max(0, m.h - layout.top_h);
    int preview_h = std::min(m.w, available_h);
    if ((available_h - preview_h) < kModulePreviewMinTableHeight) {
        preview_h = std::max(0, available_h - kModulePreviewMinTableHeight);
        preview_h = std::min(preview_h, m.w);
    }
    layout.preview_h = std::max(0, preview_h);
    layout.preview_y = std::max(0, m.h - layout.preview_h);
    layout.table_clip_h = std::max(1, layout.preview_y - layout.table_y);
    layout.drag_y = std::min(layout.top_h, kModuleTopPadding + kModuleTitleRowH);
    layout.drag_h = std::max(0, std::min(kModuleThumbRowH, layout.top_h - layout.drag_y));
    return layout;
}

static ModuleLayout module_layout_for(const GP_CanvasContextImpl* ctx, int module_idx, const GP_CanvasModuleDesc& m) {
    ModuleLayout layout = module_layout_for(m);
    bool is_stage = ctx && module_idx >= 0 && module_idx < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[module_idx];
    if (is_stage) {
        layout.top_h = 0;
        layout.table_y = 0;
        layout.preview_h = 0;
        layout.preview_y = std::max(0, m.h);
        layout.table_clip_h = std::max(1, m.h);
        layout.drag_y = 0;
        layout.drag_h = std::min(kModuleThumbRowH, std::max(0, m.h));
    }
    return layout;
}

static GP_TableCell* module_frame_led_cell(GP_CanvasContextImpl* ctx, int module_idx, int row, int idx) {
    if (!ctx) return nullptr;
    if (row < 0 || row >= kModuleExtraLedRows) return nullptr;
    if (idx < 0 || idx >= kModuleExtraLedCount) return nullptr;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_frame_leds.size())) return nullptr;
    auto &group = ctx->module_frame_leds[module_idx];
    return &group.cells[static_cast<size_t>(row)][static_cast<size_t>(idx)];
}

static void module_stack_tail_write(GP_CanvasContextImpl* ctx, int module_idx, const float* values, int count) {
    if (!ctx || module_idx < 0) return;
    if (module_idx >= static_cast<int>(ctx->module_stack_tail.size())) return;
    ModuleStackTail &tail = ctx->module_stack_tail[module_idx];
    uint32_t seq = tail.seq.load(std::memory_order_relaxed);
    tail.seq.store(seq + 1, std::memory_order_release);
    int to_copy = std::clamp(count, 0, static_cast<int>(tail.values.size()));
    for (int i = 0; i < to_copy; ++i) {
        tail.values[static_cast<size_t>(i)] = values ? values[i] : 0.0f;
    }
    tail.count.store(to_copy, std::memory_order_release);
    tail.seq.store(seq + 2, std::memory_order_release);
}

static void module_stack_tail_read(const GP_CanvasContextImpl* ctx, int module_idx, float* out_vals, int* out_count) {
    if (out_count) *out_count = 0;
    if (!ctx || !out_vals || !out_count) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_stack_tail.size())) return;
    const ModuleStackTail &tail = ctx->module_stack_tail[module_idx];
    for (int attempt = 0; attempt < 3; ++attempt) {
        uint32_t start = tail.seq.load(std::memory_order_acquire);
        if (start & 1u) continue;
        int count = tail.count.load(std::memory_order_acquire);
        int to_copy = std::clamp(count, 0, static_cast<int>(tail.values.size()));
        for (int i = 0; i < to_copy; ++i) {
            out_vals[i] = tail.values[static_cast<size_t>(i)];
        }
        uint32_t end = tail.seq.load(std::memory_order_acquire);
        if (start == end && !(end & 1u)) {
            *out_count = to_copy;
            return;
        }
    }
}

static void build_module_preview_input(const GP_CanvasContextImpl* ctx, int module_idx, GP_ModulePreviewInput &out) {
    std::memset(&out, 0, sizeof(out));
    if (!ctx) return;
    out.rows = 0;
    out.cols = 0;
    out.layout_strategy = 0;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_frame_links.size())) return;
    const auto &links = ctx->module_frame_links[module_idx];
    // populate preview with the first send/receive logical rows (left column)
    for (int i = 0; i < kModuleExtraLedCount; ++i) {
        out.send_ptrs[i] = links.ptrs[0][static_cast<size_t>(i)];
        out.receive_ptrs[i] = links.ptrs[2][static_cast<size_t>(i)];
    }
    module_stack_tail_read(ctx, module_idx, out.stack_tail, &out.stack_tail_count);
}

// forward-declare helper used by draw routines
static std::string canvas_module_label(const GP_CanvasModuleDesc& desc);

static void draw_module_top_ui(GP_CanvasContextImpl* ctx, int module_idx, const GP_CanvasModuleDesc& m, uint8_t* out_rgba, int w, int h, int pitch) {
    if (!ctx || !out_rgba) return;
    ModuleLayout layout = module_layout_for(ctx, module_idx, m);
    if (layout.top_h <= 0) return;
    int sx = m.x - ctx->offset_x;
    int sy = m.y - ctx->offset_y;
    int top_h = layout.top_h;
    memset_rect(out_rgba, w, h, pitch, sx, sy, m.w, top_h, Color{34,34,46,235});
    memset_rect(out_rgba, w, h, pitch, sx, sy + top_h - 1, m.w, 1, Color{18,18,26,255});

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

    int cursor_y = sy + kModuleTopPadding;
    std::string title = canvas_module_label(m);
    // append timing info when timing mode enabled
    if (ThreadManager::global() && ThreadManager::global()->timing_enabled()) {
        ThreadManager::ModuleTiming mt{};
        if (ThreadManager::global()->get_module_timing(module_idx, &mt)) {
            char buf[128];
            // show counts and times in milliseconds
            int n = snprintf(buf, sizeof(buf), "  [%llu runs last=%.3fms total=%.3fms]",
                             (unsigned long long)mt.run_count,
                             mt.last_run_wall_time * 1000.0,
                             mt.total_run_wall_time * 1000.0);
            if (n > 0) title += std::string(buf, static_cast<size_t>(std::max(0, n)));
        }
    }
    // append attached-table type names for diagnostics via public API
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_tables.size()) && ctx->module_tables[module_idx]) {
        GP_TableContext* tbl = ctx->module_tables[module_idx];
        const int kCap = 64;
        int ids[kCap];
        int total = gp_table_get_type_ids(tbl, ids, kCap);
        if (total > 0) {
            std::string ts;
            int take = std::min<int>(total, kCap);
            for (int i = 0; i < take; ++i) {
                const ValueType* vt = ValueTypeRegistry::global().get(ids[i]);
                if (vt) {
                    if (!ts.empty()) ts += ",";
                    // vt->name is a char array
                    ts += std::string(vt->name, vt->name + std::strlen(vt->name));
                } else {
                    if (!ts.empty()) ts += ",";
                    ts += std::to_string(static_cast<int>(ids[i]));
                }
            }
            title += std::string("  [table=") + ts + "]";
        }
    }
    blit_text(title.c_str(), sx + kModuleTopPadding, cursor_y, 1.05f, Color{220,220,230,255});
    cursor_y += kModuleTitleRowH;

    int thumb_y = cursor_y;
    int thumb_x = sx + kModuleTopPadding;
    int thumb_w = std::max(1, m.w - kModuleTopPadding * 2);
    memset_rect(out_rgba, w, h, pitch, thumb_x, thumb_y, thumb_w, kModuleThumbRowH, Color{40,40,52,255});
    for (int i = 0; i < 3; ++i) {
        int bar_w = thumb_w / 4;
        int bar_x = thumb_x + (thumb_w - bar_w) / 2;
        int bar_y = thumb_y + 3 + i * 4;
        memset_rect(out_rgba, w, h, pitch, bar_x, bar_y, bar_w, 2, Color{70,70,88,255});
    }
    cursor_y += kModuleThumbRowH + kModuleTopGap;

    int control_y = cursor_y;
    int control_h = std::max(1, kModuleControlRowH - 2);
    int gap = 6;
    auto draw_button = [&](int bx, int by, int bw, int bh, Color fill, const char* label, float scale) {
        memset_rect(out_rgba, w, h, pitch, bx, by, bw, bh, fill);
        memset_rect(out_rgba, w, h, pitch, bx, by, bw, 1, Color{20,20,28,255});
        memset_rect(out_rgba, w, h, pitch, bx, by + bh - 1, bw, 1, Color{12,12,18,255});
        if (label && label[0] != '\0') {
            auto bm = render_text_to_rgba(label, scale, {230,230,235,255});
            if (!bm.pixels.empty()) {
                int tx = bx + (bw - bm.width) / 2;
                int ty = by + (bh - bm.height) / 2;
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
    };

    int group_x = sx + kModuleTopPadding;
    int nbw = control_h;
    int num_w = std::max(40, control_h * 2);
    int btn_y = control_y + 1;
    draw_button(group_x, btn_y, nbw, control_h, Color{52,52,64,255}, LABEL_IO_MINUS, 1.0f);
    draw_button(group_x + nbw + gap, btn_y, num_w, control_h, Color{36,36,46,255}, "0", 0.9f);
    draw_button(group_x + nbw + gap + num_w + gap, btn_y, nbw, control_h, Color{52,52,64,255}, LABEL_IO_PLUS, 1.0f);

    int right_x = sx + m.w - kModuleTopPadding;
    int menu_w = std::max(30, control_h);
    int lib_w = menu_w;
    int pause_w = std::max(42, control_h * 2);
    int menu_x = right_x - menu_w;
    int lib_x = menu_x - gap - lib_w;
    int pause_x = lib_x - gap - pause_w;
    // Per-module play/pause: this button toggles the entire-module skip flag.
    bool module_paused = false;
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_skip.size())) module_paused = (ctx->module_skip[module_idx] != 0);
    else module_paused = ctx->thread_mgr_paused;
    const char* pause_label = module_paused ? LABEL_THREAD_PLAY_SHORT : LABEL_THREAD_PAUSE_SHORT;
    // Split the pause/play button: left half = module SIM toggle, right half = module play/pause
    int half_play_w = std::max(8, pause_w / 2);
    int sim_x = pause_x;
    int sim_w = half_play_w;
    int play_x = pause_x + half_play_w;
    int play_w = pause_w - half_play_w;
    bool module_sim = false;
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_sim_enabled.size())) module_sim = (ctx->module_sim_enabled[module_idx] != 0);
    const char* sim_label = module_sim ? "SIM" : "sim";
    // draw left SIM half
    draw_button(sim_x, btn_y, sim_w, control_h, Color{46,46,56,255}, sim_label, 0.8f);
    // draw right play/pause half
    draw_button(play_x, btn_y, play_w, control_h, Color{44,52,60,255}, pause_label, 1.0f);
    draw_button(lib_x, btn_y, lib_w, control_h, Color{46,54,52,255}, LABEL_MODULE_LIBRARY_SHORT, 0.85f);
    draw_button(menu_x, btn_y, menu_w, control_h, Color{50,50,62,255}, LABEL_MODULE_MENU, 0.85f);

    // Module action buttons: Clone / Clear / Destroy / Commit / Export (right-aligned)
    int module_btn_count = 5;
    int module_btn_w = nbw;
    int module_gap = 6;
    int btns_total_w = module_btn_count * (module_btn_w + module_gap) - module_gap;
    int btns_right = pause_x - module_gap;
    int btns_left = btns_right - btns_total_w;
    int bx_btn = btns_left;
    // Timing toggle button placed between pause and exec-mode group
    int timing_btn_w = nbw;
    int timing_x = btns_left - module_gap - timing_btn_w;
    bool timing_on = false;
    if (ThreadManager::global()) timing_on = ThreadManager::global()->timing_enabled();
    Color timing_col = timing_on ? Color{64,120,60,255} : Color{52,52,64,255};
    draw_button(timing_x, btn_y, timing_btn_w, control_h, timing_col, "TIME", 0.75f);
    draw_button(bx_btn, btn_y, module_btn_w, control_h, Color{52,52,64,255}, LABEL_MODULE_CLONE_SHORT, 0.9f); bx_btn += module_btn_w + module_gap;
    draw_button(bx_btn, btn_y, module_btn_w, control_h, Color{52,52,64,255}, LABEL_MODULE_CLEAR_SHORT, 0.9f); bx_btn += module_btn_w + module_gap;
    draw_button(bx_btn, btn_y, module_btn_w, control_h, Color{52,52,64,255}, LABEL_MODULE_DESTROY_SHORT, 0.9f); bx_btn += module_btn_w + module_gap;
    draw_button(bx_btn, btn_y, module_btn_w, control_h, Color{48,48,56,255}, "CMT", 0.9f); bx_btn += module_btn_w + module_gap;
    draw_button(bx_btn, btn_y, module_btn_w, control_h, Color{44,60,48,255}, "EXP", 0.9f);

    // Per-module exec-mode buttons (SEQ / POOL / SLIP / FREE) placed left of module action buttons
    int exec_mode_count = 4;
    int exec_mode_w = module_btn_w;
    int exec_mode_gap = 4;
    int exec_total_w = exec_mode_count * exec_mode_w + (exec_mode_count - 1) * exec_mode_gap;
    int bx_exec_left = btns_left - exec_total_w - module_gap;
    const char* exec_labels[4] = {"SEQ","POOL","SLIP","FREE"};
    for (int emi = 0; emi < exec_mode_count; ++emi) {
        int bx_e = bx_exec_left + emi * (exec_mode_w + exec_mode_gap);
        // draw base button
        draw_button(bx_e, btn_y, exec_mode_w, control_h, Color{46,46,56,255}, exec_labels[emi], 0.85f);
        // Render per-module selected state if module has a local override.
        bool local_selected = false;
        if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_exec_mode.size())) {
            int mval = ctx->module_exec_mode[module_idx];
            if (mval == emi) local_selected = true;
        }
        if (local_selected) {
            int y_top = btn_y;
            int y_bot = btn_y + control_h - 1;
                for (int xx = 0; xx < exec_mode_w; ++xx) {
                int xh = bx_e + xx;
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
    }

    cursor_y += kModuleControlRowH + kModuleTopGap;
    constexpr float kLedLabelScale = 0.7f;
    constexpr int kLedLabelGap = 6;
    // Four logical LED sets arranged as 2 columns x 2 rows: top row = "send", bottom row = "receive"
    const int full_cols = 2; // left/right
    int visible_total = ctx ? std::clamp(ctx->module_frame_pair_count, 1, kModuleExtraLedCount * full_cols) : kModuleExtraLedCount * full_cols; // total pair-columns across both sides
    int pair_count = (kModuleExtraLedRows + full_cols - 1) / full_cols; // logical label rows (send/receive)
    std::vector<const char*> led_labels_vec;
    led_labels_vec.reserve(static_cast<size_t>(pair_count));
    for (int gi = 0; gi < pair_count; ++gi) {
        // row 0 = send, row 1 = receive
        led_labels_vec.push_back((gi == 0) ? "send" : "receive");
    }
    std::vector<TextBitmap> label_bitmaps(static_cast<size_t>(pair_count));
    int label_w = 0;
    for (int i = 0; i < pair_count; ++i) {
        label_bitmaps[static_cast<size_t>(i)] = render_text_to_rgba(led_labels_vec[static_cast<size_t>(i)], kLedLabelScale, {190,190,205,255});
        label_w = std::max(label_w, label_bitmaps[static_cast<size_t>(i)].width);
    }
    const int label_pad = (label_w > 0) ? kLedLabelGap : 0;
    int led_row_h = kModuleLedRowH;
    const int grid_rows = pair_count;
    // reserve space for the numeric control on the left, then divide remaining width
    int ctrl_nbw = nbw;
    int ctrl_num_w = num_w;
    int ctrl_gap = gap;
    int ctrl_total_w = ctrl_nbw + ctrl_gap + ctrl_num_w + ctrl_gap + ctrl_nbw;
    int remaining_w = std::max(1, m.w - kModuleTopPadding * 2 - ctrl_total_w);
    int col_total_w = std::max(1, remaining_w / full_cols);
    int led_gap = 4;
    // visible per-side counts derived from total
    int visible_on_left = std::clamp(visible_total, 0, kModuleExtraLedCount);
    int visible_on_right = std::clamp(visible_total - kModuleExtraLedCount, 0, kModuleExtraLedCount);
    int max_visible_side = std::max(visible_on_left, visible_on_right);
    int led_area_inner_w = std::max(1, col_total_w - label_w - label_pad - kModuleTopPadding);
    int led_size = (led_area_inner_w - led_gap * std::max(0, max_visible_side - 1)) / std::max(1, max_visible_side);
    if (led_size < 8) {
        led_gap = 2;
        led_size = (led_area_inner_w - led_gap * std::max(0, max_visible_side - 1)) / std::max(1, max_visible_side);
    }
    led_size = std::max(4, std::min(led_size, led_row_h - 4));
    Color led_on{255,210,90,255};
    Color led_off{70,70,80,255};
    Color led_edge{20,20,28,255};
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_tables.size()) && ctx->module_tables[module_idx]) {
        GP_TableStyle st{};
        gp_table_get_style(ctx->module_tables[module_idx], &st);
        led_on = Color{st.led_on_rgba[0], st.led_on_rgba[1], st.led_on_rgba[2], st.led_on_rgba[3]};
        led_off = Color{st.led_off_rgba[0], st.led_off_rgba[1], st.led_off_rgba[2], st.led_off_rgba[3]};
        led_edge = Color{st.led_edge_rgba[0], st.led_edge_rgba[1], st.led_edge_rgba[2], st.led_edge_rgba[3]};
    }
    int total_rows = pair_count * full_cols;
    // draw single pair-count controls (left of left column) once
    {
        int group_x = sx + kModuleTopPadding;
        int nbw = control_h;
        int num_w = std::max(24, nbw * 2);
        int btn_y = cursor_y + (led_row_h - control_h) / 2;
        draw_button(group_x, btn_y, nbw, control_h, Color{52,52,64,255}, "-", 1.0f);
        // render count label
        auto bm_count = render_text_to_rgba(std::to_string(visible_total).c_str(), 0.9f, {230,230,235,255});
        if (!bm_count.pixels.empty()) {
            int tx = group_x + nbw + gap + (num_w - bm_count.width) / 2;
            int ty = btn_y + (control_h - bm_count.height) / 2;
            for (int yy = 0; yy < bm_count.height; ++yy) {
                int dst_y = ty + yy; if (dst_y < 0 || dst_y >= h) continue;
                for (int xx = 0; xx < bm_count.width; ++xx) {
                    int dst_x = tx + xx; if (dst_x < 0 || dst_x >= w) continue;
                    uint8_t* dst = out_rgba + dst_y * pitch + dst_x * 4;
                    const unsigned char* src = &bm_count.pixels[(yy * bm_count.width + xx) * 4];
                    float sa = src[3] / 255.0f;
                    if (sa >= 0.999f) { dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3]; }
                    else if (sa > 0.001f) {
                        for (int cch = 0; cch < 3; ++cch) dst[cch] = static_cast<uint8_t>(std::lround((src[cch]/255.0f * sa + dst[cch]/255.0f * (1.0f-sa)) * 255.0f));
                        dst[3] = 255;
                    }
                }
            }
        }
        draw_button(group_x + nbw + gap + num_w + gap, btn_y, nbw, control_h, Color{52,52,64,255}, "+", 1.0f);
    }

    for (int row = 0; row < total_rows; ++row) {
        int grid_row = row / full_cols;
        int grid_col = row % full_cols;
        int led_row_y = cursor_y + grid_row * (kModuleLedRowH + kModuleTopGap);
        // draw label only for left column (grid_col == 0) and reuse per-grid_row bitmap
        if (grid_col == 0) {
            const auto &bm = label_bitmaps[static_cast<size_t>(grid_row)];
            int label_x = sx + kModuleTopPadding + ctrl_total_w + grid_col * col_total_w;
            if (!bm.pixels.empty()) {
                int label_y = led_row_y + (led_row_h - bm.height) / 2;
                for (int yy = 0; yy < bm.height; ++yy) {
                    int dst_y = label_y + yy;
                    if (dst_y < 0 || dst_y >= h) continue;
                    for (int xx = 0; xx < bm.width; ++xx) {
                        int dst_x = label_x + xx;
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
        // left-aligned groups: compute left base then place right group immediately after left group's LEDs
        int left_base_x = sx + kModuleTopPadding + ctrl_total_w + /*label space*/ (label_w + label_pad);
        int left_led_total = led_size * visible_on_left + led_gap * std::max(0, visible_on_left - 1);
        int visible_here = (grid_col == 0) ? visible_on_left : visible_on_right;
        int led_start_x = (grid_col == 0) ? left_base_x : (left_base_x + left_led_total);
        int led_y = led_row_y + (led_row_h - led_size) / 2;
        for (int i = 0; i < visible_here; ++i) {
            int lx = led_start_x + i * (led_size + led_gap);
            const GP_TableCell* cell = module_frame_led_cell(ctx, module_idx, row, i);
            bool on = cell && ((cell->flags & 0x1u) != 0u);
            bool active = cell && ((static_cast<uint32_t>(cell->reserved0) & 0x1u) != 0u);
            Color fill = (on || active) ? led_on : led_off;
            int cx = lx + led_size / 2;
            int cy = led_y + led_size / 2;
            int radius = std::max(2, led_size / 2 - 1);
            draw_circle(out_rgba, w, h, pitch, cx, cy, radius, led_edge);
            draw_circle(out_rgba, w, h, pitch, cx, cy, std::max(1, radius - 1), fill);
        }
    }
}

template <typename Fn>
static void for_each_module_frame_led(const GP_CanvasContextImpl* ctx, int module_idx, const GP_CanvasModuleDesc& m, Fn&& fn) {
    if (!ctx) return;
    ModuleLayout layout = module_layout_for(ctx, module_idx, m);
    if (layout.top_h <= 0) return;
    int cursor_y = kModuleTopPadding;
    cursor_y += kModuleTitleRowH;
    cursor_y += kModuleThumbRowH + kModuleTopGap;
    cursor_y += kModuleControlRowH + kModuleTopGap;
    constexpr float kLedLabelScale = 0.7f;
    constexpr int kLedLabelGap = 6;
    // prepare label bitmaps for logical rows (send/receive)
    const int full_cols = 2; // left/right columns
    int visible_total = ctx ? std::clamp(ctx->module_frame_pair_count, 1, kModuleExtraLedCount * full_cols) : kModuleExtraLedCount * full_cols;
    int pair_count = (kModuleExtraLedRows + full_cols - 1) / full_cols; // logical label rows (send/receive)
    std::vector<TextBitmap> label_bitmaps(static_cast<size_t>(pair_count));
    int label_w = 0;
    for (int i = 0; i < pair_count; ++i) {
        const char* lbl = (i == 0) ? "send" : "receive";
        label_bitmaps[static_cast<size_t>(i)] = render_text_to_rgba(lbl, kLedLabelScale, {190,190,205,255});
        label_w = std::max(label_w, label_bitmaps[static_cast<size_t>(i)].width);
    }
    const int label_pad = (label_w > 0) ? kLedLabelGap : 0;
    int led_row_h = kModuleLedRowH;
    // estimate control width same as main raster so hitboxes align with drawn controls
    int nbw = std::max(1, kModuleControlRowH - 2);
    int num_w = std::max(24, nbw * 2);
    int gap2 = 6;
    int ctrl_total_w = nbw + gap2 + num_w + gap2 + nbw;
    int remaining_w2 = std::max(1, m.w - kModuleTopPadding * 2 - ctrl_total_w);
    int col_total_w = std::max(1, remaining_w2 / full_cols);
    int led_gap = 4;
    int visible_on_left = std::clamp(visible_total, 0, kModuleExtraLedCount);
    int visible_on_right = std::clamp(visible_total - kModuleExtraLedCount, 0, kModuleExtraLedCount);
    int max_visible_side = std::max(visible_on_left, visible_on_right);
    int led_area_inner_w = std::max(1, col_total_w - label_w - label_pad - kModuleTopPadding);
    int led_size = (led_area_inner_w - led_gap * std::max(0, max_visible_side - 1)) / std::max(1, max_visible_side);
    if (led_size < 8) {
        led_gap = 2;
        led_size = (led_area_inner_w - led_gap * std::max(0, max_visible_side - 1)) / std::max(1, max_visible_side);
    }
    led_size = std::max(4, std::min(led_size, led_row_h - 4));
    int total_rows = pair_count * full_cols;
    for (int row = 0; row < total_rows; ++row) {
        int grid_row = row / full_cols;
        int grid_col = row % full_cols;
        int led_row_y = cursor_y + grid_row * (kModuleLedRowH + kModuleTopGap);
        // left-aligned groups for hitbox enumeration: match raster layout
        int left_base_x = kModuleTopPadding + ctrl_total_w + (label_w + label_pad);
        int left_led_total = led_size * visible_on_left + led_gap * std::max(0, visible_on_left - 1);
        int visible_here = (grid_col == 0) ? visible_on_left : visible_on_right;
        int led_start_x = (grid_col == 0) ? left_base_x : (left_base_x + left_led_total);
        int led_y = led_row_y + (led_row_h - led_size) / 2;
        for (int i = 0; i < visible_here; ++i) {
            int lx = led_start_x + i * (led_size + led_gap);
            fn(row, i, lx, led_y, lx + led_size, led_y + led_size);
        }
    }
}

static GP_TableHitBox make_module_frame_led_hitbox(int row, int idx, int x0, int y0, int x1, int y1) {
    GP_TableHitBox hb{};
    hb.x0 = x0;
    hb.y0 = y0;
    hb.x1 = x1;
    hb.y1 = y1;
    hb.cell_kind = GP_TABLE_CELL_LEDS;
    hb.part = GP_TABLE_HIT_LED;
    const int full_cols = 2;
    int grid_row = row / full_cols;
    int grid_col = row % full_cols;
    hb.row_idx = (grid_row == 0) ? kModuleFrameRowSend : kModuleFrameRowReceive;
    hb.col_idx = (grid_col == 0) ? kModuleColLeftLed : kModuleColRightLed;
    int base = kModuleFrameContactBase + row * kModuleExtraLedCount;
    hb.aux0 = base + idx;
    hb.aux1 = 0;
    return hb;
}

static bool find_module_frame_led_hit(const GP_CanvasContextImpl* ctx, int module_idx, const GP_CanvasModuleDesc& m, int lx, int ly, GP_TableHitBox* out_hit) {
    if (!out_hit) return false;
    bool hit = false;
    for_each_module_frame_led(ctx, module_idx, m, [&](int row, int idx, int x0, int y0, int x1, int y1) {
        if (hit) return;
        if (lx >= x0 && lx < x1 && ly >= y0 && ly < y1) {
            *out_hit = make_module_frame_led_hitbox(row, idx, x0, y0, x1, y1);
            hit = true;
        }
    });
    return hit;
}

static CanvasBounds compute_canvas_bounds(const GP_CanvasContextImpl* ctx) {
    CanvasBounds b;
    if (!ctx) return b;
    if (!ctx->modules.empty()) {
        b.has_any = true;
        b.min_x = ctx->modules.front().x;
        b.max_x = ctx->modules.front().x + ctx->modules.front().w;
        b.min_y = ctx->modules.front().y;
        b.max_y = ctx->modules.front().y + ctx->modules.front().h;
        for (const auto& m : ctx->modules) {
            b.min_x = std::min(b.min_x, m.x);
            b.max_x = std::max(b.max_x, m.x + m.w);
            b.min_y = std::min(b.min_y, m.y);
            b.max_y = std::max(b.max_y, m.y + m.h);
        }
    } else {
        b.min_x = 0; b.max_x = ctx->width;
        b.min_y = 0; b.max_y = ctx->height;
    }
    return b;
}

static uint32_t lookup_molex_hash(const GP_CanvasContextImpl* ctx, int module_idx, bool is_input, int contact_idx) {
    if (!ctx || contact_idx < 0) return 0;
    const auto &layout = is_input ? ctx->module_input_layout : ctx->module_output_layout;
    if (module_idx < 0 || module_idx >= static_cast<int>(layout.size())) return 0;
    const MolexLayoutInfo &info = layout[module_idx];
    int side_idx = resolve_side_contact_index(ctx, module_idx, is_input, contact_idx);
    if (side_idx < 0 || side_idx >= static_cast<int>(info.hashes.size())) return 0;
    return info.hashes[side_idx];
}

static void clamp_offset_to_bounds(GP_CanvasContextImpl* ctx, const CanvasBounds& b) {
    if (!ctx) return;
    int view_w = std::max(1, ctx->width);
    int view_h = std::max(1, ctx->height);
    int min_off_x = std::min(0, b.min_x);
    int max_off_x = std::max(min_off_x, b.max_x - view_w);
    int min_off_y = std::min(0, b.min_y);
    int max_off_y = std::max(min_off_y, b.max_y - view_h);
    ctx->offset_x = std::clamp(ctx->offset_x, min_off_x, max_off_x);
    ctx->offset_y = std::clamp(ctx->offset_y, min_off_y, max_off_y);
    ctx->scroll_x_needed = (b.min_x < ctx->offset_x) || (b.max_x > ctx->offset_x + view_w);
    ctx->scroll_y_needed = (b.min_y < ctx->offset_y) || (b.max_y > ctx->offset_y + view_h);
}

static void sync_container_scroll(GP_CanvasContextImpl* ctx, const CanvasBounds& b) {
    if (!ctx || !ctx->container_table) return;
    float fx = 0.0f, fy = 0.0f;
    // tolerate older tables that only expose vertical scroll
    if (!gp_table_get_scroll_fraction_xy(ctx->container_table, &fx, &fy)) {
        gp_table_get_scroll_fraction(ctx->container_table, &fy);
        fx = 0.0f;
    }
    fx = std::clamp(fx, 0.0f, 1.0f);
    fy = std::clamp(fy, 0.0f, 1.0f);
    int view_w = std::max(1, ctx->width);
    int view_h = std::max(1, ctx->height);
    int min_off_x = std::min(0, b.min_x);
    int max_off_x = std::max(min_off_x, b.max_x - view_w);
    int min_off_y = std::min(0, b.min_y);
    int max_off_y = std::max(min_off_y, b.max_y - view_h);
    ctx->offset_x = min_off_x + static_cast<int>(std::lround(fx * float(max_off_x - min_off_x)));
    ctx->offset_y = min_off_y + static_cast<int>(std::lround(fy * float(max_off_y - min_off_y)));
    ctx->offset_x = std::clamp(ctx->offset_x, min_off_x, max_off_x);
    ctx->offset_y = std::clamp(ctx->offset_y, min_off_y, max_off_y);
    float out_fx = (max_off_x == min_off_x) ? 0.0f : float(ctx->offset_x - min_off_x) / float(max_off_x - min_off_x);
    float out_fy = (max_off_y == min_off_y) ? 0.0f : float(ctx->offset_y - min_off_y) / float(max_off_y - min_off_y);
    gp_table_set_scroll_fraction_xy(ctx->container_table, out_fx, out_fy);
}

static CanvasBounds update_canvas_scroll_state(GP_CanvasContextImpl* ctx, bool pull_from_container) {
    CanvasBounds b = compute_canvas_bounds(ctx);
    if (pull_from_container) sync_container_scroll(ctx, b);
    clamp_offset_to_bounds(ctx, b);
    if (ctx && ctx->container_table) {
        int view_w = std::max(1, ctx->width);
        int view_h = std::max(1, ctx->height);
        int min_off_x = std::min(0, b.min_x);
        int max_off_x = std::max(min_off_x, b.max_x - view_w);
        int min_off_y = std::min(0, b.min_y);
        int max_off_y = std::max(min_off_y, b.max_y - view_h);
        float fx = (max_off_x == min_off_x) ? 0.0f : float(ctx->offset_x - min_off_x) / float(max_off_x - min_off_x);
        float fy = (max_off_y == min_off_y) ? 0.0f : float(ctx->offset_y - min_off_y) / float(max_off_y - min_off_y);
        gp_table_set_scroll_fraction_xy(ctx->container_table, fx, fy);
    }
    return b;
}

struct InputRayLight {
    float cx = 0.0f;
    float cy = 0.0f;
    float radius = 1.0f;
    float intensity = 0.0f;
    int contact_idx = -1;
};

static void ensure_module_bg_storage(GP_CanvasContextImpl::ModuleBg& bg, int w, int h, int oversample) {
    const int mw = std::max(0, w);
    const int mh = std::max(0, h);
    const int os = std::max(1, oversample);
    const int mw_hi = mw * os;
    const int mh_hi = mh * os;

    const size_t px = static_cast<size_t>(mw) * static_cast<size_t>(mh);
    const size_t px_hi = static_cast<size_t>(std::max(0, mw_hi)) * static_cast<size_t>(std::max(0, mh_hi));

    // Frame accum is cleared by the caller each render; preserve temporal across frames.
    if (bg.accum.size() != px) bg.accum.assign(px, 0.0f);
    if (bg.temporal.size() != px) bg.temporal.assign(px, 0.0f);
    if (bg.scratch.size() != px * 4u) bg.scratch.assign(px * 4u, 0);
    if (bg.layer.size() != px_hi * 4u) bg.layer.assign(px_hi * 4u, 0);

    if (!bg.ray && mw_hi > 0 && mh_hi > 0) {
        bg.ray = raytrace2d_create(mw_hi, mh_hi);
    } else if (bg.ray && mw_hi > 0 && mh_hi > 0) {
        raytrace2d_resize(bg.ray, mw_hi, mh_hi);
    }
}

static void blit_module_buffer(uint8_t* out_rgba, int w, int h, int pitch, int sx, int sy, int mw, int mh, const std::vector<uint8_t>& src) {
    if (!out_rgba || src.empty() || mw <= 0 || mh <= 0) return;
    for (int yy = 0; yy < mh; ++yy) {
        int dst_y = sy + yy;
        if (dst_y < 0 || dst_y >= h) continue;
        uint8_t* dst_row = out_rgba + dst_y * pitch;
        const uint8_t* src_row = src.data() + static_cast<size_t>(yy) * static_cast<size_t>(mw) * 4u;
        int dst_x0 = sx;
        int src_x0 = 0;
        int copy_w = mw;
        if (dst_x0 < 0) { src_x0 = -dst_x0; copy_w -= src_x0; dst_x0 = 0; }
        copy_w = std::min(copy_w, std::max(0, w - dst_x0));
        if (copy_w <= 0) continue;
        std::memcpy(dst_row + dst_x0 * 4, src_row + static_cast<size_t>(src_x0) * 4u, static_cast<size_t>(copy_w) * 4u);
    }
}

static void blit_module_buffer_alpha(uint8_t* out_rgba, int w, int h, int pitch, int sx, int sy, int mw, int mh, const std::vector<uint8_t>& src, uint8_t global_alpha) {
    if (!out_rgba || src.empty() || mw <= 0 || mh <= 0) return;
    if (global_alpha >= 255) {
        blit_module_buffer(out_rgba, w, h, pitch, sx, sy, mw, mh, src);
        return;
    }
    const float g = global_alpha / 255.0f;
    for (int yy = 0; yy < mh; ++yy) {
        int dst_y = sy + yy;
        if (dst_y < 0 || dst_y >= h) continue;
        uint8_t* dst_row = out_rgba + dst_y * pitch;
        const uint8_t* src_row = src.data() + static_cast<size_t>(yy) * static_cast<size_t>(mw) * 4u;
        int dst_x0 = sx;
        int src_x0 = 0;
        int copy_w = mw;
        if (dst_x0 < 0) { src_x0 = -dst_x0; copy_w -= src_x0; dst_x0 = 0; }
        copy_w = std::min(copy_w, std::max(0, w - dst_x0));
        if (copy_w <= 0) continue;
        uint8_t* dst = dst_row + dst_x0 * 4;
        const uint8_t* srcp = src_row + static_cast<size_t>(src_x0) * 4u;
        for (int xx = 0; xx < copy_w; ++xx) {
            const uint8_t sa = srcp[3];
            const float a = (sa / 255.0f) * g;
            if (a >= 0.999f) {
                dst[0] = srcp[0];
                dst[1] = srcp[1];
                dst[2] = srcp[2];
                dst[3] = 255;
            } else if (a > 0.001f) {
                const float inv = 1.0f - a;
                dst[0] = static_cast<uint8_t>(std::lround(srcp[0] * a + dst[0] * inv));
                dst[1] = static_cast<uint8_t>(std::lround(srcp[1] * a + dst[1] * inv));
                dst[2] = static_cast<uint8_t>(std::lround(srcp[2] * a + dst[2] * inv));
                dst[3] = 255;
            }
            dst += 4;
            srcp += 4;
        }
    }
}

// Blend source using its own alpha (no global alpha scaling).
static void blit_module_buffer_srcalpha(uint8_t* out_rgba, int w, int h, int pitch, int sx, int sy, int mw, int mh, const std::vector<uint8_t>& src) {
    if (!out_rgba || src.empty() || mw <= 0 || mh <= 0) return;
    for (int yy = 0; yy < mh; ++yy) {
        int dst_y = sy + yy;
        if (dst_y < 0 || dst_y >= h) continue;
        uint8_t* dst_row = out_rgba + dst_y * pitch;
        const uint8_t* src_row = src.data() + static_cast<size_t>(yy) * static_cast<size_t>(mw) * 4u;
        int dst_x0 = sx;
        int src_x0 = 0;
        int copy_w = mw;
        if (dst_x0 < 0) { src_x0 = -dst_x0; copy_w -= src_x0; dst_x0 = 0; }
        copy_w = std::min(copy_w, std::max(0, w - dst_x0));
        if (copy_w <= 0) continue;
        uint8_t* dst = dst_row + dst_x0 * 4;
        const uint8_t* srcp = src_row + static_cast<size_t>(src_x0) * 4u;
        for (int xx = 0; xx < copy_w; ++xx) {
            const float a = srcp[3] / 255.0f;
            if (a >= 0.999f) {
                dst[0] = srcp[0];
                dst[1] = srcp[1];
                dst[2] = srcp[2];
                dst[3] = 255;
            } else if (a > 0.001f) {
                const float inv = 1.0f - a;
                dst[0] = static_cast<uint8_t>(std::lround(srcp[0] * a + dst[0] * inv));
                dst[1] = static_cast<uint8_t>(std::lround(srcp[1] * a + dst[1] * inv));
                dst[2] = static_cast<uint8_t>(std::lround(srcp[2] * a + dst[2] * inv));
                dst[3] = 255;
            }
            dst += 4;
            srcp += 4;
        }
    }
}

static void blit_module_buffer_alpha_scaled(uint8_t* out_rgba, int w, int h, int pitch, int dst_x, int dst_y, int dst_w, int dst_h, const std::vector<uint8_t>& src, int src_w, int src_h, uint8_t global_alpha) {
    if (!out_rgba || src.empty() || dst_w <= 0 || dst_h <= 0 || src_w <= 0 || src_h <= 0) return;
    const float g = std::clamp(global_alpha, uint8_t(0), uint8_t(255)) / 255.0f;
    for (int dy = 0; dy < dst_h; ++dy) {
        int y = dst_y + dy;
        if (y < 0 || y >= h) continue;
        float syf = (dst_h > 1) ? (float(dy) / float(dst_h - 1)) * float(src_h - 1) : 0.0f;
        int syi = std::clamp(int(std::round(syf)), 0, src_h - 1);
        const uint8_t* src_row = src.data() + static_cast<size_t>(syi) * static_cast<size_t>(src_w) * 4u;
        for (int dx = 0; dx < dst_w; ++dx) {
            int x = dst_x + dx;
            if (x < 0 || x >= w) continue;
            float sxf = (dst_w > 1) ? (float(dx) / float(dst_w - 1)) * float(src_w - 1) : 0.0f;
            int sxi = std::clamp(int(std::round(sxf)), 0, src_w - 1);
            const uint8_t* sp = src_row + static_cast<size_t>(sxi) * 4u;
            float sa = (sp[3] / 255.0f) * g;
            if (sa <= 0.0f) continue;
            float inv = 1.0f - sa;
            uint8_t* dp = out_rgba + y * pitch + x * 4;
            for (int c = 0; c < 3; ++c) {
                float s = sp[c] / 255.0f;
                float d = dp[c] / 255.0f;
                dp[c] = static_cast<uint8_t>(std::lround((s * sa + d * inv) * 255.0f));
            }
            float da = dp[3] / 255.0f;
            dp[3] = static_cast<uint8_t>(std::lround((sa + da * inv) * 255.0f));
        }
    }
}

static void render_module_raytrace_bg(GP_CanvasContextImpl::ModuleBg& bg, int mw, int mh, const std::vector<InputRayLight>& lights) {
    const int os = std::max(1, bg.oversample);
    const int mw_hi = std::max(0, mw) * os;
    const int mh_hi = std::max(0, mh) * os;
    ensure_module_bg_storage(bg, mw, mh, os);
    if (!bg.ray) return;
    std::fill(bg.accum.begin(), bg.accum.end(), 0.0f);

    // Temporal accumulation: decay per redraw (not wall-clock).
    const float temporal_decay = std::clamp(bg.temporal_decay, 0.0f, 1.0f);
    const float temporal_max = std::max(0.0f, bg.temporal_max);

    if (!lights.empty() && mw > 0 && mh > 0 && mw_hi > 0 && mh_hi > 0) {
        ++bg.temporal_frame;
        raytrace2d_set_room(bg.ray, 0.0f, 0.0f, static_cast<float>(mw), static_cast<float>(mh));
        raytrace2d_set_params(bg.ray, bg.rays, bg.reflections, bg.blur_sigma);
        raytrace2d_set_attenuation(bg.ray, bg.ray_bounce_decay, bg.ray_air_decay);
        for (const auto& light : lights) {
            if (light.intensity <= 0.0f) continue;
            uint32_t seed = static_cast<uint32_t>(light.contact_idx + 1);
            seed = seed * 0x9E3779B1u;
            seed ^= (bg.temporal_frame * 0x85EBCA6Bu) + 0xC2B2AE35u;
            if (seed == 0) seed = 1u;
            raytrace2d_set_seed(bg.ray, seed);
            raytrace2d_set_light(bg.ray, light.cx, light.cy, light.radius);
            if (!raytrace2d_render_rgba(bg.ray, bg.layer.data(), static_cast<int32_t>(bg.layer.size()))) continue;

            // Downsample (nearest-neighbor) from oversampled buffer and accumulate.
            for (int y = 0; y < mh; ++y) {
                const int hy = y * os;
                const size_t row_hi = static_cast<size_t>(hy) * static_cast<size_t>(mw_hi);
                const size_t row_lo = static_cast<size_t>(y) * static_cast<size_t>(mw);
                for (int x = 0; x < mw; ++x) {
                    const int hx = x * os;
                    const size_t idx_hi = (row_hi + static_cast<size_t>(hx)) * 4u;
                    const size_t idx_lo = row_lo + static_cast<size_t>(x);
                    float v = bg.layer[idx_hi + 0] / 255.0f;
                    bg.accum[idx_lo] += v * light.intensity;
                }
            }
        }

        // Accumulate with decay and clamp to a max intensity.
        for (size_t i = 0; i < bg.temporal.size(); ++i) {
            float v = bg.temporal[i] * temporal_decay + bg.accum[i];
            bg.temporal[i] = std::min(temporal_max, v);
        }
    } else {
        // No lights: decay the existing accumulated field.
        for (size_t i = 0; i < bg.temporal.size(); ++i) {
            bg.temporal[i] *= temporal_decay;
        }
    }

    const float exposure = std::max(0.0f, bg.ray_exposure);
    const uint8_t base_r = 40, base_g = 40, base_b = 50;
    const float add = 160.0f;
    for (int y = 0; y < mh; ++y) {
        for (int x = 0; x < mw; ++x) {
            size_t idx = static_cast<size_t>(y) * static_cast<size_t>(mw) + static_cast<size_t>(x);
            float t = std::clamp(bg.temporal[idx] * exposure, 0.0f, 1.0f);
            uint8_t r = static_cast<uint8_t>(std::clamp(base_r + t * add, 0.0f, 255.0f));
            uint8_t g = static_cast<uint8_t>(std::clamp(base_g + t * add, 0.0f, 255.0f));
            uint8_t b = static_cast<uint8_t>(std::clamp(base_b + t * add, 0.0f, 255.0f));
            size_t o = idx * 4u;
            bg.scratch[o + 0] = r;
            bg.scratch[o + 1] = g;
            bg.scratch[o + 2] = b;
            bg.scratch[o + 3] = 255;
        }
    }
}

static const char* tool_label(ModuleToolKind tool) {
    switch (tool) {
        case ModuleToolKind::Add: return LABEL_TOOL_ADD;
        case ModuleToolKind::Subtract: return LABEL_TOOL_SUB;
        case ModuleToolKind::Multiply: return LABEL_TOOL_MUL;
        case ModuleToolKind::Divide: return LABEL_TOOL_DIV;
        case ModuleToolKind::Modulo: return LABEL_TOOL_MOD;
        case ModuleToolKind::KeyboardListener: return LABEL_TOOL_KEYBOARD;
        case ModuleToolKind::MouseListener: return LABEL_TOOL_MOUSE;
        case ModuleToolKind::StackDisplay: return LABEL_TOOL_STACK;
        case ModuleToolKind::Clone: return LABEL_TOOL_CLONE;
        case ModuleToolKind::RectRgba: return LABEL_TOOL_RECT;
        case ModuleToolKind::TableNumber: return LABEL_TOOL_NUMBER;
        case ModuleToolKind::None:
        default:
            return "";
    }
}

static std::string shared_library_extension() {
#if defined(_WIN32)
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
}

static bool shared_library_matches_tool(const std::string& filename, const std::string& tool_id) {
    const std::string ext = shared_library_extension();
    if (filename == tool_id + ext) return true;
    if (filename == "lib" + tool_id + ext) return true;
    if (ext == ".so") {
        const std::string prefix = "lib" + tool_id + ".so";
        if (filename.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

static bool parse_tool_kind_from_id(const std::string& tool_id, ModuleToolKind& out_kind) {
    constexpr const char* kPrefix = "tool_";
    if (tool_id.rfind(kPrefix, 0) != 0) return false;
    std::string digits = tool_id.substr(std::strlen(kPrefix));
    if (digits.empty()) return false;
    int value = 0;
    for (char ch : digits) {
        if (ch < '0' || ch > '9') return false;
        value = value * 10 + (ch - '0');
    }
    const int max_kind = static_cast<int>(ModuleToolKind::FontRenderer);
    if (value <= 0 || value > max_kind) return false;
    out_kind = static_cast<ModuleToolKind>(value);
    return true;
}

// Clear module frame pointers and action bindings for a module without destroying it.
static void canvas_clear_module_bindings(GP_CanvasContextImpl* ctx, int module_idx) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_frame_links.size())) return;
    {
        std::lock_guard<std::mutex> lk(ctx->action_subscribers_mu);
        for (auto it = ctx->action_port_bindings.begin(); it != ctx->action_port_bindings.end(); ) {
            auto &vec = it->second;
            for (auto vit = vec.begin(); vit != vec.end(); ) {
                if (vit->module_idx == module_idx) {
                    vit = vec.erase(vit);
                } else {
                    ++vit;
                }
            }
            if (vec.empty()) it = ctx->action_port_bindings.erase(it); else ++it;
        }
    }
    auto &links = ctx->module_frame_links[static_cast<size_t>(module_idx)];
    for (int row = 0; row < kModuleExtraLedRows; ++row) {
        for (int li = 0; li < kModuleExtraLedCount; ++li) {
            links.ptrs[static_cast<size_t>(row)][static_cast<size_t>(li)] = nullptr;
        }
    }
}

static std::string canvas_module_label(const GP_CanvasModuleDesc& desc) {
    std::string label(desc.label, desc.label + sizeof(desc.label));
    size_t null_pos = label.find('\0');
    if (null_pos != std::string::npos) label.resize(null_pos);
    return label;
}

static std::string module_library_manifest_path(const std::string& module_id) {
    namespace fs = std::filesystem;
    fs::path root = gp_module_library_default_root();
    fs::path manifest = root / "serialized" / (module_id + std::string(".modlib"));
    return manifest.generic_string();
}

static bool canvas_build_module_library_for_module(GP_CanvasContextImpl* ctx,
                                                  int module_idx,
                                                  bool convert_to_tool,
                                                  GP_ModuleLibrary* out_lib) {
    if (!ctx || !out_lib) return false;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return false;
    ensure_module_row_order(ctx, module_idx);
    if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) return false;

    GP_ModuleLibrary lib{};
    lib.root_dir = gp_module_library_default_root();

    std::map<int, GP_ModuleLibraryTool> tool_registry_by_kind;
    std::map<std::string, GP_ModuleLibraryTool> plugin_registry_by_id;
    auto normalize_source = [&](const std::string& source_path) -> std::string {
        if (source_path.empty()) return {};
        namespace fs = std::filesystem;
        try {
            fs::path src = fs::path(source_path);
            if (!src.is_absolute()) return src.generic_string();
            fs::path root = fs::absolute(fs::path(lib.root_dir));
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
    const auto &rows = ctx->module_io_rows[module_idx];
    for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
        const auto &row = rows[row_idx];
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
                tool.source_path = gp_module_library_tool_source_path(std::string(), tool.id);
                tool_registry_by_kind.emplace(static_cast<int>(tool_kind), std::move(tool));
            }
        } else if (row.tool_origin == ModuleToolOrigin::Plugin && !row.plugin_id.empty()) {
            if (plugin_registry_by_id.find(row.plugin_id) == plugin_registry_by_id.end()) {
                GP_ModuleLibraryTool tool{};
                tool.kind = ModuleToolKind::None;
                tool.id = row.plugin_id;
                const auto *entry = tool_registry_global().find(row.plugin_id);
                tool.name = entry ? entry->name : row.plugin_id;
                if (entry) {
                    tool.source_path = normalize_source(entry->source_path);
                }
                plugin_registry_by_id.emplace(tool.id, std::move(tool));
            }
        }
    }

    GP_ModuleLibraryModule module{};
    module.module_idx = module_idx;
    module.id = gp_module_library_module_id(module_idx);
    module.label = canvas_module_label(ctx->modules[module_idx]);
    module.serialized_path = gp_module_library_module_serialized_path(std::string(), module.id);
    module.source_path = gp_module_library_module_source_path(std::string(), module.id);
    module.convert_to_tool = convert_to_tool;
    module.tool_caps = 0;
    module.input_count = 0;
    module.output_count = 0;

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
        GP_ModuleToolInstance instance{};
        instance.row_idx = static_cast<int>(row_idx);
        instance.attachment_count = row.attachment_count;
        if (row.tool_origin == ModuleToolOrigin::Builtin) {
            ModuleToolKind tool_kind = row.tool;
            if (tool_kind == ModuleToolKind::None) continue;
            instance.tool_id = gp_module_library_tool_id(tool_kind);
            module.tool_instances.push_back(std::move(instance));
        } else if (row.tool_origin == ModuleToolOrigin::Plugin && !row.plugin_id.empty()) {
            instance.tool_id = row.plugin_id;
            module.tool_instances.push_back(std::move(instance));
        }
    }

    lib.modules.push_back(std::move(module));
    for (auto &entry : tool_registry_by_kind) {
        lib.tool_registry.push_back(std::move(entry.second));
    }
    for (auto &entry : plugin_registry_by_id) {
        lib.tool_registry.push_back(std::move(entry.second));
    }

    *out_lib = std::move(lib);
    return true;
}

static bool canvas_write_module_library_manifest(GP_CanvasContextImpl* ctx, int module_idx) {
    GP_ModuleLibrary lib{};
    if (!canvas_build_module_library_for_module(ctx, module_idx, false, &lib)) return false;
    const std::string module_id = gp_module_library_module_id(module_idx);
    const std::string manifest_path = module_library_manifest_path(module_id);
    try {
        std::filesystem::path p = manifest_path;
        if (!p.empty()) std::filesystem::create_directories(p.parent_path());
    } catch (...) {
        return false;
    }
    return gp_module_library_write_to_file(lib, manifest_path.c_str()) != 0;
}

static void canvas_refresh_module_library(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    namespace fs = std::filesystem;
    ctx->module_library_ids.clear();
    ctx->module_library_labels.clear();
    ctx->module_library_serialized.clear();
    ctx->module_library_manifests.clear();

    fs::path root = gp_module_library_default_root();
    fs::path ser_dir = root / "serialized";
    if (!fs::exists(ser_dir)) return;

    std::vector<std::string> ids;
    std::vector<std::string> labels;
    std::vector<std::string> serialized;
    std::vector<std::string> manifests;

    for (const auto &entry : fs::directory_iterator(ser_dir, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) continue;
        const fs::path &path = entry.path();
        if (path.extension() != ".gpmod") continue;
        std::string id = path.stem().string();
        std::string manifest = module_library_manifest_path(id);
        std::string label = id;
        if (fs::exists(manifest)) {
            GP_ModuleLibrary lib{};
            if (gp_module_library_read_from_file(manifest.c_str(), &lib)) {
                for (const auto &mod : lib.modules) {
                    if (mod.id == id) {
                        if (!mod.label.empty()) label = mod.label;
                        break;
                    }
                }
            }
        }
        ids.push_back(id);
        labels.push_back(label);
        serialized.push_back(path.generic_string());
        manifests.push_back(fs::exists(manifest) ? manifest : std::string());
    }

    // stable order by id
    std::vector<size_t> order(ids.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return ids[a] < ids[b];
    });
    for (size_t idx : order) {
        ctx->module_library_ids.push_back(ids[idx]);
        ctx->module_library_labels.push_back(labels[idx]);
        ctx->module_library_serialized.push_back(serialized[idx]);
        ctx->module_library_manifests.push_back(manifests[idx]);
    }
}

static std::vector<ModuleToolKind> discover_compiled_plugin_tools() {
    namespace fs = std::filesystem;
    std::vector<ModuleToolKind> out;
    fs::path root = gp_module_library_default_root();
    fs::path tool_dir = root / "source" / "tools";
    if (!fs::exists(tool_dir)) return out;

    std::vector<std::string> tool_ids;
    for (const auto& entry : fs::directory_iterator(tool_dir, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) continue;
        const fs::path& path = entry.path();
        if (path.extension() != ".cpp") continue;
        tool_ids.push_back(path.stem().string());
    }
    std::cerr << "DEBUG: discover_compiled_plugin_tools: found tool source ids: ";
    for (const auto &tid : tool_ids) std::cerr << tid << ",";
    std::cerr << "\n";
    if (tool_ids.empty()) return out;

    std::vector<std::string> shared_libs;
    if (fs::exists(root)) {
        for (const auto& entry : fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) continue;
            shared_libs.push_back(entry.path().filename().string());
        }
    }

    for (const auto& tool_id : tool_ids) {
        ModuleToolKind kind = ModuleToolKind::None;
        if (!parse_tool_kind_from_id(tool_id, kind)) continue;
        bool compiled = false;
        for (const auto& filename : shared_libs) {
            if (shared_library_matches_tool(filename, tool_id)) {
                compiled = true;
                break;
            }
        }
            if (compiled) std::cerr << "DEBUG: discover_compiled_plugin_tools: tool_id='" << tool_id << "' matched on-disk filename\n";
        if (compiled) out.push_back(kind);
    }

    std::sort(out.begin(), out.end(), [](ModuleToolKind a, ModuleToolKind b) {
        return static_cast<int>(a) < static_cast<int>(b);
    });
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

struct ToolMenuItem {
    int action_id = 0;
    const char* label = "";
    ModuleToolKind tool = ModuleToolKind::None;
};

static const ToolMenuItem kToolMenuItems[] = {
    { CANVAS_ACT_MENU_TOOL_ADD, LABEL_TOOL_ADD, ModuleToolKind::Add },
    { CANVAS_ACT_MENU_TOOL_SUB, LABEL_TOOL_SUB, ModuleToolKind::Subtract },
    { CANVAS_ACT_MENU_TOOL_MUL, LABEL_TOOL_MUL, ModuleToolKind::Multiply },
    { CANVAS_ACT_MENU_TOOL_DIV, LABEL_TOOL_DIV, ModuleToolKind::Divide },
    { CANVAS_ACT_MENU_TOOL_MOD, LABEL_TOOL_MOD, ModuleToolKind::Modulo },
    { CANVAS_ACT_MENU_TOOL_ALLOCATOR, LABEL_TOOL_ALLOCATOR, ModuleToolKind::TensorAllocator },
    { CANVAS_ACT_MENU_TOOL_KEYBOARD, LABEL_TOOL_KEYBOARD, ModuleToolKind::KeyboardListener },
    { CANVAS_ACT_MENU_TOOL_MOUSE, LABEL_TOOL_MOUSE, ModuleToolKind::MouseListener },
    { CANVAS_ACT_MENU_TOOL_STACK, LABEL_TOOL_STACK, ModuleToolKind::StackDisplay },
    { CANVAS_ACT_MENU_TOOL_CLONE, LABEL_TOOL_CLONE, ModuleToolKind::Clone },
    { CANVAS_ACT_MENU_TOOL_RECT, LABEL_TOOL_RECT, ModuleToolKind::RectRgba },
    { CANVAS_ACT_MENU_TOOL_NUMBER, LABEL_TOOL_NUMBER, ModuleToolKind::TableNumber },
};

struct ToolMenuLayout {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    int header_h = 0;
    int row_h = 0;
    int item_start_y = 0;
    int stack_start_y = 0;
    int stack_count = 0;
    int table_section_y = 0;
    int table_control_y = 0;
    int table_control_h = 0;
};

struct PluginMenuLayout {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    int header_h = 0;
    int row_h = 0;
    int item_start_y = 0;
    int item_count = 0;
};

struct ModuleMenuLayout {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    int header_h = 0;
    int row_h = 0;
    int item_start_y = 0;
    int item_count = 0;
};

struct RopeMenuLayout {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    int header_h = 0;
    int row_h = 0;
    int item_start_y = 0;
};

struct RopeMenuCounterLayout {
    int bx_minus = 0;
    int bx_num = 0;
    int bx_plus = 0;
    int by = 0;
    int nbw = 0;
    int num_w = 0;
    int h = 0;
    int label_x = 0;
    int label_w = 0;
};

static ToolMenuLayout compute_tool_menu_layout(const GP_CanvasContextImpl* ctx) {
    ToolMenuLayout layout{};
    if (!ctx) return layout;
    const int padding = 8;
    const int margin = 12;
    const int header_h = 18;
    const int row_h = 22;
    const int w = 200;
    int stack_count = 0;
    if (ctx->focused_module >= 0 && ctx->focused_module < static_cast<int>(ctx->module_io_rows.size())) {
        stack_count = static_cast<int>(ctx->module_io_rows[ctx->focused_module].size());
    }
    int h = padding * 2 + header_h + static_cast<int>(std::size(kToolMenuItems)) * row_h;
    if (stack_count > 0) {
        h += padding + header_h + stack_count * row_h;
    }
    h += padding + header_h + row_h;
    int x = std::max(margin, ctx->width - w - margin);
    int y = ctx->rope_bar_h + ctx->control_bar_h + margin;
    layout.x = x;
    layout.y = y;
    layout.w = w;
    layout.h = h;
    layout.header_h = header_h;
    layout.row_h = row_h;
    layout.item_start_y = y + padding + header_h;
    layout.stack_start_y = layout.item_start_y + static_cast<int>(std::size(kToolMenuItems)) * row_h + padding;
    layout.stack_count = stack_count;
    int table_start_y = layout.item_start_y + static_cast<int>(std::size(kToolMenuItems)) * row_h + padding;
    if (stack_count > 0) {
        table_start_y = layout.stack_start_y + stack_count * row_h + padding;
    }
    layout.table_section_y = table_start_y;
    layout.table_control_y = table_start_y + header_h;
    layout.table_control_h = row_h;
    return layout;
}

static ModuleMenuLayout compute_module_menu_layout(const GP_CanvasContextImpl* ctx, int module_idx, int item_count) {
    ModuleMenuLayout layout{};
    if (!ctx || module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return layout;
    const auto& m = ctx->modules[module_idx];
    ModuleLayout mod_layout = module_layout_for(ctx, module_idx, m);
    const int padding = 8;
    const int margin = 12;
    const int header_h = 18;
    const int row_h = 22;
    const int w = 220;
    const int rows = std::max(1, item_count);
    int h = padding * 2 + header_h + rows * row_h;
    int sx = m.x - ctx->offset_x;
    int sy = m.y - ctx->offset_y;
    int x = sx + m.w - w - margin;
    x = std::clamp(x, margin, std::max(margin, ctx->width - w - margin));
    int y = sy + mod_layout.table_y + margin;
    y = std::clamp(y, margin, std::max(margin, ctx->height - h - margin));
    layout.x = x;
    layout.y = y;
    layout.w = w;
    layout.h = h;
    layout.header_h = header_h;
    layout.row_h = row_h;
    layout.item_start_y = y + padding + header_h;
    layout.item_count = rows;
    return layout;
}

static RopeMenuLayout compute_rope_menu_layout(const GP_CanvasContextImpl* ctx) {
    RopeMenuLayout layout{};
    if (!ctx) return layout;
    const int padding = 8;
    const int margin = 12;
    const int header_h = 18;
    const int row_h = 22;
    const int row_count = 4;
    const int w = 220;
    int h = padding * 2 + header_h + row_h * row_count;
    int bw = std::max(8, ctx->rope_bar_h - 8);
    int spacing = 8;
    int menu_btn_x = 8 + (bw + spacing) * 2;
    int x = std::clamp(menu_btn_x, margin, std::max(margin, ctx->width - w - margin));
    int y = ctx->rope_bar_h + ctx->control_bar_h + margin;
    layout.x = x;
    layout.y = y;
    layout.w = w;
    layout.h = h;
    layout.header_h = header_h;
    layout.row_h = row_h;
    layout.item_start_y = y + padding + header_h;
    return layout;
}

static RopeMenuCounterLayout compute_rope_menu_counter_layout(const RopeMenuLayout& layout, int row_idx) {
    RopeMenuCounterLayout out{};
    const int padding = 8;
    const int gap = 8;
    const int label_w = 64;
    int control_x = layout.x + padding;
    int control_w = layout.w - padding * 2;
    int control_h = std::max(18, layout.row_h - 2);
    int nbw = control_h;
    int num_w = std::max(32, control_w - label_w - nbw * 2 - gap * 2);
    int bx_minus = control_x + label_w;
    out.bx_minus = bx_minus;
    out.bx_num = bx_minus + nbw + gap;
    out.bx_plus = out.bx_num + num_w + gap;
    out.by = layout.item_start_y + row_idx * layout.row_h + 1;
    out.nbw = nbw;
    out.num_w = num_w;
    out.h = control_h;
    out.label_x = control_x;
    out.label_w = label_w;
    return out;
}

static PluginMenuLayout compute_plugin_menu_layout(const GP_CanvasContextImpl* ctx, int module_idx, int item_count) {
    PluginMenuLayout layout{};
    if (!ctx || module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return layout;
    const auto& m = ctx->modules[module_idx];
    ModuleLayout mod_layout = module_layout_for(ctx, module_idx, m);
    const int padding = 8;
    const int margin = 12;
    const int header_h = 18;
    const int row_h = 22;
    const int w = 200;
    const int rows = std::max(1, item_count);
    int h = padding * 2 + header_h + rows * row_h;
    int sx = m.x - ctx->offset_x;
    int sy = m.y - ctx->offset_y;
    int x = sx + m.w - w - margin;
    x = std::clamp(x, margin, std::max(margin, ctx->width - w - margin));
    int y = sy + mod_layout.table_y + margin;
    y = std::clamp(y, margin, std::max(margin, ctx->height - h - margin));
    layout.x = x;
    layout.y = y;
    layout.w = w;
    layout.h = h;
    layout.header_h = header_h;
    layout.row_h = row_h;
    layout.item_start_y = y + padding + header_h;
    layout.item_count = rows;
    return layout;
}

static void canvas_refresh_plugin_tools(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    // Discover compiled tool source stems and on-disk shared objects.
    namespace fs = std::filesystem;
    ctx->plugin_tool_kinds.clear();
    ctx->plugin_tool_ids.clear();
    ctx->plugin_tool_labels.clear();

    fs::path root = gp_module_library_default_root();
    fs::path tool_dir = root / "source" / "tools";
    if (!fs::exists(tool_dir)) return;

    std::vector<std::string> tool_ids;
    for (const auto& entry : fs::directory_iterator(tool_dir, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) continue;
        const fs::path& path = entry.path();
        if (path.extension() != ".cpp") continue;
        tool_ids.push_back(path.stem().string());
    }

    // gather on-disk file names under module library root so we can detect compiled artifacts
    std::vector<std::string> shared_libs;
    if (fs::exists(root)) {
        for (const auto& entry : fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) continue;
            shared_libs.push_back(entry.path().filename().string());
        }
    }

    for (const auto& tool_id : tool_ids) {
        ModuleToolKind kind = ModuleToolKind::None;
        if (parse_tool_kind_from_id(tool_id, kind)) {
            // tool id encodes a builtin kind like tool_1 etc.
            bool compiled = false;
            for (const auto& filename : shared_libs) {
                if (shared_library_matches_tool(filename, tool_id)) { compiled = true; break; }
            }
            if (compiled) {
                ctx->plugin_tool_kinds.push_back(kind);
                ctx->plugin_tool_ids.push_back(tool_id);
                ctx->plugin_tool_labels.push_back(tool_id);
            }
        } else {
            // Non-numeric generated tool names (e.g. tool_module_0_ver_...) -- still surface them
            bool compiled = false;
            for (const auto& filename : shared_libs) {
                if (shared_library_matches_tool(filename, tool_id)) { compiled = true; break; }
            }
            if (compiled) {
                ctx->plugin_tool_kinds.push_back(ModuleToolKind::None);
                ctx->plugin_tool_ids.push_back(tool_id);
                ctx->plugin_tool_labels.push_back(tool_id);
            }
        }
    }

    // deduplicate and keep order
    std::vector<std::string> ids_unique;
    std::vector<std::string> labels_unique;
    std::vector<ModuleToolKind> kinds_unique;
    for (size_t i = 0; i < ctx->plugin_tool_ids.size(); ++i) {
        if (std::find(ids_unique.begin(), ids_unique.end(), ctx->plugin_tool_ids[i]) == ids_unique.end()) {
            ids_unique.push_back(ctx->plugin_tool_ids[i]);
            labels_unique.push_back(ctx->plugin_tool_labels[i]);
            kinds_unique.push_back(ctx->plugin_tool_kinds[i]);
        }
    }
    ctx->plugin_tool_ids.swap(ids_unique);
    ctx->plugin_tool_labels.swap(labels_unique);
    ctx->plugin_tool_kinds.swap(kinds_unique);
}

static bool canvas_apply_module_library_entry(GP_CanvasContextImpl* ctx,
                                              int module_idx,
                                              const std::string &manifest_path,
                                              const std::string &serialized_path) {
    if (!ctx) return false;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return false;
    ensure_module_row_order(ctx, module_idx);

    bool applied = false;
    if (!manifest_path.empty()) {
        GP_ModuleLibrary lib{};
        if (gp_module_library_read_from_file(manifest_path.c_str(), &lib)) {
            const GP_ModuleLibraryModule* match = nullptr;
            for (const auto &mod : lib.modules) {
                if (mod.id == gp_module_library_module_id(module_idx)) {
                    match = &mod;
                    break;
                }
            }
            if (!match && !lib.modules.empty()) match = &lib.modules.front();
            if (match) {
                std::vector<ModuleIORow> rows;
                if (module_idx < static_cast<int>(ctx->module_io_rows.size())) {
                    const auto &current = ctx->module_io_rows[module_idx];
                    for (const auto &row : current) {
                        if (row.kind == ModuleRowKind::Input) {
                            rows.push_back({ModuleRowKind::Input, 0, ModuleToolKind::None, row.attachment_count});
                        }
                    }
                }
                std::vector<GP_ModuleToolInstance> sorted = match->tool_instances;
                std::sort(sorted.begin(), sorted.end(), [](const GP_ModuleToolInstance& a, const GP_ModuleToolInstance& b) {
                    return a.row_idx < b.row_idx;
                });
                for (const auto &instance : sorted) {
                    ModuleToolKind kind = ModuleToolKind::None;
                    if (parse_tool_kind_from_id(instance.tool_id, kind)) {
                        rows.push_back({ModuleRowKind::Tool, -1, kind, instance.attachment_count, ModuleToolOrigin::Builtin, std::string()});
                    } else if (!instance.tool_id.empty()) {
                        rows.push_back({ModuleRowKind::Tool, -1, ModuleToolKind::None, instance.attachment_count, ModuleToolOrigin::Plugin, instance.tool_id});
                    }
                }
                if (module_idx < static_cast<int>(ctx->module_io_rows.size())) {
                    const auto &current = ctx->module_io_rows[module_idx];
                    for (const auto &row : current) {
                        if (row.kind == ModuleRowKind::Output) {
                            rows.push_back({ModuleRowKind::Output, 0, ModuleToolKind::None, row.attachment_count});
                        }
                    }
                }
                canvas_apply_io_rows(ctx, module_idx, rows);
                sync_module_table_io_layout(ctx, module_idx);
                if (!match->label.empty()) {
                    auto &m = ctx->modules[module_idx];
                    std::snprintf(m.label, sizeof(m.label), "%s", match->label.c_str());
                    m.label[sizeof(m.label) - 1] = '\0';
                }
                applied = true;
            }
        }
    }

    if (!serialized_path.empty()) {
        try {
            std::filesystem::path full = serialized_path;
            if (std::filesystem::exists(full) && std::filesystem::file_size(full) > 0) {
                if (module_idx >= static_cast<int>(ctx->module_tables.size()) || !ctx->module_tables[module_idx]) {
                    gp_canvas_create_table(reinterpret_cast<GP_CanvasContext*>(ctx), module_idx);
                }
                GP_TableContext* t = (module_idx < static_cast<int>(ctx->module_tables.size()))
                    ? ctx->module_tables[module_idx]
                    : nullptr;
                if (t) {
                    std::ifstream ifs(full, std::ios::binary);
                    if (ifs.good()) {
                        std::vector<char> buf((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                        ifs.close();
                        if (!buf.empty()) {
                            gp_table_deserialize(t, buf.data(), static_cast<int32_t>(buf.size()));
                            applied = true;
                        }
                    }
                }
            }
        } catch (...) {
        }
    }

    if (applied) {
        update_canvas_scroll_state(ctx, /*pull_from_container=*/false);
    }
    return applied;
}

struct ToolMenuCounterLayout {
    int bx_minus = 0;
    int bx_num = 0;
    int bx_plus = 0;
    int by = 0;
    int nbw = 0;
    int num_w = 0;
    int h = 0;
};

static ToolMenuCounterLayout compute_tool_menu_counter_layout(const ToolMenuLayout& layout) {
    ToolMenuCounterLayout out{};
    int control_x = layout.x + 8;
    int control_w = layout.w - 16;
    int control_h = std::max(18, layout.table_control_h - 2);
    int gap = 8;
    int nbw = control_h;
    int num_w = std::max(32, control_w - nbw * 2 - gap * 2);
    int total_w = nbw + gap + num_w + gap + nbw;
    int bx_minus = control_x + (control_w - total_w) / 2;
    out.bx_minus = bx_minus;
    out.bx_num = bx_minus + nbw + gap;
    out.bx_plus = out.bx_num + num_w + gap;
    out.by = layout.table_control_y + 1;
    out.nbw = nbw;
    out.num_w = num_w;
    out.h = control_h;
    return out;
}

struct CanvasTableSnapshot {
    GP_TableContext* table = nullptr;
    std::vector<char> data;
};

static bool canvas_strip_table_rope_blob(const std::vector<char>& in_buf, std::vector<char>& out_buf) {
    if (in_buf.size() < 8 + 4 + 4 + sizeof(GP_TableStyle)) return false;
    const char* base = in_buf.data();
    const char* p = base;
    const char* end = base + in_buf.size();
    p += 8; // magic
    if (p + 4 > end) return false; // version
    p += 4;
    if (p + 4 > end) return false;
    int32_t atlas_count = 0;
    std::memcpy(&atlas_count, p, 4); p += 4;
    if (atlas_count < 0) return false;
    if (p + static_cast<size_t>(atlas_count) * 8 > end) return false;
    p += static_cast<size_t>(atlas_count) * 8;
    if (p + sizeof(GP_TableStyle) > end) return false;
    p += sizeof(GP_TableStyle);
    if (p + 4 > end) return false;
    int32_t col_count = 0;
    std::memcpy(&col_count, p, 4); p += 4;
    if (col_count < 0 || col_count > 8) return false;
    if (p + static_cast<size_t>(col_count) * sizeof(GP_TableColumn) > end) return false;
    p += static_cast<size_t>(col_count) * sizeof(GP_TableColumn);
    if (p + 4 > end) return false;
    int32_t row_count = 0;
    std::memcpy(&row_count, p, 4); p += 4;
    if (row_count < 0) return false;
    if (p + static_cast<size_t>(row_count) * sizeof(GP_TableRow) > end) return false;
    p += static_cast<size_t>(row_count) * sizeof(GP_TableRow);
    if (p + 4 > end) return false;
    int32_t edge_count = 0;
    std::memcpy(&edge_count, p, 4); p += 4;
    if (edge_count < 0) return false;
    if (p + static_cast<size_t>(edge_count) * sizeof(uint64_t) * 2 > end) return false;
    p += static_cast<size_t>(edge_count) * sizeof(uint64_t) * 2;
    if (p + 4 > end) return false;
    int32_t rope_id_count = 0;
    std::memcpy(&rope_id_count, p, 4); p += 4;
    if (rope_id_count < 0) return false;
    if (p + static_cast<size_t>(rope_id_count) * sizeof(uint64_t) > end) return false;
    p += static_cast<size_t>(rope_id_count) * sizeof(uint64_t);
    if (p + 4 > end) return false;
    int32_t sel_count = 0;
    std::memcpy(&sel_count, p, 4); p += 4;
    if (sel_count < 0) return false;
    if (p + static_cast<size_t>(sel_count) * sizeof(uint64_t) > end) return false;
    p += static_cast<size_t>(sel_count) * sizeof(uint64_t);
    if (p + 8 > end) return false;
    p += 8; // module uuid
    if (p + 4 > end) return false;
    int32_t fpcount = 0;
    std::memcpy(&fpcount, p, 4); p += 4;
    if (fpcount < 0) return false;
    if (p + static_cast<size_t>(fpcount) * (4 + 4 + 8) > end) return false;
    p += static_cast<size_t>(fpcount) * (4 + 4 + 8);

    const char* len_ptr = p;
    if (p + 4 > end) return false;
    int32_t rope_blob_len = 0;
    std::memcpy(&rope_blob_len, p, 4);
    if (rope_blob_len < 0) return false;
    const char* blob_start = p + 4;
    const char* blob_end = blob_start + rope_blob_len;
    if (blob_end > end) return false;
    if (rope_blob_len == 0) {
        out_buf = in_buf;
        return true;
    }
    out_buf.resize(in_buf.size() - static_cast<size_t>(rope_blob_len));
    size_t prefix_len = static_cast<size_t>(len_ptr - base);
    std::memcpy(out_buf.data(), in_buf.data(), prefix_len);
    int32_t zero = 0;
    std::memcpy(out_buf.data() + prefix_len, &zero, 4);
    size_t suffix_len = static_cast<size_t>(end - blob_end);
    std::memcpy(out_buf.data() + prefix_len + 4, blob_end, suffix_len);
    return true;
}

static void canvas_reinstantiate_all_ropes(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    std::vector<CanvasTableSnapshot> snapshots;
    std::unordered_set<GP_TableContext*> seen;
    auto capture = [&](GP_TableContext* t) {
        if (!t || seen.find(t) != seen.end()) return;
        int need = gp_table_serialize(t, nullptr, 0);
        if (need <= 0) return;
        std::vector<char> raw(static_cast<size_t>(need));
        int wrote = gp_table_serialize(t, raw.data(), need);
        if (wrote != need) return;
        CanvasTableSnapshot snap{};
        snap.table = t;
        if (!canvas_strip_table_rope_blob(raw, snap.data)) {
            snap.data = std::move(raw);
        }
        snapshots.push_back(std::move(snap));
        seen.insert(t);
    };
    capture(ctx->container_table);
    for (GP_TableContext* t : ctx->module_tables) capture(t);
    if (snapshots.empty()) return;

    for (GP_TableContext* t : seen) {
        gp_table_attach_rope_sim(t, nullptr, 0);
    }
    ctx->rope_id_map.clear();
    ctx->prospective_rope_idx = -1;
    ctx->lasso_rope_idx = -1;

    int segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : std::max(2, ctx->sim_segs);
    for (auto &snap : snapshots) {
        if (!snap.table || snap.data.empty()) continue;
        gp_table_deserialize(snap.table, snap.data.data(), static_cast<int32_t>(snap.data.size()));
        gp_table_set_debug_flags(snap.table, ctx->debug_flags);
        gp_table_set_cable_segments(snap.table, segs);
    }

    RopeSim* sim = rope_sim_create(1024, segs);
    GP_TableContext* root = ctx->container_table;
    if (root) gp_table_attach_rope_sim(root, sim, 1);
    // diagnostic: report sim creation and attachment
    printf("canvas_reinstantiate_all_ropes: created sim=%p root=%p segs=%d\n", (void*)sim, (void*)root, segs);
    for (GP_TableContext* t : ctx->module_tables) {
        if (t && t != root) gp_table_attach_rope_sim(t, sim, 0);
    }
    // Clear ephemeral overlay-created rope indices (no persistent rope_uid)
    // so the overlay raster will recreate ropes in the freshly-created sim.
    for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
        if (ctx->edges[ei].rope_uid == 0ull) ctx->edges[ei].rope_idx = -1;
    }
    if (GP_CanvasContext* cvs = gp_canvas_get_singleton()) {
        gp_canvas_mark_rope_map_dirty(cvs);
        // Immediately refresh the canvas rope->sim mapping so canvas edge
        // entries get rebound to the newly-created RopeSim indices.
        // Without this, overlay rendering may hold stale/empty indices that
        // point to no vertices (seen as verts==0 in logs).
        canvas_refresh_rope_map(ctx);
        // Deterministic rebind: for any canvas edge that has a persistent
        // `rope_uid`, consult the canvas `rope_id_map` to find the table
        // that owns that id and ask the table to resolve the id to the
        // current sim index. This avoids leaving stale numeric indices
        // (from the previous RopeSim instance) on canvas edges.
        for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
            uint64_t uid = ctx->edges[ei].rope_uid;
            if (uid == 0ull) continue;
            auto it = ctx->rope_id_map.find(uid);
            if (it == ctx->rope_id_map.end()) continue;
            GP_TableContext* t = it->second.table;
            if (!t) continue;
            int resolved = gp_table_resolve_rope_id_to_sim_index(t, uid);
            if (resolved >= 0) ctx->edges[ei].rope_idx = resolved;
            else ctx->edges[ei].rope_idx = -1;
        }
    }
}

static const GP_TableAction kCanvasRootActions[] = {
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SIM_SEGS_DEC, CANVAS_ACT_SIM_SEGS_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SIM_SEGS_INC, CANVAS_ACT_SIM_SEGS_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SIM_SLACK_DEC, CANVAS_ACT_SIM_SLACK_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SIM_SLACK_INC, CANVAS_ACT_SIM_SLACK_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_SAVE, CANVAS_ACT_SAVE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_CLEAR, CANVAS_ACT_CLEAR },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_CANVAS_0, CANVAS_ACT_TOOL_CANVAS_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_CANVAS_1, CANVAS_ACT_TOOL_CANVAS_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_CANVAS_2, CANVAS_ACT_TOOL_CANVAS_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_CANVAS_3, CANVAS_ACT_TOOL_CANVAS_3 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_0, CANVAS_ACT_TOOL_EDGE_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_1, CANVAS_ACT_TOOL_EDGE_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_2, CANVAS_ACT_TOOL_EDGE_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_3, CANVAS_ACT_TOOL_EDGE_3 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_EDGE_4, CANVAS_ACT_TOOL_EDGE_4 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EDGE_ORDER_DEC, CANVAS_ACT_EDGE_ORDER_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EDGE_ORDER_INC, CANVAS_ACT_EDGE_ORDER_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EDGE_ORDER_TOOL, CANVAS_ACT_EDGE_ORDER_TOOL },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_TABLE_0, CANVAS_ACT_TOOL_TABLE_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_TABLE_1, CANVAS_ACT_TOOL_TABLE_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_TABLE_2, CANVAS_ACT_TOOL_TABLE_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_FRAME_PAIRS_DEC, CANVAS_ACT_FRAME_PAIRS_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_FRAME_PAIRS_INC, CANVAS_ACT_FRAME_PAIRS_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_IO_COUNT_DEC, CANVAS_ACT_IO_COUNT_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_IO_COUNT_INC, CANVAS_ACT_IO_COUNT_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_IO_CONSUMER_ADD, CANVAS_ACT_IO_CONSUMER_ADD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_IO_PRODUCER_ADD, CANVAS_ACT_IO_PRODUCER_ADD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TABLE_TOOL_NUM_DEC, CANVAS_ACT_TABLE_TOOL_NUM_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TABLE_TOOL_NUM_INC, CANVAS_ACT_TABLE_TOOL_NUM_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_META_CHAN_DEC, CANVAS_ACT_META_CHAN_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_META_CHAN_INC, CANVAS_ACT_META_CHAN_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_KPN_0, CANVAS_ACT_TOOL_KPN_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_KPN_1, CANVAS_ACT_TOOL_KPN_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_KPN_2, CANVAS_ACT_TOOL_KPN_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_THREAD_TOGGLE, CANVAS_ACT_THREAD_TOGGLE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TIMING_TOGGLE, CANVAS_ACT_TIMING_TOGGLE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_THREAD_SIM_TOGGLE, CANVAS_ACT_THREAD_SIM_TOGGLE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_CANVAS_ROOT_SIM_TOGGLE, CANVAS_ACT_CANVAS_ROOT_SIM_TOGGLE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_KPN_GLOBAL_TOGGLE, CANVAS_ACT_KPN_GLOBAL_TOGGLE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_THREAD_DELAY_DEC, CANVAS_ACT_THREAD_DELAY_DEC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_THREAD_DELAY_INC, CANVAS_ACT_THREAD_DELAY_INC },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MODULE_CLONE, CANVAS_ACT_MODULE_CLONE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MODULE_CLEAR, CANVAS_ACT_MODULE_CLEAR },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MODULE_DESTROY, CANVAS_ACT_MODULE_DESTROY },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MODULE_COMMIT, CANVAS_ACT_MODULE_COMMIT },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MODULE_EXPORT, CANVAS_ACT_MODULE_EXPORT },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_0, CANVAS_ACT_TOOL_SUBGROUP_0 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_1, CANVAS_ACT_TOOL_SUBGROUP_1 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_2, CANVAS_ACT_TOOL_SUBGROUP_2 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_3, CANVAS_ACT_TOOL_SUBGROUP_3 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_4, CANVAS_ACT_TOOL_SUBGROUP_4 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_5, CANVAS_ACT_TOOL_SUBGROUP_5 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_6, CANVAS_ACT_TOOL_SUBGROUP_6 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_TOOL_SUBGROUP_7, CANVAS_ACT_TOOL_SUBGROUP_7 },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EXEC_MODE_MODULE_SEQ, CANVAS_ACT_EXEC_MODE_MODULE_SEQ },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EXEC_MODE_MODULE_POOLED, CANVAS_ACT_EXEC_MODE_MODULE_POOLED },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EXEC_MODE_MODULE_SLIP, CANVAS_ACT_EXEC_MODE_MODULE_SLIP },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_EXEC_MODE_MODULE_FREE, CANVAS_ACT_EXEC_MODE_MODULE_FREE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_ADD, CANVAS_ACT_MENU_TOOL_ADD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_SUB, CANVAS_ACT_MENU_TOOL_SUB },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_MUL, CANVAS_ACT_MENU_TOOL_MUL },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_DIV, CANVAS_ACT_MENU_TOOL_DIV },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_MOD, CANVAS_ACT_MENU_TOOL_MOD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_KEYBOARD, CANVAS_ACT_MENU_TOOL_KEYBOARD },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_MOUSE, CANVAS_ACT_MENU_TOOL_MOUSE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_STACK, CANVAS_ACT_MENU_TOOL_STACK },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_CLONE, CANVAS_ACT_MENU_TOOL_CLONE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_RECT, CANVAS_ACT_MENU_TOOL_RECT },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_MENU_TOOL_NUMBER, CANVAS_ACT_MENU_TOOL_NUMBER },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_ROPE_MENU_TOGGLE, CANVAS_ACT_ROPE_MENU_TOGGLE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_ROPE_MODE_SIMPLE, CANVAS_ACT_ROPE_MODE_SIMPLE },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, CANVAS_ACT_ROPE_MODE_FULL, CANVAS_ACT_ROPE_MODE_FULL },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_LED, GP_TABLE_ACTION_ANY, CANVAS_ACT_MODULE_LED },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_LED_ARG, GP_TABLE_ACTION_ANY, CANVAS_ACT_MODULE_LED },
    { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_LED_TABLE, GP_TABLE_ACTION_ANY, CANVAS_ACT_MODULE_LED },
};

static int canvas_handle_module_led_hit(GP_CanvasContextImpl* ctx, int module_idx, const GP_TableHitBox& hit);
static void canvas_setup_stage_table(GP_TableContext* t, int w_px);
static void canvas_setup_stage_defaults(GP_StageContext* st, int w_px, int h_px);
static void stage_bg_callback(void* user, int module_idx, int width, int height, uint8_t* out_rgba, int32_t out_pitch);
static GP_TableContext* canvas_ensure_root_table(GP_CanvasContextImpl* ctx);
static void canvas_append_io_row(GP_CanvasContextImpl* ctx, int module_idx, bool is_input, int attachment_count);
static void sync_module_table_io_layout(GP_CanvasContextImpl* ctx, int module_idx);
static void canvas_push_tool_to_focused(GP_CanvasContextImpl* ctx, ModuleToolKind tool, ModuleToolOrigin origin);

// Per-module recorder state stored by the canvas so it can be freed on table destroy.
struct KeyRecorderState {
    GP_CanvasContextImpl* canvas;
    int module_idx;
    uint64_t writer_key;
    int root_edge_idx;
};

// Create a simple table that records key presses. Attaches a key callback
// that appends a text row for each key press.
static void canvas_install_key_recorder_table(GP_CanvasContextImpl* ctx, int module_idx) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    // create table
    GP_TableContext* t = gp_table_create(nullptr);
    if (!t) return;
    // style
    GP_TableStyle st{};
    st.width_px = std::max(1, ctx->modules[module_idx].w);
    st.row_h_px = 20;
    st.name_w_px = 140;
    gp_table_set_style(t, &st);
    // one text column
    GP_TableColumn col{};
    col.kind = GP_TABLE_CELL_TEXT;
    col.width_px = std::max(64, st.width_px - st.name_w_px);
    col.align = 0;
    gp_table_set_columns(t, &col, 1);
    // initial header row
    GP_TableRow r{};
    memset(&r, 0, sizeof(r));
    r.kind = GP_TABLE_ROW_HEADER;
    r.depth = 0;
    r.expanded = 1;
    r.selected = 0;
    r.cell_count = 1;
    r.cells[0].kind = GP_TABLE_CELL_TEXT;
    std::snprintf(r.cells[0].text, sizeof(r.cells[0].text), "%s", LABEL_MENU_KEY_RECORDER);
    gp_table_set_rows(t, &r, 1);

    // We'll attach a key callback that both records the key locally and
    // publishes the key value onto a root-table edge (if present).
    // Create a small state object and store it per-module.
    KeyRecorderState* ks = new KeyRecorderState();
    ks->canvas = ctx; ks->module_idx = module_idx; ks->writer_key = 0; ks->root_edge_idx = -1;

    gp_table_set_key_callback(t, +[](void* user, int key, int scancode, int action, int mods) {
        KeyRecorderState* ks = reinterpret_cast<KeyRecorderState*>(user);
        if (!ks) return;
        printf("key-recorder callback: module=%d key=%d sc=%d action=%d mods=%d writer_key=%llu root_edge_idx=%d\n", ks->module_idx, key, scancode, action, mods, (unsigned long long)ks->writer_key, ks->root_edge_idx);
        GP_TableContext* tt = nullptr;
        // find the attached table for this module via canvas
        GP_CanvasContextImpl* c = ks->canvas;
        if (!c) return;
        int mi = ks->module_idx;
        if (mi < 0 || mi >= static_cast<int>(c->module_tables.size())) return;
        tt = c->module_tables[mi];
        if (!tt) return;
        // Only act on non-zero actions
        if (action == 0) return;
        // append local row
        int32_t n = gp_table_get_row_count(tt);
        std::vector<GP_TableRow> rows;
        rows.resize(static_cast<size_t>(n + 1));
        for (int32_t i = 0; i < n; ++i) {
            GP_TableRow tmp; memset(&tmp, 0, sizeof(tmp));
            if (gp_table_get_row(tt, i, &tmp)) rows[static_cast<size_t>(i)] = tmp;
        }
        GP_TableRow nr; memset(&nr, 0, sizeof(nr));
        nr.kind = GP_TABLE_ROW_DEVICE; nr.depth = 0; nr.expanded = 1; nr.selected = 0; nr.cell_count = 1;
        nr.cells[0].kind = GP_TABLE_CELL_TEXT;
        std::snprintf(nr.cells[0].text, sizeof(nr.cells[0].text), "K=%d S=%d A=%d M=%d", key, scancode, action, mods);
        rows[static_cast<size_t>(n)] = nr;
        gp_table_set_rows(tt, rows.data(), static_cast<int32_t>(rows.size()));

        // publish to root edge(s) if available. Use module's output count
        // to determine how many contact channels to publish to. This lets a
        // single recorder emit to multiple writer keys (contact_idx 0..N-1).
        GP_TableContext* root = canvas_ensure_root_table(c);
        if (!root) return;
        int out_count = 1;
        if (mi >= 0 && mi < static_cast<int>(c->module_io_out_count.size())) out_count = std::max(1, c->module_io_out_count[mi]);
        for (int ci = 0; ci < out_count; ++ci) {
            uint64_t writer_key = (static_cast<uint64_t>(static_cast<uint32_t>(mi)) << 32) |
                                  (static_cast<uint64_t>(static_cast<uint32_t>(ci)) << 16) |
                                  static_cast<uint64_t>(0);
            int edge_idx = -1;
            if (!gp_table_edge_index_for_key(root, writer_key, &edge_idx) || edge_idx < 0) {
                // no edge for this contact, skip
                //printf("key-recorder: no root edge for module=%d contact=%d key=%llu\n", ks->module_idx, ci, (unsigned long long)writer_key);
                continue;
            }
            float payload[1]; payload[0] = static_cast<float>(key);
            int dropped = 0;
            int ok = gp_table_edge_publish(root, edge_idx, writer_key, payload, sizeof(payload), &dropped);
            printf("key-recorder publish: module=%d contact=%d edge=%d writer_key=%llu ok=%d dropped=%d payload=%f\n", ks->module_idx, ci, edge_idx, (unsigned long long)writer_key, ok, dropped, payload[0]);
            if (!ok) {
                printf("key-recorder publish FAILED: module=%d contact=%d edge=%d writer_key=%llu\n", ks->module_idx, ci, edge_idx, (unsigned long long)writer_key);
            }
        }
    }, ks);

    // store state so we can free it on table destroy
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_key_recorder_state.size())) ctx->module_key_recorder_state[module_idx] = reinterpret_cast<void*>(ks);
    printf("canvas_install_key_recorder_table: installed recorder for module=%d writer_key=%llu\n", module_idx, (unsigned long long)ks->writer_key);

    // attach to module and let canvas own it
    gp_canvas_attach_table(reinterpret_cast<GP_CanvasContext*>(ctx), module_idx, t, 1);
    ctx->focused_module = module_idx;
    if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) ctx->module_io_rows.resize(module_idx + 1);
    ctx->module_io_rows[module_idx].clear();
    if (module_idx >= static_cast<int>(ctx->module_table_rows.size())) ctx->module_table_rows.resize(module_idx + 1);
    ctx->module_table_rows[module_idx].clear();
    if (module_idx >= static_cast<int>(ctx->module_stack_snapshots.size())) ctx->module_stack_snapshots.resize(module_idx + 1);
    ctx->module_stack_snapshots[module_idx].clear();
    // If there's another module available, create a canvas/root edge from this module to the next module
    if (ctx->modules.size() > 1) {
        int target = (module_idx + 1) % static_cast<int>(ctx->modules.size());
        if (target != module_idx) {
            GP_CanvasEdgeDesc ed{};
            ed.a_module = module_idx; ed.a_contact_idx = 0;
            ed.b_module = target; ed.b_contact_idx = 0;
            // create a canvas-level edge which will ensure root-table FIFOs
            gp_canvas_add_edge_with_type(reinterpret_cast<GP_CanvasContext*>(ctx), &ed, /*type_id=*/0);
            // Try to synchronously create the corresponding root edge for immediate publishes
            GP_TableContext* root = canvas_ensure_root_table(ctx);
                if (root) {
                    uint64_t ka = (static_cast<uint64_t>(static_cast<uint32_t>(ed.a_module)) << 32) |
                                  (static_cast<uint64_t>(static_cast<uint32_t>(ed.a_contact_idx)) << 16) |
                                  static_cast<uint64_t>(0);
                    uint64_t kb = (static_cast<uint64_t>(static_cast<uint32_t>(ed.b_module)) << 32) |
                                  (static_cast<uint64_t>(static_cast<uint32_t>(ed.b_contact_idx)) << 16) |
                                  static_cast<uint64_t>(0);
                    gp_table_add_edge(root, ka, kb);
                    // update cached edge index and writer_key if state exists
                    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_key_recorder_state.size())) {
                        void* s = ctx->module_key_recorder_state[module_idx];
                        if (s) {
                            KeyRecorderState* ks2 = reinterpret_cast<KeyRecorderState*>(s);
                            // if writer_key wasn't set (==0), use the canonical 'ka' we just created
                            if (ks2->writer_key == 0) ks2->writer_key = ka;
                            int idx = -1;
                            if (gp_table_edge_index_for_key(root, ks2->writer_key, &idx) && idx >= 0) ks2->root_edge_idx = idx;
                        }
                    }
                    (void)0;
                }
        }
    }
}

static void canvas_install_root_actions(GP_CanvasContextImpl* ctx, GP_TableContext* table) {
    if (!ctx || !table) return;
    if (ctx->root_actions_installed) return;
    gp_table_set_action_callback(table, [](void* user, int32_t action_id, const GP_TableHitBox* hit) {
        auto* c = reinterpret_cast<GP_CanvasContextImpl*>(user);
        if (!c || !hit) return;
        // Click-listen mode: capture the action intent instead of dispatching.
        if (c->click_listen_mode) {
            if (c->pending_action) {
                // Already have a pending action; overwrite (caller intent keeps most recent)
                delete c->pending_action;
                c->pending_action = nullptr;
            }
            c->pending_action = new GP_CanvasContextImpl::PendingAction();
            c->pending_action->action_id = action_id;
            c->pending_action->hit = *hit;
            printf("gp_canvas_on_click: captured action_id=%d in click-listen mode\n", action_id);
            return;
        }
        switch (action_id) {
            case CANVAS_ACT_SIM_SEGS_DEC:
                c->sim_segs = std::max(2, c->sim_segs - 1);
                printf("gp_canvas_on_click: sim_segs=%d\n", c->sim_segs);
                canvas_reinstantiate_all_ropes(c);
                break;
            case CANVAS_ACT_SIM_SEGS_INC:
                c->sim_segs = std::min(64, c->sim_segs + 1);
                printf("gp_canvas_on_click: sim_segs=%d\n", c->sim_segs);
                canvas_reinstantiate_all_ropes(c);
                break;
            case CANVAS_ACT_SIM_SLACK_DEC:
                c->sim_slack = std::max(0.0f, c->sim_slack - 0.1f);
                printf("gp_canvas_on_click: sim_slack=%.2f\n", c->sim_slack);
                break;
            case CANVAS_ACT_SIM_SLACK_INC:
                c->sim_slack = std::min(8.0f, c->sim_slack + 0.1f);
                printf("gp_canvas_on_click: sim_slack=%.2f\n", c->sim_slack);
                break;
            case CANVAS_ACT_ROPE_MENU_TOGGLE:
                c->rope_menu_open = !c->rope_menu_open;
                if (c->rope_menu_open) {
                    c->tool_menu_open = false;
                    c->plugin_menu_open = false;
                    c->plugin_menu_module_idx = -1;
                    c->module_menu_open = false;
                    c->module_menu_module_idx = -1;
                }
                printf("gp_canvas_on_click: rope_menu_open -> %d\n", c->rope_menu_open ? 1 : 0);
                break;
            case CANVAS_ACT_ROPE_MODE_SIMPLE: {
                gp_canvas_set_debug_flags(reinterpret_cast<GP_CanvasContext*>(c), GP_CANVAS_DEBUG_NORENDER_MODE);
                canvas_reinstantiate_all_ropes(c);
                printf("gp_canvas_on_click: rope_mode -> simple\n");
                break;
            }
            case CANVAS_ACT_ROPE_MODE_FULL: {
                gp_canvas_set_debug_flags(reinterpret_cast<GP_CanvasContext*>(c), 0u);
                canvas_reinstantiate_all_ropes(c);
                printf("gp_canvas_on_click: rope_mode -> full\n");
                break;
            }
            case CANVAS_ACT_SAVE:
                if (!c->autosave_path.empty()) {
                    gp_canvas_save_to_file(reinterpret_cast<GP_CanvasContext*>(c), c->autosave_path.c_str());
                } else {
                    gp_canvas_save_to_file(reinterpret_cast<GP_CanvasContext*>(c), kCanvasDefaultWorkspacePath);
                }
                break;
            case CANVAS_ACT_CLEAR:
                // Clear overlays, meta-groups, and modules via unified API
                gp_canvas_clear_meta_and_overlays(reinterpret_cast<GP_CanvasContext*>(c));
                update_canvas_scroll_state(c, /*pull_from_container=*/false);
                break;
            case CANVAS_ACT_TOOL_CANVAS_0:
            case CANVAS_ACT_TOOL_CANVAS_1:
            case CANVAS_ACT_TOOL_CANVAS_2:
            case CANVAS_ACT_TOOL_CANVAS_3: {
                int tool = static_cast<int>(action_id - CANVAS_ACT_TOOL_CANVAS_0);
                if (c->selected_tool_canvas == tool) c->selected_tool_canvas = 0; else c->selected_tool_canvas = tool;
                printf("gp_canvas_on_click: canvas tool %d toggled -> selected_tool_canvas=%d\n", tool, c->selected_tool_canvas);
                break;
            }
            case CANVAS_ACT_TOOL_EDGE_0:
            case CANVAS_ACT_TOOL_EDGE_1:
            case CANVAS_ACT_TOOL_EDGE_2:
            case CANVAS_ACT_TOOL_EDGE_3: {
                // Selecting a regular edge tool clears lasso mode
                c->lasso_mode = false;
                int tool = static_cast<int>(action_id - CANVAS_ACT_TOOL_EDGE_0);
                if (c->selected_tool_edge == tool) c->selected_tool_edge = -1; else c->selected_tool_edge = tool;
                printf("gp_canvas_on_click: edge tool %d toggled -> selected_tool_edge=%d\n", tool, c->selected_tool_edge);
                break;
            }
            case CANVAS_ACT_TOOL_EDGE_4: {
                // Lasso button: toggle lasso_mode and ensure other edge tools are deselected
                c->selected_tool_edge = -1;
                c->lasso_mode = !c->lasso_mode;
                if (c->lasso_mode) {
                    c->lasso_module_idx = -1;
                    c->lasso_rope_idx = -1;
                }
                printf("gp_canvas_on_click: lasso_mode toggled -> %d\n", c->lasso_mode);
                break;
            }
            case CANVAS_ACT_EDGE_ORDER_DEC:
            case CANVAS_ACT_EDGE_ORDER_INC: {
                int delta = (action_id == CANVAS_ACT_EDGE_ORDER_INC) ? 1 : -1;
                c->edge_order_value = std::clamp(c->edge_order_value + delta, -4, 4);
                printf("gp_canvas_on_click: edge_order_value -> %d\n", c->edge_order_value);
                break;
            }
            case CANVAS_ACT_EDGE_ORDER_TOOL: {
                c->edge_order_tool_active = c->edge_order_tool_active ? 0 : 1;
                printf("gp_canvas_on_click: edge_order_tool_active -> %d\n", c->edge_order_tool_active);
                break;
            }
            
            case CANVAS_ACT_TOOL_TABLE_0:
            case CANVAS_ACT_TOOL_TABLE_1:
            case CANVAS_ACT_TOOL_TABLE_2: {
                // debug: table-tool button clicked (before state change)
                printf("gp_canvas_on_click: table-tool button clicked action_id=%d selected_tool_table(before)=%d\n", action_id, c->selected_tool_table);
                int tool = static_cast<int>(action_id - CANVAS_ACT_TOOL_TABLE_0);
                if (tool == 2) {
                    c->tool_menu_open = !c->tool_menu_open;
                    c->selected_tool_table = c->tool_menu_open ? tool : 0;
                } else {
                    if (c->selected_tool_table == tool) c->selected_tool_table = 0; else c->selected_tool_table = tool;
                    c->tool_menu_open = false;
                }
                c->plugin_menu_open = false;
                c->plugin_menu_module_idx = -1;
                c->module_menu_open = false;
                c->module_menu_module_idx = -1;
                c->rope_menu_open = false;
                printf("gp_canvas_on_click: table tool %d toggled -> selected_tool_table=%d\n", tool, c->selected_tool_table);
                break;
            }
            case CANVAS_ACT_IO_COUNT_DEC:
            case CANVAS_ACT_IO_COUNT_INC: {
                int delta = (action_id == CANVAS_ACT_IO_COUNT_INC) ? 1 : -1;
                c->io_attachment_count = std::clamp(c->io_attachment_count + delta, 1, 32);
                printf("gp_canvas_on_click: io_attachment_count -> %d\n", c->io_attachment_count);
                break;
            }
            case CANVAS_ACT_FRAME_PAIRS_DEC:
            case CANVAS_ACT_FRAME_PAIRS_INC: {
                int delta = (action_id == CANVAS_ACT_FRAME_PAIRS_INC) ? 1 : -1;
                const int max_pairs = kModuleExtraLedCount * 2; // allow across both left+right
                c->module_frame_pair_count = std::clamp(c->module_frame_pair_count + delta, 1, max_pairs);
                printf("gp_canvas_on_click: module_frame_pair_count -> %d\n", c->module_frame_pair_count);
                break;
            }
            case CANVAS_ACT_IO_CONSUMER_ADD:
            case CANVAS_ACT_IO_PRODUCER_ADD: {
                int focused = c->focused_module;
                if (focused >= 0 && focused < static_cast<int>(c->modules.size())) {
                    bool is_input = (action_id == CANVAS_ACT_IO_CONSUMER_ADD);
                    canvas_append_io_row(c, focused, is_input, c->io_attachment_count);
                    printf("gp_canvas_on_click: module=%d add %s row (attachments=%d)\n",
                        focused,
                        is_input ? "consumer" : "producer",
                        c->io_attachment_count);
                }
                break;
            }
            case CANVAS_ACT_TABLE_TOOL_NUM_DEC:
            case CANVAS_ACT_TABLE_TOOL_NUM_INC: {
                int delta = (action_id == CANVAS_ACT_TABLE_TOOL_NUM_INC) ? 1 : -1;
                c->table_tool_number = std::clamp(c->table_tool_number + delta, 0, 99);
                printf("gp_canvas_on_click: table_tool_number -> %d\n", c->table_tool_number);
                break;
            }
            case CANVAS_ACT_META_CHAN_DEC:
            case CANVAS_ACT_META_CHAN_INC: {
                int delta = (action_id == CANVAS_ACT_META_CHAN_INC) ? 1 : -1;
                // If aux0 encodes an overlay id (ov.id), use the overlay's
                // authoritative meta binding directly to mutate the meta-group.
                int overlay_id = hit->aux0;
                if (overlay_id > 0) {
                    auto oit = c->overlays.find(overlay_id);
                    if (oit != c->overlays.end()) {
                        auto &ov = oit->second;
                        if (ov.meta_table && ov.meta_mg) {
                            int cur = 0; gp_table_meta_get_channel_group(ov.meta_table, ov.meta_mg, &cur);
                            cur = std::clamp(cur + delta, -32768, 32767);
                            gp_table_meta_set_channel_group(ov.meta_table, ov.meta_mg, cur);
                            update_canvas_scroll_state(c, /*pull_from_container=*/false);
                            printf("gp_canvas_on_click: meta mg=%p ov=%d new_group=%d\n", (void*)ov.meta_mg, overlay_id, cur);
                            break;
                        }
                    }
                }
                // Fallback: hit->aux0 carried a rope_idx; search tables for matching dangling rope
                int rope_idx = hit->aux0;
                auto try_handle = [&](GP_TableContext* t)->bool{
                    if (!t) return false;
                    int mgcount = gp_table_get_meta_group_count(t);
                    for (int mgi = 0; mgi < mgcount; ++mgi) {
                        GP_MetaGroup* mg = gp_table_get_meta_group(t, mgi);
                        if (!mg) continue;
                        int _dr=-1,_dv=-1; gp_table_meta_get_dangling_rope_info(t, mg, &_dr, &_dv);
                        if (_dr != rope_idx) continue;
                        int cur = 0; gp_table_meta_get_channel_group(t, mg, &cur);
                        cur = std::clamp(cur + delta, -32768, 32767);
                        gp_table_meta_set_channel_group(t, mg, cur);
                        update_canvas_scroll_state(c, /*pull_from_container=*/false);
                        printf("gp_canvas_on_click: meta mg=%p rope=%d new_group=%d\n", (void*)mg, rope_idx, cur);
                        return true;
                    }
                    return false;
                };
                if (try_handle(c->container_table)) break;
                for (size_t mi = 0; mi < c->module_tables.size(); ++mi) if (try_handle(c->module_tables[mi])) break;
                break;
            }
            case CANVAS_ACT_TOOL_KPN_0:
            case CANVAS_ACT_TOOL_KPN_1:
            case CANVAS_ACT_TOOL_KPN_2: {
                int tool = static_cast<int>(action_id - CANVAS_ACT_TOOL_KPN_0);
                if (c->selected_tool_kpn == tool) c->selected_tool_kpn = -1; else c->selected_tool_kpn = tool;
                printf("gp_canvas_on_click: kpn tool %d toggled -> selected_tool_kpn=%d\n", tool, c->selected_tool_kpn);
                break;
            }
            case CANVAS_ACT_TOOL_SUBGROUP_0:
            case CANVAS_ACT_TOOL_SUBGROUP_1:
            case CANVAS_ACT_TOOL_SUBGROUP_2:
            case CANVAS_ACT_TOOL_SUBGROUP_3:
            case CANVAS_ACT_TOOL_SUBGROUP_4:
            case CANVAS_ACT_TOOL_SUBGROUP_5:
            case CANVAS_ACT_TOOL_SUBGROUP_6:
            case CANVAS_ACT_TOOL_SUBGROUP_7: {
                int tool = static_cast<int>(action_id - CANVAS_ACT_TOOL_SUBGROUP_0);
                if (tool >= kSubgroupBinCount) break;
                uint32_t bit = subgroup_mask_for_index(tool);
                if (c->selected_tool_subgroup_flags & bit) c->selected_tool_subgroup_flags &= ~bit;
                else c->selected_tool_subgroup_flags |= bit;
                printf("gp_canvas_on_click: subgroup tool %d toggled -> selected_tool_subgroup_flags=0x%08x\n", tool, c->selected_tool_subgroup_flags);
                break;
            }
            case CANVAS_ACT_THREAD_TOGGLE: {
                // If a module is focused, toggle that module's sim flag (per-module play/pause).
                // If a module-specific action is in-flight (dispatch_module_idx), prefer it.
                int target_module = -1;
                if (c->dispatch_module_idx >= 0) target_module = c->dispatch_module_idx;
                else if (c->focused_module >= 0) target_module = c->focused_module;
                if (target_module >= 0 && target_module < static_cast<int>(c->modules.size())) {
                    int mi = target_module;
                    if (mi >= static_cast<int>(c->module_skip.size())) c->module_skip.resize(mi + 1, 0);
                    c->module_skip[mi] = c->module_skip[mi] ? 0 : 1;
                    printf("gp_canvas_on_click: module %d module_skip -> %d\n", mi, c->module_skip[mi]);
                } else {
                    // No module target: fall back to toggling global thread manager pause
                    // keep CANVAS_ACT_THREAD_TOGGLE module-scoped; use GLOBAL_TOGGLE for global toolbar
                    printf("gp_canvas_on_click: THREAD_TOGGLE invoked with no module target - ignoring global toggle\n");
                }
                break;
            }
            case CANVAS_ACT_TIMING_TOGGLE: {
                // Toggle global timing collection on ThreadManager
                if (ThreadManager::global()) {
                    bool nv = !ThreadManager::global()->timing_enabled();
                    ThreadManager::global()->set_timing_enabled(nv);
                    printf("gp_canvas_on_click: TIMING_TOGGLE -> %d\n", nv ? 1 : 0);
                }
                break;
            }
            case CANVAS_ACT_KPN_GLOBAL_TOGGLE: {
                c->thread_mgr_paused = !c->thread_mgr_paused;
                if (c->thread_mgr_paused) {
                    c->thread_mgr_delay_accum_s = 0.0;
                } else {
                    c->thread_mgr_delay_accum_s = static_cast<double>(std::max(0, c->thread_mgr_delay_ms)) / 1000.0;
                }
                printf("gp_canvas_on_click: GLOBAL kpn_paused -> %d\n", c->thread_mgr_paused ? 1 : 0);
                break;
            }
            case CANVAS_ACT_THREAD_SIM_TOGGLE: {
                int target_module = -1;
                if (c->dispatch_module_idx >= 0) target_module = c->dispatch_module_idx;
                else if (c->focused_module >= 0) target_module = c->focused_module;
                if (target_module >= 0 && target_module < static_cast<int>(c->modules.size())) {
                    int mi = target_module;
                    if (mi >= static_cast<int>(c->module_sim_enabled.size())) c->module_sim_enabled.resize(mi + 1, 1);
                    c->module_sim_enabled[mi] = c->module_sim_enabled[mi] ? 0 : 1;
                    // apply to attached table if present
                    if (mi >= 0 && mi < static_cast<int>(c->module_tables.size()) && c->module_tables[mi]) {
                        gp_table_set_sim_enabled(c->module_tables[mi], c->module_sim_enabled[mi]);
                    }
                    // ensure canvas re-renders rope visuals after sim toggle
                    gp_canvas_mark_rope_map_dirty(reinterpret_cast<GP_CanvasContext*>(c));
                    printf("gp_canvas_on_click: module %d sim_enabled -> %d\n", mi, c->module_sim_enabled[mi]);
                } else {
                    // No module target — treat as canvas-root simulator toggle (toolbar SIM)
                    // This handler path is only reached for module-targeted SIM toggles.
                    // Toolbar-level SIM now uses CANVAS_ACT_CANVAS_ROOT_SIM_TOGGLE and
                    // is handled separately.
                }
                break;
            }
            // Per-module exec-mode actions (module-local buttons)
            case CANVAS_ACT_EXEC_MODE_MODULE_SEQ:
            case CANVAS_ACT_EXEC_MODE_MODULE_POOLED:
            case CANVAS_ACT_EXEC_MODE_MODULE_SLIP:
            case CANVAS_ACT_EXEC_MODE_MODULE_FREE: {
                int mode = -1;
                if (action_id == CANVAS_ACT_EXEC_MODE_MODULE_SEQ) mode = 0;
                else if (action_id == CANVAS_ACT_EXEC_MODE_MODULE_POOLED) mode = 1;
                else if (action_id == CANVAS_ACT_EXEC_MODE_MODULE_SLIP) mode = 2;
                else if (action_id == CANVAS_ACT_EXEC_MODE_MODULE_FREE) mode = 3;
                int target_module = -1;
                if (c->dispatch_module_idx >= 0) target_module = c->dispatch_module_idx;
                else if (c->focused_module >= 0) target_module = c->focused_module;
                if (target_module >= 0 && target_module < static_cast<int>(c->modules.size())) {
                    int mi = target_module;
                    if (mi >= static_cast<int>(c->module_exec_mode.size())) c->module_exec_mode.resize(mi + 1, -1);
                    // Toggle: clicking the already-selected mode will deselect (set -1)
                    if (c->module_exec_mode[mi] == mode) c->module_exec_mode[mi] = -1;
                    else c->module_exec_mode[mi] = mode;
                    printf("gp_canvas_on_click: module %d exec_mode -> %d\n", mi, c->module_exec_mode[mi]);
                }
                break;
            }
            // Global exec-mode toolbar actions (override)
            case CANVAS_ACT_EXEC_MODE_SEQ:
            case CANVAS_ACT_EXEC_MODE_POOLED:
            case CANVAS_ACT_EXEC_MODE_SLIP:
            case CANVAS_ACT_EXEC_MODE_FREE: {
                int mode = -1;
                if (action_id == CANVAS_ACT_EXEC_MODE_SEQ) mode = 0;
                else if (action_id == CANVAS_ACT_EXEC_MODE_POOLED) mode = 1;
                else if (action_id == CANVAS_ACT_EXEC_MODE_SLIP) mode = 2;
                else if (action_id == CANVAS_ACT_EXEC_MODE_FREE) mode = 3;
                if (c->thread_mgr_global_exec_mode == mode) c->thread_mgr_global_exec_mode = -1;
                else c->thread_mgr_global_exec_mode = mode;
                printf("gp_canvas_on_click: GLOBAL exec_mode_override -> %d\n", c->thread_mgr_global_exec_mode);
                break;
            }
            case CANVAS_ACT_CANVAS_ROOT_SIM_TOGGLE: {
                GP_TableContext* root = canvas_ensure_root_table(c);
                if (root) {
                    int cur = 0;
                    gp_table_get_sim_enabled(root, &cur);
                    int next = cur ? 0 : 1;
                    // Toggle per-table sim enable as before
                    gp_table_set_sim_enabled(root, next);
                    // ensure canvas rope visuals are refreshed immediately when root sim toggles
                    gp_canvas_mark_rope_map_dirty(reinterpret_cast<GP_CanvasContext*>(c));
                    // Also toggle a toolbar-level global pause so the canvas root sim
                    // (and all attached table sims) defer to the global skip decision.
                    c->sim_root_paused = next ? 0 : 1;
                    if (c->sim_root_paused) {
                        // set an effectively infinite skip so gp_table_should_step_sim
                        // will return false for all tables until unpaused
                        gp_table_set_global_sim_frame_skip_count(INT32_MAX / 4);
                    } else {
                        // restore canvas-requested skip count
                        gp_table_set_global_sim_frame_skip_count(c->sim_frame_skip_count);
                    }
                    printf("gp_canvas_on_click: canvas-root sim_enabled -> %d sim_root_paused=%d\n", next, c->sim_root_paused);
                } else {
                    printf("gp_canvas_on_click: canvas-root table unavailable for SIM toggle\n");
                }
                break;
            }
            
            case CANVAS_ACT_THREAD_DELAY_DEC:
            case CANVAS_ACT_THREAD_DELAY_INC: {
                int delta = (action_id == CANVAS_ACT_THREAD_DELAY_INC) ? 10 : -10;
                c->thread_mgr_delay_ms = std::clamp(c->thread_mgr_delay_ms + delta, 0, 2000);
                printf("gp_canvas_on_click: thread_mgr_delay_ms -> %d\n", c->thread_mgr_delay_ms);
                break;
            }
            case CANVAS_ACT_MODULE_CLONE: {
                int focused = c->focused_module;
                if (focused >= 0 && focused < static_cast<int>(c->modules.size())) {
                    int new_idx = canvas_clone_module(c, focused);
                    if (new_idx >= 0) c->focused_module = new_idx;
                }
                break;
            }
            case CANVAS_ACT_MODULE_CLEAR: {
                int focused = c->focused_module;
                if (focused >= 0 && focused < static_cast<int>(c->modules.size())) {
                    canvas_clear_module_table(c, focused);
                }
                break;
            }
            case CANVAS_ACT_MODULE_DESTROY: {
                int focused = c->focused_module;
                if (focused >= 0 && focused < static_cast<int>(c->modules.size())) {
                    canvas_destroy_module(c, focused);
                }
                break;
            }
            case CANVAS_ACT_MODULE_COMMIT: {
                int focused = c->focused_module;
                if (focused >= 0 && focused < static_cast<int>(c->modules.size())) {
                    // Queue commit/build; processed later in gp_canvas_step to avoid running during input callbacks.
                    canvas_clear_module_bindings(c, focused);
                    c->pending_module_commits.push_back(focused);
                }
                break;
            }

            case CANVAS_ACT_MODULE_EXPORT: {
                int focused = c->focused_module;
                if (focused >= 0 && focused < static_cast<int>(c->modules.size())) {
                    gp_canvas_export_module_to_root(reinterpret_cast<GP_CanvasContext*>(c), focused, nullptr);
                    (void)canvas_write_module_library_manifest(c, focused);
                }
                break;
            }
            case CANVAS_ACT_MODULE_LED:
                if (c->dispatch_module_idx >= 0) {
                    canvas_handle_module_led_hit(c, c->dispatch_module_idx, *hit);
                }
                break;
            case CANVAS_ACT_MENU_TOOL_ADD:
                canvas_push_tool_to_focused(c, ModuleToolKind::Add, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_SUB:
                canvas_push_tool_to_focused(c, ModuleToolKind::Subtract, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_MUL:
                canvas_push_tool_to_focused(c, ModuleToolKind::Multiply, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_DIV:
                canvas_push_tool_to_focused(c, ModuleToolKind::Divide, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_MOD:
                canvas_push_tool_to_focused(c, ModuleToolKind::Modulo, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_KEYBOARD:
                canvas_push_tool_to_focused(c, ModuleToolKind::KeyboardListener, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_MOUSE:
                canvas_push_tool_to_focused(c, ModuleToolKind::MouseListener, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_STACK:
                canvas_push_tool_to_focused(c, ModuleToolKind::StackDisplay, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_CLONE:
                canvas_push_tool_to_focused(c, ModuleToolKind::Clone, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_RECT:
                canvas_push_tool_to_focused(c, ModuleToolKind::RectRgba, ModuleToolOrigin::Builtin);
                break;
            case CANVAS_ACT_MENU_TOOL_NUMBER:
                canvas_push_tool_to_focused(c, ModuleToolKind::TableNumber, ModuleToolOrigin::Builtin);
                break;
            default:
                break;
        }
    }, ctx);
    gp_table_set_actions(table, kCanvasRootActions, static_cast<int>(sizeof(kCanvasRootActions) / sizeof(kCanvasRootActions[0])));
    ctx->root_actions_installed = 1;
}

static GP_TableContext* canvas_ensure_root_table(GP_CanvasContextImpl* ctx) {
    if (!ctx) return nullptr;
    if (!ctx->container_table) {
        ctx->container_table = gp_table_create(nullptr);
        ctx->container_table_owned = 1;
        ctx->root_actions_installed = 0;
        gp_table_set_debug_flags(ctx->container_table, ctx->debug_flags);
        {
            int segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : std::max(2, ctx->sim_segs);
            gp_table_set_cable_segments(ctx->container_table, segs);
        }
    }
    canvas_install_root_actions(ctx, ctx->container_table);
    // Ensure a root RopeSim exists immediately so edges created later
    // on the root/container table get real rope indices and persistent uids.
    if (ctx->container_table && !gp_table_get_rope_sim(ctx->container_table)) {
        int max_segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : std::max(2, ctx->sim_segs);
        RopeSim* sim = rope_sim_create(1024, max_segs);
        gp_table_attach_rope_sim(ctx->container_table, sim, 1);
        // Attach any existing module tables to the new root sim so their
        // edges also get real ropes/uids rather than deferring until later.
        for (size_t i = 0; i < ctx->module_tables.size(); ++i) {
            if (ctx->module_tables[i]) gp_table_attach_rope_sim(ctx->module_tables[i], sim, 0);
        }
        printf("canvas_ensure_root_table: created and attached root RopeSim %p\n", (void*)sim);
    }
    return ctx->container_table;
}

// Bind up to `max_ports` currently-empty receive frame ports on `module_idx` to the
// specified action id. Returns how many were bound.
static RopeSim* canvas_root_sim(GP_CanvasContextImpl* ctx) {
    if (!ctx || !ctx->container_table) return nullptr;
    return gp_table_get_rope_sim(ctx->container_table);
}

// Ensure a synthetic canvas module that mirrors the root/container table exists.
static int canvas_ensure_root_module(GP_CanvasContextImpl* ctx) {
    if (!ctx) return -1;
    if (ctx->root_module_idx >= 0 && ctx->root_module_idx < static_cast<int>(ctx->modules.size())) return ctx->root_module_idx;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return -1;
    GP_CanvasModuleDesc d{};
    d.x = 16; d.y = 64; d.w = std::max(160, ctx->width - 32); d.h = std::max(120, ctx->height / 3);
    std::strncpy(d.label, "ROOT", sizeof(d.label) - 1);
    d.label[sizeof(d.label) - 1] = '\0';
    int new_idx = gp_canvas_add_module(reinterpret_cast<GP_CanvasContext*>(ctx), &d);
    if (new_idx < 0) return -1;
    // gp_canvas_add_module created a canvas-owned table for this module; replace it with the real root table
    if (new_idx < static_cast<int>(ctx->module_tables.size())) {
        if (ctx->module_tables[new_idx] && ctx->module_table_owned[new_idx]) {
            gp_table_destroy(ctx->module_tables[new_idx]);
        }
        ctx->module_tables[new_idx] = root;
        ctx->module_table_owned[new_idx] = 0;
    }
    // sync layout to reflect root table IO
    // Ensure the sync runs even if the canvas isn't focused on this module by
    // temporarily setting focus so table-driven IO rows are created.
    int prev_focused = ctx->focused_module;
    ctx->focused_module = new_idx;
    sync_module_table_io_layout(ctx, new_idx);
    ctx->focused_module = prev_focused;
    ctx->root_module_idx = new_idx;
    return new_idx;
}

static RopeSim* canvas_require_root_sim(GP_CanvasContextImpl* ctx) {
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return nullptr;
    RopeSim* sim = gp_table_get_rope_sim(root);
    if (!sim) {
        int max_segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : std::max(2, ctx->sim_segs);
        sim = rope_sim_create(1024, max_segs);
        gp_table_attach_rope_sim(root, sim, 1);
    }
    // diagnostic: report when a sim is returned (helps detect root-sim lifecycle)
#if defined(GP_CANVAS_DEBUG_PRINTF)
    printf("canvas_require_root_sim: root=%p sim=%p\n", (void*)root, (void*)sim);
#endif
    return sim;
}

static void canvas_attach_tables_to_root_sim(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    RopeSim* sim = canvas_root_sim(ctx);
    if (!sim) return;
    for (size_t i = 0; i < ctx->module_tables.size(); ++i) {
        if (ctx->module_tables[i]) gp_table_attach_rope_sim(ctx->module_tables[i], sim, 0);
    }
}

static int canvas_dispatch_root_hit(GP_CanvasContextImpl* ctx, const GP_TableHitBox& hit) {
    if (!ctx) return 0;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return 0;
    return gp_table_dispatch_hit(root, &hit);
}

static int canvas_dispatch_root_action(GP_CanvasContextImpl* ctx, int action_id) {
    if (!ctx) return 0;
    // fast-path: explicit spawn-root action should create the synthetic
    // root module immediately so it's visible and clickable.
    if (action_id == CANVAS_ACT_SPAWN_ROOT) {
        canvas_ensure_root_module(ctx);
        return 1;
    }
    GP_TableHitBox hb{};
    hb.part = GP_TABLE_HIT_CELL;
    hb.aux0 = action_id;
    return canvas_dispatch_root_hit(ctx, hb);
}

// Query attached table for input/output IO key counts. If table is null,
// returns zero counts.
static void get_table_io_counts(GP_TableContext* t, int &out_in_count, int &out_out_count) {
    out_in_count = 0; out_out_count = 0;
    if (!t) return;
    const int cap = 4096;
    std::vector<unsigned long long> keys(cap);
    int nin = gp_table_enumerate_io_keys(t, 0, keys.data(), cap);
    if (nin > 0) out_in_count = nin;
    int nout = gp_table_enumerate_io_keys(t, 1, keys.data(), cap);
    if (nout > 0) out_out_count = nout;
}

static void canvas_append_io_row(GP_CanvasContextImpl* ctx, int module_idx, bool is_input, int attachment_count) {
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    if (module_idx < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[module_idx]) return;
    if (module_idx >= static_cast<int>(ctx->module_io_input_rows.size())) ctx->module_io_input_rows.resize(module_idx + 1);
    if (module_idx >= static_cast<int>(ctx->module_io_output_rows.size())) ctx->module_io_output_rows.resize(module_idx + 1);
    int count = std::clamp(attachment_count, 1, 32);
    ensure_module_row_order(ctx, module_idx);
    if (is_input) {
        ctx->module_io_input_rows[module_idx].push_back(count);
        if (module_idx < static_cast<int>(ctx->module_io_rows.size())) {
            ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Input, 0, ModuleToolKind::None, count});
        }
    } else {
        ctx->module_io_output_rows[module_idx].push_back(count);
        if (module_idx < static_cast<int>(ctx->module_io_rows.size())) {
            ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Output, 0, ModuleToolKind::None, count});
        }
    }
}

static uint64_t canvas_root_key_for_contact(int module_idx, int contact_idx) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(module_idx)) << 32) |
           (static_cast<uint64_t>(static_cast<uint32_t>(contact_idx)) << 16) |
           static_cast<uint64_t>(0);
}

static void canvas_collect_edges_for_contact(const GP_CanvasContextImpl* ctx, int module_idx, int contact_idx, std::vector<int> &out_indices) {
    out_indices.clear();
    if (!ctx) return;
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto &e = ctx->edges[i].desc;
        if ((e.a_module == module_idx && e.a_contact_idx == contact_idx) ||
            (e.b_module == module_idx && e.b_contact_idx == contact_idx)) {
            out_indices.push_back(static_cast<int>(i));
        }
    }
}

static void canvas_set_delta_mode_for_edge(GP_CanvasContextImpl* ctx, const GP_CanvasEdgeDesc &desc, bool delta_mode) {
    if (!ctx) return;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return;
    uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
    uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
    int edge_idx = -1;
    if (gp_table_edge_index_for_pair(root, ka, kb, &edge_idx)) {
        gp_table_edge_set_delta_mode(root, edge_idx, delta_mode ? 1 : 0);
    } else if (gp_table_edge_index_for_pair(root, kb, ka, &edge_idx)) {
        gp_table_edge_set_delta_mode(root, edge_idx, delta_mode ? 1 : 0);
    }
}

static void canvas_set_order_mode_for_edge(GP_CanvasContextImpl* ctx, const GP_CanvasEdgeDesc &desc, int order_mode) {
    if (!ctx) return;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return;
    uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
    uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
    int edge_idx = -1;
    if (gp_table_edge_index_for_pair(root, ka, kb, &edge_idx)) {
        gp_table_edge_set_order_mode(root, edge_idx, order_mode);
    } else if (gp_table_edge_index_for_pair(root, kb, ka, &edge_idx)) {
        gp_table_edge_set_order_mode(root, edge_idx, order_mode);
    }
}

static void canvas_set_subgroup_flags_for_edge(GP_CanvasContextImpl* ctx, const GP_CanvasEdgeDesc &desc, uint32_t flags) {
    if (!ctx) return;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (!root) return;
    uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
    uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
    int edge_idx = -1;
    if (gp_table_edge_index_for_pair(root, ka, kb, &edge_idx)) {
        gp_table_edge_set_subgroup_flags(root, edge_idx, flags);
    } else if (gp_table_edge_index_for_pair(root, kb, ka, &edge_idx)) {
        gp_table_edge_set_subgroup_flags(root, edge_idx, flags);
    }
}

static void canvas_apply_subgroup_to_edge_idx(GP_CanvasContextImpl* ctx, int edge_idx, uint32_t flags) {
    if (!ctx || edge_idx < 0 || edge_idx >= static_cast<int>(ctx->edges.size())) return;
    ctx->edges[static_cast<size_t>(edge_idx)].subgroup_flags = flags;
    canvas_set_subgroup_flags_for_edge(ctx, ctx->edges[static_cast<size_t>(edge_idx)].desc, flags);
}

static void canvas_remove_edge_at(GP_CanvasContextImpl* ctx, int edge_idx) {
    if (!ctx || edge_idx < 0 || edge_idx >= static_cast<int>(ctx->edges.size())) return;
    const auto desc = ctx->edges[static_cast<size_t>(edge_idx)].desc;
    GP_TableContext* root = canvas_ensure_root_table(ctx);
    if (root) {
        uint64_t ka = canvas_root_key_for_contact(desc.a_module, desc.a_contact_idx);
        uint64_t kb = canvas_root_key_for_contact(desc.b_module, desc.b_contact_idx);
        gp_table_remove_edge_pair(root, ka, kb);
    }
    ctx->edges.erase(ctx->edges.begin() + edge_idx);
}

static float point_segment_distance_sq(float px, float py, float ax, float ay, float bx, float by) {
    float vx = bx - ax;
    float vy = by - ay;
    float wx = px - ax;
    float wy = py - ay;
    float c1 = vx * wx + vy * wy;
    if (c1 <= 0.0f) {
        float dx = px - ax;
        float dy = py - ay;
        return dx * dx + dy * dy;
    }
    float c2 = vx * vx + vy * vy;
    if (c2 <= c1) {
        float dx = px - bx;
        float dy = py - by;
        return dx * dx + dy * dy;
    }
    float t = c1 / c2;
    float projx = ax + t * vx;
    float projy = ay + t * vy;
    float dx = px - projx;
    float dy = py - projy;
    return dx * dx + dy * dy;
}

static int canvas_pick_edge_by_rope(GP_CanvasContextImpl* ctx, int world_x, int world_y, float max_dist) {
    if (!ctx) return -1;
    RopeSim* sim = canvas_root_sim(ctx);
    if (!sim) return -1;
    float max_dist_sq = max_dist * max_dist;
    int best_edge = -1;
    float best_dist = max_dist_sq;
    for (size_t ei = 0; ei < ctx->edges.size(); ++ei) {
        int ridx = ctx->edges[ei].rope_idx;
        if (ridx < 0) continue;
        int vc = rope_sim_get_vertex_count(sim, ridx);
        if (vc < 2) continue;
        std::vector<float> verts(static_cast<size_t>(vc) * 2u);
        int got = rope_sim_get_vertices(sim, ridx, verts.data(), static_cast<int>(verts.size()));
        if (got < 2) continue;
        float dx0 = verts[0] - static_cast<float>(world_x);
        float dy0 = verts[1] - static_cast<float>(world_y);
        float dx1 = verts[(got - 1) * 2] - static_cast<float>(world_x);
        float dy1 = verts[(got - 1) * 2 + 1] - static_cast<float>(world_y);
        if ((dx0 * dx0 + dy0 * dy0) <= max_dist_sq) continue;
        if ((dx1 * dx1 + dy1 * dy1) <= max_dist_sq) continue;
        for (int vi = 0; vi < got - 1; ++vi) {
            float ax = verts[vi * 2];
            float ay = verts[vi * 2 + 1];
            float bx = verts[(vi + 1) * 2];
            float by = verts[(vi + 1) * 2 + 1];
            float dist_sq = point_segment_distance_sq(static_cast<float>(world_x), static_cast<float>(world_y), ax, ay, bx, by);
            if (dist_sq <= best_dist) {
                best_dist = dist_sq;
                best_edge = static_cast<int>(ei);
            }
        }
    }
    return best_edge;
}

// Ensure the attached table reflects the canvas' requested IO counts for the module.
// Creates columns/rows and LED cells (GP_TABLE_CELL_LEDS_ARG) to display counts.
static void sync_module_table_io_layout(GP_CanvasContextImpl* ctx, int module_idx) {
    // Simplified behavior: only update the focused module's attached table
    // to contain a single LED cell representing the input count from the
    // control-bar. This avoids complex layout logic while producing a
    // stateful LED cell that the existing table rasterizer will render.
    if (!ctx) return;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return;
    // keep module_plugin_instances aligned with module_io_rows for this module
    if (module_idx >= static_cast<int>(ctx->module_plugin_instances.size())) ctx->module_plugin_instances.resize(module_idx + 1);
    if (module_idx < static_cast<int>(ctx->module_io_rows.size())) {
        const size_t want = ctx->module_io_rows[module_idx].size();
        if (ctx->module_plugin_instances[module_idx].size() < want) ctx->module_plugin_instances[module_idx].resize(want);
    }
    // Stage modules get their own fixed IO layout with explicit ports.
    if (module_idx < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[module_idx]) {
        GP_TableContext* t = (module_idx < static_cast<int>(ctx->module_tables.size())) ? ctx->module_tables[module_idx] : nullptr;
        GP_StageContext* st = (module_idx < static_cast<int>(ctx->module_stages.size())) ? ctx->module_stages[module_idx] : nullptr;
        if (!t) return;
        // ensure IO counts for stage ports (one in, one out)
        if (module_idx >= static_cast<int>(ctx->module_io_in_count.size())) ctx->module_io_in_count.resize(module_idx + 1, 0);
        if (module_idx >= static_cast<int>(ctx->module_io_out_count.size())) ctx->module_io_out_count.resize(module_idx + 1, 0);
        ctx->module_io_in_count[module_idx] = 1;
        ctx->module_io_out_count[module_idx] = 1;

        // keep the stage table compact; give it a faint opaque backing so LEDs are visible
        GP_TableStyle st_style{};
        gp_table_get_style(t, &st_style);
        const auto &mod = ctx->modules[module_idx];
        st_style.width_px = std::max(1, mod.w);
        st_style.row_h_px = 18;
        // avoid pushing columns off-screen by keeping the name gutter tiny
        st_style.name_w_px = 8;
        // keep some translucency so the stage is still visible beneath the table surface
        st_style.bg_rgba[0] = 18; st_style.bg_rgba[1] = 18; st_style.bg_rgba[2] = 24; st_style.bg_rgba[3] = 180;
        st_style.bg_sel_rgba[0] = 28; st_style.bg_sel_rgba[1] = 28; st_style.bg_sel_rgba[2] = 36; st_style.bg_sel_rgba[3] = 200;
        gp_table_set_style(t, &st_style);

        // Column sizing: content + input LED + output LED. Keep totals within table width.
        int avail_w = std::max(16, st_style.width_px - st_style.name_w_px);
        GP_TableColumn cols[3];
        cols[0].kind = GP_TABLE_CELL_TEXT;
        cols[0].width_px = std::max(64, avail_w - 144);
        cols[0].align = 0;
        cols[1].kind = GP_TABLE_CELL_LEDS_ARG;
        cols[1].width_px = 72;
        cols[1].align = 0;
        cols[2].kind = GP_TABLE_CELL_LEDS_ARG;
        cols[2].width_px = 72;
        cols[2].align = 0;
        gp_table_set_columns(t, cols, 3);

        auto fill_text = [](GP_TableCell &c, const char* txt) {
            c.kind = GP_TABLE_CELL_TEXT;
            std::snprintf(c.text, sizeof(c.text), "%s", txt);
        };
        auto fill_led_single = [](GP_TableCell &c) {
            c.kind = GP_TABLE_CELL_LEDS_ARG;
            c.value = 1.0f;      // one LED
            c.flags = 1u;        // on mask (bit0)
            c.reserved0 = 1;     // active mask bit0
        };

        // Rows: input port, output port, and a flex "content" row that the stage
        // render is bound into as a GP_TABLE_CELL_IMAGE.
        GP_TableRow rows[3];
        std::memset(rows, 0, sizeof(rows));
        for (int i = 0; i < 3; ++i) {
            rows[i].kind = GP_TABLE_ROW_DEVICE;
            rows[i].depth = 0;
            rows[i].expanded = 1;
            rows[i].selected = 0;
            rows[i].cell_count = 3;
        }
        fill_text(rows[0].cells[0], LABEL_IO_INPUT);
        fill_led_single(rows[0].cells[1]);
        fill_text(rows[0].cells[2], "");
        fill_text(rows[1].cells[0], LABEL_IO_OUTPUT);
        fill_text(rows[1].cells[1], "");
        fill_led_single(rows[1].cells[2]);

        rows[2].cells[0].kind = GP_TABLE_CELL_IMAGE;
        rows[2].cells[0].image = nullptr;
        fill_text(rows[2].cells[1], "");
        fill_text(rows[2].cells[2], "");
        // reserved0 < 0 => flex row (absorbs remaining vertical space when rendered into a larger buffer)
        rows[2].reserved0 = -1;
        gp_table_set_rows(t, rows, 3);

        // update metadata so control surfaces know there are two contacts
        if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) ctx->module_io_rows.resize(module_idx + 1);
        ctx->module_io_rows[module_idx].clear();
        ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Input, 0, ModuleToolKind::None, 1});
        ctx->module_io_rows[module_idx].push_back({ModuleRowKind::Output, 1, ModuleToolKind::None, 1});
        if (module_idx >= static_cast<int>(ctx->module_plugin_instances.size())) ctx->module_plugin_instances.resize(module_idx + 1);
        ctx->module_plugin_instances[module_idx].clear();
        ctx->module_plugin_instances[module_idx].resize(ctx->module_io_rows[module_idx].size());
        if (module_idx >= static_cast<int>(ctx->module_table_rows.size())) ctx->module_table_rows.resize(module_idx + 1);
        ctx->module_table_rows[module_idx].clear();
        if (module_idx >= static_cast<int>(ctx->module_stack_snapshots.size())) ctx->module_stack_snapshots.resize(module_idx + 1);
        ctx->module_stack_snapshots[module_idx].clear();

        auto bind_port = [&](int row_idx, int led_col_idx, bool is_output, int channel) {
            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(row_idx)) << 32) |
                (static_cast<uint64_t>(static_cast<uint32_t>(led_col_idx)) << 16) | // LED column index
                static_cast<uint64_t>(0); // single LED
            int float_tid = ValueTypeRegistry::global().builtin(VT_FLOAT32);
            gp_table_set_key_type_hint(t, key, /*type_id=*/float_tid, is_output ? 0 : 1, is_output ? 1 : 0);
            if (st) gp_table_enqueue_bind_stage_port(t, key, st, is_output ? 1 : 0, channel);
        };
        bind_port(0, /*led_col_idx=*/1, /*is_output=*/false, /*channel=*/0);
        bind_port(1, /*led_col_idx=*/2, /*is_output=*/true,  /*channel=*/0);
        return;
    }
    if (module_idx >= static_cast<int>(ctx->module_tables.size())) return;
    GP_TableContext* t = ctx->module_tables[module_idx];
    if (!t) return;
    // only touch the currently focused module to avoid clobbering other tables
    if (ctx->focused_module != module_idx) return;
    if (module_idx >= static_cast<int>(ctx->module_io_input_rows.size())) ctx->module_io_input_rows.resize(module_idx + 1);
    if (module_idx >= static_cast<int>(ctx->module_io_output_rows.size())) ctx->module_io_output_rows.resize(module_idx + 1);
    int legacy_in_count = (module_idx < static_cast<int>(ctx->module_io_in_count.size())) ? ctx->module_io_in_count[module_idx] : 0;
    int legacy_out_count = (module_idx < static_cast<int>(ctx->module_io_out_count.size())) ? ctx->module_io_out_count[module_idx] : 0;
    auto &input_rows = ctx->module_io_input_rows[module_idx];
    auto &output_rows = ctx->module_io_output_rows[module_idx];
    if (input_rows.empty() && legacy_in_count > 0) input_rows.assign(static_cast<size_t>(legacy_in_count), 1);
    if (output_rows.empty() && legacy_out_count > 0) output_rows.assign(static_cast<size_t>(legacy_out_count), 1);
    ensure_module_row_order(ctx, module_idx);
    if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) return;
    auto &ordered_rows = ctx->module_io_rows[module_idx];
    int in_count = 0;
    int out_count = 0;
    for (const auto &row : ordered_rows) {
        if (row.kind == ModuleRowKind::Input) in_count += std::max(0, row.attachment_count);
        if (row.kind == ModuleRowKind::Output) out_count += std::max(0, row.attachment_count);
    }
    if (module_idx >= static_cast<int>(ctx->module_io_in_count.size())) ctx->module_io_in_count.resize(module_idx + 1, 0);
    if (module_idx >= static_cast<int>(ctx->module_io_out_count.size())) ctx->module_io_out_count.resize(module_idx + 1, 0);
    ctx->module_io_in_count[module_idx] = in_count;
    ctx->module_io_out_count[module_idx] = out_count;

    // always rebuild the table layout to reflect ordered IO rows

    // If there's no attached table but the UI has non-zero IO counts, create
    // a canvas-owned table so the user sees the LED cells immediately.
    if (!t && (in_count > 0 || (module_idx < static_cast<int>(ctx->module_io_out_count.size()) && ctx->module_io_out_count[module_idx] > 0))) {
        GP_TableContext* nt = gp_table_create(nullptr);
        gp_table_set_prospective_mode(nt, 1);
        gp_table_prospective_set_params(nt, 8, 4.0f, 0.0f);
        t = nt;
    }
    if (!t) return;

    GP_TableStyle st{};
    gp_table_get_style(t, &st);
    const auto &mod = ctx->modules[module_idx];
    st.width_px = std::max(1, mod.w);
    st.name_w_px = 0;
    if (st.row_h_px <= 0) st.row_h_px = 22;
    gp_table_set_style(t, &st);

    int content_w = std::max(1, st.width_px);
    int led_w = std::max(30, std::min(60, content_w / 3));
    int text_w = std::max(1, content_w - 2 * led_w);
    GP_TableColumn cols[kModuleColCount];
    cols[kModuleColLeftLed] = { GP_TABLE_CELL_LEDS_ARG, led_w, 0 };
    cols[kModuleColText] = { GP_TABLE_CELL_TEXT, text_w, 0 };
    cols[kModuleColRightLed] = { GP_TABLE_CELL_LEDS_ARG, led_w, 0 };
    gp_table_set_columns(t, cols, kModuleColCount);

    int base_row_h = st.row_h_px > 0 ? st.row_h_px : 22;
    int tool_row_h = std::max(14, base_row_h - 6);
    int stack_value_row_h = std::max(12, base_row_h - 8);

    int total_rows = static_cast<int>(ordered_rows.size());
    if (total_rows <= 0) {
        GP_TableRow prow{}; memset(&prow, 0, sizeof(prow));
        prow.kind = GP_TABLE_ROW_HEADER; prow.depth = 0; prow.expanded = 1; prow.selected = 0;
        prow.cell_count = kModuleColCount;
        prow.reserved0 = tool_row_h;
        prow.cells[kModuleColLeftLed].kind = GP_TABLE_CELL_TEXT;
        prow.cells[kModuleColText].kind = GP_TABLE_CELL_TEXT;
        prow.cells[kModuleColRightLed].kind = GP_TABLE_CELL_TEXT;
        gp_table_set_rows(t, &prow, 1);
        if (module_idx >= static_cast<int>(ctx->module_table_rows.size())) ctx->module_table_rows.resize(module_idx + 1);
        ctx->module_table_rows[module_idx].clear();
        return;
    }

    auto fill_text_cell = [](GP_TableCell &cell, const char* text) {
        cell.kind = GP_TABLE_CELL_TEXT;
        size_t len = std::min<std::size_t>(std::strlen(text), sizeof(cell.text) - 1);
        std::memcpy(cell.text, text, len);
        cell.text[len] = '\0';
    };

    auto fill_counter_cell = [](GP_TableCell &cell, int value, const char* label) {
        cell.kind = GP_TABLE_CELL_COUNTER;
        cell.value = static_cast<float>(std::max(0, value));
        cell.flags = 0;
        if (label && label[0] != '\0') {
            size_t len = std::min<std::size_t>(std::strlen(label), sizeof(cell.text) - 1);
            std::memcpy(cell.text, label, len);
            cell.text[len] = '\0';
        } else {
            cell.text[0] = '\0';
        }
    };

    auto fill_led_cell = [](GP_TableCell &cell, int count, uint32_t on_mask, uint32_t active_mask) {
        cell.kind = GP_TABLE_CELL_LEDS_ARG;
        int clamped = std::clamp(count, 1, 32);
        cell.value = static_cast<float>(clamped);
        cell.flags = on_mask;
        cell.reserved0 = static_cast<int32_t>(active_mask);
    };

    std::vector<GP_TableRow> rows;
    rows.reserve(static_cast<size_t>(total_rows));
    std::vector<ModuleIORow> row_meta;
    row_meta.reserve(static_cast<size_t>(total_rows));

    auto append_row = [&](const char* label, ModuleRowKind kind, int contact_idx, int attachment_count, int tool_value, ModuleToolKind tool_kind, ModuleToolOrigin tool_origin, int row_h, bool left_active, bool right_active) {
        GP_TableRow r{};
        memset(&r, 0, sizeof(r));
        if (kind == ModuleRowKind::Tool) {
            r.kind = (tool_origin == ModuleToolOrigin::Plugin) ? GP_TABLE_ROW_DEVICE : GP_TABLE_ROW_HEADER;
        } else {
            r.kind = GP_TABLE_ROW_DEVICE;
        }
        r.depth = 0;
        r.expanded = 1;
        r.selected = 0;
        r.cell_count = kModuleColCount;
        r.reserved0 = row_h;
        uint32_t mask = (attachment_count >= 32) ? 0xFFFFFFFFu : ((1u << std::max(1, attachment_count)) - 1u);
        uint32_t left_mask = left_active ? mask : 0u;
        uint32_t right_mask = right_active ? mask : 0u;
        if (kind == ModuleRowKind::Tool) {
            fill_text_cell(r.cells[kModuleColLeftLed], "");
            fill_text_cell(r.cells[kModuleColRightLed], "");
        } else {
            if (left_active) {
                fill_led_cell(r.cells[kModuleColLeftLed], std::max(1, attachment_count), left_mask, left_mask);
            } else {
                fill_text_cell(r.cells[kModuleColLeftLed], "");
            }
            if (right_active) {
                fill_led_cell(r.cells[kModuleColRightLed], std::max(1, attachment_count), right_mask, right_mask);
            } else {
                fill_text_cell(r.cells[kModuleColRightLed], "");
            }
        }
        if (kind == ModuleRowKind::Tool && tool_kind == ModuleToolKind::TableNumber) {
            fill_counter_cell(r.cells[kModuleColText], tool_value, label);
        } else {
            fill_text_cell(r.cells[kModuleColText], label);
        }
        rows.push_back(r);
        row_meta.push_back({kind, contact_idx, tool_kind, attachment_count, tool_origin});
    };

    auto append_stack_display_rows = [&](int logical_row_idx, ModuleToolOrigin tool_origin) {
        GP_TableRow header{};
        memset(&header, 0, sizeof(header));
        header.kind = (tool_origin == ModuleToolOrigin::Plugin) ? GP_TABLE_ROW_DEVICE : GP_TABLE_ROW_HEADER;
        header.depth = 0;
        header.expanded = 1;
        header.selected = 0;
        header.cell_count = kModuleColCount;
        header.reserved0 = tool_row_h;
        fill_text_cell(header.cells[kModuleColLeftLed], "");
        fill_text_cell(header.cells[kModuleColText], LABEL_TOOL_STACK);
        fill_text_cell(header.cells[kModuleColRightLed], "");
        rows.push_back(header);
        row_meta.push_back({ModuleRowKind::Tool, logical_row_idx, ModuleToolKind::StackDisplay, 0, tool_origin, std::string()});

        const std::vector<float>* snapshot = nullptr;
        if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_stack_snapshots.size())) {
            auto &snapshots = ctx->module_stack_snapshots[module_idx];
            auto it = snapshots.find(logical_row_idx);
            if (it != snapshots.end()) snapshot = &it->second;
        }
        if (!snapshot || snapshot->empty()) return;

        int display_idx = 0;
        for (auto it = snapshot->rbegin(); it != snapshot->rend(); ++it, ++display_idx) {
            GP_TableRow vr{};
            memset(&vr, 0, sizeof(vr));
            vr.kind = GP_TABLE_ROW_NOTE;
            vr.depth = 0;
            vr.expanded = 1;
            vr.selected = 0;
            vr.cell_count = kModuleColCount;
            vr.reserved0 = stack_value_row_h;
            char buf[96];
            std::snprintf(buf, sizeof(buf), "%d: %.6g", display_idx, *it);
            fill_text_cell(vr.cells[kModuleColLeftLed], "");
            fill_text_cell(vr.cells[kModuleColText], buf);
            fill_text_cell(vr.cells[kModuleColRightLed], "");
            rows.push_back(vr);
            row_meta.push_back({ModuleRowKind::Tool, -1, ModuleToolKind::StackDisplay, 0, tool_origin, std::string()});
        }
    };

    int input_contact = 0;
    int output_contact = 0;
    for (size_t row_idx = 0; row_idx < ordered_rows.size(); ++row_idx) {
        const auto &row = ordered_rows[row_idx];
        if (row.kind == ModuleRowKind::Tool) {
            if (row.tool == ModuleToolKind::StackDisplay) {
                append_stack_display_rows(static_cast<int>(row_idx), row.tool_origin);
            } else {
                int tool_value = (row.tool == ModuleToolKind::TableNumber) ? std::max(0, row.attachment_count) : kModuleLedPerSide;
                std::string label_str;
                if (row.tool_origin == ModuleToolOrigin::Plugin) {
                    // prefer live instance name if present
                    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_plugin_instances.size())) {
                        const auto &insts = ctx->module_plugin_instances[module_idx];
                        if (row_idx < insts.size() && insts[row_idx]) {
                            try { label_str = insts[row_idx]->name(); } catch(...) { label_str = row.plugin_id; }
                        }
                    }
                    if (label_str.empty()) {
                        // fallback: use registry entry name if available
                        auto *entry = tool_registry_global().find(row.plugin_id);
                        if (entry) label_str = entry->name;
                        else label_str = row.plugin_id.empty() ? tool_label(row.tool) : row.plugin_id;
                    }
                } else {
                    label_str = tool_label(row.tool);
                }
                append_row(label_str.c_str(), ModuleRowKind::Tool, static_cast<int>(row_idx), kModuleLedPerSide, tool_value, row.tool, row.tool_origin, tool_row_h, false, false);
            }
            continue;
        }
        if (row.kind == ModuleRowKind::Input) {
            int remaining = std::clamp(row.attachment_count, 1, 32);
            while (remaining > 0) {
                int attachments = std::min(kModuleLedPerSide, remaining);
                append_row(LABEL_IO_INPUT, ModuleRowKind::Input, input_contact, attachments, attachments, ModuleToolKind::None, ModuleToolOrigin::Builtin, base_row_h, true, false);
                input_contact += attachments;
                remaining -= attachments;
            }
        } else if (row.kind == ModuleRowKind::Output) {
            int remaining = std::clamp(row.attachment_count, 1, 32);
            while (remaining > 0) {
                int attachments = std::min(kModuleLedPerSide, remaining);
                append_row(LABEL_IO_OUTPUT, ModuleRowKind::Output, in_count + output_contact, attachments, attachments, ModuleToolKind::None, ModuleToolOrigin::Builtin, base_row_h, false, true);
                output_contact += attachments;
                remaining -= attachments;
            }
        }
    }

    auto row_height_for = [&](const GP_TableRow& row) {
        int h = row.reserved0 > 0 ? row.reserved0 : base_row_h;
        return std::max(1, h);
    };
    int table_content_h = 0;
    for (const auto &r : rows) table_content_h += row_height_for(r);
    bool is_stage = (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_is_stage.size()) && ctx->module_is_stage[module_idx]);
    if (!is_stage) {
        int preview_target = std::max(1, mod.w);
        int target_h = kModuleTopUiHeight + table_content_h + preview_target;
        if (mod.h < target_h && module_idx >= 0 && module_idx < static_cast<int>(ctx->modules.size())) {
            ctx->modules[module_idx].h = target_h;
        }
    }
    gp_table_set_rows(t, rows.data(), static_cast<int>(rows.size()));
    if (module_idx >= static_cast<int>(ctx->module_table_rows.size())) ctx->module_table_rows.resize(module_idx + 1);
    ctx->module_table_rows[module_idx] = row_meta;
    // Annotate LED keys for created rows with explicit input/output hints
    // so the table renderer's glow pass can recognize IO roles.
    for (int ri = 0; ri < static_cast<int>(rows.size()); ++ri) {
        ModuleRowKind kind = row_meta[ri].kind;
        if (kind == ModuleRowKind::Tool) continue;
        bool is_input = (kind == ModuleRowKind::Input);
        int col_idx = is_input ? kModuleColLeftLed : kModuleColRightLed;
        if (col_idx >= rows[ri].cell_count) continue;
        const GP_TableCell &cell = rows[ri].cells[col_idx];
        int led_count = std::max(1, row_meta[ri].attachment_count);
        if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
            int c = std::max(0, std::min(32, static_cast<int>(cell.value)));
            if (c == 0) c = static_cast<int>(cell.flags & 0xFFu);
            if (c == 0) c = 12;
            led_count = std::min(led_count, c);
        }
        for (int li = 0; li < led_count; ++li) {
            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(ri)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(col_idx)) << 16) | static_cast<uint64_t>(static_cast<uint32_t>(li));
            int float_tid = ValueTypeRegistry::global().builtin(VT_FLOAT32);
            gp_table_set_key_type_hint(t, key, float_tid, is_input ? 1 : 0, is_input ? 0 : 1);
        }
    }
    
    auto ensure_layout = [&](std::vector<MolexLayoutInfo> &arr) {
        if (module_idx >= static_cast<int>(arr.size())) arr.resize(module_idx + 1);
    };
    ensure_layout(ctx->module_input_layout);
    ensure_layout(ctx->module_output_layout);
    ctx->module_input_layout[module_idx] = make_molex_layout(module_idx, true, in_count);
    ctx->module_output_layout[module_idx] = make_molex_layout(module_idx, false, out_count);
    return;
}

static void canvas_push_tool_to_focused(GP_CanvasContextImpl* ctx, ModuleToolKind tool, ModuleToolOrigin origin) {
    if (!ctx) return;
    int focused = ctx->focused_module;
    if (focused < 0 || focused >= static_cast<int>(ctx->modules.size())) return;
    if (focused >= static_cast<int>(ctx->module_tool_stack.size())) ctx->module_tool_stack.resize(focused + 1);
    ensure_module_row_order(ctx, focused);
    if (focused >= static_cast<int>(ctx->module_io_rows.size())) return;
    auto &rows = ctx->module_io_rows[focused];
    auto insert_tool_row = [&](size_t idx) {
        ModuleIORow row;
        row.kind = ModuleRowKind::Tool;
        row.contact_idx = -1;
        row.tool = tool;
        row.attachment_count = (tool == ModuleToolKind::TableNumber) ? 1 : 0;
        row.tool_origin = origin;
        rows.insert(rows.begin() + static_cast<ptrdiff_t>(idx), row);
    };
    if (tool == ModuleToolKind::KeyboardListener) {
        auto it = std::find_if(rows.begin(), rows.end(), [](const ModuleIORow &r) {
            return r.kind == ModuleRowKind::Tool && r.tool == ModuleToolKind::MouseListener;
        });
        if (it != rows.end()) {
            insert_tool_row(static_cast<size_t>(std::distance(rows.begin(), it) + 1));
        } else {
            rows.push_back({ModuleRowKind::Tool, -1, tool, (tool == ModuleToolKind::TableNumber) ? 1 : 0, origin, std::string()});
        }
    } else if (tool == ModuleToolKind::MouseListener) {
        auto it = std::find_if(rows.begin(), rows.end(), [](const ModuleIORow &r) {
            return r.kind == ModuleRowKind::Tool && r.tool == ModuleToolKind::KeyboardListener;
        });
        if (it != rows.end()) {
            insert_tool_row(static_cast<size_t>(std::distance(rows.begin(), it)));
        } else {
            rows.push_back({ModuleRowKind::Tool, -1, tool, (tool == ModuleToolKind::TableNumber) ? 1 : 0, origin, std::string()});
        }
    } else {
        rows.push_back({ModuleRowKind::Tool, -1, tool, (tool == ModuleToolKind::TableNumber) ? 1 : 0, origin, std::string()});
    }
    auto &tools = ctx->module_tool_stack[focused];
    tools.clear();
    for (const auto &row : rows) {
        if (row.kind == ModuleRowKind::Tool) tools.push_back(row.tool);
    }
    sync_module_table_io_layout(ctx, focused);
    update_canvas_scroll_state(ctx, /*pull_from_container=*/false);
    ctx->tool_menu_open = false;
    ctx->rope_menu_open = false;
    ctx->plugin_menu_open = false;
    ctx->plugin_menu_module_idx = -1;
    ctx->module_menu_open = false;
    ctx->module_menu_module_idx = -1;
    ctx->selected_tool_table = 0;
    // When the mouse tool is explicitly added to a module, auto-bind the first
    // available four receive ports so it can start receiving events immediately.
    if (tool == ModuleToolKind::MouseListener) {
        gp_canvas_autobind_mouse_ports(reinterpret_cast<GP_CanvasContext*>(ctx), focused, 4);
    }
}

static void canvas_push_plugin_tool_to_focused(GP_CanvasContextImpl* ctx, const std::string &plugin_id) {
    if (!ctx) return;
    int focused = ctx->focused_module;
    if (focused < 0 || focused >= static_cast<int>(ctx->modules.size())) return;
    if (focused >= static_cast<int>(ctx->module_tool_stack.size())) ctx->module_tool_stack.resize(focused + 1);
    ensure_module_row_order(ctx, focused);
    if (focused >= static_cast<int>(ctx->module_io_rows.size())) return;
    auto &rows = ctx->module_io_rows[focused];
    ModuleIORow row;
    row.kind = ModuleRowKind::Tool;
    row.contact_idx = -1;
    row.tool = ModuleToolKind::None;
    row.attachment_count = 0;
    row.tool_origin = ModuleToolOrigin::Plugin;
    row.plugin_id = plugin_id;
    rows.push_back(row);
    // ensure plugin instances vector aligns with rows
    if (focused >= static_cast<int>(ctx->module_plugin_instances.size())) ctx->module_plugin_instances.resize(focused + 1);
    if (ctx->module_plugin_instances[focused].size() < rows.size()) ctx->module_plugin_instances[focused].resize(rows.size());

    // try to create an instance for the plugin id via the ToolRegistry
    try {
        auto inst = tool_registry_global().create(plugin_id);
        if (inst) {
            // store instance in parallel vector at same index as the pushed row
            ToolInitContext tctx{};
            tctx.user = reinterpret_cast<void*>(static_cast<intptr_t>(focused));
            try { inst->initialize(tctx); } catch (...) {}
            ctx->module_plugin_instances[focused][rows.size() - 1] = std::move(inst);
        }
    } catch (...) {
        // creation failed; leave null instance
    }
    // Apply host-driven autobind hints advertised by the plugin.
    if (const auto* entry = tool_registry_global().find(plugin_id)) {
        if (entry->auto_mouse_ports > 0) {
            gp_canvas_autobind_mouse_ports(reinterpret_cast<GP_CanvasContext*>(ctx), focused, entry->auto_mouse_ports);
        }
        if (entry->auto_keyboard_ports > 0) {
            gp_canvas_autobind_keyboard_ports(reinterpret_cast<GP_CanvasContext*>(ctx), focused, entry->auto_keyboard_ports);
        }
    }

    auto &tools = ctx->module_tool_stack[focused];
    tools.clear();
}

static int canvas_handle_module_counter_hit(GP_CanvasContextImpl* ctx, int module_idx, const GP_TableHitBox& found) {
    if (!ctx) return 0;
    if (found.part != GP_TABLE_HIT_COUNTER_DEC && found.part != GP_TABLE_HIT_COUNTER_INC) return 0;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->module_table_rows.size())) return 0;
    const auto &rows = ctx->module_table_rows[module_idx];
    if (found.row_idx < 0 || found.row_idx >= static_cast<int>(rows.size())) return 0;
    const auto &meta = rows[found.row_idx];
    if (meta.kind != ModuleRowKind::Tool || meta.tool != ModuleToolKind::TableNumber) return 0;
    if (module_idx >= static_cast<int>(ctx->module_io_rows.size())) return 0;
    int tool_row_idx = meta.contact_idx;
    if (tool_row_idx < 0 || tool_row_idx >= static_cast<int>(ctx->module_io_rows[module_idx].size())) return 0;
    auto &row = ctx->module_io_rows[module_idx][tool_row_idx];
    if (row.kind != ModuleRowKind::Tool || row.tool != ModuleToolKind::TableNumber) return 0;
    int delta = (found.part == GP_TABLE_HIT_COUNTER_INC) ? 1 : -1;
    row.attachment_count = std::clamp(row.attachment_count + delta, 0, 99);
    sync_module_table_io_layout(ctx, module_idx);
    return 1;
}

static int canvas_handle_module_led_hit(GP_CanvasContextImpl* ctx, int module_idx, const GP_TableHitBox& found) {
    if (!ctx) return 0;
    if (module_idx < 0 || module_idx >= static_cast<int>(ctx->modules.size())) return 0;
    const auto& m = ctx->modules[module_idx];
    if (found.part != GP_TABLE_HIT_LED && found.part != GP_TABLE_HIT_LED_ARG && found.part != GP_TABLE_HIT_LED_TABLE) return 0;
    int ax = m.x + (found.x0 + found.x1) / 2;
    int ay = m.y + (found.y0 + found.y1) / 2;
    int contact_idx = resolve_contact_index(ctx, module_idx, found);
    if (contact_idx < 0) return 0;
    ctx->focused_module = module_idx;
    if (ctx->edge_order_tool_active) {
        std::vector<int> edge_indices;
        canvas_collect_edges_for_contact(ctx, module_idx, contact_idx, edge_indices);
        if (!edge_indices.empty()) {
            for (int idx : edge_indices) {
                if (idx < 0 || idx >= static_cast<int>(ctx->edges.size())) continue;
                canvas_set_order_mode_for_edge(ctx, ctx->edges[static_cast<size_t>(idx)].desc, ctx->edge_order_value);
            }
            ctx->selected.module = -1; ctx->selected.contact_idx = -1; ctx->selected.left = -1; ctx->selected.anchor_x = -1; ctx->selected.anchor_y = -1;
            if (ctx->prospective_rope_idx >= 0) ctx->prospective_rope_idx = -1;
            return 1;
        }
    }
    if (ctx->selected_tool_edge >= 0 && ctx->selected_tool_edge != 0) {
        std::vector<int> edge_indices;
        canvas_collect_edges_for_contact(ctx, module_idx, contact_idx, edge_indices);
        if (!edge_indices.empty()) {
            if (ctx->selected_tool_edge == 1) {
                std::sort(edge_indices.begin(), edge_indices.end(), std::greater<int>());
                for (int idx : edge_indices) canvas_remove_edge_at(ctx, idx);
            } else {
                bool delta_mode = (ctx->selected_tool_edge == 2);
                for (int idx : edge_indices) {
                    if (idx < 0 || idx >= static_cast<int>(ctx->edges.size())) continue;
                    canvas_set_delta_mode_for_edge(ctx, ctx->edges[static_cast<size_t>(idx)].desc, delta_mode);
                }
            }
            ctx->selected.module = -1; ctx->selected.contact_idx = -1; ctx->selected.left = -1; ctx->selected.anchor_x = -1; ctx->selected.anchor_y = -1;
            if (ctx->prospective_rope_idx >= 0) ctx->prospective_rope_idx = -1;
            return 1;
        }
    }
    if (ctx->selected_tool_subgroup_flags != 0u) {
        std::vector<int> edge_indices;
        canvas_collect_edges_for_contact(ctx, module_idx, contact_idx, edge_indices);
        if (!edge_indices.empty()) {
            uint32_t flags = ctx->selected_tool_subgroup_flags;
            for (int idx : edge_indices) {
                canvas_apply_subgroup_to_edge_idx(ctx, idx, flags);
            }
            ctx->selected.module = -1; ctx->selected.contact_idx = -1; ctx->selected.left = -1; ctx->selected.anchor_x = -1; ctx->selected.anchor_y = -1;
            if (ctx->prospective_rope_idx >= 0) ctx->prospective_rope_idx = -1;
            return 1;
        }
    }
    // Determine producer/consumer using row metadata when available.
    bool is_producer = false;
    bool resolved_role = false;
    if (found.row_idx == kModuleFrameRowSend || found.row_idx == kModuleFrameRowReceive) {
        is_producer = (found.row_idx == kModuleFrameRowSend);
        resolved_role = true;
    }
    const std::vector<ModuleIORow>* rows_ptr = nullptr;
    if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_table_rows.size()) &&
        !ctx->module_table_rows[module_idx].empty()) {
        rows_ptr = &ctx->module_table_rows[module_idx];
    } else if (module_idx >= 0 && module_idx < static_cast<int>(ctx->module_io_rows.size())) {
        rows_ptr = &ctx->module_io_rows[module_idx];
    }
    if (!resolved_role && rows_ptr) {
        const auto &rows = *rows_ptr;
        if (found.row_idx >= 0 && found.row_idx < static_cast<int>(rows.size())) {
            const auto &meta = rows[found.row_idx];
            if (meta.kind == ModuleRowKind::Input || meta.kind == ModuleRowKind::Output) {
                bool is_input = (meta.kind == ModuleRowKind::Input);
                is_producer = !is_input;
                resolved_role = true;
            }
        }
    }
    if (!resolved_role && module_idx < static_cast<int>(ctx->module_io_in_count.size()) && module_idx < static_cast<int>(ctx->module_io_out_count.size())) {
        int in_count = ctx->module_io_in_count[module_idx];
        int out_count = ctx->module_io_out_count[module_idx];
        int total = std::max(0, in_count) + std::max(0, out_count);
        if (total > 0 && contact_idx >= 0 && contact_idx < total) {
            bool is_input = (contact_idx < in_count);
            is_producer = !is_input;
            resolved_role = true;
        }
    }
    if (!resolved_role) {
        if (found.col_idx >= 0) is_producer = (found.col_idx != 0);
        else is_producer = (ax >= m.x + m.w / 2);
    }
    bool is_input = !is_producer;
    if (ctx->selected.module == -1) {
        ctx->selected.module = module_idx; ctx->selected.contact_idx = contact_idx; ctx->selected.left = is_producer ? 1 : 0;
        ctx->selected.anchor_x = ax; ctx->selected.anchor_y = ay;
        uint32_t connector_hash = lookup_molex_hash(ctx, module_idx, is_input, contact_idx);
        printf("gp_canvas_on_click: selecting table LED module=%d contact=%d producer=%d anchor=%d,%d hash=0x%08x\n",
            module_idx, contact_idx, ctx->selected.left, ax, ay, connector_hash);
        RopeSim* sim = canvas_require_root_sim(ctx);
        if (sim) {
            int ax0 = ax, ay0 = ay;
            int bx = ax, by = ay;
            int segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : ctx->sim_segs;
            float slack = ctx->sim_slack;
            ctx->prospective_rope_idx = rope_sim_add_rope(sim, static_cast<float>(ax0), static_cast<float>(ay0), static_cast<float>(bx), static_cast<float>(by), segs, slack);
            printf("gp_canvas_on_click: created prospective rope %d for table LED %d/%d\n", ctx->prospective_rope_idx, module_idx, contact_idx);
        }
        return 1;
    }
    bool sel_producer = (ctx->selected.left == 1);
    // allow connecting producer -> consumer only
        if (sel_producer && !is_producer) {
            GP_CanvasEdgeDesc e; e.a_module = ctx->selected.module; e.a_contact_idx = ctx->selected.contact_idx; e.b_module = module_idx; e.b_contact_idx = contact_idx;
            printf("gp_canvas_on_click: adding edge sel %d.%d->%d.%d prospective_rope=%d\n",
                ctx->selected.module, ctx->selected.contact_idx, module_idx, contact_idx, ctx->prospective_rope_idx);
            int ei = gp_canvas_add_edge_with_type(reinterpret_cast<GP_CanvasContext*>(ctx), &e, /*type_id=*/0);
            printf("gp_canvas_on_click: gp_canvas_add_edge returned %d\n", ei);
            if (ei >= 0 && ctx->prospective_rope_idx >= 0) {
                ctx->edges[ei].rope_idx = ctx->prospective_rope_idx;
                printf("gp_canvas_on_click: attached rope %d to edge %d\n", ctx->prospective_rope_idx, ei);
                ctx->prospective_rope_idx = -1;
            }
            if (ei >= 0) {
                canvas_sync_root_rope_endpoints(ctx, e,
                    static_cast<float>(ctx->selected.anchor_x), static_cast<float>(ctx->selected.anchor_y),
                    static_cast<float>(ax), static_cast<float>(ay));
            } else {
                printf("gp_canvas_on_click: no rope attached (prospective=%d)\n", ctx->prospective_rope_idx);
            }
            ctx->selected.module = -1; ctx->selected.contact_idx = -1; ctx->selected.left = -1; ctx->selected.anchor_x = -1; ctx->selected.anchor_y = -1;
            return 1;
        } else if (!sel_producer && is_producer) {
            GP_CanvasEdgeDesc e; e.a_module = module_idx; e.a_contact_idx = contact_idx; e.b_module = ctx->selected.module; e.b_contact_idx = ctx->selected.contact_idx;
            printf("gp_canvas_on_click: adding edge sel %d.%d->%d.%d prospective_rope=%d\n",
                module_idx, contact_idx, ctx->selected.module, ctx->selected.contact_idx, ctx->prospective_rope_idx);
            int ei = gp_canvas_add_edge_with_type(reinterpret_cast<GP_CanvasContext*>(ctx), &e, /*type_id=*/0);
            printf("gp_canvas_on_click: gp_canvas_add_edge returned %d\n", ei);
            if (ei >= 0 && ctx->prospective_rope_idx >= 0) {
                ctx->edges[ei].rope_idx = ctx->prospective_rope_idx;
                printf("gp_canvas_on_click: attached rope %d to edge %d\n", ctx->prospective_rope_idx, ei);
                ctx->prospective_rope_idx = -1;
            }
            if (ei >= 0) {
                canvas_sync_root_rope_endpoints(ctx, e,
                    static_cast<float>(ax), static_cast<float>(ay),
                    static_cast<float>(ctx->selected.anchor_x), static_cast<float>(ctx->selected.anchor_y));
            } else {
                printf("gp_canvas_on_click: no rope attached (prospective=%d)\n", ctx->prospective_rope_idx);
            }
            ctx->selected.module = -1; ctx->selected.contact_idx = -1; ctx->selected.left = -1; ctx->selected.anchor_x = -1; ctx->selected.anchor_y = -1;
            return 1;
    }
    // same direction: switch selection to this
    ctx->selected.module = module_idx; ctx->selected.contact_idx = contact_idx; ctx->selected.left = is_producer ? 1 : 0; ctx->selected.anchor_x = ax; ctx->selected.anchor_y = ay;
    return 1;
}
