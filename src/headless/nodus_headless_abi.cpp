#include "nodus_headless_abi.h"

#include "canvas_abi.h"
#include "table_abi.h"

#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

// Fixed subscriber/writer identity this runtime uses when it is itself the
// external party on a declared port's table edge (as opposed to another
// module in the graph). Any value distinct from module-derived keys works;
// module-derived keys are built from (module_idx<<32 | row_idx<<16), which
// never collides with this constant for valid module/row indices.
constexpr unsigned long long kHostKey = 0xE082D0A5E082D0A5ULL;

struct PortBinding {
    int32_t module_idx = -1;
    int32_t row_idx = -1;
    unsigned long long key = 0;
    int32_t edge_idx = -1;
};

} // namespace

struct NodusHeadlessGraph {
    GP_CanvasContext* ctx = nullptr;
    GP_TableContext* table = nullptr;

    std::unordered_map<std::string, PortBinding> input_ports;
    std::unordered_map<std::string, PortBinding> output_ports;

    std::mutex events_mu;
    std::deque<NodusHeadlessEvent> events;

    uint64_t tick_id = 0;
};

namespace {

void push_event(NodusHeadlessGraph* g, int32_t kind, int32_t module_idx, int32_t edge_idx, int32_t code = 0) {
    NodusHeadlessEvent ev{};
    ev.kind = kind;
    ev.module_idx = module_idx;
    ev.edge_idx = edge_idx;
    ev.tick_id = g->tick_id;
    ev.code = code;
    std::lock_guard<std::mutex> lock(g->events_mu);
    g->events.push_back(ev);
}

// Mirrors the key-resolution the real TickRequest builder uses
// (include/inl/canvas_abi_table_step.inl) so a declared port lands on the
// same table edge the scheduler itself would resolve for that row.
unsigned long long resolve_row_key(NodusHeadlessGraph* g, int32_t module_idx, int32_t row_idx) {
    unsigned long long key = gp_canvas_get_tool_row_id(g->ctx, module_idx, row_idx);
    if (key == 0ull) {
        key = (static_cast<unsigned long long>(static_cast<uint32_t>(module_idx)) << 32) |
              (static_cast<unsigned long long>(static_cast<uint32_t>(row_idx)) << 16);
    }
    return key;
}

int32_t declare_port(NodusHeadlessGraph* g, std::unordered_map<std::string, PortBinding>& ports,
                      const char* name, int32_t module_idx, int32_t row_idx, int32_t token_bytes, bool as_reader) {
    if (!g || !g->ctx || !g->table || !name || !*name || token_bytes <= 0) return 0;

    PortBinding pb;
    pb.module_idx = module_idx;
    pb.row_idx = row_idx;
    pb.key = resolve_row_key(g, module_idx, row_idx);

    int32_t edge_idx = -1;
    if (!gp_table_edge_index_for_key(g->table, pb.key, &edge_idx) || edge_idx < 0) {
        gp_table_add_edge(g->table, pb.key, pb.key);
        // gp_table_add_edge's own return value does not match the index
        // space gp_table_edge_set_tensor_spec/publish/consume validate
        // against (it can refer to a rope registration, not necessarily a
        // live ctx->edges slot yet) -- re-resolve the authoritative edge_idx
        // the same way the real TickRequest builder does
        // (canvas_abi_table_step.inl), rather than trusting that return
        // value directly.
        if (!gp_table_edge_index_for_key(g->table, pb.key, &edge_idx)) {
            edge_idx = -1;
        }
    }
    if (edge_idx < 0) return 0;
    pb.edge_idx = edge_idx;

    // The edge transport is fixed-shape and typed -- it must be configured
    // before any publish/consume succeeds (EdgeTensorFifo::push returns
    // false on an unconfigured FIFO). Treat the port as an opaque byte blob
    // of exactly `token_bytes` per sample: elem_size=1, a single dim of
    // token_bytes, Opaque layout, Unknown dtype. slots/top_k are small
    // finite defaults so a caller that never drains a port can observe
    // NODUS_EVT_OVERFLOW rather than growing unbounded.
    GP_TableEdgeTensorSpecTyped spec{};
    spec.dims[0] = token_bytes;
    spec.dim_count = 1;
    spec.slots = 4;
    spec.top_k = 1;
    spec.elem_size = 1;
    spec.type_id = -1;
    spec.layout = 2;  // nodus::tensors::TensorLayout::Opaque
    spec.dtype = 0;   // nodus::tensors::TensorDType::Unknown
    if (!gp_table_edge_set_tensor_spec(g->table, edge_idx, &spec)) return 0;

    if (as_reader) {
        gp_table_edge_subscribe_ex(g->table, edge_idx, kHostKey, /*start_at_head=*/1);
    }

    ports[name] = pb;
    return 1;
}

} // namespace

extern "C" NodusHeadlessGraph* nodus_headless_create(void) {
    auto* g = new NodusHeadlessGraph();
    g->ctx = gp_canvas_create(1, 1);
    if (!g->ctx) {
        delete g;
        return nullptr;
    }
    g->table = gp_canvas_get_container_table(g->ctx);
    return g;
}

extern "C" void nodus_headless_destroy(NodusHeadlessGraph* g) {
    if (!g) return;
    if (g->ctx) gp_canvas_destroy(g->ctx);
    delete g;
}

extern "C" int32_t nodus_headless_load_file(NodusHeadlessGraph* g, const char* path) {
    if (!g || !g->ctx || !path) return 0;
    int32_t rc = gp_canvas_load_from_file(g->ctx, path);
    g->table = gp_canvas_get_container_table(g->ctx);
    return rc;
}

extern "C" int32_t nodus_headless_save_file(NodusHeadlessGraph* g, const char* path) {
    if (!g || !g->ctx || !path) return 0;
    return gp_canvas_save_to_file(g->ctx, path);
}

extern "C" int32_t nodus_headless_add_module(NodusHeadlessGraph* g, const GP_CanvasModuleDesc* desc) {
    if (!g || !g->ctx || !desc) return -1;
    return gp_canvas_add_module(g->ctx, desc);
}

extern "C" int32_t nodus_headless_bind_builtin_tool(NodusHeadlessGraph* g, int32_t module_idx, int32_t tool_kind) {
    if (!g || !g->ctx) return -1;
    return gp_canvas_bind_builtin_tool(g->ctx, module_idx, tool_kind);
}

extern "C" int32_t nodus_headless_bind_plugin_tool(NodusHeadlessGraph* g, int32_t module_idx, const char* plugin_id) {
    if (!g || !g->ctx) return -1;
    return gp_canvas_bind_plugin_tool(g->ctx, module_idx, plugin_id);
}

extern "C" int32_t nodus_headless_add_edge(NodusHeadlessGraph* g, const GP_CanvasEdgeDesc* desc, int32_t type_id) {
    if (!g || !g->ctx || !desc) return -1;
    return gp_canvas_add_edge_with_type(g->ctx, desc, type_id);
}

extern "C" int32_t nodus_headless_declare_input_port(NodusHeadlessGraph* g, const char* name, int32_t module_idx, int32_t row_idx, int32_t token_bytes) {
    if (!g) return 0;
    // The host is the writer on an input port -- no reader subscription needed here;
    // the bound tool subscribes as a reader through its own normal tick machinery.
    return declare_port(g, g->input_ports, name, module_idx, row_idx, token_bytes, /*as_reader=*/false);
}

extern "C" int32_t nodus_headless_declare_output_port(NodusHeadlessGraph* g, const char* name, int32_t module_idx, int32_t row_idx, int32_t token_bytes) {
    if (!g) return 0;
    // The host is a reader on an output port, so it must subscribe up front.
    return declare_port(g, g->output_ports, name, module_idx, row_idx, token_bytes, /*as_reader=*/true);
}

extern "C" int32_t nodus_headless_push_token(NodusHeadlessGraph* g, const char* port_name, const void* bytes, size_t len) {
    if (!g || !g->table || !port_name) return 0;
    auto it = g->input_ports.find(port_name);
    if (it == g->input_ports.end()) return 0;
    int32_t dropped = 0;
    int32_t ok = gp_table_edge_publish(g->table, it->second.edge_idx, it->second.key, bytes, static_cast<int32_t>(len), &dropped);
    if (dropped) {
        push_event(g, NODUS_EVT_OVERFLOW, it->second.module_idx, it->second.edge_idx);
    }
    return ok;
}

extern "C" int32_t nodus_headless_pull_token(NodusHeadlessGraph* g, const char* port_name, void* out_bytes, size_t cap, size_t* out_written) {
    if (!g || !g->table || !port_name) return 0;
    auto it = g->output_ports.find(port_name);
    if (it == g->output_ports.end()) return 0;
    int32_t written = 0;
    int32_t ok = gp_table_edge_consume(g->table, it->second.edge_idx, kHostKey, out_bytes, static_cast<int32_t>(cap), &written);
    if (out_written) *out_written = static_cast<size_t>(written);
    return ok;
}

extern "C" int32_t nodus_headless_step(NodusHeadlessGraph* g, double dt) {
    if (!g || !g->ctx) return 0;
    int32_t rc = gp_canvas_step(g->ctx, static_cast<float>(dt));
    ++g->tick_id;
    return rc;
}

extern "C" int32_t nodus_headless_is_quiescent(const NodusHeadlessGraph* g) {
    if (!g || !g->table) return 0;
    for (const auto& kv : g->output_ports) {
        int32_t unread = 0;
        if (gp_table_edge_unread(g->table, kv.second.edge_idx, kHostKey, &unread) && unread > 0) {
            return 0;
        }
    }
    for (const auto& kv : g->input_ports) {
        int32_t unread = 0;
        // Input ports have no host-side subscription (see declare_input_port);
        // fall back to the host key, which reports 0/absent if never subscribed,
        // so this is a best-effort check and never falsely reports pending work.
        if (gp_table_edge_unread(g->table, kv.second.edge_idx, kHostKey, &unread) && unread > 0) {
            return 0;
        }
    }
    return 1;
}

extern "C" int32_t nodus_headless_run_to_quiescence(NodusHeadlessGraph* g, double dt, int32_t max_ticks) {
    if (!g || !g->ctx) return 0;
    for (int32_t i = 0; i < max_ticks; ++i) {
        nodus_headless_step(g, dt);
        if (nodus_headless_is_quiescent(g)) {
            push_event(g, NODUS_EVT_QUIESCENT, -1, -1);
            return 1;
        }
    }
    return 0;
}

extern "C" int32_t nodus_headless_poll_events(NodusHeadlessGraph* g, NodusHeadlessEvent* out, int32_t capacity) {
    if (!g || !out || capacity <= 0) return 0;
    std::lock_guard<std::mutex> lock(g->events_mu);
    int32_t n = 0;
    while (n < capacity && !g->events.empty()) {
        out[n++] = g->events.front();
        g->events.pop_front();
    }
    return n;
}
