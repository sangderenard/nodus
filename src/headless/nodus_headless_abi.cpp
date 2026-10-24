#include "nodus_headless_abi.h"

#include "canvas_abi.h"
#include "table_abi.h"
#include "value_types.h"
#include "plugin_manager.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Fixed subscriber/writer identity this runtime uses when it is itself the
// external party on a declared port's table edge (as opposed to another
// module in the graph). Any value distinct from module-derived keys works;
// module-derived keys are built from (module_idx<<32 | row_idx<<16), which
// never collides with this constant for valid module/row indices.
constexpr unsigned long long kHostKey = 0xE082D0A5E082D0A5ULL;
constexpr unsigned long long kEmptyComposedReaderKey = 0x8000000000000000ULL;
struct PortBinding {
    int32_t module_idx = -1;
    int32_t row_idx = -1;
    unsigned long long key = 0;
    int32_t edge_idx = -1;
    NodusHeadlessFifoConfig config{};
    int32_t start_at_head = 1;
    uint64_t declaration_order = 0;
};

struct EdgeBinding {
    GP_CanvasEdgeDesc desc{};
    int32_t table_edge_idx = -1;
    int32_t configured = 0;
};

} // namespace

struct NodusHeadlessGraph {
    GP_CanvasContext* ctx = nullptr;
    GP_TableContext* table = nullptr;

    std::unordered_map<std::string, PortBinding> input_ports;
    std::unordered_map<std::string, PortBinding> output_ports;
    std::vector<EdgeBinding> edges;
    gp_mem_backend_handle_t vulkan_backend = nullptr;

    std::mutex events_mu;
    std::deque<NodusHeadlessEvent> events;

    uint64_t tick_id = 0;
    uint64_t next_port_declaration_order = 0;
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

// ThreadManager subscribers for ordinary scalar rows use composed contact
// identity, not the table edge's stable row id.  The all-zero contact has a
// reserved nonzero fallback because the FIFO rejects reader key zero.
unsigned long long scalar_reader_key(int32_t module_idx, int32_t contact_idx) {
    unsigned long long key =
        (static_cast<unsigned long long>(static_cast<uint32_t>(module_idx)) << 32) |
        (static_cast<unsigned long long>(static_cast<uint32_t>(contact_idx)) << 16);
    return key == 0ull ? kEmptyComposedReaderKey : key;
}

int32_t declare_port(NodusHeadlessGraph* g, std::unordered_map<std::string, PortBinding>& ports,
                      const char* name, int32_t module_idx, int32_t contact_idx,
                      const NodusHeadlessFifoConfig& config, bool as_reader,
                      int32_t start_at_head) {
    if (!g || !g->ctx || !g->table || !name || !*name || config.token_bytes <= 0 ||
        config.slots <= 0 || config.top_k < 0 || config.top_k > config.slots) return 0;

    PortBinding pb;
    pb.module_idx = module_idx;
    pb.row_idx = contact_idx;
    pb.key = resolve_row_key(g, module_idx, contact_idx);
    pb.config = config;
    pb.start_at_head = start_at_head ? 1 : 0;
    pb.declaration_order = g->next_port_declaration_order++;

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
    spec.dims[0] = config.token_bytes;
    spec.dim_count = 1;
    spec.slots = config.slots;
    spec.top_k = config.top_k;
    spec.elem_size = 1;
    spec.type_id = config.type_id;
    spec.layout = config.layout;
    spec.dtype = config.dtype;
    if (!gp_table_edge_set_tensor_spec(g->table, edge_idx, &spec)) return 0;
    if (config.backend_type == GP_MEM_BACKEND_VULKAN) {
        if (!g->vulkan_backend) g->vulkan_backend = gp_mem_backend_create_vulkan(nullptr);
        if (!g->vulkan_backend ||
            !gp_table_edge_set_backend(g->table, edge_idx, g->vulkan_backend)) return 0;
    } else if (config.backend_type != GP_MEM_BACKEND_CPU) {
        return 0;
    }

    if (as_reader) {
        if (!gp_table_edge_subscribe_ex(g->table, edge_idx, kHostKey, pb.start_at_head)) return 0;
    } else {
        // The ThreadManager consumes ordinary Input rows under their canonical
        // (module, contact) identity.  Install that exact reader before the
        // host can publish so a start-at-head subscription in the first tick
        // observes the first token instead of skipping it.
        if (!gp_table_edge_subscribe_ex(g->table, edge_idx,
                                        scalar_reader_key(module_idx, contact_idx),
                                        /*start_at_head=*/1)) return 0;
    }

    ports[name] = pb;
    return 1;
}

NodusHeadlessFifoConfig default_fifo_config(int32_t token_bytes) {
    NodusHeadlessFifoConfig config{};
    config.token_bytes = token_bytes;
    config.slots = 4;
    config.top_k = 1;
    config.type_id = -1;
    config.layout = 2;  // TensorLayout::Opaque
    config.dtype = 0;   // TensorDType::Unknown
    config.backend_type = GP_MEM_BACKEND_CPU;
    return config;
}

int32_t configure_table_edge(NodusHeadlessGraph* g, int32_t edge_idx,
                             const NodusHeadlessFifoConfig& config) {
    if (!g || !g->table || edge_idx < 0 || config.token_bytes <= 0 || config.slots <= 0 ||
        config.top_k < 0 || config.top_k > config.slots) return 0;
    GP_TableEdgeTensorSpecTyped spec{};
    spec.dims[0] = config.token_bytes;
    spec.dim_count = 1;
    spec.slots = config.slots;
    spec.top_k = config.top_k;
    spec.elem_size = 1;
    spec.type_id = config.type_id;
    spec.layout = config.layout;
    spec.dtype = config.dtype;
    if (!gp_table_edge_set_tensor_spec(g->table, edge_idx, &spec)) return 0;
    if (config.backend_type == GP_MEM_BACKEND_CPU) return 1;
    if (config.backend_type != GP_MEM_BACKEND_VULKAN) return 0;
    if (!g->vulkan_backend) g->vulkan_backend = gp_mem_backend_create_vulkan(nullptr);
    return g->vulkan_backend &&
           gp_table_edge_set_backend(g->table, edge_idx, g->vulkan_backend);
}

int32_t read_table_edge_config(GP_TableContext* table, int32_t edge_idx,
                               NodusHeadlessFifoConfig* out_config) {
    if (!table || edge_idx < 0 || !out_config) return 0;
    GP_TableEdgeTensorSpecTyped spec{};
    if (!gp_table_edge_get_tensor_spec(table, edge_idx, &spec) ||
        spec.dim_count != 1 || spec.dims[0] <= 0 || spec.elem_size <= 0) return 0;
    const int64_t bytes = static_cast<int64_t>(spec.dims[0]) * spec.elem_size;
    if (bytes <= 0 || bytes > INT32_MAX) return 0;
    out_config->token_bytes = static_cast<int32_t>(bytes);
    out_config->slots = spec.slots;
    out_config->top_k = spec.top_k;
    out_config->type_id = spec.type_id;
    out_config->layout = spec.layout;
    out_config->dtype = spec.dtype;
    const int backend_type = gp_mem_backend_get_type(
        gp_table_edge_get_storage_backend(table, edge_idx));
    out_config->backend_type = backend_type == GP_MEM_BACKEND_VULKAN
        ? GP_MEM_BACKEND_VULKAN : GP_MEM_BACKEND_CPU;
    return 1;
}

int32_t attach_edge_binding(NodusHeadlessGraph* g, const GP_CanvasEdgeDesc& desc,
                            int32_t canvas_edge_idx) {
    if (!g || !g->table || canvas_edge_idx < 0) return -1;
    gp_table_apply_pending_ops(g->table);
    uint64_t ka = (static_cast<uint64_t>(static_cast<uint32_t>(desc.a_module)) << 32) |
                  (static_cast<uint64_t>(static_cast<uint32_t>(desc.a_contact_idx)) << 16);
    uint64_t kb = (static_cast<uint64_t>(static_cast<uint32_t>(desc.b_module)) << 32) |
                  (static_cast<uint64_t>(static_cast<uint32_t>(desc.b_contact_idx)) << 16);
    int32_t table_edge_idx = -1;
    if (!gp_table_edge_index_for_pair(g->table, ka, kb, &table_edge_idx) || table_edge_idx < 0) {
        return -1;
    }
    if (static_cast<size_t>(canvas_edge_idx) >= g->edges.size()) {
        g->edges.resize(static_cast<size_t>(canvas_edge_idx) + 1);
    }
    g->edges[static_cast<size_t>(canvas_edge_idx)] = EdgeBinding{desc, table_edge_idx, 0};
    return canvas_edge_idx;
}

} // namespace

extern "C" NodusHeadlessGraph* nodus_headless_create(void) {
    // The native canvas still owns process-global contact and table helpers.
    // Refuse coexistence instead of perturbing an already-running UI/bridge.
    if (gp_canvas_get_singleton() != nullptr) return nullptr;
    auto* g = new NodusHeadlessGraph();
    g->ctx = gp_canvas_create(1, 1);
    if (!g->ctx) {
        delete g;
        return nullptr;
    }
    g->table = gp_canvas_get_container_table(g->ctx);
    // gp_canvas_create is the interactive constructor: it restores the
    // default workspace and enables its autosave policy.  A headless graph is
    // an isolated programmatic document, so use the existing native clear and
    // autosave APIs immediately.  This avoids importing or overwriting the
    // user's live canvas while retaining the same Canvas/ThreadManager.
    if (!gp_canvas_set_autosave(g->ctx, nullptr, 0.0) ||
        !gp_canvas_clear_meta_and_overlays(g->ctx)) {
        gp_canvas_destroy(g->ctx);
        delete g;
        return nullptr;
    }
    g->table = gp_canvas_get_container_table(g->ctx);
    // GP_CanvasContextImpl::thread_mgr_paused defaults to true (canvas_abi_
    // context.inl); gp_canvas_step's real tick submission is entirely gated
    // on it being false (canvas_abi_table_step.inl's `if (c->thread_mgr &&
    // !c->thread_mgr_paused)`). The GUI path clears it once at canvas-head
    // init (frontend_shell.cpp) when the window opens; a headless graph has
    // no such init step, so without this call nodus_headless_step/run_to_
    // quiescence silently never submits a single tick, for any graph,
    // regardless of topology -- this is the ABI's own contract ("drive
    // ThreadManager's real scheduler to quiescence"), not an optional mode.
    if (!gp_canvas_set_thread_manager_paused(g->ctx, 0)) {
        gp_canvas_destroy(g->ctx);
        delete g;
        return nullptr;
    }
    return g;
}

extern "C" int32_t nodus_headless_abstract_tensor_type_id(void) {
    return static_cast<int32_t>(
        ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR));
}

extern "C" int32_t nodus_headless_deploy_python_row(
    const char* descriptor_path, const char* target,
    const char* python_executable, const char* turing_root,
    const char* extraction_contract, const char* product_dir,
    const char* nodus_root, const char* module_library_root,
    GP_PythonRowDeploymentReport* report,
    char* out_tool_id, int32_t out_tool_id_cap,
    char* out_retained_python, int32_t out_retained_python_cap,
    char* out_staged_wrapper, int32_t out_staged_wrapper_cap) {
    return gp_python_row_deploy_and_load(
        descriptor_path, target, python_executable, turing_root,
        extraction_contract, product_dir, nodus_root, module_library_root,
        report, out_tool_id, out_tool_id_cap,
        out_retained_python, out_retained_python_cap,
        out_staged_wrapper, out_staged_wrapper_cap);
}

extern "C" int32_t nodus_headless_unload_plugin(const char* tool_id) {
    return gp_plugin_unload(tool_id);
}

extern "C" int32_t nodus_headless_build_plugin_source(
    const char* source_path, const char* nodus_root,
    const char* module_library_root,
    char* out_tool_id, int32_t out_tool_id_cap) {
    if (!source_path || !nodus_root || !module_library_root ||
        !out_tool_id || out_tool_id_cap <= 0) return 0;
    std::filesystem::path bin =
        std::filesystem::path(module_library_root) / "bin";
    std::error_code error;
    std::filesystem::create_directories(bin, error);
    if (error) return 0;
    return gp_plugin_build_module_and_load(
        source_path, nodus_root, bin.string().c_str(), nullptr,
        out_tool_id, out_tool_id_cap);
}

extern "C" int32_t nodus_headless_compile_turing_row(
    const char* python_executable, const char* request_json,
    const char* turing_root, const char* product_dir,
    const char* nodus_root, const char* module_library_root,
    GP_TuringProductAdoptionReport* report,
    char* out_tool_id, int32_t out_tool_id_cap,
    char* out_staged_source, int32_t out_staged_source_cap) {
    return gp_turing_row_tool_compile_and_load(
        python_executable, request_json, turing_root, product_dir,
        nodus_root, module_library_root, report, out_tool_id,
        out_tool_id_cap, out_staged_source, out_staged_source_cap);
}

extern "C" uint64_t nodus_headless_tensor_create(
    int32_t dtype, int32_t rank, const uint64_t* shape) {
    return nodus_tensor_create(dtype, rank, shape);
}

extern "C" int64_t nodus_headless_tensor_write(
    uint64_t handle, uint64_t byte_offset, const void* src,
    uint64_t byte_count) {
    return nodus_tensor_write(handle, byte_offset, src, byte_count);
}

extern "C" int64_t nodus_headless_tensor_read(
    uint64_t handle, uint64_t byte_offset, void* dst,
    uint64_t byte_count) {
    return nodus_tensor_read(handle, byte_offset, dst, byte_count);
}

extern "C" int32_t nodus_headless_tensor_describe(
    uint64_t handle, NodusTensorDesc* out_desc) {
    return nodus_tensor_describe(handle, out_desc);
}

extern "C" void nodus_headless_tensor_destroy(uint64_t handle) {
    nodus_tensor_destroy(handle);
}

extern "C" void nodus_headless_destroy(NodusHeadlessGraph* g) {
    if (!g) return;
    if (g->ctx) gp_canvas_destroy(g->ctx);
    if (g->vulkan_backend) gp_mem_backend_release(g->vulkan_backend);
    delete g;
}

extern "C" int32_t nodus_headless_load_file(NodusHeadlessGraph* g, const char* path) {
    if (!g || !g->ctx || !path) return 0;
    int32_t rc = gp_canvas_load_from_file(g->ctx, path);
    if (!rc) return 0;
    g->table = gp_canvas_get_container_table(g->ctx);
    // Port bindings are process-local host views, not canvas persistence.
    // Callers reattach them after load to the restored stable identities.
    g->input_ports.clear();
    g->output_ports.clear();
    g->edges.clear();
    return rc;
}

extern "C" int32_t nodus_headless_save_file(NodusHeadlessGraph* g, const char* path) {
    if (!g || !g->ctx || !path) return 0;
    return gp_canvas_save_to_file(g->ctx, path);
}

extern "C" int32_t nodus_headless_export_module_to_root(
    NodusHeadlessGraph* g, int32_t module_idx, const char* output_root) {
    if (!g || !g->ctx || !output_root || !*output_root) return 0;
    return gp_canvas_export_module_to_root(g->ctx, module_idx, output_root);
}

extern "C" int32_t nodus_headless_add_module(NodusHeadlessGraph* g, const GP_CanvasModuleDesc* desc) {
    if (!g || !g->ctx || !desc) return -1;
    return gp_canvas_add_module(g->ctx, desc);
}

extern "C" int32_t nodus_headless_get_module_count(NodusHeadlessGraph* g) {
    if (!g || !g->ctx) return -1;
    return gp_canvas_get_module_count(g->ctx);
}

extern "C" int32_t nodus_headless_get_module_desc(
    NodusHeadlessGraph* g, int32_t module_idx, GP_CanvasModuleDesc* out_desc) {
    if (!g || !g->ctx) return 0;
    return gp_canvas_get_module_desc(g->ctx, module_idx, out_desc);
}

extern "C" int32_t nodus_headless_set_module_desc(
    NodusHeadlessGraph* g, int32_t module_idx, const GP_CanvasModuleDesc* desc) {
    if (!g || !g->ctx) return 0;
    return gp_canvas_set_module_desc(g->ctx, module_idx, desc);
}

extern "C" int32_t nodus_headless_bind_builtin_tool(NodusHeadlessGraph* g, int32_t module_idx, int32_t tool_kind) {
    if (!g || !g->ctx) return -1;
    return gp_canvas_bind_builtin_tool(g->ctx, module_idx, tool_kind);
}

extern "C" int32_t nodus_headless_bind_plugin_tool(NodusHeadlessGraph* g, int32_t module_idx, const char* plugin_id) {
    if (!g || !g->ctx) return -1;
    return gp_canvas_bind_plugin_tool(g->ctx, module_idx, plugin_id);
}

extern "C" int32_t nodus_headless_set_module_rows(
    NodusHeadlessGraph* g, int32_t module_idx,
    const GP_CanvasModuleRowDesc* rows, int32_t row_count) {
    if (!g || !g->ctx) return 0;
    return gp_canvas_set_module_rows(g->ctx, module_idx, rows, row_count);
}

extern "C" int32_t nodus_headless_get_module_row_count(
    NodusHeadlessGraph* g, int32_t module_idx) {
    if (!g || !g->ctx) return -1;
    return gp_canvas_get_module_row_count(g->ctx, module_idx);
}

extern "C" int32_t nodus_headless_get_module_row(
    NodusHeadlessGraph* g, int32_t module_idx, int32_t row_idx,
    GP_CanvasModuleRowInfo* out_info, char* plugin_id,
    int32_t plugin_id_capacity) {
    if (!g || !g->ctx) return 0;
    return gp_canvas_get_module_row(
        g->ctx, module_idx, row_idx, out_info, plugin_id, plugin_id_capacity);
}

extern "C" int32_t nodus_headless_set_module_stack_contract(
    NodusHeadlessGraph* g, int32_t module_idx, int32_t contract) {
    if (!g || !g->ctx) return 0;
    return gp_canvas_set_module_stack_contract(g->ctx, module_idx, contract);
}

extern "C" int32_t nodus_headless_get_module_stack_contract(
    NodusHeadlessGraph* g, int32_t module_idx, int32_t* out_contract) {
    if (!g || !g->ctx) return 0;
    return gp_canvas_get_module_stack_contract(g->ctx, module_idx, out_contract);
}

extern "C" int32_t nodus_headless_add_edge(NodusHeadlessGraph* g, const GP_CanvasEdgeDesc* desc, int32_t type_id) {
    if (!g || !g->ctx || !desc) return -1;
    int32_t canvas_edge_idx = gp_canvas_add_edge_with_type(g->ctx, desc, type_id);
    if (canvas_edge_idx < 0) return -1;
    return attach_edge_binding(g, *desc, canvas_edge_idx);
}

extern "C" int32_t nodus_headless_get_edge_count(NodusHeadlessGraph* g) {
    if (!g || !g->ctx) return -1;
    return gp_canvas_get_edge_count(g->ctx);
}

extern "C" int32_t nodus_headless_get_edge(
    NodusHeadlessGraph* g, int32_t edge_idx,
    GP_CanvasEdgeDesc* out_desc, int32_t* out_type_id) {
    if (!g || !g->ctx) return 0;
    return gp_canvas_get_edge(g->ctx, edge_idx, out_desc, out_type_id);
}

extern "C" int32_t nodus_headless_attach_edge(
    NodusHeadlessGraph* g, const GP_CanvasEdgeDesc* desc) {
    if (!g || !g->ctx || !desc) return -1;
    const int32_t canvas_edge_idx = gp_canvas_find_edge(g->ctx, desc);
    if (canvas_edge_idx < 0) return -1;
    return attach_edge_binding(g, *desc, canvas_edge_idx);
}


extern "C" int32_t nodus_headless_configure_edge_fifo(
    NodusHeadlessGraph* g, int32_t canvas_edge_idx,
    const NodusHeadlessFifoConfig* config) {
    if (!g || !g->table || !config || canvas_edge_idx < 0 ||
        static_cast<size_t>(canvas_edge_idx) >= g->edges.size()) return 0;
    auto& edge = g->edges[static_cast<size_t>(canvas_edge_idx)];
    if (!configure_table_edge(g, edge.table_edge_idx, *config)) return 0;
    // Tensor-spec configuration rebuilds FIFO storage, so subscribe only
    // after it succeeds.  Otherwise the producer can publish in topological
    // order before the consumer's first-tick start-at-head subscription and
    // the first inter-module token is skipped.
    const int32_t subscribed = gp_table_edge_subscribe_ex(
        g->table, edge.table_edge_idx,
        scalar_reader_key(edge.desc.b_module, edge.desc.b_contact_idx),
        /*start_at_head=*/1);
    if (subscribed) edge.configured = 1;
    return subscribed;
}

extern "C" int32_t nodus_headless_get_edge_fifo_config(
    NodusHeadlessGraph* g, int32_t canvas_edge_idx,
    NodusHeadlessFifoConfig* out_config) {
    if (!g || !g->table || !out_config || canvas_edge_idx < 0 ||
        static_cast<size_t>(canvas_edge_idx) >= g->edges.size()) return 0;
    const auto& edge = g->edges[static_cast<size_t>(canvas_edge_idx)];
    if (!edge.configured) return 0;
    return read_table_edge_config(g->table, edge.table_edge_idx, out_config);
}

extern "C" int32_t nodus_headless_get_edge_unread(
    NodusHeadlessGraph* g, int32_t canvas_edge_idx, int32_t* out_unread) {
    if (!g || !g->table || !out_unread || canvas_edge_idx < 0 ||
        static_cast<size_t>(canvas_edge_idx) >= g->edges.size()) return 0;
    const auto& edge = g->edges[static_cast<size_t>(canvas_edge_idx)];
    if (!edge.configured) return 0;
    return gp_table_edge_unread(
        g->table, edge.table_edge_idx,
        scalar_reader_key(edge.desc.b_module, edge.desc.b_contact_idx), out_unread);
}

extern "C" int32_t nodus_headless_set_module_uuid(
    NodusHeadlessGraph* g, int32_t module_idx, uint64_t module_uuid) {
    if (!g || !g->ctx || module_uuid == 0) return 0;
    return gp_canvas_set_module_uuid(g->ctx, module_idx, module_uuid);
}

extern "C" uint64_t nodus_headless_get_module_uuid(
    NodusHeadlessGraph* g, int32_t module_idx) {
    if (!g || !g->ctx) return 0;
    return gp_canvas_get_module_uuid(g->ctx, module_idx);
}

extern "C" int32_t nodus_headless_find_module_by_uuid(
    NodusHeadlessGraph* g, uint64_t module_uuid) {
    if (!g || !g->ctx) return -1;
    return gp_canvas_find_module_by_uuid(g->ctx, module_uuid);
}

extern "C" int32_t nodus_headless_set_scheduler_mode(
    NodusHeadlessGraph* g, int32_t mode) {
    if (!g || !g->ctx || (mode != 0 && mode != 1)) return 0;
    return gp_canvas_set_thread_manager_mode(g->ctx, mode);
}

extern "C" int32_t nodus_headless_get_scheduler_mode(
    NodusHeadlessGraph* g, int32_t* out_mode) {
    if (!g || !g->ctx) return 0;
    return gp_canvas_get_thread_manager_mode(g->ctx, out_mode);
}

extern "C" int32_t nodus_headless_declare_input_port(NodusHeadlessGraph* g, const char* name, int32_t module_idx, int32_t row_idx, int32_t token_bytes) {
    auto config = default_fifo_config(token_bytes);
    return nodus_headless_declare_input_port_ex(g, name, module_idx, row_idx, &config);
}

extern "C" int32_t nodus_headless_declare_output_port(NodusHeadlessGraph* g, const char* name, int32_t module_idx, int32_t row_idx, int32_t token_bytes) {
    auto config = default_fifo_config(token_bytes);
    return nodus_headless_declare_output_port_ex(g, name, module_idx, row_idx, &config,
                                                  /*start_at_head=*/1);
}

extern "C" int32_t nodus_headless_declare_input_port_ex(
    NodusHeadlessGraph* g, const char* name, int32_t module_idx,
    int32_t contact_idx, const NodusHeadlessFifoConfig* config) {
    if (!g || !config) return 0;
    return declare_port(g, g->input_ports, name, module_idx, contact_idx,
                        *config, /*as_reader=*/false, /*start_at_head=*/1);
}

extern "C" int32_t nodus_headless_declare_output_port_ex(
    NodusHeadlessGraph* g, const char* name, int32_t module_idx,
    int32_t contact_idx, const NodusHeadlessFifoConfig* config,
    int32_t start_at_head) {
    if (!g || !config) return 0;
    return declare_port(g, g->output_ports, name, module_idx, contact_idx,
                        *config, /*as_reader=*/true, start_at_head);
}

extern "C" int32_t nodus_headless_push_token(NodusHeadlessGraph* g, const char* port_name, const void* bytes, size_t len) {
    NodusHeadlessPublishResult result{};
    return nodus_headless_push_token_ex(g, port_name, bytes, len, &result) && result.accepted;
}

extern "C" int32_t nodus_headless_push_token_ex(
    NodusHeadlessGraph* g, const char* port_name, const void* bytes, size_t len,
    NodusHeadlessPublishResult* out_result) {
    if (out_result) *out_result = NodusHeadlessPublishResult{};
    if (!g || !g->table || !port_name) return 0;
    auto it = g->input_ports.find(port_name);
    if (it == g->input_ports.end()) return 0;
    if (len != static_cast<size_t>(it->second.config.token_bytes)) return 0;
    int32_t dropped = 0;
    int32_t ok = gp_table_edge_publish(g->table, it->second.edge_idx, it->second.key, bytes, static_cast<int32_t>(len), &dropped);
    if (out_result) {
        out_result->accepted = ok ? 1 : 0;
        out_result->dropped = dropped ? 1 : 0;
    }
    if (dropped) {
        push_event(g, NODUS_EVT_OVERFLOW, it->second.module_idx, it->second.edge_idx);
    }
    return 1;
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

extern "C" int32_t nodus_headless_get_port_fifo_config(
    NodusHeadlessGraph* g, const char* port_name, int32_t is_output,
    NodusHeadlessFifoConfig* out_config) {
    if (!g || !g->table || !port_name || !out_config) return 0;
    const auto& ports = is_output ? g->output_ports : g->input_ports;
    auto it = ports.find(port_name);
    if (it == ports.end()) return 0;
    return read_table_edge_config(g->table, it->second.edge_idx, out_config);
}

extern "C" int32_t nodus_headless_get_port_count(
    NodusHeadlessGraph* g, int32_t is_output) {
    if (!g) return -1;
    const auto& ports = is_output ? g->output_ports : g->input_ports;
    return static_cast<int32_t>(ports.size());
}

extern "C" int32_t nodus_headless_get_port_at(
    NodusHeadlessGraph* g, int32_t is_output, int32_t ordinal,
    char* out_name, int32_t out_name_capacity,
    int32_t* out_module_idx, int32_t* out_contact_idx,
    int32_t* out_start_at_head, NodusHeadlessFifoConfig* out_config) {
    if (!g || !g->table || ordinal < 0 || !out_name || out_name_capacity <= 0 ||
        !out_module_idx || !out_contact_idx || !out_start_at_head || !out_config) return 0;
    const auto& ports = is_output ? g->output_ports : g->input_ports;
    if (ordinal >= static_cast<int32_t>(ports.size())) return 0;
    std::vector<const std::pair<const std::string, PortBinding>*> ordered;
    ordered.reserve(ports.size());
    for (const auto& item : ports) ordered.push_back(&item);
    std::sort(ordered.begin(), ordered.end(), [](const auto* left, const auto* right) {
        return left->second.declaration_order < right->second.declaration_order;
    });
    const std::string& name = ordered[static_cast<size_t>(ordinal)]->first;
    const auto found = ports.find(name);
    if (found == ports.end()) return 0;
    const PortBinding& port = found->second;
    if (!read_table_edge_config(g->table, port.edge_idx, out_config)) return 0;
    const size_t n = std::min(name.size(), static_cast<size_t>(out_name_capacity - 1));
    std::memcpy(out_name, name.data(), n);
    out_name[n] = '\0';
    *out_module_idx = port.module_idx;
    *out_contact_idx = port.row_idx;
    *out_start_at_head = port.start_at_head;
    return 1;
}

extern "C" int32_t nodus_headless_get_port_unread(
    NodusHeadlessGraph* g, const char* port_name, int32_t is_output,
    int32_t* out_unread) {
    if (!g || !g->table || !port_name || !out_unread) return 0;
    const auto& ports = is_output ? g->output_ports : g->input_ports;
    auto it = ports.find(port_name);
    if (it == ports.end()) return 0;
    const unsigned long long reader = is_output
        ? kHostKey
        : scalar_reader_key(it->second.module_idx, it->second.row_idx);
    return gp_table_edge_unread(g->table, it->second.edge_idx, reader, out_unread);
}

extern "C" int32_t nodus_headless_get_stack_snapshot(
    NodusHeadlessGraph* g, int32_t module_idx, int32_t row_idx,
    float* out_values, int32_t capacity, int32_t* out_count) {
    if (!g || !g->ctx) return 0;
    return gp_canvas_get_module_stack_snapshot(
        g->ctx, module_idx, row_idx, out_values, capacity, out_count);
}

extern "C" int32_t nodus_headless_get_stack_snapshot_meta(
    NodusHeadlessGraph* g, int32_t module_idx, int32_t row_idx,
    int32_t* out_type_ids, int32_t* out_element_bytes,
    int32_t capacity, int32_t* out_count) {
    if (!g || !g->ctx) return 0;
    return gp_canvas_get_module_stack_snapshot_meta(
        g->ctx, module_idx, row_idx, out_type_ids, out_element_bytes,
        capacity, out_count);
}

extern "C" int32_t nodus_headless_get_stack_tail(
    NodusHeadlessGraph* g, int32_t module_idx,
    float* out_values, int32_t capacity, int32_t* out_count) {
    if (!g || !g->ctx) return 0;
    return gp_canvas_get_module_stack_tail(
        g->ctx, module_idx, out_values, capacity, out_count);
}

extern "C" int32_t nodus_headless_step(NodusHeadlessGraph* g, double dt) {
    if (!g || !g->ctx) return 0;
    int32_t rc = gp_canvas_step(g->ctx, static_cast<float>(dt));
    // gp_canvas_step intentionally remains asynchronous for the interactive
    // UI.  A headless step is an observation boundary: Python/native callers
    // must be able to pull values produced by this tick when it returns.
    if (rc) rc = gp_canvas_wait_for_thread_manager(g->ctx);
    if (rc) ++g->tick_id;
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
        // Input consumption identity depends on the bound row/tool contract;
        // the headless boundary does not have one generic reader key for all
        // plugin and builtin tools. Keep this explicitly best-effort rather
        // than misreporting scalar-row identity as universal.
        if (gp_table_edge_unread(g->table, kv.second.edge_idx, kHostKey, &unread) && unread > 0) {
            return 0;
        }
    }
    return 1;
}

extern "C" int32_t nodus_headless_run_to_quiescence(NodusHeadlessGraph* g, double dt, int32_t max_ticks) {
    if (!g || !g->ctx) return 0;
    for (int32_t i = 0; i < max_ticks; ++i) {
        if (!nodus_headless_step(g, dt)) return 0;
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
