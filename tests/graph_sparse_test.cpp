#include "common/tensors/abstraction/graph_sparse.h"
#include "common/tensors/abstraction/in_memory_backend.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace nodus::tensors;

static bool require_or_report(bool condition, const char* what) {
    if (condition) return true;
    std::cerr << "graph_sparse_test: failed: " << what << "\n";
    return false;
}

static COOMatrix make_table(uint32_t rows, uint32_t cols, uint32_t nnz, TensorBackend* backend) {
    TensorShape shape{};
    shape.dims = {rows, cols};
    return COOMatrix::create(shape, nnz, TensorDType::U32, backend, TensorDType::U32, CooIndexLayout::RowMajor);
}

static COOMatrix make_flags_table(uint32_t port_count, uint32_t nnz, TensorBackend* backend) {
    TensorShape shape{};
    shape.dims = {port_count};
    return COOMatrix::create(shape, nnz, TensorDType::U32, backend, TensorDType::U32, CooIndexLayout::RowMajor);
}

static bool write_u32(InMemoryBackend& backend, const AbstractTensor& t, const std::vector<uint32_t>& data) {
    void* raw = nullptr;
    size_t bytes = 0;
    if (!require_or_report(backend.map(t.handle(), &raw, &bytes), "backend.map")) return false;
    if (!require_or_report(bytes >= data.size() * sizeof(uint32_t), "mapped bytes too small")) return false;
    std::memcpy(raw, data.data(), data.size() * sizeof(uint32_t));
    backend.unmap(t.handle());
    return true;
}

static SparseGraph build_graph(InMemoryBackend& backend,
                               const std::vector<uint32_t>& port_flags,
                               const std::vector<uint32_t>& connection_pairs) {
    SparseGraph graph{};
    const uint32_t port_count = static_cast<uint32_t>(port_flags.size());
    graph.nodes = make_table(1, 1, 1, &backend);
    graph.node_ports = make_table(1, 2, 1, &backend);
    graph.edges = make_table(1, 1, 1, &backend);
    graph.edge_ports = make_table(1, 2, 1, &backend);
    graph.connections = make_table(port_count, port_count, static_cast<uint32_t>(connection_pairs.size() / 2), &backend);
    graph.edge_meta_ports = make_table(1, 2, 1, &backend);
    graph.port_flags = make_flags_table(port_count, port_count, &backend);

    const uint32_t node_id = graph_make_id(GraphObjectKind::Node, 1);
    const uint32_t edge_id = graph_make_id(GraphObjectKind::Edge, 3);
    if (!write_u32(backend, graph.nodes.values, {node_id})) return {};
    if (!write_u32(backend, graph.edges.values, {edge_id})) return {};

    std::vector<uint32_t> indices(port_count);
    for (uint32_t i = 0; i < port_count; ++i) indices[i] = i;
    if (!write_u32(backend, graph.port_flags.indices, indices)) return {};
    if (!write_u32(backend, graph.port_flags.values, port_flags)) return {};

    if (!write_u32(backend, graph.connections.indices, connection_pairs)) return {};
    if (!write_u32(backend, graph.connections.values, std::vector<uint32_t>(connection_pairs.size() / 2, 0u))) return {};

    return graph;
}

int main() {
    register_in_memory_backend(true);
    InMemoryBackend& backend = in_memory_backend_singleton();

    const uint32_t node_id = graph_make_id(GraphObjectKind::Node, 1);
    const uint32_t port_id = graph_make_id(GraphObjectKind::Port, 2);
    const uint32_t edge_id = graph_make_id(GraphObjectKind::Edge, 3);

    if (!require_or_report(graph_id_is_node(node_id), "graph_id_is_node")) return 1;
    if (!require_or_report(graph_id_is_port(port_id), "graph_id_is_port")) return 1;
    if (!require_or_report(graph_id_is_edge(edge_id), "graph_id_is_edge")) return 1;
    PortFlagRule pass1[] = {
        {kPortFlagOutput | kPortFlagOptionalOutput, kPortFlagInput | kPortFlagOptionalInput, PortRuleOp::And},
        {kPortFlagInput | kPortFlagOptionalInput, kPortFlagOutput | kPortFlagOptionalOutput, PortRuleOp::And},
        {0xFFFFFFFFu, 0xFFFFFFFFu, PortRuleOp::MaskedBothZero},
    };
    PortFlagRule pass2[] = {
        {kPortTypeMask, kPortTypeMask, PortRuleOp::MaskedEqual},
    };

    const uint32_t type_a = 1u << kPortTypeShift;
    const uint32_t type_b = 2u << kPortTypeShift;
    const uint32_t type_c = 3u << kPortTypeShift;
    const uint32_t f_out = kPortFlagOutput;
    const uint32_t f_in = kPortFlagInput;
    const uint32_t f_opt_out = kPortFlagOptionalOutput;
    const uint32_t f_opt_in = kPortFlagOptionalInput;
    const std::vector<uint32_t> flags = {
        f_out | type_a,                  // 0: output (A)
        f_in | type_a,                   // 1: input (A)
        f_opt_out | type_b,              // 2: optional output (B)
        f_opt_in | type_b,               // 3: optional input (B)
        0u,                              // 4: logistic
        0u,                              // 5: logistic
        f_opt_in | f_opt_out | type_c,   // 6: meta optional in/out (C)
    };

    SparseGraph valid_graph = build_graph(backend, flags, {
        0, 3,  // output (A) -> optional input (B)
        2, 1,  // optional output (B) -> input (A)
        4, 5,  // logistic -> logistic (both zero)
        6, 3   // meta optional in/out (C) -> optional input (B)
    });
    if (!require_or_report(valid_graph.validate(true, true, false, pass1, 3, pass2, 1), "valid_graph.validate")) return 1;

    SparseGraph invalid_graph = build_graph(backend, flags, {
        0, 1,  // output (A) -> input (A) (type match => disallow in pass2)
        1, 3,  // input (A) -> optional input (B) (no output on src)
        4, 6   // logistic -> meta (no IO bits to satisfy pass1)
    });
    if (!require_or_report(!invalid_graph.validate(true, true, false, pass1, 3, pass2, 1), "invalid_graph.validate")) return 1;

    std::cout << "graph_sparse_test: ok\n";
    return 0;
}
