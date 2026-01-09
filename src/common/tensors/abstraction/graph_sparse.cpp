#include "common/tensors/abstraction/graph_sparse.h"

#include "common/tensors/abstraction/in_memory_backend.h"

#include <unordered_map>

namespace nodus::tensors {

namespace {
bool validate_id_table(const COOMatrix& table, GraphObjectKind kind) {
    if (!table.valid()) return false;
    const TensorDesc& desc = table.values.desc();
    if (desc.dtype != TensorDType::U32) return false;
    const uint64_t count = desc.shape.element_count();
    if (count == 0) return true;

    auto* backend = dynamic_cast<InMemoryBackend*>(table.values.backend());
    if (!backend) return false;

    void* data = nullptr;
    size_t bytes = 0;
    if (!backend->map(table.values.handle(), &data, &bytes)) return false;
    if (bytes < count * sizeof(uint32_t)) {
        backend->unmap(table.values.handle());
        return false;
    }

    const auto* ids = static_cast<const uint32_t*>(data);
    for (uint64_t i = 0; i < count; ++i) {
        if (graph_id_kind(ids[i]) != kind) {
            backend->unmap(table.values.handle());
            return false;
        }
    }
    backend->unmap(table.values.handle());
    return true;
}
} // namespace

bool SparseGraph::validate(bool check_id_kinds,
                           bool check_port_flags,
                           bool allow_by_default,
                           const PortFlagRule* pass1_rules,
                           uint32_t pass1_count,
                           const PortFlagRule* pass2_rules,
                           uint32_t pass2_count) const {
    if (!nodes.valid() || !edges.valid()) return false;
    if (!node_ports.valid() || !edge_ports.valid()) return false;
    if (!connections.valid() || !edge_meta_ports.valid()) return false;

    if (!nodes.validate() || !edges.validate()) return false;
    if (!node_ports.validate() || !edge_ports.validate()) return false;
    if (!connections.validate() || !edge_meta_ports.validate()) return false;

    const uint32_t node_rank = nodes.rank();
    const uint32_t edge_rank = edges.rank();
    if (node_rank == 0 || edge_rank == 0) return false;

    if (check_id_kinds) {
        if (!validate_id_table(nodes, GraphObjectKind::Node)) return false;
        if (!validate_id_table(edges, GraphObjectKind::Edge)) return false;
    }

    if (check_port_flags && port_flags.valid()) {
        if (!validate_port_connections(connections,
                                        port_flags,
                                        allow_by_default,
                                        pass1_rules,
                                        pass1_count,
                                        pass2_rules,
                                        pass2_count)) {
            return false;
        }
    }

    return true;
}

bool port_rule_eval(const PortFlagRule& rule, uint32_t flags_a, uint32_t flags_b) {
    switch (rule.op) {
        case PortRuleOp::MaskedEqual:
            return (flags_a & rule.mask_a) == (flags_b & rule.mask_b);
        case PortRuleOp::MaskedNotEqual:
            return (flags_a & rule.mask_a) != (flags_b & rule.mask_b);
        case PortRuleOp::MaskedBothZero:
            return ((flags_a & rule.mask_a) == 0u) && ((flags_b & rule.mask_b) == 0u);
        default: {
            const bool a = (flags_a & rule.mask_a) != 0u;
            const bool b = (flags_b & rule.mask_b) != 0u;
            switch (rule.op) {
                case PortRuleOp::And:  return a && b;
                case PortRuleOp::Or:   return a || b;
                case PortRuleOp::Xor:  return a != b;
                case PortRuleOp::Nand: return !(a && b);
                case PortRuleOp::Nor:  return !(a || b);
                case PortRuleOp::Xnor: return a == b;
                default:               return false;
            }
        }
    }
}

bool port_pair_allowed(uint32_t flags_a,
                        uint32_t flags_b,
                        bool allow_by_default,
                        const PortFlagRule* pass1_rules,
                        uint32_t pass1_count,
                        const PortFlagRule* pass2_rules,
                        uint32_t pass2_count) {
    bool allowed = allow_by_default;
    if (pass1_rules && pass1_count > 0) {
        for (uint32_t i = 0; i < pass1_count; ++i) {
            if (port_rule_eval(pass1_rules[i], flags_a, flags_b)) {
                allowed = !allowed;
            }
        }
    }
    if (pass2_rules && pass2_count > 0) {
        for (uint32_t i = 0; i < pass2_count; ++i) {
            if (port_rule_eval(pass2_rules[i], flags_a, flags_b)) {
                allowed = !allowed;
            }
        }
    }
    return allowed;
}

bool validate_port_connections(const COOMatrix& connections,
                               const COOMatrix& port_flags,
                               bool allow_by_default,
                               const PortFlagRule* pass1_rules,
                               uint32_t pass1_count,
                               const PortFlagRule* pass2_rules,
                               uint32_t pass2_count) {
    if (!connections.valid() || !port_flags.valid()) return false;
    const TensorDesc& conn_idx_desc = connections.indices.desc();
    const TensorDesc& flags_desc = port_flags.values.desc();
    const TensorDesc& flags_idx_desc = port_flags.indices.desc();
    if (flags_desc.dtype != TensorDType::U32) return false;
    if (conn_idx_desc.dtype != TensorDType::U32) return false;
    if (flags_idx_desc.dtype != TensorDType::U32) return false;

    auto* backend = dynamic_cast<InMemoryBackend*>(connections.indices.backend());
    auto* flags_backend = dynamic_cast<InMemoryBackend*>(port_flags.values.backend());
    if (!backend || !flags_backend) return false;

    void* conn_idx_data = nullptr;
    size_t conn_idx_bytes = 0;
    if (!backend->map(connections.indices.handle(), &conn_idx_data, &conn_idx_bytes)) return false;

    void* flags_data = nullptr;
    size_t flags_bytes = 0;
    if (!flags_backend->map(port_flags.values.handle(), &flags_data, &flags_bytes)) {
        backend->unmap(connections.values.handle());
        backend->unmap(connections.indices.handle());
        return false;
    }
    void* flags_idx_data = nullptr;
    size_t flags_idx_bytes = 0;
    if (!flags_backend->map(port_flags.indices.handle(), &flags_idx_data, &flags_idx_bytes)) {
        backend->unmap(connections.values.handle());
        backend->unmap(connections.indices.handle());
        flags_backend->unmap(port_flags.values.handle());
        return false;
    }

    const auto* conn_idx = static_cast<const uint32_t*>(conn_idx_data);
    const auto* flag_vals = static_cast<const uint32_t*>(flags_data);
    const auto* flag_idx = static_cast<const uint32_t*>(flags_idx_data);

    const uint64_t conn_count = conn_idx_desc.shape.element_count() / 2;
    const uint64_t flag_count = flags_desc.shape.element_count();
    const uint32_t flag_rank = static_cast<uint32_t>(flags_idx_desc.shape.dims.size());

    std::unordered_map<uint32_t, uint32_t> flag_map;
    flag_map.reserve(static_cast<size_t>(flag_count));
    for (uint64_t i = 0; i < flag_count; ++i) {
        const uint32_t port_id = (flag_rank == 1) ? flag_idx[i] : flag_idx[i * 2 + 0];
        flag_map[port_id] = flag_vals[i];
    }

    for (uint64_t i = 0; i < conn_count; ++i) {
        const uint32_t src_port = conn_idx[i * 2 + 0];
        const uint32_t dst_port = conn_idx[i * 2 + 1];
        const uint32_t src_flags = flag_map.count(src_port) ? flag_map[src_port] : 0u;
        const uint32_t dst_flags = flag_map.count(dst_port) ? flag_map[dst_port] : 0u;
        if (!port_pair_allowed(src_flags,
                               dst_flags,
                               allow_by_default,
                               pass1_rules,
                               pass1_count,
                               pass2_rules,
                               pass2_count)) {
            backend->unmap(connections.indices.handle());
            flags_backend->unmap(port_flags.values.handle());
            flags_backend->unmap(port_flags.indices.handle());
            return false;
        }
    }

    backend->unmap(connections.indices.handle());
    flags_backend->unmap(port_flags.values.handle());
    flags_backend->unmap(port_flags.indices.handle());
    return true;
}

} // namespace nodus::tensors
