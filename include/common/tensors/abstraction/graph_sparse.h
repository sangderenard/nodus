#pragma once

#include <cstdint>

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/coo_matrix.h"

namespace nodus::tensors {

// Sparse graph representation backed by COO tensors.
// Ports are the primary attachment points; edges connect port->port and
// also own two meta-ports (meta-in/meta-out) to describe their own interface.
enum class GraphObjectKind : uint32_t {
    Node = 1,
    Port = 2,
    Edge = 3,
};

inline constexpr uint32_t kGraphIdKindShift = 29;
inline constexpr uint32_t kGraphIdKindMask = 0x7u << kGraphIdKindShift;
inline constexpr uint32_t kGraphIdValueMask = (1u << kGraphIdKindShift) - 1u;

inline constexpr uint32_t graph_make_id(GraphObjectKind kind, uint32_t value) {
    return (static_cast<uint32_t>(kind) << kGraphIdKindShift) | (value & kGraphIdValueMask);
}

inline constexpr GraphObjectKind graph_id_kind(uint32_t id) {
    return static_cast<GraphObjectKind>((id & kGraphIdKindMask) >> kGraphIdKindShift);
}

inline constexpr bool graph_id_is_node(uint32_t id) {
    return graph_id_kind(id) == GraphObjectKind::Node;
}

inline constexpr bool graph_id_is_port(uint32_t id) {
    return graph_id_kind(id) == GraphObjectKind::Port;
}

inline constexpr bool graph_id_is_edge(uint32_t id) {
    return graph_id_kind(id) == GraphObjectKind::Edge;
}

struct PortFlagRule; // forward declare for SparseGraph validation helpers

struct SparseGraph {
    COOMatrix nodes;                 // [node_count, 1] ids or metadata indices
    COOMatrix node_ports;            // [port_count, 2] (node_id, port_id)
    COOMatrix edges;                 // [edge_count, 1] ids or metadata indices
    COOMatrix edge_ports;            // [edge_port_count, 2] (edge_id, port_id)

    COOMatrix connections;           // [connection_count, 2] (src_port_id, dst_port_id)
    COOMatrix edge_meta_ports;       // [edge_count, 2] (meta_in_port_id, meta_out_port_id)

    COOMatrix node_attrs;            // optional sparse attributes per node
    COOMatrix port_attrs;            // optional sparse attributes per port
    COOMatrix edge_attrs;            // optional sparse attributes per edge
    COOMatrix port_flags;            // optional sparse flags per port (indices store port_id, rank 1 or 2)

    AbstractTensor node_attr_ptrs;   // optional dense pointer table
    AbstractTensor port_attr_ptrs;
    AbstractTensor edge_attr_ptrs;

    bool validate(bool check_id_kinds = false,
                  bool check_port_flags = false,
                  bool allow_by_default = true,
                  const PortFlagRule* pass1_rules = nullptr,
                  uint32_t pass1_count = 0,
                  const PortFlagRule* pass2_rules = nullptr,
                  uint32_t pass2_count = 0) const;
};

enum class PortRuleOp : uint8_t {
    And = 0,
    Or,
    Xor,
    Nand,
    Nor,
    Xnor,
    MaskedEqual,
    MaskedNotEqual,
    MaskedBothZero,
};

struct PortFlagRule {
    uint32_t mask_a = 0;
    uint32_t mask_b = 0;
    PortRuleOp op = PortRuleOp::And;
};

bool port_rule_eval(const PortFlagRule& rule, uint32_t flags_a, uint32_t flags_b);

bool port_pair_allowed(uint32_t flags_a,
                        uint32_t flags_b,
                        bool allow_by_default,
                        const PortFlagRule* pass1_rules,
                        uint32_t pass1_count,
                        const PortFlagRule* pass2_rules,
                        uint32_t pass2_count);

bool validate_port_connections(const COOMatrix& connections,
                               const COOMatrix& port_flags,
                               bool allow_by_default,
                               const PortFlagRule* pass1_rules,
                               uint32_t pass1_count,
                               const PortFlagRule* pass2_rules,
                               uint32_t pass2_count);

// Default port flag bits for directionality and optionality.
inline constexpr uint32_t kPortFlagInput = 1u << 0;
inline constexpr uint32_t kPortFlagOutput = 1u << 1;
inline constexpr uint32_t kPortFlagOptionalInput = 1u << 2;
inline constexpr uint32_t kPortFlagOptionalOutput = 1u << 3;
inline constexpr uint32_t kPortFlagUserShift = 4;
inline constexpr uint32_t kPortFlagUserMask = 0xFu << kPortFlagUserShift;

inline constexpr uint32_t kPortTypeShift = 8;
inline constexpr uint32_t kPortTypeMask = 0xFFFFFFu << kPortTypeShift;

inline constexpr uint32_t port_type_bits(uint32_t flags) {
    return (flags & kPortTypeMask);
}

} // namespace nodus::tensors
