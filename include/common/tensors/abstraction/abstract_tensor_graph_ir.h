#pragma once

#include "common/tensors/abstraction/graph_ir.h"

namespace nodus::tensors {

// Operations used by Turing's direct ProcessGraph -> Nodus tool-graph export:
//
//   tensor_node(op) -> node id
//   tensor_input(node, role) -> input port id
//   tensor_output(node, role) -> output port id
//
// The resulting edit log is an ordinary Nodus sparse graph. Canonical
// operations are marked from canonical_ops.h; structural ProcessGraph nodes
// remain explicit non-canonical tools rather than being rejected or hidden.
GraphIrOperatorSet make_abstract_tensor_graph_ir_ops();

} // namespace nodus::tensors

