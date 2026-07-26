#pragma once

#include "common/tensors/abstraction/graph_ir.h"

class ToolRegistry;

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

// Register one real ToolIR stack tool for every canonical in-memory
// elementwise operation. Tool ids are ``abstract_tensor.<canonical-name>`` and
// are the same ids emitted by ``tensor_node``.
size_t register_abstract_tensor_tool_ir(ToolRegistry& registry);

} // namespace nodus::tensors
