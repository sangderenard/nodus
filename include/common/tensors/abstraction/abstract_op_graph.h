#pragma once

#include "common/tensors/abstraction/graph_ir.h"

#include <string>
#include <string_view>
#include <unordered_map>

namespace nodus::tensors {

// AbstractOpGraph: executes a GraphIR program against a supplied operator set.
// The operator set is responsible for translating arbitrary operators into graph
// operation sequences (GraphEdits) emitted via ctx.edits.
//
// This is the intended pivot point: the graph is generic, and domains plug in
// as operator tables.
struct AbstractOpGraphRunResult final {
  GraphEditBuilder edits;
  std::unordered_map<std::string, GraphIrValue> symbols;
};

bool run_graph_ir_on_abstract_graph(std::string_view src,
                                   const GraphIrOperatorSet& ops,
                                   AbstractOpGraphRunResult& out,
                                   std::string* out_error = nullptr);

} // namespace nodus::tensors
