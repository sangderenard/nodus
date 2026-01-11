#include "common/tensors/abstraction/graph_ir.h"
#include "common/tensors/abstraction/graph_sparse_builder.h"
#include "common/tensors/abstraction/in_memory_backend.h"

#include <iostream>
#include <string>

using namespace nodus::tensors;

static bool require_or_report(bool condition, const char* what) {
  if (condition) return true;
  std::cerr << "[GRAPH-SPARSE-BUILDER] FAILED: " << what << "\n";
  return false;
}

int main() {
  const char* src = R"(
    n = node("Point");
    a = port(n, 1, 0);
    b = port(n, 2, 0);
    connect(a, b);
    attr(n, "w", 1.0);
  )";

  GraphIrProgram prog;
  GraphIrError err;
  if (!require_or_report(graph_ir_parse(src, prog, &err), "parse")) {
    std::cerr << "  at " << err.at.line << ":" << err.at.col << ": " << err.message << "\n";
    return 1;
  }

  GraphEditBuilder edits;
  GraphIrContext ctx;
  ctx.edits = &edits;

  const GraphIrOperatorSet ops = make_core_graph_ir_ops();
  if (!require_or_report(graph_ir_eval(prog, ops, ctx, &err), "eval")) {
    std::cerr << "  at " << err.at.line << ":" << err.at.col << ": " << err.message << "\n";
    return 1;
  }

  InMemoryBackend backend;
  SparseGraphBuildError berr;
  SparseGraph g = sparse_graph_from_edit_log(edits.edits(), &backend, &berr);
  if (!berr.message.empty()) {
    std::cerr << berr.message << "\n";
    return 1;
  }

  if (!require_or_report(g.validate(/*check_id_kinds=*/true), "validate")) return 1;

  std::cout << "[GRAPH-SPARSE-BUILDER] OK\n";
  return 0;
}
