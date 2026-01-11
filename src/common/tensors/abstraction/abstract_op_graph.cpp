#include "common/tensors/abstraction/abstract_op_graph.h"

#include <sstream>

namespace nodus::tensors {

bool run_graph_ir_on_abstract_graph(std::string_view src,
                                   const GraphIrOperatorSet& ops,
                                   AbstractOpGraphRunResult& out,
                                   std::string* out_error) {
  GraphIrProgram program;
  GraphIrError perr;
  if (!graph_ir_parse(src, program, &perr)) {
    if (out_error) {
      std::ostringstream oss;
      oss << "parse error at " << perr.at.line << ":" << perr.at.col << ": " << perr.message;
      *out_error = oss.str();
    }
    return false;
  }

  GraphIrContext ctx;
  ctx.edits = &out.edits;
  ctx.symbols = {};

  GraphIrError eerr;
  if (!graph_ir_eval(program, ops, ctx, &eerr)) {
    if (out_error) {
      std::ostringstream oss;
      oss << "eval error at " << eerr.at.line << ":" << eerr.at.col << ": " << eerr.message;
      *out_error = oss.str();
    }
    return false;
  }

  out.symbols = std::move(ctx.symbols);
  return true;
}

} // namespace nodus::tensors
