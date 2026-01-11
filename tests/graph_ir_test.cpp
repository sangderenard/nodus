#include "common/tensors/abstraction/graph_ir.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo_ir.h"

#include "common/tensors/abstraction/kpath/kpath_relgeo.h"

#include <iostream>
#include <string>

using namespace nodus::tensors;
using namespace nodus::tensors::kpath;

static bool require_or_report(bool condition, const char* what) {
  if (condition) return true;
  std::cerr << "[GRAPH-IR] FAILED: " << what << "\n";
  return false;
}

int main() {
  // 1) Core graph IR parses and emits edits.
  {
    const char* src = R"(
      n = node("Point");
      p = port(n, 1, 0);
      attr(n, "name", "A");
    )";

    GraphIrProgram prog;
    GraphIrError err;
    if (!require_or_report(graph_ir_parse(src, prog, &err), "parse core program")) {
      std::cerr << "  at " << err.at.line << ":" << err.at.col << ": " << err.message << "\n";
      return 1;
    }

    GraphEditBuilder edits;
    GraphIrContext ctx;
    ctx.edits = &edits;

    const GraphIrOperatorSet ops = make_core_graph_ir_ops();
    if (!require_or_report(graph_ir_eval(prog, ops, ctx, &err), "eval core program")) {
      std::cerr << "  at " << err.at.line << ":" << err.at.col << ": " << err.message << "\n";
      return 1;
    }

    if (!require_or_report(!edits.edits().empty(), "edits should not be empty")) return 1;
  }

  // 2) RelGeo IR builds a RelProgram and compiles an outline.
  {
    const char* src = R"(
      a = pt(0, 0);
      b = pt(100, 0);
      c = pt(0, 100);
      contour(a, b, c);
    )";

    // Evaluate with graph edit emission enabled.
    GraphIrProgram prog;
    GraphIrError perr;
    if (!require_or_report(graph_ir_parse(src, prog, &perr), "parse relgeo program")) {
      std::cerr << "  at " << perr.at.line << ":" << perr.at.col << ": " << perr.message << "\n";
      return 1;
    }

    RelProgram p;
    {
      GraphEditBuilder edits;
      GraphIrContext ctx;
      ctx.edits = &edits;

      // RelGeo operator table stores its state behind ctx.user, so we evaluate
      // then extract the RelProgram by re-parsing through the convenience API.
      // This keeps the test focused on the IR plumbing + edit emission.
      const GraphIrOperatorSet ops = make_relgeo_ir_ops();
      GraphIrError eerr;
      if (!require_or_report(graph_ir_eval(prog, ops, ctx, &eerr), "eval relgeo program")) {
        std::cerr << "  at " << eerr.at.line << ":" << eerr.at.col << ": " << eerr.message << "\n";
        return 1;
      }
      if (!require_or_report(!edits.edits().empty(), "relgeo should emit edits")) return 1;
    }

    std::string err;
    if (!require_or_report(relgeo_program_from_ir(src, p, &err), "build relgeo program from IR")) {
      std::cerr << err << "\n";
      return 1;
    }

    RelGlyph g;
    g.glyph_id = 1;
    g.program = std::move(p);

    GlyphOutline outline;
    if (!require_or_report(compile_relglyph_outline(g, outline, 0.0f, 0.0f, 1.0f, &err), "compile relglyph outline")) {
      std::cerr << err << "\n";
      return 1;
    }

    if (!require_or_report(outline.segments.size() >= 4, "expected triangle-ish outline segments")) return 1;
  }

  std::cout << "[GRAPH-IR] OK\n";
  return 0;
}
