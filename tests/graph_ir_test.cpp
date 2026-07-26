#include "common/tensors/abstraction/graph_ir.h"
#include "common/tensors/abstraction/abstract_op_graph.h"
#include "common/tensors/abstraction/abstract_tensor_graph_ir.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo_ir.h"
#include "tool_registry.h"

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
  // 0) Turing ProcessGraph-shaped programs become AbstractTensor tool graphs.
  {
    const char* src = R"(
      x = tensor_node("input");
      xo = tensor_output(x, "value");
      y = tensor_node("input");
      yo = tensor_output(y, "value");
      add = tensor_node("add");
      ai = tensor_input(add, "lhs");
      bi = tensor_input(add, "rhs");
      ao = tensor_output(add, "result");
      connect(xo, ai);
      connect(yo, bi);
    )";

    AbstractOpGraphRunResult result;
    std::string error;
    if (!require_or_report(
            run_graph_ir_on_abstract_graph(
                src, make_abstract_tensor_graph_ir_ops(), result, &error),
            "build AbstractTensor tool graph")) {
      std::cerr << error << "\n";
      return 1;
    }

    size_t nodes = 0;
    size_t ports = 0;
    size_t connections = 0;
    bool canonical_add = false;
    for (const auto& edit : result.edits.edits()) {
      nodes += edit.kind == GraphEditKind::AddNode ? 1 : 0;
      ports += edit.kind == GraphEditKind::AddPort ? 1 : 0;
      connections += edit.kind == GraphEditKind::Connect ? 1 : 0;
      if (edit.kind == GraphEditKind::SetAttr &&
          edit.key == "tensor.canonical") {
        if (const auto* value = std::get_if<bool>(&edit.value)) {
          canonical_add |= *value;
        }
      }
    }
    if (!require_or_report(nodes == 3, "expected three tensor tool nodes")) return 1;
    if (!require_or_report(ports == 5, "expected five tensor ports")) return 1;
    if (!require_or_report(connections == 2, "expected two tensor connections")) return 1;
    if (!require_or_report(canonical_add, "add should be recognized as canonical")) return 1;

    bool emitted_registered_tool_id = false;
    for (const auto& edit : result.edits.edits()) {
      if (edit.kind == GraphEditKind::AddNode &&
          edit.key == "abstract_tensor.add")
        emitted_registered_tool_id = true;
    }
    if (!require_or_report(
            emitted_registered_tool_id,
            "canonical graph node should name its real registered tool"))
      return 1;
  }

  // 0b) Canonical graph nodes are actual ToolIR stack tools, not descriptive
  // attributes on otherwise generic nodes.
  {
    ToolRegistry registry;
    const size_t registered = register_abstract_tensor_tool_ir(registry);
    if (!require_or_report(registered == 28, "register 28 elementwise tensor tools"))
      return 1;
    auto add = registry.create("abstract_tensor.add");
    if (!require_or_report(add != nullptr, "create abstract_tensor.add ToolIR"))
      return 1;

    auto& backend = in_memory_backend_singleton();
    const TensorDesc desc{TensorDType::F64, {{3}}};
    auto* left = new AbstractTensor(desc, &backend);
    auto* right = new AbstractTensor(desc, &backend);
    void* data = nullptr;
    size_t bytes = 0;
    backend.map(left->handle(), &data, &bytes);
    auto* values = static_cast<double*>(data);
    values[0] = 1.0; values[1] = 2.0; values[2] = 3.0;
    backend.unmap(left->handle());
    backend.map(right->handle(), &data, &bytes);
    values = static_cast<double*>(data);
    values[0] = 10.0; values[1] = 20.0; values[2] = 30.0;
    backend.unmap(right->handle());

    RawStackFrame raw{};
    if (!require_or_report(raw_stack_init_frame(raw, 1024), "initialize tool stack"))
      return 1;
    const ValueTypeId pointer_type =
        ValueTypeRegistry::global().builtin(VT_VOID_PTR);
    void* left_pointer = left;
    void* right_pointer = right;
    raw_stack_push_typed(raw, &left_pointer, pointer_type);
    raw_stack_push_typed(raw, &right_pointer, pointer_type);
    ToolStackContext context{};
    context.stack.raw = &raw;
    add->execute_stack(context);

    void* output_pointer = nullptr;
    const bool popped =
        raw_stack_pop_typed(raw, &output_pointer, pointer_type) != 0;
    auto* output = static_cast<AbstractTensor*>(output_pointer);
    bool correct = popped && output && output->valid() &&
                   backend.map(output->handle(), &data, &bytes);
    if (correct) {
      values = static_cast<double*>(data);
      correct = values[0] == 11.0 && values[1] == 22.0 && values[2] == 33.0;
      backend.unmap(output->handle());
    }
    delete output;
    delete right;
    delete left;
    raw_stack_free_mask(raw);
    if (!require_or_report(correct, "execute tensor add through ToolIR stack"))
      return 1;
  }

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

  // 3) Circle/arc primitives + logical constraints are preserved as RelProgram assertions.
  {
    const char* src = R"(
      c = pt(0, 0);
      through = pt(10, 0);
      C = circle(c, through);
      p = free();
      point_on_circle(p, C);
    )";

    RelProgram p;
    std::string err;
    if (!require_or_report(relgeo_program_from_ir(src, p, &err), "build relgeo program from IR (circle constraints)")) {
      std::cerr << err << "\n";
      return 1;
    }
    if (!require_or_report(!p.circles().empty(), "should build at least one circle")) return 1;

    bool found = false;
    for (const auto& a : p.assertions()) {
      if (std::holds_alternative<RelAssertPointOnCircle>(a.assertion)) {
        found = true;
        break;
      }
    }
    if (!require_or_report(found, "should record RelAssertPointOnCircle")) return 1;
  }

  {
    // Non-pure ops: fixed radius and tangent + arc angle are recorded as assertions.
    const char* src = R"(
      c = pt(0, 0);
      t = free();
      C = circle(c, t);
      fixed_radius(C, 5.0);

      a = free();
      b = free();
      L = line(a, b);
      tangent(L, C);

      s = pt(5, 0);
      e = free();
      point_on_circle(s, C);
      point_on_circle(e, C);
      A = arcse(C, s, e, "ccw");
      arc_angle(A, 1.57079632679);
    )";

    RelProgram p;
    std::string err;
    if (!require_or_report(relgeo_program_from_ir(src, p, &err), "build relgeo program from IR (full constraints)")) {
      std::cerr << err << "\n";
      return 1;
    }

    bool has_fixed_radius = false;
    bool has_tangent = false;
    bool has_arc_angle = false;
    bool has_poc = false;
    for (const auto& a : p.assertions()) {
      has_fixed_radius |= std::holds_alternative<RelAssertFixedRadius>(a.assertion);
      has_tangent |= std::holds_alternative<RelAssertTangentLineCircle>(a.assertion);
      has_arc_angle |= std::holds_alternative<RelAssertArcAngle>(a.assertion);
      has_poc |= std::holds_alternative<RelAssertPointOnCircle>(a.assertion);
    }

    if (!require_or_report(has_fixed_radius, "should record RelAssertFixedRadius")) return 1;
    if (!require_or_report(has_tangent, "should record RelAssertTangentLineCircle")) return 1;
    if (!require_or_report(has_arc_angle, "should record RelAssertArcAngle")) return 1;
    if (!require_or_report(has_poc, "should record RelAssertPointOnCircle")) return 1;
  }

  std::cout << "[GRAPH-IR] OK\n";
  return 0;
}
