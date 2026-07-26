#include "common/tensors/abstraction/abstract_tensor_graph_ir.h"

#include "canonical_ops.h"
#include "common/tensors/abstraction/graph_sparse.h"

#include <optional>

namespace nodus::tensors {

namespace {

std::optional<uint32_t> as_u32(const GraphIrValue& value) {
  if (const auto* p = std::get_if<uint32_t>(&value)) return *p;
  if (const auto* p = std::get_if<double>(&value)) {
    if (*p >= 0.0 && *p <= 4294967295.0) return static_cast<uint32_t>(*p);
  }
  return std::nullopt;
}

std::optional<std::string> as_string(const GraphIrValue& value) {
  if (const auto* p = std::get_if<std::string>(&value)) return *p;
  return std::nullopt;
}

bool require_edits(GraphIrContext& ctx, std::string& error) {
  if (ctx.edits) return true;
  error = "AbstractTensor GraphIR operation requires ctx.edits";
  return false;
}

} // namespace

GraphIrOperatorSet make_abstract_tensor_graph_ir_ops() {
  GraphIrOperatorSet ops = make_core_graph_ir_ops();

  ops.add(
      GraphIrOpSpec{"tensor_node", 1, 1, "tensor_node(op: string) -> node id"},
      [](const GraphIrOpSpec&,
         const std::vector<GraphIrValue>& args,
         GraphIrValue& out,
         GraphIrContext& ctx,
         std::string& error) {
        if (!require_edits(ctx, error)) return false;
        const auto op = as_string(args[0]);
        if (!op) {
          error = "tensor_node(): op must be a string";
          return false;
        }

        const uint32_t node = ctx.edits->add_node("abstract_tensor_tool");
        ctx.edits->set_attr(node, "tensor.op", *op);
        const auto* canonical = nodus::ops::find_op(*op);
        ctx.edits->set_attr(node, "tensor.canonical", canonical != nullptr);
        if (canonical) {
          ctx.edits->set_attr(
              node, "tensor.arity", static_cast<uint32_t>(canonical->arity));
          ctx.edits->set_attr(node, "tensor.lowerable", canonical->lowerable);
        }
        out = node;
        return true;
      });

  ops.add(
      GraphIrOpSpec{
          "tensor_input", 2, 2, "tensor_input(node: u32, role: string) -> port id"},
      [](const GraphIrOpSpec&,
         const std::vector<GraphIrValue>& args,
         GraphIrValue& out,
         GraphIrContext& ctx,
         std::string& error) {
        if (!require_edits(ctx, error)) return false;
        const auto node = as_u32(args[0]);
        const auto role = as_string(args[1]);
        if (!node || !role) {
          error = "tensor_input(): expected (u32, string)";
          return false;
        }
        const uint32_t port = ctx.edits->add_port(*node, kPortFlagInput, 0);
        ctx.edits->set_attr(port, "tensor.role", *role);
        out = port;
        return true;
      });

  ops.add(
      GraphIrOpSpec{
          "tensor_output", 2, 2, "tensor_output(node: u32, role: string) -> port id"},
      [](const GraphIrOpSpec&,
         const std::vector<GraphIrValue>& args,
         GraphIrValue& out,
         GraphIrContext& ctx,
         std::string& error) {
        if (!require_edits(ctx, error)) return false;
        const auto node = as_u32(args[0]);
        const auto role = as_string(args[1]);
        if (!node || !role) {
          error = "tensor_output(): expected (u32, string)";
          return false;
        }
        const uint32_t port = ctx.edits->add_port(*node, kPortFlagOutput, 0);
        ctx.edits->set_attr(port, "tensor.role", *role);
        out = port;
        return true;
      });

  return ops;
}

} // namespace nodus::tensors

