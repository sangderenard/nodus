#include "common/tensors/abstraction/abstract_tensor_graph_ir.h"

#include "canonical_ops.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "common/tensors/abstraction/tensor_math.h"
#include "common/tensors/abstraction/graph_sparse.h"
#include "tool_ir.h"
#include "tool_registry.h"

#include <memory>
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

        const auto* canonical = nodus::ops::find_op(*op);
        const std::string tool_id = canonical
            ? "abstract_tensor." + *op
            : "abstract_tensor.structural";
        const uint32_t node = ctx.edits->add_node(tool_id);
        ctx.edits->set_attr(node, "tensor.op", *op);
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

size_t register_abstract_tensor_tool_ir(ToolRegistry& registry) {
  size_t registered = 0;

  for (const auto& descriptor : nodus::ops::kOps) {
    if (descriptor.ct_value < 0 ||
        (descriptor.arity != 1 && descriptor.arity != 2))
      continue;

    auto ir = std::make_shared<ToolIR>();
    ir->id = "abstract_tensor." + std::string(descriptor.name);
    ir->name = "AbstractTensor " + std::string(descriptor.name);
    ir->caps = ToolCaps::None;
    const uint8_t arity = descriptor.arity;
    const auto operation =
        static_cast<nodus::ops::CanonicalOp>(descriptor.canonical_id);

    ir->port_count = [] { return 2; };
    ir->port_spec = [arity](int32_t index) {
      if (index == 0)
        return ToolPortSpec{ToolPortKind::Argument, arity};
      if (index == 1)
        return ToolPortSpec{ToolPortKind::Return, 1};
      return ToolPortSpec{};
    };
    ir->execute_stack = [operation, arity](ToolStackContext& ctx) {
      if (!ctx.stack.raw) return;
      RawStackFrame& frame = *ctx.stack.raw;
      AbstractTensorHandle right{};
      AbstractTensorHandle left{};
      if (arity == 2 &&
          !raw_stack_pop_abstract_tensor(frame, right))
        return;
      if (!raw_stack_pop_abstract_tensor(frame, left)) {
        if (right.id) raw_stack_push_abstract_tensor(frame, right);
        return;
      }
      auto restore = [&] {
        raw_stack_push_abstract_tensor(frame, left);
        if (arity == 2) raw_stack_push_abstract_tensor(frame, right);
      };
      NodusTensorDesc desc{};
      if (nodus_tensor_describe(left.id, &desc) != NODUS_OK) {
        restore();
        return;
      }
      AbstractTensorHandle output{};
      output.id = nodus_tensor_create(desc.dtype, desc.rank, desc.shape);
      const bool ok = output.id && (
          arity == 1
              ? nodus_tensor_unary(
                    static_cast<int32_t>(operation), left.id, output.id) == NODUS_OK
              : nodus_tensor_binary(
                    static_cast<int32_t>(operation), left.id, right.id,
                    output.id) == NODUS_OK);
      if (!ok) {
        if (output.id) nodus_tensor_destroy(output.id);
        restore();
        return;
      }
      if (!raw_stack_push_abstract_tensor(frame, output))
        nodus_tensor_destroy(output.id);
    };

    if (registry.register_tool(tool_ir::make_registry_entry(std::move(ir))))
      ++registered;
  }

  // Reduction tools are thin bodies over tensor_math's own full-reduction
  // executors (tensor_reduce_sum_all / tensor_reduce_mean_all), exactly as
  // the elementwise tools are thin bodies over tensor_elementwise_*. The
  // dtype POLICY lives here and follows NumPy: sum keeps the input dtype
  // (bool -> I64); mean is F64 for integer/bool inputs and keeps float
  // width. Unsupported operands are refused by pushing them back.
  struct ReductionSpec {
    const char* name;
    bool divide_by_count;
  };
  static constexpr ReductionSpec kReductions[] = {
      {"sum", false},
      {"mean", true},
  };
  for (const auto& reduction : kReductions) {
    auto ir = std::make_shared<ToolIR>();
    ir->id = std::string("abstract_tensor.") + reduction.name;
    ir->name = std::string("AbstractTensor ") + reduction.name;
    ir->caps = ToolCaps::None;
    ir->port_count = [] { return 2; };
    ir->port_spec = [](int32_t index) {
      if (index == 0)
        return ToolPortSpec{ToolPortKind::Argument, 1};
      if (index == 1)
        return ToolPortSpec{ToolPortKind::Return, 1};
      return ToolPortSpec{};
    };
    const bool divide = reduction.divide_by_count;
    ir->execute_stack = [divide](ToolStackContext& ctx) {
      if (!ctx.stack.raw) return;
      RawStackFrame& frame = *ctx.stack.raw;
      AbstractTensorHandle input{};
      if (!raw_stack_pop_abstract_tensor(frame, input)) return;
      NodusTensorDesc input_desc{};
      if (nodus_tensor_describe(input.id, &input_desc) != NODUS_OK) {
        raw_stack_push_abstract_tensor(frame, input);
        return;
      }
      int32_t output_dtype = input_desc.dtype;
      if (divide) {
        output_dtype = input_desc.dtype == NODUS_DTYPE_F32
            ? NODUS_DTYPE_F32 : NODUS_DTYPE_F64;
      } else if (input_desc.dtype == NODUS_DTYPE_BOOL) {
        output_dtype = NODUS_DTYPE_I64;
      }
      const uint64_t scalar_shape[1] = {1};
      AbstractTensorHandle output{};
      output.id = nodus_tensor_create(output_dtype, 1, scalar_shape);
      const bool ok = output.id &&
          (divide ? nodus_tensor_reduce_mean_all(input.id, output.id)
                  : nodus_tensor_reduce_sum_all(input.id, output.id)) == NODUS_OK;
      if (!ok) {
        if (output.id) nodus_tensor_destroy(output.id);
        raw_stack_push_abstract_tensor(frame, input);
        return;
      }
      if (!raw_stack_push_abstract_tensor(frame, output))
        nodus_tensor_destroy(output.id);
    };
    if (registry.register_tool(tool_ir::make_registry_entry(std::move(ir))))
      ++registered;
  }
  return registered;
}

// Live registration of the canonical vocabulary, same static-init idiom as
// REGISTER_TOOL (tool_registry.h). Until now register_abstract_tensor_tool_ir
// was reachable only from graph_ir_test.cpp, so canvas ROWS records naming
// "abstract_tensor.<op>" plugin ids had nothing to instantiate. The coverage
// filter above (ct_value present, arity 1..2) is deliberate and must not be
// widened ahead of the evaluator: tensor_math's elementwise plan executor
// implements exactly the CT-marked subset and answers quiet-NaN outside it,
// so a wider registration would mint tools that silently compute wrong.
namespace {
struct AbstractTensorVocabularyRegistrar {
  AbstractTensorVocabularyRegistrar() {
    register_abstract_tensor_tool_ir(tool_registry_global());
  }
};
static AbstractTensorVocabularyRegistrar g_abstract_tensor_vocabulary_registrar;
} // namespace

} // namespace nodus::tensors
