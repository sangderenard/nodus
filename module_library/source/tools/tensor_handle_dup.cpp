// tensor_handle_dup: pushes the AbstractTensor handle on top of the stack a second time (research case 6393486c).
//
// A composed table uses one tensor value twice by holding its handle twice; the tensor itself is not copied. This is
// sound because compiled rows read their input tensors and never destroy them (artifact_importer's buffer wrapper:
// inputs are popped and mapped read-only, only outputs and workspaces are created or destroyed), so two stack entries
// naming one tensor are two reads of it. Owner decision, 2026-11-19: "Handle dup row".
//
// The built-in Clone row repeats a scalar float; this row is its tensor-handle counterpart. A top of stack that is not
// an AbstractTensor handle is refused and left as it was.
#include "tool_api.h"
#include "value_types.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"

namespace {

class TensorHandleDupTool final : public ITool {
public:
    const char* id_cstr() const noexcept override { return "tensor_handle_dup"; }
    const char* name_cstr() const noexcept override { return "Tensor handle dup"; }
    ToolCaps caps() const override { return static_cast<ToolCaps>(0); }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    int32_t port_count() const override { return 3; }
    ToolPortSpec port_spec(int32_t idx) const override {
        if (idx == 0) return ToolPortSpec{ToolPortKind::Argument, 1};
        if (idx == 1 || idx == 2) return ToolPortSpec{ToolPortKind::Return, 1};
        return ToolPortSpec{};
    }
    void execute_stack(ToolStackContext& ctx) override {
        ctx.succeeded = false;
        if (!ctx.stack.raw) return;
        RawStackFrame& frame = *ctx.stack.raw;
        nodus::tensors::AbstractTensorHandle handle{};
        if (!raw_stack_pop_abstract_tensor(frame, handle)) return;
        if (!raw_stack_push_abstract_tensor(frame, handle)) return;
        if (!raw_stack_push_abstract_tensor(frame, handle)) {
            // Leave the stack as it was found: the one entry is already back on it.
            return;
        }
        ctx.succeeded = true;
    }
};

}  // namespace

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new TensorHandleDupTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) { delete t; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
