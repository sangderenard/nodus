// tensor_governed_intake: a row made to house two inputs and govern their priority. Owner design, 2026-11-19:
//
//   "you could make a loopback by placing two inputs in a row made specifically to house inputs and prioritize them,
//    have it always feed from the loopback, so as things fill they keep refusing new work while continuously burning
//    their last supply"
//   "three governed intake rows each obtaining supply from feedback"
//
// The row's own ports: "loop" (the feedback, wired by a port pair from a filter send row's "loop<i>") and "supply"
// (new work). Each tick it pushes one tensor handle onto the stack, always from the loopback while the loopback has
// one; it takes from supply only when the loopback is empty. What it does not take stays on its edge: the supply edge
// is lossless, so as things fill, new work is refused upstream while the loopback's supply keeps being used. With
// nothing on either, it pushes the null handle, as an Input row pads an empty edge.
#include "tool_api.h"
#include "value_types.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"

#include <cstdint>

namespace {

class TensorGovernedIntakeTool final : public ITool {
public:
    const char* id_cstr() const noexcept override { return "tensor_governed_intake"; }
    const char* name_cstr() const noexcept override { return "Tensor governed intake"; }
    ToolCaps caps() const override { return static_cast<ToolCaps>(0); }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    int32_t port_count() const override { return 1; }
    ToolPortSpec port_spec(int32_t idx) const override {
        if (idx == 0) return ToolPortSpec{ToolPortKind::Return, 1};
        return ToolPortSpec{};
    }
    int32_t row_port_count() const override { return 2; }
    ToolRowPort row_port(int32_t idx) const override {
        if (idx == 0) return ToolRowPort{ToolPortKind::Argument, 0.0f, 0.3f, "loop", 8};
        if (idx == 1) return ToolRowPort{ToolPortKind::Argument, 0.0f, 0.7f, "supply", 8};
        return ToolRowPort{};
    }
    void execute_stack(ToolStackContext& ctx) override {
        ctx.succeeded = true;
        if (!ctx.stack.raw) return;
        uint64_t id = 0;
        if (ctx.row_ports && ctx.row_ports->consume) {
            for (int32_t port = 0; port < 2 && !id; ++port) {  // the loopback first
                size_t written = 0;
                uint64_t got = 0;
                if (ctx.row_ports->consume(ctx.row_ports->host, port, &got, sizeof(got), &written) &&
                    written == sizeof(got))
                    id = got;
            }
        }
        if (!raw_stack_push_abstract_tensor(*ctx.stack.raw, nodus::tensors::AbstractTensorHandle{id}))
            ctx.succeeded = false;
    }
};

}  // namespace

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new TensorGovernedIntakeTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) { delete t; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
