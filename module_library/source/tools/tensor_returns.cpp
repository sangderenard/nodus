// tensor_returns: hands a table's used-up AbstractTensor handles on to the mint table (research case 6393486c).
//
// Owner design, 2026-11-19: tensors are lent out by a mint table, whose only job is to take returned tensors in and hand
// them out again cleaned, minting new ones when returns do not keep up with the loans. A table's Borrower rows
// (compiled rows, tensor_handle_dup) only read the handles they pop; the host records a popped handle as used up once no
// stack entry names it (thread_manager, ITool::tensor_role), and offers those to this row. A duplicated handle has
// every use as its own stack entry, so it is offered only after its last use.
//
// This row publishes each one, oldest first, on its own port "returns" (8-byte handle tokens), wired by a port pair to
// a mint row's "returned" port. The edge is lossless (top_k = 0): a full edge refuses, and what was not taken stays
// with the host for the next tick. The stack is not touched.
#include "tool_api.h"
#include "value_types.h"

#include <cstdint>

namespace {

class TensorReturnsTool final : public ITool {
public:
    // The id is the plugin's ("tensor_returns") unless an image registers its own copy under another
    // (nodus_link_rows.cpp: "nodus.link.returns", on a link's send tables).
    explicit TensorReturnsTool(const char* id = "tensor_returns") : id_(id) {}
    const char* id_cstr() const noexcept override { return id_; }
    const char* name_cstr() const noexcept override { return "Tensor returns"; }
    ToolCaps caps() const override { return static_cast<ToolCaps>(0); }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    ToolTensorRole tensor_role() const override { return ToolTensorRole::Returns; }
    int32_t port_count() const override { return 0; }
    ToolPortSpec port_spec(int32_t) const override { return ToolPortSpec{}; }
    int32_t row_port_count() const override { return 1; }
    ToolRowPort row_port(int32_t idx) const override {
        if (idx == 0) return ToolRowPort{ToolPortKind::Return, 1.0f, 0.5f, "returns", 8};
        return ToolRowPort{};
    }
    void execute_stack(ToolStackContext& ctx) override {
        ctx.succeeded = true;  // handing on nothing is not a refusal: the rest of the table still runs
        if (!ctx.returns || !ctx.row_ports || !ctx.row_ports->publish) return;
        ToolReturnedTensors& r = *ctx.returns;
        size_t taken = 0;
        while (taken < r.count) {
            const uint64_t id = r.ids[taken];
            if (!ctx.row_ports->publish(ctx.row_ports->host, 0, &id, sizeof(id))) break;
            ++taken;
        }
        r.taken = taken;
    }

private:
    const char* id_;
};

}  // namespace

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new TensorReturnsTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) { delete t; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
