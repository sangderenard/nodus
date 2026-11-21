// tensor_filter_send: sends tensors back up a table by their size, into three loopbacks. Owner design, 2026-11-19:
//
//   "you'll need a filter send row and three governed intake rows each obtaining supply from feedback. With the
//    doubling work, we need to make sure that we have a burn extra policy that will make sure the fifo is cleared to
//    have one open spot at all times for all three loopback"
//
// The row filters; it decides nothing by need (the mint dispenses sizes according to need). Its parameters are its
// stack arguments, as every Nodus row takes them: three sizes in bytes, s0 s1 s2 (s2 on top), each the size of the
// tensors its loopback carries. A 0 keeps the current value, so an idle Input row (which pushes 0) changes nothing.
//
// Under the parameters, every tensor on the stack is filtered, bottom first. A tensor of size s<i> is sent on the
// row's own port "loop<i>" (8-byte handle tokens), wired by a port pair to a governed intake row's "loop" port. A null
// handle (an Input row's pad) is nothing and is dropped. A tensor of no loopback's size is refused: it is left on the
// stack, untouched (in a table set in stack loopback it is still there next tick).
//
// Burn extra: a loopback FIFO always keeps one open spot. A tensor is sent only while its loopback can take two; when
// it cannot, the tensor is extra (the link doubles work: every crossing is a new tensor) and is burned (destroyed).
#include "tool_api.h"
#include "value_types.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int32_t kLoops = 3;
constexpr const char* kLoopNames[kLoops] = {"loop0", "loop1", "loop2"};

bool pop_number(RawStackFrame& frame, double& out) {
    ValueTypeId tid = kInvalidValueTypeId;
    if (!raw_stack_peek_type(frame, tid)) return false;
    if (tid == ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR)) return false;
    const ValueType* vt = ValueTypeRegistry::global().get(tid);
    if (!vt) return false;
    if (vt->size == sizeof(float)) {
        float f = 0.0f;
        if (!raw_stack_pop_typed(frame, &f, tid)) return false;
        out = static_cast<double>(f);
        return true;
    }
    if (vt->size == sizeof(double)) {
        double d = 0.0;
        if (!raw_stack_pop_typed(frame, &d, tid)) return false;
        out = d;
        return true;
    }
    return false;
}

class TensorFilterSendTool final : public ITool {
public:
    const char* id_cstr() const noexcept override { return "tensor_filter_send"; }
    const char* name_cstr() const noexcept override { return "Tensor filter send"; }
    ToolCaps caps() const override { return static_cast<ToolCaps>(0); }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    int32_t port_count() const override { return 1; }
    ToolPortSpec port_spec(int32_t idx) const override {
        if (idx == 0) return ToolPortSpec{ToolPortKind::Argument, kLoops};
        return ToolPortSpec{};
    }
    int32_t row_port_count() const override { return kLoops; }
    ToolRowPort row_port(int32_t idx) const override {
        if (idx >= 0 && idx < kLoops)
            return ToolRowPort{ToolPortKind::Return, 1.0f, 0.25f + 0.25f * static_cast<float>(idx), kLoopNames[idx], 8};
        return ToolRowPort{};
    }
    void execute_stack(ToolStackContext& ctx) override {
        ctx.succeeded = true;
        if (!ctx.stack.raw) return;
        RawStackFrame& frame = *ctx.stack.raw;

        std::array<double, kLoops> p{};
        for (int i = kLoops - 1; i >= 0; --i) {
            double v = 0.0;
            if (!pop_number(frame, v)) break;
            p[static_cast<size_t>(i)] = v;
        }
        for (int i = 0; i < kLoops; ++i)
            if (p[static_cast<size_t>(i)] > 0.0) size_[static_cast<size_t>(i)] = static_cast<uint64_t>(std::llround(p[static_cast<size_t>(i)]));

        std::vector<uint64_t> on_stack;
        for (;;) {
            ValueTypeId tid = kInvalidValueTypeId;
            if (!raw_stack_peek_type(frame, tid) || tid != ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR)) break;
            nodus::tensors::AbstractTensorHandle h{};
            if (!raw_stack_pop_abstract_tensor(frame, h)) break;
            on_stack.push_back(h.id);
        }
        std::vector<uint64_t> refused;
        for (auto it = on_stack.rbegin(); it != on_stack.rend(); ++it) {  // bottom first
            const uint64_t id = *it;
            if (!id) continue;  // the pad: nothing
            const int loop = loop_of(id);
            if (loop < 0) { refused.push_back(id); continue; }
            const bool room = ctx.row_ports && ctx.row_ports->can_publish && ctx.row_ports->publish &&
                              ctx.row_ports->can_publish(ctx.row_ports->host, loop, 2);
            if (room && ctx.row_ports->publish(ctx.row_ports->host, loop, &id, sizeof(id))) continue;
            nodus_tensor_destroy(id);  // extra: the loopback keeps its one open spot
            ++burned_[static_cast<size_t>(loop)];
        }
        for (uint64_t id : refused) raw_stack_push_abstract_tensor(frame, nodus::tensors::AbstractTensorHandle{id});
    }

private:
    int loop_of(uint64_t id) const {
        NodusTensorDesc d{};
        if (nodus_tensor_describe(id, &d) != NODUS_OK) return -1;
        for (int i = 0; i < kLoops; ++i)
            if (size_[static_cast<size_t>(i)] && d.total_bytes == size_[static_cast<size_t>(i)]) return i;
        return -1;
    }

    std::array<uint64_t, kLoops> size_{};
    std::array<uint64_t, kLoops> burned_{};
};

}  // namespace

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new TensorFilterSendTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) { delete t; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
