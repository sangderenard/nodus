// tensor_mint: the mint table's generator row (research case 6393486c). Owner design, 2026-11-19:
//
//   "pull preallocated tensors from a table who's sole responsibility is to take in returned tensors and pass them out
//    again cleaned or make new ones if returns aren't paying for loans, we set up a table doing that with our preferred
//    tensor size set by numeric values sent through edges to the row parameter inputs of the tensor generator row"
//
// One row mints one size. Its parameters are its stack arguments, as every Nodus row takes them: the table's Input row
// receives six numbers over edges and pushes them, dtype d0 d1 d2 d3 depth (top last). A 0 keeps the current value, so
// an idle Input row (which pushes 0) changes nothing. dtype is the Nodus tensor dtype code (2 = float64); the rank is
// the count of leading positive dims; depth is how many cleaned tensors the row keeps in hand (default 8).
//
// Returned tensors come in on the row's own port "returned" and on the stack under its parameters. Those of this row's
// size are zeroed and kept; everything else is destroyed. Owner, 2026-11-21: "change the mint trash to just trash
// anything there is no reason to have mints filtered ... there's no point to filter the garbage slot". So a mint with no
// size set is a plain trash: whatever a returns row hands it (sent files, canvas snapshots, any size) is freed. Each port "blank0".."blank7" wired to a borrowing row is kept full of blanks: the oldest kept
// tensor first, a new one only when none is kept (owner, 2026-11-19: "prefer old tensors before new ones"). Those edges
// are lossless (top_k = 0), so a full edge refuses and the tensor stays here: the loans are bounded by the edges' slots.
// The row's buffer holds depth tensors; the newest arrivals that do not fit are killed (owner, 2026-11-19).
// The returns edge feeding "returned" must hold a whole tick of the borrowing table's returns, or its backlog grows.
#include "tool_api.h"
#include "value_types.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <vector>

namespace {

constexpr int32_t kBlankPorts = 8;
constexpr const char* kBlankNames[kBlankPorts] = {"blank0", "blank1", "blank2", "blank3",
                                                   "blank4", "blank5", "blank6", "blank7"};

// One numeric stack entry as a double; false (nothing popped) at a tensor handle, an empty stack or a non-number.
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

class TensorMintTool final : public ITool {
public:
    // The id is the plugin's ("tensor_mint") unless an image registers its own copy under another (nodus_link_rows.cpp:
    // "nodus.link.trash", the trash of a link's send tables).
    explicit TensorMintTool(const char* id = "tensor_mint") : id_(id) {}
    const char* id_cstr() const noexcept override { return id_; }
    const char* name_cstr() const noexcept override { return "Tensor mint"; }
    ToolCaps caps() const override { return static_cast<ToolCaps>(0); }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {
        for (uint64_t id : kept_) nodus_tensor_destroy(id);
        kept_.clear();
    }
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    int32_t port_count() const override { return 1; }
    ToolPortSpec port_spec(int32_t idx) const override {
        if (idx == 0) return ToolPortSpec{ToolPortKind::Argument, 6};
        return ToolPortSpec{};
    }
    int32_t row_port_count() const override { return 1 + kBlankPorts; }
    ToolRowPort row_port(int32_t idx) const override {
        if (idx == 0) return ToolRowPort{ToolPortKind::Argument, 0.0f, 0.5f, "returned", 8};
        if (idx >= 1 && idx <= kBlankPorts) {
            const float x = 0.30f + 0.70f * static_cast<float>(idx - 1) / static_cast<float>(kBlankPorts - 1);
            return ToolRowPort{ToolPortKind::Return, x, 1.0f, kBlankNames[idx - 1], 8};
        }
        return ToolRowPort{};
    }
    void execute_stack(ToolStackContext& ctx) override {
        ctx.succeeded = true;
        if (!ctx.stack.raw) return;
        RawStackFrame& frame = *ctx.stack.raw;

        // Parameters: dtype d0 d1 d2 d3 depth, depth on top.
        std::array<double, 6> p{};
        for (int i = 5; i >= 0; --i) {
            double v = 0.0;
            if (!pop_number(frame, v)) break;
            p[static_cast<size_t>(i)] = v;
        }
        if (p[0] > 0.0) dtype_ = static_cast<int32_t>(std::lround(p[0]));
        if (p[1] > 0.0 || p[2] > 0.0 || p[3] > 0.0 || p[4] > 0.0) {
            rank_ = 0;
            for (int a = 0; a < 4 && p[static_cast<size_t>(1 + a)] > 0.0; ++a)
                shape_[static_cast<size_t>(rank_++)] = static_cast<uint64_t>(std::llround(p[static_cast<size_t>(1 + a)]));
        }
        if (p[5] > 0.0) depth_ = static_cast<size_t>(std::llround(p[5]));
        const bool sized = dtype_ > 0 && rank_ > 0;

        // Returns: from under the parameters and from the "returned" port. Kept if this row's size, else trash.
        auto take = [&](uint64_t id) {
            if (!id) return;
            if (sized && is_my_size(id) && nodus_tensor_zero(id, 0, 0) == NODUS_OK) kept_.push_back(id);
            else nodus_tensor_destroy(id);
        };
        std::vector<uint64_t> from_stack;
        for (;;) {
            nodus::tensors::AbstractTensorHandle h{};
            ValueTypeId tid = kInvalidValueTypeId;
            if (!raw_stack_peek_type(frame, tid) || tid != ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR)) break;
            if (!raw_stack_pop_abstract_tensor(frame, h)) break;
            from_stack.push_back(h.id);
        }
        for (auto it = from_stack.rbegin(); it != from_stack.rend(); ++it) take(*it);  // bottom first
        if (ctx.row_ports && ctx.row_ports->consume) {
            for (;;) {
                uint64_t id = 0;
                size_t written = 0;
                if (!ctx.row_ports->consume(ctx.row_ports->host, 0, &id, sizeof(id), &written) || written != sizeof(id)) break;
                take(id);
            }
        }

        // Loans: keep every wired blank port full.
        if (sized && ctx.row_ports && ctx.row_ports->publish) {
            for (int32_t port = 1; port <= kBlankPorts; ++port) {
                for (;;) {
                    uint64_t id = 0;
                    if (!kept_.empty()) {
                        id = kept_.front();  // old tensors before new ones
                        kept_.pop_front();
                    } else {
                        id = nodus_tensor_create(dtype_, static_cast<uint32_t>(rank_), shape_.data());
                        if (!id) break;
                        if (nodus_tensor_zero(id, 0, 0) != NODUS_OK) { nodus_tensor_destroy(id); break; }
                    }
                    if (!ctx.row_ports->publish(ctx.row_ports->host, port, &id, sizeof(id))) {
                        kept_.push_front(id);  // refused (full, or nothing wired): it stays here, still the oldest
                        break;
                    }
                }
            }
        }
        // The buffer holds depth tensors: the newest arrivals that do not fit are killed.
        while (kept_.size() > depth_) {
            nodus_tensor_destroy(kept_.back());
            kept_.pop_back();
        }
    }

private:
    bool is_my_size(uint64_t id) const {
        NodusTensorDesc d{};
        if (nodus_tensor_describe(id, &d) != NODUS_OK) return false;
        if (d.dtype != dtype_ || static_cast<int32_t>(d.rank) != rank_) return false;
        for (int32_t a = 0; a < rank_; ++a)
            if (d.shape[a] != shape_[static_cast<size_t>(a)]) return false;
        return true;
    }

    int32_t dtype_ = 0;
    int32_t rank_ = 0;
    std::array<uint64_t, 4> shape_{};
    size_t depth_ = 8;
    std::deque<uint64_t> kept_;  // oldest at the front: lent first; newest at the back: killed first when over depth
    const char* id_;
};

}  // namespace

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new TensorMintTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) { delete t; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
