// nodus_link: the Nodus link's four canvas rows, <id>.control.send/.receive and <id>.bulk.send/.receive
// (include/nodus_link_rows.h).
// The row tool's source, in the module library (case 74fd8157): the Menu lists it, the canvas compiler inlines it,
// it builds as its own plugin DLL, and the image registers rows from it (src/nodus_link_rows.cpp).
// Its settings are its saved state (serialize/deserialize): the row id and which of the four rows it is. The endpoint the
// rows share (a listener or dialer with the link) is process state and is not saved state; a row made with no endpoint
// refuses in execute_stack (nodus_link_endpoint.h holds the endpoint class).
// Research case 98d49cb8520e44d092c5d134c37c3c32, stage 2.
#include "nodus_link_rows.h"
#include "nodus_link_endpoint.h"

#include "common/tensors/abstraction/tensor_abi.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nodus::link {

LinkRow::LinkRow(std::string id, LinkRowRole role, std::shared_ptr<LinkEndpoint> ep)
    : id_(std::move(id)), role_(role), ep_(std::move(ep)) {}
LinkRow::LinkRow() = default;
void LinkRow::configure(std::string id, LinkRowRole role) { id_ = std::move(id); role_ = role; }
const char* LinkRow::name_cstr() const noexcept {
    switch (role_) {
        case LinkRowRole::ControlSend: return "Link control send";
        case LinkRowRole::ControlReceive: return "Link control receive";
        case LinkRowRole::BulkSend: return "Link bulk send";
        default: return "Link bulk receive";
    }
}

// Saved state: magic "NLR1", version 2, id length, id, role (u64 0..3: control send, control receive, bulk send, bulk
// receive). Version 1 was one row that both sent and received; it is not read.
namespace {
constexpr uint32_t kLinkRowMagic = 0x31524C4E, kLinkRowVersion = 2;
}
bool LinkRow::serialize(OutputArchive& out) const {
    const uint32_t head[3] = {kLinkRowMagic, kLinkRowVersion, static_cast<uint32_t>(id_.size())};
    const uint64_t values[1] = {static_cast<uint64_t>(role_)};
    return out.write(head, sizeof(head)) && out.write(id_.data(), id_.size()) && out.write(values, sizeof(values));
}
bool LinkRow::deserialize(InputArchive& in) {
    uint32_t head[3] = {};
    if (!in.read(head, sizeof(head)) || head[0] != kLinkRowMagic || head[1] != kLinkRowVersion || head[2] > 4096) return false;
    std::string id; id.resize(head[2]);
    uint64_t values[1] = {};
    if (!in.read(id.data(), id.size()) || !in.read(values, sizeof(values)) || values[0] > 3) return false;
    configure(std::move(id), static_cast<LinkRowRole>(values[0]));
    return true;
}
void LinkRow::initialize(const ToolInitContext& ctx) {
    if (ctx.serialized_path && ctx.serialized_path[0] != '\0') {
        FileInputArchive in(ctx.serialized_path);
        if (in.ok()) deserialize(in);
    }
}
void LinkRow::tick(double, HostAPI&) {
    if (!ep_) return;
    std::lock_guard<std::mutex> g(ep_->mutex()); ep_->poll();
}
void LinkRow::execute_stack(ToolStackContext& context) {
    context.succeeded = false;
    if (!ep_ || !context.stack.raw) return;
    std::lock_guard<std::mutex> g(ep_->mutex());
    ep_->poll();
    bool accepted = true;
    if (sends()) send(*context.stack.raw, accepted);
    else receive(*context.stack.raw);
    context.succeeded = accepted;
}

// Send: the tensor waiting on top of the stack. An Input row with an empty edge pads the stack with one value of the
// stack's default type (a scalar, not a tensor) or a null handle; either means nothing to send. With stack loopback a
// refused input stays on the stack across ticks -- under the next tick's pad, where a tensor pop fails on the scalar
// and the input is never seen again: on the Oracle A1 host (2026-11-21) 6 of 20 refused 8 MiB inputs sat under pads
// for minutes while the link stood idle. Every pad is dropped, so a waiting input is retried; a refused one goes back.
void LinkRow::send(RawStackFrame& stack, bool& accepted) {
    const ValueTypeId tensor_type = ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR);
    nodus::tensors::AbstractTensorHandle input{};
    bool found = false;
    ValueTypeId top = kInvalidValueTypeId;
    while (raw_stack_peek_type(stack, top)) {
        if (top == tensor_type) {
            if (!raw_stack_pop_abstract_tensor(stack, input)) break;
            if (input.id) { found = true; break; }
            continue;
        }
        const ValueType* pad = ValueTypeRegistry::global().get(top);
        if (!pad || !pad->size) break;
        std::vector<uint8_t> discard(pad->size);
        if (!raw_stack_pop_typed(stack, discard.data(), top)) break;
    }
    if (!found) return;
    if (!(bulk() ? accept_bulk(input.id) : accept_control(input.id))) {
        ++(bulk() ? ep_->report().bulk_refused : ep_->report().control_refused);
        raw_stack_push_abstract_tensor(stack, input);  // refused: it stays on the stack for the rows below
        accepted = false;
        return;
    }
    ep_->poll();
}

// Receive: every arrival onto the stack while it has room; an arrival the stack cannot hold waits in the endpoint.
void LinkRow::receive(RawStackFrame& stack) {
    const ValueType* vt = ValueTypeRegistry::global().get(ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR));
    const size_t entry = vt ? vt->size : sizeof(nodus::tensors::AbstractTensorHandle);
    // Taken in arrival order, pushed newest first, so the oldest arrival is on top: whatever pops the stack reads them in
    // the order they came.
    std::vector<uint64_t> arrived;
    size_t room = stack.byte_capacity > stack.byte_count ? (stack.byte_capacity - stack.byte_count) / entry : 0;
    while (room-- > 0) {
        const uint64_t t = bulk() ? ep_->take_bulk() : control_tensor();
        if (!t) break;
        arrived.push_back(t);
    }
    for (size_t i = arrived.size(); i-- > 0;) {
        if (!raw_stack_push_abstract_tensor(stack, nodus::tensors::AbstractTensorHandle{arrived[i]})) {
            nodus_tensor_destroy(arrived[i]);   // room was checked; a push that still fails is a fault, never silent
            ++ep_->report().bulk_dropped;
        }
    }
}
bool LinkRow::accept_control(uint64_t h) {
    NodusTensorDesc d{}; uint8_t b[kControlPacketBytes]; ControlPacket p;
    if (nodus_tensor_describe(h, &d) != NODUS_OK || d.dtype != NODUS_DTYPE_U8 || d.rank != 1 || d.total_bytes != kControlPacketBytes
        || nodus_tensor_read(h, 0, b, sizeof(b)) != int64_t(sizeof(b)) || !decode_control(b, sizeof(b), p)) return false;
    return ep_->send_control(p);
}
bool LinkRow::accept_bulk(uint64_t h) {
    NodusTensorDesc d{};
    if (!ep_->linked() || nodus_tensor_describe(h, &d) != NODUS_OK || !d.total_bytes || d.total_bytes > ep_->maximum()) return false;
    std::vector<uint8_t> bytes(size_t(d.total_bytes));
    if (nodus_tensor_read(h, 0, bytes.data(), bytes.size()) != int64_t(bytes.size())) return false;
    return ep_->queue_bulk(d, std::move(bytes));
}
uint64_t LinkRow::control_tensor() {
    ControlPacket p; if (!ep_->receive_control(p)) return 0;
    uint8_t b[kControlPacketBytes]; encode_control(p, b);
    const uint64_t shape[1] = {kControlPacketBytes};
    uint64_t t = nodus_tensor_create(NODUS_DTYPE_U8, 1, shape);
    if (!t) return 0;
    if (nodus_tensor_write(t, 0, b, sizeof(b)) != int64_t(sizeof(b))) { nodus_tensor_destroy(t); return 0; }
    return t;
}

}  // namespace nodus::link

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::link::LinkRow(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
