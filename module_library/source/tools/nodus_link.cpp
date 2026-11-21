// nodus_link: the Nodus link's two canvas rows, <id>.control and <id>.bulk (include/nodus_link_rows.h).
// The row tool's source, in the module library (case 74fd8157): the Menu lists it, the canvas compiler inlines it,
// it builds as its own plugin DLL, and the image registers rows from it (src/nodus_link_rows.cpp).
// Its settings are its saved state (serialize/deserialize): the row id and which of the two rows it is. The endpoint the
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

LinkRow::LinkRow(std::string id, bool bulk, std::shared_ptr<LinkEndpoint> ep) : id_(std::move(id)), bulk_(bulk), ep_(std::move(ep)) {}
LinkRow::LinkRow() = default;
void LinkRow::configure(std::string id, bool bulk) { id_ = std::move(id); bulk_ = bulk; }

// Saved state: magic "NLR1", version 1, id length, id, bulk (u64 0/1).
namespace {
constexpr uint32_t kLinkRowMagic = 0x31524C4E, kLinkRowVersion = 1;
}
bool LinkRow::serialize(OutputArchive& out) const {
    const uint32_t head[3] = {kLinkRowMagic, kLinkRowVersion, static_cast<uint32_t>(id_.size())};
    const uint64_t values[1] = {bulk_ ? 1u : 0u};
    return out.write(head, sizeof(head)) && out.write(id_.data(), id_.size()) && out.write(values, sizeof(values));
}
bool LinkRow::deserialize(InputArchive& in) {
    uint32_t head[3] = {};
    if (!in.read(head, sizeof(head)) || head[0] != kLinkRowMagic || head[1] != kLinkRowVersion || head[2] > 4096) return false;
    std::string id; id.resize(head[2]);
    uint64_t values[1] = {};
    if (!in.read(id.data(), id.size()) || !in.read(values, sizeof(values)) || values[0] > 1) return false;
    configure(std::move(id), values[0] == 1);
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
    auto& stack = *context.stack.raw;
    bool accepted = false;
    nodus::tensors::AbstractTensorHandle input{};
    // An Input row with an empty edge pads the stack with one value of the stack's default type (a scalar, not a
    // tensor) or a null handle; either means nothing to send. With stack loopback a refused input stays on the stack
    // across ticks -- under the next tick's pad, where a tensor pop fails on the scalar and the input is never seen
    // again: on the Oracle A1 host (2026-11-21) 6 of 20 refused 8 MiB inputs sat under pads for minutes while the link
    // stood idle. Drop every pad, so a waiting input is retried.
    const ValueTypeId tensor_type = ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR);
    bool popped = false;
    ValueTypeId top = kInvalidValueTypeId;
    while (raw_stack_peek_type(stack, top)) {
        if (top == tensor_type) {
            if (!raw_stack_pop_abstract_tensor(stack, input)) break;
            popped = true;
            if (input.id) break;
            accepted = true;
            continue;
        }
        const ValueType* pad = ValueTypeRegistry::global().get(top);
        if (!pad || !pad->size) break;
        std::vector<uint8_t> discard(pad->size);
        if (!raw_stack_pop_typed(stack, discard.data(), top)) break;
        accepted = true;
    }
    if (popped && input.id) {
        if (!(bulk_ ? accept_bulk(input.id) : accept_control(input.id))) {
            ++(bulk_ ? ep_->report().bulk_refused : ep_->report().control_refused);
            raw_stack_push_abstract_tensor(stack, input);  // refused: it stays on the stack for the rows below
            accepted = false;
        } else {
            accepted = true; ep_->poll();
        }
    }
    // Arrivals go out on the row port "received", one per token, while its edge can take them.
    const ToolRowPortAccess* ports = context.row_ports;
    while (ports && ports->publish && ports->can_publish && ports->can_publish(ports->host, 0, 1)) {
        const uint64_t out = bulk_ ? ep_->take_bulk() : control_tensor();
        if (!out) break;
        if (!ports->publish(ports->host, 0, &out, sizeof(out))) { nodus_tensor_destroy(out); break; }
    }
    context.succeeded = accepted;
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
