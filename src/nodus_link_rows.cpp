// The Nodus link's two canvas rows and the endpoint they share (include/nodus_link_rows.h).
// Research case 98d49cb8520e44d092c5d134c37c3c32, stage 2.
#include "nodus_link_rows.h"

#include "common/tensors/abstraction/tensor_abi.h"

#include <cstdio>
#include <ctime>
#include <deque>
#include <exception>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace nodus::link {
namespace {

// Bulk descriptor carried by Begin: magic "NLD1" u32 | dtype u32 | rank u32 | reserved u32 | shape u64[rank].
constexpr uint32_t kDescriptorMagic = 0x31444C4E;
constexpr size_t kPendingOut = 2;      // tensors copied and waiting to go, per endpoint
constexpr size_t kCompletedIn = 4;     // arrived tensors waiting to be popped

std::vector<uint8_t> encode_descriptor(const NodusTensorDesc& d) {
    std::vector<uint8_t> b(16 + 8 * size_t(d.rank));
    put_le(b.data(), kDescriptorMagic, 4); put_le(b.data() + 4, d.dtype, 4); put_le(b.data() + 8, d.rank, 4); put_le(b.data() + 12, 0, 4);
    for (uint32_t i = 0; i < d.rank; ++i) put_le(b.data() + 16 + 8 * i, d.shape[i], 8);
    return b;
}
bool decode_descriptor(const std::vector<uint8_t>& b, int32_t& dtype, uint32_t& rank, uint64_t* shape) {
    if (b.size() < 16 || get_le(b.data(), 4) != kDescriptorMagic) return false;
    dtype = int32_t(get_le(b.data() + 4, 4)); rank = uint32_t(get_le(b.data() + 8, 4));
    if (rank == 0 || rank > NODUS_TENSOR_MAX_RANK || b.size() != 16 + 8 * size_t(rank)) return false;
    for (uint32_t i = 0; i < rank; ++i) shape[i] = get_le(b.data() + 16 + 8 * i, 8);
    return true;
}

}  // namespace

class LinkEndpoint {
public:
    explicit LinkEndpoint(LinkHostConfig c) : maximum_(c.maximum_bulk_bytes) {
        ListenerConfig lc; lc.port = c.port; lc.prefix = c.prefix; lc.key = c.key;
        listener_ = std::make_unique<LinkListener>(std::move(lc));
        report_.listen_port = listener_->port();
    }
    explicit LinkEndpoint(LinkDialConfig c) : maximum_(c.maximum_bulk_bytes) {
        dialer_ = std::make_unique<LinkDialer>(c.target, c.control_ticket, c.bulk_ticket);
    }
    ~LinkEndpoint() {
        for (auto& p : partial_) nodus_tensor_destroy(p.second.handle);
        for (auto h : completed_) nodus_tensor_destroy(h);
    }
    uint16_t listen_port() const { return report_.listen_port; }
    std::mutex& mutex() { return mutex_; }

    // Advance whichever side this is, then the link and the bulk pumps. Caller holds mutex().
    void poll() {
        if (listener_) {
            listener_->poll(uint64_t(std::time(nullptr)));
            while (auto l = listener_->take()) {
                if (link_ && !link_->closed()) { l->close(1008); ++report_.peers_refused; continue; }
                adopt(std::move(l));
            }
        } else if (dialer_ && !link_) {
            dialer_->poll();
            if (dialer_->failed()) report_.state = EndpointState::Failed;
            else if (dialer_->ready()) adopt(dialer_->take());
        }
        if (!link_) return;
        link_->poll();
        pump_out(); pump_in();
        link_->poll();
        report_.link = link_->stats();
        if (link_->closed()) { report_.state = EndpointState::Closed; drop_partials(); }
    }
    bool linked() const { return link_ && !link_->closed(); }

    bool send_control(const ControlPacket& p) { return linked() && link_->send_control(p); }
    bool receive_control(ControlPacket& p) { return link_ && link_->receive_control(p); }

    bool queue_bulk(const NodusTensorDesc& d, std::vector<uint8_t> bytes) {
        if (!linked() || out_.size() >= kPendingOut) return false;
        Out o; o.descriptor = encode_descriptor(d); o.bytes = std::move(bytes); out_.push_back(std::move(o)); return true;
    }
    uint64_t take_bulk() {
        if (completed_.empty()) return 0;
        uint64_t h = completed_.front(); completed_.pop_front(); return h;
    }
    uint64_t maximum() const { return maximum_; }
    EndpointReport& report() { return report_; }

private:
    struct Out { std::vector<uint8_t> descriptor, bytes; uint64_t stream = 0; size_t written = 0; };
    struct Partial { uint64_t handle = 0, total = 0; bool dropped = false; };

    void adopt(std::unique_ptr<Link> l) {
        drop_partials(); out_.clear();
        link_ = std::move(l); report_.state = EndpointState::Linked; report_.account = link_->account();
    }
    void drop_partials() {
        for (auto& p : partial_) nodus_tensor_destroy(p.second.handle);
        partial_.clear();
    }
    void pump_out() {
        while (!out_.empty()) {
            Out& o = out_.front();
            if (!o.stream) {
                o.stream = link_->begin_bulk(o.descriptor.data(), o.descriptor.size(), o.bytes.size());
                if (!o.stream) return;
            }
            if (o.written < o.bytes.size()) o.written += link_->write_bulk(o.stream, o.bytes.data() + o.written, o.bytes.size() - o.written);
            if (o.written < o.bytes.size() || !link_->end_bulk(o.stream)) return;
            ++report_.bulk_tensors_sent; out_.pop_front();
        }
    }
    void pump_in() {
        BulkEvent e;
        while (completed_.size() < kCompletedIn && link_->receive_bulk(e)) {
            if (e.kind == BulkKind::Begin) {
                Partial p; p.total = e.total;
                int32_t dtype = 0; uint32_t rank = 0; uint64_t shape[NODUS_TENSOR_MAX_RANK] = {};
                NodusTensorDesc d{};
                if (e.total > maximum_ || !decode_descriptor(e.data, dtype, rank, shape)
                    || !(p.handle = nodus_tensor_create(dtype, rank, shape))
                    || nodus_tensor_describe(p.handle, &d) != NODUS_OK || d.total_bytes != e.total) {
                    if (p.handle) nodus_tensor_destroy(p.handle);
                    p.handle = 0; p.dropped = true; ++report_.bulk_dropped;
                }
                partial_[e.stream] = p;
            } else if (e.kind == BulkKind::Chunk) {
                auto it = partial_.find(e.stream);
                if (it == partial_.end() || it->second.dropped) continue;
                if (nodus_tensor_write(it->second.handle, e.offset, e.data.data(), e.data.size()) != int64_t(e.data.size())) {
                    nodus_tensor_destroy(it->second.handle); it->second.handle = 0; it->second.dropped = true; ++report_.bulk_dropped;
                }
            } else {
                auto it = partial_.find(e.stream);
                if (it == partial_.end()) continue;
                if (e.kind == BulkKind::End && !it->second.dropped) { completed_.push_back(it->second.handle); ++report_.bulk_tensors_received; }
                else if (it->second.handle) nodus_tensor_destroy(it->second.handle);
                partial_.erase(it);
            }
        }
    }

    std::mutex mutex_;
    uint64_t maximum_;
    std::unique_ptr<LinkListener> listener_;
    std::unique_ptr<LinkDialer> dialer_;
    std::unique_ptr<Link> link_;
    std::deque<Out> out_;
    std::unordered_map<uint64_t, Partial> partial_;
    std::deque<uint64_t> completed_;
    EndpointReport report_;
};

namespace {

std::mutex& endpoints_mutex() { static std::mutex m; return m; }
std::unordered_map<std::string, std::shared_ptr<LinkEndpoint>>& endpoints() {
    static std::unordered_map<std::string, std::shared_ptr<LinkEndpoint>> e; return e;
}

class LinkRow final : public ITool {
public:
    LinkRow(std::string id, bool bulk, std::shared_ptr<LinkEndpoint> ep) : id_(std::move(id)), bulk_(bulk), ep_(std::move(ep)) {}
    const char* id_cstr() const noexcept override { return id_.c_str(); }
    const char* name_cstr() const noexcept override { return bulk_ ? "Link bulk" : "Link control"; }
    ToolCaps caps() const override { return ToolCaps::None; }
    int32_t port_count() const override { return 2; }
    ToolPortSpec port_spec(int32_t index) const override {
        if (index == 0) return {ToolPortKind::Argument, 1};
        if (index == 1) return {ToolPortKind::Return, 1};
        return {};
    }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override { std::lock_guard<std::mutex> g(ep_->mutex()); ep_->poll(); }
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& context) override {
        context.succeeded = false;
        if (!context.stack.raw) return;
        std::lock_guard<std::mutex> g(ep_->mutex());
        ep_->poll();
        auto& stack = *context.stack.raw;
        bool accepted = false;
        nodus::tensors::AbstractTensorHandle input{};
        if (raw_stack_pop_abstract_tensor(stack, input)) {
            if (!(bulk_ ? accept_bulk(input.id) : accept_control(input.id))) {
                ++(bulk_ ? ep_->report().bulk_refused : ep_->report().control_refused);
                raw_stack_push_abstract_tensor(stack, input); return;
            }
            accepted = true; ep_->poll();
        }
        uint64_t out = bulk_ ? ep_->take_bulk() : control_tensor();
        if (!out) { context.succeeded = accepted; return; }
        if (!raw_stack_push_abstract_tensor(stack, {out})) { nodus_tensor_destroy(out); context.succeeded = accepted; return; }
        context.succeeded = true;
    }

private:
    bool accept_control(uint64_t h) {
        NodusTensorDesc d{}; uint8_t b[kControlPacketBytes]; ControlPacket p;
        if (nodus_tensor_describe(h, &d) != NODUS_OK || d.dtype != NODUS_DTYPE_U8 || d.rank != 1 || d.total_bytes != kControlPacketBytes
            || nodus_tensor_read(h, 0, b, sizeof(b)) != int64_t(sizeof(b)) || !decode_control(b, sizeof(b), p)) return false;
        return ep_->send_control(p);
    }
    bool accept_bulk(uint64_t h) {
        NodusTensorDesc d{};
        if (!ep_->linked() || nodus_tensor_describe(h, &d) != NODUS_OK || !d.total_bytes || d.total_bytes > ep_->maximum()) return false;
        std::vector<uint8_t> bytes(size_t(d.total_bytes));
        if (nodus_tensor_read(h, 0, bytes.data(), bytes.size()) != int64_t(bytes.size())) return false;
        return ep_->queue_bulk(d, std::move(bytes));
    }
    uint64_t control_tensor() {
        ControlPacket p; if (!ep_->receive_control(p)) return 0;
        uint8_t b[kControlPacketBytes]; encode_control(p, b);
        const uint64_t shape[1] = {kControlPacketBytes};
        uint64_t t = nodus_tensor_create(NODUS_DTYPE_U8, 1, shape);
        if (!t) return 0;
        if (nodus_tensor_write(t, 0, b, sizeof(b)) != int64_t(sizeof(b))) { nodus_tensor_destroy(t); return 0; }
        return t;
    }
    std::string id_;
    bool bulk_;
    std::shared_ptr<LinkEndpoint> ep_;
};

bool register_rows(ToolRegistry& registry, const std::string& id, std::shared_ptr<LinkEndpoint> ep, std::string* error) {
    {
        std::lock_guard<std::mutex> g(endpoints_mutex());
        if (endpoints().count(id)) { if (error) *error = "link " + id + " is already registered"; return false; }
    }
    for (bool bulk : {false, true}) {
        ToolRegistry::Entry entry;
        entry.id = id + (bulk ? ".bulk" : ".control"); entry.name = bulk ? "Link bulk" : "Link control"; entry.caps = ToolCaps::None;
        entry.factory = [rid = entry.id, bulk, ep] {
            return std::unique_ptr<ITool, std::function<void(ITool*)>>(new LinkRow(rid, bulk, ep), [](ITool* t) { delete t; });
        };
        if (!registry.register_tool(std::move(entry))) {
            if (bulk) registry.unregister_tool(id + ".control");
            if (error) *error = "tool registry refused " + id;
            return false;
        }
    }
    std::lock_guard<std::mutex> g(endpoints_mutex());
    endpoints()[id] = std::move(ep);
    return true;
}

}  // namespace

bool register_link_host(ToolRegistry& registry, LinkHostConfig config, uint16_t* listen_port, std::string* error) {
    if (config.id.empty() || config.prefix.empty() || !config.maximum_bulk_bytes) { if (error) *error = "id, prefix and size are required"; return false; }
    std::shared_ptr<LinkEndpoint> ep;
    std::string id = config.id;
    try { ep = std::make_shared<LinkEndpoint>(std::move(config)); }
    catch (const std::exception& e) { if (error) *error = e.what(); return false; }
    if (listen_port) *listen_port = ep->listen_port();
    return register_rows(registry, id, std::move(ep), error);
}

bool register_link_dialer(ToolRegistry& registry, LinkDialConfig config, std::string* error) {
    if (config.id.empty() || config.target.prefix.empty() || !config.target.port || !config.maximum_bulk_bytes) {
        if (error) *error = "id, prefix, port and size are required"; return false;
    }
    std::shared_ptr<LinkEndpoint> ep;
    std::string id = config.id;
    try { ep = std::make_shared<LinkEndpoint>(std::move(config)); }
    catch (const std::exception& e) { if (error) *error = e.what(); return false; }
    return register_rows(registry, id, std::move(ep), error);
}

bool link_endpoint_report(const std::string& id, EndpointReport* out) {
    std::shared_ptr<LinkEndpoint> ep;
    {
        std::lock_guard<std::mutex> g(endpoints_mutex());
        auto it = endpoints().find(id); if (it == endpoints().end()) return false;
        ep = it->second;
    }
    std::lock_guard<std::mutex> g(ep->mutex());
    if (out) *out = ep->report();
    return true;
}

void unregister_link(ToolRegistry& registry, const std::string& id) {
    registry.unregister_tool(id + ".control"); registry.unregister_tool(id + ".bulk");
    std::lock_guard<std::mutex> g(endpoints_mutex());
    endpoints().erase(id);
}

}  // namespace nodus::link
