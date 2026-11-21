#pragma once
// The Nodus link on a canvas: four rows, one per FIFO, sharing one link endpoint. Each row has one duty (owner,
// 2026-11-21: "rows should not be doing double duty, if you want double duty bake a plugin from a table with two rows").
//   <id>.control.send     pops a U8 [256] tensor (one ControlPacket) off the stack and sends it
//   <id>.control.receive  pushes each arrived ControlPacket onto the stack, a U8 [256] tensor
//   <id>.bulk.send        pops any dense tensor off the stack and sends it
//   <id>.bulk.receive     pushes each arrived tensor onto the stack, with the sender's dtype and shape
// Arrivals go onto the stack (owner, 2026-11-21: "at no point did I ever say arrival should be redirected by a port
// instead of being fed into the stack"); the table's Output rows carry them on. Put sending and receiving rows in
// separate tables, so a refused send waiting on a stack is never mistaken for an arrival.
// The endpoint is a host (a LinkListener on 127.0.0.1, one peer at a time) or a dialer (a LinkDialer, one dial: tickets
// are single use). Every row polls it; nothing blocks. A sent input is copied when accepted and stays the caller's; an
// arrival belongs to whoever pops it. A refused input is restored to the stack with succeeded=false and the rows below
// it run. An arrival waits in the endpoint while the stack has no room for it.
// Research case 98d49cb8520e44d092c5d134c37c3c32, stage 2. Include before <windows.h> (winsock).
#include "nodus_link.h"
#include "nodus_link_wss.h"
#include "nodus_link_udp.h"
#include "tool_registry.h"

#include <cstdint>
#include <memory>
#include <string>

namespace nodus::link {

struct LinkHostConfig {
    std::string id;                 // rows are <id>.control.send/.receive and <id>.bulk.send/.receive
    std::string prefix;             // /ws/link/<prefix>/{control,bulk}
    uint16_t port = 0;              // 0 = ephemeral
    Key key{};                      // ticket key of the issuing service
    uint64_t maximum_bulk_bytes = uint64_t(256) << 20;
};
struct LinkDialConfig {
    std::string id;
    DialTarget target;              // the loopback passage (LinkDialer), used when `wss` is false
    bool wss = false;               // dial with WinHTTP instead (WssDialer): wss:// or ws://, a name or an address
    WssTarget wss_target;
    bool udp = false;               // rendezvous at a broker, then direct UDP (nodus_link_udp.h); `target`, `wss_target`
    UdpTarget udp_target;           // and the tickets are unused
    TicketBytes control_ticket{}, bulk_ticket{};
    uint64_t maximum_bulk_bytes = uint64_t(256) << 20;
};

class LinkEndpoint;   // the shared endpoint (module_library/source/tools/nodus_link_endpoint.h)

// Which of the four rows a LinkRow is.
enum class LinkRowRole : uint32_t { ControlSend = 0, ControlReceive = 1, BulkSend = 2, BulkReceive = 3 };
inline const char* link_row_suffix(LinkRowRole r) {
    switch (r) {
        case LinkRowRole::ControlSend: return ".control.send";
        case LinkRowRole::ControlReceive: return ".control.receive";
        case LinkRowRole::BulkSend: return ".bulk.send";
        default: return ".bulk.receive";
    }
}

// One row of the four. Source: module_library/source/tools/nodus_link.cpp. Saved state is the link id and the row's
// role. The endpoint is process state, handed over by the registration; a row made with no endpoint (the no-argument
// constructor, or a canvas-compiled copy) refuses in execute_stack.
class LinkRow final : public ITool {
public:
    LinkRow(std::string id, LinkRowRole role, std::shared_ptr<LinkEndpoint> ep);
    LinkRow();
    void configure(std::string id, LinkRowRole role);
    bool serialize(OutputArchive& out) const override;
    bool deserialize(InputArchive& in) override;
    const char* id_cstr() const noexcept override { return id_.c_str(); }
    const char* name_cstr() const noexcept override;
    ToolCaps caps() const override { return ToolCaps::None; }
    void initialize(const ToolInitContext& ctx) override;
    void shutdown() override {}
    void tick(double, HostAPI&) override;
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& context) override;
    int32_t port_count() const override { return 1; }
    ToolPortSpec port_spec(int32_t index) const override {
        if (index == 0) return {sends() ? ToolPortKind::Argument : ToolPortKind::Return, 1};
        return {};
    }
    // A sending row copies what it sends and never keeps or destroys it (tool_api.h ToolTensorRole): in a table with a
    // Returns row a sent tensor goes back to the mint, e.g. a volunteer cycle batch once it is on the wire. A receiving
    // row pops nothing.
    ToolTensorRole tensor_role() const override { return sends() ? ToolTensorRole::Borrower : ToolTensorRole::Owner; }

private:
    bool sends() const { return role_ == LinkRowRole::ControlSend || role_ == LinkRowRole::BulkSend; }
    bool bulk() const { return role_ == LinkRowRole::BulkSend || role_ == LinkRowRole::BulkReceive; }
    void send(RawStackFrame& stack, bool& accepted);
    void receive(RawStackFrame& stack);
    bool accept_control(uint64_t h);
    bool accept_bulk(uint64_t h);
    uint64_t control_tensor();
    std::string id_;
    LinkRowRole role_ = LinkRowRole::ControlSend;
    std::shared_ptr<LinkEndpoint> ep_;
};

enum class EndpointState : int32_t { Waiting = 0, Linked = 1, Closed = 2, Failed = 3 };
struct EndpointReport {
    EndpointState state = EndpointState::Waiting;
    uint16_t listen_port = 0;
    uint64_t account = 0;
    LinkStats link;                         // of the current (or last) link
    uint64_t bulk_tensors_sent = 0, bulk_tensors_received = 0;
    uint64_t bulk_refused = 0;              // inputs restored: not linked, out queue full, too large, unreadable
    uint64_t control_refused = 0;           // inputs restored: not linked, not one packet, FIFO full
    uint64_t bulk_dropped = 0;              // arrivals that could not become a tensor (descriptor, create, size)
    uint64_t peers_refused = 0;             // a second peer while one is linked
    std::string reason;                     // why a wss dial failed (WinHTTP, TLS, pin), when it did
};

// The 16 counters of nodus_headless_link_report, in its order (one definition: the report and the metrics row use it).
inline void link_report_counters(const EndpointReport& r, uint64_t out[16]) {
    const uint64_t v[16] = {r.listen_port, r.account, r.link.control_sent, r.link.control_received,
        r.link.bulk_bytes_sent, r.link.bulk_bytes_received, r.bulk_tensors_sent, r.bulk_tensors_received,
        r.control_refused, r.bulk_refused, r.bulk_dropped, r.peers_refused, r.link.control_read_stalls,
        r.link.bulk_read_stalls, r.link.protocol_violations, r.link.control_refused_full + r.link.bulk_refused_full};
    for (int i = 0; i < 16; ++i) out[i] = v[i];
}

// Registers <id>.control and <id>.bulk, and <id>.metrics (the metrics row bound to this link; also the generic
// "nodus.link.metrics", whose link comes from saved state; module_library/source/tools/link_metrics.cpp); the listener is open on return (port in report). False on refusal.
bool register_link_host(ToolRegistry& registry, LinkHostConfig config, uint16_t* listen_port = nullptr, std::string* error = nullptr);
bool register_link_dialer(ToolRegistry& registry, LinkDialConfig config, std::string* error = nullptr);
// State and counters of a registered endpoint (by the id given at registration).
bool link_endpoint_report(const std::string& id, EndpointReport* out);
// The meta-edge channel group the NEXT "<id>.metrics" row created for this link starts with (0 = none, the default):
// the row then appends the meta-edge observations of that channel to its samples (module_library/source/tools/link_metrics.cpp).
// A row already created keeps its own (saved state); a canvas-loaded copy takes it from its saved state.
void set_link_metrics_channel(const std::string& id, int32_t channel);
// Forget the endpoint and unregister both rows; live rows keep their endpoint until they are destroyed.
void unregister_link(ToolRegistry& registry, const std::string& id);

}  // namespace nodus::link
