#pragma once
// The Nodus link on a canvas: two rows, modelled on TcpByteStreamTool, sharing one link endpoint.
//   <id>.control   Argument = control.out, row port "received" = control.in   U8 [256] tensors, one ControlPacket each
//   <id>.bulk      Argument = bulk.out,    row port "received" = bulk.in      any dense tensor; it arrives with the
//                                                                             sender's dtype and shape
// The endpoint is a host (a LinkListener on 127.0.0.1, one peer at a time) or a dialer (a LinkDialer, one dial: tickets
// are single use). Both rows poll it; nothing blocks. Inputs are copied when accepted and stay the caller's; outputs
// belong to whoever pops them. A refused input is restored to the stack with succeeded=false and the rows below it run.
// Research case 98d49cb8520e44d092c5d134c37c3c32, stage 2. Include before <windows.h> (winsock).
#include "nodus_link.h"
#include "nodus_link_wss.h"
#include "tool_registry.h"

#include <cstdint>
#include <memory>
#include <string>

namespace nodus::link {

struct LinkHostConfig {
    std::string id;                 // rows are <id>.control and <id>.bulk
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
    TicketBytes control_ticket{}, bulk_ticket{};
    uint64_t maximum_bulk_bytes = uint64_t(256) << 20;
};

class LinkEndpoint;   // the shared endpoint (module_library/source/tools/nodus_link_endpoint.h)

// The row: <id>.control (bulk=false) or <id>.bulk (bulk=true). Source: module_library/source/tools/nodus_link.cpp.
// Saved state is the id and which row it is. The endpoint is process state, handed over by the registration; a row
// made with no endpoint (the no-argument constructor, or a canvas-compiled copy) refuses in execute_stack.
class LinkRow final : public ITool {
public:
    LinkRow(std::string id, bool bulk, std::shared_ptr<LinkEndpoint> ep);
    LinkRow();
    void configure(std::string id, bool bulk);
    bool serialize(OutputArchive& out) const override;
    bool deserialize(InputArchive& in) override;
    const char* id_cstr() const noexcept override { return id_.c_str(); }
    const char* name_cstr() const noexcept override { return bulk_ ? "Link bulk" : "Link control"; }
    ToolCaps caps() const override { return ToolCaps::None; }
    int32_t port_count() const override { return 1; }
    ToolPortSpec port_spec(int32_t index) const override {
        if (index == 0) return {ToolPortKind::Argument, 1};
        return {};
    }
    void initialize(const ToolInitContext&) override;
    void shutdown() override {}
    void tick(double, HostAPI&) override;
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& context) override;
    // What arrives goes out on the row's own port "received" (8-byte handle tokens), never onto the stack: below the
    // row the stack holds only what it refused (owner, 2026-11-19). An arrival waits in the endpoint until that port's
    // edge can take it.
    int32_t row_port_count() const override { return 1; }
    ToolRowPort row_port(int32_t index) const override {
        if (index == 0) return ToolRowPort{ToolPortKind::Return, 1.0f, 0.5f, "received", 8};
        return ToolRowPort{};
    }
    // It copies what it sends and never keeps or destroys it (tool_api.h ToolTensorRole): in a table with a Returns
    // row a sent tensor goes back to the mint, e.g. a volunteer cycle batch once it is on the wire.
    ToolTensorRole tensor_role() const override { return ToolTensorRole::Borrower; }

private:
    bool accept_control(uint64_t h);
    bool accept_bulk(uint64_t h);
    uint64_t control_tensor();
    std::string id_;
    bool bulk_ = false;
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

// Registers <id>.control and <id>.bulk; the listener is open on return (port in report). False on refusal.
bool register_link_host(ToolRegistry& registry, LinkHostConfig config, uint16_t* listen_port = nullptr, std::string* error = nullptr);
bool register_link_dialer(ToolRegistry& registry, LinkDialConfig config, std::string* error = nullptr);
// State and counters of a registered endpoint (by the id given at registration).
bool link_endpoint_report(const std::string& id, EndpointReport* out);
// Forget the endpoint and unregister both rows; live rows keep their endpoint until they are destroyed.
void unregister_link(ToolRegistry& registry, const std::string& id);

}  // namespace nodus::link
