#pragma once
// The Nodus link on a canvas: two rows, modelled on TcpByteStreamTool, sharing one link endpoint.
//   <id>.control   Argument = control.out, Return = control.in   U8 [256] tensors, one encoded ControlPacket each
//   <id>.bulk      Argument = bulk.out,    Return = bulk.in      any dense tensor; it arrives with the sender's dtype
//                                                                and shape (nodus_tensor_create)
// The endpoint is a host (a LinkListener on 127.0.0.1, one peer at a time) or a dialer (a LinkDialer, one dial: tickets
// are single use). Both rows poll it; nothing blocks. Inputs are copied when accepted and stay the caller's; outputs
// belong to whoever pops them. A refused input is restored to the stack with succeeded=false, as TcpByteStreamTool does.
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
