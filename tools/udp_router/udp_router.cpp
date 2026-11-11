// udp_router: the service's UDP router. It holds every outside-facing UDP socket and talks to the server's own
// services over local connections only (a TCP listener on 127.0.0.1). It decides nothing: which ports are open and
// who may send are the business of the service that connects.
//
// Greedy: each round it drains everything that has arrived on every socket and gives each destination ONE
// conglomerate record holding all of it, in arrival order. A service never receives a datagram at a time.
//
// Local frames, little-endian, both directions:   u8 op | u16 port | u32 ipv4 | u16 peer_port | u32 length | payload
//   in   1 OPEN    bind a UDP socket on <bind-host>:<port>; the connection that opens a port owns it
//   in   2 CLOSE   close it
//   in   3 SEND    send payload from <port> to ipv4:peer_port
//   out  5 NOTE    payload is a line of text about <port> (a bind that failed, a send that failed)
//   out  6 READY   sent once to a new connection; payload is the bind host
//   out  7 OPENED  the socket on <port> is bound and listening
//   out  8 BATCH   the conglomerate record for destination <port>:
//                  u32 count, then count times: u32 ipv4 | u16 peer_port | u16 length | payload
//
// Usage: udp_router <bind-host> <local-control-port>     e.g. udp_router 0.0.0.0 47699
// A connection that closes gives up its ports. The control listener is bound to 127.0.0.1 and nowhere else.

#define FD_SETSIZE 1024
#include <winsock2.h>
#include <ws2tcpip.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

const uint32_t kMaxFrame = 2u * 1024u * 1024u;
const size_t kMaxRecord = 1024 * 1024;      // one conglomerate record is at most this large; the rest waits a round
const size_t kHead = 13;

struct Client {
    SOCKET s = INVALID_SOCKET;
    std::vector<char> in;
};
struct Port {
    SOCKET s = INVALID_SOCKET;
    SOCKET owner = INVALID_SOCKET;
    std::vector<char> record;                // the conglomerate being gathered this round, after its 4-byte count
    uint32_t count = 0;
};

std::map<SOCKET, Client> g_clients;
std::map<uint16_t, Port> g_ports;
in_addr g_host{};

bool send_all(SOCKET s, const char* data, size_t length) {
    while (length) {
        int sent = send(s, data, static_cast<int>(length > 65536 ? 65536 : length), 0);
        if (sent <= 0) return false;
        data += sent;
        length -= static_cast<size_t>(sent);
    }
    return true;
}

bool emit(SOCKET client, uint8_t op, uint16_t port, const char* data, uint32_t length) {
    char head[kHead] = {0};
    head[0] = static_cast<char>(op);
    std::memcpy(head + 1, &port, 2);
    std::memcpy(head + 9, &length, 4);
    return send_all(client, head, kHead) && (length == 0 || send_all(client, data, length));
}

void note(SOCKET client, uint16_t port, const std::string& text) {
    emit(client, 5, port, text.data(), static_cast<uint32_t>(text.size()));
}

void close_port(uint16_t port) {
    auto found = g_ports.find(port);
    if (found == g_ports.end()) return;
    closesocket(found->second.s);
    g_ports.erase(found);
}

void open_port(SOCKET client, uint16_t port) {
    auto found = g_ports.find(port);
    if (found != g_ports.end()) {
        if (found->second.owner == client) emit(client, 7, port, nullptr, 0);
        else note(client, port, "open refused: another service holds this port");
        return;
    }
    SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == INVALID_SOCKET) { note(client, port, "socket failed"); return; }
    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_addr = g_host;
    local.sin_port = htons(port);
    if (bind(s, reinterpret_cast<sockaddr*>(&local), sizeof(local)) != 0) {
        note(client, port, "bind failed, error " + std::to_string(WSAGetLastError()));
        closesocket(s);
        return;
    }
    u_long nonblocking = 1;
    ioctlsocket(s, FIONBIO, &nonblocking);
    BOOL off = FALSE;                        // a peer's ICMP "port unreachable" must not break the next receive
    DWORD returned = 0;
    WSAIoctl(s, _WSAIOW(IOC_VENDOR, 12), &off, sizeof(off), nullptr, 0, &returned, nullptr, nullptr);
    int buffer = 4 * 1024 * 1024;            // room for a burst between two rounds
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&buffer), sizeof(buffer));
    Port made;
    made.s = s;
    made.owner = client;
    g_ports[port] = made;
    emit(client, 7, port, nullptr, 0);
}

void drop_client(SOCKET client) {
    std::vector<uint16_t> owned;
    for (const auto& entry : g_ports) if (entry.second.owner == client) owned.push_back(entry.first);
    for (uint16_t port : owned) close_port(port);
    closesocket(client);
    g_clients.erase(client);
}

// One frame from a service. Returns false when the connection must be dropped.
bool handle(SOCKET client, const char* head, const char* payload, uint32_t length) {
    uint16_t port, peer_port;
    uint32_t ip;
    std::memcpy(&port, head + 1, 2);
    std::memcpy(&ip, head + 3, 4);
    std::memcpy(&peer_port, head + 7, 2);
    const uint8_t op = static_cast<uint8_t>(head[0]);
    if (op == 1) { open_port(client, port); return true; }
    auto found = g_ports.find(port);
    const bool mine = found != g_ports.end() && found->second.owner == client;
    if (op == 2) { if (mine) close_port(port); return true; }
    if (op != 3) return false;
    if (!mine) { note(client, port, "send refused: this connection does not hold the port"); return true; }
    sockaddr_in peer{};
    peer.sin_family = AF_INET;
    peer.sin_addr.s_addr = ip;               // already in network order
    peer.sin_port = htons(peer_port);
    if (sendto(found->second.s, payload, static_cast<int>(length), 0, reinterpret_cast<sockaddr*>(&peer), sizeof(peer)) == SOCKET_ERROR)
        note(client, port, "send failed, error " + std::to_string(WSAGetLastError()));
    return true;
}

bool read_client(Client& client) {
    char chunk[65536];
    int got = recv(client.s, chunk, sizeof(chunk), 0);
    if (got <= 0) return false;
    client.in.insert(client.in.end(), chunk, chunk + got);
    size_t at = 0;
    while (client.in.size() - at >= kHead) {
        uint32_t length;
        std::memcpy(&length, client.in.data() + at + 9, 4);
        if (length > kMaxFrame) return false;
        if (client.in.size() - at < kHead + length) break;
        if (!handle(client.s, client.in.data() + at, client.in.data() + at + kHead, length)) return false;
        at += kHead + length;
    }
    client.in.erase(client.in.begin(), client.in.begin() + static_cast<std::ptrdiff_t>(at));
    return true;
}

// Take everything that has arrived on one socket into its destination's conglomerate record.
void drain(Port& port, std::vector<char>& scratch) {
    for (;;) {
        if (port.record.size() >= kMaxRecord) return;
        sockaddr_in peer{};
        int size = sizeof(peer);
        int got = recvfrom(port.s, scratch.data(), static_cast<int>(scratch.size()), 0, reinterpret_cast<sockaddr*>(&peer), &size);
        if (got < 0) return;
        const uint32_t ip = peer.sin_addr.s_addr;
        const uint16_t peer_port = ntohs(peer.sin_port);
        const uint16_t length = static_cast<uint16_t>(got);
        const size_t at = port.record.size();
        port.record.resize(at + 8 + length);
        std::memcpy(port.record.data() + at, &ip, 4);
        std::memcpy(port.record.data() + at + 4, &peer_port, 2);
        std::memcpy(port.record.data() + at + 6, &length, 2);
        std::memcpy(port.record.data() + at + 8, scratch.data(), length);
        ++port.count;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: udp_router <bind-host> <local-control-port>\n"); return 2; }
    WSADATA data;
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 2;
    if (inet_pton(AF_INET, argv[1], &g_host) != 1) return 2;
    const int control = std::atoi(argv[2]);
    SOCKET listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_addr.s_addr = htonl(INADDR_LOOPBACK);          // local connections only, whatever the bind host is
    local.sin_port = htons(static_cast<uint16_t>(control));
    if (listener == INVALID_SOCKET || bind(listener, reinterpret_cast<sockaddr*>(&local), sizeof(local)) != 0 || listen(listener, 16) != 0) {
        std::fprintf(stderr, "udp_router: cannot listen on 127.0.0.1:%d, error %d\n", control, WSAGetLastError());
        return 2;
    }
    std::printf("udp_router: UDP on %s, local services on 127.0.0.1:%d\n", argv[1], control);
    std::fflush(stdout);
    std::vector<char> scratch(65535);
    for (;;) {
        fd_set readable;
        FD_ZERO(&readable);
        FD_SET(listener, &readable);
        for (const auto& entry : g_clients) FD_SET(entry.first, &readable);
        for (const auto& entry : g_ports) FD_SET(entry.second.s, &readable);
        timeval wait{0, 20000};
        if (select(0, &readable, nullptr, nullptr, &wait) <= 0) continue;
        if (FD_ISSET(listener, &readable)) {
            SOCKET accepted = accept(listener, nullptr, nullptr);
            if (accepted != INVALID_SOCKET) {
                DWORD timeout = 2000;        // a service that stops reading is dropped, not waited on
                setsockopt(accepted, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
                BOOL nodelay = TRUE;
                setsockopt(accepted, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));
                Client made;
                made.s = accepted;
                g_clients[accepted] = made;
                emit(accepted, 6, 0, argv[1], static_cast<uint32_t>(std::strlen(argv[1])));
            }
        }
        std::vector<SOCKET> gone;
        for (auto& entry : g_clients)
            if (FD_ISSET(entry.first, &readable) && !read_client(entry.second)) gone.push_back(entry.first);
        for (SOCKET client : gone) drop_client(client);
        gone.clear();
        for (auto& entry : g_ports) {
            Port& port = entry.second;
            if (!FD_ISSET(port.s, &readable)) continue;
            port.record.assign(4, 0);
            port.count = 0;
            drain(port, scratch);
            if (!port.count) continue;
            std::memcpy(port.record.data(), &port.count, 4);
            if (!emit(port.owner, 8, entry.first, port.record.data(), static_cast<uint32_t>(port.record.size())))
                gone.push_back(port.owner);
        }
        for (SOCKET client : gone) if (g_clients.count(client)) drop_client(client);
    }
}
