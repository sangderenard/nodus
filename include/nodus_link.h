#pragma once
// Nodus link: one authenticated connection between two Nodus instances carrying FOUR FIFOs (owner, 2026-11-18:
// "Four FIFOs, 2 pairs"):
//   control.out / control.in   fixed 256-byte packets: instruction, API, metric, receipt
//   bulk.out    / bulk.in      large streams, sent as Begin / Chunk ... / End records
// Each pair is its own WebSocket (RFC 6455) on its own TCP connection, so a stalled bulk stream never holds back
// control. Research case 98d49cb8520e44d092c5d134c37c3c32 (ticket bc214e177933475e9f8ac3f5ed3e3762), stage 1.
//
// Passage: the public side is wss://<host>/ws/link/<prefix>/{control,bulk}. Apache terminates TLS and proxies to a
// LinkListener, which binds 127.0.0.1 ONLY (same stance as contract_listener.h). The first binary message on each
// socket must be a 128-byte link ticket signed HMAC-SHA256 with the key the issuing service holds (stage 3 issues
// tickets from an existing peer grant). Both sockets of a link carry tickets with the same link id, account and role,
// and only then is a Link handed out. A ticket is accepted once per channel.
//
// Nonblocking, bounded work per poll(); every queue is bounded and refuses when full; when an incoming queue is full
// the socket is simply not read (TCP backpressure), counted in LinkStats. Nothing blocks: listener, dialer and
// link are all advanced by poll(). OS calls live in the one platform region below: Windows CNG (bcrypt.lib) + ws2_32; POSIX OpenSSL libcrypto + BSD sockets.
#include "contract_listener.h"   // ensure_sockets, WireSocket socket helpers
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ================================ platform region: Windows (CNG, Winsock) / POSIX (OpenSSL, BSD sockets) ================================
// Every OS call of this file is here; the rest of the file is platform-free.
#if defined(_WIN32)
#include <ws2tcpip.h>
#include <windows.h>
#include <bcrypt.h>
namespace nodus::link {
namespace detail {   // CNG providers; nodus_link_udp.h's Aead uses detail::provider too
inline BCRYPT_ALG_HANDLE provider(LPCWSTR alg, bool hmac) {
    BCRYPT_ALG_HANDLE h = nullptr;
    if (!BCRYPT_SUCCESS(BCryptOpenAlgorithmProvider(&h, alg, nullptr, hmac ? BCRYPT_ALG_HANDLE_HMAC_FLAG : 0)))
        throw std::runtime_error("CNG provider refused");
    return h;   // process lifetime
}
inline void hash(BCRYPT_ALG_HANDLE h, const uint8_t* key, size_t key_n, const uint8_t* data, size_t n, uint8_t* out, size_t out_n) {
    if (!BCRYPT_SUCCESS(BCryptHash(h, const_cast<PUCHAR>(key), ULONG(key_n), const_cast<PUCHAR>(data), ULONG(n), out, ULONG(out_n))))
        throw std::runtime_error("CNG hash refused");
}
}  // namespace detail
namespace platform {
using socklen = int;
constexpr int kSendFlags = 0;
// After a nonblocking connect() reported failure: true when it is merely in progress.
inline bool connect_would_block() { return WSAGetLastError() == WSAEWOULDBLOCK; }
// Readiness query for a nonblocking connect, never waits: 1 connected, 0 in progress, -1 refused, -2 query failed.
inline int connect_state(SOCKET s) {
    fd_set w, e; FD_ZERO(&w); FD_ZERO(&e); FD_SET(s, &w); FD_SET(s, &e);
    timeval zero{0, 0};
    int n = select(0, nullptr, &w, &e, &zero);           // zero timeout: a readiness query, not a wait
    if (n < 0) return -2;
    if (n == 0) return 0;
    int err = 0; int len = sizeof(err);
    getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&err), &len);
    if (FD_ISSET(s, &e) || err) return -1;
    return 1;
}
inline void sha1(const void* data, size_t n, uint8_t* out) {
    static BCRYPT_ALG_HANDLE h = detail::provider(BCRYPT_SHA1_ALGORITHM, false);
    detail::hash(h, nullptr, 0, static_cast<const uint8_t*>(data), n, out, 20);
}
inline void sha256(const void* data, size_t n, uint8_t* out) {
    static BCRYPT_ALG_HANDLE h = detail::provider(BCRYPT_SHA256_ALGORITHM, false);
    detail::hash(h, nullptr, 0, static_cast<const uint8_t*>(data), n, out, 32);
}
inline void hmac_sha256(const uint8_t* key, size_t key_n, const void* data, size_t n, uint8_t* out) {
    static BCRYPT_ALG_HANDLE h = detail::provider(BCRYPT_SHA256_ALGORITHM, true);
    detail::hash(h, key, key_n, static_cast<const uint8_t*>(data), n, out, 32);
}
inline void random_bytes(void* out, size_t n) {
    if (!BCRYPT_SUCCESS(BCryptGenRandom(nullptr, static_cast<PUCHAR>(out), ULONG(n), BCRYPT_USE_SYSTEM_PREFERRED_RNG)))
        throw std::runtime_error("CNG random refused");
}
}  // namespace platform
}  // namespace nodus::link
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <unistd.h>
#include <cerrno>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
namespace nodus::link {
namespace platform {
using socklen = socklen_t;
#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;                 // a closed peer is an error return, not SIGPIPE
#else
constexpr int kSendFlags = 0;
#endif
inline bool connect_would_block() { return errno == EINPROGRESS || errno == EINTR; }
inline int connect_state(SOCKET s) {
    pollfd p{}; p.fd = s; p.events = POLLOUT;
    int n = ::poll(&p, 1, 0);                            // zero timeout: a readiness query, not a wait
    if (n < 0) return errno == EINTR ? 0 : -2;
    if (n == 0) return 0;
    int err = 0; socklen len = sizeof(err);
    getsockopt(s, SOL_SOCKET, SO_ERROR, &err, &len);
    if ((p.revents & (POLLERR | POLLNVAL)) || err) return -1;
    if (p.revents & POLLOUT) return 1;
    return (p.revents & POLLHUP) ? -1 : 0;
}
inline void sha1(const void* data, size_t n, uint8_t* out) {
    unsigned len = 0;
    if (EVP_Digest(data, n, out, &len, EVP_sha1(), nullptr) != 1 || len != 20) throw std::runtime_error("OpenSSL hash refused");
}
inline void sha256(const void* data, size_t n, uint8_t* out) {
    unsigned len = 0;
    if (EVP_Digest(data, n, out, &len, EVP_sha256(), nullptr) != 1 || len != 32) throw std::runtime_error("OpenSSL hash refused");
}
inline void hmac_sha256(const uint8_t* key, size_t key_n, const void* data, size_t n, uint8_t* out) {
    unsigned len = 0;
    if (!HMAC(EVP_sha256(), key, int(key_n), static_cast<const unsigned char*>(data), n, out, &len) || len != 32)
        throw std::runtime_error("OpenSSL HMAC refused");
}
inline void random_bytes(void* out, size_t n) {
    if (RAND_bytes(static_cast<unsigned char*>(out), int(n)) != 1) throw std::runtime_error("OpenSSL random refused");
}
}  // namespace platform
}  // namespace nodus::link
#endif
// ============================================ end of the platform region ============================================

namespace nodus::link {
using contracts::WireSocket;

// ---- wire constants -------------------------------------------------------------------------------------------------
constexpr size_t kControlPacketBytes = 256, kControlHeaderBytes = 16, kControlPayloadBytes = 240;
constexpr size_t kTicketBytes = 128;
constexpr size_t kBulkHeaderBytes = 40, kBulkChunkBytes = size_t(1) << 20, kBulkDescriptorBytes = 4096;
constexpr size_t kBulkMaxStreams = 16;                         // concurrent streams per direction
constexpr size_t kControlQueuePackets = 1024;                  // each direction
constexpr size_t kBulkWindowBytes = size_t(16) << 20;          // queued bytes each direction
constexpr uint32_t kControlMagic = 0x314B4C4E;  // "NLK1"
constexpr uint32_t kBulkMagic = 0x31424C4E;     // "NLB1"
constexpr uint32_t kTicketMagic = 0x31544C4E;   // "NLT1"
constexpr uint16_t kTicketVersion = 1;

enum class Channel : uint8_t { Control = 1, Bulk = 2 };
enum class Role : uint8_t { Host = 1, Dialer = 2 };
// Contract = one 72-byte contract-negotiator frame (contract_socket.h WireSocket encoding) in the payload: the work
// market carried over the link (volunteer case a11b5c9b1be74c9b9fb684790f7fd5a3, stage 1).
enum class ControlKind : uint16_t { Instruction = 1, Api = 2, Metric = 3, Receipt = 4, Contract = 5 };
enum class BulkKind : uint16_t { Begin = 1, Chunk = 2, End = 3, Abort = 4 };

using Key = std::array<uint8_t, 32>;
using LinkId = std::array<uint8_t, 16>;
using TicketBytes = std::array<uint8_t, kTicketBytes>;

inline void put_le(uint8_t* p, uint64_t v, size_t n) { for (size_t i = 0; i < n; ++i) p[i] = uint8_t(v >> (8 * i)); }
inline uint64_t get_le(const uint8_t* p, size_t n) { uint64_t v = 0; for (size_t i = 0; i < n; ++i) v |= uint64_t(p[i]) << (8 * i); return v; }
inline const char* channel_name(Channel c) { return c == Channel::Control ? "control" : "bulk"; }

// ---- crypto (the platform region holds the provider) ----------------------------------------------------------------------
inline std::array<uint8_t, 20> sha1(const void* data, size_t n) { std::array<uint8_t, 20> out{}; platform::sha1(data, n, out.data()); return out; }
inline std::array<uint8_t, 32> sha256(const void* data, size_t n) { std::array<uint8_t, 32> out{}; platform::sha256(data, n, out.data()); return out; }
inline std::array<uint8_t, 32> hmac_sha256(const Key& key, const void* data, size_t n) {
    std::array<uint8_t, 32> out{}; platform::hmac_sha256(key.data(), key.size(), data, n, out.data()); return out;
}
inline void random_bytes(void* out, size_t n) { platform::random_bytes(out, n); }
inline bool equal_constant_time(const uint8_t* a, const uint8_t* b, size_t n) {
    uint8_t d = 0; for (size_t i = 0; i < n; ++i) d |= uint8_t(a[i] ^ b[i]); return d == 0;
}
inline std::string base64(const uint8_t* p, size_t n) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string s; s.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = uint32_t(p[i]) << 16 | (i + 1 < n ? uint32_t(p[i + 1]) << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        s += t[v >> 18 & 63]; s += t[v >> 12 & 63]; s += i + 1 < n ? t[v >> 6 & 63] : '='; s += i + 2 < n ? t[v & 63] : '=';
    }
    return s;
}
inline std::string websocket_accept(const std::string& key) {
    std::string k = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    auto d = sha1(k.data(), k.size()); return base64(d.data(), d.size());
}

// ---- ticket ---------------------------------------------------------------------------------------------------------
// 128 bytes, little-endian: magic u32 | version u16 | channel u8 | role u8 | link id [16] | sha256(prefix) [32] |
// account u64 | expires_at u64 | issued_at u64 | nonce [16] | HMAC-SHA256(key, bytes 0..95) [32].
// The account is the service SQL account the issuing grant names; times are unix seconds.
struct Ticket {
    Channel channel = Channel::Control;
    Role role = Role::Dialer;
    LinkId link_id{};
    std::array<uint8_t, 32> prefix_hash{};
    uint64_t account = 0, expires_at = 0, issued_at = 0;
    std::array<uint8_t, 16> nonce{};
};
inline std::array<uint8_t, 32> prefix_hash(const std::string& prefix) { return sha256(prefix.data(), prefix.size()); }
inline LinkId new_link_id() { LinkId id{}; random_bytes(id.data(), id.size()); return id; }
inline void encode_ticket_body(const Ticket& t, uint8_t* b) {
    put_le(b, kTicketMagic, 4); put_le(b + 4, kTicketVersion, 2); b[6] = uint8_t(t.channel); b[7] = uint8_t(t.role);
    std::memcpy(b + 8, t.link_id.data(), 16); std::memcpy(b + 24, t.prefix_hash.data(), 32);
    put_le(b + 56, t.account, 8); put_le(b + 64, t.expires_at, 8); put_le(b + 72, t.issued_at, 8);
    std::memcpy(b + 80, t.nonce.data(), 16);
}
inline TicketBytes sign_ticket(const Ticket& t, const Key& key) {
    TicketBytes b{}; encode_ticket_body(t, b.data());
    auto mac = hmac_sha256(key, b.data(), 96); std::memcpy(b.data() + 96, mac.data(), 32); return b;
}
// The caller's ticket for one channel of one link (issued_at = now, a fresh nonce).
inline Ticket make_ticket(Channel channel, Role role, const LinkId& id, const std::string& prefix, uint64_t account, uint64_t now, uint64_t life) {
    Ticket t; t.channel = channel; t.role = role; t.link_id = id; t.prefix_hash = prefix_hash(prefix);
    t.account = account; t.issued_at = now; t.expires_at = now + life; random_bytes(t.nonce.data(), t.nonce.size()); return t;
}
// Verifies magic, version, size and MAC; time, prefix and replay are the listener's checks.
inline bool parse_ticket(const uint8_t* b, size_t n, const Key& key, Ticket& t) {
    if (n != kTicketBytes || get_le(b, 4) != kTicketMagic || get_le(b + 4, 2) != kTicketVersion) return false;
    auto mac = hmac_sha256(key, b, 96);
    if (!equal_constant_time(mac.data(), b + 96, 32)) return false;
    if (b[6] != uint8_t(Channel::Control) && b[6] != uint8_t(Channel::Bulk)) return false;
    if (b[7] != uint8_t(Role::Host) && b[7] != uint8_t(Role::Dialer)) return false;
    t.channel = Channel(b[6]); t.role = Role(b[7]);
    std::memcpy(t.link_id.data(), b + 8, 16); std::memcpy(t.prefix_hash.data(), b + 24, 32);
    t.account = get_le(b + 56, 8); t.expires_at = get_le(b + 64, 8); t.issued_at = get_le(b + 72, 8);
    std::memcpy(t.nonce.data(), b + 80, 16); return true;
}

// ---- control packet -------------------------------------------------------------------------------------------------
// 256 bytes: magic u32 | kind u16 | flags u16 | seq u32 | length u32 (<= 240) | payload [240].
struct ControlPacket {
    uint16_t kind = 0, flags = 0;
    uint32_t seq = 0, length = 0;
    std::array<uint8_t, kControlPayloadBytes> payload{};
};
inline void encode_control(const ControlPacket& p, uint8_t* b) {
    put_le(b, kControlMagic, 4); put_le(b + 4, p.kind, 2); put_le(b + 6, p.flags, 2); put_le(b + 8, p.seq, 4); put_le(b + 12, p.length, 4);
    std::memcpy(b + kControlHeaderBytes, p.payload.data(), kControlPayloadBytes);
}
inline bool decode_control(const uint8_t* b, size_t n, ControlPacket& p) {
    if (n != kControlPacketBytes || get_le(b, 4) != kControlMagic) return false;
    p.kind = uint16_t(get_le(b + 4, 2)); p.flags = uint16_t(get_le(b + 6, 2)); p.seq = uint32_t(get_le(b + 8, 4)); p.length = uint32_t(get_le(b + 12, 4));
    if (p.length > kControlPayloadBytes) return false;
    std::memcpy(p.payload.data(), b + kControlHeaderBytes, kControlPayloadBytes); return true;
}

// ---- bulk record ----------------------------------------------------------------------------------------------------
// 40-byte header: magic u32 | kind u16 | flags u16 | stream u64 | offset u64 | total u64 | length u32 | reserved u32,
// then `length` bytes: the descriptor (Begin, <= 4096), a chunk (Chunk, 1..1 MiB at `offset`), nothing (End, Abort).
struct BulkEvent {
    BulkKind kind = BulkKind::Begin;
    uint64_t stream = 0, offset = 0, total = 0;
    std::vector<uint8_t> data;   // ownership passes to the receiver
};
inline void encode_bulk_header(uint8_t* b, BulkKind kind, uint64_t stream, uint64_t offset, uint64_t total, uint32_t length) {
    put_le(b, kBulkMagic, 4); put_le(b + 4, uint16_t(kind), 2); put_le(b + 6, 0, 2); put_le(b + 8, stream, 8);
    put_le(b + 16, offset, 8); put_le(b + 24, total, 8); put_le(b + 32, length, 4); put_le(b + 36, 0, 4);
}

// ---- one message stream: what a Link needs from each of its two sockets ---------------------------------------------
// WsStream (a raw nonblocking socket, the listener side and the loopback dialer) and nodus_link_wss.h's wss:// stream
// (WinHTTP on Windows, OpenSSL on POSIX) implement it. Every call returns at once.
class Stream {
public:
    virtual ~Stream() = default;
    virtual bool closed() const = 0;
    // One binary message gathered from two parts; false when the bounded out queue cannot take it.
    virtual bool queue(const uint8_t* a, size_t an, const uint8_t* b = nullptr, size_t bn = 0) = 0;
    virtual void flush(size_t budget) = 0;
    // True when one complete binary message is in `out`.
    virtual bool next(std::vector<uint8_t>& out, size_t budget) = 0;
    virtual void close(uint16_t code) = 0;
};

// ---- one WebSocket over one nonblocking socket ----------------------------------------------------------------------
// Binary messages only (text is a protocol violation); ping answered, close echoed; fragments reassembled up to
// max_message. The client side masks what it sends and requires unmasked input; the server side the reverse.
class WsStream : public Stream {
public:
    WsStream(SOCKET s, bool client, size_t max_message, size_t out_limit, std::vector<uint8_t> leftover = {})
        : socket_(s), client_(client), max_message_(max_message), out_limit_(out_limit), rx_(std::move(leftover)) {
        if (s == INVALID_SOCKET) throw std::invalid_argument("invalid link socket");
        if (!WireSocket::set_nonblocking(s)) { WireSocket::close_socket(s); throw std::runtime_error("nonblocking socket refused"); }
        // Small frames go at once, as the wss dialer's socket already does (nodus_link_wss.h): with Nagle on, a control
        // packet sent after the handshake waited for the handshake's delayed ACK -- about 40 ms on Linux (Oracle A1,
        // 2026-11-21: five 256-byte packets held 37 ms on loopback).
        { int on = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&on), sizeof(on)); }
        random_bytes(&mask_state_, sizeof(mask_state_)); mask_state_ |= 1;
    }
    ~WsStream() override { WireSocket::close_socket(socket_); }
    WsStream(const WsStream&) = delete;
    WsStream& operator=(const WsStream&) = delete;

    bool closed() const override { return closed_; }
    size_t queued_bytes() const { return out_bytes_; }
    size_t frame_bytes(size_t n) const { return n + 2 + (n >= 65536 ? 8 : n >= 126 ? 2 : 0) + (client_ ? 4 : 0); }
    bool can_queue(size_t n) const { return !closed_ && !closing_ && n <= max_message_ && out_bytes_ + frame_bytes(n) <= out_limit_; }
    // One binary message gathered from two parts (header + body), so a bulk chunk is copied once.
    bool queue(const uint8_t* a, size_t an, const uint8_t* b = nullptr, size_t bn = 0) override {
        if (!can_queue(an + bn)) return false;
        push_frame(2, a, an, b, bn); return true;
    }
    // Best effort close frame (code 1000 normal, 1008 policy, 1002 protocol), then the stream reads as closed.
    void close(uint16_t code) override {
        if (closed_ || closing_) return;
        uint8_t c[2] = {uint8_t(code >> 8), uint8_t(code)};
        push_frame(8, c, 2, nullptr, 0); closing_ = true; flush(size_t(1) << 16); closed_ = true;
    }
    void flush(size_t budget) override {
        while (!closed_ && !out_.empty() && budget) {
            auto& f = out_.front();
            size_t want = std::min(budget, f.size() - sent_);
            long n = io_send(f.data() + sent_, want);
            if (n == kIoAgain) return;
            if (n <= 0) { closed_ = true; return; }
            sent_ += size_t(n); budget -= std::min(budget, size_t(n));
            if (sent_ == f.size()) { out_bytes_ -= f.size(); out_.pop_front(); sent_ = 0; }
        }
    }
    // True when one complete binary message is in `out`. Reads at most `budget` bytes from the socket.
    bool next(std::vector<uint8_t>& out, size_t budget) override {
        if (closed_) return false;
        if (parse(out)) return true;
        std::array<char, 65536> buf;
        while (budget && !closed_) {
            size_t room = max_message_ + 14 + buf.size();
            if (rx_.size() - rx_pos_ >= room) break;                 // a whole frame plus one read is already held
            long n = io_recv(buf.data(), std::min(budget, buf.size()));
            if (n == kIoAgain) break;
            if (n <= 0) { closed_ = true; break; }
            rx_.insert(rx_.end(), buf.data(), buf.data() + n); budget -= std::min(budget, size_t(n));
            if (parse(out)) return true;
        }
        return false;
    }

protected:
    // The byte transport under the framing: bytes moved, kIoAgain when it would block, 0 or kIoError when the peer or the
    // transport is gone. The default is the raw socket; a TLS stream (nodus_link_wss.h) overrides both.
    static constexpr long kIoAgain = -1, kIoError = -2;
    virtual long io_send(const uint8_t* p, size_t n) {
        int r = ::send(socket_, reinterpret_cast<const char*>(p), int(std::min<size_t>(n, 1 << 30)), platform::kSendFlags);
        if (r < 0) return WireSocket::would_block() ? kIoAgain : kIoError;
        return r;
    }
    virtual long io_recv(char* p, size_t n) {
        int r = ::recv(socket_, p, int(n), 0);
        if (r < 0) return WireSocket::would_block() ? kIoAgain : kIoError;
        return r;
    }

private:
    void push_frame(uint8_t op, const uint8_t* a, size_t an, const uint8_t* b, size_t bn) {
        size_t n = an + bn;
        std::vector<uint8_t> f; f.reserve(frame_bytes(n));
        f.push_back(uint8_t(0x80 | op));
        uint8_t m = client_ ? 0x80 : 0;
        if (n < 126) f.push_back(uint8_t(m | n));
        else if (n < 65536) { f.push_back(uint8_t(m | 126)); f.push_back(uint8_t(n >> 8)); f.push_back(uint8_t(n)); }
        else { f.push_back(uint8_t(m | 127)); for (int i = 7; i >= 0; --i) f.push_back(uint8_t(uint64_t(n) >> (8 * i))); }
        uint8_t key[4] = {};
        if (client_) {
            // Masking is not security (TLS is); RFC 6455 asks for an unpredictable key, xorshift seeded from the platform RNG.
            mask_state_ ^= mask_state_ << 13; mask_state_ ^= mask_state_ >> 7; mask_state_ ^= mask_state_ << 17;
            put_le(key, mask_state_, 4); f.insert(f.end(), key, key + 4);
        }
        size_t at = f.size();
        if (an) f.insert(f.end(), a, a + an);
        if (bn) f.insert(f.end(), b, b + bn);
        if (client_) for (size_t i = 0; i < n; ++i) f[at + i] ^= key[i & 3];
        out_bytes_ += f.size(); out_.push_back(std::move(f));
    }
    void fail() { closed_ = true; }
    bool parse(std::vector<uint8_t>& out) {
        for (;;) {
            size_t have = rx_.size() - rx_pos_;
            const uint8_t* p = rx_.data() + rx_pos_;
            if (have < 2) break;
            uint8_t op = p[0] & 0x0F; bool fin = (p[0] & 0x80) != 0, masked = (p[1] & 0x80) != 0;
            if (p[0] & 0x70) { fail(); return false; }
            if (masked == client_) { fail(); return false; }       // server input must be masked, client input must not
            uint64_t len = p[1] & 0x7F; size_t head = 2;
            if (len == 126) { if (have < 4) break; len = uint64_t(p[2]) << 8 | p[3]; head = 4; }
            else if (len == 127) { if (have < 10) break; len = 0; for (int i = 0; i < 8; ++i) len = len << 8 | p[2 + i]; head = 10; }
            bool control = op >= 8;
            if ((control && (len > 125 || !fin)) || len > max_message_) { fail(); return false; }
            size_t mask_at = head; if (masked) head += 4;
            if (have < head + len) break;
            std::vector<uint8_t> payload(p + head, p + head + size_t(len));
            if (masked) for (size_t i = 0; i < payload.size(); ++i) payload[i] ^= p[mask_at + (i & 3)];
            rx_pos_ += head + size_t(len);
            if (rx_pos_ == rx_.size()) { rx_.clear(); rx_pos_ = 0; }
            else if (rx_pos_ > (size_t(1) << 20)) { rx_.erase(rx_.begin(), rx_.begin() + ptrdiff_t(rx_pos_)); rx_pos_ = 0; }
            switch (op) {
            case 0:
                if (!in_fragment_ || frag_.size() + payload.size() > max_message_) { fail(); return false; }
                frag_.insert(frag_.end(), payload.begin(), payload.end());
                if (fin) { out = std::move(frag_); frag_.clear(); in_fragment_ = false; return true; }
                break;
            case 2:
                if (in_fragment_) { fail(); return false; }
                if (fin) { out = std::move(payload); return true; }
                frag_ = std::move(payload); in_fragment_ = true; break;
            case 8: {
                uint16_t code = payload.size() >= 2 ? uint16_t(payload[0] << 8 | payload[1]) : 1000;
                close(code); return false;
            }
            case 9: push_frame(10, payload.data(), payload.size(), nullptr, 0); break;
            case 10: break;
            default: fail(); return false;                          // text and reserved opcodes
            }
        }
        return false;
    }
    SOCKET socket_;
    bool client_;
    size_t max_message_, out_limit_;
    std::vector<uint8_t> rx_; size_t rx_pos_ = 0;
    std::vector<uint8_t> frag_; bool in_fragment_ = false;
    std::deque<std::vector<uint8_t>> out_; size_t out_bytes_ = 0, sent_ = 0;
    uint64_t mask_state_ = 0;
    bool closed_ = false, closing_ = false;
};

// ---- the link: four FIFOs -------------------------------------------------------------------------------------------
struct LinkStats {
    uint64_t control_sent = 0, control_received = 0;
    uint64_t bulk_streams_sent = 0, bulk_streams_received = 0, bulk_bytes_sent = 0, bulk_bytes_received = 0;
    uint64_t control_refused_full = 0, bulk_refused_full = 0;     // a send the out FIFO refused
    uint64_t control_read_stalls = 0, bulk_read_stalls = 0;       // polls that left the socket unread: in FIFO full
    uint64_t protocol_violations = 0;
};
class Link {
public:
    Link(std::unique_ptr<Stream> control, std::unique_ptr<Stream> bulk, Role role, uint64_t account, const LinkId& id)
        : control_(std::move(control)), bulk_(std::move(bulk)), role_(role), account_(account), id_(id) {}
    Role role() const { return role_; }
    uint64_t account() const { return account_; }
    const LinkId& id() const { return id_; }
    const LinkStats& stats() const { return stats_; }
    bool closed() const { return control_->closed() || bulk_->closed(); }
    void close(uint16_t code = 1000) { control_->close(code); bulk_->close(code); }

    // control.out / control.in
    bool send_control(const ControlPacket& p) {
        uint8_t b[kControlPacketBytes]; encode_control(p, b);
        if (!control_->queue(b, sizeof(b))) { ++stats_.control_refused_full; return false; }
        ++stats_.control_sent; return true;
    }
    bool receive_control(ControlPacket& p) {
        if (control_in_.empty()) return false;
        p = control_in_.front(); control_in_.pop_front(); return true;
    }

    // bulk.out: begin (descriptor, total bytes) -> stream id (0 = refused); write as much as the window takes; end.
    uint64_t begin_bulk(const uint8_t* descriptor, size_t n, uint64_t total) {
        if (n > kBulkDescriptorBytes || out_streams_.size() >= kBulkMaxStreams || closed()) { ++stats_.bulk_refused_full; return 0; }
        uint64_t id = next_stream_ + 1; uint8_t h[kBulkHeaderBytes];
        encode_bulk_header(h, BulkKind::Begin, id, 0, total, uint32_t(n));
        if (!bulk_->queue(h, sizeof(h), descriptor, n)) { ++stats_.bulk_refused_full; return 0; }
        next_stream_ = id; out_streams_[id] = {total, 0}; return id;
    }
    size_t write_bulk(uint64_t stream, const uint8_t* data, size_t n) {
        auto it = out_streams_.find(stream); if (it == out_streams_.end()) return 0;
        size_t accepted = 0;
        while (accepted < n && it->second.written < it->second.total) {
            size_t take = size_t(std::min<uint64_t>({uint64_t(n - accepted), uint64_t(kBulkChunkBytes), it->second.total - it->second.written}));
            uint8_t h[kBulkHeaderBytes];
            encode_bulk_header(h, BulkKind::Chunk, stream, it->second.written, it->second.total, uint32_t(take));
            if (!bulk_->queue(h, sizeof(h), data + accepted, take)) break;
            accepted += take; it->second.written += take; stats_.bulk_bytes_sent += take;
        }
        if (accepted == 0 && n) ++stats_.bulk_refused_full;
        return accepted;
    }
    bool end_bulk(uint64_t stream) { return finish(stream, BulkKind::End); }
    bool abort_bulk(uint64_t stream) { return finish(stream, BulkKind::Abort); }

    // bulk.in: Begin (data = descriptor), Chunk (data at offset, in order), End, Abort.
    bool receive_bulk(BulkEvent& e) {
        if (bulk_in_.empty()) return false;
        e = std::move(bulk_in_.front()); bulk_in_.pop_front(); bulk_in_bytes_ -= e.data.size(); return true;
    }

    void poll(size_t budget = size_t(8) << 20) {
        control_->flush(budget); bulk_->flush(budget);
        std::vector<uint8_t> m;
        for (size_t i = 0; i < kControlQueuePackets && !control_->closed(); ++i) {
            if (control_in_.size() >= kControlQueuePackets) { ++stats_.control_read_stalls; break; }
            if (!control_->next(m, 1 << 16)) break;
            ControlPacket p; if (!decode_control(m.data(), m.size(), p)) { violation(); return; }
            control_in_.push_back(p); ++stats_.control_received;
        }
        size_t read = 0;
        while (read < budget && !bulk_->closed()) {
            if (bulk_in_bytes_ >= kBulkWindowBytes) { ++stats_.bulk_read_stalls; break; }
            if (!bulk_->next(m, budget - read)) break;
            read += m.size();
            if (!accept_bulk(m)) { violation(); return; }
        }
        if (control_->closed() != bulk_->closed()) close(1000);   // the four FIFOs live and die together
    }

private:
    struct OutStream { uint64_t total = 0, written = 0; };
    struct InStream { uint64_t total = 0, received = 0; };
    bool finish(uint64_t stream, BulkKind kind) {
        auto it = out_streams_.find(stream); if (it == out_streams_.end()) return false;
        if (kind == BulkKind::End && it->second.written != it->second.total) return false;
        uint8_t h[kBulkHeaderBytes]; encode_bulk_header(h, kind, stream, it->second.written, it->second.total, 0);
        if (!bulk_->queue(h, sizeof(h))) { ++stats_.bulk_refused_full; return false; }
        if (kind == BulkKind::End) ++stats_.bulk_streams_sent;
        out_streams_.erase(it); return true;
    }
    bool accept_bulk(std::vector<uint8_t>& m) {
        if (m.size() < kBulkHeaderBytes || get_le(m.data(), 4) != kBulkMagic) return false;
        BulkEvent e; uint64_t kind = get_le(m.data() + 4, 2);
        e.stream = get_le(m.data() + 8, 8); e.offset = get_le(m.data() + 16, 8); e.total = get_le(m.data() + 24, 8);
        uint64_t length = get_le(m.data() + 32, 4);
        if (length != m.size() - kBulkHeaderBytes) return false;
        auto it = in_streams_.find(e.stream);
        switch (kind) {
        case uint64_t(BulkKind::Begin):
            if (e.stream <= last_in_stream_ || in_streams_.size() >= kBulkMaxStreams || length > kBulkDescriptorBytes || e.offset) return false;
            last_in_stream_ = e.stream; in_streams_[e.stream] = {e.total, 0}; e.kind = BulkKind::Begin; break;
        case uint64_t(BulkKind::Chunk):
            if (it == in_streams_.end() || !length || length > kBulkChunkBytes || e.offset != it->second.received
                || it->second.received + length > it->second.total) return false;
            it->second.received += length; stats_.bulk_bytes_received += length; e.kind = BulkKind::Chunk; break;
        case uint64_t(BulkKind::End):
            if (it == in_streams_.end() || length || it->second.received != it->second.total) return false;
            in_streams_.erase(it); ++stats_.bulk_streams_received; e.kind = BulkKind::End; break;
        case uint64_t(BulkKind::Abort):
            if (it == in_streams_.end() || length) return false;
            in_streams_.erase(it); e.kind = BulkKind::Abort; break;
        default: return false;
        }
        e.data.assign(m.begin() + kBulkHeaderBytes, m.end());
        bulk_in_bytes_ += e.data.size(); bulk_in_.push_back(std::move(e)); return true;
    }
    void violation() { ++stats_.protocol_violations; close(1002); }

    std::unique_ptr<Stream> control_, bulk_;
    Role role_; uint64_t account_; LinkId id_;
    LinkStats stats_;
    std::deque<ControlPacket> control_in_;
    std::deque<BulkEvent> bulk_in_; size_t bulk_in_bytes_ = 0;
    std::map<uint64_t, OutStream> out_streams_; uint64_t next_stream_ = 0;
    std::map<uint64_t, InStream> in_streams_; uint64_t last_in_stream_ = 0;
};

// ---- HTTP upgrade helpers -------------------------------------------------------------------------------------------
namespace detail {
inline std::string lower(std::string s) { for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c))); return s; }
inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}
// Request line and lower-cased header names; false when malformed.
inline bool parse_http(const std::string& head, std::string& first, std::map<std::string, std::string>& headers) {
    size_t at = head.find("\r\n"); if (at == std::string::npos) return false;
    first = head.substr(0, at); at += 2;
    while (at < head.size()) {
        size_t end = head.find("\r\n", at); if (end == std::string::npos || end == at) break;
        std::string line = head.substr(at, end - at); size_t colon = line.find(':');
        if (colon == std::string::npos) return false;
        headers[lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1)); at = end + 2;
    }
    return true;
}
}  // namespace detail

// ---- listener (host side) -------------------------------------------------------------------------------------------
struct ListenerConfig {
    uint16_t port = 0;              // 0 = ephemeral; read back with port()
    std::string prefix;             // the canvas prefix in /ws/link/<prefix>/{control,bulk}
    Key key{};                      // the issuing service's ticket key
    uint64_t max_ticket_life = 300; // seconds a ticket may be valid for
    uint64_t pending_seconds = 10;  // an unauthenticated socket is dropped after this
};
struct ListenerCounters {
    uint64_t accepted = 0, links = 0;
    uint64_t refused_handshake = 0, refused_ticket = 0, refused_replay = 0, refused_pair = 0, timed_out = 0;
};
class LinkListener {
public:
    static constexpr size_t kAcceptPerPoll = 8, kMaxPending = 32, kMaxHead = 8192;
    explicit LinkListener(ListenerConfig config) : config_(std::move(config)), prefix_hash_(prefix_hash(config_.prefix)) {
        contracts::ensure_sockets();
        listener_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listener_ == INVALID_SOCKET) throw std::runtime_error("link listener socket refused");
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(config_.port);
        if (!WireSocket::reuse_listen_port(listener_) || bind(listener_, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 || listen(listener_, 16) != 0 || !WireSocket::set_nonblocking(listener_)) {
            WireSocket::close_socket(listener_); throw std::runtime_error("link listener bind/listen refused");
        }
        platform::socklen len = sizeof(a);
        if (getsockname(listener_, reinterpret_cast<sockaddr*>(&a), &len) != 0) { WireSocket::close_socket(listener_); throw std::runtime_error("getsockname failed"); }
        port_ = ntohs(a.sin_port);
    }
    ~LinkListener() { for (auto& p : pending_) if (p.s != INVALID_SOCKET) WireSocket::close_socket(p.s); WireSocket::close_socket(listener_); }
    LinkListener(const LinkListener&) = delete;
    LinkListener& operator=(const LinkListener&) = delete;
    uint16_t port() const { return port_; }
    const ListenerCounters& counters() const { return counters_; }

    // Accept, upgrade, check tickets, pair halves. `now` is unix seconds.
    void poll(uint64_t now) {
        for (size_t i = 0; i < kAcceptPerPoll && pending_.size() < kMaxPending; ++i) {
            SOCKET c = accept(listener_, nullptr, nullptr);
            if (c == INVALID_SOCKET) break;
            if (!WireSocket::set_nonblocking(c)) { WireSocket::close_socket(c); continue; }
            Pending p; p.s = c; p.started = now; pending_.push_back(std::move(p)); ++counters_.accepted;
        }
        for (size_t i = 0; i < pending_.size();) {
            if (advance(pending_[i], now)) { pending_.erase(pending_.begin() + ptrdiff_t(i)); continue; }
            ++i;
        }
        for (auto it = halves_.begin(); it != halves_.end();) {
            if (now > it->second.ticket.expires_at) { ++counters_.timed_out; it = halves_.erase(it); } else ++it;
        }
        for (auto it = used_.begin(); it != used_.end();) { if (now > it->second) it = used_.erase(it); else ++it; }
    }
    std::unique_ptr<Link> take() {
        if (ready_.empty()) return nullptr;
        auto l = std::move(ready_.front()); ready_.pop_front(); return l;
    }

private:
    enum class Stage { Request, Response, Ticket };
    struct Pending {
        SOCKET s = INVALID_SOCKET; Stage stage = Stage::Request; uint64_t started = 0;
        std::string head, response; size_t sent = 0; Channel channel = Channel::Control;
        std::unique_ptr<WsStream> ws;
    };
    struct Half { std::unique_ptr<WsStream> control, bulk; Ticket ticket; };

    void refuse_http(Pending& p) {
        static const char r[] = "HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
        ::send(p.s, r, int(sizeof(r) - 1), platform::kSendFlags); WireSocket::close_socket(p.s); p.s = INVALID_SOCKET; ++counters_.refused_handshake;
    }
    // True when the pending socket is finished with (handed on, refused or timed out).
    bool advance(Pending& p, uint64_t now) {
        if (now > p.started + config_.pending_seconds) {
            ++counters_.timed_out;
            if (p.ws) p.ws->close(1008); else WireSocket::close_socket(p.s);
            return true;
        }
        if (p.stage == Stage::Request) {
            char buf[2048];
            int n = ::recv(p.s, buf, sizeof(buf), 0);
            if (n == 0 || (n < 0 && !WireSocket::would_block())) { WireSocket::close_socket(p.s); ++counters_.refused_handshake; return true; }
            if (n < 0) return false;
            p.head.append(buf, size_t(n));
            size_t end = p.head.find("\r\n\r\n");
            if (end == std::string::npos) { if (p.head.size() > kMaxHead) { refuse_http(p); return true; } return false; }
            std::vector<uint8_t> leftover(p.head.begin() + ptrdiff_t(end + 4), p.head.end());
            p.head.resize(end + 2);
            std::string first; std::map<std::string, std::string> h;
            if (!detail::parse_http(p.head, first, h)) { refuse_http(p); return true; }
            std::string base = "GET /ws/link/" + config_.prefix + "/";
            if (first == base + "control HTTP/1.1") p.channel = Channel::Control;
            else if (first == base + "bulk HTTP/1.1") p.channel = Channel::Bulk;
            else { refuse_http(p); return true; }
            auto key = h.find("sec-websocket-key");
            if (detail::lower(h["upgrade"]) != "websocket" || detail::lower(h["connection"]).find("upgrade") == std::string::npos
                || h["sec-websocket-version"] != "13" || key == h.end() || key->second.size() != 24) { refuse_http(p); return true; }
            p.response = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: "
                + websocket_accept(key->second) + "\r\n\r\n";
            p.head.assign(leftover.begin(), leftover.end());     // frames a client sent early are kept for the stream
            p.stage = Stage::Response;
        }
        if (p.stage == Stage::Response) {
            int n = ::send(p.s, p.response.data() + p.sent, int(p.response.size() - p.sent), platform::kSendFlags);
            if (n < 0 && !WireSocket::would_block()) { WireSocket::close_socket(p.s); ++counters_.refused_handshake; return true; }
            if (n > 0) p.sent += size_t(n);
            if (p.sent < p.response.size()) return false;
            size_t max_message = p.channel == Channel::Control ? kControlPacketBytes : kBulkHeaderBytes + kBulkChunkBytes;
            size_t out_limit = p.channel == Channel::Control ? kControlQueuePackets * (kControlPacketBytes + 4) : kBulkWindowBytes + kBulkHeaderBytes + 16;
            SOCKET s = p.s; p.s = INVALID_SOCKET;
            p.ws = std::make_unique<WsStream>(s, false, max_message, out_limit, std::vector<uint8_t>(p.head.begin(), p.head.end()));
            p.stage = Stage::Ticket;
        }
        std::vector<uint8_t> m;
        if (!p.ws->next(m, kTicketBytes + 14)) {
            if (p.ws->closed()) { ++counters_.refused_handshake; return true; }
            return false;
        }
        Ticket t;
        if (!parse_ticket(m.data(), m.size(), config_.key, t) || t.channel != p.channel || t.prefix_hash != prefix_hash_
            || now > t.expires_at || t.issued_at > now + 30 || t.expires_at < t.issued_at || t.expires_at - t.issued_at > config_.max_ticket_life) {
            ++counters_.refused_ticket; p.ws->close(1008); return true;
        }
        auto use = std::make_pair(t.link_id, uint8_t(t.channel));
        if (used_.count(use)) { ++counters_.refused_replay; p.ws->close(1008); return true; }
        used_[use] = t.expires_at;
        auto& half = halves_[t.link_id];
        bool first_half = !half.control && !half.bulk;
        if (!first_half && (half.ticket.account != t.account || half.ticket.role != t.role)) {
            ++counters_.refused_pair; p.ws->close(1008);
            if (half.control) half.control->close(1008);
            if (half.bulk) half.bulk->close(1008);
            halves_.erase(t.link_id); return true;
        }
        if (first_half) half.ticket = t;
        (t.channel == Channel::Control ? half.control : half.bulk) = std::move(p.ws);
        if (half.control && half.bulk) {
            ready_.push_back(std::make_unique<Link>(std::move(half.control), std::move(half.bulk), Role::Host, t.account, t.link_id));
            halves_.erase(t.link_id); ++counters_.links;
        }
        return true;
    }

    ListenerConfig config_;
    std::array<uint8_t, 32> prefix_hash_;
    SOCKET listener_ = INVALID_SOCKET; uint16_t port_ = 0;
    ListenerCounters counters_;
    std::vector<Pending> pending_;
    std::map<LinkId, Half> halves_;
    std::map<std::pair<LinkId, uint8_t>, uint64_t> used_;
    std::deque<std::unique_ptr<Link>> ready_;
};

// ---- dialer (polled, never blocks) ----------------------------------------------------------------------------------
// Plain ws:// to a LinkListener on this machine, or to whatever terminates TLS for it. The public wss:// dial-out is
// stage 4; this is the loopback passage both share. Construct, then poll() each tick until ready() or failed(); take()
// hands out the Link. Nonblocking connect, upgrade and ticket; numeric IPv4 only (no resolver, which would block).
struct DialTarget { std::string host = "127.0.0.1"; uint16_t port = 0; std::string prefix; int timeout_ms = 5000; };
class LinkDialer {
public:
    LinkDialer(DialTarget target, const TicketBytes& control_ticket, const TicketBytes& bulk_ticket)
        : target_(std::move(target)), deadline_(std::chrono::steady_clock::now() + std::chrono::milliseconds(target_.timeout_ms)) {
        contracts::ensure_sockets();
        halves_[0].channel = Channel::Control; halves_[0].ticket = control_ticket;
        halves_[1].channel = Channel::Bulk; halves_[1].ticket = bulk_ticket;
        std::memcpy(id_.data(), control_ticket.data() + 8, 16);      // unverified: the dialer holds no key
        account_ = get_le(control_ticket.data() + 56, 8);
        for (auto& h : halves_) if (!start(h)) { fail(); return; }
    }
    ~LinkDialer() { for (auto& h : halves_) if (h.s != INVALID_SOCKET) WireSocket::close_socket(h.s); }
    LinkDialer(const LinkDialer&) = delete;
    LinkDialer& operator=(const LinkDialer&) = delete;
    bool failed() const { return failed_; }
    bool ready() const { return !failed_ && halves_[0].ws && halves_[1].ws && !link_taken_; }
    const char* reason() const { return reason_; }
    void poll() {
        if (failed_ || ready() || link_taken_) return;
        if (std::chrono::steady_clock::now() > deadline_) { fail("timed out"); return; }
        for (auto& h : halves_) if (!h.ws && !advance(h)) { fail(reason_ ? reason_ : "refused"); return; }
    }
    std::unique_ptr<Link> take() {
        if (!ready()) return nullptr;
        link_taken_ = true;
        return std::make_unique<Link>(std::move(halves_[0].ws), std::move(halves_[1].ws), Role::Dialer, account_, id_);
    }

private:
    enum class Stage { Connect, Send, Read };
    struct Half {
        Channel channel = Channel::Control; TicketBytes ticket{};
        SOCKET s = INVALID_SOCKET; Stage stage = Stage::Connect;
        std::string request, head, key; size_t sent = 0;
        std::unique_ptr<WsStream> ws;
    };
    void fail(const char* why = "refused") {
        failed_ = true; if (!reason_) reason_ = why;
        for (auto& h : halves_) { if (h.s != INVALID_SOCKET) { WireSocket::close_socket(h.s); h.s = INVALID_SOCKET; } h.ws.reset(); }
    }
    bool start(Half& h) {
        h.s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (h.s == INVALID_SOCKET || !WireSocket::set_nonblocking(h.s)) { reason_ = "socket refused"; return false; }
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(target_.port);
        if (inet_pton(AF_INET, target_.host.c_str(), &a.sin_addr) != 1) { reason_ = "host is not a numeric IPv4 address"; return false; }
        if (connect(h.s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 && !platform::connect_would_block()) { reason_ = "connect refused"; return false; }
        uint8_t raw[16]; random_bytes(raw, sizeof(raw)); h.key = base64(raw, sizeof(raw));
        h.request = "GET /ws/link/" + target_.prefix + "/" + channel_name(h.channel) + " HTTP/1.1\r\nHost: " + target_.host + ":"
            + std::to_string(target_.port) + "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: " + h.key
            + "\r\nSec-WebSocket-Version: 13\r\n\r\n";
        return true;
    }
    // False on refusal (reason_ set); true while in progress or done.
    bool advance(Half& h) {
        if (h.stage == Stage::Connect) {
            int state = platform::connect_state(h.s);
            if (state == -2) { reason_ = "select refused"; return false; }
            if (state == 0) return true;                         // still connecting
            if (state < 0) { reason_ = "connect refused"; return false; }
            h.stage = Stage::Send;
        }
        if (h.stage == Stage::Send) {
            int n = ::send(h.s, h.request.data() + h.sent, int(h.request.size() - h.sent), platform::kSendFlags);
            if (n < 0 && !WireSocket::would_block()) { reason_ = "send refused"; return false; }
            if (n > 0) h.sent += size_t(n);
            if (h.sent < h.request.size()) return true;
            h.stage = Stage::Read;
        }
        char buf[2048];
        int n = ::recv(h.s, buf, sizeof(buf), 0);
        if (n == 0 || (n < 0 && !WireSocket::would_block())) { reason_ = "upgrade refused"; return false; }
        if (n < 0) return true;
        h.head.append(buf, size_t(n));
        size_t end = h.head.find("\r\n\r\n");
        if (end == std::string::npos) { if (h.head.size() > 8192) { reason_ = "upgrade answer too long"; return false; } return true; }
        std::vector<uint8_t> leftover(h.head.begin() + ptrdiff_t(end + 4), h.head.end());
        h.head.resize(end + 2);
        std::string first; std::map<std::string, std::string> hd;
        if (!detail::parse_http(h.head, first, hd) || first.rfind("HTTP/1.1 101", 0) != 0 || hd["sec-websocket-accept"] != websocket_accept(h.key)) {
            reason_ = "upgrade refused"; return false;
        }
        size_t max_message = h.channel == Channel::Control ? kControlPacketBytes : kBulkHeaderBytes + kBulkChunkBytes;
        size_t out_limit = h.channel == Channel::Control ? kControlQueuePackets * (kControlPacketBytes + 8) : kBulkWindowBytes + kBulkHeaderBytes + 16;
        SOCKET s = h.s; h.s = INVALID_SOCKET;
        h.ws = std::make_unique<WsStream>(s, true, max_message, out_limit, std::move(leftover));
        if (!h.ws->queue(h.ticket.data(), h.ticket.size())) { reason_ = "ticket refused by own queue"; return false; }
        h.ws->flush(kTicketBytes + 16);                           // what does not go now goes on the Link's first poll
        return true;
    }

    DialTarget target_;
    std::chrono::steady_clock::time_point deadline_;
    Half halves_[2];
    LinkId id_{}; uint64_t account_ = 0;
    bool failed_ = false, link_taken_ = false;
    const char* reason_ = nullptr;
};
}  // namespace nodus::link
