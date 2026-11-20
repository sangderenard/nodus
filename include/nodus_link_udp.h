#pragma once
// Nodus link over direct UDP: rendezvous at a broker, hole punching, then the ordinary Link over two reliable, ordered,
// AES-256-GCM message streams that share one UDP socket. Research case a02ca4a5d8eb4dbb85e91a76daa0f53d (ticket
// d3da321de3234a3783e4201fec36909e). Owner, 2026-11-19: "a page that brokers and then hands off to udp"; "aes-gcm is worth
// building into nodus ... if it's intended to be a computational backend across the internet".
//
// Both Nodus instances only make outgoing requests. Each gets a rendezvous (link id, role, expiry, key K) from
// POST /api/link/rendezvous/<prefix> (turing examples/meowdoku/link_broker.py), says HELLO to the broker's UDP port from
// its own socket, and receives an INTRO sealed with K: the peer's public address and a session key. Then both send
// PUNCH datagrams to each other until one arrives, and the Link runs peer to peer; the broker never carries link data.
//
// Broker datagrams (identical to link_broker.py):
//   HELLO  "NLR1" u32 | 1 u8 | role u8 | prefix length u16 | link id [16] | expires_at u64 | local ipv4 [4] | local port u16 |
//          prefix | HMAC-SHA256(K, all before)
//   INTRO  "NLR1" u32 | 2 u8 | 0 u8 | 0 u16 | link id [16] | nonce [12] | AES-GCM(K, aad = first 24 bytes):
//          peer public ipv4 [4, network order] | peer public port u16 | peer role u8 | 0 u8 | pair id [16] | session key [32] |
//          peer local ipv4 [4] | peer local port u16
//   Two candidates per peer, as ICE has host and server-reflexive ones: punching goes to both, and the first address that
//   answers with the session key is the peer's. A pair behind one NAT meets on the local one, two networks on the public.
// Peer datagrams, all sealed with the session key S (AES-256-GCM, aad = the 16-byte header):
//   header "NLU1" u32 | kind u8 | channel u8 | 0 u16 | seq u64;   nonce = sender role | kind | channel | 0 | seq (12 bytes)
//   PUNCH 3, KEEPALIVE 6: pair id [16].   CLOSE 7: code u16.
//   DATA 4 (channel 1 control, 2 bulk): message length u32 | offset u32 | bytes; seq counts the channel's segments.
//   ACK 5: next expected seq u64 | bitmap u64 (bit i: seq next+1+i already held).
// Reliability is selective-ack with a fixed window and a fixed resend time; every message arrives once, in order.
//
// Platform: the two classes in the platform region (UdpSocket, Aead) are the only system calls. This is the Windows
// region (Winsock, CNG); another platform supplies the same two classes and nothing else changes.
#include "nodus_link.h"

#include <future>
#include <map>

namespace nodus::link {

constexpr uint32_t kRendezvousMagic = 0x31524C4E;   // "NLR1"
constexpr uint32_t kUdpMagic = 0x31554C4E;          // "NLU1"
enum class UdpKind : uint8_t { Hello = 1, Intro = 2, Punch = 3, Data = 4, Ack = 5, Keepalive = 6, Close = 7 };
constexpr size_t kUdpHeaderBytes = 16, kUdpTagBytes = 16, kUdpDataHead = 8;
constexpr size_t kUdpSegmentBytes = 1200;           // message bytes per DATA datagram: 1240 on the wire, under a 1472 MTU
constexpr size_t kUdpWindow = 2048;                 // segments in flight per channel
constexpr int kUdpResendMs = 250, kUdpHelloMs = 500, kUdpPunchMs = 100, kUdpKeepaliveMs = 5000, kUdpDeadMs = 30000;

// ================================ platform region: Windows (Winsock UDP, CNG AES-256-GCM) ================================
#if defined(_WIN32)
class UdpSocket {
public:
    UdpSocket() {
        WSADATA w{}; if (WSAStartup(MAKEWORD(2, 2), &w) != 0) return;
        started_ = true;
        s_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (s_ == INVALID_SOCKET) return;
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_ANY); a.sin_port = 0;
        u_long nb = 1;
        if (bind(s_, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 || ioctlsocket(s_, FIONBIO, &nb) != 0) { closesocket(s_); s_ = INVALID_SOCKET; }
    }
    ~UdpSocket() { if (s_ != INVALID_SOCKET) closesocket(s_); if (started_) WSACleanup(); }
    UdpSocket(const UdpSocket&) = delete; UdpSocket& operator=(const UdpSocket&) = delete;
    bool ok() const { return s_ != INVALID_SOCKET; }
    // ip in network order, port in host order
    bool send(uint32_t ip, uint16_t port, const uint8_t* p, size_t n) {
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = ip; a.sin_port = htons(port);
        return sendto(s_, reinterpret_cast<const char*>(p), int(n), 0, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == int(n);
    }
    // bytes received (0 when nothing is waiting)
    int receive(uint8_t* p, size_t cap, uint32_t& ip, uint16_t& port) {
        sockaddr_in a{}; int al = sizeof(a);
        int n = recvfrom(s_, reinterpret_cast<char*>(p), int(cap), 0, reinterpret_cast<sockaddr*>(&a), &al);
        if (n <= 0) return 0;
        ip = a.sin_addr.s_addr; port = ntohs(a.sin_port); return n;
    }
    uint16_t local_port() const {
        sockaddr_in a{}; int al = sizeof(a);
        return getsockname(s_, reinterpret_cast<sockaddr*>(&a), &al) == 0 ? ntohs(a.sin_port) : 0;
    }
    // This machine's address on the interface that reaches `ip` (network order); 0 when unknown. Sends nothing.
    static uint32_t local_ip_toward(uint32_t ip, uint16_t port) {
        SOCKET t = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (t == INVALID_SOCKET) return 0;
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = ip; a.sin_port = htons(port);
        uint32_t out = 0;
        if (connect(t, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0) {
            sockaddr_in l{}; int ll = sizeof(l);
            if (getsockname(t, reinterpret_cast<sockaddr*>(&l), &ll) == 0) out = l.sin_addr.s_addr;
        }
        closesocket(t); return out;
    }
    // A name to an IPv4 address (network order); blocking, so callers run it off the polling thread.
    static bool resolve(const std::string& host, uint32_t& ip) {
        WSADATA w{}; if (WSAStartup(MAKEWORD(2, 2), &w) != 0) return false;
        addrinfo hints{}; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_DGRAM;
        addrinfo* r = nullptr; bool ok = getaddrinfo(host.c_str(), nullptr, &hints, &r) == 0 && r;
        if (ok) ip = reinterpret_cast<sockaddr_in*>(r->ai_addr)->sin_addr.s_addr;
        if (r) freeaddrinfo(r);
        WSACleanup(); return ok;
    }
private:
    SOCKET s_ = INVALID_SOCKET; bool started_ = false;
};

class Aead {   // AES-256-GCM, 12-byte nonce, 16-byte tag
public:
    explicit Aead(const uint8_t* key32) {
        static BCRYPT_ALG_HANDLE alg = [] {
            BCRYPT_ALG_HANDLE h = detail::provider(BCRYPT_AES_ALGORITHM, false);
            if (!BCRYPT_SUCCESS(BCryptSetProperty(h, BCRYPT_CHAINING_MODE, reinterpret_cast<PUCHAR>(const_cast<wchar_t*>(BCRYPT_CHAIN_MODE_GCM)),
                                                  ULONG(sizeof(BCRYPT_CHAIN_MODE_GCM)), 0)))
                throw std::runtime_error("CNG AES-GCM refused");
            return h;
        }();
        if (!BCRYPT_SUCCESS(BCryptGenerateSymmetricKey(alg, &key_, nullptr, 0, const_cast<PUCHAR>(key32), 32, 0))) key_ = nullptr;
    }
    ~Aead() { if (key_) BCryptDestroyKey(key_); }
    Aead(const Aead&) = delete; Aead& operator=(const Aead&) = delete;
    bool ok() const { return key_ != nullptr; }
    // out holds n + 16 bytes (ciphertext then tag)
    bool seal(const uint8_t* nonce, const uint8_t* aad, size_t aad_n, const uint8_t* in, size_t n, uint8_t* out) const {
        BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO info; BCRYPT_INIT_AUTH_MODE_INFO(info);
        info.pbNonce = const_cast<PUCHAR>(nonce); info.cbNonce = 12; info.pbAuthData = const_cast<PUCHAR>(aad); info.cbAuthData = ULONG(aad_n);
        info.pbTag = out + n; info.cbTag = 16; ULONG done = 0;
        return BCRYPT_SUCCESS(BCryptEncrypt(key_, const_cast<PUCHAR>(in), ULONG(n), &info, nullptr, 0, out, ULONG(n), &done, 0)) && done == n;
    }
    // in holds n bytes (ciphertext then tag); out holds n - 16; false when it does not authenticate
    bool open(const uint8_t* nonce, const uint8_t* aad, size_t aad_n, const uint8_t* in, size_t n, uint8_t* out) const {
        if (n < 16) return false;
        BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO info; BCRYPT_INIT_AUTH_MODE_INFO(info);
        info.pbNonce = const_cast<PUCHAR>(nonce); info.cbNonce = 12; info.pbAuthData = const_cast<PUCHAR>(aad); info.cbAuthData = ULONG(aad_n);
        info.pbTag = const_cast<PUCHAR>(in + n - 16); info.cbTag = 16; ULONG done = 0;
        return BCRYPT_SUCCESS(BCryptDecrypt(key_, const_cast<PUCHAR>(in), ULONG(n - 16), &info, nullptr, 0, out, ULONG(n - 16), &done, 0));
    }
private:
    BCRYPT_KEY_HANDLE key_ = nullptr;
};
#else
#error "nodus_link_udp.h: supply UdpSocket and Aead for this platform (the Windows region above is the reference)"
#endif
// ============================================ end of the platform region ============================================

struct UdpTarget {
    std::string broker_host;        // the name the rendezvous answered with
    uint16_t broker_port = 0;
    std::string prefix;
    Role role = Role::Dialer;       // this side: host or dialer (the rendezvous's direction)
    LinkId link_id{};               // this side's rendezvous link id
    uint64_t expires_at = 0;
    Key key{};                      // this side's rendezvous key K
    uint64_t account = 0;
    int timeout_ms = 15000;         // from start to an open path (hello, intro and punch together)
};

// One UDP socket, the rendezvous and both channels. Shared by the two channel streams; driven by whoever polls.
class UdpCore {
public:
    using Clock = std::chrono::steady_clock;
    UdpCore(const UdpTarget& t, size_t max_message, size_t out_limit)
        : t_(t), max_message_(max_message), out_limit_(out_limit), start_(Clock::now()) {
        if (!sock_.ok()) { fail("no UDP socket"); return; }
        std::string host = t.broker_host;
        resolving_ = std::async(std::launch::async, [host] { uint32_t ip = 0; return UdpSocket::resolve(host, ip) ? ip : 0u; });
        rendezvous_ = std::make_unique<Aead>(t.key.data());
    }
    bool open() const { return state_ == State::Open; }
    bool closed() const { return state_ == State::Closed; }
    bool failed() const { return closed() && !was_open_; }
    const char* reason() const { return why_; }
    const LinkId& pair_id() const { return pair_; }

    void pump() {
        if (closed()) return;
        const auto now = Clock::now();
        if (state_ == State::Resolving) {
            if (resolving_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) { if (late(now)) fail("broker name did not resolve"); return; }
            broker_ip_ = resolving_.get();
            if (!broker_ip_) { fail("broker name did not resolve"); return; }
            local_ip_ = UdpSocket::local_ip_toward(broker_ip_, t_.broker_port); local_port_ = sock_.local_port();
            state_ = State::Hello;
        }
        receive_all(now);
        if (closed()) return;
        if (state_ == State::Hello) {
            if (late(now)) { fail("no introduction from the broker"); return; }
            if (now - last_hello_ >= std::chrono::milliseconds(kUdpHelloMs)) { send_hello(); last_hello_ = now; }
        } else if (state_ == State::Punching) {
            if (late(now)) { fail("hole punching timed out"); return; }
            if (now - last_punch_ >= std::chrono::milliseconds(kUdpPunchMs)) {
                send_small(UdpKind::Punch, 0, pair_.data(), pair_.size());                         // the public candidate
                if (local_peer_ip_ && (local_peer_ip_ != peer_ip_ || local_peer_port_ != peer_port_)) {
                    const uint32_t ip = peer_ip_; const uint16_t port = peer_port_;
                    peer_ip_ = local_peer_ip_; peer_port_ = local_peer_port_;                       // the local candidate
                    send_small(UdpKind::Punch, 0, pair_.data(), pair_.size());
                    peer_ip_ = ip; peer_port_ = port;
                }
                last_punch_ = now;
            }
        } else if (state_ == State::Open) {
            if (now - heard_ > std::chrono::milliseconds(kUdpDeadMs)) { fail("the peer went silent"); return; }
            for (int c = 0; c < 2; ++c) { send_acks(c); send_segments(c, now); }
            if (now - sent_ > std::chrono::milliseconds(kUdpKeepaliveMs)) send_small(UdpKind::Keepalive, 0, pair_.data(), pair_.size());
        }
    }
    // channel 0 control, 1 bulk
    bool queue(int c, const uint8_t* a, size_t an, const uint8_t* b, size_t bn) {
        const size_t n = an + bn;
        if (closed() || n == 0 || n > max_message_ || ch_[c].queued_bytes + n > out_limit_) return false;
        std::vector<uint8_t> m; m.reserve(n);
        if (an) m.insert(m.end(), a, a + an);
        if (bn) m.insert(m.end(), b, b + bn);
        Chan& ch = ch_[c];
        for (size_t off = 0; off < n; off += kUdpSegmentBytes) {
            const size_t len = std::min(kUdpSegmentBytes, n - off);
            Segment s; s.bytes.resize(kUdpDataHead + len);
            put_le(s.bytes.data(), n, 4); put_le(s.bytes.data() + 4, off, 4);
            std::memcpy(s.bytes.data() + kUdpDataHead, m.data() + off, len);
            ch.waiting.push_back(std::move(s));
        }
        ch.queued_bytes += n; return true;
    }
    bool next(int c, std::vector<uint8_t>& out) {
        Chan& ch = ch_[c];
        if (ch.complete.empty()) return false;
        out = std::move(ch.complete.front()); ch.complete.pop_front(); return true;
    }
    void close(uint16_t code) {
        if (closed()) return;
        if (state_ == State::Open) { uint8_t b[2]; put_le(b, code, 2); send_small(UdpKind::Close, 0, b, 2); }
        state_ = State::Closed; if (!why_) why_ = "closed";
    }

private:
    enum class State { Resolving, Hello, Punching, Open, Closed };
    struct Segment { std::vector<uint8_t> bytes; uint64_t seq = 0; Clock::time_point sent{}; };
    struct Chan {
        std::deque<Segment> waiting;                    // not yet given a seq
        std::map<uint64_t, Segment> unacked;            // in flight
        uint64_t next_seq = 0; size_t queued_bytes = 0;
        std::map<uint64_t, std::vector<uint8_t>> held; // arrived ahead of `expected`
        uint64_t expected = 0; bool ack_due = false;
        std::vector<uint8_t> assembling; size_t assembled = 0;
        std::deque<std::vector<uint8_t>> complete;
    };

    bool late(Clock::time_point now) const { return now - start_ > std::chrono::milliseconds(t_.timeout_ms); }
    void fail(const char* why) { if (!why_ || state_ != State::Closed) why_ = why; state_ = State::Closed; }
    uint8_t peer_role() const { return t_.role == Role::Host ? uint8_t(Role::Dialer) : uint8_t(Role::Host); }

    void send_hello() {
        std::vector<uint8_t> h(38 + t_.prefix.size());
        put_le(h.data(), kRendezvousMagic, 4); h[4] = uint8_t(UdpKind::Hello); h[5] = uint8_t(t_.role);
        put_le(h.data() + 6, t_.prefix.size(), 2); std::memcpy(h.data() + 8, t_.link_id.data(), 16);
        put_le(h.data() + 24, t_.expires_at, 8); std::memcpy(h.data() + 32, &local_ip_, 4); put_le(h.data() + 36, local_port_, 2);
        std::memcpy(h.data() + 38, t_.prefix.data(), t_.prefix.size());
        auto mac = hmac_sha256(t_.key, h.data(), h.size());
        h.insert(h.end(), mac.begin(), mac.end());
        sock_.send(broker_ip_, t_.broker_port, h.data(), h.size());
    }
    bool take_intro(const uint8_t* p, size_t n) {
        if (n != 24 + 12 + 62 + 16 || get_le(p, 4) != kRendezvousMagic || p[4] != uint8_t(UdpKind::Intro)
            || std::memcmp(p + 8, t_.link_id.data(), 16) != 0) return false;
        uint8_t plain[62];
        if (!rendezvous_->open(p + 24, p, 24, p + 36, 62 + 16, plain)) return false;
        std::memcpy(&local_peer_ip_, plain + 56, 4); local_peer_port_ = uint16_t(get_le(plain + 60, 2));
        std::memcpy(&peer_ip_, plain, 4); peer_port_ = uint16_t(get_le(plain + 4, 2));
        if (plain[6] != peer_role()) return false;
        std::memcpy(pair_.data(), plain + 8, 16);
        session_ = std::make_unique<Aead>(plain + 24);
        SecureZeroMemory(plain, sizeof(plain));
        if (!session_->ok()) { fail("session key refused"); return false; }
        state_ = State::Punching; start_ = Clock::now(); return true;
    }
    static void nonce(uint8_t* out, uint8_t role, UdpKind kind, uint8_t channel, uint64_t seq) {
        out[0] = role; out[1] = uint8_t(kind); out[2] = channel; out[3] = 0; put_le(out + 4, seq, 8);
    }
    void send_sealed(UdpKind kind, uint8_t channel, uint64_t seq, const uint8_t* p, size_t n) {
        uint8_t d[kUdpHeaderBytes + kUdpDataHead + kUdpSegmentBytes + kUdpTagBytes];
        put_le(d, kUdpMagic, 4); d[4] = uint8_t(kind); d[5] = channel; put_le(d + 6, 0, 2); put_le(d + 8, seq, 8);
        uint8_t iv[12]; nonce(iv, uint8_t(t_.role), kind, channel, seq);
        if (!session_->seal(iv, d, kUdpHeaderBytes, p, n, d + kUdpHeaderBytes)) return;
        sock_.send(peer_ip_, peer_port_, d, kUdpHeaderBytes + n + kUdpTagBytes);
        sent_ = Clock::now();
    }
    void send_small(UdpKind kind, uint8_t channel, const uint8_t* p, size_t n) { send_sealed(kind, channel, small_seq_[uint8_t(kind) & 7]++, p, n); }

    void receive_all(Clock::time_point now) {
        uint8_t d[2048], plain[2048];
        for (int i = 0; i < 8192; ++i) {
            uint32_t ip = 0; uint16_t port = 0;
            const int n = sock_.receive(d, sizeof(d), ip, port);
            if (n <= 0) return;
            if (state_ == State::Hello) {
                if (ip == broker_ip_ && port == t_.broker_port) take_intro(d, size_t(n));
                continue;
            }
            if (size_t(n) < kUdpHeaderBytes + kUdpTagBytes || get_le(d, 4) != kUdpMagic || !session_) continue;
            const UdpKind kind = UdpKind(d[4]); const uint8_t channel = d[5]; const uint64_t seq = get_le(d + 8, 8);
            uint8_t iv[12]; nonce(iv, peer_role(), kind, channel, seq);
            const size_t m = size_t(n) - kUdpHeaderBytes - kUdpTagBytes;
            if (!session_->open(iv, d, kUdpHeaderBytes, d + kUdpHeaderBytes, m + kUdpTagBytes, plain)) continue;
            peer_ip_ = ip; peer_port_ = port; heard_ = now;           // the address that authenticates is the peer's
            if (state_ == State::Punching) { state_ = State::Open; was_open_ = true; send_small(UdpKind::Punch, 0, pair_.data(), pair_.size()); }
            if (kind == UdpKind::Data && (channel == 1 || channel == 2)) take_data(ch_[channel - 1], seq, plain, m);
            else if (kind == UdpKind::Ack && (channel == 1 || channel == 2) && m == 16) take_ack(ch_[channel - 1], get_le(plain, 8), get_le(plain + 8, 8));
            else if (kind == UdpKind::Close) { state_ = State::Closed; why_ = "the peer closed"; return; }
        }
    }
    void take_data(Chan& ch, uint64_t seq, const uint8_t* p, size_t n) {
        ch.ack_due = true;
        if (seq < ch.expected || seq >= ch.expected + 2 * kUdpWindow || n < kUdpDataHead) return;
        ch.held.emplace(seq, std::vector<uint8_t>(p, p + n));
        for (auto it = ch.held.find(ch.expected); it != ch.held.end(); it = ch.held.find(ch.expected)) {
            const std::vector<uint8_t>& s = it->second;
            const size_t total = size_t(get_le(s.data(), 4)), off = size_t(get_le(s.data() + 4, 4)), len = s.size() - kUdpDataHead;
            if (total == 0 || total > max_message_ || off + len > total || (off == 0) != (ch.assembled == 0) || off != ch.assembled) {
                fail("protocol violation on a UDP channel"); return;
            }
            if (off == 0) ch.assembling.assign(total, 0);
            std::memcpy(ch.assembling.data() + off, s.data() + kUdpDataHead, len);
            ch.assembled += len;
            if (ch.assembled == total) { ch.complete.push_back(std::move(ch.assembling)); ch.assembling.clear(); ch.assembled = 0; }
            ch.held.erase(it); ++ch.expected;
        }
    }
    void take_ack(Chan& ch, uint64_t next, uint64_t bits) {
        while (!ch.unacked.empty() && ch.unacked.begin()->first < next) {
            ch.queued_bytes -= std::min(ch.queued_bytes, ch.unacked.begin()->second.bytes.size() - kUdpDataHead);
            ch.unacked.erase(ch.unacked.begin());
        }
        for (int i = 0; i < 64; ++i) {
            if (!(bits >> i & 1)) continue;
            auto it = ch.unacked.find(next + 1 + uint64_t(i));
            if (it == ch.unacked.end()) continue;
            ch.queued_bytes -= std::min(ch.queued_bytes, it->second.bytes.size() - kUdpDataHead);
            ch.unacked.erase(it);
        }
    }
    void send_acks(int c) {
        Chan& ch = ch_[c];
        if (!ch.ack_due) return;
        uint64_t bits = 0;
        for (int i = 0; i < 64; ++i) if (ch.held.count(ch.expected + 1 + uint64_t(i))) bits |= uint64_t(1) << i;
        uint8_t b[16]; put_le(b, ch.expected, 8); put_le(b + 8, bits, 8);
        send_small(UdpKind::Ack, uint8_t(c + 1), b, 16);
        ch.ack_due = false;
    }
    void send_segments(int c, Clock::time_point now) {
        Chan& ch = ch_[c];
        int resent = 0;
        for (auto& kv : ch.unacked) {
            if (resent >= 256) break;
            if (now - kv.second.sent < std::chrono::milliseconds(kUdpResendMs)) continue;
            send_sealed(UdpKind::Data, uint8_t(c + 1), kv.first, kv.second.bytes.data(), kv.second.bytes.size());
            kv.second.sent = now; ++resent;
        }
        while (!ch.waiting.empty() && ch.unacked.size() < kUdpWindow) {
            Segment s = std::move(ch.waiting.front()); ch.waiting.pop_front();
            s.seq = ch.next_seq++; s.sent = now;
            send_sealed(UdpKind::Data, uint8_t(c + 1), s.seq, s.bytes.data(), s.bytes.size());
            ch.unacked.emplace(s.seq, std::move(s));
        }
    }

    UdpTarget t_;
    size_t max_message_, out_limit_;
    UdpSocket sock_;
    std::future<uint32_t> resolving_;
    std::unique_ptr<Aead> rendezvous_, session_;
    State state_ = State::Resolving;
    bool was_open_ = false;
    const char* why_ = nullptr;
    uint32_t broker_ip_ = 0, peer_ip_ = 0, local_ip_ = 0, local_peer_ip_ = 0;
    uint16_t peer_port_ = 0, local_port_ = 0, local_peer_port_ = 0;
    LinkId pair_{};
    Clock::time_point start_, last_hello_{}, last_punch_{}, heard_{}, sent_{};
    uint64_t small_seq_[8] = {};
    Chan ch_[2];
};

// One of the two message streams a Link needs, over the shared core.
class UdpChannelStream final : public Stream {
public:
    UdpChannelStream(std::shared_ptr<UdpCore> core, int channel) : core_(std::move(core)), c_(channel) {}
    bool closed() const override { return core_->closed(); }
    bool queue(const uint8_t* a, size_t an, const uint8_t* b = nullptr, size_t bn = 0) override { return core_->queue(c_, a, an, b, bn); }
    void flush(size_t) override { core_->pump(); }
    bool next(std::vector<uint8_t>& out, size_t) override { core_->pump(); return core_->next(c_, out); }
    void close(uint16_t code) override { core_->close(code); }
private:
    std::shared_ptr<UdpCore> core_; int c_;
};

// The rendezvous, then a Link: the same shape as WssDialer.
class UdpDialer {
public:
    explicit UdpDialer(const UdpTarget& t) : role_(t.role), account_(t.account) {
        // One core for both channels; its limits fit the larger (bulk) channel and its out queue.
        core_ = std::make_shared<UdpCore>(t, kBulkHeaderBytes + kBulkChunkBytes, kBulkWindowBytes + kBulkHeaderBytes + 16);
    }
    void poll() { if (!taken_) core_->pump(); }
    bool ready() const { return core_->open() && !taken_; }
    bool failed() const { return core_->failed(); }
    const char* reason() const { return core_->reason(); }
    std::unique_ptr<Link> take() {
        if (!ready()) return nullptr;
        taken_ = true;
        return std::make_unique<Link>(std::make_unique<UdpChannelStream>(core_, 0), std::make_unique<UdpChannelStream>(core_, 1),
                                      role_, account_, core_->pair_id());
    }
private:
    std::shared_ptr<UdpCore> core_;
    Role role_; uint64_t account_;
    bool taken_ = false;
};

}  // namespace nodus::link
