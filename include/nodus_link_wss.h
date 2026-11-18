#pragma once
// Nodus link, stage 4: dial out over wss:// (or ws://) with WinHTTP in asynchronous mode. The OS does TLS and name
// resolution on its own threads; every call here returns at once and WssDialer/Link advance by poll(), like the
// loopback LinkDialer. Research case 98d49cb8520e44d092c5d134c37c3c32 (ticket bc214e177933475e9f8ac3f5ed3e3762).
//
// The public path is <path_base><prefix>/{control,bulk} (default /ws/link/), which Apache proxies to the native
// LinkListener on 127.0.0.1. Server certificates are validated by Windows unless a SHA-256 pin of the server's
// certificate (DER) is given: then only that exact certificate is accepted (a self-hosted peer without a public CA).
// The pin is checked when the TLS handshake completes, before the upgrade answer is read and before any ticket is sent.
// Link with winhttp.lib and crypt32.lib.
#include "nodus_link.h"
#include <winhttp.h>
#include <wincrypt.h>

#include <atomic>
#include <mutex>

namespace nodus::link {

struct WssTarget {
    std::string host;                       // a name or an address; resolved by WinHTTP without blocking the caller
    uint16_t port = 443;
    std::string path_base = "/ws/link/";
    std::string prefix;
    bool secure = true;                     // wss:// (false = ws://, for a peer on this machine)
    bool pinned = false;
    std::array<uint8_t, 32> pin{};          // SHA-256 of the server certificate's DER bytes
    int timeout_ms = 10000;
};

namespace detail {
inline std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(size_t(n), L'\0'); MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n); return w;
}
}  // namespace detail

// One WebSocket over WinHTTP. Callbacks arrive on WinHTTP threads (or inside a call); they only move bytes and flags
// under the core's lock. The core lives until its last handle has closed (HANDLE_CLOSING), so it outlives the stream.
class WinHttpStream final : public Stream {
    struct Core {
        std::mutex m;
        std::atomic<int> refs{1};
        enum class State { Connecting, Open, Closed } state = State::Connecting;
        const char* why = nullptr;
        HINTERNET request = nullptr, ws = nullptr;
        bool pinned = false; std::array<uint8_t, 32> pin{};
        bool send_busy = false; std::vector<uint8_t> sending;
        bool recv_busy = false; std::vector<uint8_t> rbuf, partial; std::deque<std::vector<uint8_t>> complete;
        size_t max_message = 0;
        void release() { if (--refs == 0) delete this; }
    };

public:
    WinHttpStream(const WssTarget& t, Channel channel, size_t max_message, size_t out_limit)
        : out_limit_(out_limit), deadline_(std::chrono::steady_clock::now() + std::chrono::milliseconds(t.timeout_ms)) {
        core_ = new Core; core_->pinned = t.pinned; core_->pin = t.pin; core_->max_message = max_message;
        core_->rbuf.resize(std::min<size_t>(max_message + 16, size_t(1) << 20));
        session_ = WinHttpOpen(L"nodus-link/1", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, WINHTTP_FLAG_ASYNC);
        if (!session_) { fail("WinHttpOpen refused"); return; }
        if (WinHttpSetStatusCallback(session_, &callback, WINHTTP_CALLBACK_FLAG_ALL_NOTIFICATIONS, 0) == WINHTTP_INVALID_STATUS_CALLBACK) { fail("status callback refused"); return; }
        connect_ = WinHttpConnect(session_, detail::widen(t.host).c_str(), t.port, 0);
        if (!connect_) { fail("WinHttpConnect refused"); return; }
        const std::wstring path = detail::widen(t.path_base + t.prefix + "/" + channel_name(channel));
        HINTERNET request = WinHttpOpenRequest(connect_, L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                               t.secure ? WINHTTP_FLAG_SECURE : 0);
        if (!request) { fail("WinHttpOpenRequest refused"); return; }
        DWORD_PTR ctx = reinterpret_cast<DWORD_PTR>(core_);
        WinHttpSetOption(request, WINHTTP_OPTION_CONTEXT_VALUE, &ctx, sizeof(ctx));
        ++core_->refs; core_->request = request;                  // released by the request's HANDLE_CLOSING
        bool ok = WinHttpSetOption(request, WINHTTP_OPTION_UPGRADE_TO_WEB_SOCKET, nullptr, 0) != FALSE;
        if (ok && t.secure && t.pinned) {
            DWORD flags = SECURITY_FLAG_IGNORE_UNKNOWN_CA;          // the pin decides instead of the CA chain
            ok = WinHttpSetOption(request, WINHTTP_OPTION_SECURITY_FLAGS, &flags, sizeof(flags)) != FALSE;
        }
        if (!ok || !WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, ctx)) fail("WinHttpSendRequest refused");
    }
    ~WinHttpStream() override {
        close_handles();
        if (connect_) WinHttpCloseHandle(connect_);
        if (session_) WinHttpCloseHandle(session_);
        core_->release();
    }
    WinHttpStream(const WinHttpStream&) = delete;
    WinHttpStream& operator=(const WinHttpStream&) = delete;

    bool open() const { std::lock_guard<std::mutex> g(core_->m); return core_->state == Core::State::Open; }
    const char* reason() const { std::lock_guard<std::mutex> g(core_->m); return core_->why; }
    bool closed() const override {
        std::lock_guard<std::mutex> g(core_->m);
        return core_->state == Core::State::Closed ||
               (core_->state == Core::State::Connecting && std::chrono::steady_clock::now() > deadline_);
    }
    bool queue(const uint8_t* a, size_t an, const uint8_t* b = nullptr, size_t bn = 0) override {
        if (an + bn > core_->max_message || out_bytes_ + an + bn > out_limit_ || closed()) return false;
        std::vector<uint8_t> m; m.reserve(an + bn);
        if (an) m.insert(m.end(), a, a + an);
        if (bn) m.insert(m.end(), b, b + bn);
        out_bytes_ += m.size(); out_.push_back(std::move(m)); return true;
    }
    void flush(size_t budget) override {
        for (size_t sent = 0; sent < budget && !out_.empty();) {
            HINTERNET ws = nullptr; const uint8_t* data = nullptr; size_t n = 0;
            {
                std::lock_guard<std::mutex> g(core_->m);
                if (core_->state != Core::State::Open || core_->send_busy) return;
                core_->sending = std::move(out_.front()); out_.pop_front();
                out_bytes_ -= core_->sending.size();
                core_->send_busy = true; ws = core_->ws; data = core_->sending.data(); n = core_->sending.size();
            }
            sent += n;
            if (WinHttpWebSocketSend(ws, WINHTTP_WEB_SOCKET_BINARY_MESSAGE_BUFFER_TYPE, const_cast<uint8_t*>(data), DWORD(n)) != NO_ERROR) {
                fail("WebSocket send refused"); return;
            }
        }
    }
    bool next(std::vector<uint8_t>& out, size_t) override {
        bool got = false; HINTERNET ws = nullptr; void* buf = nullptr; DWORD size = 0;
        {
            std::lock_guard<std::mutex> g(core_->m);
            if (!core_->complete.empty()) { out = std::move(core_->complete.front()); core_->complete.pop_front(); got = true; }
            // one receive in flight; at most four complete messages held, so a slow reader backpressures the peer
            if (core_->state == Core::State::Open && !core_->recv_busy && core_->complete.size() < 4) {
                core_->recv_busy = true; ws = core_->ws; buf = core_->rbuf.data(); size = DWORD(core_->rbuf.size());
            }
        }
        if (ws && WinHttpWebSocketReceive(ws, buf, size, nullptr, nullptr) != NO_ERROR) fail("WebSocket receive refused");
        return got;
    }
    void close(uint16_t) override { fail("closed by this side"); }

private:
    void fail(const char* why) {
        { std::lock_guard<std::mutex> g(core_->m); if (core_->state != Core::State::Closed) { core_->state = Core::State::Closed; if (!core_->why) core_->why = why; } }
        close_handles();
    }
    void close_handles() { close_core_handles(core_); }
    static void close_core_handles(Core* c) {
        HINTERNET ws, request;
        { std::lock_guard<std::mutex> g(c->m); ws = c->ws; request = c->request; c->ws = c->request = nullptr; c->state = Core::State::Closed; }
        if (ws) WinHttpCloseHandle(ws);
        if (request) WinHttpCloseHandle(request);
    }
    static void fail_core(Core* c, const char* why) {
        { std::lock_guard<std::mutex> g(c->m); if (!c->why) c->why = why; }
        close_core_handles(c);
    }
    static bool pin_matches(HINTERNET request, const std::array<uint8_t, 32>& pin) {
        PCCERT_CONTEXT cert = nullptr; DWORD size = sizeof(cert);
        if (!WinHttpQueryOption(request, WINHTTP_OPTION_SERVER_CERT_CONTEXT, &cert, &size) || !cert) return false;
        auto digest = sha256(cert->pbCertEncoded, cert->cbCertEncoded);
        CertFreeCertificateContext(cert);
        return equal_constant_time(digest.data(), pin.data(), 32);
    }
    static void CALLBACK callback(HINTERNET h, DWORD_PTR ctx, DWORD status, LPVOID info, DWORD length) {
        auto* c = reinterpret_cast<Core*>(ctx);
        if (!c) return;                                              // session and connect handles carry no context
        switch (status) {
        case WINHTTP_CALLBACK_STATUS_SENDREQUEST_COMPLETE:
            if (c->pinned && !pin_matches(h, c->pin)) { fail_core(c, "server certificate does not match the pin"); return; }
            if (!WinHttpReceiveResponse(h, nullptr)) fail_core(c, "receive response refused");
            break;
        case WINHTTP_CALLBACK_STATUS_HEADERS_AVAILABLE: {
            DWORD code = 0, size = sizeof(code);
            WinHttpQueryHeaders(h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &code, &size, WINHTTP_NO_HEADER_INDEX);
            if (code != 101) { fail_core(c, "upgrade refused"); return; }
            ++c->refs;                                               // released by the WebSocket handle's HANDLE_CLOSING
            HINTERNET ws = WinHttpWebSocketCompleteUpgrade(h, ctx);
            if (!ws) { c->release(); fail_core(c, "WebSocket upgrade refused"); return; }
            HINTERNET request;
            { std::lock_guard<std::mutex> g(c->m); request = c->request; c->request = nullptr; c->ws = ws; if (c->state == State::Connecting) c->state = State::Open; }
            if (request) WinHttpCloseHandle(request);                // the upgraded socket now belongs to `ws`
            break;
        }
        case WINHTTP_CALLBACK_STATUS_WRITE_COMPLETE: {
            std::lock_guard<std::mutex> g(c->m); c->send_busy = false; c->sending.clear(); break;
        }
        case WINHTTP_CALLBACK_STATUS_READ_COMPLETE: {
            auto* s = static_cast<WINHTTP_WEB_SOCKET_STATUS*>(info);
            std::lock_guard<std::mutex> g(c->m);
            c->recv_busy = false;
            if (!s || s->eBufferType == WINHTTP_WEB_SOCKET_CLOSE_BUFFER_TYPE) { c->state = State::Closed; if (!c->why) c->why = "peer closed"; break; }
            if (s->eBufferType != WINHTTP_WEB_SOCKET_BINARY_MESSAGE_BUFFER_TYPE && s->eBufferType != WINHTTP_WEB_SOCKET_BINARY_FRAGMENT_BUFFER_TYPE) {
                c->state = State::Closed; if (!c->why) c->why = "text message (protocol violation)"; break;
            }
            if (c->partial.size() + s->dwBytesTransferred > c->max_message) { c->state = State::Closed; if (!c->why) c->why = "message too large"; break; }
            c->partial.insert(c->partial.end(), c->rbuf.begin(), c->rbuf.begin() + ptrdiff_t(s->dwBytesTransferred));
            if (s->eBufferType == WINHTTP_WEB_SOCKET_BINARY_MESSAGE_BUFFER_TYPE) { c->complete.push_back(std::move(c->partial)); c->partial.clear(); }
            break;
        }
        case WINHTTP_CALLBACK_STATUS_REQUEST_ERROR: {
            auto* r = static_cast<WINHTTP_ASYNC_RESULT*>(info);
            fail_core(c, r && r->dwError == ERROR_WINHTTP_SECURE_FAILURE ? "TLS refused (certificate not trusted)" : "request error");
            break;
        }
        case WINHTTP_CALLBACK_STATUS_HANDLE_CLOSING: c->release(); break;
        default: break;
        }
        (void)length;
    }
    using State = Core::State;

    Core* core_ = nullptr;
    HINTERNET session_ = nullptr, connect_ = nullptr;
    std::deque<std::vector<uint8_t>> out_; size_t out_bytes_ = 0, out_limit_;
    std::chrono::steady_clock::time_point deadline_;
};

// Both sockets of one link through WinHTTP; the same tickets and Link as the loopback dialer.
class WssDialer {
public:
    WssDialer(const WssTarget& t, const TicketBytes& control_ticket, const TicketBytes& bulk_ticket)
        : tickets_{control_ticket, bulk_ticket} {
        std::memcpy(id_.data(), control_ticket.data() + 8, 16);    // unverified: the dialer holds no key
        account_ = get_le(control_ticket.data() + 56, 8);
        control_ = std::make_unique<WinHttpStream>(t, Channel::Control, kControlPacketBytes, kControlQueuePackets * kControlPacketBytes);
        bulk_ = std::make_unique<WinHttpStream>(t, Channel::Bulk, kBulkHeaderBytes + kBulkChunkBytes, kBulkWindowBytes + kBulkHeaderBytes + 16);
    }
    void poll() {
        if (failed_ || ready_ || taken_) return;
        for (auto* s : {control_.get(), bulk_.get()}) {
            if (s->closed()) { failed_ = true; reason_ = s->reason() ? s->reason() : "timed out"; return; }
        }
        if (control_->open() && bulk_->open()) {
            if (!control_->queue(tickets_[0].data(), kTicketBytes) || !bulk_->queue(tickets_[1].data(), kTicketBytes)) { failed_ = true; reason_ = "ticket refused by own queue"; return; }
            control_->flush(kTicketBytes); bulk_->flush(kTicketBytes);
            ready_ = true;
        }
    }
    bool ready() const { return ready_ && !taken_; }
    bool failed() const { return failed_; }
    const char* reason() const { return reason_; }
    std::unique_ptr<Link> take() {
        if (!ready()) return nullptr;
        taken_ = true;
        return std::make_unique<Link>(std::move(control_), std::move(bulk_), Role::Dialer, account_, id_);
    }

private:
    std::array<TicketBytes, 2> tickets_;
    std::unique_ptr<WinHttpStream> control_, bulk_;
    LinkId id_{}; uint64_t account_ = 0;
    bool ready_ = false, failed_ = false, taken_ = false;
    const char* reason_ = nullptr;
};

}  // namespace nodus::link
