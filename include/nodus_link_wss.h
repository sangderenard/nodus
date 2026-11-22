#pragma once
// Nodus link, stage 4: dial out over wss:// (or ws://). Windows: WinHTTP in asynchronous mode; the OS does TLS and name
// resolution on its own threads. POSIX: nonblocking TCP + OpenSSL. Every call here returns at once and WssDialer/Link advance by poll(), like the
// loopback LinkDialer. Research case 98d49cb8520e44d092c5d134c37c3c32 (ticket bc214e177933475e9f8ac3f5ed3e3762).
//
// The public path is <path_base><prefix>/{control,bulk} (default /ws/link/), which Apache proxies to the native
// LinkListener on 127.0.0.1. Server certificates are validated by Windows unless a SHA-256 pin of the server's
// certificate (DER) is given: then only that exact certificate is accepted (a self-hosted peer without a public CA).
// The pin is checked when the TLS handshake completes, before the upgrade answer is read and before any ticket is sent.
// Windows: WinHTTP does the pin check (link with winhttp.lib and crypt32.lib). POSIX: OpenSSL libssl/libcrypto (-lssl -lcrypto) does it
// after the TLS handshake with the same ordering (pin, then upgrade request, then tickets); system CA store otherwise.
#include "nodus_link.h"

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

}  // namespace nodus::link

// ================================ platform region: Windows (WinHTTP) / POSIX (OpenSSL TLS client) ================================
// WinHttpStream or TlsStream, whichever the platform has, is PlatformWssStream: a Stream with open(), reason(), advance().
#if defined(_WIN32)
#include <winhttp.h>
#include <wincrypt.h>

#include <atomic>
#include <mutex>

namespace nodus::link {

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
            // the pin decides instead of the CA chain and the name, as the OpenSSL lane skips SSL_set1_host when pinned
            DWORD flags = SECURITY_FLAG_IGNORE_UNKNOWN_CA | SECURITY_FLAG_IGNORE_CERT_CN_INVALID;
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
    void advance() {}                                          // WinHTTP runs on its own threads; nothing to poll
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
using PlatformWssStream = WinHttpStream;

}  // namespace nodus::link
#else
#include <future>
#include <climits>
#include <netinet/tcp.h>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/crypto.h>

namespace nodus::link {

// The bytes of one connection, plain (ws://) or TLS (wss://), without blocking. TLS runs over two memory BIOs: this class
// does the recv/send itself (send with MSG_NOSIGNAL, so a closed peer is an error return and not SIGPIPE). It does not
// own the socket. Return codes match WsStream::io_send/io_recv: bytes moved, kAgain, 0 = closed, kError.
class TlsConn {
public:
    static constexpr long kAgain = -1, kError = -2;
    explicit TlsConn(SOCKET s) : s_(s) {}
    ~TlsConn() {
        if (ssl_) {
            if (established_) { ERR_clear_error(); SSL_shutdown(ssl_); flush(); }   // best effort close_notify
            SSL_free(ssl_);                                                          // frees both BIOs
        }
    }
    TlsConn(const TlsConn&) = delete;
    TlsConn& operator=(const TlsConn&) = delete;

    // Prepare the client side of TLS: SNI = host, and unless pinned the system CA store plus host name verification.
    // nullptr when ready, otherwise the reason.
    const char* begin_tls(const std::string& host, bool pinned) {
        SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
        if (!ctx) return "TLS refused (no TLS context)";
        SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
        if (pinned) SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);              // the pin decides instead of the CA chain
        else {
            SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
            if (SSL_CTX_set_default_verify_paths(ctx) != 1) { SSL_CTX_free(ctx); return "TLS refused (no system CA store)"; }
        }
        ssl_ = SSL_new(ctx); SSL_CTX_free(ctx);
        if (!ssl_) return "TLS refused (no TLS session)";
        BIO* rbio = BIO_new(BIO_s_mem()); BIO* wbio = BIO_new(BIO_s_mem());
        if (!rbio || !wbio) { if (rbio) BIO_free(rbio); if (wbio) BIO_free(wbio); return "TLS refused (no BIO)"; }
        SSL_set_bio(ssl_, rbio, wbio); wbio_ = wbio;                                // ssl_ owns both from here
        SSL_set_connect_state(ssl_);
        unsigned char probe[sizeof(in6_addr)];
        const bool numeric = inet_pton(AF_INET, host.c_str(), probe) == 1 || inet_pton(AF_INET6, host.c_str(), probe) == 1;
        if (numeric) {
            if (!pinned && X509_VERIFY_PARAM_set1_ip_asc(SSL_get0_param(ssl_), host.c_str()) != 1) return "TLS refused (bad address)";
        } else {
            if (SSL_set_tlsext_host_name(ssl_, host.c_str()) != 1) return "TLS refused (bad host name)";
            if (!pinned && SSL_set1_host(ssl_, host.c_str()) != 1) return "TLS refused (bad host name)";
        }
        return nullptr;
    }
    // 1 established, 0 in progress, -1 refused.
    int handshake() {
        for (int i = 0; i < 16; ++i) {
            ERR_clear_error();
            int r = SSL_do_handshake(ssl_);
            if (!flush()) return -1;
            if (r == 1) { established_ = true; return 1; }
            int e = SSL_get_error(ssl_, r);
            if (e == SSL_ERROR_WANT_READ) { long got = fill(); if (got == kAgain) return 0; if (got <= 0) return -1; continue; }
            if (e == SSL_ERROR_WANT_WRITE) continue;
            return -1;
        }
        return 0;
    }
    // The server's certificate (DER) must hash to `pin`.
    bool pin_matches(const std::array<uint8_t, 32>& pin) const {
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
        X509* cert = SSL_get1_peer_certificate(ssl_);
#else
        X509* cert = SSL_get_peer_certificate(ssl_);
#endif
        if (!cert) return false;
        unsigned char* der = nullptr; int n = i2d_X509(cert, &der);
        X509_free(cert);
        if (n <= 0 || !der) return false;
        auto digest = sha256(der, size_t(n));
        OPENSSL_free(der);
        return equal_constant_time(digest.data(), pin.data(), 32);
    }
    // Encrypted bytes the socket did not take yet; false when the socket failed.
    bool flush() {
        if (wbio_) {
            char buf[16384];
            for (;;) {
                if (BIO_ctrl_pending(wbio_) == 0) break;
                int k = BIO_read(wbio_, buf, int(sizeof(buf)));
                if (k <= 0) break;
                out_.insert(out_.end(), buf, buf + k);
            }
        }
        while (out_pos_ < out_.size()) {
            ssize_t r = ::send(s_, out_.data() + out_pos_, out_.size() - out_pos_, platform::kSendFlags);
            if (r < 0) return WireSocket::would_block();
            if (r == 0) return false;
            out_pos_ += size_t(r);
        }
        out_.clear(); out_pos_ = 0; return true;
    }
    long send(const uint8_t* p, size_t n) {
        if (!ssl_) {
            ssize_t r = ::send(s_, p, n, platform::kSendFlags);
            if (r < 0) return WireSocket::would_block() ? kAgain : kError;
            return long(r);
        }
        if (!flush()) return kError;
        if (out_.size() - out_pos_ > (size_t(1) << 18)) return kAgain;               // the socket is behind: backpressure
        ERR_clear_error();
        int r = SSL_write(ssl_, p, int(std::min<size_t>(n, 16384)));                // one TLS record
        if (r <= 0) {
            int e = SSL_get_error(ssl_, r); flush();
            return (e == SSL_ERROR_WANT_READ || e == SSL_ERROR_WANT_WRITE) ? kAgain : kError;
        }
        return flush() ? long(r) : kError;
    }
    long recv(char* p, size_t n) {
        if (!ssl_) {
            ssize_t r = ::recv(s_, p, n, 0);
            if (r < 0) return WireSocket::would_block() ? kAgain : kError;
            return long(r);
        }
        for (int i = 0; i < 64; ++i) {
            ERR_clear_error();
            int r = SSL_read(ssl_, p, int(std::min<size_t>(n, size_t(INT_MAX))));
            if (r > 0) { flush(); return r; }
            int e = SSL_get_error(ssl_, r);
            if (e == SSL_ERROR_WANT_READ) {
                long got = fill();
                if (got == kAgain) { flush(); return kAgain; }
                if (got == 0) return 0;
                if (got < 0) return kError;
                continue;
            }
            if (e == SSL_ERROR_ZERO_RETURN) return 0;
            return kError;
        }
        return kAgain;
    }

private:
    // One socket read into the TLS input; bytes, kAgain, 0 closed, kError.
    long fill() {
        char buf[16384];
        ssize_t r = ::recv(s_, buf, sizeof(buf), 0);
        if (r < 0) return WireSocket::would_block() ? kAgain : kError;
        if (r == 0) return 0;
        if (BIO_write(SSL_get_rbio(ssl_), buf, int(r)) != int(r)) return kError;
        return long(r);
    }
    SOCKET s_;
    SSL* ssl_ = nullptr; BIO* wbio_ = nullptr;
    bool established_ = false;
    std::vector<char> out_; size_t out_pos_ = 0;
};

// WsStream's framing over a TlsConn: the one framer of nodus_link.h, with TLS under it.
class ConnWsStream final : public WsStream {
public:
    ConnWsStream(SOCKET s, std::unique_ptr<TlsConn> conn, size_t max_message, size_t out_limit, std::vector<uint8_t> leftover)
        : WsStream(s, true, max_message, out_limit, std::move(leftover)), conn_(std::move(conn)) {}
    void flush(size_t budget) override { conn_->flush(); WsStream::flush(budget); conn_->flush(); }
protected:
    long io_send(const uint8_t* p, size_t n) override { return conn_->send(p, n); }
    long io_recv(char* p, size_t n) override { return conn_->recv(p, n); }
private:
    std::unique_ptr<TlsConn> conn_;   // destroyed before the base closes the socket: close_notify goes out first
};
static_assert(TlsConn::kAgain == -1 && TlsConn::kError == -2, "TlsConn codes are WsStream's io codes");

// One WebSocket client over TCP + OpenSSL, driven by poll: name resolution on a std::async thread, then nonblocking
// connect, TLS handshake (certificate pin checked before the upgrade request is sent), HTTP upgrade, then ConnWsStream.
// advance() moves it along; flush/next/closed callers need not call it, they do. Every call returns at once.
class TlsStream final : public Stream {
public:
    TlsStream(const WssTarget& t, Channel channel, size_t max_message, size_t out_limit)
        : t_(t), max_message_(max_message),
          out_limit_(out_limit + 8 * (out_limit / (max_message ? max_message : 1) + 1)),   // WsStream counts frame headers + mask
          deadline_(std::chrono::steady_clock::now() + std::chrono::milliseconds(t.timeout_ms)) {
        path_ = t.path_base + t.prefix + "/" + channel_name(channel);
        const std::string host = t.host; const uint16_t port = t.port;
        resolving_ = std::async(std::launch::async, [host, port] { return resolve(host, port); });
    }
    ~TlsStream() override {
        ws_.reset(); conn_.reset();
        if (sock_ != INVALID_SOCKET) WireSocket::close_socket(sock_);
    }
    TlsStream(const TlsStream&) = delete;
    TlsStream& operator=(const TlsStream&) = delete;

    bool open() const { return state_ == State::Open && ws_ && !ws_->closed(); }
    const char* reason() const { return why_; }
    bool closed() const override {
        if (state_ == State::Closed) return true;
        if (state_ == State::Open) return !ws_ || ws_->closed();
        return std::chrono::steady_clock::now() > deadline_;
    }
    void advance() {
        if (state_ == State::Open || state_ == State::Closed) return;
        if (std::chrono::steady_clock::now() > deadline_) { fail("timed out"); return; }
        if (state_ == State::Resolving) {
            if (resolving_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
            Resolved r = resolving_.get();
            if (!r.ok) { fail("host did not resolve"); return; }
            begin_connect(r);
        }
        if (state_ == State::Connecting) {
            int c = platform::connect_state(sock_);
            if (c == 0) return;
            if (c < 0) { fail("connect refused"); return; }
            conn_ = std::make_unique<TlsConn>(sock_);
            if (t_.secure) {
                if (const char* why = conn_->begin_tls(t_.host, t_.pinned)) { fail(why); return; }
                state_ = State::Handshake;
            } else { make_request(); state_ = State::Send; }
        }
        if (state_ == State::Handshake) {
            int h = conn_->handshake();
            if (h == 0) return;
            if (h < 0) { fail("TLS refused (certificate not trusted)"); return; }
            if (t_.pinned && !conn_->pin_matches(t_.pin)) { fail("server certificate does not match the pin"); return; }
            make_request(); state_ = State::Send;
        }
        if (state_ == State::Send) {
            while (sent_ < request_.size()) {
                long n = conn_->send(reinterpret_cast<const uint8_t*>(request_.data()) + sent_, request_.size() - sent_);
                if (n == TlsConn::kAgain) return;
                if (n <= 0) { fail("send refused"); return; }
                sent_ += size_t(n);
            }
            state_ = State::Read;
        }
        if (state_ == State::Read) {
            if (!conn_->flush()) { fail("send refused"); return; }
            char buf[2048];
            long n = conn_->recv(buf, sizeof(buf));
            if (n == TlsConn::kAgain) return;
            if (n <= 0) { fail("upgrade refused"); return; }
            head_.append(buf, size_t(n));
            size_t end = head_.find("\r\n\r\n");
            if (end == std::string::npos) { if (head_.size() > 8192) fail("upgrade answer too long"); return; }
            std::vector<uint8_t> leftover(head_.begin() + ptrdiff_t(end + 4), head_.end());
            head_.resize(end + 2);
            std::string first; std::map<std::string, std::string> hd;
            if (!detail::parse_http(head_, first, hd) || first.rfind("HTTP/1.1 101", 0) != 0 || hd["sec-websocket-accept"] != websocket_accept(key_)) {
                fail("upgrade refused"); return;
            }
            SOCKET s = sock_; sock_ = INVALID_SOCKET;
            ws_ = std::make_unique<ConnWsStream>(s, std::move(conn_), max_message_, out_limit_, std::move(leftover));
            state_ = State::Open;
        }
    }
    bool queue(const uint8_t* a, size_t an, const uint8_t* b = nullptr, size_t bn = 0) override {
        return ws_ && state_ == State::Open && ws_->queue(a, an, b, bn);
    }
    // budget counts message bytes (as WinHTTP does); WsStream counts frame bytes, so a message gets its header and mask on top.
    void flush(size_t budget) override { advance(); if (ws_) ws_->flush(budget + 16); }
    bool next(std::vector<uint8_t>& out, size_t budget) override { advance(); return ws_ && ws_->next(out, budget); }
    void close(uint16_t code) override {
        if (ws_) ws_->close(code); else fail("closed by this side");
    }

private:
    enum class State { Resolving, Connecting, Handshake, Send, Read, Open, Closed };
    struct Resolved { bool ok = false; sockaddr_storage addr{}; socklen_t len = 0; int family = 0; };
    // Blocking: runs on the async thread. Prefers an IPv4 address, else the first one.
    static Resolved resolve(const std::string& host, uint16_t port) {
        addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM; hints.ai_flags = AI_NUMERICSERV;
        addrinfo* list = nullptr; Resolved out;
        const std::string service = std::to_string(port);
        if (getaddrinfo(host.c_str(), service.c_str(), &hints, &list) == 0 && list) {
            const addrinfo* pick = list;
            for (const addrinfo* a = list; a; a = a->ai_next) if (a->ai_family == AF_INET) { pick = a; break; }
            if (pick->ai_addrlen <= sizeof(out.addr)) {
                std::memcpy(&out.addr, pick->ai_addr, pick->ai_addrlen);
                out.len = socklen_t(pick->ai_addrlen); out.family = pick->ai_family; out.ok = true;
            }
        }
        if (list) freeaddrinfo(list);
        return out;
    }
    void fail(const char* why) {
        if (!why_) why_ = why;
        state_ = State::Closed;
        ws_.reset(); conn_.reset();
        if (sock_ != INVALID_SOCKET) { WireSocket::close_socket(sock_); sock_ = INVALID_SOCKET; }
    }
    void begin_connect(const Resolved& r) {
        sock_ = socket(r.family, SOCK_STREAM, IPPROTO_TCP);
        if (sock_ == INVALID_SOCKET || !WireSocket::set_nonblocking(sock_)) { fail("socket refused"); return; }
        int on = 1; setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
        if (connect(sock_, reinterpret_cast<const sockaddr*>(&r.addr), r.len) != 0 && !platform::connect_would_block()) { fail("connect refused"); return; }
        state_ = State::Connecting;
    }
    void make_request() {
        uint8_t raw[16]; random_bytes(raw, sizeof(raw)); key_ = base64(raw, sizeof(raw));
        std::string host = t_.host.find(':') != std::string::npos ? "[" + t_.host + "]" : t_.host;
        if (t_.port != (t_.secure ? 443 : 80)) host += ":" + std::to_string(t_.port);
        request_ = "GET " + path_ + " HTTP/1.1\r\nHost: " + host + "\r\nUser-Agent: nodus-link/1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: "
            + key_ + "\r\nSec-WebSocket-Version: 13\r\n\r\n";
    }

    WssTarget t_;
    size_t max_message_, out_limit_;
    std::chrono::steady_clock::time_point deadline_;
    std::string path_, request_, head_, key_; size_t sent_ = 0;
    std::future<Resolved> resolving_;
    State state_ = State::Resolving;
    const char* why_ = nullptr;
    SOCKET sock_ = INVALID_SOCKET;
    std::unique_ptr<TlsConn> conn_;
    std::unique_ptr<ConnWsStream> ws_;
};
using PlatformWssStream = TlsStream;

}  // namespace nodus::link
#endif
// ============================================ end of the platform region ============================================

namespace nodus::link {

// Both sockets of one link through WinHTTP; the same tickets and Link as the loopback dialer.
class WssDialer {
public:
    WssDialer(const WssTarget& t, const TicketBytes& control_ticket, const TicketBytes& bulk_ticket)
        : tickets_{control_ticket, bulk_ticket} {
        std::memcpy(id_.data(), control_ticket.data() + 8, 16);    // unverified: the dialer holds no key
        account_ = get_le(control_ticket.data() + 56, 8);
        control_ = std::make_unique<PlatformWssStream>(t, Channel::Control, kControlPacketBytes, kControlQueuePackets * kControlPacketBytes);
        bulk_ = std::make_unique<PlatformWssStream>(t, Channel::Bulk, kBulkHeaderBytes + kBulkChunkBytes, kBulkWindowBytes + kBulkHeaderBytes + 16);
    }
    void poll() {
        if (failed_ || ready_ || taken_) return;
        control_->advance(); bulk_->advance();                      // POSIX: the poll drives connect, TLS and upgrade; WinHTTP: no-op
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
    std::unique_ptr<PlatformWssStream> control_, bulk_;
    LinkId id_{}; uint64_t account_ = 0;
    bool ready_ = false, failed_ = false, taken_ = false;
    const char* reason_ = nullptr;
};

}  // namespace nodus::link
