// link_test: stage 1 of the Nodus link (research case 98d49cb8520e44d092c5d134c37c3c32). A LinkListener and a dialer
// on loopback in one process: the four FIFOs both ways, bulk integrity and throughput, control not held back by bulk,
// and every ticket refusal. No Apache, no TLS, no canvas: those are later stages.
#include "nodus_link.h"
#include <chrono>
#include <cstdio>
#include <ctime>

using namespace nodus::link;
using Clock = std::chrono::steady_clock;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d  " #cond "  ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static uint64_t now_s() { return uint64_t(std::time(nullptr)); }
static const std::string kPrefix = "lab";

static void fill(std::vector<uint8_t>& v, uint64_t seed) {
    uint64_t x = seed | 1;
    for (size_t i = 0; i < v.size(); ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; v[i] = uint8_t(x >> 24); }
}

struct Tickets { TicketBytes control, bulk; };
static Tickets tickets_for(const Key& key, const LinkId& id, uint64_t account, const std::string& prefix = kPrefix, uint64_t issued = 0, uint64_t life = 60) {
    uint64_t t = issued ? issued : now_s();
    return {sign_ticket(make_ticket(Channel::Control, Role::Dialer, id, prefix, account, t, life), key),
            sign_ticket(make_ticket(Channel::Bulk, Role::Dialer, id, prefix, account, t, life), key)};
}

// One thread: the dialer and the listener are both polled until the dialer is ready or refused.
static std::unique_ptr<Link> dial(LinkListener& l, const TicketBytes& c, const TicketBytes& b, const std::string& prefix = kPrefix) {
    DialTarget t; t.port = l.port(); t.prefix = prefix; t.timeout_ms = 3000;
    LinkDialer d(t, c, b);
    while (!d.ready() && !d.failed()) { d.poll(); l.poll(now_s()); }
    if (d.failed()) std::printf("  dialer: %s\n", d.reason());
    return d.take();
}
static std::unique_ptr<Link> wait_take(LinkListener& l, Link* dialer, int ms = 2000) {
    auto end = Clock::now() + std::chrono::milliseconds(ms);
    while (Clock::now() < end) {
        l.poll(now_s()); if (dialer) dialer->poll();
        if (auto h = l.take()) return h;
    }
    return nullptr;
}
static void pump(LinkListener& l, Link* a, Link* b, int ms) {
    auto end = Clock::now() + std::chrono::milliseconds(ms);
    while (Clock::now() < end) { l.poll(now_s()); if (a) a->poll(); if (b) b->poll(); }
}

static void control_both_ways(Link& from, Link& to, LinkListener& l, uint32_t count, const char* label) {
    uint32_t sent = 0, got = 0; bool order = true;
    auto end = Clock::now() + std::chrono::seconds(10);
    while ((sent < count || got < count) && Clock::now() < end) {
        while (sent < count) {
            ControlPacket p; p.kind = uint16_t(ControlKind::Instruction); p.seq = sent; p.length = 8;
            put_le(p.payload.data(), uint64_t(sent) * 2654435761u, 8);
            if (!from.send_control(p)) break;
            ++sent;
        }
        l.poll(now_s()); from.poll(); to.poll();
        ControlPacket r;
        while (to.receive_control(r)) {
            if (r.seq != got || get_le(r.payload.data(), 8) != uint64_t(got) * 2654435761u || r.length != 8) order = false;
            ++got;
        }
    }
    CHECK(got == count && order, "%s control: %u of %u in order=%d", label, got, count, int(order));
    std::printf("control %s: %u packets, in order\n", label, got);
}

// Sends `bytes` on bulk with a descriptor; part way it sends one control packet and records whether it arrived
// before the stream ended (control must not wait behind bulk).
static void bulk_one_way(Link& from, Link& to, LinkListener& l, size_t bytes, uint64_t seed, const char* label) {
    std::vector<uint8_t> src(bytes); fill(src, seed);
    const char desc[] = "dtype=u8 shape=[n]";
    uint64_t stream = 0;
    auto end = Clock::now() + std::chrono::seconds(60);
    while (!stream && Clock::now() < end) { stream = from.begin_bulk(reinterpret_cast<const uint8_t*>(desc), sizeof(desc), bytes); from.poll(); }
    CHECK(stream != 0, "%s begin refused", label);
    std::vector<uint8_t> dst; std::string got_desc; bool began = false, ended = false, control_sent = false, control_before_end = false;
    size_t written = 0;
    auto t0 = Clock::now();
    while (!ended && Clock::now() < end) {
        if (written < bytes) written += from.write_bulk(stream, src.data() + written, bytes - written);
        else if (from.end_bulk(stream)) written = bytes + 1;           // End queued once
        if (!control_sent && written > bytes / 4) {
            ControlPacket p; p.kind = uint16_t(ControlKind::Metric); p.seq = 777; p.length = 0;
            control_sent = from.send_control(p);
        }
        l.poll(now_s()); from.poll(); to.poll();
        ControlPacket r; while (to.receive_control(r)) if (r.seq == 777 && !ended) control_before_end = true;
        BulkEvent e;
        while (to.receive_bulk(e)) {
            if (e.kind == BulkKind::Begin) { began = true; got_desc.assign(e.data.begin(), e.data.end()); dst.reserve(size_t(e.total)); CHECK(e.stream == stream, "stream id"); }
            else if (e.kind == BulkKind::Chunk) { CHECK(e.offset == dst.size(), "%s chunk order", label); dst.insert(dst.end(), e.data.begin(), e.data.end()); }
            else if (e.kind == BulkKind::End) ended = true;
            else CHECK(false, "%s unexpected abort", label);
        }
    }
    double s = std::chrono::duration<double>(Clock::now() - t0).count();
    CHECK(began && ended, "%s began=%d ended=%d", label, int(began), int(ended));
    CHECK(got_desc == std::string(desc, sizeof(desc)), "%s descriptor", label);
    CHECK(dst.size() == src.size() && std::memcmp(dst.data(), src.data(), src.size()) == 0, "%s bytes differ (%zu of %zu)", label, dst.size(), src.size());
    CHECK(control_before_end, "%s control packet waited behind bulk", label);
    std::printf("bulk %s: %zu bytes identical in %.3f s (%.0f MB/s), control arrived mid-stream=%d\n",
                label, bytes, s, bytes / s / 1e6, int(control_before_end));
}

static void two_streams(Link& from, Link& to, LinkListener& l) {
    std::vector<uint8_t> a(5u << 20), b(3u << 20); fill(a, 11); fill(b, 12);
    uint64_t sa = from.begin_bulk(nullptr, 0, a.size()), sb = from.begin_bulk(nullptr, 0, b.size());
    CHECK(sa && sb && sa != sb, "two begins");
    size_t wa = 0, wb = 0; bool ea = false, eb = false;
    std::map<uint64_t, std::vector<uint8_t>> got; int ends = 0;
    auto end = Clock::now() + std::chrono::seconds(30);
    while (ends < 2 && Clock::now() < end) {
        if (wa < a.size()) wa += from.write_bulk(sa, a.data() + wa, std::min<size_t>(a.size() - wa, 300000));
        if (wb < b.size()) wb += from.write_bulk(sb, b.data() + wb, std::min<size_t>(b.size() - wb, 300000));
        if (wa == a.size() && !ea) ea = from.end_bulk(sa);
        if (wb == b.size() && !eb) eb = from.end_bulk(sb);
        l.poll(now_s()); from.poll(); to.poll();
        BulkEvent e;
        while (to.receive_bulk(e)) {
            if (e.kind == BulkKind::Chunk) got[e.stream].insert(got[e.stream].end(), e.data.begin(), e.data.end());
            if (e.kind == BulkKind::End) ++ends;
        }
    }
    CHECK(ends == 2 && got[sa] == a && got[sb] == b, "interleaved streams ends=%d", ends);
    std::printf("bulk interleaved: two streams (%zu, %zu bytes) identical\n", a.size(), b.size());
}

// A dial that the listener must refuse: no Link is handed out, the counter moves, the dialer's link closes.
static void refused(LinkListener& l, const char* label, const TicketBytes& c, const TicketBytes& b, uint64_t ListenerCounters::*counter,
                    const std::string& prefix = kPrefix) {
    uint64_t before = l.counters().*counter;
    auto d = dial(l, c, b, prefix);
    if (!d) {   // refused at the HTTP upgrade
        pump(l, nullptr, nullptr, 50);
    } else {
        pump(l, d.get(), nullptr, 300);
    }
    auto h = l.take();
    CHECK(!h, "%s: a link was handed out", label);
    CHECK(l.counters().*counter > before, "%s: counter did not move", label);
    CHECK(!d || d->closed(), "%s: dialer link still open", label);
    std::printf("refused %s\n", label);
}

// `link_test vector`: one ticket from fixed fields, as hex, the known answer other issuers (turing link_tickets.py)
// must reproduce byte for byte. key = 00..1f, link id = 10..1f, prefix "lab", account 42, issued 1795000000,
// expires 1795000120, nonce = a0..af, channel control, role dialer.
static int print_vector() {
    Key key{}; for (size_t i = 0; i < key.size(); ++i) key[i] = uint8_t(i);
    Ticket t; t.channel = Channel::Control; t.role = Role::Dialer;
    for (size_t i = 0; i < 16; ++i) { t.link_id[i] = uint8_t(0x10 + i); t.nonce[i] = uint8_t(0xA0 + i); }
    t.prefix_hash = prefix_hash("lab"); t.account = 42; t.issued_at = 1795000000; t.expires_at = 1795000120;
    auto b = sign_ticket(t, key);
    for (auto x : b) std::printf("%02x", x);
    std::printf("\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "vector") == 0) return print_vector();
    Key key{}; random_bytes(key.data(), key.size());
    ListenerConfig cfg; cfg.prefix = kPrefix; cfg.key = key;
    LinkListener listener(cfg);
    std::printf("listener 127.0.0.1:%u prefix %s\n", listener.port(), kPrefix.c_str());

    LinkId id = new_link_id();
    Tickets good = tickets_for(key, id, 42);
    auto dialer = dial(listener, good.control, good.bulk);
    CHECK(dialer != nullptr, "dial refused");
    if (!dialer) return 1;
    auto host = wait_take(listener, dialer.get());
    CHECK(host != nullptr, "no link handed out");
    if (!host) return 1;
    CHECK(host->account() == 42 && host->id() == id && dialer->id() == id, "identity");
    CHECK(host->role() == Role::Host && dialer->role() == Role::Dialer, "roles");
    std::printf("link up: account %llu\n", static_cast<unsigned long long>(host->account()));

    control_both_ways(*dialer, *host, listener, 5000, "dialer->host");
    control_both_ways(*host, *dialer, listener, 5000, "host->dialer");
    bulk_one_way(*dialer, *host, listener, size_t(64) << 20, 1, "dialer->host");
    bulk_one_way(*host, *dialer, listener, size_t(3) << 20, 2, "host->dialer");
    two_streams(*dialer, *host, listener);
    CHECK(!host->closed() && !dialer->closed(), "link closed during traffic");
    CHECK(host->stats().protocol_violations == 0 && dialer->stats().protocol_violations == 0, "violations");
    const auto& hs = host->stats();
    std::printf("host stats: control in %llu out %llu, bulk in %llu B / %llu streams, read stalls control %llu bulk %llu\n",
                (unsigned long long)hs.control_received, (unsigned long long)hs.control_sent, (unsigned long long)hs.bulk_bytes_received,
                (unsigned long long)hs.bulk_streams_received, (unsigned long long)hs.control_read_stalls, (unsigned long long)hs.bulk_read_stalls);

    Key wrong{}; random_bytes(wrong.data(), wrong.size());
    { auto t = tickets_for(wrong, new_link_id(), 42); refused(listener, "wrong key", t.control, t.bulk, &ListenerCounters::refused_ticket); }
    { auto t = tickets_for(key, new_link_id(), 42, kPrefix, now_s() - 400, 60); refused(listener, "expired", t.control, t.bulk, &ListenerCounters::refused_ticket); }
    { auto t = tickets_for(key, new_link_id(), 42, kPrefix, now_s(), 3600); refused(listener, "life too long", t.control, t.bulk, &ListenerCounters::refused_ticket); }
    { auto t = tickets_for(key, new_link_id(), 42, "other"); refused(listener, "wrong prefix", t.control, t.bulk, &ListenerCounters::refused_ticket); }
    { auto t = tickets_for(key, new_link_id(), 42); refused(listener, "control ticket on bulk", t.control, t.control, &ListenerCounters::refused_ticket); }
    refused(listener, "replayed tickets", good.control, good.bulk, &ListenerCounters::refused_replay);
    {
        LinkId other = new_link_id(); uint64_t t0 = now_s();
        auto c = sign_ticket(make_ticket(Channel::Control, Role::Dialer, other, kPrefix, 42, t0, 60), key);
        auto b = sign_ticket(make_ticket(Channel::Bulk, Role::Dialer, other, kPrefix, 43, t0, 60), key);
        refused(listener, "halves of two accounts", c, b, &ListenerCounters::refused_pair);
    }
    { auto t = tickets_for(key, new_link_id(), 42); refused(listener, "unknown path", t.control, t.bulk, &ListenerCounters::refused_handshake, "nope"); }

    {   // Nobody listening: the dialer reports the refusal from poll(), each poll returning at once.
        DialTarget t; t.port = 1; t.prefix = kPrefix; t.timeout_ms = 3000;
        auto tk = tickets_for(key, new_link_id(), 42);
        LinkDialer d(t, tk.control, tk.bulk);
        double worst = 0; auto t0 = Clock::now();
        while (!d.ready() && !d.failed()) {
            auto a = Clock::now(); d.poll(); worst = std::max(worst, std::chrono::duration<double>(Clock::now() - a).count());
        }
        double total = std::chrono::duration<double>(Clock::now() - t0).count();
        CHECK(d.failed() && worst < 0.01, "closed port: failed=%d worst poll %.4f s", int(d.failed()), worst);
        std::printf("refused closed port: %s after %.3f s, longest poll %.6f s\n", d.reason(), total, worst);
    }

    // The first link is untouched by the refusals.
    control_both_ways(*dialer, *host, listener, 100, "after refusals");
    dialer->close();
    pump(listener, dialer.get(), host.get(), 200);
    CHECK(host->closed(), "host did not see the close");

    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
