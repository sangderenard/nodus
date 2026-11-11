// Compile host attached through the headless ABI (nodus_headless_attach_compile_host).
// No Turing compile runs here: the python executable is a path that does not exist,
// so any attempt to spawn the compiler would show up as CompileFailed.
// Winsock headers must precede anything that may pull in <windows.h>.
#include "contract_socket.h"
#include "contract_listener.h"
#include "nodus_headless_abi.h"
#include "contract_negotiator_registry.h"
#include "turing_compile_host_tool.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace nodus::contracts;
namespace fs = std::filesystem;
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "REQUIRE failed %s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while (0)

static const char* kPrefix = "abi.compile";
static const uint64_t kHost = 7;

struct Rig {
    ContractNegotiator* neg = nullptr;
    std::deque<Result> fifo;
    std::vector<std::pair<uint64_t, uint64_t>> routed;
    std::vector<uint64_t> to_submit;
    size_t next = 0;
    void bind(uint64_t engine) {
        neg = find(kPrefix);
        REQUIRE(neg);
        REQUIRE(neg->bind_result_transport(
            [this](const Result& r) { fifo.push_back(r); return true; },
            [this](Result& r) { if (fifo.empty()) return false; r = fifo.front(); fifo.pop_front(); return true; }));
        REQUIRE(neg->add_engine({engine,
            [this](const Result& r) { routed.push_back({r.loan.work.id, r.payload}); return true; },
            [this, engine](Contract& c) { if (next >= to_submit.size()) return false; c = {to_submit[next], engine, 1, 5}; ++next; return true; }}));
    }
    void run(size_t want) {
        const auto t0 = std::chrono::steady_clock::now();
        while (routed.size() < want && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) {
            neg->worker_turn(); neg->market_turn();
        }
        REQUIRE(routed.size() == want);
    }
};

static int32_t attach(const char* prefix, uint64_t id, const char* py, const char* nodus,
                      const char* turing, const char* lib, const char* jobs, int32_t adopt = 0) {
    return nodus_headless_attach_compile_host(prefix, id, py, nodus, turing, lib, jobs, adopt);
}

// ---- Real loopback contract port + attached compile host, one engine connection per contract ----
// Mirrors the live canvas host: pair acquired with loan_ticks 54000, one compile host, engines that
// connect with the SAME id, Submit, wait for Route, send Decision, close. Every request file is
// missing so the host finishes each job at once (RequestMissing) and no compiler runs.
static SOCKET lb_connect(uint16_t port) {
    ensure_sockets();
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    REQUIRE(s != INVALID_SOCKET);
    sockaddr_in a{};
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(port);
    REQUIRE(connect(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0);
    return s;
}
static WireMessage lb_hello(uint64_t role, uint64_t id) {
    WireMessage m{WireKind::Hello, {}}; m.result.loan.work = {role, id, 2, 0}; return m;
}
static WireMessage lb_submit(Contract c) { WireMessage m{WireKind::Submit, {}}; m.result.loan.work = c; return m; }
static WireMessage lb_decision(uint64_t id, uint64_t gen, uint64_t v) {
    WireMessage m{WireKind::Decision, {}}; m.result.loan.work.id = id; m.result.loan.generation = gen; m.result.payload = v; return m;
}
static const char* state_name(ContractNegotiator::State s) {
    switch (s) { case ContractNegotiator::State::Unaccepted: return "Unaccepted"; case ContractNegotiator::State::Loaned: return "Loaned";
                 case ContractNegotiator::State::Returned: return "Returned"; default: return "Committed"; }
}
struct LoopbackRig {
    const char* prefix; NodusHeadlessGraph* g = nullptr; uint16_t port = 0; uint64_t host_id; int steps = 0;
    LoopbackRig(const char* p, uint64_t host, const fs::path& jobs, const fs::path& base)
        : prefix(p), host_id(host) {
        const std::string market = std::string(p) + ".market", worker = std::string(p) + ".worker";
        REQUIRE(nodus_headless_acquire_contract_pair_with_loan(p, 4, 54000));
        g = nodus_headless_create(); REQUIRE(g);
        GP_CanvasModuleDesc d{}; d.w = d.h = 64;
        const int m = nodus_headless_add_module(g, &d), w = nodus_headless_add_module(g, &d);
        const int mr = nodus_headless_bind_plugin_tool(g, m, market.c_str()), wr = nodus_headless_bind_plugin_tool(g, w, worker.c_str());
        REQUIRE(m >= 0 && w >= 0 && mr >= 0 && wr >= 0);
        GP_CanvasEdgeDesc e{}; e.a_module = m; e.a_contact_idx = mr; e.b_module = w; e.b_contact_idx = wr;
        const int edge = nodus_headless_add_edge(g, &e, 0); REQUIRE(edge >= 0);
        NodusHeadlessFifoConfig c{}; c.token_bytes = sizeof(Result); c.slots = 4; c.top_k = 0; c.type_id = 0x534A5231; c.layout = 2; c.dtype = 8;
        REQUIRE(nodus_headless_configure_edge_fifo(g, edge, &c));
        REQUIRE(nodus_headless_set_scheduler_mode(g, 1));
        const int32_t pt = nodus_headless_declare_contract_port(g, p, 4, edge, 0); REQUIRE(pt > 0); port = (uint16_t)pt;
        const std::string j = jobs.string(), b = base.string();
        REQUIRE(attach(p, host, "Z:/no/such/python.exe", b.c_str(), b.c_str(), b.c_str(), j.c_str()));
    }
    ~LoopbackRig() { nodus_headless_destroy(g); nodus_headless_release_contract_pair(prefix); }
    void step() { REQUIRE(nodus_headless_step(g, 1.0 / 60.0)); ++steps; }
    void dump(const char* why) {
        auto* n = find(prefix); uint64_t st[5] = {}; nodus_headless_compile_host_stats(prefix, host_id, st);
        std::fprintf(stderr, "STALL %s: step=%d market_turns=%llu entries=%zu discarded=%llu reoffered=%llu engines=%zu\n", why, steps,
                     (unsigned long long)n->turns(), n->contracts().size(), (unsigned long long)n->discarded_results(),
                     (unsigned long long)n->reoffered_loans(), n->engine_count());
        for (auto& e : n->contracts())
            std::fprintf(stderr, "  contract %llu engine %llu state %s gen %llu host %llu deadline %llu\n", (unsigned long long)e.work.id,
                         (unsigned long long)e.work.engine, state_name(e.state), (unsigned long long)e.loan.generation,
                         (unsigned long long)e.loan.host, (unsigned long long)e.loan.deadline);
        std::fprintf(stderr, "  host stats accepted=%llu completed=%llu refused=%llu acknowledged=%llu in_flight=%llu\n",
                     (unsigned long long)st[0], (unsigned long long)st[1], (unsigned long long)st[2], (unsigned long long)st[3], (unsigned long long)st[4]);
    }
    // One engine connection, id `engine_id`, one contract `cid`. wait=true: send the Decision when Route
    // arrives and close. wait=false: close as soon as Admitted arrived (the client gave up before any result).
    // Returns canvas steps from Submit to Route (or to Admitted when !wait); aborts with a dump past `bound`.
    int engine(uint64_t engine_id, uint64_t cid, bool wait, int bound) {
        ContractSocket c(lb_connect(port));
        REQUIRE(c.send_message(lb_hello(2, engine_id)));
        REQUIRE(c.send_message(lb_submit({cid, engine_id, 1, 2})));
        bool admitted = false;
        for (int i = 1; i <= bound; ++i) {
            step(); c.poll();
            WireMessage m{};
            while (c.receive(m)) {
                if (m.kind == WireKind::Admitted) admitted = true;
                if (m.kind == WireKind::Route) {
                    REQUIRE(m.result.loan.work.id == cid && m.result.payload == (uint64_t)CompileStatus::RequestMissing);
                    REQUIRE(c.send_message(lb_decision(cid, m.result.loan.generation, 1))); c.poll();
                    return i;
                }
            }
            if (!wait && admitted) return i;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        char why[96]; std::snprintf(why, sizeof why, "contract %llu admitted=%d no %s within %d steps", (unsigned long long)cid, (int)admitted, wait ? "Route" : "Admitted", bound);
        dump(why); std::abort();
    }
    // Step with no peer until every entry has left the negotiator and the host is idle, within `bound` steps.
    void settle(int bound) {
        auto* n = find(prefix);
        for (int i = 0; i < bound; ++i) {
            step();
            uint64_t st[5] = {}; nodus_headless_compile_host_stats(prefix, host_id, st);
            if (n->contracts().empty() && n->engine_count() == 0 && st[4] == 0) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        dump("settle"); std::abort();
    }
};

// Protocol bound, per contract: step 1 listener accepts+reads Hello, tick runs worker (Submit -> admit);
// step 2 market offers+host accepts and (missing request) finishes at once; step 3 market takes the
// finished Result (accept_result -> publish); step 4 worker consumes it and routes (Route frame is
// queued+sent). Hello/Submit may land one step later than they were sent, so 6 gives two spare steps.
static const int kRouteBound = 6;

static void loopback_missing_requests() {
    const fs::path base = fs::temp_directory_path() / ("turing_compile_host_abi_lb_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const fs::path jobs = base / "jobs"; fs::create_directories(jobs);
    {
        LoopbackRig r("abi.lb.a", 7, jobs, base);
        for (uint64_t cid = 1; cid <= 6; ++cid) {
            const int n = r.engine(9001, 5000 + cid, true, kRouteBound);
            std::printf("  contract %llu routed after %d steps\n", (unsigned long long)(5000 + cid), n);
        }
        r.settle(kRouteBound);
        uint64_t st[5] = {}; REQUIRE(nodus_headless_compile_host_stats(r.prefix, 7, st));
        std::printf("  stats accepted=%llu completed=%llu refused=%llu ack=%llu in_flight=%llu\n", (unsigned long long)st[0], (unsigned long long)st[1],
                    (unsigned long long)st[2], (unsigned long long)st[3], (unsigned long long)st[4]);
        REQUIRE(st[1] == 6 && st[4] == 0);
    }
    {   // Engines that leave before any result: the next one with the same id is still served promptly.
        LoopbackRig r("abi.lb.b", 7, jobs, base);
        for (uint64_t cid = 1; cid <= 3; ++cid) r.engine(9001, 6000 + cid, false, kRouteBound);       // abandoned
        for (uint64_t cid = 4; cid <= 6; ++cid) {
            const int n = r.engine(9001, 6000 + cid, true, kRouteBound);
            std::printf("  after abandon: contract %llu routed after %d steps\n", (unsigned long long)(6000 + cid), n);
        }
        r.settle(kRouteBound);
        REQUIRE(find(r.prefix)->contracts().empty());
        uint64_t st[5] = {}; REQUIRE(nodus_headless_compile_host_stats(r.prefix, 7, st));
        REQUIRE(st[4] == 0);
    }
    std::error_code ec; fs::remove_all(base, ec);
}

int main() {
    const fs::path base = fs::temp_directory_path() /
        ("turing_compile_host_abi_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const fs::path jobs = base / "jobs";
    fs::create_directories(jobs / "302");
    { std::ofstream(jobs / "302" / "nodus-row-tool-product.receipt") << "backend=c\n"; }
    fs::create_directories(jobs / "303");
    { std::ofstream(jobs / "303" / "nodus-row-tool-product.receipt") << "backend=c\n"; }
    const std::string j = jobs.string(), b = base.string();
    const char* py = "Z:/no/such/python.exe";

    // Refusals.
    REQUIRE(!attach(kPrefix, kHost, py, b.c_str(), b.c_str(), b.c_str(), j.c_str()));  // no acquired pair
    REQUIRE(nodus_headless_acquire_contract_pair(kPrefix, 8));
    REQUIRE(!attach(kPrefix, 0, py, b.c_str(), b.c_str(), b.c_str(), j.c_str()));      // host id 0
    REQUIRE(!attach(kPrefix, kHost, "", b.c_str(), b.c_str(), b.c_str(), j.c_str()));  // empty python
    REQUIRE(!attach(kPrefix, kHost, py, "", b.c_str(), b.c_str(), j.c_str()));         // empty nodus_root
    REQUIRE(!attach(kPrefix, kHost, py, b.c_str(), "", b.c_str(), j.c_str()));         // empty turing_root
    REQUIRE(!attach(kPrefix, kHost, py, b.c_str(), b.c_str(), "", j.c_str()));         // empty module root
    REQUIRE(!attach(kPrefix, kHost, py, b.c_str(), b.c_str(), b.c_str(), ""));         // empty jobs dir
    REQUIRE(!attach(kPrefix, kHost, py, b.c_str(), b.c_str(), b.c_str(), nullptr));    // null jobs dir
    uint64_t st[5] = {};
    REQUIRE(!nodus_headless_compile_host_stats(kPrefix, kHost, st));                   // nothing attached yet
    REQUIRE(!nodus_headless_detach_compile_host(kPrefix, kHost));

    // Attach, duplicate refused.
    REQUIRE(attach(kPrefix, kHost, py, b.c_str(), b.c_str(), b.c_str(), j.c_str()));
    REQUIRE(!attach(kPrefix, kHost, py, b.c_str(), b.c_str(), b.c_str(), j.c_str()));
    // The same id on another prefix is independent.
    REQUIRE(nodus_headless_acquire_contract_pair("abi.compile.other", 8));
    REQUIRE(attach("abi.compile.other", kHost, py, b.c_str(), b.c_str(), b.c_str(), j.c_str()));
    REQUIRE(nodus_headless_release_contract_pair("abi.compile.other"));

    // Contracts through the attached host: 301 has no request file, 302 has a product receipt.
    Rig rig; rig.to_submit = {301, 302}; rig.bind(10);
    rig.run(2);
    REQUIRE(rig.routed[0].first == 301 && rig.routed[0].second == (uint64_t)CompileStatus::RequestMissing);
    REQUIRE(rig.routed[1].first == 302 && rig.routed[1].second == (uint64_t)CompileStatus::ExistingProduct);
    REQUIRE(fs::is_regular_file(jobs / "301" / "compile-host.result.json"));
    REQUIRE(fs::is_regular_file(jobs / "302" / "compile-host.result.json"));
    REQUIRE(nodus_headless_compile_host_stats(kPrefix, kHost, st));
    REQUIRE(st[0] == 2 && st[1] == 2 && st[2] == 1 && st[3] == 2 && st[4] == 0);

    // Detach while idle: the host is REMOVED from the negotiator (no inert wrapper), stats gone.
    REQUIRE(rig.neg->host_count() == 1);
    REQUIRE(nodus_headless_detach_compile_host(kPrefix, kHost));
    REQUIRE(!nodus_headless_detach_compile_host(kPrefix, kHost));
    REQUIRE(!nodus_headless_compile_host_stats(kPrefix, kHost, st));
    REQUIRE(rig.neg->host_count() == 0);
    for (int i = 0; i < 5; ++i) { rig.neg->worker_turn(); rig.neg->market_turn(); }
    // A second live host on the same prefix gets the next contract promptly: with the old inert
    // wrapper registered it would sit behind the wrapper's reservation until the loan expired.
    REQUIRE(attach(kPrefix, kHost + 1, py, b.c_str(), b.c_str(), b.c_str(), j.c_str()));
    rig.to_submit.push_back(303);
    int turns = 0;
    while (rig.routed.size() < 3 && turns < 20) { rig.neg->worker_turn(); rig.neg->market_turn(); ++turns; }
    REQUIRE(rig.routed.size() == 3 && turns <= 10);
    REQUIRE(rig.routed[2].first == 303 && rig.routed[2].second == (uint64_t)CompileStatus::ExistingProduct);
    REQUIRE(rig.neg->discarded_results() == 0);
    REQUIRE(nodus_headless_detach_compile_host(kPrefix, kHost + 1));
    REQUIRE(rig.neg->host_count() == 0);
    // The detached id can be attached again (and is a registered host again).
    REQUIRE(attach(kPrefix, kHost, py, b.c_str(), b.c_str(), b.c_str(), j.c_str()));
    REQUIRE(rig.neg->host_count() == 1);
    REQUIRE(nodus_headless_compile_host_stats(kPrefix, kHost, st) && st[0] == 0);

    // Release with a host still attached: host ends first, pair goes, nothing dangles.
    REQUIRE(nodus_headless_release_contract_pair(kPrefix));
    REQUIRE(!find(kPrefix));
    REQUIRE(!nodus_headless_compile_host_stats(kPrefix, kHost, st));
    REQUIRE(!nodus_headless_detach_compile_host(kPrefix, kHost));
    REQUIRE(!attach(kPrefix, kHost, py, b.c_str(), b.c_str(), b.c_str(), j.c_str()));
    // A fresh pair under the same prefix takes the same host id again.
    REQUIRE(nodus_headless_acquire_contract_pair(kPrefix, 8));
    REQUIRE(attach(kPrefix, kHost, py, b.c_str(), b.c_str(), b.c_str(), j.c_str()));
    REQUIRE(nodus_headless_release_contract_pair(kPrefix));

    std::error_code ec; fs::remove_all(base, ec);
    loopback_missing_requests();
    std::printf("turing_compile_host_abi_test: OK\n");
    return 0;
}
