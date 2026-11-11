// Compile host attached through the headless ABI (nodus_headless_attach_compile_host).
// No Turing compile runs here: the python executable is a path that does not exist,
// so any attempt to spawn the compiler would show up as CompileFailed.
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
    std::printf("turing_compile_host_abi_test: OK\n");
    return 0;
}
