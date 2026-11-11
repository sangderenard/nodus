// Turing compile host: one-at-a-time, non-blocking, no real Turing compile.
// A stub tools/turing_row_tool_compiler.py stands in for the driver.
#include "turing_compile_host_tool.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <signal.h>
#endif
using namespace nodus::contracts;
namespace fs = std::filesystem;
using Clk = std::chrono::steady_clock;

static const char* kStub = R"PY(
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--request"); ap.add_argument("--output-dir"); ap.add_argument("--turing-root")
a = ap.parse_args()
req = json.load(open(a.request))
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(root, "invocations.txt"), "a") as f: f.write(os.path.basename(a.request) + "\n")
os.makedirs(a.output_dir, exist_ok=True)
with open(os.path.join(a.output_dir, "pid.txt"), "w") as f: f.write(str(os.getpid()))
time.sleep(req.get("sleep_ms", 0) / 1000.0)
if req.get("fail"): sys.exit(3)
if req.get("source_only"):
    with open(os.path.join(a.output_dir, "nodus-row-tool-product.json"), "w") as f:
        json.dump({"schema": "nodus-turing-source-only-product-v1", "unstable_source_only": True,
                   "state": "emitted-not-built", "loadable": bool(req.get("forged_loadable")),
                   "buffer_abi": {} if req.get("forged_abi") else None}, f)
    if req.get("forged_receipt"):
        with open(os.path.join(a.output_dir, "nodus-row-tool-product.receipt"), "w") as f:
            f.write("backend=c\nmanifest=nodus-row-tool-product.json\nabi=missing\nbinary=missing\n")
    sys.exit(0)
with open(os.path.join(a.output_dir, "nodus-row-tool-product.receipt"), "w") as f:
    f.write("backend=c\nmanifest=m.json\nabi=a.nodus_buffer_abi.txt\nbinary=b.dll\n")
)PY";

static std::string find_python() {
    if (const char* e = std::getenv("PYTHON")) if (fs::is_regular_file(e)) return e;
    const char* path = std::getenv("PATH");
    std::string p = path ? path : "";
#ifdef _WIN32
    const char sep = ';'; const char* names[] = {"python.exe"};
#else
    const char sep = ':'; const char* names[] = {"python3", "python"};
#endif
    std::stringstream ss(p); std::string dir;
    while (std::getline(ss, dir, sep)) for (auto n : names) {
        std::error_code ec; fs::path c = fs::path(dir) / n;
        if (!dir.empty() && fs::is_regular_file(c, ec)) return c.string();
    }
    return {};
}
static void write(const fs::path& p, const std::string& s) {
    fs::create_directories(p.parent_path()); std::ofstream(p, std::ios::binary) << s;
}
static int lines(const fs::path& p) {
    std::ifstream f(p); std::string l; int n = 0; while (std::getline(f, l)) ++n; return n;
}
static bool pid_alive(unsigned long pid) {
#ifdef _WIN32
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, pid);
    if (!h) return false;
    bool alive = WaitForSingleObject(h, 0) == WAIT_TIMEOUT; CloseHandle(h); return alive;
#else
    return kill((pid_t)pid, 0) == 0;
#endif
}

int main() {
    const std::string python = find_python();
    assert(!python.empty());
    const fs::path base = fs::temp_directory_path() / ("turing_compile_host_test_" + std::to_string(Clk::now().time_since_epoch().count()));
    const fs::path nodus = base / "nodus", jobs = base / "jobs", turing = base / "turing";
    fs::create_directories(turing); fs::create_directories(jobs);
    write(nodus / "tools" / "turing_row_tool_compiler.py", kStub);
    write(jobs / "201.request.json", R"({"sleep_ms":600})");
    write(jobs / "202.request.json", R"({"sleep_ms":50,"fail":true})");
    write(jobs / "203.request.json", R"({"sleep_ms":50})");
    write(jobs / "301.request.json", R"({"sleep_ms":10000})");
    const fs::path counter = nodus / "invocations.txt";

    CompileHostConfig cfg;
    cfg.python_executable = python; cfg.nodus_root = nodus.string(); cfg.turing_root = turing.string();
    cfg.module_library_root = (base / "lib").string(); cfg.jobs_dir = jobs.string();
    cfg.host_id = 7; cfg.adopt = false;

    {
        TuringCompileHost host(cfg);
        auto choose = [](const Peer&, const std::vector<ContractNegotiator::Entry>& es) {
            for (size_t i = 0; i < es.size(); ++i) if (es[i].state == ContractNegotiator::State::Unaccepted) return i;
            return size_t(-1);
        };
        ContractNegotiator box(8, choose, 100000);
        std::vector<uint64_t> to_submit = {201, 202, 203, 204};
        size_t next = 0; std::vector<std::pair<uint64_t, uint64_t>> routed;
        assert(box.add_engine({10,
            [&](const Result& r) { routed.push_back({r.loan.work.id, r.payload}); return true; },
            [&](Contract& c) { if (next >= to_submit.size()) return false; c = {to_submit[next], 10, 1, 5}; ++next; return true; }}));
        assert(box.add_host(host.peer()));
        double max_turn_ms = 0; int max_in_flight = 0;
        const auto t0 = Clk::now();
        while (routed.size() < 4 && Clk::now() - t0 < std::chrono::seconds(30)) {
            const auto a = Clk::now();
            box.market_turn(); box.worker_turn();
            max_turn_ms = std::max(max_turn_ms, std::chrono::duration<double, std::milli>(Clk::now() - a).count());
            max_in_flight = std::max(max_in_flight, host.stats().in_flight);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        assert(routed.size() == 4);
        assert(max_in_flight == 1);
        assert(max_turn_ms < 50);
        const uint64_t want[4] = {(uint64_t)CompileStatus::CompiledNotAdopted, (uint64_t)CompileStatus::CompileFailed,
                                  (uint64_t)CompileStatus::CompiledNotAdopted, (uint64_t)CompileStatus::RequestMissing};
        for (int i = 0; i < 4; ++i) { assert(routed[i].first == 201u + i); assert(routed[i].second == want[i]); }
        auto st = host.stats();
        assert(st.completed == 4 && st.accepted == 4 && st.in_flight == 0 && st.refused == 1);
        assert(fs::is_regular_file(jobs / "201" / "compile-host.result.json"));
        assert(lines(counter) == 3); // 204 never spawned
        std::printf("max turn %.2f ms\n", max_turn_ms);
    }
    // Restart-safe duplicate refusal of work: a second host over the same jobs_dir.
    {
        TuringCompileHost host2(cfg);
        Peer p = host2.peer();
        Loan loan; loan.work = {201, 10, 1, 5}; loan.generation = 9; loan.host = 7; loan.deadline = 1000;
        Exchange e = p.exchange(&loan);
        assert(e.accepted_generation == 9 && !e.has_result);
        e = p.exchange(nullptr);
        assert(e.has_result && e.result.payload == (uint64_t)CompileStatus::ExistingProduct);
        assert(lines(counter) == 3);
    }
    // Source-only products complete distinctly with adoption enabled or disabled.
    for (bool adopt : {false, true}) {
        const int id = adopt ? 402 : 401;
        write(jobs / (std::to_string(id) + ".request.json"), R"({"source_only":true})");
        auto source_cfg = cfg; source_cfg.adopt = adopt;
        TuringCompileHost host(source_cfg); Peer p = host.peer();
        Loan loan; loan.work = {static_cast<uint64_t>(id), 10, 1, 5};
        loan.generation = 1; loan.host = 7; loan.deadline = 1000;
        Exchange e = p.exchange(&loan); assert(e.accepted_generation == 1);
        const auto start = Clk::now();
        while (!e.has_result && Clk::now() - start < std::chrono::seconds(10)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); e = p.exchange(nullptr);
        }
        assert(e.has_result && e.result.payload == (uint64_t)CompileStatus::EmittedNotBuilt);
        assert(!fs::exists(jobs / std::to_string(id) / "nodus-row-tool-product.receipt"));
        assert(!fs::exists(base / "lib"));
    }
    // A forged mixed source-only/runnable product cannot become a success or adoption.
    for (int id : {403, 404, 405}) {
        write(jobs / (std::to_string(id) + ".request.json"), id == 403
              ? R"({"source_only":true,"forged_loadable":true})"
              : id == 404 ? R"({"source_only":true,"forged_receipt":true})"
              : R"({"source_only":true,"forged_abi":true})");
        auto source_cfg = cfg; source_cfg.adopt = true;
        TuringCompileHost host(source_cfg); Peer p = host.peer();
        Loan loan; loan.work = {static_cast<uint64_t>(id), 10, 1, 5};
        loan.generation = 1; loan.host = 7; loan.deadline = 1000;
        Exchange e = p.exchange(&loan);
        const auto start = Clk::now();
        while (!e.has_result && Clk::now() - start < std::chrono::seconds(10)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5)); e = p.exchange(nullptr);
        }
        assert(e.has_result && e.result.payload == (uint64_t)CompileStatus::CompileFailed);
        assert(!fs::exists(base / "lib"));
    }
    // Existing runnable receipt must not bypass source-only admission or call adoption.
    for (const std::string request : {R"({"unstable_source_only":true})",
                                     R"({"unstable\u005fsource_only":true})",
                                     R"({"unstable_source_only":false,"unstable_source_only":true})",
                                     R"({"source_names":["entry"]})"}) {
        const int before = lines(counter);
        write(jobs / "501.request.json", request);
        write(jobs / "501" / "nodus-row-tool-product.receipt",
              "NODUS ROW TOOL PRODUCT V1\nBACKEND \"c\"\nMANIFEST \"m.json\"\nABI \"a.nodus_buffer_abi.txt\"\nBINARY \"b.dll\"\n");
        write(jobs / "501" / "m.json", R"({"schema":"nodus-turing-row-tool-product-v1","loadable":true,"backend":"c"})");
        write(jobs / "501" / "a.nodus_buffer_abi.txt", "descriptor"); write(jobs / "501" / "b.dll", "binary");
        auto source_cfg = cfg; source_cfg.adopt = true;
        TuringCompileHost host(source_cfg); Peer p = host.peer();
        Loan loan; loan.work = {501, 10, 1, 5}; loan.generation = 1; loan.host = 7; loan.deadline = 1000;
        assert(p.exchange(&loan).accepted_generation == 1);
        auto e = p.exchange(nullptr);
        assert(e.has_result && e.result.payload == (uint64_t)CompileStatus::CompileFailed);
        assert(lines(counter) == before); // no driver, importer, build or load
        assert(fs::exists(jobs / "501" / "nodus-row-tool-product.receipt"));
    }
    // Cancellation: shutdown kills a 10 s child promptly.
    {
        TuringCompileHost host3(cfg);
        Peer p = host3.peer();
        Loan loan; loan.work = {301, 10, 1, 5}; loan.generation = 1; loan.host = 7; loan.deadline = 1000;
        Exchange e = p.exchange(&loan); assert(e.accepted_generation == 1);
        assert(p.exchange(nullptr).has_result == false && host3.stats().in_flight == 1);
        const fs::path pidf = jobs / "301" / "pid.txt";
        const auto w = Clk::now();
        while (!fs::is_regular_file(pidf) && Clk::now() - w < std::chrono::seconds(10)) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::ifstream pf(pidf); unsigned long pid = 0; pf >> pid; assert(pid);
        assert(pid_alive(pid));
        const auto s = Clk::now();
        host3.shutdown();
        const double secs = std::chrono::duration<double>(Clk::now() - s).count();
        assert(secs < 3.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        assert(!pid_alive(pid));
        assert(host3.stats().in_flight == 0);
        std::printf("shutdown %.2f s\n", secs);
    }
    std::error_code ec; fs::remove_all(base, ec);
    std::puts("turing compile host tests passed");
}
