// native_host_test: the native Nodus host (frontend_shell --canvas, tools/native_host.h; stage A of
// docs/NODUS_NATIVE_HOST_PLAN_2026-11-23.md) does what tools/run_canvas.py does as a process.
//
//   native_host_test <work dir> <frontend_shell executable>
//
// It saves a canvas through the headless library with a contract port (a CONTRACT_PORT record on a configured result
// edge, loan 600), then runs the host on it offscreen for 600 ticks with --acquire and --link-host, and checks:
//   - the ready line: the contract port restored on a bound port with its capacity and edge, the link's listener on its
//     port, a pid, console_port null;
//   - while it runs: the restored contract port and the link listener accept a connection;
//   - the host ends on --ticks with {"ticks": 600} and exit 0, and the canvas file still holds its records;
//   - malformed host options exit 2 before anything starts (run_canvas's checks).
#include "nodus_link.h"          // first: winsock2.h before windows.h
#include "contract_socket.h"
#include "contract_negotiator.h"
#include "exec_pipe.h"
#include "nodus_headless_abi.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

std::string read_text(const fs::path& p) { std::ifstream in(p, std::ios::binary); std::stringstream s; s << in.rdbuf(); return s.str(); }

// The integer after "<key>": in a JSON line; -1 when absent.
long long json_int(const std::string& line, const std::string& key) {
    const size_t at = line.find("\"" + key + "\": ");
    if (at == std::string::npos) return -1;
    return std::atoll(line.c_str() + at + key.size() + 4);
}

// Runs the host to its end; returns its exit code and its stdout.
int run_host(const std::string& exe, std::vector<std::string> args, const fs::path& cwd, std::string& out) {
    nodus::exec::Child child;
    std::string why;
    if (!child.start({exe, std::move(args), cwd.string(), {}}, why)) { std::printf("host did not start: %s\n", why.c_str()); return -1; }
    child.finish_input();
    for (auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(60); child.running() && std::chrono::steady_clock::now() < give_up;) {
        child.take(out);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (child.running()) { child.kill(); return -1; }
    child.take(out);
    return child.exit_code();
}

// A loopback TCP connection to `port` is accepted (as nodus_headless_runtime_test connects to a contract port).
bool tcp_accepts(uint16_t port) {
#if defined(_WIN32)
    static const bool started = [] { WSADATA d; return WSAStartup(MAKEWORD(2, 2), &d) == 0; }();
    if (!started) return false;
#endif
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return false;
    sockaddr_in a{};
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(port);
    const bool ok = connect(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0;
#if defined(_WIN32)
    closesocket(s);
#else
    close(s);
#endif
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: native_host_test <work dir> <frontend_shell executable>\n"); return 2; }
    const fs::path work = fs::absolute(fs::path(argv[1]) / ("native_host_test_" + std::to_string(std::time(nullptr))));
    fs::create_directories(work);
    const std::string exe = fs::absolute(argv[2]).string();
    const fs::path canvas = work / "host.nodus.txt";
    const char* kPrefix = "nh.port";

    {   // A canvas with a contract port, saved through the headless file path (as nodus_headless_runtime_test builds one).
        CHECK(nodus_headless_acquire_contract_pair_with_loan(kPrefix, 4, 600));
        NodusHeadlessGraph* g = nodus_headless_create();
        GP_CanvasModuleDesc desc{}; desc.w = desc.h = 64;
        const int m = nodus_headless_add_module(g, &desc), w = nodus_headless_add_module(g, &desc);
        const int mr = nodus_headless_bind_plugin_tool(g, m, (std::string(kPrefix) + ".market").c_str());
        const int wr = nodus_headless_bind_plugin_tool(g, w, (std::string(kPrefix) + ".worker").c_str());
        GP_CanvasEdgeDesc edge{}; edge.a_module = m; edge.a_contact_idx = mr; edge.b_module = w; edge.b_contact_idx = wr;
        const int edge_id = nodus_headless_add_edge(g, &edge, 0);
        NodusHeadlessFifoConfig config{};
        config.token_bytes = sizeof(nodus::contracts::Result); config.slots = 4; config.type_id = 0x534A5231; config.layout = 2; config.dtype = 8;
        CHECK(m >= 0 && w >= 0 && mr >= 0 && wr >= 0 && edge_id >= 0 && nodus_headless_configure_edge_fifo(g, edge_id, &config));
        CHECK(nodus_headless_declare_contract_port(g, kPrefix, 4, edge_id, 0) > 0);
        CHECK(nodus_headless_save_file(g, canvas.string().c_str()));
        nodus_headless_destroy(g);
        CHECK(nodus_headless_release_contract_pair(kPrefix));
    }
    const std::string saved = read_text(canvas);
    CHECK(saved.find("CONTRACT_PORT nh.port 4 ") != std::string::npos && saved.find("EDGE_FIFO ") != std::string::npos);

    nodus::link::Key key{};
    nodus::link::random_bytes(key.data(), key.size());
    { std::ofstream k(work / "link.key", std::ios::binary); k.write(reinterpret_cast<const char*>(key.data()), key.size()); }
    const uint16_t link_port = 47671;

    // The host, run to --ticks; while it runs, the restored contract port and the link listener take a connection.
    nodus::exec::Child child;
    std::string why, out;
    CHECK(child.start({exe, {"--canvas", canvas.string(), "--rate", "60", "--ticks", "600", "--offscreen",
                             "--acquire", std::string(kPrefix) + ":4:600",
                             "--link-host", "nh.link:mesh-x:" + std::to_string(link_port) + ":" + (work / "link.key").string()},
                       work.string(), {}}, why));
    child.finish_input();
    std::string ready;
    for (auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30); ready.empty() && child.running() && std::chrono::steady_clock::now() < give_up;) {
        child.take(out);
        const size_t nl = out.find('\n');
        if (nl != std::string::npos) ready = out.substr(0, nl);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::printf("ready line: %s\n", ready.c_str());
    CHECK(ready.find("\"contract_ports\": [{\"prefix\": \"nh.port\"") != std::string::npos);
    CHECK(json_int(ready, "capacity") == 4 && json_int(ready, "edge") >= 0);
    const long long contract_port = json_int(ready, "listen_port");
    CHECK(contract_port > 0 && contract_port < 65536);
    CHECK(ready.find("\"links\": [{\"id\": \"nh.link\", \"prefix\": \"mesh-x\", \"listen_port\": " + std::to_string(link_port) + "}]") != std::string::npos);
    CHECK(json_int(ready, "pid") > 0 && ready.find("\"console_port\": null") != std::string::npos && ready.find("\"cron\": []") != std::string::npos);
    CHECK(contract_port > 0 && tcp_accepts(static_cast<uint16_t>(contract_port)));
    CHECK(tcp_accepts(link_port));
    for (auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(60); child.running() && std::chrono::steady_clock::now() < give_up;) {
        child.take(out);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    child.take(out);
    std::printf("host exit %d; stdout:\n%s", child.exit_code(), out.c_str());
    CHECK(!child.running() && child.exit_code() == 0);
    CHECK(out.find("{\"ticks\": 600}") != std::string::npos);   // a text-mode line: \r\n on Windows, as run_canvas's is
    CHECK(read_text(canvas) == saved);   // nothing changed it: not rewritten, every record kept

    // run_canvas's checks: each exits 2 before anything starts.
    const std::vector<std::vector<std::string>> refused = {
        {"--canvas", canvas.string(), "--rate", "0", "--offscreen"},
        {"--canvas", canvas.string(), "--offscreen", "--compile-host", "nh.port:1:" + work.string(), "--compile-python", "py"},
        {"--canvas", canvas.string(), "--offscreen", "--acquire", "nh.port:4", "--cron", "nh.port:0:" + work.string()},
        {"--canvas", canvas.string(), "--offscreen", "--acquire", "nh.port:4:0"},
        {"--canvas", canvas.string(), "--offscreen", "--link-host", "a:b:0:" + (work / "link.key").string()},
        {"--canvas", canvas.string(), "--offscreen", "--console-port", "0", "--console-token-file", "t"},
        {"--rate", "60"},
        {"--canvas", (work / "missing.nodus.txt").string(), "--offscreen"},
    };
    for (const auto& args : refused) {
        std::string o;
        const int code = run_host(exe, args, work, o);
        std::string joined; for (const auto& a : args) joined += a + " ";
        std::printf("refused (exit %d): %s\n", code, joined.c_str());
        CHECK(code == 2);
    }

    if (failures) { std::printf("FAILED %d\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
