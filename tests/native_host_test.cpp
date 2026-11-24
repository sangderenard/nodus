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
//   - malformed host options exit 2 before anything starts (run_canvas's checks);
//   - the native console (--console-port): HELLO v1/v2, DIGEST, SNAPSHOT with STATE, the views, ACTIONs (a stale
//     revision, a type error, a viewport that changes the revision and saves the canvas with its records), the
//     refusals (read-only, bad token, bad hello, invalid action) and the ninth client refused.
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

// One console connection (docs/NODUS_CONSOLE_READ_PROTOCOL_2026-11-10.md): lines out, a line or a framed reply back.
struct ConsoleClient {
    SOCKET s = INVALID_SOCKET;
    std::string buf;
    explicit ConsoleClient(uint16_t port) {
        s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr_in a{};
        a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(port);
        if (connect(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) { close_now(); return; }
#if defined(_WIN32)
        DWORD ms = 10000;
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
#else
        timeval tv{10, 0};
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
    }
    ~ConsoleClient() { close_now(); }
    void close_now() {
        if (s == INVALID_SOCKET) return;
#if defined(_WIN32)
        closesocket(s);
#else
        close(s);
#endif
        s = INVALID_SOCKET;
    }
    void send_text(const std::string& t) { if (s != INVALID_SOCKET) ::send(s, t.data(), int(t.size()), 0); }
    bool fill() {
        char b[4096];
        const int n = ::recv(s, b, sizeof(b), 0);
        if (n <= 0) return false;
        buf.append(b, size_t(n));
        return true;
    }
    std::string line() {   // one line with its "\n", or what arrived before the end of the stream
        for (;;) {
            const size_t nl = buf.find('\n');
            if (nl != std::string::npos) { std::string l = buf.substr(0, nl + 1); buf.erase(0, nl + 1); return l; }
            if (s == INVALID_SOCKET || !fill()) { std::string l = buf; buf.clear(); return l; }
        }
    }
    std::string bytes(size_t n) {
        while (buf.size() < n && s != INVALID_SOCKET && fill()) {}
        std::string out = buf.substr(0, std::min(n, buf.size()));
        buf.erase(0, out.size());
        return out;
    }
    std::string ask(const std::string& request) {   // a verb's reply: a line, or "JSON <VERB> <n>\n" + n bytes
        send_text(request + "\n");
        std::string head = line();
        if (head.rfind("JSON ", 0) == 0) head += bytes(size_t(std::atoll(head.c_str() + head.rfind(' ') + 1)));
        return head;
    }
    std::string action(const std::string& json) {
        send_text("ACTION " + std::to_string(json.size()) + "\n" + json);
        std::string head = line();
        if (head.rfind("JSON ", 0) == 0) head += bytes(size_t(std::atoll(head.c_str() + head.rfind(' ') + 1)));
        return head;
    }
};

// The native console (tools/native_console.h): protocol replies, an ACTION that changes and saves the canvas, the
// refusals, and the client limit.
void console_checks(const std::string& exe, const fs::path& work, const fs::path& canvas_source) {
    const fs::path dir = work / "console";
    fs::create_directories(dir);
    const fs::path canvas = dir / "console.nodus.txt";
    fs::copy_file(canvas_source, canvas, fs::copy_options::overwrite_existing);
    { std::ofstream t(dir / "token", std::ios::binary); t << "console-secret\r\n"; }
    nodus::exec::Child child;
    std::string why, out;
    CHECK(child.start({exe, {"--canvas", canvas.string(), "--rate", "60", "--offscreen", "--acquire", "nh.port:4:600",
                             "--console-port", "0", "--console-token-file", (dir / "token").string()}, dir.string(), {}}, why));
    child.finish_input();
    std::string ready;
    for (auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30); ready.empty() && child.running() && std::chrono::steady_clock::now() < give_up;) {
        child.take(out);
        const size_t nl = out.find('\n');
        if (nl != std::string::npos) ready = out.substr(0, nl);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const long long port = json_int(ready, "console_port");
    std::printf("console ready line: %s\n", ready.c_str());
    CHECK(port > 0 && port < 65536);
    if (port <= 0) { child.kill(); return; }
    const uint16_t p = uint16_t(port);
    {
        ConsoleClient c(p);
        CHECK(c.ask("HELLO nodus-console/2 console-secret") == "OK console console.nodus.txt actions\n");
        const std::string digest = c.ask("DIGEST");
        std::printf("DIGEST: %s\n", digest.c_str());
        CHECK(digest.rfind("JSON DIGEST ", 0) == 0 && digest.find("\"protocol\":2") != std::string::npos &&
              digest.find("\"revision\":1") != std::string::npos && digest.find("\"contract_ports\":1") != std::string::npos &&
              digest.find("\"rate\":60.0") != std::string::npos);
        c.send_text("SNAPSHOT\n");
        const std::string head = c.line();
        CHECK(head.rfind("CANVAS 1 ", 0) == 0);
        const std::string data = c.bytes(size_t(std::atoll(head.c_str() + 9)));
        CHECK(data.rfind("CANVAS V4", 0) == 0 && data.find("CONTRACT_PORT nh.port 4 ") != std::string::npos);
        CHECK(c.line() == "\n");
        const std::string state = c.line();
        CHECK(state.rfind("STATE {\"revision\": 1, \"ticks\": ", 0) == 0 && state.find("\"events\": []}") != std::string::npos);
        CHECK(c.ask("digest") == "REFUSED read-only\n");
        CHECK(c.ask("GRAPH").rfind("JSON GRAPH ", 0) == 0);
        CHECK(c.ask("TOOLS").find("\"name\":\"nh.port.market\"") != std::string::npos);
        CHECK(c.ask("ACTIVITY").find("\"contract_ports\":[{\"engines\":0,\"hosts\":0,\"pending\":null,\"prefix\":\"nh.port\"}]") != std::string::npos);
        CHECK(c.ask("META").rfind("JSON META ", 0) == 0);
        CHECK(c.action("{\"op\": \"viewport\", \"expected_revision\": 7, \"width\": 800, \"height\": 600}") ==
              "JSON ACTION 48\n{\"error\":\"canvas revision changed\",\"revision\":1}");
        CHECK(c.action("{\"op\": \"viewport\", \"expected_revision\": 1, \"width\": 800.0, \"height\": 600}") ==
              "JSON ACTION 54\n{\"error\":\"viewport width and height must be integers\"}");
        CHECK(c.action("{\"op\": \"viewport\", \"expected_revision\": 1, \"width\": 800, \"height\": 600}") ==
              "JSON ACTION 58\n{\"handled\":true,\"revision\":2,\"canvas\":\"console.nodus.txt\"}");
        CHECK(read_text(canvas).find("CANVAS V4\r\n800 600") == 0 || read_text(canvas).find("CANVAS V4\n800 600") == 0);   // saved
        CHECK(read_text(canvas).find("CONTRACT_PORT nh.port 4 ") != std::string::npos);   // with its records
        CHECK(c.action("{\"op\": \"link_host\", \"expected_revision\": 2, \"id\": \"x\"}").find("stage C") != std::string::npos);
        CHECK(c.ask("ACTION nope") == "REFUSED invalid action\n");
    }
    {
        ConsoleClient v1(p);
        CHECK(v1.ask("HELLO nodus-console/1 console-secret") == "OK console console.nodus.txt read-only\n");
        CHECK(v1.ask("ACTION 2") == "REFUSED read-only\n");
        ConsoleClient bad(p);
        CHECK(bad.ask("HELLO nodus-console/2 wrong") == "REFUSED bad token\n");
        ConsoleClient hello(p);
        CHECK(hello.ask("HELO") == "REFUSED bad hello\n");
    }
    {   // eight clients at once; the ninth is refused
        std::vector<std::unique_ptr<ConsoleClient>> held;
        for (int k = 0; k < 8; ++k) {
            held.push_back(std::make_unique<ConsoleClient>(p));
            CHECK(held.back()->ask("HELLO nodus-console/1 console-secret") == "OK console console.nodus.txt read-only\n");
        }
        ConsoleClient ninth(p);
        CHECK(ninth.line() == "REFUSED too many clients\n");
    }
    child.kill();
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
        {"--canvas", canvas.string(), "--offscreen", "--console-port", "0", "--console-token-file", (work / "missing.token").string()},
        {"--canvas", canvas.string(), "--offscreen", "--console-port", "0"},
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

    console_checks(exe, work, canvas);

    if (failures) { std::printf("FAILED %d\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
