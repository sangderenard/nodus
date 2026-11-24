// native_host.h: the native Nodus canvas host, stage A of docs/NODUS_NATIVE_HOST_PLAN_2026-11-23.md.
//
// Owner, 2026-11-23: "I want us moving fully out of run canvas and the headless run, nodus must run with the rasterizer
// and fully native, and still have all features it has now." This is what tools/run_canvas.py does as a process, done
// natively inside frontend_shell, on the headless ABI compiled into the same image (NODUS_HEADLESS_STATIC): one copy of
// the engine, every canvas-file record loaded and saved, contract pairs, links, compile hosts and cron attached in
// run_canvas's order, the same ready line on stdout, the same fixed-rate step, signals, shutdown order and exit codes.
//
// The command line is run_canvas's, in its "--name value" form or the shell's "--name=value":
//   --canvas PATH (host mode)  --rate F (60)  --ticks N  --acquire PREFIX:CAPACITY[:LOAN_TICKS]  (repeatable)
//   --compile-host PREFIX:HOST_ID:JOBS_DIR  --compile-python EXE  --turing-root DIR  --module-library-root DIR
//   --compile-adopt  --cron PREFIX:ENGINE_ID:JOBS_DIR  --link-host ID:PREFIX:PORT:KEYFILE  (repeatable)
//   --console-port N  --console-token-file PATH   the canvas console on 127.0.0.1 (tools/native_console.h; 0: the system
//                     picks the port, reported as console_port in the ready line)
//   --offscreen   no window: no SDL video and no GL. The rasterizer draws only when something asks for a frame.
//   --nodus-root DIR   the Nodus checkout the compile host builds against (default: this build's source tree)
//   --link-route-map PATH   the front's route map (RewriteMap txt:): a routed LINK_HOST of the canvas writes its line there
// Links are declared in the canvas (LINK_HOST records, nodus_headless_abi.h) and restored as it loads; --link-host
// registers one for this run only, as run_canvas did.
// With a window, the window draws the hosted canvas and gives it input; a change made there is saved through the
// headless file path (every record kept), at most every 2 s, between steps. Closing the window stops the host.
#pragma once

#include "native_console.h"
#include "nodus_headless_abi.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

#ifndef NODUS_REPO_ROOT
#define NODUS_REPO_ROOT "."
#endif

namespace nodus::host {

// A change made through the window (the shell's input calls set it); the stepping thread saves it.
inline std::atomic<bool>& canvas_changed() { static std::atomic<bool> v{false}; return v; }
inline std::atomic<bool>& stop_requested() { static std::atomic<bool> v{false}; return v; }
// One rasterization of the canvas at a time: the window's draw and the console's FRAME both call gp_canvas_raster_rgba,
// which updates the canvas's scroll and palette state as it draws.
inline std::mutex& raster_mutex() { static std::mutex m; return m; }

struct Acquire { std::string prefix; int32_t capacity = 0; uint64_t loan = 0; };   // loan 0: the native default (60)
struct CompileHost { std::string prefix; uint64_t id = 0; std::string jobs; };
struct Cron { std::string prefix; uint64_t engine = 0; std::string jobs; };
struct LinkHost { std::string id, prefix; uint16_t port = 0; std::string key_file; };

struct Options {
    bool enabled = false;                 // --canvas given
    std::string canvas;
    double rate = 60.0;
    long long ticks = -1;                 // -1: until a signal
    std::vector<Acquire> acquires;
    std::vector<CompileHost> compile_hosts;
    std::vector<Cron> crons;
    std::vector<LinkHost> link_hosts;
    std::string compile_python, turing_root, module_library_root, nodus_root = NODUS_REPO_ROOT;
    bool compile_adopt = false;
    std::string link_route_map;           // --link-route-map: the front's route map for routed links (LINK_HOST flag 1)
    long long console_port = -1;          // --console-port (0: the system picks); -1: no console
    std::string console_token_file;
    bool offscreen = false;
};

namespace detail {

inline bool digits(const std::string& s) {
    if (s.empty() || s.size() > 19) return false;
    for (char c : s) if (c < '0' || c > '9') return false;
    return true;
}
// Splits on the first `n` colons only: the last field may hold a drive colon (C:/jobs), as in run_canvas.
inline std::vector<std::string> split_first(const std::string& s, size_t n) {
    std::vector<std::string> out;
    size_t at = 0;
    for (size_t i = 0; i < n; ++i) {
        const size_t c = s.find(':', at);
        if (c == std::string::npos) break;
        out.push_back(s.substr(at, c - at));
        at = c + 1;
    }
    out.push_back(s.substr(at));
    return out;
}
inline Acquire parse_acquire(const std::string& text) {   // run_canvas _parse_acquire
    std::vector<std::string> parts;
    for (size_t at = 0;;) {
        const size_t c = text.find(':', at);
        parts.push_back(text.substr(at, c == std::string::npos ? std::string::npos : c - at));
        if (c == std::string::npos) break;
        at = c + 1;
    }
    Acquire a;
    if (parts.size() >= 3 && digits(parts.back()) && digits(parts[parts.size() - 2])) {
        a.loan = std::stoull(parts.back());
        parts.pop_back();
        if (!a.loan) throw std::invalid_argument("--acquire LOAN_TICKS must be > 0, got '" + text + "'");
    }
    std::string prefix;
    for (size_t i = 0; i + 1 < parts.size(); ++i) prefix += (i ? ":" : "") + parts[i];
    if (parts.size() < 2 || prefix.empty() || !digits(parts.back()) || std::stoull(parts.back()) > 0x7FFFFFFF)
        throw std::invalid_argument("--acquire expects PREFIX:CAPACITY[:LOAN_TICKS], got '" + text + "'");
    a.prefix = prefix;
    a.capacity = static_cast<int32_t>(std::stoull(parts.back()));
    return a;
}
template <class T>
T parse_engine(const std::string& text, const char* flag, const char* what) {   // PREFIX:ID:JOBS_DIR, ID non-zero
    const auto p = split_first(text, 2);
    if (p.size() != 3 || p[0].empty() || !digits(p[1]) || std::stoull(p[1]) == 0 || p[2].empty())
        throw std::invalid_argument(std::string(flag) + " expects PREFIX:" + what + ":JOBS_DIR (" + what + " non-zero), got '" + text + "'");
    return T{p[0], std::stoull(p[1]), p[2]};
}
inline LinkHost parse_link_host(const std::string& text) {
    const auto p = split_first(text, 3);
    if (p.size() != 4 || p[0].empty() || p[1].empty() || !digits(p[2]) || std::stoull(p[2]) == 0 || std::stoull(p[2]) > 65535 || p[3].empty())
        throw std::invalid_argument("--link-host expects ID:PREFIX:PORT:KEYFILE (PORT 1-65535), got '" + text + "'");
    return LinkHost{p[0], p[1], static_cast<uint16_t>(std::stoull(p[2])), p[3]};
}
inline std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
                else out += static_cast<char>(c);
        }
    }
    return out + "\"";
}

}  // namespace detail

// True when the command line asks for host mode (--canvas). Throws std::invalid_argument on a malformed host option,
// with run_canvas's message; the caller prints it and exits 2.
inline bool parse(int argc, char** argv, Options& o) {
    static const std::set<std::string> kValued = {"--canvas", "--rate", "--ticks", "--acquire", "--compile-host", "--cron",
        "--link-host", "--compile-python", "--turing-root", "--module-library-root", "--console-port", "--console-token-file",
        "--nodus-root", "--link-route-map"};
    static const std::set<std::string> kFlags = {"--compile-adopt", "--offscreen"};
    bool any = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i] ? argv[i] : "", value;
        const size_t eq = arg.find('=');
        std::string name = arg.substr(0, eq);
        if (kFlags.count(name)) {
            if (eq != std::string::npos) throw std::invalid_argument(name + " takes no value");
            if (name == "--compile-adopt") o.compile_adopt = true; else o.offscreen = true;
            continue;
        }
        if (!kValued.count(name)) continue;   // the shell's own options are read by the shell
        if (eq != std::string::npos) value = arg.substr(eq + 1);
        else if (i + 1 < argc) value = argv[++i];
        else throw std::invalid_argument(name + " needs a value");
        any = true;
        if (name == "--canvas") { o.canvas = value; o.enabled = true; }
        else if (name == "--rate") { o.rate = std::stod(value); }
        else if (name == "--ticks") { o.ticks = std::stoll(value); }
        else if (name == "--acquire") o.acquires.push_back(detail::parse_acquire(value));
        else if (name == "--compile-host") o.compile_hosts.push_back(detail::parse_engine<CompileHost>(value, "--compile-host", "HOST_ID"));
        else if (name == "--cron") o.crons.push_back(detail::parse_engine<Cron>(value, "--cron", "ENGINE_ID"));
        else if (name == "--link-host") o.link_hosts.push_back(detail::parse_link_host(value));
        else if (name == "--compile-python") o.compile_python = value;
        else if (name == "--turing-root") o.turing_root = value;
        else if (name == "--module-library-root") o.module_library_root = value;
        else if (name == "--nodus-root") o.nodus_root = value;
        else if (name == "--console-port") o.console_port = std::stoll(value);
        else if (name == "--link-route-map") o.link_route_map = value;
        else o.console_token_file = value;
    }
    if (!o.enabled) {
        if (any || o.compile_adopt || o.offscreen) throw std::invalid_argument("host options need --canvas");
        return false;
    }
    if (!(o.rate > 0.0)) throw std::invalid_argument("--rate must be positive");
    if (o.ticks < -1) throw std::invalid_argument("--ticks must not be negative");
    if (o.console_port >= 0 && o.console_token_file.empty()) throw std::invalid_argument("--console-port requires --console-token-file");
    if (o.console_port > 65535) throw std::invalid_argument("--console-port must be 0..65535");
    std::set<std::string> acquired;
    for (const auto& a : o.acquires) acquired.insert(a.prefix);
    std::set<std::pair<std::string, uint64_t>> seen;
    for (const auto& c : o.compile_hosts) {
        if (!acquired.count(c.prefix)) throw std::invalid_argument("--compile-host prefix '" + c.prefix + "' must also be given to --acquire");
        if (!seen.insert({c.prefix, c.id}).second) throw std::invalid_argument("--compile-host PREFIX:HOST_ID given twice");
    }
    if (!o.compile_hosts.empty() && o.compile_python.empty())
        throw std::invalid_argument("--compile-host needs --compile-python (the Python that runs the Turing compile driver)");
    std::set<std::string> link_ids;
    for (const auto& l : o.link_hosts)
        if (!link_ids.insert(l.id).second) throw std::invalid_argument("--link-host ID given twice");
    seen.clear();
    for (const auto& c : o.crons) {
        if (!acquired.count(c.prefix)) throw std::invalid_argument("--cron prefix '" + c.prefix + "' must also be given to --acquire");
        if (!seen.insert({c.prefix, c.engine}).second) throw std::invalid_argument("--cron PREFIX:ENGINE_ID given twice");
    }
    if (o.turing_root.empty()) o.turing_root = o.nodus_root + "/../turing";
    if (o.module_library_root.empty()) o.module_library_root = o.canvas + ".modules";
    return true;
}

// One hosted canvas. start() does run_canvas's startup and prints the ready line; run() steps at the rate until a signal,
// --ticks or request_stop(); finish() detaches in run_canvas's order, destroys the graph and prints {"ticks": n}.
class Host {
public:
    explicit Host(Options o) : o_(std::move(o)) {}
    ~Host() { if (g_) nodus_headless_destroy(g_); if (out_) std::fclose(out_); }

    NodusHeadlessGraph* graph() const { return g_; }

    // 0, or 2 after printing the reason (run_canvas's exit codes).
    int start() {
        try {
            std::string token;
            if (o_.console_port >= 0) {   // run_canvas: the first line, "\r\n" stripped, must not be empty
                std::ifstream in(o_.console_token_file, std::ios::binary);
                if (!in) throw std::runtime_error("console token file could not be read: " + o_.console_token_file);
                std::getline(in, token);
                while (!token.empty() && (token.back() == '\r' || token.back() == '\n')) token.pop_back();
                if (token.empty()) throw std::runtime_error("console token file is empty");
            }
            install_signals();
            claim_stdout();
            if (o_.console_port >= 0) {   // before the graph, as run_canvas starts its ConsoleServer
                nodus::console::Context context;
                context.canvas_name = std::filesystem::path(o_.canvas).filename().string();
                context.canvas_path = o_.canvas;
                context.rate = o_.rate;
                context.started = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
                context.ticks = [this] { return ticks_; };
                context.frame = [this](std::string& rgba, int& width, int& height) {   // FRAME: what the window would show
                    GP_CanvasContext* ctx = nodus_headless_canvas_context(g_);
                    std::lock_guard<std::mutex> lk(raster_mutex());
                    if (!ctx || !gp_canvas_get_size(ctx, &width, &height) || width <= 0 || height <= 0) return false;
                    rgba.assign(size_t(width) * size_t(height) * 4, '\0');
                    return gp_canvas_raster_rgba(ctx, reinterpret_cast<uint8_t*>(rgba.data()), int32_t(rgba.size())) != 0;
                };
                console_ = std::make_unique<nodus::console::Server>(token, std::move(context));
                console_port_ = console_->start(static_cast<uint16_t>(o_.console_port));
            }
            if (!o_.link_route_map.empty()) nodus_headless_set_link_route_map(o_.link_route_map.c_str());   // before LINK_HOST loads
            g_ = nodus_headless_create();
            if (!g_) throw std::runtime_error("native graph creation failed");
            if (!nodus_headless_set_scheduler_mode(g_, 1)) throw std::runtime_error("native scheduler mode refused");
            for (const auto& a : o_.acquires) {
                const int32_t ok = a.loan ? nodus_headless_acquire_contract_pair_with_loan(a.prefix.c_str(), a.capacity, a.loan)
                                          : nodus_headless_acquire_contract_pair(a.prefix.c_str(), a.capacity);
                if (!ok) throw std::runtime_error("native contract pair acquire refused: " + a.prefix);
            }
            for (const auto& l : o_.link_hosts) {   // before the load: the canvas binds the rows
                std::ifstream in(l.key_file, std::ios::binary);
                std::string key((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                if (key.size() != 32) throw std::invalid_argument("the link key is 32 bytes: " + l.key_file);
                const int32_t bound = nodus_headless_register_link_host(g_, l.id.c_str(), l.prefix.c_str(), l.port,
                                                                        reinterpret_cast<const uint8_t*>(key.data()), 0);
                if (bound <= 0) throw std::runtime_error("native link host registration refused: " + l.id);
                links_.push_back({l, bound});
            }
            if (!nodus_headless_load_file(g_, o_.canvas.c_str())) throw std::runtime_error("native load failed: " + o_.canvas);
            check_edges();
            try {
                for (const auto& c : o_.compile_hosts) {
                    if (!nodus_headless_attach_compile_host(c.prefix.c_str(), c.id, o_.compile_python.c_str(), o_.nodus_root.c_str(),
                                                            o_.turing_root.c_str(), o_.module_library_root.c_str(), c.jobs.c_str(),
                                                            o_.compile_adopt ? 1 : 0))
                        throw std::runtime_error("native compile host attach refused: " + c.prefix + " host " + std::to_string(c.id));
                    attached_.push_back({c.prefix, c.id});
                }
            } catch (...) {
                for (const auto& [prefix, id] : attached_) nodus_headless_detach_compile_host(prefix.c_str(), id);
                attached_.clear();
                throw;
            }
            std::ostringstream cron;
            for (const auto& c : o_.crons) {   // after the load: the engine joins the restored negotiator
                // A cron tool that cannot start is reported and left off; it must not take the host down with it.
                cron << (cron.tellp() > 0 ? "," : "") << "{\"prefix\": " << detail::json_string(c.prefix) << ", \"engine\": " << c.engine;
                if (nodus_headless_attach_cron(c.prefix.c_str(), c.engine, c.jobs.c_str())) {
                    scheduled_.push_back({c.prefix, c.engine});
                    cron << ", \"attached\": true}";
                } else {
                    const std::string why = "native cron attach refused: " + c.prefix + " engine " + std::to_string(c.engine) +
                        " (no acquired pair, the id is already an engine, or cron.table is missing or malformed; see " + c.jobs + "/cron.log)";
                    std::fprintf(stderr, "nodus: %s\n", why.c_str());
                    cron << ", \"attached\": false, \"reason\": " << detail::json_string(why) << "}";
                }
            }
            if (console_) {
                std::vector<std::string> link_ids;
                for (const auto& l : links_) link_ids.push_back(l.first.id);
                console_->set_context(g_, attached_, link_ids);
            }
            ready_line(cron.str());
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "nodus: %s\n", e.what());
            std::fflush(stderr);
            return 2;
        }
    }

    // The fixed-rate loop: an absolute deadline; behind it, the deadline resets (no catch-up burst), as in run_canvas.
    void run() {
        const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / o_.rate));
        const double dt = 1.0 / o_.rate;
        auto next = std::chrono::steady_clock::now();
        auto save_after = next;
        while (!stop_requested().load() && (o_.ticks < 0 || ticks_ < o_.ticks)) {
            if (!nodus_headless_step(g_, dt)) { std::fprintf(stderr, "nodus: native graph step failed\n"); failed_ = true; break; }
            ++ticks_;
            if (console_) console_->service();   // the console's views and ACTIONs, between steps
            const auto now = std::chrono::steady_clock::now();
            if (canvas_changed().load() && now >= save_after) { save(); save_after = now + std::chrono::seconds(2); }
            next += period;
            if (next > now) std::this_thread::sleep_until(next);
            else next = now;
        }
        stop_requested().store(true);
    }
    void request_stop() { stop_requested().store(true); }

    // Detaches cron, then compile hosts (a running compile cancelled, its thread joined) before any pair is released;
    // saves a change the window made; destroys the graph; prints {"ticks": n}. Returns the exit code.
    int finish() {
        for (const auto& [prefix, engine] : scheduled_)
            if (!nodus_headless_detach_cron(prefix.c_str(), engine)) std::fprintf(stderr, "nodus: native cron detach refused: %s\n", prefix.c_str());
        for (const auto& [prefix, id] : attached_)
            if (!nodus_headless_detach_compile_host(prefix.c_str(), id)) std::fprintf(stderr, "nodus: native compile host detach refused: %s\n", prefix.c_str());
        if (canvas_changed().load()) save();
        nodus_headless_destroy(g_);
        g_ = nullptr;
        if (console_) console_->close();   // after the graph, as run_canvas closes its console
        if (out_) { std::fprintf(out_, "{\"ticks\": %lld}\n", ticks_); std::fflush(out_); }
        std::fflush(stderr);
        return failed_ ? 2 : 0;
    }

private:
    static void on_signal(int) { stop_requested().store(true); }
    static void install_signals() {
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
#if defined(SIGBREAK)
        std::signal(SIGBREAK, on_signal);   // the supervisor's CTRL_BREAK_EVENT
#endif
    }
    // The native library writes diagnostics to fd 1; the real stdout is kept for the JSON lines only (run_canvas _claim_stdout).
    void claim_stdout() {
        std::fflush(stdout);
#if defined(_WIN32)
        const int keep = _dup(1);
        _dup2(2, 1);
        out_ = keep >= 0 ? _fdopen(keep, "w") : nullptr;
#else
        const int keep = dup(1);
        dup2(2, 1);
        out_ = keep >= 0 ? fdopen(keep, "w") : nullptr;
#endif
        if (!out_) throw std::runtime_error("stdout could not be kept for the ready line");
    }
    // Graph.step's guard (nodus_graph/native.py): every edge of the loaded canvas must carry its FIFO config.
    void check_edges() {
        const int32_t n = nodus_headless_get_edge_count(g_);
        if (n < 0) throw std::runtime_error("native edge enumeration failed after load");
        std::string missing;
        for (int32_t e = 0; e < n; ++e) {
            NodusHeadlessFifoConfig c{};
            if (!nodus_headless_get_edge_fifo_config(g_, e, &c)) missing += (missing.empty() ? "" : ", ") + std::to_string(e);
        }
        if (!missing.empty())
            throw std::runtime_error("native execution requires an explicit FIFO config on edges [" + missing + "]");
    }
    void ready_line(const std::string& cron) {
        std::ostringstream s;
        s << "{\"canvas\": " << detail::json_string(o_.canvas) << ", \"contract_ports\": [";
        const int32_t n = nodus_headless_get_contract_port_count(g_);
        for (int32_t i = 0; i < n; ++i) {
            char prefix[1024] = {};
            int32_t capacity = 0, edge = 0, declared = 0, port = 0, hosts = 0, engines = 0;
            if (!nodus_headless_get_contract_port_at(g_, i, prefix, sizeof(prefix), &capacity, &edge, &declared))
                throw std::runtime_error("native contract port " + std::to_string(i) + " query failed");
            const bool bound = nodus_headless_get_contract_port(g_, prefix, &port, &hosts, &engines) != 0;
            s << (i ? ", " : "") << "{\"prefix\": " << detail::json_string(prefix) << ", \"listen_port\": " << (bound ? port : declared)
              << ", \"capacity\": " << capacity << ", \"edge\": " << edge << "}";
        }
        s << "], \"links\": [";
        for (size_t i = 0; i < links_.size(); ++i)
            s << (i ? ", " : "") << "{\"id\": " << detail::json_string(links_[i].first.id) << ", \"prefix\": "
              << detail::json_string(links_[i].first.prefix) << ", \"listen_port\": " << links_[i].second << "}";
#if defined(_WIN32)
        const long pid = static_cast<long>(_getpid());
#else
        const long pid = static_cast<long>(getpid());
#endif
        s << "], \"pid\": " << pid << ", \"console_port\": " << (console_ ? std::to_string(console_port_) : std::string("null"))
          << ", \"cron\": [" << cron << "]}";
        std::fprintf(out_, "%s\n", s.str().c_str());
        std::fflush(out_);
    }
    void save() {
        canvas_changed().store(false);
        if (!nodus_headless_save_file(g_, o_.canvas.c_str())) std::fprintf(stderr, "nodus: native save failed: %s\n", o_.canvas.c_str());
    }

    Options o_;
    std::unique_ptr<nodus::console::Server> console_;
    uint16_t console_port_ = 0;
    NodusHeadlessGraph* g_ = nullptr;
    FILE* out_ = nullptr;
    long long ticks_ = 0;
    bool failed_ = false;
    std::vector<std::pair<LinkHost, int32_t>> links_;
    std::vector<std::pair<std::string, uint64_t>> attached_, scheduled_;
};

}  // namespace nodus::host
