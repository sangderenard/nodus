// mesh_secure_test: one writer per canvas, and read privacy, across real processes (docs/MESH_CANVAS_UP_PORTS_2026-11-21.md;
// module_library/source/tools/mesh_crypto.py; include/canvas_row_tool.h).
//
// Each node is its own process with its own keys (NODUS_MESH_KEYS). Node A writes a canvas whose one reader is B:
//   writer  [canvas_row host, mesh, readers {B}] "state" -> [Input | seal | sign | returns | Output]
//           -> [Input | <id>.bulk.send | returns]   (both stack loopback; each returns row feeds a trash mint)
//   reader  [<id>.bulk.receive | verify | open | returns | Output] -> [canvas_row replica, writer W]
// Scenarios, each a writer process and a reader process over a link on 127.0.0.1:
//   B, writer A   B opens the canvas: readable, a rising generation, the canvas is held
//   C, writer A   C is no reader: it learns that A's canvas exists and its generation, and holds no canvas
//   B, writer A, but the snapshots come from D (a node B knows): never adopted, the replica stays empty
// Keys are hybrid (Ed25519 + ML-DSA-65 signatures, X25519 + ML-KEM-768 sealing), made by mesh_crypto.py keygen.
//
//   mesh_secure_test <python> <python site-packages> <canvas_row> <tensor_mint> <tensor_returns> <work dir>
#include "canvas_abi.h"
#include "exec_pipe.h"
#include "nodus_headless_abi.h"
#include "nodus_link.h"
#include "plugin_manager.h"
#include "python_row_tool.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int32_t kTool = 1, kInput = 0, kOutput = 2, kPlugin = 1;
constexpr int32_t kRowPortBase = 20000, kRowPortStride = 64;
const char* kRows[4] = {"seal", "sign", "verify", "open"};

std::vector<uint8_t> read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; }
    return s;
}
std::vector<uint8_t> unhex(const std::string& s) {
    std::vector<uint8_t> out;
    for (size_t i = 0; i + 1 < s.size(); i += 2) out.push_back(static_cast<uint8_t>(std::stoi(s.substr(i, 2), nullptr, 16)));
    return out;
}
void set_env(const char* name, const std::string& value) {
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

struct Canvas {
    NodusHeadlessGraph* g = nodus_headless_create();
    NodusHeadlessFifoConfig tensor{8, 4, 0, 0, 2, 16, 0};
    Canvas() { tensor.type_id = nodus_headless_abstract_tensor_type_id(); }
    ~Canvas() { if (g) nodus_headless_destroy(g); }
    int table(const char* label, std::vector<GP_CanvasModuleRowDesc> rows, bool loopback) {
        GP_CanvasModuleDesc d{};
        d.w = 200; d.h = 120;
        std::snprintf(d.label, sizeof(d.label), "%s", label);
        const int m = nodus_headless_add_module(g, &d);
        if (m < 0 || !nodus_headless_set_module_rows(g, m, rows.data(), static_cast<int32_t>(rows.size())) ||
            !nodus_headless_set_module_stack_contract(g, m, 1) || (loopback && !nodus_headless_set_module_stack_loopback(g, m, 1)))
            return -1;
        return m;
    }
    bool edge(int a, int ac, int b, int bc) {
        GP_CanvasEdgeDesc e{a, ac, b, bc};
        const int idx = nodus_headless_add_edge(g, &e, 0);
        return idx >= 0 && nodus_headless_configure_edge_fifo(g, idx, &tensor);
    }
    // A returns row at `row` of table `t` hands what the table's Borrower rows used up to a trash mint of its own.
    bool trash(int t, int row, const char* mint_id) {
        const int m = table("trash", {{kTool, 0, 0, kPlugin, mint_id}}, false);
        return m >= 0 && edge(t, kRowPortBase + row * kRowPortStride, m, kRowPortBase);
    }
};

// A canvas row's saved state, version 3.
std::vector<uint8_t> row_state(uint32_t role, uint32_t every, bool mesh, const std::vector<uint8_t>& readers,
                               const std::vector<uint8_t>& writer, const std::vector<uint8_t>& snap) {
    std::vector<uint8_t> s;
    auto put = [&](const void* v, size_t n) { const auto* b = static_cast<const uint8_t*>(v); s.insert(s.end(), b, b + n); };
    const uint32_t head[4] = {0x3152434E, 3, role, every};
    const uint32_t m[2] = {mesh ? 1u : 0u, static_cast<uint32_t>(readers.size() / 32)};
    const uint32_t ws = static_cast<uint32_t>(writer.size());
    const uint64_t last = 0, size = snap.size();
    put(head, sizeof(head)); put(m, sizeof(m)); put(readers.data(), readers.size()); put(&ws, 4); put(writer.data(), writer.size());
    put(&last, 8); put(&size, 8); put(snap.data(), snap.size());
    return s;
}

// The writer's canvas: one table and a meta group over nothing; its snapshot (NCS1 v2) carries the saved files.
std::vector<uint8_t> small_canvas(const fs::path& dir) {
    Canvas c;
    c.table("work", {{kInput, 0, 1, 0, nullptr}, {kOutput, 0, 1, 0, nullptr}}, false);
    const auto path = dir / "writer_canvas.nodus.txt";
    nodus_headless_save_file(c.g, path.string().c_str());
    const std::vector<uint8_t> canvas = read_file(path);
    std::vector<uint8_t> snap;
    auto put = [&](const void* v, size_t n) { const auto* b = static_cast<const uint8_t*>(v); snap.insert(snap.end(), b, b + n); };
    const uint32_t head[2] = {0x3153434E, 2};
    const uint64_t size = canvas.size();
    put(head, 8); put(&size, 8); put(canvas.data(), canvas.size());
    std::vector<std::pair<std::string, std::vector<uint8_t>>> files;
    const fs::path modules = path.string() + ".modules";
    std::error_code ec;
    for (fs::recursive_directory_iterator it(modules, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file()) files.emplace_back(fs::relative(it->path(), modules).generic_string(), read_file(it->path()));
    const uint32_t count = static_cast<uint32_t>(files.size());
    put(&count, 4);
    for (const auto& f : files) {
        const uint32_t n = static_cast<uint32_t>(f.first.size());
        const uint64_t b = f.second.size();
        put(&n, 4); put(f.first.data(), n); put(&b, 8); put(f.second.data(), f.second.size());
    }
    const uint32_t groups = 0;
    put(&groups, 4);
    return snap;
}

bool load(const char* path, std::string& id) {
    char out[512] = {};
    if (!gp_plugin_load_from_path(path, out, sizeof(out))) { std::fprintf(stderr, "cannot load %s\n", path); return false; }
    id = out;
    return true;
}

// ---------------------------------------------------------------------------------------------------- the writer
// writer <work> <port> <control ticket> <bulk ticket> <reader kid hex | -> <canvas_row> <mint> <returns> <seal> <sign>
int writer(char** argv) {
    const fs::path work = argv[2];
    std::string row, mint, returns, seal, sign;
    if (!load(argv[7], row) || !load(argv[8], mint) || !load(argv[9], returns) || !load(argv[10], seal) || !load(argv[11], sign))
        return 2;
    Canvas c;
    const auto ct = unhex(argv[4]), bt = unhex(argv[5]);
    if (!nodus_headless_register_link_dialer(c.g, "mesh.w", "127.0.0.1", uint16_t(std::stoi(argv[3])), "mesh", ct.data(),
                                             bt.data(), 10000, 0)) return 3;
    const std::vector<uint8_t> readers = std::string(argv[6]) == "-" ? std::vector<uint8_t>{} : unhex(argv[6]);
    const int node = c.table("node", {{kTool, 0, 0, kPlugin, row.c_str()}}, false);
    const auto state = row_state(0, 5, true, readers, {}, small_canvas(work));
    if (node < 0 || !nodus_headless_set_row_state(c.g, node, 0, state.data(), state.size())) return 4;
    const int secure = c.table("secure", {{kInput, 0, 1, 0, nullptr}, {kTool, 0, 0, kPlugin, seal.c_str()},
                                          {kTool, 0, 0, kPlugin, sign.c_str()}, {kTool, 0, 0, kPlugin, returns.c_str()},
                                          {kOutput, 0, 1, 0, nullptr}}, true);
    const int send = c.table("send", {{kInput, 0, 1, 0, nullptr}, {kTool, 0, 0, kPlugin, "mesh.w.bulk.send"},
                                      {kTool, 0, 0, kPlugin, returns.c_str()}}, true);
    if (secure < 0 || send < 0 || !c.edge(node, kRowPortBase, secure, 0) || !c.edge(secure, 1, send, 0) ||
        !c.trash(secure, 3, mint.c_str()) || !c.trash(send, 2, mint.c_str())) return 5;
    std::printf("[writer] up\n");
    std::fflush(stdout);
    for (;;) {                                      // until the orchestrator ends it
        nodus_headless_step(c.g, 1.0 / 60.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

// ---------------------------------------------------------------------------------------------------- the reader
// reader <port> <link key hex> <writer kid hex> <seconds> <canvas_row> <mint> <returns> <verify> <open>
int reader(char** argv) {
    std::string row, mint, returns, verify, open;
    if (!load(argv[6], row) || !load(argv[7], mint) || !load(argv[8], returns) || !load(argv[9], verify) || !load(argv[10], open))
        return 2;
    Canvas c;
    const auto key = unhex(argv[3]);
    if (nodus_headless_register_link_host(c.g, "mesh.r", "mesh", uint16_t(std::stoi(argv[2])), key.data(), 0) <= 0) return 3;
    const int receive = c.table("receive", {{kTool, 0, 0, kPlugin, "mesh.r.bulk.receive"}, {kTool, 0, 0, kPlugin, verify.c_str()},
                                            {kTool, 0, 0, kPlugin, open.c_str()}, {kTool, 0, 0, kPlugin, returns.c_str()},
                                            {kOutput, 0, 1, 0, nullptr}}, true);
    const int replica = c.table("replica", {{kTool, 0, 0, kPlugin, row.c_str()}}, false);
    const auto state = row_state(1, 0, false, {}, unhex(argv[4]), {});
    if (receive < 0 || replica < 0 || !nodus_headless_set_row_state(c.g, replica, 0, state.data(), state.size()) ||
        !c.edge(receive, 0, replica, kRowPortBase) || !c.trash(receive, 3, mint.c_str())) return 4;
    std::printf("[reader] up\n");
    std::fflush(stdout);
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(std::stoi(argv[5]));
    uint64_t first = 0, last = 0;
    uint32_t readable_seen = 0;
    size_t snap_bytes = 0;
    int32_t hs = -1;
    uint64_t hv[16] = {};
    while (std::chrono::steady_clock::now() < until) {
        nodus_headless_step(c.g, 1.0 / 60.0);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        size_t size = 0;
        nodus_headless_get_row_state(c.g, replica, 0, nullptr, 0, &size);
        std::vector<uint8_t> s(size);
        if (size < 24 || !nodus_headless_get_row_state(c.g, replica, 0, s.data(), s.size(), &size)) continue;
        // v3: head 16 | mesh 4 | n 4 | readers | writer size 4 | writer | last generation 8 | snapshot size 8 | snapshot
        size_t at = 16 + 4;
        uint32_t n = 0, ws = 0;
        std::memcpy(&n, s.data() + at, 4); at += 4 + n * 32;
        std::memcpy(&ws, s.data() + at, 4); at += 4 + ws;
        uint64_t generation = 0, sb = 0;
        std::memcpy(&generation, s.data() + at, 8); at += 8;
        std::memcpy(&sb, s.data() + at, 8);
        if (generation && !first) first = generation;
        if (generation > last) { last = generation; if (sb) ++readable_seen; }
        snap_bytes = static_cast<size_t>(sb);
    }
    nodus_headless_link_report("mesh.r", &hs, hv);
    std::printf("RESULT linked %d snapshots_received %llu first %llu last %llu readable_generations %u snapshot_bytes %zu\n",
                hs, (unsigned long long)hv[7], (unsigned long long)first, (unsigned long long)last, readable_seen, snap_bytes);
    std::fflush(stdout);
    return 0;
}

// ---------------------------------------------------------------------------------------------------- the test
int failures = 0;
#define CHECK(cond, ...)                                                                             \
    do {                                                                                             \
        if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

bool run(const nodus::exec::Program& p, std::string& out, int seconds) {
    nodus::exec::Child child;
    std::string why;
    if (!child.start(p, why)) { std::fprintf(stderr, "start %s: %s\n", p.path.c_str(), why.c_str()); return false; }
    child.finish_input();
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (child.running() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::string more;
    child.take(more);
    out += more;
    return !child.running() && child.exit_code() == 0;
}

// The plugin of one mesh_crypto row: its descriptor and generated wrapper, built and loaded the ordinary way.
std::string build_row(const char* name, const fs::path& source, const fs::path& library, const fs::path& nodus_root) {
    nodus::python_row::Descriptor d;
    d.tool_id = std::string("nodus.mesh.") + name;
    d.tool_name = std::string("Mesh ") + name;
    d.source_path = fs::absolute(source).generic_string();
    d.module_name = "mesh_crypto";
    d.callable_name = name;
    d.inputs = {{"in", nodus::python_row::PrimitiveType::Bytes}};
    d.outputs = {{"out", nodus::python_row::PrimitiveType::Bytes}};
    const fs::path wrapper = library / "source" / (std::string("mesh_") + name + ".cpp");
    fs::create_directories(wrapper.parent_path());
    std::string error;
    if (!nodus::python_row::write_plugin_source(d, wrapper.string(), &error)) { std::fprintf(stderr, "%s\n", error.c_str()); return {}; }
    char id[512] = {};
    if (!nodus_headless_build_plugin_source(wrapper.string().c_str(), nodus_root.string().c_str(), library.string().c_str(), id, sizeof(id)))
        return {};
    for (const auto& e : fs::recursive_directory_iterator(library / "bin"))
        if (e.is_regular_file() && e.path().filename().string().find(std::string("mesh_") + name) != std::string::npos &&
            (e.path().extension() == ".so" || e.path().extension() == ".dll"))
            return e.path().string();
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "writer") return writer(argv);
    if (argc >= 2 && std::string(argv[1]) == "reader") return reader(argv);
    if (argc < 7) {
        std::fprintf(stderr, "usage: mesh_secure_test <python> <site-packages> <canvas_row> <tensor_mint> <tensor_returns> <work dir>\n");
        return 2;
    }
    const fs::path work = fs::path(argv[6]) / ("mesh_secure_test_" + std::to_string(std::time(nullptr)));
    fs::create_directories(work);
    const fs::path nodus_root = fs::path(NODUS_REPO_ROOT);
    const fs::path crypto = nodus_root / "module_library" / "source" / "tools" / "mesh_crypto.py";
    set_env("PYTHONPATH", argv[2]);

    // Four nodes' keys. A, B and C know one another; B also knows D.
    std::vector<uint8_t> kid[4];
    const char* names[4] = {"A", "B", "C", "D"};
    for (int i = 0; i < 4; ++i) {
        nodus::exec::Program p;
        p.path = argv[1];
        p.working_dir = work.string();
        p.arguments = {crypto.string(), "keygen", (work / names[i]).string()};
        std::string out;
        if (!run(p, out, 60)) { std::fprintf(stderr, "keygen %s: %s\n", names[i], out.c_str()); return 1; }
        const auto block = read_file(work / names[i] / "node.pub");
        kid[i] = std::vector<uint8_t>(32);
        nodus::link::platform::sha256(block.data(), block.size(), kid[i].data());
    }
    auto know = [&](int who, int whom) {
        fs::copy_file(work / names[whom] / "node.pub", work / names[who] / "peers" / (hex(kid[whom].data(), 32) + ".pub"),
                      fs::copy_options::overwrite_existing);
    };
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) if (a != b) know(a, b);
    know(1, 3);
    know(3, 1);
    std::printf("keys: A %s.. B %s.. C %s.. D %s..\n", hex(kid[0].data(), 4).c_str(), hex(kid[1].data(), 4).c_str(),
                hex(kid[2].data(), 4).c_str(), hex(kid[3].data(), 4).c_str());

    const fs::path library = work / "module_library";
    std::string rows[4];
    for (int i = 0; i < 4; ++i) {
        rows[i] = build_row(kRows[i], crypto, library, nodus_root);
        if (rows[i].empty()) { std::fprintf(stderr, "the %s row did not build\n", kRows[i]); return 1; }
    }
    std::printf("rows built: seal sign verify open\n");

    struct Scenario { const char* what; int writer_keys; int reader_keys; int expected_writer; };
    const Scenario scenarios[3] = {{"B reads A's canvas", 0, 1, 0}, {"C is no reader of A's canvas", 0, 2, 0},
                                   {"B takes only A: D's snapshots", 3, 1, 0}};
    std::string results[3];
    for (int s = 0; s < 3; ++s) {
        const auto& sc = scenarios[s];
        nodus::link::Key key{};
        nodus::link::random_bytes(key.data(), key.size());
        std::srand(static_cast<unsigned>(std::time(nullptr)) + s);
        const int port = 41000 + (std::rand() % 2000) * 3 + s;   // the reader's link host, on 127.0.0.1
        const auto link = nodus::link::new_link_id();
        const uint64_t now = uint64_t(std::time(nullptr));
        const auto ct = nodus::link::sign_ticket(nodus::link::make_ticket(nodus::link::Channel::Control, nodus::link::Role::Dialer, link, "mesh", 7, now, 120), key);
        const auto bt = nodus::link::sign_ticket(nodus::link::make_ticket(nodus::link::Channel::Bulk, nodus::link::Role::Dialer, link, "mesh", 7, now, 120), key);

        set_env("NODUS_MESH_KEYS", (work / names[sc.reader_keys]).string());
        nodus::exec::Program rp;
        rp.path = fs::absolute(argv[0]).string();
        rp.working_dir = work.string();
        rp.arguments = {"reader", std::to_string(port), hex(key.data(), key.size()), hex(kid[sc.expected_writer].data(), 32), "25",
                        argv[3], argv[4], argv[5], rows[2], rows[3]};
        nodus::exec::Child reader_child;
        std::string why;
        if (!reader_child.start(rp, why)) { std::fprintf(stderr, "reader: %s\n", why.c_str()); return 1; }
        reader_child.finish_input();
        std::this_thread::sleep_for(std::chrono::seconds(2));

        set_env("NODUS_MESH_KEYS", (work / names[sc.writer_keys]).string());
        nodus::exec::Program wp;
        wp.path = rp.path;
        wp.working_dir = work.string();
        wp.arguments = {"writer", work.string(), std::to_string(port), hex(ct.data(), ct.size()), hex(bt.data(), bt.size()),
                        hex(kid[1].data(), 32), argv[3], argv[4], argv[5], rows[0], rows[1]};
        nodus::exec::Child writer_child;
        if (!writer_child.start(wp, why)) { std::fprintf(stderr, "writer: %s\n", why.c_str()); return 1; }
        writer_child.finish_input();

        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(40);
        while (reader_child.running() && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::string out;
        reader_child.take(out);
        writer_child.kill();
        const auto at = out.find("RESULT ");
        results[s] = at == std::string::npos ? std::string() : out.substr(at, out.find('\n', at) - at);
        std::printf("%s: %s\n", sc.what, results[s].empty() ? "(no result)" : results[s].c_str());
        if (results[s].empty()) {
            std::string wout;
            writer_child.take(wout);
            std::printf("  reader output tail: %s\n  writer output tail: %s\n", out.substr(out.size() > 1500 ? out.size() - 1500 : 0).c_str(),
                        wout.substr(wout.size() > 1500 ? wout.size() - 1500 : 0).c_str());
        }
    }
    auto field = [](const std::string& r, const char* name) -> unsigned long long {
        const auto at = r.find(std::string(" ") + name + " ");
        return at == std::string::npos ? ~0ull : std::strtoull(r.c_str() + at + std::strlen(name) + 2, nullptr, 10);
    };
    // B: snapshots arrive, at rising generations, and the canvas is held.
    CHECK(field(results[0], "snapshots_received") >= 2);
    CHECK(field(results[0], "last") > field(results[0], "first") && field(results[0], "first") > 0);
    CHECK(field(results[0], "readable_generations") >= 2 && field(results[0], "snapshot_bytes") > 0);
    // C: the same stream; it knows the writer's generation and holds no canvas.
    CHECK(field(results[1], "snapshots_received") >= 2 && field(results[1], "last") > 0);
    CHECK(field(results[1], "readable_generations") == 0 && field(results[1], "snapshot_bytes") == 0);
    // B named A as the writer; D's snapshots arrive, verify (B knows D), and are never adopted.
    CHECK(field(results[2], "snapshots_received") >= 2);
    CHECK(field(results[2], "last") == 0 && field(results[2], "snapshot_bytes") == 0);
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
