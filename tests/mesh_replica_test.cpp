// mesh_replica_test: a node's canvas and its traffic metrics, read on another node (docs/MESH_CANVAS_UP_PORTS_2026-11-21.md).
// Owner, 2026-11-21: "by having the canvas state we should collect also any meta edges supplying traffic metrics".
//
//   writer node: [feed Input | Output] -> canvas row (host, publish every 5) "in";  "out" -> the node's canvas port
//                "result"; "state" -> [Input | <id>.bulk.send | tensor_returns] (the node's own canvas, which only it
//                writes); tensor_returns -> [tensor_mint], a mint with no size: every sent snapshot is freed
//                its canvas: "in" -> pass table -> "out", a meta group listening to both edges
//   reader node: [<id>.bulk.receive | Output] -> canvas row (replica) "state"     (read only, never stepped)
// Values run through the writer's canvas while its snapshots cross the link. The replica then holds the writer's canvas
// (the same canvas ports), and its meta group holds the writer's records: the same group, the same member ropes, the
// traffic the writer saw. Saved, the reader's canvas carries the replica (ROW_STATE) and it loads again.
//
//   mesh_replica_test <work dir>   (its rows are built from module_library/source/tools in the headless runtime)
#include "canvas_abi.h"
#include "nodus_headless_abi.h"
#include "nodus_link.h"
#include "headless_rows.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;
#define CHECK(cond)                                                                                  \
    do {                                                                                             \
        if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

constexpr int32_t kTool = 1, kInput = 0, kOutput = 2, kPlugin = 1;
constexpr int32_t kRowPortBase = 20000, kRowPortStride = 64;
constexpr int32_t kRecordWords = 16;      // GP_META_RECORD_WORDS
constexpr int kRopeWord = 5, kWriteDeltaWord = 7;

// Tensor handles alive in the process, by describing every id up to `limit` (ids are small and reused).
int live_tensors(uint64_t limit = 1u << 16) {
    int n = 0;
    for (uint64_t id = 1; id <= limit; ++id) {
        NodusTensorDesc d{};
        if (nodus_tensor_describe(id, &d) == NODUS_OK) ++n;
    }
    return n;
}

std::vector<uint8_t> read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// A parsed snapshot (canvas_row.cpp): the canvas file and each meta group's records.
struct Snapshot {
    std::string canvas;
    struct Group { uint64_t id = 0; std::vector<uint64_t> words; };
    std::vector<Group> groups;
};
bool parse_snapshot(const std::vector<uint8_t>& b, Snapshot& s) {
    size_t at = 0;
    auto get = [&](void* v, size_t n) { if (n > b.size() - at) return false; std::memcpy(v, b.data() + at, n); at += n; return true; };
    uint32_t magic = 0, version = 0, groups = 0;
    uint64_t size = 0;
    if (!get(&magic, 4) || magic != 0x3153434E || !get(&version, 4) || !get(&size, 8) || size > b.size() - at) return false;
    s.canvas.assign(reinterpret_cast<const char*>(b.data() + at), static_cast<size_t>(size));
    at += static_cast<size_t>(size);
    uint32_t files = 0;
    if (!get(&files, 4)) return false;
    for (uint32_t f = 0; f < files; ++f) {                  // the files beside the canvas (<file>.modules/)
        uint32_t name = 0;
        uint64_t bytes = 0;
        if (!get(&name, 4) || name > b.size() - at) return false;
        at += name;
        if (!get(&bytes, 8) || bytes > b.size() - at) return false;
        at += static_cast<size_t>(bytes);
    }
    if (!get(&groups, 4)) return false;
    for (uint32_t g = 0; g < groups; ++g) {
        Snapshot::Group grp;
        uint32_t n = 0;
        if (!get(&grp.id, 8) || !get(&n, 4)) return false;
        grp.words.resize(static_cast<size_t>(n) * kRecordWords);
        if (!get(grp.words.data(), grp.words.size() * 8)) return false;
        s.groups.push_back(std::move(grp));
    }
    return at == b.size();
}
// A canvas row's saved state: NCR1 v2, role, interval, snapshot size, snapshot.
std::vector<uint8_t> row_state(uint32_t role, uint32_t every, const std::vector<uint8_t>& snap) {
    std::vector<uint8_t> s;
    const uint32_t head[4] = {0x3152434E, 2, role, every};
    const uint64_t size = snap.size();
    s.insert(s.end(), reinterpret_cast<const uint8_t*>(head), reinterpret_cast<const uint8_t*>(head) + sizeof(head));
    s.insert(s.end(), reinterpret_cast<const uint8_t*>(&size), reinterpret_cast<const uint8_t*>(&size) + 8);
    s.insert(s.end(), snap.begin(), snap.end());
    return s;
}
bool snapshot_of_row(NodusHeadlessGraph* g, int32_t module, int32_t row, Snapshot& out) {
    size_t size = 0;
    nodus_headless_get_row_state(g, module, row, nullptr, 0, &size);
    if (size < 24) return false;
    std::vector<uint8_t> st(size);
    if (!nodus_headless_get_row_state(g, module, row, st.data(), st.size(), &size)) return false;
    // NCR1 v4: head 16 | mesh 4 | reader count 4 | readers | writer size 4 | writer | last generation 8 | size 8 | snapshot |
    // held tokens
    size_t at = 16 + 4;
    uint32_t readers = 0, writer = 0;
    if (st.size() < at + 4) return false;
    std::memcpy(&readers, st.data() + at, 4);
    at += 4 + static_cast<size_t>(readers) * 32;
    if (st.size() < at + 4) return false;
    std::memcpy(&writer, st.data() + at, 4);
    at += 4 + writer + 8;
    uint64_t snap = 0;
    if (st.size() < at + 8) return false;
    std::memcpy(&snap, st.data() + at, 8);
    at += 8;
    if (st.size() < at + snap) return false;                   // v4: the held tokens follow the snapshot
    return parse_snapshot(std::vector<uint8_t>(st.begin() + static_cast<std::ptrdiff_t>(at),
                                               st.begin() + static_cast<std::ptrdiff_t>(at + snap)), out);
}

struct Canvas {
    NodusHeadlessGraph* g = nodus_headless_create();
    NodusHeadlessFifoConfig tensor{8, 4, 0, 0, 2, 16, 0};
    NodusHeadlessFifoConfig scalar{4, 8, 0, -1, 2, 0, 0};
    Canvas() { tensor.type_id = nodus_headless_abstract_tensor_type_id(); }
    ~Canvas() { if (g) nodus_headless_destroy(g); }
    // contract: 1 = tensor handles on the stack; -1 = the default (scalars).
    int table(const char* label, std::vector<GP_CanvasModuleRowDesc> rows, int contract) {
        GP_CanvasModuleDesc d{};
        d.w = 200; d.h = 120;
        std::snprintf(d.label, sizeof(d.label), "%s", label);
        const int m = nodus_headless_add_module(g, &d);
        if (m < 0 || !nodus_headless_set_module_rows(g, m, rows.data(), static_cast<int32_t>(rows.size())) ||
            (contract >= 0 && !nodus_headless_set_module_stack_contract(g, m, contract)))
            return -1;
        return m;
    }
    int edge(int a, int ac, int b, int bc, const NodusHeadlessFifoConfig& c) {
        GP_CanvasEdgeDesc e{a, ac, b, bc};
        const int idx = nodus_headless_add_edge(g, &e, 0);
        return idx >= 0 && nodus_headless_configure_edge_fifo(g, idx, &c) ? idx : -1;
    }
};

// The writer's own canvas: "in" -> pass table -> "out", a meta group (channel 7) listening to both edges. Its snapshot
// carries no records yet (it has never run): the canvas file, the files saved beside it, and no records.
std::vector<uint8_t> writer_canvas(const fs::path& dir) {
    Canvas c;
    const int32_t in = nodus_headless_declare_canvas_port(c.g, "in", 1, 1);
    const int32_t out = nodus_headless_declare_canvas_port(c.g, "out", 0, 1);
    const int32_t root = nodus_headless_canvas_root_module(c.g);
    const int pass = c.table("pass", {{kInput, 0, 1, 0, nullptr}, {kOutput, 0, 1, 0, nullptr}}, -1);
    const int e0 = c.edge(root, in, pass, 0, c.scalar), e1 = c.edge(pass, 1, root, out, c.scalar);
    CHECK(in >= 0 && out >= 0 && root >= 0 && pass >= 0 && e0 >= 0 && e1 >= 0);
    const int32_t members[2] = {e0, e1};
    CHECK(nodus_headless_meta_create(c.g, members, 2, 7) != 0);
    const auto path = dir / "writer_child.nodus.txt";
    CHECK(nodus_headless_save_file(c.g, path.string().c_str()));
    const std::vector<uint8_t> canvas = read_file(path);
    std::vector<uint8_t> snap;
    auto put = [&](const void* v, size_t n) { const auto* b = static_cast<const uint8_t*>(v); snap.insert(snap.end(), b, b + n); };
    const uint32_t head[2] = {0x3153434E, 2};
    const uint64_t size = canvas.size();
    put(head, 8);
    put(&size, 8);
    put(canvas.data(), canvas.size());
    std::vector<std::pair<std::string, std::vector<uint8_t>>> files;
    const fs::path modules = path.string() + ".modules";
    for (const auto& e : fs::recursive_directory_iterator(modules))
        if (e.is_regular_file()) files.emplace_back(fs::relative(e.path(), modules).generic_string(), read_file(e.path()));
    const uint32_t count = static_cast<uint32_t>(files.size());
    put(&count, 4);
    for (const auto& f : files) {
        const uint32_t name = static_cast<uint32_t>(f.first.size());
        const uint64_t bytes = f.second.size();
        put(&name, 4);
        put(f.first.data(), f.first.size());
        put(&bytes, 8);
        put(f.second.data(), f.second.size());
    }
    const uint32_t groups = 0;
    put(&groups, 4);
    return snap;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: mesh_replica_test <work dir>\n"); return 2; }
    const fs::path lib = fs::path(argv[1]) / ("mesh_replica_test_rows_" + std::to_string(std::time(nullptr)));
    const std::string row_s = headless_rows::build_named("canvas_row", lib), mint_s = headless_rows::build_named("tensor_mint", lib),
                      returns_s = headless_rows::build_named("tensor_returns", lib);
    if (row_s.empty() || mint_s.empty() || returns_s.empty()) return 2;
    const char *row_id = row_s.c_str(), *mint_id = mint_s.c_str(), *returns_id = returns_s.c_str();
    const fs::path dir = fs::path(argv[1]) / ("mesh_replica_test_" + std::to_string(std::time(nullptr)));
    fs::create_directories(dir);
    const std::vector<uint8_t> child = writer_canvas(dir);

    Canvas writer, reader;
    nodus::link::Key key{};
    nodus::link::random_bytes(key.data(), key.size());
    const int32_t port = nodus_headless_register_link_host(reader.g, "mesh.r", "mesh", 0, key.data(), 0);
    CHECK(port > 0);
    const auto id = nodus::link::new_link_id();
    const uint64_t now = uint64_t(std::time(nullptr));
    using nodus::link::Channel;
    using nodus::link::Role;
    const auto ct = nodus::link::sign_ticket(nodus::link::make_ticket(Channel::Control, Role::Dialer, id, "mesh", 9, now, 60), key);
    const auto bt = nodus::link::sign_ticket(nodus::link::make_ticket(Channel::Bulk, Role::Dialer, id, "mesh", 9, now, 60), key);
    CHECK(nodus_headless_register_link_dialer(writer.g, "mesh.w", "127.0.0.1", uint16_t(port), "mesh", ct.data(), bt.data(), 5000, 0));

    // Writer node.
    const int feed = writer.table("feed", {{kInput, 0, 1, 0, nullptr}, {kOutput, 0, 1, 0, nullptr}}, -1);
    CHECK(nodus_headless_declare_input_port_ex(writer.g, "feed", feed, 0, &writer.scalar));
    const int node = writer.table("node", {{kTool, 0, 0, kPlugin, row_id}}, 1);
    const auto host_state = row_state(0, 5, child);
    CHECK(nodus_headless_set_row_state(writer.g, node, 0, host_state.data(), host_state.size()));
    const int32_t result = nodus_headless_declare_canvas_port(writer.g, "result", 0, 1);
    const int32_t wroot = nodus_headless_canvas_root_module(writer.g);
    const int live_before = live_tensors();
    const int send = writer.table("send", {{kInput, 0, 1, 0, nullptr}, {kTool, 0, 0, kPlugin, "mesh.w.bulk.send"},
                                           {kTool, 0, 0, kPlugin, returns_id}}, 1);
    const int trash = writer.table("trash", {{kTool, 0, 0, kPlugin, mint_id}}, 1);
    CHECK(trash >= 0 && writer.edge(send, kRowPortBase + 2 * kRowPortStride, trash, kRowPortBase, writer.tensor) >= 0);
    CHECK(nodus_headless_set_module_stack_loopback(writer.g, send, 1));
    CHECK(writer.edge(feed, 1, node, kRowPortBase + 0, writer.scalar) >= 0);              // row port 0: "in"
    CHECK(writer.edge(node, kRowPortBase + 1, wroot, result, writer.scalar) >= 0);        // row port 1: "out"
    CHECK(writer.edge(node, kRowPortBase + 2, send, 0, writer.tensor) >= 0);              // row port 2: "state"

    // Reader node.
    const int recv = reader.table("receive", {{kTool, 0, 0, kPlugin, "mesh.r.bulk.receive"}, {kOutput, 0, 1, 0, nullptr}}, 1);
    CHECK(nodus_headless_set_module_stack_loopback(reader.g, recv, 1));
    const int replica = reader.table("replica of the writer", {{kTool, 0, 0, kPlugin, row_id}}, 1);
    const auto empty_replica = row_state(1, 0, {});
    CHECK(nodus_headless_set_row_state(reader.g, replica, 0, empty_replica.data(), empty_replica.size()));
    CHECK(reader.edge(recv, 0, replica, kRowPortBase + 0, reader.tensor) >= 0);           // the replica's "state"

    int32_t hs = -1, ds = -1;
    uint64_t hv[16] = {}, dv[16] = {};
    for (int i = 0; i < 2000 && !(hs == 1 && ds == 1); ++i) {
        nodus_headless_step(writer.g, 1.0 / 60.0); nodus_headless_step(reader.g, 1.0 / 60.0);
        nodus_headless_link_report("mesh.r", &hs, hv); nodus_headless_link_report("mesh.w", &ds, dv);
    }
    CHECK(hs == 1 && ds == 1);

    int next = 1, through = 0;
    for (int tick = 0; tick < 400; ++tick) {
        if (next <= 40) {
            const float v = static_cast<float>(next);
            NodusHeadlessPublishResult r{};
            if (nodus_headless_push_token_ex(writer.g, "feed", &v, sizeof(v), &r) && r.accepted) ++next;
        }
        nodus_headless_step(writer.g, 1.0 / 60.0);
        nodus_headless_step(reader.g, 1.0 / 60.0);
        for (;;) {
            float v = 0.0f;
            size_t w = 0;
            if (!nodus_headless_canvas_port_consume(writer.g, "result", &v, sizeof(v), &w) || w != sizeof(v)) break;
            if (v != 0.0f) ++through;
        }
    }
    // The writer stops; the reader steps until every snapshot the writer sent has arrived.
    nodus_headless_link_report("mesh.w", &ds, dv);
    for (int tick = 0; tick < 2000; ++tick) {
        nodus_headless_step(reader.g, 1.0 / 60.0);
        nodus_headless_link_report("mesh.r", &hs, hv);
        if (hv[7] >= dv[6]) break;
    }
    std::printf("writer: %d of 40 values through its canvas; snapshots sent %llu, received %llu\n", through,
                (unsigned long long)dv[6], (unsigned long long)hv[7]);
    for (int tick = 0; tick < 60; ++tick) { nodus_headless_step(writer.g, 1.0 / 60.0); nodus_headless_step(reader.g, 1.0 / 60.0); }
    const int live_after = live_tensors();
    std::printf("tensors alive before %d, after %d\n", live_before, live_after);
    // A few snapshots may be in flight (the writer keeps publishing); a leak would leave every one sent (about 80).
    CHECK(live_after <= live_before + 4);
    CHECK(through == 40);
    CHECK(dv[6] >= 2 && hv[7] == dv[6]);

    Snapshot w, r;
    CHECK(snapshot_of_row(writer.g, node, 0, w));
    CHECK(snapshot_of_row(reader.g, replica, 0, r));
    std::printf("meta groups: writer %zu, replica %zu\n", w.groups.size(), r.groups.size());
    CHECK(w.groups.size() == 1 && r.groups.size() == 1);
    if (w.groups.size() == 1 && r.groups.size() == 1) {
        const auto& wg = w.groups[0];
        const auto& rg = r.groups[0];
        CHECK(rg.id == wg.id);
        const size_t rn = rg.words.size() / kRecordWords, wn = wg.words.size() / kRecordWords;
        uint64_t r_writes = 0, w_writes = 0;
        for (size_t i = 0; i < rn; ++i) r_writes += rg.words[i * kRecordWords + kWriteDeltaWord];
        for (size_t i = 0; i < wn; ++i) w_writes += wg.words[i * kRecordWords + kWriteDeltaWord];
        // The replica's records are a prefix of the writer's: everything up to the last snapshot that crossed.
        bool prefix = rn > 0 && rn <= wn;
        for (size_t i = 0; prefix && i < rn * kRecordWords; ++i) prefix = rg.words[i] == wg.words[i];
        CHECK(prefix);
        CHECK(r_writes > 0);
        std::printf("replica: meta group %llu, %zu of the writer's %zu records (writes seen %llu of %llu), member rope %llu\n",
                    (unsigned long long)rg.id, rn, wn, (unsigned long long)r_writes, (unsigned long long)w_writes,
                    (unsigned long long)(rn ? rg.words[kRopeWord] : 0));
    }
    // The canvas ports' records, whatever line ends the platform wrote (CRLF on Windows: review f1a3eb8f).
    std::string lines = r.canvas;
    lines.erase(std::remove(lines.begin(), lines.end(), '\r'), lines.end());
    CHECK(lines.find("CANVAS_PORT ") != std::string::npos && lines.find(" in\n") != std::string::npos &&
          lines.find(" out\n") != std::string::npos);

    // Saved, the reader's canvas carries its replica; loaded into another graph, the replica is there with its records.
    const auto saved = dir / "reader.nodus.txt";
    CHECK(nodus_headless_save_file(reader.g, saved.string().c_str()));
    Canvas again;
    CHECK(nodus_headless_load_file(again.g, saved.string().c_str()));
    Snapshot a;
    CHECK(snapshot_of_row(again.g, replica, 0, a));
    CHECK(a.groups.size() == 1 && !r.groups.empty() && a.groups[0].words == r.groups[0].words);
    std::printf("reloaded reader: the replica's %zu records are back\n", a.groups.empty() ? 0 : a.groups[0].words.size() / kRecordWords);

    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
