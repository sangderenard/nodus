// canvas_row_test: a canvas nested in a table row (module_library/source/tools/canvas_row.cpp,
// docs/MESH_CANVAS_UP_PORTS_2026-11-21.md stage 3).
//
// A child canvas passes values from its IN canvas port "in" through one table to its OUT canvas port "out", once with
// that table stepping on the tick the canvas row gives the child, once with it free-running (exec mode Free). A parent
// canvas holds it in a canvas row: the parent's feed table drives the row's Argument port "in", and the row's Return
// port "out" ends at the parent's own OUT canvas port "result". Two canvases coexist in one process, each on its own
// scheduler; every value arrives at the top, in order. The parent is saved (the row's state, the child canvas, rides in
// a ROW_STATE record) and loaded into a third graph, whose restored canvas row carries values again.
//
//   canvas_row_test <path to the canvas_row plugin library>
#include "canvas_abi.h"
#include "nodus_headless_abi.h"
#include "headless_rows.h"
// The canvas row itself, compiled in for the custody case (its exports renamed, as the image composes a plugin source).
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME canvas_row_test_create
#define NODUS_PLUGIN_DESTROY_NAME canvas_row_test_destroy
#define NODUS_PLUGIN_SOURCE_NAME canvas_row_test_source
#include "../module_library/source/tools/canvas_row.cpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int failures = 0;
#define CHECK(cond)                                                                                  \
    do {                                                                                             \
        if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

constexpr int32_t kRowKindInput = 0, kRowKindOutput = 2;
constexpr int32_t kCanvasPortOut = 0, kCanvasPortIn = 1;
constexpr int32_t kRowPortBase = 20000, kRowPortStride = 64;    // GP_CANVAS_ROW_PORT_CONTACT_BASE / _STRIDE

NodusHeadlessFifoConfig float_fifo() { return NodusHeadlessFifoConfig{4, 8, 0, -1, 2, 0, 0}; }   // one float, lossless

int32_t add_module(NodusHeadlessGraph* g, const char* label) {
    GP_CanvasModuleDesc d{};
    d.x = 40; d.y = 40; d.w = 160; d.h = 80;
    std::snprintf(d.label, sizeof(d.label), "%s", label);
    return nodus_headless_add_module(g, &d);
}

// A table of one Input row (contact 0) and one Output row (contact 1): what comes in goes out.
int32_t pass_table(NodusHeadlessGraph* g, const char* label) {
    const int32_t m = add_module(g, label);
    const GP_CanvasModuleRowDesc rows[2] = {{kRowKindInput, 0, 1, 0, nullptr}, {kRowKindOutput, 0, 1, 0, nullptr}};
    if (m < 0 || !nodus_headless_set_module_rows(g, m, rows, 2)) return -1;
    return m;
}

bool edge(NodusHeadlessGraph* g, int32_t a, int32_t ac, int32_t b, int32_t bc, int32_t slots = 8) {
    GP_CanvasEdgeDesc d{a, ac, b, bc};
    const int32_t e = nodus_headless_add_edge(g, &d, 0);
    NodusHeadlessFifoConfig fifo = float_fifo();
    fifo.slots = slots;
    return e >= 0 && nodus_headless_configure_edge_fifo(g, e, &fifo);
}

std::vector<uint8_t> read_file(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// The child: canvas port "in" -> pass table -> canvas port "out". Saved, its bytes are a canvas row's state. The pass
// table's exec mode is the child's own choice: -1 steps on the tick the canvas row gives it, 3 (Free) free-runs on its
// own worker; the choice is saved with the child (MODULE_POLICY).
std::vector<uint8_t> child_canvas_state(const std::filesystem::path& dir, int32_t exec_mode, int32_t slots = 8) {
    NodusHeadlessGraph* child = nodus_headless_create();
    CHECK(child != nullptr);
    if (!child) return {};
    const int32_t in = nodus_headless_declare_canvas_port(child, "in", kCanvasPortIn, 1);
    const int32_t out = nodus_headless_declare_canvas_port(child, "out", kCanvasPortOut, 1);
    const int32_t root = nodus_headless_canvas_root_module(child);
    const int32_t pass = pass_table(child, "pass");
    CHECK(in >= 0 && out >= 0 && root >= 0 && pass >= 0);
    CHECK(edge(child, root, in, pass, 0, slots));
    CHECK(edge(child, pass, 1, root, out, slots));
    CHECK(nodus_headless_set_module_exec_mode(child, pass, exec_mode));
    const auto path = dir / ("child_" + std::to_string(exec_mode + 1) + "_" + std::to_string(slots) + ".nodus.txt");
    CHECK(nodus_headless_save_file(child, path.string().c_str()));
    nodus_headless_destroy(child);
    std::vector<uint8_t> canvas = read_file(path);
    const std::string text(canvas.begin(), canvas.end());
    CHECK(text.find("MODULE_POLICY " + std::to_string(pass) + " " + std::to_string(exec_mode) + " ") != std::string::npos);
    // NCR1 v1, then the canvas file's byte count and bytes (canvas_row.cpp's saved state).
    std::vector<uint8_t> state;
    const uint32_t head[2] = {0x3152434E, 1};
    const uint64_t size = canvas.size();
    state.insert(state.end(), reinterpret_cast<const uint8_t*>(head), reinterpret_cast<const uint8_t*>(head) + 8);
    state.insert(state.end(), reinterpret_cast<const uint8_t*>(&size), reinterpret_cast<const uint8_t*>(&size) + 8);
    state.insert(state.end(), canvas.begin(), canvas.end());
    return state;
}

// Push 1..n into "feed" and read "result" at the top until every value arrived; an Input row with nothing on its edge
// outputs 0, so zeros are idle ticks, not values.
std::vector<float> carry(NodusHeadlessGraph* parent, int n) {
    std::vector<float> got;
    int next = 1;
    for (int tick = 0; tick < 2000 && static_cast<int>(got.size()) < n; ++tick) {
        if (next <= n) {
            const float v = static_cast<float>(next);
            NodusHeadlessPublishResult r{};
            if (nodus_headless_push_token_ex(parent, "feed", &v, sizeof(v), &r) && r.accepted) ++next;
        }
        nodus_headless_step(parent, 1.0 / 60.0);
        for (;;) {
            float v = 0.0f;
            size_t w = 0;
            if (!nodus_headless_canvas_port_consume(parent, "result", &v, sizeof(v), &w) || w != sizeof(v)) break;
            if (v != 0.0f) got.push_back(v);
        }
    }
    return got;
}

// A parent canvas holding that child in a canvas row: its feed table drives the row's "in", and the row's "out" ends at
// the parent's canvas port "result". The top level gives a tick to the holder table, the table to the canvas row, and
// the row to the canvas it holds.
void run_case(const char* tool_id, const std::filesystem::path& dir, int32_t exec_mode, const char* label) {
    const std::vector<uint8_t> state = child_canvas_state(dir, exec_mode);
    CHECK(!state.empty());

    // The parent: feed table -> canvas row "in"; canvas row "out" -> the parent's own canvas port "result".
    NodusHeadlessGraph* parent = nodus_headless_create();
    CHECK(parent != nullptr);
    if (!parent) return;
    const int32_t feed = pass_table(parent, "feed");
    const NodusHeadlessFifoConfig fifo = float_fifo();
    CHECK(nodus_headless_declare_input_port_ex(parent, "feed", feed, 0, &fifo));
    const int32_t holder = add_module(parent, "holder");
    const int32_t row = nodus_headless_bind_plugin_tool(parent, holder, tool_id);
    CHECK(row >= 0);
    CHECK(nodus_headless_set_row_state(parent, holder, row, state.data(), state.size()));
    const int32_t result = nodus_headless_declare_canvas_port(parent, "result", kCanvasPortOut, 1);
    const int32_t root = nodus_headless_canvas_root_module(parent);
    CHECK(result >= 0 && root >= 0);
    CHECK(edge(parent, feed, 1, holder, kRowPortBase + row * kRowPortStride + 0));        // row port 0: "in"
    CHECK(edge(parent, holder, kRowPortBase + row * kRowPortStride + 1, root, result));   // row port 1: "out"

    std::vector<float> wanted;
    for (int i = 1; i <= 12; ++i) wanted.push_back(static_cast<float>(i));
    const std::vector<float> got = carry(parent, 12);
    CHECK(got == wanted);
    std::printf("%s nested: %zu of %zu values arrived in order\n", label, got.size(), wanted.size());

    // Saved, the parent carries its canvas row's canvas; loaded into another graph, the row carries values again.
    const auto saved = dir / ("parent_" + std::to_string(exec_mode + 1) + ".nodus.txt");
    CHECK(nodus_headless_save_file(parent, saved.string().c_str()));
    const std::vector<uint8_t> text = read_file(saved);
    const std::string s(text.begin(), text.end());
    CHECK(s.find("ROW_STATE " + std::to_string(holder) + " " + std::to_string(row) + " ") != std::string::npos);
    nodus_headless_destroy(parent);

    NodusHeadlessGraph* again = nodus_headless_create();
    CHECK(again != nullptr);
    if (!again) return;
    CHECK(nodus_headless_load_file(again, saved.string().c_str()));
    size_t restored_size = 0;
    nodus_headless_get_row_state(again, holder, row, nullptr, 0, &restored_size);
    CHECK(restored_size > 16);
    const std::vector<float> got_again = carry(again, 12);
    CHECK(got_again == wanted);
    std::printf("%s reloaded: %zu of %zu values arrived in order\n", label, got_again.size(), wanted.size());

    // Saved twice to one path with only the nested canvas changed in between, the second save writes it: the save's
    // change signature counts row state (review 6e1b5069). The other exec mode's child is a different nested canvas.
    const auto twice = dir / ("twice_" + std::to_string(exec_mode + 1) + ".nodus.txt");
    CHECK(nodus_headless_save_file(again, twice.string().c_str()));
    const std::vector<uint8_t> before = read_file(twice);
    const std::vector<uint8_t> other = child_canvas_state(dir, exec_mode == -1 ? 3 : -1);
    CHECK(nodus_headless_set_row_state(again, holder, row, other.data(), other.size()));
    CHECK(nodus_headless_save_file(again, twice.string().c_str()));
    const std::vector<uint8_t> after = read_file(twice);
    CHECK(!before.empty() && after != before);
    std::printf("%s a second save after only the nested canvas changed: %s\n", label, after != before ? "written" : "SKIPPED");
    nodus_headless_destroy(again);

}

// Exactly once across a save (review f1a3eb8f). A canvas row takes a token from one side only when the other side has
// room, so a token is in its custody only when the far side then refuses: the canvas's IN edge filling, or a publish
// failing after the edge said it had room. Those moments are driven here directly, with the row's ports played by
// the test (as link_rows_test plays a link row's): the feed always offers the next value, and the output edge says it
// has room but refuses every publish. The row then holds a token on each side. It is saved, restored into a new row,
// and drained with the output accepting: each held value comes out exactly once, and nothing comes out twice. Values
// that were in the nested canvas's own FIFOs at the save are that canvas's, and a canvas save keeps no FIFO contents.
struct PlayedPorts {
    std::deque<float> feed;                 // what the Argument port "in" offers, in order
    std::vector<float> published;           // what the Return port "out" accepted
    bool refuse = true;                     // the output says it has room, then refuses
    static int32_t consume(void* h, int32_t port, void* out, size_t cap, size_t* written) {
        auto* p = static_cast<PlayedPorts*>(h);
        if (written) *written = 0;
        if (port != 0 || p->feed.empty() || cap < sizeof(float)) return 0;
        std::memcpy(out, &p->feed.front(), sizeof(float));
        p->feed.pop_front();
        if (written) *written = sizeof(float);
        return 1;
    }
    static int32_t publish(void* h, int32_t port, const void* bytes, size_t len) {
        auto* p = static_cast<PlayedPorts*>(h);
        if (port != 1 || len != sizeof(float) || p->refuse) return 0;
        float v = 0.0f;
        std::memcpy(&v, bytes, sizeof(v));
        p->published.push_back(v);
        return 1;
    }
    static int32_t can_publish(void*, int32_t port, int32_t) { return port == 1 ? 1 : 0; }
    ToolRowPortAccess access() {
        ToolRowPortAccess a{};
        a.host = this; a.consume = &consume; a.publish = &publish; a.can_publish = &can_publish;
        return a;
    }
};

void execute(nodus::canvas_row::CanvasRow& row, PlayedPorts& ports) {
    RawStackFrame frame{};
    raw_stack_init_frame(frame, 1 << 12);
    const ToolRowPortAccess access = ports.access();
    ToolStackContext c{};
    c.stack.raw = &frame;
    c.row_ports = &access;
    c.dt = 1.0 / 60.0;
    row.execute_stack(c);
    raw_stack_free_mask(frame);
}

void custody_case(const std::filesystem::path& dir) {
    const std::vector<uint8_t> child = child_canvas_state(dir, -1, 1);   // one-slot edges inside the canvas
    PlayedPorts ports;
    for (int v = 1; v <= 40; ++v) ports.feed.push_back(static_cast<float>(v));
    std::vector<uint8_t> saved;
    {
        nodus::canvas_row::CanvasRow row;
        MemoryInputArchive in(child);
        CHECK(row.deserialize(in));
        for (int i = 0; i < 6; ++i) execute(row, ports);              // the output refuses: tokens pile up in custody
        MemoryOutputArchive out;
        CHECK(row.serialize(out));
        saved = out.bytes;
    }                                                                 // the row is gone; only its saved state remains
    // Its held tokens, from the saved state (NCR1 v4): after the snapshot, per port a kind and the token.
    std::vector<float> held;
    {
        size_t at = 16 + 4;
        uint32_t n = 0, ws = 0, nports = 0;
        uint64_t snap = 0;
        std::memcpy(&n, saved.data() + at, 4); at += 4 + n * 32;
        std::memcpy(&ws, saved.data() + at, 4); at += 4 + ws + 8;
        std::memcpy(&snap, saved.data() + at, 8); at += 8 + static_cast<size_t>(snap);
        std::memcpy(&nports, saved.data() + at, 4); at += 4;
        for (uint32_t i = 0; i < nports; ++i) {
            const uint8_t kind = saved[at++];
            float v = 0.0f;
            if (kind == 1) {
                uint32_t len = 0;
                std::memcpy(&len, saved.data() + at, 4); at += 4;
                if (len == sizeof(v)) std::memcpy(&v, saved.data() + at, sizeof(v));
                at += len;
            }
            held.push_back(v);
        }
    }
    std::printf("custody: held at the save: in %g, out %g (nothing published while the output refused: %zu)\n",
                held.size() > 0 ? held[0] : -1.0, held.size() > 1 ? held[1] : -1.0, ports.published.size());
    CHECK(held.size() == 2 && held[0] != 0.0f && held[1] != 0.0f && ports.published.empty());
    {
        nodus::canvas_row::CanvasRow row;
        MemoryInputArchive in(saved);
        CHECK(row.deserialize(in));
        ports.refuse = false;
        for (int i = 0; i < 120; ++i) execute(row, ports);
    }
    std::vector<float> got;
    for (float v : ports.published) if (v != 0.0f) got.push_back(v);
    auto count = [&](float v) { return std::count(got.begin(), got.end(), v); };
    bool unique = true;
    for (float v : got) unique = unique && count(v) == 1;
    std::printf("custody: after the restore %zu values came out; the held ones %ld and %ld times\n", got.size(),
                held.size() > 0 ? static_cast<long>(count(held[0])) : -1L, held.size() > 1 ? static_cast<long>(count(held[1])) : -1L);
    CHECK(held.size() == 2 && count(held[0]) == 1 && count(held[1]) == 1);
    CHECK(unique);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: canvas_row_test <work dir>\n"); return 2; }
    const auto dir = std::filesystem::path(argv[1]) / ("canvas_row_test_" + std::to_string(::time(nullptr)));
    std::filesystem::create_directories(dir);
    const std::string row_id = headless_rows::build_named("canvas_row", dir / "module_library");
    if (row_id.empty()) return 2;
    const char* tool_id = row_id.c_str();
    run_case(tool_id, dir, -1, "respects the tick:");
    run_case(tool_id, dir, 3, "free-runs:");
    custody_case(dir);

    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
