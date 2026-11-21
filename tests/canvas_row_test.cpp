// canvas_row_test: a canvas nested in a table row (module_library/source/tools/canvas_row.cpp,
// docs/MESH_CANVAS_UP_PORTS_2026-11-21.md stage 3).
//
// A child canvas passes values from its IN canvas port "in" through one table to its OUT canvas port "out". A parent
// canvas holds it in a canvas row: the parent's feed table drives the row's Argument port "in", and the row's Return
// port "out" ends at the parent's own OUT canvas port "result". Two canvases coexist in one process, each on its own
// scheduler; every value arrives at the top, in order. The parent is saved (the row's state, the child canvas, rides in
// a ROW_STATE record) and loaded into a third graph, whose restored canvas row carries values again.
//
//   canvas_row_test <path to the canvas_row plugin library>
#include "canvas_abi.h"
#include "nodus_headless_abi.h"
#include "plugin_manager.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
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

bool edge(NodusHeadlessGraph* g, int32_t a, int32_t ac, int32_t b, int32_t bc) {
    GP_CanvasEdgeDesc d{a, ac, b, bc};
    const int32_t e = nodus_headless_add_edge(g, &d, 0);
    const NodusHeadlessFifoConfig fifo = float_fifo();
    return e >= 0 && nodus_headless_configure_edge_fifo(g, e, &fifo);
}

std::vector<uint8_t> read_file(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// The child: canvas port "in" -> pass table -> canvas port "out". Saved, its bytes are a canvas row's state.
std::vector<uint8_t> child_canvas_state(const std::filesystem::path& dir) {
    NodusHeadlessGraph* child = nodus_headless_create();
    CHECK(child != nullptr);
    if (!child) return {};
    const int32_t in = nodus_headless_declare_canvas_port(child, "in", kCanvasPortIn, 1);
    const int32_t out = nodus_headless_declare_canvas_port(child, "out", kCanvasPortOut, 1);
    const int32_t root = nodus_headless_canvas_root_module(child);
    const int32_t pass = pass_table(child, "pass");
    CHECK(in >= 0 && out >= 0 && root >= 0 && pass >= 0);
    CHECK(edge(child, root, in, pass, 0));
    CHECK(edge(child, pass, 1, root, out));
    const auto path = dir / "child.nodus.txt";
    CHECK(nodus_headless_save_file(child, path.string().c_str()));
    nodus_headless_destroy(child);
    std::vector<uint8_t> canvas = read_file(path);
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: canvas_row_test <canvas_row plugin library>\n"); return 2; }
    char tool_id[256] = {};
    if (!gp_plugin_load_from_path(argv[1], tool_id, sizeof(tool_id))) {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        return 2;
    }
    const auto dir = std::filesystem::temp_directory_path() / ("canvas_row_test_" + std::to_string(::time(nullptr)));
    std::filesystem::create_directories(dir);
    const std::vector<uint8_t> state = child_canvas_state(dir);
    CHECK(!state.empty());

    // The parent: feed table -> canvas row "in"; canvas row "out" -> the parent's own canvas port "result".
    NodusHeadlessGraph* parent = nodus_headless_create();
    CHECK(parent != nullptr);
    if (!parent) return 1;
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
    std::printf("nested: %zu of %zu values arrived in order\n", got.size(), wanted.size());

    // Saved, the parent carries its canvas row's canvas; loaded into another graph, the row carries values again.
    const auto saved = dir / "parent.nodus.txt";
    CHECK(nodus_headless_save_file(parent, saved.string().c_str()));
    const std::vector<uint8_t> text = read_file(saved);
    const std::string s(text.begin(), text.end());
    CHECK(s.find("ROW_STATE " + std::to_string(holder) + " " + std::to_string(row) + " ") != std::string::npos);
    nodus_headless_destroy(parent);

    NodusHeadlessGraph* again = nodus_headless_create();
    CHECK(again != nullptr);
    if (!again) return 1;
    CHECK(nodus_headless_load_file(again, saved.string().c_str()));
    size_t restored_size = 0;
    nodus_headless_get_row_state(again, holder, row, nullptr, 0, &restored_size);
    CHECK(restored_size > 16);
    const std::vector<float> got_again = carry(again, 12);
    CHECK(got_again == wanted);
    std::printf("reloaded: %zu of %zu values arrived in order\n", got_again.size(), wanted.size());
    nodus_headless_destroy(again);

    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
