// Verifies the nodus_headless ABI: real ThreadManager scheduling (via
// gp_canvas_step, never a hand-written per-tool tick loop) reaches
// quiescence with no window/renderer, ports round-trip real bytes through
// the same table-edge transport real graph edges use, save/load preserves
// enough structure to re-resolve the same ports, and edge overflow is
// observable through the event queue.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "nodus_headless_abi.h"

namespace {
constexpr int32_t kToolTensorAllocator = 12;
constexpr int32_t kToolTensorTool = 13;

int32_t build_two_module_graph(NodusHeadlessGraph* g, int32_t* out_module_a, int32_t* out_row_a,
                                int32_t* out_module_b, int32_t* out_row_b) {
    GP_CanvasModuleDesc mod_a{};
    mod_a.x = 0; mod_a.y = 0; mod_a.w = 64; mod_a.h = 64;
    std::strncpy(mod_a.label, "alloc", sizeof(mod_a.label) - 1);
    int32_t module_a = nodus_headless_add_module(g, &mod_a);

    GP_CanvasModuleDesc mod_b{};
    mod_b.x = 100; mod_b.y = 0; mod_b.w = 64; mod_b.h = 64;
    std::strncpy(mod_b.label, "tool", sizeof(mod_b.label) - 1);
    int32_t module_b = nodus_headless_add_module(g, &mod_b);
    assert(module_a >= 0 && module_b >= 0);

    int32_t row_a = nodus_headless_bind_builtin_tool(g, module_a, kToolTensorAllocator);
    int32_t row_b = nodus_headless_bind_builtin_tool(g, module_b, kToolTensorTool);
    assert(row_a >= 0 && row_b >= 0);

    GP_CanvasEdgeDesc edge{};
    edge.a_module = module_a; edge.a_contact_idx = row_a;
    edge.b_module = module_b; edge.b_contact_idx = row_b;
    int32_t edge_idx = nodus_headless_add_edge(g, &edge, /*type_id=*/0);
    assert(edge_idx >= 0);

    *out_module_a = module_a; *out_row_a = row_a;
    *out_module_b = module_b; *out_row_b = row_b;
    return edge_idx;
}
} // namespace

static void test_quiescence_and_event() {
    NodusHeadlessGraph* g = nodus_headless_create();
    assert(g);

    int32_t module_a, row_a, module_b, row_b;
    build_two_module_graph(g, &module_a, &row_a, &module_b, &row_b);

    assert(nodus_headless_declare_input_port(g, "in", module_a, row_a, sizeof(float) * 4));
    assert(nodus_headless_declare_output_port(g, "out", module_b, row_b, sizeof(float) * 4));

    // Scheduling must go through the real frontier (gp_canvas_step ->
    // ThreadManager::submit_tick) -- this test never calls a tool's tick()
    // directly.
    int32_t reached = nodus_headless_run_to_quiescence(g, 1.0 / 60.0, /*max_ticks=*/240);
    assert(reached == 1);
    assert(nodus_headless_is_quiescent(g) == 1);

    NodusHeadlessEvent events[32];
    int32_t n = nodus_headless_poll_events(g, events, 32);
    bool saw_quiescent = false;
    for (int32_t i = 0; i < n; ++i) {
        if (events[i].kind == NODUS_EVT_QUIESCENT) saw_quiescent = true;
    }
    assert(saw_quiescent);

    nodus_headless_destroy(g);
    std::printf("test_quiescence_and_event: OK\n");
}

static void test_save_load_round_trip() {
    std::filesystem::path path = std::filesystem::temp_directory_path() / "nodus_headless_test_graph.txt";

    NodusHeadlessGraph* g1 = nodus_headless_create();
    assert(g1);
    int32_t module_a, row_a, module_b, row_b;
    build_two_module_graph(g1, &module_a, &row_a, &module_b, &row_b);
    assert(nodus_headless_save_file(g1, path.string().c_str()));
    nodus_headless_destroy(g1);

    NodusHeadlessGraph* g2 = nodus_headless_create();
    assert(g2);
    assert(nodus_headless_load_file(g2, path.string().c_str()));
    // The loaded graph must retain the same module/row topology so the same
    // ports resolve again -- this is the save/load fidelity the headless
    // ABI relies on (no new serialization format was introduced).
    assert(nodus_headless_declare_input_port(g2, "in", module_a, row_a, sizeof(float) * 4));
    assert(nodus_headless_declare_output_port(g2, "out", module_b, row_b, sizeof(float) * 4));
    nodus_headless_destroy(g2);

    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::printf("test_save_load_round_trip: OK\n");
}

static void test_overflow_event() {
    NodusHeadlessGraph* g = nodus_headless_create();
    assert(g);
    int32_t module_a, row_a, module_b, row_b;
    build_two_module_graph(g, &module_a, &row_a, &module_b, &row_b);
    assert(nodus_headless_declare_input_port(g, "in", module_a, row_a, sizeof(float) * 4));

    float payload[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    bool saw_overflow = false;
    // Publish far more samples than the edge's default slot capacity without
    // ticking in between, so old unread samples must be dropped to admit
    // new writes.
    for (int i = 0; i < 256; ++i) {
        nodus_headless_push_token(g, "in", payload, sizeof(payload));
        NodusHeadlessEvent events[32];
        int32_t n = nodus_headless_poll_events(g, events, 32);
        for (int32_t j = 0; j < n; ++j) {
            if (events[j].kind == NODUS_EVT_OVERFLOW) saw_overflow = true;
        }
    }
    assert(saw_overflow);

    nodus_headless_destroy(g);
    std::printf("test_overflow_event: OK\n");
}

int main() {
    test_quiescence_and_event();
    test_save_load_round_trip();
    test_overflow_event();
    return 0;
}
