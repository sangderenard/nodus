// Builds a minimal two-module graph and runs it to quiescence with no
// window/renderer, proving the nodus_headless ABI end-to-end from a plain
// console process. Uses existing built-in tool kinds (TensorAllocator,
// TensorTool) -- no new tool code involved.
#include "nodus_headless_abi.h"

#include <cstdio>
#include <cstring>

// ModuleToolKind values (thread_manager.h). The demo only needs the plain
// integers, matching what a ctypes caller would pass.
namespace {
constexpr int32_t kToolTensorAllocator = 12;
constexpr int32_t kToolTensorTool = 13;
} // namespace

int main() {
    NodusHeadlessGraph* g = nodus_headless_create();
    if (!g) {
        std::fprintf(stderr, "nodus_headless_create failed\n");
        return 1;
    }

    GP_CanvasModuleDesc mod_a{};
    mod_a.x = 0; mod_a.y = 0; mod_a.w = 64; mod_a.h = 64;
    std::strncpy(mod_a.label, "alloc", sizeof(mod_a.label) - 1);
    int32_t module_a = nodus_headless_add_module(g, &mod_a);

    GP_CanvasModuleDesc mod_b{};
    mod_b.x = 100; mod_b.y = 0; mod_b.w = 64; mod_b.h = 64;
    std::strncpy(mod_b.label, "tool", sizeof(mod_b.label) - 1);
    int32_t module_b = nodus_headless_add_module(g, &mod_b);

    if (module_a < 0 || module_b < 0) {
        std::fprintf(stderr, "add_module failed\n");
        nodus_headless_destroy(g);
        return 1;
    }

    int32_t row_a = nodus_headless_bind_builtin_tool(g, module_a, kToolTensorAllocator);
    int32_t row_b = nodus_headless_bind_builtin_tool(g, module_b, kToolTensorTool);
    if (row_a < 0 || row_b < 0) {
        std::fprintf(stderr, "bind_builtin_tool failed\n");
        nodus_headless_destroy(g);
        return 1;
    }

    GP_CanvasEdgeDesc edge{};
    edge.a_module = module_a; edge.a_contact_idx = row_a;
    edge.b_module = module_b; edge.b_contact_idx = row_b;
    int32_t edge_idx = nodus_headless_add_edge(g, &edge, /*type_id=*/0);
    if (edge_idx < 0) {
        std::fprintf(stderr, "add_edge failed\n");
        nodus_headless_destroy(g);
        return 1;
    }

    if (!nodus_headless_declare_input_port(g, "in", module_a, row_a, sizeof(float) * 4) ||
        !nodus_headless_declare_output_port(g, "out", module_b, row_b, sizeof(float) * 4)) {
        std::fprintf(stderr, "declare port failed\n");
        nodus_headless_destroy(g);
        return 1;
    }

    float payload[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    nodus_headless_push_token(g, "in", payload, sizeof(payload));

    int32_t reached = nodus_headless_run_to_quiescence(g, 1.0 / 60.0, /*max_ticks=*/240);
    std::printf("run_to_quiescence: %s\n", reached ? "quiescent" : "max_ticks reached");

    float out_buf[4] = {};
    size_t written = 0;
    if (nodus_headless_pull_token(g, "out", out_buf, sizeof(out_buf), &written)) {
        std::printf("pulled %zu bytes from 'out'\n", written);
    } else {
        std::printf("nothing pending on 'out' (expected -- TensorTool's own tick decides what it forwards)\n");
    }

    NodusHeadlessEvent events[16];
    int32_t n = nodus_headless_poll_events(g, events, 16);
    std::printf("polled %d event(s)\n", n);
    for (int32_t i = 0; i < n; ++i) {
        std::printf("  kind=%d module=%d edge=%d tick=%llu\n",
            events[i].kind, events[i].module_idx, events[i].edge_idx,
            static_cast<unsigned long long>(events[i].tick_id));
    }

    nodus_headless_destroy(g);
    return 0;
}
