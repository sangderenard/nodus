// fabric_state_test: the fabric row on the tensor state library against the fabric row as it is (research case
// 2a1563063ca34358899d5cb971cdf521). Both rows get the same configuration and the same sequence of ticks and commands;
// every output must be byte-identical.
//   fabric_state_test <reference spring_network_nodus_row.dll> <spring_network_step_batched__tick.dll>
//                     <spring_network_step.direct-glsl.spv> cpu|vulkan.<device> [ticks]
// The reference row is the plugin turing tools/build_spring_nodus_row.py builds; NODUS_FABRIC_BACKEND=vulkan is set
// for it in GPU mode, as the row reads it.
#include "kernel_library_session.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <windows.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace nodus::gpu;
using Clock = std::chrono::steady_clock;

namespace {
constexpr int Nets = kFabricNetworks, Nodes = kFabricNodes, Fields = kFabricFields, Side = 16;

bool invoke(ITool& tool, uint64_t input, std::vector<double>& out, double& ms) {
    RawStackFrame frame{};
    if (!raw_stack_init_frame(frame, 4096)) return false;
    bool ok = raw_stack_push_abstract_tensor(frame, {input});
    ToolStackContext c{}; c.stack.raw = &frame;
    const auto t0 = Clock::now();
    if (ok) tool.execute_stack(c);
    ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    nodus::tensors::AbstractTensorHandle r{};
    ok = ok && c.succeeded && raw_stack_pop_abstract_tensor(frame, r);
    if (ok) { out.resize(std::size_t(kFabricBatch)); ok = nodus_tensor_read(r.id, 0, out.data(), out.size() * 8) == int64_t(out.size() * 8); }
    if (r.id) nodus_tensor_destroy(r.id);
    raw_stack_free_mask(frame); if (frame.backend) gp_mem_backend_release(frame.backend);
    return ok;
}
uint64_t f64(std::vector<uint64_t> shape, const std::vector<double>& v) {
    uint64_t t = nodus_tensor_create(NODUS_DTYPE_F64, uint32_t(shape.size()), shape.empty() ? nullptr : shape.data());
    if (t && !v.empty()) nodus_tensor_write(t, 0, v.data(), v.size() * 8);
    return t;
}
// Eight 16 x 16 sheets, left/right/up/down springs with rest offsets, two pinned top corners, gravity, drag.
std::vector<double> sheets() {
    std::vector<double> d(std::size_t(Nets) * Nodes * Fields, 0.0);
    for (int net = 0; net < Nets; net++) for (int r = 0; r < Side; r++) for (int c = 0; c < Side; c++) {
        const int node = r * Side + c; double* row = &d[(std::size_t(net) * Nodes + node) * Fields];
        const double x = c * 0.05 + net * 1.0, y = 1.5 - r * 0.05, z = 0.01 * net;
        row[0] = x; row[1] = y; row[2] = z;
        row[6] = 1; row[7] = (r == 0 && (c == 0 || c == Side - 1)) ? 1 : 0;
        row[8] = x; row[9] = y; row[10] = z;
        row[11] = 1.0 / (0.2 * (1 + net % 3));   // inverse mass; stable limit 1/sqrt(2*4*40*5) = 0.025 s > 1/60 row[12] = 40; row[13] = 0.08 + 0.01 * net;
        row[15] = -9.8;
        const int nb[4][2] = {{r, c - 1}, {r, c + 1}, {r - 1, c}, {r + 1, c}};
        for (int e = 0; e < 4; e++) {
            const int rr = nb[e][0], cc = nb[e][1];
            if (rr < 0 || rr >= Side || cc < 0 || cc >= Side) continue;
            row[17 + e] = rr * Side + cc; row[33 + e] = 1;
            row[21 + e * 3 + 0] = (cc - c) * 0.05; row[21 + e * 3 + 1] = -(rr - r) * 0.05; row[21 + e * 3 + 2] = 0;
        }
        if (node == 0) row[37] = 100 + net;
    }
    return d;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) { std::printf("usage: fabric_state_test <reference row dll> <llvm dll> <spirv> cpu|vulkan.<device> [ticks]\n"); return 2; }
    const std::string backend = argv[4];
    const int ticks = argc > 5 ? std::atoi(argv[5]) : 120;
    const bool gpu = backend.rfind("vulkan.", 0) == 0;
    _putenv_s("NODUS_FABRIC_UDP_HOST", "127.0.0.1"); _putenv_s("NODUS_FABRIC_UDP_PORT", "48771");
    if (gpu) _putenv_s("NODUS_FABRIC_BACKEND", "vulkan");

    HMODULE plugin = LoadLibraryA(argv[1]);
    auto create = plugin ? reinterpret_cast<ITool* (*)()>(GetProcAddress(plugin, "create_tool")) : nullptr;
    auto destroy = plugin ? reinterpret_cast<void (*)(ITool*)>(GetProcAddress(plugin, "destroy_tool")) : nullptr;
    if (!create || !destroy) { std::printf("reference row did not load\n"); return 2; }
    ITool* reference = create(); reference->initialize({});

    std::ifstream f(argv[3], std::ios::binary); std::vector<char> b{std::istreambuf_iterator<char>(f), {}};
    FabricSpringProducts products; products.llvm_dll = argv[2]; products.vulkan_spirv.resize(b.size() / 4);
    std::memcpy(products.vulkan_spirv.data(), b.data(), b.size());
    int failures = 0, compared = 0;
    {
        KernelLibrarySession session(tool_registry_global());
        if (!session) { std::printf("library refused\n"); return 2; }
        if (!session.register_fabric_spring(products, "spring.network.library", backend, 31, 4000)) { std::printf("fabric registration refused\n"); return 2; }
        auto row = tool_registry_global().create("spring.network.library");
        row->initialize({});
        std::vector<double> want, got; double ms_ref = 0, ms_lib = 0, sum_ref = 0, sum_lib = 0;
        auto both = [&](uint64_t input, const char* what, int index) {
            const bool a = invoke(*reference, input, want, ms_ref), c = invoke(*row, input, got, ms_lib);
            ++compared;
            if (a != c || (a && std::memcmp(want.data(), got.data(), want.size() * 8) != 0)) {
                ++failures;
                size_t first = 0; while (a && c && first < want.size() && want[first] == got[first]) ++first;
                std::printf("DIFFER at %s %d (reference ok=%d, library ok=%d", what, index, int(a), int(c));
                if (a && c && first < want.size()) std::printf(", lane %zu: %.17g vs %.17g", first, want[first], got[first]);
                std::printf(")\n");
            }
            if (std::strcmp(what, "tick") == 0) { sum_ref += ms_ref; sum_lib += ms_lib; }
            return a && c;
        };
        const uint64_t config = f64({uint64_t(Nets), uint64_t(Nodes), uint64_t(Fields)}, sheets());
        const uint64_t dt = f64({}, {1.0 / 60.0});
        auto command = [&](std::vector<double> v) { return f64({v.size()}, v); };
        both(config, "configure", 0);
        for (int t = 0; t < ticks; t++) {
            if (t == 10) { auto c = command({3, 3}); both(c, "substeps 3", t); nodus_tensor_destroy(c); }
            if (t == 30) { auto c = command({5, 0.2}); both(c, "floor y=0.2", t); nodus_tensor_destroy(c); }
            if (t == 50) { auto c = command({2, 0.9, 0.0, 3.0, -1.0, 1.0}); both(c, "tabletop", t); nodus_tensor_destroy(c); }
            if (t == 70) { auto c = command({1, 1}); both(c, "freeze", t); nodus_tensor_destroy(c); }
            if (t == 75) { auto c = command({1, 0}); both(c, "thaw", t); nodus_tensor_destroy(c); }
            both(dt, "tick", t);
        }
        double low = 1e9; for (double v : got) low = std::min(low, v);
        std::printf("%s: %d outputs compared, %d differ; last tick lowest coordinate %.6f; mean tick %.3f ms (row as it is) vs %.3f ms (state library)\n",
                    backend.c_str(), compared, failures, low, sum_ref / ticks, sum_lib / ticks);
        nodus_tensor_destroy(config); nodus_tensor_destroy(dt);
        row.reset();
    }
    reference->shutdown(); destroy(reference);
    std::printf("%s\n", failures ? "FAILED" : "PASSED: byte-identical to the fabric row as it is");
    return failures ? 1 : 0;
}
