// The volunteer cycle end to end (case a11b5c9b1be74c9b9fb684790f7fd5a3): one cycle, two tables, and the mint.
//
//   tensor mint:        Input(6) | tensor_mint (U64[8,10] batches)
//   volunteer library:  Input(2) | nodus.volunteer.library | Output(1)   (Input 0: queued work; Input 1: from work)
//   volunteer work:     Input(1) | one work row per GPU | Output(1)
//   edges:   library Output -> work Input, work Output -> library Input 1 (lossless tensor FIFOs);
//            port pairs: mint blank0 > library blank, library spent > mint returned.
//
// The job is a GLSL compute shader (compiled here with glslangValidator) that adds 1 to every u32 of its state. K
// packets each name their own state address; half leave the target to the library (rule 1), half set it ahead of
// time. Passes when every packet completes through receipts (rule 2) and every state holds exactly 1 everywhere.
//
// Usage: test_volunteer_cycle <glslangValidator> <nodus root> <module library root> <work dir>
#include "kernel_library_session.h"
#include "nodus_headless_abi.h"
#include "value_types.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr std::uint64_t kCard = 7, kProcess = 1, kBase = 100;
constexpr std::uint32_t kWords = 64;
constexpr int kPackets = 16;

std::uint64_t u64_tensor(const std::vector<std::uint64_t>& words) {
    const std::uint64_t shape[1] = {words.size()};
    const auto h = nodus_tensor_create(NODUS_DTYPE_U64, 1, shape);
    nodus_tensor_write(h, 0, words.data(), words.size() * sizeof(std::uint64_t));
    return h;
}

// One ordinary library request through the session's row: control U64[12] under its payload; returns the status.
std::uint64_t request(ITool& library, const std::array<std::uint64_t, 12>& control, std::uint64_t payload) {
    RawStackFrame frame;
    raw_stack_init_frame(frame, 4096);
    const auto c = u64_tensor(std::vector<std::uint64_t>(control.begin(), control.end()));
    raw_stack_push_abstract_tensor(frame, {c});
    raw_stack_push_abstract_tensor(frame, {payload});
    ToolStackContext context;
    context.stack.raw = &frame;
    library.execute_stack(context);
    std::uint64_t status = 99;
    nodus::tensors::AbstractTensorHandle receipt{};
    if (context.succeeded && raw_stack_pop_abstract_tensor(frame, receipt)) {
        nodus_tensor_read(receipt.id, 0, &status, sizeof(status));
        nodus_tensor_destroy(receipt.id);
    }
    nodus_tensor_destroy(c);
    raw_stack_free_mask(frame);
    if (frame.backend) gp_mem_backend_release(frame.backend);
    return status;
}

int fail(const char* what) {
    std::printf("FAIL %s\n", what);
    return 1;
}
}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 5) return fail("usage: test_volunteer_cycle <glslangValidator> <nodus root> <module library> <work dir>");
    const std::filesystem::path glslang = argv[1], nodus_root = argv[2], library_root = argv[3], work = argv[4];
    std::filesystem::create_directories(work);

    // The job: GLSL, built for this machine's Vulkan by glslang.
    const auto comp = work / "increment.comp", spv = work / "increment.spv";
    std::ofstream(comp) << "#version 450\nlayout(local_size_x = 64) in;\n"
                           "layout(std430, binding = 0) buffer State { uint v[]; };\n"
                           "void main() { uint i = gl_GlobalInvocationID.x; if (i < v.length()) v[i] += 1u; }\n";
    const std::string command = "\"\"" + glslang.string() + "\" -V --target-env vulkan1.1 -o \"" + spv.string() +
                                "\" \"" + comp.string() + "\" > NUL\"";
    if (std::system(command.c_str()) != 0) return fail("glslangValidator");
    std::ifstream in(spv, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::uint32_t> spirv(bytes.size() / 4);
    std::memcpy(spirv.data(), bytes.data(), spirv.size() * 4);

    nodus::gpu::KernelLibrarySession session(tool_registry_global());
    if (!session) return fail("no state library");
    const auto devices = session.library().device_count();
    std::printf("devices: %zu\n", devices);
    if (!devices) return fail("no Vulkan device");
    nodus::gpu::GpuShaderJobContract contract;
    contract.id = "volunteer.increment";
    contract.program_id = 1;
    contract.spirv = spirv;
    contract.minimum_binding_bytes = {kWords * 4};
    contract.writable = {1};
    contract.local_size_x = 64;
    contract.maximum_invocations = kWords;
    if (!session.register_shader(contract)) return fail("register_shader");
    if (!session.register_volunteer_rows(contract.id, 4)) return fail("register_volunteer_rows");

    // The library card and one u32[64] state per packet.
    ITool& library = session.library_row();
    std::array<std::uint64_t, 12> card{1, kCard, kProcess, 0, 1ull << 20};
    for (std::size_t d = 0; d < devices && d < 3; ++d) card[5 + d] = 1ull << 20;
    const auto dummy = u64_tensor({0});
    if (request(library, card, dummy) != 0) return fail("Card");
    for (int k = 0; k < kPackets; ++k) {
        const std::uint64_t shape[1] = {kWords};
        const auto state = nodus_tensor_create(NODUS_DTYPE_U32, 1, shape);
        nodus_tensor_zero(state, 0, 0);
        if (request(library, {2, kCard, kProcess, kBase + static_cast<std::uint64_t>(k)}, state) != 0) return fail("Put");
        nodus_tensor_destroy(state);
    }
    nodus_tensor_destroy(dummy);

    // The three tables.
    auto* graph = nodus_headless_create();
    if (!graph) return fail("nodus_headless_create");
    if (!nodus_headless_set_scheduler_mode(graph, 1)) return fail("scheduler");  // SCHEDULED, as nodus_graph.Graph
    auto add = [&](const char* label, int x) {
        GP_CanvasModuleDesc desc{};
        desc.x = x; desc.y = 0; desc.w = 420; desc.h = 320;
        std::snprintf(desc.label, sizeof(desc.label), "%s", label);
        return nodus_headless_add_module(graph, &desc);
    };
    char mint_id[1024] = {};
    if (!nodus_headless_build_plugin_source((library_root / "source" / "tools" / "tensor_mint.cpp").string().c_str(),
                                            nodus_root.string().c_str(), library_root.string().c_str(), mint_id,
                                            sizeof(mint_id)))
        return fail("build tensor_mint");
    const int mint = add("tensor mint", 0), lib = add("volunteer library", 480), wrk = add("volunteer work", 960);
    {
        GP_CanvasModuleRowDesc rows[] = {{0, 0, 6, 0, nullptr}, {1, 0, 0, 1, mint_id}};
        if (!nodus_headless_set_module_rows(graph, mint, rows, 2)) return fail("mint rows");
    }
    {
        GP_CanvasModuleRowDesc rows[] = {{0, 0, 2, 0, nullptr}, {1, 0, 0, 1, "nodus.volunteer.library"}, {2, 0, 1, 0, nullptr}};
        if (!nodus_headless_set_module_rows(graph, lib, rows, 3)) return fail("library rows");
    }
    std::vector<std::string> work_ids;
    for (std::size_t d = 0; d < devices; ++d)
        work_ids.push_back("nodus.volunteer.work.volunteer.increment.vulkan." + std::to_string(d));
    {
        std::vector<GP_CanvasModuleRowDesc> rows{{0, 0, 1, 0, nullptr}};
        for (const auto& id : work_ids) rows.push_back({1, 0, 0, 1, id.c_str()});
        rows.push_back({2, 0, 1, 0, nullptr});
        if (!nodus_headless_set_module_rows(graph, wrk, rows.data(), static_cast<int32_t>(rows.size())))
            return fail("work rows");
    }
    for (const int m : {mint, lib, wrk})
        if (!nodus_headless_set_module_stack_contract(graph, m, 1)) return fail("stack contract");

    const int tensor_type = nodus_headless_abstract_tensor_type_id();
    NodusHeadlessFifoConfig lossless{8, 4, 0, tensor_type, 2, 16, 0};
    auto edge = [&](int a, int ac, int b, int bc) {
        GP_CanvasEdgeDesc e{};
        e.a_module = a; e.a_contact_idx = ac; e.b_module = b; e.b_contact_idx = bc;
        const int idx = nodus_headless_add_edge(graph, &e, 0);  // as nodus_graph connect: the FIFO carries the type
        return idx >= 0 && nodus_headless_configure_edge_fifo(graph, idx, &lossless);
    };
    // library: inputs 0,1, output 2. work: input 0, output 1.
    if (!edge(lib, 2, wrk, 0) || !edge(wrk, 1, lib, 1)) return fail("cycle edges");
    char error[512] = {};
    for (const char* pair : {"tensor mint.1:blank0 > volunteer library.1:blank",
                             "volunteer library.1:spent > tensor mint.1:returned"}) {
        const int idx = nodus_headless_apply_port_pair(graph, pair, error, sizeof(error));
        if (idx < 0 || !nodus_headless_configure_edge_fifo(graph, idx, &lossless)) {
            std::printf("port pair %s: %s\n", pair, error);
            return fail("port pair");
        }
    }
    NodusHeadlessFifoConfig scalar{4, 4, 1, -1, 2, 0, 0};
    const char* params[6] = {"dtype", "d0", "d1", "d2", "d3", "depth"};
    for (int k = 0; k < 6; ++k)
        if (!nodus_headless_declare_input_port_ex(graph, params[k], mint, k, &scalar)) return fail("mint ports");
    if (!nodus_headless_declare_input_port_ex(graph, "work", lib, 0, &lossless)) return fail("work port");
    const float mint_settings[6] = {14.0f, 8.0f, 10.0f, 0.0f, 0.0f, 4.0f};  // U64[8,10], keep 4
    for (int k = 0; k < 6; ++k) nodus_headless_push_token(graph, params[k], &mint_settings[k], sizeof(float));

    // The work: half the targets left to the library, half set ahead of time (device + 1).
    int pushed = 0;
    const auto started = std::chrono::steady_clock::now();
    nodus::gpu::TensorStateLibraryTool::CycleCounts counts{};
    for (int tick = 0; tick < 20000; ++tick) {
        while (pushed < kPackets) {
            const std::uint64_t target = (pushed % 2) ? 1 + static_cast<std::uint64_t>(pushed / 2) % devices : 0;
            const auto packet = u64_tensor({kCard, kBase + static_cast<std::uint64_t>(pushed), kWords,
                                            1000 + static_cast<std::uint64_t>(pushed), 0, target});
            NodusHeadlessPublishResult r{};
            if (!nodus_headless_push_token_ex(graph, "work", &packet, sizeof(packet), &r) || !r.accepted) {
                nodus_tensor_destroy(packet);
                break;
            }
            ++pushed;
        }
        if (!nodus_headless_step(graph, 1.0 / 60.0)) return fail("step");
        counts = session.library().cycle_counts();
        if (pushed == kPackets && counts.completed == kPackets && !counts.outstanding && !counts.queued) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::printf("pushed %d completed %llu retried %llu outstanding %llu queued %llu batches in hand %llu (%.3f s)\n",
                pushed, (unsigned long long)counts.completed, (unsigned long long)counts.retried,
                (unsigned long long)counts.outstanding, (unsigned long long)counts.queued,
                (unsigned long long)counts.batches_in_hand, seconds);
    for (std::size_t d = 0; d < devices && d < counts.completed_by_device.size(); ++d) {
        const auto perf = session.library().work_performance(nodus::gpu::TensorStateLibraryTool::ExecutionApi::Vulkan,
                                                             d, contract.program_id, kWords);
        std::printf("device %zu: %llu completed, measured mean %.1f us over %llu\n", d,
                    (unsigned long long)counts.completed_by_device[d], perf ? perf->average_total_ns / 1000.0 : 0.0,
                    perf ? (unsigned long long)perf->samples : 0ull);
    }
    for (int e = 0; e < nodus_headless_get_edge_count(graph); ++e) {
        GP_CanvasEdgeDesc d{};
        int32_t type = 0, unread = -1;
        nodus_headless_get_edge(graph, e, &d, &type);
        nodus_headless_get_edge_unread(graph, e, &unread);
        std::printf("edge %d: %d.%d -> %d.%d unread %d\n", e, d.a_module, d.a_contact_idx, d.b_module, d.b_contact_idx, unread);
    }
    nodus_headless_destroy(graph);
    if (counts.completed != kPackets) return fail("not every packet completed");

    // Every state was incremented exactly once.
    for (int k = 0; k < kPackets; ++k) {
        auto loan = session.library().borrow_cpu(kCard, kBase + static_cast<std::uint64_t>(k), false);
        if (!loan || loan.bytes() < kWords * 4) return fail("borrow_cpu");
        const auto* v = static_cast<const std::uint32_t*>(loan.data());
        for (std::uint32_t i = 0; i < kWords; ++i)
            if (v[i] != 1) {
                std::printf("state %d word %u = %u\n", k, i, v[i]);
                return fail("state value");
            }
    }
    std::printf("PASSED: %d packets through the cycle, every state incremented exactly once\n", kPackets);
    return 0;
}
