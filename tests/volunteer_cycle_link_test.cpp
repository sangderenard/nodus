// The volunteer cycle across the Nodus link (case a11b5c9b1be74c9b9fb684790f7fd5a3): the library on one Nodus host, the
// work on a volunteer process, the cycle's two lossless edges carried by the link's bulk stream.
//
// Server (this process; a LinkListener on 127.0.0.1, it never dials):
//   tensor mint:        Input(6) | tensor_mint | Input(6) | tensor_mint      (U64[8,10] batches)
//   volunteer library:  Input(2) | nodus.volunteer.library | Output(1)       (Input 0: queued work; 1: from the link)
//   server link:        intake | intake | intake | cycle.server.bulk | tensor_returns | Input(3) | tensor_filter_send
//   edges: library Output -> intake (row 2) "supply"; link "received" -> library Input 1. Port pairs: mint.1 blank0 >
//   library blank, library spent > mint.1 returned, link returns > mint.3 returned (a sent batch goes back to the
//   mint), filter loop<i> > intake (row 2 - i) loop: a batch the link refused goes back up and is offered again first.
// Volunteer (a child process; it only dials out):
//   volunteer link:     intake | intake | intake | cycle.volunteer.bulk | tensor_returns | Input(3) | tensor_filter_send
//   volunteer work:     Input(1) | one work row per device and program | Output(1)
//   tensor mint:        Input(6) | tensor_mint                               (takes the volunteer's sent batches back)
//   edges: link "received" -> work Input; work Output -> intake (row 2) "supply". Port pairs: link returns > mint
//   returned, filter loop<i> > intake (row 2 - i) loop.
// The volunteer's own KernelLibrarySession holds the states and runs the jobs; the server's library queues the work,
// chooses targets, and finishes each packet from its receipt. When every packet is done the server closes the link;
// the volunteer then checks its states (each holds its own program's value) and exits with the result.
//
// Usage: test_volunteer_cycle_link run <glslangValidator> <nodus root> <module library root> <work dir>
#include "nodus_link_rows.h"
#include "kernel_library_session.h"
#include "nodus_headless_abi.h"
#include "exec_pipe.h"
#include "value_types.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace nodus::link;
namespace fs = std::filesystem;

namespace {
constexpr std::uint64_t kCard = 7, kProcess = 1, kBase = 100, kPrograms = 2;
constexpr std::uint32_t kWords = 64;
constexpr int kPackets = 16;
std::uint64_t program_of(int k) { return 1 + static_cast<std::uint64_t>(k / 2) % kPrograms; }
const char* g_who = "server";

int fail(const char* what) {
    std::printf("[%s] FAIL %s\n", g_who, what);
    return 1;
}
std::string hex(const std::uint8_t* p, std::size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (std::size_t i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; }
    return s;
}
std::vector<std::uint8_t> unhex(const std::string& s) {
    std::vector<std::uint8_t> v(s.size() / 2);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<std::uint8_t>(std::stoul(s.substr(2 * i, 2), nullptr, 16));
    return v;
}
std::uint64_t u64_tensor(const std::vector<std::uint64_t>& words) {
    const std::uint64_t shape[1] = {words.size()};
    const auto h = nodus_tensor_create(NODUS_DTYPE_U64, 1, shape);
    nodus_tensor_write(h, 0, words.data(), words.size() * sizeof(std::uint64_t));
    return h;
}
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
bool linked(const std::string& id, bool* closed) {
    EndpointReport r;
    if (!link_endpoint_report(id, &r)) return false;
    if (closed) *closed = r.state == EndpointState::Closed || r.state == EndpointState::Failed;
    return r.state == EndpointState::Linked;
}

// One headless graph and the moves both sides make on it.
struct Canvas {
    NodusHeadlessGraph* graph = nodus_headless_create();
    int tensor_type = nodus_headless_abstract_tensor_type_id();
    NodusHeadlessFifoConfig lossless{8, 4, 0, 0, 2, 16, 0};
    NodusHeadlessFifoConfig scalar{4, 4, 1, -1, 2, 0, 0};
    Canvas() { lossless.type_id = tensor_type; nodus_headless_set_scheduler_mode(graph, 1); }
    ~Canvas() { if (graph) nodus_headless_destroy(graph); }
    int table(const char* label, int x, std::vector<GP_CanvasModuleRowDesc> rows) {
        GP_CanvasModuleDesc desc{};
        desc.x = x; desc.w = 420; desc.h = 320;
        std::snprintf(desc.label, sizeof(desc.label), "%s", label);
        const int m = nodus_headless_add_module(graph, &desc);
        if (m < 0 || !nodus_headless_set_module_rows(graph, m, rows.data(), static_cast<int32_t>(rows.size())) ||
            !nodus_headless_set_module_stack_contract(graph, m, 1))
            return -1;
        return m;
    }
    bool edge(int a, int ac, int b, int bc) {
        GP_CanvasEdgeDesc e{};
        e.a_module = a; e.a_contact_idx = ac; e.b_module = b; e.b_contact_idx = bc;
        const int idx = nodus_headless_add_edge(graph, &e, 0);
        return idx >= 0 && nodus_headless_configure_edge_fifo(graph, idx, &lossless);
    }
    bool pair(const char* text, int slots = 4) {
        char error[512] = {};
        NodusHeadlessFifoConfig config = lossless;
        config.slots = slots;
        const int idx = nodus_headless_apply_port_pair(graph, text, error, sizeof(error));
        if (idx < 0) std::printf("[%s] port pair %s: %s\n", g_who, text, error);
        return idx >= 0 && nodus_headless_configure_edge_fifo(graph, idx, &config);
    }
    // A mint row's six parameters (dtype d0 d1 d2 d3 depth) on Input `contact0`.. of table `mint`.
    bool mint_settings(int mint, int contact0, const char* prefix) {
        const float values[6] = {14.0f, 8.0f, 10.0f, 0.0f, 0.0f, 4.0f};  // U64[8,10], keep 4
        for (int k = 0; k < 6; ++k) {
            const std::string name = std::string(prefix) + std::to_string(k);
            if (!nodus_headless_declare_input_port_ex(graph, name.c_str(), mint, contact0 + k, &scalar)) return false;
            nodus_headless_push_token(graph, name.c_str(), &values[k], sizeof(float));
        }
        return true;
    }
    bool step() { return nodus_headless_step(graph, 1.0 / 60.0) != 0; }
    // A row's own port as a table contact (canvas_abi.h).
    static int row_port(int row, int port) { return GP_CANVAS_ROW_PORT_CONTACT_BASE + row * GP_CANVAS_ROW_PORT_STRIDE + port; }
    // A link table's row-based loopback: rows 0..2 governed intakes (row 2 takes the supply), row 3 the link, row 4
    // returns, row 5 Input(3), row 6 filter send. loop<i> of the filter feeds the intake at row 2 - i.
    bool loopback(const char* table, int t, const char* prefix) {
        for (int i = 0; i < 3; ++i) {
            const std::string text = std::string(table) + ".6:loop" + std::to_string(i) + " > " + table + "." +
                                     std::to_string(2 - i) + ":loop";
            if (!pair(text.c_str())) return false;
        }
        const float sizes[3] = {640.0f, 0.0f, 0.0f};  // the batch, U64[8,10]
        for (int k = 0; k < 3; ++k) {
            const std::string name = std::string(prefix) + std::to_string(k);
            if (!nodus_headless_declare_input_port_ex(graph, name.c_str(), t, k, &scalar)) return false;
            nodus_headless_push_token(graph, name.c_str(), &sizes[k], sizeof(float));
        }
        return true;
    }
};
std::vector<GP_CanvasModuleRowDesc> link_rows(const char* link_id, const char* intake_id, const char* returns_id,
                                              const char* filter_id) {
    return {{1, 0, 0, 1, intake_id}, {1, 0, 0, 1, intake_id}, {1, 0, 0, 1, intake_id}, {1, 0, 0, 1, link_id},
            {1, 0, 0, 1, returns_id}, {0, 0, 3, 0, nullptr}, {1, 0, 0, 1, filter_id}};
}

// A module library row (module_library/source/tools/<name>.cpp), built into `bin_root`/bin and loaded; `id` is its
// loaded tool id. Each process builds into its own root: a loaded plugin's DLL cannot be rebuilt in place.
bool build_row(const char* name, const fs::path& nodus_root, const fs::path& library_root, const fs::path& bin_root,
               char* id, std::size_t cap) {
    return nodus_headless_build_plugin_source((library_root / "source" / "tools" / (std::string(name) + ".cpp")).string().c_str(),
                                              nodus_root.string().c_str(), bin_root.string().c_str(), id,
                                              static_cast<int32_t>(cap)) != 0;
}

int volunteer(int argc, char** argv) {
    g_who = "volunteer";
    // volunteer <port> <control ticket> <bulk ticket> <glslang> <nodus root> <module library> <work dir>
    if (argc != 9) return fail("usage");
    const fs::path glslang = argv[5], nodus_root = argv[6], library_root = argv[7], work = argv[8];
    const fs::path bin_root = work / "module_library";
    fs::create_directories(bin_root);
    LinkDialConfig dc;
    dc.id = "cycle.volunteer";
    dc.target.port = static_cast<std::uint16_t>(std::stoi(argv[2]));
    dc.target.prefix = "cycle";
    const auto c = unhex(argv[3]), b = unhex(argv[4]);
    if (c.size() != kTicketBytes || b.size() != kTicketBytes) return fail("tickets");
    std::memcpy(dc.control_ticket.data(), c.data(), kTicketBytes);
    std::memcpy(dc.bulk_ticket.data(), b.data(), kTicketBytes);
    std::string err;
    int rc = 1;
    {
        nodus::gpu::KernelLibrarySession session(tool_registry_global());
        if (!session || !session.library().device_count()) return fail("no Vulkan device");
        std::printf("[volunteer] library open on %zu devices\n", session.library().device_count());
        const auto devices = session.library().device_count();
        // The jobs, built here from their GLSL (build where the toolchain is): program p adds p.
        std::vector<nodus::gpu::GpuShaderJobContract> contracts(kPrograms);
        for (std::uint64_t p = 1; p <= kPrograms; ++p) {
            auto& contract = contracts[p - 1];
            contract.id = "volunteer.add" + std::to_string(p);
            contract.program_id = p;
            const auto comp = work / (contract.id + ".comp"), spv = work / (contract.id + ".spv");
            std::ofstream(comp) << "#version 450\nlayout(local_size_x = 64) in;\n"
                                   "layout(std430, binding = 0) buffer State { uint v[]; };\n"
                                   "void main() { uint i = gl_GlobalInvocationID.x; if (i < v.length()) v[i] += "
                                << p << "u; }\n";
            const std::string command = "\"\"" + glslang.string() + "\" -V --target-env vulkan1.1 -o \"" +
                                        spv.string() + "\" \"" + comp.string() + "\" > NUL\"";
            if (std::system(command.c_str()) != 0) return fail("glslangValidator");
            std::ifstream in(spv, std::ios::binary);
            std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            contract.spirv.resize(bytes.size() / 4);
            std::memcpy(contract.spirv.data(), bytes.data(), contract.spirv.size() * 4);
            contract.minimum_binding_bytes = {kWords * 4};
            contract.writable = {1};
            contract.local_size_x = 64;
            contract.maximum_invocations = kWords;
            if (!session.register_shader(contract) || !session.register_volunteer_rows(contract, 4))
                return fail("register");
        }
        // The states live in the volunteer's own library.
        ITool& library = session.library_row();
        std::array<std::uint64_t, 12> card{1, kCard, kProcess, 0, 1ull << 20};
        for (std::size_t d = 0; d < devices && d < 3; ++d) card[5 + d] = 1ull << 20;
        const auto dummy = u64_tensor({0});
        if (request(library, card, dummy) != 0) return fail("Card");
        nodus_tensor_destroy(dummy);
        for (int k = 0; k < kPackets; ++k) {
            const std::uint64_t shape[1] = {kWords};
            const auto state = nodus_tensor_create(NODUS_DTYPE_U32, 1, shape);
            nodus_tensor_zero(state, 0, 0);
            if (request(library, {2, kCard, kProcess, kBase + static_cast<std::uint64_t>(k)}, state) != 0)
                return fail("Put");
            nodus_tensor_destroy(state);
        }
        std::printf("[volunteer] programs registered, %d states in the library\n", kPackets);

        char mint_id[1024] = {}, returns_id[1024] = {}, intake_id[1024] = {}, filter_id[1024] = {};
        if (!build_row("tensor_mint", nodus_root, library_root, bin_root, mint_id, sizeof(mint_id)) ||
            !build_row("tensor_returns", nodus_root, library_root, bin_root, returns_id, sizeof(returns_id)) ||
            !build_row("tensor_governed_intake", nodus_root, library_root, bin_root, intake_id, sizeof(intake_id)) ||
            !build_row("tensor_filter_send", nodus_root, library_root, bin_root, filter_id, sizeof(filter_id)))
            return fail("build rows");
        std::printf("[volunteer] rows built\n");
        // The dial's clock (DialTarget::timeout_ms) starts here and it advances only as the link rows poll it, so the
        // dialer is registered once everything else is ready and the tables are about to step.
        if (!register_link_dialer(tool_registry_global(), dc, &err)) return fail(("dialer: " + err).c_str());
        std::printf("[volunteer] dialer registered\n");
        {
            Canvas canvas;
            const int link = canvas.table("volunteer link", 0,
                                          link_rows("cycle.volunteer.bulk", intake_id, returns_id, filter_id));
            std::vector<std::string> work_ids;
            for (std::size_t d = 0; d < devices; ++d)
                for (const auto& contract : contracts)
                    work_ids.push_back("nodus.volunteer.work." + contract.id + ".vulkan." + std::to_string(d));
            std::vector<GP_CanvasModuleRowDesc> rows{{0, 0, 1, 0, nullptr}};
            for (const auto& id : work_ids) rows.push_back({1, 0, 0, 1, id.c_str()});
            rows.push_back({2, 0, 1, 0, nullptr});
            const int wrk = canvas.table("volunteer work", 480, rows);
            const int mint = canvas.table("tensor mint", 960, {{0, 0, 6, 0, nullptr}, {1, 0, 0, 1, mint_id}});
            if (link < 0 || wrk < 0 || mint < 0) return fail("tables");
            if (!canvas.edge(link, Canvas::row_port(3, 0), wrk, 0) || !canvas.edge(wrk, 1, link, Canvas::row_port(2, 1)))
                return fail("edges");
            if (!canvas.pair("volunteer link.4:returns > tensor mint.1:returned", 16)) return fail("port pair");
            if (!canvas.loopback("volunteer link", link, "filter")) return fail("loopback");
            if (!canvas.mint_settings(mint, 0, "mint")) return fail("mint settings");
            std::printf("[volunteer] tables built; stepping\n");
            bool was_linked = false, closed = false;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
            while (std::chrono::steady_clock::now() < deadline) {
                if (!canvas.step()) return fail("step");
                const bool up = linked("cycle.volunteer", &closed);
                static auto last = std::chrono::steady_clock::now();
                if (!was_linked && std::chrono::steady_clock::now() - last > std::chrono::seconds(5)) {
                    last = std::chrono::steady_clock::now();
                    EndpointReport r;
                    link_endpoint_report("cycle.volunteer", &r);
                    std::printf("[volunteer] link state %d reason '%s' control sent %llu received %llu\n",
                                static_cast<int>(r.state), r.reason.c_str(), (unsigned long long)r.link.control_sent,
                                (unsigned long long)r.link.control_received);
                }
                was_linked = was_linked || up;
                if (was_linked && closed) {  // the server is done with us
                    EndpointReport r;
                    link_endpoint_report("cycle.volunteer", &r);
                    std::printf("[volunteer] link ended: state %d reason '%s' control sent %llu received %llu bulk "
                                "tensors sent %llu received %llu refused %llu dropped %llu violations %llu\n",
                                static_cast<int>(r.state), r.reason.c_str(), (unsigned long long)r.link.control_sent,
                                (unsigned long long)r.link.control_received, (unsigned long long)r.bulk_tensors_sent,
                                (unsigned long long)r.bulk_tensors_received, (unsigned long long)r.bulk_refused,
                                (unsigned long long)r.bulk_dropped, (unsigned long long)r.link.protocol_violations);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (!was_linked) return fail("never linked");
        }
        // Each state ran once, by its own packet's program.
        rc = 0;
        for (int k = 0; k < kPackets && rc == 0; ++k) {
            auto loan = session.library().borrow_cpu(kCard, kBase + static_cast<std::uint64_t>(k), false);
            if (!loan || loan.bytes() < kWords * 4) { rc = fail("borrow_cpu"); break; }
            const auto* v = static_cast<const std::uint32_t*>(loan.data());
            for (std::uint32_t i = 0; i < kWords; ++i)
                if (v[i] != program_of(k)) {
                    std::printf("[volunteer] state %d word %u = %u, program %llu\n", k, i, v[i],
                                (unsigned long long)program_of(k));
                    rc = fail("state value");
                    break;
                }
        }
        if (rc == 0) std::printf("[volunteer] every state ran once, by its own program\n");
    }
    unregister_link(tool_registry_global(), "cycle.volunteer");
    return rc;
}

int server(int argc, char** argv) {
    // run <glslang> <nodus root> <module library> <work dir>
    if (argc != 6) return fail("usage");
    const fs::path nodus_root = argv[3], library_root = argv[4], work = argv[5];
    const fs::path bin_root = work / "server" / "module_library";
    fs::create_directories(work / "volunteer");
    fs::create_directories(bin_root);
    Key key{};
    random_bytes(key.data(), key.size());
    LinkHostConfig hc;
    hc.id = "cycle.server";
    hc.prefix = "cycle";
    hc.key = key;
    std::uint16_t port = 0;
    std::string err;
    if (!register_link_host(tool_registry_global(), hc, &port, &err)) return fail(("listener: " + err).c_str());
    int rc = 1;
    nodus::exec::Child child;
    {
        nodus::gpu::KernelLibrarySession session(tool_registry_global());
        if (!session || !session.register_volunteer_library()) return fail("library");
        char mint_id[1024] = {}, returns_id[1024] = {}, intake_id[1024] = {}, filter_id[1024] = {};
        if (!build_row("tensor_mint", nodus_root, library_root, bin_root, mint_id, sizeof(mint_id)) ||
            !build_row("tensor_returns", nodus_root, library_root, bin_root, returns_id, sizeof(returns_id)) ||
            !build_row("tensor_governed_intake", nodus_root, library_root, bin_root, intake_id, sizeof(intake_id)) ||
            !build_row("tensor_filter_send", nodus_root, library_root, bin_root, filter_id, sizeof(filter_id)))
            return fail("build rows");
        Canvas canvas;
        const int mint = canvas.table("tensor mint", 0, {{0, 0, 6, 0, nullptr}, {1, 0, 0, 1, mint_id},
                                                          {0, 0, 6, 0, nullptr}, {1, 0, 0, 1, mint_id}});
        const int lib = canvas.table("volunteer library", 480, {{0, 0, 2, 0, nullptr},
                                                                 {1, 0, 0, 1, "nodus.volunteer.library"},
                                                                 {2, 0, 1, 0, nullptr}});
        const int link = canvas.table("server link", 960, link_rows("cycle.server.bulk", intake_id, returns_id, filter_id));
        if (mint < 0 || lib < 0 || link < 0) return fail("tables");
        if (!canvas.edge(lib, 2, link, Canvas::row_port(2, 1)) || !canvas.edge(link, Canvas::row_port(3, 0), lib, 1))
            return fail("edges");
        if (!canvas.pair("tensor mint.1:blank0 > volunteer library.1:blank") ||
            !canvas.pair("volunteer library.1:spent > tensor mint.1:returned") ||
            !canvas.pair("server link.4:returns > tensor mint.3:returned", 16) ||
            !canvas.loopback("server link", link, "filter"))
            return fail("port pairs");
        if (!canvas.mint_settings(mint, 0, "mint.a") || !canvas.mint_settings(mint, 6, "mint.b"))
            return fail("mint settings");
        if (!nodus_headless_declare_input_port_ex(canvas.graph, "work", lib, 0, &canvas.lossless))
            return fail("work port");

        // The volunteer: a child process that dials out with the tickets this service issued.
        const auto id = new_link_id();
        const std::uint64_t now = static_cast<std::uint64_t>(std::time(nullptr));
        const auto c = sign_ticket(make_ticket(Channel::Control, Role::Dialer, id, "cycle", 42, now, 120), key);
        const auto b = sign_ticket(make_ticket(Channel::Bulk, Role::Dialer, id, "cycle", 42, now, 120), key);
        nodus::exec::Program program;
        program.path = fs::absolute(argv[0]).string();
        program.working_dir = (work / "volunteer").string();
        program.arguments = {"volunteer", std::to_string(port), hex(c.data(), c.size()), hex(b.data(), b.size()),
                             argv[2], nodus_root.string(), library_root.string(), (work / "volunteer").string()};
        std::string why;
        if (!child.start(program, why)) return fail(("volunteer process: " + why).c_str());

        // Work goes out only once the link is up: a batch the link refuses would not survive the tick.
        const auto started = std::chrono::steady_clock::now();
        bool up = false;
        while (!up && std::chrono::steady_clock::now() - started < std::chrono::seconds(180)) {
            if (!canvas.step()) return fail("step");
            up = linked("cycle.server", nullptr);
            static auto last = std::chrono::steady_clock::now();
            if (std::chrono::steady_clock::now() - last > std::chrono::seconds(5)) {
                last = std::chrono::steady_clock::now();
                std::string seen;
                child.take(seen);
                std::size_t lines = 0;
                for (const char ch : seen) lines += ch == '\n';
                EndpointReport r;
                link_endpoint_report("cycle.server", &r);
                std::printf("[server] waiting for the link: volunteer running %d, wrote %zu bytes (%zu lines); endpoint "
                            "state %d account %llu peers refused %llu violations %llu control received %llu\n",
                            child.running() ? 1 : 0, seen.size(), lines, static_cast<int>(r.state),
                            (unsigned long long)r.account, (unsigned long long)r.peers_refused,
                            (unsigned long long)r.link.protocol_violations, (unsigned long long)r.link.control_received);
                for (std::size_t at = seen.find("[volunteer]"); at != std::string::npos; at = seen.find("[volunteer]", at + 1))
                    std::printf("  %s\n", seen.substr(at, seen.find('\n', at) - at).c_str());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!up) {
            std::string output;
            child.take(output);
            std::printf("[server] volunteer running %d, its output (last 3000 bytes):\n%s\n", child.running() ? 1 : 0,
                        output.substr(output.size() > 3000 ? output.size() - 3000 : 0).c_str());
            return fail("the volunteer never linked");
        }
        const auto work_started = std::chrono::steady_clock::now();
        int pushed = 0;
        nodus::gpu::TensorStateLibraryTool::CycleCounts counts{};
        while (std::chrono::steady_clock::now() - work_started < std::chrono::seconds(60)) {
            while (pushed < kPackets) {
                // Targets name the volunteer's devices; on this machine both hosts see the same three.
                const std::uint64_t target = (pushed % 2) ? 1 + static_cast<std::uint64_t>(pushed / 2) % session.library().device_count() : 0;
                const auto packet = u64_tensor({kCard, kBase + static_cast<std::uint64_t>(pushed), kWords,
                                                1000 + static_cast<std::uint64_t>(pushed), 0, target, program_of(pushed)});
                NodusHeadlessPublishResult r{};
                if (!nodus_headless_push_token_ex(canvas.graph, "work", &packet, sizeof(packet), &r) || !r.accepted) {
                    nodus_tensor_destroy(packet);
                    break;
                }
                ++pushed;
            }
            if (!canvas.step()) return fail("step");
            counts = session.library().cycle_counts();
            if (pushed == kPackets && counts.completed == kPackets && !counts.outstanding && !counts.queued) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - work_started).count();
        EndpointReport report;
        link_endpoint_report("cycle.server", &report);
        std::printf("[server] pushed %d completed %llu retried %llu outstanding %llu (%.3f s); bulk tensors sent %llu "
                    "received %llu; completed by device %llu/%llu/%llu\n",
                    pushed, (unsigned long long)counts.completed, (unsigned long long)counts.retried,
                    (unsigned long long)counts.outstanding, seconds, (unsigned long long)report.bulk_tensors_sent,
                    (unsigned long long)report.bulk_tensors_received, (unsigned long long)counts.completed_by_device[0],
                    (unsigned long long)counts.completed_by_device[1], (unsigned long long)counts.completed_by_device[2]);
        rc = counts.completed == kPackets ? 0 : fail("not every packet completed");
    }
    // Done: closing the link tells the volunteer to check its states.
    unregister_link(tool_registry_global(), "cycle.server");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (child.running() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::string output;
    child.take(output);
    for (std::size_t start = 0; start < output.size();) {  // the volunteer's own verdict lines
        const auto end = output.find('\n', start);
        const std::string line = output.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (line.rfind("[volunteer]", 0) == 0) std::printf("%s\n", line.c_str());
        if (end == std::string::npos) break;
        start = end + 1;
    }
    const int volunteer_rc = child.running() ? -1 : child.exit_code();
    std::printf("[server] volunteer exit code %d\n", volunteer_rc);
    if (rc == 0 && volunteer_rc == 0)
        std::printf("PASSED: %d packets of %llu programs across the link, each state run once by its own program\n",
                    kPackets, (unsigned long long)kPrograms);
    return rc == 0 && volunteer_rc == 0 ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "volunteer") return volunteer(argc, argv);
    if (mode == "run") return server(argc, argv);
    std::printf("usage: test_volunteer_cycle_link run <glslangValidator> <nodus root> <module library> <work dir>\n");
    return 2;
}
