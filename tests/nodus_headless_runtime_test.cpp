// Verifies the nodus_headless ABI: real ThreadManager scheduling (via
// gp_canvas_step, never a hand-written per-tool tick loop) reaches
// quiescence with no window/renderer, ports round-trip real bytes through
// the same table-edge transport real graph edges use, save/load preserves
// enough structure to re-resolve the same ports, and edge overflow is
// observable through the event queue.
// Winsock headers must precede anything that may pull in <windows.h>.
#include "contract_socket.h"
#include "contract_listener.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>

#include "nodus_headless_abi.h"

#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")  // test client sockets; CMake links ws2_32 only into nodus_headless
#endif

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

    // Run from a fresh empty cwd so we can prove nothing lands in the cwd.
    namespace fs = std::filesystem;
    const fs::path prev_cwd = fs::current_path();
    const fs::path fresh_cwd = fs::temp_directory_path() / "nodus_headless_test_cwd";
    std::error_code cwd_ec;
    fs::remove_all(fresh_cwd, cwd_ec);
    fs::create_directories(fresh_cwd);
    fs::remove_all(fs::path(path.string() + ".modules"), cwd_ec);
    fs::current_path(fresh_cwd);

    NodusHeadlessGraph* g1 = nodus_headless_create();
    assert(g1);
    int32_t module_a, row_a, module_b, row_b;
    build_two_module_graph(g1, &module_a, &row_a, &module_b, &row_b);
    assert(nodus_headless_set_module_exec_mode(g1, module_a, 3));
    assert(nodus_headless_set_module_stack_contract(g1, module_a, 0));
    assert(!nodus_headless_set_module_exec_mode(g1, module_a, 4));
    assert(nodus_headless_save_file(g1, path.string().c_str()));
    nodus_headless_destroy(g1);

    // Serialized tables travel beside the canvas file, never in the cwd.
    const fs::path modules_dir = fs::path(path.string() + ".modules");
    assert(fs::is_directory(modules_dir / "module_library" / "serialized") ||
           fs::is_directory(modules_dir / "serialized"));
    {
        bool found_gpmod = false;
        for (const auto& e : fs::recursive_directory_iterator(modules_dir)) {
            if (e.path().extension() == ".gpmod") found_gpmod = true;
        }
        assert(found_gpmod);
    }
    assert(!fs::exists(fresh_cwd / "module_library" / "serialized"));

    NodusHeadlessGraph* g2 = nodus_headless_create();
    assert(g2);
    assert(nodus_headless_load_file(g2, path.string().c_str()));
    int32_t exec_mode = -1;
    int32_t stack_contract = -1;
    assert(nodus_headless_get_module_exec_mode(g2, module_a, &exec_mode) && exec_mode == 3);
    assert(nodus_headless_get_module_stack_contract(g2, module_a, &stack_contract) && stack_contract == 0);
    assert(!fs::exists(fresh_cwd / "module_library" / "serialized"));
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

static void test_duplicate_module_uuid_is_refused() {
    NodusHeadlessGraph* graph = nodus_headless_create();
    assert(graph);
    GP_CanvasModuleDesc desc{};
    std::strncpy(desc.label, "identity", sizeof(desc.label) - 1);
    int32_t first = nodus_headless_add_module(graph, &desc);
    int32_t second = nodus_headless_add_module(graph, &desc);
    assert(first >= 0 && second >= 0);
    assert(nodus_headless_set_module_uuid(graph, first, 0xD00D));
    assert(!nodus_headless_set_module_uuid(graph, second, 0xD00D));
    assert(nodus_headless_find_module_by_uuid(graph, 0xD00D) == first);
    nodus_headless_destroy(graph);
    std::printf("test_duplicate_module_uuid_is_refused: OK\n");
}

static void test_unregistered_plugin_row_is_refused() {
    NodusHeadlessGraph* graph = nodus_headless_create();
    assert(graph);
    GP_CanvasModuleDesc desc{};
    std::strncpy(desc.label, "plugin", sizeof(desc.label) - 1);
    int32_t module = nodus_headless_add_module(graph, &desc);
    assert(module >= 0);
    GP_CanvasModuleRowDesc good[2]{};
    good[0].kind = 0; good[0].attachment_count = 1;
    good[1].kind = 2; good[1].attachment_count = 1;
    assert(nodus_headless_set_module_rows(graph, module, good, 2));
    int32_t before = nodus_headless_get_module_row_count(graph, module);
    GP_CanvasModuleRowDesc bad[1]{};
    bad[0].kind = 1;
    bad[0].tool_origin = 1;
    bad[0].plugin_id = "no.such.plugin";
    assert(!nodus_headless_set_module_rows(graph, module, bad, 1));
    assert(nodus_headless_get_module_row_count(graph, module) == before);
    nodus_headless_destroy(graph);
    std::printf("test_unregistered_plugin_row_is_refused: OK\n");
}

// Exercise the real canvas -> ThreadManager route, not just an isolated
// predecessor-count helper. The partial cycle has a ready upstream node.
static void test_partial_cycle_freespinning_completes() {
    auto* graph = nodus_headless_create();
    assert(graph);
    int modules[3];
    int rows[3];
    for (int i = 0; i < 3; ++i) {
        GP_CanvasModuleDesc desc{};
        desc.w = desc.h = 64;
        std::snprintf(desc.label, sizeof(desc.label), "cycle_%d", i);
        modules[i] = nodus_headless_add_module(graph, &desc);
        assert(modules[i] >= 0);
        rows[i] = nodus_headless_bind_builtin_tool(graph, modules[i], kToolTensorTool);
        assert(rows[i] >= 0);
    }
    const int ends[3][2] = {{0, 1}, {1, 2}, {2, 1}};
    for (const auto& pair : ends) {
        GP_CanvasEdgeDesc edge{};
        edge.a_module = modules[pair[0]];
        edge.a_contact_idx = rows[pair[0]];
        edge.b_module = modules[pair[1]];
        edge.b_contact_idx = rows[pair[1]];
        assert(nodus_headless_add_edge(graph, &edge, 0) >= 0);
    }
    assert(nodus_headless_set_scheduler_mode(graph, 0));
    assert(nodus_headless_step(graph, 1.0 / 60.0));
    nodus_headless_destroy(graph);
    std::printf("test_partial_cycle_freespinning_completes: OK\n");
}

static bool same_config(const NodusHeadlessFifoConfig& a, const NodusHeadlessFifoConfig& b) {
    return a.token_bytes == b.token_bytes && a.slots == b.slots && a.top_k == b.top_k &&
           a.type_id == b.type_id && a.layout == b.layout && a.dtype == b.dtype &&
           a.backend_type == b.backend_type;
}

static void test_save_load_persists_ports_and_edge_fifo() {
    std::filesystem::path path = std::filesystem::temp_directory_path() / "nodus_headless_test_ports.txt";
    std::filesystem::path plain = std::filesystem::temp_directory_path() / "nodus_headless_test_plain_v4.txt";

    NodusHeadlessGraph* g1 = nodus_headless_create();
    assert(g1);
    int32_t module_a, row_a, module_b, row_b;
    int32_t edge_idx = build_two_module_graph(g1, &module_a, &row_a, &module_b, &row_b);
    // A V4 file saved before anything is configured carries no trailer.
    assert(nodus_headless_save_file(g1, plain.string().c_str()));

    NodusHeadlessFifoConfig edge_cfg{};
    edge_cfg.token_bytes = 24; edge_cfg.slots = 7; edge_cfg.top_k = 3;
    edge_cfg.type_id = nodus_headless_abstract_tensor_type_id();
    edge_cfg.backend_type = 0;
    assert(nodus_headless_configure_edge_fifo(g1, edge_idx, &edge_cfg));
    NodusHeadlessFifoConfig in_cfg = edge_cfg;
    in_cfg.token_bytes = 16; in_cfg.slots = 5; in_cfg.top_k = 2;
    NodusHeadlessFifoConfig out_cfg = edge_cfg;
    out_cfg.token_bytes = 8; out_cfg.slots = 9; out_cfg.top_k = 4;
    // Output declared first so declaration order differs from in/out grouping.
    assert(nodus_headless_declare_output_port_ex(g1, "result port", module_b, row_b, &out_cfg, /*start_at_head=*/0));
    assert(nodus_headless_declare_input_port_ex(g1, "feed", module_a, row_a, &in_cfg));
    assert(nodus_headless_save_file(g1, path.string().c_str()));
    nodus_headless_destroy(g1);

    NodusHeadlessGraph* g2 = nodus_headless_create();
    assert(g2);
    assert(nodus_headless_load_file(g2, path.string().c_str()));
    NodusHeadlessFifoConfig got{};
    assert(nodus_headless_get_edge_fifo_config(g2, 0, &got));
    assert(same_config(got, edge_cfg));

    assert(nodus_headless_get_port_count(g2, 0) == 1);
    assert(nodus_headless_get_port_count(g2, 1) == 1);
    char name[64];
    int32_t mod = -1, contact = -1, start = -1;
    assert(nodus_headless_get_port_at(g2, 0, 0, name, sizeof(name), &mod, &contact, &start, &got));
    assert(std::strcmp(name, "feed") == 0 && mod == module_a && contact == row_a && start == 1);
    assert(same_config(got, in_cfg));
    assert(nodus_headless_get_port_at(g2, 1, 0, name, sizeof(name), &mod, &contact, &start, &got));
    assert(std::strcmp(name, "result port") == 0 && mod == module_b && contact == row_b && start == 0);
    assert(same_config(got, out_cfg));
    assert(nodus_headless_get_port_fifo_config(g2, "feed", 0, &got) && same_config(got, in_cfg));
    assert(nodus_headless_get_port_fifo_config(g2, "result port", 1, &got) && same_config(got, out_cfg));
    nodus_headless_destroy(g2);

    NodusHeadlessGraph* g3 = nodus_headless_create();
    assert(g3);
    assert(nodus_headless_load_file(g3, plain.string().c_str()));
    assert(nodus_headless_get_port_count(g3, 0) == 0);
    assert(nodus_headless_get_port_count(g3, 1) == 0);
    assert(nodus_headless_get_edge_count(g3) == 1);
    nodus_headless_destroy(g3);

    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(plain, ec);
    std::printf("test_save_load_persists_ports_and_edge_fifo: OK\n");
}

// ---- Contract port: loopback socket transport owned by the canvas ----------
// The negotiator pair is process-owned (acquired through the ABI; the
// registry itself lives inside the DLL image). Participants reach it only
// through the listener the canvas opens and the headless step.
#define REQUIRE(x) do{if(!(x)){std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)
using namespace nodus::contracts;

static SOCKET contract_connect(uint16_t port) {
    ensure_sockets();
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    REQUIRE(s != INVALID_SOCKET);
    sockaddr_in a{};
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(port);
    REQUIRE(connect(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0);
    return s;
}
static WireMessage contract_hello(uint64_t role, uint64_t id) {
    WireMessage m{WireKind::Hello, {}}; m.result.loan.work = {role, id, 2, 0}; return m;
}
static WireMessage contract_submit(Contract c) {
    WireMessage m{WireKind::Submit, {}}; m.result.loan.work = c; return m;
}
static WireMessage contract_decision(uint64_t id, uint64_t generation, uint64_t value) {
    WireMessage m{WireKind::Decision, {}};
    m.result.loan.work.id = id; m.result.loan.generation = generation; m.result.payload = value; return m;
}

static void test_contract_port_loopback_and_persistence() {
    const char* kPrefix = "hl.proof";
    const std::string market_id = std::string(kPrefix) + ".market";
    const std::string worker_id = std::string(kPrefix) + ".worker";
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "nodus_headless_test_contract_port.txt";
    REQUIRE(nodus_headless_acquire_contract_pair(kPrefix, 4));

    NodusHeadlessGraph* g = nodus_headless_create();
    REQUIRE(g);
    GP_CanvasModuleDesc desc{}; desc.w = desc.h = 64;
    const int m = nodus_headless_add_module(g, &desc), w = nodus_headless_add_module(g, &desc);
    REQUIRE(m >= 0 && w >= 0);
    const int mr = nodus_headless_bind_plugin_tool(g, m, market_id.c_str());
    const int wr = nodus_headless_bind_plugin_tool(g, w, worker_id.c_str());
    REQUIRE(mr >= 0 && wr >= 0);
    GP_CanvasEdgeDesc edge{};
    edge.a_module = m; edge.a_contact_idx = mr; edge.b_module = w; edge.b_contact_idx = wr;
    const int edge_id = nodus_headless_add_edge(g, &edge, 0);
    REQUIRE(edge_id >= 0);
    NodusHeadlessFifoConfig config{};
    config.token_bytes = sizeof(Result); config.slots = 4; config.top_k = 0;
    config.type_id = 0x534A5231; config.layout = 2; config.dtype = 8;
    REQUIRE(nodus_headless_configure_edge_fifo(g, edge_id, &config));
    REQUIRE(nodus_headless_set_scheduler_mode(g, 1));

    // Refusals: empty prefix, bad capacity, unattached edge, whitespace prefix.
    REQUIRE(!nodus_headless_declare_contract_port(g, "", 4, edge_id, 0));
    REQUIRE(!nodus_headless_declare_contract_port(g, kPrefix, 0, edge_id, 0));
    REQUIRE(!nodus_headless_declare_contract_port(g, kPrefix, 4, edge_id + 5, 0));
    REQUIRE(!nodus_headless_declare_contract_port(g, "a b", 4, edge_id, 0));
    const int32_t port = nodus_headless_declare_contract_port(g, kPrefix, 4, edge_id, 0);
    REQUIRE(port > 0);
    REQUIRE(!nodus_headless_declare_contract_port(g, kPrefix, 4, edge_id, 0));  // already declared
    int32_t got_port = 0, hosts = -1, engines = -1;
    REQUIRE(nodus_headless_get_contract_port(g, kPrefix, &got_port, &hosts, &engines));
    REQUIRE(got_port == port && hosts == 0 && engines == 0);
    REQUIRE(nodus_headless_get_contract_port_count(g) == 1);
    {
        char name[64]; int32_t cap = 0, eidx = -1, lport = 0;
        REQUIRE(nodus_headless_get_contract_port_at(g, 0, name, sizeof(name), &cap, &eidx, &lport));
        REQUIRE(std::strcmp(name, kPrefix) == 0 && cap == 4 && eidx == edge_id && lport == port);
    }

    const uint64_t kEngine = 7, kHost = 1, kHostPayload = 555;
    ContractSocket engine_client(contract_connect(static_cast<uint16_t>(port)));
    ContractSocket host_client(contract_connect(static_cast<uint16_t>(port)));
    REQUIRE(engine_client.send_message(contract_hello(2, kEngine)));
    REQUIRE(engine_client.send_message(contract_submit({100, kEngine, 3, 4})));
    REQUIRE(host_client.send_message(contract_hello(1, kHost)));

    bool admitted = false; int offers = 0, routes = 0, receipts = 0;
    for (int tick = 0; tick < 5000 && !(admitted && offers && routes && receipts); ++tick) {
        REQUIRE(nodus_headless_step(g, 1.0 / 60.0));
        engine_client.poll(); host_client.poll();
        WireMessage msg{};
        while (engine_client.receive(msg)) {
            if (msg.kind == WireKind::Admitted) { REQUIRE(msg.result.loan.work.id == 100); admitted = true; }
            if (msg.kind == WireKind::Route) {
                REQUIRE(msg.result.payload == kHostPayload && msg.result.loan.host == kHost &&
                        msg.result.loan.work.engine == kEngine);
                ++routes;
                REQUIRE(engine_client.send_message(contract_decision(msg.result.loan.work.id, msg.result.loan.generation, 1)));
            }
        }
        while (host_client.receive(msg)) {
            if (msg.kind == WireKind::Offer) {
                ++offers;
                REQUIRE(host_client.send_message({WireKind::Accept, {msg.result.loan, 0}}));
                REQUIRE(host_client.send_message({WireKind::Result, {msg.result.loan, kHostPayload}}));
            }
            if (msg.kind == WireKind::Receipt) { REQUIRE(msg.result.payload == 1); ++receipts; }
        }
        engine_client.poll(); host_client.poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(admitted && offers == 1 && routes == 1 && receipts == 1);
    REQUIRE(nodus_headless_get_contract_port(g, kPrefix, nullptr, &hosts, &engines) && hosts == 1 && engines == 1);

    // Save, tear down (listener/sockets before graph), forget the process-owned
    // pair as a fresh process would, then load: the port must come back live.
    REQUIRE(nodus_headless_save_file(g, path.string().c_str()));
    nodus_headless_destroy(g);
    REQUIRE(nodus_headless_release_contract_pair(kPrefix));
    REQUIRE(nodus_headless_acquire_contract_pair(kPrefix, 4));

    NodusHeadlessGraph* g2 = nodus_headless_create();
    REQUIRE(g2);
    REQUIRE(nodus_headless_load_file(g2, path.string().c_str()));
    REQUIRE(nodus_headless_set_scheduler_mode(g2, 1));
    int32_t port2 = 0;
    REQUIRE(nodus_headless_get_contract_port(g2, kPrefix, &port2, &hosts, &engines));
    REQUIRE(port2 > 0 && hosts == 0 && engines == 0 && nodus_headless_get_contract_port_count(g2) == 1);

    ContractSocket engine2(contract_connect(static_cast<uint16_t>(port2)));
    REQUIRE(engine2.send_message(contract_hello(2, 8)));
    REQUIRE(engine2.send_message(contract_submit({200, 8, 1, 2})));
    bool admitted2 = false;
    for (int tick = 0; tick < 5000 && !admitted2; ++tick) {
        REQUIRE(nodus_headless_step(g2, 1.0 / 60.0));
        engine2.poll();
        WireMessage msg{};
        while (engine2.receive(msg))
            if (msg.kind == WireKind::Admitted) { REQUIRE(msg.result.loan.work.id == 200); admitted2 = true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(admitted2);
    nodus_headless_destroy(g2);
    REQUIRE(nodus_headless_release_contract_pair(kPrefix));
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::printf("test_contract_port_loopback_and_persistence: OK\n");
}

// Engine id reuse after its socket closed, duplicate LIVE id refused, loan_ticks carried by
// the pair and persisted in the CONTRACT_PORT trailer. Real loopback contract port; no compile.
static void test_contract_port_engine_reconnect_and_loan_ticks() {
    const char* kPrefix = "hl.reconnect";
    const std::string market_id = std::string(kPrefix) + ".market";
    const std::string worker_id = std::string(kPrefix) + ".worker";
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "nodus_headless_test_reconnect.txt";
    // Loan length rules.
    REQUIRE(!nodus_headless_acquire_contract_pair_with_loan(kPrefix, 4, 0));
    REQUIRE(nodus_headless_acquire_contract_pair_with_loan(kPrefix, 4, 600));
    REQUIRE(nodus_headless_contract_pair_loan_ticks(kPrefix) == 600);
    REQUIRE(nodus_headless_acquire_contract_pair_with_loan(kPrefix, 4, 600));   // same length: shared acquire
    REQUIRE(!nodus_headless_acquire_contract_pair_with_loan(kPrefix, 4, 61));   // different length refused
    REQUIRE(nodus_headless_acquire_contract_pair(kPrefix, 4));                  // unspecified: takes the pair as is
    REQUIRE(nodus_headless_contract_pair_loan_ticks(kPrefix) == 600);

    NodusHeadlessGraph* g = nodus_headless_create();
    REQUIRE(g);
    GP_CanvasModuleDesc desc{}; desc.w = desc.h = 64;
    const int m = nodus_headless_add_module(g, &desc), w = nodus_headless_add_module(g, &desc);
    const int mr = nodus_headless_bind_plugin_tool(g, m, market_id.c_str());
    const int wr = nodus_headless_bind_plugin_tool(g, w, worker_id.c_str());
    REQUIRE(m >= 0 && w >= 0 && mr >= 0 && wr >= 0);
    GP_CanvasEdgeDesc edge{};
    edge.a_module = m; edge.a_contact_idx = mr; edge.b_module = w; edge.b_contact_idx = wr;
    const int edge_id = nodus_headless_add_edge(g, &edge, 0);
    REQUIRE(edge_id >= 0);
    NodusHeadlessFifoConfig config{};
    config.token_bytes = sizeof(Result); config.slots = 4; config.top_k = 0;
    config.type_id = 0x534A5231; config.layout = 2; config.dtype = 8;
    REQUIRE(nodus_headless_configure_edge_fifo(g, edge_id, &config));
    REQUIRE(nodus_headless_set_scheduler_mode(g, 1));
    const int32_t port = nodus_headless_declare_contract_port(g, kPrefix, 4, edge_id, 0);
    REQUIRE(port > 0);

    const uint64_t kHostPayload = 777;
    ContractSocket host_client(contract_connect(static_cast<uint16_t>(port)));
    REQUIRE(host_client.send_message(contract_hello(1, 1)));
    auto engine_count = [&]() { int32_t h = 0, e = -1; REQUIRE(nodus_headless_get_contract_port(g, kPrefix, nullptr, &h, &e)); return e; };
    // One full cycle for the engine on `client`: Admitted, host Offer/Accept/Result, Route, Decision.
    auto cycle = [&](ContractSocket& client, uint64_t engine, uint64_t cid) {
        REQUIRE(client.send_message(contract_submit({cid, engine, 1, 2})));
        bool admitted = false, routed = false;
        for (int tick = 0; tick < 5000 && !routed; ++tick) {
            REQUIRE(nodus_headless_step(g, 1.0 / 60.0));
            client.poll(); host_client.poll();
            WireMessage msg{};
            while (client.receive(msg)) {
                if (msg.kind == WireKind::Admitted) { REQUIRE(msg.result.loan.work.id == cid); admitted = true; }
                if (msg.kind == WireKind::Route) {
                    REQUIRE(msg.result.loan.work.id == cid && msg.result.payload == kHostPayload);
                    REQUIRE(msg.result.loan.generation >= 1);
                    routed = true;
                    REQUIRE(client.send_message(contract_decision(cid, msg.result.loan.generation, 1)));
                }
            }
            while (host_client.receive(msg))
                if (msg.kind == WireKind::Offer) {
                    REQUIRE(host_client.send_message({WireKind::Accept, {msg.result.loan, 0}}));
                    REQUIRE(host_client.send_message({WireKind::Result, {msg.result.loan, kHostPayload}}));
                }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        REQUIRE(admitted && routed);
        for (int i = 0; i < 20; ++i) { REQUIRE(nodus_headless_step(g, 1.0 / 60.0)); client.poll(); std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    };

    {   // First life of engine 9001.
        ContractSocket first(contract_connect(static_cast<uint16_t>(port)));
        REQUIRE(first.send_message(contract_hello(2, 9001)));
        cycle(first, 9001, 1);
        REQUIRE(engine_count() == 1);
    }   // socket closed here
    for (int tick = 0; tick < 2000 && engine_count() != 0; ++tick) {
        REQUIRE(nodus_headless_step(g, 1.0 / 60.0));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(engine_count() == 0);   // the closed engine was removed

    // Reconnect with the same id; a second LIVE connection with that id is refused.
    ContractSocket again(contract_connect(static_cast<uint16_t>(port)));
    REQUIRE(again.send_message(contract_hello(2, 9001)));
    cycle(again, 9001, 2);
    REQUIRE(engine_count() == 1);
    ContractSocket dup(contract_connect(static_cast<uint16_t>(port)));
    REQUIRE(dup.send_message(contract_hello(2, 9001)));
    for (int tick = 0; tick < 3000 && !dup.closed(); ++tick) {
        REQUIRE(nodus_headless_step(g, 1.0 / 60.0));
        dup.poll(); again.poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(dup.closed() && !again.closed() && engine_count() == 1);

    // The loan length is in the trailer and a fresh process-owned pair comes back with it.
    REQUIRE(nodus_headless_save_file(g, path.string().c_str()));
    nodus_headless_destroy(g);
    REQUIRE(nodus_headless_release_contract_pair(kPrefix));
    REQUIRE(nodus_headless_contract_pair_loan_ticks(kPrefix) == 0);
    NodusHeadlessGraph* g2 = nodus_headless_create();
    REQUIRE(g2 && nodus_headless_load_file(g2, path.string().c_str()));
    REQUIRE(nodus_headless_contract_pair_loan_ticks(kPrefix) == 600);
    nodus_headless_destroy(g2);
    REQUIRE(nodus_headless_release_contract_pair(kPrefix));
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::printf("test_contract_port_engine_reconnect_and_loan_ticks: OK\n");
}

int main() {
    {
        NodusHeadlessGraph* scheduler = nodus_headless_create();
        assert(scheduler);
        int32_t persistent = -1;
        assert(nodus_headless_get_persistent_workers(scheduler, &persistent) && persistent == 1);
        assert(nodus_headless_set_persistent_workers(scheduler, 0));
        assert(nodus_headless_get_persistent_workers(scheduler, &persistent) && persistent == 0);
        assert(nodus_headless_set_persistent_workers(scheduler, 1));
        nodus_headless_destroy(scheduler);
    }
    test_partial_cycle_freespinning_completes();
    test_quiescence_and_event();
    test_save_load_round_trip();
    test_save_load_persists_ports_and_edge_fifo();
    test_overflow_event();
    test_duplicate_module_uuid_is_refused();
    test_unregistered_plugin_row_is_refused();
    test_contract_port_loopback_and_persistence();
    test_contract_port_engine_reconnect_and_loan_ticks();
    return 0;
}
