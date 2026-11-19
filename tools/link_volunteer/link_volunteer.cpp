// link_volunteer: the volunteer as the whole system (volunteer case a11b5c9b1be74c9b9fb684790f7fd5a3, stage 3).
//
//   link_volunteer volunteer --port P --prefix X --control-ticket HEX --bulk-ticket HEX --work DIR
//       dials the server, sends its compiler settings, registers every job definition it is offered in its own
//       KernelLibrarySession, joins the server's contract market as a host and serves its loans.
//   link_volunteer test <spring row dir> <fft_cffi.dll> <wave_t4.dll> <work dir>
//       the server side: the real ContractNegotiator with a local host and a volunteer child process; one engine submits
//       a mixed workload of every library job kind (spring shader jobs, a two-stage pipeline, the CPU law, fftfree, wave
//       T4, and two stateful fabric programs, one pinned to each host); each volunteer result is re-run on the server
//       and byte-compared before it is committed.
// The spring row dir is turing tools/build_spring_nodus_row.py's output (spring_network_step_batched__tick.dll and
// shader/spring_network_step.comp + .glsl.json).
#include "nodus_link_jobs.h"
#include "nodus_link_contracts.h"
#include "contract_negotiator.h"
#include "exec_pipe.h"

#include <cstdio>
#include <ctime>
#include <limits>
#include <optional>

using namespace nodus::link;
using nodus::contracts::Contract;
using nodus::contracts::ContractNegotiator;
using nodus::contracts::Producer;
using Clock = std::chrono::steady_clock;

namespace {
std::string g_who = "?";
void say(const std::string& s) { std::printf("[%s] %s\n", g_who.c_str(), s.c_str()); std::fflush(stdout); }
std::vector<uint8_t> read_file(const fs::path& p) { std::ifstream f(p, std::ios::binary); return {std::istreambuf_iterator<char>(f), {}}; }
std::string hex(const uint8_t* p, size_t n) { static const char* d = "0123456789abcdef"; std::string s; for (size_t i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; } return s; }
std::vector<uint8_t> unhex(const std::string& s) { std::vector<uint8_t> v(s.size() / 2); for (size_t i = 0; i < v.size(); ++i) v[i] = uint8_t(std::stoul(s.substr(2 * i, 2), nullptr, 16)); return v; }

// Control instructions of the job layer (ControlKind::Instruction): 1 definition ready {id, status}, 2 done.
bool send_instruction(Link& link, std::initializer_list<uint64_t> words) {
    ControlPacket p; p.kind = uint16_t(ControlKind::Instruction); p.length = uint32_t(8 * words.size());
    size_t i = 0; for (auto w : words) put_le(p.payload.data() + 8 * i++, w, 8);
    return link.send_control(p);
}
std::vector<uint64_t> words_of(const ControlPacket& p) { std::vector<uint64_t> w(p.length / 8); for (size_t i = 0; i < w.size(); ++i) w[i] = get_le(p.payload.data() + 8 * i, 8); return w; }

constexpr uint64_t kCard = 1, kLanes = uint64_t(nodus::gpu::kFabricBatch);   // the CPU law's batch is fixed at 6144
constexpr uint64_t kStepProgram = 0x7370726e67737465ull, kPongProgram = 0x7370726e67706f6eull, kCpuProgram = 0x737072696e6731ull;
enum : uint64_t { DefStep = 1, DefPong = 2, DefPipeline = 3, DefCpuLaw = 4, DefFftfree = 5, DefT4 = 6, DefFabricV = 7, DefFabricL = 8 };

// ---- the volunteer -----------------------------------------------------------------------------------------------
int volunteer(int argc, char** argv) {
    g_who = "volunteer";
    int port = 0; std::string prefix = "vol", control_hex, bulk_hex; fs::path work = ".";
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i]; auto v = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--port") port = std::stoi(v()); else if (a == "--prefix") prefix = v();
        else if (a == "--control-ticket") control_hex = v(); else if (a == "--bulk-ticket") bulk_hex = v(); else if (a == "--work") work = v();
    }
    auto ct = unhex(control_hex), bt = unhex(bulk_hex);
    if (ct.size() != kTicketBytes || bt.size() != kTicketBytes) { say("tickets missing"); return 2; }
    TicketBytes c{}, b{}; std::memcpy(c.data(), ct.data(), kTicketBytes); std::memcpy(b.data(), bt.data(), kTicketBytes);
    DialTarget t; t.port = uint16_t(port); t.prefix = prefix;
    LinkDialer dialer(t, c, b);
    std::unique_ptr<Link> link;
    for (auto end = Clock::now() + std::chrono::seconds(10); Clock::now() < end && !link && !dialer.failed();) { dialer.poll(); if (dialer.ready()) link = dialer.take(); }
    if (!link) { say("dial failed"); return 1; }
    ControlMux control(*link); BulkMux bulk(*link);

    // 1. Compiler settings.
    SettingsRecord mine = collect_settings();
    bulk.send(BulkStreamKind::Settings, 1, encode_settings(mine));
    const std::string target = mine.gpus.empty() ? std::string() : vulkan_target(mine.gpus[0].api_version);
    LibraryRunner runner(work, mine.glslang_path, target, 0);
    if (!runner) { say("library refused"); return 1; }
    say("joined: " + std::to_string(mine.gpus.size()) + " GPU(s), " + (mine.has_glslang ? "builds its own GLSL (glslang)" : "no glslang"));

    std::map<uint64_t, Definition> pending;
    std::set<uint64_t> answered;                      // definitions this volunteer has reported ready or refused
    std::map<uint64_t, WorkItem> items;
    std::map<std::string, int> served;
    bool done = false;
    LinkContractHost host(control,
        [&](const Loan& loan) { return runner.ready(loan.work.revision); },
        [&](const Loan& loan, uint64_t& payload) {
            auto it = items.find(loan.work.payload);
            if (it == items.end()) return false;                       // its inputs are still arriving
            WorkOutputs out = runner.run(it->second);
            bulk.send(BulkStreamKind::Outputs, out.id, encode_outputs(out));
            ++served[kind_name(pending.count(loan.work.revision) ? pending[loan.work.revision].kind : JobKind::ShaderJob)];
            items.erase(it);
            payload = out.status; return true;
        });
    for (auto end = Clock::now() + std::chrono::seconds(300); Clock::now() < end && !link->closed();) {
        link->poll(); bulk.pump();
        TypedStream s;
        while (bulk.take(BulkStreamKind::Artifact, s)) {
            std::array<uint8_t, 32> d = sha256(s.bytes.data(), s.bytes.size());
            if (s.aux.size() == 32 && std::memcmp(d.data(), s.aux.data(), 32) == 0) runner.add_artifact(d, s.bytes); else say("artifact refused: digest");
        }
        while (bulk.take(BulkStreamKind::Vulkan, s)) {
            if (s.aux.size() != 80) continue;
            std::array<uint8_t, 32> g{}, v{}; std::memcpy(g.data(), s.aux.data() + 16, 32); std::memcpy(v.data(), s.aux.data() + 48, 32);
            if (sha256(s.bytes.data(), s.bytes.size()) == v) runner.add_vulkan(g, s.bytes);
        }
        while (bulk.take(BulkStreamKind::Definition, s)) { Definition d; if (decode_definition(s.bytes, d)) pending[d.id] = d; }
        while (bulk.take(BulkStreamKind::WorkItem, s)) { WorkItem w; if (decode_work(s.bytes, w)) items[w.id] = std::move(w); }
        for (auto& [id, d] : pending) {
            if (answered.count(id)) continue;
            const std::string why = runner.register_definition(d);
            if (why == "pending") continue;
            answered.insert(id);
            send_instruction(*link, {1, id, why.empty() ? 0u : 1u});
            say(std::string(why.empty() ? "registered " : "REFUSED ") + kind_name(d.kind) + " " + d.name + (why.empty() ? "" : ": " + why));
        }
        host.poll();
        ControlPacket p;
        while (control.take(ControlKind::Instruction, p)) if (auto w = words_of(p); !w.empty() && w[0] == 2) done = true;
        if (done && !host.busy() && bulk.idle()) break;
    }
    std::string line = "served";
    for (const auto& [k, n] : served) line += " " + k + " x" + std::to_string(n) + ";";
    say(line + " own glslang builds identical to the server's: " + std::to_string(runner.local_builds_matching()) +
        ", different: " + std::to_string(runner.local_builds_differing()));
    for (auto end = Clock::now() + std::chrono::milliseconds(300); Clock::now() < end;) { link->poll(); bulk.pump(); }
    return done ? 0 : 1;
}

// ---- the server's workload ---------------------------------------------------------------------------------------
TensorBytes spring_column(uint64_t i, double seed) {
    if (i == 26) return tensor1<uint32_t>(NODUS_DTYPE_U32, {uint32_t(kLanes)});
    std::vector<double> v(kLanes, 0.0);
    for (uint64_t l = 0; l < kLanes; ++l) {
        double x = 0;
        if (i == 0 || i == 17) x = 1.0 + seed + double(l) * 1e-6;
        if (i == 3) x = .05; if (i == 4) x = .001; if (i == 9) x = -9.8; if (i == 10 || i == 15) x = 1.0; if (i == 22) x = 10.0;
        v[l] = x;
    }
    return tensor1<double>(NODUS_DTYPE_F64, v);
}
std::vector<double> sheets(int variant) {   // eight 16x16 sheets (as tools/fabric_state_test)
    constexpr int Nets = 8, Nodes = 256, Fields = 38, Side = 16;
    std::vector<double> d(size_t(Nets) * Nodes * Fields, 0.0);
    for (int net = 0; net < Nets; net++) for (int r = 0; r < Side; r++) for (int c = 0; c < Side; c++) {
        const int node = r * Side + c; double* row = &d[(size_t(net) * Nodes + node) * Fields];
        const double x = c * 0.05 + net * 1.0, y = 1.5 - r * 0.05 + 0.1 * variant, z = 0.01 * net;
        row[0] = x; row[1] = y; row[2] = z; row[6] = 1; row[7] = (r == 0 && (c == 0 || c == Side - 1)) ? 1 : 0;
        row[8] = x; row[9] = y; row[10] = z; row[11] = 1.0 / (0.2 * (1 + net % 3)); row[12] = 40; row[13] = 0.08 + 0.01 * net; row[15] = -9.8;
        const int nb[4][2] = {{r, c - 1}, {r, c + 1}, {r - 1, c}, {r + 1, c}};
        for (int e = 0; e < 4; e++) {
            const int rr = nb[e][0], cc = nb[e][1];
            if (rr < 0 || rr >= Side || cc < 0 || cc >= Side) continue;
            row[17 + e] = rr * Side + cc; row[33 + e] = 1; row[21 + e * 3] = (cc - c) * 0.05; row[21 + e * 3 + 1] = -(rr - r) * 0.05;
        }
        if (node == 0) row[37] = 100 + net + 10 * variant;
    }
    return d;
}

int test(int argc, char** argv) {
    g_who = "server";
    if (argc != 6) { say("usage: link_volunteer test <spring row dir> <fft_cffi.dll> <wave_t4.dll> <work dir>"); return 2; }
    const fs::path row_dir = argv[2], work = fs::absolute(argv[5]);
    const auto llvm = read_file(row_dir / "spring_network_step_batched__tick.dll");
    const auto glsl = read_file(row_dir / "shader" / "spring_network_step.comp");
    const auto fft = read_file(argv[3]), t4 = read_file(argv[4]);
    if (llvm.empty() || glsl.empty() || fft.empty() || t4.empty()) { say("products missing"); return 2; }
    const auto glsl_d = sha256(glsl.data(), glsl.size()), llvm_d = sha256(llvm.data(), llvm.size());
    const auto fft_d = sha256(fft.data(), fft.size()), t4_d = sha256(t4.data(), t4.size());
    const ShaderNeeds needs{true, 64};                               // the shader product's record: float64, local_size_x 64

    const SettingsRecord ours = collect_settings();
    LibraryRunner server(work / "server", ours.glslang_path, vulkan_target(ours.gpus.at(0).api_version), 0);
    for (const auto* p : {&llvm, &glsl, &fft, &t4}) server.add_artifact(sha256(p->data(), p->size()), *p);

    // The definitions: every library job kind.
    std::vector<Definition> defs;
    auto shader = [&](uint64_t id, const std::string& name, uint64_t program, bool pong) {
        Definition d; d.id = id; d.kind = JobKind::ShaderJob; d.name = name; d.program_id = program; d.glsl = glsl_d; d.has_glsl = true; d.needs = needs;
        for (int i = 0; i < 27; i++) d.binding_bytes.push_back(i == 26 ? 4 : kLanes * 8);
        d.writable.assign(27, 0); d.writable[24] = d.writable[25] = 1;
        if (pong) { for (uint64_t i = 0; i < 27; i++) d.offsets.push_back(i); d.offsets[17] = 24; d.offsets[23] = 25; d.offsets[24] = 17; d.offsets[25] = 23; }
        d.local_x = 64; d.max_x = uint32_t(kLanes); return d;
    };
    defs.push_back(shader(DefStep, "nodus.job.spring.step", kStepProgram, false));
    defs.push_back(shader(DefPong, "nodus.job.spring.pong", kPongProgram, true));
    { Definition d; d.id = DefPipeline; d.kind = JobKind::Pipeline; d.name = "nodus.job.spring.two";
      d.components = {{"nodus.job.spring.step", 1, kStepProgram, true}, {"nodus.job.spring.pong", 1, kPongProgram, true}}; defs.push_back(d); }
    { Definition d; d.id = DefCpuLaw; d.kind = JobKind::CpuLaw; d.name = "nodus.fabric.dispatch.cpu"; d.dll = llvm_d; d.has_dll = true;
      d.entry = "spring_network_step_batched__tick"; d.batch = kLanes; d.program_id = kCpuProgram; defs.push_back(d); }
    { Definition d; d.id = DefFftfree; d.kind = JobKind::Fftfree; d.name = "nodus.fftfree"; d.dll = fft_d; d.has_dll = true; defs.push_back(d); }
    { Definition d; d.id = DefT4; d.kind = JobKind::WaveT4; d.name = "nodus.wave.t4.state"; d.dll = t4_d; d.has_dll = true; defs.push_back(d); }
    auto fabric = [&](uint64_t id, const std::string& name, const std::string& backend, uint64_t base) {
        Definition d; d.id = id; d.kind = JobKind::FabricProgram; d.stateful = true; d.name = name; d.backend = backend; d.card = kCard; d.base = base;
        d.dll = llvm_d; d.has_dll = true; d.entry = "spring_network_step_batched__tick"; d.local_x = 64; d.glsl = glsl_d; d.has_glsl = true; d.needs = needs; return d;
    };
    defs.push_back(fabric(DefFabricV, "spring.network.library.v", "vulkan.0", 50000));
    defs.push_back(fabric(DefFabricL, "spring.network.library.l", "cpu", 60000));
    for (auto end = Clock::now() + std::chrono::seconds(60); Clock::now() < end;) {   // the server registers all of them
        bool all = true;
        for (const auto& d : defs) { auto why = server.register_definition(d); if (why == "pending") all = false; else if (!why.empty()) { say("server cannot register " + d.name + ": " + why); return 1; } }
        if (all) break;
    }

    // The link and the volunteer process.
    Key key{}; random_bytes(key.data(), key.size());
    ListenerConfig lc; lc.prefix = "vol"; lc.key = key;
    LinkListener listener(lc);
    const auto lid = new_link_id(); const uint64_t now = uint64_t(std::time(nullptr));
    auto ct = sign_ticket(make_ticket(Channel::Control, Role::Dialer, lid, "vol", 42, now, 300), key);
    auto bt = sign_ticket(make_ticket(Channel::Bulk, Role::Dialer, lid, "vol", 42, now, 300), key);
    fs::create_directories(work / "volunteer");
    nodus::exec::Program vp; vp.path = fs::absolute(argv[0]).string(); vp.working_dir = (work / "volunteer").string();
    vp.arguments = {"volunteer", "--port", std::to_string(listener.port()), "--prefix", "vol", "--control-ticket", hex(ct.data(), ct.size()),
                    "--bulk-ticket", hex(bt.data(), bt.size()), "--work", (work / "volunteer").string()};
    nodus::exec::Child child; std::string why, relay;
    if (!child.start(vp, why)) { say("volunteer refused: " + why); return 1; }
    auto relay_out = [&] { child.take(relay); size_t nl; while ((nl = relay.find('\n')) != std::string::npos) { std::printf("%s\n", relay.substr(0, nl).c_str()); relay.erase(0, nl + 1); } std::fflush(stdout); };
    std::unique_ptr<Link> link;
    for (auto end = Clock::now() + std::chrono::seconds(20); Clock::now() < end && !link;) { listener.poll(uint64_t(std::time(nullptr))); link = listener.take(); relay_out(); }
    if (!link) { say("the volunteer did not link"); return 1; }
    ControlMux control(*link); BulkMux bulk(*link);
    auto pump = [&] { link->poll(); bulk.pump(); relay_out(); };

    // Its settings decide what it is offered.
    TypedStream st; SettingsRecord theirs;
    for (auto end = Clock::now() + std::chrono::seconds(20); Clock::now() < end && !bulk.take(BulkStreamKind::Settings, st);) pump();
    if (!decode_settings(st.bytes, theirs) || theirs.gpus.empty()) { say("no settings from the volunteer"); return 1; }
    const auto& gpu = theirs.gpus[0];
    const bool gpu_ok = unmet_need(gpu, needs) == nullptr, cpu_ok = theirs.cpu_triple == ours.cpu_triple || theirs.has_clang;
    say("volunteer gpu 0 " + gpu.name + " (" + vulkan_target(gpu.api_version) + "): " + (gpu_ok ? "meets the shaders' needs" : unmet_need(gpu, needs)) +
        "; cpu " + theirs.cpu_triple + (cpu_ok ? " (CPU products built for it)" : " (no CPU jobs)"));
    GlslangBuilds builds(ours.glslang_path, work / "server-builds");
    std::vector<uint8_t> vulkan; bool failed = false;
    if (gpu_ok) for (auto end = Clock::now() + std::chrono::seconds(30); Clock::now() < end && !builds.build(glsl, vulkan_target(gpu.api_version), vulkan, failed) && !failed;) pump();
    auto send_artifact = [&](const std::vector<uint8_t>& bytes) { auto d = sha256(bytes.data(), bytes.size()); bulk.send(BulkStreamKind::Artifact, get_le(d.data(), 8), bytes, std::vector<uint8_t>(d.begin(), d.end())); };
    if (gpu_ok && !vulkan.empty()) {
        send_artifact(glsl);
        const std::string target = vulkan_target(gpu.api_version);
        std::vector<uint8_t> aux(target.begin(), target.end()); aux.resize(16, 0);
        auto vd = sha256(vulkan.data(), vulkan.size()); aux.insert(aux.end(), glsl_d.begin(), glsl_d.end()); aux.insert(aux.end(), vd.begin(), vd.end());
        bulk.send(BulkStreamKind::Vulkan, 0, vulkan, aux);
    }
    if (cpu_ok) { send_artifact(llvm); send_artifact(fft); send_artifact(t4); }
    std::set<uint64_t> offered_defs;
    for (const auto& d : defs) {
        if (d.id == DefFabricL) continue;                                         // pinned to the server's own host
        const bool gpu_kind = d.kind == JobKind::ShaderJob || d.kind == JobKind::Pipeline || (d.kind == JobKind::FabricProgram && d.backend != "cpu");
        const bool cpu_kind = d.kind == JobKind::CpuLaw || d.kind == JobKind::Fftfree || d.kind == JobKind::WaveT4 || d.kind == JobKind::FabricProgram;
        if ((gpu_kind && !gpu_ok) || (cpu_kind && !cpu_ok)) continue;
        bulk.send(BulkStreamKind::Definition, d.id, encode_definition(d)); offered_defs.insert(d.id);
    }
    std::set<uint64_t> volunteer_ready;

    // The workload: every kind, mixed. Phase 1 is a market with both hosts; phase 2 runs after the local host has left the
    // market, so every job of it can only go to the volunteer (a capability check that needs no placement policy).
    std::map<uint64_t, WorkItem> items;      // by id (stable storage)
    std::vector<uint64_t> order;             // submission order
    uint64_t next = 1;
    auto build_batch = [&](int phase) {
    const uint64_t off = uint64_t(phase) * 100000; const double ph = 0.05 * phase;
    std::vector<WorkItem> work_items;
    auto add = [&](uint64_t def, WorkItem w) { w.id = next++; w.definition = def; w.card = kCard; work_items.push_back(std::move(w)); };
    for (int k = 0; k < 6; ++k) {                                                // spring step as a shader job
        WorkItem w; const uint64_t base = off + 10000 + uint64_t(k) * 100;
        for (uint64_t i = 0; i < 27; ++i) w.puts.push_back({base + i, spring_column(i, ph + 0.1 * k)});
        w.inputs.push_back(tensor1<uint64_t>(NODUS_DTYPE_U64, {kCard, base, kLanes, 1}));
        w.outputs = {base + 24, base + 25}; add(DefStep, w);
    }
    for (int k = 0; k < 4; ++k) {                                                // two steps on the device, ping-pong
        WorkItem w; const uint64_t base = off + 20000 + uint64_t(k) * 100;
        for (uint64_t i = 0; i < 27; ++i) w.puts.push_back({base + i, spring_column(i, ph + 0.2 + 0.1 * k)});
        TensorBytes schedule; schedule.dtype = NODUS_DTYPE_U64; schedule.shape = {2, 7};
        const uint64_t rows[14] = {0, 1, kStepProgram, kCard, base, kLanes, 1, 1, 1, kPongProgram, kCard, base, kLanes, 2};
        schedule.bytes.resize(sizeof(rows)); std::memcpy(schedule.bytes.data(), rows, sizeof(rows));
        w.inputs.push_back(schedule); w.outputs = {base + 17, base + 23}; add(DefPipeline, w);
    }
    for (int k = 0; k < 4; ++k) {                                                // the CPU law (Turing's LLVM product)
        WorkItem w; const uint64_t base = off + 30000 + uint64_t(k) * 100;
        for (uint64_t i = 0; i < 27; ++i) w.puts.push_back({base + i, spring_column(i, ph + 0.5 + 0.1 * k)});
        w.inputs.push_back(tensor1<uint64_t>(NODUS_DTYPE_U64, {kCard, base, kLanes, 1})); w.outputs = {base + 24, base + 25}; add(DefCpuLaw, w);
    }
    for (int k = 0; k < 3; ++k) {                                                // fftfree (as tests/fftfree_tool_test.cpp)
        WorkItem w; const uint64_t b = off + 40000 + uint64_t(k) * 10;
        std::vector<float> input(8); for (int i = 0; i < 8; ++i) input[size_t(i)] = float(std::sin(0.7 * i * (k + 1)));
        w.puts.push_back({b, tensor1<float>(NODUS_DTYPE_F32, input)});
        for (uint64_t a = b + 1; a <= b + 3; ++a) w.puts.push_back({a, tensor1<float>(NODUS_DTYPE_F32, std::vector<float>(5, 0.f))});
        std::vector<uint64_t> wk(10, 0); wk[0] = 1; wk[1] = kCard; wk[2] = b; wk[4] = b + 1; wk[5] = b + 2; wk[6] = b + 3; wk[7] = 8; wk[8] = 1; wk[9] = 7001 + uint64_t(k);
        std::vector<uint64_t> cfg(40, 0); cfg[0] = 1; cfg[1] = 8; cfg[2] = 1; cfg[3] = 1; cfg[5] = 1; cfg[6] = 2; cfg[16] = 2; cfg[17] = 8; cfg[18] = 8; cfg[19] = 1; cfg[20] = 1; cfg[23] = 1;
        w.inputs = {tensor1<uint64_t>(NODUS_DTYPE_U64, wk), tensor1<uint64_t>(NODUS_DTYPE_U64, cfg)};
        w.outputs = {b + 1, b + 2, b + 3}; add(DefFftfree, w);
    }
    auto bits = [](double v) { uint64_t u; std::memcpy(&u, &v, 8); return u; };
    for (int k = 0; k < 3; ++k) {                                                // wave T4 (as tests/wave_t4_state_tool_test.cpp)
        WorkItem w; const uint64_t b = off + 45000 + uint64_t(k) * 10;
        w.puts.push_back({b, tensor1<double>(NODUS_DTYPE_F64, {450e-9 + 10e-9 * k, 850e-9})});
        for (uint64_t a = b + 1; a <= b + 4; ++a) { std::vector<float> f(256, 0.f); if (a == b + 1 || a == b + 3) f[8 * 16 + 8 + size_t(k)] = 1; w.puts.push_back({a, tensor1<float>(NODUS_DTYPE_F32, f)}); }
        w.inputs = {tensor1<uint64_t>(NODUS_DTYPE_U64, {kCard, b, 777, 42 + uint64_t(k), b + 1, b + 2, b + 3, b + 4}),
                    tensor1<uint64_t>(NODUS_DTYPE_U64, {1, 16, 16, 2, 1, bits(0.2e-6), bits(0.25e-6), 1})};
        w.outputs = {b + 1, b + 2, b + 3, b + 4}; add(DefT4, w);
    }
    for (int f = 0; f < (phase ? 1 : 2); ++f) {                                  // two fabric programs, configure + 30 ticks
        const uint64_t def = f ? DefFabricL : DefFabricV;       // phase 2: the volunteer's fabric continues, ticks only
        TensorBytes cfg; cfg.dtype = NODUS_DTYPE_F64; cfg.shape = {8, 256, 38}; auto s = sheets(f);
        cfg.bytes.resize(s.size() * 8); std::memcpy(cfg.bytes.data(), s.data(), cfg.bytes.size());
        if (!phase) { WorkItem w; w.inputs = {cfg}; add(def, w); }
        for (int t = 0; t < 30; ++t) { WorkItem w; TensorBytes dt; dt.dtype = NODUS_DTYPE_F64; dt.bytes.resize(8); const double v = 1.0 / 60; std::memcpy(dt.bytes.data(), &v, 8); w.inputs = {dt}; add(def, w); }
    }
    // A mixed queue: interleave the kinds (each kind keeps its own order, so a fabric program's ticks stay in sequence).
    {
        std::map<uint64_t, std::deque<WorkItem>> lanes;
        for (auto& w : work_items) lanes[w.definition].push_back(std::move(w));
        work_items.clear();
        for (bool more = true; more;) {
            more = false;
            for (auto& [def, q] : lanes) if (!q.empty()) { work_items.push_back(std::move(q.front())); q.pop_front(); more = true; }
        }
    }
    for (auto& w : work_items) { order.push_back(w.id); const uint64_t id = w.id; items.emplace(id, std::move(w)); }
    return order.size();
    };
    size_t total = build_batch(0);
    const size_t phase1 = total;
    int phase = 1;

    // The market: the real negotiator; a local host (the server's own runner) and the volunteer over the link.
    const uint64_t volunteer_id = link->account(), local_id = 7;
    auto definition_of = [&](uint64_t def) -> const Definition& { for (const auto& d : defs) if (d.id == def) return d; return defs[0]; };
    ContractNegotiator market(256, [&](const nodus::contracts::Peer& peer, const std::vector<ContractNegotiator::Entry>& entries) {
        // Stage 3's placement: a host is offered only what it has registered; a stateful program stays on its host.
        for (size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].state != ContractNegotiator::State::Unaccepted) continue;
            const auto& d = definition_of(entries[i].work.revision);
            if (d.id == DefFabricV && peer.id != volunteer_id) continue;
            if (d.id == DefFabricL && peer.id != local_id) continue;
            if (peer.id == volunteer_id && !volunteer_ready.count(d.id)) continue;
            return i;
        }
        return std::numeric_limits<size_t>::max();
    }, 1000000);
    std::map<uint64_t, WorkOutputs> local_outputs, volunteer_outputs;
    LinkHostPeer vpeer(control, volunteer_id, [&](const Loan& loan) {
        if (auto it = items.find(loan.work.payload); it != items.end()) bulk.send(BulkStreamKind::WorkItem, loan.work.payload, encode_work(it->second));
    });
    market.add_host(vpeer.peer());
    std::optional<Loan> local_loan; bool local_due = false;
    nodus::contracts::Peer local; local.id = local_id;
    local.exchange = [&](const Loan* offer) {
        nodus::contracts::Exchange e{};
        if (local_loan && local_due) { e.has_result = true; e.result = {*local_loan, local_outputs[local_loan->work.payload].status}; local_loan.reset(); local_due = false; }
        else if (local_loan) { local_outputs[local_loan->work.payload] = server.run(items.at(local_loan->work.payload)); local_due = true; }
        if (offer && !local_loan) { local_loan = *offer; e.accepted_generation = offer->generation; }
        return e;
    };
    market.add_host(local);

    size_t submitted = 0, routed = 0, mismatched = 0, rechecked = 0;
    std::map<std::string, std::array<int, 2>> by_kind;      // [local, volunteer], phase 1
    std::map<std::string, int> phase2_by_kind;              // volunteer, phase 2
    double volunteer_ms = 0, local_ms = 0;
    Producer engine; engine.id = 900;
    engine.request = [&](Contract& c) {
        if (submitted >= total) return false;
        if (volunteer_ready.size() < offered_defs.size()) return false;   // this test starts once the volunteer has registered
        const auto& w = items.at(order[submitted]); c = {w.id, 900, w.definition, w.id}; return true;
    };
    engine.admitted = [&](uint64_t) { ++submitted; };
    engine.route = [&](const nodus::contracts::Result& r) {
        const uint64_t id = r.loan.work.payload;
        const bool from_volunteer = r.loan.host == volunteer_id;
        auto& store = from_volunteer ? volunteer_outputs : local_outputs;
        auto it = store.find(id);
        if (it == store.end()) return false;                 // its outputs are still on the bulk pair: not committed yet
        if (it->second.status != 0) { say("job " + std::to_string(id) + " failed on its host"); ++mismatched; }
        if (from_volunteer) {                                // the server re-runs the volunteer's work and compares
            WorkOutputs mine = server.run(items.at(id)); ++rechecked;
            if (!(mine == it->second)) { ++mismatched; say("job " + std::to_string(id) + " (" + kind_name(definition_of(r.loan.work.revision).kind) + ") DIFFERS from the server's run"); }
            volunteer_ms += it->second.ms;
        } else local_ms += it->second.ms;
        if (phase == 1) ++by_kind[kind_name(definition_of(r.loan.work.revision).kind)][from_volunteer ? 1 : 0];
        else ++phase2_by_kind[kind_name(definition_of(r.loan.work.revision).kind)];
        ++routed; return true;
    };
    market.add_engine(engine);

    const auto t0 = Clock::now();
    for (auto end = t0 + std::chrono::seconds(240); Clock::now() < end && routed < total;) {
        pump();
        TypedStream o;
        while (bulk.take(BulkStreamKind::Outputs, o)) { WorkOutputs x; if (decode_outputs(o.bytes, x)) volunteer_outputs[x.id] = std::move(x); }
        ControlPacket p;
        while (control.take(ControlKind::Instruction, p)) {
            auto w = words_of(p);
            if (w.size() >= 3 && w[0] == 1) { if (w[2] == 0) volunteer_ready.insert(w[1]); else say("the volunteer refused definition " + std::to_string(w[1])); }
        }
        market.worker_turn(); market.market_turn();
        if (phase == 1 && routed == phase1) {                  // phase 2: the local host leaves; the volunteer alone
            market.remove_host(local_id); phase = 2; total = build_batch(1);
            say("phase 1 done; the local host leaves the market; phase 2: " + std::to_string(total - phase1) + " jobs for the volunteer alone");
        }
    }
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    send_instruction(*link, {2});
    for (auto end = Clock::now() + std::chrono::seconds(20); Clock::now() < end && !child.finished();) pump();
    relay_out(); { std::string rest; child.take(rest); if (!rest.empty()) std::printf("%s", rest.c_str()); }

    say("market: " + std::to_string(routed) + " of " + std::to_string(total) + " jobs routed in " + std::to_string(s).substr(0, 5) + " s; definitions the volunteer registered: " +
        std::to_string(volunteer_ready.size()) + " of " + std::to_string(offered_defs.size()) + " offered");
    for (const auto& [k, n] : by_kind) say("  phase 1 " + k + ": local host " + std::to_string(n[0]) + ", volunteer " + std::to_string(n[1]));
    for (const auto& [k, n] : phase2_by_kind) say("  phase 2 " + k + ": volunteer " + std::to_string(n));
    say("volunteer results re-run on the server: " + std::to_string(rechecked) + ", differing: " + std::to_string(mismatched));
    // Every kind the volunteer registered must have been served by it, not only registered.
    std::set<std::string> kinds_offered; for (const auto& d : defs) if (offered_defs.count(d.id)) kinds_offered.insert(kind_name(d.kind));
    std::string unserved;
    for (const auto& k : kinds_offered) if (by_kind[k][1] + phase2_by_kind[k] == 0) unserved += " " + k + ";";
    if (!unserved.empty()) say("the volunteer registered but was never given:" + unserved);
    const bool ok = routed == total && mismatched == 0 && rechecked > 0 && child.finished() && child.exit_code() == 0 &&
                    volunteer_ready.size() == offered_defs.size() && unserved.empty();
    say(ok ? "PASSED: the volunteer served every job kind it registered, from the market; every result matched the server's own run" : "FAILED");
    return ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "volunteer") return volunteer(argc, argv);
    if (mode == "test") return test(argc, argv);
    std::printf("usage: link_volunteer test <spring row dir> <fft_cffi.dll> <wave_t4.dll> <work dir>\n"
                "       link_volunteer volunteer --port P --prefix X --control-ticket HEX --bulk-ticket HEX --work DIR\n");
    return 2;
}
