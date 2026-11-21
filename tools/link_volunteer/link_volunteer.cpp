// link_volunteer: the volunteer as the whole system (volunteer case a11b5c9b1be74c9b9fb684790f7fd5a3, stages 3-4).
//
//   link_volunteer volunteer --port P --prefix X --control-ticket HEX --bulk-ticket HEX --work DIR
//       dials the server, sends its compiler settings, registers every job definition it is offered in its own
//       KernelLibrarySession, and joins the server's contract market as one host per device (one per GPU UUID, plus
//       its CPU), each running its loan on its own thread.
//   link_volunteer test <spring row dir> <fft_cffi.dll> <wave_t4.dll> <work dir>
//       the server side: the real ContractNegotiator with market_offer (resident_fabric_offer's ranking through a
//       view of local and volunteer hosts); one host per local device plus the CPU; a mixed workload of every library
//       job kind is run twice: on the local hosts alone, then with the volunteer's hosts joined. Each volunteer result
//       is re-run on the server on the same device slot and byte-compared before it is committed.
// The spring row dir is turing tools/build_spring_nodus_row.py's output.
#include "nodus_link_jobs.h"
#include "nodus_link_contracts.h"
#include "nodus_market_offer.h"
#include "contract_negotiator.h"
#include "exec_pipe.h"

#include <cstdio>
#include <ctime>
#include <future>
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
bool send_instruction(Link& link, std::initializer_list<uint64_t> words) {
    ControlPacket p; p.kind = uint16_t(ControlKind::Instruction); p.length = uint32_t(8 * words.size());
    size_t i = 0; for (auto w : words) put_le(p.payload.data() + 8 * i++, w, 8);
    return link.send_control(p);
}
std::vector<uint64_t> words_of(const ControlPacket& p) { std::vector<uint64_t> w(p.length / 8); for (size_t i = 0; i < w.size(); ++i) w[i] = get_le(p.payload.data() + 8 * i, 8); return w; }
uint64_t now_ns() { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count()); }

constexpr uint64_t kCard = 1, kLanes = uint64_t(nodus::gpu::kFabricBatch);   // the CPU law's batch is fixed at 6144
constexpr uint64_t kStepProgram = 0x7370726e67737465ull, kPongProgram = 0x7370726e67706f6eull, kCpuProgram = 0x737072696e6731ull;
enum : uint64_t { DefStep = 1, DefPong = 2, DefPipeline = 3, DefCpuLaw = 4, DefFftfree = 5, DefT4 = 6,
                  DefFabricA1 = 11, DefFabricA2 = 12, DefFabricB1 = 21, DefFabricB2 = 22 };
constexpr uint32_t kCpuSlot = 0xFFFF;

// One market host per GPU (first library slot per UUID) and one for the CPU; ids are the account's space.
struct HostSlot { uint64_t id = 0; uint32_t slot = 0; bool gpu = true; std::string name; std::array<uint8_t, 16> uuid{}; };
std::vector<HostSlot> hosts_of(const SettingsRecord& r, uint64_t space) {
    std::vector<HostSlot> out; std::set<std::array<uint8_t, 16>> seen;
    uint32_t slot = 0;                                                   // the library's bank: one per physical device
    for (uint32_t i = 0; i < r.gpus.size(); ++i) {
        if (!seen.insert(r.gpus[i].uuid).second) continue;               // the same physical device listed twice
        out.push_back({(space << 16) | (slot + 1), slot, true, r.gpus[i].name, r.gpus[i].uuid}); ++slot;
    }
    out.push_back({(space << 16) | kCpuSlot, kCpuSlot, false, "CPU " + r.cpu_brand});
    return out;
}
bool gpu_kind(const Definition& d) { return d.kind == JobKind::ShaderJob || d.kind == JobKind::Pipeline || (d.kind == JobKind::FabricProgram && d.backend != "cpu"); }

// A host's execution: prepare on the calling (main) thread, the row on the host's own thread.
struct Execution {
    std::unique_ptr<LibraryRunner::Lane> lane;
    std::optional<std::future<WorkOutputs>> running;
};

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
    ControlMux control(*link); BulkMux bulk(*link); ContractRouter router(control);

    SettingsRecord mine = collect_settings();
    bulk.send(BulkStreamKind::Settings, 1, encode_settings(mine));
    const std::string target = mine.gpus.empty() ? std::string() : vulkan_target(mine.gpus[0].api_version);
    LibraryRunner runner(work, mine.glslang_path, target, 0);
    if (!runner) { say("library refused"); return 1; }
    const auto slots = hosts_of(mine, link->account());
    say("joined: " + std::to_string(slots.size()) + " market hosts (" + std::to_string(slots.size() - 1) + " GPU, 1 CPU), " +
        (mine.has_glslang ? "builds its own GLSL (glslang)" : "no glslang"));

    std::map<uint64_t, Definition> definitions; std::set<uint64_t> answered;
    std::map<uint64_t, WorkItem> items;
    std::map<std::string, int> served;
    std::map<uint64_t, Execution> executions;
    std::vector<std::unique_ptr<LinkContractHost>> hosts;
    for (const auto& s : slots) {
        auto& ex = executions[s.id]; ex.lane = std::make_unique<LibraryRunner::Lane>(s.gpu ? s.slot : 0);
        hosts.push_back(std::make_unique<LinkContractHost>(router, s.id,
            [&, s](const Loan& loan) {
                auto d = definitions.find(loan.work.revision);
                return d != definitions.end() && runner.ready(d->first) && gpu_kind(d->second) == s.gpu;
            },
            [&, s](const Loan& loan, uint64_t& payload) {
                auto& e = executions[s.id];
                if (!e.running) {
                    auto it = items.find(loan.work.payload);
                    if (it == items.end()) return false;                 // its inputs are still arriving
                    if (!runner.prepare(it->second, *e.lane)) { WorkOutputs bad; bad.id = it->first; bulk.send(BulkStreamKind::Outputs, bad.id, encode_outputs(bad)); payload = 1; items.erase(it); return true; }
                    WorkItem w = std::move(it->second); items.erase(it);
                    e.running = std::async(std::launch::async, [&runner, &e, w = std::move(w)] { return runner.execute(w, *e.lane); });
                    return false;
                }
                if (e.running->wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
                WorkOutputs out = e.running->get(); e.running.reset();
                bulk.send(BulkStreamKind::Outputs, out.id, encode_outputs(out));
                ++served[kind_name(definitions[loan.work.revision].kind)];
                payload = out.status; return true;
            }));
    }
    bool done = false;
    for (auto end = Clock::now() + std::chrono::seconds(600); Clock::now() < end && !link->closed();) {
        link->poll(); bulk.pump();
        TypedStream s;
        while (bulk.take(BulkStreamKind::Artifact, s)) {
            const auto d = sha256(s.bytes.data(), s.bytes.size());
            if (s.aux.size() == 32 && std::memcmp(d.data(), s.aux.data(), 32) == 0) runner.add_artifact(d, s.bytes); else say("artifact refused: digest");
        }
        while (bulk.take(BulkStreamKind::Vulkan, s)) {
            if (s.aux.size() != 80) continue;
            std::array<uint8_t, 32> g{}, v{}; std::memcpy(g.data(), s.aux.data() + 16, 32); std::memcpy(v.data(), s.aux.data() + 48, 32);
            if (sha256(s.bytes.data(), s.bytes.size()) == v) runner.add_vulkan(g, s.bytes);
        }
        while (bulk.take(BulkStreamKind::Definition, s)) { Definition d; if (decode_definition(s.bytes, d)) definitions[d.id] = d; }
        while (bulk.take(BulkStreamKind::WorkItem, s)) { WorkItem w; if (decode_work(s.bytes, w)) items[w.id] = std::move(w); }
        for (auto& [id, d] : definitions) {
            if (answered.count(id)) continue;
            const std::string why = runner.register_definition(d);
            if (why == "pending") continue;
            answered.insert(id);
            send_instruction(*link, {1, id, why.empty() ? 0u : 1u});
            if (!why.empty()) say(std::string("REFUSED ") + kind_name(d.kind) + " " + d.name + ": " + why);
        }
        for (auto& h : hosts) h->poll();
        ControlPacket p;
        while (control.take(ControlKind::Instruction, p)) if (auto w = words_of(p); !w.empty() && w[0] == 2) done = true;
        bool busy = false; for (auto& h : hosts) busy |= h->busy();
        if (done && !busy && bulk.idle()) break;
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
    const ShaderNeeds needs{true, 64};

    const SettingsRecord ours = collect_settings();
    LibraryRunner server(work / "server", ours.glslang_path, vulkan_target(ours.gpus.at(0).api_version), 0);
    for (const auto* p : {&llvm, &glsl, &fft, &t4}) server.add_artifact(sha256(p->data(), p->size()), *p);

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
    // run A (local hosts alone): both fabrics local; run B: the GPU fabric on the volunteer's device 0, the CPU fabric local
    defs.push_back(fabric(DefFabricA1, "spring.network.library.a1", "vulkan.0", 50000));
    defs.push_back(fabric(DefFabricA2, "spring.network.library.a2", "cpu", 60000));
    defs.push_back(fabric(DefFabricB1, "spring.network.library.b1", "vulkan.0", 70000));
    defs.push_back(fabric(DefFabricB2, "spring.network.library.b2", "cpu", 80000));
    for (auto end = Clock::now() + std::chrono::seconds(60); Clock::now() < end;) {
        bool all = true;
        for (const auto& d : defs) { auto why = server.register_definition(d); if (why == "pending") all = false; else if (!why.empty()) { say("server cannot register " + d.name + ": " + why); return 1; } }
        if (all) break;
    }
    auto definition_of = [&](uint64_t def) -> const Definition& { for (const auto& d : defs) if (d.id == def) return d; return defs[0]; };

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
    ControlMux control(*link); BulkMux bulk(*link); ContractRouter router(control);
    auto pump = [&] { link->poll(); bulk.pump(); relay_out(); };

    TypedStream st; SettingsRecord theirs;
    for (auto end = Clock::now() + std::chrono::seconds(20); Clock::now() < end && !bulk.take(BulkStreamKind::Settings, st);) pump();
    if (!decode_settings(st.bytes, theirs) || theirs.gpus.empty()) { say("no settings from the volunteer"); return 1; }
    const auto volunteer_slots = hosts_of(theirs, link->account());
    const auto local_slots = hosts_of(ours, 0);
    const bool gpu_ok = unmet_need(theirs.gpus[0], needs) == nullptr, cpu_ok = theirs.cpu_triple == ours.cpu_triple || theirs.has_clang;
    say("local market hosts: " + std::to_string(local_slots.size()) + "; the volunteer brings " + std::to_string(volunteer_slots.size()) +
        " (" + std::to_string(volunteer_slots.size() - 1) + " GPU, 1 CPU); its GPUs " + (gpu_ok ? "meet" : "do not meet") + " the shaders' needs; CPU " + theirs.cpu_triple);
    GlslangBuilds builds(ours.glslang_path, work / "server-builds");
    std::vector<uint8_t> vulkan; bool failed = false;
    if (gpu_ok) for (auto end = Clock::now() + std::chrono::seconds(30); Clock::now() < end && !builds.build(glsl, vulkan_target(theirs.gpus[0].api_version), vulkan, failed) && !failed;) pump();
    auto send_artifact = [&](const std::vector<uint8_t>& bytes) { auto d = sha256(bytes.data(), bytes.size()); bulk.send(BulkStreamKind::Artifact, get_le(d.data(), 8), bytes, std::vector<uint8_t>(d.begin(), d.end())); };
    if (gpu_ok && !vulkan.empty()) {
        send_artifact(glsl);
        const std::string target = vulkan_target(theirs.gpus[0].api_version);
        std::vector<uint8_t> aux(target.begin(), target.end()); aux.resize(16, 0);
        auto vd = sha256(vulkan.data(), vulkan.size()); aux.insert(aux.end(), glsl_d.begin(), glsl_d.end()); aux.insert(aux.end(), vd.begin(), vd.end());
        bulk.send(BulkStreamKind::Vulkan, 0, vulkan, aux);
    }
    if (cpu_ok) { send_artifact(llvm); send_artifact(fft); send_artifact(t4); }
    std::set<uint64_t> offered_defs, volunteer_ready;
    for (const auto& d : defs) {
        if (d.id == DefFabricA1 || d.id == DefFabricA2 || d.id == DefFabricB2) continue;   // pinned to local hosts
        if ((gpu_kind(d) && !gpu_ok) || (!gpu_kind(d) && !cpu_ok)) continue;
        bulk.send(BulkStreamKind::Definition, d.id, encode_definition(d)); offered_defs.insert(d.id);
    }
    for (auto end = Clock::now() + std::chrono::seconds(60); Clock::now() < end && volunteer_ready.size() < offered_defs.size();) {
        pump();
        ControlPacket p;
        while (control.take(ControlKind::Instruction, p)) { auto w = words_of(p); if (w.size() >= 3 && w[0] == 1 && w[2] == 0) volunteer_ready.insert(w[1]); }
    }
    say("the volunteer registered " + std::to_string(volunteer_ready.size()) + " of " + std::to_string(offered_defs.size()) + " definitions");

    // ---- the market ----
    std::map<uint64_t, WorkItem> items;
    std::vector<uint64_t> order;
    uint64_t next = 1;
    auto build_batch = [&](int run) {     // run 0 = A, run 1 = B: the same mix at new addresses
        const uint64_t off = uint64_t(run) * 1000000; const double ph = 0.05 * run;
        std::map<uint64_t, std::deque<WorkItem>> lanes;
        auto add = [&](uint64_t def, WorkItem w) { w.id = next++; w.definition = def; w.card = kCard; lanes[def].push_back(std::move(w)); };
        for (int k = 0; k < 24; ++k) {
            WorkItem w; const uint64_t base = off + 100000 + uint64_t(k) * 100;
            for (uint64_t i = 0; i < 27; ++i) w.puts.push_back({base + i, spring_column(i, ph + 0.01 * k)});
            w.inputs.push_back(tensor1<uint64_t>(NODUS_DTYPE_U64, {kCard, base, kLanes, 1})); w.outputs = {base + 24, base + 25}; add(DefStep, w);
        }
        for (int k = 0; k < 12; ++k) {
            WorkItem w; const uint64_t base = off + 200000 + uint64_t(k) * 100;
            for (uint64_t i = 0; i < 27; ++i) w.puts.push_back({base + i, spring_column(i, ph + 0.2 + 0.01 * k)});
            TensorBytes schedule; schedule.dtype = NODUS_DTYPE_U64; schedule.shape = {2, 7};
            const uint64_t rows[14] = {0, 1, kStepProgram, kCard, base, kLanes, 1, 1, 1, kPongProgram, kCard, base, kLanes, 2};
            schedule.bytes.resize(sizeof(rows)); std::memcpy(schedule.bytes.data(), rows, sizeof(rows));
            w.inputs.push_back(schedule); w.outputs = {base + 17, base + 23}; add(DefPipeline, w);
        }
        for (int k = 0; k < 8; ++k) {
            WorkItem w; const uint64_t base = off + 300000 + uint64_t(k) * 100;
            for (uint64_t i = 0; i < 27; ++i) w.puts.push_back({base + i, spring_column(i, ph + 0.5 + 0.01 * k)});
            w.inputs.push_back(tensor1<uint64_t>(NODUS_DTYPE_U64, {kCard, base, kLanes, 1})); w.outputs = {base + 24, base + 25}; add(DefCpuLaw, w);
        }
        for (int k = 0; k < 8; ++k) {
            WorkItem w; const uint64_t b = off + 400000 + uint64_t(k) * 10;
            std::vector<float> input(8); for (int i = 0; i < 8; ++i) input[size_t(i)] = float(std::sin(0.7 * i * (k + 1) + ph));
            w.puts.push_back({b, tensor1<float>(NODUS_DTYPE_F32, input)});
            for (uint64_t a = b + 1; a <= b + 3; ++a) w.puts.push_back({a, tensor1<float>(NODUS_DTYPE_F32, std::vector<float>(5, 0.f))});
            std::vector<uint64_t> wk(10, 0); wk[0] = 1; wk[1] = kCard; wk[2] = b; wk[4] = b + 1; wk[5] = b + 2; wk[6] = b + 3; wk[7] = 8; wk[8] = 1; wk[9] = 7001 + uint64_t(k);
            std::vector<uint64_t> cfg(40, 0); cfg[0] = 1; cfg[1] = 8; cfg[2] = 1; cfg[3] = 1; cfg[5] = 1; cfg[6] = 2; cfg[16] = 2; cfg[17] = 8; cfg[18] = 8; cfg[19] = 1; cfg[20] = 1; cfg[23] = 1;
            w.inputs = {tensor1<uint64_t>(NODUS_DTYPE_U64, wk), tensor1<uint64_t>(NODUS_DTYPE_U64, cfg)}; w.outputs = {b + 1, b + 2, b + 3}; add(DefFftfree, w);
        }
        auto bits = [](double v) { uint64_t u; std::memcpy(&u, &v, 8); return u; };
        for (int k = 0; k < 8; ++k) {
            WorkItem w; const uint64_t b = off + 450000 + uint64_t(k) * 10;
            w.puts.push_back({b, tensor1<double>(NODUS_DTYPE_F64, {450e-9 + 10e-9 * k, 850e-9})});
            for (uint64_t a = b + 1; a <= b + 4; ++a) { std::vector<float> f(256, 0.f); if (a == b + 1 || a == b + 3) f[8 * 16 + 8 + size_t(k)] = 1; w.puts.push_back({a, tensor1<float>(NODUS_DTYPE_F32, f)}); }
            w.inputs = {tensor1<uint64_t>(NODUS_DTYPE_U64, {kCard, b, 777, 42 + uint64_t(k), b + 1, b + 2, b + 3, b + 4}),
                        tensor1<uint64_t>(NODUS_DTYPE_U64, {1, 16, 16, 2, 1, bits(0.2e-6), bits(0.25e-6), 1})};
            w.outputs = {b + 1, b + 2, b + 3, b + 4}; add(DefT4, w);
        }
        for (int f = 0; f < 2; ++f) {
            const uint64_t def = run ? (f ? DefFabricB2 : DefFabricB1) : (f ? DefFabricA2 : DefFabricA1);
            TensorBytes cfg; cfg.dtype = NODUS_DTYPE_F64; cfg.shape = {8, 256, 38}; auto s = sheets(f + 2 * run);
            cfg.bytes.resize(s.size() * 8); std::memcpy(cfg.bytes.data(), s.data(), cfg.bytes.size());
            { WorkItem w; w.inputs = {cfg}; add(def, w); }
            for (int t = 0; t < 25; ++t) { WorkItem w; TensorBytes dt; dt.dtype = NODUS_DTYPE_F64; dt.bytes.resize(8); const double v = 1.0 / 60; std::memcpy(dt.bytes.data(), &v, 8); w.inputs = {dt}; add(def, w); }
        }
        size_t count = 0;
        for (bool more = true; more;) {             // interleave the kinds; each keeps its own order
            more = false;
            for (auto& [def, q] : lanes) if (!q.empty()) { order.push_back(q.front().id); const uint64_t id = q.front().id; items.emplace(id, std::move(q.front())); q.pop_front(); more = true; ++count; }
        }
        return count;
    };

    // Hosts and what they may run.
    struct HostInfo { HostSlot slot; bool volunteer = false; };
    std::map<uint64_t, HostInfo> host_info;
    for (const auto& s : local_slots) host_info[s.id] = {s, false};
    uint64_t local_gpu0 = 0, local_cpu = 0, volunteer_gpu0 = 0;
    for (const auto& s : local_slots) { if (s.gpu && s.slot == 0) local_gpu0 = s.id; if (!s.gpu) local_cpu = s.id; }
    for (const auto& s : volunteer_slots) if (s.gpu && s.slot == 0) volunteer_gpu0 = s.id;
    std::set<uint64_t> in_market;
    std::map<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>> timing;   // (host, def) -> (samples, total ns)
    std::map<uint64_t, uint64_t> accepted_at, submitted_at;
    auto pinned_host = [&](uint64_t def) -> uint64_t {
        if (def == DefFabricA1) return local_gpu0; if (def == DefFabricA2 || def == DefFabricB2) return local_cpu;
        if (def == DefFabricB1) return volunteer_gpu0; return 0;
    };
    nodus::contracts::MarketView view;
    view.compatible = [&](uint64_t host, const ContractNegotiator::Entry& e) {
        auto it = host_info.find(host); if (it == host_info.end() || !in_market.count(host)) return false;
        const auto& d = definition_of(e.work.revision);
        if (const uint64_t pin = pinned_host(d.id)) return pin == host;
        if (gpu_kind(d) != it->second.slot.gpu) return false;
        return !it->second.volunteer || volunteer_ready.count(d.id) != 0;
    };
    view.job = [&](uint64_t host, const ContractNegotiator::Entry& e) {
        nodus::contracts::MarketJob j;
        j.expires_at_ns = submitted_at[e.work.payload] + 5'000'000'000ull;
        auto t = timing.find({host, e.work.revision}); if (t != timing.end() && t->second.first) j.recent_completion_ns = t->second.second / t->second.first;
        const auto& d = definition_of(e.work.revision);
        if (d.stateful) { j.total_bytes = 1; j.resident_bytes = pinned_host(d.id) == host ? 1 : 0; }   // the program's state lives on its host
        return j;
    };
    view.timing = [&](uint64_t host, const ContractNegotiator::Entry& e) {
        auto t = timing.find({host, e.work.revision});
        return t == timing.end() || !t->second.first ? nodus::contracts::MarketTiming{} : nodus::contracts::MarketTiming{t->second.first, t->second.second / t->second.first};
    };
    for (const auto& [id, _] : host_info) view.hosts.push_back(id);
    for (const auto& s : volunteer_slots) view.hosts.push_back(s.id);
    ContractNegotiator market(512, nodus::contracts::market_offer(view), 1000000);

    // Local hosts: prepare on this thread, the row on the host's own thread.
    std::map<uint64_t, Execution> local_exec;
    std::map<uint64_t, std::optional<Loan>> local_loans;
    std::map<uint64_t, WorkOutputs> outputs;      // by work id: local results, and volunteer outputs as they arrive
    for (const auto& s : local_slots) {
        local_exec[s.id].lane = std::make_unique<LibraryRunner::Lane>(s.gpu ? s.slot : 0);
        nodus::contracts::Peer p; p.id = s.id;
        p.exchange = [&, id = s.id](const Loan* offer) {
            nodus::contracts::Exchange e{};
            auto& ex = local_exec[id]; auto& loan = local_loans[id];
            if (loan && ex.running && ex.running->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                outputs[loan->work.payload] = ex.running->get(); ex.running.reset();
                e.has_result = true; e.result = {*loan, outputs[loan->work.payload].status}; loan.reset();
            }
            if (offer && !loan) {
                const WorkItem& w = items.at(offer->work.payload);
                if (!server.prepare(w, *ex.lane)) return e;               // refused: the offer stays unaccepted
                loan = *offer; e.accepted_generation = offer->generation; accepted_at[offer->work.payload] = now_ns();
                ex.running = std::async(std::launch::async, [&server, &ex, w] { return server.execute(w, *ex.lane); });
            }
            return e;
        };
        market.add_host(p); in_market.insert(s.id);
    }
    // Volunteer hosts join for run B.
    std::vector<std::unique_ptr<LinkHostPeer>> volunteer_peers;
    for (const auto& s : volunteer_slots) {
        host_info[s.id] = {s, true};
        volunteer_peers.push_back(std::make_unique<LinkHostPeer>(router, s.id, [&](const Loan& loan) {
            accepted_at[loan.work.payload] = now_ns();
            bulk.send(BulkStreamKind::WorkItem, loan.work.payload, encode_work(items.at(loan.work.payload)));
        }));
    }
    // Re-check lanes: the volunteer's device slot, on this machine (byte comparison is only meaningful on the same device).
    std::map<uint32_t, std::unique_ptr<LibraryRunner::Lane>> recheck_lanes;

    size_t submitted = 0, routed = 0, mismatched = 0, rechecked = 0, total = 0, unmatched_devices = 0;
    int run = 0;
    std::map<std::string, std::array<int, 2>> by_kind[2];   // per run: [local, volunteer]
    std::map<uint64_t, int> by_host[2];
    Producer engine; engine.id = 900;
    engine.request = [&](Contract& c) {
        if (submitted >= total) return false;
        const auto& w = items.at(order[submitted]); c = {w.id, 900, w.definition, w.id}; return true;
    };
    engine.admitted = [&](uint64_t id) { submitted_at[id] = now_ns(); ++submitted; };
    engine.route = [&](const nodus::contracts::Result& r) {
        const uint64_t id = r.loan.work.payload;
        auto it = outputs.find(id);
        if (it == outputs.end()) return false;               // its outputs are still on the bulk pair: not committed yet
        const auto& d = definition_of(r.loan.work.revision);
        const auto info = host_info.find(r.loan.host);
        const bool from_volunteer = info != host_info.end() && info->second.volunteer;
        if (it->second.status != 0) {
            say("job " + std::to_string(id) + " (" + kind_name(d.kind) + ") failed on host " + std::to_string(r.loan.host) + ": status " +
                std::to_string(it->second.status) + ", row output " + std::to_string(it->second.row_output.bytes.size()) + " bytes");
            ++mismatched;
        }
        if (from_volunteer) {
            // the server's own device with the volunteer device's UUID: the same device, so bytes must agree
            std::optional<uint32_t> same;
            if (!info->second.slot.gpu) same = 0;
            else for (const auto& l : local_slots) if (l.gpu && l.uuid == info->second.slot.uuid) same = l.slot;
            if (!same) { ++unmatched_devices; ++by_kind[run][kind_name(d.kind)][1]; ++by_host[run][r.loan.host]; ++routed; return true; }
            auto& lane = recheck_lanes[*same];
            if (!lane) lane = std::make_unique<LibraryRunner::Lane>(*same);
            const WorkItem& w = items.at(id);
            WorkOutputs mine = server.prepare(w, *lane) ? server.execute(w, *lane) : WorkOutputs{};
            ++rechecked;
            if (!(mine == it->second)) { ++mismatched; say("job " + std::to_string(id) + " (" + kind_name(d.kind) + ") DIFFERS from the server's run"); }
        }
        auto& t = timing[{r.loan.host, d.id}]; t.first++; t.second += now_ns() - accepted_at[id];
        ++by_kind[run][kind_name(d.kind)][from_volunteer ? 1 : 0]; ++by_host[run][r.loan.host]; ++routed; return true;
    };
    market.add_engine(engine);

    double seconds[2] = {0, 0};
    for (run = 0; run < 2; ++run) {
        if (run == 1) for (size_t i = 0; i < volunteer_slots.size(); ++i) { market.add_host(volunteer_peers[i]->peer()); in_market.insert(volunteer_slots[i].id); }
        const size_t before = routed;
        total += build_batch(run);
        const auto t0 = Clock::now();
        for (auto end = t0 + std::chrono::seconds(300); Clock::now() < end && routed < total;) {
            pump();
            TypedStream o;
            while (bulk.take(BulkStreamKind::Outputs, o)) { WorkOutputs x; if (decode_outputs(o.bytes, x)) outputs[x.id] = std::move(x); }
            market.worker_turn(); market.market_turn();
        }
        seconds[run] = std::chrono::duration<double>(Clock::now() - t0).count();
        say(std::string(run ? "run B (local + volunteer hosts)" : "run A (local hosts alone)") + ": " + std::to_string(routed - before) + " jobs in " +
            std::to_string(seconds[run]).substr(0, 6) + " s");
    }
    send_instruction(*link, {2});
    for (auto end = Clock::now() + std::chrono::seconds(20); Clock::now() < end && !child.finished();) pump();
    relay_out(); { std::string rest; child.take(rest); if (!rest.empty()) std::printf("%s", rest.c_str()); }

    for (const auto& [k, n] : by_kind[1]) say("  run B " + k + ": local " + std::to_string(n[0]) + ", volunteer " + std::to_string(n[1]));
    for (const auto& [h, n] : by_host[1]) { auto i = host_info.find(h); say("  run B host " + (i->second.volunteer ? std::string("volunteer ") : std::string("local ")) + i->second.slot.name + ": " + std::to_string(n) + " jobs"); }
    say("volunteer results re-run on the server on the same device (by UUID): " + std::to_string(rechecked) + ", differing: " + std::to_string(mismatched) +
        (unmatched_devices ? "; NOT re-checked (no server device with that UUID): " + std::to_string(unmatched_devices) : std::string()));
    std::set<std::string> kinds; for (const auto& d : defs) if (offered_defs.count(d.id)) kinds.insert(kind_name(d.kind));
    std::string unserved; for (const auto& k : kinds) if (by_kind[1][k][1] == 0) unserved += " " + k + ";";
    if (!unserved.empty()) say("the volunteer was given none of:" + unserved);
    say("throughput: run A " + std::to_string(seconds[0]).substr(0, 6) + " s, run B " + std::to_string(seconds[1]).substr(0, 6) + " s for the same mix" +
        " (on this machine the volunteer shares the same GPUs and CPU)");
    const bool ok = routed == total && mismatched == 0 && rechecked > 0 && unserved.empty() && child.finished() && child.exit_code() == 0;
    say(ok ? "PASSED: market placement gave the volunteer every job kind alongside the local hosts; every volunteer result matched" : "FAILED");
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
