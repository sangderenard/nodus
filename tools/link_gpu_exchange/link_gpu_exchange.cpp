// link_gpu_exchange: stage 5 of the Nodus link (research case 98d49cb8520e44d092c5d134c37c3c32), the owner's test case:
// two Nodus host processes link and share GPU work in both directions (on this server they share the same GPU).
//
// Each process is a Nodus host: the tensor state library owned by KernelLibrarySession, the fabric dispatch rows
// nodus.fabric.dispatch.vulkan.<device> registered over it, and the <id>.control / <id>.bulk link rows. Rows are driven
// through execute_stack, as the dispatch trial and KernelLibrarySession hosts do; nothing blocks.
//
// The shader travels as an OFFER: its GLSL source and the Vulkan SPIR-V glslang built from that GLSL (Turing's
// tools/build_spring_network_shader.py: spring_network_step.comp and .direct-glsl.spv). The receiver compiles the offered
// GLSL with its own glslang, requires the same Vulkan bytes, and requires them equal to the shader its dispatch rows hold
// (one shader per session), then ACCEPTs. Only then does work flow: PUT packets on control with the 27 columns on bulk,
// a DISPATCH packet, the RECEIPT (the row's event) on control and columns 24/25 (next position, velocity) on bulk. The
// requester runs the same dispatch in its own library and requires the returned columns byte-identical.
//
//   link_gpu_exchange run <shader-dir> <work-dir> <glslangValidator>
// starts the host side (A, a LinkListener) and the peer (B, a LinkDialer) as a child process of the same program.
#include "nodus_link_rows.h"
#include "kernel_library_session.h"
#include "exec_pipe.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

using namespace nodus::link;
using Clock = std::chrono::steady_clock;
using Row = std::unique_ptr<ITool, std::function<void(ITool*)>>;
namespace fs = std::filesystem;

namespace {
constexpr uint64_t kCard = 17, kLanes = 65536, kColumns = 27, kDevice = 0;
enum Op : uint64_t { Offer = 1, Accept = 2, Put = 3, Dispatch = 4, Receipt = 5, Result = 6, Done = 7 };
std::string g_who = "?";

void say(const char* fmt, const std::string& a = {}) { std::printf("[%s] ", g_who.c_str()); std::printf(fmt, a.c_str()); std::printf("\n"); std::fflush(stdout); }

std::vector<uint8_t> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}
std::string hex(const uint8_t* p, size_t n) { static const char* d = "0123456789abcdef"; std::string s; for (size_t i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; } return s; }
std::vector<uint8_t> unhex(const std::string& s) { std::vector<uint8_t> v(s.size() / 2); for (size_t i = 0; i < v.size(); ++i) v[i] = uint8_t(std::stoul(s.substr(2 * i, 2), nullptr, 16)); return v; }

uint64_t make_tensor(int32_t dtype, uint64_t count, const void* data, size_t bytes) {
    const uint64_t shape[1] = {count};
    uint64_t t = nodus_tensor_create(dtype, 1, shape);
    if (t && bytes) nodus_tensor_write(t, 0, data, bytes);
    return t;
}
std::vector<uint8_t> tensor_bytes(uint64_t t) {
    NodusTensorDesc d{}; if (nodus_tensor_describe(t, &d) != NODUS_OK) return {};
    std::vector<uint8_t> v(size_t(d.total_bytes)); nodus_tensor_read(t, 0, v.data(), v.size()); return v;
}
uint64_t packet(uint64_t op, std::initializer_list<uint64_t> words) {
    ControlPacket p; p.kind = uint16_t(op == Receipt ? ControlKind::Receipt : ControlKind::Instruction);
    p.length = uint32_t(8 * (1 + words.size()));
    put_le(p.payload.data(), op, 8); size_t i = 1; for (uint64_t w : words) put_le(p.payload.data() + 8 * i++, w, 8);
    uint8_t b[kControlPacketBytes]; encode_control(p, b);
    return make_tensor(NODUS_DTYPE_U8, kControlPacketBytes, b, sizeof(b));
}
std::vector<uint64_t> words_of(uint64_t packet_tensor) {
    auto b = tensor_bytes(packet_tensor); ControlPacket p;
    if (!decode_control(b.data(), b.size(), p)) return {};
    std::vector<uint64_t> w(p.length / 8); for (size_t i = 0; i < w.size(); ++i) w[i] = get_le(p.payload.data() + 8 * i, 8);
    return w;
}
// Column i of a job: the dispatch trial's fills (tests/tensor_state_library_tool_test.cpp) with a per-lane, per-seed ramp
// on the positions so the two directions carry different work; column 26 is the u32 lane count.
uint64_t column(uint64_t i, double seed) {
    if (i == 26) { const uint32_t n = uint32_t(kLanes); return make_tensor(NODUS_DTYPE_U32, 1, &n, sizeof(n)); }
    std::vector<double> v(kLanes, 0.0);
    for (uint64_t l = 0; l < kLanes; ++l) {
        double x = 0;
        if (i == 0 || i == 17) x = 1.0 + seed + double(l) * 1e-6;
        if (i == 3) x = .05; if (i == 4) x = .001; if (i == 9) x = -9.8; if (i == 10 || i == 15) x = 1.0; if (i == 22) x = 10.0;
        v[l] = x;
    }
    return make_tensor(NODUS_DTYPE_F64, kLanes, v.data(), v.size() * sizeof(double));
}

// One scheduler visit of a row: an optional input on the stack; returns the output it pushed (0 = none) and whether the
// input was taken.
uint64_t visit(ITool& row, RawStackFrame& f, uint64_t input, bool& taken) {
    taken = false;
    if (input) raw_stack_push_abstract_tensor(f, {input});
    ToolStackContext c{}; c.stack.raw = &f; row.execute_stack(c);
    nodus::tensors::AbstractTensorHandle out{};
    if (!raw_stack_pop_abstract_tensor(f, out)) { taken = input != 0; return 0; }
    if (input && out.id == input) return 0;                    // refused and restored
    taken = input != 0;
    return out.id;
}

struct Host {
    nodus::gpu::KernelLibrarySession session{tool_registry_global()};
    Row dispatch, control_send, control_receive, bulk_send, bulk_receive;   // the link's four rows, one duty each
    RawStackFrame frame{};
    std::vector<uint8_t> glsl, vulkan;                         // this process's shader: GLSL and the SPIR-V glslang made of it
    fs::path work, glslang;
    std::deque<uint64_t> control_out, bulk_out;                // tensors waiting for the link rows to take them
    // inbound bulk slots, in the order the peer's control packets announced them
    struct Slot { uint64_t op = 0, address = 0; };
    // The two sockets are not ordered against each other, so a bulk tensor may come before the control packet that
    // announces it; each stream is ordered, so arrivals are paired with announcements in order.
    std::deque<Slot> slots; std::deque<uint64_t> early_bulk; std::deque<std::vector<uint64_t>> instructions;
    std::vector<uint8_t> offered_glsl, offered_vulkan; int offer_parts = 0; bool offer_seen = false;
    nodus::exec::Child compile; bool compiling = false; bool accepted_peer = false, peer_accepted_us = false;
    bool requested = false, served = false, peer_done = false, sent_done = false, passed = false, failed = false, compared = false;
    std::vector<std::vector<uint8_t>> results; std::vector<uint64_t> result_addresses; std::array<uint64_t, 6> remote_event{};
    double seed = 0; uint64_t remote_base = 0, local_base = 0;
    double sent_at = 0; Clock::time_point t0 = Clock::now();
    // Roles: a requester offers its shader, sends a job and re-checks the result with its own dispatch; a worker
    // accepts an offer and serves jobs. The two-process test runs both in each process; a volunteer is a worker only.
    bool requester = true, worker = true;
    std::string link_id;

    std::array<uint64_t, 32> library_call(std::array<uint64_t, 12> request, uint64_t payload) {
        const uint64_t command = make_tensor(NODUS_DTYPE_U64, 12, request.data(), sizeof(request));
        bool own_payload = !payload; if (own_payload) payload = make_tensor(NODUS_DTYPE_U64, 1, nullptr, 0);
        raw_stack_push_abstract_tensor(frame, {command}); raw_stack_push_abstract_tensor(frame, {payload});
        ToolStackContext c{}; c.stack.raw = &frame; session.library_row().execute_stack(c);
        std::array<uint64_t, 32> r{}; r[0] = UINT64_MAX;
        nodus::tensors::AbstractTensorHandle out{};
        if (c.succeeded && raw_stack_pop_abstract_tensor(frame, out)) { nodus_tensor_read(out.id, 0, r.data(), sizeof(r)); nodus_tensor_destroy(out.id); }
        nodus_tensor_destroy(command); if (own_payload) nodus_tensor_destroy(payload);
        return r;
    }
    bool put(uint64_t address, uint64_t tensor) {
        std::array<uint64_t, 12> q{}; q[0] = 2; q[1] = kCard; q[3] = address;
        return library_call(q, tensor)[0] == 0;
    }
    std::array<uint64_t, 6> run_dispatch(uint64_t base, uint64_t generation) {
        const uint64_t token[4] = {kCard, base, kLanes, generation};
        uint64_t work = make_tensor(NODUS_DTYPE_U64, 4, token, sizeof(token));
        raw_stack_push_abstract_tensor(frame, {work});
        ToolStackContext c{}; c.stack.raw = &frame; dispatch->execute_stack(c);
        std::array<uint64_t, 6> e{}; e[0] = UINT64_MAX;
        nodus::tensors::AbstractTensorHandle out{};
        if (c.succeeded && raw_stack_pop_abstract_tensor(frame, out)) { nodus_tensor_read(out.id, 0, e.data(), sizeof(e)); nodus_tensor_destroy(out.id); }
        nodus_tensor_destroy(work);
        return e;
    }
    std::vector<uint8_t> read_back(uint64_t address) {
        auto loan = session.library().borrow_cpu(kCard, address, false);
        if (!loan || loan.bytes() < kLanes * sizeof(double)) return {};
        const auto* p = static_cast<const uint8_t*>(loan.data());
        return {p, p + kLanes * sizeof(double)};
    }

    // The dispatch rows hold one shader per session: registered from this process's own files, or (a volunteer with
    // no shader of its own) from the first offer it accepts.
    bool register_shader(const std::vector<uint8_t>& spirv) {
        if (spirv.empty() || spirv.size() % 4) return false;
        std::vector<uint32_t> words(spirv.size() / 4); std::memcpy(words.data(), spirv.data(), spirv.size());
        if (!session.register_fabric_dispatch(words, 256)) { say("fabric dispatch registration refused"); return false; }
        dispatch = tool_registry_global().create("nodus.fabric.dispatch.vulkan." + std::to_string(kDevice));
        if (!dispatch) { say("dispatch row missing"); return false; }
        vulkan = spirv;
        return true;
    }
    bool setup(const fs::path& shader_dir, const std::string& id) {
        link_id = id;
        if (!raw_stack_init_frame(frame, 1 << 16) || !session) { say("library refused"); return false; }
        if (!shader_dir.empty()) {
            glsl = read_file(shader_dir / "spring_network_step.comp");
            auto spirv = read_file(shader_dir / "spring_network_step.direct-glsl.spv");
            if (glsl.empty() || spirv.empty()) { say("shader files missing in %s", shader_dir.string()); return false; }
            if (!register_shader(spirv)) return false;
        }
        auto& reg = tool_registry_global();
        control_send = reg.create(link_id + ".control.send"); control_receive = reg.create(link_id + ".control.receive");
        bulk_send = reg.create(link_id + ".bulk.send"); bulk_receive = reg.create(link_id + ".bulk.receive");
        if (!control_send || !control_receive || !bulk_send || !bulk_receive) { say("rows missing"); return false; }
        std::array<uint64_t, 12> card{}; card[0] = 1; card[1] = kCard; card[2] = 901; card[4] = 512ull << 20;
        card[5] = card[6] = card[7] = 128ull << 20;
        auto r = library_call(card, 0);
        if (r[0] != 0) { say("card refused"); return false; }
        char msg[200]; std::snprintf(msg, sizeof(msg), "host up: %zu GPU bank(s), %s", session.library().device_count(),
            vulkan.empty() ? "no shader of its own (takes the server's offer)" : ("shader GLSL " + std::to_string(glsl.size()) +
            " B, Vulkan " + std::to_string(vulkan.size()) + " B").c_str()); say("%s", msg);
        return true;
    }

    void send_offer() {
        auto hg = sha256(glsl.data(), glsl.size()), hv = sha256(vulkan.data(), vulkan.size());
        // the full SHA-256 of both products: a worker with no glslang accepts the Vulkan bytes on this digest alone
        control_out.push_back(packet(Offer, {glsl.size(), vulkan.size(), get_le(hg.data(), 8), get_le(hg.data() + 8, 8),
            get_le(hg.data() + 16, 8), get_le(hg.data() + 24, 8), get_le(hv.data(), 8), get_le(hv.data() + 8, 8),
            get_le(hv.data() + 16, 8), get_le(hv.data() + 24, 8)}));
        bulk_out.push_back(make_tensor(NODUS_DTYPE_U8, glsl.size(), glsl.data(), glsl.size()));
        bulk_out.push_back(make_tensor(NODUS_DTYPE_U8, vulkan.size(), vulkan.data(), vulkan.size()));
    }
    void send_job() {
        // the local reference first: the same columns through this process's own library and dispatch row
        for (uint64_t i = 0; i < kColumns; ++i) { uint64_t t = column(i, seed); if (!put(local_base + i, t)) { say("local put refused"); failed = true; } nodus_tensor_destroy(t); }
        for (uint64_t i = 0; i < kColumns; ++i) {
            control_out.push_back(packet(Put, {remote_base + i}));
            bulk_out.push_back(column(i, seed));
        }
        control_out.push_back(packet(Dispatch, {remote_base, kLanes, kDevice}));
        sent_at = std::chrono::duration<double>(Clock::now() - t0).count();
        requested = true;
        say("job sent: 27 columns (13.8 MB) + DISPATCH");
    }

    void on_control(uint64_t t) {
        auto w = words_of(t); nodus_tensor_destroy(t);
        if (w.empty()) { say("bad control packet"); failed = true; return; }
        switch (w[0]) {
        case Offer: slots.push_back({Offer, 0}); slots.push_back({Offer, 1}); instructions.push_back(w); break;
        case Put: slots.push_back({Put, w[1]}); break;
        case Dispatch: instructions.push_back(w); break;
        case Accept:
            peer_accepted_us = w.size() > 3 && w[1] == 0;
            say(peer_accepted_us ? "peer accepted the offer" : "peer REFUSED the offer"); if (!peer_accepted_us) failed = true; break;
        case Receipt: for (size_t i = 0; i < 6 && i + 2 < w.size(); ++i) remote_event[i] = w[2 + i]; if (w[1]) { say("remote dispatch failed"); failed = true; } break;
        case Result: slots.push_back({Result, w[1]}); break;
        case Done: peer_done = true; break;
        default: say("unknown op"); failed = true;
        }
    }
    void on_bulk(uint64_t t) { early_bulk.push_back(t); pair_bulk(); }
    void pair_bulk() {
        while (!slots.empty() && !early_bulk.empty()) {
            uint64_t t = early_bulk.front(); early_bulk.pop_front();
            Slot s = slots.front(); slots.pop_front();
            consume(s, t);
        }
    }
    void consume(const Slot& s, uint64_t t) {
        if (s.op == Offer) { (s.address ? offered_vulkan : offered_glsl) = tensor_bytes(t); ++offer_parts; }
        else if (s.op == Put) { if (!put(s.address, t)) { say("put refused"); failed = true; } }
        else if (s.op == Result) { results.push_back(tensor_bytes(t)); result_addresses.push_back(s.address); }
        nodus_tensor_destroy(t);
    }
    // Instructions run in order; each waits for the bulk slots announced before it.
    void work_instructions() {
        while (!instructions.empty()) {
            auto& w = instructions.front();
            if (w[0] == Offer) {
                if (offer_parts < 2) return;
                auto digest_is = [&](const std::vector<uint8_t>& bytes, size_t at) {
                    if (w.size() < at + 4) return false;
                    auto d = sha256(bytes.data(), bytes.size());
                    for (size_t i = 0; i < 4; ++i) if (get_le(d.data() + 8 * i, 8) != w[at + i]) return false;
                    return true;
                };
                const bool digests = w.size() >= 11 && w[1] == offered_glsl.size() && w[2] == offered_vulkan.size() &&
                                     digest_is(offered_glsl, 3) && digest_is(offered_vulkan, 7);
                if (!digests) {
                    control_out.push_back(packet(Accept, {1u, 0u, 0u}));
                    say("offer REFUSED: the products do not match the offered SHA-256"); failed = true;
                    instructions.pop_front(); continue;
                }
                if (glslang.empty()) {          // a volunteer: no compiler here, the digest is the check
                    const bool usable = vulkan.empty() ? register_shader(offered_vulkan) : offered_vulkan == vulkan;
                    control_out.push_back(packet(Accept, {usable ? 0u : 1u, 0u, usable ? 1u : 0u}));
                    say(usable ? "offer accepted on its SHA-256 (no glslang here); dispatch rows hold the offered Vulkan"
                               : "offer REFUSED: this session already holds a different shader");
                    accepted_peer = usable; if (!usable) failed = true;
                    instructions.pop_front(); continue;
                }
                if (!compiling) {
                    std::ofstream(work / "offered.comp", std::ios::binary).write(reinterpret_cast<const char*>(offered_glsl.data()), std::streamsize(offered_glsl.size()));
                    std::error_code ec; fs::remove(work / "offered.spv", ec);
                    nodus::exec::Program p; p.path = glslang.string(); p.working_dir = work.string();
                    p.arguments = {"-V", "--target-env", "vulkan1.1", "-S", "comp", "offered.comp", "-o", "offered.spv"};
                    std::string why;
                    if (!compile.start(p, why)) { say("glslang refused: %s", why); failed = true; return; }
                    compiling = true; return;
                }
                if (!compile.finished()) { std::string sink; compile.take(sink); return; }
                std::string out; compile.take(out); compiling = false;
                auto own = read_file(work / "offered.spv");
                const bool glsl_reproduces = compile.exit_code() == 0 && own == offered_vulkan;
                const bool same_as_rows = offered_vulkan == vulkan;
                control_out.push_back(packet(Accept, {glsl_reproduces && same_as_rows ? 0u : 1u, glsl_reproduces, same_as_rows}));
                say(glsl_reproduces ? "offer: our glslang built the offered GLSL into the offered Vulkan bytes" : "offer: our glslang did NOT reproduce the offered Vulkan");
                accepted_peer = glsl_reproduces && same_as_rows; if (!accepted_peer) failed = true;
                instructions.pop_front(); continue;
            }
            if (w[0] == Dispatch) {
                for (const auto& s : slots) if (s.op == Put) return;   // columns still in flight
                auto e = run_dispatch(w[1], 7);
                control_out.push_back(packet(Receipt, {e[0] == 0 ? 0u : 1u, e[0], e[1], e[2], e[3], e[4], e[5]}));
                for (uint64_t col : {24u, 25u}) {
                    auto bytes = read_back(w[1] + col);
                    control_out.push_back(packet(Result, {w[1] + col}));
                    bulk_out.push_back(make_tensor(NODUS_DTYPE_F64, kLanes, bytes.data(), bytes.size()));
                }
                char m[160]; std::snprintf(m, sizeof(m), "served the peer's dispatch on device %llu: status %llu, %.3f ms",
                    (unsigned long long)e[1], (unsigned long long)e[0], double(e[3]) / 1e6); say("%s", m);
                served = true; instructions.pop_front(); continue;
            }
            instructions.pop_front();
        }
    }
    void check_results() {
        if (compared || results.size() < 2) return;
        compared = true;
        auto e = run_dispatch(local_base, 8);
        bool same = e[0] == 0;
        for (size_t k = 0; k < 2; ++k) same = same && results[k] == read_back(local_base + (result_addresses[k] - remote_base));
        double rtt = std::chrono::duration<double>(Clock::now() - t0).count() - sent_at;
        char m[220]; std::snprintf(m, sizeof(m), "remote result columns 24,25 %s the local dispatch (%zu B each); remote dispatch %.3f ms, round trip %.3f s",
            same ? "byte-identical to" : "DIFFER from", results[0].size(), double(remote_event[3]) / 1e6, rtt); say("%s", m);
        // what came back is the stepped state, not the input: lane 0 and the last lane of next position / velocity
        auto at = [&](size_t k, uint64_t lane) { double v; std::memcpy(&v, results[k].data() + lane * 8, 8); return v; };
        const double input0 = 1.0 + seed, input_last = 1.0 + seed + double(kLanes - 1) * 1e-6;
        bool stepped = at(0, 0) != input0 || at(1, 0) != 0.0;
        std::snprintf(m, sizeof(m), "position[0] %.17g -> %.17g, position[last] %.17g -> %.17g, velocity[0] %.17g",
            input0, at(0, 0), input_last, at(0, kLanes - 1), at(1, 0)); say("%s", m);
        if (!stepped) say("the returned state equals the input: the dispatch did not step");
        passed = same && stepped; if (!passed) failed = true;
    }
    // One loop turn: give the rows their inputs, collect their outputs, advance the work.
    void turn() {
        bool taken = false;
        uint64_t in = control_out.empty() ? 0 : control_out.front();
        bool unused = false;
        visit(*control_send, frame, in, taken);
        if (taken) { nodus_tensor_destroy(control_out.front()); control_out.pop_front(); }
        for (uint64_t got; (got = visit(*control_receive, frame, 0, unused));) { on_control(got); pair_bulk(); }
        in = bulk_out.empty() ? 0 : bulk_out.front();
        visit(*bulk_send, frame, in, taken);
        if (taken) { nodus_tensor_destroy(bulk_out.front()); bulk_out.pop_front(); }
        for (uint64_t got; (got = visit(*bulk_receive, frame, 0, unused));) on_bulk(got);
        work_instructions();
        if (requester && peer_accepted_us && !requested) send_job();
        check_results();
        if ((!requester || passed) && (!worker || served) && !sent_done) { control_out.push_back(packet(Done, {1})); sent_done = true; }
    }
    bool finished() const {
        return failed || ((!requester || passed) && (!worker || served) && sent_done && peer_done && control_out.empty());
    }
    ~Host() {
        for (auto t : early_bulk) nodus_tensor_destroy(t);
        for (auto t : control_out) nodus_tensor_destroy(t); for (auto t : bulk_out) nodus_tensor_destroy(t);
        dispatch.reset(); control_send.reset(); control_receive.reset(); bulk_send.reset(); bulk_receive.reset();
        raw_stack_free_mask(frame); if (frame.backend) gp_mem_backend_release(frame.backend);
    }
};

int serve(Host& h, const std::function<bool(const std::string&)>& linked, nodus::exec::Child* peer) {
    auto end = Clock::now() + std::chrono::seconds(180);
    bool offered = false; std::string relay;
    while (Clock::now() < end && !h.finished()) {
        h.turn();
        if (!offered && linked(h.link_id)) {
            offered = true;
            if (h.requester) { h.send_offer(); say("linked; offer sent (GLSL + Vulkan)"); } else say("linked");
        }
        if (peer) { peer->take(relay); size_t nl; while ((nl = relay.find('\n')) != std::string::npos) { std::printf("%s\n", relay.substr(0, nl).c_str()); relay.erase(0, nl + 1); } std::fflush(stdout); }
    }
    // let the last packets leave before the rows go
    auto drain = Clock::now() + std::chrono::milliseconds(500);
    while (Clock::now() < drain) { h.turn(); if (peer) { peer->take(relay); } }
    if (peer) {
        auto wait = Clock::now() + std::chrono::seconds(30);
        while (!peer->finished() && Clock::now() < wait) { std::string s; peer->take(s); relay += s; }
        std::string s; peer->take(s); relay += s;
        if (!relay.empty()) std::printf("%s", relay.c_str());
        if (!peer->finished() || peer->exit_code() != 0) { say("peer process did not pass"); return 1; }
    }
    if (!h.finished() || h.failed) { say("FAILED (passed=%s)", h.passed ? "1" : "0"); return 1; }
    say(h.requester && h.worker ? "PASSED: served the peer's GPU work and received ours back identical"
        : h.requester ? "PASSED: the volunteer's GPU result matched our own dispatch byte for byte" : "PASSED: served the server's GPU work");
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    auto linked_report = [](const std::string& id) {
        EndpointReport r;
        if (!link_endpoint_report(id, &r)) return false;
        if (r.state == EndpointState::Failed && !r.reason.empty()) { say("dial failed: %s", r.reason); std::exit(1); }
        return r.state == EndpointState::Linked;
    };
    if (mode == "run" && argc == 5) {
        g_who = "A";
        const fs::path shader_dir = argv[2], work = argv[3]; fs::create_directories(work / "A"); fs::create_directories(work / "B");
        Key key{}; random_bytes(key.data(), key.size());
        LinkHostConfig hc; hc.id = "exchange.a"; hc.prefix = "lab"; hc.key = key;
        uint16_t port = 0; std::string err;
        if (!register_link_host(tool_registry_global(), hc, &port, &err)) { say("listener refused: %s", err); return 1; }
        int rc = 1;
        {   // the host's rows and library go before the endpoint is unregistered
            Host h; h.work = work / "A"; h.glslang = argv[4]; h.seed = 0.25; h.remote_base = 1000; h.local_base = 5000;
            if (!h.setup(shader_dir, "exchange.a")) return 1;
            const auto id = new_link_id(); const uint64_t now = uint64_t(std::time(nullptr));
            auto c = sign_ticket(make_ticket(Channel::Control, Role::Dialer, id, "lab", 42, now, 120), key);
            auto b = sign_ticket(make_ticket(Channel::Bulk, Role::Dialer, id, "lab", 42, now, 120), key);
            nodus::exec::Program p; p.path = fs::absolute(argv[0]).string(); p.working_dir = (work / "B").string();
            p.arguments = {"peer", std::to_string(port), hex(c.data(), c.size()), hex(b.data(), b.size()), shader_dir.string(), (work / "B").string(), argv[4]};
            nodus::exec::Child peer; std::string why;
            if (!peer.start(p, why)) { say("peer process refused: %s", why); return 1; }
            say("listening on 127.0.0.1:%s; peer process started", std::to_string(port));
            rc = serve(h, linked_report, &peer);
        }
        unregister_link(tool_registry_global(), "exchange.a");
        return rc;
    }
    if (mode == "peer" && argc == 8) {
        g_who = "B";
        LinkDialConfig dc; dc.id = "exchange.b"; dc.target.port = uint16_t(std::stoi(argv[2])); dc.target.prefix = "lab";
        auto c = unhex(argv[3]), b = unhex(argv[4]);
        if (c.size() != kTicketBytes || b.size() != kTicketBytes) return 2;
        std::memcpy(dc.control_ticket.data(), c.data(), kTicketBytes); std::memcpy(dc.bulk_ticket.data(), b.data(), kTicketBytes);
        std::string err;
        if (!register_link_dialer(tool_registry_global(), dc, &err)) { say("dialer refused: %s", err); return 1; }
        Host h; h.work = argv[6]; h.glslang = argv[7]; h.seed = 0.75; h.remote_base = 2000; h.local_base = 6000;
        if (!h.setup(argv[5], "exchange.b")) return 1;
        return serve(h, linked_report, nullptr);
    }
    if (mode == "volunteer") {
        // The volunteer: no shader, no compiler, no listening port. It dials out (wss://, or ws:// with --ws), takes the
        // server's offer on its digest, and serves the server's jobs on its GPU.
        g_who = "volunteer";
        std::string host = "127.0.0.1", prefix = "lab", pin_hex, control_hex, bulk_hex; int port = 443; bool ws = false;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i]; auto value = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
            if (a == "--host") host = value(); else if (a == "--port") port = std::stoi(value()); else if (a == "--prefix") prefix = value();
            else if (a == "--pin") pin_hex = value(); else if (a == "--ws") ws = true;
            else if (a == "--control-ticket") control_hex = value(); else if (a == "--bulk-ticket") bulk_hex = value();
            else { say("unknown argument %s", a); return 2; }
        }
        auto c = unhex(control_hex), b = unhex(bulk_hex), pin = unhex(pin_hex);
        if (c.size() != kTicketBytes || b.size() != kTicketBytes || (!pin_hex.empty() && pin.size() != 32)) {
            say("usage: volunteer --host H --port P --prefix X [--pin SHA256HEX] [--ws] --control-ticket HEX --bulk-ticket HEX"); return 2;
        }
        LinkDialConfig dc; dc.id = "volunteer"; dc.wss = true;
        dc.wss_target.host = host; dc.wss_target.port = uint16_t(port); dc.wss_target.prefix = prefix; dc.wss_target.secure = !ws;
        if (!pin.empty()) { dc.wss_target.pinned = true; std::memcpy(dc.wss_target.pin.data(), pin.data(), 32); }
        std::memcpy(dc.control_ticket.data(), c.data(), kTicketBytes); std::memcpy(dc.bulk_ticket.data(), b.data(), kTicketBytes);
        std::string err;
        if (!register_link_dialer(tool_registry_global(), dc, &err)) { say("dialer refused: %s", err); return 1; }
        int rc = 1;
        {
            Host h; h.requester = false; h.worker = true;
            if (!h.setup({}, "volunteer")) return 1;
            say("dialing %s", std::string(ws ? "ws://" : "wss://") + host + ":" + std::to_string(port) + "/ws/link/" + prefix + "/");
            rc = serve(h, linked_report, nullptr);
        }
        unregister_link(tool_registry_global(), "volunteer");
        return rc;
    }
    if (mode == "volunteer-test" && argc == 6) {
        // The server side only requests and re-checks; a volunteer child process dials it through a TLS front (the test
        // relay, where Apache stands in production) with a pinned certificate and no shader of its own.
        g_who = "server";
        const fs::path shader_dir = argv[2], work = argv[3]; fs::create_directories(work / "relay"); fs::create_directories(work / "volunteer");
        Key key{}; random_bytes(key.data(), key.size());
        LinkHostConfig hc; hc.id = "exchange.server"; hc.prefix = "lab"; hc.key = key;
        uint16_t port = 0; std::string err;
        if (!register_link_host(tool_registry_global(), hc, &port, &err)) { say("listener refused: %s", err); return 1; }
        nodus::exec::Program rp; rp.path = argv[4]; rp.arguments = {argv[5], std::to_string(port), (work / "relay").string()};
        rp.working_dir = (work / "relay").string();
        nodus::exec::Child relay; std::string why, line;
        if (!relay.start(rp, why)) { say("relay refused: %s", why); return 1; }
        auto until = Clock::now() + std::chrono::seconds(30);
        while (line.find('\n') == std::string::npos && Clock::now() < until && !relay.finished()) relay.take(line);
        if (line.rfind("RELAY ", 0) != 0) { say("relay said: %s", line); relay.kill(); return 1; }
        const std::string relay_port = line.substr(6, line.find(' ', 6) - 6);
        const std::string pin = line.substr(line.find(' ', 6) + 1, 64);
        int rc = 1;
        {
            Host h; h.requester = true; h.worker = false; h.seed = 0.5; h.remote_base = 3000; h.local_base = 7000;
            if (!h.setup(shader_dir, "exchange.server")) return 1;
            const auto id = new_link_id(); const uint64_t now = uint64_t(std::time(nullptr));
            auto c = sign_ticket(make_ticket(Channel::Control, Role::Dialer, id, "lab", 42, now, 120), key);
            auto b = sign_ticket(make_ticket(Channel::Bulk, Role::Dialer, id, "lab", 42, now, 120), key);
            nodus::exec::Program vp; vp.path = fs::absolute(argv[0]).string(); vp.working_dir = (work / "volunteer").string();
            vp.arguments = {"volunteer", "--host", "127.0.0.1", "--port", relay_port, "--prefix", "lab", "--pin", pin,
                            "--control-ticket", hex(c.data(), c.size()), "--bulk-ticket", hex(b.data(), b.size())};
            nodus::exec::Child volunteer;
            if (!volunteer.start(vp, why)) { say("volunteer refused: %s", why); return 1; }
            say("listening on 127.0.0.1:%s; the TLS front and a volunteer process started", std::to_string(port) + " (front 127.0.0.1:" + relay_port + ")");
            rc = serve(h, linked_report, &volunteer);
        }
        relay.finish_input();
        auto stop = Clock::now() + std::chrono::seconds(5);
        while (!relay.finished() && Clock::now() < stop) { std::string sink; relay.take(sink); }
        if (!relay.finished()) relay.kill();
        unregister_link(tool_registry_global(), "exchange.server");
        return rc;
    }
    std::printf("usage: link_gpu_exchange run <shader-dir> <work-dir> <glslangValidator>\n");
    std::printf("       link_gpu_exchange volunteer --host H --port P --prefix X [--pin SHA256HEX] [--ws] --control-ticket HEX --bulk-ticket HEX\n");
    std::printf("       link_gpu_exchange volunteer-test <shader-dir> <work-dir> <python with cryptography> <tls_relay.py>\n");
    return 2;
}
