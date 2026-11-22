// link_metrics_test: the link metrics row (module_library/source/tools/link_metrics.cpp), development case
// 2115fd107eba439fb0b5e65d6e6809e9 stage A1. Two link endpoints in one process (a host and a dialer, as link_rows_test),
// their <id>.control rows and a <id>.metrics row on each, stepped the way the thread manager steps a table's rows. The
// metrics row's port "metrics" is bound to a one-slot edge that the test drains. Checks: the published tensor is float64
// [N, 18], the newest sample is the last row, control_sent / control_received grew as packets were sent, the stack is
// left as found, the link is only read, and render_table paints.
#include "nodus_link_rows.h"
#if defined(LINK_METRICS_TEST_HEADLESS)
#include "nodus_headless_abi.h"
#endif
#include "common/tensors/abstraction/tensor_abi.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <vector>

using namespace nodus::link;
using Clock = std::chrono::steady_clock;
using Row = std::unique_ptr<ITool, std::function<void(ITool*)>>;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d  " #cond "  ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

struct Frame {
    RawStackFrame raw{};
    Frame() { raw_stack_init_frame(raw, 1 << 16); }
    ~Frame() { raw_stack_free_mask(raw); if (raw.backend) gp_mem_backend_release(raw.backend); }
};
// The row's output edge: holds `capacity` tokens (8-byte handles); the test pops them.
struct Edge {
    size_t capacity = 1;
    std::deque<uint64_t> tokens;
    static int32_t publish(void* h, int32_t port, const void* bytes, size_t len) {
        auto* e = static_cast<Edge*>(h);
        if (port != 0 || len != sizeof(uint64_t) || e->tokens.size() >= e->capacity) return 0;
        uint64_t t; std::memcpy(&t, bytes, sizeof(t)); e->tokens.push_back(t); return 1;
    }
    static int32_t can_publish(void* h, int32_t port, int32_t count) {
        auto* e = static_cast<Edge*>(h);
        return port == 0 && e->tokens.size() + size_t(count) <= e->capacity;
    }
    ToolRowPortAccess access() { ToolRowPortAccess a{}; a.host = this; a.publish = &publish; a.can_publish = &can_publish; return a; }
};
// One scheduler visit; returns the entry popped off the stack (0 = none).
static uint64_t visit(ITool& row, Frame& f, uint64_t input = 0, bool* succeeded = nullptr, bool* input_restored = nullptr, const ToolRowPortAccess* ports = nullptr) {
    if (input) raw_stack_push_abstract_tensor(f.raw, {input});
    ToolStackContext c{}; c.stack.raw = &f.raw; c.row_ports = ports;
    row.execute_stack(c);
    if (succeeded) *succeeded = c.succeeded;
    nodus::tensors::AbstractTensorHandle out{};
    if (!raw_stack_pop_abstract_tensor(f.raw, out)) { if (input_restored) *input_restored = false; return 0; }
    if (input && out.id == input) { if (input_restored) *input_restored = true; return 0; }
    if (input_restored) *input_restored = false;
    return out.id;
}
static uint64_t packet_tensor(uint32_t seq) {
    ControlPacket p; p.kind = uint16_t(ControlKind::Api); p.seq = seq; p.length = 4; put_le(p.payload.data(), seq * 7u, 4);
    uint8_t b[kControlPacketBytes]; encode_control(p, b);
    const uint64_t shape[1] = {kControlPacketBytes};
    uint64_t t = nodus_tensor_create(NODUS_DTYPE_U8, 1, shape); nodus_tensor_write(t, 0, b, sizeof(b)); return t;
}
// The newest published matrix: drains the edge, keeps the last, checks the shape; false when none arrived.
static bool newest_matrix(Edge& edge, uint64_t window, std::vector<double>& out, uint64_t columns = 18) {   // false: nothing new; `out` keeps the older one
    uint64_t last = 0;
    while (!edge.tokens.empty()) { if (last) nodus_tensor_destroy(last); last = edge.tokens.front(); edge.tokens.pop_front(); }
    if (!last) return false;
    NodusTensorDesc d{};
    bool ok = nodus_tensor_describe(last, &d) == NODUS_OK && d.dtype == NODUS_DTYPE_F64 && d.rank == 2 && d.shape[0] == window && d.shape[1] == columns
              && d.total_bytes == window * columns * sizeof(double);
    CHECK(ok, "published tensor is float64 [%llu,%llu]", (unsigned long long)window, (unsigned long long)columns);
    if (ok) { out.resize(size_t(window) * size_t(columns)); ok = nodus_tensor_read(last, 0, out.data(), d.total_bytes) == int64_t(d.total_bytes); }
    nodus_tensor_destroy(last);
    return ok;
}

int main() {
    auto& reg = tool_registry_global();
    Key key{}; random_bytes(key.data(), key.size());
    LinkHostConfig hc; hc.id = "proof.metrics.host"; hc.prefix = "lab"; hc.key = key;
    uint16_t port = 0; std::string err;
    CHECK(register_link_host(reg, hc, &port, &err) && port, "host register: %s", err.c_str());
    LinkId id = new_link_id(); uint64_t now = uint64_t(std::time(nullptr));
    LinkDialConfig dc; dc.id = "proof.metrics.dialer"; dc.target.port = port; dc.target.prefix = "lab";
    dc.control_ticket = sign_ticket(make_ticket(Channel::Control, Role::Dialer, id, "lab", 42, now, 60), key);
    dc.bulk_ticket = sign_ticket(make_ticket(Channel::Bulk, Role::Dialer, id, "lab", 42, now, 60), key);
    CHECK(register_link_dialer(reg, dc, &err), "dialer register: %s", err.c_str());

    // The receive rows take what arrives off the link onto their stacks; the dialer's send row sends the packets.
    Row hcon = reg.create("proof.metrics.host.control.receive"), dcon = reg.create("proof.metrics.dialer.control.receive");
    Row dsend = reg.create("proof.metrics.dialer.control.send");
    Row hmet = reg.create("proof.metrics.host.metrics"), dmet = reg.create("proof.metrics.dialer.metrics");
    CHECK(hcon && dcon && dsend && hmet && dmet, "rows from the registry");
    if (!(hcon && dcon && dsend && hmet && dmet)) return 1;
    CHECK(std::strcmp(hmet->id_cstr(), "proof.metrics.host.metrics") == 0 && hmet->row_port_count() == 1
          && std::strcmp(hmet->row_port(0).name, "metrics") == 0 && hmet->row_port(0).token_bytes == 8, "row port");
    Row generic = reg.create("nodus.link.metrics");
    CHECK(generic && std::strcmp(generic->id_cstr(), "nodus.link.metrics") == 0, "the generic row is registered by id");

    // A generic row takes its link from saved state: serialize a bound row's state, deserialize it into the generic one.
    {
        std::string path = "link_metrics_test.state.bin";
        { FileOutputArchive out(path.c_str()); CHECK(out.ok() && dmet->serialize(out), "serialize"); }
        FileInputArchive in(path.c_str()); CHECK(in.ok() && generic->deserialize(in), "deserialize into the generic row");
        std::remove(path.c_str());
    }

    const uint64_t N = 64;
    std::vector<double> dm, hm, gm;   // the newest matrix each metrics row published
    Frame fhc, fdc, fds, fhm, fdm, fgm;
    Edge hedge, dedge, gedge;
    ToolRowPortAccess hport = hedge.access(), dport = dedge.access(), gport = gedge.access();
    auto step = [&] {
        uint64_t o;
        if ((o = visit(*hcon, fhc))) nodus_tensor_destroy(o);
        if ((o = visit(*dcon, fdc))) nodus_tensor_destroy(o);
        bool ok = false;
        o = visit(*hmet, fhm, 0, &ok, nullptr, &hport); CHECK(!o && ok, "metrics row left the stack as found");
        o = visit(*dmet, fdm, 0, &ok, nullptr, &dport); CHECK(!o && ok, "metrics row left the stack as found");
        o = visit(*generic, fgm, 0, &ok, nullptr, &gport); CHECK(!o && ok, "generic row left the stack as found");
        newest_matrix(hedge, N, hm); newest_matrix(dedge, N, dm); newest_matrix(gedge, N, gm);   // the edges' consumers
    };

    EndpointReport hr, dr;
    auto end = Clock::now() + std::chrono::seconds(5);
    while (Clock::now() < end) {
        step();
        link_endpoint_report("proof.metrics.host", &hr); link_endpoint_report("proof.metrics.dialer", &dr);
        if (hr.state == EndpointState::Linked && dr.state == EndpointState::Linked) break;
    }
    CHECK(hr.state == EndpointState::Linked && dr.state == EndpointState::Linked, "linked: host %d dialer %d", int(hr.state), int(dr.state));

    // A few control packets dialer -> host through the link rows; the metrics rows sample every step.
    const int kPackets = 5;
    for (int i = 0; i < kPackets; ++i) {
        uint64_t t = packet_tensor(100 + uint32_t(i)); bool accepted = false;
        for (auto stop = Clock::now() + std::chrono::seconds(5); !accepted && Clock::now() < stop;) {
            bool restored = true; uint64_t o = visit(*dsend, fds, t, nullptr, &restored); accepted = !restored; if (o) nodus_tensor_destroy(o);
            step();
        }
        CHECK(accepted, "packet %d accepted", i);
        nodus_tensor_destroy(t);
    }
    for (auto stop = Clock::now() + std::chrono::seconds(5); Clock::now() < stop;) {
        step();
        link_endpoint_report("proof.metrics.host", &hr);
        if (hr.link.control_received >= uint64_t(kPackets)) break;
    }
    step();
    link_endpoint_report("proof.metrics.dialer", &dr);
    CHECK(dr.link.control_sent >= uint64_t(kPackets) && hr.link.control_received >= uint64_t(kPackets), "link counters sent %llu received %llu",
          (unsigned long long)dr.link.control_sent, (unsigned long long)hr.link.control_received);

    // Columns: 0 state, 1..16 the counters of nodus_headless_link_report (control_sent = 3, control_received = 4), 17 seconds.
    CHECK(dm.size() == N * 18 && hm.size() == N * 18, "matrices published");
    CHECK(gm.size() == N * 18, "generic row published (bound by saved state to the dialer)");
    if (dm.size() == N * 18 && hm.size() == N * 18) {
        const double* dlast = &dm[(N - 1) * 18]; const double* hlast = &hm[(N - 1) * 18];
        CHECK(dlast[0] == 1.0 && hlast[0] == 1.0, "state linked (dialer %.0f host %.0f)", dlast[0], hlast[0]);
        CHECK(dlast[3] >= kPackets && dlast[3] == double(dr.link.control_sent) , "dialer control_sent %.0f (report %llu)", dlast[3], (unsigned long long)dr.link.control_sent);
        CHECK(hlast[4] >= kPackets, "host control_received %.0f", hlast[4]);
        CHECK(dlast[17] > 0.0, "monotonic seconds");
        // Oldest first: zero rows lead, the filled rows ascend in time and in control_sent, and the first filled row is before the sends.
        uint32_t first = 0; while (first < N && dm[first * 18 + 17] == 0.0) ++first;
        CHECK(first > 0 && first < N - 1, "zero rows lead until filled (first filled row %u)", first);
        bool ascending = true;
        for (uint32_t r = first + 1; r < N; ++r)
            ascending = ascending && dm[r * 18 + 17] >= dm[(r - 1) * 18 + 17] && dm[r * 18 + 3] >= dm[(r - 1) * 18 + 3];
        CHECK(ascending, "rows ascend in time and control_sent");
        CHECK(dm[first * 18 + 3] < dlast[3], "control_sent grew over the window (%.0f -> %.0f)", dm[first * 18 + 3], dlast[3]);
    }
    if (gm.size() == N * 18) CHECK(gm[(N - 1) * 18 + 3] >= kPackets, "generic row follows its saved link");

    // The picture.
    {
        const int w = 320, h = 216; std::vector<uint8_t> px(size_t(w) * h * 4, 0);
        TableRenderArgs a; a.rows = 1; a.cols = 1; a.cell_w = float(w); a.cell_h = float(h); a.frame_buffer = px.data(); a.pitch = w * 4;
        dmet->render_table(a);
        size_t lit = 0; for (size_t i = 0; i < px.size(); i += 4) lit += (px[i] | px[i + 1] | px[i + 2] | px[i + 3]) != 0;
        CHECK(lit > 100, "render_table painted %zu pixels", lit);
    }
    std::printf("dialer last row: state %.0f control_sent %.0f; host last row: control_received %.0f\n",
                dm.size() == N * 18 ? dm[(N - 1) * 18] : -1.0, dm.size() == N * 18 ? dm[(N - 1) * 18 + 3] : -1.0, hm.size() == N * 18 ? hm[(N - 1) * 18 + 4] : -1.0);

#if defined(LINK_METRICS_TEST_HEADLESS)
    // Channel group: the same row, given a meta channel, appends what the meta edges of that channel heard (columns 18..25).
    // A real headless canvas (nodus_headless.dll) holds two tables and an edge carrying tokens, a meta group on channel 7 over
    // it, and is stepped by the real scheduler; the row reads it through nodus_headless's process-wide reader.
    {
        constexpr int32_t kChannel = 7;
        NodusHeadlessGraph* graph = nodus_headless_create();
        CHECK(graph, "a headless canvas");
        if (graph) {
            GP_CanvasModuleDesc ma{}, mb{}; ma.w = ma.h = mb.w = mb.h = 64; mb.x = 100;
            std::strncpy(ma.label, "alloc", sizeof(ma.label) - 1); std::strncpy(mb.label, "tool", sizeof(mb.label) - 1);
            const int32_t a = nodus_headless_add_module(graph, &ma), b = nodus_headless_add_module(graph, &mb);
            const int32_t ra = nodus_headless_bind_builtin_tool(graph, a, 12), rb = nodus_headless_bind_builtin_tool(graph, b, 13);
            GP_CanvasEdgeDesc e{}; e.a_module = a; e.a_contact_idx = ra; e.b_module = b; e.b_contact_idx = rb;
            const int32_t edge_idx = nodus_headless_add_edge(graph, &e, 0);
            NodusHeadlessFifoConfig fc{}; fc.token_bytes = 16; fc.slots = 4; fc.top_k = 0; fc.type_id = 0x4D455441; fc.layout = 2; fc.dtype = 8;
            CHECK(edge_idx >= 0 && nodus_headless_configure_edge_fifo(graph, edge_idx, &fc), "the member edge");
            const uint64_t group = nodus_headless_meta_create(graph, &edge_idx, 1, kChannel);
            CHECK(group != 0, "a meta group on channel %d", kChannel);
            uint8_t token[16] = {9};
            for (int i = 0; i < 6; ++i) { NodusHeadlessPublishResult pr{}; nodus_headless_publish_edge_token(graph, edge_idx, token, sizeof(token), &pr); }
            CHECK(nodus_headless_step(graph, 1.0 / 60.0) && nodus_headless_step(graph, 1.0 / 60.0), "two scheduler ticks");

            set_link_metrics_channel("proof.metrics.dialer", kChannel);   // the next dialer.metrics row starts on the channel
            Row cm = reg.create("proof.metrics.dialer.metrics"), plain = reg.create("proof.metrics.host.metrics");
            CHECK(cm && plain, "rows");
            Frame fcm, fpl; Edge cedge, pedge; ToolRowPortAccess cport = cedge.access(), pport = pedge.access();
            uint64_t o;
            for (int i = 0; i < 3; ++i) {
                o = visit(*cm, fcm, 0, nullptr, nullptr, &cport); if (o) nodus_tensor_destroy(o);
                o = visit(*plain, fpl, 0, nullptr, nullptr, &pport); if (o) nodus_tensor_destroy(o);
                std::vector<double> sink; if (i < 2) { newest_matrix(cedge, N, sink, 26); newest_matrix(pedge, N, sink, 18); }
            }
            std::vector<double> m26, m18;
            CHECK(newest_matrix(cedge, N, m26, 26), "a channel row publishes [64,26]");
            CHECK(newest_matrix(pedge, N, m18, 18), "a row with no channel still publishes [64,18]");
            if (m26.size() == N * 26) {
                const double* last = &m26[(N - 1) * 26];
                CHECK(last[3] >= kPackets, "link columns intact: control_sent %.0f", last[3]);
                CHECK(last[18] == 1.0, "meta_members %.0f", last[18]);
                CHECK(last[19] >= 4.0, "meta_write_seq %.0f (the edge took four of six tokens)", last[19]);
                CHECK(last[23] == 4.0 && last[22] <= 4.0, "meta_slots %.0f occupancy %.0f", last[23], last[22]);
                std::printf("meta columns: members %.0f write_seq %.0f write_delta %.0f read_delta %.0f occupancy %.0f/%.0f refused_delta %.0f bytes_delta %.0f\n",
                            last[18], last[19], last[20], last[21], last[22], last[23], last[24], last[25]);
            }
            // The channel travels in saved state: a bound row's state loads into the generic row with 26 columns.
            {
                std::string path = "link_metrics_test.state26.bin";
                { FileOutputArchive out(path.c_str()); CHECK(out.ok() && cm->serialize(out), "serialize with a channel"); }
                Row gen = reg.create("nodus.link.metrics"); Frame fg; Edge gedge2; ToolRowPortAccess gp2 = gedge2.access();
                { FileInputArchive in(path.c_str()); CHECK(in.ok() && gen->deserialize(in), "deserialize the channel"); }
                std::remove(path.c_str());
                o = visit(*gen, fg, 0, nullptr, nullptr, &gp2); if (o) nodus_tensor_destroy(o);
                std::vector<double> mg; CHECK(newest_matrix(gedge2, N, mg, 26), "a loaded row keeps its channel (26 columns)");
            }
            nodus_headless_destroy(graph);
        }
    }
#endif

    hcon.reset(); dcon.reset(); hmet.reset(); dmet.reset(); generic.reset();
    unregister_link(reg, "proof.metrics.host"); unregister_link(reg, "proof.metrics.dialer");
    CHECK(!reg.create("proof.metrics.host.metrics") && !reg.create("proof.metrics.dialer.metrics"), "unregister left a metrics row");
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
