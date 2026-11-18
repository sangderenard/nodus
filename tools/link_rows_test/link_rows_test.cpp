// link_rows_test: stage 2 of the Nodus link (research case 98d49cb8520e44d092c5d134c37c3c32). A host endpoint and a
// dialer endpoint registered in the real tool registry; their <id>.control and <id>.bulk rows are created from the
// registry and driven through execute_stack the way the thread manager drives a row, one thread, nothing blocking.
#include "nodus_link_rows.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
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
// One scheduler visit: optional input tensor on the stack, then execute_stack; returns the popped output (0 = none).
static uint64_t visit(ITool& row, Frame& f, uint64_t input, bool* succeeded = nullptr, bool* input_restored = nullptr) {
    if (input) raw_stack_push_abstract_tensor(f.raw, {input});
    ToolStackContext c{}; c.stack.raw = &f.raw;
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
static uint64_t f32_tensor(uint64_t rows, uint64_t cols, float base) {
    const uint64_t shape[2] = {rows, cols};
    uint64_t t = nodus_tensor_create(NODUS_DTYPE_F32, 2, shape);
    std::vector<float> v(size_t(rows * cols)); for (size_t i = 0; i < v.size(); ++i) v[i] = base + float(i) * 0.25f;
    nodus_tensor_write(t, 0, v.data(), v.size() * sizeof(float)); return t;
}
static bool same_tensor(uint64_t a, uint64_t b) {
    NodusTensorDesc da{}, db{};
    if (nodus_tensor_describe(a, &da) != NODUS_OK || nodus_tensor_describe(b, &db) != NODUS_OK) return false;
    if (da.dtype != db.dtype || da.rank != db.rank || da.total_bytes != db.total_bytes) return false;
    for (uint32_t i = 0; i < da.rank; ++i) if (da.shape[i] != db.shape[i]) return false;
    std::vector<uint8_t> x(size_t(da.total_bytes)), y(size_t(db.total_bytes));
    nodus_tensor_read(a, 0, x.data(), x.size()); nodus_tensor_read(b, 0, y.data(), y.size());
    return x == y;
}

// Sends `t` into `from` until accepted, visiting `to` until a tensor comes out; returns it (0 on timeout).
static uint64_t carry(ITool& from, Frame& ff, ITool& to, Frame& tf, uint64_t t, ITool* other_a = nullptr, ITool* other_b = nullptr) {
    bool accepted = false; uint64_t got = 0;
    auto end = Clock::now() + std::chrono::seconds(20);
    Frame spare;
    while (!got && Clock::now() < end) {
        if (!accepted) { bool restored = true; uint64_t o = visit(from, ff, t, nullptr, &restored); accepted = !restored; if (o) nodus_tensor_destroy(o); }
        else { uint64_t o = visit(from, ff, 0); if (o) nodus_tensor_destroy(o); }
        if (other_a) { uint64_t o = visit(*other_a, spare, 0); if (o) nodus_tensor_destroy(o); }
        if (other_b) { uint64_t o = visit(*other_b, spare, 0); if (o) nodus_tensor_destroy(o); }
        got = visit(to, tf, 0);
    }
    return got;
}

int main() {
    auto& reg = tool_registry_global();
    Key key{}; random_bytes(key.data(), key.size());
    LinkHostConfig hc; hc.id = "proof.link.host"; hc.prefix = "lab"; hc.key = key;
    uint16_t port = 0; std::string err;
    CHECK(register_link_host(reg, hc, &port, &err) && port, "host register: %s", err.c_str());

    LinkId id = new_link_id(); uint64_t now = uint64_t(std::time(nullptr));
    LinkDialConfig dc; dc.id = "proof.link.dialer"; dc.target.port = port; dc.target.prefix = "lab";
    dc.control_ticket = sign_ticket(make_ticket(Channel::Control, Role::Dialer, id, "lab", 42, now, 60), key);
    dc.bulk_ticket = sign_ticket(make_ticket(Channel::Bulk, Role::Dialer, id, "lab", 42, now, 60), key);
    CHECK(register_link_dialer(reg, dc, &err), "dialer register: %s", err.c_str());
    CHECK(!register_link_dialer(reg, dc, &err), "the same id registered twice");

    Row hcon = reg.create("proof.link.host.control"), hbulk = reg.create("proof.link.host.bulk");
    Row dcon = reg.create("proof.link.dialer.control"), dbulk = reg.create("proof.link.dialer.bulk");
    CHECK(hcon && hbulk && dcon && dbulk, "rows from the registry");
    if (!(hcon && hbulk && dcon && dbulk)) return 1;
    CHECK(hbulk->port_count() == 2 && hbulk->port_spec(0).kind == ToolPortKind::Argument && hbulk->port_spec(1).kind == ToolPortKind::Return, "ports");
    Frame fhc, fhb, fdc, fdb;

    // Before the link is up an input is refused and restored to the stack.
    {
        uint64_t t = packet_tensor(1); bool ok = true, restored = false;
        visit(*dcon, fdc, t, &ok, &restored);
        EndpointReport r; link_endpoint_report("proof.link.dialer", &r);
        CHECK(!ok && restored, "input before link: succeeded=%d restored=%d state=%d", int(ok), int(restored), int(r.state));
        nodus_tensor_destroy(t);
    }
    // Rows are polled by the scheduler; nothing else drives the endpoints.
    auto end = Clock::now() + std::chrono::seconds(5);
    EndpointReport hr, dr;
    Frame idle;
    while (Clock::now() < end) {
        for (auto* row : {hcon.get(), hbulk.get(), dcon.get(), dbulk.get()}) { uint64_t o = visit(*row, idle, 0); if (o) nodus_tensor_destroy(o); }
        link_endpoint_report("proof.link.host", &hr); link_endpoint_report("proof.link.dialer", &dr);
        if (hr.state == EndpointState::Linked && dr.state == EndpointState::Linked) break;
    }
    CHECK(hr.state == EndpointState::Linked && dr.state == EndpointState::Linked && hr.account == 42, "linked: host %d dialer %d", int(hr.state), int(dr.state));
    std::printf("linked through rows: host listens on %u, account %llu\n", port, (unsigned long long)hr.account);

    // control.out -> control.in, both ways: one 256-byte packet tensor each.
    for (uint32_t i = 0; i < 50; ++i) {
        uint64_t t = packet_tensor(100 + i), got = carry(*dcon, fdc, *hcon, fhc, t);
        CHECK(got && same_tensor(t, got), "control dialer->host %u", i);
        nodus_tensor_destroy(t); if (got) nodus_tensor_destroy(got);
        t = packet_tensor(200 + i); got = carry(*hcon, fhc, *dcon, fdc, t);
        CHECK(got && same_tensor(t, got), "control host->dialer %u", i);
        nodus_tensor_destroy(t); if (got) nodus_tensor_destroy(got);
    }
    std::printf("control rows: 50 packets each way identical\n");
    // A tensor that is not one control packet is refused.
    {
        const uint64_t shape[1] = {100}; uint64_t t = nodus_tensor_create(NODUS_DTYPE_U8, 1, shape); bool restored = false;
        visit(*dcon, fdc, t, nullptr, &restored);
        CHECK(restored, "a 100-byte control input was not refused");
        nodus_tensor_destroy(t);
    }

    // bulk.out -> bulk.in: an F32 [1024, 4096] tensor (16 MiB) each way keeps dtype, shape and bytes.
    for (int dir = 0; dir < 2; ++dir) {
        uint64_t t = f32_tensor(1024, 4096, dir ? -3.5f : 1.0f);
        auto t0 = Clock::now();
        uint64_t got = dir ? carry(*hbulk, fhb, *dbulk, fdb, t, hcon.get(), dcon.get()) : carry(*dbulk, fdb, *hbulk, fhb, t, hcon.get(), dcon.get());
        double s = std::chrono::duration<double>(Clock::now() - t0).count();
        CHECK(got && same_tensor(t, got), "bulk %s", dir ? "host->dialer" : "dialer->host");
        std::printf("bulk rows %s: F32 [1024,4096] identical in %.3f s\n", dir ? "host->dialer" : "dialer->host", s);
        nodus_tensor_destroy(t); if (got) nodus_tensor_destroy(got);
    }
    link_endpoint_report("proof.link.host", &hr); link_endpoint_report("proof.link.dialer", &dr);
    CHECK(hr.bulk_tensors_received == 1 && hr.bulk_tensors_sent == 1 && dr.bulk_tensors_received == 1 && dr.bulk_dropped == 0 && hr.bulk_dropped == 0,
          "bulk counters host in %llu out %llu", (unsigned long long)hr.bulk_tensors_received, (unsigned long long)hr.bulk_tensors_sent);
    std::printf("host report: control in %llu out %llu, bulk tensors in %llu out %llu, refused control %llu bulk %llu\n",
                (unsigned long long)hr.link.control_received, (unsigned long long)hr.link.control_sent, (unsigned long long)hr.bulk_tensors_received,
                (unsigned long long)hr.bulk_tensors_sent, (unsigned long long)hr.control_refused, (unsigned long long)hr.bulk_refused);

    hcon.reset(); hbulk.reset(); dcon.reset(); dbulk.reset();
    unregister_link(reg, "proof.link.host"); unregister_link(reg, "proof.link.dialer");
    CHECK(!reg.create("proof.link.host.control"), "unregister left a row");
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
