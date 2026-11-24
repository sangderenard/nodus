// link_canvas_test: links declared in the canvas (stage C of docs/NODUS_NATIVE_HOST_PLAN_2026-11-23.md).
//
//   link_canvas_test <work dir>
//
// 1. A host opened on a canvas (nodus_headless_open_link_host, port 0, routed, with metrics) has its six tables, its
//    ports <id>.send / <id>.received, and its line in the front's route map; a save writes its LINK_HOST record (the key
//    file's path, never the key).
// 2. The canvas goes: its route line goes with it.
// 3. The saved canvas loaded by a new graph hosts the link again (a new port, the line written again) with its tables and
//    ports restored from the file.
// 4. A dialer opened on a third graph composes its own tables (nodus_headless_compose_link_tables) and sends a tensor
//    into <id>.send; it arrives byte for byte at the reloaded host's <id>.received, and the sent tensor is freed by the
//    send table's trash (its returns row hands it on). A dialed link is not recorded.
#include "nodus_link.h"          // first: winsock2.h before windows.h
#include "nodus_headless_abi.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

std::string read_text(const fs::path& p) { std::ifstream in(p, std::ios::binary); std::stringstream s; s << in.rdbuf(); return s.str(); }

int32_t module_named(NodusHeadlessGraph* g, const std::string& label) {
    for (int32_t m = 0, n = nodus_headless_get_module_count(g); m < n; ++m) {
        GP_CanvasModuleDesc d{};
        if (nodus_headless_get_module_desc(g, m, &d) && std::string(d.label, strnlen(d.label, sizeof(d.label))) == label) return m;
    }
    return -1;
}
bool port_named(NodusHeadlessGraph* g, const std::string& name, int32_t is_output) {
    for (int32_t k = 0, n = nodus_headless_get_port_count(g, is_output); k < n; ++k) {
        char buf[256] = {};
        int32_t mi = 0, ci = 0, head = 0;
        NodusHeadlessFifoConfig c{};
        if (nodus_headless_get_port_at(g, is_output, k, buf, sizeof(buf), &mi, &ci, &head, &c) && name == buf) return true;
    }
    return false;
}
// Tensor handles alive in the process (mesh_replica_test's count: ids are small and reused).
int live_tensors(uint64_t limit = 1u << 16) {
    int n = 0;
    for (uint64_t id = 1; id <= limit; ++id) { NodusTensorDesc d{}; if (nodus_tensor_describe(id, &d) == NODUS_OK) ++n; }
    return n;
}
uint64_t listen_port_of(const char* id) { int32_t s = 0; uint64_t c[16] = {}; return nodus_headless_link_report(id, &s, c) ? c[0] : 0; }
int32_t state_of(const char* id) { int32_t s = -1; uint64_t c[16] = {}; return nodus_headless_link_report(id, &s, c) ? s : -1; }

void check_tables(NodusHeadlessGraph* g, const std::string& id, bool routed, bool metrics) {
    for (const char* t : {" bulk send", " bulk receive", " control send", " control receive", " bulk trash", " control trash"})
        CHECK(module_named(g, "link " + id + t) >= 0);
    CHECK((module_named(g, "link " + id + " route") >= 0) == routed);
    CHECK((module_named(g, "link " + id + " metrics") >= 0) == metrics);
    CHECK(port_named(g, id + ".send", 0) && port_named(g, id + ".control", 0));
    CHECK(port_named(g, id + ".received", 1) && port_named(g, id + ".control.received", 1));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: link_canvas_test <work dir>\n"); return 2; }
    const fs::path work = fs::absolute(fs::path(argv[1]) / ("link_canvas_test_" + std::to_string(std::time(nullptr))));
    fs::create_directories(work);
    const fs::path canvas = work / "linked.nodus.txt", map = work / "link.map", key_file = work / "link.key";
    { std::ofstream m(map, std::ios::binary); m << "# routes\n"; }
    nodus::link::Key key{};
    nodus::link::random_bytes(key.data(), key.size());
    { std::ofstream k(key_file, std::ios::binary); k.write(reinterpret_cast<const char*>(key.data()), key.size()); }
    CHECK(nodus_headless_set_link_route_map(map.string().c_str()));

    // 1. Opened, composed, routed, saved.
    {
        NodusHeadlessGraph* a = nodus_headless_create();
        CHECK(!nodus_headless_open_link_host(a, "lk", "mesh lk", 0, NODUS_LINK_ROUTE, key_file.string().c_str()));   // a bad name
        CHECK(!nodus_headless_open_link_host(a, "lk", "mesh-lk", 0, NODUS_LINK_ROUTE, (work / "missing.key").string().c_str()));
        const int32_t port = nodus_headless_open_link_host(a, "lk", "mesh-lk", 0, NODUS_LINK_ROUTE | NODUS_LINK_METRICS, key_file.string().c_str());
        CHECK(port > 0);
        CHECK(!nodus_headless_open_link_host(a, "lk", "mesh-lk", 0, 0, key_file.string().c_str()));   // hosted already
        check_tables(a, "lk", true, true);
        for (int i = 0; i < 10; ++i) nodus_headless_step(a, 1.0 / 60.0);
        CHECK(read_text(map).find("mesh-lk 127.0.0.1:" + std::to_string(port) + "\n") != std::string::npos);
        CHECK(nodus_headless_save_file(a, canvas.string().c_str()));
        const std::string saved = read_text(canvas);
        CHECK(saved.find("LINK_HOST lk mesh-lk 0 3 " + key_file.string()) != std::string::npos);
        CHECK(saved.find(std::string(reinterpret_cast<const char*>(key.data()), 8)) == std::string::npos);   // the key stays in its file
        std::printf("opened: lk on %d; LINK_HOST recorded\n", port);
        nodus_headless_destroy(a);
    }
    // 2. Gone with its canvas.
    CHECK(read_text(map).find("mesh-lk") == std::string::npos);
    CHECK(state_of("lk") == -1);

    // 3. Loaded again: hosted, routed, tables and ports from the file.
    NodusHeadlessGraph* b = nodus_headless_create();
    CHECK(nodus_headless_load_file(b, canvas.string().c_str()));
    const uint64_t reloaded = listen_port_of("lk");
    CHECK(reloaded > 0);
    check_tables(b, "lk", true, true);
    for (int i = 0; i < 10; ++i) nodus_headless_step(b, 1.0 / 60.0);
    CHECK(read_text(map).find("mesh-lk 127.0.0.1:" + std::to_string(reloaded) + "\n") != std::string::npos);
    std::printf("reloaded: lk on %llu, routed\n", static_cast<unsigned long long>(reloaded));

    // 4. A dialer's tables composed; a tensor across the reloaded link.
    NodusHeadlessGraph* c = nodus_headless_create();
    const uint64_t now = uint64_t(std::time(nullptr)), life = nodus::link::ListenerConfig{}.max_ticket_life;
    const auto link = nodus::link::new_link_id();
    const auto ct = nodus::link::sign_ticket(nodus::link::make_ticket(nodus::link::Channel::Control, nodus::link::Role::Dialer, link, "mesh-lk", 7, now, life), key);
    const auto bt = nodus::link::sign_ticket(nodus::link::make_ticket(nodus::link::Channel::Bulk, nodus::link::Role::Dialer, link, "mesh-lk", 7, now, life), key);
    CHECK(nodus_headless_register_link_dialer(c, "dl", "127.0.0.1", uint16_t(reloaded), "mesh-lk", ct.data(), bt.data(), 10000, 0));
    CHECK(nodus_headless_compose_link_tables(c, "dl", 0));
    check_tables(c, "dl", false, false);
    const int alive_before = live_tensors();
    const uint64_t shape[1] = {4096};
    const uint64_t sent = nodus_tensor_create(NODUS_DTYPE_U8, 1, shape);
    const std::string payload(4096, char(0x5A));
    CHECK(sent && nodus_tensor_write(sent, 0, payload.data(), payload.size()) == int64_t(payload.size()));
    NodusHeadlessPublishResult pushed{};
    CHECK(nodus_headless_push_token_ex(c, "dl.send", &sent, sizeof(sent), &pushed) && pushed.accepted);
    uint64_t arrived = 0;
    size_t written = 0;
    for (auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(15); !arrived && std::chrono::steady_clock::now() < give_up;) {
        nodus_headless_step(c, 1.0 / 60.0);
        nodus_headless_step(b, 1.0 / 60.0);
        uint64_t h = 0;
        if (nodus_headless_pull_token(b, "lk.received", &h, sizeof(h), &written) && written == sizeof(h) && h) arrived = h;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    std::printf("dialer dl %d, host lk %d (1 = linked); arrived handle %llu\n", state_of("dl"), state_of("lk"), static_cast<unsigned long long>(arrived));
    CHECK(state_of("dl") == 1 && state_of("lk") == 1);
    CHECK(arrived != 0);
    if (arrived) {
        NodusTensorDesc d{};
        std::string got(4096, '\0');
        CHECK(nodus_tensor_describe(arrived, &d) == NODUS_OK && d.total_bytes == 4096 &&
              nodus_tensor_read(arrived, 0, got.data(), got.size()) == int64_t(got.size()) && got == payload);
        nodus_tensor_destroy(arrived);
    }
    // What was sent is not kept: the send table's returns row handed it to the trash, which freed it. Tensor ids are
    // process-wide and reused (the arrival took the sent tensor's id once it was freed), so the proof is the count of
    // live tensors: back to what it was before the push, with the arrival freed above.
    int alive = -1;
    for (int i = 0; i < 200 && alive != alive_before; ++i) {
        nodus_headless_step(c, 1.0 / 60.0);
        nodus_headless_step(b, 1.0 / 60.0);
        alive = live_tensors();
    }
    std::printf("tensors alive before the push %d, after %d (the sent one freed by the trash)\n", alive_before, alive);
    CHECK(alive == alive_before);
    const fs::path dialer_canvas = work / "dialer.nodus.txt";
    CHECK(nodus_headless_save_file(c, dialer_canvas.string().c_str()));
    CHECK(read_text(dialer_canvas).find("LINK_HOST") == std::string::npos);   // a dial is not recorded
    nodus_headless_destroy(c);
    nodus_headless_destroy(b);
    CHECK(read_text(map).find("mesh-lk") == std::string::npos);

    if (failures) { std::printf("FAILED %d\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
