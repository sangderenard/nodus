// link_route_test: the front route map that lets many links share one public port (module_library/source/tools/
// link_route.cpp, the row "nodus.link.route"; owner 2026-11-23: one front negotiating other ports, as Apache does).
//
//   link_route_test <work dir>
//     Two link hosts, each on a loopback port the system picks, and a route row for each. Checks: both lines are written
//     with their own ports; a line another writer owns (and the comment) is kept; a route name outside the front's
//     characters is refused; when the routes' canvas goes, both lines go and the other writer's line stays.
//   link_route_test <work dir> <route map> <front port>
//     The same, with the map at <route map>, read by an Apache front on 127.0.0.1:<front port> (RewriteMap txt:, see
//     tests/link_route_front.ps1). Each link is also dialed through that front by its own name and must link: a ticket
//     is signed for one link's name and key, so a connection routed to the other listener is refused.
#include "nodus_link.h"          // first: winsock2.h before windows.h
#include "nodus_headless_abi.h"
#include "headless_rows.h"

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

constexpr int32_t kTool = 1, kPlugin = 1;
int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

std::string read_text(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream s; s << in.rdbuf(); return s.str();
}
// The route row's saved state: "NLR1" version 1, link id, route name, map path.
std::vector<uint8_t> route_state(const std::string& link, const std::string& name, const std::string& map) {
    const uint32_t head[5] = {0x31524C4Eu, 1u, uint32_t(link.size()), uint32_t(name.size()), uint32_t(map.size())};
    std::vector<uint8_t> b(reinterpret_cast<const uint8_t*>(head), reinterpret_cast<const uint8_t*>(head) + sizeof(head));
    b.insert(b.end(), link.begin(), link.end()); b.insert(b.end(), name.begin(), name.end()); b.insert(b.end(), map.begin(), map.end());
    return b;
}
int table(NodusHeadlessGraph* g, const char* label, std::vector<GP_CanvasModuleRowDesc> rows) {
    GP_CanvasModuleDesc d{};
    d.w = 200; d.h = 120;
    std::snprintf(d.label, sizeof(d.label), "%s", label);
    const int m = nodus_headless_add_module(g, &d);
    return m >= 0 && nodus_headless_set_module_rows(g, m, rows.data(), int32_t(rows.size())) && nodus_headless_set_module_stack_contract(g, m, 1) ? m : -1;
}
int32_t link_state(const char* id) { int32_t s = -1; uint64_t c[16] = {}; return nodus_headless_link_report(id, &s, c) ? s : -1; }

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 4) {
        std::fprintf(stderr, "usage: link_route_test <work dir> [<route map> <front port>]\n");
        return 2;
    }
    const fs::path work = fs::path(argv[1]) / ("link_route_test_" + std::to_string(std::time(nullptr)));
    fs::create_directories(work);
    const fs::path map = argc == 4 ? fs::path(argv[2]) : work / "link.map";
    const int front = argc == 4 ? std::stoi(argv[3]) : 0;
    { std::ofstream out(map, std::ios::binary | std::ios::trunc); out << "# routes\nsomeone-else 127.0.0.1:1\n"; }
    const std::string route = headless_rows::build_named("link_route", work / "rows");
    if (route.empty()) return 2;

    nodus::link::Key key_a{}, key_b{};
    nodus::link::random_bytes(key_a.data(), key_a.size());
    nodus::link::random_bytes(key_b.data(), key_b.size());
    NodusHeadlessGraph* hosts = nodus_headless_create();
    const int32_t port_a = nodus_headless_register_link_host(hosts, "route.a", "mesh-a", 0, key_a.data(), 0);
    const int32_t port_b = nodus_headless_register_link_host(hosts, "route.b", "mesh-b", 0, key_b.data(), 0);
    CHECK(port_a > 0 && port_b > 0 && port_a != port_b);
    std::printf("hosts: mesh-a on %d, mesh-b on %d\n", port_a, port_b);
    // Each link's receive row keeps its endpoint moving (an endpoint advances when its rows run).
    const int recv = table(hosts, "receive", {{kTool, 0, 0, kPlugin, "route.a.bulk.receive"}, {kTool, 0, 0, kPlugin, "route.b.bulk.receive"}});
    const int routes = table(hosts, "routes", {{kTool, 0, 0, kPlugin, route.c_str()}, {kTool, 0, 0, kPlugin, route.c_str()},
                                               {kTool, 0, 0, kPlugin, route.c_str()}});
    CHECK(recv >= 0 && routes >= 0);
    const auto sa = route_state("route.a", "mesh-a", map.string()), sb = route_state("route.b", "mesh-b", map.string());
    const auto bad = route_state("route.a", "mesh a/../x", map.string());
    CHECK(nodus_headless_set_row_state(hosts, routes, 0, sa.data(), sa.size()));
    CHECK(nodus_headless_set_row_state(hosts, routes, 1, sb.data(), sb.size()));
    CHECK(!nodus_headless_set_row_state(hosts, routes, 2, bad.data(), bad.size()));   // refused: not a route name
    for (int i = 0; i < 20; ++i) nodus_headless_step(hosts, 1.0 / 60.0);

    std::string text = read_text(map);
    std::printf("map after the routes:\n%s", text.c_str());
    CHECK(text.find("mesh-a 127.0.0.1:" + std::to_string(port_a) + "\n") != std::string::npos);
    CHECK(text.find("mesh-b 127.0.0.1:" + std::to_string(port_b) + "\n") != std::string::npos);
    CHECK(text.find("someone-else 127.0.0.1:1\n") != std::string::npos && text.find("# routes\n") != std::string::npos);
    CHECK(text.find("mesh a") == std::string::npos);

    if (front) {
        // Dial each link by its own name through the front; each ticket is for its own link's name and key.
        NodusHeadlessGraph* dialers = nodus_headless_create();
        const uint64_t now = uint64_t(std::time(nullptr)), life = nodus::link::ListenerConfig{}.max_ticket_life;
        auto dial = [&](const char* id, const char* name, const nodus::link::Key& key) {
            const auto link = nodus::link::new_link_id();
            const auto ct = nodus::link::sign_ticket(nodus::link::make_ticket(nodus::link::Channel::Control, nodus::link::Role::Dialer, link, name, 7, now, life), key);
            const auto bt = nodus::link::sign_ticket(nodus::link::make_ticket(nodus::link::Channel::Bulk, nodus::link::Role::Dialer, link, name, 7, now, life), key);
            return nodus_headless_register_link_dialer(dialers, id, "127.0.0.1", uint16_t(front), name, ct.data(), bt.data(), 10000, 0);
        };
        CHECK(dial("dial.a", "mesh-a", key_a) && dial("dial.b", "mesh-b", key_b));
        CHECK(table(dialers, "receive", {{kTool, 0, 0, kPlugin, "dial.a.bulk.receive"}, {kTool, 0, 0, kPlugin, "dial.b.bulk.receive"}}) >= 0);
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        // Both ends of both links: a dialer is linked at the front's 101, its host once its next poll admits the tickets.
        auto all_linked = [] { return link_state("dial.a") == 1 && link_state("dial.b") == 1 && link_state("route.a") == 1 && link_state("route.b") == 1; };
        while (std::chrono::steady_clock::now() < until && !all_linked()) {
            nodus_headless_step(hosts, 1.0 / 60.0);
            nodus_headless_step(dialers, 1.0 / 60.0);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        std::printf("through the front on %d: dial.a %d route.a %d, dial.b %d route.b %d (1 = linked)\n", front,
                    link_state("dial.a"), link_state("route.a"), link_state("dial.b"), link_state("route.b"));
        CHECK(link_state("dial.a") == 1 && link_state("route.a") == 1);
        CHECK(link_state("dial.b") == 1 && link_state("route.b") == 1);
        nodus_headless_destroy(dialers);
    }

    nodus_headless_destroy(hosts);   // the route rows go, and their lines with them
    text = read_text(map);
    std::printf("map after the routes went:\n%s", text.c_str());
    CHECK(text.find("mesh-a") == std::string::npos && text.find("mesh-b") == std::string::npos);
    CHECK(text.find("someone-else 127.0.0.1:1\n") != std::string::npos);

    if (failures) { std::printf("FAILED %d\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
