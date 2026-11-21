// file_link_test: the file rows (module_library/source/tools/file_read.cpp, file_write.cpp) and the link's split rows
// (nodus_link.cpp), composed in tables -- no host code moves a byte.
//
// Tables that end in an Output row are in stack loopback, so a tensor whose edge is full waits instead of being reset.
//   local:  [file_read | Output] -> [Input | file_write]                     in one canvas
//   link:   sender canvas   [file_read | Output] -> [Input | <id>.bulk.send | tensor_returns]  (stack loopback: a
//           refused send waits); tensor_returns "returns" -> [tensor_mint] "returned": a mint with no size, the trash
//           receiver canvas [<id>.bulk.receive | Output] -> [Input | file_write]
// Two "generations" share their file names (000000.rec, 000001.rec, ...), as state_broadcast's do. Every file arrives
// byte for byte, the source folder ends empty with every file in sent/, and nothing is overwritten. Every sent tensor
// goes back to the mint and is freed: the process ends the run with the tensors it started with.
//
//   file_link_test <file_read plugin> <file_write plugin> <tensor_mint plugin> <tensor_returns plugin> <work dir>
#include "canvas_abi.h"
#include "nodus_headless_abi.h"
#include "nodus_link.h"
#include "plugin_manager.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// UTF-8 text of a path and back (C++20 paths speak char8_t).
inline std::string utf8_of(const fs::path& p) { const auto u = p.u8string(); return std::string(u.begin(), u.end()); }
inline fs::path path_of_utf8(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }

namespace {

int failures = 0;
#define CHECK(cond)                                                                                  \
    do {                                                                                             \
        if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

constexpr int32_t kTool = 1, kInput = 0, kOutput = 2, kPlugin = 1;
constexpr int32_t kRowPortBase = 20000, kRowPortStride = 64;

// Tensor handles alive in the process, by describing every id up to `limit` (ids are small and reused).
int live_tensors(uint64_t limit = 1u << 16) {
    int n = 0;
    for (uint64_t id = 1; id <= limit; ++id) {
        NodusTensorDesc d{};
        if (nodus_tensor_describe(id, &d) == NODUS_OK) ++n;
    }
    return n;
}

std::vector<uint8_t> read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// A folder setting: the rows' saved state (magic, version 1, length, folder; file_read adds its largest file).
std::vector<uint8_t> folder_state(uint32_t magic, const fs::path& folder, bool with_largest) {
    const std::string f = utf8_of(folder);
    const uint32_t head[3] = {magic, 1, static_cast<uint32_t>(f.size())};
    std::vector<uint8_t> s(reinterpret_cast<const uint8_t*>(head), reinterpret_cast<const uint8_t*>(head) + sizeof(head));
    s.insert(s.end(), f.begin(), f.end());
    if (with_largest) {
        const uint64_t largest = uint64_t(64) << 20;
        s.insert(s.end(), reinterpret_cast<const uint8_t*>(&largest), reinterpret_cast<const uint8_t*>(&largest) + 8);
    }
    return s;
}

struct Canvas {
    NodusHeadlessGraph* g = nodus_headless_create();
    NodusHeadlessFifoConfig lossless{8, 4, 0, 0, 2, 16, 0};       // one tensor handle a token
    Canvas() { lossless.type_id = nodus_headless_abstract_tensor_type_id(); }
    ~Canvas() { if (g) nodus_headless_destroy(g); }
    int table(const char* label, std::vector<GP_CanvasModuleRowDesc> rows) {
        GP_CanvasModuleDesc d{};
        d.w = 200; d.h = 120;
        std::snprintf(d.label, sizeof(d.label), "%s", label);
        const int m = nodus_headless_add_module(g, &d);
        if (m < 0 || !nodus_headless_set_module_rows(g, m, rows.data(), static_cast<int32_t>(rows.size())) ||
            !nodus_headless_set_module_stack_contract(g, m, 1))
            return -1;
        return m;
    }
    bool edge(int a, int ac, int b, int bc) {
        GP_CanvasEdgeDesc e{a, ac, b, bc};
        const int idx = nodus_headless_add_edge(g, &e, 0);
        return idx >= 0 && nodus_headless_configure_edge_fifo(g, idx, &lossless);
    }
    bool step() { return nodus_headless_step(g, 1.0 / 60.0) != 0; }
};

// Two generations of records with the same names; sizes from tiny to past one link chunk.
std::vector<std::vector<uint8_t>> make_files(const fs::path& src, int big) {
    std::mt19937_64 rng(7);
    std::vector<std::vector<uint8_t>> all;
    fs::create_directories(src);
    const size_t sizes[] = {1, 97, 4096, 65537, 1u << 20};
    for (int gen = 0; gen < 2; ++gen) {
        std::vector<size_t> these(std::begin(sizes), std::end(sizes));
        for (int b = 0; b < big; ++b) these.push_back((size_t(8) << 20) + size_t(gen * 31 + b));
        fs::create_directories(src / ("gen" + std::to_string(gen)));
        for (size_t i = 0; i < these.size(); ++i) {
            std::vector<uint8_t> bytes(these[i]);
            for (auto& x : bytes) x = static_cast<uint8_t>(rng());
            char name[32];
            std::snprintf(name, sizeof(name), "%06zu.rec", i);
            std::ofstream(src / ("gen" + std::to_string(gen)) / name, std::ios::binary)
                .write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            all.push_back(std::move(bytes));
        }
    }
    return all;
}

// The generations go into the source folder one after the other, the second once the first has been taken, as a
// writer would hand over a generation: its names collide with names already in sent/.
bool hand_over(const fs::path& src, int gen) {
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(src / ("gen" + std::to_string(gen)))) {
        fs::rename(e.path(), src / e.path().filename(), ec);
        if (ec) return false;
    }
    return true;
}

size_t visible_files(const fs::path& dir) {
    size_t n = 0;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file(ec) && utf8_of(it->path().filename())[0] != '.') ++n;
    return n;
}

bool same_contents(std::vector<std::vector<uint8_t>> sent, const fs::path& dst) {
    std::vector<std::vector<uint8_t>> got;
    for (const auto& e : fs::directory_iterator(dst))
        if (e.is_regular_file() && e.path().extension() == ".arrival") got.push_back(read_file(e.path()));
    std::sort(sent.begin(), sent.end());
    std::sort(got.begin(), got.end());
    return got == sent;
}

void local_case(const char* read_id, const char* write_id, const fs::path& dir) {
    const fs::path src = dir / "local_src", dst = dir / "local_dst";
    const auto files = make_files(src, 1);
    Canvas c;
    const int r = c.table("read", {{kTool, 0, 0, kPlugin, read_id}, {kOutput, 0, 1, 0, nullptr}});
    const int w = c.table("write", {{kInput, 0, 1, 0, nullptr}, {kTool, 0, 0, kPlugin, write_id}});
    CHECK(r >= 0 && w >= 0 && c.edge(r, 0, w, 0));
    CHECK(nodus_headless_set_module_stack_loopback(c.g, r, 1));      // a file its Output row has not moved yet stays
    const auto rs = folder_state(0x3152464E, src, true), ws = folder_state(0x3157464E, dst, false);
    CHECK(nodus_headless_set_row_state(c.g, r, 0, rs.data(), rs.size()));
    CHECK(nodus_headless_set_row_state(c.g, w, 1, ws.data(), ws.size()));
    const size_t per_gen = files.size() / 2;
    CHECK(hand_over(src, 0));
    bool second = false;
    for (int tick = 0; tick < 4000 && visible_files(dst) < files.size(); ++tick) {
        c.step();
        if (!second && visible_files(dst) >= per_gen && visible_files(src) == 0) second = hand_over(src, 1);
    }
    CHECK(second);
    CHECK(visible_files(src) == 0 && visible_files(src / "sent") == files.size());
    CHECK(same_contents(files, dst));
    std::printf("local: %zu of %zu files arrived byte for byte; %zu in sent/\n", visible_files(dst), files.size(),
                visible_files(src / "sent"));
}

void link_case(const char* read_id, const char* write_id, const char* mint_id, const char* returns_id, const fs::path& dir) {
    const fs::path src = dir / "link_src", dst = dir / "link_dst";
    const auto files = make_files(src, 6);
    nodus::link::Key key{};
    nodus::link::random_bytes(key.data(), key.size());
    Canvas sender, receiver;
    const int32_t port = nodus_headless_register_link_host(receiver.g, "files.r", "files", 0, key.data(), 0);
    CHECK(port > 0);
    const auto id = nodus::link::new_link_id();
    const uint64_t now = uint64_t(std::time(nullptr));
    using nodus::link::Channel;
    using nodus::link::Role;
    const auto ct = nodus::link::sign_ticket(nodus::link::make_ticket(Channel::Control, Role::Dialer, id, "files", 7, now, 60), key);
    const auto bt = nodus::link::sign_ticket(nodus::link::make_ticket(Channel::Bulk, Role::Dialer, id, "files", 7, now, 60), key);
    CHECK(nodus_headless_register_link_dialer(sender.g, "files.s", "127.0.0.1", uint16_t(port), "files", ct.data(), bt.data(), 5000, 0));

    const int r = sender.table("read", {{kTool, 0, 0, kPlugin, read_id}, {kOutput, 0, 1, 0, nullptr}});
    const int live_before = live_tensors();
    const int s = sender.table("send", {{kInput, 0, 1, 0, nullptr}, {kTool, 0, 0, kPlugin, "files.s.bulk.send"},
                                        {kTool, 0, 0, kPlugin, returns_id}});
    const int trash = sender.table("trash", {{kTool, 0, 0, kPlugin, mint_id}});
    CHECK(trash >= 0 && sender.edge(s, kRowPortBase + 2 * kRowPortStride, trash, kRowPortBase) );   // returns -> returned
    CHECK(r >= 0 && s >= 0 && sender.edge(r, 0, s, 0));
    CHECK(nodus_headless_set_module_stack_loopback(sender.g, r, 1));
    CHECK(nodus_headless_set_module_stack_loopback(sender.g, s, 1));                 // a refused send waits on the stack
    const int v = receiver.table("receive", {{kTool, 0, 0, kPlugin, "files.r.bulk.receive"}, {kOutput, 0, 1, 0, nullptr}});
    const int w = receiver.table("write", {{kInput, 0, 1, 0, nullptr}, {kTool, 0, 0, kPlugin, write_id}});
    CHECK(v >= 0 && w >= 0 && receiver.edge(v, 0, w, 0));
    CHECK(nodus_headless_set_module_stack_loopback(receiver.g, v, 1));   // an arrival not moved out yet stays
    const auto rs = folder_state(0x3152464E, src, true), ws = folder_state(0x3157464E, dst, false);
    CHECK(nodus_headless_set_row_state(sender.g, r, 0, rs.data(), rs.size()));
    CHECK(nodus_headless_set_row_state(receiver.g, w, 1, ws.data(), ws.size()));

    int32_t hs = -1, ds = -1;
    uint64_t hv[16] = {}, dv[16] = {};
    for (int i = 0; i < 2000 && !(hs == 1 && ds == 1); ++i) {
        sender.step(); receiver.step();
        nodus_headless_link_report("files.r", &hs, hv); nodus_headless_link_report("files.s", &ds, dv);
    }
    CHECK(hs == 1 && ds == 1);
    const size_t per_gen = files.size() / 2;
    CHECK(hand_over(src, 0));
    bool second = false;
    for (int tick = 0; tick < 20000 && visible_files(dst) < files.size(); ++tick) {
        sender.step(); receiver.step();
        if (!second && visible_files(dst) >= per_gen && visible_files(src) == 0) second = hand_over(src, 1);
    }
    nodus_headless_link_report("files.s", &ds, dv);
    nodus_headless_link_report("files.r", &hs, hv);
    std::printf("link counters: sender bulk tensors sent %llu refused %llu; receiver bulk tensors received %llu dropped %llu\n",
                (unsigned long long)dv[6], (unsigned long long)dv[9], (unsigned long long)hv[7], (unsigned long long)hv[10]);
    CHECK(second);
    CHECK(visible_files(src) == 0 && visible_files(src / "sent") == files.size());
    CHECK(same_contents(files, dst));
    // dv: 0 listen_port, 1 account, 2 control_sent, 3 control_received, 4/5 bulk bytes sent/received,
    // 6/7 bulk tensors sent/received, 8 control_refused, 9 bulk_refused, 10 bulk_dropped (nodus_headless_link_report).
    for (int tick = 0; tick < 60; ++tick) { sender.step(); receiver.step(); }      // the last returns reach the trash
    const int live_after = live_tensors();
    CHECK(live_after <= live_before);
    std::printf("link: tensors alive before %d, after %d\n", live_before, live_after);
    std::printf("link: %zu of %zu files arrived byte for byte; %zu in sent/; sender bulk tensors sent %llu, refused "
                "(retried) %llu, dropped %llu\n", visible_files(dst), files.size(), visible_files(src / "sent"),
                (unsigned long long)dv[6], (unsigned long long)dv[9], (unsigned long long)dv[10]);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: file_link_test <file_read> <file_write> <tensor_mint> <tensor_returns> <work dir>\n");
        return 2;
    }
    char read_id[256] = {}, write_id[256] = {}, mint_id[256] = {}, returns_id[256] = {};
    if (!gp_plugin_load_from_path(argv[1], read_id, sizeof(read_id)) || !gp_plugin_load_from_path(argv[2], write_id, sizeof(write_id)) ||
        !gp_plugin_load_from_path(argv[3], mint_id, sizeof(mint_id)) || !gp_plugin_load_from_path(argv[4], returns_id, sizeof(returns_id))) {
        std::fprintf(stderr, "cannot load the file rows\n");
        return 2;
    }
    const fs::path dir = fs::path(argv[5]) / ("file_link_test_" + std::to_string(std::time(nullptr)));
    fs::create_directories(dir);
    local_case(read_id, write_id, dir);
    link_case(read_id, write_id, mint_id, returns_id, dir);
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
