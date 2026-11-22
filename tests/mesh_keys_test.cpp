// mesh_keys_test: the mesh's keys in one process (src/headless/nodus_mesh_crypto.cpp; KEY_CHANGE_PROTOCOL.md).
//
//   rows      A seals a snapshot to B and signs it; B verifies and opens it (readable), C opens it (not a reader); a
//             body changed in flight, or an envelope from a key B does not know, is refused
//   rotation  stopped after its key set, then after its record (drills of an interruption): `current` still names the
//             old whole key set each time, and a thread signing the whole while gets a valid signature every call;
//             run again, it finishes with the same record. B accepts the record and verifies A's next epoch as A
//   accept    interrupted before its marker (the marker removed by hand) is applied again; a replay is refused
//   suites    a force record allowing only suite 2 refuses every suite-1 key (A's envelopes, sealing to A); a newer
//             one allowing suite 1 again restores it; an older one is refused
//
//   mesh_keys_test <work dir>
#include "nodus_headless_abi.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Bytes = std::vector<uint8_t>;

namespace {

int failures = 0;
#define CHECK(cond)                                                                                  \
    do {                                                                                             \
        if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

std::mutex g_env;   // NODUS_MESH_KEYS is process state: one node at a time
void set_env(const std::string& value) {
#if defined(_WIN32)
    _putenv_s("NODUS_MESH_KEYS", value.c_str());
#else
    setenv("NODUS_MESH_KEYS", value.c_str(), 1);
#endif
}
Bytes read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return Bytes(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}
std::string read_text(const fs::path& p) {
    const Bytes b = read_file(p);
    std::string s(b.begin(), b.end());
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}
std::string hex(const Bytes& b) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (uint8_t c : b) { s += d[c >> 4]; s += d[c & 15]; }
    return s;
}
bool op(std::vector<std::string> args, std::string* said = nullptr) {
    std::vector<const char*> a;
    for (const auto& x : args) a.push_back(x.c_str());
    char report[1024] = {};
    const int ok = nodus_mesh_operator(static_cast<int32_t>(a.size()), a.data(), report, sizeof(report));
    if (said) *said = report;
    std::printf("  %s %s: %s\n", args[0].c_str(), ok ? "done" : "refused", report);
    return ok == 1;
}
// One row step as `node`: 1 and the result, or 0 (refused).
bool row(const fs::path& node, int32_t which, const Bytes& in, Bytes& out) {
    std::lock_guard<std::mutex> lock(g_env);
    set_env(node.string());
    out.resize(static_cast<size_t>(nodus_mesh_bound(which, in.data(), in.size())));
    size_t written = 0;
    const bool ok = nodus_mesh_row(which, in.data(), in.size(), out.data(), out.size(), &written) == 1;
    out.resize(ok ? written : 0);
    return ok;
}
Bytes current_block(const fs::path& node) { return read_file(node / "epochs" / read_text(node / "current") / "node.pub"); }
Bytes unhex(const std::string& s) {
    Bytes out;
    for (size_t i = 0; i + 1 < s.size(); i += 2) out.push_back(static_cast<uint8_t>(std::stoi(s.substr(i, 2), nullptr, 16)));
    return out;
}
std::string word_after(const std::string& text, const std::string& word) {
    const size_t at = text.find(word + " ");
    if (at == std::string::npos) return {};
    const size_t start = at + word.size() + 1;
    return text.substr(start, text.find_first_of(" ;", start) - start);
}
void know(const fs::path& who, const fs::path& whom, const std::string& kid_hex) {
    fs::copy_file(whom / "epochs" / read_text(whom / "current") / "node.pub", who / "peers" / (kid_hex + ".pub"),
                  fs::copy_options::overwrite_existing);
}
Bytes request(const std::vector<Bytes>& readers, const std::string& snapshot) {
    Bytes r = {'N', 'R', 'S', '1'};
    const uint32_t n = static_cast<uint32_t>(readers.size());
    r.insert(r.end(), reinterpret_cast<const uint8_t*>(&n), reinterpret_cast<const uint8_t*>(&n) + 4);
    for (const Bytes& l : readers) r.insert(r.end(), l.begin(), l.end());
    r.insert(r.end(), snapshot.begin(), snapshot.end());
    return r;
}
// A writes `snapshot` for `readers`; `reader` takes it: verified and opened. readable: 1, 0 not a reader, -1 refused.
int carry(const fs::path& a, const fs::path& reader, const std::vector<Bytes>& readers, const std::string& snapshot,
          std::string* opened = nullptr, Bytes* envelope_out = nullptr) {
    Bytes sealed, env, verified, out;
    if (!row(a, NODUS_MESH_SEAL, request(readers, snapshot), sealed) || sealed.empty()) return -2;
    if (!row(a, NODUS_MESH_SIGN, sealed, env) || env.empty()) return -3;
    if (envelope_out) *envelope_out = env;
    if (!row(reader, NODUS_MESH_VERIFY, env, verified) || verified.empty()) return -1;
    if (!row(reader, NODUS_MESH_OPEN, verified, out) || out.size() < 4 + 32 + 8 + 1) return -1;
    if (opened) opened->assign(out.begin() + 45, out.end());
    return out[44];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: mesh_keys_test <work dir>\n"); return 2; }
    if (!nodus_mesh_available()) { std::fprintf(stderr, "this Nodus was built without the pinned OpenSSL\n"); return 2; }
    const fs::path work = fs::path(argv[1]) / ("mesh_keys_test_" + std::to_string(std::time(nullptr)));
    fs::create_directories(work);
    const fs::path A = work / "A", B = work / "B", C = work / "C", ROOT = work / "ROOT";
    std::string said;
    std::string kid[3];
    const fs::path nodes[3] = {A, B, C};
    for (int i = 0; i < 3; ++i) {
        CHECK(op({"keygen", nodes[i].string()}, &said));
        kid[i] = word_after(said, "kid");
        CHECK(kid[i].size() == 64);
    }
    CHECK(!op({"keygen", A.string()}));                                    // keys are never replaced in place
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) if (i != j) know(nodes[i], nodes[j], kid[j]);
    const Bytes lineage_a = unhex(kid[0]), lineage_b = unhex(kid[1]);

    // rows
    std::string opened;
    Bytes env;
    CHECK(carry(A, B, {lineage_b}, "the canvas", &opened, &env) == 1 && opened == "the canvas");
    CHECK(carry(A, C, {lineage_b}, "the canvas") == 0);                   // C: not a reader
    std::printf("rows: B opens A's canvas, C is not a reader\n");
    Bytes bad = env, out;
    bad[4 + 2 + 32 + 16 + 3] ^= 1;                                         // a body byte changed in flight
    CHECK(row(B, NODUS_MESH_VERIFY, bad, out) && out.empty());
    CHECK(op({"keygen", (work / "D").string()}));
    Bytes sealed, foreign;
    CHECK(row(work / "D", NODUS_MESH_SEAL, request({}, "x"), sealed) && row(work / "D", NODUS_MESH_SIGN, sealed, foreign));
    CHECK(row(B, NODUS_MESH_VERIFY, foreign, out) && out.empty());        // D's key: unknown to B
    std::printf("rows: a changed body and an unknown key are refused\n");

    // rotation, interrupted at each stage while another thread signs the whole time
    std::atomic<bool> stop{false};
    std::atomic<int> signed_ok{0}, signed_bad{0};
    std::thread signer([&] {
        const Bytes body = {'N', 'M', 'C', '0', 'x'};
        while (!stop) {
            Bytes e;
            if (row(A, NODUS_MESH_SIGN, body, e) && !e.empty()) ++signed_ok; else ++signed_bad;
        }
    });
    CHECK(op({"rotate", A.string(), "600", "--drill-stop=1"}));
    CHECK(read_text(A / "current") == "0" && fs::exists(A / "epochs" / "1" / "node.pub"));
    CHECK(op({"rotate", A.string(), "600", "--drill-stop=2"}));
    CHECK(read_text(A / "current") == "0" && fs::exists(A / "rotation-1.rec"));
    const Bytes staged_record = read_file(A / "rotation-1.rec");
    CHECK(op({"rotate", A.string(), "600"}));
    CHECK(read_text(A / "current") == "1" && read_file(A / "rotation-1.rec") == staged_record);
    for (int i = 0; i < 3; ++i) CHECK(op({"rotate", A.string(), "600"}));   // more, still under the signer
    stop = true;
    signer.join();
    std::printf("rotation: resumed after each stage; %d signatures under rotation, %d failed\n", signed_ok.load(), signed_bad.load());
    CHECK(signed_ok > 0 && signed_bad == 0);

    // accept: B takes the epoch-1 record; interrupted before its marker, it is applied again; then a replay is refused
    const fs::path rec1 = A / "rotation-1.rec";
    CHECK(op({"accept", B.string(), rec1.string()}));
    fs::remove(B / "records" / ("rotation-" + kid[0]));                    // as if the accept stopped before its marker
    CHECK(op({"accept", B.string(), rec1.string()}));
    CHECK(!op({"accept", B.string(), rec1.string()}));                    // the replay
    for (int e = 2; e <= 4; ++e) CHECK(op({"accept", B.string(), (A / ("rotation-" + std::to_string(e) + ".rec")).string()}));
    CHECK(carry(A, B, {lineage_b}, "after rotations", &opened) == 1 && opened == "after rotations");
    Bytes v;
    CHECK(row(A, NODUS_MESH_SIGN, Bytes{'N', 'M', 'C', '0'}, env) && row(B, NODUS_MESH_VERIFY, env, v) && v.size() >= 36 &&
          Bytes(v.begin() + 4, v.begin() + 36) == lineage_a);              // epoch 4 is still A
    std::printf("accept: retried after an interruption, a replay refused; A's epoch 4 is A to B\n");

    // suites: a forced crypto change under the site root
    CHECK(op({"root-keygen", ROOT.string()}));
    fs::copy_file(ROOT / "epochs" / "0" / "node.pub", B / "root.pub");
    const fs::path f1 = work / "force-1.rec", f2 = work / "force-2.rec";
    CHECK(op({"force", ROOT.string(), f1.string(), "--suites", "2"}));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(op({"force", ROOT.string(), f2.string(), "--suites", "1"}));
    CHECK(op({"accept", B.string(), f1.string()}));
    CHECK(read_text(B / "suites") == "2");
    CHECK(carry(A, B, {lineage_b}, "suite 1") == -1);                     // A's suite-1 envelope: refused
    CHECK(row(B, NODUS_MESH_SEAL, request({lineage_a}, "to A"), sealed) && sealed.empty());   // nor sealed to it
    CHECK(op({"accept", B.string(), f2.string()}));
    CHECK(!op({"accept", B.string(), f1.string()}));                      // the older one, again: refused
    CHECK(carry(A, B, {lineage_b}, "suite 1 again", &opened) == 1 && opened == "suite 1 again");
    std::printf("suites: suite 1 refused under a force allowing only 2, back under a newer one; the older refused\n");

    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
