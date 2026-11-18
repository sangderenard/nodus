// link_settings_test: compiler settings and shader builds over the Nodus link (volunteer case
// a11b5c9b1be74c9b9fb684790f7fd5a3, stage 2), on this machine's real devices.
//   link_settings_test <shader product dir: spring_network_step.comp + spring_network_step.glsl.json> <work dir>
// The volunteer sends its SettingsRecord; the server builds the GLSL with glslang for each device whose settings meet
// the product's declared needs and sends the GLSL and that Vulkan build; the volunteer rebuilds locally when it has
// glslang (and must get the same bytes), or takes the server's build on its digest when it has no toolchain.
#include "nodus_link_settings.h"

#include <chrono>
#include <cstdio>
#include <ctime>

using namespace nodus::link;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

namespace {
std::vector<uint8_t> read_file(const fs::path& p) { std::ifstream f(p, std::ios::binary); return {std::istreambuf_iterator<char>(f), {}}; }
// The two declared fields of the product record this test needs (the record is Turing's own JSON).
ShaderNeeds needs_of(const fs::path& json) {
    auto b = read_file(json); std::string s(b.begin(), b.end());
    ShaderNeeds n;
    n.float64 = s.find("\"numeric_dtype\": \"float64\"") != std::string::npos;
    const auto at = s.find("\"local_size_x\":");
    if (at != std::string::npos) n.local_size_x = uint32_t(std::strtoul(s.c_str() + at + 15, nullptr, 10));
    return n;
}
std::string hex8(const std::array<uint8_t, 32>& d) { char b[17]; for (int i = 0; i < 8; ++i) std::snprintf(b + 2 * i, 3, "%02x", d[i]); return b; }
}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::printf("usage: link_settings_test <shader product dir> <work dir>\n"); return 2; }
    const fs::path product = argv[1], work = argv[2];
    int failures = 0;
    auto check = [&](bool ok, const std::string& what) { if (!ok) { ++failures; std::printf("FAIL %s\n", what.c_str()); } };
    const auto glsl = read_file(product / "spring_network_step.comp");
    const ShaderNeeds needs = needs_of(product / "spring_network_step.glsl.json");
    check(!glsl.empty() && needs.local_size_x, "shader product readable");
    std::printf("shader product: GLSL %zu bytes, declares float64=%d local_size_x=%u\n", glsl.size(), int(needs.float64), needs.local_size_x);

    Key key{}; random_bytes(key.data(), key.size());
    ListenerConfig lc; lc.prefix = "settings"; lc.key = key;
    LinkListener listener(lc);
    const auto id = new_link_id(); const uint64_t now = uint64_t(std::time(nullptr));
    DialTarget t; t.port = listener.port(); t.prefix = "settings";
    LinkDialer dialer(t, sign_ticket(make_ticket(Channel::Control, Role::Dialer, id, "settings", 42, now, 60), key),
                      sign_ticket(make_ticket(Channel::Bulk, Role::Dialer, id, "settings", 42, now, 60), key));
    std::unique_ptr<Link> server, volunteer;
    for (auto end = Clock::now() + std::chrono::seconds(5); Clock::now() < end && !(server && volunteer);) {
        dialer.poll(); listener.poll(uint64_t(std::time(nullptr)));
        if (!volunteer && dialer.ready()) volunteer = dialer.take();
        if (!server) server = listener.take();
        if (volunteer) volunteer->poll();
    }
    check(server && volunteer, "link up");
    if (!server || !volunteer) return 1;
    BulkMux server_bulk(*server), volunteer_bulk(*volunteer);
    auto pump = [&] { server->poll(); volunteer->poll(); server_bulk.pump(); volunteer_bulk.pump(); };

    for (int pass = 0; pass < 2; ++pass) {
        const bool toolchain = pass == 0;                       // pass 1: a volunteer without glslang
        // 1. The volunteer sends its settings.
        SettingsRecord mine = collect_settings();
        if (!toolchain) { mine.has_glslang = false; mine.glslang_path.clear(); }
        volunteer_bulk.send(BulkStreamKind::Settings, 1, encode_settings(mine));
        TypedStream got;
        for (auto end = Clock::now() + std::chrono::seconds(10); Clock::now() < end && !server_bulk.take(BulkStreamKind::Settings, got);) pump();
        SettingsRecord theirs;
        check(decode_settings(got.bytes, theirs), "settings record decodes");
        if (pass == 0) {
            std::printf("volunteer settings: %s (%s), %u threads, features 0x%llx, glslang %s, clang %s\n", theirs.cpu_brand.c_str(),
                        theirs.cpu_triple.c_str(), theirs.cpu_threads, (unsigned long long)theirs.cpu_features,
                        theirs.has_glslang ? "yes" : "no", theirs.has_clang ? "yes" : "no");
            for (size_t i = 0; i < theirs.gpus.size(); ++i) {
                const auto& g = theirs.gpus[i];
                std::printf("  gpu %zu: %s, Vulkan %u.%u (target %s), float64 %s, work-group %u (x %u), subgroup %u\n", i, g.name.c_str(),
                            g.api_version >> 22, (g.api_version >> 12) & 0x3FF, vulkan_target(g.api_version).c_str(), g.float64 ? "yes" : "no",
                            g.max_invocations, g.max_size[0], g.subgroup);
            }
        }
        // 2. The server builds for every device that meets the declared needs, and sends GLSL + Vulkan.
        GlslangBuilds server_builds(detail::find_on_path("glslangValidator.exe"), work / "server");
        size_t offered = 0, refused = 0;
        for (size_t i = 0; i < theirs.gpus.size(); ++i) {
            if (const char* why = unmet_need(theirs.gpus[i], needs)) {
                ++refused; if (pass == 0) std::printf("  gpu %zu not offered: %s\n", i, why); continue;
            }
            std::vector<uint8_t> spirv; bool failed = false;
            const std::string target = vulkan_target(theirs.gpus[i].api_version);
            for (auto end = Clock::now() + std::chrono::seconds(30); Clock::now() < end && !server_builds.build(glsl, target, spirv, failed) && !failed;) pump();
            check(!failed && !spirv.empty(), "server build for gpu " + std::to_string(i));
            const auto gd = sha256(glsl.data(), glsl.size()), vd = sha256(spirv.data(), spirv.size());
            std::vector<uint8_t> aux(target.begin(), target.end()); aux.resize(16, 0);
            aux.insert(aux.end(), gd.begin(), gd.end()); aux.insert(aux.end(), vd.begin(), vd.end());
            server_bulk.send(BulkStreamKind::Glsl, i, glsl);
            server_bulk.send(BulkStreamKind::Vulkan, i, spirv, aux);
            ++offered;
        }
        std::printf("pass %d (%s): server built %zu device(s) (%llu glslang run(s), %zu cached), %zu not offered\n", pass,
                    toolchain ? "volunteer has glslang" : "volunteer without a toolchain", offered,
                    (unsigned long long)server_builds.started(), server_builds.cached(), refused);
        // 3. The volunteer receives each pair; it builds locally when it can, else takes the server's build on its digest.
        GlslangBuilds local_builds(mine.glslang_path, work / (toolchain ? "volunteer" : "volunteer-none"));
        size_t accepted = 0, matched = 0;
        for (size_t k = 0; k < offered; ++k) {
            TypedStream source, build;
            for (auto end = Clock::now() + std::chrono::seconds(20); Clock::now() < end && !(source.bytes.size() && build.bytes.size());) {
                pump();
                if (source.bytes.empty()) volunteer_bulk.take(BulkStreamKind::Glsl, source);
                if (build.bytes.empty()) volunteer_bulk.take(BulkStreamKind::Vulkan, build);
            }
            check(source.id == build.id && build.aux.size() == 80, "a GLSL and its Vulkan build for one device");
            if (build.aux.size() != 80) continue;
            const std::string target = detail::get_str(build.aux.data(), 16);
            std::array<uint8_t, 32> gd{}, vd{}; std::memcpy(gd.data(), build.aux.data() + 16, 32); std::memcpy(vd.data(), build.aux.data() + 48, 32);
            const bool digests = sha256(source.bytes.data(), source.bytes.size()) == gd && sha256(build.bytes.data(), build.bytes.size()) == vd;
            check(digests, "offered products match their digests");
            if (mine.has_glslang) {
                std::vector<uint8_t> own; bool failed = false;
                for (auto end = Clock::now() + std::chrono::seconds(30); Clock::now() < end && !local_builds.build(source.bytes, target, own, failed) && !failed;) pump();
                const bool same = !failed && own == build.bytes;
                check(same, "the volunteer's own glslang build equals the server's for gpu " + std::to_string(source.id));
                if (same) ++matched;
                std::printf("  gpu %llu: built locally for %s, %zu bytes, sha256 %s..., %s the server's\n", (unsigned long long)source.id,
                            target.c_str(), own.size(), hex8(sha256(own.data(), own.size())).c_str(), same ? "identical to" : "DIFFERENT from");
            } else {
                std::printf("  gpu %llu: no local toolchain; took the server's %s build on its digest %s...\n", (unsigned long long)source.id,
                            target.c_str(), hex8(vd).c_str());
            }
            if (digests) ++accepted;
        }
        check(accepted == offered, "every offered build accepted");
        if (toolchain) check(matched == offered, "every local build matched");
    }
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
