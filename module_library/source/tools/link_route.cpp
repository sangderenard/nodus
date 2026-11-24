// link_route: the row "nodus.link.route". Its one duty is this host's front route for one hosted link: the line
// "<name> 127.0.0.1:<port>" in the front's route map, a file the Apache front reads per connection
// (RewriteMap txt:, owner 2026-11-23: "a connector negotiating other ports which i guess is exactly what apache does").
// Each link listener binds a port of its own on loopback (ListenerConfig.port 0: the system picks it); the front routes
// wss://<host>/ws/link/<name>/{control,bulk} to that port by this line, so many links share the one public port and no
// link has to take another's. Apache only carries the socket: the listener still admits a link by its tickets alone.
//
// The row only READS the link (link_endpoint_report in the image, nodus_headless_link_report as a plugin DLL), as
// link_metrics does; it never sends on, closes or alters it. The route map is its file destination (owner, 2026-11-21:
// file I/O belongs to a row whose purpose it is): it writes only when the link's bound port differs from the line it
// last wrote, and on shutdown it takes its line out again. Every other line of the map (other links' rows, comments) is
// kept as it is.
//
// Several processes share one map. A writer holds "<map>.lock" (created exclusively; one older than 10 s is a crashed
// writer's and is removed), writes "<map>.<random>.tmp" and renames it over the map, so a reader sees the old map or the
// new one, never a part. A lock it cannot take now is tried again on the next execute; nothing here waits.
//
// Settings are saved state ("NLR1" version 1: link id, route name, map path). The route name is what the front
// matches, [A-Za-z0-9._-]{1,64}, the same characters its RewriteRule takes; a name outside them is refused.
#include "nodus_link_rows.h"
#include "nodus_headless_abi.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace nodus::link {

namespace fs = std::filesystem;

class LinkRouteRow final : public ITool {
public:
    // The generic row: its link, name and map come from saved state.
    LinkRouteRow() = default;
    // The row bound to one hosted link at registration ("<link id>.route", nodus_link_rows.cpp), as <id>.metrics is.
    LinkRouteRow(std::string row_id, std::string link_id, std::string name, std::string map)
        : row_id_(std::move(row_id)), link_id_(std::move(link_id)), name_(std::move(name)), map_(std::move(map)) {
        if (!valid_name(name_)) name_.clear();   // refused: the row then does nothing
    }
    static bool valid_name(const std::string& n) {
        if (n.empty() || n.size() > 64) return false;
        for (char c : n)
            if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
                return false;
        return true;
    }

    bool serialize(OutputArchive& out) const override {
        std::lock_guard<std::mutex> g(mutex_);
        const uint32_t head[5] = {kMagic, kVersion, uint32_t(link_id_.size()), uint32_t(name_.size()), uint32_t(map_.size())};
        return out.write(head, sizeof(head)) && out.write(link_id_.data(), link_id_.size()) &&
               out.write(name_.data(), name_.size()) && out.write(map_.data(), map_.size());
    }
    bool deserialize(InputArchive& in) override {
        uint32_t head[5] = {};
        if (!in.read(head, sizeof(head)) || head[0] != kMagic || head[1] != kVersion || head[2] > 4096 || head[3] > 64 ||
            head[4] == 0 || head[4] > 4096) return false;
        std::string id(head[2], '\0'), name(head[3], '\0'), map(head[4], '\0');
        if (!in.read(id.data(), id.size()) || !in.read(name.data(), name.size()) || !in.read(map.data(), map.size())) return false;
        if (!valid_name(name)) { std::fprintf(stderr, "nodus.link.route: route name '%s' refused\n", name.c_str()); return false; }
        std::lock_guard<std::mutex> g(mutex_);
        link_id_ = std::move(id); name_ = std::move(name); map_ = std::move(map); written_ = 0;
        return true;
    }
    const char* id_cstr() const noexcept override { return row_id_.c_str(); }
    const char* name_cstr() const noexcept override { return "Link route"; }
    void initialize(const ToolInitContext& ctx) override {
        if (ctx.serialized_path && ctx.serialized_path[0] != '\0') {
            FileInputArchive in(ctx.serialized_path);
            if (in.ok()) deserialize(in);
        }
    }
    // Its line goes when the row does: a route to a listener that is gone would send the next dialer to a closed port.
    // The graph deletes a plugin row without calling shutdown(), so the destructor takes the line out as well.
    void shutdown() override { withdraw(); }
    ~LinkRouteRow() noexcept override {
        try { withdraw(); } catch (...) { std::fprintf(stderr, "nodus.link.route: route line not withdrawn\n"); }
    }
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    // It pops and pushes nothing.
    void execute_stack(ToolStackContext& context) override {
        std::lock_guard<std::mutex> g(mutex_);
        context.succeeded = false;
        if (link_id_.empty() || name_.empty() || map_.empty()) return;
        int32_t state = 0; uint64_t c[16] = {};
#if defined(NODUS_PLUGIN_COMPOSITE)
        EndpointReport report;
        if (!link_endpoint_report(link_id_, &report)) return;
        link_report_counters(report, c);
#else
        if (!nodus_headless_link_report(link_id_.c_str(), &state, c)) return;
#endif
        const uint16_t port = uint16_t(c[0] <= 65535 ? c[0] : 0);   // listen_port: 0 for a link that dials
        context.succeeded = true;
        if (port != written_) (void)rewrite(port);
    }

private:
    static constexpr uint32_t kMagic = 0x31524C4E, kVersion = 1;   // "NLR1"

    // Takes this row's line out, waiting up to 1 s for another writer's lock.
    void withdraw() {
        std::lock_guard<std::mutex> g(mutex_);
        for (int tries = 0; written_ && tries < 50; ++tries) {
            if (rewrite(0)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    // The map with this row's line set to `port` (0: removed). Caller holds mutex_. False: not now (the lock is held or a
    // file operation failed); the next execute tries again.
    bool rewrite(uint16_t port) {
        const fs::path map(map_), lock(map_ + ".lock");
        std::error_code ec;
        FILE* held = std::fopen(lock.string().c_str(), "wx");
        if (!held) {
            const auto age = fs::file_time_type::clock::now() - fs::last_write_time(lock, ec);
            if (!ec && age > std::chrono::seconds(10)) fs::remove(lock, ec);   // a crashed writer's lock
            return false;
        }
        std::fclose(held);
        bool ok = false;
        {
            std::vector<std::string> keep;
            std::ifstream in(map, std::ios::binary);
            for (std::string line; std::getline(in, line);) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                std::istringstream words(line);
                std::string first;
                words >> first;
                if (first == name_) continue;   // this row's own line, written again below
                keep.push_back(line);
            }
            in.close();
            if (keep.empty()) keep.push_back("# Nodus link routes: <name> <loopback host:port>, one line per hosted link (nodus.link.route).");
            if (port) keep.push_back(name_ + " 127.0.0.1:" + std::to_string(port));
            std::random_device rd;
            const fs::path tmp = map_ + "." + std::to_string(rd()) + ".tmp";
            {
                std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
                for (const auto& line : keep) out << line << '\n';
                out.flush();
                ok = bool(out);
            }
            if (ok) { fs::rename(tmp, map, ec); ok = !ec; }
            if (!ok) fs::remove(tmp, ec);
        }
        fs::remove(lock, ec);
        if (ok) written_ = port;
        else std::fprintf(stderr, "nodus.link.route: route map %s not written\n", map_.c_str());
        return ok;
    }

    std::string row_id_ = "nodus.link.route";
    mutable std::mutex mutex_;
    std::string link_id_, name_, map_;
    uint16_t written_ = 0;   // the port of the line this row last wrote; 0: none
};

}  // namespace nodus::link

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::link::LinkRouteRow(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
