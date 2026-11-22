// The Nodus link's registration and endpoint table (include/nodus_link_rows.h). The rows themselves are the plugin
// source module_library/source/tools/nodus_link.cpp, compiled into this image the way a composed plugin source is:
// NODUS_PLUGIN_COMPOSITE with its exports renamed (as src/tcp_byte_stream_tool.cpp does for its row).
// Research case 98d49cb8520e44d092c5d134c37c3c32, stage 2.
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_link_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_link_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_link_source
#include "../module_library/source/tools/nodus_link.cpp"
// The metrics row (module_library/source/tools/link_metrics.cpp), likewise: registered per link as <id>.metrics.
#undef NODUS_PLUGIN_FACTORY_NAME
#undef NODUS_PLUGIN_DESTROY_NAME
#undef NODUS_PLUGIN_SOURCE_NAME
#define NODUS_PLUGIN_FACTORY_NAME nodus_link_metrics_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_link_metrics_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_link_metrics_source
#include "../module_library/source/tools/link_metrics.cpp"

namespace nodus::link {
namespace {

std::mutex& endpoints_mutex() { static std::mutex m; return m; }
std::unordered_map<std::string, std::shared_ptr<LinkEndpoint>>& endpoints() {
    static std::unordered_map<std::string, std::shared_ptr<LinkEndpoint>> e; return e;
}
std::unordered_map<std::string, int32_t>& metrics_channels() {   // guarded by endpoints_mutex()
    static std::unordered_map<std::string, int32_t> c; return c;
}
int32_t metrics_channel_of(const std::string& id) {
    std::lock_guard<std::mutex> g(endpoints_mutex());
    auto it = metrics_channels().find(id);
    return it == metrics_channels().end() ? 0 : it->second;
}

bool register_rows(ToolRegistry& registry, const std::string& id, std::shared_ptr<LinkEndpoint> ep, std::string* error) {
    {
        std::lock_guard<std::mutex> g(endpoints_mutex());
        if (endpoints().count(id)) { if (error) *error = "link " + id + " is already registered"; return false; }
    }
    constexpr LinkRowRole kRoles[4] = {LinkRowRole::ControlSend, LinkRowRole::ControlReceive, LinkRowRole::BulkSend,
                                       LinkRowRole::BulkReceive};
    for (size_t r = 0; r < 4; ++r) {
        const LinkRowRole role = kRoles[r];
        ToolRegistry::Entry entry;
        entry.id = id + link_row_suffix(role);
        entry.name = LinkRow(std::string(), role, nullptr).name_cstr();
        entry.caps = ToolCaps::None;
        entry.source_path = nodus_link_source();
        entry.factory = [rid = entry.id, role, ep] {
            return std::unique_ptr<ITool, std::function<void(ITool*)>>(new LinkRow(rid, role, ep), [](ITool* t) { nodus_link_destroy(t); });
        };
        if (!registry.register_tool(std::move(entry))) {
            for (size_t k = 0; k < r; ++k) registry.unregister_tool(id + link_row_suffix(kRoles[k]));
            if (error) *error = "tool registry refused " + id + link_row_suffix(role);
            return false;
        }
    }
    {   // The metrics row bound to this link, and the generic one (a canvas-loaded copy takes its link from saved state).
        ToolRegistry::Entry entry;
        entry.id = id + ".metrics"; entry.name = "Link metrics"; entry.caps = ToolCaps::Table;
        entry.source_path = nodus_link_metrics_source();
        entry.factory = [rid = entry.id, id] {
            return std::unique_ptr<ITool, std::function<void(ITool*)>>(new LinkMetricsRow(rid, id, LinkMetricsRow::kDefaultWindow, metrics_channel_of(id)), [](ITool* t) { nodus_link_metrics_destroy(t); });
        };
        if (!registry.register_tool(std::move(entry))) {
            for (LinkRowRole role : kRoles) registry.unregister_tool(id + link_row_suffix(role));
            if (error) *error = "tool registry refused " + id + ".metrics";
            return false;
        }
        ToolRegistry::Entry generic;
        generic.id = "nodus.link.metrics"; generic.name = "Link metrics"; generic.caps = ToolCaps::Table;
        generic.source_path = nodus_link_metrics_source();
        generic.factory = [] {
            return std::unique_ptr<ITool, std::function<void(ITool*)>>(new LinkMetricsRow(), [](ITool* t) { nodus_link_metrics_destroy(t); });
        };
        registry.register_tool(std::move(generic));   // already there from an earlier link: fine
    }
    std::lock_guard<std::mutex> g(endpoints_mutex());
    endpoints()[id] = std::move(ep);
    return true;
}

}  // namespace

bool register_link_host(ToolRegistry& registry, LinkHostConfig config, uint16_t* listen_port, std::string* error) {
    if (config.id.empty() || config.prefix.empty() || !config.maximum_bulk_bytes) { if (error) *error = "id, prefix and size are required"; return false; }
    std::shared_ptr<LinkEndpoint> ep;
    std::string id = config.id;
    try { ep = std::make_shared<LinkEndpoint>(std::move(config)); }
    catch (const std::exception& e) { if (error) *error = e.what(); return false; }
    if (listen_port) *listen_port = ep->listen_port();
    return register_rows(registry, id, std::move(ep), error);
}

bool register_link_dialer(ToolRegistry& registry, LinkDialConfig config, std::string* error) {
    const bool target_ok = config.udp ? !config.udp_target.broker_host.empty() && !config.udp_target.prefix.empty() && config.udp_target.broker_port
                         : config.wss ? !config.wss_target.host.empty() && !config.wss_target.prefix.empty() && config.wss_target.port
                                      : !config.target.prefix.empty() && config.target.port;
    if (config.id.empty() || !target_ok || !config.maximum_bulk_bytes) {
        if (error) *error = "id, prefix, port and size are required"; return false;
    }
    std::shared_ptr<LinkEndpoint> ep;
    std::string id = config.id;
    try { ep = std::make_shared<LinkEndpoint>(std::move(config)); }
    catch (const std::exception& e) { if (error) *error = e.what(); return false; }
    return register_rows(registry, id, std::move(ep), error);
}

bool link_endpoint_report(const std::string& id, EndpointReport* out) {
    std::shared_ptr<LinkEndpoint> ep;
    {
        std::lock_guard<std::mutex> g(endpoints_mutex());
        auto it = endpoints().find(id); if (it == endpoints().end()) return false;
        ep = it->second;
    }
    std::lock_guard<std::mutex> g(ep->mutex());
    if (out) *out = ep->report();
    return true;
}

void set_link_metrics_channel(const std::string& id, int32_t channel) {
    std::lock_guard<std::mutex> g(endpoints_mutex());
    if (channel > 0) metrics_channels()[id] = channel; else metrics_channels().erase(id);
}

void unregister_link(ToolRegistry& registry, const std::string& id) {
    for (LinkRowRole role : {LinkRowRole::ControlSend, LinkRowRole::ControlReceive, LinkRowRole::BulkSend, LinkRowRole::BulkReceive})
        registry.unregister_tool(id + link_row_suffix(role));
    registry.unregister_tool(id + ".metrics");
    std::lock_guard<std::mutex> g(endpoints_mutex());
    endpoints().erase(id);
    metrics_channels().erase(id);
}

}  // namespace nodus::link
