#include "plugin_loader.h"
#include "tool_registry.h"
#include "tool_api.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <iostream>
#include <sstream>

struct LoadedModuleInfo {
#ifdef _WIN32
    HMODULE handle = nullptr;
#endif
    std::string path;
    std::string base_id;
    std::string unique_id;
    std::string version_str;
    // store function pointers if available
    using destroy_fn_t = void (*)(ITool*);
    using plugin_init_fn_t = int (*)(HostAPI*);
    using plugin_shutdown_fn_t = void (*)();
    destroy_fn_t destroy_fn = nullptr;
    plugin_init_fn_t plugin_init = nullptr;
    plugin_shutdown_fn_t plugin_shutdown = nullptr;
};

PluginLoader::PluginLoader() {}

PluginLoader::~PluginLoader() {
    // Unload all modules
    for (auto it = modules_.begin(); it != modules_.end(); ) {
        unload_module(it->first);
        it = modules_.begin();
    }
}

#include <chrono>
#include <random>

static std::string make_version_tag() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    char buf[64];
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    // append a short random hex to avoid collisions within same second
    std::random_device rd;
    uint32_t v = rd();
    std::ostringstream ss;
    ss << buf << "_" << std::hex << (v & 0xFFFF);
    return ss.str();
}

std::string PluginLoader::load_module(const std::string& path, void* host) {
#ifdef _WIN32
    HMODULE h = LoadLibraryA(path.c_str());
    if (!h) {
        std::cerr << "LoadLibrary failed: " << GetLastError() << "\n";
        return {};
    }

    using create_fn_t = ITool* (*)();
    create_fn_t create_fn = (create_fn_t)GetProcAddress(h, "create_tool");
    LoadedModuleInfo* info = new LoadedModuleInfo();
    info->handle = h;
    info->path = path;

    if (!create_fn) {
        std::cerr << "create_tool not found in " << path << "\n";
        FreeLibrary(h);
        delete info;
        return {};
    }

    info->destroy_fn = (LoadedModuleInfo::destroy_fn_t)GetProcAddress(h, "destroy_tool");
    info->plugin_init = (LoadedModuleInfo::plugin_init_fn_t)GetProcAddress(h, "plugin_init");
    info->plugin_shutdown = (LoadedModuleInfo::plugin_shutdown_fn_t)GetProcAddress(h, "plugin_shutdown");

    // instantiate temporary tool to discover id/name/caps
    ITool* tmp = create_fn();
    if (!tmp) {
        std::cerr << "create_tool returned null in " << path << "\n";
        if (info->destroy_fn) info->destroy_fn(tmp);
        FreeLibrary(h);
        delete info;
        return {};
    }

    std::string base_id = tmp->id();
    std::string name = tmp->name();
    ToolCaps caps = tmp->caps();

    std::cerr << "DEBUG: plugin_loader: discovered base_id='" << base_id << "' name='" << name << "' caps=" << static_cast<int>(caps) << " from '" << path << "'\n";

    // destroy temporary instance
    if (info->destroy_fn) info->destroy_fn(tmp); else delete tmp;

    // prepare factory that uses the DLL's create/destroy
    ToolRegistry::Entry entry;
    // create a unique id for this loaded module so multiple versions can coexist
    std::string ver = make_version_tag();
    std::string unique_id = base_id + "_ver_" + ver;
    info->base_id = base_id;
    info->unique_id = unique_id;
    info->version_str = ver;

    entry.id = unique_id;
    entry.name = (name.empty() ? base_id : name) + std::string(" [v") + ver + "]";
    entry.caps = caps;
    if (create_fn) {
        auto dfn = info->destroy_fn;
        if (dfn) {
            entry.factory = [create_fn, dfn]() -> std::unique_ptr<ITool, std::function<void(ITool*)>> {
                std::function<void(ITool*)> del = [dfn](ITool* p){ dfn(p); };
                return std::unique_ptr<ITool, std::function<void(ITool*)>>(create_fn(), del);
            };
        } else {
            entry.factory = [create_fn]() -> std::unique_ptr<ITool, std::function<void(ITool*)>> {
                std::function<void(ITool*)> del = [](ITool* p){ delete p; };
                return std::unique_ptr<ITool, std::function<void(ITool*)>>(create_fn(), del);
            };
        }
    }

    if (!tool_registry_global().register_tool(std::move(entry))) {
        std::cerr << "failed to register tool " << unique_id << "\n";
        FreeLibrary(h);
        delete info;
        return {};
    }
    std::cerr << "DEBUG: PluginLoader registered tool id='" << unique_id << "' base='" << base_id << "' path='" << path << "'\n";

    // call plugin_init if present
    if (info->plugin_init) {
        HostAPI* hostapi = reinterpret_cast<HostAPI*>(host);
        int r = info->plugin_init(hostapi);
        if (!r) {
            std::cerr << "plugin_init failed for " << unique_id << "\n";
            tool_registry_global().unregister_tool(unique_id);
            FreeLibrary(h);
            delete info;
            return {};
        }
        std::cerr << "DEBUG: plugin_init succeeded for " << unique_id << "\n";
    }

    modules_.emplace(unique_id, std::unique_ptr<LoadedModuleInfo>(info));
    std::cerr << "DEBUG: plugin_loader: loaded module unique_id='" << unique_id << "' path='" << path << "'\n";
    return unique_id;
#else
    (void)path;
    std::cerr << "PluginLoader only implemented on Windows in this prototype.\n";
    return {};
#endif
}

bool PluginLoader::unload_module(const std::string& tool_id) {
    auto it = modules_.find(tool_id);
    if (it == modules_.end()) return false;
    // call plugin_shutdown if present
    if (it->second->plugin_shutdown) {
        try { it->second->plugin_shutdown(); } catch (...) {}
    }
    // unregister from registry
    bool ok = tool_registry_global().unregister_tool(tool_id);

#ifdef _WIN32
    if (it->second->handle) {
        FreeLibrary(it->second->handle);
        it->second->handle = nullptr;
    }
#endif
    modules_.erase(it);
    return ok;
}

std::vector<std::string> PluginLoader::loaded_modules() const {
    std::vector<std::string> out;
    out.reserve(modules_.size());
    for (const auto& kv : modules_) out.push_back(kv.first);
    return out;
}
