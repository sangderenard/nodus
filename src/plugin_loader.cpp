#include "plugin_loader.h"
#include "tool_registry.h"
#include "tool_api.h"
#include "module_library.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <iostream>
#include <sstream>
#include <filesystem>

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
    int auto_mouse_ports = 0;
    int auto_keyboard_ports = 0;
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

static std::string normalize_source_path(const std::string& path) {
    if (path.empty()) return {};
    namespace fs = std::filesystem;
    try {
        fs::path p(path);
        if (!p.is_absolute()) {
            return p.generic_string();
        }
        fs::path root = fs::absolute(fs::path(gp_module_library_default_root()));
        fs::path abs_p = fs::absolute(p);
        std::error_code ec;
        fs::path rel = fs::relative(abs_p, root, ec);
        if (!ec) {
            std::string rel_str = rel.generic_string();
            if (!rel_str.empty() && rel_str.rfind("..", 0) != 0) {
                return rel_str;
            }
        }
        return abs_p.generic_string();
    } catch (...) {
        return path;
    }
}

static std::string find_plugin_source_path(const std::string& dll_path, HMODULE module) {
    using source_path_fn_t = const char* (*)();
    source_path_fn_t src_fn = (source_path_fn_t)GetProcAddress(module, "plugin_source_path");
    if (src_fn) {
        const char* p = src_fn();
        if (p && p[0] != '\0') return normalize_source_path(p);
    }
    namespace fs = std::filesystem;
    try {
        fs::path dll = fs::path(dll_path);
        fs::path stem = dll.stem();
        fs::path same_dir = dll.parent_path() / (stem.string() + ".cpp");
        if (fs::exists(same_dir)) {
            return normalize_source_path(same_dir.generic_string());
        }
        fs::path lib_src = fs::path(gp_module_library_default_root()) / "source" / "tools" / (stem.string() + ".cpp");
        if (fs::exists(lib_src)) {
            return normalize_source_path(lib_src.generic_string());
        }
    } catch (...) {
    }
    return {};
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
    std::cerr << "DEBUG: plugin_loader: create_tool fn at " << reinterpret_cast<void*>(create_fn) << " for " << path << "\n";
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

    // Optional: register any new ValueType structs this DLL's tools need,
    // once, before any tool instance is created. Absent in most plugins.
    if (auto register_types_fn = reinterpret_cast<RegisterTypesFn>(GetProcAddress(h, "register_types"))) {
        try { register_types_fn(); } catch (...) {
            std::cerr << "register_types threw in " << path << "\n";
        }
    }

    // instantiate temporary tool to ensure create/destroy are viable
    ITool* tmp = create_fn();
    std::cerr << "DEBUG: plugin_loader: create_tool returned " << tmp << " for " << path << "\n";
    std::filesystem::path p(path);
    std::string fallback_id = p.stem().string();
    if (fallback_id.empty()) fallback_id = "tool";
    std::string base_id = fallback_id;
    std::string name = fallback_id;
    ToolCaps caps = ToolCaps::None;
    bool tmp_valid = (tmp != nullptr) && (reinterpret_cast<uintptr_t>(tmp) != static_cast<uintptr_t>(~0ull));
    if (tmp_valid) {
        try {
            // Prefer C-style stable pointers to avoid cross-DLL C++ ABI/allocator issues.
            const char* idc = nullptr;
            const char* namec = nullptr;
            try { idc = tmp->id_cstr(); } catch (...) { idc = nullptr; }
            try { namec = tmp->name_cstr(); } catch (...) { namec = nullptr; }
            if (idc && idc[0] != '\0') base_id = idc; else base_id = fallback_id;
            if (namec && namec[0] != '\0') name = namec; else name = base_id;
            // caps stays as before
            try { caps = tmp->caps(); } catch (...) { caps = ToolCaps::None; }
        } catch (...) {
            tmp_valid = false;
            base_id = fallback_id;
            name = fallback_id;
            caps = ToolCaps::None;
        }
    } else {
        std::cerr << "create_tool returned invalid pointer in " << path << ", falling back to filename id\n";
    }

    std::cerr << "DEBUG: plugin_loader: discovered base_id='" << base_id << "' name='" << name << "' caps=" << static_cast<int>(caps) << " from '" << path << "'\n";

    // destroy temporary instance if it looked valid
    if (tmp_valid) {
        if (info->destroy_fn) info->destroy_fn(tmp); else delete tmp;
    }

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
    entry.source_path = find_plugin_source_path(path, h);
    // Optional autobind exports
#ifdef _WIN32
    if (h) {
        using autobind_fn = int (*)();
        if (auto fn = reinterpret_cast<autobind_fn>(GetProcAddress(h, "nodus_autobind_mouse_ports"))) {
            info->auto_mouse_ports = fn();
        }
        if (auto fn = reinterpret_cast<autobind_fn>(GetProcAddress(h, "nodus_autobind_keyboard_ports"))) {
            info->auto_keyboard_ports = fn();
        }
    }
#endif
    entry.auto_mouse_ports = info->auto_mouse_ports;
    entry.auto_keyboard_ports = info->auto_keyboard_ports;
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
