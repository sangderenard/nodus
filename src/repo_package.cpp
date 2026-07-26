#include "repo_package.h"
#include "plugin_manager.h"
#include "tool_api.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace {

struct LoadedRepoPackage {
    std::string owner;
    std::string version;
    std::vector<std::string> tool_ids; // pass to gp_plugin_unload
#ifdef _WIN32
    std::vector<HMODULE> contract_handles; // pass to FreeLibrary
#endif
};

std::vector<LoadedRepoPackage> g_loaded_packages;

LoadedRepoPackage* find_loaded(const std::string& owner) {
    for (auto& p : g_loaded_packages) {
        if (p.owner == owner) return &p;
    }
    return nullptr;
}

fs::path resolve_path(const std::string& root_dir, const std::string& p) {
    if (p.empty()) return {};
    fs::path fp(p);
    if (fp.is_absolute()) return fp;
    return fs::path(root_dir) / fp;
}

fs::path find_nodus_repo_root(const fs::path& hint) {
    fs::path cur = hint;
    while (!cur.empty()) {
        if (fs::exists(cur / "include" / "tool_api.h")) return cur;
        fs::path parent = cur.parent_path();
        if (parent == cur) break;
        cur = parent;
    }
    return fs::current_path();
}

std::string timestamp_tag() {
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
    return std::string(buf);
}

#ifdef _WIN32
// Build `src` as a standalone shared library and return the built DLL path
// (empty on failure). Used for register_types()-only contract sources,
// which have no create_tool export and so can't go through
// gp_plugin_build_module_and_load (PluginLoader::load_module requires one).
fs::path build_contract_dll(const fs::path& src, const fs::path& package_root) {
    fs::path abs_src = fs::absolute(src);
    if (!fs::exists(abs_src)) return {};

    fs::path nodus_root = find_nodus_repo_root(abs_src.parent_path());
    if (!fs::exists(nodus_root / "include" / "tool_api.h")) {
        nodus_root = find_nodus_repo_root(package_root);
    }

    std::string name = abs_src.stem().string() + "_" + timestamp_tag();
    fs::path scratch = fs::temp_directory_path() / (std::string("nodus_pkg_") + name);
    fs::path build_dir = scratch / "build";
    std::error_code ec;
    fs::create_directories(build_dir, ec);

    std::ostringstream cm;
    cm << "cmake_minimum_required(VERSION 3.15)\n";
    cm << "project(" << name << " LANGUAGES CXX)\n";
    cm << "add_library(" << name << " SHARED \"" << abs_src.generic_string() << "\")\n";
    cm << "target_include_directories(" << name << " PRIVATE \""
       << nodus_root.generic_string() << "\" \""
       << (nodus_root / "include").generic_string() << "\" \""
       << package_root.generic_string() << "\")\n";
    fs::path release_dir = nodus_root / "build" / "Release";
    const std::vector<std::string> candidate_libs = {
        (release_dir / "canvas_tables.lib").generic_string(),
        (release_dir / "libcanvas_tables.a").generic_string(),
    };
    for (const auto& lib : candidate_libs) {
        if (fs::exists(lib)) {
            cm << "target_link_libraries(" << name << " PRIVATE \"" << lib << "\")\n";
            break;
        }
    }
    cm << "set_target_properties(" << name << " PROPERTIES CXX_STANDARD 17)\n";

    std::ofstream ofs(scratch / "CMakeLists.txt");
    if (!ofs.good()) return {};
    ofs << cm.str();
    ofs.close();

    std::string cfg_cmd = "cmake -S \"" + scratch.string() + "\" -B \"" + build_dir.string() + "\" -DCMAKE_BUILD_TYPE=Release";
    if (std::system(cfg_cmd.c_str()) != 0) return {};
    std::string build_cmd = "cmake --build \"" + build_dir.string() + "\" --config Release --target \"" + name + "\"";
    if (std::system(build_cmd.c_str()) != 0) return {};

    const std::vector<fs::path> candidates = {
        build_dir / "Release" / (name + ".dll"),
        build_dir / (name + ".dll"),
    };
    for (const auto& c : candidates) {
        if (fs::exists(c)) return c;
    }
    return {};
}
#endif

} // namespace

int gp_repo_package_write_to_file(const GP_RepoPackage& package, const char* path) {
    if (!path) return 0;
    std::ofstream ofs(path);
    if (!ofs.good()) return 0;
    ofs << "NODUSPKG V1\n";
    ofs << "OWNER " << std::quoted(package.owner) << "\n";
    ofs << "VERSION " << std::quoted(package.version) << "\n";
    ofs << "ROOT " << std::quoted(package.root_dir) << "\n";
    for (const auto& tool : package.tools) {
        ofs << "TOOL " << tool.id << " " << std::quoted(tool.name) << " "
            << std::quoted(tool.source_or_dll_path) << " " << tool.caps << " "
            << tool.capability_tags << " " << std::quoted(tool.notes) << "\n";
    }
    for (const auto& contract : package.contracts) {
        ofs << "CONTRACT " << std::quoted(contract.source_path) << "\n";
    }
    return 1;
}

int gp_repo_package_read_from_file(const char* path, GP_RepoPackage* out_package) {
    if (!path || !out_package) return 0;
    std::ifstream ifs(path);
    if (!ifs.good()) return 0;
    GP_RepoPackage package{};
    std::string line;
    if (!std::getline(ifs, line)) return 0;
    if (line.rfind("NODUSPKG", 0) != 0) return 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "OWNER") {
            ss >> std::quoted(package.owner);
        } else if (tag == "VERSION") {
            ss >> std::quoted(package.version);
        } else if (tag == "ROOT") {
            ss >> std::quoted(package.root_dir);
        } else if (tag == "TOOL") {
            GP_RepoPackageTool tool{};
            ss >> tool.id >> std::quoted(tool.name) >> std::quoted(tool.source_or_dll_path)
               >> tool.caps >> tool.capability_tags >> std::quoted(tool.notes);
            package.tools.push_back(std::move(tool));
        } else if (tag == "CONTRACT") {
            GP_RepoPackageContract contract{};
            ss >> std::quoted(contract.source_path);
            package.contracts.push_back(std::move(contract));
        }
    }
    *out_package = std::move(package);
    return 1;
}

int gp_repo_package_ingest(const GP_RepoPackage& package, void* host) {
    if (package.owner.empty()) {
        std::cerr << "repo_package: ingest requires a non-empty owner\n";
        return 0;
    }
    if (find_loaded(package.owner)) {
        std::cerr << "repo_package: '" << package.owner << "' is already ingested; call gp_repo_package_unload first\n";
        return 0;
    }

    LoadedRepoPackage loaded;
    loaded.owner = package.owner;
    loaded.version = package.version;

    auto rollback = [&]() {
        for (const auto& id : loaded.tool_ids) gp_plugin_unload(id.c_str());
#ifdef _WIN32
        for (HMODULE h : loaded.contract_handles) FreeLibrary(h);
#endif
    };

#ifdef _WIN32
    for (const auto& contract : package.contracts) {
        fs::path src = resolve_path(package.root_dir, contract.source_path);
        fs::path dll = build_contract_dll(src, fs::path(package.root_dir));
        if (dll.empty()) {
            std::cerr << "repo_package: failed to build contract '" << contract.source_path << "'\n";
            rollback();
            return 0;
        }
        HMODULE h = LoadLibraryA(dll.string().c_str());
        if (!h) {
            std::cerr << "repo_package: failed to load contract dll '" << dll.string() << "'\n";
            rollback();
            return 0;
        }
        auto register_types_fn = reinterpret_cast<RegisterTypesFn>(GetProcAddress(h, "register_types"));
        if (!register_types_fn) {
            std::cerr << "repo_package: contract '" << contract.source_path << "' has no register_types export\n";
            FreeLibrary(h);
            rollback();
            return 0;
        }
        // Runs register_types() so a contract can self-register into its own
        // DLL's local ValueTypeRegistry copy for its own tools' internal use.
        // This does NOT reach the host's registry (see GP_RepoPackageContract's
        // doc comment) -- do not add code here that expects it to.
        try {
            register_types_fn();
        } catch (...) {
            std::cerr << "repo_package: register_types threw in '" << contract.source_path << "'\n";
            FreeLibrary(h);
            rollback();
            return 0;
        }
        loaded.contract_handles.push_back(h);
    }
#else
    if (!package.contracts.empty()) {
        std::cerr << "repo_package: contracts only supported on Windows in this prototype\n";
        return 0;
    }
#endif

    for (const auto& tool : package.tools) {
        fs::path src_or_dll = resolve_path(package.root_dir, tool.source_or_dll_path);
        char id_buf[256] = {0};
        bool ok = false;
        if (src_or_dll.extension() == ".dll") {
            ok = gp_plugin_load_from_path_with_host(src_or_dll.string().c_str(), host, id_buf, sizeof(id_buf)) != 0;
        } else {
            ok = gp_plugin_build_module_and_load(src_or_dll.string().c_str(), package.root_dir.c_str(), nullptr,
                                                  host, id_buf, sizeof(id_buf)) != 0;
        }
        if (!ok) {
            std::cerr << "repo_package: failed to load tool '" << tool.id << "' (" << tool.source_or_dll_path << ")\n";
            rollback();
            return 0;
        }
        loaded.tool_ids.emplace_back(id_buf);
    }

    g_loaded_packages.push_back(std::move(loaded));
    return 1;
}

int gp_repo_package_ingest_from_file(const char* path, void* host) {
    GP_RepoPackage package{};
    if (!gp_repo_package_read_from_file(path, &package)) return 0;
    return gp_repo_package_ingest(package, host);
}

int gp_repo_package_unload(const std::string& owner) {
    for (auto it = g_loaded_packages.begin(); it != g_loaded_packages.end(); ++it) {
        if (it->owner != owner) continue;
        for (const auto& id : it->tool_ids) gp_plugin_unload(id.c_str());
#ifdef _WIN32
        for (HMODULE h : it->contract_handles) FreeLibrary(h);
#endif
        g_loaded_packages.erase(it);
        return 1;
    }
    return 0;
}

std::vector<std::string> gp_repo_package_list() {
    std::vector<std::string> out;
    out.reserve(g_loaded_packages.size());
    for (const auto& p : g_loaded_packages) out.push_back(p.owner);
    return out;
}
