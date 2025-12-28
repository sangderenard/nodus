#include "plugin_manager.h"
#include "plugin_loader.h"
#include "tool_registry.h"
#include "module_library.h"

#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <cstdlib>

static PluginLoader g_plugin_loader;

int gp_plugin_load_from_path(const char* path, char* out_id, int out_id_capacity) {
    if (!path || !out_id || out_id_capacity <= 0) return 0;
    std::string id = g_plugin_loader.load_module(path);
    if (id.empty()) return 0;
    // copy id into out buffer
    int copy_len = static_cast<int>(std::min<size_t>(id.size(), static_cast<size_t>(out_id_capacity - 1)));
    std::memcpy(out_id, id.c_str(), static_cast<size_t>(copy_len));
    out_id[copy_len] = '\0';
    return 1;
}

int gp_plugin_load_from_path_with_host(const char* path, void* host, char* out_id, int out_id_capacity) {
    if (!path || !out_id || out_id_capacity <= 0) return 0;
    std::string id = g_plugin_loader.load_module(path, host);
    if (id.empty()) return 0;
    int copy_len = static_cast<int>(std::min<size_t>(id.size(), static_cast<size_t>(out_id_capacity - 1)));
    std::memcpy(out_id, id.c_str(), static_cast<size_t>(copy_len));
    out_id[copy_len] = '\0';
    return 1;
}

int gp_plugin_unload(const char* id) {
    if (!id) return 0;
    return g_plugin_loader.unload_module(id) ? 1 : 0;
}

int gp_plugin_list_ids(char* out_buf, int out_buf_capacity) {
    auto ids = g_plugin_loader.loaded_modules();
    std::ostringstream ss;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) ss << ',';
        ss << ids[i];
    }
    std::string s = ss.str();
    if (out_buf && out_buf_capacity > 0) {
        int copy_len = static_cast<int>(std::min<size_t>(s.size(), static_cast<size_t>(out_buf_capacity - 1)));
        std::memcpy(out_buf, s.c_str(), static_cast<size_t>(copy_len));
        out_buf[copy_len] = '\0';
    }
    return static_cast<int>(ids.size());
}

static std::string timestamp_string() {
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

static std::string shared_library_extension() {
#if defined(_WIN32)
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
}

static std::string normalize_source_path_for_registry(const std::string& path) {
    if (path.empty()) return {};
    namespace fs = std::filesystem;
    try {
        fs::path p(path);
        if (!p.is_absolute()) return p.generic_string();
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

int gp_plugin_build_and_load(const char* build_dir,
                             const char* target,
                             const char* built_relpath,
                             const char* dest_dir,
                             char* out_id,
                             int out_id_capacity) {
    // delegate to host-aware variant with null host
    return gp_plugin_build_and_load_with_host(build_dir, target, built_relpath, dest_dir, nullptr, out_id, out_id_capacity);
}

int gp_plugin_build_and_load_with_host(const char* build_dir,
                                       const char* target,
                                       const char* built_relpath,
                                       const char* dest_dir,
                                       void* host,
                                       char* out_id,
                                       int out_id_capacity) {
    if (!build_dir || !target || !built_relpath || !out_id || out_id_capacity <= 0) return 0;
    namespace fs = std::filesystem;

    std::string builddir = build_dir;
    std::string tgt = target;
    std::string rel = built_relpath;
    std::string dest = dest_dir ? dest_dir : ".";

    std::string cmd = "cmake --build \"" + builddir + "\" --config Release --target \"" + tgt + "\"";
    int rc = std::system(cmd.c_str());
    if (rc != 0) return 0;

    fs::path built = fs::path(builddir) / fs::path(rel);
    if (!fs::exists(built)) return 0;

    try {
        fs::create_directories(dest);
        // Preserve the built library filename (module_name includes the version tag from the generated source)
        fs::path destpath = fs::path(dest) / built.filename();
        fs::copy_file(built, destpath, fs::copy_options::overwrite_existing);
        std::string id = g_plugin_loader.load_module(destpath.string(), host);
        if (id.empty()) return 0;
        int copy_len = static_cast<int>(std::min<size_t>(id.size(), static_cast<size_t>(out_id_capacity - 1)));
        std::memcpy(out_id, id.c_str(), static_cast<size_t>(copy_len));
        out_id[copy_len] = '\0';
        return 1;
    } catch (...) {
        return 0;
    }
}

int gp_plugin_build_module_and_load(const char* module_src,
                                    const char* repo_root,
                                    const char* dest_dir,
                                    void* host,
                                    char* out_id,
                                    int out_id_capacity) {
    if (!module_src || !out_id || out_id_capacity <= 0) return 0;
    namespace fs = std::filesystem;
    try {
        fs::path src = fs::path(module_src);
        if (!fs::exists(src)) return 0;

        std::string ts = timestamp_string();
        fs::path scratch = fs::temp_directory_path() / (std::string("nodus_mod_") + ts);
        fs::create_directories(scratch);
        fs::path builddir = scratch / "build";
        fs::create_directories(builddir);

        // write simple CMakeLists.txt that builds the single source as a shared lib
        std::string module_name = src.stem().string();
        std::ostringstream cm; 
        cm << "cmake_minimum_required(VERSION 3.15)\n";
        cm << "project(" << module_name << " LANGUAGES CXX)\n";
        std::filesystem::path abs_src = std::filesystem::absolute(src);
        std::string abs_src_str = abs_src.generic_string();
        cm << "add_library(" << module_name << " SHARED \"" << abs_src_str << "\")\n";
        if (repo_root && repo_root[0] != '\0') {
            std::filesystem::path rr(repo_root);
            std::filesystem::path abs_rr = std::filesystem::absolute(rr);
            cm << "target_include_directories(" << module_name << " PRIVATE \"" << abs_rr.generic_string() << "\")\n";
            std::filesystem::path release_dir = abs_rr / "build" / "Release";
            const std::vector<std::string> candidate_libs = {
                (release_dir / "canvas_tables.lib").generic_string(),
                (release_dir / "libcanvas_tables.a").generic_string(),
                (release_dir / "canvas_tables.so").generic_string(),
                (release_dir / "canvas_tables.dylib").generic_string()
            };
            for (const auto& lib_path : candidate_libs) {
                if (std::filesystem::exists(lib_path)) {
                    cm << "target_link_libraries(" << module_name << " PRIVATE \"" << lib_path << "\")\n";
                    break;
                }
            }
        }
        cm << "set_target_properties(" << module_name << " PROPERTIES CXX_STANDARD 17)\n";

        fs::path cmake_file = scratch / "CMakeLists.txt";
        {
            std::ofstream ofs(cmake_file);
            if (!ofs.good()) { fs::remove_all(scratch); return 0; }
            ofs << cm.str();
            ofs.close();
        }

        // configure
        std::string cfg_cmd = std::string("cmake -S \"") + scratch.string() + "\" -B \"" + builddir.string() + "\" -DCMAKE_BUILD_TYPE=Release";
        int rc = std::system(cfg_cmd.c_str());
        if (rc != 0) { fs::remove_all(scratch); return 0; }

        // build
        std::string build_cmd = std::string("cmake --build \"") + builddir.string() + "\" --config Release --target \"" + module_name + "\"";
        rc = std::system(build_cmd.c_str());
        if (rc != 0) { fs::remove_all(scratch); return 0; }

        // locate built library
        std::string ext = shared_library_extension();
        std::vector<fs::path> candidates = {
            builddir / "Release" / (module_name + ext),
            builddir / (module_name + ext),
            builddir / "Release" / ("lib" + module_name + ext),
            builddir / ("lib" + module_name + ext)
        };
        fs::path built;
        for (auto &p : candidates) if (fs::exists(p)) { built = p; break; }
        if (built.empty()) { fs::remove_all(scratch); return 0; }

        fs::create_directories(dest_dir ? fs::path(dest_dir) : fs::path("."));
        // Preserve the built library filename (module_name includes the version tag from the generated source)
        fs::path destpath = fs::path(dest_dir ? dest_dir : ".") / built.filename();
        fs::copy_file(built, destpath, fs::copy_options::overwrite_existing);

        std::string id = g_plugin_loader.load_module(destpath.string(), host);
        fs::remove_all(scratch);
        if (id.empty()) return 0;
        (void)tool_registry_global().set_source_path(id, normalize_source_path_for_registry(abs_src_str));
        int copy_len = static_cast<int>(std::min<size_t>(id.size(), static_cast<size_t>(out_id_capacity - 1)));
        std::memcpy(out_id, id.c_str(), static_cast<size_t>(copy_len));
        out_id[copy_len] = '\0';
        return 1;
    } catch (...) {
        return 0;
    }
}
