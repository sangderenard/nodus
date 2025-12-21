#include "module_library_actualizer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>
#include <cstdio>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
bool ensure_parent_dir(const std::filesystem::path& path, std::string& err) {
    try {
        auto parent = path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool atomic_write_file(const std::filesystem::path& target, const std::string& contents, std::string& err) {
    try {
        if (!ensure_parent_dir(target, err)) return false;
        // create a temp file name in the same directory
        std::random_device rd;
        std::mt19937_64 gen(rd());
        uint64_t v = gen();
        std::ostringstream ss;
        ss << std::hex << v;
        auto tmp = target.parent_path() / (target.filename().string() + ".tmp-" + ss.str());

        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs.good()) {
            err = "failed to open temp file for writing: " + tmp.string();
            return false;
        }
        ofs << contents;
        ofs.flush();
        ofs.close();

        // Move/rename into place. On Windows use MoveFileEx to replace existing atomically.
#ifdef _WIN32
        if (!MoveFileExW(tmp.wstring().c_str(), target.wstring().c_str(), MOVEFILE_REPLACE_EXISTING)) {
            err = "MoveFileExW failed: " + std::to_string(GetLastError());
            std::filesystem::remove(tmp);
            return false;
        }
#else
        std::filesystem::rename(tmp, target);
#endif
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        try { std::filesystem::remove(target); } catch (...) {}
        return false;
    }
}

bool write_placeholder_if_missing(const std::filesystem::path& path, const std::string& contents, std::string& err) {
    try {
        if (std::filesystem::exists(path)) return true;
        return atomic_write_file(path, contents, err);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}
} // namespace

int gp_module_library_actualize_sources(const GP_ModuleLibrary& library, const char* output_root) {
    std::string root_str = output_root ? output_root : library.root_dir;
    if (root_str.empty()) root_str = gp_module_library_default_root();
    std::filesystem::path root = root_str;

    bool ok = true;
    std::string err;

    try {
        std::filesystem::create_directories(root / "serialized");
        std::filesystem::create_directories(root / "source" / "tools");
        std::filesystem::create_directories(root / "source" / "modules");
    } catch (const std::exception& e) {
        std::cerr << "failed to create output directories: " << e.what() << "\n";
        return 0;
    }

    for (const auto& tool : library.tool_registry) {
        std::filesystem::path tool_path = tool.source_path.empty()
            ? std::filesystem::path(gp_module_library_tool_source_path(root.string(), tool.id))
            : std::filesystem::path(tool.source_path);
        if (!tool_path.is_absolute()) tool_path = root / tool_path;
        std::string contents = "// Placeholder tool source for " + tool.id + " (" + tool.name + ")\n";
        if (!write_placeholder_if_missing(tool_path, contents, err)) {
            std::cerr << "error writing tool source '" << tool_path.string() << "': " << err << "\n";
            ok = false;
        }
    }

    for (const auto& module : library.modules) {
        std::filesystem::path module_path = module.source_path.empty()
            ? std::filesystem::path(gp_module_library_module_source_path(root.string(), module.id))
            : std::filesystem::path(module.source_path);
        if (!module_path.is_absolute()) module_path = root / module_path;
        std::string contents = "// Placeholder module source for " + module.id + "\n";
        if (!write_placeholder_if_missing(module_path, contents, err)) {
            std::cerr << "error writing module source '" << module_path.string() << "': " << err << "\n";
            ok = false;
        }
        if (!module.serialized_path.empty()) {
            std::filesystem::path ser_path = std::filesystem::path(module.serialized_path);
            if (!ser_path.is_absolute()) ser_path = root / ser_path;
            if (!write_placeholder_if_missing(ser_path, "", err)) {
                std::cerr << "error writing serialized module '" << ser_path.string() << "': " << err << "\n";
                ok = false;
            }
        }
    }

    return ok ? 1 : 0;
}

int gp_module_library_actualize_from_file(const char* path, const char* output_root) {
    GP_ModuleLibrary library{};
    if (!gp_module_library_read_from_file(path, &library)) return 0;
    return gp_module_library_actualize_sources(library, output_root);
}
