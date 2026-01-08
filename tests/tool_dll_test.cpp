#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "module_library_actualizer.h"
#include "plugin_loader.h"
#include "tool_api.h"
#include "tool_registry.h"

namespace fs = std::filesystem;

class StubHost final : public HostAPI {
public:
    void log(const char* message) override {
        if (message) std::cerr << message << "\n";
    }
    void draw_text(const TextRenderArgs& /*args*/) override {}
};

static std::string make_temp_name(const char* prefix) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    uint64_t v = gen();
    return std::string(prefix) + std::to_string(v);
}

static std::vector<fs::path> list_files_with_ext(const fs::path& root, const char* ext) {
    std::vector<fs::path> out;
    if (!fs::exists(root)) return out;
    for (auto const& entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ext) out.push_back(entry.path());
    }
    return out;
}

static fs::path find_canvas_import_lib(const fs::path& build_dir) {
    const fs::path release = build_dir / "Release" / "canvas_tables.lib";
    if (fs::exists(release)) return release;
    const fs::path root = build_dir / "canvas_tables.lib";
    if (fs::exists(root)) return root;
    for (auto const& entry : fs::recursive_directory_iterator(build_dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().filename() == "canvas_tables.lib") return entry.path();
    }
    return {};
}

static fs::path find_tool_dll(const fs::path& build_dir, const std::string& tool_name) {
    const fs::path out = build_dir / "out" / (tool_name + ".dll");
    if (fs::exists(out)) return out;
    const fs::path release = build_dir / "Release" / (tool_name + ".dll");
    if (fs::exists(release)) return release;
    const fs::path root = build_dir / (tool_name + ".dll");
    if (fs::exists(root)) return root;
    for (auto const& entry : fs::recursive_directory_iterator(build_dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().filename() == (tool_name + ".dll")) return entry.path();
    }
    return {};
}

static bool run_command(const std::string& cmd) {
    int rc = std::system(cmd.c_str());
    return rc == 0;
}

static bool write_text_file(const fs::path& path, const std::string& contents) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.good()) return false;
    ofs << contents;
    return ofs.good();
}

int main() {
#ifndef _WIN32
    std::cerr << "tool_dll_test skipped: plugin loader only implemented on Windows.\n";
    return 0;
#endif

    const fs::path repo_root = fs::path(NODUS_REPO_ROOT);
    const fs::path build_dir = fs::path(NODUS_BUILD_DIR);
    const fs::path repo_module_root = repo_root / "module_library";
    const fs::path serialized_dir = repo_module_root / "serialized";

    auto serialized = list_files_with_ext(serialized_dir, ".gpmod");
    if (serialized.empty()) {
        std::cerr << "tool_dll_test skipped: no serialized modules.\n";
        return 0;
    }

    fs::path test_root = repo_module_root / "test_build" / "tool_dll_test";
    const fs::path tool_src_dir = test_root / "source" / "tools";
    fs::create_directories(test_root);

    fs::path modlib_path = test_root / "module_library.txt";
    std::ostringstream lib;
    lib << "MODULELIB V1\n";
    lib << "ROOT \"" << test_root.generic_string() << "\"\n";
    for (size_t i = 0; i < serialized.size(); ++i) {
        const auto& ser_path = serialized[i];
        std::string module_id = ser_path.stem().string();
        std::string tool_id = "tool_" + module_id;
        lib << "MODULE " << module_id << " " << i << " \"" << module_id << "\" \""
            << ser_path.generic_string() << "\" \"\" 1 \"" << tool_id << "\" 0\n";
    }
    if (!write_text_file(modlib_path, lib.str())) {
        std::cerr << "failed to write module library file\n";
        return 1;
    }

    std::set<fs::path> preexisting;
    for (const auto& p : list_files_with_ext(tool_src_dir, ".cpp")) preexisting.insert(p);

    if (!gp_module_library_actualize_from_file(modlib_path.string().c_str(),
                                               test_root.string().c_str())) {
        std::cerr << "module library actualization failed\n";
        return 1;
    }

    std::vector<fs::path> generated;
    for (const auto& p : list_files_with_ext(tool_src_dir, ".cpp")) {
        if (preexisting.find(p) == preexisting.end()) generated.push_back(p);
    }
    if (generated.empty()) {
        std::cerr << "no tool sources generated\n";
        return 1;
    }

    fs::path canvas_lib = find_canvas_import_lib(build_dir);
    if (canvas_lib.empty()) {
        std::cerr << "canvas_tables.lib not found in build dir\n";
        return 1;
    }

    StubHost host;
    PluginLoader loader;
    bool ok = true;

    for (const auto& tool_source : generated) {
        std::string tool_name = tool_source.stem().string();
        fs::path proj_dir = test_root / ("cmake_" + tool_name);
        fs::path proj_build = proj_dir / "build";
        fs::create_directories(proj_dir);

        std::ostringstream cmake;
        cmake << "cmake_minimum_required(VERSION 3.15)\n";
        cmake << "project(" << tool_name << " LANGUAGES CXX)\n";
        cmake << "add_library(" << tool_name << " SHARED \"" << tool_source.generic_string() << "\")\n";
        cmake << "target_include_directories(" << tool_name << " PRIVATE \""
              << repo_root.generic_string() << "\" \""
              << (repo_root / "include").generic_string() << "\")\n";
        cmake << "target_link_libraries(" << tool_name << " PRIVATE \""
              << canvas_lib.generic_string() << "\")\n";
        cmake << "set_target_properties(" << tool_name << " PROPERTIES CXX_STANDARD 17)\n";
        cmake << "set(CMAKE_RUNTIME_OUTPUT_DIRECTORY \"${CMAKE_BINARY_DIR}/out\")\n";
        cmake << "set(CMAKE_LIBRARY_OUTPUT_DIRECTORY \"${CMAKE_BINARY_DIR}/out\")\n";
        cmake << "set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY \"${CMAKE_BINARY_DIR}/out\")\n";
        cmake << "set(CMAKE_PDB_OUTPUT_DIRECTORY \"${CMAKE_BINARY_DIR}/out\")\n";
        cmake << "set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE \"${CMAKE_BINARY_DIR}/out\")\n";
        cmake << "set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE \"${CMAKE_BINARY_DIR}/out\")\n";
        cmake << "set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE \"${CMAKE_BINARY_DIR}/out\")\n";
        cmake << "set(CMAKE_PDB_OUTPUT_DIRECTORY_RELEASE \"${CMAKE_BINARY_DIR}/out\")\n";

        if (!write_text_file(proj_dir / "CMakeLists.txt", cmake.str())) {
            std::cerr << "failed to write CMakeLists for " << tool_name << "\n";
            ok = false;
            break;
        }

        std::string cfg_cmd = "cmake -S \"" + proj_dir.string() + "\" -B \"" +
                              proj_build.string() + "\" -DCMAKE_BUILD_TYPE=Release";
        std::string build_cmd = "cmake --build \"" + proj_build.string() + "\" --config Release";
        if (!run_command(cfg_cmd) || !run_command(build_cmd)) {
            std::cerr << "failed to build tool " << tool_name << "\n";
            ok = false;
            break;
        }

        fs::path tool_lib = find_tool_dll(proj_build, tool_name);
        if (tool_lib.empty()) {
            std::cerr << "tool DLL not found in: " << proj_build.string() << "\n";
            ok = false;
            break;
        }

        std::string id = loader.load_module(tool_lib.string(), &host);
        if (id.empty()) {
            std::cerr << "failed to load tool DLL: " << tool_lib.string() << "\n";
            ok = false;
            break;
        }

        auto tool = tool_registry_global().create(id);
        if (!tool) {
            std::cerr << "failed to create tool instance for: " << id << "\n";
            ok = false;
            break;
        }
        ToolInitContext ctx{};
        RenderContext rctx{};
        tool->initialize(ctx);
        tool->tick(0.016, host);
        tool->render(rctx);
        tool->shutdown();
        tool.reset();
        loader.unload_module(id);
    }

    return ok ? 0 : 1;
}
