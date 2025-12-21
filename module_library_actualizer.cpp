#include "module_library_actualizer.h"

#include <filesystem>
#include <fstream>

namespace {
void ensure_parent_dir(const std::filesystem::path& path) {
    auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void write_placeholder_if_missing(const std::filesystem::path& path, const std::string& contents) {
    if (std::filesystem::exists(path)) return;
    ensure_parent_dir(path);
    std::ofstream ofs(path);
    if (!ofs.good()) return;
    ofs << contents;
}
} // namespace

int gp_module_library_actualize_sources(const GP_ModuleLibrary& library, const char* output_root) {
    std::string root = output_root ? output_root : library.root_dir;
    if (root.empty()) root = gp_module_library_default_root();

    std::filesystem::create_directories(root + "/serialized");
    std::filesystem::create_directories(root + "/source/tools");
    std::filesystem::create_directories(root + "/source/modules");

    for (const auto& tool : library.tool_registry) {
        std::string tool_path = tool.source_path.empty()
            ? gp_module_library_tool_source_path(root, tool.id)
            : tool.source_path;
        std::string contents = "// Placeholder tool source for " + tool.id + " (" + tool.name + ")\n";
        write_placeholder_if_missing(tool_path, contents);
    }

    for (const auto& module : library.modules) {
        std::string module_path = module.source_path.empty()
            ? gp_module_library_module_source_path(root, module.id)
            : module.source_path;
        std::string contents = "// Placeholder module source for " + module.id + "\n";
        write_placeholder_if_missing(module_path, contents);
        if (!module.serialized_path.empty()) {
            write_placeholder_if_missing(module.serialized_path, "");
        }
    }

    return 1;
}

int gp_module_library_actualize_from_file(const char* path, const char* output_root) {
    GP_ModuleLibrary library{};
    if (!gp_module_library_read_from_file(path, &library)) return 0;
    return gp_module_library_actualize_sources(library, output_root);
}
