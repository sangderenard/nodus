#include "module_library.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace {
constexpr const char* kLibraryRoot = "module_library";
constexpr const char* kSerializedDir = "serialized";
constexpr const char* kSourceDir = "source";
constexpr const char* kToolDir = "tools";
constexpr const char* kModuleDir = "modules";

std::string join_path(const std::string& root, const std::string& subdir, const std::string& filename) {
    std::filesystem::path p = root;
    if (!subdir.empty()) p /= subdir;
    if (!filename.empty()) p /= filename;
    return p.string();
}
} // namespace

std::string gp_module_library_default_root() {
    return kLibraryRoot;
}

std::string gp_module_library_tool_id(ModuleToolKind kind) {
    return "tool_" + std::to_string(static_cast<int>(kind));
}

std::string gp_module_library_module_id(int module_idx) {
    return "module_" + std::to_string(module_idx);
}

std::string gp_module_library_tool_source_path(const std::string& root_dir, const std::string& tool_id) {
    return join_path(root_dir, std::string(kSourceDir) + "/" + std::string(kToolDir), tool_id + ".cpp");
}

std::string gp_module_library_module_source_path(const std::string& root_dir, const std::string& module_id) {
    return join_path(root_dir, std::string(kSourceDir) + "/" + std::string(kModuleDir), module_id + ".cpp");
}

std::string gp_module_library_module_serialized_path(const std::string& root_dir, const std::string& module_id) {
    return join_path(root_dir, std::string(kSerializedDir), module_id + ".gpmod");
}

const char* gp_module_tool_kind_name(ModuleToolKind kind) {
    switch (kind) {
        case ModuleToolKind::Add: return "Add";
        case ModuleToolKind::Subtract: return "Subtract";
        case ModuleToolKind::Multiply: return "Multiply";
        case ModuleToolKind::Divide: return "Divide";
        case ModuleToolKind::Modulo: return "Modulo";
        case ModuleToolKind::KeyboardListener: return "KeyboardListener";
        case ModuleToolKind::MouseListener: return "MouseListener";
        case ModuleToolKind::StackDisplay: return "StackDisplay";
        case ModuleToolKind::RectRgba: return "RectRgba";
        case ModuleToolKind::TableNumber: return "TableNumber";
        case ModuleToolKind::Clone: return "Clone";
        case ModuleToolKind::None:
        default:
            return "None";
    }
}

int gp_module_library_write_to_file(const GP_ModuleLibrary& library, const char* path) {
    if (!path) return 0;
    std::ofstream ofs(path);
    if (!ofs.good()) return 0;
    ofs << "MODULELIB V1\n";
    ofs << "ROOT " << std::quoted(library.root_dir) << "\n";
    for (const auto& tool : library.tool_registry) {
        ofs << "TOOL " << tool.id << " " << static_cast<int>(tool.kind) << " "
            << std::quoted(tool.name) << " " << std::quoted(tool.source_path) << "\n";
    }
    for (const auto& module : library.modules) {
        ofs << "MODULE " << module.id << " " << module.module_idx << " "
            << std::quoted(module.label) << " " << std::quoted(module.serialized_path) << " "
            << std::quoted(module.source_path) << " " << module.convert_to_tool << " "
            << std::quoted(module.tool_id) << " " << module.tool_caps << "\n";
        for (const auto& tool_instance : module.tool_instances) {
            ofs << "MODULETOOL " << module.id << " " << tool_instance.row_idx << " "
                << tool_instance.tool_id << " " << tool_instance.attachment_count << "\n";
        }
    }
    return 1;
}

int gp_module_library_read_from_file(const char* path, GP_ModuleLibrary* out_library) {
    if (!path || !out_library) return 0;
    std::ifstream ifs(path);
    if (!ifs.good()) return 0;
    GP_ModuleLibrary library{};
    std::string line;
    if (!std::getline(ifs, line)) return 0;
    if (line.rfind("MODULELIB", 0) != 0) return 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "ROOT") {
            ss >> std::quoted(library.root_dir);
        } else if (tag == "TOOL") {
            GP_ModuleLibraryTool tool{};
            int kind_int = 0;
            ss >> tool.id >> kind_int >> std::quoted(tool.name) >> std::quoted(tool.source_path);
            tool.kind = static_cast<ModuleToolKind>(kind_int);
            library.tool_registry.push_back(std::move(tool));
        } else if (tag == "MODULE") {
            GP_ModuleLibraryModule module{};
            ss >> module.id >> module.module_idx >> std::quoted(module.label)
               >> std::quoted(module.serialized_path) >> std::quoted(module.source_path);
            if (ss.good()) {
                int convert_flag = 0;
                ss >> convert_flag >> std::quoted(module.tool_id) >> module.tool_caps;
                module.convert_to_tool = (convert_flag != 0);
            }
            library.modules.push_back(std::move(module));
        } else if (tag == "MODULETOOL") {
            std::string module_id;
            GP_ModuleToolInstance instance{};
            ss >> module_id >> instance.row_idx >> instance.tool_id >> instance.attachment_count;
            for (auto& module : library.modules) {
                if (module.id == module_id) {
                    module.tool_instances.push_back(std::move(instance));
                    break;
                }
            }
        }
    }
    if (library.root_dir.empty()) {
        library.root_dir = gp_module_library_default_root();
    }
    *out_library = std::move(library);
    return 1;
}
