#pragma once

#include <string>
#include <vector>
#include "thread_manager.h"

struct GP_ModuleLibraryTool {
    std::string id;
    ModuleToolKind kind = ModuleToolKind::None;
    std::string name;
    std::string source_path;
};

struct GP_ModuleToolInstance {
    int row_idx = 0;
    int attachment_count = 0;
    std::string tool_id;
};

struct GP_ModuleLibraryModule {
    std::string id;
    int module_idx = -1;
    std::string label;
    std::string serialized_path;
    std::string source_path;
    std::vector<GP_ModuleToolInstance> tool_instances;
};

struct GP_ModuleLibrary {
    std::string root_dir;
    std::vector<GP_ModuleLibraryTool> tool_registry;
    std::vector<GP_ModuleLibraryModule> modules;
};

std::string gp_module_library_default_root();
std::string gp_module_library_tool_id(ModuleToolKind kind);
std::string gp_module_library_module_id(int module_idx);
std::string gp_module_library_tool_source_path(const std::string& root_dir, const std::string& tool_id);
std::string gp_module_library_module_source_path(const std::string& root_dir, const std::string& module_id);
std::string gp_module_library_module_serialized_path(const std::string& root_dir, const std::string& module_id);
const char* gp_module_tool_kind_name(ModuleToolKind kind);

int gp_module_library_write_to_file(const GP_ModuleLibrary& library, const char* path);
int gp_module_library_read_from_file(const char* path, GP_ModuleLibrary* out_library);
