#include "module_library_actualizer.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>
#include <cstdio>
#include <unordered_map>
#include <vector>

#include "tool_api.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
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
    std::filesystem::path tmp;
    try {
        if (!ensure_parent_dir(target, err)) return false;
        // create a temp file name in the same directory
        std::random_device rd;
        std::mt19937_64 gen(rd());
        uint64_t v = gen();
        std::ostringstream ss;
        ss << std::hex << v;
        tmp = target.parent_path() / (target.filename().string() + ".tmp-" + ss.str());

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
        try { if (!tmp.empty() && std::filesystem::exists(tmp)) std::filesystem::remove(tmp); } catch (...) {}
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

std::string sanitize_identifier(const std::string& id) {
    std::string out;
    out.reserve(id.size());
    for (char ch : id) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) {
            out.push_back(ch);
        } else {
            out.push_back('_');
        }
    }
    if (out.empty() || (out[0] >= '0' && out[0] <= '9')) {
        out.insert(out.begin(), '_');
    }
    return out;
}

struct ToolPortPlan {
    int32_t args = 0;
    int32_t internal = 0;
    int32_t returns = 0;
};

ToolPortPlan build_port_plan(const std::vector<ModuleToolKind>& tools) {
    int32_t stack = 0;
    int32_t args = 0;
    int32_t produced_total = 0;
    for (ModuleToolKind tool : tools) {
        ToolStackSpec spec = module_tool_stack_spec(tool);
        produced_total += spec.produces;
        if (spec.consumes > stack) {
            args += spec.consumes - stack;
            stack = 0;
        } else {
            stack -= spec.consumes;
        }
        stack += spec.produces;
    }
    ToolPortPlan plan{};
    plan.args = args;
    plan.returns = stack;
    plan.internal = std::max(0, produced_total - plan.returns);
    return plan;
}

std::string generate_tool_source(const GP_ModuleLibraryModule& module,
                                 const std::string& tool_id,
                                 const std::string& tool_name,
                                 const ToolPortPlan& plan) {
    std::ostringstream ss;
    std::string class_name = "Tool_" + sanitize_identifier(tool_id);

    ss << "// Generated from module " << module.id << "\n";
    ss << "#include \"tool_api.h\"\n\n";
    ss << "#include <cstdint>\n";
    ss << "#include <string>\n\n";
    ss << "class " << class_name << " : public ITool {\n";
    ss << "public:\n";
    ss << "    std::string id() const override { return \"" << tool_id << "\"; }\n";
    ss << "    std::string name() const override { return \"" << tool_name << "\"; }\n";
    ss << "    ToolCaps caps() const override { return static_cast<ToolCaps>(" << module.tool_caps << "); }\n\n";
    ss << "    void initialize(const ToolInitContext& ctx) override {\n";
    ss << "        if (ctx.serialized_path && ctx.serialized_path[0] != '\\0') {\n";
    ss << "            FileInputArchive in(ctx.serialized_path);\n";
    ss << "            if (in.ok()) {\n";
    ss << "                deserialize(in);\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "    }\n\n";
    ss << "    void shutdown() override {}\n\n";
    ss << "    void tick(double /*dt*/, HostAPI& /*host*/) override {}\n\n";
    ss << "    void render(RenderContext& /*ctx*/) override {}\n\n";

    ss << "    int32_t port_count() const override {\n";
    ss << "        return static_cast<int32_t>(kPortCount);\n";
    ss << "    }\n\n";
    ss << "    ToolPortSpec port_spec(int32_t idx) const override {\n";
    ss << "        if (idx < 0 || idx >= port_count()) return ToolPortSpec{};\n";
    ss << "        return kPorts[idx];\n";
    ss << "    }\n\n";
    ss << "private:\n";

    // decide how many ports we will emit
    int k_total_ports = 0;
    if (plan.args > 0) k_total_ports += 1;
    if (plan.internal > 0) k_total_ports += 1;
    if (plan.returns > 0) k_total_ports += 1;

    if (k_total_ports > 0) {
        ss << "    static constexpr ToolPortSpec kPorts[] = {\n";
        if (plan.args > 0) {
            ss << "        {ToolPortKind::Argument, " << plan.args << "},\n";
        }
        if (plan.internal > 0) {
            ss << "        {ToolPortKind::Internal, " << plan.internal << "},\n";
        }
        if (plan.returns > 0) {
            ss << "        {ToolPortKind::Return, " << plan.returns << "},\n";
        }
        ss << "    };\n";
        ss << "    static constexpr int kPortCount = static_cast<int>(sizeof(kPorts) / sizeof(kPorts[0]));\n";
    } else {
        ss << "    static constexpr const ToolPortSpec* kPorts = nullptr;\n";
        ss << "    static constexpr int kPortCount = 0;\n";
    }

    ss << "};\n\n";
    ss << "#if defined(_WIN32)\n";
    ss << "extern \"C\" __declspec(dllexport) ITool* create_tool() {\n";
    ss << "#else\n";
    ss << "extern \"C\" ITool* create_tool() {\n";
    ss << "#endif\n";
    ss << "    return new " << class_name << "();\n";
    ss << "}\n";

    return ss.str();
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

    std::unordered_map<std::string, ModuleToolKind> tool_kind_by_id;
    tool_kind_by_id.reserve(library.tool_registry.size());
    for (const auto& tool : library.tool_registry) {
        tool_kind_by_id.emplace(tool.id, tool.kind);
    }

    for (const auto& module : library.modules) {
        if (module.convert_to_tool) {
            std::string tool_id = module.tool_id.empty() ? ("tool_" + module.id) : module.tool_id;
            std::filesystem::path tool_path = gp_module_library_tool_source_path(root.string(), tool_id);
            if (!tool_path.is_absolute()) tool_path = root / tool_path;

            std::vector<GP_ModuleToolInstance> sorted_tools = module.tool_instances;
            std::sort(sorted_tools.begin(), sorted_tools.end(),
                      [](const GP_ModuleToolInstance& a, const GP_ModuleToolInstance& b) {
                          return a.row_idx < b.row_idx;
                      });
            std::vector<ModuleToolKind> tool_stack;
            tool_stack.reserve(sorted_tools.size());
            for (const auto& instance : sorted_tools) {
                auto it = tool_kind_by_id.find(instance.tool_id);
                tool_stack.push_back(it == tool_kind_by_id.end() ? ModuleToolKind::None : it->second);
            }

            ToolPortPlan plan = build_port_plan(tool_stack);
            std::string tool_name = module.label.empty() ? tool_id : module.label;
            std::string contents = generate_tool_source(module, tool_id, tool_name, plan);
            if (!write_placeholder_if_missing(tool_path, contents, err)) {
                std::cerr << "error writing tool source '" << tool_path.string() << "': " << err << "\n";
                ok = false;
            }
        } else {
            std::filesystem::path module_path = module.source_path.empty()
                ? std::filesystem::path(gp_module_library_module_source_path(root.string(), module.id))
                : std::filesystem::path(module.source_path);
            if (!module_path.is_absolute()) module_path = root / module_path;
            std::string contents = "// Placeholder module source for " + module.id + "\n";
            if (!write_placeholder_if_missing(module_path, contents, err)) {
                std::cerr << "error writing module source '" << module_path.string() << "': " << err << "\n";
                ok = false;
            }
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
