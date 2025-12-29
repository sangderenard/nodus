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
    int32_t produced_total = 0;
};

struct ToolStep {
    ModuleToolKind kind = ModuleToolKind::None;
    int attachment_count = 0;
    std::string plugin_id;
    int plugin_slot = -1;
};

struct PluginSourceInfo {
    std::string id;
    std::string include_path;
};

struct PluginSlotInfo {
    std::string create_fn;
    std::string destroy_fn;
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
    plan.produced_total = produced_total;
    return plan;
}

int compute_stack_after_tools(const std::vector<ModuleToolKind>& tools, int initial_stack) {
    int stack = std::max(0, initial_stack);
    for (ModuleToolKind tool : tools) {
        ToolStackSpec spec = module_tool_stack_spec(tool);
        int consumes = std::max(0, spec.consumes);
        int produces = std::max(0, spec.produces);
        stack = std::max(0, stack - consumes) + produces;
    }
    return stack;
}

void emit_stack_step(std::ostringstream& ss, ModuleToolKind kind, int attachment_count) {
    switch (kind) {
        case ModuleToolKind::KeyboardListener:
            ss << "        {\n";
            ss << "            float val = 0.0f;\n";
            ss << "            if (input && input->key_event) {\n";
            ss << "                val = static_cast<float>(input->key);\n";
            ss << "            }\n";
            ss << "            tool_stack_push(stack, val);\n";
            ss << "        }\n";
            break;
        case ModuleToolKind::MouseListener:
            ss << "        {\n";
            ss << "            float mx = 0.0f;\n";
            ss << "            float my = 0.0f;\n";
            ss << "            float down = 0.0f;\n";
            ss << "            float up = 0.0f;\n";
            ss << "            if (input) {\n";
            ss << "                mx = input->mouse_x;\n";
            ss << "                my = input->mouse_y;\n";
            ss << "                down = input->mouse_down ? 1.0f : 0.0f;\n";
            ss << "                up = input->mouse_up ? 1.0f : 0.0f;\n";
            ss << "            }\n";
            ss << "            tool_stack_push(stack, up);\n";
            ss << "            tool_stack_push(stack, down);\n";
            ss << "            tool_stack_push(stack, my);\n";
            ss << "            tool_stack_push(stack, mx);\n";
            ss << "        }\n";
            break;
        case ModuleToolKind::StackDisplay:
            ss << "        {\n";
            ss << "            // StackDisplay: snapshot handled externally; no stack mutation here.\n";
            ss << "        }\n";
            break;
        case ModuleToolKind::TableNumber:
            ss << "        {\n";
            ss << "            float val = static_cast<float>(std::max(0, " << attachment_count << "));\n";
            ss << "            tool_stack_push(stack, val);\n";
            ss << "        }\n";
            break;
        case ModuleToolKind::Clone:
            ss << "        {\n";
            ss << "            float count_f = tool_stack_pop(stack);\n";
            ss << "            float value = tool_stack_pop(stack);\n";
            ss << "            int count = std::max(0, static_cast<int>(std::lround(count_f)));\n";
            ss << "            for (int i = 0; i < count; ++i) {\n";
            ss << "                tool_stack_push(stack, value);\n";
            ss << "            }\n";
            ss << "        }\n";
            break;
        case ModuleToolKind::RectRgba:
            ss << "        {\n";
            ss << "            float height = tool_stack_pop(stack);\n";
            ss << "            float width = tool_stack_pop(stack);\n";
            ss << "            float bg[4] = { tool_stack_pop(stack), tool_stack_pop(stack), tool_stack_pop(stack), tool_stack_pop(stack) };\n";
            ss << "            float border[4] = { tool_stack_pop(stack), tool_stack_pop(stack), tool_stack_pop(stack), tool_stack_pop(stack) };\n";
            ss << "            float border_width = tool_stack_pop(stack);\n";
            ss << "            float corner_radius = tool_stack_pop(stack);\n";
            ss << "            int raster_h = std::max(0, static_cast<int>(std::lround(height)));\n";
            ss << "            int raster_w = std::max(0, static_cast<int>(std::lround(width)));\n";
            ss << "            float out = 0.0f;\n";
            ss << "            uint64_t cycle = tool_cycle_++;\n";
            ss << "            if (raster_w > 0 && raster_h > 0) {\n";
            ss << "                uint64_t pixel_count = static_cast<uint64_t>(raster_w) * static_cast<uint64_t>(raster_h);\n";
            ss << "                uint64_t total = pixel_count * 4ull;\n";
            ss << "                if (total > 0) {\n";
            ss << "                    uint64_t idx = cycle % total;\n";
            ss << "                    int channel = static_cast<int>(idx % 4ull);\n";
            ss << "                    uint64_t pix = idx / 4ull;\n";
            ss << "                    int x = static_cast<int>(pix % static_cast<uint64_t>(raster_w));\n";
            ss << "                    int y = static_cast<int>(pix / static_cast<uint64_t>(raster_w));\n";
            ss << "                    float px = static_cast<float>(x) + 0.5f;\n";
            ss << "                    float py = static_cast<float>(y) + 0.5f;\n";
            ss << "                    float fw = static_cast<float>(raster_w);\n";
            ss << "                    float fh = static_cast<float>(raster_h);\n";
            ss << "                    float bw = std::max(0.0f, border_width);\n";
            ss << "                    float cr = std::max(0.0f, corner_radius);\n";
            ss << "                    bool inside = point_in_rounded_rect(px, py, fw, fh, cr);\n";
            ss << "                    if (inside) {\n";
            ss << "                        bool use_bg = true;\n";
            ss << "                        float inner_w = fw - 2.0f * bw;\n";
            ss << "                        float inner_h = fh - 2.0f * bw;\n";
            ss << "                        if (bw > 0.0f && inner_w > 0.0f && inner_h > 0.0f) {\n";
            ss << "                            float inner_r = std::max(0.0f, cr - bw);\n";
            ss << "                            use_bg = point_in_rounded_rect(px - bw, py - bw, inner_w, inner_h, inner_r);\n";
            ss << "                        }\n";
            ss << "                        const float* src = use_bg ? bg : border;\n";
            ss << "                        out = src[channel];\n";
            ss << "                    }\n";
            ss << "                }\n";
            ss << "            }\n";
            ss << "            tool_stack_push(stack, out);\n";
            ss << "        }\n";
            break;
        case ModuleToolKind::Add:
        case ModuleToolKind::Subtract:
        case ModuleToolKind::Multiply:
        case ModuleToolKind::Divide:
        case ModuleToolKind::Modulo:
        case ModuleToolKind::None:
        default:
            ss << "        {\n";
            ss << "            float b = tool_stack_pop(stack);\n";
            ss << "            float a = tool_stack_pop(stack);\n";
            switch (kind) {
                case ModuleToolKind::Add:
                    ss << "            tool_stack_push(stack, a + b);\n";
                    break;
                case ModuleToolKind::Subtract:
                    ss << "            tool_stack_push(stack, a - b);\n";
                    break;
                case ModuleToolKind::Multiply:
                    ss << "            tool_stack_push(stack, a * b);\n";
                    break;
                case ModuleToolKind::Divide:
                    ss << "            tool_stack_push(stack, (b == 0.0f) ? 0.0f : (a / b));\n";
                    break;
                case ModuleToolKind::Modulo:
                    ss << "            tool_stack_push(stack, (b == 0.0f) ? 0.0f : std::fmod(a, b));\n";
                    break;
                case ModuleToolKind::None:
                default:
                    ss << "            (void)a; (void)b;\n";
                    break;
            }
            ss << "        }\n";
            break;
    }
}

void emit_stack_execution(std::ostringstream& ss, const std::vector<ToolStep>& steps) {
    ss << "    void execute_stack(ToolStackContext& ctx) override {\n";
    ss << "        ToolStackFrame& stack = ctx.stack;\n";
    ss << "        const ToolInputState* input = ctx.input;\n";
    ss << "        (void)input;\n";
    for (size_t i = 0; i < steps.size(); ++i) {
        const auto& step = steps[i];
        if (!step.plugin_id.empty()) {
            ss << "        // step " << i << ": plugin " << step.plugin_id << "\n";
            if (step.plugin_slot >= 0) {
                ss << "        if (plugin_instances_[" << step.plugin_slot << "]) {\n";
                ss << "            plugin_instances_[" << step.plugin_slot << "]->execute_stack(ctx);\n";
                ss << "        }\n";
            }
            continue;
        }
        ss << "        // step " << i << ": " << gp_module_tool_kind_name(step.kind) << "\n";
        emit_stack_step(ss, step.kind, step.attachment_count);
    }
    ss << "    }\n\n";
}

std::string generate_tool_source(const GP_ModuleLibraryModule& module,
                                 const std::string& tool_id,
                                 const std::string& tool_name,
                                 const ToolPortPlan& plan,
                                 const std::vector<ToolStep>& steps,
                                 const std::vector<PluginSourceInfo>& plugin_sources,
                                 const std::vector<PluginSlotInfo>& plugin_slots) {
    std::ostringstream ss;
    std::string class_name = "Tool_" + sanitize_identifier(tool_id);
    bool has_rect = false;
    bool needs_mouse_bindings = false;
    bool needs_keyboard_bindings = false;
    int mouse_port_need = 0;
    int keyboard_port_need = 0;
    int plugin_slot_count = static_cast<int>(plugin_slots.size());
    for (const auto& step : steps) {
        if (step.kind == ModuleToolKind::RectRgba) {
            has_rect = true;
            break;
        }
    }
    for (const auto& step : steps) {
        if (!step.plugin_id.empty()) continue; // external plugin step: skip
        if (step.kind == ModuleToolKind::MouseListener) {
            needs_mouse_bindings = true;
            mouse_port_need = std::max(mouse_port_need, 4); // x,y,down,up
        } else if (step.kind == ModuleToolKind::KeyboardListener) {
            needs_keyboard_bindings = true;
            keyboard_port_need = std::max(keyboard_port_need, 1);
        }
    }

    ss << "// Generated from module " << module.id << "\n";
    ss << "#include \"tool_api.h\"\n\n";
    if (needs_mouse_bindings || needs_keyboard_bindings) {
        ss << "#include \"canvas_abi.h\"\n";
        ss << "#include \"tool_events.h\"\n";
    }
    ss << "#include <cstdint>\n";
    ss << "#include <cmath>\n";
    ss << "#include <algorithm>\n";
    ss << "#include <string>\n";
    ss << "#include <vector>\n\n";
    ss << "// Tool stack (in order):\n";
    if (steps.empty()) {
        ss << "//  (empty)\n";
    } else {
        for (const auto& step : steps) {
            if (!step.plugin_id.empty()) {
                ss << "//  - plugin " << step.plugin_id << " (attachments=" << step.attachment_count << ")\n";
            } else {
                ss << "//  - " << gp_module_tool_kind_name(step.kind) << " (attachments=" << step.attachment_count << ")\n";
            }
        }
    }
    ss << "\n";
    if (!plugin_sources.empty()) {
        for (const auto& entry : plugin_sources) {
            std::string sym = sanitize_identifier(entry.id);
            ss << "namespace nodus_plugin_" << sym << " {\n";
            ss << "#define NODUS_PLUGIN_COMPOSITE 1\n";
            ss << "#define NODUS_PLUGIN_FACTORY_NAME nodus_create_tool_" << sym << "\n";
            ss << "#define NODUS_PLUGIN_DESTROY_NAME nodus_destroy_tool_" << sym << "\n";
            ss << "#define NODUS_PLUGIN_SOURCE_NAME nodus_plugin_source_path_" << sym << "\n";
            ss << "#define create_tool nodus_create_tool_" << sym << "\n";
            ss << "#define destroy_tool nodus_destroy_tool_" << sym << "\n";
            ss << "#define plugin_source_path nodus_plugin_source_path_" << sym << "\n";
            ss << "#include \"" << entry.include_path << "\"\n";
            ss << "#undef plugin_source_path\n";
            ss << "#undef destroy_tool\n";
            ss << "#undef create_tool\n";
            ss << "#undef NODUS_PLUGIN_SOURCE_NAME\n";
            ss << "#undef NODUS_PLUGIN_DESTROY_NAME\n";
            ss << "#undef NODUS_PLUGIN_FACTORY_NAME\n";
            ss << "#undef NODUS_PLUGIN_COMPOSITE\n";
            ss << "} // namespace nodus_plugin_" << sym << "\n\n";
        }
    }
    if (has_rect) {
        ss << "static bool point_in_rounded_rect(float px, float py, float w, float h, float r) {\n";
        ss << "    if (r <= 0.0f) return (px >= 0.0f && py >= 0.0f && px <= w && py <= h);\n";
        ss << "    float inner_w = std::max(0.0f, w - 2.0f * r);\n";
        ss << "    float inner_h = std::max(0.0f, h - 2.0f * r);\n";
        ss << "    if (px >= r && px <= r + inner_w && py >= 0.0f && py <= h) return true;\n";
        ss << "    if (py >= r && py <= r + inner_h && px >= 0.0f && px <= w) return true;\n";
        ss << "    float dx = 0.0f;\n";
        ss << "    float dy = 0.0f;\n";
        ss << "    if (px < r) dx = px - r; else if (px > w - r) dx = px - (w - r);\n";
        ss << "    if (py < r) dy = py - r; else if (py > h - r) dy = py - (h - r);\n";
        ss << "    return (dx * dx + dy * dy) <= (r * r);\n";
        ss << "}\n\n";
    }
    ss << "class " << class_name << " : public ITool {\n";
    ss << "public:\n";
    ss << "    const char* id_cstr() const noexcept override { return \"" << tool_id << "\"; }\n";
    ss << "    const char* name_cstr() const noexcept override { return \"" << tool_name << "\"; }\n";
    // `id()`/`name()` are provided by the base class and will call the
    // plugin's `id_cstr()`/`name_cstr()` implementation.
    ss << "    ToolCaps caps() const override { return static_cast<ToolCaps>(" << module.tool_caps << "); }\n\n";
    ss << "    void initialize(const ToolInitContext& ctx) override {\n";
    ss << "        if (ctx.serialized_path && ctx.serialized_path[0] != '\\0') {\n";
    ss << "            FileInputArchive in(ctx.serialized_path);\n";
    ss << "            if (in.ok()) {\n";
    ss << "                deserialize(in);\n";
    ss << "            }\n";
    ss << "        }\n";
    // Binding of module frame ports is deferred to the host when the tool is
    // actually placed on a module. Do not autobind on load/registration.
    if (plugin_slot_count > 0) {
        ss << "        ToolInitContext sub = ctx;\n";
        ss << "        sub.serialized_path = nullptr;\n";
        ss << "        for (int i = 0; i < kPluginSlotCount; ++i) {\n";
            ss << "            if (!plugin_instances_[i] && kPluginCreateFns[i]) {\n";
            ss << "                plugin_instances_[i] = kPluginCreateFns[i]();\n";
            ss << "                if (plugin_instances_[i]) {\n";
                ss << "                    plugin_instances_[i]->initialize(sub);\n";
            ss << "                }\n";
            ss << "            }\n";
        ss << "        }\n";
    }
    ss << "    }\n\n";
    ss << "    void shutdown() override {\n";
    if (plugin_slot_count > 0) {
        ss << "        for (int i = 0; i < kPluginSlotCount; ++i) {\n";
        ss << "            if (plugin_instances_[i]) {\n";
        ss << "                plugin_instances_[i]->shutdown();\n";
        ss << "                if (kPluginDestroyFns[i]) kPluginDestroyFns[i](plugin_instances_[i]);\n";
        ss << "                plugin_instances_[i] = nullptr;\n";
        ss << "            }\n";
        ss << "        }\n";
    }
    ss << "    }\n\n";
    ss << "    void tick(double /*dt*/, HostAPI& /*host*/) override {}\n\n";
    ss << "    void render(RenderContext& /*ctx*/) override {}\n\n";

    emit_stack_execution(ss, steps);

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

    if (plugin_slot_count > 0) {
        ss << "    using CreateFn = ITool* (*)();\n";
        ss << "    using DestroyFn = void (*)(ITool*);\n";
        ss << "    static constexpr int kPluginSlotCount = " << plugin_slot_count << ";\n";
        ss << "    static const CreateFn kPluginCreateFns[kPluginSlotCount];\n";
        ss << "    static const DestroyFn kPluginDestroyFns[kPluginSlotCount];\n";
        ss << "    ITool* plugin_instances_[kPluginSlotCount] = {};\n";
    }

    if (has_rect) {
        ss << "    uint64_t tool_cycle_ = 0;\n";
    }
    ss << "};\n\n";
    if (plugin_slot_count > 0) {
        ss << "const " << class_name << "::CreateFn " << class_name << "::kPluginCreateFns[kPluginSlotCount] = {\n";
        for (size_t i = 0; i < plugin_slots.size(); ++i) {
            ss << "    " << plugin_slots[i].create_fn << ",\n";
        }
        ss << "};\n";
        ss << "const " << class_name << "::DestroyFn " << class_name << "::kPluginDestroyFns[kPluginSlotCount] = {\n";
        for (size_t i = 0; i < plugin_slots.size(); ++i) {
            ss << "    " << plugin_slots[i].destroy_fn << ",\n";
        }
        ss << "};\n\n";
    }
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() {\n";
    ss << "    return new " << class_name << "();\n";
    ss << "}\n";

    ss << "extern \"C\" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) {\n";
    ss << "    delete t;\n";
    ss << "}\n\n";

    ss << "extern \"C\" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() {\n";
    ss << "    return __FILE__;\n";
    ss << "}\n\n";
    // Optional autobind hints for host: how many mouse/keyboard ports to bind when placed on a module.
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT int nodus_autobind_mouse_ports() { return " << std::max(0, mouse_port_need) << "; }\n";
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT int nodus_autobind_keyboard_ports() { return " << std::max(0, keyboard_port_need) << "; }\n\n";

    // optional lifecycle hooks
    ss << "#if defined(_WIN32)\n";
    ss << "extern \"C\" __declspec(dllexport) int plugin_init(HostAPI* host) {\n";
    ss << "#else\n";
    ss << "extern \"C\" int plugin_init(HostAPI* host) {\n";
    ss << "#endif\n";
    ss << "    (void)host;\n";
    ss << "    return 1;\n";
    ss << "}\n\n";

    ss << "#if defined(_WIN32)\n";
    ss << "extern \"C\" __declspec(dllexport) void plugin_shutdown() {\n";
    ss << "#else\n";
    ss << "extern \"C\" void plugin_shutdown() {\n";
    ss << "#endif\n";
    ss << "}\n";

    return ss.str();
}

std::string generate_builtin_tool_source(const GP_ModuleLibraryTool& tool) {
    std::vector<ToolStep> steps;
    steps.push_back(ToolStep{tool.kind, 1});
    ToolPortPlan plan = build_port_plan({tool.kind});
    GP_ModuleLibraryModule dummy{};
    dummy.id = tool.id;
    dummy.tool_caps = 0;
    std::vector<PluginSourceInfo> empty_sources;
    std::vector<PluginSlotInfo> empty_slots;
    return generate_tool_source(dummy, tool.id, tool.name.empty() ? tool.id : tool.name, plan, steps, empty_sources, empty_slots);
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
            ? std::filesystem::path(gp_module_library_tool_source_path(std::string(), tool.id))
            : std::filesystem::path(tool.source_path);
        if (!tool_path.is_absolute()) tool_path = root / tool_path;
        std::string contents;
        if (tool.kind != ModuleToolKind::None) {
            contents = generate_builtin_tool_source(tool);
        } else {
            contents = "// Placeholder tool source for " + tool.id + " (" + tool.name + ")\n";
        }
        if (!write_placeholder_if_missing(tool_path, contents, err)) {
            std::cerr << "error writing tool source '" << tool_path.string() << "': " << err << "\n";
            ok = false;
        }
    }

    std::unordered_map<std::string, ModuleToolKind> tool_kind_by_id;
    tool_kind_by_id.reserve(library.tool_registry.size());
    std::unordered_map<std::string, std::string> tool_source_by_id;
    tool_source_by_id.reserve(library.tool_registry.size());
    for (const auto& tool : library.tool_registry) {
        tool_kind_by_id.emplace(tool.id, tool.kind);
        if (!tool.source_path.empty()) {
            tool_source_by_id.emplace(tool.id, tool.source_path);
        }
    }

    for (const auto& module : library.modules) {
        if (module.convert_to_tool) {
            std::string tool_id = module.tool_id.empty() ? ("tool_" + module.id) : module.tool_id;
            std::filesystem::path tool_path = gp_module_library_tool_source_path(std::string(), tool_id);
            if (!tool_path.is_absolute()) tool_path = root / tool_path;

            std::vector<GP_ModuleToolInstance> sorted_tools = module.tool_instances;
            std::sort(sorted_tools.begin(), sorted_tools.end(),
                      [](const GP_ModuleToolInstance& a, const GP_ModuleToolInstance& b) {
                          return a.row_idx < b.row_idx;
                      });
            std::vector<ModuleToolKind> tool_stack;
            tool_stack.reserve(sorted_tools.size());
            bool has_external_tools = false;
            std::vector<std::string> external_ids;
            std::vector<std::string> missing_source_ids;
            std::unordered_map<std::string, PluginSourceInfo> plugin_source_map;
            std::vector<PluginSourceInfo> plugin_sources;
            for (const auto& instance : sorted_tools) {
                auto it = tool_kind_by_id.find(instance.tool_id);
                ModuleToolKind kind = (it == tool_kind_by_id.end()) ? ModuleToolKind::None : it->second;
                if (kind == ModuleToolKind::None) {
                    if (!instance.tool_id.empty()) {
                        has_external_tools = true;
                        external_ids.push_back(instance.tool_id);
                        auto src_it = tool_source_by_id.find(instance.tool_id);
                        if (src_it != tool_source_by_id.end() && !src_it->second.empty()) {
                            if (plugin_source_map.find(instance.tool_id) == plugin_source_map.end()) {
                                std::filesystem::path src_path = std::filesystem::path(src_it->second);
                                if (!src_path.is_absolute()) src_path = root / src_path;
                                PluginSourceInfo info{};
                                info.id = instance.tool_id;
                                info.include_path = src_path.generic_string();
                                plugin_source_map.emplace(info.id, info);
                                plugin_sources.push_back(std::move(info));
                            }
                        } else {
                            missing_source_ids.push_back(instance.tool_id);
                        }
                    }
                }
                tool_stack.push_back(kind);
            }
            if (has_external_tools) {
                std::cerr << "module actualizer: module '" << module.id << "' uses external plugin tools:";
                for (const auto& id : external_ids) std::cerr << " " << id;
                std::cerr << "\n";
            }
            if (!missing_source_ids.empty()) {
                std::cerr << "module actualizer: missing source for plugin tools:";
                for (const auto& id : missing_source_ids) std::cerr << " " << id;
                std::cerr << "\n";
            }

            ToolPortPlan plan = build_port_plan(tool_stack);
            int initial_stack = plan.args;
            if (module.input_count >= 0) {
                initial_stack = module.input_count;
                plan.args = module.input_count;
            }
            int final_stack = compute_stack_after_tools(tool_stack, initial_stack);
            if (module.output_count >= 0) {
                plan.returns = module.output_count;
            } else {
                plan.returns = final_stack;
            }
            plan.internal = std::max(0, final_stack - plan.returns);
            std::string tool_name = module.label.empty() ? tool_id : module.label;
            std::vector<ToolStep> steps;
            steps.reserve(sorted_tools.size());
            std::vector<PluginSlotInfo> plugin_slots;
            for (const auto& instance : sorted_tools) {
                auto it = tool_kind_by_id.find(instance.tool_id);
                ModuleToolKind kind = it == tool_kind_by_id.end() ? ModuleToolKind::None : it->second;
                ToolStep step{};
                step.kind = kind;
                step.attachment_count = instance.attachment_count;
                if (kind == ModuleToolKind::None && !instance.tool_id.empty()) {
                    step.plugin_id = instance.tool_id;
                    auto src_it = tool_source_by_id.find(instance.tool_id);
                    if (src_it != tool_source_by_id.end() && !src_it->second.empty()) {
                        std::string sym = sanitize_identifier(instance.tool_id);
                        step.plugin_slot = static_cast<int>(plugin_slots.size());
                        PluginSlotInfo slot{};
                        slot.create_fn = "nodus_plugin_" + sym + "::nodus_create_tool_" + sym;
                        slot.destroy_fn = "nodus_plugin_" + sym + "::nodus_destroy_tool_" + sym;
                        plugin_slots.push_back(std::move(slot));
                    }
                }
                steps.push_back(std::move(step));
            }
            std::string contents = generate_tool_source(module, tool_id, tool_name, plan, steps, plugin_sources, plugin_slots);
            try {
                // create a deterministic, versioned filename: <tool_id>_ver_<YYYYMMDD_HHMMSS>.cpp
                auto now = std::chrono::system_clock::now();
                std::time_t t = std::chrono::system_clock::to_time_t(now);
                char buf[64];
#ifdef _WIN32
                std::tm tm; localtime_s(&tm, &t);
#else
                std::tm tm; localtime_r(&t, &tm);
#endif
                std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
                std::string ts(buf);
                std::filesystem::path dir = tool_path.parent_path();
                std::string base = tool_id + std::string("_ver_") + ts;
                std::filesystem::create_directories(dir);
                std::filesystem::path outpath;
                int suffix = 0;
                do {
                    std::ostringstream name;
                    name << base;
                    if (suffix > 0) name << "_" << suffix;
                    name << ".cpp";
                    outpath = dir / name.str();
                    ++suffix;
                } while (std::filesystem::exists(outpath));
                std::ofstream ofs(outpath, std::ios::binary | std::ios::trunc);
                if (!ofs.good()) {
                    std::cerr << "error writing tool source '" << outpath.string() << "'\n";
                    ok = false;
                } else {
                    std::cerr << "writing generated tool source to: " << outpath.generic_string() << "\n";
                    ofs << contents;
                    ofs.close();
                }
            } catch (const std::exception& e) {
                std::cerr << "exception writing tool source: " << e.what() << "\n";
                ok = false;
            }
        } else {
            std::filesystem::path module_path = module.source_path.empty()
                ? std::filesystem::path(gp_module_library_module_source_path(std::string(), module.id))
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
