#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "plugin_loader.h"
#include "tool_api.h"
#include "tool_registry.h"
#include "kpath_raster_box.h"

namespace fs = std::filesystem;
using namespace nodus::tensors::kpath;

namespace {

class StubHost final : public HostAPI {
public:
    void log(const char* message) override {
        if (message) std::cerr << message << "\n";
    }
    void draw_text(const TextRenderArgs&) override {}
};

fs::path find_dll(const fs::path& build_dir, const std::string& target_name) {
    const std::vector<fs::path> candidates = {
        build_dir / "Release" / (target_name + ".dll"),
        build_dir / (target_name + ".dll"),
    };
    for (const auto& c : candidates) {
        if (fs::exists(c)) return c;
    }
    return {};
}

// Same square-outline test program for both raster paths -- deliberately not
// the full house/sun/bridge scene from kpath_relgeo_scene_demo.cpp (that
// exercises kpath's IR compiler, already covered elsewhere per the handoff);
// this test is only verifying the plugin-tool wrapper round-trips correctly,
// not re-verifying kpath's rendering correctness.
ArmatureProgram make_test_program() {
    ArmatureProgram program;
    program.points.push_back(ToolPoint{0.0f, 0.0f, 0.0f, false});
    program.points.push_back(ToolPoint{40.0f, 0.0f, 0.0f, true});
    program.points.push_back(ToolPoint{40.0f, 40.0f, 0.0f, true});
    program.points.push_back(ToolPoint{0.0f, 40.0f, 0.0f, true});
    program.points.push_back(ToolPoint{0.0f, 0.0f, 0.0f, true});
    return program;
}

bool canvases_equal(const TensorCanvas2D& a, const TensorCanvas2D& b) {
    if (a.width != b.width || a.height != b.height) return false;
    if (a.values.size() != b.values.size()) return false;
    for (size_t i = 0; i < a.values.size(); ++i) {
        if (a.values[i] != b.values[i]) return false;
    }
    return true;
}

// Loads `dll_name` via PluginLoader, drives it through `frame` with `input`,
// and returns the popped KpathRasterResultBox (nullptr on any failure).
std::unique_ptr<KpathRasterResultBox> run_tool_via_plugin(PluginLoader& loader, StubHost& host,
                                                          const fs::path& build_dir,
                                                          const std::string& target_name,
                                                          const ArmatureProgram& input) {
    fs::path dll = find_dll(build_dir, target_name);
    if (dll.empty()) {
        std::cerr << "could not find built dll for target '" << target_name << "'\n";
        return nullptr;
    }
    std::string id = loader.load_module(dll.string(), &host);
    if (id.empty()) {
        std::cerr << "failed to load plugin dll: " << dll.string() << "\n";
        return nullptr;
    }
    auto tool = tool_registry_global().create(id);
    if (!tool) {
        std::cerr << "failed to create tool instance for: " << id << "\n";
        loader.unload_module(id);
        return nullptr;
    }
    ToolInitContext init_ctx{};
    tool->initialize(init_ctx);

    RawStackFrame raw{};
    if (!raw_stack_init_frame(raw, 4096)) {
        std::cerr << "failed to init raw stack frame\n";
        tool->shutdown();
        loader.unload_module(id);
        return nullptr;
    }

    auto* box = new KpathArmatureProgramBox();
    box->program = input;
    const ValueTypeId vt_ptr = ValueTypeRegistry::global().builtin(VT_VOID_PTR);
    void* p = box;
    if (!raw_stack_push_typed(raw, &p, vt_ptr)) {
        std::cerr << "failed to push input box onto stack\n";
        delete box;
        raw_stack_free_mask(raw);
        tool->shutdown();
        loader.unload_module(id);
        return nullptr;
    }

    ToolStackFrame stack_frame{};
    stack_frame.raw = &raw;
    ToolStackContext stack_ctx{};
    stack_ctx.stack = stack_frame;
    tool->execute_stack(stack_ctx);

    std::unique_ptr<KpathRasterResultBox> result;
    ValueTypeId top_tid = kInvalidValueTypeId;
    if (raw_stack_peek_type(raw, top_tid) && top_tid == vt_ptr) {
        void* out_ptr = nullptr;
        if (raw_stack_pop_typed(raw, &out_ptr, vt_ptr)) {
            auto* out_box = static_cast<KpathRasterResultBox*>(out_ptr);
            if (out_box && out_box->magic == KpathRasterResultBox::kMagic) {
                result.reset(out_box);
            }
            // If the magic doesn't match, this pointer is NOT a
            // KpathRasterResultBox we own -- do not delete it as one (that
            // would be a type-confused free). Leave it alone.
        }
    }

    raw_stack_free_mask(raw);
    tool->shutdown();
    tool.reset();
    loader.unload_module(id);
    std::cerr << "[checkpoint] unloaded\n";
    return result;
}

} // namespace

int main() {
#ifndef _WIN32
    std::cerr << "kpath_raster_tool_test skipped: plugin loader only implemented on Windows.\n";
    return 0;
#endif

    const fs::path build_dir = fs::path(NODUS_BUILD_DIR);
    const ArmatureProgram program = make_test_program();

    MachineControlConfig machine;
    machine.step_px = 0.9f;
    machine.energy_per_px = 1.0f;
    machine.enable_thermal_guard = false;
    GaussianToolParams gaussian_tool{};
    ProgramRasterTransform xform =
        plan_program_raster_transform_refined(program, machine, 1.0f, 12.0f, gaussian_tool);
    BeamToolParams kernel_tool{};
    kernel_tool.falloff = BeamFalloffKind::Gaussian;
    kernel_tool.sigma_px = gaussian_tool.sigma_px;

    StubHost host;

    // --- kernel-transform path ---
    {
        TensorCanvas2D ref_energy, ref_temp;
        rasterize_program_with_kernel_transformed(program, ref_energy, ref_temp, machine, kernel_tool, xform);
        if (ref_energy.max_value() <= 0.0f) {
            std::cerr << "test setup invariant broken: reference kernel raster is blank\n";
            return 1;
        }

        PluginLoader loader;
        auto result = run_tool_via_plugin(loader, host, build_dir, "kpath_raster_kernel_tool", program);
        if (!result) {
            std::cerr << "kpath_raster_kernel_tool produced no result\n";
            return 1;
        }
        if (!canvases_equal(result->energy, ref_energy) || !canvases_equal(result->temp, ref_temp)) {
            std::cerr << "kpath_raster_kernel_tool output does not match direct call\n";
            return 1;
        }
        std::cerr << "kpath_raster_kernel_tool ok (energy_max=" << result->energy.max_value() << ")\n";
    }

    // --- scatter path ---
    {
        TensorCanvas2D ref_energy, ref_temp;
        ThermalSimConfig sim{};
        const bool ref_ok = rasterize_program_scatter_with_spatial_kernel(
            program, ref_energy, ref_temp, machine, xform, kernel_tool, nullptr, sim, nullptr, nullptr, nullptr);
        if (!ref_ok || ref_energy.max_value() <= 0.0f) {
            std::cerr << "test setup invariant broken: reference scatter raster is blank or failed\n";
            return 1;
        }

        PluginLoader loader;
        auto result = run_tool_via_plugin(loader, host, build_dir, "kpath_raster_scatter_tool", program);
        if (!result) {
            std::cerr << "kpath_raster_scatter_tool produced no result\n";
            return 1;
        }
        if (!canvases_equal(result->energy, ref_energy) || !canvases_equal(result->temp, ref_temp)) {
            std::cerr << "kpath_raster_scatter_tool output does not match direct call\n";
            return 1;
        }
        std::cerr << "kpath_raster_scatter_tool ok (energy_max=" << result->energy.max_value() << ")\n";
    }

    return 0;
}
