#include "tool_api.h"
#include "kpath_raster_box.h"
#include "common/tensors/abstraction/kpath/kpath_raster_c_abi.h"

#include <memory>

using namespace nodus::tensors::kpath;

// Wraps rasterize_program_scatter_with_spatial_kernel as a standalone nodus
// plugin tool -- the scatter/gather sibling of KpathRasterKernelTool
// (kpath_raster_kernel_tool.cpp; see that file's header comment for the
// shared design notes on the plugin-DLL pattern, capability tagging, and why
// this calls through the gp_kpath_* C-ABI wrappers + links canvas_tables
// (shared) instead of calling nodus::tensors::kpath:: directly).
//
// Input: pops a KpathArmatureProgramBox* (VT_VOID_PTR) off the stack and
// takes ownership of it.
// Output: on success, pushes a KpathRasterResultBox* (VT_VOID_PTR) with the
// rasterized energy/temp canvases. On failure (the underlying call returns
// false), the input is still consumed and nothing is pushed.
class KpathRasterScatterTool final : public ITool {
public:
    const char* id_cstr() const noexcept override { return "kpath_raster_scatter"; }
    const char* name_cstr() const noexcept override { return "KPath Raster (Scatter)"; }
    ToolCaps caps() const override { return ToolCaps::None; }

    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}

    void execute_stack(ToolStackContext& ctx) override {
        if (!ctx.stack.raw) return;
        RawStackFrame& frame = *ctx.stack.raw;

        KpathArmatureProgramBox* input = kpath_pop_armature_program_box(frame);
        if (!input) return;

        MachineControlConfig machine;
        machine.step_px = 0.9f;
        machine.energy_per_px = 1.0f;
        machine.enable_thermal_guard = false;

        GaussianToolParams gaussian_tool{};
        const float render_scale = 1.0f;
        const float margin_px = 12.0f;
        ProgramRasterTransform xform = gp_kpath_plan_program_raster_transform_refined(
            input->program, machine, render_scale, margin_px, gaussian_tool);

        BeamToolParams kernel_tool{};
        kernel_tool.falloff = BeamFalloffKind::Gaussian;
        kernel_tool.sigma_px = gaussian_tool.sigma_px;

        ThermalSimConfig sim{};
        auto result = std::make_unique<KpathRasterResultBox>();
        const bool ok = gp_kpath_rasterize_program_scatter_with_spatial_kernel(
            input->program, result->energy, result->temp, machine, xform, kernel_tool,
            nullptr, sim, nullptr, nullptr, nullptr);
        delete input;

        if (ok) {
            kpath_push_raster_result_box(frame, result.release());
        }
    }
};

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new KpathRasterScatterTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) { delete t; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
