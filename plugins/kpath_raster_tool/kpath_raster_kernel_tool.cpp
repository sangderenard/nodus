#include "tool_api.h"
#include "kpath_raster_box.h"
#include "common/tensors/abstraction/kpath/kpath_raster_c_abi.h"

using namespace nodus::tensors::kpath;

// Wraps rasterize_program_with_kernel_transformed as a standalone nodus
// plugin tool -- proves the plugin-DLL loading pattern (raw create_tool/
// destroy_tool exports, no REGISTER_TOOL/tool_registry.cpp link) that
// repo_package.cpp's ingestion routine uses for real cross-repo packages,
// with zero cross-repo risk since kpath lives in this same repo. Wraps
// already-compiled native code with no source-level decomposition offered --
// conceptually CAP_BINARY (see tool_api.h's ToolCapabilityTag): not
// KernelIR-decomposed today, CPU-only. (ITool itself has no capability-tag
// slot -- that field only exists on GP_RepoPackageTool manifest entries,
// which this same-repo tool isn't ingested through.)
//
// Input: pops a KpathArmatureProgramBox* (VT_VOID_PTR) off the stack and
// takes ownership of it.
// Output: pushes a KpathRasterResultBox* (VT_VOID_PTR) with the rasterized
// energy/temp canvases.
//
// Notes: machine/tool parameters are fixed defaults for now (mirroring
// tools/kpath_relgeo_scene_demo.cpp) -- exposing them as tool-configurable
// ports is future work, not required to make kpath reachable as a node.
//
// Calls kpath_raster through the gp_kpath_* C-ABI wrappers
// (kpath_raster_c_abi.h) rather than the raw nodus::tensors::kpath::
// functions directly: two independent canvas_tables_static copies in one
// process (this DLL's own, plus the host's) were observed to crash once
// either copy's kpath_raster code actually ran. Routing through canvas_tables
// (shared, see this target's CMakeLists.txt linkage) keeps a single live
// copy in the process, matching tests/tool_dll_test.cpp's existing pattern.
class KpathRasterKernelTool final : public ITool {
public:
    const char* id_cstr() const noexcept override { return "kpath_raster_kernel"; }
    const char* name_cstr() const noexcept override { return "KPath Raster (Kernel Transform)"; }
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

        auto* result = new KpathRasterResultBox();
        gp_kpath_rasterize_program_with_kernel_transformed(input->program, result->energy, result->temp,
                                                            machine, kernel_tool, xform);
        delete input;

        kpath_push_raster_result_box(frame, result);
    }
};

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new KpathRasterKernelTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) { delete t; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
