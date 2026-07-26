#include "common/tensors/abstraction/kpath/kpath_raster_c_abi.h"

using namespace nodus::tensors;
using namespace nodus::tensors::kpath;

extern "C" {

void gp_kpath_rasterize_program_with_kernel_transformed(
    const ArmatureProgram& program,
    TensorCanvas2D& out_energy,
    TensorCanvas2D& out_temp,
    const MachineControlConfig& machine,
    const BeamToolParams& tool,
    const ProgramRasterTransform& xform) {
  rasterize_program_with_kernel_transformed(program, out_energy, out_temp, machine, tool, xform);
}

ProgramRasterTransform gp_kpath_plan_program_raster_transform_refined(
    const ArmatureProgram& reference_program,
    const MachineControlConfig& machine,
    float pixels_per_unit,
    float margin_px,
    const GaussianToolParams& tool) {
  return plan_program_raster_transform_refined(reference_program, machine, pixels_per_unit, margin_px, tool);
}

bool gp_kpath_rasterize_program_scatter_with_spatial_kernel(
    const ArmatureProgram& program,
    TensorCanvas2D& out_energy,
    TensorCanvas2D& out_temp,
    const MachineControlConfig& machine,
    const ProgramRasterTransform& xform,
    const BeamToolParams& tool,
    const AbstractTensor* diffusion_kernel,
    const ThermalSimConfig& sim,
    AbstractTensor* temp_state,
    TensorBackend* backend_override,
    ScatterTimingBreakdown* timing) {
  return rasterize_program_scatter_with_spatial_kernel(program, out_energy, out_temp, machine, xform, tool,
                                                        diffusion_kernel, sim, temp_state, backend_override, timing);
}

} // extern "C"
