#pragma once

#include "kpath_raster.h"

// Flat, unmangled (extern "C") wrappers around a few kpath_raster.h
// functions, so standalone plugin DLLs (e.g. plugins/kpath_raster_tool/)
// can call them across the canvas_tables.dll boundary. canvas_tables.dll's
// export surface (cmake/canvas_tables.def) is a curated, C-linkage-only ABI
// -- raw C++ namespaced symbols like nodus::tensors::kpath::* are never
// exported (their mangled names aren't in the .def, and this project
// deliberately keeps the DLL boundary flat-C, matching canvas_abi.h/
// table_abi.h/mem_backend.h elsewhere). These wrappers just forward to the
// real implementation; parameter types stay as their normal C++ references
// since caller and callee are both built by the same compiler/ABI, so no
// marshaling is needed -- extern "C" only affects name mangling here.
//
// A standalone plugin DLL that needs kpath_raster's compute should link
// against canvas_tables (shared), not canvas_tables_static: two independent
// canvas_tables_static copies coexisting in one process (the host binary's
// own copy plus a LoadLibrary'd plugin's own copy) has been observed to
// crash once either copy's kpath_raster code actually executes, even though
// the exact same lifecycle against a canvas_tables-linked plugin (the
// existing tests/tool_dll_test.cpp pattern) does not. Route through these
// wrappers + canvas_tables so there's only ever one live copy in the process.

extern "C" {

void gp_kpath_rasterize_program_with_kernel_transformed(
    const nodus::tensors::kpath::ArmatureProgram& program,
    nodus::tensors::kpath::TensorCanvas2D& out_energy,
    nodus::tensors::kpath::TensorCanvas2D& out_temp,
    const nodus::tensors::kpath::MachineControlConfig& machine,
    const nodus::tensors::kpath::BeamToolParams& tool,
    const nodus::tensors::kpath::ProgramRasterTransform& xform);

nodus::tensors::kpath::ProgramRasterTransform gp_kpath_plan_program_raster_transform_refined(
    const nodus::tensors::kpath::ArmatureProgram& reference_program,
    const nodus::tensors::kpath::MachineControlConfig& machine,
    float pixels_per_unit,
    float margin_px,
    const nodus::tensors::kpath::GaussianToolParams& tool);

bool gp_kpath_rasterize_program_scatter_with_spatial_kernel(
    const nodus::tensors::kpath::ArmatureProgram& program,
    nodus::tensors::kpath::TensorCanvas2D& out_energy,
    nodus::tensors::kpath::TensorCanvas2D& out_temp,
    const nodus::tensors::kpath::MachineControlConfig& machine,
    const nodus::tensors::kpath::ProgramRasterTransform& xform,
    const nodus::tensors::kpath::BeamToolParams& tool,
    const nodus::tensors::AbstractTensor* diffusion_kernel,
    const nodus::tensors::kpath::ThermalSimConfig& sim,
    nodus::tensors::AbstractTensor* temp_state,
    nodus::tensors::TensorBackend* backend_override,
    nodus::tensors::kpath::ScatterTimingBreakdown* timing);

} // extern "C"
