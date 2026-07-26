#include "common/tensors/abstraction/kpath/kpath_raster.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo_ir.h"
#include "common/tensors/abstraction/kpath/kpath_image_export.h"
#include "common/tensors/abstraction/abstract_tensor.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nodus::tensors::kpath;

static std::string make_output_path_next_to_exe(const char* argv0, const char* filename) {
  try {
    std::filesystem::path exe_path(argv0);
    if (!exe_path.empty()) {
      return (exe_path.parent_path() / filename).string();
    }
  } catch (...) {
  }
  return std::string(filename);
}

static GlyphOutline compile_from_ir_or_throw(std::string_view src,
                                            uint32_t glyph_id,
                                            float tx,
                                            float ty,
                                            float scale) {
  RelProgram p;
  std::string err;
  if (!relgeo_program_from_ir(src, p, &err)) {
    throw std::runtime_error(err);
  }

  RelGlyph g;
  g.glyph_id = glyph_id;
  g.program = std::move(p);

  GlyphOutline out;
  if (!compile_relglyph_outline(g, out, tx, ty, scale, &err)) {
    throw std::runtime_error(err);
  }
  return out;
}

int main(int argc, char** argv) {
  try {
    // Scene: multiple parts defined by tiny constraint programs.
    // Coordinate system is arbitrary; we let kpath mapping fit to canvas.

    const float scene_scale = 1.0f;

    // Part 1: a "house" footprint + constrained roof apex from circle-circle intersection.
    // The apex is the intersection of equal-radius circles centered at the roof corners.
    const std::string house_ir = R"(
      a = pt(0, 0);
      b = pt(240, 0);
      c = pt(240, 140);
      d = pt(0, 140);
      contour(a, b, c, d);

      ra = pt(0, 140);
      rb = pt(240, 140);
      apex = ccint(ra, 170, rb, 170, "HigherY");
      contour(ra, rb, apex);

      door0 = lerp(a, b, 0.45);
      door1 = lerp(a, b, 0.55);
      door2 = offset(door1, 0, 70);
      door3 = offset(door0, 0, 70);
      contour(door0, door1, door2, door3);

      w0 = lerp(d, c, 0.18);
      w1 = offset(w0, 40, 0);
      w2 = offset(w1, 0, 40);
      w3 = offset(w0, 0, 40);
      contour(w0, w1, w2, w3);
    )";

    // Part 2: a "sun": constrained triangle rays around a center using lerps and offsets.
    const std::string sun_ir = R"(
      center = pt(0, 0);
      p0 = offset(center, 0, 60);
      p1 = offset(center, 52, -30);
      p2 = offset(center, -52, -30);
      contour(p0, p1, p2);

      core0 = offset(center, 0, 18);
      core1 = offset(center, 16, -9);
      core2 = offset(center, -16, -9);
      contour(core0, core1, core2);
    )";

    // Part 3: a "bridge": two pylons and a cable sag approximated by lerps.
    const std::string bridge_ir = R"(
      left = pt(0, 0);
      right = pt(360, 0);
      topL = offset(left, 0, 120);
      topR = offset(right, 0, 120);
      contour(left, topL);
      contour(right, topR);

      mid = lerp(topL, topR, 0.5);
      sag = offset(mid, 0, -55);
      contour(topL, sag, topR, "open");
    )";

    // Compile outlines at explicit placements.
    const GlyphOutline house = compile_from_ir_or_throw(house_ir, 1, 80.0f, 140.0f, scene_scale);
    const GlyphOutline sun = compile_from_ir_or_throw(sun_ir, 2, 520.0f, 360.0f, scene_scale);
    const GlyphOutline bridge = compile_from_ir_or_throw(bridge_ir, 3, 120.0f, 520.0f, scene_scale);

    ArmatureProgram program;
    append_glyph_outline_to_program(program, house, 0.0f, 0.0f, 0.0f, 24);
    append_glyph_outline_to_program(program, sun, 0.0f, 0.0f, 0.0f, 24);
    append_glyph_outline_to_program(program, bridge, 0.0f, 0.0f, 0.0f, 24);

    // Raster.
    TensorCanvas2D energy;
    TensorCanvas2D temp;

    MachineControlConfig machine;
    machine.step_px = 0.9f;
    machine.energy_per_px = 1.0f;
    machine.enable_thermal_guard = false;

    GaussianToolParams tool;

    const float render_scale = 1.0f;
    const float margin_px = 12.0f;
    ProgramRasterTransform xform = plan_program_raster_transform_refined(program, machine, render_scale, margin_px, tool);
    BeamToolParams kernel_tool{};
    kernel_tool.falloff = BeamFalloffKind::Gaussian;
    kernel_tool.sigma_px = tool.sigma_px;
    rasterize_program_with_kernel_transformed(program, energy, temp, machine, kernel_tool, xform);

    const std::string out_path = make_output_path_next_to_exe((argc > 0) ? argv[0] : "kpath_relgeo_scene_demo",
                                                              "kpath_relgeo_scene.png");
    nodus::tensors::AbstractTensor image = make_image_tensor_from_canvas(energy);
    if (!export_tensor_png(image, out_path, true)) {
      std::cerr << "Failed to write PNG: " << out_path << "\n";
      return 1;
    }

    std::cout << "[REL-GEO] wrote PNG: " << out_path << " (energy_max=" << energy.max_value() << ")\n";

    // Scatter/gather raster path (rasterize_program_scatter_with_spatial_kernel):
    // restored this session after sitting stubbed since f7e6e8b (2026-01-17).
    // Renders the same program through the fast scatter-add engine instead of
    // the kernel-transform path above, and exports its own PNG so the two are
    // directly comparable.
    TensorCanvas2D scatter_energy;
    TensorCanvas2D scatter_temp;
    ThermalSimConfig sim{};
    ScatterTimingBreakdown scatter_timing{};
    const bool scatter_ok = rasterize_program_scatter_with_spatial_kernel(
        program, scatter_energy, scatter_temp, machine, xform, kernel_tool,
        nullptr, sim, nullptr, nullptr, &scatter_timing);
    if (!scatter_ok) {
      std::cerr << "[REL-GEO] scatter raster failed\n";
      return 1;
    }
    const std::string scatter_out_path = make_output_path_next_to_exe(
        (argc > 0) ? argv[0] : "kpath_relgeo_scene_demo", "kpath_relgeo_scene_scatter.png");
    nodus::tensors::AbstractTensor scatter_image = make_image_tensor_from_canvas(scatter_energy);
    if (!export_tensor_png(scatter_image, scatter_out_path, true)) {
      std::cerr << "Failed to write PNG: " << scatter_out_path << "\n";
      return 1;
    }
    std::cout << "[REL-GEO] wrote scatter PNG: " << scatter_out_path
              << " (energy_max=" << scatter_energy.max_value()
              << " plan_ms=" << scatter_timing.plan_ms
              << " deposit_ms=" << scatter_timing.deposit_ms
              << " unique_sites=" << scatter_timing.unique_sites << ")\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[REL-GEO] error: " << e.what() << "\n";
    return 1;
  }
}
