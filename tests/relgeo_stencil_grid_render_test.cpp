#include "common/tensors/abstraction/kpath/kpath_relgeo.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo_stencil.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo_stencil_grid.h"
#include "common/tensors/abstraction/kpath/kpath_image_export.h"
#include "common/tensors/abstraction/kpath/kpath_path_builder.h"
#include "common/tensors/abstraction/kpath/kpath_fill.h"
#include "common/tensors/abstraction/kpath/kpath_raster.h"
#include "common/tensors/abstraction/abstract_tensor.h"

#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace nodus::tensors::kpath;
using nodus::tensors::AbstractTensor;

static bool require_or_report(bool condition, const char* what) {
  if (condition) return true;
  std::cerr << "[REL-GEO-STENCIL-GRID] FAILED: " << what << "\n";
  return false;
}

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

static void append_rect(PathBuilder& pb, float x0, float y0, float x1, float y1) {
  pb.move_to(x0, y0)
    .line_to(x1, y0)
    .line_to(x1, y1)
    .line_to(x0, y1)
    .close();
}

int main(int argc, char** argv) {
  RelProgram prog;
  const RelPointId p0 = prog.add_point(RelPointFixed{0.0f, 0.0f});
  const RelPointId p1 = prog.add_point(RelPointOffset{p0, 10.0f, 0.0f});
  const RelPointId p2 = prog.add_point(RelPointOffset{p0, 0.0f, 10.0f});
  const RelPointId p3 = prog.add_point(RelPointFree{});
  const RelPointId p4 = prog.add_point(RelPointFree{});
  const RelPointId p5 = prog.add_point(RelPointFree{});

  prog.add_assertion(RelAssertCoincident{p3, p4});

  RelGeoStencilOptions stencil_opts;
  stencil_opts.dims = 2;
  stencil_opts.include_local_frames = false;

  const RelGeoStencil stencil = relgeo_build_stencil(prog, stencil_opts);

  RelGeoStencilGridOptions grid_opts;
  grid_opts.unique_per_point = true;
  grid_opts.include_unconstrained = true;

  const RelGeoStencilGrid grid = relgeo_stencil_grid_from_stencil(stencil, grid_opts);

  uint32_t cols = 0;
  uint32_t rows = 0;
  for (const auto& axis : grid.axes) {
    if (axis.frame != stencil.global_frame) continue;
    if (axis.axis == 0) cols = axis.size;
    if (axis.axis == 1) rows = axis.size;
  }

  if (!require_or_report(cols > 0 && rows > 0, "global grid axes missing")) return 1;

  std::unordered_map<uint32_t, uint32_t> x_index;
  std::unordered_map<uint32_t, uint32_t> y_index;
  for (const auto& place : grid.placements) {
    if (place.frame != stencil.global_frame) continue;
    if (place.axis == 0) x_index[place.point.v] = place.index;
    if (place.axis == 1) y_index[place.point.v] = place.index;
  }

  const float cell_size = 24.0f;
  const float margin = 10.0f;
  const float inset = 4.0f;
  const float width = cols * cell_size;
  const float height = rows * cell_size;

  PathBuilder grid_paths;
  for (uint32_t c = 0; c <= cols; ++c) {
    const float x = margin + static_cast<float>(c) * cell_size;
    grid_paths.move_to(x, margin).line_to(x, margin + height);
  }
  for (uint32_t r = 0; r <= rows; ++r) {
    const float y = margin + static_cast<float>(r) * cell_size;
    grid_paths.move_to(margin, y).line_to(margin + width, y);
  }

  PathBuilder fill_paths;
  std::unordered_set<uint64_t> used_cells;
  for (const auto& p : stencil.points) {
    const auto ix = x_index.find(p.v);
    const auto iy = y_index.find(p.v);
    if (ix == x_index.end() || iy == y_index.end()) continue;
    const uint64_t key = (static_cast<uint64_t>(ix->second) << 32) | static_cast<uint64_t>(iy->second);
    if (used_cells.find(key) != used_cells.end()) continue;
    used_cells.insert(key);

    const float x0 = margin + ix->second * cell_size + inset;
    const float y0 = margin + iy->second * cell_size + inset;
    const float x1 = margin + (ix->second + 1) * cell_size - inset;
    const float y1 = margin + (iy->second + 1) * cell_size - inset;
    append_rect(fill_paths, x0, y0, x1, y1);
  }

  const GlyphOutline grid_outline = grid_paths.finalize(1);
  const GlyphOutline fill_outline = fill_paths.finalize(2);

  ArmatureProgram grid_prog = glyph_outline_to_armature_program(grid_outline, 0.0f, 8);
  ArmatureProgram fill_prog;
  FillPlanConfig fill_cfg;
  fill_cfg.tool.tool_width = 3.5f;
  fill_cfg.tool.stepover = 2.5f;
  fill_cfg.tool.overlap = 0.4f;
  if (!require_or_report(!fill_outline.segments.empty(), "fill outline is empty")) return 1;
  if (!require_or_report(plan_fill_for_glyph_outline(fill_outline, fill_prog, fill_cfg), "plan_fill_for_glyph_outline failed")) {
    return 1;
  }

  ArmatureProgram combined = grid_prog;
  combined.points.insert(combined.points.end(), fill_prog.points.begin(), fill_prog.points.end());

  MachineControlConfig machine;
  machine.step_px = 0.6f;
  GaussianToolParams tool;

  const float pixels_per_unit = 3.0f;
  const float margin_px = 8.0f;
  const ProgramRasterTransform xform =
    plan_program_raster_transform_refined(combined, machine, pixels_per_unit, margin_px, tool);

  TensorCanvas2D grid_energy;
  TensorCanvas2D grid_temp;
  TensorCanvas2D fill_energy;
  TensorCanvas2D fill_temp;
  rasterize_program_gaussian_with_thermal_transformed(grid_prog, grid_energy, grid_temp, machine, tool, xform);
  rasterize_program_gaussian_with_thermal_transformed(fill_prog, fill_energy, fill_temp, machine, tool, xform);

  const std::vector<uint8_t> grid_px = grid_energy.to_u8_normalized();
  const std::vector<uint8_t> fill_px = fill_energy.to_u8_normalized();
  if (!require_or_report(!grid_px.empty() && grid_px.size() == fill_px.size(), "raster sizes mismatch")) return 1;

  TensorCanvas2D zero_canvas(grid_energy.width, grid_energy.height);
  zero_canvas.clear(0.0f);

  const std::string out_path = make_output_path_next_to_exe(argc > 0 ? argv[0] : "relgeo_stencil_grid_render_test",
                                                            "relgeo_stencil_grid.png");
  AbstractTensor rgb_image = make_image_tensor_from_canvases_rgb(grid_energy, fill_energy, zero_canvas);
  if (!require_or_report(export_tensor_png(rgb_image, out_path, true),
                         "export_canvas_rgb failed")) {
    return 1;
  }
  const std::string grid_path = make_output_path_next_to_exe(argc > 0 ? argv[0] : "relgeo_stencil_grid_render_test",
                                                             "relgeo_stencil_grid_gray.png");
  AbstractTensor grid_image = make_image_tensor_from_canvas(grid_energy);
  require_or_report(export_tensor_png(grid_image, grid_path, true), "export_tensor_png(grid) failed");
  const std::string fill_path = make_output_path_next_to_exe(argc > 0 ? argv[0] : "relgeo_stencil_grid_render_test",
                                                             "relgeo_stencil_fill_gray.png");
  AbstractTensor fill_image = make_image_tensor_from_canvas(fill_energy);
  require_or_report(export_tensor_png(fill_image, fill_path, true), "export_tensor_png(fill) failed");

  std::cout << "[REL-GEO-STENCIL-GRID] wrote PNG: " << out_path << "\n";
  return 0;
}
