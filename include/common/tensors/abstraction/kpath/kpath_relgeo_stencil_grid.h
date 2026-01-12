#pragma once

#include "common/tensors/abstraction/graph_ir.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo_stencil.h"

#include <cstdint>
#include <vector>

namespace nodus::tensors::kpath {

struct RelGeoStencilGridAxis final {
  uint32_t frame = 0;
  uint32_t axis = 0;
  uint32_t size = 0;
};

struct RelGeoStencilGridPlacement final {
  uint32_t frame = 0;
  uint32_t axis = 0;
  RelPointId point{};
  uint32_t index = 0;
};

struct RelGeoStencilGrid final {
  std::vector<RelGeoStencilGridAxis> axes;
  std::vector<RelGeoStencilGridPlacement> placements;
  std::vector<RelStencilIssue> issues;
};

struct RelGeoStencilGridOptions final {
  bool unique_per_point = true;
  bool include_unconstrained = true;
};

RelGeoStencilGrid relgeo_stencil_grid_from_stencil(const RelGeoStencil& stencil,
                                                   const RelGeoStencilGridOptions& options = {});

// Encode a stencil grid as a sparse-graph edit log (numeric attrs only).
// Node kinds are tagged via "stencil.kind" numeric codes:
// 1=frame, 2=axis, 3=point, 4=placement.
std::vector<GraphEdit> relgeo_stencil_grid_edits(const RelGeoStencil& stencil,
                                                 const RelGeoStencilGrid& grid);

} // namespace nodus::tensors::kpath
