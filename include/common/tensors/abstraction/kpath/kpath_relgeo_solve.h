#pragma once

#include "common/tensors/abstraction/graph_ir.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo.h"

#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

namespace nodus::tensors::kpath {

// Numeric embedding inputs for a RelGeo graph.
//
// The intended flow is:
//   pure IR (symbolic) -> graph edits -> constraint solve (with anchors) -> value map
//
// Anchors are expressed as concrete coordinates for selected point node ids.
struct RelGeoSolveInputs final {
  // Anchor points by node id (the u32 values returned by operators like pt()/free()).
  std::unordered_map<uint32_t, RelVec2> anchors; // point_node_id -> value

  // Convenience: anchor by IR symbol name (e.g. "origin").
  // These are resolved at parse/eval time and merged into `anchors`.
  std::unordered_map<std::string, RelVec2> anchors_by_name;
  float eps = 1e-4f;
  uint32_t max_passes = 64;
};

// A numeric value map for the graph.
// Today this returns only point positions (enough to compile contours).
struct RelGeoSolveOutput final {
  std::unordered_map<uint32_t, RelVec2> points; // point_node_id -> value
};

// Solve a numeric embedding for a pure RelGeo IR program.
//
// Notes:
// - This currently supports deterministic propagation of a small subset of node kinds
//   (pt/free/lerp/offset/ratio/lerp_ratio/llint) plus relation checks when resolvable.
// - Underconstrained graphs will return an error listing unsolved required points.
bool relgeo_solve_points_from_pure_ir(std::string_view src,
                                     const RelGeoSolveInputs& in,
                                     RelGeoSolveOutput& out,
                                     std::string* out_error = nullptr);

// Lower-level entry: solve from an already-evaluated edit log.
bool relgeo_solve_points_from_pure_edits(std::span<const nodus::tensors::GraphEdit> edits,
                                        const RelGeoSolveInputs& in,
                                        RelGeoSolveOutput& out,
                                        std::string* out_error = nullptr);

// Build a RelProgram consisting of fixed points + contours from a pure RelGeo IR program.
// This is the intended bridge for rendering (compile_relglyph_outline) after constraint solving.
bool relgeo_program_from_pure_ir_solved(std::string_view src,
                                       const RelGeoSolveInputs& in,
                                       RelProgram& out_program,
                                       std::string* out_error = nullptr);

} // namespace nodus::tensors::kpath
