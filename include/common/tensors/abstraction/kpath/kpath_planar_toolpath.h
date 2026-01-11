#pragma once

#include "kpath_ids.h"

#include <cstdint>
#include <vector>

namespace nodus::tensors::kpath {

struct Vec2 final {
  float x = 0.0f;
  float y = 0.0f;
};

// Planner-stage representation: purely geometric 2D spans/loops plus semantic
// annotations (tool mode and winding). This is intentionally pre-device and
// pre-armature; it is the canonical boundary between planning and solving.
struct PlanarSpan final {
  ToolMode tool_mode{ToolMode::Travel};
  bool closed = false;
  RotDir winding{RotDir::Zero};
  std::vector<Vec2> points;
};

struct PlanarToolpath final {
  std::vector<PlanarSpan> spans;

  void clear() { spans.clear(); }
  bool empty() const { return spans.empty(); }
  size_t size() const { return spans.size(); }
};

// Computes signed area of a closed span. Positive => CCW.
float planar_signed_area(const PlanarSpan& span);

// Computes winding for a closed span. Non-closed spans return RotDir::Zero.
RotDir planar_winding(const PlanarSpan& span);

} // namespace nodus::tensors::kpath
