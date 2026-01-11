#pragma once

#include "kpath_kinematics.h"
#include "kpath_planar_toolpath.h"

#include <vector>

namespace nodus::tensors::kpath {

// Solver-stage entrypoint (post-planner): takes a planar toolpath definition
// (loops/spans + winding + tool_mode) and produces device/armature-specific
// actuation. This is the correct place for inverse kinematics, timing,
// axis masking, and tool power scheduling.
struct ToolpathSolveParams final {
  float nominal_z = 0.0f;
  float feed_rate = 1.0f;
};

// Placeholder output: sequence of armature configurations along the path.
// This is expected to expand into a richer multi-receiver actuation tape.
struct ToolpathSolveOutput final {
  std::vector<ArmatureConfiguration> configs;
};

// NOTE: currently a stub that returns false.
bool solve_planar_toolpath_to_armature(const ArmatureModel& model,
                                       const PlanarToolpath& path,
                                       const ToolpathSolveParams& params,
                                       ToolpathSolveOutput* out);

} // namespace nodus::tensors::kpath
