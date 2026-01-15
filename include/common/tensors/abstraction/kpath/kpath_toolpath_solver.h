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
  float base_dt = 1.0f / 120.0f;
  float max_speed = 250.0f;
  float max_accel = 1500.0f;
  float max_force = 2000.0f;
  float kp = 12.0f;
  float kd = 4.0f;
  float hold_error = 1.0f;
  float lift_z = 1.0f;
};

struct ToolpathActuationFrame final {
  float t = 0.0f;
  float dt = 0.0f;
  float dt_scale = 1.0f;
  ToolMode tool_mode{ToolMode::Travel};
  bool tool_engaged = false;
  Vec3 target_pos{};
  Quat target_rot{};
  std::vector<float> q;
  std::vector<float> dq;
  std::vector<float> u;
  uint32_t axis_mask = 0;
};

// Solver output: step-wise actuation frames (second-order) that track a toolpath.
struct ToolpathSolveOutput final {
  std::vector<ToolpathActuationFrame> frames;
};

// NOTE: currently a stub that returns false.
bool solve_planar_toolpath_to_armature(const ArmatureModel& model,
                                       const PlanarToolpath& path,
                                       const ToolpathSolveParams& params,
                                       ToolpathSolveOutput* out);

} // namespace nodus::tensors::kpath
