#pragma once

#include "kpath_ids.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nodus::tensors::kpath {

// --- Conceptual layers ---
// Planner output should be axis/armature agnostic: it defines geometry + semantics.
// Solver stages progressively introduce parameterization, timing, and finally
// actuator-specific signals.

enum class ToolpathDomainKind : uint8_t {
  Planar2D,   // curves in a 2D plane (u,v)
  Surface2D,  // curves on a 2-manifold embedded in 3D (u,v mapped to xyz)
  Volume3D,   // volumetric coverage plans (layers, isosurfaces, space-filling)
};

enum class ToolpathPrimitiveKind : uint8_t {
  Polyline,
  // Future: BezierSpline, Arc, Clothoid, ImplicitIsoCurve, etc.
};

struct ToolpathVec2 final {
  float x = 0.0f;
  float y = 0.0f;
};

struct ToolpathVec3 final {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

// Tool-state semantics attached to geometry. This is still pre-device.
struct ToolpathSemantic final {
  ToolMode tool_mode{ToolMode::Travel};

  // Optional semantic fields (not all planners will set these).
  std::optional<float> feed_rate;
  std::optional<float> tolerance;
  std::optional<float> tool_power;

  // Topology semantics.
  bool closed = false;
  RotDir winding{RotDir::Zero};

  // User tags for downstream grouping (paths/contours/operations).
  uint32_t path_id = 0;
  uint32_t contour_id = 0;
};

struct ToolpathPolyline2D final {
  std::vector<ToolpathVec2> points;
};

// Geometry + semantics for one contiguous span.
struct ToolpathSpan final {
  ToolpathPrimitiveKind primitive{ToolpathPrimitiveKind::Polyline};
  ToolpathSemantic semantic;
  ToolpathPolyline2D polyline;
};

// Axis-agnostic toolpath definition in a chosen domain.
struct ToolpathDefinition final {
  ToolpathDomainKind domain{ToolpathDomainKind::Planar2D};
  std::vector<ToolpathSpan> spans;

  void clear() { spans.clear(); }
  bool empty() const { return spans.empty(); }
};

// --- Domain mapping ---
// A mapping turns domain coordinates into a 3D task space (workcell) coordinate.
// This is where planar -> surface and future volume plans can be unified.
struct ToolpathDomainMapping {
  virtual ~ToolpathDomainMapping() = default;

  // Map a (u,v) point in the domain to task-space xyz.
  virtual ToolpathVec3 map_uv_to_xyz(ToolpathVec2 uv) const = 0;

  // Optional: map a tangent in UV to a task-space direction.
  virtual ToolpathVec3 map_duv_to_dxyz(ToolpathVec2 duv) const {
    (void)duv;
    return ToolpathVec3{0, 0, 0};
  }
};

// --- Solve stages (high level) ---

enum class ToolpathSolveStage : uint8_t {
  Geometry,          // validate/normalize loops, winding, closure, kerf rules
  TaskParameterize,  // assign path parameter s and per-span semantics
  TimeParameterize,  // apply feed/accel/jerk constraints to produce t(s)
  Actuation,         // map task trajectory to actuator signals (IK + tool IO)
};

// Not yet a concrete tape type; this is the axis-agnostic target trajectory.
struct TaskSpaceSample final {
  ToolpathVec3 p;
  // Future: orientation, normal, curvature, etc.
  ToolpathSemantic semantic;
};

struct TaskSpaceTrajectory final {
  std::vector<TaskSpaceSample> samples;
};

} // namespace nodus::tensors::kpath
