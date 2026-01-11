#pragma once

#include "kpath_planar_toolpath.h"
#include "kpath_raster.h"
#include "kpath_shaper.h"

#include <cstdint>
#include <vector>

namespace nodus::tensors::kpath {

enum class FillRule : uint8_t {
  EvenOdd,
  NonZeroWinding,
};

enum class FillPattern : uint8_t {
  Hatch,
};

// Where the tool centerline should be placed relative to the intended region.
// Center: tool centerline covers the mathematical region boundary.
// Inside: tool footprint stays inside (shrinks region by tool radius).
// Outside: tool footprint stays outside (expands region by tool radius).
enum class KerfMode : uint8_t {
  Center,
  Inside,
  Outside,
};

struct ToolGeometry final {
  // Cutter/beam effective width in the same units as the outline (font units).
  // For rasterization, this is typically expressed in pixel space after mapping,
  // but keeping it here makes STEP IR integration straightforward.
  float tool_width = 1.0f;

  // Step-over distance between fill passes.
  // If <= 0, an automatic stepover will be derived from tool_width and overlap.
  float stepover = 0.0f;

  // Overlap ratio between adjacent passes (0..0.95). Only used when stepover <= 0.
  // Example: overlap=0.5 => stepover = 0.5 * tool_width.
  float overlap = 0.5f;

  // Fill angle in degrees (0 == horizontal hatch lines in glyph space).
  float angle_degrees = 0.0f;

  // Optional cross-hatch pass (second direction). If enabled, a second hatch pass
  // is emitted at (angle_degrees + cross_angle_degrees).
  bool crosshatch = false;
  float cross_angle_degrees = 90.0f;

  // Miter limit for polygon offsetting (ratio relative to |offset|). When the
  // corner miter would exceed limit * |offset|, we fall back to a bevel.
  float offset_miter_limit = 4.0f;
};

struct OffsetContourPoint final {
  float x = 0.0f;
  float y = 0.0f;
};

// Optional capture of generated offset contours so they can be replayed or
// refined (e.g., with a smaller tool) without re-running the geometric offset.
struct OffsetContourBuffer final {
  // Annotation: effective tool width used when the buffer was generated.
  float tool_width = 0.0f;

  // Annotation: outline-space units per calibrated tool unit (e.g., px -> outline).
  float calibration_scale = 1.0f;

  // Annotation: positive allowance indicating how much smaller a finishing
  // tool may be while still respecting this offset (used for corner clean-up).
  float finishing_allowance = 0.0f;

  std::vector<std::vector<OffsetContourPoint>> loops;
};

struct OffsetContourOptions final {
  // Miter limit multiplier; defaults to ToolGeometry::offset_miter_limit.
  float miter_limit = 4.0f;

  // Lead-in/out control: extend the path before first contact and after exit.
  float lead_in_length = 0.0f;
  float lead_out_length = 0.0f;

  // Optional sweep (degrees) applied to the lead-in direction for a gentler entry.
  float lead_sweep_degrees = 0.0f;

  // Optional capture target; if set, the generated offset loops are stored here.
  OffsetContourBuffer* capture = nullptr;

  // Capture metadata overrides (used when capture != nullptr).
  float capture_tool_width = 0.0f;
  float capture_calibration_scale = 1.0f;
  float capture_finishing_allowance = 0.0f;

  bool enable_lead_in = true;

  // Geometric tolerance in outline-space units used when flattening curves
  // for offset and hatch computations. If <= 0, legacy fixed-step flattening
  // is used for speed and determinism.
  float tolerance = 0.0f;
};

struct FillPlanConfig final {
  FillRule rule = FillRule::EvenOdd;
  FillPattern pattern = FillPattern::Hatch;
  KerfMode kerf = KerfMode::Center;
  ToolGeometry tool;

  // A small padding applied to the scan domain.
  float bounds_pad = 1.0f;

  // Geometric tolerance in outline-space units used when flattening curves
  // for hatch computations. If <= 0, legacy fixed-step flattening is used.
  float tolerance = 0.0f;

  // Z conventions for emitted toolpaths.
  float safe_z = -1.0f;
  float cut_z = 0.0f;
};

// Generates interior fill toolpaths for a single glyph outline.
// Output uses disengaged rapids between hatch segments.
bool plan_fill_for_glyph_outline(const GlyphOutline& outline,
                                ArmatureProgram& out_program,
                                const FillPlanConfig& cfg);

// Generate a single offset contour (tool centerline) for the given outline.
// `offset` is in outline-space units: positive -> expand (outside), negative -> shrink (inside).
// Emits a rapid to start, plunge, cut along the offset contour, then lift.
bool plan_offset_contour_for_outline(const GlyphOutline& outline,
                                    float offset,
                                    ArmatureProgram& out_program,
                                    float safe_z = -1.0f,
                                    float cut_z = 0.0f,
                                    const OffsetContourOptions& options = {});

// Iteratively refines a fill by planning successive passes with shrinking tool widths.
// `outlines` are the outlines that describe the workpiece, and `coverage_loops`
// receives the offset contours generated during the first pass so the caller can
// decide how much finishing allowance remains. Returns true when any pass emits points.
bool plan_iterative_fill(const std::vector<GlyphOutline>& outlines,
                         ArmatureProgram& out_program,
                         const FillPlanConfig& base_cfg,
                         const GaussianToolParams& tool,
                         OffsetContourBuffer* coverage_loops = nullptr,
                         float finishing_allowance = 0.0f,
                         size_t max_passes = 4);

// --- Planner-stage planar toolpaths ---
// These are the preferred APIs going forward: they emit planar spans/loops
// annotated with ToolMode and winding, leaving armature/tool actuation to the
// solver stage.

bool plan_planar_fill_for_glyph_outline(const GlyphOutline& outline,
                                       PlanarToolpath& out_path,
                                       const FillPlanConfig& cfg);

bool plan_planar_offset_contour_for_outline(const GlyphOutline& outline,
                                            float offset,
                                            PlanarToolpath& out_path,
                                            float safe_z = -1.0f,
                                            float cut_z = 0.0f,
                                            const OffsetContourOptions& options = {});

} // namespace nodus::tensors::kpath
