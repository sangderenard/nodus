#pragma once

#include "kpath_ids.h"
#include "kpath_shaper.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace nodus::tensors::kpath {

// Relational geometry: constructive geometry programs with stable identities.
//
// Goal: provide a HarfBuzz-like authoring + layout layer for synthetic geometry
// where endpoints/anchors have identities and derived points can be referenced
// by relationships (e.g. circle intersections, lerp along a segment).

struct RelPointTag {};
struct RelSegmentTag {};
struct RelContourTag {};
struct RelLineTag {};
struct RelRayTag {};
struct RelAngleTag {};

using RelPointId = Id<RelPointTag>;
using RelSegmentId = Id<RelSegmentTag>;
using RelContourId = Id<RelContourTag>;
using RelLineId = Id<RelLineTag>;
using RelRayId = Id<RelRayTag>;
using RelAngleId = Id<RelAngleTag>;

struct RelVec2 final {
  float x = 0.0f;
  float y = 0.0f;
};

enum class RelPick : uint8_t {
  HigherY,
  LowerY,
  HigherX,
  LowerX,
};

struct RelRadiusConstant final {
  float r = 0.0f;
};

struct RelRadiusDistance final {
  RelPointId a{};
  RelPointId b{};
};

using RelRadiusExpr = std::variant<RelRadiusConstant, RelRadiusDistance>;

struct RelCircle final {
  RelPointId center{};
  RelRadiusExpr radius;
};

struct RelPointFixed final {
  float x = 0.0f;
  float y = 0.0f;
};

struct RelPointLerp final {
  RelPointId a{};
  RelPointId b{};
  float u = 0.0f; // 0..1
};

struct RelPointOffset final {
  RelPointId base{};
  float dx = 0.0f;
  float dy = 0.0f;
};

struct RelPointCircleCircleIntersection final {
  RelCircle c0;
  RelCircle c1;
  RelPick pick = RelPick::HigherY;
};

struct RelParametricSegment final {
  enum class Kind : uint8_t {
    Line,
    Quadratic,
    Cubic,
    SinWave,
  };

  Kind kind = Kind::Line;
  RelPointId start{};
  RelPointId end{};
  RelPointId ctrl1{};
  RelPointId ctrl2{};
  float amplitude = 0.0f;
  float cycles = 1.0f;
  float phase = 0.0f;
  bool tangent_forward = true;
};

struct RelPointParametric final {
  RelParametricSegment segment;
  float u = 0.0f; // normalized 0..1 along the curve
};

// Intersection of two infinite lines.
struct RelPointLineLineIntersection final {
  RelLineId l0{};
  RelLineId l1{};
};

using RelPointExpr =
  std::variant<RelPointFixed,
         RelPointLerp,
         RelPointOffset,
         RelPointCircleCircleIntersection,
         RelPointParametric,
         RelPointLineLineIntersection>;

struct RelPointDef final {
  RelPointId id{};
  RelPointExpr expr;
};

struct RelSegmentDef final {
  RelSegmentId id{};
  RelPointId a{};
  RelPointId b{};
};

// Infinite line through points a and b.
struct RelLineDef final {
  RelLineId id{};
  RelPointId a{};
  RelPointId b{};
};

// Half-infinite ray starting at origin and passing through "through".
struct RelRayDef final {
  RelRayId id{};
  RelPointId origin{};
  RelPointId through{};
};

// Oriented angle A-V-B (vertex at V).
struct RelAngleDef final {
  RelAngleId id{};
  RelPointId a{};
  RelPointId v{};
  RelPointId b{};
};

enum class RelContourWinding : uint8_t {
  Unknown = 0,
  CCW = 1,
  CW = 2,
};

struct RelAssertPointOnLine final {
  RelPointId p{};
  RelLineId line{};
  float tol = 1e-3f;
};

struct RelAssertCoincident final {
  RelPointId a{};
  RelPointId b{};
  float tol = 1e-3f;
};

struct RelAssertParallelLines final {
  RelLineId a{};
  RelLineId b{};
  float tol = 1e-3f; // angular tolerance (radians)
};

struct RelAssertPerpendicularLines final {
  RelLineId a{};
  RelLineId b{};
  float tol = 1e-3f; // angular tolerance (radians)
};

using RelAssertion =
  std::variant<RelAssertPointOnLine, RelAssertCoincident, RelAssertParallelLines, RelAssertPerpendicularLines>;

struct RelContourDef final {
  RelContourId id{};
  // For now, a contour is an ordered vertex list (polyline). If closed, the
  // outline will emit a Close and connect back to the first vertex.
  std::vector<RelPointId> vertices;
  bool closed = true;
  // Winding is interpreted in the RelGeo coordinate system (x right, y up).
  // Convention: CCW is outer boundary, CW is inner (hole).
  RelContourWinding winding = RelContourWinding::Unknown;
};

class RelProgram final {
 public:
  RelPointId add_point(RelPointExpr expr);
  bool set_point_expr(RelPointId id, RelPointExpr expr);
  RelSegmentId add_segment(RelPointId a, RelPointId b);
  RelLineId add_line(RelPointId a, RelPointId b);
  RelRayId add_ray(RelPointId origin, RelPointId through);
  RelAngleId add_angle(RelPointId a, RelPointId v, RelPointId b);
  RelContourId add_contour(std::vector<RelPointId> vertices,
                           bool closed = true,
                           RelContourWinding winding = RelContourWinding::Unknown);

  void add_assertion(RelAssertion a);

  const std::vector<RelPointDef>& points() const { return points_; }
  const std::vector<RelSegmentDef>& segments() const { return segments_; }
  const std::vector<RelLineDef>& lines() const { return lines_; }
  const std::vector<RelRayDef>& rays() const { return rays_; }
  const std::vector<RelAngleDef>& angles() const { return angles_; }
  const std::vector<RelContourDef>& contours() const { return contours_; }
  const std::vector<RelAssertion>& assertions() const { return assertions_; }

  // Evaluate all points. Returns false on failure (cycles, invalid refs, no intersections).
  bool evaluate(std::vector<RelVec2>& out_positions, std::string* out_error = nullptr) const;

  // Convenience: evaluate a single point.
  std::optional<RelVec2> eval_point(RelPointId id) const;

  // Validates post-conditions (incidence/coincidence) by evaluating and checking assertions.
  bool validate(std::string* out_error = nullptr) const;

 private:
  std::vector<RelPointDef> points_;
  std::vector<RelSegmentDef> segments_;
  std::vector<RelLineDef> lines_;
  std::vector<RelRayDef> rays_;
  std::vector<RelAngleDef> angles_;
  std::vector<RelContourDef> contours_;
  std::vector<RelAssertion> assertions_;
};

struct RelGlyph final {
  uint32_t codepoint = 0;
  uint32_t glyph_id = 0;
  RelProgram program;
  float advance_x = 0.0f;
  float advance_y = 0.0f;
};

// Compiles a relational glyph into a concrete outline.
// The outline is translated by (tx,ty) after evaluation.
bool compile_relglyph_outline(const RelGlyph& glyph,
                              GlyphOutline& out_outline,
                              float tx = 0.0f,
                              float ty = 0.0f,
                              float scale = 1.0f,
                              std::string* out_error = nullptr);

// Simple line layout: advances by glyph.advance_x and emits outlines.
std::vector<GlyphOutline> layout_relglyphs_line(const std::vector<RelGlyph>& glyphs,
                                               float start_x,
                                               float start_y,
                                               float scale = 1.0f);

} // namespace nodus::tensors::kpath
