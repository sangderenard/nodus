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

using RelPointId = Id<RelPointTag>;
using RelSegmentId = Id<RelSegmentTag>;
using RelContourId = Id<RelContourTag>;

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

using RelPointExpr = std::variant<RelPointFixed, RelPointLerp, RelPointOffset, RelPointCircleCircleIntersection>;

struct RelPointDef final {
  RelPointId id{};
  RelPointExpr expr;
};

struct RelSegmentDef final {
  RelSegmentId id{};
  RelPointId a{};
  RelPointId b{};
};

struct RelContourDef final {
  RelContourId id{};
  // For now, a contour is an ordered vertex list (polyline). If closed, the
  // outline will emit a Close and connect back to the first vertex.
  std::vector<RelPointId> vertices;
  bool closed = true;
};

class RelProgram final {
 public:
  RelPointId add_point(RelPointExpr expr);
  bool set_point_expr(RelPointId id, RelPointExpr expr);
  RelSegmentId add_segment(RelPointId a, RelPointId b);
  RelContourId add_contour(std::vector<RelPointId> vertices, bool closed = true);

  const std::vector<RelPointDef>& points() const { return points_; }
  const std::vector<RelSegmentDef>& segments() const { return segments_; }
  const std::vector<RelContourDef>& contours() const { return contours_; }

  // Evaluate all points. Returns false on failure (cycles, invalid refs, no intersections).
  bool evaluate(std::vector<RelVec2>& out_positions, std::string* out_error = nullptr) const;

  // Convenience: evaluate a single point.
  std::optional<RelVec2> eval_point(RelPointId id) const;

 private:
  std::vector<RelPointDef> points_;
  std::vector<RelSegmentDef> segments_;
  std::vector<RelContourDef> contours_;
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
