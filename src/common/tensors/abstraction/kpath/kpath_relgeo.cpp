#include "common/tensors/abstraction/kpath/kpath_relgeo.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>

namespace nodus::tensors::kpath {

namespace {

constexpr float kEps = 1e-6f;

static float dist(RelVec2 a, RelVec2 b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

static bool circle_circle_intersections(RelVec2 c0, float r0, RelVec2 c1, float r1, RelVec2& out_a, RelVec2& out_b) {
  const float dx = c1.x - c0.x;
  const float dy = c1.y - c0.y;
  const float d = std::sqrt(dx * dx + dy * dy);
  if (d < kEps) return false;
  if (d > r0 + r1) return false;
  if (d < std::fabs(r0 - r1)) return false;

  const float a = (r0 * r0 - r1 * r1 + d * d) / (2.0f * d);
  const float h2 = std::max(0.0f, r0 * r0 - a * a);
  const float h = std::sqrt(h2);

  const float xm = c0.x + a * dx / d;
  const float ym = c0.y + a * dy / d;

  const float rx = -dy * (h / d);
  const float ry = dx * (h / d);

  out_a = RelVec2{xm + rx, ym + ry};
  out_b = RelVec2{xm - rx, ym - ry};
  return true;
}

static RelVec2 pick(RelPick p, RelVec2 a, RelVec2 b) {
  switch (p) {
    case RelPick::HigherY:
      return (a.y >= b.y) ? a : b;
    case RelPick::LowerY:
      return (a.y <= b.y) ? a : b;
    case RelPick::HigherX:
      return (a.x >= b.x) ? a : b;
    case RelPick::LowerX:
      return (a.x <= b.x) ? a : b;
  }
  return a;
}

static OutlineSegment seg_move(RelVec2 p) {
  OutlineSegment s;
  s.op = OutlineOp::MoveTo;
  s.x1 = p.x; s.y1 = p.y;
  s.x2 = p.x; s.y2 = p.y;
  s.x3 = p.x; s.y3 = p.y;
  return s;
}

static OutlineSegment seg_line(RelVec2 a, RelVec2 b) {
  OutlineSegment s;
  s.op = OutlineOp::LineTo;
  s.x1 = a.x; s.y1 = a.y;
  s.x2 = 0.0f; s.y2 = 0.0f;
  s.x3 = b.x; s.y3 = b.y;
  return s;
}

static OutlineSegment seg_close(RelVec2 last) {
  OutlineSegment s;
  s.op = OutlineOp::Close;
  s.x1 = last.x; s.y1 = last.y;
  s.x2 = 0.0f; s.y2 = 0.0f;
  s.x3 = 0.0f; s.y3 = 0.0f;
  return s;
}

} // namespace

RelPointId RelProgram::add_point(RelPointExpr expr) {
  RelPointId id(static_cast<uint32_t>(points_.size() + 1));
  points_.push_back(RelPointDef{id, std::move(expr)});
  return id;
}

bool RelProgram::set_point_expr(RelPointId id, RelPointExpr expr) {
  if (!id) return false;
  const size_t idx = static_cast<size_t>(id.v - 1);
  if (idx >= points_.size()) return false;
  points_[idx].expr = std::move(expr);
  return true;
}

RelSegmentId RelProgram::add_segment(RelPointId a, RelPointId b) {
  RelSegmentId id(static_cast<uint32_t>(segments_.size() + 1));
  segments_.push_back(RelSegmentDef{id, a, b});
  return id;
}

RelContourId RelProgram::add_contour(std::vector<RelPointId> vertices, bool closed) {
  RelContourId id(static_cast<uint32_t>(contours_.size() + 1));
  contours_.push_back(RelContourDef{id, std::move(vertices), closed});
  return id;
}

bool RelProgram::evaluate(std::vector<RelVec2>& out_positions, std::string* out_error) const {
  out_positions.clear();
  out_positions.resize(points_.size());

  // id.v is 1-based index into points_.
  auto idx_of = [&](RelPointId id) -> std::optional<size_t> {
    if (!id) return std::nullopt;
    const size_t idx = static_cast<size_t>(id.v - 1);
    if (idx >= points_.size()) return std::nullopt;
    return idx;
  };

  enum class Mark : uint8_t { Unvisited, Visiting, Done };
  std::vector<Mark> marks(points_.size(), Mark::Unvisited);

  std::function<bool(RelPointId)> eval_rec;
  eval_rec = [&](RelPointId id) -> bool {
    auto idx_opt = idx_of(id);
    if (!idx_opt) {
      if (out_error) *out_error = "invalid point id";
      return false;
    }
    const size_t idx = *idx_opt;
    if (marks[idx] == Mark::Done) return true;
    if (marks[idx] == Mark::Visiting) {
      if (out_error) *out_error = "cycle detected in relational program";
      return false;
    }
    marks[idx] = Mark::Visiting;

    const RelPointDef& def = points_[idx];

    auto setp = [&](RelVec2 p) {
      out_positions[idx] = p;
      marks[idx] = Mark::Done;
      return true;
    };

    if (std::holds_alternative<RelPointFixed>(def.expr)) {
      const auto& e = std::get<RelPointFixed>(def.expr);
      return setp(RelVec2{e.x, e.y});
    }

    if (std::holds_alternative<RelPointLerp>(def.expr)) {
      const auto& e = std::get<RelPointLerp>(def.expr);
      if (!eval_rec(e.a) || !eval_rec(e.b)) return false;
      const auto ia = idx_of(e.a);
      const auto ib = idx_of(e.b);
      if (!ia || !ib) return false;
      const float u = std::clamp(e.u, 0.0f, 1.0f);
      RelVec2 a = out_positions[*ia];
      RelVec2 b = out_positions[*ib];
      return setp(RelVec2{a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u});
    }

    if (std::holds_alternative<RelPointOffset>(def.expr)) {
      const auto& e = std::get<RelPointOffset>(def.expr);
      if (!eval_rec(e.base)) return false;
      const auto ib = idx_of(e.base);
      if (!ib) return false;
      RelVec2 base = out_positions[*ib];
      return setp(RelVec2{base.x + e.dx, base.y + e.dy});
    }

    if (std::holds_alternative<RelPointCircleCircleIntersection>(def.expr)) {
      const auto& e = std::get<RelPointCircleCircleIntersection>(def.expr);
      if (!eval_rec(e.c0.center) || !eval_rec(e.c1.center)) return false;

      const auto ic0 = idx_of(e.c0.center);
      const auto ic1 = idx_of(e.c1.center);
      if (!ic0 || !ic1) return false;
      const RelVec2 c0 = out_positions[*ic0];
      const RelVec2 c1 = out_positions[*ic1];

      auto eval_radius = [&](const RelRadiusExpr& r, float& out_r) -> bool {
        if (std::holds_alternative<RelRadiusConstant>(r)) {
          out_r = std::get<RelRadiusConstant>(r).r;
          return out_r > 0.0f;
        }
        const auto& d = std::get<RelRadiusDistance>(r);
        if (!eval_rec(d.a) || !eval_rec(d.b)) return false;
        const auto ia = idx_of(d.a);
        const auto ib = idx_of(d.b);
        if (!ia || !ib) return false;
        out_r = dist(out_positions[*ia], out_positions[*ib]);
        return out_r > 0.0f;
      };

      float r0 = 0.0f;
      float r1 = 0.0f;
      if (!eval_radius(e.c0.radius, r0) || !eval_radius(e.c1.radius, r1)) {
        if (out_error) *out_error = "failed to evaluate circle radii";
        return false;
      }

      RelVec2 a{}, b{};
      if (!circle_circle_intersections(c0, r0, c1, r1, a, b)) {
        if (out_error) *out_error = "circles do not intersect";
        return false;
      }
      return setp(pick(e.pick, a, b));
    }

    if (out_error) *out_error = "unknown relational point expr";
    return false;
  };

  for (const auto& def : points_) {
    if (!eval_rec(def.id)) return false;
  }

  return true;
}

std::optional<RelVec2> RelProgram::eval_point(RelPointId id) const {
  std::vector<RelVec2> pos;
  if (!evaluate(pos, nullptr)) return std::nullopt;
  if (!id) return std::nullopt;
  const size_t idx = static_cast<size_t>(id.v - 1);
  if (idx >= pos.size()) return std::nullopt;
  return pos[idx];
}

bool compile_relglyph_outline(const RelGlyph& glyph,
                              GlyphOutline& out_outline,
                              float tx,
                              float ty,
                              float scale,
                              std::string* out_error) {
  std::vector<RelVec2> pos;
  if (!glyph.program.evaluate(pos, out_error)) return false;

  out_outline = GlyphOutline{};
  out_outline.glyph_id = glyph.glyph_id;

  auto getp = [&](RelPointId id, RelVec2& out) -> bool {
    if (!id) return false;
    const size_t idx = static_cast<size_t>(id.v - 1);
    if (idx >= pos.size()) return false;
    out = pos[idx];
    out.x = out.x * scale + tx;
    out.y = out.y * scale + ty;
    return true;
  };

  if (glyph.program.contours().empty()) {
    if (out_error) *out_error = "relglyph has no contours";
    return false;
  }

  for (const auto& contour : glyph.program.contours()) {
    if (contour.vertices.size() < 2) continue;
    RelVec2 p0{};
    if (!getp(contour.vertices.front(), p0)) return false;
    out_outline.segments.push_back(seg_move(p0));

    RelVec2 last = p0;
    for (size_t i = 1; i < contour.vertices.size(); ++i) {
      RelVec2 pi{};
      if (!getp(contour.vertices[i], pi)) return false;
      out_outline.segments.push_back(seg_line(last, pi));
      last = pi;
    }

    if (contour.closed) {
      out_outline.segments.push_back(seg_close(last));
    }
  }

  return !out_outline.segments.empty();
}

std::vector<GlyphOutline> layout_relglyphs_line(const std::vector<RelGlyph>& glyphs,
                                               float start_x,
                                               float start_y,
                                               float scale) {
  std::vector<GlyphOutline> out;
  float pen_x = start_x;
  float pen_y = start_y;

  for (const auto& g : glyphs) {
    GlyphOutline outline;
    std::string err;
    if (compile_relglyph_outline(g, outline, pen_x, pen_y, scale, &err)) {
      out.push_back(std::move(outline));
    }
    pen_x += g.advance_x * scale;
  }

  return out;
}

} // namespace nodus::tensors::kpath
