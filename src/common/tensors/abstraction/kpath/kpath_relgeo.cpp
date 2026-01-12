#include "common/tensors/abstraction/kpath/kpath_relgeo.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>

namespace nodus::tensors::kpath {

namespace {

constexpr float kEps = 1e-6f;
constexpr float kTwoPi = 6.283185307179586f;

static float dot(RelVec2 a, RelVec2 b) { return a.x * b.x + a.y * b.y; }

static float len(RelVec2 v) { return std::sqrt(v.x * v.x + v.y * v.y); }

static RelVec2 sub(RelVec2 a, RelVec2 b) { return RelVec2{a.x - b.x, a.y - b.y}; }

static float cross(RelVec2 a, RelVec2 b) { return a.x * b.y - a.y * b.x; }

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

static bool line_line_intersection(RelVec2 p0, RelVec2 p1, RelVec2 p2, RelVec2 p3, RelVec2& out_p) {
  // Solve p0 + t*(p1-p0) = p2 + u*(p3-p2)
  const RelVec2 r = sub(p1, p0);
  const RelVec2 s = sub(p3, p2);
  const float denom = cross(r, s);
  if (std::fabs(denom) < kEps) return false;
  const float t = cross(sub(p2, p0), s) / denom;
  out_p = RelVec2{p0.x + r.x * t, p0.y + r.y * t};
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

static RelVec2 lerp(const RelVec2& a, const RelVec2& b, float t) {
  return RelVec2{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
}

static RelVec2 evaluate_parametric_segment(const RelParametricSegment& seg,
                                           const RelVec2& start,
                                           const RelVec2& ctrl1,
                                           const RelVec2& ctrl2,
                                           const RelVec2& end,
                                           float u,
                                           RelVec2* out_tangent) {
  const float t = std::clamp(u, 0.0f, 1.0f);
  RelVec2 tangent{};
  RelVec2 pos{};

  switch (seg.kind) {
    case RelParametricSegment::Kind::Line: {
      tangent = RelVec2{end.x - start.x, end.y - start.y};
      pos = lerp(start, end, t);
      break;
    }
    case RelParametricSegment::Kind::Quadratic: {
      const float it = 1.0f - t;
      pos = RelVec2{
          it * it * start.x + 2.0f * it * t * ctrl1.x + t * t * end.x,
          it * it * start.y + 2.0f * it * t * ctrl1.y + t * t * end.y,
      };
      tangent = RelVec2{
          2.0f * it * (ctrl1.x - start.x) + 2.0f * t * (end.x - ctrl1.x),
          2.0f * it * (ctrl1.y - start.y) + 2.0f * t * (end.y - ctrl1.y),
      };
      break;
    }
    case RelParametricSegment::Kind::Cubic: {
      const float it = 1.0f - t;
      const float it2 = it * it;
      const float t2 = t * t;
      pos = RelVec2{
          it * it2 * start.x + 3.0f * it2 * t * ctrl1.x + 3.0f * it * t2 * ctrl2.x + t * t2 * end.x,
          it * it2 * start.y + 3.0f * it2 * t * ctrl1.y + 3.0f * it * t2 * ctrl2.y + t * t2 * end.y,
      };
      tangent = RelVec2{
          3.0f * it2 * (ctrl1.x - start.x) + 6.0f * it * t * (ctrl2.x - ctrl1.x) + 3.0f * t2 * (end.x - ctrl2.x),
          3.0f * it2 * (ctrl1.y - start.y) + 6.0f * it * t * (ctrl2.y - ctrl1.y) + 3.0f * t2 * (end.y - ctrl2.y),
      };
      break;
    }
    case RelParametricSegment::Kind::SinWave: {
      const RelVec2 base = lerp(start, end, t);
      const RelVec2 dir = RelVec2{end.x - start.x, end.y - start.y};
      tangent = dir;
      float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
      RelVec2 perp{0.0f, 0.0f};
      if (len > kEps) {
        perp = RelVec2{-dir.y / len, dir.x / len};
      }
      const float angle = kTwoPi * seg.cycles * t + seg.phase;
      const float sin_offset = seg.amplitude * std::sin(angle);
      const float cos_offset = seg.amplitude * kTwoPi * seg.cycles * std::cos(angle);
      pos = RelVec2{base.x + perp.x * sin_offset, base.y + perp.y * sin_offset};
      tangent.x += perp.x * cos_offset;
      tangent.y += perp.y * cos_offset;
      break;
    }
  }

  if (out_tangent) {
    if (seg.tangent_forward) *out_tangent = tangent;
    else *out_tangent = RelVec2{-tangent.x, -tangent.y};
  }

  return pos;
}

static float signed_area(const std::vector<RelVec2>& pts) {
  if (pts.size() < 3) return 0.0f;
  double a2 = 0.0;
  for (size_t i = 0; i < pts.size(); ++i) {
    const RelVec2 p = pts[i];
    const RelVec2 q = pts[(i + 1) % pts.size()];
    a2 += static_cast<double>(p.x) * static_cast<double>(q.y) - static_cast<double>(q.x) * static_cast<double>(p.y);
  }
  return static_cast<float>(0.5 * a2);
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

RelLineId RelProgram::add_line(RelPointId a, RelPointId b) {
  RelLineId id(static_cast<uint32_t>(lines_.size() + 1));
  lines_.push_back(RelLineDef{id, a, b});
  return id;
}

RelRayId RelProgram::add_ray(RelPointId origin, RelPointId through) {
  RelRayId id(static_cast<uint32_t>(rays_.size() + 1));
  rays_.push_back(RelRayDef{id, origin, through});
  return id;
}

RelAngleId RelProgram::add_angle(RelPointId a, RelPointId v, RelPointId b) {
  RelAngleId id(static_cast<uint32_t>(angles_.size() + 1));
  angles_.push_back(RelAngleDef{id, a, v, b});
  return id;
}

RelContourId RelProgram::add_contour(std::vector<RelPointId> vertices, bool closed, RelContourWinding winding) {
  RelContourId id(static_cast<uint32_t>(contours_.size() + 1));
  contours_.push_back(RelContourDef{id, std::move(vertices), closed, winding});
  return id;
}

void RelProgram::add_assertion(RelAssertion a) { assertions_.push_back(std::move(a)); }

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

  auto idx_of_line = [&](RelLineId id) -> std::optional<size_t> {
    if (!id) return std::nullopt;
    const size_t idx = static_cast<size_t>(id.v - 1);
    if (idx >= lines_.size()) return std::nullopt;
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

    if (std::holds_alternative<RelPointParametric>(def.expr)) {
      const auto& e = std::get<RelPointParametric>(def.expr);
      auto require_anchor = [&](RelPointId id, RelVec2& out, const char* msg) -> bool {
        if (!id) {
          if (out_error) *out_error = msg;
          return false;
        }
        if (!eval_rec(id)) return false;
        const auto ip = idx_of(id);
        if (!ip) return false;
        out = out_positions[*ip];
        return true;
      };

      RelVec2 start{}, end{}, ctrl1{}, ctrl2{};
      if (!require_anchor(e.segment.start, start, "relgeo.parametric missing start")) return false;
      if (!require_anchor(e.segment.end, end, "relgeo.parametric missing end")) return false;
      if (e.segment.kind == RelParametricSegment::Kind::Quadratic) {
        if (!require_anchor(e.segment.ctrl1, ctrl1, "relgeo.parametric missing ctrl1")) return false;
      } else if (e.segment.kind == RelParametricSegment::Kind::Cubic) {
        if (!require_anchor(e.segment.ctrl1, ctrl1, "relgeo.parametric missing ctrl1")) return false;
        if (!require_anchor(e.segment.ctrl2, ctrl2, "relgeo.parametric missing ctrl2")) return false;
      }

      RelVec2 pos = evaluate_parametric_segment(e.segment, start, ctrl1, ctrl2, end, e.u, nullptr);
      return setp(pos);
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

    if (std::holds_alternative<RelPointLineLineIntersection>(def.expr)) {
      const auto& e = std::get<RelPointLineLineIntersection>(def.expr);
      const auto il0 = idx_of_line(e.l0);
      const auto il1 = idx_of_line(e.l1);
      if (!il0 || !il1) {
        if (out_error) *out_error = "invalid line id";
        return false;
      }
      const RelLineDef& l0 = lines_[*il0];
      const RelLineDef& l1 = lines_[*il1];
      if (!eval_rec(l0.a) || !eval_rec(l0.b) || !eval_rec(l1.a) || !eval_rec(l1.b)) return false;
      const auto ip0 = idx_of(l0.a);
      const auto ip1 = idx_of(l0.b);
      const auto ip2 = idx_of(l1.a);
      const auto ip3 = idx_of(l1.b);
      if (!ip0 || !ip1 || !ip2 || !ip3) return false;
      RelVec2 p{};
      if (!line_line_intersection(out_positions[*ip0], out_positions[*ip1], out_positions[*ip2], out_positions[*ip3], p)) {
        if (out_error) *out_error = "lines do not intersect (parallel)";
        return false;
      }
      return setp(p);
    }

    if (out_error) *out_error = "unknown relational point expr";
    return false;
  };

  for (const auto& def : points_) {
    if (!eval_rec(def.id)) return false;
  }

  return true;
}

bool RelProgram::validate(std::string* out_error) const {
  if (assertions_.empty()) return true;
  std::vector<RelVec2> pos;
  if (!evaluate(pos, out_error)) return false;

  auto getp = [&](RelPointId id, RelVec2& out) -> bool {
    if (!id) return false;
    const size_t idx = static_cast<size_t>(id.v - 1);
    if (idx >= pos.size()) return false;
    out = pos[idx];
    return true;
  };
  auto getline = [&](RelLineId id, RelVec2& a, RelVec2& b) -> bool {
    if (!id) return false;
    const size_t idx = static_cast<size_t>(id.v - 1);
    if (idx >= lines_.size()) return false;
    const RelLineDef& l = lines_[idx];
    return getp(l.a, a) && getp(l.b, b);
  };

  for (const auto& a : assertions_) {
    if (std::holds_alternative<RelAssertCoincident>(a)) {
      const auto& c = std::get<RelAssertCoincident>(a);
      RelVec2 pa{}, pb{};
      if (!getp(c.a, pa) || !getp(c.b, pb)) {
        if (out_error) *out_error = "assert coincident: invalid point id";
        return false;
      }
      const float d = dist(pa, pb);
      if (d > c.tol) {
        if (out_error) *out_error = "assert coincident failed";
        return false;
      }
      continue;
    }
    if (std::holds_alternative<RelAssertPointOnLine>(a)) {
      const auto& inc = std::get<RelAssertPointOnLine>(a);
      RelVec2 p{}, a0{}, b0{};
      if (!getp(inc.p, p) || !getline(inc.line, a0, b0)) {
        if (out_error) *out_error = "assert point-on-line: invalid ids";
        return false;
      }
      const RelVec2 ab = sub(b0, a0);
      const RelVec2 ap = sub(p, a0);
      const float denom = len(ab);
      if (denom < kEps) {
        if (out_error) *out_error = "assert point-on-line: degenerate line";
        return false;
      }
      const float area2 = std::fabs(cross(ab, ap));
      const float dist_line = area2 / denom;
      if (dist_line > inc.tol) {
        if (out_error) *out_error = "assert point-on-line failed";
        return false;
      }
      continue;
    }
    if (std::holds_alternative<RelAssertParallelLines>(a)) {
      const auto& pl = std::get<RelAssertParallelLines>(a);
      RelVec2 a0{}, a1{}, b0{}, b1{};
      if (!getline(pl.a, a0, a1) || !getline(pl.b, b0, b1)) {
        if (out_error) *out_error = "assert parallel: invalid line id";
        return false;
      }
      const RelVec2 da = sub(a1, a0);
      const RelVec2 db = sub(b1, b0);
      if (len(da) < kEps || len(db) < kEps) {
        if (out_error) *out_error = "assert parallel: degenerate line";
        return false;
      }
      const float s = std::fabs(cross(da, db)) / (len(da) * len(db));
      const float ang = std::asin(std::clamp(s, 0.0f, 1.0f));
      if (ang > pl.tol) {
        if (out_error) *out_error = "assert parallel failed";
        return false;
      }
      continue;
    }
    if (std::holds_alternative<RelAssertPerpendicularLines>(a)) {
      const auto& pp = std::get<RelAssertPerpendicularLines>(a);
      RelVec2 a0{}, a1{}, b0{}, b1{};
      if (!getline(pp.a, a0, a1) || !getline(pp.b, b0, b1)) {
        if (out_error) *out_error = "assert perpendicular: invalid line id";
        return false;
      }
      const RelVec2 da = sub(a1, a0);
      const RelVec2 db = sub(b1, b0);
      if (len(da) < kEps || len(db) < kEps) {
        if (out_error) *out_error = "assert perpendicular: degenerate line";
        return false;
      }
      const float c = std::fabs(dot(da, db)) / (len(da) * len(db));
      const float ang = std::acos(std::clamp(c, 0.0f, 1.0f));
      const float dev = std::fabs(ang - 1.5707963267948966f);
      if (dev > pp.tol) {
        if (out_error) *out_error = "assert perpendicular failed";
        return false;
      }
      continue;
    }
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

    // Optionally normalize vertex order for closed contours with explicit winding.
    std::vector<RelPointId> verts = contour.vertices;
    if (contour.closed && contour.winding != RelContourWinding::Unknown && verts.size() >= 3) {
      std::vector<RelVec2> poly;
      poly.reserve(verts.size());
      for (RelPointId vid : verts) {
        if (!vid) return false;
        const size_t idx = static_cast<size_t>(vid.v - 1);
        if (idx >= pos.size()) return false;
        poly.push_back(pos[idx]);
      }
      const float a = signed_area(poly);
      const bool is_ccw = a > 0.0f;
      const bool want_ccw = contour.winding == RelContourWinding::CCW;
      if (is_ccw != want_ccw) {
        std::reverse(verts.begin(), verts.end());
      }
    }

    RelVec2 p0{};
    if (!getp(verts.front(), p0)) return false;
    out_outline.segments.push_back(seg_move(p0));

    RelVec2 last = p0;
    for (size_t i = 1; i < verts.size(); ++i) {
      RelVec2 pi{};
      if (!getp(verts[i], pi)) return false;
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
