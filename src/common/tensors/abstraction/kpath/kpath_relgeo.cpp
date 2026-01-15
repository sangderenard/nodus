#include "common/tensors/abstraction/kpath/kpath_relgeo.h"
#include "common/tensors/abstraction/in_memory_backend.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>

namespace nodus::tensors::kpath {

namespace {

constexpr float kEps = 1e-6f;
constexpr float kTwoPi = 6.283185307179586f;

struct P2 final {
  float x = 0.0f;
  float y = 0.0f;
};

static float dot(P2 a, P2 b) { return a.x * b.x + a.y * b.y; }

static float len(P2 v) { return std::sqrt(v.x * v.x + v.y * v.y); }

static bool normalize(P2 v, P2& out) {
  const float l = len(v);
  if (l < kEps) return false;
  out = P2{v.x / l, v.y / l};
  return true;
}

static P2 sub(P2 a, P2 b) { return P2{a.x - b.x, a.y - b.y}; }

static float cross(P2 a, P2 b) { return a.x * b.y - a.y * b.x; }

static float dist(P2 a, P2 b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

static float wrap_0_2pi(float a) {
  float r = std::fmod(a, kTwoPi);
  if (r < 0.0f) r += kTwoPi;
  return r;
}

static float wrap_ccw_delta(float delta) {
  float d = std::fmod(delta, kTwoPi);
  if (d < 0.0f) d += kTwoPi;
  return d;
}

static bool angle_is_between_ccw(float a0, float a1, float am) {
  // All angles are expected wrapped to [0,2pi).
  const float d01 = wrap_ccw_delta(a1 - a0);
  const float d0m = wrap_ccw_delta(am - a0);
  return d0m <= d01 + 1e-5f;
}

static bool circumcircle(P2 p0, P2 p1, P2 p2, P2& out_center, float& out_r) {
  // Compute circumcenter using determinant formula.
  // https://mathworld.wolfram.com/Circumcircle.html
  const float ax = p0.x;
  const float ay = p0.y;
  const float bx = p1.x;
  const float by = p1.y;
  const float cx = p2.x;
  const float cy = p2.y;

  const float d = 2.0f * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  if (std::fabs(d) < kEps) return false;

  const float a2 = ax * ax + ay * ay;
  const float b2 = bx * bx + by * by;
  const float c2 = cx * cx + cy * cy;

  const float ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
  const float uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
  out_center = P2{ux, uy};
  out_r = dist(out_center, p0);
  return out_r > kEps;
}

static bool circle_circle_intersections(P2 c0, float r0, P2 c1, float r1, P2& out_a, P2& out_b) {
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

  out_a = P2{xm + rx, ym + ry};
  out_b = P2{xm - rx, ym - ry};
  return true;
}

static bool line_line_intersection(P2 p0, P2 p1, P2 p2, P2 p3, P2& out_p) {
  // Solve p0 + t*(p1-p0) = p2 + u*(p3-p2)
  const P2 r = sub(p1, p0);
  const P2 s = sub(p3, p2);
  const float denom = cross(r, s);
  if (std::fabs(denom) < kEps) return false;
  const float t = cross(sub(p2, p0), s) / denom;
  out_p = P2{p0.x + r.x * t, p0.y + r.y * t};
  return true;
}

static P2 pick(RelPick p, P2 a, P2 b) {
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

static OutlineSegment seg_move(P2 p) {
  OutlineSegment s;
  s.op = OutlineOp::MoveTo;
  s.x1 = p.x; s.y1 = p.y;
  s.x2 = p.x; s.y2 = p.y;
  s.x3 = p.x; s.y3 = p.y;
  return s;
}

static OutlineSegment seg_line(P2 a, P2 b) {
  OutlineSegment s;
  s.op = OutlineOp::LineTo;
  s.x1 = a.x; s.y1 = a.y;
  s.x2 = 0.0f; s.y2 = 0.0f;
  s.x3 = b.x; s.y3 = b.y;
  return s;
}

static OutlineSegment seg_close(P2 last) {
  OutlineSegment s;
  s.op = OutlineOp::Close;
  s.x1 = last.x; s.y1 = last.y;
  s.x2 = 0.0f; s.y2 = 0.0f;
  s.x3 = 0.0f; s.y3 = 0.0f;
  return s;
}

static P2 lerp(const P2& a, const P2& b, float t) {
  return P2{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
}

static P2 evaluate_parametric_segment(const RelParametricSegment& seg,
                                      const P2& start,
                                      const P2& ctrl1,
                                      const P2& ctrl2,
                                      const P2& end,
                                      float u,
                                      P2* out_tangent) {
  const float t = std::clamp(u, 0.0f, 1.0f);
  P2 tangent{};
  P2 pos{};

  switch (seg.kind) {
    case RelParametricSegment::Kind::Line: {
      tangent = P2{end.x - start.x, end.y - start.y};
      pos = lerp(start, end, t);
      break;
    }
    case RelParametricSegment::Kind::Quadratic: {
      const float it = 1.0f - t;
      pos = P2{
          it * it * start.x + 2.0f * it * t * ctrl1.x + t * t * end.x,
          it * it * start.y + 2.0f * it * t * ctrl1.y + t * t * end.y,
      };
      tangent = P2{
          2.0f * it * (ctrl1.x - start.x) + 2.0f * t * (end.x - ctrl1.x),
          2.0f * it * (ctrl1.y - start.y) + 2.0f * t * (end.y - ctrl1.y),
      };
      break;
    }
    case RelParametricSegment::Kind::Cubic: {
      const float it = 1.0f - t;
      const float it2 = it * it;
      const float t2 = t * t;
      pos = P2{
          it * it2 * start.x + 3.0f * it2 * t * ctrl1.x + 3.0f * it * t2 * ctrl2.x + t * t2 * end.x,
          it * it2 * start.y + 3.0f * it2 * t * ctrl1.y + 3.0f * it * t2 * ctrl2.y + t * t2 * end.y,
      };
      tangent = P2{
          3.0f * it2 * (ctrl1.x - start.x) + 6.0f * it * t * (ctrl2.x - ctrl1.x) + 3.0f * t2 * (end.x - ctrl2.x),
          3.0f * it2 * (ctrl1.y - start.y) + 6.0f * it * t * (ctrl2.y - ctrl1.y) + 3.0f * t2 * (end.y - ctrl2.y),
      };
      break;
    }
    case RelParametricSegment::Kind::SinWave: {
      const P2 base = lerp(start, end, t);
      const P2 dir = P2{end.x - start.x, end.y - start.y};
      tangent = dir;
      float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
      P2 perp{0.0f, 0.0f};
      if (len > kEps) {
        perp = P2{-dir.y / len, dir.x / len};
      }
      const float angle = kTwoPi * seg.cycles * t + seg.phase;
      const float sin_offset = seg.amplitude * std::sin(angle);
      const float cos_offset = seg.amplitude * kTwoPi * seg.cycles * std::cos(angle);
      pos = P2{base.x + perp.x * sin_offset, base.y + perp.y * sin_offset};
      tangent.x += perp.x * cos_offset;
      tangent.y += perp.y * cos_offset;
      break;
    }
  }

  if (out_tangent) {
    if (seg.tangent_forward) *out_tangent = tangent;
    else *out_tangent = P2{-tangent.x, -tangent.y};
  }

  return pos;
}

static float signed_area(const float* data, size_t count, const std::vector<RelPointId>& verts) {
  if (!data || verts.size() < 3) return 0.0f;
  double a2 = 0.0;
  const size_t n = verts.size();
  for (size_t i = 0; i < n; ++i) {
    const size_t pi = static_cast<size_t>(verts[i].v ? verts[i].v - 1 : 0);
    const size_t qi = static_cast<size_t>(verts[(i + 1) % n].v ? verts[(i + 1) % n].v - 1 : 0);
    if (pi >= count || qi >= count) continue;
    const float px = data[pi * 2u + 0u];
    const float py = data[pi * 2u + 1u];
    const float qx = data[qi * 2u + 0u];
    const float qy = data[qi * 2u + 1u];
    a2 += static_cast<double>(px) * static_cast<double>(qy) - static_cast<double>(qx) * static_cast<double>(py);
  }
  return static_cast<float>(0.5 * a2);
}

static bool read_point(const float* data, size_t count, RelPointId id, P2& out) {
  if (!data || !id) return false;
  const size_t idx = static_cast<size_t>(id.v - 1);
  if (idx >= count) return false;
  out = P2{data[idx * 2u + 0u], data[idx * 2u + 1u]};
  return true;
}

static void write_point(float* data, size_t idx, P2 p) {
  data[idx * 2u + 0u] = p.x;
  data[idx * 2u + 1u] = p.y;
}

struct TensorMapGuard final {
  AbstractTensor* tensor = nullptr;
  InMemoryBackend* backend = nullptr;
  bool mapped = false;
  ~TensorMapGuard() {
    if (mapped && tensor && backend) backend->unmap(tensor->handle());
  }
};

static TensorMapGuard map_points_guard(AbstractTensor& tensor,
                                       float*& data,
                                       size_t& count,
                                       std::string* out_error) {
  TensorMapGuard guard;
  guard.tensor = &tensor;
  guard.backend = dynamic_cast<InMemoryBackend*>(tensor.backend());
  if (!guard.backend) {
    if (out_error) *out_error = "relgeo expects in-memory tensor backend for mapping";
    return guard;
  }
  const TensorDesc& desc = tensor.desc();
  if (desc.dtype != TensorDType::F32 || desc.layout != TensorLayout::Dense) {
    if (out_error) *out_error = "relgeo expects dense f32 tensor for positions";
    return guard;
  }
  if (desc.shape.dims.size() != 2 || desc.shape.dims[1] != 2u) {
    if (out_error) *out_error = "relgeo expects position tensor shape [point_count,2]";
    return guard;
  }
  count = desc.shape.dims[0];
  void* raw = nullptr;
  size_t bytes = 0;
  if (!guard.backend->map(tensor.handle(), &raw, &bytes)) {
    if (out_error) *out_error = "relgeo failed to map tensor storage";
    return guard;
  }
  const size_t needed = count * 2u * sizeof(float);
  if (bytes < needed) {
    guard.backend->unmap(tensor.handle());
    if (out_error) *out_error = "relgeo tensor storage too small";
    return guard;
  }
  data = static_cast<float*>(raw);
  guard.mapped = true;
  return guard;
}

static std::optional<AbstractTensor> make_point_tensor(P2 p, TensorBackend* backend, std::string* out_error) {
  TensorDesc desc;
  desc.dtype = TensorDType::F32;
  desc.layout = TensorLayout::Dense;
  desc.shape.dims = {1u, 2u};
  AbstractTensor t = AbstractTensor::create(desc, backend);
  if (!t.valid()) {
    if (out_error) *out_error = "relgeo failed to allocate point tensor";
    return std::nullopt;
  }
  auto* mem_backend = dynamic_cast<InMemoryBackend*>(t.backend());
  if (!mem_backend) {
    if (out_error) *out_error = "relgeo expects in-memory tensor backend for point mapping";
    return std::nullopt;
  }
  void* raw = nullptr;
  size_t bytes = 0;
  if (!mem_backend->map(t.handle(), &raw, &bytes)) {
    if (out_error) *out_error = "relgeo failed to map point tensor";
    return std::nullopt;
  }
  if (bytes < 2u * sizeof(float)) {
    mem_backend->unmap(t.handle());
    if (out_error) *out_error = "relgeo point tensor storage too small";
    return std::nullopt;
  }
  float* dst = static_cast<float*>(raw);
  dst[0] = p.x;
  dst[1] = p.y;
  mem_backend->unmap(t.handle());
  return t;
}

static bool read_point_tensor(const AbstractTensor& t, P2& out, std::string* out_error) {
  auto* mem_backend = dynamic_cast<InMemoryBackend*>(t.backend());
  if (!mem_backend) {
    if (out_error) *out_error = "relgeo expects in-memory tensor backend for point read";
    return false;
  }
  const TensorDesc& desc = t.desc();
  if (desc.dtype != TensorDType::F32 || desc.layout != TensorLayout::Dense ||
      desc.shape.dims.size() != 2 || desc.shape.dims[0] == 0 || desc.shape.dims[1] != 2u) {
    if (out_error) *out_error = "relgeo expects point tensor shape [1,2]";
    return false;
  }
  void* raw = nullptr;
  size_t bytes = 0;
  if (!mem_backend->map(t.handle(), &raw, &bytes)) {
    if (out_error) *out_error = "relgeo failed to map point tensor";
    return false;
  }
  if (bytes < 2u * sizeof(float)) {
    mem_backend->unmap(t.handle());
    if (out_error) *out_error = "relgeo point tensor storage too small";
    return false;
  }
  float* dst = static_cast<float*>(raw);
  out = P2{dst[0], dst[1]};
  mem_backend->unmap(t.handle());
  return true;
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

RelCircleId RelProgram::add_circle(RelCircle circle) {
  RelCircleId id(static_cast<uint32_t>(circles_.size() + 1));
  circles_.push_back(RelCircleDef{id, std::move(circle)});
  return id;
}

RelArcId RelProgram::add_arc(RelArcExpr expr) {
  RelArcId id(static_cast<uint32_t>(arcs_.size() + 1));
  arcs_.push_back(RelArcDef{id, std::move(expr)});
  return id;
}

RelBezierId RelProgram::add_bezier(RelPointId p0, RelPointId c0, RelPointId c1, RelPointId p1) {
  RelBezierId id(static_cast<uint32_t>(beziers_.size() + 1));
  beziers_.push_back(RelBezierDef{id, p0, c0, c1, p1});
  return id;
}

RelContourId RelProgram::add_contour(std::vector<RelPointId> vertices, bool closed, RelContourWinding winding) {
  RelContourId id(static_cast<uint32_t>(contours_.size() + 1));
  contours_.push_back(RelContourDef{id, std::move(vertices), closed, winding});
  return id;
}

void RelProgram::add_assertion(RelAssertion a, RelRuleScope scope) {
  assertions_.push_back(RelScopedAssertion{std::move(a), scope});
}

bool RelProgram::evaluate(RelTensorPoint& out_positions,
                          TensorBackend* backend_override,
                          std::string* out_error) const {
  TensorDesc desc;
  desc.dtype = TensorDType::F32;
  desc.layout = TensorLayout::Dense;
  desc.shape.dims = {static_cast<uint32_t>(points_.size()), 2u};

  TensorBackend* backend = backend_override ? backend_override : &in_memory_backend_singleton();
  AbstractTensor positions = AbstractTensor::create(desc, backend);
  if (!positions.valid()) {
    if (out_error) *out_error = "relgeo failed to allocate positions tensor";
    return false;
  }

  float* data = nullptr;
  size_t count = 0;
  auto guard = map_points_guard(positions, data, count, out_error);
  if (!guard.mapped) return false;
  std::fill_n(data, count * 2u, 0.0f);

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

  auto read_idx = [&](size_t idx, P2& out) -> bool {
    if (idx >= count) return false;
    out = P2{data[idx * 2u + 0u], data[idx * 2u + 1u]};
    return true;
  };

  std::function<bool(RelPointId)> eval_rec;
  eval_rec = [&](RelPointId id) -> bool {
    auto idx_opt = idx_of(id);
    if (!idx_opt) {
      if (out_error) {
        *out_error = "invalid point id " + std::to_string(id.v) + " (points=" + std::to_string(points_.size()) + ")";
      }
      return false;
    }
    const size_t idx = *idx_opt;
    if (marks[idx] == Mark::Done) return true;
    if (marks[idx] == Mark::Visiting) {
      if (out_error) *out_error = "cycle detected in relational program (at point id " + std::to_string(id.v) + ")";
      return false;
    }
    marks[idx] = Mark::Visiting;

    const RelPointDef& def = points_[idx];

    auto setp = [&](P2 p) {
      write_point(data, idx, p);
      marks[idx] = Mark::Done;
      return true;
    };

    if (std::holds_alternative<RelPointFixed>(def.expr)) {
      const auto& e = std::get<RelPointFixed>(def.expr);
      return setp(P2{e.x, e.y});
    }

    if (std::holds_alternative<RelPointFree>(def.expr)) {
      if (out_error) *out_error = "relgeo.free_point requires a solver or anchors";
      return false;
    }

    if (std::holds_alternative<RelPointLerp>(def.expr)) {
      const auto& e = std::get<RelPointLerp>(def.expr);
      if (!eval_rec(e.a) || !eval_rec(e.b)) return false;
      const auto ia = idx_of(e.a);
      const auto ib = idx_of(e.b);
      if (!ia || !ib) return false;
      const float u = std::clamp(e.u, 0.0f, 1.0f);
      P2 a{}, b{};
      if (!read_idx(*ia, a) || !read_idx(*ib, b)) return false;
      return setp(P2{a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u});
    }

    if (std::holds_alternative<RelPointOffset>(def.expr)) {
      const auto& e = std::get<RelPointOffset>(def.expr);
      if (!eval_rec(e.base)) return false;
      const auto ib = idx_of(e.base);
      if (!ib) return false;
      P2 base{};
      if (!read_idx(*ib, base)) return false;
      return setp(P2{base.x + e.dx, base.y + e.dy});
    }

    if (std::holds_alternative<RelPointParametric>(def.expr)) {
      const auto& e = std::get<RelPointParametric>(def.expr);
      auto require_anchor = [&](RelPointId pid, P2& out, const char* msg) -> bool {
        if (!pid) {
          if (out_error) *out_error = msg;
          return false;
        }
        if (!eval_rec(pid)) return false;
        const auto ip = idx_of(pid);
        if (!ip) return false;
        return read_idx(*ip, out);
      };

      P2 start{}, end{}, ctrl1{}, ctrl2{};
      if (!require_anchor(e.segment.start, start, "relgeo.parametric missing start")) return false;
      if (!require_anchor(e.segment.end, end, "relgeo.parametric missing end")) return false;
      if (e.segment.kind == RelParametricSegment::Kind::Quadratic) {
        if (!require_anchor(e.segment.ctrl1, ctrl1, "relgeo.parametric missing ctrl1")) return false;
      } else if (e.segment.kind == RelParametricSegment::Kind::Cubic) {
        if (!require_anchor(e.segment.ctrl1, ctrl1, "relgeo.parametric missing ctrl1")) return false;
        if (!require_anchor(e.segment.ctrl2, ctrl2, "relgeo.parametric missing ctrl2")) return false;
      }

      P2 pos = evaluate_parametric_segment(e.segment, start, ctrl1, ctrl2, end, e.u, nullptr);
      return setp(pos);
    }

    if (std::holds_alternative<RelPointCircleCircleIntersection>(def.expr)) {
      const auto& e = std::get<RelPointCircleCircleIntersection>(def.expr);
      if (!eval_rec(e.c0.center) || !eval_rec(e.c1.center)) return false;

      const auto ic0 = idx_of(e.c0.center);
      const auto ic1 = idx_of(e.c1.center);
      if (!ic0 || !ic1) return false;
      P2 c0{}, c1{};
      if (!read_idx(*ic0, c0) || !read_idx(*ic1, c1)) return false;

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
        P2 pa{}, pb{};
        if (!read_idx(*ia, pa) || !read_idx(*ib, pb)) return false;
        out_r = dist(pa, pb);
        return out_r > 0.0f;
      };

      float r0 = 0.0f;
      float r1 = 0.0f;
      if (!eval_radius(e.c0.radius, r0) || !eval_radius(e.c1.radius, r1)) {
        if (out_error) *out_error = "failed to evaluate circle radii";
        return false;
      }

      P2 a{}, b{};
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
        if (out_error) {
          *out_error = "invalid line id in line-line intersection (l0=" + std::to_string(e.l0.v) + ", l1=" +
                       std::to_string(e.l1.v) + ")";
        }
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
      P2 p0{}, p1{}, p2{}, p3{};
      if (!read_idx(*ip0, p0) || !read_idx(*ip1, p1) || !read_idx(*ip2, p2) || !read_idx(*ip3, p3)) return false;
      P2 p{};
      if (!line_line_intersection(p0, p1, p2, p3, p)) {
        if (out_error) *out_error = "lines do not intersect (parallel)";
        return false;
      }
      return setp(p);
    }

    if (out_error) *out_error = "unknown relational point expr";
    return false;
  };

  bool ok = true;
  for (const auto& def : points_) {
    if (!eval_rec(def.id)) {
      ok = false;
      break;
    }
  }

  if (!ok) return false;
  out_positions = std::move(positions);
  return true;
}

bool RelProgram::validate(std::string* out_error) const {
  if (assertions_.empty()) return true;
  RelTensorPoint pos;
  if (!evaluate(pos, nullptr, out_error)) return false;

  float* data = nullptr;
  size_t count = 0;
  auto guard = map_points_guard(pos, data, count, out_error);
  if (!guard.mapped) return false;

  const RelRuleContext ctx{};

  auto getp = [&](RelPointId id, P2& out) -> bool {
    return read_point(data, count, id, out);
  };
  auto getline = [&](RelLineId id, P2& a, P2& b) -> bool {
    if (!id) return false;
    const size_t idx = static_cast<size_t>(id.v - 1);
    if (idx >= lines_.size()) return false;
    const RelLineDef& l = lines_[idx];
    return getp(l.a, a) && getp(l.b, b);
  };
  auto line_distance = [&](P2 a, P2 b, P2 p, float& out_dist) -> bool {
    const P2 ab = sub(b, a);
    const float denom = len(ab);
    if (denom < kEps) return false;
    const float area2 = std::fabs(cross(ab, sub(p, a)));
    out_dist = area2 / denom;
    return true;
  };
  auto line_angle = [&](P2 a0, P2 a1, P2 b0, P2 b1, float& out_angle) -> bool {
    const P2 da = sub(a1, a0);
    const P2 db = sub(b1, b0);
    const float la = len(da);
    const float lb = len(db);
    if (la < kEps || lb < kEps) return false;
    const float c = std::fabs(dot(da, db)) / (la * lb);
    out_angle = std::acos(std::clamp(c, 0.0f, 1.0f));
    return true;
  };

  auto has_circle = [&](RelCircleId id) -> bool {
    if (!id) return false;
    const size_t idx = static_cast<size_t>(id.v - 1);
    return idx < circles_.size();
  };
  auto has_arc = [&](RelArcId id) -> bool {
    if (!id) return false;
    const size_t idx = static_cast<size_t>(id.v - 1);
    return idx < arcs_.size();
  };

  for (const auto& scoped : assertions_) {
    if (!relgeo_rule_applies(scoped.scope, ctx)) continue;
    const auto& a = scoped.assertion;
    if (std::holds_alternative<RelAssertCoincident>(a)) {
      const auto& c = std::get<RelAssertCoincident>(a);
      P2 pa{}, pb{};
      if (!getp(c.a, pa) || !getp(c.b, pb)) {
        if (out_error) {
          *out_error = "assert coincident: invalid point id (a=" + std::to_string(c.a.v) + ", b=" + std::to_string(c.b.v) + ")";
        }
        return false;
      }
      const float d = dist(pa, pb);
      if (d > c.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert coincident failed: a=" << c.a.v << ", b=" << c.b.v << ", d=" << d << " > tol=" << c.tol;
          *out_error = oss.str();
        }
        return false;
      }
      continue;
    }
    if (std::holds_alternative<RelAssertPointOnLine>(a)) {
      const auto& inc = std::get<RelAssertPointOnLine>(a);
      P2 p{}, a0{}, b0{};
      if (!getp(inc.p, p) || !getline(inc.line, a0, b0)) {
        if (out_error) {
          *out_error = "assert point-on-line: invalid ids (p=" + std::to_string(inc.p.v) + ", line=" + std::to_string(inc.line.v) + ")";
        }
        return false;
      }
      const P2 ab = sub(b0, a0);
      const P2 ap = sub(p, a0);
      const float denom = len(ab);
      if (denom < kEps) {
        if (out_error) *out_error = "assert point-on-line: degenerate line";
        return false;
      }
      const float area2 = std::fabs(cross(ab, ap));
      const float dist_line = area2 / denom;
      if (dist_line > inc.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert point-on-line failed: p=" << inc.p.v << ", line=" << inc.line.v << ", dist=" << dist_line
              << " > tol=" << inc.tol;
          *out_error = oss.str();
        }
        return false;
      }
      continue;
    }
    if (std::holds_alternative<RelAssertParallelLines>(a)) {
      const auto& pl = std::get<RelAssertParallelLines>(a);
      P2 a0{}, a1{}, b0{}, b1{};
      if (!getline(pl.a, a0, a1) || !getline(pl.b, b0, b1)) {
        if (out_error) {
          *out_error = "assert parallel: invalid line id (a=" + std::to_string(pl.a.v) + ", b=" + std::to_string(pl.b.v) + ")";
        }
        return false;
      }
      const P2 da = sub(a1, a0);
      const P2 db = sub(b1, b0);
      if (len(da) < kEps || len(db) < kEps) {
        if (out_error) *out_error = "assert parallel: degenerate line";
        return false;
      }
      const float s = std::fabs(cross(da, db)) / (len(da) * len(db));
      const float ang = std::asin(std::clamp(s, 0.0f, 1.0f));
      if (ang > pl.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert parallel failed: a=" << pl.a.v << ", b=" << pl.b.v << ", angle=" << ang << " > tol=" << pl.tol;
          *out_error = oss.str();
        }
        return false;
      }
      continue;
    }
    if (std::holds_alternative<RelAssertPerpendicularLines>(a)) {
      const auto& pp = std::get<RelAssertPerpendicularLines>(a);
      P2 a0{}, a1{}, b0{}, b1{};
      if (!getline(pp.a, a0, a1) || !getline(pp.b, b0, b1)) {
        if (out_error) {
          *out_error = "assert perpendicular: invalid line id (a=" + std::to_string(pp.a.v) + ", b=" + std::to_string(pp.b.v) + ")";
        }
        return false;
      }
      const P2 da = sub(a1, a0);
      const P2 db = sub(b1, b0);
      if (len(da) < kEps || len(db) < kEps) {
        if (out_error) *out_error = "assert perpendicular: degenerate line";
        return false;
      }
      const float c = std::fabs(dot(da, db)) / (len(da) * len(db));
      const float ang = std::acos(std::clamp(c, 0.0f, 1.0f));
      const float dev = std::fabs(ang - 1.5707963267948966f);
      if (dev > pp.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert perpendicular failed: a=" << pp.a.v << ", b=" << pp.b.v << ", dev=" << dev << " > tol=" << pp.tol;
          *out_error = oss.str();
        }
        return false;
      }
      continue;
    }
    if (std::holds_alternative<RelAssertPerpAt>(a)) {
      const auto& pp = std::get<RelAssertPerpAt>(a);
      P2 v{}, a0{}, b0{};
      if (!getp(pp.v, v) || !getp(pp.a, a0) || !getp(pp.b, b0)) {
        if (out_error) {
          *out_error = "assert perp-at: invalid point id (v=" + std::to_string(pp.v.v) + ", a=" +
                       std::to_string(pp.a.v) + ", b=" + std::to_string(pp.b.v) + ")";
        }
        return false;
      }
      const P2 va = sub(a0, v);
      const P2 vb = sub(b0, v);
      if (len(va) < kEps || len(vb) < kEps) {
        if (out_error) *out_error = "assert perp-at: degenerate segment";
        return false;
      }
      const float c = std::fabs(dot(va, vb)) / (len(va) * len(vb));
      const float ang = std::acos(std::clamp(c, 0.0f, 1.0f));
      const float dev = std::fabs(ang - 1.5707963267948966f);
      if (dev > pp.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert perp-at failed: v=" << pp.v.v << ", a=" << pp.a.v << ", b=" << pp.b.v
              << ", dev=" << dev << " > tol=" << pp.tol;
          *out_error = oss.str();
        }
        return false;
      }
      continue;
    }

    // Circle/arc constraints are stored symbolically for now.
    // Validate ids are well-formed, but do not attempt numeric satisfaction here.
    if (std::holds_alternative<RelAssertPointOnCircle>(a)) {
      const auto& pc = std::get<RelAssertPointOnCircle>(a);
      if (!pc.p || !has_circle(pc.circle)) {
        if (out_error) {
          *out_error = "assert point-on-circle: invalid ids (p=" + std::to_string(pc.p.v) + ", circle=" +
                       std::to_string(pc.circle.v) + ")";
        }
        return false;
      }
      continue;
    }
    if (std::holds_alternative<RelAssertTangentLineCircle>(a)) {
      const auto& tc = std::get<RelAssertTangentLineCircle>(a);
      if (!tc.line || !has_circle(tc.circle)) {
        if (out_error) {
          *out_error = "assert tangent: invalid ids (line=" + std::to_string(tc.line.v) + ", circle=" +
                       std::to_string(tc.circle.v) + ")";
        }
        return false;
      }
      continue;
    }
    if (std::holds_alternative<RelAssertFixedRadius>(a)) {
      const auto& fr = std::get<RelAssertFixedRadius>(a);
      if (!has_circle(fr.circle) || fr.r <= 0.0f) {
        if (out_error) {
          *out_error = "assert fixed-radius: invalid (circle=" + std::to_string(fr.circle.v) + ", r=" +
                       std::to_string(fr.r) + ")";
        }
        return false;
      }
      continue;
    }
    if (std::holds_alternative<RelAssertArcAngle>(a)) {
      const auto& aa = std::get<RelAssertArcAngle>(a);
      if (!has_arc(aa.arc) || aa.angle <= 0.0f) {
        if (out_error) {
          *out_error = "assert arc-angle: invalid (arc=" + std::to_string(aa.arc.v) + ", angle=" +
                       std::to_string(aa.angle) + ")";
        }
        return false;
      }
      continue;
    }

    if (std::holds_alternative<RelAssertCollinear>(a)) {
      const auto& c = std::get<RelAssertCollinear>(a);
      P2 pa{}, pb{}, pc{};
      if (!getp(c.a, pa) || !getp(c.b, pb) || !getp(c.c, pc)) {
        if (out_error) {
          *out_error = "assert collinear: invalid point id (a=" + std::to_string(c.a.v) + ", b=" +
                       std::to_string(c.b.v) + ", c=" + std::to_string(c.c.v) + ")";
        }
        return false;
      }
      float d = 0.0f;
      if (!line_distance(pa, pb, pc, d)) {
        if (out_error) *out_error = "assert collinear: degenerate line";
        return false;
      }
      if (d > c.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert collinear failed: a=" << c.a.v << ", b=" << c.b.v << ", c=" << c.c.v
              << ", dist=" << d << " > tol=" << c.tol;
          *out_error = oss.str();
        }
        return false;
      }
      continue;
    }

    if (std::holds_alternative<RelAssertEqualDistance>(a)) {
      const auto& ed = std::get<RelAssertEqualDistance>(a);
      P2 a0{}, b0{}, c0{}, d0{};
      if (!getp(ed.a, a0) || !getp(ed.b, b0) || !getp(ed.c, c0) || !getp(ed.d, d0)) {
        if (out_error) {
          *out_error = "assert equal-dist: invalid point id (a=" + std::to_string(ed.a.v) + ", b=" +
                       std::to_string(ed.b.v) + ", c=" + std::to_string(ed.c.v) + ", d=" + std::to_string(ed.d.v) + ")";
        }
        return false;
      }
      const float d0ab = dist(a0, b0);
      const float d0cd = dist(c0, d0);
      const float diff = std::fabs(d0ab - d0cd);
      if (diff > ed.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert equal-dist failed: d_ab=" << d0ab << ", d_cd=" << d0cd << ", diff=" << diff
              << " > tol=" << ed.tol;
          *out_error = oss.str();
        }
        return false;
      }
      continue;
    }

    if (std::holds_alternative<RelAssertMidpoint>(a)) {
      const auto& mp = std::get<RelAssertMidpoint>(a);
      P2 m{}, a0{}, b0{};
      if (!getp(mp.m, m) || !getp(mp.a, a0) || !getp(mp.b, b0)) {
        if (out_error) {
          *out_error = "assert midpoint: invalid point id (m=" + std::to_string(mp.m.v) + ", a=" +
                       std::to_string(mp.a.v) + ", b=" + std::to_string(mp.b.v) + ")";
        }
        return false;
      }
      const P2 mid{(a0.x + b0.x) * 0.5f, (a0.y + b0.y) * 0.5f};
      const float dmid = dist(m, mid);
      if (dmid > mp.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert midpoint failed: m=" << mp.m.v << ", d_mid=" << dmid << " > tol=" << mp.tol;
          *out_error = oss.str();
        }
        return false;
      }
      continue;
    }

    if (std::holds_alternative<RelAssertPointOnSegment>(a)) {
      const auto& ps = std::get<RelAssertPointOnSegment>(a);
      P2 p{}, a0{}, b0{};
      if (!getp(ps.p, p) || !getp(ps.a, a0) || !getp(ps.b, b0)) {
        if (out_error) {
          *out_error = "assert on-segment: invalid point id (p=" + std::to_string(ps.p.v) + ", a=" +
                       std::to_string(ps.a.v) + ", b=" + std::to_string(ps.b.v) + ")";
        }
        return false;
      }
      float dline = 0.0f;
      if (!line_distance(a0, b0, p, dline)) {
        if (out_error) *out_error = "assert on-segment: degenerate segment";
        return false;
      }
      if (dline > ps.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert on-segment failed: p=" << ps.p.v << ", dist=" << dline << " > tol=" << ps.tol;
          *out_error = oss.str();
        }
        return false;
      }
      const P2 ab = sub(b0, a0);
      const float ab2 = dot(ab, ab);
      if (ab2 < kEps) {
        if (out_error) *out_error = "assert on-segment: degenerate segment";
        return false;
      }
      const float t = dot(sub(p, a0), ab) / ab2;
      if (t < -1e-4f || t > 1.0f + 1e-4f) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert on-segment failed: p=" << ps.p.v << ", t=" << t << " outside [0,1]";
          *out_error = oss.str();
        }
        return false;
      }
      continue;
    }

    if (std::holds_alternative<RelAssertPointOnSegmentRatio>(a)) {
      const auto& ps = std::get<RelAssertPointOnSegmentRatio>(a);
      P2 p{}, a0{}, b0{};
      if (!getp(ps.p, p) || !getp(ps.a, a0) || !getp(ps.b, b0)) {
        if (out_error) {
          *out_error = "assert on-segment-ratio: invalid point id (p=" + std::to_string(ps.p.v) + ", a=" +
                       std::to_string(ps.a.v) + ", b=" + std::to_string(ps.b.v) + ")";
        }
        return false;
      }
      float dline = 0.0f;
      if (!line_distance(a0, b0, p, dline)) {
        if (out_error) *out_error = "assert on-segment-ratio: degenerate segment";
        return false;
      }
      if (dline > ps.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert on-segment-ratio failed: p=" << ps.p.v << ", dist=" << dline << " > tol=" << ps.tol;
          *out_error = oss.str();
        }
        return false;
      }
      const P2 ab = sub(b0, a0);
      const float ab2 = dot(ab, ab);
      if (ab2 < kEps) {
        if (out_error) *out_error = "assert on-segment-ratio: degenerate segment";
        return false;
      }
      const float t = dot(sub(p, a0), ab) / ab2;
      if (t < -1e-4f || t > 1.0f + 1e-4f) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert on-segment-ratio failed: p=" << ps.p.v << ", t=" << t << " outside [0,1]";
          *out_error = oss.str();
        }
        return false;
      }
      const float diff = std::fabs(t - ps.ratio);
      if (diff > ps.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert on-segment-ratio failed: p=" << ps.p.v << ", ratio=" << ps.ratio << ", t=" << t
              << ", diff=" << diff << " > tol=" << ps.tol;
          *out_error = oss.str();
        }
        return false;
      }
      continue;
    }

    if (std::holds_alternative<RelAssertParallelLinePairs>(a)) {
      const auto& pp = std::get<RelAssertParallelLinePairs>(a);
      P2 a0{}, a1{}, b0{}, b1{};
      if (!getline(pp.a0, a0, a1) || !getline(pp.a1, b0, b1)) {
        if (out_error) {
          *out_error = "assert parallel-pairs: invalid line id (a0=" + std::to_string(pp.a0.v) + ", a1=" +
                       std::to_string(pp.a1.v) + ")";
        }
        return false;
      }
      float ang0 = 0.0f;
      if (!line_angle(a0, a1, b0, b1, ang0)) {
        if (out_error) *out_error = "assert parallel-pairs: degenerate line";
        return false;
      }
      if (ang0 > pp.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert parallel-pairs failed: a0=" << pp.a0.v << ", a1=" << pp.a1.v
              << ", angle=" << ang0 << " > tol=" << pp.tol;
          *out_error = oss.str();
        }
        return false;
      }
      if (!getline(pp.b0, a0, a1) || !getline(pp.b1, b0, b1)) {
        if (out_error) {
          *out_error = "assert parallel-pairs: invalid line id (b0=" + std::to_string(pp.b0.v) + ", b1=" +
                       std::to_string(pp.b1.v) + ")";
        }
        return false;
      }
      float ang1 = 0.0f;
      if (!line_angle(a0, a1, b0, b1, ang1)) {
        if (out_error) *out_error = "assert parallel-pairs: degenerate line";
        return false;
      }
      if (ang1 > pp.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert parallel-pairs failed: b0=" << pp.b0.v << ", b1=" << pp.b1.v
              << ", angle=" << ang1 << " > tol=" << pp.tol;
          *out_error = oss.str();
        }
        return false;
      }
      continue;
    }

    if (std::holds_alternative<RelAssertEqualAngleLines>(a)) {
      const auto& ea = std::get<RelAssertEqualAngleLines>(a);
      P2 a0{}, a1{}, b0{}, b1{};
      if (!getline(ea.a0, a0, a1) || !getline(ea.a1, b0, b1)) {
        if (out_error) {
          *out_error = "assert equal-angle: invalid line id (a0=" + std::to_string(ea.a0.v) + ", a1=" +
                       std::to_string(ea.a1.v) + ")";
        }
        return false;
      }
      float ang0 = 0.0f;
      if (!line_angle(a0, a1, b0, b1, ang0)) {
        if (out_error) *out_error = "assert equal-angle: degenerate line";
        return false;
      }
      if (!getline(ea.b0, a0, a1) || !getline(ea.b1, b0, b1)) {
        if (out_error) {
          *out_error = "assert equal-angle: invalid line id (b0=" + std::to_string(ea.b0.v) + ", b1=" +
                       std::to_string(ea.b1.v) + ")";
        }
        return false;
      }
      float ang1 = 0.0f;
      if (!line_angle(a0, a1, b0, b1, ang1)) {
        if (out_error) *out_error = "assert equal-angle: degenerate line";
        return false;
      }
      const float diff = std::fabs(ang0 - ang1);
      if (diff > ea.tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "assert equal-angle failed: diff=" << diff << " > tol=" << ea.tol;
          *out_error = oss.str();
        }
        return false;
      }
      continue;
    }

    if (std::holds_alternative<RelAssertTriangle>(a)) {
      const auto& t = std::get<RelAssertTriangle>(a);
      if (!t.a || !t.b || !t.c) {
        if (out_error) {
          *out_error = "assert triangle: invalid point id (a=" + std::to_string(t.a.v) + ", b=" +
                       std::to_string(t.b.v) + ", c=" + std::to_string(t.c.v) + ")";
        }
        return false;
      }
      continue;
    }
  }

  return true;
}

std::optional<RelTensorPoint> RelProgram::eval_point(RelPointId id,
                                                     TensorBackend* backend_override,
                                                     std::string* out_error) const {
  RelTensorPoint pos;
  if (!evaluate(pos, backend_override, out_error)) return std::nullopt;
  if (!id) return std::nullopt;

  float* data = nullptr;
  size_t count = 0;
  auto guard = map_points_guard(pos, data, count, out_error);
  if (!guard.mapped) return std::nullopt;

  P2 p{};
  if (!read_point(data, count, id, p)) return std::nullopt;

  TensorDesc desc;
  desc.dtype = TensorDType::F32;
  desc.layout = TensorLayout::Dense;
  desc.shape.dims = {1u, 2u};
  TensorBackend* backend = backend_override ? backend_override : &in_memory_backend_singleton();
  AbstractTensor out = AbstractTensor::create(desc, backend);
  if (!out.valid()) return std::nullopt;
  InMemoryBackend* out_backend = dynamic_cast<InMemoryBackend*>(out.backend());
  if (!out_backend) return std::nullopt;
  void* raw = nullptr;
  size_t bytes = 0;
  if (!out_backend->map(out.handle(), &raw, &bytes)) return std::nullopt;
  if (bytes < 2u * sizeof(float)) {
    out_backend->unmap(out.handle());
    return std::nullopt;
  }
  float* dst = static_cast<float*>(raw);
  dst[0] = p.x;
  dst[1] = p.y;
  out_backend->unmap(out.handle());
  return out;
}

std::optional<RelCircleEval> RelProgram::eval_circle(RelCircleId id) const {
  if (!id) return std::nullopt;
  const size_t idx = static_cast<size_t>(id.v - 1);
  if (idx >= circles_.size()) return std::nullopt;

  RelTensorPoint pos;
  if (!evaluate(pos, nullptr, nullptr)) return std::nullopt;
  float* data = nullptr;
  size_t count = 0;
  auto guard = map_points_guard(pos, data, count, nullptr);
  if (!guard.mapped) return std::nullopt;

  const RelCircle& c = circles_[idx].circle;
  if (!c.center) return std::nullopt;
  P2 center{};
  if (!read_point(data, count, c.center, center)) return std::nullopt;

  auto eval_radius = [&](const RelRadiusExpr& r, float& out_r) -> bool {
    if (std::holds_alternative<RelRadiusConstant>(r)) {
      out_r = std::get<RelRadiusConstant>(r).r;
      return out_r > 0.0f;
    }
    const auto& d = std::get<RelRadiusDistance>(r);
    if (!d.a || !d.b) return false;
    P2 pa{}, pb{};
    if (!read_point(data, count, d.a, pa) || !read_point(data, count, d.b, pb)) return false;
    out_r = dist(pa, pb);
    return out_r > 0.0f;
  };

  float r = 0.0f;
  if (!eval_radius(c.radius, r)) return std::nullopt;
  auto center_tensor = make_point_tensor(center, &in_memory_backend_singleton(), nullptr);
  if (!center_tensor) return std::nullopt;
  return RelCircleEval{std::move(*center_tensor), r};
}

std::optional<RelBezierEval> RelProgram::eval_bezier(RelBezierId id) const {
  if (!id) return std::nullopt;
  const size_t idx = static_cast<size_t>(id.v - 1);
  if (idx >= beziers_.size()) return std::nullopt;

  RelTensorPoint pos;
  if (!evaluate(pos, nullptr, nullptr)) return std::nullopt;
  float* data = nullptr;
  size_t count = 0;
  auto guard = map_points_guard(pos, data, count, nullptr);
  if (!guard.mapped) return std::nullopt;

  const RelBezierDef& b = beziers_[idx];
  P2 p0{}, c0{}, c1{}, p1{};
  if (!read_point(data, count, b.p0, p0) || !read_point(data, count, b.c0, c0) ||
      !read_point(data, count, b.c1, c1) || !read_point(data, count, b.p1, p1)) return std::nullopt;

  auto tp0 = make_point_tensor(p0, &in_memory_backend_singleton(), nullptr);
  auto tc0 = make_point_tensor(c0, &in_memory_backend_singleton(), nullptr);
  auto tc1 = make_point_tensor(c1, &in_memory_backend_singleton(), nullptr);
  auto tp1 = make_point_tensor(p1, &in_memory_backend_singleton(), nullptr);
  if (!tp0 || !tc0 || !tc1 || !tp1) return std::nullopt;

  RelBezierEval out{std::move(*tp0), std::move(*tc0), std::move(*tc1), std::move(*tp1)};
  return out;
}

std::optional<RelArcEval> RelProgram::eval_arc(RelArcId id) const {
  if (!id) return std::nullopt;
  const size_t idx = static_cast<size_t>(id.v - 1);
  if (idx >= arcs_.size()) return std::nullopt;

  RelTensorPoint pos;
  if (!evaluate(pos, nullptr, nullptr)) return std::nullopt;
  float* data = nullptr;
  size_t count = 0;
  auto guard = map_points_guard(pos, data, count, nullptr);
  if (!guard.mapped) return std::nullopt;

  const RelArcExpr& e = arcs_[idx].expr;

  if (std::holds_alternative<RelArcOnCircleAngles>(e)) {
    const auto& a = std::get<RelArcOnCircleAngles>(e);
    auto ce = eval_circle(a.circle);
    if (!ce) return std::nullopt;
    return RelArcEval{std::move(ce->center), ce->r, wrap_0_2pi(a.a0), wrap_0_2pi(a.a1), a.ccw};
  }

  if (std::holds_alternative<RelArcOnCircleEndpoints>(e)) {
    const auto& a = std::get<RelArcOnCircleEndpoints>(e);
    auto ce = eval_circle(a.circle);
    if (!ce) return std::nullopt;
    P2 p0{}, p1{};
    if (!read_point(data, count, a.start, p0) || !read_point(data, count, a.end, p1)) return std::nullopt;
    P2 center{};
    if (!read_point_tensor(ce->center, center, nullptr)) return std::nullopt;
    const float a0 = wrap_0_2pi(std::atan2(p0.y - center.y, p0.x - center.x));
    const float a1 = wrap_0_2pi(std::atan2(p1.y - center.y, p1.x - center.x));
    const bool ccw = a.normal_z >= 0.0f;
    return RelArcEval{std::move(ce->center), ce->r, a0, a1, ccw};
  }

  const auto& a3 = std::get<RelArc3>(e);
  P2 p0{}, p1{}, p2{};
  if (!read_point(data, count, a3.p0, p0) || !read_point(data, count, a3.p1, p1) ||
      !read_point(data, count, a3.p2, p2)) return std::nullopt;
  P2 center{};
  float r = 0.0f;
  if (!circumcircle(p0, p1, p2, center, r)) return std::nullopt;

  const float a0 = wrap_0_2pi(std::atan2(p0.y - center.y, p0.x - center.x));
  const float a2 = wrap_0_2pi(std::atan2(p2.y - center.y, p2.x - center.x));
  const float am = wrap_0_2pi(std::atan2(p1.y - center.y, p1.x - center.x));

  const bool ccw = angle_is_between_ccw(a0, a2, am);
  auto center_tensor = make_point_tensor(center, &in_memory_backend_singleton(), nullptr);
  if (!center_tensor) return std::nullopt;
  return RelArcEval{std::move(*center_tensor), r, a0, a2, ccw};
}

bool compile_relglyph_outline(const RelGlyph& glyph,
                              GlyphOutline& out_outline,
                              float tx,
                              float ty,
                              float scale,
                              std::string* out_error) {
  RelTensorPoint pos;
  if (!glyph.program.evaluate(pos, nullptr, out_error)) return false;

  float* data = nullptr;
  size_t count = 0;
  auto guard = map_points_guard(pos, data, count, out_error);
  if (!guard.mapped) return false;

  out_outline = GlyphOutline{};
  out_outline.glyph_id = glyph.glyph_id;

  auto getp = [&](RelPointId id, P2& out) -> bool {
    if (!read_point(data, count, id, out)) return false;
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
      const float a = signed_area(data, count, verts);
      const bool is_ccw = a > 0.0f;
      const bool want_ccw = contour.winding == RelContourWinding::CCW;
      if (is_ccw != want_ccw) {
        std::reverse(verts.begin(), verts.end());
      }
    }

    P2 p0{};
    if (!getp(verts.front(), p0)) return false;
    out_outline.segments.push_back(seg_move(p0));

    P2 last = p0;
    for (size_t i = 1; i < verts.size(); ++i) {
      P2 pi{};
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
