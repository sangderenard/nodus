#include "common/tensors/abstraction/kpath/kpath_relgeo_stencil.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>

namespace nodus::tensors::kpath {

namespace {

struct FrameKey final {
  uint32_t a = 0;
  uint32_t b = 0;
  bool operator==(const FrameKey& other) const { return a == other.a && b == other.b; }
};

struct FrameKeyHash final {
  size_t operator()(const FrameKey& k) const noexcept {
    return (static_cast<size_t>(k.a) << 32) ^ static_cast<size_t>(k.b);
  }
};

struct PerpKey final {
  uint32_t v = 0;
  uint32_t a = 0;
  uint32_t b = 0;
  bool operator==(const PerpKey& other) const { return v == other.v && a == other.a && b == other.b; }
};

struct PerpKeyHash final {
  size_t operator()(const PerpKey& k) const noexcept {
    size_t h = static_cast<size_t>(k.v);
    h ^= static_cast<size_t>(k.a) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= static_cast<size_t>(k.b) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

static uint64_t pair_key(uint32_t a, uint32_t b) {
  return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

struct AxisState final {
  std::unordered_map<uint64_t, RelAxisOrder> order;

  bool set_order(RelPointId a, RelPointId b, RelAxisOrder rel, std::vector<RelStencilIssue>& issues) {
    if (!a || !b || rel == RelAxisOrder::Unknown) return true;
    if (a == b) {
      if (rel == RelAxisOrder::Less || rel == RelAxisOrder::Greater) {
        issues.push_back(RelStencilIssue{RelStencilIssue::Kind::Conflict, "axis order conflict: point ordered against itself"});
        return false;
      }
      return true;
    }

    const uint64_t key = pair_key(a.v, b.v);
    auto it = order.find(key);
    if (it == order.end()) {
      order.emplace(key, rel);
      return true;
    }
    if (it->second != rel) {
      issues.push_back(RelStencilIssue{RelStencilIssue::Kind::Conflict, "axis order conflict"});
      return false;
    }
    return true;
  }
};

struct FrameState final {
  RelGeoStencilFrame def;
  std::vector<AxisState> axes;
};

} // namespace

RelGeoStencil relgeo_build_stencil(const RelProgram& program, const RelGeoStencilOptions& options) {
  RelGeoStencil out;

  FrameState global;
  global.def.id = 1;
  global.def.dims = std::max(1u, options.dims);
  global.def.label = "global";
  global.axes.resize(global.def.dims);
  out.global_frame = global.def.id;

  std::vector<FrameState> frames;
  frames.push_back(global);

  std::unordered_map<uint32_t, uint32_t> line_to_frame;
  std::unordered_map<FrameKey, uint32_t, FrameKeyHash> segment_to_frame;
  std::unordered_map<PerpKey, uint32_t, PerpKeyHash> perp_to_frame;

  auto add_frame = [&](std::string label, RelPointId origin) -> uint32_t {
    FrameState fs;
    fs.def.id = static_cast<uint32_t>(frames.size() + 1);
    fs.def.dims = std::max(1u, options.dims);
    fs.def.label = std::move(label);
    fs.def.origin = origin;
    fs.axes.resize(fs.def.dims);
    frames.push_back(std::move(fs));
    return frames.back().def.id;
  };

  if (options.include_local_frames) {
    for (const auto& def : program.points()) {
      const std::string label = "local_point_" + std::to_string(def.id.v);
      (void)add_frame(label, def.id);
    }
  }

  out.points.reserve(program.points().size());
  for (const auto& def : program.points()) {
    out.points.push_back(def.id);
  }

  auto ensure_line_frame = [&](RelLineId line, RelPointId a, RelPointId b) -> uint32_t {
    if (!line) return 0;
    auto it = line_to_frame.find(line.v);
    if (it != line_to_frame.end()) return it->second;
    const std::string label = "line_" + std::to_string(line.v);
    const uint32_t fid = add_frame(label, a);
    line_to_frame.emplace(line.v, fid);
    (void)b;
    return fid;
  };

  auto ensure_segment_frame = [&](RelPointId a, RelPointId b) -> uint32_t {
    if (!a || !b) return 0;
    const uint32_t lo = std::min(a.v, b.v);
    const uint32_t hi = std::max(a.v, b.v);
    FrameKey key{lo, hi};
    auto it = segment_to_frame.find(key);
    if (it != segment_to_frame.end()) return it->second;
    const std::string label = "segment_" + std::to_string(lo) + "_" + std::to_string(hi);
    const uint32_t fid = add_frame(label, a);
    segment_to_frame.emplace(key, fid);
    return fid;
  };

  auto ensure_perp_frame = [&](RelPointId v, RelPointId a, RelPointId b) -> uint32_t {
    const uint32_t lo = std::min(a.v, b.v);
    const uint32_t hi = std::max(a.v, b.v);
    PerpKey key{v.v, lo, hi};
    auto it = perp_to_frame.find(key);
    if (it != perp_to_frame.end()) return it->second;
    const std::string label = "perp_at_" + std::to_string(v.v);
    const uint32_t fid = add_frame(label, v);
    perp_to_frame.emplace(key, fid);
    return fid;
  };

  auto add_axis_order = [&](uint32_t frame, uint32_t axis, RelPointId a, RelPointId b, RelAxisOrder order) {
    if (frame == 0 || axis >= frames[frame - 1].axes.size()) return;
    (void)frames[frame - 1].axes[axis].set_order(a, b, order, out.issues);
    out.axis_orders.push_back(RelAxisOrderConstraint{frame, axis, a, b, order});
  };

  auto add_axis_sign = [&](uint32_t frame, uint32_t axis, RelPointId p, RelAxisSign sign) {
    if (!p || frame == 0 || axis >= frames[frame - 1].axes.size()) return;
    out.axis_signs.push_back(RelAxisSignConstraint{frame, axis, p, sign});
  };

  auto add_between = [&](uint32_t frame, uint32_t axis, RelPointId a, RelPointId m, RelPointId b) {
    if (!a || !m || !b || frame == 0) return;
    out.betweens.push_back(RelBetweenConstraint{frame, axis, a, m, b});
  };

  for (const auto& def : program.points()) {
    if (std::holds_alternative<RelPointFree>(def.expr)) {
      out.free_points.push_back(def.id);
    }
  }

  for (const auto& def : program.lines()) {
    ensure_line_frame(def.id, def.a, def.b);
  }

  for (const auto& a : program.assertions()) {
    if (const auto* c = std::get_if<RelAssertCoincident>(&a)) {
      for (uint32_t axis = 0; axis < options.dims; ++axis) {
        add_axis_order(out.global_frame, axis, c->a, c->b, RelAxisOrder::Equal);
      }
      continue;
    }

    if (const auto* inc = std::get_if<RelAssertPointOnLine>(&a)) {
      if (!inc->line || !inc->p) continue;
      const auto idx = static_cast<size_t>(inc->line.v - 1);
      if (idx >= program.lines().size()) continue;
      const RelLineDef& l = program.lines()[idx];
      const uint32_t fid = ensure_line_frame(inc->line, l.a, l.b);
      add_axis_sign(fid, 1, inc->p, RelAxisSign::Zero);
      add_axis_sign(fid, 1, l.a, RelAxisSign::Zero);
      add_axis_sign(fid, 1, l.b, RelAxisSign::Zero);
      continue;
    }

    if (const auto* perp = std::get_if<RelAssertPerpAt>(&a)) {
      if (!perp->v || !perp->a || !perp->b) continue;
      const uint32_t fid = ensure_perp_frame(perp->v, perp->a, perp->b);
      // Local frame: axis0 along v->a, axis1 along v->b.
      add_axis_sign(fid, 0, perp->v, RelAxisSign::Zero);
      add_axis_sign(fid, 1, perp->v, RelAxisSign::Zero);
      add_axis_sign(fid, 0, perp->a, RelAxisSign::Positive);
      add_axis_sign(fid, 1, perp->a, RelAxisSign::Zero);
      add_axis_sign(fid, 0, perp->b, RelAxisSign::Zero);
      add_axis_sign(fid, 1, perp->b, RelAxisSign::Positive);
      continue;
    }

    if (const auto* par = std::get_if<RelAssertParallelLines>(&a)) {
      if (!par->a || !par->b) continue;
      const auto ia = static_cast<size_t>(par->a.v - 1);
      const auto ib = static_cast<size_t>(par->b.v - 1);
      if (ia >= program.lines().size() || ib >= program.lines().size()) continue;
      const uint32_t fa = ensure_line_frame(par->a, program.lines()[ia].a, program.lines()[ia].b);
      const uint32_t fb = ensure_line_frame(par->b, program.lines()[ib].a, program.lines()[ib].b);
      out.frame_relations.push_back(RelFrameAxisConstraint{fa, 0, fb, 0, RelFrameAxisRelation::Parallel, 0});
      out.frame_relations.push_back(RelFrameAxisConstraint{fa, 1, fb, 1, RelFrameAxisRelation::Parallel, 0});
      continue;
    }

    if (const auto* perp = std::get_if<RelAssertPerpendicularLines>(&a)) {
      if (!perp->a || !perp->b) continue;
      const auto ia = static_cast<size_t>(perp->a.v - 1);
      const auto ib = static_cast<size_t>(perp->b.v - 1);
      if (ia >= program.lines().size() || ib >= program.lines().size()) continue;
      const uint32_t fa = ensure_line_frame(perp->a, program.lines()[ia].a, program.lines()[ia].b);
      const uint32_t fb = ensure_line_frame(perp->b, program.lines()[ib].a, program.lines()[ib].b);
      out.frame_relations.push_back(RelFrameAxisConstraint{fa, 0, fb, 0, RelFrameAxisRelation::Perpendicular, 0});
      out.frame_relations.push_back(RelFrameAxisConstraint{fa, 0, fb, 1, RelFrameAxisRelation::Parallel, 0});
      out.frame_relations.push_back(RelFrameAxisConstraint{fa, 1, fb, 0, RelFrameAxisRelation::Parallel, 0});
      continue;
    }
  }

  for (const auto& def : program.points()) {
    if (std::holds_alternative<RelPointLerp>(def.expr)) {
      const auto& e = std::get<RelPointLerp>(def.expr);
      const uint32_t fid = ensure_segment_frame(e.a, e.b);
      if (fid != 0) {
        if (e.u <= 0.0f) {
          add_axis_order(fid, 0, def.id, e.a, RelAxisOrder::Equal);
        } else if (e.u >= 1.0f) {
          add_axis_order(fid, 0, def.id, e.b, RelAxisOrder::Equal);
        } else {
          add_between(fid, 0, e.a, def.id, e.b);
          add_axis_order(fid, 0, e.a, def.id, RelAxisOrder::Less);
          add_axis_order(fid, 0, def.id, e.b, RelAxisOrder::Less);
        }
      }
      continue;
    }

    if (std::holds_alternative<RelPointOffset>(def.expr)) {
      const auto& e = std::get<RelPointOffset>(def.expr);
      if (e.dx > 0.0f) add_axis_order(out.global_frame, 0, e.base, def.id, RelAxisOrder::Less);
      if (e.dx < 0.0f) add_axis_order(out.global_frame, 0, e.base, def.id, RelAxisOrder::Greater);
      if (e.dy > 0.0f) add_axis_order(out.global_frame, 1, e.base, def.id, RelAxisOrder::Less);
      if (e.dy < 0.0f) add_axis_order(out.global_frame, 1, e.base, def.id, RelAxisOrder::Greater);
      continue;
    }

    if (std::holds_alternative<RelPointLineLineIntersection>(def.expr)) {
      const auto& e = std::get<RelPointLineLineIntersection>(def.expr);
      if (e.l0) {
        const auto idx = static_cast<size_t>(e.l0.v - 1);
        if (idx < program.lines().size()) {
          const auto& l = program.lines()[idx];
          const uint32_t fid = ensure_line_frame(e.l0, l.a, l.b);
          add_axis_sign(fid, 1, def.id, RelAxisSign::Zero);
        }
      }
      if (e.l1) {
        const auto idx = static_cast<size_t>(e.l1.v - 1);
        if (idx < program.lines().size()) {
          const auto& l = program.lines()[idx];
          const uint32_t fid = ensure_line_frame(e.l1, l.a, l.b);
          add_axis_sign(fid, 1, def.id, RelAxisSign::Zero);
        }
      }
      continue;
    }
  }

  out.frames.reserve(frames.size());
  for (const auto& f : frames) out.frames.push_back(f.def);

  return out;
}

} // namespace nodus::tensors::kpath
