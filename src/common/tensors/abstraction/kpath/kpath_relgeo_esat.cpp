#include "common/tensors/abstraction/kpath/kpath_relgeo_esat.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nodus::tensors::kpath {

namespace {

static int64_t to_microunits(float v) {
  // Quantize to 1e-6 to keep constants stable in hashing/equality.
  return static_cast<int64_t>(std::llround(static_cast<double>(v) * 1'000'000.0));
}

static bool ratio_to_rational(float ratio, int32_t& out_num, uint32_t& out_den) {
  if (!std::isfinite(ratio)) return false;
  const double r = std::clamp(static_cast<double>(ratio), 0.0, 1.0);
  const int64_t den = 1'000'000;
  const int64_t num = static_cast<int64_t>(std::llround(r * static_cast<double>(den)));
  if (num > std::numeric_limits<int32_t>::max()) return false;
  out_num = static_cast<int32_t>(num);
  out_den = static_cast<uint32_t>(den);
  return true;
}

static uint32_t gcd_u32(uint32_t a, uint32_t b) {
  while (b != 0u) {
    const uint32_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

static void reduce_pi_frac(int32_t& num, uint32_t& den) {
  if (den == 0u) {
    num = 0;
    den = 1u;
    return;
  }
  if (num == 0) {
    den = 1u;
    return;
  }

  const uint32_t absn = static_cast<uint32_t>(num < 0 ? -static_cast<int64_t>(num) : static_cast<int64_t>(num));
  const uint32_t g = gcd_u32(absn, den);
  if (g > 1u) {
    num = static_cast<int32_t>(static_cast<int64_t>(num) / static_cast<int64_t>(g));
    den /= g;
  }
}

static void reduce_rational(int32_t& num, uint32_t& den) {
  if (den == 0u) {
    num = 0;
    den = 1u;
    return;
  }
  if (num == 0) {
    den = 1u;
    return;
  }

  const uint32_t absn = static_cast<uint32_t>(num < 0 ? -static_cast<int64_t>(num) : static_cast<int64_t>(num));
  const uint32_t g = gcd_u32(absn, den);
  if (g > 1u) {
    num = static_cast<int32_t>(static_cast<int64_t>(num) / static_cast<int64_t>(g));
    den /= g;
  }
}

struct TermHash {
  size_t operator()(const RelGeoEsatTerm& t) const noexcept {
    // Simple mixing.
    const auto mix = [](size_t h, size_t x) {
      h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      return h;
    };

    size_t h = static_cast<size_t>(t.kind);
    h = mix(h, static_cast<size_t>(t.a));
    h = mix(h, static_cast<size_t>(t.b));
    h = mix(h, static_cast<size_t>(t.c));
    return h;
  }
};

class UnionFind final {
 public:
  uint32_t add() {
    const uint32_t id = static_cast<uint32_t>(parent_.size());
    parent_.push_back(id);
    rank_.push_back(0);
    return id;
  }

  uint32_t find(uint32_t x) {
    uint32_t p = parent_[x];
    if (p == x) return x;
    parent_[x] = find(p);
    return parent_[x];
  }

  bool unite(uint32_t a, uint32_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return false;
    if (rank_[a] < rank_[b]) std::swap(a, b);
    parent_[b] = a;
    if (rank_[a] == rank_[b]) rank_[a]++;
    return true;
  }

 private:
  std::vector<uint32_t> parent_;
  std::vector<uint8_t> rank_;
};

struct CircleInfo final {
  RelPointId center{};
  RelRadiusExpr radius;
};

} // namespace

RelGeoEsatTerm RelGeoEsatTerm::point(RelPointId p) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::Point;
  t.a = p.v;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::dist(RelPointId p, RelPointId q) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::Dist;
  const uint32_t a = std::min(p.v, q.v);
  const uint32_t b = std::max(p.v, q.v);
  t.a = a;
  t.b = b;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::dist2(RelPointId p, RelPointId q) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::Dist2;
  const uint32_t a = std::min(p.v, q.v);
  const uint32_t b = std::max(p.v, q.v);
  t.a = a;
  t.b = b;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::radius(RelCircleId circle) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::CircleRadius;
  t.a = circle.v;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::const_length(float value) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::ConstLength;
  t.c = to_microunits(value);
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::const_rational(int32_t num, uint32_t den) {
  reduce_rational(num, den);
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::ConstRational;
  t.c = static_cast<int64_t>(num);
  t.b = den;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::add(uint32_t left_term, uint32_t right_term) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::Add;
  const uint32_t a = std::min(left_term, right_term);
  const uint32_t b = std::max(left_term, right_term);
  t.a = a;
  t.b = b;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::mul(uint32_t left_term, uint32_t right_term) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::Mul;
  const uint32_t a = std::min(left_term, right_term);
  const uint32_t b = std::max(left_term, right_term);
  t.a = a;
  t.b = b;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::sqrt(uint32_t term) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::Sqrt;
  t.a = term;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::dir(RelLineId line) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::LineDir;
  t.a = line.v;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::dir_perp(RelLineId line) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::LineDirPerp;
  t.a = line.v;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::angle_between(RelLineId l0, RelLineId l1) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::AngleBetweenLines;
  const uint32_t a = std::min(l0.v, l1.v);
  const uint32_t b = std::max(l0.v, l1.v);
  t.a = a;
  t.b = b;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::const_angle_pi(int32_t num, uint32_t den) {
  reduce_pi_frac(num, den);
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::ConstAnglePiFrac;
  // Store signed numerator in c to avoid uint32_t sign issues.
  t.c = static_cast<int64_t>(num);
  t.b = den;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::angle_at(RelPointId a, RelPointId v, RelPointId b) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::AngleAtPoints;
  t.a = v.v;
  const uint32_t lo = std::min(a.v, b.v);
  const uint32_t hi = std::max(a.v, b.v);
  t.b = lo;
  t.c = static_cast<int64_t>(hi);
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::sin(uint32_t angle_term) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::SinAngle;
  t.a = angle_term;
  return t;
}

RelGeoEsatTerm RelGeoEsatTerm::cos(uint32_t angle_term) {
  RelGeoEsatTerm t;
  t.kind = RelGeoEsatTermKind::CosAngle;
  t.a = angle_term;
  return t;
}

struct RelGeoEsatResult::Impl final {
  std::unordered_map<RelGeoEsatTerm, uint32_t, TermHash> term_to_id;
  std::vector<RelGeoEsatTerm> id_to_term;
  UnionFind uf;

  uint32_t intern(const RelGeoEsatTerm& t) {
    auto it = term_to_id.find(t);
    if (it != term_to_id.end()) return it->second;
    const uint32_t id = uf.add();
    term_to_id.emplace(t, id);
    id_to_term.push_back(t);
    return id;
  }

  bool equate(const RelGeoEsatTerm& a, const RelGeoEsatTerm& b) {
    const uint32_t ia = intern(a);
    const uint32_t ib = intern(b);
    return uf.unite(ia, ib);
  }

  bool are_equal(const RelGeoEsatTerm& a, const RelGeoEsatTerm& b) {
    const auto ita = term_to_id.find(a);
    const auto itb = term_to_id.find(b);
    if (ita == term_to_id.end() || itb == term_to_id.end()) return false;
    return uf.find(ita->second) == uf.find(itb->second);
  }
};

bool RelGeoEsatResult::are_equal(const RelGeoEsatTerm& x, const RelGeoEsatTerm& y) const {
  if (!impl_) return false;
  return impl_->are_equal(x, y);
}

std::optional<uint32_t> RelGeoEsatResult::term_id(const RelGeoEsatTerm& term) const {
  if (!impl_) return std::nullopt;
  const auto it = impl_->term_to_id.find(term);
  if (it == impl_->term_to_id.end()) return std::nullopt;
  return it->second;
}

RelGeoEsatResult relgeo_esaturate(const RelProgram& program, const RelGeoEsatOptions& options) {
  RelGeoEsatResult result;
  result.impl_ = std::make_shared<RelGeoEsatResult::Impl>();

  // Index circles for quick center lookup.
  std::unordered_map<uint32_t, CircleInfo> circles;
  circles.reserve(program.circles().size());
  for (const auto& def : program.circles()) {
    circles.emplace(def.id.v, CircleInfo{def.circle.center, def.circle.radius});

    // Circle definition itself can relate Radius(circle) to a term.
    const RelGeoEsatTerm rad = RelGeoEsatTerm::radius(def.id);
    if (const auto* rc = std::get_if<RelRadiusConstant>(&def.circle.radius)) {
      result.impl_->equate(rad, RelGeoEsatTerm::const_length(rc->r));
    } else if (const auto* rd = std::get_if<RelRadiusDistance>(&def.circle.radius)) {
      result.impl_->equate(rad, RelGeoEsatTerm::dist(rd->a, rd->b));
    }
  }

  // Seed equalities from assertions.
  std::vector<uint32_t> mentioned_points;
  mentioned_points.reserve(program.points().size());

  std::vector<uint32_t> mentioned_lines;
  mentioned_lines.reserve(program.lines().size());

  std::vector<RelAssertPerpAt> perp_at;
  perp_at.reserve(8);

  std::vector<RelAssertTriangle> triangles;
  triangles.reserve(8);

  const auto mention_point = [&](RelPointId pid) {
    if (!pid) return;
    mentioned_points.push_back(pid.v);
  };

  const auto mention_line = [&](RelLineId lid) {
    if (!lid) return;
    mentioned_lines.push_back(lid.v);
  };

  for (const auto& scoped : program.assertions()) {
    if (!relgeo_rule_applies(scoped.scope, options.rule_context)) continue;
    const auto& a = scoped.assertion;
    if (const auto* c = std::get_if<RelAssertCoincident>(&a)) {
      result.impl_->equate(RelGeoEsatTerm::point(c->a), RelGeoEsatTerm::point(c->b));
      mention_point(c->a);
      mention_point(c->b);
      continue;
    }

    if (const auto* ed = std::get_if<RelAssertEqualDistance>(&a)) {
      result.impl_->equate(RelGeoEsatTerm::dist(ed->a, ed->b), RelGeoEsatTerm::dist(ed->c, ed->d));
      mention_point(ed->a);
      mention_point(ed->b);
      mention_point(ed->c);
      mention_point(ed->d);
      continue;
    }

    if (const auto* mp = std::get_if<RelAssertMidpoint>(&a)) {
      result.impl_->equate(RelGeoEsatTerm::dist(mp->a, mp->m), RelGeoEsatTerm::dist(mp->m, mp->b));
      mention_point(mp->a);
      mention_point(mp->m);
      mention_point(mp->b);
      continue;
    }

    if (const auto* pp = std::get_if<RelAssertParallelLinePairs>(&a)) {
      result.impl_->equate(RelGeoEsatTerm::dir(pp->a0), RelGeoEsatTerm::dir(pp->a1));
      result.impl_->equate(RelGeoEsatTerm::dir(pp->b0), RelGeoEsatTerm::dir(pp->b1));
      result.impl_->equate(RelGeoEsatTerm::angle_between(pp->a0, pp->b0), RelGeoEsatTerm::angle_between(pp->a1, pp->b1));
      mention_line(pp->a0);
      mention_line(pp->a1);
      mention_line(pp->b0);
      mention_line(pp->b1);
      continue;
    }

    if (const auto* ea = std::get_if<RelAssertEqualAngleLines>(&a)) {
      result.impl_->equate(RelGeoEsatTerm::angle_between(ea->a0, ea->a1), RelGeoEsatTerm::angle_between(ea->b0, ea->b1));
      mention_line(ea->a0);
      mention_line(ea->a1);
      mention_line(ea->b0);
      mention_line(ea->b1);
      continue;
    }

    if (const auto* ps = std::get_if<RelAssertPointOnSegmentRatio>(&a)) {
      int32_t num = 0;
      uint32_t den = 1;
      if (ratio_to_rational(ps->ratio, num, den)) {
        const int32_t num_other = static_cast<int32_t>(static_cast<int64_t>(den) - static_cast<int64_t>(num));
        const uint32_t d_ap = result.impl_->intern(RelGeoEsatTerm::dist(ps->a, ps->p));
        const uint32_t d_pb = result.impl_->intern(RelGeoEsatTerm::dist(ps->p, ps->b));
        const uint32_t r_other = result.impl_->intern(RelGeoEsatTerm::const_rational(num_other, den));
        const uint32_t r_num = result.impl_->intern(RelGeoEsatTerm::const_rational(num, den));
        result.impl_->equate(RelGeoEsatTerm::mul(d_ap, r_other), RelGeoEsatTerm::mul(d_pb, r_num));
      }
      mention_point(ps->a);
      mention_point(ps->p);
      mention_point(ps->b);
      continue;
    }

    if (const auto* tri = std::get_if<RelAssertTriangle>(&a)) {
      triangles.push_back(*tri);
      mention_point(tri->a);
      mention_point(tri->b);
      mention_point(tri->c);
      continue;
    }

    if (const auto* par = std::get_if<RelAssertParallelLines>(&a)) {
      result.impl_->equate(RelGeoEsatTerm::dir(par->a), RelGeoEsatTerm::dir(par->b));
      mention_line(par->a);
      mention_line(par->b);
      continue;
    }

    if (const auto* perp = std::get_if<RelAssertPerpendicularLines>(&a)) {
      // Perpendicular implies both an angle-between fact and a direction consequence.
      result.impl_->equate(RelGeoEsatTerm::angle_between(perp->a, perp->b), RelGeoEsatTerm::const_angle_pi(1, 2));
      result.impl_->equate(RelGeoEsatTerm::dir(perp->a), RelGeoEsatTerm::dir_perp(perp->b));
      result.impl_->equate(RelGeoEsatTerm::dir(perp->b), RelGeoEsatTerm::dir_perp(perp->a));
      mention_line(perp->a);
      mention_line(perp->b);
      continue;
    }

    if (const auto* perp = std::get_if<RelAssertPerpAt>(&a)) {
      perp_at.push_back(*perp);
      mention_point(perp->v);
      mention_point(perp->a);
      mention_point(perp->b);
      continue;
    }

    if (const auto* fr = std::get_if<RelAssertFixedRadius>(&a)) {
      result.impl_->equate(RelGeoEsatTerm::radius(fr->circle), RelGeoEsatTerm::const_length(fr->r));
      continue;
    }

    if (const auto* poc = std::get_if<RelAssertPointOnCircle>(&a)) {
      const auto it = circles.find(poc->circle.v);
      if (it != circles.end()) {
        const RelPointId center = it->second.center;
        result.impl_->equate(RelGeoEsatTerm::dist(center, poc->p), RelGeoEsatTerm::radius(poc->circle));
        mention_point(center);
        mention_point(poc->p);
      }
      continue;
    }
  }

  // De-dup mentioned points.
  std::sort(mentioned_points.begin(), mentioned_points.end());
  mentioned_points.erase(std::unique(mentioned_points.begin(), mentioned_points.end()), mentioned_points.end());

  // De-dup mentioned lines.
  std::sort(mentioned_lines.begin(), mentioned_lines.end());
  mentioned_lines.erase(std::unique(mentioned_lines.begin(), mentioned_lines.end()), mentioned_lines.end());

  // Ensure all mentioned point terms exist in the UF so we can group them.
  for (const uint32_t pv : mentioned_points) {
    result.impl_->intern(RelGeoEsatTerm::point(RelPointId(pv)));
  }

  // Ensure all mentioned line dir terms exist.
  for (const uint32_t lv : mentioned_lines) {
    result.impl_->intern(RelGeoEsatTerm::dir(RelLineId(lv)));
    result.impl_->intern(RelGeoEsatTerm::dir_perp(RelLineId(lv)));
  }

  // Map line definitions by their unordered point pairs.
  std::unordered_map<uint64_t, RelLineId> line_by_points;
  line_by_points.reserve(program.lines().size());
  for (const auto& l : program.lines()) {
    if (!l.a || !l.b) continue;
    const uint32_t lo = std::min(l.a.v, l.b.v);
    const uint32_t hi = std::max(l.a.v, l.b.v);
    const uint64_t key = (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    line_by_points.emplace(key, l.id);
  }

  const auto find_line = [&](RelPointId p, RelPointId q) -> std::optional<RelLineId> {
    if (!p || !q) return std::nullopt;
    const uint32_t lo = std::min(p.v, q.v);
    const uint32_t hi = std::max(p.v, q.v);
    const uint64_t key = (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    auto it = line_by_points.find(key);
    if (it == line_by_points.end()) return std::nullopt;
    return it->second;
  };

  // Connect explicit vertex angles to line-pair angles when both lines exist.
  for (const auto& def : program.angles()) {
    if (!def.a || !def.v || !def.b) continue;
    const auto l0 = find_line(def.v, def.a);
    const auto l1 = find_line(def.v, def.b);
    if (!l0 || !l1) continue;
    const RelGeoEsatTerm ang_lines = RelGeoEsatTerm::angle_between(*l0, *l1);
    const RelGeoEsatTerm ang_pts = RelGeoEsatTerm::angle_at(def.a, def.v, def.b);
    result.impl_->equate(ang_lines, ang_pts);
    mention_line(*l0);
    mention_line(*l1);
    mention_point(def.a);
    mention_point(def.v);
    mention_point(def.b);
  }

  const auto equate_dist_sqrt = [&](RelPointId a, RelPointId b) {
    const uint32_t d2 = result.impl_->intern(RelGeoEsatTerm::dist2(a, b));
    const RelGeoEsatTerm root = RelGeoEsatTerm::sqrt(d2);
    result.impl_->equate(RelGeoEsatTerm::dist(a, b), root);
  };

  for (const auto& perp : perp_at) {
    const uint32_t d2_av = result.impl_->intern(RelGeoEsatTerm::dist2(perp.a, perp.v));
    const uint32_t d2_vb = result.impl_->intern(RelGeoEsatTerm::dist2(perp.v, perp.b));
    const RelGeoEsatTerm sum = RelGeoEsatTerm::add(d2_av, d2_vb);
    result.impl_->equate(RelGeoEsatTerm::dist2(perp.a, perp.b), sum);
    equate_dist_sqrt(perp.a, perp.b);
    equate_dist_sqrt(perp.a, perp.v);
    equate_dist_sqrt(perp.v, perp.b);
  }

  // Triangle trigonometric identities (symbolic only).
  for (const auto& tri : triangles) {
    if (!tri.a || !tri.b || !tri.c) continue;
    const RelGeoEsatTerm ang_a = RelGeoEsatTerm::angle_at(tri.b, tri.a, tri.c);
    const RelGeoEsatTerm ang_b = RelGeoEsatTerm::angle_at(tri.a, tri.b, tri.c);
    const RelGeoEsatTerm ang_c = RelGeoEsatTerm::angle_at(tri.a, tri.c, tri.b);

    const uint32_t ang_a_id = result.impl_->intern(ang_a);
    const uint32_t ang_b_id = result.impl_->intern(ang_b);
    const uint32_t ang_c_id = result.impl_->intern(ang_c);

    const uint32_t sum_ab = result.impl_->intern(RelGeoEsatTerm::add(ang_a_id, ang_b_id));
    result.impl_->equate(RelGeoEsatTerm::add(sum_ab, ang_c_id), RelGeoEsatTerm::const_angle_pi(1, 1));

    const uint32_t sin_a = result.impl_->intern(RelGeoEsatTerm::sin(ang_a_id));
    const uint32_t sin_b = result.impl_->intern(RelGeoEsatTerm::sin(ang_b_id));
    const uint32_t sin_c = result.impl_->intern(RelGeoEsatTerm::sin(ang_c_id));

    const uint32_t len_a = result.impl_->intern(RelGeoEsatTerm::dist(tri.b, tri.c));
    const uint32_t len_b = result.impl_->intern(RelGeoEsatTerm::dist(tri.a, tri.c));
    const uint32_t len_c = result.impl_->intern(RelGeoEsatTerm::dist(tri.a, tri.b));

    result.impl_->equate(RelGeoEsatTerm::mul(sin_a, len_b), RelGeoEsatTerm::mul(sin_b, len_a));

    result.impl_->equate(RelGeoEsatTerm::mul(sin_a, len_c), RelGeoEsatTerm::mul(sin_c, len_a));

    const uint32_t cos_a = result.impl_->intern(RelGeoEsatTerm::cos(ang_a_id));
    const uint32_t cos_b = result.impl_->intern(RelGeoEsatTerm::cos(ang_b_id));
    const uint32_t cos_c = result.impl_->intern(RelGeoEsatTerm::cos(ang_c_id));

    const uint32_t len_a2 = result.impl_->intern(RelGeoEsatTerm::dist2(tri.b, tri.c));
    const uint32_t len_b2 = result.impl_->intern(RelGeoEsatTerm::dist2(tri.a, tri.c));
    const uint32_t len_c2 = result.impl_->intern(RelGeoEsatTerm::dist2(tri.a, tri.b));

    const uint32_t two_neg = result.impl_->intern(RelGeoEsatTerm::const_rational(-2, 1));

    const uint32_t ab = result.impl_->intern(RelGeoEsatTerm::mul(len_b, len_c));
    const uint32_t ac = result.impl_->intern(RelGeoEsatTerm::mul(len_a, len_c));
    const uint32_t bc = result.impl_->intern(RelGeoEsatTerm::mul(len_a, len_b));

    const uint32_t term_a = result.impl_->intern(RelGeoEsatTerm::mul(cos_a, ab));
    const uint32_t term_b = result.impl_->intern(RelGeoEsatTerm::mul(cos_b, ac));
    const uint32_t term_c = result.impl_->intern(RelGeoEsatTerm::mul(cos_c, bc));

    const uint32_t neg_a = result.impl_->intern(RelGeoEsatTerm::mul(two_neg, term_a));
    const uint32_t neg_b = result.impl_->intern(RelGeoEsatTerm::mul(two_neg, term_b));
    const uint32_t neg_c = result.impl_->intern(RelGeoEsatTerm::mul(two_neg, term_c));

    const uint32_t sum_ba = result.impl_->intern(RelGeoEsatTerm::add(len_b2, len_c2));
    const uint32_t sum_ca = result.impl_->intern(RelGeoEsatTerm::add(len_a2, len_c2));
    const uint32_t sum_ab2 = result.impl_->intern(RelGeoEsatTerm::add(len_a2, len_b2));

    result.impl_->equate(RelGeoEsatTerm::dist2(tri.b, tri.c), RelGeoEsatTerm::add(sum_ba, neg_a));
    result.impl_->equate(RelGeoEsatTerm::dist2(tri.a, tri.c), RelGeoEsatTerm::add(sum_ca, neg_b));
    result.impl_->equate(RelGeoEsatTerm::dist2(tri.a, tri.b), RelGeoEsatTerm::add(sum_ab2, neg_c));
  }

  const uint32_t sqrt2 = result.impl_->intern(RelGeoEsatTerm::sqrt(
    result.impl_->intern(RelGeoEsatTerm::const_rational(2, 1))));
  const uint32_t zero_rat = result.impl_->intern(RelGeoEsatTerm::const_rational(0, 1));
  const uint32_t zero_len = result.impl_->intern(RelGeoEsatTerm::const_length(0.0f));
  result.impl_->equate(RelGeoEsatTerm::sqrt(zero_rat), RelGeoEsatTerm::const_rational(0, 1));
  result.impl_->equate(RelGeoEsatTerm::sqrt(zero_len), RelGeoEsatTerm::const_length(0.0f));

  // Basic saturation:
  // - if Point(a) == Point(b), then Dist(a,x) == Dist(b,x) for mentioned x.
  // - if Dir(l0) == Dir(l1), then AngleBetweenLines(l0,l1) == 0.
  // - if Dir(l1) == DirPerp(l0), then AngleBetweenLines(l0,l1) == pi/2.
  const int iters = std::max(0, options.max_iterations);
  for (int iter = 0; iter < iters; ++iter) {
    bool changed = false;

    // Group mentioned points by UF representative of their Point(...) term.
    std::unordered_map<uint32_t, std::vector<uint32_t>> groups;
    groups.reserve(mentioned_points.size());
    for (const uint32_t pv : mentioned_points) {
      const RelGeoEsatTerm pt = RelGeoEsatTerm::point(RelPointId(pv));
      const uint32_t tid = result.impl_->intern(pt);
      const uint32_t root = result.impl_->uf.find(tid);
      groups[root].push_back(pv);
    }

    for (const auto& kv : groups) {
      const std::vector<uint32_t>& pts = kv.second;
      if (pts.size() < 2) continue;

      const uint32_t rep = pts[0];
      for (size_t i = 1; i < pts.size(); ++i) {
        const uint32_t other = pts[i];
        for (const uint32_t xv : mentioned_points) {
          changed |= result.impl_->equate(
            RelGeoEsatTerm::dist(RelPointId(rep), RelPointId(xv)),
            RelGeoEsatTerm::dist(RelPointId(other), RelPointId(xv)));
        }
      }
    }

    if (!mentioned_lines.empty()) {
      std::unordered_map<uint32_t, std::vector<uint32_t>> line_groups;
      line_groups.reserve(mentioned_lines.size());
      for (const uint32_t lv : mentioned_lines) {
        const RelGeoEsatTerm d = RelGeoEsatTerm::dir(RelLineId(lv));
        const uint32_t tid = result.impl_->intern(d);
        const uint32_t root = result.impl_->uf.find(tid);
        line_groups[root].push_back(lv);
      }

      const RelGeoEsatTerm zero = RelGeoEsatTerm::const_angle_pi(0, 1);
      for (const auto& kv : line_groups) {
        const std::vector<uint32_t>& ls = kv.second;
        if (ls.size() < 2) continue;
        const uint32_t rep = ls[0];
        for (size_t i = 1; i < ls.size(); ++i) {
          const uint32_t other = ls[i];
          changed |= result.impl_->equate(RelGeoEsatTerm::angle_between(RelLineId(rep), RelLineId(other)), zero);
        }
      }

      // Perp-direction consequences.
      // For each pair (L0, L1), if Dir(L1) == DirPerp(L0), derive AngleBetween(L0,L1) == pi/2.
      const RelGeoEsatTerm half_pi = RelGeoEsatTerm::const_angle_pi(1, 2);
      for (const uint32_t l0v : mentioned_lines) {
        const RelGeoEsatTerm d0p = RelGeoEsatTerm::dir_perp(RelLineId(l0v));
        for (const uint32_t l1v : mentioned_lines) {
          if (l0v == l1v) continue;
          const RelGeoEsatTerm d1 = RelGeoEsatTerm::dir(RelLineId(l1v));
          // If the two direction terms are equal, we can assert the angle-between fact.
          if (result.impl_->are_equal(d1, d0p)) {
            changed |= result.impl_->equate(RelGeoEsatTerm::angle_between(RelLineId(l0v), RelLineId(l1v)), half_pi);
          }
        }
      }
    }

    for (const auto& perp : perp_at) {
      const RelGeoEsatTerm d2_av = RelGeoEsatTerm::dist2(perp.a, perp.v);
      const RelGeoEsatTerm d2_vb = RelGeoEsatTerm::dist2(perp.v, perp.b);
      if (result.impl_->are_equal(d2_av, d2_vb)) {
        const uint32_t dist_av = result.impl_->intern(RelGeoEsatTerm::dist(perp.v, perp.a));
        const RelGeoEsatTerm scaled = RelGeoEsatTerm::mul(sqrt2, dist_av);
        changed |= result.impl_->equate(RelGeoEsatTerm::dist(perp.a, perp.b), scaled);
      }
    }

    if (!changed) break;
  }

  return result;
}

} // namespace nodus::tensors::kpath
