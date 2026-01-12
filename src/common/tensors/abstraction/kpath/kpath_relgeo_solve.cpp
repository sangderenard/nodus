#include "common/tensors/abstraction/kpath/kpath_relgeo_solve.h"

#include "common/tensors/abstraction/graph_sparse.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo_ir.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <sstream>
#include <unordered_set>

namespace nodus::tensors::kpath {

namespace {

static std::optional<double> as_f64(const GraphIrValue& v) {
  if (const auto* p = std::get_if<double>(&v)) return *p;
  if (const auto* u = std::get_if<uint32_t>(&v)) return static_cast<double>(*u);
  if (const auto* b = std::get_if<bool>(&v)) return *b ? 1.0 : 0.0;
  return std::nullopt;
}

static std::optional<uint32_t> as_u32(const GraphIrValue& v) {
  if (const auto* p = std::get_if<uint32_t>(&v)) return *p;
  if (const auto* d = std::get_if<double>(&v)) {
    if (*d >= 0.0 && *d <= 4294967295.0) return static_cast<uint32_t>(*d);
  }
  return std::nullopt;
}

static std::optional<bool> as_bool(const GraphIrValue& v) {
  if (const auto* p = std::get_if<bool>(&v)) return *p;
  if (const auto* d = std::get_if<double>(&v)) return *d >= 0.5;
  return std::nullopt;
}

struct NodeRec final {
  std::string kind;
  std::unordered_map<std::string, GraphIrValue> attrs;
};

struct EdgeRec final {
  std::string kind;
  std::unordered_map<std::string, GraphIrValue> attrs;
};

static RelVec2 sub(RelVec2 a, RelVec2 b) {
  return RelVec2{a.x - b.x, a.y - b.y};
}

static RelVec2 add(RelVec2 a, RelVec2 b) {
  return RelVec2{a.x + b.x, a.y + b.y};
}

static RelVec2 mul(RelVec2 a, float s) {
  return RelVec2{a.x * s, a.y * s};
}

static float dot(RelVec2 a, RelVec2 b) {
  return a.x * b.x + a.y * b.y;
}

static float cross(RelVec2 a, RelVec2 b) {
  return a.x * b.y - a.y * b.x;
}

static float len(RelVec2 a) {
  return std::sqrt(dot(a, a));
}

static float dist(RelVec2 a, RelVec2 b) {
  return len(sub(a, b));
}

static bool line_line_intersection(RelVec2 p0, RelVec2 p1, RelVec2 q0, RelVec2 q1, float eps, RelVec2& out) {
  const RelVec2 r = sub(p1, p0);
  const RelVec2 s = sub(q1, q0);
  const float denom = cross(r, s);
  if (std::fabs(denom) <= eps) return false;

  const RelVec2 qp = sub(q0, p0);
  const float t = cross(qp, s) / denom;
  out = add(p0, mul(r, t));
  return true;
}

static bool point_on_line(RelVec2 p, RelVec2 a, RelVec2 b, float eps) {
  const RelVec2 ab = sub(b, a);
  const float ab_len = len(ab);
  if (ab_len <= eps) return false;

  // Distance from p to infinite line through a->b.
  const float area2 = std::fabs(cross(sub(p, a), ab));
  const float dist = area2 / ab_len;
  return dist <= eps;
}

static RelVec2 perp(RelVec2 a) {
  return RelVec2{-a.y, a.x};
}

static bool normalize(RelVec2 v, float eps, RelVec2& out) {
  const float l = len(v);
  if (l <= eps) return false;
  out = RelVec2{v.x / l, v.y / l};
  return true;
}

static RelVec2 project_point_to_line(RelVec2 p, RelVec2 a, RelVec2 b, float eps) {
  const RelVec2 ab = sub(b, a);
  const float dd = dot(ab, ab);
  if (dd <= eps) return a;
  const float t = dot(sub(p, a), ab) / dd;
  return add(a, mul(ab, t));
}

static RelVec2 project_point_to_circle(RelVec2 p, RelVec2 center, float r, float eps) {
  RelVec2 dir;
  if (!normalize(sub(p, center), eps, dir)) {
    dir = RelVec2{1.0f, 0.0f};
  }
  return add(center, mul(dir, r));
}

static float signed_distance_to_line(RelVec2 p, RelVec2 a, RelVec2 b, float eps, RelVec2* out_unit_normal = nullptr) {
  RelVec2 dir;
  if (!normalize(sub(b, a), eps, dir)) {
    if (out_unit_normal) *out_unit_normal = RelVec2{0.0f, 0.0f};
    return 0.0f;
  }
  RelVec2 n = perp(dir);
  // n is already unit since dir is unit.
  if (out_unit_normal) *out_unit_normal = n;
  return dot(sub(p, a), n);
}

static uint64_t mix_u64(uint64_t x) {
  // splitmix64
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  x = x ^ (x >> 31);
  return x;
}

static RelVec2 default_free_point_seed(uint32_t id) {
  // Deterministic non-degenerate seed so projection constraints have something to work with.
  const uint64_t h = mix_u64(static_cast<uint64_t>(id) * 0x1000003dull);
  const double a = (static_cast<double>(h & 0xFFFFFFFFull) / static_cast<double>(0xFFFFFFFFull)) * (2.0 * 3.141592653589793);
  const double r = 1.0 + 0.5 * (static_cast<double>((h >> 32) & 0xFFFFFFFFull) / static_cast<double>(0xFFFFFFFFull));
  return RelVec2{static_cast<float>(r * std::cos(a)), static_cast<float>(r * std::sin(a))};
}

} // namespace

bool relgeo_solve_points_from_pure_edits(std::span<const nodus::tensors::GraphEdit> edits,
                                        const RelGeoSolveInputs& in,
                                        RelGeoSolveOutput& out,
                                        std::string* out_error) {
  out.points.clear();

  std::unordered_map<uint32_t, NodeRec> nodes;
  std::unordered_map<uint32_t, EdgeRec> edges;
  std::vector<uint32_t> contour_nodes;

  for (const auto& e : edits) {
    if (e.kind == GraphEditKind::AddNode) {
      nodes[e.a].kind = e.key;
      if (e.key == "relgeo.contour") contour_nodes.push_back(e.a);
      continue;
    }
    if (e.kind == GraphEditKind::AddEdge) {
      edges[e.a].kind = e.key;
      continue;
    }
    if (e.kind == GraphEditKind::SetAttr) {
      const GraphObjectKind kind = graph_id_kind(e.a);
      if (kind == GraphObjectKind::Node) {
        nodes[e.a].attrs[e.key] = e.value;
      } else if (kind == GraphObjectKind::Edge) {
        edges[e.a].attrs[e.key] = e.value;
      }
      continue;
    }
  }

  auto get_u32_attr_node = [&](uint32_t nid, const char* key, uint32_t& out_u) -> bool {
    auto itn = nodes.find(nid);
    if (itn == nodes.end()) return false;
    auto ita = itn->second.attrs.find(key);
    if (ita == itn->second.attrs.end()) return false;
    auto u = as_u32(ita->second);
    if (!u) return false;
    out_u = *u;
    return true;
  };

  auto get_f64_attr_node = [&](uint32_t nid, const char* key, double& out_d) -> bool {
    auto itn = nodes.find(nid);
    if (itn == nodes.end()) return false;
    auto ita = itn->second.attrs.find(key);
    if (ita == itn->second.attrs.end()) return false;
    auto d = as_f64(ita->second);
    if (!d) return false;
    out_d = *d;
    return true;
  };

  auto get_u32_attr_edge = [&](uint32_t eid, const char* key, uint32_t& out_u) -> bool {
    auto ite = edges.find(eid);
    if (ite == edges.end()) return false;
    auto ita = ite->second.attrs.find(key);
    if (ita == ite->second.attrs.end()) return false;
    auto u = as_u32(ita->second);
    if (!u) return false;
    out_u = *u;
    return true;
  };

  auto get_f64_attr_edge = [&](uint32_t eid, const char* key, double& out_d) -> bool {
    auto ite = edges.find(eid);
    if (ite == edges.end()) return false;
    auto ita = ite->second.attrs.find(key);
    if (ita == ite->second.attrs.end()) return false;
    auto d = as_f64(ita->second);
    if (!d) return false;
    out_d = *d;
    return true;
  };

  // Figure out which point nodes are required (reachable from contours/nouns/relations).
  std::unordered_set<uint32_t> required_points;

  // Contours.
  for (uint32_t nid : contour_nodes) {
    for (uint32_t i = 0;; ++i) {
      const std::string k = "v" + std::to_string(i);
      uint32_t v = 0;
      if (!get_u32_attr_node(nid, k.c_str(), v)) break;
      required_points.insert(v);
    }
  }

  // Nouns and derived points.
  for (const auto& [nid, nr] : nodes) {
    if (nr.kind == "relgeo.segment" || nr.kind == "relgeo.line") {
      uint32_t a = 0, b = 0;
      if (get_u32_attr_node(nid, "a", a)) required_points.insert(a);
      if (get_u32_attr_node(nid, "b", b)) required_points.insert(b);
    } else if (nr.kind == "relgeo.ray") {
      uint32_t o = 0, t = 0;
      if (get_u32_attr_node(nid, "origin", o)) required_points.insert(o);
      if (get_u32_attr_node(nid, "through", t)) required_points.insert(t);
    } else if (nr.kind == "relgeo.angle") {
      uint32_t a = 0, v = 0, b = 0;
      if (get_u32_attr_node(nid, "a", a)) required_points.insert(a);
      if (get_u32_attr_node(nid, "v", v)) required_points.insert(v);
      if (get_u32_attr_node(nid, "b", b)) required_points.insert(b);
    } else if (nr.kind == "relgeo.lerp") {
      uint32_t a = 0, b = 0;
      if (get_u32_attr_node(nid, "a", a)) required_points.insert(a);
      if (get_u32_attr_node(nid, "b", b)) required_points.insert(b);
      required_points.insert(nid);
    } else if (nr.kind == "relgeo.offset") {
      uint32_t base = 0;
      if (get_u32_attr_node(nid, "base", base)) required_points.insert(base);
      required_points.insert(nid);
    } else if (nr.kind == "relgeo.lerp_ratio") {
      uint32_t a = 0, b = 0;
      if (get_u32_attr_node(nid, "a", a)) required_points.insert(a);
      if (get_u32_attr_node(nid, "b", b)) required_points.insert(b);
      required_points.insert(nid);
    } else if (nr.kind == "relgeo.llint") {
      required_points.insert(nid);
      // Line endpoints will be pulled in once we see the line nodes.
    } else if (nr.kind == "relgeo.point" || nr.kind == "relgeo.free_point") {
      // A base point is required if referenced.
      // (We do not force all points in the graph to be solved.)
    } else if (nr.kind == "relgeo.circle") {
      uint32_t center = 0;
      if (get_u32_attr_node(nid, "center", center)) required_points.insert(center);
      uint32_t through = 0;
      if (get_u32_attr_node(nid, "through", through)) required_points.insert(through);
    } else if (nr.kind == "relgeo.arc3") {
      uint32_t p0 = 0, p1 = 0, p2 = 0;
      if (get_u32_attr_node(nid, "p0", p0)) required_points.insert(p0);
      if (get_u32_attr_node(nid, "p1", p1)) required_points.insert(p1);
      if (get_u32_attr_node(nid, "p2", p2)) required_points.insert(p2);
    } else if (nr.kind == "relgeo.arcse") {
      uint32_t start = 0, end = 0, circle = 0;
      if (get_u32_attr_node(nid, "circle", circle)) {
        // circle will pull in its center/through via the circle node scan.
      }
      if (get_u32_attr_node(nid, "start", start)) required_points.insert(start);
      if (get_u32_attr_node(nid, "end", end)) required_points.insert(end);
    } else if (nr.kind == "relgeo.bezier") {
      uint32_t p0 = 0, c0 = 0, c1 = 0, p1 = 0;
      if (get_u32_attr_node(nid, "p0", p0)) required_points.insert(p0);
      if (get_u32_attr_node(nid, "c0", c0)) required_points.insert(c0);
      if (get_u32_attr_node(nid, "c1", c1)) required_points.insert(c1);
      if (get_u32_attr_node(nid, "p1", p1)) required_points.insert(p1);
    }
  }

  // Relations: require endpoints.
  for (const auto& [eid, er] : edges) {
    if (er.kind == "relgeo.coincident") {
      uint32_t a = 0, b = 0;
      if (get_u32_attr_edge(eid, "a", a)) required_points.insert(a);
      if (get_u32_attr_edge(eid, "b", b)) required_points.insert(b);
    } else if (er.kind == "relgeo.incident") {
      uint32_t p = 0;
      if (get_u32_attr_edge(eid, "a", p)) required_points.insert(p);
    } else if (er.kind == "relgeo.point_on_circle") {
      uint32_t p = 0;
      if (get_u32_attr_edge(eid, "a", p)) required_points.insert(p);
    } else if (er.kind == "relgeo.tangent") {
      // line endpoints will be pulled in by line nodes; circle pulls by circle nodes.
    } else if (er.kind == "relgeo.fixed_radius") {
      // radius constraint will operate on circle's points.
    } else if (er.kind == "relgeo.arc_angle") {
      // arc constraint will operate on arc endpoints/circle.
    }
  }

  // Seed known points from anchors and from literal points.
  std::unordered_map<uint32_t, RelVec2> points = in.anchors;

  // Fixed points: explicit anchors and literal pt() nodes are treated as fixed reference.
  std::unordered_set<uint32_t> fixed;
  fixed.reserve(points.size() + 64);
  for (const auto& [id, _] : in.anchors) fixed.insert(id);

  for (const auto& [nid, nr] : nodes) {
    if (nr.kind != "relgeo.point") continue;
    if (points.find(nid) != points.end()) continue; // anchor overrides

    double x = 0.0, y = 0.0;
    if (get_f64_attr_node(nid, "x", x) && get_f64_attr_node(nid, "y", y)) {
      points[nid] = RelVec2{static_cast<float>(x), static_cast<float>(y)};
      fixed.insert(nid);
    }
  }

  // Non-movable derived points (recomputed from their inputs each pass).
  std::unordered_set<uint32_t> derived;
  derived.reserve(128);
  for (const auto& [nid, nr] : nodes) {
    if (nr.kind == "relgeo.lerp" || nr.kind == "relgeo.offset" || nr.kind == "relgeo.lerp_ratio" ||
        nr.kind == "relgeo.llint") {
      derived.insert(nid);
    }
  }

  auto is_movable = [&](uint32_t pid) -> bool {
    if (fixed.find(pid) != fixed.end()) return false;
    if (derived.find(pid) != derived.end()) return false;
    return true;
  };

  auto set_point_blend = [&](uint32_t pid, RelVec2 target, float alpha) {
    auto it = points.find(pid);
    if (it == points.end()) {
      points[pid] = target;
      return;
    }
    it->second = add(mul(it->second, 1.0f - alpha), mul(target, alpha));
  };

  auto translate_point = [&](uint32_t pid, RelVec2 delta, float alpha) {
    auto it = points.find(pid);
    if (it == points.end()) {
      points[pid] = mul(delta, alpha);
      return;
    }
    it->second = add(it->second, mul(delta, alpha));
  };

  // Seed free points so constraints can rotate/translate them.
  for (uint32_t p : required_points) {
    if (points.find(p) != points.end()) continue;
    auto itn = nodes.find(p);
    if (itn == nodes.end()) continue;
    if (itn->second.kind == "relgeo.free_point") {
      points[p] = default_free_point_seed(p);
    }
  }

  // Precompute ratio nodes.
  std::unordered_map<uint32_t, float> ratios;
  for (const auto& [nid, nr] : nodes) {
    if (nr.kind != "relgeo.ratio") continue;
    uint32_t n = 0, d = 0;
    if (!get_u32_attr_node(nid, "n", n) || !get_u32_attr_node(nid, "d", d) || d == 0u) continue;
    ratios[nid] = static_cast<float>(static_cast<double>(n) / static_cast<double>(d));
  }

  // Iterative constraint projector: recompute derived points, then enforce relation constraints.
  // This is not a full symbolic solver; it is meant to produce a stable numeric embedding given anchors.
  const float alpha = 0.65f; // damping
  for (uint32_t pass = 0; pass < std::max(1u, in.max_passes); ++pass) {
    // Cache line endpoints and (if solvable) directions.
    struct LineDef {
      uint32_t a_id = 0;
      uint32_t b_id = 0;
      RelVec2 a{};
      RelVec2 b{};
      RelVec2 dir{};
      float length = 0.0f;
      bool ok = false;
    };
    std::unordered_map<uint32_t, LineDef> lines;
    lines.reserve(64);

    struct CircleDef {
      uint32_t center_id = 0;
      uint32_t through_id = 0; // 0 if constant radius
      float r = 0.0f;
      RelVec2 center{};
      bool ok = false;
    };

    std::unordered_map<uint32_t, CircleDef> circles;
    circles.reserve(64);

    // Recompute derived points first (order-independent; they will update over passes as inputs move).
    for (const auto& [nid, nr] : nodes) {
      if (nr.kind == "relgeo.lerp") {
        uint32_t a = 0, b = 0;
        double u = 0.0;
        if (!get_u32_attr_node(nid, "a", a) || !get_u32_attr_node(nid, "b", b) || !get_f64_attr_node(nid, "u", u)) continue;
        auto ia = points.find(a);
        auto ib = points.find(b);
        if (ia == points.end() || ib == points.end()) continue;
        const float uf = static_cast<float>(u);
        points[nid] = add(mul(ia->second, 1.0f - uf), mul(ib->second, uf));
      } else if (nr.kind == "relgeo.offset") {
        uint32_t base = 0;
        double dx = 0.0, dy = 0.0;
        if (!get_u32_attr_node(nid, "base", base) || !get_f64_attr_node(nid, "dx", dx) || !get_f64_attr_node(nid, "dy", dy)) continue;
        auto ib = points.find(base);
        if (ib == points.end()) continue;
        points[nid] = add(ib->second, RelVec2{static_cast<float>(dx), static_cast<float>(dy)});
      } else if (nr.kind == "relgeo.lerp_ratio") {
        uint32_t a = 0, b = 0, u = 0;
        if (!get_u32_attr_node(nid, "a", a) || !get_u32_attr_node(nid, "b", b) || !get_u32_attr_node(nid, "u", u)) continue;
        auto ia = points.find(a);
        auto ib = points.find(b);
        auto iu = ratios.find(u);
        if (ia == points.end() || ib == points.end() || iu == ratios.end()) continue;
        const float uf = iu->second;
        points[nid] = add(mul(ia->second, 1.0f - uf), mul(ib->second, uf));
      }
    }

    for (const auto& [nid, nr] : nodes) {
      if (nr.kind != "relgeo.line") continue;
      uint32_t a_id = 0, b_id = 0;
      if (!get_u32_attr_node(nid, "a", a_id) || !get_u32_attr_node(nid, "b", b_id)) continue;
      auto ia = points.find(a_id);
      auto ib = points.find(b_id);
      if (ia == points.end() || ib == points.end()) continue;
      LineDef ld;
      ld.a_id = a_id;
      ld.b_id = b_id;
      ld.a = ia->second;
      ld.b = ib->second;
      const RelVec2 d = sub(ld.b, ld.a);
      ld.length = len(d);
      ld.ok = normalize(d, in.eps, ld.dir);
      lines[nid] = ld;
    }

    // Cache circle centers and radii.
    for (const auto& [nid, nr] : nodes) {
      if (nr.kind != "relgeo.circle") continue;
      uint32_t center_id = 0;
      if (!get_u32_attr_node(nid, "center", center_id)) continue;
      auto ic = points.find(center_id);
      if (ic == points.end()) continue;
      CircleDef cd;
      cd.center_id = center_id;
      cd.center = ic->second;

      // Constant radius.
      double rconst = 0.0;
      if (get_f64_attr_node(nid, "r", rconst)) {
        cd.r = static_cast<float>(rconst);
        cd.ok = cd.r > in.eps;
        circles[nid] = cd;
        continue;
      }

      // Through-point radius.
      uint32_t through_id = 0;
      if (get_u32_attr_node(nid, "through", through_id)) {
        auto itp = points.find(through_id);
        if (itp != points.end()) {
          cd.through_id = through_id;
          cd.r = dist(cd.center, itp->second);
          cd.ok = cd.r > in.eps;
          circles[nid] = cd;
        }
      }
    }

    // Update llint points using current line estimates.
    for (const auto& [nid, nr] : nodes) {
      if (nr.kind != "relgeo.llint") continue;
      uint32_t l0 = 0, l1 = 0;
      if (!get_u32_attr_node(nid, "l0", l0) || !get_u32_attr_node(nid, "l1", l1)) continue;
      auto i0 = lines.find(l0);
      auto i1 = lines.find(l1);
      if (i0 == lines.end() || i1 == lines.end()) continue;
      RelVec2 p;
      if (!line_line_intersection(i0->second.a, i0->second.b, i1->second.a, i1->second.b, in.eps, p)) continue;
      points[nid] = p;
    }

    float max_delta = 0.0f;

    // Enforce relation constraints.
    for (const auto& [eid, er] : edges) {
      if (er.kind == "relgeo.coincident") {
        uint32_t a = 0, b = 0;
        if (!get_u32_attr_edge(eid, "a", a) || !get_u32_attr_edge(eid, "b", b)) continue;
        auto ia = points.find(a);
        auto ib = points.find(b);
        if (ia == points.end() || ib == points.end()) continue;

        const bool ma = is_movable(a);
        const bool mb = is_movable(b);
        if (!ma && !mb) continue;
        const RelVec2 avg = mul(add(ia->second, ib->second), 0.5f);
        if (ma && mb) {
          const RelVec2 da = sub(avg, ia->second);
          const RelVec2 db = sub(avg, ib->second);
          max_delta = std::max(max_delta, len(da));
          max_delta = std::max(max_delta, len(db));
          set_point_blend(a, avg, alpha);
          set_point_blend(b, avg, alpha);
        } else if (ma) {
          const RelVec2 da = sub(ib->second, ia->second);
          max_delta = std::max(max_delta, len(da));
          set_point_blend(a, ib->second, alpha);
        } else {
          const RelVec2 db = sub(ia->second, ib->second);
          max_delta = std::max(max_delta, len(db));
          set_point_blend(b, ia->second, alpha);
        }
        continue;
      }

      if (er.kind == "relgeo.incident") {
        uint32_t p = 0, l = 0;
        if (!get_u32_attr_edge(eid, "a", p) || !get_u32_attr_edge(eid, "b", l)) continue;
        auto ip = points.find(p);
        if (ip == points.end()) continue;
        auto ln = lines.find(l);
        if (ln == lines.end() || !ln->second.ok) continue;

        const uint32_t a_id = ln->second.a_id;
        const uint32_t b_id = ln->second.b_id;
        const bool mp = is_movable(p);
        const bool ma = is_movable(a_id);
        const bool mb = is_movable(b_id);

        // If the point is movable, project it onto the current line.
        if (mp) {
          const RelVec2 a0 = ln->second.a;
          const RelVec2 d = sub(ln->second.b, ln->second.a);
          const float dd = dot(d, d);
          if (dd <= in.eps) continue;
          const float t = dot(sub(ip->second, a0), d) / dd;
          const RelVec2 proj = add(a0, mul(d, t));
          const RelVec2 delta = sub(proj, ip->second);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(p, proj, alpha);
          continue;
        }

        // Otherwise, try to rotate/translate the line to pass through fixed p by moving a movable endpoint.
        // Prefer keeping the other endpoint as pivot.
        if (ma || mb) {
          const RelVec2 ppos = ip->second;
          if (mb) {
            const RelVec2 pivot = points[a_id];
            RelVec2 dir_to_p;
            if (!normalize(sub(ppos, pivot), in.eps, dir_to_p)) continue;
            const float L = std::max(ln->second.length, 1.0f);
            const RelVec2 target = add(pivot, mul(dir_to_p, L));
            const RelVec2 delta = sub(target, points[b_id]);
            max_delta = std::max(max_delta, len(delta));
            set_point_blend(b_id, target, alpha);
          } else if (ma) {
            const RelVec2 pivot = points[b_id];
            RelVec2 dir_to_p;
            if (!normalize(sub(ppos, pivot), in.eps, dir_to_p)) continue;
            const float L = std::max(ln->second.length, 1.0f);
            const RelVec2 target = add(pivot, mul(dir_to_p, L));
            const RelVec2 delta = sub(target, points[a_id]);
            max_delta = std::max(max_delta, len(delta));
            set_point_blend(a_id, target, alpha);
          }
        }
        continue;
      }

      if (er.kind == "relgeo.parallel" || er.kind == "relgeo.perp") {
        uint32_t l0 = 0, l1 = 0;
        if (!get_u32_attr_edge(eid, "a", l0) || !get_u32_attr_edge(eid, "b", l1)) continue;
        auto i0 = lines.find(l0);
        auto i1 = lines.find(l1);
        if (i0 == lines.end() || i1 == lines.end()) continue;
        if (!i0->second.ok || !i1->second.ok) continue;

        auto enforce_dir = [&](const LineDef& line, RelVec2 target_dir) {
          uint32_t pivot_id = line.a_id;
          uint32_t move_id = line.b_id;
          if (!is_movable(move_id) && is_movable(pivot_id)) {
            std::swap(pivot_id, move_id);
          }
          if (!is_movable(move_id)) return;

          const RelVec2 pivot = points[pivot_id];
          const RelVec2 cur_end = points[move_id];
          const float L = std::max(len(sub(cur_end, pivot)), 1.0f);
          // Keep orientation stable.
          RelVec2 cur_dir;
          if (normalize(sub(cur_end, pivot), in.eps, cur_dir)) {
            if (dot(cur_dir, target_dir) < 0.0f) target_dir = mul(target_dir, -1.0f);
          }
          const RelVec2 target = add(pivot, mul(target_dir, L));
          const RelVec2 delta = sub(target, cur_end);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(move_id, target, alpha);
        };

        RelVec2 d0 = i0->second.dir;
        RelVec2 d1 = i1->second.dir;
        if (er.kind == "relgeo.parallel") {
          enforce_dir(i0->second, d1);
          enforce_dir(i1->second, d0);
        } else {
          enforce_dir(i0->second, perp(d1));
          enforce_dir(i1->second, perp(d0));
        }
        continue;
      }

      if (er.kind == "relgeo.point_on_circle") {
        uint32_t p = 0, c = 0;
        if (!get_u32_attr_edge(eid, "a", p) || !get_u32_attr_edge(eid, "b", c)) continue;
        auto ip = points.find(p);
        auto ic = circles.find(c);
        if (ip == points.end() || ic == circles.end() || !ic->second.ok) continue;

        const bool mp = is_movable(p);
        const bool mc = is_movable(ic->second.center_id);

        if (mp) {
          const RelVec2 target = project_point_to_circle(ip->second, ic->second.center, ic->second.r, in.eps);
          const RelVec2 delta = sub(target, ip->second);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(p, target, alpha);
        } else if (mc) {
          // Move center along current direction so that fixed p lies on the circle.
          RelVec2 dir;
          if (!normalize(sub(ic->second.center, ip->second), in.eps, dir)) dir = RelVec2{1.0f, 0.0f};
          const RelVec2 target_center = add(ip->second, mul(dir, ic->second.r));
          const RelVec2 delta = sub(target_center, ic->second.center);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(ic->second.center_id, target_center, alpha);
        }
        continue;
      }

      if (er.kind == "relgeo.fixed_radius") {
        uint32_t c = 0;
        double r = 0.0;
        if (!get_u32_attr_edge(eid, "a", c) || !get_f64_attr_edge(eid, "r", r)) continue;
        auto ic = circles.find(c);
        if (ic == circles.end()) continue;
        if (r <= 0.0) continue;

        // Only meaningful for through-point circles; constant-radius circles are already fixed.
        if (ic->second.through_id == 0) continue;
        const uint32_t center_id = ic->second.center_id;
        const uint32_t through_id = ic->second.through_id;

        auto ip_center = points.find(center_id);
        auto ip_through = points.find(through_id);
        if (ip_center == points.end() || ip_through == points.end()) continue;

        const bool mt = is_movable(through_id);
        const bool mc = is_movable(center_id);
        if (!mt && !mc) continue;

        const float rf = static_cast<float>(r);
        if (mt) {
          const RelVec2 target = project_point_to_circle(ip_through->second, ip_center->second, rf, in.eps);
          const RelVec2 delta = sub(target, ip_through->second);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(through_id, target, alpha);
        } else if (mc) {
          // Move center relative to fixed through point.
          RelVec2 dir;
          if (!normalize(sub(ip_center->second, ip_through->second), in.eps, dir)) dir = RelVec2{1.0f, 0.0f};
          const RelVec2 target_center = add(ip_through->second, mul(dir, rf));
          const RelVec2 delta = sub(target_center, ip_center->second);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(center_id, target_center, alpha);
        }
        continue;
      }

      if (er.kind == "relgeo.tangent") {
        uint32_t l = 0, c = 0;
        if (!get_u32_attr_edge(eid, "a", l) || !get_u32_attr_edge(eid, "b", c)) continue;
        auto il = lines.find(l);
        auto ic = circles.find(c);
        if (il == lines.end() || ic == circles.end()) continue;
        if (!il->second.ok || !ic->second.ok) continue;

        const uint32_t a_id = il->second.a_id;
        const uint32_t b_id = il->second.b_id;
        const uint32_t center_id = ic->second.center_id;
        const bool ma = is_movable(a_id);
        const bool mb = is_movable(b_id);
        const bool mc = is_movable(center_id);
        if (!ma && !mb && !mc) continue;

        RelVec2 n;
        const float sd = signed_distance_to_line(ic->second.center, il->second.a, il->second.b, in.eps, &n);
        if (len(n) <= in.eps) continue;

        // Want |sd| == r. Translate along n.
        const float want = (sd >= 0.0f ? ic->second.r : -ic->second.r);
        const float delta_s = want - sd;
        const RelVec2 delta = mul(n, delta_s);
        max_delta = std::max(max_delta, len(delta));

        if (ma || mb) {
          if (ma) translate_point(a_id, delta, alpha);
          if (mb) translate_point(b_id, delta, alpha);
        } else if (mc) {
          translate_point(center_id, mul(delta, -1.0f), alpha);
        }
        continue;
      }

      if (er.kind == "relgeo.arc_angle") {
        uint32_t arc = 0;
        double angle = 0.0;
        if (!get_u32_attr_edge(eid, "a", arc) || !get_f64_attr_edge(eid, "angle", angle)) continue;
        auto itn = nodes.find(arc);
        if (itn == nodes.end()) continue;
        if (itn->second.kind != "relgeo.arcse") continue;

        uint32_t circle = 0, start_id = 0, end_id = 0;
        if (!get_u32_attr_node(arc, "circle", circle) || !get_u32_attr_node(arc, "start", start_id) || !get_u32_attr_node(arc, "end", end_id)) continue;
        auto ic = circles.find(circle);
        if (ic == circles.end() || !ic->second.ok) continue;
        auto is = points.find(start_id);
        auto ie = points.find(end_id);
        if (is == points.end() || ie == points.end()) continue;

        bool ccw = true;
        if (auto it = nodes[arc].attrs.find("ccw"); it != nodes[arc].attrs.end()) {
          if (auto b = as_bool(it->second)) ccw = *b;
        }

        const bool ms = is_movable(start_id);
        const bool me = is_movable(end_id);
        if (!ms && !me) continue;

        const float a0 = std::atan2(is->second.y - ic->second.center.y, is->second.x - ic->second.center.x);
        const float da = static_cast<float>(angle);
        const float a1 = ccw ? (a0 + da) : (a0 - da);
        const RelVec2 target_end = add(ic->second.center, RelVec2{ic->second.r * std::cos(a1), ic->second.r * std::sin(a1)});

        if (me) {
          const RelVec2 delta = sub(target_end, ie->second);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(end_id, target_end, alpha);
        } else if (ms) {
          // Move start instead, keeping end fixed.
          const float a_end = std::atan2(ie->second.y - ic->second.center.y, ie->second.x - ic->second.center.x);
          const float a_start = ccw ? (a_end - da) : (a_end + da);
          const RelVec2 target_start = add(ic->second.center, RelVec2{ic->second.r * std::cos(a_start), ic->second.r * std::sin(a_start)});
          const RelVec2 delta = sub(target_start, is->second);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(start_id, target_start, alpha);
        }
        continue;
      }
    }

    if (max_delta <= in.eps) break;
  }

  // Final validation pass (when resolvable).
  for (const auto& [eid, er] : edges) {
    if (er.kind == "relgeo.coincident") {
      uint32_t a = 0, b = 0;
      if (!get_u32_attr_edge(eid, "a", a) || !get_u32_attr_edge(eid, "b", b)) continue;
      auto ia = points.find(a);
      auto ib = points.find(b);
      if (ia == points.end() || ib == points.end()) continue;
      const float dx = ia->second.x - ib->second.x;
      const float dy = ia->second.y - ib->second.y;
      const float d = std::sqrt(dx * dx + dy * dy);
      const float tol = 5.0f * in.eps;
      if (d > tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "constraint failed: edge " << eid << " relgeo.coincident(a=" << a << ", b=" << b << "): d=" << d
              << " > tol=" << tol;
          *out_error = oss.str();
        }
        return false;
      }
    }
    if (er.kind == "relgeo.incident") {
      uint32_t p = 0, l = 0;
      if (!get_u32_attr_edge(eid, "a", p) || !get_u32_attr_edge(eid, "b", l)) continue;
      auto ip = points.find(p);
      if (ip == points.end()) continue;
      auto ln = nodes.find(l);
      if (ln == nodes.end() || ln->second.kind != "relgeo.line") continue;
      uint32_t a = 0, b = 0;
      if (!get_u32_attr_node(l, "a", a) || !get_u32_attr_node(l, "b", b)) continue;
      auto ia = points.find(a);
      auto ib = points.find(b);
      if (ia == points.end() || ib == points.end()) continue;
      const float tol = 5.0f * in.eps;
      if (!point_on_line(ip->second, ia->second, ib->second, tol)) {
        if (out_error) {
          const RelVec2 ab = sub(ib->second, ia->second);
          const float denom = std::max(len(ab), in.eps);
          const float area2 = std::fabs(cross(sub(ip->second, ia->second), ab));
          const float dist_line = area2 / denom;
          std::ostringstream oss;
          oss << "constraint failed: edge " << eid << " relgeo.incident(p=" << p << ", line=" << l << "): dist="
              << dist_line << " > tol=" << tol;
          *out_error = oss.str();
        }
        return false;
      }
    }
    if (er.kind == "relgeo.parallel" || er.kind == "relgeo.perp") {
      uint32_t l0 = 0, l1 = 0;
      if (!get_u32_attr_edge(eid, "a", l0) || !get_u32_attr_edge(eid, "b", l1)) continue;
      auto n0 = nodes.find(l0);
      auto n1 = nodes.find(l1);
      if (n0 == nodes.end() || n1 == nodes.end()) continue;
      if (n0->second.kind != "relgeo.line" || n1->second.kind != "relgeo.line") continue;
      uint32_t a0 = 0, b0 = 0, a1 = 0, b1 = 0;
      if (!get_u32_attr_node(l0, "a", a0) || !get_u32_attr_node(l0, "b", b0)) continue;
      if (!get_u32_attr_node(l1, "a", a1) || !get_u32_attr_node(l1, "b", b1)) continue;
      auto ia0 = points.find(a0);
      auto ib0 = points.find(b0);
      auto ia1 = points.find(a1);
      auto ib1 = points.find(b1);
      if (ia0 == points.end() || ib0 == points.end() || ia1 == points.end() || ib1 == points.end()) continue;
      RelVec2 u0, u1;
      if (!normalize(sub(ib0->second, ia0->second), in.eps, u0) || !normalize(sub(ib1->second, ia1->second), in.eps, u1)) continue;
      const float c = std::fabs(cross(u0, u1));
      const float dp = std::fabs(dot(u0, u1));
      if (er.kind == "relgeo.parallel") {
        const float tol = 5.0f * in.eps;
        if (c > tol) {
          if (out_error) {
            std::ostringstream oss;
            oss << "constraint failed: edge " << eid << " relgeo.parallel(l0=" << l0 << ", l1=" << l1 << "): |cross|="
                << c << " > tol=" << tol;
            *out_error = oss.str();
          }
          return false;
        }
      } else {
        const float tol = 5.0f * in.eps;
        if (dp > tol) {
          if (out_error) {
            std::ostringstream oss;
            oss << "constraint failed: edge " << eid << " relgeo.perp(l0=" << l0 << ", l1=" << l1 << "): |dot|=" << dp
                << " > tol=" << tol;
            *out_error = oss.str();
          }
          return false;
        }
      }
    }

    if (er.kind == "relgeo.point_on_circle") {
      uint32_t p = 0, c = 0;
      if (!get_u32_attr_edge(eid, "a", p) || !get_u32_attr_edge(eid, "b", c)) continue;
      auto ip = points.find(p);
      auto cn = nodes.find(c);
      if (ip == points.end() || cn == nodes.end() || cn->second.kind != "relgeo.circle") continue;

      uint32_t center_id = 0;
      if (!get_u32_attr_node(c, "center", center_id)) continue;
      auto ic = points.find(center_id);
      if (ic == points.end()) continue;

      float r = 0.0f;
      double rconst = 0.0;
      if (get_f64_attr_node(c, "r", rconst)) {
        r = static_cast<float>(rconst);
      } else {
        uint32_t through = 0;
        if (!get_u32_attr_node(c, "through", through)) continue;
        auto itp = points.find(through);
        if (itp == points.end()) continue;
        r = dist(ic->second, itp->second);
      }
      const float tol = 10.0f * in.eps;
      const float d = dist(ip->second, ic->second);
      if (std::fabs(d - r) > tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "constraint failed: edge " << eid << " relgeo.point_on_circle(p=" << p << ", circle=" << c << "): |d-r|="
              << std::fabs(d - r) << " > tol=" << tol;
          *out_error = oss.str();
        }
        return false;
      }
    }

    if (er.kind == "relgeo.fixed_radius") {
      uint32_t c = 0;
      double r = 0.0;
      if (!get_u32_attr_edge(eid, "a", c) || !get_f64_attr_edge(eid, "r", r)) continue;
      auto cn = nodes.find(c);
      if (cn == nodes.end() || cn->second.kind != "relgeo.circle") continue;
      uint32_t center_id = 0, through_id = 0;
      if (!get_u32_attr_node(c, "center", center_id) || !get_u32_attr_node(c, "through", through_id)) continue;
      auto icp = points.find(center_id);
      auto itp = points.find(through_id);
      if (icp == points.end() || itp == points.end()) continue;
      const float tol = 10.0f * in.eps;
      const float d = dist(icp->second, itp->second);
      if (std::fabs(d - static_cast<float>(r)) > tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "constraint failed: edge " << eid << " relgeo.fixed_radius(circle=" << c << "): |d-r|="
              << std::fabs(d - static_cast<float>(r)) << " > tol=" << tol;
          *out_error = oss.str();
        }
        return false;
      }
    }

    if (er.kind == "relgeo.tangent") {
      uint32_t l = 0, c = 0;
      if (!get_u32_attr_edge(eid, "a", l) || !get_u32_attr_edge(eid, "b", c)) continue;
      auto ln = nodes.find(l);
      auto cn = nodes.find(c);
      if (ln == nodes.end() || cn == nodes.end()) continue;
      if (ln->second.kind != "relgeo.line" || cn->second.kind != "relgeo.circle") continue;

      uint32_t a_id = 0, b_id = 0;
      if (!get_u32_attr_node(l, "a", a_id) || !get_u32_attr_node(l, "b", b_id)) continue;
      uint32_t center_id = 0;
      if (!get_u32_attr_node(c, "center", center_id)) continue;
      auto ia = points.find(a_id);
      auto ib = points.find(b_id);
      auto icp = points.find(center_id);
      if (ia == points.end() || ib == points.end() || icp == points.end()) continue;

      float r = 0.0f;
      double rconst = 0.0;
      if (get_f64_attr_node(c, "r", rconst)) {
        r = static_cast<float>(rconst);
      } else {
        uint32_t through = 0;
        if (!get_u32_attr_node(c, "through", through)) continue;
        auto itp = points.find(through);
        if (itp == points.end()) continue;
        r = dist(icp->second, itp->second);
      }

      RelVec2 n;
      const float sd = signed_distance_to_line(icp->second, ia->second, ib->second, in.eps, &n);
      const float tol = 10.0f * in.eps;
      if (std::fabs(std::fabs(sd) - r) > tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "constraint failed: edge " << eid << " relgeo.tangent(line=" << l << ", circle=" << c << "): | |sd|-r |="
              << std::fabs(std::fabs(sd) - r) << " > tol=" << tol;
          *out_error = oss.str();
        }
        return false;
      }
    }

    if (er.kind == "relgeo.arc_angle") {
      uint32_t arc = 0;
      double angle = 0.0;
      if (!get_u32_attr_edge(eid, "a", arc) || !get_f64_attr_edge(eid, "angle", angle)) continue;
      auto an = nodes.find(arc);
      if (an == nodes.end() || an->second.kind != "relgeo.arcse") continue;
      uint32_t circle = 0, start_id = 0, end_id = 0;
      if (!get_u32_attr_node(arc, "circle", circle) || !get_u32_attr_node(arc, "start", start_id) || !get_u32_attr_node(arc, "end", end_id)) continue;
      auto cn = nodes.find(circle);
      if (cn == nodes.end() || cn->second.kind != "relgeo.circle") continue;
      uint32_t center_id = 0;
      if (!get_u32_attr_node(circle, "center", center_id)) continue;
      auto icp = points.find(center_id);
      auto is = points.find(start_id);
      auto ie = points.find(end_id);
      if (icp == points.end() || is == points.end() || ie == points.end()) continue;

      const float a0 = std::atan2(is->second.y - icp->second.y, is->second.x - icp->second.x);
      const float a1 = std::atan2(ie->second.y - icp->second.y, ie->second.x - icp->second.x);
      float da = a1 - a0;
      // Wrap to [-pi,pi] for comparison.
      while (da > 3.141592653589793f) da -= 2.0f * 3.141592653589793f;
      while (da < -3.141592653589793f) da += 2.0f * 3.141592653589793f;
      const float tol = 20.0f * in.eps;
      if (std::fabs(std::fabs(da) - static_cast<float>(angle)) > tol) {
        if (out_error) {
          std::ostringstream oss;
          oss << "constraint failed: edge " << eid << " relgeo.arc_angle(arc=" << arc << "): | |da|-angle |="
              << std::fabs(std::fabs(da) - static_cast<float>(angle)) << " > tol=" << tol;
          *out_error = oss.str();
        }
        return false;
      }
    }
  }

  // Check required points.
  std::vector<uint32_t> missing;
  missing.reserve(16);
  for (uint32_t p : required_points) {
    if (points.find(p) == points.end()) missing.push_back(p);
  }

  if (!missing.empty()) {
    if (out_error) {
      std::ostringstream oss;
      oss << "underconstrained relgeo graph: missing " << missing.size() << " required point(s); e.g. ";
      const size_t show = std::min<size_t>(missing.size(), 8);
      for (size_t i = 0; i < show; ++i) {
        const uint32_t id = missing[i];
        auto itn = nodes.find(id);
        oss << id;
        if (itn != nodes.end() && !itn->second.kind.empty()) oss << "(" << itn->second.kind << ")";
        if (i + 1 < show) oss << ", ";
      }
      *out_error = oss.str();
    }
    return false;
  }

  out.points = std::move(points);
  return true;
}

bool relgeo_solve_points_from_pure_ir(std::string_view src,
                                     const RelGeoSolveInputs& in,
                                     RelGeoSolveOutput& out,
                                     std::string* out_error) {
  GraphIrProgram program;
  GraphIrError perr;
  if (!graph_ir_parse(src, program, &perr)) {
    if (out_error) {
      std::ostringstream oss;
      oss << "parse error at " << perr.at.line << ":" << perr.at.col << ": " << perr.message;
      *out_error = oss.str();
    }
    return false;
  }

  GraphEditBuilder builder;
  GraphIrContext ctx;
  ctx.edits = &builder;

  GraphIrError eerr;
  const GraphIrOperatorSet ops = make_relgeo_ir_ops_pure();
  if (!graph_ir_eval(program, ops, ctx, &eerr)) {
    if (out_error) {
      std::ostringstream oss;
      oss << "eval error at " << eerr.at.line << ":" << eerr.at.col << ": " << eerr.message;
      *out_error = oss.str();
    }
    return false;
  }

  // Merge anchors-by-name into the anchor map.
  RelGeoSolveInputs merged = in;
  for (const auto& [name, value] : in.anchors_by_name) {
    auto it = ctx.symbols.find(name);
    if (it == ctx.symbols.end()) {
      if (out_error) *out_error = "anchor name not found in IR symbols: " + name;
      return false;
    }
    auto id = as_u32(it->second);
    if (!id) {
      if (out_error) *out_error = "anchor symbol is not a u32 id: " + name;
      return false;
    }
    merged.anchors[static_cast<uint32_t>(*id)] = value;
  }
  merged.anchors_by_name.clear();

  std::vector<nodus::tensors::GraphEdit> edits;
  edits.assign(builder.edits().begin(), builder.edits().end());
  return relgeo_solve_points_from_pure_edits(edits, merged, out, out_error);
}

bool relgeo_program_from_pure_ir_solved(std::string_view src,
                                       const RelGeoSolveInputs& in,
                                       RelProgram& out_program,
                                       std::string* out_error) {
  // Evaluate pure IR to get edit log + symbol table (for anchors_by_name).
  GraphIrProgram program;
  GraphIrError perr;
  if (!graph_ir_parse(src, program, &perr)) {
    if (out_error) {
      std::ostringstream oss;
      oss << "parse error at " << perr.at.line << ":" << perr.at.col << ": " << perr.message;
      *out_error = oss.str();
    }
    return false;
  }

  GraphEditBuilder builder;
  GraphIrContext ctx;
  ctx.edits = &builder;

  GraphIrError eerr;
  const GraphIrOperatorSet ops = make_relgeo_ir_ops_pure();
  if (!graph_ir_eval(program, ops, ctx, &eerr)) {
    if (out_error) {
      std::ostringstream oss;
      oss << "eval error at " << eerr.at.line << ":" << eerr.at.col << ": " << eerr.message;
      *out_error = oss.str();
    }
    return false;
  }

  std::vector<nodus::tensors::GraphEdit> edits;
  edits.assign(builder.edits().begin(), builder.edits().end());

  RelGeoSolveInputs merged = in;
  for (const auto& [name, value] : in.anchors_by_name) {
    auto it = ctx.symbols.find(name);
    if (it == ctx.symbols.end()) {
      if (out_error) *out_error = "anchor name not found in IR symbols: " + name;
      return false;
    }
    auto id = as_u32(it->second);
    if (!id) {
      if (out_error) *out_error = "anchor symbol is not a u32 id: " + name;
      return false;
    }
    merged.anchors[static_cast<uint32_t>(*id)] = value;
  }
  merged.anchors_by_name.clear();

  RelGeoSolveOutput solved;
  if (!relgeo_solve_points_from_pure_edits(edits, merged, solved, out_error)) return false;

  // Decode contour nodes and build a fixed RelProgram.
  struct ContourRec {
    uint32_t node_id = 0;
    bool closed = true;
    uint32_t winding_u32 = 0;
    std::vector<uint32_t> verts;
  };

  std::unordered_map<uint32_t, std::string> node_kind;
  std::unordered_map<uint32_t, std::unordered_map<std::string, GraphIrValue>> node_attrs;
  std::vector<ContourRec> contours;
  contours.reserve(16);

  for (const auto& e : edits) {
    if (e.kind == GraphEditKind::AddNode) {
      node_kind[e.a] = e.key;
      continue;
    }
    if (e.kind == GraphEditKind::SetAttr && graph_id_kind(e.a) == GraphObjectKind::Node) {
      node_attrs[e.a][e.key] = e.value;
      continue;
    }
  }

  for (const auto& [nid, kind] : node_kind) {
    if (kind != "relgeo.contour") continue;
    ContourRec cr;
    cr.node_id = nid;
    auto it = node_attrs.find(nid);
    if (it != node_attrs.end()) {
      if (auto b = it->second.find("closed"); b != it->second.end()) {
        if (const auto* pb = std::get_if<bool>(&b->second)) cr.closed = *pb;
        else if (auto d = as_f64(b->second)) cr.closed = (*d >= 0.5);
      }
      if (auto w = it->second.find("winding"); w != it->second.end()) {
        if (auto u = as_u32(w->second)) cr.winding_u32 = *u;
      }
      for (uint32_t i = 0;; ++i) {
        const std::string k = "v" + std::to_string(i);
        auto vit = it->second.find(k);
        if (vit == it->second.end()) break;
        auto u = as_u32(vit->second);
        if (!u) break;
        cr.verts.push_back(*u);
      }
    }
    if (cr.verts.size() >= 2) contours.push_back(std::move(cr));
  }

  if (contours.empty()) {
    if (out_error) *out_error = "pure IR solved but produced no contours";
    return false;
  }

  // Build point node id -> RelPointId mapping for all points referenced by contours.
  std::unordered_map<uint32_t, RelPointId> pid;
  pid.reserve(128);
  out_program = RelProgram{};

  auto ensure_point = [&](uint32_t point_node_id) -> bool {
    if (pid.find(point_node_id) != pid.end()) return true;
    auto itp = solved.points.find(point_node_id);
    if (itp == solved.points.end()) {
      if (out_error) *out_error = "missing solved point for node id " + std::to_string(point_node_id);
      return false;
    }
    RelPointId rid = out_program.add_point(RelPointFixed{itp->second.x, itp->second.y});
    pid[point_node_id] = rid;
    return true;
  };

  for (const auto& c : contours) {
    for (uint32_t v : c.verts) {
      if (!ensure_point(v)) return false;
    }
  }

  for (const auto& c : contours) {
    std::vector<RelPointId> verts;
    verts.reserve(c.verts.size());
    for (uint32_t v : c.verts) verts.push_back(pid[v]);
    RelContourWinding w = RelContourWinding::Unknown;
    if (c.winding_u32 == 1u) w = RelContourWinding::CCW;
    if (c.winding_u32 == 2u) w = RelContourWinding::CW;
    out_program.add_contour(std::move(verts), c.closed, w);
  }

  return true;
}

} // namespace nodus::tensors::kpath
