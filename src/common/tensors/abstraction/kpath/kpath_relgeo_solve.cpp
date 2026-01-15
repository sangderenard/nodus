#include "common/tensors/abstraction/kpath/kpath_relgeo_solve.h"

#include "common/tensors/abstraction/graph_sparse.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo_ir.h"
#include "common/tensors/abstraction/in_memory_backend.h"

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

struct P2 final {
  float x = 0.0f;
  float y = 0.0f;
};

static std::optional<AbstractTensor> make_point_tensor(P2 p) {
  TensorDesc desc;
  desc.dtype = TensorDType::F32;
  desc.layout = TensorLayout::Dense;
  desc.shape.dims = {1u, 2u};
  AbstractTensor t = AbstractTensor::create(desc, &in_memory_backend_singleton());
  if (!t.valid()) return std::nullopt;
  auto* backend = dynamic_cast<InMemoryBackend*>(t.backend());
  if (!backend) return std::nullopt;
  void* raw = nullptr;
  size_t bytes = 0;
  if (!backend->map(t.handle(), &raw, &bytes)) return std::nullopt;
  if (bytes < 2u * sizeof(float)) {
    backend->unmap(t.handle());
    return std::nullopt;
  }
  float* dst = static_cast<float*>(raw);
  dst[0] = p.x;
  dst[1] = p.y;
  backend->unmap(t.handle());
  return t;
}

static bool read_point_tensor(const AbstractTensor& t, P2& out) {
  auto* backend = dynamic_cast<InMemoryBackend*>(t.backend());
  if (!backend) return false;
  const TensorDesc& desc = t.desc();
  if (desc.dtype != TensorDType::F32 || desc.layout != TensorLayout::Dense ||
      desc.shape.dims.size() != 2 || desc.shape.dims[0] == 0 || desc.shape.dims[1] != 2u) {
    return false;
  }
  void* raw = nullptr;
  size_t bytes = 0;
  if (!backend->map(t.handle(), &raw, &bytes)) return false;
  if (bytes < 2u * sizeof(float)) {
    backend->unmap(t.handle());
    return false;
  }
  float* src = static_cast<float*>(raw);
  out = P2{src[0], src[1]};
  backend->unmap(t.handle());
  return true;
}

static bool clone_point_tensor(const RelTensorPoint& src, RelTensorPoint& out) {
  P2 p{};
  if (!read_point_tensor(src, p)) return false;
  auto t = make_point_tensor(p);
  if (!t) return false;
  out = std::move(*t);
  return true;
}

static bool get_point(const std::unordered_map<uint32_t, RelTensorPoint>& points, uint32_t id, P2& out) {
  auto it = points.find(id);
  if (it == points.end()) return false;
  return read_point_tensor(it->second, out);
}

static bool set_point(std::unordered_map<uint32_t, RelTensorPoint>& points, uint32_t id, P2 p) {
  auto t = make_point_tensor(p);
  if (!t) return false;
  points[id] = std::move(*t);
  return true;
}

static P2 sub(P2 a, P2 b) {
  return P2{a.x - b.x, a.y - b.y};
}

static P2 add(P2 a, P2 b) {
  return P2{a.x + b.x, a.y + b.y};
}

static P2 mul(P2 a, float s) {
  return P2{a.x * s, a.y * s};
}

static float dot(P2 a, P2 b) {
  return a.x * b.x + a.y * b.y;
}

static float cross(P2 a, P2 b) {
  return a.x * b.y - a.y * b.x;
}

static float len(P2 a) {
  return std::sqrt(dot(a, a));
}

static float dist(P2 a, P2 b) {
  return len(sub(a, b));
}

static bool line_line_intersection(P2 p0, P2 p1, P2 q0, P2 q1, float eps, P2& out) {
  const P2 r = sub(p1, p0);
  const P2 s = sub(q1, q0);
  const float denom = cross(r, s);
  if (std::fabs(denom) <= eps) return false;

  const P2 qp = sub(q0, p0);
  const float t = cross(qp, s) / denom;
  out = add(p0, mul(r, t));
  return true;
}

static bool point_on_line(P2 p, P2 a, P2 b, float eps) {
  const P2 ab = sub(b, a);
  const float ab_len = len(ab);
  if (ab_len <= eps) return false;

  // Distance from p to infinite line through a->b.
  const float area2 = std::fabs(cross(sub(p, a), ab));
  const float dist = area2 / ab_len;
  return dist <= eps;
}

static P2 perp(P2 a) {
  return P2{-a.y, a.x};
}

static bool normalize(P2 v, float eps, P2& out) {
  const float l = len(v);
  if (l <= eps) return false;
  out = P2{v.x / l, v.y / l};
  return true;
}

static P2 project_point_to_line(P2 p, P2 a, P2 b, float eps) {
  const P2 ab = sub(b, a);
  const float dd = dot(ab, ab);
  if (dd <= eps) return a;
  const float t = dot(sub(p, a), ab) / dd;
  return add(a, mul(ab, t));
}

static P2 project_point_to_circle(P2 p, P2 center, float r, float eps) {
  P2 dir;
  if (!normalize(sub(p, center), eps, dir)) {
    dir = P2{1.0f, 0.0f};
  }
  return add(center, mul(dir, r));
}

static float signed_distance_to_line(P2 p, P2 a, P2 b, float eps, P2* out_unit_normal = nullptr) {
  P2 dir;
  if (!normalize(sub(b, a), eps, dir)) {
    if (out_unit_normal) *out_unit_normal = P2{0.0f, 0.0f};
    return 0.0f;
  }
  P2 n = perp(dir);
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

static P2 default_free_point_seed(uint32_t id) {
  // Deterministic non-degenerate seed so projection constraints have something to work with.
  const uint64_t h = mix_u64(static_cast<uint64_t>(id) * 0x1000003dull);
  const double a = (static_cast<double>(h & 0xFFFFFFFFull) / static_cast<double>(0xFFFFFFFFull)) * (2.0 * 3.141592653589793);
  const double r = 1.0 + 0.5 * (static_cast<double>((h >> 32) & 0xFFFFFFFFull) / static_cast<double>(0xFFFFFFFFull));
  return P2{static_cast<float>(r * std::cos(a)), static_cast<float>(r * std::sin(a))};
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
    } else if (er.kind == "relgeo.perp_at") {
      uint32_t v = 0, a = 0, b = 0;
      if (get_u32_attr_edge(eid, "v", v)) required_points.insert(v);
      if (get_u32_attr_edge(eid, "a", a)) required_points.insert(a);
      if (get_u32_attr_edge(eid, "b", b)) required_points.insert(b);
    } else if (er.kind == "relgeo.tangent") {
      // line endpoints will be pulled in by line nodes; circle pulls by circle nodes.
    } else if (er.kind == "relgeo.fixed_radius") {
      // radius constraint will operate on circle's points.
    } else if (er.kind == "relgeo.arc_angle") {
      // arc constraint will operate on arc endpoints/circle.
    }
  }

  // Seed known points from anchors and from literal points.
  std::unordered_map<uint32_t, RelTensorPoint> points;
  points.reserve(in.anchors.size());
  for (const auto& [id, t] : in.anchors) {
    P2 p{};
    if (!read_point_tensor(t, p)) continue;
    (void)set_point(points, id, p);
  }

  // Fixed points: explicit anchors and literal pt() nodes are treated as fixed reference.
  std::unordered_set<uint32_t> fixed;
  fixed.reserve(points.size() + 64);
  for (const auto& [id, _] : in.anchors) fixed.insert(id);

  for (const auto& [nid, nr] : nodes) {
    if (nr.kind != "relgeo.point") continue;
    if (points.find(nid) != points.end()) continue; // anchor overrides

    double x = 0.0, y = 0.0;
    if (get_f64_attr_node(nid, "x", x) && get_f64_attr_node(nid, "y", y)) {
      if (!set_point(points, nid, P2{static_cast<float>(x), static_cast<float>(y)})) {
        if (out_error) *out_error = "relgeo.solve failed to allocate point tensor for fixed point";
        return false;
      }
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

  auto set_point_blend = [&](uint32_t pid, P2 target, float alpha) {
    P2 cur{};
    if (!get_point(points, pid, cur)) {
      (void)set_point(points, pid, target);
      return;
    }
    (void)set_point(points, pid, add(mul(cur, 1.0f - alpha), mul(target, alpha)));
  };

  auto translate_point = [&](uint32_t pid, P2 delta, float alpha) {
    P2 cur{};
    if (!get_point(points, pid, cur)) {
      (void)set_point(points, pid, mul(delta, alpha));
      return;
    }
    (void)set_point(points, pid, add(cur, mul(delta, alpha)));
  };

  // Seed free points so constraints can rotate/translate them.
  for (uint32_t p : required_points) {
    if (points.find(p) != points.end()) continue;
    auto itn = nodes.find(p);
    if (itn == nodes.end()) continue;
    if (itn->second.kind == "relgeo.free_point") {
      (void)set_point(points, p, default_free_point_seed(p));
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
      P2 a{};
      P2 b{};
      P2 dir{};
      float length = 0.0f;
      bool ok = false;
    };
    std::unordered_map<uint32_t, LineDef> lines;
    lines.reserve(64);

    struct CircleDef {
      uint32_t center_id = 0;
      uint32_t through_id = 0; // 0 if constant radius
      float r = 0.0f;
      P2 center{};
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
        P2 pa{}, pb{};
        if (!get_point(points, a, pa) || !get_point(points, b, pb)) continue;
        const float uf = static_cast<float>(u);
        (void)set_point(points, nid, add(mul(pa, 1.0f - uf), mul(pb, uf)));
      } else if (nr.kind == "relgeo.offset") {
        uint32_t base = 0;
        double dx = 0.0, dy = 0.0;
        if (!get_u32_attr_node(nid, "base", base) || !get_f64_attr_node(nid, "dx", dx) || !get_f64_attr_node(nid, "dy", dy)) continue;
        P2 pb{};
        if (!get_point(points, base, pb)) continue;
        (void)set_point(points, nid, add(pb, P2{static_cast<float>(dx), static_cast<float>(dy)}));
      } else if (nr.kind == "relgeo.lerp_ratio") {
        uint32_t a = 0, b = 0, u = 0;
        if (!get_u32_attr_node(nid, "a", a) || !get_u32_attr_node(nid, "b", b) || !get_u32_attr_node(nid, "u", u)) continue;
        auto iu = ratios.find(u);
        if (iu == ratios.end()) continue;
        P2 pa{}, pb{};
        if (!get_point(points, a, pa) || !get_point(points, b, pb)) continue;
        const float uf = iu->second;
        (void)set_point(points, nid, add(mul(pa, 1.0f - uf), mul(pb, uf)));
      }
    }

    for (const auto& [nid, nr] : nodes) {
      if (nr.kind != "relgeo.line") continue;
      uint32_t a_id = 0, b_id = 0;
      if (!get_u32_attr_node(nid, "a", a_id) || !get_u32_attr_node(nid, "b", b_id)) continue;
      P2 pa{}, pb{};
      if (!get_point(points, a_id, pa) || !get_point(points, b_id, pb)) continue;
      LineDef ld;
      ld.a_id = a_id;
      ld.b_id = b_id;
      ld.a = pa;
      ld.b = pb;
      const P2 d = sub(ld.b, ld.a);
      ld.length = len(d);
      ld.ok = normalize(d, in.eps, ld.dir);
      lines[nid] = ld;
    }

    // Cache circle centers and radii.
    for (const auto& [nid, nr] : nodes) {
      if (nr.kind != "relgeo.circle") continue;
      uint32_t center_id = 0;
      if (!get_u32_attr_node(nid, "center", center_id)) continue;
      P2 center{};
      if (!get_point(points, center_id, center)) continue;
      CircleDef cd;
      cd.center_id = center_id;
      cd.center = center;

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
        P2 through{};
        if (get_point(points, through_id, through)) {
          cd.through_id = through_id;
          cd.r = dist(cd.center, through);
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
      P2 p;
      if (!line_line_intersection(i0->second.a, i0->second.b, i1->second.a, i1->second.b, in.eps, p)) continue;
      (void)set_point(points, nid, p);
    }

    float max_delta = 0.0f;

    // Enforce relation constraints.
    for (const auto& [eid, er] : edges) {
      if (er.kind == "relgeo.coincident") {
        uint32_t a = 0, b = 0;
        if (!get_u32_attr_edge(eid, "a", a) || !get_u32_attr_edge(eid, "b", b)) continue;
        P2 pa{}, pb{};
        if (!get_point(points, a, pa) || !get_point(points, b, pb)) continue;

        const bool ma = is_movable(a);
        const bool mb = is_movable(b);
        if (!ma && !mb) continue;
        const P2 avg = mul(add(pa, pb), 0.5f);
        if (ma && mb) {
          const P2 da = sub(avg, pa);
          const P2 db = sub(avg, pb);
          max_delta = std::max(max_delta, len(da));
          max_delta = std::max(max_delta, len(db));
          set_point_blend(a, avg, alpha);
          set_point_blend(b, avg, alpha);
        } else if (ma) {
          const P2 da = sub(pb, pa);
          max_delta = std::max(max_delta, len(da));
          set_point_blend(a, pb, alpha);
        } else {
          const P2 db = sub(pa, pb);
          max_delta = std::max(max_delta, len(db));
          set_point_blend(b, pa, alpha);
        }
        continue;
      }

      if (er.kind == "relgeo.incident") {
        uint32_t p = 0, l = 0;
        if (!get_u32_attr_edge(eid, "a", p) || !get_u32_attr_edge(eid, "b", l)) continue;
        P2 ppos{};
        if (!get_point(points, p, ppos)) continue;
        auto ln = lines.find(l);
        if (ln == lines.end() || !ln->second.ok) continue;

        const uint32_t a_id = ln->second.a_id;
        const uint32_t b_id = ln->second.b_id;
        const bool mp = is_movable(p);
        const bool ma = is_movable(a_id);
        const bool mb = is_movable(b_id);

        // If the point is movable, project it onto the current line.
        if (mp) {
          const P2 a0 = ln->second.a;
          const P2 d = sub(ln->second.b, ln->second.a);
          const float dd = dot(d, d);
          if (dd <= in.eps) continue;
          const float t = dot(sub(ppos, a0), d) / dd;
          const P2 proj = add(a0, mul(d, t));
          const P2 delta = sub(proj, ppos);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(p, proj, alpha);
          continue;
        }

        // Otherwise, try to rotate/translate the line to pass through fixed p by moving a movable endpoint.
        // Prefer keeping the other endpoint as pivot.
        if (ma || mb) {
          if (mb) {
            P2 pivot{};
            if (!get_point(points, a_id, pivot)) continue;
            P2 dir_to_p;
            if (!normalize(sub(ppos, pivot), in.eps, dir_to_p)) continue;
            const float L = std::max(ln->second.length, 1.0f);
            const P2 target = add(pivot, mul(dir_to_p, L));
            P2 cur_b{};
            if (!get_point(points, b_id, cur_b)) continue;
            const P2 delta = sub(target, cur_b);
            max_delta = std::max(max_delta, len(delta));
            set_point_blend(b_id, target, alpha);
          } else if (ma) {
            P2 pivot{};
            if (!get_point(points, b_id, pivot)) continue;
            P2 dir_to_p;
            if (!normalize(sub(ppos, pivot), in.eps, dir_to_p)) continue;
            const float L = std::max(ln->second.length, 1.0f);
            const P2 target = add(pivot, mul(dir_to_p, L));
            P2 cur_a{};
            if (!get_point(points, a_id, cur_a)) continue;
            const P2 delta = sub(target, cur_a);
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

        auto enforce_dir = [&](const LineDef& line, P2 target_dir) {
          uint32_t pivot_id = line.a_id;
          uint32_t move_id = line.b_id;
          if (!is_movable(move_id) && is_movable(pivot_id)) {
            std::swap(pivot_id, move_id);
          }
          if (!is_movable(move_id)) return;

          P2 pivot{}, cur_end{};
          if (!get_point(points, pivot_id, pivot) || !get_point(points, move_id, cur_end)) return;
          const float L = std::max(len(sub(cur_end, pivot)), 1.0f);
          // Keep orientation stable.
          P2 cur_dir;
          if (normalize(sub(cur_end, pivot), in.eps, cur_dir)) {
            if (dot(cur_dir, target_dir) < 0.0f) target_dir = mul(target_dir, -1.0f);
          }
          const P2 target = add(pivot, mul(target_dir, L));
          const P2 delta = sub(target, cur_end);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(move_id, target, alpha);
        };

        P2 d0 = i0->second.dir;
        P2 d1 = i1->second.dir;
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
        auto ic = circles.find(c);
        if (ic == circles.end() || !ic->second.ok) continue;
        P2 ppos{};
        if (!get_point(points, p, ppos)) continue;

        const bool mp = is_movable(p);
        const bool mc = is_movable(ic->second.center_id);

        if (mp) {
          const P2 target = project_point_to_circle(ppos, ic->second.center, ic->second.r, in.eps);
          const P2 delta = sub(target, ppos);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(p, target, alpha);
        } else if (mc) {
          // Move center along current direction so that fixed p lies on the circle.
          P2 dir;
          if (!normalize(sub(ic->second.center, ppos), in.eps, dir)) dir = P2{1.0f, 0.0f};
          const P2 target_center = add(ppos, mul(dir, ic->second.r));
          const P2 delta = sub(target_center, ic->second.center);
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

        P2 pc{}, pt{};
        if (!get_point(points, center_id, pc) || !get_point(points, through_id, pt)) continue;

        const bool mt = is_movable(through_id);
        const bool mc = is_movable(center_id);
        if (!mt && !mc) continue;

        const float rf = static_cast<float>(r);
        if (mt) {
          const P2 target = project_point_to_circle(pt, pc, rf, in.eps);
          const P2 delta = sub(target, pt);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(through_id, target, alpha);
        } else if (mc) {
          // Move center relative to fixed through point.
          P2 dir;
          if (!normalize(sub(pc, pt), in.eps, dir)) dir = P2{1.0f, 0.0f};
          const P2 target_center = add(pt, mul(dir, rf));
          const P2 delta = sub(target_center, pc);
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

        P2 n;
        const float sd = signed_distance_to_line(ic->second.center, il->second.a, il->second.b, in.eps, &n);
        if (len(n) <= in.eps) continue;

        // Want |sd| == r. Translate along n.
        const float want = (sd >= 0.0f ? ic->second.r : -ic->second.r);
        const float delta_s = want - sd;
        const P2 delta = mul(n, delta_s);
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
        P2 ps{}, pe{};
        if (!get_point(points, start_id, ps) || !get_point(points, end_id, pe)) continue;

        bool ccw = true;
        if (auto it = nodes[arc].attrs.find("ccw"); it != nodes[arc].attrs.end()) {
          if (auto b = as_bool(it->second)) ccw = *b;
        }

        const bool ms = is_movable(start_id);
        const bool me = is_movable(end_id);
        if (!ms && !me) continue;

        const float a0 = std::atan2(ps.y - ic->second.center.y, ps.x - ic->second.center.x);
        const float da = static_cast<float>(angle);
        const float a1 = ccw ? (a0 + da) : (a0 - da);
        const P2 target_end = add(ic->second.center, P2{ic->second.r * std::cos(a1), ic->second.r * std::sin(a1)});

        if (me) {
          const P2 delta = sub(target_end, pe);
          max_delta = std::max(max_delta, len(delta));
          set_point_blend(end_id, target_end, alpha);
        } else if (ms) {
          // Move start instead, keeping end fixed.
          const float a_end = std::atan2(pe.y - ic->second.center.y, pe.x - ic->second.center.x);
          const float a_start = ccw ? (a_end - da) : (a_end + da);
          const P2 target_start = add(ic->second.center, P2{ic->second.r * std::cos(a_start), ic->second.r * std::sin(a_start)});
          const P2 delta = sub(target_start, ps);
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
      P2 pa{}, pb{};
      if (!get_point(points, a, pa) || !get_point(points, b, pb)) continue;
      const float dx = pa.x - pb.x;
      const float dy = pa.y - pb.y;
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
      P2 pp{};
      if (!get_point(points, p, pp)) continue;
      auto ln = nodes.find(l);
      if (ln == nodes.end() || ln->second.kind != "relgeo.line") continue;
      uint32_t a = 0, b = 0;
      if (!get_u32_attr_node(l, "a", a) || !get_u32_attr_node(l, "b", b)) continue;
      P2 pa{}, pb{};
      if (!get_point(points, a, pa) || !get_point(points, b, pb)) continue;
      const float tol = 5.0f * in.eps;
      if (!point_on_line(pp, pa, pb, tol)) {
        if (out_error) {
          const P2 ab = sub(pb, pa);
          const float denom = std::max(len(ab), in.eps);
          const float area2 = std::fabs(cross(sub(pp, pa), ab));
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
      P2 pa0{}, pb0{}, pa1{}, pb1{};
      if (!get_point(points, a0, pa0) || !get_point(points, b0, pb0) ||
          !get_point(points, a1, pa1) || !get_point(points, b1, pb1)) continue;
      P2 u0, u1;
      if (!normalize(sub(pb0, pa0), in.eps, u0) || !normalize(sub(pb1, pa1), in.eps, u1)) continue;
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
      auto cn = nodes.find(c);
      if (cn == nodes.end() || cn->second.kind != "relgeo.circle") continue;
      P2 pp{};
      if (!get_point(points, p, pp)) continue;

      uint32_t center_id = 0;
      if (!get_u32_attr_node(c, "center", center_id)) continue;
      P2 pc{};
      if (!get_point(points, center_id, pc)) continue;

      float r = 0.0f;
      double rconst = 0.0;
      if (get_f64_attr_node(c, "r", rconst)) {
        r = static_cast<float>(rconst);
      } else {
        uint32_t through = 0;
        if (!get_u32_attr_node(c, "through", through)) continue;
        P2 pt{};
        if (!get_point(points, through, pt)) continue;
        r = dist(pc, pt);
      }
      const float tol = 10.0f * in.eps;
      const float d = dist(pp, pc);
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
      P2 pc{}, pt{};
      if (!get_point(points, center_id, pc) || !get_point(points, through_id, pt)) continue;
      const float tol = 10.0f * in.eps;
      const float d = dist(pc, pt);
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
      P2 pa{}, pb{}, pc{};
      if (!get_point(points, a_id, pa) || !get_point(points, b_id, pb) || !get_point(points, center_id, pc)) continue;

      float r = 0.0f;
      double rconst = 0.0;
      if (get_f64_attr_node(c, "r", rconst)) {
        r = static_cast<float>(rconst);
      } else {
        uint32_t through = 0;
        if (!get_u32_attr_node(c, "through", through)) continue;
        P2 pt{};
        if (!get_point(points, through, pt)) continue;
        r = dist(pc, pt);
      }

      P2 n;
      const float sd = signed_distance_to_line(pc, pa, pb, in.eps, &n);
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
      P2 pc{}, ps{}, pe{};
      if (!get_point(points, center_id, pc) || !get_point(points, start_id, ps) || !get_point(points, end_id, pe)) continue;

      const float a0 = std::atan2(ps.y - pc.y, ps.x - pc.x);
      const float a1 = std::atan2(pe.y - pc.y, pe.x - pc.x);
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
  RelGeoSolveInputs merged;
  merged.eps = in.eps;
  merged.max_passes = in.max_passes;
  for (const auto& [id, value] : in.anchors) {
    RelTensorPoint t;
    if (!clone_point_tensor(value, t)) {
      if (out_error) *out_error = "failed to clone anchor tensor";
      return false;
    }
    merged.anchors.emplace(id, std::move(t));
  }
  for (const auto& [name, value] : in.anchors_by_name) {
    RelTensorPoint t;
    if (!clone_point_tensor(value, t)) {
      if (out_error) *out_error = "failed to clone anchor-by-name tensor";
      return false;
    }
    merged.anchors_by_name.emplace(name, std::move(t));
  }

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
    RelTensorPoint t;
    if (!clone_point_tensor(value, t)) {
      if (out_error) *out_error = "failed to clone anchor-by-name tensor";
      return false;
    }
    merged.anchors[static_cast<uint32_t>(*id)] = std::move(t);
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

  RelGeoSolveInputs merged;
  merged.eps = in.eps;
  merged.max_passes = in.max_passes;
  for (const auto& [id, value] : in.anchors) {
    RelTensorPoint t;
    if (!clone_point_tensor(value, t)) {
      if (out_error) *out_error = "failed to clone anchor tensor";
      return false;
    }
    merged.anchors.emplace(id, std::move(t));
  }
  for (const auto& [name, value] : in.anchors_by_name) {
    RelTensorPoint t;
    if (!clone_point_tensor(value, t)) {
      if (out_error) *out_error = "failed to clone anchor-by-name tensor";
      return false;
    }
    merged.anchors_by_name.emplace(name, std::move(t));
  }

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
    RelTensorPoint t;
    if (!clone_point_tensor(value, t)) {
      if (out_error) *out_error = "failed to clone anchor-by-name tensor";
      return false;
    }
    merged.anchors[static_cast<uint32_t>(*id)] = std::move(t);
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
    P2 p{};
    if (!read_point_tensor(itp->second, p)) {
      if (out_error) *out_error = "failed to read solved point tensor for node id " + std::to_string(point_node_id);
      return false;
    }
    RelPointId rid = out_program.add_point(RelPointFixed{p.x, p.y});
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
