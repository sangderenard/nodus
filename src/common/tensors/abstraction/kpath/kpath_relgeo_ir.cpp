#include "common/tensors/abstraction/kpath/kpath_relgeo_ir.h"

#include "common/tensors/abstraction/graph_sparse.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace nodus::tensors::kpath {

namespace {

static std::optional<double> as_f64(const GraphIrValue& v) {
  if (const auto* p = std::get_if<double>(&v)) return *p;
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

static std::optional<std::string> as_string(const GraphIrValue& v) {
  if (const auto* p = std::get_if<std::string>(&v)) return *p;
  return std::nullopt;
}

static std::optional<bool> as_bool(const GraphIrValue& v) {
  if (const auto* p = std::get_if<bool>(&v)) return *p;
  if (const auto* d = std::get_if<double>(&v)) return *d >= 0.5;
  return std::nullopt;
}

static std::string to_lower(std::string_view s) {
  std::string result(s);
  std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return result;
}

static std::optional<bool> direction_from_string(std::string_view s) {
  const std::string lowered = to_lower(s);
  if (lowered == "ccw" || lowered == "counterclockwise" || lowered == "counter_clockwise") return true;
  if (lowered == "cw" || lowered == "clockwise") return false;
  if (lowered == "forward" || lowered == "fwd" || lowered == "front") return true;
  if (lowered == "reverse" || lowered == "rev" || lowered == "backward" || lowered == "back") return false;
  return std::nullopt;
}

static std::optional<bool> parse_tangent_direction(const GraphIrValue& v) {
  if (const auto* p = std::get_if<bool>(&v)) return *p;
  if (const auto* d = std::get_if<double>(&v)) return *d >= 0.5;
  if (const auto* s = std::get_if<std::string>(&v)) return direction_from_string(*s);
  return std::nullopt;
}

static std::optional<RelPick> parse_pick(std::string_view s) {
  if (s == "HigherY" || s == "higher_y" || s == "hy") return RelPick::HigherY;
  if (s == "LowerY" || s == "lower_y" || s == "ly") return RelPick::LowerY;
  if (s == "HigherX" || s == "higher_x" || s == "hx") return RelPick::HigherX;
  if (s == "LowerX" || s == "lower_x" || s == "lx") return RelPick::LowerX;
  return std::nullopt;
}

static std::optional<RelContourWinding> parse_contour_winding_string(std::string_view s) {
  const std::string lowered = to_lower(s);
  if (lowered == "ccw" || lowered == "counterclockwise" || lowered == "counter_clockwise" || lowered == "outer") {
    return RelContourWinding::CCW;
  }
  if (lowered == "cw" || lowered == "clockwise" || lowered == "hole" || lowered == "inner") {
    return RelContourWinding::CW;
  }
  return std::nullopt;
}

static std::optional<RelContourWinding> parse_contour_winding_value(const GraphIrValue& v) {
  if (const auto* s = std::get_if<std::string>(&v)) return parse_contour_winding_string(*s);
  if (const auto* d = std::get_if<double>(&v)) {
    if (*d > 0.0) return RelContourWinding::CCW;
    if (*d < 0.0) return RelContourWinding::CW;
    return RelContourWinding::Unknown;
  }
  if (const auto* u = std::get_if<uint32_t>(&v)) {
    if (*u == 0u) return RelContourWinding::Unknown;
    if (*u == 1u) return RelContourWinding::CCW;
    if (*u == 2u) return RelContourWinding::CW;
  }
  return std::nullopt;
}

static void emit_node_point(GraphIrContext& ctx, uint32_t rel_point_id, double x, double y) {
  if (!ctx.edits) return;
  const uint32_t n = ctx.edits->add_node("relgeo.point");
  ctx.edits->set_attr(n, "relgeo.point_id", static_cast<uint32_t>(rel_point_id));
  ctx.edits->set_attr(n, "x", x);
  ctx.edits->set_attr(n, "y", y);
}

static void emit_node_lerp(GraphIrContext& ctx, uint32_t rel_point_id, uint32_t a, uint32_t b, double u) {
  if (!ctx.edits) return;
  const uint32_t n = ctx.edits->add_node("relgeo.lerp");
  ctx.edits->set_attr(n, "relgeo.point_id", static_cast<uint32_t>(rel_point_id));
  ctx.edits->set_attr(n, "a", static_cast<uint32_t>(a));
  ctx.edits->set_attr(n, "b", static_cast<uint32_t>(b));
  ctx.edits->set_attr(n, "u", u);
}

static void emit_node_offset(GraphIrContext& ctx, uint32_t rel_point_id, uint32_t base, double dx, double dy) {
  if (!ctx.edits) return;
  const uint32_t n = ctx.edits->add_node("relgeo.offset");
  ctx.edits->set_attr(n, "relgeo.point_id", static_cast<uint32_t>(rel_point_id));
  ctx.edits->set_attr(n, "base", static_cast<uint32_t>(base));
  ctx.edits->set_attr(n, "dx", dx);
  ctx.edits->set_attr(n, "dy", dy);
}

static void emit_node_ccint(GraphIrContext& ctx,
                            uint32_t rel_point_id,
                            uint32_t c0,
                            double r0,
                            uint32_t c1,
                            double r1,
                            std::string_view pick) {
  if (!ctx.edits) return;
  const uint32_t n = ctx.edits->add_node("relgeo.ccint");
  ctx.edits->set_attr(n, "relgeo.point_id", static_cast<uint32_t>(rel_point_id));
  ctx.edits->set_attr(n, "c0", static_cast<uint32_t>(c0));
  ctx.edits->set_attr(n, "r0", r0);
  ctx.edits->set_attr(n, "c1", static_cast<uint32_t>(c1));
  ctx.edits->set_attr(n, "r1", r1);
  if (auto p = parse_pick(pick)) {
    ctx.edits->set_attr(n, "pick", static_cast<uint32_t>(*p));
  }
}

static uint32_t emit_node_segment(GraphIrContext& ctx, uint32_t a, uint32_t b) {
  if (!ctx.edits) return 0;
  const uint32_t n = ctx.edits->add_node("relgeo.segment");
  ctx.edits->set_attr(n, "a", static_cast<uint32_t>(a));
  ctx.edits->set_attr(n, "b", static_cast<uint32_t>(b));
  return n;
}

static uint32_t emit_node_line(GraphIrContext& ctx, uint32_t a, uint32_t b) {
  if (!ctx.edits) return 0;
  const uint32_t n = ctx.edits->add_node("relgeo.line");
  ctx.edits->set_attr(n, "a", static_cast<uint32_t>(a));
  ctx.edits->set_attr(n, "b", static_cast<uint32_t>(b));
  return n;
}

static uint32_t emit_node_ray(GraphIrContext& ctx, uint32_t origin, uint32_t through) {
  if (!ctx.edits) return 0;
  const uint32_t n = ctx.edits->add_node("relgeo.ray");
  ctx.edits->set_attr(n, "origin", static_cast<uint32_t>(origin));
  ctx.edits->set_attr(n, "through", static_cast<uint32_t>(through));
  return n;
}

static uint32_t emit_node_angle(GraphIrContext& ctx, uint32_t a, uint32_t v, uint32_t b) {
  if (!ctx.edits) return 0;
  const uint32_t n = ctx.edits->add_node("relgeo.angle");
  ctx.edits->set_attr(n, "a", static_cast<uint32_t>(a));
  ctx.edits->set_attr(n, "v", static_cast<uint32_t>(v));
  ctx.edits->set_attr(n, "b", static_cast<uint32_t>(b));
  return n;
}

static uint32_t emit_node_llint(GraphIrContext& ctx, uint32_t l0, uint32_t l1) {
  if (!ctx.edits) return 0;
  const uint32_t n = ctx.edits->add_node("relgeo.llint");
  ctx.edits->set_attr(n, "l0", static_cast<uint32_t>(l0));
  ctx.edits->set_attr(n, "l1", static_cast<uint32_t>(l1));
  return n;
}

static uint32_t emit_node_circle(GraphIrContext& ctx, uint32_t center, const GraphIrValue& r_or_through) {
  if (!ctx.edits) return 0;
  const uint32_t n = ctx.edits->add_node("relgeo.circle");
  ctx.edits->set_attr(n, "center", static_cast<uint32_t>(center));
  if (auto r = as_f64(r_or_through)) {
    ctx.edits->set_attr(n, "r", *r);
  } else if (auto through = as_u32(r_or_through)) {
    ctx.edits->set_attr(n, "through", static_cast<uint32_t>(*through));
  }
  return n;
}

static uint32_t emit_node_arc_angles(GraphIrContext& ctx, uint32_t circle, double a0, double a1, bool ccw) {
  if (!ctx.edits) return 0;
  const uint32_t n = ctx.edits->add_node("relgeo.arc");
  ctx.edits->set_attr(n, "circle", static_cast<uint32_t>(circle));
  ctx.edits->set_attr(n, "a0", a0);
  ctx.edits->set_attr(n, "a1", a1);
  ctx.edits->set_attr(n, "ccw", ccw);
  return n;
}

static uint32_t emit_node_arc_endpoints(GraphIrContext& ctx, uint32_t circle, uint32_t start, uint32_t end, bool ccw) {
  if (!ctx.edits) return 0;
  const uint32_t n = ctx.edits->add_node("relgeo.arcse");
  ctx.edits->set_attr(n, "circle", static_cast<uint32_t>(circle));
  ctx.edits->set_attr(n, "start", static_cast<uint32_t>(start));
  ctx.edits->set_attr(n, "end", static_cast<uint32_t>(end));
  ctx.edits->set_attr(n, "ccw", ccw);
  return n;
}

static uint32_t emit_node_arc3(GraphIrContext& ctx, uint32_t p0, uint32_t p1, uint32_t p2) {
  if (!ctx.edits) return 0;
  const uint32_t n = ctx.edits->add_node("relgeo.arc3");
  ctx.edits->set_attr(n, "p0", static_cast<uint32_t>(p0));
  ctx.edits->set_attr(n, "p1", static_cast<uint32_t>(p1));
  ctx.edits->set_attr(n, "p2", static_cast<uint32_t>(p2));
  return n;
}

static uint32_t emit_node_bezier(GraphIrContext& ctx, uint32_t p0, uint32_t c0, uint32_t c1, uint32_t p1) {
  if (!ctx.edits) return 0;
  const uint32_t n = ctx.edits->add_node("relgeo.bezier");
  ctx.edits->set_attr(n, "p0", static_cast<uint32_t>(p0));
  ctx.edits->set_attr(n, "c0", static_cast<uint32_t>(c0));
  ctx.edits->set_attr(n, "c1", static_cast<uint32_t>(c1));
  ctx.edits->set_attr(n, "p1", static_cast<uint32_t>(p1));
  return n;
}

static void emit_edge_relation(GraphIrContext& ctx, std::string_view kind, uint32_t a, uint32_t b) {
  if (!ctx.edits) return;
  const uint32_t e = ctx.edits->add_edge(kind);
  ctx.edits->set_attr(e, "a", static_cast<uint32_t>(a));
  ctx.edits->set_attr(e, "b", static_cast<uint32_t>(b));
}

static void emit_edge_perp_at(GraphIrContext& ctx, uint32_t v, uint32_t a, uint32_t b) {
  if (!ctx.edits) return;
  const uint32_t e = ctx.edits->add_edge("relgeo.perp_at");
  ctx.edits->set_attr(e, "v", static_cast<uint32_t>(v));
  ctx.edits->set_attr(e, "a", static_cast<uint32_t>(a));
  ctx.edits->set_attr(e, "b", static_cast<uint32_t>(b));
}

static void emit_edge_fixed_radius(GraphIrContext& ctx, uint32_t circle, double r) {
  if (!ctx.edits) return;
  const uint32_t e = ctx.edits->add_edge("relgeo.fixed_radius");
  ctx.edits->set_attr(e, "a", static_cast<uint32_t>(circle));
  ctx.edits->set_attr(e, "r", r);
}

static void emit_edge_arc_angle(GraphIrContext& ctx, uint32_t arc, double angle) {
  if (!ctx.edits) return;
  const uint32_t e = ctx.edits->add_edge("relgeo.arc_angle");
  ctx.edits->set_attr(e, "a", static_cast<uint32_t>(arc));
  ctx.edits->set_attr(e, "angle", angle);
}

static uint32_t emit_node_ratio(GraphIrContext& ctx, uint32_t n, uint32_t d) {
  if (!ctx.edits) return 0;
  const uint32_t node_id = ctx.edits->add_node("relgeo.ratio");
  ctx.edits->set_attr(node_id, "n", static_cast<uint32_t>(n));
  ctx.edits->set_attr(node_id, "d", static_cast<uint32_t>(d));
  return node_id;
}

static uint32_t emit_node_lerp_ratio(GraphIrContext& ctx, uint32_t a, uint32_t b, uint32_t ratio) {
  if (!ctx.edits) return 0;
  const uint32_t node_id = ctx.edits->add_node("relgeo.lerp_ratio");
  ctx.edits->set_attr(node_id, "a", static_cast<uint32_t>(a));
  ctx.edits->set_attr(node_id, "b", static_cast<uint32_t>(b));
  ctx.edits->set_attr(node_id, "u", static_cast<uint32_t>(ratio));
  return node_id;
}

static void emit_node_contour(GraphIrContext& ctx,
                              const std::vector<uint32_t>& verts,
                              bool closed,
                              RelContourWinding winding) {
  if (!ctx.edits) return;
  const uint32_t n = ctx.edits->add_node("relgeo.contour");
  ctx.edits->set_attr(n, "closed", closed);
  if (winding == RelContourWinding::CCW) ctx.edits->set_attr(n, "winding", static_cast<uint32_t>(1u));
  if (winding == RelContourWinding::CW) ctx.edits->set_attr(n, "winding", static_cast<uint32_t>(2u));
  for (size_t i = 0; i < verts.size(); ++i) {
    ctx.edits->set_attr(n, "v" + std::to_string(i), static_cast<uint32_t>(verts[i]));
  }
}

static std::string segment_kind_to_string(RelParametricSegment::Kind kind) {
  switch (kind) {
    case RelParametricSegment::Kind::Line:
      return "line";
    case RelParametricSegment::Kind::Quadratic:
      return "quadratic";
    case RelParametricSegment::Kind::Cubic:
      return "cubic";
    case RelParametricSegment::Kind::SinWave:
      return "sin";
  }
  return "line";
}

static std::optional<RelParametricSegment::Kind> parse_segment_kind(std::string_view s) {
  const std::string lowered = to_lower(s);
  if (lowered == "line") return RelParametricSegment::Kind::Line;
  if (lowered == "quadratic") return RelParametricSegment::Kind::Quadratic;
  if (lowered == "cubic") return RelParametricSegment::Kind::Cubic;
  if (lowered == "sin" || lowered == "sinwave") return RelParametricSegment::Kind::SinWave;
  return std::nullopt;
}

static uint32_t emit_node_parametric(GraphIrContext& ctx,
                                     RelParametricSegment::Kind kind,
                                     uint32_t start,
                                     uint32_t end,
                                     double u,
                                     bool tangent_forward,
                                     std::optional<uint32_t> ctrl1 = std::nullopt,
                                     std::optional<uint32_t> ctrl2 = std::nullopt,
                                     std::optional<double> amplitude = std::nullopt,
                                     std::optional<double> cycles = std::nullopt,
                                     std::optional<double> phase = std::nullopt) {
  if (!ctx.edits) return 0;
  const uint32_t n = ctx.edits->add_node("relgeo.parametric");
  ctx.edits->set_attr(n, "segment_kind", segment_kind_to_string(kind));
  ctx.edits->set_attr(n, "start", start);
  ctx.edits->set_attr(n, "end", end);
  ctx.edits->set_attr(n, "u", u);
  ctx.edits->set_attr(n, "tangent_forward", tangent_forward);
  if (ctrl1) ctx.edits->set_attr(n, "ctrl1", static_cast<uint32_t>(*ctrl1));
  if (ctrl2) ctx.edits->set_attr(n, "ctrl2", static_cast<uint32_t>(*ctrl2));
  if (amplitude) ctx.edits->set_attr(n, "amplitude", *amplitude);
  if (cycles) ctx.edits->set_attr(n, "cycles", *cycles);
  if (phase) ctx.edits->set_attr(n, "phase", *phase);
  return n;
}

} // namespace

GraphIrOperatorSet make_relgeo_ir_ops() {
  GraphIrOperatorSet set;

  set.add(GraphIrOpSpec{"free", 0, 0, "free() -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>&, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "free(): ctx.edits is null";
              return false;
            }
            const uint32_t node_id = ctx.edits->add_node("relgeo.free_point");
            out = node_id;
            return true;
          });

  set.add(GraphIrOpSpec{"pt", 2, 2, "pt(x: f64, y: f64) -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "pt(): ctx.edits is null";
              return false;
            }
            auto x = as_f64(args[0]);
            auto y = as_f64(args[1]);
            if (!x || !y) {
              err = "pt(): expected (number, number)";
              return false;
            }
            const uint32_t node_id = ctx.edits->add_node("relgeo.point");
            ctx.edits->set_attr(node_id, "x", *x);
            ctx.edits->set_attr(node_id, "y", *y);
            out = node_id;
            return true;
          });

  set.add(GraphIrOpSpec{"lerp", 3, 3, "lerp(a: u32, b: u32, u: f64) -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "lerp(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            auto u = as_f64(args[2]);
            if (!a || !b || !u) {
              err = "lerp(): expected (u32,u32,number)";
              return false;
            }
            const uint32_t node_id = ctx.edits->add_node("relgeo.lerp");
            ctx.edits->set_attr(node_id, "a", static_cast<uint32_t>(*a));
            ctx.edits->set_attr(node_id, "b", static_cast<uint32_t>(*b));
            ctx.edits->set_attr(node_id, "u", *u);
            out = node_id;
            return true;
          });

  set.add(GraphIrOpSpec{"offset", 3, 3, "offset(base: u32, dx: f64, dy: f64) -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "offset(): ctx.edits is null";
              return false;
            }
            auto base = as_u32(args[0]);
            auto dx = as_f64(args[1]);
            auto dy = as_f64(args[2]);
            if (!base || !dx || !dy) {
              err = "offset(): expected (u32,number,number)";
              return false;
            }

            const uint32_t node_id = ctx.edits->add_node("relgeo.offset");
            ctx.edits->set_attr(node_id, "base", static_cast<uint32_t>(*base));
            ctx.edits->set_attr(node_id, "dx", *dx);
            ctx.edits->set_attr(node_id, "dy", *dy);
            out = node_id;
            return true;
          });

  set.add(GraphIrOpSpec{"ccint", 5, 5, "ccint(c0: u32, r0: f64, c1: u32, r1: f64, pick: string) -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "ccint(): ctx.edits is null";
              return false;
            }
            auto c0 = as_u32(args[0]);
            auto r0 = as_f64(args[1]);
            auto c1 = as_u32(args[2]);
            auto r1 = as_f64(args[3]);
            auto pick_s = as_string(args[4]);
            if (!c0 || !r0 || !c1 || !r1 || !pick_s) {
              err = "ccint(): expected (u32,number,u32,number,string)";
              return false;
            }
            auto pick = parse_pick(*pick_s);
            if (!pick) {
              err = "ccint(): invalid pick string";
              return false;
            }

            const uint32_t node_id = ctx.edits->add_node("relgeo.ccint");
            ctx.edits->set_attr(node_id, "c0", static_cast<uint32_t>(*c0));
            ctx.edits->set_attr(node_id, "r0", *r0);
            ctx.edits->set_attr(node_id, "c1", static_cast<uint32_t>(*c1));
            ctx.edits->set_attr(node_id, "r1", *r1);
            ctx.edits->set_attr(node_id, "pick", static_cast<uint32_t>(*pick));
            out = node_id;
            return true;
          });

  set.add(GraphIrOpSpec{"segment", 2, 2, "segment(a: u32, b: u32) -> u32 segment"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "segment(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            if (!a || !b) {
              err = "segment(): expected (u32,u32)";
              return false;
            }
            out = emit_node_segment(ctx, static_cast<uint32_t>(*a), static_cast<uint32_t>(*b));
            return true;
          });

  set.add(GraphIrOpSpec{"line", 2, 2, "line(a: u32, b: u32) -> u32 line"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "line(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            if (!a || !b) {
              err = "line(): expected (u32,u32)";
              return false;
            }
            out = emit_node_line(ctx, static_cast<uint32_t>(*a), static_cast<uint32_t>(*b));
            return true;
          });

  set.add(GraphIrOpSpec{"ray", 2, 2, "ray(origin: u32, through: u32) -> u32 ray"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "ray(): ctx.edits is null";
              return false;
            }
            auto origin = as_u32(args[0]);
            auto through = as_u32(args[1]);
            if (!origin || !through) {
              err = "ray(): expected (u32,u32)";
              return false;
            }
            out = emit_node_ray(ctx, static_cast<uint32_t>(*origin), static_cast<uint32_t>(*through));
            return true;
          });

  set.add(GraphIrOpSpec{"angle", 3, 3, "angle(a: u32, v: u32, b: u32) -> u32 angle"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "angle(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto v = as_u32(args[1]);
            auto b = as_u32(args[2]);
            if (!a || !v || !b) {
              err = "angle(): expected (u32,u32,u32)";
              return false;
            }
            out = emit_node_angle(ctx, static_cast<uint32_t>(*a), static_cast<uint32_t>(*v), static_cast<uint32_t>(*b));
            return true;
          });

  set.add(GraphIrOpSpec{"llint", 2, 2, "llint(l0: u32, l1: u32) -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "llint(): ctx.edits is null";
              return false;
            }
            auto l0 = as_u32(args[0]);
            auto l1 = as_u32(args[1]);
            if (!l0 || !l1) {
              err = "llint(): expected (u32,u32)";
              return false;
            }
            out = emit_node_llint(ctx, static_cast<uint32_t>(*l0), static_cast<uint32_t>(*l1));
            return true;
          });

  set.add(GraphIrOpSpec{"param_line", 3, 4, "param_line(start: u32, end: u32, u: f64[, dir: string|bool]) -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "param_line(): ctx.edits is null";
              return false;
            }
            auto start = as_u32(args[0]);
            auto end = as_u32(args[1]);
            auto u = as_f64(args[2]);
            if (!start || !end || !u) {
              err = "param_line(): expected (u32,u32,number[,string|bool])";
              return false;
            }
            bool tangent_forward = true;
            if (args.size() == 4) {
              auto dir = parse_tangent_direction(args[3]);
              if (!dir) {
                err = "param_line(): invalid direction argument";
                return false;
              }
              tangent_forward = *dir;
            }
            const uint32_t node_id =
                emit_node_parametric(ctx, RelParametricSegment::Kind::Line, static_cast<uint32_t>(*start),
                                     static_cast<uint32_t>(*end), *u, tangent_forward);
            out = node_id;
            return true;
          });

  set.add(GraphIrOpSpec{"param_quad", 4, 5, "param_quad(start: u32, ctrl: u32, end: u32, u: f64[, dir: string|bool]) -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "param_quad(): ctx.edits is null";
              return false;
            }
            auto start = as_u32(args[0]);
            auto ctrl = as_u32(args[1]);
            auto end = as_u32(args[2]);
            auto u = as_f64(args[3]);
            if (!start || !ctrl || !end || !u) {
              err = "param_quad(): expected (u32,u32,u32,number[,string|bool])";
              return false;
            }
            bool tangent_forward = true;
            if (args.size() == 5) {
              auto dir = parse_tangent_direction(args[4]);
              if (!dir) {
                err = "param_quad(): invalid direction argument";
                return false;
              }
              tangent_forward = *dir;
            }
            const uint32_t node_id =
                emit_node_parametric(ctx, RelParametricSegment::Kind::Quadratic, static_cast<uint32_t>(*start),
                                     static_cast<uint32_t>(*end), *u, tangent_forward, static_cast<uint32_t>(*ctrl));
            out = node_id;
            return true;
          });

  set.add(GraphIrOpSpec{"param_cubic", 5, 6, "param_cubic(start: u32, ctrl0: u32, ctrl1: u32, end: u32, u: f64[, dir: string|bool]) -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "param_cubic(): ctx.edits is null";
              return false;
            }
            auto start = as_u32(args[0]);
            auto ctrl0 = as_u32(args[1]);
            auto ctrl1 = as_u32(args[2]);
            auto end = as_u32(args[3]);
            auto u = as_f64(args[4]);
            if (!start || !ctrl0 || !ctrl1 || !end || !u) {
              err = "param_cubic(): expected (u32,u32,u32,u32,number[,string|bool])";
              return false;
            }
            bool tangent_forward = true;
            if (args.size() == 6) {
              auto dir = parse_tangent_direction(args[5]);
              if (!dir) {
                err = "param_cubic(): invalid direction argument";
                return false;
              }
              tangent_forward = *dir;
            }
            const uint32_t node_id =
                emit_node_parametric(ctx, RelParametricSegment::Kind::Cubic, static_cast<uint32_t>(*start),
                                     static_cast<uint32_t>(*end), *u, tangent_forward,
                                     static_cast<uint32_t>(*ctrl0), static_cast<uint32_t>(*ctrl1));
            out = node_id;
            return true;
          });

  set.add(GraphIrOpSpec{"param_sin", 6, 7, "param_sin(start: u32, end: u32, u: f64, amplitude: f64, cycles: f64, phase: f64[, dir: string|bool]) -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "param_sin(): ctx.edits is null";
              return false;
            }
            auto start = as_u32(args[0]);
            auto end = as_u32(args[1]);
            auto u = as_f64(args[2]);
            auto amplitude = as_f64(args[3]);
            auto cycles = as_f64(args[4]);
            auto phase = as_f64(args[5]);
            if (!start || !end || !u || !amplitude || !cycles || !phase) {
              err = "param_sin(): expected (u32,u32,number,number,number,number[,string|bool])";
              return false;
            }
            bool tangent_forward = true;
            if (args.size() == 7) {
              auto dir = parse_tangent_direction(args[6]);
              if (!dir) {
                err = "param_sin(): invalid direction argument";
                return false;
              }
              tangent_forward = *dir;
            }
            const uint32_t node_id =
                emit_node_parametric(ctx, RelParametricSegment::Kind::SinWave, static_cast<uint32_t>(*start),
                                     static_cast<uint32_t>(*end), *u, tangent_forward,
                                     {} /*ctrl1*/, {} /*ctrl2*/, *amplitude, *cycles, *phase);
            out = node_id;
            return true;
          });

  set.add(GraphIrOpSpec{"circle", 2, 2, "circle(center: u32, r: f64|u32) -> u32 circle"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "circle(): ctx.edits is null";
              return false;
            }
            auto center = as_u32(args[0]);
            if (!center) {
              err = "circle(): expected (u32, number|u32)";
              return false;
            }

            // Second arg may be numeric radius or a point id the circle passes through.
            if (!as_f64(args[1]) && !as_u32(args[1])) {
              err = "circle(): expected (u32, number|u32)";
              return false;
            }

            out = emit_node_circle(ctx, static_cast<uint32_t>(*center), args[1]);
            return true;
          });

  set.add(GraphIrOpSpec{"arc", 3, 4, "arc(circle: u32circle, a0: f64, a1: f64[, dir: string|bool]) -> u32 arc"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "arc(): ctx.edits is null";
              return false;
            }
            auto circle = as_u32(args[0]);
            auto a0 = as_f64(args[1]);
            auto a1 = as_f64(args[2]);
            if (!circle || !a0 || !a1) {
              err = "arc(): expected (u32, number, number[, string|bool])";
              return false;
            }
            bool ccw = true;
            if (args.size() == 4) {
              auto dir = parse_tangent_direction(args[3]);
              if (!dir) {
                err = "arc(): invalid direction argument";
                return false;
              }
              ccw = *dir;
            }
            out = emit_node_arc_angles(ctx, static_cast<uint32_t>(*circle), *a0, *a1, ccw);
            return true;
          });

  set.add(GraphIrOpSpec{"arcse", 3, 4, "arcse(circle: u32circle, start: u32, end: u32[, dir: string|bool]) -> u32 arc"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "arcse(): ctx.edits is null";
              return false;
            }
            auto circle = as_u32(args[0]);
            auto start = as_u32(args[1]);
            auto end = as_u32(args[2]);
            if (!circle || !start || !end) {
              err = "arcse(): expected (u32, u32, u32[, string|bool])";
              return false;
            }
            bool ccw = true;
            if (args.size() == 4) {
              auto dir = parse_tangent_direction(args[3]);
              if (!dir) {
                err = "arcse(): invalid direction argument";
                return false;
              }
              ccw = *dir;
            }
            out = emit_node_arc_endpoints(ctx, static_cast<uint32_t>(*circle), static_cast<uint32_t>(*start), static_cast<uint32_t>(*end), ccw);
            return true;
          });

  set.add(GraphIrOpSpec{"arc3", 3, 3, "arc3(p0: u32, p1: u32, p2: u32) -> u32 arc"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "arc3(): ctx.edits is null";
              return false;
            }
            auto p0 = as_u32(args[0]);
            auto p1 = as_u32(args[1]);
            auto p2 = as_u32(args[2]);
            if (!p0 || !p1 || !p2) {
              err = "arc3(): expected (u32,u32,u32)";
              return false;
            }
            out = emit_node_arc3(ctx, static_cast<uint32_t>(*p0), static_cast<uint32_t>(*p1), static_cast<uint32_t>(*p2));
            return true;
          });

  set.add(GraphIrOpSpec{"bezier", 4, 4, "bezier(p0: u32, c0: u32, c1: u32, p1: u32) -> u32 bezier"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "bezier(): ctx.edits is null";
              return false;
            }
            auto p0 = as_u32(args[0]);
            auto c0 = as_u32(args[1]);
            auto c1 = as_u32(args[2]);
            auto p1 = as_u32(args[3]);
            if (!p0 || !c0 || !c1 || !p1) {
              err = "bezier(): expected (u32,u32,u32,u32)";
              return false;
            }
            out = emit_node_bezier(ctx, static_cast<uint32_t>(*p0), static_cast<uint32_t>(*c0), static_cast<uint32_t>(*c1), static_cast<uint32_t>(*p1));
            return true;
          });

  set.add(GraphIrOpSpec{"contour",
                        2,
                        0xFFFFFFFFu,
                        "contour(p0: u32, p1: u32, ..., [\"open\"|\"closed\"], [\"ccw\"|\"cw\"|\"outer\"|\"inner\"]) -> void"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "contour(): ctx.edits is null";
              return false;
            }

            bool closed = true;
            RelContourWinding winding = RelContourWinding::Unknown;
            size_t n = args.size();

            // Allow up to two trailing strings for closure and winding (in any order).
            while (n >= 1) {
              auto last_s = as_string(args[n - 1]);
              if (!last_s) break;
              const std::string lowered = to_lower(*last_s);

              bool consumed = false;
              if (lowered == "open") {
                closed = false;
                consumed = true;
              } else if (lowered == "closed") {
                closed = true;
                consumed = true;
              } else if (auto w = parse_contour_winding_string(lowered)) {
                winding = *w;
                consumed = true;
              }

              if (!consumed) break;
              n -= 1;
            }

            if (n < 2) {
              err = "contour(): requires at least 2 points";
              return false;
            }

            std::vector<uint32_t> raw;
            raw.reserve(n);
            for (size_t i = 0; i < n; ++i) {
              auto pid = as_u32(args[i]);
              if (!pid) {
                err = "contour(): point args must be u32";
                return false;
              }
              raw.push_back(*pid);
            }

            emit_node_contour(ctx, raw, closed, winding);
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"parallel", 2, 2, "parallel(l0: u32, l1: u32) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "parallel(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            if (!a || !b) {
              err = "parallel(): expected (u32,u32)";
              return false;
            }
            emit_edge_relation(ctx, "relgeo.parallel", static_cast<uint32_t>(*a), static_cast<uint32_t>(*b));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"perp", 2, 2, "perp(l0: u32, l1: u32) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "perp(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            if (!a || !b) {
              err = "perp(): expected (u32,u32)";
              return false;
            }
            emit_edge_relation(ctx, "relgeo.perp", static_cast<uint32_t>(*a), static_cast<uint32_t>(*b));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"perp_at", 3, 3, "perp_at(v: u32, a: u32, b: u32) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "perp_at(): ctx.edits is null";
              return false;
            }
            auto v = as_u32(args[0]);
            auto a = as_u32(args[1]);
            auto b = as_u32(args[2]);
            if (!v || !a || !b) {
              err = "perp_at(): expected (u32,u32,u32)";
              return false;
            }
            emit_edge_perp_at(ctx, static_cast<uint32_t>(*v), static_cast<uint32_t>(*a), static_cast<uint32_t>(*b));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"perp_at", 3, 3, "perp_at(v: u32, a: u32, b: u32) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "perp_at(): ctx.edits is null";
              return false;
            }
            auto v = as_u32(args[0]);
            auto a = as_u32(args[1]);
            auto b = as_u32(args[2]);
            if (!v || !a || !b) {
              err = "perp_at(): expected (u32,u32,u32)";
              return false;
            }
            emit_edge_perp_at(ctx, static_cast<uint32_t>(*v), static_cast<uint32_t>(*a), static_cast<uint32_t>(*b));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"incident", 2, 2, "incident(p: u32, l: u32) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "incident(): ctx.edits is null";
              return false;
            }
            auto p = as_u32(args[0]);
            auto l = as_u32(args[1]);
            if (!p || !l) {
              err = "incident(): expected (u32,u32)";
              return false;
            }
            emit_edge_relation(ctx, "relgeo.incident", static_cast<uint32_t>(*p), static_cast<uint32_t>(*l));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"coincident", 2, 2, "coincident(p0: u32, p1: u32) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "coincident(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            if (!a || !b) {
              err = "coincident(): expected (u32,u32)";
              return false;
            }
            emit_edge_relation(ctx, "relgeo.coincident", static_cast<uint32_t>(*a), static_cast<uint32_t>(*b));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"point_on_circle", 2, 2, "point_on_circle(p: u32, circle: u32circle) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "point_on_circle(): ctx.edits is null";
              return false;
            }
            auto p = as_u32(args[0]);
            auto c = as_u32(args[1]);
            if (!p || !c) {
              err = "point_on_circle(): expected (u32,u32)";
              return false;
            }
            emit_edge_relation(ctx, "relgeo.point_on_circle", static_cast<uint32_t>(*p), static_cast<uint32_t>(*c));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"tangent", 2, 2, "tangent(line: u32line, circle: u32circle) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "tangent(): ctx.edits is null";
              return false;
            }
            auto l = as_u32(args[0]);
            auto c = as_u32(args[1]);
            if (!l || !c) {
              err = "tangent(): expected (u32,u32)";
              return false;
            }
            emit_edge_relation(ctx, "relgeo.tangent", static_cast<uint32_t>(*l), static_cast<uint32_t>(*c));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"fixed_radius", 2, 2, "fixed_radius(circle: u32circle, r: f64) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "fixed_radius(): ctx.edits is null";
              return false;
            }
            auto c = as_u32(args[0]);
            auto r = as_f64(args[1]);
            if (!c || !r) {
              err = "fixed_radius(): expected (u32, number)";
              return false;
            }
            emit_edge_fixed_radius(ctx, static_cast<uint32_t>(*c), *r);
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"arc_angle", 2, 2, "arc_angle(arc: u32arc, angle: f64) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "arc_angle(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto ang = as_f64(args[1]);
            if (!a || !ang) {
              err = "arc_angle(): expected (u32, number)";
              return false;
            }
            emit_edge_arc_angle(ctx, static_cast<uint32_t>(*a), *ang);
            out = std::monostate{};
            return true;
          });

  return set;
}

GraphIrOperatorSet make_relgeo_ir_ops_pure() {
  GraphIrOperatorSet set;

  // Allow the noun+relation vocabulary plus contouring.
  // Deliberately omit float-based operators (offset/ccint/param_*).

  set.add(GraphIrOpSpec{"free", 0, 0, "free() -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>&, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "free(): ctx.edits is null";
              return false;
            }
            const uint32_t node_id = ctx.edits->add_node("relgeo.free_point");
            out = node_id;
            return true;
          });

  set.add(GraphIrOpSpec{"pt", 2, 2, "pt(x: f64, y: f64) -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "pt(): ctx.edits is null";
              return false;
            }
            auto x = as_f64(args[0]);
            auto y = as_f64(args[1]);
            if (!x || !y) {
              err = "pt(): expected (number, number)";
              return false;
            }
            const uint32_t node_id = ctx.edits->add_node("relgeo.point");
            ctx.edits->set_attr(node_id, "x", *x);
            ctx.edits->set_attr(node_id, "y", *y);
            out = node_id;
            return true;
          });

  set.add(GraphIrOpSpec{"segment", 2, 2, "segment(a: u32, b: u32) -> u32 segment"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "segment(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            if (!a || !b) {
              err = "segment(): expected (u32,u32)";
              return false;
            }
            out = emit_node_segment(ctx, static_cast<uint32_t>(*a), static_cast<uint32_t>(*b));
            return true;
          });

  set.add(GraphIrOpSpec{"line", 2, 2, "line(a: u32, b: u32) -> u32 line"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "line(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            if (!a || !b) {
              err = "line(): expected (u32,u32)";
              return false;
            }
            out = emit_node_line(ctx, static_cast<uint32_t>(*a), static_cast<uint32_t>(*b));
            return true;
          });

  set.add(GraphIrOpSpec{"ray", 2, 2, "ray(origin: u32, through: u32) -> u32 ray"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "ray(): ctx.edits is null";
              return false;
            }
            auto origin = as_u32(args[0]);
            auto through = as_u32(args[1]);
            if (!origin || !through) {
              err = "ray(): expected (u32,u32)";
              return false;
            }
            out = emit_node_ray(ctx, static_cast<uint32_t>(*origin), static_cast<uint32_t>(*through));
            return true;
          });

  set.add(GraphIrOpSpec{"angle", 3, 3, "angle(a: u32, v: u32, b: u32) -> u32 angle"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "angle(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto v = as_u32(args[1]);
            auto b = as_u32(args[2]);
            if (!a || !v || !b) {
              err = "angle(): expected (u32,u32,u32)";
              return false;
            }
            out = emit_node_angle(ctx, static_cast<uint32_t>(*a), static_cast<uint32_t>(*v), static_cast<uint32_t>(*b));
            return true;
          });

  set.add(GraphIrOpSpec{"llint", 2, 2, "llint(l0: u32, l1: u32) -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "llint(): ctx.edits is null";
              return false;
            }
            auto l0 = as_u32(args[0]);
            auto l1 = as_u32(args[1]);
            if (!l0 || !l1) {
              err = "llint(): expected (u32,u32)";
              return false;
            }
            out = emit_node_llint(ctx, static_cast<uint32_t>(*l0), static_cast<uint32_t>(*l1));
            return true;
          });

  set.add(GraphIrOpSpec{"ratio", 2, 2, "ratio(n: u32, d: u32) -> u32 ratio"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "ratio(): ctx.edits is null";
              return false;
            }
            auto n = as_u32(args[0]);
            auto d = as_u32(args[1]);
            if (!n || !d || *d == 0u) {
              err = "ratio(): expected (u32,u32) with d!=0";
              return false;
            }
            out = emit_node_ratio(ctx, static_cast<uint32_t>(*n), static_cast<uint32_t>(*d));
            return true;
          });

  set.add(GraphIrOpSpec{"lerp_ratio", 3, 3, "lerp_ratio(a: u32, b: u32, u: u32ratio) -> u32 point"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "lerp_ratio(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            auto u = as_u32(args[2]);
            if (!a || !b || !u) {
              err = "lerp_ratio(): expected (u32,u32,u32)";
              return false;
            }
            out = emit_node_lerp_ratio(ctx, static_cast<uint32_t>(*a), static_cast<uint32_t>(*b), static_cast<uint32_t>(*u));
            return true;
          });

  set.add(GraphIrOpSpec{"circle", 2, 2, "circle(center: u32, through: u32) -> u32 circle"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "circle(): ctx.edits is null";
              return false;
            }
            auto center = as_u32(args[0]);
            auto through = as_u32(args[1]);
            if (!center || !through) {
              err = "circle(): expected (u32,u32)";
              return false;
            }
            out = emit_node_circle(ctx, static_cast<uint32_t>(*center), args[1]);
            return true;
          });

  set.add(GraphIrOpSpec{"arcse", 3, 4, "arcse(circle: u32circle, start: u32, end: u32[, dir: string|bool]) -> u32 arc"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "arcse(): ctx.edits is null";
              return false;
            }
            auto circle = as_u32(args[0]);
            auto start = as_u32(args[1]);
            auto end = as_u32(args[2]);
            if (!circle || !start || !end) {
              err = "arcse(): expected (u32,u32,u32[,string|bool])";
              return false;
            }
            bool ccw = true;
            if (args.size() == 4) {
              auto dir = parse_tangent_direction(args[3]);
              if (!dir) {
                err = "arcse(): invalid direction argument";
                return false;
              }
              ccw = *dir;
            }
            out = emit_node_arc_endpoints(ctx, static_cast<uint32_t>(*circle), static_cast<uint32_t>(*start), static_cast<uint32_t>(*end), ccw);
            return true;
          });

  set.add(GraphIrOpSpec{"arc3", 3, 3, "arc3(p0: u32, p1: u32, p2: u32) -> u32 arc"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "arc3(): ctx.edits is null";
              return false;
            }
            auto p0 = as_u32(args[0]);
            auto p1 = as_u32(args[1]);
            auto p2 = as_u32(args[2]);
            if (!p0 || !p1 || !p2) {
              err = "arc3(): expected (u32,u32,u32)";
              return false;
            }
            out = emit_node_arc3(ctx, static_cast<uint32_t>(*p0), static_cast<uint32_t>(*p1), static_cast<uint32_t>(*p2));
            return true;
          });

  set.add(GraphIrOpSpec{"bezier", 4, 4, "bezier(p0: u32, c0: u32, c1: u32, p1: u32) -> u32 bezier"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "bezier(): ctx.edits is null";
              return false;
            }
            auto p0 = as_u32(args[0]);
            auto c0 = as_u32(args[1]);
            auto c1 = as_u32(args[2]);
            auto p1 = as_u32(args[3]);
            if (!p0 || !c0 || !c1 || !p1) {
              err = "bezier(): expected (u32,u32,u32,u32)";
              return false;
            }
            out = emit_node_bezier(ctx, static_cast<uint32_t>(*p0), static_cast<uint32_t>(*c0), static_cast<uint32_t>(*c1), static_cast<uint32_t>(*p1));
            return true;
          });

  set.add(GraphIrOpSpec{"contour",
                        2,
                        0xFFFFFFFFu,
                        "contour(p0: u32, p1: u32, ..., [\"open\"|\"closed\"], [\"ccw\"|\"cw\"|\"outer\"|\"inner\"]) -> void"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "contour(): ctx.edits is null";
              return false;
            }

            bool closed = true;
            RelContourWinding winding = RelContourWinding::Unknown;
            size_t n = args.size();

            while (n >= 1) {
              auto last_s = as_string(args[n - 1]);
              if (!last_s) break;
              const std::string lowered = to_lower(*last_s);

              bool consumed = false;
              if (lowered == "open") {
                closed = false;
                consumed = true;
              } else if (lowered == "closed") {
                closed = true;
                consumed = true;
              } else if (auto w = parse_contour_winding_string(lowered)) {
                winding = *w;
                consumed = true;
              }

              if (!consumed) break;
              n -= 1;
            }

            if (n < 2) {
              err = "contour(): requires at least 2 points";
              return false;
            }

            std::vector<uint32_t> raw;
            raw.reserve(n);
            for (size_t i = 0; i < n; ++i) {
              auto pid = as_u32(args[i]);
              if (!pid) {
                err = "contour(): point args must be u32";
                return false;
              }
              raw.push_back(*pid);
            }

            emit_node_contour(ctx, raw, closed, winding);
            out = std::monostate{};
            return true;
          });

  // Relations as edges.
  set.add(GraphIrOpSpec{"parallel", 2, 2, "parallel(l0: u32, l1: u32) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "parallel(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            if (!a || !b) {
              err = "parallel(): expected (u32,u32)";
              return false;
            }
            emit_edge_relation(ctx, "relgeo.parallel", static_cast<uint32_t>(*a), static_cast<uint32_t>(*b));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"perp", 2, 2, "perp(l0: u32, l1: u32) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "perp(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            if (!a || !b) {
              err = "perp(): expected (u32,u32)";
              return false;
            }
            emit_edge_relation(ctx, "relgeo.perp", static_cast<uint32_t>(*a), static_cast<uint32_t>(*b));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"incident", 2, 2, "incident(p: u32, l: u32) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "incident(): ctx.edits is null";
              return false;
            }
            auto p = as_u32(args[0]);
            auto l = as_u32(args[1]);
            if (!p || !l) {
              err = "incident(): expected (u32,u32)";
              return false;
            }
            emit_edge_relation(ctx, "relgeo.incident", static_cast<uint32_t>(*p), static_cast<uint32_t>(*l));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"coincident", 2, 2, "coincident(p0: u32, p1: u32) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "coincident(): ctx.edits is null";
              return false;
            }
            auto a = as_u32(args[0]);
            auto b = as_u32(args[1]);
            if (!a || !b) {
              err = "coincident(): expected (u32,u32)";
              return false;
            }
            emit_edge_relation(ctx, "relgeo.coincident", static_cast<uint32_t>(*a), static_cast<uint32_t>(*b));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"point_on_circle", 2, 2, "point_on_circle(p: u32, circle: u32circle) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "point_on_circle(): ctx.edits is null";
              return false;
            }
            auto p = as_u32(args[0]);
            auto c = as_u32(args[1]);
            if (!p || !c) {
              err = "point_on_circle(): expected (u32,u32)";
              return false;
            }
            emit_edge_relation(ctx, "relgeo.point_on_circle", static_cast<uint32_t>(*p), static_cast<uint32_t>(*c));
            out = std::monostate{};
            return true;
          });

  set.add(GraphIrOpSpec{"tangent", 2, 2, "tangent(line: u32line, circle: u32circle) -> void edge"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "tangent(): ctx.edits is null";
              return false;
            }
            auto l = as_u32(args[0]);
            auto c = as_u32(args[1]);
            if (!l || !c) {
              err = "tangent(): expected (u32,u32)";
              return false;
            }
            emit_edge_relation(ctx, "relgeo.tangent", static_cast<uint32_t>(*l), static_cast<uint32_t>(*c));
            out = std::monostate{};
            return true;
          });

  return set;
}

bool relgeo_validate_pure_ir(std::string_view src, std::string* out_error) {
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

  GraphEditBuilder edits;
  GraphIrContext ctx;
  ctx.edits = &edits;

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

  // Additional structural checks could go here (e.g. ensure sufficient anchors are provided
  // for a constraint solver to obtain a numeric embedding).
  return true;
}

bool relgeo_edits_from_pure_ir(std::string_view src, std::vector<nodus::tensors::GraphEdit>& out_edits, std::string* out_error) {
  out_edits.clear();

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

  GraphEditBuilder edits;
  GraphIrContext ctx;
  ctx.edits = &edits;

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

  out_edits.assign(edits.edits().begin(), edits.edits().end());
  return true;
}

bool relgeo_program_from_ir(std::string_view src, RelProgram& out_program, std::string* out_error) {
  // Parse + run against a pure graph-edit operator set.
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

  GraphEditBuilder edits;
  GraphIrContext ctx;
  ctx.edits = &edits;

  GraphIrError eerr;
  const GraphIrOperatorSet ops = make_relgeo_ir_ops();
  if (!graph_ir_eval(program, ops, ctx, &eerr)) {
    if (out_error) {
      std::ostringstream oss;
      oss << "eval error at " << eerr.at.line << ":" << eerr.at.col << ": " << eerr.message;
      *out_error = oss.str();
    }
    return false;
  }

  // Interpret the resulting graph edits into a RelProgram (temporary bridge).
  // Source of truth is the graph; this just adapts into the existing RelProgram evaluator.
  struct NodeRec {
    std::string kind;
    std::unordered_map<std::string, GraphIrValue> attrs;
  };
  struct EdgeRec {
    std::string kind;
    std::unordered_map<std::string, GraphIrValue> attrs;
  };
  std::unordered_map<uint32_t, NodeRec> nodes;
  std::unordered_map<uint32_t, EdgeRec> edges;
  std::vector<uint32_t> point_nodes;
  std::vector<uint32_t> segment_nodes;
  std::vector<uint32_t> line_nodes;
  std::vector<uint32_t> ray_nodes;
  std::vector<uint32_t> angle_nodes;
  std::vector<uint32_t> circle_nodes;
  std::vector<uint32_t> arc_nodes;
  std::vector<uint32_t> bezier_nodes;
  std::vector<uint32_t> contour_nodes;
  std::vector<uint32_t> relation_edges;

  for (const auto& e : edits.edits()) {
    if (e.kind == GraphEditKind::AddNode) {
      nodes[e.a].kind = e.key;
      if (e.key.rfind("relgeo.", 0) == 0) {
        if (e.key == "relgeo.contour") contour_nodes.push_back(e.a);
        else if (e.key == "relgeo.segment") segment_nodes.push_back(e.a);
        else if (e.key == "relgeo.line") line_nodes.push_back(e.a);
        else if (e.key == "relgeo.ray") ray_nodes.push_back(e.a);
        else if (e.key == "relgeo.angle") angle_nodes.push_back(e.a);
        else if (e.key == "relgeo.circle") circle_nodes.push_back(e.a);
        else if (e.key == "relgeo.arc" || e.key == "relgeo.arc3" || e.key == "relgeo.arcse") arc_nodes.push_back(e.a);
        else if (e.key == "relgeo.bezier") bezier_nodes.push_back(e.a);
        else point_nodes.push_back(e.a);
      }
      continue;
    }
    if (e.kind == GraphEditKind::AddEdge) {
      edges[e.a].kind = e.key;
      if (e.key.rfind("relgeo.", 0) == 0) relation_edges.push_back(e.a);
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

  std::sort(point_nodes.begin(), point_nodes.end());
  std::sort(segment_nodes.begin(), segment_nodes.end());
  std::sort(line_nodes.begin(), line_nodes.end());
  std::sort(ray_nodes.begin(), ray_nodes.end());
  std::sort(angle_nodes.begin(), angle_nodes.end());
  std::sort(circle_nodes.begin(), circle_nodes.end());
  std::sort(arc_nodes.begin(), arc_nodes.end());
  std::sort(bezier_nodes.begin(), bezier_nodes.end());
  std::sort(contour_nodes.begin(), contour_nodes.end());
  std::sort(relation_edges.begin(), relation_edges.end());

  // Map graph node ids -> RelPointId (dense 1..N).
  out_program = RelProgram{};

  auto get_u32_attr = [&](uint32_t nid, const char* key, uint32_t& out_u) -> bool {
    auto itn = nodes.find(nid);
    if (itn == nodes.end()) return false;
    auto ita = itn->second.attrs.find(key);
    if (ita == itn->second.attrs.end()) return false;
    auto u = as_u32(ita->second);
    if (!u) return false;
    out_u = *u;
    return true;
  };
  auto get_f64_attr = [&](uint32_t nid, const char* key, double& out_d) -> bool {
    auto itn = nodes.find(nid);
    if (itn == nodes.end()) return false;
    auto ita = itn->second.attrs.find(key);
    if (ita == itn->second.attrs.end()) return false;
    auto d = as_f64(ita->second);
    if (!d) return false;
    out_d = *d;
    return true;
  };
  auto get_str_attr = [&](uint32_t nid, const char* key, std::string& out_s) -> bool {
    auto itn = nodes.find(nid);
    if (itn == nodes.end()) return false;
    auto ita = itn->second.attrs.find(key);
    if (ita == itn->second.attrs.end()) return false;
    auto s = as_string(ita->second);
    if (!s) return false;
    out_s = *s;
    return true;
  };

  auto get_bool_attr = [&](uint32_t nid, const char* key, bool& out_b) -> bool {
    auto itn = nodes.find(nid);
    if (itn == nodes.end()) return false;
    auto ita = itn->second.attrs.find(key);
    if (ita == itn->second.attrs.end()) return false;
    auto b = as_bool(ita->second);
    if (!b) return false;
    out_b = *b;
    return true;
  };

  auto get_edge_u32_attr = [&](uint32_t eid, const char* key, uint32_t& out_u) -> bool {
    auto itn = edges.find(eid);
    if (itn == edges.end()) return false;
    auto ita = itn->second.attrs.find(key);
    if (ita == itn->second.attrs.end()) return false;
    auto u = as_u32(ita->second);
    if (!u) return false;
    out_u = *u;
    return true;
  };

  auto get_edge_f64_attr = [&](uint32_t eid, const char* key, double& out_d) -> bool {
    auto ite = edges.find(eid);
    if (ite == edges.end()) return false;
    auto ita = ite->second.attrs.find(key);
    if (ita == ite->second.attrs.end()) return false;
    auto d = as_f64(ita->second);
    if (!d) return false;
    out_d = *d;
    return true;
  };

  // First pass: allocate point ids.
  for (uint32_t nid : point_nodes) {
    // exclude contours
    if (nodes[nid].kind == "relgeo.contour") continue;
    (void)out_program.add_point(RelPointFixed{0.0f, 0.0f});
  }

  // Second pass: build node->point mapping in the same order we allocated above.
  std::unordered_map<uint32_t, RelPointId> nid_to_pid;
  {
    uint32_t next = 1;
    for (uint32_t nid : point_nodes) {
      if (nodes[nid].kind == "relgeo.contour") continue;
      nid_to_pid[nid] = RelPointId(next++);
    }
  }

  // Noun nodes.
  std::unordered_map<uint32_t, RelSegmentId> nid_to_sid;
  for (uint32_t nid : segment_nodes) {
    uint32_t a = 0, b = 0;
    if (!get_u32_attr(nid, "a", a) || !get_u32_attr(nid, "b", b)) {
      if (out_error) *out_error = "relgeo.segment missing a/b";
      return false;
    }
    auto ia = nid_to_pid.find(a);
    auto ib = nid_to_pid.find(b);
    if (ia == nid_to_pid.end() || ib == nid_to_pid.end()) {
      if (out_error) *out_error = "relgeo.segment references unknown point";
      return false;
    }
    nid_to_sid[nid] = out_program.add_segment(ia->second, ib->second);
  }

  std::unordered_map<uint32_t, RelLineId> nid_to_lid;
  for (uint32_t nid : line_nodes) {
    uint32_t a = 0, b = 0;
    if (!get_u32_attr(nid, "a", a) || !get_u32_attr(nid, "b", b)) {
      if (out_error) *out_error = "relgeo.line missing a/b";
      return false;
    }
    auto ia = nid_to_pid.find(a);
    auto ib = nid_to_pid.find(b);
    if (ia == nid_to_pid.end() || ib == nid_to_pid.end()) {
      if (out_error) *out_error = "relgeo.line references unknown point";
      return false;
    }
    nid_to_lid[nid] = out_program.add_line(ia->second, ib->second);
  }

  std::unordered_map<uint32_t, RelRayId> nid_to_rid;
  for (uint32_t nid : ray_nodes) {
    uint32_t origin = 0, through = 0;
    if (!get_u32_attr(nid, "origin", origin) || !get_u32_attr(nid, "through", through)) {
      if (out_error) *out_error = "relgeo.ray missing origin/through";
      return false;
    }
    auto io = nid_to_pid.find(origin);
    auto it = nid_to_pid.find(through);
    if (io == nid_to_pid.end() || it == nid_to_pid.end()) {
      if (out_error) *out_error = "relgeo.ray references unknown point";
      return false;
    }
    nid_to_rid[nid] = out_program.add_ray(io->second, it->second);
  }

  std::unordered_map<uint32_t, RelAngleId> nid_to_aid;
  for (uint32_t nid : angle_nodes) {
    uint32_t a = 0, v = 0, b = 0;
    if (!get_u32_attr(nid, "a", a) || !get_u32_attr(nid, "v", v) || !get_u32_attr(nid, "b", b)) {
      if (out_error) *out_error = "relgeo.angle missing a/v/b";
      return false;
    }
    auto ia = nid_to_pid.find(a);
    auto iv = nid_to_pid.find(v);
    auto ib = nid_to_pid.find(b);
    if (ia == nid_to_pid.end() || iv == nid_to_pid.end() || ib == nid_to_pid.end()) {
      if (out_error) *out_error = "relgeo.angle references unknown point";
      return false;
    }
    nid_to_aid[nid] = out_program.add_angle(ia->second, iv->second, ib->second);
  }

  std::unordered_map<uint32_t, RelCircleId> nid_to_cid;
  for (uint32_t nid : circle_nodes) {
    uint32_t center = 0;
    if (!get_u32_attr(nid, "center", center)) {
      if (out_error) *out_error = "relgeo.circle missing center";
      return false;
    }
    auto ic = nid_to_pid.find(center);
    if (ic == nid_to_pid.end()) {
      if (out_error) *out_error = "relgeo.circle references unknown center point";
      return false;
    }

    RelCircle c;
    c.center = ic->second;
    // radius may be constant (r) or derived from through point.
    if (auto it = nodes[nid].attrs.find("r"); it != nodes[nid].attrs.end()) {
      auto r = as_f64(it->second);
      if (!r) {
        if (out_error) *out_error = "relgeo.circle invalid r";
        return false;
      }
      c.radius = RelRadiusConstant{static_cast<float>(*r)};
    } else if (auto it = nodes[nid].attrs.find("through"); it != nodes[nid].attrs.end()) {
      auto through = as_u32(it->second);
      if (!through) {
        if (out_error) *out_error = "relgeo.circle invalid through";
        return false;
      }
      auto itp = nid_to_pid.find(*through);
      if (itp == nid_to_pid.end()) {
        if (out_error) *out_error = "relgeo.circle through references unknown point";
        return false;
      }
      c.radius = RelRadiusDistance{c.center, itp->second};
    } else {
      if (out_error) *out_error = "relgeo.circle missing radius (r or through)";
      return false;
    }

    nid_to_cid[nid] = out_program.add_circle(std::move(c));
  }

  std::unordered_map<uint32_t, RelArcId> nid_to_arcid;
  for (uint32_t nid : arc_nodes) {
    const std::string& kind = nodes[nid].kind;
    if (kind == "relgeo.arc") {
      uint32_t circle = 0;
      double a0 = 0.0, a1 = 0.0;
      bool ccw = true;
      if (!get_u32_attr(nid, "circle", circle) || !get_f64_attr(nid, "a0", a0) || !get_f64_attr(nid, "a1", a1)) {
        if (out_error) *out_error = "relgeo.arc missing circle/a0/a1";
        return false;
      }
      (void)get_bool_attr(nid, "ccw", ccw);
      auto ic = nid_to_cid.find(circle);
      if (ic == nid_to_cid.end()) {
        if (out_error) *out_error = "relgeo.arc references unknown circle";
        return false;
      }
      nid_to_arcid[nid] = out_program.add_arc(RelArcOnCircleAngles{ic->second, static_cast<float>(a0), static_cast<float>(a1), ccw});
    } else if (kind == "relgeo.arcse") {
      uint32_t circle = 0, start = 0, end = 0;
      bool ccw = true;
      if (!get_u32_attr(nid, "circle", circle) || !get_u32_attr(nid, "start", start) || !get_u32_attr(nid, "end", end)) {
        if (out_error) *out_error = "relgeo.arcse missing circle/start/end";
        return false;
      }
      (void)get_bool_attr(nid, "ccw", ccw);
      auto ic = nid_to_cid.find(circle);
      if (ic == nid_to_cid.end()) {
        if (out_error) *out_error = "relgeo.arcse references unknown circle";
        return false;
      }
      auto is = nid_to_pid.find(start);
      auto ie = nid_to_pid.find(end);
      if (is == nid_to_pid.end() || ie == nid_to_pid.end()) {
        if (out_error) *out_error = "relgeo.arcse references unknown endpoint point";
        return false;
      }
      RelArcOnCircleEndpoints ae;
      ae.circle = ic->second;
      ae.start = is->second;
      ae.end = ie->second;
      ae.normal_z = ccw ? 1.0f : -1.0f;
      nid_to_arcid[nid] = out_program.add_arc(ae);
    } else if (kind == "relgeo.arc3") {
      uint32_t p0 = 0, p1 = 0, p2 = 0;
      if (!get_u32_attr(nid, "p0", p0) || !get_u32_attr(nid, "p1", p1) || !get_u32_attr(nid, "p2", p2)) {
        if (out_error) *out_error = "relgeo.arc3 missing p0/p1/p2";
        return false;
      }
      auto i0 = nid_to_pid.find(p0);
      auto i1 = nid_to_pid.find(p1);
      auto i2 = nid_to_pid.find(p2);
      if (i0 == nid_to_pid.end() || i1 == nid_to_pid.end() || i2 == nid_to_pid.end()) {
        if (out_error) *out_error = "relgeo.arc3 references unknown point";
        return false;
      }
      nid_to_arcid[nid] = out_program.add_arc(RelArc3{i0->second, i1->second, i2->second});
    }
  }

  std::unordered_map<uint32_t, RelBezierId> nid_to_bezid;
  for (uint32_t nid : bezier_nodes) {
    uint32_t p0 = 0, c0 = 0, c1 = 0, p1 = 0;
    if (!get_u32_attr(nid, "p0", p0) || !get_u32_attr(nid, "c0", c0) || !get_u32_attr(nid, "c1", c1) ||
        !get_u32_attr(nid, "p1", p1)) {
      if (out_error) *out_error = "relgeo.bezier missing p0/c0/c1/p1";
      return false;
    }
    auto ip0 = nid_to_pid.find(p0);
    auto ic0 = nid_to_pid.find(c0);
    auto ic1 = nid_to_pid.find(c1);
    auto ip1 = nid_to_pid.find(p1);
    if (ip0 == nid_to_pid.end() || ic0 == nid_to_pid.end() || ic1 == nid_to_pid.end() || ip1 == nid_to_pid.end()) {
      if (out_error) *out_error = "relgeo.bezier references unknown point";
      return false;
    }
    nid_to_bezid[nid] = out_program.add_bezier(ip0->second, ic0->second, ic1->second, ip1->second);
  }

  // Third pass: set expressions for each point.
  for (uint32_t nid : point_nodes) {
    if (nodes[nid].kind == "relgeo.contour") continue;
    NodeRec& nr = nodes[nid];
    const RelPointId pid = nid_to_pid[nid];

    if (nr.kind == "relgeo.point") {
      double x = 0.0, y = 0.0;
      if (!get_f64_attr(nid, "x", x) || !get_f64_attr(nid, "y", y)) {
        if (out_error) *out_error = "relgeo.point missing x/y";
        return false;
      }
      if (!out_program.set_point_expr(pid, RelPointFixed{static_cast<float>(x), static_cast<float>(y)})) return false;
    } else if (nr.kind == "relgeo.lerp") {
      uint32_t a = 0, b = 0;
      double u = 0.0;
      if (!get_u32_attr(nid, "a", a) || !get_u32_attr(nid, "b", b) || !get_f64_attr(nid, "u", u)) {
        if (out_error) *out_error = "relgeo.lerp missing a/b/u";
        return false;
      }
      if (!out_program.set_point_expr(pid, RelPointLerp{nid_to_pid[a], nid_to_pid[b], static_cast<float>(u)})) return false;
    } else if (nr.kind == "relgeo.offset") {
      uint32_t base = 0;
      double dx = 0.0, dy = 0.0;
      if (!get_u32_attr(nid, "base", base) || !get_f64_attr(nid, "dx", dx) || !get_f64_attr(nid, "dy", dy)) {
        if (out_error) *out_error = "relgeo.offset missing base/dx/dy";
        return false;
      }
      if (!out_program.set_point_expr(pid, RelPointOffset{nid_to_pid[base], static_cast<float>(dx), static_cast<float>(dy)})) return false;
    } else if (nr.kind == "relgeo.ccint") {
      uint32_t c0 = 0, c1 = 0;
      uint32_t pick_u32 = 0;
      double r0 = 0.0, r1 = 0.0;
      if (!get_u32_attr(nid, "c0", c0) || !get_f64_attr(nid, "r0", r0) ||
          !get_u32_attr(nid, "c1", c1) || !get_f64_attr(nid, "r1", r1) ||
          !get_u32_attr(nid, "pick", pick_u32)) {
        if (out_error) *out_error = "relgeo.ccint missing c0/r0/c1/r1/pick";
        return false;
      }
      RelPick pick = static_cast<RelPick>(pick_u32);
      RelCircle C0{nid_to_pid[c0], RelRadiusConstant{static_cast<float>(r0)}};
      RelCircle C1{nid_to_pid[c1], RelRadiusConstant{static_cast<float>(r1)}};
      if (!out_program.set_point_expr(pid, RelPointCircleCircleIntersection{C0, C1, pick})) return false;
    } else if (nr.kind == "relgeo.llint") {
      uint32_t l0 = 0, l1 = 0;
      if (!get_u32_attr(nid, "l0", l0) || !get_u32_attr(nid, "l1", l1)) {
        if (out_error) *out_error = "relgeo.llint missing l0/l1";
        return false;
      }
      auto il0 = nid_to_lid.find(l0);
      auto il1 = nid_to_lid.find(l1);
      if (il0 == nid_to_lid.end() || il1 == nid_to_lid.end()) {
        if (out_error) *out_error = "relgeo.llint references unknown line";
        return false;
      }
      if (!out_program.set_point_expr(pid, RelPointLineLineIntersection{il0->second, il1->second})) return false;
    } else if (nr.kind == "relgeo.parametric") {
      std::string kind_s;
      if (!get_str_attr(nid, "segment_kind", kind_s)) {
        if (out_error) *out_error = "relgeo.parametric missing segment_kind";
        return false;
      }
      auto kind_opt = parse_segment_kind(kind_s);
      if (!kind_opt) {
        if (out_error) *out_error = "relgeo.parametric invalid segment_kind";
        return false;
      }
      RelParametricSegment seg;
      seg.kind = *kind_opt;

      auto resolve_point = [&](const char* key, RelPointId& out, const char* msg) -> bool {
        uint32_t raw = 0;
        if (!get_u32_attr(nid, key, raw)) {
          if (out_error) *out_error = msg;
          return false;
        }
        auto it = nid_to_pid.find(raw);
        if (it == nid_to_pid.end()) {
          if (out_error) *out_error = std::string(msg) + " references unknown point";
          return false;
        }
        out = it->second;
        return true;
      };

      if (!resolve_point("start", seg.start, "relgeo.parametric missing start")) return false;
      if (!resolve_point("end", seg.end, "relgeo.parametric missing end")) return false;
      if (seg.kind == RelParametricSegment::Kind::Quadratic || seg.kind == RelParametricSegment::Kind::Cubic) {
        if (!resolve_point("ctrl1", seg.ctrl1, "relgeo.parametric missing ctrl1")) return false;
      }
      if (seg.kind == RelParametricSegment::Kind::Cubic) {
        if (!resolve_point("ctrl2", seg.ctrl2, "relgeo.parametric missing ctrl2")) return false;
      }

      if (seg.kind == RelParametricSegment::Kind::SinWave) {
        double amp = 0.0, cycles = 0.0, phase = 0.0;
        if (!get_f64_attr(nid, "amplitude", amp) || !get_f64_attr(nid, "cycles", cycles) || !get_f64_attr(nid, "phase", phase)) {
          if (out_error) *out_error = "relgeo.parametric sin segment missing amplitude/cycles/phase";
          return false;
        }
        seg.amplitude = static_cast<float>(amp);
        seg.cycles = static_cast<float>(cycles);
        seg.phase = static_cast<float>(phase);
      }

      double u = 0.0;
      if (!get_f64_attr(nid, "u", u)) {
        if (out_error) *out_error = "relgeo.parametric missing u";
        return false;
      }

      bool tangent_forward = true;
      if (auto it = nr.attrs.find("tangent_forward"); it != nr.attrs.end()) {
        auto tf = as_bool(it->second);
        if (!tf) {
          if (out_error) *out_error = "relgeo.parametric invalid tangent_forward value";
          return false;
        }
        tangent_forward = *tf;
      }
      seg.tangent_forward = tangent_forward;

      if (!out_program.set_point_expr(pid, RelPointParametric{seg, static_cast<float>(u)})) return false;
    } else {
      if (out_error) *out_error = "unknown relgeo node kind: " + nr.kind;
      return false;
    }
  }

  // Relation edges -> assertions.
  auto line_endpoints = [&](RelLineId lid, RelPointId& out_a, RelPointId& out_b) -> bool {
    if (!lid) return false;
    const size_t idx = static_cast<size_t>(lid.v - 1);
    if (idx >= out_program.lines().size()) return false;
    const RelLineDef& def = out_program.lines()[idx];
    out_a = def.a;
    out_b = def.b;
    return true;
  };

  auto shared_line_vertex = [&](RelLineId l0, RelLineId l1, RelPointId& out_v, RelPointId& out_a, RelPointId& out_b) -> bool {
    RelPointId l0a{}, l0b{}, l1a{}, l1b{};
    if (!line_endpoints(l0, l0a, l0b) || !line_endpoints(l1, l1a, l1b)) return false;
    if ((l0a == l1a && l0b == l1b) || (l0a == l1b && l0b == l1a)) return false;
    if (l0a == l1a) {
      out_v = l0a;
      out_a = l0b;
      out_b = l1b;
      return true;
    }
    if (l0a == l1b) {
      out_v = l0a;
      out_a = l0b;
      out_b = l1a;
      return true;
    }
    if (l0b == l1a) {
      out_v = l0b;
      out_a = l0a;
      out_b = l1b;
      return true;
    }
    if (l0b == l1b) {
      out_v = l0b;
      out_a = l0a;
      out_b = l1a;
      return true;
    }
    return false;
  };

  for (uint32_t eid : relation_edges) {
    const auto it = edges.find(eid);
    if (it == edges.end()) continue;
    const std::string& k = it->second.kind;

    uint32_t a = 0, b = 0;
    (void)get_edge_u32_attr(eid, "a", a);
    (void)get_edge_u32_attr(eid, "b", b);

    if (k == "relgeo.coincident") {
      auto ia = nid_to_pid.find(a);
      auto ib = nid_to_pid.find(b);
      if (ia != nid_to_pid.end() && ib != nid_to_pid.end()) {
        out_program.add_assertion(RelAssertCoincident{ia->second, ib->second});
      }
    } else if (k == "relgeo.incident") {
      auto ip = nid_to_pid.find(a);
      auto il = nid_to_lid.find(b);
      if (ip != nid_to_pid.end() && il != nid_to_lid.end()) {
        out_program.add_assertion(RelAssertPointOnLine{ip->second, il->second});
      }
    } else if (k == "relgeo.parallel") {
      auto il0 = nid_to_lid.find(a);
      auto il1 = nid_to_lid.find(b);
      if (il0 != nid_to_lid.end() && il1 != nid_to_lid.end()) {
        out_program.add_assertion(RelAssertParallelLines{il0->second, il1->second});
      }
    } else if (k == "relgeo.perp") {
      auto il0 = nid_to_lid.find(a);
      auto il1 = nid_to_lid.find(b);
      if (il0 != nid_to_lid.end() && il1 != nid_to_lid.end()) {
        out_program.add_assertion(RelAssertPerpendicularLines{il0->second, il1->second});
        RelPointId v{}, pa{}, pb{};
        if (shared_line_vertex(il0->second, il1->second, v, pa, pb)) {
          out_program.add_assertion(RelAssertPerpAt{v, pa, pb});
        }
      }
    } else if (k == "relgeo.perp_at") {
      uint32_t v = 0, pa = 0, pb = 0;
      if (get_edge_u32_attr(eid, "v", v) && get_edge_u32_attr(eid, "a", pa) && get_edge_u32_attr(eid, "b", pb)) {
        auto iv = nid_to_pid.find(v);
        auto ia = nid_to_pid.find(pa);
        auto ib = nid_to_pid.find(pb);
        if (iv != nid_to_pid.end() && ia != nid_to_pid.end() && ib != nid_to_pid.end()) {
          out_program.add_assertion(RelAssertPerpAt{iv->second, ia->second, ib->second});
        }
      }
    } else if (k == "relgeo.point_on_circle") {
      auto ip = nid_to_pid.find(a);
      auto ic = nid_to_cid.find(b);
      if (ip != nid_to_pid.end() && ic != nid_to_cid.end()) {
        out_program.add_assertion(RelAssertPointOnCircle{ip->second, ic->second});
      }
    } else if (k == "relgeo.tangent") {
      auto il = nid_to_lid.find(a);
      auto ic = nid_to_cid.find(b);
      if (il != nid_to_lid.end() && ic != nid_to_cid.end()) {
        out_program.add_assertion(RelAssertTangentLineCircle{il->second, ic->second});
      }
    } else if (k == "relgeo.fixed_radius") {
      uint32_t circle_nid = 0;
      double r = 0.0;
      if (get_edge_u32_attr(eid, "a", circle_nid) && get_edge_f64_attr(eid, "r", r)) {
        auto ic = nid_to_cid.find(circle_nid);
        if (ic != nid_to_cid.end()) {
          out_program.add_assertion(RelAssertFixedRadius{ic->second, static_cast<float>(r)});
        }
      }
    } else if (k == "relgeo.arc_angle") {
      uint32_t arc_nid = 0;
      double angle = 0.0;
      if (get_edge_u32_attr(eid, "a", arc_nid) && get_edge_f64_attr(eid, "angle", angle)) {
        auto ia = nid_to_arcid.find(arc_nid);
        if (ia != nid_to_arcid.end()) {
          out_program.add_assertion(RelAssertArcAngle{ia->second, static_cast<float>(angle)});
        }
      }
    }
  }

  // Contours.
  for (uint32_t nid : contour_nodes) {
    NodeRec& nr = nodes[nid];
    if (nr.kind != "relgeo.contour") continue;
    bool closed = true;
    if (auto it = nr.attrs.find("closed"); it != nr.attrs.end()) {
      auto v = as_f64(it->second);
      if (v) closed = (*v >= 0.5);
    }

    RelContourWinding winding = RelContourWinding::Unknown;
    if (auto it = nr.attrs.find("winding"); it != nr.attrs.end()) {
      if (auto w = parse_contour_winding_value(it->second)) winding = *w;
    }

    // gather v0..vN
    std::vector<std::pair<size_t, uint32_t>> verts_tmp;
    for (const auto& kv : nr.attrs) {
      if (kv.first.size() >= 2 && kv.first[0] == 'v') {
        size_t idx = 0;
        try {
          idx = static_cast<size_t>(std::stoul(kv.first.substr(1)));
        } catch (...) {
          continue;
        }
        auto u = as_u32(kv.second);
        if (!u) continue;
        verts_tmp.push_back({idx, *u});
      }
    }
    std::sort(verts_tmp.begin(), verts_tmp.end(), [](auto a, auto b) { return a.first < b.first; });
    std::vector<RelPointId> verts;
    verts.reserve(verts_tmp.size());
    for (auto [_, ref] : verts_tmp) {
      auto it = nid_to_pid.find(ref);
      if (it == nid_to_pid.end()) {
        if (out_error) *out_error = "contour references unknown point";
        return false;
      }
      verts.push_back(it->second);
    }
    out_program.add_contour(std::move(verts), closed, winding);
  }
  return true;
}

} // namespace nodus::tensors::kpath
