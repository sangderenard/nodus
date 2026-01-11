#include "common/tensors/abstraction/kpath/kpath_relgeo_ir.h"

#include <algorithm>
#include <sstream>

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

static std::optional<std::string> as_string(const GraphIrValue& v) {
  if (const auto* p = std::get_if<std::string>(&v)) return *p;
  return std::nullopt;
}

static std::optional<RelPick> parse_pick(std::string_view s) {
  if (s == "HigherY" || s == "higher_y" || s == "hy") return RelPick::HigherY;
  if (s == "LowerY" || s == "lower_y" || s == "ly") return RelPick::LowerY;
  if (s == "HigherX" || s == "higher_x" || s == "hx") return RelPick::HigherX;
  if (s == "LowerX" || s == "lower_x" || s == "lx") return RelPick::LowerX;
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

static void emit_node_contour(GraphIrContext& ctx, const std::vector<uint32_t>& verts, bool closed) {
  if (!ctx.edits) return;
  const uint32_t n = ctx.edits->add_node("relgeo.contour");
  ctx.edits->set_attr(n, "closed", closed ? 1.0 : 0.0);
  for (size_t i = 0; i < verts.size(); ++i) {
    ctx.edits->set_attr(n, "v" + std::to_string(i), static_cast<uint32_t>(verts[i]));
  }
}

} // namespace

GraphIrOperatorSet make_relgeo_ir_ops() {
  GraphIrOperatorSet set;

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

  set.add(GraphIrOpSpec{"contour", 2, 0xFFFFFFFFu, "contour(p0: u32, p1: u32, ..., [\"open\"|\"closed\"]) -> void"},
          [](const GraphIrOpSpec&, const std::vector<GraphIrValue>& args, GraphIrValue& out, GraphIrContext& ctx, std::string& err) {
            if (!ctx.edits) {
              err = "contour(): ctx.edits is null";
              return false;
            }

            bool closed = true;
            size_t n = args.size();
            if (n >= 1) {
              if (auto last_s = as_string(args.back())) {
                std::string v = *last_s;
                std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (v == "open") closed = false;
                if (v == "closed") closed = true;
                n -= 1;
              }
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

            emit_node_contour(ctx, raw, closed);
            out = std::monostate{};
            return true;
          });

  return set;
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
  std::unordered_map<uint32_t, NodeRec> nodes;
  std::vector<uint32_t> point_nodes;
  std::vector<uint32_t> contour_nodes;

  for (const auto& e : edits.edits()) {
    if (e.kind == GraphEditKind::AddNode) {
      nodes[e.a].kind = e.key;
      if (e.key.rfind("relgeo.", 0) == 0) {
        if (e.key == "relgeo.contour") contour_nodes.push_back(e.a);
        else point_nodes.push_back(e.a);
      }
      continue;
    }
    if (e.kind == GraphEditKind::SetAttr) {
      nodes[e.a].attrs[e.key] = e.value;
      continue;
    }
  }

  std::sort(point_nodes.begin(), point_nodes.end());
  std::sort(contour_nodes.begin(), contour_nodes.end());

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
    } else {
      if (out_error) *out_error = "unknown relgeo node kind: " + nr.kind;
      return false;
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
    out_program.add_contour(std::move(verts), closed);
  }
  return true;
}

} // namespace nodus::tensors::kpath
