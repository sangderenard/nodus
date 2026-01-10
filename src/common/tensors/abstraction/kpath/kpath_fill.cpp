#include "common/tensors/abstraction/kpath/kpath_fill.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nodus::tensors::kpath {

namespace {

constexpr float kEps = 1e-6f;
constexpr float kPi = 3.14159265358979323846f;

struct Pt final {
  float x = 0.0f;
  float y = 0.0f;
};

static Pt rot(Pt p, float c, float s) {
  return Pt{p.x * c - p.y * s, p.x * s + p.y * c};
}

static Pt normalize(Pt v) {
  float l = std::sqrt(v.x * v.x + v.y * v.y);
  if (l < kEps) return Pt{0.0f, 0.0f};
  return Pt{v.x / l, v.y / l};
}

static void bounds_update(float x, float y, float& min_x, float& min_y, float& max_x, float& max_y) {
  min_x = std::min(min_x, x);
  min_y = std::min(min_y, y);
  max_x = std::max(max_x, x);
  max_y = std::max(max_y, y);
}

// Very simple curve flattening (fixed steps). Good enough for now; we can swap
// to adaptive subdivision later.
static void flatten_outline(const GlyphOutline& outline,
                            std::vector<std::vector<Pt>>& out_loops,
                            uint32_t steps_per_curve = 16) {
  out_loops.clear();
  std::vector<Pt> cur_loop;
  Pt cur{0, 0};
  Pt start{0, 0};
  bool have_loop = false;

  auto push = [&](Pt p) {
    cur_loop.push_back(p);
    cur = p;
  };

  for (const auto& seg : outline.segments) {
    switch (seg.op) {
      case OutlineOp::MoveTo: {
        if (!cur_loop.empty()) {
          out_loops.push_back(cur_loop);
          cur_loop.clear();
        }
        cur = Pt{seg.x1, seg.y1};
        start = cur;
        have_loop = true;
        push(cur);
        break;
      }
      case OutlineOp::LineTo: {
        Pt p1{seg.x3, seg.y3};
        push(p1);
        break;
      }
      case OutlineOp::QuadTo: {
        Pt p0{seg.x1, seg.y1};
        Pt p1{seg.x2, seg.y2};
        Pt p2{seg.x3, seg.y3};
        for (uint32_t i = 1; i <= steps_per_curve; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(steps_per_curve);
          float a = 1.0f - t;
          Pt p;
          p.x = a * a * p0.x + 2.0f * a * t * p1.x + t * t * p2.x;
          p.y = a * a * p0.y + 2.0f * a * t * p1.y + t * t * p2.y;
          push(p);
        }
        break;
      }
      case OutlineOp::CubicTo: {
        Pt p0{cur.x, cur.y};
        Pt p1{seg.x1, seg.y1};
        Pt p2{seg.x2, seg.y2};
        Pt p3{seg.x3, seg.y3};
        for (uint32_t i = 1; i <= steps_per_curve; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(steps_per_curve);
          float a = 1.0f - t;
          Pt p;
          p.x = a * a * a * p0.x + 3.0f * a * a * t * p1.x + 3.0f * a * t * t * p2.x + t * t * t * p3.x;
          p.y = a * a * a * p0.y + 3.0f * a * a * t * p1.y + 3.0f * a * t * t * p2.y + t * t * t * p3.y;
          push(p);
        }
        break;
      }
      case OutlineOp::Close: {
        if (have_loop) {
          // Ensure closure.
          if (cur_loop.size() >= 2) {
            Pt last = cur_loop.back();
            if (std::fabs(last.x - start.x) > kEps || std::fabs(last.y - start.y) > kEps) {
              cur_loop.push_back(start);
            }
          }
          out_loops.push_back(cur_loop);
          cur_loop.clear();
          have_loop = false;
        }
        break;
      }
    }
  }

  if (!cur_loop.empty()) out_loops.push_back(cur_loop);
}

static bool segment_intersect_scanline(Pt a, Pt b, float y, float* out_x) {
  // Skip horizontal edges.
  if (std::fabs(a.y - b.y) < kEps) return false;

  // Half-open rule to avoid double hits at vertices.
  float y0 = a.y;
  float y1 = b.y;
  if (y0 > y1) {
    std::swap(y0, y1);
    std::swap(a, b);
  }

  if (!(y >= y0 && y < y1)) return false;

  float t = (y - a.y) / (b.y - a.y);
  *out_x = a.x + (b.x - a.x) * t;
  return true;
}

static void emit_hatch_segments_evenodd(const std::vector<std::vector<Pt>>& loops,
                                       ArmatureProgram& out,
                                       const FillPlanConfig& cfg) {
  // Rotate geometry to make hatch generation axis-aligned.
  float ang = cfg.tool.angle_degrees * (kPi / 180.0f);
  float c = std::cos(ang);
  float s = std::sin(ang);

  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();

  std::vector<std::vector<Pt>> rloops;
  rloops.reserve(loops.size());
  for (const auto& loop : loops) {
    std::vector<Pt> r;
    r.reserve(loop.size());
    for (const auto& p : loop) {
      Pt rp = rot(p, c, -s); // rotate by -ang
      r.push_back(rp);
      bounds_update(rp.x, rp.y, min_x, min_y, max_x, max_y);
    }
    rloops.push_back(std::move(r));
  }

  float pad = cfg.bounds_pad;
  min_y -= pad;
  max_y += pad;

  float stepover = std::max(cfg.tool.stepover, 0.25f);
  float tool_radius = 0.5f * std::max(cfg.tool.tool_width, 0.0f);

  for (float y = min_y; y <= max_y; y += stepover) {
    std::vector<float> xs;

    for (const auto& loop : rloops) {
      if (loop.size() < 2) continue;
      for (size_t i = 1; i < loop.size(); ++i) {
        float x = 0.0f;
        if (segment_intersect_scanline(loop[i - 1], loop[i], y, &x)) {
          xs.push_back(x);
        }
      }
    }

    if (xs.size() < 2) continue;
    std::sort(xs.begin(), xs.end());

    // Even-odd: pair intersections.
    for (size_t i = 0; i + 1 < xs.size(); i += 2) {
      float x0 = xs[i];
      float x1 = xs[i + 1];

      // Kerf/work-side model (in hatch-space): shrink/expand each interior span.
      // This is a deterministic first step toward true offsetting.
      if (tool_radius > 0.0f) {
        switch (cfg.kerf) {
          case KerfMode::Center:
            break;
          case KerfMode::Inside:
            x0 += tool_radius;
            x1 -= tool_radius;
            break;
          case KerfMode::Outside:
            x0 -= tool_radius;
            x1 += tool_radius;
            break;
        }
      }

      if (x1 - x0 < kEps) continue;

      // Rotate segment endpoints back to original space.
      Pt a = rot(Pt{x0, y}, c, s);
      Pt b = rot(Pt{x1, y}, c, s);

      // Emit as: rapid to a (disengaged, safe), plunge/engage (handled by outline->program),
      // but here we directly express it with engaged states.
      ToolPoint rapid;
      rapid.x = a.x;
      rapid.y = a.y;
      rapid.z = cfg.safe_z;
      rapid.engaged = false;
      out.points.push_back(rapid);

      ToolPoint plunge;
      plunge.x = a.x;
      plunge.y = a.y;
      plunge.z = cfg.cut_z;
      plunge.engaged = true;
      out.points.push_back(plunge);

      ToolPoint cut;
      cut.x = b.x;
      cut.y = b.y;
      cut.z = cfg.cut_z;
      cut.engaged = true;
      out.points.push_back(cut);
    }
  }
}

static void emit_offset_path(const std::vector<Pt>& off,
                             const OffsetContourOptions& options,
                             ArmatureProgram& out,
                             float safe_z,
                             float cut_z) {
  if (off.size() < 2) return;

  Pt start = off.front();
  Pt entry_dir = normalize(Pt{off[1].x - off[0].x, off[1].y - off[0].y});
  if (std::fabs(entry_dir.x) < kEps && std::fabs(entry_dir.y) < kEps && off.size() >= 3) {
    entry_dir = normalize(Pt{off[2].x - off[1].x, off[2].y - off[1].y});
  }

  Pt exit_dir = normalize(Pt{off.front().x - off.back().x, off.front().y - off.back().y});
  if (std::fabs(exit_dir.x) < kEps && std::fabs(exit_dir.y) < kEps && off.size() >= 3) {
    exit_dir = normalize(Pt{off[off.size() - 1].x - off[off.size() - 2].x,
                           off[off.size() - 1].y - off[off.size() - 2].y});
  }
  if (std::fabs(exit_dir.x) < kEps && std::fabs(exit_dir.y) < kEps) exit_dir = entry_dir;

  float lead_in = std::max(options.lead_in_length, 0.0f);
  float lead_out = std::max(options.lead_out_length, 0.0f);

  Pt lead_start = start;
  Pt lead_exit = start;

  if (lead_in > kEps) {
    float ang = options.lead_sweep_degrees * (kPi / 180.0f);
    float c = std::cos(ang);
    float s = std::sin(ang);
    Pt swept = rot(entry_dir, c, s);
    lead_start = Pt{start.x - swept.x * lead_in, start.y - swept.y * lead_in};
  }

  if (lead_out > kEps) {
    lead_exit = Pt{start.x + exit_dir.x * lead_out, start.y + exit_dir.y * lead_out};
  }

  ToolPoint rapid{lead_start.x, lead_start.y, safe_z, false};
  out.points.push_back(rapid);

  ToolPoint plunge{lead_start.x, lead_start.y, cut_z, true};
  out.points.push_back(plunge);

  if (std::fabs(lead_start.x - start.x) > kEps || std::fabs(lead_start.y - start.y) > kEps) {
    out.points.push_back(ToolPoint{start.x, start.y, cut_z, true});
  }

  for (size_t i = 1; i < off.size(); ++i) {
    out.points.push_back(ToolPoint{off[i].x, off[i].y, cut_z, true});
  }

  // Close loop.
  out.points.push_back(ToolPoint{start.x, start.y, cut_z, true});

  if (lead_out > kEps) {
    out.points.push_back(ToolPoint{lead_exit.x, lead_exit.y, cut_z, true});
  }

  ToolPoint lift;
  lift.x = (lead_out > kEps) ? lead_exit.x : start.x;
  lift.y = (lead_out > kEps) ? lead_exit.y : start.y;
  lift.z = safe_z;
  lift.engaged = false;
  out.points.push_back(lift);
}

// Minkowski-like offset via mitered edge-offset intersection. Falls back to
// vertex-normal approximation if intersections fail. Positive offset expands,
// negative shrinks (outline units).
static bool plan_offset_contour(const std::vector<std::vector<Pt>>& loops,
                                float offset,
                                const OffsetContourOptions& options,
                                ArmatureProgram& out,
                                float safe_z,
                                float cut_z) {
  if (loops.empty()) return false;

  float mlimit = options.miter_limit > 0.0f ? options.miter_limit : 4.0f;

  for (const auto& loop : loops) {
    if (loop.size() < 2) continue;
    size_t N = loop.size();

    std::vector<Pt> off;
    off.clear();

    // Precompute edge directions and normals.
    std::vector<Pt> dirs(N);
    std::vector<Pt> norms(N);
    for (size_t i = 0; i < N; ++i) {
      const Pt& p0 = loop[i];
      const Pt& p1 = loop[(i + 1) % N];
      Pt d{p1.x - p0.x, p1.y - p0.y};
      dirs[i] = d;
      norms[i] = normalize(Pt{-d.y, d.x});
    }

    for (size_t i = 0; i < N; ++i) {
      const Pt& p = loop[i];
      Pt n_prev = norms[(i + N - 1) % N];
      Pt n_cur = norms[i];

      // Offset lines for edges i-1 and i.
      Pt p_prev0{loop[(i + N - 1) % N].x + n_prev.x * offset, loop[(i + N - 1) % N].y + n_prev.y * offset};
      Pt p_prev1{p.x + n_prev.x * offset, p.y + n_prev.y * offset};
      Pt p_cur0{p.x + n_cur.x * offset, p.y + n_cur.y * offset};
      Pt p_cur1{loop[(i + 1) % N].x + n_cur.x * offset, loop[(i + 1) % N].y + n_cur.y * offset};

      Pt d_prev{p_prev1.x - p_prev0.x, p_prev1.y - p_prev0.y};
      Pt d_cur{p_cur1.x - p_cur0.x, p_cur1.y - p_cur0.y};

      float denom = d_prev.x * d_cur.y - d_prev.y * d_cur.x;
      Pt corner{};
      bool beveled = false;
      if (std::fabs(denom) < 1e-6f) {
        // Parallel; bevel between the two offset segments.
        corner.x = 0.5f * (p_prev1.x + p_cur0.x);
        corner.y = 0.5f * (p_prev1.y + p_cur0.y);
        beveled = true;
      } else {
        float t = ((p_cur0.x - p_prev0.x) * d_cur.y - (p_cur0.y - p_prev0.y) * d_cur.x) / denom;
        corner.x = p_prev0.x + d_prev.x * t;
        corner.y = p_prev0.y + d_prev.y * t;

        // Miter length check: if too large, bevel instead.
        float vx = corner.x - p.x;
        float vy = corner.y - p.y;
        float miter_len = std::sqrt(vx * vx + vy * vy);
        float limit = std::max(mlimit, 1.0f) * std::fabs(offset);
        if (miter_len > limit) {
          beveled = true;
          // Bevel: end of prev offset edge and start of next offset edge.
          off.push_back(Pt{p_prev1.x, p_prev1.y});
          off.push_back(Pt{p_cur0.x, p_cur0.y});
          continue;
        }
      }

      off.push_back(corner);
    }

    if (options.capture) {
      OffsetContourBuffer& buf = *options.capture;
      if (options.capture_tool_width > 0.0f) buf.tool_width = options.capture_tool_width;
      if (options.capture_calibration_scale > 0.0f) buf.calibration_scale = options.capture_calibration_scale;
      if (options.capture_finishing_allowance > 0.0f) buf.finishing_allowance = options.capture_finishing_allowance;

      std::vector<OffsetContourPoint> stored;
      stored.reserve(off.size());
      for (const auto& p : off) stored.push_back(OffsetContourPoint{p.x, p.y});
      buf.loops.push_back(std::move(stored));
    }

    emit_offset_path(off, options, out, safe_z, cut_z);
  }

  return true;
}

// (removed internal definition — exported wrapper implemented after anonymous namespace)

static float compute_stepover(const ToolGeometry& tool) {
  // If stepover is explicitly provided, trust it (but keep it sane).
  if (tool.stepover > 0.0f) return tool.stepover;

  float width = std::max(tool.tool_width, 0.25f);
  float overlap = std::clamp(tool.overlap, 0.0f, 0.95f);
  return std::max(width * (1.0f - overlap), 0.25f);
}

} // namespace

bool plan_fill_for_glyph_outline(const GlyphOutline& outline, ArmatureProgram& out_program, const FillPlanConfig& cfg) {
  std::vector<std::vector<Pt>> loops;
  flatten_outline(outline, loops, 20);
  if (loops.empty()) return false;

  // For now: only EvenOdd + Hatch.
  if (cfg.rule != FillRule::EvenOdd) return false;
  if (cfg.pattern != FillPattern::Hatch) return false;

  FillPlanConfig eff = cfg;
  eff.tool.stepover = compute_stepover(eff.tool);

  emit_hatch_segments_evenodd(loops, out_program, eff);
  if (eff.tool.crosshatch) {
    FillPlanConfig cross = eff;
    cross.tool.angle_degrees = eff.tool.angle_degrees + eff.tool.cross_angle_degrees;
    emit_hatch_segments_evenodd(loops, out_program, cross);
  }
  return !out_program.points.empty();
}

// Exported offset-contour generator: flatten the outline and emit an offset
// contour toolpath. This is implemented here (outside the anonymous helpers)
// so it matches the header declaration and links properly.
bool plan_offset_contour_for_outline(const GlyphOutline& outline,
                                     float offset,
                                     ArmatureProgram& out_program,
                                     float safe_z,
                                     float cut_z,
                                     const OffsetContourOptions& options) {
  struct P2 { float x, y; };
  std::vector<std::vector<P2>> loops;
  loops.clear();

  std::vector<P2> cur_loop;
  P2 cur{0.0f, 0.0f};
  P2 start{0.0f, 0.0f};
  bool have_loop = false;

  auto push = [&](P2 p) {
    cur_loop.push_back(p);
    cur = p;
  };

  const uint32_t steps_per_curve = 20;
  for (const auto& seg : outline.segments) {
    switch (seg.op) {
      case OutlineOp::MoveTo: {
        if (!cur_loop.empty()) { loops.push_back(cur_loop); cur_loop.clear(); }
        cur = P2{seg.x1, seg.y1}; start = cur; have_loop = true; push(cur);
        break;
      }
      case OutlineOp::LineTo: {
        push(P2{seg.x3, seg.y3}); break;
      }
      case OutlineOp::QuadTo: {
        P2 p0{seg.x1, seg.y1}; P2 p1{seg.x2, seg.y2}; P2 p2{seg.x3, seg.y3};
        for (uint32_t i = 1; i <= steps_per_curve; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(steps_per_curve);
          float a = 1.0f - t;
          P2 p;
          p.x = a * a * p0.x + 2.0f * a * t * p1.x + t * t * p2.x;
          p.y = a * a * p0.y + 2.0f * a * t * p1.y + t * t * p2.y;
          push(p);
        }
        break;
      }
      case OutlineOp::CubicTo: {
        P2 p0{cur.x, cur.y}; P2 p1{seg.x1, seg.y1}; P2 p2{seg.x2, seg.y2}; P2 p3{seg.x3, seg.y3};
        for (uint32_t i = 1; i <= steps_per_curve; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(steps_per_curve);
          float a = 1.0f - t;
          P2 p;
          p.x = a * a * a * p0.x + 3.0f * a * a * t * p1.x + 3.0f * a * t * t * p2.x + t * t * t * p3.x;
          p.y = a * a * a * p0.y + 3.0f * a * a * t * p1.y + 3.0f * a * t * t * p2.y + t * t * t * p3.y;
          push(p);
        }
        break;
      }
      case OutlineOp::Close: {
        if (have_loop) {
          if (cur_loop.size() >= 2) {
            P2 last = cur_loop.back();
            if (std::fabs(last.x - start.x) > kEps || std::fabs(last.y - start.y) > kEps) cur_loop.push_back(start);
          }
          loops.push_back(cur_loop);
          cur_loop.clear();
          have_loop = false;
        }
        break;
      }
    }
  }
  if (!cur_loop.empty()) loops.push_back(cur_loop);

  if (loops.empty()) return false;

  // Reuse the robust/miter-limited offset routine in float space.
  std::vector<std::vector<Pt>> floops;
  floops.reserve(loops.size());
  for (const auto& loop : loops) {
    std::vector<Pt> lf;
    lf.reserve(loop.size());
    for (const auto& p : loop) lf.push_back(Pt{p.x, p.y});
    floops.push_back(std::move(lf));
  }

  return plan_offset_contour(floops, offset, options, out_program, safe_z, cut_z);
}

} // namespace nodus::tensors::kpath
