#include "common/tensors/abstraction/kpath/kpath_fill.h"

#include "common/tensors/abstraction/kpath/kpath_planar_toolpath.h"

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

static uint32_t f32_bits(float v) {
  uint32_t u = 0;
  static_assert(sizeof(uint32_t) == sizeof(float));
  std::memcpy(&u, &v, sizeof(uint32_t));
  return u;
}

static uint64_t fnv1a_u64(const void* data, size_t len) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < len; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= 1099511628211ull;
  }
  return h;
}

static uint64_t hash_combine_u64(uint64_t a, uint64_t b) {
  // A simple reversible-ish mix.
  a ^= b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2);
  return a;
}

static uint64_t hash_outline(const GlyphOutline& outline) {
  uint64_t h = 14695981039346656037ull;
  h = hash_combine_u64(h, static_cast<uint64_t>(outline.glyph_id));
  for (const auto& seg : outline.segments) {
    const uint8_t op = static_cast<uint8_t>(seg.op);
    h = hash_combine_u64(h, fnv1a_u64(&op, sizeof(op)));
    const uint32_t bits[6] = {f32_bits(seg.x1), f32_bits(seg.y1), f32_bits(seg.x2), f32_bits(seg.y2), f32_bits(seg.x3), f32_bits(seg.y3)};
    h = hash_combine_u64(h, fnv1a_u64(bits, sizeof(bits)));
  }
  h = hash_combine_u64(h, static_cast<uint64_t>(outline.segments.size()));
  return h;
}

static uint64_t hash_fill_cfg(const FillPlanConfig& cfg) {
  uint64_t h = 14695981039346656037ull;
  const uint32_t u[10] = {
    static_cast<uint32_t>(cfg.rule),
    static_cast<uint32_t>(cfg.pattern),
    static_cast<uint32_t>(cfg.kerf),
    f32_bits(cfg.tool.tool_width),
    f32_bits(cfg.tool.stepover),
    f32_bits(cfg.tool.overlap),
    f32_bits(cfg.tool.angle_degrees),
    static_cast<uint32_t>(cfg.tool.crosshatch ? 1u : 0u),
    f32_bits(cfg.tool.cross_angle_degrees),
    f32_bits(cfg.tool.offset_miter_limit),
  };
  h = hash_combine_u64(h, fnv1a_u64(u, sizeof(u)));
  const uint32_t u2[5] = {f32_bits(cfg.bounds_pad), f32_bits(cfg.tolerance), f32_bits(cfg.safe_z), f32_bits(cfg.cut_z), 0u};
  h = hash_combine_u64(h, fnv1a_u64(u2, sizeof(u2)));
  return h;
}

static uint64_t hash_offset_opts(const OffsetContourOptions& opt) {
  uint64_t h = 14695981039346656037ull;
  const uint32_t u[8] = {
    f32_bits(opt.miter_limit),
    f32_bits(opt.lead_in_length),
    f32_bits(opt.lead_out_length),
    f32_bits(opt.lead_sweep_degrees),
    static_cast<uint32_t>(opt.enable_lead_in ? 1u : 0u),
    f32_bits(opt.capture_tool_width),
    f32_bits(opt.capture_calibration_scale),
    f32_bits(opt.capture_finishing_allowance),
  };
  h = hash_combine_u64(h, fnv1a_u64(u, sizeof(u)));
  const uint32_t u2[1] = {f32_bits(opt.tolerance)};
  h = hash_combine_u64(h, fnv1a_u64(u2, sizeof(u2)));
  // Intentionally do NOT hash capture pointer.
  return h;
}

struct PlanCacheEntry final {
  uint64_t key{};
  PlanarToolpath path;
};

static PlanarToolpath* cache_find(std::vector<PlanCacheEntry>& cache, uint64_t key) {
  for (auto& e : cache) {
    if (e.key == key) return &e.path;
  }
  return nullptr;
}

static void cache_store(std::vector<PlanCacheEntry>& cache, uint64_t key, PlanarToolpath value, size_t cap = 256) {
  if (PlanarToolpath* existing = cache_find(cache, key)) {
    *existing = std::move(value);
    return;
  }
  if (cache.size() >= cap) {
    // Deterministic eviction: FIFO.
    cache.erase(cache.begin());
  }
  cache.push_back(PlanCacheEntry{key, std::move(value)});
}

static float dist_point_to_line(Pt p, Pt a, Pt b) {
  float vx = b.x - a.x;
  float vy = b.y - a.y;
  float wx = p.x - a.x;
  float wy = p.y - a.y;
  float denom = vx * vx + vy * vy;
  if (denom < kEps) {
    float dx = p.x - a.x;
    float dy = p.y - a.y;
    return std::sqrt(dx * dx + dy * dy);
  }
  float t = (wx * vx + wy * vy) / denom;
  t = std::clamp(t, 0.0f, 1.0f);
  float projx = a.x + t * vx;
  float projy = a.y + t * vy;
  float dx = p.x - projx;
  float dy = p.y - projy;
  return std::sqrt(dx * dx + dy * dy);
}

static void flatten_quad_adaptive(Pt p0, Pt p1, Pt p2, float tol, std::vector<Pt>& out) {
  // Iterative subdivision stack.
  struct Q { Pt a, b, c; };
  std::vector<Q> stack;
  stack.push_back(Q{p0, p1, p2});

  auto flat_enough = [&](const Q& q) {
    return dist_point_to_line(q.b, q.a, q.c) <= tol;
  };

  while (!stack.empty()) {
    Q q = stack.back();
    stack.pop_back();
    if (flat_enough(q)) {
      out.push_back(q.c);
      continue;
    }
    // Subdivide at t=0.5 via De Casteljau.
    Pt ab{0.5f * (q.a.x + q.b.x), 0.5f * (q.a.y + q.b.y)};
    Pt bc{0.5f * (q.b.x + q.c.x), 0.5f * (q.b.y + q.c.y)};
    Pt mid{0.5f * (ab.x + bc.x), 0.5f * (ab.y + bc.y)};
    // Push right then left so left is processed first (deterministic order).
    stack.push_back(Q{mid, bc, q.c});
    stack.push_back(Q{q.a, ab, mid});
  }
}

static void flatten_cubic_adaptive(Pt p0, Pt p1, Pt p2, Pt p3, float tol, std::vector<Pt>& out) {
  struct C { Pt a, b, c, d; };
  std::vector<C> stack;
  stack.push_back(C{p0, p1, p2, p3});

  auto flat_enough = [&](const C& c) {
    float d1 = dist_point_to_line(c.b, c.a, c.d);
    float d2 = dist_point_to_line(c.c, c.a, c.d);
    return std::max(d1, d2) <= tol;
  };

  while (!stack.empty()) {
    C c = stack.back();
    stack.pop_back();
    if (flat_enough(c)) {
      out.push_back(c.d);
      continue;
    }
    // De Casteljau at t=0.5
    Pt ab{0.5f * (c.a.x + c.b.x), 0.5f * (c.a.y + c.b.y)};
    Pt bc{0.5f * (c.b.x + c.c.x), 0.5f * (c.b.y + c.c.y)};
    Pt cd{0.5f * (c.c.x + c.d.x), 0.5f * (c.c.y + c.d.y)};
    Pt abbc{0.5f * (ab.x + bc.x), 0.5f * (ab.y + bc.y)};
    Pt bccd{0.5f * (bc.x + cd.x), 0.5f * (bc.y + cd.y)};
    Pt mid{0.5f * (abbc.x + bccd.x), 0.5f * (abbc.y + bccd.y)};
    // Push right then left for deterministic order.
    stack.push_back(C{mid, bccd, cd, c.d});
    stack.push_back(C{c.a, ab, abbc, mid});
  }
}

// Very simple curve flattening (fixed steps). Good enough for now; we can swap
// to adaptive subdivision later.
static void flatten_outline(const GlyphOutline& outline,
                            std::vector<std::vector<Pt>>& out_loops,
                            float tolerance,
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
        if (tolerance > 0.0f) {
          flatten_quad_adaptive(p0, p1, p2, tolerance, cur_loop);
          cur = cur_loop.back();
        } else {
          for (uint32_t i = 1; i <= steps_per_curve; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(steps_per_curve);
            float a = 1.0f - t;
            Pt p;
            p.x = a * a * p0.x + 2.0f * a * t * p1.x + t * t * p2.x;
            p.y = a * a * p0.y + 2.0f * a * t * p1.y + t * t * p2.y;
            push(p);
          }
        }
        break;
      }
      case OutlineOp::CubicTo: {
        Pt p0{cur.x, cur.y};
        Pt p1{seg.x1, seg.y1};
        Pt p2{seg.x2, seg.y2};
        Pt p3{seg.x3, seg.y3};
        if (tolerance > 0.0f) {
          flatten_cubic_adaptive(p0, p1, p2, p3, tolerance, cur_loop);
          cur = cur_loop.back();
        } else {
          for (uint32_t i = 1; i <= steps_per_curve; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(steps_per_curve);
            float a = 1.0f - t;
            Pt p;
            p.x = a * a * a * p0.x + 3.0f * a * a * t * p1.x + 3.0f * a * t * t * p2.x + t * t * t * p3.x;
            p.y = a * a * a * p0.y + 3.0f * a * a * t * p1.y + 3.0f * a * t * t * p2.y + t * t * t * p3.y;
            push(p);
          }
        }
        break;
      }
      case OutlineOp::Arc: {
        // Params: x1/y1=center, x2=radius, y2=start angle (rad), x3=sweep (rad)
        const float cx = seg.x1;
        const float cy = seg.y1;
        const float r = std::max(seg.x2, kEps);
        const float start = seg.y2;
        const float sweep = seg.x3;
        uint32_t steps = std::max<uint32_t>(4, static_cast<uint32_t>(std::ceil(std::fabs(sweep) / (kPi / 8.0f))));
        if (tolerance > 0.0f) {
          // Sagitta <= tol: r*(1-cos(dtheta/2)) <= tol
          const float tol = std::max(tolerance, 1e-4f);
          const float ratio = std::clamp(1.0f - tol / r, -1.0f, 1.0f);
          float dtheta = 2.0f * std::acos(ratio);
          if (!(dtheta > 0.0f)) dtheta = kPi / 8.0f;
          steps = std::max<uint32_t>(steps, static_cast<uint32_t>(std::ceil(std::fabs(sweep) / dtheta)));
        } else {
          steps = std::max<uint32_t>(steps, steps_per_curve);
        }
        for (uint32_t i = 1; i <= steps; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(steps);
          float ang = start + sweep * t;
          Pt p{cx + r * std::cos(ang), cy + r * std::sin(ang)};
          push(p);
        }
        break;
      }
      case OutlineOp::Sin: {
        // Params: x1/y1=end point; x2=amplitude; y2=cycles; x3=phase (rad)
        Pt start_pt{cur.x, cur.y};
        Pt end_pt{seg.x1, seg.y1};
        float dx = end_pt.x - start_pt.x;
        float dy = end_pt.y - start_pt.y;
        float len = std::max(std::sqrt(dx * dx + dy * dy), kEps);
        float amp = seg.x2;
        float cycles = seg.y2;
        float phase = seg.x3;

        uint32_t steps = std::max<uint32_t>(steps_per_curve,
            static_cast<uint32_t>(std::ceil(len / 0.75f) + std::fabs(cycles) * 4.0f));
        if (tolerance > 0.0f) {
          // Heuristic: more samples for higher amplitude and tighter tolerances.
          const float tol = std::max(tolerance, 1e-4f);
          float amp_factor = std::max(std::fabs(amp) / tol, 1.0f);
          steps = std::max<uint32_t>(steps, static_cast<uint32_t>(std::ceil(amp_factor * 8.0f + std::fabs(cycles) * 8.0f)));
        }

        float tx_dir = dx / len;
        float ty_dir = dy / len;
        float nx_dir = -ty_dir;
        float ny_dir = tx_dir;

        for (uint32_t i = 1; i <= steps; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(steps);
          float base_x = start_pt.x + dx * t;
          float base_y = start_pt.y + dy * t;
          float s = std::sin((2.0f * kPi * cycles * t) + phase);
          Pt p{base_x + nx_dir * amp * s, base_y + ny_dir * amp * s};
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

static bool point_inside(const std::vector<Pt>& loop, Pt p) {
  bool inside = false;
  size_t n = loop.size();
  if (n < 3) return false;
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    const Pt& pi = loop[i];
    const Pt& pj = loop[j];
    bool intersect = ((pi.y > p.y) != (pj.y > p.y)) &&
                     (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y + kEps) + pi.x);
    if (intersect) inside = !inside;
  }
  return inside;
}

static bool point_inside_any(const std::vector<std::vector<Pt>>& loops, Pt p) {
  for (const auto& loop : loops) {
    if (point_inside(loop, p)) return true;
  }
  return false;
}

static void emit_offset_path(const std::vector<Pt>& off,
                             const std::vector<Pt>& base_loop,
                             const std::vector<std::vector<Pt>>& loops,
                             const OffsetContourOptions& options,
                             ArmatureProgram& out,
                             float safe_z,
                             float cut_z,
                             float loop_area) {
  if (off.size() < 2) return;

  size_t entry_idx = 0;
  float max_len_sq = 0.0f;
  for (size_t i = 0; i < off.size(); ++i) {
    const Pt& a = off[i];
    const Pt& b = off[(i + 1) % off.size()];
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float len_sq = dx * dx + dy * dy;
    if (len_sq > max_len_sq) {
      max_len_sq = len_sq;
      entry_idx = i;
    }
  }

  const Pt& start = off[entry_idx];
  const Pt& prev = off[(entry_idx + off.size() - 1) % off.size()];
  const Pt& next = off[(entry_idx + 1) % off.size()];

  Pt tangent_back = normalize(Pt{start.x - prev.x, start.y - prev.y});
  if (std::fabs(tangent_back.x) < kEps && std::fabs(tangent_back.y) < kEps) {
    tangent_back = normalize(Pt{next.x - start.x, next.y - start.y});
  }
  Pt tangent_forward = normalize(Pt{next.x - start.x, next.y - start.y});
  if (std::fabs(tangent_forward.x) < kEps && std::fabs(tangent_forward.y) < kEps) tangent_forward = tangent_back;

  float lead_in = std::max(options.lead_in_length, 0.0f);
  float lead_out = std::max(options.lead_out_length, 0.0f);

  Pt lead_start = start;
  bool built_lead = false;
  if (options.enable_lead_in && lead_in > kEps &&
      (std::fabs(tangent_forward.x) > kEps || std::fabs(tangent_forward.y) > kEps)) {
    Pt candidate{start.x - tangent_forward.x * lead_in, start.y - tangent_forward.y * lead_in};
    bool has_clearance = true;
    const int samples = 4;
    for (int s = 0; s < samples; ++s) {
      float t = static_cast<float>(s + 1) / static_cast<float>(samples);
      Pt sample{candidate.x + tangent_forward.x * lead_in * t,
                candidate.y + tangent_forward.y * lead_in * t};
      if (point_inside_any(loops, sample)) {
        has_clearance = false;
        break;
      }
    }
    if (has_clearance) {
      lead_start = candidate;
      built_lead = true;
    }
  }

  Pt lead_exit = start;
  if (lead_out > kEps && (std::fabs(tangent_forward.x) > kEps || std::fabs(tangent_forward.y) > kEps)) {
    lead_exit.x += tangent_forward.x * lead_out;
    lead_exit.y += tangent_forward.y * lead_out;
  }

  if (built_lead)
    out.points.push_back(ToolPoint{lead_start.x, lead_start.y, safe_z, false});
  else
    out.points.push_back(ToolPoint{start.x, start.y, safe_z, false});
  out.points.push_back(ToolPoint{lead_start.x, lead_start.y, cut_z, true});
  if (built_lead) {
    out.points.push_back(ToolPoint{start.x, start.y, cut_z, true});
  }

  for (size_t i = 1; i < off.size(); ++i) {
    const Pt& p = off[(entry_idx + i) % off.size()];
    out.points.push_back(ToolPoint{p.x, p.y, cut_z, true});
  }
  out.points.push_back(ToolPoint{start.x, start.y, cut_z, true});

  if (lead_out > kEps) {
    out.points.push_back(ToolPoint{lead_exit.x, lead_exit.y, cut_z, true});
  }

  Pt lift_p = (lead_out > kEps) ? lead_exit : start;
  out.points.push_back(ToolPoint{lift_p.x, lift_p.y, safe_z, false});
}

// Minkowski-like offset via mitered edge-offset intersection. Falls back to
// vertex-normal approximation if intersections fail. Positive offset expands,
// negative shrinks (outline units).
static float signed_area(const std::vector<Pt>& loop) {
  if (loop.size() < 3) return 0.0f;
  float area = 0.0f;
  for (size_t i = 1; i < loop.size(); ++i) {
    const Pt& a = loop[i - 1];
    const Pt& b = loop[i];
    area += a.x * b.y - b.x * a.y;
  }
  return area * 0.5f;
}

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
    float area = signed_area(loop);
    // Keep offsets pointing "outward" relative to material: outer loops (CCW) use +offset,
    // holes (likely CW) get offset flipped so the tool stays consistent.
    float loop_offset = (area < 0.0f) ? -offset : offset;

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
      Pt p_prev0{loop[(i + N - 1) % N].x + n_prev.x * loop_offset, loop[(i + N - 1) % N].y + n_prev.y * loop_offset};
      Pt p_prev1{p.x + n_prev.x * loop_offset, p.y + n_prev.y * loop_offset};
      Pt p_cur0{p.x + n_cur.x * loop_offset, p.y + n_cur.y * loop_offset};
      Pt p_cur1{loop[(i + 1) % N].x + n_cur.x * loop_offset, loop[(i + 1) % N].y + n_cur.y * loop_offset};

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
        float limit = std::max(mlimit, 1.0f) * std::fabs(loop_offset);
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

    emit_offset_path(off, loop, loops, options, out, safe_z, cut_z, area);
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
  flatten_outline(outline, loops, cfg.tolerance, 20);
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

bool plan_iterative_fill(const std::vector<GlyphOutline>& outlines,
                         ArmatureProgram& out_program,
                         const FillPlanConfig& base_cfg,
                         const GaussianToolParams& tool,
                         OffsetContourBuffer* coverage_loops,
                         float finishing_allowance,
                         size_t max_passes) {
  if (outlines.empty()) return false;

  out_program.points.clear();
  bool emitted = false;
  (void)tool;
  float width = std::max(base_cfg.tool.tool_width, 0.5f);
  float reduction_factor = 0.65f;

  OffsetContourOptions capture_opts;
  if (coverage_loops) {
    coverage_loops->loops.clear();
    capture_opts.capture = coverage_loops;
    capture_opts.capture_tool_width = width;
    capture_opts.capture_calibration_scale = 1.0f;
    capture_opts.capture_finishing_allowance = finishing_allowance;
    capture_opts.tolerance = base_cfg.tolerance;
  }

  for (size_t pass = 0; pass < max_passes; ++pass) {
    if (width <= finishing_allowance) break;

    FillPlanConfig cfg = base_cfg;
    cfg.tool.tool_width = width;
    cfg.tool.stepover = compute_stepover(cfg.tool);

    ArmatureProgram pass_prog;
    bool pass_ok = false;
    for (const auto& outline : outlines) {
      ArmatureProgram glyph_prog;
      if (plan_fill_for_glyph_outline(outline, glyph_prog, cfg)) {
        pass_prog.points.insert(pass_prog.points.end(), glyph_prog.points.begin(), glyph_prog.points.end());
        pass_ok = true;
      }
    }
    if (!pass_ok) break;

    if (capture_opts.capture) {
      for (const auto& outline : outlines) {
        ArmatureProgram dummy;
        (void)plan_offset_contour_for_outline(outline, width * 0.5f, dummy, cfg.safe_z, cfg.cut_z, capture_opts);
      }
      capture_opts.capture = nullptr; // only capture once
    }

    out_program.points.insert(out_program.points.end(), pass_prog.points.begin(), pass_prog.points.end());
    emitted = true;
    width *= reduction_factor;
  }

  return emitted;
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
  std::vector<std::vector<Pt>> loops;
  flatten_outline(outline, loops, options.tolerance, 20);
  if (loops.empty()) return false;
  return plan_offset_contour(loops, offset, options, out_program, safe_z, cut_z);
}

namespace {

static bool nearly_equal(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) <= eps;
}

static void append_span_from_points(PlanarToolpath& out,
                                    ToolMode mode,
                                    const std::vector<ToolPoint>& pts,
                                    size_t begin,
                                    size_t end) {
  if (end <= begin) return;
  PlanarSpan span;
  span.tool_mode = mode;
  span.points.reserve(end - begin);
  for (size_t i = begin; i < end; ++i) {
    span.points.push_back(Vec2{pts[i].x, pts[i].y});
  }
  if (span.points.size() >= 2) {
    const Vec2& first = span.points.front();
    const Vec2& last = span.points.back();
    span.closed = nearly_equal(first.x, last.x) && nearly_equal(first.y, last.y);
  }
  span.winding = planar_winding(span);
  out.spans.push_back(std::move(span));
}

static void planarize_armature_program(const ArmatureProgram& prog,
                                       ToolMode engaged_mode,
                                       PlanarToolpath& out) {
  out.clear();
  if (prog.points.empty()) return;

  const auto& pts = prog.points;
  size_t seg_begin = 0;
  bool cur_engaged = pts[0].engaged;

  for (size_t i = 1; i < pts.size(); ++i) {
    if (pts[i].engaged != cur_engaged) {
      append_span_from_points(out, cur_engaged ? engaged_mode : ToolMode::Travel, pts, seg_begin, i);
      seg_begin = i;
      cur_engaged = pts[i].engaged;
    }
  }
  append_span_from_points(out, cur_engaged ? engaged_mode : ToolMode::Travel, pts, seg_begin, pts.size());
}

} // namespace

bool plan_planar_fill_for_glyph_outline(const GlyphOutline& outline,
                                       PlanarToolpath& out_path,
                                       const FillPlanConfig& cfg) {
  static std::vector<PlanCacheEntry> s_cache;
  const uint64_t key = hash_combine_u64(hash_outline(outline), hash_combine_u64(hash_fill_cfg(cfg), 0xF11F11F11ull));
  if (PlanarToolpath* cached = cache_find(s_cache, key)) {
    out_path = *cached;
    return !out_path.empty();
  }

  ArmatureProgram tmp;
  if (!plan_fill_for_glyph_outline(outline, tmp, cfg)) {
    out_path.clear();
    return false;
  }
  planarize_armature_program(tmp, ToolMode::Cut, out_path);
  cache_store(s_cache, key, out_path);
  return !out_path.empty();
}

bool plan_planar_offset_contour_for_outline(const GlyphOutline& outline,
                                            float offset,
                                            PlanarToolpath& out_path,
                                            float safe_z,
                                            float cut_z,
                                            const OffsetContourOptions& options) {
  static std::vector<PlanCacheEntry> s_cache;
  uint64_t h = hash_outline(outline);
  h = hash_combine_u64(h, fnv1a_u64(&offset, sizeof(offset)));
  h = hash_combine_u64(h, fnv1a_u64(&safe_z, sizeof(safe_z)));
  h = hash_combine_u64(h, fnv1a_u64(&cut_z, sizeof(cut_z)));
  const uint64_t key = hash_combine_u64(h, hash_combine_u64(hash_offset_opts(options), 0x0FF5E7u));
  if (PlanarToolpath* cached = cache_find(s_cache, key)) {
    out_path = *cached;
    return !out_path.empty();
  }

  ArmatureProgram tmp;
  if (!plan_offset_contour_for_outline(outline, offset, tmp, safe_z, cut_z, options)) {
    out_path.clear();
    return false;
  }
  planarize_armature_program(tmp, ToolMode::Cut, out_path);
  cache_store(s_cache, key, out_path);
  return !out_path.empty();
}

} // namespace nodus::tensors::kpath
