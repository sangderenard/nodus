#include "common/tensors/abstraction/kpath/kpath_geoglyph.h"

#include <algorithm>
#include <cmath>

namespace nodus::tensors::kpath {

namespace {

constexpr uint32_t kPUA_Base = 0xE000u;
constexpr uint32_t kPUA_Triangle = 0xE000u;
constexpr uint32_t kPUA_Square = 0xE001u;

constexpr float kPi = 3.14159265358979323846f;

struct P2 final {
  float x = 0.0f;
  float y = 0.0f;
};

static float dist(P2 a, P2 b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

static bool nearly_equal(float a, float b, float eps) {
  return std::fabs(a - b) <= eps;
}

struct Circle final {
  P2 c;
  float r = 0.0f;
};

static bool circle_circle_intersections(const Circle& c0, const Circle& c1, P2& out_a, P2& out_b) {
  // Standard two-circle intersection.
  const float dx = c1.c.x - c0.c.x;
  const float dy = c1.c.y - c0.c.y;
  const float d = std::sqrt(dx * dx + dy * dy);
  if (d < 1e-6f) return false;
  if (d > c0.r + c1.r) return false;
  if (d < std::fabs(c0.r - c1.r)) return false;

  const float a = (c0.r * c0.r - c1.r * c1.r + d * d) / (2.0f * d);
  const float h2 = std::max(0.0f, c0.r * c0.r - a * a);
  const float h = std::sqrt(h2);

  const float xm = c0.c.x + a * dx / d;
  const float ym = c0.c.y + a * dy / d;

  const float rx = -dy * (h / d);
  const float ry = dx * (h / d);

  out_a = P2{xm + rx, ym + ry};
  out_b = P2{xm - rx, ym - ry};
  return true;
}

static uint32_t synthetic_glyph_id_from_codepoint(uint32_t cp) {
  // Keep it obviously synthetic and stable.
  return 0xF0000000u | (cp & 0x0000FFFFu);
}

static void outline_move_to(GlyphOutline& o, P2 p, P2& last, bool& have_last) {
  OutlineSegment s;
  s.op = OutlineOp::MoveTo;
  s.x1 = p.x; s.y1 = p.y;
  s.x2 = p.x; s.y2 = p.y;
  s.x3 = p.x; s.y3 = p.y;
  o.segments.push_back(s);
  last = p;
  have_last = true;
}

static void outline_line_to(GlyphOutline& o, P2 p, P2& last, bool& have_last) {
  if (!have_last) {
    // Best-effort: treat as MoveTo.
    outline_move_to(o, p, last, have_last);
    return;
  }
  OutlineSegment s;
  s.op = OutlineOp::LineTo;
  s.x1 = last.x; s.y1 = last.y;
  s.x2 = 0.0f; s.y2 = 0.0f;
  s.x3 = p.x; s.y3 = p.y;
  o.segments.push_back(s);
  last = p;
}

static void outline_close(GlyphOutline& o, P2 last, bool have_last) {
  OutlineSegment s;
  s.op = OutlineOp::Close;
  s.x1 = have_last ? last.x : 0.0f;
  s.y1 = have_last ? last.y : 0.0f;
  s.x2 = 0.0f; s.y2 = 0.0f;
  s.x3 = 0.0f; s.y3 = 0.0f;
  o.segments.push_back(s);
}

static void compute_bounds(const GlyphOutline& o, float& min_x, float& min_y, float& max_x, float& max_y) {
  min_x = 1e30f; min_y = 1e30f;
  max_x = -1e30f; max_y = -1e30f;
  for (const auto& seg : o.segments) {
    min_x = std::min(min_x, std::min(seg.x1, std::min(seg.x2, seg.x3)));
    min_y = std::min(min_y, std::min(seg.y1, std::min(seg.y2, seg.y3)));
    max_x = std::max(max_x, std::max(seg.x1, std::max(seg.x2, seg.x3)));
    max_y = std::max(max_y, std::max(seg.y1, std::max(seg.y2, seg.y3)));
  }
  if (!(min_x <= max_x)) { min_x = min_y = 0.0f; max_x = max_y = 0.0f; }
}

static bool make_equilateral_triangle(GlyphOutline& out, float side_len, float& adv_x) {
  // Compass construction:
  // A,B define base segment. C is intersection of circles centered at A and B with radius AB.
  const float s = std::max(side_len, 1.0f);
  const P2 A{0.0f, 0.0f};
  const P2 B{s, 0.0f};

  Circle ca{A, s};
  Circle cb{B, s};
  P2 C1{}, C2{};
  if (!circle_circle_intersections(ca, cb, C1, C2)) return false;
  const P2 C = (C1.y >= C2.y) ? C1 : C2;

  // Assert equal side lengths.
  const float eps = 1e-2f;
  const float dAB = dist(A, B);
  const float dBC = dist(B, C);
  const float dCA = dist(C, A);
  if (!nearly_equal(dAB, dBC, eps) || !nearly_equal(dAB, dCA, eps)) return false;

  out.segments.clear();
  P2 last{};
  bool have_last = false;
  outline_move_to(out, A, last, have_last);
  outline_line_to(out, B, last, have_last);
  outline_line_to(out, C, last, have_last);
  outline_close(out, last, have_last);

  float min_x, min_y, max_x, max_y;
  compute_bounds(out, min_x, min_y, max_x, max_y);
  adv_x = (max_x - min_x) + 0.15f * s;
  return true;
}

static bool make_square(GlyphOutline& out, float side_len, float& adv_x) {
  // Straightedge construction from a base segment AB.
  const float s = std::max(side_len, 1.0f);
  const P2 A{0.0f, 0.0f};
  const P2 B{s, 0.0f};

  // Perpendicular at A, same length: D.
  const P2 D{0.0f, s};
  // Complete square with parallels: C.
  const P2 C{s, s};

  const float eps = 1e-2f;
  if (!nearly_equal(dist(A, B), dist(A, D), eps)) return false;
  if (!nearly_equal(dist(B, C), dist(A, B), eps)) return false;
  if (!nearly_equal(dist(D, C), dist(A, B), eps)) return false;

  out.segments.clear();
  P2 last{};
  bool have_last = false;
  outline_move_to(out, A, last, have_last);
  outline_line_to(out, B, last, have_last);
  outline_line_to(out, C, last, have_last);
  outline_line_to(out, D, last, have_last);
  outline_close(out, last, have_last);

  float min_x, min_y, max_x, max_y;
  compute_bounds(out, min_x, min_y, max_x, max_y);
  adv_x = (max_x - min_x) + 0.15f * s;
  return true;
}

static uint32_t synthetic_glyph_id_from_ngon(int sides) {
  const uint32_t s = static_cast<uint32_t>(std::clamp(sides, 0, 0xFFFF));
  return 0xF1000000u | s;
}

static bool make_regular_ngon(GlyphOutline& out, int sides, float radius, float& adv_x) {
  const int n = std::max(sides, 3);
  const float r = std::max(radius, 1.0f);

  out.segments.clear();
  P2 last{};
  bool have_last = false;

  // Vertex 0 on +X.
  const float step = 2.0f * kPi / static_cast<float>(n);
  P2 p0{r, 0.0f};
  outline_move_to(out, p0, last, have_last);
  for (int i = 1; i < n; ++i) {
    const float a = step * static_cast<float>(i);
    P2 p{r * std::cos(a), r * std::sin(a)};
    outline_line_to(out, p, last, have_last);
  }
  outline_close(out, last, have_last);

  float min_x, min_y, max_x, max_y;
  compute_bounds(out, min_x, min_y, max_x, max_y);
  adv_x = (max_x - min_x) + 0.15f * (2.0f * r);
  return true;
}

} // namespace

bool is_geoglyph_codepoint(uint32_t codepoint) {
  return codepoint == kPUA_Triangle || codepoint == kPUA_Square;
}

std::vector<uint32_t> default_geoglyph_codepoints() {
  return {kPUA_Triangle, kPUA_Square};
}

bool make_geoglyph(uint32_t codepoint, GeoGlyph& out) {
  // Default size tuned to match the typical outline extents when a font is
  // loaded around ~18px in tests.
  return make_geoglyph(codepoint, 24.0f, out);
}

bool make_geoglyph(uint32_t codepoint, float size, GeoGlyph& out) {
  out = GeoGlyph{};
  out.codepoint = codepoint;
  out.outline.glyph_id = synthetic_glyph_id_from_codepoint(codepoint);

  float adv_x = 0.0f;
  bool ok = false;
  switch (codepoint) {
    case kPUA_Triangle:
      ok = make_equilateral_triangle(out.outline, size, adv_x);
      break;
    case kPUA_Square:
      ok = make_square(out.outline, size, adv_x);
      break;
    default:
      return false;
  }
  if (!ok) return false;

  out.advance_x = adv_x;
  out.advance_y = 0.0f;
  return true;
}

bool make_regular_ngon_glyph(uint32_t codepoint, int sides, float radius, GeoGlyph& out) {
  out = GeoGlyph{};
  out.codepoint = codepoint;
  out.outline.glyph_id = synthetic_glyph_id_from_codepoint(codepoint);

  float adv_x = 0.0f;
  if (!make_regular_ngon(out.outline, sides, radius, adv_x)) return false;
  out.advance_x = adv_x;
  out.advance_y = 0.0f;
  return true;
}

bool make_regular_ngon_glyph(int sides, float radius, GeoGlyph& out) {
  out = GeoGlyph{};
  out.codepoint = 0;
  out.outline.glyph_id = synthetic_glyph_id_from_ngon(sides);

  float adv_x = 0.0f;
  if (!make_regular_ngon(out.outline, sides, radius, adv_x)) return false;
  out.advance_x = adv_x;
  out.advance_y = 0.0f;
  return true;
}

} // namespace nodus::tensors::kpath
