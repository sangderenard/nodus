#include "common/tensors/abstraction/kpath/kpath_ui_layout.h"

#include "common/tensors/abstraction/kpath/kpath_pipeline.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

namespace nodus::tensors::kpath {

namespace {

static constexpr float kTwoPi = 6.283185307179586f;

static GlyphOutline translate_outline(const GlyphOutline& outline, float dx, float dy) {
  GlyphOutline result;
  result.glyph_id = outline.glyph_id;
  result.segments.reserve(outline.segments.size());
  for (const auto& seg : outline.segments) {
    OutlineSegment translated = seg;
    translated.x1 += dx;
    translated.y1 += dy;
    translated.x2 += dx;
    translated.y2 += dy;
    translated.x3 += dx;
    translated.y3 += dy;
    result.segments.push_back(translated);
  }
  return result;
}

static void outline_bounds(const GlyphOutline& outline, float& min_x, float& min_y, float& max_x, float& max_y) {
  if (outline.segments.empty()) {
    min_x = min_y = max_x = max_y = 0.0f;
    return;
  }
  min_x = min_y = std::numeric_limits<float>::max();
  max_x = max_y = std::numeric_limits<float>::lowest();
  auto expand = [&](float x, float y) {
    min_x = std::min(min_x, x);
    max_x = std::max(max_x, x);
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
  };
  for (const auto& seg : outline.segments) {
    expand(seg.x1, seg.y1);
    expand(seg.x2, seg.y2);
    expand(seg.x3, seg.y3);
  }
}

} // namespace

RelGlyph make_rel_square_panel(uint32_t codepoint, float size) {
  RelGlyph glyph;
  glyph.codepoint = codepoint;
  glyph.glyph_id = 0xF2100000u | (codepoint & 0x0000FFFFu);
  glyph.advance_x = size + 4.0f;

  const RelPointId a = glyph.program.add_point(RelPointFixed{0.0f, 0.0f});
  const RelPointId b = glyph.program.add_point(RelPointFixed{size, 0.0f});
  const RelPointId c = glyph.program.add_point(RelPointFixed{size, size});
  const RelPointId d = glyph.program.add_point(RelPointFixed{0.0f, size});

  auto make_mid = [&](RelPointId start, RelPointId end) {
    RelParametricSegment seg;
    seg.kind = RelParametricSegment::Kind::Line;
    seg.start = start;
    seg.end = end;
    seg.tangent_forward = true;
    return glyph.program.add_point(RelPointParametric{seg, 0.5f});
  };

  const RelPointId m0 = make_mid(a, b);
  const RelPointId m1 = make_mid(b, c);
  const RelPointId m2 = make_mid(c, d);
  const RelPointId m3 = make_mid(d, a);
  glyph.program.add_contour({a, m0, b, m1, c, m2, d, m3}, true, RelContourWinding::CCW);
  return glyph;
}

RelGlyph make_rel_hex_panel(uint32_t codepoint, float radius, float waviness_amplitude) {
  RelGlyph glyph;
  glyph.codepoint = codepoint;
  glyph.glyph_id = 0xF2200000u | (codepoint & 0x0000FFFFu);
  glyph.advance_x = radius * 2.5f;

  const float cx = radius;
  const float cy = radius;
  std::array<RelPointId, 6> corners{};
  for (int i = 0; i < 6; ++i) {
    const float angle = kTwoPi * static_cast<float>(i) / 6.0f - 3.14159265f / 6.0f;
    const float x = cx + radius * std::cos(angle);
    const float y = cy + radius * std::sin(angle);
    corners[i] = glyph.program.add_point(RelPointFixed{x, y});
  }

  std::vector<RelPointId> contour;
  contour.reserve(12);
  for (int i = 0; i < 6; ++i) {
    const RelPointId current = corners[i];
    const RelPointId next = corners[(i + 1) % 6];
    contour.push_back(current);

    RelParametricSegment seg;
    seg.start = current;
    seg.end = next;
    seg.tangent_forward = true;

    if (waviness_amplitude > 0.0f) {
      seg.kind = RelParametricSegment::Kind::SinWave;
      seg.amplitude = waviness_amplitude;
      seg.cycles = 1.0f;
      seg.phase = kTwoPi * static_cast<float>(i) / 6.0f;
    } else {
      seg.kind = RelParametricSegment::Kind::Line;
    }

    contour.push_back(glyph.program.add_point(RelPointParametric{seg, 0.5f}));
  }

  glyph.program.add_contour(contour, true, RelContourWinding::CCW);
  return glyph;
}

bool add_relglyph_shape(AtlasBuilder& builder,
                        ArmatureProgram& program,
                        std::vector<GlyphOutline>& outlines,
                        const RelGlyph& glyph,
                        float tx,
                        float ty,
                        const char* label) {
  GlyphOutline outline;
  std::string err;
  if (!compile_relglyph_outline(glyph, outline, 0.0f, 0.0f, 1.0f, &err)) {
    std::fprintf(stderr, "[KPATH-FACILITIES] %s compile failed: %s\n", label ? label : "(null)", err.c_str());
    return false;
  }
  append_glyph_outline_to_program(program, outline, tx, ty, 0.0f, /*samples*/16);
  outlines.push_back(translate_outline(outline, tx, ty));
  TokenLayoutPlan plan;
  if (!build_codepoint_glyph_token(builder, glyph.codepoint, outline, glyph.advance_x, glyph.advance_y, plan)) {
    std::fprintf(stderr, "[KPATH-FACILITIES] %s atlas token build failed\n", label ? label : "(null)");
    return false;
  }
  return true;
}

bool append_ui_panel_with_text(AtlasBuilder& atlas_builder,
                              ArmatureProgram& program,
                              std::vector<GlyphOutline>& outlines,
                              Shaper& shaper,
                              const std::vector<std::string>& words,
                              float panel_x,
                              float panel_y,
                              float panel_size) {
  const uint32_t panel_cp = 0xE020u;
  RelGlyph panel_square = make_rel_square_panel(panel_cp, panel_size);
  if (!add_relglyph_shape(atlas_builder, program, outlines, panel_square, panel_x, panel_y, "ui panel square")) {
    return false;
  }

  const float line_spacing = panel_size / static_cast<float>(words.size() + 1);
  float baseline = panel_y + panel_size - line_spacing;

  for (const auto& word : words) {
    CodepointSequence seq;
    seq.codepoints.reserve(word.size());
    for (unsigned char ch : word) seq.codepoints.push_back(static_cast<uint32_t>(ch));

    std::vector<Cluster> clusters;
    if (!shaper.shape(seq, clusters) || clusters.empty() || clusters.front().glyph_ids.empty()) {
      std::fprintf(stderr, "[KPATH-FACILITIES] UI panel shape failed for word '%s'\n", word.c_str());
      return false;
    }

    GlyphOutline outline;
    if (!shaper.extract_outline(clusters.front().glyph_ids.front(), outline)) {
      std::fprintf(stderr, "[KPATH-FACILITIES] UI panel outline missing for word '%s'\n", word.c_str());
      return false;
    }

    float min_x, min_y, max_x, max_y;
    outline_bounds(outline, min_x, min_y, max_x, max_y);
    const float translate_x = panel_x + (panel_size - (max_x - min_x)) * 0.5f - min_x;
    const float translate_y = baseline - min_y;

    outlines.push_back(translate_outline(outline, translate_x, translate_y));
    append_glyph_outline_to_program(program, outline, translate_x, translate_y, 0.0f, /*samples*/16);

    baseline -= line_spacing;
  }

  return true;
}

} // namespace nodus::tensors::kpath
