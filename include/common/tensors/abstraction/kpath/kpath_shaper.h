#pragma once

#include "kpath_ids.h"

#if defined(__has_include)
#if __has_include(<hb-ft.h>)
#include <hb-ft.h>
#else
#include <harfbuzz/hb-ft.h>
#endif
#else
#include <hb-ft.h>
#endif

#include <hb.h>

#include <ft2build.h>
#include FT_FREETYPE_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nodus::tensors::kpath {

struct CodepointSequence final {
  std::vector<uint32_t> codepoints; // Unicode scalar values
};

struct Cluster final {
  size_t start = 0;   // index into codepoints
  size_t count = 0;   // number of codepoints
  std::vector<uint32_t> glyph_ids; // HarfBuzz cluster glyphs
  float advance_x = 0.0f;
  float advance_y = 0.0f;
};

// Outline operations describing vector geometry. Arc/Sin are custom analytical
// primitives that are sampled later by raster/toolpath converters; the rest are
// standard move/Bezier/close.
enum class OutlineOp : uint8_t {
  MoveTo,
  LineTo,
  QuadTo,
  CubicTo,
  Arc,   // circle arc: center, radius, start angle, sweep angle (radians)
  Sin,   // sinusoid along chord: end point, amplitude, cycles, phase
  Close
};

struct OutlineSegment final {
  OutlineOp op;
  float x1, y1;
  float x2, y2;
  float x3, y3;
};

struct GlyphOutline final {
  uint32_t glyph_id = 0;
  std::vector<OutlineSegment> segments;
};

class GlyphCache final {
 public:
  void store_glyph(uint32_t glyph_id, GlyphOutline outline) {
    outlines_[glyph_id] = std::move(outline);
  }

  const GlyphOutline* find(uint32_t glyph_id) const {
    auto it = outlines_.find(glyph_id);
    return it == outlines_.end() ? nullptr : &it->second;
  }

 private:
  std::unordered_map<uint32_t, GlyphOutline> outlines_;
};

class Shaper final {
 public:
  Shaper();
  ~Shaper();

  bool load_font(const std::string& path, float point_size = 12.0f);

  bool shape(const CodepointSequence& seq, std::vector<Cluster>& out_clusters) const;
  bool extract_outline(uint32_t glyph_id, GlyphOutline& out) const;

 private:
  struct FontResource final {
    std::vector<uint8_t> data;
    FT_Face face{};
    hb_font_t* hb_font{};

    ~FontResource();
  };

  bool ensure_library();
  bool font_ready() const {
    return font_resource_ && font_resource_->face && font_resource_->hb_font;
  }

  FT_Library ft_library_{};
  std::unique_ptr<FontResource> font_resource_;
  mutable GlyphCache glyph_cache_;
};

} // namespace nodus::tensors::kpath
