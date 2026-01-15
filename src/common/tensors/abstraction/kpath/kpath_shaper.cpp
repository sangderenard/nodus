#include "common/tensors/abstraction/kpath/kpath_shaper.h"

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
#include FT_OUTLINE_H

#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>

namespace nodus::tensors::kpath {

namespace {

float to_float(FT_Pos v) {
  return static_cast<float>(v) / 64.0f;
}

struct OutlineBuilder {
  GlyphOutline* outline{};
  float last_x{};
  float last_y{};

  static int move_to(const FT_Vector* to, void* user) {
    auto* builder = static_cast<OutlineBuilder*>(user);
    float x = to_float(to->x);
    float y = to_float(to->y);
    builder->outline->segments.push_back({OutlineOp::MoveTo, x, y, 0.0f, 0.0f, 0.0f, 0.0f});
    builder->last_x = x;
    builder->last_y = y;
    return 0;
  }

  static int line_to(const FT_Vector* to, void* user) {
    auto* builder = static_cast<OutlineBuilder*>(user);
    float x = to_float(to->x);
    float y = to_float(to->y);
    builder->outline->segments.push_back({OutlineOp::LineTo, builder->last_x, builder->last_y, 0.0f, 0.0f, x, y});
    builder->last_x = x;
    builder->last_y = y;
    return 0;
  }

  static int conic_to(const FT_Vector* control, const FT_Vector* to, void* user) {
    auto* builder = static_cast<OutlineBuilder*>(user);
    float cx = to_float(control->x);
    float cy = to_float(control->y);
    float x = to_float(to->x);
    float y = to_float(to->y);
    builder->outline->segments.push_back({OutlineOp::QuadTo, builder->last_x, builder->last_y, cx, cy, x, y});
    builder->last_x = x;
    builder->last_y = y;
    return 0;
  }

  static int cubic_to(const FT_Vector* control1, const FT_Vector* control2, const FT_Vector* to, void* user) {
    auto* builder = static_cast<OutlineBuilder*>(user);
    float c1x = to_float(control1->x);
    float c1y = to_float(control1->y);
    float c2x = to_float(control2->x);
    float c2y = to_float(control2->y);
    float x = to_float(to->x);
    float y = to_float(to->y);
    builder->outline->segments.push_back({OutlineOp::CubicTo, c1x, c1y, c2x, c2y, x, y});
    builder->last_x = x;
    builder->last_y = y;
    return 0;
  }

  static int close_path(void* user) {
    auto* builder = static_cast<OutlineBuilder*>(user);
    builder->outline->segments.push_back({OutlineOp::Close, builder->last_x, builder->last_y, 0.0f, 0.0f, 0.0f, 0.0f});
    return 0;
  }
};

}

Shaper::FontResource::~FontResource() {
  if (hb_font) {
    hb_font_destroy(hb_font);
    hb_font = nullptr;
  }
  if (face) {
    FT_Done_Face(face);
    face = nullptr;
  }
}

Shaper::Shaper() {
  FT_Init_FreeType(&ft_library_);
}

Shaper::Shaper(Shaper&& other) noexcept
    : ft_library_(other.ft_library_),
      font_resource_(std::move(other.font_resource_)),
      glyph_cache_(std::move(other.glyph_cache_)) {
  other.ft_library_ = nullptr;
}

Shaper::~Shaper() {
  font_resource_.reset();
  if (ft_library_) {
    FT_Done_FreeType(ft_library_);
    ft_library_ = nullptr;
  }
}

Shaper& Shaper::operator=(Shaper&& other) noexcept {
  if (this == &other) return *this;
  font_resource_.reset();
  if (ft_library_) {
    FT_Done_FreeType(ft_library_);
  }
  ft_library_ = other.ft_library_;
  font_resource_ = std::move(other.font_resource_);
  glyph_cache_ = std::move(other.glyph_cache_);
  other.ft_library_ = nullptr;
  return *this;
}

bool Shaper::ensure_library() {
  return ft_library_ != nullptr;
}

bool Shaper::load_font(const std::string& path, float point_size) {
  if (!ensure_library()) return false;
  std::ifstream file(path, std::ios::binary);
  if (!file) return false;
  std::vector<uint8_t> buffer(std::istreambuf_iterator<char>(file), {});
  FT_Face face = nullptr;
  if (FT_New_Memory_Face(ft_library_, buffer.data(), static_cast<FT_Long>(buffer.size()), 0, &face)) return false;
  FT_Set_Char_Size(face, 0, static_cast<FT_F26Dot6>(point_size * 64.0f), 0, 0);

  hb_font_t* hb_font = hb_ft_font_create(face, nullptr);
  if (!hb_font) {
    FT_Done_Face(face);
    return false;
  }

  auto resource = std::make_unique<FontResource>();
  resource->data = std::move(buffer);
  resource->face = face;
  resource->hb_font = hb_font;
  font_resource_ = std::move(resource);
  return true;
}

bool Shaper::shape(const CodepointSequence& seq, std::vector<Cluster>& out_clusters) const {
  if (!font_ready()) {
    out_clusters.clear();
    return false;
  }

  if (seq.codepoints.empty()) {
    out_clusters.clear();
    return true;
  }

  auto* hb_font = font_resource_->hb_font;
  hb_buffer_t* buffer = hb_buffer_create();
  hb_buffer_set_direction(buffer, HB_DIRECTION_LTR);
  hb_buffer_set_script(buffer, HB_SCRIPT_LATIN);
  hb_buffer_set_language(buffer, hb_language_from_string("und", -1));
  hb_buffer_add_utf32(buffer, seq.codepoints.data(), static_cast<int>(seq.codepoints.size()), 0,
                      static_cast<int>(seq.codepoints.size()));
  hb_shape(hb_font, buffer, nullptr, 0);

  unsigned int glyph_count = 0;
  hb_glyph_info_t* infos = hb_buffer_get_glyph_infos(buffer, &glyph_count);
  hb_glyph_position_t* positions = hb_buffer_get_glyph_positions(buffer, &glyph_count);

  std::map<uint32_t, Cluster> clusters;
  const float inv = 1.0f / 64.0f;
  for (unsigned int i = 0; i < glyph_count; ++i) {
    auto& cluster = clusters[infos[i].cluster];
    cluster.glyph_ids.push_back(infos[i].codepoint);
    cluster.advance_x += positions[i].x_advance * inv;
    cluster.advance_y += positions[i].y_advance * inv;
  }

  out_clusters.clear();
  if (clusters.empty()) {
    hb_buffer_destroy(buffer);
    return true;
  }

  uint32_t last_key = 0;
  std::vector<uint32_t> keys;
  keys.reserve(clusters.size());
  for (const auto& kv : clusters) keys.push_back(kv.first);

  std::sort(keys.begin(), keys.end());
  for (size_t idx = 0; idx < keys.size(); ++idx) {
    Cluster cluster = clusters[keys[idx]];
    cluster.start = keys[idx];
    uint32_t next = (idx + 1 < keys.size()) ? keys[idx + 1]
                                             : static_cast<uint32_t>(seq.codepoints.size());
    cluster.count = next > cluster.start ? next - cluster.start : 1;
    out_clusters.push_back(std::move(cluster));
  }

  hb_buffer_destroy(buffer);
  return true;
}

bool Shaper::extract_outline(uint32_t glyph_id, GlyphOutline& out) const {
  if (!font_ready()) return false;

  if (const GlyphOutline* cached = glyph_cache_.find(glyph_id)) {
    out = *cached;
    return true;
  }

  FT_Face face = font_resource_->face;
  if (!face) return false;
  FT_UInt glyph_index = glyph_id;
  if (FT_Load_Glyph(face, glyph_index, FT_LOAD_NO_HINTING | FT_LOAD_NO_BITMAP)) return false;

  GlyphOutline outline;
  outline.glyph_id = glyph_id;
  OutlineBuilder builder{&outline, 0.0f, 0.0f};

  FT_Outline_Funcs funcs{};
  funcs.move_to = OutlineBuilder::move_to;
  funcs.line_to = OutlineBuilder::line_to;
  funcs.conic_to = OutlineBuilder::conic_to;
  funcs.cubic_to = OutlineBuilder::cubic_to;
  funcs.shift = 0;
  funcs.delta = 0;

  if (FT_Outline_Decompose(&face->glyph->outline, &funcs, &builder)) return false;

  outline.segments.push_back({OutlineOp::Close, builder.last_x, builder.last_y, 0.0f, 0.0f, 0.0f, 0.0f});
  glyph_cache_.store_glyph(glyph_id, outline);
  out = std::move(outline);
  return true;
}

} // namespace nodus::tensors::kpath
