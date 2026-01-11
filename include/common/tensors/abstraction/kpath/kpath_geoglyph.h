#pragma once

#include "kpath_shaper.h"

#include <cstdint>
#include <vector>

namespace nodus::tensors::kpath {

// A small set of synthetic glyphs defined by compass-and-straightedge style
// constructions (circles + lines + intersections).
//
// These are intended to be injected into an atlas using Private Use Area
// codepoints (U+E000..), enabling tests/tools to exercise the pipeline
// without relying on font files.

struct GeoGlyph final {
  uint32_t codepoint = 0;
  GlyphOutline outline;
  float advance_x = 0.0f;
  float advance_y = 0.0f;
};

// Returns true if `codepoint` is a supported geoglyph PUA codepoint.
bool is_geoglyph_codepoint(uint32_t codepoint);

// Returns the default set of PUA codepoints implemented by this module.
std::vector<uint32_t> default_geoglyph_codepoints();

// Builds the geometric glyph associated with `codepoint`.
//
// The resulting outline is in an arbitrary, consistent outline-space with
// y-up orientation.
bool make_geoglyph(uint32_t codepoint, GeoGlyph& out);

// Size-controlled variant. `size` is interpreted as the nominal side length
// of the constructed primitive in outline units.
bool make_geoglyph(uint32_t codepoint, float size, GeoGlyph& out);

// Constructs a regular n-gon (n>=3) centered at the origin.
//
// This is a general "program dispenser" for synthetic glyph geometry.
// The outline is emitted as a single closed contour of LineTo segments.
bool make_regular_ngon_glyph(uint32_t codepoint,
                             int sides,
                             float radius,
                             GeoGlyph& out);

// Same as above but does not require a codepoint; a stable synthetic glyph_id
// is generated from `sides`.
bool make_regular_ngon_glyph(int sides,
                             float radius,
                             GeoGlyph& out);

} // namespace nodus::tensors::kpath
