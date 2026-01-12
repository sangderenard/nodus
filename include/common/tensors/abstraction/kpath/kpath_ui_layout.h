#pragma once

#include "common/tensors/abstraction/kpath/kpath_atlas.h"
#include "common/tensors/abstraction/kpath/kpath_pipeline.h"
#include "common/tensors/abstraction/kpath/kpath_program.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo.h"
#include "common/tensors/abstraction/kpath/kpath_shaper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nodus::tensors::kpath {

// UI-oriented helpers that combine relational geometry (RelGlyph) and text layout.

// Create a square panel glyph using relational geometry.
RelGlyph make_rel_square_panel(uint32_t codepoint, float size);

// Create a hex panel glyph. If waviness_amplitude > 0, each side uses a sin-wave
// parametric midpoint to intentionally create slightly bowed sides.
RelGlyph make_rel_hex_panel(uint32_t codepoint, float radius, float waviness_amplitude = 0.0f);

// Compile a relational glyph, append it to an ArmatureProgram at (tx,ty), store
// a translated outline copy, and add it to the AtlasBuilder as a codepoint token.
bool add_relglyph_shape(AtlasBuilder& builder,
                        ArmatureProgram& program,
                        std::vector<GlyphOutline>& outlines,
                        const RelGlyph& glyph,
                        float tx,
                        float ty,
                        const char* label);

// Builds a square panel at (panel_x,panel_y) and places one word per line inside
// it using the supplied Shaper.
bool append_ui_panel_with_text(AtlasBuilder& atlas_builder,
                              ArmatureProgram& program,
                              std::vector<GlyphOutline>& outlines,
                              Shaper& shaper,
                              const std::vector<std::string>& words,
                              float panel_x,
                              float panel_y,
                              float panel_size);

} // namespace nodus::tensors::kpath
