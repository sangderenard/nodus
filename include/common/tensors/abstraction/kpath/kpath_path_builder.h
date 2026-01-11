#pragma once

#include "kpath_shaper.h"

#include <functional>
#include <vector>

namespace nodus::tensors::kpath {

// Lightweight path composer that produces GlyphOutline using the existing
// OutlineOp primitives (including Arc and Sin). Supports chaining and reuse
// without re-encoding raster/fill logic.
class PathBuilder {
public:
  PathBuilder() = default;

  // Reset state and keep storage.
  void clear();

  // Move without drawing.
  PathBuilder& move_to(float x, float y);

  // Straight line.
  PathBuilder& line_to(float x, float y);

  // Quadratic and cubic segments.
  PathBuilder& quad_to(float cx, float cy, float x, float y);
  PathBuilder& cubic_to(float c1x, float c1y, float c2x, float c2y, float x, float y);

  // Circular arc: center (cx, cy), radius r, start angle, sweep angle (radians).
  PathBuilder& arc(float cx, float cy, float radius, float start_angle_rad, float sweep_angle_rad);

  // Sinusoid along the chord to (x, y): amplitude, cycles, phase (radians).
  PathBuilder& sin_to(float x, float y, float amplitude, float cycles, float phase_rad = 0.0f);

  // Append an existing outline (optionally translated).
  PathBuilder& append_outline(const GlyphOutline& outline, float tx = 0.0f, float ty = 0.0f);

  // Close current subpath.
  PathBuilder& close();

  // Warp the current path by sampling its parametric u in [0,1] and applying
  // warp_fn(u, x, y). The result replaces the path with Move/Line segments that
  // can themselves be warped again (idempotent nesting).
  PathBuilder& warp(const std::function<void(float, float&, float&)>& warp_fn,
                    uint32_t samples_per_curve = 32);

  // Finalize into a GlyphOutline; optional glyph_id is stored for metadata use.
  GlyphOutline finalize(uint32_t glyph_id = 0) const;

private:
  void ensure_subpath();

  std::vector<OutlineSegment> segments_;
  float cur_x_ = 0.0f;
  float cur_y_ = 0.0f;
  bool has_subpath_ = false;
};

} // namespace nodus::tensors::kpath
