#pragma once

#include "kpath_toolpath_concept.h"

namespace nodus::tensors::kpath {

// Signed area for polyline loops in UV. Returns 0 for non-closed / too-short.
float toolpath_signed_area_uv(const ToolpathSpan& span);

// Computes winding based on signed area (CCW => Pos, CW => Neg).
RotDir toolpath_winding_uv(const ToolpathSpan& span);

// Normalizes closure + winding fields based on geometry. Safe no-op for non-polylines.
void toolpath_refresh_topology(ToolpathSpan& span);

} // namespace nodus::tensors::kpath
