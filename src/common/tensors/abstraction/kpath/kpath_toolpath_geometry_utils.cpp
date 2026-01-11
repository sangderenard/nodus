#include "common/tensors/abstraction/kpath/kpath_toolpath_geometry_utils.h"

#include <cmath>

namespace nodus::tensors::kpath {

static bool nearly_equal(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) <= eps;
}

float toolpath_signed_area_uv(const ToolpathSpan& span) {
  if (span.primitive != ToolpathPrimitiveKind::Polyline) return 0.0f;
  if (!span.semantic.closed) return 0.0f;
  const auto& pts = span.polyline.points;
  if (pts.size() < 3) return 0.0f;

  float area2 = 0.0f;
  for (size_t i = 1; i < pts.size(); ++i) {
    const auto& a = pts[i - 1];
    const auto& b = pts[i];
    area2 += a.x * b.y - b.x * a.y;
  }

  const auto& first = pts.front();
  const auto& last = pts.back();
  if (!nearly_equal(first.x, last.x) || !nearly_equal(first.y, last.y)) {
    area2 += last.x * first.y - first.x * last.y;
  }

  return 0.5f * area2;
}

RotDir toolpath_winding_uv(const ToolpathSpan& span) {
  const float area = toolpath_signed_area_uv(span);
  if (area > 0.0f) return RotDir::Pos;
  if (area < 0.0f) return RotDir::Neg;
  return RotDir::Zero;
}

void toolpath_refresh_topology(ToolpathSpan& span) {
  if (span.primitive != ToolpathPrimitiveKind::Polyline) return;
  const auto& pts = span.polyline.points;
  if (pts.size() >= 2) {
    const auto& first = pts.front();
    const auto& last = pts.back();
    span.semantic.closed = nearly_equal(first.x, last.x) && nearly_equal(first.y, last.y);
  } else {
    span.semantic.closed = false;
  }
  span.semantic.winding = toolpath_winding_uv(span);
}

} // namespace nodus::tensors::kpath
