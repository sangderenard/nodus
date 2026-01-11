#include "common/tensors/abstraction/kpath/kpath_planar_toolpath.h"

#include <cmath>

namespace nodus::tensors::kpath {

float planar_signed_area(const PlanarSpan& span) {
  if (!span.closed) return 0.0f;
  if (span.points.size() < 3) return 0.0f;

  // Shoelace; accept either explicit closure or implied closure.
  float area2 = 0.0f;
  const size_t n = span.points.size();
  for (size_t i = 1; i < n; ++i) {
    const Vec2& a = span.points[i - 1];
    const Vec2& b = span.points[i];
    area2 += a.x * b.y - b.x * a.y;
  }

  const Vec2& first = span.points.front();
  const Vec2& last = span.points.back();
  if (std::fabs(first.x - last.x) > 1e-5f || std::fabs(first.y - last.y) > 1e-5f) {
    area2 += last.x * first.y - first.x * last.y;
  }

  return 0.5f * area2;
}

RotDir planar_winding(const PlanarSpan& span) {
  if (!span.closed) return RotDir::Zero;
  const float area = planar_signed_area(span);
  if (area > 0.0f) return RotDir::Pos;
  if (area < 0.0f) return RotDir::Neg;
  return RotDir::Zero;
}

} // namespace nodus::tensors::kpath
