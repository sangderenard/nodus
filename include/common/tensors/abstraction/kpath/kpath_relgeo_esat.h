#pragma once

#include "common/tensors/abstraction/kpath/kpath_relgeo.h"

#include <cstdint>
#include <memory>

namespace nodus::tensors::kpath {

// Minimal symbolic e-saturation result.
//
// This is intentionally small: it proves simple equalities between RelGeo terms
// without introducing any numeric solving or coordinate embedding.
struct RelGeoEsatOptions final {
  int max_iterations = 4;
};

enum class RelGeoEsatTermKind : uint8_t {
  Point,
  Dist,
  CircleRadius,
  ConstLength,
  LineDir,
  LineDirPerp,
  AngleBetweenLines,
  ConstAnglePiFrac,
};

struct RelGeoEsatTerm final {
  RelGeoEsatTermKind kind = RelGeoEsatTermKind::Point;

  // For most terms, payload is packed into these fields.
  // - Point: a = point id
  // - Dist:  a = point id (min), b = point id (max)
  // - CircleRadius: a = circle id
  // - ConstLength: c = micro-units
  // - LineDir: a = line id
  // - LineDirPerp: a = line id
  // - AngleBetweenLines: a = line id (min), b = line id (max)
  // - ConstAnglePiFrac: a = signed numerator, b = denominator (both reduced)
  uint32_t a = 0;
  uint32_t b = 0;
  int64_t c = 0;

  static RelGeoEsatTerm point(RelPointId p);
  static RelGeoEsatTerm dist(RelPointId p, RelPointId q);
  static RelGeoEsatTerm radius(RelCircleId circle);
  static RelGeoEsatTerm const_length(float value);
  static RelGeoEsatTerm dir(RelLineId line);
  static RelGeoEsatTerm dir_perp(RelLineId line);
  static RelGeoEsatTerm angle_between(RelLineId l0, RelLineId l1);
  static RelGeoEsatTerm const_angle_pi(int32_t num, uint32_t den);

  friend bool operator==(const RelGeoEsatTerm& x, const RelGeoEsatTerm& y) {
    return x.kind == y.kind && x.a == y.a && x.b == y.b && x.c == y.c;
  }
};

struct RelGeoEsatResult final {
  bool are_equal(const RelGeoEsatTerm& x, const RelGeoEsatTerm& y) const;

 private:
  friend RelGeoEsatResult relgeo_esaturate(const RelProgram& program, const RelGeoEsatOptions& options);

  // Opaque storage owned by the result.
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

RelGeoEsatResult relgeo_esaturate(const RelProgram& program, const RelGeoEsatOptions& options = {});

} // namespace nodus::tensors::kpath
