#pragma once

#include "common/tensors/abstraction/kpath/kpath_relgeo.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace nodus::tensors::kpath {

// Minimal symbolic e-saturation result.
//
// This is intentionally small: it proves simple equalities between RelGeo terms
// without introducing any numeric solving or coordinate embedding.
struct RelGeoEsatOptions final {
  int max_iterations = 4;
  RelRuleContext rule_context{};
};

enum class RelGeoEsatTermKind : uint8_t {
  Point,
  Dist,
  Dist2,
  CircleRadius,
  ConstLength,
  ConstRational,
  Add,
  Mul,
  Sqrt,
  LineDir,
  LineDirPerp,
  AngleBetweenLines,
  ConstAnglePiFrac,
  AngleAtPoints,
  SinAngle,
  CosAngle,
};

struct RelGeoEsatTerm final {
  RelGeoEsatTermKind kind = RelGeoEsatTermKind::Point;

  // For most terms, payload is packed into these fields.
  // - Point: a = point id
  // - Dist:  a = point id (min), b = point id (max)
  // - Dist2: a = point id (min), b = point id (max)
  // - CircleRadius: a = circle id
  // - ConstLength: c = micro-units
  // - ConstRational: c = signed numerator, b = denominator (both reduced)
  // - Add: a = term id (min), b = term id (max)
  // - Mul: a = term id (min), b = term id (max)
  // - Sqrt: a = term id
  // - LineDir: a = line id
  // - LineDirPerp: a = line id
  // - AngleBetweenLines: a = line id (min), b = line id (max)
  // - ConstAnglePiFrac: c = signed numerator, b = denominator (both reduced)
  // - AngleAtPoints: a = vertex point id, b = arm point id (min), c = arm point id (max)
  // - SinAngle: a = angle term id
  // - CosAngle: a = angle term id
  uint32_t a = 0;
  uint32_t b = 0;
  int64_t c = 0;

  static RelGeoEsatTerm point(RelPointId p);
  static RelGeoEsatTerm dist(RelPointId p, RelPointId q);
  static RelGeoEsatTerm dist2(RelPointId p, RelPointId q);
  static RelGeoEsatTerm radius(RelCircleId circle);
  static RelGeoEsatTerm const_length(float value);
  static RelGeoEsatTerm const_rational(int32_t num, uint32_t den);
  // These constructors take term ids from RelGeoEsatResult::term_id().
  static RelGeoEsatTerm add(uint32_t left_term, uint32_t right_term);
  static RelGeoEsatTerm mul(uint32_t left_term, uint32_t right_term);
  static RelGeoEsatTerm sqrt(uint32_t term);
  static RelGeoEsatTerm dir(RelLineId line);
  static RelGeoEsatTerm dir_perp(RelLineId line);
  static RelGeoEsatTerm angle_between(RelLineId l0, RelLineId l1);
  static RelGeoEsatTerm const_angle_pi(int32_t num, uint32_t den);
  static RelGeoEsatTerm angle_at(RelPointId a, RelPointId v, RelPointId b);
  // These constructors take angle term ids from RelGeoEsatResult::term_id().
  static RelGeoEsatTerm sin(uint32_t angle_term);
  static RelGeoEsatTerm cos(uint32_t angle_term);

  friend bool operator==(const RelGeoEsatTerm& x, const RelGeoEsatTerm& y) {
    return x.kind == y.kind && x.a == y.a && x.b == y.b && x.c == y.c;
  }
};

struct RelGeoEsatResult final {
  bool are_equal(const RelGeoEsatTerm& x, const RelGeoEsatTerm& y) const;
  std::optional<uint32_t> term_id(const RelGeoEsatTerm& term) const;

 private:
  friend RelGeoEsatResult relgeo_esaturate(const RelProgram& program, const RelGeoEsatOptions& options);

  // Opaque storage owned by the result.
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

RelGeoEsatResult relgeo_esaturate(const RelProgram& program, const RelGeoEsatOptions& options = {});

} // namespace nodus::tensors::kpath
