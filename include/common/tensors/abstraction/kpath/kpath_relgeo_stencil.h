#pragma once

#include "common/tensors/abstraction/kpath/kpath_relgeo.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nodus::tensors::kpath {

struct RelGeoEsatResult;

// Qualitative (non-numeric) spatial constraints inferred from RelProgram.
//
// A stencil captures partial ordering and sign information per axis inside
// one or more local frames. Frames can be related (parallel/perpendicular)
// without committing to numeric coordinates.

enum class RelAxisOrder : uint8_t {
  Unknown = 0,
  Less,
  Equal,
  Greater,
};

enum class RelAxisSign : uint8_t {
  Unknown = 0,
  Negative,
  Zero,
  Positive,
};

enum class RelFrameAxisRelation : uint8_t {
  Unknown = 0,
  Parallel,
  Perpendicular,
};

struct RelStencilIssue final {
  enum class Kind : uint8_t {
    Conflict,
  };
  Kind kind = Kind::Conflict;
  std::string message;
};

struct RelGeoStencilFrame final {
  uint32_t id = 0;
  uint32_t dims = 2;
  std::string label;
  RelPointId origin{};
};

struct RelAxisOrderConstraint final {
  uint32_t frame = 0;
  uint32_t axis = 0;
  RelPointId a{};
  RelPointId b{};
  RelAxisOrder order = RelAxisOrder::Unknown;
};

struct RelAxisSignConstraint final {
  uint32_t frame = 0;
  uint32_t axis = 0;
  RelPointId p{};
  RelAxisSign sign = RelAxisSign::Unknown;
};

struct RelBetweenConstraint final {
  uint32_t frame = 0;
  uint32_t axis = 0;
  RelPointId a{};
  RelPointId mid{};
  RelPointId b{};
};

struct RelSegmentRatioConstraint final {
  uint32_t frame = 0;
  uint32_t axis = 0;
  RelPointId a{};
  RelPointId mid{};
  RelPointId b{};
  float ratio = 0.5f; // 0..1 along a->b
};

struct RelFrameAxisConstraint final {
  uint32_t frame_a = 0;
  uint32_t axis_a = 0;
  uint32_t frame_b = 0;
  uint32_t axis_b = 0;
  RelFrameAxisRelation relation = RelFrameAxisRelation::Unknown;
  // -1 = opposite, 0 = unknown, 1 = same direction (only meaningful for Parallel).
  int8_t alignment = 0;
};

struct RelGeoStencilOptions final {
  uint32_t dims = 2;
  bool include_local_frames = true;
  RelRuleContext rule_context{};
};

struct RelGeoStencil final {
  uint32_t global_frame = 0;
  std::vector<RelGeoStencilFrame> frames;
  std::vector<RelAxisOrderConstraint> axis_orders;
  std::vector<RelAxisSignConstraint> axis_signs;
  std::vector<RelBetweenConstraint> betweens;
  std::vector<RelSegmentRatioConstraint> ratios;
  std::vector<RelFrameAxisConstraint> frame_relations;
  std::vector<RelPointId> points;
  std::vector<RelPointId> free_points;
  std::vector<RelStencilIssue> issues;
};

RelGeoStencil relgeo_build_stencil(const RelProgram& program, const RelGeoStencilOptions& options = {});
RelGeoStencil relgeo_build_stencil(const RelProgram& program,
                                   const RelGeoEsatResult& esat,
                                   const RelGeoStencilOptions& options = {});

} // namespace nodus::tensors::kpath
