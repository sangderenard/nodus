#pragma once

#include "kpath_ids.h"

#include <cstdint>
#include <vector>

namespace nodus::tensors::kpath {

enum class KernelKind : uint8_t {
  AnalyticDisc,
  AnalyticSphere,
  AnalyticCapsule,
  AnalyticWedge,
  Discrete2D,
  Discrete3D,
  PointCloud
};

struct KernelSample final {
  float x{}, y{}, z{};
  float w{}; // occupancy weight in [0,1]
};

struct DiscreteGrid final {
  uint32_t nx{}, ny{}, nz{};
  float dx{}, dy{}, dz{};
  std::vector<float> w;
};

struct ToolKernel final {
  KernelId id{};
  KernelKind kind{KernelKind::AnalyticDisc};
  float p0{}, p1{}, p2{}, p3{};
  DiscreteGrid grid;
  std::vector<KernelSample> samples;
};

} // namespace nodus::tensors::kpath
