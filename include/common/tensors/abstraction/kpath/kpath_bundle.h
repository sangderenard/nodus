#pragma once

#include "kpath_atlas.h"
#include "kpath_kernel.h"
#include "kpath_kinematics.h"
#include "kpath_tape.h"

#include <vector>

namespace nodus::tensors::kpath {

struct ContextKey final {
  uint32_t script{};
  uint32_t language{};
  uint32_t features{};
  DeviceId device{};
  uint32_t tool_id{};
};

struct DeviceModel final {
  DeviceId id{};
};

struct ResourceTable final {
  std::vector<ToolKernel> kernels;
  std::vector<DeviceModel> devices;
  std::vector<ArmatureModel> armatures;
};

struct Program final {
  ProgramId id{};
  TokenId source{};
  ContextKey ctx{};
  StepTape tape;
};

struct ProgramBundle final {
  MetricSchema schema;
  Atlas atlas;
  ResourceTable resources;
  std::vector<Program> programs;
};

class BundleBuilder final {
 public:
  explicit BundleBuilder(MetricSchema schema) : schema_(std::move(schema)) {}

  AtlasBuilder& atlas_builder() noexcept { return atlas_b_; }
  ResourceTable& resources() noexcept { return resources_; }

  ProgramId add_program(Program p) {
    p.id = ProgramId{static_cast<uint32_t>(programs_.size() + 1)};
    programs_.push_back(std::move(p));
    return programs_.back().id;
  }

  ProgramBundle finalize() {
    ProgramBundle b;
    b.schema = std::move(schema_);
    b.atlas = atlas_b_.finalize();
    b.resources = std::move(resources_);
    b.programs = std::move(programs_);
    return b;
  }

 private:
  MetricSchema schema_;
  AtlasBuilder atlas_b_;
  ResourceTable resources_;
  std::vector<Program> programs_;
};

} // namespace nodus::tensors::kpath
