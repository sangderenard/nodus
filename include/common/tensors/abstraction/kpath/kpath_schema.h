#pragma once

#include "kpath_ids.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nodus::tensors::kpath {

enum class ChannelKind : uint8_t { Kinematic, Tool, GeometrySemantic, Scheduling, Orientation, Provenance };
enum class ChannelDType : uint8_t { F32, I32, U32 };
enum class ChannelScope : uint8_t { PerStep, PerContour, PerGlyph, PerProgram };
enum class ChannelInterp : uint8_t { Stepped, Continuous };

struct ChannelDef final {
  std::string name;
  ChannelKind kind{};
  ChannelDType dtype{};
  ChannelScope scope{};
  ChannelInterp interp{};
};

class MetricSchema final {
 public:
  explicit MetricSchema(SchemaId id = SchemaId{}) : id_(id) {}

  SchemaId id() const noexcept { return id_; }
  int channel_count() const noexcept { return static_cast<int>(channels_.size()); }
  const ChannelDef& channel(int idx) const { return channels_.at(static_cast<size_t>(idx)); }

  int find_channel(std::string_view name) const noexcept {
    for (size_t i = 0; i < channels_.size(); ++i)
      if (channels_[i].name == name) return static_cast<int>(i);
    return -1;
  }

  void add_channel(ChannelDef def) { channels_.push_back(std::move(def)); }

  static MetricSchema MakeCoreV01();

private:
  SchemaId id_{};
  std::vector<ChannelDef> channels_;
};

inline MetricSchema MetricSchema::MakeCoreV01() {
  MetricSchema schema(SchemaId{1});
  schema.add_channel({"dt", ChannelKind::Kinematic, ChannelDType::F32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"feed_rate", ChannelKind::Kinematic, ChannelDType::F32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"tolerance", ChannelKind::Kinematic, ChannelDType::F32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"interp_mode", ChannelKind::Kinematic, ChannelDType::U32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"tool_mode", ChannelKind::Tool, ChannelDType::U32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"tool_id", ChannelKind::Tool, ChannelDType::U32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"tool_power", ChannelKind::Tool, ChannelDType::F32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"kernel_id", ChannelKind::Tool, ChannelDType::U32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"path_id", ChannelKind::GeometrySemantic, ChannelDType::U32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"contour_id", ChannelKind::GeometrySemantic, ChannelDType::U32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"winding_dir", ChannelKind::GeometrySemantic, ChannelDType::I32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"axis_mask", ChannelKind::Scheduling, ChannelDType::U32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"lock_group", ChannelKind::Scheduling, ChannelDType::U32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"sync_id", ChannelKind::Scheduling, ChannelDType::U32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"quat_w", ChannelKind::Orientation, ChannelDType::F32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"quat_x", ChannelKind::Orientation, ChannelDType::F32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"quat_y", ChannelKind::Orientation, ChannelDType::F32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"quat_z", ChannelKind::Orientation, ChannelDType::F32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"token_id", ChannelKind::Provenance, ChannelDType::U32, ChannelScope::PerStep, ChannelInterp::Stepped});
  schema.add_channel({"token_edge_offset", ChannelKind::Provenance, ChannelDType::U32, ChannelScope::PerStep, ChannelInterp::Stepped});
  return schema;
}

} // namespace nodus::tensors::kpath
