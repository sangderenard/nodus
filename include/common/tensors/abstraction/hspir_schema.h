#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nodus::tensors::hspir {

enum class ChannelKind : uint8_t {
  Kinematic,
  Tool,
  GeometrySemantic,
  Scheduling,
  Provenance,
};

enum class Interpolation : uint8_t { Stepped, Continuous };

enum class ChannelScope : uint8_t { PerStep, PerContour, PerGlyph, PerProgram };

struct ChannelDef {
  std::string name;
  ChannelKind kind = ChannelKind::GeometrySemantic;
  uint32_t width_bytes = 4;
  Interpolation interp = Interpolation::Stepped;
  ChannelScope scope = ChannelScope::PerStep;
};

class MetricSchema {
 public:
  explicit MetricSchema(uint32_t id);
  uint32_t id() const noexcept { return id_; }
  void add_channel(ChannelDef def);
  const ChannelDef& channel(size_t idx) const;
  size_t channel_count() const noexcept;

 private:
  uint32_t id_;
  std::vector<ChannelDef> channels_;
};

inline MetricSchema::MetricSchema(uint32_t id) : id_(id) {}

inline void MetricSchema::add_channel(ChannelDef def) {
  channels_.push_back(std::move(def));
}

inline const ChannelDef& MetricSchema::channel(size_t idx) const {
  return channels_.at(idx);
}

inline size_t MetricSchema::channel_count() const noexcept {
  return channels_.size();
}

}  // namespace nodus::tensors::hspir
