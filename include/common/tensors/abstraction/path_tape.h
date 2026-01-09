#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace nodus::tensors::path_tape {

// Short, descriptive name for the "abstract spatial program" subfeature.
// PathTape describes atlas-backed execution paths as scheduled tape steps.

using AxisIndex = uint32_t;
using StepIndex = uint32_t;
using ChannelIndex = uint32_t;
using TokenId = uint32_t;
using EdgeId = uint32_t;

enum class ChannelKind : uint8_t {
  Kinematic,
  Tool,
  Geometry,
  Scheduling,
  Provenance,
};

// Describes how a channel participates in the tape.  Keep this small so the
// future schema can grow without breaking ABI.
struct ChannelDef {
  std::string name;
  ChannelKind kind = ChannelKind::Geometry;
  uint32_t width_bytes = 4;
};

// Layout information that clients can use to interpret axis/channel spans.
struct TapeLayout {
  AxisIndex axis_count = 0;
  StepIndex step_count = 0;
  ChannelIndex channel_count = 0;
};

// Represents a single token path (atlas edges in order) with optional padding hints.
struct TokenPath {
  TokenId token_id = 0;
  std::vector<EdgeId> edges;
  uint32_t capacity_hint = 0;
};

// Provenance link for each step; keeps atlas/token context close to tape.
struct StepProvenance {
  TokenId token_id = 0;
  EdgeId edge_id = 0;
  StepIndex offset_in_token = 0;
};

// Immutable, SoA tape of per-axis deltas and channel values.
class StepTape {
 public:
  explicit StepTape(TapeLayout layout);
  StepTape(const StepTape&) = delete;
  StepTape& operator=(const StepTape&) = delete;
  StepTape(StepTape&&) noexcept = default;

  TapeLayout layout() const noexcept { return layout_; }

  float axis_delta(AxisIndex axis, StepIndex step) const;
  float channel_value(ChannelIndex channel, StepIndex step) const;

  void set_axis_delta(AxisIndex axis, StepIndex step, float delta);
  void set_channel_value(ChannelIndex channel, StepIndex step, float value);
  void set_provenance(StepIndex step, StepProvenance prov);

  std::span<const float> axis_deltas(AxisIndex axis) const;
  std::span<const float> channel_values(ChannelIndex channel) const;
  std::span<const StepProvenance> provenance() const;

 private:
  TapeLayout layout_;
  std::vector<float> axis_deltas_;  // axis_count * step_count
  std::vector<float> channel_values_;  // channel_count * step_count
  std::vector<StepProvenance> provenance_;  // size = step_count
};

}  // namespace nodus::tensors::path_tape
