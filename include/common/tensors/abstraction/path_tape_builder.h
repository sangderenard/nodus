#pragma once

#include "common/tensors/abstraction/path_tape.h"

#include <cstdint>
#include <optional>

namespace nodus::tensors::path_tape {

class StepTapeBuilder {
 public:
  explicit StepTapeBuilder(TapeLayout layout)
      : layout_(layout), tape_(layout) {}

  bool set_axis_delta(AxisIndex axis, StepIndex step, float delta) {
    if (!valid_axis(axis) || !valid_step(step)) return false;
    tape_.set_axis_delta(axis, step, delta);
    return true;
  }

  bool set_channel_value(ChannelIndex channel, StepIndex step, float value) {
    if (!valid_channel(channel) || !valid_step(step)) return false;
    tape_.set_channel_value(channel, step, value);
    return true;
  }

  bool set_provenance(StepIndex step, StepProvenance prov) {
    if (!valid_step(step)) return false;
    tape_.set_provenance(step, prov);
    return true;
  }

  StepTape finalize() { return std::move(tape_); }

 private:
  TapeLayout layout_;
  StepTape tape_;

  bool valid_axis(AxisIndex axis) const { return axis < layout_.axis_count; }
  bool valid_channel(ChannelIndex channel) const {
    return channel < layout_.channel_count;
  }
  bool valid_step(StepIndex step) const { return step < layout_.step_count; }
};

}  // namespace nodus::tensors::path_tape
