#pragma once

#include "kpath_schema.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace nodus::tensors::kpath {

struct TapeLayout final {
  uint32_t axis_count{};
  uint32_t step_count{};
  SchemaId schema_id{};
};

struct StepProvenance final {
  TokenId token{};
  uint32_t token_edge_offset{};
  uint32_t glyph_id{};
  uint32_t contour_id{};
};

class StepTape final {
 public:
  StepTape() = default;
  constexpr TapeLayout layout() const noexcept { return layout_; }

  std::span<const float> axis_delta(uint32_t axis) const noexcept {
    const size_t S = layout_.step_count;
    const size_t off = static_cast<size_t>(axis) * S;
    return std::span<const float>(axis_delta_.data() + off, S);
  }

  std::span<const float> ch_f32(uint32_t ch) const noexcept { return ch_f32_.at(ch); }
  std::span<const uint32_t> ch_u32(uint32_t ch) const noexcept { return ch_u32_.at(ch); }
  std::span<const int32_t> ch_i32(uint32_t ch) const noexcept { return ch_i32_.at(ch); }

  std::span<const StepProvenance> provenance() const noexcept { return provenance_; }

 private:
  friend class StepTapeBuilder;

  TapeLayout layout_{};
  std::vector<float> axis_delta_;
  std::vector<std::span<const float>> ch_f32_;
  std::vector<std::span<const uint32_t>> ch_u32_;
  std::vector<std::span<const int32_t>> ch_i32_;
  std::vector<float> ch_f32_store_;
  std::vector<uint32_t> ch_u32_store_;
  std::vector<int32_t> ch_i32_store_;
  std::vector<StepProvenance> provenance_;
};

class StepTapeBuilder final {
 public:
  struct ChannelEntry final {
    ChannelDType dtype{};
    uint32_t bucket_idx{};
  };

  StepTapeBuilder(const MetricSchema& schema, uint32_t axis_count, uint32_t step_count)
      : tape_(), schema_(&schema) {
    tape_.layout_ = TapeLayout{axis_count, step_count, schema.id()};
    tape_.axis_delta_.assign(static_cast<size_t>(axis_count) * step_count, 0.0f);
    tape_.provenance_.resize(step_count);

    const int channel_count = schema.channel_count();
    step_count_ = step_count;

    int f32 = 0, u32 = 0, i32 = 0;
    for (int i = 0; i < channel_count; ++i) {
      switch (schema.channel(i).dtype) {
        case ChannelDType::F32: ++f32; break;
        case ChannelDType::U32: ++u32; break;
        case ChannelDType::I32: ++i32; break;
      }
    }

    tape_.ch_f32_store_.assign(static_cast<size_t>(f32) * step_count_, 0.0f);
    tape_.ch_u32_store_.assign(static_cast<size_t>(u32) * step_count_, 0);
    tape_.ch_i32_store_.assign(static_cast<size_t>(i32) * step_count_, 0);

    tape_.ch_f32_.reserve(f32);
    tape_.ch_u32_.reserve(u32);
    tape_.ch_i32_.reserve(i32);

    int f32_idx = 0, u32_idx = 0, i32_idx = 0;
    channel_entries_.reserve(channel_count);
    for (int i = 0; i < channel_count; ++i) {
      switch (schema.channel(i).dtype) {
        case ChannelDType::F32:
          tape_.ch_f32_.emplace_back(tape_.ch_f32_store_.data() + static_cast<size_t>(f32_idx) * step_count_,
                                     step_count_);
          channel_entries_.push_back({ChannelDType::F32, static_cast<uint32_t>(f32_idx)});
          ++f32_idx;
          break;
        case ChannelDType::U32:
          tape_.ch_u32_.emplace_back(tape_.ch_u32_store_.data() + static_cast<size_t>(u32_idx) * step_count_,
                                     step_count_);
          channel_entries_.push_back({ChannelDType::U32, static_cast<uint32_t>(u32_idx)});
          ++u32_idx;
          break;
        case ChannelDType::I32:
          tape_.ch_i32_.emplace_back(tape_.ch_i32_store_.data() + static_cast<size_t>(i32_idx) * step_count_,
                                     step_count_);
          channel_entries_.push_back({ChannelDType::I32, static_cast<uint32_t>(i32_idx)});
          ++i32_idx;
          break;
      }
    }
  }

  void set_axis_delta(uint32_t axis, uint32_t step, float v) {
    if (axis < tape_.layout_.axis_count && step < tape_.layout_.step_count) {
      tape_.axis_delta_[static_cast<size_t>(axis) * tape_.layout_.step_count + step] = v;
    }
  }

  bool set_channel_f32(uint32_t channel, uint32_t step, float value) {
    if (!valid_index(channel, step)) return false;
    const ChannelEntry& entry = channel_entries_[channel];
    if (entry.dtype != ChannelDType::F32) return false;
    tape_.ch_f32_store_[static_cast<size_t>(entry.bucket_idx) * step_count_ + step] = value;
    return true;
  }

  bool set_channel_u32(uint32_t channel, uint32_t step, uint32_t value) {
    if (!valid_index(channel, step)) return false;
    const ChannelEntry& entry = channel_entries_[channel];
    if (entry.dtype != ChannelDType::U32) return false;
    tape_.ch_u32_store_[static_cast<size_t>(entry.bucket_idx) * step_count_ + step] = value;
    return true;
  }

  bool set_channel_i32(uint32_t channel, uint32_t step, int32_t value) {
    if (!valid_index(channel, step)) return false;
    const ChannelEntry& entry = channel_entries_[channel];
    if (entry.dtype != ChannelDType::I32) return false;
    tape_.ch_i32_store_[static_cast<size_t>(entry.bucket_idx) * step_count_ + step] = value;
    return true;
  }

  bool set_channel_f32(std::string_view name, uint32_t step, float value) {
    int idx = schema_ ? schema_->find_channel(name) : -1;
    return idx >= 0 && set_channel_f32(static_cast<uint32_t>(idx), step, value);
  }

  bool set_channel_u32(std::string_view name, uint32_t step, uint32_t value) {
    int idx = schema_ ? schema_->find_channel(name) : -1;
    return idx >= 0 && set_channel_u32(static_cast<uint32_t>(idx), step, value);
  }

  bool set_channel_i32(std::string_view name, uint32_t step, int32_t value) {
    int idx = schema_ ? schema_->find_channel(name) : -1;
    return idx >= 0 && set_channel_i32(static_cast<uint32_t>(idx), step, value);
  }

  void set_provenance(uint32_t step, StepProvenance p) {
    if (step < tape_.layout_.step_count) tape_.provenance_[step] = p;
  }

  StepTape finalize() { return std::move(tape_); }

 private:
  bool valid_index(uint32_t channel, uint32_t step) const {
    return channel < channel_entries_.size() && step < step_count_;
  }

  const MetricSchema* schema_{};
  StepTape tape_;
  uint32_t step_count_{0};
  std::vector<ChannelEntry> channel_entries_;
};

} // namespace nodus::tensors::kpath
