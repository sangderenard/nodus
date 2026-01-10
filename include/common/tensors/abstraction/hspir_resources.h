#pragma once

#include <cstdint>
#include <vector>

namespace nodus::tensors::hspir {

using FrameId = uint16_t;
using FontFaceId = uint16_t;
using DeviceModelId = uint16_t;

struct Transform {
  float tx = 0.0f;
  float ty = 0.0f;
  float tz = 0.0f;
  float qw = 1.0f;
  float qx = 0.0f;
  float qy = 0.0f;
  float qz = 0.0f;
};

struct Frame {
  FrameId id = 0;
  FrameId parent = 0;
  Transform pose;
};

struct FontFace {
  FontFaceId id = 0;
  uint32_t units_per_em = 1000;
};

struct DeviceModel {
  DeviceModelId id = 0;
  uint32_t axis_count = 0;
};

class ResourceTable {
 public:
  FrameId add_frame(Frame frame) {
    frame.id = static_cast<FrameId>(frames_.size());
    frames_.push_back(frame);
    return frame.id;
  }
  FontFaceId add_font(FontFace font) {
    font.id = static_cast<FontFaceId>(fonts_.size());
    fonts_.push_back(font);
    return font.id;
  }
  DeviceModelId add_device(DeviceModel device) {
    device.id = static_cast<DeviceModelId>(devices_.size());
    devices_.push_back(device);
    return device.id;
  }

 private:
  std::vector<Frame> frames_;
  std::vector<FontFace> fonts_;
  std::vector<DeviceModel> devices_;
};

}  // namespace nodus::tensors::hspir
