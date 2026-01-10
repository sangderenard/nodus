#pragma once

#include "common/tensors/abstraction/hspir_resources.h"

#include <vector>

namespace nodus::tensors::hspir {

class ResourceBuilder {
 public:
  ResourceBuilder() = default;

  FrameId add_frame(Frame frame) {
    return resource_table_.add_frame(frame);
  }

  FontFaceId add_font(FontFace font) {
    return resource_table_.add_font(font);
  }

  DeviceModelId add_device(DeviceModel device) {
    return resource_table_.add_device(device);
  }

  ResourceTable finalize() { return std::move(resource_table_); }

 private:
  ResourceTable resource_table_;
};

}  // namespace nodus::tensors::hspir
