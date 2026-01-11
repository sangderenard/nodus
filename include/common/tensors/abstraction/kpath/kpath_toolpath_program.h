#pragma once

#include "kpath_atlas.h"
#include "kpath_schema.h"
#include "kpath_tape.h"

#include <cstdint>

namespace nodus::tensors::kpath {

// Opaque-ish heap object intended to be passed via the tool raw stack as a VT_VOID_PTR.
// This represents a pre-device, pre-armature toolpath derived from an atlas token.
struct ToolpathProgram final {
  static constexpr uint32_t kMagic = 0x4B505450u; // 'KPTP'

  uint32_t magic = kMagic;
  uint32_t version = 1;

  MetricSchema schema;
  StepTape tape;
  Atlas atlas;
  TokenId token{};

  ToolpathProgram() = default;
  ToolpathProgram(const ToolpathProgram&) = delete;
  ToolpathProgram& operator=(const ToolpathProgram&) = delete;
  ToolpathProgram(ToolpathProgram&&) noexcept = default;
  ToolpathProgram& operator=(ToolpathProgram&&) noexcept = default;
};

} // namespace nodus::tensors::kpath
