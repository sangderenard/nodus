#pragma once

#include "kpath_raster.h"
#include "kpath_shaper.h"

#include <cstdint>
#include <string>

namespace nodus::tensors::kpath {

// Parameters that drive how glyph outlines become armature programs.
struct ProgramBuildParams final {
  float nominal_z = 0.0f;
  uint32_t samples_per_segment = 16;
};

// Parameters that control how an armature program is rasterized into a tensor.
struct ProgramRasterParams final {
  float margin = 8.0f;
  float initialization_value = 0.0f;
  float deposition_value = 1.0f;
};

// Converts a UTF-8 string into the scalar codepoints expected by the shaper.
CodepointSequence codepoints_from_utf8(const std::string& utf8_text);

// Shapes a codepoint sequence and fills an armature program that follows the
// resulting outline, clearing `out_program` before appending.
bool build_armature_program_from_sequence(Shaper& shaper,
                                          const CodepointSequence& sequence,
                                          ArmatureProgram& out_program,
                                          const ProgramBuildParams& params = {});

// Convenience wrapper that decodes a UTF-8 string and builds a matching program.
bool build_armature_program_from_utf8(Shaper& shaper,
                                      const std::string& text_utf8,
                                      ArmatureProgram& out_program,
                                      const ProgramBuildParams& params = {});

// Rasterizes a program into the smallest tensor that still respects the `margin`
// while allowing the caller to pick an initialization baseline and a deposit
// amount (negative values produce dark-on-light results).
bool rasterize_program_into_minimal_tensor(const ArmatureProgram& program,
                                           TensorCanvas2D& out_texture,
                                           const MachineControlConfig& machine,
                                           const GaussianToolParams& tool,
                                           const ProgramRasterParams& params = {});

} // namespace nodus::tensors::kpath
