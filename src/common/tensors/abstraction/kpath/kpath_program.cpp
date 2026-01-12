#include "common/tensors/abstraction/kpath/kpath_program.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace nodus::tensors::kpath {

namespace {

[[nodiscard]] uint32_t decode_utf8_codepoint(const std::string& text, size_t& idx) {
  constexpr uint32_t kReplacement = 0xFFFD;
  const size_t len = text.size();
  if (idx >= len) return kReplacement;

  unsigned char lead = static_cast<unsigned char>(text[idx]);
  if (lead < 0x80u) {
    ++idx;
    return lead;
  }

  size_t total_bytes = 0;
  uint32_t cp = 0;
  if ((lead & 0xE0u) == 0xC0u) {
    total_bytes = 2;
    cp = lead & 0x1Fu;
  } else if ((lead & 0xF0u) == 0xE0u) {
    total_bytes = 3;
    cp = lead & 0x0Fu;
  } else if ((lead & 0xF8u) == 0xF0u) {
    total_bytes = 4;
    cp = lead & 0x07u;
  } else {
    ++idx;
    return kReplacement;
  }

  if (idx + total_bytes > len) {
    ++idx;
    return kReplacement;
  }

  for (size_t i = 1; i < total_bytes; ++i) {
    unsigned char cont = static_cast<unsigned char>(text[idx + i]);
    if ((cont & 0xC0u) != 0x80u) {
      ++idx;
      return kReplacement;
    }
    cp = (cp << 6) | (cont & 0x3Fu);
  }

  idx += total_bytes;

  if ((total_bytes == 2 && cp < 0x80u) ||
      (total_bytes == 3 && cp < 0x800u) ||
      (total_bytes == 4 && cp < 0x10000u) ||
      (cp > 0x10FFFFu) ||
      (cp >= 0xD800u && cp <= 0xDFFFu)) {
    return kReplacement;
  }

  return cp;
}

[[nodiscard]] uint32_t decode_utf16_codepoint(std::u16string_view text, size_t& idx) {
  constexpr uint32_t kReplacement = 0xFFFD;
  if (idx >= text.size()) return kReplacement;

  const uint16_t lead = static_cast<uint16_t>(text[idx]);
  ++idx;

  // Not a surrogate.
  if (lead < 0xD800u || lead > 0xDFFFu) return static_cast<uint32_t>(lead);

  // Lone low surrogate.
  if (lead >= 0xDC00u) return kReplacement;

  // High surrogate; must have a following low surrogate.
  if (idx >= text.size()) return kReplacement;
  const uint16_t trail = static_cast<uint16_t>(text[idx]);
  if (trail < 0xDC00u || trail > 0xDFFFu) return kReplacement;
  ++idx;

  const uint32_t hi = static_cast<uint32_t>(lead - 0xD800u);
  const uint32_t lo = static_cast<uint32_t>(trail - 0xDC00u);
  return 0x10000u + ((hi << 10) | lo);
}

[[nodiscard]] bool compute_program_bounds(const ArmatureProgram& program,
                                         float& min_x,
                                         float& min_y,
                                         float& max_x,
                                         float& max_y) {
  if (program.points.empty()) return false;

  min_x = min_y = std::numeric_limits<float>::infinity();
  max_x = max_y = -std::numeric_limits<float>::infinity();

  for (const auto& pt : program.points) {
    min_x = std::min(min_x, pt.x);
    min_y = std::min(min_y, pt.y);
    max_x = std::max(max_x, pt.x);
    max_y = std::max(max_y, pt.y);
  }
  return true;
}

} // namespace

CodepointSequence codepoints_from_utf8(const std::string& utf8_text) {
  CodepointSequence result;
  result.codepoints.reserve(utf8_text.size());
  size_t idx = 0;
  while (idx < utf8_text.size()) {
    result.codepoints.push_back(decode_utf8_codepoint(utf8_text, idx));
  }
  return result;
}

CodepointSequence codepoints_from_utf16(std::u16string_view utf16_text) {
  CodepointSequence result;
  result.codepoints.reserve(utf16_text.size());
  size_t idx = 0;
  while (idx < utf16_text.size()) {
    result.codepoints.push_back(decode_utf16_codepoint(utf16_text, idx));
  }
  return result;
}

bool build_armature_program_from_sequence(Shaper& shaper,
                                          const CodepointSequence& sequence,
                                          ArmatureProgram& out_program,
                                          const ProgramBuildParams& params) {
  if (sequence.codepoints.empty()) return false;
  std::vector<Cluster> clusters;
  if (!shaper.shape(sequence, clusters) || clusters.empty()) return false;

  uint32_t samples = params.samples_per_segment == 0 ? 1 : params.samples_per_segment;
  out_program.points.clear();
  float pen_x = 0.0f;
  float pen_y = 0.0f;

  for (const Cluster& cluster : clusters) {
    if (cluster.glyph_ids.empty()) return false;
    for (uint32_t glyph_id : cluster.glyph_ids) {
      GlyphOutline outline;
      if (!shaper.extract_outline(glyph_id, outline)) return false;
      append_glyph_outline_to_program(out_program, outline, pen_x, pen_y, params.nominal_z, samples);
    }
    pen_x += cluster.advance_x;
    pen_y += cluster.advance_y;
  }

  return !out_program.points.empty();
}

bool build_armature_program_from_utf8(Shaper& shaper,
                                      const std::string& text_utf8,
                                      ArmatureProgram& out_program,
                                      const ProgramBuildParams& params) {
  CodepointSequence seq = codepoints_from_utf8(text_utf8);
  return build_armature_program_from_sequence(shaper, seq, out_program, params);
}

bool rasterize_program_into_minimal_tensor(const ArmatureProgram& program,
                                           TensorCanvas2D& out_texture,
                                           const MachineControlConfig& machine,
                                           const GaussianToolParams& tool,
                                           const ProgramRasterParams& params) {
  if (program.points.empty()) return false;

  // Minimal tensor is defined as a 1:1 program-unit-to-pixel raster with margin.
  // This avoids the implicit fit-scaling that can otherwise vary by aspect/rounding.
  ProgramRasterPlan plan = plan_program_raster(program,
                                               /*pixels_per_unit=*/1.0f,
                                               /*margin_px=*/params.margin,
                                               tool);
  if (plan.width_px == 0 || plan.height_px == 0) return false;

  MachineControlConfig machine_copy = machine;
  machine_copy.energy_per_px = params.deposition_value;

  out_texture.resize(plan.width_px, plan.height_px, 0.0f);
  TensorCanvas2D temp;
  temp.resize(plan.width_px, plan.height_px, 0.0f);
  rasterize_program_gaussian_with_thermal_mapped(program, out_texture, temp, machine_copy, tool, plan.mapping);

  if (params.initialization_value != 0.0f) {
    for (float& v : out_texture.values) {
      v += params.initialization_value;
    }
  }
  return true;
}

} // namespace nodus::tensors::kpath
