#include "common/tensors/abstraction/kpath/kpath_program.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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

  float min_x = 0.0f;
  float min_y = 0.0f;
  float max_x = 0.0f;
  float max_y = 0.0f;
  if (!compute_program_bounds(program, min_x, min_y, max_x, max_y)) {
    return false;
  }

  float margin = std::max(0.0f, params.margin);
  float span_x = std::max(max_x - min_x, 1.0f);
  float span_y = std::max(max_y - min_y, 1.0f);
  float width_f = span_x + margin * 2.0f;
  float height_f = span_y + margin * 2.0f;

  uint32_t width = static_cast<uint32_t>(std::max(1.0f, std::ceil(width_f)));
  uint32_t height = static_cast<uint32_t>(std::max(1.0f, std::ceil(height_f)));

  ProgramMapping mapping = compute_program_mapping(program, width, height, margin);

  MachineControlConfig machine_copy = machine;
  machine_copy.energy_per_px = params.deposition_value;

  TensorCanvas2D canvas(width, height);
  TensorCanvas2D temp(width, height);
  rasterize_program_gaussian_with_thermal_mapped(program, canvas, temp, machine_copy, tool, mapping);

  if (params.initialization_value != 0.0f) {
    for (float& v : canvas.values) {
      v += params.initialization_value;
    }
  }

  out_texture = std::move(canvas);
  return true;
}

} // namespace nodus::tensors::kpath
