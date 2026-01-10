#pragma once

#include "kpath_shaper.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace nodus::tensors::kpath {

struct ToolPoint final {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f; // optional depth component (for future 3D tensors)
  bool engaged = true; // false => rapid/travel move (no deposition)
};

struct ArmatureProgram final {
  std::vector<ToolPoint> points;
};

// Beam-based (gimbaled) program: origin+direction for each shot/step.
struct BeamPoint final {
  float ox = 0.0f;
  float oy = 0.0f;
  float oz = 0.0f;
  float dx = 0.0f;
  float dy = 0.0f;
  float dz = -1.0f; // pointing toward -Z by default
  bool engaged = true;
};

struct GimbalProgram final {
  std::vector<BeamPoint> beams;
};

// Models a simplified motion-control configuration.
// This is intentionally feed/energy-centric (like real CNC/robot controllers).
struct MachineControlConfig final {
  // Polyline execution step in image/pixel space.
  // Smaller values produce smoother results but more work.
  float step_px = 0.75f;

  // Constant energy deposited per unit length of travel (in pixel distance).
  // Total energy for a step is energy_per_px * step_length.
  float energy_per_px = 1.0f;

  // --- Optional laser-like thermal guardrails ---
  // If enabled, the executor maintains a "temperature" field and clamps
  // deposition near points that are still hot (cooling not complete).
  bool enable_thermal_guard = false;

  // Travel speed in pixel space. Used to convert motion into dt for cooling.
  float feed_rate_px_per_s = 400.0f;

  // Exponential cooling time constant (seconds). Larger => slower cooling.
  float cooling_tau_s = 0.25f;

  // Converts deposited energy into temperature increase in the local footprint.
  float energy_to_temp = 1.0f;

  // Soft limit. If local temperature exceeds this, deposition is scaled down.
  float max_temp = 5.0f;

  // Z conventions (for future 3D tensors / motion export):
  // safe_z should be < cut_z to represent "lift" away from the material.
  float safe_z = -1.0f;
  float cut_z = 0.0f;
};

struct TensorCanvas2D final {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<float> values;

  TensorCanvas2D() = default;
  TensorCanvas2D(uint32_t w, uint32_t h);

  float& at(uint32_t x, uint32_t y);
  float at(uint32_t x, uint32_t y) const;
  void clear(float v = 0.0f);

  float max_value() const;
  float min_value() const;

  // Converts values into 8-bit grayscale image (row-major, top-down).
  std::vector<uint8_t> to_u8_normalized() const;
};

// Tool parameters for a Gaussian "energy kernel".
// Note: sigma is purely a tool parameter now; z does not modulate thickness.
struct GaussianToolParams final {
  float sigma_px = 2.5f;
};

// Simple calibration summary for a tool kernel.
// This is currently based on the rasterizer's impulse response (single stamp).
// It gives us deterministic "effective width" knobs for spacing/offset decisions.
struct ToolCalibration final {
  float peak_value = 0.0f;
  float total_energy = 0.0f;

  // Radial mean of the impulse response in integer pixel rings.
  // radial_mean[r] corresponds to mean intensity for distances in [r, r+1).
  std::vector<float> radial_mean;

  // Cumulative sum of energy within radius r (inclusive), normalized to [0..1].
  std::vector<float> cumulative_energy_fraction;

  // Returns the smallest radius where mean intensity <= fraction * peak.
  // If fraction <= 0, returns 0. If fraction >= 1, returns 0.
  float radius_at_value_fraction(float fraction_of_peak) const;

  // Returns the smallest radius where cumulative energy >= fraction.
  float radius_at_cumulative_fraction(float fraction) const;
};

// Calibrates the Gaussian tool by stamping an impulse at the canvas center and
// measuring radial falloff. This is deterministic and independent of any path.
ToolCalibration calibrate_gaussian_tool_impulse(uint32_t canvas_size_px,
                                               const GaussianToolParams& tool);

// A shared mapping from outline space to image space so multiple raster passes
// (e.g., RGB channels) can be aligned exactly.
struct ProgramMapping final {
  float min_x = 0.0f;
  float min_y = 0.0f;
  float scale = 1.0f;
  float margin = 0.0f;
};

ProgramMapping compute_program_mapping(const ArmatureProgram& reference_program,
                                       uint32_t canvas_width,
                                       uint32_t canvas_height,
                                       float margin);

// Convert a beam program (origins + directions) into surface hits on a plane
// at z = plane_z. Ignores beams whose direction does not intersect the plane.
// Keeps engagement flag.
bool project_beam_program_to_plane(const GimbalProgram& beam_prog,
                                   float plane_z,
                                   ArmatureProgram& out_program);

// Converts a glyph outline (Move/Line/Quad/Cubic) into a sampled tool-path.
// The returned points are in the outline coordinate system.
ArmatureProgram glyph_outline_to_armature_program(const GlyphOutline& outline,
                                                  float nominal_z = 0.0f,
                                                  uint32_t samples_per_segment = 16);

// Append a translated outline into an existing program.
void append_glyph_outline_to_program(ArmatureProgram& program,
                                     const GlyphOutline& outline,
                                     float tx,
                                     float ty,
                                     float tz = 0.0f,
                                     uint32_t samples_per_segment = 16);

// Rasterizes an armature program into a tensor canvas using a Gaussian toolhead.
// Rasterizes an armature program into a tensor canvas using a Gaussian tool kernel.
// Implements constant energy density (energy per unit length) in image/pixel space.
void rasterize_program_gaussian(const ArmatureProgram& program,
                                TensorCanvas2D& canvas,
                                const MachineControlConfig& machine,
                                const GaussianToolParams& tool,
                                float margin = 8.0f);

// Variant that also returns the final temperature field (if thermal guard enabled).
// When disabled, out_temp is cleared to zeros.
void rasterize_program_gaussian_with_thermal(const ArmatureProgram& program,
                                            TensorCanvas2D& out_energy,
                                            TensorCanvas2D& out_temp,
                                            const MachineControlConfig& machine,
                                            const GaussianToolParams& tool,
                                            float margin = 8.0f);

// Rasterizes using a caller-provided mapping (typically computed from a master
// program). This keeps multiple passes perfectly aligned.
void rasterize_program_gaussian_with_thermal_mapped(const ArmatureProgram& program,
                                                   TensorCanvas2D& out_energy,
                                                   TensorCanvas2D& out_temp,
                                                   const MachineControlConfig& machine,
                                                   const GaussianToolParams& tool,
                                                   const ProgramMapping& mapping);

// Writes a grayscale PNG to disk. Returns false on failure.
bool write_png_grayscale_u8(const std::string& path,
                            uint32_t width,
                            uint32_t height,
                            std::span<const uint8_t> pixels);

// Writes an RGB PNG to disk. pixels is packed RGB (row-major, top-down).
bool write_png_rgb_u8(const std::string& path,
                      uint32_t width,
                      uint32_t height,
                      std::span<const uint8_t> pixels_rgb);

} // namespace nodus::tensors::kpath
