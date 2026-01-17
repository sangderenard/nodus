#pragma once

#include "kpath_shaper.h"

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/abstract_tensor_pool.h"
#include "common/tensors/abstraction/coo_matrix.h"

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace nodus::tensors::kpath {

struct ScatterTimingBreakdown final {
  double plan_ms = 0.0;
  double clear_ms = 0.0;
  double map_ms = 0.0;
  double deposit_ms = 0.0;
  double temp_state_accum_ms = 0.0;
  double diffusion_prep_ms = 0.0;
  double diffusion_iter_ms = 0.0;
  double diffusion_copyback_ms = 0.0;
  double temp_state_writeback_ms = 0.0;
  double total_ms = 0.0;

  // --- Work/efficiency counters ---
  // Number of points in the input vector program (including travel points).
  uint64_t program_points = 0;
  // Number of resampled engaged visits in image space (sum of per-site visit counts).
  uint64_t plan_moments = 0;
  // Number of unique image-space sites after compression.
  uint64_t unique_sites = 0;
  // Total per-pixel site activations (kernel footprint visits), counting multiplicity.
  uint64_t site_activations = 0;
  // Number of unique pixels touched at least once during deposition.
  uint64_t pixels_touched = 0;
};

struct ToolPoint final {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f; // optional depth component (for future 3D tensors)
  bool engaged = true; // false => rapid/travel move (no deposition)
};

struct ArmatureProgram final {
  std::vector<ToolPoint> points;
};

// Beam-based program: origin+direction for each shot/step.
struct BeamPoint final {
  float ox = 0.0f;
  float oy = 0.0f;
  float oz = 0.0f;
  float dx = 0.0f;
  float dy = 0.0f;
  float dz = -1.0f; // pointing toward -Z by default
  bool engaged = true;
};

struct BeamProgram final {
  std::vector<BeamPoint> beams;
};

enum class BeamFalloffKind : uint8_t {
  Colinear = 0, // linear radial falloff to zero at radius
  Gaussian = 1,
  Airy = 2
};

struct BeamToolParams final {
  BeamFalloffKind falloff = BeamFalloffKind::Airy;
  float radius_px = 2.0f; // used for colinear falloff
  float sigma_px = 1.0f;  // used for Gaussian falloff
  // Airy: either set airy_alpha directly or provide aperture/focal/wavelength.
  float airy_alpha = 0.0f; // alpha = pi * D / (lambda * f)
  float aperture_d = 0.0f; // D
  float focal_length = 0.0f; // f
  float wavelength = 0.0f; // lambda
  float airy_lut_step = 0.0f; // step (px) for LUT sampling
  std::vector<float> airy_lut; // precomputed I(r) for r = i * airy_lut_step
  float axial_falloff = 0.0f; // scale by 1 / (1 + axial_falloff * t^2)
  float energy_per_shot = 1.0f; // overrides machine.energy_per_px when > 0
  float shot_dt = 0.0f; // optional override; 0 => infer from feed_rate_px_per_s
};

// Build an Airy LUT for a radius (px) and step. Uses I(r) = [2 J1(alpha r)/(alpha r)]^2.
std::vector<float> make_airy_lut(float alpha, float radius_px, float step_px);

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

enum class ThermalMaterialPreset : uint8_t {
  SteelThin = 0,
  SteelThick,
  AluminumThin,
  AluminumThick,
  CopperThin,
  CopperThick,
  Custom,
};

struct ThermalSimConfig final {
  // Decay applied to the accumulated temperature field each tick (0..1).
  float decay = 0.95f;
  // Diffusion time step for the Laplacian stencil.
  float diffusion_dt = 0.12f;
  // Number of diffusion iterations per tick.
  uint32_t diffusion_steps = 1;
  // Multiplier applied to incoming energy when accumulating heat.
  float heat_gain = 1.0f;
};

ThermalSimConfig thermal_sim_from_preset(ThermalMaterialPreset preset);

struct TensorCanvas2D final {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<float> values;

  TensorCanvas2D() = default;
  TensorCanvas2D(uint32_t w, uint32_t h);

  // Ensures underlying storage can hold at least w*h elements without changing
  // the logical dimensions.
  void reserve(uint32_t w, uint32_t h);

  // Resize the logical tensor dimensions. Preserves allocated capacity when
  // possible and clears values to `init_value`.
  void resize(uint32_t w, uint32_t h, float init_value = 0.0f);

  float& at(uint32_t x, uint32_t y);
  float at(uint32_t x, uint32_t y) const;
  void clear(float v = 0.0f);

  float max_value() const;
  float min_value() const;

  // Converts values into 8-bit grayscale image (row-major, top-down).
  std::vector<uint8_t> to_u8_normalized() const;
};

struct ProgramBounds final {
  float min_x = 0.0f;
  float min_y = 0.0f;
  float max_x = 0.0f;
  float max_y = 0.0f;
};

// Computes bounds over all points in a program. Returns false if program is empty.
bool compute_program_bounds(const ArmatureProgram& program, ProgramBounds& out_bounds);

// Tool parameters for a Gaussian "energy kernel".
// Note: sigma is purely a tool parameter now; z does not modulate thickness.
struct GaussianToolParams final {
  // Default to an effectively single-pixel tool footprint.
  // The rasterizer treats sufficiently small sigma as a point-stamp fast path.
  float sigma_px = 0.35f;
};

struct ProgramRasterTransform;

// Rasterizes using a configurable spatial kernel (Gaussian, Airy, or colinear).
void rasterize_program_with_kernel_transformed(const ArmatureProgram& program,
                                               TensorCanvas2D& out_energy,
                                               TensorCanvas2D& out_temp,
                                               const MachineControlConfig& machine,
                                               const BeamToolParams& tool,
                                               const ProgramRasterTransform& xform);

// A shared mapping from outline space to image space so multiple raster passes
// (e.g., RGB channels) can be aligned exactly.
struct ProgramMapping final {
  float min_x = 0.0f;
  float min_y = 0.0f;
  float scale = 1.0f;
  float margin = 0.0f;
};

struct ProgramRasterPlan final {
  uint32_t width_px = 0;
  uint32_t height_px = 0;
  ProgramMapping mapping;
};

// Plans a raster output for a program at a caller-chosen scale (pixels per program unit).
// This is intentionally NOT a "fit to existing canvas" mapping: instead it computes
// the canvas size required to cover the program at the given scale.
ProgramRasterPlan plan_program_raster(const ArmatureProgram& program,
                                      float pixels_per_unit,
                                      float margin_px,
                                      const GaussianToolParams& tool);

// A raster plan expressed as a direct affine transform on program points:
//   x_img = x_prog * scale + shift_x
//   y_img = (height_px - 1) - (y_prog * scale + shift_y)
// where (shift_x, shift_y) are chosen based on the refined toolpath polyline.
struct ProgramRasterTransform final {
  uint32_t width_px = 0;
  uint32_t height_px = 0;
  float scale = 1.0f;   // pixels per program unit
  float shift_x = 0.0f; // pixels
  float shift_y = 0.0f; // pixels
};

struct ProgramScatterPlan final {
  uint32_t width_px = 0;
  uint32_t height_px = 0;
  bool compressed = true;
  float energy_per_visit = 0.0f;

  // Sparse boolean mask of image-space pixel indices (x,y) in COO form.
  COOMatrix points;
  bool points_sorted = false;
  // Flat list of linearized pixel indices (y * width + x).
  AbstractTensorPool::PooledTensor flat_points;
  // [N] F32 energy values (counts * energy_per_visit).
  AbstractTensorPool::PooledTensor values;
  // [N] F32 counts per site (optional, same length as values).
  AbstractTensorPool::PooledTensor visit_counts;
};

// Plans output dimensions by first compiling the program into a resampled toolpath
// polyline (using machine.step_px in pixel space), tracking min/max during that
// refinement step, then choosing a tight tensor size with sufficient padding for
// the Gaussian kernel footprint.
ProgramRasterTransform plan_program_raster_transform_refined(const ArmatureProgram& reference_program,
                                                            const MachineControlConfig& machine,
                                                            float pixels_per_unit,
                                                            float margin_px,
                                                            const GaussianToolParams& tool);

// Rasterizes using the provided refined transform. This funnels all toolpaths
// through the same refinement step (image-space polyline resampling) while keeping
// multiple passes aligned (e.g., outline + fill).
void rasterize_program_gaussian_with_thermal_transformed(const ArmatureProgram& program,
                                                        TensorCanvas2D& out_energy,
                                                        TensorCanvas2D& out_temp,
                                                        const MachineControlConfig& machine,
                                                        const GaussianToolParams& tool,
                                                        const ProgramRasterTransform& xform);

using RasterKeyframeCallback = std::function<void(uint32_t frame_index,
                                                  uint32_t frame_total,
                                                  const TensorCanvas2D& energy,
                                                  const TensorCanvas2D& temp)>;

// Rasterize and emit keyframe callbacks during the process.
void rasterize_program_gaussian_with_thermal_transformed_keyframes(const ArmatureProgram& program,
                                                                  TensorCanvas2D& out_energy,
                                                                  TensorCanvas2D& out_temp,
                                                                  const MachineControlConfig& machine,
                                                                  const GaussianToolParams& tool,
                                                                  const ProgramRasterTransform& xform,
                                                                  RasterKeyframeCallback on_keyframe);

// Plan scatter-ready indices/values for a program in image space using xform.
// When compressed, duplicate pixel hits are combined and stored as counts.
bool plan_program_scatter(const ArmatureProgram& program,
                          const MachineControlConfig& machine,
                          const ProgramRasterTransform& xform,
                          ProgramScatterPlan* out_plan,
                          bool compress = true,
                          bool sort_points = true,
                          TensorBackend* backend_override = nullptr);

// New rasterization entry that consumes a boolean mask and prepares the beam tool.
bool rasterize_program_from_mask(const AbstractTensor& mask,
                                 const BeamToolParams& tool,
                                 TensorBackend* backend_override = nullptr);

// Utility: materialize COO point indices as dense [N,2] F32 tensor (x,y).
AbstractTensor coo_points_to_dense_f32(const COOMatrix& points, TensorBackend* backend);

// Rasterize via scatter + kernel bank. kernel_bank is [B,K,C], kernel_ids is [N].
// Optionally apply a diffusion stencil (Laplacian-like) for temp evolution.
bool rasterize_program_scatter_with_kernels(const ArmatureProgram& program,
                                            TensorCanvas2D& out_energy,
                                            TensorCanvas2D& out_temp,
                                            const MachineControlConfig& machine,
                                            const ProgramRasterTransform& xform,
                                            const AbstractTensor& kernel_bank,
                                            const AbstractTensor& kernel_ids,
                                            const AbstractTensor* temp_kernel_ids = nullptr,
                                            const AbstractTensor* diffusion_kernel = nullptr,
                                            uint32_t diffusion_steps = 0,
                                            float diffusion_dt = 1.0f,
                                            float decay = 1.0f,
                                            AbstractTensor* temp_state = nullptr,
                                            TensorBackend* backend_override = nullptr,
                                            ScatterTimingBreakdown* timing = nullptr);

// Scatter mode that splats a spatial kernel for each point (non-histogram kernel).
bool rasterize_program_scatter_with_spatial_kernel(const ArmatureProgram& program,
                                                   TensorCanvas2D& out_energy,
                                                   TensorCanvas2D& out_temp,
                                                   const MachineControlConfig& machine,
                                                   const ProgramRasterTransform& xform,
                                                   const BeamToolParams& tool,
                                                   const AbstractTensor* diffusion_kernel,
                                                   const ThermalSimConfig& sim,
                                                   AbstractTensor* temp_state = nullptr,
                                                   TensorBackend* backend_override = nullptr,
                                                   ScatterTimingBreakdown* timing = nullptr);

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

ProgramMapping compute_program_mapping(const ArmatureProgram& reference_program,
                                       uint32_t canvas_width,
                                       uint32_t canvas_height,
                                       float margin);

// Convert a beam program (origins + directions) into surface hits on a plane
// at z = plane_z. Ignores beams whose direction does not intersect the plane.
// Keeps engagement flag.
bool project_beam_program_to_plane(const BeamProgram& beam_prog,
                                   float plane_z,
                                   ArmatureProgram& out_program);

// Rasterize a beam program onto a plane with thermal (optional).
// Uses a circular tool profile with colinear (linear radial) or Gaussian falloff.
bool rasterize_beam_program_with_thermal(const BeamProgram& beam_prog,
                                        float plane_z,
                                        TensorCanvas2D& out_energy,
                                        TensorCanvas2D& out_temp,
                                        const MachineControlConfig& machine,
                                        const BeamToolParams& tool,
                                        const ProgramRasterTransform& xform);

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

// Writes an RGBA PNG to disk. pixels is packed RGBA (row-major, top-down).
bool write_png_rgba_u8(const std::string& path,
                       uint32_t width,
                       uint32_t height,
                       std::span<const uint8_t> pixels_rgba);

} // namespace nodus::tensors::kpath
