#include "common/tensors/abstraction/kpath/kpath_raster.h"

#include "common/tensors/abstraction/tensor_math.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "common/tensors/abstraction/in_memory_backend.h"

#include <png.h>

namespace nodus::tensors::kpath {

namespace {

constexpr float kEps = 1e-6f;
constexpr float kPi = 3.14159265358979323846f;

static float lerp(float a, float b, float t) { return a + (b - a) * t; }

static inline std::chrono::high_resolution_clock::time_point now_hr() {
  return std::chrono::high_resolution_clock::now();
}

static inline double elapsed_ms(std::chrono::high_resolution_clock::time_point a,
                                std::chrono::high_resolution_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

static ToolPoint eval_quad(const ToolPoint& p0, const ToolPoint& p1, const ToolPoint& p2, float t) {
  float a = 1.0f - t;
  float x = a * a * p0.x + 2.0f * a * t * p1.x + t * t * p2.x;
  float y = a * a * p0.y + 2.0f * a * t * p1.y + t * t * p2.y;
  float z = lerp(p0.z, p2.z, t);
  return ToolPoint{x, y, z};
}

static ToolPoint eval_cubic(const ToolPoint& p0,
                           const ToolPoint& p1,
                           const ToolPoint& p2,
                           const ToolPoint& p3,
                           float t) {
  float a = 1.0f - t;
  float x = a * a * a * p0.x + 3.0f * a * a * t * p1.x + 3.0f * a * t * t * p2.x + t * t * t * p3.x;
  float y = a * a * a * p0.y + 3.0f * a * a * t * p1.y + 3.0f * a * t * t * p2.y + t * t * t * p3.y;
  float z = lerp(p0.z, p3.z, t);
  return ToolPoint{x, y, z};
}

static void bounds_update(float x, float y, float& min_x, float& min_y, float& max_x, float& max_y) {
  min_x = std::min(min_x, x);
  min_y = std::min(min_y, y);
  max_x = std::max(max_x, x);
  max_y = std::max(max_y, y);
}

static float distance2(float ax, float ay, float bx, float by) {
  float dx = bx - ax;
  float dy = by - ay;
  return dx * dx + dy * dy;
}

static float distance(float ax, float ay, float bx, float by) {
  return std::sqrt(distance2(ax, ay, bx, by));
}

static void resample_polyline_equal_arclen_into(const std::vector<ToolPoint>& pts, float step, std::vector<ToolPoint>& out) {
  out.clear();
  if (pts.size() < 2) return;

  step = std::max(step, 0.05f);
  if (out.capacity() < static_cast<size_t>((pts.size() - 1) * 2)) {
    out.reserve(static_cast<size_t>((pts.size() - 1) * 2));
  }

  ToolPoint cur = pts.front();
  out.push_back(cur);

  float carry = 0.0f;
  for (size_t i = 1; i < pts.size(); ++i) {
    ToolPoint next = pts[i];
    float seg_len = distance(cur.x, cur.y, next.x, next.y);
    if (seg_len < kEps) {
      cur = next;
      continue;
    }

    float t0 = 0.0f;
    while (carry + seg_len * (1.0f - t0) >= step) {
      float remain = step - carry;
      float dt = remain / seg_len;
      float t = t0 + dt;
      ToolPoint p;
      p.x = lerp(cur.x, next.x, t);
      p.y = lerp(cur.y, next.y, t);
      p.z = lerp(cur.z, next.z, t);
      p.engaged = cur.engaged && next.engaged;
      out.push_back(p);
      t0 = t;
      carry = 0.0f;
    }

    carry += seg_len * (1.0f - t0);
    cur = next;
  }

  if (distance(out.back().x, out.back().y, pts.back().x, pts.back().y) > 0.5f * step) {
    out.push_back(pts.back());
  }
}

static std::vector<ToolPoint> resample_polyline_equal_arclen(const std::vector<ToolPoint>& pts, float step) {
  std::vector<ToolPoint> out;
  resample_polyline_equal_arclen_into(pts, step, out);
  return out;
}

struct ScatterPlanScratch final {
  std::vector<ToolPoint> img_pts;
  std::vector<ToolPoint> exec_pts;
  std::vector<std::pair<uint32_t, uint32_t>> hits;
  std::vector<std::pair<uint32_t, uint32_t>> sites;
  std::vector<float> counts;

  // Per-frame stamp buffer used for unique pixel counting during kernel deposition.
  // Stored as host memory (not a tensor) because it is purely diagnostic.
  std::vector<uint32_t> pixel_stamp;
  uint32_t pixel_epoch = 1;
};

static AbstractTensorPool& scatter_plan_pool() {
  static thread_local AbstractTensorPool pool([] {
    AbstractTensorPool::Options opt;
    opt.clear_on_release = false;
    // Scatter plan tensor sizes can vary slightly frame-to-frame (e.g., unique hit site count).
    // Without caps, a thread-local pool can grow unbounded and appear as a memory leak.
    opt.max_cached_handles_total = 96;
    opt.max_cached_handles_per_key = 2;
    return opt;
  }());
  return pool;
}

static ScatterPlanScratch& scatter_plan_scratch() {
  static thread_local ScatterPlanScratch scratch;
  return scratch;
}

static bool copy_tensor_to_canvas_f32(const AbstractTensor& tensor, TensorCanvas2D& canvas) {
  if (!tensor.valid()) return false;
  const TensorDesc& desc = tensor.desc();
  if (desc.dtype != TensorDType::F32 || desc.layout != TensorLayout::Dense) return false;
  if (desc.shape.dims.size() != 2) return false;
  const uint32_t height = desc.shape.dims[0];
  const uint32_t width = desc.shape.dims[1];

  auto* mem = dynamic_cast<InMemoryBackend*>(tensor.backend());
  if (!mem) return false;
  void* ptr_v = nullptr;
  size_t bytes = 0;
  if (!mem->map(tensor.handle(), &ptr_v, &bytes)) return false;
  const float* src = static_cast<const float*>(ptr_v);

  if (canvas.width != width || canvas.height != height) {
    canvas.resize(width, height, 0.0f);
  } else {
    canvas.clear(0.0f);
  }
  const size_t count = static_cast<size_t>(width) * height;
  for (size_t i = 0; i < count; ++i) {
    canvas.values[i] = src[i];
  }
  mem->unmap(tensor.handle());
  return true;
}

static AbstractTensor clone_dense_f32(const AbstractTensor& src, TensorBackend* backend) {
  if (!src.valid() || !backend) return {};
  const TensorDesc& desc = src.desc();
  if (desc.dtype != TensorDType::F32 || desc.layout != TensorLayout::Dense) return {};
  AbstractTensor out(desc, backend);
  if (!out.valid()) return {};
  auto* mem = dynamic_cast<InMemoryBackend*>(backend);
  if (!mem) return {};
  void* src_ptr_v = nullptr;
  size_t src_bytes = 0;
  if (!mem->map(src.handle(), &src_ptr_v, &src_bytes)) return {};
  void* dst_ptr_v = nullptr;
  size_t dst_bytes = 0;
  if (!mem->map(out.handle(), &dst_ptr_v, &dst_bytes)) {
    mem->unmap(src.handle());
    return {};
  }
  const size_t copy_bytes = std::min(src_bytes, dst_bytes);
  std::memcpy(dst_ptr_v, src_ptr_v, copy_bytes);
  mem->unmap(src.handle());
  mem->unmap(out.handle());
  return out;
}

static bool add_scaled_inplace_f32(AbstractTensor& dst, const AbstractTensor& add, float scale) {
  return tensor_axpby_f32(dst, 1.0f, add, scale, &dst);
}

static bool scale_inplace_f32(AbstractTensor& dst, float scale) {
  return tensor_axpby_f32(dst, scale, dst, 0.0f, &dst);
}

struct SpatialKernel final {
  int radius = 0;
  std::vector<float> weights; // (2r+1)^2 row-major

  float at(int dx, int dy) const {
    int s = 2 * radius + 1;
    int ix = dx + radius;
    int iy = dy + radius;
    return weights[static_cast<size_t>(iy) * s + ix];
  }
};

static uint32_t gaussian_kernel_radius_px(float sigma_px) {
  const float sigma = std::max(sigma_px, 0.0f);
  if (sigma <= 0.5f) return 0;
  return static_cast<uint32_t>(std::ceil(3.0f * std::max(sigma, 0.25f)));
}

static SpatialKernel make_spatial_kernel_gaussian(float sigma_px) {
  SpatialKernel k;
  // Fast path: treat very small sigma as a point-stamp (single pixel).
  // This avoids kernel allocation and the convolution loops.
  const float sigma = std::max(sigma_px, 0.0f);
  if (sigma <= 0.5f) {
    k.radius = 0;
    k.weights = {1.0f};
    return k;
  }

  // Clamp to a small minimum to keep the discrete kernel stable.
  const float sigma_clamped = std::max(sigma, 0.25f);
  k.radius = static_cast<int>(std::ceil(3.0f * sigma_clamped));
  int s = 2 * k.radius + 1;
  k.weights.resize(static_cast<size_t>(s) * s);

  float inv2 = 1.0f / (2.0f * sigma_clamped * sigma_clamped);
  float sum = 0.0f;
  for (int y = -k.radius; y <= k.radius; ++y) {
    for (int x = -k.radius; x <= k.radius; ++x) {
      float r2 = static_cast<float>(x * x + y * y);
      float w = std::exp(-r2 * inv2);
      k.weights[static_cast<size_t>(y + k.radius) * s + (x + k.radius)] = w;
      sum += w;
    }
  }

  // Normalize so sum(weights) == 1 (discrete approximation of unit-integral kernel).
  sum = std::max(sum, kEps);
  for (float& w : k.weights) w /= sum;
  return k;
}

static SpatialKernel make_spatial_kernel(const BeamToolParams& tool) {
  SpatialKernel k;
  if (tool.falloff == BeamFalloffKind::Gaussian) {
    k = make_spatial_kernel_gaussian(tool.sigma_px);
    return k;
  }

  if (tool.falloff == BeamFalloffKind::Airy) {
    const float radius = std::max(tool.radius_px, 0.0f);
    if (radius <= 0.5f) {
      k.radius = 0;
      k.weights = {1.0f};
      return k;
    }

    float alpha = tool.airy_alpha;
    if (alpha <= 0.0f && tool.aperture_d > 0.0f && tool.focal_length > 0.0f && tool.wavelength > 0.0f) {
      alpha = static_cast<float>(kPi) * tool.aperture_d / (tool.wavelength * tool.focal_length);
    }
    if (alpha <= 0.0f) alpha = 1.0f;

    const bool use_lut = (!tool.airy_lut.empty() && tool.airy_lut_step > 0.0f);
    k.radius = static_cast<int>(std::ceil(radius));
    int s = 2 * k.radius + 1;
    k.weights.resize(static_cast<size_t>(s) * s, 0.0f);

    float sum = 0.0f;
    for (int y = -k.radius; y <= k.radius; ++y) {
      for (int x = -k.radius; x <= k.radius; ++x) {
        float r = std::sqrt(static_cast<float>(x * x + y * y));
        if (r > radius) continue;
        float w = 0.0f;
        if (use_lut) {
          float idx = r / tool.airy_lut_step;
          size_t i0 = static_cast<size_t>(std::floor(idx));
          size_t i1 = std::min(i0 + 1, tool.airy_lut.size() - 1);
          float t = idx - static_cast<float>(i0);
          float v0 = tool.airy_lut[i0];
          float v1 = tool.airy_lut[i1];
          w = v0 + (v1 - v0) * t;
        } else {
          float xarg = alpha * r;
          if (xarg <= kEps) {
            w = 1.0f;
          } else {
            float j1 = static_cast<float>(std::cyl_bessel_j(1, xarg));
            float v = (2.0f * j1 / xarg);
            w = v * v;
          }
        }
        k.weights[static_cast<size_t>(y + k.radius) * s + (x + k.radius)] = w;
        sum += w;
      }
    }
    sum = std::max(sum, kEps);
    for (float& w : k.weights) w /= sum;
    return k;
  }

  const float radius = std::max(tool.radius_px, 0.0f);
  if (radius <= 0.5f) {
    k.radius = 0;
    k.weights = {1.0f};
    return k;
  }

  k.radius = static_cast<int>(std::ceil(radius));
  int s = 2 * k.radius + 1;
  k.weights.resize(static_cast<size_t>(s) * s, 0.0f);

  float sum = 0.0f;
  for (int y = -k.radius; y <= k.radius; ++y) {
    for (int x = -k.radius; x <= k.radius; ++x) {
      float r = std::sqrt(static_cast<float>(x * x + y * y));
      if (r > radius) continue;
      float t = (radius > kEps) ? (1.0f - (r / radius)) : 1.0f;
      float w = std::max(t, 0.0f);
      k.weights[static_cast<size_t>(y + k.radius) * s + (x + k.radius)] = w;
      sum += w;
    }
  }
  sum = std::max(sum, kEps);
  for (float& w : k.weights) w /= sum;
  return k;
}

std::vector<float> make_airy_lut(float alpha, float radius_px, float step_px) {
  std::vector<float> lut;
  if (alpha <= 0.0f || radius_px <= 0.0f || step_px <= 0.0f) return lut;
  const size_t count = static_cast<size_t>(std::ceil(radius_px / step_px)) + 1;
  lut.resize(count, 0.0f);
  for (size_t i = 0; i < count; ++i) {
    float r = static_cast<float>(i) * step_px;
    float xarg = alpha * r;
    if (xarg <= kEps) {
      lut[i] = 1.0f;
    } else {
      float j1 = static_cast<float>(std::cyl_bessel_j(1, xarg));
      float v = (2.0f * j1 / xarg);
      lut[i] = v * v;
    }
  }
  return lut;
}

static void kernel_stamp_energy(TensorCanvas2D& canvas,
                                  const SpatialKernel& kernel,
                                  float cx,
                                  float cy,
                                  float energy) {
  if (kernel.radius == 0) {
    uint32_t x = static_cast<uint32_t>(std::clamp(static_cast<int>(std::floor(cx)), 0, static_cast<int>(canvas.width - 1)));
    uint32_t y = static_cast<uint32_t>(std::clamp(static_cast<int>(std::floor(cy)), 0, static_cast<int>(canvas.height - 1)));
    canvas.at(x, y) += energy;
    return;
  }

  int r = kernel.radius;
  int x0 = static_cast<int>(std::floor(cx)) - r;
  int x1 = static_cast<int>(std::floor(cx)) + r;
  int y0 = static_cast<int>(std::floor(cy)) - r;
  int y1 = static_cast<int>(std::floor(cy)) + r;

  x0 = std::max(x0, 0);
  y0 = std::max(y0, 0);
  x1 = std::min(x1, static_cast<int>(canvas.width - 1));
  y1 = std::min(y1, static_cast<int>(canvas.height - 1));

  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      int dx = x - static_cast<int>(std::floor(cx));
      int dy = y - static_cast<int>(std::floor(cy));
      if (dx < -r || dx > r || dy < -r || dy > r) continue;
      canvas.at(static_cast<uint32_t>(x), static_cast<uint32_t>(y)) += kernel.at(dx, dy) * energy;
    }
  }
}

static void compute_radial_profile(const TensorCanvas2D& canvas,
                                  float cx,
                                  float cy,
                                  std::vector<float>& out_mean,
                                  std::vector<float>& out_cumfrac,
                                  float& out_peak,
                                  float& out_total) {
  out_peak = 0.0f;
  out_total = 0.0f;

  uint32_t w = canvas.width;
  uint32_t h = canvas.height;
  if (w == 0 || h == 0) {
    out_mean.clear();
    out_cumfrac.clear();
    return;
  }

  int cx_i = std::clamp(static_cast<int>(std::floor(cx)), 0, static_cast<int>(w - 1));
  int cy_i = std::clamp(static_cast<int>(std::floor(cy)), 0, static_cast<int>(h - 1));
  out_peak = canvas.at(static_cast<uint32_t>(cx_i), static_cast<uint32_t>(cy_i));

  float max_r = std::min(std::min(cx, cy),
                         std::min(static_cast<float>(w - 1) - cx, static_cast<float>(h - 1) - cy));
  int rmax = std::max(0, static_cast<int>(std::floor(max_r)));
  out_mean.assign(static_cast<size_t>(rmax + 1), 0.0f);
  std::vector<uint32_t> counts(static_cast<size_t>(rmax + 1), 0);
  std::vector<float> sums(static_cast<size_t>(rmax + 1), 0.0f);

  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      float v = canvas.at(x, y);
      out_total += v;
      float dx = static_cast<float>(x) - cx;
      float dy = static_cast<float>(y) - cy;
      float r = std::sqrt(dx * dx + dy * dy);
      int ri = static_cast<int>(std::floor(r));
      if (ri < 0 || ri > rmax) continue;
      sums[static_cast<size_t>(ri)] += v;
      counts[static_cast<size_t>(ri)] += 1;
    }
  }

  for (int r = 0; r <= rmax; ++r) {
    uint32_t c = counts[static_cast<size_t>(r)];
    out_mean[static_cast<size_t>(r)] = (c > 0) ? (sums[static_cast<size_t>(r)] / static_cast<float>(c)) : 0.0f;
  }

  // Cumulative energy fraction within radius r (inclusive).
  out_cumfrac.assign(static_cast<size_t>(rmax + 1), 0.0f);
  float denom = std::max(out_total, kEps);
  float cum = 0.0f;
  for (int r = 0; r <= rmax; ++r) {
    cum += sums[static_cast<size_t>(r)];
    out_cumfrac[static_cast<size_t>(r)] = std::clamp(cum / denom, 0.0f, 1.0f);
  }
}

static void kernel_stamp_energy_and_temp(TensorCanvas2D& energy_canvas,
                                           TensorCanvas2D& temp_canvas,
                                           const SpatialKernel& kernel,
                                           float cx,
                                           float cy,
                                           float energy,
                                           float dt,
                                           float cooling_tau,
                                           float energy_to_temp,
                                           float max_temp) {
  if (kernel.radius == 0) {
    float decay = 1.0f;
    if (cooling_tau > kEps) {
      decay = std::exp(-dt / cooling_tau);
    }

    int cx_i = std::clamp(static_cast<int>(std::floor(cx)), 0, static_cast<int>(temp_canvas.width - 1));
    int cy_i = std::clamp(static_cast<int>(std::floor(cy)), 0, static_cast<int>(temp_canvas.height - 1));
    uint32_t x = static_cast<uint32_t>(cx_i);
    uint32_t y = static_cast<uint32_t>(cy_i);

    float center_temp = temp_canvas.at(x, y) * decay;
    float scale = 1.0f;
    if (max_temp > kEps && center_temp > max_temp) {
      scale = max_temp / center_temp;
      scale = std::clamp(scale, 0.0f, 1.0f);
    }

    float e = energy * scale;
    energy_canvas.at(x, y) += e;

    float prev_t = temp_canvas.at(x, y);
    float new_t = prev_t * decay + e * energy_to_temp;
    temp_canvas.at(x, y) = new_t;
    return;
  }

  int r = kernel.radius;
  int x0 = static_cast<int>(std::floor(cx)) - r;
  int x1 = static_cast<int>(std::floor(cx)) + r;
  int y0 = static_cast<int>(std::floor(cy)) - r;
  int y1 = static_cast<int>(std::floor(cy)) + r;

  x0 = std::max(x0, 0);
  y0 = std::max(y0, 0);
  x1 = std::min(x1, static_cast<int>(energy_canvas.width - 1));
  y1 = std::min(y1, static_cast<int>(energy_canvas.height - 1));

  float decay = 1.0f;
  if (cooling_tau > kEps) {
    decay = std::exp(-dt / cooling_tau);
  }

  // Compute a local center temp after decay to decide scaling.
  int cx_i = std::clamp(static_cast<int>(std::floor(cx)), 0, static_cast<int>(temp_canvas.width - 1));
  int cy_i = std::clamp(static_cast<int>(std::floor(cy)), 0, static_cast<int>(temp_canvas.height - 1));
  float center_temp = temp_canvas.at(static_cast<uint32_t>(cx_i), static_cast<uint32_t>(cy_i)) * decay;
  float scale = 1.0f;
  if (max_temp > kEps && center_temp > max_temp) {
    // Soft clamp: never negative, smoothly approaches 0 with overheating.
    scale = max_temp / center_temp;
    scale = std::clamp(scale, 0.0f, 1.0f);
  }

  float e = energy * scale;
  float t_gain = e * energy_to_temp;

  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      int dx = x - static_cast<int>(std::floor(cx));
      int dy = y - static_cast<int>(std::floor(cy));
      if (dx < -r || dx > r || dy < -r || dy > r) continue;
      float w = kernel.at(dx, dy);

      // Update energy.
      energy_canvas.at(static_cast<uint32_t>(x), static_cast<uint32_t>(y)) += w * e;

      // Update temperature with local cooling + local heat input.
      float prev_t = temp_canvas.at(static_cast<uint32_t>(x), static_cast<uint32_t>(y));
      float new_t = prev_t * decay + w * t_gain;
      temp_canvas.at(static_cast<uint32_t>(x), static_cast<uint32_t>(y)) = new_t;
    }
  }
}

} // namespace

ThermalSimConfig thermal_sim_from_preset(ThermalMaterialPreset preset) {
  ThermalSimConfig cfg{};
  switch (preset) {
    case ThermalMaterialPreset::SteelThin:
      cfg.decay = 0.92f;
      cfg.diffusion_dt = 0.18f;
      cfg.diffusion_steps = 2;
      cfg.heat_gain = 1.0f;
      break;
    case ThermalMaterialPreset::SteelThick:
      cfg.decay = 0.97f;
      cfg.diffusion_dt = 0.10f;
      cfg.diffusion_steps = 1;
      cfg.heat_gain = 1.0f;
      break;
    case ThermalMaterialPreset::AluminumThin:
      cfg.decay = 0.90f;
      cfg.diffusion_dt = 0.22f;
      cfg.diffusion_steps = 2;
      cfg.heat_gain = 0.9f;
      break;
    case ThermalMaterialPreset::AluminumThick:
      cfg.decay = 0.96f;
      cfg.diffusion_dt = 0.12f;
      cfg.diffusion_steps = 1;
      cfg.heat_gain = 0.9f;
      break;
    case ThermalMaterialPreset::CopperThin:
      cfg.decay = 0.88f;
      cfg.diffusion_dt = 0.26f;
      cfg.diffusion_steps = 2;
      cfg.heat_gain = 0.85f;
      break;
    case ThermalMaterialPreset::CopperThick:
      cfg.decay = 0.95f;
      cfg.diffusion_dt = 0.14f;
      cfg.diffusion_steps = 1;
      cfg.heat_gain = 0.85f;
      break;
    case ThermalMaterialPreset::Custom:
    default:
      break;
  }
  return cfg;
}

float ToolCalibration::radius_at_value_fraction(float fraction_of_peak) const {
  if (radial_mean.empty()) return 0.0f;
  if (fraction_of_peak <= 0.0f) return 0.0f;
  if (fraction_of_peak >= 1.0f) return 0.0f;
  float thr = peak_value * fraction_of_peak;
  for (size_t r = 0; r < radial_mean.size(); ++r) {
    if (radial_mean[r] <= thr) return static_cast<float>(r);
  }
  return static_cast<float>(radial_mean.size() - 1);
}

float ToolCalibration::radius_at_cumulative_fraction(float fraction) const {
  if (cumulative_energy_fraction.empty()) return 0.0f;
  float f = std::clamp(fraction, 0.0f, 1.0f);
  for (size_t r = 0; r < cumulative_energy_fraction.size(); ++r) {
    if (cumulative_energy_fraction[r] >= f) return static_cast<float>(r);
  }
  return static_cast<float>(cumulative_energy_fraction.size() - 1);
}

ToolCalibration calibrate_gaussian_tool_impulse(uint32_t canvas_size_px, const GaussianToolParams& tool) {
  ToolCalibration cal;
  uint32_t n = std::max<uint32_t>(canvas_size_px, 32);
  TensorCanvas2D canvas(n, n);
  canvas.clear(0.0f);

  SpatialKernel kernel = make_spatial_kernel_gaussian(tool.sigma_px);
  float cx = 0.5f * static_cast<float>(n - 1);
  float cy = 0.5f * static_cast<float>(n - 1);

  // Unit-energy impulse response in raster space.
  kernel_stamp_energy(canvas, kernel, cx, cy, 1.0f);

  compute_radial_profile(canvas, cx, cy, cal.radial_mean, cal.cumulative_energy_fraction, cal.peak_value, cal.total_energy);
  return cal;
}

ProgramMapping compute_program_mapping(const ArmatureProgram& reference_program,
                                       uint32_t canvas_width,
                                       uint32_t canvas_height,
                                       float margin) {
  ProgramMapping m;
  m.margin = margin;

  if (canvas_width == 0 || canvas_height == 0) return m;
  if (reference_program.points.empty()) return m;

  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  for (const auto& p : reference_program.points) {
    bounds_update(p.x, p.y, min_x, min_y, max_x, max_y);
  }

  float span_x = std::max(max_x - min_x, 1.0f);
  float span_y = std::max(max_y - min_y, 1.0f);

  float target_w = std::max(1.0f, static_cast<float>(canvas_width) - 2.0f * margin);
  float target_h = std::max(1.0f, static_cast<float>(canvas_height) - 2.0f * margin);
  float s = std::min(target_w / span_x, target_h / span_y);

  m.min_x = min_x;
  m.min_y = min_y;
  m.scale = s;
  return m;
}

bool compute_program_bounds(const ArmatureProgram& program, ProgramBounds& out_bounds) {
  if (program.points.empty()) return false;

  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  for (const auto& p : program.points) {
    min_x = std::min(min_x, p.x);
    min_y = std::min(min_y, p.y);
    max_x = std::max(max_x, p.x);
    max_y = std::max(max_y, p.y);
  }

  out_bounds.min_x = min_x;
  out_bounds.min_y = min_y;
  out_bounds.max_x = max_x;
  out_bounds.max_y = max_y;
  return true;
}

ProgramRasterPlan plan_program_raster(const ArmatureProgram& program,
                                      float pixels_per_unit,
                                      float margin_px,
                                      const GaussianToolParams& tool) {
  ProgramRasterPlan plan;
  if (program.points.empty()) return plan;

  ProgramBounds b;
  if (!compute_program_bounds(program, b)) return plan;

  const float s = std::max(pixels_per_unit, 1e-6f);
  const float span_x = std::max(b.max_x - b.min_x, 1.0f);
  const float span_y = std::max(b.max_y - b.min_y, 1.0f);

  // Ensure the kernel footprint stays inside the tensor even when stamping at the margin.
  // For a radius-r kernel, we need >= r+1 pixels of padding to avoid the (w-1) boundary
  // when floor(cx) lands at the last in-bounds pixel.
  const float r = static_cast<float>(gaussian_kernel_radius_px(tool.sigma_px));
  const float margin = std::max(0.0f, margin_px) + r + 1.0f;

  const float width_f = span_x * s + 2.0f * margin;
  const float height_f = span_y * s + 2.0f * margin;

  plan.width_px = static_cast<uint32_t>(std::max(1.0f, std::ceil(width_f)));
  plan.height_px = static_cast<uint32_t>(std::max(1.0f, std::ceil(height_f)));

  plan.mapping.min_x = b.min_x;
  plan.mapping.min_y = b.min_y;
  plan.mapping.scale = s;
  plan.mapping.margin = margin;
  return plan;
}

ProgramRasterTransform plan_program_raster_transform_refined(const ArmatureProgram& reference_program,
                                                            const MachineControlConfig& machine,
                                                            float pixels_per_unit,
                                                            float margin_px,
                                                            const GaussianToolParams& tool) {
  ProgramRasterTransform xform;
  if (reference_program.points.empty()) return xform;

  const float scale = std::max(pixels_per_unit, 1e-6f);
  const uint32_t r_px = gaussian_kernel_radius_px(tool.sigma_px);
  const float pad = std::max(0.0f, margin_px) + static_cast<float>(r_px) + 1.0f;

  std::vector<ToolPoint> scaled;
  scaled.reserve(reference_program.points.size());
  for (const auto& p : reference_program.points) {
    ToolPoint q = p;
    q.x = p.x * scale;
    q.y = p.y * scale;
    scaled.push_back(q);
  }

  // Refine into a pixel-space polyline and track bounds over engaged points.
  std::vector<ToolPoint> exec_pts = resample_polyline_equal_arclen(scaled, machine.step_px);
  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  bool any = false;
  for (const auto& p : exec_pts) {
    if (!p.engaged) continue;
    any = true;
    min_x = std::min(min_x, p.x);
    min_y = std::min(min_y, p.y);
    max_x = std::max(max_x, p.x);
    max_y = std::max(max_y, p.y);
  }
  if (!any) return xform;

  const float span_x = std::max(max_x - min_x, 1.0f);
  const float span_y = std::max(max_y - min_y, 1.0f);

  xform.width_px = static_cast<uint32_t>(std::max(1.0f, std::ceil(span_x + 2.0f * pad)));
  xform.height_px = static_cast<uint32_t>(std::max(1.0f, std::ceil(span_y + 2.0f * pad)));
  xform.scale = scale;
  xform.shift_x = -min_x + pad;
  xform.shift_y = -min_y + pad;
  return xform;
}

void rasterize_program_gaussian_with_thermal_transformed(const ArmatureProgram& program,
                                                        TensorCanvas2D& out_energy,
                                                        TensorCanvas2D& out_temp,
                                                        const MachineControlConfig& machine,
                                                        const GaussianToolParams& tool,
                                                        const ProgramRasterTransform& xform) {
  BeamToolParams kernel_tool{};
  kernel_tool.falloff = BeamFalloffKind::Gaussian;
  kernel_tool.sigma_px = tool.sigma_px;
  rasterize_program_with_kernel_transformed(program, out_energy, out_temp, machine, kernel_tool, xform);
}

void rasterize_program_with_kernel_transformed(const ArmatureProgram& program,
                                               TensorCanvas2D& out_energy,
                                               TensorCanvas2D& out_temp,
                                               const MachineControlConfig& machine,
                                               const BeamToolParams& tool,
                                               const ProgramRasterTransform& xform) {
  if (xform.width_px == 0 || xform.height_px == 0) return;

  if (out_energy.width != xform.width_px || out_energy.height != xform.height_px) {
    out_energy.resize(xform.width_px, xform.height_px, 0.0f);
  } else {
    out_energy.clear(0.0f);
  }
  if (out_temp.width != xform.width_px || out_temp.height != xform.height_px) {
    out_temp.resize(xform.width_px, xform.height_px, 0.0f);
  } else {
    out_temp.clear(0.0f);
  }

  if (program.points.empty()) return;

  SpatialKernel kernel = make_spatial_kernel(tool);

  std::vector<ToolPoint> img_pts;
  img_pts.reserve(program.points.size());
  for (const auto& p : program.points) {
    ToolPoint ip = p;
    const float x = p.x * xform.scale + xform.shift_x;
    const float y_unflipped = p.y * xform.scale + xform.shift_y;
    ip.x = x;
    ip.y = (static_cast<float>(xform.height_px) - 1.0f) - y_unflipped;
    img_pts.push_back(ip);
  }

  std::vector<ToolPoint> exec_pts = resample_polyline_equal_arclen(img_pts, machine.step_px);
  if (exec_pts.size() < 2) return;

  const float step = std::max(machine.step_px, 0.05f);
  const float energy_step = machine.energy_per_px * step;
  const float feed = std::max(machine.feed_rate_px_per_s, 1.0f);
  const float dt = step / feed;

  if (!machine.enable_thermal_guard) {
    bool have_last = false;
    ToolPoint last{};
    for (const auto& p : exec_pts) {
      if (!p.engaged) continue;
      if (have_last && distance2(p.x, p.y, last.x, last.y) < 1e-6f) continue;
      kernel_stamp_energy(out_energy, kernel, p.x, p.y, energy_step);
      last = p;
      have_last = true;
    }
    out_temp.clear(0.0f);
    return;
  }

  bool have_last = false;
  ToolPoint last{};
  for (const auto& p : exec_pts) {
    if (!p.engaged) continue;
    if (have_last && distance2(p.x, p.y, last.x, last.y) < 1e-6f) continue;
    kernel_stamp_energy_and_temp(out_energy,
                                 out_temp,
                                 kernel,
                                 p.x,
                                 p.y,
                                 energy_step,
                                 dt,
                                 std::max(machine.cooling_tau_s, 0.0f),
                                 machine.energy_to_temp,
                                 std::max(machine.max_temp, 0.0f));
    last = p;
    have_last = true;
  }
}

AbstractTensor coo_points_to_dense_f32(const COOMatrix& points, TensorBackend* backend) {
  if (!points.valid() || points.layout != CooIndexLayout::RowMajor) return {};
  if (points.shape.rank() != 2) return {};
  const uint32_t nnz = points.nnz();
  if (nnz == 0) return {};

  TensorDesc pts_desc{};
  pts_desc.dtype = TensorDType::F32;
  pts_desc.layout = TensorLayout::Dense;
  pts_desc.shape.dims = {nnz, 2u};
  AbstractTensor pts = AbstractTensor::create(pts_desc, backend);
  if (!pts.valid()) return {};

  auto* mem = dynamic_cast<InMemoryBackend*>(backend);
  if (!mem) return {};

  void* idx_ptr_v = nullptr;
  size_t idx_bytes = 0;
  if (!mem->map(points.indices.handle(), &idx_ptr_v, &idx_bytes)) return {};
  void* pts_ptr_v = nullptr;
  size_t pts_bytes = 0;
  if (!mem->map(pts.handle(), &pts_ptr_v, &pts_bytes)) {
    mem->unmap(points.indices.handle());
    return {};
  }

  const TensorDesc& idx_desc = points.indices.desc();
  const uint32_t rank = points.shape.rank();
  if (idx_desc.shape.dims.size() != 2 || idx_desc.shape.dims[0] != nnz || idx_desc.shape.dims[1] != rank) {
    mem->unmap(points.indices.handle());
    mem->unmap(pts.handle());
    return {};
  }

  auto read_coord = [&](uint32_t i, uint32_t d) -> uint64_t {
    const size_t idx = static_cast<size_t>(i) * rank + d;
    switch (idx_desc.dtype) {
      case TensorDType::I32: return static_cast<uint64_t>(static_cast<int32_t*>(idx_ptr_v)[idx]);
      case TensorDType::I64: return static_cast<uint64_t>(static_cast<int64_t*>(idx_ptr_v)[idx]);
      case TensorDType::U32: return static_cast<uint64_t>(static_cast<uint32_t*>(idx_ptr_v)[idx]);
      case TensorDType::U64: return static_cast<uint64_t>(static_cast<uint64_t*>(idx_ptr_v)[idx]);
      default: return 0;
    }
  };

  auto* out_ptr = static_cast<float*>(pts_ptr_v);
  for (uint32_t i = 0; i < nnz; ++i) {
    const uint64_t x = read_coord(i, 1);
    const uint64_t y = read_coord(i, 0);
    out_ptr[i * 2 + 0] = static_cast<float>(x);
    out_ptr[i * 2 + 1] = static_cast<float>(y);
  }

  mem->unmap(points.indices.handle());
  mem->unmap(pts.handle());
  return pts;
}

bool plan_program_scatter(const ArmatureProgram& program,
                          const MachineControlConfig& machine,
                          const ProgramRasterTransform& xform,
                          ProgramScatterPlan* out_plan,
                          bool compress,
                          bool sort_points,
                          TensorBackend* backend_override) {
  if (!out_plan) return false;
  out_plan->points = {};
  out_plan->points_sorted = false;
  out_plan->flat_points = {};
  out_plan->values = {};
  out_plan->visit_counts = {};
  out_plan->width_px = xform.width_px;
  out_plan->height_px = xform.height_px;
  if (machine.enable_thermal_guard) compress = false;
  out_plan->compressed = compress;

  if (program.points.empty()) return false;
  if (xform.width_px == 0 || xform.height_px == 0) return false;

  ScatterPlanScratch& scratch = scatter_plan_scratch();
  scratch.img_pts.clear();
  if (scratch.img_pts.capacity() < program.points.size()) {
    scratch.img_pts.reserve(program.points.size());
  }
  for (const auto& p : program.points) {
    ToolPoint ip = p;
    const float x = p.x * xform.scale + xform.shift_x;
    const float y_unflipped = p.y * xform.scale + xform.shift_y;
    ip.x = x;
    ip.y = (static_cast<float>(xform.height_px) - 1.0f) - y_unflipped;
    scratch.img_pts.push_back(ip);
  }

  resample_polyline_equal_arclen_into(scratch.img_pts, machine.step_px, scratch.exec_pts);
  if (scratch.exec_pts.size() < 2) return false;

  const float step = std::max(machine.step_px, 0.05f);
  const float energy_step = machine.energy_per_px * step;
  out_plan->energy_per_visit = energy_step;

  scratch.hits.clear();
  if (scratch.hits.capacity() < scratch.exec_pts.size()) {
    scratch.hits.reserve(scratch.exec_pts.size());
  }

  bool have_last = false;
  ToolPoint last{};
  for (const auto& p : scratch.exec_pts) {
    if (!p.engaged) continue;
    if (have_last && distance2(p.x, p.y, last.x, last.y) < 1e-6f) continue;
    const int64_t xi = static_cast<int64_t>(std::llround(static_cast<double>(p.x)));
    const int64_t yi = static_cast<int64_t>(std::llround(static_cast<double>(p.y)));
    if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(xform.width_px) ||
        yi >= static_cast<int64_t>(xform.height_px)) {
      last = p;
      have_last = true;
      continue;
    }
    scratch.hits.emplace_back(static_cast<uint32_t>(xi), static_cast<uint32_t>(yi));
    last = p;
    have_last = true;
  }

  if (scratch.hits.empty()) return false;

  scratch.sites.clear();
  scratch.counts.clear();
  if (compress) {
    std::sort(scratch.hits.begin(), scratch.hits.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second) return a.second < b.second;
      return a.first < b.first;
    });
    scratch.sites.reserve(scratch.hits.size());
    scratch.counts.reserve(scratch.hits.size());
    size_t i = 0;
    while (i < scratch.hits.size()) {
      const auto site = scratch.hits[i];
      size_t j = i + 1;
      while (j < scratch.hits.size() && scratch.hits[j] == site) ++j;
      scratch.sites.push_back(site);
      scratch.counts.push_back(static_cast<float>(j - i));
      i = j;
    }
  } else {
    scratch.sites.reserve(scratch.hits.size());
    scratch.sites.insert(scratch.sites.end(), scratch.hits.begin(), scratch.hits.end());
    scratch.counts.resize(scratch.hits.size());
    std::fill(scratch.counts.begin(), scratch.counts.end(), 1.0f);
  }

  TensorBackend* backend = backend_override ? backend_override : &in_memory_backend_singleton();
  if (!backend) return false;
  auto* mem = dynamic_cast<InMemoryBackend*>(backend);
  if (!mem) return false;

  const uint32_t n = static_cast<uint32_t>(scratch.sites.size());

  TensorShape mask_shape{};
  mask_shape.dims = {xform.height_px, xform.width_px};
  COOMatrix points = COOMatrix::create(mask_shape, n, TensorDType::Bool, backend, TensorDType::U32, CooIndexLayout::RowMajor);
  if (!points.valid()) return false;

  TensorIndexSpec spec; spec.dims = {points.indices.handle()};
  points.values.set_slice(spec, points.values);

  TensorDesc val_desc{};
  val_desc.dtype = TensorDType::F32;
  val_desc.layout = TensorLayout::Dense;
  val_desc.shape.dims = {n};
  auto values = scatter_plan_pool().acquire(val_desc, backend);
  if (!values.valid()) return false;

  TensorDesc flat_desc{};
  flat_desc.dtype = TensorDType::U32;
  flat_desc.layout = TensorLayout::Dense;
  flat_desc.shape.dims = {n};
  auto flat_points = scatter_plan_pool().acquire(flat_desc, backend);
  if (!flat_points.valid()) return false;

  auto visit_counts = scatter_plan_pool().acquire(val_desc, backend);
  if (!visit_counts.valid()) return false;

  void* val_ptr_v = nullptr;
  size_t val_bytes = 0;
  if (!mem->map(values->handle(), &val_ptr_v, &val_bytes)) {
    return false;
  }
  float* val_ptr = static_cast<float*>(val_ptr_v);

  void* flat_ptr_v = nullptr;
  size_t flat_bytes = 0;
  if (!mem->map(flat_points->handle(), &flat_ptr_v, &flat_bytes)) {
    mem->unmap(values->handle());
    return false;
  }
  auto* flat_ptr = static_cast<uint32_t*>(flat_ptr_v);

  void* cnt_ptr_v = nullptr;
  size_t cnt_bytes = 0;
  if (!mem->map(visit_counts->handle(), &cnt_ptr_v, &cnt_bytes)) {
    mem->unmap(values->handle());
    mem->unmap(flat_points->handle());
    return false;
  }
  float* cnt_ptr = static_cast<float*>(cnt_ptr_v);

  void* idx_ptr_v = nullptr;
  size_t idx_bytes = 0;
  if (!mem->map(points.indices.handle(), &idx_ptr_v, &idx_bytes)) {
    mem->unmap(values->handle());
    mem->unmap(visit_counts->handle());
    return false;
  }
  void* val_mask_v = nullptr;
  size_t val_mask_bytes = 0;
  if (!mem->map(points.values.handle(), &val_mask_v, &val_mask_bytes)) {
    mem->unmap(points.indices.handle());
    mem->unmap(values->handle());
    mem->unmap(flat_points->handle());
    mem->unmap(visit_counts->handle());
    return false;
  }
  auto* idx_ptr = static_cast<uint32_t*>(idx_ptr_v);
  auto* mask_ptr = static_cast<uint8_t*>(val_mask_v);

  for (uint32_t i = 0; i < n; ++i) {
    idx_ptr[i * 2 + 0] = scratch.sites[i].second;
    idx_ptr[i * 2 + 1] = scratch.sites[i].first;
    mask_ptr[i] = 1u;
    cnt_ptr[i] = scratch.counts[i];
    val_ptr[i] = scratch.counts[i] * energy_step;
    flat_ptr[i] = static_cast<uint32_t>(scratch.sites[i].second) * xform.width_px + static_cast<uint32_t>(scratch.sites[i].first);
  }

  mem->unmap(points.values.handle());
  mem->unmap(points.indices.handle());
  mem->unmap(values->handle());
  mem->unmap(flat_points->handle());
  mem->unmap(visit_counts->handle());

  if (sort_points) {
    out_plan->points_sorted = points.sort_indices(true);
  }

  out_plan->points = std::move(points);
  out_plan->values = std::move(values);
  out_plan->visit_counts = std::move(visit_counts);
  out_plan->flat_points = std::move(flat_points);
  return true;
}

bool rasterize_program_from_mask(const AbstractTensor& mask,
                                 const BeamToolParams& tool,
                                 TensorBackend* /*backend_override*/) {
  (void)mask;
  SpatialKernel kernel = make_spatial_kernel(tool);
  (void)kernel;
  return true;
}

bool rasterize_program_scatter_with_kernels(const ArmatureProgram& program,
                                            TensorCanvas2D& out_energy,
                                            TensorCanvas2D& out_temp,
                                            const MachineControlConfig& machine,
                                            const ProgramRasterTransform& xform,
                                            const AbstractTensor& kernel_bank,
                                            const AbstractTensor& kernel_ids,
                                            const AbstractTensor* temp_kernel_ids,
                                            const AbstractTensor* diffusion_kernel,
                                            uint32_t diffusion_steps,
                                            float diffusion_dt,
                                            float decay,
                                            AbstractTensor* temp_state,
                                            TensorBackend* backend_override,
                                            ScatterTimingBreakdown* timing) {
  if (timing) *timing = {};
  const auto t_total_start = now_hr();

  if (xform.width_px == 0 || xform.height_px == 0) return false;

  ProgramScatterPlan plan;
  {
    const auto t0 = now_hr();
    if (!plan_program_scatter(program, machine, xform, &plan, true, true, backend_override)) return false;
    const auto t1 = now_hr();
    if (timing) timing->plan_ms += elapsed_ms(t0, t1);
  }

  TensorBackend* backend = backend_override ? backend_override : &in_memory_backend_singleton();
  if (!backend) return false;

  AbstractTensor dense_points = coo_points_to_dense_f32(plan.points, backend);
  if (!dense_points.valid()) return false;

  TensorDesc base_desc{};
  base_desc.dtype = TensorDType::F32;
  base_desc.layout = TensorLayout::Dense;
  base_desc.shape.dims = {xform.height_px, xform.width_px};
  auto base = scatter_plan_pool().acquire(base_desc, backend);
  if (!base.valid()) return false;
  auto* mem = dynamic_cast<InMemoryBackend*>(backend);
  if (!mem) return false;
  void* base_ptr_v = nullptr;
  size_t base_bytes = 0;
  {
    const auto t0 = now_hr();
    if (!mem->map(base->handle(), &base_ptr_v, &base_bytes)) return false;
    std::memset(base_ptr_v, 0, base_bytes);
    mem->unmap(base->handle());
    const auto t1 = now_hr();
    if (timing) timing->clear_ms += elapsed_ms(t0, t1);
  }

  AbstractTensor energy;
  {
    const auto t0 = now_hr();
    if (!tensor_scatter_add_2d_kernel_bank_f32(base.tensor(),
                           dense_points,
                                               plan.values.tensor(),
                                               kernel_bank,
                                               kernel_ids,
                                               &energy,
                                               true)) {
      return false;
    }
    const auto t1 = now_hr();
    if (timing) timing->deposit_ms += elapsed_ms(t0, t1);
  }

  {
    const auto t0 = now_hr();
    if (!copy_tensor_to_canvas_f32(energy, out_energy)) return false;
    const auto t1 = now_hr();
    if (timing) timing->diffusion_copyback_ms += elapsed_ms(t0, t1);
  }

  AbstractTensor temp_field;
  if (temp_kernel_ids) {
    const auto t0 = now_hr();
    if (!tensor_scatter_add_2d_kernel_bank_f32(base.tensor(),
                           dense_points,
                                               plan.values.tensor(),
                                               kernel_bank,
                                               *temp_kernel_ids,
                                               &temp_field,
                                               true)) {
      return false;
    }
    const auto t1 = now_hr();
    if (timing) timing->deposit_ms += elapsed_ms(t0, t1);
  } else {
    temp_field = clone_dense_f32(energy, backend);
    if (!temp_field.valid()) return false;
  }

  if (machine.energy_to_temp != 1.0f) {
    const auto t0 = now_hr();
    if (!scale_inplace_f32(temp_field, machine.energy_to_temp)) return false;
    const auto t1 = now_hr();
    if (timing) timing->temp_state_accum_ms += elapsed_ms(t0, t1);
  }

  if (temp_state) {
    if (!temp_state->valid()) {
      *temp_state = clone_dense_f32(temp_field, backend);
      if (!temp_state->valid()) return false;
    }
    if (decay > 0.0f && decay < 1.0f) {
      const auto t0 = now_hr();
      if (!scale_inplace_f32(*temp_state, decay)) return false;
      const auto t1 = now_hr();
      if (timing) timing->temp_state_accum_ms += elapsed_ms(t0, t1);
    }
    {
      const auto t0 = now_hr();
      if (!add_scaled_inplace_f32(*temp_state, temp_field, 1.0f)) return false;
      const auto t1 = now_hr();
      if (timing) timing->temp_state_accum_ms += elapsed_ms(t0, t1);
    }
    temp_field = AbstractTensor::wrap(temp_state->handle(), temp_state->desc(), temp_state->backend(), false);
  } else {
    if (decay > 0.0f && decay < 1.0f) {
      const auto t0 = now_hr();
      if (!scale_inplace_f32(temp_field, decay)) return false;
      const auto t1 = now_hr();
      if (timing) timing->temp_state_accum_ms += elapsed_ms(t0, t1);
    }
  }

  if (diffusion_kernel && diffusion_steps > 0) {
    TensorDesc lap_desc = temp_field.desc();
    auto lap_tensor = scatter_plan_pool().acquire(lap_desc, backend);
    if (!lap_tensor.valid()) return false;
    for (uint32_t i = 0; i < diffusion_steps; ++i) {
      const auto t0 = now_hr();
      if (!tensor_apply_stencil_2d_f32_into(temp_field, *diffusion_kernel, &lap_tensor.tensor())) return false;
      if (!add_scaled_inplace_f32(temp_field, lap_tensor.tensor(), diffusion_dt)) return false;
      const auto t1 = now_hr();
      if (timing) timing->diffusion_iter_ms += elapsed_ms(t0, t1);
    }
  }

  {
    const auto t0 = now_hr();
    if (!copy_tensor_to_canvas_f32(temp_field, out_temp)) return false;
    const auto t1 = now_hr();
    if (timing) timing->diffusion_copyback_ms += elapsed_ms(t0, t1);
  }

  if (timing) timing->total_ms = elapsed_ms(t_total_start, now_hr());
  return true;
}

bool rasterize_program_scatter_with_spatial_kernel(const ArmatureProgram& program,
                                                   TensorCanvas2D& out_energy,
                                                   TensorCanvas2D& out_temp,
                                                   const MachineControlConfig& machine,
                                                   const ProgramRasterTransform& xform,
                                                   const BeamToolParams& tool,
                                                   const AbstractTensor* diffusion_kernel,
                                                   const ThermalSimConfig& sim,
                                                   AbstractTensor* temp_state,
                                                   TensorBackend* backend_override,
                                                   ScatterTimingBreakdown* timing) {
  if (timing) *timing = {};
  const auto t_total_start = now_hr();

  if (xform.width_px == 0 || xform.height_px == 0) return false;

  ProgramScatterPlan plan;
  {
    const auto t0 = now_hr();
    if (!plan_program_scatter(program, machine, xform, &plan, true, true, backend_override)) return false;
    const auto t1 = now_hr();
    if (timing) timing->plan_ms += elapsed_ms(t0, t1);
  }

  if (timing) {
    timing->program_points = static_cast<uint64_t>(program.points.size());
  }

  {
    const auto t0 = now_hr();
  if (out_energy.width != xform.width_px || out_energy.height != xform.height_px) {
    out_energy.resize(xform.width_px, xform.height_px, 0.0f);
  } else {
    out_energy.clear(0.0f);
  }
  if (out_temp.width != xform.width_px || out_temp.height != xform.height_px) {
    out_temp.resize(xform.width_px, xform.height_px, 0.0f);
  } else {
    out_temp.clear(0.0f);
  }
    const auto t1 = now_hr();
    if (timing) timing->clear_ms += elapsed_ms(t0, t1);
  }

  TensorBackend* backend = backend_override ? backend_override : &in_memory_backend_singleton();
  if (!backend) return false;
  auto* mem = dynamic_cast<InMemoryBackend*>(backend);
  if (!mem) return false;

  AbstractTensor dense_points = coo_points_to_dense_f32(plan.points, backend);
  if (!dense_points.valid()) return false;

  const TensorDesc& pts_desc = dense_points.desc();
  const TensorDesc& val_desc = plan.values->desc();
  if (pts_desc.dtype != TensorDType::F32 || val_desc.dtype != TensorDType::F32) return false;
  if (pts_desc.layout != TensorLayout::Dense || val_desc.layout != TensorLayout::Dense) return false;
  if (pts_desc.shape.dims.size() != 2 || pts_desc.shape.dims[1] != 2) return false;
  if (val_desc.shape.dims.size() != 1 || val_desc.shape.dims[0] != pts_desc.shape.dims[0]) return false;

  void* pts_ptr_v = nullptr;
  size_t pts_bytes = 0;
  const auto t_map_start = now_hr();
  if (!mem->map(dense_points.handle(), &pts_ptr_v, &pts_bytes)) return false;
  void* val_ptr_v = nullptr;
  size_t val_bytes = 0;
  if (!mem->map(plan.values->handle(), &val_ptr_v, &val_bytes)) {
    mem->unmap(dense_points.handle());
    return false;
  }
  void* cnt_ptr_v = nullptr;
  size_t cnt_bytes = 0;
  if (!mem->map(plan.visit_counts->handle(), &cnt_ptr_v, &cnt_bytes)) {
    mem->unmap(dense_points.handle());
    mem->unmap(plan.values->handle());
    return false;
  }
  const auto t_map_end = now_hr();
  if (timing) timing->map_ms += elapsed_ms(t_map_start, t_map_end);

  const float* pts = static_cast<const float*>(pts_ptr_v);
  const float* vals = static_cast<const float*>(val_ptr_v);
  const float* counts = static_cast<const float*>(cnt_ptr_v);
  const SpatialKernel kernel = make_spatial_kernel(tool);
  const float energy_to_temp = machine.energy_to_temp * sim.heat_gain;

  const uint32_t n = pts_desc.shape.dims[0];

  uint64_t plan_moments = 0;
  if (timing) {
    timing->unique_sites = static_cast<uint64_t>(n);
  }

  struct KernelOffset {
    int dx = 0;
    int dy = 0;
    float w = 0.0f;
  };
  std::vector<KernelOffset> offsets;
  if (kernel.radius == 0) {
    offsets.push_back(KernelOffset{0, 0, 1.0f});
  } else {
    const int r = kernel.radius;
    offsets.reserve(static_cast<size_t>((2 * r + 1) * (2 * r + 1)));
    for (int dy = -r; dy <= r; ++dy) {
      for (int dx = -r; dx <= r; ++dx) {
        const float w = kernel.at(dx, dy);
        if (w == 0.0f) continue;
        offsets.push_back(KernelOffset{dx, dy, w});
      }
    }
  }

  // Prepare stamp buffer for unique pixel counting.
  ScatterPlanScratch& scratch = scatter_plan_scratch();
  const uint32_t w = out_energy.width;
  const uint32_t h = out_energy.height;
  const size_t pixel_count = static_cast<size_t>(w) * h;
  if (scratch.pixel_stamp.size() != pixel_count) {
    scratch.pixel_stamp.assign(pixel_count, 0u);
    scratch.pixel_epoch = 1u;
  } else {
    ++scratch.pixel_epoch;
    if (scratch.pixel_epoch == 0u) {
      std::fill(scratch.pixel_stamp.begin(), scratch.pixel_stamp.end(), 0u);
      scratch.pixel_epoch = 1u;
    }
  }
  const uint32_t epoch = scratch.pixel_epoch;
  uint64_t pixels_touched = 0;
  uint64_t site_activations = 0;

  auto stamp_pixel = [&](uint32_t px, uint32_t py, uint32_t visits) {
    const size_t idx = static_cast<size_t>(py) * w + px;
    if (scratch.pixel_stamp[idx] != epoch) {
      scratch.pixel_stamp[idx] = epoch;
      ++pixels_touched;
    }
    site_activations += static_cast<uint64_t>(visits);
  };

  uint64_t expanded_count = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t visits = static_cast<uint32_t>(std::max<int64_t>(1, std::llround(static_cast<double>(counts[i]))));
    plan_moments += static_cast<uint64_t>(visits);
    const int cx = static_cast<int>(std::floor(pts[i * 2 + 0]));
    const int cy = static_cast<int>(std::floor(pts[i * 2 + 1]));
    for (const auto& off : offsets) {
      const int px = cx + off.dx;
      const int py = cy + off.dy;
      if (px < 0 || py < 0 || px >= static_cast<int>(w) || py >= static_cast<int>(h)) continue;
      ++expanded_count;
      stamp_pixel(static_cast<uint32_t>(px), static_cast<uint32_t>(py), visits);
    }
  }

  AbstractTensorPool::PooledTensor expanded_points;
  AbstractTensorPool::PooledTensor expanded_values;
  const AbstractTensor* scatter_points = &dense_points;
  const AbstractTensor* scatter_values = &plan.values.tensor();
  if (offsets.size() != 1 || offsets[0].dx != 0 || offsets[0].dy != 0 || offsets[0].w != 1.0f) {
    TensorDesc pts_expand{};
    pts_expand.dtype = TensorDType::F32;
    pts_expand.layout = TensorLayout::Dense;
    pts_expand.shape.dims = {static_cast<uint32_t>(expanded_count), 2u};
    expanded_points = scatter_plan_pool().acquire(pts_expand, backend);
    if (!expanded_points.valid()) return false;

    TensorDesc vals_expand{};
    vals_expand.dtype = TensorDType::F32;
    vals_expand.layout = TensorLayout::Dense;
    vals_expand.shape.dims = {static_cast<uint32_t>(expanded_count)};
    expanded_values = scatter_plan_pool().acquire(vals_expand, backend);
    if (!expanded_values.valid()) return false;

    void* exp_pts_v = nullptr;
    size_t exp_pts_bytes = 0;
    void* exp_vals_v = nullptr;
    size_t exp_vals_bytes = 0;
    if (!mem->map(expanded_points->handle(), &exp_pts_v, &exp_pts_bytes)) return false;
    if (!mem->map(expanded_values->handle(), &exp_vals_v, &exp_vals_bytes)) {
      mem->unmap(expanded_points->handle());
      return false;
    }
    auto* exp_pts = static_cast<float*>(exp_pts_v);
    auto* exp_vals = static_cast<float*>(exp_vals_v);

    uint64_t write = 0;
    for (uint32_t i = 0; i < n; ++i) {
      const int cx = static_cast<int>(std::floor(pts[i * 2 + 0]));
      const int cy = static_cast<int>(std::floor(pts[i * 2 + 1]));
      const float energy = vals[i];
      for (const auto& off : offsets) {
        const int px = cx + off.dx;
        const int py = cy + off.dy;
        if (px < 0 || py < 0 || px >= static_cast<int>(w) || py >= static_cast<int>(h)) continue;
        exp_pts[write * 2 + 0] = static_cast<float>(px);
        exp_pts[write * 2 + 1] = static_cast<float>(py);
        exp_vals[write] = energy * off.w;
        ++write;
      }
    }

    mem->unmap(expanded_points->handle());
    mem->unmap(expanded_values->handle());

    scatter_points = &expanded_points.tensor();
    scatter_values = &expanded_values.tensor();
  }

  mem->unmap(dense_points.handle());
  mem->unmap(plan.values->handle());
  mem->unmap(plan.visit_counts->handle());

  if (timing) {
    timing->plan_moments = plan_moments;
    timing->site_activations = site_activations;
    timing->pixels_touched = pixels_touched;
  }

  TensorDesc base_desc{};
  base_desc.dtype = TensorDType::F32;
  base_desc.layout = TensorLayout::Dense;
  base_desc.shape.dims = {xform.height_px, xform.width_px};
  auto base = scatter_plan_pool().acquire(base_desc, backend);
  if (!base.valid()) return false;
  {
    const auto t0 = now_hr();
    void* base_ptr_v = nullptr;
    size_t base_bytes = 0;
    if (!mem->map(base->handle(), &base_ptr_v, &base_bytes)) return false;
    std::memset(base_ptr_v, 0, base_bytes);
    mem->unmap(base->handle());
    const auto t1 = now_hr();
    if (timing) timing->clear_ms += elapsed_ms(t0, t1);
  }

  AbstractTensor energy_tensor;
  {
    const auto t0 = now_hr();
    if (!tensor_scatter_add_2d_f32(base.tensor(), *scatter_points, *scatter_values, &energy_tensor, true)) return false;
    const auto t1 = now_hr();
    if (timing) timing->deposit_ms += elapsed_ms(t0, t1);
  }

  {
    const auto t0 = now_hr();
    if (!copy_tensor_to_canvas_f32(energy_tensor, out_energy)) return false;
    const auto t1 = now_hr();
    if (timing) timing->diffusion_copyback_ms += elapsed_ms(t0, t1);
  }

  AbstractTensor temp_tensor;
  if (energy_to_temp != 0.0f) {
    temp_tensor = clone_dense_f32(energy_tensor, backend);
    if (!temp_tensor.valid()) return false;
    if (energy_to_temp != 1.0f) {
      const auto t0 = now_hr();
      if (!scale_inplace_f32(temp_tensor, energy_to_temp)) return false;
      const auto t1 = now_hr();
      if (timing) timing->temp_state_accum_ms += elapsed_ms(t0, t1);
    }
  } else {
    temp_tensor = clone_dense_f32(base.tensor(), backend);
    if (!temp_tensor.valid()) return false;
  }

  const float decay = std::clamp(sim.decay, 0.0f, 1.0f);
  if (temp_state) {
    if (!temp_state->valid()) {
      const auto t0 = now_hr();
      *temp_state = clone_dense_f32(temp_tensor, backend);
      if (!temp_state->valid()) return false;
      const auto t1 = now_hr();
      if (timing) timing->diffusion_prep_ms += elapsed_ms(t0, t1);
    }
    if (decay > 0.0f && decay < 1.0f) {
      const auto t0 = now_hr();
      if (!scale_inplace_f32(*temp_state, decay)) return false;
      const auto t1 = now_hr();
      if (timing) timing->temp_state_accum_ms += elapsed_ms(t0, t1);
    }
    {
      const auto t0 = now_hr();
      if (!add_scaled_inplace_f32(*temp_state, temp_tensor, 1.0f)) return false;
      const auto t1 = now_hr();
      if (timing) timing->temp_state_accum_ms += elapsed_ms(t0, t1);
    }
    temp_tensor = AbstractTensor::wrap(temp_state->handle(), temp_state->desc(), temp_state->backend(), false);
  } else {
    if (decay > 0.0f && decay < 1.0f) {
      const auto t0 = now_hr();
      if (!scale_inplace_f32(temp_tensor, decay)) return false;
      const auto t1 = now_hr();
      if (timing) timing->temp_state_accum_ms += elapsed_ms(t0, t1);
    }
  }

  if (diffusion_kernel && sim.diffusion_steps > 0) {
    TensorDesc lap_desc = temp_tensor.desc();
    auto lap_tensor = scatter_plan_pool().acquire(lap_desc, backend);
    if (!lap_tensor.valid()) return false;
    for (uint32_t i = 0; i < sim.diffusion_steps; ++i) {
      const auto t0 = now_hr();
      if (!tensor_apply_stencil_2d_f32_into(temp_tensor, *diffusion_kernel, &lap_tensor.tensor())) return false;
      if (!add_scaled_inplace_f32(temp_tensor, lap_tensor.tensor(), sim.diffusion_dt)) return false;
      const auto t1 = now_hr();
      if (timing) timing->diffusion_iter_ms += elapsed_ms(t0, t1);
    }
  }

  {
    const auto t0 = now_hr();
    if (!copy_tensor_to_canvas_f32(temp_tensor, out_temp)) return false;
    const auto t1 = now_hr();
    if (timing) timing->diffusion_copyback_ms += elapsed_ms(t0, t1);
  }

  if (timing) timing->total_ms = elapsed_ms(t_total_start, now_hr());
  return true;
}

bool project_beam_program_to_plane(const BeamProgram& beam_prog,
                                   float plane_z,
                                   ArmatureProgram& out_program) {
  out_program.points.clear();
  for (const auto& b : beam_prog.beams) {
    float dz = b.dz;
    if (std::fabs(dz) < kEps) continue; // no intersection with horizontal plane
    float t = (plane_z - b.oz) / dz;
    if (t < 0.0f) continue; // beam points away from plane
    ToolPoint p;
    p.x = b.ox + b.dx * t;
    p.y = b.oy + b.dy * t;
    p.z = plane_z;
    p.engaged = b.engaged;
    out_program.points.push_back(p);
  }
  return !out_program.points.empty();
}

bool rasterize_beam_program_with_thermal(const BeamProgram& beam_prog,
                                        float plane_z,
                                        TensorCanvas2D& out_energy,
                                        TensorCanvas2D& out_temp,
                                        const MachineControlConfig& machine,
                                        const BeamToolParams& tool,
                                        const ProgramRasterTransform& xform) {
  if (xform.width_px == 0 || xform.height_px == 0) return false;

  if (out_energy.width != xform.width_px || out_energy.height != xform.height_px) {
    out_energy.resize(xform.width_px, xform.height_px, 0.0f);
  } else {
    out_energy.clear(0.0f);
  }
  if (out_temp.width != xform.width_px || out_temp.height != xform.height_px) {
    out_temp.resize(xform.width_px, xform.height_px, 0.0f);
  } else {
    out_temp.clear(0.0f);
  }

  if (beam_prog.beams.empty()) return false;

  const float base_energy = (tool.energy_per_shot > 0.0f) ? tool.energy_per_shot : machine.energy_per_px;
  const float feed = std::max(machine.feed_rate_px_per_s, 1.0f);
  const float dt = (tool.shot_dt > 0.0f) ? tool.shot_dt : (1.0f / feed);

  SpatialKernel kernel = make_spatial_kernel(tool);

  for (const auto& b : beam_prog.beams) {
    if (!b.engaged) continue;
    float dz = b.dz;
    if (std::fabs(dz) < kEps) continue;
    float t = (plane_z - b.oz) / dz;
    if (t < 0.0f) continue;

    float energy = base_energy;
    if (tool.axial_falloff > kEps) {
      energy *= 1.0f / (1.0f + tool.axial_falloff * t * t);
    }

    float x = b.ox + b.dx * t;
    float y_unflipped = b.oy + b.dy * t;
    float px = x * xform.scale + xform.shift_x;
    float py = (static_cast<float>(xform.height_px) - 1.0f) -
               (y_unflipped * xform.scale + xform.shift_y);

    kernel_stamp_energy_and_temp(out_energy,
                                 out_temp,
                                 kernel,
                                 px,
                                 py,
                                 energy,
                                 dt,
                                 std::max(machine.cooling_tau_s, 0.0f),
                                 machine.energy_to_temp,
                                 std::max(machine.max_temp, 0.0f));
  }

  return true;
}

TensorCanvas2D::TensorCanvas2D(uint32_t w, uint32_t h) : width(w), height(h), values(w * h, 0.0f) {}

void TensorCanvas2D::reserve(uint32_t w, uint32_t h) {
  values.reserve(static_cast<size_t>(w) * h);
}

void TensorCanvas2D::resize(uint32_t w, uint32_t h, float init_value) {
  width = w;
  height = h;
  values.resize(static_cast<size_t>(w) * h);
  std::fill(values.begin(), values.end(), init_value);
}

float& TensorCanvas2D::at(uint32_t x, uint32_t y) {
  return values[static_cast<size_t>(y) * width + x];
}

float TensorCanvas2D::at(uint32_t x, uint32_t y) const {
  return values[static_cast<size_t>(y) * width + x];
}

void TensorCanvas2D::clear(float v) {
  std::fill(values.begin(), values.end(), v);
}

float TensorCanvas2D::max_value() const {
  float m = -std::numeric_limits<float>::infinity();
  for (float v : values) m = std::max(m, v);
  return m;
}

float TensorCanvas2D::min_value() const {
  float m = std::numeric_limits<float>::infinity();
  for (float v : values) m = std::min(m, v);
  return m;
}

std::vector<uint8_t> TensorCanvas2D::to_u8_normalized() const {
  std::vector<uint8_t> out;
  out.resize(static_cast<size_t>(width) * height);

  float min_v = min_value();
  float max_v = max_value();
  float denom = std::max(max_v - min_v, kEps);

  for (size_t i = 0; i < values.size(); ++i) {
    float t = (values[i] - min_v) / denom;
    t = std::clamp(t, 0.0f, 1.0f);
    out[i] = static_cast<uint8_t>(std::lround(t * 255.0f));
  }
  return out;
}

ArmatureProgram glyph_outline_to_armature_program(const GlyphOutline& outline,
                                                  float nominal_z,
                                                  uint32_t samples_per_segment) {
  ArmatureProgram program;
  append_glyph_outline_to_program(program, outline, 0.0f, 0.0f, nominal_z, samples_per_segment);
  return program;
}

void append_glyph_outline_to_program(ArmatureProgram& program,
                                     const GlyphOutline& outline,
                                     float tx,
                                     float ty,
                                     float tz,
                                     uint32_t samples_per_segment) {
  program.points.reserve(program.points.size() + outline.segments.size() * std::max<uint32_t>(1, samples_per_segment));

  // NOTE: tz is interpreted as the cutting Z. Safe-Z is injected automatically on MoveTo.
  constexpr float kDefaultSafeZ = -1.0f;
  const float cut_z = tz;
  const float safe_z = std::min(kDefaultSafeZ, cut_z - 1.0f);

  // When appending multiple outlines into a single program, the starting cursor
  // must be the program's last emitted point. Using a fabricated (tx,ty,safe_z)
  // cursor can accidentally suppress the first MoveTo for geometry that begins
  // at (0,0), which then creates an unintended engaged connection.
  ToolPoint cur = program.points.empty() ? ToolPoint{tx, ty, safe_z, false} : program.points.back();
  ToolPoint subpath_start = cur;
  bool have_subpath = false;
  bool need_plunge = false;

  auto push_point = [&](const ToolPoint& p) {
    // Avoid emitting identical consecutive points (common at joins/degenerate segments).
    // This prevents zero-length segments, which can otherwise lead to uneven energy deposition.
    if (!program.points.empty()) {
      constexpr float kDupEps = 1e-6f;
      const ToolPoint& last_emitted = program.points.back();
      const float dx = p.x - last_emitted.x;
      const float dy = p.y - last_emitted.y;
      const float dz = p.z - last_emitted.z;
      const bool same_xy = (dx * dx + dy * dy) <= (kDupEps * kDupEps);
      const bool same_z = std::fabs(dz) <= kDupEps;
      const bool same_engaged = (p.engaged == last_emitted.engaged);
      if (same_xy && same_z && same_engaged) {
        return;
      }
    }
    program.points.push_back(p);
    cur = p;
  };

  for (const auto& seg : outline.segments) {
    switch (seg.op) {
      case OutlineOp::MoveTo: {
        // Rapid travel: disengaged at safe Z.
        ToolPoint rapid{seg.x1 + tx, seg.y1 + ty, safe_z, false};
        push_point(rapid);
        // Next drawable segment will automatically plunge.
        need_plunge = true;
        subpath_start = rapid;
        have_subpath = true;
        break;
      }
      case OutlineOp::LineTo: {
        if (need_plunge) {
          ToolPoint plunge{cur.x, cur.y, cut_z, true};
          push_point(plunge);
          need_plunge = false;
        }
        ToolPoint p0{seg.x1 + tx, seg.y1 + ty, tz, true};
        ToolPoint p1{seg.x3 + tx, seg.y3 + ty, tz, true};
        for (uint32_t i = 1; i <= samples_per_segment; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(samples_per_segment);
          ToolPoint p{lerp(p0.x, p1.x, t), lerp(p0.y, p1.y, t), tz, true};
          push_point(p);
        }
        break;
      }
      case OutlineOp::QuadTo: {
        if (need_plunge) {
          ToolPoint plunge{cur.x, cur.y, cut_z, true};
          push_point(plunge);
          need_plunge = false;
        }
        ToolPoint p0{seg.x1 + tx, seg.y1 + ty, tz, true};
        ToolPoint p1{seg.x2 + tx, seg.y2 + ty, tz, true};
        ToolPoint p2{seg.x3 + tx, seg.y3 + ty, tz, true};
        for (uint32_t i = 1; i <= samples_per_segment; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(samples_per_segment);
          ToolPoint p = eval_quad(p0, p1, p2, t);
          p.z = tz;
          p.engaged = true;
          push_point(p);
        }
        break;
      }
      case OutlineOp::CubicTo: {
        if (need_plunge) {
          ToolPoint plunge{cur.x, cur.y, cut_z, true};
          push_point(plunge);
          need_plunge = false;
        }
        ToolPoint p0 = cur;
        p0.engaged = true;
        ToolPoint p1{seg.x1 + tx, seg.y1 + ty, tz, true};
        ToolPoint p2{seg.x2 + tx, seg.y2 + ty, tz, true};
        ToolPoint p3{seg.x3 + tx, seg.y3 + ty, tz, true};
        for (uint32_t i = 1; i <= samples_per_segment; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(samples_per_segment);
          ToolPoint p = eval_cubic(p0, p1, p2, p3, t);
          p.z = tz;
          p.engaged = true;
          push_point(p);
        }
        break;
      }
      case OutlineOp::Arc: {
        if (need_plunge) {
          ToolPoint plunge{cur.x, cur.y, cut_z, true};
          push_point(plunge);
          need_plunge = false;
        }
        // Parameters: x1/y1=center, x2=radius, y2=start angle (rad), x3=sweep (rad)
        const float cx = seg.x1 + tx;
        const float cy = seg.y1 + ty;
        const float r = std::max(seg.x2, kEps);
        const float start = seg.y2;
        const float sweep = seg.x3;
        // Adaptive samples: keep chord error small; base on 45-degree chunks.
        float abs_sweep = std::fabs(sweep);
        uint32_t steps = std::max<uint32_t>(4, static_cast<uint32_t>(std::ceil(abs_sweep / (kPi / 8.0f))));
        for (uint32_t i = 1; i <= steps; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(steps);
          float ang = start + sweep * t;
          ToolPoint p;
          p.x = cx + r * std::cos(ang);
          p.y = cy + r * std::sin(ang);
          p.z = tz;
          p.engaged = true;
          push_point(p);
        }
        break;
      }
      case OutlineOp::Sin: {
        if (need_plunge) {
          ToolPoint plunge{cur.x, cur.y, cut_z, true};
          push_point(plunge);
          need_plunge = false;
        }
        // Parameters: x1/y1=end point; x2=amplitude; y2=cycles; x3=phase (rad).
        ToolPoint start_pt = cur;
        ToolPoint end_pt{seg.x1 + tx, seg.y1 + ty, tz, true};
        const float amp = seg.x2;
        const float cycles = seg.y2;
        const float phase = seg.x3;

        float dx = end_pt.x - start_pt.x;
        float dy = end_pt.y - start_pt.y;
        float len = std::max(std::sqrt(dx * dx + dy * dy), kEps);

        // Base samples on segment length and cycles to keep good fidelity.
        uint32_t steps = std::max<uint32_t>(
          6u,
          static_cast<uint32_t>(std::ceil(len / 0.75f) + std::fabs(cycles) * 4.0f));
        steps = std::max<uint32_t>(steps, samples_per_segment);

        // Unit tangent and normal.
        float tx_dir = dx / len;
        float ty_dir = dy / len;
        float nx_dir = -ty_dir;
        float ny_dir = tx_dir;

        for (uint32_t i = 1; i <= steps; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(steps);
          float base_x = start_pt.x + dx * t;
          float base_y = start_pt.y + dy * t;
          float s = std::sin((2.0f * kPi * cycles * t) + phase);
          float off_x = nx_dir * amp * s;
          float off_y = ny_dir * amp * s;
          ToolPoint p;
          p.x = base_x + off_x;
          p.y = base_y + off_y;
          p.z = tz;
          p.engaged = true;
          push_point(p);
        }
        break;
      }
      case OutlineOp::Close: {
        if (need_plunge) {
          ToolPoint plunge{cur.x, cur.y, cut_z, true};
          push_point(plunge);
          need_plunge = false;
        }
        if (have_subpath) {
          ToolPoint p0 = cur;
          ToolPoint p1 = subpath_start;
          for (uint32_t i = 1; i <= samples_per_segment; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(samples_per_segment);
            ToolPoint p{lerp(p0.x, p1.x, t), lerp(p0.y, p1.y, t), tz, true};
            push_point(p);
          }
        }
        break;
      }
    }
  }
}

void rasterize_program_gaussian_with_thermal(const ArmatureProgram& program,
                                            TensorCanvas2D& out_energy,
                                            TensorCanvas2D& out_temp,
                                            const MachineControlConfig& machine,
                                            const GaussianToolParams& tool,
                                            float margin) {
  if (out_energy.width == 0 || out_energy.height == 0) return;
  out_energy.clear(0.0f);
  if (out_temp.width != out_energy.width || out_temp.height != out_energy.height) {
    out_temp.resize(out_energy.width, out_energy.height, 0.0f);
  }
  out_temp.clear(0.0f);

  if (program.points.empty()) return;

  SpatialKernel kernel = make_spatial_kernel_gaussian(tool.sigma_px);

  float min_x = std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();

  for (const auto& p : program.points) {
    bounds_update(p.x, p.y, min_x, min_y, max_x, max_y);
  }

  float span_x = std::max(max_x - min_x, 1.0f);
  float span_y = std::max(max_y - min_y, 1.0f);

  float target_w = std::max(1.0f, static_cast<float>(out_energy.width) - 2.0f * margin);
  float target_h = std::max(1.0f, static_cast<float>(out_energy.height) - 2.0f * margin);
  float s = std::min(target_w / span_x, target_h / span_y);

  std::vector<ToolPoint> img_pts;
  img_pts.reserve(program.points.size());
  for (const auto& p : program.points) {
    float x = (p.x - min_x) * s + margin;
    float y = (p.y - min_y) * s + margin;
    ToolPoint ip;
    ip.x = x;
    ip.y = (static_cast<float>(out_energy.height) - 1.0f) - y;
    ip.z = p.z;
    ip.engaged = p.engaged;
    img_pts.push_back(ip);
  }

  // Execute in image space with constant energy per unit length.
  std::vector<ToolPoint> exec_pts = resample_polyline_equal_arclen(img_pts, machine.step_px);
  if (exec_pts.size() < 2) return;

  float step = std::max(machine.step_px, 0.05f);
  float energy_step = machine.energy_per_px * step;
  float feed = std::max(machine.feed_rate_px_per_s, 1.0f);
  float dt = step / feed;

  if (!machine.enable_thermal_guard) {
    bool have_last = false;
    ToolPoint last{};
    for (const auto& p : exec_pts) {
      if (!p.engaged) continue;
      if (have_last && distance2(p.x, p.y, last.x, last.y) < 1e-6f) continue;
      kernel_stamp_energy(out_energy, kernel, p.x, p.y, energy_step);
      last = p;
      have_last = true;
    }
    out_temp.clear(0.0f);
    return;
  }

  bool have_last = false;
  ToolPoint last{};
  for (const auto& p : exec_pts) {
    if (!p.engaged) continue;
    if (have_last && distance2(p.x, p.y, last.x, last.y) < 1e-6f) continue;
    kernel_stamp_energy_and_temp(out_energy,
                                 out_temp,
                                 kernel,
                                 p.x,
                                 p.y,
                                 energy_step,
                                 dt,
                                 std::max(machine.cooling_tau_s, 0.0f),
                                 machine.energy_to_temp,
                                 std::max(machine.max_temp, 0.0f));
    last = p;
    have_last = true;
  }
}

void rasterize_program_gaussian_with_thermal_mapped(const ArmatureProgram& program,
                                                   TensorCanvas2D& out_energy,
                                                   TensorCanvas2D& out_temp,
                                                   const MachineControlConfig& machine,
                                                   const GaussianToolParams& tool,
                                                   const ProgramMapping& mapping) {
  if (out_energy.width == 0 || out_energy.height == 0) return;
  out_energy.clear(0.0f);
  if (out_temp.width != out_energy.width || out_temp.height != out_energy.height) {
    out_temp.resize(out_energy.width, out_energy.height, 0.0f);
  }
  out_temp.clear(0.0f);

  if (program.points.empty()) return;

  SpatialKernel kernel = make_spatial_kernel_gaussian(tool.sigma_px);
  float min_x = mapping.min_x;
  float min_y = mapping.min_y;
  float s = std::max(mapping.scale, 1e-6f);
  float margin = mapping.margin;

  std::vector<ToolPoint> img_pts;
  img_pts.reserve(program.points.size());
  for (const auto& p : program.points) {
    float x = (p.x - min_x) * s + margin;
    float y = (p.y - min_y) * s + margin;
    ToolPoint ip;
    ip.x = x;
    ip.y = (static_cast<float>(out_energy.height) - 1.0f) - y;
    ip.z = p.z;
    ip.engaged = p.engaged;
    img_pts.push_back(ip);
  }

  std::vector<ToolPoint> exec_pts = resample_polyline_equal_arclen(img_pts, machine.step_px);
  if (exec_pts.size() < 2) return;

  float step = std::max(machine.step_px, 0.05f);
  float energy_step = machine.energy_per_px * step;
  float feed = std::max(machine.feed_rate_px_per_s, 1.0f);
  float dt = step / feed;

  if (!machine.enable_thermal_guard) {
    bool have_last = false;
    ToolPoint last{};
    for (const auto& p : exec_pts) {
      if (!p.engaged) continue;
      if (have_last && distance2(p.x, p.y, last.x, last.y) < 1e-6f) continue;
      kernel_stamp_energy(out_energy, kernel, p.x, p.y, energy_step);
      last = p;
      have_last = true;
    }
    out_temp.clear(0.0f);
    return;
  }

  bool have_last = false;
  ToolPoint last{};
  for (const auto& p : exec_pts) {
    if (!p.engaged) continue;
    if (have_last && distance2(p.x, p.y, last.x, last.y) < 1e-6f) continue;
    kernel_stamp_energy_and_temp(out_energy,
                                 out_temp,
                                 kernel,
                                 p.x,
                                 p.y,
                                 energy_step,
                                 dt,
                                 std::max(machine.cooling_tau_s, 0.0f),
                                 machine.energy_to_temp,
                                 std::max(machine.max_temp, 0.0f));
    last = p;
    have_last = true;
  }
}

void rasterize_program_gaussian(const ArmatureProgram& program,
                                TensorCanvas2D& canvas,
                                const MachineControlConfig& machine,
                                const GaussianToolParams& tool,
                                float margin) {
  TensorCanvas2D temp(canvas.width, canvas.height);
  rasterize_program_gaussian_with_thermal(program, canvas, temp, machine, tool, margin);
}

bool write_png_grayscale_u8(const std::string& path,
                            uint32_t width,
                            uint32_t height,
                            std::span<const uint8_t> pixels) {
  if (width == 0 || height == 0) return false;
  if (pixels.size() < static_cast<size_t>(width) * height) return false;

  FILE* fp = nullptr;
#if defined(_MSC_VER)
  if (fopen_s(&fp, path.c_str(), "wb") != 0) fp = nullptr;
#else
  fp = std::fopen(path.c_str(), "wb");
#endif
  if (!fp) return false;

  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    std::fclose(fp);
    return false;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_write_struct(&png_ptr, nullptr);
    std::fclose(fp);
    return false;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    std::fclose(fp);
    return false;
  }

  png_init_io(png_ptr, fp);

  png_set_IHDR(
      png_ptr,
      info_ptr,
      width,
      height,
      8,
      PNG_COLOR_TYPE_GRAY,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_BASE,
      PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  std::vector<png_bytep> row_ptrs;
  row_ptrs.resize(height);
  for (uint32_t y = 0; y < height; ++y) {
    auto* row = const_cast<uint8_t*>(pixels.data() + static_cast<size_t>(y) * width);
    row_ptrs[y] = reinterpret_cast<png_bytep>(row);
  }

  png_write_image(png_ptr, row_ptrs.data());
  png_write_end(png_ptr, info_ptr);

  png_destroy_write_struct(&png_ptr, &info_ptr);
  std::fclose(fp);
  return true;
}

bool write_png_rgb_u8(const std::string& path,
                      uint32_t width,
                      uint32_t height,
                      std::span<const uint8_t> pixels_rgb) {
  if (width == 0 || height == 0) return false;
  if (pixels_rgb.size() < static_cast<size_t>(width) * height * 3) return false;

  FILE* fp = nullptr;
#if defined(_MSC_VER)
  if (fopen_s(&fp, path.c_str(), "wb") != 0) fp = nullptr;
#else
  fp = std::fopen(path.c_str(), "wb");
#endif
  if (!fp) return false;

  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) {
    std::fclose(fp);
    return false;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    png_destroy_write_struct(&png_ptr, nullptr);
    std::fclose(fp);
    return false;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    std::fclose(fp);
    return false;
  }

  png_init_io(png_ptr, fp);

  png_set_IHDR(
      png_ptr,
      info_ptr,
      width,
      height,
      8,
      PNG_COLOR_TYPE_RGB,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_BASE,
      PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  std::vector<png_bytep> row_ptrs;
  row_ptrs.resize(height);
  size_t row_bytes = static_cast<size_t>(width) * 3;
  for (uint32_t y = 0; y < height; ++y) {
    auto* row = const_cast<uint8_t*>(pixels_rgb.data() + static_cast<size_t>(y) * row_bytes);
    row_ptrs[y] = reinterpret_cast<png_bytep>(row);
  }

  png_write_image(png_ptr, row_ptrs.data());
  png_write_end(png_ptr, info_ptr);

  png_destroy_write_struct(&png_ptr, &info_ptr);
  std::fclose(fp);
  return true;
}

} // namespace nodus::tensors::kpath
