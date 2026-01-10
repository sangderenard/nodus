#include "common/tensors/abstraction/kpath/kpath_raster.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include <png.h>

namespace nodus::tensors::kpath {

namespace {

constexpr float kEps = 1e-6f;
constexpr float kPi = 3.14159265358979323846f;

static float lerp(float a, float b, float t) { return a + (b - a) * t; }

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

static std::vector<ToolPoint> resample_polyline_equal_arclen(const std::vector<ToolPoint>& pts, float step) {
  std::vector<ToolPoint> out;
  if (pts.size() < 2) return out;

  step = std::max(step, 0.05f);
  out.reserve(static_cast<size_t>((pts.size() - 1) * 2));

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

  return out;
}

struct GaussianKernel final {
  int radius = 0;
  float sigma = 1.0f;
  std::vector<float> weights; // (2r+1)^2 row-major

  float at(int dx, int dy) const {
    int s = 2 * radius + 1;
    int ix = dx + radius;
    int iy = dy + radius;
    return weights[static_cast<size_t>(iy) * s + ix];
  }
};

static GaussianKernel make_gaussian_kernel(float sigma_px) {
  GaussianKernel k;
  k.sigma = std::max(sigma_px, 0.25f);
  k.radius = static_cast<int>(std::ceil(3.0f * k.sigma));
  int s = 2 * k.radius + 1;
  k.weights.resize(static_cast<size_t>(s) * s);

  float inv2 = 1.0f / (2.0f * k.sigma * k.sigma);
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

static void gaussian_stamp_energy(TensorCanvas2D& canvas,
                                  const GaussianKernel& kernel,
                                  float cx,
                                  float cy,
                                  float energy) {
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

static void gaussian_stamp_energy_and_temp(TensorCanvas2D& energy_canvas,
                                           TensorCanvas2D& temp_canvas,
                                           const GaussianKernel& kernel,
                                           float cx,
                                           float cy,
                                           float energy,
                                           float dt,
                                           float cooling_tau,
                                           float energy_to_temp,
                                           float max_temp) {
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

  GaussianKernel kernel = make_gaussian_kernel(tool.sigma_px);
  float cx = 0.5f * static_cast<float>(n - 1);
  float cy = 0.5f * static_cast<float>(n - 1);

  // Unit-energy impulse response in raster space.
  gaussian_stamp_energy(canvas, kernel, cx, cy, 1.0f);

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

bool project_beam_program_to_plane(const GimbalProgram& beam_prog,
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

TensorCanvas2D::TensorCanvas2D(uint32_t w, uint32_t h) : width(w), height(h), values(w * h, 0.0f) {}

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

  ToolPoint cur{tx, ty, safe_z, false};
  ToolPoint subpath_start{tx, ty, safe_z, false};
  bool have_subpath = false;
  bool need_plunge = false;

  auto push_point = [&](const ToolPoint& p) {
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
    out_temp = TensorCanvas2D(out_energy.width, out_energy.height);
  }
  out_temp.clear(0.0f);

  if (program.points.empty()) return;

  GaussianKernel kernel = make_gaussian_kernel(tool.sigma_px);

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
    for (const auto& p : exec_pts) {
      if (!p.engaged) continue;
      gaussian_stamp_energy(out_energy, kernel, p.x, p.y, energy_step);
    }
    out_temp.clear(0.0f);
    return;
  }

  for (const auto& p : exec_pts) {
    if (!p.engaged) continue;
    gaussian_stamp_energy_and_temp(out_energy,
                                   out_temp,
                                   kernel,
                                   p.x,
                                   p.y,
                                   energy_step,
                                   dt,
                                   std::max(machine.cooling_tau_s, 0.0f),
                                   machine.energy_to_temp,
                                   std::max(machine.max_temp, 0.0f));
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
    out_temp = TensorCanvas2D(out_energy.width, out_energy.height);
  }
  out_temp.clear(0.0f);

  if (program.points.empty()) return;

  GaussianKernel kernel = make_gaussian_kernel(tool.sigma_px);
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
    for (const auto& p : exec_pts) {
      if (!p.engaged) continue;
      gaussian_stamp_energy(out_energy, kernel, p.x, p.y, energy_step);
    }
    out_temp.clear(0.0f);
    return;
  }

  for (const auto& p : exec_pts) {
    if (!p.engaged) continue;
    gaussian_stamp_energy_and_temp(out_energy,
                                   out_temp,
                                   kernel,
                                   p.x,
                                   p.y,
                                   energy_step,
                                   dt,
                                   std::max(machine.cooling_tau_s, 0.0f),
                                   machine.energy_to_temp,
                                   std::max(machine.max_temp, 0.0f));
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
