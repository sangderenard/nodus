#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstring>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/tensor_index.h"
#include "common/tensors/abstraction/tensor_math.h"
#include "common/tensors/abstraction/tensor_types.h"

#include "common/tensors/abstraction/tensor_index.h"
#include "common/tensors/abstraction/tensor_math.h"
#include "common/tensors/abstraction/tensor_types.h"

namespace nodus::tensors::kpath {

// Shared spectral basis descriptor. Emission and reactance histograms each carry
// their own weights but must agree on this basis (bin ordering/centers) to
// broadcast meaningfully.
struct SpectrumBasis final {
  std::string name;
  std::vector<float> bin_centers; // e.g., wavelengths or arbitrary spectral bins

  bool compatible_with(const SpectrumBasis& other) const {
    return bin_centers.size() == other.bin_centers.size();
  }
};

// A simple plate: single-channel reactance over a 2D domain. Useful for the
// legacy rasterizer path where we only care about scalar energy response.
struct PlateTensor2D final {
  uint32_t width = 0;
  uint32_t height = 0;
  // Per-pixel reactance (single channel). Length = width * height.
  std::vector<float> reactance;

  PlateTensor2D() = default;
  PlateTensor2D(uint32_t w, uint32_t h) { resize(w, h); }

  void resize(uint32_t w, uint32_t h) {
    width = w; height = h;
    reactance.assign(static_cast<size_t>(w) * h, 0.0f);
  }

  float& at(uint32_t x, uint32_t y) {
    return reactance[static_cast<size_t>(y) * width + x];
  }
  float at(uint32_t x, uint32_t y) const {
    return reactance[static_cast<size_t>(y) * width + x];
  }
  void clear(float v = 0.0f) {
    std::fill(reactance.begin(), reactance.end(), v);
  }
};

// Multi-channel film: spectral reactance layers live along the channel axis and
// exposures accumulate per pixel per channel. Reactance describes how sensitive
// the layer is to a given spectral bin; emission supplies bin weights.
struct FilmTensor2D final {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  SpectrumBasis basis;

  // Layer sensitivity (chemistry) per channel; length = channels.
  std::vector<float> layer_reactance;
  // Accumulated exposure per pixel per channel; length = width * height * channels.
  std::vector<float> exposure;

  FilmTensor2D() = default;
  FilmTensor2D(uint32_t w, uint32_t h, uint32_t c, SpectrumBasis b = {}) {
    resize(w, h, c, std::move(b));
  }

  void resize(uint32_t w, uint32_t h, uint32_t c, SpectrumBasis b = {}) {
    width = w; height = h; channels = c; basis = std::move(b);
    layer_reactance.assign(c, 1.0f);
    exposure.assign(static_cast<size_t>(w) * h * c, 0.0f);
  }

  size_t index(uint32_t x, uint32_t y, uint32_t c) const {
    return (static_cast<size_t>(y) * width + x) * channels + c;
  }

  float& at(uint32_t x, uint32_t y, uint32_t c) { return exposure[index(x, y, c)]; }
  float at(uint32_t x, uint32_t y, uint32_t c) const { return exposure[index(x, y, c)]; }

  // Broadcast multiply-add: emission histogram (length = channels) is applied
  // against the per-channel reactance and accumulated into the exposure slice
  // for the hit pixel. Caller must ensure basis compatibility.
  void accumulate(uint32_t x, uint32_t y, std::span<const float> emission_hist) {
    if (emission_hist.size() != channels) return;
    const size_t base = (static_cast<size_t>(y) * width + x) * channels;
    for (size_t c = 0; c < channels; ++c) {
      exposure[base + c] += emission_hist[c] * layer_reactance[c];
    }
  }

  void clear(float v = 0.0f) {
    std::fill(exposure.begin(), exposure.end(), v);
  }
};

// Holographic plate: each pixel references a stencil tensor capturing
// directional/wavelength traversal data. The stencil itself is a 3D tensor
// over (theta, phi, rho) where rho encodes wavelength or path length.
struct HolographicStencilTensor final {
  uint32_t dim_theta = 0;
  uint32_t dim_phi = 0;
  uint32_t dim_rho = 0;
  // Optional metadata per stencil (single traversal annotation).
  float theta_value = 0.0f;
  float phi_value = 0.0f;
  float rho_value = 0.0f;
  // Flattened samples of size dim_theta * dim_phi * dim_rho.
  std::vector<float> samples;

  void resize(uint32_t t, uint32_t p, uint32_t r) {
    dim_theta = t; dim_phi = p; dim_rho = r;
    samples.assign(static_cast<size_t>(t) * p * r, 0.0f);
  }

  size_t index(uint32_t t, uint32_t p, uint32_t r) const {
    return (static_cast<size_t>(t) * dim_phi + p) * dim_rho + r;
  }

  float& at(uint32_t t, uint32_t p, uint32_t r) { return samples[index(t, p, r)]; }
  float at(uint32_t t, uint32_t p, uint32_t r) const { return samples[index(t, p, r)]; }
};

struct HolographicPlateTensor2D final {
  uint32_t width = 0;
  uint32_t height = 0;

  // Per-pixel list of references to stencils (append-on-hit, no integration).
  // Each entry holds indices into the stencils array below.
  std::vector<std::vector<uint32_t>> pixel_refs;
  std::vector<HolographicStencilTensor> stencils;

  void resize(uint32_t w, uint32_t h) {
    width = w; height = h;
    pixel_refs.clear();
    pixel_refs.resize(static_cast<size_t>(w) * h);
  }

  std::vector<uint32_t>& at(uint32_t x, uint32_t y) {
    return pixel_refs[static_cast<size_t>(y) * width + x];
  }
  const std::vector<uint32_t>& at(uint32_t x, uint32_t y) const {
    return pixel_refs[static_cast<size_t>(y) * width + x];
  }

  uint32_t add_stencil(HolographicStencilTensor stencil) {
    stencils.push_back(std::move(stencil));
    return static_cast<uint32_t>(stencils.size() - 1);
  }

  // Record a traversal by appending a stencil reference at (x,y).
  void record(uint32_t x, uint32_t y, uint32_t stencil_idx) {
    if (stencil_idx >= stencils.size()) return;
    pixel_refs[static_cast<size_t>(y) * width + x].push_back(stencil_idx);
  }
};

// Batch of ray strikes with matching lengths across spans.
struct RayStrikeBatch final {
  std::span<const float> xs;       // pixel-space X
  std::span<const float> ys;       // pixel-space Y
  std::span<const float> wavelengths; // spectral coordinate (rho)
  std::span<const float> thetas;   // theta angle
  std::span<const float> phis;     // phi angle
  std::span<const float> intensities; // deposited energy

  size_t size() const {
    return std::min({xs.size(), ys.size(), wavelengths.size(), thetas.size(), phis.size(), intensities.size()});
  }
};

inline uint32_t clamp_pixel(float v, uint32_t limit) {
  if (v < 0.0f) return 0;
  float m = std::floor(v);
  if (m >= static_cast<float>(limit)) return limit ? limit - 1 : 0;
  return static_cast<uint32_t>(m);
}

struct FilmHistogramDescriptor final {
  SpectrumBasis basis;
  std::vector<float> layer_reactance;

  uint32_t channel_count() const {
    return static_cast<uint32_t>(layer_reactance.size());
  }
};

class FilmTensor {
 public:
  FilmTensor() = default;
  bool configure(uint32_t width,
                 uint32_t height,
                 std::vector<FilmHistogramDescriptor> descriptors);
  bool valid() const { return width_ != 0 && height_ != 0 && total_channels_ != 0 && !histograms_.empty(); }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }
  uint32_t histogram_count() const { return static_cast<uint32_t>(histograms_.size()); }
  uint32_t total_channels() const { return total_channels_; }

  AbstractTensor histogram_tensor(const AbstractTensor& exposures, uint32_t index) const;
  bool reduce_histogram(const AbstractTensor& exposures, uint32_t index, AbstractTensor* out) const;
  bool reduce_all(const AbstractTensor& exposures, AbstractTensor* out) const;
  void clear() { histograms_.clear(); width_ = height_ = total_channels_ = 0; }

 private:
  struct HistogramMeta final {
    FilmHistogramDescriptor desc;
    uint32_t offset = 0;
  };

  std::vector<HistogramMeta> histograms_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t total_channels_ = 0;
};

// Map wavelength to nearest basis bin.
inline uint32_t nearest_bin(const SpectrumBasis& basis, float wavelength) {
  if (basis.bin_centers.empty()) return 0;
  uint32_t best = 0;
  float best_d = std::numeric_limits<float>::infinity();
  for (uint32_t i = 0; i < basis.bin_centers.size(); ++i) {
    float d = std::fabs(basis.bin_centers[i] - wavelength);
    if (d < best_d || (d == best_d && basis.bin_centers[i] > basis.bin_centers[best])) {
      best_d = d;
      best = i;
    }
  }
  return best;
}

// Integrate strikes into a plate: scalar accumulation of intensity.
inline void apply_strikes(PlateTensor2D& plate, const RayStrikeBatch& batch) {
  const size_t n = batch.size();
  if (!plate.width || !plate.height) return;
  for (size_t i = 0; i < n; ++i) {
    uint32_t px = clamp_pixel(batch.xs[i], plate.width);
    uint32_t py = clamp_pixel(batch.ys[i], plate.height);
    plate.at(px, py) += batch.intensities[i];
  }
}

// Integrate strikes into a film: spectral accumulation using nearest-bin mapping.
inline void apply_strikes(FilmTensor2D& film, const RayStrikeBatch& batch) {
  const size_t n = batch.size();
  if (!film.width || !film.height || film.channels == 0) return;
  std::vector<float> one_hot;
  one_hot.resize(film.channels, 0.0f);

  for (size_t i = 0; i < n; ++i) {
    uint32_t px = clamp_pixel(batch.xs[i], film.width);
    uint32_t py = clamp_pixel(batch.ys[i], film.height);
    uint32_t bin = nearest_bin(film.basis, batch.wavelengths[i]);
    if (bin >= film.channels) bin = film.channels - 1;
    std::fill(one_hot.begin(), one_hot.end(), 0.0f);
    one_hot[bin] = batch.intensities[i];
    film.accumulate(px, py, one_hot);
  }
}

// Record strikes into a holographic plate: append a per-hit stencil reference.
inline void apply_strikes(HolographicPlateTensor2D& plate, const RayStrikeBatch& batch) {
  const size_t n = batch.size();
  if (!plate.width || !plate.height) return;
  for (size_t i = 0; i < n; ++i) {
    uint32_t px = clamp_pixel(batch.xs[i], plate.width);
    uint32_t py = clamp_pixel(batch.ys[i], plate.height);

    HolographicStencilTensor stencil;
    stencil.resize(1, 1, 1);
    stencil.theta_value = batch.thetas[i];
    stencil.phi_value = batch.phis[i];
    stencil.rho_value = batch.wavelengths[i];
    stencil.at(0, 0, 0) = batch.intensities[i];
    uint32_t idx = plate.add_stencil(std::move(stencil));
    plate.record(px, py, idx);
  }
}

} // namespace nodus::tensors::kpath
