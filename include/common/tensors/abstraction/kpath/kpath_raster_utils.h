#pragma once

#include <span>
#include <vector>

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/abstract_tensor_pool.h"
#include "common/tensors/abstraction/kpath/kpath_film.h"
#include "common/tensors/abstraction/kpath/kpath_raster.h"

namespace nodus::tensors::kpath {

struct FilmTimingBreakdown final {
  double map_ms = 0.0;
  double normalize_energy_ms = 0.0;
  double normalize_heat_ms = 0.0;
  double build_ms = 0.0;
  double scatter_ms = 0.0;
  double reduce_ms = 0.0;
  double pixel_loop_ms = 0.0;
  double unmap_ms = 0.0;
  double total_ms = 0.0;
};

struct BlackbodyResponseConfig final {
  // Normalized heat value 0..1 maps to [min_kelvin, max_kelvin].
  float min_kelvin = 800.0f;
  float max_kelvin = 2000.0f;
  float intensity = 1.0f;
};

struct BeamHistogram final {
  float r = 1.0f;
  float g = 0.85f;
  float b = 0.7f;
};

// Utility helpers shared by demos and kpath tools.
AbstractTensor make_flip_tensor(TensorBackend* backend, uint32_t width, uint32_t height);
AbstractTensorPool::PooledTensor make_flip_tensor(AbstractTensorPool& pool,
                                                  TensorBackend* backend,
                                                  uint32_t width,
                                                  uint32_t height);
AbstractTensor make_kernel_bank(TensorBackend* backend, std::span<const float> weights);
AbstractTensorPool::PooledTensor make_kernel_bank(AbstractTensorPool& pool,
                                                  TensorBackend* backend,
                                                  std::span<const float> weights);
AbstractTensor make_kernel_ids_for_heads(const AbstractTensor& points,
                                         uint32_t width,
                                         uint32_t height,
                                         uint32_t heads,
                                         TensorBackend* backend);
AbstractTensorPool::PooledTensor make_kernel_ids_for_heads(AbstractTensorPool& pool,
                                                           const AbstractTensor& points,
                                                           uint32_t width,
                                                           uint32_t height,
                                                           uint32_t heads,
                                                           TensorBackend* backend);
AbstractTensor make_laplacian_kernel(TensorBackend* backend);
AbstractTensorPool::PooledTensor make_laplacian_kernel(AbstractTensorPool& pool, TensorBackend* backend);

// Fill a U32 RGBA tensor with a tone-mapped view of an energy canvas.
bool fill_flip_rgba(const TensorCanvas2D& energy,
                    FilmTensor2D& film,
                    AbstractTensor& rgba_u32,
                    float exposure_gain,
                    float film_decay);

// Combine beam histogram from energy with a blackbody response from heat.
bool fill_flip_rgba_dual(const TensorCanvas2D& energy,
                         const TensorCanvas2D& heat,
                         FilmTensor2D& film,
                         AbstractTensor& rgba_u32,
                         float exposure_gain,
                         float film_decay,
                         const BeamHistogram& beam,
                         const BlackbodyResponseConfig& blackbody,
                         bool normalize_energy = true,
                         bool normalize_heat = true,
                         float energy_norm_max = 1.0f,
                         float heat_norm_max = 1.0f,
                         FilmTimingBreakdown* timing = nullptr);

// AbstractTensor variant: film_rgb_f32 is a dense [H,W,3] F32 tensor updated in-place
// (decay + add) and used to produce the packed RGBA output.
bool fill_flip_rgba_dual_tensor(const TensorCanvas2D& energy,
                                const TensorCanvas2D& heat,
                                FilmTensor& film,
                                AbstractTensorPool::PooledTensor& film_exposures,
                                AbstractTensorPool::PooledTensor& film_exposures_scratch,
                                AbstractTensor& film_rgb_f32,
                                AbstractTensor& rgba_u32,
                                float exposure_gain,
                                float film_decay,
                                const BeamHistogram& beam,
                                const BlackbodyResponseConfig& blackbody,
                                bool normalize_energy = true,
                                bool normalize_heat = true,
                                float energy_norm_max = 1.0f,
                                float heat_norm_max = 1.0f,
                                FilmTimingBreakdown* timing = nullptr);

} // namespace nodus::tensors::kpath
