#pragma once

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/kpath/kpath_film.h"
#include "common/tensors/abstraction/kpath/kpath_raster.h"

#include <functional>
#include <string>

namespace nodus::tensors::kpath {

// Aggregator for film exposures.
using FilmValueAggregator = std::function<float(const FilmTensor2D&, uint32_t, uint32_t)>;

// Builds a canvas from the provided structures.
TensorCanvas2D plate_to_canvas(const PlateTensor2D& plate);
TensorCanvas2D film_to_canvas(const FilmTensor2D& film,
                              FilmValueAggregator aggregator = nullptr,
                              bool normalize = true);
TensorCanvas2D holographic_plate_to_canvas(const HolographicPlateTensor2D& plate,
                                           bool count_stencils = true,
                                           bool normalize = true);

struct RgbColor final {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

struct HslColor final {
    float h = 0.0f; // [0,1]
    float s = 0.0f; // [0,1]
    float l = 0.0f; // [0,1]
};

// Color conversion helpers (normalized floats).
RgbColor hsl_to_rgb(const HslColor& hsl);
HslColor rgb_to_hsl(const RgbColor& rgb);

// Abstract-tensor image helpers (HWC layout, channel dim required).
AbstractTensor make_image_tensor_from_canvas(const TensorCanvas2D& canvas,
                                                                                         TensorBackend* backend = nullptr);
AbstractTensor make_image_tensor_from_canvases_rgb(const TensorCanvas2D& r,
                                                                                                     const TensorCanvas2D& g,
                                                                                                     const TensorCanvas2D& b,
                                                                                                     TensorBackend* backend = nullptr);
AbstractTensor make_image_tensor_from_canvases_rgba(const TensorCanvas2D& r,
                                                                                                        const TensorCanvas2D& g,
                                                                                                        const TensorCanvas2D& b,
                                                                                                        const TensorCanvas2D& a,
                                                                                                        TensorBackend* backend = nullptr);

// Export helpers: channels must be 1 (gray), 3 (rgb), or 4 (rgba).
// These enqueue background jobs and return immediately.
bool export_tensor_png(const AbstractTensor& image, const std::string& path,
                       bool normalize = true);

// Export HSL tensors (channels=3) by converting to RGB.
// Enqueues a background job and returns immediately.
bool export_hsl_tensor_png(const AbstractTensor& image_hsl, const std::string& path,
                           bool normalize = true);

// Optional: block until pending image saves finish.
void wait_for_pending_image_saves();

} // namespace nodus::tensors::kpath
