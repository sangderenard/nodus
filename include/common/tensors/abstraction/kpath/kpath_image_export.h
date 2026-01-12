#pragma once

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

// Writing helpers.
bool export_canvas_to_png(const TensorCanvas2D& canvas, const std::string& path);
bool export_plate_png(const PlateTensor2D& plate, const std::string& path);
bool export_film_png(const FilmTensor2D& film, const std::string& path,
                     FilmValueAggregator aggregator = nullptr);
bool export_holographic_plate_png(const HolographicPlateTensor2D& plate, const std::string& path,
                                  bool count_stencils = true);

bool export_canvas_rgb(const TensorCanvas2D& r,
                       const TensorCanvas2D& g,
                       const TensorCanvas2D& b,
                       const std::string& path);

} // namespace nodus::tensors::kpath
