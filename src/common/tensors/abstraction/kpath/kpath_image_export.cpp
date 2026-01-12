#include "common/tensors/abstraction/kpath/kpath_image_export.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace nodus::tensors::kpath {

namespace {

static TensorCanvas2D canvas_from_plate(const PlateTensor2D& plate) {
  TensorCanvas2D canvas;
  if (!plate.width || !plate.height) return canvas;
  canvas.resize(plate.width, plate.height);
  for (uint32_t y = 0; y < plate.height; ++y) {
    for (uint32_t x = 0; x < plate.width; ++x) {
      canvas.at(x, y) = plate.at(x, y);
    }
  }
  return canvas;
}

static float default_film_aggregator(const FilmTensor2D& film, uint32_t x, uint32_t y) {
  float sum = 0.0f;
  for (uint32_t c = 0; c < film.channels; ++c) {
    sum += film.layer_reactance[c] * film.at(x, y, c);
  }
  return sum;
}

static TensorCanvas2D canvas_from_film(const FilmTensor2D& film,
                                       FilmValueAggregator aggregator,
                                       bool normalize) {
  TensorCanvas2D canvas;
  if (!film.width || !film.height) return canvas;
  if (!aggregator) aggregator = default_film_aggregator;
  canvas.resize(film.width, film.height, normalize ? 0.0f : std::numeric_limits<float>::quiet_NaN());
  for (uint32_t y = 0; y < film.height; ++y) {
    for (uint32_t x = 0; x < film.width; ++x) {
      canvas.at(x, y) = aggregator(film, x, y);
    }
  }
  return canvas;
}

static TensorCanvas2D canvas_from_holographic(const HolographicPlateTensor2D& plate,
                                               bool count_stencils,
                                               bool normalize) {
  TensorCanvas2D canvas;
  if (!plate.width || !plate.height) return canvas;
  canvas.resize(plate.width, plate.height, normalize ? 0.0f : std::numeric_limits<float>::quiet_NaN());
  for (uint32_t y = 0; y < plate.height; ++y) {
    for (uint32_t x = 0; x < plate.width; ++x) {
      const auto& refs = plate.at(x, y);
      float value = 0.0f;
      if (count_stencils) {
        value = static_cast<float>(refs.size());
      } else {
        for (uint32_t idx : refs) {
          if (idx >= plate.stencils.size()) continue;
          const auto& stencil = plate.stencils[idx];
          for (float sample : stencil.samples) value += sample;
        }
      }
      canvas.at(x, y) = value;
    }
  }
  return canvas;
}

} // namespace

TensorCanvas2D plate_to_canvas(const PlateTensor2D& plate) {
  return canvas_from_plate(plate);
}

TensorCanvas2D film_to_canvas(const FilmTensor2D& film,
                              FilmValueAggregator aggregator,
                              bool normalize) {
  return canvas_from_film(film, aggregator, normalize);
}

TensorCanvas2D holographic_plate_to_canvas(const HolographicPlateTensor2D& plate,
                                           bool count_stencils,
                                           bool normalize) {
  return canvas_from_holographic(plate, count_stencils, normalize);
}

bool export_canvas_to_png(const TensorCanvas2D& canvas, const std::string& path) {
  if (!canvas.width || !canvas.height) return false;
  const std::vector<uint8_t> pixels = canvas.to_u8_normalized();
  return write_png_grayscale_u8(path, canvas.width, canvas.height, pixels);
}

bool export_plate_png(const PlateTensor2D& plate, const std::string& path) {
  return export_canvas_to_png(canvas_from_plate(plate), path);
}

bool export_film_png(const FilmTensor2D& film, const std::string& path,
                     FilmValueAggregator aggregator) {
  return export_canvas_to_png(canvas_from_film(film, aggregator, true), path);
}

bool export_holographic_plate_png(const HolographicPlateTensor2D& plate,
                                  const std::string& path,
                                  bool count_stencils) {
  return export_canvas_to_png(canvas_from_holographic(plate, count_stencils, true), path);
}

bool export_canvas_rgb(const TensorCanvas2D& r,
                       const TensorCanvas2D& g,
                       const TensorCanvas2D& b,
                       const std::string& path) {
  if (r.width == 0 || r.height == 0) return false;
  if (g.width != r.width || g.height != r.height) return false;
  if (b.width != r.width || b.height != r.height) return false;

  const size_t pixel_count = static_cast<size_t>(r.width) * r.height;
  const std::vector<uint8_t> rpx = r.to_u8_normalized();
  const std::vector<uint8_t> gpx = g.to_u8_normalized();
  const std::vector<uint8_t> bpx = b.to_u8_normalized();
  std::vector<uint8_t> rgb;
  rgb.resize(pixel_count * 3);
  for (size_t i = 0; i < pixel_count; ++i) {
    rgb[i * 3 + 0] = rpx[i];
    rgb[i * 3 + 1] = gpx[i];
    rgb[i * 3 + 2] = bpx[i];
  }
  return write_png_rgb_u8(path, r.width, r.height, rgb);
}

} // namespace nodus::tensors::kpath
