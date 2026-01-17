#include "common/tensors/abstraction/kpath/kpath_image_export.h"

#include "common/tensors/abstraction/in_memory_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <condition_variable>
#include <deque>
#include <thread>
#include <type_traits>
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

static bool map_in_memory_tensor(const AbstractTensor& tensor, void** out_data,
                                 size_t* out_bytes, InMemoryBackend** out_backend) {
  if (!tensor.valid() || !tensor.backend()) return false;
  auto* backend = dynamic_cast<InMemoryBackend*>(tensor.backend());
  if (!backend) return false;
  if (!backend->map(tensor.handle(), out_data, out_bytes)) return false;
  if (out_backend) *out_backend = backend;
  return true;
}

static void unmap_in_memory_tensor(InMemoryBackend* backend, const AbstractTensor& tensor) {
  if (backend) backend->unmap(tensor.handle());
}

static bool is_float_dtype(TensorDType dtype) {
  return dtype == TensorDType::F32 || dtype == TensorDType::F64;
}

static bool is_unsigned_dtype(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::U8:
    case TensorDType::U16:
    case TensorDType::U32:
    case TensorDType::U64:
      return true;
    default:
      return false;
  }
}

static double dtype_max_value(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::U8: return static_cast<double>(std::numeric_limits<uint8_t>::max());
    case TensorDType::U16: return static_cast<double>(std::numeric_limits<uint16_t>::max());
    case TensorDType::U32: return static_cast<double>(std::numeric_limits<uint32_t>::max());
    case TensorDType::U64: return static_cast<double>(std::numeric_limits<uint64_t>::max());
    case TensorDType::I8: return static_cast<double>(std::numeric_limits<int8_t>::max());
    case TensorDType::I16: return static_cast<double>(std::numeric_limits<int16_t>::max());
    case TensorDType::I32: return static_cast<double>(std::numeric_limits<int32_t>::max());
    case TensorDType::I64: return static_cast<double>(std::numeric_limits<int64_t>::max());
    default: return 1.0;
  }
}

static float read_value_as_float(const void* data, TensorDType dtype, size_t idx) {
  switch (dtype) {
    case TensorDType::F32:
      return static_cast<const float*>(data)[idx];
    case TensorDType::F64:
      return static_cast<float>(static_cast<const double*>(data)[idx]);
    case TensorDType::I8:
      return static_cast<float>(static_cast<const int8_t*>(data)[idx]);
    case TensorDType::I16:
      return static_cast<float>(static_cast<const int16_t*>(data)[idx]);
    case TensorDType::I32:
      return static_cast<float>(static_cast<const int32_t*>(data)[idx]);
    case TensorDType::I64:
      return static_cast<float>(static_cast<const int64_t*>(data)[idx]);
    case TensorDType::U8:
      return static_cast<float>(static_cast<const uint8_t*>(data)[idx]);
    case TensorDType::U16:
      return static_cast<float>(static_cast<const uint16_t*>(data)[idx]);
    case TensorDType::U32:
      return static_cast<float>(static_cast<const uint32_t*>(data)[idx]);
    case TensorDType::U64:
      return static_cast<float>(static_cast<const uint64_t*>(data)[idx]);
    default:
      return 0.0f;
  }
}

static uint8_t to_u8_from_normalized(float v) {
  float t = std::clamp(v, 0.0f, 1.0f);
  return static_cast<uint8_t>(std::lround(t * 255.0f));
}

enum class ImageSaveColorSpace : uint8_t {
  Gray = 1,
  RGB = 3,
  RGBA = 4,
  HSL = 5,
};

struct ImageSaveJob final {
  std::string path;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  bool normalize = true;
  ImageSaveColorSpace colorspace = ImageSaveColorSpace::Gray;
  std::vector<float> values;
};

class ImageSaveQueue final {
public:
  ImageSaveQueue() {
    worker_ = std::thread([this]() { run(); });
  }

  ~ImageSaveQueue() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  void enqueue(ImageSaveJob&& job) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push_back(std::move(job));
    }
    cv_.notify_one();
  }

  void wait_for_idle() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_cv_.wait(lock, [this]() { return queue_.empty() && !busy_; });
  }

private:
  void run() {
    for (;;) {
      ImageSaveJob job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) break;
        job = std::move(queue_.front());
        queue_.pop_front();
        busy_ = true;
      }

      process_job(job);

      {
        std::lock_guard<std::mutex> lock(mutex_);
        busy_ = false;
        if (queue_.empty()) idle_cv_.notify_all();
      }
    }
  }

  void process_job(const ImageSaveJob& job) {
    if (job.width == 0 || job.height == 0 || job.channels == 0) return;
    const size_t pixel_count = static_cast<size_t>(job.width) * job.height;
    const size_t elem_count = pixel_count * job.channels;
    if (job.values.size() < elem_count) return;

    std::vector<float> min_v(job.channels, std::numeric_limits<float>::infinity());
    std::vector<float> max_v(job.channels, -std::numeric_limits<float>::infinity());
    if (job.normalize) {
      for (size_t i = 0; i < pixel_count; ++i) {
        const size_t base = i * job.channels;
        for (uint32_t c = 0; c < job.channels; ++c) {
          float v = job.values[base + c];
          min_v[c] = std::min(min_v[c], v);
          max_v[c] = std::max(max_v[c], v);
        }
      }
    }

    auto norm_value = [&](float v, uint32_t c) -> float {
      if (job.normalize) {
        float denom = std::max(max_v[c] - min_v[c], 1e-8f);
        return std::clamp((v - min_v[c]) / denom, 0.0f, 1.0f);
      }
      return std::clamp(v, 0.0f, 1.0f);
    };

    if (job.colorspace == ImageSaveColorSpace::Gray) {
      std::vector<uint8_t> gray(pixel_count);
      for (size_t i = 0; i < pixel_count; ++i) {
        gray[i] = to_u8_from_normalized(norm_value(job.values[i], 0));
      }
      write_png_grayscale_u8(job.path, job.width, job.height, gray);
      return;
    }

    if (job.colorspace == ImageSaveColorSpace::RGB) {
      std::vector<uint8_t> rgb(pixel_count * 3);
      for (size_t i = 0; i < pixel_count; ++i) {
        const size_t base = i * 3;
        rgb[base + 0] = to_u8_from_normalized(norm_value(job.values[base + 0], 0));
        rgb[base + 1] = to_u8_from_normalized(norm_value(job.values[base + 1], 1));
        rgb[base + 2] = to_u8_from_normalized(norm_value(job.values[base + 2], 2));
      }
      write_png_rgb_u8(job.path, job.width, job.height, rgb);
      return;
    }

    if (job.colorspace == ImageSaveColorSpace::RGBA) {
      std::vector<uint8_t> rgba(pixel_count * 4);
      for (size_t i = 0; i < pixel_count; ++i) {
        const size_t base = i * 4;
        rgba[base + 0] = to_u8_from_normalized(norm_value(job.values[base + 0], 0));
        rgba[base + 1] = to_u8_from_normalized(norm_value(job.values[base + 1], 1));
        rgba[base + 2] = to_u8_from_normalized(norm_value(job.values[base + 2], 2));
        rgba[base + 3] = to_u8_from_normalized(norm_value(job.values[base + 3], 3));
      }
      write_png_rgba_u8(job.path, job.width, job.height, rgba);
      return;
    }

    if (job.colorspace == ImageSaveColorSpace::HSL) {
      std::vector<uint8_t> rgb(pixel_count * 3);
      for (size_t i = 0; i < pixel_count; ++i) {
        const size_t base = i * 3;
        HslColor hsl{};
        hsl.h = norm_value(job.values[base + 0], 0);
        hsl.s = norm_value(job.values[base + 1], 1);
        hsl.l = norm_value(job.values[base + 2], 2);
        const RgbColor rgbf = hsl_to_rgb(hsl);
        rgb[base + 0] = to_u8_from_normalized(rgbf.r);
        rgb[base + 1] = to_u8_from_normalized(rgbf.g);
        rgb[base + 2] = to_u8_from_normalized(rgbf.b);
      }
      write_png_rgb_u8(job.path, job.width, job.height, rgb);
      return;
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable idle_cv_;
  std::deque<ImageSaveJob> queue_;
  std::thread worker_;
  bool stop_ = false;
  bool busy_ = false;
};

static ImageSaveQueue& image_save_queue() {
  static ImageSaveQueue queue;
  return queue;
}

static bool snapshot_tensor_to_float(const AbstractTensor& image,
                                     uint32_t width,
                                     uint32_t height,
                                     uint32_t channels,
                                     std::vector<float>& out_values) {
  void* data = nullptr;
  size_t bytes = 0;
  InMemoryBackend* mem = nullptr;
  if (!map_in_memory_tensor(image, &data, &bytes, &mem)) return false;

  const TensorDesc& desc = image.desc();
  const size_t pixel_count = static_cast<size_t>(width) * height;
  const size_t elem_count = pixel_count * channels;
  const size_t elem_bytes = tensor_dtype_size_bytes(desc.dtype);
  if (elem_bytes == 0 || bytes < elem_count * elem_bytes) {
    unmap_in_memory_tensor(mem, image);
    return false;
  }

  out_values.resize(elem_count);
  for (size_t i = 0; i < elem_count; ++i) {
    out_values[i] = read_value_as_float(data, desc.dtype, i);
  }

  unmap_in_memory_tensor(mem, image);
  return true;
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

RgbColor hsl_to_rgb(const HslColor& hsl) {
  float h = hsl.h;
  float s = hsl.s;
  float l = hsl.l;
  h = h - std::floor(h);
  s = std::clamp(s, 0.0f, 1.0f);
  l = std::clamp(l, 0.0f, 1.0f);

  RgbColor out{};
  if (s <= 0.0f) {
    out.r = l;
    out.g = l;
    out.b = l;
    return out;
  }

  auto hue2rgb = [](float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f / 2.0f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
  };

  float q = (l < 0.5f) ? (l * (1.0f + s)) : (l + s - l * s);
  float p = 2.0f * l - q;
  out.r = hue2rgb(p, q, h + 1.0f / 3.0f);
  out.g = hue2rgb(p, q, h);
  out.b = hue2rgb(p, q, h - 1.0f / 3.0f);
  return out;
}

HslColor rgb_to_hsl(const RgbColor& rgb) {
  float r = std::clamp(rgb.r, 0.0f, 1.0f);
  float g = std::clamp(rgb.g, 0.0f, 1.0f);
  float b = std::clamp(rgb.b, 0.0f, 1.0f);
  float max_v = std::max({r, g, b});
  float min_v = std::min({r, g, b});
  float l = 0.5f * (max_v + min_v);

  HslColor out{};
  out.l = l;

  float delta = max_v - min_v;
  if (delta <= 0.0f) {
    out.h = 0.0f;
    out.s = 0.0f;
    return out;
  }

  out.s = (l < 0.5f) ? (delta / (max_v + min_v)) : (delta / (2.0f - max_v - min_v));
  if (max_v == r) {
    out.h = (g - b) / delta + (g < b ? 6.0f : 0.0f);
  } else if (max_v == g) {
    out.h = (b - r) / delta + 2.0f;
  } else {
    out.h = (r - g) / delta + 4.0f;
  }
  out.h /= 6.0f;
  return out;
}

AbstractTensor make_image_tensor_from_canvas(const TensorCanvas2D& canvas,
                                             TensorBackend* backend) {
  if (canvas.width == 0 || canvas.height == 0) return {};
  TensorBackend* use_backend = backend ? backend : &in_memory_backend_singleton();
  TensorDesc desc{};
  desc.dtype = TensorDType::F32;
  desc.layout = TensorLayout::Dense;
  desc.shape.dims = {canvas.height, canvas.width, 1u};
  AbstractTensor image = AbstractTensor::create(desc, use_backend);
  void* data = nullptr;
  size_t bytes = 0;
  InMemoryBackend* mem = nullptr;
  if (!map_in_memory_tensor(image, &data, &bytes, &mem)) return {};

  const size_t expected = static_cast<size_t>(canvas.width) * canvas.height;
  if (bytes < expected * sizeof(float)) {
    unmap_in_memory_tensor(mem, image);
    return {};
  }
  std::memcpy(data, canvas.values.data(), expected * sizeof(float));
  unmap_in_memory_tensor(mem, image);
  return image;
}

AbstractTensor make_image_tensor_from_canvases_rgb(const TensorCanvas2D& r,
                                                   const TensorCanvas2D& g,
                                                   const TensorCanvas2D& b,
                                                   TensorBackend* backend) {
  if (r.width == 0 || r.height == 0) return {};
  if (g.width != r.width || g.height != r.height) return {};
  if (b.width != r.width || b.height != r.height) return {};

  TensorBackend* use_backend = backend ? backend : &in_memory_backend_singleton();
  TensorDesc desc{};
  desc.dtype = TensorDType::F32;
  desc.layout = TensorLayout::Dense;
  desc.shape.dims = {r.height, r.width, 3u};
  AbstractTensor image = AbstractTensor::create(desc, use_backend);
  void* data = nullptr;
  size_t bytes = 0;
  InMemoryBackend* mem = nullptr;
  if (!map_in_memory_tensor(image, &data, &bytes, &mem)) return {};

  const size_t pixel_count = static_cast<size_t>(r.width) * r.height;
  if (bytes < pixel_count * 3 * sizeof(float)) {
    unmap_in_memory_tensor(mem, image);
    return {};
  }

  auto* dst = static_cast<float*>(data);
  for (size_t i = 0; i < pixel_count; ++i) {
    dst[i * 3 + 0] = r.values[i];
    dst[i * 3 + 1] = g.values[i];
    dst[i * 3 + 2] = b.values[i];
  }
  unmap_in_memory_tensor(mem, image);
  return image;
}

AbstractTensor make_image_tensor_from_canvases_rgba(const TensorCanvas2D& r,
                                                    const TensorCanvas2D& g,
                                                    const TensorCanvas2D& b,
                                                    const TensorCanvas2D& a,
                                                    TensorBackend* backend) {
  if (r.width == 0 || r.height == 0) return {};
  if (g.width != r.width || g.height != r.height) return {};
  if (b.width != r.width || b.height != r.height) return {};
  if (a.width != r.width || a.height != r.height) return {};

  TensorBackend* use_backend = backend ? backend : &in_memory_backend_singleton();
  TensorDesc desc{};
  desc.dtype = TensorDType::F32;
  desc.layout = TensorLayout::Dense;
  desc.shape.dims = {r.height, r.width, 4u};
  AbstractTensor image = AbstractTensor::create(desc, use_backend);
  void* data = nullptr;
  size_t bytes = 0;
  InMemoryBackend* mem = nullptr;
  if (!map_in_memory_tensor(image, &data, &bytes, &mem)) return {};

  const size_t pixel_count = static_cast<size_t>(r.width) * r.height;
  if (bytes < pixel_count * 4 * sizeof(float)) {
    unmap_in_memory_tensor(mem, image);
    return {};
  }

  auto* dst = static_cast<float*>(data);
  for (size_t i = 0; i < pixel_count; ++i) {
    dst[i * 4 + 0] = r.values[i];
    dst[i * 4 + 1] = g.values[i];
    dst[i * 4 + 2] = b.values[i];
    dst[i * 4 + 3] = a.values[i];
  }
  unmap_in_memory_tensor(mem, image);
  return image;
}

bool export_tensor_png(const AbstractTensor& image, const std::string& path, bool normalize) {
  if (!image.valid()) return false;
  const TensorDesc& desc = image.desc();
  if (desc.shape.rank() != 3) return false;
  if (desc.layout != TensorLayout::Dense) return false;

  const uint32_t height = desc.shape.dims[0];
  const uint32_t width = desc.shape.dims[1];
  const uint32_t channels = desc.shape.dims[2];
  if (height == 0 || width == 0) return false;
  if (!(channels == 1 || channels == 3 || channels == 4)) return false;

  const size_t pixel_count = static_cast<size_t>(width) * height;
  std::vector<float> values;
  if (!snapshot_tensor_to_float(image, width, height, channels, values)) return false;

  ImageSaveJob job{};
  job.path = path;
  job.width = width;
  job.height = height;
  job.channels = channels;
  job.normalize = normalize;
  job.values = std::move(values);
  if (channels == 1) job.colorspace = ImageSaveColorSpace::Gray;
  else if (channels == 3) job.colorspace = ImageSaveColorSpace::RGB;
  else job.colorspace = ImageSaveColorSpace::RGBA;

  image_save_queue().enqueue(std::move(job));
  return true;
}

bool export_hsl_tensor_png(const AbstractTensor& image_hsl, const std::string& path, bool normalize) {
  if (!image_hsl.valid()) return false;
  const TensorDesc& desc = image_hsl.desc();
  if (desc.shape.rank() != 3) return false;
  if (desc.layout != TensorLayout::Dense) return false;
  if (desc.shape.dims[2] != 3) return false;

  const uint32_t height = desc.shape.dims[0];
  const uint32_t width = desc.shape.dims[1];
  if (height == 0 || width == 0) return false;

  const size_t pixel_count = static_cast<size_t>(width) * height;
  std::vector<float> values;
  if (!snapshot_tensor_to_float(image_hsl, width, height, 3, values)) return false;

  ImageSaveJob job{};
  job.path = path;
  job.width = width;
  job.height = height;
  job.channels = 3;
  job.normalize = normalize;
  job.colorspace = ImageSaveColorSpace::HSL;
  job.values = std::move(values);

  image_save_queue().enqueue(std::move(job));
  return true;
}

void wait_for_pending_image_saves() {
  image_save_queue().wait_for_idle();
}

} // namespace nodus::tensors::kpath
