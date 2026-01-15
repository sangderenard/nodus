#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/kpath/kpath_film.h"
#include "common/tensors/abstraction/tensor_compare.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

using namespace nodus::tensors;
using namespace nodus::tensors::kpath;

static bool require_or_report(bool condition, const char* what) {
  if (condition) return true;
  std::cerr << "[KP-FILM] FAILED: " << what << "\n";
  return false;
}

static std::vector<float> tensor_to_floats(const AbstractTensor& tensor) {
  std::vector<float> out;
  if (!tensor.valid()) return out;
  auto* backend = dynamic_cast<InMemoryBackend*>(tensor.backend());
  if (!backend) return out;
  void* ptr = nullptr;
  size_t bytes = 0;
  if (!backend->map(tensor.handle(), &ptr, &bytes)) return out;
  const size_t elem_count = static_cast<size_t>(tensor.desc().shape.element_count());
  const size_t avail = bytes / sizeof(float);
  const size_t copy_count = std::min(elem_count, avail);
  out.assign(elem_count, 0.0f);
  if (copy_count > 0) {
    std::memcpy(out.data(), ptr, copy_count * sizeof(float));
  }
  backend->unmap(tensor.handle());
  return out;
}

static bool check_vector(const std::vector<float>& got,
                         const std::vector<float>& expected,
                         const char* label) {
  if (got.size() != expected.size()) {
    std::cerr << "[KP-FILM] size mismatch for " << label << ": expected " << expected.size()
              << " got " << got.size() << "\n";
    return false;
  }
  constexpr float kEpsilon = 1e-4f;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(got[i] - expected[i]) > kEpsilon) {
      std::cerr << "[KP-FILM] value mismatch for " << label << " at [" << i << "]: expected "
                << expected[i] << " got " << got[i] << "\n";
      return false;
    }
  }
  return true;
}

static std::vector<FilmHistogramDescriptor> make_test_descriptors() {
  FilmHistogramDescriptor coarse;
  coarse.basis.name = "coarse";
  coarse.layer_reactance = {1.0f};
  FilmHistogramDescriptor detail;
  detail.basis.name = "detail";
  detail.layer_reactance = {1.0f, 0.5f};
  return {std::move(coarse), std::move(detail)};
}

static AbstractTensor make_dense_tensor(TensorBackend* backend,
                                        const TensorDesc& desc,
                                        const void* bytes,
                                        size_t byte_count) {
  return tensor_from_bytes(desc, backend, bytes, byte_count);
}

static bool run_reduce_all_test(TensorBackend* backend) {
  FilmTensor film;
  if (!film.configure(2, 2, make_test_descriptors())) {
    return require_or_report(false, "film.configure");
  }

  TensorDesc exposures_desc;
  exposures_desc.dtype = TensorDType::F32;
  exposures_desc.layout = TensorLayout::Dense;
  exposures_desc.shape.dims = {film.height(), film.width(), film.total_channels()};
  const std::vector<float> exposures_data = {
      1.0f, 2.0f, 3.0f,  // pix (0,0)
      4.0f, 5.0f, 6.0f,  // pix (1,0)
      7.0f, 8.0f, 9.0f,  // pix (0,1)
      10.0f, 11.0f, 12.0f // pix (1,1)
  };
  if (exposures_data.size() != exposures_desc.shape.element_count()) {
    return require_or_report(false, "exposure data size (histogram)");
  }
  const AbstractTensor exposures = make_dense_tensor(
      backend, exposures_desc, exposures_data.data(), exposures_data.size() * sizeof(float));
  if (!require_or_report(exposures.valid(), "make exposures tensor")) return false;

  TensorDesc out_desc;
  out_desc.dtype = TensorDType::F32;
  out_desc.layout = TensorLayout::Dense;
  out_desc.shape.dims = {film.height(), film.width(), film.histogram_count()};
  const size_t out_elements = static_cast<size_t>(out_desc.shape.element_count());
  std::vector<float> out_scratch(out_elements, 0.0f);
  AbstractTensor out_all = make_dense_tensor(
      backend, out_desc, out_scratch.data(), out_scratch.size() * sizeof(float));
  if (!require_or_report(out_all.valid(), "allocate out tensor")) return false;

  if (!require_or_report(film.reduce_all(exposures, &out_all), "film.reduce_all")) {
    return false;
  }

  const std::vector<float> got = tensor_to_floats(out_all);
  const std::vector<float> expected = {1.0f, 5.0f, 4.0f, 11.0f, 7.0f, 17.0f, 10.0f, 23.0f};
  return check_vector(got, expected, "reduce_all");
}

static bool run_reduce_histogram_test(TensorBackend* backend) {
  FilmTensor film;
  if (!film.configure(2, 2, make_test_descriptors())) {
    return require_or_report(false, "film.configure (histogram)");
  }

  TensorDesc exposures_desc;
  exposures_desc.dtype = TensorDType::F32;
  exposures_desc.layout = TensorLayout::Dense;
  exposures_desc.shape.dims = {film.height(), film.width(), film.total_channels()};
  const std::vector<float> exposures_data = {
      1.0f, 2.0f, 3.0f,  // pix (0,0)
      4.0f, 5.0f, 6.0f,  // pix (1,0)
      7.0f, 8.0f, 9.0f,  // pix (0,1)
      10.0f, 11.0f, 12.0f // pix (1,1)
  };
  const AbstractTensor exposures = make_dense_tensor(
      backend, exposures_desc, exposures_data.data(), exposures_data.size() * sizeof(float));
  if (!require_or_report(exposures.valid(), "make exposures tensor (histogram)")) return false;

  TensorDesc hist_out_desc;
  hist_out_desc.dtype = TensorDType::F32;
  hist_out_desc.layout = TensorLayout::Dense;
  hist_out_desc.shape.dims = {film.height(), film.width()};
  const size_t hist_elements = static_cast<size_t>(hist_out_desc.shape.element_count());
  std::vector<float> hist_scratch(hist_elements, 0.0f);
  AbstractTensor hist_out = make_dense_tensor(
      backend, hist_out_desc, hist_scratch.data(), hist_scratch.size() * sizeof(float));
  if (!require_or_report(hist_out.valid(), "allocate histogram out")) return false;

  if (!require_or_report(film.reduce_histogram(exposures, 1, &hist_out), "film.reduce_histogram")) {
    return false;
  }

  const std::vector<float> got = tensor_to_floats(hist_out);
  const std::vector<float> expected = {5.0f, 11.0f, 17.0f, 23.0f};
  return check_vector(got, expected, "reduce_histogram");
}

int main() {
  register_in_memory_backend(true);
  TensorBackend* backend = &in_memory_backend_singleton();
  const bool all = run_reduce_all_test(backend);
  const bool hist = run_reduce_histogram_test(backend);
  return (all && hist) ? 0 : 1;
}
