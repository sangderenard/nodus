#include "common/tensors/abstraction/kpath/kpath_film.h"

#include "common/tensors/abstraction/tensor_math.h"

namespace nodus::tensors::kpath {

bool FilmTensor::configure(uint32_t width,
                           uint32_t height,
                           std::vector<FilmHistogramDescriptor> descriptors) {
  width_ = width;
  height_ = height;
  histograms_.clear();
  total_channels_ = 0;

  for (auto& descriptor : descriptors) {
    const uint32_t channel_count = descriptor.channel_count();
    if (channel_count == 0) {
      continue;
    }
    HistogramMeta meta;
    meta.desc = std::move(descriptor);
    meta.offset = total_channels_;
    histograms_.push_back(std::move(meta));
    total_channels_ += channel_count;
  }

  return width_ != 0 && height_ != 0 && !histograms_.empty() && total_channels_ != 0;
}

AbstractTensor FilmTensor::histogram_tensor(const AbstractTensor& exposures, uint32_t index) const {
  if (!valid()) return {};
  if (index >= histogram_count()) return {};
  if (!exposures.valid()) return {};
  const TensorDesc& desc = exposures.desc();
  if (desc.dtype != TensorDType::F32 || desc.layout != TensorLayout::Dense) return {};
  if (desc.shape.dims.size() != 3) return {};
  if (desc.shape.dims[0] != height_ || desc.shape.dims[1] != width_) return {};

  const auto& meta = histograms_[index];
  const uint32_t channel_count = meta.desc.channel_count();
  if (channel_count == 0) return {};

  TensorIndexSpec spec;
  spec.dims = {
      TensorSlice::all(),
      TensorSlice::all(),
      TensorSlice{
          .start = static_cast<int64_t>(meta.offset),
          .stop = static_cast<int64_t>(meta.offset + channel_count),
          .step = 1,
          .is_all = false,
      },
  };

  return exposures(spec);
}

bool FilmTensor::reduce_histogram(const AbstractTensor& exposures, uint32_t index, AbstractTensor* out) const {
  if (!out || !out->valid()) return false;
  const AbstractTensor hist = histogram_tensor(exposures, index);
  if (!hist.valid()) return false;
  return tensor_reduce_sum_axis_f32(hist, 2, out);
}

bool FilmTensor::reduce_all(const AbstractTensor& exposures, AbstractTensor* out) const {
  if (!out || !out->valid()) return false;
  if (!valid()) return false;
  if (histogram_count() == 0) return false;

  const TensorDesc& od = out->desc();
  if (od.dtype != TensorDType::F32 || od.layout != TensorLayout::Dense) return false;
  if (od.shape.dims.size() != 3) return false;
  if (od.shape.dims[0] != height_ || od.shape.dims[1] != width_ ||
      od.shape.dims[2] != histogram_count()) {
    return false;
  }

  bool direct_copy = (total_channels_ == histogram_count());
  if (direct_copy) {
    for (uint32_t i = 0; i < histogram_count(); ++i) {
      if (histograms_[i].desc.channel_count() != 1 || histograms_[i].offset != i) {
        direct_copy = false;
        break;
      }
    }
  }

  if (direct_copy) {
    return tensor_copy_f32_into(exposures, out);
  }

  for (uint32_t i = 0; i < histogram_count(); ++i) {
    TensorIndexSpec spec;
    spec.dims = {TensorSlice::all(), TensorSlice::all(), static_cast<int64_t>(i)};
    AbstractTensor slice = out->slice(spec);
    if (!slice.valid()) return false;
    if (!reduce_histogram(exposures, i, &slice)) return false;
  }

  return true;
}

} // namespace nodus::tensors::kpath
