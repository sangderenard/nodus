#include "common/tensors/abstraction/tensor_tiling_strategy.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace nodus::tensors;

namespace {

constexpr uint32_t kDims = 2;
constexpr uint32_t kCount = 1u << 20;
constexpr uint32_t kIters = 8;

inline const char* dtype_name(TensorDType dtype) {
    switch (dtype) {
        case TensorDType::U8: return "U8";
        case TensorDType::U16: return "U16";
        default: return "Unknown";
    }
}

inline TensorDesc make_points_desc(TensorDType dtype) {
    TensorDesc desc{};
    desc.dtype = dtype;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {kCount, kDims};
    return desc;
}

template <typename T>
void fill_points(std::vector<T>& points, std::mt19937& rng) {
    using Lim = std::numeric_limits<T>;
    if constexpr (std::is_signed_v<T>) {
        std::uniform_int_distribution<int> dist(static_cast<int>(Lim::min()), static_cast<int>(Lim::max()));
        for (auto& v : points) v = static_cast<T>(dist(rng));
    } else {
        std::uniform_int_distribution<uint32_t> dist(0u, static_cast<uint32_t>(Lim::max()));
        for (auto& v : points) v = static_cast<T>(dist(rng));
    }
}

template <typename T>
void run_case(TensorDType dtype) {
    std::vector<T> points(static_cast<size_t>(kCount) * kDims);
    std::vector<int64_t> out_coords(static_cast<size_t>(kCount) * kDims);

    std::mt19937 rng(1337u);
    fill_points(points, rng);

    const TensorDesc pd = make_points_desc(dtype);
    const void* points_ptr = points.data();
    const uint32_t* bounds = nullptr;
    const bool points_match = false;
    const bool use_affine = false;
    const float* affine = nullptr;
    const bool clamp_to_bounds = false;
    uint8_t* out_in_bounds = nullptr;

    // Warm-up
    convert_points_to_int_coords(pd, points_ptr, kCount, kDims, points_match,
                                 use_affine, affine, bounds, clamp_to_bounds,
                                 out_coords.data(), out_in_bounds);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < kIters; ++i) {
        convert_points_to_int_coords(pd, points_ptr, kCount, kDims, points_match,
                                     use_affine, affine, bounds, clamp_to_bounds,
                                     out_coords.data(), out_in_bounds);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double per_iter = ms / static_cast<double>(kIters);

    // Simple sanity check on a couple of samples.
    if (!out_coords.empty()) {
        const int64_t x0 = out_coords[0];
        const int64_t y0 = out_coords[1];
        if (x0 == std::numeric_limits<int64_t>::min() || y0 == std::numeric_limits<int64_t>::min()) {
            std::cerr << "Sanity check failed for " << dtype_name(dtype) << "\n";
        }
    }

    std::cout << "Tensor tiling (" << dtype_name(dtype) << ") "
              << kCount << "x" << kDims << " points: "
              << per_iter << " ms/iter (" << kIters << " iters)\n";
}

} // namespace

int main() {
    std::cout << "\nTensor tiling integer timing test\n";
    std::cout << "count=" << kCount << ", dims=" << kDims << ", iters=" << kIters << "\n";

    run_case<uint8_t>(TensorDType::U8);
    run_case<uint16_t>(TensorDType::U16);

    return 0;
}
