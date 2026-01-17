#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace nodus::tensors {

template <typename Scalar>
struct SpatialKernelT final {
    int radius = 0;
    std::vector<Scalar> weights; // (2r+1)^2 row-major

    Scalar at(int dx, int dy) const {
        const int s = 2 * radius + 1;
        const int ix = dx + radius;
        const int iy = dy + radius;
        return weights[static_cast<size_t>(iy) * s + ix];
    }
};

namespace detail {

inline constexpr double kEps = 1.0e-9;

template <typename Scalar>
inline double kernel_scale_factor() {
    if constexpr (std::is_floating_point_v<Scalar>) {
        return 1.0;
    } else {
        return static_cast<double>(std::numeric_limits<Scalar>::max());
    }
}

template <typename Scalar>
inline Scalar kernel_quantize(double w) {
    if constexpr (std::is_floating_point_v<Scalar>) {
        return static_cast<Scalar>(w);
    } else {
        const double scaled = w * kernel_scale_factor<Scalar>();
        const double clamped = std::min(std::max(scaled, 0.0), kernel_scale_factor<Scalar>());
        return static_cast<Scalar>(std::llround(clamped));
    }
}

} // namespace detail

// Gaussian kernel (runtime parameterized). Compiled per dtype.
template <typename Scalar>
inline SpatialKernelT<Scalar> make_spatial_kernel_gaussian_t(double sigma_px) {
    SpatialKernelT<Scalar> k;
    const double sigma = std::max(sigma_px, 0.0);
    if (sigma <= 0.5) {
        k.radius = 0;
        k.weights = {detail::kernel_quantize<Scalar>(1.0)};
        return k;
    }

    const double sigma_clamped = std::max(sigma, 0.25);
    k.radius = static_cast<int>(std::ceil(3.0 * sigma_clamped));
    const int s = 2 * k.radius + 1;
    k.weights.resize(static_cast<size_t>(s) * s);

    const double inv2 = 1.0 / (2.0 * sigma_clamped * sigma_clamped);
    double sum = 0.0;
    for (int y = -k.radius; y <= k.radius; ++y) {
        for (int x = -k.radius; x <= k.radius; ++x) {
            const double r2 = static_cast<double>(x * x + y * y);
            const double w = std::exp(-r2 * inv2);
            k.weights[static_cast<size_t>(y + k.radius) * s + (x + k.radius)] =
                detail::kernel_quantize<Scalar>(w);
            sum += w;
        }
    }

    sum = std::max(sum, detail::kEps);
    const double inv_sum = 1.0 / sum;
    if constexpr (std::is_floating_point_v<Scalar>) {
        for (auto& w : k.weights) w = static_cast<Scalar>(static_cast<double>(w) * inv_sum);
    } else {
        for (auto& w : k.weights) w = detail::kernel_quantize<Scalar>(static_cast<double>(w) * inv_sum);
    }
    return k;
}

// Airy kernel (runtime parameterized). LUT optional.
template <typename Scalar>
inline SpatialKernelT<Scalar> make_spatial_kernel_airy_t(double radius_px,
                                                         double alpha,
                                                         double lut_step_px,
                                                         std::span<const float> lut = {}) {
    SpatialKernelT<Scalar> k;
    const double radius = std::max(radius_px, 0.0);
    if (radius <= 0.5) {
        k.radius = 0;
        k.weights = {detail::kernel_quantize<Scalar>(1.0)};
        return k;
    }

    const bool use_lut = (!lut.empty() && lut_step_px > 0.0);
    k.radius = static_cast<int>(std::ceil(radius));
    const int s = 2 * k.radius + 1;
    k.weights.resize(static_cast<size_t>(s) * s, detail::kernel_quantize<Scalar>(0.0));

    double sum = 0.0;
    for (int y = -k.radius; y <= k.radius; ++y) {
        for (int x = -k.radius; x <= k.radius; ++x) {
            const double r = std::sqrt(static_cast<double>(x * x + y * y));
            if (r > radius) continue;
            double w = 0.0;
            if (use_lut) {
                const double idx = r / lut_step_px;
                const size_t i0 = static_cast<size_t>(std::floor(idx));
                const size_t i1 = std::min(i0 + 1, lut.size() - 1);
                const double t = idx - static_cast<double>(i0);
                const double v0 = static_cast<double>(lut[i0]);
                const double v1 = static_cast<double>(lut[i1]);
                w = v0 + (v1 - v0) * t;
            } else {
                const double xarg = alpha * r;
                if (xarg <= detail::kEps) {
                    w = 1.0;
                } else {
                    const double j1 = std::cyl_bessel_j(1, xarg);
                    const double v = (2.0 * j1 / xarg);
                    w = v * v;
                }
            }
            k.weights[static_cast<size_t>(y + k.radius) * s + (x + k.radius)] =
                detail::kernel_quantize<Scalar>(w);
            sum += w;
        }
    }

    sum = std::max(sum, detail::kEps);
    const double inv_sum = 1.0 / sum;
    if constexpr (std::is_floating_point_v<Scalar>) {
        for (auto& w : k.weights) w = static_cast<Scalar>(static_cast<double>(w) * inv_sum);
    } else {
        for (auto& w : k.weights) w = detail::kernel_quantize<Scalar>(static_cast<double>(w) * inv_sum);
    }
    return k;
}

#define NODUS_KERNEL_DEFINE_TYPED(SUFFIX, SCALAR)                                              \
    inline SpatialKernelT<SCALAR> make_spatial_kernel_gaussian_##SUFFIX(double sigma_px) {     \
        return make_spatial_kernel_gaussian_t<SCALAR>(sigma_px);                               \
    }                                                                                           \
    inline SpatialKernelT<SCALAR> make_spatial_kernel_airy_##SUFFIX(double radius_px,          \
                                                                    double alpha,              \
                                                                    double lut_step_px,         \
                                                                    std::span<const float> lut = {}) { \
        return make_spatial_kernel_airy_t<SCALAR>(radius_px, alpha, lut_step_px, lut);          \
    }

NODUS_KERNEL_DEFINE_TYPED(i8, int8_t)
NODUS_KERNEL_DEFINE_TYPED(i16, int16_t)
NODUS_KERNEL_DEFINE_TYPED(i32, int32_t)
NODUS_KERNEL_DEFINE_TYPED(i64, int64_t)
NODUS_KERNEL_DEFINE_TYPED(u8, uint8_t)
NODUS_KERNEL_DEFINE_TYPED(u16, uint16_t)
NODUS_KERNEL_DEFINE_TYPED(u32, uint32_t)
NODUS_KERNEL_DEFINE_TYPED(u64, uint64_t)
NODUS_KERNEL_DEFINE_TYPED(f32, float)
NODUS_KERNEL_DEFINE_TYPED(f64, double)

#undef NODUS_KERNEL_DEFINE_TYPED

} // namespace nodus::tensors
