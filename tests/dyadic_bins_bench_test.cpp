#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/dyadic_bins.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_math.h"
#include "common/tensors/abstraction/kpath/kpath_image_export.h"
//#define DYADIC_TEST_LOGGING(...) std::fprintf(stderr, __VA_ARGS__)
#define DYADIC_TEST_LOGGING(...) ((void)0)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace nodus::tensors;
using namespace nodus::tensors::kpath;
using ValueT = uint16_t;
constexpr uint32_t kOutHeight = 1080u;
constexpr uint32_t kOutWidth = 1920u;
constexpr uint32_t kOutChannels = 3u;
static std::vector<ValueT> eigen_scatter_reference(const std::vector<int32_t>& coords,
                                                   const std::vector<ValueT>& values) {
    const size_t out_elems = static_cast<size_t>(kOutHeight) * kOutWidth * kOutChannels;
    std::vector<ValueT> out(out_elems, ValueT{});
    if (coords.empty()) {
        return out;
    }
    const size_t count = coords.size() / 2u;
    for (size_t i = 0; i < count; ++i) {
        const int32_t x = coords[i * 2 + 0];
        const int32_t y = coords[i * 2 + 1];
        if (x < 0 || y < 0 ||
            x >= static_cast<int32_t>(kOutWidth) ||
            y >= static_cast<int32_t>(kOutHeight)) {
            continue;
        }
        const size_t base = (static_cast<size_t>(y) * kOutWidth + static_cast<size_t>(x)) * kOutChannels;
        const size_t vbase = i * kOutChannels;
        for (uint32_t c = 0; c < kOutChannels; ++c) {
            out[base + c] = static_cast<ValueT>(out[base + c] + values[vbase + c]);
        }
    }
    return out;
}

static std::string make_output_path_next_to_exe(const char* argv0, const char* filename) {
    try {
        std::filesystem::path exe_path(argv0 ? argv0 : "");
        if (!exe_path.empty()) {
            return (exe_path.parent_path() / filename).string();
        }
    } catch (...) {
    }
    return std::string(filename);
}

static bool validate_png_output(const char* label,
                                const AbstractTensor& output_tensor,
                                InMemoryBackend& backend) {
    if (!output_tensor.valid()) {
        DYADIC_TEST_LOGGING(stderr, "%s: export precheck failed (invalid output tensor)\n", label);
        return false;
    }
    const TensorDesc& desc = output_tensor.desc();
    if (desc.shape.rank() != 3) {
        DYADIC_TEST_LOGGING(stderr, "%s: export precheck failed (rank %u)\n",
                     label, desc.shape.rank());
        return false;
    }
    if (desc.layout != TensorLayout::Dense) {
        DYADIC_TEST_LOGGING(stderr, "%s: export precheck failed (layout %d)\n",
                     label, static_cast<int>(desc.layout));
        return false;
    }
    const uint32_t channels = desc.shape.dims[2];
    if (!(channels == 1u || channels == 3u || channels == 4u)) {
        DYADIC_TEST_LOGGING(stderr, "%s: export precheck failed (channels %u)\n",
                     label, channels);
        return false;
    }
    void* output_ptr = nullptr;
    size_t output_bytes = 0;
    if (!backend.map(output_tensor.handle(), &output_ptr, &output_bytes)) {
        DYADIC_TEST_LOGGING(stderr, "%s: export precheck failed (map output)\n", label);
        return false;
    }
    const size_t elem_bytes = tensor_dtype_size_bytes(desc.dtype);
    const size_t elem_count = desc.shape.element_count();
    const size_t expected_bytes = elem_count * elem_bytes;
    backend.unmap(output_tensor.handle());
    if (elem_bytes == 0 || output_bytes < expected_bytes) {
        DYADIC_TEST_LOGGING(stderr, "%s: export precheck failed (bytes %zu < %zu)\n",
                     label, output_bytes, expected_bytes);
        return false;
    }
    return true;
}

}

struct CaseResult {
    uint32_t mismatches = 0;
};

struct Quantiles {
    double p50 = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
};

static double quantile_sorted(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double pos = q * static_cast<double>(sorted.size() - 1u);
    const size_t idx = static_cast<size_t>(pos);
    const size_t idx_next = std::min(sorted.size() - 1u, idx + 1u);
    const double frac = pos - static_cast<double>(idx);
    return sorted[idx] + (sorted[idx_next] - sorted[idx]) * frac;
}

static Quantiles compute_quantiles(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    Quantiles q{};
    q.p50 = quantile_sorted(values, 0.50);
    q.p90 = quantile_sorted(values, 0.90);
    q.p99 = quantile_sorted(values, 0.99);
    return q;
}

static CaseResult run_case(const char* label,
                           uint32_t index_count,
                           std::mt19937& rng,
                           const char* argv0,
                           nodus::tensors::InMemoryBackend& backend,
                           nodus::tensors::AbstractTensorPool& pool) {
    const uint32_t iterations = (index_count >= 1'000'000u) ? 10u
        : (index_count >= 100'000u) ? 20u : 50u;
    std::uniform_int_distribution<int32_t> dist_x(0, static_cast<int32_t>(kOutWidth - 1u));
    std::uniform_int_distribution<int32_t> dist_y(0, static_cast<int32_t>(kOutHeight - 1u));
    std::uniform_int_distribution<uint32_t> dist_val(0u, (uint32_t)std::numeric_limits<ValueT>::max());

    nodus::tensors::TensorDesc desc_points{};
    desc_points.dtype = nodus::tensors::TensorDType::I32;
    desc_points.layout = nodus::tensors::TensorLayout::Dense;
    desc_points.shape.dims = { index_count, 2u };

    nodus::tensors::TensorDesc desc_values{};
    desc_values.dtype = dyadic_value_dtype<ValueT>();
    desc_values.layout = nodus::tensors::TensorLayout::Dense;
    desc_values.shape.dims = { index_count, kOutChannels };

    nodus::tensors::TensorDesc desc_output{};
    desc_output.dtype = dyadic_value_dtype<ValueT>();
    desc_output.layout = nodus::tensors::TensorLayout::Dense;
    desc_output.shape.dims = { kOutHeight, kOutWidth, kOutChannels };

    std::vector<double> impl_times;
    std::vector<double> ref_times;
    impl_times.reserve(iterations);
    ref_times.reserve(iterations);

    std::vector<int32_t> coords(static_cast<size_t>(index_count) * 2u);
    std::vector<ValueT> values(static_cast<size_t>(index_count) * kOutChannels);
    std::vector<uint32_t> order(index_count);
    std::vector<int32_t> coords_shuf(static_cast<size_t>(index_count) * 2u);
    std::vector<ValueT> values_shuf(static_cast<size_t>(index_count) * kOutChannels);
    uint32_t mismatch_count = 0;

    // iter==0 is an untimed warm-up: it pays for first-touch pool/arena
    // allocation (and, for "small", the very first scatter call in the
    // whole binary) so that cost doesn't pollute the p50/p90/p99 numbers
    // below, especially for "large" where only 10 real iterations are
    // sampled and a single cold iteration would visibly skew p90/p99.
    for (uint32_t iter = 0; iter < iterations + 1u; ++iter) {
        const bool is_warmup = (iter == 0u);
        for (uint32_t i = 0; i < index_count; ++i) {
            const int32_t x = dist_x(rng);
            const int32_t y = dist_y(rng);
            const size_t cbase = static_cast<size_t>(i) * 2u;
            coords[cbase + 0] = x;
            coords[cbase + 1] = y;
            const size_t vbase = static_cast<size_t>(i) * kOutChannels;
            for (uint32_t c = 0; c < kOutChannels; ++c) {
                values[vbase + c] = (ValueT)dist_val(rng);
            }
        }
        std::iota(order.begin(), order.end(), 0u);
        std::shuffle(order.begin(), order.end(), rng);
        for (uint32_t i = 0; i < index_count; ++i) {
            const uint32_t src = order[i];
            const size_t src_cbase = static_cast<size_t>(src) * 2u;
            const size_t dst_cbase = static_cast<size_t>(i) * 2u;
            coords_shuf[dst_cbase + 0] = coords[src_cbase + 0];
            coords_shuf[dst_cbase + 1] = coords[src_cbase + 1];
            const size_t src_base = static_cast<size_t>(src) * kOutChannels;
            const size_t dst_base = static_cast<size_t>(i) * kOutChannels;
            for (uint32_t c = 0; c < kOutChannels; ++c) {
                values_shuf[dst_base + c] = values[src_base + c];
            }
        }

        auto points_tensor = pool.acquire_tensor(desc_points, &backend);
        auto values_tensor = pool.acquire_tensor(desc_values, &backend);
        nodus::tensors::AbstractTensor output_tensor = nodus::tensors::AbstractTensor::create(desc_output, &backend);
        if (!output_tensor.valid()) {
            DYADIC_TEST_LOGGING(stderr, "%s: output tensor create failed\n", label);
            return {1u};
        }

        void* points_ptr = nullptr;
        size_t points_bytes = 0;
        if (!backend.map(points_tensor.handle(), &points_ptr, &points_bytes)) {
            DYADIC_TEST_LOGGING(stderr, "%s: points map failed\n", label);
            return {1u};
        }
        auto* points_out = static_cast<int32_t*>(points_ptr);
        for (uint32_t i = 0; i < index_count; ++i) {
            const size_t cbase = static_cast<size_t>(i) * 2u;
            points_out[cbase + 0] = coords_shuf[cbase + 0];
            points_out[cbase + 1] = coords_shuf[cbase + 1];
        }
        backend.unmap(points_tensor.handle());
        void* values_ptr = nullptr;
        size_t values_bytes = 0;
        if (!backend.map(values_tensor.handle(), &values_ptr, &values_bytes)) {
            DYADIC_TEST_LOGGING(stderr, "%s: values map failed\n", label);
            return {1u};
        }
        std::memcpy(values_ptr, values_shuf.data(), values_shuf.size() * sizeof(ValueT));
        backend.unmap(values_tensor.handle());

        backend.ensure_zeroed(output_tensor.handle(), output_tensor.desc());
        const auto t0 = std::chrono::high_resolution_clock::now();
        nodus::tensors::TensorTransferConfig cfg{};
        cfg.premix_scatter = nodus::tensors::TensorMixPolicy::Add;
        cfg.postmix_scatter = nodus::tensors::TensorMixPolicy::Add;
        cfg.clamp = true;
        // Without this, dyadic_thread_count_from_overrides() defaults to 1
        // (see tensor_math.cpp) and the whole multithreaded engine runs
        // single-threaded silently -- this benchmark previously never
        // exercised the threading it exists to validate.
        const uint32_t hw_threads = std::thread::hardware_concurrency();
        cfg.thread_count = hw_threads >= 2u ? hw_threads : 8u;

        if (!values_tensor.scatter(nodus::tensors::AbstractTensor{}, output_tensor, points_tensor, cfg)) {
            DYADIC_TEST_LOGGING(stderr, "%s: scatter failed\n", label);
            return {1u};
        }
        const auto t1 = std::chrono::high_resolution_clock::now();

        {
            const std::string path = make_output_path_next_to_exe(
                argv0, (std::string("dyadic_bins_") + label + "_iter_" + std::to_string(iter) + ".png").c_str());
            if (!validate_png_output(label, output_tensor, backend)) {
                return {1u};
            }
            if (!export_tensor_png(output_tensor, path, true)) {
                DYADIC_TEST_LOGGING(stderr, "%s: export_tensor_png failed (%s)\n", label, path.c_str());
                return {1u};
            }
        }

        const auto tref0 = std::chrono::high_resolution_clock::now();
        const std::vector<ValueT> ref_stage = eigen_scatter_reference(coords_shuf, values_shuf);
        const auto tref1 = std::chrono::high_resolution_clock::now();

        const double impl_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double ref_ms = std::chrono::duration<double, std::milli>(tref1 - tref0).count();
        if (!is_warmup) {
            impl_times.push_back(impl_ms);
            ref_times.push_back(ref_ms);
        }

        void* output_ptr = nullptr;
        size_t output_bytes = 0;
        if (!backend.map(output_tensor.handle(), &output_ptr, &output_bytes)) {
            DYADIC_TEST_LOGGING(stderr, "%s: output map failed\n", label);
            return {1u};
        }
        const auto* output = static_cast<const ValueT*>(output_ptr);
        const size_t output_elems = output_tensor.desc().shape.element_count();
        for (size_t i = 0; i < output_elems; ++i) {
            if (output[i] != ref_stage[i]) {
                mismatch_count += 1u;
            }
        }
        backend.unmap(output_tensor.handle());

    }

    const Quantiles impl_q = compute_quantiles(std::move(impl_times));
    const Quantiles ref_q = compute_quantiles(std::move(ref_times));
    std::printf("%s: impl p50 %.3f ms p90 %.3f ms p99 %.3f ms | ref p50 %.3f ms p90 %.3f ms p99 %.3f ms\n",
                label, impl_q.p50, impl_q.p90, impl_q.p99, ref_q.p50, ref_q.p90, ref_q.p99);

    if (mismatch_count == 0) {
        std::printf("%s: ok (%u iters)\n", label, iterations);
        return {0u};
    }

    std::printf("%s: %u mismatches\n", label, mismatch_count);
    return {mismatch_count};
}

int main(int argc, char** argv) {
    nodus::tensors::InMemoryBackend backend;
    nodus::tensors::AbstractTensorPool pool;

    const uint32_t small_count = 8u;
    const uint32_t medium_count = 100'000u;
    const uint32_t dense_count = 1'000'000u;
    std::mt19937 rng(1337u);

    const char* argv0 = (argc > 0) ? argv[0] : "test_dyadic_bins_bench";
    const CaseResult small_result = run_case("small", small_count, rng, argv0, backend, pool);
    const CaseResult medium_result = run_case("medium", medium_count, rng, argv0, backend, pool);
    const CaseResult large_result = run_case("large", dense_count, rng, argv0, backend, pool);

    wait_for_pending_image_saves();
    const bool ok = (small_result.mismatches == 0) &&
                    (medium_result.mismatches == 0) &&
                    (large_result.mismatches == 0);
    return ok ? 0 : 1;
}
