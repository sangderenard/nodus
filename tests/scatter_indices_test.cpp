#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_math.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace nodus::tensors;

#define THREADS 12

static TensorDesc make_dense_desc(TensorDType dtype, std::initializer_list<uint32_t> dims) {
    TensorDesc desc{};
    desc.dtype = dtype;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = dims;
    return desc;
}

static void log_tensor_meta(const char* tag, const AbstractTensor& t) {
    const TensorDesc& d = t.desc();
    std::fprintf(stderr, "[scatter_indices][%s] valid=%d handle=%llu dtype=%d layout=%d rank=%zu dims=",
                 tag,
                 t.valid() ? 1 : 0,
                 static_cast<unsigned long long>(t.handle().id),
                 static_cast<int>(d.dtype),
                 static_cast<int>(d.layout),
                 d.shape.dims.size());
    std::fprintf(stderr, "[");
    for (size_t i = 0; i < d.shape.dims.size(); ++i) {
        std::fprintf(stderr, "%u", d.shape.dims[i]);
        if (i + 1u < d.shape.dims.size()) {
            std::fprintf(stderr, ",");
        }
    }
    std::fprintf(stderr, "]\n");
}

static void set_threads_env(bool enable, uint32_t threads = THREADS) {
#if defined(_WIN32)
    if (enable) {
        _putenv_s("NODUS_TENSOR_OP_THREADS", std::to_string(threads).c_str());
    } else {
        _putenv_s("NODUS_TENSOR_OP_THREADS", "");
    }
#else
    if (enable) {
        setenv("NODUS_TENSOR_OP_THREADS", std::to_string(threads).c_str(), 1);
    } else {
        unsetenv("NODUS_TENSOR_OP_THREADS");
    }
#endif
}

static void fill_random_points(AbstractTensor& points,
                               uint32_t count,
                               uint32_t width,
                               uint32_t height,
                               std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> dist_x(0, width - 1);
    std::uniform_int_distribution<uint32_t> dist_y(0, height - 1);

    auto* mem = dynamic_cast<InMemoryBackend*>(points.backend());
    assert(mem);
    void* ptr = nullptr;
    size_t bytes = 0;
    assert(mem->map(points.handle(), &ptr, &bytes));

    if (points.desc().dtype == TensorDType::F32) {
        auto* out = static_cast<float*>(ptr);
        for (uint32_t i = 0; i < count; ++i) {
            out[i * 2 + 0] = static_cast<float>(dist_x(rng));
            out[i * 2 + 1] = static_cast<float>(dist_y(rng));
        }
    } else if (points.desc().dtype == TensorDType::I32) {
        auto* out = static_cast<int32_t*>(ptr);
        for (uint32_t i = 0; i < count; ++i) {
            out[i * 2 + 0] = static_cast<int32_t>(dist_x(rng));
            out[i * 2 + 1] = static_cast<int32_t>(dist_y(rng));
        }
    }

    mem->unmap(points.handle());
}

static void fill_random_values(AbstractTensor& values, uint32_t count, std::mt19937& rng) {
    auto* mem = dynamic_cast<InMemoryBackend*>(values.backend());
    assert(mem);
    void* ptr = nullptr;
    size_t bytes = 0;
    assert(mem->map(values.handle(), &ptr, &bytes));

    const TensorDesc& vd = values.desc();
    const uint32_t channels = (vd.shape.dims.size() > 1) ? vd.shape.dims[1] : 1u;

    if (vd.dtype == TensorDType::F32) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        auto* out = static_cast<float*>(ptr);
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t c = 0; c < channels; ++c) {
                out[i * channels + c] = dist(rng);
            }
        }
    } else if (vd.dtype == TensorDType::I32) {
        std::uniform_int_distribution<int32_t> dist(0, 1024);
        auto* out = static_cast<int32_t*>(ptr);
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t c = 0; c < channels; ++c) {
                out[i * channels + c] = dist(rng);
            }
        }
    } else if (vd.dtype == TensorDType::U8) {
        std::uniform_int_distribution<int32_t> dist(0, 255);
        auto* out = static_cast<uint8_t*>(ptr);
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t c = 0; c < channels; ++c) {
                out[i * channels + c] = static_cast<uint8_t>(dist(rng));
            }
        }
    } else {
        assert(false && "Unsupported values dtype in fill_random_values");
    }

    mem->unmap(values.handle());
}

static void fill_random_dense(AbstractTensor& tensor, std::mt19937& rng) {
    auto* mem = dynamic_cast<InMemoryBackend*>(tensor.backend());
    assert(mem);
    void* ptr = nullptr;
    size_t bytes = 0;
    assert(mem->map(tensor.handle(), &ptr, &bytes));

    const TensorDesc& td = tensor.desc();
    const size_t elems = static_cast<size_t>(td.shape.element_count());
    if (td.dtype == TensorDType::F32) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        auto* out = static_cast<float*>(ptr);
        for (size_t i = 0; i < elems; ++i) {
            out[i] = dist(rng);
        }
    } else if (td.dtype == TensorDType::I32) {
        std::uniform_int_distribution<int32_t> dist(0, 1024);
        auto* out = static_cast<int32_t*>(ptr);
        for (size_t i = 0; i < elems; ++i) {
            out[i] = dist(rng);
        }
    } else if (td.dtype == TensorDType::U8) {
        std::uniform_int_distribution<int32_t> dist(0, 255);
        auto* out = static_cast<uint8_t*>(ptr);
        for (size_t i = 0; i < elems; ++i) {
            out[i] = static_cast<uint8_t>(dist(rng));
        }
    } else {
        assert(false && "Unsupported dense dtype in fill_random_dense");
    }

    mem->unmap(tensor.handle());
}

static TensorStencil build_unit_stencil(TensorDType weight_dtype, TensorBackend* backend) {
    TensorStencil stencil{};
    TensorDesc offsets_desc = make_dense_desc(TensorDType::I32, {1, 2});
    TensorDesc weights_desc = make_dense_desc(weight_dtype, {1});
    stencil.offsets = AbstractTensor::create(offsets_desc, backend);
    stencil.weights = AbstractTensor::create(weights_desc, backend);
    if (!stencil.offsets.valid() || !stencil.weights.valid()) return {};

    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem) return {};

    void* optr = nullptr;
    size_t obytes = 0;
    if (!mem->map(stencil.offsets.handle(), &optr, &obytes)) return {};
    auto* off = static_cast<int32_t*>(optr);
    off[0] = 0;
    off[1] = 0;
    mem->unmap(stencil.offsets.handle());

    void* wptr = nullptr;
    size_t wbytes = 0;
    if (!mem->map(stencil.weights.handle(), &wptr, &wbytes)) return {};
    switch (weight_dtype) {
        case TensorDType::F32:
            static_cast<float*>(wptr)[0] = 1.0f;
            break;
        case TensorDType::F64:
            static_cast<double*>(wptr)[0] = 1.0;
            break;
        case TensorDType::I32:
            static_cast<int32_t*>(wptr)[0] = 1;
            break;
        case TensorDType::U8:
            static_cast<uint8_t*>(wptr)[0] = 1u;
            break;
        default:
            mem->unmap(stencil.weights.handle());
            return {};
    }
    mem->unmap(stencil.weights.handle());

    stencil.support.min_offset = {0, 0};
    stencil.support.max_offset = {0, 0};
    stencil.support.radius = {0, 0};
    return stencil;
}

static TensorFootprint2D build_unit_footprint(TensorDType weight_dtype, TensorBackend* backend) {
    TensorFootprint2D fp{};
    TensorDesc kdesc{};
    kdesc.dtype = weight_dtype;
    kdesc.layout = TensorLayout::Dense;
    kdesc.shape.dims = {1, 1};
    fp.weights = AbstractTensor::create(kdesc, backend);
    if (!fp.weights.valid()) return {};

    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem) return {};
    void* ptr = nullptr;
    size_t bytes = 0;
    if (!mem->map(fp.weights.handle(), &ptr, &bytes)) return {};
    switch (weight_dtype) {
        case TensorDType::F32:
            static_cast<float*>(ptr)[0] = 1.0f;
            break;
        case TensorDType::F64:
            static_cast<double*>(ptr)[0] = 1.0;
            break;
        default:
            mem->unmap(fp.weights.handle());
            return {};
    }
    mem->unmap(fp.weights.handle());

    fp.support.min_offset = {0, 0};
    fp.support.max_offset = {0, 0};
    fp.support.radius = {0, 0};
    return fp;
}

static void report_stub(const char* name, bool ok) {
    if (!ok) {
        std::cout << "stubbed: " << name << "\n";
    }
}

static TensorFootprint2D build_gaussian_footprint(uint32_t ks, float sigma, TensorBackend* backend) {
    TensorFootprint2D fp{};
    TensorDesc kdesc{};
    kdesc.dtype = TensorDType::F32;
    kdesc.layout = TensorLayout::Dense;
    kdesc.shape.dims = {ks, ks};
    fp.weights = AbstractTensor::create(kdesc, backend);
    if (!fp.weights.valid()) return {};

    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    assert(mem);
    void* ptr = nullptr;
    size_t bytes = 0;
    assert(mem->map(fp.weights.handle(), &ptr, &bytes));
    auto* out = static_cast<float*>(ptr);
    const int32_t r = static_cast<int32_t>(ks / 2);
    const float inv2 = 1.0f / (2.0f * sigma * sigma);
    float sum = 0.0f;
    for (int32_t y = -r; y <= r; ++y) {
        for (int32_t x = -r; x <= r; ++x) {
            const float w = std::exp(-(x * x + y * y) * inv2);
            const size_t idx = static_cast<size_t>(y + r) * ks + static_cast<size_t>(x + r);
            out[idx] = w;
            sum += w;
        }
    }
    if (sum > 0.0f) {
        for (uint32_t i = 0; i < ks * ks; ++i) {
            out[i] /= sum;
        }
    }
    mem->unmap(fp.weights.handle());

    fp.support.min_offset = {-r, -r};
    fp.support.max_offset = {r, r};
    fp.support.radius = {r, r};
    return fp;
}

static TensorFootprint2D build_airy_footprint(uint32_t ks, float scale, TensorBackend* backend) {
    TensorFootprint2D fp{};
    TensorDesc kdesc{};
    kdesc.dtype = TensorDType::F32;
    kdesc.layout = TensorLayout::Dense;
    kdesc.shape.dims = {ks, ks};
    fp.weights = AbstractTensor::create(kdesc, backend);
    if (!fp.weights.valid()) return {};

    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    assert(mem);
    void* ptr = nullptr;
    size_t bytes = 0;
    assert(mem->map(fp.weights.handle(), &ptr, &bytes));
    auto* out = static_cast<float*>(ptr);
    const int32_t r = static_cast<int32_t>(ks / 2);
    constexpr float kPi = 3.14159265358979323846f;
    float sum = 0.0f;
    for (int32_t y = -r; y <= r; ++y) {
        for (int32_t x = -r; x <= r; ++x) {
            const float rf = std::sqrt(static_cast<float>(x * x + y * y));
            float w = 1.0f;
            if (rf > 0.0f) {
                const float t = kPi * rf / scale;
                const float j1 = static_cast<float>(std::cyl_bessel_j(1.0f, t));
                const float v = (t == 0.0f) ? 1.0f : (2.0f * j1 / t);
                w = v * v;
            }
            const size_t idx = static_cast<size_t>(y + r) * ks + static_cast<size_t>(x + r);
            out[idx] = w;
            sum += w;
        }
    }
    if (sum > 0.0f) {
        for (uint32_t i = 0; i < ks * ks; ++i) {
            out[i] /= sum;
        }
    }
    mem->unmap(fp.weights.handle());

    fp.support.min_offset = {-r, -r};
    fp.support.max_offset = {r, r};
    fp.support.radius = {r, r};
    return fp;
}

static TensorProbeKernel build_probe_kernel_from_stencil(const TensorStencil& stencil) {
    TensorProbeKernel probe{};
    if (stencil.offsets.valid()) {
        probe.offsets = AbstractTensor::wrap(stencil.offsets.handle(), stencil.offsets.desc(),
                                             stencil.offsets.backend(), false);
    }
    if (stencil.weights.valid()) {
        probe.weights = AbstractTensor::wrap(stencil.weights.handle(), stencil.weights.desc(),
                                             stencil.weights.backend(), false);
    }
    probe.support = stencil.support;
    return probe;
}

static TensorCSR build_csr_from_edges(uint32_t rows,
                                      uint32_t cols,
                                      const std::vector<uint32_t>& edge_rows,
                                      const std::vector<uint32_t>& edge_cols,
                                      const std::vector<float>& edge_w,
                                      TensorBackend* backend) {
    TensorCSR csr{};
    if (edge_rows.size() != edge_cols.size() || edge_rows.size() != edge_w.size()) return csr;

    const uint32_t edges = static_cast<uint32_t>(edge_rows.size());
    std::vector<int32_t> row_ptr(rows + 1, 0);
    for (uint32_t e = 0; e < edges; ++e) {
        const uint32_t r = edge_rows[e];
        if (r < rows) {
            row_ptr[r + 1] += 1;
        }
    }
    for (uint32_t r = 0; r < rows; ++r) {
        row_ptr[r + 1] += row_ptr[r];
    }

    std::vector<int32_t> col(edge_rows.size(), 0);
    std::vector<float> weight(edge_rows.size(), 0.0f);
    std::vector<int32_t> write = row_ptr;
    for (uint32_t e = 0; e < edges; ++e) {
        const uint32_t r = edge_rows[e];
        if (r >= rows) continue;
        const int32_t idx = write[r]++;
        col[idx] = static_cast<int32_t>(edge_cols[e]);
        weight[idx] = edge_w[e];
    }

    TensorDesc rp_desc = make_dense_desc(TensorDType::I32, {static_cast<uint32_t>(row_ptr.size())});
    TensorDesc col_desc = make_dense_desc(TensorDType::I32, {static_cast<uint32_t>(col.size())});
    TensorDesc w_desc = make_dense_desc(TensorDType::F32, {static_cast<uint32_t>(weight.size())});
    csr.row_ptr = AbstractTensor::create(rp_desc, backend);
    csr.col_gli = AbstractTensor::create(col_desc, backend);
    csr.edge_weight = AbstractTensor::create(w_desc, backend);
    if (!csr.row_ptr.valid() || !csr.col_gli.valid() || !csr.edge_weight.valid()) return {};

    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    assert(mem);
    void* ptr = nullptr;
    size_t bytes = 0;
    assert(mem->map(csr.row_ptr.handle(), &ptr, &bytes));
    std::memcpy(ptr, row_ptr.data(), row_ptr.size() * sizeof(int32_t));
    mem->unmap(csr.row_ptr.handle());

    assert(mem->map(csr.col_gli.handle(), &ptr, &bytes));
    std::memcpy(ptr, col.data(), col.size() * sizeof(int32_t));
    mem->unmap(csr.col_gli.handle());

    assert(mem->map(csr.edge_weight.handle(), &ptr, &bytes));
    std::memcpy(ptr, weight.data(), weight.size() * sizeof(float));
    mem->unmap(csr.edge_weight.handle());

    csr.rows = rows;
    csr.cols = cols;
    return csr;
}

int main() {
    register_in_memory_backend(true);
    InMemoryBackend& backend = in_memory_backend_singleton();

    const uint32_t height = 1024;
    const uint32_t width = 1024;
    const uint32_t channels = 3;
    const uint32_t count = 1'000'000;
    const uint32_t rounds = 10;
    TensorDesc base_desc = make_dense_desc(TensorDType::F32, {height, width, channels});
    TensorDesc base_i32_desc = make_dense_desc(TensorDType::I32, {height, width, channels});
    TensorDesc base_u8_desc = make_dense_desc(TensorDType::U8, {height, width, channels});
    TensorDesc points_desc = make_dense_desc(TensorDType::F32, {count, 2});
    TensorDesc points_i_desc = make_dense_desc(TensorDType::I32, {count, 2});
    TensorDesc values_desc = make_dense_desc(TensorDType::F32, {count, channels});
    TensorDesc values_i32_desc = make_dense_desc(TensorDType::I32, {count, channels});
    TensorDesc values_u8_desc = make_dense_desc(TensorDType::U8, {count, channels});

    AbstractTensor points = AbstractTensor::create(points_desc, &backend);
    AbstractTensor points_i = AbstractTensor::create(points_i_desc, &backend);
    AbstractTensor values = AbstractTensor::create(values_desc, &backend);
    AbstractTensor values_i32 = AbstractTensor::create(values_i32_desc, &backend);
    AbstractTensor values_u8 = AbstractTensor::create(values_u8_desc, &backend);
    assert(points.valid());
    assert(points_i.valid());
    assert(values.valid());
    assert(values_i32.valid());
    assert(values_u8.valid());

    std::mt19937 rng(1337u);

    auto scatter_add_dispatch = [&](const AbstractTensor& base_tensor,
                                    const AbstractTensor& points_tensor,
                                    const AbstractTensor& values_tensor,
                                    AbstractTensor* out) -> bool {
        TensorTransferConfig cfg{};
        cfg.premix_scatter = TensorMixPolicy::Add;
        cfg.postmix_scatter = TensorMixPolicy::Add;
        cfg.clamp = true;
        if (!out || !out->valid()) return false;
        static bool logged_first = false;
        if (!logged_first) {
            logged_first = true;
            const char* threads_env = std::getenv("NODUS_TENSOR_OP_THREADS");
            std::fprintf(stderr, "[scatter_indices] first scatter_add_dispatch threads_env=%s\n",
                         threads_env ? threads_env : "(unset)");
            log_tensor_meta("base", base_tensor);
            log_tensor_meta("points", points_tensor);
            log_tensor_meta("values", values_tensor);
            log_tensor_meta("out", *out);
            std::fprintf(stderr, "[scatter_indices] first scatter_add_dispatch begin\n");
        }
        const bool ok = values_tensor.scatter(AbstractTensor{}, *out, points_tensor, cfg);
        if (logged_first) {
            std::fprintf(stderr, "[scatter_indices] first scatter_add_dispatch end ok=%d\n", ok ? 1 : 0);
        }
        return ok;
    };

    auto benchmark = [&](const char* label,
                         const AbstractTensor& base_tensor,
                         const AbstractTensor& points_tensor,
                         AbstractTensor& values_tensor) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            const uint32_t base_h = base_tensor.desc().shape.dims[0];
            const uint32_t base_w = base_tensor.desc().shape.dims[1];
            fill_random_points(const_cast<AbstractTensor&>(points_tensor), count, base_w, base_h, rng);
            fill_random_values(values_tensor, count, rng);
            AbstractTensor out = AbstractTensor::create(base_tensor.desc(), base_tensor.backend());
            if (!out.valid()) return;
            TensorTransferConfig copy_cfg{};
            if (!base_tensor.transfer(AbstractTensor{}, out, AbstractTensor{}, true, copy_cfg)){
                printf("Failed to copy base tensor to out tensor\n");
                return;
            }
            const auto t0 = clock::now();
            bool ok = scatter_add_dispatch(base_tensor, points_tensor, values_tensor, &out);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto gather_dispatch = [&](const AbstractTensor& base_tensor,
                               const AbstractTensor& points_tensor,
                               AbstractTensor* out) -> bool {
        TensorTransferConfig cfg{};
        cfg.clamp = true;
        if (!out) {
            std::cerr << "gather_dispatch: out is null\n";
            return false;
        }
        return base_tensor.gather(points_tensor, *out, AbstractTensor{}, cfg);
    };

    auto benchmark_gather = [&](const char* label,
                                const AbstractTensor& base_tensor,
                                const AbstractTensor& points_tensor) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            const uint32_t base_h = base_tensor.desc().shape.dims[0];
            const uint32_t base_w = base_tensor.desc().shape.dims[1];
            fill_random_points(const_cast<AbstractTensor&>(points_tensor), count, base_w, base_h, rng);
            AbstractTensor out;
            const auto t0 = clock::now();
            bool ok = gather_dispatch(base_tensor, points_tensor, &out);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto gather_add_dispatch = [&](const AbstractTensor& base_tensor,
                                   const AbstractTensor& points_tensor,
                                   AbstractTensor* out) -> bool {
        TensorTransferConfig cfg{};
        cfg.postmix_gather = TensorMixPolicy::Add;
        cfg.clamp = true;
        if (!out) return false;
        return base_tensor.gather(points_tensor, *out, AbstractTensor{}, cfg);
    };

    auto benchmark_gather_add = [&](const char* label,
                                    const AbstractTensor& base_tensor,
                                    const AbstractTensor& points_tensor) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            const uint32_t base_h = base_tensor.desc().shape.dims[0];
            const uint32_t base_w = base_tensor.desc().shape.dims[1];
            fill_random_points(const_cast<AbstractTensor&>(points_tensor), count, base_w, base_h, rng);

            TensorDesc out_desc{};
            out_desc.dtype = base_tensor.desc().dtype;
            out_desc.layout = TensorLayout::Dense;
            const uint32_t channels = (base_tensor.desc().shape.dims.size() == 3)
                                          ? base_tensor.desc().shape.dims[2]
                                          : 1u;
            if (channels == 1) {
                out_desc.shape.dims = {count};
            } else {
                out_desc.shape.dims = {count, channels};
            }
            AbstractTensor out = AbstractTensor::create(out_desc, base_tensor.backend());
            assert(out.valid());

            const auto t0 = clock::now();
            bool ok = gather_add_dispatch(base_tensor, points_tensor, &out);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto benchmark_gather_stencil = [&](const char* label,
                                        const AbstractTensor& base_tensor,
                                        const TensorStencil& stencil) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            AbstractTensor out;
            const auto t0 = clock::now();
            bool ok = tensor_gather_stencil_2d_f32(
                base_tensor, stencil, &out, StencilBoundaryMode::Clamp, nullptr, nullptr, true);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto benchmark_gather_stencil_nd = [&](const char* label,
                                           const AbstractTensor& base_tensor,
                                           const TensorStencil& stencil) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            AbstractTensor out;
            const auto t0 = clock::now();
            bool ok = tensor_gather_stencil_nd_f32(
                base_tensor, stencil, &out, StencilBoundaryMode::Clamp, nullptr, nullptr, true);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto benchmark_gather_footprint = [&](const char* label,
                                          const AbstractTensor& base_tensor,
                                          const TensorFootprint2D& footprint) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            AbstractTensor out;
            const auto t0 = clock::now();
            bool ok = tensor_gather_footprint_2d_f32(
                base_tensor, footprint, &out, StencilOrientation::Correlation,
                StencilBoundaryMode::Clamp, nullptr, nullptr, true);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto benchmark_scatter_stencil_2d = [&](const char* label,
                                            const AbstractTensor& base_tensor,
                                            const AbstractTensor& points_tensor,
                                            const AbstractTensor& values_tensor,
                                            const TensorStencil& stencil) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            const uint32_t base_h = base_tensor.desc().shape.dims[0];
            const uint32_t base_w = base_tensor.desc().shape.dims[1];
            fill_random_points(const_cast<AbstractTensor&>(points_tensor), count, base_w, base_h, rng);
            fill_random_values(const_cast<AbstractTensor&>(values_tensor), count, rng);
            AbstractTensor out;
            const auto t0 = clock::now();
            bool ok = tensor_scatter_stencil_2d_f32(
                base_tensor, points_tensor, values_tensor, stencil, &out,
                StencilBoundaryMode::Clamp, true);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto benchmark_scatter_footprint_2d = [&](const char* label,
                                              const AbstractTensor& base_tensor,
                                              const AbstractTensor& points_tensor,
                                              const AbstractTensor& values_tensor,
                                              const TensorFootprint2D& footprint) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            const uint32_t base_h = base_tensor.desc().shape.dims[0];
            const uint32_t base_w = base_tensor.desc().shape.dims[1];
            fill_random_points(const_cast<AbstractTensor&>(points_tensor), count, base_w, base_h, rng);
            fill_random_values(const_cast<AbstractTensor&>(values_tensor), count, rng);
            AbstractTensor out;
            const auto t0 = clock::now();
            bool ok = tensor_scatter_footprint_2d_f32(
                base_tensor, points_tensor, values_tensor, footprint, &out,
                StencilOrientation::Correlation, StencilBoundaryMode::Clamp, true);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto benchmark_scatter_probe_csr = [&](const char* label,
                                           const AbstractTensor& base_tensor,
                                           const TensorProbeKernel& probe,
                                           const TensorCSR& targets_per_center) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            AbstractTensor out;
            const auto t0 = clock::now();
            bool ok = tensor_scatter_probe_csr_f32(base_tensor, probe, targets_per_center, &out, true);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto benchmark_gather_probe_csr = [&](const char* label,
                                          const AbstractTensor& base_tensor,
                                          const TensorProbeKernel& probe,
                                          const TensorCSR& centers_per_target) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            AbstractTensor out;
            const auto t0 = clock::now();
            bool ok = tensor_gather_probe_csr_f32(base_tensor, probe, centers_per_target, &out, true);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto benchmark_scatter_probe_csr_stencil = [&](const char* label,
                                                    const AbstractTensor& base_tensor,
                                                    const TensorStencil& stencil,
                                                    const TensorCSR& targets_per_center) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            AbstractTensor out;
            const auto t0 = clock::now();
            bool ok = tensor_scatter_probe_csr_stencil_f32(base_tensor, stencil, targets_per_center, &out, true);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto benchmark_gather_probe_csr_stencil = [&](const char* label,
                                                   const AbstractTensor& base_tensor,
                                                   const TensorStencil& stencil,
                                                   const TensorCSR& centers_per_target) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            AbstractTensor out;
            const auto t0 = clock::now();
            bool ok = tensor_gather_probe_csr_stencil_f32(base_tensor, stencil, centers_per_target, &out, true);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto benchmark_scatter_probe_csr_footprint = [&](const char* label,
                                                      const AbstractTensor& base_tensor,
                                                      const TensorFootprint2D& footprint,
                                                      const TensorCSR& targets_per_center) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            AbstractTensor out;
            const auto t0 = clock::now();
            bool ok = tensor_scatter_probe_csr_footprint_f32(
                base_tensor, footprint, StencilOrientation::Correlation, targets_per_center, &out, true);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    auto benchmark_gather_probe_csr_footprint = [&](const char* label,
                                                     const AbstractTensor& base_tensor,
                                                     const TensorFootprint2D& footprint,
                                                     const TensorCSR& centers_per_target) {
        using clock = std::chrono::steady_clock;
        std::chrono::duration<double, std::milli> total{0};
        for (uint32_t r = 0; r < rounds; ++r) {
            AbstractTensor out;
            const auto t0 = clock::now();
            bool ok = tensor_gather_probe_csr_footprint_f32(
                base_tensor, footprint, StencilOrientation::Correlation, centers_per_target, &out, true);
            const auto t1 = clock::now();
            assert(ok);
            total += (t1 - t0);
        }
        const double avg_ms = total.count() / static_cast<double>(rounds);
        std::cout << label << ": " << avg_ms << " ms (avg over " << rounds << ")\n";
    };

    const bool index_modes[] = {false, true};
    const bool affine_modes[] = {false, true};
    const bool saturate_modes[] = {false};

    struct ThreadMode {
        bool use_threads;
        uint32_t threads;
    };

    const ThreadMode thread_modes[] = {
        {false, 1},
        {true, 2},
        {true, 4},
        {true, 8},
        {true, 12}
    };

    struct RunFeatures {
        const char* prefix;
        const char* kernel;
        const char* data_label;
        const char* index_label;
        bool use_affine;
        bool use_saturate;
        bool use_add;
        bool use_threads;
        uint32_t threads;
    };

    auto make_label = [](const RunFeatures& f) {
        std::string label = f.prefix;
        label += ":";
        if (f.kernel) {
            label += " ";
            label += f.kernel;
        }

        bool has_tokens = false;
        auto append = [&](const std::string& token) {
            if (!has_tokens) {
                label += f.kernel ? "+" : " ";
            } else {
                label += "+";
            }
            label += token;
            has_tokens = true;
        };

        if (f.data_label) {
            append(f.data_label);
        }
        if (f.index_label) {
            append(f.index_label);
        }
        append(f.use_affine ? "affine" : "noaffine");
        append(f.use_saturate ? "saturate" : "nosaturate");
        append(f.use_add ? "add" : "nonadd");
        if (f.use_threads) {
            append(std::string("mt") + std::to_string(f.threads));
        } else {
            append("st");
        }

        return label;
    };

    struct ScatterDataMode {
        const char* label;
        const TensorDesc* base_desc;
        AbstractTensor* values_tensor;
        bool allow_float_points;
    };

    ScatterDataMode data_modes[] = {
        {nullptr, &base_desc, &values, true},
        {"i32", &base_i32_desc, &values_i32, false},
        {"u8", &base_u8_desc, &values_u8, false}
    };

    for (const auto& thread_mode : thread_modes) {
        set_threads_env(thread_mode.use_threads, thread_mode.threads);
        for (const auto& data_mode : data_modes) {
            for (bool use_int : index_modes) {
                if (!data_mode.allow_float_points && !use_int) continue;
                for (bool use_affine : affine_modes) {
                    for (bool use_saturate : saturate_modes) {
                        AbstractTensor base = AbstractTensor::create(*data_mode.base_desc, &backend);
                        assert(base.valid());
                        void* base_ptr = nullptr;
                        size_t base_bytes = 0;
                        assert(backend.map(base.handle(), &base_ptr, &base_bytes));
                        std::memset(base_ptr, 0, base_bytes);
                        backend.unmap(base.handle());

                        if (use_affine) {
                            float affine[16] = {
                                1.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 1.0f, 0.0f,
                                8.0f, 5.0f, 0.0f, 1.0f
                            };
                            base.set_slice_affine_row_major(affine);
                        }
                        if (use_saturate) {
                            base.set_slice_saturate_threshold(1.0f);
                        }

                        const AbstractTensor& points_tensor = use_int ? points_i : points;
                        RunFeatures features{
                            "scatter indices",
                            nullptr,
                            data_mode.label,
                            use_int ? "intiger indices" : "float indices",
                            use_affine,
                            use_saturate,
                            true,
                            thread_mode.use_threads,
                            thread_mode.threads
                        };
                        const std::string label = make_label(features);
                        benchmark(label.c_str(), base, points_tensor, *data_mode.values_tensor);
                    }
                }
            }
        }
    }

    struct GatherDataMode {
        const char* label;
        const TensorDesc* base_desc;
        bool allow_float_points;
    };

    GatherDataMode gather_modes[] = {
        {nullptr, &base_desc, true},
        {"i32", &base_i32_desc, false},
        {"u8", &base_u8_desc, false}
    };

    for (const auto& thread_mode : thread_modes) {
        set_threads_env(thread_mode.use_threads, thread_mode.threads);
        for (const auto& data_mode : gather_modes) {
            for (bool use_int : index_modes) {
                if (!data_mode.allow_float_points && !use_int) continue;
                for (bool use_affine : affine_modes) {
                    for (bool use_saturate : saturate_modes) {
                        AbstractTensor gather_base = AbstractTensor::create(*data_mode.base_desc, &backend);
                        assert(gather_base.valid());
                        fill_random_dense(gather_base, rng);

                        if (use_affine) {
                            float affine[16] = {
                                1.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 1.0f, 0.0f,
                                8.0f, 5.0f, 0.0f, 1.0f
                            };
                            gather_base.set_slice_affine_row_major(affine);
                        }
                        if (use_saturate) {
                            gather_base.set_slice_saturate_threshold(1.0f);
                        }

                        const AbstractTensor& points_tensor = use_int ? points_i : points;
                        RunFeatures features_nonadd{
                            "gather indices",
                            nullptr,
                            data_mode.label,
                            use_int ? "intiger indices" : "float indices",
                            use_affine,
                            use_saturate,
                            false,
                            thread_mode.use_threads,
                            thread_mode.threads
                        };
                        const std::string label = make_label(features_nonadd);
                        benchmark_gather(label.c_str(), gather_base, points_tensor);

                        RunFeatures features_add{
                            "gather indices",
                            nullptr,
                            data_mode.label,
                            use_int ? "intiger indices" : "float indices",
                            use_affine,
                            use_saturate,
                            true,
                            thread_mode.use_threads,
                            thread_mode.threads
                        };
                        const std::string label_add = make_label(features_add);
                        benchmark_gather_add(label_add.c_str(), gather_base, points_tensor);
                    }
                }
            }
        }
    }

    {
        const uint32_t smoke_h = 8;
        const uint32_t smoke_w = 8;
        const uint32_t smoke_c = 2;
        const uint32_t smoke_n = 32;

        TensorDesc smoke_f32_desc = make_dense_desc(TensorDType::F32, {smoke_h, smoke_w, smoke_c});
        TensorDesc smoke_f64_desc = make_dense_desc(TensorDType::F64, {smoke_h, smoke_w, smoke_c});
        TensorDesc smoke_i32_desc = make_dense_desc(TensorDType::I32, {smoke_h, smoke_w, smoke_c});
        TensorDesc smoke_u8_desc = make_dense_desc(TensorDType::U8, {smoke_h, smoke_w, smoke_c});
        TensorDesc smoke_points_f32_desc = make_dense_desc(TensorDType::F32, {smoke_n, 2});
        TensorDesc smoke_points_i32_desc = make_dense_desc(TensorDType::I32, {smoke_n, 2});
        TensorDesc smoke_values_f32_desc = make_dense_desc(TensorDType::F32, {smoke_n, smoke_c});
        TensorDesc smoke_values_f64_desc = make_dense_desc(TensorDType::F64, {smoke_n, smoke_c});
        TensorDesc smoke_values_i32_desc = make_dense_desc(TensorDType::I32, {smoke_n, smoke_c});
        TensorDesc smoke_values_u8_desc = make_dense_desc(TensorDType::U8, {smoke_n, smoke_c});

        AbstractTensor smoke_base_f32 = AbstractTensor::create(smoke_f32_desc, &backend);
        AbstractTensor smoke_base_f64 = AbstractTensor::create(smoke_f64_desc, &backend);
        AbstractTensor smoke_base_i32 = AbstractTensor::create(smoke_i32_desc, &backend);
        AbstractTensor smoke_base_u8 = AbstractTensor::create(smoke_u8_desc, &backend);
        AbstractTensor smoke_points_f32 = AbstractTensor::create(smoke_points_f32_desc, &backend);
        AbstractTensor smoke_points_i32 = AbstractTensor::create(smoke_points_i32_desc, &backend);
        AbstractTensor smoke_values_f32 = AbstractTensor::create(smoke_values_f32_desc, &backend);
        AbstractTensor smoke_values_f64 = AbstractTensor::create(smoke_values_f64_desc, &backend);
        AbstractTensor smoke_values_i32 = AbstractTensor::create(smoke_values_i32_desc, &backend);
        AbstractTensor smoke_values_u8 = AbstractTensor::create(smoke_values_u8_desc, &backend);
        assert(smoke_base_f32.valid() && smoke_base_f64.valid());
        assert(smoke_base_i32.valid() && smoke_base_u8.valid());
        assert(smoke_points_f32.valid() && smoke_points_i32.valid());
        assert(smoke_values_f32.valid() && smoke_values_f64.valid());
        assert(smoke_values_i32.valid() && smoke_values_u8.valid());

        fill_random_dense(smoke_base_f32, rng);
        fill_random_dense(smoke_base_i32, rng);
        fill_random_dense(smoke_base_u8, rng);
        {
            auto* mem = dynamic_cast<InMemoryBackend*>(smoke_base_f64.backend());
            assert(mem);
            void* ptr = nullptr;
            size_t bytes = 0;
            assert(mem->map(smoke_base_f64.handle(), &ptr, &bytes));
            std::memset(ptr, 0, bytes);
            mem->unmap(smoke_base_f64.handle());
        }
        fill_random_points(smoke_points_f32, smoke_n, smoke_w, smoke_h, rng);
        fill_random_points(smoke_points_i32, smoke_n, smoke_w, smoke_h, rng);
        fill_random_values(smoke_values_f32, smoke_n, rng);
        fill_random_values(smoke_values_i32, smoke_n, rng);
        fill_random_values(smoke_values_u8, smoke_n, rng);

        {
            auto* mem = dynamic_cast<InMemoryBackend*>(smoke_values_f64.backend());
            assert(mem);
            void* ptr = nullptr;
            size_t bytes = 0;
            assert(mem->map(smoke_values_f64.handle(), &ptr, &bytes));
            auto* out = static_cast<double*>(ptr);
            for (uint32_t i = 0; i < smoke_n * smoke_c; ++i) {
                out[i] = static_cast<double>(i % 7) * 0.125;
            }
            mem->unmap(smoke_values_f64.handle());
        }

        TensorStencil stencil_f32 = build_unit_stencil(TensorDType::F32, &backend);
        TensorStencil stencil_f64 = build_unit_stencil(TensorDType::F64, &backend);
        TensorStencil stencil_i32 = build_unit_stencil(TensorDType::I32, &backend);
        TensorStencil stencil_u8 = build_unit_stencil(TensorDType::U8, &backend);
        TensorFootprint2D footprint_f32 = build_unit_footprint(TensorDType::F32, &backend);
        TensorFootprint2D footprint_f64 = build_unit_footprint(TensorDType::F64, &backend);

        AbstractTensor out;
        TensorTransferConfig cfg_overwrite{};
        cfg_overwrite.clamp = true;
        TensorTransferConfig cfg_scatter_add{};
        cfg_scatter_add.premix_scatter = TensorMixPolicy::Add;
        cfg_scatter_add.postmix_scatter = TensorMixPolicy::Add;
        cfg_scatter_add.clamp = true;
        TensorTransferConfig cfg_gather_add{};
        cfg_gather_add.postmix_gather = TensorMixPolicy::Add;
        cfg_gather_add.clamp = true;

        auto scatter_with_base = [&](const AbstractTensor& base_tensor,
                                     const AbstractTensor& points_tensor,
                                     const AbstractTensor& values_tensor,
                                     const TensorTransferConfig& cfg) -> bool {
            AbstractTensor base_copy = AbstractTensor::create(base_tensor.desc(), base_tensor.backend());
            if (!base_copy.valid()){
                printf("Failed to create base copy tensor\n");
                return false;
            }
            TensorTransferConfig copy_cfg{};
            if (!base_tensor.transfer(AbstractTensor{}, base_copy, AbstractTensor{}, true, copy_cfg)){
                printf("Failed to copy base tensor to out tensor\n");
                return false;
            }
            const bool ok = values_tensor.scatter(AbstractTensor{}, base_copy, points_tensor, cfg);
            if(!ok){
                printf("Scatter operation failed\n");
                return false;
            }
            out = std::move(base_copy);
            return ok;
        };

        report_stub("tensor_scatter_2d", scatter_with_base(smoke_base_f32, smoke_points_f32, smoke_values_f32, cfg_overwrite));
        report_stub("tensor_scatter_2d", scatter_with_base(smoke_base_i32, smoke_points_i32, smoke_values_i32, cfg_overwrite));
        report_stub("tensor_scatter_2d", scatter_with_base(smoke_base_u8, smoke_points_i32, smoke_values_u8, cfg_overwrite));

        report_stub("tensor_scatter_add_2d", scatter_with_base(smoke_base_f32, smoke_points_f32, smoke_values_f32, cfg_scatter_add));
        report_stub("tensor_scatter_add_2d", scatter_with_base(smoke_base_f64, smoke_points_i32, smoke_values_f64, cfg_scatter_add));
        report_stub("tensor_scatter_add_2d", scatter_with_base(smoke_base_i32, smoke_points_i32, smoke_values_i32, cfg_scatter_add));
        report_stub("tensor_scatter_add_2d", scatter_with_base(smoke_base_u8, smoke_points_i32, smoke_values_u8, cfg_scatter_add));

        report_stub("tensor_gather_2d", smoke_base_f32.gather(smoke_points_f32, out, AbstractTensor{}, cfg_overwrite));
        report_stub("tensor_gather_2d", smoke_base_f64.gather(smoke_points_i32, out, AbstractTensor{}, cfg_overwrite));
        report_stub("tensor_gather_2d", smoke_base_i32.gather(smoke_points_i32, out, AbstractTensor{}, cfg_overwrite));
        report_stub("tensor_gather_2d", smoke_base_u8.gather(smoke_points_i32, out, AbstractTensor{}, cfg_overwrite));

        TensorDesc gather_out_desc{};
        gather_out_desc.layout = TensorLayout::Dense;
        gather_out_desc.shape.dims = {smoke_n, smoke_c};

        gather_out_desc.dtype = TensorDType::F32;
        AbstractTensor gather_add_f32_out = AbstractTensor::create(gather_out_desc, &backend);
        report_stub("tensor_gather_add_2d", smoke_base_f32.gather(smoke_points_f32, gather_add_f32_out, AbstractTensor{}, cfg_gather_add));

        gather_out_desc.dtype = TensorDType::F64;
        AbstractTensor gather_add_f64_out = AbstractTensor::create(gather_out_desc, &backend);
        report_stub("tensor_gather_add_2d", smoke_base_f64.gather(smoke_points_i32, gather_add_f64_out, AbstractTensor{}, cfg_gather_add));

        gather_out_desc.dtype = TensorDType::I32;
        AbstractTensor gather_add_i32_out = AbstractTensor::create(gather_out_desc, &backend);
        report_stub("tensor_gather_add_2d", smoke_base_i32.gather(smoke_points_i32, gather_add_i32_out, AbstractTensor{}, cfg_gather_add));

        gather_out_desc.dtype = TensorDType::U8;
        AbstractTensor gather_add_u8_out = AbstractTensor::create(gather_out_desc, &backend);
        report_stub("tensor_gather_add_2d", smoke_base_u8.gather(smoke_points_i32, gather_add_u8_out, AbstractTensor{}, cfg_gather_add));

        report_stub("tensor_scatter_nd", scatter_with_base(smoke_base_f32, smoke_points_f32, smoke_values_f32, cfg_overwrite));
        report_stub("tensor_scatter_nd", scatter_with_base(smoke_base_i32, smoke_points_i32, smoke_values_i32, cfg_overwrite));
        report_stub("tensor_scatter_nd", scatter_with_base(smoke_base_u8, smoke_points_i32, smoke_values_u8, cfg_overwrite));

        report_stub("tensor_scatter_add_nd", scatter_with_base(smoke_base_f32, smoke_points_f32, smoke_values_f32, cfg_scatter_add));
        report_stub("tensor_scatter_add_nd", scatter_with_base(smoke_base_f64, smoke_points_i32, smoke_values_f64, cfg_scatter_add));
        report_stub("tensor_scatter_add_nd", scatter_with_base(smoke_base_i32, smoke_points_i32, smoke_values_i32, cfg_scatter_add));
        report_stub("tensor_scatter_add_nd", scatter_with_base(smoke_base_u8, smoke_points_i32, smoke_values_u8, cfg_scatter_add));

        report_stub("tensor_gather_nd", smoke_base_f32.gather(smoke_points_f32, out, AbstractTensor{}, cfg_overwrite));
        report_stub("tensor_gather_nd", smoke_base_f64.gather(smoke_points_i32, out, AbstractTensor{}, cfg_overwrite));
        report_stub("tensor_gather_nd", smoke_base_i32.gather(smoke_points_i32, out, AbstractTensor{}, cfg_overwrite));
        report_stub("tensor_gather_nd", smoke_base_u8.gather(smoke_points_i32, out, AbstractTensor{}, cfg_overwrite));

        TensorDesc gather_nd_out_desc{};
        gather_nd_out_desc.layout = TensorLayout::Dense;
        gather_nd_out_desc.shape.dims = {smoke_n, smoke_c};

        gather_nd_out_desc.dtype = TensorDType::F32;
        AbstractTensor gather_add_nd_f32_out = AbstractTensor::create(gather_nd_out_desc, &backend);
        report_stub("tensor_gather_add_nd", smoke_base_f32.gather(smoke_points_f32, gather_add_nd_f32_out, AbstractTensor{}, cfg_gather_add));

        gather_nd_out_desc.dtype = TensorDType::F64;
        AbstractTensor gather_add_nd_f64_out = AbstractTensor::create(gather_nd_out_desc, &backend);
        report_stub("tensor_gather_add_nd", smoke_base_f64.gather(smoke_points_i32, gather_add_nd_f64_out, AbstractTensor{}, cfg_gather_add));

        gather_nd_out_desc.dtype = TensorDType::I32;
        AbstractTensor gather_add_nd_i32_out = AbstractTensor::create(gather_nd_out_desc, &backend);
        report_stub("tensor_gather_add_nd", smoke_base_i32.gather(smoke_points_i32, gather_add_nd_i32_out, AbstractTensor{}, cfg_gather_add));

        gather_nd_out_desc.dtype = TensorDType::U8;
        AbstractTensor gather_add_nd_u8_out = AbstractTensor::create(gather_nd_out_desc, &backend);
        report_stub("tensor_gather_add_nd", smoke_base_u8.gather(smoke_points_i32, gather_add_nd_u8_out, AbstractTensor{}, cfg_gather_add));

        report_stub("tensor_gather_stencil_2d_f32", tensor_gather_stencil_2d_f32(
            smoke_base_f32, stencil_f32, &out, StencilBoundaryMode::Clamp, nullptr, nullptr, false));
        report_stub("tensor_gather_stencil_2d_f64", tensor_gather_stencil_2d_f64(
            smoke_base_f64, stencil_f64, &out, StencilBoundaryMode::Clamp, nullptr, nullptr, false));
        report_stub("tensor_gather_stencil_2d_i32", tensor_gather_stencil_2d_i32(
            smoke_base_i32, stencil_i32, &out, StencilBoundaryMode::Clamp, nullptr, nullptr, false));
        report_stub("tensor_gather_stencil_2d_u8", tensor_gather_stencil_2d_u8(
            smoke_base_u8, stencil_u8, &out, StencilBoundaryMode::Clamp, nullptr, nullptr, false));

        report_stub("tensor_gather_stencil_nd_f32", tensor_gather_stencil_nd_f32(
            smoke_base_f32, stencil_f32, &out, StencilBoundaryMode::Clamp, nullptr, nullptr, false));
        report_stub("tensor_gather_stencil_nd_f64", tensor_gather_stencil_nd_f64(
            smoke_base_f64, stencil_f64, &out, StencilBoundaryMode::Clamp, nullptr, nullptr, false));
        report_stub("tensor_gather_stencil_nd_i32", tensor_gather_stencil_nd_i32(
            smoke_base_i32, stencil_i32, &out, StencilBoundaryMode::Clamp, nullptr, nullptr, false));
        report_stub("tensor_gather_stencil_nd_u8", tensor_gather_stencil_nd_u8(
            smoke_base_u8, stencil_u8, &out, StencilBoundaryMode::Clamp, nullptr, nullptr, false));

        report_stub("tensor_gather_footprint_2d_f32", tensor_gather_footprint_2d_f32(
            smoke_base_f32, footprint_f32, &out, StencilOrientation::Correlation,
            StencilBoundaryMode::Clamp, nullptr, nullptr, false));
        report_stub("tensor_gather_footprint_2d_f64", tensor_gather_footprint_2d_f64(
            smoke_base_f64, footprint_f64, &out, StencilOrientation::Correlation,
            StencilBoundaryMode::Clamp, nullptr, nullptr, false));

        report_stub("tensor_scatter_stencil_2d_f32", tensor_scatter_stencil_2d_f32(
            smoke_base_f32, smoke_points_f32, smoke_values_f32, stencil_f32, &out,
            StencilBoundaryMode::Clamp, true));
        report_stub("tensor_scatter_stencil_2d_f64", tensor_scatter_stencil_2d_f64(
            smoke_base_f64, smoke_points_i32, smoke_values_f64, stencil_f64, &out,
            StencilBoundaryMode::Clamp, true));
        report_stub("tensor_scatter_stencil_2d_i32", tensor_scatter_stencil_2d_i32(
            smoke_base_i32, smoke_points_i32, smoke_values_i32, stencil_i32, &out,
            StencilBoundaryMode::Clamp, true));
        report_stub("tensor_scatter_stencil_2d_u8", tensor_scatter_stencil_2d_u8(
            smoke_base_u8, smoke_points_i32, smoke_values_u8, stencil_u8, &out,
            StencilBoundaryMode::Clamp, true));

        report_stub("tensor_scatter_stencil_nd_f32", tensor_scatter_stencil_nd_f32(
            smoke_base_f32, smoke_points_f32, smoke_values_f32, stencil_f32, &out,
            StencilBoundaryMode::Clamp, true));
        report_stub("tensor_scatter_stencil_nd_f64", tensor_scatter_stencil_nd_f64(
            smoke_base_f64, smoke_points_i32, smoke_values_f64, stencil_f64, &out,
            StencilBoundaryMode::Clamp, true));
        report_stub("tensor_scatter_stencil_nd_i32", tensor_scatter_stencil_nd_i32(
            smoke_base_i32, smoke_points_i32, smoke_values_i32, stencil_i32, &out,
            StencilBoundaryMode::Clamp, true));
        report_stub("tensor_scatter_stencil_nd_u8", tensor_scatter_stencil_nd_u8(
            smoke_base_u8, smoke_points_i32, smoke_values_u8, stencil_u8, &out,
            StencilBoundaryMode::Clamp, true));

        report_stub("tensor_scatter_footprint_2d_f32", tensor_scatter_footprint_2d_f32(
            smoke_base_f32, smoke_points_f32, smoke_values_f32, footprint_f32, &out,
            StencilOrientation::Correlation, StencilBoundaryMode::Clamp, true));
        report_stub("tensor_scatter_footprint_2d_f64", tensor_scatter_footprint_2d_f64(
            smoke_base_f64, smoke_points_i32, smoke_values_f64, footprint_f64, &out,
            StencilOrientation::Correlation, StencilBoundaryMode::Clamp, true));
    }

    TensorFootprint2D footprint = build_gaussian_footprint(9, 2.0f, &backend);
    assert(footprint.weights.valid());
    TensorStencil stencil;
    bool ok_build = tensor_build_stencil_from_footprint_2d_f32(
        footprint, StencilOrientation::Correlation, &stencil);
    assert(ok_build);

    const uint32_t reduced_height = 256;
    const uint32_t reduced_width = 256;
    TensorDesc reduced_base_desc = make_dense_desc(TensorDType::F32, {reduced_height, reduced_width, channels});

    for (const auto& thread_mode : thread_modes) {
        set_threads_env(thread_mode.use_threads, thread_mode.threads);
        for (bool use_affine : affine_modes) {
            for (bool use_saturate : saturate_modes) {
                AbstractTensor reduced_base = AbstractTensor::create(reduced_base_desc, &backend);
                assert(reduced_base.valid());
                fill_random_dense(reduced_base, rng);

                if (use_affine) {
                    float affine[16] = {
                        1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f, 0.0f,
                        8.0f, 5.0f, 0.0f, 1.0f
                    };
                    reduced_base.set_slice_affine_row_major(affine);
                }
                if (use_saturate) {
                    reduced_base.set_slice_saturate_threshold(1.0f);
                }

                RunFeatures features_stencil{
                    "gather stencil 2d",
                    "gaussian9",
                    nullptr,
                    nullptr,
                    use_affine,
                    use_saturate,
                    false,
                    thread_mode.use_threads,
                    thread_mode.threads
                };
                const std::string label_stencil = make_label(features_stencil);
                benchmark_gather_stencil(label_stencil.c_str(), reduced_base, stencil);

                RunFeatures features_stencil_nd{
                    "gather stencil nd",
                    "gaussian9",
                    nullptr,
                    nullptr,
                    use_affine,
                    use_saturate,
                    false,
                    thread_mode.use_threads,
                    thread_mode.threads
                };
                const std::string label_stencil_nd = make_label(features_stencil_nd);
                benchmark_gather_stencil_nd(label_stencil_nd.c_str(), reduced_base, stencil);

                RunFeatures features_gather_fp{
                    "gather footprint 2d",
                    "gaussian9",
                    nullptr,
                    nullptr,
                    use_affine,
                    use_saturate,
                    false,
                    thread_mode.use_threads,
                    thread_mode.threads
                };
                const std::string label_gather_fp = make_label(features_gather_fp);
                benchmark_gather_footprint(label_gather_fp.c_str(), reduced_base, footprint);

                RunFeatures features_scatter_stencil{
                    "scatter stencil 2d",
                    "gaussian9",
                    nullptr,
                    nullptr,
                    use_affine,
                    use_saturate,
                    true,
                    thread_mode.use_threads,
                    thread_mode.threads
                };
                const std::string label_scatter_stencil = make_label(features_scatter_stencil);
                benchmark_scatter_stencil_2d(label_scatter_stencil.c_str(), reduced_base, points, values, stencil);

                RunFeatures features_scatter_fp{
                    "scatter footprint 2d",
                    "gaussian9",
                    nullptr,
                    nullptr,
                    use_affine,
                    use_saturate,
                    true,
                    thread_mode.use_threads,
                    thread_mode.threads
                };
                const std::string label_scatter_fp = make_label(features_scatter_fp);
                benchmark_scatter_footprint_2d(label_scatter_fp.c_str(), reduced_base, points, values, footprint);
            }
        }
    }

    const uint32_t csr_height = 512;
    const uint32_t csr_width = 512;
    const uint32_t csr_degree = 8;
    TensorDesc csr_desc = make_dense_desc(TensorDType::F32, {csr_height, csr_width, channels});
    AbstractTensor csr_base = AbstractTensor::create(csr_desc, &backend);
    assert(csr_base.valid());
    fill_random_dense(csr_base, rng);

    const uint32_t csr_rows = csr_height * csr_width;
    const uint32_t csr_cols = csr_rows;
    std::uniform_int_distribution<uint32_t> dist_target(0, csr_cols - 1);
    std::uniform_real_distribution<float> dist_w(0.0f, 1.0f);
    std::vector<uint32_t> edge_rows;
    std::vector<uint32_t> edge_cols;
    std::vector<float> edge_w;
    edge_rows.reserve(static_cast<size_t>(csr_rows) * csr_degree);
    edge_cols.reserve(static_cast<size_t>(csr_rows) * csr_degree);
    edge_w.reserve(static_cast<size_t>(csr_rows) * csr_degree);
    for (uint32_t row = 0; row < csr_rows; ++row) {
        for (uint32_t k = 0; k < csr_degree; ++k) {
            edge_rows.push_back(row);
            edge_cols.push_back(dist_target(rng));
            edge_w.push_back(dist_w(rng));
        }
    }

    TensorCSR targets_per_center = build_csr_from_edges(
        csr_rows, csr_cols, edge_rows, edge_cols, edge_w, &backend);
    assert(targets_per_center.row_ptr.valid());
    TensorCSR centers_per_target = build_csr_from_edges(
        csr_cols, csr_rows, edge_cols, edge_rows, edge_w, &backend);
    assert(centers_per_target.row_ptr.valid());

    TensorFootprint2D gaussian_fp = build_gaussian_footprint(9, 2.0f, &backend);
    TensorStencil gaussian_stencil;
    bool ok_gauss = tensor_build_stencil_from_footprint_2d_f32(
        gaussian_fp, StencilOrientation::Correlation, &gaussian_stencil);
    assert(ok_gauss);
    TensorProbeKernel gaussian_probe = build_probe_kernel_from_stencil(gaussian_stencil);

    TensorFootprint2D airy_fp = build_airy_footprint(11, 3.0f, &backend);
    TensorStencil airy_stencil;
    bool ok_airy = tensor_build_stencil_from_footprint_2d_f32(
        airy_fp, StencilOrientation::Correlation, &airy_stencil);
    assert(ok_airy);
    TensorProbeKernel airy_probe = build_probe_kernel_from_stencil(airy_stencil);

    RunFeatures label_scatter_gauss{
        "scatter probe csr",
        "gaussian9",
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        1
    };
    const std::string label_scatter_gauss_str = make_label(label_scatter_gauss);
    benchmark_scatter_probe_csr(label_scatter_gauss_str.c_str(), csr_base, gaussian_probe, targets_per_center);

    RunFeatures label_gather_gauss{
        "gather probe csr",
        "gaussian9",
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        1
    };
    const std::string label_gather_gauss_str = make_label(label_gather_gauss);
    benchmark_gather_probe_csr(label_gather_gauss_str.c_str(), csr_base, gaussian_probe, centers_per_target);

    RunFeatures label_scatter_gauss_stencil{
        "scatter probe csr (stencil)",
        "gaussian9",
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        1
    };
    const std::string label_scatter_gauss_stencil_str = make_label(label_scatter_gauss_stencil);
    benchmark_scatter_probe_csr_stencil(label_scatter_gauss_stencil_str.c_str(), csr_base, gaussian_stencil, targets_per_center);

    RunFeatures label_gather_gauss_stencil{
        "gather probe csr (stencil)",
        "gaussian9",
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        1
    };
    const std::string label_gather_gauss_stencil_str = make_label(label_gather_gauss_stencil);
    benchmark_gather_probe_csr_stencil(label_gather_gauss_stencil_str.c_str(), csr_base, gaussian_stencil, centers_per_target);

    RunFeatures label_scatter_gauss_fp{
        "scatter probe csr (footprint)",
        "gaussian9",
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        1
    };
    const std::string label_scatter_gauss_fp_str = make_label(label_scatter_gauss_fp);
    benchmark_scatter_probe_csr_footprint(label_scatter_gauss_fp_str.c_str(), csr_base, gaussian_fp, targets_per_center);

    RunFeatures label_gather_gauss_fp{
        "gather probe csr (footprint)",
        "gaussian9",
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        1
    };
    const std::string label_gather_gauss_fp_str = make_label(label_gather_gauss_fp);
    benchmark_gather_probe_csr_footprint(label_gather_gauss_fp_str.c_str(), csr_base, gaussian_fp, centers_per_target);

    RunFeatures label_scatter_airy{
        "scatter probe csr",
        "airy11",
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        1
    };
    const std::string label_scatter_airy_str = make_label(label_scatter_airy);
    benchmark_scatter_probe_csr(label_scatter_airy_str.c_str(), csr_base, airy_probe, targets_per_center);

    RunFeatures label_gather_airy{
        "gather probe csr",
        "airy11",
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        1
    };
    const std::string label_gather_airy_str = make_label(label_gather_airy);
    benchmark_gather_probe_csr(label_gather_airy_str.c_str(), csr_base, airy_probe, centers_per_target);

    RunFeatures label_scatter_airy_stencil{
        "scatter probe csr (stencil)",
        "airy11",
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        1
    };
    const std::string label_scatter_airy_stencil_str = make_label(label_scatter_airy_stencil);
    benchmark_scatter_probe_csr_stencil(label_scatter_airy_stencil_str.c_str(), csr_base, airy_stencil, targets_per_center);

    RunFeatures label_gather_airy_stencil{
        "gather probe csr (stencil)",
        "airy11",
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        1
    };
    const std::string label_gather_airy_stencil_str = make_label(label_gather_airy_stencil);
    benchmark_gather_probe_csr_stencil(label_gather_airy_stencil_str.c_str(), csr_base, airy_stencil, centers_per_target);

    RunFeatures label_scatter_airy_fp{
        "scatter probe csr (footprint)",
        "airy11",
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        1
    };
    const std::string label_scatter_airy_fp_str = make_label(label_scatter_airy_fp);
    benchmark_scatter_probe_csr_footprint(label_scatter_airy_fp_str.c_str(), csr_base, airy_fp, targets_per_center);

    RunFeatures label_gather_airy_fp{
        "gather probe csr (footprint)",
        "airy11",
        nullptr,
        nullptr,
        false,
        false,
        false,
        false,
        1
    };
    const std::string label_gather_airy_fp_str = make_label(label_gather_airy_fp);
    benchmark_gather_probe_csr_footprint(label_gather_airy_fp_str.c_str(), csr_base, airy_fp, centers_per_target);

    std::cout << "scatter_indices_test: ok\n";
    return 0;
}
