#pragma once

#include "common/tensors/abstraction/tensor_types.h"
#include "common/tensors/abstraction/abstract_tensor_pool.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_math.h"
#include "common/tensors/abstraction/rounding.h"
#include "common/tensors/abstraction/affine_xy.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <mutex>

namespace nodus::tensors {

class TensorBackend;

struct SpanRecordStencil {
    uint64_t linear = 0;
    uint32_t point = 0;
    uint32_t tap = 0;
};

enum class SpanAggregateOp : uint8_t {
    Replace = 0,
    Add = 1,
    Mul = 2,
};

struct SpanOpConfig {
    SpanAggregateOp aggregate = SpanAggregateOp::Add;
    bool use_stencil = false;
    bool use_footprint = false;
    bool clamp = false;
    bool normalize = false;

    inline bool is_replace() const { return aggregate == SpanAggregateOp::Replace; }
    inline bool is_add() const { return aggregate == SpanAggregateOp::Add; }
    inline bool is_mul() const { return aggregate == SpanAggregateOp::Mul; }
};

struct TileBin2D {
    std::vector<uint32_t> counts;
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> write_pos;
    std::vector<uint32_t> indices;
    std::vector<uint64_t> tile_mask;
};

struct TileBinning2D {
    uint32_t tile_px = 0;
    uint32_t tiles_x = 0;
    uint32_t tiles_y = 0;
    uint32_t tile_count = 0;
    float dense_threshold = 0.0f;
    bool dense_grid = false;
};

inline TileBin2D& tile_bins_2d() {
    static thread_local TileBin2D bins;
    return bins;
}

inline uint32_t default_tile_px_2d(const TensorDesc& desc) {
    static uint32_t kTilePx = 0;
    if (kTilePx == 0) {
        constexpr uint64_t kDefaultTileCacheBytes = 6ull * 1024ull * 1024ull * 1024ull;
        TileShape2D shape = choose_tile_shape_2d(desc, 0u, 0u, kDefaultTileCacheBytes,
                                                TileContiguityStrategy::Auto);
        const uint32_t t = std::max<uint32_t>(1u, std::min<uint32_t>(shape.x, shape.y));
        kTilePx = t;
    }
    return kTilePx;
}

inline float scatter_tile_dense_threshold() {
    static const float cached = []() -> float {
        float dense_threshold = 0.85f;
        if (const char* v = std::getenv("NODUS_SCATTER_TILE_DENSE")) {
            if (*v) {
                char* end = nullptr;
                double parsed = std::strtod(v, &end);
                if (end != v) {
                    dense_threshold = static_cast<float>(std::clamp(parsed, 0.0, 1.0));
                }
            }
        }
        return dense_threshold;
    }();
    return cached;
}

inline bool detect_dense_grid_2d(uint32_t pcols,
                                 uint32_t count,
                                 uint32_t width,
                                 uint32_t height,
                                 const int64_t* coords) {
    if (pcols != 2 || !coords) return false;
    if (count != static_cast<uint32_t>(static_cast<uint64_t>(width) * height)) return false;
    auto check_point = [&](size_t idx, uint32_t ex, uint32_t ey) -> bool {
        if (idx >= count) return false;
        const int64_t xi = coords[idx * 2 + 0];
        const int64_t yi = coords[idx * 2 + 1];
        return (xi == static_cast<int64_t>(ex) && yi == static_cast<int64_t>(ey));
    };
    const size_t last = static_cast<size_t>(height - 1) * width + (width - 1);
    return check_point(0, 0u, 0u) &&
           (width < 2 || check_point(1, 1u, 0u)) &&
           (height < 2 || check_point(static_cast<size_t>(width), 0u, 1u)) &&
           check_point(last, width - 1, height - 1);
}

inline TileBinning2D build_tile_bins_2d(const TensorDesc& desc,
                                        const int64_t* coords,
                                        const uint8_t* in_bounds,
                                        uint32_t count,
                                        uint32_t width,
                                        uint32_t height,
                                        uint32_t pcols,
                                        TileBin2D& bins) {
    TileBinning2D out{};
    out.tile_px = default_tile_px_2d(desc);
    out.tiles_x = (width + out.tile_px - 1) / out.tile_px;
    out.tiles_y = (height + out.tile_px - 1) / out.tile_px;
    out.tile_count = out.tiles_x * out.tiles_y;

    bins.counts.assign(out.tile_count, 0u);
    bins.offsets.assign(out.tile_count + 1, 0u);
    bins.write_pos.assign(out.tile_count, 0u);
    bins.indices.resize(count);
    bins.tile_mask.assign((out.tile_count + 63u) / 64u, 0ull);

    for (uint32_t i = 0; i < count; ++i) {
        if (!in_bounds || !in_bounds[i]) {
            continue;
        }
        const int64_t xi = coords[static_cast<size_t>(i) * 2 + 0];
        const int64_t yi = coords[static_cast<size_t>(i) * 2 + 1];
        const uint32_t tx = static_cast<uint32_t>(xi) / out.tile_px;
        const uint32_t ty = static_cast<uint32_t>(yi) / out.tile_px;
        const uint32_t tid = ty * out.tiles_x + tx;
        bins.counts[tid]++;
        const uint32_t word = tid >> 6;
        const uint32_t bit = tid & 63u;
        bins.tile_mask[word] |= (1ull << bit);
    }

    uint32_t running = 0;
    for (uint32_t t = 0; t < out.tile_count; ++t) {
        bins.offsets[t] = running;
        running += bins.counts[t];
    }
    bins.offsets[out.tile_count] = running;
    bins.write_pos = bins.offsets;

    out.dense_threshold = scatter_tile_dense_threshold();
    out.dense_grid = detect_dense_grid_2d(pcols, count, width, height, coords);

    for (uint32_t i = 0; i < count; ++i) {
        if (!in_bounds || !in_bounds[i]) {
            continue;
        }
        const int64_t xi = coords[static_cast<size_t>(i) * 2 + 0];
        const int64_t yi = coords[static_cast<size_t>(i) * 2 + 1];
        const uint32_t tx = static_cast<uint32_t>(xi) / out.tile_px;
        const uint32_t ty = static_cast<uint32_t>(yi) / out.tile_px;
        const uint32_t tid = ty * out.tiles_x + tx;
        const uint32_t pos = bins.write_pos[tid]++;
        if (pos < bins.indices.size()) {
            bins.indices[pos] = i;
        }
    }

    return out;
}

template <typename Scalar>
struct SpanStencilRowSlice2D {
    int32_t dy = 0;
    uint32_t begin = 0;
    uint32_t end = 0;
    int32_t min_dx = 0;
    int32_t max_dx = 0;

    inline uint32_t count() const { return (end > begin) ? (end - begin) : 0u; }
};

template <typename Scalar>
struct SpanStencilRowView2D {
    int32_t dy = 0;
    const int32_t* dx = nullptr;
    const Scalar* weights = nullptr;
    uint32_t count = 0;
    int32_t min_dx = 0;
    int32_t max_dx = 0;

    inline bool empty() const { return count == 0; }
};

template <typename Scalar>
struct SpanStencilPlan2D {
    std::vector<int32_t> dx;
    std::vector<Scalar> weights;
    std::vector<SpanStencilRowSlice2D<Scalar>> rows;
    TensorSupport support;
    uint32_t taps = 0;

    inline void reset() {
        dx.clear();
        weights.clear();
        rows.clear();
        support.reset();
        taps = 0;
    }

    inline bool empty() const { return taps == 0 || dx.empty() || weights.empty(); }

    inline const SpanStencilRowSlice2D<Scalar>* row_slice(int32_t dy) const {
        if (rows.empty()) return nullptr;
        auto it = std::lower_bound(rows.begin(), rows.end(), dy,
                                   [](const SpanStencilRowSlice2D<Scalar>& r, int32_t v) {
                                       return r.dy < v;
                                   });
        if (it == rows.end() || it->dy != dy) return nullptr;
        return &(*it);
    }

    inline SpanStencilRowView2D<Scalar> row_view(int32_t dy) const {
        SpanStencilRowView2D<Scalar> view{};
        const auto* row = row_slice(dy);
        if (!row) return view;
        view.dy = row->dy;
        view.count = row->count();
        view.min_dx = row->min_dx;
        view.max_dx = row->max_dx;
        view.dx = dx.data() + row->begin;
        view.weights = weights.data() + row->begin;
        return view;
    }
};

template <typename Scalar>
inline void build_span_codex_from_plan_2d(const SpanStencilPlan2D<Scalar>& plan,
                                          TensorSpanOffsetCodex2D& out) {
    out.reset();
    if (plan.taps == 0 || plan.dx.empty()) return;

    out.dx = plan.dx;
    out.rows.clear();
    out.rows.reserve(plan.rows.size());
    for (const auto& row : plan.rows) {
        TensorSpanRowSlice2D slice{};
        slice.dy = row.dy;
        slice.begin = row.begin;
        slice.end = row.end;
        slice.min_dx = row.min_dx;
        slice.max_dx = row.max_dx;
        out.rows.push_back(slice);
    }

    out.support = plan.support;
    out.taps = plan.taps;

    if constexpr (std::is_same_v<Scalar, float>) {
        out.weight_dtype = TensorDType::F32;
        out.weights_f32 = plan.weights;
    } else if constexpr (std::is_same_v<Scalar, double>) {
        out.weight_dtype = TensorDType::F64;
        out.weights_f64 = plan.weights;
    } else {
        out.weight_dtype = TensorDType::Unknown;
    }
}

template <typename Scalar>
inline void build_span_codex_from_offsets_2d(const int32_t* offsets,
                                             const Scalar* weights,
                                             uint32_t taps,
                                             TensorSpanOffsetCodex2D& out) {
    SpanStencilPlan2D<Scalar> plan{};
    build_span_stencil_plan_2d_from_offsets(offsets, weights, taps, plan);
    build_span_codex_from_plan_2d(plan, out);
}

template <typename Scalar>
inline bool build_span_codex_from_stencil_2d(const TensorStencil& stencil,
                                             TensorSpanOffsetCodex2D& out) {
    SpanStencilPlan2D<Scalar> plan{};
    if (!build_span_stencil_plan_2d_from_stencil<Scalar>(stencil, plan)) return false;
    build_span_codex_from_plan_2d(plan, out);
    return true;
}

template <typename Scalar>
inline bool build_span_codex_from_footprint_2d(const TensorFootprint2D& footprint,
                                               StencilOrientation orientation,
                                               TensorSpanOffsetCodex2D& out) {
    SpanStencilPlan2D<Scalar> plan{};
    if (!build_span_stencil_plan_2d_from_footprint<Scalar>(footprint, orientation, plan)) return false;
    build_span_codex_from_plan_2d(plan, out);
    return true;
}

template <typename Scalar>
inline constexpr TensorDType span_scalar_dtype() {
    if constexpr (std::is_same_v<Scalar, float>) {
        return TensorDType::F32;
    } else if constexpr (std::is_same_v<Scalar, double>) {
        return TensorDType::F64;
    } else {
        return TensorDType::Unknown;
    }
}

template <typename Scalar>
inline void build_span_stencil_plan_2d_from_offsets(const int32_t* offsets,
                                                    const Scalar* weights,
                                                    uint32_t taps,
                                                    SpanStencilPlan2D<Scalar>& out) {
    out.reset();
    if (!offsets || !weights || taps == 0) return;

    std::vector<uint32_t> order(taps);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        const int32_t dy_a = offsets[a * 2 + 1];
        const int32_t dy_b = offsets[b * 2 + 1];
        if (dy_a != dy_b) return dy_a < dy_b;
        const int32_t dx_a = offsets[a * 2 + 0];
        const int32_t dx_b = offsets[b * 2 + 0];
        return dx_a < dx_b;
    });

    out.dx.resize(taps);
    out.weights.resize(taps);
    out.rows.clear();
    out.rows.reserve(taps);

    int32_t min_dx = std::numeric_limits<int32_t>::max();
    int32_t max_dx = std::numeric_limits<int32_t>::min();
    int32_t min_dy = std::numeric_limits<int32_t>::max();
    int32_t max_dy = std::numeric_limits<int32_t>::min();

    int32_t current_dy = std::numeric_limits<int32_t>::max();
    SpanStencilRowSlice2D<Scalar> row{};

    for (uint32_t i = 0; i < taps; ++i) {
        const uint32_t idx = order[i];
        const int32_t dx = offsets[idx * 2 + 0];
        const int32_t dy = offsets[idx * 2 + 1];

        out.dx[i] = dx;
        out.weights[i] = weights[idx];

        min_dx = std::min(min_dx, dx);
        max_dx = std::max(max_dx, dx);
        min_dy = std::min(min_dy, dy);
        max_dy = std::max(max_dy, dy);

        if (i == 0 || dy != current_dy) {
            if (i > 0) {
                row.end = i;
                out.rows.push_back(row);
            }
            current_dy = dy;
            row = SpanStencilRowSlice2D<Scalar>{};
            row.dy = dy;
            row.begin = i;
            row.min_dx = dx;
            row.max_dx = dx;
        } else {
            row.min_dx = std::min(row.min_dx, dx);
            row.max_dx = std::max(row.max_dx, dx);
        }
    }

    if (taps > 0) {
        row.end = taps;
        out.rows.push_back(row);
    }

    out.support.min_offset = {min_dx, min_dy};
    out.support.max_offset = {max_dx, max_dy};
    out.support.radius = {std::max(std::abs(min_dx), std::abs(max_dx)),
                          std::max(std::abs(min_dy), std::abs(max_dy))};
    out.taps = taps;
}

template <typename Scalar>
inline bool build_span_stencil_plan_2d_from_stencil(const TensorStencil& stencil,
                                                    SpanStencilPlan2D<Scalar>& out) {
    out.reset();
    if (!stencil.offsets.valid() || !stencil.weights.valid()) return false;
    const TensorDesc& od = stencil.offsets.desc();
    const TensorDesc& wd = stencil.weights.desc();
    if (od.dtype != TensorDType::I32) return false;
    if (wd.dtype != span_scalar_dtype<Scalar>()) return false;
    if (od.layout != TensorLayout::Dense || wd.layout != TensorLayout::Dense) return false;
    if (od.shape.dims.size() != 2 || od.shape.dims[1] != 2) return false;
    const uint32_t taps = od.shape.dims[0];
    if (wd.shape.dims.size() != 1 || wd.shape.dims[0] != taps) return false;

    auto* mem = dynamic_cast<InMemoryBackend*>(stencil.offsets.backend());
    if (!mem) return false;
    void* optr = nullptr;
    size_t obytes = 0;
    void* wptr = nullptr;
    size_t wbytes = 0;
    if (!mem->map(stencil.offsets.handle(), &optr, &obytes)) return false;
    if (!mem->map(stencil.weights.handle(), &wptr, &wbytes)) {
        mem->unmap(stencil.offsets.handle());
        return false;
    }

    build_span_stencil_plan_2d_from_offsets(static_cast<const int32_t*>(optr),
                                            static_cast<const Scalar*>(wptr),
                                            taps,
                                            out);

    mem->unmap(stencil.offsets.handle());
    mem->unmap(stencil.weights.handle());
    return true;
}

template <typename Scalar>
inline bool build_span_stencil_plan_2d_from_footprint(const TensorFootprint2D& footprint,
                                                      StencilOrientation orientation,
                                                      SpanStencilPlan2D<Scalar>& out) {
    out.reset();
    if (!footprint.weights.valid()) return false;
    const TensorDesc& wd = footprint.weights.desc();
    if (wd.dtype != span_scalar_dtype<Scalar>()) return false;
    if (wd.layout != TensorLayout::Dense) return false;
    if (wd.shape.dims.size() != 2) return false;
    const uint32_t ks = wd.shape.dims[0];
    if (ks == 0 || wd.shape.dims[1] != ks) return false;

    auto* mem = dynamic_cast<InMemoryBackend*>(footprint.weights.backend());
    if (!mem) return false;
    void* wptr = nullptr;
    size_t wbytes = 0;
    if (!mem->map(footprint.weights.handle(), &wptr, &wbytes)) return false;
    const auto* wmap = static_cast<const Scalar*>(wptr);

    const int32_t radius = static_cast<int32_t>(ks / 2);
    std::vector<int32_t> offsets;
    std::vector<Scalar> weights;
    offsets.reserve(static_cast<size_t>(ks) * ks * 2u);
    weights.reserve(static_cast<size_t>(ks) * ks);

    for (uint32_t y = 0; y < ks; ++y) {
        const int32_t dy0 = static_cast<int32_t>(y) - radius;
        const uint64_t row = static_cast<uint64_t>(y) * ks;
        for (uint32_t x = 0; x < ks; ++x) {
            const Scalar w = wmap[row + x];
            if (w == static_cast<Scalar>(0)) continue;
            const int32_t dx0 = static_cast<int32_t>(x) - radius;
            const int32_t dx = (orientation == StencilOrientation::Convolution) ? -dx0 : dx0;
            const int32_t dy = (orientation == StencilOrientation::Convolution) ? -dy0 : dy0;
            offsets.push_back(dx);
            offsets.push_back(dy);
            weights.push_back(w);
        }
    }

    mem->unmap(footprint.weights.handle());

    const uint32_t taps = static_cast<uint32_t>(weights.size());
    if (taps == 0) return true;
    build_span_stencil_plan_2d_from_offsets(offsets.data(), weights.data(), taps, out);
    return true;
}

inline bool& span_tiling_enabled_flag() {
    static bool enabled = true;
    return enabled;
}

inline void set_span_tiling_enabled(bool enabled) {
    span_tiling_enabled_flag() = enabled;
}

inline bool span_tiling_enabled() {
    return span_tiling_enabled_flag();
}

struct TileSpan {
    uint64_t start = 0;
    uint32_t len = 0;
    uint32_t record_begin = 0;
    uint32_t record_end = 0;
};

struct SpanRecord2D {
    uint64_t linear = 0;
    uint32_t point = 0;
};

inline bool span_tiling_enabled_from_env(const char* env_name) {
    if (!env_name) return span_tiling_enabled();

    static std::unordered_map<std::string, bool> cache;
    static std::mutex cache_mutex;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(env_name);
        if (it != cache.end()) return it->second;
    }

    bool enabled = span_tiling_enabled();
    if (const char* v = std::getenv(env_name)) {
        if (*v) {
            const char c = *v;
            if (c == '0' || c == 'f' || c == 'F' || c == 'n' || c == 'N') {
                enabled = false;
            } else {
                enabled = true;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.emplace(env_name, enabled);
    }
    return enabled;
}

inline uint32_t span_tiling_max_from_env(const char* env_name, uint32_t default_value) {
    if (!env_name) return default_value;

    static std::unordered_map<std::string, uint32_t> cache;
    static std::mutex cache_mutex;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(env_name);
        if (it != cache.end()) return it->second;
    }

    uint32_t value = default_value;
    if (const char* v = std::getenv(env_name)) {
        if (*v) {
            char* end = nullptr;
            const long parsed = std::strtol(v, &end, 10);
            if (end != v && parsed > 0) {
                value = static_cast<uint32_t>(parsed);
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.emplace(env_name, value);
    }
    return value;
}

inline uint32_t span_tiling_default_max_from_tile_dims(uint32_t tile_x, uint32_t tile_y) {
    const uint64_t area = static_cast<uint64_t>(tile_x) * static_cast<uint64_t>(tile_y);
    if (area == 0) return 1u;
    const uint64_t max_u32 = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    return static_cast<uint32_t>(std::min<uint64_t>(area, max_u32));
}

inline uint32_t span_tiling_default_max_from_tile_px(uint32_t tile_px) {
    return span_tiling_default_max_from_tile_dims(tile_px, tile_px);
}

inline AbstractTensorPool::Options span_bin_pool_options() {
    AbstractTensorPool::Options opt;
    opt.clear_on_release = false;
    opt.cache_handles = true;
    opt.enable_shape_bucketing = false;
    opt.max_cached_handles_total = 8;
    opt.max_cached_handles_per_key = 2;
    return opt;
}

inline AbstractTensorPool& span_bin_pool() {
    static thread_local AbstractTensorPool pool(span_bin_pool_options());
    return pool;
}

inline bool span_ensure_tensor_zeroed(AbstractTensor& tensor) {
    if (!tensor.valid()) return false;
    auto* mem = dynamic_cast<InMemoryBackend*>(tensor.backend());
    if (!mem) return false;
    return mem->ensure_zeroed(tensor.handle(), tensor.desc());
}

struct SpanBinBuffers {
    AbstractTensorPool::PooledTensor counts;
    AbstractTensorPool::PooledTensor offsets;
    uint32_t* counts_ptr = nullptr;
    uint32_t* offsets_ptr = nullptr;
    uint32_t len = 0;
};

inline bool acquire_span_bin_buffers(uint32_t len,
                                     TensorBackend* backend,
                                     SpanBinBuffers& out) {
    out = SpanBinBuffers{};
    if (!backend || len == 0) return false;

    TensorDesc cdesc{};
    cdesc.dtype = TensorDType::U32;
    cdesc.layout = TensorLayout::Dense;
    cdesc.shape.dims = {len};

    TensorDesc odesc{};
    odesc.dtype = TensorDType::U32;
    odesc.layout = TensorLayout::Dense;
    odesc.shape.dims = {len + 1u};

    out.counts = span_bin_pool().acquire(cdesc, backend);
    out.offsets = span_bin_pool().acquire(odesc, backend);
    if (!out.counts.valid() || !out.offsets.valid()) return false;

    (void)span_ensure_tensor_zeroed(out.counts.tensor());

    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem) return false;

    void* cptr = nullptr;
    size_t cbytes = 0;
    void* optr = nullptr;
    size_t obytes = 0;
    if (!mem->map(out.counts.tensor().handle(), &cptr, &cbytes)) return false;
    if (!mem->map(out.offsets.tensor().handle(), &optr, &obytes)) {
        mem->unmap(out.counts.tensor().handle());
        return false;
    }
    out.counts_ptr = static_cast<uint32_t*>(cptr);
    out.offsets_ptr = static_cast<uint32_t*>(optr);
    out.len = len;
    return true;
}

inline void release_span_bin_buffers(TensorBackend* backend, SpanBinBuffers& buf) {
    if (!backend) return;
    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem) return;
    if (buf.counts.valid()) mem->unmap(buf.counts.tensor().handle());
    if (buf.offsets.valid()) mem->unmap(buf.offsets.tensor().handle());
    buf.counts_ptr = nullptr;
    buf.offsets_ptr = nullptr;
    buf.len = 0;
}

inline void build_span_records_from_coords_2d(const int64_t* coords,
                                               const uint8_t* in_bounds,
                                               uint32_t count,
                                               uint32_t width,
                                               uint32_t height,
                                               std::vector<SpanRecord2D>& out) {
    out.clear();
    if (!coords || !in_bounds || count == 0) return;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (!in_bounds[i]) continue;
        const int64_t xi = coords[static_cast<size_t>(i) * 2 + 0];
        const int64_t yi = coords[static_cast<size_t>(i) * 2 + 1];
        if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(width) ||
            yi >= static_cast<int64_t>(height)) {
            continue;
        }
        const uint64_t linear = static_cast<uint64_t>(yi) * width + static_cast<uint64_t>(xi);
        out.push_back(SpanRecord2D{linear, i});
    }
}

// Span tiling execution shells (centralized span management). These will be filled in later.
template <typename Scalar>
inline bool span_scatter_add_2d(const int64_t* /*coords*/, const uint8_t* /*in_bounds*/, uint32_t /*count*/,
                                uint32_t /*width*/, uint32_t /*height*/, uint32_t /*channels*/,
                                bool /*val_scalar*/, const Scalar* /*values*/, Scalar* /*out*/,
                                TensorBackend* /*backend*/, uint32_t /*tile_px*/,
                                const SpanOpConfig& op) {
    if constexpr (!std::is_integral_v<Scalar> || !std::is_unsigned_v<Scalar>) {
        return false;
    }
    if (!op.is_add()) return false;
    return false;
}

template <typename Scalar>
inline bool span_gather_2d(const int64_t* /*coords*/, const uint8_t* /*in_bounds*/, uint32_t /*count*/,
                           uint32_t /*width*/, uint32_t /*height*/, uint32_t /*channels*/,
                           const Scalar* /*base*/, Scalar* /*out*/, TensorBackend* /*backend*/, uint32_t /*tile_px*/,
                           const SpanOpConfig& op) {
    if constexpr (!std::is_integral_v<Scalar> || !std::is_unsigned_v<Scalar>) {
        return false;
    }
    if (!op.is_add() && !op.is_replace()) return false;
    return false;
}

template <typename Scalar>
inline bool span_gather_add_2d(const int64_t* /*coords*/, const uint8_t* /*in_bounds*/, uint32_t /*count*/,
                               uint32_t /*width*/, uint32_t /*height*/, uint32_t /*channels*/,
                               const Scalar* /*base*/, Scalar* /*out*/, TensorBackend* /*backend*/, uint32_t /*tile_px*/,
                               const SpanOpConfig& op) {
    if constexpr (!std::is_integral_v<Scalar> || !std::is_unsigned_v<Scalar>) {
        return false;
    }
    if (!op.is_add()) return false;
    return false;
}

template <typename Scalar>
inline bool span_scatter_stencil_2d(const int64_t* /*coords*/, const uint8_t* /*in_bounds*/, uint32_t /*count*/,
                                    uint32_t /*width*/, uint32_t /*height*/, uint32_t /*channels*/,
                                    uint32_t /*taps*/, const int32_t* /*offsets*/, const Scalar* /*weights*/,
                                    bool /*val_scalar*/, const Scalar* /*values*/, Scalar* /*out*/,
                                    TensorBackend* /*backend*/, uint32_t /*tile_px_x*/, uint32_t /*tile_px_y*/,
                                    const SpanOpConfig& op,
                                    const SpanStencilPlan2D<Scalar>* /*plan*/ = nullptr) {
    if constexpr (!std::is_integral_v<Scalar> || !std::is_unsigned_v<Scalar>) {
        return false;
    }
    if (!op.is_add() || !op.use_stencil) return false;
    return false;
}

template <typename Scalar>
inline bool span_scatter_add_2d_kernel(const int64_t* /*coords*/, const uint8_t* /*in_bounds*/, uint32_t /*count*/,
                                       uint32_t /*width*/, uint32_t /*height*/, uint32_t /*channels*/,
                                       uint32_t /*kcols*/, const Scalar* /*kernel*/, const Scalar* /*values*/,
                                       bool /*val_scalar*/, Scalar* /*out*/, TensorBackend* /*backend*/,
                                       uint32_t /*tile_px_x*/, uint32_t /*tile_px_y*/,
                                       const SpanOpConfig& op) {
    if constexpr (!std::is_integral_v<Scalar> || !std::is_unsigned_v<Scalar>) {
        return false;
    }
    if (!op.is_add()) return false;
    return false;
}

template <typename RecordT>
inline void build_contiguous_spans_from_dense_bins_arena(const std::vector<RecordT>& records,
                                                         uint64_t linear_len,
                                                         uint32_t max_len,
                                                         std::vector<RecordT>& out_sorted,
                                                         uint32_t* counts,
                                                         uint32_t* offsets,
                                                         std::vector<TileSpan>& out_spans) {
    out_spans.clear();
    out_sorted.clear();
    if (records.empty() || linear_len == 0 || !counts || !offsets) return;
    if (max_len == 0) max_len = 1u;

    const size_t len = static_cast<size_t>(linear_len);
    out_sorted.resize(records.size());

    for (const auto& rec : records) {
        if (rec.linear >= linear_len) continue;
        counts[static_cast<size_t>(rec.linear)]++;
    }

    uint32_t running = 0;
    for (size_t i = 0; i < len; ++i) {
        offsets[i] = running;
        running += counts[i];
    }
    offsets[len] = running;

    std::vector<uint32_t> write_pos(offsets, offsets + len);
    for (const auto& rec : records) {
        if (rec.linear >= linear_len) continue;
        const size_t idx = static_cast<size_t>(rec.linear);
        out_sorted[write_pos[idx]++] = rec;
    }

    size_t i = 0;
    while (i < len) {
        if (counts[i] == 0u) {
            ++i;
            continue;
        }
        const size_t run_start = i;
        while (i < len && counts[i] > 0u) {
            ++i;
        }
        const size_t run_end = i;

        size_t chunk_start = run_start;
        while (chunk_start < run_end) {
            const size_t chunk_end = std::min(run_end, chunk_start + static_cast<size_t>(max_len));
            TileSpan span{};
            span.start = static_cast<uint64_t>(chunk_start);
            span.len = static_cast<uint32_t>(chunk_end - chunk_start);
            span.record_begin = offsets[chunk_start];
            span.record_end = offsets[chunk_end];
            out_spans.push_back(span);
            chunk_start = chunk_end;
        }
    }
}

template <typename RecordT>
inline void build_contiguous_spans_from_sorted_records(const std::vector<RecordT>& records,
                                                       uint32_t max_len,
                                                       std::vector<TileSpan>& out) {
    out.clear();
    if (records.empty()) return;
    if (max_len == 0) max_len = 1u;

    size_t r = 0;
    const size_t n = records.size();
    while (r < n) {
        const uint64_t span_start = records[r].linear;
        uint64_t prev = span_start;
        const size_t span_begin = r;
        ++r;
        for (; r < n; ++r) {
            const uint64_t cur = records[r].linear;

            if (cur == prev + 1u) {
                prev = cur;
                continue;
            }
            break;
        }
        const size_t span_end = r;
        uint64_t span_len = prev - span_start + 1u;

        uint64_t chunk_start = span_start;
        size_t chunk_begin = span_begin;
        while (span_len > 0) {
            const uint32_t chunk_len = static_cast<uint32_t>(std::min<uint64_t>(span_len, max_len));
            const uint64_t chunk_last = chunk_start + static_cast<uint64_t>(chunk_len - 1u);
            size_t chunk_end = chunk_begin;
            while (chunk_end < span_end && records[chunk_end].linear <= chunk_last) {
                ++chunk_end;
            }

            TileSpan span{};
            span.start = chunk_start;
            span.len = chunk_len;
            span.record_begin = static_cast<uint32_t>(chunk_begin);
            span.record_end = static_cast<uint32_t>(chunk_end);
            out.push_back(span);

            chunk_start += static_cast<uint64_t>(chunk_len);
            span_len -= static_cast<uint64_t>(chunk_len);
            chunk_begin = chunk_end;
        }
    }
}

inline void apply_affine_row_major(const float* m,
                                   float x,
                                   float y,
                                   float z,
                                   float& ox,
                                   float& oy,
                                   float& oz) {
    // Row-vector convention: v' = v * M (translation in last row).
    ox = x * m[0] + y * m[4] + z * m[8] + m[12];
    oy = x * m[1] + y * m[5] + z * m[9] + m[13];
    oz = x * m[2] + y * m[6] + z * m[10] + m[14];
}

inline int64_t quantize_round_fast(float v) {
    // Bitwise round-to-nearest-even for float32 in the index-dominant range.
    // Assumes IEEE-754 binary32 and finite inputs on the hot path.
    uint32_t b = 0u;
    std::memcpy(&b, &v, sizeof(b));
    const uint32_t absb = b & 0x7FFFFFFFu;

    // |v| >= 2^23 -> float already has integer resolution or is extreme.
    // Let the cast handle the rare/external slow-path domain.
    if (absb >= 0x4B000000u) {
        return static_cast<int64_t>(v);
    }

    const uint32_t exp_field = (absb >> 23) & 0xFFu;
    const int32_t exp = static_cast<int32_t>(exp_field) - 127;

    if (exp < 0) {
        // |v| < 1.0f : round-to-nearest-even around zero.
        if (absb < 0x3F000000u) return 0; // |v| < 0.5f
        return (b >> 31) ? -1 : 1;
    }

    // Normalized mantissa with implicit leading 1.
    const uint32_t mant = (absb & 0x7FFFFFu) | 0x800000u;
    const uint32_t shift = 23u - static_cast<uint32_t>(exp);
    const uint32_t intpart = mant >> shift;
    const uint32_t frac = mant & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1u);
    const uint32_t inc = (frac > half) | ((frac == half) & (intpart & 1u));
    const int64_t r = static_cast<int64_t>(intpart + inc);
    return (b >> 31) ? -r : r;
}
struct PointIndexReader {
    const TensorDesc* pd = nullptr;
    const void* points_ptr = nullptr;
    uint32_t dims = 0;
    bool points_match = false;
    bool use_affine = false;
    const float* affine = nullptr;

    inline float read_point_f(uint64_t idx) const {
        if (!pd || !points_ptr) return 0.0f;
        switch (pd->dtype) {
            case TensorDType::F32:
                return static_cast<const float*>(points_ptr)[idx];
            case TensorDType::F64:
                return static_cast<float>(static_cast<const double*>(points_ptr)[idx]);
            default:
                return 0.0f;
        }
    }

    inline float read_point_as_float(uint64_t idx) const {
        if (!pd || !points_ptr) return 0.0f;
        switch (pd->dtype) {
            case TensorDType::F32:
                return static_cast<const float*>(points_ptr)[idx];
            case TensorDType::F64:
                return static_cast<float>(static_cast<const double*>(points_ptr)[idx]);
            case TensorDType::I8:
                return static_cast<float>(static_cast<const int8_t*>(points_ptr)[idx]);
            case TensorDType::I16:
                return static_cast<float>(static_cast<const int16_t*>(points_ptr)[idx]);
            case TensorDType::I32:
                return static_cast<float>(static_cast<const int32_t*>(points_ptr)[idx]);
            case TensorDType::I64:
                return static_cast<float>(static_cast<const int64_t*>(points_ptr)[idx]);
            case TensorDType::U8:
                return static_cast<float>(static_cast<const uint8_t*>(points_ptr)[idx]);
            case TensorDType::U16:
                return static_cast<float>(static_cast<const uint16_t*>(points_ptr)[idx]);
            case TensorDType::U32:
                return static_cast<float>(static_cast<const uint32_t*>(points_ptr)[idx]);
            case TensorDType::U64:
                return static_cast<float>(static_cast<const uint64_t*>(points_ptr)[idx]);
            default:
                return 0.0f;
        }
    }

    inline int64_t read_index(uint32_t i, uint32_t d) const {
        if (!pd || !points_ptr) return 0;
        const uint64_t idx = static_cast<uint64_t>(i) * dims + d;
        switch (pd->dtype) {
            case TensorDType::I8:
                return static_cast<int64_t>(static_cast<const int8_t*>(points_ptr)[idx]);
            case TensorDType::I16:
                return static_cast<int64_t>(static_cast<const int16_t*>(points_ptr)[idx]);
            case TensorDType::I32:
                return static_cast<int64_t>(static_cast<const int32_t*>(points_ptr)[idx]);
            case TensorDType::I64:
                return static_cast<const int64_t*>(points_ptr)[idx];
            case TensorDType::U8:
                return static_cast<int64_t>(static_cast<const uint8_t*>(points_ptr)[idx]);
            case TensorDType::U16:
                return static_cast<int64_t>(static_cast<const uint16_t*>(points_ptr)[idx]);
            case TensorDType::U32:
                return static_cast<int64_t>(static_cast<const uint32_t*>(points_ptr)[idx]);
            case TensorDType::U64:
                return static_cast<int64_t>(static_cast<const uint64_t*>(points_ptr)[idx]);
            default:
                return 0;
        }
    }

    inline void read_xy(uint32_t i, int64_t& xi, int64_t& yi) const {
        if (use_affine) {
            float xf = read_point_as_float(static_cast<uint64_t>(i) * dims + 0);
            float yf = read_point_as_float(static_cast<uint64_t>(i) * dims + 1);
            float xo = 0.0f, yo = 0.0f, zo = 0.0f;
            apply_affine_row_major(affine, xf, yf, 0.0f, xo, yo, zo);
            xi = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(xo));
            yi = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(yo));
            return;
        }

        if (points_match) {
            float xf = read_point_f(static_cast<uint64_t>(i) * dims + 0);
            float yf = read_point_f(static_cast<uint64_t>(i) * dims + 1);
            xi = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(xf));
            yi = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(yf));
            return;
        }

        xi = read_index(i, 0);
        yi = read_index(i, 1);
    }

    inline void read_coords(uint32_t i, int64_t* out) const {
        if (!out || dims == 0) return;
        if (use_affine) {
            float xf = read_point_as_float(static_cast<uint64_t>(i) * dims + 0);
            float yf = (dims > 1) ? read_point_as_float(static_cast<uint64_t>(i) * dims + 1) : 0.0f;
            float zf = (dims > 2) ? read_point_as_float(static_cast<uint64_t>(i) * dims + 2) : 0.0f;
            float xo = 0.0f, yo = 0.0f, zo = 0.0f;
            apply_affine_row_major(affine, xf, yf, zf, xo, yo, zo);
            out[0] = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(xo));
            if (dims > 1) out[1] = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(yo));
            if (dims > 2) out[2] = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(zo));
            for (uint32_t d = 3; d < dims; ++d) {
                out[d] = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(
                    read_point_as_float(static_cast<uint64_t>(i) * dims + d)));
            }
            return;
        }

        if (points_match) {
            float xf = read_point_f(static_cast<uint64_t>(i) * dims + 0);
            float yf = (dims > 1) ? read_point_f(static_cast<uint64_t>(i) * dims + 1) : 0.0f;
            float zf = (dims > 2) ? read_point_f(static_cast<uint64_t>(i) * dims + 2) : 0.0f;
            out[0] = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(xf));
            if (dims > 1) out[1] = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(yf));
            if (dims > 2) out[2] = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(zf));
            for (uint32_t d = 3; d < dims; ++d) {
                out[d] = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(
                    read_point_f(static_cast<uint64_t>(i) * dims + d)));
            }
            return;
        }

        for (uint32_t d = 0; d < dims; ++d) {
            out[d] = read_index(i, d);
        }
    }
};

inline void convert_points_to_int_coords(const TensorDesc& pd,
                                         const void* points_ptr,
                                         uint32_t count,
                                         uint32_t dims,
                                         bool points_match,
                                         bool use_affine,
                                         const float* affine,
                                         const uint32_t* bounds,
                                         bool clamp_to_bounds,
                                         int64_t* out_coords,
                                         uint8_t* out_in_bounds) {
    const bool use_bounds = bounds != nullptr;
    const bool emit_in_bounds = out_in_bounds != nullptr;


    // Aggressive XY fast paths: avoid per-dim loops and runtime dtype switches.
    if (dims == 2 && points_ptr) {
        const int64_t bx = use_bounds ? static_cast<int64_t>(bounds[0]) : 0;
        const int64_t by = use_bounds ? static_cast<int64_t>(bounds[1]) : 0;

        auto run_xy_dispatch = [&](auto* typed_ptr) {
            using SrcT = std::remove_pointer_t<decltype(typed_ptr)>;
            if (use_bounds) {
                if (clamp_to_bounds) {
                    if (emit_in_bounds) {
                        run_xy<true, true, true, SrcT>(typed_ptr, count, out_coords, out_in_bounds, bx, by);
                    } else {
                        run_xy<true, true, false, SrcT>(typed_ptr, count, out_coords, out_in_bounds, bx, by);
                    }
                } else {
                    if (emit_in_bounds) {
                        run_xy<true, false, true, SrcT>(typed_ptr, count, out_coords, out_in_bounds, bx, by);
                    } else {
                        run_xy<true, false, false, SrcT>(typed_ptr, count, out_coords, out_in_bounds, bx, by);
                    }
                }
            } else {
                if (emit_in_bounds) {
                    run_xy<false, false, true, SrcT>(typed_ptr, count, out_coords, out_in_bounds, bx, by);
                } else {
                    run_xy<false, false, false, SrcT>(typed_ptr, count, out_coords, out_in_bounds, bx, by);
                }
            }
        };

        auto run_xy_affine_dispatch = [&](auto* typed_ptr, auto m0, auto m1, auto m4, auto m5, auto m12, auto m13) {
            using SrcT = std::remove_pointer_t<decltype(typed_ptr)>;
            using AffineT = decltype(m0);
            if (use_bounds) {
                if (clamp_to_bounds) {
                    if (emit_in_bounds) {
                        run_xy_affine<true, true, true, SrcT, AffineT>(typed_ptr, count, out_coords, out_in_bounds,
                                                                       bx, by, m0, m1, m4, m5, m12, m13);
                    } else {
                        run_xy_affine<true, true, false, SrcT, AffineT>(typed_ptr, count, out_coords, out_in_bounds,
                                                                        bx, by, m0, m1, m4, m5, m12, m13);
                    }
                } else {
                    if (emit_in_bounds) {
                        run_xy_affine<true, false, true, SrcT, AffineT>(typed_ptr, count, out_coords, out_in_bounds,
                                                                        bx, by, m0, m1, m4, m5, m12, m13);
                    } else {
                        run_xy_affine<true, false, false, SrcT, AffineT>(typed_ptr, count, out_coords, out_in_bounds,
                                                                         bx, by, m0, m1, m4, m5, m12, m13);
                    }
                }
            } else {
                if (emit_in_bounds) {
                    run_xy_affine<false, false, true, SrcT, AffineT>(typed_ptr, count, out_coords, out_in_bounds,
                                                                     bx, by, m0, m1, m4, m5, m12, m13);
                } else {
                    run_xy_affine<false, false, false, SrcT, AffineT>(typed_ptr, count, out_coords, out_in_bounds,
                                                                      bx, by, m0, m1, m4, m5, m12, m13);
                }
            }
        };

        if (!use_affine) {
            switch (pd.dtype) {
                case TensorDType::F32:
                    run_xy_dispatch(static_cast<const float*>(points_ptr));
                    return;
                case TensorDType::F64:
                    run_xy_dispatch(static_cast<const double*>(points_ptr));
                    return;
                case TensorDType::I8:
                    run_xy_dispatch(static_cast<const int8_t*>(points_ptr));
                    return;
                case TensorDType::I16:
                    run_xy_dispatch(static_cast<const int16_t*>(points_ptr));
                    return;
                case TensorDType::I32:
                    run_xy_dispatch(static_cast<const int32_t*>(points_ptr));
                    return;
                case TensorDType::I64:
                    run_xy_dispatch(static_cast<const int64_t*>(points_ptr));
                    return;
                case TensorDType::U8:
                    run_xy_dispatch(static_cast<const uint8_t*>(points_ptr));
                    return;
                case TensorDType::U16:
                    run_xy_dispatch(static_cast<const uint16_t*>(points_ptr));
                    return;
                case TensorDType::U32:
                    run_xy_dispatch(static_cast<const uint32_t*>(points_ptr));
                    return;
                case TensorDType::U64:
                    run_xy_dispatch(static_cast<const uint64_t*>(points_ptr));
                    return;
                default:
                    break;
            }
        } else if (affine) {
            const float m0f = affine[0];
            const float m1f = affine[1];
            const float m4f = affine[4];
            const float m5f = affine[5];
            const float m12f = affine[12];
            const float m13f = affine[13];
            switch (pd.dtype) {
                case TensorDType::F64: {
                    const double m0 = static_cast<double>(m0f);
                    const double m1 = static_cast<double>(m1f);
                    const double m4 = static_cast<double>(m4f);
                    const double m5 = static_cast<double>(m5f);
                    const double m12 = static_cast<double>(m12f);
                    const double m13 = static_cast<double>(m13f);
                    run_xy_affine_dispatch(static_cast<const double*>(points_ptr), m0, m1, m4, m5, m12, m13);
                    return;
                }
                case TensorDType::F32:
                    run_xy_affine_dispatch(static_cast<const float*>(points_ptr), m0f, m1f, m4f, m5f, m12f, m13f);
                    return;
                case TensorDType::I8:
                    run_xy_affine_dispatch(static_cast<const int8_t*>(points_ptr), m0f, m1f, m4f, m5f, m12f, m13f);
                    return;
                case TensorDType::I16:
                    run_xy_affine_dispatch(static_cast<const int16_t*>(points_ptr), m0f, m1f, m4f, m5f, m12f, m13f);
                    return;
                case TensorDType::I32:
                    run_xy_affine_dispatch(static_cast<const int32_t*>(points_ptr), m0f, m1f, m4f, m5f, m12f, m13f);
                    return;
                case TensorDType::I64:
                    run_xy_affine_dispatch(static_cast<const int64_t*>(points_ptr), m0f, m1f, m4f, m5f, m12f, m13f);
                    return;
                case TensorDType::U8:
                    run_xy_affine_dispatch(static_cast<const uint8_t*>(points_ptr), m0f, m1f, m4f, m5f, m12f, m13f);
                    return;
                case TensorDType::U16:
                    run_xy_affine_dispatch(static_cast<const uint16_t*>(points_ptr), m0f, m1f, m4f, m5f, m12f, m13f);
                    return;
                case TensorDType::U32:
                    run_xy_affine_dispatch(static_cast<const uint32_t*>(points_ptr), m0f, m1f, m4f, m5f, m12f, m13f);
                    return;
                case TensorDType::U64:
                    run_xy_affine_dispatch(static_cast<const uint64_t*>(points_ptr), m0f, m1f, m4f, m5f, m12f, m13f);
                    return;
                default:
                    break;
            }
        }
    }

    if (!use_affine && points_ptr &&
        (pd.dtype == TensorDType::I8 || pd.dtype == TensorDType::I16 ||
         pd.dtype == TensorDType::I32 || pd.dtype == TensorDType::I64 ||
         pd.dtype == TensorDType::U8 || pd.dtype == TensorDType::U16 ||
         pd.dtype == TensorDType::U32 || pd.dtype == TensorDType::U64)) {
        auto convert_int = [&](auto* src) {
            for (uint32_t i = 0; i < count; ++i) {
                int64_t* dst = out_coords + static_cast<size_t>(i) * dims;
                bool in_bounds = true;
                for (uint32_t d = 0; d < dims; ++d) {
                    const uint64_t idx = static_cast<uint64_t>(i) * dims + d;
                    int64_t v = static_cast<int64_t>(src[idx]);
                    if (bounds) {
                        const int64_t limit = static_cast<int64_t>(bounds[d]);
                        if (v < 0 || v >= limit) {
                            in_bounds = false;
                        }
                        if (clamp_to_bounds && limit > 0) {
                            v = std::clamp<int64_t>(v, 0, limit - 1);
                        }
                    }
                    dst[d] = v;
                }
                if (out_in_bounds) {
                    out_in_bounds[i] = in_bounds ? 1u : 0u;
                }
            }
        };

        switch (pd.dtype) {
            case TensorDType::I8:
                convert_int(static_cast<const int8_t*>(points_ptr));
                return;
            case TensorDType::I16:
                convert_int(static_cast<const int16_t*>(points_ptr));
                return;
            case TensorDType::I32:
                convert_int(static_cast<const int32_t*>(points_ptr));
                return;
            case TensorDType::I64:
                convert_int(static_cast<const int64_t*>(points_ptr));
                return;
            case TensorDType::U8:
                convert_int(static_cast<const uint8_t*>(points_ptr));
                return;
            case TensorDType::U16:
                convert_int(static_cast<const uint16_t*>(points_ptr));
                return;
            case TensorDType::U32:
                convert_int(static_cast<const uint32_t*>(points_ptr));
                return;
            case TensorDType::U64:
                convert_int(static_cast<const uint64_t*>(points_ptr));
                return;
            default:
                break;
        }
    }

    PointIndexReader reader{};
    reader.pd = &pd;
    reader.points_ptr = points_ptr;
    reader.dims = dims;
    reader.points_match = points_match;
    reader.use_affine = use_affine;
    reader.affine = affine;

    for (uint32_t i = 0; i < count; ++i) {
        int64_t* dst = out_coords + static_cast<size_t>(i) * dims;
        reader.read_coords(i, dst);

        bool in_bounds = true;
        if (bounds) {
            for (uint32_t d = 0; d < dims; ++d) {
                const int64_t limit = static_cast<int64_t>(bounds[d]);
                if (dst[d] < 0 || dst[d] >= limit) {
                    in_bounds = false;
                }
                if (clamp_to_bounds) {
                    if (limit > 0) {
                        dst[d] = std::clamp<int64_t>(dst[d], 0, limit - 1);
                    }
                }
            }
        }

        if (out_in_bounds) {
            out_in_bounds[i] = in_bounds ? 1u : 0u;
        }
    }
}

} // namespace nodus::tensors
