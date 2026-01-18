#include "common/tensors/abstraction/tensor_math.h"

#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/abstract_tensor_pool.h"
#include "common/tensors/abstraction/dyadic_bins.h"
#include "common/tensors/abstraction/microkernels.h"
#include "common/tensors/abstraction/tensor_types.h"
#include "common/tensors/abstraction/tensor_tiling_strategy.h"
#include "common/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <limits>
#include <unordered_map>
#include <type_traits>

#if defined(NODUS_DYADIC_SCATTER_DEBUG)
#define DYADIC_SCATTER_LOGF(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define DYADIC_SCATTER_LOGF(...) ((void)0)
#endif

namespace nodus::tensors {

struct TensorOpTensor;
static bool compute_dense_strides_u64(const std::vector<uint32_t>& dims, std::vector<uint64_t>& out) {
    if (dims.empty()) return false;
    out.resize(dims.size());
    uint64_t stride = 1;
    for (size_t i = dims.size(); i-- > 0;) {
        out[i] = stride;
        stride *= static_cast<uint64_t>(dims[i]);
    }
    return true;
}

static bool compute_strides_for_desc_u64(const TensorDesc& desc, std::vector<uint64_t>& out) {
    if (desc.layout == TensorLayout::Strided && desc.strides.elems.size() == desc.shape.dims.size()) {
        out = desc.strides.elems;
        return true;
    }
    return compute_dense_strides_u64(desc.shape.dims, out);
}

bool tensor_gli_from_coords(const TensorDesc& desc,
                            const std::vector<uint32_t>& coords,
                            uint64_t* out_gli) {
    if (!out_gli) return false;
    *out_gli = 0;
    const size_t rank = desc.shape.dims.size();
    if (coords.size() != rank) return false;
    std::vector<uint64_t> dense_strides;
    if (!compute_dense_strides_u64(desc.shape.dims, dense_strides)) return false;
    uint64_t gli = 0;
    for (size_t d = 0; d < rank; ++d) {
        if (coords[d] >= desc.shape.dims[d]) return false;
        gli += static_cast<uint64_t>(coords[d]) * dense_strides[d];
    }
    *out_gli = gli;
    return true;
}

bool tensor_elem_offset_from_gli(const TensorDesc& desc,
                                 uint64_t gli,
                                 uint64_t* out_elem_offset) {
    if (!out_elem_offset) return false;
    *out_elem_offset = 0;
    const size_t rank = desc.shape.dims.size();
    if (rank == 0) return false;
    std::vector<uint64_t> dense_strides;
    if (!compute_dense_strides_u64(desc.shape.dims, dense_strides)) return false;
    std::vector<uint64_t> elem_strides;
    if (!compute_strides_for_desc_u64(desc, elem_strides)) return false;
    if (elem_strides.size() != rank) return false;
    uint64_t rem = gli;
    uint64_t elem_off = 0;
    for (size_t d = 0; d < rank; ++d) {
        const uint64_t stride = dense_strides[d];
        const uint64_t dim = desc.shape.dims[d];
        uint64_t coord = 0;
        if (stride > 0) {
            coord = rem / stride;
            rem = rem % stride;
        }
        if (coord >= dim) return false;
        elem_off += coord * elem_strides[d];
    }
    *out_elem_offset = elem_off;
    return true;
}

bool tensor_byte_offset_from_gli(const TensorDesc& desc,
                                 uint64_t gli,
                                 uint64_t* out_byte_offset) {
    if (!out_byte_offset) return false;
    uint64_t elem_off = 0;
    if (!tensor_elem_offset_from_gli(desc, gli, &elem_off)) return false;
    const uint32_t elem_bytes = tensor_dtype_size_bytes(desc.dtype);
    if (elem_bytes == 0) return false;
    *out_byte_offset = elem_off * static_cast<uint64_t>(elem_bytes);
    return true;
}

namespace {

struct TensorOpTensor;
static bool build_op_tensor(const TensorDesc& desc, const std::vector<uint64_t>& shape, void* base, TensorOpTensor& out);


struct TileLockToken;

struct TileLockTable {
    static constexpr uint32_t kLockCount = 4096u;
    std::atomic_flag locks[kLockCount];

    TileLockTable() {
        for (uint32_t i = 0; i < kLockCount; ++i) {
            locks[i].clear(std::memory_order_release);
        }
    }

    TileLockToken acquire(uint64_t tile_id);
    void release(uint32_t index) {
        locks[index].clear(std::memory_order_release);
    }
};

struct TileLockToken {
    TileLockTable* table = nullptr;
    uint32_t index = 0;

    TileLockToken() = default;
    TileLockToken(TileLockTable* t, uint32_t i) : table(t), index(i) {}
    TileLockToken(const TileLockToken&) = delete;
    TileLockToken& operator=(const TileLockToken&) = delete;

    TileLockToken(TileLockToken&& other) noexcept {
        table = other.table;
        index = other.index;
        other.table = nullptr;
    }
    TileLockToken& operator=(TileLockToken&& other) noexcept {
        if (this == &other) return *this;
        if (table) table->release(index);
        table = other.table;
        index = other.index;
        other.table = nullptr;
        return *this;
    }

    ~TileLockToken() {
        if (table) table->release(index);
    }
};

inline TileLockToken TileLockTable::acquire(uint64_t tile_id) {
    const uint32_t index = static_cast<uint32_t>(tile_id) & (kLockCount - 1u);
    while (locks[index].test_and_set(std::memory_order_acquire)) {
    }
    return TileLockToken(this, index);
}

static TileLockTable& tile_lock_table() {
    static TileLockTable table;
    return table;
}

static TileLockToken acquire_tile_lock(uint64_t tile_id) {
    return tile_lock_table().acquire(tile_id);
}

static uint64_t choose_linear_tile_elems(const TensorDesc& desc,
                                         uint32_t dims,
                                         uint64_t cache_budget_bytes) {
    if (dims == 0) return 1u;
    const uint32_t elem_bytes = tensor_dtype_size_bytes(desc.dtype);
    if (elem_bytes == 0) return 1u;
    const uint64_t budget_elems = std::max<uint64_t>(1u, cache_budget_bytes / elem_bytes);
    const double base = std::pow(static_cast<double>(budget_elems), 1.0 / static_cast<double>(dims));
    const uint64_t side = std::max<uint64_t>(1u, static_cast<uint64_t>(base));
    uint64_t tile_elems = 1u;
    for (uint32_t d = 0; d < dims; ++d) {
        if (tile_elems > budget_elems / side) {
            tile_elems = budget_elems;
            break;
        }
        tile_elems *= side;
    }
    return std::max<uint64_t>(1u, tile_elems);
}

static TileShape2D choose_tile_shape_2d_impl(const TensorDesc& desc,
                                             uint32_t radius_x,
                                             uint32_t radius_y,
                                             uint64_t cache_budget_bytes,
                                             TileContiguityStrategy strategy) {
    TileShape2D shape{};
    if (desc.shape.dims.size() < 2) return shape;
    const uint32_t height = desc.shape.dims[0];
    const uint32_t width = desc.shape.dims[1];
    if (height == 0 || width == 0) return shape;

    const uint32_t elem_bytes = tensor_dtype_size_bytes(desc.dtype);
    if (elem_bytes == 0) return shape;
    const uint64_t budget_elems = cache_budget_bytes / static_cast<uint64_t>(elem_bytes);
    if (budget_elems == 0) return shape;

    std::vector<uint64_t> elem_strides;
    if (!compute_strides_for_desc_u64(desc, elem_strides) || elem_strides.size() < 2) return shape;

    struct DimRank {
        uint32_t dim = 0;
        uint64_t stride = 0;
    };
    DimRank dims[2] = {
        {0u, elem_strides[0]},
        {1u, elem_strides[1]}
    };
    if (dims[0].stride > dims[1].stride) {
        std::swap(dims[0], dims[1]);
    }

    float alpha[2] = {0.0f, 0.0f};
    switch (strategy) {
        case TileContiguityStrategy::PreferRowMajor:
            alpha[0] = 0.25f;
            alpha[1] = 0.75f;
            break;
        case TileContiguityStrategy::PreferColumnMajor:
            alpha[0] = 0.75f;
            alpha[1] = 0.25f;
            break;
        case TileContiguityStrategy::PreferFastestStride:
        case TileContiguityStrategy::Auto:
        default: {
            float denom = 0.0f;
            for (uint32_t i = 0; i < 2; ++i) {
                const float v = 1.0f / static_cast<float>(1u + i);
                alpha[dims[i].dim] = v;
                denom += v;
            }
            if (denom <= 0.0f) return shape;
            alpha[0] /= denom;
            alpha[1] /= denom;
            break;
        }
    }

    auto volume_ok = [&](uint64_t t, uint32_t& out_x, uint32_t& out_y) -> bool {
        double tx = std::pow(static_cast<double>(t), static_cast<double>(alpha[1]));
        double ty = std::pow(static_cast<double>(t), static_cast<double>(alpha[0]));
        uint32_t lx = std::max<uint32_t>(1u, static_cast<uint32_t>(tx));
        uint32_t ly = std::max<uint32_t>(1u, static_cast<uint32_t>(ty));
        lx = std::min<uint32_t>(lx, width);
        ly = std::min<uint32_t>(ly, height);

        uint64_t vx = static_cast<uint64_t>(lx) + static_cast<uint64_t>(radius_x) * 2u;
        uint64_t vy = static_cast<uint64_t>(ly) + static_cast<uint64_t>(radius_y) * 2u;
        if (vx == 0 || vy == 0) return false;
        if (vx > budget_elems || vy > budget_elems) return false;
        if (vx > 0 && vy > budget_elems / vx) return false;

        out_x = lx;
        out_y = ly;
        return true;
    };

    uint64_t lo = 1u;
    uint64_t hi = std::max<uint64_t>(1u, budget_elems);
    uint32_t best_x = 1u;
    uint32_t best_y = 1u;
    while (lo <= hi) {
        const uint64_t mid = (lo + hi) / 2u;
        uint32_t lx = 1u;
        uint32_t ly = 1u;
        if (volume_ok(mid, lx, ly)) {
            best_x = lx;
            best_y = ly;
            lo = mid + 1u;
        } else {
            if (mid == 0) break;
            hi = mid - 1u;
        }
    }

    const uint32_t simd_multiple = (elem_bytes == 8) ? 4u : 8u;
    uint32_t fast_dim = (dims[0].dim == 1u) ? 1u : 0u;
    if (strategy == TileContiguityStrategy::PreferRowMajor) {
        fast_dim = 1u;
    } else if (strategy == TileContiguityStrategy::PreferColumnMajor) {
        fast_dim = 0u;
    }
    if (fast_dim == 1u && best_x >= simd_multiple) {
        best_x = std::max<uint32_t>(1u, (best_x / simd_multiple) * simd_multiple);
    } else if (fast_dim == 0u && best_y >= simd_multiple) {
        best_y = std::max<uint32_t>(1u, (best_y / simd_multiple) * simd_multiple);
    }

    uint64_t vx = static_cast<uint64_t>(best_x) + static_cast<uint64_t>(radius_x) * 2u;
    uint64_t vy = static_cast<uint64_t>(best_y) + static_cast<uint64_t>(radius_y) * 2u;
    while (vx > 0 && vy > 0 && vx <= budget_elems && vy <= budget_elems && vx > 0 && vy > 0 && vx > budget_elems / vy) {
        if (fast_dim == 1u && best_x > 1u) {
            --best_x;
        } else if (fast_dim == 0u && best_y > 1u) {
            --best_y;
        } else if (best_x > 1u) {
            --best_x;
        } else if (best_y > 1u) {
            --best_y;
        } else {
            break;
        }
        vx = static_cast<uint64_t>(best_x) + static_cast<uint64_t>(radius_x) * 2u;
        vy = static_cast<uint64_t>(best_y) + static_cast<uint64_t>(radius_y) * 2u;
    }

    shape.x = std::max<uint32_t>(1u, best_x);
    shape.y = std::max<uint32_t>(1u, best_y);
    return shape;
}

static AbstractTensorPool::Options tile_pool_options() {
    AbstractTensorPool::Options opt;
    opt.clear_on_release = false;
    opt.cache_handles = true;
    opt.enable_shape_bucketing = false;
    opt.max_cached_handles_total = 8;
    opt.max_cached_handles_per_key = 2;
    return opt;
}

static AbstractTensorPool::Options coord_pool_options() {
    AbstractTensorPool::Options opt;
    opt.clear_on_release = false;
    opt.cache_handles = true;
    opt.enable_shape_bucketing = false;
    opt.max_cached_handles_total = 8;
    opt.max_cached_handles_per_key = 2;
    return opt;
}

static AbstractTensorPool& coord_pool() {
    static thread_local AbstractTensorPool pool(coord_pool_options());
    return pool;
}

static AbstractTensorPool& scatter_pool() {
    static thread_local AbstractTensorPool pool(coord_pool_options());
    return pool;
}

static bool ensure_tensor_zeroed(AbstractTensor& tensor) {
    if (!tensor.valid()) return false;
    auto* mem = dynamic_cast<InMemoryBackend*>(tensor.backend());
    if (!mem) return false;
    return mem->ensure_zeroed(tensor.handle(), tensor.desc());
}

static bool scatter_dyadic_enabled_from_env(const char* env) {
    if (!env) return false;
    if (const char* v = std::getenv(env)) {
        return (*v != 0 && std::strcmp(v, "0") != 0);
    }
    return false;
}

struct CoordBuffer {
    AbstractTensorPool::PooledTensor coords;
    AbstractTensorPool::PooledTensor mask;
    int64_t* coords_ptr = nullptr;
    uint8_t* mask_ptr = nullptr;
};

static inline uint64_t compute_index_range_u64(const uint32_t* shape, uint32_t dims) {
    uint64_t range = 1u;
    for (uint32_t d = 0; d < dims; ++d) {
        range *= static_cast<uint64_t>(shape[d]);
    }
    return range;
}

static inline TensorDType pick_unsigned_index_dtype(uint64_t range) {
    if (range <= static_cast<uint64_t>(std::numeric_limits<uint8_t>::max())) return TensorDType::U8;
    if (range <= static_cast<uint64_t>(std::numeric_limits<uint16_t>::max())) return TensorDType::U16;
    if (range <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) return TensorDType::U32;
    return TensorDType::U64;
}

template <typename IndexT>
constexpr TensorDType unsigned_index_dtype() {
    if constexpr (std::is_same_v<IndexT, uint8_t>) return TensorDType::U8;
    if constexpr (std::is_same_v<IndexT, uint16_t>) return TensorDType::U16;
    if constexpr (std::is_same_v<IndexT, uint32_t>) return TensorDType::U32;
    return TensorDType::U64;
}

template <typename IndexT>
static bool linearize_coords_to_tensor(const CoordBuffer& coord_buf,
                                       uint32_t count,
                                       uint32_t dims,
                                       const uint32_t* shape,
                                       AbstractTensor& out,
                                       TensorBackend* backend,
                                       TensorDType out_dtype = TensorDType::Unknown) {
    if (!backend || !coord_buf.coords_ptr || !shape || dims == 0) return false;
    const uint64_t index_range = compute_index_range_u64(shape, dims);
    if (out_dtype == TensorDType::Unknown) {
        out_dtype = pick_unsigned_index_dtype(index_range);
    }
    if (!out.valid()) {
        TensorDesc desc{};
        desc.dtype = out_dtype;
        desc.layout = TensorLayout::Dense;
        desc.shape.dims = {count};
        out = AbstractTensor::create(desc, backend);
        if (!out.valid()) return false;
    } else {
        if (out.desc().dtype != unsigned_index_dtype<IndexT>()) return false;
    }

    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem) return false;
    void* out_ptr = nullptr;
    size_t out_bytes = 0;
    if (!mem->map(out.handle(), &out_ptr, &out_bytes)) return false;

    std::vector<uint64_t> strides;
    if (!compute_dense_strides_u64(std::vector<uint32_t>(shape, shape + dims), strides)) {
        mem->unmap(out.handle());
        return false;
    }

    IndexT* out_idx = static_cast<IndexT*>(out_ptr);
    const int64_t* coords = coord_buf.coords_ptr;
    const uint8_t* in_bounds = coord_buf.mask_ptr;
    for (uint32_t i = 0; i < count; ++i) {
        if (in_bounds && !in_bounds[i]) {
            out_idx[i] = static_cast<IndexT>(0);
            continue;
        }
        const int64_t* coord = coords + static_cast<size_t>(i) * dims;
        uint64_t linear = 0;
        for (uint32_t d = 0; d < dims; ++d) {
            linear += static_cast<uint64_t>(coord[d]) * strides[d];
        }
        out_idx[i] = static_cast<IndexT>(linear);
    }

    mem->unmap(out.handle());
    return true;
}

template <typename IndexT>
static bool delinearize_tensor_to_coords(const AbstractTensor& linear,
                                         CoordBuffer& coord_buf,
                                         uint32_t count,
                                         uint32_t dims,
                                         const uint32_t* shape) {
    if (!linear.valid() || !coord_buf.coords_ptr || !shape || dims == 0) return false;
    if (linear.desc().dtype != unsigned_index_dtype<IndexT>()) return false;
    auto* mem = dynamic_cast<InMemoryBackend*>(linear.backend());
    if (!mem) return false;
    void* in_ptr = nullptr;
    size_t in_bytes = 0;
    if (!mem->map(linear.handle(), &in_ptr, &in_bytes)) return false;

    const IndexT* in_idx = static_cast<const IndexT*>(in_ptr);
    const uint64_t index_range = compute_index_range_u64(shape, dims);
    int64_t* coords = coord_buf.coords_ptr;
    uint8_t* in_bounds = coord_buf.mask_ptr;

    std::vector<uint64_t> strides;
    if (!compute_dense_strides_u64(std::vector<uint32_t>(shape, shape + dims), strides)) {
        mem->unmap(linear.handle());
        return false;
    }

    for (uint32_t i = 0; i < count; ++i) {
        uint64_t idx = static_cast<uint64_t>(in_idx[i]);
        if (idx >= index_range) {
            if (in_bounds) in_bounds[i] = 0;
            for (uint32_t d = 0; d < dims; ++d) {
                coords[static_cast<size_t>(i) * dims + d] = 0;
            }
            continue;
        }
        if (in_bounds) in_bounds[i] = 1;
        for (uint32_t d = 0; d < dims; ++d) {
            const uint64_t stride = strides[d];
            const uint64_t v = idx / stride;
            idx -= v * stride;
            coords[static_cast<size_t>(i) * dims + d] = static_cast<int64_t>(v);
        }
    }

    mem->unmap(linear.handle());
    return true;
}

template <typename IndexT, typename Scalar, typename PremixPol, typename OutmixPol>
static bool dyadic_scatter_from_coords(const CoordBuffer& coord_buf,
                                       uint32_t count,
                                       uint32_t dims,
                                       const uint32_t* shape,
                                       uint32_t channels,
                                       bool val_scalar,
                                       const Scalar* values,
                                       AbstractTensor& output,
                                       TensorBackend* backend) {
    if (!backend || !values || !output.valid() || count == 0 || dims == 0) return false;
    const uint64_t spatial_range = compute_index_range_u64(shape, dims);
    const uint64_t index_range = spatial_range * static_cast<uint64_t>(channels);

    DYADIC_SCATTER_LOGF("dyadic_scatter: count=%u dims=%u channels=%u val_scalar=%d index_range=%llu\n",
                        count, dims, channels, val_scalar ? 1 : 0, (unsigned long long)index_range);

    AbstractTensor linear_indices;
    if (!linearize_coords_to_tensor<IndexT>(coord_buf,
                                            count,
                                            dims,
                                            shape,
                                            linear_indices,
                                            backend,
                                            unsigned_index_dtype<IndexT>())) {
        DYADIC_SCATTER_LOGF("dyadic_scatter: linearize failed\n");
        return false;
    }

    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem) {
        DYADIC_SCATTER_LOGF("dyadic_scatter: backend not in-memory\n");
        return false;
    }
    void* idx_ptr = nullptr;
    size_t idx_bytes = 0;
    if (!mem->map(linear_indices.handle(), &idx_ptr, &idx_bytes)) {
        DYADIC_SCATTER_LOGF("dyadic_scatter: map linear_indices failed\n");
        return false;
    }
    IndexT* idx_data = static_cast<IndexT*>(idx_ptr);
    const uint8_t* in_bounds = coord_buf.mask_ptr;

    if (channels == 1) {
        AbstractTensor masked_values;
        Scalar* values_ptr = const_cast<Scalar*>(values);
        void* mv_ptr = nullptr;
        size_t mv_bytes = 0;
        if (in_bounds) {
            TensorDesc desc_values{};
            desc_values.dtype = dyadic_value_dtype<Scalar>();
            desc_values.layout = TensorLayout::Dense;
            desc_values.shape.dims = {count};
            masked_values = scatter_pool().acquire_tensor(desc_values, backend);
            if (!masked_values.valid()) {
                DYADIC_SCATTER_LOGF("dyadic_scatter: masked_values alloc failed\n");
                mem->unmap(linear_indices.handle());
                return false;
            }
            if (!mem->map(masked_values.handle(), &mv_ptr, &mv_bytes)) {
                DYADIC_SCATTER_LOGF("dyadic_scatter: map masked_values failed\n");
                mem->unmap(linear_indices.handle());
                return false;
            }
            auto* mv = static_cast<Scalar*>(mv_ptr);
            for (uint32_t i = 0; i < count; ++i) {
                mv[i] = in_bounds[i] ? values[i] : Scalar{};
            }
            values_ptr = mv;
        }

        const bool ok = dyadic_binning_tensor<PremixPol, OutmixPol>(
            static_cast<IndexT>(index_range),
            count,
            idx_data,
            values_ptr,
            8u,
            1u,
            1u,
            1u,
            1u,
            output,
            scatter_pool());
        DYADIC_SCATTER_LOGF("dyadic_scatter: binning (channels=1) ok=%d\n", ok ? 1 : 0);
        if (masked_values.valid()) {
            mem->unmap(masked_values.handle());
        }
        mem->unmap(linear_indices.handle());
        return ok;
    }

    const uint64_t expanded_count = static_cast<uint64_t>(count) * channels;
    TensorDesc desc_indices{};
    desc_indices.dtype = unsigned_index_dtype<IndexT>();
    desc_indices.layout = TensorLayout::Dense;
    desc_indices.shape.dims = {static_cast<uint32_t>(expanded_count)};
    TensorDesc desc_values{};
    desc_values.dtype = dyadic_value_dtype<Scalar>();
    desc_values.layout = TensorLayout::Dense;
    desc_values.shape.dims = {static_cast<uint32_t>(expanded_count)};

    AbstractTensor expanded_indices = scatter_pool().acquire_tensor(desc_indices, backend);
    AbstractTensor expanded_values = scatter_pool().acquire_tensor(desc_values, backend);
    if (!expanded_indices.valid() || !expanded_values.valid()) {
        DYADIC_SCATTER_LOGF("dyadic_scatter: expanded buffers alloc failed\n");
        mem->unmap(linear_indices.handle());
        return false;
    }

    void* ex_idx_ptr = nullptr;
    size_t ex_idx_bytes = 0;
    if (!mem->map(expanded_indices.handle(), &ex_idx_ptr, &ex_idx_bytes)) {
        DYADIC_SCATTER_LOGF("dyadic_scatter: map expanded_indices failed\n");
        mem->unmap(linear_indices.handle());
        return false;
    }
    void* ex_val_ptr = nullptr;
    size_t ex_val_bytes = 0;
    if (!mem->map(expanded_values.handle(), &ex_val_ptr, &ex_val_bytes)) {
        DYADIC_SCATTER_LOGF("dyadic_scatter: map expanded_values failed\n");
        mem->unmap(expanded_indices.handle());
        mem->unmap(linear_indices.handle());
        return false;
    }

    auto* ex_idx = static_cast<IndexT*>(ex_idx_ptr);
    auto* ex_val = static_cast<Scalar*>(ex_val_ptr);
    uint64_t out_i = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t base = static_cast<uint64_t>(idx_data[i]) * channels;
        const bool inb = !in_bounds || in_bounds[i];
        if (val_scalar) {
            const Scalar v = inb ? values[i] : Scalar{};
            for (uint32_t c = 0; c < channels; ++c) {
                ex_idx[out_i] = static_cast<IndexT>(base + c);
                ex_val[out_i] = v;
                out_i += 1u;
            }
        } else {
            const Scalar* src = values + static_cast<uint64_t>(i) * channels;
            for (uint32_t c = 0; c < channels; ++c) {
                ex_idx[out_i] = static_cast<IndexT>(base + c);
                ex_val[out_i] = inb ? src[c] : Scalar{};
                out_i += 1u;
            }
        }
    }

    const bool ok = dyadic_binning_tensor<PremixPol, OutmixPol>(
        static_cast<IndexT>(index_range),
        static_cast<uint32_t>(expanded_count),
        ex_idx,
        ex_val,
        8u,
        1u,
        1u,
        1u,
        1u,
        output,
        scatter_pool());
    DYADIC_SCATTER_LOGF("dyadic_scatter: binning (channels>1) ok=%d\n", ok ? 1 : 0);
    mem->unmap(expanded_values.handle());
    mem->unmap(expanded_indices.handle());
    mem->unmap(linear_indices.handle());
    return ok;
}

template <typename Scalar, typename PremixPol, typename OutmixPol>
static bool dyadic_scatter_from_coords_auto(const CoordBuffer& coord_buf,
                                            uint32_t count,
                                            uint32_t dims,
                                            const uint32_t* shape,
                                            uint32_t channels,
                                            bool val_scalar,
                                            const Scalar* values,
                                            AbstractTensor& output,
                                            TensorBackend* backend) {
    const uint64_t spatial_range = compute_index_range_u64(shape, dims);
    const uint64_t index_range = spatial_range * static_cast<uint64_t>(channels);
    switch (pick_unsigned_index_dtype(index_range)) {
        case TensorDType::U8:
            return dyadic_scatter_from_coords<uint8_t, Scalar, PremixPol, OutmixPol>(
                coord_buf, count, dims, shape, channels, val_scalar, values, output, backend);
        case TensorDType::U16:
            return dyadic_scatter_from_coords<uint16_t, Scalar, PremixPol, OutmixPol>(
                coord_buf, count, dims, shape, channels, val_scalar, values, output, backend);
        case TensorDType::U32:
            return dyadic_scatter_from_coords<uint32_t, Scalar, PremixPol, OutmixPol>(
                coord_buf, count, dims, shape, channels, val_scalar, values, output, backend);
        default:
            return dyadic_scatter_from_coords<uint64_t, Scalar, PremixPol, OutmixPol>(
                coord_buf, count, dims, shape, channels, val_scalar, values, output, backend);
    }
}

static bool acquire_coord_buffer(uint32_t count,
                                 uint32_t dims,
                                 TensorBackend* backend,
                                 CoordBuffer& out) {
    if (!backend || count == 0 || dims == 0) return false;
    TensorDesc cdesc{};
    cdesc.dtype = TensorDType::I64;
    cdesc.layout = TensorLayout::Dense;
    if (dims == 1) {
        cdesc.shape.dims = {count};
    } else {
        cdesc.shape.dims = {count, dims};
    }

    TensorDesc mdesc{};
    mdesc.dtype = TensorDType::Bool;
    mdesc.layout = TensorLayout::Dense;
    mdesc.shape.dims = {count};

    out.coords = coord_pool().acquire(cdesc, backend);
    out.mask = coord_pool().acquire(mdesc, backend);
    if (!out.coords.valid() || !out.mask.valid()) return false;

    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem) return false;
    void* cptr = nullptr;
    size_t cbytes = 0;
    if (!mem->map(out.coords.tensor().handle(), &cptr, &cbytes)) return false;
    void* mptr = nullptr;
    size_t mbytes = 0;
    if (!mem->map(out.mask.tensor().handle(), &mptr, &mbytes)) {
        mem->unmap(out.coords.tensor().handle());
        return false;
    }
    out.coords_ptr = static_cast<int64_t*>(cptr);
    out.mask_ptr = static_cast<uint8_t*>(mptr);
    return true;
}

static void release_coord_buffer(TensorBackend* backend, CoordBuffer& buf) {
    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem) return;
    if (buf.coords.valid()) mem->unmap(buf.coords.tensor().handle());
    if (buf.mask.valid()) mem->unmap(buf.mask.tensor().handle());
    buf.coords_ptr = nullptr;
    buf.mask_ptr = nullptr;
}

struct TensorOpTensor {
    void* base = nullptr;
    std::vector<uint64_t> byte_strides;
    uint32_t elem_bytes = 0;
    bool contiguous = false;
};

struct TensorOpPlan {
    uint32_t rank = 0;
    std::vector<uint64_t> shape;
    std::vector<uint64_t> outer_shape;
    std::vector<uint64_t> outer_strides;
    TensorOpTensor a;
    TensorOpTensor b;
    TensorOpTensor out;
    uint64_t inner_count = 0;
    uint64_t outer_count = 0;
};

static bool build_unary_plan(const AbstractTensor& src,
                             const AbstractTensor& dst,
                             TensorOpPlan& plan) {
    if (!src.valid() || !dst.valid()) return false;
    if (src.backend() != dst.backend()) return false;
    const TensorDesc& sd = src.desc();
    const TensorDesc& dd = dst.desc();
    if (sd.dtype != TensorDType::F32 || dd.dtype != TensorDType::F32) return false;
    if (!((sd.layout == TensorLayout::Dense || sd.layout == TensorLayout::Strided) &&
          (dd.layout == TensorLayout::Dense || dd.layout == TensorLayout::Strided))) {
        return false;
    }
    if (sd.shape.dims != dd.shape.dims) return false;

    plan.shape.assign(sd.shape.dims.begin(), sd.shape.dims.end());
    plan.rank = static_cast<uint32_t>(plan.shape.size());

    auto* mem = dynamic_cast<InMemoryBackend*>(src.backend());
    if (!mem) return false;
    void* sp = nullptr;
    void* dp = nullptr;
    size_t sb = 0;
    size_t db = 0;
    if (!mem->map(src.handle(), &sp, &sb)) return false;
    if (!mem->map(dst.handle(), &dp, &db)) {
        mem->unmap(src.handle());
        return false;
    }

    if (!build_op_tensor(sd, plan.shape, sp, plan.a) ||
        !build_op_tensor(dd, plan.shape, dp, plan.out)) {
        mem->unmap(dst.handle());
        mem->unmap(src.handle());
        return false;
    }

    plan.inner_count = plan.shape.empty() ? 1u : plan.shape.back();
    plan.outer_count = 1u;
    plan.outer_shape.clear();
    plan.outer_strides.clear();
    if (plan.shape.size() > 1) {
        for (size_t i = 0; i + 1 < plan.shape.size(); ++i) {
            plan.outer_count *= plan.shape[i];
            plan.outer_shape.push_back(plan.shape[i]);
        }
        if (!compute_dense_strides_u64(
                std::vector<uint32_t>(plan.outer_shape.begin(), plan.outer_shape.end()), plan.outer_strides)) {
            return false;
        }
    } else {
        plan.outer_count = 1u;
    }

    return true;
}

static void copy_range(const TensorOpPlan& plan, uint64_t outer_begin, uint64_t outer_end) {
    const uint64_t inner = plan.inner_count;
    const size_t rank = plan.shape.size();
    if (rank == 0) {
        auto* s = reinterpret_cast<float*>(plan.a.base);
        auto* d = reinterpret_cast<float*>(plan.out.base);
        *d = *s;
        return;
    }

    auto* s0 = static_cast<uint8_t*>(plan.a.base);
    auto* d0 = static_cast<uint8_t*>(plan.out.base);

    for (uint64_t outer_idx = outer_begin; outer_idx < outer_end; ++outer_idx) {
        uint64_t s_off = 0;
        uint64_t d_off = 0;
        if (rank > 1) {
            uint64_t tmp = outer_idx;
            for (size_t d = 0; d < plan.outer_shape.size(); ++d) {
                const uint64_t stride = plan.outer_strides[d];
                const uint64_t coord = (stride == 0) ? 0 : (tmp / stride);
                tmp = (stride == 0) ? tmp : (tmp % stride);
                s_off += coord * plan.a.byte_strides[d];
                d_off += coord * plan.out.byte_strides[d];
            }
        }

        auto* s_ptr = reinterpret_cast<float*>(s0 + s_off);
        auto* d_ptr = reinterpret_cast<float*>(d0 + d_off);
        const uint64_t s_step = plan.a.byte_strides.back() / plan.a.elem_bytes;
        const uint64_t d_step = plan.out.byte_strides.back() / plan.out.elem_bytes;

        if (plan.a.contiguous && plan.out.contiguous && s_step == 1 && d_step == 1) {
            std::memcpy(d_ptr, s_ptr, static_cast<size_t>(inner * sizeof(float)));
        } else {
            uint64_t si = 0;
            uint64_t di = 0;
            for (uint64_t i = 0; i < inner; ++i) {
                d_ptr[di] = s_ptr[si];
                si += s_step;
                di += d_step;
            }
        }
    }
}

static inline bool should_parallelize(nodus::ThreadPool* pool,
                                      uint64_t work_items,
                                      uint32_t min_per_thread = 2) {
    if (!pool) return false;
    const uint32_t max_jobs = pool->thread_count();
    if (max_jobs < 2) return false;
    return work_items >= static_cast<uint64_t>(max_jobs) * static_cast<uint64_t>(min_per_thread);
}

static void execute_copy_plan(const TensorOpPlan& plan) {
    const uint64_t outer = plan.outer_count;
    if (outer == 0 || plan.inner_count == 0) return;

    nodus::ThreadPool* pool = tensor_op_pool();
    if (!pool) {
        copy_range(plan, 0, outer);
        return;
    }

    if (!should_parallelize(pool, outer)) {
        copy_range(plan, 0, outer);
        return;
    }

    const uint32_t max_jobs = pool->thread_count();
    const uint32_t job_count = static_cast<uint32_t>(std::min<uint64_t>(outer, max_jobs));
    if (job_count <= 1) {
        copy_range(plan, 0, outer);
        return;
    }

    struct CopyJobParams {
        const TensorOpPlan* plan = nullptr;
        uint64_t begin = 0;
        uint64_t end = 0;
    };

    auto job_fn = [](const nodus::ThreadPool::Job& job, uint32_t /*worker_id*/) {
        const auto* p = static_cast<const CopyJobParams*>(job.params);
        copy_range(*p->plan, p->begin, p->end);
    };

    std::vector<CopyJobParams> params(job_count);
    std::vector<nodus::ThreadPool::Job> jobs(job_count);
    const uint64_t chunk = (outer + job_count - 1) / job_count;
    for (uint32_t i = 0; i < job_count; ++i) {
        const uint64_t begin = static_cast<uint64_t>(i) * chunk;
        const uint64_t end = std::min<uint64_t>(outer, begin + chunk);
        params[i] = CopyJobParams{&plan, begin, end};
        nodus::ThreadPool::Job job{};
        job.fn = job_fn;
        job.params = &params[i];
        job.params_size = static_cast<uint32_t>(sizeof(CopyJobParams));
        jobs[i] = job;
    }

    auto batch = pool->submit_batch(jobs.data(), job_count);
    if (batch) batch->wait();
}

static bool compute_broadcast_shape(const TensorDesc& a,
                                    const TensorDesc& b,
                                    std::vector<uint64_t>& out_shape) {
    const size_t ar = a.shape.dims.size();
    const size_t br = b.shape.dims.size();
    const size_t rank = std::max(ar, br);
    out_shape.assign(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const size_t ai = (i < rank - ar) ? static_cast<size_t>(-1) : (i - (rank - ar));
        const size_t bi = (i < rank - br) ? static_cast<size_t>(-1) : (i - (rank - br));
        const uint64_t ad = (ai == static_cast<size_t>(-1)) ? 1u : a.shape.dims[ai];
        const uint64_t bd = (bi == static_cast<size_t>(-1)) ? 1u : b.shape.dims[bi];
        if (ad != bd && ad != 1 && bd != 1) return false;
        out_shape[i] = std::max(ad, bd);
    }
    return true;
}

static bool build_op_tensor(const TensorDesc& desc,
                            const std::vector<uint64_t>& out_shape,
                            void* base,
                            TensorOpTensor& out) {
    const size_t rank = out_shape.size();
    std::vector<uint64_t> elem_strides;
    if (!compute_strides_for_desc_u64(desc, elem_strides)) return false;
    out.elem_bytes = tensor_dtype_size_bytes(desc.dtype);
    if (out.elem_bytes == 0) return false;
    out.byte_strides.assign(rank, 0);
    out.base = base;

    const size_t dr = desc.shape.dims.size();
    const size_t pad = (rank >= dr) ? (rank - dr) : 0;
    for (size_t i = 0; i < rank; ++i) {
        if (i < pad) {
            out.byte_strides[i] = 0;
            continue;
        }
        const size_t di = i - pad;
        const uint64_t dim = desc.shape.dims[di];
        const uint64_t stride = elem_strides[di];
        out.byte_strides[i] = (dim == 1 && out_shape[i] > 1) ? 0 : stride * out.elem_bytes;
    }

    // Contiguous if dense and no broadcast in trailing dimension.
    out.contiguous = (desc.layout == TensorLayout::Dense);
    return true;
}

static bool build_axpby_plan(const AbstractTensor& a,
                             const AbstractTensor& b,
                             const AbstractTensor& out,
                             TensorOpPlan& plan) {
    if (!a.valid() || !b.valid() || !out.valid()) return false;
    if (a.backend() != b.backend() || a.backend() != out.backend()) return false;

    const TensorDesc& ad = a.desc();
    const TensorDesc& bd = b.desc();
    const TensorDesc& od = out.desc();
    if (ad.dtype != TensorDType::F32 || bd.dtype != TensorDType::F32 || od.dtype != TensorDType::F32) return false;
    if (!((ad.layout == TensorLayout::Dense || ad.layout == TensorLayout::Strided) &&
          (bd.layout == TensorLayout::Dense || bd.layout == TensorLayout::Strided) &&
          (od.layout == TensorLayout::Dense || od.layout == TensorLayout::Strided))) {
        return false;
    }

    if (!compute_broadcast_shape(ad, bd, plan.shape)) return false;
    plan.rank = static_cast<uint32_t>(plan.shape.size());

    // Out must match broadcast shape (no implicit output broadcast).
    if (od.shape.dims.size() != plan.shape.size()) return false;
    for (size_t i = 0; i < plan.shape.size(); ++i) {
        if (od.shape.dims[i] != plan.shape[i]) return false;
    }

    auto* mem = dynamic_cast<InMemoryBackend*>(a.backend());
    if (!mem) return false;

    void* ap = nullptr;
    void* bp = nullptr;
    void* op = nullptr;
    size_t ab = 0;
    size_t bb = 0;
    size_t ob = 0;
    if (!mem->map(a.handle(), &ap, &ab)) return false;
    if (!mem->map(b.handle(), &bp, &bb)) {
        mem->unmap(a.handle());
        return false;
    }
    if (!mem->map(out.handle(), &op, &ob)) {
        mem->unmap(a.handle());
        mem->unmap(b.handle());
        return false;
    }

    if (!build_op_tensor(ad, plan.shape, ap, plan.a) ||
        !build_op_tensor(bd, plan.shape, bp, plan.b) ||
        !build_op_tensor(od, plan.shape, op, plan.out)) {
        mem->unmap(out.handle());
        mem->unmap(b.handle());
        mem->unmap(a.handle());
        return false;
    }

    plan.inner_count = plan.shape.empty() ? 1u : plan.shape.back();
    plan.outer_count = 1u;
    plan.outer_shape.clear();
    plan.outer_strides.clear();
    if (plan.shape.size() > 1) {
        for (size_t i = 0; i + 1 < plan.shape.size(); ++i) {
            plan.outer_count *= plan.shape[i];
            plan.outer_shape.push_back(plan.shape[i]);
        }
        if (!compute_dense_strides_u64(
                std::vector<uint32_t>(plan.outer_shape.begin(), plan.outer_shape.end()), plan.outer_strides)) {
            return false;
        }
    } else {
        plan.outer_count = 1u;
    }

    return true;
}

static void axpby_range(const TensorOpPlan& plan,
                        float alpha,
                        float beta,
                        uint64_t outer_begin,
                        uint64_t outer_end) {
    const uint64_t inner = plan.inner_count;
    const size_t rank = plan.shape.size();

    if (rank == 0) {
        auto* a_ptr = reinterpret_cast<float*>(plan.a.base);
        auto* b_ptr = reinterpret_cast<float*>(plan.b.base);
        auto* o_ptr = reinterpret_cast<float*>(plan.out.base);
        *o_ptr = alpha * (*a_ptr) + beta * (*b_ptr);
        return;
    }

    const bool all_contig = plan.a.contiguous && plan.b.contiguous && plan.out.contiguous &&
                            plan.a.byte_strides.back() == plan.a.elem_bytes &&
                            plan.b.byte_strides.back() == plan.b.elem_bytes &&
                            plan.out.byte_strides.back() == plan.out.elem_bytes;

    auto* a0 = static_cast<uint8_t*>(plan.a.base);
    auto* b0 = static_cast<uint8_t*>(plan.b.base);
    auto* o0 = static_cast<uint8_t*>(plan.out.base);

    for (uint64_t outer_idx = outer_begin; outer_idx < outer_end; ++outer_idx) {
        uint64_t a_off = 0;
        uint64_t b_off = 0;
        uint64_t o_off = 0;
        if (rank > 1) {
            uint64_t tmp = outer_idx;
            for (size_t d = 0; d < plan.outer_shape.size(); ++d) {
                const uint64_t stride = plan.outer_strides[d];
                const uint64_t coord = (stride == 0) ? 0 : (tmp / stride);
                tmp = (stride == 0) ? tmp : (tmp % stride);
                a_off += coord * plan.a.byte_strides[d];
                b_off += coord * plan.b.byte_strides[d];
                o_off += coord * plan.out.byte_strides[d];
            }
        }

        auto* a_ptr = reinterpret_cast<float*>(a0 + a_off);
        auto* b_ptr = reinterpret_cast<float*>(b0 + b_off);
        auto* o_ptr = reinterpret_cast<float*>(o0 + o_off);

        const uint64_t a_step = plan.a.byte_strides.back() / plan.a.elem_bytes;
        const uint64_t b_step = plan.b.byte_strides.back() / plan.b.elem_bytes;
        const uint64_t o_step = plan.out.byte_strides.back() / plan.out.elem_bytes;

        if (all_contig && a_step == 1 && b_step == 1 && o_step == 1) {
            for (uint64_t i = 0; i < inner; ++i) {
                o_ptr[i] = alpha * a_ptr[i] + beta * b_ptr[i];
            }
        } else {
            uint64_t ai = 0;
            uint64_t bi = 0;
            uint64_t oi = 0;
            for (uint64_t i = 0; i < inner; ++i) {
                o_ptr[oi] = alpha * a_ptr[ai] + beta * b_ptr[bi];
                ai += a_step;
                bi += b_step;
                oi += o_step;
            }
        }

    }
}

static void execute_axpby_plan(const TensorOpPlan& plan, float alpha, float beta) {
    const uint64_t outer = plan.outer_count;
    const uint64_t inner = plan.inner_count;
    if (outer == 0 || inner == 0) return;

    nodus::ThreadPool* pool = tensor_op_pool();
    if (!pool) {
        axpby_range(plan, alpha, beta, 0, outer);
        return;
    }

    if (!should_parallelize(pool, outer)) {
        axpby_range(plan, alpha, beta, 0, outer);
        return;
    }

    const uint32_t max_jobs = pool->thread_count();
    const uint32_t job_count = static_cast<uint32_t>(std::min<uint64_t>(outer, max_jobs));
    if (job_count <= 1) {
        axpby_range(plan, alpha, beta, 0, outer);
        return;
    }

    struct AxpbyJobParams {
        const TensorOpPlan* plan = nullptr;
        float alpha = 0.0f;
        float beta = 0.0f;
        uint64_t begin = 0;
        uint64_t end = 0;
    };

    auto job_fn = [](const nodus::ThreadPool::Job& job, uint32_t /*worker_id*/) {
        const auto* p = static_cast<const AxpbyJobParams*>(job.params);
        axpby_range(*p->plan, p->alpha, p->beta, p->begin, p->end);
    };

    std::vector<AxpbyJobParams> params(job_count);
    std::vector<nodus::ThreadPool::Job> jobs(job_count);
    const uint64_t chunk = (outer + job_count - 1) / job_count;
    for (uint32_t i = 0; i < job_count; ++i) {
        const uint64_t begin = static_cast<uint64_t>(i) * chunk;
        const uint64_t end = std::min<uint64_t>(outer, begin + chunk);
        params[i] = AxpbyJobParams{&plan, alpha, beta, begin, end};
        nodus::ThreadPool::Job job{};
        job.fn = job_fn;
        job.params = &params[i];
        job.params_size = static_cast<uint32_t>(sizeof(AxpbyJobParams));
        jobs[i] = job;
    }

    auto batch = pool->submit_batch(jobs.data(), job_count);
    if (batch) batch->wait();
}

template <typename Scalar, TensorDType DType>
struct TensorMathImpl {
    static constexpr Scalar kQuatEps = static_cast<Scalar>(1e-8);
    static constexpr Scalar kPlaneEps = static_cast<Scalar>(1e-8);

    struct MappedDense {
        InMemoryBackend* backend = nullptr;
        AbstractTensorHandle handle{};
        Scalar* data = nullptr;
        uint64_t elems = 0;
        bool ok = false;

        void unmap() {
            if (backend && data) backend->unmap(handle);
            backend = nullptr;
            data = nullptr;
            elems = 0;
            ok = false;
        }
    };

    struct MappedDenseI32 {
        InMemoryBackend* backend = nullptr;
        AbstractTensorHandle handle{};
        int32_t* data = nullptr;
        uint64_t elems = 0;
        bool ok = false;

        void unmap() {
            if (backend && data) backend->unmap(handle);
            backend = nullptr;
            data = nullptr;
            elems = 0;
            ok = false;
        }
    };

    struct MappedDenseU8 {
        InMemoryBackend* backend = nullptr;
        AbstractTensorHandle handle{};
        uint8_t* data = nullptr;
        uint64_t elems = 0;
        bool ok = false;

        void unmap() {
            if (backend && data) backend->unmap(handle);
            backend = nullptr;
            data = nullptr;
            elems = 0;
            ok = false;
        }
    };

    static MappedDense map_dense(const AbstractTensor& t) {
        MappedDense out{};
        if (!t.valid()) return out;
        const TensorDesc& d = t.desc();
        if (d.dtype != DType || d.layout != TensorLayout::Dense) return out;
        auto* mem = dynamic_cast<InMemoryBackend*>(t.backend());
        if (!mem) return out;
        void* ptr = nullptr;
        size_t bytes = 0;
        if (!mem->map(t.handle(), &ptr, &bytes)) return out;
        out.backend = mem;
        out.handle = t.handle();
        out.data = static_cast<Scalar*>(ptr);
        out.elems = d.shape.element_count();
        out.ok = true;
        return out;
    }

    static MappedDenseI32 map_dense_i32(const AbstractTensor& t) {
        MappedDenseI32 out{};
        if (!t.valid()) return out;
        const TensorDesc& d = t.desc();
        if (d.dtype != TensorDType::I32 || d.layout != TensorLayout::Dense) return out;
        auto* mem = dynamic_cast<InMemoryBackend*>(t.backend());
        if (!mem) return out;
        void* ptr = nullptr;
        size_t bytes = 0;
        if (!mem->map(t.handle(), &ptr, &bytes)) return out;
        out.backend = mem;
        out.handle = t.handle();
        out.data = static_cast<int32_t*>(ptr);
        out.elems = d.shape.element_count();
        out.ok = true;
        return out;
    }

    static MappedDenseU8 map_dense_u8(const AbstractTensor& t) {
        MappedDenseU8 out{};
        if (!t.valid()) return out;
        const TensorDesc& d = t.desc();
        if (d.dtype != TensorDType::Bool || d.layout != TensorLayout::Dense) return out;
        auto* mem = dynamic_cast<InMemoryBackend*>(t.backend());
        if (!mem) return out;
        void* ptr = nullptr;
        size_t bytes = 0;
        if (!mem->map(t.handle(), &ptr, &bytes)) return out;
        out.backend = mem;
        out.handle = t.handle();
        out.data = static_cast<uint8_t*>(ptr);
        out.elems = d.shape.element_count();
        out.ok = true;
        return out;
    }

    static MappedDense map_dense_mut(const AbstractTensor& t) { return map_dense(t); }

    static bool shape_is(const TensorDesc& desc, std::initializer_list<uint32_t> dims) {
        if (desc.shape.dims.size() != dims.size()) return false;
        size_t i = 0;
        for (uint32_t d : dims) {
            if (desc.shape.dims[i++] != d) return false;
        }
        return true;
    }

    static bool read_vec3(const AbstractTensor& t, Scalar* out3) {
        if (!out3 || !t.valid()) return false;
        const TensorDesc& d = t.desc();
        if (d.dtype != DType || d.layout != TensorLayout::Dense) return false;
        if (!(shape_is(d, {3}) || shape_is(d, {1, 3}) ||
              (d.shape.dims.size() == 2 && d.shape.dims[1] == 3))) {
            return false;
        }
        MappedDense map = map_dense(t);
        if (!map.ok || map.elems < 3) {
            map.unmap();
            return false;
        }
        out3[0] = map.data[0];
        out3[1] = map.data[1];
        out3[2] = map.data[2];
        map.unmap();
        return true;
    }

    static void quat_to_mat4_row(Scalar w, Scalar x, Scalar y, Scalar z, const Scalar* t3, Scalar* out16) {
        const Scalar xx = x * x;
        const Scalar yy = y * y;
        const Scalar zz = z * z;
        const Scalar xy = x * y;
        const Scalar xz = x * z;
        const Scalar yz = y * z;
        const Scalar wx = w * x;
        const Scalar wy = w * y;
        const Scalar wz = w * z;

        out16[0] = static_cast<Scalar>(1) - static_cast<Scalar>(2) * (yy + zz);
        out16[1] = static_cast<Scalar>(2) * (xy - wz);
        out16[2] = static_cast<Scalar>(2) * (xz + wy);
        out16[3] = static_cast<Scalar>(0);

        out16[4] = static_cast<Scalar>(2) * (xy + wz);
        out16[5] = static_cast<Scalar>(1) - static_cast<Scalar>(2) * (xx + zz);
        out16[6] = static_cast<Scalar>(2) * (yz - wx);
        out16[7] = static_cast<Scalar>(0);

        out16[8] = static_cast<Scalar>(2) * (xz - wy);
        out16[9] = static_cast<Scalar>(2) * (yz + wx);
        out16[10] = static_cast<Scalar>(1) - static_cast<Scalar>(2) * (xx + yy);
        out16[11] = static_cast<Scalar>(0);

        out16[12] = t3 ? t3[0] : static_cast<Scalar>(0);
        out16[13] = t3 ? t3[1] : static_cast<Scalar>(0);
        out16[14] = t3 ? t3[2] : static_cast<Scalar>(0);
        out16[15] = static_cast<Scalar>(1);
    }

    static AbstractTensor matmul(const AbstractTensor& a, const AbstractTensor& b) {
        if (!a.valid() || !b.valid()) return {};
        if (a.backend() != b.backend()) return {};
        const TensorDesc& ad = a.desc();
        const TensorDesc& bd = b.desc();
        if (ad.dtype != DType || bd.dtype != DType) return {};
        if (ad.layout != TensorLayout::Dense || bd.layout != TensorLayout::Dense) return {};
        if (ad.shape.dims.size() != 2 || bd.shape.dims.size() != 2) return {};
        const uint32_t m = ad.shape.dims[0];
        const uint32_t n = ad.shape.dims[1];
        const uint32_t n2 = bd.shape.dims[0];
        const uint32_t k = bd.shape.dims[1];
        if (n == 0 || n2 == 0 || m == 0 || k == 0) return {};
        if (n != n2) return {};

        TensorDesc out_desc{};
        out_desc.dtype = DType;
        out_desc.layout = TensorLayout::Dense;
        out_desc.shape.dims = {m, k};
        AbstractTensor out = AbstractTensor::create(out_desc, a.backend());
        if (!out.valid()) return {};

        MappedDense amap = map_dense(a);
        MappedDense bmap = map_dense(b);
        MappedDense omap = map_dense_mut(out);
        if (!amap.ok || !bmap.ok || !omap.ok) {
            amap.unmap();
            bmap.unmap();
            omap.unmap();
            return {};
        }

        const Scalar* A = amap.data;
        const Scalar* B = bmap.data;
        Scalar* C = omap.data;

        struct MatmulCtx {
            const Scalar* A = nullptr;
            const Scalar* B = nullptr;
            Scalar* C = nullptr;
            uint32_t m = 0;
            uint32_t n = 0;
            uint32_t k = 0;
        };

        auto matmul_rows = [](const void* vctx, uint32_t r0, uint32_t r1) {
            const auto* ctx = static_cast<const MatmulCtx*>(vctx);
            const Scalar* A = ctx->A;
            const Scalar* B = ctx->B;
            Scalar* C = ctx->C;
            const uint32_t n = ctx->n;
            const uint32_t k = ctx->k;
            for (uint32_t r = r0; r < r1; ++r) {
                for (uint32_t c = 0; c < k; ++c) {
                    Scalar acc = static_cast<Scalar>(0);
                    for (uint32_t i = 0; i < n; ++i) {
                        acc += A[r * n + i] * B[i * k + c];
                    }
                    C[r * k + c] = acc;
                }
            }
        };

        MatmulCtx ctx{};
        ctx.A = A;
        ctx.B = B;
        ctx.C = C;
        ctx.m = m;
        ctx.n = n;
        ctx.k = k;

        submit_row_jobs(tensor_op_pool(), matmul_rows, &ctx, 0, m);

        amap.unmap();
        bmap.unmap();
        omap.unmap();
        return out;
    }

    static AbstractTensor affine_identity(TensorBackend* backend) {
        TensorDesc desc{};
        desc.dtype = DType;
        desc.layout = TensorLayout::Dense;
        desc.shape.dims = {4, 4};
        AbstractTensor out = AbstractTensor::create(desc, backend);
        if (!out.valid()) return {};
        MappedDense map = map_dense_mut(out);
        if (!map.ok || map.elems < 16) {
            map.unmap();
            return {};
        }
        std::fill(map.data, map.data + 16, static_cast<Scalar>(0));
        map.data[0] = map.data[5] = map.data[10] = map.data[15] = static_cast<Scalar>(1);
        map.unmap();
        return out;
    }

    static AbstractTensor affine_translation(const AbstractTensor& t) {
        if (!t.valid()) return {};
        const TensorDesc& td = t.desc();
        if (td.dtype != DType || td.layout != TensorLayout::Dense) return {};
        const bool t_vec = shape_is(td, {3}) || shape_is(td, {1, 3});
        const bool t_batch = (td.shape.dims.size() == 2 && td.shape.dims[1] == 3);
        if (!t_vec && !t_batch) return {};
        const uint32_t batch = t_batch ? td.shape.dims[0] : 1;

        TensorDesc desc{};
        desc.dtype = DType;
        desc.layout = TensorLayout::Dense;
        desc.shape.dims = (batch > 1) ? std::vector<uint32_t>{batch, 4, 4}
                                      : std::vector<uint32_t>{4, 4};
        AbstractTensor out = AbstractTensor::create(desc, t.backend());
        if (!out.valid()) return {};
        MappedDense tmap = map_dense(t);
        MappedDense map = map_dense_mut(out);
        if (!tmap.ok || !map.ok) {
            tmap.unmap();
            map.unmap();
            return {};
        }

        for (uint32_t i = 0; i < batch; ++i) {
            const Scalar* v = tmap.data + (t_vec ? 0u : static_cast<uint64_t>(i) * 3u);
            Scalar* dst = map.data + (static_cast<uint64_t>(i) * 16u);
            std::fill(dst, dst + 16, static_cast<Scalar>(0));
            dst[0] = dst[5] = dst[10] = dst[15] = static_cast<Scalar>(1);
            dst[12] = v[0];
            dst[13] = v[1];
            dst[14] = v[2];
        }

        tmap.unmap();
        map.unmap();
        return out;
    }

    static AbstractTensor affine_scale(const AbstractTensor& s) {
        if (!s.valid()) return {};
        const TensorDesc& sd = s.desc();
        if (sd.dtype != DType || sd.layout != TensorLayout::Dense) return {};
        const bool s_vec = shape_is(sd, {3}) || shape_is(sd, {1, 3});
        const bool s_batch = (sd.shape.dims.size() == 2 && sd.shape.dims[1] == 3);
        if (!s_vec && !s_batch) return {};
        const uint32_t batch = s_batch ? sd.shape.dims[0] : 1;

        TensorDesc desc{};
        desc.dtype = DType;
        desc.layout = TensorLayout::Dense;
        desc.shape.dims = (batch > 1) ? std::vector<uint32_t>{batch, 4, 4}
                                      : std::vector<uint32_t>{4, 4};
        AbstractTensor out = AbstractTensor::create(desc, s.backend());
        if (!out.valid()) return {};
        MappedDense smap = map_dense(s);
        MappedDense map = map_dense_mut(out);
        if (!smap.ok || !map.ok) {
            smap.unmap();
            map.unmap();
            return {};
        }

        for (uint32_t i = 0; i < batch; ++i) {
            const Scalar* v = smap.data + (s_vec ? 0u : static_cast<uint64_t>(i) * 3u);
            Scalar* dst = map.data + (static_cast<uint64_t>(i) * 16u);
            std::fill(dst, dst + 16, static_cast<Scalar>(0));
            dst[0] = v[0];
            dst[5] = v[1];
            dst[10] = v[2];
            dst[15] = static_cast<Scalar>(1);
        }

        smap.unmap();
        map.unmap();
        return out;
    }

    static AbstractTensor quat_identity(TensorBackend* backend) {
        TensorDesc desc{};
        desc.dtype = DType;
        desc.layout = TensorLayout::Dense;
        desc.shape.dims = {4};
        AbstractTensor out = AbstractTensor::create(desc, backend);
        if (!out.valid()) return {};
        MappedDense map = map_dense_mut(out);
        if (!map.ok || map.elems < 4) {
            map.unmap();
            return {};
        }
        map.data[0] = static_cast<Scalar>(1);
        map.data[1] = static_cast<Scalar>(0);
        map.data[2] = static_cast<Scalar>(0);
        map.data[3] = static_cast<Scalar>(0);
        map.unmap();
        return out;
    }

    static AbstractTensor quat_from_axis_angle(const AbstractTensor& axis, Scalar angle) {
        Scalar v[3]{};
        if (!read_vec3(axis, v)) return {};
        Scalar len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (len < kQuatEps) return quat_identity(axis.backend());
        Scalar inv = static_cast<Scalar>(1) / len;
        v[0] *= inv;
        v[1] *= inv;
        v[2] *= inv;
        const Scalar half = static_cast<Scalar>(0.5) * angle;
        const Scalar s = std::sin(half);
        const Scalar c = std::cos(half);

        TensorDesc desc{};
        desc.dtype = DType;
        desc.layout = TensorLayout::Dense;
        desc.shape.dims = {4};
        AbstractTensor out = AbstractTensor::create(desc, axis.backend());
        if (!out.valid()) return {};
        MappedDense map = map_dense_mut(out);
        if (!map.ok || map.elems < 4) {
            map.unmap();
            return {};
        }
        map.data[0] = c;
        map.data[1] = v[0] * s;
        map.data[2] = v[1] * s;
        map.data[3] = v[2] * s;
        map.unmap();
        return out;
    }

    static AbstractTensor quat_normalize(const AbstractTensor& q) {
        if (!q.valid()) return {};
        const TensorDesc& d = q.desc();
        if (d.dtype != DType || d.layout != TensorLayout::Dense) return {};
        if (!(shape_is(d, {4}) || (d.shape.dims.size() == 2 && d.shape.dims[1] == 4))) return {};

        TensorDesc out_desc = d;
        AbstractTensor out = AbstractTensor::create(out_desc, q.backend());
        if (!out.valid()) return {};
        MappedDense in = map_dense(q);
        MappedDense outm = map_dense_mut(out);
        if (!in.ok || !outm.ok) {
            in.unmap();
            outm.unmap();
            return {};
        }

        const uint64_t count = (d.shape.dims.size() == 2) ? d.shape.dims[0] : 1;
        for (uint64_t i = 0; i < count; ++i) {
            const Scalar* src = in.data + i * 4;
            Scalar* dst = outm.data + i * 4;
            const Scalar w = src[0], x = src[1], y = src[2], z = src[3];
            const Scalar len = std::sqrt(w * w + x * x + y * y + z * z);
            if (len < kQuatEps) {
                dst[0] = static_cast<Scalar>(1);
                dst[1] = dst[2] = dst[3] = static_cast<Scalar>(0);
            } else {
                const Scalar inv = static_cast<Scalar>(1) / len;
                dst[0] = w * inv;
                dst[1] = x * inv;
                dst[2] = y * inv;
                dst[3] = z * inv;
            }
        }

        in.unmap();
        outm.unmap();
        return out;
    }

    static AbstractTensor quat_mul(const AbstractTensor& a, const AbstractTensor& b) {
        if (!a.valid() || !b.valid()) return {};
        if (a.backend() != b.backend()) return {};
        const TensorDesc& ad = a.desc();
        const TensorDesc& bd = b.desc();
        if (ad.dtype != DType || bd.dtype != DType) return {};
        if (ad.layout != TensorLayout::Dense || bd.layout != TensorLayout::Dense) return {};

        const bool a_vec = shape_is(ad, {4});
        const bool b_vec = shape_is(bd, {4});
        const bool a_batch = (ad.shape.dims.size() == 2 && ad.shape.dims[1] == 4);
        const bool b_batch = (bd.shape.dims.size() == 2 && bd.shape.dims[1] == 4);
        if (!(a_vec || a_batch) || !(b_vec || b_batch)) return {};

        uint32_t batch = 1;
        if (a_batch && b_batch) {
            if (ad.shape.dims[0] != bd.shape.dims[0]) return {};
            batch = ad.shape.dims[0];
        } else if (a_batch) {
            batch = ad.shape.dims[0];
        } else if (b_batch) {
            batch = bd.shape.dims[0];
        }

        TensorDesc out_desc{};
        out_desc.dtype = DType;
        out_desc.layout = TensorLayout::Dense;
        out_desc.shape.dims = (batch == 1 && a_vec && b_vec) ? std::vector<uint32_t>{4}
                                                             : std::vector<uint32_t>{batch, 4};
        AbstractTensor out = AbstractTensor::create(out_desc, a.backend());
        if (!out.valid()) return {};

        MappedDense amap = map_dense(a);
        MappedDense bmap = map_dense(b);
        MappedDense omap = map_dense_mut(out);
        if (!amap.ok || !bmap.ok || !omap.ok) {
            amap.unmap();
            bmap.unmap();
            omap.unmap();
            return {};
        }

        for (uint32_t i = 0; i < batch; ++i) {
            const Scalar* qa = amap.data + (a_vec ? 0u : static_cast<uint64_t>(i) * 4u);
            const Scalar* qb = bmap.data + (b_vec ? 0u : static_cast<uint64_t>(i) * 4u);
            Scalar* qc = omap.data + static_cast<uint64_t>(i) * 4u;
            const Scalar w1 = qa[0], x1 = qa[1], y1 = qa[2], z1 = qa[3];
            const Scalar w2 = qb[0], x2 = qb[1], y2 = qb[2], z2 = qb[3];
            qc[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
            qc[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
            qc[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
            qc[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
        }

        amap.unmap();
        bmap.unmap();
        omap.unmap();
        return out;
    }

    static AbstractTensor quat_to_mat4(const AbstractTensor& q, const AbstractTensor& t) {
        if (!q.valid()) return {};
        const TensorDesc& qd = q.desc();
        if (qd.dtype != DType || qd.layout != TensorLayout::Dense) return {};
        const bool q_vec = shape_is(qd, {4});
        const bool q_batch = (qd.shape.dims.size() == 2 && qd.shape.dims[1] == 4);
        if (!q_vec && !q_batch) return {};

        const bool have_t = t.valid();
        if (have_t && t.backend() != q.backend()) return {};
        const TensorDesc& td = t.desc();
        const bool t_vec = have_t && shape_is(td, {3});
        const bool t_batch = have_t && (td.shape.dims.size() == 2 && td.shape.dims[1] == 3);
        if (have_t && !(t_vec || t_batch)) return {};

        uint32_t batch = q_batch ? qd.shape.dims[0] : 1;
        if (t_batch) {
            if (batch != 1 && td.shape.dims[0] != batch) return {};
            batch = td.shape.dims[0];
        }

        TensorDesc out_desc{};
        out_desc.dtype = DType;
        out_desc.layout = TensorLayout::Dense;
        if (batch == 1 && q_vec && !t_batch) {
            out_desc.shape.dims = {4, 4};
        } else {
            out_desc.shape.dims = {batch, 4, 4};
        }

        AbstractTensor out = AbstractTensor::create(out_desc, q.backend());
        if (!out.valid()) return {};

        MappedDense qmap = map_dense(q);
        MappedDense tmap = have_t ? map_dense(t) : MappedDense{};
        MappedDense omap = map_dense_mut(out);
        if (!qmap.ok || !omap.ok || (have_t && !tmap.ok)) {
            qmap.unmap();
            tmap.unmap();
            omap.unmap();
            return {};
        }

        for (uint32_t i = 0; i < batch; ++i) {
            const Scalar* qq = qmap.data + (q_vec ? 0u : static_cast<uint64_t>(i) * 4u);
            Scalar qw = qq[0], qx = qq[1], qy = qq[2], qz = qq[3];
            const Scalar len = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            if (len >= kQuatEps) {
                const Scalar inv = static_cast<Scalar>(1) / len;
                qw *= inv;
                qx *= inv;
                qy *= inv;
                qz *= inv;
            } else {
                qw = static_cast<Scalar>(1);
                qx = qy = qz = static_cast<Scalar>(0);
            }

            const Scalar* tt = nullptr;
            if (have_t) {
                tt = tmap.data + (t_vec ? 0u : static_cast<uint64_t>(i) * 3u);
            }
            Scalar* dst = omap.data + static_cast<uint64_t>(i) * 16u;
            quat_to_mat4_row(qw, qx, qy, qz, tt, dst);
        }

        qmap.unmap();
        tmap.unmap();
        omap.unmap();
        return out;
    }

    static AbstractTensor affine_from_quat_translation(const AbstractTensor& q, const AbstractTensor& t) {
        return quat_to_mat4(q, t);
    }

    static AbstractTensor transform_points(const AbstractTensor& points, const AbstractTensor& mat4) {
        if (!points.valid() || !mat4.valid()) return {};
        if (points.backend() != mat4.backend()) return {};

        const TensorDesc& pd = points.desc();
        const TensorDesc& md = mat4.desc();
        if (pd.dtype != DType || md.dtype != DType) return {};
        if (pd.layout != TensorLayout::Dense || md.layout != TensorLayout::Dense) return {};

        const bool p_vec = shape_is(pd, {3});
        const bool p_batch = (pd.shape.dims.size() == 2 && pd.shape.dims[1] == 3);
        if (!p_vec && !p_batch) return {};

        const bool m_single = shape_is(md, {4, 4});
        const bool m_batch = (md.shape.dims.size() == 3 && md.shape.dims[1] == 4 && md.shape.dims[2] == 4);
        if (!m_single && !m_batch) return {};

        uint32_t batch = p_vec ? 1 : pd.shape.dims[0];
        if (m_batch && md.shape.dims[0] != batch) return {};

        TensorDesc out_desc = pd;
        AbstractTensor out = AbstractTensor::create(out_desc, points.backend());
        if (!out.valid()) return {};

        MappedDense pmap = map_dense(points);
        MappedDense mmap = map_dense(mat4);
        MappedDense omap = map_dense_mut(out);
        if (!pmap.ok || !mmap.ok || !omap.ok) {
            pmap.unmap();
            mmap.unmap();
            omap.unmap();
            return {};
        }

        struct TransformCtx {
            const Scalar* p = nullptr;
            const Scalar* m = nullptr;
            Scalar* out = nullptr;
            bool p_vec = false;
            bool m_single = false;
        };

        auto transform_rows = [](const void* vctx, uint32_t i0, uint32_t i1) {
            const auto* ctx = static_cast<const TransformCtx*>(vctx);
            for (uint32_t i = i0; i < i1; ++i) {
                const Scalar* p = ctx->p + (ctx->p_vec ? 0u : static_cast<uint64_t>(i) * 3u);
                const Scalar* m = ctx->m + (ctx->m_single ? 0u : static_cast<uint64_t>(i) * 16u);
                Scalar* dst = ctx->out + (ctx->p_vec ? 0u : static_cast<uint64_t>(i) * 3u);
                const Scalar x = p[0];
                const Scalar y = p[1];
                const Scalar z = p[2];
                // Row-vector convention: v' = v * M (translation in last row).
                dst[0] = x * m[0] + y * m[4] + z * m[8] + m[12];
                dst[1] = x * m[1] + y * m[5] + z * m[9] + m[13];
                dst[2] = x * m[2] + y * m[6] + z * m[10] + m[14];
            }
        };

        TransformCtx ctx{};
        ctx.p = pmap.data;
        ctx.m = mmap.data;
        ctx.out = omap.data;
        ctx.p_vec = p_vec;
        ctx.m_single = m_single;

        submit_row_jobs(tensor_op_pool(), transform_rows, &ctx, 0, batch);

        pmap.unmap();
        mmap.unmap();
        omap.unmap();
        return out;
    }

    static bool intersect_plane_z(const AbstractTensor& origins,
                                  const AbstractTensor& dirs,
                                  Scalar plane_z,
                                  AbstractTensor* out_hits,
                                  AbstractTensor* out_mask) {
        if (!out_hits || !out_mask) return false;
        out_hits->reset();
        out_mask->reset();
        if (!origins.valid() || !dirs.valid()) return false;
        if (origins.backend() != dirs.backend()) return false;

        const TensorDesc& od = origins.desc();
        const TensorDesc& dd = dirs.desc();
        if (od.dtype != DType || dd.dtype != DType) return false;
        if (od.layout != TensorLayout::Dense || dd.layout != TensorLayout::Dense) return false;
        if (od.shape.dims.size() != 2 || dd.shape.dims.size() != 2) return false;
        const uint32_t batch = od.shape.dims[0];
        if (dd.shape.dims[0] != batch) return false;
        if (od.shape.dims[1] != 3 || dd.shape.dims[1] != 3) return false;

        auto* mem = dynamic_cast<InMemoryBackend*>(origins.backend());
        if (!mem) return false;

        TensorDesc hit_desc{};
        hit_desc.dtype = DType;
        hit_desc.layout = TensorLayout::Dense;
        hit_desc.shape.dims = {batch, 3u};
        *out_hits = AbstractTensor::create(hit_desc, origins.backend());
        if (!out_hits->valid()) return false;

        TensorDesc mask_desc{};
        mask_desc.dtype = TensorDType::Bool;
        mask_desc.layout = TensorLayout::Dense;
        mask_desc.shape.dims = {batch};
        *out_mask = AbstractTensor::create(mask_desc, origins.backend());
        if (!out_mask->valid()) return false;

        MappedDense omap = map_dense(origins);
        MappedDense dmap = map_dense(dirs);
        MappedDense hmap = map_dense_mut(*out_hits);
        if (!omap.ok || !dmap.ok || !hmap.ok) {
            omap.unmap();
            dmap.unmap();
            hmap.unmap();
            return false;
        }

        void* m_ptr_v = nullptr;
        size_t m_bytes = 0;
        if (!mem->map(out_mask->handle(), &m_ptr_v, &m_bytes)) {
            omap.unmap();
            dmap.unmap();
            hmap.unmap();
            return false;
        }
        auto* m_ptr = static_cast<uint8_t*>(m_ptr_v);

        struct IntersectCtx {
            const Scalar* o = nullptr;
            const Scalar* d = nullptr;
            Scalar* h = nullptr;
            uint8_t* m = nullptr;
            Scalar plane_z = static_cast<Scalar>(0);
        };

        auto intersect_rows = [](const void* vctx, uint32_t b0, uint32_t b1) {
            const auto* ctx = static_cast<const IntersectCtx*>(vctx);
            for (uint32_t b = b0; b < b1; ++b) {
                const Scalar ox = ctx->o[3 * b + 0];
                const Scalar oy = ctx->o[3 * b + 1];
                const Scalar oz = ctx->o[3 * b + 2];

                const Scalar dx = ctx->d[3 * b + 0];
                const Scalar dy = ctx->d[3 * b + 1];
                const Scalar dz = ctx->d[3 * b + 2];

                bool ok = std::fabs(static_cast<double>(dz)) > static_cast<double>(kPlaneEps);
                Scalar t = ok ? (ctx->plane_z - oz) / dz : static_cast<Scalar>(0);
                ok = ok && (t >= static_cast<Scalar>(0));
                ctx->m[b] = ok ? 1u : 0u;

                const Scalar hx = ok ? (ox + t * dx) : static_cast<Scalar>(0);
                const Scalar hy = ok ? (oy + t * dy) : static_cast<Scalar>(0);
                const Scalar hz = ok ? ctx->plane_z : static_cast<Scalar>(0);
                ctx->h[3 * b + 0] = hx;
                ctx->h[3 * b + 1] = hy;
                ctx->h[3 * b + 2] = hz;
            }
        };

        IntersectCtx ctx{};
        ctx.o = omap.data;
        ctx.d = dmap.data;
        ctx.h = hmap.data;
        ctx.m = m_ptr;
        ctx.plane_z = plane_z;

        submit_row_jobs(tensor_op_pool(), intersect_rows, &ctx, 0, batch);

        omap.unmap();
        dmap.unmap();
        hmap.unmap();
        mem->unmap(out_mask->handle());
        return true;
    }

    static bool scatter_add_2d(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        if (bd.dtype != DType || vd.dtype != DType) return false;
        const bool points_match = (pd.dtype == DType);
        if (!points_match) {
            if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
                pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
                return false;
            }
        }
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
            return false;

        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        bool val_scalar = false;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            val_scalar = true;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            if (vd.shape.dims[1] != channels) return false;
        } else {
            return false;
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        (void)ensure_tensor_zeroed(*out);
        (void)ensure_tensor_zeroed(*out);
        (void)ensure_tensor_zeroed(*out);
        (void)ensure_tensor_zeroed(*out);
        (void)ensure_tensor_zeroed(*out);
        (void)ensure_tensor_zeroed(*out);
        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense vmap = map_dense(values);
        MappedDense pmap{};
        void* points_ptr = nullptr;
        size_t points_bytes = 0;
        if (points_match) {
            pmap = map_dense(points);
            points_ptr = pmap.data; // Store pointer to mapped data
        } else {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
                bmap.unmap();
                omap.unmap();
                vmap.unmap();
                return false;
            }
        }
        if (!bmap.ok || !omap.ok || !vmap.ok || (points_match && !pmap.ok)) {
            bmap.unmap();
            omap.unmap();
            vmap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }

        if constexpr (std::is_same_v<Scalar, float>) {
            if (!tensor_copy_f32_into(base, out)) {
                bmap.unmap();
                omap.unmap();
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            bmap.unmap();
            omap.unmap();
            MappedDense omap2 = map_dense_mut(*out);
            if (!omap2.ok) {
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            omap = omap2;
        } else {
            const uint64_t elems = bd.shape.element_count();
            std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
        }

        CoordBuffer coord_buf{};
        if (!acquire_coord_buffer(count, 2u, base.backend(), coord_buf)) {
            bmap.unmap();
            omap.unmap();
            vmap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }
        int64_t* coords = coord_buf.coords_ptr;
        uint8_t* in_bounds = coord_buf.mask_ptr;

        const uint32_t tile_px = default_tile_px_2d(bd);

        const bool use_affine = bd.slice.valid && bd.slice.has_affine;

        convert_points_to_int_coords(pd,
                                     points_ptr,
                                     count,
                                     2u,
                                     points_match,
                                     use_affine,
                                     bd.slice.affine,
                                     bd.shape.dims.data(),
                                     true,
                                     coords,
                                     in_bounds);

        const bool use_span_tiling = span_tiling_enabled_from_env("NODUS_SCATTER_SPAN_TILING");
        if (use_span_tiling) {
            SpanOpConfig span_op{};
            span_op.aggregate = SpanAggregateOp::Add;
            if (span_scatter_add_2d<Scalar>(coords,
                                            in_bounds,
                                            count,
                                            width,
                                            height,
                                            channels,
                                            val_scalar,
                                            vmap.data,
                                            omap.data,
                                            base.backend(),
                                            tile_px,
                                            span_op)) {
                bmap.unmap();
                omap.unmap();
                vmap.unmap();
                release_coord_buffer(base.backend(), coord_buf);
                if (points_match) {
                    pmap.unmap();
                } else if (points_ptr) {
                    auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                    if (mem) mem->unmap(points.handle());
                }
                return true;
            }
        }

        auto& bins = tile_bins_2d();
        const TileBinning2D tile = build_tile_bins_2d(bd, coords, in_bounds,
                                                      count, width, height, pcols, bins);

        struct ScatterCtx {
            const int64_t* coords = nullptr;
            const Scalar* values = nullptr;
            Scalar* out = nullptr;
            const uint32_t* offsets = nullptr;
            const uint32_t* indices = nullptr;
            const uint64_t* tile_mask = nullptr;
            uint32_t tile_count = 0;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t channels = 0;
            uint32_t pcols = 0;
            uint32_t tiles_x = 0;
            uint32_t tile_px = 0;
            uint32_t tile_len = 0;
            TensorBackend* backend = nullptr;
            bool val_scalar = false;
            bool dense_grid = false;
            float dense_threshold = 0.0f;
        };

        auto scatter_rows = [](const void* vctx, uint32_t t0, uint32_t t1) {
            const auto* ctx = static_cast<const ScatterCtx*>(vctx);
            const uint32_t width = ctx->width;
            const uint32_t channels = ctx->channels;
            auto* mem = dynamic_cast<InMemoryBackend*>(ctx->backend);
            if (!mem) return;
            static thread_local AbstractTensorPool tile_pool(tile_pool_options());
            static thread_local AbstractTensorPool::PooledTensor tile_buf;
            TensorDesc tile_desc{};
            tile_desc.dtype = DType;
            tile_desc.layout = TensorLayout::Dense;
            tile_desc.shape.dims = {ctx->tile_len, channels};

            if (!tile_buf.valid() ||
                tile_buf.tensor().desc().shape.dims != tile_desc.shape.dims) {
                tile_buf = tile_pool.acquire(tile_desc, ctx->backend);
            }
            if (!tile_buf.valid()) return;

            (void)ensure_tensor_zeroed(tile_buf.tensor());
            void* tile_ptr_v = nullptr;
            size_t tile_bytes = 0;
            if (!mem->map(tile_buf.tensor().handle(), &tile_ptr_v, &tile_bytes)) return;
            auto* tile = static_cast<Scalar*>(tile_ptr_v);
            const size_t tile_elems = static_cast<size_t>(ctx->tile_len) * channels;

            for (uint32_t ti = t0; ti < t1; ++ti) {
                const uint32_t tid = ti;
                const uint32_t word = tid >> 6;
                const uint32_t bit = tid & 63u;
                if ((ctx->tile_mask[word] & (1ull << bit)) == 0ull) continue;
                const uint32_t begin = ctx->offsets[tid];
                const uint32_t end = ctx->offsets[tid + 1];
                if (begin == end) continue;
                auto tile_lock = acquire_tile_lock(tid);
                const uint32_t ty = tid / ctx->tiles_x;
                const uint32_t tx = tid - ty * ctx->tiles_x;
                const uint32_t tile_origin_x = tx * ctx->tile_px;
                const uint32_t tile_origin_y = ty * ctx->tile_px;
                const uint32_t y_end = std::min<uint32_t>(ctx->height, tile_origin_y + ctx->tile_px);
                const uint32_t x_end = std::min<uint32_t>(ctx->width, tile_origin_x + ctx->tile_px);
                const uint32_t rows = y_end - tile_origin_y;
                const uint32_t cols = x_end - tile_origin_x;

                const uint32_t hits = end - begin;
                const uint32_t tile_area = rows * cols;
                if (ctx->dense_grid && tile_area > 0) {
                    const float density = static_cast<float>(hits) / static_cast<float>(tile_area);
                    if (density >= ctx->dense_threshold) {
                        const uint32_t out_row_stride = width * channels;
                        Scalar* out_ptr =
                            ctx->out + (static_cast<uint64_t>(tile_origin_y) * width + tile_origin_x) * channels;
                        for (uint32_t ly = 0; ly < rows; ++ly) {
                            const uint32_t y = tile_origin_y + ly;
                            Scalar* dst = out_ptr + static_cast<uint64_t>(ly) * out_row_stride;
                            if (ctx->val_scalar) {
                                const Scalar* src = ctx->values + static_cast<size_t>(y) * width + tile_origin_x;
                                for (uint32_t lx = 0; lx < cols; ++lx) {
                                    const Scalar v = src[lx];
                                    for (uint32_t c = 0; c < channels; ++c) {
                                        dst[lx * channels + c] += v;
                                    }
                                }
                            } else {
                                const Scalar* src = ctx->values +
                                    (static_cast<size_t>(y) * width + tile_origin_x) * channels;
                                for (uint32_t lx = 0; lx < cols; ++lx) {
                                    const Scalar* src_row = src + static_cast<uint64_t>(lx) * channels;
                                    for (uint32_t c = 0; c < channels; ++c) {
                                        dst[lx * channels + c] += src_row[c];
                                    }
                                }
                            }
                        }
                        continue;
                    }
                }

                for (uint32_t idx = begin; idx < end; ++idx) {
                    const uint32_t i = ctx->indices[idx];
                    const int64_t xi = ctx->coords[static_cast<size_t>(i) * 2 + 0];
                    const int64_t yi = ctx->coords[static_cast<size_t>(i) * 2 + 1];
                    if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(width) ||
                        yi >= static_cast<int64_t>(ctx->height)) {
                        continue;
                    }
                    const uint32_t lx = static_cast<uint32_t>(xi) - tile_origin_x;
                    const uint32_t ly = static_cast<uint32_t>(yi) - tile_origin_y;
                    if (lx >= ctx->tile_px || ly >= ctx->tile_px) continue;
                    const uint32_t local = ly * ctx->tile_px + lx;
                    const uint64_t base_idx = static_cast<uint64_t>(local) * channels;
                    if (ctx->val_scalar) {
                        const Scalar v = ctx->values[i];
                        for (uint32_t c = 0; c < channels; ++c) {
                            tile[base_idx + c] += v;
                        }
                    } else {
                        const Scalar* src = ctx->values + static_cast<uint64_t>(i) * channels;
                        for (uint32_t c = 0; c < channels; ++c) {
                            tile[base_idx + c] += src[c];
                        }
                    }
                }

                const uint32_t out_row_stride = width * channels;
                const uint32_t x_row_stride = ctx->tile_px * channels;
                if constexpr (std::is_same_v<Scalar, float>) {
                    float* out_ptr =
                        ctx->out + (static_cast<uint64_t>(tile_origin_y) * width + tile_origin_x) * channels;
                    nodus_fused_add_f32_strided(out_ptr,
                                                out_row_stride,
                                                tile,
                                                x_row_stride,
                                                rows,
                                                cols,
                                                channels,
                                                0u,
                                                0.0f);
                } else {
                    Scalar* out_ptr =
                        ctx->out + (static_cast<uint64_t>(tile_origin_y) * width + tile_origin_x) * channels;
                    for (uint32_t r = 0; r < rows; ++r) {
                        Scalar* dst = out_ptr + static_cast<uint64_t>(r) * out_row_stride;
                        const Scalar* xv = tile + static_cast<uint64_t>(r) * x_row_stride;
                        for (uint32_t j = 0; j < cols; ++j) {
                            Scalar* drow = dst + static_cast<uint64_t>(j) * channels;
                            const Scalar* xrow = xv + static_cast<uint64_t>(j) * channels;
                            for (uint32_t cc = 0; cc < channels; ++cc) {
                                drow[cc] += xrow[cc];
                            }
                        }
                    }
                }
            }
            mem->unmap(tile_buf.tensor().handle());
        };

        ScatterCtx ctx{};
        ctx.coords = coords;
        ctx.values = vmap.data;
        ctx.out = omap.data;
        ctx.offsets = bins.offsets.data();
        ctx.indices = bins.indices.data();
        ctx.tile_mask = bins.tile_mask.data();
        ctx.tile_count = tile.tile_count;
        ctx.width = width;
        ctx.height = height;
        ctx.channels = channels;
        ctx.pcols = pcols;
        ctx.tiles_x = tile.tiles_x;
        ctx.tile_px = tile.tile_px;
        ctx.tile_len = tile.tile_px * tile.tile_px;
        ctx.backend = base.backend();
        ctx.val_scalar = val_scalar;
        ctx.dense_grid = tile.dense_grid;
        ctx.dense_threshold = tile.dense_threshold;

        submit_row_jobs(tensor_op_pool(), scatter_rows, &ctx, 0, ctx.tile_count);

        bmap.unmap();
        omap.unmap();
        vmap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return true;
    }

    static bool gather_2d(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid()) return false;
        if (base.backend() != points.backend()) return false;
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        if (bd.dtype != DType) return false;
        const bool points_match = (pd.dtype == DType);
        if (!points_match) {
            if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
                pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
                return false;
            }
        }
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense) return false;

        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        TensorDesc out_desc{};
        out_desc.dtype = DType;
        out_desc.layout = TensorLayout::Dense;
        if (channels == 1) {
            out_desc.shape.dims = {count};
        } else {
            out_desc.shape.dims = {count, channels};
        }
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap{};
        void* points_ptr = nullptr;
        size_t points_bytes = 0;
        if (points_match) {
            pmap = map_dense(points);
            points_ptr = pmap.data;
        } else {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
                bmap.unmap();
                omap.unmap();
                return false;
            }
        }
        if (!bmap.ok || !omap.ok || (points_match && !pmap.ok)) {
            bmap.unmap();
            omap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }

        CoordBuffer coord_buf{};
        if (!acquire_coord_buffer(count, 2u, base.backend(), coord_buf)) {
            bmap.unmap();
            omap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }
        int64_t* coords = coord_buf.coords_ptr;
        uint8_t* in_bounds = coord_buf.mask_ptr;
        const bool use_affine = bd.slice.valid && bd.slice.has_affine;
        convert_points_to_int_coords(pd,
                                     points_ptr,
                                     count,
                                     2u,
                                     points_match,
                                     use_affine,
                                     bd.slice.affine,
                                     bd.shape.dims.data(),
                                     true,
                                     coords,
                                     in_bounds);

        const bool use_span_tiling = span_tiling_enabled_from_env("NODUS_GATHER_SPAN_TILING");
        if (use_span_tiling) {
            SpanOpConfig span_op{};
            span_op.aggregate = SpanAggregateOp::Replace;
            span_op.clamp = clamp;
            if (span_gather_2d<Scalar>(coords,
                                       in_bounds,
                                       count,
                                       width,
                                       height,
                                       channels,
                                       bmap.data,
                                       omap.data,
                                       base.backend(),
                                       0u,
                                       span_op)) {
                bmap.unmap();
                omap.unmap();
                release_coord_buffer(base.backend(), coord_buf);
                if (points_match) {
                    pmap.unmap();
                } else if (points_ptr) {
                    auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                    if (mem) mem->unmap(points.handle());
                }
                return true;
            }
        }

        struct GatherCtx {
            const Scalar* base = nullptr;
            Scalar* out = nullptr;
            const int64_t* coords = nullptr;
            const uint8_t* in_bounds = nullptr;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t channels = 0;
            uint32_t pcols = 0;
            bool clamp = false;
        };

        GatherCtx ctx{};
        ctx.base = bmap.data;
        ctx.out = omap.data;
        ctx.coords = coords;
        ctx.in_bounds = in_bounds;
        ctx.width = width;
        ctx.height = height;
        ctx.channels = channels;
        ctx.pcols = pcols;
        ctx.clamp = clamp;

        auto gather_rows = [](const void* vctx, uint32_t i0, uint32_t i1) {
            const auto* ctx = static_cast<const GatherCtx*>(vctx);
            for (uint32_t i = i0; i < i1; ++i) {
                if (!ctx->in_bounds[i]) {
                    if (ctx->clamp) continue;
                    continue;
                }

                const int64_t xi = ctx->coords[static_cast<size_t>(i) * 2 + 0];
                const int64_t yi = ctx->coords[static_cast<size_t>(i) * 2 + 1];

                const uint64_t base_idx = (static_cast<uint64_t>(yi) * ctx->width +
                                           static_cast<uint64_t>(xi)) * ctx->channels;
                if (ctx->channels == 1) {
                    ctx->out[i] = ctx->base[base_idx];
                } else {
                    Scalar* dst = ctx->out + static_cast<uint64_t>(i) * ctx->channels;
                    const Scalar* src = ctx->base + base_idx;
                    for (uint32_t c = 0; c < ctx->channels; ++c) {
                        dst[c] = src[c];
                    }
                }
            }
        };

        submit_row_jobs(tensor_op_pool(), gather_rows, &ctx, 0, count);

        bmap.unmap();
        omap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return true;
    }

    static bool gather_add_2d(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp) {
        if (!out || !out->valid()) return false;
        if (!base.valid() || !points.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != out->backend()) return false;
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& od = out->desc();
        if (bd.dtype != DType || od.dtype != DType) return false;
        const bool points_match = (pd.dtype == DType);
        if (!points_match) {
            if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
                pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
                return false;
            }
        }
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || od.layout != TensorLayout::Dense)
            return false;

        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        if (channels == 1) {
            if (!shape_is(od, {count})) return false;
        } else {
            if (!shape_is(od, {count, channels})) return false;
        }

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap{};
        void* points_ptr = nullptr;
        size_t points_bytes = 0;
        if (points_match) {
            pmap = map_dense(points);
            points_ptr = pmap.data;
        } else {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
                bmap.unmap();
                omap.unmap();
                return false;
            }
        }
        if (!bmap.ok || !omap.ok || (points_match && !pmap.ok)) {
            bmap.unmap();
            omap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }

        CoordBuffer coord_buf{};
        if (!acquire_coord_buffer(count, 2u, base.backend(), coord_buf)) {
            bmap.unmap();
            omap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }
        int64_t* coords = coord_buf.coords_ptr;
        uint8_t* in_bounds = coord_buf.mask_ptr;
        const bool use_affine = bd.slice.valid && bd.slice.has_affine;
        convert_points_to_int_coords(pd,
                                     points_ptr,
                                     count,
                                     2u,
                                     points_match,
                                     use_affine,
                                     bd.slice.affine,
                                     bd.shape.dims.data(),
                                     true,
                                     coords,
                                     in_bounds);

        const bool use_span_tiling = span_tiling_enabled_from_env("NODUS_GATHER_SPAN_TILING");
        if (use_span_tiling) {
            SpanOpConfig span_op{};
            span_op.aggregate = SpanAggregateOp::Add;
            span_op.clamp = clamp;
            if (span_gather_add_2d<Scalar>(coords,
                                           in_bounds,
                                           count,
                                           width,
                                           height,
                                           channels,
                                           bmap.data,
                                           omap.data,
                                           base.backend(),
                                           0u,
                                           span_op)) {
                bmap.unmap();
                omap.unmap();
                release_coord_buffer(base.backend(), coord_buf);
                if (points_match) {
                    pmap.unmap();
                } else if (points_ptr) {
                    auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                    if (mem) mem->unmap(points.handle());
                }
                return true;
            }
        }

        struct GatherCtx {
            const Scalar* base = nullptr;
            Scalar* out = nullptr;
            const int64_t* coords = nullptr;
            const uint8_t* in_bounds = nullptr;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t channels = 0;
            uint32_t pcols = 0;
            bool clamp = false;
        };

        GatherCtx ctx{};
        ctx.base = bmap.data;
        ctx.out = omap.data;
        ctx.coords = coords;
        ctx.in_bounds = in_bounds;
        ctx.width = width;
        ctx.height = height;
        ctx.channels = channels;
        ctx.pcols = pcols;
        ctx.clamp = clamp;

        auto gather_rows = [](const void* vctx, uint32_t i0, uint32_t i1) {
            const auto* ctx = static_cast<const GatherCtx*>(vctx);
            for (uint32_t i = i0; i < i1; ++i) {
                if (!ctx->in_bounds[i]) {
                    if (ctx->clamp) continue;
                    continue;
                }

                const int64_t xi = ctx->coords[static_cast<size_t>(i) * 2 + 0];
                const int64_t yi = ctx->coords[static_cast<size_t>(i) * 2 + 1];

                const uint64_t base_idx = (static_cast<uint64_t>(yi) * ctx->width +
                                           static_cast<uint64_t>(xi)) * ctx->channels;
                if (ctx->channels == 1) {
                    ctx->out[i] += ctx->base[base_idx];
                } else {
                    Scalar* dst = ctx->out + static_cast<uint64_t>(i) * ctx->channels;
                    const Scalar* src = ctx->base + base_idx;
                    for (uint32_t c = 0; c < ctx->channels; ++c) {
                        dst[c] += src[c];
                    }
                }
            }
        };

        submit_row_jobs(tensor_op_pool(), gather_rows, &ctx, 0, count);

        bmap.unmap();
        omap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return true;
    }

    static bool scatter_add_nd_1d(const AbstractTensor& base,
                                  const AbstractTensor& points,
                                  const AbstractTensor& values,
                                  AbstractTensor* out,
                                  bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        if (bd.dtype != DType || vd.dtype != DType) return false;
        const bool points_match = (pd.dtype == DType);
        if (!points_match) {
            if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
                pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
                return false;
            }
        }
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
            return false;
        if (pd.shape.dims.size() != 2 || pd.shape.dims[1] != 1) return false;
        const uint32_t count = pd.shape.dims[0];

        const size_t out_rank = bd.shape.dims.size();
        if (out_rank != 1 && out_rank != 2) return false;
        const uint32_t channels = (out_rank == 2) ? bd.shape.dims[1] : 1u;

        bool val_scalar = false;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            val_scalar = true;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            if (vd.shape.dims[1] != channels) return false;
        } else {
            return false;
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap = map_dense(points);
        MappedDense vmap = map_dense(values);
        if (!bmap.ok || !omap.ok || !pmap.ok || !vmap.ok) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            return false;
        }

        if constexpr (std::is_same_v<Scalar, float>) {
            if (!tensor_copy_f32_into(base, out)) {
                bmap.unmap();
                omap.unmap();
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            bmap.unmap();
            omap.unmap();
            MappedDense omap2 = map_dense_mut(*out);
            if (!omap2.ok) {
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            omap = omap2;
        } else {
            const uint64_t elems = bd.shape.element_count();
            std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
        }

        const bool use_affine = bd.slice.valid && bd.slice.has_affine;
        const uint64_t stride0 = channels;
        const uint64_t limit0 = bd.shape.dims[0];
        constexpr uint64_t kDefaultTileCacheBytes = 6ull * 1024ull * 1024ull * 1024ull;
        const uint64_t tile_elems = std::max<uint64_t>(1u, choose_linear_tile_elems(bd, 1u, kDefaultTileCacheBytes));

        CoordBuffer coord_buf{};
        if (!acquire_coord_buffer(count, 1u, base.backend(), coord_buf)) {
            pmap.unmap();
            vmap.unmap();
            omap.unmap();
            return false;
        }
        int64_t* coords = coord_buf.coords_ptr;
        uint8_t* in_bounds = coord_buf.mask_ptr;
        convert_points_to_int_coords(pd,
                                     pmap.data,
                                     count,
                                     1u,
                                     true,
                                     use_affine,
                                     bd.slice.affine,
                                     bd.shape.dims.data(),
                                     true,
                                     coords,
                                     in_bounds);

        bmap.unmap();
        omap.unmap();

        if (dyadic_scatter_from_coords_auto<Scalar, policies::Add, policies::Add>(
                coord_buf,
                count,
                1u,
                bd.shape.dims.data(),
                channels,
                val_scalar,
                vmap.data,
                *out,
                base.backend())) {
            pmap.unmap();
            vmap.unmap();
            release_coord_buffer(base.backend(), coord_buf);
            return true;
        }
        pmap.unmap();
        vmap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        return false;
    }

    static bool scatter_add_nd_3d(const AbstractTensor& base,
                                  const AbstractTensor& points,
                                  const AbstractTensor& values,
                                  AbstractTensor* out,
                                  bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        if (bd.dtype != DType || vd.dtype != DType) return false;
        const bool points_match = (pd.dtype == DType);
        if (!points_match) {
            if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
                pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
                return false;
            }
        }
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
            return false;
        if (pd.shape.dims.size() != 2 || pd.shape.dims[1] != 3) return false;
        const uint32_t count = pd.shape.dims[0];

        const size_t out_rank = bd.shape.dims.size();
        if (out_rank != 3 && out_rank != 4) return false;
        const uint32_t channels = (out_rank == 4) ? bd.shape.dims[3] : 1u;

        bool val_scalar = false;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            val_scalar = true;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            if (vd.shape.dims[1] != channels) return false;
        } else {
            return false;
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap = map_dense(points);
        MappedDense vmap = map_dense(values);
        if (!bmap.ok || !omap.ok || !pmap.ok || !vmap.ok) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            return false;
        }

        if constexpr (std::is_same_v<Scalar, float>) {
            if (!tensor_copy_f32_into(base, out)) {
                bmap.unmap();
                omap.unmap();
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            bmap.unmap();
            omap.unmap();
            MappedDense omap2 = map_dense_mut(*out);
            if (!omap2.ok) {
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            omap = omap2;
        } else {
            const uint64_t elems = bd.shape.element_count();
            std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
        }

        std::vector<uint64_t> strides;
        if (!compute_dense_strides_u64(bd.shape.dims, strides)) {
            pmap.unmap();
            vmap.unmap();
            omap.unmap();
            return false;
        }

        std::vector<uint64_t> dense_strides;
        if (!compute_dense_strides_u64(std::vector<uint32_t>(bd.shape.dims.begin(), bd.shape.dims.begin() + 3),
                                       dense_strides)) {
            pmap.unmap();
            vmap.unmap();
            omap.unmap();
            return false;
        }
        constexpr uint64_t kDefaultTileCacheBytes = 6ull * 1024ull * 1024ull * 1024ull;
        const uint64_t tile_elems = std::max<uint64_t>(1u, choose_linear_tile_elems(bd, 3u, kDefaultTileCacheBytes));

        const bool use_affine = bd.slice.valid && bd.slice.has_affine;

        CoordBuffer coord_buf{};
        if (!acquire_coord_buffer(count, 3u, base.backend(), coord_buf)) {
            pmap.unmap();
            vmap.unmap();
            omap.unmap();
            return false;
        }
        int64_t* coords = coord_buf.coords_ptr;
        uint8_t* in_bounds = coord_buf.mask_ptr;
        convert_points_to_int_coords(pd,
                                     pmap.data,
                                     count,
                                     3u,
                                     true,
                                     use_affine,
                                     bd.slice.affine,
                                     bd.shape.dims.data(),
                                     true,
                                     coords,
                                     in_bounds);

        bmap.unmap();
        omap.unmap();
        if (dyadic_scatter_from_coords_auto<Scalar, policies::Add, policies::Add>(
                coord_buf,
                count,
                3u,
                bd.shape.dims.data(),
                channels,
                val_scalar,
                vmap.data,
                *out,
                base.backend())) {
            pmap.unmap();
            vmap.unmap();
            release_coord_buffer(base.backend(), coord_buf);
            return true;
        }
        pmap.unmap();
        vmap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        return false;
    }

    static bool scatter_add_nd_4d(const AbstractTensor& base,
                                  const AbstractTensor& points,
                                  const AbstractTensor& values,
                                  AbstractTensor* out,
                                  bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        if (bd.dtype != DType || vd.dtype != DType) return false;
        const bool points_match = (pd.dtype == DType);
        if (!points_match) {
            if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
                pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
                return false;
            }
        }
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
            return false;
        if (pd.shape.dims.size() != 2 || pd.shape.dims[1] != 4) return false;
        const uint32_t count = pd.shape.dims[0];

        const size_t out_rank = bd.shape.dims.size();
        if (out_rank != 4 && out_rank != 5) return false;
        const uint32_t channels = (out_rank == 5) ? bd.shape.dims[4] : 1u;

        bool val_scalar = false;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            val_scalar = true;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            if (vd.shape.dims[1] != channels) return false;
        } else {
            return false;
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap = map_dense(points);
        MappedDense vmap = map_dense(values);
        if (!bmap.ok || !omap.ok || !pmap.ok || !vmap.ok) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            return false;
        }

        if constexpr (std::is_same_v<Scalar, float>) {
            if (!tensor_copy_f32_into(base, out)) {
                bmap.unmap();
                omap.unmap();
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            bmap.unmap();
            omap.unmap();
            MappedDense omap2 = map_dense_mut(*out);
            if (!omap2.ok) {
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            omap = omap2;
        } else {
            const uint64_t elems = bd.shape.element_count();
            std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
        }

        std::vector<uint64_t> strides;
        if (!compute_dense_strides_u64(bd.shape.dims, strides)) {
            pmap.unmap();
            vmap.unmap();
            omap.unmap();
            return false;
        }

        std::vector<uint64_t> dense_strides;
        if (!compute_dense_strides_u64(std::vector<uint32_t>(bd.shape.dims.begin(), bd.shape.dims.begin() + 4),
                                       dense_strides)) {
            pmap.unmap();
            vmap.unmap();
            omap.unmap();
            return false;
        }
        constexpr uint64_t kDefaultTileCacheBytes = 6ull * 1024ull * 1024ull * 1024ull;
        const uint64_t tile_elems = std::max<uint64_t>(1u, choose_linear_tile_elems(bd, 4u, kDefaultTileCacheBytes));

        CoordBuffer coord_buf{};
        if (!acquire_coord_buffer(count, 4u, base.backend(), coord_buf)) {
            pmap.unmap();
            vmap.unmap();
            omap.unmap();
            return false;
        }
        int64_t* coords = coord_buf.coords_ptr;
        uint8_t* in_bounds = coord_buf.mask_ptr;
        convert_points_to_int_coords(pd,
                                     pmap.data,
                                     count,
                                     4u,
                                     true,
                                     false,
                                     nullptr,
                                     bd.shape.dims.data(),
                                     true,
                                     coords,
                                     in_bounds);

        bmap.unmap();
        omap.unmap();
        if (dyadic_scatter_from_coords_auto<Scalar, policies::Add, policies::Add>(
                coord_buf,
                count,
                4u,
                bd.shape.dims.data(),
                channels,
                val_scalar,
                vmap.data,
                *out,
                base.backend())) {
            pmap.unmap();
            vmap.unmap();
            release_coord_buffer(base.backend(), coord_buf);
            return true;
        }
        pmap.unmap();
        vmap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        return false;
    }

    static bool scatter_add_nd(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;

        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        if (bd.dtype != DType || vd.dtype != DType) return false;
        const bool points_match = (pd.dtype == DType);
        if (!points_match) {
            if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
                pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
                return false;
            }
        }
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
            return false;
        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t dims = pd.shape.dims[1];
        if (dims == 0) return false;
        const size_t out_rank = bd.shape.dims.size();
        if (out_rank < dims || out_rank > dims + 1) return false;

        if (dims == 1 && points_match) return scatter_add_nd_1d(base, points, values, out, clamp);
        if (dims == 2) return scatter_add_2d(base, points, values, out, clamp);
        if (dims == 3 && points_match) return scatter_add_nd_3d(base, points, values, out, clamp);
        if (dims == 4 && points_match) return scatter_add_nd_4d(base, points, values, out, clamp);

        const uint32_t channels = (out_rank == dims + 1) ? bd.shape.dims.back() : 1u;
        bool val_scalar = false;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            val_scalar = true;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            if (vd.shape.dims[1] != channels) return false;
        } else {
            return false;
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense vmap = map_dense(values);
        MappedDense pmap{};
        void* points_ptr = nullptr;
        size_t points_bytes = 0;
        if (points_match) {
            pmap = map_dense(points);
            points_ptr = pmap.data;
        } else {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
                bmap.unmap();
                omap.unmap();
                vmap.unmap();
                return false;
            }
        }
        if (!bmap.ok || !omap.ok || !vmap.ok || (points_match && !pmap.ok)) {
            bmap.unmap();
            omap.unmap();
            vmap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }

        if (!tensor_copy_f32_into(base, out)) {
            bmap.unmap();
            omap.unmap();
            vmap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }
        bmap.unmap();
        omap.unmap();
        MappedDense omap2 = map_dense_mut(*out);
        if (!omap2.ok) {
            vmap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }
        omap = omap2;

        std::vector<uint64_t> strides;
        if (!compute_dense_strides_u64(bd.shape.dims, strides)) {
            pmap.unmap();
            vmap.unmap();
            omap.unmap();
            return false;
        }

        const bool use_affine = bd.slice.valid && bd.slice.has_affine && dims <= 3;

        CoordBuffer coord_buf{};
        if (!acquire_coord_buffer(count, dims, base.backend(), coord_buf)) {
            vmap.unmap();
            omap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }
        int64_t* coords = coord_buf.coords_ptr;
        uint8_t* in_bounds = coord_buf.mask_ptr;
        convert_points_to_int_coords(pd,
                                     points_ptr,
                                     count,
                                     dims,
                                     points_match,
                                     use_affine,
                                     bd.slice.affine,
                                     bd.shape.dims.data(),
                                     true,
                                     coords,
                                     in_bounds);

        bmap.unmap();
        omap.unmap();

        if (dyadic_scatter_from_coords_auto<Scalar, policies::Add, policies::Add>(
                coord_buf,
                count,
                dims,
                bd.shape.dims.data(),
                channels,
                val_scalar,
                vmap.data,
                *out,
                base.backend())) {
            vmap.unmap();
            release_coord_buffer(base.backend(), coord_buf);
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return true;
        }
        vmap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return false;
    }

    static bool gather_nd(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid()) return false;
        if (base.backend() != points.backend()) return false;

        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        if (bd.dtype != DType) return false;
        const bool points_match = (pd.dtype == DType);
        if (!points_match) {
            if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
                pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
                return false;
            }
        }
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense) return false;
        if (pd.shape.dims.size() != 2) return false;

        const uint32_t count = pd.shape.dims[0];
        const uint32_t dims = pd.shape.dims[1];
        if (dims == 0) return false;

        const size_t in_rank = bd.shape.dims.size();
        if (in_rank < dims || in_rank > dims + 1) return false;

        if (dims == 2) return gather_2d(base, points, out, clamp);

        const uint32_t channels = (in_rank == dims + 1) ? bd.shape.dims.back() : 1u;

        TensorDesc out_desc{};
        out_desc.dtype = DType;
        out_desc.layout = TensorLayout::Dense;
        if (channels == 1) {
            out_desc.shape.dims = {count};
        } else {
            out_desc.shape.dims = {count, channels};
        }
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap{};
        void* points_ptr = nullptr;
        size_t points_bytes = 0;
        if (points_match) {
            pmap = map_dense(points);
            points_ptr = pmap.data;
        } else {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
                bmap.unmap();
                omap.unmap();
                return false;
            }
        }
        if (!bmap.ok || !omap.ok || (points_match && !pmap.ok)) {
            bmap.unmap();
            omap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }

        std::vector<uint64_t> strides;
        if (!compute_dense_strides_u64(bd.shape.dims, strides)) {
            bmap.unmap();
            omap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }

        const bool use_affine = bd.slice.valid && bd.slice.has_affine && dims <= 3;
        CoordBuffer coord_buf{};
        if (!acquire_coord_buffer(count, dims, base.backend(), coord_buf)) {
            bmap.unmap();
            omap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }
        int64_t* coords = coord_buf.coords_ptr;
        uint8_t* in_bounds = coord_buf.mask_ptr;
        convert_points_to_int_coords(pd,
                                     points_ptr,
                                     count,
                                     dims,
                                     points_match,
                                     use_affine,
                                     bd.slice.affine,
                                     bd.shape.dims.data(),
                                     true,
                                     coords,
                                     in_bounds);
        for (uint32_t i = 0; i < count; ++i) {
            if (!in_bounds[i]) {
                if (clamp) continue;
                continue;
            }

            uint64_t base_idx = 0;
            for (uint32_t d = 0; d < dims; ++d) {
                base_idx += static_cast<uint64_t>(coords[static_cast<size_t>(i) * dims + d]) * strides[d];
            }

            if (channels == 1) {
                omap.data[i] = bmap.data[base_idx];
            } else {
                Scalar* dst = omap.data + static_cast<uint64_t>(i) * channels;
                const Scalar* src = bmap.data + base_idx;
                for (uint32_t c = 0; c < channels; ++c) {
                    dst[c] = src[c];
                }
            }
        }

        bmap.unmap();
        omap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return true;
    }

    static bool gather_add_nd(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp) {
        if (!out || !out->valid()) return false;
        if (!base.valid() || !points.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != out->backend()) return false;

        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& od = out->desc();
        if (bd.dtype != DType || od.dtype != DType) return false;
        const bool points_match = (pd.dtype == DType);
        if (!points_match) {
            if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
                pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
                return false;
            }
        }
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || od.layout != TensorLayout::Dense)
            return false;
        if (pd.shape.dims.size() != 2) return false;

        const uint32_t count = pd.shape.dims[0];
        const uint32_t dims = pd.shape.dims[1];
        if (dims == 0) return false;

        const size_t in_rank = bd.shape.dims.size();
        if (in_rank < dims || in_rank > dims + 1) return false;

        if (dims == 2) return gather_add_2d(base, points, out, clamp);

        const uint32_t channels = (in_rank == dims + 1) ? bd.shape.dims.back() : 1u;
        if (channels == 1) {
            if (!shape_is(od, {count})) return false;
        } else {
            if (!shape_is(od, {count, channels})) return false;
        }

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap{};
        void* points_ptr = nullptr;
        size_t points_bytes = 0;
        if (points_match) {
            pmap = map_dense(points);
            points_ptr = pmap.data;
        } else {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
                bmap.unmap();
                omap.unmap();
                return false;
            }
        }
        if (!bmap.ok || !omap.ok || (points_match && !pmap.ok)) {
            bmap.unmap();
            omap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }

        std::vector<uint64_t> strides;
        if (!compute_dense_strides_u64(bd.shape.dims, strides)) {
            bmap.unmap();
            omap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }

        const bool use_affine = bd.slice.valid && bd.slice.has_affine && dims <= 3;
        CoordBuffer coord_buf{};
        if (!acquire_coord_buffer(count, dims, base.backend(), coord_buf)) {
            bmap.unmap();
            omap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }
        int64_t* coords = coord_buf.coords_ptr;
        uint8_t* in_bounds = coord_buf.mask_ptr;
        convert_points_to_int_coords(pd,
                                     points_ptr,
                                     count,
                                     dims,
                                     points_match,
                                     use_affine,
                                     bd.slice.affine,
                                     bd.shape.dims.data(),
                                     true,
                                     coords,
                                     in_bounds);
        for (uint32_t i = 0; i < count; ++i) {
            if (!in_bounds[i]) {
                if (clamp) continue;
                continue;
            }

            uint64_t base_idx = 0;
            for (uint32_t d = 0; d < dims; ++d) {
                base_idx += static_cast<uint64_t>(coords[static_cast<size_t>(i) * dims + d]) * strides[d];
            }

            if (channels == 1) {
                omap.data[i] += bmap.data[base_idx];
            } else {
                Scalar* dst = omap.data + static_cast<uint64_t>(i) * channels;
                const Scalar* src = bmap.data + base_idx;
                for (uint32_t c = 0; c < channels; ++c) {
                    dst[c] += src[c];
                }
            }
        }

        bmap.unmap();
        omap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return true;
    }

    static bool build_stencil_from_footprint_2d(const TensorFootprint2D& footprint,
                                                StencilOrientation orientation,
                                                TensorStencil* out_stencil) {
        if (!out_stencil) return false;
        out_stencil->offsets.reset();
        out_stencil->weights.reset();
        out_stencil->support.reset();
        out_stencil->orientation = orientation;
        if (!footprint.weights.valid()) return false;
        if (footprint.weights.desc().dtype != DType) return false;
        if (footprint.weights.desc().layout != TensorLayout::Dense) return false;
        if (footprint.weights.desc().shape.dims.size() != 2) return false;
        const uint32_t ks = footprint.weights.desc().shape.dims[0];
        if (ks == 0 || footprint.weights.desc().shape.dims[1] != ks) return false;
        const int32_t radius = static_cast<int32_t>(ks / 2);

        MappedDense kmap = map_dense(footprint.weights);
        if (!kmap.ok) {
            kmap.unmap();
            return false;
        }

        uint32_t count = 0;
        for (uint32_t y = 0; y < ks; ++y) {
            const uint64_t row = static_cast<uint64_t>(y) * ks;
            for (uint32_t x = 0; x < ks; ++x) {
                const Scalar w = kmap.data[row + x];
                if (w != static_cast<Scalar>(0)) ++count;
            }
        }

        TensorDesc off_desc{};
        off_desc.dtype = TensorDType::I32;
        off_desc.layout = TensorLayout::Dense;
        off_desc.shape.dims = {count, 2u};
        out_stencil->offsets = AbstractTensor::create(off_desc, footprint.weights.backend());
        if (!out_stencil->offsets.valid()) {
            kmap.unmap();
            return false;
        }

        TensorDesc w_desc{};
        w_desc.dtype = DType;
        w_desc.layout = TensorLayout::Dense;
        w_desc.shape.dims = {count};
        out_stencil->weights = AbstractTensor::create(w_desc, footprint.weights.backend());
        if (!out_stencil->weights.valid()) {
            kmap.unmap();
            return false;
        }

        MappedDenseI32 omap = map_dense_i32(out_stencil->offsets);
        MappedDense wmap = map_dense_mut(out_stencil->weights);
        if (!omap.ok || !wmap.ok) {
            kmap.unmap();
            omap.unmap();
            wmap.unmap();
            return false;
        }

        uint32_t write = 0;
        for (uint32_t y = 0; y < ks; ++y) {
            const int32_t dy = static_cast<int32_t>(y) - radius;
            const uint64_t row = static_cast<uint64_t>(y) * ks;
            for (uint32_t x = 0; x < ks; ++x) {
                const Scalar w = kmap.data[row + x];
                if (w == static_cast<Scalar>(0)) continue;
                const int32_t dx = static_cast<int32_t>(x) - radius;
                const int32_t ox = (orientation == StencilOrientation::Convolution) ? -dx : dx;
                const int32_t oy = (orientation == StencilOrientation::Convolution) ? -dy : dy;
                omap.data[write * 2 + 0] = ox;
                omap.data[write * 2 + 1] = oy;
                wmap.data[write] = w;
                ++write;
            }
        }

        kmap.unmap();
        omap.unmap();
        wmap.unmap();
        (void)tensor_support_from_offsets(out_stencil->offsets, &out_stencil->support);
        return true;
    }

    static bool gather_stencil_2d(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary,
                                  const AbstractTensor* target_mask,
                                  const AbstractTensor* source_mask,
                                  bool normalize) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !stencil.offsets.valid() || !stencil.weights.valid()) return false;
        if (base.backend() != stencil.offsets.backend() || base.backend() != stencil.weights.backend()) return false;
        if (target_mask && base.backend() != target_mask->backend()) return false;
        if (source_mask && base.backend() != source_mask->backend()) return false;

        const TensorDesc& bd = base.desc();
        const TensorDesc& od = stencil.offsets.desc();
        const TensorDesc& wd = stencil.weights.desc();
        if (bd.dtype != DType || wd.dtype != DType) return false;
        if (od.dtype != TensorDType::I32) return false;
        if (bd.layout != TensorLayout::Dense || od.layout != TensorLayout::Dense || wd.layout != TensorLayout::Dense)
            return false;
        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

        if (od.shape.dims.size() != 2 || od.shape.dims[1] != 2) return false;
        const uint32_t count = od.shape.dims[0];
        if (wd.shape.dims.size() != 1 || wd.shape.dims[0] != count) return false;

        if (target_mask) {
            const TensorDesc& md = target_mask->desc();
            if (md.dtype != TensorDType::Bool || md.layout != TensorLayout::Dense) return false;
            if (md.shape.dims.size() != 2 || md.shape.dims[0] != height || md.shape.dims[1] != width) return false;
        }
        if (source_mask) {
            const TensorDesc& md = source_mask->desc();
            if (md.dtype != TensorDType::Bool || md.layout != TensorLayout::Dense) return false;
            if (md.shape.dims.size() != 2 || md.shape.dims[0] != height || md.shape.dims[1] != width) return false;
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDenseI32 omap_off = map_dense_i32(stencil.offsets);
        MappedDense wmap = map_dense(stencil.weights);
        MappedDenseU8 mmap{};
        if (target_mask) {
            mmap = map_dense_u8(*target_mask);
        }
        MappedDenseU8 smap{};
        if (source_mask) {
            smap = map_dense_u8(*source_mask);
        }
        if (!bmap.ok || !omap.ok || !omap_off.ok || !wmap.ok ||
            (target_mask && !mmap.ok) || (source_mask && !smap.ok)) {
            bmap.unmap();
            omap.unmap();
            omap_off.unmap();
            wmap.unmap();
            if (target_mask) mmap.unmap();
            if (source_mask) smap.unmap();
            return false;
        }

        const auto* off_ptr = omap_off.data;
        const Scalar* w_ptr = wmap.data;

        int32_t radius_x = 0;
        int32_t radius_y = 0;
        for (uint32_t i = 0; i < count; ++i) {
            radius_x = std::max(radius_x, std::abs(off_ptr[i * 2 + 0]));
            radius_y = std::max(radius_y, std::abs(off_ptr[i * 2 + 1]));
        }

        static uint32_t kTilePxX = 0;
        static uint32_t kTilePxY = 0;
        if (kTilePxX == 0 || kTilePxY == 0) {
            constexpr uint64_t kDefaultTileCacheBytes = 6ull * 1024ull * 1024ull * 1024ull;
            TileShape2D shape = choose_tile_shape_2d_impl(bd, static_cast<uint32_t>(radius_x),
                                                         static_cast<uint32_t>(radius_y), kDefaultTileCacheBytes,
                                                         TileContiguityStrategy::Auto);
            kTilePxX = (shape.x > 0) ? shape.x : 32u;
            kTilePxY = (shape.y > 0) ? shape.y : 32u;
        }

        const uint32_t tiles_x = (width + kTilePxX - 1) / kTilePxX;
        const uint32_t tiles_y = (height + kTilePxY - 1) / kTilePxY;

        struct GatherCtx {
            const Scalar* src = nullptr;
            Scalar* dst = nullptr;
            const int32_t* offsets = nullptr;
            const Scalar* weights = nullptr;
            const uint8_t* mask = nullptr;
            const uint8_t* src_mask = nullptr;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t channels = 0;
            uint32_t count = 0;
            uint32_t tiles_x = 0;
            uint32_t tile_px_x = 0;
            uint32_t tile_px_y = 0;
            int32_t radius_x = 0;
            int32_t radius_y = 0;
            StencilBoundaryMode boundary = StencilBoundaryMode::Zero;
            bool normalize = false;
            TensorBackend* backend = nullptr;
        };

        auto gather_tiles = [](const void* vctx, uint32_t ty0, uint32_t ty1) {
            const auto* ctx = static_cast<const GatherCtx*>(vctx);
            auto* mem = dynamic_cast<InMemoryBackend*>(ctx->backend);
            if (!mem) return;
            static thread_local AbstractTensorPool tile_pool(tile_pool_options());
            static thread_local AbstractTensorPool::PooledTensor tile_buf;

            for (uint32_t ty = ty0; ty < ty1; ++ty) {
                const uint32_t y0 = ty * ctx->tile_px_y;
                const uint32_t y1 = std::min<uint32_t>(ctx->height, y0 + ctx->tile_px_y);
                const uint32_t rows = y1 - y0;
                for (uint32_t tx = 0; tx < ctx->tiles_x; ++tx) {
                    const uint32_t x0 = tx * ctx->tile_px_x;
                    const uint32_t x1 = std::min<uint32_t>(ctx->width, x0 + ctx->tile_px_x);
                    const uint32_t cols = x1 - x0;

                    const uint32_t halo_h = rows + static_cast<uint32_t>(ctx->radius_y * 2);
                    const uint32_t halo_w = cols + static_cast<uint32_t>(ctx->radius_x * 2);

                    TensorDesc tile_desc{};
                    tile_desc.dtype = DType;
                    tile_desc.layout = TensorLayout::Dense;
                    if (ctx->channels == 1) {
                        tile_desc.shape.dims = {halo_h, halo_w};
                    } else {
                        tile_desc.shape.dims = {halo_h, halo_w, ctx->channels};
                    }

                    if (!tile_buf.valid() || tile_buf.tensor().desc().shape.dims != tile_desc.shape.dims) {
                        tile_buf = tile_pool.acquire(tile_desc, ctx->backend);
                    }
                    if (!tile_buf.valid()) return;

                    void* tile_ptr_v = nullptr;
                    size_t tile_bytes = 0;
                    if (!mem->map(tile_buf.tensor().handle(), &tile_ptr_v, &tile_bytes)) return;
                    auto* tile = static_cast<Scalar*>(tile_ptr_v);

                    auto map_coord_local = [&](int64_t v, int64_t limit) -> int64_t {
                        if (limit <= 0) return -1;
                        if (v >= 0 && v < limit) return v;
                        switch (ctx->boundary) {
                            case StencilBoundaryMode::Zero:
                                return -1;
                            case StencilBoundaryMode::Clamp:
                                return std::clamp<int64_t>(v, 0, limit - 1);
                            case StencilBoundaryMode::Mirror: {
                                if (limit == 1) return 0;
                                int64_t period = (limit - 1) * 2;
                                int64_t m = v % period;
                                if (m < 0) m += period;
                                if (m >= limit) m = period - m;
                                return m;
                            }
                            case StencilBoundaryMode::Wrap: {
                                int64_t m = v % limit;
                                if (m < 0) m += limit;
                                return m;
                            }
                            default:
                                return -1;
                        }
                    };

                    const uint32_t tile_row_stride = halo_w * ctx->channels;
                    for (uint32_t tyh = 0; tyh < halo_h; ++tyh) {
                        const int64_t sy = static_cast<int64_t>(y0) + static_cast<int64_t>(tyh) - ctx->radius_y;
                        const int64_t my = map_coord_local(sy, ctx->height);
                        for (uint32_t txh = 0; txh < halo_w; ++txh) {
                            const int64_t sx = static_cast<int64_t>(x0) + static_cast<int64_t>(txh) - ctx->radius_x;
                            const int64_t mx = map_coord_local(sx, ctx->width);
                            const uint64_t tile_idx = (static_cast<uint64_t>(tyh) * halo_w + txh) * ctx->channels;
                            if (my < 0 || mx < 0) {
                                for (uint32_t c = 0; c < ctx->channels; ++c) {
                                    tile[tile_idx + c] = static_cast<Scalar>(0);
                                }
                                continue;
                            }
                            if (ctx->src_mask) {
                                const uint64_t m_idx = static_cast<uint64_t>(my) * ctx->width +
                                                       static_cast<uint64_t>(mx);
                                if (ctx->src_mask[m_idx] == 0u) {
                                    for (uint32_t c = 0; c < ctx->channels; ++c) {
                                        tile[tile_idx + c] = static_cast<Scalar>(0);
                                    }
                                    continue;
                                }
                            }
                            const uint64_t src_idx = (static_cast<uint64_t>(my) * ctx->width +
                                                      static_cast<uint64_t>(mx)) * ctx->channels;
                            for (uint32_t c = 0; c < ctx->channels; ++c) {
                                tile[tile_idx + c] = ctx->src[src_idx + c];
                            }
                        }
                    }

                    static thread_local std::vector<int64_t> offset_delta;
                    offset_delta.resize(ctx->count);
                    for (uint32_t i = 0; i < ctx->count; ++i) {
                        const int32_t dx = ctx->offsets[i * 2 + 0];
                        const int32_t dy = ctx->offsets[i * 2 + 1];
                        offset_delta[i] = (static_cast<int64_t>(dy) * static_cast<int64_t>(halo_w) +
                                           static_cast<int64_t>(dx)) * static_cast<int64_t>(ctx->channels);
                    }

                    for (uint32_t ly = 0; ly < rows; ++ly) {
                        const uint32_t y = y0 + ly;
                        for (uint32_t lx = 0; lx < cols; ++lx) {
                            const uint32_t x = x0 + lx;
                            if (ctx->mask) {
                                const uint8_t m = ctx->mask[static_cast<uint64_t>(y) * ctx->width + x];
                                if (m == 0u) continue;
                            }
                            const uint64_t out_idx = (static_cast<uint64_t>(y) * ctx->width + x) * ctx->channels;
                            const int64_t base = (static_cast<int64_t>(ly + ctx->radius_y) *
                                                  static_cast<int64_t>(halo_w) +
                                                  static_cast<int64_t>(lx + ctx->radius_x)) *
                                                 static_cast<int64_t>(ctx->channels);

                            if (!ctx->normalize) {
                                for (uint32_t c = 0; c < ctx->channels; ++c) {
                                    Scalar acc = static_cast<Scalar>(0);
                                    const int64_t base_c = base + static_cast<int64_t>(c);
                                    for (uint32_t i = 0; i < ctx->count; ++i) {
                                        const int64_t tile_idx = base_c + offset_delta[i];
                                        acc += tile[static_cast<uint64_t>(tile_idx)] * ctx->weights[i];
                                    }
                                    ctx->dst[out_idx + c] = acc;
                                }
                            } else {
                                for (uint32_t c = 0; c < ctx->channels; ++c) {
                                    Scalar acc = static_cast<Scalar>(0);
                                    Scalar sum_w = static_cast<Scalar>(0);
                                    for (uint32_t i = 0; i < ctx->count; ++i) {
                                        const int32_t dx = ctx->offsets[i * 2 + 0];
                                        const int32_t dy = ctx->offsets[i * 2 + 1];
                                        const int64_t sx = static_cast<int64_t>(x) + dx;
                                        const int64_t sy = static_cast<int64_t>(y) + dy;
                                        const int64_t mx = map_coord_local(sx, ctx->width);
                                        const int64_t my = map_coord_local(sy, ctx->height);
                                        if (mx < 0 || my < 0) continue;
                                        if (ctx->src_mask) {
                                            const uint64_t m_idx = static_cast<uint64_t>(my) * ctx->width +
                                                                   static_cast<uint64_t>(mx);
                                            if (ctx->src_mask[m_idx] == 0u) continue;
                                        }
                                        const uint32_t tyi = static_cast<uint32_t>(static_cast<int32_t>(ly) + ctx->radius_y + dy);
                                        const uint32_t txi = static_cast<uint32_t>(static_cast<int32_t>(lx) + ctx->radius_x + dx);
                                        const uint64_t tile_idx =
                                            (static_cast<uint64_t>(tyi) * halo_w + txi) * ctx->channels + c;
                                        const Scalar w = ctx->weights[i];
                                        acc += tile[tile_idx] * w;
                                        sum_w += w;
                                    }
                                    if (sum_w != static_cast<Scalar>(0)) {
                                        acc /= sum_w;
                                    }
                                    ctx->dst[out_idx + c] = acc;
                                }
                            }
                        }
                    }

                    mem->unmap(tile_buf.tensor().handle());
                }
            }
        };

        GatherCtx ctx{};
        ctx.src = bmap.data;
        ctx.dst = omap.data;
        ctx.offsets = off_ptr;
        ctx.weights = w_ptr;
        ctx.mask = target_mask ? static_cast<const uint8_t*>(mmap.data) : nullptr;
        ctx.src_mask = source_mask ? static_cast<const uint8_t*>(smap.data) : nullptr;
        ctx.width = width;
        ctx.height = height;
        ctx.channels = channels;
        ctx.count = count;
        ctx.tiles_x = tiles_x;
        ctx.tile_px_x = kTilePxX;
        ctx.tile_px_y = kTilePxY;
        ctx.radius_x = radius_x;
        ctx.radius_y = radius_y;
        ctx.boundary = boundary;
        ctx.normalize = normalize;
        ctx.backend = base.backend();

        submit_row_jobs(tensor_op_pool(), gather_tiles, &ctx, 0, tiles_y);

        bmap.unmap();
        omap.unmap();
        omap_off.unmap();
        wmap.unmap();
        if (target_mask) mmap.unmap();
        if (source_mask) smap.unmap();
        return true;
    }

    static bool gather_stencil_nd(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary,
                                  const AbstractTensor* target_mask,
                                  const AbstractTensor* source_mask,
                                  bool normalize) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !stencil.offsets.valid() || !stencil.weights.valid()) return false;
        if (base.backend() != stencil.offsets.backend() || base.backend() != stencil.weights.backend()) return false;
        if (target_mask && base.backend() != target_mask->backend()) return false;
        if (source_mask && base.backend() != source_mask->backend()) return false;

        const TensorDesc& bd = base.desc();
        const TensorDesc& od = stencil.offsets.desc();
        const TensorDesc& wd = stencil.weights.desc();
        if (bd.dtype != DType || wd.dtype != DType) return false;
        if (od.dtype != TensorDType::I32) return false;
        if (bd.layout != TensorLayout::Dense || od.layout != TensorLayout::Dense || wd.layout != TensorLayout::Dense)
            return false;
        if (od.shape.dims.size() != 2) return false;
        const uint32_t count = od.shape.dims[0];
        const uint32_t dims = od.shape.dims[1];
        if (dims == 0) return false;
        if (wd.shape.dims.size() != 1 || wd.shape.dims[0] != count) return false;

        const size_t rank = bd.shape.dims.size();
        if (rank < dims || rank > dims + 1) return false;
        const uint32_t channels = (rank == dims + 1) ? bd.shape.dims.back() : 1u;

        if (target_mask) {
            const TensorDesc& md = target_mask->desc();
            if (md.dtype != TensorDType::Bool || md.layout != TensorLayout::Dense) return false;
            if (md.shape.dims.size() != dims) return false;
            for (uint32_t d = 0; d < dims; ++d) {
                if (md.shape.dims[d] != bd.shape.dims[d]) return false;
            }
        }
        if (source_mask) {
            const TensorDesc& md = source_mask->desc();
            if (md.dtype != TensorDType::Bool || md.layout != TensorLayout::Dense) return false;
            if (md.shape.dims.size() != dims) return false;
            for (uint32_t d = 0; d < dims; ++d) {
                if (md.shape.dims[d] != bd.shape.dims[d]) return false;
            }
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDenseI32 omap_off = map_dense_i32(stencil.offsets);
        MappedDense wmap = map_dense(stencil.weights);
        MappedDenseU8 mmap{};
        if (target_mask) {
            mmap = map_dense_u8(*target_mask);
        }
        MappedDenseU8 smap{};
        if (source_mask) {
            smap = map_dense_u8(*source_mask);
        }
        if (!bmap.ok || !omap.ok || !omap_off.ok || !wmap.ok ||
            (target_mask && !mmap.ok) || (source_mask && !smap.ok)) {
            bmap.unmap();
            omap.unmap();
            omap_off.unmap();
            wmap.unmap();
            if (target_mask) mmap.unmap();
            if (source_mask) smap.unmap();
            return false;
        }

        std::vector<uint64_t> strides;
        if (!compute_dense_strides_u64(bd.shape.dims, strides)) {
            bmap.unmap();
            omap.unmap();
            omap_off.unmap();
            wmap.unmap();
            if (target_mask) mmap.unmap();
            return false;
        }

        auto map_coord = [&](int64_t v, int64_t limit) -> int64_t {
            if (limit <= 0) return -1;
            if (v >= 0 && v < limit) return v;
            switch (boundary) {
                case StencilBoundaryMode::Zero:
                    return -1;
                case StencilBoundaryMode::Clamp:
                    return std::clamp<int64_t>(v, 0, limit - 1);
                case StencilBoundaryMode::Mirror: {
                    if (limit == 1) return 0;
                    int64_t period = (limit - 1) * 2;
                    int64_t m = v % period;
                    if (m < 0) m += period;
                    if (m >= limit) m = period - m;
                    return m;
                }
                case StencilBoundaryMode::Wrap: {
                    int64_t m = v % limit;
                    if (m < 0) m += limit;
                    return m;
                }
                default:
                    return -1;
            }
        };

        const auto* off_ptr = omap_off.data;
        const Scalar* w_ptr = wmap.data;

        uint64_t spatial_elems = 1;
        for (uint32_t d = 0; d < dims; ++d) {
            spatial_elems *= bd.shape.dims[d];
        }

        auto compute_range = [&](uint64_t begin, uint64_t end) {
            std::vector<uint32_t> coords(dims, 0);
            std::vector<int64_t> src_coords(dims, 0);
            for (uint64_t lin = begin; lin < end; ++lin) {
                uint64_t rem = lin;
                for (size_t d = dims; d-- > 0;) {
                    const uint32_t dim = bd.shape.dims[d];
                    coords[d] = (dim == 0) ? 0u : static_cast<uint32_t>(rem % dim);
                    rem /= dim == 0 ? 1u : dim;
                }

                if (target_mask) {
                    const uint8_t m = mmap.data[lin];
                    if (m == 0u) continue;
                }

                uint64_t out_idx = 0;
                for (uint32_t d = 0; d < dims; ++d) {
                    out_idx += static_cast<uint64_t>(coords[d]) * strides[d];
                }

                for (uint32_t c = 0; c < channels; ++c) {
                    Scalar acc = static_cast<Scalar>(0);
                    Scalar sum_w = static_cast<Scalar>(0);
                    for (uint32_t i = 0; i < count; ++i) {
                        bool oob = false;
                        for (uint32_t d = 0; d < dims; ++d) {
                            const int64_t v = static_cast<int64_t>(coords[d]) +
                                              static_cast<int64_t>(off_ptr[i * dims + d]);
                            const int64_t mv = map_coord(v, bd.shape.dims[d]);
                            if (mv < 0) {
                                oob = true;
                                break;
                            }
                            src_coords[d] = mv;
                        }
                        if (oob) continue;
                        uint64_t src_idx = 0;
                        for (uint32_t d = 0; d < dims; ++d) {
                            src_idx += static_cast<uint64_t>(src_coords[d]) * strides[d];
                        }
                        if (source_mask) {
                            const uint8_t m = smap.data[src_idx];
                            if (m == 0u) continue;
                        }
                        const Scalar w = w_ptr[i];
                        acc += bmap.data[src_idx + c] * w;
                        if (normalize) sum_w += w;
                    }
                    if (normalize && sum_w != static_cast<Scalar>(0)) {
                        acc /= sum_w;
                    }
                    omap.data[out_idx + c] = acc;
                }
            }
        };

        nodus::ThreadPool* pool = tensor_op_pool();
        if (!pool || !should_parallelize(pool, spatial_elems)) {
            compute_range(0, spatial_elems);
        } else {
            const uint32_t max_jobs = pool->thread_count();
            const uint32_t job_count = std::min<uint32_t>(
                max_jobs,
                static_cast<uint32_t>(std::min<uint64_t>(spatial_elems, static_cast<uint64_t>(max_jobs))));
            const uint64_t chunk = (spatial_elems + job_count - 1) / job_count;
            struct RangeCtx {
                decltype(compute_range)* fn = nullptr;
                uint64_t begin = 0;
                uint64_t end = 0;
            };
            std::vector<RangeCtx> ranges(job_count);
            std::vector<nodus::ThreadPool::Job> jobs(job_count);
            auto job_fn = [](const nodus::ThreadPool::Job& job, uint32_t /*worker_id*/) {
                const auto* rc = static_cast<const RangeCtx*>(job.params);
                (*rc->fn)(rc->begin, rc->end);
            };
            uint32_t count_jobs = 0;
            for (uint32_t i = 0; i < job_count; ++i) {
                const uint64_t begin = static_cast<uint64_t>(i) * chunk;
                const uint64_t end = std::min<uint64_t>(spatial_elems, begin + chunk);
                if (begin >= end) break;
                ranges[count_jobs] = RangeCtx{&compute_range, begin, end};
                nodus::ThreadPool::Job job{};
                job.fn = job_fn;
                job.params = &ranges[count_jobs];
                job.params_size = static_cast<uint32_t>(sizeof(RangeCtx));
                jobs[count_jobs] = job;
                ++count_jobs;
            }
            auto batch = pool->submit_batch(jobs.data(), count_jobs);
            if (batch) batch->wait();
        }

        bmap.unmap();
        omap.unmap();
        omap_off.unmap();
        wmap.unmap();
        if (target_mask) mmap.unmap();
        if (source_mask) smap.unmap();
        return true;
    }

    static bool gather_footprint_2d(const AbstractTensor& base,
                                    const TensorFootprint2D& footprint,
                                    AbstractTensor* out,
                                    StencilOrientation orientation,
                                    StencilBoundaryMode boundary,
                                    const AbstractTensor* target_mask,
                                    const AbstractTensor* source_mask,
                                    bool normalize) {
        TensorStencil stencil{};
        if (!build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return gather_stencil_2d(base, stencil, out, boundary, target_mask, source_mask, normalize);
    }

    static bool scatter_stencil_2d(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary,
                                   bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (!stencil.offsets.valid() || !stencil.weights.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
        if (base.backend() != stencil.offsets.backend() || base.backend() != stencil.weights.backend()) return false;

        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        const TensorDesc& od = stencil.offsets.desc();
        const TensorDesc& wd = stencil.weights.desc();
        if (bd.dtype != DType || vd.dtype != DType || wd.dtype != DType) return false;
        if (od.dtype != TensorDType::I32) return false;
        const bool points_match = (pd.dtype == DType);
        if (!points_match) {
            if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
                pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
                return false;
            }
        }
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense ||
            vd.layout != TensorLayout::Dense || od.layout != TensorLayout::Dense ||
            wd.layout != TensorLayout::Dense) {
            return false;
        }
        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1u;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        bool val_scalar = false;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            val_scalar = true;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count || vd.shape.dims[1] != channels) return false;
        } else {
            return false;
        }

        if (od.shape.dims.size() != 2 || od.shape.dims[1] != 2) return false;
        const uint32_t taps = od.shape.dims[0];
        if (wd.shape.dims.size() != 1 || wd.shape.dims[0] != taps) return false;

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense vmap = map_dense(values);
        MappedDenseI32 omap_off = map_dense_i32(stencil.offsets);
        MappedDense wmap = map_dense(stencil.weights);
        MappedDense pmap{};
        void* points_ptr = nullptr;
        size_t points_bytes = 0;
        if (points_match) {
            pmap = map_dense(points);
            points_ptr = pmap.data;
        } else {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
                bmap.unmap();
                omap.unmap();
                vmap.unmap();
                return false;
            }
        }
        if (!bmap.ok || !omap.ok || !vmap.ok || !omap_off.ok || !wmap.ok ||
            (points_match && !pmap.ok)) {
            bmap.unmap();
            omap.unmap();
            vmap.unmap();
            omap_off.unmap();
            wmap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }

        if constexpr (std::is_same_v<Scalar, float>) {
            if (!tensor_copy_f32_into(base, out)) {
                bmap.unmap();
                omap.unmap();
                vmap.unmap();
                omap_off.unmap();
                wmap.unmap();
                if (points_match) {
                    pmap.unmap();
                } else if (points_ptr) {
                    auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                    if (mem) mem->unmap(points.handle());
                }
                return false;
            }
            bmap.unmap();
            omap.unmap();
            MappedDense omap2 = map_dense_mut(*out);
            if (!omap2.ok) {
                vmap.unmap();
                omap_off.unmap();
                wmap.unmap();
                if (points_match) {
                    pmap.unmap();
                } else if (points_ptr) {
                    auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                    if (mem) mem->unmap(points.handle());
                }
                return false;
            }
            omap = omap2;
        } else {
            const uint64_t elems = bd.shape.element_count();
            std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
        }

        auto read_index = [&](uint32_t i, uint32_t col) -> int64_t {
            const uint64_t idx = static_cast<uint64_t>(i) * pcols + col;
            switch (pd.dtype) {
                case TensorDType::I32:
                    return static_cast<int64_t>(static_cast<const int32_t*>(points_ptr)[idx]);
                case TensorDType::I64:
                    return static_cast<const int64_t*>(points_ptr)[idx];
                case TensorDType::U32:
                    return static_cast<int64_t>(static_cast<const uint32_t*>(points_ptr)[idx]);
                case TensorDType::U64:
                    return static_cast<int64_t>(static_cast<const uint64_t*>(points_ptr)[idx]);
                default:
                    return 0;
            }
        };

        auto map_coord_local = [&](int64_t v, int64_t limit) -> int64_t {
            if (limit <= 0) return -1;
            if (v >= 0 && v < limit) return v;
            switch (boundary) {
                case StencilBoundaryMode::Zero:
                    return -1;
                case StencilBoundaryMode::Clamp:
                    return std::clamp<int64_t>(v, 0, limit - 1);
                case StencilBoundaryMode::Mirror: {
                    if (limit == 1) return 0;
                    int64_t period = (limit - 1) * 2;
                    int64_t m = v % period;
                    if (m < 0) m += period;
                    if (m >= limit) m = period - m;
                    return m;
                }
                case StencilBoundaryMode::Wrap: {
                    int64_t m = v % limit;
                    if (m < 0) m += limit;
                    return m;
                }
                default:
                    return -1;
            }
        };

        int32_t radius_x = 0;
        int32_t radius_y = 0;
        const int32_t* off_ptr = omap_off.data;
        const Scalar* w_ptr = wmap.data;
        for (uint32_t i = 0; i < taps; ++i) {
            radius_x = std::max(radius_x, std::abs(off_ptr[i * 2 + 0]));
            radius_y = std::max(radius_y, std::abs(off_ptr[i * 2 + 1]));
        }

        constexpr uint64_t kDefaultTileCacheBytes = 6ull * 1024ull * 1024ull * 1024ull;
        TileShape2D tile_shape = choose_tile_shape_2d_impl(bd, static_cast<uint32_t>(radius_x),
                                   static_cast<uint32_t>(radius_y), kDefaultTileCacheBytes,
                                   TileContiguityStrategy::Auto);
        const uint32_t tile_px_x = (tile_shape.x > 0) ? tile_shape.x : 32u;
        const uint32_t tile_px_y = (tile_shape.y > 0) ? tile_shape.y : 32u;
        const uint32_t tiles_x = (width + tile_px_x - 1) / tile_px_x;

        const bool use_affine = bd.slice.valid && bd.slice.has_affine;
        CoordBuffer coord_buf{};
        if (!acquire_coord_buffer(count, 2u, base.backend(), coord_buf)) {
            bmap.unmap();
            omap.unmap();
            vmap.unmap();
            omap_off.unmap();
            wmap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }
        int64_t* coords = coord_buf.coords_ptr;
        uint8_t* in_bounds = coord_buf.mask_ptr;
        const bool clamp_to_bounds = (boundary == StencilBoundaryMode::Clamp);
        convert_points_to_int_coords(pd,
                                     points_ptr,
                                     count,
                                     2u,
                                     points_match,
                                     use_affine,
                                     bd.slice.affine,
                                     bd.shape.dims.data(),
                                     clamp_to_bounds,
                                     coords,
                                     in_bounds);
        const bool use_span_tiling = span_tiling_enabled_from_env("NODUS_SCATTER_SPAN_TILING");
        if (use_span_tiling) {
            SpanOpConfig span_op{};
            span_op.aggregate = SpanAggregateOp::Add;
            span_op.use_stencil = true;
            span_op.clamp = clamp;
            if (span_scatter_stencil_2d<Scalar>(coords,
                                                in_bounds,
                                                count,
                                                width,
                                                height,
                                                channels,
                                                taps,
                                                off_ptr,
                                                w_ptr,
                                                val_scalar,
                                                vmap.data,
                                                omap.data,
                                                base.backend(),
                                                tile_px_x,
                                                tile_px_y,
                                                span_op)) {
                if (points_match) {
                    pmap.unmap();
                } else if (points_ptr) {
                    auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                    if (mem) mem->unmap(points.handle());
                }
                vmap.unmap();
                omap.unmap();
                omap_off.unmap();
                wmap.unmap();
                release_coord_buffer(base.backend(), coord_buf);
                return true;
            }
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (clamp && !in_bounds[i]) {
                continue;
            }

            const int64_t xi = coords[static_cast<size_t>(i) * 2 + 0];
            const int64_t yi = coords[static_cast<size_t>(i) * 2 + 1];

            const Scalar vscalar = val_scalar ? vmap.data[i] : static_cast<Scalar>(0);
            const Scalar* vrow = val_scalar ? nullptr : (vmap.data + static_cast<uint64_t>(i) * channels);

            for (uint32_t t = 0; t < taps; ++t) {
                const int32_t dx = off_ptr[t * 2 + 0];
                const int32_t dy = off_ptr[t * 2 + 1];
                const int64_t mx = map_coord_local(xi + dx, width);
                const int64_t my = map_coord_local(yi + dy, height);
                if (mx < 0 || my < 0) continue;
                const uint32_t tx = static_cast<uint32_t>(mx) / tile_px_x;
                const uint32_t ty = static_cast<uint32_t>(my) / tile_px_y;
                const uint64_t tile_id = static_cast<uint64_t>(ty) * tiles_x + tx;
                auto tile_lock = acquire_tile_lock(tile_id);
                const uint64_t base_idx = (static_cast<uint64_t>(my) * width +
                                           static_cast<uint64_t>(mx)) * channels;
                const Scalar w = w_ptr[t];
                if (channels == 1) {
                    const Scalar vv = val_scalar ? vscalar : vrow[0];
                    omap.data[base_idx] += vv * w;
                } else {
                    if (val_scalar) {
                        for (uint32_t c = 0; c < channels; ++c) {
                            omap.data[base_idx + c] += vscalar * w;
                        }
                    } else {
                        for (uint32_t c = 0; c < channels; ++c) {
                            omap.data[base_idx + c] += vrow[c] * w;
                        }
                    }
                }
            }
        }

        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        vmap.unmap();
        omap.unmap();
        omap_off.unmap();
        wmap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        return true;
    }

    static bool scatter_stencil_nd(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary,
                                   bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (!stencil.offsets.valid() || !stencil.weights.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
        if (base.backend() != stencil.offsets.backend() || base.backend() != stencil.weights.backend()) return false;

        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        const TensorDesc& od = stencil.offsets.desc();
        const TensorDesc& wd = stencil.weights.desc();
        if (bd.dtype != DType || vd.dtype != DType || wd.dtype != DType) return false;
        if (od.dtype != TensorDType::I32) return false;
        const bool points_match = (pd.dtype == DType);
        if (!points_match) {
            if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
                pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
                return false;
            }
        }
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense ||
            vd.layout != TensorLayout::Dense || od.layout != TensorLayout::Dense ||
            wd.layout != TensorLayout::Dense) {
            return false;
        }
        if (pd.shape.dims.size() != 2 || od.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t dims = pd.shape.dims[1];
        if (dims == 0) return false;
        if (od.shape.dims[1] != dims) return false;
        const uint32_t taps = od.shape.dims[0];
        if (wd.shape.dims.size() != 1 || wd.shape.dims[0] != taps) return false;

        const size_t out_rank = bd.shape.dims.size();
        if (out_rank < dims || out_rank > dims + 1) return false;
        const uint32_t channels = (out_rank == dims + 1) ? bd.shape.dims.back() : 1u;

        bool val_scalar = false;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            val_scalar = true;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count || vd.shape.dims[1] != channels) return false;
        } else {
            return false;
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense vmap = map_dense(values);
        MappedDenseI32 omap_off = map_dense_i32(stencil.offsets);
        MappedDense wmap = map_dense(stencil.weights);
        MappedDense pmap{};
        void* points_ptr = nullptr;
        size_t points_bytes = 0;
        if (points_match) {
            pmap = map_dense(points);
        } else {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
                bmap.unmap();
                omap.unmap();
                vmap.unmap();
                return false;
            }
        }
        if (!bmap.ok || !omap.ok || !vmap.ok || !omap_off.ok || !wmap.ok ||
            (points_match && !pmap.ok)) {
            bmap.unmap();
            omap.unmap();
            vmap.unmap();
            omap_off.unmap();
            wmap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }

        if constexpr (std::is_same_v<Scalar, float>) {
            if (!tensor_copy_f32_into(base, out)) {
                bmap.unmap();
                omap.unmap();
                vmap.unmap();
                omap_off.unmap();
                wmap.unmap();
                if (points_match) {
                    pmap.unmap();
                } else if (points_ptr) {
                    auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                    if (mem) mem->unmap(points.handle());
                }
                return false;
            }
            bmap.unmap();
            omap.unmap();
            MappedDense omap2 = map_dense_mut(*out);
            if (!omap2.ok) {
                vmap.unmap();
                omap_off.unmap();
                wmap.unmap();
                if (points_match) {
                    pmap.unmap();
                } else if (points_ptr) {
                    auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                    if (mem) mem->unmap(points.handle());
                }
                return false;
            }
            omap = omap2;
        } else {
            const uint64_t elems = bd.shape.element_count();
            std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
        }

        std::vector<uint64_t> elem_strides;
        if (!compute_strides_for_desc_u64(bd, elem_strides)) {
            vmap.unmap();
            omap.unmap();
            omap_off.unmap();
            wmap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }

        std::vector<uint64_t> dense_strides;
        if (!compute_dense_strides_u64(std::vector<uint32_t>(bd.shape.dims.begin(),
                                                             bd.shape.dims.begin() + dims),
                                       dense_strides)) {
            vmap.unmap();
            omap.unmap();
            omap_off.unmap();
            wmap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }

        constexpr uint64_t kDefaultTileCacheBytes = 6ull * 1024ull * 1024ull * 1024ull;
        const uint64_t tile_elems = std::max<uint64_t>(1u, choose_linear_tile_elems(bd, dims, kDefaultTileCacheBytes));

        auto map_coord = [&](int64_t v, int64_t limit) -> int64_t {
            if (limit <= 0) return -1;
            if (v >= 0 && v < limit) return v;
            switch (boundary) {
                case StencilBoundaryMode::Zero:
                    return -1;
                case StencilBoundaryMode::Clamp:
                    return std::clamp<int64_t>(v, 0, limit - 1);
                case StencilBoundaryMode::Mirror: {
                    if (limit == 1) return 0;
                    int64_t period = (limit - 1) * 2;
                    int64_t m = v % period;
                    if (m < 0) m += period;
                    if (m >= limit) m = period - m;
                    return m;
                }
                case StencilBoundaryMode::Wrap: {
                    int64_t m = v % limit;
                    if (m < 0) m += limit;
                    return m;
                }
                default:
                    return -1;
            }
        };

        const int32_t* off_ptr = omap_off.data;
        const Scalar* w_ptr = wmap.data;

        CoordBuffer coord_buf{};
        if (!acquire_coord_buffer(count, dims, base.backend(), coord_buf)) {
            vmap.unmap();
            omap.unmap();
            omap_off.unmap();
            wmap.unmap();
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return false;
        }
        int64_t* coords = coord_buf.coords_ptr;
        std::vector<int64_t> target(dims, 0);
        const bool clamp_to_bounds = (boundary == StencilBoundaryMode::Clamp);
        convert_points_to_int_coords(pd,
                         points_ptr,
                         count,
                         dims,
                         points_match,
                         false,
                         nullptr,
                         bd.shape.dims.data(),
                         clamp_to_bounds,
                                     coords,
                                     nullptr);

        for (uint32_t i = 0; i < count; ++i) {
            const int64_t* coord = coords + static_cast<size_t>(i) * dims;

            const Scalar vscalar = val_scalar ? vmap.data[i] : static_cast<Scalar>(0);
            const Scalar* vrow = val_scalar ? nullptr : (vmap.data + static_cast<uint64_t>(i) * channels);

            for (uint32_t t = 0; t < taps; ++t) {
                bool oob = false;
                for (uint32_t d = 0; d < dims; ++d) {
                    const int64_t v = coord[d] + static_cast<int64_t>(off_ptr[t * dims + d]);
                    const int64_t mv = map_coord(v, bd.shape.dims[d]);
                    if (mv < 0) {
                        oob = true;
                        break;
                    }
                    target[d] = mv;
                }
                if (oob) {
                    if (clamp) continue;
                    continue;
                }

                uint64_t out_idx = 0;
                uint64_t gli = 0;
                for (uint32_t d = 0; d < dims; ++d) {
                    out_idx += static_cast<uint64_t>(target[d]) * elem_strides[d];
                    gli += static_cast<uint64_t>(target[d]) * dense_strides[d];
                }
                const uint64_t tile_id = gli / tile_elems;
                auto tile_lock = acquire_tile_lock(tile_id);

                const Scalar w = w_ptr[t];
                if (channels == 1) {
                    const Scalar vv = val_scalar ? vscalar : vrow[0];
                    omap.data[out_idx] += vv * w;
                } else {
                    if (val_scalar) {
                        for (uint32_t c = 0; c < channels; ++c) {
                            omap.data[out_idx + c] += vscalar * w;
                        }
                    } else {
                        for (uint32_t c = 0; c < channels; ++c) {
                            omap.data[out_idx + c] += vrow[c] * w;
                        }
                    }
                }
            }
        }

        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        vmap.unmap();
        omap.unmap();
        omap_off.unmap();
        wmap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        return true;
    }

    static bool scatter_footprint_2d(const AbstractTensor& base,
                                     const AbstractTensor& points,
                                     const AbstractTensor& values,
                                     const TensorFootprint2D& footprint,
                                     AbstractTensor* out,
                                     StencilOrientation orientation,
                                     StencilBoundaryMode boundary,
                                     bool clamp) {
        TensorStencil stencil{};
        if (!build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return scatter_stencil_2d(base, points, values, stencil, out, boundary, clamp);
    }

    static bool scatter_probe_csr(const AbstractTensor& base,
                                  const TensorProbeKernel& probe,
                                  const TensorCSR& targets_per_center,
                                  AbstractTensor* out,
                                  bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !probe.offsets.valid() || !probe.weights.valid()) return false;
        if (!targets_per_center.row_ptr.valid() || !targets_per_center.col_gli.valid() ||
            !targets_per_center.edge_weight.valid()) return false;
        if (base.backend() != probe.offsets.backend() || base.backend() != probe.weights.backend()) return false;
        if (base.backend() != targets_per_center.row_ptr.backend() ||
            base.backend() != targets_per_center.col_gli.backend() ||
            base.backend() != targets_per_center.edge_weight.backend()) {
            return false;
        }

        const TensorDesc& bd = base.desc();
        const TensorDesc& od = probe.offsets.desc();
        const TensorDesc& wd = probe.weights.desc();
        if (bd.dtype != DType || wd.dtype != DType) return false;
        if (od.dtype != TensorDType::I32) return false;
        if (bd.layout != TensorLayout::Dense || od.layout != TensorLayout::Dense || wd.layout != TensorLayout::Dense)
            return false;
        if (od.shape.dims.size() != 2) return false;
        const uint32_t count = od.shape.dims[0];
        const uint32_t dims = od.shape.dims[1];
        if (dims == 0) return false;
        if (wd.shape.dims.size() != 1 || wd.shape.dims[0] != count) return false;

        const size_t rank = bd.shape.dims.size();
        if (rank < dims || rank > dims + 1) return false;
        const uint32_t channels = (rank == dims + 1) ? bd.shape.dims.back() : 1u;

        const TensorDesc& rpd = targets_per_center.row_ptr.desc();
        const TensorDesc& cpd = targets_per_center.col_gli.desc();
        const TensorDesc& epd = targets_per_center.edge_weight.desc();
        if (rpd.dtype != TensorDType::I32 || cpd.dtype != TensorDType::I32 || epd.dtype != DType) return false;
        if (rpd.layout != TensorLayout::Dense || cpd.layout != TensorLayout::Dense || epd.layout != TensorLayout::Dense)
            return false;
        if (rpd.shape.dims.size() != 1 || cpd.shape.dims.size() != 1 || epd.shape.dims.size() != 1) return false;
        if (cpd.shape.dims[0] != epd.shape.dims[0]) return false;

        uint32_t rows = targets_per_center.rows;
        if (rows == 0) {
            if (rpd.shape.dims[0] == 0) return false;
            rows = rpd.shape.dims[0] - 1;
        }
        if (rpd.shape.dims[0] != rows + 1) return false;

        uint64_t spatial_elems = 1;
        for (uint32_t d = 0; d < dims; ++d) {
            spatial_elems *= bd.shape.dims[d];
        }
        if (targets_per_center.cols != 0 && targets_per_center.cols != spatial_elems) return false;
        if (rows > spatial_elems) return false;

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDenseI32 omap_off = map_dense_i32(probe.offsets);
        MappedDense wmap = map_dense(probe.weights);
        MappedDenseI32 rpmap = map_dense_i32(targets_per_center.row_ptr);
        MappedDenseI32 cpmap = map_dense_i32(targets_per_center.col_gli);
        MappedDense epmap = map_dense(targets_per_center.edge_weight);
        if (!bmap.ok || !omap.ok || !omap_off.ok || !wmap.ok || !rpmap.ok || !cpmap.ok || !epmap.ok) {
            bmap.unmap();
            omap.unmap();
            omap_off.unmap();
            wmap.unmap();
            rpmap.unmap();
            cpmap.unmap();
            epmap.unmap();
            return false;
        }

        std::vector<uint64_t> elem_strides;
        if (!compute_strides_for_desc_u64(bd, elem_strides)) {
            bmap.unmap();
            omap.unmap();
            omap_off.unmap();
            wmap.unmap();
            rpmap.unmap();
            cpmap.unmap();
            epmap.unmap();
            return false;
        }

        const auto* off_ptr = omap_off.data;
        const Scalar* w_ptr = wmap.data;
        const auto* row_ptr = rpmap.data;
        const auto* col_ptr = cpmap.data;
        const Scalar* edge_ptr = epmap.data;
        constexpr uint64_t kDefaultTileCacheBytes = 6ull * 1024ull * 1024ull * 1024ull;
        const uint64_t tile_elems = std::max<uint64_t>(1u, choose_linear_tile_elems(bd, dims, kDefaultTileCacheBytes));

        std::vector<uint32_t> coords(dims, 0);
        std::vector<int64_t> src_coords(dims, 0);
        std::vector<Scalar> probe_val(channels, static_cast<Scalar>(0));
        std::vector<Scalar> probe_edge(channels, static_cast<Scalar>(0));

        auto decode_coords = [&](uint64_t gli) {
            uint64_t rem = gli;
            for (size_t d = dims; d-- > 0;) {
                const uint32_t dim = bd.shape.dims[d];
                coords[d] = (dim == 0) ? 0u : static_cast<uint32_t>(rem % dim);
                rem /= dim == 0 ? 1u : dim;
            }
        };

        auto elem_offset_from_coords = [&](const std::vector<int64_t>& in_coords, uint64_t* out_idx) -> bool {
            uint64_t idx = 0;
            for (uint32_t d = 0; d < dims; ++d) {
                if (in_coords[d] < 0 || in_coords[d] >= static_cast<int64_t>(bd.shape.dims[d])) return false;
                idx += static_cast<uint64_t>(in_coords[d]) * elem_strides[d];
            }
            *out_idx = idx;
            return true;
        };

        auto elem_offset_from_gli_spatial = [&](uint64_t gli, uint64_t* out_idx) -> bool {
            uint64_t rem = gli;
            uint64_t idx = 0;
            for (size_t d = dims; d-- > 0;) {
                const uint32_t dim = bd.shape.dims[d];
                if (dim == 0) return false;
                const uint32_t c = static_cast<uint32_t>(rem % dim);
                rem /= dim;
                idx += static_cast<uint64_t>(c) * elem_strides[d];
            }
            *out_idx = idx;
            return true;
        };

        const uint64_t channel_stride = (channels > 1) ? elem_strides[dims] : 1u;

        for (uint32_t row = 0; row < rows; ++row) {
            const uint64_t center_gli = row;
            if (center_gli >= spatial_elems) break;
            decode_coords(center_gli);

            for (uint32_t c = 0; c < channels; ++c) {
                probe_val[c] = static_cast<Scalar>(0);
            }

            for (uint32_t i = 0; i < count; ++i) {
                bool oob = false;
                for (uint32_t d = 0; d < dims; ++d) {
                    const int64_t v = static_cast<int64_t>(coords[d]) +
                                      static_cast<int64_t>(off_ptr[i * dims + d]);
                    if (v < 0 || v >= static_cast<int64_t>(bd.shape.dims[d])) {
                        oob = true;
                        break;
                    }
                    src_coords[d] = v;
                }
                if (oob) {
                    if (clamp) continue;
                    continue;
                }
                uint64_t src_idx = 0;
                if (!elem_offset_from_coords(src_coords, &src_idx)) continue;
                const Scalar w = w_ptr[i];
                for (uint32_t c = 0; c < channels; ++c) {
                    probe_val[c] += bmap.data[src_idx + c * channel_stride] * w;
                }
            }

            const int32_t begin = row_ptr[row];
            const int32_t end = row_ptr[row + 1];
            if (begin < 0 || end < begin) continue;
            for (int32_t e = begin; e < end; ++e) {
                const uint64_t target_gli = static_cast<uint64_t>(col_ptr[e]);
                if (target_gli >= spatial_elems) continue;
                const uint64_t tile_id = target_gli / tile_elems;
                auto tile_lock = acquire_tile_lock(tile_id);
                uint64_t out_idx = 0;
                if (!elem_offset_from_gli_spatial(target_gli, &out_idx)) continue;
                const Scalar ew = edge_ptr[e];
                for (uint32_t c = 0; c < channels; ++c) {
                    omap.data[out_idx + c * channel_stride] += probe_val[c] * ew;
                }
            }
        }

        bmap.unmap();
        omap.unmap();
        omap_off.unmap();
        wmap.unmap();
        rpmap.unmap();
        cpmap.unmap();
        epmap.unmap();
        return true;
    }

    static bool gather_probe_csr(const AbstractTensor& base,
                                 const TensorProbeKernel& probe,
                                 const TensorCSR& centers_per_target,
                                 AbstractTensor* out,
                                 bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !probe.offsets.valid() || !probe.weights.valid()) return false;
        if (!centers_per_target.row_ptr.valid() || !centers_per_target.col_gli.valid() ||
            !centers_per_target.edge_weight.valid()) return false;
        if (base.backend() != probe.offsets.backend() || base.backend() != probe.weights.backend()) return false;
        if (base.backend() != centers_per_target.row_ptr.backend() ||
            base.backend() != centers_per_target.col_gli.backend() ||
            base.backend() != centers_per_target.edge_weight.backend()) {
            return false;
        }

        const TensorDesc& bd = base.desc();
        const TensorDesc& od = probe.offsets.desc();
        const TensorDesc& wd = probe.weights.desc();
        if (bd.dtype != DType || wd.dtype != DType) return false;
        if (od.dtype != TensorDType::I32) return false;
        if (bd.layout != TensorLayout::Dense || od.layout != TensorLayout::Dense || wd.layout != TensorLayout::Dense)
            return false;
        if (od.shape.dims.size() != 2) return false;
        const uint32_t count = od.shape.dims[0];
        const uint32_t dims = od.shape.dims[1];
        if (dims == 0) return false;
        if (wd.shape.dims.size() != 1 || wd.shape.dims[0] != count) return false;

        const size_t rank = bd.shape.dims.size();
        if (rank < dims || rank > dims + 1) return false;
        const uint32_t channels = (rank == dims + 1) ? bd.shape.dims.back() : 1u;

        const TensorDesc& rpd = centers_per_target.row_ptr.desc();
        const TensorDesc& cpd = centers_per_target.col_gli.desc();
        const TensorDesc& epd = centers_per_target.edge_weight.desc();
        if (rpd.dtype != TensorDType::I32 || cpd.dtype != TensorDType::I32 || epd.dtype != DType) return false;
        if (rpd.layout != TensorLayout::Dense || cpd.layout != TensorLayout::Dense || epd.layout != TensorLayout::Dense)
            return false;
        if (rpd.shape.dims.size() != 1 || cpd.shape.dims.size() != 1 || epd.shape.dims.size() != 1) return false;
        if (cpd.shape.dims[0] != epd.shape.dims[0]) return false;

        uint32_t rows = centers_per_target.rows;
        if (rows == 0) {
            if (rpd.shape.dims[0] == 0) return false;
            rows = rpd.shape.dims[0] - 1;
        }
        if (rpd.shape.dims[0] != rows + 1) return false;

        uint64_t spatial_elems = 1;
        for (uint32_t d = 0; d < dims; ++d) {
            spatial_elems *= bd.shape.dims[d];
        }
        if (centers_per_target.cols != 0 && centers_per_target.cols != spatial_elems) return false;
        if (rows > spatial_elems) return false;

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDenseI32 omap_off = map_dense_i32(probe.offsets);
        MappedDense wmap = map_dense(probe.weights);
        MappedDenseI32 rpmap = map_dense_i32(centers_per_target.row_ptr);
        MappedDenseI32 cpmap = map_dense_i32(centers_per_target.col_gli);
        MappedDense epmap = map_dense(centers_per_target.edge_weight);
        if (!bmap.ok || !omap.ok || !omap_off.ok || !wmap.ok || !rpmap.ok || !cpmap.ok || !epmap.ok) {
            bmap.unmap();
            omap.unmap();
            omap_off.unmap();
            wmap.unmap();
            rpmap.unmap();
            cpmap.unmap();
            epmap.unmap();
            return false;
        }

        std::vector<uint64_t> elem_strides;
        if (!compute_strides_for_desc_u64(bd, elem_strides)) {
            bmap.unmap();
            omap.unmap();
            omap_off.unmap();
            wmap.unmap();
            rpmap.unmap();
            cpmap.unmap();
            epmap.unmap();
            return false;
        }

        const auto* off_ptr = omap_off.data;
        const Scalar* w_ptr = wmap.data;
        const auto* row_ptr = rpmap.data;
        const auto* col_ptr = cpmap.data;
        const Scalar* edge_ptr = epmap.data;

        const uint64_t channel_stride = (channels > 1) ? elem_strides[dims] : 1u;

        std::vector<Scalar> probe_cache(static_cast<size_t>(spatial_elems) * channels,
                                        static_cast<Scalar>(0));
        const std::vector<uint32_t> spatial_shape(bd.shape.dims.begin(), bd.shape.dims.begin() + dims);

        struct ProbeCtx {
            const Scalar* src = nullptr;
            const int32_t* offsets = nullptr;
            const Scalar* weights = nullptr;
            Scalar* probe = nullptr;
            const uint64_t* elem_strides = nullptr;
            const uint32_t* shape = nullptr;
            uint32_t dims = 0;
            uint32_t channels = 0;
            uint32_t count = 0;
            uint64_t spatial_elems = 0;
            uint64_t channel_stride = 0;
            bool clamp = false;
        };

        ProbeCtx pctx{};
        pctx.src = bmap.data;
        pctx.offsets = off_ptr;
        pctx.weights = w_ptr;
        pctx.probe = probe_cache.data();
        pctx.elem_strides = elem_strides.data();
        pctx.shape = spatial_shape.data();
        pctx.dims = dims;
        pctx.channels = channels;
        pctx.count = count;
        pctx.spatial_elems = spatial_elems;
        pctx.channel_stride = channel_stride;
        pctx.clamp = clamp;

        auto compute_probe = [](const void* vctx, uint32_t begin, uint32_t end) {
            const auto* ctx = static_cast<const ProbeCtx*>(vctx);
            std::vector<uint32_t> coords(ctx->dims, 0);
            std::vector<int64_t> src_coords(ctx->dims, 0);

            for (uint32_t row = begin; row < end; ++row) {
                const uint64_t center_gli = row;
                if (center_gli >= ctx->spatial_elems) break;

                uint64_t rem = center_gli;
                for (size_t d = ctx->dims; d-- > 0;) {
                    const uint32_t dim = ctx->shape[d];
                    coords[d] = (dim == 0) ? 0u : static_cast<uint32_t>(rem % dim);
                    rem /= dim == 0 ? 1u : dim;
                }

                const uint64_t probe_base = center_gli * ctx->channels;
                for (uint32_t c = 0; c < ctx->channels; ++c) {
                    ctx->probe[probe_base + c] = static_cast<Scalar>(0);
                }

                for (uint32_t i = 0; i < ctx->count; ++i) {
                    bool oob = false;
                    for (uint32_t d = 0; d < ctx->dims; ++d) {
                        const int64_t v = static_cast<int64_t>(coords[d]) +
                                          static_cast<int64_t>(ctx->offsets[i * ctx->dims + d]);
                        if (v < 0 || v >= static_cast<int64_t>(ctx->shape[d])) {
                            oob = true;
                            break;
                        }
                        src_coords[d] = v;
                    }
                    if (oob) {
                        if (ctx->clamp) continue;
                        continue;
                    }
                    uint64_t src_idx = 0;
                    for (uint32_t d = 0; d < ctx->dims; ++d) {
                        src_idx += static_cast<uint64_t>(src_coords[d]) * ctx->elem_strides[d];
                    }
                    const Scalar w = ctx->weights[i];
                    for (uint32_t c = 0; c < ctx->channels; ++c) {
                        ctx->probe[probe_base + c] +=
                            ctx->src[src_idx + c * ctx->channel_stride] * w;
                    }
                }
            }
        };

        nodus::ThreadPool* pool = tensor_op_pool();
        if (!pool || !should_parallelize(pool, spatial_elems) ||
            spatial_elems > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            compute_probe(&pctx, 0u, static_cast<uint32_t>(std::min<uint64_t>(spatial_elems, std::numeric_limits<uint32_t>::max())));
        } else {
            submit_row_jobs(pool, compute_probe, &pctx, 0u, static_cast<uint32_t>(spatial_elems));
        }

        struct GatherCtx {
            const int32_t* row_ptr = nullptr;
            const int32_t* col_ptr = nullptr;
            const Scalar* edge_ptr = nullptr;
            const Scalar* probe = nullptr;
            Scalar* out = nullptr;
            const uint64_t* elem_strides = nullptr;
            const uint32_t* shape = nullptr;
            uint32_t dims = 0;
            uint32_t channels = 0;
            uint64_t spatial_elems = 0;
            uint64_t channel_stride = 0;
        };

        GatherCtx gctx{};
        gctx.row_ptr = row_ptr;
        gctx.col_ptr = col_ptr;
        gctx.edge_ptr = edge_ptr;
        gctx.probe = probe_cache.data();
        gctx.out = omap.data;
        gctx.elem_strides = elem_strides.data();
        gctx.shape = spatial_shape.data();
        gctx.dims = dims;
        gctx.channels = channels;
        gctx.spatial_elems = spatial_elems;
        gctx.channel_stride = channel_stride;

        auto gather_rows = [](const void* vctx, uint32_t begin, uint32_t end) {
            const auto* ctx = static_cast<const GatherCtx*>(vctx);
            std::vector<Scalar> acc(ctx->channels, static_cast<Scalar>(0));
            for (uint32_t row = begin; row < end; ++row) {
                const uint64_t target_gli = row;
                if (target_gli >= ctx->spatial_elems) break;

                const int32_t r0 = ctx->row_ptr[row];
                const int32_t r1 = ctx->row_ptr[row + 1];
                if (r0 < 0 || r1 < r0) continue;
                std::fill(acc.begin(), acc.end(), static_cast<Scalar>(0));
                for (int32_t e = r0; e < r1; ++e) {
                    const uint64_t center_gli = static_cast<uint64_t>(ctx->col_ptr[e]);
                    if (center_gli >= ctx->spatial_elems) continue;
                    const uint64_t probe_base = center_gli * ctx->channels;
                    const Scalar ew = ctx->edge_ptr[e];
                    for (uint32_t c = 0; c < ctx->channels; ++c) {
                        acc[c] += ctx->probe[probe_base + c] * ew;
                    }
                }

                uint64_t out_idx = 0;
                uint64_t rem = target_gli;
                bool valid = true;
                for (size_t d = ctx->dims; d-- > 0;) {
                    const uint32_t dim = ctx->shape[d];
                    if (dim == 0) {
                        valid = false;
                        break;
                    }
                    const uint32_t c = static_cast<uint32_t>(rem % dim);
                    rem /= dim;
                    out_idx += static_cast<uint64_t>(c) * ctx->elem_strides[d];
                }
                if (!valid) continue;

                for (uint32_t c = 0; c < ctx->channels; ++c) {
                    ctx->out[out_idx + c * ctx->channel_stride] = acc[c];
                }
            }
        };

        submit_row_jobs(tensor_op_pool(), gather_rows, &gctx, 0, rows);

        bmap.unmap();
        omap.unmap();
        omap_off.unmap();
        wmap.unmap();
        rpmap.unmap();
        cpmap.unmap();
        epmap.unmap();
        return true;
    }

    static bool scatter_add_2d_kernel(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out,
                                      bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid() || !kernel.valid()) return false;
        if (base.backend() != points.backend() ||
            base.backend() != values.backend() ||
            base.backend() != kernel.backend()) {
            return false;
        }
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        const TensorDesc& kd = kernel.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType || kd.dtype != DType) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense ||
            vd.layout != TensorLayout::Dense || kd.layout != TensorLayout::Dense) {
            return false;
        }
        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;
        constexpr uint64_t kDefaultTileCacheBytes = 6ull * 1024ull * 1024ull * 1024ull;
        TileShape2D tile_shape = choose_tile_shape_2d_impl(bd, 0u, 0u, kDefaultTileCacheBytes,
                                   TileContiguityStrategy::Auto);
        const uint32_t tile_px_x = (tile_shape.x > 0) ? tile_shape.x : 32u;
        const uint32_t tile_px_y = (tile_shape.y > 0) ? tile_shape.y : 32u;
        const uint32_t tiles_x = (width + tile_px_x - 1) / tile_px_x;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        uint32_t kcols = 1;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            kcols = 1;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            kcols = vd.shape.dims[1];
        } else {
            return false;
        }

        if (kd.shape.dims.size() != 2) return false;
        if (kd.shape.dims[0] != kcols) return false;
        if (kd.shape.dims[1] != channels) return false;

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap = map_dense(points);
        MappedDense vmap = map_dense(values);
        MappedDense kmap = map_dense(kernel);
        if (!bmap.ok || !omap.ok || !pmap.ok || !vmap.ok || !kmap.ok) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }

        const bool use_affine = bd.slice.valid && bd.slice.has_affine;

        CoordBuffer coord_buf{};
        if (!acquire_coord_buffer(count, 2u, base.backend(), coord_buf)) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }
        int64_t* coords = coord_buf.coords_ptr;
        uint8_t* in_bounds = coord_buf.mask_ptr;
        convert_points_to_int_coords(pd,
                         pmap.data,
                         count,
                         2u,
                         true,
                         use_affine,
                         bd.slice.affine,
                         bd.shape.dims.data(),
                         true,
                         coords,
                         in_bounds);

        if (!tensor_copy_f32_into(base, out)) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }
        bmap.unmap();
        omap.unmap();
        MappedDense omap2 = map_dense_mut(*out);
        if (!omap2.ok) {
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }
        omap = omap2;

        const bool use_span_tiling = span_tiling_enabled_from_env("NODUS_SCATTER_SPAN_TILING");
        if (use_span_tiling) {
            SpanOpConfig span_op{};
            span_op.aggregate = SpanAggregateOp::Add;
            if (span_scatter_add_2d_kernel<Scalar>(coords,
                                                   in_bounds,
                                                   count,
                                                   width,
                                                   height,
                                                   channels,
                                                   kcols,
                                                   kmap.data,
                                                   vmap.data,
                                                   (vd.shape.dims.size() == 1),
                                                   omap.data,
                                                   base.backend(),
                                                   tile_px_x,
                                                   tile_px_y,
                                                   span_op)) {
                bmap.unmap();
                omap.unmap();
                pmap.unmap();
                vmap.unmap();
                kmap.unmap();
                release_coord_buffer(base.backend(), coord_buf);
                return true;
            }
        }

        for (uint32_t i = 0; i < count; ++i) {
            if (!in_bounds[i]) {
                if (clamp) continue;
                continue;
            }
            const int64_t xi = coords[static_cast<size_t>(i) * 2 + 0];
            const int64_t yi = coords[static_cast<size_t>(i) * 2 + 1];
            const uint32_t tx = static_cast<uint32_t>(xi) / tile_px_x;
            const uint32_t ty = static_cast<uint32_t>(yi) / tile_px_y;
            const uint64_t tile_id = static_cast<uint64_t>(ty) * tiles_x + tx;
            auto tile_lock = acquire_tile_lock(tile_id);
            const uint64_t base_idx = (static_cast<uint64_t>(yi) * width + static_cast<uint64_t>(xi)) * channels;
            for (uint32_t c = 0; c < channels; ++c) {
                Scalar acc = static_cast<Scalar>(0);
                if (vd.shape.dims.size() == 1) {
                    const Scalar v = vmap.data[i];
                    acc = v * kmap.data[c];
                } else {
                    const Scalar* vrow = vmap.data + static_cast<uint64_t>(i) * kcols;
                    const Scalar* kcol = kmap.data + c;
                    for (uint32_t k = 0; k < kcols; ++k) {
                        acc += vrow[k] * kcol[k * channels];
                    }
                }
                omap.data[base_idx + c] += acc;
            }
        }

        bmap.unmap();
        omap.unmap();
        pmap.unmap();
        vmap.unmap();
        kmap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        return true;
    }

    static bool scatter_add_2d_kernel_fn_impl(const AbstractTensor& base,
                                              const AbstractTensor& points,
                                              const AbstractTensor& values,
                                              const std::function<Scalar(uint32_t, uint32_t, void*)>& kernel_fn,
                                              void* user,
                                              AbstractTensor* out,
                                              KernelDispatchPlan* out_plan,
                                              bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (!kernel_fn) return false;
        if (!out_plan) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
            return false;
        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        uint32_t kcols = 1;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            kcols = 1;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            kcols = vd.shape.dims[1];
        } else {
            return false;
        }

        TensorDesc kdesc{};
        kdesc.dtype = DType;
        kdesc.layout = TensorLayout::Dense;
        kdesc.shape.dims = {kcols, channels};
        AbstractTensor kernel = AbstractTensor::create(kdesc, base.backend());
        if (!kernel.valid()) return false;

        out_plan->entries.clear();
        out_plan->slot_ids.clear();
        KernelDispatchEntry entry{};
        entry.id = 0;
        entry.fn = nullptr;
        entry.user = user;
        out_plan->entries.push_back(entry);
        out_plan->slot_ids.resize(static_cast<size_t>(kcols) * channels, 0u);

        MappedDense kmap = map_dense_mut(kernel);
        if (!kmap.ok) {
            kmap.unmap();
            return false;
        }
        for (uint32_t k = 0; k < kcols; ++k) {
            for (uint32_t c = 0; c < channels; ++c) {
                kmap.data[static_cast<size_t>(k) * channels + c] = kernel_fn(k, c, user);
            }
        }
        kmap.unmap();

        // Single broadcast-style kernel op: reuse the standard kernel scatter once.
        return scatter_add_2d_kernel(base, points, values, kernel, out, clamp);
    }

    static bool scatter_add_2d_kernel_ptr_impl(const AbstractTensor& base,
                                              const AbstractTensor& points,
                                              const AbstractTensor& values,
                                              const AbstractTensor& kernel_ptrs,
                                              AbstractTensor* out,
                                              KernelDispatchPlan* out_plan,
                                              bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid() || !kernel_ptrs.valid()) return false;
        if (!out_plan) return false;
        if (base.backend() != points.backend() ||
            base.backend() != values.backend() ||
            base.backend() != kernel_ptrs.backend()) {
            return false;
        }
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        const TensorDesc& kd = kernel_ptrs.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType) return false;
        if (kd.dtype != TensorDType::Ptr) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense ||
            vd.layout != TensorLayout::Dense || kd.layout != TensorLayout::Dense) {
            return false;
        }
        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        uint32_t kcols = 1;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            kcols = 1;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            kcols = vd.shape.dims[1];
        } else {
            return false;
        }

        if (kd.shape.dims.size() != 2) return false;
        if (kd.shape.dims[0] != kcols) return false;
        if (kd.shape.dims[1] != channels) return false;

        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (!mem) return false;
        void* k_ptr_v = nullptr;
        size_t k_bytes = 0;
        if (!mem->map(kernel_ptrs.handle(), &k_ptr_v, &k_bytes)) {
            return false;
        }
        auto** k_ptrs = static_cast<void**>(k_ptr_v);

        using KernelEntry = TensorKernelPtrEntry<Scalar>;

        TensorDesc kdesc{};
        kdesc.dtype = DType;
        kdesc.layout = TensorLayout::Dense;
        kdesc.shape.dims = {kcols, channels};
        AbstractTensor kernel = AbstractTensor::create(kdesc, base.backend());
        if (!kernel.valid()) {
            mem->unmap(kernel_ptrs.handle());
            return false;
        }

        out_plan->entries.clear();
        out_plan->slot_ids.clear();

        MappedDense kmap = map_dense_mut(kernel);
        if (!kmap.ok) {
            mem->unmap(kernel_ptrs.handle());
            kmap.unmap();
            return false;
        }

        for (uint32_t k = 0; k < kcols; ++k) {
            for (uint32_t c = 0; c < channels; ++c) {
                const size_t slot = static_cast<size_t>(k) * channels + c;
                auto* entry = static_cast<KernelEntry*>(k_ptrs[slot]);
                if (entry && entry->fn) {
                    kmap.data[slot] = static_cast<Scalar>(entry->fn(k, c, entry->user));
                } else {
                    kmap.data[slot] = static_cast<Scalar>(0);
                }
                const void* fn_ptr = entry ? reinterpret_cast<const void*>(entry->fn) : nullptr;
                void* user_ptr = entry ? entry->user : nullptr;
                uint32_t id = 0;
                bool found = false;
                for (size_t i = 0; i < out_plan->entries.size(); ++i) {
                    if (out_plan->entries[i].fn == fn_ptr && out_plan->entries[i].user == user_ptr) {
                        id = out_plan->entries[i].id;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    id = static_cast<uint32_t>(out_plan->entries.size());
                    KernelDispatchEntry info{};
                    info.id = id;
                    info.fn = fn_ptr;
                    info.user = user_ptr;
                    out_plan->entries.push_back(info);
                }
                out_plan->slot_ids.push_back(id);
            }
        }

        kmap.unmap();
        mem->unmap(kernel_ptrs.handle());

        // Single broadcast-style kernel op: reuse the standard kernel scatter once.
        return scatter_add_2d_kernel(base, points, values, kernel, out, clamp);
    }

    static bool scatter_add_2d_kernel_bank(const AbstractTensor& base,
                                           const AbstractTensor& points,
                                           const AbstractTensor& values,
                                           const AbstractTensor& kernel_bank,
                                           const AbstractTensor& kernel_ids,
                                           AbstractTensor* out,
                                           bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid() ||
            !kernel_bank.valid() || !kernel_ids.valid()) {
            return false;
        }
        if (base.backend() != points.backend() ||
            base.backend() != values.backend() ||
            base.backend() != kernel_bank.backend() ||
            base.backend() != kernel_ids.backend()) {
            return false;
        }
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        const TensorDesc& kd = kernel_bank.desc();
        const TensorDesc& idd = kernel_ids.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType || kd.dtype != DType) return false;
        if (idd.dtype != TensorDType::I32 && idd.dtype != TensorDType::U32) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense ||
            vd.layout != TensorLayout::Dense || kd.layout != TensorLayout::Dense ||
            idd.layout != TensorLayout::Dense) {
            return false;
        }
        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        if (idd.shape.dims.size() != 1 || idd.shape.dims[0] != count) return false;

        uint32_t kcols = 1;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            kcols = 1;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            kcols = vd.shape.dims[1];
        } else {
            return false;
        }

        if (kd.shape.dims.size() != 3) return false;
        const uint32_t kernel_count = kd.shape.dims[0];
        if (kd.shape.dims[1] != kcols) return false;
        if (kd.shape.dims[2] != channels) return false;

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap = map_dense(points);
        MappedDense vmap = map_dense(values);
        MappedDense kmap = map_dense(kernel_bank);
        if (!bmap.ok || !omap.ok || !pmap.ok || !vmap.ok || !kmap.ok) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }

        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (!mem) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }
        void* ids_ptr_v = nullptr;
        size_t ids_bytes = 0;
        if (!mem->map(kernel_ids.handle(), &ids_ptr_v, &ids_bytes)) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }

        const bool use_affine = bd.slice.valid && bd.slice.has_affine;

        if (!tensor_copy_f32_into(base, out)) {
            mem->unmap(kernel_ids.handle());
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }
        bmap.unmap();
        omap.unmap();
        MappedDense omap2 = map_dense_mut(*out);
        if (!omap2.ok) {
            mem->unmap(kernel_ids.handle());
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }
        omap = omap2;

        for (uint32_t i = 0; i < count; ++i) {
            uint32_t kid = 0;
            if (idd.dtype == TensorDType::I32) {
                kid = static_cast<uint32_t>(static_cast<int32_t*>(ids_ptr_v)[i]);
            } else {
                kid = static_cast<uint32_t>(static_cast<uint32_t*>(ids_ptr_v)[i]);
            }
            if (kid >= kernel_count) continue;

            float xf = static_cast<float>(pmap.data[i * pcols + 0]);
            float yf = static_cast<float>(pmap.data[i * pcols + 1]);
            if (use_affine) {
                float xo = 0.0f, yo = 0.0f, zo = 0.0f;
                apply_affine_row_major(bd.slice.affine, xf, yf, 0.0f, xo, yo, zo);
                xf = xo;
                yf = yo;
            }
            const int64_t xi = quantize_round_fast(xf);
            const int64_t yi = quantize_round_fast(yf);
            if (clamp) {
                if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(width) ||
                    yi >= static_cast<int64_t>(height)) {
                    continue;
                }
            }
            if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(width) ||
                yi >= static_cast<int64_t>(height)) {
                continue;
            }
            const uint64_t base_idx = (static_cast<uint64_t>(yi) * width + static_cast<uint64_t>(xi)) * channels;
            const uint64_t kernel_base = (static_cast<uint64_t>(kid) * kcols * channels);
            for (uint32_t c = 0; c < channels; ++c) {
                Scalar acc = static_cast<Scalar>(0);
                if (vd.shape.dims.size() == 1) {
                    const Scalar v = vmap.data[i];
                    acc = v * kmap.data[kernel_base + c];
                } else {
                    const Scalar* vrow = vmap.data + static_cast<uint64_t>(i) * kcols;
                    for (uint32_t k = 0; k < kcols; ++k) {
                        acc += vrow[k] * kmap.data[kernel_base + static_cast<uint64_t>(k) * channels + c];
                    }
                }
                omap.data[base_idx + c] += acc;
            }
        }

        mem->unmap(kernel_ids.handle());
        bmap.unmap();
        omap.unmap();
        pmap.unmap();
        vmap.unmap();
        kmap.unmap();
        return true;
    }

    static AbstractTensor apply_stencil_2d(const AbstractTensor& field,
                                           const AbstractTensor& kernel) {
        TensorDesc out_desc = field.desc();
        AbstractTensor out = AbstractTensor::create(out_desc, field.backend());
        if (!out.valid()) return {};
        if (!apply_stencil_2d_into(field, kernel, &out)) return {};
        return out;
    }

    static bool apply_stencil_2d_into(const AbstractTensor& field,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out) {
        if (!out) return false;
        if (!field.valid() || !kernel.valid() || !out->valid()) return false;
        if (field.backend() != kernel.backend() || field.backend() != out->backend()) return false;
        const TensorDesc& fd = field.desc();
        const TensorDesc& kd = kernel.desc();
        const TensorDesc& od = out->desc();
        if (fd.dtype != DType || kd.dtype != DType || od.dtype != DType) return false;
        if (fd.layout != TensorLayout::Dense || kd.layout != TensorLayout::Dense || od.layout != TensorLayout::Dense) return false;
        if (fd.shape.dims.size() != 2 || kd.shape.dims.size() != 2 || od.shape.dims.size() != 2) return false;
        if (od.shape.dims != fd.shape.dims) return false;

        const uint32_t height = fd.shape.dims[0];
        const uint32_t width = fd.shape.dims[1];
        const uint32_t ks = kd.shape.dims[0];
        if (ks == 0 || kd.shape.dims[1] != ks) return false;
        if (ks % 2 == 0) return false;
        const int32_t radius = static_cast<int32_t>(ks / 2);

        MappedDense fmap = map_dense(field);
        MappedDense kmap = map_dense(kernel);
        MappedDense omap = map_dense_mut(*out);
        if (!fmap.ok || !kmap.ok || !omap.ok) {
            fmap.unmap();
            kmap.unmap();
            omap.unmap();
            return false;
        }

        const Scalar* f = fmap.data;
        const Scalar* k = kmap.data;
        Scalar* o = omap.data;

        nodus::ThreadPool* pool = tensor_op_pool();

        struct StencilCtx {
            const Scalar* f = nullptr;
            const Scalar* k = nullptr;
            Scalar* o = nullptr;
            uint32_t width = 0;
            uint32_t height = 0;
            int32_t radius = 0;
            uint32_t ks = 0;
            uint32_t x0 = 0;
            uint32_t x1 = 0;
            uint32_t y0 = 0;
            uint32_t y1 = 0;
        };

        auto border_3 = [](const void* vctx, uint32_t y0r, uint32_t y1r) {
            const auto* ctx = static_cast<const StencilCtx*>(vctx);
            const uint32_t width = ctx->width;
            const uint32_t height = ctx->height;
            const Scalar* f = ctx->f;
            const Scalar* k = ctx->k;
            Scalar* o = ctx->o;
            auto clamp_i32_local = [](int32_t v, int32_t lo, int32_t hi) -> int32_t {
                if (v < lo) return lo;
                if (v > hi) return hi;
                return v;
            };
            for (uint32_t y = y0r; y < y1r; ++y) {
                const bool border_y = (y == 0 || y + 1 == height);
                const uint64_t row = static_cast<uint64_t>(y) * width;
                for (uint32_t x = 0; x < width; ++x) {
                    const bool border = border_y || (x == 0 || x + 1 == width);
                    if (!border) continue;
                    const int32_t y_i = static_cast<int32_t>(y);
                    const int32_t x_i = static_cast<int32_t>(x);

                    const int32_t y0 = clamp_i32_local(y_i - 1, 0, static_cast<int32_t>(height) - 1);
                    const int32_t y1 = y_i;
                    const int32_t y2 = clamp_i32_local(y_i + 1, 0, static_cast<int32_t>(height) - 1);
                    const int32_t x0 = clamp_i32_local(x_i - 1, 0, static_cast<int32_t>(width) - 1);
                    const int32_t x1 = x_i;
                    const int32_t x2 = clamp_i32_local(x_i + 1, 0, static_cast<int32_t>(width) - 1);

                    const uint64_t r0 = static_cast<uint64_t>(y0) * width;
                    const uint64_t r1 = static_cast<uint64_t>(y1) * width;
                    const uint64_t r2 = static_cast<uint64_t>(y2) * width;

                    Scalar acc = static_cast<Scalar>(0);
                    acc += f[r0 + static_cast<uint64_t>(x0)] * k[0];
                    acc += f[r0 + static_cast<uint64_t>(x1)] * k[1];
                    acc += f[r0 + static_cast<uint64_t>(x2)] * k[2];
                    acc += f[r1 + static_cast<uint64_t>(x0)] * k[3];
                    acc += f[r1 + static_cast<uint64_t>(x1)] * k[4];
                    acc += f[r1 + static_cast<uint64_t>(x2)] * k[5];
                    acc += f[r2 + static_cast<uint64_t>(x0)] * k[6];
                    acc += f[r2 + static_cast<uint64_t>(x1)] * k[7];
                    acc += f[r2 + static_cast<uint64_t>(x2)] * k[8];
                    o[row + x] = acc;
                }
            }
        };

        auto interior_3 = [](const void* vctx, uint32_t y0r, uint32_t y1r) {
            const auto* ctx = static_cast<const StencilCtx*>(vctx);
            const uint32_t width = ctx->width;
            const uint32_t height = ctx->height;
            const Scalar* f = ctx->f;
            const Scalar* k = ctx->k;
            Scalar* o = ctx->o;
            const uint32_t y_start = std::max<uint32_t>(1, y0r);
            const uint32_t y_end = std::min<uint32_t>(height - 1, y1r);
            if (y_start >= y_end) return;
            for (uint32_t y = y_start; y < y_end; ++y) {
                const uint64_t r0 = static_cast<uint64_t>(y - 1) * width;
                const uint64_t r1 = static_cast<uint64_t>(y) * width;
                const uint64_t r2 = static_cast<uint64_t>(y + 1) * width;
                for (uint32_t x = 1; x + 1 < width; ++x) {
                    const uint64_t i0 = static_cast<uint64_t>(x - 1);
                    const uint64_t i1 = static_cast<uint64_t>(x);
                    const uint64_t i2 = static_cast<uint64_t>(x + 1);
                    Scalar acc = static_cast<Scalar>(0);
                    acc += f[r0 + i0] * k[0];
                    acc += f[r0 + i1] * k[1];
                    acc += f[r0 + i2] * k[2];
                    acc += f[r1 + i0] * k[3];
                    acc += f[r1 + i1] * k[4];
                    acc += f[r1 + i2] * k[5];
                    acc += f[r2 + i0] * k[6];
                    acc += f[r2 + i1] * k[7];
                    acc += f[r2 + i2] * k[8];
                    o[r1 + x] = acc;
                }
            }
        };

        auto border_general = [](const void* vctx, uint32_t y0r, uint32_t y1r) {
            const auto* ctx = static_cast<const StencilCtx*>(vctx);
            const uint32_t width = ctx->width;
            const uint32_t height = ctx->height;
            const int32_t radius = ctx->radius;
            const uint32_t ks = ctx->ks;
            const Scalar* f = ctx->f;
            const Scalar* k = ctx->k;
            Scalar* o = ctx->o;
            const uint32_t x0 = ctx->x0;
            const uint32_t x1 = ctx->x1;
            const uint32_t y0 = ctx->y0;
            const uint32_t y1 = ctx->y1;
            auto clamp_i32_local = [](int32_t v, int32_t lo, int32_t hi) -> int32_t {
                if (v < lo) return lo;
                if (v > hi) return hi;
                return v;
            };
            const int32_t h_i = static_cast<int32_t>(height);
            const int32_t w_i = static_cast<int32_t>(width);
            for (uint32_t y = y0r; y < y1r; ++y) {
                const bool border_y = (y < y0) || (y >= y1);
                const uint64_t row = static_cast<uint64_t>(y) * width;
                for (uint32_t x = 0; x < width; ++x) {
                    const bool border = border_y || (x < x0) || (x >= x1);
                    if (!border) continue;
                    Scalar acc = static_cast<Scalar>(0);
                    const int32_t y_i = static_cast<int32_t>(y);
                    const int32_t x_i = static_cast<int32_t>(x);
                    for (int32_t ky = -radius; ky <= radius; ++ky) {
                        const int32_t sy = clamp_i32_local(y_i + ky, 0, h_i - 1);
                        const uint64_t frow = static_cast<uint64_t>(sy) * width;
                        const uint64_t krow = static_cast<uint64_t>(ky + radius) * ks;
                        for (int32_t kx = -radius; kx <= radius; ++kx) {
                            const int32_t sx = clamp_i32_local(x_i + kx, 0, w_i - 1);
                            acc += f[frow + static_cast<uint64_t>(sx)] * k[krow + static_cast<uint64_t>(kx + radius)];
                        }
                    }
                    o[row + x] = acc;
                }
            }
        };

        auto interior_general = [](const void* vctx, uint32_t y0r, uint32_t y1r) {
            const auto* ctx = static_cast<const StencilCtx*>(vctx);
            const uint32_t width = ctx->width;
            const uint32_t height = ctx->height;
            const int32_t radius = ctx->radius;
            const uint32_t ks = ctx->ks;
            const Scalar* f = ctx->f;
            const Scalar* k = ctx->k;
            Scalar* o = ctx->o;
            const uint32_t x0 = ctx->x0;
            const uint32_t x1 = ctx->x1;
            const uint32_t y0 = ctx->y0;
            const uint32_t y1 = ctx->y1;
            if (x0 >= x1) return;
            const uint32_t y_start = std::max<uint32_t>(y0, y0r);
            const uint32_t y_end = std::min<uint32_t>(y1, y1r);
            if (y_start >= y_end) return;
            for (uint32_t y = y_start; y < y_end; ++y) {
                const uint64_t row = static_cast<uint64_t>(y) * width;
                for (uint32_t x = x0; x < x1; ++x) {
                    Scalar acc = static_cast<Scalar>(0);
                    for (int32_t ky = -radius; ky <= radius; ++ky) {
                        const uint64_t frow = static_cast<uint64_t>(static_cast<int32_t>(y) + ky) * width;
                        const uint64_t krow = static_cast<uint64_t>(ky + radius) * ks;
                        for (int32_t kx = -radius; kx <= radius; ++kx) {
                            acc += f[frow + static_cast<uint64_t>(static_cast<int32_t>(x) + kx)] *
                                   k[krow + static_cast<uint64_t>(kx + radius)];
                        }
                    }
                    o[row + x] = acc;
                }
            }
        };

        StencilCtx ctx{};
        ctx.f = f;
        ctx.k = k;
        ctx.o = o;
        ctx.width = width;
        ctx.height = height;
        ctx.radius = radius;
        ctx.ks = ks;
        ctx.x0 = 0;
        ctx.x1 = 0;
        ctx.y0 = 0;
        ctx.y1 = 0;

        // Small kernels are worth special-casing for better vectorization.
        if (ks == 3 && width > 0 && height > 0) {
            // Border: y=0 and y=H-1 and x=0 and x=W-1. Cheap because it's O(W+H).
            submit_row_jobs(pool, border_3, &ctx, 0, height);

            // Interior (no clamps).
            if (width >= 3 && height >= 3) {
                submit_row_jobs(pool, interior_3, &ctx, 1, height - 1);
            }

            fmap.unmap();
            kmap.unmap();
            omap.unmap();
            return true;
        }

        const uint32_t x0 = static_cast<uint32_t>(radius);
        const uint32_t y0 = static_cast<uint32_t>(radius);
        const uint32_t x1 = (width > static_cast<uint32_t>(radius)) ? (width - static_cast<uint32_t>(radius)) : 0u;
        const uint32_t y1 = (height > static_cast<uint32_t>(radius)) ? (height - static_cast<uint32_t>(radius)) : 0u;
        ctx.x0 = x0;
        ctx.x1 = x1;
        ctx.y0 = y0;
        ctx.y1 = y1;

        // Border region (clamped).
        submit_row_jobs(pool, border_general, &ctx, 0, height);

        // Interior region (no clamps).
        if (x0 < x1 && y0 < y1) {
            submit_row_jobs(pool, interior_general, &ctx, y0, y1);
        }

        fmap.unmap();
        kmap.unmap();
        omap.unmap();
        return true;
    }

    static bool scatter_add_2d_affine(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& mat4,
                                      AbstractTensor* out,
                                      bool clamp) {
        if (!out) return false;
        if (!mat4.valid()) return false;
        if (base.backend() != mat4.backend()) return false;
        const TensorDesc& md = mat4.desc();
        if (md.dtype != DType || md.layout != TensorLayout::Dense) return false;
        const bool use_mat4 = shape_is(md, {4, 4});
        const bool use_mat3 = shape_is(md, {3, 3});
        if (!use_mat4 && !use_mat3) return false;

        MappedDense mmap = map_dense(mat4);
        if (!mmap.ok) {
            mmap.unmap();
            return false;
        }

        if (!points.valid()) {
            mmap.unmap();
            return false;
        }
        const TensorDesc& pd = points.desc();
        if (pd.dtype != DType || pd.layout != TensorLayout::Dense) {
            mmap.unmap();
            return false;
        }
        if (pd.shape.dims.size() != 2) {
            mmap.unmap();
            return false;
        }
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) {
            mmap.unmap();
            return false;
        }

        float affine[16] = {0};
        if (use_mat4) {
            for (uint32_t i = 0; i < 16; ++i) {
                affine[i] = static_cast<float>(mmap.data[i]);
            }
        } else {
            affine[0] = static_cast<float>(mmap.data[0]);
            affine[1] = static_cast<float>(mmap.data[1]);
            affine[4] = static_cast<float>(mmap.data[3]);
            affine[5] = static_cast<float>(mmap.data[4]);
            affine[12] = static_cast<float>(mmap.data[6]);
            affine[13] = static_cast<float>(mmap.data[7]);
            affine[10] = 1.0f;
            affine[15] = 1.0f;
        }
        mmap.unmap();

        TensorDesc bd = base.desc();
        bd.slice.has_affine = false;
        AbstractTensor base_view = AbstractTensor::wrap(base.handle(), bd, base.backend(), false);

        if (pcols == 2) {
            base_view.set_slice_affine_row_major(affine);
            return scatter_add_nd(base_view, points, values, out, clamp);
        }

        MappedDense pmap = map_dense(points);
        if (!pmap.ok) {
            pmap.unmap();
            return false;
        }

        TensorDesc tdesc = pd;
        tdesc.shape.dims[1] = 2;
        AbstractTensor tmp = AbstractTensor::create(tdesc, base.backend());
        if (!tmp.valid()) {
            pmap.unmap();
            return false;
        }
        MappedDense tmap = map_dense_mut(tmp);
        if (!tmap.ok) {
            pmap.unmap();
            tmap.unmap();
            return false;
        }

        for (uint32_t i = 0; i < count; ++i) {
            const Scalar x = pmap.data[i * pcols + 0];
            const Scalar y = pmap.data[i * pcols + 1];
            const Scalar z = use_mat4 ? pmap.data[i * pcols + 2] : static_cast<Scalar>(0);
            float tx = 0.0f;
            float ty = 0.0f;
            if (use_mat4) {
                tx = static_cast<float>(x) * affine[0] +
                     static_cast<float>(y) * affine[4] +
                     static_cast<float>(z) * affine[8] +
                     affine[12];
                ty = static_cast<float>(x) * affine[1] +
                     static_cast<float>(y) * affine[5] +
                     static_cast<float>(z) * affine[9] +
                     affine[13];
            } else {
                tx = static_cast<float>(x) * affine[0] +
                     static_cast<float>(y) * affine[4] +
                     affine[12];
                ty = static_cast<float>(x) * affine[1] +
                     static_cast<float>(y) * affine[5] +
                     affine[13];
            }
            tmap.data[i * 2 + 0] = static_cast<Scalar>(tx);
            tmap.data[i * 2 + 1] = static_cast<Scalar>(ty);
        }

        pmap.unmap();
        tmap.unmap();

        return scatter_add_nd(base_view, tmp, values, out, clamp);
    }
};

struct MappedDenseU8Value {
    InMemoryBackend* backend = nullptr;
    AbstractTensorHandle handle{};
    uint8_t* data = nullptr;
    uint64_t elems = 0;
    bool ok = false;

    void unmap() {
        if (backend && data) backend->unmap(handle);
        backend = nullptr;
        data = nullptr;
        elems = 0;
        ok = false;
    }
};

struct MappedDenseI32Local {
    InMemoryBackend* backend = nullptr;
    AbstractTensorHandle handle{};
    int32_t* data = nullptr;
    uint64_t elems = 0;
    bool ok = false;

    void unmap() {
        if (backend && data) backend->unmap(handle);
        backend = nullptr;
        data = nullptr;
        elems = 0;
        ok = false;
    }
};

struct MappedDenseU8Local {
    InMemoryBackend* backend = nullptr;
    AbstractTensorHandle handle{};
    uint8_t* data = nullptr;
    uint64_t elems = 0;
    bool ok = false;

    void unmap() {
        if (backend && data) backend->unmap(handle);
        backend = nullptr;
        data = nullptr;
        elems = 0;
        ok = false;
    }
};

static MappedDenseI32Local map_dense_i32_local(const AbstractTensor& t) {
    MappedDenseI32Local out{};
    if (!t.valid()) return out;
    const TensorDesc& d = t.desc();
    if (d.dtype != TensorDType::I32 || d.layout != TensorLayout::Dense) return out;
    auto* mem = dynamic_cast<InMemoryBackend*>(t.backend());
    if (!mem) return out;
    void* ptr = nullptr;
    size_t bytes = 0;
    if (!mem->map(t.handle(), &ptr, &bytes)) return out;
    out.backend = mem;
    out.handle = t.handle();
    out.data = static_cast<int32_t*>(ptr);
    out.elems = d.shape.element_count();
    out.ok = true;
    return out;
}

static MappedDenseU8Local map_dense_u8_local(const AbstractTensor& t) {
    MappedDenseU8Local out{};
    if (!t.valid()) return out;
    const TensorDesc& d = t.desc();
    if (d.dtype != TensorDType::Bool || d.layout != TensorLayout::Dense) return out;
    auto* mem = dynamic_cast<InMemoryBackend*>(t.backend());
    if (!mem) return out;
    void* ptr = nullptr;
    size_t bytes = 0;
    if (!mem->map(t.handle(), &ptr, &bytes)) return out;
    out.backend = mem;
    out.handle = t.handle();
    out.data = static_cast<uint8_t*>(ptr);
    out.elems = d.shape.element_count();
    out.ok = true;
    return out;
}

static bool shape_is_local(const TensorDesc& desc, std::initializer_list<uint32_t> dims) {
    if (desc.shape.dims.size() != dims.size()) return false;
    size_t i = 0;
    for (uint32_t d : dims) {
        if (desc.shape.dims[i++] != d) return false;
    }
    return true;
}

template <typename Scalar, TensorDType DTypeValue>
struct MappedDenseTyped {
    InMemoryBackend* backend = nullptr;
    AbstractTensorHandle handle{};
    Scalar* data = nullptr;
    uint64_t elems = 0;
    bool ok = false;

    void unmap() {
        if (backend && data) backend->unmap(handle);
        backend = nullptr;
        data = nullptr;
        elems = 0;
        ok = false;
    }
};

template <typename Scalar, TensorDType DTypeValue>
static MappedDenseTyped<Scalar, DTypeValue> map_dense_typed(const AbstractTensor& t) {
    MappedDenseTyped<Scalar, DTypeValue> out{};
    if (!t.valid()) return out;
    const TensorDesc& d = t.desc();
    if (d.dtype != DTypeValue || d.layout != TensorLayout::Dense) return out;
    auto* mem = dynamic_cast<InMemoryBackend*>(t.backend());
    if (!mem) return out;
    void* ptr = nullptr;
    size_t bytes = 0;
    if (!mem->map(t.handle(), &ptr, &bytes)) return out;
    out.backend = mem;
    out.handle = t.handle();
    out.data = static_cast<Scalar*>(ptr);
    out.elems = d.shape.element_count();
    out.ok = true;
    return out;
}

template <typename Scalar, typename AccumT>
static inline Scalar clamp_accum_to_scalar(AccumT v) {
    if constexpr (std::is_signed_v<Scalar>) {
        const AccumT lo = static_cast<AccumT>(std::numeric_limits<Scalar>::min());
        const AccumT hi = static_cast<AccumT>(std::numeric_limits<Scalar>::max());
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        return static_cast<Scalar>(v);
    } else {
        const AccumT hi = static_cast<AccumT>(std::numeric_limits<Scalar>::max());
        if (v > hi) v = hi;
        return static_cast<Scalar>(v);
    }
}

template <typename Scalar, TensorDType DTypeValue>
static bool gather_stencil_2d_int_impl(const AbstractTensor& base,
                                       const TensorStencil& stencil,
                                       AbstractTensor* out,
                                       StencilBoundaryMode boundary,
                                       const AbstractTensor* target_mask,
                                       const AbstractTensor* source_mask,
                                       bool normalize) {
    using AccumT = std::conditional_t<std::is_signed_v<Scalar>, int64_t, uint64_t>;
    if (!out) return false;
    out->reset();
    if (!base.valid() || !stencil.offsets.valid() || !stencil.weights.valid()) return false;
    if (base.backend() != stencil.offsets.backend() || base.backend() != stencil.weights.backend()) return false;
    if (target_mask && base.backend() != target_mask->backend()) return false;
    if (source_mask && base.backend() != source_mask->backend()) return false;

    const TensorDesc& bd = base.desc();
    const TensorDesc& od = stencil.offsets.desc();
    const TensorDesc& wd = stencil.weights.desc();
    if (bd.dtype != DTypeValue || wd.dtype != DTypeValue) return false;
    if (od.dtype != TensorDType::I32) return false;
    if (bd.layout != TensorLayout::Dense || od.layout != TensorLayout::Dense || wd.layout != TensorLayout::Dense)
        return false;
    if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
    const uint32_t height = bd.shape.dims[0];
    const uint32_t width = bd.shape.dims[1];
    const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

    if (od.shape.dims.size() != 2 || od.shape.dims[1] != 2) return false;
    const uint32_t count = od.shape.dims[0];
    if (wd.shape.dims.size() != 1 || wd.shape.dims[0] != count) return false;

    if (target_mask) {
        const TensorDesc& md = target_mask->desc();
        if (md.dtype != TensorDType::Bool || md.layout != TensorLayout::Dense) return false;
        if (md.shape.dims.size() != 2 || md.shape.dims[0] != height || md.shape.dims[1] != width) return false;
    }
    if (source_mask) {
        const TensorDesc& md = source_mask->desc();
        if (md.dtype != TensorDType::Bool || md.layout != TensorLayout::Dense) return false;
        if (md.shape.dims.size() != 2 || md.shape.dims[0] != height || md.shape.dims[1] != width) return false;
    }

    TensorDesc out_desc = bd;
    *out = AbstractTensor::create(out_desc, base.backend());
    if (!out->valid()) return false;

    auto bmap = map_dense_typed<Scalar, DTypeValue>(base);
    auto omap = map_dense_typed<Scalar, DTypeValue>(*out);
    MappedDenseI32Local omap_off = map_dense_i32_local(stencil.offsets);
    auto wmap = map_dense_typed<Scalar, DTypeValue>(stencil.weights);
    MappedDenseU8Local mmap{};
    if (target_mask) {
        mmap = map_dense_u8_local(*target_mask);
    }
    MappedDenseU8Local smap{};
    if (source_mask) {
        smap = map_dense_u8_local(*source_mask);
    }
    if (!bmap.ok || !omap.ok || !omap_off.ok || !wmap.ok ||
        (target_mask && !mmap.ok) || (source_mask && !smap.ok)) {
        bmap.unmap();
        omap.unmap();
        omap_off.unmap();
        wmap.unmap();
        if (target_mask) mmap.unmap();
        if (source_mask) smap.unmap();
        return false;
    }

    const auto* off_ptr = omap_off.data;
    const Scalar* w_ptr = wmap.data;

    int32_t radius_x = 0;
    int32_t radius_y = 0;
    for (uint32_t i = 0; i < count; ++i) {
        radius_x = std::max(radius_x, std::abs(off_ptr[i * 2 + 0]));
        radius_y = std::max(radius_y, std::abs(off_ptr[i * 2 + 1]));
    }

    static uint32_t kTilePxX = 0;
    static uint32_t kTilePxY = 0;
    if (kTilePxX == 0 || kTilePxY == 0) {
        constexpr uint64_t kDefaultTileCacheBytes = 6ull * 1024ull * 1024ull * 1024ull;
        TileShape2D shape = choose_tile_shape_2d_impl(bd, static_cast<uint32_t>(radius_x),
                                  static_cast<uint32_t>(radius_y), kDefaultTileCacheBytes,
                                  TileContiguityStrategy::Auto);
        kTilePxX = (shape.x > 0) ? shape.x : 32u;
        kTilePxY = (shape.y > 0) ? shape.y : 32u;
    }

    const uint32_t tiles_x = (width + kTilePxX - 1) / kTilePxX;
    const uint32_t tiles_y = (height + kTilePxY - 1) / kTilePxY;

    struct GatherCtx {
        const Scalar* src = nullptr;
        Scalar* dst = nullptr;
        const int32_t* offsets = nullptr;
        const Scalar* weights = nullptr;
        const uint8_t* mask = nullptr;
        const uint8_t* src_mask = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t channels = 0;
        uint32_t count = 0;
        uint32_t tiles_x = 0;
        uint32_t tile_px_x = 0;
        uint32_t tile_px_y = 0;
        int32_t radius_x = 0;
        int32_t radius_y = 0;
        StencilBoundaryMode boundary = StencilBoundaryMode::Zero;
        bool normalize = false;
        TensorBackend* backend = nullptr;
    };

    auto gather_tiles = [](const void* vctx, uint32_t ty0, uint32_t ty1) {
        const auto* ctx = static_cast<const GatherCtx*>(vctx);
        auto* mem = dynamic_cast<InMemoryBackend*>(ctx->backend);
        if (!mem) return;
        static thread_local AbstractTensorPool tile_pool(tile_pool_options());
        static thread_local AbstractTensorPool::PooledTensor tile_buf;

        for (uint32_t ty = ty0; ty < ty1; ++ty) {
            const uint32_t y0 = ty * ctx->tile_px_y;
            const uint32_t y1 = std::min<uint32_t>(ctx->height, y0 + ctx->tile_px_y);
            const uint32_t rows = y1 - y0;
            for (uint32_t tx = 0; tx < ctx->tiles_x; ++tx) {
                const uint32_t x0 = tx * ctx->tile_px_x;
                const uint32_t x1 = std::min<uint32_t>(ctx->width, x0 + ctx->tile_px_x);
                const uint32_t cols = x1 - x0;

                const uint32_t halo_h = rows + static_cast<uint32_t>(ctx->radius_y * 2);
                const uint32_t halo_w = cols + static_cast<uint32_t>(ctx->radius_x * 2);

                TensorDesc tile_desc{};
                tile_desc.dtype = DTypeValue;
                tile_desc.layout = TensorLayout::Dense;
                if (ctx->channels == 1) {
                    tile_desc.shape.dims = {halo_h, halo_w};
                } else {
                    tile_desc.shape.dims = {halo_h, halo_w, ctx->channels};
                }

                if (!tile_buf.valid() || tile_buf.tensor().desc().shape.dims != tile_desc.shape.dims) {
                    tile_buf = tile_pool.acquire(tile_desc, ctx->backend);
                }
                if (!tile_buf.valid()) return;

                void* tile_ptr_v = nullptr;
                size_t tile_bytes = 0;
                if (!mem->map(tile_buf.tensor().handle(), &tile_ptr_v, &tile_bytes)) return;
                auto* tile = static_cast<Scalar*>(tile_ptr_v);

                auto map_coord_local = [&](int64_t v, int64_t limit) -> int64_t {
                    if (limit <= 0) return -1;
                    if (v >= 0 && v < limit) return v;
                    switch (ctx->boundary) {
                        case StencilBoundaryMode::Zero:
                            return -1;
                        case StencilBoundaryMode::Clamp:
                            return std::clamp<int64_t>(v, 0, limit - 1);
                        case StencilBoundaryMode::Mirror: {
                            if (limit == 1) return 0;
                            int64_t period = (limit - 1) * 2;
                            int64_t m = v % period;
                            if (m < 0) m += period;
                            if (m >= limit) m = period - m;
                            return m;
                        }
                        case StencilBoundaryMode::Wrap: {
                            int64_t m = v % limit;
                            if (m < 0) m += limit;
                            return m;
                        }
                        default:
                            return -1;
                    }
                };

                for (uint32_t tyh = 0; tyh < halo_h; ++tyh) {
                    const int64_t sy = static_cast<int64_t>(y0) + static_cast<int64_t>(tyh) - ctx->radius_y;
                    const int64_t my = map_coord_local(sy, ctx->height);
                    for (uint32_t txh = 0; txh < halo_w; ++txh) {
                        const int64_t sx = static_cast<int64_t>(x0) + static_cast<int64_t>(txh) - ctx->radius_x;
                        const int64_t mx = map_coord_local(sx, ctx->width);
                        const uint64_t tile_idx = (static_cast<uint64_t>(tyh) * halo_w + txh) * ctx->channels;
                        if (my < 0 || mx < 0) {
                            for (uint32_t c = 0; c < ctx->channels; ++c) {
                                tile[tile_idx + c] = static_cast<Scalar>(0);
                            }
                            continue;
                        }
                        if (ctx->src_mask) {
                            const uint64_t m_idx = static_cast<uint64_t>(my) * ctx->width +
                                                   static_cast<uint64_t>(mx);
                            if (ctx->src_mask[m_idx] == 0u) {
                                for (uint32_t c = 0; c < ctx->channels; ++c) {
                                    tile[tile_idx + c] = static_cast<Scalar>(0);
                                }
                                continue;
                            }
                        }
                        const uint64_t src_idx = (static_cast<uint64_t>(my) * ctx->width +
                                                  static_cast<uint64_t>(mx)) * ctx->channels;
                        for (uint32_t c = 0; c < ctx->channels; ++c) {
                            tile[tile_idx + c] = ctx->src[src_idx + c];
                        }
                    }
                }

                static thread_local std::vector<int64_t> offset_delta;
                offset_delta.resize(ctx->count);
                for (uint32_t i = 0; i < ctx->count; ++i) {
                    const int32_t dx = ctx->offsets[i * 2 + 0];
                    const int32_t dy = ctx->offsets[i * 2 + 1];
                    offset_delta[i] = (static_cast<int64_t>(dy) * static_cast<int64_t>(halo_w) +
                                       static_cast<int64_t>(dx)) * static_cast<int64_t>(ctx->channels);
                }

                for (uint32_t ly = 0; ly < rows; ++ly) {
                    const uint32_t y = y0 + ly;
                    for (uint32_t lx = 0; lx < cols; ++lx) {
                        const uint32_t x = x0 + lx;
                        if (ctx->mask) {
                            const uint8_t m = ctx->mask[static_cast<uint64_t>(y) * ctx->width + x];
                            if (m == 0u) continue;
                        }
                        const uint64_t out_idx = (static_cast<uint64_t>(y) * ctx->width + x) * ctx->channels;
                        const int64_t base = (static_cast<int64_t>(ly + ctx->radius_y) *
                                              static_cast<int64_t>(halo_w) +
                                              static_cast<int64_t>(lx + ctx->radius_x)) *
                                             static_cast<int64_t>(ctx->channels);

                        if (!ctx->normalize) {
                            for (uint32_t c = 0; c < ctx->channels; ++c) {
                                AccumT acc = 0;
                                const int64_t base_c = base + static_cast<int64_t>(c);
                                for (uint32_t i = 0; i < ctx->count; ++i) {
                                    const int64_t tile_idx = base_c + offset_delta[i];
                                    acc += static_cast<AccumT>(tile[static_cast<uint64_t>(tile_idx)]) *
                                           static_cast<AccumT>(ctx->weights[i]);
                                }
                                ctx->dst[out_idx + c] = clamp_accum_to_scalar<Scalar>(acc);
                            }
                        } else {
                            for (uint32_t c = 0; c < ctx->channels; ++c) {
                                AccumT acc = 0;
                                AccumT sum_w = 0;
                                for (uint32_t i = 0; i < ctx->count; ++i) {
                                    const int32_t dx = ctx->offsets[i * 2 + 0];
                                    const int32_t dy = ctx->offsets[i * 2 + 1];
                                    const int64_t sx = static_cast<int64_t>(x) + dx;
                                    const int64_t sy = static_cast<int64_t>(y) + dy;
                                    const int64_t mx = map_coord_local(sx, ctx->width);
                                    const int64_t my = map_coord_local(sy, ctx->height);
                                    if (mx < 0 || my < 0) continue;
                                    if (ctx->src_mask) {
                                        const uint64_t m_idx = static_cast<uint64_t>(my) * ctx->width +
                                                               static_cast<uint64_t>(mx);
                                        if (ctx->src_mask[m_idx] == 0u) continue;
                                    }
                                    const uint32_t tyi = static_cast<uint32_t>(static_cast<int32_t>(ly) + ctx->radius_y + dy);
                                    const uint32_t txi = static_cast<uint32_t>(static_cast<int32_t>(lx) + ctx->radius_x + dx);
                                    const uint64_t tile_idx =
                                        (static_cast<uint64_t>(tyi) * halo_w + txi) * ctx->channels + c;
                                    const AccumT w = static_cast<AccumT>(ctx->weights[i]);
                                    acc += static_cast<AccumT>(tile[tile_idx]) * w;
                                    sum_w += w;
                                }
                                if (sum_w != 0) {
                                    acc /= sum_w;
                                }
                                ctx->dst[out_idx + c] = clamp_accum_to_scalar<Scalar>(acc);
                            }
                        }
                    }
                }

                mem->unmap(tile_buf.tensor().handle());
            }
        }
    };

    GatherCtx ctx{};
    ctx.src = bmap.data;
    ctx.dst = omap.data;
    ctx.offsets = off_ptr;
    ctx.weights = w_ptr;
    ctx.mask = target_mask ? static_cast<const uint8_t*>(mmap.data) : nullptr;
    ctx.src_mask = source_mask ? static_cast<const uint8_t*>(smap.data) : nullptr;
    ctx.width = width;
    ctx.height = height;
    ctx.channels = channels;
    ctx.count = count;
    ctx.tiles_x = tiles_x;
    ctx.tile_px_x = kTilePxX;
    ctx.tile_px_y = kTilePxY;
    ctx.radius_x = radius_x;
    ctx.radius_y = radius_y;
    ctx.boundary = boundary;
    ctx.normalize = normalize;
    ctx.backend = base.backend();

    submit_row_jobs(tensor_op_pool(), gather_tiles, &ctx, 0, tiles_y);

    bmap.unmap();
    omap.unmap();
    omap_off.unmap();
    wmap.unmap();
    if (target_mask) mmap.unmap();
    if (source_mask) smap.unmap();
    return true;
}

template <typename Scalar, TensorDType DTypeValue>
static bool gather_2d_typed(const AbstractTensor& base,
                            const AbstractTensor& points,
                            AbstractTensor* out,
                            bool clamp) {
    if (!out) return false;
    out->reset();
    if (!base.valid() || !points.valid()) return false;
    if (base.backend() != points.backend()) return false;
    const TensorDesc& bd = base.desc();
    const TensorDesc& pd = points.desc();
    if (bd.dtype != DTypeValue) return false;
    const bool points_match = (pd.dtype == DTypeValue);
    if (!points_match) {
        if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
            pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
            return false;
        }
    }
    if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense) return false;

    if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
    const uint32_t height = bd.shape.dims[0];
    const uint32_t width = bd.shape.dims[1];
    const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

    if (pd.shape.dims.size() != 2) return false;
    const uint32_t count = pd.shape.dims[0];
    const uint32_t pcols = pd.shape.dims[1];
    if (pcols != 2 && pcols != 3) return false;

    TensorDesc out_desc{};
    out_desc.dtype = DTypeValue;
    out_desc.layout = TensorLayout::Dense;
    if (channels == 1) {
        out_desc.shape.dims = {count};
    } else {
        out_desc.shape.dims = {count, channels};
    }
    *out = AbstractTensor::create(out_desc, base.backend());
    if (!out->valid()) return false;

    auto bmap = map_dense_typed<Scalar, DTypeValue>(base);
    auto omap = map_dense_typed<Scalar, DTypeValue>(*out);
    MappedDenseTyped<Scalar, DTypeValue> pmap{};
    void* points_ptr = nullptr;
    size_t points_bytes = 0;
    if (points_match) {
        pmap = map_dense_typed<Scalar, DTypeValue>(points);
        points_ptr = pmap.data;
    } else {
        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
            bmap.unmap();
            omap.unmap();
            return false;
        }
    }
    if (!bmap.ok || !omap.ok || (points_match && !pmap.ok)) {
        bmap.unmap();
        omap.unmap();
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return false;
    }

    CoordBuffer coord_buf{};
    if (!acquire_coord_buffer(count, 2u, base.backend(), coord_buf)) {
        bmap.unmap();
        omap.unmap();
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return false;
    }
    int64_t* coords = coord_buf.coords_ptr;
    uint8_t* in_bounds = coord_buf.mask_ptr;
    const bool use_affine = bd.slice.valid && bd.slice.has_affine;
    convert_points_to_int_coords(pd,
                                 points_ptr,
                                 count,
                                 2u,
                                 points_match,
                                 use_affine,
                                 bd.slice.affine,
                                 bd.shape.dims.data(),
                                 true,
                                 coords,
                                 in_bounds);


    const bool use_span_tiling = span_tiling_enabled_from_env("NODUS_GATHER_SPAN_TILING");
    if (use_span_tiling) {
        SpanOpConfig span_op{};
        span_op.aggregate = SpanAggregateOp::Replace;
        span_op.clamp = clamp;
        if (span_gather_2d<Scalar>(coords,
                                   in_bounds,
                                   count,
                                   width,
                                   height,
                                   channels,
                                   bmap.data,
                                   omap.data,
                                   base.backend(),
                                   0u,
                                   span_op)) {
            bmap.unmap();
            omap.unmap();
            release_coord_buffer(base.backend(), coord_buf);
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return true;
        }
    }

    struct GatherCtx {
        const Scalar* base = nullptr;
        Scalar* out = nullptr;
        const int64_t* coords = nullptr;
        const uint8_t* in_bounds = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t channels = 0;
        uint32_t pcols = 0;
        bool clamp = false;
    };

    GatherCtx ctx{};
    ctx.base = bmap.data;
    ctx.out = omap.data;
    ctx.coords = coords;
    ctx.in_bounds = in_bounds;
    ctx.width = width;
    ctx.height = height;
    ctx.channels = channels;
    ctx.pcols = pcols;
    ctx.clamp = clamp;

    auto gather_rows = [](const void* vctx, uint32_t i0, uint32_t i1) {
        const auto* ctx = static_cast<const GatherCtx*>(vctx);
        for (uint32_t i = i0; i < i1; ++i) {
            if (!ctx->in_bounds[i]) {
                if (ctx->clamp) continue;
                continue;
            }

            const int64_t xi = ctx->coords[static_cast<size_t>(i) * 2 + 0];
            const int64_t yi = ctx->coords[static_cast<size_t>(i) * 2 + 1];

            const uint64_t base_idx = (static_cast<uint64_t>(yi) * ctx->width +
                                       static_cast<uint64_t>(xi)) * ctx->channels;
            if (ctx->channels == 1) {
                ctx->out[i] = ctx->base[base_idx];
            } else {
                Scalar* dst = ctx->out + static_cast<uint64_t>(i) * ctx->channels;
                const Scalar* src = ctx->base + base_idx;
                for (uint32_t c = 0; c < ctx->channels; ++c) {
                    dst[c] = src[c];
                }
            }
        }
    };

    submit_row_jobs(tensor_op_pool(), gather_rows, &ctx, 0, count);

    bmap.unmap();
    omap.unmap();
    release_coord_buffer(base.backend(), coord_buf);
    if (points_match) {
        pmap.unmap();
    } else if (points_ptr) {
        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (mem) mem->unmap(points.handle());
    }
    return true;
}

template <typename Scalar, TensorDType DTypeValue>
static bool gather_add_2d_typed(const AbstractTensor& base,
                                const AbstractTensor& points,
                                AbstractTensor* out,
                                bool clamp) {
    if (!out || !out->valid()) return false;
    if (!base.valid() || !points.valid()) return false;
    if (base.backend() != points.backend() || base.backend() != out->backend()) return false;
    const TensorDesc& bd = base.desc();
    const TensorDesc& pd = points.desc();
    const TensorDesc& od = out->desc();
    if (bd.dtype != DTypeValue || od.dtype != DTypeValue) return false;
    const bool points_match = (pd.dtype == DTypeValue);
    if (!points_match) {
        if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
            pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
            return false;
        }
    }
    if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || od.layout != TensorLayout::Dense)
        return false;

    if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
    const uint32_t height = bd.shape.dims[0];
    const uint32_t width = bd.shape.dims[1];
    const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

    if (pd.shape.dims.size() != 2) return false;
    const uint32_t count = pd.shape.dims[0];
    const uint32_t pcols = pd.shape.dims[1];
    if (pcols != 2 && pcols != 3) return false;

    if (channels == 1) {
        if (!shape_is_local(od, {count})) return false;
    } else {
        if (!shape_is_local(od, {count, channels})) return false;
    }

    auto bmap = map_dense_typed<Scalar, DTypeValue>(base);
    auto omap = map_dense_typed<Scalar, DTypeValue>(*out);
    MappedDenseTyped<Scalar, DTypeValue> pmap{};
    void* points_ptr = nullptr;
    size_t points_bytes = 0;
    if (points_match) {
        pmap = map_dense_typed<Scalar, DTypeValue>(points);
        points_ptr = pmap.data;
    } else {
        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
            bmap.unmap();
            omap.unmap();
            return false;
        }
    }
    if (!bmap.ok || !omap.ok || (points_match && !pmap.ok)) {
        bmap.unmap();
        omap.unmap();
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return false;
    }

    CoordBuffer coord_buf{};
    if (!acquire_coord_buffer(count, 2u, base.backend(), coord_buf)) {
        bmap.unmap();
        omap.unmap();
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return false;
    }
    int64_t* coords = coord_buf.coords_ptr;
    uint8_t* in_bounds = coord_buf.mask_ptr;
    const bool use_affine = bd.slice.valid && bd.slice.has_affine;
    convert_points_to_int_coords(pd,
                                 points_ptr,
                                 count,
                                 2u,
                                 points_match,
                                 use_affine,
                                 bd.slice.affine,
                                 bd.shape.dims.data(),
                                 true,
                                 coords,
                                 in_bounds);

    const bool use_span_tiling = span_tiling_enabled_from_env("NODUS_GATHER_SPAN_TILING");
    if (use_span_tiling) {
        SpanOpConfig span_op{};
        span_op.aggregate = SpanAggregateOp::Add;
        span_op.clamp = clamp;
        if (span_gather_add_2d<Scalar>(coords,
                                       in_bounds,
                                       count,
                                       width,
                                       height,
                                       channels,
                                       bmap.data,
                                       omap.data,
                                       base.backend(),
                                       0u,
                                       span_op)) {
            bmap.unmap();
            omap.unmap();
            release_coord_buffer(base.backend(), coord_buf);
            if (points_match) {
                pmap.unmap();
            } else if (points_ptr) {
                auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
                if (mem) mem->unmap(points.handle());
            }
            return true;
        }
    }

    struct GatherCtx {
        const Scalar* base = nullptr;
        Scalar* out = nullptr;
        const int64_t* coords = nullptr;
        const uint8_t* in_bounds = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t channels = 0;
        uint32_t pcols = 0;
        bool clamp = false;
    };

    GatherCtx ctx{};
    ctx.base = bmap.data;
    ctx.out = omap.data;
    ctx.coords = coords;
    ctx.in_bounds = in_bounds;
    ctx.width = width;
    ctx.height = height;
    ctx.channels = channels;
    ctx.pcols = pcols;
    ctx.clamp = clamp;

    auto gather_rows = [](const void* vctx, uint32_t i0, uint32_t i1) {
        const auto* ctx = static_cast<const GatherCtx*>(vctx);
        for (uint32_t i = i0; i < i1; ++i) {
            if (!ctx->in_bounds[i]) {
                if (ctx->clamp) continue;
                continue;
            }

            const int64_t xi = ctx->coords[static_cast<size_t>(i) * 2 + 0];
            const int64_t yi = ctx->coords[static_cast<size_t>(i) * 2 + 1];

            const uint64_t base_idx = (static_cast<uint64_t>(yi) * ctx->width +
                                       static_cast<uint64_t>(xi)) * ctx->channels;
            if (ctx->channels == 1) {
                ctx->out[i] += ctx->base[base_idx];
            } else {
                Scalar* dst = ctx->out + static_cast<uint64_t>(i) * ctx->channels;
                const Scalar* src = ctx->base + base_idx;
                for (uint32_t c = 0; c < ctx->channels; ++c) {
                    dst[c] += src[c];
                }
            }
        }
    };

    submit_row_jobs(tensor_op_pool(), gather_rows, &ctx, 0, count);

    bmap.unmap();
    omap.unmap();
    release_coord_buffer(base.backend(), coord_buf);
    if (points_match) {
        pmap.unmap();
    } else if (points_ptr) {
        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (mem) mem->unmap(points.handle());
    }
    return true;
}

template <typename Scalar, TensorDType DTypeValue>
static bool scatter_add_2d_typed(const AbstractTensor& base,
                                 const AbstractTensor& points,
                                 const AbstractTensor& values,
                                 AbstractTensor* out,
                                 bool clamp) {
    if (!out) return false;
    out->reset();
    if (!base.valid() || !points.valid() || !values.valid()) return false;
    if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
    const TensorDesc& bd = base.desc();
    const TensorDesc& pd = points.desc();
    const TensorDesc& vd = values.desc();
    if (bd.dtype != DTypeValue || vd.dtype != DTypeValue) return false;
    const bool points_match = (pd.dtype == DTypeValue);
    if (!points_match) {
        if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
            pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
            return false;
        }
    }
    if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
        return false;

    if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
    const uint32_t height = bd.shape.dims[0];
    const uint32_t width = bd.shape.dims[1];
    const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

    if (pd.shape.dims.size() != 2) return false;
    const uint32_t count = pd.shape.dims[0];
    const uint32_t pcols = pd.shape.dims[1];
    if (pcols != 2 && pcols != 3) return false;

    bool val_scalar = false;
    if (vd.shape.dims.size() == 1) {
        if (vd.shape.dims[0] != count) return false;
        val_scalar = true;
    } else if (vd.shape.dims.size() == 2) {
        if (vd.shape.dims[0] != count) return false;
        if (vd.shape.dims[1] != channels) return false;
    } else {
        return false;
    }

    TensorDesc out_desc = bd;
    *out = AbstractTensor::create(out_desc, base.backend());
    if (!out->valid()) return false;

    (void)ensure_tensor_zeroed(*out);
    auto bmap = map_dense_typed<Scalar, DTypeValue>(base);
    auto omap = map_dense_typed<Scalar, DTypeValue>(*out);
    auto vmap = map_dense_typed<Scalar, DTypeValue>(values);
    MappedDenseTyped<Scalar, DTypeValue> pmap{};
    void* points_ptr = nullptr;
    size_t points_bytes = 0;
    if (points_match) {
        pmap = map_dense_typed<Scalar, DTypeValue>(points);
        points_ptr = pmap.data;
    } else {
        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
            bmap.unmap();
            omap.unmap();
            vmap.unmap();
            return false;
        }
    }
    if (!bmap.ok || !omap.ok || !vmap.ok || (points_match && !pmap.ok)) {
        bmap.unmap();
        omap.unmap();
        vmap.unmap();
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return false;
    }

    if constexpr (std::is_same_v<Scalar, float>) {
        if (!tensor_copy_f32_into(base, out)) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            return false;
        }
        bmap.unmap();
        omap.unmap();
        auto omap2 = map_dense_typed<Scalar, DTypeValue>(*out);
        if (!omap2.ok) {
            pmap.unmap();
            vmap.unmap();
            return false;
        }
        omap = omap2;
    } else {
        const uint64_t elems = bd.shape.element_count();
        std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
    }

    CoordBuffer coord_buf{};
    if (!acquire_coord_buffer(count, 2u, base.backend(), coord_buf)) {
        bmap.unmap();
        omap.unmap();
        vmap.unmap();
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return false;
    }
    int64_t* coords = coord_buf.coords_ptr;
    uint8_t* in_bounds = coord_buf.mask_ptr;

    const bool use_affine = bd.slice.valid && bd.slice.has_affine;

    convert_points_to_int_coords(pd,
                                 points_ptr,
                                 count,
                                 2u,
                                 points_match,
                                 use_affine,
                                 bd.slice.affine,
                                 bd.shape.dims.data(),
                                 true,
                                 coords,
                                 in_bounds);

    bmap.unmap();
    omap.unmap();

    if (dyadic_scatter_from_coords_auto<Scalar, policies::Add, policies::Add>(
            coord_buf,
            count,
            2u,
            bd.shape.dims.data(),
            channels,
            val_scalar,
            vmap.data,
            *out,
            base.backend())) {
        vmap.unmap();
        release_coord_buffer(base.backend(), coord_buf);
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return true;
    }

    vmap.unmap();
    release_coord_buffer(base.backend(), coord_buf);
    if (points_match) {
        pmap.unmap();
    } else if (points_ptr) {
        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (mem) mem->unmap(points.handle());
    }
    return false;

}

template <typename Scalar, TensorDType DTypeValue>
static bool scatter_2d_typed(const AbstractTensor& base,
                             const AbstractTensor& points,
                             const AbstractTensor& values,
                             AbstractTensor* out,
                             bool clamp) {
    if (!out) return false;
    out->reset();
    if (!base.valid() || !points.valid() || !values.valid()) return false;
    if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
    const TensorDesc& bd = base.desc();
    const TensorDesc& pd = points.desc();
    const TensorDesc& vd = values.desc();
    if (bd.dtype != DTypeValue || vd.dtype != DTypeValue) return false;
    const bool points_match = (pd.dtype == DTypeValue);
    if (!points_match) {
        if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
            pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
            return false;
        }
    }
    if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
        return false;

    if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
    const uint32_t height = bd.shape.dims[0];
    const uint32_t width = bd.shape.dims[1];
    const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

    if (pd.shape.dims.size() != 2) return false;
    const uint32_t count = pd.shape.dims[0];
    const uint32_t pcols = pd.shape.dims[1];
    if (pcols != 2 && pcols != 3) return false;

    bool val_scalar = false;
    if (vd.shape.dims.size() == 1) {
        if (vd.shape.dims[0] != count) return false;
        val_scalar = true;
    } else if (vd.shape.dims.size() == 2) {
        if (vd.shape.dims[0] != count) return false;
        if (vd.shape.dims[1] != channels) return false;
    } else {
        return false;
    }

    TensorDesc out_desc = bd;
    *out = AbstractTensor::create(out_desc, base.backend());
    if (!out->valid()) return false;

    auto bmap = map_dense_typed<Scalar, DTypeValue>(base);
    auto omap = map_dense_typed<Scalar, DTypeValue>(*out);
    auto vmap = map_dense_typed<Scalar, DTypeValue>(values);
    MappedDenseTyped<Scalar, DTypeValue> pmap{};
    void* points_ptr = nullptr;
    size_t points_bytes = 0;
    if (points_match) {
        pmap = map_dense_typed<Scalar, DTypeValue>(points);
        points_ptr = pmap.data;
    } else {
        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
            bmap.unmap();
            omap.unmap();
            vmap.unmap();
            return false;
        }
    }
    if (!bmap.ok || !omap.ok || !vmap.ok || (points_match && !pmap.ok)) {
        bmap.unmap();
        omap.unmap();
        vmap.unmap();
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return false;
    }

    if constexpr (std::is_same_v<Scalar, float>) {
        if (!tensor_copy_f32_into(base, out)) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            return false;
        }
        bmap.unmap();
        omap.unmap();
        auto omap2 = map_dense_typed<Scalar, DTypeValue>(*out);
        if (!omap2.ok) {
            pmap.unmap();
            vmap.unmap();
            return false;
        }
        omap = omap2;
    } else {
        const uint64_t elems = bd.shape.element_count();
        std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
    }

    CoordBuffer coord_buf{};
    if (!acquire_coord_buffer(count, 2u, base.backend(), coord_buf)) {
        bmap.unmap();
        omap.unmap();
        vmap.unmap();
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return false;
    }
    int64_t* coords = coord_buf.coords_ptr;
    uint8_t* in_bounds = coord_buf.mask_ptr;

    const uint32_t tile_px = default_tile_px_2d(bd);

    const bool use_affine = bd.slice.valid && bd.slice.has_affine;

    convert_points_to_int_coords(pd,
                                 points_ptr,
                                 count,
                                 2u,
                                 points_match,
                                 use_affine,
                                 bd.slice.affine,
                                 bd.shape.dims.data(),
                                 true,
                                 coords,
                                 in_bounds);

    auto& bins = tile_bins_2d();
    const TileBinning2D tile = build_tile_bins_2d(bd, coords, in_bounds,
                                                  count, width, height, pcols, bins);

    struct ScatterCtx {
        const int64_t* coords = nullptr;
        const Scalar* values = nullptr;
        Scalar* out = nullptr;
        const uint32_t* offsets = nullptr;
        const uint32_t* indices = nullptr;
        const uint64_t* tile_mask = nullptr;
        uint32_t tile_count = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t channels = 0;
        uint32_t pcols = 0;
        uint32_t tiles_x = 0;
        uint32_t tile_px = 0;
        bool val_scalar = false;
        bool dense_grid = false;
        float dense_threshold = 0.0f;
    };

    auto scatter_rows = [](const void* vctx, uint32_t t0, uint32_t t1) {
        const auto* ctx = static_cast<const ScatterCtx*>(vctx);
        const uint32_t width = ctx->width;
        const uint32_t channels = ctx->channels;

        for (uint32_t ti = t0; ti < t1; ++ti) {
            const uint32_t tid = ti;
            const uint32_t word = tid >> 6;
            const uint32_t bit = tid & 63u;
            if ((ctx->tile_mask[word] & (1ull << bit)) == 0ull) continue;
            const uint32_t begin = ctx->offsets[tid];
            const uint32_t end = ctx->offsets[tid + 1];
            if (begin == end) continue;
            auto tile_lock = acquire_tile_lock(tid);
            const uint32_t ty = tid / ctx->tiles_x;
            const uint32_t tx = tid - ty * ctx->tiles_x;
            const uint32_t tile_origin_x = tx * ctx->tile_px;
            const uint32_t tile_origin_y = ty * ctx->tile_px;
            const uint32_t y_end = std::min<uint32_t>(ctx->height, tile_origin_y + ctx->tile_px);
            const uint32_t x_end = std::min<uint32_t>(ctx->width, tile_origin_x + ctx->tile_px);
            const uint32_t rows = y_end - tile_origin_y;
            const uint32_t cols = x_end - tile_origin_x;

            const uint32_t hits = end - begin;
            const uint32_t tile_area = rows * cols;
            if (ctx->dense_grid && tile_area > 0) {
                const float density = static_cast<float>(hits) / static_cast<float>(tile_area);
                if (density >= ctx->dense_threshold) {
                    const uint32_t out_row_stride = width * channels;
                    Scalar* out_ptr =
                        ctx->out + (static_cast<uint64_t>(tile_origin_y) * width + tile_origin_x) * channels;
                    for (uint32_t ly = 0; ly < rows; ++ly) {
                        const uint32_t y = tile_origin_y + ly;
                        Scalar* dst = out_ptr + static_cast<uint64_t>(ly) * out_row_stride;
                        if (ctx->val_scalar) {
                            const Scalar* src = ctx->values + static_cast<size_t>(y) * width + tile_origin_x;
                            for (uint32_t lx = 0; lx < cols; ++lx) {
                                const Scalar v = src[lx];
                                for (uint32_t c = 0; c < channels; ++c) {
                                    dst[lx * channels + c] = v;
                                }
                            }
                        } else {
                            const Scalar* src = ctx->values +
                                (static_cast<size_t>(y) * width + tile_origin_x) * channels;
                            for (uint32_t lx = 0; lx < cols; ++lx) {
                                const Scalar* src_row = src + static_cast<uint64_t>(lx) * channels;
                                Scalar* dst_row = dst + static_cast<uint64_t>(lx) * channels;
                                for (uint32_t c = 0; c < channels; ++c) {
                                    dst_row[c] = src_row[c];
                                }
                            }
                        }
                    }
                    continue;
                }
            }

            for (uint32_t idx = begin; idx < end; ++idx) {
                const uint32_t i = ctx->indices[idx];
                const int64_t xi = ctx->coords[static_cast<size_t>(i) * 2 + 0];
                const int64_t yi = ctx->coords[static_cast<size_t>(i) * 2 + 1];
                if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(width) ||
                    yi >= static_cast<int64_t>(ctx->height)) {
                    continue;
                }
                const uint64_t base_idx = (static_cast<uint64_t>(yi) * width +
                                           static_cast<uint64_t>(xi)) * channels;
                if (ctx->val_scalar) {
                    const Scalar v = ctx->values[i];
                    for (uint32_t c = 0; c < channels; ++c) {
                        ctx->out[base_idx + c] = v;
                    }
                } else {
                    const Scalar* src = ctx->values + static_cast<uint64_t>(i) * channels;
                    for (uint32_t c = 0; c < channels; ++c) {
                        ctx->out[base_idx + c] = src[c];
                    }
                }
            }
        }
    };

    ScatterCtx ctx{};
    ctx.coords = coords;
    ctx.values = vmap.data;
    ctx.out = omap.data;
    ctx.offsets = bins.offsets.data();
    ctx.indices = bins.indices.data();
    ctx.tile_mask = bins.tile_mask.data();
    ctx.tile_count = tile.tile_count;
    ctx.width = width;
    ctx.height = height;
    ctx.channels = channels;
    ctx.pcols = pcols;
    ctx.tiles_x = tile.tiles_x;
    ctx.tile_px = tile.tile_px;
    ctx.val_scalar = val_scalar;
    ctx.dense_grid = tile.dense_grid;
    ctx.dense_threshold = tile.dense_threshold;

    submit_row_jobs(tensor_op_pool(), scatter_rows, &ctx, 0, ctx.tile_count);

    bmap.unmap();
    omap.unmap();
    vmap.unmap();
    release_coord_buffer(base.backend(), coord_buf);
    if (points_match) {
        pmap.unmap();
    } else if (points_ptr) {
        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (mem) mem->unmap(points.handle());
    }
    return true;
}

template <typename Scalar, TensorDType DTypeValue>
static bool scatter_nd_typed(const AbstractTensor& base,
                             const AbstractTensor& points,
                             const AbstractTensor& values,
                             AbstractTensor* out,
                             bool clamp) {
    if (!out) return false;
    out->reset();
    if (!base.valid() || !points.valid() || !values.valid()) return false;
    if (base.backend() != points.backend() || base.backend() != values.backend()) return false;

    const TensorDesc& bd = base.desc();
    const TensorDesc& pd = points.desc();
    const TensorDesc& vd = values.desc();
    if (bd.dtype != DTypeValue || vd.dtype != DTypeValue) return false;
    const bool points_match = (pd.dtype == DTypeValue);
    if (!points_match) {
        if (pd.dtype != TensorDType::I32 && pd.dtype != TensorDType::I64 &&
            pd.dtype != TensorDType::U32 && pd.dtype != TensorDType::U64) {
            return false;
        }
    }
    if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
        return false;
    if (pd.shape.dims.size() != 2) return false;
    const uint32_t count = pd.shape.dims[0];
    const uint32_t dims = pd.shape.dims[1];
    if (dims == 0) return false;

    const size_t out_rank = bd.shape.dims.size();
    if (out_rank < dims || out_rank > dims + 1) return false;

    if (dims == 2) return scatter_2d_typed<Scalar, DTypeValue>(base, points, values, out, clamp);

    const uint32_t channels = (out_rank == dims + 1) ? bd.shape.dims.back() : 1u;
    bool val_scalar = false;
    if (vd.shape.dims.size() == 1) {
        if (vd.shape.dims[0] != count) return false;
        val_scalar = true;
    } else if (vd.shape.dims.size() == 2) {
        if (vd.shape.dims[0] != count) return false;
        if (vd.shape.dims[1] != channels) return false;
    } else {
        return false;
    }

    TensorDesc out_desc = bd;
    *out = AbstractTensor::create(out_desc, base.backend());
    if (!out->valid()) return false;

    auto bmap = map_dense_typed<Scalar, DTypeValue>(base);
    auto omap = map_dense_typed<Scalar, DTypeValue>(*out);
    auto vmap = map_dense_typed<Scalar, DTypeValue>(values);
    MappedDenseTyped<Scalar, DTypeValue> pmap{};
    void* points_ptr = nullptr;
    size_t points_bytes = 0;
    if (points_match) {
        pmap = map_dense_typed<Scalar, DTypeValue>(points);
        points_ptr = pmap.data;
    } else {
        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (!mem || !mem->map(points.handle(), &points_ptr, &points_bytes)) {
            bmap.unmap();
            omap.unmap();
            vmap.unmap();
            return false;
        }
    }
    if (!bmap.ok || !omap.ok || !vmap.ok || (points_match && !pmap.ok)) {
        bmap.unmap();
        omap.unmap();
        vmap.unmap();
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return false;
    }

    if constexpr (std::is_same_v<Scalar, float>) {
        if (!tensor_copy_f32_into(base, out)) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            return false;
        }
        bmap.unmap();
        omap.unmap();
        auto omap2 = map_dense_typed<Scalar, DTypeValue>(*out);
        if (!omap2.ok) {
            pmap.unmap();
            vmap.unmap();
            return false;
        }
        omap = omap2;
    } else {
        const uint64_t elems = bd.shape.element_count();
        std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
    }

    std::vector<uint64_t> strides;
    if (!compute_dense_strides_u64(bd.shape.dims, strides)) {
        vmap.unmap();
        omap.unmap();
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return false;
    }

    const bool use_affine = bd.slice.valid && bd.slice.has_affine && dims <= 3;
    CoordBuffer coord_buf{};
    if (!acquire_coord_buffer(count, dims, base.backend(), coord_buf)) {
        vmap.unmap();
        omap.unmap();
        if (points_match) {
            pmap.unmap();
        } else if (points_ptr) {
            auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
            if (mem) mem->unmap(points.handle());
        }
        return false;
    }
    int64_t* coords = coord_buf.coords_ptr;
    uint8_t* in_bounds = coord_buf.mask_ptr;
    convert_points_to_int_coords(pd,
                                 points_ptr,
                                 count,
                                 dims,
                                 points_match,
                                 use_affine,
                                 bd.slice.affine,
                                 bd.shape.dims.data(),
                                 true,
                                 coords,
                                 in_bounds);

    for (uint32_t i = 0; i < count; ++i) {
        if (!in_bounds[i]) {
            if (clamp) continue;
            continue;
        }

        uint64_t base_idx = 0;
        const int64_t* coord = coords + static_cast<size_t>(i) * dims;
        for (uint32_t d = 0; d < dims; ++d) {
            base_idx += static_cast<uint64_t>(coord[d]) * strides[d];
        }

        if (channels == 1) {
            const Scalar v = val_scalar ? vmap.data[i] : vmap.data[i * channels];
            omap.data[base_idx] = v;
        } else {
            if (val_scalar) {
                const Scalar v = vmap.data[i];
                for (uint32_t c = 0; c < channels; ++c) {
                    omap.data[base_idx + c] = v;
                }
            } else {
                const Scalar* src = vmap.data + static_cast<uint64_t>(i) * channels;
                for (uint32_t c = 0; c < channels; ++c) {
                    omap.data[base_idx + c] = src[c];
                }
            }
        }
    }

    vmap.unmap();
    omap.unmap();
    release_coord_buffer(base.backend(), coord_buf);
    if (points_match) {
        pmap.unmap();
    } else if (points_ptr) {
        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (mem) mem->unmap(points.handle());
    }
    return true;
}
} // namespace

TileShape2D choose_tile_shape_2d(const TensorDesc& desc,
                                 uint32_t radius_x,
                                 uint32_t radius_y,
                                 uint64_t cache_budget_bytes,
                                 TileContiguityStrategy strategy) {
    return choose_tile_shape_2d_impl(desc, radius_x, radius_y, cache_budget_bytes, strategy);
}

void submit_row_jobs(::nodus::ThreadPool* pool,
                     RowRangeFn fn,
                     const void* ctx,
                     uint32_t y0,
                     uint32_t y1) {
    if (!pool || y1 <= y0) {
        if (fn) fn(ctx, y0, y1);
        return;
    }

    const uint32_t span = y1 - y0;
    if (!should_parallelize(pool, span)) {
        if (fn) fn(ctx, y0, y1);
        return;
    }

    const uint32_t max_jobs = pool->thread_count();

    const uint32_t job_count = std::min<uint32_t>(max_jobs, span);
    const uint32_t chunk = (span + job_count - 1) / job_count;

    struct RowJobParams {
        RowRangeFn fn = nullptr;
        const void* ctx = nullptr;
        uint32_t y0 = 0;
        uint32_t y1 = 0;
    };

    std::vector<RowJobParams> params(job_count);
    std::vector<nodus::ThreadPool::Job> jobs(job_count);

    auto job_fn = [](const nodus::ThreadPool::Job& job, uint32_t /*worker_id*/) {
        const auto* p = static_cast<const RowJobParams*>(job.params);
        p->fn(p->ctx, p->y0, p->y1);
    };

    uint32_t count = 0;
    for (uint32_t i = 0; i < job_count; ++i) {
        const uint32_t begin = y0 + i * chunk;
        const uint32_t end = std::min<uint32_t>(y1, begin + chunk);
        if (begin >= end) break;
        params[count] = RowJobParams{fn, ctx, begin, end};
        nodus::ThreadPool::Job job{};
        job.fn = job_fn;
        job.params = &params[count];
        job.params_size = static_cast<uint32_t>(sizeof(RowJobParams));
        jobs[count] = job;
        ++count;
    }

    auto batch = pool->submit_batch(jobs.data(), count);
    if (batch) batch->wait();
}

::nodus::ThreadPool* tensor_op_pool() {
    static nodus::ThreadPool* pool = nullptr;
    static uint32_t pool_threads = 0;

    const char* v = std::getenv("NODUS_TENSOR_OP_THREADS");
    if (!v || !*v) {
        return nullptr;
    }
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(v, &end, 10);
    if (end == v || parsed < 2) {
        return nullptr;
    }
    const uint32_t desired = static_cast<uint32_t>(parsed);
    if (pool && pool_threads == desired) {
        return pool;
    }
    if (pool) {
        delete pool;
        pool = nullptr;
        pool_threads = 0;
    }
    nodus::ThreadPool::Options opt{};
    opt.thread_count = desired;
    opt.start_immediately = true;
    pool = new nodus::ThreadPool(opt);
    pool_threads = desired;
    return pool;
}

bool tensor_axpby_f32(const AbstractTensor& a,
                      float alpha,
                      const AbstractTensor& b,
                      float beta,
                      AbstractTensor* out) {
    if (!out || !out->valid() || !a.valid() || !b.valid()) return false;

    TensorOpPlan plan{};
    if (!build_axpby_plan(a, b, *out, plan)) return false;

    execute_axpby_plan(plan, alpha, beta);

    auto* mem = dynamic_cast<InMemoryBackend*>(a.backend());
    if (mem) {
        mem->unmap(out->handle());
        mem->unmap(b.handle());
        mem->unmap(a.handle());
    }
    return true;
}

bool tensor_copy_f32_into(const AbstractTensor& src, AbstractTensor* dst) {
    if (!dst || !dst->valid() || !src.valid()) return false;

    TensorOpPlan plan{};
    if (!build_unary_plan(src, *dst, plan)) return false;

    execute_copy_plan(plan);

    auto* mem = dynamic_cast<InMemoryBackend*>(src.backend());
    if (mem) {
        mem->unmap(dst->handle());
        mem->unmap(src.handle());
    }
    return true;
}

#define NODUS_TENSOR_MATH_DEFINE(SUFFIX, SCALAR, DTYPE)                              \
    AbstractTensor tensor_matmul_##SUFFIX(const AbstractTensor& a,                  \
                                          const AbstractTensor& b) {                \
        return TensorMathImpl<SCALAR, DTYPE>::matmul(a, b);                          \
    }                                                                               \
    AbstractTensor tensor_affine_identity_##SUFFIX(TensorBackend* backend) {        \
        return TensorMathImpl<SCALAR, DTYPE>::affine_identity(backend);             \
    }                                                                               \
    AbstractTensor tensor_affine_translation_##SUFFIX(const AbstractTensor& t) {    \
        return TensorMathImpl<SCALAR, DTYPE>::affine_translation(t);                \
    }                                                                               \
    AbstractTensor tensor_affine_scale_##SUFFIX(const AbstractTensor& s) {          \
        return TensorMathImpl<SCALAR, DTYPE>::affine_scale(s);                      \
    }                                                                               \
    AbstractTensor tensor_affine_from_quat_translation_##SUFFIX(                    \
        const AbstractTensor& q, const AbstractTensor& t) {                         \
        return TensorMathImpl<SCALAR, DTYPE>::affine_from_quat_translation(q, t);   \
    }                                                                               \
    AbstractTensor tensor_quat_identity_##SUFFIX(TensorBackend* backend) {          \
        return TensorMathImpl<SCALAR, DTYPE>::quat_identity(backend);               \
    }                                                                               \
    AbstractTensor tensor_quat_from_axis_angle_##SUFFIX(const AbstractTensor& axis, \
                                                        SCALAR angle) {             \
        return TensorMathImpl<SCALAR, DTYPE>::quat_from_axis_angle(axis, angle);    \
    }                                                                               \
    AbstractTensor tensor_quat_normalize_##SUFFIX(const AbstractTensor& q) {        \
        return TensorMathImpl<SCALAR, DTYPE>::quat_normalize(q);                    \
    }                                                                               \
    AbstractTensor tensor_quat_mul_##SUFFIX(const AbstractTensor& a,                \
                                            const AbstractTensor& b) {              \
        return TensorMathImpl<SCALAR, DTYPE>::quat_mul(a, b);                        \
    }                                                                               \
    AbstractTensor tensor_quat_to_mat4_##SUFFIX(const AbstractTensor& q,            \
                                                const AbstractTensor& t) {          \
        return TensorMathImpl<SCALAR, DTYPE>::quat_to_mat4(q, t);                    \
    }                                                                               \
    AbstractTensor tensor_transform_points_##SUFFIX(const AbstractTensor& points,  \
                                                     const AbstractTensor& mat4) {  \
        return TensorMathImpl<SCALAR, DTYPE>::transform_points(points, mat4);       \
    }

NODUS_TENSOR_MATH_DEFINE(f32, float, TensorDType::F32)
NODUS_TENSOR_MATH_DEFINE(f64, double, TensorDType::F64)

#undef NODUS_TENSOR_MATH_DEFINE

bool tensor_intersect_plane_z_f32(const AbstractTensor& origins,
                                  const AbstractTensor& dirs,
                                  float plane_z,
                                  AbstractTensor* out_hits,
                                  AbstractTensor* out_mask) {
    return TensorMathImpl<float, TensorDType::F32>::intersect_plane_z(
        origins, dirs, plane_z, out_hits, out_mask);
}

bool tensor_intersect_plane_z_f64(const AbstractTensor& origins,
                                  const AbstractTensor& dirs,
                                  double plane_z,
                                  AbstractTensor* out_hits,
                                  AbstractTensor* out_mask) {
    return TensorMathImpl<double, TensorDType::F64>::intersect_plane_z(
        origins, dirs, plane_z, out_hits, out_mask);
}

#define NODUS_TENSOR_SCATTER_ADD_2D_WRAP(SUFFIX, ...)                              \
    bool tensor_scatter_add_2d_##SUFFIX(const AbstractTensor& base,                \
                                        const AbstractTensor& points,             \
                                        const AbstractTensor& values,             \
                                        AbstractTensor* out,                       \
                                        bool clamp) {                              \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_SCATTER_ADD_2D_WRAP_TYPED(SUFFIX, TYPE, DTYPE)               \
    NODUS_TENSOR_SCATTER_ADD_2D_WRAP(                                              \
        SUFFIX,                                                                    \
        scatter_add_2d_typed<TYPE, DTYPE>(base, points, values, out, clamp))

#define NODUS_TENSOR_GATHER_2D_WRAP(SUFFIX, ...)                                   \
    bool tensor_gather_2d_##SUFFIX(const AbstractTensor& base,                     \
                                   const AbstractTensor& points,                  \
                                   AbstractTensor* out,                            \
                                   bool clamp) {                                   \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_GATHER_2D_WRAP_TYPED(SUFFIX, TYPE, DTYPE)                     \
    NODUS_TENSOR_GATHER_2D_WRAP(                                                   \
        SUFFIX,                                                                    \
        gather_2d_typed<TYPE, DTYPE>(base, points, out, clamp))

#define NODUS_TENSOR_GATHER_ADD_2D_WRAP(SUFFIX, ...)                               \
    bool tensor_gather_add_2d_##SUFFIX(const AbstractTensor& base,                 \
                                       const AbstractTensor& points,              \
                                       AbstractTensor* out,                        \
                                       bool clamp) {                               \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_GATHER_ADD_2D_WRAP_TYPED(SUFFIX, TYPE, DTYPE)                 \
    NODUS_TENSOR_GATHER_ADD_2D_WRAP(                                               \
        SUFFIX,                                                                    \
        gather_add_2d_typed<TYPE, DTYPE>(base, points, out, clamp))

NODUS_TENSOR_SCATTER_ADD_2D_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::scatter_add_2d(base, points, values, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::scatter_add_2d(base, points, values, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_WRAP_TYPED(i8, int8_t, TensorDType::I8)
NODUS_TENSOR_SCATTER_ADD_2D_WRAP_TYPED(i16, int16_t, TensorDType::I16)
NODUS_TENSOR_SCATTER_ADD_2D_WRAP_TYPED(i32, int32_t, TensorDType::I32)
NODUS_TENSOR_SCATTER_ADD_2D_WRAP_TYPED(i64, int64_t, TensorDType::I64)
NODUS_TENSOR_SCATTER_ADD_2D_WRAP_TYPED(u8, uint8_t, TensorDType::U8)
NODUS_TENSOR_SCATTER_ADD_2D_WRAP_TYPED(u16, uint16_t, TensorDType::U16)
NODUS_TENSOR_SCATTER_ADD_2D_WRAP_TYPED(u32, uint32_t, TensorDType::U32)
NODUS_TENSOR_SCATTER_ADD_2D_WRAP_TYPED(u64, uint64_t, TensorDType::U64)

NODUS_TENSOR_GATHER_2D_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::gather_2d(base, points, out, clamp))
NODUS_TENSOR_GATHER_2D_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::gather_2d(base, points, out, clamp))
NODUS_TENSOR_GATHER_2D_WRAP_TYPED(i8, int8_t, TensorDType::I8)
NODUS_TENSOR_GATHER_2D_WRAP_TYPED(i16, int16_t, TensorDType::I16)
NODUS_TENSOR_GATHER_2D_WRAP_TYPED(i32, int32_t, TensorDType::I32)
NODUS_TENSOR_GATHER_2D_WRAP_TYPED(i64, int64_t, TensorDType::I64)
NODUS_TENSOR_GATHER_2D_WRAP_TYPED(u8, uint8_t, TensorDType::U8)
NODUS_TENSOR_GATHER_2D_WRAP_TYPED(u16, uint16_t, TensorDType::U16)
NODUS_TENSOR_GATHER_2D_WRAP_TYPED(u32, uint32_t, TensorDType::U32)
NODUS_TENSOR_GATHER_2D_WRAP_TYPED(u64, uint64_t, TensorDType::U64)

NODUS_TENSOR_GATHER_ADD_2D_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::gather_add_2d(base, points, out, clamp))
NODUS_TENSOR_GATHER_ADD_2D_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::gather_add_2d(base, points, out, clamp))
NODUS_TENSOR_GATHER_ADD_2D_WRAP_TYPED(i8, int8_t, TensorDType::I8)
NODUS_TENSOR_GATHER_ADD_2D_WRAP_TYPED(i16, int16_t, TensorDType::I16)
NODUS_TENSOR_GATHER_ADD_2D_WRAP_TYPED(i32, int32_t, TensorDType::I32)
NODUS_TENSOR_GATHER_ADD_2D_WRAP_TYPED(i64, int64_t, TensorDType::I64)
NODUS_TENSOR_GATHER_ADD_2D_WRAP_TYPED(u8, uint8_t, TensorDType::U8)
NODUS_TENSOR_GATHER_ADD_2D_WRAP_TYPED(u16, uint16_t, TensorDType::U16)
NODUS_TENSOR_GATHER_ADD_2D_WRAP_TYPED(u32, uint32_t, TensorDType::U32)
NODUS_TENSOR_GATHER_ADD_2D_WRAP_TYPED(u64, uint64_t, TensorDType::U64)

#undef NODUS_TENSOR_SCATTER_ADD_2D_WRAP_TYPED
#undef NODUS_TENSOR_GATHER_2D_WRAP_TYPED
#undef NODUS_TENSOR_GATHER_ADD_2D_WRAP_TYPED
#undef NODUS_TENSOR_SCATTER_ADD_2D_WRAP
#undef NODUS_TENSOR_GATHER_2D_WRAP
#undef NODUS_TENSOR_GATHER_ADD_2D_WRAP

#define NODUS_TENSOR_STUB_FALSE(...) false

#define NODUS_TENSOR_SCATTER_2D_WRAP(SUFFIX, ...)                                 \
    bool tensor_scatter_2d_##SUFFIX(const AbstractTensor& base,                    \
                                    const AbstractTensor& points,                 \
                                    const AbstractTensor& values,                 \
                                    AbstractTensor* out,                           \
                                    bool clamp) {                                  \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_SCATTER_2D_WRAP_TYPED(SUFFIX, TYPE, DTYPE)                    \
    NODUS_TENSOR_SCATTER_2D_WRAP(                                                  \
        SUFFIX,                                                                    \
        scatter_2d_typed<TYPE, DTYPE>(base, points, values, out, clamp))

#define NODUS_TENSOR_SCATTER_ADD_ND_WRAP(SUFFIX, ...)                              \
    bool tensor_scatter_add_nd_##SUFFIX(const AbstractTensor& base,                \
                                        const AbstractTensor& points,             \
                                        const AbstractTensor& values,             \
                                        AbstractTensor* out,                       \
                                        bool clamp) {                              \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_SCATTER_ND_WRAP(SUFFIX, ...)                                  \
    bool tensor_scatter_nd_##SUFFIX(const AbstractTensor& base,                    \
                                    const AbstractTensor& points,                 \
                                    const AbstractTensor& values,                 \
                                    AbstractTensor* out,                           \
                                    bool clamp) {                                  \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_SCATTER_ND_WRAP_TYPED(SUFFIX, TYPE, DTYPE)                    \
    NODUS_TENSOR_SCATTER_ND_WRAP(                                                  \
        SUFFIX,                                                                    \
        scatter_nd_typed<TYPE, DTYPE>(base, points, values, out, clamp))

#define NODUS_TENSOR_GATHER_ND_WRAP(SUFFIX, ...)                                   \
    bool tensor_gather_nd_##SUFFIX(const AbstractTensor& base,                     \
                                   const AbstractTensor& points,                  \
                                   AbstractTensor* out,                            \
                                   bool clamp) {                                   \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_GATHER_ADD_ND_WRAP(SUFFIX, ...)                               \
    bool tensor_gather_add_nd_##SUFFIX(const AbstractTensor& base,                 \
                                       const AbstractTensor& points,              \
                                       AbstractTensor* out,                        \
                                       bool clamp) {                               \
        return (__VA_ARGS__);                                                      \
    }

NODUS_TENSOR_SCATTER_2D_WRAP_TYPED(i8, int8_t, TensorDType::I8)
NODUS_TENSOR_SCATTER_2D_WRAP_TYPED(i16, int16_t, TensorDType::I16)
NODUS_TENSOR_SCATTER_2D_WRAP_TYPED(i32, int32_t, TensorDType::I32)
NODUS_TENSOR_SCATTER_2D_WRAP_TYPED(i64, int64_t, TensorDType::I64)
NODUS_TENSOR_SCATTER_2D_WRAP_TYPED(u8, uint8_t, TensorDType::U8)
NODUS_TENSOR_SCATTER_2D_WRAP_TYPED(u16, uint16_t, TensorDType::U16)
NODUS_TENSOR_SCATTER_2D_WRAP_TYPED(u32, uint32_t, TensorDType::U32)
NODUS_TENSOR_SCATTER_2D_WRAP_TYPED(u64, uint64_t, TensorDType::U64)
NODUS_TENSOR_SCATTER_2D_WRAP_TYPED(f32, float, TensorDType::F32)
NODUS_TENSOR_SCATTER_2D_WRAP_TYPED(f64, double, TensorDType::F64)

NODUS_TENSOR_SCATTER_ADD_ND_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::scatter_add_nd(base, points, values, out, clamp))
NODUS_TENSOR_SCATTER_ADD_ND_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::scatter_add_nd(base, points, values, out, clamp))
NODUS_TENSOR_SCATTER_ADD_ND_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::scatter_add_nd(base, points, values, out, clamp))
NODUS_TENSOR_SCATTER_ADD_ND_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::scatter_add_nd(base, points, values, out, clamp))
NODUS_TENSOR_SCATTER_ADD_ND_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::scatter_add_nd(base, points, values, out, clamp))
NODUS_TENSOR_SCATTER_ADD_ND_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::scatter_add_nd(base, points, values, out, clamp))
NODUS_TENSOR_SCATTER_ADD_ND_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::scatter_add_nd(base, points, values, out, clamp))
NODUS_TENSOR_SCATTER_ADD_ND_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::scatter_add_nd(base, points, values, out, clamp))
NODUS_TENSOR_SCATTER_ADD_ND_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::scatter_add_nd(base, points, values, out, clamp))
NODUS_TENSOR_SCATTER_ADD_ND_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::scatter_add_nd(base, points, values, out, clamp))

NODUS_TENSOR_SCATTER_ND_WRAP_TYPED(i8, int8_t, TensorDType::I8)
NODUS_TENSOR_SCATTER_ND_WRAP_TYPED(i16, int16_t, TensorDType::I16)
NODUS_TENSOR_SCATTER_ND_WRAP_TYPED(i32, int32_t, TensorDType::I32)
NODUS_TENSOR_SCATTER_ND_WRAP_TYPED(i64, int64_t, TensorDType::I64)
NODUS_TENSOR_SCATTER_ND_WRAP_TYPED(u8, uint8_t, TensorDType::U8)
NODUS_TENSOR_SCATTER_ND_WRAP_TYPED(u16, uint16_t, TensorDType::U16)
NODUS_TENSOR_SCATTER_ND_WRAP_TYPED(u32, uint32_t, TensorDType::U32)
NODUS_TENSOR_SCATTER_ND_WRAP_TYPED(u64, uint64_t, TensorDType::U64)
NODUS_TENSOR_SCATTER_ND_WRAP_TYPED(f32, float, TensorDType::F32)
NODUS_TENSOR_SCATTER_ND_WRAP_TYPED(f64, double, TensorDType::F64)

NODUS_TENSOR_GATHER_ND_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::gather_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ND_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::gather_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ND_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::gather_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ND_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::gather_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ND_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::gather_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ND_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::gather_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ND_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::gather_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ND_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::gather_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ND_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::gather_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ND_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::gather_nd(base, points, out, clamp))

NODUS_TENSOR_GATHER_ADD_ND_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::gather_add_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ADD_ND_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::gather_add_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ADD_ND_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::gather_add_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ADD_ND_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::gather_add_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ADD_ND_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::gather_add_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ADD_ND_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::gather_add_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ADD_ND_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::gather_add_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ADD_ND_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::gather_add_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ADD_ND_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::gather_add_nd(base, points, out, clamp))
NODUS_TENSOR_GATHER_ADD_ND_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::gather_add_nd(base, points, out, clamp))

#undef NODUS_TENSOR_SCATTER_2D_WRAP_TYPED
#undef NODUS_TENSOR_SCATTER_2D_WRAP
#undef NODUS_TENSOR_SCATTER_ADD_ND_WRAP
#undef NODUS_TENSOR_SCATTER_ND_WRAP_TYPED
#undef NODUS_TENSOR_SCATTER_ND_WRAP
#undef NODUS_TENSOR_GATHER_ND_WRAP
#undef NODUS_TENSOR_GATHER_ADD_ND_WRAP
#undef NODUS_TENSOR_STUB_FALSE

bool tensor_build_stencil_from_footprint_2d_f32(const TensorFootprint2D& footprint,
                                                StencilOrientation orientation,
                                                TensorStencil* out_stencil) {
    return TensorMathImpl<float, TensorDType::F32>::build_stencil_from_footprint_2d(
        footprint, orientation, out_stencil);
}

bool tensor_build_stencil_from_footprint_2d_f64(const TensorFootprint2D& footprint,
                                                StencilOrientation orientation,
                                                TensorStencil* out_stencil) {
    return TensorMathImpl<double, TensorDType::F64>::build_stencil_from_footprint_2d(
        footprint, orientation, out_stencil);
}

bool tensor_support_from_offsets(const AbstractTensor& offsets, TensorSupport* out_support) {
    if (!out_support) return false;
    out_support->reset();
    if (!offsets.valid()) return false;
    const TensorDesc& od = offsets.desc();
    if (od.dtype != TensorDType::I32) return false;
    if (od.layout != TensorLayout::Dense) return false;
    if (od.shape.dims.size() != 2) return false;
    const uint32_t count = od.shape.dims[0];
    const uint32_t dims = od.shape.dims[1];
    if (dims == 0) return false;

    auto* mem = dynamic_cast<InMemoryBackend*>(offsets.backend());
    if (!mem) return false;
    void* ptr = nullptr;
    size_t bytes = 0;
    if (!mem->map(offsets.handle(), &ptr, &bytes)) return false;
    const auto* off_ptr = static_cast<const int32_t*>(ptr);

    out_support->min_offset.assign(dims, std::numeric_limits<int32_t>::max());
    out_support->max_offset.assign(dims, std::numeric_limits<int32_t>::min());
    out_support->radius.assign(dims, 0);

    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t d = 0; d < dims; ++d) {
            const int32_t v = off_ptr[i * dims + d];
            out_support->min_offset[d] = std::min(out_support->min_offset[d], v);
            out_support->max_offset[d] = std::max(out_support->max_offset[d], v);
        }
    }

    for (uint32_t d = 0; d < dims; ++d) {
        const int32_t mn = out_support->min_offset[d];
        const int32_t mx = out_support->max_offset[d];
        out_support->radius[d] = std::max(std::abs(mn), std::abs(mx));
    }

    mem->unmap(offsets.handle());
    return true;
}

static bool add_into_f32(const AbstractTensor& src, AbstractTensor* dst) {
    if (!dst || !dst->valid() || !src.valid()) return false;
    if (src.backend() != dst->backend()) return false;
    if (src.desc().dtype != TensorDType::F32 || dst->desc().dtype != TensorDType::F32) return false;
    if (src.desc().layout != TensorLayout::Dense || dst->desc().layout != TensorLayout::Dense) return false;
    if (src.desc().shape.dims != dst->desc().shape.dims) return false;
    return tensor_axpby_f32(src, 1.0f, *dst, 1.0f, dst);
}

static bool add_into_f64(const AbstractTensor& src, AbstractTensor* dst) {
    if (!dst || !dst->valid() || !src.valid()) return false;
    if (src.backend() != dst->backend()) return false;
    if (src.desc().dtype != TensorDType::F64 || dst->desc().dtype != TensorDType::F64) return false;
    if (src.desc().layout != TensorLayout::Dense || dst->desc().layout != TensorLayout::Dense) return false;
    if (src.desc().shape.dims != dst->desc().shape.dims) return false;
    auto smap = map_dense_typed<double, TensorDType::F64>(src);
    auto dmap = map_dense_typed<double, TensorDType::F64>(*dst);
    if (!smap.ok || !dmap.ok) {
        smap.unmap();
        dmap.unmap();
        return false;
    }
    const size_t elems = static_cast<size_t>(src.desc().shape.element_count());
    for (size_t i = 0; i < elems; ++i) {
        dmap.data[i] += smap.data[i];
    }
    smap.unmap();
    dmap.unmap();
    return true;
}

template <typename Scalar, TensorDType DTypeValue>
static bool add_into_typed(const AbstractTensor& src, AbstractTensor* dst) {
    if (!dst || !dst->valid() || !src.valid()) return false;
    if (src.backend() != dst->backend()) return false;
    if (src.desc().dtype != DTypeValue || dst->desc().dtype != DTypeValue) return false;
    if (src.desc().layout != TensorLayout::Dense || dst->desc().layout != TensorLayout::Dense) return false;
    if (src.desc().shape.dims != dst->desc().shape.dims) return false;
    auto smap = map_dense_typed<Scalar, DTypeValue>(src);
    auto dmap = map_dense_typed<Scalar, DTypeValue>(*dst);
    if (!smap.ok || !dmap.ok) {
        smap.unmap();
        dmap.unmap();
        return false;
    }
    const size_t elems = static_cast<size_t>(src.desc().shape.element_count());
    for (size_t i = 0; i < elems; ++i) {
        dmap.data[i] += smap.data[i];
    }
    smap.unmap();
    dmap.unmap();
    return true;
}

#define NODUS_TENSOR_GATHER_STENCIL_2D_WRAP(SUFFIX, ...)                           \
    bool tensor_gather_stencil_2d_##SUFFIX(const AbstractTensor& base,             \
                                           const TensorStencil& stencil,          \
                                           AbstractTensor* out,                    \
                                           StencilBoundaryMode boundary,           \
                                           const AbstractTensor* target_mask,      \
                                           const AbstractTensor* source_mask,      \
                                           bool normalize) {                       \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_GATHER_ADD_STENCIL_2D_WRAP(SUFFIX, SCALAR, DTYPE)             \
    bool tensor_gather_add_stencil_2d_##SUFFIX(const AbstractTensor& base,         \
                                               const TensorStencil& stencil,      \
                                               AbstractTensor* out,                \
                                               StencilBoundaryMode boundary,       \
                                               const AbstractTensor* target_mask,  \
                                               const AbstractTensor* source_mask,  \
                                               bool normalize) {                   \
        if (!out || !out->valid()) return false;                                   \
        if (base.backend() != out->backend()) return false;                        \
        if (base.desc().shape.dims != out->desc().shape.dims) return false;        \
        AbstractTensor tmp;                                                        \
        if (!tensor_gather_stencil_2d_##SUFFIX(                                    \
            base, stencil, &tmp, boundary, target_mask, source_mask, normalize)) { \
            return false;                                                          \
        }                                                                          \
        return add_into_typed<SCALAR, DTYPE>(tmp, out);                             \
    }

#define NODUS_TENSOR_GATHER_STENCIL_ND_WRAP(SUFFIX, ...)                           \
    bool tensor_gather_stencil_nd_##SUFFIX(const AbstractTensor& base,             \
                                           const TensorStencil& stencil,          \
                                           AbstractTensor* out,                    \
                                           StencilBoundaryMode boundary,           \
                                           const AbstractTensor* target_mask,      \
                                           const AbstractTensor* source_mask,      \
                                           bool normalize) {                       \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_GATHER_ADD_STENCIL_ND_WRAP(SUFFIX, SCALAR, DTYPE)             \
    bool tensor_gather_add_stencil_nd_##SUFFIX(const AbstractTensor& base,         \
                                               const TensorStencil& stencil,      \
                                               AbstractTensor* out,                \
                                               StencilBoundaryMode boundary,       \
                                               const AbstractTensor* target_mask,  \
                                               const AbstractTensor* source_mask,  \
                                               bool normalize) {                   \
        if (!out || !out->valid()) return false;                                   \
        if (base.backend() != out->backend()) return false;                        \
        if (base.desc().shape.dims != out->desc().shape.dims) return false;        \
        AbstractTensor tmp;                                                        \
        if (!tensor_gather_stencil_nd_##SUFFIX(                                    \
            base, stencil, &tmp, boundary, target_mask, source_mask, normalize)) { \
            return false;                                                          \
        }                                                                          \
        return add_into_typed<SCALAR, DTYPE>(tmp, out);                             \
    }

#define NODUS_TENSOR_GATHER_FOOTPRINT_2D_WRAP(SUFFIX, ...)                         \
    bool tensor_gather_footprint_2d_##SUFFIX(const AbstractTensor& base,           \
                                             const TensorFootprint2D& footprint,  \
                                             AbstractTensor* out,                  \
                                             StencilOrientation orientation,       \
                                             StencilBoundaryMode boundary,         \
                                             const AbstractTensor* target_mask,    \
                                             const AbstractTensor* source_mask,    \
                                             bool normalize) {                     \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_GATHER_ADD_FOOTPRINT_2D_WRAP(SUFFIX, SCALAR, DTYPE)           \
    bool tensor_gather_add_footprint_2d_##SUFFIX(const AbstractTensor& base,       \
                                                 const TensorFootprint2D& footprint, \
                                                 AbstractTensor* out,              \
                                                 StencilOrientation orientation,   \
                                                 StencilBoundaryMode boundary,     \
                                                 const AbstractTensor* target_mask, \
                                                 const AbstractTensor* source_mask, \
                                                 bool normalize) {                 \
        TensorStencil stencil{};                                                   \
        if (!TensorMathImpl<SCALAR, DTYPE>::build_stencil_from_footprint_2d(       \
                footprint, orientation, &stencil)) {                              \
            return false;                                                          \
        }                                                                          \
        return tensor_gather_add_stencil_2d_##SUFFIX(                              \
            base, stencil, out, boundary, target_mask, source_mask, normalize);   \
    }

#define NODUS_TENSOR_SCATTER_STENCIL_2D_WRAP(SUFFIX, ...)                          \
    bool tensor_scatter_stencil_2d_##SUFFIX(const AbstractTensor& base,            \
                                            const AbstractTensor& points,         \
                                            const AbstractTensor& values,         \
                                            const TensorStencil& stencil,         \
                                            AbstractTensor* out,                  \
                                            StencilBoundaryMode boundary,         \
                                            bool clamp) {                          \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_SCATTER_STENCIL_ND_WRAP(SUFFIX, ...)                          \
    bool tensor_scatter_stencil_nd_##SUFFIX(const AbstractTensor& base,            \
                                            const AbstractTensor& points,         \
                                            const AbstractTensor& values,         \
                                            const TensorStencil& stencil,         \
                                            AbstractTensor* out,                  \
                                            StencilBoundaryMode boundary,         \
                                            bool clamp) {                          \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_SCATTER_FOOTPRINT_2D_WRAP(SUFFIX, ...)                        \
    bool tensor_scatter_footprint_2d_##SUFFIX(const AbstractTensor& base,          \
                                              const AbstractTensor& points,       \
                                              const AbstractTensor& values,       \
                                              const TensorFootprint2D& footprint, \
                                              AbstractTensor* out,                \
                                              StencilOrientation orientation,     \
                                              StencilBoundaryMode boundary,       \
                                              bool clamp) {                        \
        return (__VA_ARGS__);                                                      \
    }

NODUS_TENSOR_GATHER_STENCIL_2D_WRAP(u8,
    gather_stencil_2d_int_impl<uint8_t, TensorDType::U8>(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_2D_WRAP(u16,
    gather_stencil_2d_int_impl<uint16_t, TensorDType::U16>(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_2D_WRAP(u32,
    gather_stencil_2d_int_impl<uint32_t, TensorDType::U32>(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_2D_WRAP(u64,
    gather_stencil_2d_int_impl<uint64_t, TensorDType::U64>(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_2D_WRAP(i8,
    gather_stencil_2d_int_impl<int8_t, TensorDType::I8>(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_2D_WRAP(i16,
    gather_stencil_2d_int_impl<int16_t, TensorDType::I16>(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_2D_WRAP(i32,
    gather_stencil_2d_int_impl<int32_t, TensorDType::I32>(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_2D_WRAP(i64,
    gather_stencil_2d_int_impl<int64_t, TensorDType::I64>(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_2D_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::gather_stencil_2d(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_2D_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::gather_stencil_2d(
        base, stencil, out, boundary, target_mask, source_mask, normalize))

NODUS_TENSOR_GATHER_ADD_STENCIL_2D_WRAP(u8, uint8_t, TensorDType::U8)
NODUS_TENSOR_GATHER_ADD_STENCIL_2D_WRAP(u16, uint16_t, TensorDType::U16)
NODUS_TENSOR_GATHER_ADD_STENCIL_2D_WRAP(u32, uint32_t, TensorDType::U32)
NODUS_TENSOR_GATHER_ADD_STENCIL_2D_WRAP(u64, uint64_t, TensorDType::U64)
NODUS_TENSOR_GATHER_ADD_STENCIL_2D_WRAP(i8, int8_t, TensorDType::I8)
NODUS_TENSOR_GATHER_ADD_STENCIL_2D_WRAP(i16, int16_t, TensorDType::I16)
NODUS_TENSOR_GATHER_ADD_STENCIL_2D_WRAP(i32, int32_t, TensorDType::I32)
NODUS_TENSOR_GATHER_ADD_STENCIL_2D_WRAP(i64, int64_t, TensorDType::I64)
NODUS_TENSOR_GATHER_ADD_STENCIL_2D_WRAP(f32, float, TensorDType::F32)
NODUS_TENSOR_GATHER_ADD_STENCIL_2D_WRAP(f64, double, TensorDType::F64)

NODUS_TENSOR_GATHER_STENCIL_ND_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::gather_stencil_nd(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_ND_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::gather_stencil_nd(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_ND_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::gather_stencil_nd(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_ND_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::gather_stencil_nd(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_ND_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::gather_stencil_nd(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_ND_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::gather_stencil_nd(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_ND_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::gather_stencil_nd(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_ND_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::gather_stencil_nd(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_ND_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::gather_stencil_nd(
        base, stencil, out, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_STENCIL_ND_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::gather_stencil_nd(
        base, stencil, out, boundary, target_mask, source_mask, normalize))

NODUS_TENSOR_GATHER_ADD_STENCIL_ND_WRAP(u8, uint8_t, TensorDType::U8)
NODUS_TENSOR_GATHER_ADD_STENCIL_ND_WRAP(u16, uint16_t, TensorDType::U16)
NODUS_TENSOR_GATHER_ADD_STENCIL_ND_WRAP(u32, uint32_t, TensorDType::U32)
NODUS_TENSOR_GATHER_ADD_STENCIL_ND_WRAP(u64, uint64_t, TensorDType::U64)
NODUS_TENSOR_GATHER_ADD_STENCIL_ND_WRAP(i8, int8_t, TensorDType::I8)
NODUS_TENSOR_GATHER_ADD_STENCIL_ND_WRAP(i16, int16_t, TensorDType::I16)
NODUS_TENSOR_GATHER_ADD_STENCIL_ND_WRAP(i32, int32_t, TensorDType::I32)
NODUS_TENSOR_GATHER_ADD_STENCIL_ND_WRAP(i64, int64_t, TensorDType::I64)
NODUS_TENSOR_GATHER_ADD_STENCIL_ND_WRAP(f32, float, TensorDType::F32)
NODUS_TENSOR_GATHER_ADD_STENCIL_ND_WRAP(f64, double, TensorDType::F64)

NODUS_TENSOR_GATHER_FOOTPRINT_2D_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::gather_footprint_2d(
        base, footprint, out, orientation, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_FOOTPRINT_2D_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::gather_footprint_2d(
        base, footprint, out, orientation, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_FOOTPRINT_2D_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::gather_footprint_2d(
        base, footprint, out, orientation, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_FOOTPRINT_2D_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::gather_footprint_2d(
        base, footprint, out, orientation, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_FOOTPRINT_2D_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::gather_footprint_2d(
        base, footprint, out, orientation, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_FOOTPRINT_2D_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::gather_footprint_2d(
        base, footprint, out, orientation, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_FOOTPRINT_2D_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::gather_footprint_2d(
        base, footprint, out, orientation, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_FOOTPRINT_2D_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::gather_footprint_2d(
        base, footprint, out, orientation, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_FOOTPRINT_2D_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::gather_footprint_2d(
        base, footprint, out, orientation, boundary, target_mask, source_mask, normalize))
NODUS_TENSOR_GATHER_FOOTPRINT_2D_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::gather_footprint_2d(
        base, footprint, out, orientation, boundary, target_mask, source_mask, normalize))

NODUS_TENSOR_GATHER_ADD_FOOTPRINT_2D_WRAP(u8, uint8_t, TensorDType::U8)
NODUS_TENSOR_GATHER_ADD_FOOTPRINT_2D_WRAP(u16, uint16_t, TensorDType::U16)
NODUS_TENSOR_GATHER_ADD_FOOTPRINT_2D_WRAP(u32, uint32_t, TensorDType::U32)
NODUS_TENSOR_GATHER_ADD_FOOTPRINT_2D_WRAP(u64, uint64_t, TensorDType::U64)
NODUS_TENSOR_GATHER_ADD_FOOTPRINT_2D_WRAP(i8, int8_t, TensorDType::I8)
NODUS_TENSOR_GATHER_ADD_FOOTPRINT_2D_WRAP(i16, int16_t, TensorDType::I16)
NODUS_TENSOR_GATHER_ADD_FOOTPRINT_2D_WRAP(i32, int32_t, TensorDType::I32)
NODUS_TENSOR_GATHER_ADD_FOOTPRINT_2D_WRAP(i64, int64_t, TensorDType::I64)
NODUS_TENSOR_GATHER_ADD_FOOTPRINT_2D_WRAP(f32, float, TensorDType::F32)
NODUS_TENSOR_GATHER_ADD_FOOTPRINT_2D_WRAP(f64, double, TensorDType::F64)

NODUS_TENSOR_SCATTER_STENCIL_2D_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::scatter_stencil_2d(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_2D_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::scatter_stencil_2d(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_2D_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::scatter_stencil_2d(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_2D_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::scatter_stencil_2d(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_2D_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::scatter_stencil_2d(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_2D_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::scatter_stencil_2d(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_2D_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::scatter_stencil_2d(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_2D_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::scatter_stencil_2d(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_2D_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::scatter_stencil_2d(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_2D_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::scatter_stencil_2d(
        base, points, values, stencil, out, boundary, clamp))

NODUS_TENSOR_SCATTER_STENCIL_ND_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::scatter_stencil_nd(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_ND_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::scatter_stencil_nd(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_ND_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::scatter_stencil_nd(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_ND_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::scatter_stencil_nd(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_ND_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::scatter_stencil_nd(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_ND_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::scatter_stencil_nd(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_ND_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::scatter_stencil_nd(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_ND_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::scatter_stencil_nd(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_ND_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::scatter_stencil_nd(
        base, points, values, stencil, out, boundary, clamp))
NODUS_TENSOR_SCATTER_STENCIL_ND_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::scatter_stencil_nd(
        base, points, values, stencil, out, boundary, clamp))

NODUS_TENSOR_SCATTER_FOOTPRINT_2D_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::scatter_footprint_2d(
        base, points, values, footprint, out, orientation, boundary, clamp))
NODUS_TENSOR_SCATTER_FOOTPRINT_2D_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::scatter_footprint_2d(
        base, points, values, footprint, out, orientation, boundary, clamp))
NODUS_TENSOR_SCATTER_FOOTPRINT_2D_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::scatter_footprint_2d(
        base, points, values, footprint, out, orientation, boundary, clamp))
NODUS_TENSOR_SCATTER_FOOTPRINT_2D_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::scatter_footprint_2d(
        base, points, values, footprint, out, orientation, boundary, clamp))
NODUS_TENSOR_SCATTER_FOOTPRINT_2D_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::scatter_footprint_2d(
        base, points, values, footprint, out, orientation, boundary, clamp))
NODUS_TENSOR_SCATTER_FOOTPRINT_2D_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::scatter_footprint_2d(
        base, points, values, footprint, out, orientation, boundary, clamp))
NODUS_TENSOR_SCATTER_FOOTPRINT_2D_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::scatter_footprint_2d(
        base, points, values, footprint, out, orientation, boundary, clamp))
NODUS_TENSOR_SCATTER_FOOTPRINT_2D_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::scatter_footprint_2d(
        base, points, values, footprint, out, orientation, boundary, clamp))
NODUS_TENSOR_SCATTER_FOOTPRINT_2D_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::scatter_footprint_2d(
        base, points, values, footprint, out, orientation, boundary, clamp))
NODUS_TENSOR_SCATTER_FOOTPRINT_2D_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::scatter_footprint_2d(
        base, points, values, footprint, out, orientation, boundary, clamp))

#undef NODUS_TENSOR_GATHER_STENCIL_2D_WRAP
#undef NODUS_TENSOR_GATHER_ADD_STENCIL_2D_WRAP
#undef NODUS_TENSOR_GATHER_STENCIL_ND_WRAP
#undef NODUS_TENSOR_GATHER_ADD_STENCIL_ND_WRAP
#undef NODUS_TENSOR_GATHER_FOOTPRINT_2D_WRAP
#undef NODUS_TENSOR_GATHER_ADD_FOOTPRINT_2D_WRAP
#undef NODUS_TENSOR_SCATTER_STENCIL_2D_WRAP
#undef NODUS_TENSOR_SCATTER_STENCIL_ND_WRAP
#undef NODUS_TENSOR_SCATTER_FOOTPRINT_2D_WRAP

#define NODUS_TENSOR_CSR_SCATTER_WRAP(SUFFIX, ...)                                 \
    bool tensor_scatter_probe_csr_##SUFFIX(const AbstractTensor& base,             \
                                           const TensorProbeKernel& probe,         \
                                           const TensorCSR& targets_per_center,    \
                                           AbstractTensor* out,                    \
                                           bool clamp) {                           \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_CSR_GATHER_WRAP(SUFFIX, ...)                                  \
    bool tensor_gather_probe_csr_##SUFFIX(const AbstractTensor& base,              \
                                          const TensorProbeKernel& probe,          \
                                          const TensorCSR& centers_per_target,     \
                                          AbstractTensor* out,                     \
                                          bool clamp) {                            \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_CSR_STENCIL_SCATTER_WRAP(SUFFIX, ...)                         \
    bool tensor_scatter_probe_csr_stencil_##SUFFIX(const AbstractTensor& base,     \
                                                   const TensorStencil& stencil,  \
                                                   const TensorCSR& targets_per_center, \
                                                   AbstractTensor* out,            \
                                                   bool clamp) {                   \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_CSR_STENCIL_GATHER_WRAP(SUFFIX, ...)                          \
    bool tensor_gather_probe_csr_stencil_##SUFFIX(const AbstractTensor& base,      \
                                                  const TensorStencil& stencil,   \
                                                  const TensorCSR& centers_per_target, \
                                                  AbstractTensor* out,             \
                                                  bool clamp) {                    \
        return (__VA_ARGS__);                                                      \
    }

#define NODUS_TENSOR_CSR_FOOTPRINT_SCATTER_WRAP(SUFFIX, BODY)                      \
    bool tensor_scatter_probe_csr_footprint_##SUFFIX(const AbstractTensor& base,   \
                                                     const TensorFootprint2D& footprint, \
                                                     StencilOrientation orientation, \
                                                     const TensorCSR& targets_per_center, \
                                                     AbstractTensor* out,           \
                                                     bool clamp) {                  \
        return (BODY);                                                             \
    }

#define NODUS_TENSOR_CSR_FOOTPRINT_GATHER_WRAP(SUFFIX, BODY)                       \
    bool tensor_gather_probe_csr_footprint_##SUFFIX(const AbstractTensor& base,    \
                                                    const TensorFootprint2D& footprint, \
                                                    StencilOrientation orientation, \
                                                    const TensorCSR& centers_per_target, \
                                                    AbstractTensor* out,            \
                                                    bool clamp) {                   \
        return (BODY);                                                             \
    }

NODUS_TENSOR_CSR_SCATTER_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::scatter_probe_csr(base, probe, targets_per_center, out, clamp))
NODUS_TENSOR_CSR_SCATTER_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::scatter_probe_csr(base, probe, targets_per_center, out, clamp))
NODUS_TENSOR_CSR_SCATTER_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::scatter_probe_csr(base, probe, targets_per_center, out, clamp))
NODUS_TENSOR_CSR_SCATTER_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::scatter_probe_csr(base, probe, targets_per_center, out, clamp))
NODUS_TENSOR_CSR_SCATTER_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::scatter_probe_csr(base, probe, targets_per_center, out, clamp))
NODUS_TENSOR_CSR_SCATTER_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::scatter_probe_csr(base, probe, targets_per_center, out, clamp))
NODUS_TENSOR_CSR_SCATTER_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::scatter_probe_csr(base, probe, targets_per_center, out, clamp))
NODUS_TENSOR_CSR_SCATTER_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::scatter_probe_csr(base, probe, targets_per_center, out, clamp))
NODUS_TENSOR_CSR_SCATTER_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::scatter_probe_csr(base, probe, targets_per_center, out, clamp))
NODUS_TENSOR_CSR_SCATTER_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::scatter_probe_csr(base, probe, targets_per_center, out, clamp))

NODUS_TENSOR_CSR_GATHER_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::gather_probe_csr(base, probe, centers_per_target, out, clamp))
NODUS_TENSOR_CSR_GATHER_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::gather_probe_csr(base, probe, centers_per_target, out, clamp))
NODUS_TENSOR_CSR_GATHER_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::gather_probe_csr(base, probe, centers_per_target, out, clamp))
NODUS_TENSOR_CSR_GATHER_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::gather_probe_csr(base, probe, centers_per_target, out, clamp))
NODUS_TENSOR_CSR_GATHER_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::gather_probe_csr(base, probe, centers_per_target, out, clamp))
NODUS_TENSOR_CSR_GATHER_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::gather_probe_csr(base, probe, centers_per_target, out, clamp))
NODUS_TENSOR_CSR_GATHER_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::gather_probe_csr(base, probe, centers_per_target, out, clamp))
NODUS_TENSOR_CSR_GATHER_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::gather_probe_csr(base, probe, centers_per_target, out, clamp))
NODUS_TENSOR_CSR_GATHER_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::gather_probe_csr(base, probe, centers_per_target, out, clamp))
NODUS_TENSOR_CSR_GATHER_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::gather_probe_csr(base, probe, centers_per_target, out, clamp))

static TensorProbeKernel probe_from_stencil(const TensorStencil& stencil) {
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

NODUS_TENSOR_CSR_STENCIL_SCATTER_WRAP(i8,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_scatter_probe_csr_i8(base, probe, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_SCATTER_WRAP(i16,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_scatter_probe_csr_i16(base, probe, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_SCATTER_WRAP(i32,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_scatter_probe_csr_i32(base, probe, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_SCATTER_WRAP(i64,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_scatter_probe_csr_i64(base, probe, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_SCATTER_WRAP(u8,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_scatter_probe_csr_u8(base, probe, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_SCATTER_WRAP(u16,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_scatter_probe_csr_u16(base, probe, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_SCATTER_WRAP(u32,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_scatter_probe_csr_u32(base, probe, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_SCATTER_WRAP(u64,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_scatter_probe_csr_u64(base, probe, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_SCATTER_WRAP(f32,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_scatter_probe_csr_f32(base, probe, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_SCATTER_WRAP(f64,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_scatter_probe_csr_f64(base, probe, targets_per_center, out, clamp);
    })())

NODUS_TENSOR_CSR_STENCIL_GATHER_WRAP(i8,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_gather_probe_csr_i8(base, probe, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_GATHER_WRAP(i16,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_gather_probe_csr_i16(base, probe, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_GATHER_WRAP(i32,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_gather_probe_csr_i32(base, probe, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_GATHER_WRAP(i64,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_gather_probe_csr_i64(base, probe, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_GATHER_WRAP(u8,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_gather_probe_csr_u8(base, probe, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_GATHER_WRAP(u16,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_gather_probe_csr_u16(base, probe, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_GATHER_WRAP(u32,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_gather_probe_csr_u32(base, probe, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_GATHER_WRAP(u64,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_gather_probe_csr_u64(base, probe, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_GATHER_WRAP(f32,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_gather_probe_csr_f32(base, probe, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_STENCIL_GATHER_WRAP(f64,
    ([&]() {
        const TensorProbeKernel probe = probe_from_stencil(stencil);
        return tensor_gather_probe_csr_f64(base, probe, centers_per_target, out, clamp);
    })())

NODUS_TENSOR_CSR_FOOTPRINT_SCATTER_WRAP(i8,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<int8_t, TensorDType::I8>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_scatter_probe_csr_stencil_i8(base, stencil, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_SCATTER_WRAP(i16,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<int16_t, TensorDType::I16>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_scatter_probe_csr_stencil_i16(base, stencil, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_SCATTER_WRAP(i32,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<int32_t, TensorDType::I32>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_scatter_probe_csr_stencil_i32(base, stencil, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_SCATTER_WRAP(i64,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<int64_t, TensorDType::I64>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_scatter_probe_csr_stencil_i64(base, stencil, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_SCATTER_WRAP(u8,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<uint8_t, TensorDType::U8>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_scatter_probe_csr_stencil_u8(base, stencil, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_SCATTER_WRAP(u16,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<uint16_t, TensorDType::U16>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_scatter_probe_csr_stencil_u16(base, stencil, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_SCATTER_WRAP(u32,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<uint32_t, TensorDType::U32>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_scatter_probe_csr_stencil_u32(base, stencil, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_SCATTER_WRAP(u64,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<uint64_t, TensorDType::U64>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_scatter_probe_csr_stencil_u64(base, stencil, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_SCATTER_WRAP(f32,
    ([&]() {
        TensorStencil stencil{};
        if (!tensor_build_stencil_from_footprint_2d_f32(footprint, orientation, &stencil)) return false;
        return tensor_scatter_probe_csr_stencil_f32(base, stencil, targets_per_center, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_SCATTER_WRAP(f64,
    ([&]() {
        TensorStencil stencil{};
        if (!tensor_build_stencil_from_footprint_2d_f64(footprint, orientation, &stencil)) return false;
        return tensor_scatter_probe_csr_stencil_f64(base, stencil, targets_per_center, out, clamp);
    })())

NODUS_TENSOR_CSR_FOOTPRINT_GATHER_WRAP(i8,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<int8_t, TensorDType::I8>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_gather_probe_csr_stencil_i8(base, stencil, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_GATHER_WRAP(i16,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<int16_t, TensorDType::I16>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_gather_probe_csr_stencil_i16(base, stencil, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_GATHER_WRAP(i32,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<int32_t, TensorDType::I32>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_gather_probe_csr_stencil_i32(base, stencil, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_GATHER_WRAP(i64,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<int64_t, TensorDType::I64>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_gather_probe_csr_stencil_i64(base, stencil, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_GATHER_WRAP(u8,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<uint8_t, TensorDType::U8>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_gather_probe_csr_stencil_u8(base, stencil, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_GATHER_WRAP(u16,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<uint16_t, TensorDType::U16>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_gather_probe_csr_stencil_u16(base, stencil, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_GATHER_WRAP(u32,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<uint32_t, TensorDType::U32>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_gather_probe_csr_stencil_u32(base, stencil, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_GATHER_WRAP(u64,
    ([&]() {
        TensorStencil stencil{};
        if (!TensorMathImpl<uint64_t, TensorDType::U64>::build_stencil_from_footprint_2d(footprint, orientation, &stencil)) return false;
        return tensor_gather_probe_csr_stencil_u64(base, stencil, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_GATHER_WRAP(f32,
    ([&]() {
        TensorStencil stencil{};
        if (!tensor_build_stencil_from_footprint_2d_f32(footprint, orientation, &stencil)) return false;
        return tensor_gather_probe_csr_stencil_f32(base, stencil, centers_per_target, out, clamp);
    })())
NODUS_TENSOR_CSR_FOOTPRINT_GATHER_WRAP(f64,
    ([&]() {
        TensorStencil stencil{};
        if (!tensor_build_stencil_from_footprint_2d_f64(footprint, orientation, &stencil)) return false;
        return tensor_gather_probe_csr_stencil_f64(base, stencil, centers_per_target, out, clamp);
    })())

#undef NODUS_TENSOR_CSR_SCATTER_WRAP
#undef NODUS_TENSOR_CSR_GATHER_WRAP
#undef NODUS_TENSOR_CSR_STENCIL_SCATTER_WRAP
#undef NODUS_TENSOR_CSR_STENCIL_GATHER_WRAP
#undef NODUS_TENSOR_CSR_FOOTPRINT_SCATTER_WRAP
#undef NODUS_TENSOR_CSR_FOOTPRINT_GATHER_WRAP

bool tensor_scatter_add_2d_affine_f32(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& mat4,
                                      AbstractTensor* out,
                                      bool clamp) {
    return TensorMathImpl<float, TensorDType::F32>::scatter_add_2d_affine(
        base, points, values, mat4, out, clamp);
}

bool tensor_scatter_add_2d_affine_f64(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& mat4,
                                      AbstractTensor* out,
                                      bool clamp) {
    return TensorMathImpl<double, TensorDType::F64>::scatter_add_2d_affine(
        base, points, values, mat4, out, clamp);
}

#define NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_WRAP(SUFFIX, ...)                         \
    bool tensor_scatter_add_2d_kernel_##SUFFIX(const AbstractTensor& base,           \
                                               const AbstractTensor& points,        \
                                               const AbstractTensor& values,        \
                                               const AbstractTensor& kernel,        \
                                               AbstractTensor* out,                 \
                                               bool clamp) {                         \
        return (__VA_ARGS__);                                                       \
    }

#define NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_FN_WRAP(SUFFIX, KFN_TYPE, ...)            \
    bool tensor_scatter_add_2d_kernel_fn_##SUFFIX(const AbstractTensor& base,        \
                                                  const AbstractTensor& points,     \
                                                  const AbstractTensor& values,     \
                                                  const KFN_TYPE& kernel_fn,         \
                                                  void* user,                        \
                                                  AbstractTensor* out,               \
                                                  KernelDispatchPlan* out_plan,      \
                                                  bool clamp) {                      \
        return (__VA_ARGS__);                                                       \
    }

#define NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_PTR_WRAP(SUFFIX, ...)                     \
    bool tensor_scatter_add_2d_kernel_ptr_##SUFFIX(const AbstractTensor& base,       \
                                                   const AbstractTensor& points,    \
                                                   const AbstractTensor& values,    \
                                                   const AbstractTensor& kernel_ptrs, \
                                                   AbstractTensor* out,              \
                                                   KernelDispatchPlan* out_plan,     \
                                                   bool clamp) {                     \
        return (__VA_ARGS__);                                                       \
    }

#define NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_BANK_WRAP(SUFFIX, ...)                    \
    bool tensor_scatter_add_2d_kernel_bank_##SUFFIX(const AbstractTensor& base,      \
                                                    const AbstractTensor& points,   \
                                                    const AbstractTensor& values,   \
                                                    const AbstractTensor& kernel_bank, \
                                                    const AbstractTensor& kernel_ids, \
                                                    AbstractTensor* out,             \
                                                    bool clamp) {                    \
        return (__VA_ARGS__);                                                       \
    }

#define NODUS_TENSOR_APPLY_STENCIL_2D_WRAP(SUFFIX, ...)                              \
    AbstractTensor tensor_apply_stencil_2d_##SUFFIX(const AbstractTensor& field,    \
                                                    const AbstractTensor& kernel) { \
        return (__VA_ARGS__);                                                       \
    }

#define NODUS_TENSOR_APPLY_STENCIL_2D_INTO_WRAP(SUFFIX, ...)                         \
    bool tensor_apply_stencil_2d_##SUFFIX##_into(const AbstractTensor& field,       \
                                                 const AbstractTensor& kernel,      \
                                                 AbstractTensor* out) {             \
        return (__VA_ARGS__);                                                       \
    }

NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::scatter_add_2d_kernel(
        base, points, values, kernel, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::scatter_add_2d_kernel(
        base, points, values, kernel, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::scatter_add_2d_kernel(
        base, points, values, kernel, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::scatter_add_2d_kernel(
        base, points, values, kernel, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::scatter_add_2d_kernel(
        base, points, values, kernel, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::scatter_add_2d_kernel(
        base, points, values, kernel, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::scatter_add_2d_kernel(
        base, points, values, kernel, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::scatter_add_2d_kernel(
        base, points, values, kernel, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::scatter_add_2d_kernel(
        base, points, values, kernel, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::scatter_add_2d_kernel(
        base, points, values, kernel, out, clamp))

NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_FN_WRAP(u8, TensorKernelFnU8,
    TensorMathImpl<uint8_t, TensorDType::U8>::scatter_add_2d_kernel_fn_impl(
        base, points, values, kernel_fn, user, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_FN_WRAP(u16, TensorKernelFnU16,
    TensorMathImpl<uint16_t, TensorDType::U16>::scatter_add_2d_kernel_fn_impl(
        base, points, values, kernel_fn, user, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_FN_WRAP(u32, TensorKernelFnU32,
    TensorMathImpl<uint32_t, TensorDType::U32>::scatter_add_2d_kernel_fn_impl(
        base, points, values, kernel_fn, user, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_FN_WRAP(u64, TensorKernelFnU64,
    TensorMathImpl<uint64_t, TensorDType::U64>::scatter_add_2d_kernel_fn_impl(
        base, points, values, kernel_fn, user, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_FN_WRAP(i8, TensorKernelFnI8,
    TensorMathImpl<int8_t, TensorDType::I8>::scatter_add_2d_kernel_fn_impl(
        base, points, values, kernel_fn, user, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_FN_WRAP(i16, TensorKernelFnI16,
    TensorMathImpl<int16_t, TensorDType::I16>::scatter_add_2d_kernel_fn_impl(
        base, points, values, kernel_fn, user, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_FN_WRAP(i32, TensorKernelFnI32,
    TensorMathImpl<int32_t, TensorDType::I32>::scatter_add_2d_kernel_fn_impl(
        base, points, values, kernel_fn, user, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_FN_WRAP(i64, TensorKernelFnI64,
    TensorMathImpl<int64_t, TensorDType::I64>::scatter_add_2d_kernel_fn_impl(
        base, points, values, kernel_fn, user, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_FN_WRAP(f32, TensorKernelFnF32,
    TensorMathImpl<float, TensorDType::F32>::scatter_add_2d_kernel_fn_impl(
        base, points, values, kernel_fn, user, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_FN_WRAP(f64, TensorKernelFnF64,
    TensorMathImpl<double, TensorDType::F64>::scatter_add_2d_kernel_fn_impl(
        base, points, values, kernel_fn, user, out, out_plan, clamp))

NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_PTR_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::scatter_add_2d_kernel_ptr_impl(
        base, points, values, kernel_ptrs, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_PTR_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::scatter_add_2d_kernel_ptr_impl(
        base, points, values, kernel_ptrs, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_PTR_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::scatter_add_2d_kernel_ptr_impl(
        base, points, values, kernel_ptrs, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_PTR_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::scatter_add_2d_kernel_ptr_impl(
        base, points, values, kernel_ptrs, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_PTR_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::scatter_add_2d_kernel_ptr_impl(
        base, points, values, kernel_ptrs, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_PTR_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::scatter_add_2d_kernel_ptr_impl(
        base, points, values, kernel_ptrs, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_PTR_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::scatter_add_2d_kernel_ptr_impl(
        base, points, values, kernel_ptrs, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_PTR_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::scatter_add_2d_kernel_ptr_impl(
        base, points, values, kernel_ptrs, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_PTR_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::scatter_add_2d_kernel_ptr_impl(
        base, points, values, kernel_ptrs, out, out_plan, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_PTR_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::scatter_add_2d_kernel_ptr_impl(
        base, points, values, kernel_ptrs, out, out_plan, clamp))

NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_BANK_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::scatter_add_2d_kernel_bank(
        base, points, values, kernel_bank, kernel_ids, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_BANK_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::scatter_add_2d_kernel_bank(
        base, points, values, kernel_bank, kernel_ids, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_BANK_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::scatter_add_2d_kernel_bank(
        base, points, values, kernel_bank, kernel_ids, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_BANK_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::scatter_add_2d_kernel_bank(
        base, points, values, kernel_bank, kernel_ids, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_BANK_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::scatter_add_2d_kernel_bank(
        base, points, values, kernel_bank, kernel_ids, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_BANK_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::scatter_add_2d_kernel_bank(
        base, points, values, kernel_bank, kernel_ids, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_BANK_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::scatter_add_2d_kernel_bank(
        base, points, values, kernel_bank, kernel_ids, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_BANK_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::scatter_add_2d_kernel_bank(
        base, points, values, kernel_bank, kernel_ids, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_BANK_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::scatter_add_2d_kernel_bank(
        base, points, values, kernel_bank, kernel_ids, out, clamp))
NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_BANK_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::scatter_add_2d_kernel_bank(
        base, points, values, kernel_bank, kernel_ids, out, clamp))

NODUS_TENSOR_APPLY_STENCIL_2D_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::apply_stencil_2d(field, kernel))
NODUS_TENSOR_APPLY_STENCIL_2D_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::apply_stencil_2d(field, kernel))
NODUS_TENSOR_APPLY_STENCIL_2D_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::apply_stencil_2d(field, kernel))
NODUS_TENSOR_APPLY_STENCIL_2D_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::apply_stencil_2d(field, kernel))
NODUS_TENSOR_APPLY_STENCIL_2D_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::apply_stencil_2d(field, kernel))
NODUS_TENSOR_APPLY_STENCIL_2D_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::apply_stencil_2d(field, kernel))
NODUS_TENSOR_APPLY_STENCIL_2D_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::apply_stencil_2d(field, kernel))
NODUS_TENSOR_APPLY_STENCIL_2D_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::apply_stencil_2d(field, kernel))
NODUS_TENSOR_APPLY_STENCIL_2D_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::apply_stencil_2d(field, kernel))
NODUS_TENSOR_APPLY_STENCIL_2D_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::apply_stencil_2d(field, kernel))

NODUS_TENSOR_APPLY_STENCIL_2D_INTO_WRAP(u8,
    TensorMathImpl<uint8_t, TensorDType::U8>::apply_stencil_2d_into(field, kernel, out))
NODUS_TENSOR_APPLY_STENCIL_2D_INTO_WRAP(u16,
    TensorMathImpl<uint16_t, TensorDType::U16>::apply_stencil_2d_into(field, kernel, out))
NODUS_TENSOR_APPLY_STENCIL_2D_INTO_WRAP(u32,
    TensorMathImpl<uint32_t, TensorDType::U32>::apply_stencil_2d_into(field, kernel, out))
NODUS_TENSOR_APPLY_STENCIL_2D_INTO_WRAP(u64,
    TensorMathImpl<uint64_t, TensorDType::U64>::apply_stencil_2d_into(field, kernel, out))
NODUS_TENSOR_APPLY_STENCIL_2D_INTO_WRAP(i8,
    TensorMathImpl<int8_t, TensorDType::I8>::apply_stencil_2d_into(field, kernel, out))
NODUS_TENSOR_APPLY_STENCIL_2D_INTO_WRAP(i16,
    TensorMathImpl<int16_t, TensorDType::I16>::apply_stencil_2d_into(field, kernel, out))
NODUS_TENSOR_APPLY_STENCIL_2D_INTO_WRAP(i32,
    TensorMathImpl<int32_t, TensorDType::I32>::apply_stencil_2d_into(field, kernel, out))
NODUS_TENSOR_APPLY_STENCIL_2D_INTO_WRAP(i64,
    TensorMathImpl<int64_t, TensorDType::I64>::apply_stencil_2d_into(field, kernel, out))
NODUS_TENSOR_APPLY_STENCIL_2D_INTO_WRAP(f32,
    TensorMathImpl<float, TensorDType::F32>::apply_stencil_2d_into(field, kernel, out))
NODUS_TENSOR_APPLY_STENCIL_2D_INTO_WRAP(f64,
    TensorMathImpl<double, TensorDType::F64>::apply_stencil_2d_into(field, kernel, out))

#undef NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_WRAP
#undef NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_FN_WRAP
#undef NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_PTR_WRAP
#undef NODUS_TENSOR_SCATTER_ADD_2D_KERNEL_BANK_WRAP
#undef NODUS_TENSOR_APPLY_STENCIL_2D_WRAP
#undef NODUS_TENSOR_APPLY_STENCIL_2D_INTO_WRAP

bool tensor_reduce_sum_axis_f32(const AbstractTensor& src, uint32_t axis, AbstractTensor* out) {
    if (!out || !out->valid() || !src.valid()) return false;
    if (src.backend() != out->backend()) return false;
    const TensorDesc& sd = src.desc();
    const TensorDesc& od = out->desc();
    if (sd.dtype != TensorDType::F32 || od.dtype != TensorDType::F32) return false;
    const size_t rank = sd.shape.dims.size();
    if (rank == 0 || axis >= rank) return false;

    std::vector<uint32_t> out_shape = sd.shape.dims;
    out_shape.erase(out_shape.begin() + axis);
    if (out_shape != od.shape.dims) return false;

    auto* mem = dynamic_cast<InMemoryBackend*>(src.backend());
    if (!mem) return false;

    void* src_ptr_v = nullptr;
    size_t src_bytes = 0;
    if (!mem->map(src.handle(), &src_ptr_v, &src_bytes)) return false;
    void* dst_ptr_v = nullptr;
    size_t dst_bytes = 0;
    if (!mem->map(out->handle(), &dst_ptr_v, &dst_bytes)) {
        mem->unmap(src.handle());
        return false;
    }

    const float* src_ptr = static_cast<const float*>(src_ptr_v);
    float* dst_ptr = static_cast<float*>(dst_ptr_v);

    std::vector<uint64_t> src_strides;
    if (!compute_strides_for_desc_u64(sd, src_strides)) {
        mem->unmap(src.handle());
        mem->unmap(out->handle());
        return false;
    }

    std::vector<uint64_t> dst_strides;
    if (!compute_strides_for_desc_u64(od, dst_strides)) {
        mem->unmap(src.handle());
        mem->unmap(out->handle());
        return false;
    }

    std::vector<uint64_t> dense_out_strides;
    if (!out_shape.empty() && !compute_dense_strides_u64(out_shape, dense_out_strides)) {
        mem->unmap(src.handle());
        mem->unmap(out->handle());
        return false;
    }

    const bool dst_dense = (od.layout == TensorLayout::Dense);
    const uint64_t output_elems = od.shape.element_count();
    std::vector<uint32_t> src_coords(rank, 0);
    std::vector<uint32_t> out_coords(out_shape.size(), 0);

    for (uint64_t lin = 0; lin < output_elems; ++lin) {
        if (!out_shape.empty()) {
            uint64_t rem = lin;
            for (size_t d = 0; d < out_shape.size(); ++d) {
                const uint64_t stride = dense_out_strides[d];
                const uint32_t dim = out_shape[d];
                if (dim == 0) {
                    out_coords[d] = 0;
                    continue;
                }
                if (stride > 0) {
                    out_coords[d] = static_cast<uint32_t>(rem / stride);
                    rem = rem % stride;
                } else {
                    out_coords[d] = 0;
                }
            }
        }

        size_t out_idx = 0;
        for (size_t d = 0; d < rank; ++d) {
            if (d == axis) continue;
            src_coords[d] = out_shape.empty() ? 0u : out_coords[out_idx++];
        }

        uint64_t base_offset = 0;
        for (size_t d = 0; d < rank; ++d) {
            if (d == axis) continue;
            base_offset += static_cast<uint64_t>(src_coords[d]) * src_strides[d];
        }

        float acc = 0.0f;
        const uint32_t axis_dim = sd.shape.dims[axis];
        const uint64_t axis_stride = src_strides[axis];
        for (uint32_t a = 0; a < axis_dim; ++a) {
            const uint64_t src_offset = base_offset + static_cast<uint64_t>(a) * axis_stride;
            acc += src_ptr[src_offset];
        }

        if (dst_dense) {
            dst_ptr[lin] = acc;
        } else {
            uint64_t dst_offset = 0;
            for (size_t d = 0; d < out_shape.size(); ++d) {
                dst_offset += static_cast<uint64_t>(out_coords[d]) * dst_strides[d];
            }
            dst_ptr[dst_offset] = acc;
        }
    }

    mem->unmap(src.handle());
    mem->unmap(out->handle());
    return true;
}

bool tensor_pack_rgba_from_film_f32(const AbstractTensor& film,
                                     AbstractTensor* rgba,
                                     uint32_t check_shift) {
    if (!film.valid() || !rgba || !rgba->valid()) return false;
    if (film.backend() != rgba->backend()) return false;
    const TensorDesc& fdesc = film.desc();
    const TensorDesc& rdesc = rgba->desc();
    if (fdesc.dtype != TensorDType::F32 || fdesc.layout != TensorLayout::Dense) return false;
    if (rdesc.dtype != TensorDType::U32 || rdesc.layout != TensorLayout::Dense) return false;
    if (fdesc.shape.dims.size() != 3) return false;
    if (rdesc.shape.dims.size() != 2) return false;

    const uint32_t height = fdesc.shape.dims[0];
    const uint32_t width = fdesc.shape.dims[1];
    const uint32_t channels = fdesc.shape.dims[2];
    if (channels < 3) return false;
    if (rdesc.shape.dims[0] != height || rdesc.shape.dims[1] != width) return false;

    auto* mem = dynamic_cast<InMemoryBackend*>(film.backend());
    if (!mem) return false;

    void* film_ptr = nullptr;
    size_t film_bytes = 0;
    if (!mem->map(film.handle(), &film_ptr, &film_bytes)) return false;
    void* rgba_ptr = nullptr;
    size_t rgba_bytes = 0;
    if (!mem->map(rgba->handle(), &rgba_ptr, &rgba_bytes)) {
        mem->unmap(film.handle());
        return false;
    }

    const float* src = static_cast<const float*>(film_ptr);
    uint32_t* dst = static_cast<uint32_t*>(rgba_ptr);
    const uint32_t check_mask = (1u << check_shift) - 1u;

    for (uint32_t y = 0; y < height; ++y) {
        const uint32_t ycheck = (y >> check_shift);
        const bool yodd = (ycheck & 1u) != 0u;
        for (uint32_t x = 0; x < width; ++x) {
            const size_t base = (static_cast<size_t>(y) * width + x) * channels;
            const bool odd = ((x >> check_shift) ^ ycheck) & 1u;
            const float bias = odd ? 0.18f : 0.28f;

            auto to_clamped = [&](uint32_t channel) -> float {
                float v = bias + src[base + static_cast<size_t>(channel)];
                v = std::clamp(v, 0.0f, 1.0f);
                return v;
            };

            const uint8_t r = static_cast<uint8_t>(std::lround(to_clamped(0) * 255.0f));
            const uint8_t g = static_cast<uint8_t>(std::lround(to_clamped(1) * 255.0f));
            const uint8_t b = static_cast<uint8_t>(std::lround(to_clamped(2) * 255.0f));
            dst[static_cast<size_t>(y) * width + x] =
                static_cast<uint32_t>(r) |
                (static_cast<uint32_t>(g) << 8) |
                (static_cast<uint32_t>(b) << 16) |
                (static_cast<uint32_t>(255) << 24);
        }
    }

    mem->unmap(rgba->handle());
    mem->unmap(film.handle());
    return true;
}

AbstractTensor tensor_quat_normalize(const AbstractTensor& q) {
    if (!q.valid()) return {};
    if (q.desc().dtype == TensorDType::F32) return tensor_quat_normalize_f32(q);
    if (q.desc().dtype == TensorDType::F64) return tensor_quat_normalize_f64(q);
    return {};
}

AbstractTensor tensor_quat_mul(const AbstractTensor& a, const AbstractTensor& b) {
    if (!a.valid() || !b.valid()) return {};
    if (a.desc().dtype == TensorDType::F32 && b.desc().dtype == TensorDType::F32) {
        return tensor_quat_mul_f32(a, b);
    }
    if (a.desc().dtype == TensorDType::F64 && b.desc().dtype == TensorDType::F64) {
        return tensor_quat_mul_f64(a, b);
    }
    return {};
}

AbstractTensor tensor_quat_from_axis_angle(const AbstractTensor& axis, double angle) {
    if (!axis.valid()) return {};
    if (axis.desc().dtype == TensorDType::F32) {
        return tensor_quat_from_axis_angle_f32(axis, static_cast<float>(angle));
    }
    if (axis.desc().dtype == TensorDType::F64) {
        return tensor_quat_from_axis_angle_f64(axis, angle);
    }
    return {};
}

} // namespace nodus::tensors
