#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "common/tensors/abstraction/abstract_tensor.h"

namespace nodus::tensors {

class AbstractTensorPool {
public:
    struct Options {
        bool clear_on_release = true;
        // If false, the pool will NOT cache handles; it will destroy the handle on release.
        // This makes the backend allocator the single source of truth for reuse and enables
        // cross-shape reuse (any large-enough freed region can satisfy any future request).
        bool cache_handles = true;
        // If enabled, the pool will round requested shapes up to bucketed "capacity" shapes
        // for caching purposes. The returned tensor will still report the requested logical
        // shape, but its backing allocation will be at least the bucketed size.
        //
        // This is critical for workloads where the first dimension varies frame-to-frame
        // (e.g. scatter site counts), to avoid desc-key explosion and allocator churn.
        bool enable_shape_bucketing = false;
        // Bucketing policy: round each dimension up to a power-of-two for small sizes.
        // 0 disables power-of-two rounding.
        uint32_t bucket_pow2_max = 4096;
        // For sizes above bucket_pow2_max (or when pow2 is disabled), round up to a multiple.
        // 0 disables multiple rounding.
        uint32_t bucket_multiple = 1024;
        // 0 means unlimited.
        // These caps are useful to prevent unbounded memory growth when tensor shapes vary frame-to-frame.
        size_t max_cached_handles_total = 0;
        size_t max_cached_handles_per_key = 0;
    };

    struct DebugStats {
        uint64_t acquire_calls = 0;
        uint64_t cache_hits = 0;
        uint64_t cache_misses = 0;
        uint64_t created_handles = 0;
        uint64_t bucketed_acquires = 0;
    };

    class PooledTensor {
    public:
        PooledTensor() = default;
        PooledTensor(AbstractTensorPool* pool, AbstractTensor tensor, TensorDesc pool_desc);
        PooledTensor(const PooledTensor&) = delete;
        PooledTensor& operator=(const PooledTensor&) = delete;
        PooledTensor(PooledTensor&& other) noexcept;
        PooledTensor& operator=(PooledTensor&& other) noexcept;
        ~PooledTensor();

        AbstractTensor& tensor() { return tensor_; }
        const AbstractTensor& tensor() const { return tensor_; }
        AbstractTensor* operator->() { return &tensor_; }
        const AbstractTensor* operator->() const { return &tensor_; }
        AbstractTensor& operator*() { return tensor_; }
        const AbstractTensor& operator*() const { return tensor_; }
        bool valid() const { return tensor_.valid(); }

    private:
        friend class AbstractTensorPool;
        AbstractTensorPool* pool_ = nullptr;
        AbstractTensor tensor_{};
        TensorDesc pool_desc_{};
    };

    explicit AbstractTensorPool(Options options = {});
    AbstractTensorPool(const AbstractTensorPool&) = delete;
    AbstractTensorPool& operator=(const AbstractTensorPool&) = delete;
    ~AbstractTensorPool();

    PooledTensor acquire(const TensorDesc& desc, TensorBackend* backend = nullptr);
    AbstractTensor acquire_tensor(const TensorDesc& desc, TensorBackend* backend = nullptr);
    void release(AbstractTensor&& tensor);
    void preallocate(const TensorDesc& desc, TensorBackend* backend, size_t count);
    void clear();

    DebugStats stats() const { return stats_; }
    void reset_stats() { stats_ = {}; }

private:
    friend class AbstractTensor;
    struct DescKey {
        TensorDType dtype = TensorDType::Unknown;
        TensorLayout layout = TensorLayout::Dense;
        std::vector<uint32_t> dims;
        std::vector<uint64_t> strides;
    };

    struct PoolKey {
        TensorBackend* backend = nullptr;
        DescKey desc;
    };

    struct PoolKeyHash {
        size_t operator()(const PoolKey& key) const noexcept;
    };

    struct PoolKeyEq {
        bool operator()(const PoolKey& a, const PoolKey& b) const noexcept;
    };

    static DescKey make_desc_key(const TensorDesc& desc);
    static uint32_t round_up_pow2(uint32_t v) noexcept;
    static uint32_t round_up_multiple(uint32_t v, uint32_t multiple) noexcept;
    TensorDesc bucket_desc(const TensorDesc& desc) const;
    static size_t hash_bytes(const void* data, size_t len) noexcept;
    static size_t combine_hash(size_t a, size_t b) noexcept;
    void clear_tensor(TensorBackend* backend, AbstractTensorHandle handle, const TensorDesc& desc) const;
    void release_handle(AbstractTensorHandle handle, TensorBackend* backend, const TensorDesc& pool_desc);
    size_t total_cached_handles() const;
    void enforce_cache_limits();

    Options options_{};
    std::unordered_map<PoolKey, std::vector<AbstractTensorHandle>, PoolKeyHash, PoolKeyEq> pool_{};
    DebugStats stats_{};
};

} // namespace nodus::tensors
