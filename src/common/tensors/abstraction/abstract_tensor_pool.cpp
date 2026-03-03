#include "common/tensors/abstraction/abstract_tensor_pool.h"

#include <algorithm>
#include <cstring>
#include <cstdio>

#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_registry.h"

namespace nodus::tensors {

// Uncomment to enable pool logging while debugging.
#define NODUS_POOL_LOGGING 1
#if defined(NODUS_POOL_LOGGING)
#define NODUS_POOL_LOGF(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define NODUS_POOL_LOGF(...) ((void)0)
#endif

AbstractTensorPool::PooledTensor::PooledTensor(AbstractTensorPool* pool, AbstractTensor tensor, TensorDesc pool_desc)
    : pool_(pool), tensor_(std::move(tensor)), pool_desc_(std::move(pool_desc)) {}

AbstractTensorPool::PooledTensor::PooledTensor(PooledTensor&& other) noexcept {
    pool_ = other.pool_;
    tensor_ = std::move(other.tensor_);
    pool_desc_ = std::move(other.pool_desc_);
    other.pool_ = nullptr;
}

AbstractTensorPool::PooledTensor& AbstractTensorPool::PooledTensor::operator=(PooledTensor&& other) noexcept {
    if (this == &other) return *this;
    if (pool_) {
        // Return via handle path so we use the correct pool key (bucketed desc).
        if (tensor_.valid()) {
            AbstractTensorHandle handle = tensor_.handle();
            TensorBackend* backend = tensor_.backend();
            TensorDesc pool_desc = pool_desc_;
            tensor_.release();
            tensor_.reset();
            pool_->release_handle(handle, backend, pool_desc);
        }
    }
    pool_ = other.pool_;
    tensor_ = std::move(other.tensor_);
    pool_desc_ = std::move(other.pool_desc_);
    other.pool_ = nullptr;
    return *this;
}

AbstractTensorPool::PooledTensor::~PooledTensor() {
    if (pool_) {
        if (tensor_.valid()) {
            NODUS_POOL_LOGF("[tensor_pool] pooled dtor release handle\n");
            AbstractTensorHandle handle = tensor_.handle();
            TensorBackend* backend = tensor_.backend();
            TensorDesc pool_desc = pool_desc_;
            tensor_.release();
            tensor_.reset();
            pool_->release_handle(handle, backend, pool_desc);
        }
    }
}

AbstractTensorPool::AbstractTensorPool(Options options)
    : options_(options) {}

AbstractTensorPool::~AbstractTensorPool() {
    clear();
}

AbstractTensorPool::PooledTensor AbstractTensorPool::acquire(const TensorDesc& desc, TensorBackend* backend) {
    TensorBackend* use_backend = backend ? backend : default_backend();
    if (!use_backend) return PooledTensor{};

    stats_.acquire_calls++;
    const TensorDesc pool_desc = bucket_desc(desc);
    if (pool_desc.shape.dims != desc.shape.dims ||
        pool_desc.layout != desc.layout ||
        pool_desc.dtype != desc.dtype) {
        stats_.bucketed_acquires++;
    }

    AbstractTensorHandle handle{};
    if (options_.cache_handles) {
        const PoolKey key{use_backend, make_desc_key(pool_desc)};
        auto it = pool_.find(key);
        if (it != pool_.end() && !it->second.empty()) {
            handle = it->second.back();
            it->second.pop_back();
            if (it->second.empty()) {
                pool_.erase(it);
            }
            stats_.cache_hits++;
            NODUS_POOL_LOGF("[tensor_pool] acquire hit dtype=%d layout=%d rank=%zu\n",
                            static_cast<int>(pool_desc.dtype),
                            static_cast<int>(pool_desc.layout),
                            pool_desc.shape.dims.size());
        } else {
            stats_.cache_misses++;
            NODUS_POOL_LOGF("[tensor_pool] acquire miss dtype=%d layout=%d rank=%zu\n",
                            static_cast<int>(pool_desc.dtype),
                            static_cast<int>(pool_desc.layout),
                            pool_desc.shape.dims.size());
        }
    }

    if (!abstract_tensor_handle_is_valid(handle)) {
        NODUS_POOL_LOGF("[tensor_pool] create_raw begin dtype=%d layout=%d rank=%zu\n",
                        static_cast<int>(pool_desc.dtype),
                        static_cast<int>(pool_desc.layout),
                        pool_desc.shape.dims.size());
        AbstractTensor cap = AbstractTensor::create_raw(pool_desc, use_backend);
        if (!cap.valid()) {
            std::fprintf(stderr, "[tensor_pool] create_raw failed: dtype=%d layout=%d rank=%zu dims=[",
                         static_cast<int>(pool_desc.dtype),
                         static_cast<int>(pool_desc.layout),
                         pool_desc.shape.dims.size());
            for (size_t i = 0; i < pool_desc.shape.dims.size(); ++i) {
                std::fprintf(stderr, "%u", pool_desc.shape.dims[i]);
                if (i + 1u < pool_desc.shape.dims.size()) {
                    std::fprintf(stderr, ",");
                }
            }
            std::fprintf(stderr, "]\n");
            return PooledTensor{};
        }
        stats_.created_handles++;
        handle = cap.handle();
        cap.release();
        cap.reset();
        NODUS_POOL_LOGF("[tensor_pool] create_raw ok\n");
    }

    AbstractTensor tensor = AbstractTensor::wrap(handle, desc, use_backend, true);
    NODUS_POOL_LOGF("[tensor_pool] pooled acquire wrap ok\n");
    return PooledTensor(this, std::move(tensor), pool_desc);
}

AbstractTensor AbstractTensorPool::acquire_tensor(const TensorDesc& desc, TensorBackend* backend) {
    PooledTensor pooled = acquire(desc, backend);
    if (!pooled.valid()) return {};

    AbstractTensor out = std::move(pooled.tensor_);
    out.pool_ = this;
    out.pool_desc_ = pooled.pool_desc_;

    pooled.pool_ = nullptr;
    pooled.pool_desc_ = {};
    pooled.tensor_ = {};
    return out;
}

void AbstractTensorPool::release(AbstractTensor&& tensor) {
    if (!tensor.valid()) return;
    AbstractTensorHandle handle = tensor.handle();
    TensorBackend* backend = tensor.backend();
    if (!backend) return;
    TensorDesc pool_desc = tensor.desc();
    tensor.release();
    tensor.reset();
    release_handle(handle, backend, pool_desc);
}

void AbstractTensorPool::release_handle(AbstractTensorHandle handle, TensorBackend* backend, const TensorDesc& pool_desc) {
    if (!backend || !abstract_tensor_handle_is_valid(handle)) return;
    if (!options_.cache_handles) {
        NODUS_POOL_LOGF("[tensor_pool] release_handle destroy (no cache)\n");
        backend->destroy(handle);
        return;
    }

    if (options_.clear_on_release) {
        clear_tensor(backend, handle, pool_desc);
    }

    const PoolKey key{backend, make_desc_key(pool_desc)};
    auto& list = pool_[key];
    list.push_back(handle);

    if (options_.max_cached_handles_per_key > 0) {
        while (list.size() > options_.max_cached_handles_per_key) {
            const AbstractTensorHandle to_destroy = list.back();
            list.pop_back();
            backend->destroy(to_destroy);
        }
        if (list.empty()) {
            pool_.erase(key);
        }
    }
    enforce_cache_limits();
}

void AbstractTensorPool::preallocate(const TensorDesc& desc, TensorBackend* backend, size_t count) {
    TensorBackend* use_backend = backend ? backend : default_backend();
    if (!use_backend || count == 0) return;
    if (!options_.cache_handles) return;
    const TensorDesc pool_desc = bucket_desc(desc);
    const PoolKey key{use_backend, make_desc_key(pool_desc)};

    NODUS_POOL_LOGF("[tensor_pool] preallocate begin count=%zu dtype=%d layout=%d rank=%zu\n",
                    count,
                    static_cast<int>(pool_desc.dtype),
                    static_cast<int>(pool_desc.layout),
                    pool_desc.shape.dims.size());

    auto& list = pool_[key];
    while (count-- > 0) {
        AbstractTensor cap = AbstractTensor::create_raw(pool_desc, use_backend);
        if (!cap.valid()) {
            NODUS_POOL_LOGF("[tensor_pool] preallocate create_raw failed\n");
            break;
        }
        stats_.created_handles++;
        AbstractTensorHandle handle = cap.handle();
        cap.release();
        cap.reset();
        if (options_.clear_on_release) {
            clear_tensor(use_backend, handle, pool_desc);
        }
        list.push_back(handle);
        if (options_.max_cached_handles_per_key > 0 && list.size() > options_.max_cached_handles_per_key) {
            const AbstractTensorHandle to_destroy = list.back();
            list.pop_back();
            use_backend->destroy(to_destroy);
        }
    }
    enforce_cache_limits();
    NODUS_POOL_LOGF("[tensor_pool] preallocate end\n");
}

void AbstractTensorPool::clear() {
    for (auto& kv : pool_) {
        TensorBackend* backend = kv.first.backend;
        if (!backend) continue;
        for (auto handle : kv.second) {
            backend->destroy(handle);
        }
    }
    pool_.clear();
}

size_t AbstractTensorPool::total_cached_handles() const {
    size_t total = 0;
    for (const auto& kv : pool_) {
        total += kv.second.size();
    }
    return total;
}

void AbstractTensorPool::enforce_cache_limits() {
    if (options_.max_cached_handles_total == 0) return;

    while (total_cached_handles() > options_.max_cached_handles_total) {
        bool removed_any = false;
        for (auto it = pool_.begin(); it != pool_.end();) {
            TensorBackend* backend = it->first.backend;
            auto& handles = it->second;

            if (handles.empty()) {
                it = pool_.erase(it);
                continue;
            }
            if (!backend) {
                handles.clear();
                it = pool_.erase(it);
                continue;
            }

            AbstractTensorHandle to_destroy = handles.back();
            handles.pop_back();
            backend->destroy(to_destroy);

            if (handles.empty()) {
                it = pool_.erase(it);
            } else {
                ++it;
            }
            removed_any = true;
            break;
        }
        if (!removed_any) break;
    }
}

AbstractTensorPool::DescKey AbstractTensorPool::make_desc_key(const TensorDesc& desc) {
    DescKey key;
    key.dtype = desc.dtype;
    key.layout = desc.layout;
    key.dims = desc.shape.dims;
    key.strides = desc.strides.elems;
    return key;
}

uint32_t AbstractTensorPool::round_up_pow2(uint32_t v) noexcept {
    if (v <= 1) return 1;
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

uint32_t AbstractTensorPool::round_up_multiple(uint32_t v, uint32_t multiple) noexcept {
    if (multiple == 0) return v;
    if (v == 0) return 0;
    const uint32_t r = v % multiple;
    return r == 0 ? v : (v + (multiple - r));
}

TensorDesc AbstractTensorPool::bucket_desc(const TensorDesc& desc) const {
    if (!options_.enable_shape_bucketing) return desc;
    if (desc.layout != TensorLayout::Dense) return desc;
    if (!desc.strides.elems.empty()) return desc;
    if (desc.shape.dims.empty()) return desc;

    TensorDesc out = desc;
    for (auto& d : out.shape.dims) {
        if (d == 0) continue;
        if (options_.bucket_pow2_max > 0 && d <= options_.bucket_pow2_max) {
            d = round_up_pow2(d);
        } else if (options_.bucket_multiple > 0) {
            d = round_up_multiple(d, options_.bucket_multiple);
        }
    }
    return out;
}

size_t AbstractTensorPool::hash_bytes(const void* data, size_t len) noexcept {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    size_t h = 14695981039346656037ull;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<size_t>(bytes[i]);
        h *= 1099511628211ull;
    }
    return h;
}

size_t AbstractTensorPool::combine_hash(size_t a, size_t b) noexcept {
    return a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6) + (a >> 2));
}

size_t AbstractTensorPool::PoolKeyHash::operator()(const PoolKey& key) const noexcept {
    size_t h = 0;
    h = AbstractTensorPool::combine_hash(h, std::hash<void*>{}(key.backend));
    h = AbstractTensorPool::combine_hash(h, static_cast<size_t>(key.desc.dtype));
    h = AbstractTensorPool::combine_hash(h, static_cast<size_t>(key.desc.layout));
    if (!key.desc.dims.empty()) {
        h = AbstractTensorPool::combine_hash(h,
                                             AbstractTensorPool::hash_bytes(key.desc.dims.data(),
                                                                            key.desc.dims.size() * sizeof(uint32_t)));
    }
    if (!key.desc.strides.empty()) {
        h = AbstractTensorPool::combine_hash(h,
                                             AbstractTensorPool::hash_bytes(key.desc.strides.data(),
                                                                            key.desc.strides.size() * sizeof(uint64_t)));
    }
    return h;
}

bool AbstractTensorPool::PoolKeyEq::operator()(const PoolKey& a, const PoolKey& b) const noexcept {
    return a.backend == b.backend &&
           a.desc.dtype == b.desc.dtype &&
           a.desc.layout == b.desc.layout &&
           a.desc.dims == b.desc.dims &&
           a.desc.strides == b.desc.strides;
}

void AbstractTensorPool::clear_tensor(TensorBackend* backend, AbstractTensorHandle handle, const TensorDesc& desc) const {
    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem) return;
    void* data = nullptr;
    size_t bytes = 0;
    if (!mem->map(handle, &data, &bytes)) return;
    if (bytes == 0) {
        mem->unmap(handle);
        return;
    }
    const size_t expected = static_cast<size_t>(desc.shape.element_count()) * tensor_dtype_size_bytes(desc.dtype);
    std::memset(data, 0, std::min(bytes, expected));
    mem->unmap(handle);
}

} // namespace nodus::tensors
