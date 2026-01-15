#include "common/tensors/abstraction/abstract_tensor_pool.h"

#include <algorithm>
#include <cstring>

#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_registry.h"

namespace nodus::tensors {

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
    const TensorDesc pool_desc = desc;

    AbstractTensor cap = AbstractTensor::create_raw(pool_desc, use_backend);
    if (!cap.valid()) return PooledTensor{};
    stats_.created_handles++;
    AbstractTensorHandle handle = cap.handle();
    cap.release();
    cap.reset();
    AbstractTensor tensor = AbstractTensor::wrap(handle, desc, use_backend, true);
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
    (void)pool_desc;
    backend->destroy(handle);
}

void AbstractTensorPool::preallocate(const TensorDesc& desc, TensorBackend* backend, size_t count) {
    TensorBackend* use_backend = backend ? backend : default_backend();
    if (!use_backend || count == 0) return;
    (void)desc;
    (void)count;
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
