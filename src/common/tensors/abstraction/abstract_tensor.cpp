#include "common/tensors/abstraction/abstract_tensor.h"

#include "common/tensors/abstraction/abstract_tensor_pool.h"
#include "common/tensors/abstraction/tensor_backend.h"
#include "common/tensors/abstraction/tensor_index.h"
#include "common/tensors/abstraction/tensor_registry.h"

#include <utility>
#include <cstring>

namespace nodus::tensors {

AbstractTensor::AbstractTensor(AbstractTensor&& other) noexcept {
    handle_ = other.handle_;
    desc_ = std::move(other.desc_);
    backend_ = other.backend_;
    owns_handle_ = other.owns_handle_;
    pool_ = other.pool_;
    pool_desc_ = std::move(other.pool_desc_);
    other.handle_ = {};
    other.backend_ = nullptr;
    other.owns_handle_ = false;
    other.pool_ = nullptr;
    other.pool_desc_ = {};
}

AbstractTensor::AbstractTensor(const TensorDesc& desc, TensorBackend* backend) {
    *this = AbstractTensor::create(desc, backend);
}

AbstractTensor& AbstractTensor::operator=(AbstractTensor&& other) noexcept {
    if (this == &other) return *this;
    reset();
    handle_ = other.handle_;
    desc_ = std::move(other.desc_);
    backend_ = other.backend_;
    owns_handle_ = other.owns_handle_;
    pool_ = other.pool_;
    pool_desc_ = std::move(other.pool_desc_);
    other.handle_ = {};
    other.backend_ = nullptr;
    other.owns_handle_ = false;
    other.pool_ = nullptr;
    other.pool_desc_ = {};
    return *this;
}

AbstractTensor::~AbstractTensor() {
    reset();
}

namespace {
AbstractTensorPool& default_tensor_pool() {
    static AbstractTensorPool pool([] {
        AbstractTensorPool::Options opt;
        opt.cache_handles = false;
        return opt;
    }());
    return pool;
}
}

AbstractTensor AbstractTensor::create(const TensorDesc& desc, TensorBackend* backend) {
    return default_tensor_pool().acquire_tensor(desc, backend);
}

AbstractTensor AbstractTensor::create_raw(const TensorDesc& desc, TensorBackend* backend) {
    TensorBackend* use_backend = backend ? backend : default_backend();
    AbstractTensor out;
    if (!use_backend) return out;
    AbstractTensorHandle h = use_backend->create(desc);
    if (!abstract_tensor_handle_is_valid(h)) return out;
    out.handle_ = h;
    out.desc_ = desc;
    out.backend_ = use_backend;
    out.owns_handle_ = true;
    return out;
}

AbstractTensor AbstractTensor::wrap(AbstractTensorHandle handle,
                                    const TensorDesc& desc,
                                    TensorBackend* backend,
                                    bool owns_handle) {
    AbstractTensor out;
    out.handle_ = handle;
    out.desc_ = desc;
    out.backend_ = backend;
    out.owns_handle_ = owns_handle;
    out.pool_ = nullptr;
    out.pool_desc_ = {};
    return out;
}

void AbstractTensor::set_dtype(TensorDType dtype) {
    desc_.dtype = dtype;
}

void AbstractTensor::set_slice_affine_row_major(const float* affine16) {
    if (!affine16) return;
    desc_.slice.valid = true;
    desc_.slice.has_affine = true;
    std::memcpy(desc_.slice.affine, affine16, sizeof(desc_.slice.affine));
}

void AbstractTensor::set_slice_saturate_threshold(float threshold) {
    desc_.slice.valid = true;
    desc_.slice.saturate = true;
    desc_.slice.saturate_threshold = threshold;
}

void AbstractTensor::reset() {
    if (owns_handle_ && backend_ && abstract_tensor_handle_is_valid(handle_)) {
        if (pool_) {
            AbstractTensorPool* pool = pool_;
            AbstractTensorHandle handle = handle_;
            TensorBackend* backend = backend_;
            TensorDesc pool_desc = pool_desc_.shape.dims.empty() ? desc_ : pool_desc_;
            handle_ = {};
            backend_ = nullptr;
            owns_handle_ = true;
            desc_ = {};
            pool_ = nullptr;
            pool_desc_ = {};
            pool->release_handle(handle, backend, pool_desc);
            return;
        }
        backend_->destroy(handle_);
    }
    handle_ = {};
    backend_ = nullptr;
    owns_handle_ = true;
    desc_ = {};
    pool_ = nullptr;
    pool_desc_ = {};
}

void AbstractTensor::release() {
    owns_handle_ = false;
}

bool AbstractTensor::refresh_desc() {
    if (!backend_ || !abstract_tensor_handle_is_valid(handle_)) return false;
    TensorDesc tmp{};
    if (!backend_->describe(handle_, &tmp)) return false;
    desc_ = std::move(tmp);
    return true;
}

AbstractTensor AbstractTensor::slice(const TensorIndexSpec& index) const {
    if (!backend_ || !abstract_tensor_handle_is_valid(handle_)) return {};
    AbstractTensorHandle out_handle{};
    TensorDesc out_desc{};
    if (!backend_->get_item(handle_, index, &out_handle, &out_desc)) return {};
    return AbstractTensor::wrap(out_handle, out_desc, backend_, true);
}

bool AbstractTensor::set_slice(const TensorIndexSpec& index, const AbstractTensor& value) {
    if (!backend_ || !abstract_tensor_handle_is_valid(handle_)) return false;
    if (!value.backend_ || !abstract_tensor_handle_is_valid(value.handle_)) return false;
    if (backend_ != value.backend_) return false;
    return backend_->set_item(handle_, index, value.handle_);
}

AbstractTensor AbstractTensor::operator()(const TensorIndexSpec& index) const {
    return slice(index);
}

AbstractTensor AbstractTensor::operator()(std::initializer_list<TensorIndex> dims) const {
    TensorIndexSpec spec;
    spec.dims.assign(dims.begin(), dims.end());
    return slice(spec);
}

bool AbstractTensor::set(std::initializer_list<TensorIndex> dims, const AbstractTensor& value) {
    TensorIndexSpec spec;
    spec.dims.assign(dims.begin(), dims.end());
    return set_slice(spec, value);
}

} // namespace nodus::tensors
