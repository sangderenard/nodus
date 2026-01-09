#include "common/tensors/abstraction/abstract_tensor.h"

#include "common/tensors/abstraction/tensor_backend.h"
#include "common/tensors/abstraction/tensor_registry.h"

#include <utility>

namespace nodus::tensors {

AbstractTensor::AbstractTensor(AbstractTensor&& other) noexcept {
    handle_ = other.handle_;
    desc_ = std::move(other.desc_);
    backend_ = other.backend_;
    owns_handle_ = other.owns_handle_;
    other.handle_ = {};
    other.backend_ = nullptr;
    other.owns_handle_ = false;
}

AbstractTensor& AbstractTensor::operator=(AbstractTensor&& other) noexcept {
    if (this == &other) return *this;
    reset();
    handle_ = other.handle_;
    desc_ = std::move(other.desc_);
    backend_ = other.backend_;
    owns_handle_ = other.owns_handle_;
    other.handle_ = {};
    other.backend_ = nullptr;
    other.owns_handle_ = false;
    return *this;
}

AbstractTensor::~AbstractTensor() {
    reset();
}

AbstractTensor AbstractTensor::create(const TensorDesc& desc, TensorBackend* backend) {
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
    return out;
}

void AbstractTensor::set_dtype(TensorDType dtype) {
    desc_.dtype = dtype;
}

void AbstractTensor::reset() {
    if (owns_handle_ && backend_ && abstract_tensor_handle_is_valid(handle_)) {
        backend_->destroy(handle_);
    }
    handle_ = {};
    backend_ = nullptr;
    owns_handle_ = true;
    desc_ = {};
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

} // namespace nodus::tensors
