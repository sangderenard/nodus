#pragma once

#include <cstdint>

#include "common/tensors/abstraction/tensor_types.h"

namespace nodus::tensors {

struct AbstractTensorHandle {
    uint64_t id = 0;
};

inline bool abstract_tensor_handle_is_valid(AbstractTensorHandle handle) {
    return handle.id != 0;
}

class TensorBackend;

// Move-only handle wrapper with optional ownership semantics.
class AbstractTensor {
public:
    AbstractTensor() = default;
    AbstractTensor(const AbstractTensor&) = delete;
    AbstractTensor& operator=(const AbstractTensor&) = delete;

    AbstractTensor(AbstractTensor&& other) noexcept;
    AbstractTensor& operator=(AbstractTensor&& other) noexcept;
    ~AbstractTensor();

    static AbstractTensor create(const TensorDesc& desc, TensorBackend* backend = nullptr);
    static AbstractTensor wrap(AbstractTensorHandle handle,
                               const TensorDesc& desc,
                               TensorBackend* backend,
                               bool owns_handle);

    bool valid() const { return abstract_tensor_handle_is_valid(handle_); }
    const TensorDesc& desc() const { return desc_; }
    TensorBackend* backend() const { return backend_; }
    AbstractTensorHandle handle() const { return handle_; }

    void set_dtype(TensorDType dtype);
    void reset();
    void release();
    bool refresh_desc();

private:
    AbstractTensorHandle handle_{};
    TensorDesc desc_{};
    TensorBackend* backend_ = nullptr;
    bool owns_handle_ = true;
};

} // namespace nodus::tensors
