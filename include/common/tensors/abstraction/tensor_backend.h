#pragma once

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/tensor_types.h"

namespace nodus::tensors {

class TensorBackend {
public:
    virtual ~TensorBackend() = default;

    virtual const char* name() const = 0;

    virtual AbstractTensorHandle create(const TensorDesc& desc) = 0;
    virtual void destroy(AbstractTensorHandle handle) = 0;
    virtual bool describe(AbstractTensorHandle handle, TensorDesc* out) const = 0;
};

} // namespace nodus::tensors
