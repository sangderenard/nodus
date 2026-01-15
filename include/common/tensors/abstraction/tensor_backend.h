#pragma once

#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include "common/tensors/abstraction/tensor_index.h"
#include "common/tensors/abstraction/tensor_types.h"

namespace nodus::tensors {

class TensorBackend {
public:
    virtual ~TensorBackend() = default;

    virtual const char* name() const = 0;

    virtual AbstractTensorHandle create(const TensorDesc& desc) = 0;
    virtual void destroy(AbstractTensorHandle handle) = 0;
    virtual bool describe(AbstractTensorHandle handle, TensorDesc* out) const = 0;

    virtual bool get_item(AbstractTensorHandle handle,
                          const TensorIndexSpec& index,
                          AbstractTensorHandle* out_handle,
                          TensorDesc* out_desc) = 0;
    virtual bool set_item(AbstractTensorHandle handle,
                          const TensorIndexSpec& index,
                          AbstractTensorHandle value) = 0;
};

} // namespace nodus::tensors
