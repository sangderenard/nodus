#pragma once

#include <cstddef>

#include "common/tensors/abstraction/tensor_backend.h"

namespace nodus::tensors {

class InMemoryBackend final : public TensorBackend {
public:
    const char* name() const override;

    AbstractTensorHandle create(const TensorDesc& desc) override;
    void destroy(AbstractTensorHandle handle) override;
    bool describe(AbstractTensorHandle handle, TensorDesc* out) const override;

    struct AllocationInfo {
        void* data = nullptr;
        size_t bytes = 0;
        uint16_t bucket = 0;
        bool alive = false;
    };

    bool get_allocation_info(AbstractTensorHandle handle, AllocationInfo* out) const;
    bool map(AbstractTensorHandle handle, void** out_data, size_t* out_bytes) const;
    void unmap(AbstractTensorHandle handle) const;
};

InMemoryBackend& in_memory_backend_singleton();
void register_in_memory_backend(bool make_default = false);

} // namespace nodus::tensors
