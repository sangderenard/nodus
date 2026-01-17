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
    bool get_item(AbstractTensorHandle handle,
                  const TensorIndexSpec& index,
                  AbstractTensorHandle* out_handle,
                  TensorDesc* out_desc) override;
    bool set_item(AbstractTensorHandle handle,
                  const TensorIndexSpec& index,
                  AbstractTensorHandle value) override;

    struct AllocationInfo {
        void* data = nullptr;
        size_t bytes = 0;
        uint16_t bucket = 0;
        bool alive = false;
    };

    struct ArenaStats {
        uint64_t reserve_bytes = 0;
        uint64_t active_leases = 0;
        uint64_t active_leased_bytes = 0;
        uint64_t peak_active_leased_bytes = 0;
        uint64_t alloc_calls = 0;
        uint64_t free_calls = 0;
        uint64_t alloc_fail_oom = 0;
        uint64_t free_drop_no_nodes = 0;
        uint32_t span_nodes_capacity = 0;
        uint32_t span_nodes_free = 0;
        uint64_t free_spans = 0;
        uint64_t free_bytes = 0;
        uint64_t largest_free_span = 0;
    };

    bool get_allocation_info(AbstractTensorHandle handle, AllocationInfo* out) const;
    bool get_arena_stats(ArenaStats* out) const;
    bool map(AbstractTensorHandle handle, void** out_data, size_t* out_bytes) const;
    void unmap(AbstractTensorHandle handle) const;

    // Ensure the payload for this tensor is zeroed, preferably by swapping in a clean arena lease.
    // Falls back to clearing the existing lease when swap is not possible.
    bool ensure_zeroed(AbstractTensorHandle handle, const TensorDesc& desc);
};

InMemoryBackend& in_memory_backend_singleton();
void register_in_memory_backend(bool make_default = false);

} // namespace nodus::tensors
