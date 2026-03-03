#pragma once

#include <cstddef>
#include <cstdint>

#include "common/tensors/abstraction/tensor_backend.h"

namespace nodus::tensors {

class InMemoryBackend final : public TensorBackend {
public:
    enum class ArenaBackingPolicy : uint8_t {
        VirtualOnly,
        AllowNonVirtual
    };

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
    // When keep_in_place is true, never swap the lease and only clear the specified span.
    // byte_offset/byte_count are interpreted in bytes; byte_count == 0 means "clear full tensor payload".
    bool ensure_zeroed(AbstractTensorHandle handle,
                       const TensorDesc& desc,
                       bool keep_in_place = false,
                       uint64_t byte_offset = 0,
                       uint64_t byte_count = 0);

    // Must be set before first arena initialization to select backing strategy.
    static void set_arena_backing_policy(ArenaBackingPolicy policy);
    // Must be set before first arena initialization to control maximum arena size.
    static void set_arena_reserve_bytes(uint64_t bytes);
    // Must be set before first arena initialization to control initial commit size.
    static void set_arena_min_commit_bytes(uint64_t bytes);
    // Must be set before first arena initialization to control span node capacity.
    static void set_arena_span_nodes(uint64_t nodes);
    // For test-only use: clear all arenas so a new policy/size can be applied.
    static void reset_arena_for_testing();
    // Expose system-available bytes for test sizing.
    static uint64_t get_system_available_bytes();
};

InMemoryBackend& in_memory_backend_singleton();
void register_in_memory_backend(bool make_default = false);

} // namespace nodus::tensors
