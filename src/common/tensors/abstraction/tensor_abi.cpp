// C ABI over the in-memory tensor backend. See tensor_abi.h for why this
// exists: it is the transport a host language uses to reach nodus tensors
// without any shared dependency beyond its own standard library.
//
// Every function here is a translation, not a policy: the arena's behaviour
// (reserve/commit, span coalescing, clean/dirty leases, exodus) stays entirely
// inside InMemoryBackend. What this file adds is a boundary those C++ types
// cannot cross on their own -- TensorDesc holds vectors, ArenaBackingPolicy and
// TensorDType are enum classes, and AbstractTensorHandle is a struct.

#include "common/tensors/abstraction/tensor_abi.h"

#include <algorithm>
#include <cstring>

#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_types.h"

namespace {

using nodus::tensors::AbstractTensorHandle;
using nodus::tensors::InMemoryBackend;
using nodus::tensors::TensorDesc;
using nodus::tensors::TensorDType;
using nodus::tensors::TensorLayout;

InMemoryBackend& backend() {
    return nodus::tensors::in_memory_backend_singleton();
}

AbstractTensorHandle to_handle(uint64_t id) {
    AbstractTensorHandle handle{};
    handle.id = id;
    return handle;
}

bool dtype_from_int(int32_t value, TensorDType* out) {
    if (value < 0 || value > static_cast<int32_t>(TensorDType::Ptr)) {
        return false;
    }
    *out = static_cast<TensorDType>(static_cast<uint8_t>(value));
    return true;
}

// Byte length of a described tensor. Strides are in elements, so a strided
// tensor's footprint is decided by its furthest reachable element rather than
// by the product of its dimensions.
uint64_t payload_bytes(const TensorDesc& desc) {
    const uint64_t element_bytes =
        nodus::tensors::tensor_dtype_size_bytes(desc.dtype);
    if (desc.shape.dims.empty()) {
        return element_bytes;
    }
    if (desc.layout == TensorLayout::Strided &&
        desc.strides.elems.size() == desc.shape.dims.size()) {
        uint64_t furthest = 0;
        for (size_t axis = 0; axis < desc.shape.dims.size(); ++axis) {
            const uint64_t extent = desc.shape.dims[axis];
            if (extent == 0) {
                return 0;
            }
            furthest += (extent - 1) * desc.strides.elems[axis];
        }
        return (furthest + 1) * element_bytes;
    }
    return desc.shape.element_count() * element_bytes;
}

}  // namespace

extern "C" {

int32_t nodus_tensor_arena_configure(uint64_t reserve_bytes,
                                     uint64_t min_commit_bytes,
                                     uint64_t span_nodes,
                                     int32_t backing_policy) {
    if (backing_policy != NODUS_ARENA_VIRTUAL_ONLY &&
        backing_policy != NODUS_ARENA_ALLOW_NON_VIRTUAL) {
        return NODUS_ERR_INVALID_ARG;
    }
    // Zero means "leave this one alone", so a caller can set one knob without
    // restating the others it does not care about.
    if (reserve_bytes) {
        InMemoryBackend::set_arena_reserve_bytes(reserve_bytes);
    }
    if (min_commit_bytes) {
        InMemoryBackend::set_arena_min_commit_bytes(min_commit_bytes);
    }
    if (span_nodes) {
        InMemoryBackend::set_arena_span_nodes(span_nodes);
    }
    InMemoryBackend::set_arena_backing_policy(
        backing_policy == NODUS_ARENA_VIRTUAL_ONLY
            ? InMemoryBackend::ArenaBackingPolicy::VirtualOnly
            : InMemoryBackend::ArenaBackingPolicy::AllowNonVirtual);
    return NODUS_OK;
}

int32_t nodus_tensor_arena_stats(NodusArenaStats* out) {
    if (!out) {
        return NODUS_ERR_INVALID_ARG;
    }
    InMemoryBackend::ArenaStats stats{};
    if (!backend().get_arena_stats(&stats)) {
        return NODUS_ERR_UNSUPPORTED;
    }
    out->reserve_bytes = stats.reserve_bytes;
    out->active_leases = stats.active_leases;
    out->active_leased_bytes = stats.active_leased_bytes;
    out->peak_active_leased_bytes = stats.peak_active_leased_bytes;
    out->alloc_calls = stats.alloc_calls;
    out->free_calls = stats.free_calls;
    out->alloc_fail_oom = stats.alloc_fail_oom;
    out->free_drop_no_nodes = stats.free_drop_no_nodes;
    out->span_nodes_capacity = stats.span_nodes_capacity;
    out->span_nodes_free = stats.span_nodes_free;
    out->free_spans = stats.free_spans;
    out->free_bytes = stats.free_bytes;
    out->largest_free_span = stats.largest_free_span;
    return NODUS_OK;
}

uint64_t nodus_tensor_arena_system_available_bytes(void) {
    return InMemoryBackend::get_system_available_bytes();
}

void nodus_tensor_arena_reset(void) {
    InMemoryBackend::reset_arena_for_testing();
}

uint64_t nodus_tensor_create(int32_t dtype, uint32_t rank, const uint64_t* shape) {
    TensorDType typed = TensorDType::Unknown;
    if (!dtype_from_int(dtype, &typed) || typed == TensorDType::Unknown) {
        return 0;
    }
    if (rank > NODUS_TENSOR_MAX_RANK || (rank > 0 && !shape)) {
        return 0;
    }
    TensorDesc desc{};
    desc.dtype = typed;
    desc.layout = TensorLayout::Dense;
    desc.is_buffer = true;
    desc.shape.dims.reserve(rank);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        const uint64_t extent = shape[axis];
        // TensorShape carries 32-bit dimensions; a larger extent would wrap
        // silently and hand back a tensor smaller than the caller asked for.
        if (extent > 0xFFFFFFFFull) {
            return 0;
        }
        desc.shape.dims.push_back(static_cast<uint32_t>(extent));
    }
    return backend().create(desc).id;
}

void nodus_tensor_destroy(uint64_t handle) {
    if (handle) {
        backend().destroy(to_handle(handle));
    }
}

int32_t nodus_tensor_describe(uint64_t handle, NodusTensorDesc* out) {
    if (!handle || !out) {
        return NODUS_ERR_INVALID_ARG;
    }
    TensorDesc desc{};
    if (!backend().describe(to_handle(handle), &desc)) {
        return NODUS_ERR_UNKNOWN_HANDLE;
    }
    if (desc.shape.dims.size() > NODUS_TENSOR_MAX_RANK) {
        return NODUS_ERR_RANK_TOO_HIGH;
    }
    std::memset(out, 0, sizeof(*out));
    out->dtype = static_cast<uint8_t>(desc.dtype);
    out->layout = static_cast<uint8_t>(desc.layout);
    out->is_readonly = desc.is_readonly ? 1u : 0u;
    out->is_buffer = desc.is_buffer ? 1u : 0u;
    out->rank = static_cast<uint32_t>(desc.shape.dims.size());
    for (uint32_t axis = 0; axis < out->rank; ++axis) {
        out->shape[axis] = desc.shape.dims[axis];
        out->strides[axis] = axis < desc.strides.elems.size()
                                 ? desc.strides.elems[axis]
                                 : 0ull;
    }
    out->element_count = desc.shape.element_count();
    out->element_bytes = nodus::tensors::tensor_dtype_size_bytes(desc.dtype);
    out->total_bytes = payload_bytes(desc);
    return NODUS_OK;
}

int32_t nodus_tensor_map(uint64_t handle, void** out_data, uint64_t* out_bytes) {
    if (!handle || !out_data) {
        return NODUS_ERR_INVALID_ARG;
    }
    void* data = nullptr;
    size_t bytes = 0;
    if (!backend().map(to_handle(handle), &data, &bytes)) {
        return NODUS_ERR_UNKNOWN_HANDLE;
    }
    *out_data = data;
    if (out_bytes) {
        *out_bytes = static_cast<uint64_t>(bytes);
    }
    return NODUS_OK;
}

void nodus_tensor_unmap(uint64_t handle) {
    if (handle) {
        backend().unmap(to_handle(handle));
    }
}

int32_t nodus_tensor_zero(uint64_t handle, uint64_t byte_offset, uint64_t byte_count) {
    if (!handle) {
        return NODUS_ERR_INVALID_ARG;
    }
    TensorDesc desc{};
    if (!backend().describe(to_handle(handle), &desc)) {
        return NODUS_ERR_UNKNOWN_HANDLE;
    }
    // A partial clear must stay in place: swapping the lease would zero the
    // bytes outside the requested span as well.
    const bool keep_in_place = byte_offset != 0 || byte_count != 0;
    return backend().ensure_zeroed(to_handle(handle), desc, keep_in_place,
                                   byte_offset, byte_count)
               ? NODUS_OK
               : NODUS_ERR_UNSUPPORTED;
}

int64_t nodus_tensor_write(uint64_t handle, uint64_t byte_offset,
                           const void* source, uint64_t bytes) {
    if (!handle || !source) {
        return NODUS_ERR_INVALID_ARG;
    }
    TensorDesc desc{};
    if (!backend().describe(to_handle(handle), &desc)) {
        return NODUS_ERR_UNKNOWN_HANDLE;
    }
    void* data = nullptr;
    size_t mapped = 0;
    if (!backend().map(to_handle(handle), &data, &mapped)) {
        return NODUS_ERR_UNKNOWN_HANDLE;
    }
    // The lease is aligned up (64 bytes), so it is routinely larger than the
    // tensor. Clamping to the mapping would let a caller run past the
    // tensor's own extent and still land inside allocated memory, which is
    // the kind of overrun that is not caught until something else reads it.
    const uint64_t limit =
        std::min<uint64_t>(payload_bytes(desc), static_cast<uint64_t>(mapped));
    mapped = static_cast<size_t>(limit);
    int64_t written = 0;
    if (byte_offset < static_cast<uint64_t>(mapped)) {
        const uint64_t room = static_cast<uint64_t>(mapped) - byte_offset;
        const uint64_t count = std::min(room, bytes);
        std::memcpy(static_cast<uint8_t*>(data) + byte_offset, source,
                    static_cast<size_t>(count));
        written = static_cast<int64_t>(count);
    }
    backend().unmap(to_handle(handle));
    return written;
}

int64_t nodus_tensor_read(uint64_t handle, uint64_t byte_offset,
                          void* destination, uint64_t bytes) {
    if (!handle || !destination) {
        return NODUS_ERR_INVALID_ARG;
    }
    TensorDesc desc{};
    if (!backend().describe(to_handle(handle), &desc)) {
        return NODUS_ERR_UNKNOWN_HANDLE;
    }
    void* data = nullptr;
    size_t mapped = 0;
    if (!backend().map(to_handle(handle), &data, &mapped)) {
        return NODUS_ERR_UNKNOWN_HANDLE;
    }
    // The lease is aligned up (64 bytes), so it is routinely larger than the
    // tensor. Clamping to the mapping would let a caller run past the
    // tensor's own extent and still land inside allocated memory, which is
    // the kind of overrun that is not caught until something else reads it.
    const uint64_t limit =
        std::min<uint64_t>(payload_bytes(desc), static_cast<uint64_t>(mapped));
    mapped = static_cast<size_t>(limit);
    int64_t got = 0;
    if (byte_offset < static_cast<uint64_t>(mapped)) {
        const uint64_t room = static_cast<uint64_t>(mapped) - byte_offset;
        const uint64_t count = std::min(room, bytes);
        std::memcpy(destination, static_cast<const uint8_t*>(data) + byte_offset,
                    static_cast<size_t>(count));
        got = static_cast<int64_t>(count);
    }
    backend().unmap(to_handle(handle));
    return got;
}

int32_t nodus_tensor_allocation(uint64_t handle,
                                void** out_data,
                                uint64_t* out_bytes,
                                uint16_t* out_bucket) {
    if (!handle) {
        return NODUS_ERR_INVALID_ARG;
    }
    InMemoryBackend::AllocationInfo info{};
    if (!backend().get_allocation_info(to_handle(handle), &info)) {
        return NODUS_ERR_UNKNOWN_HANDLE;
    }
    if (out_data) {
        *out_data = info.data;
    }
    if (out_bytes) {
        *out_bytes = static_cast<uint64_t>(info.bytes);
    }
    if (out_bucket) {
        *out_bucket = info.bucket;
    }
    return info.alive ? NODUS_OK : NODUS_ERR_UNKNOWN_HANDLE;
}

uint32_t nodus_tensor_dtype_size(int32_t dtype) {
    TensorDType typed = TensorDType::Unknown;
    if (!dtype_from_int(dtype, &typed)) {
        return 0;
    }
    return nodus::tensors::tensor_dtype_size_bytes(typed);
}

}  // extern "C"
