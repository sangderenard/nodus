#pragma once

#include "mem_backend.h"
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <new>
#include <torch/torch.h>

// Simple runtime type description used to register primitive and struct
// shapes that can be carried on table edges and stacks. The registry is
// intentionally small and C-friendly in layout to make integration simpler
// for allocator and schema tools later.

using ValueTypeId = int32_t;
static constexpr ValueTypeId kInvalidValueTypeId = -1;

// Builtin type enum. Use these for lookup instead of string comparisons.
enum ValueTypeBuiltin : int {
    VT_FLOAT32 = 0,
    VT_FLOAT64,
    VT_INT8,
    VT_INT16,
    VT_INT32,
    VT_INT64,
    VT_UINT8,
    VT_UINT16,
    VT_UINT32,
    VT_UINT64,
    VT_VOID_PTR,
    VT_EIGEN_TENSOR,
    VT_TORCH_TENSOR,
    VT_UTF16_VIEW,
    VT_BUILTIN_COUNT
};
static constexpr int kValueTypePlaceholderCount = 16; // reserved slots for dynamic type building

struct ValueField {
    std::string name;
    uint32_t offset = 0; // bytes from struct base
    uint32_t size = 0;   // bytes of the field
    ValueTypeId type = kInvalidValueTypeId; // nested type id (primitive or struct)
};

struct ValueType {
    char name[64]; // char-array only for rendering state; do not use for lookup
    uint32_t size = 0; // bytes
    bool is_primitive = false;
    std::vector<ValueField> fields; // empty for primitives
};

// Simple singleton registry.
class ValueTypeRegistry {
public:
    static ValueTypeRegistry& global() {
        static ValueTypeRegistry inst;
        return inst;
    }

    // Register a primitive type (name is copied into internal char array). Returns id.
    ValueTypeId register_primitive(const std::string& name, uint32_t size) {
        std::lock_guard<std::mutex> lk(mu_);
        // disallow repeated names via scanning the stored name arrays (cheap here)
        for (size_t i = 0; i < types_.size(); ++i) {
            if (std::strncmp(types_[i].name, name.c_str(), sizeof(types_[i].name)) == 0) return static_cast<ValueTypeId>(i);
        }
        ValueTypeId id = static_cast<ValueTypeId>(types_.size());
        ValueType t;
        std::memset(t.name, 0, sizeof(t.name));
        std::strncpy(t.name, name.c_str(), sizeof(t.name) - 1);
        t.size = size;
        t.is_primitive = true;
        types_.push_back(t);
        return id;
    }

    // Register a struct type. Field offsets and sizes must be supplied.
    ValueTypeId register_struct(const std::string& name, const std::vector<ValueField>& fields, uint32_t size) {
        std::lock_guard<std::mutex> lk(mu_);
        // check existing by name
        for (size_t i = 0; i < types_.size(); ++i) {
            if (std::strncmp(types_[i].name, name.c_str(), sizeof(types_[i].name)) == 0) return static_cast<ValueTypeId>(i);
        }
        ValueTypeId id = static_cast<ValueTypeId>(types_.size());
        ValueType t;
        std::memset(t.name, 0, sizeof(t.name));
        std::strncpy(t.name, name.c_str(), sizeof(t.name) - 1);
        t.size = size;
        t.is_primitive = false;
        t.fields = fields;
        types_.push_back(t);
        return id;
    }

    const ValueType* get(ValueTypeId id) const {
        std::lock_guard<std::mutex> lk(mu_);
        if (id < 0 || static_cast<size_t>(id) >= types_.size()) return nullptr;
        return &types_[static_cast<size_t>(id)];
    }

    // Deprecated: avoid using name-based lookup. Prefer builtin() or stored ids.
    ValueTypeId find_by_name(const std::string& name) const {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < types_.size(); ++i) {
            if (std::strncmp(types_[i].name, name.c_str(), sizeof(types_[i].name)) == 0) return static_cast<ValueTypeId>(i);
        }
        return kInvalidValueTypeId;
    }

    // Return the registered id for a builtin enum. Always valid after construction.
    ValueTypeId builtin(ValueTypeBuiltin b) const {
        std::lock_guard<std::mutex> lk(mu_);
        if (b < 0 || b >= VT_BUILTIN_COUNT) return kInvalidValueTypeId;
        return builtin_ids_[static_cast<size_t>(b)];
    }

private:
    mutable std::mutex mu_;
    std::vector<ValueType> types_;
    std::vector<ValueTypeId> builtin_ids_; // map ValueTypeBuiltin -> ValueTypeId
    ValueTypeRegistry() {
        // Reserve space for builtins + placeholders
        types_.reserve(static_cast<size_t>(VT_BUILTIN_COUNT + kValueTypePlaceholderCount));
        builtin_ids_.assign(static_cast<size_t>(VT_BUILTIN_COUNT), kInvalidValueTypeId);
        // Pre-register common primitives and save builtin ids
        builtin_ids_[VT_FLOAT32] = register_primitive("float32", sizeof(float));
        builtin_ids_[VT_FLOAT64] = register_primitive("float64", sizeof(double));
        builtin_ids_[VT_INT8] = register_primitive("int8", sizeof(int8_t));
        builtin_ids_[VT_INT16] = register_primitive("int16", sizeof(int16_t));
        builtin_ids_[VT_INT32] = register_primitive("int32", sizeof(int32_t));
        builtin_ids_[VT_INT64] = register_primitive("int64", sizeof(int64_t));
        builtin_ids_[VT_UINT8] = register_primitive("uint8", sizeof(uint8_t));
        builtin_ids_[VT_UINT16] = register_primitive("uint16", sizeof(uint16_t));
        builtin_ids_[VT_UINT32] = register_primitive("uint32", sizeof(uint32_t));
        builtin_ids_[VT_UINT64] = register_primitive("uint64", sizeof(uint64_t));
        builtin_ids_[VT_VOID_PTR] = register_primitive("ptr", static_cast<uint32_t>(sizeof(void*)));
        builtin_ids_[VT_EIGEN_TENSOR] = register_primitive("eigen_tensor_view", static_cast<uint32_t>(sizeof(void*)));
        builtin_ids_[VT_TORCH_TENSOR] = register_primitive("torch_tensor_ptr", static_cast<uint32_t>(sizeof(void*)));
        builtin_ids_[VT_UTF16_VIEW] = register_primitive("utf16_view", static_cast<uint32_t>(sizeof(void*)));
        // Reserve some placeholder entries for dynamic types (names are not meaningful for lookup)
        for (int i = 0; i < kValueTypePlaceholderCount; ++i) {
            char tmp[32];
            std::snprintf(tmp, sizeof(tmp), "_ph%d", i);
            register_primitive(tmp, 1);
        }
    }
};

// Raw stack frame which holds contiguous typed bytes (variable element sizes supported).
struct RawStackFrame {
    // Storage is now owned via a backend handle. By default newly-initialized
    // frames allocate a host-backed buffer via `gp_mem_backend_create_host`.
    gp_mem_backend_handle_t backend = nullptr; // backend that owns the storage
    size_t byte_capacity = 0;          // bytes allocated in backend
    size_t byte_count = 0;             // bytes currently occupied
    ValueTypeId* types_per_byte = nullptr; // per-byte type indicator mask (host-side)
};

// Allocate storage and mask for `frame` with the given byte capacity. The
// type mask is filled with `init_type` (defaults to `VT_UINT8`).
inline bool raw_stack_init_frame(RawStackFrame& frame, size_t byte_capacity, ValueTypeId init_type = kInvalidValueTypeId) {
    if (byte_capacity == 0) return false;
    // Allocate the type mask on host
    try {
        frame.types_per_byte = new ValueTypeId[byte_capacity];
    } catch (...) {
        frame.types_per_byte = nullptr;
        return false;
    }
    frame.byte_capacity = byte_capacity;
    frame.byte_count = 0;
    ValueTypeId fill = init_type;
    if (fill == kInvalidValueTypeId) fill = ValueTypeRegistry::global().builtin(VT_UINT8);
    for (size_t i = 0; i < byte_capacity; ++i) frame.types_per_byte[i] = fill;

    // Allocate a host-backed storage buffer via the mem backend API so that
    // higher-level code can treat storage uniformly via backend handles.
    gp_mem_backend_handle_t h = gp_mem_backend_create_host(byte_capacity);
    if (!h) {
        delete [] frame.types_per_byte;
        frame.types_per_byte = nullptr;
        frame.byte_capacity = 0;
        return false;
    }
    frame.backend = h;

    // Zero the backend storage if possible
    const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(frame.backend);
    if (vt && vt->copy_to_backend) {
        std::vector<uint8_t> zeros;
        try { zeros.resize(byte_capacity); } catch (...) { /* ignore */ }
        if (zeros.size() == byte_capacity) vt->copy_to_backend(frame.backend, 0, zeros.data(), byte_capacity);
    } else {
        void* m = gp_mem_backend_map_or_null(frame.backend);
        if (m) {
            std::memset(m, 0, byte_capacity);
            gp_mem_backend_unmap(frame.backend);
        }
    }
    return true;
}

inline void raw_stack_free_mask(RawStackFrame& frame) {
    if (frame.types_per_byte) {
        delete [] frame.types_per_byte;
        frame.types_per_byte = nullptr;
    }
}

// Reset the byte count so the frame appears empty (mask contents are preserved);
// callers should set `byte_count` to zero before reusing the frame.
inline void raw_stack_reset(RawStackFrame& frame) {
    frame.byte_count = 0;
}

// Push a single typed element onto the raw stack. Returns 1 on success, 0 otherwise.
inline int raw_stack_push_typed(RawStackFrame& frame, const void* in, ValueTypeId type_id) {
    if (!frame.types_per_byte || !in) return 0;
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt) return 0;
    size_t needed = vt->size;
    if (frame.byte_count + needed > frame.byte_capacity) return 0;
    // If frame is backed by a mem backend with copy hooks, perform optimized copy
    if (frame.backend) {
        const gp_mem_backend_vtable_t* bvt = gp_mem_backend_get_vtable(frame.backend);
        size_t offset = frame.byte_count;
        if (bvt && bvt->copy_to_backend) {
            if (!bvt->copy_to_backend(frame.backend, offset, in, needed)) return 0;
        } else {
            void* m = gp_mem_backend_map_or_null(frame.backend);
            if (!m) return 0;
            std::memcpy(reinterpret_cast<uint8_t*>(m) + offset, in, needed);
            gp_mem_backend_unmap(frame.backend);
        }
        size_t start = frame.byte_count;
        for (size_t i = 0; i < needed; ++i) frame.types_per_byte[start + i] = type_id;
        frame.byte_count += needed;
        return 1;
    }
    return 0; // without backend we no longer maintain a host heap buffer
}

// Attach/get backend for a RawStackFrame
extern "C" inline int gp_raw_stack_frame_set_backend(RawStackFrame* frame, gp_mem_backend_handle_t h) {
    if (!frame) return 0;
    frame->backend = h;
    return 1;
}

extern "C" inline gp_mem_backend_handle_t gp_raw_stack_frame_get_backend(RawStackFrame* frame) {
    if (!frame) return nullptr;
    return frame->backend;
}

// Swap the backend attached to a RawStackFrame. If `copy_over` is non-zero,
// attempt to copy existing frame bytes (up to `byte_capacity`) from the old
// backend or host memory into the new backend. Returns 1 on success.
extern "C" inline int gp_raw_stack_frame_swap_backend(RawStackFrame* frame, gp_mem_backend_handle_t new_h, int copy_over) {
    if (!frame) return 0;
    gp_mem_backend_handle_t old_h = frame->backend;
    if (old_h == new_h) return 1;
    if (!copy_over) { frame->backend = new_h; return 1; }
    size_t total = frame->byte_capacity;
    if (total == 0) { frame->backend = new_h; return 1; }
    std::vector<uint8_t> tmp;
    try { tmp.resize(total); } catch (...) { return 0; }

    bool read_ok = false;
    if (old_h) {
        const gp_mem_backend_vtable_t* old_vt = gp_mem_backend_get_vtable(old_h);
        if (old_vt && old_vt->copy_from_backend) {
            if (old_vt->copy_from_backend(old_h, 0, tmp.data(), total)) read_ok = true;
        }
        if (!read_ok) {
            void* m = gp_mem_backend_map_or_null(old_h);
            if (m) {
                std::memcpy(tmp.data(), m, total);
                gp_mem_backend_unmap(old_h);
                read_ok = true;
            }
        }
    }
    if (!read_ok) {
        // No old backend: nothing to read from host-side since we store types only.
        // Treat unread data as zeros.
        std::memset(tmp.data(), 0, total);
        read_ok = true;
    }
    if (!read_ok) return 0;

    // Attach new backend and write into it if possible.
    frame->backend = new_h;
    if (new_h) {
        const gp_mem_backend_vtable_t* new_vt = gp_mem_backend_get_vtable(new_h);
        bool write_ok = false;
        if (new_vt && new_vt->copy_to_backend) {
            if (new_vt->copy_to_backend(new_h, 0, tmp.data(), total)) write_ok = true;
        }
        if (!write_ok) {
            void* m = gp_mem_backend_map_or_null(new_h);
            if (m) {
                std::memcpy(m, tmp.data(), total);
                gp_mem_backend_unmap(new_h);
                write_ok = true;
            }
        }
        if (!write_ok) {
            // revert
            frame->backend = old_h;
            return 0;
        }
    }

    // Host-side mask is already in sync; no separate host-side storage is kept.
    return 1;
}

// Pop a single typed element from the raw stack into out. Returns 1 on success, 0 otherwise.
inline int raw_stack_pop_typed(RawStackFrame& frame, void* out, ValueTypeId type_id) {
    if (!frame.types_per_byte || !out) return 0;
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt) return 0;
    size_t needed = vt->size;
    if (frame.byte_count < needed) return 0;
    size_t start = frame.byte_count - needed;
    for (size_t i = 0; i < needed; ++i) {
        if (frame.types_per_byte[start + i] != type_id) return 0;
    }
    // If backend attached and has copy_from_backend, use it
    if (frame.backend) {
        const gp_mem_backend_vtable_t* bvt = gp_mem_backend_get_vtable(frame.backend);
        size_t offset = start;
        if (bvt && bvt->copy_from_backend) {
            if (!bvt->copy_from_backend(frame.backend, offset, out, needed)) return 0;
        } else {
            void* m = gp_mem_backend_map_or_null(frame.backend);
            if (!m) return 0;
            std::memcpy(out, reinterpret_cast<uint8_t*>(m) + offset, needed);
            gp_mem_backend_unmap(frame.backend);
        }
        for (size_t i = 0; i < needed; ++i) frame.types_per_byte[start + i] = kInvalidValueTypeId;
        frame.byte_count -= needed;
        return 1;
    }
    return 0; // no host heap storage without backend
}

// Push up to n elements (typed) using single memcpy. Returns number pushed.
inline int raw_stack_push_n(RawStackFrame& frame, const void* in, int n, ValueTypeId type_id) {
    if (!frame.types_per_byte || !in || n <= 0) return 0;
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt) return 0;
    size_t per = vt->size;
    size_t needed = per * static_cast<size_t>(n);
    if (frame.byte_count + needed > frame.byte_capacity) {
        size_t free_bytes = frame.byte_capacity - frame.byte_count;
        int to = static_cast<int>(free_bytes / per);
        if (to <= 0) return 0;
        needed = per * static_cast<size_t>(to);
        n = to;
    }
    if (frame.backend) {
        const gp_mem_backend_vtable_t* bvt = gp_mem_backend_get_vtable(frame.backend);
        size_t offset = frame.byte_count;
        if (bvt && bvt->copy_to_backend) {
            if (!bvt->copy_to_backend(frame.backend, offset, in, needed)) return 0;
        } else {
            void* m = gp_mem_backend_map_or_null(frame.backend);
            if (!m) return 0;
            std::memcpy(reinterpret_cast<uint8_t*>(m) + offset, in, needed);
            gp_mem_backend_unmap(frame.backend);
        }
        for (int i = 0; i < n; ++i) {
            size_t elem_start = frame.byte_count + static_cast<size_t>(i) * per;
            for (size_t b = 0; b < per; ++b) frame.types_per_byte[elem_start + b] = type_id;
        }
        frame.byte_count += needed;
        return n;
    }
    return 0;
}

// Pop up to n elements (typed) preserving element-order using memcpy.
// Returns number popped.
inline int raw_stack_pop_block(RawStackFrame& frame, void* out, int n, ValueTypeId type_id) {
    if (!frame.types_per_byte || !out || n <= 0) return 0;
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt) return 0;
    size_t per = vt->size;
    size_t total_bytes = per * static_cast<size_t>(n);
    if (frame.byte_count < total_bytes) {
        n = static_cast<int>(frame.byte_count / per);
        total_bytes = per * static_cast<size_t>(n);
        if (n <= 0) return 0;
    }
    size_t start = frame.byte_count - total_bytes;
    for (size_t offset = start; offset < frame.byte_count; offset += per) {
        if (frame.types_per_byte[offset] != type_id) return 0;
    }
    if (frame.backend) {
        const gp_mem_backend_vtable_t* bvt = gp_mem_backend_get_vtable(frame.backend);
        size_t offset = start;
        if (bvt && bvt->copy_from_backend) {
            if (!bvt->copy_from_backend(frame.backend, offset, out, total_bytes)) return 0;
        } else {
            void* m = gp_mem_backend_map_or_null(frame.backend);
            if (!m) return 0;
            std::memcpy(out, reinterpret_cast<uint8_t*>(m) + offset, total_bytes);
            gp_mem_backend_unmap(frame.backend);
        }
        for (size_t i = start; i < start + total_bytes; ++i) frame.types_per_byte[i] = kInvalidValueTypeId;
        frame.byte_count -= total_bytes;
        return n;
    }
    return 0;
}

// Push a single torch::Tensor object (by constructing the object in-place
// inside the RawStackFrame storage). Returns 1 on success, 0 otherwise.
// Push a pointer to a heap-allocated torch::Tensor. The stack stores
// the pointer value (caller owns the pointed-to tensor lifetime).
inline int raw_stack_push_torch(RawStackFrame& frame, torch::Tensor* tensor_ptr) {
    const ValueTypeId tid = ValueTypeRegistry::global().builtin(VT_TORCH_TENSOR);
    return raw_stack_push_typed(frame, &tensor_ptr, tid);
}

// Pop a pointer to a heap-allocated torch::Tensor from the raw stack.
// Returns 1 on success and writes the pointer into `out`.
inline int raw_stack_pop_torch(RawStackFrame& frame, torch::Tensor*& out) {
    const ValueTypeId tid = ValueTypeRegistry::global().builtin(VT_TORCH_TENSOR);
    return raw_stack_pop_typed(frame, &out, tid);
}

// Push bytes from a backend source into this RawStackFrame. The source
// provides `src_h` and `src_offset` (byte offset within that backend). If
// this frame has an attached backend, the transfer will try to move bytes
// directly between backends using `gp_mem_backend_transfer`. Otherwise the
// bytes will be copied into the host buffer. Returns 1 on success.
extern "C" inline int gp_raw_stack_frame_push_from_backend(RawStackFrame* frame, gp_mem_backend_handle_t src_h, size_t src_offset, ValueTypeId type_id) {
    if (!frame) return 0;
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt) return 0;
    size_t needed = vt->size;
    if (frame->byte_count + needed > frame->byte_capacity) return 0;

    // If the frame has a backend attached, prefer backend-to-backend transfer

    if (frame->backend && src_h) {
        if (gp_mem_backend_transfer(src_h, frame->backend, src_offset, frame->byte_count, needed)) {
            for (size_t i = 0; i < needed; ++i) frame->types_per_byte[frame->byte_count + i] = type_id;
            frame->byte_count += needed;
            return 1;
        }
        // fallthrough to host-mediated path
    }

    // Host-mediated: read from src backend into host buffer then memcpy into frame
    std::vector<uint8_t> tmp;
    try { tmp.resize(needed); } catch (...) { return 0; }
    if (src_h) {
        const gp_mem_backend_vtable_t* src_vt = gp_mem_backend_get_vtable(src_h);
        bool ok = false;
        if (src_vt && src_vt->copy_from_backend) ok = src_vt->copy_from_backend(src_h, src_offset, tmp.data(), needed) != 0;
        if (!ok) {
            void* m = gp_mem_backend_map_or_null(src_h);
            if (m) {
                std::memcpy(tmp.data(), reinterpret_cast<uint8_t*>(m) + src_offset, needed);
                gp_mem_backend_unmap(src_h);
                ok = true;
            }
        }
        if (!ok) return 0;
    } else {
        return 0; // no src provided
    }

    // Write into frame's backend storage (or fail)
    if (!frame->backend) return 0;
    const gp_mem_backend_vtable_t* dst_vt = gp_mem_backend_get_vtable(frame->backend);
    bool wrote = false;
    if (dst_vt && dst_vt->copy_to_backend) wrote = dst_vt->copy_to_backend(frame->backend, frame->byte_count, tmp.data(), needed) != 0;
    if (!wrote) {
        void* m = gp_mem_backend_map_or_null(frame->backend);
        if (m) {
            std::memcpy(reinterpret_cast<uint8_t*>(m) + frame->byte_count, tmp.data(), needed);
            gp_mem_backend_unmap(frame->backend);
            wrote = true;
        }
    }
    if (!wrote) return 0;
    for (size_t i = 0; i < needed; ++i) frame->types_per_byte[frame->byte_count + i] = type_id;
    frame->byte_count += needed;
    return 1;
}

// Pop bytes from this RawStackFrame into a destination backend. If the
// destination backend equals the frame backend, prefer an in-backend transfer
// (optimized). Otherwise attempt backend-to-backend transfer via the transfer
// API. Returns 1 on success.
extern "C" inline int gp_raw_stack_frame_pop_into_backend(RawStackFrame* frame, gp_mem_backend_handle_t dst_h, size_t dst_offset, ValueTypeId type_id) {
    if (!frame) return 0;
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt) return 0;
    size_t needed = vt->size;
    if (frame->byte_count < needed) return 0;
    size_t start = frame->byte_count - needed;
    for (size_t i = 0; i < needed; ++i) if (frame->types_per_byte[start + i] != type_id) return 0;

    // If frame has backend and dst is same/different backend, try transfer
    if (frame->backend && dst_h) {
        if (frame->backend == dst_h) {
            if (gp_mem_backend_transfer(frame->backend, dst_h, start, dst_offset, needed)) {
                for (size_t i = 0; i < needed; ++i) frame->types_per_byte[start + i] = kInvalidValueTypeId;
                frame->byte_count -= needed;
                return 1;
            }
        } else {
            if (gp_mem_backend_transfer(frame->backend, dst_h, start, dst_offset, needed)) {
                for (size_t i = 0; i < needed; ++i) frame->types_per_byte[start + i] = kInvalidValueTypeId;
                frame->byte_count -= needed;
                return 1;
            }
        }
        // fallthrough to host-mediated
    }

    // Host-mediated path: read from frame backend (or error) then write into dst
    if (!dst_h) return 0;
    // Read source bytes into tmp
    std::vector<uint8_t> tmp;
    try { tmp.resize(needed); } catch (...) { return 0; }
    bool read_ok = false;
    if (frame->backend) {
        const gp_mem_backend_vtable_t* src_vt = gp_mem_backend_get_vtable(frame->backend);
        if (src_vt && src_vt->copy_from_backend) read_ok = src_vt->copy_from_backend(frame->backend, start, tmp.data(), needed) != 0;
        if (!read_ok) {
            void* m = gp_mem_backend_map_or_null(frame->backend);
            if (m) {
                std::memcpy(tmp.data(), reinterpret_cast<uint8_t*>(m) + start, needed);
                gp_mem_backend_unmap(frame->backend);
                read_ok = true;
            }
        }
    }
    if (!read_ok) return 0;
    const gp_mem_backend_vtable_t* dst_vt = gp_mem_backend_get_vtable(dst_h);
    bool ok = false;
    if (dst_vt && dst_vt->copy_to_backend) ok = dst_vt->copy_to_backend(dst_h, dst_offset, tmp.data(), needed) != 0;
    if (!ok) {
        void* m = gp_mem_backend_map_or_null(dst_h);
        if (m) {
            std::memcpy(reinterpret_cast<uint8_t*>(m) + dst_offset, tmp.data(), needed);
            gp_mem_backend_unmap(dst_h);
            ok = true;
        }
    }
    if (!ok) return 0;
    for (size_t i = 0; i < needed; ++i) frame->types_per_byte[start + i] = kInvalidValueTypeId;
    frame->byte_count -= needed;
    return 1;
}

// Peek the ValueTypeId of the top-most element without popping.
// Returns 1 and writes `out` on success, 0 if stack empty or types not available.
inline int raw_stack_peek_type(const RawStackFrame& frame, ValueTypeId& out) {
    if (frame.byte_count == 0) return 0;
    if (!frame.types_per_byte) {
        out = ValueTypeRegistry::global().builtin(VT_UINT8);
        return 1;
    }
    out = frame.types_per_byte[frame.byte_count - 1];
    if (out == kInvalidValueTypeId) {
        out = ValueTypeRegistry::global().builtin(VT_UINT8);
    }
    return 1;
}

// Pop consecutive 64-bit integer elements (signed) from the top of the
// raw stack while their registered type is an integer builtin. Popped
// integers are appended to `out` in pop order (top-most first). Returns
// the number of integers popped.
inline int raw_stack_pop_int64s_while(RawStackFrame& frame, std::vector<int64_t>& out) {
    int cnt = 0;
    while (frame.byte_count > 0) {
        if (!frame.types_per_byte) break;
        ValueTypeId tid = frame.types_per_byte[frame.byte_count - 1];
        if (tid == kInvalidValueTypeId) break;
        const ValueType* vt = ValueTypeRegistry::global().get(tid);
        if (!vt) break;
        size_t elem_size = vt->size;
        if (elem_size == 0 || frame.byte_count < elem_size) break;
        ValueTypeId int_tid = tid;
        const ValueTypeId vt_int64 = ValueTypeRegistry::global().builtin(VT_INT64);
        const ValueTypeId vt_uint64 = ValueTypeRegistry::global().builtin(VT_UINT64);
        const ValueTypeId vt_int32 = ValueTypeRegistry::global().builtin(VT_INT32);
        const ValueTypeId vt_uint32 = ValueTypeRegistry::global().builtin(VT_UINT32);
        if (int_tid == vt_int64 || int_tid == vt_uint64) {
            int64_t v = 0;
            if (!raw_stack_pop_typed(frame, &v, int_tid)) break;
            out.push_back(v);
        } else if (int_tid == vt_int32 || int_tid == vt_uint32) {
            int32_t v32 = 0;
            if (!raw_stack_pop_typed(frame, &v32, int_tid)) break;
            out.push_back(static_cast<int64_t>(v32));
        } else {
            break;
        }
        cnt += 1;
    }
    return cnt;
}

// Destroy a RawStackFrame instance: free any internal masks and storage
// but do NOT delete or free any pointer values that may be stored inside
// the frame bytes (callers are responsible for those). Safe to call
// on frames allocated with `new RawStackFrame()` whose `bytes` and
// `types_per_byte` were separately allocated.
inline void raw_stack_destroy_frame(RawStackFrame* frame) {
    if (!frame) return;
    raw_stack_free_mask(*frame);
    // Release any backend-owned storage and clear metadata. Callers are
    // responsible for releasing any pointer values stored inside the
    // frame bytes prior to destroying the frame.
    if (frame->backend) {
        gp_mem_backend_release(frame->backend);
        frame->backend = nullptr;
    }
    delete frame;
}

// Return the serialized size (in bytes) required to encode `frame` into a
// compact byte representation. Format:
//   [uint64_t byte_count][byte_count bytes data][byte_count int32 type ids]
extern "C" inline int gp_raw_stack_frame_serialized_size(const RawStackFrame* frame, size_t* out_size) {
    if (!frame || !out_size) return 0;
    uint64_t bc = static_cast<uint64_t>(frame->byte_count);
    // size = 8 (uint64_t) + bc (data) + bc * sizeof(ValueTypeId)
    size_t need = sizeof(uint64_t);
    // Protect against overflow
    if (bc > SIZE_MAX / 2) return 0;
    need += static_cast<size_t>(bc);
    size_t mask_bytes = static_cast<size_t>(bc) * sizeof(ValueTypeId);
    if (mask_bytes > SIZE_MAX - need) return 0;
    need += mask_bytes;
    *out_size = need;
    return 1;
}

// Serialize the given RawStackFrame into a caller-provided buffer. See
// gp_raw_stack_frame_serialized_size for format. Returns 1 on success and
// writes the number of bytes produced to out_written.
extern "C" inline int gp_raw_stack_frame_serialize(const RawStackFrame* frame, void* out_buf, size_t out_buf_len, size_t* out_written) {
    if (!frame || !out_buf || !out_written) return 0;
    size_t need = 0;
    if (!gp_raw_stack_frame_serialized_size(frame, &need)) return 0;
    if (out_buf_len < need) return 0;
    uint8_t* dst = reinterpret_cast<uint8_t*>(out_buf);
    // write byte_count as uint64_t (host endianness)
    uint64_t bc = static_cast<uint64_t>(frame->byte_count);
    std::memcpy(dst, &bc, sizeof(bc));
    size_t off = sizeof(bc);

    // Read frame bytes from backend into buffer[off .. off+bc)
    if (bc > 0) {
        bool read_ok = false;
        if (frame->backend) {
            const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(frame->backend);
            if (vt && vt->copy_from_backend) {
                if (vt->copy_from_backend(frame->backend, 0, dst + off, static_cast<size_t>(bc))) read_ok = true;
            }
            if (!read_ok) {
                void* m = gp_mem_backend_map_or_null(frame->backend);
                if (m) {
                    std::memcpy(dst + off, m, static_cast<size_t>(bc));
                    gp_mem_backend_unmap(frame->backend);
                    read_ok = true;
                }
            }
        }
        if (!read_ok) return 0;
        off += static_cast<size_t>(bc);

        // Copy type mask for the bytes (as int32 per entry)
        for (size_t i = 0; i < static_cast<size_t>(bc); ++i) {
            ValueTypeId tid = frame->types_per_byte[i];
            std::memcpy(dst + off + i * sizeof(ValueTypeId), &tid, sizeof(ValueTypeId));
        }
        off += static_cast<size_t>(bc) * sizeof(ValueTypeId);
    }
    *out_written = off;
    return 1;
}

// Deserialize a serialized RawStackFrame buffer into `frame`. The caller
// must provide a mutable RawStackFrame instance. If `frame` capacity is
// insufficient, this will attempt to resize/replace backend storage and the
// types mask. Returns 1 on success.
extern "C" inline int gp_raw_stack_frame_deserialize(RawStackFrame* frame, const void* buf, size_t buf_len) {
    if (!frame || !buf) return 0;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(buf);
    if (buf_len < sizeof(uint64_t)) return 0;
    uint64_t bc = 0;
    std::memcpy(&bc, src, sizeof(bc));
    size_t need = sizeof(uint64_t) + static_cast<size_t>(bc) + static_cast<size_t>(bc) * sizeof(ValueTypeId);
    if (buf_len < need) return 0;

    // Ensure capacity: if insufficient, try to resize backend or allocate host backend and swap
    if (frame->byte_capacity < static_cast<size_t>(bc)) {
        size_t new_cap = static_cast<size_t>(bc);
        // Try to resize existing backend
        if (frame->backend) {
            const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(frame->backend);
            bool resized = false;
            if (vt && vt->resize) {
                if (vt->resize(frame->backend, new_cap)) resized = true;
            }
            if (!resized) {
                // create host backend and copy existing contents
                gp_mem_backend_handle_t h = gp_mem_backend_create_host(new_cap);
                if (!h) return 0;
                if (!gp_raw_stack_frame_swap_backend(frame, h, 1)) {
                    gp_mem_backend_release(h);
                    return 0;
                }
            }
        } else {
            // No backend present: allocate host backend
            gp_mem_backend_handle_t h = gp_mem_backend_create_host(new_cap);
            if (!h) return 0;
            frame->backend = h;
        }
        // Reallocate mask
        try {
            ValueTypeId* nm = new ValueTypeId[new_cap];
            // Initialize to invalid
            for (size_t i = 0; i < new_cap; ++i) nm[i] = kInvalidValueTypeId;
            // copy existing values if present
            if (frame->types_per_byte) {
                size_t copy_len = std::min(frame->byte_capacity, new_cap);
                for (size_t i = 0; i < copy_len; ++i) nm[i] = frame->types_per_byte[i];
                delete [] frame->types_per_byte;
            }
            frame->types_per_byte = nm;
            frame->byte_capacity = new_cap;
        } catch(...) {
            return 0;
        }
    }

    size_t off = sizeof(uint64_t);
    // Write data bytes into backend at offset 0
    if (bc > 0) {
        bool wrote = false;
        const gp_mem_backend_vtable_t* dst_vt = frame->backend ? gp_mem_backend_get_vtable(frame->backend) : nullptr;
        if (dst_vt && dst_vt->copy_to_backend) {
            if (dst_vt->copy_to_backend(frame->backend, 0, src + off, static_cast<size_t>(bc))) wrote = true;
        }
        if (!wrote) {
            void* m = gp_mem_backend_map_or_null(frame->backend);
            if (!m) return 0;
            std::memcpy(m, src + off, static_cast<size_t>(bc));
            gp_mem_backend_unmap(frame->backend);
            wrote = true;
        }
        if (!wrote) return 0;
        off += static_cast<size_t>(bc);

        // Read mask entries
        for (size_t i = 0; i < static_cast<size_t>(bc); ++i) {
            ValueTypeId tid = kInvalidValueTypeId;
            std::memcpy(&tid, src + off + i * sizeof(ValueTypeId), sizeof(ValueTypeId));
            frame->types_per_byte[i] = tid;
        }
        off += static_cast<size_t>(bc) * sizeof(ValueTypeId);
    }

    frame->byte_count = static_cast<size_t>(bc);
    return 1;
}
