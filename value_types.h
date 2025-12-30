#pragma once

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

// Raw stack frame which holds contiguous elements of a fixed byte stride.
struct RawStackFrame {
    uint8_t* bytes = nullptr; // storage pointer
    int32_t elem_size = 0;    // bytes per element
    int32_t count = 0;        // number of elements currently present
    int32_t capacity = 0;     // capacity in elements
    // Optional per-byte type mask. If non-null, callers must ensure the
    // array has length >= capacity * elem_size and entries correspond to
    // the underlying storage bytes. This keeps the mask in sync even if
    // element sizes vary and avoids ctor/dtor semantics in the raw stack.
    ValueTypeId* types_per_byte = nullptr;
};

// Initialize a RawStackFrame's `types_per_byte` mask. Allocates an array of
// size `capacity * elem_size` and fills it with the default primitive
// `VT_UINT8`. Returns true on success, false on allocation failure.
inline bool raw_stack_init_types_mask(RawStackFrame& frame) {
    if (frame.elem_size <= 0 || frame.capacity <= 0) return false;
    size_t len = static_cast<size_t>(frame.capacity) * static_cast<size_t>(frame.elem_size);
    try {
        frame.types_per_byte = new ValueTypeId[len];
    } catch (...) {
        frame.types_per_byte = nullptr;
        return false;
    }
    ValueTypeId def = ValueTypeRegistry::global().builtin(VT_UINT8);
    for (size_t i = 0; i < len; ++i) frame.types_per_byte[i] = def;
    return true;
}

// Free a previously allocated `types_per_byte` mask. Leaves pointer null.
inline void raw_stack_free_types_mask(RawStackFrame& frame) {
    if (frame.types_per_byte) {
        delete [] frame.types_per_byte;
        frame.types_per_byte = nullptr;
    }
}

// Push a single typed element onto the raw stack. Returns 1 on success, 0 otherwise.
inline int raw_stack_push_typed(RawStackFrame& frame, const void* in, ValueTypeId type_id) {
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt || !in) return 0;
    if (frame.elem_size != 0 && frame.elem_size != static_cast<int32_t>(vt->size)) return 0;
    if (frame.elem_size == 0) frame.elem_size = static_cast<int32_t>(vt->size);
    int free_space = frame.capacity - frame.count;
    if (free_space <= 0) return 0;
    uint8_t* dst = frame.bytes + static_cast<size_t>(frame.count) * frame.elem_size;
    std::memcpy(dst, in, vt->size);
    if (frame.types_per_byte) {
        size_t start = static_cast<size_t>(frame.count) * frame.elem_size;
        for (size_t i = 0; i < vt->size; ++i) frame.types_per_byte[start + i] = type_id;
    }
    frame.count += 1;
    return 1;
}

// Pop a single typed element from the raw stack into out. Returns 1 on success, 0 otherwise.
inline int raw_stack_pop_typed(RawStackFrame& frame, void* out, ValueTypeId type_id) {
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt || !out) return 0;
    if (frame.elem_size != 0 && frame.elem_size != static_cast<int32_t>(vt->size)) return 0;
    if (frame.count <= 0) return 0;
    if (frame.elem_size == 0) frame.elem_size = static_cast<int32_t>(vt->size);
    uint8_t* src = frame.bytes + static_cast<size_t>(frame.count - 1) * frame.elem_size;
    std::memcpy(out, src, vt->size);
    if (frame.types_per_byte) {
        size_t start = static_cast<size_t>(frame.count - 1) * frame.elem_size;
        for (size_t i = 0; i < vt->size; ++i) frame.types_per_byte[start + i] = kInvalidValueTypeId;
    }
    frame.count -= 1;
    return 1;
}

// Push up to n elements (typed) using single memcpy. Returns number pushed.
inline int raw_stack_push_n(RawStackFrame& frame, const void* in, int n, ValueTypeId type_id) {
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt || !in || n <= 0) return 0;
    if (frame.elem_size != 0 && frame.elem_size != static_cast<int32_t>(vt->size)) return 0;
    if (frame.elem_size == 0) frame.elem_size = static_cast<int32_t>(vt->size);
    int free_space = frame.capacity - frame.count;
    if (free_space <= 0) return 0;
    int to = (n < free_space) ? n : free_space;
    uint8_t* dst = frame.bytes + static_cast<size_t>(frame.count) * frame.elem_size;
    std::memcpy(dst, in, static_cast<size_t>(to) * frame.elem_size);
    if (frame.types_per_byte) {
        size_t start = static_cast<size_t>(frame.count) * frame.elem_size;
        for (int i = 0; i < to; ++i) {
            for (size_t b = 0; b < vt->size; ++b) frame.types_per_byte[start + static_cast<size_t>(i) * frame.elem_size + b] = type_id;
        }
    }
    frame.count += to;
    return to;
}

// Pop up to n elements (typed) preserving element-order using memcpy.
// The first element written to out corresponds to the oldest of the popped
// block (i.e. same order as they were pushed). Returns number popped.
inline int raw_stack_pop_block(RawStackFrame& frame, void* out, int n, ValueTypeId type_id) {
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt || !out || n <= 0) return 0;
    if (frame.elem_size != 0 && frame.elem_size != static_cast<int32_t>(vt->size)) return 0;
    if (frame.count <= 0) return 0;
    if (frame.elem_size == 0) frame.elem_size = static_cast<int32_t>(vt->size);
    int avail = frame.count;
    int to = (n < avail) ? n : avail;
    uint8_t* src = frame.bytes + static_cast<size_t>(frame.count - to) * frame.elem_size;
    std::memcpy(out, src, static_cast<size_t>(to) * frame.elem_size);
    if (frame.types_per_byte) {
        size_t start = static_cast<size_t>(frame.count - to) * frame.elem_size;
        for (int i = 0; i < to; ++i) {
            for (int b = 0; b < vt->size; ++b) frame.types_per_byte[start + static_cast<size_t>(i) * frame.elem_size + b] = kInvalidValueTypeId;
        }
    }
    frame.count -= to;
    return to;
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

// Peek the ValueTypeId of the top-most element without popping.
// Returns 1 and writes `out` on success, 0 if stack empty or types not available.
inline int raw_stack_peek_type(const RawStackFrame& frame, ValueTypeId& out) {
    if (frame.count <= 0) return 0;
    int32_t es = frame.elem_size;
    if (es <= 0) return 0;
    if (!frame.types_per_byte) {
        out = ValueTypeRegistry::global().builtin(VT_UINT8); // default to byte/char
        return 1;
    }
    out = frame.types_per_byte[static_cast<size_t>(frame.count - 1) * es];
    return 1;
}

// Pop consecutive 64-bit integer elements (signed) from the top of the
// raw stack while their registered type is an integer builtin. Popped
// integers are appended to `out` in pop order (top-most first). Returns
// the number of integers popped.
inline int raw_stack_pop_int64s_while(RawStackFrame& frame, std::vector<int64_t>& out) {
    int cnt = 0;
    ValueTypeId tid = kInvalidValueTypeId;
    while (frame.count > 0) {
        if (!frame.types_per_byte) break;
        int32_t es = frame.elem_size;
        if (es <= 0) break;
        tid = frame.types_per_byte[static_cast<size_t>(frame.count - 1) * es];
        // Check against registered integer builtin ids
        if (tid == ValueTypeRegistry::global().builtin(VT_INT64) || tid == ValueTypeRegistry::global().builtin(VT_UINT64) || tid == ValueTypeRegistry::global().builtin(VT_INT32) || tid == ValueTypeRegistry::global().builtin(VT_UINT32)) {
            // Pop into a 64-bit container depending on elem_size
            if (es == static_cast<int32_t>(sizeof(int64_t))) {
                int64_t v = 0;
                raw_stack_pop_typed(frame, &v, tid);
                out.push_back(v);
            } else if (es == static_cast<int32_t>(sizeof(int32_t))) {
                int32_t v32 = 0;
                raw_stack_pop_typed(frame, &v32, tid);
                out.push_back(static_cast<int64_t>(v32));
            } else {
                break; // element size mismatch
            }
            cnt += 1;
            continue;
        }
        break; // non-integer type encountered
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
    raw_stack_free_types_mask(*frame);
    if (frame->bytes) {
        delete [] frame->bytes;
        frame->bytes = nullptr;
    }
    delete frame;
}
