#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstring>

// Simple runtime type description used to register primitive and struct
// shapes that can be carried on table edges and stacks. The registry is
// intentionally small and C-friendly in layout to make integration simpler
// for allocator and schema tools later.

using ValueTypeId = int32_t;
static constexpr ValueTypeId kInvalidValueTypeId = -1;

struct ValueField {
    std::string name;
    uint32_t offset = 0; // bytes from struct base
    uint32_t size = 0;   // bytes of the field
    ValueTypeId type = kInvalidValueTypeId; // nested type id (primitive or struct)
};

struct ValueType {
    std::string name;
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

    // Register a primitive type (e.g., "float32", "int32"). Returns id.
    ValueTypeId register_primitive(const std::string& name, uint32_t size) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = name_to_id_.find(name);
        if (it != name_to_id_.end()) return it->second;
        ValueTypeId id = static_cast<ValueTypeId>(types_.size());
        ValueType t;
        t.name = name;
        t.size = size;
        t.is_primitive = true;
        types_.push_back(t);
        name_to_id_[name] = id;
        return id;
    }

    // Register a struct type. Field offsets and sizes must be supplied.
    ValueTypeId register_struct(const std::string& name, const std::vector<ValueField>& fields, uint32_t size) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = name_to_id_.find(name);
        if (it != name_to_id_.end()) return it->second;
        ValueTypeId id = static_cast<ValueTypeId>(types_.size());
        ValueType t;
        t.name = name;
        t.size = size;
        t.is_primitive = false;
        t.fields = fields;
        types_.push_back(t);
        name_to_id_[name] = id;
        return id;
    }

    const ValueType* get(ValueTypeId id) const {
        std::lock_guard<std::mutex> lk(mu_);
        if (id < 0 || static_cast<size_t>(id) >= types_.size()) return nullptr;
        return &types_[static_cast<size_t>(id)];
    }

    ValueTypeId find_by_name(const std::string& name) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = name_to_id_.find(name);
        if (it == name_to_id_.end()) return kInvalidValueTypeId;
        return it->second;
    }

private:
    mutable std::mutex mu_;
    std::vector<ValueType> types_;
    std::unordered_map<std::string, ValueTypeId> name_to_id_;
    ValueTypeRegistry() {
        // Pre-register common primitives
        register_primitive("float32", sizeof(float));
        register_primitive("int32", sizeof(int32_t));
        register_primitive("uint32", sizeof(uint32_t));
        register_primitive("ptr", sizeof(void*));
    }
};

// Raw stack frame which holds contiguous elements of a fixed byte stride.
struct RawStackFrame {
    uint8_t* bytes = nullptr; // storage pointer
    int32_t elem_size = 0;    // bytes per element
    int32_t count = 0;        // number of elements currently present
    int32_t capacity = 0;     // capacity in elements
};

// Push a single typed element onto the raw stack. Returns 1 on success, 0 otherwise.
inline int raw_stack_push_typed(RawStackFrame& frame, const void* in, ValueTypeId type_id) {
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt || !in) return 0;
    if (frame.elem_size != static_cast<int32_t>(vt->size)) return 0;
    int free_space = frame.capacity - frame.count;
    if (free_space <= 0) return 0;
    uint8_t* dst = frame.bytes + static_cast<size_t>(frame.count) * frame.elem_size;
    std::memcpy(dst, in, vt->size);
    frame.count += 1;
    return 1;
}

// Pop a single typed element from the raw stack into out. Returns 1 on success, 0 otherwise.
inline int raw_stack_pop_typed(RawStackFrame& frame, void* out, ValueTypeId type_id) {
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt || !out) return 0;
    if (frame.elem_size != static_cast<int32_t>(vt->size)) return 0;
    if (frame.count <= 0) return 0;
    uint8_t* src = frame.bytes + static_cast<size_t>(frame.count - 1) * frame.elem_size;
    std::memcpy(out, src, vt->size);
    frame.count -= 1;
    return 1;
}

// Push up to n elements (typed) using single memcpy. Returns number pushed.
inline int raw_stack_push_n(RawStackFrame& frame, const void* in, int n, ValueTypeId type_id) {
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt || !in || n <= 0) return 0;
    if (frame.elem_size != static_cast<int32_t>(vt->size)) return 0;
    int free_space = frame.capacity - frame.count;
    if (free_space <= 0) return 0;
    int to = (n < free_space) ? n : free_space;
    uint8_t* dst = frame.bytes + static_cast<size_t>(frame.count) * frame.elem_size;
    std::memcpy(dst, in, static_cast<size_t>(to) * frame.elem_size);
    frame.count += to;
    return to;
}

// Pop up to n elements (typed) preserving element-order using memcpy.
// The first element written to out corresponds to the oldest of the popped
// block (i.e. same order as they were pushed). Returns number popped.
inline int raw_stack_pop_block(RawStackFrame& frame, void* out, int n, ValueTypeId type_id) {
    const ValueType* vt = ValueTypeRegistry::global().get(type_id);
    if (!vt || !out || n <= 0) return 0;
    if (frame.elem_size != static_cast<int32_t>(vt->size)) return 0;
    if (frame.count <= 0) return 0;
    int avail = frame.count;
    int to = (n < avail) ? n : avail;
    uint8_t* src = frame.bytes + static_cast<size_t>(frame.count - to) * frame.elem_size;
    std::memcpy(out, src, static_cast<size_t>(to) * frame.elem_size);
    frame.count -= to;
    return to;
}
