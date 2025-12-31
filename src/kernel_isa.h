// kernel_isa.h
// Defines the minimal, portable KernelISA for lowering to SPIR-V
// This is the abstract op set for translation, not raw SPIR-V opcodes.
// See spirv_translation.cpp for usage and expansion.

#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <variant>

namespace nodus {
namespace spirv {


// Scalar types for tensors
enum class ScalarType : uint8_t {
    I32,
    U32,
    F32,
    // TODO: F16, I16, U16, F64, etc.
};

// Tensor type
struct TensorType {
    ScalarType scalar{};
    std::vector<uint32_t> shape;
    bool is_buffer = true;
    bool is_readonly = false;
};

// Value reference (SSA)
struct ValueRef {
    uint32_t id = 0;
};

// Operand: ValueRef or immediate
struct Operand {
    std::variant<ValueRef, int32_t, uint32_t, float> v;

    static Operand ref(ValueRef r) { return Operand{r}; }
    static Operand i(int32_t x) { return Operand{x}; }
    static Operand u(uint32_t x) { return Operand{x}; }
    static Operand f(float x) { return Operand{x}; }
};

// Kernel IR opcodes: original 20-op set for maximal compatibility
enum class OpCode : uint16_t {
    MODULE_BEGIN,      // env, capabilities, memory_model
    KERNEL_ENTRY,      // name, local_size_xyz, interface_signature
    TYPE,              // kind, layout
    CONST,             // type, literal
    SPEC_CONST,        // type, id, default
    VAR,               // storage_class, type, binding/set, init?
    ADDR,              // base_ptr, index_chain, bounds_policy
    LOAD,              // addr
    STORE,             // addr, value
    MEMCPY,            // dst, src_or_value, bytes
    UNARY,             // op, x
    BINARY,            // op, a, b
    TERNARY,           // op, a, b, c
    CMP,               // pred, a, b
    SELECT,            // cond, t, f
    CAST,              // kind, x, dst_type
    EXTRACT,           // composite, idx
    INSERT,            // composite, idx, value
    SHUFFLE,           // vecA, vecB, mask
    IF,                // cond, then_block, else_block
    BARRIER,           // kind, scope, semantics
    ATOMIC             // kind, scope, semantics, addr, args...
};

// IR instruction
struct Instruction {
    OpCode op{};
    std::vector<Operand> inputs;
    std::vector<ValueRef> outputs;
};

// Value definition
struct ValueDef {
    TensorType type;
    std::string debug_name;
};

// KernelIR: full compute kernel region
struct KernelIR {
    std::string name;
    std::vector<ValueDef> values;
    std::vector<uint32_t> buffer_value_ids;
    std::vector<Instruction> instrs;
    std::array<uint32_t, 3> suggested_local_size{16, 16, 1};
    uint32_t element_count = 0;
};

} // namespace spirv
} // namespace nodus
