// kernel_isa.h
// Defines the minimal, portable KernelISA for lowering to SPIR-V
// This is the abstract op set for translation, not raw SPIR-V opcodes.
// See spirv_translation.cpp for usage and expansion.

#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <variant>
#include <array>

// Windows headers define CONST as a macro; undef to keep enum names intact.
#ifdef CONST
#undef CONST
#endif

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
// [SIC 2026-07-25] "20-op set" is the original design count, preserved as-is. The enum
// below now also appends logical ops (AND/OR/NOT/XOR), so the live total is higher. The
// "20" reflects historical intent, not the current count.
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
    ATOMIC,            // kind, scope, semantics, addr, args...

    // Logical ops
    AND,               // a, b
    OR,                // a, b
    NOT,               // x
    XOR,               // a, b

    // Bounded iteration. THE missing Tier-0 primitive: without it nothing
    // that walks a buffer -- no reduction, contraction, remap or scan -- can
    // be DEFINED in KernelIR, which is why such operations were previously
    // recorded as permanent holes. It is abundantly safe to add: every
    // target has bounded iteration (SPIR-V structured OpLoopMerge, GLSL/C
    // for-loops, a plain loop on CPU/Eigen), so a Tier-1 recipe built on it
    // lowers everywhere rather than obliging each emitter to re-implement a
    // composite.
    //
    // LOOP_BEGIN/LOOP_END bracket a body in the flat instruction list, in
    // the same structured spirit as IF: no arbitrary branching, one entry,
    // one exit, statically bounded trip count.
    //   LOOP_BEGIN  inputs=[trip_count]  outputs=[index]
    //   ... body instructions, which may read `index` ...
    //   LOOP_END    (no inputs, no outputs)
    //
    // Cooperative yielding rides here: when KernelIR::budget_value_id names
    // a buffer, the loop's condition also consults that budget so a kernel
    // stops of its own accord before a watchdog (Windows TDR) stops it for
    // us. Dressing the control primitive means every generated kernel is
    // bounded by construction, with no caller obliged to remember.
    LOOP_BEGIN,        // trip_count -> index
    LOOP_END           //
};

// IR instruction
//
// `sub_op` supplies the operation selector that UNARY / BINARY / TERNARY / CMP / CAST
// have always implied in their comments above ("op, x", "op, a, b", "pred, a, b",
// "kind, x, dst_type") but never had a value space for -- so those opcodes were
// unspecifiable and no emitter could lower them. Non-negative values are
// `nodus::ops::CanonicalOp` (see include/canonical_ops.h, generated from
// ops/canonical_ops.json); -1 means "not applicable to this opcode".
//
// Deliberately a plain int32_t rather than the enum type: kernel_isa.h stays free of
// any dependency on the generated header. Canonical IDs are append-only catalog
// positions; CTensorOp ordinals are a separate backend capability field.
struct Instruction {
    OpCode op{};
    int32_t sub_op = -1;
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
    // Optional buffer holding a cooperative work budget. When set, LOOP
    // conditions consult it so long-running kernels yield voluntarily
    // instead of being killed by a driver watchdog; the host re-dispatches
    // until the kernel reports completion. kInvalidValueId means "no budget
    // declared", the behavior every existing kernel already has.
    static constexpr uint32_t kNoBudget = 0xFFFFFFFFu;
    uint32_t budget_value_id = kNoBudget;
};

} // namespace spirv
} // namespace nodus
