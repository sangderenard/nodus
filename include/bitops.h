// bitops.h
// Bit-level operations and type system for IR and codegen
#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <type_traits>

namespace nodus {
namespace bitops {


// Math/semantic object kinds (nouns)
enum class BitObjectKind {
    INTEGER,
    RATIONAL,
    FLOAT,
    COMPLEX,
    INFINITESIMAL,
    LIMIT,
    DOMAIN_TYPE,
    BOUNDARY,
    TENSOR,
    TAYLOR_SERIES,
    SUM,
    INTEGRAL,
    DERIVATIVE,
    GRAPH,
    DAG,
    STRUCT,
    FUNCTION,
    TABLE,
    MANIFOLD,
    SYMBOLIC,
    EXPRESSION,
    SERIALIZATION
};

// Control/statement forms (syntax)
enum class BitStmtKind {
    IF,
    ELSE_,
    MATCH,
    FOR_,
    WHILE_,
    SWITCH_,
    GOTO_,
    BREAK_,
    CONTINUE_,
    RETURN_,
    ASSIGN
};

// Storage/layout/allocator/backend artifacts
enum class BitStorageKind {
    BITLAYOUT,
    TYPE,
    SCHEMA,
    BITTENSOR,
    BITTENSORMEMORY,
    BITTENSORMEMORYGRAPH
};

// Bit-level operation kinds (verbs)
enum class BitOpKind {
    INT_TO_GRAY_U32,
    GRAY_TO_INT_U32,
    GETBIT_U32,
    SETBIT_U32,
    ADD_U32,
    SUB_U32,
    AND_U32,
    OR_U32,
    XOR_U32,
    MUL_U32,
    DIV_U32,
    MOD_U32,
    NOT_U32,
    SHL_U32,
    SHR_U32
    // ...extend as needed
};

// BitStruct covers logical ops, control forms, and storage/layout tags
enum class BitStructType {
    AND,
    OR,
    XOR,
    NOT,
    INTEGER,
    ELSE_,
    MATCH,
    FOR_,
    WHILE_,
    FLAG,
    SWITCH_,
    GOTO_,
    BREAK_,
    CONTINUE_,
    RETURN_,
    ASSIGN,
    BITLAYOUT,
    TYPE,
    SCHEMA,
    BITTENSOR,
    BITTENSORMEMORY,
    BITTENSORMEMORYGRAPH
};

// BitOps: static bitwise operations
struct BitOps {
    static uint32_t int_to_gray(uint32_t n) { return n ^ (n >> 1); }
    static uint32_t gray_to_int(uint32_t g) {
        uint32_t n = g;
        for (uint32_t shift = 1; shift < 32; shift <<= 1) n ^= (g >> shift);
        return n;
    }
    static uint32_t getbit_u32(uint32_t x, uint32_t bit) { return (x >> bit) & 1u; }
    static uint32_t setbit_u32(uint32_t x, uint32_t bit, uint32_t v) { return (x & ~(1u << bit)) | ((v & 1u) << bit); }
    static uint32_t bit_add(uint32_t x, uint32_t y) { return x + y; }
    static uint32_t bit_sub(uint32_t x, uint32_t y) { return x - y; }
    static uint32_t bit_and(uint32_t x, uint32_t y) { return x & y; }
    static uint32_t bit_or(uint32_t x, uint32_t y) { return x | y; }
    static uint32_t bit_xor(uint32_t x, uint32_t y) { return x ^ y; }
    static uint32_t bit_mul(uint32_t x, uint32_t y) { return x * y; }
    static uint32_t bit_div(uint32_t x, uint32_t y) { if (y == 0) throw std::runtime_error("div by zero"); return x / y; }
    static uint32_t bit_mod(uint32_t x, uint32_t y) { if (y == 0) throw std::runtime_error("mod by zero"); return x % y; }
    static uint32_t bit_not(uint32_t x) { return ~x; }
    static uint32_t bit_shift_left(uint32_t x, uint32_t shift) { return x << shift; }
    static uint32_t bit_shift_right(uint32_t x, uint32_t shift) { return x >> shift; }
};

// BitStruct: base for bit-level types
struct BitStruct {
    BitStructType type;
    BitObjectKind object_kind;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> bits;
    BitStruct(BitStructType s, BitObjectKind t, std::vector<uint32_t> in, std::vector<uint32_t> b = {})
        : type(s), object_kind(t), inputs(std::move(in)), bits(std::move(b)) {}
};

// Integer, Rational, Float, Complex, etc. can be added as needed

} // namespace bitops
} // namespace nodus
