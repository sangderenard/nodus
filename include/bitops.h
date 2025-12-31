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

// Enum for bit-struct types
enum class BitStructType {
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
    SERIALIZATION,
    IF,
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
    std::vector<uint32_t> bits;
    BitStruct(BitStructType t, std::vector<uint32_t> b) : type(t), bits(std::move(b)) {}
};

// Integer, Rational, Float, Complex, etc. can be added as needed

} // namespace bitops
} // namespace nodus
