// GENERATED FILE -- DO NOT EDIT.
// Source:    ops/canonical_ops.json
// Generator: ops/generate_canonical_ops.py
// Verifier:  ops/verify_canonical_ops.py  (checks ct_value against turing's live header)
// Rationale: research/15_the_missing_function_table.md
//
// The canonical operation set shared by turing (Python) and nodus (C++). Both sides
// reduce their own op spellings to these names; anything that agrees here is
// interchangeable regardless of which side produced it.
//
// Canonical IDs are append-only catalog positions. CTensorOp ordinals remain a
// separate, verified backend field for the currently implemented C subset.

#pragma once

#include <cstdint>
#include <string_view>

namespace nodus {
namespace ops {

// Sub-opcode carried by KernelIR's UNARY / BINARY / CMP / CAST instructions,
// which reserve the slot but never defined the index space (research/15).
// Every catalog operation has an ID, including operations not yet available
// in the C dispatcher. IDs are append-only catalog positions.
enum class CanonicalOp : uint16_t {
    ADD = 0,  // CT_OP_ADD
    SUB = 1,  // CT_OP_SUB
    MUL = 2,  // CT_OP_MUL
    TRUEDIV = 3,  // CT_OP_DIV
    POW = 4,  // CT_OP_POW
    MOD = 5,  // CT_OP_MOD
    FLOORDIV = 6,  // CT_OP_FLOORDIV
    SQRT = 7,  // CT_OP_SQRT
    EXP = 8,  // CT_OP_EXP
    LOG = 9,  // CT_OP_LOG
    NEG = 10,  // CT_OP_NEG
    ABS = 11,  // CT_OP_ABS
    ROUND = 12,  // CT_OP_ROUND
    TRUNC = 13,  // CT_OP_TRUNC
    FLOOR = 14,  // CT_OP_FLOOR
    CEIL = 15,  // CT_OP_CEIL
    ISFINITE = 16,  // CT_OP_ISFINITE
    ISNAN = 17,  // CT_OP_ISNAN
    ISINF = 18,  // CT_OP_ISINF
    LOGICAL_NOT = 19,  // CT_OP_LOGICAL_NOT
    LESS = 20,  // CT_OP_LT
    LESS_EQUAL = 21,  // CT_OP_LE
    GREATER = 22,  // CT_OP_GT
    GREATER_EQUAL = 23,  // CT_OP_GE
    EQUAL = 24,  // CT_OP_EQ
    NOT_EQUAL = 25,  // CT_OP_NE
    MAXIMUM = 26,  // CT_OP_MAXIMUM
    MINIMUM = 27,  // CT_OP_MINIMUM
    SIGN = 28,
    INVERT = 29,
    SIN = 30,  // CT_OP_SIN
    COS = 31,  // CT_OP_COS
    TAN = 32,  // CT_OP_TAN
    ASIN = 33,  // CT_OP_ASIN
    ACOS = 34,  // CT_OP_ACOS
    ATAN = 35,  // CT_OP_ATAN
    SINH = 36,  // CT_OP_SINH
    COSH = 37,  // CT_OP_COSH
    TANH = 38,  // CT_OP_TANH
    ASINH = 39,  // CT_OP_ASINH
    ACOSH = 40,  // CT_OP_ACOSH
    ATANH = 41,  // CT_OP_ATANH
    BITAND = 42,
    BITOR = 43,
    BITXOR = 44,
    SHL = 45,
    SHR = 46,
    LOGICAL_AND = 47,
    LOGICAL_OR = 48,
    INT_TRUNC = 49,
    ZEXT = 50,
    SEXT = 51,
    FPTOSI = 52,
    FPTOUI = 53,
    SITOFP = 54,
    UITOFP = 55,
    MATMUL = 56,
    SUM = 57,
    MEAN = 58,
    TOPK = 59,
    LOG_SOFTMAX = 60,
    PAD = 61,
    STACK = 62,
    CAT = 63,
    GATHER = 64,
    ARANGE = 65,
    COUNT = 66,
};

enum class OpClass : uint8_t { Unary, Binary, Compare, Cast, Opaque };

struct OpDesc {
    std::string_view name;         // canonical name -- the one true key
    uint16_t         canonical_id; // append-only shared operation ID
    OpClass          op_class;
    int32_t          ct_value;     // turing CTensorOp ordinal, or -1 (no C target)
    std::string_view ct_op;        // CTensorOp member name, or ""
    uint8_t          arity;
    bool             returns_bool;
    bool             lowerable;    // true => expressible as ONE Tier-0 instruction
    bool             reflectable;  // has a distinct reversed-operand form
    std::string_view kernel_op;    // nodus::spirv::OpCode name, or ""
    std::string_view handler;      // turing Handler member, or ""
    std::string_view c_fn;         // named C function outside the dispatcher, or ""
    std::string_view tier1_class;  // Tier-1 composition family, or "" if Tier-0
};

inline constexpr OpDesc kOps[] = {
    {"add", 0, OpClass::Binary, 0, "CT_OP_ADD", 2, false, true, false, "BINARY", "Add", "", ""},
    {"sub", 1, OpClass::Binary, 1, "CT_OP_SUB", 2, false, true, true, "BINARY", "Sub", "", ""},
    {"mul", 2, OpClass::Binary, 2, "CT_OP_MUL", 2, false, true, false, "BINARY", "Mul", "", ""},
    {"truediv", 3, OpClass::Binary, 3, "CT_OP_DIV", 2, false, true, true, "BINARY", "Div", "", ""},
    {"pow", 4, OpClass::Binary, 4, "CT_OP_POW", 2, false, true, true, "BINARY", "Pow", "", ""},
    {"mod", 5, OpClass::Binary, 5, "CT_OP_MOD", 2, false, true, true, "BINARY", "Mod", "", ""},
    {"floordiv", 6, OpClass::Binary, 6, "CT_OP_FLOORDIV", 2, false, true, true, "BINARY", "", "", ""},
    {"sqrt", 7, OpClass::Unary, 7, "CT_OP_SQRT", 1, false, true, false, "UNARY", "Call", "", ""},
    {"exp", 8, OpClass::Unary, 8, "CT_OP_EXP", 1, false, true, false, "UNARY", "Call", "", ""},
    {"log", 9, OpClass::Unary, 9, "CT_OP_LOG", 1, false, true, false, "UNARY", "Call", "", ""},
    {"neg", 10, OpClass::Unary, 10, "CT_OP_NEG", 1, false, true, false, "UNARY", "Neg", "", ""},
    {"abs", 11, OpClass::Unary, 11, "CT_OP_ABS", 1, false, true, false, "UNARY", "Abs", "", ""},
    {"round", 12, OpClass::Unary, 12, "CT_OP_ROUND", 1, false, true, false, "UNARY", "Call", "", ""},
    {"trunc", 13, OpClass::Unary, 13, "CT_OP_TRUNC", 1, false, true, false, "UNARY", "", "", ""},
    {"floor", 14, OpClass::Unary, 14, "CT_OP_FLOOR", 1, false, true, false, "UNARY", "Call", "", ""},
    {"ceil", 15, OpClass::Unary, 15, "CT_OP_CEIL", 1, false, true, false, "UNARY", "Call", "", ""},
    {"isfinite", 16, OpClass::Unary, 16, "CT_OP_ISFINITE", 1, true, true, false, "UNARY", "", "", ""},
    {"isnan", 17, OpClass::Unary, 17, "CT_OP_ISNAN", 1, true, true, false, "UNARY", "", "", ""},
    {"isinf", 18, OpClass::Unary, 18, "CT_OP_ISINF", 1, true, true, false, "UNARY", "", "", ""},
    {"logical_not", 19, OpClass::Unary, 19, "CT_OP_LOGICAL_NOT", 1, true, true, false, "NOT", "LNot", "", ""},
    {"less", 20, OpClass::Compare, 20, "CT_OP_LT", 2, true, true, false, "CMP", "Lt", "", ""},
    {"less_equal", 21, OpClass::Compare, 21, "CT_OP_LE", 2, true, true, false, "CMP", "Le", "", ""},
    {"greater", 22, OpClass::Compare, 22, "CT_OP_GT", 2, true, true, false, "CMP", "Gt", "", ""},
    {"greater_equal", 23, OpClass::Compare, 23, "CT_OP_GE", 2, true, true, false, "CMP", "Ge", "", ""},
    {"equal", 24, OpClass::Compare, 24, "CT_OP_EQ", 2, true, true, false, "CMP", "Eq", "", ""},
    {"not_equal", 25, OpClass::Compare, 25, "CT_OP_NE", 2, true, true, false, "CMP", "Ne", "", ""},
    {"maximum", 26, OpClass::Binary, 26, "CT_OP_MAXIMUM", 2, false, true, false, "BINARY", "", "", ""},
    {"minimum", 27, OpClass::Binary, 27, "CT_OP_MINIMUM", 2, false, true, false, "BINARY", "", "", ""},
    {"sign", 28, OpClass::Unary, -1, "", 1, false, true, false, "UNARY", "", "", ""},
    {"invert", 29, OpClass::Unary, -1, "", 1, false, true, false, "NOT", "Not", "", ""},
    {"sin", 30, OpClass::Unary, 29, "CT_OP_SIN", 1, false, true, false, "UNARY", "Call", "", ""},
    {"cos", 31, OpClass::Unary, 30, "CT_OP_COS", 1, false, true, false, "UNARY", "Call", "", ""},
    {"tan", 32, OpClass::Unary, 31, "CT_OP_TAN", 1, false, true, false, "UNARY", "Call", "", ""},
    {"asin", 33, OpClass::Unary, 32, "CT_OP_ASIN", 1, false, true, false, "UNARY", "Call", "", ""},
    {"acos", 34, OpClass::Unary, 33, "CT_OP_ACOS", 1, false, true, false, "UNARY", "Call", "", ""},
    {"atan", 35, OpClass::Unary, 34, "CT_OP_ATAN", 1, false, true, false, "UNARY", "Call", "", ""},
    {"sinh", 36, OpClass::Unary, 35, "CT_OP_SINH", 1, false, true, false, "UNARY", "Call", "", ""},
    {"cosh", 37, OpClass::Unary, 36, "CT_OP_COSH", 1, false, true, false, "UNARY", "Call", "", ""},
    {"tanh", 38, OpClass::Unary, 28, "CT_OP_TANH", 1, false, true, false, "UNARY", "Call", "", ""},
    {"asinh", 39, OpClass::Unary, 37, "CT_OP_ASINH", 1, false, true, false, "UNARY", "Call", "", ""},
    {"acosh", 40, OpClass::Unary, 38, "CT_OP_ACOSH", 1, false, true, false, "UNARY", "Call", "", ""},
    {"atanh", 41, OpClass::Unary, 39, "CT_OP_ATANH", 1, false, true, false, "UNARY", "Call", "", ""},
    {"bitand", 42, OpClass::Binary, -1, "", 2, false, true, false, "AND", "And", "", ""},
    {"bitor", 43, OpClass::Binary, -1, "", 2, false, true, false, "OR", "Or", "", ""},
    {"bitxor", 44, OpClass::Binary, -1, "", 2, false, true, false, "XOR", "Xor", "", ""},
    {"shl", 45, OpClass::Binary, -1, "", 2, false, true, true, "BINARY", "Shl", "", ""},
    {"shr", 46, OpClass::Binary, -1, "", 2, false, true, true, "BINARY", "Shr", "", ""},
    {"logical_and", 47, OpClass::Binary, -1, "", 2, true, true, false, "AND", "LAnd", "", ""},
    {"logical_or", 48, OpClass::Binary, -1, "", 2, true, true, false, "OR", "LOr", "", ""},
    {"int_trunc", 49, OpClass::Cast, -1, "", 1, false, true, false, "CAST", "Trunc", "", ""},
    {"zext", 50, OpClass::Cast, -1, "", 1, false, true, false, "CAST", "ZExt", "", ""},
    {"sext", 51, OpClass::Cast, -1, "", 1, false, true, false, "CAST", "SExt", "", ""},
    {"fptosi", 52, OpClass::Cast, -1, "", 1, false, true, false, "CAST", "FpToSi", "cast_double_to_int_values", ""},
    {"fptoui", 53, OpClass::Cast, -1, "", 1, false, true, false, "CAST", "FpToUi", "", ""},
    {"sitofp", 54, OpClass::Cast, -1, "", 1, false, true, false, "CAST", "SiToFp", "cast_double_to_float_values", ""},
    {"uitofp", 55, OpClass::Cast, -1, "", 1, false, true, false, "CAST", "UiToFp", "", ""},
    {"matmul", 56, OpClass::Opaque, -1, "", 2, false, false, true, "", "Call", "matmul_double", "contract"},
    {"sum", 57, OpClass::Opaque, -1, "", 1, false, false, false, "", "Call", "sum_double", "reduce"},
    {"mean", 58, OpClass::Opaque, -1, "", 1, false, false, false, "", "Call", "mean_dim", "reduce"},
    {"topk", 59, OpClass::Opaque, -1, "", 1, false, false, false, "", "Call", "topk_double", "order"},
    {"log_softmax", 60, OpClass::Opaque, -1, "", 1, false, false, false, "", "Call", "log_softmax_dim", "reduce"},
    {"pad", 61, OpClass::Opaque, -1, "", 1, false, false, false, "", "Call", "pad_double_nd", "remap"},
    {"stack", 62, OpClass::Opaque, -1, "", 2, false, false, false, "", "Call", "stack_double", "remap"},
    {"cat", 63, OpClass::Opaque, -1, "", 2, false, false, false, "", "Call", "cat_double", "remap"},
    {"gather", 64, OpClass::Opaque, -1, "", 2, false, false, false, "", "Load", "gather_pairs_2d", "remap"},
    {"arange", 65, OpClass::Opaque, -1, "", 0, false, false, false, "", "", "create_arange", "generate"},
};

inline constexpr size_t kOpCount = 66;

// Lookups return nullptr when unknown. A hard null beats a silent zero: research/06
// and research/12 document what silently-defaulting lookups have already cost here.
inline constexpr const OpDesc* find_op(std::string_view name) {
    for (size_t i = 0; i < kOpCount; ++i) {
        if (kOps[i].name == name) return &kOps[i];
    }
    return nullptr;
}

inline constexpr const OpDesc* find_op(CanonicalOp op) {
    for (size_t i = 0; i < kOpCount; ++i) {
        if (kOps[i].canonical_id == static_cast<uint16_t>(op)) return &kOps[i];
    }
    return nullptr;
}

// Resolve a turing Handler member. Handler::Call fans out across every elementary
// function, so this returns the first match only and is unsuitable for Call --
// prefer find_op(name). Present to make the correspondence expressible, not to
// dispatch on.
inline constexpr const OpDesc* find_op_by_handler(std::string_view handler) {
    for (size_t i = 0; i < kOpCount; ++i) {
        if (!kOps[i].handler.empty() && kOps[i].handler == handler) return &kOps[i];
    }
    return nullptr;
}

} // namespace ops
} // namespace nodus
