// kernel_spirv.cpp
// Direct SPIR-V assembly from KernelIR. See include/kernel_spirv_assembler.h
// for the contract; this file owns the words.
//
// Layout follows the SPIR-V 1.3 logical module order: capabilities, extended
// instruction imports, memory model, entry points, execution modes, debug
// names, decorations, types/constants/global variables, function bodies.
// Types and constants are module-scoped and pooled -- two uses of u32 or of
// the literal 1u share one id (the same discipline turing's
// ssa_spirv_backend.py enforces with its _ModuleBuilder).

#include "../kernel_isa.h"
#include "kernel_spirv_assembler.h"
#include "canonical_ops.h"
#include "translation_matrix.h"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace nodus {
namespace kernels {

namespace {

using spirv::KernelIR;
using spirv::Operand;
using spirv::ScalarType;
using spirv::ValueRef;
using ops::CanonicalOp;

using u32 = std::uint32_t;

// --- SPIR-V opcode numbers used here (core spec, not exhaustive) -----------
enum : u32 {
    kOpExtInstImport = 11, kOpExtInst = 12, kOpMemoryModel = 14,
    kOpEntryPoint = 15, kOpExecutionMode = 16, kOpCapability = 17,
    kOpTypeVoid = 19, kOpTypeBool = 20, kOpTypeInt = 21, kOpTypeFloat = 22,
    kOpTypeVector = 23, kOpTypeRuntimeArray = 29, kOpTypeStruct = 30,
    kOpTypePointer = 32, kOpTypeFunction = 33,
    kOpConstantTrue = 41, kOpConstantFalse = 42, kOpConstant = 43,
    kOpFunction = 54, kOpFunctionEnd = 56, kOpVariable = 59,
    kOpLoad = 61, kOpStore = 62, kOpAccessChain = 65,
    kOpDecorate = 71, kOpMemberDecorate = 72, kOpCompositeExtract = 81,
    kOpConvertFToU = 109, kOpConvertFToS = 110,
    kOpConvertSToF = 111, kOpConvertUToF = 112, kOpBitcast = 124,
    kOpSNegate = 126, kOpFNegate = 127,
    kOpIAdd = 128, kOpFAdd = 129, kOpISub = 130, kOpFSub = 131,
    kOpIMul = 132, kOpFMul = 133, kOpUDiv = 134, kOpSDiv = 135,
    kOpFDiv = 136, kOpUMod = 137, kOpSMod = 139, kOpFRem = 140,
    kOpLogicalOr = 166, kOpLogicalAnd = 167, kOpLogicalNot = 168,
    kOpSelect = 169,
    kOpIEqual = 170, kOpINotEqual = 171,
    kOpUGreaterThan = 172, kOpSGreaterThan = 173,
    kOpUGreaterThanEqual = 174, kOpSGreaterThanEqual = 175,
    kOpULessThan = 176, kOpSLessThan = 177,
    kOpULessThanEqual = 178, kOpSLessThanEqual = 179,
    kOpFOrdEqual = 180, kOpFOrdNotEqual = 182,
    kOpFOrdLessThan = 184, kOpFOrdGreaterThan = 186,
    kOpFOrdLessThanEqual = 188, kOpFOrdGreaterThanEqual = 190,
    kOpShiftRightLogical = 194, kOpShiftRightArithmetic = 195,
    kOpShiftLeftLogical = 196,
    kOpBitwiseOr = 197, kOpBitwiseXor = 198, kOpBitwiseAnd = 199,
    kOpNot = 200,
    kOpLabel = 248, kOpBranch = 249, kOpBranchConditional = 250,
    kOpSelectionMerge = 247, kOpReturn = 253,
};

// GLSL.std.450 extended instruction numbers used here.
enum : u32 {
    kGlslRound = 1, kGlslTrunc = 3, kGlslFAbs = 4, kGlslSAbs = 5,
    kGlslFloor = 8, kGlslCeil = 9,
    kGlslSin = 13, kGlslCos = 14, kGlslTan = 15,
    kGlslAsin = 16, kGlslAcos = 17, kGlslAtan = 18,
    kGlslSinh = 19, kGlslCosh = 20, kGlslTanh = 21,
    kGlslAsinh = 22, kGlslAcosh = 23, kGlslAtanh = 24,
    kGlslPow = 26, kGlslExp = 27, kGlslLog = 28, kGlslSqrt = 31,
    kGlslFMin = 37, kGlslUMin = 38, kGlslSMin = 39,
    kGlslFMax = 40, kGlslUMax = 41, kGlslSMax = 42,
};

// Decorations / enums.
enum : u32 {
    kDecoBlock = 2, kDecoArrayStride = 6, kDecoBuiltIn = 11,
    kDecoNonWritable = 24, kDecoBinding = 33, kDecoDescriptorSet = 34,
    kDecoOffset = 35,
    kBuiltInGlobalInvocationId = 28,
    kStorageInput = 1, kStorageStorageBuffer = 12,
    kExecutionModelGLCompute = 5, kExecutionModeLocalSize = 17,
    kCapabilityShader = 1,
    kMemoryModelLogical = 0, kMemoryModelGLSL450 = 1,
};

std::string scalar_name(ScalarType s) {
    switch (s) {
        case ScalarType::I32: return "i32";
        case ScalarType::U32: return "u32";
        case ScalarType::F32: return "f32";
    }
    return "?";
}

// One instruction stream (words) with the standard word_count<<16|opcode head.
struct Stream {
    std::vector<u32> words;

    void op(u32 opcode, std::initializer_list<u32> operands) {
        words.push_back((u32(operands.size() + 1) << 16) | opcode);
        words.insert(words.end(), operands.begin(), operands.end());
    }
    void op(u32 opcode, const std::vector<u32>& operands) {
        words.push_back((u32(operands.size() + 1) << 16) | opcode);
        words.insert(words.end(), operands.begin(), operands.end());
    }
};

// Pack a UTF-8 literal into null-terminated little-endian words.
std::vector<u32> packed(const std::string& text) {
    std::vector<u32> out;
    u32 word = 0;
    int shift = 0;
    for (unsigned char c : text) {
        word |= u32(c) << shift;
        shift += 8;
        if (shift == 32) { out.push_back(word); word = 0; shift = 0; }
    }
    out.push_back(word); // includes the terminating NUL (word may be 0)
    return out;
}

class ModuleAssembler {
public:
    explicit ModuleAssembler(const KernelIR& kernel) : k_(kernel) {}

    SpirvAssembly run();

private:
    // --- id/type/constant pools (module scope) ---
    u32 fresh() { return next_id_++; }

    u32 type_key(const std::string& key, u32 opcode, std::vector<u32> tail) {
        auto found = types_.find(key);
        if (found != types_.end()) return found->second;
        const u32 id = fresh();
        std::vector<u32> operands{id};
        operands.insert(operands.end(), tail.begin(), tail.end());
        types_stream_.op(opcode, operands);
        types_.emplace(key, id);
        return id;
    }

    u32 t_void() { return type_key("void", kOpTypeVoid, {}); }
    u32 t_bool() { return type_key("bool", kOpTypeBool, {}); }
    u32 t_u32()  { return type_key("u32", kOpTypeInt, {32u, 0u}); }
    u32 t_i32()  { return type_key("i32", kOpTypeInt, {32u, 1u}); }
    u32 t_f32()  { return type_key("f32", kOpTypeFloat, {32u}); }
    u32 t_scalar(ScalarType s) {
        switch (s) {
            case ScalarType::I32: return t_i32();
            case ScalarType::U32: return t_u32();
            case ScalarType::F32: return t_f32();
        }
        throw std::runtime_error("unreachable scalar type");
    }
    u32 t_v3u32() { return type_key("v3u32", kOpTypeVector, {t_u32(), 3u}); }
    u32 t_ptr(const std::string& tag, u32 storage, u32 pointee) {
        return type_key("ptr:" + tag, kOpTypePointer, {storage, pointee});
    }
    u32 t_fn_void() { return type_key("fn:void", kOpTypeFunction, {t_void()}); }

    u32 const_scalar(ScalarType s, u32 bits) {
        const std::string key = scalar_name(s) + ":" + std::to_string(bits);
        auto found = constants_.find(key);
        if (found != constants_.end()) return found->second;
        const u32 id = fresh();
        types_stream_.op(kOpConstant, {t_scalar(s), id, bits});
        constants_.emplace(key, id);
        return id;
    }
    u32 const_u32(u32 value) { return const_scalar(ScalarType::U32, value); }
    u32 const_i32(std::int32_t value) {
        return const_scalar(ScalarType::I32, static_cast<u32>(value));
    }
    u32 const_f32(float value) {
        u32 bits;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        return const_scalar(ScalarType::F32, bits);
    }

    // --- shortfall bookkeeping (fail closed, fail visible) ---
    void shortfall(const std::string& reason) { shortfalls_.push_back(reason); }

    // --- value handling ---
    ScalarType value_scalar(u32 value_id) const {
        return k_.values.at(value_id).type.scalar;
    }

    u32 resolve(const Operand& operand, ScalarType context, bool& ok) {
        ok = true;
        if (std::holds_alternative<ValueRef>(operand.v)) {
            const u32 id = std::get<ValueRef>(operand.v).id;
            auto found = value_results_.find(id);
            if (found == value_results_.end()) {
                ok = false;
                shortfall(
                    "value %" + std::to_string(id) +
                    " is used before any instruction or buffer load defines it");
                return 0;
            }
            return found->second;
        }
        if (std::holds_alternative<std::int32_t>(operand.v)) {
            // An i32 immediate feeding a u32/f32 context keeps the context's
            // type: the IR's Operand::i is a literal spelling, not a cast.
            const std::int32_t raw = std::get<std::int32_t>(operand.v);
            if (context == ScalarType::F32) return const_f32(float(raw));
            return const_scalar(context, static_cast<u32>(raw));
        }
        if (std::holds_alternative<u32>(operand.v)) {
            const u32 raw = std::get<u32>(operand.v);
            if (context == ScalarType::F32) return const_f32(float(raw));
            return const_scalar(context, raw);
        }
        const float raw = std::get<float>(operand.v);
        if (context != ScalarType::F32) {
            ok = false;
            shortfall("float immediate used in integer context");
            return 0;
        }
        return const_f32(raw);
    }

    // Emit a bool from a scalar condition value (nonzero => true).
    u32 to_bool(u32 value_id, ScalarType s) {
        const u32 result = fresh();
        if (s == ScalarType::F32) {
            body_.op(kOpFOrdNotEqual, {t_bool(), result, value_id, const_f32(0.0f)});
        } else {
            body_.op(kOpINotEqual,
                     {t_bool(), result, value_id, const_scalar(s, 0u)});
        }
        return result;
    }

    // Bool -> the instruction's declared scalar type (1/0).
    u32 from_bool(u32 bool_id, ScalarType s) {
        const u32 result = fresh();
        const u32 one = s == ScalarType::F32 ? const_f32(1.0f) : const_scalar(s, 1u);
        const u32 zero = s == ScalarType::F32 ? const_f32(0.0f) : const_scalar(s, 0u);
        body_.op(kOpSelect, {t_scalar(s), result, bool_id, one, zero});
        return result;
    }

    u32 ext_inst(u32 inst, ScalarType s, std::vector<u32> args) {
        const u32 result = fresh();
        std::vector<u32> operands{t_scalar(s), result, glsl_import_, inst};
        operands.insert(operands.end(), args.begin(), args.end());
        body_.op(kOpExtInst, operands);
        return result;
    }

    void lower_instruction(const spirv::Instruction& ins);
    void lower_logical(const spirv::Instruction& ins, u32 bit_opcode);
    void lower_binary(const spirv::Instruction& ins);
    void lower_unary(const spirv::Instruction& ins);
    void lower_cmp(const spirv::Instruction& ins);
    void lower_select(const spirv::Instruction& ins);
    void lower_cast(const spirv::Instruction& ins);
    void lower_addr(const spirv::Instruction& ins);
    void lower_load(const spirv::Instruction& ins);
    void lower_store(const spirv::Instruction& ins);

    bool single_output(const spirv::Instruction& ins, const char* what) {
        if (ins.outputs.size() == 1) return true;
        shortfall(std::string(what) + " requires exactly one output value");
        return false;
    }

    const KernelIR& k_;
    u32 next_id_ = 1;
    u32 glsl_import_ = 0;

    Stream types_stream_;   // types, constants, global variables
    Stream decorations_;
    Stream names_;
    Stream body_;           // function body instructions (inside the guard)

    std::unordered_map<std::string, u32> types_;
    std::unordered_map<std::string, u32> constants_;
    std::unordered_map<u32, u32> value_results_;   // KernelIR value id -> SPIR-V id
    std::unordered_map<u32, u32> value_pointers_;  // ADDR results -> pointer id
    std::unordered_map<u32, ScalarType> pointer_scalars_;
    std::unordered_map<u32, u32> buffer_variables_; // value id -> OpVariable id
    std::unordered_map<u32, ScalarType> buffer_scalars_;
    std::vector<std::string> shortfalls_;
};

void ModuleAssembler::lower_logical(const spirv::Instruction& ins, u32 bit_opcode) {
    if (!single_output(ins, "logical op")) return;
    const u32 out = ins.outputs[0].id;
    const ScalarType s = value_scalar(out);
    if (s == ScalarType::F32) {
        shortfall("bitwise op on f32 value %" + std::to_string(out));
        return;
    }
    bool ok = true;
    std::vector<u32> args;
    for (const Operand& operand : ins.inputs) {
        args.push_back(resolve(operand, s, ok));
        if (!ok) return;
    }
    const u32 result = fresh();
    if (bit_opcode == kOpNot) {
        if (args.size() != 1) { shortfall("NOT takes one input"); return; }
        body_.op(kOpNot, {t_scalar(s), result, args[0]});
    } else {
        if (args.size() != 2) { shortfall("bitwise op takes two inputs"); return; }
        body_.op(bit_opcode, {t_scalar(s), result, args[0], args[1]});
    }
    value_results_[out] = result;
}

void ModuleAssembler::lower_binary(const spirv::Instruction& ins) {
    if (!single_output(ins, "BINARY")) return;
    const u32 out = ins.outputs[0].id;
    const ScalarType s = value_scalar(out);
    if (ins.inputs.size() != 2) { shortfall("BINARY takes two inputs"); return; }
    bool ok = true;
    const u32 a = resolve(ins.inputs[0], s, ok);
    if (!ok) return;
    const u32 b = resolve(ins.inputs[1], s, ok);
    if (!ok) return;

    const bool f = s == ScalarType::F32;
    const bool sign = s == ScalarType::I32;
    const auto sub = static_cast<CanonicalOp>(ins.sub_op);
    u32 opcode = 0;
    switch (sub) {
        case CanonicalOp::ADD: opcode = f ? kOpFAdd : kOpIAdd; break;
        case CanonicalOp::SUB: opcode = f ? kOpFSub : kOpISub; break;
        case CanonicalOp::MUL: opcode = f ? kOpFMul : kOpIMul; break;
        case CanonicalOp::TRUEDIV:
            opcode = f ? kOpFDiv : (sign ? kOpSDiv : kOpUDiv); break;
        case CanonicalOp::MOD:
            opcode = f ? kOpFRem : (sign ? kOpSMod : kOpUMod); break;
        case CanonicalOp::BITAND: opcode = kOpBitwiseAnd; break;
        case CanonicalOp::BITOR: opcode = kOpBitwiseOr; break;
        case CanonicalOp::BITXOR: opcode = kOpBitwiseXor; break;
        case CanonicalOp::SHL: opcode = kOpShiftLeftLogical; break;
        case CanonicalOp::SHR:
            opcode = sign ? kOpShiftRightArithmetic : kOpShiftRightLogical; break;
        case CanonicalOp::POW:
            if (!f) { shortfall("POW is lowered for f32 only"); return; }
            value_results_[out] = ext_inst(kGlslPow, s, {a, b});
            return;
        case CanonicalOp::MAXIMUM:
            value_results_[out] = ext_inst(
                f ? kGlslFMax : (sign ? kGlslSMax : kGlslUMax), s, {a, b});
            return;
        case CanonicalOp::MINIMUM:
            value_results_[out] = ext_inst(
                f ? kGlslFMin : (sign ? kGlslSMin : kGlslUMin), s, {a, b});
            return;
        default:
            shortfall(
                "BINARY sub_op " + std::to_string(ins.sub_op) +
                " has no direct SPIR-V lowering yet");
            return;
    }
    if ((opcode == kOpBitwiseAnd || opcode == kOpBitwiseOr ||
         opcode == kOpBitwiseXor || opcode == kOpShiftLeftLogical ||
         opcode == kOpShiftRightLogical) && f) {
        shortfall("bitwise BINARY sub_op on f32 value %" + std::to_string(out));
        return;
    }
    const u32 result = fresh();
    body_.op(opcode, {t_scalar(s), result, a, b});
    value_results_[out] = result;
}

void ModuleAssembler::lower_unary(const spirv::Instruction& ins) {
    if (!single_output(ins, "UNARY")) return;
    const u32 out = ins.outputs[0].id;
    const ScalarType s = value_scalar(out);
    if (ins.inputs.size() != 1) { shortfall("UNARY takes one input"); return; }
    bool ok = true;
    const u32 x = resolve(ins.inputs[0], s, ok);
    if (!ok) return;

    const bool f = s == ScalarType::F32;
    const auto sub = static_cast<CanonicalOp>(ins.sub_op);
    switch (sub) {
        case CanonicalOp::NEG: {
            const u32 result = fresh();
            body_.op(f ? kOpFNegate : kOpSNegate, {t_scalar(s), result, x});
            value_results_[out] = result;
            return;
        }
        case CanonicalOp::ABS:
            value_results_[out] = ext_inst(f ? kGlslFAbs : kGlslSAbs, s, {x});
            return;
        case CanonicalOp::INVERT: {
            if (f) { shortfall("INVERT on f32 value %" + std::to_string(out)); return; }
            const u32 result = fresh();
            body_.op(kOpNot, {t_scalar(s), result, x});
            value_results_[out] = result;
            return;
        }
        default: break;
    }
    if (!f) {
        shortfall("UNARY sub_op " + std::to_string(ins.sub_op) +
                  " is lowered for f32 only");
        return;
    }
    u32 inst = 0;
    switch (sub) {
        case CanonicalOp::SQRT: inst = kGlslSqrt; break;
        case CanonicalOp::EXP: inst = kGlslExp; break;
        case CanonicalOp::LOG: inst = kGlslLog; break;
        case CanonicalOp::SIN: inst = kGlslSin; break;
        case CanonicalOp::COS: inst = kGlslCos; break;
        case CanonicalOp::TAN: inst = kGlslTan; break;
        case CanonicalOp::ASIN: inst = kGlslAsin; break;
        case CanonicalOp::ACOS: inst = kGlslAcos; break;
        case CanonicalOp::ATAN: inst = kGlslAtan; break;
        case CanonicalOp::SINH: inst = kGlslSinh; break;
        case CanonicalOp::COSH: inst = kGlslCosh; break;
        case CanonicalOp::TANH: inst = kGlslTanh; break;
        case CanonicalOp::ASINH: inst = kGlslAsinh; break;
        case CanonicalOp::ACOSH: inst = kGlslAcosh; break;
        case CanonicalOp::ATANH: inst = kGlslAtanh; break;
        case CanonicalOp::FLOOR: inst = kGlslFloor; break;
        case CanonicalOp::CEIL: inst = kGlslCeil; break;
        case CanonicalOp::TRUNC: inst = kGlslTrunc; break;
        case CanonicalOp::ROUND: inst = kGlslRound; break;
        default:
            shortfall("UNARY sub_op " + std::to_string(ins.sub_op) +
                      " has no direct SPIR-V lowering yet");
            return;
    }
    value_results_[out] = ext_inst(inst, s, {x});
}

void ModuleAssembler::lower_cmp(const spirv::Instruction& ins) {
    if (!single_output(ins, "CMP")) return;
    const u32 out = ins.outputs[0].id;
    if (ins.inputs.size() != 2) { shortfall("CMP takes two inputs"); return; }
    // Operand type comes from the operands, not the (integer) result value.
    ScalarType operand_type = value_scalar(out);
    for (const Operand& operand : ins.inputs) {
        if (std::holds_alternative<ValueRef>(operand.v)) {
            operand_type = value_scalar(std::get<ValueRef>(operand.v).id);
            break;
        }
    }
    bool ok = true;
    const u32 a = resolve(ins.inputs[0], operand_type, ok);
    if (!ok) return;
    const u32 b = resolve(ins.inputs[1], operand_type, ok);
    if (!ok) return;

    const bool f = operand_type == ScalarType::F32;
    const bool sign = operand_type == ScalarType::I32;
    u32 opcode = 0;
    switch (static_cast<CanonicalOp>(ins.sub_op)) {
        case CanonicalOp::EQUAL: opcode = f ? kOpFOrdEqual : kOpIEqual; break;
        case CanonicalOp::NOT_EQUAL:
            opcode = f ? kOpFOrdNotEqual : kOpINotEqual; break;
        case CanonicalOp::LESS:
            opcode = f ? kOpFOrdLessThan : (sign ? kOpSLessThan : kOpULessThan);
            break;
        case CanonicalOp::LESS_EQUAL:
            opcode = f ? kOpFOrdLessThanEqual
                       : (sign ? kOpSLessThanEqual : kOpULessThanEqual);
            break;
        case CanonicalOp::GREATER:
            opcode = f ? kOpFOrdGreaterThan
                       : (sign ? kOpSGreaterThan : kOpUGreaterThan);
            break;
        case CanonicalOp::GREATER_EQUAL:
            opcode = f ? kOpFOrdGreaterThanEqual
                       : (sign ? kOpSGreaterThanEqual : kOpUGreaterThanEqual);
            break;
        default:
            shortfall("CMP sub_op " + std::to_string(ins.sub_op) +
                      " has no direct SPIR-V lowering yet");
            return;
    }
    const u32 as_bool = fresh();
    body_.op(opcode, {t_bool(), as_bool, a, b});
    value_results_[out] = from_bool(as_bool, value_scalar(out));
}

void ModuleAssembler::lower_select(const spirv::Instruction& ins) {
    if (!single_output(ins, "SELECT")) return;
    const u32 out = ins.outputs[0].id;
    const ScalarType s = value_scalar(out);
    if (ins.inputs.size() != 3) { shortfall("SELECT takes cond, t, f"); return; }
    ScalarType cond_type = ScalarType::U32;
    if (std::holds_alternative<ValueRef>(ins.inputs[0].v)) {
        cond_type = value_scalar(std::get<ValueRef>(ins.inputs[0].v).id);
    }
    bool ok = true;
    const u32 cond = resolve(ins.inputs[0], cond_type, ok);
    if (!ok) return;
    const u32 t = resolve(ins.inputs[1], s, ok);
    if (!ok) return;
    const u32 fv = resolve(ins.inputs[2], s, ok);
    if (!ok) return;
    const u32 result = fresh();
    body_.op(kOpSelect, {t_scalar(s), result, to_bool(cond, cond_type), t, fv});
    value_results_[out] = result;
}

void ModuleAssembler::lower_cast(const spirv::Instruction& ins) {
    if (!single_output(ins, "CAST")) return;
    const u32 out = ins.outputs[0].id;
    const ScalarType dst = value_scalar(out);
    if (ins.inputs.size() != 1) { shortfall("CAST takes one input"); return; }
    if (!std::holds_alternative<ValueRef>(ins.inputs[0].v)) {
        shortfall("CAST of an immediate; spell the constant in the target type");
        return;
    }
    const u32 src_value = std::get<ValueRef>(ins.inputs[0].v).id;
    const ScalarType src = value_scalar(src_value);
    bool ok = true;
    const u32 x = resolve(ins.inputs[0], src, ok);
    if (!ok) return;

    u32 opcode = 0;
    switch (static_cast<CanonicalOp>(ins.sub_op)) {
        case CanonicalOp::FPTOSI: opcode = kOpConvertFToS; break;
        case CanonicalOp::FPTOUI: opcode = kOpConvertFToU; break;
        case CanonicalOp::SITOFP: opcode = kOpConvertSToF; break;
        case CanonicalOp::UITOFP: opcode = kOpConvertUToF; break;
        case CanonicalOp::ZEXT:
        case CanonicalOp::SEXT:
        case CanonicalOp::INT_TRUNC:
            // All integer types here are 32-bit, so width casts degenerate to
            // a signedness reinterpretation.
            opcode = kOpBitcast;
            break;
        default:
            shortfall("CAST sub_op " + std::to_string(ins.sub_op) +
                      " has no direct SPIR-V lowering yet");
            return;
    }
    if (opcode == kOpBitcast && src == dst) {
        value_results_[out] = x; // identity
        return;
    }
    const u32 result = fresh();
    body_.op(opcode, {t_scalar(dst), result, x});
    value_results_[out] = result;
}

void ModuleAssembler::lower_addr(const spirv::Instruction& ins) {
    if (!single_output(ins, "ADDR")) return;
    if (ins.inputs.size() < 2) {
        shortfall("ADDR takes a buffer value and at least one index");
        return;
    }
    if (!std::holds_alternative<ValueRef>(ins.inputs[0].v)) {
        shortfall("ADDR base must be a buffer value reference");
        return;
    }
    const u32 base_value = std::get<ValueRef>(ins.inputs[0].v).id;
    auto buffer = buffer_variables_.find(base_value);
    if (buffer == buffer_variables_.end()) {
        shortfall("ADDR base value %" + std::to_string(base_value) +
                  " is not a declared buffer");
        return;
    }
    const ScalarType s = buffer_scalars_.at(base_value);
    bool ok = true;
    std::vector<u32> operands{
        t_ptr("sb-elem-" + scalar_name(s), kStorageStorageBuffer, t_scalar(s)),
        0, // result placeholder, patched below
        buffer->second,
        const_u32(0u), // step through the Block struct's single member
    };
    for (std::size_t index = 1; index < ins.inputs.size(); ++index) {
        operands.push_back(resolve(ins.inputs[index], ScalarType::U32, ok));
        if (!ok) return;
    }
    const u32 result = fresh();
    operands[1] = result;
    body_.op(kOpAccessChain, operands);
    value_pointers_[ins.outputs[0].id] = result;
    pointer_scalars_[ins.outputs[0].id] = s;
}

void ModuleAssembler::lower_load(const spirv::Instruction& ins) {
    if (!single_output(ins, "LOAD")) return;
    if (ins.inputs.size() != 1 ||
        !std::holds_alternative<ValueRef>(ins.inputs[0].v)) {
        shortfall("LOAD takes one ADDR result");
        return;
    }
    const u32 addr_value = std::get<ValueRef>(ins.inputs[0].v).id;
    auto pointer = value_pointers_.find(addr_value);
    if (pointer == value_pointers_.end()) {
        shortfall("LOAD address %" + std::to_string(addr_value) +
                  " is not an ADDR result");
        return;
    }
    const ScalarType s = pointer_scalars_.at(addr_value);
    const u32 result = fresh();
    body_.op(kOpLoad, {t_scalar(s), result, pointer->second});
    value_results_[ins.outputs[0].id] = result;
}

void ModuleAssembler::lower_store(const spirv::Instruction& ins) {
    if (ins.inputs.size() != 2 ||
        !std::holds_alternative<ValueRef>(ins.inputs[0].v)) {
        shortfall("STORE takes an ADDR result and a value");
        return;
    }
    const u32 addr_value = std::get<ValueRef>(ins.inputs[0].v).id;
    auto pointer = value_pointers_.find(addr_value);
    if (pointer == value_pointers_.end()) {
        shortfall("STORE address %" + std::to_string(addr_value) +
                  " is not an ADDR result");
        return;
    }
    const ScalarType s = pointer_scalars_.at(addr_value);
    bool ok = true;
    const u32 value = resolve(ins.inputs[1], s, ok);
    if (!ok) return;
    body_.op(kOpStore, {pointer->second, value});
}

void ModuleAssembler::lower_instruction(const spirv::Instruction& ins) {
    using spirv::OpCode;
    switch (ins.op) {
        // The kernel frame comes from KernelIR fields, so these are metadata.
        case OpCode::MODULE_BEGIN:
        case OpCode::KERNEL_ENTRY:
            return;
        case OpCode::AND: lower_logical(ins, kOpBitwiseAnd); return;
        case OpCode::OR: lower_logical(ins, kOpBitwiseOr); return;
        case OpCode::XOR: lower_logical(ins, kOpBitwiseXor); return;
        case OpCode::NOT: lower_logical(ins, kOpNot); return;
        case OpCode::BINARY: lower_binary(ins); return;
        case OpCode::UNARY: lower_unary(ins); return;
        case OpCode::CMP: lower_cmp(ins); return;
        case OpCode::SELECT: lower_select(ins); return;
        case OpCode::CAST: lower_cast(ins); return;
        case OpCode::ADDR: lower_addr(ins); return;
        case OpCode::LOAD: lower_load(ins); return;
        case OpCode::STORE: lower_store(ins); return;
        default:
            shortfall("OpCode " + std::to_string(static_cast<int>(ins.op)) +
                      " has no direct SPIR-V lowering yet");
            return;
    }
}

SpirvAssembly ModuleAssembler::run() {
    SpirvAssembly out;

    glsl_import_ = fresh();

    // Buffers: one Block-decorated struct of a runtime array per entry,
    // descriptor set 0, binding = position in buffer_value_ids.
    u32 binding = 0;
    for (const u32 value_id : k_.buffer_value_ids) {
        if (value_id >= k_.values.size()) {
            shortfall("buffer value id %" + std::to_string(value_id) +
                      " is outside the value table");
            ++binding;
            continue;
        }
        const auto& def = k_.values[value_id];
        if (!def.type.is_buffer) {
            shortfall("buffer value %" + std::to_string(value_id) +
                      " is not declared is_buffer");
            ++binding;
            continue;
        }
        const ScalarType s = def.type.scalar;
        const std::string tag = scalar_name(s);
        const u32 array = type_key("rta:" + tag, kOpTypeRuntimeArray,
                                   {t_scalar(s)});
        // Decorations must be unique per decorated id; the pooled struct is
        // decorated once when first created.
        const std::string struct_key = "sb-struct:" + tag;
        const bool new_struct = types_.find(struct_key) == types_.end();
        const u32 wrapper = type_key(struct_key, kOpTypeStruct, {array});
        if (new_struct) {
            decorations_.op(kOpDecorate, {array, kDecoArrayStride, 4u});
            decorations_.op(kOpDecorate, {wrapper, kDecoBlock});
            decorations_.op(kOpMemberDecorate, {wrapper, 0u, kDecoOffset, 0u});
        }
        const u32 pointer =
            t_ptr("sb-struct-" + tag, kStorageStorageBuffer, wrapper);
        const u32 variable = fresh();
        types_stream_.op(kOpVariable, {pointer, variable, kStorageStorageBuffer});
        decorations_.op(kOpDecorate, {variable, kDecoDescriptorSet, 0u});
        decorations_.op(kOpDecorate, {variable, kDecoBinding, binding});
        if (def.type.is_readonly) {
            decorations_.op(kOpDecorate, {variable, kDecoNonWritable});
        }
        if (!def.debug_name.empty()) {
            std::vector<u32> name_operands{variable};
            for (u32 word : packed(def.debug_name)) name_operands.push_back(word);
            names_.op(5 /*OpName*/, name_operands);
        }
        buffer_variables_[value_id] = variable;
        buffer_scalars_[value_id] = s;
        ++binding;
    }

    // gl_GlobalInvocationID is declared whenever the kernel is element-wise
    // (element_count guard) so the guard has an index to compare.
    u32 gid_variable = 0;
    const bool guarded = k_.element_count > 0;
    if (guarded) {
        gid_variable = fresh();
        types_stream_.op(kOpVariable,
                         {t_ptr("in-v3u32", kStorageInput, t_v3u32()),
                          gid_variable, kStorageInput});
        decorations_.op(kOpDecorate,
                        {gid_variable, kDecoBuiltIn, kBuiltInGlobalInvocationId});
    }

    // Function skeleton.
    const u32 fn = fresh();
    const u32 entry_label = fresh();
    Stream fn_stream;
    fn_stream.op(kOpFunction, {t_void(), fn, 0u /*None*/, t_fn_void()});
    fn_stream.op(kOpLabel, {entry_label});

    u32 body_label = 0, merge_label = 0;
    if (guarded) {
        const u32 gid_vector = fresh();
        body_.words.clear();
        // The guard prologue is emitted straight into the function stream so
        // `body_` holds only the guarded region.
        fn_stream.op(kOpLoad, {t_v3u32(), gid_vector, gid_variable});
        const u32 gid = fresh();
        fn_stream.op(kOpCompositeExtract, {t_u32(), gid, gid_vector, 0u});
        value_results_[kGlobalIndexValue] = gid; // see note below
        const u32 in_range = fresh();
        fn_stream.op(kOpULessThan,
                     {t_bool(), in_range, gid, const_u32(k_.element_count)});
        body_label = fresh();
        merge_label = fresh();
        fn_stream.op(kOpSelectionMerge, {merge_label, 0u /*None*/});
        fn_stream.op(kOpBranchConditional, {in_range, body_label, merge_label});
        fn_stream.op(kOpLabel, {body_label});
    }

    for (const auto& ins : k_.instrs) lower_instruction(ins);

    // Assemble function: prologue (+guard), body, close.
    for (u32 word : body_.words) fn_stream.words.push_back(word);
    if (guarded) {
        fn_stream.op(kOpBranch, {merge_label});
        fn_stream.op(kOpLabel, {merge_label});
    }
    fn_stream.op(kOpReturn, {});
    fn_stream.op(kOpFunctionEnd, {});

    // Header + module-order sections.
    std::vector<u32>& words = out.binary.words;
    words = {0x07230203u, 0x00010300u /*SPIR-V 1.3*/, 0u /*generator*/,
             next_id_, 0u};
    Stream prelude;
    prelude.op(kOpCapability, {kCapabilityShader});
    {
        std::vector<u32> operands{glsl_import_};
        for (u32 word : packed("GLSL.std.450")) operands.push_back(word);
        prelude.op(kOpExtInstImport, operands);
    }
    prelude.op(kOpMemoryModel, {kMemoryModelLogical, kMemoryModelGLSL450});
    {
        std::vector<u32> operands{kExecutionModelGLCompute, fn};
        for (u32 word : packed(k_.name.empty() ? "main" : k_.name)) {
            operands.push_back(word);
        }
        // Pre-1.4 interface lists Input/Output variables only.
        if (gid_variable) operands.push_back(gid_variable);
        prelude.op(kOpEntryPoint, operands);
    }
    prelude.op(kOpExecutionMode,
               {fn, kExecutionModeLocalSize,
                k_.suggested_local_size[0], k_.suggested_local_size[1],
                k_.suggested_local_size[2]});

    for (u32 word : prelude.words) words.push_back(word);
    for (u32 word : names_.words) words.push_back(word);
    for (u32 word : decorations_.words) words.push_back(word);
    for (u32 word : types_stream_.words) words.push_back(word);
    for (u32 word : fn_stream.words) words.push_back(word);
    words[3] = next_id_; // final id bound

    out.id_bound = next_id_;
    out.shortfalls = shortfalls_;
    return out;
}

} // namespace

SpirvAssembly assemble_kernel_ir_to_spirv(const spirv::KernelIR& kernel) {
    ModuleAssembler assembler(kernel);
    return assembler.run();
}

bool register_spirv_backend() {
    TranslationMatrix::instance().register_backend(
        "spirv", [](const spirv::KernelIR& kernel) {
            SpirvAssembly assembly = assemble_kernel_ir_to_spirv(kernel);
            if (!assembly.complete()) {
                std::ostringstream message;
                message << "spirv backend refused kernel '" << kernel.name
                        << "':";
                for (const auto& reason : assembly.shortfalls) {
                    message << "\n  - " << reason;
                }
                throw std::runtime_error(message.str());
            }
        });
    return true;
}

// Kept for source compatibility with the original placeholder entry point.
// Returns the number of SPIR-V words assembled, 0 when the kernel was refused.
int assemble_spirv_kernel(const spirv::KernelIR& kernel) {
    SpirvAssembly assembly = assemble_kernel_ir_to_spirv(kernel);
    if (!assembly.complete()) return 0;
    return static_cast<int>(assembly.binary.words.size());
}

} // namespace kernels
} // namespace nodus
