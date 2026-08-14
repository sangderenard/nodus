// kernel_eigen.cpp
// CPU execution of KernelIR. See include/kernel_eigen_runner.h for the
// contract; src/kernels/kernel_spirv.cpp is the authority for what every
// opcode and sub_op MEANS, and this file mirrors it -- where the assembler
// picks OpSMod, so does this; where it refuses, so does this.
//
// Deliberately a plain scalar interpreter rather than Eigen expressions. The
// IR is per-invocation SSA over scalars with a mutable-cell (VAR) accumulator
// and a flat loop bracket; batching it into array expressions would mean
// re-deriving structure the IR does not carry, and a reference implementation
// earns its keep by being obviously right, not by being fast.

#include "../kernel_isa.h"
#include "kernel_eigen_runner.h"
#include "kernel_spirv_assembler.h"
#include "canonical_ops.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
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
using i32 = std::int32_t;

const char* scalar_name(ScalarType s) {
    switch (s) {
        case ScalarType::I32: return "i32";
        case ScalarType::U32: return "u32";
        case ScalarType::F32: return "f32";
    }
    return "?";
}

// A KernelIR scalar. All three scalar types are 32 bits wide, so the payload
// is kept as raw bits: the signedness reinterpretations the assembler performs
// with OpBitcast then cost nothing and cannot lose information.
struct Value {
    ScalarType type = ScalarType::U32;
    u32 bits = 0;

    static Value from_u32(u32 x) { return Value{ScalarType::U32, x}; }
    static Value from_i32(i32 x) { return Value{ScalarType::I32, static_cast<u32>(x)}; }
    static Value from_f32(float x) {
        u32 b;
        static_assert(sizeof(b) == sizeof(x));
        std::memcpy(&b, &x, sizeof(b));
        return Value{ScalarType::F32, b};
    }

    u32 u() const { return bits; }
    i32 i() const { return static_cast<i32>(bits); }
    float f() const {
        float x;
        std::memcpy(&x, &bits, sizeof(x));
        return x;
    }
};

// A 32-bit slot, either inside a bound buffer (ADDR) or a function-local cell
// (VAR). Both are just addresses, which is exactly why LOAD/STORE serve both.
struct Pointer {
    std::uint8_t* addr = nullptr;
    ScalarType scalar = ScalarType::U32;
};

struct BufferView {
    std::uint8_t* data = nullptr;
    std::size_t elements = 0;
    ScalarType scalar = ScalarType::U32;
};

class Interpreter {
public:
    Interpreter(const KernelIR& kernel, std::vector<EigenBufferBinding>& buffers)
        : k_(kernel), buffers_(buffers) {}

    bool run(std::string* error) {
        const bool ok = bind_buffers() && run_all_invocations();
        if (!ok && error) *error = error_;
        return ok;
    }

private:
    bool fail(const std::string& reason) {
        error_ = reason;
        return false;
    }

    // --- setup -------------------------------------------------------------

    bool bind_buffers() {
        if (buffers_.size() != k_.buffer_value_ids.size()) {
            return fail("kernel declares " +
                        std::to_string(k_.buffer_value_ids.size()) +
                        " buffers but " + std::to_string(buffers_.size()) +
                        " bindings were supplied");
        }
        for (std::size_t slot = 0; slot < k_.buffer_value_ids.size(); ++slot) {
            const u32 value_id = k_.buffer_value_ids[slot];
            if (value_id >= k_.values.size()) {
                return fail("buffer value %" + std::to_string(value_id) +
                            " is outside the value table");
            }
            const auto& def = k_.values[value_id];
            if (!def.type.is_buffer) {
                return fail("buffer value %" + std::to_string(value_id) +
                            " is not declared is_buffer");
            }
            const EigenBufferBinding& binding = buffers_[slot];
            if (binding.data == nullptr && binding.bytes != 0) {
                return fail("binding " + std::to_string(slot) +
                            " has a size but no data pointer");
            }
            if (binding.bytes % 4 != 0) {
                return fail("binding " + std::to_string(slot) + " is " +
                            std::to_string(binding.bytes) +
                            " bytes, not a whole number of 32-bit elements");
            }
            BufferView view;
            view.data = static_cast<std::uint8_t*>(binding.data);
            view.elements = binding.bytes / 4;
            view.scalar = def.type.scalar;
            buffer_views_[value_id] = view;
        }
        return true;
    }

    bool run_all_invocations() {
        if (k_.element_count == 0) {
            // The assembler emits no gl_GlobalInvocationID guard in this case
            // and the dispatch grid decides the invocation count; on the host
            // nothing supplies one, so there is no honest number to run.
            return fail("kernel '" + k_.name +
                        "' declares element_count 0, so the invocation count "
                        "is undefined on the CPU");
        }
        for (u32 gid = 0; gid < k_.element_count; ++gid) {
            if (!run_invocation(gid)) return false;
        }
        return true;
    }

    // --- value plumbing ----------------------------------------------------

    bool scalar_of(u32 value_id, ScalarType* out) {
        // The global index is a sentinel with no ValueDef; the assembler
        // publishes it as gl_GlobalInvocationID.x, a u32.
        if (value_id == kGlobalIndexValue) {
            *out = ScalarType::U32;
            return true;
        }
        if (value_id >= k_.values.size()) {
            return fail("value %" + std::to_string(value_id) +
                        " is outside the value table");
        }
        *out = k_.values[value_id].type.scalar;
        return true;
    }

    bool resolve(const Operand& operand, ScalarType context, Value* out) {
        if (std::holds_alternative<ValueRef>(operand.v)) {
            const u32 id = std::get<ValueRef>(operand.v).id;
            if (id == kGlobalIndexValue) {
                *out = Value::from_u32(gid_);
                return true;
            }
            auto found = values_.find(id);
            if (found == values_.end()) {
                return fail("value %" + std::to_string(id) +
                            " is used before any instruction or buffer load "
                            "defines it");
            }
            *out = found->second;
            return true;
        }
        if (std::holds_alternative<i32>(operand.v)) {
            // An i32 immediate feeding a u32/f32 context keeps the context's
            // type: Operand::i is a literal spelling, not a cast.
            const i32 raw = std::get<i32>(operand.v);
            *out = context == ScalarType::F32 ? Value::from_f32(float(raw))
                                              : Value{context, static_cast<u32>(raw)};
            return true;
        }
        if (std::holds_alternative<u32>(operand.v)) {
            const u32 raw = std::get<u32>(operand.v);
            *out = context == ScalarType::F32 ? Value::from_f32(float(raw))
                                              : Value{context, raw};
            return true;
        }
        const float raw = std::get<float>(operand.v);
        if (context != ScalarType::F32) {
            return fail("float immediate used in integer context");
        }
        *out = Value::from_f32(raw);
        return true;
    }

    // Bring a resolved value to the type the consuming instruction declares.
    // I32 and U32 are the same 32-bit machine type differing only in how the
    // op reads them, which is why the assembler lowers width casts between
    // them to OpBitcast; anything involving f32 is a genuine type error.
    bool coerce(const Value& in, ScalarType want, const char* where, Value* out) {
        if (in.type == want) {
            *out = in;
            return true;
        }
        if (in.type != ScalarType::F32 && want != ScalarType::F32) {
            *out = Value{want, in.bits};
            return true;
        }
        return fail(std::string(where) + ": a " + scalar_name(in.type) +
                    " value is used where " + scalar_name(want) + " is required");
    }

    bool operand_as(const spirv::Instruction& ins, std::size_t index,
                    ScalarType want, const char* where, Value* out) {
        Value raw;
        if (!resolve(ins.inputs[index], want, &raw)) return false;
        return coerce(raw, want, where, out);
    }

    bool single_output(const spirv::Instruction& ins, const char* what,
                       u32* out_id, ScalarType* out_type) {
        if (ins.outputs.size() != 1) {
            return fail(std::string(what) + " requires exactly one output value");
        }
        *out_id = ins.outputs[0].id;
        return scalar_of(*out_id, out_type);
    }

    static bool to_bool(const Value& v) {
        return v.type == ScalarType::F32 ? v.f() != 0.0f : v.bits != 0;
    }

    static Value from_bool(bool b, ScalarType s) {
        if (s == ScalarType::F32) return Value::from_f32(b ? 1.0f : 0.0f);
        return Value{s, b ? 1u : 0u};
    }

    Value read(const Pointer& p) const {
        u32 bits;
        std::memcpy(&bits, p.addr, sizeof(bits));
        return Value{p.scalar, bits};
    }

    static void write(const Pointer& p, const Value& v) {
        std::memcpy(p.addr, &v.bits, sizeof(v.bits));
    }

    // --- instruction execution ---------------------------------------------

    bool run_invocation(u32 gid) {
        gid_ = gid;
        values_.clear();
        pointers_.clear();
        locals_.clear();
        loops_.clear();

        std::size_t pc = 0;
        while (pc < k_.instrs.size()) {
            const spirv::Instruction& ins = k_.instrs[pc];
            if (ins.op == spirv::OpCode::LOOP_BEGIN) {
                if (!exec_loop_begin(ins, &pc)) return false;
                continue;
            }
            if (ins.op == spirv::OpCode::LOOP_END) {
                if (!exec_loop_end(&pc)) return false;
                continue;
            }
            if (!exec(ins)) return false;
            ++pc;
        }
        if (!loops_.empty()) {
            return fail("LOOP_BEGIN without a matching LOOP_END");
        }
        return true;
    }

    bool exec(const spirv::Instruction& ins) {
        using spirv::OpCode;
        switch (ins.op) {
            // The kernel frame comes from KernelIR fields, so these are metadata.
            case OpCode::MODULE_BEGIN:
            case OpCode::KERNEL_ENTRY:
                return true;
            case OpCode::AND: return exec_logical(ins, CanonicalOp::BITAND);
            case OpCode::OR: return exec_logical(ins, CanonicalOp::BITOR);
            case OpCode::XOR: return exec_logical(ins, CanonicalOp::BITXOR);
            case OpCode::NOT: return exec_logical(ins, CanonicalOp::INVERT);
            case OpCode::BINARY: return exec_binary(ins);
            case OpCode::UNARY: return exec_unary(ins);
            case OpCode::CMP: return exec_cmp(ins);
            case OpCode::SELECT: return exec_select(ins);
            case OpCode::CAST: return exec_cast(ins);
            case OpCode::ADDR: return exec_addr(ins);
            case OpCode::LOAD: return exec_load(ins);
            case OpCode::STORE: return exec_store(ins);
            case OpCode::VAR: return exec_var(ins);
            default:
                return fail("OpCode " + std::to_string(static_cast<int>(ins.op)) +
                            " has no CPU execution yet");
        }
    }

    // AND/OR/XOR/NOT are the bitwise ops on integers, exactly as the assembler
    // lowers them (OpBitwiseAnd/Or/Xor, OpNot).
    bool exec_logical(const spirv::Instruction& ins, CanonicalOp which) {
        u32 out = 0;
        ScalarType s{};
        if (!single_output(ins, "logical op", &out, &s)) return false;
        if (s == ScalarType::F32) {
            return fail("bitwise op on f32 value %" + std::to_string(out));
        }
        const std::size_t arity = which == CanonicalOp::INVERT ? 1u : 2u;
        if (ins.inputs.size() != arity) {
            return fail(arity == 1 ? "NOT takes one input"
                                   : "bitwise op takes two inputs");
        }
        Value a;
        if (!operand_as(ins, 0, s, "logical op", &a)) return false;
        if (arity == 1) {
            values_[out] = Value{s, ~a.u()};
            return true;
        }
        Value b;
        if (!operand_as(ins, 1, s, "logical op", &b)) return false;
        switch (which) {
            case CanonicalOp::BITAND: values_[out] = Value{s, a.u() & b.u()}; break;
            case CanonicalOp::BITOR: values_[out] = Value{s, a.u() | b.u()}; break;
            default: values_[out] = Value{s, a.u() ^ b.u()}; break;
        }
        return true;
    }

    bool exec_binary(const spirv::Instruction& ins) {
        u32 out = 0;
        ScalarType s{};
        if (!single_output(ins, "BINARY", &out, &s)) return false;
        if (ins.inputs.size() != 2) return fail("BINARY takes two inputs");
        Value a, b;
        if (!operand_as(ins, 0, s, "BINARY", &a)) return false;
        if (!operand_as(ins, 1, s, "BINARY", &b)) return false;

        const bool f = s == ScalarType::F32;
        const bool sign = s == ScalarType::I32;
        const auto sub = static_cast<CanonicalOp>(ins.sub_op);
        switch (sub) {
            // Integer add/sub/mul wrap, which is precisely what OpIAdd/ISub/
            // IMul specify, so unsigned arithmetic is both correct and free of
            // signed-overflow UB.
            case CanonicalOp::ADD:
                values_[out] = f ? Value::from_f32(a.f() + b.f())
                                 : Value{s, a.u() + b.u()};
                return true;
            case CanonicalOp::SUB:
                values_[out] = f ? Value::from_f32(a.f() - b.f())
                                 : Value{s, a.u() - b.u()};
                return true;
            case CanonicalOp::MUL:
                values_[out] = f ? Value::from_f32(a.f() * b.f())
                                 : Value{s, a.u() * b.u()};
                return true;
            case CanonicalOp::TRUEDIV: {
                if (f) {
                    values_[out] = Value::from_f32(a.f() / b.f());
                    return true;
                }
                if (b.u() == 0) return fail("integer division by zero in BINARY");
                if (sign) {
                    if (a.i() == INT32_MIN && b.i() == -1) {
                        return fail("integer division overflow (INT32_MIN / -1)");
                    }
                    values_[out] = Value::from_i32(a.i() / b.i());
                } else {
                    values_[out] = Value::from_u32(a.u() / b.u());
                }
                return true;
            }
            case CanonicalOp::MOD: {
                // OpFRem takes the sign of operand 1 (C fmod); OpSMod takes the
                // sign of operand 2, which C's % does not, hence the fixup.
                if (f) {
                    values_[out] = Value::from_f32(std::fmod(a.f(), b.f()));
                    return true;
                }
                if (b.u() == 0) return fail("integer modulo by zero in BINARY");
                if (sign) {
                    if (a.i() == INT32_MIN && b.i() == -1) {
                        return fail("integer modulo overflow (INT32_MIN % -1)");
                    }
                    i32 r = a.i() % b.i();
                    if (r != 0 && ((r < 0) != (b.i() < 0))) r += b.i();
                    values_[out] = Value::from_i32(r);
                } else {
                    values_[out] = Value::from_u32(a.u() % b.u());
                }
                return true;
            }
            case CanonicalOp::BITAND:
            case CanonicalOp::BITOR:
            case CanonicalOp::BITXOR:
            case CanonicalOp::SHL:
            case CanonicalOp::SHR: {
                if (f) {
                    return fail("bitwise BINARY sub_op on f32 value %" +
                                std::to_string(out));
                }
                if (sub == CanonicalOp::BITAND) {
                    values_[out] = Value{s, a.u() & b.u()};
                    return true;
                }
                if (sub == CanonicalOp::BITOR) {
                    values_[out] = Value{s, a.u() | b.u()};
                    return true;
                }
                if (sub == CanonicalOp::BITXOR) {
                    values_[out] = Value{s, a.u() ^ b.u()};
                    return true;
                }
                // A shift of 32 or more is undefined in SPIR-V and in C++;
                // refusing beats inventing a result the GPU would not produce.
                if (b.u() >= 32) {
                    return fail("shift amount " + std::to_string(b.u()) +
                                " is not below the 32-bit operand width");
                }
                if (sub == CanonicalOp::SHL) {
                    values_[out] = Value{s, a.u() << b.u()};
                } else if (sign) {
                    values_[out] = Value::from_i32(a.i() >> b.u()); // arithmetic
                } else {
                    values_[out] = Value::from_u32(a.u() >> b.u()); // logical
                }
                return true;
            }
            case CanonicalOp::POW:
                if (!f) return fail("POW is defined for f32 only");
                values_[out] = Value::from_f32(std::pow(a.f(), b.f()));
                return true;
            case CanonicalOp::MAXIMUM:
                values_[out] = f ? Value::from_f32(std::fmax(a.f(), b.f()))
                                 : (sign ? Value::from_i32(a.i() > b.i() ? a.i() : b.i())
                                         : Value::from_u32(a.u() > b.u() ? a.u() : b.u()));
                return true;
            case CanonicalOp::MINIMUM:
                values_[out] = f ? Value::from_f32(std::fmin(a.f(), b.f()))
                                 : (sign ? Value::from_i32(a.i() < b.i() ? a.i() : b.i())
                                         : Value::from_u32(a.u() < b.u() ? a.u() : b.u()));
                return true;
            default:
                return fail("BINARY sub_op " + std::to_string(ins.sub_op) +
                            " has no CPU execution yet");
        }
    }

    bool exec_unary(const spirv::Instruction& ins) {
        u32 out = 0;
        ScalarType s{};
        if (!single_output(ins, "UNARY", &out, &s)) return false;
        if (ins.inputs.size() != 1) return fail("UNARY takes one input");
        Value x;
        if (!operand_as(ins, 0, s, "UNARY", &x)) return false;

        const bool f = s == ScalarType::F32;
        const auto sub = static_cast<CanonicalOp>(ins.sub_op);
        switch (sub) {
            case CanonicalOp::NEG:
                // OpSNegate is two's-complement negation on the raw bits.
                values_[out] = f ? Value::from_f32(-x.f()) : Value{s, 0u - x.u()};
                return true;
            case CanonicalOp::ABS:
                if (f) {
                    values_[out] = Value::from_f32(std::fabs(x.f()));
                } else {
                    // GLSL SAbs reads the operand as signed regardless of the
                    // declared signedness, matching the assembler's choice.
                    const i32 v = x.i();
                    values_[out] = Value{s, v < 0 ? 0u - x.u() : x.u()};
                }
                return true;
            case CanonicalOp::INVERT:
                if (f) return fail("INVERT on f32 value %" + std::to_string(out));
                values_[out] = Value{s, ~x.u()};
                return true;
            default:
                break;
        }
        if (!f) {
            return fail("UNARY sub_op " + std::to_string(ins.sub_op) +
                        " is defined for f32 only");
        }
        const float v = x.f();
        float r = 0.0f;
        switch (sub) {
            case CanonicalOp::SQRT: r = std::sqrt(v); break;
            case CanonicalOp::EXP: r = std::exp(v); break;
            case CanonicalOp::LOG: r = std::log(v); break;
            case CanonicalOp::SIN: r = std::sin(v); break;
            case CanonicalOp::COS: r = std::cos(v); break;
            case CanonicalOp::TAN: r = std::tan(v); break;
            case CanonicalOp::ASIN: r = std::asin(v); break;
            case CanonicalOp::ACOS: r = std::acos(v); break;
            case CanonicalOp::ATAN: r = std::atan(v); break;
            case CanonicalOp::SINH: r = std::sinh(v); break;
            case CanonicalOp::COSH: r = std::cosh(v); break;
            case CanonicalOp::TANH: r = std::tanh(v); break;
            case CanonicalOp::ASINH: r = std::asinh(v); break;
            case CanonicalOp::ACOSH: r = std::acosh(v); break;
            case CanonicalOp::ATANH: r = std::atanh(v); break;
            case CanonicalOp::FLOOR: r = std::floor(v); break;
            case CanonicalOp::CEIL: r = std::ceil(v); break;
            case CanonicalOp::TRUNC: r = std::trunc(v); break;
            // GLSL Round leaves the halfway case implementation-defined, so any
            // consistent tie rule is conformant; away-from-zero is the choice.
            case CanonicalOp::ROUND: r = std::round(v); break;
            default:
                return fail("UNARY sub_op " + std::to_string(ins.sub_op) +
                            " has no CPU execution yet");
        }
        values_[out] = Value::from_f32(r);
        return true;
    }

    bool exec_cmp(const spirv::Instruction& ins) {
        u32 out = 0;
        ScalarType result_type{};
        if (!single_output(ins, "CMP", &out, &result_type)) return false;
        if (ins.inputs.size() != 2) return fail("CMP takes two inputs");
        // Operand type comes from the operands, not the (integer) result value.
        ScalarType operand_type = result_type;
        for (const Operand& operand : ins.inputs) {
            if (std::holds_alternative<ValueRef>(operand.v)) {
                if (!scalar_of(std::get<ValueRef>(operand.v).id, &operand_type)) {
                    return false;
                }
                break;
            }
        }
        Value a, b;
        if (!operand_as(ins, 0, operand_type, "CMP", &a)) return false;
        if (!operand_as(ins, 1, operand_type, "CMP", &b)) return false;

        const bool f = operand_type == ScalarType::F32;
        const bool sign = operand_type == ScalarType::I32;
        bool r = false;
        switch (static_cast<CanonicalOp>(ins.sub_op)) {
            case CanonicalOp::EQUAL:
                r = f ? a.f() == b.f() : a.bits == b.bits;
                break;
            case CanonicalOp::NOT_EQUAL:
                // OpFOrdNotEqual is ordered: a NaN operand makes it false.
                r = f ? (a.f() == a.f() && b.f() == b.f() && a.f() != b.f())
                      : a.bits != b.bits;
                break;
            case CanonicalOp::LESS:
                r = f ? a.f() < b.f() : (sign ? a.i() < b.i() : a.u() < b.u());
                break;
            case CanonicalOp::LESS_EQUAL:
                r = f ? a.f() <= b.f() : (sign ? a.i() <= b.i() : a.u() <= b.u());
                break;
            case CanonicalOp::GREATER:
                r = f ? a.f() > b.f() : (sign ? a.i() > b.i() : a.u() > b.u());
                break;
            case CanonicalOp::GREATER_EQUAL:
                r = f ? a.f() >= b.f() : (sign ? a.i() >= b.i() : a.u() >= b.u());
                break;
            default:
                return fail("CMP sub_op " + std::to_string(ins.sub_op) +
                            " has no CPU execution yet");
        }
        values_[out] = from_bool(r, result_type);
        return true;
    }

    bool exec_select(const spirv::Instruction& ins) {
        u32 out = 0;
        ScalarType s{};
        if (!single_output(ins, "SELECT", &out, &s)) return false;
        if (ins.inputs.size() != 3) return fail("SELECT takes cond, t, f");
        ScalarType cond_type = ScalarType::U32;
        if (std::holds_alternative<ValueRef>(ins.inputs[0].v)) {
            if (!scalar_of(std::get<ValueRef>(ins.inputs[0].v).id, &cond_type)) {
                return false;
            }
        }
        Value cond, t, fv;
        if (!operand_as(ins, 0, cond_type, "SELECT", &cond)) return false;
        if (!operand_as(ins, 1, s, "SELECT", &t)) return false;
        if (!operand_as(ins, 2, s, "SELECT", &fv)) return false;
        values_[out] = to_bool(cond) ? t : fv;
        return true;
    }

    bool exec_cast(const spirv::Instruction& ins) {
        u32 out = 0;
        ScalarType dst{};
        if (!single_output(ins, "CAST", &out, &dst)) return false;
        if (ins.inputs.size() != 1) return fail("CAST takes one input");
        if (!std::holds_alternative<ValueRef>(ins.inputs[0].v)) {
            return fail("CAST of an immediate; spell the constant in the target type");
        }
        ScalarType src{};
        if (!scalar_of(std::get<ValueRef>(ins.inputs[0].v).id, &src)) return false;
        Value x;
        if (!operand_as(ins, 0, src, "CAST", &x)) return false;

        switch (static_cast<CanonicalOp>(ins.sub_op)) {
            case CanonicalOp::FPTOSI:
                if (src != ScalarType::F32) return fail("FPTOSI source is not f32");
                values_[out] = Value{dst, static_cast<u32>(static_cast<i32>(x.f()))};
                return true;
            case CanonicalOp::FPTOUI:
                if (src != ScalarType::F32) return fail("FPTOUI source is not f32");
                values_[out] = Value{dst, static_cast<u32>(x.f())};
                return true;
            case CanonicalOp::SITOFP:
                if (src == ScalarType::F32) return fail("SITOFP source is f32");
                values_[out] = Value::from_f32(static_cast<float>(x.i()));
                return true;
            case CanonicalOp::UITOFP:
                if (src == ScalarType::F32) return fail("UITOFP source is f32");
                values_[out] = Value::from_f32(static_cast<float>(x.u()));
                return true;
            case CanonicalOp::ZEXT:
            case CanonicalOp::SEXT:
            case CanonicalOp::INT_TRUNC:
                // Every integer type here is 32-bit, so width casts degenerate
                // to a signedness reinterpretation (the assembler's OpBitcast).
                values_[out] = Value{dst, x.bits};
                return true;
            default:
                return fail("CAST sub_op " + std::to_string(ins.sub_op) +
                            " has no CPU execution yet");
        }
    }

    bool exec_addr(const spirv::Instruction& ins) {
        if (ins.outputs.size() != 1) {
            return fail("ADDR requires exactly one output value");
        }
        if (ins.inputs.size() < 2) {
            return fail("ADDR takes a buffer value and at least one index");
        }
        if (!std::holds_alternative<ValueRef>(ins.inputs[0].v)) {
            return fail("ADDR base must be a buffer value reference");
        }
        const u32 base_value = std::get<ValueRef>(ins.inputs[0].v).id;
        auto buffer = buffer_views_.find(base_value);
        if (buffer == buffer_views_.end()) {
            return fail("ADDR base value %" + std::to_string(base_value) +
                        " is not a declared buffer");
        }
        if (ins.inputs.size() != 2) {
            // A storage buffer here wraps ONE runtime array of scalars, so a
            // longer index chain has nothing left to step through.
            return fail("ADDR into a flat buffer takes exactly one index");
        }
        Value index;
        if (!operand_as(ins, 1, ScalarType::U32, "ADDR index", &index)) return false;
        const BufferView& view = buffer->second;
        if (index.u() >= view.elements) {
            // The GPU would read whatever the descriptor's robustness rules
            // give it; here an out-of-range index is a bug worth naming.
            return fail("ADDR index " + std::to_string(index.u()) +
                        " is outside buffer value %" + std::to_string(base_value) +
                        " of " + std::to_string(view.elements) + " elements");
        }
        Pointer p;
        p.addr = view.data + std::size_t(index.u()) * 4u;
        p.scalar = view.scalar;
        pointers_[ins.outputs[0].id] = p;
        return true;
    }

    bool exec_load(const spirv::Instruction& ins) {
        if (ins.outputs.size() != 1) {
            return fail("LOAD requires exactly one output value");
        }
        if (ins.inputs.size() != 1 ||
            !std::holds_alternative<ValueRef>(ins.inputs[0].v)) {
            return fail("LOAD takes one ADDR result");
        }
        const u32 addr_value = std::get<ValueRef>(ins.inputs[0].v).id;
        auto pointer = pointers_.find(addr_value);
        if (pointer == pointers_.end()) {
            return fail("LOAD address %" + std::to_string(addr_value) +
                        " is not an ADDR result");
        }
        values_[ins.outputs[0].id] = read(pointer->second);
        return true;
    }

    bool exec_store(const spirv::Instruction& ins) {
        if (ins.inputs.size() != 2 ||
            !std::holds_alternative<ValueRef>(ins.inputs[0].v)) {
            return fail("STORE takes an ADDR result and a value");
        }
        const u32 addr_value = std::get<ValueRef>(ins.inputs[0].v).id;
        auto pointer = pointers_.find(addr_value);
        if (pointer == pointers_.end()) {
            return fail("STORE address %" + std::to_string(addr_value) +
                        " is not an ADDR result");
        }
        Value v;
        if (!operand_as(ins, 1, pointer->second.scalar, "STORE", &v)) return false;
        write(pointer->second, v);
        return true;
    }

    // VAR: a function-local mutable cell whose result is a POINTER, so LOAD
    // and STORE work on it exactly as on an ADDR result -- which is what lets
    // a Tier-1 recipe carry an accumulator across loop iterations with no phi
    // nodes in the IR at all. The cell lives for one invocation.
    bool exec_var(const spirv::Instruction& ins) {
        u32 out = 0;
        ScalarType s{};
        if (!single_output(ins, "VAR", &out, &s)) return false;
        // deque, not vector: earlier cells must keep their addresses as later
        // VARs are allocated.
        locals_.push_back(0u);
        Pointer p;
        p.addr = reinterpret_cast<std::uint8_t*>(&locals_.back());
        p.scalar = s;
        pointers_[out] = p;
        if (!ins.inputs.empty()) {
            Value initial;
            if (!operand_as(ins, 0, s, "VAR initializer", &initial)) return false;
            write(p, initial);
        }
        return true;
    }

    // --- bounded iteration --------------------------------------------------
    //
    // The instruction list is flat, so the structured loop the assembler builds
    // out of OpLoopMerge blocks becomes a frame stack plus a back-jump: the
    // trip count is read once on entry (as the SPIR-V header's OpULessThan
    // reads it), the index is republished each iteration, and LOOP_END either
    // jumps back to the first body instruction or pops the frame.

    struct LoopFrame {
        u32 index_value = 0;
        u32 counter = 0;
        u32 trip = 0;
        std::size_t body_start = 0;
    };

    bool matching_loop_end(std::size_t begin_pc, std::size_t* out) {
        int depth = 0;
        for (std::size_t pc = begin_pc; pc < k_.instrs.size(); ++pc) {
            if (k_.instrs[pc].op == spirv::OpCode::LOOP_BEGIN) {
                ++depth;
            } else if (k_.instrs[pc].op == spirv::OpCode::LOOP_END) {
                if (--depth == 0) {
                    *out = pc;
                    return true;
                }
            }
        }
        return fail("LOOP_BEGIN without a matching LOOP_END");
    }

    // Cooperative budget: budget[0] == 0 means "yield now". The assembler folds
    // this into the loop condition itself, so the interpreter must consult it
    // at the same two points -- entry and each back-edge.
    bool budget_remaining(bool* remaining) {
        *remaining = true;
        if (k_.budget_value_id == KernelIR::kNoBudget) return true;
        auto budget = buffer_views_.find(k_.budget_value_id);
        if (budget == buffer_views_.end()) {
            return fail("budget_value_id is not a declared buffer");
        }
        if (budget->second.elements == 0) {
            return fail("budget buffer holds no elements");
        }
        Pointer p{budget->second.data, budget->second.scalar};
        const Value v = read(p);
        if (v.type == ScalarType::F32) {
            return fail("budget buffer is f32; the budget test is an integer compare");
        }
        *remaining = v.bits != 0;
        return true;
    }

    bool exec_loop_begin(const spirv::Instruction& ins, std::size_t* pc) {
        u32 index_value = 0;
        ScalarType index_type{};
        if (!single_output(ins, "LOOP_BEGIN", &index_value, &index_type)) return false;
        if (ins.inputs.size() != 1) return fail("LOOP_BEGIN takes a trip count");
        Value trip;
        if (!operand_as(ins, 0, ScalarType::U32, "LOOP_BEGIN trip count", &trip)) {
            return false;
        }
        bool permitted = true;
        if (!budget_remaining(&permitted)) return false;
        if (trip.u() == 0 || !permitted) {
            std::size_t end_pc = 0;
            if (!matching_loop_end(*pc, &end_pc)) return false;
            *pc = end_pc + 1;
            return true;
        }
        values_[index_value] = Value::from_u32(0u);
        loops_.push_back(LoopFrame{index_value, 0u, trip.u(), *pc + 1});
        ++*pc;
        return true;
    }

    bool exec_loop_end(std::size_t* pc) {
        if (loops_.empty()) return fail("LOOP_END without a matching LOOP_BEGIN");
        LoopFrame& frame = loops_.back();
        ++frame.counter;
        bool permitted = true;
        if (!budget_remaining(&permitted)) return false;
        if (frame.counter < frame.trip && permitted) {
            values_[frame.index_value] = Value::from_u32(frame.counter);
            *pc = frame.body_start;
            return true;
        }
        loops_.pop_back();
        ++*pc;
        return true;
    }

    const KernelIR& k_;
    std::vector<EigenBufferBinding>& buffers_;
    std::unordered_map<u32, BufferView> buffer_views_;

    u32 gid_ = 0;
    std::unordered_map<u32, Value> values_;
    std::unordered_map<u32, Pointer> pointers_;
    std::deque<u32> locals_;
    std::vector<LoopFrame> loops_;
    std::string error_;
};

} // namespace

bool execute_kernel_eigen(const spirv::KernelIR& kernel,
                          std::vector<EigenBufferBinding>& buffers,
                          std::string* error) {
    Interpreter interpreter(kernel, buffers);
    return interpreter.run(error);
}

} // namespace kernels
} // namespace nodus
