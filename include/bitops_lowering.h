#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "bitops.h"
#include "canonical_ops.h"
#include "kernel_isa.h"

namespace nodus::bitops {

// These enums define the meaning of the first operand to OpCode::UNARY / OpCode::BINARY.
// They are intentionally small and "Tier-0": the emitter/backend can pattern-match them,
// or a later legalization pass can rewrite them to target-specific ops.
enum class UnaryOp : std::int32_t {
  // (Reserved)
};

// Tiny helper for building spirv::KernelIR in SSA form.
// This is meant for Tier-1 lowering: bitops / composites / sugar -> Tier-0 kernel IR.
class KernelIrBuilder {
public:
  explicit KernelIrBuilder(std::string name = "bitops") { k_.name = std::move(name); }

  nodus::spirv::KernelIR& kernel() { return k_; }
  const nodus::spirv::KernelIR& kernel() const { return k_; }

  // Create a new SSA value of scalar type.
  nodus::spirv::ValueRef new_scalar(nodus::spirv::ScalarType s, std::string debug_name = {}) {
    nodus::spirv::ValueRef out{static_cast<std::uint32_t>(k_.values.size())};
    nodus::spirv::ValueDef def;
    def.type.scalar = s;
    def.type.shape = {};
    def.type.is_buffer = false;
    def.type.is_readonly = false;
    def.debug_name = std::move(debug_name);
    k_.values.push_back(std::move(def));
    return out;
  }

  // Convenience for declaring an input value (still just an SSA value in this IR).
  nodus::spirv::ValueRef arg_u32(std::string debug_name) {
    return new_scalar(nodus::spirv::ScalarType::U32, std::move(debug_name));
  }

  // Emit helpers
  nodus::spirv::ValueRef emit_and(nodus::spirv::Operand a, nodus::spirv::Operand b, std::string name = {}) {
    auto out = new_scalar(nodus::spirv::ScalarType::U32, std::move(name));
    nodus::spirv::Instruction ins;
    ins.op = nodus::spirv::OpCode::AND;
    ins.inputs = {a, b};
    ins.outputs = {out};
    k_.instrs.push_back(std::move(ins));
    return out;
  }

  nodus::spirv::ValueRef emit_or(nodus::spirv::Operand a, nodus::spirv::Operand b, std::string name = {}) {
    auto out = new_scalar(nodus::spirv::ScalarType::U32, std::move(name));
    nodus::spirv::Instruction ins;
    ins.op = nodus::spirv::OpCode::OR;
    ins.inputs = {a, b};
    ins.outputs = {out};
    k_.instrs.push_back(std::move(ins));
    return out;
  }

  nodus::spirv::ValueRef emit_xor(nodus::spirv::Operand a, nodus::spirv::Operand b, std::string name = {}) {
    auto out = new_scalar(nodus::spirv::ScalarType::U32, std::move(name));
    nodus::spirv::Instruction ins;
    ins.op = nodus::spirv::OpCode::XOR;
    ins.inputs = {a, b};
    ins.outputs = {out};
    k_.instrs.push_back(std::move(ins));
    return out;
  }

  nodus::spirv::ValueRef emit_not(nodus::spirv::Operand x, std::string name = {}) {
    auto out = new_scalar(nodus::spirv::ScalarType::U32, std::move(name));
    nodus::spirv::Instruction ins;
    ins.op = nodus::spirv::OpCode::NOT;
    ins.inputs = {x};
    ins.outputs = {out};
    k_.instrs.push_back(std::move(ins));
    return out;
  }

  nodus::spirv::ValueRef emit_binary(nodus::ops::CanonicalOp op,
                                    nodus::spirv::Operand a,
                                    nodus::spirv::Operand b,
                                    std::string name = {}) {
    auto out = new_scalar(nodus::spirv::ScalarType::U32, std::move(name));
    nodus::spirv::Instruction ins;
    ins.op = nodus::spirv::OpCode::BINARY;
    ins.sub_op = static_cast<std::int32_t>(op);
    ins.inputs = {a, b};
    ins.outputs = {out};
    k_.instrs.push_back(std::move(ins));
    return out;
  }

private:
  nodus::spirv::KernelIR k_{};
};

// -----------------------------------------------------------------------------
// BitOps lowering recipes (u32)
// -----------------------------------------------------------------------------

inline nodus::spirv::ValueRef lower_int_to_gray_u32(KernelIrBuilder& b, nodus::spirv::Operand n) {
  auto n_shr1 = b.emit_binary(nodus::ops::CanonicalOp::SHR, n, nodus::spirv::Operand::u(1u), "n_shr1");
  return b.emit_xor(n, nodus::spirv::Operand::ref(n_shr1), "gray");
}

// Mirrors the semantics in BitOps::gray_to_int (note it uses `g >> shift`, not `n >> shift`).
inline nodus::spirv::ValueRef lower_gray_to_int_u32(KernelIrBuilder& b, nodus::spirv::Operand g) {
  auto n = b.new_scalar(nodus::spirv::ScalarType::U32, "n");
  // Model "n = g" via a synthetic XOR with 0 (keeps everything SSA-only for now).
  // If you later add OpCode::MOV / OpCode::COPY, replace this.
  {
    nodus::spirv::Instruction ins;
    ins.op = nodus::spirv::OpCode::XOR;
    ins.inputs = {g, nodus::spirv::Operand::u(0u)};
    ins.outputs = {n};
    b.kernel().instrs.push_back(std::move(ins));
  }
  for (std::uint32_t shift = 1; shift < 32; shift <<= 1) {
    auto g_shr = b.emit_binary(nodus::ops::CanonicalOp::SHR, g, nodus::spirv::Operand::u(shift), "g_shr");
    // n = n ^ g_shr
    auto n2 = b.emit_xor(nodus::spirv::Operand::ref(n), nodus::spirv::Operand::ref(g_shr), "n_xor");
    n = n2;
  }
  return n;
}

inline nodus::spirv::ValueRef lower_getbit_u32(KernelIrBuilder& b,
                                              nodus::spirv::Operand x,
                                              nodus::spirv::Operand bit) {
  auto shr = b.emit_binary(nodus::ops::CanonicalOp::SHR, x, bit, "x_shr");
  return b.emit_and(nodus::spirv::Operand::ref(shr), nodus::spirv::Operand::u(1u), "bit");
}

inline nodus::spirv::ValueRef lower_setbit_u32(KernelIrBuilder& b,
                                              nodus::spirv::Operand x,
                                              nodus::spirv::Operand bit,
                                              nodus::spirv::Operand v) {
  auto one_shl = b.emit_binary(nodus::ops::CanonicalOp::SHL, nodus::spirv::Operand::u(1u), bit, "one_shl");
  auto inv = b.emit_not(nodus::spirv::Operand::ref(one_shl), "inv_mask");
  auto cleared = b.emit_and(x, nodus::spirv::Operand::ref(inv), "cleared");
  auto v1 = b.emit_and(v, nodus::spirv::Operand::u(1u), "v_lsb");
  auto vshl = b.emit_binary(nodus::ops::CanonicalOp::SHL, nodus::spirv::Operand::ref(v1), bit, "v_shl");
  return b.emit_or(nodus::spirv::Operand::ref(cleared), nodus::spirv::Operand::ref(vshl), "setbit");
}

} // namespace nodus::bitops
