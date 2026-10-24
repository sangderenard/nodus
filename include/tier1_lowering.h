#pragma once

// Tier-1 tensor composites lowered exclusively to the established Tier-0
// KernelIR.  This is the authority named by docs/TIERS.md: emitters never
// learn REDUCE/CONTRACT/GENERATE opcodes; they receive the primitive program
// produced here.  bitops::KernelIrBuilder is the repository's existing SSA
// KernelIR builder and is deliberately reused rather than shadowed.

#include "bitops_lowering.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nodus::tier1 {

namespace detail {

inline spirv::ValueRef emit_value(
    bitops::KernelIrBuilder& builder, spirv::OpCode opcode,
    spirv::ScalarType scalar, std::vector<spirv::Operand> inputs,
    std::string name, std::int32_t sub_op = -1) {
  const auto out = builder.new_scalar(scalar, std::move(name));
  spirv::Instruction ins;
  ins.op = opcode;
  ins.sub_op = sub_op;
  ins.inputs = std::move(inputs);
  ins.outputs = {out};
  builder.kernel().instrs.push_back(std::move(ins));
  return out;
}

inline void emit_void(bitops::KernelIrBuilder& builder, spirv::OpCode opcode,
                      std::vector<spirv::Operand> inputs = {}) {
  spirv::Instruction ins;
  ins.op = opcode;
  ins.inputs = std::move(inputs);
  builder.kernel().instrs.push_back(std::move(ins));
}

inline spirv::ValueRef binary(bitops::KernelIrBuilder& builder,
                              ops::CanonicalOp op,
                              spirv::ScalarType scalar,
                              spirv::Operand left,
                              spirv::Operand right,
                              std::string name) {
  const ops::OpDesc* desc = ops::find_op(op);
  if (!desc || desc->kernel_op != "BINARY") {
    throw std::invalid_argument("Tier-1 binary combiner is not a BINARY Tier-0 op");
  }
  return emit_value(builder, spirv::OpCode::BINARY, scalar,
                    {left, right}, std::move(name),
                    static_cast<std::int32_t>(op));
}

inline spirv::ValueRef address(bitops::KernelIrBuilder& builder,
                               spirv::ValueRef buffer,
                               spirv::Operand index,
                               spirv::ScalarType scalar,
                               std::string name) {
  return emit_value(builder, spirv::OpCode::ADDR, scalar,
                    {spirv::Operand::ref(buffer), index}, std::move(name));
}

inline spirv::ValueRef load(bitops::KernelIrBuilder& builder,
                            spirv::ValueRef pointer,
                            spirv::ScalarType scalar,
                            std::string name) {
  return emit_value(builder, spirv::OpCode::LOAD, scalar,
                    {spirv::Operand::ref(pointer)}, std::move(name));
}

inline void store(bitops::KernelIrBuilder& builder,
                  spirv::ValueRef pointer, spirv::Operand value) {
  emit_void(builder, spirv::OpCode::STORE,
            {spirv::Operand::ref(pointer), value});
}

} // namespace detail

// Reduce one flat source buffer to one scalar. The caller owns launch shape,
// buffer declarations, and the final output store; this recipe owns only the
// composite's semantic body.
inline spirv::ValueRef lower_reduce(
    bitops::KernelIrBuilder& builder,
    spirv::ValueRef source_buffer,
    spirv::ScalarType scalar,
    spirv::Operand extent,
    ops::CanonicalOp combine,
    spirv::Operand identity) {
  auto accumulator = detail::emit_value(
      builder, spirv::OpCode::VAR, scalar, {identity}, "reduce_accumulator");
  auto index = detail::emit_value(
      builder, spirv::OpCode::LOOP_BEGIN, spirv::ScalarType::U32,
      {extent}, "reduce_index");
  auto source_pointer = detail::address(
      builder, source_buffer, spirv::Operand::ref(index), scalar, "reduce_source_ptr");
  auto source_value = detail::load(
      builder, source_pointer, scalar, "reduce_source");
  auto accumulated = detail::load(
      builder, accumulator, scalar, "reduce_prior");
  auto next = detail::binary(
      builder, combine, scalar, spirv::Operand::ref(accumulated),
      spirv::Operand::ref(source_value), "reduce_next");
  detail::store(builder, accumulator, spirv::Operand::ref(next));
  detail::emit_void(builder, spirv::OpCode::LOOP_END);
  return detail::load(builder, accumulator, scalar, "reduce_result");
}

// One output element of row-major A[M,K] @ B[K,N]. `output_index` is the
// flattened C index for the current invocation.
inline spirv::ValueRef lower_contract_f32(
    bitops::KernelIrBuilder& builder,
    spirv::ValueRef left_buffer,
    spirv::ValueRef right_buffer,
    spirv::Operand inner_extent,
    spirv::Operand right_columns,
    spirv::Operand output_index) {
  using spirv::Operand;
  using spirv::ScalarType;
  const auto row = detail::binary(builder, ops::CanonicalOp::FLOORDIV,
                                  ScalarType::U32, output_index, right_columns,
                                  "contract_row");
  const auto column = detail::binary(builder, ops::CanonicalOp::MOD,
                                     ScalarType::U32, output_index, right_columns,
                                     "contract_column");
  const auto row_base = detail::binary(
      builder, ops::CanonicalOp::MUL, ScalarType::U32, Operand::ref(row),
      inner_extent, "contract_row_base");
  const auto accumulator = detail::emit_value(
      builder, spirv::OpCode::VAR, ScalarType::F32, {Operand::f(0.0f)},
      "contract_accumulator");
  const auto index = detail::emit_value(
      builder, spirv::OpCode::LOOP_BEGIN, ScalarType::U32, {inner_extent},
      "contract_index");
  const auto left_index = detail::binary(
      builder, ops::CanonicalOp::ADD, ScalarType::U32, Operand::ref(row_base),
      Operand::ref(index), "contract_left_index");
  const auto left_pointer = detail::address(
      builder, left_buffer, Operand::ref(left_index), ScalarType::F32,
      "contract_left_ptr");
  const auto left = detail::load(
      builder, left_pointer, ScalarType::F32, "contract_left");
  const auto right_row = detail::binary(
      builder, ops::CanonicalOp::MUL, ScalarType::U32, Operand::ref(index),
      right_columns, "contract_right_row");
  const auto right_index = detail::binary(
      builder, ops::CanonicalOp::ADD, ScalarType::U32,
      Operand::ref(right_row), Operand::ref(column), "contract_right_index");
  const auto right_pointer = detail::address(
      builder, right_buffer, Operand::ref(right_index), ScalarType::F32,
      "contract_right_ptr");
  const auto right = detail::load(
      builder, right_pointer, ScalarType::F32, "contract_right");
  const auto product = detail::binary(
      builder, ops::CanonicalOp::MUL, ScalarType::F32, Operand::ref(left),
      Operand::ref(right), "contract_product");
  const auto prior = detail::load(
      builder, accumulator, ScalarType::F32, "contract_prior");
  const auto next = detail::binary(
      builder, ops::CanonicalOp::ADD, ScalarType::F32, Operand::ref(prior),
      Operand::ref(product), "contract_next");
  detail::store(builder, accumulator, Operand::ref(next));
  detail::emit_void(builder, spirv::OpCode::LOOP_END);
  return detail::load(builder, accumulator, ScalarType::F32, "contract_result");
}

// start + index * step, the primitive body used by the generate/arange family.
inline spirv::ValueRef lower_affine_generate(
    bitops::KernelIrBuilder& builder,
    spirv::ScalarType scalar,
    spirv::Operand index,
    spirv::Operand start,
    spirv::Operand step) {
  spirv::Operand typed_index = index;
  if (scalar == spirv::ScalarType::F32) {
    const auto converted = detail::emit_value(
        builder, spirv::OpCode::CAST, scalar, {index}, "generate_index",
        static_cast<std::int32_t>(ops::CanonicalOp::UITOFP));
    typed_index = spirv::Operand::ref(converted);
  } else if (scalar == spirv::ScalarType::I32) {
    const auto converted = detail::emit_value(
        builder, spirv::OpCode::CAST, scalar, {index}, "generate_index",
        static_cast<std::int32_t>(ops::CanonicalOp::INT_TRUNC));
    typed_index = spirv::Operand::ref(converted);
  }
  const auto scaled = detail::binary(
      builder, ops::CanonicalOp::MUL, scalar, typed_index, step,
      "generate_scaled_index");
  return detail::binary(builder, ops::CanonicalOp::ADD, scalar, start,
                        spirv::Operand::ref(scaled), "generate_value");
}

} // namespace nodus::tier1
