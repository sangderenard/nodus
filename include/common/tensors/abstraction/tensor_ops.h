#pragma once

#include <string>

#include "common/tensors/abstraction/tensor_types.h"

namespace nodus::bitops {
class KernelIrBuilder;
}
namespace nodus::spirv {
struct Operand;
struct ValueRef;
enum class OpCode : uint16_t;
}

namespace nodus::tensors {

enum class TensorOp : uint8_t {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Shl,
    Shr,
    And,
    Or,
    Xor,
    Not,
};

struct KernelOpDesc {
    nodus::spirv::OpCode op;
    int32_t binary = 0; // uses nodus::bitops::BinaryOp values when op == OpCode::BINARY
    bool is_unary = false;
};

bool tensor_op_to_kernel(TensorOp op, KernelOpDesc* out);

bool emit_binary_kernel_op(nodus::bitops::KernelIrBuilder& b,
                           TensorOp op,
                           nodus::spirv::Operand a,
                           nodus::spirv::Operand b_in,
                           const std::string& name,
                           nodus::spirv::ValueRef* out);

bool emit_unary_kernel_op(nodus::bitops::KernelIrBuilder& b,
                          TensorOp op,
                          nodus::spirv::Operand x,
                          const std::string& name,
                          nodus::spirv::ValueRef* out);

} // namespace nodus::tensors
