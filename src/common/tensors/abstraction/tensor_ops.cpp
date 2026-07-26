#include "common/tensors/abstraction/tensor_ops.h"

#include "bitops_lowering.h"

namespace nodus::tensors {

bool tensor_op_to_kernel(TensorOp op, KernelOpDesc* out) {
    if (!out) return false;
    switch (op) {
        case TensorOp::Add:
            out->op = nodus::spirv::OpCode::BINARY;
            out->canonical = nodus::ops::CanonicalOp::ADD;
            out->is_unary = false;
            return true;
        case TensorOp::Sub:
            out->op = nodus::spirv::OpCode::BINARY;
            out->canonical = nodus::ops::CanonicalOp::SUB;
            out->is_unary = false;
            return true;
        case TensorOp::Mul:
            out->op = nodus::spirv::OpCode::BINARY;
            out->canonical = nodus::ops::CanonicalOp::MUL;
            out->is_unary = false;
            return true;
        case TensorOp::Div:
            out->op = nodus::spirv::OpCode::BINARY;
            out->canonical = nodus::ops::CanonicalOp::TRUEDIV;
            out->is_unary = false;
            return true;
        case TensorOp::Mod:
            out->op = nodus::spirv::OpCode::BINARY;
            out->canonical = nodus::ops::CanonicalOp::MOD;
            out->is_unary = false;
            return true;
        case TensorOp::Shl:
            out->op = nodus::spirv::OpCode::BINARY;
            out->canonical = nodus::ops::CanonicalOp::SHL;
            out->is_unary = false;
            return true;
        case TensorOp::Shr:
            out->op = nodus::spirv::OpCode::BINARY;
            out->canonical = nodus::ops::CanonicalOp::SHR;
            out->is_unary = false;
            return true;
        case TensorOp::And:
            out->op = nodus::spirv::OpCode::AND;
            out->canonical = nodus::ops::CanonicalOp::BITAND;
            out->is_unary = false;
            return true;
        case TensorOp::Or:
            out->op = nodus::spirv::OpCode::OR;
            out->canonical = nodus::ops::CanonicalOp::BITOR;
            out->is_unary = false;
            return true;
        case TensorOp::Xor:
            out->op = nodus::spirv::OpCode::XOR;
            out->canonical = nodus::ops::CanonicalOp::BITXOR;
            out->is_unary = false;
            return true;
        case TensorOp::Not:
            out->op = nodus::spirv::OpCode::NOT;
            out->canonical = nodus::ops::CanonicalOp::INVERT;
            out->is_unary = true;
            return true;
        default:
            return false;
    }
}

bool emit_binary_kernel_op(nodus::bitops::KernelIrBuilder& b,
                           TensorOp op,
                           nodus::spirv::Operand a,
                           nodus::spirv::Operand b_in,
                           const std::string& name,
                           nodus::spirv::ValueRef* out) {
    if (!out) return false;
    KernelOpDesc desc{};
    if (!tensor_op_to_kernel(op, &desc)) return false;
    if (desc.is_unary) return false;

    switch (desc.op) {
        case nodus::spirv::OpCode::AND:
            *out = b.emit_and(a, b_in, name);
            return true;
        case nodus::spirv::OpCode::OR:
            *out = b.emit_or(a, b_in, name);
            return true;
        case nodus::spirv::OpCode::XOR:
            *out = b.emit_xor(a, b_in, name);
            return true;
        case nodus::spirv::OpCode::BINARY:
            *out = b.emit_binary(desc.canonical, a, b_in, name);
            return true;
        default:
            return false;
    }
}

bool emit_unary_kernel_op(nodus::bitops::KernelIrBuilder& b,
                          TensorOp op,
                          nodus::spirv::Operand x,
                          const std::string& name,
                          nodus::spirv::ValueRef* out) {
    if (!out) return false;
    KernelOpDesc desc{};
    if (!tensor_op_to_kernel(op, &desc)) return false;
    if (!desc.is_unary) return false;

    switch (desc.op) {
        case nodus::spirv::OpCode::NOT:
            *out = b.emit_not(x, name);
            return true;
        default:
            return false;
    }
}

} // namespace nodus::tensors
