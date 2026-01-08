#include "common/tensors/abstraction/tensor_ops.h"

#include "bitops_lowering.h"

#include <cassert>
#include <iostream>

using namespace nodus;
using namespace nodus::tensors;

int main() {
    bitops::KernelIrBuilder b("tensor_ops");
    auto a = b.arg_u32("a");
    auto c = b.arg_u32("c");

    spirv::ValueRef out{};
    bool ok = emit_binary_kernel_op(b,
                                    TensorOp::Add,
                                    spirv::Operand::ref(a),
                                    spirv::Operand::ref(c),
                                    "add",
                                    &out);
    assert(ok);
    assert(!b.kernel().instrs.empty());
    assert(b.kernel().instrs.back().op == spirv::OpCode::BINARY);

    ok = emit_unary_kernel_op(b, TensorOp::Not, spirv::Operand::ref(a), "not", &out);
    assert(ok);
    assert(b.kernel().instrs.back().op == spirv::OpCode::NOT);

    std::cout << "tensor_ops_lowering_test: ok\n";
    return 0;
}
