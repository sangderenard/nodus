#include "bitops_lowering.h"

#include <cassert>
#include <iostream>

using namespace nodus;

static void expect_last_op(const spirv::KernelIR& k, spirv::OpCode op) {
    assert(!k.instrs.empty());
    assert(k.instrs.back().op == op);
}

int main() {
    {
        bitops::KernelIrBuilder b("int_to_gray_u32");
        auto n = b.arg_u32("n");
        (void)bitops::lower_int_to_gray_u32(b, spirv::Operand::ref(n));
        // SHR is encoded as OpCode::BINARY, then XOR.
        assert(b.kernel().instrs.size() == 2);
        assert(b.kernel().instrs[0].op == spirv::OpCode::BINARY);
        assert(b.kernel().instrs[0].sub_op ==
               static_cast<int32_t>(ops::CanonicalOp::SHR));
        assert(b.kernel().instrs[0].inputs.size() == 2);
        expect_last_op(b.kernel(), spirv::OpCode::XOR);
    }

    {
        bitops::KernelIrBuilder b("getbit_u32");
        auto x = b.arg_u32("x");
        (void)bitops::lower_getbit_u32(b, spirv::Operand::ref(x), spirv::Operand::u(7u));
        assert(b.kernel().instrs.size() == 2);
        assert(b.kernel().instrs[0].op == spirv::OpCode::BINARY);
        assert(b.kernel().instrs[0].sub_op ==
               static_cast<int32_t>(ops::CanonicalOp::SHR));
        expect_last_op(b.kernel(), spirv::OpCode::AND);
    }

    {
        bitops::KernelIrBuilder b("setbit_u32");
        auto x = b.arg_u32("x");
        auto v = b.arg_u32("v");
        (void)bitops::lower_setbit_u32(b, spirv::Operand::ref(x), spirv::Operand::u(3u), spirv::Operand::ref(v));
        // Pattern: shl, not, and, and, shl, or
        assert(b.kernel().instrs.size() == 6);
        assert(b.kernel().instrs[0].op == spirv::OpCode::BINARY);
        assert(b.kernel().instrs[0].sub_op ==
               static_cast<int32_t>(ops::CanonicalOp::SHL));
        assert(b.kernel().instrs[1].op == spirv::OpCode::NOT);
        assert(b.kernel().instrs[2].op == spirv::OpCode::AND);
        assert(b.kernel().instrs[3].op == spirv::OpCode::AND);
        assert(b.kernel().instrs[4].op == spirv::OpCode::BINARY);
        expect_last_op(b.kernel(), spirv::OpCode::OR);
    }

    {
        bitops::KernelIrBuilder b("gray_to_int_u32");
        auto g = b.arg_u32("g");
        (void)bitops::lower_gray_to_int_u32(b, spirv::Operand::ref(g));
        // n = g via XOR with 0 + 5 rounds for shifts 1,2,4,8,16 (each round emits BINARY+XOR).
        assert(b.kernel().instrs.size() == 1 + (5 * 2));
        expect_last_op(b.kernel(), spirv::OpCode::XOR);
    }

    std::cout << "All bitops lowering tests passed.\n";
    return 0;
}
