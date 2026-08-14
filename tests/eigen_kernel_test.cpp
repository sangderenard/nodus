// eigen_kernel_test.cpp
//
// The other half of the loop: KernelIR -> CPU (kernel_eigen.cpp).
//
// vulkan_dispatch_test.cpp runs the assembled module on a device and compares
// it against a host reference written by hand, which only works on a machine
// that has a device. This runs the SAME two kernels -- built by the same
// recipes, from the same builders -- through the CPU interpreter, so the
// meaning of every opcode is checked on every machine, and the interpreter
// itself becomes the reference the device path can be diffed against.
#include "bitops_lowering.h"
#include "kernel_eigen_runner.h"
#include "kernel_spirv_assembler.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace nodus;
using spirv::Instruction;
using spirv::OpCode;
using spirv::Operand;
using spirv::ScalarType;
using spirv::ValueRef;

namespace {

constexpr std::uint32_t kElements = 64;

// out[i] = x ^ (x >> 1), x = in[i] -- the Tier-1 gray-code recipe from
// bitops_lowering.h lowered to Tier-0 instructions.
spirv::KernelIR build_gray_kernel() {
    bitops::KernelIrBuilder b("gray_encode_u32");
    auto& k = b.kernel();

    const auto in_buffer = b.new_scalar(ScalarType::U32, "values_in");
    const auto out_buffer = b.new_scalar(ScalarType::U32, "gray_out");
    k.values[in_buffer.id].type.is_buffer = true;
    k.values[in_buffer.id].type.is_readonly = true;
    k.values[out_buffer.id].type.is_buffer = true;
    k.buffer_value_ids = {in_buffer.id, out_buffer.id};
    k.element_count = kElements;
    k.suggested_local_size = {kElements, 1, 1};

    const ValueRef gid{kernels::kGlobalIndexValue};
    const auto in_ptr = b.new_scalar(ScalarType::U32, "in_ptr");
    {
        Instruction ins;
        ins.op = OpCode::ADDR;
        ins.inputs = {Operand::ref(in_buffer), Operand::ref(gid)};
        ins.outputs = {in_ptr};
        k.instrs.push_back(ins);
    }
    const auto x = b.new_scalar(ScalarType::U32, "x");
    {
        Instruction ins;
        ins.op = OpCode::LOAD;
        ins.inputs = {Operand::ref(in_ptr)};
        ins.outputs = {x};
        k.instrs.push_back(ins);
    }
    const auto gray = bitops::lower_int_to_gray_u32(b, Operand::ref(x));
    const auto out_ptr = b.new_scalar(ScalarType::U32, "out_ptr");
    {
        Instruction ins;
        ins.op = OpCode::ADDR;
        ins.inputs = {Operand::ref(out_buffer), Operand::ref(gid)};
        ins.outputs = {out_ptr};
        k.instrs.push_back(ins);
    }
    {
        Instruction ins;
        ins.op = OpCode::STORE;
        ins.inputs = {Operand::ref(out_ptr), Operand::ref(gray)};
        k.instrs.push_back(ins);
    }
    return k;
}
} // namespace

int main() {
    {
        const spirv::KernelIR k = build_gray_kernel();

        std::vector<std::uint32_t> input(kElements);
        std::vector<std::uint32_t> output(kElements, 0xDEADBEEFu);
        for (std::uint32_t i = 0; i < kElements; ++i) input[i] = i;

        std::vector<kernels::EigenBufferBinding> buffers = {
            {input.data(), input.size() * sizeof(std::uint32_t), false},
            {output.data(), output.size() * sizeof(std::uint32_t), true},
        };
        std::string error;
        if (!kernels::execute_kernel_eigen(k, buffers, &error)) {
            std::cerr << "[CPU] FAILED: " << error << "\n";
            return 1;
        }

        int wrong = 0;
        for (std::uint32_t i = 0; i < kElements; ++i) {
            const std::uint32_t want = i ^ (i >> 1); // host reference
            if (output[i] != want) {
                if (wrong < 5) {
                    std::cerr << "[CPU] mismatch at " << i << ": got " << output[i]
                              << " want " << want << "\n";
                }
                ++wrong;
            }
        }
        if (wrong) {
            std::cerr << "[CPU] FAILED: " << wrong << " of " << kElements
                      << " elements wrong\n";
            return 1;
        }
        std::cout << "[CPU] PASS: " << kElements
                  << " elements gray-encoded by the KernelIR interpreter, all match"
                  << " the host reference (e.g. in[7]=7 -> out[7]=" << output[7]
                  << ")\n";
    }

    return 0;
}
