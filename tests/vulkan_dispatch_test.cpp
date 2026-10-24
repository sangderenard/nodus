// vulkan_dispatch_test.cpp
//
// The loop closed: KernelIR -> SPIR-V (kernel_spirv.cpp) -> a real GPU.
//
// Everything before this validated structure -- spirv-val says a module is
// well formed, which is necessary and not sufficient. Here the assembled
// gray-code kernel runs on the device and its output is compared against the
// same recipe computed on the host, so "the assembler is correct" becomes a
// measured result.
//
// Skips (exit 0) when no Vulkan device is present, so the suite stays green
// on machines without one.
#include "bitops_lowering.h"
#include "canonical_ops.h"
#include "kernel_spirv_assembler.h"
#include "kernel_vulkan_runner.h"

#include <array>
#include <cstdint>
#include <cmath>
#include <cstdio>
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

spirv::KernelIR build_scalar_concordance_kernel() {
    spirv::KernelIR k;
    k.name = "canonical_scalar_concordance";
    k.element_count = 1;
    k.suggested_local_size = {1, 1, 1};

    const struct Probe {
        OpCode opcode;
        ops::CanonicalOp sub_op;
        std::vector<Operand> inputs;
    } probes[] = {
        {OpCode::BINARY, ops::CanonicalOp::MOD,
         {Operand::f(-5.5f), Operand::f(2.0f)}},
        {OpCode::BINARY, ops::CanonicalOp::FLOORDIV,
         {Operand::f(-5.5f), Operand::f(2.0f)}},
        {OpCode::AND, ops::CanonicalOp::LOGICAL_AND,
         {Operand::f(-2.0f), Operand::f(0.0f)}},
        {OpCode::OR, ops::CanonicalOp::LOGICAL_OR,
         {Operand::f(0.0f), Operand::f(-2.0f)}},
        {OpCode::NOT, ops::CanonicalOp::LOGICAL_NOT,
         {Operand::f(0.0f)}},
        {OpCode::UNARY, ops::CanonicalOp::SIGN,
         {Operand::f(-2.0f)}},
        {OpCode::UNARY, ops::CanonicalOp::SIGMOID,
         {Operand::f(0.0f)}},
        {OpCode::UNARY, ops::CanonicalOp::ISFINITE,
         {Operand::f(-2.0f)}},
        {OpCode::UNARY, ops::CanonicalOp::ISNAN,
         {Operand::f(NAN)}},
        {OpCode::UNARY, ops::CanonicalOp::ISINF,
         {Operand::f(INFINITY)}},
    };

    for (const auto& probe : probes) {
        const std::uint32_t result = static_cast<std::uint32_t>(k.values.size());
        k.values.push_back({{ScalarType::F32, {}, false, false}, "probe"});
        k.instrs.push_back({probe.opcode, static_cast<std::int32_t>(probe.sub_op),
                            probe.inputs, {{result}}});
        const std::uint32_t output = static_cast<std::uint32_t>(k.values.size());
        k.values.push_back({{ScalarType::F32, {}, true, false}, "probe_out"});
        k.buffer_value_ids.push_back(output);
        const std::uint32_t pointer = static_cast<std::uint32_t>(k.values.size());
        k.values.push_back({{ScalarType::F32, {}, false, false}, "probe_ptr"});
        k.instrs.push_back({OpCode::ADDR, -1,
                            {Operand::ref({output}),
                             Operand::ref({kernels::kGlobalIndexValue})},
                            {{pointer}}});
        k.instrs.push_back({OpCode::STORE, -1,
                            {Operand::ref({pointer}), Operand::ref({result})}, {}});
    }
    return k;
}

} // namespace

int main() {
    if (!kernels::vulkan_available()) {
        std::cout << "[VK] skipped: no Vulkan compute device present\n";
        return 0;
    }
    std::cout << "[VK] device: " << kernels::vulkan_device_name() << "\n";

    const spirv::KernelIR k = build_gray_kernel();
    const kernels::SpirvAssembly assembly = kernels::assemble_kernel_ir_to_spirv(k);
    for (const auto& reason : assembly.shortfalls) {
        std::cerr << "[VK] shortfall: " << reason << "\n";
    }
    if (!assembly.complete()) {
        std::cerr << "[VK] FAILED: assembly incomplete\n";
        return 1;
    }

    std::vector<std::uint32_t> input(kElements);
    std::vector<std::uint32_t> output(kElements, 0xDEADBEEFu);
    for (std::uint32_t i = 0; i < kElements; ++i) input[i] = i;

    std::vector<kernels::VulkanBufferBinding> buffers = {
        {input.data(), input.size() * sizeof(std::uint32_t), false},
        {output.data(), output.size() * sizeof(std::uint32_t), true},
    };
    std::string error;
    // One workgroup: local_size_x == kElements, so a single group covers the
    // whole domain and the assembler's bounds guard handles the remainder.
    if (!kernels::dispatch_spirv(assembly.binary.words, k.name.c_str(), buffers,
                                 1, &error)) {
        std::cerr << "[VK] FAILED: " << error << "\n";
        return 1;
    }

    int wrong = 0;
    for (std::uint32_t i = 0; i < kElements; ++i) {
        const std::uint32_t want = i ^ (i >> 1); // host reference
        if (output[i] != want) {
            if (wrong < 5) {
                std::cerr << "[VK] mismatch at " << i << ": got " << output[i]
                          << " want " << want << "\n";
            }
            ++wrong;
        }
    }
    if (wrong) {
        std::cerr << "[VK] FAILED: " << wrong << " of " << kElements
                  << " elements wrong\n";
        return 1;
    }

    std::cout << "[VK] PASS: " << kElements
              << " elements gray-encoded on device, all match the host reference"
              << " (e.g. in[7]=7 -> out[7]=" << output[7] << ")\n";

    const spirv::KernelIR concordance = build_scalar_concordance_kernel();
    const kernels::SpirvAssembly concordance_assembly =
        kernels::assemble_kernel_ir_to_spirv(concordance);
    if (!concordance_assembly.complete()) {
        for (const auto& reason : concordance_assembly.shortfalls) {
            std::cerr << "[VK] scalar concordance shortfall: " << reason << "\n";
        }
        return 1;
    }
    std::array<float, 10> observed{};
    std::vector<kernels::VulkanBufferBinding> concordance_buffers;
    for (float& value : observed) {
        concordance_buffers.push_back({&value, sizeof(value), true});
    }
    if (!kernels::dispatch_spirv(
            concordance_assembly.binary.words, concordance.name.c_str(),
            concordance_buffers, 1, &error)) {
        std::cerr << "[VK] scalar concordance dispatch failed: " << error << "\n";
        return 1;
    }
    const std::array<float, 10> expected{
        0.5f, -3.0f, 0.0f, 1.0f, 1.0f, -1.0f, 0.5f, 1.0f, 1.0f, 1.0f,
    };
    for (std::size_t i = 0; i < observed.size(); ++i) {
        if (std::fabs(observed[i] - expected[i]) > 1e-6f) {
            std::cerr << "[VK] scalar concordance mismatch " << i << ": got "
                      << observed[i] << " want " << expected[i] << "\n";
            return 1;
        }
    }
    std::cout << "[VK] PASS: canonical scalar semantics agree on device\n";

    return 0;
}
