// test_spirv_assembler.cpp
// Direct KernelIR -> SPIR-V assembly: a real element-wise gray-code kernel
// (out[i] = x ^ (x >> 1), the Tier-1 recipe from bitops_lowering.h) assembles
// to a well-formed module, the "spirv" TranslationMatrix entry accepts it, and
// an instruction outside the assembler's contract is refused by name.
//
// The module header and structure are asserted here; full semantic validation
// against the Khronos validator is one command away and worth running when the
// Vulkan SDK is present:  spirv-val --target-env vulkan1.1 <dumped module>

#include "bitops_lowering.h"
#include "kernel_ir_text.h"
#include "kernel_spirv_assembler.h"
#include "translation_matrix.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>

using namespace nodus;
using spirv::Instruction;
using spirv::OpCode;
using spirv::Operand;
using spirv::ScalarType;
using spirv::ValueRef;

static spirv::KernelIR build_gray_kernel() {
    bitops::KernelIrBuilder b("gray_encode_u32");
    auto& k = b.kernel();

    const auto in_buffer = b.new_scalar(ScalarType::U32, "values_in");
    const auto out_buffer = b.new_scalar(ScalarType::U32, "gray_out");
    k.values[in_buffer.id].type.is_buffer = true;
    k.values[in_buffer.id].type.is_readonly = true;
    k.values[out_buffer.id].type.is_buffer = true;
    k.buffer_value_ids = {in_buffer.id, out_buffer.id};
    k.element_count = 64;
    k.suggested_local_size = {64, 1, 1};

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

int main() {
    const spirv::KernelIR k = build_gray_kernel();

    // Direct assembly is complete and produces a well-formed module header.
    const kernels::SpirvAssembly assembly =
        kernels::assemble_kernel_ir_to_spirv(k);
    for (const auto& reason : assembly.shortfalls) {
        std::cerr << "shortfall: " << reason << "\n";
    }
    assert(assembly.complete());
    assert(assembly.binary.words.size() > 5);
    assert(assembly.binary.words[0] == 0x07230203u); // SPIR-V magic
    assert(assembly.binary.words[1] == 0x00010300u); // version 1.3
    assert(assembly.binary.words[3] == assembly.id_bound);

    // The registry path: TranslationMatrix's first real backend entry.
    kernels::register_spirv_backend();
    auto translate = kernels::TranslationMatrix::instance().get("spirv");
    assert(translate != nullptr);
    translate(k); // complete kernel: must not throw

    // Fail closed: an instruction outside the contract is refused by name.
    spirv::KernelIR bad = k;
    {
        Instruction ins;
        ins.op = OpCode::ATOMIC;
        bad.instrs.push_back(ins);
    }
    bool refused = false;
    try {
        translate(bad);
    } catch (const std::runtime_error& error) {
        refused = true;
        assert(std::string(error.what()).find("no direct SPIR-V lowering") !=
               std::string::npos);
    }
    assert(refused);

    // KIRTEXT crossing: this document is verbatim turing output
    // (kernel_ir_lowering.serialize_kernel_ir over the same gray-code chain,
    // lowered from repository SSA through the canonical-op membrane —
    // BINARY sub_op 46 = shr, XOR sub_op 44 = bitxor, ref 4294967295 =
    // kGlobalIndexValue). Parsing it must yield a kernel the assembler
    // completes.
    const std::string kirtext =
        "kirtext 1\n"
        "kernel gray_encode\n"
        "local_size 64 1 1\n"
        "element_count 64\n"
        "value U32 buffer readonly t0_in\n"
        "value U32 plain writable t0_ptr\n"
        "value U32 plain writable t0\n"
        "value U32 plain writable t1\n"
        "value U32 plain writable t2\n"
        "value U32 buffer writable t2_out\n"
        "value U32 plain writable t2_optr\n"
        "buffers 0 5\n"
        "instr ADDR -1 in ref:0 ref:4294967295 out 1\n"
        "instr LOAD -1 in ref:1 out 2\n"
        "instr BINARY 46 in ref:2 u:1 out 3\n"
        "instr XOR 44 in ref:2 ref:3 out 4\n"
        "instr ADDR -1 in ref:5 ref:4294967295 out 6\n"
        "instr STORE -1 in ref:6 ref:4 out\n";
    const spirv::KernelIR parsed = kernels::parse_kernel_ir_text(kirtext);
    assert(parsed.name == "gray_encode");
    assert(parsed.element_count == 64);
    assert(parsed.buffer_value_ids.size() == 2);
    assert(parsed.instrs.size() == 6);
    const kernels::SpirvAssembly crossed =
        kernels::assemble_kernel_ir_to_spirv(parsed);
    for (const auto& reason : crossed.shortfalls) {
        std::cerr << "kirtext shortfall: " << reason << "\n";
    }
    assert(crossed.complete());
    assert(crossed.binary.words[0] == 0x07230203u);

    // Malformed input fails by line, not by guess.
    bool parse_refused = false;
    try {
        kernels::parse_kernel_ir_text("kirtext 1\ninstr FROB 0 in out\n");
    } catch (const std::runtime_error&) {
        parse_refused = true;
    }
    assert(parse_refused);

    std::cout << "spirv assembler: " << assembly.binary.words.size()
              << " words, id bound " << assembly.id_bound
              << ", registry + refusal + kirtext OK\n";
    return 0;
}
