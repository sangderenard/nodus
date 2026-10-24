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
#include "canonical_ops.h"
#include "kernel_eigen_runner.h"
#include "kernel_spirv_assembler.h"
#include "tier1_lowering.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nodus;
using spirv::Instruction;
using spirv::OpCode;
using spirv::Operand;
using spirv::ScalarType;
using spirv::ValueRef;

namespace {

constexpr std::uint32_t kElements = 64;

spirv::OpCode opcode_for(std::string_view name) {
    if (name == "UNARY") return OpCode::UNARY;
    if (name == "BINARY") return OpCode::BINARY;
    if (name == "CMP") return OpCode::CMP;
    if (name == "CAST") return OpCode::CAST;
    if (name == "SELECT") return OpCode::SELECT;
    if (name == "AND") return OpCode::AND;
    if (name == "OR") return OpCode::OR;
    if (name == "XOR") return OpCode::XOR;
    if (name == "NOT") return OpCode::NOT;
    throw std::runtime_error("test has no opcode spelling for " + std::string(name));
}

spirv::ScalarType input_scalar(const ops::OpDesc& op) {
    using ops::CanonicalOp;
    switch (static_cast<CanonicalOp>(op.canonical_id)) {
        case CanonicalOp::BITAND:
        case CanonicalOp::BITOR:
        case CanonicalOp::BITXOR:
        case CanonicalOp::SHL:
        case CanonicalOp::SHR:
        case CanonicalOp::INVERT:
        case CanonicalOp::UITOFP:
        case CanonicalOp::INT_TRUNC:
        case CanonicalOp::SEXT:
            return ScalarType::U32;
        case CanonicalOp::SITOFP:
        case CanonicalOp::ZEXT:
            return ScalarType::I32;
        default:
            return ScalarType::F32;
    }
}

spirv::ScalarType output_scalar(const ops::OpDesc& op) {
    using ops::CanonicalOp;
    switch (static_cast<CanonicalOp>(op.canonical_id)) {
        case CanonicalOp::BITAND:
        case CanonicalOp::BITOR:
        case CanonicalOp::BITXOR:
        case CanonicalOp::SHL:
        case CanonicalOp::SHR:
        case CanonicalOp::INVERT:
        case CanonicalOp::FPTOUI:
        case CanonicalOp::ZEXT:
            return ScalarType::U32;
        case CanonicalOp::FPTOSI:
        case CanonicalOp::INT_TRUNC:
        case CanonicalOp::SEXT:
            return ScalarType::I32;
        default:
            return ScalarType::F32;
    }
}

struct ScalarKernel {
    spirv::KernelIR kernel;
    std::vector<std::array<std::uint32_t, 1>> storage;
    std::vector<kernels::EigenBufferBinding> bindings;
};

std::uint32_t f32_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_f32(std::uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

ScalarKernel build_scalar_kernel(const ops::OpDesc& op,
                                 std::array<float, 3> f_inputs = {0.5f, 2.0f, -1.0f}) {
    ScalarKernel fixture;
    auto& k = fixture.kernel;
    k.name = "catalog_" + std::string(op.name);
    k.element_count = 1;
    k.suggested_local_size = {1, 1, 1};

    const ScalarType in_scalar = input_scalar(op);
    const ScalarType out_scalar = output_scalar(op);
    std::vector<ValueRef> loaded;
    for (int i = 0; i < op.arity; ++i) {
        const std::uint32_t buffer = static_cast<std::uint32_t>(k.values.size());
        k.values.push_back({{in_scalar, {}, true, true}, "input"});
        k.buffer_value_ids.push_back(buffer);
        const std::uint32_t pointer = static_cast<std::uint32_t>(k.values.size());
        k.values.push_back({{in_scalar, {}, false, false}, "input_ptr"});
        const std::uint32_t value = static_cast<std::uint32_t>(k.values.size());
        k.values.push_back({{in_scalar, {}, false, false}, "input_value"});
        k.instrs.push_back({OpCode::ADDR, -1,
                            {Operand::ref({buffer}), Operand::ref({kernels::kGlobalIndexValue})},
                            {{pointer}}});
        k.instrs.push_back({OpCode::LOAD, -1, {Operand::ref({pointer})}, {{value}}});
        loaded.push_back({value});
    }

    const std::uint32_t result = static_cast<std::uint32_t>(k.values.size());
    k.values.push_back({{out_scalar, {}, false, false}, "result"});
    Instruction operation;
    operation.op = opcode_for(op.kernel_op);
    operation.sub_op = op.canonical_id;
    for (const auto value : loaded) operation.inputs.push_back(Operand::ref(value));
    operation.outputs = {{result}};
    k.instrs.push_back(operation);

    const std::uint32_t output = static_cast<std::uint32_t>(k.values.size());
    k.values.push_back({{out_scalar, {}, true, false}, "output"});
    k.buffer_value_ids.push_back(output);
    const std::uint32_t output_ptr = static_cast<std::uint32_t>(k.values.size());
    k.values.push_back({{out_scalar, {}, false, false}, "output_ptr"});
    k.instrs.push_back({OpCode::ADDR, -1,
                        {Operand::ref({output}), Operand::ref({kernels::kGlobalIndexValue})},
                        {{output_ptr}}});
    k.instrs.push_back({OpCode::STORE, -1,
                        {Operand::ref({output_ptr}), Operand::ref({result})}, {}});

    fixture.storage.resize(static_cast<std::size_t>(op.arity) + 1u);
    for (int i = 0; i < op.arity; ++i) {
        fixture.storage[static_cast<std::size_t>(i)][0] =
            in_scalar == ScalarType::F32 ? f32_bits(f_inputs[static_cast<std::size_t>(i)])
                                         : static_cast<std::uint32_t>(i == 0 ? 7 : 1);
    }
    for (std::size_t i = 0; i < fixture.storage.size(); ++i) {
        fixture.bindings.push_back({fixture.storage[i].data(), sizeof(std::uint32_t),
                                    i + 1u == fixture.storage.size()});
    }
    return fixture;
}

float execute_scalar(const ops::OpDesc& op,
                     std::array<float, 3> inputs = {0.5f, 2.0f, -1.0f}) {
    ScalarKernel fixture = build_scalar_kernel(op, inputs);
    std::string error;
    if (!kernels::execute_kernel_eigen(fixture.kernel, fixture.bindings, &error)) {
        throw std::runtime_error(std::string(op.name) + ": " + error);
    }
    return bits_f32(fixture.storage.back()[0]);
}

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

void append_store(bitops::KernelIrBuilder& builder, ValueRef output_buffer,
                  Operand index, ValueRef value, ScalarType scalar) {
    auto& k = builder.kernel();
    const auto pointer = builder.new_scalar(scalar, "result_ptr");
    k.instrs.push_back({OpCode::ADDR, -1,
                        {Operand::ref(output_buffer), index}, {{pointer}}});
    k.instrs.push_back({OpCode::STORE, -1,
                        {Operand::ref(pointer), Operand::ref(value)}, {}});
}

spirv::KernelIR build_sum_kernel(std::uint32_t extent) {
    bitops::KernelIrBuilder builder("tier1_sum_f32");
    auto& k = builder.kernel();
    const auto input = builder.new_scalar(ScalarType::F32, "values_in");
    const auto output = builder.new_scalar(ScalarType::F32, "total_out");
    k.values[input.id].type.is_buffer = true;
    k.values[input.id].type.is_readonly = true;
    k.values[output.id].type.is_buffer = true;
    k.buffer_value_ids = {input.id, output.id};
    k.element_count = 1;
    k.suggested_local_size = {1, 1, 1};
    const auto total = tier1::lower_reduce(
        builder, input, ScalarType::F32, Operand::u(extent),
        ops::CanonicalOp::ADD, Operand::f(0.0f));
    append_store(builder, output, Operand::ref({kernels::kGlobalIndexValue}),
                 total, ScalarType::F32);
    return k;
}

spirv::KernelIR build_matmul_kernel(std::uint32_t rows,
                                    std::uint32_t inner,
                                    std::uint32_t columns) {
    bitops::KernelIrBuilder builder("tier1_matmul_f32");
    auto& k = builder.kernel();
    const auto left = builder.new_scalar(ScalarType::F32, "left");
    const auto right = builder.new_scalar(ScalarType::F32, "right");
    const auto output = builder.new_scalar(ScalarType::F32, "product");
    k.values[left.id].type.is_buffer = true;
    k.values[left.id].type.is_readonly = true;
    k.values[right.id].type.is_buffer = true;
    k.values[right.id].type.is_readonly = true;
    k.values[output.id].type.is_buffer = true;
    k.buffer_value_ids = {left.id, right.id, output.id};
    k.element_count = rows * columns;
    k.suggested_local_size = {rows * columns, 1, 1};
    const Operand gid = Operand::ref({kernels::kGlobalIndexValue});
    const auto value = tier1::lower_contract_f32(
        builder, left, right, Operand::u(inner), Operand::u(columns), gid);
    append_store(builder, output, gid, value, ScalarType::F32);
    return k;
}

spirv::KernelIR build_arange_kernel(std::uint32_t count) {
    bitops::KernelIrBuilder builder("tier1_arange_f32");
    auto& k = builder.kernel();
    const auto output = builder.new_scalar(ScalarType::F32, "values_out");
    k.values[output.id].type.is_buffer = true;
    k.buffer_value_ids = {output.id};
    k.element_count = count;
    k.suggested_local_size = {count, 1, 1};
    const Operand gid = Operand::ref({kernels::kGlobalIndexValue});
    const auto value = tier1::lower_affine_generate(
        builder, ScalarType::F32, gid, Operand::f(2.5f), Operand::f(0.25f));
    append_store(builder, output, gid, value, ScalarType::F32);
    return k;
}
} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path dump_dir =
        argc == 3 && std::string(argv[1]) == "--dump-dir"
            ? std::filesystem::path(argv[2]) : std::filesystem::path{};
    if (!dump_dir.empty()) std::filesystem::create_directories(dump_dir);
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

    {
        const spirv::KernelIR k = build_sum_kernel(64);
        std::array<float, 64> input{};
        float output = -1.0f;
        for (std::size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(i + 1);
        }
        std::vector<kernels::EigenBufferBinding> buffers = {
            {input.data(), sizeof(input), false}, {&output, sizeof(output), true},
        };
        std::string error;
        if (!kernels::execute_kernel_eigen(k, buffers, &error) || output != 2080.0f) {
            std::cerr << "[TIER1] sum failed: " << error << " output=" << output << "\n";
            return 1;
        }
        if (!kernels::assemble_kernel_ir_to_spirv(k).complete()) {
            std::cerr << "[TIER1] sum did not assemble\n";
            return 1;
        }
    }

    {
        const spirv::KernelIR k = build_matmul_kernel(2, 3, 2);
        std::array<float, 6> left{1, 2, 3, 4, 5, 6};
        std::array<float, 6> right{7, 8, 9, 10, 11, 12};
        std::array<float, 4> output{};
        std::vector<kernels::EigenBufferBinding> buffers = {
            {left.data(), sizeof(left), false}, {right.data(), sizeof(right), false},
            {output.data(), sizeof(output), true},
        };
        std::string error;
        const std::array<float, 4> expected{58, 64, 139, 154};
        if (!kernels::execute_kernel_eigen(k, buffers, &error) || output != expected) {
            std::cerr << "[TIER1] matmul failed: " << error << "\n";
            return 1;
        }
        if (!kernels::assemble_kernel_ir_to_spirv(k).complete()) {
            std::cerr << "[TIER1] matmul did not assemble\n";
            return 1;
        }
    }

    {
        constexpr std::uint32_t count = 8;
        const spirv::KernelIR k = build_arange_kernel(count);
        std::array<float, count> output{};
        std::vector<kernels::EigenBufferBinding> buffers = {
            {output.data(), sizeof(output), true},
        };
        std::string error;
        if (!kernels::execute_kernel_eigen(k, buffers, &error)) {
            std::cerr << "[TIER1] affine generate failed: " << error << "\n";
            return 1;
        }
        for (std::size_t i = 0; i < output.size(); ++i) {
            const float expected = 2.5f + static_cast<float>(i) * 0.25f;
            if (output[i] != expected) {
                std::cerr << "[TIER1] affine generate mismatch at " << i << "\n";
                return 1;
            }
        }
        if (!kernels::assemble_kernel_ir_to_spirv(k).complete()) {
            std::cerr << "[TIER1] affine generate did not assemble\n";
            return 1;
        }
    }
    std::cout << "[TIER1] PASS: reduce, contract, and affine generate recipes\n";

    // Catalog honesty: every operation that claims to be one Tier-0 scalar
    // instruction must be accepted by both concrete authorities. This is the
    // executable capability table; a new catalog claim cannot outrun either
    // the direct SPIR-V assembler or its CPU differential reference.
    std::size_t covered = 0;
    for (const auto& op : ops::kOps) {
        if (!op.lowerable) continue;
        ScalarKernel fixture = build_scalar_kernel(op);
        const kernels::SpirvAssembly assembly =
            kernels::assemble_kernel_ir_to_spirv(fixture.kernel);
        if (!assembly.complete()) {
            std::cerr << "[CATALOG] SPIR-V refused " << op.name << ":";
            for (const auto& reason : assembly.shortfalls) std::cerr << " " << reason;
            std::cerr << "\n";
            return 1;
        }
        if (!dump_dir.empty()) {
            const auto path = dump_dir / (std::string(op.name) + ".spv");
            std::ofstream stream(path, std::ios::binary);
            stream.write(reinterpret_cast<const char*>(assembly.binary.words.data()),
                         static_cast<std::streamsize>(assembly.binary.words.size() *
                                                      sizeof(std::uint32_t)));
            if (!stream) {
                std::cerr << "[CATALOG] could not write " << path << "\n";
                return 1;
            }
        }
        std::string error;
        if (!kernels::execute_kernel_eigen(fixture.kernel, fixture.bindings, &error)) {
            std::cerr << "[CATALOG] CPU refused " << op.name << ": " << error << "\n";
            return 1;
        }
        ++covered;
    }
    if (covered != 58) {
        std::cerr << "[CATALOG] expected 58 Tier-0 operations, covered " << covered << "\n";
        return 1;
    }

    const auto* mod = ops::find_op("mod");
    const auto* floordiv = ops::find_op("floordiv");
    const auto* logical_and = ops::find_op("logical_and");
    const auto* logical_or = ops::find_op("logical_or");
    const auto* logical_not = ops::find_op("logical_not");
    const auto* sign = ops::find_op("sign");
    const auto* sigmoid = ops::find_op("sigmoid");
    if (!mod || !floordiv || !logical_and || !logical_or || !logical_not ||
        !sign || !sigmoid) {
        std::cerr << "[CATALOG] semantic probes are absent from the catalog\n";
        return 1;
    }
    if (std::fabs(execute_scalar(*mod, {-5.5f, 2.0f, 0.0f}) - 0.5f) > 1e-6f ||
        std::fabs(execute_scalar(*floordiv, {-5.5f, 2.0f, 0.0f}) + 3.0f) > 1e-6f ||
        execute_scalar(*logical_and, {-2.0f, 0.0f, 0.0f}) != 0.0f ||
        execute_scalar(*logical_or, {0.0f, -2.0f, 0.0f}) != 1.0f ||
        execute_scalar(*logical_not, {0.0f, 0.0f, 0.0f}) != 1.0f ||
        execute_scalar(*sign, {-2.0f, 0.0f, 0.0f}) != -1.0f ||
        std::fabs(execute_scalar(*sigmoid, {0.0f, 0.0f, 0.0f}) - 0.5f) > 1e-6f) {
        std::cerr << "[CATALOG] scalar semantic concordance failed\n";
        return 1;
    }
    std::cout << "[CATALOG] PASS: " << covered
              << " Tier-0 operations accepted by CPU and SPIR-V; semantic probes agree\n";

    return 0;
}
