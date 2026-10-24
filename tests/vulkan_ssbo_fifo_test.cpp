// Two shader rows communicate through the ordinary EdgeTensorFifo while all
// payload bytes remain in persistent Vulkan SSBO allocations.
#include "bitops_lowering.h"
#include "edge_tensor_fifo.h"
#include "kernel_spirv_assembler.h"
#include "kernel_vulkan_runner.h"
#include "table_abi.h"

#include <array>
#include <cstdint>
#include <iostream>
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
constexpr std::size_t kBytes = kElements * sizeof(std::uint32_t);

spirv::KernelIR gray_kernel(const char* name) {
    bitops::KernelIrBuilder b(name);
    auto& k = b.kernel();
    const auto input = b.new_scalar(ScalarType::U32, "input");
    const auto output = b.new_scalar(ScalarType::U32, "output");
    k.values[input.id].type.is_buffer = true;
    k.values[input.id].type.is_readonly = true;
    k.values[output.id].type.is_buffer = true;
    k.buffer_value_ids = {input.id, output.id};
    k.element_count = kElements;
    k.suggested_local_size = {kElements, 1, 1};
    const ValueRef gid{kernels::kGlobalIndexValue};
    const auto input_ptr = b.new_scalar(ScalarType::U32, "input_ptr");
    k.instrs.push_back({OpCode::ADDR, -1,
                        {Operand::ref(input), Operand::ref(gid)}, {input_ptr}});
    const auto value = b.new_scalar(ScalarType::U32, "value");
    k.instrs.push_back({OpCode::LOAD, -1, {Operand::ref(input_ptr)}, {value}});
    const auto gray = bitops::lower_int_to_gray_u32(b, Operand::ref(value));
    const auto output_ptr = b.new_scalar(ScalarType::U32, "output_ptr");
    k.instrs.push_back({OpCode::ADDR, -1,
                        {Operand::ref(output), Operand::ref(gid)}, {output_ptr}});
    k.instrs.push_back({OpCode::STORE, -1,
                        {Operand::ref(output_ptr), Operand::ref(gray)}, {}});
    return k;
}

bool bind_frame_to_buffer(RawStackFrame* frame, gp_mem_backend_handle_t buffer,
                          ValueTypeId type, bool full) {
    if (!raw_stack_init_frame(*frame, kBytes, type)) return false;
    gp_mem_backend_release(frame->backend);
    frame->backend = buffer;
    frame->byte_count = full ? kBytes : 0;
    for (std::size_t i = 0; i < kBytes; ++i) frame->types_per_byte[i] = type;
    return true;
}
} // namespace

int main() {
    if (!kernels::vulkan_available()) {
        std::cout << "[VK-SSBO-FIFO] skipped: no Vulkan compute device\n";
        return 0;
    }
    gp_mem_backend_handle_t context = gp_mem_backend_create_vulkan(nullptr);
    if (!context) return 1;
    const auto* vtable = gp_mem_backend_get_vtable(context);
    if (!vtable || !vtable->alloc ||
        !(gp_mem_backend_get_capabilities(context) & GP_MEM_CAP_D2D_NATIVE)) {
        std::cerr << "Vulkan backend does not advertise native D2D\n";
        return 2;
    }
    auto input = vtable->alloc(context, kBytes, 0);
    auto row_a_output = vtable->alloc(context, kBytes, 0);
    auto row_b_input = vtable->alloc(context, kBytes, 0);
    auto row_b_output = vtable->alloc(context, kBytes, 0);
    if (!input || !row_a_output || !row_b_input || !row_b_output) return 3;

    // The public table boundary must re-home an already configured edge's
    // ring allocation, not merely retain a Vulkan context hint beside host
    // storage.  This is what graph materialization calls.
    GP_TableContext* table = gp_table_create(nullptr);
    if (!table || !gp_table_add_edge(table, 9001, 9002)) return 31;
    GP_TableEdgeTensorSpecTyped spec{};
    spec.dims[0] = static_cast<std::int32_t>(kElements);
    spec.dim_count = 1;
    spec.slots = 2;
    spec.elem_size = sizeof(std::uint32_t);
    spec.type_id = 0x5353424f;
    spec.layout = static_cast<std::int32_t>(nodus::tensors::TensorLayout::Dense);
    spec.dtype = static_cast<std::int32_t>(nodus::tensors::TensorDType::U32);
    if (!gp_table_edge_set_tensor_spec(table, 0, &spec) ||
        !gp_table_edge_set_backend(table, 0, context) ||
        gp_mem_backend_get_type(gp_table_edge_get_storage_backend(table, 0)) !=
            GP_MEM_BACKEND_VULKAN) return 32;

    gp_table_destroy(table);

    std::array<std::uint32_t, kElements> source{};
    for (std::uint32_t i = 0; i < kElements; ++i) source[i] = i;
    if (!vtable->copy_to_backend(input, 0, source.data(), kBytes)) return 4;

    const auto row_a = gray_kernel("shader_row_a");
    const auto row_b = gray_kernel("shader_row_b");
    const auto spirv_a = kernels::assemble_kernel_ir_to_spirv(row_a);
    const auto spirv_b = kernels::assemble_kernel_ir_to_spirv(row_b);
    if (!spirv_a.complete() || !spirv_b.complete()) return 5;
    std::string error;
    if (!kernels::dispatch_spirv_backends(
            spirv_a.binary.words, row_a.name.c_str(),
            {{input, 0, kBytes}, {row_a_output, 0, kBytes}}, 1, &error)) {
        std::cerr << "shader row A failed: " << error << '\n';
        return 6;
    }

    const ValueTypeId packet_type =
        ValueTypeRegistry::global().register_primitive("vulkan_u32x64", kBytes);
    constexpr std::uint64_t writer = 101;
    constexpr std::uint64_t reader = 202;
    {
        EdgeTensorFifo fifo;
        fifo.set_backend(context);
        fifo.configure({static_cast<std::int32_t>(kElements)}, 2, 0,
                       sizeof(std::uint32_t), packet_type);
        if (gp_mem_backend_get_type(fifo.impl->storage_handle) !=
            GP_MEM_BACKEND_VULKAN) return 7;
        if (!fifo.subscribe(reader, true)) return 8;

        auto* producer = new RawStackFrame();
        if (!bind_frame_to_buffer(producer, row_a_output, packet_type, true)) return 9;
        bool dropped = false;
        if (!fifo.push_from_frame(77, writer, producer, packet_type, &dropped) ||
            dropped || producer->byte_count != 0) return 10;
        raw_stack_destroy_frame(producer); // owns row_a_output
        row_a_output = nullptr;

        auto* consumer = new RawStackFrame();
        if (!bind_frame_to_buffer(consumer, row_b_input, packet_type, false)) return 11;
        if (!fifo.pop_to_frame(reader, consumer, packet_type) ||
            consumer->byte_count != kBytes) return 12;

        if (!kernels::dispatch_spirv_backends(
                spirv_b.binary.words, row_b.name.c_str(),
                {{consumer->backend, 0, kBytes}, {row_b_output, 0, kBytes}},
                1, &error)) {
            std::cerr << "shader row B failed: " << error << '\n';
            return 13;
        }
        raw_stack_destroy_frame(consumer); // owns row_b_input
        row_b_input = nullptr;
    }

    std::array<std::uint32_t, kElements> observed{};
    if (!vtable->copy_from_backend(row_b_output, 0, observed.data(), kBytes)) return 14;
    for (std::uint32_t i = 0; i < kElements; ++i) {
        const auto once = i ^ (i >> 1);
        const auto expected = once ^ (once >> 1);
        if (observed[i] != expected) {
            std::cerr << "mismatch at " << i << ": got " << observed[i]
                      << " expected " << expected << '\n';
            return 15;
        }
    }
    gp_mem_backend_release(input);
    gp_mem_backend_release(row_b_output);
    gp_mem_backend_release(context);
    std::cout << "[VK-SSBO-FIFO] PASS: shader A -> Vulkan EdgeTensorFifo -> "
                 "shader B, 256-byte packet remained D2D\n";
    return 0;
}
