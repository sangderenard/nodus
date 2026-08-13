// kernel_spirv_assembler.h
// Direct KernelIR -> SPIR-V binary assembly, with no external toolchain.
//
// This is the counterpart of SpirvTranslator (spirv_translation.h), which routes
// KernelIR through generated GLSL and an external glslangValidator process. The
// two paths are deliberately both kept: the external-compiler route is the
// differential reference; this assembler owns the words itself, so a machine
// without the Vulkan SDK can still produce a runnable module (the same reason
// turing's wasm_binary.py assembles WebAssembly directly instead of requiring
// wat2wasm).
//
// Fail-closed discipline: an operation, type, or value shape outside the
// assembler's contract is reported by name in `shortfalls` and the result is
// marked incomplete. Nothing is guessed, approximated, or silently dropped.

#pragma once

#include <string>
#include <vector>

#include "spirv_translation.h" // SpirvBinary

namespace nodus {
namespace spirv {
struct KernelIR;
}

namespace kernels {

// A KernelIR reserves no value id for "the element index", but an element-wise
// kernel must address its buffers by one. When element_count > 0 the assembler
// publishes gl_GlobalInvocationID.x under this sentinel value id, so an ADDR
// instruction reaches it with an ordinary ValueRef{kGlobalIndexValue} operand.
inline constexpr std::uint32_t kGlobalIndexValue = 0xFFFFFFFFu;

struct SpirvAssembly {
    spirv::SpirvBinary binary{};
    // Named refusals. Empty means every instruction of the kernel was assembled.
    std::vector<std::string> shortfalls;
    // SPIR-V id bound actually used (for diagnostics).
    std::uint32_t id_bound = 0;

    bool complete() const { return shortfalls.empty(); }
};

// Assemble one KernelIR compute kernel into a SPIR-V 1.3 (Vulkan 1.1) module:
// OpEntryPoint GLCompute, LocalSize from suggested_local_size, one storage
// buffer per entry of buffer_value_ids (descriptor set 0, binding = position),
// and a gl_GlobalInvocationID.x bounds guard when element_count > 0.
// Values listed in buffer_value_ids must be is_buffer ValueDefs; everything
// else is straight-line SSA lowered in instruction order.
SpirvAssembly assemble_kernel_ir_to_spirv(const spirv::KernelIR& kernel);

// Register this assembler in the TranslationMatrix under the name "spirv".
// Returns true when (re)registered. The registered function assembles and
// fails loudly (throws std::runtime_error naming the shortfalls) rather than
// letting an incomplete module masquerade as a lowered kernel.
bool register_spirv_backend();

} // namespace kernels
} // namespace nodus
