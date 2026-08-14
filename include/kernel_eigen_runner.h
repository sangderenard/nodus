// kernel_eigen_runner.h
//
// Execute a KernelIR on the CPU, so a device result has something to be right
// or wrong against.
//
// kernel_vulkan_runner.h runs the assembled SPIR-V on a real GPU; this runs the
// same KernelIR here, instruction by instruction, and the two must agree. That
// makes it the differential reference for the assembler -- and, on a machine
// with no Vulkan device at all, the only way a kernel's numbers get checked
// rather than merely validated for structure.
//
// Buffer binding follows the assembler's convention exactly: buffers[i]
// corresponds to KernelIR::buffer_value_ids[i], the same order the Vulkan
// runner takes, so a caller can hand the identical vector to either path.
//
// Fail-closed like the assembler: an opcode, sub_op, type or value shape
// outside the interpreter's contract is named in `error` and execution stops.
// Nothing is guessed or approximated -- a wrong number here would discredit
// the device result it exists to check.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace nodus {
namespace spirv {
struct KernelIR;
}

namespace kernels {

struct EigenBufferBinding {
    void* data = nullptr;      // read and written in place
    std::size_t bytes = 0;
    // Carried for parity with VulkanBufferBinding so the same bindings serve
    // both runners. The CPU path works directly in host memory, so there is no
    // device copy to bring back and this field has no effect here.
    bool readback = false;
};

// Interpret `kernel` once per invocation for gid in [0, element_count), in
// order. Returns false and fills `error` (when non-null) on the first refusal.
bool execute_kernel_eigen(const spirv::KernelIR& kernel,
                          std::vector<EigenBufferBinding>& buffers,
                          std::string* error);

} // namespace kernels
} // namespace nodus
