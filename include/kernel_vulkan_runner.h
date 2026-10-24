// kernel_vulkan_runner.h
//
// Execute an assembled SPIR-V compute kernel on a real device.
//
// This closes nodus's own loop: kernel_spirv.cpp assembles KernelIR into a
// SPIR-V module, and this dispatches that module on Vulkan and reads the
// results back. Until now nodus could produce device code but never run it,
// so "the kernel is correct" rested on spirv-val's structural verdict alone.
//
// Buffer binding follows the assembler's own convention exactly: descriptor
// set 0, binding i = the i-th entry of KernelIR::buffer_value_ids, each a
// Block-decorated struct holding one runtime array. Callers therefore pass
// buffers in that same order.
//
// Built only when the Vulkan SDK is present (NODUS_HAVE_VULKAN); otherwise
// vulkan_available() answers false and dispatch reports a named reason, so a
// machine without a device degrades to a clear skip rather than a build break.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mem_backend.h"

namespace nodus {
namespace kernels {

struct VulkanBufferBinding {
    void* data = nullptr;      // host memory: uploaded before, read back after
    std::size_t bytes = 0;
    bool readback = false;     // copy device contents back into `data` after
};

// Persistent backend binding. `backend` must be a buffer allocated from a
// GP_MEM_BACKEND_VULKAN context. Offsets make FIFO ring slots bindable without
// copying them into a temporary Vulkan allocation.
struct VulkanBackendBinding {
    gp_mem_backend_handle_t backend = nullptr;
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

// True when a Vulkan instance and a compute-capable physical device exist.
bool vulkan_available();

// Human-readable device name, or "" when unavailable.
std::string vulkan_device_name();

// Dispatch `spirv` (a complete module) with `entry_name` as its entry point.
// `group_count_x` workgroups are launched; the module's own OpExecutionMode
// LocalSize fixes the workgroup shape. Returns false and fills `error` on any
// failure. Buffers are host-visible and coherent -- no staging -- which keeps
// this runner small enough to audit.
bool dispatch_spirv(const std::vector<std::uint32_t>& spirv,
                    const char* entry_name,
                    std::vector<VulkanBufferBinding>& buffers,
                    std::uint32_t group_count_x,
                    std::string* error);

// Dispatch directly against persistent Vulkan backend buffers. All bindings
// must originate from one compatible Vulkan context.
bool dispatch_spirv_backends(
    const std::vector<std::uint32_t>& spirv,
    const char* entry_name,
    const std::vector<VulkanBackendBinding>& buffers,
    std::uint32_t group_count_x,
    std::string* error);

} // namespace kernels
} // namespace nodus
