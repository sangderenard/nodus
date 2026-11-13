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
#include <array>
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

// Physical-device identity from the SAME Vulkan enumeration used for context
// creation. Ordinals are snapshot-local; compare UUID/LUID after reselection.
// No claim is made about OpenGL/SDL adapter selection or other API ordinals.
struct VulkanDeviceInfo {
    std::uint32_t index = 0;
    std::string name;
    std::uint32_t vendor_id = 0, device_id = 0, device_type = 0;
    std::uint32_t api_version = 0, driver_version = 0;
    std::array<std::uint8_t, 16> uuid{};
    std::array<std::uint8_t, 8> luid{};
    bool uuid_valid = false, luid_valid = false, compute = false;
    std::uint32_t queue_family = UINT32_MAX;
    std::uint32_t available_compute_queues = 0;
    std::uint32_t max_storage_buffers = 0, max_storage_buffer_bytes = 0;
};

struct VulkanMemoryHeapStatus {
    std::uint64_t size_bytes = 0, budget_bytes = 0, usage_bytes = 0;
    bool device_local = false;
};

struct VulkanBackendStatus {
    VulkanDeviceInfo device;
    std::uint32_t active_compute_queues = 0;
    // VK_EXT_memory_budget estimates, only valid when budget_available is true.
    // These are NOT GPU utilization percentages or process RAM measurements.
    bool budget_available = false;
    std::vector<VulkanMemoryHeapStatus> heaps;
};

struct VulkanBufferAllocationStatus {
    std::uint32_t heap_index = UINT32_MAX;
    std::uint64_t allocation_bytes = 0;
    bool host_visible = false;
    bool device_local = false;
};

struct VulkanPipelineCacheStats {
    std::uint64_t resident = 0, hits = 0, created = 0;
};

// Creates an instance for physical queries only; no logical device, shader,
// buffer, command pool, dispatch or rendering is created by enumeration.
bool enumerate_vulkan_devices(std::vector<VulkanDeviceInfo>* devices,
                              std::string* error);
// Same read-only instance/query lifecycle, without creating a logical device.
bool query_vulkan_device_status(std::uint32_t index, VulkanBackendStatus* status,
                                std::string* error);
// Uses the existing persistent GP_MEM_BACKEND_VULKAN implementation. Errors
// retain the selected device and failed operation/result; no implicit retry.
gp_mem_backend_handle_t create_vulkan_backend(
    const gp_mem_backend_vulkan_params_t* params,
    VulkanDeviceInfo* selected, std::string* error);
// Read-only identity/heap snapshot of an existing context or its buffer.
bool query_vulkan_backend_status(gp_mem_backend_handle_t backend,
                                 VulkanBackendStatus* status,
                                 std::string* error);
// Exact heap and allocation size of a live Vulkan buffer. Use this with the
// matching backend budget snapshot to account for resident state.
bool query_vulkan_buffer_allocation(gp_mem_backend_handle_t buffer,
                                    VulkanBufferAllocationStatus* status,
                                    std::string* error);
bool query_vulkan_pipeline_cache_stats(gp_mem_backend_handle_t backend,
                                       VulkanPipelineCacheStats* stats,
                                       std::string* error);

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

// Explicit device selection for the existing host-buffer entry. Failures
// return to the caller, never trigger another device or workload replay.
bool dispatch_spirv_on_device(
    const std::vector<std::uint32_t>& spirv, const char* entry_name,
    std::vector<VulkanBufferBinding>& buffers, std::uint32_t group_count_x,
    const gp_mem_backend_vulkan_params_t* params, std::string* error);

// Dispatch directly against persistent Vulkan backend buffers. All bindings
// must originate from one compatible Vulkan context.
bool dispatch_spirv_backends(
    const std::vector<std::uint32_t>& spirv,
    const char* entry_name,
    const std::vector<VulkanBackendBinding>& buffers,
    std::uint32_t group_count_x,
    std::string* error);

// Two-dimensional compute dispatch for tiled kernels such as authored T5.
// Existing one-dimensional callers continue to use dispatch_spirv_backends.
bool dispatch_spirv_backends_2d(
    const std::vector<std::uint32_t>& spirv,
    const char* entry_name,
    const std::vector<VulkanBackendBinding>& buffers,
    std::uint32_t group_count_x,
    std::uint32_t group_count_y,
    std::string* error);

// Keep one shader/buffer loadout resident for `steps` dispatches. Between
// dispatches, copy the completed position/velocity outputs into the current
// input columns behind a transfer-to-compute barrier. One queue submit and
// one completion fence cover the entire contract quantum.
bool dispatch_spirv_resident_loop(
    const std::vector<std::uint32_t>& spirv, const char* entry_name,
    const std::vector<VulkanBackendBinding>& buffers,
    std::uint32_t group_count_x, std::uint32_t steps,
    std::uint32_t current_position, std::uint32_t current_velocity,
    std::uint32_t next_position, std::uint32_t next_velocity,
    std::string* error);

} // namespace kernels
} // namespace nodus
