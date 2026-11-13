// Explicit opt-in integration verification of kernel_vulkan_runner's actual
// physical inventory. NOT a selector/service. Only instances/physical queries
// reach the driver; executable-local guards prohibit device/workload entry.
#include "kernel_vulkan_runner.h"
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
unsigned opened = 0, closed = 0;
[[noreturn]] void forbidden(const char* operation) {
    std::fprintf(stderr, "FORBIDDEN non-query operation: %s\n", operation);
    std::fflush(stderr);
    std::_Exit(97);
}
template<std::size_t N> std::string hex_bytes(const std::array<std::uint8_t, N>& bytes) {
    std::ostringstream text;
    for (auto byte : bytes) text << std::hex << std::setfill('0') << std::setw(2)
                                 << static_cast<unsigned>(byte);
    return text.str();
}
}

extern "C" {
VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo* info,
    const VkAllocationCallbacks* allocator, VkInstance* result) {
    auto create = reinterpret_cast<PFN_vkCreateInstance>(
        vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!create) return VK_ERROR_INITIALIZATION_FAILED;
    const auto status = create(info, allocator, result);
    if (status == VK_SUCCESS) ++opened;
    return status;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance instance,
    const VkAllocationCallbacks* allocator) {
    auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(
        vkGetInstanceProcAddr(instance, "vkDestroyInstance"));
    if (!destroy) forbidden("missing vkDestroyInstance");
    destroy(instance, allocator);
    ++closed;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice,
    const VkDeviceCreateInfo*, const VkAllocationCallbacks*, VkDevice*) {
    forbidden("vkCreateDevice");
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(VkDevice,
    const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool*) {
    forbidden("vkCreateCommandPool");
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice, const VkBufferCreateInfo*,
    const VkAllocationCallbacks*, VkBuffer*) { forbidden("vkCreateBuffer"); }
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice, const VkMemoryAllocateInfo*,
    const VkAllocationCallbacks*, VkDeviceMemory*) { forbidden("vkAllocateMemory"); }
VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice,
    const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule*) {
    forbidden("vkCreateShaderModule");
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice, VkPipelineCache,
    std::uint32_t, const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline*) {
    forbidden("vkCreateComputePipelines");
}
VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue, std::uint32_t,
    const VkSubmitInfo*, VkFence) { forbidden("vkQueueSubmit"); }
VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer, std::uint32_t,
    std::uint32_t, std::uint32_t) { forbidden("vkCmdDispatch"); }
}

int main(int argc, char** argv) {
    if (argc != 2 || std::string(argv[1]) != "--physical-queries-only") return 64;
    using namespace nodus::kernels;
    std::cout << std::unitbuf;
    std::vector<VulkanDeviceInfo> devices;
    std::string error;
    bool ok = enumerate_vulkan_devices(&devices, &error);
    if (!ok) std::cerr << "enumeration_error=" << error << '\n';
    for (const auto& device : devices) {
        std::cout << "device index=" << device.index
            << " name=" << std::quoted(device.name)
            << " vendor_id=" << device.vendor_id << " device_id=" << device.device_id
            << " device_type=" << device.device_type
            << " api_version=" << device.api_version << " driver_version=" << device.driver_version
            << " uuid_valid=" << device.uuid_valid << " uuid=" << hex_bytes(device.uuid)
            << " luid_valid=" << device.luid_valid << " luid=" << hex_bytes(device.luid)
            << " compute=" << device.compute << " queue_family=" << device.queue_family
            << " max_storage_buffers=" << device.max_storage_buffers
            << " max_storage_buffer_bytes=" << device.max_storage_buffer_bytes << '\n';
        VulkanBackendStatus status;
        if (!query_vulkan_device_status(device.index, &status, &error)) {
            std::cerr << "status_error index=" << device.index << " error=" << error << '\n';
            ok = false;
            continue;
        }
        if (status.device.name != device.name || status.device.uuid != device.uuid ||
            status.device.luid != device.luid) {
            std::cerr << "inventory_changed index=" << device.index << '\n';
            ok = false;
            continue;
        }
        std::cout << "memory index=" << device.index
                  << " budget_available=" << status.budget_available << '\n';
        for (std::size_t i = 0; i < status.heaps.size(); ++i) {
            const auto& heap = status.heaps[i];
            std::cout << "heap index=" << device.index << " heap=" << i
                << " device_local=" << heap.device_local << " size_bytes=" << heap.size_bytes
                << " budget_bytes=" << heap.budget_bytes << " usage_bytes=" << heap.usage_bytes << '\n';
        }
    }
    std::cout << "cleanup instances_created=" << opened << " instances_destroyed=" << closed
              << " logical_devices=0 shaders=0 pipelines=0 submissions=0\n";
    return ok && opened == closed ? 0 : 1;
}
