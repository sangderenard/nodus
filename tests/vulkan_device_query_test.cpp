// CPU-only native regression: physical inventory and context API calls below
// are fake. Forbidden shader/queue operations abort, never reach the driver.
#include "kernel_vulkan_runner.h"
#include "mem_backend_registry_internal.h"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace {
int instances = 0, destroyed_instances = 0, device_calls = 0;
int destroyed_devices = 0, pool_calls = 0, lists = 0;
bool incomplete = false, budget = true, throw_register = false;
VkResult instance_result = VK_SUCCESS, device_result = VK_SUCCESS;
VkResult list_result = VK_SUCCESS, pool_result = VK_SUCCESS;
std::uint32_t last_device = UINT32_MAX;
std::unordered_map<void*, const gp_mem_backend_vtable_t*> handles;
template<class T> T id(std::uintptr_t value) { return reinterpret_cast<T>(value); }
std::uint32_t ordinal(VkPhysicalDevice d) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(d) - 1);
}
void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}
[[noreturn]] void forbidden() { std::abort(); }
}

extern "C" {
VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo*,
    const VkAllocationCallbacks*, VkInstance* out) {
    ++instances;
    if (instance_result == VK_SUCCESS) *out = id<VkInstance>(1);
    return instance_result;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance, const VkAllocationCallbacks*) {
    ++destroyed_instances;
}
VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(VkInstance,
    std::uint32_t* count, VkPhysicalDevice* out) {
    if (!out) { *count = 3; return VK_SUCCESS; }
    ++lists;
    if (incomplete) return VK_INCOMPLETE;
    if (list_result != VK_SUCCESS) return list_result;
    for (std::uint32_t i = 0; i < 3; ++i) out[i] = id<VkPhysicalDevice>(i + 1);
    *count = 3;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(VkPhysicalDevice d,
    VkPhysicalDeviceProperties* out) {
    *out = {};
    out->apiVersion = VK_API_VERSION_1_1;
    out->vendorID = 0x10de;
    out->deviceID = 100 + ordinal(d);
    out->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    std::strcpy(out->deviceName, ordinal(d) == 0 ? "no compute" :
                               ordinal(d) == 1 ? "GPU one" : "GPU two");
    out->limits.maxPerStageDescriptorStorageBuffers = 16;
    out->limits.maxDescriptorSetStorageBuffers = 32;
    out->limits.maxStorageBufferRange = 1u << 27;
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(VkPhysicalDevice d,
    VkPhysicalDeviceProperties2* out) {
    vkGetPhysicalDeviceProperties(d, &out->properties);
    auto* ids = static_cast<VkPhysicalDeviceIDProperties*>(out->pNext);
    ids->deviceUUID[0] = static_cast<std::uint8_t>(ordinal(d) + 20);
    ids->deviceLUID[0] = static_cast<std::uint8_t>(ordinal(d) + 40);
    ids->deviceLUIDValid = VK_TRUE;
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice d,
    std::uint32_t* count, VkQueueFamilyProperties* out) {
    if (!out) { *count = 2; return; }
    out[0] = {}; out[1] = {};
    out[0].queueFlags = VK_QUEUE_COMPUTE_BIT; // zero queueCount is not usable
    out[1].queueCount = 1;
    out[1].queueFlags = ordinal(d) ? VK_QUEUE_COMPUTE_BIT : VK_QUEUE_GRAPHICS_BIT;
    *count = 2;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice d,
    const VkDeviceCreateInfo* info, const VkAllocationCallbacks*, VkDevice* out) {
    ++device_calls; last_device = ordinal(d);
    require(info->pQueueCreateInfos[0].queueFamilyIndex == 1, "wrong queue family");
    if (device_result == VK_SUCCESS) *out = id<VkDevice>(1);
    return device_result;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice, const VkAllocationCallbacks*) {
    ++destroyed_devices;
}
VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice, std::uint32_t, std::uint32_t,
    VkQueue* out) { *out = id<VkQueue>(1); }
VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(VkDevice,
    const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool* out) {
    ++pool_calls;
    if (pool_result == VK_SUCCESS) *out = id<VkCommandPool>(1);
    return pool_result;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice, VkCommandPool,
    const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice,
    const char*, std::uint32_t* count, VkExtensionProperties* out) {
    *count = budget ? 1 : 0;
    if (out && budget) std::strcpy(out[0].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties2(VkPhysicalDevice,
    VkPhysicalDeviceMemoryProperties2* out) {
    out->memoryProperties = {};
    out->memoryProperties.memoryHeapCount = 1;
    out->memoryProperties.memoryHeaps[0] = {4096, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    if (out->pNext) {
        auto* b = static_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(out->pNext);
        b->heapBudget[0] = 3072; b->heapUsage[0] = 1024;
    }
}
// Never allow a regression to turn this CPU test into a GPU workload.
VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice, const VkBufferCreateInfo*,
    const VkAllocationCallbacks*, VkBuffer*) { forbidden(); }
VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice,
    const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule*) { forbidden(); }
VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice, VkPipelineCache,
    std::uint32_t, const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline*) { forbidden(); }
VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue, std::uint32_t,
    const VkSubmitInfo*, VkFence) { forbidden(); }

const gp_mem_backend_vtable_t* gp_mem_backend_get_vtable(gp_mem_backend_handle_t h) {
    auto i = handles.find(h); return i == handles.end() ? nullptr : i->second;
}
int gp_mem_backend_get_type(gp_mem_backend_handle_t h) {
    return handles.count(h) ? GP_MEM_BACKEND_VULKAN : -1;
}
void gp_mem_backend_release(gp_mem_backend_handle_t h) {
    auto* table = gp_mem_backend_get_vtable(h); if (table) table->free(h);
}
}
namespace nodus::mem_backend_internal {
void register_external_handle(gp_mem_backend_handle_t h, GP_MemBackendType,
    const gp_mem_backend_vtable_t* vtable, std::size_t, std::uint64_t) {
    handles[h] = vtable;
    if (throw_register) throw std::bad_alloc(); // failure after a partial insert
}
void unregister_external_handle(gp_mem_backend_handle_t h) { handles.erase(h); }
void update_external_size(gp_mem_backend_handle_t, std::size_t) {}
}

int main() try {
    using namespace nodus::kernels;
    std::string error = "stale";
    std::vector<VulkanDeviceInfo> devices;
    require(enumerate_vulkan_devices(&devices, &error), "enumeration failed");
    require(error.empty() && devices.size() == 3, "stale error or count");
    require(devices[0].index == 0 && !devices[0].compute, "filtered raw ordinal");
    require(devices[2].index == 2 && devices[2].compute && devices[2].queue_family == 1,
            "wrong compute mapping");
    require(devices[2].uuid_valid && devices[2].uuid[0] == 22 &&
            devices[2].luid_valid && devices[2].luid[0] == 42, "identity lost");
    require(devices[2].max_storage_buffers == 16, "limit intersection wrong");
    require(device_calls == 0 && pool_calls == 0, "enumeration created a device");
    incomplete = true; const auto start_lists = lists;
    require(!enumerate_vulkan_devices(&devices, &error) && devices.empty(), "partial inventory escaped");
    require(lists - start_lists == 3 && error.find("VK_INCOMPLETE") != error.npos,
            "unbounded inventory retry");
    incomplete = false;
    list_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    require(!enumerate_vulkan_devices(&devices, &error) &&
            error.find("out of host memory") != error.npos, "list error lost");
    list_result = VK_SUCCESS;
    instance_result = VK_ERROR_INCOMPATIBLE_DRIVER;
    require(!enumerate_vulkan_devices(&devices, &error) &&
            error.find("vkCreateInstance") != error.npos, "instance error lost");
    instance_result = VK_SUCCESS;

    VulkanBackendStatus status;
    require(query_vulkan_device_status(2, &status, &error), "physical status failed");
    require(status.device.index == 2 && status.budget_available &&
            status.heaps[0].usage_bytes == 1024 && status.heaps[0].budget_bytes == 3072,
            "budget identity mismatch");
    require(device_calls == 0, "physical query created device");
    budget = false;
    require(query_vulkan_device_status(2, &status, &error) && !status.budget_available &&
            status.heaps[0].budget_bytes == 0, "unsupported budget fabricated");
    require(!query_vulkan_device_status(99, &status, &error) && status.heaps.empty(),
            "invalid index or stale status");
    require(!query_vulkan_backend_status(id<void*>(99), &status, &error), "unregistered handle accepted");

    gp_mem_backend_vulkan_params_t params{99, 0};
    require(!gp_mem_backend_create_vulkan(&params) && device_calls == 0,
            "invalid device silently fell back");
    require(std::strstr(gp_mem_backend_vulkan_last_error(), "outside enumeration"), "C diagnostic lost");
    params = {0, 0};
    require(!gp_mem_backend_create_vulkan(&params) && device_calls == 0,
            "noncompute device silently fell back");
    require(std::strstr(gp_mem_backend_vulkan_last_error(), "no compute queue"), "queue diagnostic lost");
    params = {2, 2};
    const auto before_flags = instances;
    require(!gp_mem_backend_create_vulkan(&params) && instances == before_flags,
            "reserved flags created a context");
    params = {2, 0};
    VulkanDeviceInfo selected;
    auto handle = create_vulkan_backend(&params, &selected, &error);
    require(handle && last_device == 2 && selected.index == 2 && error.empty(), "exact selection failed");
    require(query_vulkan_backend_status(handle, &status, &error) && status.device.index == 2,
            "context identity changed");
    gp_mem_backend_release(handle);
    handle = gp_mem_backend_create_vulkan(nullptr);
    require(handle && last_device == 1, "default did not choose first compute raw ordinal");
    gp_mem_backend_release(handle);
    device_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    const auto before_failure = device_calls;
    require(!gp_mem_backend_create_vulkan(nullptr) && device_calls == before_failure + 1,
            "unexpected alternate-device retry");
    require(std::strstr(gp_mem_backend_vulkan_last_error(), "GPU one") &&
            std::strstr(gp_mem_backend_vulkan_last_error(), "vkCreateDevice: out of host memory"),
            "device operation/result missing");
    device_result = VK_SUCCESS; pool_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    const auto before_cleanup = destroyed_devices;
    require(!gp_mem_backend_create_vulkan(&params) && destroyed_devices == before_cleanup + 1,
            "failed partial context leaked");
    pool_result = VK_SUCCESS; throw_register = true;
    require(!gp_mem_backend_create_vulkan(&params), "exception escaped C API");
    throw_register = false;
    std::vector<VulkanBufferBinding> bindings;
    const auto before_invalid = instances;
    require(!dispatch_spirv_on_device({}, "main", bindings, 1, &params, &error) &&
            instances == before_invalid, "invalid request initialized device");
    require(handles.empty(), "context handles leaked");
    require(instances == destroyed_instances + 1, "instance leak (one creation deliberately failed)");
    std::cout << "PASS: Vulkan inventory, exact selection, failures, identity and budgets (mock API only)\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n'; return 1;
}
