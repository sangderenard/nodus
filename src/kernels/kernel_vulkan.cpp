// Persistent Vulkan storage buffers and KernelIR SPIR-V dispatch.
#include "kernel_vulkan_runner.h"
#include "mem_backend_registry_internal.h"

#if defined(NODUS_HAVE_VULKAN)
#include <vulkan/vulkan.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <exception>

namespace nodus::kernels {

namespace {
thread_local std::string creation_error;
thread_local const char* creation_fatal = nullptr;
bool fail_with(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}
}

#if !defined(NODUS_HAVE_VULKAN)

bool vulkan_available() { return false; }
std::string vulkan_device_name() { return {}; }
bool enumerate_vulkan_devices(std::vector<VulkanDeviceInfo>* devices,
                              std::string* error) {
    if (devices) devices->clear();
    return fail_with(error, "built without Vulkan (NODUS_HAVE_VULKAN undefined)");
}
gp_mem_backend_handle_t create_vulkan_backend(
    const gp_mem_backend_vulkan_params_t*, VulkanDeviceInfo* selected,
    std::string* error) {
    creation_fatal = nullptr;
    if (selected) *selected = {};
    creation_error = "built without Vulkan (NODUS_HAVE_VULKAN undefined)";
    fail_with(error, creation_error);
    return nullptr;
}
bool query_vulkan_backend_status(gp_mem_backend_handle_t,
                                 VulkanBackendStatus* status, std::string* error) {
    if (status) *status = {};
    return fail_with(error, "built without Vulkan (NODUS_HAVE_VULKAN undefined)");
}
bool query_vulkan_buffer_allocation(gp_mem_backend_handle_t,
                                    VulkanBufferAllocationStatus* status,
                                    std::string* error) {
    if (status) *status = {};
    return fail_with(error, "built without Vulkan (NODUS_HAVE_VULKAN undefined)");
}
bool query_vulkan_pipeline_cache_stats(gp_mem_backend_handle_t,
                                       VulkanPipelineCacheStats* stats,
                                       std::string* error) {
    if (stats) *stats = {};
    return fail_with(error, "built without Vulkan (NODUS_HAVE_VULKAN undefined)");
}
bool query_vulkan_device_status(std::uint32_t, VulkanBackendStatus* status,
                                std::string* error) {
    return query_vulkan_backend_status(nullptr, status, error);
}
bool dispatch_spirv_on_device(const std::vector<std::uint32_t>&, const char*,
                              std::vector<VulkanBufferBinding>&, std::uint32_t,
                              const gp_mem_backend_vulkan_params_t*,
                              std::string* error) {
    return fail_with(error, "built without Vulkan (NODUS_HAVE_VULKAN undefined)");
}
bool dispatch_spirv(const std::vector<std::uint32_t>&, const char*,
                    std::vector<VulkanBufferBinding>&, std::uint32_t,
                    std::string* error) {
    if (error) *error = "built without Vulkan (NODUS_HAVE_VULKAN undefined)";
    return false;
}
bool dispatch_spirv_backends(const std::vector<std::uint32_t>&, const char*,
                             const std::vector<VulkanBackendBinding>&,
                             std::uint32_t, std::string* error) {
    if (error) *error = "built without Vulkan (NODUS_HAVE_VULKAN undefined)";
    return false;
}
bool dispatch_spirv_backends_2d(const std::vector<std::uint32_t>&, const char*,
                                const std::vector<VulkanBackendBinding>&,
                                std::uint32_t, std::uint32_t, std::string* error) {
    return fail_with(error, "built without Vulkan (NODUS_HAVE_VULKAN undefined)");
}
bool dispatch_spirv_resident_loop(const std::vector<std::uint32_t>&, const char*,
                                  const std::vector<VulkanBackendBinding>&,
                                  std::uint32_t, std::uint32_t, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t,
                                  std::string* error) {
    if (error) *error = "built without Vulkan (NODUS_HAVE_VULKAN undefined)";
    return false;
}

#else

namespace detail {

constexpr std::uint32_t kMagic = 0x4e564b42u; // NVKB
constexpr std::uint64_t kCapabilities =
    GP_MEM_CAP_MAP_HOST | GP_MEM_CAP_DEVICE_POINTER | GP_MEM_CAP_D2D_NATIVE;

struct CachedPipeline;

struct DeviceState {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    struct ExtraQueue {
        VkQueue queue=VK_NULL_HANDLE;
        VkCommandPool command_pool=VK_NULL_HANDLE;
        std::mutex mutex;
    };
    std::vector<std::unique_ptr<ExtraQueue>> extra_queues;
    std::atomic<std::uint32_t> next_dispatch_queue{0};
    VkPhysicalDeviceProperties properties{};
    VulkanDeviceInfo identity;
    bool device_local_allocations = false;
    std::mutex queue_mutex;
    struct PipelineEntry {
        std::shared_ptr<CachedPipeline> pipeline;
        std::uint64_t last_use = 0;
    };
    std::mutex pipeline_mutex;
    std::unordered_map<std::string,PipelineEntry> pipelines;
    std::uint64_t pipeline_clock = 0;
    std::uint64_t pipeline_hits = 0, pipeline_created = 0;

    ~DeviceState() {
        if (device) {
            vkDeviceWaitIdle(device);
            pipelines.clear();
            for(auto& lane:extra_queues)
                if(lane->command_pool)
                    vkDestroyCommandPool(device,lane->command_pool,nullptr);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

struct CachedPipeline {
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    ~CachedPipeline() {
        if (pipeline) vkDestroyPipeline(device,pipeline,nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(device,pipeline_layout,nullptr);
        if (set_layout) vkDestroyDescriptorSetLayout(device,set_layout,nullptr);
    }
};

enum class HandleKind : std::uint32_t { Context, Buffer };

struct VulkanHandle {
    std::uint32_t magic = kMagic;
    HandleKind kind = HandleKind::Context;
    std::shared_ptr<DeviceState> state;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::size_t bytes = 0;
    std::size_t allocation_bytes = 0;
    std::uint32_t heap_index = UINT32_MAX;
    bool host_visible = false;

    ~VulkanHandle() {
        if (state && state->device) {
            if (buffer) vkDestroyBuffer(state->device, buffer, nullptr);
            if (memory) vkFreeMemory(state->device, memory, nullptr);
        }
    }
};

extern const gp_mem_backend_vtable_t g_vulkan_vtable;

bool vk_result(VkResult result, const char* operation, std::string* error) {
    if (result == VK_SUCCESS) return true;
    const char* reason = "Vulkan error";
    switch (result) {
    case VK_ERROR_OUT_OF_HOST_MEMORY: reason = "out of host memory"; break;
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: reason = "out of device memory"; break;
    case VK_ERROR_DEVICE_LOST: reason = "device lost"; break;
    case VK_ERROR_INITIALIZATION_FAILED: reason = "initialization failed"; break;
    case VK_ERROR_INCOMPATIBLE_DRIVER: reason = "incompatible driver"; break;
    default: break;
    }
    return fail_with(error, std::string(operation) + ": " + reason +
                     " (VkResult " + std::to_string(result) + ")");
}

bool create_instance(VkInstance* out, std::string* error) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "nodus";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    return vk_result(vkCreateInstance(&info, nullptr, out), "vkCreateInstance", error);
}

bool physical_devices(VkInstance instance, std::vector<VkPhysicalDevice>& devices,
                       std::string* error) {
    // A hotplug may change the count between the two calls. Never publish a
    // partial index map or loop forever in a changing driver inventory.
    for (int attempt = 0; attempt < 3; ++attempt) {
        std::uint32_t count = 0;
        if (!vk_result(vkEnumeratePhysicalDevices(instance, &count, nullptr),
                       "vkEnumeratePhysicalDevices(count)", error)) return false;
        devices.resize(count);
        if (!count) return true;
        const auto result = vkEnumeratePhysicalDevices(instance, &count, devices.data());
        if (result == VK_INCOMPLETE) continue;
        if (!vk_result(result, "vkEnumeratePhysicalDevices(list)", error)) return false;
        devices.resize(count);
        return true;
    }
    devices.clear();
    return fail_with(error, "vkEnumeratePhysicalDevices inventory kept changing (VK_INCOMPLETE)");
}

VulkanDeviceInfo device_info(VkPhysicalDevice physical, std::uint32_t index) {
    VulkanDeviceInfo info;
    info.index = index;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    info.name = properties.deviceName;
    info.vendor_id = properties.vendorID;
    info.device_id = properties.deviceID;
    info.device_type = properties.deviceType;
    info.api_version = properties.apiVersion;
    info.driver_version = properties.driverVersion;
    info.max_storage_buffers = std::min(properties.limits.maxPerStageDescriptorStorageBuffers,
                                        properties.limits.maxDescriptorSetStorageBuffers);
    info.max_storage_buffer_bytes = properties.limits.maxStorageBufferRange;
    if (properties.apiVersion >= VK_API_VERSION_1_1) {
        VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props.pNext = &ids;
        vkGetPhysicalDeviceProperties2(physical, &props);
        std::copy_n(ids.deviceUUID, info.uuid.size(), info.uuid.begin());
        std::copy_n(ids.deviceLUID, info.luid.size(), info.luid.begin());
        info.uuid_valid = true;
        info.luid_valid = ids.deviceLUIDValid == VK_TRUE;
    }
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    bool dedicated=false;
    for (std::uint32_t f = 0; f < count; ++f) {
        const auto& family=families[f];
        if(!family.queueCount || !(family.queueFlags & VK_QUEUE_COMPUTE_BIT))
            continue;
        const bool compute_only=!(family.queueFlags & VK_QUEUE_GRAPHICS_BIT);
        if(!info.compute || (compute_only&&!dedicated) ||
           (compute_only==dedicated &&
            family.queueCount>info.available_compute_queues)){
            info.compute=true;info.queue_family=f;
            info.available_compute_queues=family.queueCount;
            dedicated=compute_only;
        }
    }
    return info;
}

bool initialize_device(DeviceState& state, std::string* error) {
    // Runtime discovery is authoritative. The override permits a measured
    // one-queue baseline without recompiling any shader or device program.
    std::uint32_t desired=2;
    if(const char* setting=std::getenv("NODUS_VULKAN_COMPUTE_QUEUES")){
        char* end=nullptr;
        const auto parsed=std::strtoul(setting,&end,10);
        if(!*setting||!end||*end||parsed<1||parsed>16)
            return fail_with(error,"NODUS_VULKAN_COMPUTE_QUEUES must be 1..16");
        desired=static_cast<std::uint32_t>(parsed);
    }
    const auto count=std::min(desired,state.identity.available_compute_queues);
    if(!count)return fail_with(error,"selected compute family has no queues");
    std::vector<float> priorities(count,1.0f);
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = state.queue_family;
    queue_info.queueCount = count;
    queue_info.pQueuePriorities = priorities.data();
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    if (!vk_result(vkCreateDevice(state.physical, &device_info, nullptr, &state.device),
                   "vkCreateDevice", error)) return false;
    vkGetDeviceQueue(state.device, state.queue_family, 0, &state.queue);
    vkGetPhysicalDeviceProperties(state.physical, &state.properties);
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = state.queue_family;
    if(!vk_result(vkCreateCommandPool(state.device, &pool_info, nullptr,
                                     &state.command_pool), "vkCreateCommandPool", error))
        return false;
    for(std::uint32_t i=1;i<count;++i){
        auto lane=std::make_unique<DeviceState::ExtraQueue>();
        vkGetDeviceQueue(state.device,state.queue_family,i,&lane->queue);
        if(!vk_result(vkCreateCommandPool(state.device,&pool_info,nullptr,
                                         &lane->command_pool),
                      "vkCreateCommandPool(extra)",error))return false;
        state.extra_queues.push_back(std::move(lane));
    }
    return true;
}

std::shared_ptr<DeviceState> create_device(std::uint32_t requested, std::string* error) {
    auto state = std::make_shared<DeviceState>();
    if (!create_instance(&state->instance, error)) return {};
    std::vector<VkPhysicalDevice> devices;
    if (!physical_devices(state->instance, devices, error)) return {};
    if (requested != UINT32_MAX && requested >= devices.size()) {
        fail_with(error, "Vulkan device index " + std::to_string(requested) +
                  " is outside enumeration (" + std::to_string(devices.size()) + ")");
        return {}; // A typo is not permission to silently pick another GPU.
    }
    std::vector<std::uint32_t> order;
    if (requested != UINT32_MAX) order.push_back(requested);
    if (requested == UINT32_MAX) {
        for (std::uint32_t i = 0; i < devices.size(); ++i)
            if (i != requested) order.push_back(i);
    }
    std::string attempts;
    for (auto index : order) {
        auto identity = device_info(devices[index], index);
        if (!identity.compute) {
            if (index == requested) {
                attempts += "device " + std::to_string(index) + " (" + identity.name +
                            "): no compute queue; ";
                break;
            }
            continue;
        }
        state->physical = devices[index];
        state->queue_family = identity.queue_family;
        state->identity = std::move(identity);
        std::string reason;
        if (initialize_device(*state, &reason)) {
            if (error) *error = attempts; // prior failures remain observable
            return state;
        }
        attempts += "device " + std::to_string(index) + " (" + state->identity.name +
                    "): " + reason + "; ";
        if (state->command_pool) vkDestroyCommandPool(state->device, state->command_pool, nullptr);
        if (state->device) vkDestroyDevice(state->device, nullptr);
        state->command_pool = VK_NULL_HANDLE;
        state->device = VK_NULL_HANDLE;
        state->queue = VK_NULL_HANDLE;
        break;
    }
    fail_with(error, attempts.empty() ? "no Vulkan compute device" : attempts);
    return {};
}

bool find_memory_type(VkPhysicalDevice physical, std::uint32_t bits,
                      VkMemoryPropertyFlags required, std::uint32_t* out) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & required) == required) {
            *out = i;
            return true;
        }
    }
    return false;
}

VulkanHandle* as_handle(gp_mem_backend_handle_t h) {
    auto* value = reinterpret_cast<VulkanHandle*>(h);
    return value && value->magic == kMagic ? value : nullptr;
}

VulkanHandle* allocate_buffer(const std::shared_ptr<DeviceState>& state,
                              std::size_t bytes, bool device_local) {
    if (!state || !bytes) return nullptr;
    auto value = std::make_unique<VulkanHandle>();
    value->kind = HandleKind::Buffer;
    value->state = state;
    value->bytes = bytes;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = static_cast<VkDeviceSize>(bytes);
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(state->device, &info, nullptr, &value->buffer) != VK_SUCCESS)
        return nullptr;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(state->device, value->buffer, &requirements);
    std::uint32_t type = 0;
    const VkMemoryPropertyFlags required = device_local
        ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!find_memory_type(state->physical, requirements.memoryTypeBits,
                          required, &type)) return nullptr;
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(state->physical, &memory_properties);
    value->host_visible = (memory_properties.memoryTypes[type].propertyFlags &
                           (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                          (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    value->heap_index = memory_properties.memoryTypes[type].heapIndex;
    value->allocation_bytes = static_cast<std::size_t>(requirements.size);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = type;
    if (vkAllocateMemory(state->device, &alloc, nullptr, &value->memory) != VK_SUCCESS)
        return nullptr;
    if (vkBindBufferMemory(state->device, value->buffer, value->memory, 0) != VK_SUCCESS)
        return nullptr;
    try {
        mem_backend_internal::register_external_handle(
            value.get(), GP_MEM_BACKEND_VULKAN, &g_vulkan_vtable, bytes,
            value->host_visible ? kCapabilities : kCapabilities & ~GP_MEM_CAP_MAP_HOST);
    } catch (...) {
        // Registry insertion spans several maps; a later allocation may fail
        // after the first map already holds this pointer.
        mem_backend_internal::unregister_external_handle(value.get());
        throw;
    }
    return value.release();
}

void* backend_map(gp_mem_backend_handle_t h) {
    auto* value = as_handle(h);
    if (!value || value->kind != HandleKind::Buffer || !value->host_visible) return nullptr;
    void* mapped = nullptr;
    return vkMapMemory(value->state->device, value->memory, 0, VK_WHOLE_SIZE, 0,
                       &mapped) == VK_SUCCESS ? mapped : nullptr;
}
void backend_unmap(gp_mem_backend_handle_t h) {
    auto* value = as_handle(h);
    if (value && value->kind == HandleKind::Buffer && value->host_visible)
        vkUnmapMemory(value->state->device, value->memory);
}
gp_mem_backend_handle_t backend_alloc(gp_mem_backend_handle_t owner,
                                      std::size_t bytes, std::uint32_t) {
    auto* value = as_handle(owner);
    if (!value) return nullptr;
    auto* allocated = allocate_buffer(value->state, bytes,
                                      value->state->device_local_allocations);
    return reinterpret_cast<gp_mem_backend_handle_t>(allocated);
}
void backend_free(gp_mem_backend_handle_t h) {
    auto* value = as_handle(h);
    if (!value) return;
    mem_backend_internal::unregister_external_handle(h);
    delete value;
}
bool submit_copy(VulkanHandle* src, VulkanHandle* dst, std::size_t src_offset,
                 std::size_t dst_offset, std::size_t bytes);

int backend_copy_to(gp_mem_backend_handle_t h, std::size_t offset,
                    const void* src, std::size_t bytes) {
    auto* value = as_handle(h);
    if (!value || value->kind != HandleKind::Buffer || !src ||
        offset > value->bytes || bytes > value->bytes - offset) return 0;
    if (!bytes) return 1;
    if (!value->host_visible) {
        std::unique_ptr<void, decltype(&gp_mem_backend_release)> staging(
            allocate_buffer(value->state, bytes, false), &gp_mem_backend_release);
        if (!staging) return 0;
        void* mapped_staging = backend_map(staging.get());
        if (!mapped_staging) return 0;
        std::memcpy(mapped_staging, src, bytes);
        backend_unmap(staging.get());
        return submit_copy(as_handle(staging.get()), value, 0, offset, bytes) ? 1 : 0;
    }
    void* mapped = backend_map(h);
    if (!mapped) return 0;
    std::memcpy(static_cast<std::uint8_t*>(mapped) + offset, src, bytes);
    backend_unmap(h);
    return 1;
}
int backend_copy_from(gp_mem_backend_handle_t h, std::size_t offset,
                      void* dst, std::size_t bytes) {
    auto* value = as_handle(h);
    if (!value || value->kind != HandleKind::Buffer || !dst ||
        offset > value->bytes || bytes > value->bytes - offset) return 0;
    if (!bytes) return 1;
    if (!value->host_visible) {
        std::unique_ptr<void, decltype(&gp_mem_backend_release)> staging(
            allocate_buffer(value->state, bytes, false), &gp_mem_backend_release);
        if (!staging || !submit_copy(value, as_handle(staging.get()), offset, 0, bytes)) return 0;
        void* mapped_staging = backend_map(staging.get());
        if (!mapped_staging) return 0;
        std::memcpy(dst, mapped_staging, bytes);
        backend_unmap(staging.get());
        return 1;
    }
    void* mapped = backend_map(h);
    if (!mapped) return 0;
    std::memcpy(dst, static_cast<std::uint8_t*>(mapped) + offset, bytes);
    backend_unmap(h);
    return 1;
}

bool submit_copy(VulkanHandle* src, VulkanHandle* dst, std::size_t src_offset,
                 std::size_t dst_offset, std::size_t bytes) {
    if (!src || !dst || src->kind != HandleKind::Buffer ||
        dst->kind != HandleKind::Buffer || src->state.get() != dst->state.get() ||
        src_offset > src->bytes || bytes > src->bytes - src_offset ||
        dst_offset > dst->bytes || bytes > dst->bytes - dst_offset) return false;
    auto& s = *src->state;
    std::lock_guard<std::mutex> lock(s.queue_mutex);
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = s.command_pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(s.device, &alloc, &command) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command, &begin);
    VkBufferCopy region{static_cast<VkDeviceSize>(src_offset),
                        static_cast<VkDeviceSize>(dst_offset),
                        static_cast<VkDeviceSize>(bytes)};
    vkCmdCopyBuffer(command, src->buffer, dst->buffer, 1, &region);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(command);
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    bool ok = vkCreateFence(s.device, &fence_info, nullptr, &fence) == VK_SUCCESS;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    if (ok) ok = vkQueueSubmit(s.queue, 1, &submit, fence) == VK_SUCCESS;
    if (ok) ok = vkWaitForFences(s.device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    if (fence) vkDestroyFence(s.device, fence, nullptr);
    vkFreeCommandBuffers(s.device, s.command_pool, 1, &command);
    return ok;
}

int backend_copy_between(gp_mem_backend_handle_t src,
                         gp_mem_backend_handle_t dst,
                         std::size_t src_offset, std::size_t dst_offset,
                         std::size_t bytes) {
    return submit_copy(as_handle(src), as_handle(dst), src_offset, dst_offset,
                       bytes) ? 1 : 0;
}
uintptr_t backend_native(gp_mem_backend_handle_t h) {
    auto* value = as_handle(h);
    return value && value->kind == HandleKind::Buffer
               ? reinterpret_cast<uintptr_t>(value->buffer) : 0;
}
uintptr_t backend_stream(gp_mem_backend_handle_t h) {
    auto* value = as_handle(h);
    return value ? reinterpret_cast<uintptr_t>(value->state->queue) : 0;
}

const gp_mem_backend_vtable_t g_vulkan_vtable = {
    &backend_map, &backend_unmap, &backend_alloc, nullptr, &backend_free,
    &backend_copy_to, &backend_copy_from, &backend_copy_between,
    &backend_native, nullptr, nullptr, &backend_stream, nullptr};

struct DispatchObjects {
    std::shared_ptr<DeviceState> state;
    std::shared_ptr<CachedPipeline> cached;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    ~DispatchObjects() {
        if (!state) return;
        auto device = state->device;
        if (!cached) {
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (shader) vkDestroyShaderModule(device, shader, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        }
        if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    }
};

std::shared_ptr<CachedPipeline> get_cached_pipeline(
    const std::shared_ptr<DeviceState>& state,
    const std::vector<std::uint32_t>& spirv,const char* entry_name,
    std::uint32_t count,std::string* error) {
    const char* entry=entry_name&&*entry_name?entry_name:"main";
    const std::uint64_t words=spirv.size();
    std::string key(reinterpret_cast<const char*>(&words),sizeof(words));
    key.append(reinterpret_cast<const char*>(spirv.data()),
               spirv.size()*sizeof(std::uint32_t));
    key.append(reinterpret_cast<const char*>(&count),sizeof(count));
    key.append(entry);
    std::lock_guard<std::mutex> lock(state->pipeline_mutex);
    if(auto found=state->pipelines.find(key);found!=state->pipelines.end()){
        found->second.last_use=++state->pipeline_clock;
        ++state->pipeline_hits;
        return found->second.pipeline;
    }
    auto cached=std::make_shared<CachedPipeline>();
    cached->device=state->device;
    std::vector<VkDescriptorSetLayoutBinding> layout(count);
    for(std::uint32_t i=0;i<count;i++){
        layout[i].binding=i;
        layout[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layout[i].descriptorCount=1;
        layout[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount=count;
    layout_info.pBindings=layout.data();
    if(vkCreateDescriptorSetLayout(state->device,&layout_info,nullptr,
                                    &cached->set_layout)!=VK_SUCCESS){
        fail_with(error,"vkCreateDescriptorSetLayout failed");return {};
    }
    VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount=1;
    pipeline_layout_info.pSetLayouts=&cached->set_layout;
    if(vkCreatePipelineLayout(state->device,&pipeline_layout_info,nullptr,
                               &cached->pipeline_layout)!=VK_SUCCESS){
        fail_with(error,"vkCreatePipelineLayout failed");return {};
    }
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize=spirv.size()*sizeof(std::uint32_t);
    shader_info.pCode=spirv.data();
    VkShaderModule module=VK_NULL_HANDLE;
    if(vkCreateShaderModule(state->device,&shader_info,nullptr,&module)!=VK_SUCCESS){
        fail_with(error,"vkCreateShaderModule rejected module");return {};
    }
    VkComputePipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module=module;
    pipeline_info.stage.pName=entry;
    pipeline_info.layout=cached->pipeline_layout;
    const auto created=vkCreateComputePipelines(state->device,VK_NULL_HANDLE,
        1,&pipeline_info,nullptr,&cached->pipeline);
    vkDestroyShaderModule(state->device,module,nullptr);
    if(created!=VK_SUCCESS){
        fail_with(error,"vkCreateComputePipelines failed");return {};
    }
    state->pipelines.emplace(std::move(key),DeviceState::PipelineEntry{
        cached,++state->pipeline_clock});
    ++state->pipeline_created;
    if(state->pipelines.size()>32){
        auto oldest=std::min_element(state->pipelines.begin(),state->pipelines.end(),
            [](const auto& a,const auto& b){
                return a.second.last_use<b.second.last_use;});
        state->pipelines.erase(oldest);
    }
    return cached;
}

gp_mem_backend_handle_t create_backend(const gp_mem_backend_vulkan_params_t* params,
                                       std::string* error = nullptr) {
    const std::uint32_t requested = params ? params->device_index
        : std::numeric_limits<std::uint32_t>::max();
    if (params && (params->flags & ~GP_VULKAN_ALLOC_DEVICE_LOCAL)) {
        fail_with(error, "unsupported Vulkan backend flags");
        return nullptr;
    }
    auto state = create_device(requested, error);
    if (!state) return nullptr;
    state->device_local_allocations = params &&
        (params->flags & GP_VULKAN_ALLOC_DEVICE_LOCAL);
    auto value = std::unique_ptr<VulkanHandle>(new (std::nothrow) VulkanHandle());
    if (!value) {
        fail_with(error, "Vulkan context handle allocation: out of host memory");
        return nullptr;
    }
    value->state = std::move(state);
    try {
        mem_backend_internal::register_external_handle(
            value.get(), GP_MEM_BACKEND_VULKAN, &g_vulkan_vtable, 0,
            value->state->device_local_allocations ? kCapabilities & ~GP_MEM_CAP_MAP_HOST
                                                   : kCapabilities);
    } catch (...) {
        mem_backend_internal::unregister_external_handle(value.get());
        throw;
    }
    return value.release();
}

} // namespace detail

bool enumerate_vulkan_devices(std::vector<VulkanDeviceInfo>* devices,
                              std::string* error) {
    if (error) error->clear();
    if (!devices) return fail_with(error, "Vulkan device output is null");
    devices->clear();
    try {
        detail::DeviceState instance;
        if (!detail::create_instance(&instance.instance, error)) return false;
        std::vector<VkPhysicalDevice> physical;
        if (!detail::physical_devices(instance.instance, physical, error)) return false;
        std::vector<VulkanDeviceInfo> result;
        for (std::uint32_t i = 0; i < physical.size(); ++i)
            result.push_back(detail::device_info(physical[i], i));
        *devices = std::move(result);
        return true;
    } catch (const std::exception& ex) {
        return fail_with(error, std::string("Vulkan enumeration exception: ") + ex.what());
    } catch (...) {
        return fail_with(error, "Vulkan enumeration: unknown exception");
    }
}

gp_mem_backend_handle_t create_vulkan_backend(
    const gp_mem_backend_vulkan_params_t* params, VulkanDeviceInfo* selected,
    std::string* error) {
    creation_error.clear();
    creation_fatal = nullptr;
    if (selected) *selected = {};
    if (error) error->clear();
    try {
        std::unique_ptr<void, decltype(&gp_mem_backend_release)> handle(
            detail::create_backend(params, &creation_error), &gp_mem_backend_release);
        if (handle && selected) *selected = detail::as_handle(handle.get())->state->identity;
        if (error) *error = creation_error;
        return handle.release();
    } catch (const std::exception& ex) {
        creation_error = std::string("Vulkan context exception: ") + ex.what();
    } catch (...) {
        creation_error = "Vulkan context: unknown exception";
    }
    fail_with(error, creation_error);
    return nullptr;
}

namespace {
bool physical_status(VkPhysicalDevice physical, const VulkanDeviceInfo& identity,
                      VulkanBackendStatus* status, std::string* error) {
    if (error) error->clear();
    if (!status) return fail_with(error, "Vulkan status output is null");
    *status = {};
    try {
        VulkanBackendStatus result;
        result.device = identity;
        std::uint32_t count = 0;
        if (!detail::vk_result(vkEnumerateDeviceExtensionProperties(
                physical, nullptr, &count, nullptr),
                "vkEnumerateDeviceExtensionProperties(count)", error)) return false;
        std::vector<VkExtensionProperties> extensions(count);
        if (count && !detail::vk_result(vkEnumerateDeviceExtensionProperties(
                physical, nullptr, &count, extensions.data()),
                "vkEnumerateDeviceExtensionProperties(list)", error)) return false;
        result.budget_available = std::any_of(extensions.begin(), extensions.begin() + count,
            [](const auto& ext) { return std::strcmp(ext.extensionName,
                VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0; });
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 memory{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        memory.pNext = result.budget_available ? &budget : nullptr;
        vkGetPhysicalDeviceMemoryProperties2(physical, &memory);
        for (std::uint32_t i = 0; i < memory.memoryProperties.memoryHeapCount; ++i) {
            const auto& heap = memory.memoryProperties.memoryHeaps[i];
            result.heaps.push_back({heap.size,
                result.budget_available ? budget.heapBudget[i] : 0,
                result.budget_available ? budget.heapUsage[i] : 0,
                (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0});
        }
        *status = std::move(result);
        return true;
    } catch (const std::exception& ex) {
        return fail_with(error, std::string("Vulkan status exception: ") + ex.what());
    } catch (...) {
        return fail_with(error, "Vulkan status: unknown exception");
    }
}
} // namespace

bool query_vulkan_backend_status(gp_mem_backend_handle_t backend,
                                 VulkanBackendStatus* status, std::string* error) {
    if (status) *status = {};
    if (gp_mem_backend_get_type(backend) != GP_MEM_BACKEND_VULKAN)
        return fail_with(error, "status requires a registered Vulkan context or buffer");
    auto* handle = detail::as_handle(backend);
    if (!handle || !handle->state)
        return fail_with(error, "status requires a live Vulkan context or buffer");
    if(!physical_status(handle->state->physical, handle->state->identity,
                        status, error))return false;
    status->active_compute_queues=static_cast<std::uint32_t>(
        1+handle->state->extra_queues.size());
    return true;
}

bool query_vulkan_buffer_allocation(gp_mem_backend_handle_t buffer,
                                    VulkanBufferAllocationStatus* status,
                                    std::string* error) {
    if (status) *status = {};
    if (!status || gp_mem_backend_get_type(buffer) != GP_MEM_BACKEND_VULKAN)
        return fail_with(error, "allocation query requires a live Vulkan buffer");
    auto* handle = detail::as_handle(buffer);
    if (!handle || handle->kind != detail::HandleKind::Buffer)
        return fail_with(error, "allocation query requires a Vulkan buffer");
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(handle->state->physical, &memory);
    if (handle->heap_index >= memory.memoryHeapCount)
        return fail_with(error, "Vulkan buffer heap index is invalid");
    status->heap_index = handle->heap_index;
    status->allocation_bytes = handle->allocation_bytes;
    status->host_visible = handle->host_visible;
    status->device_local = (memory.memoryHeaps[handle->heap_index].flags &
                            VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    if (error) error->clear();
    return true;
}

bool query_vulkan_pipeline_cache_stats(gp_mem_backend_handle_t backend,
                                       VulkanPipelineCacheStats* stats,
                                       std::string* error) {
    if (stats) *stats = {};
    if (!stats || gp_mem_backend_get_type(backend) != GP_MEM_BACKEND_VULKAN)
        return fail_with(error,"pipeline cache query requires a Vulkan backend");
    auto* handle=detail::as_handle(backend);
    if (!handle || !handle->state)
        return fail_with(error,"pipeline cache query requires a live backend");
    std::lock_guard<std::mutex> lock(handle->state->pipeline_mutex);
    stats->resident=handle->state->pipelines.size();
    stats->hits=handle->state->pipeline_hits;
    stats->created=handle->state->pipeline_created;
    if(error)error->clear();
    return true;
}

bool query_vulkan_device_status(std::uint32_t index, VulkanBackendStatus* status,
                                std::string* error) {
    if (status) *status = {};
    if (error) error->clear();
    if (!status) return fail_with(error, "Vulkan status output is null");
    try {
        detail::DeviceState instance;
        if (!detail::create_instance(&instance.instance, error)) return false;
        std::vector<VkPhysicalDevice> devices;
        if (!detail::physical_devices(instance.instance, devices, error)) return false;
        if (index >= devices.size())
            return fail_with(error, "Vulkan device index is outside enumeration");
        return physical_status(devices[index], detail::device_info(devices[index], index),
                               status, error);
    } catch (const std::exception& ex) {
        return fail_with(error, std::string("Vulkan status exception: ") + ex.what());
    } catch (...) {
        return fail_with(error, "Vulkan status: unknown exception");
    }
}

bool vulkan_available() {
    auto h = create_vulkan_backend(nullptr, nullptr, nullptr);
    if (!h) return false;
    gp_mem_backend_release(h);
    return true;
}

std::string vulkan_device_name() {
    auto h = create_vulkan_backend(nullptr, nullptr, nullptr);
    if (!h) return {};
    auto* value = detail::as_handle(h);
    std::string name = value ? value->state->properties.deviceName : "";
    gp_mem_backend_release(h);
    return name;
}

static bool dispatch_spirv_backends_impl(const std::vector<std::uint32_t>& spirv,
                             const char* entry_name,
                             const std::vector<VulkanBackendBinding>& bindings,
                             std::uint32_t group_count_x,
                             std::uint32_t group_count_y,
                             std::uint32_t steps,
                             const std::array<std::uint32_t,4>* feedback,
                             std::string* error) {
    auto fail = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (spirv.empty()) return fail("empty SPIR-V module");
    if (bindings.empty()) return fail("no buffers bound");
    if (!group_count_x || !group_count_y ||
        group_count_x > 65535 || group_count_y > 65535)
        return fail("Vulkan workgroup counts outside 1..65535");
    if (!steps || steps > 4096) return fail("resident step count outside 1..4096");
    if (feedback) {
        for (auto index : *feedback)
            if (index >= bindings.size()) return fail("resident feedback binding is absent");
        if ((*feedback)[0] == (*feedback)[2] ||
            (*feedback)[1] == (*feedback)[3] ||
            bindings[(*feedback)[0]].backend == bindings[(*feedback)[2]].backend ||
            bindings[(*feedback)[1]].backend == bindings[(*feedback)[3]].backend ||
            bindings[(*feedback)[0]].bytes != bindings[(*feedback)[2]].bytes ||
            bindings[(*feedback)[1]].bytes != bindings[(*feedback)[3]].bytes)
            return fail("resident feedback buffers overlap or differ in size");
    }
    auto* first = detail::as_handle(bindings.front().backend);
    if (!first || first->kind != detail::HandleKind::Buffer)
        return fail("binding is not a Vulkan buffer");
    const auto state = first->state;
    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        1, state->properties.limits.minStorageBufferOffsetAlignment);
    std::vector<detail::VulkanHandle*> values;
    values.reserve(bindings.size());
    for (const auto& binding : bindings) {
        auto* value = detail::as_handle(binding.backend);
        if (!value || value->kind != detail::HandleKind::Buffer ||
            value->state.get() != state.get())
            return fail("all bindings must be Vulkan buffers from one context");
        if (!binding.bytes || binding.offset > value->bytes ||
            binding.bytes > value->bytes - binding.offset)
            return fail("Vulkan binding range exceeds its buffer");
        if (binding.offset % alignment)
            return fail("Vulkan binding offset violates SSBO alignment");
        values.push_back(value);
    }

    detail::DispatchObjects objects;
    objects.state = state;
    const std::uint32_t count = static_cast<std::uint32_t>(bindings.size());
    objects.cached=detail::get_cached_pipeline(state,spirv,entry_name,count,error);
    if(!objects.cached)return false;
    objects.set_layout=objects.cached->set_layout;
    objects.pipeline_layout=objects.cached->pipeline_layout;
    objects.pipeline=objects.cached->pipeline;
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(state->device, &pool_info, nullptr,
                               &objects.descriptor_pool) != VK_SUCCESS)
        return fail("vkCreateDescriptorPool failed");
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = objects.descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &objects.set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(state->device, &set_info, &set) != VK_SUCCESS)
        return fail("vkAllocateDescriptorSets failed");
    std::vector<VkDescriptorBufferInfo> buffer_info(count);
    std::vector<VkWriteDescriptorSet> writes(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        buffer_info[i] = {values[i]->buffer,
                          static_cast<VkDeviceSize>(bindings[i].offset),
                          static_cast<VkDeviceSize>(bindings[i].bytes)};
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_info[i];
    }
    vkUpdateDescriptorSets(state->device, count, writes.data(), 0, nullptr);

    const auto lane_index=state->next_dispatch_queue.fetch_add(
        1,std::memory_order_relaxed)%
        static_cast<std::uint32_t>(1+state->extra_queues.size());
    auto* extra=lane_index?state->extra_queues[lane_index-1].get():nullptr;
    auto& queue_mutex=extra?extra->mutex:state->queue_mutex;
    const auto command_pool=extra?extra->command_pool:state->command_pool;
    const auto queue=extra?extra->queue:state->queue;
    std::lock_guard<std::mutex> lock(queue_mutex);
    VkCommandBufferAllocateInfo command_alloc{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_alloc.commandPool = command_pool;
    command_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_alloc.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(state->device, &command_alloc, &command) != VK_SUCCESS)
        return fail("vkAllocateCommandBuffers failed");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command, &begin);
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT |
                                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &before, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, objects.pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            objects.pipeline_layout, 0, 1, &set, 0, nullptr);
    for (std::uint32_t step = 0; step < steps; ++step) {
        vkCmdDispatch(command, group_count_x, group_count_y, 1);
        if (!feedback) continue;
        VkMemoryBarrier to_copy{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        to_copy.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        to_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &to_copy,
                             0, nullptr, 0, nullptr);
        for (int pair = 0; pair < 2; ++pair) {
            const auto dst_index = (*feedback)[pair];
            const auto src_index = (*feedback)[pair + 2];
            VkBufferCopy region{
                static_cast<VkDeviceSize>(bindings[src_index].offset),
                static_cast<VkDeviceSize>(bindings[dst_index].offset),
                static_cast<VkDeviceSize>(bindings[src_index].bytes)};
            vkCmdCopyBuffer(command, values[src_index]->buffer,
                            values[dst_index]->buffer, 1, &region);
        }
        VkMemoryBarrier to_compute{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        to_compute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                                   VK_ACCESS_TRANSFER_READ_BIT;
        to_compute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &to_compute, 0, nullptr, 0, nullptr);
    }
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &after, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(command);
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    bool ok = vkCreateFence(state->device, &fence_info, nullptr, &fence) == VK_SUCCESS;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    if (ok) ok = vkQueueSubmit(queue, 1, &submit, fence) == VK_SUCCESS;
    if (ok) ok = vkWaitForFences(state->device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    if (fence) vkDestroyFence(state->device, fence, nullptr);
    vkFreeCommandBuffers(state->device, command_pool, 1, &command);
    return ok ? true : fail("Vulkan queue submission failed");
}

bool dispatch_spirv_backends(const std::vector<std::uint32_t>& spirv,
                             const char* entry_name,
                             const std::vector<VulkanBackendBinding>& bindings,
                             std::uint32_t group_count_x,
                             std::string* error) {
    return dispatch_spirv_backends_impl(spirv, entry_name, bindings,
                                        group_count_x, 1, 1, nullptr, error);
}

bool dispatch_spirv_backends_2d(const std::vector<std::uint32_t>& spirv,
                                const char* entry_name,
                                const std::vector<VulkanBackendBinding>& bindings,
                                std::uint32_t group_count_x,
                                std::uint32_t group_count_y,
                                std::string* error) {
    return dispatch_spirv_backends_impl(spirv, entry_name, bindings,
                                        group_count_x, group_count_y, 1, nullptr, error);
}

bool dispatch_spirv_resident_loop(const std::vector<std::uint32_t>& spirv,
                                  const char* entry_name,
                                  const std::vector<VulkanBackendBinding>& bindings,
                                  std::uint32_t group_count_x,
                                  std::uint32_t steps,
                                  std::uint32_t current_position,
                                  std::uint32_t current_velocity,
                                  std::uint32_t next_position,
                                  std::uint32_t next_velocity,
                                  std::string* error) {
    const std::array<std::uint32_t,4> feedback{
        current_position,current_velocity,next_position,next_velocity};
    return dispatch_spirv_backends_impl(spirv,entry_name,bindings,group_count_x,
                                        1,steps,&feedback,error);
}

bool dispatch_spirv(const std::vector<std::uint32_t>& spirv,
                    const char* entry_name,
                    std::vector<VulkanBufferBinding>& buffers,
                    std::uint32_t group_count_x,
                    std::string* error) {
    return dispatch_spirv_on_device(spirv, entry_name, buffers, group_count_x,
                                    nullptr, error);
}

bool dispatch_spirv_on_device(const std::vector<std::uint32_t>& spirv,
                             const char* entry_name,
                             std::vector<VulkanBufferBinding>& buffers,
                             std::uint32_t group_count_x,
                             const gp_mem_backend_vulkan_params_t* params,
                             std::string* error) {
    if (error) error->clear();
    if (spirv.empty()) return fail_with(error, "empty SPIR-V module");
    if (buffers.empty()) return fail_with(error, "no buffers bound");
    gp_mem_backend_handle_t context = create_vulkan_backend(params, nullptr, error);
    if (!context) {
        return false;
    }
    const auto* vtable = gp_mem_backend_get_vtable(context);
    std::vector<gp_mem_backend_handle_t> allocations;
    std::vector<VulkanBackendBinding> bindings;
    bool ok = vtable && vtable->alloc;
    for (auto& buffer : buffers) {
        auto h = ok ? vtable->alloc(context, buffer.bytes, 0) : nullptr;
        if (!h || (buffer.data && !vtable->copy_to_backend(
                       h, 0, buffer.data, buffer.bytes))) {
            ok = false;
            break;
        }
        allocations.push_back(h);
        bindings.push_back({h, 0, buffer.bytes});
    }
    if (ok) ok = dispatch_spirv_backends(spirv, entry_name, bindings,
                                         group_count_x, error);
    if (ok) {
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            if (buffers[i].readback && buffers[i].data &&
                !vtable->copy_from_backend(allocations[i], 0, buffers[i].data,
                                           buffers[i].bytes)) ok = false;
        }
    }
    for (auto h : allocations) gp_mem_backend_release(h);
    gp_mem_backend_release(context);
    if (!ok && error && error->empty()) *error = "Vulkan buffer transfer failed";
    return ok;
}

#endif

} // namespace nodus::kernels

extern "C" gp_mem_backend_handle_t gp_mem_backend_create_vulkan(
    const gp_mem_backend_vulkan_params_t* params) {
    try {
        return nodus::kernels::create_vulkan_backend(params, nullptr, nullptr);
    } catch (...) {
        // Also survives allocation failure while formatting another failure.
        nodus::kernels::creation_fatal = "Vulkan context creation failed (exception)";
        return nullptr;
    }
}

extern "C" const char* gp_mem_backend_vulkan_last_error(void) {
    return nodus::kernels::creation_fatal ? nodus::kernels::creation_fatal
        : nodus::kernels::creation_error.c_str();
}
