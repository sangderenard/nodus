// Persistent Vulkan storage buffers and KernelIR SPIR-V dispatch.
#include "kernel_vulkan_runner.h"
#include "mem_backend_registry_internal.h"

#if defined(NODUS_HAVE_VULKAN)
#include <vulkan/vulkan.h>
#endif

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace nodus::kernels {

#if !defined(NODUS_HAVE_VULKAN)

bool vulkan_available() { return false; }
std::string vulkan_device_name() { return {}; }
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

#else

namespace detail {

constexpr std::uint32_t kMagic = 0x4e564b42u; // NVKB
constexpr std::uint64_t kCapabilities =
    GP_MEM_CAP_MAP_HOST | GP_MEM_CAP_DEVICE_POINTER | GP_MEM_CAP_D2D_NATIVE;

struct DeviceState {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    std::mutex queue_mutex;

    ~DeviceState() {
        if (device) {
            vkDeviceWaitIdle(device);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
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

    ~VulkanHandle() {
        if (state && state->device) {
            if (buffer) vkDestroyBuffer(state->device, buffer, nullptr);
            if (memory) vkFreeMemory(state->device, memory, nullptr);
        }
    }
};

extern const gp_mem_backend_vtable_t g_vulkan_vtable;

bool create_instance(VkInstance* out) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "nodus";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    return vkCreateInstance(&info, nullptr, out) == VK_SUCCESS;
}

bool find_compute_device(VkInstance instance, std::uint32_t requested,
                         VkPhysicalDevice* out_device,
                         std::uint32_t* out_family) {
    std::uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || !count)
        return false;
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    for (std::uint32_t d = 0; d < count; ++d) {
        if (requested != std::numeric_limits<std::uint32_t>::max() && d != requested)
            continue;
        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[d], &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[d], &family_count,
                                                 families.data());
        for (std::uint32_t f = 0; f < family_count; ++f) {
            if (families[f].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                *out_device = devices[d];
                *out_family = f;
                return true;
            }
        }
    }
    return false;
}

std::shared_ptr<DeviceState> create_device(std::uint32_t requested) {
    auto state = std::make_shared<DeviceState>();
    if (!create_instance(&state->instance)) return {};
    if (!find_compute_device(state->instance, requested, &state->physical,
                             &state->queue_family)) return {};
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = state->queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    if (vkCreateDevice(state->physical, &device_info, nullptr, &state->device) !=
        VK_SUCCESS) return {};
    vkGetDeviceQueue(state->device, state->queue_family, 0, &state->queue);
    vkGetPhysicalDeviceProperties(state->physical, &state->properties);
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = state->queue_family;
    if (vkCreateCommandPool(state->device, &pool_info, nullptr,
                            &state->command_pool) != VK_SUCCESS) return {};
    return state;
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
                              std::size_t bytes) {
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
    if (!find_memory_type(state->physical, requirements.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          &type)) return nullptr;
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = type;
    if (vkAllocateMemory(state->device, &alloc, nullptr, &value->memory) != VK_SUCCESS)
        return nullptr;
    if (vkBindBufferMemory(state->device, value->buffer, value->memory, 0) != VK_SUCCESS)
        return nullptr;
    auto* released = value.release();
    mem_backend_internal::register_external_handle(
        released, GP_MEM_BACKEND_VULKAN, &g_vulkan_vtable, bytes, kCapabilities);
    return released;
}

void* backend_map(gp_mem_backend_handle_t h) {
    auto* value = as_handle(h);
    if (!value || value->kind != HandleKind::Buffer) return nullptr;
    void* mapped = nullptr;
    return vkMapMemory(value->state->device, value->memory, 0, VK_WHOLE_SIZE, 0,
                       &mapped) == VK_SUCCESS ? mapped : nullptr;
}
void backend_unmap(gp_mem_backend_handle_t h) {
    auto* value = as_handle(h);
    if (value && value->kind == HandleKind::Buffer)
        vkUnmapMemory(value->state->device, value->memory);
}
gp_mem_backend_handle_t backend_alloc(gp_mem_backend_handle_t owner,
                                      std::size_t bytes, std::uint32_t) {
    auto* value = as_handle(owner);
    if (!value) return nullptr;
    auto* allocated = allocate_buffer(value->state, bytes);
    return reinterpret_cast<gp_mem_backend_handle_t>(allocated);
}
void backend_free(gp_mem_backend_handle_t h) {
    auto* value = as_handle(h);
    if (!value) return;
    mem_backend_internal::unregister_external_handle(h);
    delete value;
}
int backend_copy_to(gp_mem_backend_handle_t h, std::size_t offset,
                    const void* src, std::size_t bytes) {
    auto* value = as_handle(h);
    if (!value || value->kind != HandleKind::Buffer || !src ||
        offset > value->bytes || bytes > value->bytes - offset) return 0;
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
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    ~DispatchObjects() {
        if (!state) return;
        auto device = state->device;
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (shader) vkDestroyShaderModule(device, shader, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    }
};

gp_mem_backend_handle_t create_backend(const gp_mem_backend_vulkan_params_t* params) {
    const std::uint32_t requested = params ? params->device_index
        : std::numeric_limits<std::uint32_t>::max();
    if (params && params->flags) return nullptr;
    auto state = create_device(requested);
    if (!state) return nullptr;
    auto* value = new (std::nothrow) VulkanHandle();
    if (!value) return nullptr;
    value->state = std::move(state);
    mem_backend_internal::register_external_handle(
        value, GP_MEM_BACKEND_VULKAN, &g_vulkan_vtable, 0, kCapabilities);
    return value;
}

} // namespace detail

bool vulkan_available() {
    auto h = detail::create_backend(nullptr);
    if (!h) return false;
    gp_mem_backend_release(h);
    return true;
}

std::string vulkan_device_name() {
    auto h = detail::create_backend(nullptr);
    if (!h) return {};
    auto* value = detail::as_handle(h);
    std::string name = value ? value->state->properties.deviceName : "";
    gp_mem_backend_release(h);
    return name;
}

bool dispatch_spirv_backends(const std::vector<std::uint32_t>& spirv,
                             const char* entry_name,
                             const std::vector<VulkanBackendBinding>& bindings,
                             std::uint32_t group_count_x,
                             std::string* error) {
    auto fail = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (spirv.empty()) return fail("empty SPIR-V module");
    if (bindings.empty()) return fail("no buffers bound");
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
    std::vector<VkDescriptorSetLayoutBinding> layout(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        layout[i].binding = i;
        layout[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layout[i].descriptorCount = 1;
        layout[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = count;
    layout_info.pBindings = layout.data();
    if (vkCreateDescriptorSetLayout(state->device, &layout_info, nullptr,
                                    &objects.set_layout) != VK_SUCCESS)
        return fail("vkCreateDescriptorSetLayout failed");
    VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &objects.set_layout;
    if (vkCreatePipelineLayout(state->device, &pipeline_layout_info, nullptr,
                               &objects.pipeline_layout) != VK_SUCCESS)
        return fail("vkCreatePipelineLayout failed");
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = spirv.size() * sizeof(std::uint32_t);
    shader_info.pCode = spirv.data();
    if (vkCreateShaderModule(state->device, &shader_info, nullptr,
                             &objects.shader) != VK_SUCCESS)
        return fail("vkCreateShaderModule rejected module");
    VkComputePipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = objects.shader;
    pipeline_info.stage.pName = entry_name && *entry_name ? entry_name : "main";
    pipeline_info.layout = objects.pipeline_layout;
    if (vkCreateComputePipelines(state->device, VK_NULL_HANDLE, 1, &pipeline_info,
                                 nullptr, &objects.pipeline) != VK_SUCCESS)
        return fail("vkCreateComputePipelines failed");
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

    std::lock_guard<std::mutex> lock(state->queue_mutex);
    VkCommandBufferAllocateInfo command_alloc{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_alloc.commandPool = state->command_pool;
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
    vkCmdDispatch(command, group_count_x ? group_count_x : 1, 1, 1);
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
    if (ok) ok = vkQueueSubmit(state->queue, 1, &submit, fence) == VK_SUCCESS;
    if (ok) ok = vkWaitForFences(state->device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    if (fence) vkDestroyFence(state->device, fence, nullptr);
    vkFreeCommandBuffers(state->device, state->command_pool, 1, &command);
    return ok ? true : fail("Vulkan queue submission failed");
}

bool dispatch_spirv(const std::vector<std::uint32_t>& spirv,
                    const char* entry_name,
                    std::vector<VulkanBufferBinding>& buffers,
                    std::uint32_t group_count_x,
                    std::string* error) {
    gp_mem_backend_handle_t context = detail::create_backend(nullptr);
    if (!context) {
        if (error) *error = "no Vulkan compute device";
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
#if defined(NODUS_HAVE_VULKAN)
    return nodus::kernels::detail::create_backend(params);
#else
    (void)params;
    return nullptr;
#endif
}
