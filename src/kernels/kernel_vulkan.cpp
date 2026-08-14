// kernel_vulkan.cpp
// Vulkan compute execution for assembled KernelIR modules.
//
// Deliberately minimal and linear: one instance, one device, one queue, one
// pipeline, host-visible coherent buffers, one submit, one wait. No staging
// buffers, no pipeline cache, no async. The purpose is to make nodus able to
// RUN what it assembles so a kernel's correctness is a measured fact rather
// than a structural verdict from spirv-val.
#include "../kernel_isa.h"

#if defined(NODUS_HAVE_VULKAN)
#include <vulkan/vulkan.h>
#endif

#include "kernel_vulkan_runner.h"

#include <cstring>

namespace nodus {
namespace kernels {

#if !defined(NODUS_HAVE_VULKAN)

bool vulkan_available() { return false; }
std::string vulkan_device_name() { return {}; }
bool dispatch_spirv(const std::vector<std::uint32_t>&, const char*,
                    std::vector<VulkanBufferBinding>&, std::uint32_t,
                    std::string* error) {
    if (error) *error = "built without Vulkan (NODUS_HAVE_VULKAN undefined)";
    return false;
}

#else

namespace {

// One-shot context. Everything is torn down in reverse order by the
// destructor, so every early return below is leak-free.
struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    std::vector<VkBuffer> buffers;
    std::vector<VkDeviceMemory> memories;

    ~Context() {
        if (device) {
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (shader) vkDestroyShaderModule(device, shader, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            for (VkBuffer b : buffers) if (b) vkDestroyBuffer(device, b, nullptr);
            for (VkDeviceMemory m : memories) if (m) vkFreeMemory(device, m, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

bool create_instance(VkInstance* out) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "nodus-kernel-runner";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    return vkCreateInstance(&info, nullptr, out) == VK_SUCCESS;
}

// First physical device exposing a compute queue family.
bool pick_device(VkInstance instance, VkPhysicalDevice* out_device,
                 std::uint32_t* out_family) {
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) return false;
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    for (VkPhysicalDevice device : devices) {
        std::uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &families, nullptr);
        std::vector<VkQueueFamilyProperties> props(families);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &families, props.data());
        for (std::uint32_t i = 0; i < families; ++i) {
            if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                *out_device = device;
                *out_family = i;
                return true;
            }
        }
    }
    return false;
}

// Host-visible + coherent, so writes land without explicit flushes.
bool find_memory_type(VkPhysicalDevice physical, std::uint32_t type_bits,
                      VkMemoryPropertyFlags want, std::uint32_t* out) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & want) == want) {
            *out = i;
            return true;
        }
    }
    return false;
}

} // namespace

bool vulkan_available() {
    VkInstance instance = VK_NULL_HANDLE;
    if (!create_instance(&instance)) return false;
    VkPhysicalDevice device = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    const bool ok = pick_device(instance, &device, &family);
    vkDestroyInstance(instance, nullptr);
    return ok;
}

std::string vulkan_device_name() {
    VkInstance instance = VK_NULL_HANDLE;
    if (!create_instance(&instance)) return {};
    VkPhysicalDevice device = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    std::string name;
    if (pick_device(instance, &device, &family)) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(device, &props);
        name = props.deviceName;
    }
    vkDestroyInstance(instance, nullptr);
    return name;
}

bool dispatch_spirv(const std::vector<std::uint32_t>& spirv,
                    const char* entry_name,
                    std::vector<VulkanBufferBinding>& buffers,
                    std::uint32_t group_count_x,
                    std::string* error) {
    auto fail = [&](const char* why) {
        if (error) *error = why;
        return false;
    };
    if (spirv.empty()) return fail("empty SPIR-V module");
    if (buffers.empty()) return fail("no buffers bound");

    Context ctx;
    if (!create_instance(&ctx.instance)) return fail("vkCreateInstance failed");
    if (!pick_device(ctx.instance, &ctx.physical, &ctx.queue_family)) {
        return fail("no Vulkan device with a compute queue");
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = ctx.queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    if (vkCreateDevice(ctx.physical, &device_info, nullptr, &ctx.device) != VK_SUCCESS) {
        return fail("vkCreateDevice failed");
    }
    vkGetDeviceQueue(ctx.device, ctx.queue_family, 0, &ctx.queue);

    // Buffers: one storage buffer per binding, host-visible so the upload and
    // readback are plain memcpy through a mapped pointer.
    const std::uint32_t count = static_cast<std::uint32_t>(buffers.size());
    ctx.buffers.resize(count, VK_NULL_HANDLE);
    ctx.memories.resize(count, VK_NULL_HANDLE);
    for (std::uint32_t i = 0; i < count; ++i) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = buffers[i].bytes;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(ctx.device, &info, nullptr, &ctx.buffers[i]) != VK_SUCCESS) {
            return fail("vkCreateBuffer failed");
        }
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(ctx.device, ctx.buffers[i], &requirements);
        std::uint32_t type_index = 0;
        if (!find_memory_type(ctx.physical, requirements.memoryTypeBits,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              &type_index)) {
            return fail("no host-visible coherent memory type");
        }
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = requirements.size;
        alloc.memoryTypeIndex = type_index;
        if (vkAllocateMemory(ctx.device, &alloc, nullptr, &ctx.memories[i]) != VK_SUCCESS) {
            return fail("vkAllocateMemory failed");
        }
        vkBindBufferMemory(ctx.device, ctx.buffers[i], ctx.memories[i], 0);
        void* mapped = nullptr;
        if (vkMapMemory(ctx.device, ctx.memories[i], 0, buffers[i].bytes, 0, &mapped) != VK_SUCCESS) {
            return fail("vkMapMemory (upload) failed");
        }
        if (buffers[i].data) std::memcpy(mapped, buffers[i].data, buffers[i].bytes);
        else std::memset(mapped, 0, buffers[i].bytes);
        vkUnmapMemory(ctx.device, ctx.memories[i]);
    }

    // Descriptor set 0, bindings 0..N-1 -- the assembler's own convention.
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        layout_bindings[i].binding = i;
        layout_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layout_bindings[i].descriptorCount = 1;
        layout_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = count;
    layout_info.pBindings = layout_bindings.data();
    if (vkCreateDescriptorSetLayout(ctx.device, &layout_info, nullptr, &ctx.set_layout) != VK_SUCCESS) {
        return fail("vkCreateDescriptorSetLayout failed");
    }
    VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &ctx.set_layout;
    if (vkCreatePipelineLayout(ctx.device, &pipeline_layout_info, nullptr,
                               &ctx.pipeline_layout) != VK_SUCCESS) {
        return fail("vkCreatePipelineLayout failed");
    }

    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = spirv.size() * sizeof(std::uint32_t);
    shader_info.pCode = spirv.data();
    if (vkCreateShaderModule(ctx.device, &shader_info, nullptr, &ctx.shader) != VK_SUCCESS) {
        return fail("vkCreateShaderModule rejected the assembled module");
    }
    VkComputePipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = ctx.shader;
    pipeline_info.stage.pName = entry_name && *entry_name ? entry_name : "main";
    pipeline_info.layout = ctx.pipeline_layout;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info,
                                 nullptr, &ctx.pipeline) != VK_SUCCESS) {
        return fail("vkCreateComputePipelines failed (entry point name mismatch?)");
    }

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &ctx.descriptor_pool) != VK_SUCCESS) {
        return fail("vkCreateDescriptorPool failed");
    }
    VkDescriptorSetAllocateInfo set_alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_alloc.descriptorPool = ctx.descriptor_pool;
    set_alloc.descriptorSetCount = 1;
    set_alloc.pSetLayouts = &ctx.set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(ctx.device, &set_alloc, &set) != VK_SUCCESS) {
        return fail("vkAllocateDescriptorSets failed");
    }
    std::vector<VkDescriptorBufferInfo> buffer_infos(count);
    std::vector<VkWriteDescriptorSet> writes(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        buffer_infos[i].buffer = ctx.buffers[i];
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = buffers[i].bytes;
        writes[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(ctx.device, count, writes.data(), 0, nullptr);

    VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_info.queueFamilyIndex = ctx.queue_family;
    if (vkCreateCommandPool(ctx.device, &command_pool_info, nullptr, &ctx.command_pool) != VK_SUCCESS) {
        return fail("vkCreateCommandPool failed");
    }
    VkCommandBufferAllocateInfo command_alloc{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_alloc.commandPool = ctx.command_pool;
    command_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_alloc.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(ctx.device, &command_alloc, &command) != VK_SUCCESS) {
        return fail("vkAllocateCommandBuffers failed");
    }
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command, &begin);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx.pipeline_layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(command, group_count_x ? group_count_x : 1, 1, 1);
    vkEndCommandBuffer(command);

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(ctx.device, &fence_info, nullptr, &fence);
    if (vkQueueSubmit(ctx.queue, 1, &submit, fence) != VK_SUCCESS) {
        vkDestroyFence(ctx.device, fence, nullptr);
        return fail("vkQueueSubmit failed");
    }
    const VkResult waited =
        vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 * 1000);
    vkDestroyFence(ctx.device, fence, nullptr);
    if (waited != VK_SUCCESS) return fail("dispatch did not complete within 5s");

    for (std::uint32_t i = 0; i < count; ++i) {
        if (!buffers[i].readback || !buffers[i].data) continue;
        void* mapped = nullptr;
        if (vkMapMemory(ctx.device, ctx.memories[i], 0, buffers[i].bytes, 0, &mapped) != VK_SUCCESS) {
            return fail("vkMapMemory (readback) failed");
        }
        std::memcpy(buffers[i].data, mapped, buffers[i].bytes);
        vkUnmapMemory(ctx.device, ctx.memories[i]);
    }
    return true;
}

#endif // NODUS_HAVE_VULKAN

} // namespace kernels
} // namespace nodus
