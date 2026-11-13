#include "kernel_vulkan_runner.h"
#include "mem_backend.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main() {
    gp_mem_backend_vulkan_params_t params{UINT32_MAX, GP_VULKAN_ALLOC_DEVICE_LOCAL};
    nodus::kernels::VulkanDeviceInfo selected;
    std::string error;
    using Handle = std::unique_ptr<void, decltype(&gp_mem_backend_release)>;
    Handle context(nodus::kernels::create_vulkan_backend(&params, &selected, &error),
                   &gp_mem_backend_release);
    if (!context) {
        std::cerr << "device-local context: " << error << '\n';
        return 1;
    }
    const auto* vtable = gp_mem_backend_get_vtable(context.get());
    if (!vtable || !vtable->alloc || !vtable->copy_to_backend ||
        !vtable->copy_from_backend) return 2;
    constexpr std::size_t bytes = 4 * 1024 * 1024;
    Handle source(vtable->alloc(context.get(), bytes, 0), &gp_mem_backend_release);
    Handle target(vtable->alloc(context.get(), bytes, 0), &gp_mem_backend_release);
    if (!source || !target) return 3;
    nodus::kernels::VulkanBufferAllocationStatus allocation;
    nodus::kernels::VulkanBackendStatus budget;
    if (!nodus::kernels::query_vulkan_buffer_allocation(source.get(), &allocation, &error) ||
        !nodus::kernels::query_vulkan_backend_status(context.get(), &budget, &error) ||
        !allocation.device_local || allocation.heap_index >= budget.heaps.size() ||
        allocation.allocation_bytes < bytes) {
        std::cerr << "allocation is not in a measured device-local heap: " << error << '\n';
        return 4;
    }
    std::vector<std::uint8_t> input(bytes), observed(bytes);
    for (std::size_t i = 0; i < bytes; ++i)
        input[i] = static_cast<std::uint8_t>((i * 97 + 13) & 255);
    if (!vtable->copy_to_backend(source.get(), 0, input.data(), bytes) ||
        !gp_mem_backend_transfer(source.get(), target.get(), 0, 0, bytes) ||
        !vtable->copy_from_backend(target.get(), 0, observed.data(), bytes) ||
        input != observed) return 5;
    std::cout << "device=" << selected.name << " heap=" << allocation.heap_index
              << " allocation_bytes=" << allocation.allocation_bytes
              << " budget_bytes=" << budget.heaps[allocation.heap_index].budget_bytes
              << " host_visible=" << allocation.host_visible
              << " native_copy=pass staging=pass\n";
    return 0;
}
