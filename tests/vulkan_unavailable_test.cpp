// Compile kernel_vulkan.cpp without NODUS_HAVE_VULKAN. No driver is loaded.
#include "kernel_vulkan_runner.h"
#include <cstring>
#include <iostream>

int main() {
    using namespace nodus::kernels;
    std::vector<VulkanDeviceInfo> devices(1);
    std::string error;
    if (enumerate_vulkan_devices(&devices, &error) || !devices.empty() ||
        error.find("built without Vulkan") == error.npos) return 1;
    VulkanBackendStatus status;
    status.heaps.resize(1);
    if (query_vulkan_device_status(0, &status, &error) || !status.heaps.empty()) return 2;
    VulkanDeviceInfo selected;
    selected.name = "stale";
    if (create_vulkan_backend(nullptr, &selected, &error) || !selected.name.empty()) return 3;
    if (gp_mem_backend_create_vulkan(nullptr) ||
        !std::strstr(gp_mem_backend_vulkan_last_error(), "built without Vulkan")) return 4;
    std::vector<VulkanBufferBinding> buffers;
    if (dispatch_spirv_on_device({}, "main", buffers, 1, nullptr, &error)) return 5;
    std::cout << "PASS: Vulkan unavailable paths fail explicitly without a driver\n";
    return 0;
}
