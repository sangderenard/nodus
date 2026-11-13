#include "vulkan_state_bank.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main() {
    using Handle = std::unique_ptr<void, decltype(&gp_mem_backend_release)>;
    gp_mem_backend_vulkan_params_t params{UINT32_MAX, GP_VULKAN_ALLOC_DEVICE_LOCAL};
    nodus::kernels::VulkanDeviceInfo device;
    std::string error;
    Handle context(nodus::kernels::create_vulkan_backend(&params, &device, &error),
                   &gp_mem_backend_release);
    if (!context) { std::cerr << error << '\n'; return 1; }
    constexpr std::size_t bytes = 8 * 1024 * 1024;
    Handle home_a(gp_mem_backend_create_host(bytes), &gp_mem_backend_release);
    Handle home_b(gp_mem_backend_create_host(bytes), &gp_mem_backend_release);
    if (!home_a || !home_b) return 2;
    const auto* host = gp_mem_backend_get_vtable(home_a.get());
    const auto* gpu = gp_mem_backend_get_vtable(context.get());
    std::vector<std::uint8_t> pattern(bytes, 17), changed(bytes, 93), observed(bytes);
    if (!host->copy_to_backend(home_a.get(), 0, pattern.data(), bytes) ||
        !host->copy_to_backend(home_b.get(), 0, pattern.data(), bytes)) return 3;
    nodus::gpu::VulkanStateBank bank(context.get(), bytes);
    if (!bank.add(1001, bytes, home_a.get()) || !bank.add(1002, bytes, home_b.get()))
        return 4;
    {
        auto first = bank.lease(1001, true);
        if (!first || bank.lease(1002)) return 5; // pinned lease blocks eviction
        if (!gpu->copy_to_backend(first.handle(), 0, changed.data(), bytes)) return 6;
    }
    {
        auto second = bank.lease(1002);
        if (!second) return 7;
        if (!gpu->copy_from_backend(second.handle(), 0, observed.data(), bytes) ||
            observed != pattern) return 8;
    }
    if (!host->copy_from_backend(home_a.get(), 0, observed.data(), bytes) ||
        observed != changed) return 9;
    {
        auto first = bank.lease(1001);
        if (!first || !gpu->copy_from_backend(first.handle(), 0, observed.data(), bytes) ||
            observed != changed) return 10;
    }
    if (!bank.flush()) return 11;
    const auto stats = bank.stats();
    if (stats.resident_bytes > stats.budget_bytes || stats.evicted < 2 ||
        stats.reused < 2 || stats.saved != 1 || stats.restored != 3) return 12;
    std::cout << "device=" << device.name << " budget=" << stats.budget_bytes
              << " resident=" << stats.resident_bytes << " reused=" << stats.reused
              << " evicted=" << stats.evicted << " restored=" << stats.restored
              << " saved=" << stats.saved << " keyed_lease=pass\n";
    nodus::gpu::VulkanStateBank warm(context.get(), 2 * bytes);
    if (!warm.add(2001, bytes, home_a.get()) ||
        !warm.add(2002, bytes, home_b.get())) return 13;
    for (int round = 0; round < 3; ++round) {
        auto first = warm.lease(2001);
        if (!first) return 14;
        first.reset();
        auto second = warm.lease(2002);
        if (!second) return 15;
    }
    const auto warm_stats = warm.stats();
    if (warm_stats.evicted || warm_stats.restored != 2 ||
        warm_stats.resident_bytes != 2 * bytes) return 16;
    std::cout << "warm_resident=" << warm_stats.resident_bytes
              << " warm_evictions=" << warm_stats.evicted << '\n';
    return 0;
}
