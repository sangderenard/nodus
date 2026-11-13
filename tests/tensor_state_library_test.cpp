#include "tensor_state_library.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

int main() {
    using Handle = std::unique_ptr<void, decltype(&gp_mem_backend_release)>;
    std::vector<nodus::gpu::TensorStateLibrary::Device> devices;
    std::vector<Handle> contexts;
    std::vector<nodus::kernels::VulkanDeviceInfo> inventory;
    std::string error;
    if (!nodus::kernels::enumerate_vulkan_devices(&inventory, &error)) return 1;
    for (const auto& identity : inventory) {
        if (!identity.compute) continue;
        gp_mem_backend_vulkan_params_t params{identity.index,
                                              GP_VULKAN_ALLOC_DEVICE_LOCAL};
        Handle context(nodus::kernels::create_vulkan_backend(
                           &params, nullptr, &error), &gp_mem_backend_release);
        if (!context) { std::cerr << error << '\n'; return 2; }
        devices.push_back({context.get(), 16 * 1024 * 1024});
        contexts.push_back(std::move(context));
    }
    if (devices.size() < 2) return 3;
    nodus::gpu::TensorStateLibrary library(devices, 32 * 1024 * 1024);
    using DType = nodus::tensors::TensorDType;
    nodus::gpu::TensorStateLibrary::Spec spec{{1024 * 1024}, DType::F64};
    std::vector<double> initial(1024 * 1024, 2.0);
    if (!library.add(101, spec, initial.data()) ||
        !library.add(102, spec, initial.data())) return 4;
    const auto* gpu0 = gp_mem_backend_get_vtable(contexts[0].get());
    const auto* gpu1 = gp_mem_backend_get_vtable(contexts[1].get());
    {
        auto lease = library.lease_gpu(101, 0, true);
        if (!lease) return 5;
        std::fill(initial.begin(), initial.end(), 5.0);
        if (!gpu0->copy_to_backend(lease.handle(), 0, initial.data(),
                                   initial.size() * sizeof(double))) return 6;
        if (library.lease_cpu(101)) return 7; // one writer per state
    }
    {
        auto lease = library.lease_gpu(101, 1);
        if (!lease) return 8;
        std::vector<double> observed(initial.size());
        if (!gpu1->copy_from_backend(lease.handle(), 0, observed.data(),
                                     observed.size() * sizeof(double)) ||
            observed != initial) return 9;
    }
    const auto restored_before = library.stats().vram[1].restored;
    {
        auto lease = library.lease_gpu(101, 1);
        if (!lease || library.stats().vram[1].restored != restored_before)
            return 17;
    }
    if(library.resident_bytes(101,0)!=initial.size()*sizeof(double))return 18;
    {
        auto lease=library.lease_gpu(101,1,true);
        if(!lease)return 19;
        // The old GPU0 copy remains allocated, but GPU1 is now the newer writer.
    }
    if(library.resident_bytes(101,0)!=0)return 20;
    {
        auto lease = library.lease_cpu(101);
        if (!lease || *static_cast<double*>(lease.data()) != 5.0) return 10;
    }
    {
        auto lease = library.lease_cpu(102, true);
        if (!lease) return 11;
        std::fill_n(static_cast<double*>(lease.data()), initial.size(), 7.0);
    }
    {
        auto lease = library.lease_gpu(102, devices.size() - 1);
        if (!lease) return 12;
        double observed = 0;
        const auto* gpu = gp_mem_backend_get_vtable(contexts.back().get());
        if (!gpu->copy_from_backend(lease.handle(), 0, &observed,
                                     sizeof(observed)) || observed != 7.0) return 13;
    }
    std::thread cpu([&] {
        auto lease = library.lease_cpu(102, true);
        if (lease) static_cast<double*>(lease.data())[0] = 11.0;
    });
    std::thread gpu([&] {
        auto lease = library.lease_gpu(101, 0, true);
        if (lease) {
            const double value = 13.0;
            gpu0->copy_to_backend(lease.handle(), 0, &value, sizeof(value));
        }
    });
    cpu.join();gpu.join();
    {
        auto lease = library.lease_cpu(101);
        if (!lease || *static_cast<double*>(lease.data()) != 13.0) return 14;
    }
    {
        auto lease = library.lease_gpu(102, 0);
        double observed = 0;
        if (!lease || !gpu0->copy_from_backend(lease.handle(), 0,
                                                &observed, sizeof(observed)) ||
            observed != 11.0) return 15;
    }
    const auto stats = library.stats();
    if (!library.remove(101) || !library.remove(102) ||
        stats.vram.size() != devices.size() ||
        stats.ram_bytes != 16 * 1024 * 1024) return 16;
    std::cout << "devices=" << devices.size() << " cpu_and_gpu_parallel=pass"
              << " cross_device_restore=pass ram_bytes=" << stats.ram_bytes << '\n';
    return 0;
}
