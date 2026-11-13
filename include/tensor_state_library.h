#pragma once
// Addressed tensor state shared by CPU rows and a device-local Vulkan bank.
// A row leases at a job boundary; no lock is taken in its numerical loop.
#include "vulkan_state_bank.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nodus::gpu {

class TensorStateLibrary {
public:
    using Buffer = std::unique_ptr<void, decltype(&gp_mem_backend_release)>;
    struct Spec {
        std::vector<std::uint64_t> shape;
        nodus::tensors::TensorDType dtype = nodus::tensors::TensorDType::Unknown;
    };
    struct Stats {
        std::uint64_t ram_bytes = 0, ram_budget_bytes = 0;
        std::vector<VulkanStateBank::Stats> vram;
    };

    class CpuLease {
    public:
        CpuLease() = default;
        CpuLease(TensorStateLibrary* owner, std::uint64_t address, void* data,
                 std::size_t bytes, bool writable) noexcept
            : owner_(owner), address_(address), data_(data), bytes_(bytes),
              writable_(writable) {}
        CpuLease(const CpuLease&) = delete;
        CpuLease& operator=(const CpuLease&) = delete;
        CpuLease(CpuLease&& other) noexcept { *this = std::move(other); }
        CpuLease& operator=(CpuLease&& other) noexcept {
            if (this != &other) {
                reset();
                owner_ = std::exchange(other.owner_, nullptr);
                address_ = other.address_;
                data_ = std::exchange(other.data_, nullptr);
                bytes_ = other.bytes_;
                writable_ = other.writable_;
            }
            return *this;
        }
        ~CpuLease() { reset(); }
        void* data() const noexcept { return data_; }
        std::size_t bytes() const noexcept { return bytes_; }
        explicit operator bool() const noexcept { return data_ != nullptr; }
        void reset() noexcept {
            if (owner_) owner_->release_cpu(address_, writable_);
            owner_ = nullptr;
            data_ = nullptr;
        }
    private:
        TensorStateLibrary* owner_ = nullptr;
        std::uint64_t address_ = 0;
        void* data_ = nullptr;
        std::size_t bytes_ = 0;
        bool writable_ = false;
    };

    class GpuLease {
    public:
        GpuLease() = default;
        GpuLease(TensorStateLibrary* owner, std::uint64_t address,
                 std::size_t device,
                 VulkanStateBank::Lease lease) noexcept
            : owner_(owner), address_(address), device_(device),
              lease_(std::move(lease)) {}
        GpuLease(const GpuLease&) = delete;
        GpuLease& operator=(const GpuLease&) = delete;
        GpuLease(GpuLease&& other) noexcept { *this = std::move(other); }
        GpuLease& operator=(GpuLease&& other) noexcept {
            if (this != &other) {
                reset();
                owner_ = std::exchange(other.owner_, nullptr);
                address_ = other.address_;
                device_ = other.device_;
                lease_ = std::move(other.lease_);
            }
            return *this;
        }
        ~GpuLease() { reset(); }
        void* handle() const noexcept { return lease_.handle(); }
        std::size_t bytes() const noexcept { return lease_.bytes(); }
        explicit operator bool() const noexcept { return static_cast<bool>(lease_); }
        void reset() noexcept {
            if (owner_) owner_->release_gpu(address_, device_, lease_);
            owner_ = nullptr;
        }
    private:
        TensorStateLibrary* owner_ = nullptr;
        std::uint64_t address_ = 0;
        std::size_t device_ = 0;
        VulkanStateBank::Lease lease_;
    };

    struct Device {
        gp_mem_backend_handle_t context = nullptr;
        std::uint64_t budget_bytes = 0;
    };
    TensorStateLibrary(std::vector<Device> devices,
                       std::uint64_t ram_budget_bytes)
        : ram_budget_bytes_(ram_budget_bytes) {
        for (const auto& device : devices) {
            banks_.push_back(std::make_unique<VulkanStateBank>(
                device.context, device.budget_bytes));
            bank_mutexes_.push_back(std::make_unique<std::mutex>());
        }
    }
    TensorStateLibrary(gp_mem_backend_handle_t vulkan_context,
                       std::uint64_t ram_budget_bytes,
                       std::uint64_t vram_budget_bytes)
        : TensorStateLibrary({Device{vulkan_context, vram_budget_bytes}},
                             ram_budget_bytes) {}
    TensorStateLibrary(const TensorStateLibrary&) = delete;
    TensorStateLibrary& operator=(const TensorStateLibrary&) = delete;

    bool add(std::uint64_t address, Spec spec, const void* initial = nullptr) {
        const auto element_bytes = nodus::tensors::tensor_dtype_size_bytes(spec.dtype);
        if (!address || !element_bytes || spec.shape.empty()) return false;
        std::size_t count = 1;
        for (const auto extent : spec.shape) {
            if (!extent || extent > std::numeric_limits<std::size_t>::max() / count)
                return false;
            count *= static_cast<std::size_t>(extent);
        }
        if (count > std::numeric_limits<std::size_t>::max() /
                         element_bytes) return false;
        const std::size_t bytes = count * element_bytes;
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.count(address) || bytes > ram_budget_bytes_ - ram_bytes_)
            return false;
        Buffer host(gp_mem_backend_create_host(bytes), &gp_mem_backend_release);
        if (!host) return false;
        if (initial) {
            const auto* vt = gp_mem_backend_get_vtable(host.get());
            if (!vt || !vt->copy_to_backend ||
                !vt->copy_to_backend(host.get(), 0, initial, bytes)) return false;
        }
        std::size_t added = 0;
        for (std::size_t i = 0; i < banks_.size(); ++i) {
            std::lock_guard<std::mutex> bank_lock(*bank_mutexes_[i]);
            if (!banks_[i]->add(address, bytes, host.get())) {
                for (std::size_t undo = 0; undo < added; ++undo) {
                    std::lock_guard<std::mutex> undo_lock(*bank_mutexes_[undo]);
                    banks_[undo]->remove(address);
                }
                return false;
            }
            ++added;
        }
        entries_.emplace(address, Entry{std::move(spec), bytes, std::move(host)});
        ram_bytes_ += bytes;
        return true;
    }

    CpuLease lease_cpu(std::uint64_t address, bool writable = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(address);
        if (it == entries_.end() || it->second.leased ||
            !flush_latest(address, it->second))
            return {};
        void* data = gp_mem_backend_map(it->second.host.get());
        if (!data) return {};
        it->second.leased = true;
        return CpuLease(this, address, data, it->second.bytes, writable);
    }

    GpuLease lease_gpu(std::uint64_t address, std::size_t device,
                       bool writable = false) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto it = entries_.find(address);
        if (it == entries_.end() || it->second.leased || device >= banks_.size()) return {};
        if (it->second.latest_gpu >= 0 &&
            it->second.latest_gpu != static_cast<std::int64_t>(device)) {
            if (!flush_latest(address, it->second)) return {};
        }
        it->second.leased = true;
        lock.unlock();
        VulkanStateBank::Lease lease;
        {
            std::lock_guard<std::mutex> bank_lock(*bank_mutexes_[device]);
            lease = banks_[device]->lease(address, writable);
        }
        lock.lock();
        it = entries_.find(address);
        if (it == entries_.end()) return {};
        if (!lease) {
            it->second.leased = false;
            return {};
        }
        it->second.gpu_writable = writable;
        return GpuLease(this, address, device, std::move(lease));
    }
    bool remove(std::uint64_t address) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(address);
        if (it == entries_.end() || it->second.leased ||
            !flush_latest(address, it->second)) return false;
        for (std::size_t i = 0; i < banks_.size(); ++i) {
            std::lock_guard<std::mutex> bank_lock(*bank_mutexes_[i]);
            if (!banks_[i]->remove(address)) return false;
        }
        ram_bytes_ -= it->second.bytes;
        entries_.erase(it);
        return true;
    }

    bool flush_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [address, entry] : entries_)
            if (entry.leased || !flush_latest(address, entry)) return false;
        return true;
    }

    std::size_t device_count() const noexcept { return banks_.size(); }

    std::size_t resident_bytes(std::uint64_t address,std::size_t device) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found=entries_.find(address);
        if(device>=banks_.size() || found==entries_.end() ||
           (found->second.latest_gpu>=0 &&
            found->second.latest_gpu!=static_cast<std::int64_t>(device)))
            return 0; // a newer writer on another device requires a transfer
        std::lock_guard<std::mutex> bank_lock(*bank_mutexes_[device]);
        return banks_[device]->resident_bytes(address);
    }

    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        Stats result{ram_bytes_, ram_budget_bytes_, {}};
        for (std::size_t i = 0; i < banks_.size(); ++i) {
            std::lock_guard<std::mutex> bank_lock(*bank_mutexes_[i]);
            result.vram.push_back(banks_[i]->stats());
        }
        return result;
    }

    std::optional<Spec> describe(std::uint64_t address) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found=entries_.find(address);
        if(found==entries_.end())return {};
        return found->second.spec;
    }

private:
    struct Entry {
        Spec spec;
        std::size_t bytes;
        Buffer host{nullptr, &gp_mem_backend_release};
        bool leased = false;
        bool gpu_writable = false;
        std::int64_t latest_gpu = -1;
        Entry(Spec value, std::size_t size, Buffer home)
            : spec(std::move(value)), bytes(size), host(std::move(home)) {}
    };
    void release_cpu(std::uint64_t address, bool writable) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(address);
        if (it == entries_.end()) return;
        gp_mem_backend_unmap(it->second.host.get());
        if (writable) {
            for (std::size_t i = 0; i < banks_.size(); ++i) {
                std::lock_guard<std::mutex> bank_lock(*bank_mutexes_[i]);
                banks_[i]->invalidate(address);
            }
            it->second.latest_gpu = -1;
        }
        it->second.leased = false;
    }
    void release_gpu(std::uint64_t address, std::size_t device,
                     VulkanStateBank::Lease& lease) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        {
            std::lock_guard<std::mutex> bank_lock(*bank_mutexes_[device]);
            lease.reset();
        }
        auto it = entries_.find(address);
        if (it != entries_.end()) {
            if (it->second.gpu_writable) {
                it->second.latest_gpu = static_cast<std::int64_t>(device);
                for (std::size_t i = 0; i < banks_.size(); ++i)
                    if (i != device) {
                        std::lock_guard<std::mutex> bank_lock(*bank_mutexes_[i]);
                        banks_[i]->invalidate(address);
                    }
            }
            it->second.leased = false;
            it->second.gpu_writable = false;
        }
    }

    bool flush_latest(std::uint64_t address, Entry& entry) {
        if (entry.latest_gpu < 0) return true;
        const auto device=static_cast<std::size_t>(entry.latest_gpu);
        std::lock_guard<std::mutex> bank_lock(*bank_mutexes_[device]);
        if (!banks_[device]->flush(address))
            return false;
        entry.latest_gpu = -1;
        return true;
    }

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<VulkanStateBank>> banks_;
    std::vector<std::unique_ptr<std::mutex>> bank_mutexes_;
    std::uint64_t ram_budget_bytes_, ram_bytes_ = 0;
    std::unordered_map<std::uint64_t, Entry> entries_;
};

} // namespace nodus::gpu
