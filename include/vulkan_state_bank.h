#pragma once
// One GPU row's addressable, budgeted resident state. EdgeTensorFifo remains
// the sequencing/transport authority; this bank only leases Vulkan buffers.
#include "kernel_vulkan_runner.h"
#include "mem_backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nodus::gpu {

class VulkanStateBank {
public:
    using Buffer = std::unique_ptr<void, decltype(&gp_mem_backend_release)>;
    struct Stats {
        std::uint64_t resident_bytes = 0, budget_bytes = 0;
        std::uint64_t reused = 0, evicted = 0, restored = 0, saved = 0;
    };
    class Lease {
    public:
        Lease() = default;
        Lease(VulkanStateBank* bank, std::uint64_t address, void* handle,
              std::size_t bytes, bool writable) noexcept
            : bank_(bank), address_(address), handle_(handle), bytes_(bytes),
              writable_(writable) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept { *this = std::move(other); }
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                reset();
                bank_ = std::exchange(other.bank_, nullptr);
                address_ = other.address_;
                handle_ = std::exchange(other.handle_, nullptr);
                bytes_ = other.bytes_;
                writable_ = other.writable_;
            }
            return *this;
        }
        ~Lease() { reset(); }
        void* handle() const noexcept { return handle_; }
        std::size_t bytes() const noexcept { return bytes_; }
        explicit operator bool() const noexcept { return handle_ != nullptr; }
        void reset() noexcept {
            if (bank_) bank_->release(address_, writable_);
            bank_ = nullptr;
            handle_ = nullptr;
        }
    private:
        VulkanStateBank* bank_ = nullptr;
        std::uint64_t address_ = 0;
        void* handle_ = nullptr;
        std::size_t bytes_ = 0;
        bool writable_ = false;
    };

    // The contexts are borrowed from the GPU row and its state library.
    // A backing handle may be host, file, or compatible device memory. It
    // must remain alive until the address is removed or the bank is closed.
    VulkanStateBank(gp_mem_backend_handle_t context, std::uint64_t budget_bytes)
        : context_(context), budget_bytes_(budget_bytes),
          vtable_(gp_mem_backend_get_vtable(context)) {}
    VulkanStateBank(const VulkanStateBank&) = delete;
    VulkanStateBank& operator=(const VulkanStateBank&) = delete;

    static std::uint64_t suggested_budget(gp_mem_backend_handle_t context,
                                          double share = .5) {
        if (!(share > 0 && share <= 1)) return 0;
        kernels::VulkanBackendStatus status;
        std::string error;
        if (!kernels::query_vulkan_backend_status(context, &status, &error)) return 0;
        std::uint64_t free = 0;
        for (const auto& heap : status.heaps) {
            if (!heap.device_local) continue;
            const auto available = status.budget_available
                ? (heap.budget_bytes > heap.usage_bytes
                       ? heap.budget_bytes - heap.usage_bytes : 0)
                : heap.size_bytes / 2;
            free = std::max(free, available);
        }
        return static_cast<std::uint64_t>(free * share);
    }

    bool add(std::uint64_t address, std::size_t bytes,
             gp_mem_backend_handle_t backing = nullptr) {
        if (!address || !bytes || entries_.count(address) ||
            (backing && gp_mem_backend_size(backing) < bytes)) return false;
        entries_.emplace(address, Entry{bytes, backing});
        return true;
    }

    Lease lease(std::uint64_t address, bool writable = false) {
        auto it = entries_.find(address);
        if (it == entries_.end() || it->second.leased || !vtable_ ||
            !vtable_->alloc || !budget_bytes_) return {};
        Entry& entry = it->second;
        if (!entry.buffer) {
            Buffer buffer = take_free(entry.bytes);
            if (!buffer) {
                if (entry.bytes > budget_bytes_) return {};
                while (resident_bytes_ > budget_bytes_ - entry.bytes) {
                    if (evict_one()) {
                        buffer = take_free(entry.bytes);
                        if (buffer) break;
                    }
                    if (!trim_one_free()) return {};
                }
                if (!buffer) {
                    buffer = Buffer(vtable_->alloc(context_, entry.bytes, 0),
                                    &gp_mem_backend_release);
                    if (!buffer) return {};
                    kernels::VulkanBufferAllocationStatus allocation;
                    std::string error;
                    if (!kernels::query_vulkan_buffer_allocation(
                            buffer.get(), &allocation, &error) ||
                        !allocation.device_local ||
                        allocation.allocation_bytes > budget_bytes_ - resident_bytes_)
                        return {};
                    resident_bytes_ += allocation.allocation_bytes;
                }
            }
            if (entry.backing && !gp_mem_backend_transfer(
                    entry.backing, buffer.get(), 0, 0, entry.bytes)) {
                return_free(std::move(buffer));
                return {};
            }
            if (entry.backing) ++restored_;
            entry.buffer = std::move(buffer);
        }
        entry.leased = true;
        entry.last_use = ++clock_;
        return Lease(this, address, entry.buffer.get(), entry.bytes, writable);
    }

    // Dirty state without backing cannot be evicted. A source-only scratch
    // address can be removed explicitly when its consumer is complete.
    bool remove(std::uint64_t address) {
        auto it = entries_.find(address);
        if (it == entries_.end() || it->second.leased) return false;
        if (it->second.dirty && !save(it->second)) return false;
        if (it->second.buffer) return_free(std::move(it->second.buffer));
        entries_.erase(it);
        return true;
    }

    bool flush() {
        for (auto& [_, entry] : entries_)
            if (entry.dirty && (entry.leased || !save(entry))) return false;
        return true;
    }
    bool flush(std::uint64_t address) {
        auto it = entries_.find(address);
        return it != entries_.end() && !it->second.leased && save(it->second);
    }
    // CPU has published a newer value in the backing handle. Drop the stale
    // resident copy; the next GPU lease restores from that backing.
    bool invalidate(std::uint64_t address) {
        auto it = entries_.find(address);
        if (it == entries_.end() || it->second.leased) return false;
        if (it->second.buffer) return_free(std::move(it->second.buffer));
        it->second.dirty = false;
        return true;
    }
    Stats stats() const noexcept {
        return {resident_bytes_, budget_bytes_, reused_, evicted_, restored_, saved_};
    }
    std::size_t resident_bytes(std::uint64_t address) const noexcept {
        const auto it = entries_.find(address);
        return it != entries_.end() && it->second.buffer ? it->second.bytes : 0;
    }

private:
    struct Entry {
        std::size_t bytes;
        gp_mem_backend_handle_t backing;
        Buffer buffer{nullptr, &gp_mem_backend_release};
        bool leased = false, dirty = false;
        std::uint64_t last_use = 0;
        Entry(std::size_t size, gp_mem_backend_handle_t home)
            : bytes(size), backing(home) {}
    };
    struct FreeBuffer {
        Buffer buffer{nullptr, &gp_mem_backend_release};
        std::size_t bytes = 0;
        std::uint64_t allocation_bytes = 0;
    };
    void release(std::uint64_t address, bool dirty) noexcept {
        auto it = entries_.find(address);
        if (it != entries_.end()) {
            it->second.leased = false;
            it->second.dirty |= dirty;
            it->second.last_use = ++clock_;
        }
    }
    bool save(Entry& entry) {
        if (!entry.dirty) return true;
        if (!entry.backing || !entry.buffer ||
            !gp_mem_backend_transfer(entry.buffer.get(), entry.backing,
                                     0, 0, entry.bytes)) return false;
        entry.dirty = false;
        ++saved_;
        return true;
    }
    void return_free(Buffer buffer) {
        if (!buffer) return;
        kernels::VulkanBufferAllocationStatus allocation;
        std::string error;
        const auto bytes = gp_mem_backend_size(buffer.get());
        if (!kernels::query_vulkan_buffer_allocation(buffer.get(), &allocation,
                                                      &error)) return;
        free_.push_back({std::move(buffer), bytes, allocation.allocation_bytes});
    }
    bool trim_one_free() {
        if (free_.empty()) return false;
        resident_bytes_ -= free_.back().allocation_bytes;
        free_.pop_back();
        return true;
    }
    Buffer take_free(std::size_t bytes) {
        auto it = std::find_if(free_.begin(), free_.end(),
                               [bytes](const FreeBuffer& b) { return b.bytes >= bytes; });
        if (it == free_.end()) return Buffer(nullptr, &gp_mem_backend_release);
        Buffer result = std::move(it->buffer);
        free_.erase(it);
        ++reused_;
        return result;
    }
    bool evict_one() {
        auto victim = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it)
            if (it->second.buffer && !it->second.leased &&
                (!it->second.dirty || it->second.backing) &&
                (victim == entries_.end() ||
                 it->second.last_use < victim->second.last_use)) victim = it;
        if (victim == entries_.end() || !save(victim->second)) return false;
        return_free(std::move(victim->second.buffer));
        ++evicted_;
        return true;
    }

    gp_mem_backend_handle_t context_;
    std::uint64_t budget_bytes_, resident_bytes_ = 0, clock_ = 0;
    const gp_mem_backend_vtable_t* vtable_;
    std::unordered_map<std::uint64_t, Entry> entries_;
    std::vector<FreeBuffer> free_;
    std::uint64_t reused_ = 0, evicted_ = 0, restored_ = 0, saved_ = 0;
};

} // namespace nodus::gpu
