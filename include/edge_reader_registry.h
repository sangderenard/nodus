#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>

// Domain-neutral accounting for the external reader slots associated with
// runtime edges. This is deliberately independent of ThreadManager, canvas,
// stages, and table rendering so the FIFO runtime can reuse the same
// multi-reader frontier contract without acquiring UI/scheduler dependencies.
//
// EdgeTensorFifo remains the authority for its own local reader cursors. This
// registry is the coordinator-facing mirror used by process schedulers and
// diagnostics. Callers must not use a registry shared by unrelated runtime
// contexts: edge ids are only unique within their owning context.
class EdgeReaderRegistry {
public:
    using Slot = int;
    static constexpr Slot kInvalidSlot = 0;

    Slot register_reader(uint64_t edge_id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (next_slot_ == std::numeric_limits<Slot>::max()) {
            return kInvalidSlot;
        }
        const Slot slot = ++next_slot_;
        slots_.emplace(slot, ReaderState{edge_id, 0});
        return slot;
    }

    void unregister_reader(Slot slot) {
        std::lock_guard<std::mutex> lock(mu_);
        slots_.erase(slot);
    }

    bool update_reader(Slot slot, uint64_t seq) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto found = slots_.find(slot);
        if (found == slots_.end()) {
            return false;
        }
        found->second.seq = seq;
        return true;
    }

    uint64_t min_reader_seq(uint64_t edge_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        uint64_t minimum = std::numeric_limits<uint64_t>::max();
        for (const auto& entry : slots_) {
            if (entry.second.edge_id == edge_id) {
                minimum = std::min(minimum, entry.second.seq);
            }
        }
        return minimum;
    }

    size_t reader_count(uint64_t edge_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        size_t count = 0;
        for (const auto& entry : slots_) {
            if (entry.second.edge_id == edge_id) {
                ++count;
            }
        }
        return count;
    }

private:
    struct ReaderState {
        uint64_t edge_id;
        uint64_t seq;
    };

    mutable std::mutex mu_;
    Slot next_slot_ = kInvalidSlot;
    std::unordered_map<Slot, ReaderState> slots_;
};
