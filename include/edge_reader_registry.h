#pragma once

#include <algorithm>
#include <array>
#include <atomic>
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
    static constexpr size_t kSlotsPerSegment = 1024;
    static constexpr size_t kSegmentCount = 1024;

    EdgeReaderRegistry() = default;
    ~EdgeReaderRegistry() {
        for (auto& segment : segments_) delete segment.load(std::memory_order_relaxed);
    }
    EdgeReaderRegistry(const EdgeReaderRegistry&) = delete;
    EdgeReaderRegistry& operator=(const EdgeReaderRegistry&) = delete;

    Slot register_reader(uint64_t edge_id) {
        std::lock_guard<std::mutex> lock(mu_);
        if (next_slot_ >= static_cast<Slot>(kSlotsPerSegment * kSegmentCount)) {
            return kInvalidSlot;
        }
        const Slot slot = ++next_slot_;
        const size_t segment_idx = static_cast<size_t>(slot - 1) / kSlotsPerSegment;
        Segment* segment = segments_[segment_idx].load(std::memory_order_relaxed);
        if (!segment) {
            segment = new Segment();
            segments_[segment_idx].store(segment, std::memory_order_release);
        }
        ReaderState* state = &segment->readers[static_cast<size_t>(slot - 1) % kSlotsPerSegment];
        state->edge_id = edge_id;
        state->seq.store(0, std::memory_order_relaxed);
        state->active.store(true, std::memory_order_release);
        slots_.emplace(slot, state);
        return slot;
    }

    void unregister_reader(Slot slot) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto found = slots_.find(slot);
        if (found == slots_.end()) return;
        found->second->active.store(false, std::memory_order_release);
        slots_.erase(found);
    }

    bool update_reader(Slot slot, uint64_t seq) {
        if (slot <= 0 || slot > static_cast<Slot>(kSlotsPerSegment * kSegmentCount)) return false;
        const size_t segment_idx = static_cast<size_t>(slot - 1) / kSlotsPerSegment;
        Segment* segment = segments_[segment_idx].load(std::memory_order_acquire);
        if (!segment) return false;
        ReaderState& state = segment->readers[static_cast<size_t>(slot - 1) % kSlotsPerSegment];
        if (!state.active.load(std::memory_order_acquire)) return false;
        state.seq.store(seq, std::memory_order_release);
        return true;
    }

    uint64_t min_reader_seq(uint64_t edge_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        uint64_t minimum = std::numeric_limits<uint64_t>::max();
        for (const auto& entry : slots_) {
            if (entry.second->edge_id == edge_id) {
                minimum = std::min(minimum, entry.second->seq.load(std::memory_order_acquire));
            }
        }
        return minimum;
    }

    size_t reader_count(uint64_t edge_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        size_t count = 0;
        for (const auto& entry : slots_) {
            if (entry.second->edge_id == edge_id) {
                ++count;
            }
        }
        return count;
    }

private:
    struct ReaderState {
        uint64_t edge_id;
        std::atomic<uint64_t> seq{0};
        std::atomic<bool> active{false};
    };

    struct Segment { std::array<ReaderState, kSlotsPerSegment> readers{}; };

    mutable std::mutex mu_;
    Slot next_slot_ = kInvalidSlot;
    std::array<std::atomic<Segment*>, kSegmentCount> segments_{};
    std::unordered_map<Slot, ReaderState*> slots_;
};
