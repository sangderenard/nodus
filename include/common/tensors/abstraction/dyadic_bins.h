// dyadic binning (tensor-backed)
#pragma once

#include "common/tensors/abstraction/abstract_tensor_pool.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/stream_add.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <climits>
#include <limits>
#include <type_traits>
#include <cstdio>
#include <typeinfo>
#include <cstdlib>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <array>
#include <utility>

#include "common/tensors/abstraction/tensor_types.h"
#include "common/thread_pool.h"

#ifndef WIN32
#if defined(_WIN32) || defined(_WIN64)
#define WIN32 1
#else
#define WIN32 0
#endif
#endif

#if WIN32
#include <intrin.h>
#endif

#define DYADIC_RAW_WRITE(dst, src, len) std::memcpy((dst), (src), (len))

namespace nodus::tensors {
#define UNINITIALIZED_STAGE_OFFSET -1
#define UNCLAIMED_STAGE_OFFSET -3
#define EXPIRED_STAGE_OFFSET -2
#define STAGE_OFFSET_IN_USE -4

namespace policies {
struct Overwrite;
}

template <typename INDEX_T, typename VALUE_T>
struct DyadicBinIO3 {
    uint8_t*  bin_data = nullptr;
    uint32_t* page_count = nullptr;
    uint8_t*  page_active = nullptr;
    uint8_t*  page_retain = nullptr;
    uint8_t*  page_inbox = nullptr;
    uint32_t  index_count = 0;
    uint32_t  elem_bytes = 0;
    uint32_t  slot_bytes = 0;
    uint32_t  value_stride = 1;
    uint32_t  value_bytes = 0;
    uint32_t  total_bins = 0;

    inline uint8_t* page_base(uint32_t bin, uint8_t page) const {
        const uint64_t page_index = (uint64_t)bin * 3ull + (uint64_t)page;
        return bin_data + page_index * (uint64_t)index_count * (uint64_t)slot_bytes;
    }

    inline uint32_t& cnt(uint32_t bin, uint8_t page) const {
        return page_count[(uint64_t)bin * 3ull + (uint64_t)page];
    }

    inline uint8_t* slot_ptr(uint32_t bin, uint8_t page, uint32_t slot) const {
        return page_base(bin, page) + (uint64_t)slot * (uint64_t)slot_bytes;
    }

    inline void write_slot_to_page(uint32_t bin, uint8_t page, INDEX_T idx, const VALUE_T* vals) {
        uint32_t slot = cnt(bin, page)++;
        uint8_t* p = slot_ptr(bin, page, slot);
        DYADIC_RAW_WRITE(p, &idx, sizeof(INDEX_T));
        DYADIC_RAW_WRITE(p + elem_bytes, vals, (size_t)value_bytes);
    }

    inline void write_slot_to_page_bytes(uint32_t bin, uint8_t page, INDEX_T idx, const uint8_t* val_bytes) {
        uint32_t slot = cnt(bin, page)++;
        uint8_t* p = slot_ptr(bin, page, slot);
        DYADIC_RAW_WRITE(p, &idx, sizeof(INDEX_T));
        DYADIC_RAW_WRITE(p + elem_bytes, val_bytes, (size_t)value_bytes);
    }

    inline void push_inbox(uint32_t bin, INDEX_T idx, const VALUE_T* vals) {
        write_slot_to_page(bin, page_inbox[bin], idx, vals);
    }

    inline void push_retain(uint32_t bin, INDEX_T idx, const VALUE_T* vals) {
        write_slot_to_page(bin, page_retain[bin], idx, vals);
    }

    inline void push_inbox_bytes(uint32_t bin, INDEX_T idx, const uint8_t* val_bytes) {
        write_slot_to_page_bytes(bin, page_inbox[bin], idx, val_bytes);
    }

    inline void push_retain_bytes(uint32_t bin, INDEX_T idx, const uint8_t* val_bytes) {
        write_slot_to_page_bytes(bin, page_retain[bin], idx, val_bytes);
    }

    inline void prepare_for_scan(uint32_t bin) {
        uint8_t A = page_active[bin];
        uint8_t R = page_retain[bin];
        uint8_t I = page_inbox[bin];
        uint32_t cntA = cnt(bin, A);
        uint32_t cntI = cnt(bin, I);
        cnt(bin, R) = 0;
        if (cntI == 0) {
            return;
        }
        if (cntA == 0) {
            page_active[bin] = I;
            page_retain[bin] = A;
            page_inbox[bin] = R;
            cnt(bin, page_retain[bin]) = 0;
            cnt(bin, page_inbox[bin]) = 0;
            return;
        }
        page_inbox[bin] = R;
        page_retain[bin] = I;
        uint8_t I_old = I;
        uint32_t base = cntA;
        uint32_t add = cntI;
        DYADIC_RAW_WRITE(
            page_base(bin, A) + (uint64_t)base * slot_bytes,
            page_base(bin, I_old),
            (size_t)add * (size_t)slot_bytes);
        cnt(bin, A) = base + add;
        cnt(bin, I_old) = 0;
        cnt(bin, page_inbox[bin]) = 0;
    }

    inline void finish_scan(uint32_t bin) {
        uint8_t A = page_active[bin];
        uint8_t R = page_retain[bin];
        uint8_t I = page_inbox[bin];
        const uint32_t kept = cnt(bin, R);
        const uint32_t add = cnt(bin, I);
        page_active[bin] = R;
        if (add) {
            DYADIC_RAW_WRITE(
                page_base(bin, R) + (uint64_t)kept * slot_bytes,
                page_base(bin, I),
                (size_t)add * (size_t)slot_bytes);
            cnt(bin, R) = kept + add;
        }
        page_inbox[bin] = A;
        page_retain[bin] = I;
        cnt(bin, A) = 0;
        cnt(bin, I) = 0;
    }

    inline uint8_t active_page(uint32_t bin) const { return page_active[bin]; }
    inline uint32_t active_count(uint32_t bin) const { return cnt(bin, page_active[bin]); }
};

struct DyadicStagePage {
    AbstractTensorHandle tensor{};
    std::atomic<int32_t> write_offset = -1;
    void* mapped = nullptr;
    size_t mapped_bytes = 0;
    uint64_t byte_offset = 0;
};

enum class DyadicStageReleasePolicy : uint8_t {
    Hold = 0,
    Reuse = 1,
    Expire = 2
};

struct DyadicStagePagePool {
    DyadicStagePage pages[256];
    std::atomic<int32_t> earliest_spot = 0;
    std::atomic<uint32_t> page_count = 0;
    AbstractTensorPool* tensor_pool = nullptr;
    TensorBackend* backend = nullptr;
    InMemoryBackend* mem = nullptr;
    TensorDesc page_desc{};
    TensorDesc backing_desc{};
    AbstractTensorHandle backing_tensor{};
    void* backing_ptr = nullptr;
    size_t backing_bytes = 0;
    uint64_t page_bytes = 0;
    uint32_t max_pages = 0;
    bool use_backing = false;
    bool backing_external = false;
    uint32_t total_bins = 0;
    uint32_t stage_entries = 0;
    std::vector<int32_t> page_for_bin;
    std::vector<uint8_t> dirty;

    inline void init(AbstractTensorPool* pool,
                     TensorBackend* backend_in,
                     InMemoryBackend* mem_in,
                     const TensorDesc& page_desc_in,
                     uint32_t total_bins_in,
                     uint32_t stage_entries_in,
                     uint64_t page_bytes_in,
                     uint32_t max_pages_in,
                     AbstractTensorHandle backing_handle = {},
                     const TensorDesc* backing_desc_in = nullptr,
                     bool backing_is_external = false,
                     bool clear_backing = true) {
        tensor_pool = pool;
        backend = backend_in;
        mem = mem_in;
        page_desc = page_desc_in;
        total_bins = total_bins_in;
        stage_entries = stage_entries_in;
        page_bytes = page_bytes_in;
        const uint32_t max_possible_pages = (total_bins == 0u)
            ? 1u
            : (static_cast<uint32_t>(1u) << total_bins) + 1u;
        max_pages = max_pages_in == 0 ? max_possible_pages : std::min<uint32_t>(max_pages_in, max_possible_pages);
        use_backing = true;
        backing_external = backing_is_external;

        if (!mem || !use_backing) {
            max_pages = 0;
        } else if (abstract_tensor_handle_is_valid(backing_handle)) {
            backing_tensor = backing_handle;
            if (backing_desc_in) {
                backing_desc = *backing_desc_in;
            } else {
                backing_desc = page_desc;
            }
        } else {
            if (!tensor_pool || !backend) {
                max_pages = 0;
            } else {
                backing_desc = page_desc;
                backing_desc.shape.dims = {
                    static_cast<uint32_t>(stage_entries * std::max<uint32_t>(1u, max_pages))
                };
                AbstractTensor backing = tensor_pool->acquire_tensor(backing_desc, backend);
                backing_tensor = backing.handle();
            }
        }

        if (clear_backing && abstract_tensor_handle_is_valid(backing_tensor)) {
            mem->ensure_zeroed(backing_tensor, backing_desc, backing_external, 0u, 0u);
        }
        if (abstract_tensor_handle_is_valid(backing_tensor)) {
            mem->map(backing_tensor, &backing_ptr, &backing_bytes);
        }

        page_for_bin.assign(total_bins, -1);
        dirty.assign(total_bins, 0u);
        page_count.store(0u);
        earliest_spot.store(0);
        for (uint32_t i = 0; i < 256u; ++i) {
            pages[i].tensor = {};
            pages[i].byte_offset = page_bytes * static_cast<uint64_t>(i);
            if (backing_ptr && i < max_pages) {
                pages[i].mapped = static_cast<uint8_t*>(backing_ptr) + pages[i].byte_offset;
                pages[i].mapped_bytes = static_cast<size_t>(page_bytes);
                pages[i].write_offset.store(UNINITIALIZED_STAGE_OFFSET);
            } else {
                pages[i].mapped = nullptr;
                pages[i].mapped_bytes = 0;
                pages[i].write_offset.store(STAGE_OFFSET_IN_USE);
            }
        }
    }

    template <typename VALUE_T>
    inline VALUE_T* stage_ptr(uint32_t bin) {
        if (!mem || bin >= total_bins || max_pages == 0) return nullptr;
        int32_t idx = page_for_bin[bin];
        if (idx < 0) {
            DyadicStagePage* page = add_new_page<VALUE_T>(this, bin);
            if (!page) return nullptr;
            idx = (int32_t)(page - pages);
            page_for_bin[bin] = idx;
        }
        DyadicStagePage& page = pages[idx];
        if (!page.mapped) return nullptr;
        return static_cast<VALUE_T*>(page.mapped);
    }

    inline void mark_dirty(uint32_t bin) {
        if (bin < dirty.size()) {
            dirty[bin] = 1u;
        }
    }

    inline bool try_claim_ready(uint32_t page_index, int32_t* out_offset) {
        if (page_index >= max_pages || !out_offset) return false;
        int32_t offset = pages[page_index].write_offset.load(std::memory_order_acquire);
        while (offset >= 0) {
            int32_t expected = offset;
            if (pages[page_index].write_offset.compare_exchange_strong(
                    expected, STAGE_OFFSET_IN_USE,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                *out_offset = offset;
                return true;
            }
            offset = expected;
        }
        return false;
    }

    inline void finish_page(uint32_t page_index, DyadicStageReleasePolicy policy) {
        if (page_index >= max_pages) return;
        switch (policy) {
            case DyadicStageReleasePolicy::Reuse:
                pages[page_index].write_offset.store(UNCLAIMED_STAGE_OFFSET, std::memory_order_release);
                break;
            case DyadicStageReleasePolicy::Expire:
                pages[page_index].write_offset.store(EXPIRED_STAGE_OFFSET, std::memory_order_release);
                break;
            case DyadicStageReleasePolicy::Hold:
            default:
                break;
        }
    }

    inline bool clear_page(uint32_t page_index) {
        if (!mem || !use_backing || !abstract_tensor_handle_is_valid(backing_tensor)) return false;
        if (page_index >= max_pages) return false;
        const uint64_t offset = pages[page_index].byte_offset;
        return mem->ensure_zeroed(backing_tensor, backing_desc, true, offset, page_bytes);
    }

    template <typename OutmixPol, typename VALUE_T>
    inline void emit_all(VALUE_T* output,
                         uint32_t* output_offset,
                         DyadicStageReleasePolicy release_policy = DyadicStageReleasePolicy::Expire) {
        if (!output || !output_offset) return;
        const uint32_t total = total_bins;
        for (uint32_t bin = 0; bin < total; ++bin) {
            if (!dirty[bin]) continue;
            const int32_t idx = page_for_bin[bin];
            if (idx < 0) continue;
            DyadicStagePage& page = pages[idx];
            if (!page.mapped) continue;
            VALUE_T* out = output + *output_offset;
            if constexpr (std::is_same_v<OutmixPol, ::nodus::tensors::policies::Overwrite>) {
                DYADIC_RAW_WRITE(out, page.mapped, (uint32_t)(stage_entries * sizeof(VALUE_T)));
            } else {
                OutmixPol::apply_span(out, reinterpret_cast<const uint8_t*>(page.mapped),
                                      (uint32_t)sizeof(VALUE_T), stage_entries);
            }
            *output_offset = (uint32_t)(*output_offset + stage_entries);
            dirty[bin] = 0u;
            if (!use_backing && page.mapped) {
                mem->unmap(page.tensor);
                page.mapped = nullptr;
                page.mapped_bytes = 0;
            }
            finish_page(static_cast<uint32_t>(idx), release_policy);
        }
    }
};

template <typename VALUE_T>
inline uint32_t find_earliest_spot(
    DyadicStagePagePool* pool,
    uint32_t start,
    uint64_t requested_offset)
{
    // NOTE: leaving your "scan ring" style intact, but making it correct.
    // We will scan at most max_pages slots starting at 'start' (mod max_pages).
    if (!pool || pool->max_pages == 0) return std::numeric_limits<uint32_t>::max();

    uint32_t clean = 0;
    const uint32_t max_pages = pool->max_pages;

    for (uint32_t step = 0; step < max_pages; ++step) {
        const uint32_t idx = (start + step) % max_pages;
        const int32_t offset = pool->pages[idx].write_offset.load();

        if (offset < 0) {
            if (offset == STAGE_OFFSET_IN_USE) {
                // busy, do nothing
            }
            else if (offset == UNCLAIMED_STAGE_OFFSET) {
                int32_t expected = UNCLAIMED_STAGE_OFFSET;
                clean = pool->pages[idx].write_offset.compare_exchange_strong(expected, STAGE_OFFSET_IN_USE);
                if (clean) {
                    // finalize claim immediately
                    expected = STAGE_OFFSET_IN_USE;
                    const bool success = pool->pages[idx].write_offset.compare_exchange_strong(
                        expected, (int32_t)requested_offset);
                    if (!success) {
                        fprintf(stderr, "Dyadic binning: failed to claim stage page %u (saw %d)\n", (unsigned)idx, expected);
                        fprintf(stderr, "This is a multithreading failure.\n");
                        return std::numeric_limits<uint32_t>::max();
                    }
                    return (uint32_t)idx;
                }
            }
            else if (offset == EXPIRED_STAGE_OFFSET) {
                int32_t expected = EXPIRED_STAGE_OFFSET;
                clean = pool->pages[idx].write_offset.compare_exchange_strong(expected, STAGE_OFFSET_IN_USE);
                if (clean) {
                    if (!pool->clear_page(idx)) {
                        return std::numeric_limits<uint32_t>::max();
                    }

                    expected = STAGE_OFFSET_IN_USE;
                    const bool success = pool->pages[idx].write_offset.compare_exchange_strong(
                        expected, (int32_t)requested_offset);
                    if (!success) {
                        fprintf(stderr, "Dyadic binning: failed to claim stage page %u (saw %d)\n", (unsigned)idx, expected);
                        fprintf(stderr, "This is a multithreading failure.\n");
                        return std::numeric_limits<uint32_t>::max();
                    }
                    return (uint32_t)idx;
                }
            }
            else if (offset == UNINITIALIZED_STAGE_OFFSET) {
                int32_t expected = UNINITIALIZED_STAGE_OFFSET;
                clean = pool->pages[idx].write_offset.compare_exchange_strong(expected, STAGE_OFFSET_IN_USE);
                if (clean) {
                    if (!pool->clear_page(idx)) {
                        return std::numeric_limits<uint32_t>::max();
                    }

                    expected = STAGE_OFFSET_IN_USE;
                    const bool success = pool->pages[idx].write_offset.compare_exchange_strong(
                        expected, (int32_t)requested_offset);
                    if (!success) {
                        fprintf(stderr, "Dyadic binning: failed to claim stage page %u (saw %d)\n", (unsigned)idx, expected);
                        fprintf(stderr, "This is a multithreading failure.\n");
                        return std::numeric_limits<uint32_t>::max();
                    }
                    return (uint32_t)idx;
                }
            }
        }
    }
    return std::numeric_limits<uint32_t>::max();
}


template <typename VALUE_T>
inline DyadicStagePage* add_new_page(DyadicStagePagePool* pool, uint32_t offset) {
    if (!pool || pool->max_pages == 0) return nullptr;
    uint32_t earliest_spot = pool->earliest_spot.load();
    if (earliest_spot == std::numeric_limits<uint32_t>::max()) {
        earliest_spot = 0;
        pool->earliest_spot.store(find_earliest_spot<VALUE_T>(pool, earliest_spot, offset));
        earliest_spot = pool->earliest_spot.load();
        if (earliest_spot == std::numeric_limits<uint32_t>::max()) {
            return nullptr;
        }
    } else {
        earliest_spot = find_earliest_spot<VALUE_T>(pool, earliest_spot, offset);
        pool->earliest_spot.store(earliest_spot);
    }
    
    return &pool->pages[earliest_spot];

}


#ifndef MAX_BINS
#define MAX_BINS 3
#endif
static_assert(MAX_BINS >= 3, "MAX_BINS must be at least 3.");

#define DYADIC_BIN_SORT_BASE 1000u
#define DYADIC_BIN_SWAP_INBOX_BASE (DYADIC_BIN_SORT_BASE + MAX_BINS)
#define DYADIC_BIN_SWAP_RETAIN_BASE (DYADIC_BIN_SWAP_INBOX_BASE + MAX_BINS)

#define DYADIC_ENUM_BIN_SORT(n) Bin##n##Sort,
#define DYADIC_ENUM_BIN_SWAP_INBOX(n) Bin##n##SwapInbox,
#define DYADIC_ENUM_BIN_SWAP_RETAIN(n) Bin##n##SwapRetain,

#if MAX_BINS > 1
#define DYADIC_FOR_EACH_BIN_FROM_1(macro) \
    macro(1) \
    IF_DYADIC_BIN_2(macro) \
    IF_DYADIC_BIN_3(macro) \
    IF_DYADIC_BIN_4(macro) \
    IF_DYADIC_BIN_5(macro) \
    IF_DYADIC_BIN_6(macro) \
    IF_DYADIC_BIN_7(macro) \
    IF_DYADIC_BIN_8(macro) \
    IF_DYADIC_BIN_9(macro) \
    IF_DYADIC_BIN_10(macro) \
    IF_DYADIC_BIN_11(macro) \
    IF_DYADIC_BIN_12(macro) \
    IF_DYADIC_BIN_13(macro) \
    IF_DYADIC_BIN_14(macro) \
    IF_DYADIC_BIN_15(macro) \
    IF_DYADIC_BIN_16(macro)
#else
#define DYADIC_FOR_EACH_BIN_FROM_1(macro)
#endif

#if MAX_BINS > 2
#define IF_DYADIC_BIN_2(macro) macro(2)
#else
#define IF_DYADIC_BIN_2(macro)
#endif

#if MAX_BINS > 3
#define IF_DYADIC_BIN_3(macro) macro(3)
#else
#define IF_DYADIC_BIN_3(macro)
#endif

#if MAX_BINS > 4
#define IF_DYADIC_BIN_4(macro) macro(4)
#else
#define IF_DYADIC_BIN_4(macro)
#endif

#if MAX_BINS > 5
#define IF_DYADIC_BIN_5(macro) macro(5)
#else
#define IF_DYADIC_BIN_5(macro)
#endif

#if MAX_BINS > 6
#define IF_DYADIC_BIN_6(macro) macro(6)
#else
#define IF_DYADIC_BIN_6(macro)
#endif

#if MAX_BINS > 7
#define IF_DYADIC_BIN_7(macro) macro(7)
#else
#define IF_DYADIC_BIN_7(macro)
#endif

#if MAX_BINS > 8
#define IF_DYADIC_BIN_8(macro) macro(8)
#else
#define IF_DYADIC_BIN_8(macro)
#endif

#if MAX_BINS > 9
#define IF_DYADIC_BIN_9(macro) macro(9)
#else
#define IF_DYADIC_BIN_9(macro)
#endif

#if MAX_BINS > 10
#define IF_DYADIC_BIN_10(macro) macro(10)
#else
#define IF_DYADIC_BIN_10(macro)
#endif

#if MAX_BINS > 11
#define IF_DYADIC_BIN_11(macro) macro(11)
#else
#define IF_DYADIC_BIN_11(macro)
#endif

#if MAX_BINS > 12
#define IF_DYADIC_BIN_12(macro) macro(12)
#else
#define IF_DYADIC_BIN_12(macro)
#endif

#if MAX_BINS > 13
#define IF_DYADIC_BIN_13(macro) macro(13)
#else
#define IF_DYADIC_BIN_13(macro)
#endif

#if MAX_BINS > 14
#define IF_DYADIC_BIN_14(macro) macro(14)
#else
#define IF_DYADIC_BIN_14(macro)
#endif

#if MAX_BINS > 15
#define IF_DYADIC_BIN_15(macro) macro(15)
#else
#define IF_DYADIC_BIN_15(macro)
#endif

#if MAX_BINS > 16
#define IF_DYADIC_BIN_16(macro) macro(16)
#else
#define IF_DYADIC_BIN_16(macro)
#endif

enum class DyadicScheduleOp : uint32_t {
    GlobalSort,
    StageWrite,
    LoopSentinel,
    Bin0Sort = DYADIC_BIN_SORT_BASE,
    DYADIC_FOR_EACH_BIN_FROM_1(DYADIC_ENUM_BIN_SORT)
    Bin0SwapInbox = DYADIC_BIN_SWAP_INBOX_BASE,
    DYADIC_FOR_EACH_BIN_FROM_1(DYADIC_ENUM_BIN_SWAP_INBOX)
    Bin0SwapRetain = DYADIC_BIN_SWAP_RETAIN_BASE,
    DYADIC_FOR_EACH_BIN_FROM_1(DYADIC_ENUM_BIN_SWAP_RETAIN)
};

#define DYADIC_OP_BIN_SORT(n) DyadicScheduleOp::Bin##n##Sort
#define DYADIC_OP_BIN_SWAP_INBOX(n) DyadicScheduleOp::Bin##n##SwapInbox
#define DYADIC_OP_BIN_SWAP_RETAIN(n) DyadicScheduleOp::Bin##n##SwapRetain

struct DyadicScheduleWriter {
    DyadicScheduleOp* out = nullptr;
    uint32_t count = 0;

    constexpr void emit(DyadicScheduleOp op) {
        if (out) {
            out[count] = op;
        }
        ++count;
    }
};

template <uint32_t Bin>
consteval void dyadic_emit_lower_cycle(DyadicScheduleWriter& w) {
    if constexpr (Bin > 0u) {
        w.emit(DYADIC_OP_BIN_SWAP_INBOX(Bin));
        w.emit(DYADIC_OP_BIN_SORT(Bin));
        w.emit(DYADIC_OP_BIN_SWAP_RETAIN(Bin));
        dyadic_emit_lower_cycle<Bin - 1u>(w);
    } else {
        w.emit(DYADIC_OP_BIN_SWAP_INBOX(0));
        w.emit(DYADIC_OP_BIN_SORT(0));
        w.emit(DyadicScheduleOp::StageWrite);
    }
}

template <uint32_t Bin, uint32_t MaxBins>
consteval void dyadic_emit_recursive_bins(DyadicScheduleWriter& w) {
    if constexpr (Bin < MaxBins) {
        w.emit(DYADIC_OP_BIN_SORT(Bin));
        w.emit(DYADIC_OP_BIN_SWAP_RETAIN(Bin));
        dyadic_emit_lower_cycle<Bin - 1u>(w);
        dyadic_emit_lower_cycle<Bin - 1u>(w);
        dyadic_emit_recursive_bins<Bin + 1u, MaxBins>(w);
    }
}

template <uint32_t MaxBins>
consteval void dyadic_emit_single_thread_schedule(DyadicScheduleWriter& w) {
    // MSB STRIPPING IS INTERNAL TO OPERATIONS BUT VITAL
    // FOR THIS REASON BIN SORTING IS STATEFUL

    // INITIAL SORT YEILDS 1 STAGE AND ALL BINS
    w.emit(DyadicScheduleOp::GlobalSort);
    // WRITE THE STAGE
    w.emit(DyadicScheduleOp::StageWrite);
    // BIN 0 ALWAYS SORTS INTO STAGE
    w.emit(DYADIC_OP_BIN_SORT(0));
    w.emit(DyadicScheduleOp::StageWrite);

    if constexpr (MaxBins > 1u) {
        // NOW BIN 0 IS EMPTY, BIN 1 IS FULL
        w.emit(DYADIC_OP_BIN_SORT(1));
        w.emit(DYADIC_OP_BIN_SWAP_RETAIN(1));
        w.emit(DYADIC_OP_BIN_SWAP_INBOX(0));
        w.emit(DYADIC_OP_BIN_SORT(0));
        w.emit(DyadicScheduleOp::StageWrite);
        // NOW BIN 0 IS EMPTY, BIN 1 HAS 1/2 ON RETAIN SWAPPED TO CONTENTS
        w.emit(DYADIC_OP_BIN_SORT(1));
        w.emit(DYADIC_OP_BIN_SWAP_INBOX(0));
        w.emit(DYADIC_OP_BIN_SORT(0));
        w.emit(DyadicScheduleOp::StageWrite);
    }

    if constexpr (MaxBins > 2u) {
        // NOW BIN 0 IS EMPTY, BIN 1 IS EMPTY, BIN 2 IS FULL
        w.emit(DYADIC_OP_BIN_SORT(2));
        w.emit(DYADIC_OP_BIN_SWAP_RETAIN(2));
        // NOW BIN 2 HAS 2/4 ON RETAIN SWAPPED TO CONTENTS
        w.emit(DYADIC_OP_BIN_SWAP_INBOX(1));
        w.emit(DYADIC_OP_BIN_SORT(1));
        w.emit(DYADIC_OP_BIN_SWAP_RETAIN(1));
        w.emit(DYADIC_OP_BIN_SWAP_INBOX(0));
        w.emit(DYADIC_OP_BIN_SORT(0));
        w.emit(DyadicScheduleOp::StageWrite);
        // NOW BIN 0 IS EMPTY, BIN 1 IS 1/2 ON RETAIN SWAPPED TO CONTENTS
        w.emit(DYADIC_OP_BIN_SORT(1));
        w.emit(DYADIC_OP_BIN_SWAP_INBOX(0));
        w.emit(DYADIC_OP_BIN_SORT(0));
        w.emit(DyadicScheduleOp::StageWrite);
        // NOW BIN 0 IS EMPTY, BIN 1 IS EMPTY, BIN 2 IS 2/4 ON RETAIN SWAPPED TO CONTENTS
        w.emit(DYADIC_OP_BIN_SORT(2));
        w.emit(DYADIC_OP_BIN_SWAP_RETAIN(2));
        w.emit(DYADIC_OP_BIN_SWAP_INBOX(1));
        w.emit(DYADIC_OP_BIN_SORT(1));
        w.emit(DYADIC_OP_BIN_SWAP_RETAIN(1));
        w.emit(DYADIC_OP_BIN_SWAP_INBOX(0));
        w.emit(DYADIC_OP_BIN_SORT(0));
        w.emit(DyadicScheduleOp::StageWrite);
        // NOW THE PATTERN SETS IN
        w.emit(DYADIC_OP_BIN_SORT(1));
        w.emit(DYADIC_OP_BIN_SWAP_RETAIN(1));
        w.emit(DYADIC_OP_BIN_SWAP_INBOX(0));
        w.emit(DYADIC_OP_BIN_SORT(0));
        w.emit(DyadicScheduleOp::StageWrite);
    }
    if constexpr (MaxBins > 3u) {
        dyadic_emit_recursive_bins<3u, MaxBins>(w);
    }
}

template <uint32_t MaxBins>
consteval size_t dyadic_single_thread_schedule_size() {
    DyadicScheduleWriter counter{};
    dyadic_emit_single_thread_schedule<MaxBins>(counter);
    return (size_t)counter.count;
}

template <uint32_t MaxBins>
consteval auto dyadic_make_single_thread_schedule() {
    std::array<DyadicScheduleOp, dyadic_single_thread_schedule_size<MaxBins>()> ops{};
    DyadicScheduleWriter writer{ ops.data(), 0u };
    dyadic_emit_single_thread_schedule<MaxBins>(writer);
    return ops;
}

template <uint32_t MaxBins>
struct DyadicSingleThreadSchedule {
    static constexpr auto ops = dyadic_make_single_thread_schedule<MaxBins>();
    static constexpr uint32_t count = (uint32_t)ops.size();
};

constexpr auto single_threaded_dyadic_schedule = DyadicSingleThreadSchedule<MAX_BINS>::ops;

template <uint32_t MaxBins, typename Fn, size_t... I>
constexpr void dyadic_for_each_schedule_op_impl(Fn&& fn, std::index_sequence<I...>) {
    (fn(std::integral_constant<DyadicScheduleOp, DyadicSingleThreadSchedule<MaxBins>::ops[I]>{}), ...);
}

template <uint32_t MaxBins, typename Fn>
constexpr void dyadic_for_each_schedule_op(Fn&& fn) {
    dyadic_for_each_schedule_op_impl<MaxBins>(
        std::forward<Fn>(fn),
        std::make_index_sequence<DyadicSingleThreadSchedule<MaxBins>::ops.size()>{});
}

#undef DYADIC_OP_BIN_SWAP_RETAIN
#undef DYADIC_OP_BIN_SWAP_INBOX
#undef DYADIC_OP_BIN_SORT
#undef DYADIC_ENUM_BIN_SWAP_RETAIN
#undef DYADIC_ENUM_BIN_SWAP_INBOX
#undef DYADIC_ENUM_BIN_SORT
#undef IF_DYADIC_BIN_16
#undef IF_DYADIC_BIN_15
#undef IF_DYADIC_BIN_14
#undef IF_DYADIC_BIN_13
#undef IF_DYADIC_BIN_12
#undef IF_DYADIC_BIN_11
#undef IF_DYADIC_BIN_10
#undef IF_DYADIC_BIN_9
#undef IF_DYADIC_BIN_8
#undef IF_DYADIC_BIN_7
#undef IF_DYADIC_BIN_6
#undef IF_DYADIC_BIN_5
#undef IF_DYADIC_BIN_4
#undef IF_DYADIC_BIN_3
#undef IF_DYADIC_BIN_2
#undef DYADIC_FOR_EACH_BIN_FROM_1
#undef DYADIC_BIN_SWAP_RETAIN_BASE
#undef DYADIC_BIN_SWAP_INBOX_BASE
#undef DYADIC_BIN_SORT_BASE

const int32_t ALL_BINS = UINT32_MAX;
const int32_t NO_BINS = -2;
const int32_t INITIAL_INPUT_BIN = -1;

static constexpr uint32_t kBinSortCount = MAX_BINS;

inline DyadicScheduleOp dyadic_bin_swap_inbox_op(uint32_t bin);
inline DyadicScheduleOp dyadic_bin_swap_retain_op(uint32_t bin);

inline void dyadic_exec_swap_inbox(
    uint32_t bin_id,
    std::atomic<uint8_t>* duty_state,
    uint8_t* page_active,
    uint8_t* page_inbox,
    uint8_t* page_retain);

inline void dyadic_exec_swap_retain(
    uint32_t bin_id,
    std::atomic<uint8_t>* duty_state,
    uint8_t* page_active,
    uint8_t* page_inbox,
    uint8_t* page_retain);

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_exec_bin_sort(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* phase_counter);

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline bool run_dyadic_schedule_step(
    DyadicScheduleOp op,
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    std::atomic<uint8_t>* duty_state,
    uint8_t* phase_counter)
{
    const uint32_t u = static_cast<uint32_t>(op);
    const uint32_t base = static_cast<uint32_t>(DyadicScheduleOp::Bin0Sort);
    if (u >= base && u < base + kBinSortCount) {
        const uint32_t bin = u - base;
        dyadic_exec_bin_sort<INDEX_T, VALUE_T, PremixPol>(
            io,
            stage_pages,
            stage_bits,
            bin,
            phase_counter);
        return true;
    }
    const uint32_t inbox_base = static_cast<uint32_t>(DyadicScheduleOp::Bin0SwapInbox);
    if (u >= inbox_base && u < inbox_base + kBinSortCount && (u - inbox_base) == bin_id) {
        dyadic_exec_swap_inbox(bin_id, duty_state, io.page_active, io.page_inbox, io.page_retain);
        return true;
    }
    const uint32_t retain_base = static_cast<uint32_t>(DyadicScheduleOp::Bin0SwapRetain);
    if (u >= retain_base && u < retain_base + kBinSortCount && (u - retain_base) == bin_id) {
        dyadic_exec_swap_retain(bin_id, duty_state, io.page_active, io.page_inbox, io.page_retain);
        return true;
    }
    return false;
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_exec_bin_sort_single_thread(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    VALUE_T* stage,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* phase_counter) {
    if (!stage) {
        return;
    }
    if (bin_id == 0u) {
        dyadic_bin0_drain_paged<INDEX_T, VALUE_T, PremixPol>(io, stage, stage_bits);
    } else if (bin_id == 1u) {
        dyadic_cascade_phase_a_paged<INDEX_T, VALUE_T, PremixPol>(io, bin_id, stage, stage_bits);
    } else {
        bool do_phase_a = true;
        if (phase_counter) {
            do_phase_a = ((phase_counter[bin_id] & 1u) == 0u);
            phase_counter[bin_id] = (uint8_t)(phase_counter[bin_id] + 1u);
        }
        if (do_phase_a) {
            dyadic_cascade_phase_a_paged<INDEX_T, VALUE_T, PremixPol>(io, bin_id, stage, stage_bits);
        } else {
            dyadic_cascade_phase_b_paged<INDEX_T, VALUE_T, PremixPol>(io, bin_id, stage, stage_bits);
        }
    }
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol, size_t N>
inline void dyadic_run_single_thread_schedule(
    const std::array<DyadicScheduleOp, N>& schedule,
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    VALUE_T* stage,
    uint32_t stage_bits,
    std::atomic<uint8_t>* duty_state,
    uint8_t* phase_counter) {
    const uint32_t sort_base = static_cast<uint32_t>(DyadicScheduleOp::Bin0Sort);
    const uint32_t inbox_base = static_cast<uint32_t>(DyadicScheduleOp::Bin0SwapInbox);
    const uint32_t retain_base = static_cast<uint32_t>(DyadicScheduleOp::Bin0SwapRetain);
    for (size_t i = 0; i < N; ++i) {
        const uint32_t u = static_cast<uint32_t>(schedule[i]);
        if (u >= sort_base && u < sort_base + kBinSortCount) {
            const uint32_t bin = u - sort_base;
            if (bin >= io.total_bins) {
                continue;
            }
            dyadic_exec_bin_sort_single_thread<INDEX_T, VALUE_T, PremixPol>(
                io,
                stage,
                stage_bits,
                bin,
                phase_counter);
            continue;
        }
        if (u >= inbox_base && u < inbox_base + kBinSortCount) {
            const uint32_t bin = u - inbox_base;
            if (bin >= io.total_bins) {
                continue;
            }
            dyadic_exec_swap_inbox(bin, duty_state, io.page_active, io.page_inbox, io.page_retain);
            continue;
        }
        if (u >= retain_base && u < retain_base + kBinSortCount) {
            const uint32_t bin = u - retain_base;
            if (bin >= io.total_bins) {
                continue;
            }
            dyadic_exec_swap_retain(bin, duty_state, io.page_active, io.page_inbox, io.page_retain);
            continue;
        }
    }
}


struct DyadicThreadPolicy {
    uint32_t thread_count = 0;
    uint32_t bin_workers = 1;
    uint32_t write_heads = 1;
    uint32_t ratio = 2;
    bool force_single_threaded_bins = false;
    bool force_single_threaded_writers = false;
    bool force_single_thread = false;
    bool allow_parallel_bins = false;
};

const DyadicThreadPolicy NO_THREAD_POLICY = DyadicThreadPolicy{
    1u,
    std::numeric_limits<uint32_t>::max(),
    std::numeric_limits<uint32_t>::max(),
    0u,
    false,
    false,
    true,
    false
};

inline bool dyadic_is_single_threaded(const DyadicThreadPolicy& policy) {
    return policy.force_single_thread ||
           (policy.thread_count <= 1) ||
           (policy.bin_workers <= 1 && policy.write_heads <= 1) ||
           (policy.force_single_threaded_bins && policy.force_single_threaded_writers);
}


inline uint32_t dyadic_read_env_u32(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(v, &end, 10);
    if (end == v) return fallback;
    return static_cast<uint32_t>(parsed);
}

inline DyadicThreadPolicy dyadic_read_thread_policy(uint32_t requested_threads, uint32_t total_bins) {
    DyadicThreadPolicy policy{};
    policy.thread_count = requested_threads;
    if (policy.thread_count == 0) {
        policy.thread_count = std::thread::hardware_concurrency();
    }
    if (policy.thread_count == 0) {
        policy.thread_count = 1u;
        policy.force_single_thread = true;
    }else if (policy.thread_count == 1u) {
        policy.force_single_thread = true;
    }else if (policy.thread_count <= 2u){
        policy.force_single_threaded_bins = true;
        policy.force_single_threaded_writers = true;
        policy.ratio = 2u;
    }else if (policy.thread_count > 16u) {
        policy.thread_count = 16u;
    }
    if (!policy.force_single_thread){
        if (policy.ratio == 0u) policy.ratio = 2u;
        if ((float)policy.thread_count / (float)(policy.ratio + 1u) < 1.0f) {
            policy.ratio = policy.thread_count - 1u;
        }
        if (policy.bin_workers <= 0u || policy.write_heads <= 0u) {
            policy.bin_workers = policy.thread_count / (policy.ratio + 1u);
            policy.write_heads = policy.thread_count - policy.bin_workers;
        }
        policy.allow_parallel_bins = !policy.force_single_threaded_bins;
    } else { 
        return NO_THREAD_POLICY;
    }
    return policy;
}

constexpr uint8_t dyadic_duty_contents_mask = 0x03u;
constexpr uint8_t dyadic_duty_inbox_mask = 0x0Cu;
constexpr uint8_t dyadic_duty_retain_mask = 0x30u;
constexpr uint8_t dyadic_duty_flip_mask = 0xC0u;

inline uint8_t dyadic_duty_pack(uint8_t contents, uint8_t inbox, uint8_t retain, uint8_t flip) {
    return (uint8_t)((contents & 0x3u) | ((inbox & 0x3u) << 2u) | ((retain & 0x3u) << 4u) | ((flip & 0x3u) << 6u));
}

inline uint8_t dyadic_duty_contents(uint8_t state) { return (uint8_t)(state & dyadic_duty_contents_mask); }
inline uint8_t dyadic_duty_inbox(uint8_t state) { return (uint8_t)((state & dyadic_duty_inbox_mask) >> 2u); }
inline uint8_t dyadic_duty_retain(uint8_t state) { return (uint8_t)((state & dyadic_duty_retain_mask) >> 4u); }
inline uint8_t dyadic_duty_flip(uint8_t state) { return (uint8_t)((state & dyadic_duty_flip_mask) >> 6u); }

inline uint8_t dyadic_duty_inc_flip(uint8_t state) {
    const uint8_t flip = (uint8_t)((dyadic_duty_flip(state) + 1u) & 0x3u);
    return (uint8_t)((state & ~dyadic_duty_flip_mask) | (uint8_t)(flip << 6u));
}

inline uint8_t dyadic_duty_swap_contents_inbox(uint8_t state) {
    const uint8_t contents = dyadic_duty_contents(state);
    const uint8_t inbox = dyadic_duty_inbox(state);
    const uint8_t retain = dyadic_duty_retain(state);
    uint8_t next = dyadic_duty_pack(inbox, contents, retain, dyadic_duty_flip(state));
    return dyadic_duty_inc_flip(next);
}

inline uint8_t dyadic_duty_swap_contents_retain(uint8_t state) {
    const uint8_t contents = dyadic_duty_contents(state);
    const uint8_t inbox = dyadic_duty_inbox(state);
    const uint8_t retain = dyadic_duty_retain(state);
    uint8_t next = dyadic_duty_pack(retain, inbox, contents, dyadic_duty_flip(state));
    return dyadic_duty_inc_flip(next);
}

namespace policies {

struct Overwrite {
    template <typename T>
    static inline void apply(T& dest, const T& src) {
        dest = src;
    }

    template <typename T>
    static inline void apply_span(T* dest, const uint8_t* src_bytes, uint32_t elem_bytes, uint32_t value_stride) {
        for (uint32_t c = 0; c < value_stride; ++c) {
            T v{};
            DYADIC_RAW_WRITE(&v, src_bytes + elem_bytes * c, sizeof(T));
            dest[c] = v;
        }
    }
};

struct Add {
    template <typename T>
    static inline void apply(T& dest, const T& src) {
        dest = (T)(dest + src);
    }

    template <typename T>
    static inline void apply_span(T* dest, const uint8_t* src_bytes, uint32_t elem_bytes, uint32_t value_stride) {
        if constexpr (std::is_unsigned<T>::value) {
            if (elem_bytes == sizeof(T)) {
                const uintptr_t addr = reinterpret_cast<uintptr_t>(src_bytes);
                if ((addr % alignof(T)) == 0) {
                    auto* src = reinterpret_cast<const T*>(src_bytes);
                    (void)nodus::stream_add::add_wrap_inplace<T, false>(dest, src, value_stride);
                    return;
                }
            }
        }
        for (uint32_t c = 0; c < value_stride; ++c) {
            T v{};
            DYADIC_RAW_WRITE(&v, src_bytes + elem_bytes * c, sizeof(T));
            dest[c] = (T)(dest[c] + v);
        }
    }
};

struct Multiply {
    template <typename T>
    static inline void apply(T& dest, const T& src) {
        dest = (T)(dest * src);
    }

    template <typename T>
    static inline void apply_span(T* dest, const uint8_t* src_bytes, uint32_t elem_bytes, uint32_t value_stride) {
        for (uint32_t c = 0; c < value_stride; ++c) {
            T v{};
            DYADIC_RAW_WRITE(&v, src_bytes + elem_bytes * c, sizeof(T));
            dest[c] = (T)(dest[c] * v);
        }
    }
};

struct Max {
    template <typename T>
    static inline void apply(T& dest, const T& src) {
        dest = (src > dest) ? src : dest;
    }

    template <typename T>
    static inline void apply_span(T* dest, const uint8_t* src_bytes, uint32_t elem_bytes, uint32_t value_stride) {
        for (uint32_t c = 0; c < value_stride; ++c) {
            T v{};
            DYADIC_RAW_WRITE(&v, src_bytes + elem_bytes * c, sizeof(T));
            dest[c] = (v > dest[c]) ? v : dest[c];
        }
    }
};

struct FusedMulAdd {
    template <typename T>
    static inline void apply(T& dest, const T& src) {
        dest = (T)((dest * src) + src);
    }

    template <typename T>
    static inline void apply_span(T* dest, const uint8_t* src_bytes, uint32_t elem_bytes, uint32_t value_stride) {
        for (uint32_t c = 0; c < value_stride; ++c) {
            T v{};
            DYADIC_RAW_WRITE(&v, src_bytes + elem_bytes * c, sizeof(T));
            dest[c] = (T)((dest[c] * v) + v);
        }
    }
};

} // namespace policies

template <typename T>
inline constexpr uint32_t dyadic_uint_bits_v = (uint32_t)(sizeof(T) * CHAR_BIT);

template <typename IndexT, typename ValueT>
inline constexpr uint32_t dyadic_uint_max_bits_v =
    (dyadic_uint_bits_v<IndexT> > dyadic_uint_bits_v<ValueT>)
        ? dyadic_uint_bits_v<IndexT>
        : dyadic_uint_bits_v<ValueT>;

template <typename IndexT, typename ValueT>
inline constexpr TensorDType dyadic_uint_bytes_dtype() {
    if constexpr (dyadic_uint_max_bits_v<IndexT, ValueT> <= (uint32_t)(sizeof(uint8_t) * CHAR_BIT)) {
        return TensorDType::Bytes;
    } else if constexpr (dyadic_uint_max_bits_v<IndexT, ValueT> <= (uint32_t)(sizeof(uint16_t) * CHAR_BIT)) {
        return TensorDType::Bytes2;
    } else if constexpr (dyadic_uint_max_bits_v<IndexT, ValueT> <= (uint32_t)(sizeof(uint32_t) * CHAR_BIT)) {
        return TensorDType::Bytes4;
    } else {
        return TensorDType::Bytes8;
    }
}

template <typename T>
inline constexpr TensorDType dyadic_value_dtype() {
    if constexpr (std::is_same<T, float>::value) {
        return TensorDType::F32;
    } else if constexpr (std::is_same<T, double>::value) {
        return TensorDType::F64;
    } else if constexpr (std::is_same<T, int8_t>::value) {
        return TensorDType::I8;
    } else if constexpr (std::is_same<T, int16_t>::value) {
        return TensorDType::I16;
    } else if constexpr (std::is_same<T, int32_t>::value) {
        return TensorDType::I32;
    } else if constexpr (std::is_same<T, int64_t>::value) {
        return TensorDType::I64;
    } else if constexpr (std::is_same<T, uint8_t>::value) {
        return TensorDType::U8;
    } else if constexpr (std::is_same<T, uint16_t>::value) {
        return TensorDType::U16;
    } else if constexpr (std::is_same<T, uint32_t>::value) {
        return TensorDType::U32;
    } else if constexpr (std::is_same<T, uint64_t>::value) {
        return TensorDType::U64;
    } else if constexpr (std::is_same<T, bool>::value) {
        return TensorDType::Bool;
    } else {
        return TensorDType::Unknown;
    }
}

inline const char* dyadic_dtype_name(TensorDType dtype) {
    switch (dtype) {
        case TensorDType::F32: return "F32";
        case TensorDType::F64: return "F64";
        case TensorDType::I8: return "I8";
        case TensorDType::I16: return "I16";
        case TensorDType::I32: return "I32";
        case TensorDType::I64: return "I64";
        case TensorDType::U8: return "U8";
        case TensorDType::U16: return "U16";
        case TensorDType::U32: return "U32";
        case TensorDType::U64: return "U64";
        case TensorDType::Bool: return "Bool";
        case TensorDType::Bytes: return "Bytes";
        case TensorDType::Bytes2: return "Bytes2";
        case TensorDType::Bytes4: return "Bytes4";
        case TensorDType::Bytes8: return "Bytes8";
        case TensorDType::Ptr: return "Ptr";
        case TensorDType::Unknown:
        default:
            return "Unknown";
    }
}

inline const char* dyadic_layout_name(TensorLayout layout) {
    switch (layout) {
        case TensorLayout::Dense: return "Dense";
        case TensorLayout::Strided: return "Strided";
        case TensorLayout::Opaque: return "Opaque";
        default:
            return "Unknown";
    }
}

template <typename T>
inline void dyadic_log_scalar(const char* name, T value) {
    if constexpr (std::is_same<T, bool>::value) {
        std::fprintf(stderr, "[dyadic] %s=%d\n", name, value ? 1 : 0);
    } else if constexpr (std::is_floating_point<T>::value) {
        std::fprintf(stderr, "[dyadic] %s=%g\n", name, static_cast<double>(value));
    } else if constexpr (std::is_signed<T>::value) {
        std::fprintf(stderr, "[dyadic] %s=%lld\n", name, static_cast<long long>(value));
    } else {
        std::fprintf(stderr, "[dyadic] %s=%llu\n", name, static_cast<unsigned long long>(value));
    }
}

inline void dyadic_log_tensor(const char* name, const AbstractTensor& t) {
    const TensorDesc& desc = t.desc();
    std::fprintf(
        stderr,
        "[dyadic] %s: valid=%d handle=%llu backend=%p dtype=%s layout=%s rank=%u dims=[",
        name,
        t.valid() ? 1 : 0,
        static_cast<unsigned long long>(t.handle().id),
        static_cast<void*>(t.backend()),
        dyadic_dtype_name(desc.dtype),
        dyadic_layout_name(desc.layout),
        static_cast<unsigned int>(desc.shape.rank()));
    for (size_t i = 0; i < desc.shape.dims.size(); ++i) {
        std::fprintf(stderr, "%u", desc.shape.dims[i]);
        if (i + 1u < desc.shape.dims.size()) {
            std::fprintf(stderr, ",");
        }
    }
    std::fprintf(stderr, "]\n");
}

template <typename T>
inline uint32_t dyadic_lzcnt(T v);

template <typename T>
inline T dyadic_is_zero_mask(T v) {
    return (T)(~((v | (T)(0 - v)) >> (dyadic_uint_bits_v<T> - 1)));
}

#if WIN32
template <>
inline uint32_t dyadic_lzcnt<uint32_t>(uint32_t v) { return (uint32_t)__lzcnt(v); }
template <>
inline uint32_t dyadic_lzcnt<uint64_t>(uint64_t v) { return (uint32_t)__lzcnt64(v); }
template <>
inline uint32_t dyadic_lzcnt<uint16_t>(uint16_t v) {
    return v ? (uint32_t)(__lzcnt((uint32_t)v) - 16u) : 16u;
}
template <>
inline uint32_t dyadic_lzcnt<uint8_t>(uint8_t v) {
    return v ? (uint32_t)(__lzcnt((uint32_t)v) - 24u) : 8u;
}
#else
template <>
inline uint32_t dyadic_lzcnt<uint32_t>(uint32_t v) { return v ? (uint32_t)__builtin_clz(v) : 32u; }
template <>
inline uint32_t dyadic_lzcnt<uint64_t>(uint64_t v) { return v ? (uint32_t)__builtin_clzll(v) : 64u; }
template <>
inline uint32_t dyadic_lzcnt<uint16_t>(uint16_t v) {
    return v ? (uint32_t)(__builtin_clz((uint32_t)v) - 16u) : 16u;
}
template <>
inline uint32_t dyadic_lzcnt<uint8_t>(uint8_t v) {
    return v ? (uint32_t)(__builtin_clz((uint32_t)v) - 24u) : 8u;
}
#endif

template <typename T>
inline uint32_t dyadic_msb_pos_nonzero(T v) {
    return (uint32_t)(dyadic_uint_bits_v<T> - 1u - dyadic_lzcnt<T>(v));
}

template <typename T>
inline T dyadic_stage_sentinel() {
    if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
        return std::numeric_limits<T>::quiet_NaN();
    } else if constexpr (std::is_unsigned<T>::value) {
        return std::numeric_limits<T>::max();
    } else if constexpr (std::is_signed<T>::value) {
        return std::numeric_limits<T>::lowest();
    } else if constexpr (std::is_same<T, bool>::value) {
        return true;
    } else {
        return T{};
    }
}

template <typename INDEX_T>
inline INDEX_T floor_pow_2(INDEX_T x) {
    if (x == 0) {
        return 0;
    }
    INDEX_T p = 1;
    while (p <= x) {
        p <<= 1;
    }
    return (INDEX_T)(p >> 1);
}

template <typename INDEX_T>
inline uint32_t dyadic_bin_count(INDEX_T index_range, uint32_t stage_bits) {
    const INDEX_T hi_max = (INDEX_T)((index_range - 1u) >> stage_bits);
    return (uint32_t)(dyadic_msb_pos_nonzero<INDEX_T>(hi_max) + 1u);
}

template <typename INDEX_T, typename VALUE_T>
inline bool dyadic_bins_empty(const DyadicBinIO3<INDEX_T, VALUE_T>& io) {
    if (!io.page_count || io.total_bins == 0u) return true;
    const uint64_t total_pages = (uint64_t)io.total_bins * 3ull;
    for (uint64_t i = 0; i < total_pages; ++i) {
        if (io.page_count[i] != 0u) return false;
    }
    return true;
}

inline void dyadic_sync_pages_from_state(uint32_t bin, uint8_t state,
                                         uint8_t* page_active,
                                         uint8_t* page_inbox,
                                         uint8_t* page_retain) {
    page_active[bin] = dyadic_duty_contents(state);
    page_inbox[bin] = dyadic_duty_inbox(state);
    page_retain[bin] = dyadic_duty_retain(state);
}

template <typename INDEX_T, typename VALUE_T>
inline void dyadic_push_page_bytes(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    uint32_t bin,
    uint8_t page,
    INDEX_T idx,
    const uint8_t* val_bytes) {
    io.write_slot_to_page_bytes(bin, page, idx, val_bytes);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_classify_to_stage_or_bin_paged_stage_pages(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    INDEX_T idx,
    const VALUE_T* vals,
    uint32_t src_bin);

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_scan_page_phase_a_custom(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    uint32_t src_bin,
    uint8_t src_page,
    uint8_t keep_page,
    VALUE_T* stage,
    uint32_t stage_bits) {
    const uint32_t n = io.cnt(src_bin, src_page);
    if (!n) return;
    const INDEX_T lo_mask = ((INDEX_T)1u << stage_bits) - 1u;
    const uint32_t msb0_pos = src_bin;
    const INDEX_T msb0_mask = (INDEX_T)1u << msb0_pos;
    const INDEX_T msb1_mask = (INDEX_T)1u << (msb0_pos - 1u);
    uint8_t* base = io.page_base(src_bin, src_page);
    uint32_t slot = 0;
    while (slot ^ n) {
        uint8_t* p = base + (uint64_t)slot * (uint64_t)io.slot_bytes;
        INDEX_T idx;
        DYADIC_RAW_WRITE(&idx, p, sizeof(INDEX_T));
        const uint8_t* val_bytes = p + io.elem_bytes;
        const INDEX_T lo = (INDEX_T)(idx & lo_mask);
        const INDEX_T hi = (INDEX_T)(idx >> stage_bits);
        const bool gate_move = ((hi & msb1_mask) == 0);
        if (!gate_move) {
            dyadic_push_page_bytes<INDEX_T, VALUE_T>(io, src_bin, keep_page, idx, val_bytes);
            slot += 1u;
            continue;
        }
        const INDEX_T hi_move = (INDEX_T)(hi ^ msb0_mask);
        const INDEX_T idx_move = (INDEX_T)((hi_move << stage_bits) | lo);
        dyadic_classify_to_stage_or_bin_paged_bytes<INDEX_T, VALUE_T, PremixPol>(io, stage, stage_bits, idx_move, val_bytes);
        slot += 1u;
    }
    io.cnt(src_bin, src_page) = 0;
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_scan_page_phase_b_custom(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    uint32_t src_bin,
    uint8_t src_page,
    VALUE_T* stage,
    uint32_t stage_bits) {
    const uint32_t n = io.cnt(src_bin, src_page);
    if (!n) return;
    const INDEX_T lo_mask = ((INDEX_T)1u << stage_bits) - 1u;
    const uint32_t msb0_pos = src_bin;
    const INDEX_T msb0_mask = (INDEX_T)1u << msb0_pos;
    const INDEX_T msb1_mask = (INDEX_T)1u << (msb0_pos - 1u);
    uint8_t* base = io.page_base(src_bin, src_page);
    uint32_t slot = 0;
    while (slot ^ n) {
        uint8_t* p = base + (uint64_t)slot * (uint64_t)io.slot_bytes;
        INDEX_T idx;
        DYADIC_RAW_WRITE(&idx, p, sizeof(INDEX_T));
        const uint8_t* val_bytes = p + io.elem_bytes;
        const INDEX_T lo = (INDEX_T)(idx & lo_mask);
        INDEX_T hi = (INDEX_T)(idx >> stage_bits);
        hi = (INDEX_T)(hi ^ (msb0_mask | msb1_mask));
        const INDEX_T idx2 = (INDEX_T)((hi << stage_bits) | lo);
        dyadic_classify_to_stage_or_bin_paged_bytes<INDEX_T, VALUE_T, PremixPol>(io, stage, stage_bits, idx2, val_bytes);
        slot += 1u;
    }
    io.cnt(src_bin, src_page) = 0;
}


template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_scan_page_bin0_custom(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    uint32_t src_bin,
    uint8_t src_page,
    VALUE_T* stage,
    uint32_t stage_bits) {
    const uint32_t n = io.cnt(src_bin, src_page);
    if (!n) return;
    const INDEX_T lo_mask = ((INDEX_T)1u << stage_bits) - 1u;
    uint8_t* base = io.page_base(src_bin, src_page);
    uint32_t slot = 0;
    while (slot ^ n) {
        uint8_t* p = base + (uint64_t)slot * (uint64_t)io.slot_bytes;
        INDEX_T idx;
        DYADIC_RAW_WRITE(&idx, p, sizeof(INDEX_T));
        const uint8_t* val_bytes = p + io.elem_bytes;
        const INDEX_T lo = (INDEX_T)(idx & lo_mask);
        VALUE_T* dst = stage + (uint64_t)lo * (uint64_t)io.value_stride;
        PremixPol::apply_span(dst, val_bytes, io.elem_bytes, io.value_stride);
        slot += 1u;
    }
    io.cnt(src_bin, src_page) = 0;
}



template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_classify_to_stage_or_bin_paged(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    VALUE_T* stage,
    uint32_t stage_bits,
    INDEX_T idx,
    const VALUE_T* vals) {
    const INDEX_T S = (INDEX_T)1u << stage_bits;
    const INDEX_T lo = (INDEX_T)(idx & (S - 1u));
    const INDEX_T hi = (INDEX_T)(idx >> stage_bits);
    if (hi == 0) {
        VALUE_T* dst = stage + (uint64_t)lo * (uint64_t)io.value_stride;
        PremixPol::apply_span(dst, reinterpret_cast<const uint8_t*>(vals), io.elem_bytes, io.value_stride);
        return;
    }
    uint32_t b = dyadic_msb_pos_nonzero<INDEX_T>(hi);
    io.push_inbox(b, idx, vals);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_classify_to_stage_or_bin_paged_stage_pages(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    INDEX_T idx,
    const VALUE_T* vals,
    uint32_t src_bin) {
    const INDEX_T S = (INDEX_T)1u << stage_bits;
    const INDEX_T lo = (INDEX_T)(idx & (S - 1u));
    const INDEX_T hi = (INDEX_T)(idx >> stage_bits);
    if (hi == 0) {
        VALUE_T* stage = stage_pages.template stage_ptr<VALUE_T>(src_bin);
        if (stage) {
            VALUE_T* dst = stage + (uint64_t)lo * (uint64_t)io.value_stride;
            PremixPol::apply_span(dst, reinterpret_cast<const uint8_t*>(vals), io.elem_bytes, io.value_stride);
            stage_pages.mark_dirty(src_bin);
        }
        return;
    }
    uint32_t b = dyadic_msb_pos_nonzero<INDEX_T>(hi);
    io.push_inbox(b, idx, vals);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_classify_to_stage_or_bin_paged_bytes(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    VALUE_T* stage,
    uint32_t stage_bits,
    INDEX_T idx,
    const uint8_t* val_bytes) {
    const INDEX_T S = (INDEX_T)1u << stage_bits;
    const INDEX_T lo = (INDEX_T)(idx & (S - 1u));
    const INDEX_T hi = (INDEX_T)(idx >> stage_bits);
    if (hi == 0) {
        VALUE_T* dst = stage + (uint64_t)lo * (uint64_t)io.value_stride;
        PremixPol::apply_span(dst, val_bytes, io.elem_bytes, io.value_stride);
        return;
    }
    uint32_t b = dyadic_msb_pos_nonzero<INDEX_T>(hi);
    io.push_inbox_bytes(b, idx, val_bytes);
}


template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_cascade_phase_a_paged(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    uint32_t src_bin,
    VALUE_T* stage,
    uint32_t stage_bits) {
    io.prepare_for_scan(src_bin);
    const uint8_t A = io.active_page(src_bin);
    const uint32_t n = io.cnt(src_bin, A);
    const INDEX_T lo_mask = ((INDEX_T)1u << stage_bits) - 1u;
    const uint32_t msb0_pos = src_bin;
    const INDEX_T msb0_mask = (INDEX_T)1u << msb0_pos;
    const INDEX_T msb1_mask = (INDEX_T)1u << (msb0_pos - 1u);
    uint8_t* base = io.page_base(src_bin, A);
    io.cnt(src_bin, io.page_retain[src_bin]) = 0;
    uint32_t slot = 0;
    while (slot ^ n) {
        uint8_t* p = base + (uint64_t)slot * (uint64_t)io.slot_bytes;
        INDEX_T idx;
        DYADIC_RAW_WRITE(&idx, p, sizeof(INDEX_T));
        const uint8_t* val_bytes = p + io.elem_bytes;
        const INDEX_T lo = (INDEX_T)(idx & lo_mask);
        const INDEX_T hi = (INDEX_T)(idx >> stage_bits);
        const bool gate_move = ((hi & msb1_mask) == 0);
        if (!gate_move) {
            io.push_retain_bytes(src_bin, idx, val_bytes);
            slot += 1u;
            continue;
        }
        const INDEX_T hi_move = (INDEX_T)(hi ^ msb0_mask);
        const INDEX_T idx_move = (INDEX_T)((hi_move << stage_bits) | lo);
        dyadic_classify_to_stage_or_bin_paged_bytes<INDEX_T, VALUE_T, PremixPol>(io, stage, stage_bits, idx_move, val_bytes);
        slot += 1u;
    }
    io.finish_scan(src_bin);
}


template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_cascade_phase_b_paged(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    uint32_t src_bin,
    VALUE_T* stage,
    uint32_t stage_bits) {
    io.prepare_for_scan(src_bin);
    const uint8_t A = io.active_page(src_bin);
    const uint32_t n = io.cnt(src_bin, A);
    const INDEX_T lo_mask = ((INDEX_T)1u << stage_bits) - 1u;
    const uint32_t msb0_pos = src_bin;
    const INDEX_T msb0_mask = (INDEX_T)1u << msb0_pos;
    const INDEX_T msb1_mask = (INDEX_T)1u << (msb0_pos - 1u);
    uint8_t* base = io.page_base(src_bin, A);
    io.cnt(src_bin, io.page_retain[src_bin]) = 0;
    uint32_t slot = 0;
    while (slot ^ n) {
        uint8_t* p = base + (uint64_t)slot * (uint64_t)io.slot_bytes;
        INDEX_T idx;
        DYADIC_RAW_WRITE(&idx, p, sizeof(INDEX_T));
        const uint8_t* val_bytes = p + io.elem_bytes;
        const INDEX_T lo = (INDEX_T)(idx & lo_mask);
        INDEX_T hi = (INDEX_T)(idx >> stage_bits);
        hi = (INDEX_T)(hi ^ (msb0_mask | msb1_mask));
        const INDEX_T idx2 = (INDEX_T)((hi << stage_bits) | lo);
        dyadic_classify_to_stage_or_bin_paged_bytes<INDEX_T, VALUE_T, PremixPol>(io, stage, stage_bits, idx2, val_bytes);
        slot += 1u;
    }
    io.finish_scan(src_bin);
}


template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_bin0_drain_paged(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    VALUE_T* stage,
    uint32_t stage_bits) {
    const uint32_t bin = 0;
    io.prepare_for_scan(bin);
    const uint8_t A = io.active_page(bin);
    const uint32_t n = io.cnt(bin, A);
    const INDEX_T lo_mask = ((INDEX_T)1u << stage_bits) - 1u;
    uint8_t* base = io.page_base(bin, A);
    uint32_t slot = 0;
    while (slot ^ n) {
        uint8_t* p = base + (uint64_t)slot * (uint64_t)io.slot_bytes;
        INDEX_T idx;
        DYADIC_RAW_WRITE(&idx, p, sizeof(INDEX_T));
        const uint8_t* val_bytes = p + io.elem_bytes;
        const INDEX_T lo = (INDEX_T)(idx & lo_mask);
        VALUE_T* dst = stage + (uint64_t)lo * (uint64_t)io.value_stride;
        PremixPol::apply_span(dst, val_bytes, io.elem_bytes, io.value_stride);
        slot += 1u;
    }
    io.finish_scan(bin);
}

inline DyadicScheduleOp dyadic_bin_sort_op(uint32_t bin) {
    const uint32_t base = static_cast<uint32_t>(DyadicScheduleOp::Bin0Sort);
    return static_cast<DyadicScheduleOp>(base + bin);
}

inline DyadicScheduleOp dyadic_bin_swap_inbox_op(uint32_t bin) {
    const uint32_t base = static_cast<uint32_t>(DyadicScheduleOp::Bin0SwapInbox);
    return static_cast<DyadicScheduleOp>(base + bin);
}

inline DyadicScheduleOp dyadic_bin_swap_retain_op(uint32_t bin) {
    const uint32_t base = static_cast<uint32_t>(DyadicScheduleOp::Bin0SwapRetain);
    return static_cast<DyadicScheduleOp>(base + bin);
}

inline void dyadic_exec_swap_inbox(
    uint32_t bin_id,
    std::atomic<uint8_t>* duty_state,
    uint8_t* page_active,
    uint8_t* page_inbox,
    uint8_t* page_retain) {
    const uint8_t state = duty_state[bin_id].load(std::memory_order_acquire);
    const uint8_t next = dyadic_duty_swap_contents_inbox(state);
    duty_state[bin_id].store(next, std::memory_order_release);
    dyadic_sync_pages_from_state(bin_id, next, page_active, page_inbox, page_retain);
}

inline void dyadic_exec_swap_retain(
    uint32_t bin_id,
    std::atomic<uint8_t>* duty_state,
    uint8_t* page_active,
    uint8_t* page_inbox,
    uint8_t* page_retain) {
    const uint8_t state = duty_state[bin_id].load(std::memory_order_acquire);
    const uint8_t next = dyadic_duty_swap_contents_retain(state);
    duty_state[bin_id].store(next, std::memory_order_release);
    dyadic_sync_pages_from_state(bin_id, next, page_active, page_inbox, page_retain);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_exec_bin_sort(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* phase_counter) {
    VALUE_T* stage = stage_pages.template stage_ptr<VALUE_T>(bin_id);
    if (!stage) {
        return;
    }
    if (bin_id == 0u) {
        dyadic_bin0_drain_paged<INDEX_T, VALUE_T, PremixPol>(io, stage, stage_bits);
    } else if (bin_id == 1u) {
        dyadic_cascade_phase_a_paged<INDEX_T, VALUE_T, PremixPol>(io, bin_id, stage, stage_bits);
    } else {
        bool do_phase_a = true;
        if (phase_counter) {
            do_phase_a = ((phase_counter[bin_id] & 1u) == 0u);
            phase_counter[bin_id] = (uint8_t)(phase_counter[bin_id] + 1u);
        }
        if (do_phase_a) {
            dyadic_cascade_phase_a_paged<INDEX_T, VALUE_T, PremixPol>(io, bin_id, stage, stage_bits);
        } else {
            dyadic_cascade_phase_b_paged<INDEX_T, VALUE_T, PremixPol>(io, bin_id, stage, stage_bits);
        }
    }
    stage_pages.mark_dirty(bin_id);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol, size_t N>
inline void dyadic_bin_worker_run_schedule(
    const DyadicScheduleOp (&schedule)[N],
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    std::atomic<uint8_t>* duty_state,
    uint8_t* last_upstream_flip,
    uint8_t* last_downstream_flip) {
    (void)stage_pages;
    (void)last_upstream_flip;
    for (size_t i = 0; i < N; ++i) {
        const DyadicScheduleOp op = schedule[i];
        if (op == dyadic_bin_sort_op(bin_id)) {
            dyadic_exec_bin_sort<INDEX_T, VALUE_T, PremixPol>(
                io,
                stage_pages,
                stage_bits,
                bin_id,
                last_downstream_flip);
            continue;
        }
        if (op == dyadic_bin_swap_inbox_op(bin_id)) {
            dyadic_exec_swap_inbox(bin_id, duty_state, io.page_active, io.page_inbox, io.page_retain);
            continue;
        }
        if (op == dyadic_bin_swap_retain_op(bin_id)) {
            dyadic_exec_swap_retain(bin_id, duty_state, io.page_active, io.page_inbox, io.page_retain);
            continue;
        }
        (void)op;
    }
}


template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_bin_worker_step_bin0(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    std::atomic<uint8_t>* duty_state,
    uint8_t* last_upstream_flip,
    uint8_t* last_downstream_flip) {
    static constexpr DyadicScheduleOp schedule[] = {
        DyadicScheduleOp::Bin0SwapInbox,
        DyadicScheduleOp::Bin0Sort,
    };
    dyadic_bin_worker_run_schedule<INDEX_T, VALUE_T, PremixPol>(
        schedule,
        io,
        stage_pages,
        stage_bits,
        bin_id,
        duty_state,
        last_upstream_flip,
        last_downstream_flip);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_bin_worker_step_bin1(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    std::atomic<uint8_t>* duty_state,
    uint8_t* last_upstream_flip,
    uint8_t* last_downstream_flip) {
    static constexpr DyadicScheduleOp schedule[] = {
        DyadicScheduleOp::Bin1Sort,
        DyadicScheduleOp::Bin1SwapRetain,
    };
    dyadic_bin_worker_run_schedule<INDEX_T, VALUE_T, PremixPol>(
        schedule,
        io,
        stage_pages,
        stage_bits,
        bin_id,
        duty_state,
        last_upstream_flip,
        last_downstream_flip);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_bin_worker_step_binn(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    std::atomic<uint8_t>* duty_state,
    uint8_t* last_upstream_flip,
    uint8_t* last_downstream_flip) {
    const DyadicScheduleOp schedule[] = {
        dyadic_bin_sort_op(bin_id),
        dyadic_bin_swap_retain_op(bin_id),
    };
    dyadic_bin_worker_run_schedule<INDEX_T, VALUE_T, PremixPol>(
        schedule,
        io,
        stage_pages,
        stage_bits,
        bin_id,
        duty_state,
        last_upstream_flip,
        last_downstream_flip);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_bin_worker_step_last(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    std::atomic<uint8_t>* duty_state,
    uint8_t* last_upstream_flip,
    uint8_t* last_downstream_flip) {
    const DyadicScheduleOp schedule[] = {
        dyadic_bin_sort_op(bin_id),
        dyadic_bin_swap_retain_op(bin_id),
    };
    dyadic_bin_worker_run_schedule<INDEX_T, VALUE_T, PremixPol>(
        schedule,
        io,
        stage_pages,
        stage_bits,
        bin_id,
        duty_state,
        last_upstream_flip,
        last_downstream_flip);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_bin_worker_step(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    std::atomic<uint8_t>* duty_state,
    uint8_t* last_upstream_flip,
    uint8_t* last_downstream_flip) {
    if (bin_id == 0u) {
        dyadic_bin_worker_step_bin0<INDEX_T, VALUE_T, PremixPol>(
            io, stage_pages, stage_bits, bin_id, duty_state, last_upstream_flip, last_downstream_flip);
        return;
    }
    if (bin_id == 1u) {
        dyadic_bin_worker_step_bin1<INDEX_T, VALUE_T, PremixPol>(
            io, stage_pages, stage_bits, bin_id, duty_state, last_upstream_flip, last_downstream_flip);
        return;
    }
    if (bin_id + 1u == io.total_bins) {
        dyadic_bin_worker_step_last<INDEX_T, VALUE_T, PremixPol>(
            io, stage_pages, stage_bits, bin_id, duty_state, last_upstream_flip, last_downstream_flip);
        return;
    }
    dyadic_bin_worker_step_binn<INDEX_T, VALUE_T, PremixPol>(
        io, stage_pages, stage_bits, bin_id, duty_state, last_upstream_flip, last_downstream_flip);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
struct DyadicBinWorkerJobCtx {
    DyadicBinIO3<INDEX_T, VALUE_T>* io = nullptr;
    DyadicStagePagePool* stage_pages = nullptr;
    uint32_t stage_bits = 0;
    uint32_t bin_id = 0;
    std::atomic<uint8_t>* duty_state = nullptr;
    uint8_t* last_upstream_flip = nullptr;
    uint8_t* last_downstream_flip = nullptr;
    std::atomic<bool>* stop = nullptr;
};

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_bin_worker_job_fn(const nodus::ThreadPool::Job& job, uint32_t /*worker_id*/) {
    (void)job;
}

template <typename INDEX_T, typename VALUE_T, typename OutmixPol>
struct DyadicStageWriterJobCtx {
    DyadicBinIO3<INDEX_T, VALUE_T>* io = nullptr;
    DyadicStagePagePool* stage_pages = nullptr;
    VALUE_T* output = nullptr;
    uint32_t* output_offset = nullptr;
    uint32_t stage_bits = 0;
    std::atomic<bool>* stop = nullptr;
};

template <typename INDEX_T, typename VALUE_T, typename OutmixPol>
inline void dyadic_stage_writer_job_fn(const nodus::ThreadPool::Job& job, uint32_t /*worker_id*/) {
    (void)job;
}

template <typename VALUE_T, typename OutmixPol>
inline void dyadic_emit_stage_linear(VALUE_T* output, uint32_t* output_offset, const VALUE_T* stage,
                                     uint32_t stage_bits, uint32_t value_stride) {
    const uint32_t stage_len = (uint32_t)(1u << stage_bits);
    const uint64_t total = (uint64_t)stage_len * (uint64_t)value_stride;
    VALUE_T* out = output + *output_offset;
    if constexpr (std::is_same_v<OutmixPol, policies::Overwrite>) {
        DYADIC_RAW_WRITE(out, stage, (uint32_t)(total * (uint64_t)sizeof(VALUE_T)));
    } else {
        OutmixPol::apply_span(out, reinterpret_cast<const uint8_t*>(stage),
                              (uint32_t)sizeof(VALUE_T), (uint32_t)total);
    }
    *output_offset = (uint32_t)(*output_offset + (uint32_t)total);
}

template <typename VALUE_T, typename OutmixPol>
inline bool dyadic_emit_stage_linear(AbstractTensor& output,
                                     uint32_t* output_offset,
                                     const VALUE_T* stage,
                                     uint32_t stage_bits,
                                     uint32_t value_stride) {
    if (!output.valid()) return false;
    auto* mem = dynamic_cast<InMemoryBackend*>(output.backend());
    if (!mem) return false;
    void* out_ptr = nullptr;
    size_t out_bytes = 0;
    if (!mem->map(output.handle(), &out_ptr, &out_bytes)) return false;
    dyadic_emit_stage_linear<VALUE_T, OutmixPol>(static_cast<VALUE_T*>(out_ptr), output_offset, stage, stage_bits, value_stride);
    mem->unmap(output.handle());
    return true;
}


#define DYADIC_BINNING_UINT_DEFINE(INDEX_T, VALUE_T) \
    template <typename PremixPol = policies::Add, typename OutmixPol = policies::Overwrite> \
    inline void dyadic_mt_bitmask_algo_parallel( \
        DyadicBinIO3<INDEX_T, VALUE_T>& io, \
        DyadicStagePagePool& stage_pages, \
        uint32_t stage_bits, \
        uint32_t phase_a_rounds, \
        uint32_t phase_b_rounds, \
        uint32_t bin0_passes, \
        const DyadicThreadPolicy& policy) { \
        (void)policy; \
        (void)phase_a_rounds; \
        (void)phase_b_rounds; \
        (void)bin0_passes; \
        if (!io.total_bins) return; \
        std::vector<std::atomic<uint8_t>> duty_state(io.total_bins); \
        std::vector<uint8_t> last_up(io.total_bins, 0xFFu); \
        std::vector<uint8_t> last_down(io.total_bins, 0xFFu); \
        for (uint32_t bin = 0; bin < io.total_bins; ++bin) { \
            uint8_t state = dyadic_duty_pack(io.page_active[bin], io.page_inbox[bin], io.page_retain[bin], 0u); \
            duty_state[bin].store(state, std::memory_order_release); \
            dyadic_sync_pages_from_state(bin, state, io.page_active, io.page_inbox, io.page_retain); \
        } \
        std::atomic<bool> stop{false}; \
        nodus::ThreadPool::Options opt{}; \
        opt.thread_count = policy.bin_workers; \
        opt.start_immediately = true; \
        nodus::ThreadPool pool(opt); \
        std::vector<DyadicBinWorkerJobCtx<INDEX_T, VALUE_T, PremixPol>> ctxs(io.total_bins); \
        std::vector<nodus::ThreadPool::Job> jobs(io.total_bins); \
        for (uint32_t bin = 0; bin < io.total_bins; ++bin) { \
            ctxs[bin].io = &io; \
            ctxs[bin].stage_pages = &stage_pages; \
            ctxs[bin].stage_bits = stage_bits; \
            ctxs[bin].bin_id = bin; \
            ctxs[bin].duty_state = duty_state.data(); \
            ctxs[bin].last_upstream_flip = last_up.data(); \
            ctxs[bin].last_downstream_flip = last_down.data(); \
            ctxs[bin].stop = &stop; \
            jobs[bin].fn = &dyadic_bin_worker_job_fn<INDEX_T, VALUE_T, PremixPol>; \
            jobs[bin].user = &ctxs[bin]; \
        } \
        auto batch = pool.submit_batch(jobs.data(), (uint32_t)jobs.size()); \
        if (batch) batch->wait(); \
    }\
    template <typename PremixPol = policies::Add, typename OutmixPol = policies::Overwrite> \
    inline void dyadic_mt_bitmask_algo(INDEX_T index_range, uint32_t index_count, INDEX_T* indices, VALUE_T* values, \
                                       uint32_t value_stride, \
                                       uint32_t stage_bits, uint32_t thread_count, \
                                       uint32_t phase_a_rounds, uint32_t phase_b_rounds, uint32_t bin0_passes, \
                                       VALUE_T* output, uint32_t* output_offset, \
                                       AbstractTensorPool& pool, \
                                       AbstractTensor& bins, AbstractTensor& staging, AbstractTensor& counters, \
                                       AbstractTensor& page_active, AbstractTensor& page_retain, \
                                       AbstractTensor& page_inbox, \
                                       uint32_t stage_pages_max = 0) { \
        InMemoryBackend* mem = static_cast<InMemoryBackend*>(bins.backend()); \
        void* bin_data_void = nullptr; \
        size_t bin_bytes = 0; \
        mem->map(bins.handle(), &bin_data_void, &bin_bytes); \
        (void)mem->ensure_zeroed(staging.handle(), staging.desc()); \
        (void)mem->ensure_zeroed(counters.handle(), counters.desc()); \
        void* counter_data_void = nullptr; \
        size_t counter_bytes = 0; \
        mem->map(counters.handle(), &counter_data_void, &counter_bytes); \
        void* page_active_data_void = nullptr; \
        size_t page_active_bytes = 0; \
        mem->map(page_active.handle(), &page_active_data_void, &page_active_bytes); \
        void* page_retain_data_void = nullptr; \
        size_t page_retain_bytes = 0; \
        mem->map(page_retain.handle(), &page_retain_data_void, &page_retain_bytes); \
        void* page_inbox_data_void = nullptr; \
        size_t page_inbox_bytes = 0; \
        mem->map(page_inbox.handle(), &page_inbox_data_void, &page_inbox_bytes); \
        (void)index_range; \
        (void)thread_count; \
        (void)bin_bytes; \
        (void)counter_bytes; \
        DyadicBinIO3<INDEX_T, VALUE_T> io{}; \
        io.bin_data = reinterpret_cast<uint8_t*>(bin_data_void); \
        io.page_count = static_cast<uint32_t*>(counter_data_void); \
        io.page_active = static_cast<uint8_t*>(page_active_data_void); \
        io.page_retain = static_cast<uint8_t*>(page_retain_data_void); \
        io.page_inbox = static_cast<uint8_t*>(page_inbox_data_void); \
        io.index_count = index_count; \
        io.elem_bytes = (uint32_t)(dyadic_uint_max_bits_v<INDEX_T, VALUE_T> >> 3); \
        io.value_stride = value_stride; \
        io.value_bytes = io.elem_bytes * value_stride; \
        io.slot_bytes = io.elem_bytes * (1u + value_stride); \
        io.total_bins = dyadic_bin_count(index_range, stage_bits); \
        const uint32_t stage_entries = (uint32_t)((1u << stage_bits) * value_stride); \
        const uint64_t stage_page_bytes = (uint64_t)stage_entries * (uint64_t)sizeof(VALUE_T); \
        uint32_t page_count = stage_pages_max; \
        const uint32_t max_possible_pages = (io.total_bins == 0u) \
            ? 1u \
            : (static_cast<uint32_t>(1u) << io.total_bins) + 1u; \
        if (page_count == 0u) { \
            page_count = max_possible_pages; \
        } \
        page_count = std::max<uint32_t>(1u, std::min<uint32_t>(page_count, max_possible_pages)); \
        TensorDesc stage_page_desc = staging.desc(); \
        stage_page_desc.shape.dims = { stage_entries }; \
        DyadicStagePagePool stage_pages; \
        stage_pages.init(&pool, bins.backend(), mem, stage_page_desc, io.total_bins, stage_entries, \
                         stage_page_bytes, page_count, staging.handle(), &staging.desc(), false, false); \
        const DyadicThreadPolicy policy = dyadic_read_thread_policy(thread_count, io.total_bins); \
        uint32_t b = 0; \
        while (b ^ io.total_bins) { \
            io.page_active[b] = 0; \
            io.page_retain[b] = 1; \
            io.page_inbox[b] = 2; \
            b += 1u; \
        } \
        void* stage_data_void = nullptr; \
        size_t stage_bytes = 0; \
        mem->map(staging.handle(), &stage_data_void, &stage_bytes); \
        VALUE_T* stage = reinterpret_cast<VALUE_T*>(stage_data_void); \
        (void)stage_bytes; \
        uint32_t i = 0; \
        while (i ^ index_count) { \
            const VALUE_T* vals = values + (uint64_t)i * (uint64_t)value_stride; \
            if (policy.allow_parallel_bins) { \
                dyadic_classify_to_stage_or_bin_paged_stage_pages<INDEX_T, VALUE_T, PremixPol>( \
                    io, stage_pages, stage_bits, indices[i], vals, 0u); \
            } else { \
                dyadic_classify_to_stage_or_bin_paged<INDEX_T, VALUE_T, PremixPol>(io, stage, stage_bits, indices[i], vals); \
            } \
            i += 1u; \
        } \
        if (policy.allow_parallel_bins) { \
            dyadic_mt_bitmask_algo_parallel<PremixPol, OutmixPol>( \
                io, \
                stage_pages, \
                stage_bits, \
                phase_a_rounds, \
                phase_b_rounds, \
                bin0_passes, \
                policy); \
            stage_pages.emit_all<OutmixPol>(output, output_offset); \
        } else { \
            std::vector<std::atomic<uint8_t>> duty_state(io.total_bins); \
            std::vector<uint8_t> phase_counter(io.total_bins, 0u); \
            for (uint32_t bin = 0; bin < io.total_bins; ++bin) { \
                const uint8_t state = dyadic_duty_pack(io.page_active[bin], io.page_inbox[bin], io.page_retain[bin], 0u); \
                duty_state[bin].store(state, std::memory_order_release); \
                dyadic_sync_pages_from_state(bin, state, io.page_active, io.page_inbox, io.page_retain); \
            } \
            dyadic_run_single_thread_schedule<INDEX_T, VALUE_T, PremixPol>( \
                single_threaded_dyadic_schedule, \
                io, \
                stage, \
                stage_bits, \
                duty_state.data(), \
                phase_counter.data()); \
            dyadic_emit_stage_linear<VALUE_T, OutmixPol>(output, output_offset, stage, stage_bits, value_stride); \
        } \
        mem->unmap(staging.handle()); \
        mem->unmap(bins.handle()); \
        mem->unmap(counters.handle()); \
        mem->unmap(page_active.handle()); \
        mem->unmap(page_retain.handle()); \
        mem->unmap(page_inbox.handle()); \
    } \
    template <typename PremixPol = policies::Add, typename OutmixPol = policies::Overwrite> \
    inline bool dyadic_mt_bitmask_algo(INDEX_T index_range, uint32_t index_count, INDEX_T* indices, VALUE_T* values, \
                                       uint32_t value_stride, \
                                       uint32_t stage_bits, uint32_t thread_count, \
                                       uint32_t phase_a_rounds, uint32_t phase_b_rounds, uint32_t bin0_passes, \
                                       AbstractTensor& output, uint32_t* output_offset, \
                                       AbstractTensorPool& pool, \
                                       AbstractTensor& bins, AbstractTensor& staging, AbstractTensor& counters, \
                                       AbstractTensor& page_active, AbstractTensor& page_retain, \
                                       AbstractTensor& page_inbox, \
                                       uint32_t stage_pages_max = 0) { \
        if (!output.valid()) return false; \
        auto* mem = dynamic_cast<InMemoryBackend*>(output.backend()); \
        if (!mem) return false; \
        void* out_ptr = nullptr; \
        size_t out_bytes = 0; \
        if (!mem->map(output.handle(), &out_ptr, &out_bytes)) return false; \
        dyadic_mt_bitmask_algo<PremixPol, OutmixPol>( \
            index_range, \
            index_count, \
            indices, \
            values, \
            value_stride, \
            stage_bits, \
            thread_count, \
            phase_a_rounds, \
            phase_b_rounds, \
            bin0_passes, \
            static_cast<VALUE_T*>(out_ptr), \
            output_offset, \
            pool, \
            bins, \
            staging, \
            counters, \
            page_active, \
            page_retain, \
            page_inbox, \
            stage_pages_max); \
        mem->unmap(output.handle()); \
        return true; \
    } \
    template <typename PremixPol = policies::Add, typename OutmixPol = policies::Overwrite> \
    inline bool dyadic_binning_tensor(INDEX_T index_range, uint32_t index_count, INDEX_T* indices, VALUE_T* values, \
                                      uint32_t value_stride, \
                                      uint32_t stage_bits, uint32_t thread_count, \
                                      uint32_t phase_a_rounds, uint32_t phase_b_rounds, uint32_t bin0_passes, \
                                      AbstractTensor& output, AbstractTensorPool& pool, \
                                      uint32_t stage_pages_max = 0) { \
        if (!output.valid()) return false; \
        TensorBackend* backend = output.backend(); \
        if (!backend) return false; \
        const uint32_t total_bins = dyadic_bin_count(index_range, stage_bits); \
        TensorDesc desc_dyadic_bins{}; \
        desc_dyadic_bins.dtype = dyadic_uint_bytes_dtype<INDEX_T, VALUE_T>(); \
        desc_dyadic_bins.layout = TensorLayout::Dense; \
        desc_dyadic_bins.shape.dims = { (uint32_t)(total_bins * 3u), index_count, (uint32_t)(1u + value_stride) }; \
        TensorDesc desc_staging{}; \
        desc_staging.dtype = dyadic_value_dtype<VALUE_T>(); \
        desc_staging.layout = TensorLayout::Dense; \
        desc_staging.shape.dims = { (uint32_t)((1u << stage_bits) * value_stride) }; \
        TensorDesc desc_counters{}; \
        desc_counters.dtype = TensorDType::Bytes4; \
        desc_counters.layout = TensorLayout::Dense; \
        desc_counters.shape.dims = { (uint32_t)(total_bins * 3u) }; \
        TensorDesc desc_page_active{}; \
        desc_page_active.dtype = TensorDType::Bytes; \
        desc_page_active.layout = TensorLayout::Dense; \
        desc_page_active.shape.dims = { total_bins }; \
        TensorDesc desc_page_retain{}; \
        desc_page_retain.dtype = TensorDType::Bytes; \
        desc_page_retain.layout = TensorLayout::Dense; \
        desc_page_retain.shape.dims = { total_bins }; \
        TensorDesc desc_page_inbox{}; \
        desc_page_inbox.dtype = TensorDType::Bytes; \
        desc_page_inbox.layout = TensorLayout::Dense; \
        desc_page_inbox.shape.dims = { total_bins }; \
        const uint32_t max_possible_pages = (total_bins == 0u) \
            ? 1u \
            : (static_cast<uint32_t>(1u) << total_bins) + 1u; \
        uint32_t page_count = stage_pages_max == 0u \
            ? max_possible_pages \
            : std::min<uint32_t>(stage_pages_max, max_possible_pages); \
        page_count = std::max<uint32_t>(1u, page_count); \
        TensorDesc desc_stage_backing = desc_staging; \
        desc_stage_backing.shape.dims = { (uint32_t)(desc_staging.shape.dims[0] * page_count) }; \
        AbstractTensor dyadic_bins_tensor = pool.acquire_tensor(desc_dyadic_bins, backend); \
        AbstractTensor staging_tensor = pool.acquire_tensor(desc_stage_backing, backend); \
        AbstractTensor counters_tensor = pool.acquire_tensor(desc_counters, backend); \
        AbstractTensor page_active_tensor = pool.acquire_tensor(desc_page_active, backend); \
        AbstractTensor page_retain_tensor = pool.acquire_tensor(desc_page_retain, backend); \
        AbstractTensor page_inbox_tensor = pool.acquire_tensor(desc_page_inbox, backend); \
        uint32_t output_offset = 0; \
        return dyadic_mt_bitmask_algo<PremixPol, OutmixPol>( \
            index_range, \
            index_count, \
            indices, \
            values, \
            value_stride, \
            stage_bits, \
            thread_count, \
            phase_a_rounds, \
            phase_b_rounds, \
            bin0_passes, \
            output, \
            &output_offset, \
            pool, \
            dyadic_bins_tensor, \
            staging_tensor, \
            counters_tensor, \
            page_active_tensor, \
            page_retain_tensor, \
            page_inbox_tensor, \
            stage_pages_max); \
    } \
        inline void dyadic_binning(INDEX_T index_range, uint32_t index_count, INDEX_T* indices, VALUE_T* values, \
                                   uint32_t value_stride, \
                       float_t dense_threshold, float_t* tile_dense_ratios, uint32_t thread_count, \
                       uint32_t phase_a_rounds, uint32_t phase_b_rounds, uint32_t bin0_passes, \
                       VALUE_T* output, uint32_t* output_offset) { \
        const uint32_t stage_bits = 8; \
        const uint32_t total_bins = dyadic_bin_count(index_range, stage_bits); \
        const uint32_t stage_bins = (uint32_t)(1u << stage_bits); \
        const uint32_t output_area = (uint32_t)index_range; \
        const bool should_use_dense = index_count >= output_area * dense_threshold; \
        (void)total_bins; \
        (void)stage_bins; \
        (void)output_area; \
        (void)should_use_dense; \
        (void)indices; \
        (void)values; \
        (void)tile_dense_ratios; \
        AbstractTensorPool pool; \
        TensorDesc desc_dyadic_bins{}; \
        desc_dyadic_bins.dtype = dyadic_uint_bytes_dtype<INDEX_T, VALUE_T>(); \
        desc_dyadic_bins.layout = TensorLayout::Dense; \
        desc_dyadic_bins.shape.dims = { (uint32_t)(total_bins * 3u), index_count, (uint32_t)(1u + value_stride) }; \
        TensorDesc desc_staging{}; \
        desc_staging.dtype = dyadic_value_dtype<VALUE_T>(); \
        desc_staging.layout = TensorLayout::Dense; \
        desc_staging.shape.dims = { (uint32_t)((1u << stage_bits) * value_stride) }; \
        TensorDesc desc_counters{}; \
        desc_counters.dtype = TensorDType::Bytes4; \
        desc_counters.layout = TensorLayout::Dense; \
        desc_counters.shape.dims = { (uint32_t)(total_bins * 3u) }; \
        TensorDesc desc_page_active{}; \
        desc_page_active.dtype = TensorDType::Bytes; \
        desc_page_active.layout = TensorLayout::Dense; \
        desc_page_active.shape.dims = { total_bins }; \
        TensorDesc desc_page_retain{}; \
        desc_page_retain.dtype = TensorDType::Bytes; \
        desc_page_retain.layout = TensorLayout::Dense; \
        desc_page_retain.shape.dims = { total_bins }; \
        TensorDesc desc_page_inbox{}; \
        desc_page_inbox.dtype = TensorDType::Bytes; \
        desc_page_inbox.layout = TensorLayout::Dense; \
        desc_page_inbox.shape.dims = { total_bins }; \
        const uint32_t page_count = (total_bins == 0u) \
            ? 1u \
            : (static_cast<uint32_t>(1u) << total_bins) + 1u; \
        TensorDesc desc_stage_backing = desc_staging; \
        desc_stage_backing.shape.dims = { (uint32_t)(desc_staging.shape.dims[0] * std::max<uint32_t>(1u, page_count)) }; \
        AbstractTensor dyadic_bins_tensor = pool.acquire_tensor(desc_dyadic_bins); \
        AbstractTensor staging_tensor = pool.acquire_tensor(desc_stage_backing); \
        AbstractTensor counters_tensor = pool.acquire_tensor(desc_counters); \
        AbstractTensor page_active_tensor = pool.acquire_tensor(desc_page_active); \
        AbstractTensor page_retain_tensor = pool.acquire_tensor(desc_page_retain); \
        AbstractTensor page_inbox_tensor = pool.acquire_tensor(desc_page_inbox); \
        dyadic_mt_bitmask_algo<policies::Add, policies::Overwrite>( \
            index_range, \
            index_count, \
            indices, \
            values, \
            value_stride, \
            stage_bits, \
            thread_count, \
            phase_a_rounds, \
            phase_b_rounds, \
            bin0_passes, \
            output, \
            output_offset, \
            pool, \
            dyadic_bins_tensor, \
            staging_tensor, \
            counters_tensor, \
            page_active_tensor, \
            page_retain_tensor, \
            page_inbox_tensor); \
    }

#define DYADIC_BINNING_VALUE_LIST(INDEX_T) \
    DYADIC_BINNING_UINT_DEFINE(INDEX_T, uint8_t) \
    DYADIC_BINNING_UINT_DEFINE(INDEX_T, uint16_t) \
    DYADIC_BINNING_UINT_DEFINE(INDEX_T, uint32_t) \
    DYADIC_BINNING_UINT_DEFINE(INDEX_T, uint64_t) \
    DYADIC_BINNING_UINT_DEFINE(INDEX_T, int8_t) \
    DYADIC_BINNING_UINT_DEFINE(INDEX_T, int16_t) \
    DYADIC_BINNING_UINT_DEFINE(INDEX_T, int32_t) \
    DYADIC_BINNING_UINT_DEFINE(INDEX_T, int64_t) \
    DYADIC_BINNING_UINT_DEFINE(INDEX_T, bool) \
    DYADIC_BINNING_UINT_DEFINE(INDEX_T, float) \
    DYADIC_BINNING_UINT_DEFINE(INDEX_T, double)

DYADIC_BINNING_VALUE_LIST(uint8_t)
DYADIC_BINNING_VALUE_LIST(uint16_t)
DYADIC_BINNING_VALUE_LIST(uint32_t)
DYADIC_BINNING_VALUE_LIST(uint64_t)

#undef DYADIC_BINNING_VALUE_LIST

} // namespace nodus::tensors
