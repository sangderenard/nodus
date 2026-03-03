// dyadic binning (tensor-backed)
#pragma once

#include "common/tensors/abstraction/abstract_tensor_pool.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/stream_add.h"

#include <atomic>
#include <cstdint>
#include <cstdarg>
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
#include <condition_variable>
#include <algorithm>
#include <array>
#include <utility>
#include <memory>
#include <chrono>

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
inline void dyadic_logf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
}
inline void dyadic_logf(FILE* file, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(file ? file : stderr, fmt, args);
    va_end(args);
}
} // namespace nodus::tensors
#define DYADIC_LOGGING(...) ::nodus::tensors::dyadic_logf(__VA_ARGS__)
//#define DYADIC_LOGGING(...) ((void)0)
#define DYADIC_TRACE_LOGF(...) std::fprintf(stderr, __VA_ARGS__)
constexpr uint32_t kMaxStagePages = 4096u;

namespace nodus::tensors {
#define UNINITIALIZED_STAGE_OFFSET -1
#define UNCLAIMED_STAGE_OFFSET -3
#define EXPIRED_STAGE_OFFSET -2
#define STAGE_OFFSET_IN_USE -4

namespace policies {
struct Overwrite;
}

struct DyadicStagePagePool;

template <typename IndexPol, typename ValuePol, typename T>
inline void dyadic_apply_span_split(T* dest,
                                    const uint8_t* src_bytes,
                                    uint32_t elem_bytes,
                                    uint32_t value_stride,
                                    uint32_t value_prefix) {
    if (value_prefix) {
        IndexPol::apply_span(dest, src_bytes, elem_bytes, value_prefix);
    }
    const uint32_t value_count = value_stride - value_prefix;
    if (value_count) {
        ValuePol::apply_span(dest + value_prefix,
                             src_bytes + (uint64_t)elem_bytes * value_prefix,
                             elem_bytes,
                             value_count);
    }
}

template <typename INDEX_T, typename VALUE_T>
struct DyadicBinIO3 {
    uint8_t*  bin_data = nullptr;
    uint32_t* page_count = nullptr;
    uint8_t*  page_active = nullptr;
    uint8_t*  page_retain = nullptr;
    uint8_t*  page_inbox = nullptr;
    AbstractTensorHandle ready_pages_tensor{};
    uint32_t* ready_pages_ptr = nullptr;
    uint32_t  ready_pages_capacity = 0;
    std::vector<uint32_t> ready_pages_head;
    std::vector<uint32_t> ready_pages_tail;
    AbstractTensorHandle epoch_tensor{};
    uint32_t* epoch_ptr = nullptr;
    std::vector<uint32_t> epoch_offsets;
    std::vector<uint32_t> epoch_capacity;
    std::vector<uint32_t> epoch_head;
    std::vector<uint32_t> epoch_count;
    uint32_t  index_count = 0;
    uint32_t  elem_bytes = 0;
    uint32_t  slot_bytes = 0;
    uint32_t  value_stride = 1;
    uint32_t  value_prefix = 0;
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

    inline bool enqueue_ready_page(uint32_t bin, uint32_t page_index) {
        if (!ready_pages_ptr || ready_pages_capacity == 0u || bin >= total_bins) return false;
        std::atomic_ref<uint32_t> head(ready_pages_head[bin]);
        std::atomic_ref<uint32_t> tail(ready_pages_tail[bin]);
        const uint32_t h = head.load(std::memory_order_acquire);
        const uint32_t t = tail.load(std::memory_order_relaxed);
        if ((t - h) >= ready_pages_capacity) return false;
        ready_pages_ptr[(size_t)bin * (size_t)ready_pages_capacity + (t % ready_pages_capacity)] = page_index;
        tail.store(t + 1u, std::memory_order_release);
        return true;
    }

    inline bool dequeue_ready_page(uint32_t bin, uint32_t* out_page_index) {
        if (!out_page_index || !ready_pages_ptr || ready_pages_capacity == 0u || bin >= total_bins) return false;
        std::atomic_ref<uint32_t> head(ready_pages_head[bin]);
        std::atomic_ref<uint32_t> tail(ready_pages_tail[bin]);
        const uint32_t h = head.load(std::memory_order_relaxed);
        const uint32_t t = tail.load(std::memory_order_acquire);
        if (h >= t) return false;
        *out_page_index = ready_pages_ptr[(size_t)bin * (size_t)ready_pages_capacity + (h % ready_pages_capacity)];
        head.store(h + 1u, std::memory_order_release);
        return true;
    }


    inline uint32_t* epoch_base(uint32_t bin) const {
        if (!epoch_ptr || bin >= epoch_offsets.size()) return nullptr;
        return epoch_ptr + epoch_offsets[bin];
    }

    inline bool epoch_pop(uint32_t bin, uint32_t* out_epoch) {
        if (!out_epoch || bin >= epoch_count.size()) return false;
        const uint32_t count = epoch_count[bin];
        if (count == 0u) return false;
        const uint32_t cap = epoch_capacity[bin];
        uint32_t* base = epoch_base(bin);
        if (!base || cap == 0u) return false;
        const uint32_t head = epoch_head[bin];
        *out_epoch = base[head % cap];
        epoch_head[bin] = (head + 1u) % cap;
        epoch_count[bin] = count - 1u;
        return true;
    }

    inline bool epoch_push(uint32_t bin, uint32_t epoch) {
        if (bin >= epoch_count.size()) return false;
        const uint32_t cap = epoch_capacity[bin];
        const uint32_t count = epoch_count[bin];
        if (cap == 0u || count >= cap) return false;
        uint32_t* base = epoch_base(bin);
        if (!base) return false;
        const uint32_t head = epoch_head[bin];
        base[(head + count) % cap] = epoch;
        epoch_count[bin] = count + 1u;
        return true;
    }

    inline void epoch_transfer_lower_half(uint32_t bin) {
        if (bin == 0u || bin >= epoch_count.size()) return;
        const uint32_t count = epoch_count[bin];
        const uint32_t half = count / 2u;
        for (uint32_t i = 0; i < half; ++i) {
            uint32_t epoch = 0u;
            if (!epoch_pop(bin, &epoch)) break;
            (void)epoch_push(bin - 1u, epoch);
        }
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

struct alignas(64) DyadicStagePage {
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

struct alignas(64) DyadicSortClassifierThreadCtx {
    void* ctx = nullptr;
    bool (*tick)(void*) = nullptr;
};

struct alignas(64) DyadicStagePagePool {
    DyadicStagePage pages[kMaxStagePages];
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
    bool use_relative_offsets = false;
    bool direct_output = false;
    bool clear_pages = true;
    uint32_t total_bins = 0;
    uint32_t stage_entries = 0;
    uint32_t value_stride = 0;
    uint32_t value_prefix = 0;
    std::vector<int32_t> page_for_bin;
    AbstractTensorHandle queue_tensor{};
    TensorDesc queue_desc{};
    uint8_t* queue_ptr = nullptr;
    size_t queue_bytes = 0;
    uint32_t queue_capacity = 0;
    std::vector<uint32_t> queue_head;
    std::vector<uint32_t> queue_tail;
    AbstractTensorHandle release_tensor{};
    uint32_t* release_ptr = nullptr;
    size_t release_bytes = 0;
    AbstractTensorHandle release_head_tensor{};
    AbstractTensorHandle release_tail_tensor{};
    uint32_t* release_head = nullptr;
    uint32_t* release_tail = nullptr;
    uint32_t release_heads = 0;
    std::atomic<uint32_t> open_pages{0u};
    std::thread recycler;
    std::atomic<bool> recycler_stop_local{false};
    std::atomic<bool>* recycler_stop = nullptr;
    std::thread sort_classifier_thread;
    std::mutex sort_classifier_mutex;
    std::condition_variable sort_classifier_cv;
    DyadicSortClassifierThreadCtx sort_classifier_ctx{};
    std::atomic<bool> sort_classifier_stop{false};
    bool sort_classifier_wake = false;

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
                     bool clear_backing = true,
                     void* backing_ptr_override = nullptr,
                     size_t backing_bytes_override = 0,
                     bool use_relative_offsets_in = false,
                     bool direct_output_in = false,
                     uint32_t value_stride_in = 0,
                     uint32_t value_prefix_in = 0,
                     uint32_t release_heads_in = 0) {
        tensor_pool = pool;
        backend = backend_in;
        mem = mem_in;
        page_desc = page_desc_in;
        total_bins = total_bins_in;
        stage_entries = stage_entries_in;
        value_stride = value_stride_in ? value_stride_in : stage_entries_in;
        value_prefix = std::min<uint32_t>(value_prefix_in, value_stride);
        page_bytes = page_bytes_in;
        use_relative_offsets = use_relative_offsets_in;
        direct_output = direct_output_in;
        clear_pages = clear_backing;
        const uint32_t max_possible_pages = (total_bins == 0u)
            ? 1u
            : (static_cast<uint32_t>(1u) << total_bins) + 1u;
        const uint32_t base_max_pages = max_pages_in == 0 ? max_possible_pages : std::min<uint32_t>(max_pages_in, max_possible_pages);
        const uint32_t stage_span = (stage_entries > 0u) ? (stage_entries / (value_stride ? value_stride : 1u)) : 0u;
        const uint32_t min_pages_for_8k = (stage_span > 0u) ? (uint32_t)((8192u + stage_span - 1u) / stage_span) : 1u;
        max_pages = std::max<uint32_t>(base_max_pages, min_pages_for_8k);
        max_pages = std::min<uint32_t>(max_pages, kMaxStagePages);
        max_pages = std::max<uint32_t>(1u, max_pages);
        use_backing = true;
        backing_external = backing_is_external;
        const bool has_backing_ptr = backing_ptr_override != nullptr;

        if (!mem || !use_backing) {
            max_pages = 0;
        } else if (has_backing_ptr) {
            backing_ptr = backing_ptr_override;
            backing_bytes = backing_bytes_override;
            if (backing_desc_in) {
                backing_desc = *backing_desc_in;
            } else {
                backing_desc = page_desc;
            }
            if (abstract_tensor_handle_is_valid(backing_handle)) {
                backing_tensor = backing_handle;
            }
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

        if (clear_backing && !has_backing_ptr && abstract_tensor_handle_is_valid(backing_tensor)) {
            mem->ensure_zeroed(backing_tensor, backing_desc, backing_external, 0u, 0u);
        }
        if (!has_backing_ptr && abstract_tensor_handle_is_valid(backing_tensor)) {
            mem->map(backing_tensor, &backing_ptr, &backing_bytes);
        }

        page_for_bin.assign(total_bins, -1);
        queue_capacity = std::max<uint32_t>(1u, max_pages);
        queue_tensor = {};
        queue_desc = {};
        queue_ptr = nullptr;
        queue_bytes = 0;
        queue_head.clear();
        queue_tail.clear();
        queue_head.resize(total_bins, 0u);
        queue_tail.resize(total_bins, 0u);
        if (tensor_pool && backend && total_bins > 0u) {
            queue_desc.dtype = TensorDType::Bytes;
            queue_desc.layout = TensorLayout::Dense;
            queue_desc.shape.dims = { total_bins, queue_capacity };
            AbstractTensor queue = tensor_pool->acquire_tensor(queue_desc, backend);
            queue_tensor = queue.handle();
            if (mem && abstract_tensor_handle_is_valid(queue_tensor)) {
                mem->ensure_zeroed(queue_tensor, queue_desc, false, 0u, 0u);
                mem->map(queue_tensor, reinterpret_cast<void**>(&queue_ptr), &queue_bytes);
            }
        }
        page_count.store(0u);
        earliest_spot.store(0);
        for (uint32_t i = 0; i < kMaxStagePages; ++i) {
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
        release_tensor = {};
        release_head_tensor = {};
        release_tail_tensor = {};
        release_ptr = nullptr;
        release_head = nullptr;
        release_tail = nullptr;
        release_bytes = 0;
        release_heads = release_heads_in;
        open_pages.store(0u, std::memory_order_relaxed);
        if (tensor_pool && backend && release_heads > 0u) {
            TensorDesc release_desc{};
            release_desc.dtype = TensorDType::Bytes4;
            release_desc.layout = TensorLayout::Dense;
            release_desc.shape.dims = { release_heads, max_pages };
            AbstractTensor rel = tensor_pool->acquire_tensor(release_desc, backend);
            release_tensor = rel.handle();
            TensorDesc head_desc{};
            head_desc.dtype = TensorDType::Bytes4;
            head_desc.layout = TensorLayout::Dense;
            head_desc.shape.dims = { release_heads };
            AbstractTensor head = tensor_pool->acquire_tensor(head_desc, backend);
            AbstractTensor tail = tensor_pool->acquire_tensor(head_desc, backend);
            release_head_tensor = head.handle();
            release_tail_tensor = tail.handle();
            if (mem && abstract_tensor_handle_is_valid(release_tensor)) {
                mem->ensure_zeroed(release_tensor, release_desc, false, 0u, 0u);
                mem->map(release_tensor, reinterpret_cast<void**>(&release_ptr), &release_bytes);
            }
            if (mem && abstract_tensor_handle_is_valid(release_head_tensor)) {
                mem->ensure_zeroed(release_head_tensor, head_desc, false, 0u, 0u);
                mem->map(release_head_tensor, reinterpret_cast<void**>(&release_head), &release_bytes);
            }
            if (mem && abstract_tensor_handle_is_valid(release_tail_tensor)) {
                mem->ensure_zeroed(release_tail_tensor, head_desc, false, 0u, 0u);
                mem->map(release_tail_tensor, reinterpret_cast<void**>(&release_tail), &release_bytes);
            }
        }
    }

    inline bool assign_page_offset(uint32_t page_index, uint64_t byte_offset) {
        if (page_index >= max_pages) return false;
        const uint64_t resolved_offset = byte_offset * page_bytes;
        pages[page_index].byte_offset = resolved_offset;
        if (backing_ptr) {
            if (backing_bytes && resolved_offset + page_bytes > backing_bytes) return false;
            pages[page_index].mapped = static_cast<uint8_t*>(backing_ptr) + resolved_offset;
            pages[page_index].mapped_bytes = static_cast<size_t>(page_bytes);
        }
        return true;
    }

    template <typename VALUE_T>
    inline VALUE_T* stage_ptr(uint32_t bin) {
        if (!mem || bin >= total_bins || max_pages == 0) return nullptr;
        int32_t idx = page_for_bin[bin];
        if (idx < 0) {
            const uint64_t requested_offset = static_cast<uint64_t>(bin);
            DyadicStagePage* page = add_new_page<VALUE_T>(this, requested_offset);
            if (!page) return nullptr;
            idx = (int32_t)(page - pages);
            page_for_bin[bin] = idx;
        }
        DyadicStagePage& page = pages[idx];
        if (!page.mapped) return nullptr;
        return static_cast<VALUE_T*>(page.mapped);
    }

    inline void mark_dirty(uint32_t bin) {
        if (bin >= total_bins) return;
        const int32_t idx = page_for_bin[bin];
        if (idx >= 0) {
            (void)enqueue_page(bin, (uint32_t)idx);
        }
    }

    inline bool enqueue_page(uint32_t bin, uint32_t page_index) {
        if (!queue_ptr) return false;
        if (bin >= total_bins || queue_capacity == 0u) return false;
        std::atomic_ref<uint32_t> head(queue_head[bin]);
        std::atomic_ref<uint32_t> tail(queue_tail[bin]);
        const uint32_t h = head.load(std::memory_order_acquire);
        const uint32_t t = tail.load(std::memory_order_relaxed);
        if ((t - h) >= queue_capacity) {
            return false;
        }
        queue_ptr[(size_t)bin * (size_t)queue_capacity + (t % queue_capacity)] = (uint8_t)page_index;
        tail.store(t + 1u, std::memory_order_release);
        return true;
    }

    inline bool dequeue_page(uint32_t bin, uint32_t* out_page_index) {
        if (!out_page_index || !queue_ptr) return false;
        if (bin >= total_bins || queue_capacity == 0u) return false;
        std::atomic_ref<uint32_t> head(queue_head[bin]);
        std::atomic_ref<uint32_t> tail(queue_tail[bin]);
        const uint32_t h = head.load(std::memory_order_relaxed);
        const uint32_t t = tail.load(std::memory_order_acquire);
        if (h >= t) {
            return false;
        }
        *out_page_index = queue_ptr[(size_t)bin * (size_t)queue_capacity + (h % queue_capacity)];
        head.store(h + 1u, std::memory_order_release);
        return true;
    }

    inline bool enqueue_release(uint32_t head_id, uint32_t page_index) {
        if (!release_ptr || !release_head || !release_tail || release_heads == 0u) return false;
        if (head_id >= release_heads || max_pages == 0u) return false;
        std::atomic_ref<uint32_t> head(release_head[head_id]);
        std::atomic_ref<uint32_t> tail(release_tail[head_id]);
        const uint32_t h = head.load(std::memory_order_acquire);
        const uint32_t t = tail.load(std::memory_order_relaxed);
        if ((t - h) >= max_pages) return false;
        release_ptr[(size_t)head_id * (size_t)max_pages + (t % max_pages)] = page_index;
        tail.store(t + 1u, std::memory_order_release);
        return true;
    }

    inline bool dequeue_release(uint32_t head_id, uint32_t* out_page_index) {
        if (!out_page_index || !release_ptr || !release_head || !release_tail || release_heads == 0u) return false;
        if (head_id >= release_heads || max_pages == 0u) return false;
        std::atomic_ref<uint32_t> head(release_head[head_id]);
        std::atomic_ref<uint32_t> tail(release_tail[head_id]);
        const uint32_t h = head.load(std::memory_order_relaxed);
        const uint32_t t = tail.load(std::memory_order_acquire);
        if (h >= t) return false;
        *out_page_index = release_ptr[(size_t)head_id * (size_t)max_pages + (h % max_pages)];
        head.store(h + 1u, std::memory_order_release);
        return true;
    }

    inline void start_recycler(std::atomic<bool>* stop_flag) {
        if (recycler.joinable() || release_heads == 0u) return;
        recycler_stop = stop_flag ? stop_flag : &recycler_stop_local;
        recycler_stop->store(false, std::memory_order_release);
        recycler = std::thread([this, stop_flag]() {
            for (;;) {
                DYADIC_LOGGING("Dyadic recycler tick...\n");
                bool drained = false;
                for (uint32_t h = 0; h < release_heads; ++h) {
                    DYADIC_LOGGING("Dyadic recycler checking release queue %u...\n", h);
                    uint32_t page_index = 0;
                    while (dequeue_release(h, &page_index)) {
                        DYADIC_LOGGING("Dyadic recycler releasing page %u from head %u...\n", page_index, h);
                        finish_page(page_index, DyadicStageReleasePolicy::Expire);
                        drained = true;
                    }
                }
                if (recycler_stop && recycler_stop->load(std::memory_order_acquire)) {
                    DYADIC_LOGGING("Dyadic recycler stopping as requested...\n");
                    bool queues_empty = true;
                    for (uint32_t h = 0; h < release_heads; ++h) {
                        DYADIC_LOGGING("Dyadic recycler checking queue %u for emptiness...\n", h);
                        std::atomic_ref<uint32_t> rh(release_head[h]);
                        std::atomic_ref<uint32_t> rt(release_tail[h]);
                        const uint32_t rhv = rh.load(std::memory_order_acquire);
                        const uint32_t rtv = rt.load(std::memory_order_acquire);
                        if (rhv != rtv) {
                            queues_empty = false;
                            break;
                        }
                    }
                    DYADIC_LOGGING("Dyadic recycler queues empty: %s\n", queues_empty ? "yes" : "no");
                    if (queues_empty) break;
                }
                if (!drained) {
                    DYADIC_LOGGING("Dyadic recycler idle...\n");
                    if (open_pages.load(std::memory_order_acquire) == 0u) {
                        std::this_thread::yield();
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(4));
                    }
                }
            }
        });
    }

    inline void stop_recycler() {
        if (recycler.joinable()) {
            if (recycler_stop) {
                DYADIC_LOGGING("Stopping dyadic recycler thread...\n");
                recycler_stop->store(true, std::memory_order_release);
            }
            DYADIC_LOGGING("Joining dyadic recycler thread...\n");
            recycler.join();
        }
        DYADIC_LOGGING("Dyadic recycler thread stopped.\n");
        recycler_stop = nullptr;
    }

    inline void start_sort_classifier_thread(const DyadicSortClassifierThreadCtx& ctx) {
        if (sort_classifier_thread.joinable()) return;
        sort_classifier_ctx = ctx;
        sort_classifier_stop.store(false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(sort_classifier_mutex);
            sort_classifier_wake = false;
        }
        sort_classifier_thread = std::thread([this]() {
            for (;;) {
                std::unique_lock<std::mutex> lock(sort_classifier_mutex);
                sort_classifier_cv.wait(lock, [this]() {
                    return sort_classifier_stop.load(std::memory_order_acquire) || sort_classifier_wake;
                });
                if (sort_classifier_stop.load(std::memory_order_acquire)) {
                    break;
                }
                sort_classifier_wake = false;
                lock.unlock();
                if (sort_classifier_ctx.tick) {
                    while (sort_classifier_ctx.tick(sort_classifier_ctx.ctx)) {
                    }
                }
            }
        });
    }

    inline void signal_sort_classifier_thread() {
        std::lock_guard<std::mutex> lock(sort_classifier_mutex);
        sort_classifier_wake = true;
        sort_classifier_cv.notify_one();
    }

    inline void stop_sort_classifier_thread() {
        {
            std::lock_guard<std::mutex> lock(sort_classifier_mutex);
            sort_classifier_stop.store(true, std::memory_order_release);
            sort_classifier_wake = true;
        }
        sort_classifier_cv.notify_one();
        if (sort_classifier_thread.joinable()) {
            sort_classifier_thread.join();
        }
    }

    inline void attach_queue_ptr(uint8_t* ptr) {
        queue_ptr = ptr;
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
        if (policy != DyadicStageReleasePolicy::Hold) {
            uint32_t prev = open_pages.load(std::memory_order_relaxed);
            while (prev > 0u && !open_pages.compare_exchange_weak(prev, prev - 1u, std::memory_order_relaxed)) {
            }
        }
    }

    inline bool clear_page(uint32_t page_index) {
        if (!clear_pages) return true;
        if (page_index >= max_pages) return false;
        const uint64_t offset = pages[page_index].byte_offset;
        if (backing_ptr) {
            if (backing_bytes && offset + page_bytes > backing_bytes) return false;
            std::memset(static_cast<uint8_t*>(backing_ptr) + offset, 0, static_cast<size_t>(page_bytes));
            return true;
        }
        if (!mem || !use_backing || !abstract_tensor_handle_is_valid(backing_tensor)) return false;
        return mem->ensure_zeroed(backing_tensor, backing_desc, true, offset, page_bytes);
    }

    template <typename OutmixValuePol, typename OutmixIndexPol, typename VALUE_T>
    inline void emit_all(VALUE_T* output,
                         uint32_t* output_offset,
                         DyadicStageReleasePolicy release_policy = DyadicStageReleasePolicy::Expire) {
        if (!output || !output_offset) return;
        const uint32_t total = total_bins;
        for (uint32_t bin = 0; bin < total; ++bin) {
            uint32_t page_index = 0;
            while (dequeue_page(bin, &page_index)) {
                int32_t offset = -1;
                if (!try_claim_ready(page_index, &offset)) {
                    continue;
                }
                DyadicStagePage& page = pages[page_index];
                if (offset >= 0) {
                    dyadic_write_stage_page<VALUE_T, OutmixValuePol, OutmixIndexPol>(
                        *this,
                        output,
                        static_cast<uint32_t>(offset),
                        page);
                    const uint32_t end = ((uint32_t)offset + 1u) * stage_entries;
                    if (*output_offset < end) {
                        *output_offset = end;
                    }
                }
                if (!use_backing && page.mapped) {
                    mem->unmap(page.tensor);
                    page.mapped = nullptr;
                    page.mapped_bytes = 0;
                }
                finish_page(page_index, release_policy);
            }
        }
    }
};

template <typename INDEX_T, typename VALUE_T>
inline void dyadic_drain_ready_pages(DyadicBinIO3<INDEX_T, VALUE_T>& io,
                                     DyadicStagePagePool& stage_pages,
                                     uint32_t bin) {
    if (!io.ready_pages_ptr) return;
    uint32_t page_index = 0u;
    while (io.dequeue_ready_page(bin, &page_index)) {
        (void)stage_pages.enqueue_page(bin, page_index);
    }
}

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
                    if (pool->use_relative_offsets) {
                        if (!pool->assign_page_offset(idx, requested_offset)) {
                            return std::numeric_limits<uint32_t>::max();
                        }
                    }
                    // finalize claim immediately
                    expected = STAGE_OFFSET_IN_USE;
                    const bool success = pool->pages[idx].write_offset.compare_exchange_strong(
                        expected, (int32_t)requested_offset);
                    if (!success) {
                        DYADIC_LOGGING(stderr, "Dyadic binning: failed to claim stage page %u (saw %d)\n", (unsigned)idx, expected);
                        DYADIC_LOGGING(stderr, "This is a multithreading failure.\n");
                        return std::numeric_limits<uint32_t>::max();
                    }
                    pool->open_pages.fetch_add(1u, std::memory_order_relaxed);
                    return (uint32_t)idx;
                }
            }
            else if (offset == EXPIRED_STAGE_OFFSET) {
                int32_t expected = EXPIRED_STAGE_OFFSET;
                clean = pool->pages[idx].write_offset.compare_exchange_strong(expected, STAGE_OFFSET_IN_USE);
                if (clean) {
                    if (pool->use_relative_offsets) {
                        if (!pool->assign_page_offset(idx, requested_offset)) {
                            return std::numeric_limits<uint32_t>::max();
                        }
                    }
                    if (!pool->clear_page(idx)) {
                        return std::numeric_limits<uint32_t>::max();
                    }

                    expected = STAGE_OFFSET_IN_USE;
                    const bool success = pool->pages[idx].write_offset.compare_exchange_strong(
                        expected, (int32_t)requested_offset);
                    if (!success) {
                        DYADIC_LOGGING(stderr, "Dyadic binning: failed to claim stage page %u (saw %d)\n", (unsigned)idx, expected);
                        DYADIC_LOGGING(stderr, "This is a multithreading failure.\n");
                        return std::numeric_limits<uint32_t>::max();
                    }
                    pool->open_pages.fetch_add(1u, std::memory_order_relaxed);
                    return (uint32_t)idx;
                }
            }
            else if (offset == UNINITIALIZED_STAGE_OFFSET) {
                int32_t expected = UNINITIALIZED_STAGE_OFFSET;
                clean = pool->pages[idx].write_offset.compare_exchange_strong(expected, STAGE_OFFSET_IN_USE);
                if (clean) {
                    if (pool->use_relative_offsets) {
                        if (!pool->assign_page_offset(idx, requested_offset)) {
                            return std::numeric_limits<uint32_t>::max();
                        }
                    }
                    if (!pool->clear_page(idx)) {
                        return std::numeric_limits<uint32_t>::max();
                    }

                    expected = STAGE_OFFSET_IN_USE;
                    const bool success = pool->pages[idx].write_offset.compare_exchange_strong(
                        expected, (int32_t)requested_offset);
                    if (!success) {
                        DYADIC_LOGGING(stderr, "Dyadic binning: failed to claim stage page %u (saw %d)\n", (unsigned)idx, expected);
                        DYADIC_LOGGING(stderr, "This is a multithreading failure.\n");
                        return std::numeric_limits<uint32_t>::max();
                    }
                    pool->open_pages.fetch_add(1u, std::memory_order_relaxed);
                    return (uint32_t)idx;
                }
            }
        }
    }
    return std::numeric_limits<uint32_t>::max();
}


template <typename VALUE_T>
inline DyadicStagePage* add_new_page(DyadicStagePagePool* pool, uint64_t offset) {
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

struct DyadicJobCtx;

inline void dyadic_exec_swap_inbox(
    uint32_t bin_id,
    uint8_t* duty_state,
    uint8_t* page_active,
    uint8_t* page_inbox,
    uint8_t* page_retain);

inline void dyadic_exec_swap_retain(
    uint32_t bin_id,
    uint8_t* duty_state,
    uint8_t* page_active,
    uint8_t* page_inbox,
    uint8_t* page_retain);

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_exec_bin_sort(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* phase_counter,
    DyadicJobCtx* job,
    bool tick_or_block,
    bool use_msb_override,
    INDEX_T msb_mask,
    uint32_t sort_epoch);

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline bool run_dyadic_schedule_step(
    DyadicScheduleOp op,
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* duty_state,
    uint8_t* phase_counter)
{
    const uint32_t u = static_cast<uint32_t>(op);
    const uint32_t base = static_cast<uint32_t>(DyadicScheduleOp::Bin0Sort);
    if (u >= base && u < base + kBinSortCount) {
        const uint32_t bin = u - base;
        dyadic_exec_bin_sort<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
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

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_exec_bin_sort_single_thread(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    VALUE_T* stage,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* phase_counter,
    DyadicStagePagePool* stage_pages = nullptr,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = false) {
    if (!stage && !stage_pages) {
        return;
    }
    VALUE_T* stage_ptr = stage;
    if (stage_pages) {
        stage_ptr = stage_pages->template stage_ptr<VALUE_T>(bin_id);
    }
    if (!stage_ptr) {
        return;
    }
    if (bin_id == 0u) {
        dyadic_bin0_drain_paged<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
            io, stage_ptr, stage_bits, stage_pages, job, tick_or_block);
    } else if (bin_id == 1u) {
        dyadic_cascade_phase_a_paged<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
            io, bin_id, stage_ptr, stage_bits, stage_pages, job, tick_or_block);
    } else {
        bool do_phase_a = true;
        if (phase_counter) {
            do_phase_a = ((phase_counter[bin_id] & 1u) == 0u);
            phase_counter[bin_id] = (uint8_t)(phase_counter[bin_id] + 1u);
        }
        if (do_phase_a) {
            dyadic_cascade_phase_a_paged<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                io, bin_id, stage_ptr, stage_bits, stage_pages, job, tick_or_block);
        } else {
            dyadic_cascade_phase_b_paged<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                io, bin_id, stage_ptr, stage_bits, stage_pages, job, tick_or_block);
        }
    }
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol, size_t N>
inline void dyadic_run_single_thread_schedule(
    const std::array<DyadicScheduleOp, N>& schedule,
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    VALUE_T* stage,
    uint32_t stage_bits,
    uint8_t* duty_state,
    uint8_t* phase_counter,
    DyadicStagePagePool* stage_pages = nullptr,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = false) {
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
            dyadic_exec_bin_sort_single_thread<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                io,
                stage,
                stage_bits,
                bin,
                phase_counter,
                stage_pages,
                job,
                tick_or_block);
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

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol, size_t N>
inline void dyadic_run_single_thread_schedule_stage_pages(
    const std::array<DyadicScheduleOp, N>& schedule,
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint8_t* duty_state,
    uint8_t* phase_counter,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = false) {
    const uint32_t sort_base = static_cast<uint32_t>(DyadicScheduleOp::Bin0Sort);
    const uint32_t inbox_base = static_cast<uint32_t>(DyadicScheduleOp::Bin0SwapInbox);
    const uint32_t retain_base = static_cast<uint32_t>(DyadicScheduleOp::Bin0SwapRetain);
    DYADIC_LOGGING("N: %zu\n", N);
    for (size_t i = 0; i < N; ++i) {
        DYADIC_LOGGING("Step %zu: Op %u\n", i, static_cast<uint32_t>(schedule[i]));
        const uint32_t u = static_cast<uint32_t>(schedule[i]);
        if (u >= sort_base && u < sort_base + kBinSortCount) {
            DYADIC_LOGGING("  Bin Sort\n");
            const uint32_t bin = u - sort_base;
            if (bin >= io.total_bins) {
                continue;
            }
            dyadic_exec_bin_sort<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                io,
                stage_pages,
                stage_bits,
                bin,
                phase_counter,
                job,
                tick_or_block);
            continue;
        }
        if (u >= inbox_base && u < inbox_base + kBinSortCount) {
            DYADIC_LOGGING("  Swap Inbox\n");
            const uint32_t bin = u - inbox_base;
            if (bin >= io.total_bins) {
                continue;
            }
            dyadic_exec_swap_inbox(bin, duty_state, io.page_active, io.page_inbox, io.page_retain);
            continue;
        }
        if (u >= retain_base && u < retain_base + kBinSortCount) {
            DYADIC_LOGGING("  Swap Retain\n");
            const uint32_t bin = u - retain_base;
            if (bin >= io.total_bins) {
                continue;
            }
            dyadic_exec_swap_retain(bin, duty_state, io.page_active, io.page_inbox, io.page_retain);
            continue;
        }
    }
    DYADIC_LOGGING("Done\n");
}


struct alignas(64) DyadicThreadPolicy {
    uint32_t thread_count = 0;
    uint32_t bin_workers = 1;
    uint32_t write_heads = 1;
    uint32_t write_head_queue_len = 2;
    uint32_t ratio = 2;
    bool force_single_threaded_bins = false;
    bool force_single_threaded_writers = false;
    bool force_single_thread = false;
    bool allow_parallel_bins = false;
};

enum DyadicJobRingFlags : uint32_t {
    DyadicJobRingNone = 0u,
    DyadicJobRingTerminal = 1u << 0
};

struct alignas(64) DyadicJobRingEntry {
    AbstractTensorHandle rows{};
    uint32_t row_count = 0;
    uint32_t bins_offset = 0;
    uint32_t bins_count = 0;
    uint32_t stage_page_index = 0;
    uint32_t flags = DyadicJobRingNone;
};

struct alignas(64) DyadicJobCtx {
    uint32_t job_id = 0;
    AbstractTensorHandle ring_tensor{};
    DyadicJobRingEntry* ring_ptr = nullptr;
    uint32_t ring_capacity = 0;
    uint32_t ring_head = 0;
    uint32_t ring_tail = 0;
    std::atomic<uint8_t> done{0};
    std::atomic<uint32_t> pending{0u};
    std::atomic<uint32_t> enqueued{0u};
    std::atomic<uint32_t> completed{0u};
    std::mutex pending_mutex;
    std::condition_variable pending_cv;
    AbstractTensorHandle bins_tensor{};
    uint32_t* bins_ptr = nullptr;
    uint32_t bins_count = 0;
    uint32_t stage_page_index = 0;
};

inline bool dyadic_job_ring_enqueue(DyadicJobCtx& job,
                                    const DyadicJobRingEntry& entry,
                                    DyadicStagePagePool* stage_pages,
                                    bool blocking);

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
struct alignas(64) DyadicSortClassifierCtx {
    DyadicBinIO3<INDEX_T, VALUE_T>* io = nullptr;
    DyadicStagePagePool* stage_pages = nullptr;
    InMemoryBackend* mem = nullptr;
    DyadicJobCtx* jobs = nullptr;
    uint32_t job_count = 0;
    uint32_t stage_bits = 0;
    std::atomic<bool>* stop = nullptr;
};

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_classify_job_rows_bytes(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    const uint8_t* rows,
    uint32_t row_count,
    uint32_t stage_bin,
    const DyadicJobCtx& job,
    const DyadicJobRingEntry& entry) {
    if (!rows || row_count == 0u) return;
    const uint32_t stride_bytes = io.elem_bytes * (1u + io.value_stride);
    for (uint32_t r = 0; r < row_count; ++r) {
        const uint8_t* row = rows + (uint64_t)r * (uint64_t)stride_bytes;
        INDEX_T idx{};
        DYADIC_RAW_WRITE(&idx, row, sizeof(INDEX_T));
        const uint8_t* val_bytes = row + io.elem_bytes;
        dyadic_classify_to_stage_or_bin_paged_stage_pages_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
            io, stage_pages, stage_bits, idx, val_bytes, stage_bin);
    }
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_classify_job_entry(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    InMemoryBackend& mem,
    const DyadicJobCtx& job,
    const DyadicJobRingEntry& entry) {
    if (!abstract_tensor_handle_is_valid(entry.rows)) return;
    if (entry.row_count == 0u) return;
    void* rows_ptr = nullptr;
    size_t rows_bytes = 0;
    if (!mem.map(entry.rows, &rows_ptr, &rows_bytes)) return;
    dyadic_classify_job_rows_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
        io,
        stage_pages,
        stage_bits,
        static_cast<const uint8_t*>(rows_ptr),
        entry.row_count,
        entry.stage_page_index,
        job,
        entry);
    mem.unmap(entry.rows);
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline bool dyadic_enqueue_job_row_bytes(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    DyadicJobCtx& job,
    INDEX_T idx,
    const uint8_t* val_bytes,
    uint32_t stage_bin,
    bool blocking) {
    if (!stage_pages.tensor_pool || !stage_pages.backend || !stage_pages.mem) return false;
    TensorDesc row_desc{};
    row_desc.dtype = TensorDType::Bytes;
    row_desc.layout = TensorLayout::Dense;
    row_desc.shape.dims = { 1u, io.slot_bytes };
    AbstractTensor row_tensor = stage_pages.tensor_pool->acquire_tensor(row_desc, stage_pages.backend);
    if (!row_tensor.valid()) return false;
    void* row_ptr = nullptr;
    size_t row_bytes = 0;
    if (!stage_pages.mem->map(row_tensor.handle(), &row_ptr, &row_bytes)) return false;
    if (row_bytes < io.slot_bytes) {
        stage_pages.mem->unmap(row_tensor.handle());
        return false;
    }
    uint8_t* dst = static_cast<uint8_t*>(row_ptr);
    DYADIC_RAW_WRITE(dst, &idx, sizeof(INDEX_T));
    std::memcpy(dst + io.elem_bytes, val_bytes, (size_t)io.value_stride * (size_t)io.elem_bytes);
    stage_pages.mem->unmap(row_tensor.handle());
    DyadicJobRingEntry entry{};
    entry.rows = row_tensor.handle();
    entry.row_count = 1u;
    entry.stage_page_index = stage_bin;
    DYADIC_LOGGING("Enqueueing job row idx %llu to stage bin %u\n", (unsigned long long)idx, stage_bin);
    return dyadic_job_ring_enqueue(job, entry, &stage_pages, blocking);
}

template <typename INDEX_T, typename VALUE_T>
inline bool dyadic_job_enqueue_terminal_blocking(DyadicBinIO3<INDEX_T, VALUE_T>& io,
                                                 DyadicJobCtx& job,
                                                 DyadicStagePagePool& stage_pages,
                                                 uint32_t bin_id,
                                                 uint32_t sort_epoch,
                                                 bool enqueue_ready) {
    DyadicJobRingEntry entry{};
    entry.flags = DyadicJobRingTerminal;
    job.done.store(1u, std::memory_order_release);
    (void)bin_id;
    (void)sort_epoch;
    (void)enqueue_ready;
    return dyadic_job_ring_enqueue(job, entry, &stage_pages, true);
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline bool dyadic_sort_classifier_tick(DyadicSortClassifierCtx<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>& ctx) {
    if (!ctx.io || !ctx.stage_pages || !ctx.mem || !ctx.jobs || ctx.job_count == 0u) return false;
    bool did_work = false;
    for (uint32_t j = 0; j < ctx.job_count; ++j) {
        DyadicJobCtx& job = ctx.jobs[j];
        DyadicJobRingEntry entry{};
        while (dyadic_job_ring_dequeue(job, &entry)) {
            dyadic_classify_job_entry<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                *ctx.io,
                *ctx.stage_pages,
                ctx.stage_bits,
                *ctx.mem,
                job,
                entry);
            job.pending.fetch_sub(1u, std::memory_order_acq_rel);
            job.completed.fetch_add(1u, std::memory_order_acq_rel);
            {
                std::lock_guard<std::mutex> lock(job.pending_mutex);
            }
            job.pending_cv.notify_all();
            did_work = true;
        }
    }
    return did_work;
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline bool dyadic_sort_classifier_tick_void(void* ctx) {
    auto* typed = static_cast<DyadicSortClassifierCtx<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>*>(ctx);
    if (!typed) return false;
    return dyadic_sort_classifier_tick(*typed);
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_sort_classifier_loop(DyadicSortClassifierCtx<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>& ctx) {
    if (!ctx.io || !ctx.stage_pages || !ctx.mem || !ctx.jobs || ctx.job_count == 0u) return;
    for (;;) {
        const bool did_work = dyadic_sort_classifier_tick(ctx);
        if (!did_work) {
            if (ctx.stop && ctx.stop->load(std::memory_order_acquire)) {
                break;
            }
            std::this_thread::yield();
        }
    }
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline bool dyadic_sort_classifier_all_done(DyadicSortClassifierCtx<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>& ctx) {
    if (!ctx.jobs || ctx.job_count == 0u) return true;
    for (uint32_t j = 0; j < ctx.job_count; ++j) {
        DyadicJobCtx& job = ctx.jobs[j];
        if (job.done.load(std::memory_order_acquire) == 0u) {
            return false;
        }
        std::atomic_ref<uint32_t> head(job.ring_head);
        std::atomic_ref<uint32_t> tail(job.ring_tail);
        const uint32_t h = head.load(std::memory_order_acquire);
        const uint32_t t = tail.load(std::memory_order_acquire);
        if (h != t) {
            return false;
        }
    }
    return true;
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_sort_classifier_blocking(DyadicSortClassifierCtx<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>& ctx) {
    if (!ctx.io || !ctx.stage_pages || !ctx.mem || !ctx.jobs || ctx.job_count == 0u) return;
    while (!dyadic_sort_classifier_all_done(ctx)) {
        (void)dyadic_sort_classifier_tick(ctx);
        std::this_thread::yield();
    }
}

inline bool dyadic_job_ring_enqueue(DyadicJobCtx& job,
                                    const DyadicJobRingEntry& entry,
                                    DyadicStagePagePool* stage_pages = nullptr,
                                    bool blocking = false) {
    if (!job.ring_ptr || job.ring_capacity == 0u) return false;
    std::atomic_ref<uint32_t> head(job.ring_head);
    std::atomic_ref<uint32_t> tail(job.ring_tail);
    const uint32_t h = head.load(std::memory_order_acquire);
    const uint32_t t = tail.load(std::memory_order_relaxed);
    if ((t - h) >= job.ring_capacity) return false;
    const uint32_t idx = t % job.ring_capacity;
    job.ring_ptr[idx] = entry;
    const uint32_t seq = job.enqueued.fetch_add(1u, std::memory_order_acq_rel) + 1u;
    job.pending.fetch_add(1u, std::memory_order_acq_rel);
    tail.store(t + 1u, std::memory_order_release);
    DYADIC_LOGGING("[dyadic] Enqueued job at index %u\n", idx);
    if (stage_pages) {
        DYADIC_LOGGING("[dyadic] Signaling sort classifier thread\n");
        stage_pages->signal_sort_classifier_thread();
    }
    if (blocking) {
        DYADIC_LOGGING("[dyadic] Waiting for job to complete\n");
        std::unique_lock<std::mutex> lock(job.pending_mutex);
        job.pending_cv.wait(lock, [&job, seq]() {
            DYADIC_LOGGING("[dyadic] Waiting for job completion, seq=%u, completed=%u\n", seq, job.completed.load(std::memory_order_acquire));
            return job.completed.load(std::memory_order_acquire) >= seq;
        });
        return true;
    }
    return true;
}

inline bool dyadic_job_ring_dequeue(DyadicJobCtx& job, DyadicJobRingEntry* out_entry) {
    if (!job.ring_ptr || job.ring_capacity == 0u) return false;
    std::atomic_ref<uint32_t> head(job.ring_head);
    std::atomic_ref<uint32_t> tail(job.ring_tail);
    const uint32_t h = head.load(std::memory_order_relaxed);
    const uint32_t t = tail.load(std::memory_order_acquire);
    if (h >= t) return false;
    const uint32_t idx = h % job.ring_capacity;
    if (out_entry) {
        *out_entry = job.ring_ptr[idx];
    }
    head.store(h + 1u, std::memory_order_release);
    return true;
}

const DyadicThreadPolicy NO_THREAD_POLICY = DyadicThreadPolicy{
    1u,
    std::numeric_limits<uint32_t>::max(),
    std::numeric_limits<uint32_t>::max(),
    2u,
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
        policy.write_head_queue_len = std::max<uint32_t>(1u, policy.write_head_queue_len);
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
        DYADIC_LOGGING(stderr, "[dyadic] %s=%d\n", name, value ? 1 : 0);
    } else if constexpr (std::is_floating_point<T>::value) {
        DYADIC_LOGGING(stderr, "[dyadic] %s=%g\n", name, static_cast<double>(value));
    } else if constexpr (std::is_signed<T>::value) {
        DYADIC_LOGGING(stderr, "[dyadic] %s=%lld\n", name, static_cast<long long>(value));
    } else {
        DYADIC_LOGGING(stderr, "[dyadic] %s=%llu\n", name, static_cast<unsigned long long>(value));
    }
}

inline void dyadic_log_tensor(const char* name, const AbstractTensor& t) {
    const TensorDesc& desc = t.desc();
    DYADIC_LOGGING(
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
        DYADIC_LOGGING(stderr, "%u", desc.shape.dims[i]);
        if (i + 1u < desc.shape.dims.size()) {
            DYADIC_LOGGING(stderr, ",");
        }
    }
    DYADIC_LOGGING(stderr, "]\n");
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

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_classify_to_stage_or_bin_paged_stage_pages(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    INDEX_T idx,
    const VALUE_T* vals,
    uint32_t src_bin);

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_classify_to_stage_or_bin_paged_stage_pages_bytes(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    INDEX_T idx,
    const uint8_t* val_bytes,
    uint32_t src_bin);

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
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
        dyadic_classify_to_stage_or_bin_paged_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
            io, stage, stage_bits, idx_move, val_bytes);
        slot += 1u;
    }
    io.cnt(src_bin, src_page) = 0;
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
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
        dyadic_classify_to_stage_or_bin_paged_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
            io, stage, stage_bits, idx2, val_bytes);
        slot += 1u;
    }
    io.cnt(src_bin, src_page) = 0;
}


template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
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
        dyadic_apply_span_split<PremixIndexPol, PremixValuePol>(
            dst, val_bytes, io.elem_bytes, io.value_stride, io.value_prefix);
        slot += 1u;
    }
    io.cnt(src_bin, src_page) = 0;
}



template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
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
        dyadic_apply_span_split<PremixIndexPol, PremixValuePol>(
            dst, reinterpret_cast<const uint8_t*>(vals), io.elem_bytes, io.value_stride, io.value_prefix);
        return;
    }
    uint32_t b = dyadic_msb_pos_nonzero<INDEX_T>(hi);
    io.push_inbox(b, idx, vals);
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_classify_to_stage_or_bin_paged_stage_pages(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    INDEX_T idx,
    const VALUE_T* vals,
    uint32_t src_bin) {
    const INDEX_T hi = (INDEX_T)(idx >> stage_bits);
    VALUE_T* stage = stage_pages.template stage_ptr<VALUE_T>(src_bin);
    if (!stage && hi == 0) {
        return;
    }
    dyadic_classify_to_stage_or_bin_paged<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
        io, stage, stage_bits, idx, vals);
    if (hi == 0 && stage) {
        stage_pages.mark_dirty(src_bin);
    }
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_classify_to_stage_or_bin_paged_stage_pages_bytes(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    INDEX_T idx,
    const uint8_t* val_bytes,
    uint32_t src_bin) {
    const INDEX_T hi = (INDEX_T)(idx >> stage_bits);
    VALUE_T* stage = stage_pages.template stage_ptr<VALUE_T>(src_bin);
    if (!stage && hi == 0) {
        return;
    }
    dyadic_classify_to_stage_or_bin_paged_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
        io, stage, stage_bits, idx, val_bytes);
    if (hi == 0 && stage) {
        stage_pages.mark_dirty(src_bin);
    }
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
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
        dyadic_apply_span_split<PremixIndexPol, PremixValuePol>(
            dst, val_bytes, io.elem_bytes, io.value_stride, io.value_prefix);
        return;
    }
    uint32_t b = dyadic_msb_pos_nonzero<INDEX_T>(hi);
    io.push_inbox_bytes(b, idx, val_bytes);
}


template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_cascade_phase_a_paged(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    uint32_t src_bin,
    VALUE_T* stage,
    uint32_t stage_bits,
    DyadicStagePagePool* stage_pages = nullptr,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = true,
    bool use_msb_override = false,
    INDEX_T msb_mask = 0,
    uint32_t sort_epoch = 0) {
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
        if (job && stage_pages) {
            if (use_msb_override) {
                const INDEX_T hi_override = (INDEX_T)(idx_move >> stage_bits);
                if ((hi_override & msb_mask) == msb_mask) {
                    const INDEX_T hi_stripped = (INDEX_T)(hi_override & ~msb_mask);
                    const INDEX_T idx_stripped = (INDEX_T)((hi_stripped << stage_bits) | lo);
                    (void)dyadic_enqueue_job_row_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                        io, *stage_pages, *job, idx_stripped, val_bytes, sort_epoch, false);
                    slot += 1u;
                    continue;
                }
            }
            (void)dyadic_enqueue_job_row_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                io, *stage_pages, *job, idx_move, val_bytes, src_bin, false);
        } else {
            dyadic_classify_to_stage_or_bin_paged_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                io, stage, stage_bits, idx_move, val_bytes);
        }
        slot += 1u;
    }
    if (job && stage_pages && tick_or_block) {
        (void)dyadic_job_enqueue_terminal_blocking(io, *job, *stage_pages, src_bin, sort_epoch, use_msb_override);
    }
    io.finish_scan(src_bin);
}


template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_cascade_phase_b_paged(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    uint32_t src_bin,
    VALUE_T* stage,
    uint32_t stage_bits,
    DyadicStagePagePool* stage_pages = nullptr,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = true,
    bool use_msb_override = false,
    INDEX_T msb_mask = 0,
    uint32_t sort_epoch = 0) {
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
        if (job && stage_pages) {
            if (use_msb_override) {
                const INDEX_T hi_override = (INDEX_T)(idx2 >> stage_bits);
                if ((hi_override & msb_mask) == msb_mask) {
                    const INDEX_T hi_stripped = (INDEX_T)(hi_override & ~msb_mask);
                    const INDEX_T idx_stripped = (INDEX_T)((hi_stripped << stage_bits) | lo);
                    (void)dyadic_enqueue_job_row_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                        io, *stage_pages, *job, idx_stripped, val_bytes, sort_epoch, false);
                    slot += 1u;
                    continue;
                }
            }
            (void)dyadic_enqueue_job_row_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                io, *stage_pages, *job, idx2, val_bytes, src_bin, false);
        } else {
            dyadic_classify_to_stage_or_bin_paged_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                io, stage, stage_bits, idx2, val_bytes);
        }
        slot += 1u;
    }
    if (job && stage_pages && tick_or_block) {
        (void)dyadic_job_enqueue_terminal_blocking(io, *job, *stage_pages, src_bin, sort_epoch, use_msb_override);
    }
    io.finish_scan(src_bin);
}
#define TICK false
#define BLOCK true

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_bin0_drain_paged(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    VALUE_T* stage,
    uint32_t stage_bits,
    DyadicStagePagePool* stage_pages = nullptr,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = TICK,
    bool use_msb_override = false,
    INDEX_T msb_mask = 0,
    uint32_t sort_epoch = 0) {
    DYADIC_LOGGING("[dyadic] Draining bin 0\n");
    DYADIC_LOGGING("[dyadic]   use_msb_override=%d msb_mask=0x%llx sort_epoch=%u\n",
           use_msb_override ? 1 : 0,
           static_cast<unsigned long long>(msb_mask),
           sort_epoch);
    const uint32_t bin = 0;
    io.prepare_for_scan(bin);
    const uint8_t A = io.active_page(bin);
    const uint32_t n = io.cnt(bin, A);
    const INDEX_T lo_mask = ((INDEX_T)1u << stage_bits) - 1u;
    uint8_t* base = io.page_base(bin, A);
    uint32_t slot = 0;
    while (slot ^ n) {
        DYADIC_LOGGING("[dyadic]   draining slot %u / %u\r", slot + 1u, n);
        uint8_t* p = base + (uint64_t)slot * (uint64_t)io.slot_bytes;
        INDEX_T idx;
        DYADIC_RAW_WRITE(&idx, p, sizeof(INDEX_T));
        const uint8_t* val_bytes = p + io.elem_bytes;
        const INDEX_T lo = (INDEX_T)(idx & lo_mask);
        if (job && stage_pages) {
            if (use_msb_override) {
                const INDEX_T hi_override = (INDEX_T)(idx >> stage_bits);
                if ((hi_override & msb_mask) == msb_mask) {
                    const INDEX_T hi_stripped = (INDEX_T)(hi_override & ~msb_mask);
                    const INDEX_T idx_stripped = (INDEX_T)((hi_stripped << stage_bits) | lo);
                    (void)dyadic_enqueue_job_row_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                        io, *stage_pages, *job, idx_stripped, val_bytes, sort_epoch, false);
                    slot += 1u;
                    continue;
                }
            }
            (void)dyadic_enqueue_job_row_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                io, *stage_pages, *job, idx, val_bytes, bin, false);
        } else {
            VALUE_T* dst = stage + (uint64_t)lo * (uint64_t)io.value_stride;
            dyadic_apply_span_split<PremixIndexPol, PremixValuePol>(
                dst, val_bytes, io.elem_bytes, io.value_stride, io.value_prefix);
        }
        slot += 1u;
    }
    if (job && stage_pages && tick_or_block) {
        (void)dyadic_job_enqueue_terminal_blocking(io, *job, *stage_pages, bin, sort_epoch, use_msb_override);
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
    uint8_t* duty_state,
    uint8_t* page_active,
    uint8_t* page_inbox,
    uint8_t* page_retain) {
    std::atomic_ref<uint8_t> state_ref(duty_state[bin_id]);
    const uint8_t state = state_ref.load(std::memory_order_acquire);
    const uint8_t next = dyadic_duty_swap_contents_inbox(state);
    state_ref.store(next, std::memory_order_release);
    dyadic_sync_pages_from_state(bin_id, next, page_active, page_inbox, page_retain);
}

inline void dyadic_exec_swap_retain(
    uint32_t bin_id,
    uint8_t* duty_state,
    uint8_t* page_active,
    uint8_t* page_inbox,
    uint8_t* page_retain) {
    std::atomic_ref<uint8_t> state_ref(duty_state[bin_id]);
    const uint8_t state = state_ref.load(std::memory_order_acquire);
    const uint8_t next = dyadic_duty_swap_contents_retain(state);
    state_ref.store(next, std::memory_order_release);
    dyadic_sync_pages_from_state(bin_id, next, page_active, page_inbox, page_retain);
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_exec_bin_sort(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* phase_counter,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = TICK,
    bool use_msb_override = false,
    INDEX_T msb_mask = 0,
    uint32_t sort_epoch = 0) {
    DYADIC_LOGGING("[dyadic] Sorting bin %u\n", bin_id);
    DYADIC_LOGGING("[dyadic]   use_msb_override=%d msb_mask=0x%llx sort_epoch=%u\n",
           use_msb_override ? 1 : 0,
           static_cast<unsigned long long>(msb_mask),
           sort_epoch);
    VALUE_T* stage = stage_pages.template stage_ptr<VALUE_T>(bin_id);
    if (!stage) {
        return;
    }
    if (bin_id == 0u) {
        uint32_t epoch_local = sort_epoch;
        if (use_msb_override) {
            (void)io.epoch_pop(bin_id, &epoch_local);
        }
        dyadic_bin0_drain_paged<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
            io, stage, stage_bits, &stage_pages, job, tick_or_block, use_msb_override, msb_mask, epoch_local);
    } else if (bin_id == 1u) {
        uint32_t epoch_local = sort_epoch;
        if (use_msb_override) {
            (void)io.epoch_pop(bin_id, &epoch_local);
        }
        dyadic_cascade_phase_a_paged<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
            io, bin_id, stage, stage_bits, &stage_pages, job, tick_or_block, use_msb_override, msb_mask, epoch_local);
    } else {
        bool do_phase_a = true;
        if (phase_counter) {
            do_phase_a = ((phase_counter[bin_id] & 1u) == 0u);
            phase_counter[bin_id] = (uint8_t)(phase_counter[bin_id] + 1u);
        }
        uint32_t epoch_local = sort_epoch;
        if (use_msb_override) {
            (void)io.epoch_pop(bin_id, &epoch_local);
        }
        if (do_phase_a) {
            dyadic_cascade_phase_a_paged<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                io, bin_id, stage, stage_bits, &stage_pages, job, tick_or_block, use_msb_override, msb_mask, epoch_local);
        } else {
            dyadic_cascade_phase_b_paged<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                io, bin_id, stage, stage_bits, &stage_pages, job, tick_or_block, use_msb_override, msb_mask, epoch_local);
        }
    }
    stage_pages.mark_dirty(bin_id);
    if (bin_id > 0u && !use_msb_override) {
        io.epoch_transfer_lower_half(bin_id);
    }
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol, size_t N>
inline void dyadic_bin_worker_run_schedule(
    const DyadicScheduleOp (&schedule)[N],
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* duty_state,
    uint8_t* last_upstream_flip,
    uint8_t* last_downstream_flip,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = TICK,
    bool use_msb_override = false,
    INDEX_T msb_mask = 0,
    uint32_t sort_epoch = 0) {
    (void)stage_pages;
    (void)last_upstream_flip;
    for (size_t i = 0; i < N; ++i) {
        const DyadicScheduleOp op = schedule[i];
        if (op == dyadic_bin_sort_op(bin_id)) {
            dyadic_exec_bin_sort<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
                io,
                stage_pages,
                stage_bits,
                bin_id,
                last_downstream_flip,
                job,
                tick_or_block,
                use_msb_override,
                msb_mask,
                sort_epoch);
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


template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_bin_worker_step_bin0(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* duty_state,
    uint8_t* last_upstream_flip,
    uint8_t* last_downstream_flip,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = TICK,
    bool use_msb_override = false,
    INDEX_T msb_mask = 0,
    uint32_t sort_epoch = 0) {
    static constexpr DyadicScheduleOp schedule[] = {
        DyadicScheduleOp::Bin0SwapInbox,
        DyadicScheduleOp::Bin0Sort,
    };
    dyadic_bin_worker_run_schedule<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
        schedule,
        io,
        stage_pages,
        stage_bits,
        bin_id,
        duty_state,
        last_upstream_flip,
        last_downstream_flip,
        job,
        tick_or_block,
        use_msb_override,
        msb_mask,
        sort_epoch);
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_bin_worker_step_bin1(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* duty_state,
    uint8_t* last_upstream_flip,
    uint8_t* last_downstream_flip,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = TICK,
    bool use_msb_override = false,
    INDEX_T msb_mask = 0,
    uint32_t sort_epoch = 0) {
    static constexpr DyadicScheduleOp schedule[] = {
        DyadicScheduleOp::Bin1Sort,
        DyadicScheduleOp::Bin1SwapRetain,
    };
    dyadic_bin_worker_run_schedule<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
        schedule,
        io,
        stage_pages,
        stage_bits,
        bin_id,
        duty_state,
        last_upstream_flip,
        last_downstream_flip,
        job,
        tick_or_block,
        use_msb_override,
        msb_mask,
        sort_epoch);
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_bin_worker_step_binn(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* duty_state,
    uint8_t* last_upstream_flip,
    uint8_t* last_downstream_flip,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = TICK,
    bool use_msb_override = false,
    INDEX_T msb_mask = 0,
    uint32_t sort_epoch = 0) {
    const DyadicScheduleOp schedule[] = {
        dyadic_bin_sort_op(bin_id),
        dyadic_bin_swap_retain_op(bin_id),
    };
    dyadic_bin_worker_run_schedule<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
        schedule,
        io,
        stage_pages,
        stage_bits,
        bin_id,
        duty_state,
        last_upstream_flip,
        last_downstream_flip,
        job,
        tick_or_block,
        use_msb_override,
        msb_mask,
        sort_epoch);
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_bin_worker_step_last(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* duty_state,
    uint8_t* last_upstream_flip,
    uint8_t* last_downstream_flip,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = TICK,
    bool use_msb_override = false,
    INDEX_T msb_mask = 0,
    uint32_t sort_epoch = 0) {
    const DyadicScheduleOp schedule[] = {
        dyadic_bin_sort_op(bin_id),
        dyadic_bin_swap_retain_op(bin_id),
    };
    dyadic_bin_worker_run_schedule<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
        schedule,
        io,
        stage_pages,
        stage_bits,
        bin_id,
        duty_state,
        last_upstream_flip,
        last_downstream_flip,
        job,
        tick_or_block,
        use_msb_override,
        msb_mask,
        sort_epoch);
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_bin_worker_step(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    DyadicStagePagePool& stage_pages,
    uint32_t stage_bits,
    uint32_t bin_id,
    uint8_t* duty_state,
    uint8_t* last_upstream_flip,
    uint8_t* last_downstream_flip,
    DyadicJobCtx* job = nullptr,
    bool tick_or_block = TICK,
    bool use_msb_override = false,
    INDEX_T msb_mask = 0,
    uint32_t sort_epoch = 0) {
    if (bin_id == 0u) {
        dyadic_bin_worker_step_bin0<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
            io, stage_pages, stage_bits, bin_id, duty_state, last_upstream_flip, last_downstream_flip, job, tick_or_block, use_msb_override, msb_mask, sort_epoch);
    } else if (bin_id == 1u) {
        dyadic_bin_worker_step_bin1<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
            io, stage_pages, stage_bits, bin_id, duty_state, last_upstream_flip, last_downstream_flip, job, tick_or_block, use_msb_override, msb_mask, sort_epoch);
    } else if (bin_id + 1u == io.total_bins) {
        dyadic_bin_worker_step_last<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
            io, stage_pages, stage_bits, bin_id, duty_state, last_upstream_flip, last_downstream_flip, job, tick_or_block, use_msb_override, msb_mask, sort_epoch);
    } else {
        dyadic_bin_worker_step_binn<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
            io, stage_pages, stage_bits, bin_id, duty_state, last_upstream_flip, last_downstream_flip, job, tick_or_block, use_msb_override, msb_mask, sort_epoch);
    }

    if (tick_or_block == BLOCK) {
        dyadic_drain_ready_pages(io, stage_pages, bin_id);
    }
}

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
struct alignas(64) DyadicBinWorkerJobCtx {
    DyadicBinIO3<INDEX_T, VALUE_T>* io = nullptr;
    DyadicStagePagePool* stage_pages = nullptr;
    uint32_t stage_bits = 0;
    uint32_t bin_id = 0;
    uint8_t* duty_state = nullptr;
    uint8_t* last_upstream_flip = nullptr;
    uint8_t* last_downstream_flip = nullptr;
    DyadicJobCtx* job = nullptr;
    bool tick_or_block = true;
    bool use_msb_override = false;
    INDEX_T msb_mask = 0;
    uint32_t sort_epoch = 0;
    std::atomic<bool>* stop = nullptr;
};

template <typename INDEX_T, typename VALUE_T, typename PremixValuePol, typename PremixIndexPol>
inline void dyadic_bin_worker_job_fn(const nodus::ThreadPool::Job& job, uint32_t /*worker_id*/) {
    auto* ctx = static_cast<DyadicBinWorkerJobCtx<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>*>(job.user);
    if (!ctx || !ctx->io || !ctx->stage_pages) return;
    dyadic_bin_worker_step<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(
        *ctx->io,
        *ctx->stage_pages,
        ctx->stage_bits,
        ctx->bin_id,
        ctx->duty_state,
        ctx->last_upstream_flip,
        ctx->last_downstream_flip,
        ctx->job,
        ctx->tick_or_block,
        ctx->use_msb_override,
        ctx->msb_mask,
        ctx->sort_epoch);
}

template <typename INDEX_T, typename VALUE_T, typename OutmixValuePol, typename OutmixIndexPol>
struct alignas(64) DyadicStageWriterJobCtx {
    DyadicBinIO3<INDEX_T, VALUE_T>* io = nullptr;
    DyadicStagePagePool* stage_pages = nullptr;
    VALUE_T* output = nullptr;
    uint32_t* output_offset = nullptr;
    uint32_t stage_bits = 0;
    uint32_t* inbox_ptr = nullptr;
    uint32_t* write_idx = nullptr;
    uint32_t* read_idx = nullptr;
    uint32_t head_id = 0;
    uint32_t head_count = 0;
    uint32_t queue_len = 0;
    std::atomic<bool>* stop = nullptr;
};

struct alignas(64) DyadicWriteHeadInboxView {
    uint32_t* ptr = nullptr;
    uint32_t* write_idx = nullptr;
    uint32_t* read_idx = nullptr;
    uint32_t head_count = 0;
    uint32_t queue_len = 0;
};

inline bool dyadic_write_head_push(DyadicWriteHeadInboxView inbox,
                                   uint32_t head_id,
                                   uint32_t page_index,
                                   uint32_t write_offset) {
    if (!inbox.ptr || !inbox.write_idx || !inbox.read_idx || head_id >= inbox.head_count) return false;
    if (inbox.queue_len == 0u) return false;
    std::atomic_ref<uint32_t> widx(inbox.write_idx[head_id]);
    std::atomic_ref<uint32_t> ridx(inbox.read_idx[head_id]);
    const uint32_t r = ridx.load(std::memory_order_acquire);
    const uint32_t w = widx.load(std::memory_order_relaxed);
    const uint32_t capacity = inbox.queue_len * 2u;
    if ((w - r) >= capacity) return false;
    const uint32_t idx = widx.fetch_add(1u, std::memory_order_acq_rel);
    const uint32_t slot = idx % inbox.queue_len;
    const uint32_t buffer = (idx / inbox.queue_len) & 1u;
    const size_t base = ((size_t)head_id * 2u + (size_t)buffer) * (size_t)inbox.queue_len * 2u
                      + (size_t)slot * 2u;
    inbox.ptr[base + 0u] = page_index;
    inbox.ptr[base + 1u] = write_offset;
    return true;
}

inline bool dyadic_write_head_pop(DyadicWriteHeadInboxView inbox,
                                  uint32_t head_id,
                                  uint32_t* out_page_index,
                                  uint32_t* out_write_offset) {
    if (!inbox.ptr || !inbox.write_idx || !inbox.read_idx || head_id >= inbox.head_count) return false;
    if (inbox.queue_len == 0u) return false;
    std::atomic_ref<uint32_t> widx(inbox.write_idx[head_id]);
    std::atomic_ref<uint32_t> ridx(inbox.read_idx[head_id]);
    const uint32_t r = ridx.load(std::memory_order_relaxed);
    const uint32_t w = widx.load(std::memory_order_acquire);
    if (r >= w) return false;
    const uint32_t idx = ridx.fetch_add(1u, std::memory_order_acq_rel);
    const uint32_t slot = idx % inbox.queue_len;
    const uint32_t buffer = (idx / inbox.queue_len) & 1u;
    const size_t base = ((size_t)head_id * 2u + (size_t)buffer) * (size_t)inbox.queue_len * 2u
                      + (size_t)slot * 2u;
    if (out_page_index) *out_page_index = inbox.ptr[base + 0u];
    if (out_write_offset) *out_write_offset = inbox.ptr[base + 1u];
    return true;
}

template <typename INDEX_T, typename VALUE_T>
inline void dyadic_write_head_tick(DyadicWriteHeadInboxView inbox,
                                   DyadicStagePagePool& stage_pages,
                                   VALUE_T* output,
                                   uint32_t head_id) {
    for (;;) {
        uint32_t page_index = 0u;
        uint32_t write_offset = 0u;
        if (!dyadic_write_head_pop(inbox, head_id, &page_index, &write_offset)) {
            break;
        }
        if (!stage_pages.direct_output && stage_pages.backing_ptr) {
            const uint32_t offset_bytes = write_offset * (uint32_t)stage_pages.page_bytes;
            const uint8_t* src = static_cast<const uint8_t*>(stage_pages.backing_ptr) + offset_bytes;
            uint8_t* dst = reinterpret_cast<uint8_t*>(output) + offset_bytes;
            std::memcpy(dst, src, stage_pages.page_bytes);
        }
        (void)stage_pages.enqueue_release(head_id, page_index);
    }
}

inline bool dyadic_write_head_empty(DyadicWriteHeadInboxView inbox) {
    if (!inbox.write_idx || !inbox.read_idx) return true;
    for (uint32_t h = 0; h < inbox.head_count; ++h) {
        std::atomic_ref<uint32_t> rref(inbox.read_idx[h]);
        std::atomic_ref<uint32_t> wref(inbox.write_idx[h]);
        const uint32_t r = rref.load(std::memory_order_acquire);
        const uint32_t w = wref.load(std::memory_order_acquire);
        if (r != w) return false;
    }
    return true;
}

template <typename INDEX_T, typename VALUE_T>
struct alignas(64) DyadicQueueScannerJobCtx {
    DyadicStagePagePool* stage_pages = nullptr;
    DyadicWriteHeadInboxView inbox{};
    uint32_t* head_cursor = nullptr;
    std::atomic<bool>* stop = nullptr;
};

template <typename VALUE_T, typename OutmixValuePol, typename OutmixIndexPol>
inline void dyadic_write_stage_page(DyadicStagePagePool& stage_pages,
                                    VALUE_T* output,
                                    uint32_t offset,
                                    DyadicStagePage& page) {
    if (!output || !page.mapped) return;
    if (stage_pages.direct_output) return;
    VALUE_T* out = output + (uint64_t)offset * (uint64_t)stage_pages.stage_entries;
    const uint32_t stride = stage_pages.value_stride ? stage_pages.value_stride : stage_pages.stage_entries;
    const uint32_t prefix = std::min<uint32_t>(stage_pages.value_prefix, stride);
    const uint32_t elem_count = (stride == 0) ? 0u : (stage_pages.stage_entries / stride);
    for (uint32_t i = 0; i < elem_count; ++i) {
        const auto* src = static_cast<const VALUE_T*>(page.mapped) + (uint64_t)i * stride;
        auto* dst = out + (uint64_t)i * stride;
        dyadic_apply_span_split<OutmixIndexPol, OutmixValuePol>(
            dst,
            reinterpret_cast<const uint8_t*>(src),
            (uint32_t)sizeof(VALUE_T),
            stride,
            prefix);
    }
}

template <typename INDEX_T, typename VALUE_T, typename OutmixValuePol, typename OutmixIndexPol>
inline void dyadic_drain_stage_queues(DyadicStagePagePool& stage_pages,
                                      VALUE_T* output,
                                      uint32_t* output_offset) {
    const uint32_t total = stage_pages.total_bins;
    for (uint32_t bin = 0; bin < total; ++bin) {
        uint32_t page_index = 0;
        while (stage_pages.dequeue_page(bin, &page_index)) {
            int32_t offset = -1;
            if (!stage_pages.try_claim_ready(page_index, &offset)) {
                continue;
            }
            DyadicStagePage& page = stage_pages.pages[page_index];
            if (offset >= 0) {
                dyadic_write_stage_page<VALUE_T, OutmixValuePol, OutmixIndexPol>(
                    stage_pages,
                    output,
                    static_cast<uint32_t>(offset),
                    page);
                if (output_offset && offset >= 0) {
                    const uint32_t end = ((uint32_t)offset + 1u) * stage_pages.stage_entries;
                    if (*output_offset < end) {
                        *output_offset = end;
                    }
                }
            }
            if (!stage_pages.use_backing && page.mapped) {
                stage_pages.mem->unmap(page.tensor);
                page.mapped = nullptr;
                page.mapped_bytes = 0;
            }
            stage_pages.finish_page(page_index, DyadicStageReleasePolicy::Expire);
        }
    }
}

template <typename INDEX_T, typename VALUE_T, typename OutmixValuePol, typename OutmixIndexPol>
inline void dyadic_stage_writer_job_fn(const nodus::ThreadPool::Job& job, uint32_t /*worker_id*/) {
    auto* ctx = static_cast<DyadicStageWriterJobCtx<INDEX_T, VALUE_T, OutmixValuePol, OutmixIndexPol>*>(job.user);
    if (!ctx || !ctx->stage_pages || !ctx->output) return;
    DyadicStagePagePool& stage_pages = *ctx->stage_pages;
    DyadicWriteHeadInboxView inbox{ ctx->inbox_ptr, ctx->write_idx, ctx->read_idx, ctx->head_count, ctx->queue_len };
    while (!ctx->stop || !ctx->stop->load(std::memory_order_acquire)) {
        uint32_t page_index = 0u;
        uint32_t write_offset = 0u;
        if (!dyadic_write_head_pop(inbox, ctx->head_id, &page_index, &write_offset)) {
            if (ctx->stop && ctx->stop->load(std::memory_order_acquire)) break;
            continue;
        }
        if (!stage_pages.direct_output && stage_pages.backing_ptr) {
            const uint32_t offset_bytes = write_offset * (uint32_t)stage_pages.page_bytes;
            const uint8_t* src = static_cast<const uint8_t*>(stage_pages.backing_ptr) + offset_bytes;
            uint8_t* dst = reinterpret_cast<uint8_t*>(ctx->output) + offset_bytes;
            std::memcpy(dst, src, stage_pages.page_bytes);
        }
        (void)stage_pages.enqueue_release(ctx->head_id, page_index);
    }
}

template <typename INDEX_T, typename VALUE_T>
inline void dyadic_queue_scanner_job_fn(const nodus::ThreadPool::Job& job, uint32_t /*worker_id*/) {
    auto* ctx = static_cast<DyadicQueueScannerJobCtx<INDEX_T, VALUE_T>*>(job.user);
    if (!ctx || !ctx->stage_pages || !ctx->head_cursor) return;
    DyadicStagePagePool& stage_pages = *ctx->stage_pages;
    DyadicWriteHeadInboxView inbox = ctx->inbox;
    std::atomic_ref<uint32_t> cursor_ref(*ctx->head_cursor);
    const uint32_t total_bins = stage_pages.total_bins;
    for (;;) {
        bool did_work = false;
        for (uint32_t bin = 0; bin < total_bins; ++bin) {
            uint32_t page_index = 0;
            while (stage_pages.dequeue_page(bin, &page_index)) {
                int32_t offset = -1;
                if (!stage_pages.try_claim_ready(page_index, &offset)) {
                    continue;
                }
                if (offset < 0) continue;
                const uint32_t cursor = cursor_ref.fetch_add(1u, std::memory_order_relaxed);
                const uint32_t head_id = cursor % inbox.head_count;
                if (!dyadic_write_head_push(inbox, head_id, (uint32_t)page_index, (uint32_t)offset)) {
                    // If inbox is full, drop for now.
                }
                did_work = true;
            }
        }
        if (!did_work) {
            if (ctx->stop && ctx->stop->load(std::memory_order_acquire)) break;
        }
    }
}

template <typename INDEX_T, typename VALUE_T>
inline void dyadic_queue_scanner_tick(DyadicQueueScannerJobCtx<INDEX_T, VALUE_T>& ctx) {
    if (!ctx.stage_pages || !ctx.head_cursor) return;
    DyadicStagePagePool& stage_pages = *ctx.stage_pages;
    DyadicWriteHeadInboxView inbox = ctx.inbox;
    std::atomic_ref<uint32_t> cursor_ref(*ctx.head_cursor);
    const uint32_t total_bins = stage_pages.total_bins;
    for (uint32_t bin = 0; bin < total_bins; ++bin) {
        uint32_t page_index = 0;
        while (stage_pages.dequeue_page(bin, &page_index)) {
            int32_t offset = -1;
            if (!stage_pages.try_claim_ready(page_index, &offset)) {
                continue;
            }
            if (offset < 0) continue;
            const uint32_t cursor = cursor_ref.fetch_add(1u, std::memory_order_relaxed);
            const uint32_t head_id = cursor % inbox.head_count;
            (void)dyadic_write_head_push(inbox, head_id, (uint32_t)page_index, (uint32_t)offset);
        }
    }
}

template <typename VALUE_T, typename OutmixValuePol, typename OutmixIndexPol>
inline void dyadic_emit_stage_linear(VALUE_T* output, uint32_t* output_offset, const VALUE_T* stage,
                                     uint32_t stage_bits, uint32_t value_stride, uint32_t value_prefix) {
    const uint32_t stage_len = (uint32_t)(1u << stage_bits);
    VALUE_T* out = output + *output_offset;
    const uint32_t prefix = std::min<uint32_t>(value_prefix, value_stride);
    for (uint32_t i = 0; i < stage_len; ++i) {
        const auto* src = stage + (uint64_t)i * value_stride;
        auto* dst = out + (uint64_t)i * value_stride;
        dyadic_apply_span_split<OutmixIndexPol, OutmixValuePol>(
            dst,
            reinterpret_cast<const uint8_t*>(src),
            (uint32_t)sizeof(VALUE_T),
            value_stride,
            prefix);
    }
    *output_offset = (uint32_t)(*output_offset + stage_len * value_stride);
}

template <typename VALUE_T, typename OutmixValuePol, typename OutmixIndexPol>
inline bool dyadic_emit_stage_linear(AbstractTensor& output,
                                     uint32_t* output_offset,
                                     const VALUE_T* stage,
                                     uint32_t stage_bits,
                                     uint32_t value_stride,
                                     uint32_t value_prefix) {
    if (!output.valid()) return false;
    auto* mem = dynamic_cast<InMemoryBackend*>(output.backend());
    if (!mem) return false;
    void* out_ptr = nullptr;
    size_t out_bytes = 0;
    if (!mem->map(output.handle(), &out_ptr, &out_bytes)) return false;
    dyadic_emit_stage_linear<VALUE_T, OutmixValuePol, OutmixIndexPol>(
        static_cast<VALUE_T*>(out_ptr),
        output_offset,
        stage,
        stage_bits,
        value_stride,
        value_prefix);
    mem->unmap(output.handle());
    return true;
}


#define DYADIC_BINNING_UINT_DEFINE(INDEX_T, VALUE_T) \
    template <typename PremixValuePol = policies::Add, \
              typename PremixIndexPol = policies::Overwrite, \
              typename OutmixValuePol = policies::Overwrite, \
              typename OutmixIndexPol = policies::Overwrite> \
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
        std::vector<uint8_t> duty_state(io.total_bins, 0u); \
        std::vector<uint8_t> last_up(io.total_bins, 0xFFu); \
        std::vector<uint8_t> last_down(io.total_bins, 0xFFu); \
        for (uint32_t bin = 0; bin < io.total_bins; ++bin) { \
            uint8_t state = dyadic_duty_pack(io.page_active[bin], io.page_inbox[bin], io.page_retain[bin], 0u); \
            duty_state[bin] = state; \
            dyadic_sync_pages_from_state(bin, state, io.page_active, io.page_inbox, io.page_retain); \
        } \
        std::atomic<bool> stop{false}; \
        nodus::ThreadPool::Options opt{}; \
        opt.thread_count = policy.bin_workers; \
        opt.start_immediately = true; \
        nodus::ThreadPool pool(opt); \
        std::vector<DyadicBinWorkerJobCtx<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>> ctxs(io.total_bins); \
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
            jobs[bin].fn = &dyadic_bin_worker_job_fn<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>; \
            jobs[bin].user = &ctxs[bin]; \
        } \
        auto batch = pool.submit_batch(jobs.data(), (uint32_t)jobs.size()); \
        if (batch) batch->wait(); \
    }\
    template <typename PremixValuePol = policies::Add, \
              typename PremixIndexPol = policies::Overwrite, \
              typename OutmixValuePol = policies::Overwrite, \
              typename OutmixIndexPol = policies::Overwrite> \
    inline void dyadic_mt_bitmask_algo(INDEX_T index_range, uint32_t index_count, INDEX_T* indices, VALUE_T* values, \
                                       uint32_t value_stride, \
                                       uint32_t stage_bits, uint32_t thread_count, \
                                       uint32_t phase_a_rounds, uint32_t phase_b_rounds, uint32_t bin0_passes, \
                                       VALUE_T* output, uint32_t* output_offset, \
                                       AbstractTensorPool& pool, \
                                       AbstractTensor& bins, AbstractTensor& staging, AbstractTensor& counters, \
                                       AbstractTensor& page_active, AbstractTensor& page_retain, \
                                       AbstractTensor& page_inbox, \
                                       uint32_t stage_pages_max = 0, \
                                       bool output_as_stage_pool = true, \
                                       bool clear_output_as_stage_pool = true, \
                                       size_t output_bytes = 0, \
                                       uint32_t index_stride = 0, \
                                       uint32_t index_channels = 1, \
                                       DyadicJobCtx* jobs = nullptr, \
                                       uint32_t job_count = 0, \
                                       bool run_sort_classifier_thread = true) { \
        InMemoryBackend* mem = static_cast<InMemoryBackend*>(bins.backend()); \
        void* bin_data_void = nullptr; \
        size_t bin_bytes = 0; \
        mem->map(bins.handle(), &bin_data_void, &bin_bytes); \
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
        const uint32_t channel_count = std::max<uint32_t>(1u, index_channels); \
        const uint32_t value_prefix = channel_count - 1u; \
        const uint32_t packed_value_stride = value_stride + value_prefix; \
        const uint32_t stride = (index_stride == 0u) ? index_count : index_stride; \
        io.value_stride = packed_value_stride; \
        io.value_prefix = std::min<uint32_t>(value_prefix, packed_value_stride); \
        io.total_bins = dyadic_bin_count(index_range, stage_bits); \
        io.value_bytes = io.elem_bytes * packed_value_stride; \
        io.slot_bytes = io.elem_bytes * (1u + packed_value_stride); \
        io.ready_pages_head.assign(io.total_bins, 0u); \
        io.ready_pages_tail.assign(io.total_bins, 0u); \
        if (io.total_bins > 0u) { \
            const uint32_t ready_capacity = std::max<uint32_t>(1u, stage_pages_max ? stage_pages_max : io.total_bins); \
            TensorDesc ready_desc{}; \
            ready_desc.dtype = TensorDType::Bytes4; \
            ready_desc.layout = TensorLayout::Dense; \
            ready_desc.shape.dims = { io.total_bins, ready_capacity }; \
            AbstractTensor ready_tensor = pool.acquire_tensor(ready_desc, bins.backend()); \
            void* ready_ptr_void = nullptr; \
            size_t ready_bytes = 0; \
            if (ready_tensor.valid() && mem->map(ready_tensor.handle(), &ready_ptr_void, &ready_bytes)) { \
                mem->ensure_zeroed(ready_tensor.handle(), ready_desc, true, 0u, 0u); \
                io.ready_pages_tensor = ready_tensor.handle(); \
                io.ready_pages_ptr = static_cast<uint32_t*>(ready_ptr_void); \
                io.ready_pages_capacity = ready_capacity; \
            } \
        } \
        io.epoch_offsets.assign(io.total_bins, 0u); \
        io.epoch_capacity.assign(io.total_bins, 0u); \
        io.epoch_head.assign(io.total_bins, 0u); \
        io.epoch_count.assign(io.total_bins, 0u); \
        if (io.total_bins > 0u) { \
            uint64_t epoch_total = 0u; \
            for (uint32_t b = 0; b < io.total_bins; ++b) { \
                const uint32_t cap = (b < 31u) ? (1u << b) : 0u; \
                io.epoch_offsets[b] = (uint32_t)epoch_total; \
                io.epoch_capacity[b] = cap; \
                io.epoch_head[b] = 0u; \
                io.epoch_count[b] = cap; \
                epoch_total += cap; \
            } \
            if (epoch_total > 0u) { \
                TensorDesc epoch_desc{}; \
                epoch_desc.dtype = TensorDType::Bytes4; \
                epoch_desc.layout = TensorLayout::Dense; \
                epoch_desc.shape.dims = { (uint32_t)epoch_total }; \
                AbstractTensor epoch_tensor = pool.acquire_tensor(epoch_desc, bins.backend()); \
                void* epoch_ptr_void = nullptr; \
                size_t epoch_bytes = 0; \
                if (epoch_tensor.valid() && mem->map(epoch_tensor.handle(), &epoch_ptr_void, &epoch_bytes)) { \
                    io.epoch_tensor = epoch_tensor.handle(); \
                    io.epoch_ptr = static_cast<uint32_t*>(epoch_ptr_void); \
                    for (uint32_t b = 0; b < io.total_bins; ++b) { \
                        const uint32_t cap = io.epoch_capacity[b]; \
                        const uint32_t start = (b < 31u) ? (1u << b) : 0u; \
                        uint32_t* base = io.epoch_ptr + io.epoch_offsets[b]; \
                        for (uint32_t i = 0; i < cap; ++i) { \
                            base[i] = start + i; \
                        } \
                    } \
                } \
            } \
        } \
        DYADIC_LOGGING("Dyadic Binning: index_count=%u, total_bins=%u, value_stride=%u, packed_value_stride=%u\n", \
               io.index_count, io.total_bins, value_stride, packed_value_stride); \
        const uint32_t stage_entries = (uint32_t)((1u << stage_bits) * packed_value_stride); \
        const uint64_t stage_page_bytes = (uint64_t)stage_entries * (uint64_t)sizeof(VALUE_T); \
        bool use_output_stage_pool = output_as_stage_pool && output && output_offset; \
        uint32_t output_base_offset = 0u; \
        VALUE_T* output_base_ptr = output; \
        size_t output_pool_bytes = output_bytes; \
        if (use_output_stage_pool) { \
            output_base_offset = output_offset ? *output_offset : 0u; \
            output_base_ptr = output + output_base_offset; \
            if (output_bytes > 0u) { \
                const uint64_t base_bytes = static_cast<uint64_t>(output_base_offset) * sizeof(VALUE_T); \
                if (base_bytes > output_bytes) { \
                    use_output_stage_pool = false; \
                    output_base_ptr = output; \
                    output_pool_bytes = output_bytes; \
                } else { \
                    output_pool_bytes = output_bytes - static_cast<size_t>(base_bytes); \
                } \
            } \
        } \
        uint32_t page_count = stage_pages_max; \
        const uint32_t max_possible_pages = (io.total_bins == 0u) \
            ? 1u \
            : (static_cast<uint32_t>(1u) << io.total_bins) + 1u; \
        if (use_output_stage_pool) { \
            page_count = std::max<uint32_t>(1u, io.total_bins); \
            if (output_pool_bytes > 0u) { \
                const uint64_t required_bytes = stage_page_bytes * static_cast<uint64_t>(page_count); \
                if (required_bytes > output_pool_bytes) { \
                    use_output_stage_pool = false; \
                } \
            } \
        } else { \
            if (page_count == 0u) { \
                page_count = max_possible_pages; \
            } \
            page_count = std::max<uint32_t>(1u, std::min<uint32_t>(page_count, max_possible_pages)); \
        } \
        if (!use_output_stage_pool) { \
            (void)mem->ensure_zeroed(staging.handle(), staging.desc()); \
        } \
        TensorDesc stage_page_desc = staging.desc(); \
        stage_page_desc.shape.dims = { stage_entries }; \
        const DyadicThreadPolicy policy = dyadic_read_thread_policy(thread_count, io.total_bins); \
        const uint32_t release_heads = (policy.write_heads < std::numeric_limits<uint32_t>::max()) ? std::max<uint32_t>(1u, policy.write_heads): 1u; \
        DyadicStagePagePool stage_pages; \
        stage_pages.init(&pool, bins.backend(), mem, stage_page_desc, io.total_bins, stage_entries, \
                         stage_page_bytes, page_count, \
                         use_output_stage_pool ? AbstractTensorHandle{} : staging.handle(), \
                         use_output_stage_pool ? nullptr : &staging.desc(), \
                         false, \
                         use_output_stage_pool ? clear_output_as_stage_pool : false, \
                         use_output_stage_pool ? static_cast<void*>(output_base_ptr) : nullptr, \
                         use_output_stage_pool ? output_pool_bytes : 0u, \
                         use_output_stage_pool, \
                         use_output_stage_pool, \
                         packed_value_stride, \
                         io.value_prefix, \
                         release_heads); \
        std::atomic<bool> classifier_stop{false}; \
        DyadicSortClassifierCtx<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol> classifier_ctx{}; \
        DyadicSortClassifierThreadCtx classifier_thread_ctx{}; \
        DyadicJobCtx local_job{}; \
        bool local_job_active = false; \
        AbstractTensor local_ring_tensor{}; \
        if (!jobs && job_count == 0u && dyadic_is_single_threaded(policy)) { \
            const uint32_t ring_capacity = std::max<uint32_t>(1u, index_count + 1u); \
            TensorDesc ring_desc{}; \
            ring_desc.dtype = TensorDType::Bytes; \
            ring_desc.layout = TensorLayout::Dense; \
            ring_desc.shape.dims = { ring_capacity, (uint32_t)sizeof(DyadicJobRingEntry) }; \
            local_ring_tensor = pool.acquire_tensor(ring_desc, bins.backend()); \
            void* ring_ptr_void = nullptr; \
            size_t ring_bytes = 0; \
            if (local_ring_tensor.valid() && mem->map(local_ring_tensor.handle(), &ring_ptr_void, &ring_bytes)) { \
                local_job.ring_tensor = local_ring_tensor.handle(); \
                local_job.ring_ptr = static_cast<DyadicJobRingEntry*>(ring_ptr_void); \
                local_job.ring_capacity = ring_capacity; \
                jobs = &local_job; \
                job_count = 1u; \
                local_job_active = true; \
            } \
        } \
        if (jobs && job_count > 0u) { \
            classifier_ctx.io = &io; \
            classifier_ctx.stage_pages = &stage_pages; \
            classifier_ctx.mem = mem; \
            classifier_ctx.jobs = jobs; \
            classifier_ctx.job_count = job_count; \
            classifier_ctx.stage_bits = stage_bits; \
            classifier_ctx.stop = &classifier_stop; \
            classifier_thread_ctx.ctx = &classifier_ctx; \
            classifier_thread_ctx.tick = &dyadic_sort_classifier_tick_void<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>; \
            if (!dyadic_is_single_threaded(policy) && run_sort_classifier_thread) { \
                stage_pages.start_sort_classifier_thread(classifier_thread_ctx); \
                stage_pages.signal_sort_classifier_thread(); \
            } \
        } \
        uint32_t b = 0; \
        while (b ^ io.total_bins) { \
            io.page_active[b] = 0; \
            io.page_retain[b] = 1; \
            io.page_inbox[b] = 2; \
            b += 1u; \
        } \
        (void)clear_output_as_stage_pool; \
        AbstractTensorPool::PooledTensor packed_tensor; \
        void* packed_ptr = nullptr; \
        size_t packed_bytes_len = 0; \
        if (value_prefix) { \
            TensorDesc packed_desc{}; \
            packed_desc.dtype = TensorDType::Bytes; \
            packed_desc.layout = TensorLayout::Dense; \
            packed_desc.shape.dims = { packed_value_stride, io.elem_bytes }; \
            packed_tensor = pool.acquire(packed_desc, bins.backend()); \
            if (packed_tensor.valid()) { \
                mem->map(packed_tensor.tensor().handle(), &packed_ptr, &packed_bytes_len); \
            } else { \
                packed_ptr = nullptr; \
                packed_bytes_len = 0; \
            } \
        } \
        uint32_t i = 0; \
        while (i ^ index_count) { \
            const INDEX_T idx = indices[(uint64_t)i + (uint64_t)0u * (uint64_t)stride]; \
            const VALUE_T* vals = values + (uint64_t)i * (uint64_t)value_stride; \
            if (value_prefix) { \
                if (!packed_ptr || packed_bytes_len < (size_t)packed_value_stride * io.elem_bytes) { \
                    return; \
                } \
                std::memset(packed_ptr, 0, packed_bytes_len); \
                for (uint32_t c = 1u; c < channel_count; ++c) { \
                    const INDEX_T idxc = indices[(uint64_t)i + (uint64_t)c * (uint64_t)stride]; \
                    const size_t offset = static_cast<size_t>(io.elem_bytes) * (c - 1u); \
                    DYADIC_RAW_WRITE(static_cast<uint8_t*>(packed_ptr) + offset, &idxc, sizeof(INDEX_T)); \
                } \
                for (uint32_t v = 0u; v < value_stride; ++v) { \
                    const VALUE_T val = vals[v]; \
                    const size_t offset = static_cast<size_t>(io.elem_bytes) * (value_prefix + v); \
                    DYADIC_RAW_WRITE(static_cast<uint8_t*>(packed_ptr) + offset, &val, sizeof(VALUE_T)); \
                } \
                if (jobs && job_count > 0u) { \
                    (void)dyadic_enqueue_job_row_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>( \
                        io, stage_pages, jobs[0], idx, static_cast<const uint8_t*>(packed_ptr), 0u, false); \
                } else { \
                    dyadic_classify_to_stage_or_bin_paged_stage_pages_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>( \
                        io, stage_pages, stage_bits, idx, static_cast<const uint8_t*>(packed_ptr), 0u); \
                } \
            } else { \
                if (jobs && job_count > 0u) { \
                    (void)dyadic_enqueue_job_row_bytes<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>( \
                        io, stage_pages, jobs[0], idx, reinterpret_cast<const uint8_t*>(vals), 0u, false); \
                } else { \
                    dyadic_classify_to_stage_or_bin_paged_stage_pages<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>( \
                        io, stage_pages, stage_bits, idx, vals, 0u); \
                } \
            } \
            i += 1u; \
        } \
        DYADIC_LOGGING("Dyadic Binning: classification complete, draining stage pages...\n"); \
        if (local_job_active) { \
            DYADIC_LOGGING("Dyadic Binning: finalizing local job ring...\n"); \
            DyadicJobRingEntry terminal{}; \
            terminal.flags = DyadicJobRingTerminal; \
            local_job.done.store(1u, std::memory_order_release); \
            (void)dyadic_job_ring_enqueue(local_job, terminal, &stage_pages, false); \
            dyadic_sort_classifier_blocking<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(classifier_ctx); \
        } \
        if (packed_ptr) { \
            mem->unmap(packed_tensor.tensor().handle()); \
        } \
        if (policy.allow_parallel_bins) { \
            DYADIC_LOGGING("Dyadic Binning: using parallel stage page writers (%u heads)...\n", policy.write_heads); \
            dyadic_mt_bitmask_algo_parallel<PremixValuePol, PremixIndexPol, OutmixValuePol, OutmixIndexPol>( \
                io, \
                stage_pages, \
                stage_bits, \
                phase_a_rounds, \
                phase_b_rounds, \
                bin0_passes, \
                policy); \
            DYADIC_LOGGING("Dyadic Binning: parallel stage page writers complete.\n"); \
            const uint32_t write_heads = std::max<uint32_t>(1u, policy.write_heads); \
            const uint32_t queue_len = std::max<uint32_t>(1u, policy.write_head_queue_len); \
            TensorDesc inbox_desc{}; \
            inbox_desc.dtype = TensorDType::Bytes4; \
            inbox_desc.layout = TensorLayout::Dense; \
            inbox_desc.shape.dims = { write_heads, 2u, queue_len, 2u }; \
            AbstractTensor inbox_tensor = pool.acquire_tensor(inbox_desc, bins.backend()); \
            void* inbox_ptr_void = nullptr; \
            size_t inbox_bytes = 0; \
            if (mem->map(inbox_tensor.handle(), &inbox_ptr_void, &inbox_bytes)) { \
                mem->ensure_zeroed(inbox_tensor.handle(), inbox_desc, true, 0u, 0u); \
            } \
            DyadicWriteHeadInboxView inbox_view{}; \
            inbox_view.ptr = static_cast<uint32_t*>(inbox_ptr_void); \
            std::vector<uint32_t> write_idx(write_heads, 0u); \
            std::vector<uint32_t> read_idx(write_heads, 0u); \
            inbox_view.write_idx = write_idx.data(); \
            inbox_view.read_idx = read_idx.data(); \
            inbox_view.head_count = write_heads; \
            inbox_view.queue_len = queue_len; \
            uint32_t head_cursor = 0u; \
            std::atomic<bool> stop{false}; \
            stage_pages.start_recycler(&stop); \
            nodus::ThreadPool::Options wopt{}; \
            wopt.thread_count = write_heads; \
            wopt.start_immediately = true; \
            nodus::ThreadPool wpool(wopt); \
            auto wctxs = std::make_unique<DyadicStageWriterJobCtx<INDEX_T, VALUE_T, OutmixValuePol, OutmixIndexPol>[]>(write_heads); \
            auto wjobs = std::make_unique<nodus::ThreadPool::Job[]>(write_heads); \
            for (uint32_t w = 0; w < write_heads; ++w) { \
                wctxs[w].io = &io; \
                wctxs[w].stage_pages = &stage_pages; \
                wctxs[w].output = output; \
                wctxs[w].output_offset = nullptr; \
                wctxs[w].stage_bits = stage_bits; \
                wctxs[w].inbox_ptr = inbox_view.ptr; \
                wctxs[w].write_idx = inbox_view.write_idx; \
                wctxs[w].read_idx = inbox_view.read_idx; \
                wctxs[w].head_id = w; \
                wctxs[w].head_count = write_heads; \
                wctxs[w].queue_len = queue_len; \
                wctxs[w].stop = &stop; \
                wjobs[w].fn = &dyadic_stage_writer_job_fn<INDEX_T, VALUE_T, OutmixValuePol, OutmixIndexPol>; \
                wjobs[w].user = &wctxs[w]; \
            } \
            auto wbatch = wpool.submit_batch(wjobs.get(), write_heads); \
            DyadicQueueScannerJobCtx<INDEX_T, VALUE_T> scan_ctx{}; \
            scan_ctx.stage_pages = &stage_pages; \
            scan_ctx.inbox = inbox_view; \
            scan_ctx.head_cursor = &head_cursor; \
            scan_ctx.stop = &stop; \
            nodus::ThreadPool::Options sopt{}; \
            sopt.thread_count = 1u; \
            sopt.start_immediately = true; \
            nodus::ThreadPool spool(sopt); \
            nodus::ThreadPool::Job sjob{}; \
            sjob.fn = &dyadic_queue_scanner_job_fn<INDEX_T, VALUE_T>; \
            sjob.user = &scan_ctx; \
            auto sbatch = spool.submit_batch(&sjob, 1u); \
            DYADIC_LOGGING("Dyadic Binning: waiting for stage pages to drain...\n"); \
            if (sbatch) sbatch->wait(); \
            DYADIC_LOGGING("Dyadic Binning: stage pages drained, finalizing writes...\n"); \
            for (;;) { \
                bool queues_empty = true; \
                for (uint32_t bin = 0; bin < stage_pages.total_bins; ++bin) { \
                    std::atomic_ref<uint32_t> href(stage_pages.queue_head[bin]); \
                    std::atomic_ref<uint32_t> tref(stage_pages.queue_tail[bin]); \
                    const uint32_t h = href.load(std::memory_order_acquire); \
                    const uint32_t t = tref.load(std::memory_order_acquire); \
                    if (h != t) { \
                        queues_empty = false; \
                        break; \
                    } \
                } \
                if (queues_empty && dyadic_write_head_empty(inbox_view)) { \
                    stop.store(true, std::memory_order_release); \
                    break; \
                } \
            } \
            if (wbatch) wbatch->wait(); \
            stage_pages.stop_recycler(); \
            if (inbox_ptr_void) mem->unmap(inbox_tensor.handle()); \
        } else { \
            DYADIC_LOGGING("Dyadic Binning: using single-threaded stage page writer...\n"); \
            std::vector<uint8_t> duty_state(io.total_bins, 0u); \
            std::vector<uint8_t> phase_counter(io.total_bins, 0u); \
            for (uint32_t bin = 0; bin < io.total_bins; ++bin) { \
                const uint8_t state = dyadic_duty_pack(io.page_active[bin], io.page_inbox[bin], io.page_retain[bin], 0u); \
                duty_state[bin] = state; \
                dyadic_sync_pages_from_state(bin, state, io.page_active, io.page_inbox, io.page_retain); \
            } \
            DYADIC_LOGGING("Dyadic Binning: draining stage pages...\n"); \
            dyadic_run_single_thread_schedule_stage_pages<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>( \
                single_threaded_dyadic_schedule, \
                io, \
                stage_pages, \
                stage_bits, \
                duty_state.data(), \
                phase_counter.data(), \
                jobs, \
                TICK); \
            DYADIC_LOGGING("Dyadic Binning: stage pages drained.\n"); \
            const uint32_t write_heads = policy.write_heads < std::numeric_limits<uint32_t>::max() ? std::max<uint32_t>(1u, policy.write_heads) : 1u; \
            const uint32_t queue_len = policy.write_head_queue_len < std::numeric_limits<uint32_t>::max() ? std::max<uint32_t>(1u, policy.write_head_queue_len) : 1u; \
            DYADIC_LOGGING("Dyadic Binning: finalizing writes with single-threaded writer (%u heads)...\n", write_heads); \
            DYADIC_LOGGING("Dyadic Binning: queue-length=%u\n", queue_len); \
            TensorDesc inbox_desc{}; \
            inbox_desc.dtype = TensorDType::Bytes4; \
            inbox_desc.layout = TensorLayout::Dense; \
            inbox_desc.shape.dims = { write_heads, 2u, queue_len, 2u }; \
            AbstractTensor inbox_tensor = pool.acquire_tensor(inbox_desc, bins.backend()); \
            void* inbox_ptr_void = nullptr; \
            size_t inbox_bytes = 0; \
            if (mem->map(inbox_tensor.handle(), &inbox_ptr_void, &inbox_bytes)) { \
                mem->ensure_zeroed(inbox_tensor.handle(), inbox_desc, true, 0u, 0u); \
            } \
            DyadicWriteHeadInboxView inbox_view{}; \
            inbox_view.ptr = static_cast<uint32_t*>(inbox_ptr_void); \
            std::vector<uint32_t> write_idx(write_heads, 0u); \
            std::vector<uint32_t> read_idx(write_heads, 0u); \
            inbox_view.write_idx = write_idx.data(); \
            inbox_view.read_idx = read_idx.data(); \
            inbox_view.head_count = write_heads; \
            inbox_view.queue_len = queue_len; \
            uint32_t head_cursor = 0u; \
            std::atomic<bool> stop{false}; \
            stage_pages.start_recycler(&stop); \
            DyadicQueueScannerJobCtx<INDEX_T, VALUE_T> scan_ctx{}; \
            scan_ctx.stage_pages = &stage_pages; \
            scan_ctx.inbox = inbox_view; \
            scan_ctx.head_cursor = &head_cursor; \
            scan_ctx.stop = &stop; \
            dyadic_queue_scanner_tick<INDEX_T, VALUE_T>(scan_ctx); \
            dyadic_write_head_tick<INDEX_T, VALUE_T>(inbox_view, stage_pages, output, 0u); \
            DYADIC_LOGGING("Dyadic Binning: writes finalized.\n"); \
            stop.store(true, std::memory_order_release); \
            DYADIC_LOGGING("Dyadic Binning: stopping recycler...\n"); \
            stage_pages.stop_recycler(); \
            DYADIC_LOGGING("Dyadic Binning: unmapping inbox tensor...\n"); \
            if (inbox_ptr_void) mem->unmap(inbox_tensor.handle()); \
        } \
        if (jobs && job_count > 0u) { \
            if (dyadic_is_single_threaded(policy) || !run_sort_classifier_thread) { \
                DYADIC_LOGGING("Dyadic Binning: finalizing classifier job...\n"); \
                dyadic_sort_classifier_blocking<INDEX_T, VALUE_T, PremixValuePol, PremixIndexPol>(classifier_ctx); \
            } \
            classifier_stop.store(true, std::memory_order_release); \
            stage_pages.stop_sort_classifier_thread(); \
        } \
        if (local_job_active && local_job.ring_ptr && mem) { \
            DYADIC_LOGGING("Dyadic Binning: unmapping local job ring...\n"); \
            mem->unmap(local_job.ring_tensor); \
        } \
        DYADIC_LOGGING("Dyadic Binning: unmapping resources...\n"); \
        mem->unmap(bins.handle()); \
        mem->unmap(counters.handle()); \
        mem->unmap(page_active.handle()); \
        mem->unmap(page_retain.handle()); \
        mem->unmap(page_inbox.handle()); \
        if (abstract_tensor_handle_is_valid(stage_pages.queue_tensor)) { \
            DYADIC_LOGGING("Dyadic Binning: unmapping stage pages queue tensor...\n"); \
            mem->unmap(stage_pages.queue_tensor); \
        } \
        if (abstract_tensor_handle_is_valid(stage_pages.release_tensor)) { \
            DYADIC_LOGGING("Dyadic Binning: unmapping stage pages release tensor...\n"); \
            mem->unmap(stage_pages.release_tensor); \
        } \
        if (abstract_tensor_handle_is_valid(stage_pages.release_head_tensor)) { \
            DYADIC_LOGGING("Dyadic Binning: unmapping stage pages release head tensor...\n"); \
            mem->unmap(stage_pages.release_head_tensor); \
        } \
        if (abstract_tensor_handle_is_valid(stage_pages.release_tail_tensor)) { \
            DYADIC_LOGGING("Dyadic Binning: unmapping stage pages release tail tensor...\n"); \
            mem->unmap(stage_pages.release_tail_tensor); \
        } \
        if (abstract_tensor_handle_is_valid(io.ready_pages_tensor)) { \
            DYADIC_LOGGING("Dyadic Binning: unmapping ready pages tensor...\n"); \
            mem->unmap(io.ready_pages_tensor); \
        } \
        if (abstract_tensor_handle_is_valid(io.epoch_tensor)) { \
            DYADIC_LOGGING("Dyadic Binning: unmapping epoch tensor...\n"); \
            mem->unmap(io.epoch_tensor); \
        } \
        DYADIC_LOGGING("Dyadic Binning: complete.\n"); \
    } \
    template <typename PremixValuePol = policies::Add, \
              typename PremixIndexPol = policies::Overwrite, \
              typename OutmixValuePol = policies::Overwrite, \
              typename OutmixIndexPol = policies::Overwrite> \
    inline bool dyadic_mt_bitmask_algo(INDEX_T index_range, uint32_t index_count, INDEX_T* indices, VALUE_T* values, \
                                       uint32_t value_stride, \
                                       uint32_t stage_bits, uint32_t thread_count, \
                                       uint32_t phase_a_rounds, uint32_t phase_b_rounds, uint32_t bin0_passes, \
                                       AbstractTensor& output, uint32_t* output_offset, \
                                       AbstractTensorPool& pool, \
                                       AbstractTensor& bins, AbstractTensor& staging, AbstractTensor& counters, \
                                       AbstractTensor& page_active, AbstractTensor& page_retain, \
                                       AbstractTensor& page_inbox, \
                                       uint32_t stage_pages_max = 0, \
                                       bool output_as_stage_pool = true, \
                                       bool clear_output_as_stage_pool = true, \
                                       uint32_t index_stride = 0, \
                                       uint32_t index_channels = 1, \
                                       DyadicJobCtx* jobs = nullptr, \
                                       uint32_t job_count = 0, \
                                       bool run_sort_classifier_thread = true) { \
        if (!output.valid()) return false; \
        auto* mem = dynamic_cast<InMemoryBackend*>(output.backend()); \
        if (!mem) return false; \
        const TensorDesc& out_desc = output.desc(); \
        const size_t out_rank = out_desc.shape.dims.size(); \
        const uint32_t out_d0 = out_rank > 0 ? out_desc.shape.dims[0] : 0u; \
        const uint32_t out_d1 = out_rank > 1 ? out_desc.shape.dims[1] : 0u; \
        const uint32_t out_d2 = out_rank > 2 ? out_desc.shape.dims[2] : 0u; \
        DYADIC_TRACE_LOGF("[dyadic] mt_bitmask_tensor: out_id=%llu dtype=%d rank=%zu dims=%u,%u,%u stage_bits=%u threads=%u\n", \
            (unsigned long long)output.handle().id, (int)out_desc.dtype, out_rank, out_d0, out_d1, out_d2, stage_bits, thread_count); \
        void* out_ptr = nullptr; \
        size_t out_bytes = 0; \
        if (!mem->map(output.handle(), &out_ptr, &out_bytes)) return false; \
        DYADIC_TRACE_LOGF("[dyadic] mt_bitmask_tensor: out_ptr=%p out_bytes=%zu index_range=%llu index_count=%u stride=%u index_stride=%u index_channels=%u\n", \
            out_ptr, out_bytes, (unsigned long long)index_range, index_count, value_stride, index_stride, index_channels); \
        DYADIC_LOGGING("Dyadic MT Bitmask Algo Output Mapped: %zu bytes\n", out_bytes); \
        dyadic_mt_bitmask_algo<PremixValuePol, PremixIndexPol, OutmixValuePol, OutmixIndexPol>( \
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
            stage_pages_max, \
            output_as_stage_pool, \
            clear_output_as_stage_pool, \
            out_bytes, \
            index_stride, \
            index_channels, \
            jobs, \
            job_count, \
            run_sort_classifier_thread); \
        DYADIC_TRACE_LOGF("[dyadic] mt_bitmask_tensor: complete\n"); \
        mem->unmap(output.handle()); \
        return true; \
    } \
    template <typename PremixValuePol = policies::Add, \
              typename PremixIndexPol = policies::Overwrite, \
              typename OutmixValuePol = policies::Overwrite, \
              typename OutmixIndexPol = policies::Overwrite> \
    inline bool dyadic_binning_tensor(INDEX_T index_range, uint32_t index_count, INDEX_T* indices, VALUE_T* values, \
                                      uint32_t value_stride, \
                                      uint32_t stage_bits, uint32_t thread_count, \
                                      uint32_t phase_a_rounds, uint32_t phase_b_rounds, uint32_t bin0_passes, \
                                      AbstractTensor& output, AbstractTensorPool& pool, \
                                      uint32_t stage_pages_max = 0, \
                                      bool output_as_stage_pool = true, \
                                      bool clear_output_as_stage_pool = true, \
                                      uint32_t index_stride = 0, \
                                      uint32_t index_channels = 1) { \
        if (!output.valid()) return false; \
        TensorBackend* backend = output.backend(); \
        if (!backend) return false; \
        DYADIC_TRACE_LOGF("[dyadic] binning_tensor: out_id=%llu index_range=%llu index_count=%u value_stride=%u stage_bits=%u threads=%u phase_a=%u phase_b=%u bin0=%u\n", \
            (unsigned long long)output.handle().id, (unsigned long long)index_range, index_count, value_stride, stage_bits, thread_count, phase_a_rounds, phase_b_rounds, bin0_passes); \
        DYADIC_TRACE_LOGF("[dyadic] binning_tensor: output_as_stage_pool=%d clear_output=%d stage_pages_max=%u index_stride=%u index_channels=%u\n", \
            output_as_stage_pool ? 1 : 0, clear_output_as_stage_pool ? 1 : 0, stage_pages_max, index_stride, index_channels); \
        const uint32_t total_bins = dyadic_bin_count(index_range, stage_bits); \
        const uint32_t channel_count = std::max<uint32_t>(1u, index_channels); \
        const uint32_t value_prefix = channel_count - 1u; \
        const uint32_t packed_value_stride = value_stride + value_prefix; \
        TensorDesc desc_dyadic_bins{}; \
        desc_dyadic_bins.dtype = dyadic_uint_bytes_dtype<INDEX_T, VALUE_T>(); \
        desc_dyadic_bins.layout = TensorLayout::Dense; \
        desc_dyadic_bins.shape.dims = { (uint32_t)(total_bins * 3u), index_count, (uint32_t)(1u + packed_value_stride) }; \
        TensorDesc desc_staging{}; \
        desc_staging.dtype = dyadic_value_dtype<VALUE_T>(); \
        desc_staging.layout = TensorLayout::Dense; \
        desc_staging.shape.dims = { (uint32_t)((1u << stage_bits) * packed_value_stride) }; \
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
        DYADIC_TRACE_LOGF("[dyadic] binning_tensor: total_bins=%u packed_value_stride=%u index_count=%u\n", \
            total_bins, packed_value_stride, index_count); \
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
        DYADIC_TRACE_LOGF("[dyadic] binning_tensor: bins_id=%llu staging_id=%llu counters_id=%llu active_id=%llu retain_id=%llu inbox_id=%llu\n", \
            (unsigned long long)dyadic_bins_tensor.handle().id, (unsigned long long)staging_tensor.handle().id, \
            (unsigned long long)counters_tensor.handle().id, (unsigned long long)page_active_tensor.handle().id, \
            (unsigned long long)page_retain_tensor.handle().id, (unsigned long long)page_inbox_tensor.handle().id); \
        uint32_t output_offset = 0; \
        return dyadic_mt_bitmask_algo<PremixValuePol, PremixIndexPol, OutmixValuePol, OutmixIndexPol>( \
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
            stage_pages_max, \
            output_as_stage_pool, \
            clear_output_as_stage_pool, \
            index_stride, \
            index_channels); \
    } \
        inline void dyadic_binning(INDEX_T index_range, uint32_t index_count, INDEX_T* indices, VALUE_T* values, \
                                   uint32_t value_stride, \
                       float_t dense_threshold, float_t* tile_dense_ratios, uint32_t thread_count, \
                       uint32_t phase_a_rounds, uint32_t phase_b_rounds, uint32_t bin0_passes, \
                       VALUE_T* output, uint32_t* output_offset, \
                       bool output_as_stage_pool = true, \
                       bool clear_output_as_stage_pool = true, \
                       uint32_t index_stride = 0, \
                       uint32_t index_channels = 1) { \
        const uint32_t stage_bits = 8; \
        const uint32_t total_bins = dyadic_bin_count(index_range, stage_bits); \
        const uint32_t stage_bins = (uint32_t)(1u << stage_bits); \
        const uint32_t output_area = (uint32_t)index_range; \
        const bool should_use_dense = index_count >= output_area * dense_threshold; \
        const uint32_t channel_count = std::max<uint32_t>(1u, index_channels); \
        const uint32_t value_prefix = channel_count - 1u; \
        const uint32_t packed_value_stride = value_stride + value_prefix; \
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
        desc_dyadic_bins.shape.dims = { (uint32_t)(total_bins * 3u), index_count, (uint32_t)(1u + packed_value_stride) }; \
        TensorDesc desc_staging{}; \
        desc_staging.dtype = dyadic_value_dtype<VALUE_T>(); \
        desc_staging.layout = TensorLayout::Dense; \
        desc_staging.shape.dims = { (uint32_t)((1u << stage_bits) * packed_value_stride) }; \
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
        dyadic_mt_bitmask_algo<policies::Add, policies::Overwrite, policies::Overwrite, policies::Overwrite>( \
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
            page_inbox_tensor, \
            0u, \
            output_as_stage_pool, \
            clear_output_as_stage_pool, \
            index_stride, \
            index_channels); \
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
