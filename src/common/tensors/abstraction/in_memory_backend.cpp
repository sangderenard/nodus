#include "common/tensors/abstraction/in_memory_backend.h"

#include "common/tensors/abstraction/tensor_registry.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>

namespace nodus::tensors {

namespace {
constexpr uint32_t kMagic = 0x54454E53u; // "TENS"
constexpr size_t kAlignment = 64;
constexpr size_t kHeaderAlign = kAlignment;
constexpr size_t kBucketMin = 64;
constexpr size_t kBucketMax = 1u << 20; // 1 MiB
constexpr size_t kBucketCount = 15; // 64 -> 1MiB inclusive
constexpr size_t kSegmentSize = 1024;
constexpr size_t kDefaultSlabSize = 1u << 20; // 1 MiB

constexpr size_t align_up(size_t v, size_t alignment) {
    return (v + alignment - 1) & ~(alignment - 1);
}

inline size_t bucket_index_for_size(size_t bytes) {
    size_t v = bytes < kBucketMin ? kBucketMin : bytes;
    size_t b = kBucketMin;
    size_t idx = 0;
    while (b < v && b < kBucketMax) {
        b <<= 1;
        ++idx;
    }
    if (b < v || b > kBucketMax) return kBucketCount;
    return idx;
}

inline size_t bucket_size(size_t index) {
    return kBucketMin << index;
}

struct BlockHeader {
    uint32_t magic = kMagic;
    uint16_t bucket = 0;
    uint16_t flags = 0;
    uint64_t payload_size = 0;
    BlockHeader* next = nullptr;
};

constexpr size_t kHeaderSize = align_up(sizeof(BlockHeader), kHeaderAlign);

struct Slab {
    uint8_t* data = nullptr;
    size_t size = 0;
    std::atomic<size_t> offset{0};
    std::atomic<Slab*> next{nullptr};
};

struct TensorRecord {
    TensorDesc desc{};
    void* data = nullptr;
    size_t bytes = 0;
    std::atomic<bool> alive{false};
};

struct Segment {
    uint64_t base = 0;
    std::atomic<TensorRecord*> slots[kSegmentSize];
    std::atomic<Segment*> next{nullptr};
    explicit Segment(uint64_t base_id) : base(base_id) {
        for (size_t i = 0; i < kSegmentSize; ++i) slots[i].store(nullptr, std::memory_order_relaxed);
    }
};

std::atomic<uint64_t> g_next_id{1};
std::atomic<Segment*> g_segments{nullptr};
std::atomic<Slab*> g_slabs{nullptr};
std::atomic<BlockHeader*> g_free_lists[kBucketCount];

Segment* find_segment(uint64_t base) {
    Segment* seg = g_segments.load(std::memory_order_acquire);
    while (seg) {
        if (seg->base == base) return seg;
        seg = seg->next.load(std::memory_order_acquire);
    }
    return nullptr;
}

Segment* get_or_create_segment(uint64_t base) {
    if (Segment* existing = find_segment(base)) return existing;
    Segment* seg = new Segment(base);
    Segment* head = g_segments.load(std::memory_order_acquire);
    do {
        seg->next.store(head, std::memory_order_relaxed);
    } while (!g_segments.compare_exchange_weak(head, seg, std::memory_order_release, std::memory_order_acquire));
    return seg;
}

TensorRecord* find_record(uint64_t id) {
    uint64_t base = (id / kSegmentSize) * kSegmentSize;
    size_t slot = static_cast<size_t>(id - base);
    Segment* seg = g_segments.load(std::memory_order_acquire);
    TensorRecord* found = nullptr;
    while (seg) {
        if (seg->base == base) {
            TensorRecord* rec = seg->slots[slot].load(std::memory_order_acquire);
            if (rec) found = rec;
        }
        seg = seg->next.load(std::memory_order_acquire);
    }
    return found;
}

bool try_pop_free(size_t bucket, BlockHeader** out) {
    BlockHeader* head = g_free_lists[bucket].load(std::memory_order_acquire);
    while (head) {
        BlockHeader* next = head->next;
        if (g_free_lists[bucket].compare_exchange_weak(head, next, std::memory_order_release, std::memory_order_acquire)) {
            *out = head;
            return true;
        }
    }
    return false;
}

void push_free(size_t bucket, BlockHeader* block) {
    BlockHeader* head = g_free_lists[bucket].load(std::memory_order_acquire);
    do {
        block->next = head;
    } while (!g_free_lists[bucket].compare_exchange_weak(head, block, std::memory_order_release, std::memory_order_acquire));
}

bool try_alloc_from_slab(Slab* slab, size_t bytes, void** out) {
    size_t off = slab->offset.load(std::memory_order_relaxed);
    while (true) {
        size_t aligned = align_up(off, kAlignment);
        size_t next = aligned + bytes;
        if (next > slab->size) return false;
        if (slab->offset.compare_exchange_weak(off, next, std::memory_order_release, std::memory_order_relaxed)) {
            *out = slab->data + aligned;
            return true;
        }
    }
}

Slab* alloc_slab(size_t bytes) {
    size_t size = bytes < kDefaultSlabSize ? kDefaultSlabSize : align_up(bytes, kAlignment);
    uint8_t* data = new (std::nothrow) uint8_t[size];
    if (!data) return nullptr;
    Slab* slab = new (std::nothrow) Slab{};
    if (!slab) {
        delete [] data;
        return nullptr;
    }
    slab->data = data;
    slab->size = size;
    slab->offset.store(0, std::memory_order_relaxed);
    slab->next.store(nullptr, std::memory_order_relaxed);
    return slab;
}

void* pool_alloc(size_t payload_size, size_t* out_bucket) {
    size_t bucket = bucket_index_for_size(payload_size);
    size_t alloc_payload = payload_size;
    if (bucket < kBucketCount) {
        BlockHeader* block = nullptr;
        if (try_pop_free(bucket, &block)) {
            block->magic = kMagic;
            block->bucket = static_cast<uint16_t>(bucket);
            block->payload_size = payload_size;
            *out_bucket = bucket;
            return reinterpret_cast<uint8_t*>(block) + kHeaderSize;
        }
        alloc_payload = bucket_size(bucket);
    }

    size_t total = kHeaderSize + alloc_payload;
    Slab* slab = g_slabs.load(std::memory_order_acquire);
    while (slab) {
        void* ptr = nullptr;
        if (try_alloc_from_slab(slab, total, &ptr)) {
            auto* header = reinterpret_cast<BlockHeader*>(ptr);
            header->magic = kMagic;
            header->bucket = bucket < kBucketCount ? static_cast<uint16_t>(bucket) : 0xFFFFu;
            header->payload_size = payload_size;
            *out_bucket = header->bucket;
            return reinterpret_cast<uint8_t*>(header) + kHeaderSize;
        }
        slab = slab->next.load(std::memory_order_acquire);
    }

    Slab* fresh = alloc_slab(total);
    if (!fresh) return nullptr;
    Slab* head = g_slabs.load(std::memory_order_acquire);
    do {
        fresh->next.store(head, std::memory_order_relaxed);
    } while (!g_slabs.compare_exchange_weak(head, fresh, std::memory_order_release, std::memory_order_acquire));

    void* ptr = nullptr;
    if (!try_alloc_from_slab(fresh, total, &ptr)) return nullptr;
    auto* header = reinterpret_cast<BlockHeader*>(ptr);
    header->magic = kMagic;
    header->bucket = bucket < kBucketCount ? static_cast<uint16_t>(bucket) : 0xFFFFu;
    header->payload_size = payload_size;
    *out_bucket = header->bucket;
    return reinterpret_cast<uint8_t*>(header) + kHeaderSize;
}

void pool_free(void* data) {
    if (!data) return;
    auto* header = reinterpret_cast<BlockHeader*>(reinterpret_cast<uint8_t*>(data) - kHeaderSize);
    if (header->magic != kMagic) return;
    size_t bucket = header->bucket;
    if (bucket < kBucketCount) {
        push_free(bucket, header);
    }
}
} // namespace

const char* InMemoryBackend::name() const {
    return "in_memory";
}

AbstractTensorHandle InMemoryBackend::create(const TensorDesc& desc) {
    AbstractTensorHandle handle{};
    size_t bytes = static_cast<size_t>(desc.shape.element_count()) * tensor_dtype_size_bytes(desc.dtype);
    size_t bucket = 0;
    void* data = pool_alloc(bytes, &bucket);
    if (!data) return handle;

    handle.id = g_next_id.fetch_add(1, std::memory_order_relaxed);
    uint64_t base = (handle.id / kSegmentSize) * kSegmentSize;
    size_t slot = static_cast<size_t>(handle.id - base);
    Segment* seg = get_or_create_segment(base);

    TensorRecord* rec = new (std::nothrow) TensorRecord{};
    if (!rec) {
        pool_free(data);
        return AbstractTensorHandle{};
    }
    rec->desc = desc;
    rec->data = data;
    rec->bytes = bytes;
    rec->alive.store(true, std::memory_order_release);

    TensorRecord* expected = nullptr;
    if (!seg->slots[slot].compare_exchange_strong(expected, rec, std::memory_order_release, std::memory_order_acquire)) {
        delete rec;
        pool_free(data);
        return AbstractTensorHandle{};
    }
    return handle;
}

void InMemoryBackend::destroy(AbstractTensorHandle handle) {
    if (!abstract_tensor_handle_is_valid(handle)) return;
    TensorRecord* rec = find_record(handle.id);
    if (!rec) return;
    bool expected = true;
    if (!rec->alive.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) return;
    pool_free(rec->data);
}

bool InMemoryBackend::describe(AbstractTensorHandle handle, TensorDesc* out) const {
    if (!out || !abstract_tensor_handle_is_valid(handle)) return false;
    TensorRecord* rec = find_record(handle.id);
    if (!rec || !rec->alive.load(std::memory_order_acquire)) return false;
    *out = rec->desc;
    return true;
}

bool InMemoryBackend::get_allocation_info(AbstractTensorHandle handle, AllocationInfo* out) const {
    if (!out || !abstract_tensor_handle_is_valid(handle)) return false;
    TensorRecord* rec = find_record(handle.id);
    if (!rec) return false;
    out->data = rec->data;
    out->bytes = rec->bytes;
    out->alive = rec->alive.load(std::memory_order_acquire);
    out->bucket = 0xFFFFu;
    if (rec->data) {
        auto* header = reinterpret_cast<BlockHeader*>(reinterpret_cast<uint8_t*>(rec->data) - kHeaderSize);
        if (header->magic == kMagic) {
            out->bucket = header->bucket;
        }
    }
    return true;
}

bool InMemoryBackend::map(AbstractTensorHandle handle, void** out_data, size_t* out_bytes) const {
    if (!out_data || !out_bytes || !abstract_tensor_handle_is_valid(handle)) return false;
    TensorRecord* rec = find_record(handle.id);
    if (!rec || !rec->alive.load(std::memory_order_acquire)) return false;
    *out_data = rec->data;
    *out_bytes = rec->bytes;
    return true;
}

void InMemoryBackend::unmap(AbstractTensorHandle /*handle*/) const {
    // No-op for in-memory backend; data is always host-accessible.
}

InMemoryBackend& in_memory_backend_singleton() {
    static InMemoryBackend backend;
    return backend;
}

void register_in_memory_backend(bool make_default) {
    register_backend(&in_memory_backend_singleton(), make_default);
}

} // namespace nodus::tensors
