#include "common/tensors/abstraction/in_memory_backend.h"

#include "common/tensors/abstraction/tensor_registry.h"
#include "common/tensors/abstraction/tensor_math.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <cstring>
#include <cstdlib>
#include <mutex>

namespace nodus::tensors {

namespace {
constexpr size_t kSegmentSize = 1024;
constexpr size_t kAlignment = 64;

constexpr size_t align_up(size_t v, size_t alignment) {
    return (v + alignment - 1) & ~(alignment - 1);
}

struct TensorRecord {
    TensorDesc desc{};
    void* data = nullptr;
    uint64_t offset = 0;
    size_t bytes = 0;
    // View/aliasing support:
    // - Records with owns_lease=true are the allocation owners (arena lease holders).
    // - Views set owns_lease=false and point at the same lease via lease_owner_id.
    // - lease_refs is stored on the owner record and counts all live records
    //   (owner + all views) that reference the lease.
    uint64_t lease_owner_id = 0;
    bool owns_lease = false;
    std::atomic<uint32_t> lease_refs{0};
    std::atomic<bool> alive{false};
    std::atomic<uint64_t> next_free_id{0};
};

// A single contiguous arena from which all tensor payloads are leased.
//
// Key properties:
// - One contiguous malloc-backed region.
// - Leases are best-fit from the free list and aligned.
// - Lease returns are coalesced by address.
// - A single mutex guards the lease/free-span table so multithreaded callers
//   cannot corrupt allocator state. No other allocator-side locking exists.
// - No heap allocations occur in arena_alloc/arena_free after init; span nodes
//   are served from a preallocated pool.
//
// Configure size via env var:
//   NODUS_INMEM_ARENA_MB (default: 1024)
// Configure max span nodes via env var:
//   NODUS_INMEM_ARENA_SPANS (default: 262144)
struct ArenaSpanNode {
    uint64_t offset = 0;
    uint64_t size = 0;
    int32_t next = -1;
};

struct Arena {
    void* malloc_base = nullptr;
    uint8_t* base = nullptr;
    uint64_t reserve_bytes = 0;
    bool initialized = false;

    std::mutex lease_mu;
    ArenaSpanNode* nodes = nullptr;
    uint32_t node_capacity = 0;
    int32_t free_node_head = -1;
    int32_t free_list_head = -1; // sorted by offset

    // Debug/telemetry counters (guarded by lease_mu).
    uint64_t active_leases = 0;
    uint64_t active_leased_bytes = 0;
    uint64_t peak_active_leased_bytes = 0;
    uint64_t alloc_calls = 0;
    uint64_t free_calls = 0;
    uint64_t alloc_fail_oom = 0;
    uint64_t free_drop_no_nodes = 0;
};

static Arena g_arena;

static uint64_t getenv_u64_mb(const char* name, uint64_t default_mb) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_mb;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(v, &end, 10);
    if (end == v) return default_mb;
    if (parsed == 0) return default_mb;
    return static_cast<uint64_t>(parsed);
}

static uint64_t getenv_u64(const char* name, uint64_t default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(v, &end, 10);
    if (end == v) return default_value;
    return static_cast<uint64_t>(parsed);
}

static uint64_t align_up_u64(uint64_t v, uint64_t alignment) {
    return (v + alignment - 1) & ~(alignment - 1);
}

static bool arena_ensure_initialized() {
    if (g_arena.initialized) return g_arena.base != nullptr;

    // This is the only place we allow a one-time heap allocation besides the arena itself.
    // Guard init with the same mutex that guards the lease table.
    std::lock_guard<std::mutex> lock(g_arena.lease_mu);
    if (g_arena.initialized) return g_arena.base != nullptr;

    const uint64_t mb = getenv_u64_mb("NODUS_INMEM_ARENA_MB", 1024);
    const uint64_t reserve = mb * 1024ull * 1024ull;
    g_arena.reserve_bytes = reserve;

    const uint64_t span_nodes = getenv_u64("NODUS_INMEM_ARENA_SPANS", 262144);
    if (span_nodes == 0 || span_nodes > 50'000'000ull) {
        g_arena.initialized = true;
        return false;
    }

    // One contiguous region. Keep the original pointer so we could free later if needed.
    // Align the usable base pointer to kAlignment.
    const uint64_t alloc_bytes = reserve + static_cast<uint64_t>(kAlignment);
    void* raw = std::malloc(static_cast<size_t>(alloc_bytes));
    if (!raw) {
        g_arena.initialized = true;
        return false;
    }
    g_arena.malloc_base = raw;
    uintptr_t p = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned = (p + (kAlignment - 1)) & ~(static_cast<uintptr_t>(kAlignment - 1));
    g_arena.base = reinterpret_cast<uint8_t*>(aligned);

    // Preallocate span-node pool (one-time).
    g_arena.nodes = static_cast<ArenaSpanNode*>(std::calloc(static_cast<size_t>(span_nodes), sizeof(ArenaSpanNode)));
    if (!g_arena.nodes) {
        std::free(g_arena.malloc_base);
        g_arena.malloc_base = nullptr;
        g_arena.base = nullptr;
        g_arena.initialized = true;
        return false;
    }
    g_arena.node_capacity = static_cast<uint32_t>(span_nodes);
    g_arena.free_node_head = 0;
    for (uint32_t i = 0; i < g_arena.node_capacity; ++i) {
        g_arena.nodes[i].next = (i + 1 < g_arena.node_capacity) ? static_cast<int32_t>(i + 1) : -1;
    }

    // Initialize free list with one span covering the entire arena.
    const int32_t root = g_arena.free_node_head;
    g_arena.free_node_head = g_arena.nodes[root].next;
    g_arena.nodes[root].offset = 0;
    g_arena.nodes[root].size = reserve;
    g_arena.nodes[root].next = -1;
    g_arena.free_list_head = root;

    g_arena.initialized = true;
    return true;
}

static bool arena_ensure_committed(uint64_t end_offset) {
    if (!arena_ensure_initialized()) return false;
    return end_offset <= g_arena.reserve_bytes;
}

static int32_t arena_pop_node() {
    const int32_t idx = g_arena.free_node_head;
    if (idx < 0) return -1;
    g_arena.free_node_head = g_arena.nodes[idx].next;
    g_arena.nodes[idx].next = -1;
    g_arena.nodes[idx].offset = 0;
    g_arena.nodes[idx].size = 0;
    return idx;
}

static void arena_push_node(int32_t idx) {
    if (idx < 0) return;
    g_arena.nodes[idx].next = g_arena.free_node_head;
    g_arena.free_node_head = idx;
}

struct ArenaStatsSnapshot {
    uint64_t reserve_bytes = 0;
    uint64_t active_leases = 0;
    uint64_t active_leased_bytes = 0;
    uint64_t peak_active_leased_bytes = 0;
    uint64_t alloc_calls = 0;
    uint64_t free_calls = 0;
    uint64_t alloc_fail_oom = 0;
    uint64_t free_drop_no_nodes = 0;
    uint32_t span_nodes_capacity = 0;
    uint32_t span_nodes_free = 0;
    uint64_t free_spans = 0;
    uint64_t free_bytes = 0;
    uint64_t largest_free_span = 0;
};

static ArenaStatsSnapshot arena_stats_locked() {
    ArenaStatsSnapshot s;
    s.reserve_bytes = g_arena.reserve_bytes;
    s.active_leases = g_arena.active_leases;
    s.active_leased_bytes = g_arena.active_leased_bytes;
    s.peak_active_leased_bytes = g_arena.peak_active_leased_bytes;
    s.alloc_calls = g_arena.alloc_calls;
    s.free_calls = g_arena.free_calls;
    s.alloc_fail_oom = g_arena.alloc_fail_oom;
    s.free_drop_no_nodes = g_arena.free_drop_no_nodes;
    s.span_nodes_capacity = g_arena.node_capacity;

    uint32_t free_nodes = 0;
    for (int32_t idx = g_arena.free_node_head; idx >= 0; idx = g_arena.nodes[idx].next) {
        ++free_nodes;
    }
    s.span_nodes_free = free_nodes;

    for (int32_t cur = g_arena.free_list_head; cur >= 0; cur = g_arena.nodes[cur].next) {
        ++s.free_spans;
        s.free_bytes += g_arena.nodes[cur].size;
        s.largest_free_span = std::max<uint64_t>(s.largest_free_span, g_arena.nodes[cur].size);
    }
    return s;
}

static bool arena_alloc(uint64_t bytes, uint64_t* out_offset, uint64_t* out_capacity) {
    if (!out_offset || !out_capacity) return false;
    if (!arena_ensure_initialized()) return false;
    if (bytes == 0) bytes = 1;

    const uint64_t need = align_up_u64(bytes, kAlignment);

    std::lock_guard<std::mutex> lock(g_arena.lease_mu);
    g_arena.alloc_calls++;

    // Best-fit search: scan all free spans and pick the smallest that satisfies need.
    int32_t best = -1;
    uint64_t best_size = 0;
    int32_t best_prev = -1;
    int32_t prev = -1;
    for (int32_t cur = g_arena.free_list_head; cur >= 0; prev = cur, cur = g_arena.nodes[cur].next) {
        const uint64_t sz = g_arena.nodes[cur].size;
        if (sz < need) continue;
        if (best < 0 || sz < best_size) {
            best = cur;
            best_size = sz;
            best_prev = prev;
            if (sz == need) break;
        }
    }
    if (best < 0) {
        g_arena.alloc_fail_oom++;
        return false;
    }

    const uint64_t span_offset = g_arena.nodes[best].offset;
    const uint64_t span_size = g_arena.nodes[best].size;

    // Remove chosen node from free list.
    if (best_prev < 0) {
        g_arena.free_list_head = g_arena.nodes[best].next;
    } else {
        g_arena.nodes[best_prev].next = g_arena.nodes[best].next;
    }
    g_arena.nodes[best].next = -1;

    const uint64_t alloc_offset = span_offset; // free list always aligned
    const uint64_t remaining = span_size - need;
    if (remaining > 0) {
        // Reuse the existing node for the remainder span.
        g_arena.nodes[best].offset = alloc_offset + need;
        g_arena.nodes[best].size = remaining;

        // Insert remainder back into free list at the correct position (offset order).
        int32_t insert_prev = -1;
        int32_t insert_cur = g_arena.free_list_head;
        while (insert_cur >= 0 && g_arena.nodes[insert_cur].offset < g_arena.nodes[best].offset) {
            insert_prev = insert_cur;
            insert_cur = g_arena.nodes[insert_cur].next;
        }
        if (insert_prev < 0) {
            g_arena.nodes[best].next = g_arena.free_list_head;
            g_arena.free_list_head = best;
        } else {
            g_arena.nodes[best].next = g_arena.nodes[insert_prev].next;
            g_arena.nodes[insert_prev].next = best;
        }
    } else {
        // Exact fit: return node to node pool.
        arena_push_node(best);
    }

    const uint64_t end = alloc_offset + need;
    if (end > g_arena.reserve_bytes) {
        return false;
    }
    if (!arena_ensure_committed(end)) {
        return false;
    }

    g_arena.active_leases++;
    g_arena.active_leased_bytes += need;
    g_arena.peak_active_leased_bytes = std::max<uint64_t>(g_arena.peak_active_leased_bytes, g_arena.active_leased_bytes);
    *out_offset = alloc_offset;
    *out_capacity = need;
    return true;
}

static void arena_free(uint64_t offset, uint64_t size) {
    if (size == 0) return;
    if (!arena_ensure_initialized()) return;
    const uint64_t sz = align_up_u64(size, kAlignment);

    std::lock_guard<std::mutex> lock(g_arena.lease_mu);
    g_arena.free_calls++;
    if (g_arena.active_leases > 0) g_arena.active_leases--;
    if (g_arena.active_leased_bytes >= sz) g_arena.active_leased_bytes -= sz;

    // Insert in offset order, then coalesce with neighbors.
    int32_t node = arena_pop_node();
    if (node < 0) {
        // No metadata capacity; drop the free (out of span nodes).
        g_arena.free_drop_no_nodes++;
        return;
    }
    g_arena.nodes[node].offset = offset;
    g_arena.nodes[node].size = sz;

    int32_t prev = -1;
    int32_t cur = g_arena.free_list_head;
    while (cur >= 0 && g_arena.nodes[cur].offset < offset) {
        prev = cur;
        cur = g_arena.nodes[cur].next;
    }

    // Link in.
    if (prev < 0) {
        g_arena.nodes[node].next = g_arena.free_list_head;
        g_arena.free_list_head = node;
    } else {
        g_arena.nodes[node].next = g_arena.nodes[prev].next;
        g_arena.nodes[prev].next = node;
    }

    // Coalesce with next.
    int32_t next = g_arena.nodes[node].next;
    if (next >= 0 && g_arena.nodes[node].offset + g_arena.nodes[node].size == g_arena.nodes[next].offset) {
        g_arena.nodes[node].size += g_arena.nodes[next].size;
        g_arena.nodes[node].next = g_arena.nodes[next].next;
        arena_push_node(next);
    }

    // Coalesce with prev.
    if (prev >= 0 && g_arena.nodes[prev].offset + g_arena.nodes[prev].size == g_arena.nodes[node].offset) {
        g_arena.nodes[prev].size += g_arena.nodes[node].size;
        g_arena.nodes[prev].next = g_arena.nodes[node].next;
        arena_push_node(node);
    }
}

static bool compute_dense_strides(const TensorDesc& desc, std::vector<uint64_t>& out) {
    const auto& dims = desc.shape.dims;
    if (dims.empty()) return false;
    out.resize(dims.size());
    uint64_t stride = 1;
    for (size_t i = dims.size(); i-- > 0;) {
        out[i] = stride;
        stride *= static_cast<uint64_t>(dims[i]);
    }
    return true;
}

struct IndexSpan {
    bool is_int = false;
    int64_t index = 0;
    TensorSlice slice{};
};

static bool normalize_index(const TensorIndex& in,
                            int64_t dim,
                            IndexSpan& out) {
    if (std::holds_alternative<int64_t>(in)) {
        int64_t idx = std::get<int64_t>(in);
        if (idx < 0) idx += dim;
        if (idx < 0 || idx >= dim) return false;
        out.is_int = true;
        out.index = idx;
        return true;
    }
    if (std::holds_alternative<TensorSlice>(in)) {
        TensorSlice s = std::get<TensorSlice>(in);
        if (s.is_all) {
            s.start = 0;
            s.stop = dim;
            s.step = 1;
        }
        if (s.step == 0) return false;
        if (s.step < 0) return false;
        int64_t start = s.start;
        int64_t stop = s.stop;
        if (start < 0) start += dim;
        if (stop < 0) stop += dim;
        if (start < 0) start = 0;
        if (stop > dim) stop = dim;
        if (stop < 0) stop = 0;
        s.start = start;
        s.stop = stop;
        out.is_int = false;
        out.slice = s;
        return true;
    }
    // Tensor-based indexing not supported yet.
    return false;
}

static uint64_t slice_len(const TensorSlice& s) {
    if (s.step <= 0) return 0;
    if (s.stop <= s.start) return 0;
    const int64_t span = s.stop - s.start;
    return static_cast<uint64_t>((span + s.step - 1) / s.step);
}

static bool build_index_plan(const TensorDesc& desc,
                             const TensorIndexSpec& index,
                             std::vector<IndexSpan>& spans,
                             TensorDesc& out_desc) {
    const auto& dims = desc.shape.dims;
    if (dims.empty()) return false;

    spans.clear();
    spans.reserve(dims.size());

    size_t provided = index.dims.size();
    if (provided > dims.size()) return false;

    for (size_t i = 0; i < dims.size(); ++i) {
        TensorIndex idx = (i < provided) ? index.dims[i] : TensorIndex(TensorSlice::all());
        IndexSpan span{};
        if (!normalize_index(idx, static_cast<int64_t>(dims[i]), span)) return false;
        spans.push_back(span);
    }

    // Out desc will be finalized by the caller (view strides vs. materialized dense).
    out_desc = desc;
    out_desc.shape.dims.clear();
    for (size_t i = 0; i < spans.size(); ++i) {
        if (spans[i].is_int) continue;
        out_desc.shape.dims.push_back(static_cast<uint32_t>(slice_len(spans[i].slice)));
    }
    out_desc.layout = TensorLayout::Dense;
    out_desc.strides.elems.clear();
    return true;
}

static bool compute_strides_for_desc(const TensorDesc& desc, std::vector<uint64_t>& out) {
    if (desc.layout == TensorLayout::Strided && desc.strides.elems.size() == desc.shape.dims.size()) {
        out = desc.strides.elems;
        return true;
    }
    return compute_dense_strides(desc, out);
}

static bool compute_view_desc_and_offset(const TensorDesc& base_desc,
                                        const std::vector<IndexSpan>& spans,
                                        TensorDesc& out_desc,
                                        uint64_t& out_elem_offset) {
    out_elem_offset = 0;
    std::vector<uint64_t> base_strides;
    if (!compute_strides_for_desc(base_desc, base_strides)) return false;

    out_desc = base_desc;
    out_desc.shape.dims.clear();
    out_desc.strides.elems.clear();
    out_desc.slice = {};

    for (size_t d = 0; d < spans.size(); ++d) {
        const auto& span = spans[d];
        if (span.is_int) {
            out_elem_offset += static_cast<uint64_t>(span.index) * base_strides[d];
            continue;
        }
        out_elem_offset += static_cast<uint64_t>(span.slice.start) * base_strides[d];
        out_desc.shape.dims.push_back(static_cast<uint32_t>(slice_len(span.slice)));
        out_desc.strides.elems.push_back(base_strides[d] * static_cast<uint64_t>(span.slice.step));
    }

    if (out_desc.shape.dims.empty()) {
        // Scalar view.
        out_desc.shape.dims.push_back(1u);
        out_desc.layout = TensorLayout::Dense;
        out_desc.strides.elems.clear();
    } else {
        out_desc.layout = TensorLayout::Strided;
        if (out_desc.strides.elems.size() != out_desc.shape.dims.size()) return false;
        std::vector<uint64_t> dense_strides;
        if (compute_dense_strides(out_desc, dense_strides) &&
            dense_strides == out_desc.strides.elems) {
            out_desc.layout = TensorLayout::Dense;
            out_desc.strides.elems.clear();
        }
    }

    out_desc.slice.valid = true;
    out_desc.slice.base_shape = base_desc.shape.dims;
    out_desc.slice.start.assign(spans.size(), 0);
    out_desc.slice.step.assign(spans.size(), 1);
    out_desc.slice.is_int.assign(spans.size(), 0);
    for (size_t d = 0; d < spans.size(); ++d) {
        const auto& span = spans[d];
        if (span.is_int) {
            out_desc.slice.start[d] = span.index;
            out_desc.slice.step[d] = 0;
            out_desc.slice.is_int[d] = 1;
        } else {
            out_desc.slice.start[d] = span.slice.start;
            out_desc.slice.step[d] = span.slice.step;
            out_desc.slice.is_int[d] = 0;
        }
    }

    return true;
}

static bool check_record_alive(const TensorRecord* rec) {
    return rec && rec->alive.load(std::memory_order_acquire);
}

struct Segment {
    uint64_t base = 0;
    std::atomic<TensorRecord*> slots[kSegmentSize];
    std::atomic<Segment*> next{nullptr};
    explicit Segment(uint64_t base_id) : base(base_id) {
        for (size_t i = 0; i < kSegmentSize; ++i) slots[i].store(nullptr, std::memory_order_relaxed);
    }
};

std::atomic<uint64_t> g_next_id{1};
std::atomic<uint64_t> g_free_ids{0};
std::atomic<Segment*> g_segments{nullptr};

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

bool try_pop_free_id(uint64_t* out_id) {
    if (!out_id) return false;
    uint64_t head = g_free_ids.load(std::memory_order_acquire);
    while (head != 0) {
        TensorRecord* rec = find_record(head);
        if (!rec) {
            // Corrupt free list; drop it.
            if (g_free_ids.compare_exchange_weak(head, 0, std::memory_order_release, std::memory_order_acquire)) {
                return false;
            }
            continue;
        }
        const uint64_t next = rec->next_free_id.load(std::memory_order_relaxed);
        if (g_free_ids.compare_exchange_weak(head, next, std::memory_order_release, std::memory_order_acquire)) {
            rec->next_free_id.store(0, std::memory_order_relaxed);
            *out_id = head;
            return true;
        }
    }
    return false;
}

void push_free_id(uint64_t id) {
    if (id == 0) return;
    TensorRecord* rec = find_record(id);
    if (!rec) return;
    uint64_t head = g_free_ids.load(std::memory_order_acquire);
    do {
        rec->next_free_id.store(head, std::memory_order_relaxed);
    } while (!g_free_ids.compare_exchange_weak(head, id, std::memory_order_release, std::memory_order_acquire));
}

static AbstractTensorHandle create_record_for_view(const TensorDesc& desc,
                                                   void* data,
                                                   size_t bytes,
                                                   uint64_t lease_owner_id) {
    AbstractTensorHandle handle{};

    uint64_t reuse_id = 0;
    if (try_pop_free_id(&reuse_id)) {
        TensorRecord* rec = find_record(reuse_id);
        if (rec) {
            rec->desc = desc;
            rec->data = data;
            rec->offset = 0;
            rec->bytes = bytes;
            rec->lease_owner_id = lease_owner_id;
            rec->owns_lease = false;
            rec->lease_refs.store(0, std::memory_order_relaxed);
            rec->alive.store(true, std::memory_order_release);
            handle.id = reuse_id;
            return handle;
        }
        // If reuse failed, fall through to allocating a fresh id.
    }

    handle.id = g_next_id.fetch_add(1, std::memory_order_relaxed);
    uint64_t base = (handle.id / kSegmentSize) * kSegmentSize;
    size_t slot = static_cast<size_t>(handle.id - base);
    Segment* seg = get_or_create_segment(base);

    TensorRecord* rec = new (std::nothrow) TensorRecord{};
    if (!rec) return AbstractTensorHandle{};
    rec->desc = desc;
    rec->data = data;
    rec->offset = 0;
    rec->bytes = bytes;
    rec->lease_owner_id = lease_owner_id;
    rec->owns_lease = false;
    rec->lease_refs.store(0, std::memory_order_relaxed);
    rec->alive.store(true, std::memory_order_release);

    TensorRecord* expected = nullptr;
    if (!seg->slots[slot].compare_exchange_strong(expected, rec, std::memory_order_release, std::memory_order_acquire)) {
        delete rec;
        return AbstractTensorHandle{};
    }
    return handle;
}

static void maybe_free_owner_lease(uint64_t owner_id) {
    if (owner_id == 0) return;
    TensorRecord* owner = find_record(owner_id);
    if (!owner) return;
    // Only the owner record ever frees the arena lease.
    if (!owner->owns_lease) return;
    // Ensure we only free once.
    bool expected_alive = true;
    if (!owner->alive.compare_exchange_strong(expected_alive, false, std::memory_order_acq_rel)) {
        return;
    }

    arena_free(owner->offset, static_cast<uint64_t>(owner->bytes));
    owner->data = nullptr;
    owner->offset = 0;
    owner->bytes = 0;
    owner->lease_owner_id = 0;
    owner->owns_lease = false;
    owner->lease_refs.store(0, std::memory_order_relaxed);
    push_free_id(owner_id);
}
} // namespace

const char* InMemoryBackend::name() const {
    return "in_memory";
}

AbstractTensorHandle InMemoryBackend::create(const TensorDesc& desc) {
    AbstractTensorHandle handle{};
    const size_t bytes_req = static_cast<size_t>(desc.shape.element_count()) * tensor_dtype_size_bytes(desc.dtype);
    uint64_t offset = 0;
    uint64_t cap = 0;
    if (!arena_alloc(static_cast<uint64_t>(bytes_req), &offset, &cap)) return handle;
    void* data = g_arena.base + offset;

    uint64_t reuse_id = 0;
    if (try_pop_free_id(&reuse_id)) {
        TensorRecord* rec = find_record(reuse_id);
        if (rec) {
            // Reinitialize an existing record.
            rec->desc = desc;
            rec->data = data;
            rec->offset = offset;
            rec->bytes = static_cast<size_t>(cap);
            rec->lease_owner_id = reuse_id;
            rec->owns_lease = true;
            rec->lease_refs.store(1, std::memory_order_release);
            rec->alive.store(true, std::memory_order_release);
            handle.id = reuse_id;
            return handle;
        }
        // If reuse failed, fall through to allocating a fresh id.
    }

    handle.id = g_next_id.fetch_add(1, std::memory_order_relaxed);
    uint64_t base = (handle.id / kSegmentSize) * kSegmentSize;
    size_t slot = static_cast<size_t>(handle.id - base);
    Segment* seg = get_or_create_segment(base);

    TensorRecord* rec = new (std::nothrow) TensorRecord{};
    if (!rec) {
        arena_free(offset, cap);
        return AbstractTensorHandle{};
    }
    rec->desc = desc;
    rec->data = data;
    rec->offset = offset;
    rec->bytes = static_cast<size_t>(cap);
    rec->lease_owner_id = handle.id;
    rec->owns_lease = true;
    rec->lease_refs.store(1, std::memory_order_release);
    rec->alive.store(true, std::memory_order_release);

    TensorRecord* expected = nullptr;
    if (!seg->slots[slot].compare_exchange_strong(expected, rec, std::memory_order_release, std::memory_order_acquire)) {
        delete rec;
        arena_free(offset, cap);
        return AbstractTensorHandle{};
    }
    return handle;
}

void InMemoryBackend::destroy(AbstractTensorHandle handle) {
    if (!abstract_tensor_handle_is_valid(handle)) return;
    TensorRecord* rec = find_record(handle.id);
    if (!rec) return;

    if (rec->owns_lease) {
        // Owner record: decrement lease refs. Only free when refs reach 0.
        const uint32_t prev = rec->lease_refs.fetch_sub(1, std::memory_order_acq_rel);
        if (prev == 0) {
            // Corrupt refcount; restore.
            rec->lease_refs.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (prev == 1) {
            maybe_free_owner_lease(handle.id);
        }
        return;
    }

    // View record: mark dead and return its id to the free list, then decrement owner refs.
    bool expected = true;
    if (!rec->alive.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) return;
    const uint64_t owner_id = rec->lease_owner_id;
    rec->data = nullptr;
    rec->offset = 0;
    rec->bytes = 0;
    rec->lease_owner_id = 0;
    rec->owns_lease = false;
    rec->lease_refs.store(0, std::memory_order_relaxed);
    push_free_id(handle.id);

    TensorRecord* owner = find_record(owner_id);
    if (!owner || !owner->owns_lease) return;
    const uint32_t prev = owner->lease_refs.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 0) {
        owner->lease_refs.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (prev == 1) {
        maybe_free_owner_lease(owner_id);
    }
}

bool InMemoryBackend::describe(AbstractTensorHandle handle, TensorDesc* out) const {
    if (!out || !abstract_tensor_handle_is_valid(handle)) return false;
    TensorRecord* rec = find_record(handle.id);
    if (!rec || !rec->alive.load(std::memory_order_acquire)) return false;
    *out = rec->desc;
    return true;
}

bool InMemoryBackend::get_item(AbstractTensorHandle handle,
                               const TensorIndexSpec& index,
                               AbstractTensorHandle* out_handle,
                               TensorDesc* out_desc) {
    if (!out_handle || !out_desc || !abstract_tensor_handle_is_valid(handle)) return false;
    TensorRecord* rec = find_record(handle.id);
    if (!check_record_alive(rec)) return false;

    if (index.dims.size() == 1 && std::holds_alternative<AbstractTensorHandle>(index.dims[0])) {
        const AbstractTensorHandle idx_handle = std::get<AbstractTensorHandle>(index.dims[0]);
        TensorRecord* idx = find_record(idx_handle.id);
        if (!check_record_alive(idx)) return false;
        if (idx->desc.layout != TensorLayout::Dense) return false;
        if (idx->desc.shape.dims.size() != 2) return false;
        const uint32_t rank = static_cast<uint32_t>(rec->desc.shape.dims.size());
        if (idx->desc.shape.dims[1] != rank) return false;

        TensorDesc view_desc = rec->desc;
        view_desc.shape.dims = {idx->desc.shape.dims[0]};
        view_desc.layout = TensorLayout::Opaque;
        view_desc.strides.elems.clear();
        view_desc.slice = {};
        view_desc.slice.valid = true;
        view_desc.slice.base_shape = rec->desc.shape.dims;
        view_desc.slice.indexed = true;
        view_desc.slice.index_handle = idx_handle;

        const uint64_t owner_id = rec->owns_lease ? handle.id : rec->lease_owner_id;
        TensorRecord* owner = find_record(owner_id);
        if (!check_record_alive(owner) || !owner->owns_lease) return false;
        owner->lease_refs.fetch_add(1, std::memory_order_acq_rel);

        AbstractTensorHandle out = create_record_for_view(view_desc, rec->data, 0, owner_id);
        if (!abstract_tensor_handle_is_valid(out)) {
            const uint32_t prev = owner->lease_refs.fetch_sub(1, std::memory_order_acq_rel);
            if (prev == 1) maybe_free_owner_lease(owner_id);
            return false;
        }

        *out_handle = out;
        *out_desc = view_desc;
        return true;
    }

    std::vector<IndexSpan> spans;
    TensorDesc tmp_desc;
    if (!build_index_plan(rec->desc, index, spans, tmp_desc)) return false;

    uint64_t elem_offset = 0;
    TensorDesc view_desc;
    if (!compute_view_desc_and_offset(rec->desc, spans, view_desc, elem_offset)) return false;

    const uint32_t elem_size = tensor_dtype_size_bytes(rec->desc.dtype);
    if (elem_size == 0) return false;

    const uint64_t owner_id = rec->owns_lease ? handle.id : rec->lease_owner_id;
    TensorRecord* owner = find_record(owner_id);
    if (!check_record_alive(owner) || !owner->owns_lease) return false;
    owner->lease_refs.fetch_add(1, std::memory_order_acq_rel);

    uint8_t* base_bytes = static_cast<uint8_t*>(rec->data);
    void* view_ptr = base_bytes + elem_offset * static_cast<uint64_t>(elem_size);
    const size_t view_bytes = static_cast<size_t>(view_desc.shape.element_count()) * static_cast<size_t>(elem_size);

    AbstractTensorHandle out = create_record_for_view(view_desc, view_ptr, view_bytes, owner_id);
    if (!abstract_tensor_handle_is_valid(out)) {
        const uint32_t prev = owner->lease_refs.fetch_sub(1, std::memory_order_acq_rel);
        if (prev == 1) maybe_free_owner_lease(owner_id);
        return false;
    }

    *out_handle = out;
    *out_desc = view_desc;
    return true;
}

bool InMemoryBackend::set_item(AbstractTensorHandle handle,
                               const TensorIndexSpec& index,
                               AbstractTensorHandle value) {
    if (!abstract_tensor_handle_is_valid(handle) || !abstract_tensor_handle_is_valid(value)) return false;
    TensorRecord* rec = find_record(handle.id);
    TensorRecord* val = find_record(value.id);
    if (!check_record_alive(rec) || !check_record_alive(val)) return false;
    if (rec->desc.dtype != val->desc.dtype) return false;

    if (index.dims.size() == 1 && std::holds_alternative<AbstractTensorHandle>(index.dims[0])) {
        const AbstractTensorHandle idx_handle = std::get<AbstractTensorHandle>(index.dims[0]);
        TensorRecord* idx = find_record(idx_handle.id);
        if (!check_record_alive(idx)) return false;
        if (idx->desc.layout != TensorLayout::Dense) return false;
        if (idx->desc.shape.dims.size() != 2) return false;
        const uint32_t rank = static_cast<uint32_t>(rec->desc.shape.dims.size());
        if (idx->desc.shape.dims[1] != rank) return false;

        const uint64_t n = idx->desc.shape.dims[0];
        const bool val_scalar = (val->desc.shape.element_count() == 1);
        const bool val_1d = (!val_scalar && val->desc.shape.dims.size() == 1 && val->desc.shape.dims[0] == static_cast<uint32_t>(n));
        if (!val_scalar && !val_1d) return false;

        std::vector<uint64_t> in_strides;
        if (!compute_strides_for_desc(rec->desc, in_strides)) return false;
        std::vector<uint64_t> idx_strides;
        if (!compute_strides_for_desc(idx->desc, idx_strides)) return false;
        std::vector<uint64_t> val_strides;
        if (!compute_strides_for_desc(val->desc, val_strides)) return false;

        const uint32_t elem_size = tensor_dtype_size_bytes(rec->desc.dtype);
        if (elem_size == 0) return false;

        struct IndexSetCtx {
            const uint8_t* idx_data = nullptr;
            const uint8_t* val_data = nullptr;
            uint8_t* out_data = nullptr;
            const uint64_t* in_strides = nullptr;
            const uint64_t* idx_strides = nullptr;
            const uint64_t* val_strides = nullptr;
            const uint32_t* shape = nullptr;
            uint32_t rank = 0;
            TensorDType idx_dtype = TensorDType::I32;
            uint32_t elem_size = 0;
            bool val_scalar = false;
            bool val_1d = false;
        };

        IndexSetCtx ctx;
        ctx.idx_data = static_cast<const uint8_t*>(idx->data);
        ctx.val_data = static_cast<const uint8_t*>(val->data);
        ctx.out_data = static_cast<uint8_t*>(rec->data);
        ctx.in_strides = in_strides.data();
        ctx.idx_strides = idx_strides.data();
        ctx.val_strides = val_strides.data();
        ctx.shape = rec->desc.shape.dims.data();
        ctx.rank = rank;
        ctx.idx_dtype = idx->desc.dtype;
        ctx.elem_size = elem_size;
        ctx.val_scalar = val_scalar;
        ctx.val_1d = val_1d;

        auto job_fn = [](const void* c, uint32_t y0, uint32_t y1) {
            const auto* ctx = static_cast<const IndexSetCtx*>(c);
            auto read_index = [&](uint64_t i, uint32_t d) -> int64_t {
                const uint64_t offset = i * ctx->idx_strides[0] + static_cast<uint64_t>(d) * ctx->idx_strides[1];
                switch (ctx->idx_dtype) {
                    case TensorDType::I32: return static_cast<int64_t>(reinterpret_cast<const int32_t*>(ctx->idx_data)[offset]);
                    case TensorDType::I64: return static_cast<int64_t>(reinterpret_cast<const int64_t*>(ctx->idx_data)[offset]);
                    case TensorDType::U32: return static_cast<int64_t>(reinterpret_cast<const uint32_t*>(ctx->idx_data)[offset]);
                    case TensorDType::U64: return static_cast<int64_t>(reinterpret_cast<const uint64_t*>(ctx->idx_data)[offset]);
                    default: return 0;
                }
            };

            for (uint64_t i = y0; i < y1; ++i) {
                uint64_t in_offset = 0;
                for (uint32_t d = 0; d < ctx->rank; ++d) {
                    int64_t coord = read_index(i, d);
                    const int64_t dim = static_cast<int64_t>(ctx->shape[d]);
                    if (coord < 0) coord += dim;
                    if (coord < 0 || coord >= dim) return;
                    in_offset += static_cast<uint64_t>(coord) * ctx->in_strides[d];
                }

                uint64_t val_offset = 0;
                if (!ctx->val_scalar) {
                    val_offset = i * ctx->val_strides[0];
                }

                std::memcpy(ctx->out_data + in_offset * ctx->elem_size,
                            ctx->val_data + val_offset * ctx->elem_size,
                            ctx->elem_size);
            }
        };

        submit_row_jobs(tensor_op_pool(), job_fn, &ctx, 0, static_cast<uint32_t>(n));

        return true;
    }

    std::vector<IndexSpan> spans;
    TensorDesc slice_desc;
    if (!build_index_plan(rec->desc, index, spans, slice_desc)) return false;
    if (slice_desc.shape.dims.empty()) {
        slice_desc.shape.dims.push_back(1u);
    }

    if (val->desc.shape.dims != slice_desc.shape.dims) return false;

    std::vector<uint64_t> in_strides;
    if (!compute_strides_for_desc(rec->desc, in_strides)) return false;

    std::vector<uint64_t> val_strides;
    if (!compute_strides_for_desc(val->desc, val_strides)) return false;

    std::vector<uint64_t> out_dense_strides;
    if (!compute_dense_strides(slice_desc, out_dense_strides)) return false;

    const uint32_t elem_size = tensor_dtype_size_bytes(rec->desc.dtype);
    if (elem_size == 0) return false;

    const size_t out_elems = static_cast<size_t>(slice_desc.shape.element_count());
    uint8_t* in_bytes = static_cast<uint8_t*>(rec->data);
    const uint8_t* val_bytes = static_cast<uint8_t*>(val->data);

    std::vector<uint64_t> out_coords(out_dense_strides.size(), 0);
    for (size_t linear = 0; linear < out_elems; ++linear) {
        size_t tmp = linear;
        for (size_t i = 0; i < out_dense_strides.size(); ++i) {
            const uint64_t stride = out_dense_strides[i];
            if (stride == 0) { out_coords[i] = 0; continue; }
            out_coords[i] = tmp / stride;
            tmp %= stride;
        }

        size_t out_dim_idx = 0;
        uint64_t in_offset = 0;
        for (size_t d = 0; d < spans.size(); ++d) {
            const auto& span = spans[d];
            uint64_t coord = 0;
            if (span.is_int) {
                coord = static_cast<uint64_t>(span.index);
            } else {
                coord = static_cast<uint64_t>(span.slice.start + static_cast<int64_t>(out_coords[out_dim_idx]) * span.slice.step);
                ++out_dim_idx;
            }
            in_offset += coord * in_strides[d];
        }

        uint64_t val_offset = 0;
        for (size_t d = 0; d < out_coords.size(); ++d) {
            val_offset += out_coords[d] * val_strides[d];
        }

        std::memcpy(in_bytes + in_offset * elem_size, val_bytes + val_offset * elem_size, elem_size);
    }

    return true;
}

bool InMemoryBackend::get_allocation_info(AbstractTensorHandle handle, AllocationInfo* out) const {
    if (!out || !abstract_tensor_handle_is_valid(handle)) return false;
    TensorRecord* rec = find_record(handle.id);
    if (!rec) return false;
    out->data = rec->data;
    out->bytes = rec->bytes;
    out->alive = rec->alive.load(std::memory_order_acquire);
    out->bucket = 0;
    return true;
}

bool InMemoryBackend::get_arena_stats(ArenaStats* out) const {
    if (!out) return false;
    if (!arena_ensure_initialized()) return false;
    std::lock_guard<std::mutex> lock(g_arena.lease_mu);
    const ArenaStatsSnapshot s = arena_stats_locked();
    out->reserve_bytes = s.reserve_bytes;
    out->active_leases = s.active_leases;
    out->active_leased_bytes = s.active_leased_bytes;
    out->peak_active_leased_bytes = s.peak_active_leased_bytes;
    out->alloc_calls = s.alloc_calls;
    out->free_calls = s.free_calls;
    out->alloc_fail_oom = s.alloc_fail_oom;
    out->free_drop_no_nodes = s.free_drop_no_nodes;
    out->span_nodes_capacity = s.span_nodes_capacity;
    out->span_nodes_free = s.span_nodes_free;
    out->free_spans = s.free_spans;
    out->free_bytes = s.free_bytes;
    out->largest_free_span = s.largest_free_span;
    return true;
}

bool InMemoryBackend::map(AbstractTensorHandle handle, void** out_data, size_t* out_bytes) const {
    if (!out_data || !out_bytes || !abstract_tensor_handle_is_valid(handle)) return false;
    TensorRecord* rec = find_record(handle.id);
    if (!rec || !rec->alive.load(std::memory_order_acquire)) return false;
    if (rec->desc.slice.valid && rec->desc.slice.indexed) return false;
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
