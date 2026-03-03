#include "common/tensors/abstraction/in_memory_backend.h"

#include "common/tensors/abstraction/tensor_registry.h"
#include "common/tensors/abstraction/tensor_math.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <memory>
#if defined(_WIN32) || defined(_WIN64)
#define NODUS_OS_WINDOWS 1
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <windows.h>
#elif defined(__APPLE__)
#define NODUS_OS_APPLE 1
#include <sys/sysctl.h>
#include <sys/mman.h>
#elif defined(__linux__)
#define NODUS_OS_LINUX 1
#include <sys/mman.h>
#include <fstream>
#endif
#if WIN32
#include <malloc.h>
#endif
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>

namespace nodus::tensors {

//#define NODUS_INMEM_LOGGING(...)
#ifndef NODUS_INMEM_LOGGING
#define NODUS_INMEM_LOGGING(...) ((void)0)
#endif

namespace {
constexpr size_t kSegmentSize = 1024;
constexpr size_t kPageAlignment = 4096;
constexpr size_t kLeaseAlignment = 64;

constexpr size_t align_up(size_t v, size_t alignment) {
    return (v + alignment - 1) & ~(alignment - 1);
}

struct Arena;

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
    std::atomic<uint32_t> map_refs{0};
    std::atomic<bool> alive{false};
    std::atomic<uint64_t> next_free_id{0};
    bool external = false;
    Arena* arena = nullptr;
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
    uint64_t base_offset = 0;
    size_t malloc_bytes = 0;
    bool uses_virtual = false;
    uint64_t committed_bytes = 0;
    uint64_t reserve_bytes = 0;
    bool initialized = false;
    bool retired = false;
    uint64_t trim_low = 0;
    uint64_t trim_high = 0;

    std::mutex lease_mu;
    ArenaSpanNode* nodes = nullptr;
    uint32_t node_capacity = 0;
    int32_t free_node_head = -1;
    int32_t clean_list_head = -1; // sorted by offset, zeroed spans
    int32_t dirty_list_head = -1; // sorted by offset, not yet zeroed

    std::condition_variable cleaner_cv;
    std::atomic<bool> cleaner_started{false};
    std::atomic<bool> cleaner_stop{false};
    std::thread cleaner_thread;

    // Debug/telemetry counters (guarded by lease_mu).
    uint64_t active_leases = 0;
    uint64_t active_leased_bytes = 0;
    uint64_t peak_active_leased_bytes = 0;
    uint64_t alloc_calls = 0;
    uint64_t free_calls = 0;
    uint64_t alloc_fail_oom = 0;
    uint64_t free_drop_no_nodes = 0;
};

static std::vector<std::unique_ptr<Arena>> g_arena_books;
static Arena* g_active_arena = nullptr;

static std::mutex g_pause_mu;
static std::condition_variable g_pause_cv;
static std::atomic<bool> g_pause_maps{false};
static std::atomic<uint64_t> g_active_maps{0};
static std::atomic<bool> g_destroy_arena_when_empty{false};
static std::atomic<InMemoryBackend::ArenaBackingPolicy> g_arena_backing_policy{InMemoryBackend::ArenaBackingPolicy::VirtualOnly};
static std::atomic<uint64_t> g_arena_reserve_bytes{10ull * 1024ull * 1024ull * 1024ull};
static std::atomic<uint64_t> g_arena_min_commit_bytes{256ull * 1024ull * 1024ull};
static std::atomic<uint64_t> g_arena_span_nodes{262144ull};

#define g_arena (*g_active_arena)

#define g_arena (*g_active_arena)

static int32_t arena_pop_node();
static void arena_push_node(int32_t idx);

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

static uint64_t get_system_free_bytes() {
#if NODUS_OS_WINDOWS
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<uint64_t>(status.ullAvailPhys);
    }
    return 0;
#elif NODUS_OS_LINUX
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    uint64_t value = 0;
    std::string unit;
    while (meminfo >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            return value * 1024ull;
        }
    }
    return 0;
#elif NODUS_OS_APPLE
    int64_t free_bytes = 0;
    size_t len = sizeof(free_bytes);
    if (sysctlbyname("hw.memsize", &free_bytes, &len, nullptr, 0) == 0) {
        return static_cast<uint64_t>(free_bytes);
    }
    return 0;
#else
    return 0;
#endif
}

static void os_release_pages(void* base, uint64_t offset, uint64_t size) {
    if (!base || size == 0) return;
    const uint64_t page = kPageAlignment;
    const uint64_t start = align_up_u64(offset, page);
    const uint64_t end = (offset + size) / page * page;
    if (end <= start) return;
#if NODUS_OS_WINDOWS
    VirtualFree(static_cast<uint8_t*>(base) + start, static_cast<SIZE_T>(end - start), MEM_DECOMMIT);
#elif NODUS_OS_LINUX
    madvise(static_cast<uint8_t*>(base) + start, static_cast<size_t>(end - start), MADV_DONTNEED);
#elif NODUS_OS_APPLE
    madvise(static_cast<uint8_t*>(base) + start, static_cast<size_t>(end - start), MADV_FREE_REUSABLE);
#else
    (void)base;
    (void)offset;
    (void)size;
#endif
}

static bool arena_init_book(Arena* arena, uint64_t reserve, uint64_t span_nodes) {
    if (!arena) return false;
    if (span_nodes == 0 || span_nodes > 50'000'000ull) return false;
    reserve = align_up_u64(reserve, kPageAlignment);

    const uint64_t alloc_bytes = reserve + static_cast<uint64_t>(kPageAlignment);
    void* raw = nullptr;
#if NODUS_OS_WINDOWS
    const auto policy = g_arena_backing_policy.load(std::memory_order_acquire);
    if (policy == InMemoryBackend::ArenaBackingPolicy::VirtualOnly) {
        raw = VirtualAlloc(nullptr, static_cast<SIZE_T>(alloc_bytes), MEM_RESERVE, PAGE_READWRITE);
        if (!raw) return false;
        arena->uses_virtual = true;
    } else {
        raw = std::malloc(static_cast<size_t>(alloc_bytes));
        if (!raw) return false;
        arena->uses_virtual = false;
    }
#else
    raw = std::malloc(static_cast<size_t>(alloc_bytes));
    if (!raw) return false;
    arena->uses_virtual = false;
#endif

    arena->malloc_base = raw;
    arena->malloc_bytes = static_cast<size_t>(alloc_bytes);
    uintptr_t p = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned = (p + (kPageAlignment - 1)) & ~(static_cast<uintptr_t>(kPageAlignment - 1));
    arena->base = reinterpret_cast<uint8_t*>(aligned);
    arena->base_offset = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(arena->base) - reinterpret_cast<uintptr_t>(arena->malloc_base));
    arena->reserve_bytes = reserve;
    arena->trim_low = 0;
    arena->trim_high = reserve;
    arena->committed_bytes = arena->uses_virtual ? 0u : reserve;

    arena->nodes = static_cast<ArenaSpanNode*>(std::calloc(static_cast<size_t>(span_nodes), sizeof(ArenaSpanNode)));
    if (!arena->nodes) {
#if NODUS_OS_WINDOWS
        if (arena->uses_virtual && arena->malloc_base) {
            VirtualFree(arena->malloc_base, 0, MEM_RELEASE);
        } else
#endif
        if (arena->malloc_base) {
            std::free(arena->malloc_base);
        }
        arena->malloc_base = nullptr;
        arena->base = nullptr;
        arena->malloc_bytes = 0;
        arena->reserve_bytes = 0;
        arena->uses_virtual = false;
        return false;
    }

    arena->node_capacity = static_cast<uint32_t>(span_nodes);
    arena->free_node_head = 0;
    for (uint32_t i = 0; i < arena->node_capacity; ++i) {
        arena->nodes[i].next = (i + 1 < arena->node_capacity) ? static_cast<int32_t>(i + 1) : -1;
    }

    const int32_t root = arena->free_node_head;
    arena->free_node_head = arena->nodes[root].next;
    arena->nodes[root].offset = 0;
    arena->nodes[root].size = reserve;
    arena->nodes[root].next = -1;
    arena->clean_list_head = root;
    arena->dirty_list_head = -1;

    if (!arena->cleaner_started.exchange(true)) {
        arena->cleaner_stop.store(false, std::memory_order_release);
        arena->cleaner_thread = std::thread([arena] {
            for (;;) {
                int32_t node = -1;
                uint64_t offset = 0;
                uint64_t size = 0;
                std::unique_lock<std::mutex> lock(arena->lease_mu);
                arena->cleaner_cv.wait_for(lock, std::chrono::milliseconds(10), [arena] {
                    return arena->dirty_list_head >= 0 || arena->cleaner_stop.load(std::memory_order_acquire);
                });
                if (arena->cleaner_stop.load(std::memory_order_acquire)) {
                    break;
                }
                if (arena->dirty_list_head < 0) {
                    continue;
                }
                node = arena->dirty_list_head;
                arena->dirty_list_head = arena->nodes[node].next;
                arena->nodes[node].next = -1;
                offset = arena->nodes[node].offset;
                size = arena->nodes[node].size;

                if (size > 0 && arena->base) {
                    if (offset <= arena->reserve_bytes && size <= arena->reserve_bytes && offset <= arena->reserve_bytes - size) {
                        std::memset(arena->base + offset, 0, static_cast<size_t>(size));
                    }
                }

                {
                    int32_t prev = -1;
                    int32_t cur = arena->clean_list_head;
                    while (cur >= 0 && arena->nodes[cur].offset < offset) {
                        prev = cur;
                        cur = arena->nodes[cur].next;
                    }
                    if (prev < 0) {
                        arena->nodes[node].next = arena->clean_list_head;
                        arena->clean_list_head = node;
                    } else {
                        arena->nodes[node].next = arena->nodes[prev].next;
                        arena->nodes[prev].next = node;
                    }

                    int32_t next = arena->nodes[node].next;
                    if (next >= 0 && arena->nodes[node].offset + arena->nodes[node].size == arena->nodes[next].offset) {
                        arena->nodes[node].size += arena->nodes[next].size;
                        arena->nodes[node].next = arena->nodes[next].next;
                        arena->nodes[next].next = arena->free_node_head;
                        arena->free_node_head = next;
                    }

                    if (prev >= 0 && arena->nodes[prev].offset + arena->nodes[prev].size == arena->nodes[node].offset) {
                        arena->nodes[prev].size += arena->nodes[node].size;
                        arena->nodes[prev].next = arena->nodes[node].next;
                        arena->nodes[node].next = arena->free_node_head;
                        arena->free_node_head = node;
                    }
                }
            }
        });
    }

    arena->initialized = true;

    const uint64_t min_bytes_cfg = g_arena_min_commit_bytes.load(std::memory_order_acquire);
    const uint64_t min_bytes = std::min<uint64_t>(reserve, min_bytes_cfg);
    if (min_bytes < reserve) {
#if NODUS_OS_WINDOWS
        if (arena->uses_virtual) {
            const uint64_t commit_bytes = align_up_u64(min_bytes, kPageAlignment);
            if (commit_bytes > 0) {
                void* res = VirtualAlloc(arena->base, static_cast<SIZE_T>(commit_bytes), MEM_COMMIT, PAGE_READWRITE);
                if (!res) return false;
            }
            arena->committed_bytes = commit_bytes;
        }
#else
        os_release_pages(arena->base, min_bytes, reserve - min_bytes);
#endif
    } else {
#if NODUS_OS_WINDOWS
        if (arena->uses_virtual) {
            void* res = VirtualAlloc(arena->base, static_cast<SIZE_T>(reserve), MEM_COMMIT, PAGE_READWRITE);
            if (!res) return false;
            arena->committed_bytes = reserve;
        }
#endif
    }
    return true;
}

static bool arena_ensure_initialized() {
    if (!g_active_arena) {
        g_arena_books.emplace_back(std::make_unique<Arena>());
        g_active_arena = g_arena_books.back().get();
    }
    if (g_arena.initialized) return g_arena.base != nullptr;

    // This is the only place we allow a one-time heap allocation besides the arena itself.
    // Guard init with the same mutex that guards the lease table.
    std::lock_guard<std::mutex> lock(g_arena.lease_mu);
    if (g_arena.initialized) return g_arena.base != nullptr;

    const uint64_t reserve = g_arena_reserve_bytes.load(std::memory_order_acquire);
    const uint64_t span_nodes = g_arena_span_nodes.load(std::memory_order_acquire);
    if (!arena_init_book(g_active_arena, reserve, span_nodes)) {
        g_arena.initialized = true;
        return false;
    }
    return true;
}

static void arena_insert_clean_span(uint64_t offset, uint64_t size) {
    if (size == 0) return;
    uint64_t aligned_off = align_up_u64(offset, kLeaseAlignment);
    if (aligned_off > offset) {
        const uint64_t delta = aligned_off - offset;
        if (size <= delta) return;
        size -= delta;
        offset = aligned_off;
    }
    const uint64_t sz = align_up_u64(size, kLeaseAlignment);
    int32_t node = arena_pop_node();
    if (node < 0) {
        g_arena.free_drop_no_nodes++;
        return;
    }
    g_arena.nodes[node].offset = offset;
    g_arena.nodes[node].size = sz;

    int32_t prev = -1;
    int32_t cur = g_arena.clean_list_head;
    while (cur >= 0 && g_arena.nodes[cur].offset < offset) {
        prev = cur;
        cur = g_arena.nodes[cur].next;
    }

    if (prev < 0) {
        g_arena.nodes[node].next = g_arena.clean_list_head;
        g_arena.clean_list_head = node;
    } else {
        g_arena.nodes[node].next = g_arena.nodes[prev].next;
        g_arena.nodes[prev].next = node;
    }

    int32_t next = g_arena.nodes[node].next;
    if (next >= 0 && g_arena.nodes[node].offset + g_arena.nodes[node].size == g_arena.nodes[next].offset) {
        g_arena.nodes[node].size += g_arena.nodes[next].size;
        g_arena.nodes[node].next = g_arena.nodes[next].next;
        arena_push_node(next);
    }

    if (prev >= 0 && g_arena.nodes[prev].offset + g_arena.nodes[prev].size == g_arena.nodes[node].offset) {
        g_arena.nodes[prev].size += g_arena.nodes[node].size;
        g_arena.nodes[prev].next = g_arena.nodes[node].next;
        arena_push_node(node);
    }
}

static bool arena_try_expand_locked(uint64_t min_reserve) {
    if (!g_arena.malloc_base || g_arena.base_offset > g_arena.malloc_bytes) return false;
    uint64_t new_reserve = g_arena.reserve_bytes;
    if (new_reserve == 0) return false;
    while (new_reserve < min_reserve) {
        new_reserve = new_reserve * 2u;
        if (new_reserve < g_arena.reserve_bytes) return false;
    }
    if (new_reserve == g_arena.reserve_bytes) return false;

    const uint64_t avail = get_system_free_bytes();
    if (avail > 0) {
        const uint64_t extra = new_reserve - g_arena.reserve_bytes;
        if (extra > avail) return false;
    }

    const uint64_t target_bytes = new_reserve + g_arena.base_offset;
    if (target_bytes <= g_arena.malloc_bytes) {
        const uint64_t old_reserve = g_arena.reserve_bytes;
        g_arena.reserve_bytes = new_reserve;
        if (g_arena.reserve_bytes > old_reserve) {
            arena_insert_clean_span(old_reserve, g_arena.reserve_bytes - old_reserve);
        }
        return true;
    }

#if WIN32
    void* expanded = _expand(g_arena.malloc_base, static_cast<size_t>(target_bytes));
    if (!expanded || expanded != g_arena.malloc_base) {
        return false;
    }
    g_arena.malloc_bytes = static_cast<size_t>(target_bytes);
    const uint64_t old_reserve = g_arena.reserve_bytes;
    g_arena.reserve_bytes = new_reserve;
    if (g_arena.reserve_bytes > old_reserve) {
        arena_insert_clean_span(old_reserve, g_arena.reserve_bytes - old_reserve);
    }
    return true;
#else
    (void)target_bytes;
    return false;
#endif
}



static int32_t arena_pop_node();
static void arena_push_node(int32_t idx);

static bool arena_ensure_committed(uint64_t end_offset) {
    if (!arena_ensure_initialized()) return false;
    if (end_offset > g_arena.reserve_bytes) return false;
#if NODUS_OS_WINDOWS
    if (g_arena.uses_virtual) {
        const uint64_t page = kPageAlignment;
        uint64_t want = align_up_u64(end_offset, page);
        if (want > g_arena.reserve_bytes) want = g_arena.reserve_bytes;
        if (want > g_arena.committed_bytes) {
            uint64_t delta = want - g_arena.committed_bytes;
            void* res = VirtualAlloc(g_arena.base + g_arena.committed_bytes,
                                     static_cast<SIZE_T>(delta),
                                     MEM_COMMIT,
                                     PAGE_READWRITE);
            if (!res) return false;
            g_arena.committed_bytes = want;
        }
    }
#endif
    return true;
}

static bool arena_exodus_locked(uint64_t min_need);



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

    for (int32_t cur = g_arena.clean_list_head; cur >= 0; cur = g_arena.nodes[cur].next) {
        ++s.free_spans;
        s.free_bytes += g_arena.nodes[cur].size;
        s.largest_free_span = std::max<uint64_t>(s.largest_free_span, g_arena.nodes[cur].size);
    }
    for (int32_t cur = g_arena.dirty_list_head; cur >= 0; cur = g_arena.nodes[cur].next) {
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

    const uint64_t need = align_up_u64(bytes, kLeaseAlignment);

    std::lock_guard<std::mutex> lock(g_arena.lease_mu);
    g_arena.alloc_calls++;

    if (!g_arena.nodes || g_arena.node_capacity == 0) {
        std::fprintf(stderr, "[in_memory_backend] arena_alloc failed: node pool missing\n");
        return false;
    }

    auto check_node = [&](int32_t idx, const char* list) -> bool {
        if (idx < 0) return true;
        if (g_arena.node_capacity == 0) return false;
        if (idx >= static_cast<int32_t>(g_arena.node_capacity)) {
            std::fprintf(stderr,
                         "[in_memory_backend] arena list corruption: list=%s idx=%d capacity=%u\n",
                         list,
                         idx,
                         g_arena.node_capacity);
            return false;
        }
        return true;
    };

    auto validate_list = [&](int32_t head, const char* name) -> bool {
        uint64_t last_offset = 0;
        bool first = true;
        uint32_t seen = 0;
        for (int32_t cur = head; cur >= 0; cur = g_arena.nodes[cur].next) {
            if (!check_node(cur, name)) return false;
            const uint64_t off = g_arena.nodes[cur].offset;
            const uint64_t sz = g_arena.nodes[cur].size;
            if (off > g_arena.reserve_bytes || sz > g_arena.reserve_bytes || off > g_arena.reserve_bytes - sz) {
                std::fprintf(stderr,
                             "[in_memory_backend] arena list invalid range: list=%s idx=%d off=%llu sz=%llu reserve=%llu\n",
                             name,
                             cur,
                             static_cast<unsigned long long>(off),
                             static_cast<unsigned long long>(sz),
                             static_cast<unsigned long long>(g_arena.reserve_bytes));
                return false;
            }
            if (!first && off < last_offset) {
                std::fprintf(stderr,
                             "[in_memory_backend] arena list not sorted: list=%s idx=%d off=%llu last=%llu\n",
                             name,
                             cur,
                             static_cast<unsigned long long>(off),
                             static_cast<unsigned long long>(last_offset));
                return false;
            }
            last_offset = off;
            first = false;
            if (++seen > g_arena.node_capacity) {
                std::fprintf(stderr,
                             "[in_memory_backend] arena list cycle: list=%s head=%d\n",
                             name,
                             head);
                return false;
            }
        }
        return true;
    };

    if (bytes >= (1ull << 30)) {
        if (!validate_list(g_arena.clean_list_head, "clean") ||
            !validate_list(g_arena.dirty_list_head, "dirty")) {
            return false;
        }
    }

    // Best-fit search: scan all free spans and pick the smallest that satisfies need.
    auto find_best_fit = [&](int32_t head, int32_t& out_best, uint64_t& out_best_size, int32_t& out_prev) {
        out_best = -1;
        out_best_size = 0;
        out_prev = -1;
        int32_t prev = -1;
        for (int32_t cur = head; cur >= 0; prev = cur, cur = g_arena.nodes[cur].next) {
            if (!check_node(cur, "best_fit")) {
                out_best = -1;
                out_prev = -1;
                return;
            }
            const uint64_t sz = g_arena.nodes[cur].size;
            if (sz < need) continue;
            if (out_best < 0 || sz < out_best_size) {
                out_best = cur;
                out_best_size = sz;
                out_prev = prev;
                if (sz == need) break;
            }
        }
    };

    auto try_alloc_from_combined = [&]() -> bool {
        int32_t c = g_arena.clean_list_head;
        int32_t d = g_arena.dirty_list_head;
        if (!check_node(c, "clean_head") || !check_node(d, "dirty_head")) {
            return false;
        }
        bool have_run = false;
        uint64_t run_start = 0;
        uint64_t run_end = 0;
        uint64_t run_dirty = 0;

        bool have_best = false;
        uint64_t best_start = 0;
        uint64_t best_end = 0;
        uint64_t best_dirty = 0;
        uint64_t best_run_size = 0;

        auto consider_run = [&]() {
            if (!have_run) return;
            const uint64_t run_size = run_end - run_start;
            if (run_size < need) return;
            if (!have_best || run_dirty < best_dirty || (run_dirty == best_dirty && run_size < best_run_size)) {
                have_best = true;
                best_start = run_start;
                best_end = run_end;
                best_dirty = run_dirty;
                best_run_size = run_size;
            }
        };

        while (c >= 0 || d >= 0) {
            bool take_clean = false;
            if (c >= 0 && d >= 0) {
                take_clean = g_arena.nodes[c].offset <= g_arena.nodes[d].offset;
            } else {
                take_clean = (c >= 0);
            }

            int32_t idx = take_clean ? c : d;
            if (!check_node(idx, take_clean ? "clean" : "dirty")) return false;
            const uint64_t off = g_arena.nodes[idx].offset;
            const uint64_t sz = g_arena.nodes[idx].size;
            const bool is_dirty = !take_clean;

            if (!have_run) {
                run_start = off;
                run_end = off + sz;
                run_dirty = is_dirty ? sz : 0;
                have_run = true;
            } else if (off == run_end) {
                run_end = off + sz;
                if (is_dirty) run_dirty += sz;
            } else {
                consider_run();
                run_start = off;
                run_end = off + sz;
                run_dirty = is_dirty ? sz : 0;
            }

            if (take_clean) {
                c = g_arena.nodes[idx].next;
            } else {
                d = g_arena.nodes[idx].next;
            }
            if (!check_node(c, "clean_next") || !check_node(d, "dirty_next")) return false;
        }
        consider_run();

        if (!have_best) return false;

        const uint64_t cand_start = best_start;
        const uint64_t cand_end = best_end;
        const uint64_t cand_size = cand_end - cand_start;

        int32_t remainder_node = -1;

        auto remove_spans_in_range = [&](int32_t& head, bool is_dirty) {
            int32_t prev = -1;
            int32_t cur = head;
            while (cur >= 0) {
                const int32_t next = g_arena.nodes[cur].next;
                const uint64_t off = g_arena.nodes[cur].offset;
                const uint64_t sz = g_arena.nodes[cur].size;
                if (off >= cand_start && off + sz <= cand_end) {
                    if (prev < 0) {
                        head = next;
                    } else {
                        g_arena.nodes[prev].next = next;
                    }
                    g_arena.nodes[cur].next = -1;
                    if (is_dirty && g_arena.base && sz > 0) {
                        if (off > g_arena.reserve_bytes || sz > g_arena.reserve_bytes || off > g_arena.reserve_bytes - sz) {
                            std::fprintf(stderr,
                                         "[in_memory_backend] dirty span out of range: off=%llu sz=%llu reserve=%llu\n",
                                         static_cast<unsigned long long>(off),
                                         static_cast<unsigned long long>(sz),
                                         static_cast<unsigned long long>(g_arena.reserve_bytes));
                            return;
                        }
                        std::fprintf(stderr,
                                     "[in_memory_backend] dirty span memset base=%p off=%llu sz=%llu\n",
                                     static_cast<void*>(g_arena.base),
                                     static_cast<unsigned long long>(off),
                                     static_cast<unsigned long long>(sz));
                        std::memset(g_arena.base + off, 0, static_cast<size_t>(sz));
                    }
                    if (remainder_node < 0) {
                        remainder_node = cur;
                    } else {
                        arena_push_node(cur);
                    }
                } else {
                    prev = cur;
                }
                cur = next;
            }
        };

        remove_spans_in_range(g_arena.clean_list_head, false);
        remove_spans_in_range(g_arena.dirty_list_head, true);

        const uint64_t remainder = cand_size > need ? (cand_size - need) : 0u;
        if (remainder > 0) {
            if (remainder_node < 0) {
                remainder_node = arena_pop_node();
            }
            if (remainder_node < 0) {
                g_arena.free_drop_no_nodes++;
            } else {
                g_arena.nodes[remainder_node].offset = cand_start + need;
                g_arena.nodes[remainder_node].size = remainder;

                int32_t insert_prev = -1;
                int32_t insert_cur = g_arena.clean_list_head;
                while (insert_cur >= 0 && g_arena.nodes[insert_cur].offset < g_arena.nodes[remainder_node].offset) {
                    insert_prev = insert_cur;
                    insert_cur = g_arena.nodes[insert_cur].next;
                }
                if (insert_prev < 0) {
                    g_arena.nodes[remainder_node].next = g_arena.clean_list_head;
                    g_arena.clean_list_head = remainder_node;
                } else {
                    g_arena.nodes[remainder_node].next = g_arena.nodes[insert_prev].next;
                    g_arena.nodes[insert_prev].next = remainder_node;
                }
            }
        } else if (remainder_node >= 0) {
            arena_push_node(remainder_node);
        }

        g_arena.active_leases++;
        g_arena.active_leased_bytes += need;
        g_arena.peak_active_leased_bytes = std::max<uint64_t>(g_arena.peak_active_leased_bytes, g_arena.active_leased_bytes);
        *out_offset = cand_start;
        *out_capacity = need;
        return true;
    };

    auto log_alloc_failure = [&](const char* reason) {
        const ArenaStatsSnapshot s = arena_stats_locked();
        std::fprintf(stderr,
                     "[in_memory_backend] arena_alloc failed (%s): bytes=%llu need=%llu reserve=%llu active=%llu free_bytes=%llu largest_free=%llu spans=%llu free_nodes=%u\n",
                     reason,
                     static_cast<unsigned long long>(bytes),
                     static_cast<unsigned long long>(need),
                     static_cast<unsigned long long>(s.reserve_bytes),
                     static_cast<unsigned long long>(s.active_leased_bytes),
                     static_cast<unsigned long long>(s.free_bytes),
                     static_cast<unsigned long long>(s.largest_free_span),
                     static_cast<unsigned long long>(s.free_spans),
                     s.span_nodes_free);
    };

    for (int attempt = 0; attempt < 2; ++attempt) {
        int32_t best_clean = -1;
        uint64_t best_clean_size = 0;
        int32_t best_clean_prev = -1;
        find_best_fit(g_arena.clean_list_head, best_clean, best_clean_size, best_clean_prev);

        if (best_clean < 0) {
            if (try_alloc_from_combined()) {
                return true;
            }
            if (attempt == 0 && arena_try_expand_locked(g_arena.reserve_bytes + need)) {
                continue;
            }
            if (attempt == 0 && arena_exodus_locked(need)) {
                continue;
            }
            g_arena.alloc_fail_oom++;
            const ArenaStatsSnapshot s = arena_stats_locked();
            NODUS_INMEM_LOGGING(
                "[in_memory_backend] arena_alloc failed: need=%llu reserve=%llu free_bytes=%llu largest_free=%llu active_bytes=%llu free_spans=%llu\n",
                (unsigned long long)need,
                (unsigned long long)s.reserve_bytes,
                (unsigned long long)s.free_bytes,
                (unsigned long long)s.largest_free_span,
                (unsigned long long)s.active_leased_bytes,
                (unsigned long long)s.free_spans);
            NODUS_INMEM_LOGGING(
                "[in_memory_backend] arena_alloc stats: alloc_calls=%llu alloc_fail_oom=%llu free_calls=%llu span_nodes=%u free_nodes=%u drop_no_nodes=%llu\n",
                (unsigned long long)s.alloc_calls,
                (unsigned long long)s.alloc_fail_oom,
                (unsigned long long)s.free_calls,
                s.span_nodes_capacity,
                s.span_nodes_free,
                (unsigned long long)s.free_drop_no_nodes);
            return false;
        }

        const bool from_clean = true;
        const int32_t best = best_clean;
        const uint64_t best_size = best_clean_size;
        const int32_t best_prev = best_clean_prev;

    int32_t& list_head = from_clean ? g_arena.clean_list_head : g_arena.dirty_list_head;

    // Best-fit search: scan all free spans and pick the smallest that satisfies need.
    // (done above)

    int32_t prev = best_prev;
    for (int32_t cur = list_head; cur >= 0; prev = cur, cur = g_arena.nodes[cur].next) {
        if (!check_node(cur, "remove_scan")) {
            log_alloc_failure("bad_list_node");
            return false;
        }
        if (cur == best) break;
    }
    if (best_prev != prev) {
        // best_prev is already correct, but keep prev consistent for removal.
        prev = best_prev;
    }

        const uint64_t span_offset = g_arena.nodes[best].offset;
        const uint64_t span_size = g_arena.nodes[best].size;

    // Remove chosen node from free list.
    if (best_prev < 0) {
        list_head = g_arena.nodes[best].next;
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
        int32_t insert_cur = list_head;
        while (insert_cur >= 0 && g_arena.nodes[insert_cur].offset < g_arena.nodes[best].offset) {
            insert_prev = insert_cur;
            insert_cur = g_arena.nodes[insert_cur].next;
        }
        if (insert_prev < 0) {
            g_arena.nodes[best].next = list_head;
            list_head = best;
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
            if (attempt == 0 && arena_try_expand_locked(end)) {
                continue;
            }
            if (attempt == 0 && arena_exodus_locked(need)) {
                continue;
            }
            log_alloc_failure("end>reserve");
            return false;
        }
        if (!arena_ensure_committed(end)) {
            if (attempt == 0 && arena_try_expand_locked(end)) {
                continue;
            }
            if (attempt == 0 && arena_exodus_locked(need)) {
                continue;
            }
            log_alloc_failure("ensure_committed");
            return false;
        }

        g_arena.active_leases++;
        g_arena.active_leased_bytes += need;
        g_arena.peak_active_leased_bytes = std::max<uint64_t>(g_arena.peak_active_leased_bytes, g_arena.active_leased_bytes);
        *out_offset = alloc_offset;
        *out_capacity = need;
        return true;
    }

    log_alloc_failure("no_span");
    return false;
}

static bool arena_alloc_clean_only(uint64_t bytes, uint64_t* out_offset, uint64_t* out_capacity) {
    if (!out_offset || !out_capacity) return false;
    if (!arena_ensure_initialized()) return false;
    if (bytes == 0) bytes = 1;

    const uint64_t need = align_up_u64(bytes, kLeaseAlignment);

    std::lock_guard<std::mutex> lock(g_arena.lease_mu);
    g_arena.alloc_calls++;

    int32_t best = -1;
    uint64_t best_size = 0;
    int32_t best_prev = -1;
    int32_t prev = -1;
    for (int32_t cur = g_arena.clean_list_head; cur >= 0; prev = cur, cur = g_arena.nodes[cur].next) {
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

    if (best_prev < 0) {
        g_arena.clean_list_head = g_arena.nodes[best].next;
    } else {
        g_arena.nodes[best_prev].next = g_arena.nodes[best].next;
    }
    g_arena.nodes[best].next = -1;

    const uint64_t alloc_offset = span_offset;
    const uint64_t remaining = span_size - need;
    if (remaining > 0) {
        g_arena.nodes[best].offset = alloc_offset + need;
        g_arena.nodes[best].size = remaining;

        int32_t insert_prev = -1;
        int32_t insert_cur = g_arena.clean_list_head;
        while (insert_cur >= 0 && g_arena.nodes[insert_cur].offset < g_arena.nodes[best].offset) {
            insert_prev = insert_cur;
            insert_cur = g_arena.nodes[insert_cur].next;
        }
        if (insert_prev < 0) {
            g_arena.nodes[best].next = g_arena.clean_list_head;
            g_arena.clean_list_head = best;
        } else {
            g_arena.nodes[best].next = g_arena.nodes[insert_prev].next;
            g_arena.nodes[insert_prev].next = best;
        }
    } else {
        arena_push_node(best);
    }

    const uint64_t end = alloc_offset + need;
    if (end > g_arena.reserve_bytes) return false;
    if (!arena_ensure_committed(end)) return false;

    g_arena.active_leases++;
    g_arena.active_leased_bytes += need;
    g_arena.peak_active_leased_bytes = std::max<uint64_t>(g_arena.peak_active_leased_bytes, g_arena.active_leased_bytes);
    *out_offset = alloc_offset;
    *out_capacity = need;
    return true;
}

static void arena_free_locked_for(Arena* arena, uint64_t offset, uint64_t size) {
    if (!arena || size == 0) return;
    Arena& a = *arena;
    const uint64_t sz = align_up_u64(size, kLeaseAlignment);

    if (offset >= a.reserve_bytes || sz > a.reserve_bytes || offset + sz > a.reserve_bytes) {
        std::fprintf(stderr,
                     "[in_memory_backend] arena_free invalid range: offset=%llu size=%llu reserve=%llu\n",
                     static_cast<unsigned long long>(offset),
                     static_cast<unsigned long long>(sz),
                     static_cast<unsigned long long>(a.reserve_bytes));
        return;
    }

    a.free_calls++;
    if (a.active_leases > 0) a.active_leases--;
    if (a.active_leased_bytes >= sz) a.active_leased_bytes -= sz;

    int32_t node = a.free_node_head;
    if (node < 0) {
        a.free_drop_no_nodes++;
        return;
    }
    a.free_node_head = a.nodes[node].next;
    a.nodes[node].next = -1;
    a.nodes[node].offset = offset;
    a.nodes[node].size = sz;

    int32_t prev = -1;
    int32_t cur = a.dirty_list_head;
    while (cur >= 0 && a.nodes[cur].offset < offset) {
        prev = cur;
        cur = a.nodes[cur].next;
    }

    if (prev < 0) {
        a.nodes[node].next = a.dirty_list_head;
        a.dirty_list_head = node;
    } else {
        a.nodes[node].next = a.nodes[prev].next;
        a.nodes[prev].next = node;
    }

    int32_t next = a.nodes[node].next;
    if (next >= 0 && a.nodes[node].offset + a.nodes[node].size == a.nodes[next].offset) {
        a.nodes[node].size += a.nodes[next].size;
        a.nodes[node].next = a.nodes[next].next;
        a.nodes[next].next = a.free_node_head;
        a.free_node_head = next;
    }

    if (prev >= 0 && a.nodes[prev].offset + a.nodes[prev].size == a.nodes[node].offset) {
        a.nodes[prev].size += a.nodes[node].size;
        a.nodes[prev].next = a.nodes[node].next;
        a.nodes[node].next = a.free_node_head;
        a.free_node_head = node;
    }

    a.cleaner_cv.notify_one();
}

static void arena_free_for(Arena* arena, uint64_t offset, uint64_t size) {
    if (size == 0) return;
    if (!arena) return;
    std::lock_guard<std::mutex> lock(arena->lease_mu);
    arena_free_locked_for(arena, offset, size);
}

static void arena_free(uint64_t offset, uint64_t size) {
    if (size == 0) return;
    if (!arena_ensure_initialized()) return;
    arena_free_for(g_active_arena, offset, size);
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

static void pause_maps() {
    std::lock_guard<std::mutex> lock(g_pause_mu);
    g_pause_maps.store(true, std::memory_order_release);
}

static void resume_maps() {
    {
        std::lock_guard<std::mutex> lock(g_pause_mu);
        g_pause_maps.store(false, std::memory_order_release);
    }
    g_pause_cv.notify_all();
}

static void update_owner_views(uint64_t owner_id, uint8_t* old_ptr, uint8_t* new_ptr, Arena* new_arena) {
    if (!old_ptr || !new_ptr) return;
    Segment* seg = g_segments.load(std::memory_order_acquire);
    while (seg) {
        for (size_t i = 0; i < kSegmentSize; ++i) {
            TensorRecord* rec = seg->slots[i].load(std::memory_order_acquire);
            if (!rec || !rec->alive.load(std::memory_order_acquire)) continue;
            const uint64_t id = seg->base + i;
            if (id == owner_id || rec->lease_owner_id == owner_id) {
                if (rec->data) {
                    auto* rec_ptr = static_cast<uint8_t*>(rec->data);
                    const ptrdiff_t delta = rec_ptr - old_ptr;
                    rec->data = new_ptr + delta;
                }
                rec->arena = new_arena;
            }
        }
        seg = seg->next.load(std::memory_order_acquire);
    }
}

static void arena_trim_retired_locked(Arena* arena) {
    if (!arena) return;
    if (!arena->retired) return;
    if (!arena->nodes) return;

    uint64_t min_off = UINT64_MAX;
    uint64_t max_end = 0;
    Segment* seg = g_segments.load(std::memory_order_acquire);
    while (seg) {
        for (size_t i = 0; i < kSegmentSize; ++i) {
            TensorRecord* rec = seg->slots[i].load(std::memory_order_acquire);
            if (!rec || !rec->alive.load(std::memory_order_acquire)) continue;
            if (!rec->owns_lease) continue;
            if (rec->arena != arena) continue;
            if (rec->bytes == 0) continue;
            min_off = std::min<uint64_t>(min_off, rec->offset);
            max_end = std::max<uint64_t>(max_end, rec->offset + static_cast<uint64_t>(rec->bytes));
        }
        seg = seg->next.load(std::memory_order_acquire);
    }

    if (min_off == UINT64_MAX || max_end == 0) {
        arena->trim_low = 0;
        arena->trim_high = 0;
        return;
    }

    const uint64_t page = 4096u;
    uint64_t trim_low = (min_off / page) * page;
    uint64_t trim_high = ((max_end + page - 1) / page) * page;
    trim_low = std::min(trim_low, arena->reserve_bytes);
    trim_high = std::min(trim_high, arena->reserve_bytes);
    if (trim_high < trim_low) trim_high = trim_low;

    arena->trim_low = trim_low;
    arena->trim_high = trim_high;

    auto trim_list = [&](int32_t& head) {
        int32_t prev = -1;
        int32_t cur = head;
        while (cur >= 0) {
            const int32_t next = arena->nodes[cur].next;
            uint64_t off = arena->nodes[cur].offset;
            uint64_t sz = arena->nodes[cur].size;
            uint64_t end = off + sz;

            if (end <= trim_low || off >= trim_high) {
#if NODUS_OS_WINDOWS
                if (arena->uses_virtual) {
                    os_release_pages(arena->base, off, sz);
                }
#else
                os_release_pages(arena->base, off, sz);
#endif
                if (prev < 0) {
                    head = next;
                } else {
                    arena->nodes[prev].next = next;
                }
                arena->nodes[cur].next = arena->free_node_head;
                arena->free_node_head = cur;
                cur = next;
                continue;
            }

            if (off < trim_low) {
                const uint64_t delta = trim_low - off;
                off = trim_low;
                sz = sz > delta ? sz - delta : 0;
            }
            if (off + sz > trim_high) {
                sz = trim_high > off ? trim_high - off : 0;
            }
            arena->nodes[cur].offset = off;
            arena->nodes[cur].size = sz;

            prev = cur;
            cur = next;
        }
    };

    trim_list(arena->clean_list_head);
    trim_list(arena->dirty_list_head);
}

static void arena_release_free_spans_locked(Arena* arena) {
    if (!arena || !arena->base || !arena->nodes) return;
    if (!arena->retired) return;
#if NODUS_OS_WINDOWS
    if (!arena->uses_virtual) return;
#endif
    auto release_list = [&](int32_t head) {
        for (int32_t cur = head; cur >= 0; cur = arena->nodes[cur].next) {
            const uint64_t off = arena->nodes[cur].offset;
            const uint64_t sz = arena->nodes[cur].size;
            os_release_pages(arena->base, off, sz);
        }
    };
    release_list(arena->clean_list_head);
    release_list(arena->dirty_list_head);
}

static bool arena_exodus_locked(uint64_t min_need) {
    pause_maps();

    Arena* old_arena = g_active_arena;
    if (old_arena) {
        old_arena->retired = true;
        if (old_arena->active_leases == 0) {
            arena_release_free_spans_locked(old_arena);
            if (old_arena->nodes) {
                std::free(old_arena->nodes);
                old_arena->nodes = nullptr;
            }
            if (old_arena->malloc_base) {
#if NODUS_OS_WINDOWS
                if (old_arena->uses_virtual) {
                    VirtualFree(old_arena->malloc_base, 0, MEM_RELEASE);
                } else
#endif
                {
                    std::free(old_arena->malloc_base);
                }
                old_arena->malloc_base = nullptr;
            }
            old_arena->base = nullptr;
            old_arena->malloc_bytes = 0;
            old_arena->uses_virtual = false;
            old_arena->committed_bytes = 0;
            old_arena->reserve_bytes = 0;
            old_arena->node_capacity = 0;
            old_arena->free_node_head = -1;
            old_arena->clean_list_head = -1;
            old_arena->dirty_list_head = -1;
            old_arena->initialized = false;
        }
        arena_release_free_spans_locked(old_arena);
    }

    struct MoveCandidate {
        uint64_t owner_id = 0;
        TensorRecord* owner = nullptr;
        uint8_t* old_ptr = nullptr;
        uint64_t offset = 0;
        size_t bytes = 0;
    };
    std::vector<MoveCandidate> candidates;
    uint64_t packed_bytes = 0;

    Segment* seg = g_segments.load(std::memory_order_acquire);
    while (seg) {
        for (size_t i = 0; i < kSegmentSize; ++i) {
            TensorRecord* rec = seg->slots[i].load(std::memory_order_acquire);
            if (!rec || !rec->alive.load(std::memory_order_acquire)) continue;
            if (!rec->owns_lease) continue;
            if (rec->arena != old_arena) continue;
            if (rec->map_refs.load(std::memory_order_acquire) != 0) continue;
            const uint64_t owner_id = seg->base + i;
            candidates.push_back(MoveCandidate{
                owner_id,
                rec,
                static_cast<uint8_t*>(rec->data),
                rec->offset,
                rec->bytes
            });
            packed_bytes += align_up_u64(static_cast<uint64_t>(rec->bytes), kLeaseAlignment);
        }
        seg = seg->next.load(std::memory_order_acquire);
    }

    const uint64_t aligned_need = align_up_u64(min_need ? min_need : 1u, kLeaseAlignment);
    const uint64_t compact_reserve = align_up_u64(packed_bytes, kPageAlignment);
    const uint64_t avail = get_system_free_bytes();
    const uint64_t span_nodes = old_arena && old_arena->node_capacity > 0
        ? old_arena->node_capacity
        : g_arena_span_nodes.load(std::memory_order_acquire);

    Arena* compact_arena = nullptr;
    if (compact_reserve > 0) {
        if (avail > 0 && compact_reserve > avail) {
            std::fprintf(stderr,
                         "[in_memory_backend] arena_exodus failed: compact_reserve=%llu exceeds avail=%llu\n",
                         static_cast<unsigned long long>(compact_reserve),
                         static_cast<unsigned long long>(avail));
            resume_maps();
            return false;
        }

        g_arena_books.emplace_back(std::make_unique<Arena>());
        compact_arena = g_arena_books.back().get();
        if (!arena_init_book(compact_arena, compact_reserve, span_nodes)) {
            g_arena_books.pop_back();
            resume_maps();
            return false;
        }
        compact_arena->retired = true;
    }

    if (compact_arena) {
        Arena* saved_active = g_active_arena;
        g_active_arena = compact_arena;

        size_t moved = 0;
        for (const auto& cand : candidates) {
            uint64_t new_offset = 0;
            uint64_t new_cap = 0;
            if (!arena_alloc(static_cast<uint64_t>(cand.bytes), &new_offset, &new_cap)) {
                std::fprintf(stderr,
                             "[in_memory_backend] arena_exodus failed: compact alloc bytes=%llu\n",
                             static_cast<unsigned long long>(cand.bytes));
                g_active_arena = saved_active;
                resume_maps();
                return false;
            }
            uint8_t* new_ptr = compact_arena->base + new_offset;
            std::memcpy(new_ptr, cand.old_ptr, cand.bytes);

            update_owner_views(cand.owner_id, cand.old_ptr, new_ptr, compact_arena);

            cand.owner->data = new_ptr;
            cand.owner->offset = new_offset;
            cand.owner->arena = compact_arena;

            if (old_arena) {
                arena_free_locked_for(old_arena, cand.offset, static_cast<uint64_t>(cand.bytes));
            }
            ++moved;
        }

        g_active_arena = saved_active;

        if (old_arena) {
            old_arena->retired = true;
            std::lock_guard<std::mutex> lock(old_arena->lease_mu);
            arena_trim_retired_locked(old_arena);
        }

        if (moved > 0 || min_need > 0) {
            g_destroy_arena_when_empty.store(true, std::memory_order_release);
        }
    }

    uint64_t default_reserve = g_arena_reserve_bytes.load(std::memory_order_acquire);
    default_reserve = align_up_u64(default_reserve, kPageAlignment);
    const uint64_t need_reserve = align_up_u64(aligned_need, kPageAlignment);
    uint64_t planned = std::max<uint64_t>(default_reserve, need_reserve);
    if (avail > 0) {
        const uint64_t cap = avail > 1u ? (avail - 1u) : 0u;
        planned = std::min<uint64_t>(planned, cap);
        if (planned == 0u) {
            std::fprintf(stderr,
                         "[in_memory_backend] arena_exodus failed: planned=0 after 1-byte gap (avail=%llu)\n",
                         static_cast<unsigned long long>(avail));
            resume_maps();
            return false;
        }
    }

    g_arena_books.emplace_back(std::make_unique<Arena>());
    Arena* new_arena = g_arena_books.back().get();
    if (!arena_init_book(new_arena, planned, span_nodes)) {
        g_arena_books.pop_back();
        std::fprintf(stderr,
                     "[in_memory_backend] arena_exodus failed: new_arena reserve=%llu\n",
                     static_cast<unsigned long long>(planned));
        resume_maps();
        return false;
    }
    new_arena->retired = false;
    g_active_arena = new_arena;

    if (planned < default_reserve / 2u) {
        const uint64_t page = kPageAlignment;
        uint64_t half = align_up_u64(planned / 2u, page);
        if (half < planned) {
            os_release_pages(new_arena->base, half, planned - half);
#if NODUS_OS_WINDOWS
            if (new_arena->uses_virtual) {
                new_arena->committed_bytes = half;
            }
#endif
        }
    }

    resume_maps();
    return true;
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
            rec->map_refs.store(0, std::memory_order_relaxed);
            rec->external = false;
            if (TensorRecord* owner = find_record(lease_owner_id)) {
                rec->arena = owner->arena;
            } else {
                rec->arena = g_active_arena;
            }
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
    rec->map_refs.store(0, std::memory_order_relaxed);
    rec->external = false;
    if (TensorRecord* owner = find_record(lease_owner_id)) {
        rec->arena = owner->arena;
    } else {
        rec->arena = g_active_arena;
    }
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
    // TODO: Investigate potential free-while-mapped race. We currently do not
    // gate on owner->map_refs here, so a mapped tensor could be freed and its
    // span zeroed by the cleaner while still in use.
    // Ensure we only free once.
    bool expected_alive = true;
    if (!owner->alive.compare_exchange_strong(expected_alive, false, std::memory_order_acq_rel)) {
        return;
    }

    Arena* owner_arena = owner->arena;
    Arena* arena = owner_arena ? owner_arena : g_active_arena;
    if (!arena || owner->offset >= arena->reserve_bytes ||
        static_cast<uint64_t>(owner->bytes) > arena->reserve_bytes ||
        owner->offset + static_cast<uint64_t>(owner->bytes) > arena->reserve_bytes) {
        std::fprintf(stderr,
                     "[in_memory_backend] owner lease invalid: id=%llu offset=%llu bytes=%llu reserve=%llu\n",
                     static_cast<unsigned long long>(owner_id),
                     static_cast<unsigned long long>(owner->offset),
                     static_cast<unsigned long long>(owner->bytes),
                     static_cast<unsigned long long>(arena ? arena->reserve_bytes : 0));
        return;
    }
    arena_free_for(arena,
                   owner->offset,
                   static_cast<uint64_t>(owner->bytes));
    owner->data = nullptr;
    owner->offset = 0;
    owner->bytes = 0;
    owner->lease_owner_id = 0;
    owner->owns_lease = false;
    owner->external = false;
    owner->lease_refs.store(0, std::memory_order_relaxed);
    push_free_id(owner_id);
    if (owner_arena && owner_arena->retired && g_destroy_arena_when_empty.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(owner_arena->lease_mu);
        if (owner_arena->active_leases == 0 && owner_arena->malloc_base && owner_arena->nodes) {
            std::free(owner_arena->nodes);
            owner_arena->nodes = nullptr;
            if (owner_arena->malloc_base) {
#if NODUS_OS_WINDOWS
                if (owner_arena->uses_virtual) {
                    VirtualFree(owner_arena->malloc_base, 0, MEM_RELEASE);
                } else
#endif
                {
                    std::free(owner_arena->malloc_base);
                }
            }
            owner_arena->malloc_base = nullptr;
            owner_arena->base = nullptr;
            owner_arena->malloc_bytes = 0;
            owner_arena->uses_virtual = false;
            owner_arena->committed_bytes = 0;
            owner_arena->reserve_bytes = 0;
            owner_arena->node_capacity = 0;
            owner_arena->free_node_head = -1;
            owner_arena->clean_list_head = -1;
            owner_arena->dirty_list_head = -1;
            owner_arena->initialized = false;
        }
    }
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
    if (!arena_alloc(static_cast<uint64_t>(bytes_req), &offset, &cap)) {
        std::fprintf(stderr,
                     "[in_memory_backend] create failed: bytes=%llu dtype=%d rank=%zu\n",
                     static_cast<unsigned long long>(bytes_req),
                     static_cast<int>(desc.dtype),
                     desc.shape.dims.size());
        return handle;
    }
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
            rec->map_refs.store(0, std::memory_order_relaxed);
            rec->external = false;
            rec->arena = g_active_arena;
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
    rec->map_refs.store(0, std::memory_order_relaxed);
    rec->external = false;
    rec->arena = g_active_arena;
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
    rec->map_refs.store(0, std::memory_order_relaxed);
    rec->external = false;
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
    {
        std::unique_lock<std::mutex> lock(g_pause_mu);
        while (g_pause_maps.load(std::memory_order_acquire)) {
            g_pause_cv.wait(lock);
        }
    }
    TensorRecord* rec = find_record(handle.id);
    if (!rec || !rec->alive.load(std::memory_order_acquire)) return false;
    if (rec->desc.slice.valid && rec->desc.slice.indexed) return false;
    TensorRecord* owner = rec->owns_lease ? rec : find_record(rec->lease_owner_id);
    if (owner) {
        owner->map_refs.fetch_add(1, std::memory_order_acq_rel);
    }
    g_active_maps.fetch_add(1, std::memory_order_acq_rel);
    *out_data = rec->data;
    *out_bytes = rec->bytes;
    return true;
}

void InMemoryBackend::unmap(AbstractTensorHandle handle) const {
    if (!abstract_tensor_handle_is_valid(handle)) return;
    TensorRecord* rec = find_record(handle.id);
    if (!rec) return;
    TensorRecord* owner = rec->owns_lease ? rec : find_record(rec->lease_owner_id);
    if (owner) {
        owner->map_refs.fetch_sub(1, std::memory_order_acq_rel);
    }
    g_active_maps.fetch_sub(1, std::memory_order_acq_rel);
}

bool InMemoryBackend::ensure_zeroed(AbstractTensorHandle handle,
                                    const TensorDesc& desc,
                                    bool keep_in_place,
                                    uint64_t byte_offset,
                                    uint64_t byte_count) {
    if (!abstract_tensor_handle_is_valid(handle)) return false;
    TensorRecord* rec = find_record(handle.id);
    if (!check_record_alive(rec)) return false;

    TensorRecord* owner = rec->owns_lease ? rec : find_record(rec->lease_owner_id);
    if (!owner || !owner->owns_lease) return false;

    const size_t expected = static_cast<size_t>(desc.shape.element_count()) * tensor_dtype_size_bytes(desc.dtype);
    if (byte_offset > owner->bytes) return false;
    const size_t remaining = owner->bytes - static_cast<size_t>(byte_offset);
    size_t clear_bytes = 0;
    if (byte_count == 0) {
        clear_bytes = std::min(remaining, expected);
    } else {
        clear_bytes = std::min(remaining, static_cast<size_t>(byte_count));
    }

    const bool full_clear = (byte_offset == 0 && (byte_count == 0 || clear_bytes >= expected));
    if (!keep_in_place && full_clear &&
        owner->lease_refs.load(std::memory_order_acquire) == 1) {
        uint64_t new_offset = 0;
        uint64_t new_cap = 0;
        if (arena_alloc_clean_only(static_cast<uint64_t>(owner->bytes), &new_offset, &new_cap)) {
            const uint64_t old_offset = owner->offset;
            const size_t old_bytes = owner->bytes;
            owner->offset = new_offset;
            owner->bytes = static_cast<size_t>(new_cap);
            owner->data = g_arena.base + new_offset;
            arena_free(old_offset, static_cast<uint64_t>(old_bytes));
            return true;
        }
    }

    if (owner->data && clear_bytes > 0) {
        std::memset(static_cast<uint8_t*>(owner->data) + byte_offset, 0, clear_bytes);
        return true;
    }
    return false;
}

InMemoryBackend& in_memory_backend_singleton() {
    static InMemoryBackend backend;
    return backend;
}

void InMemoryBackend::set_arena_backing_policy(ArenaBackingPolicy policy) {
    g_arena_backing_policy.store(policy, std::memory_order_release);
}

void InMemoryBackend::set_arena_reserve_bytes(uint64_t bytes) {
    g_arena_reserve_bytes.store(bytes, std::memory_order_release);
}

void InMemoryBackend::set_arena_min_commit_bytes(uint64_t bytes) {
    g_arena_min_commit_bytes.store(bytes, std::memory_order_release);
}

void InMemoryBackend::set_arena_span_nodes(uint64_t nodes) {
    g_arena_span_nodes.store(nodes, std::memory_order_release);
}

void InMemoryBackend::reset_arena_for_testing() {
    pause_maps();
    for (auto& arena_ptr : g_arena_books) {
        Arena* arena = arena_ptr.get();
        if (!arena) continue;
        arena->cleaner_stop.store(true, std::memory_order_release);
        arena->cleaner_cv.notify_all();
        if (arena->cleaner_thread.joinable()) {
            arena->cleaner_thread.join();
        }
        std::lock_guard<std::mutex> lock(arena->lease_mu);
        if (arena->nodes) {
            std::free(arena->nodes);
            arena->nodes = nullptr;
        }
        if (arena->malloc_base) {
#if NODUS_OS_WINDOWS
            if (arena->uses_virtual) {
                VirtualFree(arena->malloc_base, 0, MEM_RELEASE);
            } else
#endif
            {
                std::free(arena->malloc_base);
            }
            arena->malloc_base = nullptr;
        }
        arena->base = nullptr;
        arena->malloc_bytes = 0;
        arena->uses_virtual = false;
        arena->committed_bytes = 0;
        arena->reserve_bytes = 0;
        arena->node_capacity = 0;
        arena->free_node_head = -1;
        arena->clean_list_head = -1;
        arena->dirty_list_head = -1;
        arena->initialized = false;
        arena->retired = false;
        arena->trim_low = 0;
        arena->trim_high = 0;
        arena->active_leases = 0;
        arena->active_leased_bytes = 0;
        arena->peak_active_leased_bytes = 0;
        arena->cleaner_started.store(false, std::memory_order_release);
    }
    g_arena_books.clear();
    g_active_arena = nullptr;
    g_destroy_arena_when_empty.store(false, std::memory_order_release);
    resume_maps();
}

uint64_t InMemoryBackend::get_system_available_bytes() {
    return get_system_free_bytes();
}

void register_in_memory_backend(bool make_default) {
    register_backend(&in_memory_backend_singleton(), make_default);
}

} // namespace nodus::tensors
