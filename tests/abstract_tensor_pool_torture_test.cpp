#include "common/tensors/abstraction/abstract_tensor_pool.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

using namespace nodus::tensors;

namespace {
constexpr uint64_t kLeaseAlignment = 64u;

struct BucketOptions {
    bool enabled = false;
    uint32_t bucket_pow2_max = 0;
    uint32_t bucket_multiple = 0;
};

static uint32_t round_up_pow2(uint32_t v) {
    if (v <= 1u) return 1u;
    v -= 1u;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    return v + 1u;
}

static uint32_t round_up_multiple(uint32_t v, uint32_t multiple) {
    if (multiple == 0u) return v;
    if (v == 0u) return 0u;
    const uint32_t r = v % multiple;
    return r == 0u ? v : (v + (multiple - r));
}

static std::vector<uint32_t> bucket_dims(const std::vector<uint32_t>& dims, const BucketOptions& opt) {
    if (!opt.enabled) return dims;
    std::vector<uint32_t> out = dims;
    for (auto& d : out) {
        if (d == 0u) continue;
        if (opt.bucket_pow2_max > 0u && d <= opt.bucket_pow2_max) {
            d = round_up_pow2(d);
        } else if (opt.bucket_multiple > 0u) {
            d = round_up_multiple(d, opt.bucket_multiple);
        }
    }
    return out;
}

static uint64_t element_count_u64(const std::vector<uint32_t>& dims) {
    if (dims.empty()) return 0u;
    uint64_t prod = 1u;
    for (uint32_t d : dims) {
        prod *= static_cast<uint64_t>(d);
    }
    return prod;
}

static uint64_t per_alloc_budget_elems(uint64_t max_bytes, uint32_t elem_bytes, uint32_t alloc_count) {
    if (alloc_count == 0u) return 0u;
    if (elem_bytes == 0u) return max_bytes;
    const uint64_t align_guard = kLeaseAlignment > 0u ? (kLeaseAlignment - 1u) : 0u;
    uint64_t usable_bytes = max_bytes > (1u + align_guard) ? (max_bytes - 1u - align_guard) : 0u;
    uint64_t budget_bytes = usable_bytes / static_cast<uint64_t>(alloc_count);
    budget_bytes = (budget_bytes / kLeaseAlignment) * kLeaseAlignment;
    if (budget_bytes < elem_bytes) budget_bytes = elem_bytes;
    return budget_bytes / static_cast<uint64_t>(elem_bytes);
}

static void shrink_dims_to_fit(std::vector<uint32_t>& dims, const BucketOptions& opt, uint64_t max_elems) {
    if (dims.empty()) return;
    if (max_elems == 0u) {
        for (auto& d : dims) d = 1u;
        return;
    }

    for (uint32_t iter = 0; iter < 64u; ++iter) {
        const uint64_t bucketed_elems = element_count_u64(bucket_dims(dims, opt));
        if (bucketed_elems <= max_elems) return;

        size_t max_idx = 0u;
        uint32_t max_dim = dims[0];
        for (size_t i = 1u; i < dims.size(); ++i) {
            if (dims[i] > max_dim) {
                max_dim = dims[i];
                max_idx = i;
            }
        }
        if (max_dim <= 1u) break;
        dims[max_idx] = std::max<uint32_t>(1u, max_dim / 2u);
    }

    for (auto& d : dims) d = 1u;
}
} // namespace

static TensorDesc make_dense_desc(TensorDType dtype, const std::vector<uint32_t>& dims) {
    TensorDesc desc{};
    desc.dtype = dtype;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = dims;
    return desc;
}

static uint32_t rand_in_range(std::mt19937& rng, uint32_t lo, uint32_t hi) {
    std::uniform_int_distribution<uint32_t> dist(lo, hi);
    return dist(rng);
}

static TensorDType rand_dtype(std::mt19937& rng) {
    const TensorDType dtypes[] = {TensorDType::F32, TensorDType::I32, TensorDType::U8};
    const uint32_t idx = rand_in_range(rng, 0u, 2u);
    return dtypes[idx];
}

static std::vector<uint32_t> rand_dims(std::mt19937& rng, uint32_t rank, uint64_t max_elems) {
    std::vector<uint32_t> dims;
    dims.reserve(rank);
    uint64_t remaining = max_elems ? max_elems : 1u;
    for (uint32_t i = 0; i < rank; ++i) {
        const uint32_t remaining_dims = rank - i;
        const double side_f = std::pow(static_cast<double>(remaining), 1.0 / static_cast<double>(remaining_dims));
        const double max_dim_f = static_cast<double>(std::numeric_limits<uint32_t>::max());
        const uint64_t max_dim_u64 = (side_f >= max_dim_f) ? static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
                                                           : static_cast<uint64_t>(side_f);
        const uint32_t max_dim = std::max<uint32_t>(1u, static_cast<uint32_t>(max_dim_u64));
        const uint32_t dim = rand_in_range(rng, 1u, max_dim);
        dims.push_back(dim);
        remaining = std::max<uint64_t>(1u, remaining / dim);
    }
    return dims;
}

static uint64_t cap_bytes_for_system(uint64_t requested,
                                     uint64_t available,
                                     uint64_t divisor,
                                     uint64_t min_bytes) {
    if (available == 0u || divisor == 0u) return requested;
    uint64_t cap = available / divisor;
    if (cap < min_bytes) cap = min_bytes;
    return std::min<uint64_t>(requested, cap);
}

static void touch_tensor(const AbstractTensor& tensor) {
    auto* mem = dynamic_cast<InMemoryBackend*>(tensor.backend());
    assert(mem);
    void* ptr = nullptr;
    size_t bytes = 0;
    std::fprintf(stderr, "[pool_torture] touch_tensor begin\n");
    std::fflush(stderr);
    std::fprintf(stderr, "[pool_torture] touch_tensor map call\n");
    std::fflush(stderr);
    if (!mem->map(tensor.handle(), &ptr, &bytes)) {
        std::fprintf(stderr, "[pool_torture] touch_tensor map failed\n");
        std::fflush(stderr);
        return;
    }
    std::fprintf(stderr,
                 "[pool_torture] touch_tensor map ok ptr=%p bytes=%llu\n",
                 ptr,
                 static_cast<unsigned long long>(bytes));
    std::fflush(stderr);
    // TODO: If a write failure occurs here, compare mapped bytes vs logical tensor bytes
    // and confirm no concurrent free/zeroing is happening for this lease.
    if (ptr && bytes) {
        const TensorDesc desc = tensor.desc();
        const uint64_t logical_bytes = desc.shape.element_count() * static_cast<uint64_t>(tensor_dtype_size_bytes(desc.dtype));
        if (bytes > logical_bytes) {
            std::fprintf(stderr,
                         "[pool_torture] touch_tensor logical_bytes=%llu (capacity_slack=%llu)\n",
                         static_cast<unsigned long long>(logical_bytes),
                         static_cast<unsigned long long>(bytes - logical_bytes));
        } else {
            std::fprintf(stderr,
                         "[pool_torture] touch_tensor logical_bytes=%llu\n",
                         static_cast<unsigned long long>(logical_bytes));
        }
        std::fflush(stderr);
        std::fprintf(stderr, "[pool_torture] touch_tensor memset begin\n");
        std::fflush(stderr);
        constexpr size_t kChunkBytes = 64ull * 1024ull * 1024ull;
        const size_t to_touch = static_cast<size_t>(std::min<uint64_t>(logical_bytes, bytes));
        uint8_t* p = static_cast<uint8_t*>(ptr);
        size_t remaining = to_touch;
        size_t touched = 0;
        while (remaining > 0) {
            const size_t chunk = remaining > kChunkBytes ? kChunkBytes : remaining;
            std::memset(p, 0xA5, chunk);
            p += chunk;
            remaining -= chunk;
            touched += chunk;
            if ((touched % (512ull * 1024ull * 1024ull)) == 0u) {
                std::fprintf(stderr,
                             "[pool_torture] touch_tensor memset progress=%llu/%llu\n",
                             static_cast<unsigned long long>(touched),
                             static_cast<unsigned long long>(to_touch));
                std::fflush(stderr);
            }
        }
        std::fprintf(stderr, "[pool_torture] touch_tensor memset end\n");
        std::fflush(stderr);
    }
    std::fprintf(stderr, "[pool_torture] touch_tensor unmap call\n");
    std::fflush(stderr);
    mem->unmap(tensor.handle());
    std::fprintf(stderr, "[pool_torture] touch_tensor end\n");
    std::fflush(stderr);
}

static void log_arena_stats(const char* label, InMemoryBackend& backend) {
    InMemoryBackend::ArenaStats stats{};
    if (!backend.get_arena_stats(&stats)) {
        std::fprintf(stderr, "[pool_torture] %s arena_stats unavailable\n", label);
        return;
    }
    std::fprintf(stderr,
                 "[pool_torture] %s arena_stats reserve=%llu active_leases=%llu active_bytes=%llu peak=%llu free_bytes=%llu largest_free=%llu free_spans=%llu free_nodes=%u drop_no_nodes=%llu\n",
                 label,
                 static_cast<unsigned long long>(stats.reserve_bytes),
                 static_cast<unsigned long long>(stats.active_leases),
                 static_cast<unsigned long long>(stats.active_leased_bytes),
                 static_cast<unsigned long long>(stats.peak_active_leased_bytes),
                 static_cast<unsigned long long>(stats.free_bytes),
                 static_cast<unsigned long long>(stats.largest_free_span),
                 static_cast<unsigned long long>(stats.free_spans),
                 stats.span_nodes_free,
                 static_cast<unsigned long long>(stats.free_drop_no_nodes));
}

static void run_pool_case(const char* label,
                          AbstractTensorPool& pool,
                          InMemoryBackend& backend,
                          std::mt19937& rng,
                          uint32_t iterations,
                          uint64_t max_bytes,
                          bool prealloc_enabled,
                          uint32_t prealloc_batch,
                          BucketOptions bucket_opts) {
    std::fprintf(stderr, "[pool_torture] begin %s iterations=%u\n", label, iterations);
    const bool clear_each_iter = true;
    std::fprintf(stderr,
                 "[pool_torture] %s cap_bytes=%llu clear_each=%d\n",
                 label,
                 static_cast<unsigned long long>(max_bytes),
                 clear_each_iter ? 1 : 0);

    for (uint32_t i = 0; i < iterations; ++i) {
        const uint32_t rank = rand_in_range(rng, 1u, 4u);
        const TensorDType dtype = rand_dtype(rng);
        const uint32_t elem_bytes = tensor_dtype_size_bytes(dtype);
        const uint32_t prealloc_count = (prealloc_enabled && (i % 7u) == 0u) ? prealloc_batch : 0u;
        const uint32_t alloc_count = prealloc_count + 1u;
        const uint64_t per_alloc_elems = std::max<uint64_t>(1u, per_alloc_budget_elems(max_bytes, elem_bytes, alloc_count));
        std::vector<uint32_t> dims = rand_dims(rng, rank, per_alloc_elems);
        shrink_dims_to_fit(dims, bucket_opts, per_alloc_elems);
        TensorDesc desc = make_dense_desc(dtype, dims);
        const uint64_t bytes_req = desc.shape.element_count() * static_cast<uint64_t>(elem_bytes);
        const uint64_t bucketed_elems = element_count_u64(bucket_dims(dims, bucket_opts));
        const uint64_t bucketed_bytes = bucketed_elems * static_cast<uint64_t>(elem_bytes);
        // TODO: Bucketing only applies to the "virtual-bucket" case; keep this in mind when revisiting sizing.
        assert(bucketed_elems <= per_alloc_elems);
        if (i < 4u) {
            std::fprintf(stderr,
                         "[pool_torture] %s iter=%u dtype=%d rank=%u elems=%llu bytes=%llu bucket_bytes=%llu prealloc=%u allocs=%u\n",
                         label,
                         i,
                         static_cast<int>(dtype),
                         rank,
                         static_cast<unsigned long long>(desc.shape.element_count()),
                         static_cast<unsigned long long>(bytes_req),
                         static_cast<unsigned long long>(bucketed_bytes),
                         prealloc_count,
                         alloc_count);
        }

        if (prealloc_count > 0u) {
            pool.preallocate(desc, &backend, prealloc_count);
        }

        if ((i % 2u) == 0u) {
            if (i < 4u) {
                std::fprintf(stderr, "[pool_torture] %s iter=%u acquire begin\n", label, i);
            }
            auto pooled = pool.acquire(desc, &backend);
            if (!pooled.valid()) {
                std::fprintf(stderr,
                             "[pool_torture] %s acquire failed: iter=%u dtype=%d rank=%u dims=[",
                             label,
                             i,
                             static_cast<int>(dtype),
                             rank);
                for (uint32_t d = 0; d < rank; ++d) {
                    std::fprintf(stderr, "%u", dims[d]);
                    if (d + 1u < rank) {
                        std::fprintf(stderr, ",");
                    }
                }
                std::fprintf(stderr, "]\n");
                log_arena_stats(label, backend);
                return;
            }
            if ((i % 3u) == 0u) {
                if (i < 4u) {
                    std::fprintf(stderr,
                                 "[pool_torture] %s iter=%u touch begin bytes=%llu\n",
                                 label,
                                 i,
                                 static_cast<unsigned long long>(bytes_req));
                }
                touch_tensor(pooled.tensor());
                if (i < 4u) {
                    std::fprintf(stderr, "[pool_torture] %s iter=%u touch end\n", label, i);
                }
            }
            if (i < 4u) {
                std::fprintf(stderr, "[pool_torture] %s iter=%u acquire ok\n", label, i);
            }
        } else {
            if (i < 4u) {
                std::fprintf(stderr, "[pool_torture] %s iter=%u acquire_tensor begin\n", label, i);
            }
            AbstractTensor tensor = pool.acquire_tensor(desc, &backend);
            if (!tensor.valid()) {
                std::fprintf(stderr,
                             "[pool_torture] %s acquire_tensor failed: iter=%u dtype=%d rank=%u dims=[",
                             label,
                             i,
                             static_cast<int>(dtype),
                             rank);
                for (uint32_t d = 0; d < rank; ++d) {
                    std::fprintf(stderr, "%u", dims[d]);
                    if (d + 1u < rank) {
                        std::fprintf(stderr, ",");
                    }
                }
                std::fprintf(stderr, "]\n");
                log_arena_stats(label, backend);
                return;
            }
            if ((i % 5u) == 0u) {
                if (i < 4u) {
                    std::fprintf(stderr,
                                 "[pool_torture] %s iter=%u touch begin bytes=%llu\n",
                                 label,
                                 i,
                                 static_cast<unsigned long long>(bytes_req));
                }
                touch_tensor(tensor);
                if (i < 4u) {
                    std::fprintf(stderr, "[pool_torture] %s iter=%u touch end\n", label, i);
                }
            }
            pool.release(std::move(tensor));
            if (i < 4u) {
                std::fprintf(stderr, "[pool_torture] %s iter=%u acquire_tensor ok\n", label, i);
            }
        }

        if (clear_each_iter) {
            pool.clear();
        }
    }

    const auto stats = pool.stats();
    std::fprintf(stderr,
                 "[pool_torture] %s stats: acquire=%llu hits=%llu misses=%llu created=%llu bucketed=%llu\n",
                 label,
                 static_cast<unsigned long long>(stats.acquire_calls),
                 static_cast<unsigned long long>(stats.cache_hits),
                 static_cast<unsigned long long>(stats.cache_misses),
                 static_cast<unsigned long long>(stats.created_handles),
                 static_cast<unsigned long long>(stats.bucketed_acquires));
    std::fprintf(stderr, "[pool_torture] end %s\n", label);
}

int main() {
    register_in_memory_backend(true);

    uint64_t fixed_max_bytes = 10ull * 1024ull * 1024ull * 1024ull;
    uint64_t virtual_max_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    const uint64_t available_bytes = InMemoryBackend::get_system_available_bytes();
    if (available_bytes > 0u) {
        fixed_max_bytes = cap_bytes_for_system(fixed_max_bytes, available_bytes, 4u, 512ull * 1024ull * 1024ull);
        virtual_max_bytes = cap_bytes_for_system(virtual_max_bytes, available_bytes, 2u, 1024ull * 1024ull * 1024ull);
    }
    const uint64_t fixed_cap_bytes = (fixed_max_bytes * 70ull) / 100ull;

    InMemoryBackend::reset_arena_for_testing();
    InMemoryBackend::set_arena_backing_policy(InMemoryBackend::ArenaBackingPolicy::AllowNonVirtual);
    InMemoryBackend::set_arena_reserve_bytes(fixed_cap_bytes);
    InMemoryBackend::set_arena_min_commit_bytes(fixed_cap_bytes);
    InMemoryBackend& backend_fixed = in_memory_backend_singleton();

    std::mt19937 rng_fixed(0xC0FFEEu);
    AbstractTensorPool::Options opt_default{};
    AbstractTensorPool pool_default(opt_default);
    run_pool_case("fixed",
                  pool_default,
                  backend_fixed,
                  rng_fixed,
                  256,
                  fixed_cap_bytes,
                  false,
                  0,
                  BucketOptions{});

    InMemoryBackend::reset_arena_for_testing();
    InMemoryBackend::set_arena_backing_policy(InMemoryBackend::ArenaBackingPolicy::VirtualOnly);
    InMemoryBackend::set_arena_reserve_bytes(virtual_max_bytes);
    InMemoryBackend::set_arena_min_commit_bytes(256ull * 1024ull * 1024ull);
    InMemoryBackend& backend_virtual = in_memory_backend_singleton();

    std::mt19937 rng(0xC0FFEEu);
    AbstractTensorPool::Options opt_cache{};
    opt_cache.cache_handles = true;
    opt_cache.max_cached_handles_total = 64;
    opt_cache.max_cached_handles_per_key = 8;
    AbstractTensorPool pool_cache(opt_cache);

    AbstractTensorPool::Options opt_bucket{};
    opt_bucket.cache_handles = true;
    opt_bucket.enable_shape_bucketing = true;
    opt_bucket.bucket_pow2_max = 512;
    opt_bucket.bucket_multiple = 128;
    opt_bucket.max_cached_handles_total = 128;
    opt_bucket.max_cached_handles_per_key = 8;
    AbstractTensorPool pool_bucket(opt_bucket);

    run_pool_case("virtual-cache",
                  pool_cache,
                  backend_virtual,
                  rng,
                  384,
                  virtual_max_bytes,
                  true,
                  2,
                  BucketOptions{});
    run_pool_case("virtual-bucket",
                  pool_bucket,
                  backend_virtual,
                  rng,
                  512,
                  virtual_max_bytes,
                  true,
                  2,
                  BucketOptions{true, opt_bucket.bucket_pow2_max, opt_bucket.bucket_multiple});

    std::cout << "abstract_tensor_pool_torture_test: ok\n";
    return 0;
}
