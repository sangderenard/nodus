#include "common/tensors/abstraction/dyadic_bins.h"
#include "common/tensors/abstraction/in_memory_backend.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace {
#if defined(DYADIC_BINS_LOG)
#define DYADIC_BINS_LOGF(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define DYADIC_BINS_LOGF(...) ((void)0)
#endif

using namespace nodus::tensors;
using IndexT = uint16_t;
using ValueT = uint16_t;
constexpr uint32_t kStageBits = 8u;
constexpr uint32_t kStageSize = 1u << kStageBits;

struct RefPages {
    std::vector<std::vector<std::pair<IndexT, ValueT>>> pages;
    uint8_t active = 0;
    uint8_t retain = 1;
    uint8_t inbox = 2;
    RefPages() : pages(3) {}
};

inline uint32_t msb_pos_nonzero(IndexT v) {
    return (uint32_t)(dyadic_uint_bits_v<IndexT> - 1u - dyadic_lzcnt<IndexT>(v));
}

struct RefBins {
    std::vector<RefPages> bins;
    std::vector<ValueT> stage;
    uint32_t total_bins = 0;
    std::vector<uint8_t> phase_counter;

    explicit RefBins(uint32_t total)
        : bins(total), stage(kStageSize, 0), total_bins(total), phase_counter(total, 0u) {}

    void push_inbox(uint32_t b, IndexT idx, ValueT val) {
        bins[b].pages[bins[b].inbox].push_back({idx, val});
    }

    void push_retain(uint32_t b, IndexT idx, ValueT val) {
        bins[b].pages[bins[b].retain].push_back({idx, val});
    }

    void stage_add(IndexT idx, ValueT val) {
        const IndexT lo = (IndexT)(idx & (kStageSize - 1u));
        stage[(uint32_t)lo] = (ValueT)(stage[(uint32_t)lo] + val);
    }

    void classify(IndexT idx, ValueT val) {
        const IndexT hi = (IndexT)(idx >> kStageBits);
        if (hi == 0) {
            stage_add(idx, val);
            return;
        }
        uint32_t b = msb_pos_nonzero(hi);
        push_inbox(b, idx, val);
    }

    void prepare_for_scan(uint32_t b) {
        auto& bin = bins[b];
        bin.pages[bin.retain].clear();
        if (bin.pages[bin.inbox].empty()) return;
        if (bin.pages[bin.active].empty()) {
            bin.active = bin.inbox;
            bin.inbox = bin.retain;
            bin.retain = (uint8_t)(3u - bin.active - bin.inbox);
            bin.pages[bin.inbox].clear();
            bin.pages[bin.retain].clear();
            return;
        }
        auto& active = bin.pages[bin.active];
        auto& inbox = bin.pages[bin.inbox];
        active.insert(active.end(), inbox.begin(), inbox.end());
        inbox.clear();
        bin.inbox = bin.retain;
        bin.retain = (uint8_t)(3u - bin.active - bin.inbox);
        bin.pages[bin.inbox].clear();
        bin.pages[bin.retain].clear();
    }

    void swap_inbox(uint32_t b) {
        std::swap(bins[b].active, bins[b].inbox);
    }

    void swap_retain(uint32_t b) {
        std::swap(bins[b].active, bins[b].retain);
    }

    void finish_scan(uint32_t b) {
        auto& bin = bins[b];
        auto& A = bin.pages[bin.active];
        auto& R = bin.pages[bin.retain];
        auto& I = bin.pages[bin.inbox];

        if (!I.empty()) {
            R.insert(R.end(), I.begin(), I.end());
            I.clear();
        }

        A.clear();

        const uint8_t oldA = bin.active;
        const uint8_t oldI = bin.inbox;
        const uint8_t oldR = bin.retain;

        bin.active = oldR;
        bin.inbox = oldA;
        bin.retain = oldI;

        bin.pages[bin.inbox].clear();
        bin.pages[bin.retain].clear();
    }

    void phase_a(uint32_t b) {
        prepare_for_scan(b);
        const IndexT lo_mask = ((IndexT)1u << kStageBits) - 1u;
        const uint32_t msb0_pos = b;
        const IndexT msb0_mask = (IndexT)1u << msb0_pos;
        const IndexT msb1_mask = (IndexT)1u << (msb0_pos - 1u);
        auto& active = bins[b].pages[bins[b].active];
        for (const auto& kv : active) {
            const IndexT idx = kv.first;
            const ValueT val = kv.second;
            const IndexT lo = (IndexT)(idx & lo_mask);
            const IndexT hi = (IndexT)(idx >> kStageBits);
            const bool gate_move = ((hi & msb1_mask) == 0);
            if (!gate_move) {
                push_retain(b, idx, val);
                continue;
            }
            const IndexT hi_move = (IndexT)(hi ^ msb0_mask);
            const IndexT idx_move = (IndexT)((hi_move << kStageBits) | lo);
            classify(idx_move, val);
        }
        finish_scan(b);
    }

    void phase_b(uint32_t b) {
        prepare_for_scan(b);
        const IndexT lo_mask = ((IndexT)1u << kStageBits) - 1u;
        const uint32_t msb0_pos = b;
        const IndexT msb0_mask = (IndexT)1u << msb0_pos;
        const IndexT msb1_mask = (IndexT)1u << (msb0_pos - 1u);
        auto& active = bins[b].pages[bins[b].active];
        for (const auto& kv : active) {
            const IndexT idx = kv.first;
            const ValueT val = kv.second;
            const IndexT lo = (IndexT)(idx & lo_mask);
            IndexT hi = (IndexT)(idx >> kStageBits);
            hi = (IndexT)(hi ^ (msb0_mask | msb1_mask));
            const IndexT idx2 = (IndexT)((hi << kStageBits) | lo);
            classify(idx2, val);
        }
        finish_scan(b);
    }

    void bin0_drain() {
        prepare_for_scan(0);
        auto& active = bins[0].pages[bins[0].active];
        for (const auto& kv : active) {
            stage_add(kv.first, kv.second);
        }
        finish_scan(0);
    }
};
}

struct CaseResult {
    uint32_t mismatches = 0;
};

static CaseResult run_case(const char* label,
                           const std::vector<IndexT>& indices,
                           const std::vector<ValueT>& values,
                           nodus::tensors::AbstractTensor& output_tensor,
                           nodus::tensors::InMemoryBackend& backend,
                           nodus::tensors::AbstractTensorPool& pool) {
    const uint32_t index_count = (uint32_t)indices.size();
    const IndexT index_range = 1024u;

    const uint32_t total_bins = dyadic_bin_count(index_range, kStageBits);

    nodus::tensors::TensorDesc desc_bins{};
    desc_bins.dtype = dyadic_uint_bytes_dtype<IndexT, ValueT>();
    desc_bins.layout = nodus::tensors::TensorLayout::Dense;
    desc_bins.shape.dims = { (uint32_t)(total_bins * 3u), index_count, 2u };

    nodus::tensors::TensorDesc desc_stage{};
    desc_stage.dtype = dyadic_value_dtype<ValueT>();
    desc_stage.layout = nodus::tensors::TensorLayout::Dense;
    desc_stage.shape.dims = { kStageSize };

    nodus::tensors::TensorDesc desc_counts{};
    desc_counts.dtype = nodus::tensors::TensorDType::Bytes4;
    desc_counts.layout = nodus::tensors::TensorLayout::Dense;
    desc_counts.shape.dims = { (uint32_t)(total_bins * 3u) };

    nodus::tensors::TensorDesc desc_pages{};
    desc_pages.dtype = nodus::tensors::TensorDType::Bytes;
    desc_pages.layout = nodus::tensors::TensorLayout::Dense;
    desc_pages.shape.dims = { total_bins };

    auto bins_tensor = pool.acquire_tensor(desc_bins, &backend);
    auto stage_tensor = pool.acquire_tensor(desc_stage, &backend);
    auto counts_tensor = pool.acquire_tensor(desc_counts, &backend);
    auto page_active_tensor = pool.acquire_tensor(desc_pages, &backend);
    auto page_retain_tensor = pool.acquire_tensor(desc_pages, &backend);
    auto page_inbox_tensor = pool.acquire_tensor(desc_pages, &backend);

    if (!output_tensor.valid()) {
        nodus::tensors::TensorDesc desc_output{};
        desc_output.dtype = dyadic_value_dtype<ValueT>();
        desc_output.layout = nodus::tensors::TensorLayout::Dense;
        desc_output.shape.dims = { kStageSize * 2u };
        output_tensor = nodus::tensors::AbstractTensor::create(desc_output, &backend);
        if (!output_tensor.valid()) {
            std::fprintf(stderr, "%s: output tensor create failed\n", label);
            return {1u};
        }
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    uint32_t output_offset = 0;
    if (!dyadic_mt_bitmask_algo(
            index_range,
            index_count,
            const_cast<IndexT*>(indices.data()),
            const_cast<ValueT*>(values.data()),
            1u,
            kStageBits,
            1u,
            1u,
            1u,
            1u,
            output_tensor,
            &output_offset,
            pool,
            bins_tensor,
            stage_tensor,
            counts_tensor,
            page_active_tensor,
            page_retain_tensor,
            page_inbox_tensor)) {
        std::fprintf(stderr, "%s: dyadic_mt_bitmask_algo failed\n", label);
        return {1u};
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    void* stage_ptr = nullptr;
    size_t stage_bytes = 0;
    if (!backend.map(stage_tensor.handle(), &stage_ptr, &stage_bytes)) {
        std::fprintf(stderr, "%s: stage map failed\n", label);
        return {1u};
    }

    std::vector<ValueT> stage_out(kStageSize);
    std::memcpy(stage_out.data(), stage_ptr, kStageSize * sizeof(ValueT));
    backend.unmap(stage_tensor.handle());

    for (size_t i = 0; i < indices.size(); ++i) {
        DYADIC_BINS_LOGF("%s input[%zu]: idx=%u val=%u\n", label, i, indices[i], values[i]);
    }

    const auto tref0 = std::chrono::high_resolution_clock::now();
    RefBins ref(total_bins);
    for (size_t i = 0; i < indices.size(); ++i) {
        ref.classify(indices[i], values[i]);
    }
    const uint32_t sort_base = static_cast<uint32_t>(DyadicScheduleOp::Bin0Sort);
    const uint32_t inbox_base = static_cast<uint32_t>(DyadicScheduleOp::Bin0SwapInbox);
    const uint32_t retain_base = static_cast<uint32_t>(DyadicScheduleOp::Bin0SwapRetain);
    for (const DyadicScheduleOp op : single_threaded_dyadic_schedule) {
        const uint32_t u = static_cast<uint32_t>(op);
        if (u >= sort_base && u < sort_base + kBinSortCount) {
            const uint32_t bin = u - sort_base;
            if (bin >= ref.total_bins) continue;
            if (bin == 0u) {
                ref.bin0_drain();
            } else if (bin == 1u) {
                ref.phase_a(bin);
            } else {
                const bool do_phase_a = ((ref.phase_counter[bin] & 1u) == 0u);
                ref.phase_counter[bin] = (uint8_t)(ref.phase_counter[bin] + 1u);
                if (do_phase_a) {
                    ref.phase_a(bin);
                } else {
                    ref.phase_b(bin);
                }
            }
            continue;
        }
        if (u >= inbox_base && u < inbox_base + kBinSortCount) {
            const uint32_t bin = u - inbox_base;
            if (bin >= ref.total_bins) continue;
            ref.swap_inbox(bin);
            continue;
        }
        if (u >= retain_base && u < retain_base + kBinSortCount) {
            const uint32_t bin = u - retain_base;
            if (bin >= ref.total_bins) continue;
            ref.swap_retain(bin);
            continue;
        }
    }
    const auto tref1 = std::chrono::high_resolution_clock::now();

    const double ms_impl = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_ref = std::chrono::duration<double, std::milli>(tref1 - tref0).count();
    std::printf("%s: impl %.3f ms, ref %.3f ms\n", label, ms_impl, ms_ref);

    void* output_ptr = nullptr;
    size_t output_bytes = 0;
    if (!backend.map(output_tensor.handle(), &output_ptr, &output_bytes)) {
        std::fprintf(stderr, "%s: output map failed\n", label);
        return {1u};
    }
    const auto* output = static_cast<const ValueT*>(output_ptr);
    uint32_t mismatch_count = 0;
    for (uint32_t i = 0; i < kStageSize; ++i) {
        if (output[i] != ref.stage[i]) {
            DYADIC_BINS_LOGF("%s mismatch at %u (got %u, expected %u, stage %u)\n",
                             label, i, output[i], ref.stage[i], stage_out[i]);
            mismatch_count += 1u;
        }
    }
    backend.unmap(output_tensor.handle());

    if (mismatch_count == 0) {
        std::printf("%s: ok\n", label);
        return {0u};
    }

    std::fprintf(stderr, "%s: %u mismatches\n", label, mismatch_count);
    return {mismatch_count};
}

int main() {
    nodus::tensors::InMemoryBackend backend;
    nodus::tensors::AbstractTensorPool pool;
    nodus::tensors::AbstractTensor output_tensor;

    const std::vector<IndexT> sparse_indices = {0u, 3u, 255u, 256u, 260u, 511u, 512u, 777u};
    const std::vector<ValueT> sparse_values = {1u, 2u, 0u, 3u, 4u, 5u, 6u, 7u};

    constexpr IndexT index_range = 1024u;
    const uint32_t medium_count = 100'000u;
    const uint32_t dense_count = 1'000'000u;
    std::vector<IndexT> medium_indices(medium_count);
    std::vector<ValueT> medium_values(medium_count);
    std::vector<IndexT> dense_indices(dense_count);
    std::vector<ValueT> dense_values(dense_count);
    std::mt19937 rng(1337u);
    std::uniform_int_distribution<uint32_t> dist_idx(0u, (uint32_t)index_range - 1u);
    std::uniform_int_distribution<uint32_t> dist_val(0u, (uint32_t)std::numeric_limits<ValueT>::max());
    for (uint32_t i = 0; i < medium_count; ++i) {
        medium_indices[i] = (IndexT)dist_idx(rng);
        medium_values[i] = (ValueT)dist_val(rng);
    }
    for (uint32_t i = 0; i < dense_count; ++i) {
        dense_indices[i] = (IndexT)dist_idx(rng);
        dense_values[i] = (ValueT)dist_val(rng);
    }

    const CaseResult small_result_1 = run_case("small", sparse_indices, sparse_values, output_tensor, backend, pool);
    const CaseResult medium_result_1 = run_case("medium", medium_indices, medium_values, output_tensor, backend, pool);
    const CaseResult large_result = run_case("large", dense_indices, dense_values, output_tensor, backend, pool);
    const CaseResult medium_result_2 = run_case("medium2", medium_indices, medium_values, output_tensor, backend, pool);
    const CaseResult small_result_2 = run_case("small2", sparse_indices, sparse_values, output_tensor, backend, pool);

    const bool ok = (small_result_1.mismatches == 0) && (medium_result_1.mismatches == 0) &&
                    (large_result.mismatches == 0) && (medium_result_2.mismatches == 0) &&
                    (small_result_2.mismatches == 0);
    return ok ? 0 : 1;
}
