#include "common/tensors/abstraction/dyadic_bins.h"
#include "common/tensors/abstraction/in_memory_backend.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>

namespace {
constexpr uint32_t kWidth = 1920u;
constexpr uint32_t kHeight = 1080u;
constexpr uint32_t kCount = 1'000'000u;

inline void encode_u32_be(uint32_t v, uint8_t out[4]) {
    out[0] = static_cast<uint8_t>((v >> 24) & 0xFFu);
    out[1] = static_cast<uint8_t>((v >> 16) & 0xFFu);
    out[2] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    out[3] = static_cast<uint8_t>((v) & 0xFFu);
}
}

int main() {
    using clock = std::chrono::high_resolution_clock;

    nodus::tensors::InMemoryBackend backend;

    nodus::tensors::DyadicBinsConfig cfg{};
    cfg.idx_bytes = 4;
    cfg.chunk_cap = 8192;
    cfg.stage_bits = 8;
    cfg.max_nodes = 1'200'000;
    cfg.max_chunks = 4096;
    cfg.preallocate_chunks = false;

    nodus::tensors::DyadicBins bins;
    if (!bins.init(cfg, &backend)) {
        std::fprintf(stderr, "dyadic_bins_bench: init failed\n");
        return 1;
    }

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_int_distribution<uint32_t> dist_x(0u, kWidth - 1u);
    std::uniform_int_distribution<uint32_t> dist_y(0u, kHeight - 1u);

    uint8_t idx_be[4];

    const auto t0 = clock::now();
    for (uint32_t i = 0; i < kCount; ++i) {
        const uint32_t x = dist_x(rng);
        const uint32_t y = dist_y(rng);
        const uint32_t linear = y * kWidth + x;
        encode_u32_be(linear, idx_be);
        if (!bins.add(idx_be, 1.0)) {
            std::fprintf(stderr, "dyadic_bins_bench: add failed at %u\n", i);
            return 1;
        }
    }
    const auto t1 = clock::now();

    const auto t2 = clock::now();
    if (!bins.refine_all()) {
        std::fprintf(stderr, "dyadic_bins_bench: refine failed\n");
        return 1;
    }
    const auto t3 = clock::now();

    const double add_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double refine_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::printf("dyadic_bins_bench: 1080p, 1,000,000 random indices (no dedupe)\n");
    std::printf("  add:    %.3f ms\n", add_ms);
    std::printf("  refine: %.3f ms\n", refine_ms);

    bins.close();
    return 0;
}
