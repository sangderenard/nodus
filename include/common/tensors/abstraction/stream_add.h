#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>

#if defined(_MSC_VER)
  #define NODUS_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
  #define NODUS_FORCE_INLINE inline __attribute__((always_inline))
#else
  #define NODUS_FORCE_INLINE inline
#endif

// ---------- SIMD feature detection ----------
#if defined(__AVX2__) || defined(_M_AVX2)
  #define NODUS_HAS_AVX2 1
  #include <immintrin.h>
#else
  #define NODUS_HAS_AVX2 0
#endif

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && (_M_IX86_FP >= 2))
  #define NODUS_HAS_SSE2 1
  #if !NODUS_HAS_AVX2
    #include <emmintrin.h>
    #include <tmmintrin.h>
  #endif
#else
  #define NODUS_HAS_SSE2 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #define NODUS_HAS_NEON 1
  #include <arm_neon.h>
#else
  #define NODUS_HAS_NEON 0
#endif

namespace nodus::stream_add {

// -----------------------
// Stats (compiled out by default via template bool)
// -----------------------
struct AddStats {
    bool any_overflow = false; // "something overflowed" (not a count)
};

// -----------------------
// SWAR helpers (lane-wise wrap inside a u64 word)
// These avoid inter-lane carries for u8/u16/u32 lanes using mask-split.
// -----------------------
NODUS_FORCE_INLINE uint64_t swar_add_u8_lanes(uint64_t a, uint64_t b, bool* any_ovf /*nullable*/) {
    constexpr uint64_t m = 0x00FF00FF00FF00FFULL;
    uint64_t lo = (a & m) + (b & m);                 // sums in 16-bit fields
    uint64_t hi = ((a >> 8) & m) + ((b >> 8) & m);   // sums in 16-bit fields
    if (any_ovf) {
        // carry-out bit of each 8-bit lane sits at bit 8 of each 16-bit field
        constexpr uint64_t c = 0x0100010001000100ULL;
        *any_ovf |= ((lo & c) | (hi & c)) != 0;
    }
    return (lo & m) | ((hi & m) << 8);
}

NODUS_FORCE_INLINE uint64_t swar_add_u16_lanes(uint64_t a, uint64_t b, bool* any_ovf /*nullable*/) {
    constexpr uint64_t m = 0x0000FFFF0000FFFFULL;
    uint64_t lo = (a & m) + (b & m);                   // sums in 32-bit fields
    uint64_t hi = ((a >> 16) & m) + ((b >> 16) & m);   // sums in 32-bit fields
    if (any_ovf) {
        // carry-out bit of each 16-bit lane sits at bit 16 of each 32-bit field
        constexpr uint64_t c = 0x0001000000010000ULL;
        *any_ovf |= ((lo & c) | (hi & c)) != 0;
    }
    return (lo & m) | ((hi & m) << 16);
}

NODUS_FORCE_INLINE uint64_t swar_add_u32_lanes(uint64_t a, uint64_t b, bool* any_ovf /*nullable*/) {
    // two 32-bit lanes per u64
    uint32_t a0 = (uint32_t)(a);
    uint32_t a1 = (uint32_t)(a >> 32);
    uint32_t b0 = (uint32_t)(b);
    uint32_t b1 = (uint32_t)(b >> 32);
    uint32_t s0 = a0 + b0;
    uint32_t s1 = a1 + b1;
    if (any_ovf) {
        *any_ovf |= (s0 < a0) | (s1 < a1);
    }
    return (uint64_t)s0 | (uint64_t(s1) << 32);
}

// -----------------------
// 1) Lane-wise wrapping add, in-place
//    dst[i] = dst[i] + src[i]  (mod 2^lane_bits), NO carry between lanes.
//    Template TrackOverflow compiles overflow checks out when false.
// -----------------------
template <typename LaneT, bool TrackOverflow = false>
NODUS_FORCE_INLINE AddStats add_wrap_inplace(LaneT* dst, const LaneT* src, size_t lanes) {
    static_assert(std::is_unsigned<LaneT>::value, "LaneT must be unsigned");
    static_assert(sizeof(LaneT) == 1 || sizeof(LaneT) == 2 || sizeof(LaneT) == 4 || sizeof(LaneT) == 8,
                  "LaneT must be u8/u16/u32/u64 sized");

    AddStats st{};

    // SIMD path works for true lane adds of u8/u16/u32/u64.
#if NODUS_HAS_AVX2
    {
        constexpr size_t VBYTES = 32;
        constexpr size_t LANES_PER_V = VBYTES / sizeof(LaneT);
        size_t i = 0;

        auto* d = reinterpret_cast<uint8_t*>(dst);
        auto* s = reinterpret_cast<const uint8_t*>(src);

        const size_t bytes = lanes * sizeof(LaneT);
        for (; i + VBYTES <= bytes; i += VBYTES) {
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + i));
            __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
            __m256i r;
            if constexpr (sizeof(LaneT) == 1) r = _mm256_add_epi8(a, b);
            else if constexpr (sizeof(LaneT) == 2) r = _mm256_add_epi16(a, b);
            else if constexpr (sizeof(LaneT) == 4) r = _mm256_add_epi32(a, b);
            else r = _mm256_add_epi64(a, b);

            if constexpr (TrackOverflow) {
                // Cheap "any overflow": for u32/u64 we can detect sum < a per lane via compare.
                // For u8/u16, overflow detect here is more work; skip (still true semantics).
                if constexpr (sizeof(LaneT) == 4) {
                    __m256i lt = _mm256_cmpgt_epi32(a, r); // a > r  => overflow in unsigned? not exact for all bits
                    (void)lt;
                } else if constexpr (sizeof(LaneT) == 8) {
                    // No unsigned cmpgt_epi64 pre-AVX512; skip.
                }
            }

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + i), r);
        }

        // tail: scalar/SWAR
        size_t tail_lanes = (bytes - i) / sizeof(LaneT);
        dst = reinterpret_cast<LaneT*>(d + i);
        src = reinterpret_cast<const LaneT*>(s + i);
        lanes = tail_lanes;
    }
#elif NODUS_HAS_SSE2
    {
        constexpr size_t VBYTES = 16;
        size_t i = 0;

        auto* d = reinterpret_cast<uint8_t*>(dst);
        auto* s = reinterpret_cast<const uint8_t*>(src);

        const size_t bytes = lanes * sizeof(LaneT);
        for (; i + VBYTES <= bytes; i += VBYTES) {
            __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d + i));
            __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + i));
            __m128i r;
            if constexpr (sizeof(LaneT) == 1) r = _mm_add_epi8(a, b);
            else if constexpr (sizeof(LaneT) == 2) r = _mm_add_epi16(a, b);
            else if constexpr (sizeof(LaneT) == 4) r = _mm_add_epi32(a, b);
            else r = _mm_add_epi64(a, b);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(d + i), r);
        }

        size_t tail_lanes = (bytes - i) / sizeof(LaneT);
        dst = reinterpret_cast<LaneT*>(d + i);
        src = reinterpret_cast<const LaneT*>(s + i);
        lanes = tail_lanes;
    }
#elif NODUS_HAS_NEON
    {
        constexpr size_t VBYTES = 16;
        size_t i = 0;

        auto* d = reinterpret_cast<uint8_t*>(dst);
        auto* s = reinterpret_cast<const uint8_t*>(src);

        const size_t bytes = lanes * sizeof(LaneT);
        for (; i + VBYTES <= bytes; i += VBYTES) {
            if constexpr (sizeof(LaneT) == 1) {
                uint8x16_t a = vld1q_u8(d + i);
                uint8x16_t b = vld1q_u8(s + i);
                vst1q_u8(d + i, vaddq_u8(a, b));
            } else if constexpr (sizeof(LaneT) == 2) {
                uint16x8_t a = vld1q_u16(reinterpret_cast<const uint16_t*>(d + i));
                uint16x8_t b = vld1q_u16(reinterpret_cast<const uint16_t*>(s + i));
                vst1q_u16(reinterpret_cast<uint16_t*>(d + i), vaddq_u16(a, b));
            } else if constexpr (sizeof(LaneT) == 4) {
                uint32x4_t a = vld1q_u32(reinterpret_cast<const uint32_t*>(d + i));
                uint32x4_t b = vld1q_u32(reinterpret_cast<const uint32_t*>(s + i));
                vst1q_u32(reinterpret_cast<uint32_t*>(d + i), vaddq_u32(a, b));
            } else {
                uint64x2_t a = vld1q_u64(reinterpret_cast<const uint64_t*>(d + i));
                uint64x2_t b = vld1q_u64(reinterpret_cast<const uint64_t*>(s + i));
                vst1q_u64(reinterpret_cast<uint64_t*>(d + i), vaddq_u64(a, b));
            }
        }

        size_t tail_lanes = (bytes - i) / sizeof(LaneT);
        dst = reinterpret_cast<LaneT*>(d + i);
        src = reinterpret_cast<const LaneT*>(s + i);
        lanes = tail_lanes;
    }
#endif

    // Scalar / SWAR fallback (also handles SIMD tail)
    if constexpr (sizeof(LaneT) == 8) {
        for (size_t i = 0; i < lanes; ++i) {
            uint64_t a = dst[i];
            uint64_t r = a + src[i];
            if constexpr (TrackOverflow) st.any_overflow |= (r < a);
            dst[i] = r;
        }
    } else if constexpr (sizeof(LaneT) == 4) {
        // Do 2x u32 lanes packed into u64 via SWAR to reduce loop overhead on scalar.
        size_t i = 0;
        bool any = false;
        for (; i + 2 <= lanes; i += 2) {
            uint64_t a64, b64;
            std::memcpy(&a64, dst + i, 8);
            std::memcpy(&b64, src + i, 8);
            uint64_t r64 = swar_add_u32_lanes(a64, b64, (TrackOverflow ? &any : nullptr));
            std::memcpy(dst + i, &r64, 8);
        }
        for (; i < lanes; ++i) {
            uint32_t a = dst[i];
            uint32_t r = a + src[i];
            if constexpr (TrackOverflow) any |= (r < a);
            dst[i] = r;
        }
        if constexpr (TrackOverflow) st.any_overflow |= any;
    } else if constexpr (sizeof(LaneT) == 2) {
        // 4x u16 lanes per u64
        size_t i = 0;
        bool any = false;
        for (; i + 4 <= lanes; i += 4) {
            uint64_t a64, b64;
            std::memcpy(&a64, dst + i, 8);
            std::memcpy(&b64, src + i, 8);
            uint64_t r64 = swar_add_u16_lanes(a64, b64, (TrackOverflow ? &any : nullptr));
            std::memcpy(dst + i, &r64, 8);
        }
        for (; i < lanes; ++i) {
            uint16_t a = dst[i];
            uint16_t r = (uint16_t)(a + src[i]);
            if constexpr (TrackOverflow) any |= (r < a);
            dst[i] = r;
        }
        if constexpr (TrackOverflow) st.any_overflow |= any;
    } else { // sizeof(LaneT) == 1
        // 8x u8 lanes per u64
        size_t i = 0;
        bool any = false;
        for (; i + 8 <= lanes; i += 8) {
            uint64_t a64, b64;
            std::memcpy(&a64, dst + i, 8);
            std::memcpy(&b64, src + i, 8);
            uint64_t r64 = swar_add_u8_lanes(a64, b64, (TrackOverflow ? &any : nullptr));
            std::memcpy(dst + i, &r64, 8);
        }
        for (; i < lanes; ++i) {
            uint8_t a = dst[i];
            uint8_t r = (uint8_t)(a + src[i]);
            if constexpr (TrackOverflow) any |= (r < a);
            dst[i] = r;
        }
        if constexpr (TrackOverflow) st.any_overflow |= any;
    }

    return st;
}

// Convenience for raw byte pointers (treat as u8 lanes)
template <bool TrackOverflow = false>
NODUS_FORCE_INLINE AddStats add_wrap_bytes_inplace(uint8_t* dst, const uint8_t* src, size_t bytes) {
    return add_wrap_inplace<uint8_t, TrackOverflow>(dst, src, bytes);
}

// -----------------------
// 2) Multi-precision byte add WITH carry propagation (big-int style).
//    Endian-agnostic: define view as (LSB pointer, step).
//    - Step = +1 for LSB at start (little-end digit order in memory)
//    - Step = -1 for LSB at end   (big-end digit order in memory)
// Returns carry-out (0/1). Also flags overflow if src has remaining nonzero beyond dst.
// -----------------------
struct DigitView {
    uint8_t* p_lsb;
    ptrdiff_t step; // +1 or -1
    size_t n;       // digits (bytes)
};

template <bool TrackOverflow = false>
NODUS_FORCE_INLINE uint8_t add_mp_bytes_inplace(DigitView dst, const DigitView src, bool* overflow_src_beyond_dst /*nullable*/ = nullptr) {
    const size_t m = (dst.n < src.n) ? dst.n : src.n;
    uint16_t carry = 0;

    for (size_t i = 0; i < m; ++i) {
        uint8_t* d = dst.p_lsb + (ptrdiff_t)i * dst.step;
        const uint8_t* s = src.p_lsb + (ptrdiff_t)i * src.step;
        uint16_t t = (uint16_t)(*d) + (uint16_t)(*s) + carry;
        *d = (uint8_t)t;
        carry = t >> 8;
    }

    for (size_t i = m; carry && i < dst.n; ++i) {
        uint8_t* d = dst.p_lsb + (ptrdiff_t)i * dst.step;
        uint16_t t = (uint16_t)(*d) + carry;
        *d = (uint8_t)t;
        carry = t >> 8;
    }

    if (overflow_src_beyond_dst) {
        bool ovf = (carry != 0);
        if (!ovf) {
            for (size_t i = dst.n; i < src.n; ++i) {
                const uint8_t* s = src.p_lsb + (ptrdiff_t)i * src.step;
                if (*s) { ovf = true; break; }
            }
        }
        *overflow_src_beyond_dst = ovf;
    }

    (void)TrackOverflow; // (hook if you later want more telemetry)
    return (uint8_t)carry;
}

// Helpers to build DigitViews
NODUS_FORCE_INLINE DigitView view_le(uint8_t* base, size_t n) { return DigitView{ base, +1, n }; }
NODUS_FORCE_INLINE DigitView view_be(uint8_t* base, size_t n) { return DigitView{ base + (n ? (n - 1) : 0), -1, n }; }
NODUS_FORCE_INLINE DigitView view_le(const uint8_t* base, size_t n) { return DigitView{ const_cast<uint8_t*>(base), +1, n }; }
NODUS_FORCE_INLINE DigitView view_be(const uint8_t* base, size_t n) { return DigitView{ const_cast<uint8_t*>(base + (n ? (n - 1) : 0)), -1, n }; }

// -----------------------
// 3) Pure bitwise full-adder (per byte). This is correctness/experimentation, not performance.
// -----------------------
NODUS_FORCE_INLINE uint8_t add_u8_bitwise(uint8_t a, uint8_t b, uint8_t& carry_inout) {
    uint8_t sum = 0;
    uint8_t c = (uint8_t)(carry_inout & 1u);
    for (int bit = 0; bit < 8; ++bit) {
        uint8_t ai = (uint8_t)((a >> bit) & 1u);
        uint8_t bi = (uint8_t)((b >> bit) & 1u);
        uint8_t s  = (uint8_t)(ai ^ bi ^ c);
        uint8_t co = (uint8_t)((ai & bi) | (ai & c) | (bi & c));
        sum = (uint8_t)(sum | (uint8_t)(s << bit));
        c = co;
    }
    carry_inout = c;
    return sum;
}

NODUS_FORCE_INLINE uint8_t add_mp_bytes_inplace_bitwise(DigitView dst, const DigitView src) {
    const size_t m = (dst.n < src.n) ? dst.n : src.n;
    uint8_t carry = 0;
    for (size_t i = 0; i < m; ++i) {
        uint8_t* d = dst.p_lsb + (ptrdiff_t)i * dst.step;
        const uint8_t* s = src.p_lsb + (ptrdiff_t)i * src.step;
        *d = add_u8_bitwise(*d, *s, carry);
    }
    for (size_t i = m; carry && i < dst.n; ++i) {
        uint8_t* d = dst.p_lsb + (ptrdiff_t)i * dst.step;
        *d = add_u8_bitwise(*d, 0, carry);
    }
    return carry;
}

// -----------------------
// 4) Optional Eigen path (portable-ish vectorization), good for u32/u64 streams.
//    Enable by defining NODUS_STREAM_ADD_USE_EIGEN before including this header.
// -----------------------
#ifdef NODUS_STREAM_ADD_USE_EIGEN
  #include <Eigen/Core>

template <typename LaneT>
NODUS_FORCE_INLINE AddStats add_wrap_inplace_eigen(LaneT* dst, const LaneT* src, size_t lanes) {
    static_assert(std::is_unsigned<LaneT>::value, "LaneT must be unsigned");
    // Eigen is typically strongest for 32/64-bit lanes. For u8/u16, you often lose.
    Eigen::Map<Eigen::Array<LaneT, Eigen::Dynamic, 1>, Eigen::Unaligned> D(dst, (Eigen::Index)lanes);
    Eigen::Map<const Eigen::Array<LaneT, Eigen::Dynamic, 1>, Eigen::Unaligned> S(src, (Eigen::Index)lanes);
    D += S; // wraps naturally for unsigned LaneT
    return {};
}
#endif

} // namespace nodus::stream_add
