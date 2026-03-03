#pragma once
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <algorithm>

#include "common/tensors/abstraction/rounding.h"

#if defined(_MSC_VER)
    #include <intrin.h>
    #ifndef NODUS_FORCEINLINE
        #define NODUS_FORCEINLINE __forceinline
    #endif
    #ifndef NODUS_RESTRICT
        #define NODUS_RESTRICT __restrict
    #endif
#else
    #include <immintrin.h>
    #ifndef NODUS_FORCEINLINE
        #define NODUS_FORCEINLINE __attribute__((always_inline)) inline
    #endif
    #ifndef NODUS_RESTRICT
        #define NODUS_RESTRICT __restrict__
    #endif
#endif

namespace nodus::tensors {

// --------------------------
// store_xy: scalar, forced-inline
// --------------------------
template <bool UseBounds, bool Clamp, bool Emit>
NODUS_FORCEINLINE void store_xy_i64(int64_t*& dst,
                                    uint8_t*& inb_ptr,
                                    int64_t xi,
                                    int64_t yi,
                                    int64_t bx,
                                    int64_t by) {
    if constexpr (UseBounds) {
        // unsigned-compare trick makes negatives fail bounds naturally.
        const bool inb =
            (static_cast<uint64_t>(xi) < static_cast<uint64_t>(bx)) &&
            (static_cast<uint64_t>(yi) < static_cast<uint64_t>(by));

        if constexpr (Clamp) {
            // Clamp only makes sense when bounds exist.
            if (bx > 0) {
                xi = (xi < 0) ? 0 : (xi >= bx ? (bx - 1) : xi);
            }
            if (by > 0) {
                yi = (yi < 0) ? 0 : (yi >= by ? (by - 1) : yi);
            }
        }

        if constexpr (Emit) {
            *inb_ptr++ = inb ? 1u : 0u;
        }
    } else {
        if constexpr (Emit) {
            *inb_ptr++ = 1u;
        }
    }

    dst[0] = xi;
    dst[1] = yi;
    dst += 2;
}

// --------------------------
// Scalar conversion helpers
// --------------------------
template <typename SrcT>
NODUS_FORCEINLINE int64_t to_i64_rne_unsafe(SrcT v) {
    if constexpr (std::is_same_v<SrcT, float>) {
        return static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(v));
    } else if constexpr (std::is_same_v<SrcT, double>) {
        return rounding::rne_i64_from_f64_unsafe(v);
    } else if constexpr (std::is_integral_v<SrcT>) {
        return static_cast<int64_t>(v);
    } else {
        return rounding::rne_i64_from_f64_unsafe(static_cast<double>(v));
    }
}

// --------------------------
// run_xy: scalar universal
// src points are interleaved: [x0,y0,x1,y1,...]
// --------------------------
template <bool UseBounds, bool Clamp, bool Emit, typename SrcT>
NODUS_FORCEINLINE void run_xy(const SrcT* NODUS_RESTRICT src,
                              uint32_t count,
                              int64_t* NODUS_RESTRICT out_coords,
                              uint8_t* NODUS_RESTRICT out_in_bounds,
                              int64_t bx,
                              int64_t by) {
    int64_t* dst = out_coords;
    uint8_t* inb_ptr = out_in_bounds;

    for (uint32_t i = 0; i < count; ++i) {
        const int64_t xi = to_i64_rne_unsafe(src[0]);
        const int64_t yi = to_i64_rne_unsafe(src[1]);
        store_xy_i64<UseBounds, Clamp, Emit>(dst, inb_ptr, xi, yi, bx, by);
        src += 2;
    }
}

// --------------------------
// Affine application: 2D row-major subset
// ox = x*m0 + y*m4 + m12
// oy = x*m1 + y*m5 + m13
// --------------------------
template <bool UseBounds, bool Clamp, bool Emit, typename SrcT, typename AffineT>
NODUS_FORCEINLINE void run_xy_affine(const SrcT* NODUS_RESTRICT src,
                                     uint32_t count,
                                     int64_t* NODUS_RESTRICT out_coords,
                                     uint8_t* NODUS_RESTRICT out_in_bounds,
                                     int64_t bx,
                                     int64_t by,
                                     AffineT m0, AffineT m1,
                                     AffineT m4, AffineT m5,
                                     AffineT m12, AffineT m13) {
    int64_t* dst = out_coords;
    uint8_t* inb_ptr = out_in_bounds;

    for (uint32_t i = 0; i < count; ++i) {
        const AffineT x = static_cast<AffineT>(src[0]);
        const AffineT y = static_cast<AffineT>(src[1]);
        const AffineT ox = x * m0 + y * m4 + m12;
        const AffineT oy = x * m1 + y * m5 + m13;

        const int64_t xi = to_i64_rne_unsafe(ox);
        const int64_t yi = to_i64_rne_unsafe(oy);

        store_xy_i64<UseBounds, Clamp, Emit>(dst, inb_ptr, xi, yi, bx, by);
        src += 2;
    }
}

// ============================================================================
// AVX2 FAST LANES (F32 only), for the common case:
//   - dims==2
//   - no bounds, no clamp
//   - optionally emit all-ones (memset once)
// These avoid per-point branching and do vector conversion + widen + store.
// ============================================================================

#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))

NODUS_FORCEINLINE __m128i sign_extend2x32_to2x64(__m128i v32) {
#if defined(__SSE4_1__) || (defined(_MSC_VER) && defined(__SSE4_1__))
    return _mm_cvtepi32_epi64(v32); // low 2 int32 -> 2 int64
#else
    // SSE2 fallback: sign-extend via unpack with sign mask.
    __m128i sign = _mm_srai_epi32(v32, 31);
    return _mm_unpacklo_epi32(v32, sign);
#endif
}

NODUS_FORCEINLINE void store_4points_i64_from_8x_i32(__m256i v32,
                                                     int64_t*& dst) {
    // v32 = [x0 y0 x1 y1 x2 y2 x3 y3] as int32
    __m128i lo = _mm256_castsi256_si128(v32);       // x0 y0 x1 y1
    __m128i hi = _mm256_extracti128_si256(v32, 1);  // x2 y2 x3 y3

    // For each 128-bit half: widen (x0,y0) then widen (x1,y1)
    __m128i lo01 = sign_extend2x32_to2x64(lo);                    // x0 y0 (int64)
    __m128i lo23 = sign_extend2x32_to2x64(_mm_srli_si128(lo, 8)); // x1 y1
    __m128i hi01 = sign_extend2x32_to2x64(hi);                    // x2 y2
    __m128i hi23 = sign_extend2x32_to2x64(_mm_srli_si128(hi, 8)); // x3 y3

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 0), lo01);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 2), lo23);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 4), hi01);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 6), hi23);
    dst += 8; // 4 points * 2 coords
}

// no bounds, no emit
NODUS_FORCEINLINE void run_xy_f32_nobounds_noemit_avx2(const float* NODUS_RESTRICT src,
                                                       uint32_t count,
                                                       int64_t* NODUS_RESTRICT out_coords) {
    int64_t* dst = out_coords;
    uint32_t i = 0;

    // 4 points per iter => 8 floats
    for (; i + 4 <= count; i += 4) {
        __m256 xy = _mm256_loadu_ps(src);               // x0 y0 x1 y1 x2 y2 x3 y3
        __m256i v32 = _mm256_cvtps_epi32(xy);           // int32 RNE under MXCSR
        store_4points_i64_from_8x_i32(v32, dst);
        src += 8;
    }

    // tail scalar
    for (; i < count; ++i) {
        const int64_t xi = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(src[0]));
        const int64_t yi = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(src[1]));
        dst[0] = xi; dst[1] = yi;
        dst += 2;
        src += 2;
    }
}

// no bounds, emit all ones
NODUS_FORCEINLINE void run_xy_f32_nobounds_emit_avx2(const float* NODUS_RESTRICT src,
                                                     uint32_t count,
                                                     int64_t* NODUS_RESTRICT out_coords,
                                                     uint8_t* NODUS_RESTRICT out_in_bounds) {
    if (out_in_bounds) {
        std::memset(out_in_bounds, 1, count);
    }
    run_xy_f32_nobounds_noemit_avx2(src, count, out_coords);
}

// affine, no bounds, no emit
NODUS_FORCEINLINE void run_xy_affine_f32_nobounds_noemit_avx2(const float* NODUS_RESTRICT src,
                                                              uint32_t count,
                                                              int64_t* NODUS_RESTRICT out_coords,
                                                              float m0, float m1,
                                                              float m4, float m5,
                                                              float m12, float m13) {
    int64_t* dst = out_coords;
    uint32_t i = 0;

    const __m128 m0v  = _mm_set1_ps(m0);
    const __m128 m1v  = _mm_set1_ps(m1);
    const __m128 m4v  = _mm_set1_ps(m4);
    const __m128 m5v  = _mm_set1_ps(m5);
    const __m128 m12v = _mm_set1_ps(m12);
    const __m128 m13v = _mm_set1_ps(m13);

    // 4 points per iter
    for (; i + 4 <= count; i += 4) {
        // load 8 floats: x0 y0 x1 y1 x2 y2 x3 y3
        __m256 xy8 = _mm256_loadu_ps(src);

        // Extract x0 x1 x2 x3 into low 128, y0 y1 y2 y3 into low 128
        const __m256i idx_x = _mm256_setr_epi32(0,2,4,6, 0,2,4,6);
        const __m256i idx_y = _mm256_setr_epi32(1,3,5,7, 1,3,5,7);
        __m256 xdup = _mm256_permutevar8x32_ps(xy8, idx_x);
        __m256 ydup = _mm256_permutevar8x32_ps(xy8, idx_y);

        __m128 x4 = _mm256_castps256_ps128(xdup);
        __m128 y4 = _mm256_castps256_ps128(ydup);

#if defined(__FMA__) || (defined(_MSC_VER) && defined(__FMA__))
        __m128 ox = _mm_fmadd_ps(y4, m4v, _mm_fmadd_ps(x4, m0v, m12v));
        __m128 oy = _mm_fmadd_ps(y4, m5v, _mm_fmadd_ps(x4, m1v, m13v));
#else
        __m128 ox = _mm_add_ps(_mm_add_ps(_mm_mul_ps(x4, m0v), _mm_mul_ps(y4, m4v)), m12v);
        __m128 oy = _mm_add_ps(_mm_add_ps(_mm_mul_ps(x4, m1v), _mm_mul_ps(y4, m5v)), m13v);
#endif

        __m128i xi32 = _mm_cvtps_epi32(ox);
        __m128i yi32 = _mm_cvtps_epi32(oy);

        // Pack into v32 layout [x0 y0 x1 y1 x2 y2 x3 y3] (int32) using 2x128 halves.
        // We’ll interleave within 128-bit lanes.
        __m128i lo = _mm_unpacklo_epi32(xi32, yi32); // x0 y0 x1 y1
        __m128i hi = _mm_unpackhi_epi32(xi32, yi32); // x2 y2 x3 y3
        __m256i v32 = _mm256_set_m128i(hi, lo);

        store_4points_i64_from_8x_i32(v32, dst);
        src += 8;
    }

    // tail scalar
    for (; i < count; ++i) {
        const float x = src[0], y = src[1];
        const float ox = x*m0 + y*m4 + m12;
        const float oy = x*m1 + y*m5 + m13;
        dst[0] = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(ox));
        dst[1] = static_cast<int64_t>(rounding::rne_i32_from_f32_unsafe(oy));
        dst += 2;
        src += 2;
    }
}

NODUS_FORCEINLINE void run_xy_affine_f32_nobounds_emit_avx2(const float* NODUS_RESTRICT src,
                                                            uint32_t count,
                                                            int64_t* NODUS_RESTRICT out_coords,
                                                            uint8_t* NODUS_RESTRICT out_in_bounds,
                                                            float m0, float m1,
                                                            float m4, float m5,
                                                            float m12, float m13) {
    if (out_in_bounds) {
        std::memset(out_in_bounds, 1, count);
    }
    run_xy_affine_f32_nobounds_noemit_avx2(src, count, out_coords, m0,m1,m4,m5,m12,m13);
}

#endif // AVX2

} // namespace nodus::tensors
