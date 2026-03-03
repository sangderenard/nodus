// rounding.h
// UNSAFE / NO-CHECKS quantization + bit-slicing utilities.
// Contract: the caller guarantees all preconditions (finite, non-negative when required,
// in-range for the target integer width, and within “real index” envelopes if desired).
// Violations produce hard, unexplained failures (UB / wrap / impl-defined behavior).

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
  #define NODUS_FORCEINLINE __forceinline
  #define NODUS_RESTRICT   __restrict
#else
  #define NODUS_FORCEINLINE inline __attribute__((always_inline))
  #define NODUS_RESTRICT   __restrict__
#endif

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
  #include <immintrin.h>
#endif

namespace nodus::tensors::rounding {

// --------------------------------------------
// IEEE-754 masks (field masks, not “precision”)
// --------------------------------------------

constexpr std::uint64_t f64_frac_mask = 0x000FFFFFFFFFFFFFull; // 52 fraction bits
constexpr std::uint64_t f64_exp_mask  = 0x7FF0000000000000ull; // 11 exponent bits
constexpr std::uint64_t f64_sign_mask = 0x8000000000000000ull; // 1 sign bit

constexpr std::uint32_t f32_frac_mask = 0x007FFFFFu;           // 23 fraction bits
constexpr std::uint32_t f32_exp_mask  = 0x7F800000u;           // 8 exponent bits
constexpr std::uint32_t f32_sign_mask = 0x80000000u;           // 1 sign bit

constexpr std::uint64_t u8_mask  = 0xFFull;
constexpr std::uint64_t u16_mask = 0xFFFFull;
constexpr std::uint64_t u32_mask = 0xFFFFFFFFull;
constexpr std::uint64_t u64_mask = 0xFFFFFFFFFFFFFFFFull;

// “Real index” envelopes (unit-step representable)
constexpr std::uint64_t kF32ExactIntLimit = (1ull << 24); // float: exact integers for |x| < 2^24
constexpr std::uint64_t kF64ExactIntLimit = (1ull << 53); // double: exact integers for |x| < 2^53

// Convenience: 53-bit mask (low 53 bits)
constexpr std::uint64_t kU53Mask = (1ull << 53) - 1ull;

// -----------------------------
// Bit-cast helpers (no checks)
// -----------------------------

NODUS_FORCEINLINE std::uint64_t bits_from_f64(double x) noexcept {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

NODUS_FORCEINLINE std::uint32_t bits_from_f32(float x) noexcept {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

NODUS_FORCEINLINE std::uint64_t load_u64_unaligned(const void* p) noexcept {
    std::uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// -----------------------------------------------------
// Rounding/conversion modes (caller picks, no policing)
// -----------------------------------------------------

enum class RoundMode : std::uint8_t {
    TruncTowardZero,   // truncate toward zero
    NearestEven,       // round to nearest, ties-to-even
};

// -----------------------------------------------------
// Scalar float->int conversion (UNSAFE / NO CHECKS)
// -----------------------------------------------------
// Contracts (caller):
// - Inputs are finite and within representable integer range for destination.
// - If you care about “real indexing”: float < 2^24, double < 2^53.
// - Any NaN/Inf/out-of-range behavior is caller-owned.

NODUS_FORCEINLINE std::int64_t trunc_i64_from_f64_unsafe(double x) noexcept {
#if defined(__SSE2__) || defined(_M_X64)
    return _mm_cvttsd_si64(_mm_set_sd(x)); // trunc toward zero
#else
    return static_cast<std::int64_t>(x);
#endif
}

NODUS_FORCEINLINE std::int64_t rne_i64_from_f64_unsafe(double x) noexcept {
#if defined(__SSE4_1__)
    __m128d v = _mm_set_sd(x);
    v = _mm_round_sd(v, v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm_cvttsd_si64(v);
#elif defined(__SSE2__) || defined(_M_X64)
    // Rounds per MXCSR (typical default is nearest-even).
    return _mm_cvtsd_si64(_mm_set_sd(x));
#else
    return static_cast<std::int64_t>(x);
#endif
}

NODUS_FORCEINLINE std::int32_t trunc_i32_from_f32_unsafe(float x) noexcept {
#if defined(__SSE2__) || defined(_M_X64)
    return _mm_cvttss_si32(_mm_set_ss(x)); // trunc toward zero
#else
    return static_cast<std::int32_t>(x);
#endif
}

NODUS_FORCEINLINE std::int32_t rne_i32_from_f32_unsafe(float x) noexcept {
#if defined(__SSE4_1__)
    __m128 v = _mm_set_ss(x);
    v = _mm_round_ss(v, v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm_cvttss_si32(v);
#elif defined(__SSE2__) || defined(_M_X64)
    // Rounds per MXCSR (typical default is nearest-even).
    return _mm_cvtss_si32(_mm_set_ss(x));
#else
    return static_cast<std::int32_t>(x);
#endif
}

// Quantize helpers (UNSAFE): non-negative assumed by design (caller guarantees).
NODUS_FORCEINLINE std::uint64_t quant_u64_from_f64_unsafe(double x, RoundMode rm) noexcept {
    const std::int64_t q = (rm == RoundMode::NearestEven) ? rne_i64_from_f64_unsafe(x)
                                                          : trunc_i64_from_f64_unsafe(x);
    return static_cast<std::uint64_t>(q);
}

NODUS_FORCEINLINE std::uint32_t quant_u32_from_f32_unsafe(float x, RoundMode rm) noexcept {
    const std::int32_t q = (rm == RoundMode::NearestEven) ? rne_i32_from_f32_unsafe(x)
                                                          : trunc_i32_from_f32_unsafe(x);
    return static_cast<std::uint32_t>(q);
}

// -----------------------------------------------------
// Kernel signature and kinds (no checks, no negotiation)
// -----------------------------------------------------

using QuantFn = void(*)(void* NODUS_RESTRICT dst,
                        const void* NODUS_RESTRICT src,
                        std::size_t n,
                        std::uint64_t base) noexcept;

enum class InKind : std::uint8_t {
    F32,
    F64,
    U32,
    U64,
    I32,
    I64,
};

enum class OutKind : std::uint8_t {
    U8,
    U16,
    U32,
    U64,
};

// Plan: caller constructs/chooses it; this header provides a mapper.
// No validation, no clamping, no safety branches in kernels.
struct QuantPlan {
    InKind in_kind;
    OutKind out_kind;
    RoundMode round;
    bool use_base;          // if true: store (q - base) into dst
    std::uint64_t base;     // base for offset packing
    QuantFn fn;             // specialized kernel
};

// --------------------------------------------
// UNSAFE kernels (no range checks, no NaN checks)
// --------------------------------------------

// ---- f32 kernels ----

template <typename OutT, RoundMode RM, bool UseBase>
NODUS_FORCEINLINE void k_f32(void* NODUS_RESTRICT dst,
                             const void* NODUS_RESTRICT src,
                             std::size_t n,
                             std::uint64_t base) noexcept {
    auto* out = static_cast<OutT*>(dst);
    auto* in  = static_cast<const float*>(src);

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t q = quant_u32_from_f32_unsafe(in[i], RM);
        const std::uint64_t v = UseBase ? (static_cast<std::uint64_t>(q) - base)
                                        : static_cast<std::uint64_t>(q);
        out[i] = static_cast<OutT>(v);
    }
}

// ---- f64 kernels ----

template <typename OutT, RoundMode RM, bool UseBase>
NODUS_FORCEINLINE void k_f64(void* NODUS_RESTRICT dst,
                             const void* NODUS_RESTRICT src,
                             std::size_t n,
                             std::uint64_t base) noexcept {
    auto* out = static_cast<OutT*>(dst);
    auto* in  = static_cast<const double*>(src);

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t q = quant_u64_from_f64_unsafe(in[i], RM);
        const std::uint64_t v = UseBase ? (q - base) : q;
        out[i] = static_cast<OutT>(v);
    }
}

// ---- integer kernels ----

template <typename InT, typename OutT, bool UseBase>
NODUS_FORCEINLINE void k_int(void* NODUS_RESTRICT dst,
                             const void* NODUS_RESTRICT src,
                             std::size_t n,
                             std::uint64_t base) noexcept {
    auto* out = static_cast<OutT*>(dst);
    auto* in  = static_cast<const InT*>(src);

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t q = static_cast<std::uint64_t>(in[i]);
        const std::uint64_t v = UseBase ? (q - base) : q;
        out[i] = static_cast<OutT>(v);
    }
}

// --------------------------------------------
// Function-pointer wrappers (to avoid template in type)
// --------------------------------------------

#define NODUS_DEFINE_F32_KERNEL(OUTT, RM, USEBASE, NAME) \
    NODUS_FORCEINLINE void NAME(void* NODUS_RESTRICT dst, const void* NODUS_RESTRICT src, std::size_t n, std::uint64_t base) noexcept { \
        k_f32<OUTT, RM, USEBASE>(dst, src, n, base); \
    }

#define NODUS_DEFINE_F64_KERNEL(OUTT, RM, USEBASE, NAME) \
    NODUS_FORCEINLINE void NAME(void* NODUS_RESTRICT dst, const void* NODUS_RESTRICT src, std::size_t n, std::uint64_t base) noexcept { \
        k_f64<OUTT, RM, USEBASE>(dst, src, n, base); \
    }

#define NODUS_DEFINE_INT_KERNEL(INTT, OUTT, USEBASE, NAME) \
    NODUS_FORCEINLINE void NAME(void* NODUS_RESTRICT dst, const void* NODUS_RESTRICT src, std::size_t n, std::uint64_t base) noexcept { \
        k_int<INTT, OUTT, USEBASE>(dst, src, n, base); \
    }

// f32 -> {u8,u16,u32,u64}, {trunc,rne}, {base/no-base}
NODUS_DEFINE_F32_KERNEL(std::uint8_t,  RoundMode::TruncTowardZero, false, q_f32_u8_trunc)
NODUS_DEFINE_F32_KERNEL(std::uint8_t,  RoundMode::NearestEven,     false, q_f32_u8_rne)
NODUS_DEFINE_F32_KERNEL(std::uint16_t, RoundMode::TruncTowardZero, false, q_f32_u16_trunc)
NODUS_DEFINE_F32_KERNEL(std::uint16_t, RoundMode::NearestEven,     false, q_f32_u16_rne)
NODUS_DEFINE_F32_KERNEL(std::uint32_t, RoundMode::TruncTowardZero, false, q_f32_u32_trunc)
NODUS_DEFINE_F32_KERNEL(std::uint32_t, RoundMode::NearestEven,     false, q_f32_u32_rne)
NODUS_DEFINE_F32_KERNEL(std::uint64_t, RoundMode::TruncTowardZero, false, q_f32_u64_trunc)
NODUS_DEFINE_F32_KERNEL(std::uint64_t, RoundMode::NearestEven,     false, q_f32_u64_rne)

NODUS_DEFINE_F32_KERNEL(std::uint8_t,  RoundMode::TruncTowardZero, true,  q_f32_u8_off_trunc)
NODUS_DEFINE_F32_KERNEL(std::uint8_t,  RoundMode::NearestEven,     true,  q_f32_u8_off_rne)
NODUS_DEFINE_F32_KERNEL(std::uint16_t, RoundMode::TruncTowardZero, true,  q_f32_u16_off_trunc)
NODUS_DEFINE_F32_KERNEL(std::uint16_t, RoundMode::NearestEven,     true,  q_f32_u16_off_rne)
NODUS_DEFINE_F32_KERNEL(std::uint32_t, RoundMode::TruncTowardZero, true,  q_f32_u32_off_trunc)
NODUS_DEFINE_F32_KERNEL(std::uint32_t, RoundMode::NearestEven,     true,  q_f32_u32_off_rne)
NODUS_DEFINE_F32_KERNEL(std::uint64_t, RoundMode::TruncTowardZero, true,  q_f32_u64_off_trunc)
NODUS_DEFINE_F32_KERNEL(std::uint64_t, RoundMode::NearestEven,     true,  q_f32_u64_off_rne)

// f64 -> {u8,u16,u32,u64}, {trunc,rne}, {base/no-base}
NODUS_DEFINE_F64_KERNEL(std::uint8_t,  RoundMode::TruncTowardZero, false, q_f64_u8_trunc)
NODUS_DEFINE_F64_KERNEL(std::uint8_t,  RoundMode::NearestEven,     false, q_f64_u8_rne)
NODUS_DEFINE_F64_KERNEL(std::uint16_t, RoundMode::TruncTowardZero, false, q_f64_u16_trunc)
NODUS_DEFINE_F64_KERNEL(std::uint16_t, RoundMode::NearestEven,     false, q_f64_u16_rne)
NODUS_DEFINE_F64_KERNEL(std::uint32_t, RoundMode::TruncTowardZero, false, q_f64_u32_trunc)
NODUS_DEFINE_F64_KERNEL(std::uint32_t, RoundMode::NearestEven,     false, q_f64_u32_rne)
NODUS_DEFINE_F64_KERNEL(std::uint64_t, RoundMode::TruncTowardZero, false, q_f64_u64_trunc)
NODUS_DEFINE_F64_KERNEL(std::uint64_t, RoundMode::NearestEven,     false, q_f64_u64_rne)

NODUS_DEFINE_F64_KERNEL(std::uint8_t,  RoundMode::TruncTowardZero, true,  q_f64_u8_off_trunc)
NODUS_DEFINE_F64_KERNEL(std::uint8_t,  RoundMode::NearestEven,     true,  q_f64_u8_off_rne)
NODUS_DEFINE_F64_KERNEL(std::uint16_t, RoundMode::TruncTowardZero, true,  q_f64_u16_off_trunc)
NODUS_DEFINE_F64_KERNEL(std::uint16_t, RoundMode::NearestEven,     true,  q_f64_u16_off_rne)
NODUS_DEFINE_F64_KERNEL(std::uint32_t, RoundMode::TruncTowardZero, true,  q_f64_u32_off_trunc)
NODUS_DEFINE_F64_KERNEL(std::uint32_t, RoundMode::NearestEven,     true,  q_f64_u32_off_rne)
NODUS_DEFINE_F64_KERNEL(std::uint64_t, RoundMode::TruncTowardZero, true,  q_f64_u64_off_trunc)
NODUS_DEFINE_F64_KERNEL(std::uint64_t, RoundMode::NearestEven,     true,  q_f64_u64_off_rne)

// u32/u64/i32/i64 -> {u8,u16,u32,u64}, {base/no-base}
// (No rounding; it’s integer.)
NODUS_DEFINE_INT_KERNEL(std::uint32_t, std::uint8_t,  false, q_u32_u8)
NODUS_DEFINE_INT_KERNEL(std::uint32_t, std::uint16_t, false, q_u32_u16)
NODUS_DEFINE_INT_KERNEL(std::uint32_t, std::uint32_t, false, q_u32_u32)
NODUS_DEFINE_INT_KERNEL(std::uint32_t, std::uint64_t, false, q_u32_u64)

NODUS_DEFINE_INT_KERNEL(std::uint32_t, std::uint8_t,  true,  q_u32_u8_off)
NODUS_DEFINE_INT_KERNEL(std::uint32_t, std::uint16_t, true,  q_u32_u16_off)
NODUS_DEFINE_INT_KERNEL(std::uint32_t, std::uint32_t, true,  q_u32_u32_off)
NODUS_DEFINE_INT_KERNEL(std::uint32_t, std::uint64_t, true,  q_u32_u64_off)

NODUS_DEFINE_INT_KERNEL(std::uint64_t, std::uint8_t,  false, q_u64_u8)
NODUS_DEFINE_INT_KERNEL(std::uint64_t, std::uint16_t, false, q_u64_u16)
NODUS_DEFINE_INT_KERNEL(std::uint64_t, std::uint32_t, false, q_u64_u32)
NODUS_DEFINE_INT_KERNEL(std::uint64_t, std::uint64_t, false, q_u64_u64)

NODUS_DEFINE_INT_KERNEL(std::uint64_t, std::uint8_t,  true,  q_u64_u8_off)
NODUS_DEFINE_INT_KERNEL(std::uint64_t, std::uint16_t, true,  q_u64_u16_off)
NODUS_DEFINE_INT_KERNEL(std::uint64_t, std::uint32_t, true,  q_u64_u32_off)
NODUS_DEFINE_INT_KERNEL(std::uint64_t, std::uint64_t, true,  q_u64_u64_off)

NODUS_DEFINE_INT_KERNEL(std::int32_t, std::uint8_t,  false, q_i32_u8)
NODUS_DEFINE_INT_KERNEL(std::int32_t, std::uint16_t, false, q_i32_u16)
NODUS_DEFINE_INT_KERNEL(std::int32_t, std::uint32_t, false, q_i32_u32)
NODUS_DEFINE_INT_KERNEL(std::int32_t, std::uint64_t, false, q_i32_u64)

NODUS_DEFINE_INT_KERNEL(std::int32_t, std::uint8_t,  true,  q_i32_u8_off)
NODUS_DEFINE_INT_KERNEL(std::int32_t, std::uint16_t, true,  q_i32_u16_off)
NODUS_DEFINE_INT_KERNEL(std::int32_t, std::uint32_t, true,  q_i32_u32_off)
NODUS_DEFINE_INT_KERNEL(std::int32_t, std::uint64_t, true,  q_i32_u64_off)

NODUS_DEFINE_INT_KERNEL(std::int64_t, std::uint8_t,  false, q_i64_u8)
NODUS_DEFINE_INT_KERNEL(std::int64_t, std::uint16_t, false, q_i64_u16)
NODUS_DEFINE_INT_KERNEL(std::int64_t, std::uint32_t, false, q_i64_u32)
NODUS_DEFINE_INT_KERNEL(std::int64_t, std::uint64_t, false, q_i64_u64)

NODUS_DEFINE_INT_KERNEL(std::int64_t, std::uint8_t,  true,  q_i64_u8_off)
NODUS_DEFINE_INT_KERNEL(std::int64_t, std::uint16_t, true,  q_i64_u16_off)
NODUS_DEFINE_INT_KERNEL(std::int64_t, std::uint32_t, true,  q_i64_u32_off)
NODUS_DEFINE_INT_KERNEL(std::int64_t, std::uint64_t, true,  q_i64_u64_off)

#undef NODUS_DEFINE_F32_KERNEL
#undef NODUS_DEFINE_F64_KERNEL
#undef NODUS_DEFINE_INT_KERNEL

// --------------------------------------------
// Plan mapper (NO VALIDATION)
// --------------------------------------------
// This is “dispatch-time” selection only. No checks, no fallbacks.
// If you give an unsupported combination, the function pointer is unspecified.

NODUS_FORCEINLINE QuantFn select_quant_fn(InKind in, OutKind out, RoundMode rm, bool use_base) noexcept {
    // f32
    if (in == InKind::F32) {
        if (!use_base) {
            if (out == OutKind::U8)  return (rm == RoundMode::NearestEven) ? q_f32_u8_rne  : q_f32_u8_trunc;
            if (out == OutKind::U16) return (rm == RoundMode::NearestEven) ? q_f32_u16_rne : q_f32_u16_trunc;
            if (out == OutKind::U32) return (rm == RoundMode::NearestEven) ? q_f32_u32_rne : q_f32_u32_trunc;
            return                 (rm == RoundMode::NearestEven) ? q_f32_u64_rne : q_f32_u64_trunc;
        } else {
            if (out == OutKind::U8)  return (rm == RoundMode::NearestEven) ? q_f32_u8_off_rne  : q_f32_u8_off_trunc;
            if (out == OutKind::U16) return (rm == RoundMode::NearestEven) ? q_f32_u16_off_rne : q_f32_u16_off_trunc;
            if (out == OutKind::U32) return (rm == RoundMode::NearestEven) ? q_f32_u32_off_rne : q_f32_u32_off_trunc;
            return                 (rm == RoundMode::NearestEven) ? q_f32_u64_off_rne : q_f32_u64_off_trunc;
        }
    }

    // f64
    if (in == InKind::F64) {
        if (!use_base) {
            if (out == OutKind::U8)  return (rm == RoundMode::NearestEven) ? q_f64_u8_rne  : q_f64_u8_trunc;
            if (out == OutKind::U16) return (rm == RoundMode::NearestEven) ? q_f64_u16_rne : q_f64_u16_trunc;
            if (out == OutKind::U32) return (rm == RoundMode::NearestEven) ? q_f64_u32_rne : q_f64_u32_trunc;
            return                 (rm == RoundMode::NearestEven) ? q_f64_u64_rne : q_f64_u64_trunc;
        } else {
            if (out == OutKind::U8)  return (rm == RoundMode::NearestEven) ? q_f64_u8_off_rne  : q_f64_u8_off_trunc;
            if (out == OutKind::U16) return (rm == RoundMode::NearestEven) ? q_f64_u16_off_rne : q_f64_u16_off_trunc;
            if (out == OutKind::U32) return (rm == RoundMode::NearestEven) ? q_f64_u32_off_rne : q_f64_u32_off_trunc;
            return                 (rm == RoundMode::NearestEven) ? q_f64_u64_off_rne : q_f64_u64_off_trunc;
        }
    }

    // u32
    if (in == InKind::U32) {
        if (!use_base) {
            if (out == OutKind::U8)  return q_u32_u8;
            if (out == OutKind::U16) return q_u32_u16;
            if (out == OutKind::U32) return q_u32_u32;
            return q_u32_u64;
        } else {
            if (out == OutKind::U8)  return q_u32_u8_off;
            if (out == OutKind::U16) return q_u32_u16_off;
            if (out == OutKind::U32) return q_u32_u32_off;
            return q_u32_u64_off;
        }
    }

    // u64
    if (in == InKind::U64) {
        if (!use_base) {
            if (out == OutKind::U8)  return q_u64_u8;
            if (out == OutKind::U16) return q_u64_u16;
            if (out == OutKind::U32) return q_u64_u32;
            return q_u64_u64;
        } else {
            if (out == OutKind::U8)  return q_u64_u8_off;
            if (out == OutKind::U16) return q_u64_u16_off;
            if (out == OutKind::U32) return q_u64_u32_off;
            return q_u64_u64_off;
        }
    }

    // i32
    if (in == InKind::I32) {
        if (!use_base) {
            if (out == OutKind::U8)  return q_i32_u8;
            if (out == OutKind::U16) return q_i32_u16;
            if (out == OutKind::U32) return q_i32_u32;
            return q_i32_u64;
        } else {
            if (out == OutKind::U8)  return q_i32_u8_off;
            if (out == OutKind::U16) return q_i32_u16_off;
            if (out == OutKind::U32) return q_i32_u32_off;
            return q_i32_u64_off;
        }
    }

    // i64
    if (in == InKind::I64) {
        if (!use_base) {
            if (out == OutKind::U8)  return q_i64_u8;
            if (out == OutKind::U16) return q_i64_u16;
            if (out == OutKind::U32) return q_i64_u32;
            return q_i64_u64;
        } else {
            if (out == OutKind::U8)  return q_i64_u8_off;
            if (out == OutKind::U16) return q_i64_u16_off;
            if (out == OutKind::U32) return q_i64_u32_off;
            return q_i64_u64_off;
        }
    }

    // Unspecified.
    return nullptr;
}

NODUS_FORCEINLINE QuantPlan make_plan(InKind in, OutKind out, RoundMode rm, bool use_base, std::uint64_t base) noexcept {
    QuantPlan p;
    p.in_kind  = in;
    p.out_kind = out;
    p.round    = rm;
    p.use_base = use_base;
    p.base     = base;
    p.fn       = select_quant_fn(in, out, rm, use_base);
    return p;
}

// --------------------------------------------
// Branchless bit slicing from byte stream (LE)
// --------------------------------------------
// Little-endian within bytes: bit 0 is LSB of byte 0.
// Contract (caller):
// - There must be at least 16 readable bytes starting at (p + (bitpos>>3)) on every read.
// - nbits must be 1..64 (nbits==0 is UB; nbits>64 is UB).
// - This is designed for “packed control streams” with external padding.

struct BitSlicerLE {
    const std::uint8_t* p = nullptr;
    std::size_t bitpos = 0;

    NODUS_FORCEINLINE explicit BitSlicerLE(const void* data) noexcept
        : p(static_cast<const std::uint8_t*>(data)), bitpos(0) {}

    NODUS_FORCEINLINE std::uint64_t read_u64(unsigned nbits) noexcept {
        const std::size_t byte_i = (bitpos >> 3);
        const unsigned shift = static_cast<unsigned>(bitpos & 7);

        const std::uint64_t w0 = load_u64_unaligned(p + byte_i);
        const std::uint64_t w1 = load_u64_unaligned(p + byte_i + 8);

        // Branchless combine using a mask to zero the hi contribution when shift==0.
        std::uint64_t hi = (w1 << ((64u - shift) & 63u));
        hi &= static_cast<std::uint64_t>(-static_cast<std::int64_t>(shift != 0));

        const std::uint64_t x = (w0 >> shift) | hi;

        std::uint64_t mask = ~0ull;
        mask >>= (64u - nbits); // valid for nbits in 1..64

        bitpos += nbits;
        return x & mask;
    }

    NODUS_FORCEINLINE void skip(unsigned nbits) noexcept {
        bitpos += nbits;
    }

    NODUS_FORCEINLINE std::size_t bits_read() const noexcept { return bitpos; }
};

} // namespace nodus::tensors::rounding
