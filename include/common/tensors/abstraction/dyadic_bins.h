// dyadic binning (tensor-backed)
#pragma once

#include "common/tensors/abstraction/abstract_tensor_pool.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/stream_add.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <climits>
#include <limits>
#include <type_traits>
#include <cstdio>
#include <typeinfo>

#include "common/tensors/abstraction/tensor_types.h"

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
        std::fprintf(stderr, "[dyadic] %s=%d\n", name, value ? 1 : 0);
    } else if constexpr (std::is_floating_point<T>::value) {
        std::fprintf(stderr, "[dyadic] %s=%g\n", name, static_cast<double>(value));
    } else if constexpr (std::is_signed<T>::value) {
        std::fprintf(stderr, "[dyadic] %s=%lld\n", name, static_cast<long long>(value));
    } else {
        std::fprintf(stderr, "[dyadic] %s=%llu\n", name, static_cast<unsigned long long>(value));
    }
}

inline void dyadic_log_tensor(const char* name, const AbstractTensor& t) {
    const TensorDesc& desc = t.desc();
    std::fprintf(
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
        std::fprintf(stderr, "%u", desc.shape.dims[i]);
        if (i + 1u < desc.shape.dims.size()) {
            std::fprintf(stderr, ",");
        }
    }
    std::fprintf(stderr, "]\n");
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
struct DyadicBinIO3 {
    uint8_t*  bin_data = nullptr;
    uint32_t* page_count = nullptr;
    uint8_t*  page_active = nullptr;
    uint8_t*  page_retain = nullptr;
    uint8_t*  page_inbox = nullptr;
    uint32_t  index_count = 0;
    uint32_t  elem_bytes = 0;
    uint32_t  slot_bytes = 0;
    uint32_t  value_stride = 1;
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


template <typename INDEX_T, typename VALUE_T, typename PremixPol>
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
        PremixPol::apply_span(dst, reinterpret_cast<const uint8_t*>(vals), io.elem_bytes, io.value_stride);
        return;
    }
    uint32_t b = dyadic_msb_pos_nonzero<INDEX_T>(hi);
    io.push_inbox(b, idx, vals);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
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
        PremixPol::apply_span(dst, val_bytes, io.elem_bytes, io.value_stride);
        return;
    }
    uint32_t b = dyadic_msb_pos_nonzero<INDEX_T>(hi);
    io.push_inbox_bytes(b, idx, val_bytes);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_cascade_phase_a_paged(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    uint32_t src_bin,
    VALUE_T* stage,
    uint32_t stage_bits) {
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
        dyadic_classify_to_stage_or_bin_paged_bytes<INDEX_T, VALUE_T, PremixPol>(io, stage, stage_bits, idx_move, val_bytes);
        slot += 1u;
    }
    io.finish_scan(src_bin);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_cascade_phase_b_paged(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    uint32_t src_bin,
    VALUE_T* stage,
    uint32_t stage_bits) {
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
        dyadic_classify_to_stage_or_bin_paged_bytes<INDEX_T, VALUE_T, PremixPol>(io, stage, stage_bits, idx2, val_bytes);
        slot += 1u;
    }
    io.finish_scan(src_bin);
}

template <typename INDEX_T, typename VALUE_T, typename PremixPol>
inline void dyadic_bin0_drain_paged(
    DyadicBinIO3<INDEX_T, VALUE_T>& io,
    VALUE_T* stage,
    uint32_t stage_bits) {
    const uint32_t bin = 0;
    io.prepare_for_scan(bin);
    const uint8_t A = io.active_page(bin);
    const uint32_t n = io.cnt(bin, A);
    const INDEX_T lo_mask = ((INDEX_T)1u << stage_bits) - 1u;
    uint8_t* base = io.page_base(bin, A);
    uint32_t slot = 0;
    while (slot ^ n) {
        uint8_t* p = base + (uint64_t)slot * (uint64_t)io.slot_bytes;
        INDEX_T idx;
        DYADIC_RAW_WRITE(&idx, p, sizeof(INDEX_T));
        const uint8_t* val_bytes = p + io.elem_bytes;
        const INDEX_T lo = (INDEX_T)(idx & lo_mask);
        VALUE_T* dst = stage + (uint64_t)lo * (uint64_t)io.value_stride;
        PremixPol::apply_span(dst, val_bytes, io.elem_bytes, io.value_stride);
        slot += 1u;
    }
    io.finish_scan(bin);
}

template <typename VALUE_T, typename OutmixPol>
inline void dyadic_emit_stage_linear(VALUE_T* output, uint32_t* output_offset, const VALUE_T* stage,
                                     uint32_t stage_bits, uint32_t value_stride) {
    const uint32_t stage_len = (uint32_t)(1u << stage_bits);
    const uint64_t total = (uint64_t)stage_len * (uint64_t)value_stride;
    VALUE_T* out = output + *output_offset;
    if constexpr (std::is_same_v<OutmixPol, policies::Overwrite>) {
        DYADIC_RAW_WRITE(out, stage, (uint32_t)(total * (uint64_t)sizeof(VALUE_T)));
    } else {
        OutmixPol::apply_span(out, reinterpret_cast<const uint8_t*>(stage),
                              (uint32_t)sizeof(VALUE_T), (uint32_t)total);
    }
    *output_offset = (uint32_t)(*output_offset + (uint32_t)total);
}

template <typename VALUE_T, typename OutmixPol>
inline bool dyadic_emit_stage_linear(AbstractTensor& output,
                                     uint32_t* output_offset,
                                     const VALUE_T* stage,
                                     uint32_t stage_bits,
                                     uint32_t value_stride) {
    if (!output.valid()) return false;
    auto* mem = dynamic_cast<InMemoryBackend*>(output.backend());
    if (!mem) return false;
    void* out_ptr = nullptr;
    size_t out_bytes = 0;
    if (!mem->map(output.handle(), &out_ptr, &out_bytes)) return false;
    dyadic_emit_stage_linear<VALUE_T, OutmixPol>(static_cast<VALUE_T*>(out_ptr), output_offset, stage, stage_bits, value_stride);
    mem->unmap(output.handle());
    return true;
}


#define DYADIC_BINNING_UINT_DEFINE(INDEX_T, VALUE_T) \
    template <typename PremixPol = policies::Add, typename OutmixPol = policies::Overwrite> \
    inline void dyadic_mt_bitmask_algo(INDEX_T index_range, uint32_t index_count, INDEX_T* indices, VALUE_T* values, \
                                       uint32_t value_stride, \
                                       uint32_t stage_bits, uint32_t thread_count, \
                                       uint32_t phase_a_rounds, uint32_t phase_b_rounds, uint32_t bin0_passes, \
                                       VALUE_T* output, uint32_t* output_offset, \
                                       AbstractTensor& bins, AbstractTensor& staging, AbstractTensor& counters, \
                                       AbstractTensor& page_active, AbstractTensor& page_retain, \
                                       AbstractTensor& page_inbox) { \
        InMemoryBackend* mem = static_cast<InMemoryBackend*>(bins.backend()); \
        void* bin_data_void = nullptr; \
        size_t bin_bytes = 0; \
        mem->map(bins.handle(), &bin_data_void, &bin_bytes); \
        (void)mem->ensure_zeroed(staging.handle(), staging.desc()); \
        (void)mem->ensure_zeroed(counters.handle(), counters.desc()); \
        void* stage_data_void = nullptr; \
        size_t stage_bytes = 0; \
        mem->map(staging.handle(), &stage_data_void, &stage_bytes); \
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
        (void)stage_data_void; \
        (void)stage_bytes; \
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
        io.value_stride = value_stride; \
        io.value_bytes = io.elem_bytes * value_stride; \
        io.slot_bytes = io.elem_bytes * (1u + value_stride); \
        io.total_bins = dyadic_bin_count(index_range, stage_bits); \
        VALUE_T* stage = reinterpret_cast<VALUE_T*>(stage_data_void); \
        uint32_t b = 0; \
        while (b ^ io.total_bins) { \
            io.page_active[b] = 0; \
            io.page_retain[b] = 1; \
            io.page_inbox[b] = 2; \
            b += 1u; \
        } \
        uint32_t i = 0; \
        while (i ^ index_count) { \
            const VALUE_T* vals = values + (uint64_t)i * (uint64_t)value_stride; \
            dyadic_classify_to_stage_or_bin_paged<INDEX_T, VALUE_T, PremixPol>(io, stage, stage_bits, indices[i], vals); \
            i += 1u; \
        } \
        uint32_t round_a = 0; \
        while (round_a ^ phase_a_rounds) { \
            for (uint32_t bin = io.total_bins; bin-- > 1u;) { \
                dyadic_cascade_phase_a_paged<INDEX_T, VALUE_T, PremixPol>(io, bin, stage, stage_bits); \
            } \
            round_a += 1u; \
        } \
        uint32_t round_b = 0; \
        while (round_b ^ phase_b_rounds) { \
            for (uint32_t bin = io.total_bins; bin-- > 1u;) { \
                dyadic_cascade_phase_b_paged<INDEX_T, VALUE_T, PremixPol>(io, bin, stage, stage_bits); \
            } \
            round_b += 1u; \
        } \
        uint32_t pass0 = 0; \
        while (pass0 ^ bin0_passes) { \
            dyadic_bin0_drain_paged<INDEX_T, VALUE_T, PremixPol>(io, stage, stage_bits); \
            pass0 += 1u; \
        } \
        dyadic_emit_stage_linear<VALUE_T, OutmixPol>(output, output_offset, stage, stage_bits, value_stride); \
        mem->unmap(bins.handle()); \
        mem->unmap(staging.handle()); \
        mem->unmap(counters.handle()); \
        mem->unmap(page_active.handle()); \
        mem->unmap(page_retain.handle()); \
        mem->unmap(page_inbox.handle()); \
    } \
    template <typename PremixPol = policies::Add, typename OutmixPol = policies::Overwrite> \
    inline bool dyadic_mt_bitmask_algo(INDEX_T index_range, uint32_t index_count, INDEX_T* indices, VALUE_T* values, \
                                       uint32_t value_stride, \
                                       uint32_t stage_bits, uint32_t thread_count, \
                                       uint32_t phase_a_rounds, uint32_t phase_b_rounds, uint32_t bin0_passes, \
                                       AbstractTensor& output, uint32_t* output_offset, \
                                       AbstractTensor& bins, AbstractTensor& staging, AbstractTensor& counters, \
                                       AbstractTensor& page_active, AbstractTensor& page_retain, \
                                       AbstractTensor& page_inbox) { \
        if (!output.valid()) return false; \
        auto* mem = dynamic_cast<InMemoryBackend*>(output.backend()); \
        if (!mem) return false; \
        void* out_ptr = nullptr; \
        size_t out_bytes = 0; \
        if (!mem->map(output.handle(), &out_ptr, &out_bytes)) return false; \
        dyadic_mt_bitmask_algo<PremixPol, OutmixPol>( \
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
            bins, \
            staging, \
            counters, \
            page_active, \
            page_retain, \
            page_inbox); \
        mem->unmap(output.handle()); \
        return true; \
    } \
    template <typename PremixPol = policies::Add, typename OutmixPol = policies::Overwrite> \
    inline bool dyadic_binning_tensor(INDEX_T index_range, uint32_t index_count, INDEX_T* indices, VALUE_T* values, \
                                      uint32_t value_stride, \
                                      uint32_t stage_bits, uint32_t thread_count, \
                                      uint32_t phase_a_rounds, uint32_t phase_b_rounds, uint32_t bin0_passes, \
                                      AbstractTensor& output, AbstractTensorPool& pool) { \
        if (!output.valid()) return false; \
        TensorBackend* backend = output.backend(); \
        if (!backend) return false; \
        const uint32_t total_bins = dyadic_bin_count(index_range, stage_bits); \
        TensorDesc desc_dyadic_bins{}; \
        desc_dyadic_bins.dtype = dyadic_uint_bytes_dtype<INDEX_T, VALUE_T>(); \
        desc_dyadic_bins.layout = TensorLayout::Dense; \
        desc_dyadic_bins.shape.dims = { (uint32_t)(total_bins * 3u), index_count, (uint32_t)(1u + value_stride) }; \
        TensorDesc desc_staging{}; \
        desc_staging.dtype = dyadic_value_dtype<VALUE_T>(); \
        desc_staging.layout = TensorLayout::Dense; \
        desc_staging.shape.dims = { (uint32_t)((1u << stage_bits) * value_stride) }; \
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
        AbstractTensor dyadic_bins_tensor = pool.acquire_tensor(desc_dyadic_bins, backend); \
        AbstractTensor staging_tensor = pool.acquire_tensor(desc_staging, backend); \
        AbstractTensor counters_tensor = pool.acquire_tensor(desc_counters, backend); \
        AbstractTensor page_active_tensor = pool.acquire_tensor(desc_page_active, backend); \
        AbstractTensor page_retain_tensor = pool.acquire_tensor(desc_page_retain, backend); \
        AbstractTensor page_inbox_tensor = pool.acquire_tensor(desc_page_inbox, backend); \
        uint32_t output_offset = 0; \
        return dyadic_mt_bitmask_algo<PremixPol, OutmixPol>( \
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
            dyadic_bins_tensor, \
            staging_tensor, \
            counters_tensor, \
            page_active_tensor, \
            page_retain_tensor, \
            page_inbox_tensor); \
    } \
        inline void dyadic_binning(INDEX_T index_range, uint32_t index_count, INDEX_T* indices, VALUE_T* values, \
                                   uint32_t value_stride, \
                       float_t dense_threshold, float_t* tile_dense_ratios, uint32_t thread_count, \
                       uint32_t phase_a_rounds, uint32_t phase_b_rounds, uint32_t bin0_passes, \
                       VALUE_T* output, uint32_t* output_offset) { \
        const uint32_t stage_bits = 8; \
        const uint32_t total_bins = dyadic_bin_count(index_range, stage_bits); \
        const uint32_t stage_bins = (uint32_t)(1u << stage_bits); \
        const uint32_t output_area = (uint32_t)index_range; \
        const bool should_use_dense = index_count >= output_area * dense_threshold; \
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
        desc_dyadic_bins.shape.dims = { (uint32_t)(total_bins * 3u), index_count, (uint32_t)(1u + value_stride) }; \
        TensorDesc desc_staging{}; \
        desc_staging.dtype = dyadic_value_dtype<VALUE_T>(); \
        desc_staging.layout = TensorLayout::Dense; \
        desc_staging.shape.dims = { (uint32_t)((1u << stage_bits) * value_stride) }; \
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
        AbstractTensor dyadic_bins_tensor = pool.acquire_tensor(desc_dyadic_bins); \
        AbstractTensor staging_tensor = pool.acquire_tensor(desc_staging); \
        AbstractTensor counters_tensor = pool.acquire_tensor(desc_counters); \
        AbstractTensor page_active_tensor = pool.acquire_tensor(desc_page_active); \
        AbstractTensor page_retain_tensor = pool.acquire_tensor(desc_page_retain); \
        AbstractTensor page_inbox_tensor = pool.acquire_tensor(desc_page_inbox); \
        dyadic_mt_bitmask_algo<policies::Add, policies::Overwrite>( \
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
            dyadic_bins_tensor, \
            staging_tensor, \
            counters_tensor, \
            page_active_tensor, \
            page_retain_tensor, \
            page_inbox_tensor); \
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