#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "common/tensors/abstraction/abstract_tensor_handle.h"

namespace nodus::tensors {

enum class TensorDType : uint8_t {
    Unknown = 0,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    Ptr,
};

// Layout semantics for the tensor descriptor; backends may ignore Opaque.
enum class TensorLayout : uint8_t {
    Dense = 0,
    Strided = 1,
    Opaque = 2,
};

enum class TensorQuantize : uint8_t {
    RoundNearest = 0,
};

struct TensorSliceMeta {
    bool valid = false;
    TensorQuantize quantize = TensorQuantize::RoundNearest;
    std::vector<uint32_t> base_shape;
    std::vector<int64_t> start;
    std::vector<int64_t> step;
    std::vector<uint8_t> is_int;
    bool indexed = false;
    AbstractTensorHandle index_handle{};
    bool saturate = false;
    float saturate_threshold = 0.0f;
    bool has_affine = false;
    float affine[16] = {0};
};

// Canonical floating constants for diagnostics or kernel lowering.
inline constexpr double kTensorInf = std::numeric_limits<double>::infinity();
inline constexpr double kTensorNegInf = -std::numeric_limits<double>::infinity();
inline constexpr double kTensorNaN = std::numeric_limits<double>::quiet_NaN();

inline constexpr uint32_t tensor_dtype_size_bytes(TensorDType dtype) {
    switch (dtype) {
        case TensorDType::F32: return 4;
        case TensorDType::F64: return 8;
        case TensorDType::I8: return 1;
        case TensorDType::I16: return 2;
        case TensorDType::I32: return 4;
        case TensorDType::I64: return 8;
        case TensorDType::U8: return 1;
        case TensorDType::U16: return 2;
        case TensorDType::U32: return 4;
        case TensorDType::U64: return 8;
        case TensorDType::Bool: return 1;
        case TensorDType::Ptr: return static_cast<uint32_t>(sizeof(void*));
        case TensorDType::Unknown:
        default:
            return 0;
    }
}

struct TensorShape {
    std::vector<uint32_t> dims;

    uint32_t rank() const { return static_cast<uint32_t>(dims.size()); }
    uint64_t element_count() const {
        uint64_t total = 1;
        for (uint32_t d : dims) total *= static_cast<uint64_t>(d);
        return total;
    }
};

// Strides measured in elements (not bytes) per dimension.
struct TensorStrides {
    std::vector<uint64_t> elems;
};

// Primary schema used by backends and KernelIR lowering.
struct TensorDesc {
    TensorDType dtype = TensorDType::Unknown;
    TensorShape shape;
    TensorLayout layout = TensorLayout::Dense;
    TensorStrides strides;
    bool is_buffer = true;
    bool is_readonly = false;
    TensorSliceMeta slice;
};

} // namespace nodus::tensors
