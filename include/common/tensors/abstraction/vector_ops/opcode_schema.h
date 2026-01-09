#pragma once

#include <cstdint>

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/coo_matrix.h"

namespace nodus::tensors {

// 16-bit opcodes with headroom for procedural/extended ops.
enum class VectorOpCode : uint16_t {
    Invalid = 0,
    MoveTo = 1,
    LineTo = 2,
    QuadTo = 3,
    CubicTo = 4,
    ClosePath = 5,
    CustomBase = 0x8000
};

// Dense opcode tensor + companion operand tables.
struct VectorOpTensorSchema {
    AbstractTensor opcodes;          // uint16: [op_count]
    AbstractTensor operands0;        // uint32: [op_count]
    AbstractTensor operands1;        // uint32: [op_count]
    AbstractTensor operands2;        // uint32: [op_count]
    AbstractTensor operands3;        // uint32: [op_count]
    AbstractTensor u_src;            // float: [op_count]
    AbstractTensor u_dst;            // float: [op_count]
    AbstractTensor output_index;     // uint32: [op_count] -> arena slot
};

// Sparse opcode tensor for sparse domains (e.g., coo layout).
struct SparseVectorOpTensorSchema {
    COOMatrix opcodes;               // values hold VectorOpCode, indices map to domain
    COOMatrix operands0;
    COOMatrix operands1;
    COOMatrix operands2;
    COOMatrix operands3;
    COOMatrix u_src;
    COOMatrix u_dst;
    COOMatrix output_index;
};

} // namespace nodus::tensors
