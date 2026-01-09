#pragma once

#include <cstdint>
#include <cstddef>

namespace nodus::tensors {

struct VectorOpEval {
    float x = 0.0f;
    float y = 0.0f;
    float dx_du = 0.0f;
    float dy_du = 0.0f;
};

using VectorOpPositionFn = VectorOpEval (*)(const void* op_data, double u_src, double u_dst);
using VectorOpLengthFn = double (*)(const void* op_data);

struct VectorOperatorVTable {
    VectorOpPositionFn eval = nullptr;
    VectorOpLengthFn length = nullptr;
};

struct VectorOperator {
    const VectorOperatorVTable* vtable = nullptr;
    const void* data = nullptr;
};

inline bool vector_operator_valid(const VectorOperator& op) {
    return op.vtable && op.vtable->eval;
}

} // namespace nodus::tensors
