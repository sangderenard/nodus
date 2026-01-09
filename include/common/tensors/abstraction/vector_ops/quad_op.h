#pragma once

#include "common/tensors/abstraction/vector_ops/operator.h"

namespace nodus::tensors {

struct QuadOpData {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
};

const VectorOperatorVTable* quad_op_vtable();

} // namespace nodus::tensors
