#pragma once

#include "common/tensors/abstraction/vector_ops/operator.h"

namespace nodus::tensors {

struct CubicOpData {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float cx0 = 0.0f;
    float cy0 = 0.0f;
    float cx1 = 0.0f;
    float cy1 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
};

const VectorOperatorVTable* cubic_op_vtable();

} // namespace nodus::tensors
