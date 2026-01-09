#pragma once

#include "common/tensors/abstraction/vector_ops/operator.h"

namespace nodus::tensors {

struct LineOpData {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
};

const VectorOperatorVTable* line_op_vtable();

} // namespace nodus::tensors
