#pragma once

#include "common/tensors/abstraction/vector_ops/operator.h"

namespace nodus::tensors {

struct MoveOpData {
    float x = 0.0f;
    float y = 0.0f;
};

const VectorOperatorVTable* move_op_vtable();

} // namespace nodus::tensors
