#include "common/tensors/abstraction/vector_ops/move_op.h"

namespace nodus::tensors {

namespace {
VectorOpEval move_eval(const void* op_data, double /*u_src*/, double /*u_dst*/) {
    const auto* data = static_cast<const MoveOpData*>(op_data);
    VectorOpEval out{};
    if (!data) return out;
    out.x = data->x;
    out.y = data->y;
    return out;
}

double move_length(const void* /*op_data*/) {
    return 0.0;
}

constexpr VectorOperatorVTable kMoveVTable{&move_eval, &move_length};
} // namespace

const VectorOperatorVTable* move_op_vtable() {
    return &kMoveVTable;
}

} // namespace nodus::tensors
