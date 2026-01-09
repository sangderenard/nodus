#include "common/tensors/abstraction/vector_ops/line_op.h"

#include <cmath>

namespace nodus::tensors {

namespace {
VectorOpEval line_eval(const void* op_data, double /*u_src*/, double u_dst) {
    const auto* data = static_cast<const LineOpData*>(op_data);
    VectorOpEval out{};
    if (!data) return out;
    const double t = u_dst;
    const double x0 = data->x0;
    const double y0 = data->y0;
    const double x1 = data->x1;
    const double y1 = data->y1;
    out.x = static_cast<float>((1.0 - t) * x0 + t * x1);
    out.y = static_cast<float>((1.0 - t) * y0 + t * y1);
    out.dx_du = static_cast<float>(x1 - x0);
    out.dy_du = static_cast<float>(y1 - y0);
    return out;
}

double line_length(const void* op_data) {
    const auto* data = static_cast<const LineOpData*>(op_data);
    if (!data) return 0.0;
    const double dx = static_cast<double>(data->x1) - static_cast<double>(data->x0);
    const double dy = static_cast<double>(data->y1) - static_cast<double>(data->y0);
    return std::sqrt(dx * dx + dy * dy);
}

constexpr VectorOperatorVTable kLineVTable{&line_eval, &line_length};
} // namespace

const VectorOperatorVTable* line_op_vtable() {
    return &kLineVTable;
}

} // namespace nodus::tensors
