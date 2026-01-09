#include "common/tensors/abstraction/vector_ops/quad_op.h"

#include <cmath>

namespace nodus::tensors {

namespace {
VectorOpEval quad_eval(const void* op_data, double /*u_src*/, double u_dst) {
    const auto* data = static_cast<const QuadOpData*>(op_data);
    VectorOpEval out{};
    if (!data) return out;
    const double t = u_dst;
    const double omt = 1.0 - t;
    const double x0 = data->x0;
    const double y0 = data->y0;
    const double cx = data->cx;
    const double cy = data->cy;
    const double x1 = data->x1;
    const double y1 = data->y1;
    const double px = omt * omt * x0 + 2.0 * omt * t * cx + t * t * x1;
    const double py = omt * omt * y0 + 2.0 * omt * t * cy + t * t * y1;
    const double dx = 2.0 * omt * (cx - x0) + 2.0 * t * (x1 - cx);
    const double dy = 2.0 * omt * (cy - y0) + 2.0 * t * (y1 - cy);
    out.x = static_cast<float>(px);
    out.y = static_cast<float>(py);
    out.dx_du = static_cast<float>(dx);
    out.dy_du = static_cast<float>(dy);
    return out;
}

double quad_length(const void* op_data) {
    const auto* data = static_cast<const QuadOpData*>(op_data);
    if (!data) return 0.0;
    constexpr int kSamples = 16;
    double length = 0.0;
    double prev_x = data->x0;
    double prev_y = data->y0;
    for (int i = 1; i <= kSamples; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(kSamples);
        const double omt = 1.0 - t;
        const double x = omt * omt * data->x0 + 2.0 * omt * t * data->cx + t * t * data->x1;
        const double y = omt * omt * data->y0 + 2.0 * omt * t * data->cy + t * t * data->y1;
        const double dx = x - prev_x;
        const double dy = y - prev_y;
        length += std::sqrt(dx * dx + dy * dy);
        prev_x = x;
        prev_y = y;
    }
    return length;
}

constexpr VectorOperatorVTable kQuadVTable{&quad_eval, &quad_length};
} // namespace

const VectorOperatorVTable* quad_op_vtable() {
    return &kQuadVTable;
}

} // namespace nodus::tensors
