#include "common/tensors/abstraction/vector_ops/cubic_op.h"

#include <cmath>

namespace nodus::tensors {

namespace {
VectorOpEval cubic_eval(const void* op_data, double /*u_src*/, double u_dst) {
    const auto* data = static_cast<const CubicOpData*>(op_data);
    VectorOpEval out{};
    if (!data) return out;
    const double t = u_dst;
    const double omt = 1.0 - t;
    const double omt2 = omt * omt;
    const double t2 = t * t;
    const double x0 = data->x0;
    const double y0 = data->y0;
    const double x1 = data->x1;
    const double y1 = data->y1;
    const double cx0 = data->cx0;
    const double cy0 = data->cy0;
    const double cx1 = data->cx1;
    const double cy1 = data->cy1;
    const double px = omt2 * omt * x0
        + 3.0 * omt2 * t * cx0
        + 3.0 * omt * t2 * cx1
        + t2 * t * x1;
    const double py = omt2 * omt * y0
        + 3.0 * omt2 * t * cy0
        + 3.0 * omt * t2 * cy1
        + t2 * t * y1;
    const double dx = 3.0 * omt2 * (cx0 - x0)
        + 6.0 * omt * t * (cx1 - cx0)
        + 3.0 * t2 * (x1 - cx1);
    const double dy = 3.0 * omt2 * (cy0 - y0)
        + 6.0 * omt * t * (cy1 - cy0)
        + 3.0 * t2 * (y1 - cy1);
    out.x = static_cast<float>(px);
    out.y = static_cast<float>(py);
    out.dx_du = static_cast<float>(dx);
    out.dy_du = static_cast<float>(dy);
    return out;
}

double cubic_length(const void* op_data) {
    const auto* data = static_cast<const CubicOpData*>(op_data);
    if (!data) return 0.0;
    constexpr int kSamples = 24;
    double length = 0.0;
    double prev_x = data->x0;
    double prev_y = data->y0;
    for (int i = 1; i <= kSamples; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(kSamples);
        const double omt = 1.0 - t;
        const double omt2 = omt * omt;
        const double t2 = t * t;
        const double x = omt2 * omt * data->x0
            + 3.0 * omt2 * t * data->cx0
            + 3.0 * omt * t2 * data->cx1
            + t2 * t * data->x1;
        const double y = omt2 * omt * data->y0
            + 3.0 * omt2 * t * data->cy0
            + 3.0 * omt * t2 * data->cy1
            + t2 * t * data->y1;
        const double dx = x - prev_x;
        const double dy = y - prev_y;
        length += std::sqrt(dx * dx + dy * dy);
        prev_x = x;
        prev_y = y;
    }
    return length;
}

constexpr VectorOperatorVTable kCubicVTable{&cubic_eval, &cubic_length};
} // namespace

const VectorOperatorVTable* cubic_op_vtable() {
    return &kCubicVTable;
}

} // namespace nodus::tensors
