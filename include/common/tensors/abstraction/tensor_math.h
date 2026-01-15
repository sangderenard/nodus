#pragma once

#include "common/tensors/abstraction/abstract_tensor.h"

#include <functional>
#include <vector>

namespace nodus::tensors {

namespace nodus {
class ThreadPool;
} // namespace nodus

using RowRangeFn = void (*)(const void* ctx, uint32_t y0, uint32_t y1);

nodus::ThreadPool* tensor_op_pool();
void submit_row_jobs(nodus::ThreadPool* pool,
                     RowRangeFn fn,
                     const void* ctx,
                     uint32_t y0,
                     uint32_t y1);

// Dense helpers for affine transforms and quaternions.
// Note: these are in-memory backend only and return empty tensors on mismatch.

#define NODUS_TENSOR_MATH_DECL(SUFFIX, SCALAR)                                      \
    AbstractTensor tensor_matmul_##SUFFIX(const AbstractTensor& a,                  \
                                          const AbstractTensor& b);                \
    AbstractTensor tensor_affine_identity_##SUFFIX(TensorBackend* backend = nullptr); \
    AbstractTensor tensor_affine_translation_##SUFFIX(const AbstractTensor& t);    \
    AbstractTensor tensor_affine_scale_##SUFFIX(const AbstractTensor& s);          \
    AbstractTensor tensor_affine_from_quat_translation_##SUFFIX(                   \
        const AbstractTensor& q,                                                   \
        const AbstractTensor& t);                                                  \
    AbstractTensor tensor_quat_identity_##SUFFIX(TensorBackend* backend = nullptr); \
    AbstractTensor tensor_quat_from_axis_angle_##SUFFIX(const AbstractTensor& axis, \
                                                        SCALAR angle);             \
    AbstractTensor tensor_quat_normalize_##SUFFIX(const AbstractTensor& q);        \
    AbstractTensor tensor_quat_mul_##SUFFIX(const AbstractTensor& a,                \
                                            const AbstractTensor& b);              \
    AbstractTensor tensor_quat_to_mat4_##SUFFIX(const AbstractTensor& q,            \
                                                const AbstractTensor& t);          \
    AbstractTensor tensor_transform_points_##SUFFIX(const AbstractTensor& points,  \
                                                     const AbstractTensor& mat4)

NODUS_TENSOR_MATH_DECL(f32, float);
NODUS_TENSOR_MATH_DECL(f64, double);

#undef NODUS_TENSOR_MATH_DECL

// Intersect beams with plane z = plane_z. Origins/dirs are [B,3] dense tensors.
bool tensor_intersect_plane_z_f32(const AbstractTensor& origins,
                                  const AbstractTensor& dirs,
                                  float plane_z,
                                  AbstractTensor* out_hits,
                                  AbstractTensor* out_mask);
bool tensor_intersect_plane_z_f64(const AbstractTensor& origins,
                                  const AbstractTensor& dirs,
                                  double plane_z,
                                  AbstractTensor* out_hits,
                                  AbstractTensor* out_mask);

// Scatter-add values into a dense 2D grid. points are [N,2] or [N,3] (x,y[,z]).
// base/out are [H,W] or [H,W,C], values are [N] or [N,C].
bool tensor_scatter_add_2d_f32(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
bool tensor_scatter_add_2d_f64(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
// General N-D scatter-add: points are [N, D] with D <= output rank.
// Output is [S0..S{D-1}] or [S0..S{D-1}, C]; values are [N] or [N, C].
bool tensor_scatter_add_nd_f32(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
bool tensor_scatter_add_nd_f64(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);

// Reduce along axis: sums every value across `axis` and writes result to `out`.
bool tensor_reduce_sum_axis_f32(const AbstractTensor& src, uint32_t axis, AbstractTensor* out);

// Affine scatter-add: applies a [4,4] transform to points before scatter.
bool tensor_scatter_add_2d_affine_f32(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& mat4,
                                      AbstractTensor* out,
                                      bool clamp = true);
bool tensor_scatter_add_2d_affine_f64(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& mat4,
                                      AbstractTensor* out,
                                      bool clamp = true);

using TensorKernelFnF32 = std::function<float(uint32_t k, uint32_t c, void* user)>;
using TensorKernelFnF64 = std::function<double(uint32_t k, uint32_t c, void* user)>;

struct TensorKernelPtrEntryF32 {
    using Fn = float (*)(uint32_t k, uint32_t c, void* user);
    Fn fn = nullptr;
    void* user = nullptr;
};

struct TensorKernelPtrEntryF64 {
    using Fn = double (*)(uint32_t k, uint32_t c, void* user);
    Fn fn = nullptr;
    void* user = nullptr;
};

struct KernelDispatchEntry {
    uint32_t id = 0;
    const void* fn = nullptr;
    void* user = nullptr;
};

struct KernelDispatchPlan {
    // Unique kernel function-pointer entries, assigned ids [0..entries.size()).
    std::vector<KernelDispatchEntry> entries;
    // Per-slot ids for [K,C] kernel slots (size = K*C).
    std::vector<uint32_t> slot_ids;
};

// Scatter-add values with a dense kernel (values[K] * kernel[K,C] -> out[C]).
bool tensor_scatter_add_2d_kernel_f32(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out,
                                      bool clamp = true);
bool tensor_scatter_add_2d_kernel_f64(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out,
                                      bool clamp = true);

// Kernel values from hook: fn(k, c, user) queried for each KxC entry.
bool tensor_scatter_add_2d_kernel_fn_f32(const AbstractTensor& base,
                                         const AbstractTensor& points,
                                         const AbstractTensor& values,
                                         const TensorKernelFnF32& kernel_fn,
                                         void* user,
                                         AbstractTensor* out,
                                         KernelDispatchPlan* out_plan,
                                         bool clamp = true);
bool tensor_scatter_add_2d_kernel_fn_f64(const AbstractTensor& base,
                                         const AbstractTensor& points,
                                         const AbstractTensor& values,
                                         const TensorKernelFnF64& kernel_fn,
                                         void* user,
                                         AbstractTensor* out,
                                         KernelDispatchPlan* out_plan,
                                         bool clamp = true);

// Kernel function pointers supplied via a [K,C] Ptr tensor of TensorKernelPtrEntry.
bool tensor_scatter_add_2d_kernel_ptr_f32(const AbstractTensor& base,
                                          const AbstractTensor& points,
                                          const AbstractTensor& values,
                                          const AbstractTensor& kernel_ptrs,
                                          AbstractTensor* out,
                                          KernelDispatchPlan* out_plan,
                                          bool clamp = true);
bool tensor_scatter_add_2d_kernel_ptr_f64(const AbstractTensor& base,
                                          const AbstractTensor& points,
                                          const AbstractTensor& values,
                                          const AbstractTensor& kernel_ptrs,
                                          AbstractTensor* out,
                                          KernelDispatchPlan* out_plan,
                                          bool clamp = true);

// Kernel bank: kernels are [B,K,C], kernel_ids are [N] (I32/U32).
bool tensor_scatter_add_2d_kernel_bank_f32(const AbstractTensor& base,
                                           const AbstractTensor& points,
                                           const AbstractTensor& values,
                                           const AbstractTensor& kernel_bank,
                                           const AbstractTensor& kernel_ids,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_scatter_add_2d_kernel_bank_f64(const AbstractTensor& base,
                                           const AbstractTensor& points,
                                           const AbstractTensor& values,
                                           const AbstractTensor& kernel_bank,
                                           const AbstractTensor& kernel_ids,
                                           AbstractTensor* out,
                                           bool clamp = true);

// Apply a 2D stencil kernel [S,S] to a dense field [H,W] -> [H,W].
AbstractTensor tensor_apply_stencil_2d_f32(const AbstractTensor& field,
                                           const AbstractTensor& kernel);
AbstractTensor tensor_apply_stencil_2d_f64(const AbstractTensor& field,
                                           const AbstractTensor& kernel);

// Apply a 2D stencil into an existing output tensor (avoids per-call allocation).
// out must be a valid dense tensor matching field's desc/shape and on the same backend.
bool tensor_apply_stencil_2d_f32_into(const AbstractTensor& field,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out);
bool tensor_apply_stencil_2d_f64_into(const AbstractTensor& field,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out);

// Copy src into dst (InMemoryBackend only). Supports dense/strided.
bool tensor_copy_f32_into(const AbstractTensor& src, AbstractTensor* dst);

// Dense elementwise out = alpha * a + beta * b (InMemoryBackend only).
// a, b, out must be dense F32 and have identical shapes.
bool tensor_axpby_f32(const AbstractTensor& a,
                      float alpha,
                      const AbstractTensor& b,
                      float beta,
                      AbstractTensor* out);

bool tensor_pack_rgba_from_film_f32(const AbstractTensor& film,
                                     AbstractTensor* rgba,
                                     uint32_t check_shift);

// Dtype-dispatched quaternion helpers.
AbstractTensor tensor_quat_normalize(const AbstractTensor& q);
AbstractTensor tensor_quat_mul(const AbstractTensor& a, const AbstractTensor& b);
AbstractTensor tensor_quat_from_axis_angle(const AbstractTensor& axis, double angle);

} // namespace nodus::tensors
