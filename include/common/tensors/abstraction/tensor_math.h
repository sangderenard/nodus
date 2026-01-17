#pragma once

#include "common/tensors/abstraction/abstract_tensor.h"

#include <functional>
#include <vector>

namespace nodus {
class ThreadPool;
} // namespace nodus

namespace nodus::tensors {

using RowRangeFn = void (*)(const void* ctx, uint32_t y0, uint32_t y1);

::nodus::ThreadPool* tensor_op_pool();
void submit_row_jobs(::nodus::ThreadPool* pool,
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

// Scatter (replace) values into a dense 2D grid. points are [N,2] or [N,3] (x,y[,z]).
// base/out are [H,W] or [H,W,C], values are [N] or [N,C].
bool tensor_scatter_2d_i8(const AbstractTensor& base,
                          const AbstractTensor& points,
                          const AbstractTensor& values,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_scatter_2d_i16(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_2d_i32(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_2d_i64(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_2d_u8(const AbstractTensor& base,
                          const AbstractTensor& points,
                          const AbstractTensor& values,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_scatter_2d_u16(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_2d_u32(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_2d_u64(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_2d_f32(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_2d_f64(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);

// Scatter-add values into a dense 2D grid. points are [N,2] or [N,3] (x,y[,z]).
// base/out are [H,W] or [H,W,C], values are [N] or [N,C].
bool tensor_scatter_add_2d_i8(const AbstractTensor& base,
                              const AbstractTensor& points,
                              const AbstractTensor& values,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_scatter_add_2d_i16(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
bool tensor_scatter_add_2d_i32(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
bool tensor_scatter_add_2d_i64(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
bool tensor_scatter_add_2d_u8(const AbstractTensor& base,
                              const AbstractTensor& points,
                              const AbstractTensor& values,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_scatter_add_2d_u16(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
bool tensor_scatter_add_2d_u32(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
bool tensor_scatter_add_2d_u64(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
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
// Gather values from a dense 2D grid. points are [N,2] or [N,3] (x,y[,z]).
// base is [H,W] or [H,W,C], out is [N] or [N,C].
bool tensor_gather_2d_i8(const AbstractTensor& base,
                         const AbstractTensor& points,
                         AbstractTensor* out,
                         bool clamp = true);
bool tensor_gather_2d_i16(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_2d_i32(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_2d_i64(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_2d_u8(const AbstractTensor& base,
                         const AbstractTensor& points,
                         AbstractTensor* out,
                         bool clamp = true);
bool tensor_gather_2d_u16(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_2d_u32(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_2d_u64(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_2d_f32(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_2d_f64(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
// Gather-add into an existing output tensor (out must be valid and correctly shaped).
bool tensor_gather_add_2d_i8(const AbstractTensor& base,
                             const AbstractTensor& points,
                             AbstractTensor* out,
                             bool clamp = true);
bool tensor_gather_add_2d_i16(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_2d_i32(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_2d_i64(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_2d_u8(const AbstractTensor& base,
                             const AbstractTensor& points,
                             AbstractTensor* out,
                             bool clamp = true);
bool tensor_gather_add_2d_u16(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_2d_u32(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_2d_u64(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_2d_f32(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_2d_f64(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
// General N-D scatter-add: points are [N, D] with D <= output rank.
// Output is [S0..S{D-1}] or [S0..S{D-1}, C]; values are [N] or [N, C].
bool tensor_scatter_add_nd_i8(const AbstractTensor& base,
                              const AbstractTensor& points,
                              const AbstractTensor& values,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_scatter_add_nd_i16(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
bool tensor_scatter_add_nd_i32(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
bool tensor_scatter_add_nd_i64(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
bool tensor_scatter_add_nd_u8(const AbstractTensor& base,
                              const AbstractTensor& points,
                              const AbstractTensor& values,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_scatter_add_nd_u16(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
bool tensor_scatter_add_nd_u32(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
bool tensor_scatter_add_nd_u64(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp = true);
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
// General N-D scatter (replace): points are [N, D] with D <= output rank.
bool tensor_scatter_nd_i8(const AbstractTensor& base,
                          const AbstractTensor& points,
                          const AbstractTensor& values,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_scatter_nd_i16(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_nd_i32(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_nd_i64(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_nd_u8(const AbstractTensor& base,
                          const AbstractTensor& points,
                          const AbstractTensor& values,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_scatter_nd_u16(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_nd_u32(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_nd_u64(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_nd_f32(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
bool tensor_scatter_nd_f64(const AbstractTensor& base,
                           const AbstractTensor& points,
                           const AbstractTensor& values,
                           AbstractTensor* out,
                           bool clamp = true);
// General N-D gather: points are [N, D] with D <= input rank.
// base is [S0..S{D-1}] or [S0..S{D-1}, C]; out is [N] or [N, C].
bool tensor_gather_nd_i8(const AbstractTensor& base,
                         const AbstractTensor& points,
                         AbstractTensor* out,
                         bool clamp = true);
bool tensor_gather_nd_i16(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_nd_i32(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_nd_i64(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_nd_u8(const AbstractTensor& base,
                         const AbstractTensor& points,
                         AbstractTensor* out,
                         bool clamp = true);
bool tensor_gather_nd_u16(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_nd_u32(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_nd_u64(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_nd_f32(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
bool tensor_gather_nd_f64(const AbstractTensor& base,
                          const AbstractTensor& points,
                          AbstractTensor* out,
                          bool clamp = true);
// Gather-add for N-D (out must be valid and correctly shaped).
bool tensor_gather_add_nd_i8(const AbstractTensor& base,
                             const AbstractTensor& points,
                             AbstractTensor* out,
                             bool clamp = true);
bool tensor_gather_add_nd_i16(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_nd_i32(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_nd_i64(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_nd_u8(const AbstractTensor& base,
                             const AbstractTensor& points,
                             AbstractTensor* out,
                             bool clamp = true);
bool tensor_gather_add_nd_u16(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_nd_u32(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_nd_u64(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_nd_f32(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);
bool tensor_gather_add_nd_f64(const AbstractTensor& base,
                              const AbstractTensor& points,
                              AbstractTensor* out,
                              bool clamp = true);

enum class StencilBoundaryMode : uint32_t {
    Zero = 0,
    Clamp = 1,
    Mirror = 2,
    Wrap = 3,
};

enum class StencilOrientation : uint32_t {
    // Industry standard default for stencil-apply APIs (no kernel flip).
    Correlation = 0,
    // Convolution flips offsets (equivalent to k(-dx,-dy)).
    Convolution = 1,
};

enum class TileContiguityStrategy : uint32_t {
    // Use stride order (fastest stride gets largest tile extent).
    Auto = 0,
    // Bias the fastest stride dimension explicitly.
    PreferFastestStride = 1,
    // Bias width (last dim) as most contiguous (row-major).
    PreferRowMajor = 2,
    // Bias height (first dim) as most contiguous (column-major).
    PreferColumnMajor = 3,
};

struct TileShape2D {
    uint32_t x = 0;
    uint32_t y = 0;
};

TileShape2D choose_tile_shape_2d(const TensorDesc& desc,
                                 uint32_t radius_x,
                                 uint32_t radius_y,
                                 uint64_t cache_budget_bytes,
                                 TileContiguityStrategy strategy);

// Naming conventions (avoid ambiguous "kernel"):
// - Footprint/PSF: weights over offsets describing spread-of-influence in output space.
// - ProbeKernel: weights over offsets used for local dot-product feature extraction.
// - Stencil: (offsets, weights) used in gather-style apply (one write per output).
// - Splat: scatter-style apply (fan-out writes of a footprint).

struct TensorSupport {
    std::vector<int32_t> min_offset;
    std::vector<int32_t> max_offset;
    std::vector<int32_t> radius;

    void reset() {
        min_offset.clear();
        max_offset.clear();
        radius.clear();
    }

    uint32_t dims() const { return static_cast<uint32_t>(radius.size()); }
};

struct TensorSpanRowSlice2D {
    int32_t dy = 0;
    uint32_t begin = 0;
    uint32_t end = 0;
    int32_t min_dx = 0;
    int32_t max_dx = 0;

    uint32_t count() const { return (end > begin) ? (end - begin) : 0u; }
};

struct TensorSpanOffsetCodex2D {
    std::vector<int32_t> dx;                 // dx per tap (sorted by dy, then dx)
    std::vector<float> weights_f32;          // optional weights (F32)
    std::vector<double> weights_f64;         // optional weights (F64)
    std::vector<TensorSpanRowSlice2D> rows;  // dy-sorted row slices
    TensorSupport support;
    TensorDType weight_dtype = TensorDType::Unknown;
    uint32_t taps = 0;

    void reset() {
        dx.clear();
        weights_f32.clear();
        weights_f64.clear();
        rows.clear();
        support.reset();
        weight_dtype = TensorDType::Unknown;
        taps = 0;
    }

    bool empty() const { return taps == 0 || dx.empty(); }
    bool has_weights() const { return weight_dtype == TensorDType::F32 || weight_dtype == TensorDType::F64; }
};

struct TensorFootprint2D {
    // Dense weights [K,K] in output space.
    AbstractTensor weights;
    TensorSupport support;
    mutable TensorSpanOffsetCodex2D span_codex;
};

struct TensorProbeKernel2D {
    // Sparse offsets [M,2] (I32) + weights [M].
    AbstractTensor offsets;
    AbstractTensor weights;
    TensorSupport support;
    mutable TensorSpanOffsetCodex2D span_codex;
};

struct TensorProbeKernel {
    // Sparse offsets [M,D] (I32) + weights [M].
    AbstractTensor offsets;
    AbstractTensor weights;
    TensorSupport support;
    mutable TensorSpanOffsetCodex2D span_codex;
};

struct TensorStencil {
    // Sparse offsets [M,D] (I32) + weights [M].
    AbstractTensor offsets;
    AbstractTensor weights;
    TensorSupport support;
    StencilOrientation orientation = StencilOrientation::Correlation;
    mutable TensorSpanOffsetCodex2D span_codex;
};

struct TensorCSR {
    // Row pointers [rows+1] I32, column indices [E] I32 (GLI), edge weights [E].
    AbstractTensor row_ptr;
    AbstractTensor col_gli;
    AbstractTensor edge_weight;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

// GLI helpers (Global Linear Index) for slice-aware index math.
// GLI is computed in logical row-major order on the tensor's shape.
bool tensor_gli_from_coords(const TensorDesc& desc,
                            const std::vector<uint32_t>& coords,
                            uint64_t* out_gli);
// Compute element offset from GLI using desc strides (respects strided views).
bool tensor_elem_offset_from_gli(const TensorDesc& desc,
                                 uint64_t gli,
                                 uint64_t* out_elem_offset);
// Compute byte offset from GLI using desc strides and dtype size.
bool tensor_byte_offset_from_gli(const TensorDesc& desc,
                                 uint64_t gli,
                                 uint64_t* out_byte_offset);

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

template <typename Scalar>
using TensorKernelFn = std::function<Scalar(uint32_t k, uint32_t c, void* user)>;

using TensorKernelFnU8 = TensorKernelFn<uint8_t>;
using TensorKernelFnU16 = TensorKernelFn<uint16_t>;
using TensorKernelFnU32 = TensorKernelFn<uint32_t>;
using TensorKernelFnU64 = TensorKernelFn<uint64_t>;
using TensorKernelFnI8 = TensorKernelFn<int8_t>;
using TensorKernelFnI16 = TensorKernelFn<int16_t>;
using TensorKernelFnI32 = TensorKernelFn<int32_t>;
using TensorKernelFnI64 = TensorKernelFn<int64_t>;
using TensorKernelFnF32 = TensorKernelFn<float>;
using TensorKernelFnF64 = TensorKernelFn<double>;

template <typename Scalar>
struct TensorKernelPtrEntry {
    using Fn = Scalar (*)(uint32_t k, uint32_t c, void* user);
    Fn fn = nullptr;
    void* user = nullptr;
};

using TensorKernelPtrEntryU8 = TensorKernelPtrEntry<uint8_t>;
using TensorKernelPtrEntryU16 = TensorKernelPtrEntry<uint16_t>;
using TensorKernelPtrEntryU32 = TensorKernelPtrEntry<uint32_t>;
using TensorKernelPtrEntryU64 = TensorKernelPtrEntry<uint64_t>;
using TensorKernelPtrEntryI8 = TensorKernelPtrEntry<int8_t>;
using TensorKernelPtrEntryI16 = TensorKernelPtrEntry<int16_t>;
using TensorKernelPtrEntryI32 = TensorKernelPtrEntry<int32_t>;
using TensorKernelPtrEntryI64 = TensorKernelPtrEntry<int64_t>;
using TensorKernelPtrEntryF32 = TensorKernelPtrEntry<float>;
using TensorKernelPtrEntryF64 = TensorKernelPtrEntry<double>;

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

// Build a stencil (offsets + weights) from a dense footprint (PSF).
// Footprint weights are [K,K] dense; stencil offsets are [M,2] I32, weights [M] F32/F64.
// Use Correlation for industry-standard stencil application.
bool tensor_build_stencil_from_footprint_2d_f32(const TensorFootprint2D& footprint,
                                                StencilOrientation orientation,
                                                TensorStencil* out_stencil);
bool tensor_build_stencil_from_footprint_2d_f64(const TensorFootprint2D& footprint,
                                                StencilOrientation orientation,
                                                TensorStencil* out_stencil);

// Apply a gather stencil built from a footprint (PSF).
// Apply a gather footprint and add into an existing output tensor.
#define NODUS_TENSOR_FOOTPRINT_2D_DECL(SUFFIX)                                     \
    bool tensor_gather_footprint_2d_##SUFFIX(const AbstractTensor& base,          \
                                             const TensorFootprint2D& footprint, \
                                             AbstractTensor* out,                 \
                                             StencilOrientation orientation = StencilOrientation::Correlation, \
                                             StencilBoundaryMode boundary = StencilBoundaryMode::Zero, \
                                             const AbstractTensor* target_mask = nullptr, \
                                             const AbstractTensor* source_mask = nullptr, \
                                             bool normalize = false);             \
    bool tensor_gather_add_footprint_2d_##SUFFIX(const AbstractTensor& base,      \
                                                 const TensorFootprint2D& footprint, \
                                                 AbstractTensor* out,             \
                                                 StencilOrientation orientation = StencilOrientation::Correlation, \
                                                 StencilBoundaryMode boundary = StencilBoundaryMode::Zero, \
                                                 const AbstractTensor* target_mask = nullptr, \
                                                 const AbstractTensor* source_mask = nullptr, \
                                                 bool normalize = false);          \
    bool tensor_scatter_footprint_2d_##SUFFIX(const AbstractTensor& base,         \
                                              const AbstractTensor& points,      \
                                              const AbstractTensor& values,      \
                                              const TensorFootprint2D& footprint, \
                                              AbstractTensor* out,               \
                                              StencilOrientation orientation = StencilOrientation::Correlation, \
                                              StencilBoundaryMode boundary = StencilBoundaryMode::Zero, \
                                              bool clamp = true)

NODUS_TENSOR_FOOTPRINT_2D_DECL(u8);
NODUS_TENSOR_FOOTPRINT_2D_DECL(u16);
NODUS_TENSOR_FOOTPRINT_2D_DECL(u32);
NODUS_TENSOR_FOOTPRINT_2D_DECL(u64);
NODUS_TENSOR_FOOTPRINT_2D_DECL(i8);
NODUS_TENSOR_FOOTPRINT_2D_DECL(i16);
NODUS_TENSOR_FOOTPRINT_2D_DECL(i32);
NODUS_TENSOR_FOOTPRINT_2D_DECL(i64);
NODUS_TENSOR_FOOTPRINT_2D_DECL(f32);
NODUS_TENSOR_FOOTPRINT_2D_DECL(f64);

#undef NODUS_TENSOR_FOOTPRINT_2D_DECL

// Derive support from offsets [M,D] (I32). Fills min/max/radius per dimension.
bool tensor_support_from_offsets(const AbstractTensor& offsets, TensorSupport* out_support);

// Apply a gather stencil given offsets+weights.
// base is [H,W] or [H,W,C], out matches base shape.
bool tensor_gather_stencil_2d_f32(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
// Apply a gather stencil and add into an existing output tensor.
bool tensor_gather_add_stencil_2d_f32(const AbstractTensor& base,
                                      const TensorStencil& stencil,
                                      AbstractTensor* out,
                                      StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                      const AbstractTensor* target_mask = nullptr,
                                      const AbstractTensor* source_mask = nullptr,
                                      bool normalize = false);
bool tensor_gather_stencil_2d_f64(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
// Apply a gather stencil and add into an existing output tensor.
bool tensor_gather_add_stencil_2d_f64(const AbstractTensor& base,
                                      const TensorStencil& stencil,
                                      AbstractTensor* out,
                                      StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                      const AbstractTensor* target_mask = nullptr,
                                      const AbstractTensor* source_mask = nullptr,
                                      bool normalize = false);

bool tensor_gather_stencil_2d_i8(const AbstractTensor& base,
                                 const TensorStencil& stencil,
                                 AbstractTensor* out,
                                 StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                 const AbstractTensor* target_mask = nullptr,
                                 const AbstractTensor* source_mask = nullptr,
                                 bool normalize = false);
bool tensor_gather_stencil_2d_i16(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
bool tensor_gather_stencil_2d_i32(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
bool tensor_gather_stencil_2d_i64(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
bool tensor_gather_stencil_2d_u8(const AbstractTensor& base,
                                 const TensorStencil& stencil,
                                 AbstractTensor* out,
                                 StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                 const AbstractTensor* target_mask = nullptr,
                                 const AbstractTensor* source_mask = nullptr,
                                 bool normalize = false);
bool tensor_gather_stencil_2d_u16(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
bool tensor_gather_stencil_2d_u32(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
bool tensor_gather_stencil_2d_u64(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);

// Apply a gather stencil in N-D given offsets+weights.
// base is [S0..S{D-1}] or [S0..S{D-1}, C], offsets are [M,D] I32, weights are [M].
// out matches base shape.
bool tensor_gather_stencil_nd_f32(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
// Apply a gather stencil (N-D) and add into an existing output tensor.
bool tensor_gather_add_stencil_nd_f32(const AbstractTensor& base,
                                      const TensorStencil& stencil,
                                      AbstractTensor* out,
                                      StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                      const AbstractTensor* target_mask = nullptr,
                                      const AbstractTensor* source_mask = nullptr,
                                      bool normalize = false);
bool tensor_gather_stencil_nd_f64(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
bool tensor_gather_stencil_nd_i8(const AbstractTensor& base,
                                 const TensorStencil& stencil,
                                 AbstractTensor* out,
                                 StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                 const AbstractTensor* target_mask = nullptr,
                                 const AbstractTensor* source_mask = nullptr,
                                 bool normalize = false);
bool tensor_gather_stencil_nd_i16(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
bool tensor_gather_stencil_nd_i32(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
bool tensor_gather_stencil_nd_i64(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
bool tensor_gather_stencil_nd_u8(const AbstractTensor& base,
                                 const TensorStencil& stencil,
                                 AbstractTensor* out,
                                 StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                 const AbstractTensor* target_mask = nullptr,
                                 const AbstractTensor* source_mask = nullptr,
                                 bool normalize = false);
bool tensor_gather_stencil_nd_u16(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
bool tensor_gather_stencil_nd_u32(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
bool tensor_gather_stencil_nd_u64(const AbstractTensor& base,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  const AbstractTensor* target_mask = nullptr,
                                  const AbstractTensor* source_mask = nullptr,
                                  bool normalize = false);
// Apply a gather stencil (N-D) and add into an existing output tensor.
bool tensor_gather_add_stencil_nd_f64(const AbstractTensor& base,
                                      const TensorStencil& stencil,
                                      AbstractTensor* out,
                                      StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                      const AbstractTensor* target_mask = nullptr,
                                      const AbstractTensor* source_mask = nullptr,
                                      bool normalize = false);

// Scatter/splat using a stencil (fan-out writes).
bool tensor_scatter_stencil_2d_f32(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_2d_f64(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_2d_i8(const AbstractTensor& base,
                                  const AbstractTensor& points,
                                  const AbstractTensor& values,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  bool clamp = true);
bool tensor_scatter_stencil_2d_i16(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_2d_i32(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_2d_i64(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_2d_u8(const AbstractTensor& base,
                                  const AbstractTensor& points,
                                  const AbstractTensor& values,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  bool clamp = true);
bool tensor_scatter_stencil_2d_u16(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_2d_u32(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_2d_u64(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);

bool tensor_scatter_stencil_nd_f32(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_nd_f64(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_nd_i8(const AbstractTensor& base,
                                  const AbstractTensor& points,
                                  const AbstractTensor& values,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  bool clamp = true);
bool tensor_scatter_stencil_nd_i16(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_nd_i32(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_nd_i64(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_nd_u8(const AbstractTensor& base,
                                  const AbstractTensor& points,
                                  const AbstractTensor& values,
                                  const TensorStencil& stencil,
                                  AbstractTensor* out,
                                  StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                  bool clamp = true);
bool tensor_scatter_stencil_nd_u16(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_nd_u32(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);
bool tensor_scatter_stencil_nd_u64(const AbstractTensor& base,
                                   const AbstractTensor& points,
                                   const AbstractTensor& values,
                                   const TensorStencil& stencil,
                                   AbstractTensor* out,
                                   StencilBoundaryMode boundary = StencilBoundaryMode::Zero,
                                   bool clamp = true);

// Scatter/splat using a footprint (PSF) by first building a stencil.

// CSR-based scatter/gather kernels (GLI indices, edge weights kept).
bool tensor_scatter_probe_csr_i8(const AbstractTensor& base,
                                 const TensorProbeKernel& probe,
                                 const TensorCSR& targets_per_center,
                                 AbstractTensor* out,
                                 bool clamp = true);
bool tensor_scatter_probe_csr_i16(const AbstractTensor& base,
                                  const TensorProbeKernel& probe,
                                  const TensorCSR& targets_per_center,
                                  AbstractTensor* out,
                                  bool clamp = true);
bool tensor_scatter_probe_csr_i32(const AbstractTensor& base,
                                  const TensorProbeKernel& probe,
                                  const TensorCSR& targets_per_center,
                                  AbstractTensor* out,
                                  bool clamp = true);
bool tensor_scatter_probe_csr_i64(const AbstractTensor& base,
                                  const TensorProbeKernel& probe,
                                  const TensorCSR& targets_per_center,
                                  AbstractTensor* out,
                                  bool clamp = true);
bool tensor_scatter_probe_csr_u8(const AbstractTensor& base,
                                 const TensorProbeKernel& probe,
                                 const TensorCSR& targets_per_center,
                                 AbstractTensor* out,
                                 bool clamp = true);
bool tensor_scatter_probe_csr_u16(const AbstractTensor& base,
                                  const TensorProbeKernel& probe,
                                  const TensorCSR& targets_per_center,
                                  AbstractTensor* out,
                                  bool clamp = true);
bool tensor_scatter_probe_csr_u32(const AbstractTensor& base,
                                  const TensorProbeKernel& probe,
                                  const TensorCSR& targets_per_center,
                                  AbstractTensor* out,
                                  bool clamp = true);
bool tensor_scatter_probe_csr_u64(const AbstractTensor& base,
                                  const TensorProbeKernel& probe,
                                  const TensorCSR& targets_per_center,
                                  AbstractTensor* out,
                                  bool clamp = true);
bool tensor_scatter_probe_csr_f32(const AbstractTensor& base,
                                  const TensorProbeKernel& probe,
                                  const TensorCSR& targets_per_center,
                                  AbstractTensor* out,
                                  bool clamp = true);
bool tensor_scatter_probe_csr_f64(const AbstractTensor& base,
                                  const TensorProbeKernel& probe,
                                  const TensorCSR& targets_per_center,
                                  AbstractTensor* out,
                                  bool clamp = true);
bool tensor_gather_probe_csr_i8(const AbstractTensor& base,
                                const TensorProbeKernel& probe,
                                const TensorCSR& centers_per_target,
                                AbstractTensor* out,
                                bool clamp = true);
bool tensor_gather_probe_csr_i16(const AbstractTensor& base,
                                 const TensorProbeKernel& probe,
                                 const TensorCSR& centers_per_target,
                                 AbstractTensor* out,
                                 bool clamp = true);
bool tensor_gather_probe_csr_i32(const AbstractTensor& base,
                                 const TensorProbeKernel& probe,
                                 const TensorCSR& centers_per_target,
                                 AbstractTensor* out,
                                 bool clamp = true);
bool tensor_gather_probe_csr_i64(const AbstractTensor& base,
                                 const TensorProbeKernel& probe,
                                 const TensorCSR& centers_per_target,
                                 AbstractTensor* out,
                                 bool clamp = true);
bool tensor_gather_probe_csr_u8(const AbstractTensor& base,
                                const TensorProbeKernel& probe,
                                const TensorCSR& centers_per_target,
                                AbstractTensor* out,
                                bool clamp = true);
bool tensor_gather_probe_csr_u16(const AbstractTensor& base,
                                 const TensorProbeKernel& probe,
                                 const TensorCSR& centers_per_target,
                                 AbstractTensor* out,
                                 bool clamp = true);
bool tensor_gather_probe_csr_u32(const AbstractTensor& base,
                                 const TensorProbeKernel& probe,
                                 const TensorCSR& centers_per_target,
                                 AbstractTensor* out,
                                 bool clamp = true);
bool tensor_gather_probe_csr_u64(const AbstractTensor& base,
                                 const TensorProbeKernel& probe,
                                 const TensorCSR& centers_per_target,
                                 AbstractTensor* out,
                                 bool clamp = true);
bool tensor_gather_probe_csr_f32(const AbstractTensor& base,
                                 const TensorProbeKernel& probe,
                                 const TensorCSR& centers_per_target,
                                 AbstractTensor* out,
                                 bool clamp = true);
bool tensor_gather_probe_csr_f64(const AbstractTensor& base,
                                 const TensorProbeKernel& probe,
                                 const TensorCSR& centers_per_target,
                                 AbstractTensor* out,
                                 bool clamp = true);

// CSR scatter/gather using a stencil or footprint (probe kernel derived from stencil).
bool tensor_scatter_probe_csr_stencil_i8(const AbstractTensor& base,
                                         const TensorStencil& stencil,
                                         const TensorCSR& targets_per_center,
                                         AbstractTensor* out,
                                         bool clamp = true);
bool tensor_scatter_probe_csr_stencil_i16(const AbstractTensor& base,
                                          const TensorStencil& stencil,
                                          const TensorCSR& targets_per_center,
                                          AbstractTensor* out,
                                          bool clamp = true);
bool tensor_scatter_probe_csr_stencil_i32(const AbstractTensor& base,
                                          const TensorStencil& stencil,
                                          const TensorCSR& targets_per_center,
                                          AbstractTensor* out,
                                          bool clamp = true);
bool tensor_scatter_probe_csr_stencil_i64(const AbstractTensor& base,
                                          const TensorStencil& stencil,
                                          const TensorCSR& targets_per_center,
                                          AbstractTensor* out,
                                          bool clamp = true);
bool tensor_scatter_probe_csr_stencil_u8(const AbstractTensor& base,
                                         const TensorStencil& stencil,
                                         const TensorCSR& targets_per_center,
                                         AbstractTensor* out,
                                         bool clamp = true);
bool tensor_scatter_probe_csr_stencil_u16(const AbstractTensor& base,
                                          const TensorStencil& stencil,
                                          const TensorCSR& targets_per_center,
                                          AbstractTensor* out,
                                          bool clamp = true);
bool tensor_scatter_probe_csr_stencil_u32(const AbstractTensor& base,
                                          const TensorStencil& stencil,
                                          const TensorCSR& targets_per_center,
                                          AbstractTensor* out,
                                          bool clamp = true);
bool tensor_scatter_probe_csr_stencil_u64(const AbstractTensor& base,
                                          const TensorStencil& stencil,
                                          const TensorCSR& targets_per_center,
                                          AbstractTensor* out,
                                          bool clamp = true);
bool tensor_scatter_probe_csr_stencil_f32(const AbstractTensor& base,
                                          const TensorStencil& stencil,
                                          const TensorCSR& targets_per_center,
                                          AbstractTensor* out,
                                          bool clamp = true);
bool tensor_scatter_probe_csr_stencil_f64(const AbstractTensor& base,
                                          const TensorStencil& stencil,
                                          const TensorCSR& targets_per_center,
                                          AbstractTensor* out,
                                          bool clamp = true);
bool tensor_gather_probe_csr_stencil_i8(const AbstractTensor& base,
                                        const TensorStencil& stencil,
                                        const TensorCSR& centers_per_target,
                                        AbstractTensor* out,
                                        bool clamp = true);
bool tensor_gather_probe_csr_stencil_i16(const AbstractTensor& base,
                                         const TensorStencil& stencil,
                                         const TensorCSR& centers_per_target,
                                         AbstractTensor* out,
                                         bool clamp = true);
bool tensor_gather_probe_csr_stencil_i32(const AbstractTensor& base,
                                         const TensorStencil& stencil,
                                         const TensorCSR& centers_per_target,
                                         AbstractTensor* out,
                                         bool clamp = true);
bool tensor_gather_probe_csr_stencil_i64(const AbstractTensor& base,
                                         const TensorStencil& stencil,
                                         const TensorCSR& centers_per_target,
                                         AbstractTensor* out,
                                         bool clamp = true);
bool tensor_gather_probe_csr_stencil_u8(const AbstractTensor& base,
                                        const TensorStencil& stencil,
                                        const TensorCSR& centers_per_target,
                                        AbstractTensor* out,
                                        bool clamp = true);
bool tensor_gather_probe_csr_stencil_u16(const AbstractTensor& base,
                                         const TensorStencil& stencil,
                                         const TensorCSR& centers_per_target,
                                         AbstractTensor* out,
                                         bool clamp = true);
bool tensor_gather_probe_csr_stencil_u32(const AbstractTensor& base,
                                         const TensorStencil& stencil,
                                         const TensorCSR& centers_per_target,
                                         AbstractTensor* out,
                                         bool clamp = true);
bool tensor_gather_probe_csr_stencil_u64(const AbstractTensor& base,
                                         const TensorStencil& stencil,
                                         const TensorCSR& centers_per_target,
                                         AbstractTensor* out,
                                         bool clamp = true);
bool tensor_gather_probe_csr_stencil_f32(const AbstractTensor& base,
                                         const TensorStencil& stencil,
                                         const TensorCSR& centers_per_target,
                                         AbstractTensor* out,
                                         bool clamp = true);
bool tensor_gather_probe_csr_stencil_f64(const AbstractTensor& base,
                                         const TensorStencil& stencil,
                                         const TensorCSR& centers_per_target,
                                         AbstractTensor* out,
                                         bool clamp = true);

bool tensor_scatter_probe_csr_footprint_f32(const AbstractTensor& base,
                                            const TensorFootprint2D& footprint,
                                            StencilOrientation orientation,
                                            const TensorCSR& targets_per_center,
                                            AbstractTensor* out,
                                            bool clamp = true);
bool tensor_scatter_probe_csr_footprint_f64(const AbstractTensor& base,
                                            const TensorFootprint2D& footprint,
                                            StencilOrientation orientation,
                                            const TensorCSR& targets_per_center,
                                            AbstractTensor* out,
                                            bool clamp = true);
bool tensor_scatter_probe_csr_footprint_i8(const AbstractTensor& base,
                                           const TensorFootprint2D& footprint,
                                           StencilOrientation orientation,
                                           const TensorCSR& targets_per_center,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_scatter_probe_csr_footprint_i16(const AbstractTensor& base,
                                            const TensorFootprint2D& footprint,
                                            StencilOrientation orientation,
                                            const TensorCSR& targets_per_center,
                                            AbstractTensor* out,
                                            bool clamp = true);
bool tensor_scatter_probe_csr_footprint_i32(const AbstractTensor& base,
                                            const TensorFootprint2D& footprint,
                                            StencilOrientation orientation,
                                            const TensorCSR& targets_per_center,
                                            AbstractTensor* out,
                                            bool clamp = true);
bool tensor_scatter_probe_csr_footprint_i64(const AbstractTensor& base,
                                            const TensorFootprint2D& footprint,
                                            StencilOrientation orientation,
                                            const TensorCSR& targets_per_center,
                                            AbstractTensor* out,
                                            bool clamp = true);
bool tensor_scatter_probe_csr_footprint_u8(const AbstractTensor& base,
                                           const TensorFootprint2D& footprint,
                                           StencilOrientation orientation,
                                           const TensorCSR& targets_per_center,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_scatter_probe_csr_footprint_u16(const AbstractTensor& base,
                                            const TensorFootprint2D& footprint,
                                            StencilOrientation orientation,
                                            const TensorCSR& targets_per_center,
                                            AbstractTensor* out,
                                            bool clamp = true);
bool tensor_scatter_probe_csr_footprint_u32(const AbstractTensor& base,
                                            const TensorFootprint2D& footprint,
                                            StencilOrientation orientation,
                                            const TensorCSR& targets_per_center,
                                            AbstractTensor* out,
                                            bool clamp = true);
bool tensor_scatter_probe_csr_footprint_u64(const AbstractTensor& base,
                                            const TensorFootprint2D& footprint,
                                            StencilOrientation orientation,
                                            const TensorCSR& targets_per_center,
                                            AbstractTensor* out,
                                            bool clamp = true);
bool tensor_gather_probe_csr_footprint_i8(const AbstractTensor& base,
                                          const TensorFootprint2D& footprint,
                                          StencilOrientation orientation,
                                          const TensorCSR& centers_per_target,
                                          AbstractTensor* out,
                                          bool clamp = true);
bool tensor_gather_probe_csr_footprint_i16(const AbstractTensor& base,
                                           const TensorFootprint2D& footprint,
                                           StencilOrientation orientation,
                                           const TensorCSR& centers_per_target,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_gather_probe_csr_footprint_i32(const AbstractTensor& base,
                                           const TensorFootprint2D& footprint,
                                           StencilOrientation orientation,
                                           const TensorCSR& centers_per_target,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_gather_probe_csr_footprint_i64(const AbstractTensor& base,
                                           const TensorFootprint2D& footprint,
                                           StencilOrientation orientation,
                                           const TensorCSR& centers_per_target,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_gather_probe_csr_footprint_u8(const AbstractTensor& base,
                                          const TensorFootprint2D& footprint,
                                          StencilOrientation orientation,
                                          const TensorCSR& centers_per_target,
                                          AbstractTensor* out,
                                          bool clamp = true);
bool tensor_gather_probe_csr_footprint_u16(const AbstractTensor& base,
                                           const TensorFootprint2D& footprint,
                                           StencilOrientation orientation,
                                           const TensorCSR& centers_per_target,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_gather_probe_csr_footprint_u32(const AbstractTensor& base,
                                           const TensorFootprint2D& footprint,
                                           StencilOrientation orientation,
                                           const TensorCSR& centers_per_target,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_gather_probe_csr_footprint_u64(const AbstractTensor& base,
                                           const TensorFootprint2D& footprint,
                                           StencilOrientation orientation,
                                           const TensorCSR& centers_per_target,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_gather_probe_csr_footprint_f32(const AbstractTensor& base,
                                           const TensorFootprint2D& footprint,
                                           StencilOrientation orientation,
                                           const TensorCSR& centers_per_target,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_gather_probe_csr_footprint_f64(const AbstractTensor& base,
                                           const TensorFootprint2D& footprint,
                                           StencilOrientation orientation,
                                           const TensorCSR& centers_per_target,
                                           AbstractTensor* out,
                                           bool clamp = true);

// Scatter-add values with a dense kernel (values[K] * kernel[K,C] -> out[C]).
bool tensor_scatter_add_2d_kernel_u8(const AbstractTensor& base,
                                     const AbstractTensor& points,
                                     const AbstractTensor& values,
                                     const AbstractTensor& kernel,
                                     AbstractTensor* out,
                                     bool clamp = true);
bool tensor_scatter_add_2d_kernel_u16(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out,
                                      bool clamp = true);
bool tensor_scatter_add_2d_kernel_u32(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out,
                                      bool clamp = true);
bool tensor_scatter_add_2d_kernel_u64(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out,
                                      bool clamp = true);
bool tensor_scatter_add_2d_kernel_i8(const AbstractTensor& base,
                                     const AbstractTensor& points,
                                     const AbstractTensor& values,
                                     const AbstractTensor& kernel,
                                     AbstractTensor* out,
                                     bool clamp = true);
bool tensor_scatter_add_2d_kernel_i16(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out,
                                      bool clamp = true);
bool tensor_scatter_add_2d_kernel_i32(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out,
                                      bool clamp = true);
bool tensor_scatter_add_2d_kernel_i64(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out,
                                      bool clamp = true);
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
bool tensor_scatter_add_2d_kernel_fn_u8(const AbstractTensor& base,
                                        const AbstractTensor& points,
                                        const AbstractTensor& values,
                                        const TensorKernelFnU8& kernel_fn,
                                        void* user,
                                        AbstractTensor* out,
                                        KernelDispatchPlan* out_plan,
                                        bool clamp = true);
bool tensor_scatter_add_2d_kernel_fn_u16(const AbstractTensor& base,
                                         const AbstractTensor& points,
                                         const AbstractTensor& values,
                                         const TensorKernelFnU16& kernel_fn,
                                         void* user,
                                         AbstractTensor* out,
                                         KernelDispatchPlan* out_plan,
                                         bool clamp = true);
bool tensor_scatter_add_2d_kernel_fn_u32(const AbstractTensor& base,
                                         const AbstractTensor& points,
                                         const AbstractTensor& values,
                                         const TensorKernelFnU32& kernel_fn,
                                         void* user,
                                         AbstractTensor* out,
                                         KernelDispatchPlan* out_plan,
                                         bool clamp = true);
bool tensor_scatter_add_2d_kernel_fn_u64(const AbstractTensor& base,
                                         const AbstractTensor& points,
                                         const AbstractTensor& values,
                                         const TensorKernelFnU64& kernel_fn,
                                         void* user,
                                         AbstractTensor* out,
                                         KernelDispatchPlan* out_plan,
                                         bool clamp = true);
bool tensor_scatter_add_2d_kernel_fn_i8(const AbstractTensor& base,
                                        const AbstractTensor& points,
                                        const AbstractTensor& values,
                                        const TensorKernelFnI8& kernel_fn,
                                        void* user,
                                        AbstractTensor* out,
                                        KernelDispatchPlan* out_plan,
                                        bool clamp = true);
bool tensor_scatter_add_2d_kernel_fn_i16(const AbstractTensor& base,
                                         const AbstractTensor& points,
                                         const AbstractTensor& values,
                                         const TensorKernelFnI16& kernel_fn,
                                         void* user,
                                         AbstractTensor* out,
                                         KernelDispatchPlan* out_plan,
                                         bool clamp = true);
bool tensor_scatter_add_2d_kernel_fn_i32(const AbstractTensor& base,
                                         const AbstractTensor& points,
                                         const AbstractTensor& values,
                                         const TensorKernelFnI32& kernel_fn,
                                         void* user,
                                         AbstractTensor* out,
                                         KernelDispatchPlan* out_plan,
                                         bool clamp = true);
bool tensor_scatter_add_2d_kernel_fn_i64(const AbstractTensor& base,
                                         const AbstractTensor& points,
                                         const AbstractTensor& values,
                                         const TensorKernelFnI64& kernel_fn,
                                         void* user,
                                         AbstractTensor* out,
                                         KernelDispatchPlan* out_plan,
                                         bool clamp = true);
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
bool tensor_scatter_add_2d_kernel_ptr_u8(const AbstractTensor& base,
                                         const AbstractTensor& points,
                                         const AbstractTensor& values,
                                         const AbstractTensor& kernel_ptrs,
                                         AbstractTensor* out,
                                         KernelDispatchPlan* out_plan,
                                         bool clamp = true);
bool tensor_scatter_add_2d_kernel_ptr_u16(const AbstractTensor& base,
                                          const AbstractTensor& points,
                                          const AbstractTensor& values,
                                          const AbstractTensor& kernel_ptrs,
                                          AbstractTensor* out,
                                          KernelDispatchPlan* out_plan,
                                          bool clamp = true);
bool tensor_scatter_add_2d_kernel_ptr_u32(const AbstractTensor& base,
                                          const AbstractTensor& points,
                                          const AbstractTensor& values,
                                          const AbstractTensor& kernel_ptrs,
                                          AbstractTensor* out,
                                          KernelDispatchPlan* out_plan,
                                          bool clamp = true);
bool tensor_scatter_add_2d_kernel_ptr_u64(const AbstractTensor& base,
                                          const AbstractTensor& points,
                                          const AbstractTensor& values,
                                          const AbstractTensor& kernel_ptrs,
                                          AbstractTensor* out,
                                          KernelDispatchPlan* out_plan,
                                          bool clamp = true);
bool tensor_scatter_add_2d_kernel_ptr_i8(const AbstractTensor& base,
                                         const AbstractTensor& points,
                                         const AbstractTensor& values,
                                         const AbstractTensor& kernel_ptrs,
                                         AbstractTensor* out,
                                         KernelDispatchPlan* out_plan,
                                         bool clamp = true);
bool tensor_scatter_add_2d_kernel_ptr_i16(const AbstractTensor& base,
                                          const AbstractTensor& points,
                                          const AbstractTensor& values,
                                          const AbstractTensor& kernel_ptrs,
                                          AbstractTensor* out,
                                          KernelDispatchPlan* out_plan,
                                          bool clamp = true);
bool tensor_scatter_add_2d_kernel_ptr_i32(const AbstractTensor& base,
                                          const AbstractTensor& points,
                                          const AbstractTensor& values,
                                          const AbstractTensor& kernel_ptrs,
                                          AbstractTensor* out,
                                          KernelDispatchPlan* out_plan,
                                          bool clamp = true);
bool tensor_scatter_add_2d_kernel_ptr_i64(const AbstractTensor& base,
                                          const AbstractTensor& points,
                                          const AbstractTensor& values,
                                          const AbstractTensor& kernel_ptrs,
                                          AbstractTensor* out,
                                          KernelDispatchPlan* out_plan,
                                          bool clamp = true);
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
bool tensor_scatter_add_2d_kernel_bank_u8(const AbstractTensor& base,
                                          const AbstractTensor& points,
                                          const AbstractTensor& values,
                                          const AbstractTensor& kernel_bank,
                                          const AbstractTensor& kernel_ids,
                                          AbstractTensor* out,
                                          bool clamp = true);
bool tensor_scatter_add_2d_kernel_bank_u16(const AbstractTensor& base,
                                           const AbstractTensor& points,
                                           const AbstractTensor& values,
                                           const AbstractTensor& kernel_bank,
                                           const AbstractTensor& kernel_ids,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_scatter_add_2d_kernel_bank_u32(const AbstractTensor& base,
                                           const AbstractTensor& points,
                                           const AbstractTensor& values,
                                           const AbstractTensor& kernel_bank,
                                           const AbstractTensor& kernel_ids,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_scatter_add_2d_kernel_bank_u64(const AbstractTensor& base,
                                           const AbstractTensor& points,
                                           const AbstractTensor& values,
                                           const AbstractTensor& kernel_bank,
                                           const AbstractTensor& kernel_ids,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_scatter_add_2d_kernel_bank_i8(const AbstractTensor& base,
                                          const AbstractTensor& points,
                                          const AbstractTensor& values,
                                          const AbstractTensor& kernel_bank,
                                          const AbstractTensor& kernel_ids,
                                          AbstractTensor* out,
                                          bool clamp = true);
bool tensor_scatter_add_2d_kernel_bank_i16(const AbstractTensor& base,
                                           const AbstractTensor& points,
                                           const AbstractTensor& values,
                                           const AbstractTensor& kernel_bank,
                                           const AbstractTensor& kernel_ids,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_scatter_add_2d_kernel_bank_i32(const AbstractTensor& base,
                                           const AbstractTensor& points,
                                           const AbstractTensor& values,
                                           const AbstractTensor& kernel_bank,
                                           const AbstractTensor& kernel_ids,
                                           AbstractTensor* out,
                                           bool clamp = true);
bool tensor_scatter_add_2d_kernel_bank_i64(const AbstractTensor& base,
                                           const AbstractTensor& points,
                                           const AbstractTensor& values,
                                           const AbstractTensor& kernel_bank,
                                           const AbstractTensor& kernel_ids,
                                           AbstractTensor* out,
                                           bool clamp = true);
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
AbstractTensor tensor_apply_stencil_2d_u8(const AbstractTensor& field,
                                          const AbstractTensor& kernel);
AbstractTensor tensor_apply_stencil_2d_u16(const AbstractTensor& field,
                                           const AbstractTensor& kernel);
AbstractTensor tensor_apply_stencil_2d_u32(const AbstractTensor& field,
                                           const AbstractTensor& kernel);
AbstractTensor tensor_apply_stencil_2d_u64(const AbstractTensor& field,
                                           const AbstractTensor& kernel);
AbstractTensor tensor_apply_stencil_2d_i8(const AbstractTensor& field,
                                          const AbstractTensor& kernel);
AbstractTensor tensor_apply_stencil_2d_i16(const AbstractTensor& field,
                                           const AbstractTensor& kernel);
AbstractTensor tensor_apply_stencil_2d_i32(const AbstractTensor& field,
                                           const AbstractTensor& kernel);
AbstractTensor tensor_apply_stencil_2d_i64(const AbstractTensor& field,
                                           const AbstractTensor& kernel);
AbstractTensor tensor_apply_stencil_2d_f32(const AbstractTensor& field,
                                           const AbstractTensor& kernel);
AbstractTensor tensor_apply_stencil_2d_f64(const AbstractTensor& field,
                                           const AbstractTensor& kernel);

// Apply a 2D stencil into an existing output tensor (avoids per-call allocation).
// out must be a valid dense tensor matching field's desc/shape and on the same backend.
bool tensor_apply_stencil_2d_u8_into(const AbstractTensor& field,
                                     const AbstractTensor& kernel,
                                     AbstractTensor* out);
bool tensor_apply_stencil_2d_u16_into(const AbstractTensor& field,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out);
bool tensor_apply_stencil_2d_u32_into(const AbstractTensor& field,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out);
bool tensor_apply_stencil_2d_u64_into(const AbstractTensor& field,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out);
bool tensor_apply_stencil_2d_i8_into(const AbstractTensor& field,
                                     const AbstractTensor& kernel,
                                     AbstractTensor* out);
bool tensor_apply_stencil_2d_i16_into(const AbstractTensor& field,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out);
bool tensor_apply_stencil_2d_i32_into(const AbstractTensor& field,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out);
bool tensor_apply_stencil_2d_i64_into(const AbstractTensor& field,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out);
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
