#include "common/tensors/abstraction/tensor_math.h"

#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/abstract_tensor_pool.h"
#include "common/tensors/abstraction/microkernels.h"
#include "common/tensors/abstraction/tensor_types.h"
#include "common/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <unordered_map>
#include <type_traits>

namespace nodus::tensors {

namespace {
struct TensorOpTensor;
static bool compute_dense_strides_u64(const std::vector<uint32_t>& dims, std::vector<uint64_t>& out) {
    if (dims.empty()) return false;
    out.resize(dims.size());
    uint64_t stride = 1;
    for (size_t i = dims.size(); i-- > 0;) {
        out[i] = stride;
        stride *= static_cast<uint64_t>(dims[i]);
    }
    return true;
}
static bool compute_strides_for_desc_u64(const TensorDesc& desc, std::vector<uint64_t>& out) {
    if (desc.layout == TensorLayout::Strided && desc.strides.elems.size() == desc.shape.dims.size()) {
        out = desc.strides.elems;
        return true;
    }
    return compute_dense_strides_u64(desc.shape.dims, out);
}

static bool build_op_tensor(const TensorDesc& desc, const std::vector<uint64_t>& shape, void* base, TensorOpTensor& out);

static inline void apply_affine_row_major(const float* m,
                                          float x,
                                          float y,
                                          float z,
                                          float& ox,
                                          float& oy,
                                          float& oz) {
    // Row-vector convention: v' = v * M (translation in last row).
    ox = x * m[0] + y * m[4] + z * m[8] + m[12];
    oy = x * m[1] + y * m[5] + z * m[9] + m[13];
    oz = x * m[2] + y * m[6] + z * m[10] + m[14];
}

static inline int64_t quantize_round_fast(float v) {
    return static_cast<int64_t>(std::lround(v));
}

static AbstractTensorPool::Options tile_pool_options() {
    AbstractTensorPool::Options opt;
    opt.clear_on_release = false;
    opt.cache_handles = true;
    opt.enable_shape_bucketing = false;
    opt.max_cached_handles_total = 8;
    opt.max_cached_handles_per_key = 2;
    return opt;
}

struct TensorOpTensor {
    void* base = nullptr;
    std::vector<uint64_t> byte_strides;
    uint32_t elem_bytes = 0;
    bool contiguous = false;
};

struct TensorOpPlan {
    uint32_t rank = 0;
    std::vector<uint64_t> shape;
    std::vector<uint64_t> outer_shape;
    std::vector<uint64_t> outer_strides;
    TensorOpTensor a;
    TensorOpTensor b;
    TensorOpTensor out;
    uint64_t inner_count = 0;
    uint64_t outer_count = 0;
};

static bool build_unary_plan(const AbstractTensor& src,
                             const AbstractTensor& dst,
                             TensorOpPlan& plan) {
    if (!src.valid() || !dst.valid()) return false;
    if (src.backend() != dst.backend()) return false;
    const TensorDesc& sd = src.desc();
    const TensorDesc& dd = dst.desc();
    if (sd.dtype != TensorDType::F32 || dd.dtype != TensorDType::F32) return false;
    if (!((sd.layout == TensorLayout::Dense || sd.layout == TensorLayout::Strided) &&
          (dd.layout == TensorLayout::Dense || dd.layout == TensorLayout::Strided))) {
        return false;
    }
    if (sd.shape.dims != dd.shape.dims) return false;

    plan.shape.assign(sd.shape.dims.begin(), sd.shape.dims.end());
    plan.rank = static_cast<uint32_t>(plan.shape.size());

    auto* mem = dynamic_cast<InMemoryBackend*>(src.backend());
    if (!mem) return false;
    void* sp = nullptr;
    void* dp = nullptr;
    size_t sb = 0;
    size_t db = 0;
    if (!mem->map(src.handle(), &sp, &sb)) return false;
    if (!mem->map(dst.handle(), &dp, &db)) {
        mem->unmap(src.handle());
        return false;
    }

    if (!build_op_tensor(sd, plan.shape, sp, plan.a) ||
        !build_op_tensor(dd, plan.shape, dp, plan.out)) {
        mem->unmap(dst.handle());
        mem->unmap(src.handle());
        return false;
    }

    plan.inner_count = plan.shape.empty() ? 1u : plan.shape.back();
    plan.outer_count = 1u;
    plan.outer_shape.clear();
    plan.outer_strides.clear();
    if (plan.shape.size() > 1) {
        for (size_t i = 0; i + 1 < plan.shape.size(); ++i) {
            plan.outer_count *= plan.shape[i];
            plan.outer_shape.push_back(plan.shape[i]);
        }
        if (!compute_dense_strides_u64(
                std::vector<uint32_t>(plan.outer_shape.begin(), plan.outer_shape.end()), plan.outer_strides)) {
            return false;
        }
    } else {
        plan.outer_count = 1u;
    }

    return true;
}

static void copy_range(const TensorOpPlan& plan, uint64_t outer_begin, uint64_t outer_end) {
    const uint64_t inner = plan.inner_count;
    const size_t rank = plan.shape.size();
    if (rank == 0) {
        auto* s = reinterpret_cast<float*>(plan.a.base);
        auto* d = reinterpret_cast<float*>(plan.out.base);
        *d = *s;
        return;
    }

    auto* s0 = static_cast<uint8_t*>(plan.a.base);
    auto* d0 = static_cast<uint8_t*>(plan.out.base);

    for (uint64_t outer_idx = outer_begin; outer_idx < outer_end; ++outer_idx) {
        uint64_t s_off = 0;
        uint64_t d_off = 0;
        if (rank > 1) {
            uint64_t tmp = outer_idx;
            for (size_t d = 0; d < plan.outer_shape.size(); ++d) {
                const uint64_t stride = plan.outer_strides[d];
                const uint64_t coord = (stride == 0) ? 0 : (tmp / stride);
                tmp = (stride == 0) ? tmp : (tmp % stride);
                s_off += coord * plan.a.byte_strides[d];
                d_off += coord * plan.out.byte_strides[d];
            }
        }

        auto* s_ptr = reinterpret_cast<float*>(s0 + s_off);
        auto* d_ptr = reinterpret_cast<float*>(d0 + d_off);
        const uint64_t s_step = plan.a.byte_strides.back() / plan.a.elem_bytes;
        const uint64_t d_step = plan.out.byte_strides.back() / plan.out.elem_bytes;

        if (plan.a.contiguous && plan.out.contiguous && s_step == 1 && d_step == 1) {
            std::memcpy(d_ptr, s_ptr, static_cast<size_t>(inner * sizeof(float)));
        } else {
            uint64_t si = 0;
            uint64_t di = 0;
            for (uint64_t i = 0; i < inner; ++i) {
                d_ptr[di] = s_ptr[si];
                si += s_step;
                di += d_step;
            }
        }
    }
}

static void execute_copy_plan(const TensorOpPlan& plan) {
    const uint64_t outer = plan.outer_count;
    if (outer == 0 || plan.inner_count == 0) return;

    nodus::ThreadPool* pool = tensor_op_pool();
    const uint64_t kMinOuterPerJob = 256;
    if (!pool || outer < kMinOuterPerJob) {
        copy_range(plan, 0, outer);
        return;
    }

    const uint32_t max_jobs = pool->thread_count();
    const uint32_t job_count = static_cast<uint32_t>(std::min<uint64_t>(outer, max_jobs));
    if (job_count <= 1) {
        copy_range(plan, 0, outer);
        return;
    }

    struct CopyJobParams {
        const TensorOpPlan* plan = nullptr;
        uint64_t begin = 0;
        uint64_t end = 0;
    };

    auto job_fn = [](const nodus::ThreadPool::Job& job, uint32_t /*worker_id*/) {
        const auto* p = static_cast<const CopyJobParams*>(job.params);
        copy_range(*p->plan, p->begin, p->end);
    };

    std::vector<CopyJobParams> params(job_count);
    std::vector<nodus::ThreadPool::Job> jobs(job_count);
    const uint64_t chunk = (outer + job_count - 1) / job_count;
    for (uint32_t i = 0; i < job_count; ++i) {
        const uint64_t begin = static_cast<uint64_t>(i) * chunk;
        const uint64_t end = std::min<uint64_t>(outer, begin + chunk);
        params[i] = CopyJobParams{&plan, begin, end};
        nodus::ThreadPool::Job job{};
        job.fn = job_fn;
        job.params = &params[i];
        job.params_size = static_cast<uint32_t>(sizeof(CopyJobParams));
        jobs[i] = job;
    }

    auto batch = pool->submit_batch(jobs.data(), job_count);
    if (batch) batch->wait();
}

void submit_row_jobs(nodus::ThreadPool* pool,
                     RowRangeFn fn,
                     const void* ctx,
                     uint32_t y0,
                     uint32_t y1) {
    if (!pool || y1 <= y0) {
        if (fn) fn(ctx, y0, y1);
        return;
    }

    const uint32_t span = y1 - y0;
    const uint32_t max_jobs = pool->thread_count();
    if (max_jobs < 2 || span < 64) {
        if (fn) fn(ctx, y0, y1);
        return;
    }

    const uint32_t job_count = std::min<uint32_t>(max_jobs, span);
    const uint32_t chunk = (span + job_count - 1) / job_count;

    struct RowJobParams {
        RowRangeFn fn = nullptr;
        const void* ctx = nullptr;
        uint32_t y0 = 0;
        uint32_t y1 = 0;
    };

    std::vector<RowJobParams> params(job_count);
    std::vector<nodus::ThreadPool::Job> jobs(job_count);

    auto job_fn = [](const nodus::ThreadPool::Job& job, uint32_t /*worker_id*/) {
        const auto* p = static_cast<const RowJobParams*>(job.params);
        p->fn(p->ctx, p->y0, p->y1);
    };

    uint32_t count = 0;
    for (uint32_t i = 0; i < job_count; ++i) {
        const uint32_t begin = y0 + i * chunk;
        const uint32_t end = std::min<uint32_t>(y1, begin + chunk);
        if (begin >= end) break;
        params[count] = RowJobParams{fn, ctx, begin, end};
        nodus::ThreadPool::Job job{};
        job.fn = job_fn;
        job.params = &params[count];
        job.params_size = static_cast<uint32_t>(sizeof(RowJobParams));
        jobs[count] = job;
        ++count;
    }

    auto batch = pool->submit_batch(jobs.data(), count);
    if (batch) batch->wait();
}

static bool compute_broadcast_shape(const TensorDesc& a,
                                    const TensorDesc& b,
                                    std::vector<uint64_t>& out_shape) {
    const size_t ar = a.shape.dims.size();
    const size_t br = b.shape.dims.size();
    const size_t rank = std::max(ar, br);
    out_shape.assign(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const size_t ai = (i < rank - ar) ? static_cast<size_t>(-1) : (i - (rank - ar));
        const size_t bi = (i < rank - br) ? static_cast<size_t>(-1) : (i - (rank - br));
        const uint64_t ad = (ai == static_cast<size_t>(-1)) ? 1u : a.shape.dims[ai];
        const uint64_t bd = (bi == static_cast<size_t>(-1)) ? 1u : b.shape.dims[bi];
        if (ad != bd && ad != 1 && bd != 1) return false;
        out_shape[i] = std::max(ad, bd);
    }
    return true;
}

static bool build_op_tensor(const TensorDesc& desc,
                            const std::vector<uint64_t>& out_shape,
                            void* base,
                            TensorOpTensor& out) {
    const size_t rank = out_shape.size();
    std::vector<uint64_t> elem_strides;
    if (!compute_strides_for_desc_u64(desc, elem_strides)) return false;
    out.elem_bytes = tensor_dtype_size_bytes(desc.dtype);
    if (out.elem_bytes == 0) return false;
    out.byte_strides.assign(rank, 0);
    out.base = base;

    const size_t dr = desc.shape.dims.size();
    const size_t pad = (rank >= dr) ? (rank - dr) : 0;
    for (size_t i = 0; i < rank; ++i) {
        if (i < pad) {
            out.byte_strides[i] = 0;
            continue;
        }
        const size_t di = i - pad;
        const uint64_t dim = desc.shape.dims[di];
        const uint64_t stride = elem_strides[di];
        out.byte_strides[i] = (dim == 1 && out_shape[i] > 1) ? 0 : stride * out.elem_bytes;
    }

    // Contiguous if dense and no broadcast in trailing dimension.
    out.contiguous = (desc.layout == TensorLayout::Dense);
    return true;
}

static bool build_axpby_plan(const AbstractTensor& a,
                             const AbstractTensor& b,
                             const AbstractTensor& out,
                             TensorOpPlan& plan) {
    if (!a.valid() || !b.valid() || !out.valid()) return false;
    if (a.backend() != b.backend() || a.backend() != out.backend()) return false;

    const TensorDesc& ad = a.desc();
    const TensorDesc& bd = b.desc();
    const TensorDesc& od = out.desc();
    if (ad.dtype != TensorDType::F32 || bd.dtype != TensorDType::F32 || od.dtype != TensorDType::F32) return false;
    if (!((ad.layout == TensorLayout::Dense || ad.layout == TensorLayout::Strided) &&
          (bd.layout == TensorLayout::Dense || bd.layout == TensorLayout::Strided) &&
          (od.layout == TensorLayout::Dense || od.layout == TensorLayout::Strided))) {
        return false;
    }

    if (!compute_broadcast_shape(ad, bd, plan.shape)) return false;
    plan.rank = static_cast<uint32_t>(plan.shape.size());

    // Out must match broadcast shape (no implicit output broadcast).
    if (od.shape.dims.size() != plan.shape.size()) return false;
    for (size_t i = 0; i < plan.shape.size(); ++i) {
        if (od.shape.dims[i] != plan.shape[i]) return false;
    }

    auto* mem = dynamic_cast<InMemoryBackend*>(a.backend());
    if (!mem) return false;

    void* ap = nullptr;
    void* bp = nullptr;
    void* op = nullptr;
    size_t ab = 0;
    size_t bb = 0;
    size_t ob = 0;
    if (!mem->map(a.handle(), &ap, &ab)) return false;
    if (!mem->map(b.handle(), &bp, &bb)) {
        mem->unmap(a.handle());
        return false;
    }
    if (!mem->map(out.handle(), &op, &ob)) {
        mem->unmap(a.handle());
        mem->unmap(b.handle());
        return false;
    }

    if (!build_op_tensor(ad, plan.shape, ap, plan.a) ||
        !build_op_tensor(bd, plan.shape, bp, plan.b) ||
        !build_op_tensor(od, plan.shape, op, plan.out)) {
        mem->unmap(out.handle());
        mem->unmap(b.handle());
        mem->unmap(a.handle());
        return false;
    }

    plan.inner_count = plan.shape.empty() ? 1u : plan.shape.back();
    plan.outer_count = 1u;
    plan.outer_shape.clear();
    plan.outer_strides.clear();
    if (plan.shape.size() > 1) {
        for (size_t i = 0; i + 1 < plan.shape.size(); ++i) {
            plan.outer_count *= plan.shape[i];
            plan.outer_shape.push_back(plan.shape[i]);
        }
        if (!compute_dense_strides_u64(
                std::vector<uint32_t>(plan.outer_shape.begin(), plan.outer_shape.end()), plan.outer_strides)) {
            return false;
        }
    } else {
        plan.outer_count = 1u;
    }

    return true;
}

nodus::ThreadPool* tensor_op_pool() {
    static nodus::ThreadPool* pool = []() -> nodus::ThreadPool* {
        const char* v = std::getenv("NODUS_TENSOR_OP_THREADS");
        if (!v || !*v) return nullptr;
        char* end = nullptr;
        unsigned long long parsed = std::strtoull(v, &end, 10);
        if (end == v || parsed < 2) return nullptr;
        nodus::ThreadPool::Options opt{};
        opt.thread_count = static_cast<uint32_t>(parsed);
        opt.start_immediately = true;
        return new nodus::ThreadPool(opt);
    }();
    return pool;
}

static void axpby_range(const TensorOpPlan& plan,
                        float alpha,
                        float beta,
                        uint64_t outer_begin,
                        uint64_t outer_end) {
    const uint64_t inner = plan.inner_count;
    const size_t rank = plan.shape.size();

    if (rank == 0) {
        auto* a_ptr = reinterpret_cast<float*>(plan.a.base);
        auto* b_ptr = reinterpret_cast<float*>(plan.b.base);
        auto* o_ptr = reinterpret_cast<float*>(plan.out.base);
        *o_ptr = alpha * (*a_ptr) + beta * (*b_ptr);
        return;
    }

    const bool all_contig = plan.a.contiguous && plan.b.contiguous && plan.out.contiguous &&
                            plan.a.byte_strides.back() == plan.a.elem_bytes &&
                            plan.b.byte_strides.back() == plan.b.elem_bytes &&
                            plan.out.byte_strides.back() == plan.out.elem_bytes;

    auto* a0 = static_cast<uint8_t*>(plan.a.base);
    auto* b0 = static_cast<uint8_t*>(plan.b.base);
    auto* o0 = static_cast<uint8_t*>(plan.out.base);

    for (uint64_t outer_idx = outer_begin; outer_idx < outer_end; ++outer_idx) {
        uint64_t a_off = 0;
        uint64_t b_off = 0;
        uint64_t o_off = 0;
        if (rank > 1) {
            uint64_t tmp = outer_idx;
            for (size_t d = 0; d < plan.outer_shape.size(); ++d) {
                const uint64_t stride = plan.outer_strides[d];
                const uint64_t coord = (stride == 0) ? 0 : (tmp / stride);
                tmp = (stride == 0) ? tmp : (tmp % stride);
                a_off += coord * plan.a.byte_strides[d];
                b_off += coord * plan.b.byte_strides[d];
                o_off += coord * plan.out.byte_strides[d];
            }
        }

        auto* a_ptr = reinterpret_cast<float*>(a0 + a_off);
        auto* b_ptr = reinterpret_cast<float*>(b0 + b_off);
        auto* o_ptr = reinterpret_cast<float*>(o0 + o_off);

        const uint64_t a_step = plan.a.byte_strides.back() / plan.a.elem_bytes;
        const uint64_t b_step = plan.b.byte_strides.back() / plan.b.elem_bytes;
        const uint64_t o_step = plan.out.byte_strides.back() / plan.out.elem_bytes;

        if (all_contig && a_step == 1 && b_step == 1 && o_step == 1) {
            for (uint64_t i = 0; i < inner; ++i) {
                o_ptr[i] = alpha * a_ptr[i] + beta * b_ptr[i];
            }
        } else {
            uint64_t ai = 0;
            uint64_t bi = 0;
            uint64_t oi = 0;
            for (uint64_t i = 0; i < inner; ++i) {
                o_ptr[oi] = alpha * a_ptr[ai] + beta * b_ptr[bi];
                ai += a_step;
                bi += b_step;
                oi += o_step;
            }
        }

    }
}

static void execute_axpby_plan(const TensorOpPlan& plan, float alpha, float beta) {
    const uint64_t outer = plan.outer_count;
    const uint64_t inner = plan.inner_count;
    if (outer == 0 || inner == 0) return;

    nodus::ThreadPool* pool = tensor_op_pool();
    const uint64_t kMinOuterPerJob = 256;
    if (!pool || outer < kMinOuterPerJob) {
        axpby_range(plan, alpha, beta, 0, outer);
        return;
    }

    const uint32_t max_jobs = pool->thread_count();
    const uint32_t job_count = static_cast<uint32_t>(std::min<uint64_t>(outer, max_jobs));
    if (job_count <= 1) {
        axpby_range(plan, alpha, beta, 0, outer);
        return;
    }

    struct AxpbyJobParams {
        const TensorOpPlan* plan = nullptr;
        float alpha = 0.0f;
        float beta = 0.0f;
        uint64_t begin = 0;
        uint64_t end = 0;
    };

    auto job_fn = [](const nodus::ThreadPool::Job& job, uint32_t /*worker_id*/) {
        const auto* p = static_cast<const AxpbyJobParams*>(job.params);
        axpby_range(*p->plan, p->alpha, p->beta, p->begin, p->end);
    };

    std::vector<AxpbyJobParams> params(job_count);
    std::vector<nodus::ThreadPool::Job> jobs(job_count);
    const uint64_t chunk = (outer + job_count - 1) / job_count;
    for (uint32_t i = 0; i < job_count; ++i) {
        const uint64_t begin = static_cast<uint64_t>(i) * chunk;
        const uint64_t end = std::min<uint64_t>(outer, begin + chunk);
        params[i] = AxpbyJobParams{&plan, alpha, beta, begin, end};
        nodus::ThreadPool::Job job{};
        job.fn = job_fn;
        job.params = &params[i];
        job.params_size = static_cast<uint32_t>(sizeof(AxpbyJobParams));
        jobs[i] = job;
    }

    auto batch = pool->submit_batch(jobs.data(), job_count);
    if (batch) batch->wait();
}

template <typename Scalar, TensorDType DType>
struct TensorMathImpl {
    static constexpr Scalar kQuatEps = static_cast<Scalar>(1e-8);
    static constexpr Scalar kPlaneEps = static_cast<Scalar>(1e-8);

    struct MappedDense {
        InMemoryBackend* backend = nullptr;
        AbstractTensorHandle handle{};
        Scalar* data = nullptr;
        uint64_t elems = 0;
        bool ok = false;

        void unmap() {
            if (backend && data) backend->unmap(handle);
            backend = nullptr;
            data = nullptr;
            elems = 0;
            ok = false;
        }
    };

    static MappedDense map_dense(const AbstractTensor& t) {
        MappedDense out{};
        if (!t.valid()) return out;
        const TensorDesc& d = t.desc();
        if (d.dtype != DType || d.layout != TensorLayout::Dense) return out;
        auto* mem = dynamic_cast<InMemoryBackend*>(t.backend());
        if (!mem) return out;
        void* ptr = nullptr;
        size_t bytes = 0;
        if (!mem->map(t.handle(), &ptr, &bytes)) return out;
        out.backend = mem;
        out.handle = t.handle();
        out.data = static_cast<Scalar*>(ptr);
        out.elems = d.shape.element_count();
        out.ok = true;
        return out;
    }

    static MappedDense map_dense_mut(const AbstractTensor& t) { return map_dense(t); }

    static bool shape_is(const TensorDesc& desc, std::initializer_list<uint32_t> dims) {
        if (desc.shape.dims.size() != dims.size()) return false;
        size_t i = 0;
        for (uint32_t d : dims) {
            if (desc.shape.dims[i++] != d) return false;
        }
        return true;
    }

    static bool read_vec3(const AbstractTensor& t, Scalar* out3) {
        if (!out3 || !t.valid()) return false;
        const TensorDesc& d = t.desc();
        if (d.dtype != DType || d.layout != TensorLayout::Dense) return false;
        if (!(shape_is(d, {3}) || shape_is(d, {1, 3}) ||
              (d.shape.dims.size() == 2 && d.shape.dims[1] == 3))) {
            return false;
        }
        MappedDense map = map_dense(t);
        if (!map.ok || map.elems < 3) {
            map.unmap();
            return false;
        }
        out3[0] = map.data[0];
        out3[1] = map.data[1];
        out3[2] = map.data[2];
        map.unmap();
        return true;
    }

    static void quat_to_mat4_row(Scalar w, Scalar x, Scalar y, Scalar z, const Scalar* t3, Scalar* out16) {
        const Scalar xx = x * x;
        const Scalar yy = y * y;
        const Scalar zz = z * z;
        const Scalar xy = x * y;
        const Scalar xz = x * z;
        const Scalar yz = y * z;
        const Scalar wx = w * x;
        const Scalar wy = w * y;
        const Scalar wz = w * z;

        out16[0] = static_cast<Scalar>(1) - static_cast<Scalar>(2) * (yy + zz);
        out16[1] = static_cast<Scalar>(2) * (xy - wz);
        out16[2] = static_cast<Scalar>(2) * (xz + wy);
        out16[3] = static_cast<Scalar>(0);

        out16[4] = static_cast<Scalar>(2) * (xy + wz);
        out16[5] = static_cast<Scalar>(1) - static_cast<Scalar>(2) * (xx + zz);
        out16[6] = static_cast<Scalar>(2) * (yz - wx);
        out16[7] = static_cast<Scalar>(0);

        out16[8] = static_cast<Scalar>(2) * (xz - wy);
        out16[9] = static_cast<Scalar>(2) * (yz + wx);
        out16[10] = static_cast<Scalar>(1) - static_cast<Scalar>(2) * (xx + yy);
        out16[11] = static_cast<Scalar>(0);

        out16[12] = t3 ? t3[0] : static_cast<Scalar>(0);
        out16[13] = t3 ? t3[1] : static_cast<Scalar>(0);
        out16[14] = t3 ? t3[2] : static_cast<Scalar>(0);
        out16[15] = static_cast<Scalar>(1);
    }

    static AbstractTensor matmul(const AbstractTensor& a, const AbstractTensor& b) {
        if (!a.valid() || !b.valid()) return {};
        if (a.backend() != b.backend()) return {};
        const TensorDesc& ad = a.desc();
        const TensorDesc& bd = b.desc();
        if (ad.dtype != DType || bd.dtype != DType) return {};
        if (ad.layout != TensorLayout::Dense || bd.layout != TensorLayout::Dense) return {};
        if (ad.shape.dims.size() != 2 || bd.shape.dims.size() != 2) return {};
        const uint32_t m = ad.shape.dims[0];
        const uint32_t n = ad.shape.dims[1];
        const uint32_t n2 = bd.shape.dims[0];
        const uint32_t k = bd.shape.dims[1];
        if (n == 0 || n2 == 0 || m == 0 || k == 0) return {};
        if (n != n2) return {};

        TensorDesc out_desc{};
        out_desc.dtype = DType;
        out_desc.layout = TensorLayout::Dense;
        out_desc.shape.dims = {m, k};
        AbstractTensor out = AbstractTensor::create(out_desc, a.backend());
        if (!out.valid()) return {};

        MappedDense amap = map_dense(a);
        MappedDense bmap = map_dense(b);
        MappedDense omap = map_dense_mut(out);
        if (!amap.ok || !bmap.ok || !omap.ok) {
            amap.unmap();
            bmap.unmap();
            omap.unmap();
            return {};
        }

        const Scalar* A = amap.data;
        const Scalar* B = bmap.data;
        Scalar* C = omap.data;

        struct MatmulCtx {
            const Scalar* A = nullptr;
            const Scalar* B = nullptr;
            Scalar* C = nullptr;
            uint32_t m = 0;
            uint32_t n = 0;
            uint32_t k = 0;
        };

        auto matmul_rows = [](const void* vctx, uint32_t r0, uint32_t r1) {
            const auto* ctx = static_cast<const MatmulCtx*>(vctx);
            const Scalar* A = ctx->A;
            const Scalar* B = ctx->B;
            Scalar* C = ctx->C;
            const uint32_t n = ctx->n;
            const uint32_t k = ctx->k;
            for (uint32_t r = r0; r < r1; ++r) {
                for (uint32_t c = 0; c < k; ++c) {
                    Scalar acc = static_cast<Scalar>(0);
                    for (uint32_t i = 0; i < n; ++i) {
                        acc += A[r * n + i] * B[i * k + c];
                    }
                    C[r * k + c] = acc;
                }
            }
        };

        MatmulCtx ctx{};
        ctx.A = A;
        ctx.B = B;
        ctx.C = C;
        ctx.m = m;
        ctx.n = n;
        ctx.k = k;

        submit_row_jobs(tensor_op_pool(), matmul_rows, &ctx, 0, m);

        amap.unmap();
        bmap.unmap();
        omap.unmap();
        return out;
    }

    static AbstractTensor affine_identity(TensorBackend* backend) {
        TensorDesc desc{};
        desc.dtype = DType;
        desc.layout = TensorLayout::Dense;
        desc.shape.dims = {4, 4};
        AbstractTensor out = AbstractTensor::create(desc, backend);
        if (!out.valid()) return {};
        MappedDense map = map_dense_mut(out);
        if (!map.ok || map.elems < 16) {
            map.unmap();
            return {};
        }
        std::fill(map.data, map.data + 16, static_cast<Scalar>(0));
        map.data[0] = map.data[5] = map.data[10] = map.data[15] = static_cast<Scalar>(1);
        map.unmap();
        return out;
    }

    static AbstractTensor affine_translation(const AbstractTensor& t) {
        if (!t.valid()) return {};
        const TensorDesc& td = t.desc();
        if (td.dtype != DType || td.layout != TensorLayout::Dense) return {};
        const bool t_vec = shape_is(td, {3}) || shape_is(td, {1, 3});
        const bool t_batch = (td.shape.dims.size() == 2 && td.shape.dims[1] == 3);
        if (!t_vec && !t_batch) return {};
        const uint32_t batch = t_batch ? td.shape.dims[0] : 1;

        TensorDesc desc{};
        desc.dtype = DType;
        desc.layout = TensorLayout::Dense;
        desc.shape.dims = (batch > 1) ? std::vector<uint32_t>{batch, 4, 4}
                                      : std::vector<uint32_t>{4, 4};
        AbstractTensor out = AbstractTensor::create(desc, t.backend());
        if (!out.valid()) return {};
        MappedDense tmap = map_dense(t);
        MappedDense map = map_dense_mut(out);
        if (!tmap.ok || !map.ok) {
            tmap.unmap();
            map.unmap();
            return {};
        }

        for (uint32_t i = 0; i < batch; ++i) {
            const Scalar* v = tmap.data + (t_vec ? 0u : static_cast<uint64_t>(i) * 3u);
            Scalar* dst = map.data + (static_cast<uint64_t>(i) * 16u);
            std::fill(dst, dst + 16, static_cast<Scalar>(0));
            dst[0] = dst[5] = dst[10] = dst[15] = static_cast<Scalar>(1);
            dst[12] = v[0];
            dst[13] = v[1];
            dst[14] = v[2];
        }

        tmap.unmap();
        map.unmap();
        return out;
    }

    static AbstractTensor affine_scale(const AbstractTensor& s) {
        if (!s.valid()) return {};
        const TensorDesc& sd = s.desc();
        if (sd.dtype != DType || sd.layout != TensorLayout::Dense) return {};
        const bool s_vec = shape_is(sd, {3}) || shape_is(sd, {1, 3});
        const bool s_batch = (sd.shape.dims.size() == 2 && sd.shape.dims[1] == 3);
        if (!s_vec && !s_batch) return {};
        const uint32_t batch = s_batch ? sd.shape.dims[0] : 1;

        TensorDesc desc{};
        desc.dtype = DType;
        desc.layout = TensorLayout::Dense;
        desc.shape.dims = (batch > 1) ? std::vector<uint32_t>{batch, 4, 4}
                                      : std::vector<uint32_t>{4, 4};
        AbstractTensor out = AbstractTensor::create(desc, s.backend());
        if (!out.valid()) return {};
        MappedDense smap = map_dense(s);
        MappedDense map = map_dense_mut(out);
        if (!smap.ok || !map.ok) {
            smap.unmap();
            map.unmap();
            return {};
        }

        for (uint32_t i = 0; i < batch; ++i) {
            const Scalar* v = smap.data + (s_vec ? 0u : static_cast<uint64_t>(i) * 3u);
            Scalar* dst = map.data + (static_cast<uint64_t>(i) * 16u);
            std::fill(dst, dst + 16, static_cast<Scalar>(0));
            dst[0] = v[0];
            dst[5] = v[1];
            dst[10] = v[2];
            dst[15] = static_cast<Scalar>(1);
        }

        smap.unmap();
        map.unmap();
        return out;
    }

    static AbstractTensor quat_identity(TensorBackend* backend) {
        TensorDesc desc{};
        desc.dtype = DType;
        desc.layout = TensorLayout::Dense;
        desc.shape.dims = {4};
        AbstractTensor out = AbstractTensor::create(desc, backend);
        if (!out.valid()) return {};
        MappedDense map = map_dense_mut(out);
        if (!map.ok || map.elems < 4) {
            map.unmap();
            return {};
        }
        map.data[0] = static_cast<Scalar>(1);
        map.data[1] = static_cast<Scalar>(0);
        map.data[2] = static_cast<Scalar>(0);
        map.data[3] = static_cast<Scalar>(0);
        map.unmap();
        return out;
    }

    static AbstractTensor quat_from_axis_angle(const AbstractTensor& axis, Scalar angle) {
        Scalar v[3]{};
        if (!read_vec3(axis, v)) return {};
        Scalar len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (len < kQuatEps) return quat_identity(axis.backend());
        Scalar inv = static_cast<Scalar>(1) / len;
        v[0] *= inv;
        v[1] *= inv;
        v[2] *= inv;
        const Scalar half = static_cast<Scalar>(0.5) * angle;
        const Scalar s = std::sin(half);
        const Scalar c = std::cos(half);

        TensorDesc desc{};
        desc.dtype = DType;
        desc.layout = TensorLayout::Dense;
        desc.shape.dims = {4};
        AbstractTensor out = AbstractTensor::create(desc, axis.backend());
        if (!out.valid()) return {};
        MappedDense map = map_dense_mut(out);
        if (!map.ok || map.elems < 4) {
            map.unmap();
            return {};
        }
        map.data[0] = c;
        map.data[1] = v[0] * s;
        map.data[2] = v[1] * s;
        map.data[3] = v[2] * s;
        map.unmap();
        return out;
    }

    static AbstractTensor quat_normalize(const AbstractTensor& q) {
        if (!q.valid()) return {};
        const TensorDesc& d = q.desc();
        if (d.dtype != DType || d.layout != TensorLayout::Dense) return {};
        if (!(shape_is(d, {4}) || (d.shape.dims.size() == 2 && d.shape.dims[1] == 4))) return {};

        TensorDesc out_desc = d;
        AbstractTensor out = AbstractTensor::create(out_desc, q.backend());
        if (!out.valid()) return {};
        MappedDense in = map_dense(q);
        MappedDense outm = map_dense_mut(out);
        if (!in.ok || !outm.ok) {
            in.unmap();
            outm.unmap();
            return {};
        }

        const uint64_t count = (d.shape.dims.size() == 2) ? d.shape.dims[0] : 1;
        for (uint64_t i = 0; i < count; ++i) {
            const Scalar* src = in.data + i * 4;
            Scalar* dst = outm.data + i * 4;
            const Scalar w = src[0], x = src[1], y = src[2], z = src[3];
            const Scalar len = std::sqrt(w * w + x * x + y * y + z * z);
            if (len < kQuatEps) {
                dst[0] = static_cast<Scalar>(1);
                dst[1] = dst[2] = dst[3] = static_cast<Scalar>(0);
            } else {
                const Scalar inv = static_cast<Scalar>(1) / len;
                dst[0] = w * inv;
                dst[1] = x * inv;
                dst[2] = y * inv;
                dst[3] = z * inv;
            }
        }

        in.unmap();
        outm.unmap();
        return out;
    }

    static AbstractTensor quat_mul(const AbstractTensor& a, const AbstractTensor& b) {
        if (!a.valid() || !b.valid()) return {};
        if (a.backend() != b.backend()) return {};
        const TensorDesc& ad = a.desc();
        const TensorDesc& bd = b.desc();
        if (ad.dtype != DType || bd.dtype != DType) return {};
        if (ad.layout != TensorLayout::Dense || bd.layout != TensorLayout::Dense) return {};

        const bool a_vec = shape_is(ad, {4});
        const bool b_vec = shape_is(bd, {4});
        const bool a_batch = (ad.shape.dims.size() == 2 && ad.shape.dims[1] == 4);
        const bool b_batch = (bd.shape.dims.size() == 2 && bd.shape.dims[1] == 4);
        if (!(a_vec || a_batch) || !(b_vec || b_batch)) return {};

        uint32_t batch = 1;
        if (a_batch && b_batch) {
            if (ad.shape.dims[0] != bd.shape.dims[0]) return {};
            batch = ad.shape.dims[0];
        } else if (a_batch) {
            batch = ad.shape.dims[0];
        } else if (b_batch) {
            batch = bd.shape.dims[0];
        }

        TensorDesc out_desc{};
        out_desc.dtype = DType;
        out_desc.layout = TensorLayout::Dense;
        out_desc.shape.dims = (batch == 1 && a_vec && b_vec) ? std::vector<uint32_t>{4}
                                                             : std::vector<uint32_t>{batch, 4};
        AbstractTensor out = AbstractTensor::create(out_desc, a.backend());
        if (!out.valid()) return {};

        MappedDense amap = map_dense(a);
        MappedDense bmap = map_dense(b);
        MappedDense omap = map_dense_mut(out);
        if (!amap.ok || !bmap.ok || !omap.ok) {
            amap.unmap();
            bmap.unmap();
            omap.unmap();
            return {};
        }

        for (uint32_t i = 0; i < batch; ++i) {
            const Scalar* qa = amap.data + (a_vec ? 0u : static_cast<uint64_t>(i) * 4u);
            const Scalar* qb = bmap.data + (b_vec ? 0u : static_cast<uint64_t>(i) * 4u);
            Scalar* qc = omap.data + static_cast<uint64_t>(i) * 4u;
            const Scalar w1 = qa[0], x1 = qa[1], y1 = qa[2], z1 = qa[3];
            const Scalar w2 = qb[0], x2 = qb[1], y2 = qb[2], z2 = qb[3];
            qc[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
            qc[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
            qc[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
            qc[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
        }

        amap.unmap();
        bmap.unmap();
        omap.unmap();
        return out;
    }

    static AbstractTensor quat_to_mat4(const AbstractTensor& q, const AbstractTensor& t) {
        if (!q.valid()) return {};
        const TensorDesc& qd = q.desc();
        if (qd.dtype != DType || qd.layout != TensorLayout::Dense) return {};
        const bool q_vec = shape_is(qd, {4});
        const bool q_batch = (qd.shape.dims.size() == 2 && qd.shape.dims[1] == 4);
        if (!q_vec && !q_batch) return {};

        const bool have_t = t.valid();
        if (have_t && t.backend() != q.backend()) return {};
        const TensorDesc& td = t.desc();
        const bool t_vec = have_t && shape_is(td, {3});
        const bool t_batch = have_t && (td.shape.dims.size() == 2 && td.shape.dims[1] == 3);
        if (have_t && !(t_vec || t_batch)) return {};

        uint32_t batch = q_batch ? qd.shape.dims[0] : 1;
        if (t_batch) {
            if (batch != 1 && td.shape.dims[0] != batch) return {};
            batch = td.shape.dims[0];
        }

        TensorDesc out_desc{};
        out_desc.dtype = DType;
        out_desc.layout = TensorLayout::Dense;
        if (batch == 1 && q_vec && !t_batch) {
            out_desc.shape.dims = {4, 4};
        } else {
            out_desc.shape.dims = {batch, 4, 4};
        }

        AbstractTensor out = AbstractTensor::create(out_desc, q.backend());
        if (!out.valid()) return {};

        MappedDense qmap = map_dense(q);
        MappedDense tmap = have_t ? map_dense(t) : MappedDense{};
        MappedDense omap = map_dense_mut(out);
        if (!qmap.ok || !omap.ok || (have_t && !tmap.ok)) {
            qmap.unmap();
            tmap.unmap();
            omap.unmap();
            return {};
        }

        for (uint32_t i = 0; i < batch; ++i) {
            const Scalar* qq = qmap.data + (q_vec ? 0u : static_cast<uint64_t>(i) * 4u);
            Scalar qw = qq[0], qx = qq[1], qy = qq[2], qz = qq[3];
            const Scalar len = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            if (len >= kQuatEps) {
                const Scalar inv = static_cast<Scalar>(1) / len;
                qw *= inv;
                qx *= inv;
                qy *= inv;
                qz *= inv;
            } else {
                qw = static_cast<Scalar>(1);
                qx = qy = qz = static_cast<Scalar>(0);
            }

            const Scalar* tt = nullptr;
            if (have_t) {
                tt = tmap.data + (t_vec ? 0u : static_cast<uint64_t>(i) * 3u);
            }
            Scalar* dst = omap.data + static_cast<uint64_t>(i) * 16u;
            quat_to_mat4_row(qw, qx, qy, qz, tt, dst);
        }

        qmap.unmap();
        tmap.unmap();
        omap.unmap();
        return out;
    }

    static AbstractTensor affine_from_quat_translation(const AbstractTensor& q, const AbstractTensor& t) {
        return quat_to_mat4(q, t);
    }

    static AbstractTensor transform_points(const AbstractTensor& points, const AbstractTensor& mat4) {
        if (!points.valid() || !mat4.valid()) return {};
        if (points.backend() != mat4.backend()) return {};

        const TensorDesc& pd = points.desc();
        const TensorDesc& md = mat4.desc();
        if (pd.dtype != DType || md.dtype != DType) return {};
        if (pd.layout != TensorLayout::Dense || md.layout != TensorLayout::Dense) return {};

        const bool p_vec = shape_is(pd, {3});
        const bool p_batch = (pd.shape.dims.size() == 2 && pd.shape.dims[1] == 3);
        if (!p_vec && !p_batch) return {};

        const bool m_single = shape_is(md, {4, 4});
        const bool m_batch = (md.shape.dims.size() == 3 && md.shape.dims[1] == 4 && md.shape.dims[2] == 4);
        if (!m_single && !m_batch) return {};

        uint32_t batch = p_vec ? 1 : pd.shape.dims[0];
        if (m_batch && md.shape.dims[0] != batch) return {};

        TensorDesc out_desc = pd;
        AbstractTensor out = AbstractTensor::create(out_desc, points.backend());
        if (!out.valid()) return {};

        MappedDense pmap = map_dense(points);
        MappedDense mmap = map_dense(mat4);
        MappedDense omap = map_dense_mut(out);
        if (!pmap.ok || !mmap.ok || !omap.ok) {
            pmap.unmap();
            mmap.unmap();
            omap.unmap();
            return {};
        }

        struct TransformCtx {
            const Scalar* p = nullptr;
            const Scalar* m = nullptr;
            Scalar* out = nullptr;
            bool p_vec = false;
            bool m_single = false;
        };

        auto transform_rows = [](const void* vctx, uint32_t i0, uint32_t i1) {
            const auto* ctx = static_cast<const TransformCtx*>(vctx);
            for (uint32_t i = i0; i < i1; ++i) {
                const Scalar* p = ctx->p + (ctx->p_vec ? 0u : static_cast<uint64_t>(i) * 3u);
                const Scalar* m = ctx->m + (ctx->m_single ? 0u : static_cast<uint64_t>(i) * 16u);
                Scalar* dst = ctx->out + (ctx->p_vec ? 0u : static_cast<uint64_t>(i) * 3u);
                const Scalar x = p[0];
                const Scalar y = p[1];
                const Scalar z = p[2];
                // Row-vector convention: v' = v * M (translation in last row).
                dst[0] = x * m[0] + y * m[4] + z * m[8] + m[12];
                dst[1] = x * m[1] + y * m[5] + z * m[9] + m[13];
                dst[2] = x * m[2] + y * m[6] + z * m[10] + m[14];
            }
        };

        TransformCtx ctx{};
        ctx.p = pmap.data;
        ctx.m = mmap.data;
        ctx.out = omap.data;
        ctx.p_vec = p_vec;
        ctx.m_single = m_single;

        submit_row_jobs(tensor_op_pool(), transform_rows, &ctx, 0, batch);

        pmap.unmap();
        mmap.unmap();
        omap.unmap();
        return out;
    }

    static bool intersect_plane_z(const AbstractTensor& origins,
                                  const AbstractTensor& dirs,
                                  Scalar plane_z,
                                  AbstractTensor* out_hits,
                                  AbstractTensor* out_mask) {
        if (!out_hits || !out_mask) return false;
        out_hits->reset();
        out_mask->reset();
        if (!origins.valid() || !dirs.valid()) return false;
        if (origins.backend() != dirs.backend()) return false;

        const TensorDesc& od = origins.desc();
        const TensorDesc& dd = dirs.desc();
        if (od.dtype != DType || dd.dtype != DType) return false;
        if (od.layout != TensorLayout::Dense || dd.layout != TensorLayout::Dense) return false;
        if (od.shape.dims.size() != 2 || dd.shape.dims.size() != 2) return false;
        const uint32_t batch = od.shape.dims[0];
        if (dd.shape.dims[0] != batch) return false;
        if (od.shape.dims[1] != 3 || dd.shape.dims[1] != 3) return false;

        auto* mem = dynamic_cast<InMemoryBackend*>(origins.backend());
        if (!mem) return false;

        TensorDesc hit_desc{};
        hit_desc.dtype = DType;
        hit_desc.layout = TensorLayout::Dense;
        hit_desc.shape.dims = {batch, 3u};
        *out_hits = AbstractTensor::create(hit_desc, origins.backend());
        if (!out_hits->valid()) return false;

        TensorDesc mask_desc{};
        mask_desc.dtype = TensorDType::Bool;
        mask_desc.layout = TensorLayout::Dense;
        mask_desc.shape.dims = {batch};
        *out_mask = AbstractTensor::create(mask_desc, origins.backend());
        if (!out_mask->valid()) return false;

        MappedDense omap = map_dense(origins);
        MappedDense dmap = map_dense(dirs);
        MappedDense hmap = map_dense_mut(*out_hits);
        if (!omap.ok || !dmap.ok || !hmap.ok) {
            omap.unmap();
            dmap.unmap();
            hmap.unmap();
            return false;
        }

        void* m_ptr_v = nullptr;
        size_t m_bytes = 0;
        if (!mem->map(out_mask->handle(), &m_ptr_v, &m_bytes)) {
            omap.unmap();
            dmap.unmap();
            hmap.unmap();
            return false;
        }
        auto* m_ptr = static_cast<uint8_t*>(m_ptr_v);

        struct IntersectCtx {
            const Scalar* o = nullptr;
            const Scalar* d = nullptr;
            Scalar* h = nullptr;
            uint8_t* m = nullptr;
            Scalar plane_z = static_cast<Scalar>(0);
        };

        auto intersect_rows = [](const void* vctx, uint32_t b0, uint32_t b1) {
            const auto* ctx = static_cast<const IntersectCtx*>(vctx);
            for (uint32_t b = b0; b < b1; ++b) {
                const Scalar ox = ctx->o[3 * b + 0];
                const Scalar oy = ctx->o[3 * b + 1];
                const Scalar oz = ctx->o[3 * b + 2];

                const Scalar dx = ctx->d[3 * b + 0];
                const Scalar dy = ctx->d[3 * b + 1];
                const Scalar dz = ctx->d[3 * b + 2];

                bool ok = std::fabs(static_cast<double>(dz)) > static_cast<double>(kPlaneEps);
                Scalar t = ok ? (ctx->plane_z - oz) / dz : static_cast<Scalar>(0);
                ok = ok && (t >= static_cast<Scalar>(0));
                ctx->m[b] = ok ? 1u : 0u;

                const Scalar hx = ok ? (ox + t * dx) : static_cast<Scalar>(0);
                const Scalar hy = ok ? (oy + t * dy) : static_cast<Scalar>(0);
                const Scalar hz = ok ? ctx->plane_z : static_cast<Scalar>(0);
                ctx->h[3 * b + 0] = hx;
                ctx->h[3 * b + 1] = hy;
                ctx->h[3 * b + 2] = hz;
            }
        };

        IntersectCtx ctx{};
        ctx.o = omap.data;
        ctx.d = dmap.data;
        ctx.h = hmap.data;
        ctx.m = m_ptr;
        ctx.plane_z = plane_z;

        submit_row_jobs(tensor_op_pool(), intersect_rows, &ctx, 0, batch);

        omap.unmap();
        dmap.unmap();
        hmap.unmap();
        mem->unmap(out_mask->handle());
        return true;
    }

    static bool scatter_add_2d(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
            return false;

        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        bool val_scalar = false;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            val_scalar = true;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            if (vd.shape.dims[1] != channels) return false;
        } else {
            return false;
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap = map_dense(points);
        MappedDense vmap = map_dense(values);
        if (!bmap.ok || !omap.ok || !pmap.ok || !vmap.ok) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            return false;
        }

        if constexpr (std::is_same_v<Scalar, float>) {
            if (!tensor_copy_f32_into(base, out)) {
                bmap.unmap();
                omap.unmap();
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            bmap.unmap();
            omap.unmap();
            MappedDense omap2 = map_dense_mut(*out);
            if (!omap2.ok) {
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            omap = omap2;
        } else {
            const uint64_t elems = bd.shape.element_count();
            std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
        }

        struct ScatterBins {
            std::vector<uint32_t> counts;
            std::vector<uint32_t> offsets;
            std::vector<uint32_t> write_pos;
            std::vector<uint32_t> indices;
            std::vector<uint64_t> tile_mask;
        };
        static thread_local ScatterBins bins;

        static uint32_t kTilePx = 0;
        if (kTilePx == 0) {
            const char* v = std::getenv("NODUS_SCATTER_TILE_PX");
            uint32_t t = 32;
            if (v && *v) {
                char* end = nullptr;
                unsigned long long parsed = std::strtoull(v, &end, 10);
                if (end != v && parsed >= 4 && parsed <= 256) {
                    t = static_cast<uint32_t>(parsed);
                }
            }
            kTilePx = t;
        }

        const uint32_t tiles_x = (width + kTilePx - 1) / kTilePx;
        const uint32_t tiles_y = (height + kTilePx - 1) / kTilePx;
        const uint32_t tile_count = tiles_x * tiles_y;

        bins.counts.assign(tile_count, 0u);
        bins.offsets.assign(tile_count + 1, 0u);
        bins.write_pos.assign(tile_count, 0u);
        bins.indices.resize(count);
        bins.tile_mask.assign((tile_count + 63u) / 64u, 0ull);

        for (uint32_t i = 0; i < count; ++i) {
            float xf = static_cast<float>(pmap.data[i * pcols + 0]);
            float yf = static_cast<float>(pmap.data[i * pcols + 1]);
            if (bd.slice.valid && bd.slice.has_affine) {
                float xo = 0.0f, yo = 0.0f, zo = 0.0f;
                apply_affine_row_major(bd.slice.affine, xf, yf, 0.0f, xo, yo, zo);
                xf = xo;
                yf = yo;
            }
            const int64_t xi = quantize_round_fast(xf);
            const int64_t yi = quantize_round_fast(yf);
            if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(width) ||
                yi >= static_cast<int64_t>(height)) {
                continue;
            }
            if (bd.slice.saturate) {
                const uint64_t base_idx = (static_cast<uint64_t>(yi) * width + static_cast<uint64_t>(xi)) * channels;
                bool locked = false;
                for (uint32_t c = 0; c < channels; ++c) {
                    if (omap.data[base_idx + c] >= static_cast<Scalar>(bd.slice.saturate_threshold)) {
                        locked = true;
                        break;
                    }
                }
                if (locked) continue;
            }
            const uint32_t tx = static_cast<uint32_t>(xi) / kTilePx;
            const uint32_t ty = static_cast<uint32_t>(yi) / kTilePx;
            const uint32_t tid = ty * tiles_x + tx;
            bins.counts[tid]++;
            const uint32_t word = tid >> 6;
            const uint32_t bit = tid & 63u;
            bins.tile_mask[word] |= (1ull << bit);
        }

        uint32_t running = 0;
        for (uint32_t t = 0; t < tile_count; ++t) {
            bins.offsets[t] = running;
            running += bins.counts[t];
        }
        bins.offsets[tile_count] = running;
        bins.write_pos = bins.offsets;

        float dense_threshold = 0.85f;
        if (const char* v = std::getenv("NODUS_SCATTER_TILE_DENSE")) {
            if (*v) {
                char* end = nullptr;
                double parsed = std::strtod(v, &end);
                if (end != v) {
                    dense_threshold = static_cast<float>(std::clamp(parsed, 0.0, 1.0));
                }
            }
        }

        bool dense_grid = false;
        if (pcols == 2 && count == static_cast<uint32_t>(static_cast<uint64_t>(width) * height)) {
            auto check_point = [&](size_t idx, uint32_t ex, uint32_t ey) -> bool {
                if (idx >= count) return false;
                const float xf = static_cast<float>(pmap.data[idx * 2 + 0]);
                const float yf = static_cast<float>(pmap.data[idx * 2 + 1]);
                const int64_t xi = quantize_round_fast(xf);
                const int64_t yi = quantize_round_fast(yf);
                return (xi == static_cast<int64_t>(ex) && yi == static_cast<int64_t>(ey));
            };
            const size_t last = static_cast<size_t>(height - 1) * width + (width - 1);
            dense_grid = check_point(0, 0u, 0u) &&
                         (width < 2 || check_point(1, 1u, 0u)) &&
                         (height < 2 || check_point(static_cast<size_t>(width), 0u, 1u)) &&
                         check_point(last, width - 1, height - 1);
        }

        for (uint32_t i = 0; i < count; ++i) {
            float xf = static_cast<float>(pmap.data[i * pcols + 0]);
            float yf = static_cast<float>(pmap.data[i * pcols + 1]);
            if (bd.slice.valid && bd.slice.has_affine) {
                float xo = 0.0f, yo = 0.0f, zo = 0.0f;
                apply_affine_row_major(bd.slice.affine, xf, yf, 0.0f, xo, yo, zo);
                xf = xo;
                yf = yo;
            }
            const int64_t xi = quantize_round_fast(xf);
            const int64_t yi = quantize_round_fast(yf);
            if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(width) ||
                yi >= static_cast<int64_t>(height)) {
                continue;
            }
            if (bd.slice.saturate) {
                const uint64_t base_idx = (static_cast<uint64_t>(yi) * width + static_cast<uint64_t>(xi)) * channels;
                bool locked = false;
                for (uint32_t c = 0; c < channels; ++c) {
                    if (omap.data[base_idx + c] >= static_cast<Scalar>(bd.slice.saturate_threshold)) {
                        locked = true;
                        break;
                    }
                }
                if (locked) continue;
            }
            const uint32_t tx = static_cast<uint32_t>(xi) / kTilePx;
            const uint32_t ty = static_cast<uint32_t>(yi) / kTilePx;
            const uint32_t tid = ty * tiles_x + tx;
            const uint32_t pos = bins.write_pos[tid]++;
            if (pos < bins.indices.size()) {
                bins.indices[pos] = i;
            }
        }

        struct ScatterCtx {
            const Scalar* points = nullptr;
            const Scalar* values = nullptr;
            Scalar* out = nullptr;
            const uint32_t* offsets = nullptr;
            const uint32_t* indices = nullptr;
            const uint64_t* tile_mask = nullptr;
            uint32_t tile_count = 0;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t channels = 0;
            uint32_t pcols = 0;
            uint32_t tiles_x = 0;
            uint32_t tile_px = 0;
            uint32_t tile_len = 0;
            TensorBackend* backend = nullptr;
            bool use_affine = false;
            float affine[16] = {0};
            bool val_scalar = false;
            bool saturate = false;
            float saturate_threshold = 0.0f;
            bool dense_grid = false;
            float dense_threshold = 0.0f;
        };

        auto scatter_rows = [](const void* vctx, uint32_t t0, uint32_t t1) {
            const auto* ctx = static_cast<const ScatterCtx*>(vctx);
            const uint32_t width = ctx->width;
            const uint32_t channels = ctx->channels;
            auto* mem = dynamic_cast<InMemoryBackend*>(ctx->backend);
            if (!mem) return;
            static thread_local AbstractTensorPool tile_pool(tile_pool_options());
            static thread_local AbstractTensorPool::PooledTensor tile_buf;
            static thread_local std::vector<uint8_t> lock_mask;
            TensorDesc tile_desc{};
            tile_desc.dtype = DType;
            tile_desc.layout = TensorLayout::Dense;
            tile_desc.shape.dims = {ctx->tile_len, channels};

            if (!tile_buf.valid() ||
                tile_buf.tensor().desc().shape.dims != tile_desc.shape.dims) {
                tile_buf = tile_pool.acquire(tile_desc, ctx->backend);
            }
            if (!tile_buf.valid()) return;

            void* tile_ptr_v = nullptr;
            size_t tile_bytes = 0;
            if (!mem->map(tile_buf.tensor().handle(), &tile_ptr_v, &tile_bytes)) return;
            auto* tile = static_cast<Scalar*>(tile_ptr_v);
            const size_t tile_elems = static_cast<size_t>(ctx->tile_len) * channels;

            for (uint32_t ti = t0; ti < t1; ++ti) {
                const uint32_t tid = ti;
                const uint32_t word = tid >> 6;
                const uint32_t bit = tid & 63u;
                if ((ctx->tile_mask[word] & (1ull << bit)) == 0ull) continue;
                const uint32_t begin = ctx->offsets[tid];
                const uint32_t end = ctx->offsets[tid + 1];
                if (begin == end) continue;
                const uint32_t ty = tid / ctx->tiles_x;
                const uint32_t tx = tid - ty * ctx->tiles_x;
                const uint32_t tile_origin_x = tx * ctx->tile_px;
                const uint32_t tile_origin_y = ty * ctx->tile_px;
                const uint32_t y_end = std::min<uint32_t>(ctx->height, tile_origin_y + ctx->tile_px);
                const uint32_t x_end = std::min<uint32_t>(ctx->width, tile_origin_x + ctx->tile_px);
                const uint32_t rows = y_end - tile_origin_y;
                const uint32_t cols = x_end - tile_origin_x;

                const uint32_t hits = end - begin;
                const uint32_t tile_area = rows * cols;
                if (ctx->dense_grid && tile_area > 0) {
                    const float density = static_cast<float>(hits) / static_cast<float>(tile_area);
                    if (density >= ctx->dense_threshold) {
                        const uint32_t out_row_stride = width * channels;
                        Scalar* out_ptr =
                            ctx->out + (static_cast<uint64_t>(tile_origin_y) * width + tile_origin_x) * channels;
                        for (uint32_t ly = 0; ly < rows; ++ly) {
                            const uint32_t y = tile_origin_y + ly;
                            Scalar* dst = out_ptr + static_cast<uint64_t>(ly) * out_row_stride;
                            if (ctx->val_scalar) {
                                const Scalar* src = ctx->values + static_cast<size_t>(y) * width + tile_origin_x;
                                for (uint32_t lx = 0; lx < cols; ++lx) {
                                    if (ctx->saturate) {
                                        bool locked = false;
                                        for (uint32_t c = 0; c < channels; ++c) {
                                            if (dst[lx * channels + c] >=
                                                static_cast<Scalar>(ctx->saturate_threshold)) {
                                                locked = true;
                                                break;
                                            }
                                        }
                                        if (locked) continue;
                                    }
                                    const Scalar v = src[lx];
                                    for (uint32_t c = 0; c < channels; ++c) {
                                        Scalar out_v = dst[lx * channels + c] + v;
                                        if (ctx->saturate && out_v > static_cast<Scalar>(ctx->saturate_threshold)) {
                                            out_v = static_cast<Scalar>(ctx->saturate_threshold);
                                        }
                                        dst[lx * channels + c] = out_v;
                                    }
                                }
                            } else {
                                const Scalar* src = ctx->values +
                                    (static_cast<size_t>(y) * width + tile_origin_x) * channels;
                                for (uint32_t lx = 0; lx < cols; ++lx) {
                                    if (ctx->saturate) {
                                        bool locked = false;
                                        for (uint32_t c = 0; c < channels; ++c) {
                                            if (dst[lx * channels + c] >=
                                                static_cast<Scalar>(ctx->saturate_threshold)) {
                                                locked = true;
                                                break;
                                            }
                                        }
                                        if (locked) continue;
                                    }
                                    const Scalar* src_row = src + static_cast<uint64_t>(lx) * channels;
                                    for (uint32_t c = 0; c < channels; ++c) {
                                        Scalar out_v = dst[lx * channels + c] + src_row[c];
                                        if (ctx->saturate && out_v > static_cast<Scalar>(ctx->saturate_threshold)) {
                                            out_v = static_cast<Scalar>(ctx->saturate_threshold);
                                        }
                                        dst[lx * channels + c] = out_v;
                                    }
                                }
                            }
                        }
                        continue;
                    }
                }

                std::memset(tile, 0, tile_elems * sizeof(Scalar));
                if (ctx->saturate) {
                    if (lock_mask.size() != ctx->tile_len) {
                        lock_mask.resize(ctx->tile_len);
                    }
                    std::fill(lock_mask.begin(), lock_mask.end(), 0u);
                    for (uint32_t ly = 0; ly < rows; ++ly) {
                        const uint32_t y = tile_origin_y + ly;
                        const uint64_t row_base = static_cast<uint64_t>(y) * width;
                        for (uint32_t lx = 0; lx < cols; ++lx) {
                            const uint32_t x = tile_origin_x + lx;
                            const uint64_t base_idx = (row_base + x) * channels;
                            bool locked = false;
                            for (uint32_t c = 0; c < channels; ++c) {
                                if (ctx->out[base_idx + c] >=
                                    static_cast<Scalar>(ctx->saturate_threshold)) {
                                    locked = true;
                                    break;
                                }
                            }
                            if (locked) {
                                lock_mask[ly * ctx->tile_px + lx] = 1u;
                            }
                        }
                    }
                }
                for (uint32_t idx = begin; idx < end; ++idx) {
                    const uint32_t i = ctx->indices[idx];
                    float xf = static_cast<float>(ctx->points[i * ctx->pcols + 0]);
                    float yf = static_cast<float>(ctx->points[i * ctx->pcols + 1]);
                    if (ctx->use_affine) {
                        float xo = 0.0f, yo = 0.0f, zo = 0.0f;
                        apply_affine_row_major(ctx->affine, xf, yf, 0.0f, xo, yo, zo);
                        xf = xo;
                        yf = yo;
                    }
                    const int64_t xi = quantize_round_fast(xf);
                    const int64_t yi = quantize_round_fast(yf);
                    if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(width) ||
                        yi >= static_cast<int64_t>(ctx->height)) {
                        continue;
                    }
                    const uint32_t lx = static_cast<uint32_t>(xi) - tile_origin_x;
                    const uint32_t ly = static_cast<uint32_t>(yi) - tile_origin_y;
                    if (lx >= ctx->tile_px || ly >= ctx->tile_px) continue;
                    const uint32_t local = ly * ctx->tile_px + lx;
                    const uint64_t base_idx = static_cast<uint64_t>(local) * channels;
                    if (ctx->saturate && lock_mask[local]) continue;
                    if (ctx->val_scalar) {
                        const Scalar v = ctx->values[i];
                        for (uint32_t c = 0; c < channels; ++c) {
                            tile[base_idx + c] += v;
                        }
                    } else {
                        const Scalar* src = ctx->values + static_cast<uint64_t>(i) * channels;
                        for (uint32_t c = 0; c < channels; ++c) {
                            tile[base_idx + c] += src[c];
                        }
                    }
                    if (ctx->saturate) {
                        const uint64_t out_idx =
                            (static_cast<uint64_t>(yi) * width + static_cast<uint64_t>(xi)) * channels;
                        bool hit = false;
                        for (uint32_t c = 0; c < channels; ++c) {
                            const Scalar base_v = ctx->out[out_idx + c];
                            Scalar v = base_v + tile[base_idx + c];
                            if (v >= static_cast<Scalar>(ctx->saturate_threshold)) {
                                v = static_cast<Scalar>(ctx->saturate_threshold) - base_v;
                                tile[base_idx + c] = v;
                                hit = true;
                            }
                        }
                        if (hit) {
                            lock_mask[local] = 1u;
                        }
                    }
                }

                const uint32_t out_row_stride = width * channels;
                const uint32_t x_row_stride = ctx->tile_px * channels;
                if constexpr (std::is_same_v<Scalar, float>) {
                    float* out_ptr =
                        ctx->out + (static_cast<uint64_t>(tile_origin_y) * width + tile_origin_x) * channels;
                    nodus_fused_add_f32_strided(out_ptr,
                                                out_row_stride,
                                                tile,
                                                x_row_stride,
                                                rows,
                                                cols,
                                                channels,
                                                ctx->saturate ? 1u : 0u,
                                                ctx->saturate_threshold);
                } else {
                    Scalar* out_ptr =
                        ctx->out + (static_cast<uint64_t>(tile_origin_y) * width + tile_origin_x) * channels;
                    for (uint32_t r = 0; r < rows; ++r) {
                        Scalar* dst = out_ptr + static_cast<uint64_t>(r) * out_row_stride;
                        const Scalar* xv = tile + static_cast<uint64_t>(r) * x_row_stride;
                        for (uint32_t j = 0; j < cols; ++j) {
                            Scalar* drow = dst + static_cast<uint64_t>(j) * channels;
                            const Scalar* xrow = xv + static_cast<uint64_t>(j) * channels;
                            for (uint32_t cc = 0; cc < channels; ++cc) {
                                Scalar v = drow[cc] + xrow[cc];
                                if (ctx->saturate && v > static_cast<Scalar>(ctx->saturate_threshold)) {
                                    v = static_cast<Scalar>(ctx->saturate_threshold);
                                }
                                drow[cc] = v;
                            }
                        }
                    }
                }
            }
            mem->unmap(tile_buf.tensor().handle());
        };

        ScatterCtx ctx{};
        ctx.points = pmap.data;
        ctx.values = vmap.data;
        ctx.out = omap.data;
        ctx.offsets = bins.offsets.data();
        ctx.indices = bins.indices.data();
        ctx.tile_mask = bins.tile_mask.data();
        ctx.tile_count = tile_count;
        ctx.width = width;
        ctx.height = height;
        ctx.channels = channels;
        ctx.pcols = pcols;
        ctx.tiles_x = tiles_x;
        ctx.tile_px = kTilePx;
        ctx.tile_len = kTilePx * kTilePx;
        ctx.backend = base.backend();
        ctx.use_affine = bd.slice.valid && bd.slice.has_affine;
        if (ctx.use_affine) {
            std::memcpy(ctx.affine, bd.slice.affine, sizeof(ctx.affine));
        }
        ctx.val_scalar = val_scalar;
        ctx.saturate = bd.slice.saturate;
        ctx.saturate_threshold = bd.slice.saturate_threshold;
        ctx.dense_grid = dense_grid;
        ctx.dense_threshold = dense_threshold;

        submit_row_jobs(tensor_op_pool(), scatter_rows, &ctx, 0, ctx.tile_count);

        bmap.unmap();
        omap.unmap();
        pmap.unmap();
        vmap.unmap();
        return true;
    }

    static bool scatter_add_nd_1d(const AbstractTensor& base,
                                  const AbstractTensor& points,
                                  const AbstractTensor& values,
                                  AbstractTensor* out,
                                  bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
            return false;
        if (pd.shape.dims.size() != 2 || pd.shape.dims[1] != 1) return false;
        const uint32_t count = pd.shape.dims[0];

        const size_t out_rank = bd.shape.dims.size();
        if (out_rank != 1 && out_rank != 2) return false;
        const uint32_t channels = (out_rank == 2) ? bd.shape.dims[1] : 1u;

        bool val_scalar = false;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            val_scalar = true;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            if (vd.shape.dims[1] != channels) return false;
        } else {
            return false;
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap = map_dense(points);
        MappedDense vmap = map_dense(values);
        if (!bmap.ok || !omap.ok || !pmap.ok || !vmap.ok) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            return false;
        }

        if constexpr (std::is_same_v<Scalar, float>) {
            if (!tensor_copy_f32_into(base, out)) {
                bmap.unmap();
                omap.unmap();
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            bmap.unmap();
            omap.unmap();
            MappedDense omap2 = map_dense_mut(*out);
            if (!omap2.ok) {
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            omap = omap2;
        } else {
            const uint64_t elems = bd.shape.element_count();
            std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
        }

        const bool use_affine = bd.slice.valid && bd.slice.has_affine;
        const uint64_t stride0 = channels;
        const uint64_t limit0 = bd.shape.dims[0];

        for (uint32_t i = 0; i < count; ++i) {
            float xf = static_cast<float>(pmap.data[i]);
            if (use_affine) {
                float xo = 0.0f, yo = 0.0f, zo = 0.0f;
                apply_affine_row_major(bd.slice.affine, xf, 0.0f, 0.0f, xo, yo, zo);
                xf = xo;
            }
            const int64_t xi = quantize_round_fast(xf);
            if (xi < 0 || xi >= static_cast<int64_t>(limit0)) {
                if (clamp) continue;
                continue;
            }
            const uint64_t base_idx = static_cast<uint64_t>(xi) * stride0;
            if (bd.slice.saturate) {
                bool locked = false;
                for (uint32_t c = 0; c < channels; ++c) {
                    if (omap.data[base_idx + c] >= static_cast<Scalar>(bd.slice.saturate_threshold)) {
                        locked = true;
                        break;
                    }
                }
                if (locked) continue;
            }
            if (channels == 1) {
                const Scalar v = val_scalar ? vmap.data[i] : vmap.data[i * channels];
                Scalar out_v = omap.data[base_idx] + v;
                if (bd.slice.saturate && out_v > static_cast<Scalar>(bd.slice.saturate_threshold)) {
                    out_v = static_cast<Scalar>(bd.slice.saturate_threshold);
                }
                omap.data[base_idx] = out_v;
            } else {
                if (val_scalar) {
                    const Scalar v = vmap.data[i];
                    for (uint32_t c = 0; c < channels; ++c) {
                        Scalar out_v = omap.data[base_idx + c] + v;
                        if (bd.slice.saturate && out_v > static_cast<Scalar>(bd.slice.saturate_threshold)) {
                            out_v = static_cast<Scalar>(bd.slice.saturate_threshold);
                        }
                        omap.data[base_idx + c] = out_v;
                    }
                } else {
                    const Scalar* src = vmap.data + static_cast<uint64_t>(i) * channels;
                    for (uint32_t c = 0; c < channels; ++c) {
                        Scalar out_v = omap.data[base_idx + c] + src[c];
                        if (bd.slice.saturate && out_v > static_cast<Scalar>(bd.slice.saturate_threshold)) {
                            out_v = static_cast<Scalar>(bd.slice.saturate_threshold);
                        }
                        omap.data[base_idx + c] = out_v;
                    }
                }
            }
        }

        pmap.unmap();
        vmap.unmap();
        omap.unmap();
        return true;
    }

    static bool scatter_add_nd_3d(const AbstractTensor& base,
                                  const AbstractTensor& points,
                                  const AbstractTensor& values,
                                  AbstractTensor* out,
                                  bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
            return false;
        if (pd.shape.dims.size() != 2 || pd.shape.dims[1] != 3) return false;
        const uint32_t count = pd.shape.dims[0];

        const size_t out_rank = bd.shape.dims.size();
        if (out_rank != 3 && out_rank != 4) return false;
        const uint32_t channels = (out_rank == 4) ? bd.shape.dims[3] : 1u;

        bool val_scalar = false;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            val_scalar = true;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            if (vd.shape.dims[1] != channels) return false;
        } else {
            return false;
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap = map_dense(points);
        MappedDense vmap = map_dense(values);
        if (!bmap.ok || !omap.ok || !pmap.ok || !vmap.ok) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            return false;
        }

        if constexpr (std::is_same_v<Scalar, float>) {
            if (!tensor_copy_f32_into(base, out)) {
                bmap.unmap();
                omap.unmap();
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            bmap.unmap();
            omap.unmap();
            MappedDense omap2 = map_dense_mut(*out);
            if (!omap2.ok) {
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            omap = omap2;
        } else {
            const uint64_t elems = bd.shape.element_count();
            std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
        }

        std::vector<uint64_t> strides;
        if (!compute_dense_strides_u64(bd.shape.dims, strides)) {
            pmap.unmap();
            vmap.unmap();
            omap.unmap();
            return false;
        }

        const bool use_affine = bd.slice.valid && bd.slice.has_affine;
        for (uint32_t i = 0; i < count; ++i) {
            float xf = static_cast<float>(pmap.data[i * 3 + 0]);
            float yf = static_cast<float>(pmap.data[i * 3 + 1]);
            float zf = static_cast<float>(pmap.data[i * 3 + 2]);
            if (use_affine) {
                float xo = 0.0f, yo = 0.0f, zo = 0.0f;
                apply_affine_row_major(bd.slice.affine, xf, yf, zf, xo, yo, zo);
                xf = xo;
                yf = yo;
                zf = zo;
            }
            const int64_t xi = quantize_round_fast(xf);
            const int64_t yi = quantize_round_fast(yf);
            const int64_t zi = quantize_round_fast(zf);
            if (xi < 0 || yi < 0 || zi < 0 ||
                xi >= static_cast<int64_t>(bd.shape.dims[0]) ||
                yi >= static_cast<int64_t>(bd.shape.dims[1]) ||
                zi >= static_cast<int64_t>(bd.shape.dims[2])) {
                if (clamp) continue;
                continue;
            }
            const uint64_t base_idx =
                static_cast<uint64_t>(xi) * strides[0] +
                static_cast<uint64_t>(yi) * strides[1] +
                static_cast<uint64_t>(zi) * strides[2];
            if (bd.slice.saturate) {
                bool locked = false;
                for (uint32_t c = 0; c < channels; ++c) {
                    if (omap.data[base_idx + c] >= static_cast<Scalar>(bd.slice.saturate_threshold)) {
                        locked = true;
                        break;
                    }
                }
                if (locked) continue;
            }
            if (channels == 1) {
                const Scalar v = val_scalar ? vmap.data[i] : vmap.data[i * channels];
                Scalar out_v = omap.data[base_idx] + v;
                if (bd.slice.saturate && out_v > static_cast<Scalar>(bd.slice.saturate_threshold)) {
                    out_v = static_cast<Scalar>(bd.slice.saturate_threshold);
                }
                omap.data[base_idx] = out_v;
            } else {
                if (val_scalar) {
                    const Scalar v = vmap.data[i];
                    for (uint32_t c = 0; c < channels; ++c) {
                        Scalar out_v = omap.data[base_idx + c] + v;
                        if (bd.slice.saturate && out_v > static_cast<Scalar>(bd.slice.saturate_threshold)) {
                            out_v = static_cast<Scalar>(bd.slice.saturate_threshold);
                        }
                        omap.data[base_idx + c] = out_v;
                    }
                } else {
                    const Scalar* src = vmap.data + static_cast<uint64_t>(i) * channels;
                    for (uint32_t c = 0; c < channels; ++c) {
                        Scalar out_v = omap.data[base_idx + c] + src[c];
                        if (bd.slice.saturate && out_v > static_cast<Scalar>(bd.slice.saturate_threshold)) {
                            out_v = static_cast<Scalar>(bd.slice.saturate_threshold);
                        }
                        omap.data[base_idx + c] = out_v;
                    }
                }
            }
        }

        pmap.unmap();
        vmap.unmap();
        omap.unmap();
        return true;
    }

    static bool scatter_add_nd_4d(const AbstractTensor& base,
                                  const AbstractTensor& points,
                                  const AbstractTensor& values,
                                  AbstractTensor* out,
                                  bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
            return false;
        if (pd.shape.dims.size() != 2 || pd.shape.dims[1] != 4) return false;
        const uint32_t count = pd.shape.dims[0];

        const size_t out_rank = bd.shape.dims.size();
        if (out_rank != 4 && out_rank != 5) return false;
        const uint32_t channels = (out_rank == 5) ? bd.shape.dims[4] : 1u;

        bool val_scalar = false;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            val_scalar = true;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            if (vd.shape.dims[1] != channels) return false;
        } else {
            return false;
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap = map_dense(points);
        MappedDense vmap = map_dense(values);
        if (!bmap.ok || !omap.ok || !pmap.ok || !vmap.ok) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            return false;
        }

        if constexpr (std::is_same_v<Scalar, float>) {
            if (!tensor_copy_f32_into(base, out)) {
                bmap.unmap();
                omap.unmap();
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            bmap.unmap();
            omap.unmap();
            MappedDense omap2 = map_dense_mut(*out);
            if (!omap2.ok) {
                pmap.unmap();
                vmap.unmap();
                return false;
            }
            omap = omap2;
        } else {
            const uint64_t elems = bd.shape.element_count();
            std::memcpy(omap.data, bmap.data, static_cast<size_t>(elems * sizeof(Scalar)));
        }

        std::vector<uint64_t> strides;
        if (!compute_dense_strides_u64(bd.shape.dims, strides)) {
            pmap.unmap();
            vmap.unmap();
            omap.unmap();
            return false;
        }

        for (uint32_t i = 0; i < count; ++i) {
            const int64_t xi = quantize_round_fast(static_cast<float>(pmap.data[i * 4 + 0]));
            const int64_t yi = quantize_round_fast(static_cast<float>(pmap.data[i * 4 + 1]));
            const int64_t zi = quantize_round_fast(static_cast<float>(pmap.data[i * 4 + 2]));
            const int64_t wi = quantize_round_fast(static_cast<float>(pmap.data[i * 4 + 3]));
            if (xi < 0 || yi < 0 || zi < 0 || wi < 0 ||
                xi >= static_cast<int64_t>(bd.shape.dims[0]) ||
                yi >= static_cast<int64_t>(bd.shape.dims[1]) ||
                zi >= static_cast<int64_t>(bd.shape.dims[2]) ||
                wi >= static_cast<int64_t>(bd.shape.dims[3])) {
                if (clamp) continue;
                continue;
            }
            const uint64_t base_idx =
                static_cast<uint64_t>(xi) * strides[0] +
                static_cast<uint64_t>(yi) * strides[1] +
                static_cast<uint64_t>(zi) * strides[2] +
                static_cast<uint64_t>(wi) * strides[3];
            if (bd.slice.saturate) {
                bool locked = false;
                for (uint32_t c = 0; c < channels; ++c) {
                    if (omap.data[base_idx + c] >= static_cast<Scalar>(bd.slice.saturate_threshold)) {
                        locked = true;
                        break;
                    }
                }
                if (locked) continue;
            }
            if (channels == 1) {
                const Scalar v = val_scalar ? vmap.data[i] : vmap.data[i * channels];
                Scalar out_v = omap.data[base_idx] + v;
                if (bd.slice.saturate && out_v > static_cast<Scalar>(bd.slice.saturate_threshold)) {
                    out_v = static_cast<Scalar>(bd.slice.saturate_threshold);
                }
                omap.data[base_idx] = out_v;
            } else {
                if (val_scalar) {
                    const Scalar v = vmap.data[i];
                    for (uint32_t c = 0; c < channels; ++c) {
                        Scalar out_v = omap.data[base_idx + c] + v;
                        if (bd.slice.saturate && out_v > static_cast<Scalar>(bd.slice.saturate_threshold)) {
                            out_v = static_cast<Scalar>(bd.slice.saturate_threshold);
                        }
                        omap.data[base_idx + c] = out_v;
                    }
                } else {
                    const Scalar* src = vmap.data + static_cast<uint64_t>(i) * channels;
                    for (uint32_t c = 0; c < channels; ++c) {
                        Scalar out_v = omap.data[base_idx + c] + src[c];
                        if (bd.slice.saturate && out_v > static_cast<Scalar>(bd.slice.saturate_threshold)) {
                            out_v = static_cast<Scalar>(bd.slice.saturate_threshold);
                        }
                        omap.data[base_idx + c] = out_v;
                    }
                }
            }
        }

        pmap.unmap();
        vmap.unmap();
        omap.unmap();
        return true;
    }

    static bool scatter_add_nd(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;

        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
            return false;
        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t dims = pd.shape.dims[1];
        if (dims == 0) return false;

        const size_t out_rank = bd.shape.dims.size();
        if (out_rank < dims || out_rank > dims + 1) return false;

        if (dims == 1) return scatter_add_nd_1d(base, points, values, out, clamp);
        if (dims == 2) return scatter_add_2d(base, points, values, out, clamp);
        if (dims == 3) return scatter_add_nd_3d(base, points, values, out, clamp);
        if (dims == 4) return scatter_add_nd_4d(base, points, values, out, clamp);

        const uint32_t channels = (out_rank == dims + 1) ? bd.shape.dims.back() : 1u;
        bool val_scalar = false;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            val_scalar = true;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            if (vd.shape.dims[1] != channels) return false;
        } else {
            return false;
        }

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap = map_dense(points);
        MappedDense vmap = map_dense(values);
        if (!bmap.ok || !omap.ok || !pmap.ok || !vmap.ok) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            return false;
        }

        if (!tensor_copy_f32_into(base, out)) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            return false;
        }
        bmap.unmap();
        omap.unmap();
        MappedDense omap2 = map_dense_mut(*out);
        if (!omap2.ok) {
            pmap.unmap();
            vmap.unmap();
            return false;
        }
        omap = omap2;

        std::vector<uint64_t> strides;
        if (!compute_dense_strides_u64(bd.shape.dims, strides)) {
            pmap.unmap();
            vmap.unmap();
            omap.unmap();
            return false;
        }

        const bool use_affine = bd.slice.valid && bd.slice.has_affine && dims <= 3;

        for (uint32_t i = 0; i < count; ++i) {
            float xf = static_cast<float>(pmap.data[i * dims + 0]);
            float yf = (dims > 1) ? static_cast<float>(pmap.data[i * dims + 1]) : 0.0f;
            float zf = (dims > 2) ? static_cast<float>(pmap.data[i * dims + 2]) : 0.0f;
            if (use_affine) {
                float xo = 0.0f, yo = 0.0f, zo = 0.0f;
                apply_affine_row_major(bd.slice.affine, xf, yf, zf, xo, yo, zo);
                xf = xo;
                yf = yo;
                zf = zo;
            }

            int64_t coords[4] = {0, 0, 0, 0};
            coords[0] = quantize_round_fast(xf);
            if (dims > 1) coords[1] = quantize_round_fast(yf);
            if (dims > 2) coords[2] = quantize_round_fast(zf);
            if (dims > 3) {
                coords[3] = quantize_round_fast(static_cast<float>(pmap.data[i * dims + 3]));
            }

            bool oob = false;
            for (uint32_t d = 0; d < dims; ++d) {
                if (coords[d] < 0 || coords[d] >= static_cast<int64_t>(bd.shape.dims[d])) {
                    oob = true;
                    break;
                }
            }
            if (oob) {
                if (clamp) continue;
                continue;
            }

            uint64_t base_idx = 0;
            for (uint32_t d = 0; d < dims; ++d) {
                base_idx += static_cast<uint64_t>(coords[d]) * strides[d];
            }

            if (bd.slice.saturate) {
                bool locked = false;
                for (uint32_t c = 0; c < channels; ++c) {
                    if (omap.data[base_idx + c] >= static_cast<Scalar>(bd.slice.saturate_threshold)) {
                        locked = true;
                        break;
                    }
                }
                if (locked) continue;
            }

            if (channels == 1) {
                const Scalar v = val_scalar ? vmap.data[i] : vmap.data[i * channels];
                Scalar out_v = omap.data[base_idx] + v;
                if (bd.slice.saturate && out_v > static_cast<Scalar>(bd.slice.saturate_threshold)) {
                    out_v = static_cast<Scalar>(bd.slice.saturate_threshold);
                }
                omap.data[base_idx] = out_v;
            } else {
                if (val_scalar) {
                    const Scalar v = vmap.data[i];
                    for (uint32_t c = 0; c < channels; ++c) {
                        Scalar out_v = omap.data[base_idx + c] + v;
                        if (bd.slice.saturate && out_v > static_cast<Scalar>(bd.slice.saturate_threshold)) {
                            out_v = static_cast<Scalar>(bd.slice.saturate_threshold);
                        }
                        omap.data[base_idx + c] = out_v;
                    }
                } else {
                    const Scalar* src = vmap.data + static_cast<uint64_t>(i) * channels;
                    for (uint32_t c = 0; c < channels; ++c) {
                        Scalar out_v = omap.data[base_idx + c] + src[c];
                        if (bd.slice.saturate && out_v > static_cast<Scalar>(bd.slice.saturate_threshold)) {
                            out_v = static_cast<Scalar>(bd.slice.saturate_threshold);
                        }
                        omap.data[base_idx + c] = out_v;
                    }
                }
            }
        }

        pmap.unmap();
        vmap.unmap();
        omap.unmap();
        return true;
    }

    static bool scatter_add_2d_kernel(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out,
                                      bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid() || !kernel.valid()) return false;
        if (base.backend() != points.backend() ||
            base.backend() != values.backend() ||
            base.backend() != kernel.backend()) {
            return false;
        }
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        const TensorDesc& kd = kernel.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType || kd.dtype != DType) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense ||
            vd.layout != TensorLayout::Dense || kd.layout != TensorLayout::Dense) {
            return false;
        }
        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        uint32_t kcols = 1;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            kcols = 1;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            kcols = vd.shape.dims[1];
        } else {
            return false;
        }

        if (kd.shape.dims.size() != 2) return false;
        if (kd.shape.dims[0] != kcols) return false;
        if (kd.shape.dims[1] != channels) return false;

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap = map_dense(points);
        MappedDense vmap = map_dense(values);
        MappedDense kmap = map_dense(kernel);
        if (!bmap.ok || !omap.ok || !pmap.ok || !vmap.ok || !kmap.ok) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }

        const bool use_affine = bd.slice.valid && bd.slice.has_affine;

        if (!tensor_copy_f32_into(base, out)) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }
        bmap.unmap();
        omap.unmap();
        MappedDense omap2 = map_dense_mut(*out);
        if (!omap2.ok) {
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }
        omap = omap2;

        for (uint32_t i = 0; i < count; ++i) {
            float xf = static_cast<float>(pmap.data[i * pcols + 0]);
            float yf = static_cast<float>(pmap.data[i * pcols + 1]);
            if (use_affine) {
                float xo = 0.0f, yo = 0.0f, zo = 0.0f;
                apply_affine_row_major(bd.slice.affine, xf, yf, 0.0f, xo, yo, zo);
                xf = xo;
                yf = yo;
            }
            const int64_t xi = quantize_round_fast(xf);
            const int64_t yi = quantize_round_fast(yf);
            if (clamp) {
                if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(width) ||
                    yi >= static_cast<int64_t>(height)) {
                    continue;
                }
            }
            if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(width) ||
                yi >= static_cast<int64_t>(height)) {
                continue;
            }
            const uint64_t base_idx = (static_cast<uint64_t>(yi) * width + static_cast<uint64_t>(xi)) * channels;
            for (uint32_t c = 0; c < channels; ++c) {
                Scalar acc = static_cast<Scalar>(0);
                if (vd.shape.dims.size() == 1) {
                    const Scalar v = vmap.data[i];
                    acc = v * kmap.data[c];
                } else {
                    const Scalar* vrow = vmap.data + static_cast<uint64_t>(i) * kcols;
                    const Scalar* kcol = kmap.data + c;
                    for (uint32_t k = 0; k < kcols; ++k) {
                        acc += vrow[k] * kcol[k * channels];
                    }
                }
                omap.data[base_idx + c] += acc;
            }
        }

        bmap.unmap();
        omap.unmap();
        pmap.unmap();
        vmap.unmap();
        kmap.unmap();
        return true;
    }

    static bool scatter_add_2d_kernel_fn_impl(const AbstractTensor& base,
                                              const AbstractTensor& points,
                                              const AbstractTensor& values,
                                              const std::function<Scalar(uint32_t, uint32_t, void*)>& kernel_fn,
                                              void* user,
                                              AbstractTensor* out,
                                              KernelDispatchPlan* out_plan,
                                              bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid()) return false;
        if (!kernel_fn) return false;
        if (!out_plan) return false;
        if (base.backend() != points.backend() || base.backend() != values.backend()) return false;
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense || vd.layout != TensorLayout::Dense)
            return false;
        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        uint32_t kcols = 1;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            kcols = 1;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            kcols = vd.shape.dims[1];
        } else {
            return false;
        }

        TensorDesc kdesc{};
        kdesc.dtype = DType;
        kdesc.layout = TensorLayout::Dense;
        kdesc.shape.dims = {kcols, channels};
        AbstractTensor kernel = AbstractTensor::create(kdesc, base.backend());
        if (!kernel.valid()) return false;

        out_plan->entries.clear();
        out_plan->slot_ids.clear();
        KernelDispatchEntry entry{};
        entry.id = 0;
        entry.fn = nullptr;
        entry.user = user;
        out_plan->entries.push_back(entry);
        out_plan->slot_ids.resize(static_cast<size_t>(kcols) * channels, 0u);

        MappedDense kmap = map_dense_mut(kernel);
        if (!kmap.ok) {
            kmap.unmap();
            return false;
        }
        for (uint32_t k = 0; k < kcols; ++k) {
            for (uint32_t c = 0; c < channels; ++c) {
                kmap.data[static_cast<size_t>(k) * channels + c] = kernel_fn(k, c, user);
            }
        }
        kmap.unmap();

        // Single broadcast-style kernel op: reuse the standard kernel scatter once.
        return scatter_add_2d_kernel(base, points, values, kernel, out, clamp);
    }

    static bool scatter_add_2d_kernel_ptr_impl(const AbstractTensor& base,
                                              const AbstractTensor& points,
                                              const AbstractTensor& values,
                                              const AbstractTensor& kernel_ptrs,
                                              AbstractTensor* out,
                                              KernelDispatchPlan* out_plan,
                                              bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid() || !kernel_ptrs.valid()) return false;
        if (!out_plan) return false;
        if (base.backend() != points.backend() ||
            base.backend() != values.backend() ||
            base.backend() != kernel_ptrs.backend()) {
            return false;
        }
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        const TensorDesc& kd = kernel_ptrs.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType) return false;
        if (kd.dtype != TensorDType::Ptr) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense ||
            vd.layout != TensorLayout::Dense || kd.layout != TensorLayout::Dense) {
            return false;
        }
        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        uint32_t kcols = 1;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            kcols = 1;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            kcols = vd.shape.dims[1];
        } else {
            return false;
        }

        if (kd.shape.dims.size() != 2) return false;
        if (kd.shape.dims[0] != kcols) return false;
        if (kd.shape.dims[1] != channels) return false;

        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (!mem) return false;
        void* k_ptr_v = nullptr;
        size_t k_bytes = 0;
        if (!mem->map(kernel_ptrs.handle(), &k_ptr_v, &k_bytes)) {
            return false;
        }
        auto** k_ptrs = static_cast<void**>(k_ptr_v);

        using KernelEntry = std::conditional_t<std::is_same_v<Scalar, float>,
                                               TensorKernelPtrEntryF32,
                                               TensorKernelPtrEntryF64>;

        TensorDesc kdesc{};
        kdesc.dtype = DType;
        kdesc.layout = TensorLayout::Dense;
        kdesc.shape.dims = {kcols, channels};
        AbstractTensor kernel = AbstractTensor::create(kdesc, base.backend());
        if (!kernel.valid()) {
            mem->unmap(kernel_ptrs.handle());
            return false;
        }

        out_plan->entries.clear();
        out_plan->slot_ids.clear();

        MappedDense kmap = map_dense_mut(kernel);
        if (!kmap.ok) {
            mem->unmap(kernel_ptrs.handle());
            kmap.unmap();
            return false;
        }

        for (uint32_t k = 0; k < kcols; ++k) {
            for (uint32_t c = 0; c < channels; ++c) {
                const size_t slot = static_cast<size_t>(k) * channels + c;
                auto* entry = static_cast<KernelEntry*>(k_ptrs[slot]);
                if (entry && entry->fn) {
                    kmap.data[slot] = static_cast<Scalar>(entry->fn(k, c, entry->user));
                } else {
                    kmap.data[slot] = static_cast<Scalar>(0);
                }
                const void* fn_ptr = entry ? reinterpret_cast<const void*>(entry->fn) : nullptr;
                void* user_ptr = entry ? entry->user : nullptr;
                uint32_t id = 0;
                bool found = false;
                for (size_t i = 0; i < out_plan->entries.size(); ++i) {
                    if (out_plan->entries[i].fn == fn_ptr && out_plan->entries[i].user == user_ptr) {
                        id = out_plan->entries[i].id;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    id = static_cast<uint32_t>(out_plan->entries.size());
                    KernelDispatchEntry info{};
                    info.id = id;
                    info.fn = fn_ptr;
                    info.user = user_ptr;
                    out_plan->entries.push_back(info);
                }
                out_plan->slot_ids.push_back(id);
            }
        }

        kmap.unmap();
        mem->unmap(kernel_ptrs.handle());

        // Single broadcast-style kernel op: reuse the standard kernel scatter once.
        return scatter_add_2d_kernel(base, points, values, kernel, out, clamp);
    }

    static bool scatter_add_2d_kernel_bank(const AbstractTensor& base,
                                           const AbstractTensor& points,
                                           const AbstractTensor& values,
                                           const AbstractTensor& kernel_bank,
                                           const AbstractTensor& kernel_ids,
                                           AbstractTensor* out,
                                           bool clamp) {
        if (!out) return false;
        out->reset();
        if (!base.valid() || !points.valid() || !values.valid() ||
            !kernel_bank.valid() || !kernel_ids.valid()) {
            return false;
        }
        if (base.backend() != points.backend() ||
            base.backend() != values.backend() ||
            base.backend() != kernel_bank.backend() ||
            base.backend() != kernel_ids.backend()) {
            return false;
        }
        const TensorDesc& bd = base.desc();
        const TensorDesc& pd = points.desc();
        const TensorDesc& vd = values.desc();
        const TensorDesc& kd = kernel_bank.desc();
        const TensorDesc& idd = kernel_ids.desc();
        if (bd.dtype != DType || pd.dtype != DType || vd.dtype != DType || kd.dtype != DType) return false;
        if (idd.dtype != TensorDType::I32 && idd.dtype != TensorDType::U32) return false;
        if (bd.layout != TensorLayout::Dense || pd.layout != TensorLayout::Dense ||
            vd.layout != TensorLayout::Dense || kd.layout != TensorLayout::Dense ||
            idd.layout != TensorLayout::Dense) {
            return false;
        }
        if (bd.shape.dims.size() != 2 && bd.shape.dims.size() != 3) return false;
        const uint32_t height = bd.shape.dims[0];
        const uint32_t width = bd.shape.dims[1];
        const uint32_t channels = (bd.shape.dims.size() == 3) ? bd.shape.dims[2] : 1;

        if (pd.shape.dims.size() != 2) return false;
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) return false;

        if (idd.shape.dims.size() != 1 || idd.shape.dims[0] != count) return false;

        uint32_t kcols = 1;
        if (vd.shape.dims.size() == 1) {
            if (vd.shape.dims[0] != count) return false;
            kcols = 1;
        } else if (vd.shape.dims.size() == 2) {
            if (vd.shape.dims[0] != count) return false;
            kcols = vd.shape.dims[1];
        } else {
            return false;
        }

        if (kd.shape.dims.size() != 3) return false;
        const uint32_t kernel_count = kd.shape.dims[0];
        if (kd.shape.dims[1] != kcols) return false;
        if (kd.shape.dims[2] != channels) return false;

        TensorDesc out_desc = bd;
        *out = AbstractTensor::create(out_desc, base.backend());
        if (!out->valid()) return false;

        MappedDense bmap = map_dense(base);
        MappedDense omap = map_dense_mut(*out);
        MappedDense pmap = map_dense(points);
        MappedDense vmap = map_dense(values);
        MappedDense kmap = map_dense(kernel_bank);
        if (!bmap.ok || !omap.ok || !pmap.ok || !vmap.ok || !kmap.ok) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }

        auto* mem = dynamic_cast<InMemoryBackend*>(base.backend());
        if (!mem) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }
        void* ids_ptr_v = nullptr;
        size_t ids_bytes = 0;
        if (!mem->map(kernel_ids.handle(), &ids_ptr_v, &ids_bytes)) {
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }

        const bool use_affine = bd.slice.valid && bd.slice.has_affine;

        if (!tensor_copy_f32_into(base, out)) {
            mem->unmap(kernel_ids.handle());
            bmap.unmap();
            omap.unmap();
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }
        bmap.unmap();
        omap.unmap();
        MappedDense omap2 = map_dense_mut(*out);
        if (!omap2.ok) {
            mem->unmap(kernel_ids.handle());
            pmap.unmap();
            vmap.unmap();
            kmap.unmap();
            return false;
        }
        omap = omap2;

        for (uint32_t i = 0; i < count; ++i) {
            uint32_t kid = 0;
            if (idd.dtype == TensorDType::I32) {
                kid = static_cast<uint32_t>(static_cast<int32_t*>(ids_ptr_v)[i]);
            } else {
                kid = static_cast<uint32_t>(static_cast<uint32_t*>(ids_ptr_v)[i]);
            }
            if (kid >= kernel_count) continue;

            float xf = static_cast<float>(pmap.data[i * pcols + 0]);
            float yf = static_cast<float>(pmap.data[i * pcols + 1]);
            if (use_affine) {
                float xo = 0.0f, yo = 0.0f, zo = 0.0f;
                apply_affine_row_major(bd.slice.affine, xf, yf, 0.0f, xo, yo, zo);
                xf = xo;
                yf = yo;
            }
            const int64_t xi = quantize_round_fast(xf);
            const int64_t yi = quantize_round_fast(yf);
            if (clamp) {
                if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(width) ||
                    yi >= static_cast<int64_t>(height)) {
                    continue;
                }
            }
            if (xi < 0 || yi < 0 || xi >= static_cast<int64_t>(width) ||
                yi >= static_cast<int64_t>(height)) {
                continue;
            }
            const uint64_t base_idx = (static_cast<uint64_t>(yi) * width + static_cast<uint64_t>(xi)) * channels;
            const uint64_t kernel_base = (static_cast<uint64_t>(kid) * kcols * channels);
            for (uint32_t c = 0; c < channels; ++c) {
                Scalar acc = static_cast<Scalar>(0);
                if (vd.shape.dims.size() == 1) {
                    const Scalar v = vmap.data[i];
                    acc = v * kmap.data[kernel_base + c];
                } else {
                    const Scalar* vrow = vmap.data + static_cast<uint64_t>(i) * kcols;
                    for (uint32_t k = 0; k < kcols; ++k) {
                        acc += vrow[k] * kmap.data[kernel_base + static_cast<uint64_t>(k) * channels + c];
                    }
                }
                omap.data[base_idx + c] += acc;
            }
        }

        mem->unmap(kernel_ids.handle());
        bmap.unmap();
        omap.unmap();
        pmap.unmap();
        vmap.unmap();
        kmap.unmap();
        return true;
    }

    static AbstractTensor apply_stencil_2d(const AbstractTensor& field,
                                           const AbstractTensor& kernel) {
        TensorDesc out_desc = field.desc();
        AbstractTensor out = AbstractTensor::create(out_desc, field.backend());
        if (!out.valid()) return {};
        if (!apply_stencil_2d_into(field, kernel, &out)) return {};
        return out;
    }

    static bool apply_stencil_2d_into(const AbstractTensor& field,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out) {
        if (!out) return false;
        if (!field.valid() || !kernel.valid() || !out->valid()) return false;
        if (field.backend() != kernel.backend() || field.backend() != out->backend()) return false;
        const TensorDesc& fd = field.desc();
        const TensorDesc& kd = kernel.desc();
        const TensorDesc& od = out->desc();
        if (fd.dtype != DType || kd.dtype != DType || od.dtype != DType) return false;
        if (fd.layout != TensorLayout::Dense || kd.layout != TensorLayout::Dense || od.layout != TensorLayout::Dense) return false;
        if (fd.shape.dims.size() != 2 || kd.shape.dims.size() != 2 || od.shape.dims.size() != 2) return false;
        if (od.shape.dims != fd.shape.dims) return false;

        const uint32_t height = fd.shape.dims[0];
        const uint32_t width = fd.shape.dims[1];
        const uint32_t ks = kd.shape.dims[0];
        if (ks == 0 || kd.shape.dims[1] != ks) return false;
        if (ks % 2 == 0) return false;
        const int32_t radius = static_cast<int32_t>(ks / 2);

        MappedDense fmap = map_dense(field);
        MappedDense kmap = map_dense(kernel);
        MappedDense omap = map_dense_mut(*out);
        if (!fmap.ok || !kmap.ok || !omap.ok) {
            fmap.unmap();
            kmap.unmap();
            omap.unmap();
            return false;
        }

        const Scalar* f = fmap.data;
        const Scalar* k = kmap.data;
        Scalar* o = omap.data;

        nodus::ThreadPool* pool = tensor_op_pool();

        struct StencilCtx {
            const Scalar* f = nullptr;
            const Scalar* k = nullptr;
            Scalar* o = nullptr;
            uint32_t width = 0;
            uint32_t height = 0;
            int32_t radius = 0;
            uint32_t ks = 0;
            uint32_t x0 = 0;
            uint32_t x1 = 0;
            uint32_t y0 = 0;
            uint32_t y1 = 0;
        };

        auto border_3 = [](const void* vctx, uint32_t y0r, uint32_t y1r) {
            const auto* ctx = static_cast<const StencilCtx*>(vctx);
            const uint32_t width = ctx->width;
            const uint32_t height = ctx->height;
            const Scalar* f = ctx->f;
            const Scalar* k = ctx->k;
            Scalar* o = ctx->o;
            auto clamp_i32_local = [](int32_t v, int32_t lo, int32_t hi) -> int32_t {
                if (v < lo) return lo;
                if (v > hi) return hi;
                return v;
            };
            for (uint32_t y = y0r; y < y1r; ++y) {
                const bool border_y = (y == 0 || y + 1 == height);
                const uint64_t row = static_cast<uint64_t>(y) * width;
                for (uint32_t x = 0; x < width; ++x) {
                    const bool border = border_y || (x == 0 || x + 1 == width);
                    if (!border) continue;
                    const int32_t y_i = static_cast<int32_t>(y);
                    const int32_t x_i = static_cast<int32_t>(x);

                    const int32_t y0 = clamp_i32_local(y_i - 1, 0, static_cast<int32_t>(height) - 1);
                    const int32_t y1 = y_i;
                    const int32_t y2 = clamp_i32_local(y_i + 1, 0, static_cast<int32_t>(height) - 1);
                    const int32_t x0 = clamp_i32_local(x_i - 1, 0, static_cast<int32_t>(width) - 1);
                    const int32_t x1 = x_i;
                    const int32_t x2 = clamp_i32_local(x_i + 1, 0, static_cast<int32_t>(width) - 1);

                    const uint64_t r0 = static_cast<uint64_t>(y0) * width;
                    const uint64_t r1 = static_cast<uint64_t>(y1) * width;
                    const uint64_t r2 = static_cast<uint64_t>(y2) * width;

                    Scalar acc = static_cast<Scalar>(0);
                    acc += f[r0 + static_cast<uint64_t>(x0)] * k[0];
                    acc += f[r0 + static_cast<uint64_t>(x1)] * k[1];
                    acc += f[r0 + static_cast<uint64_t>(x2)] * k[2];
                    acc += f[r1 + static_cast<uint64_t>(x0)] * k[3];
                    acc += f[r1 + static_cast<uint64_t>(x1)] * k[4];
                    acc += f[r1 + static_cast<uint64_t>(x2)] * k[5];
                    acc += f[r2 + static_cast<uint64_t>(x0)] * k[6];
                    acc += f[r2 + static_cast<uint64_t>(x1)] * k[7];
                    acc += f[r2 + static_cast<uint64_t>(x2)] * k[8];
                    o[row + x] = acc;
                }
            }
        };

        auto interior_3 = [](const void* vctx, uint32_t y0r, uint32_t y1r) {
            const auto* ctx = static_cast<const StencilCtx*>(vctx);
            const uint32_t width = ctx->width;
            const uint32_t height = ctx->height;
            const Scalar* f = ctx->f;
            const Scalar* k = ctx->k;
            Scalar* o = ctx->o;
            const uint32_t y_start = std::max<uint32_t>(1, y0r);
            const uint32_t y_end = std::min<uint32_t>(height - 1, y1r);
            if (y_start >= y_end) return;
            for (uint32_t y = y_start; y < y_end; ++y) {
                const uint64_t r0 = static_cast<uint64_t>(y - 1) * width;
                const uint64_t r1 = static_cast<uint64_t>(y) * width;
                const uint64_t r2 = static_cast<uint64_t>(y + 1) * width;
                for (uint32_t x = 1; x + 1 < width; ++x) {
                    const uint64_t i0 = static_cast<uint64_t>(x - 1);
                    const uint64_t i1 = static_cast<uint64_t>(x);
                    const uint64_t i2 = static_cast<uint64_t>(x + 1);
                    Scalar acc = static_cast<Scalar>(0);
                    acc += f[r0 + i0] * k[0];
                    acc += f[r0 + i1] * k[1];
                    acc += f[r0 + i2] * k[2];
                    acc += f[r1 + i0] * k[3];
                    acc += f[r1 + i1] * k[4];
                    acc += f[r1 + i2] * k[5];
                    acc += f[r2 + i0] * k[6];
                    acc += f[r2 + i1] * k[7];
                    acc += f[r2 + i2] * k[8];
                    o[r1 + x] = acc;
                }
            }
        };

        auto border_general = [](const void* vctx, uint32_t y0r, uint32_t y1r) {
            const auto* ctx = static_cast<const StencilCtx*>(vctx);
            const uint32_t width = ctx->width;
            const uint32_t height = ctx->height;
            const int32_t radius = ctx->radius;
            const uint32_t ks = ctx->ks;
            const Scalar* f = ctx->f;
            const Scalar* k = ctx->k;
            Scalar* o = ctx->o;
            const uint32_t x0 = ctx->x0;
            const uint32_t x1 = ctx->x1;
            const uint32_t y0 = ctx->y0;
            const uint32_t y1 = ctx->y1;
            auto clamp_i32_local = [](int32_t v, int32_t lo, int32_t hi) -> int32_t {
                if (v < lo) return lo;
                if (v > hi) return hi;
                return v;
            };
            const int32_t h_i = static_cast<int32_t>(height);
            const int32_t w_i = static_cast<int32_t>(width);
            for (uint32_t y = y0r; y < y1r; ++y) {
                const bool border_y = (y < y0) || (y >= y1);
                const uint64_t row = static_cast<uint64_t>(y) * width;
                for (uint32_t x = 0; x < width; ++x) {
                    const bool border = border_y || (x < x0) || (x >= x1);
                    if (!border) continue;
                    Scalar acc = static_cast<Scalar>(0);
                    const int32_t y_i = static_cast<int32_t>(y);
                    const int32_t x_i = static_cast<int32_t>(x);
                    for (int32_t ky = -radius; ky <= radius; ++ky) {
                        const int32_t sy = clamp_i32_local(y_i + ky, 0, h_i - 1);
                        const uint64_t frow = static_cast<uint64_t>(sy) * width;
                        const uint64_t krow = static_cast<uint64_t>(ky + radius) * ks;
                        for (int32_t kx = -radius; kx <= radius; ++kx) {
                            const int32_t sx = clamp_i32_local(x_i + kx, 0, w_i - 1);
                            acc += f[frow + static_cast<uint64_t>(sx)] * k[krow + static_cast<uint64_t>(kx + radius)];
                        }
                    }
                    o[row + x] = acc;
                }
            }
        };

        auto interior_general = [](const void* vctx, uint32_t y0r, uint32_t y1r) {
            const auto* ctx = static_cast<const StencilCtx*>(vctx);
            const uint32_t width = ctx->width;
            const uint32_t height = ctx->height;
            const int32_t radius = ctx->radius;
            const uint32_t ks = ctx->ks;
            const Scalar* f = ctx->f;
            const Scalar* k = ctx->k;
            Scalar* o = ctx->o;
            const uint32_t x0 = ctx->x0;
            const uint32_t x1 = ctx->x1;
            const uint32_t y0 = ctx->y0;
            const uint32_t y1 = ctx->y1;
            if (x0 >= x1) return;
            const uint32_t y_start = std::max<uint32_t>(y0, y0r);
            const uint32_t y_end = std::min<uint32_t>(y1, y1r);
            if (y_start >= y_end) return;
            for (uint32_t y = y_start; y < y_end; ++y) {
                const uint64_t row = static_cast<uint64_t>(y) * width;
                for (uint32_t x = x0; x < x1; ++x) {
                    Scalar acc = static_cast<Scalar>(0);
                    for (int32_t ky = -radius; ky <= radius; ++ky) {
                        const uint64_t frow = static_cast<uint64_t>(static_cast<int32_t>(y) + ky) * width;
                        const uint64_t krow = static_cast<uint64_t>(ky + radius) * ks;
                        for (int32_t kx = -radius; kx <= radius; ++kx) {
                            acc += f[frow + static_cast<uint64_t>(static_cast<int32_t>(x) + kx)] *
                                   k[krow + static_cast<uint64_t>(kx + radius)];
                        }
                    }
                    o[row + x] = acc;
                }
            }
        };

        StencilCtx ctx{};
        ctx.f = f;
        ctx.k = k;
        ctx.o = o;
        ctx.width = width;
        ctx.height = height;
        ctx.radius = radius;
        ctx.ks = ks;
        ctx.x0 = 0;
        ctx.x1 = 0;
        ctx.y0 = 0;
        ctx.y1 = 0;

        // Small kernels are worth special-casing for better vectorization.
        if (ks == 3 && width > 0 && height > 0) {
            // Border: y=0 and y=H-1 and x=0 and x=W-1. Cheap because it's O(W+H).
            submit_row_jobs(pool, border_3, &ctx, 0, height);

            // Interior (no clamps).
            if (width >= 3 && height >= 3) {
                submit_row_jobs(pool, interior_3, &ctx, 1, height - 1);
            }

            fmap.unmap();
            kmap.unmap();
            omap.unmap();
            return true;
        }

        const uint32_t x0 = static_cast<uint32_t>(radius);
        const uint32_t y0 = static_cast<uint32_t>(radius);
        const uint32_t x1 = (width > static_cast<uint32_t>(radius)) ? (width - static_cast<uint32_t>(radius)) : 0u;
        const uint32_t y1 = (height > static_cast<uint32_t>(radius)) ? (height - static_cast<uint32_t>(radius)) : 0u;
        ctx.x0 = x0;
        ctx.x1 = x1;
        ctx.y0 = y0;
        ctx.y1 = y1;

        // Border region (clamped).
        submit_row_jobs(pool, border_general, &ctx, 0, height);

        // Interior region (no clamps).
        if (x0 < x1 && y0 < y1) {
            submit_row_jobs(pool, interior_general, &ctx, y0, y1);
        }

        fmap.unmap();
        kmap.unmap();
        omap.unmap();
        return true;
    }

    static bool scatter_add_2d_affine(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& mat4,
                                      AbstractTensor* out,
                                      bool clamp) {
        if (!out) return false;
        if (!mat4.valid()) return false;
        if (base.backend() != mat4.backend()) return false;
        const TensorDesc& md = mat4.desc();
        if (md.dtype != DType || md.layout != TensorLayout::Dense) return false;
        const bool use_mat4 = shape_is(md, {4, 4});
        const bool use_mat3 = shape_is(md, {3, 3});
        if (!use_mat4 && !use_mat3) return false;

        MappedDense mmap = map_dense(mat4);
        if (!mmap.ok) {
            mmap.unmap();
            return false;
        }

        if (!points.valid()) {
            mmap.unmap();
            return false;
        }
        const TensorDesc& pd = points.desc();
        if (pd.dtype != DType || pd.layout != TensorLayout::Dense) {
            mmap.unmap();
            return false;
        }
        if (pd.shape.dims.size() != 2) {
            mmap.unmap();
            return false;
        }
        const uint32_t count = pd.shape.dims[0];
        const uint32_t pcols = pd.shape.dims[1];
        if (pcols != 2 && pcols != 3) {
            mmap.unmap();
            return false;
        }

        float affine[16] = {0};
        if (use_mat4) {
            for (uint32_t i = 0; i < 16; ++i) {
                affine[i] = static_cast<float>(mmap.data[i]);
            }
        } else {
            affine[0] = static_cast<float>(mmap.data[0]);
            affine[1] = static_cast<float>(mmap.data[1]);
            affine[4] = static_cast<float>(mmap.data[3]);
            affine[5] = static_cast<float>(mmap.data[4]);
            affine[12] = static_cast<float>(mmap.data[6]);
            affine[13] = static_cast<float>(mmap.data[7]);
            affine[10] = 1.0f;
            affine[15] = 1.0f;
        }
        mmap.unmap();

        TensorDesc bd = base.desc();
        bd.slice.has_affine = false;
        AbstractTensor base_view = AbstractTensor::wrap(base.handle(), bd, base.backend(), false);

        if (pcols == 2) {
            base_view.set_slice_affine_row_major(affine);
            return scatter_add_nd(base_view, points, values, out, clamp);
        }

        MappedDense pmap = map_dense(points);
        if (!pmap.ok) {
            pmap.unmap();
            return false;
        }

        TensorDesc tdesc = pd;
        tdesc.shape.dims[1] = 2;
        AbstractTensor tmp = AbstractTensor::create(tdesc, base.backend());
        if (!tmp.valid()) {
            pmap.unmap();
            return false;
        }
        MappedDense tmap = map_dense_mut(tmp);
        if (!tmap.ok) {
            pmap.unmap();
            tmap.unmap();
            return false;
        }

        for (uint32_t i = 0; i < count; ++i) {
            const Scalar x = pmap.data[i * pcols + 0];
            const Scalar y = pmap.data[i * pcols + 1];
            const Scalar z = use_mat4 ? pmap.data[i * pcols + 2] : static_cast<Scalar>(0);
            float tx = 0.0f;
            float ty = 0.0f;
            if (use_mat4) {
                tx = static_cast<float>(x) * affine[0] +
                     static_cast<float>(y) * affine[4] +
                     static_cast<float>(z) * affine[8] +
                     affine[12];
                ty = static_cast<float>(x) * affine[1] +
                     static_cast<float>(y) * affine[5] +
                     static_cast<float>(z) * affine[9] +
                     affine[13];
            } else {
                tx = static_cast<float>(x) * affine[0] +
                     static_cast<float>(y) * affine[4] +
                     affine[12];
                ty = static_cast<float>(x) * affine[1] +
                     static_cast<float>(y) * affine[5] +
                     affine[13];
            }
            tmap.data[i * 2 + 0] = static_cast<Scalar>(tx);
            tmap.data[i * 2 + 1] = static_cast<Scalar>(ty);
        }

        pmap.unmap();
        tmap.unmap();

        return scatter_add_nd(base_view, tmp, values, out, clamp);
    }
};

} // namespace

bool tensor_axpby_f32(const AbstractTensor& a,
                      float alpha,
                      const AbstractTensor& b,
                      float beta,
                      AbstractTensor* out) {
    if (!out || !out->valid() || !a.valid() || !b.valid()) return false;

    TensorOpPlan plan{};
    if (!build_axpby_plan(a, b, *out, plan)) return false;

    execute_axpby_plan(plan, alpha, beta);

    auto* mem = dynamic_cast<InMemoryBackend*>(a.backend());
    if (mem) {
        mem->unmap(out->handle());
        mem->unmap(b.handle());
        mem->unmap(a.handle());
    }
    return true;
}

bool tensor_copy_f32_into(const AbstractTensor& src, AbstractTensor* dst) {
    if (!dst || !dst->valid() || !src.valid()) return false;

    TensorOpPlan plan{};
    if (!build_unary_plan(src, *dst, plan)) return false;

    execute_copy_plan(plan);

    auto* mem = dynamic_cast<InMemoryBackend*>(src.backend());
    if (mem) {
        mem->unmap(dst->handle());
        mem->unmap(src.handle());
    }
    return true;
}

#define NODUS_TENSOR_MATH_DEFINE(SUFFIX, SCALAR, DTYPE)                              \
    AbstractTensor tensor_matmul_##SUFFIX(const AbstractTensor& a,                  \
                                          const AbstractTensor& b) {                \
        return TensorMathImpl<SCALAR, DTYPE>::matmul(a, b);                          \
    }                                                                               \
    AbstractTensor tensor_affine_identity_##SUFFIX(TensorBackend* backend) {        \
        return TensorMathImpl<SCALAR, DTYPE>::affine_identity(backend);             \
    }                                                                               \
    AbstractTensor tensor_affine_translation_##SUFFIX(const AbstractTensor& t) {    \
        return TensorMathImpl<SCALAR, DTYPE>::affine_translation(t);                \
    }                                                                               \
    AbstractTensor tensor_affine_scale_##SUFFIX(const AbstractTensor& s) {          \
        return TensorMathImpl<SCALAR, DTYPE>::affine_scale(s);                      \
    }                                                                               \
    AbstractTensor tensor_affine_from_quat_translation_##SUFFIX(                    \
        const AbstractTensor& q, const AbstractTensor& t) {                         \
        return TensorMathImpl<SCALAR, DTYPE>::affine_from_quat_translation(q, t);   \
    }                                                                               \
    AbstractTensor tensor_quat_identity_##SUFFIX(TensorBackend* backend) {          \
        return TensorMathImpl<SCALAR, DTYPE>::quat_identity(backend);               \
    }                                                                               \
    AbstractTensor tensor_quat_from_axis_angle_##SUFFIX(const AbstractTensor& axis, \
                                                        SCALAR angle) {             \
        return TensorMathImpl<SCALAR, DTYPE>::quat_from_axis_angle(axis, angle);    \
    }                                                                               \
    AbstractTensor tensor_quat_normalize_##SUFFIX(const AbstractTensor& q) {        \
        return TensorMathImpl<SCALAR, DTYPE>::quat_normalize(q);                    \
    }                                                                               \
    AbstractTensor tensor_quat_mul_##SUFFIX(const AbstractTensor& a,                \
                                            const AbstractTensor& b) {              \
        return TensorMathImpl<SCALAR, DTYPE>::quat_mul(a, b);                        \
    }                                                                               \
    AbstractTensor tensor_quat_to_mat4_##SUFFIX(const AbstractTensor& q,            \
                                                const AbstractTensor& t) {          \
        return TensorMathImpl<SCALAR, DTYPE>::quat_to_mat4(q, t);                    \
    }                                                                               \
    AbstractTensor tensor_transform_points_##SUFFIX(const AbstractTensor& points,  \
                                                     const AbstractTensor& mat4) {  \
        return TensorMathImpl<SCALAR, DTYPE>::transform_points(points, mat4);       \
    }

NODUS_TENSOR_MATH_DEFINE(f32, float, TensorDType::F32)
NODUS_TENSOR_MATH_DEFINE(f64, double, TensorDType::F64)

#undef NODUS_TENSOR_MATH_DEFINE

bool tensor_intersect_plane_z_f32(const AbstractTensor& origins,
                                  const AbstractTensor& dirs,
                                  float plane_z,
                                  AbstractTensor* out_hits,
                                  AbstractTensor* out_mask) {
    return TensorMathImpl<float, TensorDType::F32>::intersect_plane_z(
        origins, dirs, plane_z, out_hits, out_mask);
}

bool tensor_intersect_plane_z_f64(const AbstractTensor& origins,
                                  const AbstractTensor& dirs,
                                  double plane_z,
                                  AbstractTensor* out_hits,
                                  AbstractTensor* out_mask) {
    return TensorMathImpl<double, TensorDType::F64>::intersect_plane_z(
        origins, dirs, plane_z, out_hits, out_mask);
}

bool tensor_scatter_add_2d_f32(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp) {
    return TensorMathImpl<float, TensorDType::F32>::scatter_add_2d(
        base, points, values, out, clamp);
}

bool tensor_scatter_add_2d_f64(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp) {
    return TensorMathImpl<double, TensorDType::F64>::scatter_add_2d(
        base, points, values, out, clamp);
}

bool tensor_scatter_add_nd_f32(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp) {
    return TensorMathImpl<float, TensorDType::F32>::scatter_add_nd(
        base, points, values, out, clamp);
}

bool tensor_scatter_add_nd_f64(const AbstractTensor& base,
                               const AbstractTensor& points,
                               const AbstractTensor& values,
                               AbstractTensor* out,
                               bool clamp) {
    return TensorMathImpl<double, TensorDType::F64>::scatter_add_nd(
        base, points, values, out, clamp);
}

bool tensor_scatter_add_2d_affine_f32(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& mat4,
                                      AbstractTensor* out,
                                      bool clamp) {
    return TensorMathImpl<float, TensorDType::F32>::scatter_add_2d_affine(
        base, points, values, mat4, out, clamp);
}

bool tensor_scatter_add_2d_affine_f64(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& mat4,
                                      AbstractTensor* out,
                                      bool clamp) {
    return TensorMathImpl<double, TensorDType::F64>::scatter_add_2d_affine(
        base, points, values, mat4, out, clamp);
}

bool tensor_scatter_add_2d_kernel_f32(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out,
                                      bool clamp) {
    return TensorMathImpl<float, TensorDType::F32>::scatter_add_2d_kernel(
        base, points, values, kernel, out, clamp);
}

bool tensor_scatter_add_2d_kernel_f64(const AbstractTensor& base,
                                      const AbstractTensor& points,
                                      const AbstractTensor& values,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out,
                                      bool clamp) {
    return TensorMathImpl<double, TensorDType::F64>::scatter_add_2d_kernel(
        base, points, values, kernel, out, clamp);
}

bool tensor_scatter_add_2d_kernel_fn_f32(const AbstractTensor& base,
                                         const AbstractTensor& points,
                                         const AbstractTensor& values,
                                         const TensorKernelFnF32& kernel_fn,
                                         void* user,
                                         AbstractTensor* out,
                                         KernelDispatchPlan* out_plan,
                                         bool clamp) {
    return TensorMathImpl<float, TensorDType::F32>::scatter_add_2d_kernel_fn_impl(
        base, points, values, kernel_fn, user, out, out_plan, clamp);
}

bool tensor_scatter_add_2d_kernel_fn_f64(const AbstractTensor& base,
                                         const AbstractTensor& points,
                                         const AbstractTensor& values,
                                         const TensorKernelFnF64& kernel_fn,
                                         void* user,
                                         AbstractTensor* out,
                                         KernelDispatchPlan* out_plan,
                                         bool clamp) {
    return TensorMathImpl<double, TensorDType::F64>::scatter_add_2d_kernel_fn_impl(
        base, points, values, kernel_fn, user, out, out_plan, clamp);
}

bool tensor_scatter_add_2d_kernel_ptr_f32(const AbstractTensor& base,
                                          const AbstractTensor& points,
                                          const AbstractTensor& values,
                                          const AbstractTensor& kernel_ptrs,
                                          AbstractTensor* out,
                                          KernelDispatchPlan* out_plan,
                                          bool clamp) {
    return TensorMathImpl<float, TensorDType::F32>::scatter_add_2d_kernel_ptr_impl(
        base, points, values, kernel_ptrs, out, out_plan, clamp);
}

bool tensor_scatter_add_2d_kernel_ptr_f64(const AbstractTensor& base,
                                          const AbstractTensor& points,
                                          const AbstractTensor& values,
                                          const AbstractTensor& kernel_ptrs,
                                          AbstractTensor* out,
                                          KernelDispatchPlan* out_plan,
                                          bool clamp) {
    return TensorMathImpl<double, TensorDType::F64>::scatter_add_2d_kernel_ptr_impl(
        base, points, values, kernel_ptrs, out, out_plan, clamp);
}

bool tensor_scatter_add_2d_kernel_bank_f32(const AbstractTensor& base,
                                           const AbstractTensor& points,
                                           const AbstractTensor& values,
                                           const AbstractTensor& kernel_bank,
                                           const AbstractTensor& kernel_ids,
                                           AbstractTensor* out,
                                           bool clamp) {
    return TensorMathImpl<float, TensorDType::F32>::scatter_add_2d_kernel_bank(
        base, points, values, kernel_bank, kernel_ids, out, clamp);
}

bool tensor_scatter_add_2d_kernel_bank_f64(const AbstractTensor& base,
                                           const AbstractTensor& points,
                                           const AbstractTensor& values,
                                           const AbstractTensor& kernel_bank,
                                           const AbstractTensor& kernel_ids,
                                           AbstractTensor* out,
                                           bool clamp) {
    return TensorMathImpl<double, TensorDType::F64>::scatter_add_2d_kernel_bank(
        base, points, values, kernel_bank, kernel_ids, out, clamp);
}

AbstractTensor tensor_apply_stencil_2d_f32(const AbstractTensor& field,
                                           const AbstractTensor& kernel) {
    return TensorMathImpl<float, TensorDType::F32>::apply_stencil_2d(field, kernel);
}

bool tensor_apply_stencil_2d_f32_into(const AbstractTensor& field,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out) {
    return TensorMathImpl<float, TensorDType::F32>::apply_stencil_2d_into(field, kernel, out);
}

AbstractTensor tensor_apply_stencil_2d_f64(const AbstractTensor& field,
                                           const AbstractTensor& kernel) {
    return TensorMathImpl<double, TensorDType::F64>::apply_stencil_2d(field, kernel);
}

bool tensor_apply_stencil_2d_f64_into(const AbstractTensor& field,
                                      const AbstractTensor& kernel,
                                      AbstractTensor* out) {
    return TensorMathImpl<double, TensorDType::F64>::apply_stencil_2d_into(field, kernel, out);
}

bool tensor_reduce_sum_axis_f32(const AbstractTensor& src, uint32_t axis, AbstractTensor* out) {
    if (!out || !out->valid() || !src.valid()) return false;
    if (src.backend() != out->backend()) return false;
    const TensorDesc& sd = src.desc();
    const TensorDesc& od = out->desc();
    if (sd.dtype != TensorDType::F32 || od.dtype != TensorDType::F32) return false;
    const size_t rank = sd.shape.dims.size();
    if (rank == 0 || axis >= rank) return false;

    std::vector<uint32_t> out_shape = sd.shape.dims;
    out_shape.erase(out_shape.begin() + axis);
    if (out_shape != od.shape.dims) return false;

    auto* mem = dynamic_cast<InMemoryBackend*>(src.backend());
    if (!mem) return false;

    void* src_ptr_v = nullptr;
    size_t src_bytes = 0;
    if (!mem->map(src.handle(), &src_ptr_v, &src_bytes)) return false;
    void* dst_ptr_v = nullptr;
    size_t dst_bytes = 0;
    if (!mem->map(out->handle(), &dst_ptr_v, &dst_bytes)) {
        mem->unmap(src.handle());
        return false;
    }

    const float* src_ptr = static_cast<const float*>(src_ptr_v);
    float* dst_ptr = static_cast<float*>(dst_ptr_v);

    std::vector<uint64_t> src_strides;
    if (!compute_strides_for_desc_u64(sd, src_strides)) {
        mem->unmap(src.handle());
        mem->unmap(out->handle());
        return false;
    }

    std::vector<uint64_t> dst_strides;
    const bool dst_dense = (od.layout == TensorLayout::Dense);
    if (!dst_dense) {
        if (!compute_strides_for_desc_u64(od, dst_strides) ||
            dst_strides.size() != out_shape.size()) {
            mem->unmap(src.handle());
            mem->unmap(out->handle());
            return false;
        }
    }

    const uint32_t axis_dim = sd.shape.dims[axis];
    const uint64_t output_elems = od.shape.element_count();
    std::vector<uint32_t> src_coords(rank, 0);
    std::vector<uint32_t> out_coords(out_shape.size(), 0);

    for (uint64_t lin = 0; lin < output_elems; ++lin) {
        uint64_t rem = lin;
        for (size_t d = out_shape.size(); d-- > 0;) {
            const uint32_t dim = out_shape[d];
            if (dim == 0) {
                out_coords[d] = 0;
                continue;
            }
            out_coords[d] = static_cast<uint32_t>(rem % dim);
            rem /= dim;
        }
        size_t out_idx = 0;
        for (size_t d = 0; d < rank; ++d) {
            if (d == axis) continue;
            src_coords[d] = out_coords[out_idx++];
        }
        float acc = 0.0f;
        for (uint32_t a = 0; a < axis_dim; ++a) {
            src_coords[axis] = a;
            uint64_t src_offset = 0;
            for (size_t d = 0; d < rank; ++d) {
                src_offset += static_cast<uint64_t>(src_coords[d]) * src_strides[d];
            }
            acc += src_ptr[src_offset];
        }
        if (dst_dense) {
            dst_ptr[lin] = acc;
        } else {
            uint64_t dst_offset = 0;
            for (size_t d = 0; d < out_shape.size(); ++d) {
                dst_offset += static_cast<uint64_t>(out_coords[d]) * dst_strides[d];
            }
            dst_ptr[dst_offset] = acc;
        }
    }

    mem->unmap(src.handle());
    mem->unmap(out->handle());
    return true;
}

bool tensor_pack_rgba_from_film_f32(const AbstractTensor& film,
                                     AbstractTensor* rgba,
                                     uint32_t check_shift) {
    if (!film.valid() || !rgba || !rgba->valid()) return false;
    if (film.backend() != rgba->backend()) return false;
    const TensorDesc& fdesc = film.desc();
    const TensorDesc& rdesc = rgba->desc();
    if (fdesc.dtype != TensorDType::F32 || fdesc.layout != TensorLayout::Dense) return false;
    if (rdesc.dtype != TensorDType::U32 || rdesc.layout != TensorLayout::Dense) return false;
    if (fdesc.shape.dims.size() != 3) return false;
    if (rdesc.shape.dims.size() != 2) return false;

    const uint32_t height = fdesc.shape.dims[0];
    const uint32_t width = fdesc.shape.dims[1];
    const uint32_t channels = fdesc.shape.dims[2];
    if (channels < 3) return false;
    if (rdesc.shape.dims[0] != height || rdesc.shape.dims[1] != width) return false;

    auto* mem = dynamic_cast<InMemoryBackend*>(film.backend());
    if (!mem) return false;

    void* film_ptr = nullptr;
    size_t film_bytes = 0;
    if (!mem->map(film.handle(), &film_ptr, &film_bytes)) return false;
    void* rgba_ptr = nullptr;
    size_t rgba_bytes = 0;
    if (!mem->map(rgba->handle(), &rgba_ptr, &rgba_bytes)) {
        mem->unmap(film.handle());
        return false;
    }

    const float* src = static_cast<const float*>(film_ptr);
    uint32_t* dst = static_cast<uint32_t*>(rgba_ptr);
    const uint32_t check_mask = (1u << check_shift) - 1u;

    for (uint32_t y = 0; y < height; ++y) {
        const uint32_t ycheck = (y >> check_shift);
        const bool yodd = (ycheck & 1u) != 0u;
        for (uint32_t x = 0; x < width; ++x) {
            const size_t base = (static_cast<size_t>(y) * width + x) * channels;
            const bool odd = ((x >> check_shift) ^ ycheck) & 1u;
            const float bias = odd ? 0.18f : 0.28f;

            auto to_clamped = [&](uint32_t channel) -> float {
                float v = bias + src[base + static_cast<size_t>(channel)];
                v = std::clamp(v, 0.0f, 1.0f);
                return v;
            };

            const uint8_t r = static_cast<uint8_t>(std::lround(to_clamped(0) * 255.0f));
            const uint8_t g = static_cast<uint8_t>(std::lround(to_clamped(1) * 255.0f));
            const uint8_t b = static_cast<uint8_t>(std::lround(to_clamped(2) * 255.0f));
            dst[static_cast<size_t>(y) * width + x] =
                static_cast<uint32_t>(r) |
                (static_cast<uint32_t>(g) << 8) |
                (static_cast<uint32_t>(b) << 16) |
                (static_cast<uint32_t>(255) << 24);
        }
    }

    mem->unmap(rgba->handle());
    mem->unmap(film.handle());
    return true;
}

AbstractTensor tensor_quat_normalize(const AbstractTensor& q) {
    if (!q.valid()) return {};
    if (q.desc().dtype == TensorDType::F32) return tensor_quat_normalize_f32(q);
    if (q.desc().dtype == TensorDType::F64) return tensor_quat_normalize_f64(q);
    return {};
}

AbstractTensor tensor_quat_mul(const AbstractTensor& a, const AbstractTensor& b) {
    if (!a.valid() || !b.valid()) return {};
    if (a.desc().dtype == TensorDType::F32 && b.desc().dtype == TensorDType::F32) {
        return tensor_quat_mul_f32(a, b);
    }
    if (a.desc().dtype == TensorDType::F64 && b.desc().dtype == TensorDType::F64) {
        return tensor_quat_mul_f64(a, b);
    }
    return {};
}

AbstractTensor tensor_quat_from_axis_angle(const AbstractTensor& axis, double angle) {
    if (!axis.valid()) return {};
    if (axis.desc().dtype == TensorDType::F32) {
        return tensor_quat_from_axis_angle_f32(axis, static_cast<float>(angle));
    }
    if (axis.desc().dtype == TensorDType::F64) {
        return tensor_quat_from_axis_angle_f64(axis, angle);
    }
    return {};
}

} // namespace nodus::tensors
