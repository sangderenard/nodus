#include "common/tensors/abstraction/tensor_compare.h"

#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace nodus::tensors {

namespace {
template <typename T>
bool compare_into_mask(const T* a,
                       const T* b,
                       uint8_t* mask,
                       uint64_t count,
                       double threshold,
                       bool keep_within,
                       std::vector<uint32_t>* out_indices) {
    if (!a || !b) return false;
    const double thr = std::max(0.0, threshold);
    for (uint64_t i = 0; i < count; ++i) {
        double dv = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        bool ok = keep_within ? (dv <= thr) : (dv > thr);
        if (mask) mask[i] = ok ? 1u : 0u;
        if (ok && out_indices) {
            out_indices->push_back(static_cast<uint32_t>(i));
        }
    }
    return true;
}

bool same_shape(const TensorDesc& a, const TensorDesc& b) {
    if (a.shape.dims.size() != b.shape.dims.size()) return false;
    for (size_t i = 0; i < a.shape.dims.size(); ++i) {
        if (a.shape.dims[i] != b.shape.dims[i]) return false;
    }
    return true;
}

bool to_coords(uint32_t linear, const TensorShape& shape, std::vector<uint32_t>& coords) {
    const size_t rank = shape.dims.size();
    coords.assign(rank, 0u);
    if (rank == 0) return false;
    uint64_t idx = linear;
    for (size_t r = rank; r-- > 0;) {
        uint32_t dim = shape.dims[r];
        if (dim == 0) return false;
        coords[r] = static_cast<uint32_t>(idx % dim);
        idx /= dim;
    }
    return true;
}

template <typename T>
bool fill_sparse(const T* values,
                 const std::vector<uint32_t>& linear_indices,
                 const TensorShape& shape,
                 COOMatrix* out_sparse) {
    if (!out_sparse || !values) return false;
    const uint32_t rank = shape.rank();
    const uint32_t nnz = static_cast<uint32_t>(linear_indices.size());
    if (rank == 0) return false;

    auto* backend = out_sparse->values.backend();
    auto* mem_backend = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem_backend) return false;

    void* idx_data = nullptr;
    size_t idx_bytes = 0;
    if (!mem_backend->map(out_sparse->indices.handle(), &idx_data, &idx_bytes)) return false;
    void* val_data = nullptr;
    size_t val_bytes = 0;
    if (!mem_backend->map(out_sparse->values.handle(), &val_data, &val_bytes)) {
        mem_backend->unmap(out_sparse->indices.handle());
        return false;
    }

    auto* idx_out = static_cast<uint32_t*>(idx_data);
    auto* val_out = static_cast<T*>(val_data);
    std::vector<uint32_t> coords;
    coords.reserve(rank);
    for (uint32_t i = 0; i < nnz; ++i) {
        const uint32_t linear = linear_indices[i];
        if (!to_coords(linear, shape, coords)) {
            mem_backend->unmap(out_sparse->indices.handle());
            mem_backend->unmap(out_sparse->values.handle());
            return false;
        }
        for (uint32_t r = 0; r < rank; ++r) {
            idx_out[i * rank + r] = coords[r];
        }
        val_out[i] = values[linear];
    }

    mem_backend->unmap(out_sparse->indices.handle());
    mem_backend->unmap(out_sparse->values.handle());
    return true;
}

bool map_dense_pair(const AbstractTensor& a,
                    const AbstractTensor& b,
                    void** out_a,
                    void** out_b,
                    size_t* out_bytes,
                    InMemoryBackend** out_backend) {
    if (!out_a || !out_b || !out_bytes || !out_backend) return false;
    auto* mem_backend = dynamic_cast<InMemoryBackend*>(a.backend());
    if (!mem_backend || mem_backend != dynamic_cast<InMemoryBackend*>(b.backend())) return false;
    void* a_ptr = nullptr;
    size_t a_bytes = 0;
    if (!mem_backend->map(a.handle(), &a_ptr, &a_bytes)) return false;
    void* b_ptr = nullptr;
    size_t b_bytes = 0;
    if (!mem_backend->map(b.handle(), &b_ptr, &b_bytes)) {
        mem_backend->unmap(a.handle());
        return false;
    }
    *out_a = a_ptr;
    *out_b = b_ptr;
    *out_bytes = std::min(a_bytes, b_bytes);
    *out_backend = mem_backend;
    return true;
}
} // namespace

AbstractTensor tensor_from_bytes(const TensorDesc& desc,
                                 TensorBackend* backend,
                                 const void* bytes,
                                 size_t bytes_len) {
    if (!backend || !bytes || bytes_len == 0) return {};
    AbstractTensor out = AbstractTensor::create(desc, backend);
    if (!out.valid()) return {};
    if (!tensor_copy_from_bytes(&out, bytes, bytes_len)) return {};
    return out;
}

bool tensor_copy_from_bytes(AbstractTensor* dst, const void* bytes, size_t bytes_len) {
    if (!dst || !dst->valid() || !bytes || bytes_len == 0) return false;
    auto* mem_backend = dynamic_cast<InMemoryBackend*>(dst->backend());
    if (!mem_backend) return false;
    void* dst_ptr = nullptr;
    size_t dst_bytes = 0;
    if (!mem_backend->map(dst->handle(), &dst_ptr, &dst_bytes)) return false;
    size_t copy_bytes = std::min(dst_bytes, bytes_len);
    std::memcpy(dst_ptr, bytes, copy_bytes);
    mem_backend->unmap(dst->handle());
    return true;
}

bool coo_extract_linear_values(const COOMatrix& coo,
                               std::vector<uint32_t>& out_linear,
                               std::vector<uint8_t>& out_values_bytes) {
    out_linear.clear();
    out_values_bytes.clear();
    if (!coo.valid()) return false;
    const TensorDesc& idx_desc = coo.indices.desc();
    const TensorDesc& val_desc = coo.values.desc();
    if (idx_desc.dtype != TensorDType::U32) return false;
    auto* mem_backend = dynamic_cast<InMemoryBackend*>(coo.values.backend());
    if (!mem_backend || mem_backend != dynamic_cast<InMemoryBackend*>(coo.indices.backend())) return false;

    void* idx_ptr = nullptr;
    size_t idx_bytes = 0;
    if (!mem_backend->map(coo.indices.handle(), &idx_ptr, &idx_bytes)) return false;
    void* val_ptr = nullptr;
    size_t val_bytes = 0;
    if (!mem_backend->map(coo.values.handle(), &val_ptr, &val_bytes)) {
        mem_backend->unmap(coo.indices.handle());
        return false;
    }

    const uint32_t rank = static_cast<uint32_t>(idx_desc.shape.dims.size() == 2 ? idx_desc.shape.dims[1] : idx_desc.shape.dims.size());
    const uint64_t nnz = val_desc.shape.element_count();
    const auto* idx = static_cast<const uint32_t*>(idx_ptr);
    const uint8_t* vals = static_cast<const uint8_t*>(val_ptr);
    const size_t elem_size = tensor_dtype_size_bytes(val_desc.dtype);
    out_linear.reserve(static_cast<size_t>(nnz));
    out_values_bytes.reserve(static_cast<size_t>(nnz) * elem_size);
    const TensorShape& shape = coo.shape;

    for (uint64_t i = 0; i < nnz; ++i) {
        uint64_t lin = 0;
        uint64_t stride = 1;
        for (uint32_t r = 0; r < rank; ++r) {
            const uint32_t dim = shape.dims[rank - 1 - r];
            const uint32_t coord = idx[i * rank + (rank - 1 - r)];
            lin += static_cast<uint64_t>(coord) * stride;
            stride *= static_cast<uint64_t>(dim ? dim : 1u);
        }
        out_linear.push_back(static_cast<uint32_t>(lin));
        const size_t off = static_cast<size_t>(i) * elem_size;
        out_values_bytes.insert(out_values_bytes.end(), vals + off, vals + off + elem_size);
    }

    mem_backend->unmap(coo.indices.handle());
    mem_backend->unmap(coo.values.handle());
    return true;
}

bool tensor_compare_within_threshold(const AbstractTensor& a,
                                     const AbstractTensor& b,
                                     double threshold,
                                     AbstractTensor* out_mask) {
    if (!out_mask) return false;
    if (!a.valid() || !b.valid()) return false;
    const TensorDesc& ad = a.desc();
    const TensorDesc& bd = b.desc();
    if (!same_shape(ad, bd)) return false;
    if (ad.layout != TensorLayout::Dense || bd.layout != TensorLayout::Dense) return false;

    TensorDesc mask_desc{};
    mask_desc.dtype = TensorDType::Bool;
    mask_desc.layout = TensorLayout::Dense;
    mask_desc.shape = ad.shape;
    *out_mask = AbstractTensor::create(mask_desc, a.backend());
    if (!out_mask->valid()) return false;

    void* a_ptr = nullptr;
    void* b_ptr = nullptr;
    size_t bytes = 0;
    InMemoryBackend* backend = nullptr;
    if (!map_dense_pair(a, b, &a_ptr, &b_ptr, &bytes, &backend)) return false;

    void* mask_ptr = nullptr;
    size_t mask_bytes = 0;
    if (!backend->map(out_mask->handle(), &mask_ptr, &mask_bytes)) {
        backend->unmap(a.handle());
        backend->unmap(b.handle());
        return false;
    }

    const uint64_t count = ad.shape.element_count();
    bool ok = false;
    switch (ad.dtype) {
        case TensorDType::F32:
            ok = compare_into_mask(reinterpret_cast<const float*>(a_ptr),
                                   reinterpret_cast<const float*>(b_ptr),
                                   reinterpret_cast<uint8_t*>(mask_ptr),
                                   count,
                                   threshold,
                                   true,
                                   nullptr);
            break;
        case TensorDType::F64:
            ok = compare_into_mask(reinterpret_cast<const double*>(a_ptr),
                                   reinterpret_cast<const double*>(b_ptr),
                                   reinterpret_cast<uint8_t*>(mask_ptr),
                                   count,
                                   threshold,
                                   true,
                                   nullptr);
            break;
        case TensorDType::I32:
            ok = compare_into_mask(reinterpret_cast<const int32_t*>(a_ptr),
                                   reinterpret_cast<const int32_t*>(b_ptr),
                                   reinterpret_cast<uint8_t*>(mask_ptr),
                                   count,
                                   threshold,
                                   true,
                                   nullptr);
            break;
        case TensorDType::I64:
            ok = compare_into_mask(reinterpret_cast<const int64_t*>(a_ptr),
                                   reinterpret_cast<const int64_t*>(b_ptr),
                                   reinterpret_cast<uint8_t*>(mask_ptr),
                                   count,
                                   threshold,
                                   true,
                                   nullptr);
            break;
        case TensorDType::U32:
            ok = compare_into_mask(reinterpret_cast<const uint32_t*>(a_ptr),
                                   reinterpret_cast<const uint32_t*>(b_ptr),
                                   reinterpret_cast<uint8_t*>(mask_ptr),
                                   count,
                                   threshold,
                                   true,
                                   nullptr);
            break;
        case TensorDType::U64:
            ok = compare_into_mask(reinterpret_cast<const uint64_t*>(a_ptr),
                                   reinterpret_cast<const uint64_t*>(b_ptr),
                                   reinterpret_cast<uint8_t*>(mask_ptr),
                                   count,
                                   threshold,
                                   true,
                                   nullptr);
            break;
        default:
            ok = false;
            break;
    }

    backend->unmap(a.handle());
    backend->unmap(b.handle());
    backend->unmap(out_mask->handle());
    return ok;
}

bool tensor_compare_outside_threshold(const AbstractTensor& a,
                                      const AbstractTensor& b,
                                      double threshold,
                                      AbstractTensor* out_mask) {
    if (!out_mask) return false;
    if (!a.valid() || !b.valid()) return false;
    const TensorDesc& ad = a.desc();
    const TensorDesc& bd = b.desc();
    if (!same_shape(ad, bd)) return false;
    if (ad.layout != TensorLayout::Dense || bd.layout != TensorLayout::Dense) return false;

    TensorDesc mask_desc{};
    mask_desc.dtype = TensorDType::Bool;
    mask_desc.layout = TensorLayout::Dense;
    mask_desc.shape = ad.shape;
    *out_mask = AbstractTensor::create(mask_desc, a.backend());
    if (!out_mask->valid()) return false;

    void* a_ptr = nullptr;
    void* b_ptr = nullptr;
    size_t bytes = 0;
    InMemoryBackend* backend = nullptr;
    if (!map_dense_pair(a, b, &a_ptr, &b_ptr, &bytes, &backend)) return false;

    void* mask_ptr = nullptr;
    size_t mask_bytes = 0;
    if (!backend->map(out_mask->handle(), &mask_ptr, &mask_bytes)) {
        backend->unmap(a.handle());
        backend->unmap(b.handle());
        return false;
    }

    const uint64_t count = ad.shape.element_count();
    bool ok = false;
    switch (ad.dtype) {
        case TensorDType::F32:
            ok = compare_into_mask(reinterpret_cast<const float*>(a_ptr),
                                   reinterpret_cast<const float*>(b_ptr),
                                   reinterpret_cast<uint8_t*>(mask_ptr),
                                   count,
                                   threshold,
                                   false,
                                   nullptr);
            break;
        case TensorDType::F64:
            ok = compare_into_mask(reinterpret_cast<const double*>(a_ptr),
                                   reinterpret_cast<const double*>(b_ptr),
                                   reinterpret_cast<uint8_t*>(mask_ptr),
                                   count,
                                   threshold,
                                   false,
                                   nullptr);
            break;
        case TensorDType::I32:
            ok = compare_into_mask(reinterpret_cast<const int32_t*>(a_ptr),
                                   reinterpret_cast<const int32_t*>(b_ptr),
                                   reinterpret_cast<uint8_t*>(mask_ptr),
                                   count,
                                   threshold,
                                   false,
                                   nullptr);
            break;
        case TensorDType::I64:
            ok = compare_into_mask(reinterpret_cast<const int64_t*>(a_ptr),
                                   reinterpret_cast<const int64_t*>(b_ptr),
                                   reinterpret_cast<uint8_t*>(mask_ptr),
                                   count,
                                   threshold,
                                   false,
                                   nullptr);
            break;
        case TensorDType::U32:
            ok = compare_into_mask(reinterpret_cast<const uint32_t*>(a_ptr),
                                   reinterpret_cast<const uint32_t*>(b_ptr),
                                   reinterpret_cast<uint8_t*>(mask_ptr),
                                   count,
                                   threshold,
                                   false,
                                   nullptr);
            break;
        case TensorDType::U64:
            ok = compare_into_mask(reinterpret_cast<const uint64_t*>(a_ptr),
                                   reinterpret_cast<const uint64_t*>(b_ptr),
                                   reinterpret_cast<uint8_t*>(mask_ptr),
                                   count,
                                   threshold,
                                   false,
                                   nullptr);
            break;
        default:
            ok = false;
            break;
    }

    backend->unmap(a.handle());
    backend->unmap(b.handle());
    backend->unmap(out_mask->handle());
    return ok;
}

COOMatrix tensor_compare_within_threshold_sparse(const AbstractTensor& a,
                                                 const AbstractTensor& b,
                                                 double threshold,
                                                 TensorBackend* backend) {
    COOMatrix out{};
    if (!a.valid() || !b.valid()) return out;
    const TensorDesc& ad = a.desc();
    const TensorDesc& bd = b.desc();
    if (!same_shape(ad, bd)) return out;
    if (ad.layout != TensorLayout::Dense || bd.layout != TensorLayout::Dense) return out;

    void* a_ptr = nullptr;
    void* b_ptr = nullptr;
    size_t bytes = 0;
    InMemoryBackend* mem_backend = nullptr;
    if (!map_dense_pair(a, b, &a_ptr, &b_ptr, &bytes, &mem_backend)) return out;

    const uint64_t count = ad.shape.element_count();
    std::vector<uint32_t> linear;
    bool ok = false;
    switch (ad.dtype) {
        case TensorDType::F32:
            ok = compare_into_mask(reinterpret_cast<const float*>(a_ptr),
                                   reinterpret_cast<const float*>(b_ptr),
                                   nullptr,
                                   count,
                                   threshold,
                                   true,
                                   &linear);
            break;
        case TensorDType::F64:
            ok = compare_into_mask(reinterpret_cast<const double*>(a_ptr),
                                   reinterpret_cast<const double*>(b_ptr),
                                   nullptr,
                                   count,
                                   threshold,
                                   true,
                                   &linear);
            break;
        case TensorDType::I32:
            ok = compare_into_mask(reinterpret_cast<const int32_t*>(a_ptr),
                                   reinterpret_cast<const int32_t*>(b_ptr),
                                   nullptr,
                                   count,
                                   threshold,
                                   true,
                                   &linear);
            break;
        case TensorDType::I64:
            ok = compare_into_mask(reinterpret_cast<const int64_t*>(a_ptr),
                                   reinterpret_cast<const int64_t*>(b_ptr),
                                   nullptr,
                                   count,
                                   threshold,
                                   true,
                                   &linear);
            break;
        case TensorDType::U32:
            ok = compare_into_mask(reinterpret_cast<const uint32_t*>(a_ptr),
                                   reinterpret_cast<const uint32_t*>(b_ptr),
                                   nullptr,
                                   count,
                                   threshold,
                                   true,
                                   &linear);
            break;
        case TensorDType::U64:
            ok = compare_into_mask(reinterpret_cast<const uint64_t*>(a_ptr),
                                   reinterpret_cast<const uint64_t*>(b_ptr),
                                   nullptr,
                                   count,
                                   threshold,
                                   true,
                                   &linear);
            break;
        default:
            ok = false;
            break;
    }

    mem_backend->unmap(a.handle());
    mem_backend->unmap(b.handle());
    if (!ok) return COOMatrix{};

    TensorBackend* use_backend = backend ? backend : a.backend();
    out = COOMatrix::create(ad.shape,
                            static_cast<uint32_t>(linear.size()),
                            ad.dtype,
                            use_backend,
                            TensorDType::U32,
                            CooIndexLayout::RowMajor);
    if (!out.valid()) return COOMatrix{};

    bool filled = false;
    void* a_ptr2 = nullptr;
    size_t a_bytes2 = 0;
    auto* use_mem_backend = dynamic_cast<InMemoryBackend*>(use_backend);
    if (!use_mem_backend) return COOMatrix{};
    if (use_mem_backend != dynamic_cast<InMemoryBackend*>(a.backend())) return COOMatrix{};
    if (!use_mem_backend->map(a.handle(), &a_ptr2, &a_bytes2)) return COOMatrix{};
    switch (ad.dtype) {
        case TensorDType::F32:
            filled = fill_sparse(reinterpret_cast<const float*>(a_ptr2), linear, ad.shape, &out);
            break;
        case TensorDType::F64:
            filled = fill_sparse(reinterpret_cast<const double*>(a_ptr2), linear, ad.shape, &out);
            break;
        case TensorDType::I32:
            filled = fill_sparse(reinterpret_cast<const int32_t*>(a_ptr2), linear, ad.shape, &out);
            break;
        case TensorDType::I64:
            filled = fill_sparse(reinterpret_cast<const int64_t*>(a_ptr2), linear, ad.shape, &out);
            break;
        case TensorDType::U32:
            filled = fill_sparse(reinterpret_cast<const uint32_t*>(a_ptr2), linear, ad.shape, &out);
            break;
        case TensorDType::U64:
            filled = fill_sparse(reinterpret_cast<const uint64_t*>(a_ptr2), linear, ad.shape, &out);
            break;
        default:
            filled = false;
            break;
    }
    use_mem_backend->unmap(a.handle());
    if (!filled) return COOMatrix{};
    return out;
}

COOMatrix tensor_compare_outside_threshold_sparse(const AbstractTensor& a,
                                                  const AbstractTensor& b,
                                                  double threshold,
                                                  TensorBackend* backend) {
    COOMatrix out{};
    if (!a.valid() || !b.valid()) return out;
    const TensorDesc& ad = a.desc();
    const TensorDesc& bd = b.desc();
    if (!same_shape(ad, bd)) return out;
    if (ad.layout != TensorLayout::Dense || bd.layout != TensorLayout::Dense) return out;

    void* a_ptr = nullptr;
    void* b_ptr = nullptr;
    size_t bytes = 0;
    InMemoryBackend* mem_backend = nullptr;
    if (!map_dense_pair(a, b, &a_ptr, &b_ptr, &bytes, &mem_backend)) return out;

    const uint64_t count = ad.shape.element_count();
    std::vector<uint32_t> linear;
    bool ok = false;
    switch (ad.dtype) {
        case TensorDType::F32:
            ok = compare_into_mask(reinterpret_cast<const float*>(a_ptr),
                                   reinterpret_cast<const float*>(b_ptr),
                                   nullptr,
                                   count,
                                   threshold,
                                   false,
                                   &linear);
            break;
        case TensorDType::F64:
            ok = compare_into_mask(reinterpret_cast<const double*>(a_ptr),
                                   reinterpret_cast<const double*>(b_ptr),
                                   nullptr,
                                   count,
                                   threshold,
                                   false,
                                   &linear);
            break;
        case TensorDType::I32:
            ok = compare_into_mask(reinterpret_cast<const int32_t*>(a_ptr),
                                   reinterpret_cast<const int32_t*>(b_ptr),
                                   nullptr,
                                   count,
                                   threshold,
                                   false,
                                   &linear);
            break;
        case TensorDType::I64:
            ok = compare_into_mask(reinterpret_cast<const int64_t*>(a_ptr),
                                   reinterpret_cast<const int64_t*>(b_ptr),
                                   nullptr,
                                   count,
                                   threshold,
                                   false,
                                   &linear);
            break;
        case TensorDType::U32:
            ok = compare_into_mask(reinterpret_cast<const uint32_t*>(a_ptr),
                                   reinterpret_cast<const uint32_t*>(b_ptr),
                                   nullptr,
                                   count,
                                   threshold,
                                   false,
                                   &linear);
            break;
        case TensorDType::U64:
            ok = compare_into_mask(reinterpret_cast<const uint64_t*>(a_ptr),
                                   reinterpret_cast<const uint64_t*>(b_ptr),
                                   nullptr,
                                   count,
                                   threshold,
                                   false,
                                   &linear);
            break;
        default:
            ok = false;
            break;
    }

    mem_backend->unmap(a.handle());
    mem_backend->unmap(b.handle());
    if (!ok) return COOMatrix{};

    TensorBackend* use_backend = backend ? backend : a.backend();
    out = COOMatrix::create(ad.shape,
                            static_cast<uint32_t>(linear.size()),
                            ad.dtype,
                            use_backend,
                            TensorDType::U32,
                            CooIndexLayout::RowMajor);
    if (!out.valid()) return COOMatrix{};

    bool filled = false;
    void* a_ptr2 = nullptr;
    size_t a_bytes2 = 0;
    auto* use_mem_backend = dynamic_cast<InMemoryBackend*>(use_backend);
    if (!use_mem_backend) return COOMatrix{};
    if (use_mem_backend != dynamic_cast<InMemoryBackend*>(a.backend())) return COOMatrix{};
    if (!use_mem_backend->map(a.handle(), &a_ptr2, &a_bytes2)) return COOMatrix{};
    switch (ad.dtype) {
        case TensorDType::F32:
            filled = fill_sparse(reinterpret_cast<const float*>(a_ptr2), linear, ad.shape, &out);
            break;
        case TensorDType::F64:
            filled = fill_sparse(reinterpret_cast<const double*>(a_ptr2), linear, ad.shape, &out);
            break;
        case TensorDType::I32:
            filled = fill_sparse(reinterpret_cast<const int32_t*>(a_ptr2), linear, ad.shape, &out);
            break;
        case TensorDType::I64:
            filled = fill_sparse(reinterpret_cast<const int64_t*>(a_ptr2), linear, ad.shape, &out);
            break;
        case TensorDType::U32:
            filled = fill_sparse(reinterpret_cast<const uint32_t*>(a_ptr2), linear, ad.shape, &out);
            break;
        case TensorDType::U64:
            filled = fill_sparse(reinterpret_cast<const uint64_t*>(a_ptr2), linear, ad.shape, &out);
            break;
        default:
            filled = false;
            break;
    }
    use_mem_backend->unmap(a.handle());
    if (!filled) return COOMatrix{};
    return out;
}

} // namespace nodus::tensors
