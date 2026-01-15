#include "common/tensors/abstraction/coo_matrix.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "common/tensors/abstraction/in_memory_backend.h"

namespace nodus::tensors {

namespace {
bool is_integer_dtype(TensorDType dtype) {
    switch (dtype) {
        case TensorDType::I32:
        case TensorDType::I64:
        case TensorDType::U32:
        case TensorDType::U64:
            return true;
        default:
            return false;
    }
}

static bool compute_strides(const TensorShape& shape, std::vector<uint64_t>& out) {
    const auto& dims = shape.dims;
    if (dims.empty()) return false;
    out.resize(dims.size());
    uint64_t stride = 1;
    for (size_t i = dims.size(); i-- > 0;) {
        out[i] = stride;
        stride *= static_cast<uint64_t>(dims[i]);
    }
    return true;
}
} // namespace

COOMatrix COOMatrix::create(const TensorShape& shape,
                            uint32_t nnz,
                            TensorDType value_dtype,
                            TensorBackend* backend,
                            TensorDType index_dtype,
                            CooIndexLayout layout) {
    COOMatrix out;
    out.shape = shape;
    out.layout = layout;

    const uint32_t rank = shape.rank();
    if (rank == 0) return out;

    TensorDesc indices_desc{};
    indices_desc.dtype = index_dtype;
    indices_desc.layout = TensorLayout::Dense;
    if (layout == CooIndexLayout::RowMajor) {
        indices_desc.shape.dims = {nnz, rank};
    } else {
        indices_desc.shape.dims = {rank, nnz};
    }

    TensorDesc values_desc{};
    values_desc.dtype = value_dtype;
    values_desc.layout = TensorLayout::Dense;
    values_desc.shape.dims = {nnz};

    out.indices = AbstractTensor::create(indices_desc, backend);
    if (!out.indices.valid()) return COOMatrix{};

    out.values = AbstractTensor::create(values_desc, backend);
    if (!out.values.valid()) return COOMatrix{};

    return out;
}

bool COOMatrix::valid() const {
    return indices.valid() && values.valid();
}

uint32_t COOMatrix::rank() const {
    return shape.rank();
}

uint32_t COOMatrix::nnz() const {
    if (!values.valid()) return 0;
    return static_cast<uint32_t>(values.desc().shape.element_count());
}

bool COOMatrix::validate() const {
    if (!valid()) return false;
    const uint32_t rank = shape.rank();
    if (rank == 0) return false;

    const TensorDesc& indices_desc = indices.desc();
    const TensorDesc& values_desc = values.desc();

    if (!is_integer_dtype(indices_desc.dtype)) return false;
    if (values_desc.shape.dims.size() != 1) return false;

    const uint32_t nnz_count = static_cast<uint32_t>(values_desc.shape.element_count());
    if (layout == CooIndexLayout::RowMajor) {
        if (indices_desc.shape.dims.size() != 2) return false;
        if (indices_desc.shape.dims[0] != nnz_count) return false;
        if (indices_desc.shape.dims[1] != rank) return false;
    } else {
        if (indices_desc.shape.dims.size() != 2) return false;
        if (indices_desc.shape.dims[0] != rank) return false;
        if (indices_desc.shape.dims[1] != nnz_count) return false;
    }

    return true;
}

bool COOMatrix::sort_indices(bool ascending) {
    if (!valid()) return false;
    if (layout != CooIndexLayout::RowMajor) return false;
    if (!is_integer_dtype(indices.desc().dtype)) return false;
    if (shape.rank() == 0) return false;

    auto* mem = dynamic_cast<InMemoryBackend*>(indices.backend());
    if (!mem || mem != dynamic_cast<InMemoryBackend*>(values.backend())) return false;

    void* idx_ptr_v = nullptr;
    size_t idx_bytes = 0;
    if (!mem->map(indices.handle(), &idx_ptr_v, &idx_bytes)) return false;
    void* val_ptr_v = nullptr;
    size_t val_bytes = 0;
    if (!mem->map(values.handle(), &val_ptr_v, &val_bytes)) {
        mem->unmap(indices.handle());
        return false;
    }

    const TensorDesc& idx_desc = indices.desc();
    const uint32_t rank = shape.rank();
    const uint32_t nnz_count = static_cast<uint32_t>(values.desc().shape.element_count());
    if (idx_desc.shape.dims.size() != 2 || idx_desc.shape.dims[0] != nnz_count || idx_desc.shape.dims[1] != rank) {
        mem->unmap(indices.handle());
        mem->unmap(values.handle());
        return false;
    }

    std::vector<uint64_t> strides;
    if (!compute_strides(shape, strides)) {
        mem->unmap(indices.handle());
        mem->unmap(values.handle());
        return false;
    }

    const size_t value_stride = tensor_dtype_size_bytes(values.desc().dtype);
    if (value_stride == 0) {
        mem->unmap(indices.handle());
        mem->unmap(values.handle());
        return false;
    }

    std::vector<uint64_t> keys(nnz_count);
    std::vector<uint32_t> order(nnz_count);

    auto read_coord = [&](uint32_t i, uint32_t d) -> uint64_t {
        const size_t idx = static_cast<size_t>(i) * rank + d;
        switch (idx_desc.dtype) {
            case TensorDType::I32: return static_cast<uint64_t>(static_cast<int32_t*>(idx_ptr_v)[idx]);
            case TensorDType::I64: return static_cast<uint64_t>(static_cast<int64_t*>(idx_ptr_v)[idx]);
            case TensorDType::U32: return static_cast<uint64_t>(static_cast<uint32_t*>(idx_ptr_v)[idx]);
            case TensorDType::U64: return static_cast<uint64_t>(static_cast<uint64_t*>(idx_ptr_v)[idx]);
            default: return 0;
        }
    };

    for (uint32_t i = 0; i < nnz_count; ++i) {
        uint64_t key = 0;
        for (uint32_t d = 0; d < rank; ++d) {
            const uint64_t coord = read_coord(i, d);
            key += coord * strides[d];
        }
        keys[i] = key;
        order[i] = i;
    }

    auto cmp = [&](uint32_t a, uint32_t b) {
        if (keys[a] == keys[b]) return a < b;
        return ascending ? (keys[a] < keys[b]) : (keys[a] > keys[b]);
    };
    std::sort(order.begin(), order.end(), cmp);

    std::vector<uint8_t> idx_copy(idx_bytes);
    std::vector<uint8_t> val_copy(nnz_count * value_stride);
    std::memcpy(idx_copy.data(), idx_ptr_v, idx_bytes);
    std::memcpy(val_copy.data(), val_ptr_v, nnz_count * value_stride);

    auto write_coord = [&](uint32_t i, uint32_t d, uint64_t value) {
        const size_t idx = static_cast<size_t>(i) * rank + d;
        switch (idx_desc.dtype) {
            case TensorDType::I32: static_cast<int32_t*>(idx_ptr_v)[idx] = static_cast<int32_t>(value); break;
            case TensorDType::I64: static_cast<int64_t*>(idx_ptr_v)[idx] = static_cast<int64_t>(value); break;
            case TensorDType::U32: static_cast<uint32_t*>(idx_ptr_v)[idx] = static_cast<uint32_t>(value); break;
            case TensorDType::U64: static_cast<uint64_t*>(idx_ptr_v)[idx] = static_cast<uint64_t>(value); break;
            default: break;
        }
    };

    for (uint32_t i = 0; i < nnz_count; ++i) {
        const uint32_t src = order[i];
        for (uint32_t d = 0; d < rank; ++d) {
            const size_t idx = static_cast<size_t>(src) * rank + d;
            uint64_t value = 0;
            switch (idx_desc.dtype) {
                case TensorDType::I32: value = static_cast<uint64_t>(reinterpret_cast<int32_t*>(idx_copy.data())[idx]); break;
                case TensorDType::I64: value = static_cast<uint64_t>(reinterpret_cast<int64_t*>(idx_copy.data())[idx]); break;
                case TensorDType::U32: value = static_cast<uint64_t>(reinterpret_cast<uint32_t*>(idx_copy.data())[idx]); break;
                case TensorDType::U64: value = reinterpret_cast<uint64_t*>(idx_copy.data())[idx]; break;
                default: break;
            }
            write_coord(i, d, value);
        }
        std::memcpy(static_cast<uint8_t*>(val_ptr_v) + static_cast<size_t>(i) * value_stride,
                    val_copy.data() + static_cast<size_t>(src) * value_stride,
                    value_stride);
    }

    mem->unmap(indices.handle());
    mem->unmap(values.handle());
    return true;
}

} // namespace nodus::tensors
