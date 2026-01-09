#include "common/tensors/abstraction/coo_matrix.h"

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

} // namespace nodus::tensors
