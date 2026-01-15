#pragma once

#include <cstdint>

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/tensor_types.h"

namespace nodus::tensors {

enum class CooIndexLayout : uint8_t {
    RowMajor = 0, // [nnz, rank]
    ColumnMajor,  // [rank, nnz]
};

struct COOMatrix {
    TensorShape shape;
    AbstractTensor indices;
    AbstractTensor values;
    CooIndexLayout layout = CooIndexLayout::RowMajor;

    static COOMatrix create(const TensorShape& shape,
                            uint32_t nnz,
                            TensorDType value_dtype,
                            TensorBackend* backend = nullptr,
                            TensorDType index_dtype = TensorDType::I32,
                            CooIndexLayout layout = CooIndexLayout::RowMajor);

    bool valid() const;
    uint32_t rank() const;
    uint32_t nnz() const;
    bool validate() const;
    bool sort_indices(bool ascending = true);
};

} // namespace nodus::tensors
