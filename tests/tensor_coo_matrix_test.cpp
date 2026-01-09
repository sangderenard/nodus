#include "common/tensors/abstraction/coo_matrix.h"
#include "common/tensors/abstraction/in_memory_backend.h"

#include <cassert>
#include <cstdint>
#include <iostream>

using namespace nodus::tensors;

int main() {
    register_in_memory_backend(true);
    InMemoryBackend& backend = in_memory_backend_singleton();

    TensorShape shape{};
    shape.dims = {3, 4};

    COOMatrix coo = COOMatrix::create(shape, 3, TensorDType::F32, &backend);
    assert(coo.valid());
    assert(coo.rank() == 2);
    assert(coo.nnz() == 3);
    assert(coo.validate());

    const TensorDesc& idx_desc = coo.indices.desc();
    const TensorDesc& val_desc = coo.values.desc();
    assert(idx_desc.shape.dims.size() == 2);
    assert(idx_desc.shape.dims[0] == 3);
    assert(idx_desc.shape.dims[1] == 2);
    assert(idx_desc.dtype == TensorDType::I32);
    assert(val_desc.shape.dims.size() == 1);
    assert(val_desc.shape.dims[0] == 3);
    assert(val_desc.dtype == TensorDType::F32);

    void* idx_data = nullptr;
    size_t idx_bytes = 0;
    assert(backend.map(coo.indices.handle(), &idx_data, &idx_bytes));
    assert(idx_bytes == sizeof(int32_t) * 3 * 2);
    int32_t* idx = static_cast<int32_t*>(idx_data);
    idx[0] = 0; idx[1] = 1;
    idx[2] = 2; idx[3] = 3;
    idx[4] = 1; idx[5] = 0;
    backend.unmap(coo.indices.handle());

    void* val_data = nullptr;
    size_t val_bytes = 0;
    assert(backend.map(coo.values.handle(), &val_data, &val_bytes));
    assert(val_bytes == sizeof(float) * 3);
    float* val = static_cast<float*>(val_data);
    val[0] = 4.0f;
    val[1] = 5.0f;
    val[2] = 6.0f;
    backend.unmap(coo.values.handle());

    std::cout << "tensor_coo_matrix_test: ok\n";
    return 0;
}
