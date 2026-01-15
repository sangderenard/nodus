#include "common/tensors/abstraction/coo_matrix.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <cstdint>
#include <iostream>

using namespace nodus::tensors;

static bool require_or_report(bool condition, const char* what) {
    if (condition) return true;
    std::cerr << "tensor_coo_matrix_test: failed: " << what << "\n";
    return false;
}

int main() {
    register_in_memory_backend(true);
    InMemoryBackend& backend = in_memory_backend_singleton();

    TensorShape shape{};
    shape.dims = {3, 4};

    COOMatrix coo = COOMatrix::create(shape, 3, TensorDType::F32, &backend);
    if (!require_or_report(coo.valid(), "COOMatrix::create")) return 1;
    if (!require_or_report(coo.rank() == 2, "coo.rank")) return 1;
    if (!require_or_report(coo.nnz() == 3, "coo.nnz")) return 1;
    if (!require_or_report(coo.validate(), "coo.validate")) return 1;

    const TensorDesc& idx_desc = coo.indices.desc();
    const TensorDesc& val_desc = coo.values.desc();
    if (!require_or_report(idx_desc.shape.dims.size() == 2, "idx_desc rank")) return 1;
    if (!require_or_report(idx_desc.shape.dims[0] == 3, "idx_desc rows")) return 1;
    if (!require_or_report(idx_desc.shape.dims[1] == 2, "idx_desc cols")) return 1;
    if (!require_or_report(idx_desc.dtype == TensorDType::I32, "idx_desc dtype")) return 1;
    if (!require_or_report(val_desc.shape.dims.size() == 1, "val_desc rank")) return 1;
    if (!require_or_report(val_desc.shape.dims[0] == 3, "val_desc len")) return 1;
    if (!require_or_report(val_desc.dtype == TensorDType::F32, "val_desc dtype")) return 1;

    void* idx_data = nullptr;
    size_t idx_bytes = 0;
    if (!require_or_report(backend.map(coo.indices.handle(), &idx_data, &idx_bytes), "backend.map(indices)")) return 1;
    const size_t idx_elem_bytes = tensor_dtype_size_bytes(idx_desc.dtype);
    const size_t expected_idx_bytes = idx_desc.shape.element_count() * idx_elem_bytes;
    if (!require_or_report(idx_bytes >= expected_idx_bytes, "indices byte size")) return 1;
    int32_t* idx = static_cast<int32_t*>(idx_data);
    idx[0] = 0; idx[1] = 1;
    idx[2] = 2; idx[3] = 3;
    idx[4] = 1; idx[5] = 0;
    backend.unmap(coo.indices.handle());

    void* val_data = nullptr;
    size_t val_bytes = 0;
    if (!require_or_report(backend.map(coo.values.handle(), &val_data, &val_bytes), "backend.map(values)")) return 1;
    const size_t val_elem_bytes = tensor_dtype_size_bytes(val_desc.dtype);
    const size_t expected_val_bytes = val_desc.shape.element_count() * val_elem_bytes;
    if (!require_or_report(val_bytes >= expected_val_bytes, "values byte size")) return 1;
    float* val = static_cast<float*>(val_data);
    val[0] = 4.0f;
    val[1] = 5.0f;
    val[2] = 6.0f;
    backend.unmap(coo.values.handle());

    std::cout << "tensor_coo_matrix_test: ok\n";
    return 0;
}
