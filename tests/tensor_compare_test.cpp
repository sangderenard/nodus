#include "common/tensors/abstraction/tensor_compare.h"
#include "common/tensors/abstraction/in_memory_backend.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

using namespace nodus::tensors;

static AbstractTensor make_tensor(InMemoryBackend& backend, const std::vector<float>& values) {
    TensorDesc desc{};
    desc.dtype = TensorDType::F32;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {static_cast<uint32_t>(values.size())};
    AbstractTensor t = AbstractTensor::create(desc, &backend);
    assert(t.valid());
    void* data = nullptr;
    size_t bytes = 0;
    assert(backend.map(t.handle(), &data, &bytes));
    assert(bytes >= values.size() * sizeof(float));
    std::memcpy(data, values.data(), values.size() * sizeof(float));
    backend.unmap(t.handle());
    return t;
}

int main() {
    register_in_memory_backend(true);
    InMemoryBackend& backend = in_memory_backend_singleton();

    AbstractTensor a = make_tensor(backend, {0.0f, 1.0f, 2.0f, 3.0f});
    AbstractTensor b = make_tensor(backend, {0.1f, 1.05f, 2.5f, 3.0f});

    AbstractTensor mask;
    assert(tensor_compare_within_threshold(a, b, 0.1, &mask));

    void* mask_data = nullptr;
    size_t mask_bytes = 0;
    assert(backend.map(mask.handle(), &mask_data, &mask_bytes));
    const auto* m = static_cast<const uint8_t*>(mask_data);
    assert(m[0] == 1);
    assert(m[1] == 1);
    assert(m[2] == 0);
    assert(m[3] == 1);
    backend.unmap(mask.handle());

    COOMatrix sparse = tensor_compare_within_threshold_sparse(a, b, 0.1, &backend);
    assert(sparse.valid());
    assert(sparse.nnz() == 3);

    void* idx_data = nullptr;
    size_t idx_bytes = 0;
    assert(backend.map(sparse.indices.handle(), &idx_data, &idx_bytes));
    const auto* idx = static_cast<const uint32_t*>(idx_data);
    assert(idx[0] == 0);
    assert(idx[1] == 1);
    assert(idx[2] == 3);
    backend.unmap(sparse.indices.handle());

    void* val_data = nullptr;
    size_t val_bytes = 0;
    assert(backend.map(sparse.values.handle(), &val_data, &val_bytes));
    const auto* vals = static_cast<const float*>(val_data);
    assert(vals[0] == 0.0f);
    assert(vals[1] == 1.0f);
    assert(vals[2] == 3.0f);
    backend.unmap(sparse.values.handle());

    std::cout << "tensor_compare_test: ok\n";
    return 0;
}
