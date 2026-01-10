#include "common/tensors/abstraction/tensor_compare.h"
#include "common/tensors/abstraction/in_memory_backend.h"

#include <cstdint>
#include <cstring>
#include <iostream>

using namespace nodus::tensors;

static bool require_or_report(bool condition, const char* what) {
    if (condition) return true;
    std::cerr << "tensor_compare_test: failed: " << what << "\n";
    return false;
}

static AbstractTensor make_tensor(InMemoryBackend& backend, const std::vector<float>& values) {
    TensorDesc desc{};
    desc.dtype = TensorDType::F32;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {static_cast<uint32_t>(values.size())};
    AbstractTensor t = AbstractTensor::create(desc, &backend);
    if (!require_or_report(t.valid(), "AbstractTensor::create")) return {};
    void* data = nullptr;
    size_t bytes = 0;
    if (!require_or_report(backend.map(t.handle(), &data, &bytes), "backend.map")) return {};
    if (!require_or_report(bytes >= values.size() * sizeof(float), "mapped bytes too small")) return {};
    std::memcpy(data, values.data(), values.size() * sizeof(float));
    backend.unmap(t.handle());
    return t;
}

int main() {
    register_in_memory_backend(true);
    InMemoryBackend& backend = in_memory_backend_singleton();

    AbstractTensor a = make_tensor(backend, {0.0f, 1.0f, 2.0f, 3.0f});
    AbstractTensor b = make_tensor(backend, {0.1f, 1.05f, 2.5f, 3.0f});
    if (!require_or_report(a.valid() && b.valid(), "make_tensor")) return 1;

    // Note: 0.1f is not exactly representable; use a slightly larger threshold
    // to avoid false negatives from float->double rounding.
    const double thr = 0.1001;

    AbstractTensor mask;
    if (!require_or_report(tensor_compare_within_threshold(a, b, thr, &mask), "tensor_compare_within_threshold")) return 1;

    void* mask_data = nullptr;
    size_t mask_bytes = 0;
    if (!require_or_report(backend.map(mask.handle(), &mask_data, &mask_bytes), "backend.map(mask)")) return 1;
    const auto* m = static_cast<const uint8_t*>(mask_data);
    if (!require_or_report(m[0] == 1, "mask[0]") ||
        !require_or_report(m[1] == 1, "mask[1]") ||
        !require_or_report(m[2] == 0, "mask[2]") ||
        !require_or_report(m[3] == 1, "mask[3]")) {
        return 1;
    }
    backend.unmap(mask.handle());

    COOMatrix sparse = tensor_compare_within_threshold_sparse(a, b, thr, &backend);
    if (!require_or_report(sparse.valid(), "tensor_compare_within_threshold_sparse")) return 1;
    if (!require_or_report(sparse.nnz() == 3, "sparse.nnz")) return 1;

    void* idx_data = nullptr;
    size_t idx_bytes = 0;
    if (!require_or_report(backend.map(sparse.indices.handle(), &idx_data, &idx_bytes), "backend.map(sparse.indices)")) return 1;
    const auto* idx = static_cast<const uint32_t*>(idx_data);
    if (!require_or_report(idx[0] == 0, "idx[0]") ||
        !require_or_report(idx[1] == 1, "idx[1]") ||
        !require_or_report(idx[2] == 3, "idx[2]")) {
        return 1;
    }
    backend.unmap(sparse.indices.handle());

    void* val_data = nullptr;
    size_t val_bytes = 0;
    if (!require_or_report(backend.map(sparse.values.handle(), &val_data, &val_bytes), "backend.map(sparse.values)")) return 1;
    const auto* vals = static_cast<const float*>(val_data);
    if (!require_or_report(vals[0] == 0.0f, "vals[0]") ||
        !require_or_report(vals[1] == 1.0f, "vals[1]") ||
        !require_or_report(vals[2] == 3.0f, "vals[2]")) {
        return 1;
    }
    backend.unmap(sparse.values.handle());

    std::cout << "tensor_compare_test: ok\n";
    return 0;
}
