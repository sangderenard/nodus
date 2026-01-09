#include "table_abi.h"

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/table_edge_tensor_bridge.h"
#include "common/tensors/abstraction/tensor_compare.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace nodus::tensors;

static AbstractTensor make_storage(InMemoryBackend& backend, size_t count) {
    TensorDesc desc{};
    desc.dtype = TensorDType::F32;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {static_cast<uint32_t>(count)};
    AbstractTensor t = AbstractTensor::create(desc, &backend);
    assert(t.valid());
    void* data = nullptr;
    size_t bytes = 0;
    assert(backend.map(t.handle(), &data, &bytes));
    std::memset(data, 0, bytes);
    backend.unmap(t.handle());
    return t;
}

static void configure_edge(GP_TableContext* ctx, int32_t edge_idx, int32_t count) {
    GP_TableEdgeTensorSpecTyped spec{};
    spec.dim_count = 1;
    spec.dims[0] = count;
    spec.slots = 4;
    spec.top_k = 0;
    spec.layout = static_cast<int32_t>(TensorLayout::Dense);
    spec.dtype = static_cast<int32_t>(TensorDType::F32);
    spec.elem_size = 0;
    spec.type_id = -1;
    assert(gp_table_edge_set_tensor_spec(ctx, edge_idx, &spec));
}

static std::unordered_map<uint32_t, float> coo_to_map(const COOMatrix& coo) {
    std::vector<uint32_t> linear;
    std::vector<uint8_t> values;
    assert(coo_extract_linear_values(coo, linear, values));
    std::unordered_map<uint32_t, float> out;
    const size_t elem_size = sizeof(float);
    assert(values.size() == linear.size() * elem_size);
    for (size_t i = 0; i < linear.size(); ++i) {
        float v = 0.0f;
        std::memcpy(&v, values.data() + i * elem_size, elem_size);
        out[linear[i]] = v;
    }
    return out;
}

static COOMatrix* consume_sparse(GP_TableContext* ctx, int32_t edge_idx, uint64_t subscriber) {
    void* sparse = nullptr;
    int32_t ok = gp_table_edge_consume_sparse(ctx, edge_idx, subscriber, &sparse);
    assert(ok);
    assert(sparse != nullptr);
    return reinterpret_cast<COOMatrix*>(sparse);
}

int main() {
    register_in_memory_backend(true);
    InMemoryBackend& backend = in_memory_backend_singleton();

    GP_TableContext* table = gp_table_create(nullptr);
    assert(table != nullptr);

    const uint64_t writer = 1001;
    const uint64_t subscriber = 2001;
    const int32_t count = 4;

    // Non-accumulated sparse delta.
    assert(gp_table_add_edge(table, 1ull, 2ull));
    int32_t edge_idx = gp_table_get_edge_count(table) - 1;
    configure_edge(table, edge_idx, count);
    AbstractTensor storage = make_storage(backend, count);
    assert(set_table_edge_tensor_storage(table, edge_idx, storage.handle(), &backend));
    assert(gp_table_edge_set_delta_sparse_mode(table, edge_idx, 1, 0, 0.0f));
    assert(gp_table_edge_subscribe(table, edge_idx, subscriber));

    const float sample1[] = {0.0f, 1.0f, 2.0f, 3.0f};
    int32_t dropped = 0;
    assert(gp_table_edge_publish(table,
                                 edge_idx,
                                 writer,
                                 sample1,
                                 static_cast<int32_t>(sizeof(sample1)),
                                 &dropped));
    assert(dropped == 0);

    COOMatrix* sparse = consume_sparse(table, edge_idx, subscriber);
    assert(sparse->nnz() == 3);
    auto values = coo_to_map(*sparse);
    gp_table_edge_free_sparse(sparse);
    assert(values.size() == 3);
    assert(values[1] == 1.0f);
    assert(values[2] == 2.0f);
    assert(values[3] == 3.0f);

    // Accumulated sparse delta.
    assert(gp_table_add_edge(table, 3ull, 4ull));
    int32_t edge_idx_accum = gp_table_get_edge_count(table) - 1;
    configure_edge(table, edge_idx_accum, count);
    AbstractTensor storage_accum = make_storage(backend, count);
    assert(set_table_edge_tensor_storage(table, edge_idx_accum, storage_accum.handle(), &backend));
    assert(gp_table_edge_set_delta_sparse_mode(table, edge_idx_accum, 1, 1, 0.0f));
    assert(gp_table_edge_subscribe(table, edge_idx_accum, subscriber));

    const float sample2[] = {0.0f, 1.0f, 2.0f, 3.0f};
    const float sample3[] = {4.0f, 1.0f, 2.0f, 5.0f};
    assert(gp_table_edge_publish(table,
                                 edge_idx_accum,
                                 writer,
                                 sample2,
                                 static_cast<int32_t>(sizeof(sample2)),
                                 &dropped));
    assert(dropped == 0);
    assert(gp_table_edge_publish(table,
                                 edge_idx_accum,
                                 writer,
                                 sample3,
                                 static_cast<int32_t>(sizeof(sample3)),
                                 &dropped));
    assert(dropped == 0);

    COOMatrix* accum_sparse = consume_sparse(table, edge_idx_accum, subscriber);
    auto accum_values = coo_to_map(*accum_sparse);
    gp_table_edge_free_sparse(accum_sparse);
    assert(accum_values.size() == 4);
    assert(accum_values[0] == 4.0f);
    assert(accum_values[1] == 1.0f);
    assert(accum_values[2] == 2.0f);
    assert(accum_values[3] == 5.0f);

    // Drain the second token to avoid leaking it.
    COOMatrix* empty_sparse = consume_sparse(table, edge_idx_accum, subscriber);
    assert(empty_sparse->nnz() == 0);
    gp_table_edge_free_sparse(empty_sparse);

    gp_table_destroy(table);
    std::cout << "edge_fifo_sparse_delta_test: ok\n";
    return 0;
}
