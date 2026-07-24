#include "table_abi.h"

#include "common/tensors/abstraction/tensor_types.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

using nodus::tensors::TensorDType;
using nodus::tensors::TensorLayout;

static int32_t add_edge(
    GP_TableContext* table,
    int32_t slots,
    int32_t top_k
) {
    const int32_t edge_idx = gp_table_get_edge_count(table);
    assert(gp_table_add_edge(
        table,
        static_cast<uint64_t>(edge_idx * 2 + 1),
        static_cast<uint64_t>(edge_idx * 2 + 2)
    ));
    GP_TableEdgeTensorSpecTyped spec{};
    spec.dims[0] = 1;
    spec.dim_count = 1;
    spec.slots = slots;
    spec.top_k = top_k;
    spec.elem_size = sizeof(uint64_t);
    spec.type_id = 0x4F525431;
    spec.layout = static_cast<int32_t>(TensorLayout::Opaque);
    spec.dtype = static_cast<int32_t>(TensorDType::Bytes8);
    assert(gp_table_edge_set_tensor_spec(table, edge_idx, &spec));
    return edge_idx;
}

int main() {
    GP_TableContext* table = gp_table_create(nullptr);
    assert(table != nullptr);
    constexpr uint64_t writer = 71;
    constexpr uint64_t reader = 91;
    const uint64_t samples[] = {1, 2, 3};

    const int32_t lossless = add_edge(table, 2, 0);
    assert(gp_table_edge_subscribe(table, lossless, reader));
    int32_t dropped = -1;
    assert(gp_table_edge_publish(
        table, lossless, writer, &samples[0], sizeof(uint64_t), &dropped
    ));
    assert(dropped == 0);
    assert(gp_table_edge_publish(
        table, lossless, writer, &samples[1], sizeof(uint64_t), &dropped
    ));
    assert(dropped == 0);
    assert(!gp_table_edge_publish(
        table, lossless, writer, &samples[2], sizeof(uint64_t), &dropped
    ));
    assert(dropped == 0);

    size_t snapshot_size = 0;
    assert(gp_table_edge_transaction_snapshot_size(
        table, lossless, &snapshot_size
    ));
    std::vector<uint8_t> snapshot(snapshot_size);
    assert(gp_table_edge_transaction_snapshot_fill(
        table, lossless, snapshot.data(), snapshot.size()
    ));
    assert(gp_table_edge_subscribe(table, lossless, reader + 1));
    assert(!gp_table_edge_transaction_snapshot_restore(
        table, lossless, snapshot.data(), snapshot.size()
    ));
    assert(gp_table_edge_unsubscribe(table, lossless, reader + 1));
    uint64_t consumed = 0;
    int32_t written = 0;
    assert(gp_table_edge_consume(
        table, lossless, reader, &consumed, sizeof(consumed), &written
    ));
    assert(written == sizeof(consumed) && consumed == samples[0]);
    assert(gp_table_edge_transaction_snapshot_restore(
        table, lossless, snapshot.data(), snapshot.size()
    ));
    consumed = 0;
    assert(gp_table_edge_consume(
        table, lossless, reader, &consumed, sizeof(consumed), &written
    ));
    assert(consumed == samples[0]);
    assert(!gp_table_edge_publish(
        table, lossless, writer + 1, &samples[2], sizeof(uint64_t), &dropped
    ));
    assert(dropped == 0);
    assert(gp_table_edge_publish(
        table, lossless, writer, &samples[2], sizeof(uint64_t), &dropped
    ));
    assert(dropped == 0);

    const int32_t lossy = add_edge(table, 2, 1);
    assert(gp_table_edge_subscribe(table, lossy, reader));
    assert(gp_table_edge_publish(
        table, lossy, writer, &samples[0], sizeof(uint64_t), &dropped
    ));
    assert(dropped == 0);
    assert(gp_table_edge_publish(
        table, lossy, writer, &samples[1], sizeof(uint64_t), &dropped
    ));
    assert(dropped == 0);
    assert(gp_table_edge_publish(
        table, lossy, writer, &samples[2], sizeof(uint64_t), &dropped
    ));
    assert(dropped == 1);

    gp_table_destroy(table);
    std::cout << "edge_fifo_drop_contract_test: ok\n";
    return 0;
}
