#include "nodus_runtime_abi.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

int main() {
    NodusEdgeRuntime* runtime = nodus_edge_runtime_create();
    assert(runtime);
    const int32_t dims[] = {4};
    assert(!nodus_edge_runtime_configure(
        runtime, dims, 1, 1, 0, sizeof(uint32_t), 0, 99, 0
    ));
    assert(!nodus_edge_runtime_configure(
        runtime,
        dims,
        1,
        std::numeric_limits<size_t>::max(),
        0,
        sizeof(uint32_t),
        0,
        0,
        0
    ));
    assert(nodus_edge_runtime_configure(
        runtime, dims, 1, 1, 0, sizeof(uint32_t), 0, 0, 0
    ));
    assert(nodus_edge_runtime_subscribe(runtime, 101, 1));
    assert(nodus_edge_runtime_subscribe(runtime, 202, 1));

    const std::array<uint32_t, 4> first{1, 2, 3, 4};
    int32_t dropped = -1;
    assert(nodus_edge_runtime_publish(
        runtime, 7, first.data(), sizeof(first), &dropped
    ));
    assert(dropped == 0);
    assert(!nodus_edge_runtime_is_quiescent(runtime));

    size_t snapshot_size = 0;
    assert(nodus_edge_runtime_snapshot_size(runtime, &snapshot_size));
    std::vector<uint8_t> snapshot(snapshot_size);
    assert(nodus_edge_runtime_snapshot_fill(
        runtime, snapshot.data(), snapshot.size()
    ));

    std::array<uint32_t, 4> output{};
    size_t written = 0;
    assert(nodus_edge_runtime_consume(
        runtime, 101, output.data(), sizeof(output), &written
    ));
    assert(output == first && written == sizeof(first));
    assert(!nodus_edge_runtime_is_quiescent(runtime));
    assert(!nodus_edge_runtime_publish(
        runtime, 7, first.data(), sizeof(first), &dropped
    ));
    assert(dropped == 0);

    assert(nodus_edge_runtime_snapshot_restore(
        runtime, snapshot.data(), snapshot.size()
    ));
    assert(nodus_edge_runtime_unread(runtime, 101) == 1);
    assert(nodus_edge_runtime_unread(runtime, 202) == 1);

    assert(nodus_edge_runtime_consume(
        runtime, 101, output.data(), sizeof(output), &written
    ));
    assert(nodus_edge_runtime_consume(
        runtime, 202, output.data(), sizeof(output), &written
    ));
    assert(nodus_edge_runtime_is_quiescent(runtime));
    nodus_edge_runtime_destroy(runtime);
    return 0;
}
