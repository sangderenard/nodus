#include "nodus_runtime_abi.h"

#include "edge_tensor_fifo.h"
#include "edge_tensor_fifo_transaction.h"

#include <limits>
#include <new>
#include <vector>

struct NodusEdgeRuntime {
    EdgeTensorFifo fifo;
};

extern "C" NODUS_RUNTIME_API NodusEdgeRuntime* nodus_edge_runtime_create(void) {
    try {
        return new (std::nothrow) NodusEdgeRuntime();
    } catch (...) {
        return nullptr;
    }
}

extern "C" NODUS_RUNTIME_API void nodus_edge_runtime_destroy(
    NodusEdgeRuntime* runtime
) {
    delete runtime;
}

extern "C" NODUS_RUNTIME_API int32_t nodus_edge_runtime_configure(
    NodusEdgeRuntime* runtime,
    const int32_t* dims,
    int32_t dim_count,
    size_t slots,
    size_t top_k,
    size_t elem_size,
    int32_t type_id,
    int32_t layout,
    int32_t dtype
) {
    if (!runtime || !dims || dim_count <= 0 || dim_count > 16 ||
        slots == 0 || elem_size == 0 || layout < 0 || layout > 2 ||
        dtype < 0 || dtype > 18) return 0;
    size_t stride = 1;
    for (int32_t i = 0; i < dim_count; ++i) {
        if (dims[i] <= 0 ||
            static_cast<size_t>(dims[i]) >
                std::numeric_limits<size_t>::max() / stride) return 0;
        stride *= static_cast<size_t>(dims[i]);
    }
    if (slots > std::numeric_limits<size_t>::max() / stride ||
        elem_size > std::numeric_limits<size_t>::max() / (stride * slots)) {
        return 0;
    }
    try {
        std::vector<int32_t> shape(dims, dims + dim_count);
        runtime->fifo.configure(
            shape,
            slots,
            top_k,
            elem_size,
            type_id,
            static_cast<nodus::tensors::TensorLayout>(layout),
            static_cast<nodus::tensors::TensorDType>(dtype)
        );
    } catch (...) {
        return 0;
    }
    return runtime->fifo.impl && runtime->fifo.impl->configured ? 1 : 0;
}

extern "C" NODUS_RUNTIME_API int32_t nodus_edge_runtime_subscribe(
    NodusEdgeRuntime* runtime, uint64_t reader_id, int32_t start_at_head
) {
    return runtime && runtime->fifo.subscribe(reader_id, start_at_head != 0)
        ? 1 : 0;
}

extern "C" NODUS_RUNTIME_API void nodus_edge_runtime_unsubscribe(
    NodusEdgeRuntime* runtime, uint64_t reader_id
) {
    if (runtime) runtime->fifo.unsubscribe(reader_id);
}

extern "C" NODUS_RUNTIME_API int32_t nodus_edge_runtime_publish(
    NodusEdgeRuntime* runtime,
    uint64_t writer_id,
    const void* sample,
    size_t sample_bytes,
    int32_t* out_dropped
) {
    if (!runtime || !out_dropped) return 0;
    bool dropped = false;
    bool accepted = false;
    try {
        accepted = runtime->fifo.push(
            0, writer_id, sample, sample_bytes, &dropped
        );
    } catch (...) {
        *out_dropped = 0;
        return 0;
    }
    *out_dropped = dropped ? 1 : 0;
    return accepted ? 1 : 0;
}

extern "C" NODUS_RUNTIME_API int32_t nodus_edge_runtime_consume(
    NodusEdgeRuntime* runtime,
    uint64_t reader_id,
    void* sample,
    size_t sample_capacity,
    size_t* out_written
) {
    if (!runtime || !out_written) return 0;
    size_t written = 0;
    bool consumed = false;
    try {
        consumed = runtime->fifo.pop(
            reader_id, sample, sample_capacity, written
        );
    } catch (...) {
        *out_written = 0;
        return 0;
    }
    *out_written = written;
    return consumed ? 1 : 0;
}

extern "C" NODUS_RUNTIME_API uint64_t nodus_edge_runtime_unread(
    const NodusEdgeRuntime* runtime, uint64_t reader_id
) {
    return runtime ? runtime->fifo.unread(reader_id) : 0;
}

extern "C" NODUS_RUNTIME_API uint64_t nodus_edge_runtime_write_sequence(
    const NodusEdgeRuntime* runtime
) {
    return runtime ? runtime->fifo.write_seq() : 0;
}

extern "C" NODUS_RUNTIME_API int32_t nodus_edge_runtime_is_quiescent(
    const NodusEdgeRuntime* runtime
) {
    if (!runtime || !runtime->fifo.impl || !runtime->fifo.impl->configured) {
        return 0;
    }
    const uint64_t head = runtime->fifo.write_seq();
    for (size_t i = 0; i < EdgeTensorFifo::kMaxReaders; ++i) {
        const auto& reader = runtime->fifo.impl->readers[i];
        if (reader.key.load(std::memory_order_acquire) != 0 &&
            reader.seq.load(std::memory_order_acquire) != head) {
            return 0;
        }
    }
    return 1;
}

extern "C" NODUS_RUNTIME_API int32_t nodus_edge_runtime_snapshot_size(
    const NodusEdgeRuntime* runtime, size_t* out_size
) {
    if (!runtime || !out_size) return 0;
    size_t size = 0;
    try {
        size = nodus::runtime::edge_transaction_snapshot_size(runtime->fifo);
    } catch (...) {
        return 0;
    }
    if (size == 0) return 0;
    *out_size = size;
    return 1;
}

extern "C" NODUS_RUNTIME_API int32_t nodus_edge_runtime_snapshot_fill(
    NodusEdgeRuntime* runtime, void* out_buffer, size_t buffer_size
) {
    if (!runtime) return 0;
    try {
        return nodus::runtime::edge_transaction_snapshot_fill(
            runtime->fifo, out_buffer, buffer_size
        ) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

extern "C" NODUS_RUNTIME_API int32_t nodus_edge_runtime_snapshot_restore(
    NodusEdgeRuntime* runtime, const void* buffer, size_t buffer_size
) {
    if (!runtime) return 0;
    try {
        return nodus::runtime::edge_transaction_snapshot_restore(
            runtime->fifo, buffer, buffer_size
        ) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}
