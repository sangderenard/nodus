#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NODUS_RUNTIME_BUILD)
#    define NODUS_RUNTIME_API __declspec(dllexport)
#  else
#    define NODUS_RUNTIME_API __declspec(dllimport)
#  endif
#else
#  define NODUS_RUNTIME_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NodusEdgeRuntime NodusEdgeRuntime;

NODUS_RUNTIME_API NodusEdgeRuntime* nodus_edge_runtime_create(void);
NODUS_RUNTIME_API void nodus_edge_runtime_destroy(NodusEdgeRuntime* runtime);

NODUS_RUNTIME_API int32_t nodus_edge_runtime_configure(
    NodusEdgeRuntime* runtime,
    const int32_t* dims,
    int32_t dim_count,
    size_t slots,
    size_t top_k,
    size_t elem_size,
    int32_t type_id,
    int32_t layout,
    int32_t dtype
);
NODUS_RUNTIME_API int32_t nodus_edge_runtime_subscribe(
    NodusEdgeRuntime* runtime, uint64_t reader_id, int32_t start_at_head
);
NODUS_RUNTIME_API void nodus_edge_runtime_unsubscribe(
    NodusEdgeRuntime* runtime, uint64_t reader_id
);
NODUS_RUNTIME_API int32_t nodus_edge_runtime_publish(
    NodusEdgeRuntime* runtime,
    uint64_t writer_id,
    const void* sample,
    size_t sample_bytes,
    int32_t* out_dropped
);
NODUS_RUNTIME_API int32_t nodus_edge_runtime_consume(
    NodusEdgeRuntime* runtime,
    uint64_t reader_id,
    void* sample,
    size_t sample_capacity,
    size_t* out_written
);
NODUS_RUNTIME_API uint64_t nodus_edge_runtime_unread(
    const NodusEdgeRuntime* runtime, uint64_t reader_id
);
NODUS_RUNTIME_API uint64_t nodus_edge_runtime_write_sequence(
    const NodusEdgeRuntime* runtime
);
NODUS_RUNTIME_API int32_t nodus_edge_runtime_is_quiescent(
    const NodusEdgeRuntime* runtime
);

NODUS_RUNTIME_API int32_t nodus_edge_runtime_snapshot_size(
    const NodusEdgeRuntime* runtime, size_t* out_size
);
NODUS_RUNTIME_API int32_t nodus_edge_runtime_snapshot_fill(
    NodusEdgeRuntime* runtime, void* out_buffer, size_t buffer_size
);
NODUS_RUNTIME_API int32_t nodus_edge_runtime_snapshot_restore(
    NodusEdgeRuntime* runtime, const void* buffer, size_t buffer_size
);

#ifdef __cplusplus
}
#endif
