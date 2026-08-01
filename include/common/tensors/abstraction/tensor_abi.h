// tensor_abi.h -- C ABI over the in-memory tensor backend.
//
// This is the transport layer between a host language and nodus tensors.
// Everything crossing it is a plain integer, a pointer, or a POD struct, so a
// caller needs nothing beyond its own standard library -- ctypes on the Python
// side, with no numpy, torch, pybind11, or DLPack anywhere in the fundamental
// path. Tensor payloads move by mapping arena memory directly, not by copying
// through an intermediate array type.
//
// The C++ types this wraps (TensorDesc, TensorShape, ArenaStats) hold vectors
// and enum classes and cannot cross a C boundary; NodusTensorDesc and
// NodusArenaStats are their flat mirrors. Shapes are fixed-rank here so a
// descriptor is a single POD read with no ownership question -- a tensor of
// higher rank than NODUS_TENSOR_MAX_RANK is rejected rather than truncated.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NODUS_TENSOR_MAX_RANK 8

// Mirrors nodus::tensors::TensorDType.
typedef enum {
    NODUS_DTYPE_UNKNOWN = 0,
    NODUS_DTYPE_F32 = 1,
    NODUS_DTYPE_F64 = 2,
    NODUS_DTYPE_I8 = 3,
    NODUS_DTYPE_I16 = 4,
    NODUS_DTYPE_I32 = 5,
    NODUS_DTYPE_I64 = 6,
    NODUS_DTYPE_U8 = 7,
    NODUS_DTYPE_BYTES = 8,
    NODUS_DTYPE_BYTES2 = 9,
    NODUS_DTYPE_BYTES4 = 10,
    NODUS_DTYPE_BYTES8 = 11,
    NODUS_DTYPE_U16 = 12,
    NODUS_DTYPE_U32 = 13,
    NODUS_DTYPE_U64 = 14,
    NODUS_DTYPE_BOOL = 15,
    NODUS_DTYPE_PTR = 16
} NodusTensorDType;

// Mirrors nodus::tensors::InMemoryBackend::ArenaBackingPolicy.
typedef enum {
    NODUS_ARENA_VIRTUAL_ONLY = 0,
    NODUS_ARENA_ALLOW_NON_VIRTUAL = 1
} NodusArenaBackingPolicy;

typedef enum {
    NODUS_OK = 0,
    NODUS_ERR_INVALID_ARG = -1,
    NODUS_ERR_UNKNOWN_HANDLE = -2,
    NODUS_ERR_RANK_TOO_HIGH = -3,
    NODUS_ERR_ALLOC_FAILED = -4,
    NODUS_ERR_UNSUPPORTED = -5
} NodusStatus;

// Flat mirror of nodus::tensors::TensorDesc. Strides are in elements, matching
// TensorStrides, not bytes -- a caller converting to a byte-addressed view must
// scale by element_bytes.
typedef struct {
    uint8_t dtype;
    uint8_t layout;
    uint8_t is_readonly;
    uint8_t is_buffer;
    uint32_t rank;
    uint64_t shape[NODUS_TENSOR_MAX_RANK];
    uint64_t strides[NODUS_TENSOR_MAX_RANK];
    uint64_t element_count;
    uint32_t element_bytes;
    uint32_t reserved;
    uint64_t total_bytes;
} NodusTensorDesc;

// Flat mirror of InMemoryBackend::ArenaStats.
typedef struct {
    uint64_t reserve_bytes;
    uint64_t active_leases;
    uint64_t active_leased_bytes;
    uint64_t peak_active_leased_bytes;
    uint64_t alloc_calls;
    uint64_t free_calls;
    uint64_t alloc_fail_oom;
    uint64_t free_drop_no_nodes;
    uint32_t span_nodes_capacity;
    uint32_t span_nodes_free;
    uint64_t free_spans;
    uint64_t free_bytes;
    uint64_t largest_free_span;
} NodusArenaStats;

// -- arena ---------------------------------------------------------------
//
// These configure the process-wide arena and must be called before the first
// allocation; the backend fixes its backing at first use.

int32_t nodus_tensor_arena_configure(uint64_t reserve_bytes,
                                     uint64_t min_commit_bytes,
                                     uint64_t span_nodes,
                                     int32_t backing_policy);
int32_t nodus_tensor_arena_stats(NodusArenaStats* out);
uint64_t nodus_tensor_arena_system_available_bytes(void);
// Test-only: drop every arena so a new policy can be applied.
void nodus_tensor_arena_reset(void);

// -- tensors -------------------------------------------------------------

// Create a dense tensor of the given shape. Returns 0 on failure.
uint64_t nodus_tensor_create(int32_t dtype, uint32_t rank, const uint64_t* shape);
void nodus_tensor_destroy(uint64_t handle);
int32_t nodus_tensor_describe(uint64_t handle, NodusTensorDesc* out);

// Borrow the tensor's arena storage. The pointer stays valid until unmap, and
// the arena will not move underneath it -- that is what unmap releases.
int32_t nodus_tensor_map(uint64_t handle, void** out_data, uint64_t* out_bytes);
void nodus_tensor_unmap(uint64_t handle);

// Zero a tensor's payload, preferring a swap to an already-clean arena lease
// over clearing in place. byte_count == 0 means the whole payload.
int32_t nodus_tensor_zero(uint64_t handle, uint64_t byte_offset, uint64_t byte_count);

// Copy in/out without requiring the caller to hold a mapping. Both return the
// number of bytes transferred, clamped to the tensor payload.
int64_t nodus_tensor_write(uint64_t handle, uint64_t byte_offset,
                           const void* source, uint64_t bytes);
int64_t nodus_tensor_read(uint64_t handle, uint64_t byte_offset,
                          void* destination, uint64_t bytes);

// Where this tensor sits in the arena, for a backend that addresses the arena
// directly (a compiled Fortran or SPIR-V kernel reading a descriptor table)
// rather than through a mapped pointer.
int32_t nodus_tensor_allocation(uint64_t handle,
                                void** out_data,
                                uint64_t* out_bytes,
                                uint16_t* out_bucket);

uint32_t nodus_tensor_dtype_size(int32_t dtype);

// -- operators -----------------------------------------------------------
//
// Dispatched by nodus::ops::CanonicalOp rather than one entry point per
// operation: the operator table already exists, the CT_OP_* codes it carries
// already agree with the caller's, and a per-operation ABI would be a second
// copy of that agreement that could drift from it.
//
// These call tensor_elementwise_* directly, which the tensor math header
// names as the CPU semantics every caller must share instead of keeping a
// private copy of the operator math.

// Arity is deliberately not exposed here. The classification lives inside
// tensor_math.cpp as file-static range checks over the enum, and re-deriving
// it at this boundary would be a second copy of the operator table that could
// disagree with the first. A caller routes from its own canonical table.

int32_t nodus_tensor_unary(int32_t op, uint64_t input, uint64_t output);
int32_t nodus_tensor_binary(int32_t op, uint64_t left, uint64_t right,
                            uint64_t output);
// scalar_on_left distinguishes ``scalar - tensor`` from ``tensor - scalar``;
// for a commutative op it makes no difference and may be 0.
int32_t nodus_tensor_scalar(int32_t op, uint64_t tensor, double scalar,
                            int32_t scalar_on_left, uint64_t output);

// Matrix multiply, which is not elementwise and so is not a CanonicalOp: it
// calls tensor_matmul_f32/f64 in tensor_math.h, the same "in-memory backend
// only" dense helpers named there. f32 and f64 only -- any other dtype is
// NODUS_ERR_UNSUPPORTED rather than a quiet conversion. Those helpers report
// a shape or backend mismatch by returning an empty tensor, which becomes
// NODUS_ERR_INVALID_ARG here rather than a silently empty output.
int32_t nodus_tensor_matmul(uint64_t left, uint64_t right, uint64_t output);

#ifdef __cplusplus
}  // extern "C"
#endif
