// mem_backend.h -- Memory backend abstraction (prototype)
// Guarded by NODUS_MEMBACKEND_PROTOTYPE
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdbool.h>

// Expose SPIR-V translation integration for use by other components.
// Keep C++-only declarations outside the C ABI block to avoid C linkage
// applying to C++ headers and templates.
#ifdef __cplusplus
#include "spirv_translation.h"
void gp_mem_backend_translate_to_spirv(const nodus::spirv::KernelIR& ir);
extern "C" {
#endif

// Forward declare table context so backends can expose table-edge attach helpers.
typedef struct GP_TableContext GP_TableContext;

// Opaque handle (exported early so top-level helpers can use it)
typedef void* gp_mem_backend_handle_t;

// Helpers to attach a backend to a table edge. Implemented in the table
// ABI translation unit. These are convenient for consumers (e.g., ThreadManager)
// to query backend capabilities without depending on table internals.
gp_mem_backend_handle_t gp_table_edge_get_backend(GP_TableContext* ctx, int32_t edge_idx);
int32_t gp_table_edge_set_backend(GP_TableContext* ctx, int32_t edge_idx, gp_mem_backend_handle_t h);

// Storage backend types (future-proofed list)
typedef enum {
    GP_MEM_BACKEND_CPU = 0,         // ordinary host memory
    GP_MEM_BACKEND_TORCH = 1,       // PyTorch tensor (CPU/GPU)
    GP_MEM_BACKEND_CUDA = 2,        // CUDA device buffer
    GP_MEM_BACKEND_VULKAN = 3,      // Vulkan device buffer
    GP_MEM_BACKEND_OPENGL = 4,      // OpenGL buffer object
    GP_MEM_BACKEND_SHARED_MEM = 5,  // POSIX/Windows shared memory
    GP_MEM_BACKEND_NETWORK = 6,     // Network/remote-backed memory descriptor
    GP_MEM_BACKEND_PYTHON = 7,      // Python object-backed buffer (pybuffer protocol)
    GP_MEM_BACKEND_CUSTOM = 0x8000  // reserved for user extensions
} GP_MemBackendType;

// Capability flags (bitmask)
typedef enum {
    GP_MEM_CAP_MAP_HOST        = 1u << 0, // supports gp_mem_backend_map / unmap
    GP_MEM_CAP_RESIZE          = 1u << 1, // supports resize
    GP_MEM_CAP_DEVICE_POINTER  = 1u << 2, // provides a device pointer/native handle for device interop
    GP_MEM_CAP_SHARED_HANDLE   = 1u << 3, // can export/import shared handles (OS or GL/VK handles)
    GP_MEM_CAP_PYBUFFER        = 1u << 4, // supports Python buffer protocol
    GP_MEM_CAP_ZERO_COPY       = 1u << 5, // zero-copy between host and device where applicable
    GP_MEM_CAP_BYREF_SAFE      = 1u << 6, // safe to store raw pointers (size/alignment sufficient)
    // Backend supports device-to-device native/D2D transfers (cross-handle
    // copies without host staging). Backends that can perform peer copies or
    // expose native interop should set this flag.
    GP_MEM_CAP_D2D_NATIVE     = 1u << 7
} GP_MemBackendCapability;

// Backend descriptor: describes properties at creation or runtime.
typedef struct {
    GP_MemBackendType type;
    uint64_t capabilities; // combination of GP_MemBackendCapability bits
    size_t size;          // current allocation size in bytes (0 allowed)
    uint32_t alignment;   // alignment guarantee in bytes
    // native handle for backend-specific interop; opaque value (e.g., GL name, FD, pointer)
    uintptr_t native_handle;
    // runtime limits and advisory guidance for using this backend
    uint32_t cpu_thread_ceiling; // suggested max host threads for compute on this backend (0 == unknown)
    size_t max_alloc_bytes;      // advisory maximum allocation size (0 == no explicit limit)
    uint32_t compute_deploy_limit; // advisory number of compute units / queues recommended
} gp_mem_backend_desc_t;

// Vtable for backend-optimized operations. Implementations should populate
// these where supported to allow zero-copy or device-native memcpy.
typedef struct {
    // map/unmap may be provided, but copy functions are preferred for safety.
    void* (*map)(gp_mem_backend_handle_t h);
    void (*unmap)(gp_mem_backend_handle_t h);
    // allocate a new backend-local buffer/handle. Returns a new
    // `gp_mem_backend_handle_t` representing the allocated buffer, or NULL
    // on failure. The first argument is the owning backend handle (for
    // backends that differentiate between backend context and buffers).
    gp_mem_backend_handle_t (*alloc)(gp_mem_backend_handle_t backend_h, size_t bytes, uint32_t alignment);
    // resize an existing buffer/handle. Returns 1 on success.
    int (*resize)(gp_mem_backend_handle_t buffer_h, size_t bytes);
    // free an allocated buffer/handle previously returned by alloc().
    void (*free)(gp_mem_backend_handle_t buffer_h);
    // copy bytes from host memory into backend at byte offset. Return 1 on success.
    int (*copy_to_backend)(gp_mem_backend_handle_t h, size_t offset, const void* src, size_t bytes);
    // copy bytes from backend into host memory at byte offset. Return 1 on success.
    int (*copy_from_backend)(gp_mem_backend_handle_t h, size_t offset, void* dst, size_t bytes);
    // optional device-to-device optimized transfer between two backend handles.
    // If provided, implementations should perform an efficient device-native
    // copy from `src` to `dst`. Return 1 on success.
    int (*copy_between_backends)(gp_mem_backend_handle_t src, gp_mem_backend_handle_t dst, size_t src_offset, size_t dst_offset, size_t bytes);
    // Return an opaque native handle or pointer for the provided buffer/handle
    // (e.g., device pointer, file descriptor, GL name). Return 0 if none.
    uintptr_t (*get_native_handle)(gp_mem_backend_handle_t h);

    // Synchronization helpers: allow recording or waiting on backend-native
    // events/streams. The event pointers are opaque to the ABI and may be
    // backend-specific (e.g., CUDA event, Vulkan semaphore). Return 1 on
    // success, 0 on failure / unsupported.
    int (*record_event)(gp_mem_backend_handle_t h, void* out_event, size_t out_event_size);
    int (*wait_event)(gp_mem_backend_handle_t h, const void* event, size_t event_size, uint64_t timeout_ms);
    // Optionally return an opaque native stream/queue handle for the buffer
    // (e.g., CUDA stream pointer). Return 0 if unsupported.
    uintptr_t (*get_native_stream)(gp_mem_backend_handle_t h);

    // Dispatch/execute a kernel described by a KernelIR (SPIR-V operator set)
    // Returns 1 on success, 0 on failure. May be a stub for backends that do not support execution.
#ifdef __cplusplus
    int (*dispatch_kernel)(gp_mem_backend_handle_t backend_h, const nodus::spirv::KernelIR* kernel_ir);
#else
    void* dispatch_kernel; // for C compatibility
#endif
} gp_mem_backend_vtable_t;

// Retrieve vtable for a backend handle (may return NULL if none).
const gp_mem_backend_vtable_t* gp_mem_backend_get_vtable(gp_mem_backend_handle_t h);

// Per-backend creation parameter structs. These are embedded or pointed-to
// by higher-level language/integration layers when creating a backend.

// CPU backend params (OS memory)
typedef struct {
    // initial allocation size
    size_t initial_bytes;
    // optional hint for alignment (0 == default)
    uint32_t alignment;
    // reserved for future flags
    uint32_t flags;
} gp_mem_backend_cpu_params_t;

// Torch backend params (language-agnostic descriptor)
typedef struct {
    // opaque pointer to a torch tensor object (up to caller to manage lifetime)
    void* torch_tensor_obj;
    // device ordinal (CPU:-1, GPU:0..N)
    int32_t device;
    // element size hint (bytes) - 0 if unknown
    uint32_t elem_size_hint;
    uint32_t flags;
} gp_mem_backend_torch_params_t;

// OpenGL backend params
typedef struct {
    // GL buffer object name (GLuint) promoted to uintptr_t for ABI stability
    uintptr_t gl_buffer_name;
    // GL target (e.g., GL_ARRAY_BUFFER). Caller may pass 0 if unknown.
    uint32_t gl_target;
    // usage hint (GL_STATIC_DRAW etc) encoded by caller
    uint32_t gl_usage;
    // optional GL context id (opaque to ABI) - frontend may register contexts and pass id here
    uintptr_t gl_context_id;
} gp_mem_backend_gl_params_t;

// Python backend params - opaque Python object that supports buffer protocol.
typedef struct {
    // opaque pointer to a Python object (PyObject*) if embedding Python; caller owns reference
    void* py_obj;
    // optional requirements about mutability or lifetime
    uint32_t flags;
} gp_mem_backend_python_params_t;

// Param/activation hook ABI
typedef struct {
    // generic params blob pointer (caller-owned memory copied by backend on set)
    const void* params_blob;
    size_t params_size;
} gp_mem_backend_params_blob_t;

// Activation hook signature. Return 1 on success, 0 on failure.
typedef int (*gp_mem_backend_activate_fn)(gp_mem_backend_handle_t h, const void* args, size_t args_size, void* user_ctx);

// Set/get params and activation hook
int gp_mem_backend_set_params(gp_mem_backend_handle_t h, const void* params, size_t params_size);
int gp_mem_backend_get_params(gp_mem_backend_handle_t h, void* out_params, size_t* inout_size);
int gp_mem_backend_set_activation_hook(gp_mem_backend_handle_t h, gp_mem_backend_activate_fn fn, void* user_ctx);
int gp_mem_backend_activate(gp_mem_backend_handle_t h, const void* args, size_t args_size);

// Main C API: create/release/map/resize/size
gp_mem_backend_handle_t gp_mem_backend_create_cpu(size_t initial_bytes);
// New alias for host-backed memory creation (preferred name). Deprecated
// wrapper around `gp_mem_backend_create_cpu` for clarity.
gp_mem_backend_handle_t gp_mem_backend_create_host(size_t initial_bytes);
gp_mem_backend_handle_t gp_mem_backend_create_from_params(GP_MemBackendType type, const void* params, size_t params_size);
// Filesystem-backed backend: stores buffers as files under `base_dir`.
gp_mem_backend_handle_t gp_mem_backend_create_filesystem(const char* base_dir, size_t initial_bytes);
void gp_mem_backend_release(gp_mem_backend_handle_t h);
void* gp_mem_backend_map(gp_mem_backend_handle_t h);
void gp_mem_backend_unmap(gp_mem_backend_handle_t h);
int gp_mem_backend_resize(gp_mem_backend_handle_t h, size_t bytes);
size_t gp_mem_backend_size(gp_mem_backend_handle_t h);

// Query helpers
uint64_t gp_mem_backend_get_capabilities(gp_mem_backend_handle_t h);
int gp_mem_backend_get_type(gp_mem_backend_handle_t h);
// Returns an opaque native handle (e.g., GL name, FD, device ptr) if available
uintptr_t gp_mem_backend_native_handle(gp_mem_backend_handle_t h);

// Manifest helpers: write a manifest listing native chunk handles and sizes
// and read a manifest to recreate/import backend buffer handles. The
// write function returns 1 on success, 0 on failure. The read function
// will allocate an array of `gp_mem_backend_handle_t` via `malloc` and
// assign it to `out_buffers`; caller is responsible for freeing that
// array and releasing each buffer handle.
int gp_mem_backend_write_manifest(const char* manifest_path, gp_mem_backend_handle_t* chunk_handles, size_t count);
int gp_mem_backend_read_manifest_and_allocate(gp_mem_backend_handle_t backend_ctx, const char* manifest_path, gp_mem_backend_handle_t** out_buffers, size_t* out_count);

// Transfer bytes directly between two backends (device-to-device) or
// between a backend and host (use NULL for host). Attempts to use any
// available backend vtable hooks to perform optimized device-to-device
// copies. Returns 1 on success, 0 on failure.
int gp_mem_backend_transfer(gp_mem_backend_handle_t src, gp_mem_backend_handle_t dst, size_t src_offset, size_t dst_offset, size_t bytes);

// Convenience helper: returns non-zero if backend is safe to store raw pointers
static inline int gp_mem_backend_supports_byref(gp_mem_backend_handle_t h) {
    return (gp_mem_backend_get_capabilities(h) & (uint64_t)GP_MEM_CAP_BYREF_SAFE) != 0;
}

// Small mapping helper: ensures host mapping and returns pointer or NULL.
// This helper will call map() and return the pointer; caller must call unmap().
static inline void* gp_mem_backend_map_or_null(gp_mem_backend_handle_t h) {
    if (!h) return NULL;
    uint64_t caps = gp_mem_backend_get_capabilities(h);
    if ((caps & (uint64_t)GP_MEM_CAP_MAP_HOST) == 0) return NULL;
    return gp_mem_backend_map(h);
}

// Utility: map backend type to a human-readable string. Returns a static string.
const char* gp_mem_backend_type_to_string(GP_MemBackendType t);

// OpenGL context registration helpers. Frontend should register its GL context
// by providing an opaque id and a native pointer/handle value. Returns 1 on
// success.
int gp_mem_backend_register_gl_context(uintptr_t ctx_id, uintptr_t native_ctx_ptr);
int gp_mem_backend_unregister_gl_context(uintptr_t ctx_id);
// Returns native context pointer or 0 if not found.
uintptr_t gp_mem_backend_get_registered_gl_context(uintptr_t ctx_id);

// Torch capability query, isolated behind the C ABI so callers never need to
// include <torch/torch.h> just to ask "is CUDA available?". Implemented in
// mem_backend_torch.cpp (the one translation unit allowed to depend on real
// libtorch headers). Returns 1 if libtorch reports a usable CUDA device, 0
// otherwise (including builds where the Torch backend isn't compiled in).
int gp_mem_backend_torch_cuda_available(void);

#ifdef __cplusplus
}
#endif
