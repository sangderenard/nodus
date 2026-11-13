// Process-local OpenGL storage in the canonical Nodus memory-backend registry.
// The caller owns an already-current registered context and its manager lane.
// This API never creates a context, chooses a GPU, compiles, or dispatches.
#pragma once
#include "mem_backend.h"
#include "kernel_opengl_runner.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t max_buffer_bytes; // Measured device block limit intersected with policy.
    size_t max_total_bytes;  // Explicit reservation for this owner, not free VRAM.
    uint32_t max_buffers;
} gp_mem_backend_opengl_limits;

typedef struct {
    size_t live_bytes;
    uint32_t live_buffers;
    uint32_t pending_cleanup_buffers;
} gp_mem_backend_opengl_usage;

// No allocation is made until the canonical vtable's alloc() is called.
// Every allocated buffer is immutable in extent and initially zeroed. No host
// mapping, implicit resize, context activation, or cross-context copying exists.
// Neither this backend handle nor its GLuints may cross a process boundary.
gp_mem_backend_handle_t gp_mem_backend_create_opengl(
    const gp_gl_compute_context* context, const gp_mem_backend_opengl_limits* limits);

// Checked alternative to the void release/free ABI. A failed release retains
// both registry identity and ownership and may ONLY retry cleanup on this lane.
// The context owner cannot be released while buffers/pending cleanup remain.
int gp_mem_backend_opengl_release(gp_mem_backend_handle_t handle);
int gp_mem_backend_opengl_collect(gp_mem_backend_handle_t owner);
int gp_mem_backend_opengl_get_usage(gp_mem_backend_handle_t owner,
                                  gp_mem_backend_opengl_usage* usage);
// Proves that this registered buffer is owned by this exact process-local
// backend and context callback set on its current manager lane. CPU preflight
// only; no driver query or activation. A host/receipt ID cannot satisfy it.
int gp_mem_backend_opengl_validate_owner(gp_mem_backend_handle_t buffer,
                                       gp_mem_backend_handle_t owner,
                                       const gp_gl_compute_context* context);
const char* gp_mem_backend_opengl_last_error(void);

// Clear precisely this uint32-aligned view, not adjacent scalar/response/state
// views. Shader visibility is still the authored schedule owner's responsibility.
int gp_mem_backend_opengl_clear_u32(gp_mem_backend_handle_t buffer,
                                  size_t offset, size_t bytes, uint32_t value);
// Borrow a process-local binding for gp_gl_compute_submit. Extent is checked
// against the registered allocation; ownership remains with the backend.
int gp_mem_backend_opengl_storage_view(gp_mem_backend_handle_t buffer,
                                      uint32_t binding, size_t offset, size_t bytes,
                                      gp_gl_storage_binding* view);

#ifdef __cplusplus
}
#endif
