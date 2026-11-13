// Native OpenGL compute packaging and ordered submission on a caller-owned
// current context. No context creation, renderer, shader rewriting or RNG.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* (*gp_gl_get_proc_fn)(const char* name, void* user);
typedef uintptr_t (*gp_gl_current_context_fn)(void* user);

typedef struct {
    // Must match gp_mem_backend_register_gl_context and the current context.
    uintptr_t context_id;
    gp_gl_get_proc_fn get_proc;
    gp_gl_current_context_fn current_context;
    void* user;
} gp_gl_compute_context;

typedef struct {
    const char* label;
    const char* source;
    size_t source_bytes;
} gp_gl_compute_source;

typedef struct gp_gl_compute_program gp_gl_compute_program;

// Every source is one GL_COMPUTE_SHADER object, attached to ONE program.
// Source bytes are sent unchanged. This is not function partitioning: the
// author/compiler must supply independently compilable units with one main.
// A combined link still exposes one reachable graph to the driver's linker.
// Logs identify object/compile/link phase. On failure, *program is normally
// null; if cleanup cannot be verified it instead owns remaining driver objects
// and MUST be released on the owning context. That owner is not executable.
// Native driver calls are synchronous. Use an externally bounded process for
// an untrusted or previously memory-exhausting driver compilation.
int gp_gl_compute_link(const gp_gl_compute_context* context,
                       const gp_gl_compute_source* sources, size_t count,
                       gp_gl_compute_program** program,
                       char* error, size_t error_capacity);
uint32_t gp_gl_compute_program_name(const gp_gl_compute_program* program);
// Requires the owning context current. Does not release on context mismatch.
int gp_gl_compute_release(const gp_gl_compute_context* context,
                          gp_gl_compute_program* program,
                          char* error, size_t error_capacity);

typedef struct {
    uint32_t binding;
    uint32_t buffer;
    // bytes == 0 means bind the complete buffer (offset must then be zero).
    size_t offset, bytes;
} gp_gl_storage_binding;

typedef struct {
    uint32_t unit, texture;
    int32_t level, layer;
    uint8_t layered;
    uint32_t access, format; // GL_READ_ONLY/WRITE_ONLY/READ_WRITE and format
} gp_gl_image_binding;

typedef enum {
    GP_GL_UNIFORM_I32 = 1,
    GP_GL_UNIFORM_U32 = 2,
    GP_GL_UNIFORM_F32 = 3,
    GP_GL_UNIFORM_MATRIX_F32 = 4
} gp_gl_uniform_type;
typedef struct {
    const char* name;
    gp_gl_uniform_type type;
    uint32_t components; // vectors 1..4, square matrices 2..4
    uint32_t count;
    const void* values;
    // Inactive uniforms are legitimate after source specialization, but only
    // when explicitly optional. Required missing uniforms refuse submission.
    uint8_t optional;
} gp_gl_uniform;

typedef enum {
    // Each dispatch starts and finishes its own native invocation. Existing
    // complete forward/reactive/sensor passes use this boundary, not a cut
    // inside a ray's bounce loop.
    GP_GL_COMPLETE_INVOCATION = 0,
    // Refused until an actual compiler-owned state ABI is implemented. A label
    // or a Boolean cannot prove preservation of locals, RNG and private globals.
    GP_GL_UNVERIFIED_CONTINUATION = 1
} gp_gl_compute_boundary;

typedef struct {
    const char* label;
    gp_gl_compute_program* program;
    uint32_t groups[3];
    const gp_gl_uniform* uniforms;
    size_t uniform_count;
    // The native schedule retains original order. Supply the exact required
    // visibility masks, including final visibility to the original consumer.
    uint32_t barrier_before, barrier_after;
    gp_gl_compute_boundary boundary;
} gp_gl_compute_stage;

typedef struct {
    // Same borrowed resources are rebound for every stage. The caller retains
    // ownership and lifetime. No clear, realloc, copy, readback or reseeding.
    const gp_gl_storage_binding* storage;
    size_t storage_count;
    const gp_gl_image_binding* images;
    size_t image_count;
    const gp_gl_compute_stage* stages;
    size_t stage_count;
} gp_gl_compute_schedule;

typedef struct {
    size_t dispatch_calls; // submitted calls, NOT GPU completion
    size_t stages_submitted; // dispatch and requested barrier accepted
    size_t failed_stage; // SIZE_MAX for preflight failure or success
    uint8_t dispatch_may_have_run; // failure after any dispatch: never replay
} gp_gl_compute_receipt;

// Preflight every program, uniform, resource and group shape before ANY
// dispatch. Then submit the authored sequence in this native call.
// Repeated use of one program must declare the same uniform names/types/extents
// in every stage, so preflight cannot leak a later stage's uniform into an
// earlier one. Each stage restores that complete scheduled set before dispatch.
// No automatic combined-link fallback, device switch, retry or replay. Success means commands
// submitted without an observed GL error, NOT completion/numerical parity.
// Leaves the last program/bindings current, like the existing Studio owner.
int gp_gl_compute_submit(const gp_gl_compute_context* context,
                         const gp_gl_compute_schedule* schedule,
                         gp_gl_compute_receipt* receipt,
                         char* error, size_t error_capacity);

// Read the caller's currently bound GL_READ_FRAMEBUFFER after its draw has
// completed. Returns an AbstractTensor U8[height,width,4] in native GL
// bottom-left row order. The caller owns the tensor handle and framebuffer;
// this operation changes only GL_PACK_ALIGNMENT and restores it.
// max_bytes is a host quota, checked before allocation or any GL call.
int gp_gl_frame_read_rgba8(const gp_gl_compute_context* context,
                           int32_t x, int32_t y, uint32_t width, uint32_t height,
                           uint64_t max_bytes, uint64_t* tensor_handle,
                           char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
