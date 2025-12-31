// mem_backend_torch.cpp -- Minimal Torch-backed memory vtable (CPU-only)
#include "mem_backend.h"
#if defined(__has_include)
# if __has_include(<torch/torch.h>)
#  include <torch/torch.h>
#  define NODUS_HAVE_LIBTORCH 1
# else
#  define NODUS_HAVE_LIBTORCH 0
namespace torch { class Tensor; namespace detail { } }
# endif
#else
# include <torch/torch.h>
# define NODUS_HAVE_LIBTORCH 1
#endif
#include <cstdint>
#include <cstring>

// Helper to get torch tensor pointer from a backend handle via native_handle
static torch::Tensor* torch_tensor_from_handle(gp_mem_backend_handle_t h) {
    if (!h) return nullptr;
    uintptr_t nh = gp_mem_backend_native_handle(h);
    if (nh == 0) return nullptr;
    return reinterpret_cast<torch::Tensor*>(nh);
}

static void* torch_map(gp_mem_backend_handle_t h) {
    torch::Tensor* t = torch_tensor_from_handle(h);
    if (!t) return nullptr;
    if (!t->device().is_cpu()) return nullptr;
    return t->data_ptr();
}

static void torch_unmap(gp_mem_backend_handle_t h) { (void)h; }

static gp_mem_backend_handle_t torch_alloc(gp_mem_backend_handle_t backend_h, size_t bytes, uint32_t alignment) {
    (void)backend_h; (void)alignment;
    // Allocate a CPU uint8 tensor to hold raw bytes
    try {
        auto t = new torch::Tensor(torch::empty({(int64_t)bytes}, torch::kUInt8));
        return reinterpret_cast<gp_mem_backend_handle_t>(t);
    } catch (...) { return nullptr; }
}

static int torch_resize(gp_mem_backend_handle_t buffer_h, size_t bytes) {
    if (!buffer_h) return 0;
    torch::Tensor* t = reinterpret_cast<torch::Tensor*>(buffer_h);
    try {
        *t = torch::empty({(int64_t)bytes}, torch::kUInt8);
        return 1;
    } catch (...) { return 0; }
}

static void torch_free(gp_mem_backend_handle_t buffer_h) {
    if (!buffer_h) return;
    torch::Tensor* t = reinterpret_cast<torch::Tensor*>(buffer_h);
    delete t;
}

static int torch_copy_to_backend(gp_mem_backend_handle_t h, size_t offset, const void* src, size_t bytes) {
    torch::Tensor* t = torch_tensor_from_handle(h);
    if (!t) return 0;
    if (!t->device().is_cpu()) return 0; // CPU-only for now
    uint8_t* dst = reinterpret_cast<uint8_t*>(t->data_ptr());
    if (offset + bytes > static_cast<size_t>(t->numel())) return 0;
    std::memcpy(dst + offset, src, bytes);
    return 1;
}

static int torch_copy_from_backend(gp_mem_backend_handle_t h, size_t offset, void* dst, size_t bytes) {
    torch::Tensor* t = torch_tensor_from_handle(h);
    if (!t) return 0;
    if (!t->device().is_cpu()) return 0; // CPU-only for now
    uint8_t* src = reinterpret_cast<uint8_t*>(t->data_ptr());
    if (offset + bytes > static_cast<size_t>(t->numel())) return 0;
    std::memcpy(dst, src + offset, bytes);
    return 1;
}

static int torch_copy_between(gp_mem_backend_handle_t src, gp_mem_backend_handle_t dst, size_t src_offset, size_t dst_offset, size_t bytes) {
    // For simplicity, support only when both handles are CPU tensors allocated by this vtable
    torch::Tensor* ts = reinterpret_cast<torch::Tensor*>(src);
    torch::Tensor* td = reinterpret_cast<torch::Tensor*>(dst);
    if (!ts || !td) return 0;
    if (!ts->device().is_cpu() || !td->device().is_cpu()) return 0;
    uint8_t* s = reinterpret_cast<uint8_t*>(ts->data_ptr());
    uint8_t* d = reinterpret_cast<uint8_t*>(td->data_ptr());
    if (src_offset + bytes > static_cast<size_t>(ts->numel())) return 0;
    if (dst_offset + bytes > static_cast<size_t>(td->numel())) return 0;
    std::memcpy(d + dst_offset, s + src_offset, bytes);
    return 1;
}

static uintptr_t torch_get_native_handle(gp_mem_backend_handle_t h) {
    // If the handle is a GenericBackend (from gp_mem_backend_create_from_params), the native
    // handle is available via gp_mem_backend_native_handle. Otherwise if a raw torch::Tensor*
    // buffer was allocated by this vtable, return its pointer.
    uintptr_t nh = gp_mem_backend_native_handle(h);
    if (nh != 0) return nh;
    return reinterpret_cast<uintptr_t>(h);
}

static int torch_record_event(gp_mem_backend_handle_t h, void* out_event, size_t out_event_size) { (void)h; (void)out_event; (void)out_event_size; return 0; }
static int torch_wait_event(gp_mem_backend_handle_t h, const void* event, size_t event_size, uint64_t timeout_ms) { (void)h; (void)event; (void)event_size; (void)timeout_ms; return 0; }
static uintptr_t torch_get_native_stream(gp_mem_backend_handle_t h) { (void)h; return 0; }

// Expose vtable
extern "C" const gp_mem_backend_vtable_t g_torch_vtable_real = {
    &torch_map,
    &torch_unmap,
    &torch_alloc,
    &torch_resize,
    &torch_free,
    &torch_copy_to_backend,
    &torch_copy_from_backend,
    &torch_copy_between,
    &torch_get_native_handle,
    &torch_record_event,
    &torch_wait_event,
    &torch_get_native_stream
};
