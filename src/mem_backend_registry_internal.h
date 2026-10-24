#pragma once

#include "mem_backend.h"

#include <cstddef>
#include <cstdint>

// Implementation-side registration for backend handles whose concrete type
// lives outside mem_backend_host.cpp.  The registry itself remains in
// nodus_tensor_core, alongside the existing CPU/filesystem registry, so a
// Vulkan handle has the same process-wide identity when used by hosts,
// generated row DLLs, RawStackFrame, and EdgeTensorFifo.
namespace nodus::mem_backend_internal {

void register_external_handle(
    gp_mem_backend_handle_t handle,
    GP_MemBackendType type,
    const gp_mem_backend_vtable_t* vtable,
    std::size_t bytes,
    std::uint64_t capabilities);

void update_external_size(gp_mem_backend_handle_t handle, std::size_t bytes);

void unregister_external_handle(gp_mem_backend_handle_t handle);

} // namespace nodus::mem_backend_internal
