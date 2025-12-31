// Deprecated shim: this file was renamed to mem_backend_host.cpp
// to better represent the host-backed memory backend implementation.
// The real implementation lives in mem_backend_host.cpp; keep this
// compilation unit as a no-op placeholder to avoid breaking older build
// setups that still reference mem_backend_cpu.cpp. It must not define
// any symbols to avoid duplicate-linker issues.

// Intentionally empty.

extern "C" const gp_mem_backend_vtable_t* gp_mem_backend_get_vtable(gp_mem_backend_handle_t h) {
    if (!h) return nullptr;
    // CPU
    CpuBackend* cb = reinterpret_cast<CpuBackend*>(h);
        std::lock_guard<std::mutex> lk(g_vt_mu);
        auto it = g_handle_vtables.find(h);
        if (it == g_handle_vtables.end()) return nullptr;
        return it->second;
    // Generic
    GenericBackend* gb = reinterpret_cast<GenericBackend*>(h);
    return gb->caps ? nullptr : nullptr; // generic backends currently have no vtable
}

// Transfer bytes between two backends (or host if handle is NULL).
extern "C" int gp_mem_backend_transfer(gp_mem_backend_handle_t src, gp_mem_backend_handle_t dst, size_t src_offset, size_t dst_offset, size_t bytes) {
    if (bytes == 0) return 1;
    // Prefer vtable-level direct copy_between_backends if available on src or dst
    if (src && dst) {
        const gp_mem_backend_vtable_t* s_vt = gp_mem_backend_get_vtable(src);
        if (s_vt && s_vt->copy_between_backends) {
            if (s_vt->copy_between_backends(src, dst, src_offset, dst_offset, bytes)) return 1;
        }
        const gp_mem_backend_vtable_t* d_vt = gp_mem_backend_get_vtable(dst);
        if (d_vt && d_vt->copy_between_backends) {
            if (d_vt->copy_between_backends(src, dst, src_offset, dst_offset, bytes)) return 1;
        }
    }
    // Fast-path: same handle, try to map once and memmove
    if (src && dst && src == dst) {
        const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(src);
        if (vt && vt->map && vt->unmap) {
            void* m = vt->map(src);
            if (m) {
                uint8_t* base = reinterpret_cast<uint8_t*>(m);
                std::memmove(base + dst_offset, base + src_offset, bytes);
                vt->unmap(src);
                return 1;
            }
        }
        // Fallback to copy_from + copy_to via temp buffer
    }

    // If both backends provide copy_from / copy_to hooks, do chunked transfer
    const gp_mem_backend_vtable_t* src_vt = src ? gp_mem_backend_get_vtable(src) : nullptr;
    const gp_mem_backend_vtable_t* dst_vt = dst ? gp_mem_backend_get_vtable(dst) : nullptr;

        // This transfer API currently requires both src and dst backend handles.
        // Host-to-backend or backend-to-host transfers should be implemented by
        // callers using map/copy or the per-backend vtable copy_{to,from}_backend
        if (!src || !dst) return 0;

    // General case: both sides present (device-to-device) or fallbacks.
    // Use a host-side staging buffer in chunks.
    size_t max_chunk = 4 * 1024 * 1024; // 4MB chunk
    size_t remaining = bytes;
    size_t off = 0;
    std::vector<uint8_t> tmp;
    try { tmp.resize(std::min<size_t>(remaining, max_chunk)); } catch (...) { return 0; }
    while (remaining > 0) {
        size_t cur = std::min<size_t>(remaining, tmp.size());
        // read from src
        bool ok_read = false;
        if (src_vt && src_vt->copy_from_backend) {
            ok_read = src_vt->copy_from_backend(src, src_offset + off, tmp.data(), cur) != 0;
        }
        if (!ok_read) {
            void* m = gp_mem_backend_map_or_null(src);
            if (m) {
                std::memcpy(tmp.data(), reinterpret_cast<uint8_t*>(m) + src_offset + off, cur);
                gp_mem_backend_unmap(src);
                ok_read = true;
            }
        }
        if (!ok_read) return 0;

        // write to dst
        bool ok_write = false;
        if (dst_vt && dst_vt->copy_to_backend) {
            ok_write = dst_vt->copy_to_backend(dst, dst_offset + off, tmp.data(), cur) != 0;
        }
        if (!ok_write) {
            void* m = gp_mem_backend_map_or_null(dst);
            if (m) {
                std::memcpy(reinterpret_cast<uint8_t*>(m) + dst_offset + off, tmp.data(), cur);
                gp_mem_backend_unmap(dst);
                ok_write = true;
            }
        }
        if (!ok_write) return 0;

        remaining -= cur;
        off += cur;
    }
    return 1;
}

extern "C" gp_mem_backend_handle_t gp_mem_backend_create_from_params(GP_MemBackendType type, const void* params, size_t params_size) {
    if (type == GP_MEM_BACKEND_CPU) {
        size_t initial = 0;
        uint32_t alignment = 0;
        if (params && params_size >= sizeof(gp_mem_backend_cpu_params_t)) {
            const gp_mem_backend_cpu_params_t* p = reinterpret_cast<const gp_mem_backend_cpu_params_t*>(params);
            initial = p->initial_bytes;
            alignment = p->alignment;
        }
        CpuBackend* b = reinterpret_cast<CpuBackend*>(gp_mem_backend_create_cpu(initial));
        if (!b) return nullptr;
        if (alignment != 0) b->alignment = alignment;
        return reinterpret_cast<gp_mem_backend_handle_t>(b);
    }

    // For other backends create a lightweight GenericBackend that stores params and basic caps.
    GenericBackend* g = new (std::nothrow) GenericBackend(type);
    if (!g) return nullptr;
    // copy params blob if provided
    if (params && params_size > 0) {
        g->params_blob = std::malloc(params_size);
        if (g->params_blob) {
            std::memcpy(g->params_blob, params, params_size);
            g->params_size = params_size;
        }
    }
    // set conservative capability hints based on type
    switch (type) {
        case GP_MEM_BACKEND_TORCH:
            g->caps = (uint64_t)GP_MEM_CAP_PYBUFFER | (uint64_t)GP_MEM_CAP_DEVICE_POINTER | (uint64_t)GP_MEM_CAP_BYREF_SAFE;
            g->cpu_thread_ceiling = 0;
            g->max_alloc_bytes = 0;
            g->compute_deploy_limit = 0;
            // If params contain gp_mem_backend_torch_params_t extract opaque tensor ptr
            if (params && params_size >= sizeof(gp_mem_backend_torch_params_t)) {
                const gp_mem_backend_torch_params_t* tp = reinterpret_cast<const gp_mem_backend_torch_params_t*>(params);
                if (tp->torch_tensor_obj) g->native_handle = reinterpret_cast<uintptr_t>(tp->torch_tensor_obj);
            }
            // register torch vtable if we've got a native handle (best-effort). Prefer
            // the real torch-enabled vtable when compiled with libtorch support.
            if (g->native_handle != 0) {
                std::lock_guard<std::mutex> lk(g_vt_mu);
                g_handle_vtables[reinterpret_cast<gp_mem_backend_handle_t>(g)] = &g_torch_vtable_real;
            }
            break;
        case GP_MEM_BACKEND_OPENGL:
            g->caps = (uint64_t)GP_MEM_CAP_SHARED_HANDLE | (uint64_t)GP_MEM_CAP_DEVICE_POINTER;
            // If params contain a gp_mem_backend_gl_params_t with a registered context id, attach it
            if (params && params_size >= sizeof(gp_mem_backend_gl_params_t)) {
                const gp_mem_backend_gl_params_t* gp = reinterpret_cast<const gp_mem_backend_gl_params_t*>(params);
                if (gp->gl_context_id != 0) {
                    uintptr_t ctxptr = gp_mem_backend_get_registered_gl_context(gp->gl_context_id);
                    if (ctxptr != 0) g->gl_context_ptr = ctxptr;
                }
                if (gp->gl_buffer_name) g->native_handle = gp->gl_buffer_name;
            }
            g->cpu_thread_ceiling = 0;
            g->max_alloc_bytes = 0;
            g->compute_deploy_limit = 0;
            // If we have a native GL buffer name and context pointer, register GL vtable
            if (g->native_handle != 0 && g->gl_context_ptr != 0) {
                std::lock_guard<std::mutex> lk(g_vt_mu);
                g_handle_vtables[reinterpret_cast<gp_mem_backend_handle_t>(g)] = &g_gl_vtable_ext;
            }
            break;
        case GP_MEM_BACKEND_PYTHON:
            g->caps = (uint64_t)GP_MEM_CAP_PYBUFFER | (uint64_t)GP_MEM_CAP_MAP_HOST;
            g->cpu_thread_ceiling = std::min<unsigned>(std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4u, 8u);
            g->max_alloc_bytes = 0;
            g->compute_deploy_limit = 0;
            break;
        default:
            g->caps = 0;
            break;
    }
    // Assign a default vtable for generic backends that do not implement custom copy hooks.
    static const gp_mem_backend_vtable_t g_generic_vtable = { nullptr, nullptr, nullptr, nullptr };
    g->caps |= 0; // no-op to avoid unused warning
    // store vtable pointer (may be overridden by concrete backend implementations later)
    (void)g_generic_vtable;
    // For GenericBackend keep vtable null (interpreted as no hooks)
    return reinterpret_cast<gp_mem_backend_handle_t>(g);
}

// params set/get/activate implementations
extern "C" int gp_mem_backend_set_params(gp_mem_backend_handle_t h, const void* params, size_t params_size) {
    if (!h) return 0;
    int t = gp_mem_backend_get_type(h);
    if (t == (int)GP_MEM_BACKEND_CPU) {
        CpuBackend* b = reinterpret_cast<CpuBackend*>(h);
        std::lock_guard<std::mutex> lk(b->mu);
        if (b->params_blob) std::free(b->params_blob);
        b->params_blob = nullptr; b->params_size = 0;
        if (params && params_size > 0) {
            b->params_blob = std::malloc(params_size);
            if (!b->params_blob) return 0;
            std::memcpy(b->params_blob, params, params_size);
            b->params_size = params_size;
        }
        return 1;
    }
    GenericBackend* g = reinterpret_cast<GenericBackend*>(h);
    std::lock_guard<std::mutex> lk(g->mu);
    if (g->params_blob) std::free(g->params_blob);
    g->params_blob = nullptr; g->params_size = 0;
    if (params && params_size > 0) {
        g->params_blob = std::malloc(params_size);
        if (!g->params_blob) return 0;
        std::memcpy(g->params_blob, params, params_size);
        g->params_size = params_size;
    }
    return 1;
}

extern "C" int gp_mem_backend_get_params(gp_mem_backend_handle_t h, void* out_params, size_t* inout_size) {
    if (!h || !inout_size) return 0;
    int t = gp_mem_backend_get_type(h);
    if (t == (int)GP_MEM_BACKEND_CPU) {
        CpuBackend* b = reinterpret_cast<CpuBackend*>(h);
        std::lock_guard<std::mutex> lk(b->mu);
        if (!b->params_blob) { *inout_size = 0; return 1; }
        if (!out_params) { *inout_size = b->params_size; return 1; }
        size_t copy = std::min(*inout_size, b->params_size);
        std::memcpy(out_params, b->params_blob, copy);
        *inout_size = copy;
        return 1;
    }
    GenericBackend* g = reinterpret_cast<GenericBackend*>(h);
    std::lock_guard<std::mutex> lk(g->mu);
    if (!g->params_blob) { *inout_size = 0; return 1; }
    if (!out_params) { *inout_size = g->params_size; return 1; }
    size_t copy = std::min(*inout_size, g->params_size);
    std::memcpy(out_params, g->params_blob, copy);
    *inout_size = copy;
    return 1;
}

extern "C" int gp_mem_backend_set_activation_hook(gp_mem_backend_handle_t h, gp_mem_backend_activate_fn fn, void* user_ctx) {
    if (!h) return 0;
    int t = gp_mem_backend_get_type(h);
    if (t == (int)GP_MEM_BACKEND_CPU) {
        CpuBackend* b = reinterpret_cast<CpuBackend*>(h);
        std::lock_guard<std::mutex> lk(b->mu);
        b->activate_fn = fn;
        b->activate_ctx = user_ctx;
        return 1;
    }
    GenericBackend* g = reinterpret_cast<GenericBackend*>(h);
    std::lock_guard<std::mutex> lk(g->mu);
    g->activate_fn = fn;
    g->activate_ctx = user_ctx;
    return 1;
}

extern "C" int gp_mem_backend_activate(gp_mem_backend_handle_t h, const void* args, size_t args_size) {
    if (!h) return 0;
    int t = gp_mem_backend_get_type(h);
    if (t == (int)GP_MEM_BACKEND_CPU) {
        CpuBackend* b = reinterpret_cast<CpuBackend*>(h);
        std::lock_guard<std::mutex> lk(b->mu);
        if (!b->activate_fn) return 1; // no-op success
        return b->activate_fn(h, args, args_size, b->activate_ctx);
    }
    GenericBackend* g = reinterpret_cast<GenericBackend*>(h);
    std::lock_guard<std::mutex> lk(g->mu);
    if (!g->activate_fn) return 1;
    return g->activate_fn(h, args, args_size, g->activate_ctx);
}

extern "C" const char* gp_mem_backend_type_to_string(GP_MemBackendType t) {
    switch (t) {
        case GP_MEM_BACKEND_CPU: return "cpu";
        case GP_MEM_BACKEND_TORCH: return "torch";
        case GP_MEM_BACKEND_CUDA: return "cuda";
        case GP_MEM_BACKEND_VULKAN: return "vulkan";
        case GP_MEM_BACKEND_OPENGL: return "opengl";
        case GP_MEM_BACKEND_SHARED_MEM: return "shared_mem";
        case GP_MEM_BACKEND_NETWORK: return "network";
        case GP_MEM_BACKEND_PYTHON: return "python";
        default: return "custom";
    }
}

// (mem backend prototype compiled unconditionally)
