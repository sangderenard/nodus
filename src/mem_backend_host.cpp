// Dispatch/execute a kernel described by a KernelIR (SPIR-V operator set)
static int cpu_dispatch_kernel(gp_mem_backend_handle_t backend_h, const nodus::spirv::KernelIR* kernel_ir) {
    if (!backend_h || !kernel_ir) return 0;
    // Translate the kernel IR to SPIR-V (and optionally execute or simulate)
    using namespace nodus::spirv;
    SpirvCompileOptions opts;
    SpirvTranslator translator(opts);
    auto result = translator.translate_kernel_to_spirv(*kernel_ir);
    // For now, just log the translation and return success (stub)
    std::fprintf(stderr, "[mem_backend] cpu_dispatch_kernel: translated kernel '%s' to SPIR-V (%zu words)\n", kernel_ir->name.c_str(), result.spirv.words.size());
    // TODO: Actually execute or simulate the kernel on the CPU backend
    return 1;
}
// mem_backend_host.cpp -- host/CPU MemoryBackend adapted to new vtable
#include "mem_backend.h"
#include <cstdlib>
#include <cstring>
#include "spirv_translation.h"
#include <mutex>
#include "kernel_isa.h" // Integrate KernelISA for KernelIR usage


#include <atomic>
#include <thread>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <filesystem>
#include <cstdio>
#include <sstream>
#include <string>
#include <fstream>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

struct CpuBackend {
    std::mutex mu;
    void* ptr;
    size_t size;
    std::atomic<int> refs;
    void* params_blob;
    size_t params_size;
    gp_mem_backend_activate_fn activate_fn;
    void* activate_ctx;
    uint32_t cpu_thread_ceiling;
    size_t max_alloc_bytes;
    uint32_t compute_deploy_limit;
    uint32_t alignment;
    const gp_mem_backend_vtable_t* vtbl;
    CpuBackend(size_t n): ptr(nullptr), size(0), refs(1), params_blob(nullptr), params_size(0), activate_fn(nullptr), activate_ctx(nullptr),
        cpu_thread_ceiling(0), max_alloc_bytes(0), compute_deploy_limit(0), alignment(alignof(void*)), vtbl(nullptr) {
        if (n > 0) {
            ptr = std::malloc(n);
            if (ptr) std::memset(ptr, 0, n);

// Example stub: Integrate SPIR-V translation into the memory backend
// (Replace or extend this with actual logic as needed)
void gp_mem_backend_translate_to_spirv(const nodus::spirv::KernelIR& ir) {
    using namespace nodus::spirv;
    SpirvCompileOptions opts;
    SpirvTranslator translator(opts);
    auto result = translator.translate_kernel_to_spirv(ir);
    // TODO: Store/use result.spirv as needed
}
            size = ptr ? n : 0;
        }
        unsigned hc = std::thread::hardware_concurrency();
        cpu_thread_ceiling = hc ? std::min<unsigned>(hc, 8u) : 4u;
        max_alloc_bytes = 1024ull * 1024ull * 1024ull;
        compute_deploy_limit = 1;
    }
    ~CpuBackend() {
        if (ptr) std::free(ptr);
        ptr = nullptr; size = 0;
        if (params_blob) std::free(params_blob);
        params_blob = nullptr; params_size = 0;
    }
};

// forward declaration of vtable defined later in this TU
extern const gp_mem_backend_vtable_t g_cpu_vtable;

// Helpers and registries reused from prior implementation
static std::mutex g_gl_ctx_mu;
static std::unordered_map<uintptr_t, uintptr_t> g_gl_ctx_registry;
static std::mutex g_vt_mu;
static std::unordered_map<gp_mem_backend_handle_t, const gp_mem_backend_vtable_t*> g_handle_vtables;
static std::unordered_map<gp_mem_backend_handle_t, int> g_handle_types;
static std::unordered_set<gp_mem_backend_handle_t> g_backend_contexts;

// map/unmap
static void* cpu_map(gp_mem_backend_handle_t h) {
    if (!h) return nullptr;
    CpuBackend* b = reinterpret_cast<CpuBackend*>(h);
    std::lock_guard<std::mutex> lk(b->mu);
    return b->ptr;
}
static void cpu_unmap(gp_mem_backend_handle_t h) { (void)h; }

// alloc/resize/free operate on buffer handles; for host backend we represent
// buffers as independent CpuBackend instances allocated via alloc().
static gp_mem_backend_handle_t cpu_alloc(gp_mem_backend_handle_t backend_h, size_t bytes, uint32_t alignment) {
    (void)backend_h; (void)alignment;
    CpuBackend* b = new (std::nothrow) CpuBackend(bytes);
    if (!b) return nullptr;
    b->vtbl = &g_cpu_vtable;
    std::lock_guard<std::mutex> lk(g_vt_mu);
    g_handle_vtables[reinterpret_cast<gp_mem_backend_handle_t>(b)] = b->vtbl;
    g_handle_types[reinterpret_cast<gp_mem_backend_handle_t>(b)] = (int)GP_MEM_BACKEND_CPU;
    return reinterpret_cast<gp_mem_backend_handle_t>(b);
}

static int cpu_resize(gp_mem_backend_handle_t buffer_h, size_t bytes) {
    if (!buffer_h) return 0;
    CpuBackend* b = reinterpret_cast<CpuBackend*>(buffer_h);
    std::lock_guard<std::mutex> lk(b->mu);
    if (bytes == b->size) return 1;
    void* p = nullptr;
    if (bytes > 0) {
        p = std::realloc(b->ptr, bytes);
        if (!p) return 0;
        if (bytes > b->size && p) std::memset(reinterpret_cast<char*>(p) + b->size, 0, bytes - b->size);
    } else {
        if (b->ptr) { std::free(b->ptr); b->ptr = nullptr; }
        b->size = 0; return 1;
    }
    b->ptr = p; b->size = bytes; return 1;
}

static void cpu_free(gp_mem_backend_handle_t buffer_h) {
    if (!buffer_h) return;
    // reduce refcount and delete when no longer used
    CpuBackend* b = reinterpret_cast<CpuBackend*>(buffer_h);
    int prev = b->refs.fetch_sub(1);
    if (prev <= 1) {
        {
            std::lock_guard<std::mutex> lk(g_vt_mu);
            auto it = g_handle_vtables.find(buffer_h);
            if (it != g_handle_vtables.end()) g_handle_vtables.erase(it);
        }
        delete b;
    }
}

static int cpu_copy_to_backend(gp_mem_backend_handle_t h, size_t offset, const void* src, size_t bytes) {
    if (!h || !src) return 0;
    CpuBackend* b = reinterpret_cast<CpuBackend*>(h);
    std::lock_guard<std::mutex> lk(b->mu);
    if (offset + bytes > b->size) return 0;
    std::memcpy(reinterpret_cast<char*>(b->ptr) + offset, src, bytes);
    return 1;
}

static int cpu_copy_from_backend(gp_mem_backend_handle_t h, size_t offset, void* dst, size_t bytes) {
    if (!h || !dst) return 0;
    CpuBackend* b = reinterpret_cast<CpuBackend*>(h);
    std::lock_guard<std::mutex> lk(b->mu);
    if (offset + bytes > b->size) return 0;
    std::memcpy(dst, reinterpret_cast<char*>(b->ptr) + offset, bytes);
    return 1;
}

// Conservative device-to-device copy: staged via host memory for host backend
static int cpu_copy_between(gp_mem_backend_handle_t src, gp_mem_backend_handle_t dst, size_t src_offset, size_t dst_offset, size_t bytes) {
    if (!src || !dst) return 0;
    size_t max_chunk = 4 * 1024 * 1024;
    size_t remaining = bytes;
    std::vector<uint8_t> tmp;
    try { tmp.resize(std::min<size_t>(remaining, max_chunk)); } catch (...) { return 0; }
    size_t off = 0;
    while (remaining > 0) {
        size_t cur = std::min<size_t>(remaining, tmp.size());
        if (!cpu_copy_from_backend(src, src_offset + off, tmp.data(), cur)) return 0;
        if (!cpu_copy_to_backend(dst, dst_offset + off, tmp.data(), cur)) return 0;
        remaining -= cur; off += cur;
    }
    return 1;
}

static uintptr_t cpu_get_native_handle(gp_mem_backend_handle_t h) {
    if (!h) return 0;
    CpuBackend* b = reinterpret_cast<CpuBackend*>(h);
    std::lock_guard<std::mutex> lk(b->mu);
    return reinterpret_cast<uintptr_t>(b->ptr);
}

// Sync helpers are unsupported for host backend
static int cpu_record_event(gp_mem_backend_handle_t h, void* out_event, size_t out_event_size) { (void)h; (void)out_event; (void)out_event_size; return 0; }
static int cpu_wait_event(gp_mem_backend_handle_t h, const void* event, size_t event_size, uint64_t timeout_ms) { (void)h; (void)event; (void)event_size; (void)timeout_ms; return 0; }
static uintptr_t cpu_get_native_stream(gp_mem_backend_handle_t h) { (void)h; return 0; }

// Populate vtable (order must match mem_backend.h)
static const gp_mem_backend_vtable_t g_cpu_vtable = {
    &cpu_map,
    &cpu_unmap,
    &cpu_alloc,
    &cpu_resize,
    &cpu_free,
    &cpu_copy_to_backend,
    &cpu_copy_from_backend,
    &cpu_copy_between,
    &cpu_get_native_handle,
    &cpu_record_event,
    &cpu_wait_event,
    &cpu_get_native_stream,
    &cpu_dispatch_kernel
};

// Forward declarations for filesystem backend functions (defined later)
static void* fs_map(gp_mem_backend_handle_t h);
static void fs_unmap(gp_mem_backend_handle_t h);
static gp_mem_backend_handle_t fs_alloc(gp_mem_backend_handle_t backend_h, size_t bytes, uint32_t alignment);
static int fs_resize(gp_mem_backend_handle_t buffer_h, size_t bytes);
static void fs_free(gp_mem_backend_handle_t buffer_h);
static int fs_copy_to_backend(gp_mem_backend_handle_t h, size_t offset, const void* src, size_t bytes);
static int fs_copy_from_backend(gp_mem_backend_handle_t h, size_t offset, void* dst, size_t bytes);
static int fs_copy_between(gp_mem_backend_handle_t src, gp_mem_backend_handle_t dst, size_t src_offset, size_t dst_offset, size_t bytes);
static uintptr_t fs_get_native_handle(gp_mem_backend_handle_t h);

// Filesystem backend vtable
static const gp_mem_backend_vtable_t g_fs_vtable = {
    &fs_map,
    &fs_unmap,
    &fs_alloc,
    &fs_resize,
    &fs_free,
    &fs_copy_to_backend,
    &fs_copy_from_backend,
    &fs_copy_between,
    &fs_get_native_handle,
    nullptr,
    nullptr,
    nullptr
};

// Minimal GenericBackend used for other backends within this file
struct GenericBackend {
    std::mutex mu;
    GP_MemBackendType type;
    void* params_blob;
    size_t params_size;
    gp_mem_backend_activate_fn activate_fn;
    void* activate_ctx;
    uint64_t caps;
    uintptr_t native_handle;
    size_t size;
    std::atomic<int> refs;
    uint32_t cpu_thread_ceiling;
    size_t max_alloc_bytes;
    uint32_t compute_deploy_limit;
    uint32_t alignment;
    uintptr_t gl_context_ptr;
    GenericBackend(GP_MemBackendType t): params_blob(nullptr), params_size(0), activate_fn(nullptr), activate_ctx(nullptr), caps(0), native_handle(0), size(0), refs(1), cpu_thread_ceiling(0), max_alloc_bytes(0), compute_deploy_limit(0), alignment(1), type(t) {}
    ~GenericBackend() { if (params_blob) std::free(params_blob); params_blob = nullptr; }
};

static void* generic_map(gp_mem_backend_handle_t h) {
    if (!h) return nullptr;
    GenericBackend* g = reinterpret_cast<GenericBackend*>(h);
    std::lock_guard<std::mutex> lk(g->mu);
    if (g->native_handle == 0) return nullptr;
    return reinterpret_cast<void*>(g->native_handle);
}
static void generic_unmap(gp_mem_backend_handle_t h) { (void)h; }
static int generic_copy_to_backend(gp_mem_backend_handle_t h, size_t offset, const void* src, size_t bytes) {
    if (!h || !src) return 0;
    GenericBackend* g = reinterpret_cast<GenericBackend*>(h);
    std::lock_guard<std::mutex> lk(g->mu);
    if (g->native_handle == 0) return 0;
    void* dst = reinterpret_cast<void*>(g->native_handle);
    std::memcpy(reinterpret_cast<uint8_t*>(dst) + offset, src, bytes);
    return 1;
}
static int generic_copy_from_backend(gp_mem_backend_handle_t h, size_t offset, void* dst, size_t bytes) {
    if (!h || !dst) return 0;
    GenericBackend* g = reinterpret_cast<GenericBackend*>(h);
    std::lock_guard<std::mutex> lk(g->mu);
    if (g->native_handle == 0) return 0;
    void* src = reinterpret_cast<void*>(g->native_handle);
    std::memcpy(dst, reinterpret_cast<uint8_t*>(src) + offset, bytes);
    return 1;
}

static int generic_copy_between(gp_mem_backend_handle_t src, gp_mem_backend_handle_t dst, size_t src_offset, size_t dst_offset, size_t bytes) {
    if (!src || !dst) return 0;
    size_t max_chunk = 4 * 1024 * 1024;
    size_t remaining = bytes;
    std::vector<uint8_t> tmp;
    try { tmp.resize(std::min<size_t>(remaining, max_chunk)); } catch (...) { return 0; }
    size_t off = 0;
    while (remaining > 0) {
        size_t cur = std::min<size_t>(remaining, tmp.size());
        if (!generic_copy_from_backend(src, src_offset + off, tmp.data(), cur)) return 0;
        if (!generic_copy_to_backend(dst, dst_offset + off, tmp.data(), cur)) return 0;
        remaining -= cur; off += cur;
    }
    return 1;
}

static const gp_mem_backend_vtable_t g_generic_vtable = { &generic_map, &generic_unmap, nullptr, nullptr, nullptr, &generic_copy_to_backend, &generic_copy_from_backend, &generic_copy_between, nullptr, nullptr, nullptr, nullptr };

extern "C" int gp_mem_backend_register_gl_context(uintptr_t ctx_id, uintptr_t native_ctx_ptr) {
    std::lock_guard<std::mutex> lk(g_gl_ctx_mu);
    g_gl_ctx_registry[ctx_id] = native_ctx_ptr;
    return 1;
}

extern "C" int gp_mem_backend_unregister_gl_context(uintptr_t ctx_id) {
    std::lock_guard<std::mutex> lk(g_gl_ctx_mu);
    auto it = g_gl_ctx_registry.find(ctx_id);
    if (it == g_gl_ctx_registry.end()) return 0;
    g_gl_ctx_registry.erase(it);
    return 1;
}

extern "C" uintptr_t gp_mem_backend_get_registered_gl_context(uintptr_t ctx_id) {
    std::lock_guard<std::mutex> lk(g_gl_ctx_mu);
    auto it = g_gl_ctx_registry.find(ctx_id);
    if (it == g_gl_ctx_registry.end()) return 0;
    return it->second;
}

extern "C" gp_mem_backend_handle_t gp_mem_backend_create_cpu(size_t initial_bytes) {
    CpuBackend* b = new (std::nothrow) CpuBackend(initial_bytes);
    if (b) {
        b->vtbl = &g_cpu_vtable;
        std::lock_guard<std::mutex> lk(g_vt_mu);
        g_handle_vtables[reinterpret_cast<gp_mem_backend_handle_t>(b)] = b->vtbl;
        g_handle_types[reinterpret_cast<gp_mem_backend_handle_t>(b)] = (int)GP_MEM_BACKEND_CPU;
    }
    return reinterpret_cast<gp_mem_backend_handle_t>(b);
}

extern "C" gp_mem_backend_handle_t gp_mem_backend_create_host(size_t initial_bytes) {
    return gp_mem_backend_create_cpu(initial_bytes);
}

// Filesystem-backed buffer implementation.
struct FileBuffer {
    std::mutex mu;
    GP_MemBackendType type;
    void* params_blob;
    size_t params_size;
    gp_mem_backend_activate_fn activate_fn;
    void* activate_ctx;
    uint64_t caps;
    uintptr_t native_handle;
    size_t size;
    std::atomic<int> refs;
    uint32_t cpu_thread_ceiling;
    size_t max_alloc_bytes;
    uint32_t compute_deploy_limit;
    uint32_t alignment;
    uintptr_t gl_context_ptr;

    // file-specific
    std::string path;
    FILE* f;
    char* native_path_cstr;
    FileBuffer(): params_blob(nullptr), params_size(0), activate_fn(nullptr), activate_ctx(nullptr), caps(0), native_handle(0), size(0), refs(1), cpu_thread_ceiling(0), max_alloc_bytes(0), compute_deploy_limit(0), alignment(1), gl_context_ptr(0), f(nullptr), native_path_cstr(nullptr) {}
    ~FileBuffer() { if (f) fclose(f); if (native_path_cstr) std::free(native_path_cstr); if (params_blob) std::free(params_blob); }
};

struct FsBackendCtx {
    std::mutex mu;
    std::string base_dir;
    std::atomic<uint64_t> counter;
    FsBackendCtx(const std::string &d): base_dir(d), counter(0) {}
};

static void* fs_map(gp_mem_backend_handle_t h) { (void)h; return nullptr; }
static void fs_unmap(gp_mem_backend_handle_t h) { (void)h; }

static gp_mem_backend_handle_t fs_alloc(gp_mem_backend_handle_t backend_h, size_t bytes, uint32_t alignment) {
    FsBackendCtx* ctx = reinterpret_cast<FsBackendCtx*>(backend_h);
    if (!ctx) return nullptr;
    FileBuffer* b = new (std::nothrow) FileBuffer();
    if (!b) return nullptr;
    b->type = GP_MEM_BACKEND_SHARED_MEM;
    b->alignment = alignment ? alignment : 1;
    // generate unique filename
    uint64_t id = ctx->counter.fetch_add(1);
    std::ostringstream ss;
    ss << ctx->base_dir << "/fb_" << std::this_thread::get_id() << "_" << id << ".bin";
    b->path = ss.str();
    // open file
    b->f = std::fopen(b->path.c_str(), "w+b");
    if (!b->f) { delete b; return nullptr; }
    if (bytes > 0) {
        // write zeros to allocate
        const size_t chunk = 64 * 1024;
        std::vector<char> z(chunk);
        size_t remaining = bytes;
        while (remaining > 0) {
            size_t cur = std::min<size_t>(remaining, chunk);
            if (std::fwrite(z.data(), 1, cur, b->f) != cur) { std::fclose(b->f); delete b; return nullptr; }
            remaining -= cur;
        }
        std::fflush(b->f);
        b->size = bytes;
    }
    // register vtable for this buffer handle
    std::lock_guard<std::mutex> lk(g_vt_mu);
    g_handle_vtables[reinterpret_cast<gp_mem_backend_handle_t>(b)] = &g_fs_vtable;
    g_handle_types[reinterpret_cast<gp_mem_backend_handle_t>(b)] = (int)GP_MEM_BACKEND_SHARED_MEM;
    std::fprintf(stderr, "[mem_backend] fs_alloc: path=%s size=%zu handle=%p\n", b->path.c_str(), b->size, (void*)b);
    return reinterpret_cast<gp_mem_backend_handle_t>(b);
}

// Forward declarations for functions used in vtable
static int fs_copy_to_backend(gp_mem_backend_handle_t h, size_t offset, const void* src, size_t bytes);
static int fs_copy_from_backend(gp_mem_backend_handle_t h, size_t offset, void* dst, size_t bytes);
static int fs_copy_between(gp_mem_backend_handle_t src, gp_mem_backend_handle_t dst, size_t src_offset, size_t dst_offset, size_t bytes);
static uintptr_t fs_get_native_handle(gp_mem_backend_handle_t h);

static int fs_resize(gp_mem_backend_handle_t buffer_h, size_t bytes) {
    if (!buffer_h) return 0;
    FileBuffer* b = reinterpret_cast<FileBuffer*>(buffer_h);
    std::lock_guard<std::mutex> lk(b->mu);
    if (!b->f) return 0;
#ifdef _WIN32
    int fd = _fileno(b->f);
    if (bytes == 0) {
        _chsize_s(fd, 0);
        b->size = 0; return 1;
    }
    if (_chsize_s(fd, bytes) != 0) return 0;
    b->size = bytes; return 1;
#else
    int fd = fileno(b->f);
    if (ftruncate(fd, (off_t)bytes) != 0) return 0;
    b->size = bytes; return 1;
#endif
}

static void fs_free(gp_mem_backend_handle_t buffer_h) {
    if (!buffer_h) return;
    FileBuffer* b = reinterpret_cast<FileBuffer*>(buffer_h);
    int prev = b->refs.fetch_sub(1);
    std::fprintf(stderr, "[mem_backend] fs_free: handle=%p prev_refs=%d\n", (void*)b, prev);
    if (prev <= 1) {
        std::lock_guard<std::mutex> lk(g_vt_mu);
        auto it = g_handle_vtables.find(buffer_h);
        if (it != g_handle_vtables.end()) g_handle_vtables.erase(it);
        if (b->f) { std::fclose(b->f); b->f = nullptr; }
        if (!b->path.empty()) { /* keep files for persistence; do not unlink */ }
        if (b->native_path_cstr) std::free(b->native_path_cstr);
        std::fprintf(stderr, "[mem_backend] fs_free: deleting handle=%p\n", (void*)b);
        delete b;
    }
}

static int fs_copy_to_backend(gp_mem_backend_handle_t h, size_t offset, const void* src, size_t bytes) {
    if (!h || !src) return 0;
    FileBuffer* b = reinterpret_cast<FileBuffer*>(h);
    std::lock_guard<std::mutex> lk(b->mu);
    if (!b->f) return 0;
    if (std::fseek(b->f, static_cast<long>(offset), SEEK_SET) != 0) return 0;
    size_t written = std::fwrite(src, 1, bytes, b->f);
    if (written != bytes) return 0;
    std::fflush(b->f);
    if (offset + bytes > b->size) b->size = offset + bytes;
    return 1;
}

static int fs_copy_from_backend(gp_mem_backend_handle_t h, size_t offset, void* dst, size_t bytes) {
    if (!h || !dst) return 0;
    FileBuffer* b = reinterpret_cast<FileBuffer*>(h);
    std::lock_guard<std::mutex> lk(b->mu);
    if (!b->f) return 0;
    if (offset + bytes > b->size) return 0;
    if (std::fseek(b->f, static_cast<long>(offset), SEEK_SET) != 0) return 0;
    size_t r = std::fread(dst, 1, bytes, b->f);
    return r == bytes ? 1 : 0;
}

static int fs_copy_between(gp_mem_backend_handle_t src, gp_mem_backend_handle_t dst, size_t src_offset, size_t dst_offset, size_t bytes) {
    if (!src || !dst) return 0;
    const size_t max_chunk = 4 * 1024 * 1024;
    std::vector<uint8_t> tmp;
    try { tmp.resize(std::min<size_t>(bytes, max_chunk)); } catch(...) { return 0; }
    size_t remaining = bytes; size_t off = 0;
    while (remaining > 0) {
        size_t cur = std::min<size_t>(remaining, tmp.size());
        if (!fs_copy_from_backend(src, src_offset + off, tmp.data(), cur)) return 0;
        if (!fs_copy_to_backend(dst, dst_offset + off, tmp.data(), cur)) return 0;
        remaining -= cur; off += cur;
    }
    return 1;
}

static uintptr_t fs_get_native_handle(gp_mem_backend_handle_t h) {
    if (!h) return 0;
    FileBuffer* b = reinterpret_cast<FileBuffer*>(h);
    std::lock_guard<std::mutex> lk(b->mu);
    if (!b->native_path_cstr) {
#ifdef _WIN32
        b->native_path_cstr = _strdup(b->path.c_str());
#else
        b->native_path_cstr = strdup(b->path.c_str());
#endif
    }
    return reinterpret_cast<uintptr_t>(b->native_path_cstr);
}

// Public factory: create a filesystem backend context that can alloc file buffers
extern "C" gp_mem_backend_handle_t gp_mem_backend_create_filesystem(const char* base_dir, size_t initial_bytes) {
    if (!base_dir) return nullptr;
    FsBackendCtx* ctx = new (std::nothrow) FsBackendCtx(std::string(base_dir));
    if (!ctx) return nullptr;
    // Ensure directory exists
    std::error_code ec;
    std::filesystem::create_directories(ctx->base_dir, ec);
    // store ctx in vtable map with filesystem vtable so callers can alloc/copy
    std::lock_guard<std::mutex> lk(g_vt_mu);
    g_handle_vtables[reinterpret_cast<gp_mem_backend_handle_t>(ctx)] = &g_fs_vtable;
    g_handle_types[reinterpret_cast<gp_mem_backend_handle_t>(ctx)] = (int)GP_MEM_BACKEND_SHARED_MEM;
    g_backend_contexts.insert(reinterpret_cast<gp_mem_backend_handle_t>(ctx));
    std::fprintf(stderr, "[mem_backend] gp_mem_backend_create_filesystem: ctx=%p base_dir=%s\n", (void*)ctx, ctx->base_dir.c_str());
    (void)initial_bytes; // initial is applied per-allocated buffer
    return reinterpret_cast<gp_mem_backend_handle_t>(ctx);
}

extern "C" void gp_mem_backend_release(gp_mem_backend_handle_t h) {
    if (!h) return;
    {
        std::lock_guard<std::mutex> lk(g_vt_mu);
        auto it = g_handle_vtables.find(h);
        if (it != g_handle_vtables.end()) g_handle_vtables.erase(it);
    }
    // Determine handle type and perform type-correct release
    std::lock_guard<std::mutex> lk(g_vt_mu);
    auto it_type = g_handle_types.find(h);
    int t = (it_type != g_handle_types.end()) ? it_type->second : -1;
    if (it_type != g_handle_types.end()) g_handle_types.erase(it_type);
    // If this handle is a registered backend context, treat it accordingly
    if (g_backend_contexts.find(h) != g_backend_contexts.end()) {
        // filesystem backend context
        FsBackendCtx* ctx = reinterpret_cast<FsBackendCtx*>(h);
        std::fprintf(stderr, "[mem_backend] gp_mem_backend_release: deleting backend ctx=%p base_dir=%s\n", (void*)ctx, ctx->base_dir.c_str());
        g_backend_contexts.erase(h);
        delete ctx;
        return;
    }
    std::fprintf(stderr, "[mem_backend] gp_mem_backend_release: handle=%p type=%d\n", (void*)h, t);
    if (t == (int)GP_MEM_BACKEND_CPU) {
        CpuBackend* b = reinterpret_cast<CpuBackend*>(h);
        int prev = 0;
        try { prev = b->refs.fetch_sub(1); } catch (...) { return; }
        std::fprintf(stderr, "[mem_backend] gp_mem_backend_release: CPU handle=%p prev_refs=%d\n", (void*)b, prev);
        if (prev <= 1) { std::fprintf(stderr, "[mem_backend] gp_mem_backend_release: deleting CPU handle=%p\n", (void*)b); delete b; return; }
        return;
    }
    if (t == (int)GP_MEM_BACKEND_SHARED_MEM) {
        FileBuffer* fb = reinterpret_cast<FileBuffer*>(h);
        int prev = 0;
        try { prev = fb->refs.fetch_sub(1); } catch (...) { return; }
        std::fprintf(stderr, "[mem_backend] gp_mem_backend_release: FILE handle=%p prev_refs=%d path=%s\n", (void*)fb, prev, fb->path.c_str());
        if (prev <= 1) {
            std::fprintf(stderr, "[mem_backend] gp_mem_backend_release: deleting FILE handle=%p\n", (void*)fb);
            delete fb;
        }
        return;
    }
    // Fallback: generic backend
    GenericBackend* g = reinterpret_cast<GenericBackend*>(h);
    int gprev = 0;
    try { gprev = g->refs.fetch_sub(1); } catch (...) { return; }
    if (gprev <= 1) delete g;
}

extern "C" void* gp_mem_backend_map(gp_mem_backend_handle_t h) {
    return cpu_map(h);
}

extern "C" void gp_mem_backend_unmap(gp_mem_backend_handle_t h) { cpu_unmap(h); }

extern "C" int gp_mem_backend_resize(gp_mem_backend_handle_t h, size_t bytes) { return cpu_resize(h, bytes); }

extern "C" size_t gp_mem_backend_size(gp_mem_backend_handle_t h) {
    if (!h) return 0;
    std::lock_guard<std::mutex> lk(g_vt_mu);
    auto it_type = g_handle_types.find(h);
    int t = (it_type != g_handle_types.end()) ? it_type->second : -1;
    if (t == (int)GP_MEM_BACKEND_CPU) {
        CpuBackend* b = reinterpret_cast<CpuBackend*>(h);
        std::lock_guard<std::mutex> lk2(b->mu);
        return b->size;
    }
    if (t == (int)GP_MEM_BACKEND_SHARED_MEM) {
        FileBuffer* fb = reinterpret_cast<FileBuffer*>(h);
        std::lock_guard<std::mutex> lk2(fb->mu);
        return fb->size;
    }
    // fallback: try generic backend
    GenericBackend* gb = reinterpret_cast<GenericBackend*>(h);
    std::lock_guard<std::mutex> lk2(gb->mu);
    return gb->size;
}

extern "C" uint64_t gp_mem_backend_get_capabilities(gp_mem_backend_handle_t h) {
    if (!h) return 0;
    int t = gp_mem_backend_get_type(h);
    if (t == (int)GP_MEM_BACKEND_CPU) {
        uint64_t caps = 0;
        caps |= (uint64_t)GP_MEM_CAP_MAP_HOST;
        caps |= (uint64_t)GP_MEM_CAP_RESIZE;
        caps |= (uint64_t)GP_MEM_CAP_ZERO_COPY;
        caps |= (uint64_t)GP_MEM_CAP_BYREF_SAFE;
        return caps;
    }
    GenericBackend* gb = reinterpret_cast<GenericBackend*>(h);
    std::lock_guard<std::mutex> lk(gb->mu);
    return gb->caps;
}

extern "C" int gp_mem_backend_get_type(gp_mem_backend_handle_t h) {
    if (!h) return -1;
    std::lock_guard<std::mutex> lk(g_vt_mu);
    auto it = g_handle_types.find(h);
    if (it != g_handle_types.end()) return it->second;
    return -1;
}

extern "C" uintptr_t gp_mem_backend_native_handle(gp_mem_backend_handle_t h) {
    if (!h) return 0;
    const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(h);
    if (vt && vt->get_native_handle) return vt->get_native_handle(h);
    return 0;
}

// Return vtable pointer for a given handle if known.
extern "C" const gp_mem_backend_vtable_t* gp_mem_backend_get_vtable(gp_mem_backend_handle_t h) {
    if (!h) return nullptr;
    std::lock_guard<std::mutex> lk(g_vt_mu);
    auto it = g_handle_vtables.find(h);
    if (it != g_handle_vtables.end()) return it->second;
    return nullptr;
}

// Transfer helper - attempt vtable optimized copy first, otherwise staged copy
extern "C" int gp_mem_backend_transfer(gp_mem_backend_handle_t src, gp_mem_backend_handle_t dst, size_t src_offset, size_t dst_offset, size_t bytes) {
    if (!src && !dst) return 0;
    const gp_mem_backend_vtable_t* src_vt = gp_mem_backend_get_vtable(src);
    const gp_mem_backend_vtable_t* dst_vt = gp_mem_backend_get_vtable(dst);
    if (src_vt && dst_vt && src_vt->copy_between_backends) {
        if (src_vt->copy_between_backends(src, dst, src_offset, dst_offset, bytes)) return 1;
    }
    // fallback to staged copy
    size_t max_chunk = 4 * 1024 * 1024;
    size_t remaining = bytes;
    std::vector<uint8_t> tmp;
    try { tmp.resize(std::min<size_t>(remaining, max_chunk)); } catch (...) { return 0; }
    size_t off = 0;
    while (remaining > 0) {
        size_t cur = std::min<size_t>(remaining, tmp.size());
        if (src_vt && src_vt->copy_from_backend) {
            if (!src_vt->copy_from_backend(src, src_offset + off, tmp.data(), cur)) return 0;
        } else {
            void* m = gp_mem_backend_map_or_null(src);
            if (!m) return 0;
            std::memcpy(tmp.data(), reinterpret_cast<uint8_t*>(m) + src_offset + off, cur);
            gp_mem_backend_unmap(src);
        }
        if (dst_vt && dst_vt->copy_to_backend) {
            if (!dst_vt->copy_to_backend(dst, dst_offset + off, tmp.data(), cur)) return 0;
        } else {
            void* m = gp_mem_backend_map_or_null(dst);
            if (!m) return 0;
            std::memcpy(reinterpret_cast<uint8_t*>(m) + dst_offset + off, tmp.data(), cur);
            gp_mem_backend_unmap(dst);
        }
        remaining -= cur; off += cur;
    }
    return 1;
}

// Helper: import an existing file-backed buffer by path and register it
static gp_mem_backend_handle_t fs_import_buffer_from_path(const std::string &path) {
    FileBuffer* b = new (std::nothrow) FileBuffer();
    if (!b) return nullptr;
    b->path = path;
    // open existing file for read/write
    b->f = std::fopen(path.c_str(), "r+b");
    if (!b->f) {
        // try read-only open as a fallback
        b->f = std::fopen(path.c_str(), "rb");
        if (!b->f) { delete b; return nullptr; }
    }
    // determine size
    std::error_code ec;
    uint64_t sz = 0;
    try {
        sz = (uint64_t)std::filesystem::file_size(path, ec);
    } catch (...) { ec = std::make_error_code(std::errc::io_error); }
    if (ec) {
        if (b->f) std::fclose(b->f);
        delete b;
        return nullptr;
    }
    b->size = (size_t)sz;
    // set native path cstr for get_native_handle
#ifdef _WIN32
    b->native_path_cstr = _strdup(b->path.c_str());
#else
    b->native_path_cstr = strdup(b->path.c_str());
#endif
    // register vtable and type
    std::lock_guard<std::mutex> lk(g_vt_mu);
    g_handle_vtables[reinterpret_cast<gp_mem_backend_handle_t>(b)] = &g_fs_vtable;
    g_handle_types[reinterpret_cast<gp_mem_backend_handle_t>(b)] = (int)GP_MEM_BACKEND_SHARED_MEM;
    std::fprintf(stderr, "[mem_backend] fs_import_buffer_from_path: imported path=%s handle=%p size=%zu\n", b->path.c_str(), (void*)b, b->size);
    return reinterpret_cast<gp_mem_backend_handle_t>(b);
}

// Write a simple manifest (one entry per line): <path> <size>\n
extern "C" int gp_mem_backend_write_manifest(const char* manifest_path, gp_mem_backend_handle_t* chunk_handles, size_t count) {
    if (!manifest_path || (!chunk_handles && count != 0)) return 0;
    std::ofstream out(manifest_path, std::ios::binary);
    if (!out.is_open()) return 0;
    for (size_t i = 0; i < count; ++i) {
        gp_mem_backend_handle_t h = chunk_handles[i];
        const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(h);
        if (!vt || !vt->get_native_handle) { out.close(); return 0; }
        uintptr_t native = vt->get_native_handle(h);
        if (!native) { out.close(); return 0; }
        const char* path = reinterpret_cast<const char*>(native);
        size_t sz = gp_mem_backend_size(h);
        out << path << " " << sz << "\n";
        std::fprintf(stderr, "[mem_backend] gp_mem_backend_write_manifest: entry path=%s size=%zu handle=%p\n", path, sz, (void*)h);
        if (!out) { out.close(); return 0; }
    }
    out.close();
    return 1;
}

// Read manifest and import buffers. Allocates an array of handles via malloc
// and assigns to out_buffers; caller frees the array and releases handles.
extern "C" int gp_mem_backend_read_manifest_and_allocate(gp_mem_backend_handle_t backend_ctx, const char* manifest_path, gp_mem_backend_handle_t** out_buffers, size_t* out_count) {
    if (!manifest_path || !out_buffers || !out_count) return 0;
    std::ifstream in(manifest_path, std::ios::binary);
    if (!in.is_open()) return 0;
    std::vector<gp_mem_backend_handle_t> tmp;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        // parse: path and size (size is optional here; primary is path)
        std::istringstream ss(line);
        std::string path; uint64_t size = 0;
        ss >> path >> size;
        if (path.empty()) { in.close(); return 0; }
        gp_mem_backend_handle_t h = fs_import_buffer_from_path(path);
        if (!h) { in.close(); // cleanup already-created
            for (auto hh : tmp) gp_mem_backend_release(hh);
            return 0;
        }
        tmp.push_back(h);
    }
    in.close();
    size_t n = tmp.size();
    gp_mem_backend_handle_t* arr = (gp_mem_backend_handle_t*)std::malloc(sizeof(gp_mem_backend_handle_t) * n);
    if (!arr) {
        for (auto hh : tmp) gp_mem_backend_release(hh);
        return 0;
    }
    for (size_t i = 0; i < n; ++i) arr[i] = tmp[i];
    std::fprintf(stderr, "[mem_backend] gp_mem_backend_read_manifest_and_allocate: allocated arr=%p count=%zu\n", (void*)arr, n);
    *out_buffers = arr; *out_count = n;
    return 1;
}
