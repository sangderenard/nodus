#include "meta_edge_snapshot_wrapper.h"
#include <cstdlib>
#include <cstring>
#include "table_abi.h"
#include "mem_backend.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

// Forward-declare the underlying snapshot functions implemented in table_abi_edge_io.inl
extern "C" int32_t gp_table_edge_serialize_snapshot(GP_TableContext* ctx, int32_t edge_idx, void* out_buf, size_t out_len, size_t* out_written);
extern "C" int32_t gp_table_edge_deserialize_snapshot(GP_TableContext* ctx, int32_t edge_idx, const void* buf, size_t buf_len);

extern "C" int32_t gp_meta_edge_snapshot_capture(GP_TableContext* ctx, int32_t edge_idx, void** out_buf, size_t* out_len, size_t* out_written) {
    if (!ctx || !out_buf || !out_len || !out_written) return 0;
    if (edge_idx < 0) return 0;

    // Try exponential allocation until serialize succeeds. Start modest and grow.
    size_t attempt = 4096;
    const size_t max_attempt = 1u << 24; // 16MiB cap to avoid runaway
    void* buf = nullptr;
    size_t written = 0;
    int32_t ok = 0;
    while (attempt <= max_attempt) {
        void* nb = std::realloc(buf, attempt);
        if (!nb) { std::free(buf); return 0; }
        buf = nb;
        written = 0;
        ok = gp_table_edge_serialize_snapshot(ctx, edge_idx, buf, attempt, &written);
        if (ok) break;
        attempt *= 2;
    }
    if (!ok) { if (buf) std::free(buf); return 0; }
    *out_buf = buf;
    *out_len = attempt;
    *out_written = written;
    return 1;
}

extern "C" int32_t gp_meta_edge_snapshot_restore(GP_TableContext* ctx, int32_t edge_idx, const void* buf, size_t buf_len) {
    if (!ctx || !buf) return 0;
    return gp_table_edge_deserialize_snapshot(ctx, edge_idx, buf, buf_len);
}

extern "C" int32_t gp_meta_edge_snapshot_free(void* buf) {
    if (!buf) return 0;
    std::free(buf);
    return 1;
}

// New chunked capture -> write files into dir and return a small manifest
extern "C" int32_t gp_meta_edge_snapshot_capture_to_dir(GP_TableContext* ctx, int32_t edge_idx, const char* dir, size_t chunk_size, char** out_manifest, size_t* out_manifest_size) {
    if (!ctx || !dir || !out_manifest || !out_manifest_size) return 0;
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::create_directories(dir, ec) && ec) return 0;

    struct Ctx { std::string dir; size_t idx; std::vector<std::string> parts; std::vector<size_t> sizes; std::vector<uint8_t> meta; bool meta_stored; gp_mem_backend_handle_t fs_backend; const gp_mem_backend_vtable_t* fs_vt; } cctx;
    cctx.dir = dir; cctx.idx = 0; cctx.meta_stored = false; cctx.fs_backend = nullptr; cctx.fs_vt = nullptr;

    // Create a filesystem backend to allocate file-backed buffers for chunks
    gp_mem_backend_handle_t fsb = gp_mem_backend_create_filesystem(dir, 0);
    if (!fsb) return 0;
    const gp_mem_backend_vtable_t* fsvt = gp_mem_backend_get_vtable(fsb);
    if (!fsvt) return 0;
    cctx.fs_backend = fsb; cctx.fs_vt = fsvt;

    // writer: allocate a file buffer via backend and copy chunk into it
    int (*writer_fn_ptr)(void*, const void*, size_t, int) = [](void* user, const void* data, size_t data_len, int is_final) -> int {
        Ctx* ctx = reinterpret_cast<Ctx*>(user);
        if (!ctx->meta_stored) {
            try { ctx->meta.assign(reinterpret_cast<const uint8_t*>(data), reinterpret_cast<const uint8_t*>(data) + data_len); } catch (...) { return 0; }
            ctx->meta_stored = true;
            return 1;
        }
        // allocate a backend buffer for this chunk
        if (!ctx->fs_vt || !ctx->fs_vt->alloc || !ctx->fs_vt->copy_to_backend) return 0;
        gp_mem_backend_handle_t buf = ctx->fs_vt->alloc(ctx->fs_backend, data_len, 1);
        if (!buf) return 0;
        if (!ctx->fs_vt->copy_to_backend(buf, 0, data, data_len)) return 0;
        // query native handle (expected to be a path C string for filesystem backend)
        uintptr_t nh = gp_mem_backend_native_handle(buf);
        const char* path = reinterpret_cast<const char*>(nh);
        std::string fname = std::filesystem::path(path).filename().string();
        ctx->parts.push_back(fname);
        ctx->sizes.push_back(data_len);
        ctx->idx++;
        return 1;
    };

        // call chunked serializer from table ABI (writer signature: void*(user,const void*,size_t,int))
        extern int32_t gp_table_edge_serialize_snapshot_chunked(GP_TableContext*, int32_t, size_t, int (*)(void*, const void*, size_t, int), void*);
        int32_t ok = gp_table_edge_serialize_snapshot_chunked(ctx, edge_idx, chunk_size, writer_fn_ptr, &cctx);
        if (!ok) return 0;

        // Deprecated behavior: if caller wanted an on-disk manifest we still
        // support it by creating a metadata backend buffer; otherwise the caller
        // should use `gp_meta_edge_snapshot_capture_to_edge` to publish chunks
        // into a FIFO (see below).
        std::ostringstream manifest;
        manifest << "METADATA:backend_meta:" << cctx.meta.size() << "\n";
        for (size_t i = 0; i < cctx.parts.size(); ++i) {
            manifest << "CHUNK:" << cctx.parts[i] << ":" << cctx.sizes[i] << "\n";
        }
        std::string mstr = manifest.str();
        char* out = static_cast<char*>(std::malloc(mstr.size() + 1));
        if (!out) return 0;
        std::memcpy(out, mstr.data(), mstr.size()); out[mstr.size()] = '\0';
        *out_manifest = out;
        *out_manifest_size = mstr.size();
        return 1;
}

    // Capture to a target edge: allocate filesystem-backed buffers for each
    // chunk and publish the buffer handle into `target_edge_idx` FIFO as a
    // pointer. Caller is responsible for consuming/releasing buffers later.
    extern "C" int32_t gp_meta_edge_snapshot_capture_to_edge(GP_TableContext* ctx, int32_t src_edge_idx, int32_t target_edge_idx, unsigned long long writer_key, size_t chunk_size) {
        if (!ctx) return 0;
        // Create a filesystem backend context rooted at a per-table directory
        // (use tmp path). Caller may instead create a shared fs backend and
        // publish its handle via other means; keep this simple for now.
        std::string dir = "./.fs_snapshot";
        gp_mem_backend_handle_t fsb = gp_mem_backend_create_filesystem(dir.c_str(), 0);
        if (!fsb) return 0;
        const gp_mem_backend_vtable_t* fsvt = gp_mem_backend_get_vtable(fsb);
        if (!fsvt) return 0;

        struct EdgeCtx { gp_mem_backend_handle_t fsb; const gp_mem_backend_vtable_t* fsvt; GP_TableContext* ctx; int32_t target_edge; unsigned long long writer_key; } ectx;
        ectx.fsb = fsb; ectx.fsvt = fsvt; ectx.ctx = ctx; ectx.target_edge = target_edge_idx; ectx.writer_key = writer_key;

        // writer: allocate backend buffer and copy chunk into it, then publish pointer
        int (*writer_fn_ptr)(void*, const void*, size_t, int) = [](void* user, const void* data, size_t data_len, int is_final) -> int {
            EdgeCtx* ec = reinterpret_cast<EdgeCtx*>(user);
            if (!ec->fsvt || !ec->fsvt->alloc || !ec->fsvt->copy_to_backend) return 0;
            gp_mem_backend_handle_t buf = ec->fsvt->alloc(ec->fsb, data_len, 1);
            if (!buf) return 0;
            if (!ec->fsvt->copy_to_backend(buf, 0, data, data_len)) return 0;
            // publish the backend handle as an opaque pointer into the target FIFO
            int32_t dropped = 0;
            if (!gp_table_edge_publish_ptr(ec->ctx, ec->target_edge, ec->writer_key, reinterpret_cast<void*>(buf), &dropped)) return 0;
            return 1;
        };

        extern int32_t gp_table_edge_serialize_snapshot_chunked(GP_TableContext*, int32_t, size_t, int (*)(void*, const void*, size_t, int), void*);
        int32_t ok = gp_table_edge_serialize_snapshot_chunked(ctx, src_edge_idx, chunk_size, writer_fn_ptr, &ectx);
        return ok;
    }

// Restore from dir + manifest
extern "C" int32_t gp_meta_edge_snapshot_restore_from_dir(GP_TableContext* ctx, int32_t edge_idx, const char* dir, const char* manifest) {
    if (!ctx || !dir) return 0;
    std::string manifest_content;
    if (manifest) manifest_content = manifest;
    else {
        std::string mpath = std::string(dir) + "/manifest.txt";
        std::ifstream mfile(mpath);
        if (!mfile) return 0;
        std::ostringstream ss; ss << mfile.rdbuf(); manifest_content = ss.str();
    }
    if (manifest_content.empty()) return 0;

    // Parse manifest: lines of METADATA:filename:size and CHUNK:filename:size
    std::istringstream in(manifest_content);
    std::string line;
    std::string meta_fname;
    size_t meta_size = 0;
    std::vector<std::string> chunks;
    std::vector<size_t> chunk_sizes;
    while (std::getline(in, line)) {
        if (line.rfind("METADATA:", 0) == 0) {
            // METADATA:metadata.bin:123
            size_t p1 = line.find(':', 9);
            if (p1 == std::string::npos) continue;
            meta_fname = line.substr(9, p1 - 9);
            meta_size = static_cast<size_t>(std::stoull(line.substr(p1 + 1)));
        } else if (line.rfind("CHUNK:", 0) == 0) {
            size_t p1 = line.find(':', 6);
            if (p1 == std::string::npos) continue;
            std::string cf = line.substr(6, p1 - 6);
            size_t cs = static_cast<size_t>(std::stoull(line.substr(p1 + 1)));
            chunks.push_back(cf);
            chunk_sizes.push_back(cs);
        }
    }
    if (meta_fname.empty()) return 0;

    // Read metadata
    std::string meta_path = std::string(dir) + "/" + meta_fname;
    std::ifstream mf(meta_path, std::ios::binary);
    if (!mf) return 0;
    std::vector<uint8_t> meta_buf((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());

    // compute total size and assemble contiguous buffer
    size_t total = meta_buf.size();
    for (size_t s : chunk_sizes) total += s;
    std::vector<uint8_t> all;
    try { all.reserve(total); } catch (...) { return 0; }
    all.insert(all.end(), meta_buf.begin(), meta_buf.end());
    for (size_t i = 0; i < chunks.size(); ++i) {
        std::string cp = std::string(dir) + "/" + chunks[i];
        std::ifstream cf(cp, std::ios::binary);
        if (!cf) return 0;
        std::vector<uint8_t> cb((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        if (cb.size() != chunk_sizes[i]) return 0;
        all.insert(all.end(), cb.begin(), cb.end());
    }

    // call deserialize
    int32_t ok = gp_table_edge_deserialize_snapshot(ctx, edge_idx, all.data(), all.size());
    return ok;
}
