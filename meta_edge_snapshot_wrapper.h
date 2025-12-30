#pragma once
#include <cstddef>
#include <cstdint>

struct GP_TableContext;

extern "C" {
// Allocates and captures a snapshot for the given edge. On success returns 1
// and sets *out_buf to a malloc'ed buffer of size *out_len and *out_written
// to the number of bytes written. Caller must free via gp_meta_edge_snapshot_free.
int32_t gp_meta_edge_snapshot_capture(GP_TableContext* ctx, int32_t edge_idx, void** out_buf, size_t* out_len, size_t* out_written);

// Restore a snapshot into the given edge. Returns 1 on success.
int32_t gp_meta_edge_snapshot_restore(GP_TableContext* ctx, int32_t edge_idx, const void* buf, size_t buf_len);

// Free a buffer returned by gp_meta_edge_snapshot_capture.
int32_t gp_meta_edge_snapshot_free(void* buf);
// Capture snapshot and write chunk files into `dir`. Returns a small manifest
// blob (malloc'ed) describing metadata and chunk filenames; free with
// `gp_meta_edge_snapshot_free`.
// Deprecated: capture to a directory and return manifest blob.
int32_t gp_meta_edge_snapshot_capture_to_dir(GP_TableContext* ctx, int32_t edge_idx, const char* dir, size_t chunk_size, char** out_manifest, size_t* out_manifest_size);
// New: capture chunks and publish their backend buffer handles into the
// specified `target_edge_idx` FIFO using `writer_key`. Returns 1 on success.
int32_t gp_meta_edge_snapshot_capture_to_edge(GP_TableContext* ctx, int32_t src_edge_idx, int32_t target_edge_idx, unsigned long long writer_key, size_t chunk_size);
// Restore a snapshot from a capture dir using the manifest returned by
// `gp_meta_edge_snapshot_capture_to_dir`. If `manifest` is NULL, the function
// will attempt to read a `manifest.txt` file from `dir`.
int32_t gp_meta_edge_snapshot_restore_from_dir(GP_TableContext* ctx, int32_t edge_idx, const char* dir, const char* manifest);
}
