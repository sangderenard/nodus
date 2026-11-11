#pragma once

#include <cstdint>
#ifdef __cplusplus
#include <atomic>
#endif

// Nodus-side adoption of a complete product emitted by
// tools/turing_row_tool_compiler.py. This layer does not compile, link, or
// reinterpret SSA itself: it launches the compiler driver when requested,
// consumes its final receipt, delegates wrapper generation to the existing
// artifact importers, and delegates build/load to the existing plugin path.
struct GP_TuringProductAdoptionReport {
    int32_t receipt_validated = 0;
    int32_t wrapper_imported = 0;
    int32_t wrapper_staged = 0;
    int32_t plugin_loaded = 0;
};

// Cancel handle for the compiler-driver wait. Set `requested` from any thread;
// the waiting thread kills the child (and its job tree on Windows) and
// gp_turing_row_tool_compile_only returns 2. nullptr keeps the INFINITE wait.
#ifdef __cplusplus
struct GP_TuringCompileCancel {
    std::atomic<bool> requested{false};
};
#else
typedef struct GP_TuringCompileCancel GP_TuringCompileCancel;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Presence of source-only request fields forbids the existing-row receipt shortcut.
int gp_turing_row_tool_request_has_source_mode(const char* request_json);

// First half of compile_and_load: validate, run the driver, require the
// receipt. Touches no registry/canvas, so it is safe on a worker thread.
// Returns 1 adoptable product, 0 failure, 2 cancelled, 3 source-only emission.
// Source-only emission must never be adopted or loaded. out_error may be null.
int gp_turing_row_tool_compile_only(
    const char* python_executable,
    const char* nodus_root,
    const char* request_json,
    const char* turing_root,
    const char* product_dir,
    GP_TuringCompileCancel* cancel,
    char* out_error,
    int32_t out_error_cap);

// Adopt an already-compiled product. All receipt paths are product-root
// relative and are refused if missing or if they escape receipt.parent.
int gp_turing_product_adopt_and_load(
    const char* receipt_path,
    const char* nodus_root,
    const char* module_library_root,
    GP_TuringProductAdoptionReport* report,
    char* out_tool_id,
    int32_t out_tool_id_cap,
    char* out_staged_source,
    int32_t out_staged_source_cap);

// Launch the Nodus-owned compiler driver with an explicit Python executable,
// request, Turing root, and product directory; then adopt its completed
// `nodus-row-tool-product.receipt`. No shell is used to launch the process.
int gp_turing_row_tool_compile_and_load(
    const char* python_executable,
    const char* request_json,
    const char* turing_root,
    const char* product_dir,
    const char* nodus_root,
    const char* module_library_root,
    GP_TuringProductAdoptionReport* report,
    char* out_tool_id,
    int32_t out_tool_id_cap,
    char* out_staged_source,
    int32_t out_staged_source_cap);

#ifdef __cplusplus
}
#endif
