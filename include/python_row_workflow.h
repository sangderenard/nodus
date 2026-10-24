#pragma once

#include <cstdint>

// One retained Python source/ABI can be deployed either as a live CPython
// row or translated by Turing to a native backend.  Deployment selection is
// explicit and a failed native translation never falls back to CPython.
struct GP_PythonRowDeploymentReport {
    int32_t abi_loaded = 0;
    int32_t source_retained = 0;
    int32_t request_written = 0;
    int32_t compiler_invoked = 0;
    int32_t wrapper_staged = 0;
    int32_t plugin_loaded = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

// target is exactly one of: "cpython", "c", "llvm", "fortran".
//
// The descriptor is a NODUSPYROW V1 file.  For cpython, compiler-specific
// arguments may be null.  Native targets require python_executable,
// turing_root, extraction_contract, and product_dir.  nodus_root and
// module_library_root are always required.
int gp_python_row_deploy_and_load(
    const char* descriptor_path,
    const char* target,
    const char* python_executable,
    const char* turing_root,
    const char* extraction_contract,
    const char* product_dir,
    const char* nodus_root,
    const char* module_library_root,
    GP_PythonRowDeploymentReport* report,
    char* out_tool_id,
    int32_t out_tool_id_cap,
    char* out_retained_python,
    int32_t out_retained_python_cap,
    char* out_staged_wrapper,
    int32_t out_staged_wrapper_cap);

#ifdef __cplusplus
}
#endif
