// artifact_importer.h
//
// Total programmatic import, first seam: ingest a compiled artifact produced
// by turing's translation suite (C / LLVM / Fortran section DLLs) and
// actualize the C++ wiring nodus needs to launch it as a tool.
//
// The importer consumes the artifact's machine-readable calling contract --
// the `turing-compiled-program-api-v1` YAML descriptor emitted beside every
// artifact by turing's compiled_program_api.py (generated from the same
// Function objects codegen used, so it states c_type/passing/role/shape
// authoritatively; parsing emitted source is never necessary or wanted).
//
// From (artifact.dll, artifact.api.yaml) it emits, under output_root:
//   - source/tools/imported/<tool>.cpp   -- a generated ITool wrapper that
//     pops VT_ABSTRACT_TENSOR handles for the entry point's input
//     parameters, maps payloads through the nodus_tensor_* transport,
//     resolves extent parameters from the arrays that name them, calls the
//     artifact's entry symbol with its exact declared signature, and pushes
//     the output tensor's handle;
//   - <tool>.nodus_package.txt           -- a NODUSPKG V1 manifest whose
//     TOOL entry names the wrapper source, ready for
//     gp_repo_package_ingest_from_file, with the artifact's language as its
//     capability tag (backend-group labeling for imported operator sets).
//
// v1 honest boundaries (each refused with a reported shortfall, never
// silently narrowed): float64 tensors only (the C tensor-op layer is
// double-only today), exactly one output parameter, static output shape or
// one inferable from a same-extent input, no card-program/arena artifacts
// (entry_points with parameters only).
#pragma once

#include <cstdint>

struct GP_ArtifactImportReport {
    int32_t entry_points_seen = 0;
    int32_t wrappers_written = 0;
    int32_t shortfalls = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

// Returns 1 on success (>=1 wrapper written), 0 on failure or if every entry
// point was a shortfall. out_package_path (optional, cap bytes) receives the
// generated NODUSPKG manifest path.
int gp_artifact_import(const char* api_yaml_path,
                       const char* artifact_path,
                       const char* output_root,
                       GP_ArtifactImportReport* report,
                       char* out_package_path,
                       int32_t out_package_cap);

// Import a C/LLVM artifact exposing the uniform buffer entry ABI
//
//     C:    void entry(void** buffers, int64_t* extents)
//     LLVM: void entry(void** buffers, int32_t* extents)
//
// described by a strict, line-oriented descriptor:
//
//     NODUS BUFFER ABI V1
//     BACKEND "c"                         # "c" or "llvm"
//     EXTENT_DTYPE "i64"                  # C: i64, LLVM: i32
//     TOOL "compiled.example.c" "Example"
//     ENTRY "example_entry"
//     VALUE INPUT  "ssa.input"  "f64" 1 8
//     VALUE OUTPUT "ssa.output" "f64" 1 8
//     VALUE WORKSPACE "ssa.tmp"  "f64" 1 8
//     BUFFER "ssa.input"
//     BUFFER "ssa.tmp"
//     BUFFER "ssa.output"
//     EXTENT "ssa.input" 0
//
// VALUE order is the row stack order. BUFFER and EXTENT records are passed
// to the backend in their exact declared order. EXTENT_DTYPE is mandatory
// and is checked against BACKEND rather than conflating C's i64 and LLVM's
// i32 contracts.
// Shapes must be fixed and
// ranks must fit NODUS_TENSOR_MAX_RANK. The generated source is written
// directly under output_root/source/tools with a backend-qualified filename,
// ready for the existing source+same-stem-binary row-tool staging path.
// Multiple outputs are supported and pushed in VALUE declaration order.
// WORKSPACE tensors are zero-initialized, passed only through BUFFER order,
// and destroyed after the call.
//
// Returns 1 on success and writes both a macro-composable ITool source and a
// NODUSPKG V1 manifest. Optional output buffers receive their paths.
int gp_artifact_import_buffer_abi(const char* descriptor_path,
                                  const char* artifact_path,
                                  const char* output_root,
                                  GP_ArtifactImportReport* report,
                                  char* out_wrapper_path,
                                  int32_t out_wrapper_cap,
                                  char* out_package_path,
                                  int32_t out_package_cap);

#ifdef __cplusplus
}
#endif
