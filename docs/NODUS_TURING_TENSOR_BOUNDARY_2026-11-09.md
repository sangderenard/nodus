# Nodus / Turing tensor boundary qualification, 2026-11-09

## Finding

The requested boundary already exists. `tools/turing_row_tool_compiler.py` enters Turing through `lower_ast_source_to_ssa`, retains authored/submitted source and repository SSA, and publishes C, LLVM or Fortran products. `src/artifact_importer.cpp::generate_buffer_wrapper_source` retains AbstractTensor handles at the tool ports. It describes and maps Nodus-owned tensors, supplies the compiled entry's buffer and extent order, allocates Nodus outputs/workspaces, unmaps them, destroys workspaces, and publishes output handles. No alternative tensor registry is required.

`include/common/tensors/abstraction/tensor_abi.h` is the flat C boundary: uint64 handles, POD descriptors, maximum rank 8, element (not byte) strides. The native C++ AbstractTensor is move-only, with explicit handle ownership. Turing's Python NodusTensorOperations remains NumPy-backed at the Python interface; this is not evidence that native compiled rows must retain NumPy. Python development and native row products are complementary modes.

Turing additionally has canonical operator KernelIR and canvas/SSA roundtrip bridges. Those preserve operator identity, but must not be mistaken for the native handle transport above. The extraction contract and resolved identity replacement mechanisms already provide the sanctioned place for explicit native boundaries; arbitrary Python connections do not automatically acquire the new socket negotiator semantics.

## Bounded correction

The generated Buffer ABI tool returned with `ToolStackContext.succeeded` still true if its entry, raw stack, or inputs were missing. Output publication refusal also inherited success. It now starts failed and marks success only after all outputs are published. Existing cleanup and input restoration are unchanged. Added refusal tests and positive success assertion to `tests/artifact_buffer_import_test.cpp`.

## Measured verification

- Existing generated wrapper: missing entry, missing stack, missing inputs each incorrectly reported success; native probe exit 1.
- Current importer and wrapper rebuilt directly with installed MSVC: all three refusal cases pass, exit 0.
- Reused actual acceptance helpers with newly generated wrapper: 4- and 6-element dynamic buffers, two outputs, zeroed workspace, numerical values, runtime validation code 77, restored input handles, and refusal assertions pass, exit 0.
- `python -m pytest tests/test_turing_row_tool_compiler.py -q`: 13 passed, 3 skipped.
- Existing installed tensor core from October 24 was linked; this is a current importer/wrapper qualification, not a full runtime rebuild. Full CMake acceptance executable was not rerun. Qualification files are under `out/tensor-abi-qualification`.

## Actual engine product

Selected the existing `turing/examples/meowdoku/entity_motion.py::damped_axis`, unchanged source, ten float64 scalar inputs and two outputs. Request: `out/entity-motion-native/request.json`. Current sanctioned row compilation successfully produced repository SSA and C, with final concordance detector reporting zero disagreements. Descriptor: `out/entity-motion-native/product-c/backend/meowdoku_damped_axis.nodus_buffer_abi.txt`.

Native artifact compilation was blocked by tool network policy at `https://dc.services.visualstudio.com:443`. No completed DLL/adoption receipt or end-to-end engine execution is claimed. Do not reroute the denied compile without relevant authorization. Once permitted, finish this existing product and use the existing Nodus adopter/GraphDefinition path to compare real Python, native and client results. No server was restarted and no product was deployed by this work.
