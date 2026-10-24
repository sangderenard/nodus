# Nodus continuation handoff — 2026-10-23

This is the closeout queue for the Turing/Nodus integration session. Detailed
producer and platform evidence remains in
`docs/TURING_DELIVERY_INVENTORY_2026-10-24.md`; this file says what a later
Nodus worker should do next and what it must not rediscover or redesign.

## 2026-10-24 priority correction

The ProcessGraph intake item below is historical evidence, not the integration
direction. Sending Turing's ProcessGraph into Nodus was a category error:
Turing should lower that graph to SSA and execute the deployable result as a
row tool. Nodus instead needs to receive and return a complete definition of
its own tables, connectors, and properties.

Current priority order is:

1. Complete and prove the bidirectional `nodus.graph-definition.v1` boundary.
2. Bind SSA-backed Turing row products into definitions without handwritten
   connector reconstruction.
3. Prove definition -> native graph -> definition concordance plus execution.
4. Resume the SPIR-V and runtime-shaped crossing items below after the graph
   communication boundary is established.

Priorities 1-3 are now proved. V1 carries module/table layout and rows, stable
connector references, logical tensor dtype/shape/shape-source properties,
edge type/FIFO properties, scheduler mode, and named ports. A real Turing C
row is compiled through SSA, its graph definition is derived from `NODUS
BUFFER ABI V2`, the native graph returns the identical definition, and the
row executes through typed `AbstractTensorHandle` ports. No ProcessGraph is
sent to Nodus. Interactive overlay/meta-rope and live runtime state remain
explicitly outside V1.

The concordance proof no longer trusts the Python adoption caches. Read-only
headless introspection enumerates native Canvas modules, descriptors, stable
IDs, row programs and stack effects, edges, scheduler state, Table FIFO state,
and named ports in declaration order. The acceptance test corrupts the cached
Python scheduler/module/edge/port values after adoption and still reconstructs
the exact definition from native state. Connector tensor metadata remains part
of the Nodus-owned communication schema and is returned only after its stable
references are validated against native row-derived connector arity.

Priority 4's narrow SPIR-V lane is now proved as well. A real Turing
`turing.spirv-product.v1` is strictly adopted without recompiling its SPIR-V,
materialized as an ordinary Nodus row, and executed through the existing
Vulkan runner over 1000 f32 values with numerical and past-end-guard checks.
The scheduler now gives an explicitly declared `abstract_tensor_pointer`
stack contract precedence over the legacy row marker; FIFO type remains an
independent transport property. Arena map capacity may exceed logical tensor
bytes and is validated as capacity, not mistaken for the logical extent.

The web proof is no longer an empty pair of module shells. Programmatic row
installation synchronizes the explicitly addressed module table even when no
UI module is focused, and the headless API can export that concrete table to
an isolated root. The acceptance test checks the serialized row labels; the
retained PNG at `out/turing-web-proof-v4/turing-created-nodus-graph.png`
visibly contains both row programs and their edge.

The separate Eigen/operator audit is recorded in
`docs/EIGEN_OPERATOR_LANE_AUDIT_2026-10-24.md`. Its result is not a second
tensor hierarchy: 40 tested F32 elementwise Eigen tools reuse
`AbstractTensorHandle`; `eigen_tensor_view` itself has no users, and the
generic-object shell is not yet automatically bound to those tools.

The runtime-shaped C and LLVM crossings are now proved for rank 1 and rank 2
against canonical Turing `a6fdb31a` plus the measured Windows LLVM export
repair: the same graph-definition derivation carries dynamic
`shape_sources`, Buffer ABI V2 preserves `ELEMENT_COUNT` rather than
mislabeling it as shape axis zero, and both shapes execute numerically through
both backends. A Windows `.def` file now keeps the descriptor-declared LLVM
entry exported when the validation runtime is linked. The generated Nodus row
also consumes the validation-error ABI: a `(2,3)` versus `(3,2)` case reports
code 77, restores its raw-stack inputs, and publishes no output FIFO token.
Dynamic Fortran remains deliberately unsupported.

## Added queue: shader rows communicating through SSBO-backed FIFOs

The intended destination is a graph of Vulkan compute-shader rows whose
declared ports communicate through FIFO edges resident in shader storage
buffers. This does **not** make every edge a FIFO and does not remove stack
execution: FIFO-versus-stack remains a graph/interface choice, and
`RawStackFrame` remains a valid row boundary and transfer vehicle.

Past implementation evidence lives in the FIFO/storage types:

- `GP_MEM_BACKEND_VULKAN`, device/native-handle, zero-copy, and native-D2D
  capabilities already exist in `include/mem_backend.h`;
- `EdgeTensorFifo` allocates through an attached backend and its
  `push_from_frame` / `pop_to_frame` paths prefer backend-to-backend transfer;
- the table ABI can attach or swap an edge backend;
- the Turing SPIR-V product already describes set-0 storage-buffer ports and
  materializes as an ordinary shader row.

The storage halves are now joined. `GP_MEM_BACKEND_VULKAN` has a persistent
instance/device/queue context and Vulkan buffer handles, `EdgeTensorFifo`
rehomes its actual ring allocation when its table backend changes, and Vulkan
dispatch can bind those existing backend buffers directly as descriptor-set-0
SSBOs. The graph-definition FIFO property `storage` communicates `host` or
`vulkan` through Python, the headless ABI, native adoption, and native
readback. Old V1 mappings remain valid and default to `host`.

`test_vulkan_ssbo_fifo` is the lower-level execution proof: shader A writes a
256-byte packet, the packet crosses an `EdgeTensorFifo` by Vulkan-native D2D,
and shader B consumes it, with numerical verification and no host staging in
that transfer. It also proves that changing a public Table edge to the Vulkan
backend changes its real ring storage rather than only recording a context
hint. Explicit compute/transfer/host barriers and a fence wait define the
current synchronous row boundary.

The retained web-GUI proof at
`out/turing-spirv-ssbo-proof/turing-spirv-ssbo-fifo-web.png` visibly contains
two Turing-created SPIR-V row tables and their cable. Its definition and
native readback say that cable is a Vulkan-backed FIFO. The generated ordinary
SPIR-V row wrapper separately retains the established
`AbstractTensorHandle`/`RawStackFrame` execution option. That is not a defect in
the SSBO FIFO and must not be replaced by a second slot-borrowing protocol.

Ordered continuation after the completed KernelIR reconciliation and LLVM
crossing:

1. retain the existing `EdgeTensorFifo` publish/consume authority and its
   `push_from_frame` / `pop_to_frame` transfer boundary;
2. keep Vulkan-to-Vulkan FIFO transfers fail-closed: a failed native D2D copy
   must fail the transfer rather than silently staging bytes through host;
3. retain the two-shader numerical acceptance, graph-definition storage
   readback, FIFO order/full policy, barriers, and lifecycle gates;
4. keep `AbstractTensorHandle`/`RawStackFrame` transport as an independent
   option rather than interpreting Vulkan storage as permission to make every
   edge or row use a new protocol.

Do not implement this by exposing non-destructive FIFO slot reservations,
inventing a second GPU-edge abstraction beside `EdgeTensorFifo`, or treating
the one-shot Vulkan runner's temporary buffers as persistent graph state.

## Pinned state

- Nodus: `nogodsnomasters` at `e4fbcd7` before the current uncommitted boundary work.
- Turing: canonical checkout observed at `a6fdb31a` during this continuation.
- The canonical Turing checkout has no tracked local changes. Its `.claude/`
  and `out/` paths are pre-existing untracked material and must be preserved.
- The secondary Turing integration-table review worktree contains the named
  tensor-call patch represented by `580a8eb4`. That commit is already an
  ancestor of current Turing main and its files agree with main. Do not merge
  or recreate it.

## What is already real in Nodus

- A Python row can execute live through CPython with a retained source file,
  typed row ABI, stack ordering, multiple returns, and failure restoration.
- The same row declaration can request a Turing-compiled C, LLVM, or Fortran
  product; retained products are validated and adopted as ordinary row tools.
- `NODUS BUFFER ABI V2` carries fixed and runtime tensor ranks, ordered extent
  inputs, and output shape derivation. The C++ importer allocates dynamic
  outputs from Nodus `AbstractTensorHandle` values.
- Historical Turing ProcessGraph intake remains regression archaeology only.
  It must not be expanded: Turing lowers ProcessGraph computation to SSA and
  Nodus receives the resulting row plus a Nodus-owned graph definition.
- The source-shader importer validates the logical component contract,
  compiles GLSL/HLSL to SPIR-V, validates it, derives other shader languages,
  and retains versioned success/failure receipts.
- Graph entities and the proposed LLM-inhabited execution environment are
  documented by commit `2e31027`. They are architecture work, not a reason to
  change row or graph execution in the compiler-integration lane.

## Completed Nodus-owned work and remaining blockers

### 1. Adopt Turing's direct deployable SPIR-V product — complete

Add a strict consumer for `turing.spirv-product.v1`. It must validate hashes,
require `product_kind == deployable-shader`, require a complete product,
validate the declared target environment, and cross-check component identity
against the referenced `turing-compiled-program-api-v1` sidecar. It must not
reinterpret a `linkable-spirv-library` as a runnable shader and must not
recompile the retained `.spv` from source.

### 2. Consume the physical shader-deployment contract — complete for the published narrow profile

Parse `turing.shader-deployment.v1` into a Nodus-owned immutable description.
The logical `turing.shader-component.v1` ports remain the value ABI; the
deployment record supplies descriptor set/binding, layout, byte view, launch
controls, local size, group-count rule, target, provenance, completeness, and
shortfalls. Refuse unknown required resource kinds and any published
shortfall the selected runtime cannot satisfy. Do not infer physical bindings
from logical port order.

### 3. Make the narrow Vulkan product an ordinary row tool — complete

Use Nodus's existing Vulkan dispatch facilities for the profile Turing
already publishes: compute stage, set 0, contiguous storage-buffer bindings,
one storage control containing element count, declared local size, and a
recorded group-count rule. The generated row must accept and return
`AbstractTensorHandle` values, validate dtype/rank/shape against both logical
and physical records, and retain the original product files as its source
provenance.

The crossing test is the completion condition: publish a supported Turing
product, adopt it without handwritten bindings, dispatch it, compare output,
and prove the extra-workgroup past-end guard from the record alone.

### 4. Finish the runtime-shaped CPU crossing tests — complete for C and LLVM

Runtime-rank-1 and runtime-rank-2 C and LLVM rows are complete: shapes absent
during compilation were fed through native Nodus tensors and output shape and
values were checked. The LLVM producer validation-error-77 mismatch case is
also crossed through the Nodus scheduler and proves that a refusing stack row
cannot be mistaken for a successful FIFO producer. Keep runtime-shaped
Fortran explicitly refused until Nodus has a real dynamic Fortran extent ABI;
do not route it through the legacy fixed-shape YAML contract.

### 5. Make platform evidence first-class

Every crossing receipt should separately retain:

- contract/schema versions;
- Turing and Nodus revisions;
- OS release/build or Linux distribution/kernel and architecture;
- Python and relevant dependency versions;
- compiler, linker, shared-library suffix, and calling convention;
- shader frontend, SPIRV-Tools, SPIRV-Cross, Vulkan loader/API, device, and
  driver or Mesa/lavapipe version;
- exact command and pass/fail/skip result.

On Windows, store a canonical logical tool ID such as `spirv-as` separately
from the observed path/basename such as `spirv-as.EXE`. On Linux, do not infer
tool versions from a basename or from an unrelated cloud test. The current
lavapipe result is numerically strong but does not retain enough Linux version
information for reproduction; the detailed inventory lists every unknown.

## Remaining Turing-owned blockers; observe rather than patch here

1. The complete Python `SSAReferenceEvaluator` is not yet a deployable native
   interpreter. Current main correctly refuses instead of falsely passing an
   `__init__`-only library; `typing.Any` record treatment is the first current
   frontier. Do not package the historical partial library as the one-row SSA
   interpreter.
2. The complete AbstractTensor operator vocabulary has not been proved as
   deployable row tools. The delivered runtime-shape matrix establishes its
   tested elementwise subset only.
3. Direct deployable SPIR-V intentionally refuses array-shaped SSA values,
   bool storage without a physical spelling, and multiple results. Nodus must
   surface these named shortfalls rather than hide them.
4. Desktop GLSL compute declares OpenGL default-block launch controls and is
   not Vulkan-ready; WebGL needs a graphics pipeline and companion vertex
   stage; extracted host shaders still lack a component ABI.

## Latest measured gates

At Turing `05d9d2bd` on the recorded Windows host:

```text
python -m pytest tests/test_whole_class_compile_trial.py \
  tests/test_runtime_sized_spans.py tests/test_shader_deployment.py -q

139 passed, 23 failed, 3 skipped
```

The three whole-class/false-acceptance tests added by the latest pull pass.
Twenty-two failures are the Windows C validation-export defect; the remaining
failure is the case/extension-sensitive assertion against `spirv-as.EXE` and
`spirv-val.EXE`, although assembly and validation succeeded.

The current Nodus boundary and execution gate is:

```text
10/10 passed

nodus_python_graph_test
turing_nodus_graph_crossing_test
turing_nodus_runtime_shape_crossing_test
turing_spirv_product_test
turing_nodus_spirv_crossing_test
spirv_translation_stage_test
vocab_tool_dll_test
vulkan_dispatch_test
eigen_kernel_test
generic_object_test
```

Re-run the focused test belonging to each continuation item; do not use the
repository's historical success receipts as evidence that a new product or
platform still passes.

## Scope guard

This continuation is consumer work inside Nodus. It does not authorize edits
to Turing internals, new graph-edge types, a replacement graph runtime, a
second tensor abstraction, or a CPython fallback for the native SSA
interpreter. If a Turing producer fact is absent, record the exact missing
contract and hand it to the Turing workers.
