# Nodus continuation handoff — 2026-10-23

This is the closeout queue for the Turing/Nodus integration session. Detailed
producer and platform evidence remains in
`docs/TURING_DELIVERY_INVENTORY_2026-10-24.md`; this file says what a later
Nodus worker should do next and what it must not rediscover or redesign.

## Pinned state

- Nodus: `nogodsnomasters` at `924d43d` before this handoff.
- Turing: canonical `main` at `05d9d2bd`, equal to `origin/main`.
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
- Turing process-graph intake is exercised end to end by commit `8e5150e`.
  There is no need to invent control-flow edge types or alter the graph
  runtime for this integration.
- The source-shader importer validates the logical component contract,
  compiles GLSL/HLSL to SPIR-V, validates it, derives other shader languages,
  and retains versioned success/failure receipts.
- Graph entities and the proposed LLM-inhabited execution environment are
  documented by commit `2e31027`. They are architecture work, not a reason to
  change row or graph execution in the compiler-integration lane.

## Next Nodus-owned work, in order

### 1. Adopt Turing's direct deployable SPIR-V product

Add a strict consumer for `turing.spirv-product.v1`. It must validate hashes,
require `product_kind == deployable-shader`, require a complete product,
validate the declared target environment, and cross-check component identity
against the referenced `turing-compiled-program-api-v1` sidecar. It must not
reinterpret a `linkable-spirv-library` as a runnable shader and must not
recompile the retained `.spv` from source.

### 2. Consume the physical shader-deployment contract

Parse `turing.shader-deployment.v1` into a Nodus-owned immutable description.
The logical `turing.shader-component.v1` ports remain the value ABI; the
deployment record supplies descriptor set/binding, layout, byte view, launch
controls, local size, group-count rule, target, provenance, completeness, and
shortfalls. Refuse unknown required resource kinds and any published
shortfall the selected runtime cannot satisfy. Do not infer physical bindings
from logical port order.

### 3. Make the narrow Vulkan product an ordinary row tool

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

### 4. Finish the runtime-shaped CPU crossing tests

When Turing's Windows C validation-runtime exports are restored, compile and
adopt a real runtime-rank-1 and runtime-rank-2 C row and then the equivalent
LLVM rows. Feed shapes not present during compilation, check output shape and
values, and check mismatch error 77. Keep runtime-shaped Fortran explicitly
refused until Nodus has a real dynamic Fortran extent ABI; do not route it
through the legacy fixed-shape YAML contract.

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

## Turing-owned blockers; observe rather than patch here

1. Runtime-shaped C execution on Windows still lacks exported
   `turing_validation_error_reset` and `turing_validation_error_take` symbols.
   This produced 22 focused failures at `05d9d2bd`.
2. The complete Python `SSAReferenceEvaluator` is not yet a deployable native
   interpreter. Current main correctly refuses instead of falsely passing an
   `__init__`-only library; `typing.Any` record treatment is the first current
   frontier. Do not package the historical partial library as the one-row SSA
   interpreter.
3. The complete AbstractTensor operator vocabulary has not been proved as
   deployable row tools. The delivered runtime-shape matrix establishes its
   tested elementwise subset only.
4. Direct deployable SPIR-V intentionally refuses array-shaped SSA values,
   bool storage without a physical spelling, and multiple results. Nodus must
   surface these named shortfalls rather than hide them.
5. Desktop GLSL compute declares OpenGL default-block launch controls and is
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

The last measured Nodus boundary checks before this closeout were:

```text
23 passed, 3 skipped  # Python row-tool and shader-artifact suites
3 passed              # opt-in real scalar Turing C/LLVM/Fortran products
2/2 passed            # C++ buffer importer and Turing product adopter
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
