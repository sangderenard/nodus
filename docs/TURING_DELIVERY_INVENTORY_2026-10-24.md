# Turing delivery inventory after the 2026-10-24 pull

Status: initially observed against canonical Turing `main` at `a8548ab6` and refreshed after a fast-forward to `05d9d2bd`; Nodus was at `d5e62bd` before this refresh. No Turing source was changed during either audit.

## Scope

This reconciles the two Turing request sets made for Nodus:

1. rank-known, runtime-sized tensor lowering and a host-readable runtime ABI for row-tool adoption;
2. retained shader source, deployable shader products, and physical launch metadata for Nodus shader ingestion and execution.

The distinction used below is:

- **provided**: present in Turing main;
- **consumed**: Nodus has a corresponding reader/adopter;
- **proved together**: a focused test crosses the producer/consumer boundary;
- **open**: one side still refuses, ignores, or cannot execute the product.

## Pull receipt

Canonical `C:/dev/Powershell/turing` fast-forwarded from `6058d247` to `a8548ab6`.

Relevant merged commits:

- `40a497d5` — rank-known runtime-sized spans and `turing-runtime-abi-v1`;
- `93b97b3d` — the Nodus shader deployment request;
- `010ca120` — shader source and deployable SPIR-V products;
- `7e0cae6a` — shader product receipts and convergence documentation;
- `6df23cdd` — recorded shader gate result.

The Turing checkout retained only its pre-existing untracked `.claude/` and `out/` paths.

### Refresh after false-acceptance work

Canonical Turing subsequently fast-forwarded from `a8548ab6` to `05d9d2bd`
(PRs 834–836). The twelve commits add the whole-class
`SSAReferenceEvaluator` trial, collection-parameter element typing, and the
one-branch list-rebound repair. They also make a material false acceptance
loud: the class compilation route previously reported success after lowering
only `SSAReferenceEvaluator.__init__` and silently omitted the other requested
methods. Current main adds every dependency seed to the activation roots and
refuses library emission if any planned activation root has no lowered
function.

That is progress for Nodus even though it does not produce the requested
single-row SSA interpreter yet. The current whole-class trial stops honestly
at its first unresolved compiler contract (`typing.Any` treated as a record
identity); its per-method catalogue records further named frontiers. Nodus
must not package the earlier `__init__`-only library or treat a created shared
library as proof that the interpreter surface exists. A future interpreter
product must prove its complete requested method/export surface from retained
source-function identities.

The focused Windows refresh ran
`test_whole_class_compile_trial.py`, `test_runtime_sized_spans.py`, and
`test_shader_deployment.py`: **139 passed, 23 failed, 3 skipped**. The new
whole-class false-acceptance guards passed. Twenty-two failures remain the
known missing Windows exports for `turing_validation_error_reset`/take on the
runtime-shaped C lane; the other remains the case/extension-sensitive
`spirv-as.EXE` / `spirv-val.EXE` assertion. No new Nodus boundary regression
was exposed by these commits.

## Versioning and platform evidence

The words "version" and "platform" refer to three different things at this
boundary. They must not be collapsed into one number:

1. **Contract version** describes the bytes and fields exchanged between
   Turing and Nodus. A contract-major change can break a consumer.
2. **Producer/consumer revision** identifies the exact implementations that
   emitted and consumed those bytes.
3. **Execution evidence** identifies the OS, architecture, runtime, compiler,
   shader tools, loader, and device used for one test run. Updating one of
   these does not silently change a contract version.

Every cross-project receipt should therefore retain all three. A statement
such as "tested on Windows" or "tested on lavapipe" is not a reproducible
platform receipt without the available version fields. Unknown values must be
written as unknown, not inferred from a different test run in the repository.

### Contract versions observed

| Boundary | Versioned identifier | Owner |
|---|---|---|
| Compiler runtime shapes | `turing-runtime-abi-v1` | Turing |
| Nodus native buffer call | `NODUS BUFFER ABI V2` | Nodus |
| Retained row-tool package | `nodus-turing-row-tool-product-v1` / `NODUS ROW TOOL PRODUCT V1` | Nodus |
| Compiled program API | `turing-compiled-program-api-v1` | Turing |
| Logical shader component | `turing.shader-component.v1` | Turing |
| Physical shader deployment | `turing.shader-deployment.v1` | Turing |
| Source shader handoff | `nodus-turing-shader-artifact-v1` | Shared request, produced by Turing and consumed by Nodus |
| Nodus shader import receipt | `nodus-turing-shader-artifact-receipt-v1` | Nodus |
| Direct SPIR-V handoff | `turing.spirv-product.v1` | Turing |

### Windows audit host

This audit's Windows evidence was measured locally rather than inferred from
package constraints:

| Layer | Exact observed value |
|---|---|
| OS | Windows build reported by Python as `Windows-10-10.0.26300-SP0`; x64 toolchain |
| Turing/Nodus Python | CPython 3.11.7, NumPy 2.3.5, pytest 9.0.3 |
| Build tools | CMake 4.1.0-rc1, Ninja 1.13.0, Git 2.43.0.windows.1 |
| MSVC selected by Nodus | compiler 19.44.35215 x64; linker 14.44.35215.0 |
| Windows SDK selected by CMake | 10.0.26100.0, targeting Windows 10.0.19045 |
| Vulkan SDK directory | `C:/VulkanSDK/1.4.335.0/Bin` |
| `glslangValidator` | Glslang 11:16.0.0; GLSL/ESSL Khronos 16.0.0; SPIR-V 1.6 |
| `spirv-as`, `spirv-val` | SPIRV-Tools v2025.5, `v2025.5.rc1-0-g262bdab4` |
| `spirv-cross` | revision `vulkan-sdk-1.4.328.0-34-gfb0c1a30`, 2025-12-08 |
| DXC | `dxcompiler.dll` 1.10, compiler 1.8.0.5123 (`3f85295cf`) |
| Vulkan instance | 1.3.301 |
| Vulkan devices visible | GTX 1660 SUPER and RTX 3060: API 1.3.280, NVIDIA driver 560.94, conformance 1.3.8.2; AMD Radeon integrated: API 1.3.280, driver 24.10.24 / 2.0.302, conformance 1.3.5.2 |

The SDK directory version and SPIRV-Cross's own revision intentionally remain
separate: the executable reports a 1.4.328.0 lineage while installed under the
1.4.335.0 SDK. A receipt must preserve both facts instead of replacing one
with the other. Vulkan loader warnings about stale Steam layer JSON files were
also observed, but were not the cause of the focused shader failure.

`zig`, `clang`, and `gcc` were not on this PowerShell host's `PATH`. Native C++
tests used MSVC; Turing backend-specific compiler discovery remains whatever
the sanctioned compiler entry recorded for each retained product.

### Linux evidence and its limits

Turing's current shader receipt records a cloud execution on **lavapipe
(CPU)**: float32 add, int32 multiply, and int32 subtract matched NumPy bit for
bit over 1000 elements, with the past-end guard checked by dispatching an extra
workgroup. It does not record the Linux distribution or kernel, architecture,
Mesa/lavapipe version, Vulkan loader/API version, SPIRV-Tools version,
glslang version, SPIRV-Cross version, Python version, or GPU/CPU identity.
Those fields are therefore unknown for that shader run.

The baseline contains a separate, well-versioned Linux receipt for the
2026-10-03 NumPy scatter work: Linux, CPython 3.12.14, NumPy 2.3.5, pytest
9.1.1. That receipt is useful evidence for those Python/NumPy tests only. It
must not be presented as the environment of the 2026-10-24 lavapipe run.
Turing's `requirements.txt` supplies broad dependency ranges rather than a
Linux lockfile and does not reconstruct either environment.

Before claiming a repeatable Windows/Linux parity gate, each new crossing run
should record at least: OS family and exact release/build/kernel; machine
architecture; Python and dependency versions; native compiler and linker
versions; shared-library suffix and calling convention; shader frontend,
SPIRV-Tools and cross-compiler versions; Vulkan loader/API; physical or
software device and driver/Mesa version; Turing and Nodus commit IDs; contract
identifiers; command; and pass/fail/skip counts. On Windows, receipts should
store a canonical logical tool name such as `spirv-as` alongside the exact
path and basename such as `spirv-as.EXE`; comparisons of the logical name must
be case-insensitive. On Linux, the exact basename remains evidence but must not
stand in for the package version.

## Set 1: runtime-shaped row tools

### What Turing provided

| Facility | Location | Observed contract |
|---|---|---|
| Runtime-rank tensor law | `src/compiler/glsl_deployment_strategy.py` | Elementwise operations over same-rank runtime-sized spans and rank-0 scalars produce runtime-sized spans, not accidental scalars. Mixed runtime ranks or unsupported broadcasts remain unsettled. |
| SSA physical rank receipt | `src/compiler/precompile_to_ssa.py` | `_runtime_rank_physical` posts `storage: span` and rank from the existing shape-state identity. |
| Public runtime ABI | `src/compiler/runtime_abi.py` | `turing-runtime-abi-v1`; every buffer records index, value id, role, dtype, rank, static shape or null, and `extents_from`; extent slots record buffer, kind, and axis. |
| C backend | `src/compiler/ssa_c_backend.py` | Runtime loops, C `int64` extent vector, runtime result allocation metadata, and axis mismatch validation code 77. |
| LLVM backend | `src/compiler/ssa_llvm_backend.py` | Runtime loops, LLVM `int32` extent vector, equivalent output-shape derivation and validation. |
| Fortran backend | `src/compiler/ssa_fortran_backend.py` | Runtime-sized arithmetic/comparison loops and named shared extent symbols in the Fortran API product. |
| Regression matrix | `tests/test_runtime_sized_spans.py` | Rank 0, fixed `(4,)`, fixed `(2,3)`, runtime rank 1, runtime rank 2; C/LLVM/Fortran; 14 arithmetic/comparison/composite expressions; representative unseen extents and mismatch refusal. |

Turing therefore supplied the compiler facts Nodus had requested: rank is independent of static shape, runtime extents are ordered ABI inputs, outputs name the input whose extents determine allocation, and disagreement is an explicit validation failure rather than an overrun.

### What Nodus already consumes

| Facility | Location | State |
|---|---|---|
| Row declaration | `tools/turing_row_tool_compiler.py` | Accepts scalar `shape: []`, fixed shapes, and `rank: N, shape: null`; dynamic output/workspace columns must name `shape_from`. |
| Product descriptor | same | Emits `NODUS BUFFER ABI V2` with `VALUE`, exact `BUFFER` order, axis-addressed `EXTENT`, and per-axis `SHAPE ... FROM ...`. C uses `i64`; LLVM uses `i32`. |
| ABI importer | `src/artifact_importer.cpp` | Parses V1/V2, validates ranks/dtypes/extent sources, checks input shapes, allocates dynamic outputs/workspaces from Nodus tensor handles, and calls the artifact with exact buffer/extent order. |
| Product adoption | `src/turing_product_adopter.cpp` | Validates the retained product receipt/manifest, generates the wrapper, stages it, builds it, and loads it into the tool registry. |
| Runtime carrier | tensor ABI and generated wrapper | Inputs/outputs are `AbstractTensorHandle` values. Dynamic allocation stays owned by Nodus; Turing sees pointer and extent vectors, not Nodus handles. |

### Evidence from this audit

- Nodus Python producer tests: `23 passed, 3 skipped` across row-tool and shader-artifact suites.
- Explicit real Turing public-boundary row compilation: C, LLVM, and Fortran scalar products all compiled and retained successfully: `3 passed`.
- Nodus C++ buffer importer and product adopter: both focused CTest cases passed. The existing build target invocation itself named no MSBuild project after CMake regeneration, but the already generated CTest executables ran successfully: `2/2 passed`.
- Initial Turing combined runtime-span and shader suite at `a8548ab6`: `136 passed, 23 failed, 3 skipped`.
- Refreshed suite at `05d9d2bd`, including the new whole-class trial: `139 passed, 23 failed, 3 skipped`; all three added passing tests are the false-acceptance/whole-class coverage.

### Current runtime gaps

1. **Current-main C execution is red on this Windows host.** Every runtime-shaped C test reaches the compiled DLL and then fails because `CModuleArtifact.reset_validation_error()` requests `turing_validation_error_reset`, which the linked DLL does not export. The validation runtime source defines `turing_validation_error`, `turing_validation_error_reset`, and `turing_validation_error_take` as ordinary external C functions but supplies no Windows `__declspec(dllexport)` decoration. Default ELF visibility can expose the same spelling on Linux; a Windows DLL does not receive that guarantee. LLVM and Fortran cases in the same matrix were not the source of those failures. This is a platform-specific Turing shared-library export defect, not a Nodus shape-contract mismatch.

2. **Real dynamic-shape Nodus adoption is not yet an end-to-end test.** Nodus's dynamic tests exercise its descriptor generation and C++ importer with controlled artifacts. Its opt-in real-Turing test is scalar. A final crossing test must compile a real runtime-rank-1 or rank-2 C/LLVM row, adopt it, push Nodus tensor handles, execute the row, and verify result shape and values.

3. **Dynamic Fortran is intentionally refused by Nodus.** Turing now lowers runtime-sized Fortran tensors, but `turing_row_tool_compiler.py` refuses a dynamic Fortran request before the legacy YAML importer can narrow it. Nodus needs a Fortran dynamic-extent adoption contract or a common buffer ABI before this lane is usable as a runtime-shaped row.

4. **The 14-expression matrix is not the full AbstractTensor operator vocabulary.** It proves the corrected elementwise/runtime-span class. It does not establish all roughly mid-hundreds AbstractTensor operations, reductions, structured operations, multi-output operations, or every dtype as deployable Nodus row tools.

### Runtime readiness judgment

| Path | Judgment |
|---|---|
| Fixed/scalar C, LLVM, Fortran product generation | Available and crossed at compile/product level |
| Runtime-shaped LLVM product contract | Provided and structurally consumed; final real Nodus execute test still needed |
| Runtime-shaped C product contract | Provided and structurally consumed; current Turing DLL execution is blocked by missing validation-runtime exports |
| Runtime-shaped Fortran product | Compiler provided; Nodus consumer deliberately refuses it |
| Full AbstractTensor operator set | Not provided by this delivery |

## Set 2: shader deployment products

The authoritative producer report is Turing `docs/compiler_convergence/38_shader_deployment_products_2026_10_24.md`. It explicitly answers `docs/NODUS_SHADER_DEPLOYMENT_REQUEST.md` without changing planner or SSA lowering.

### What Turing provided

#### Source products

Each supported shader writer can publish:

- byte-identical emitted source;
- `turing-compiled-program-api-v1` sidecar with the authoritative `turing.shader-component.v1` logical ABI;
- `metadata.shader_deployment` using `turing.shader-deployment.v1`;
- exactly the four-key `nodus-turing-shader-artifact-v1` request manifest Nodus already accepts;
- `SHA256SUMS` over the source, sidecar, manifest, and companion source where present.

Writers covered are desktop GLSL compute, WGSL, WebGL fragment plus its vertex companion, and extracted host shaders.

#### Physical deployment record

The deployment record supplies what cannot safely be inferred from logical port order:

- language, version/profile, stage, entrypoint, target client/environment, and SPIR-V version;
- physical resources with set, binding, kind, access, layout, dtype, element size, byte offsets/extents, extensions, capabilities, and specialization constants;
- the logical port view each resource realizes, including rank, static or runtime shape, and extent source;
- local size, exact or rule-based group counts, element-count source, extent order, and launch controls with derivations;
- provenance, structured completeness, and stable shortfall codes.

The deployment record refers to the component ABI by `component_id`; it does not duplicate logical ports as a second authority.

#### Separate SPIR-V products

Turing now distinguishes:

- `linkable-spirv-library`: no entrypoint or deployment API;
- `deployable-shader`: GLCompute `main`, LocalSize, descriptor/interface variables, element-count guard, target environment, deployment API, assembly, optional binary, manifest, and hashes.

The deployable scalar compute path uses contiguous set-0 storage buffers for feeds/output plus a storage-buffer element-count control. The recorded Turing evidence assembled, validated, and executed float32 add, int32 multiply, and int32 subtract through Vulkan/lavapipe over 1000 elements, bit-for-bit against NumPy, including an extra guarded workgroup.

### What Nodus already consumes

`tools/turing_shader_artifact.py` consumes the four-key source manifest. It:

- rejects path escape and unknown fields;
- validates the compiled-program API and logical component identity/language/stage/entrypoint;
- retains source and API byte-for-byte;
- compiles GLSL-family or HLSL source to SPIR-V;
- validates SPIR-V;
- derives GLSL, ESSL, HLSL, and MSL through SPIRV-Cross;
- records hashes, exact commands, stdout/stderr, exit codes, and failure receipts.

The focused Nodus shader tests passed in this audit. Real tool tests run when their executables are present; WGSL refusals and malformed contracts are covered.

### Current shader gaps

1. **Nodus ignores `metadata.shader_deployment`.** It validates the logical component ABI but does not parse the physical resources, launch controls, group-count rule, target environment, or structured shortfalls Turing now supplies. Consequently it cannot yet instantiate a runnable row directly from the richer record.

2. **Nodus does not ingest `turing.spirv-product.v1`.** The most directly deployable Turing product—the validated `.spv` with deployable-shader manifest and API—has no Nodus adopter. The existing Nodus path starts from source and recompiles it.

3. **WGSL source remains refused.** Turing can publish WGSL source products; Nodus has no configured WGSL-to-SPIR-V frontend. The new Turing direct-SPIR-V sibling is the natural bypass, but gap 2 prevents using it.

4. **Desktop GLSL compute is OpenGL-ready, not Vulkan-ready.** Its `u_count`, `u_extent`, and `u_slot` controls are default-block uniforms. Turing publishes the stable shortfall `vulkan-default-uniform-controls`; Nodus's Vulkan-source compilation path should refuse rather than guess. Direct OpenGL execution would require an OpenGL consumer of the published record.

5. **WebGL is a graphics pipeline.** Turing retains the companion vertex stage and marks `graphics-stage`. Nodus's current importer handles one source stage and does not assemble the full raster pipeline from the deployment record.

6. **Extracted host shaders lack a component ABI.** Turing correctly publishes provenance and `no-component-abi`; Nodus's strict logical-ABI validation rejects these until a real ABI is derived.

7. **The direct deployable SPIR-V emitter is intentionally narrow.** It refuses array-shaped SSA values, bool storage without a physical element spelling, and multiple results. Those are named Turing shortfalls, not silent Nodus failures.

8. **One Windows-only test assertion is red.** Turing's shader validation records executable basenames as `spirv-as.EXE` and `spirv-val.EXE`; one test asserts the Unix spellings `spirv-as` and `spirv-val`. The tools ran and the module validated. The portable contract is a canonical logical tool ID plus the exact observed executable path/name/version; the failure is receipt-name normalization in the test, not shader generation or validation.

### Shader readiness judgment

| Product | Judgment |
|---|---|
| GLSL/HLSL source + logical ABI conversion | Nodus path exists and focused tests pass |
| Rich physical deployment metadata | Fully produced by Turing; unconsumed by Nodus |
| Deployable SPIR-V binary product | Fully produced and measured by Turing; no Nodus adopter |
| WGSL source product | Produced by Turing; refused by Nodus source importer |
| Desktop GLSL compute on Vulkan | Explicitly incomplete; Turing publishes the reason |
| WebGL graphics product | Produced with companion stage; no Nodus pipeline adopter |
| Extracted shader | Provenance retained; no component ABI, so not runnable in Nodus |

## Nodus work now unlocked

The Turing workers supplied enough contract to proceed without more compiler design for these Nodus tasks:

1. Add a strict `turing.spirv-product.v1` adopter that validates hashes, `product_kind == deployable-shader`, `complete`, target environment, component identity, and API/deployment agreement before loading the existing `.spv`.
2. Parse `turing.shader-deployment.v1` into a Nodus-owned immutable deployment description. Refuse unknown required resource kinds and every published shortfall that the selected runtime cannot satisfy.
3. Map the narrow deployable-SPIR-V profile—compute stage, set 0, contiguous storage buffers, one storage control, declared local size/group rule—onto Nodus's existing Vulkan dispatch facilities.
4. Wrap that dispatch as a registered row tool whose `AbstractTensorHandle` inputs/outputs are checked against the logical component ports and physical resource dtypes/shapes.
5. Add one true crossing test: Turing publishes a scalar/vector-safe deployable SPIR-V product; Nodus adopts it, dispatches it, and verifies output and past-end guard behavior from the published record only.
6. Add one true runtime-shaped CPU crossing test for C and LLVM after the C validation-export issue is resolved; retain the explicit Fortran refusal until Nodus has a dynamic Fortran ABI.

These are Nodus consumer tasks. They do not require changing Turing's compiler internals or adding graph-edge types.

## Bottom line

Both requested producer contracts are now substantially present in Turing. Runtime-shaped C/LLVM metadata aligns with Nodus Buffer ABI V2, and shader products now include the source, logical ABI, physical launch contract, provenance, hashes, and separate deployable SPIR-V product that were requested.

The remaining boundary is measurable rather than conceptual: Nodus must consume the richer shader/SPIR-V records and prove real dispatch; dynamic Fortran remains unsupported in Nodus; and current Turing main must restore Windows-visible C validation-runtime exports before the runtime-shaped C execution matrix is green on this host.
