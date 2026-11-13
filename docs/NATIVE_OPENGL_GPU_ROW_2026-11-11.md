# Native OpenGL GPU row checkpoint

## Current scope

Native source is in `include/kernel_opengl_runner.h`,
`src/kernels/kernel_opengl.cpp`, `include/opengl_gpu_row_tool.h` and
`include/opengl_gpu_graph_binding.h`.
The row only accepts **link-only** work. It does not execute the Studio shader,
create a renderer/context, select a display adapter, or change physics/RNG.
The protected native process/provider and three actual forward source stages now
exist. OS containment and the canonical row-to-real-CPU-child path are tested;
all three emitted stages passed authorized offline syntax-only validation.
Separately authorized native driver-link attempts now have identity-verified
receipts, including known successes and driver-memory failures; see
`docs/NATIVE_PROTECTED_OPENGL_PROCESS.md` for the exact current results. Those
receipts are not execution validation. Production registration, complete shader
execution and application readiness remain unvalidated; execution hold is unchanged.

The native package helper accepts exact GLSL source objects and links them into
one compute program. Multiple attached objects still have one reachable program
graph at link time; this is not a promise to reduce NVIDIA linker memory.
`gp_gl_compute_link` has no dispatch path. The separate native submission helper
can submit already-authored complete-invocation stages with explicit uniforms,
unchanged borrowed SSBO/image names and exact barrier masks. It refuses a cut
inside an invocation because the authored candidate continuation ABI has not
been admitted to this runner or validated numerically.
Repeated use of one program requires the same full scheduled uniform set at each
stage, preventing later preflight values from leaking into earlier stages.

## Canonical queue and row ownership

`OpenGLGpuHost::peer()` is a host in the existing `ContractNegotiator`; its
exchange only admits an engine-owned source reference or returns a completed
receipt. No extra queue, scheduling thread or Python stage coordinator exists.
The existing `Contract` and `Result` tokens remain authoritative for work and
native result FIFO transport. The GPU ITool has typed Control and Telemetry ports.
The GPU row performs the link, releases its program on the owning context, and
reports link-success separately from FIFO acceptance, shader submission and
execution completion. Link-only never reports execution completion.
Telemetry v2 also carries protected-child containment/identity/reaping proof,
unknown-outcome status, PID, exit code, elapsed time and process/job memory peaks.
Process-resource reclamation and verified GL teardown are separate fields.
Requested and observed Windows adapter LUIDs are exposed, and successful protected
link receipts must contain a nonzero exact match as well as an actual attempt.

`first_suitable_free` is an Offer policy supplied to the existing negotiator.
For the local path, suitability requires compatible link-only source, a current
registered GL context, admitted supervision and free capacity. The protected
path uses structural provider availability before exact-loan authorization;
actual context identity is verified later in the child, before shader work.
Neither eligibility path currently ranks GPUs by measured load. It does **not**
mean the physical GPU is globally idle. Physical utilization is explicitly
unavailable. Vulkan ordinals, memory budget and UUID/LUID inventory are not used
as claims of OpenGL adapter affinity.

Intake retains at most one in-flight job, 4096 source objects, 16 MiB aggregate
source plus labels, and 64 outcome records by default. The engine explicitly
retires outcomes after commit. Before linking, the row protects its exact loan
in the canonical contract Entry. Protected entries do not reoffer on expiry or
rejection. Removing peers still detaches callbacks normally; unresolved protected
entries remain tombstones rather than being replayed on a replacement row.
Tombstones are bounded by negotiator capacity. An administrative resolution API
for abandoned protected work is not provided yet. Protection is not durable over
destruction of the negotiator or process restart.
For protected-process work, the native Exchange opt-in holds the exact loan at
acceptance so an explicit approval wait cannot expire and silently change its
generation. Structural provider availability allows admission; exact source/loan
authorization is checked separately before starting. Pending approval leaves the
same work Queued, without a process or driver call. Cancelling queued work returns
the ordinary Cancelled result for engine commit. This ownership hold grants no
permission to link. Existing exchanges default to ordinary lease behavior.

## Native protection implemented; hardware proof still required

The GL helper checks Nodus's existing context registry against the actual current
context callback. A quiescent GPU row can bind its control lane on the first Query
without a GL context, so the canonical scheduler can deliver honest refusal
telemetry. This confers no GL readiness. The separate current-context lane claim
requires that registered context to be already current there. Neither operation
activates or transfers a context. Wrong-lane/context work is refused. Cleanup failures retain
the program and block result publication until release on the owning context;
Advance then retries cleanup only, never compilation. The owner must call
shutdown successfully before destruction. A violated destruction contract reports
unreleased GL ownership rather than deleting on another context.
The same rule now covers failed compile/link: a false return can retain a
non-executable opaque cleanup owner. Verified deletes remove individual objects
from that owner. Remaining shader/program names survive errors, and eventual
cleanup does not turn an original link failure into success.
Successful delete/error-report paths do not allocate before recording ownership
changes. Deterministic allocation faults immediately after mock program/shader
deletion and during error construction are covered without escaping the C ABI.

The existing ThreadManager `ExecMode::Sequential` is the canonical manager-lane
mechanism. A separate bounded repair restores that documented behavior. Typed GPU
edges must use the existing `ModuleStackContract::AbstractTensorPointer` and
registered Control/Telemetry type/extent, not the default scalar/Opaque route.
The graph binding helper validates exact lossless FIFO type/extent/capacity,
installs the bound native factory and sets both typed stack and Sequential
contracts. The actual ThreadManager regression now exercises this path through
GraphRuntime and real table FIFOs, including control/refusal telemetry without a
context. Mock context availability is not real GL activation. No production
context activation is installed by this patch.

The supervisor admission callback defaults to refusal. It is not itself a
supervisor or proof of protection. A real native context-owning child must be
isolated with verified memory and wall-time bounds before using the known-risk
Studio shader. Ordinary C++ exception handling cannot contain driver crashes,
OOM or hangs. The existing Turing compile host launches a Python compiler process
and does not supply this GL ownership/protection seam. No actual next shader link
is authorized by these CPU tests.

The optional `Config.protected_provider` route is separate from the local gate.
It never calls parent-context callbacks or local GL entrypoints. It starts one
protected request after pinning the exact canonical loan, then returns Running
to the manager lane. Later Advance commands poll that same operation; CancelActive
requests termination without releasing ownership or retrying. Only a matching
host/contract/generation receipt can finish it. Unreaped/uncleaned outcomes remain
busy, and cleanup polling never starts another link. Reaped uncertain outcomes
can return failure but never claim verified GL cleanup or link success.
The native provider/context owner is implemented in
`opengl_protected_process.cpp`, `opengl_protected_child.cpp` and the separate WGL
worker. See `NATIVE_PROTECTED_OPENGL_PROCESS.md`. Its real Windows CPU-child tests
cover containment, cancellation, watchdog, exact source/loan receipt and cleanup.
`tests/opengl_protected_row_threadmanager_test.cpp` also passed the real canonical
manager/FIFO/provider/CPU-child/result/commit path, including approval waiting
beyond a one-tick lease and nonblocking progress, with no WGL or GPU callback.
The actual hardware context selection/identity and driver behavior remain
unvalidated; CPU fixture identity is explicitly not a hardware result.

CancelActive also handles completion racing cancellation: a provider's false
cancel result causes a poll of the original operation, delivering final typed
Telemetry instead of restoring Control or relinking. Its deterministic row
regression passed at 2026-11-11 21:27:33 UTC; see
`build/dot-opengl-native-cpu/cancel-completion-race.log`.
The final context-selection telemetry extent then passed a fresh row and actual
ThreadManager rebuild/run at 21:30:05 UTC. See `cancel-completion-race-final.log`
and `out/opengl-gpu-threadmanager/cancel-schema-final.log`. Frozen row header
SHA-256: `caa938d02918d6f3e7da6bee9c1f91f7c6e00a52953ec3744ffce0fae375ffd9`.

## Three actual forward sources exist; execution admission remains blocked

The current Turing source divider performs Python source triage and canonical
compiler emission. Its GLSL reader currently supports scalar helper parameter
semantics, not the actual shader's structs, vector/out/inout parameters, SSBOs and
private mutable globals. No substitute parser/compiler has been introduced.
The separate closed-input native packager now authors initialize, nearest-hit
and advance entries from the exact SHA-guarded frozen source, with copied byte
provenance and a 128-word carried-state ABI. See
`STUDIO_FORWARD_STAGE_PACKAGE.md` and `studio_forward_stage_package.cpp`.
These three emitted GLSL files passed offline `glslangValidator -S comp` syntax
validation. They remain `GP_GL_UNVERIFIED_CONTINUATION`: no driver link or stage
execution was performed, and the native runner continues to refuse execution.

The native Studio main has complete forward, sensor and reactive families, but
the prebuilt pump currently dispatches only forward. Dividing whole existing pass
entries is not equivalent to splitting a forward bounce. A nearest-hit boundary
must preserve invocation/root/bounce/RNG identity; ray origin/direction/packet;
current medium and `studio_active[32]`; sphere and partition selected-hit outputs;
region sampling/hit globals; and latched copied response state, as well as all
original resource ownership, atomics and visibility. The candidate now inventories
these values explicitly, but that is not numerical or all-path equivalence proof.
Existing recorded source already uses the
measured 16 compute SSBO blocks. A continuation buffer cannot simply add a block;
any use of shared opt-in auxiliary storage requires an explicit non-overlapping
layout and bounded reservation, with the ordinary path unchanged.
The candidate layout appends a checked tail in existing binding 23; it allocates
nothing and the producer has not been wired to reserve it. Cross-invocation
atomic append order can alter bounded retained prefixes and later reactive
RNG-to-slot associations. Exact secondary-path replay is not claimed. Sensor and
reactive integration, the complete response/consumer path and execution review
remain unfinished.

## Remaining shared-canvas integration

- Existing ContractNegotiator owns placement turns and FIFO results. The Offer
  helper scans compatible work for the host visited by its existing round-robin
  turn; it is not a fastest/least-loaded device ranking algorithm.
- Supplied native inventory now has a source/CPU-verified registration path in
  `gpu_inventory_rows.h/.cpp`, using UUID-keyed ordinary factories, the existing
  custom-Offer registry and existing module-library serialization. Three-row
  tests cover unavailable/busy exclusion, native FIFO/commit and identity
  rebinding after ordinal changes. This has not been installed in the production
  canvas, and snapshot/provider availability is not validated OpenGL capability.
- Native Control/Telemetry types and graph binding are implemented and tested.
  The existing module format retains UUID plugin IDs; explicit stopped-graph
  restoration verifies actual factory-instance ownership and exact typed FIFO
  metadata. Automatic persistence of canonical execution/stack requirements,
  production bootstrap, shared automation producers/consumers and monitoring
  display bindings are not installed or validated. See
  `docs/NATIVE_GPU_INVENTORY_ROWS_2026-11-11.md`. This is integration in existing mechanisms,
  not a need for another scheduler or FIFO system.
  Setup-created owners also need the initial Query on the Sequential manager lane
  (or an explicit quiescent lane claim) before admission. Source lookup/retention
  and engine commit/outcome retirement remain caller adapters.
- Telemetry exposes this harness's occupancy and operation results, process/job
  peaks and requested/observed adapter identity. Whole-GPU utilization, external
  process load and globally idle availability are unknown; Vulkan budget queries
  do not supply these values or establish GL affinity.
- The row accepts link-only packages and tears down the child after its receipt.
  It does not retain cross-process GL handles, execute stage schedules, or own
  the producer's scene buffers. Full producer/consumer execution remains separate
  from routing a shader package and cannot be inferred from these CPU checks.

## Verification

CPU-only native MSVC tests cover exact source bytes, object packaging, link-only
cleanup, same-program uniform state, preflight failure, persistent resources,
barriers, partial-submission receipts, no replay, native FIFO and raw-stack tool
ports, source ownership, callback rollback, diagnostics and deferred cleanup.
GL entrypoints are call-recording mocks. They are not driver/link/physics results.
The protected-loan test uses the actual native FIFO. The original unprotected
contract regression also passed because this is opt-in behavior.

`tests/opengl_gpu_threadmanager_test.cpp` passed with freshly compiled
ThreadManager and GraphRuntime: setup/caller thread 38044 and persistent manager
thread 39704 were different. Eight typed Control/Telemetry FIFO ticks covered
ContextUnavailable, SupervisionUnavailable, one mock link/release, result FIFO
acceptance and engine commit. No shader dispatch or real GL call occurred.
Its reproducible direct compiler/link response file is
`out/opengl-gpu-threadmanager/build.rsp`; it uses the intact `build-bridging`
dependency libraries, not the partially rebuilt `build-service` dependency tree.
The graph test then passed 10/10 repeated process runs; logs and exit codes are
in `out/opengl-gpu-threadmanager/run-*.log` and `verification.txt`. The four
focused package, row, protected-loan and original contract regressions all
returned zero in the final CPU check at 2026-11-11 21:10:02 UTC, after the
allocation-fault, approval-wait and provider integration fixes; their logs and
receipt are in `build/dot-opengl-native-cpu`.

Build directly into `build/dot-opengl-native-cpu` with the existing MSVC compiler
and existing `build-service/Release/nodus_runtime.lib` and `nodus_tensor_core.lib`.
Put that Release directory on the test process PATH. Do not regenerate the
existing build-service dependency tree merely to run these focused tests.

## Installation is not complete

No live DLL, service, application or canvas configuration has been replaced.
The added ContractNegotiator Entry and Exchange fields change inline shared-header layout.
Before installation, rebuild all actual owners consistently, including contract
service, canvas/static owners and consumers; do not deploy only the new helper
against older class implementations. Focused tests compile these class/header
implementations freshly and use existing native FIFO C ABI libraries. They do
not constitute a full coherent application build or a socket service deployment.
