# Native protected OpenGL link owner

## Scope and current evidence

Implemented 2026-11-11 in native C++. The existing GPU ITool, contract market,
typed FIFOs and ThreadManager remain the work scheduler. A provider owns one
child process and its safety watchdog, not another work queue or placement
policy. The parent/service never creates a GL context or links the submitted
shader on this path. The worker only links and releases; it has no dispatch,
draw, render, exposure or readback operation.

The Windows process containment fixture passed with real child processes and
**no OpenGL/Vulkan dependency**. After the initial CPU-only implementation and
registry-sentinel checks, one explicitly authorized `studio.initialize.v1`
link succeeded on the requested RTX 3060 through the real protected path at
2026-11-11 21:32 UTC. The separately authorized nearest-hit link also succeeded
at 21:34 UTC. The separately authorized advance link failed with driver C9999
out-of-memory at 21:35 UTC; cleanup and process exit were verified. All receipts
are recorded below. No further hardware attempts are authorized. There has
been no dispatch, rendering, exposure, live deployment or service restart.

## Files and interfaces

- `include/opengl_protected_process.h`: nonblocking provider, exact request,
  bounded receipt, authorization and protection policy.
- `src/kernels/opengl_protected_process.cpp`: native parent process owner.
- `src/kernels/opengl_protected_protocol.h` and
  `opengl_protected_child.cpp`: private, versioned local memory-section protocol.
- `src/kernels/opengl_protected_wgl.cpp`: child-only exact-affinity context owner.
- `src/kernels/opengl_link_worker_main.cpp`: production executable entry.
- `tests/opengl_protected_process_test.cpp`: separate CPU fixture executable.
- `include/opengl_gpu_row_tool.h`: the existing row's provider binding; its
  canonical graph/FIFO wiring is in `opengl_gpu_graph_binding.h`.

The provider exposes `available`, `authorized`, `link`, `poll`, `cancel` and
`poll_cleanup`. `available` is structural admission, not proof of actual GL
support. `ready` additionally reports whether an explicit one-link approval has
been armed. `link` returns Running after bounded source copy and native owner
startup. Later control ticks poll; they do not wait for the driver.

## Authorization and ordering

1. Configure one row/provider for an exact Windows adapter LUID and node mask,
   and an absolute path to the purpose-built native worker.
2. Structural availability allows the canonical contract system to assign a
   loan generation. GPU row admission holds that exact generation while waiting
   for approval; it does not authorize a link.
3. Obtain explicit approval for this one real link. Compute the exact labeled
   source-object SHA-256 and arm `authorize_once` with host, contract and loan
   generation. The engine must provide the actual queued generation, not guess.
4. Advance checks `authorized` before submission. No approval leaves the work
   queued, with no process or GL call. The provider rechecks and consumes the
   exact approval itself on submission. Changed source/loan also consumes it.
5. Link failure, timeout, cancellation, crash and invalid receipt never trigger
   a retry, alternate device, source rewrite or replay. A new real attempt
   requires a new explicit authorization.

Authorization capture by a live user-facing engine is not installed here.
There is no automatic provider installation into the running canvas service.
The explicit standalone development entry below can bind a source/target
approval to its actual canonical queued generation without guessing it.

## Verified process containment

The provider creates an unnamed JobObject, sets process and aggregate job memory
limits, kill-on-close and an active-process limit, then queries back the exact
configuration. The child is created with CREATE_SUSPENDED and CREATE_NO_WINDOW.
Assignment, exact job membership and limits are verified before an independent
watchdog is armed and ResumeThread is called. Failure before resume terminates
and reaps the suspended process; it never executes the shader worker.

Production policy refuses process/job caps above 4 GiB, a system available-RAM
floor below 8 GiB, or a wall deadline above 90 seconds. The watchdog checks the
child, deadline and RAM floor at most every 100 ms (default 25 ms). Cancellation
terminates the same owned job. Termination is followed by observed process exit;
unconfirmed exit retains process and job handles, blocks new admission, and can
only be reclaimed with `poll_cleanup`. Allocation/exception paths use the same
ownership rule and an unconditional watchdog stop/join guard.

The current active-process limit is **one**. Descendants are rejected rather
than escaping aggregate limits. This is stricter than the earlier Python-based
protected attempt, whose job accounting reported three total processes without
recording their identities. Native driver helper-process compatibility is
therefore unproven and helpers are currently unsupported. Do not silently raise
this limit if a driver needs them; report the refusal and obtain authorization
for a deliberately revised policy, retaining aggregate limits and lifetime
supervision.

This is process/resource containment, not a security sandbox for malicious
executables and not protection against every kernel/driver or system-wide GPU
failure. The configured worker must be the trusted native build.

## Bounded local transport and receipts

Only two unnamed memory sections are inherited through an explicit Win32 handle
list: read-only request and writable fixed-size result. No parent service
handles or JobObject handle are inherited. Sources are never uploaded or stored
as temporary shader files. Bounds are 16 MiB including labels, 4096 objects, and
a fixed result with a 2048-byte diagnostic. Sources retain object boundaries and
byte counts, including embedded NULs in source bytes.

The child validates section size, version, ranges, exact loan, source digest and
its own OS job limits before the native context callback. The parent accepts a
result only after observed child exit and validates the receipt's exact loan,
digest, status, flags and requested/observed identity. No process-local GLuint
or native context pointer is returned as a transferable resource.

`resources_released` means the child exit was observed (or no child was created).
`gl_cleanup_verified` separately means explicit program/context release was
reported. A killed/crashed child can have its process resources gone while GL
cleanup and link outcome remain unknown. A linked success requires explicit GL
cleanup, verified identity, containment, child exit and a non-uncertain receipt.

## Exact GL identity and fail-closed limits

The child creates an invisible bootstrap solely to load WGL entrypoints.
Selection is explicit. `nv-affinity` requires WGL_NV_gpu_affinity, maps its
GPU_DEVICE GDI display names to Windows DisplayConfig source adapter LUIDs,
and creates an OpenGL 4.3 core affinity context for the exact requested LUID.
`verified-default-dc` creates one OpenGL 4.3 core context on the ordinary DC.
Both require GL_EXT_memory_object_win32 and exact equality of actual
GL_DEVICE_LUID_EXT and GL_DEVICE_NODE_MASK_EXT before the first shader call.
The default-DC policy does not claim hardware affinity support. It is accepted
only when the actual context proves it belongs to the requested row's GPU.

Missing proof or mismatched identity refuses under either policy. NV affinity
also refuses missing/ambiguous display mappings and unsupported affinity.
There is no automatic policy switch, device retry, name comparison, WMI order
or Vulkan-ordinal-to-GL inference. The default-DC path never retries a mismatch
on another adapter. The initialize receipt verifies the default-DC identity
path on this RTX 3060 and node mask 1. It does not establish NV affinity or
identity support for other devices.

The worker registers, queries and unregisters its own context through the same
`nodus_tensor_core.dll` registry. The standalone worker import table confirms
all three functions come from that DLL; it does not link `nodus_runtime`'s
separate registry implementation. The current `kernel_opengl.cpp` is linked
directly into the worker and receives no parent-owned GL object.
The real worker's `--verify-registry-only` sentinel check passed using the
actual linked DLL: registration, lookup, removal and subsequent absence, with
no context or shader call. Receipt: `out/protected-gl/registry-verification.txt`.

## CPU verification and reproducibility

From `C:\dev\Powershell\nodus`, using installed MSVC only:

```powershell
cmd /d /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cl @out\protected-gl\build.rsp'
& .\out\protected-gl\opengl_protected_process_test.exe
cmd /d /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cl @out\protected-gl\worker-build.rsp'
```

The last command builds the real worker only. **Do not enter its protected GL
path without the specific next real-link authorization.** Its explicitly named
registry-only branch is CPU-only. The CPU test never links that WGL
implementation. `NODUS_PROTECTED_PROCESS_TEST_HOOKS` is compiled only into the
test executable, never the production library or worker.

Receipts are in `out/protected-gl/cpu-verification.txt`, build response files,
and `worker-imports.txt`. Tests cover nonblocking start/poll/cancel, preapproval
refusal, structural admission without a future generation, exact source/loan
binding, one-shot consumption, transport bounds, observed child exit,
allocation refusal under a verified 64 MiB cap, descendant refusal, malformed
identity receipt, crash, timeout, cancellation, and injected supervisor
exceptions both before job assignment and after resume.

`tests/opengl_protected_row_threadmanager_test.cpp` also passed the complete
canonical CPU path: real ThreadManager manager lane, typed Control FIFO, GPU
row, exact-generation approval wait, real protected CPU child, typed Telemetry
FIFO, existing contract result FIFO and one engine commit. The independent
consumer and manager continued ticking while the child ran. This fixture traps
any parent-context access and asserts zero mocked compile/link/dispatch calls.
Its build arguments and receipt are in `out/protected-gl-row/`. It links native
runtime libraries but never invokes their GPU paths. This is native transport
and supervision proof, not a successful shader-driver link.

CMake targets are registered but no CMake configure/build/regeneration was run.
The existing build-service dependency-tree recovery remains outstanding, as
does a coherent rebuild of all users of the changed contract/telemetry layouts
before any deployment. Existing Release DLLs were not replaced.

## Prepared one-link native development entry

`src/opengl_link_once.cpp` builds `nodus_opengl_link_once`. It creates a bounded
three-module native graph using `bind_opengl_gpu_module`: Control producer,
GPU row, and Telemetry consumer. Existing ThreadManager executes the row on
its Sequential manager lane and the existing ContractNegotiator owns the exact
loan/result/commit. The command has no renderer or dispatch mode, and never
invokes the protected worker directly.

`--prepare` reads and hashes the source only, before any graph/provider/process
is constructed. `--link-once` requires both `--approve-one-link` and the exact
`--source-digest`. These switches are a caller attestation, not a substitute for
obtaining the user's explicit approval immediately before the actual attempt.
After canonical queue admission, the entry binds that source commitment to the
actual loan generation once. It reports failures and exits, never retries.
Ctrl+C requests cancellation of that same child. The parent observation loop
is bounded at 105 seconds, in addition to the child's 90-second watchdog.

Built with `out/protected-gl-once/build.rsp`; no CMake regeneration. Each of
initialize, nearest-hit and advance has completed one separately authorized
attempt. Initialize and nearest-hit linked; advance failed in the driver.
Preparation receipt: `out/protected-gl-once/prepare-initialize.txt`.

Frozen first input:

- File: `build-studio-stages/package/0-studio.initialize.v1.comp`
- Object label: `studio.initialize.v1`
- Bytes: 107061
- File SHA-256: `0c6341eff6ff640775cf353af3f4e0346b784ae9506e6b61cba167a7313f22b9`
- Exact labeled-object commitment:
  `add46deee2138c75c3dc6c2baaac1475b87ae3be13f49a858783316ea0d175e6`
- Offline syntax receipt: `build-studio-stages/package/offline-syntax-receipt.txt`,
  official glslang 16.0.0 `-S comp`, exit 0, no diagnostics.
- Requested RTX 3060 Windows LUID: `22f1000000000000`, from the earlier authorized
  inventory. Node mask 1 was subsequently verified by the successful initialize
  child; every new child must verify both values again before shader calls.

The labeled-object commitment is intentionally different from the raw file
hash. It commits to the source-object count, label, byte count and unchanged
source bytes. The emitted stage still declares unverified continuation and is
not authorized for execution by a successful link.

The following is the **historical initialize command, executed once**, using
the explicit identity-verified default-DC policy. It is preserved for
provenance, not permission to repeat it. It retains fixed 4 GiB process/aggregate caps, 8 GiB RAM floor,
90-second watchdog and one-process policy:

```powershell
$env:PATH = 'C:\dev\Powershell\nodus\build-service\Release;C:\dev\Powershell\nodus\build-bridging\vcpkg_installed\x64-windows\bin;C:\Users\alber\AppData\Local\Programs\Python\Python311;' + $env:PATH
& 'C:\dev\Powershell\nodus\out\protected-gl-once\nodus_opengl_link_once.exe' --link-once --approve-one-link --source-file 'C:\dev\Powershell\nodus\build-studio-stages\package\0-studio.initialize.v1.comp' --source-label studio.initialize.v1 --worker 'C:\dev\Powershell\nodus\out\protected-gl\nodus_opengl_link_worker.exe' --adapter-luid 22f1000000000000 --node-mask 1 --context-policy verified-default-dc --host-id 3060 --contract-id 2026111101 --source-digest add46deee2138c75c3dc6c2baaac1475b87ae3be13f49a858783316ea0d175e6
```

The Python directory is only an existing native DLL dependency search path for
the linked canvas runtime, not a Python coordinator or interpreter invocation.
The command may legitimately return a known affinity/identity refusal without
ever reaching shader compilation. Identity was verified in all three attempted
stages, but only initialize and nearest-hit linked successfully.

## Authorized stage results

The first real initialize attempt completed on 2026-11-11 at 21:32:41 UTC:

- Contract 2026111101, generation 1, child PID 77860.
- Child elapsed 766 ms; outer command elapsed 906 ms; exit 0.
- Peak process commit 176197632 bytes; peak job commit 177442816 bytes.
- Requested/observed LUID `22f1000000000000`, verified-default-DC selection.
- Identity, containment, child reaping, explicit GL cleanup and process-resource
  release all verified; outcome not uncertain.
- Exactly one contract commit, no retry, zero dispatch calls, execution held.

Receipt directory:
`out/protected-gl-once/initialize-attempt-20261111T213240330Z/`
(`native-link.log`, `exit.txt`). This proves this source linked and was released,
not a rendered result, continuation correctness or shader execution.

The next two source-object commitments were each attempted once, with unchanged
binaries, the same target/context policy and all original limits:

- Nearest-hit: label `studio.nearest_hit.v1`, file
  `build-studio-stages/package/1-studio.nearest_hit.v1.comp`, 106821 bytes,
  raw SHA-256 `556b3e8bb66ce35075c3eb5df72a22284d4efbd40a6f45486624f70423565100`,
  contract 2026111102, exact approval digest
  `71e437b642151b12a4d4c3a5e470d4426036b3ed5c730f8bf711b5f53cb48fef`.
  Frozen commands: `out/protected-gl-once/frozen-nearest-hit-link.txt`.
- Advance: label `studio.advance.v1`, file
  `build-studio-stages/package/2-studio.advance.v1.comp`, 122116 bytes,
  raw SHA-256 `67cc5b45fb1099f48252d9c194fa7e9c4e29b2b735d624661cb7ca878730cdda`,
  contract 2026111103, exact approval digest
  `7ed8946ca781955165022b224cd13b8f92fd88034c563117335ab7ded8b572b6`.
  Frozen commands: `out/protected-gl-once/frozen-advance-link.txt`.

The command files now preserve historical attempts, not permission to repeat
them. Prepare receipts remain `prepare-nearest-hit.txt` and
`prepare-advance.txt` in the same directory.

### Nearest-hit: successful link

Completed 2026-11-11 at 21:34:45 UTC, contract 2026111102, generation 1:

- Child PID 76980, elapsed 922 ms; outer command elapsed 1058 ms.
- Runner exit 0, child exit 0; exactly one submission and commit.
- Peak process commit 178589696 bytes; peak job commit 179847168 bytes.
- Exact requested LUID, identity, containment, reaping, explicit GL cleanup and
  process-resource release verified; known successful outcome.
- Zero dispatch calls, execution held, no retry.

Receipt: `out/protected-gl-once/nearest-hit-attempt-20261111T213444093Z/`
(`native-link.log`, `exit.txt`).

### Advance: known driver link failure, resources released

Completed 2026-11-11 at 21:35:54 UTC, contract 2026111103, generation 1:

- Child PID 58720, elapsed 5688 ms; outer command elapsed 5813 ms.
- Combined program link returned `fatal error C9999: out of memory - internal
  malloc failed`.
- Runner exit 4 reports the failed link; child exit 0 reports successful
  protocol completion and cleanup. Exactly one submission and failure-result
  commit, with no retry.
- Peak process commit 4293746688 bytes; reported aggregate job peak 4295008256
  bytes. The job peak is **40960 bytes (40 KiB) above** the configured 4294967296
  byte (4 GiB) limit. Do not claim the observed aggregate peak stayed strictly
  below the cap. The reason for this accounting excess has not been established.
- The configured process/job limits, membership and supervision were verified.
  `containment_verified` records those checks, not a claim that every reported
  memory peak is below the configured value.
- Exact requested LUID, identity, child reaping, explicit GL cleanup and
  process-resource release verified. Outcome is a known link failure, not an
  unknown or still-running operation.
- Zero dispatch calls, execution held.

Receipt: `out/protected-gl-once/advance-attempt-20261111T213548964Z/`
(`native-link.log`, `exit.txt`). A compact summary of all three attempts is in
`out/protected-gl-once/completed-link-receipts.tsv`.

No further hardware attempt is authorized. Smaller advance source units are
being prepared separately; the tested source files and native binaries remain
frozen. All stages retain their unverified-continuation execution boundary.
These attempts did not dispatch, render, expose a frame or take a screenshot.

## V2 split preparation and material receipt (2026-11-11 UTC)

The source divider froze the new ABI2/256-word stage package in
`build-studio-stages-v2/package`. Native source tests, official offline syntax
checks and preprocessed source/liveness review passed before preparation.
The unchanged native runner's `--prepare` mode passed for material, segment and
response_flush without creating a graph, child, context or calling a shader API.

- Material `studio.material.v2`: 132574 bytes, raw SHA-256
  `7870fad3695927d30afd07aadcd7f4b36aff6be1c55c6f66201541389cefaf80`;
  contract 2026111104, exact labeled-object approval digest
  `684adbecd44a822f2b80429f0620e6d61c26c24523c51d659d5f9a4ca1d13e46`.
- Segment `studio.segment.v2`: 119974 bytes, raw SHA-256
  `11d42d1698b3fc60509894742832712bb7e2cb67a4dff64553a3ce5fe79ce9b5`;
  contract 2026111105, exact approval digest
  `d29c1e989bc11f30e058beb0578640607d5cb7edad4568f4a51c8307df18b4f9`.
- Flush `studio.response_flush.v2`: 117579 bytes, raw SHA-256
  `553845ada3e09b190d21b5f9ddd794735df97c6f5e7c6467ce61383a1e3eeddd`;
  contract 2026111106, exact approval digest
  `b942d85d6294b969ece1724f6ba4b0605d50225b72a86349aec871063543a267`.

Exact command receipts are `out/protected-gl-once/frozen-v2-*-link.txt`.
The native binaries and original v1 source/receipts are unchanged.

### Material v2: successful link

After a separate root notice and go-ahead, the parent executed material exactly
once. Completed 2026-11-11 at 21:51:17.6681041 UTC:

- Child PID 13880, elapsed 2235 ms; runner/child exit 0.
- Peak host process commit 274366464 bytes; aggregate job peak 275636224 bytes.
- Exact requested LUID `22f1000000000000`, node mask 1; verified-default-dc.
- Identity, configured containment, child reaping, explicit GL cleanup and
  process-resource release verified; known success, one submission and commit.
- Zero dispatches and no retry. Link success does not validate stage execution.

Receipt: `out/protected-gl-once/v2-material-attempt-20261111T215115273Z/`.
Segment and flush remain held pending their own separate notice and go-ahead.
All v2 shaders remain `GP_GL_UNVERIFIED_CONTINUATION`.

A read-only visible console can refresh completed receipts using
`out/protected-gl-once/watch-completed-link-receipts.ps1`. It only reads attempt
directories after a final `exit.txt` exists; it never launches native jobs,
contexts, shaders or rendering. Refresh is bounded to 30 minutes by default.

### Segment v2: known driver link failure, cleanup complete

The parent separately authorized and executed segment once on 2026-11-11 at
21:52 UTC, contract 2026111105, generation 1:

- Child PID 64056, elapsed 4344 ms; runner exit 4, child exit 0.
- Exact driver error: `(0) : fatal error C9999: out of memory - internal malloc failed`.
- Peak host process commit 4293746688 bytes; reported aggregate job peak
  4294991872 bytes, **24576 bytes (24 KiB) above** the configured 4294967296-byte
  cap. This is the reported OS peak, not a claim of strict below-cap usage or
  VRAM exhaustion. The cause of the accounting excess is not established.
- Exact requested LUID, configured containment, reaping, explicit GL cleanup
  and process-resource release verified. Known failure, one submission and
  failure-result commit, no retry, zero dispatch calls.

Receipt: `out/protected-gl-once/v2-segment-attempt-20261111T215220355Z/`.
The read-only visible log view includes this completed failure automatically.
Flush remains held pending its separate notice and go-ahead; no new GPU action
is authorized by this receipt update.
