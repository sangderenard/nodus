# Native Studio storage ownership checkpoint

Historical checkpoint. On 2026-11-11 the user directed removal of the
task-specific stage splitter. Its stage packages, generated stages, preflight,
dependent auxiliary-storage helper and associated test/build references were
removed. The generic OpenGL backend and its independent test source remain.
The auxiliary-layout and v1/v2 results below describe the removed experiment,
not currently available APIs. The retained generic test was not rebuilt or run
after this removal; no compiler, shader link or renderer was invoked.

This is implemented native source and CPU-tested ownership, not an admitted
shader execution path. The existing protected link-only child is unchanged: it
destroys its programs and context before returning a receipt. Its historical
hardware links did **not** exercise this new memory backend. No live DLL was
replaced, CMake regenerated, service reloaded, or renderer invoked for this work.

## Canonical memory backend

`include/mem_backend_opengl.h` and `src/mem_backend_opengl.cpp` implement actual
OpenGL buffer ownership through the existing `gp_mem_backend_vtable_t` and the
single shared `mem_backend_internal::register_external_handle` registry. The
old `create_from_params(GP_MEM_BACKEND_OPENGL)` path was a `GenericBackend` with
no allocation/resize/free hooks. A GLuint/context-id blob did not establish
usable storage. That unowned path now refuses; ordinary CPU callers do not need
new callbacks and keep their existing behavior.

The existing filesystem manifest writer also refuses non-filesystem backends
before opening its output. Its path-string format cannot serialize a GLuint,
Vulkan buffer, or CPU allocation pointer; treating those as strings was unsafe.

`gp_mem_backend_create_opengl` borrows the existing registered
`gp_gl_compute_context` on the caller's manager lane. It does not discover a
GPU, create or activate a context, or infer GL identity from Vulkan ordinals.
The provider/context owner must already have established its device identity.
The explicit limits intersect the measured per-block limit with the owner's
reserved total bytes and buffer count. They do not claim globally free VRAM.

The canonical vtable provides bounded, zero-initialized allocation, exact
host upload/readback, same-owner native D2D copy and checked release. It does
not advertise host mapping, resize, synchronization events or shader execution.
Extents remain immutable. No GLuint is interpreted as a host pointer. Binding
errors are checked before any dependent operation, so a failed bind cannot
modify the previously bound allocation. Overlapping copies, foreign owners,
wrong lanes, wrong contexts, missing callbacks and invalid extents refuse.

Release has an explicit checked API. The ordinary void release delegates to it
without first removing registry identity. Failed deletion leaves the original
handle registered for cleanup on its actual owner. Failed allocation retains an
undeleted driver name in a pre-reserved cleanup list and blocks new allocations.
The root cannot be released while live or pending buffer ownership remains.
Cleanup collection never retries the original allocation or work.

`gp_mem_backend_opengl_validate_owner` provides CPU preflight proof of the
exact buffer/root shared owner, callback set, registered context and manager
lane. It does not expose a cross-process handle. The existing context registry
has no registration-generation counter: its caller must not recycle a context
id/native handle while live allocations or programs still refer to it.

## Existing field allocation, four nonoverlapping views

`include/studio_auxiliary_storage.h` and
`src/kernels/studio_auxiliary_storage.cpp` connect this canonical allocator to
the actual Studio binding-23 layout. No extra SSBO block is introduced.

For `V = dimensions.x * dimensions.y * dimensions.z`, `L = layers`,
`D = original dispatch capacity`, and `B = original bounce budget`:

- Visits occupy `8 * V` bytes from byte zero.
- Original scalar columns occupy `16 * V * L` following the visits.
- Original response header/rows occupy `8 + 512 * D * B` following the columns.
- Optional authored continuation begins after 16-byte alignment of that exact
  high-water mark. V1 reserves `512 * D` bytes; V2 reserves `1024 * D` bytes.

The actual v1 `checked_auxiliary_layout` and v2 `checked_layout` APIs validate
their tail ABI. Proposed extents are not called existing allocations: after
allocation, the helper checks the actual registered size again. All arithmetic,
block limits, uint32 view boundaries and signed GLSL array-length limits are
checked. Unsupported ABI versions refuse.

Absent `evaluated_responses_enabled` returns an empty plan and allocates
nothing, without consulting a GL owner. The ordinary unsplit observer uses
continuation version zero. Reserving v1/v2 storage does not admit either source
candidate; `GP_GL_UNVERIFIED_CONTINUATION` is unchanged.

The owner returns a borrowed binding descriptor for the entire original SSBO
at slot 23. Individual clears affect only the selected visits, scalar-column,
terminal or continuation range, preserving adjacent views and padding. Views
never delete the underlying buffer. Failed owner release retains its resource.
No helper inserts a dispatch, clears between stages, reseeds rays, compacts
queues, changes uniforms or supplies physics.

## Production handoff still required

These APIs are registered as source dependencies in `nodus_tensor_core`, with
CPU tests in CMake; the legacy canvas source glob excludes duplicate copies.
Those CMake edits have not been configured/built into the running service.

The future protected-session owner must create this backend **inside the
existing context-owning child**, retain that context and manager lane, pass exact
producer bytes into canonical native allocations, and finish checked cleanup
before context teardown. Only opaque session/resource descriptors may cross the
process boundary. A link receipt, host ID or GLuint from the exited child is
not an executable resource. The existing typed row/ContractNegotiator/
ThreadManager path remains the sole scheduling mechanism.

This checkpoint does not implement the persistent child protocol, native scene
image ownership, producer-to-child buffer transfer, complete family scheduling,
numerical admission, or camera/material completeness. The prebuilt producer
still only pumps forward work and does not drain the native reactive queue.
Camera volume integration still has its documented air-only/authored-zero-alpha
limits. Raw forward power is not application RGB. The evaluated-responses
worker hold and application-ready=false contract remain unchanged.

## CPU receipts

`out/native-field-storage/build.cmd` builds the actual changed host registry,
native backend, auxiliary owner, and both frozen stage layout implementations
using MSVC C++20 `/W4 /Gy`. `build-test.log` records PASS. No actual GL function
or device was used: test procedures retain byte arrays and injected failures.
The pre-existing tensor-core import library resolves the host CPU vtable's
SpirvTranslator symbols, not the new ownership implementation. No live binary
was overwritten.

Tests cover exact allocation bytes, zero initialization, canonical type/extent/
capabilities, D2D copying, bind failures preserving old allocations, range and
overflow refusal, ownership/context/lane mismatch, release failure and retry,
pending cleanup, no-hook/legacy CPU behavior, absent opt-in, v1/v2 tails and
terminal clear preserving all adjacent views.

Additional unchanged producer/consumer gates:

- Spectral `test_studio_scene_responses.py`, `test_studio_response_factory.py`,
  `test_studio_forward_responses.py`, `test_studio_response_lineage.py`: 56 passed.
  Shader checks in that suite use the installed offline syntax validator only.
- Node `meowdoku_scene_responses_check.mjs`,
  `meowdoku_secondary_field_check.mjs`, and
  `meowdoku_studio_light_field_check.mjs`: passed CPU-only exact copy,
  availability/fallback and orchestration tests.
- Broader `meowdoku_studio_scene_builder_check.mjs`: failed its captured-scene
  hash, actual `4585b7f98c74413ff67276d273e67aa045f63b6ddb45cd8ca1c94523c6f8bf1d`
  versus expected `be5f1adbb4998704bfe24f81f4683c0066d1ed48682adc9e03442077e0696216`.
  No scene-builder file was changed in this storage work. The cause and baseline
  status remain unestablished; the broader suite is not called passed.

Logs are `producer-cpu-regressions.log` and `consumer-cpu-regressions.log` in
the same directory. This is not a full-repository, live producer, hardware,
numerical parity, performance or end-to-end rendering acceptance receipt.
