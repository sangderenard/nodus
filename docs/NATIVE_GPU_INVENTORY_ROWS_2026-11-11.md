# Native inventory registration for GPU rows

This is a source/CPU-verified registration step, not a production bootstrap or
proof that hardware is idle. It performs no enumeration, process launch, context
creation, shader work, rendering or deployment.

## Existing mechanisms used

`include/gpu_inventory_rows.h` and `src/gpu_inventory_rows.cpp` consume supplied
`VulkanDeviceInfo` snapshots and explicit per-device native configuration. Every
configured device gets one ordinary `OpenGLGpuTool` factory and Peer. The factory
ID contains the full 128-bit device UUID, not a display name or Vulkan ordinal.
The caller provides the persisted contract host ID. Both identities must be
distinct and nonzero. Registration is bounded to 64 rows; snapshot names are
bounded to 1024 bytes.

The existing `contract_negotiator_registry` gains `create_with_offer`. It creates
a new process-owned market/worker pair with the supplied domain eligibility
policy and refuses an existing prefix. Ordinary `acquire` still uses its original
default and can subsequently find a custom pair without replacing its policy.
No extra registry authority, work queue, scheduling thread or load ranking is
introduced. The same native result FIFO and engine commit remain authoritative.

`first_suitable_free` operates inside the existing host-turn order. An unsuitable
host does not reserve work, and a row already owning work is not reused. An
unsupported operation can be skipped in favor of eligible work. This is neither
a strict global arrival-order promise across heterogeneous work nor an algorithm
that ranks hardware by speed/load. The existing caller-supplied host order is
preserved, including after explicit configuration reload.

## Provider identity and unavailable devices

The default factory constructs the existing `ProtectedOpenGLProcess`; it does
not start it. Its requested adapter LUID is copied from the supplied device
snapshot. A conflicting configured LUID is refused. A node mask must be supplied
explicitly; this layer does not invent one or treat a Vulkan compute capability
as OpenGL capability.

Missing LUID, missing provider configuration, missing node mask or a null provider
factory produces a visible unavailable row. Query returns the specific refusal
through the existing typed Telemetry port. `InventoryRowStatus` separates the
snapshot UUID/LUID/ordinal, structural provider availability, harness occupancy
and operation receipt. Physical utilization is always unavailable here.

Provider availability is structural admission, not execution authorization or
hardware validation. Exact source/loan approval, one-shot start, identity checks,
containment, cancellation, cleanup and no-replay behavior remain in the existing
row/provider. The per-device provider construction callback is the integration
point for future reviewed native resource/session ownership, not another factory
registry or scheduler. CPU tests inject providers that never execute anything.

## Persistence and automation boundary

`library_tool` and `library_module` return the existing `GP_ModuleLibraryTool` and
`GP_ModuleLibraryModule` configuration types. Their existing writer/reader retain
the UUID-keyed plugin ID. Tests save/read this native format, reorder inventory
and change Vulkan ordinals, then rebind the same physical UUID and host identity.
No GPU-specific save format was added. Provider objects, approvals, in-flight
work and GL handles are never serialized.

`bind_module` uses the existing typed GPU graph binding. `restore_module` is an
explicit operation on a stopped, freshly loaded native GraphRuntime. It verifies
the saved Input/Plugin/Output program, current factory, exact FIFO types/extents
and capacity, then restores AbstractTensorPointer and Sequential requirements.
Wrong UUIDs and stale numeric type IDs are refused, not reinterpreted.
The registration layer tracks each actual factory-created instance until its
deleter runs. Editing only the saved UUID while swapping in another GPU's tool
cannot satisfy restoration. Explicit close refuses live bound GPU instances.

Inspection found these existing interfaces:

- CANVAS `ROWS` persists plugin IDs and recreates tools through ToolRegistry.
- Headless `EDGE_FIFO`, `PORT` and `CONTRACT_PORT` records persist transport and
  named automation endpoints, using the same table-edge FIFOs.
- Native Control and Telemetry are the row's existing automation payloads.

Still required for production:

- Bootstrap these factories in the same native image/registry before loading the
  canvas. No live GUI or service has been reloaded or patched.
- Restore canonical execution/stack metadata. Current canvas save/load does not
  persist `module_exec_mode` or `module_stack_contract`; this patch does not claim
  an automatic persistent reload. A general canonical persistence hook remains
  needed. ValueType IDs are process-local, so transport restoration must verify
  their current meaning and extent before accepting samples.
- Bind the shared automation producers/consumers and monitoring display through
  existing named ports. This patch does not install a new public automation API
  or a GUI monitoring panel. Initial Query/lane adoption must precede admission.
- Provide immutable source lookup/retention and engine commit/outcome retirement
  adapters. Complete shader execution and scene-buffer lifetime are separate from
  this link-only row registration.

Each row belongs to one canonical manager lane. Do not bind one participant into
multiple independently running GraphRuntime managers. Stop/destroy bound graph
instances before explicit `close`. Close refuses unfinished contracts, busy or
wrong-lane ownership rather than dropping replay protection. An abandoned C++
registration object does not silently unregister its process-owned canonical
participants; the caller must retain its configuration/adapters and close it.

## CPU verification

`tests/gpu_inventory_rows_test.cpp` uses real in-image ToolRegistry factories,
market/worker ITools, native result FIFO, module-library serializer and GraphRuntime
binding. It covers distinct identities, exact configured LUIDs, unavailable rows,
unsupported-work exclusion, busy exclusion, accepted-generation hold, callback
removal without replay, pending-close refusal, cancellation/commit, UUID rebinding
after ordinal/order changes and explicit typed/lane restoration. Provider, GL and
dispatch call counts remain zero.

`tests/contract_registry_transaction_test.cpp` injects allocation failure through
native pair creation, including after the first factory is published. Canonical
ownership is allocated first, every failed registration rolls back both newly
introduced IDs, and release uses preallocated keys. Existing foreign factory
collisions remain untouched. Registry diagnostic output cannot turn an inserted
or removed entry into an exception outcome.

Direct build response file: `out/gpu-inventory-row/build.rsp`.
Run output: `out/gpu-inventory-row/run.log`.
The existing uncustomized contract regression is also rerun. CMake target wiring
is source-only; no CMake regeneration or dependency transaction is run.

The frozen protected row/provider/CLI paths are unchanged by this step. A coherent
native rebuild of all changed contract owners is required before installation.
