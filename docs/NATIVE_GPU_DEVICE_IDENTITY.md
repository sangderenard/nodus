# Native GPU inventory and row-owned contexts

## Boundary

GPU work belongs to native `ITool` rows in the existing canvas runtime. The
row owns its harness/context; existing tool ports, `EdgeTensorFifo`, and
`ThreadManager` route and schedule work. The functions below extend the
existing `kernel_vulkan_runner` backend. They do not introduce another queue,
scheduler, automatic retry policy, renderer, or device-selection authority.

**Vulkan inventory is not a list of live OpenGL contexts.** A Vulkan-capable
adapter cannot accept an OpenGL program merely because its name matches.
OpenGL source requires a verified current GL context/harness in its owning
GPU row. Studio's earlier native GLSL 4.30 linker failure was on the NVIDIA
RTX 3060 context, not evidence that the AMD integrated GPU was selected.

## Interfaces for GPU rows

Declared in `include/kernel_vulkan_runner.h`, implemented in the existing
`src/kernels/kernel_vulkan.cpp`:

- `enumerate_vulkan_devices`: raw Vulkan enumeration ordinal, name, vendor and
  device IDs, API and driver versions, UUID/LUID availability, compute queue,
  and storage-buffer limits. Includes devices without a compute queue so
  filtering cannot silently renumber adapters.
- `query_vulkan_device_status(index)`: instance/query-only identity and memory
  heap snapshot. No logical device, buffer, shader, command pool, or workload
  is created by this query or enumeration.
- `create_vulkan_backend(params, selected, error)`: creates the existing
  persistent Vulkan backend and returns its actual selected identity.
- `query_vulkan_backend_status(handle)`: queries that context's physical
  device, preserving identity for its persistent buffers.
- `dispatch_spirv_on_device`: exact-index version of the existing host-buffer
  runner. The ordinary `dispatch_spirv` entry retains its default selection.
  Persistent-buffer callers still use `dispatch_spirv_backends`; they cannot
  migrate buffers or replay work by changing an index.

The C creation API remains `gp_mem_backend_create_vulkan`, with the existing
`{device_index, flags}` parameter layout. `gp_mem_backend_vulkan_last_error`
provides the last creation diagnostic on that thread. A failed creation
returns null; no C++ exception escapes this C entry. Errors identify the
selected device, failed Vulkan operation, result code, and named memory or
driver failure where available.

An index is local to a particular `vkEnumeratePhysicalDevices` snapshot. It
is **not** a display, Win32_VideoController, DXGI, CUDA, or GL index. A row
that persists identity should compare the newly created context's UUID/LUID
with the intended identity before allocating/submitting work. Selection by
stable UUID/LUID is not added in this patch.

`UINT32_MAX` keeps the existing first-compute-device default. Explicit indices
are exact: out-of-range, noncompute, and context-initialization failures are
reported rather than trying another adapter. Reserved flags remain zero.
There is no automatic alternative-device retry or CPU fallback, and no replay
after shader compilation, queue submission, or readback failure.

## Telemetry limits

Heap size is a 64-bit physical capacity. `VK_EXT_memory_budget`, when present,
supplies driver-estimated application heap budget and usage. The caller must check
`budget_available`; zero fields when absent do not mean zero utilization.
These values are not GPU-busy percentages or process/system RAM consumption.
Win32_VideoController's 32-bit `AdapterRAM` is not used for admission.

Enumeration alone is not proof that context initialization, a particular
shader's limits/features, or its driver compiler will succeed. No thermal,
utilization, fair-sharing, best-open-device ranking, or row availability
policy is implemented here. Those decisions belong to the native GPU row and
the existing runtime's routing/ports.

Normal C++ exceptions and returned driver errors can be reported gracefully.
An in-process driver crash or system RAM exhaustion is not made safe by a
try/catch. Existing protected shader-link memory limits remain necessary;
this patch does not add process isolation.

## Verification (2026-11-11)

- `tests/vulkan_device_query_test.cpp` compiles the current backend against
  mocked inventory/context operations. It checks raw ordinals, noncompute and
  invalid selections, UUID/LUID, limits, bounded changing-inventory handling,
  optional memory budgets, exact-index creation, named OOM/driver errors,
  partial-context cleanup, exception containment, and no implicit retry.
  Shader creation, buffer creation, pipeline creation and queue submission
  abort if accidentally reached. Passed with MSVC 19.44, C++20.
- `tests/vulkan_unavailable_test.cpp` compiles the same source without Vulkan
  support or driver linkage and verifies explicit unavailable failures. Passed.
- The initial regression pass did not query real hardware. The separately
  authorized, instance/query-only integration receipt below verifies actual
  Vulkan inventory. Logical-device creation, shader compilation/link,
  dispatch, GL adapter selection, and rendering remain unrun.
- No live library was replaced, server restarted, registry changed, or driver
  installed. Full-project build and live GPU-row integration remain separate
  verification steps.
- Attempting only the two new CMake test targets in `build-service` triggered
  automatic vcpkg dependency regeneration. That build was stopped after
  verifying its process tree rather than continuing unrelated rebuilds. Its
  private `build-service/vcpkg_installed` tree needs dependency regeneration
  before a later full build. Existing Release DLLs were not rebuilt. The
  standalone MSVC test builds do not depend on that tree.

To resume that existing build tree when a full native build is wanted, ensure
no other regeneration is running and use the same manifest/install roots:

```powershell
& 'C:\vcpkg\vcpkg.exe' install --triplet x64-windows --vcpkg-root C:/vcpkg --x-wait-for-lock --x-manifest-root=C:/dev/Powershell/nodus --x-install-root=C:/dev/Powershell/nodus/build-service/vcpkg_installed
```

Let vcpkg reconcile its existing `vcpkg/updates` journal. Do not delete that
journal, trust the stale base `status` file alone, or copy packages manually.
After successful restoration, build the intended targets normally. At the
stop point ten dependencies were absent and `vcpkg-cmake` had been restored.
The owned build process tree was verified stopped. The existing Release
`nodus_tensor_core.dll` retained its 2026-11-10 timestamp and 5,068,288-byte
size. This recovery was documented, not executed.

## Actual physical-query receipt

One opt-in native integration invocation at **2026-11-11 20:46:09 UTC**
returned exit 0 in 1,126 ms (30-second process timeout, not reached).
`tests/vulkan_inventory_integration_test.cpp` calls the current runner APIs;
its executable-local guards terminate if logical-device creation, buffers,
memory allocation, command pools, shaders, pipelines, submission or dispatch
is attempted. Instance wrappers forward to the real Vulkan loader and count
cleanup: **four created, four destroyed**. No GPU workload was submitted.

It was built directly with installed MSVC/Vulkan SDK and existing unchanged
`build-service/Release/nodus_tensor_core.lib`, without CMake or vcpkg. Build
arguments: `build-device-query/inventory-build.rsp`. Native output:
`build-device-query/inventory-20261111T204609Z.stdout.log` (stderr empty).

| Vulkan raw ordinal | Adapter | PCI vendor:device | Device UUID | LUID bytes |
| --- | --- | --- | --- | --- |
| 0 | NVIDIA GeForce GTX 1660 SUPER | 10DE:21C4 | 348c2d2a7e8cc8b93a556aa4c1cf2715 | d902010000000000 |
| 1 | AMD Radeon(TM) Graphics | 1002:164E | 000000000e0000000000000000000000 | 3413010000000000 |
| 2 | NVIDIA GeForce RTX 3060 | 10DE:2504 | b0230bdae57fb7f1d2a850d5ea426da5 | 22f1000000000000 |

All three advertise a compute queue in family 0, valid UUID/LUID, and
`VK_EXT_memory_budget`. These are snapshot identities, not permanently
assigned ordinals. The query rechecks identity before associating each memory
snapshot with its adapter. Raw limits and every 64-bit heap measurement are
in the output receipt. Vulkan storage limits must not be substituted for GL
limits. Low/zero usage in this fresh query process is not evidence that the
whole GPU is idle, and budgets do not establish row availability.

Read-only Windows inventory listed RTX 3060, GTX 1660 SUPER, then AMD Radeon.
The different Vulkan ordering demonstrates why Windows list positions cannot
be reused as Vulkan ordinals. This receipt still establishes **no OpenGL
adapter affinity or live GL context**.
