# Three-dimensional wave stencil on Nodus GPU rows (2026-11-13)

The independent Turing `wave_volume_3d_step_f32` product is a centered,
damped scalar wave update for the interior of a 3D grid. The Nodus trial uses
the existing `GpuShaderJobTool` and `TensorStateLibraryTool`, with a full
64×64×64 voxel volume (262,144 shader invocations) and an initially stationary
central impulse. All three Vulkan devices returned the exact f32 center value
0.625, all six axial-neighbor values 0.0625, and zero elsewhere within the
test's tolerance. The executable exited zero twice.

The first process run reported 608.952 ms, 11.6943 ms and 12.2801 ms for
devices 0/1/2. A second process run, after driver shader cache creation,
reported 1.3472 ms, 2.472 ms and 1.5197 ms. These are host dispatch receipts
for one volume per device, not sustained throughput, GPU-only time, or an
apples-to-apples speed comparison with the 2D angular-spectrum solver.

The product has fifteen f32 feed buffers: center/previous state, six spatial
neighbors, spacings, time step, speed, damping and source. The trial constructs
the neighbor buffers from a known 3D impulse and validates the compiled kernel
on device. A production 3D arena still needs persistent two-time-level state,
voxel-neighbor address mapping, authored boundary conditions, a stability gate
using the 3D Courant number, and exchange with Nodus's tensor state library.
This specialized stencil is distinct physics from Spectral T4's transverse
2D angular-spectrum optical propagation; running one does not validate the
other or make the thick-lens aperture handoff complete.

Reproduction (after building `test_spectral_wave_vulkan` from this branch):

```powershell
python -m src.compiler.wave_volume_3d --output C:\Users\alber\nodus-build\wave-volume-3d-artifact
C:\Users\alber\nodus-build\wave-volume-cmake\Release\test_spectral_wave_vulkan.exe volume3d C:\Users\alber\nodus-build\wave-volume-3d-artifact\wave_volume_3d_step_f32.spv
```

The first command runs from the Turing `codex/wave-volume-3d` checkout; the
second runs the native Nodus test from this branch.
