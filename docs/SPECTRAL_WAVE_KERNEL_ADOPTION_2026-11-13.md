# Compiled wave math through the Nodus GPU row

**2026-11-13 UTC.** Spectral T4 owns the physical equations and arena state in
`spectral-analyzer/csrc/kernels/wave_t4.cpp`. Turing's
`src/compiler/spectral_wave_laws.py` now authors four of those arithmetic sites
as SymPy laws and compiles them through AbstractTensor/SSA into native f64 batch
pieces and deployable f32 SPIR-V products. These products do not replace T4's
fftfree transforms, arena scheduling, or continuation state.

| Product | T4 source site | Input domain |
|---|---|---|
| `angular_propagating` | `FftfreeExecutor::transfer_for`, propagating `kz2 >= 0` and complex multiply | positive wavelength and nonnegative `kz2`; direction ±1 |
| `angular_evanescent` | same transfer table, `kz2 < 0` | `gap = -kz2 > 0`; decay applies in either direction |
| `aperture_material` | `aperture_material_exact` | closed material pixels only; positive wavelength/distance, nonnegative imaginary index |
| `jones_interface` | `rigid_interface_exact` complex 2×2 multiply | coordinate map and source sampling are owned by the caller |

Generate the four validated GPU products in user build space from the Turing
review branch:

```powershell
python -m src.compiler.spectral_wave_laws --output C:\Users\alber\nodus-build\wave-kernel-artifacts
```

Each product directory contains the `.spv`, deployable shader record, source,
and validation manifest. The native Nodus test consumes those compiled SPIR-V
products through `GpuShaderJobTool` and `TensorStateLibraryTool`, using the
product's feed, output, and control binding order. It checks every output in a
512×512 plane on each of the three enumerated Vulkan devices. Receipts are in
`C:/Users/alber/nodus-build/wave-<kind>-512.log`:

| Kernel | GPU 0 first dispatch | GPU 1 | GPU 2 |
|---|---:|---:|---:|
| propagating | 24.778 ms | 15.362 ms | 8.540 ms |
| evanescent | 15.621 ms | 8.360 ms | 7.326 ms |
| aperture material | 18.105 ms | 10.802 ms | 8.325 ms |
| Jones interface | 1.437 ms | 1.183 ms | 1.644 ms |

These are first-dispatch host timings, not sustained kernel throughput or GPU
occupancy. Every row reported `lanes=262144 verified=1`. The independent
f64 LLVM gate and f32 SPIR-V assembly/validation gate in Turing each pass for
all four laws. The GPU trial compares against the T4 arithmetic at f32
precision, not bit parity with T4's cached f64 transfer table.

To continue toward a T4 production row, keep the existing fftfree forward and
inverse transforms, the source's cache/transfer ordering, aperture mask and
power accounting, Jones coordinate mapping, and bounded arena state. The
source's separated-arena energy and polarization xfails remain. A thick-lens
aperture run must carry the actual branch handle and Jones state through T1–T3
before this math can claim correct optical continuation. The new shader rows
are kernel products and a Nodus execution proof, not that completed pipeline.

## Joined wave work schedule (2026-11-13 continuation)

`test_spectral_wave_vulkan` now accepts the angular-propagating product and
the aperture-material product together. It registers both existing GPU shader
jobs in one `KernelPipelineTool` schedule. The first row writes complex field
planes at state-library offsets 7 and 8. The second row binds those same
addresses as its real and imaginary inputs, then writes offsets 13 and 14.
The schedule has two ordered receipts. It does not copy or reconstruct the
intermediate field on the host.

The 512×512 trial verified all 262,144 complex outputs against the composed
two-operator calculation on Vulkan devices 0, 1, and 2. Combined GPU dispatch
times reported by the two row receipts were 1.561, 1.658, and 1.766 ms in
that run. The original single-operator mode also passed on all three devices.
Logs: `C:/Users/alber/nodus-build/wave-joined-angular-material-512.log` and
`C:/Users/alber/nodus-build/wave-single-angular-512.log`. These are warm
host-observed dispatch durations for these arithmetic products, not a T4
arena propagation time or an estimate of a full ray exposure.

An optional final argument writes the verified device-0 complex field as a
512×512 PNG in user build space. The trial image is
`C:/Users/alber/nodus-build/nodus-joined-wave-field.png`; red encodes signed
real amplitude, green signed imaginary amplitude, and blue normalized power.
The test fixture deliberately varies input values across cells, creating
diagonal bands. This is an actual Nodus output tensor visualization, not a
scene photograph or diffraction result. Its run receipt is
`C:/Users/alber/nodus-build/wave-joined-angular-material-image-512.log`.

The remaining handoff is explicit: a ray-stage work token must identify its
T4 branch and arena field in the state library, execute the existing fftfree
and aperture/transfer sequence around these arithmetic rows, and return the
authored continuation state to T3/T5. The current joined schedule proves
ordered complex-field ownership within Nodus; it does not yet supply the
T3-to-T4 ray/field conversion or the completed image contract.

### Existing multi-reader edge for wave receipts

`EdgeTensorFifo` already has 64 independent subscriber cursors. A table edge
exposes `gp_table_edge_subscribe_ex` and keyed consume operations; with
`top_k == 0`, the slowest subscribed reader holds the lossless write frontier.
`top_k > 0` explicitly advances lagging cursors and reports a drop. Wave
continuation receipts can therefore fan out on the existing typed edge to a
continuation row and observers without a second queue or duplicated field
tensor. The FIFO owns the receipt bytes, not the tensor-library state named by
an address in the receipt. State removal must follow the continuation contract
and every required reader's completion; FIFO consumption alone cannot be used
as a state-lifetime lease.
