# Camera and transport kernel pipeline adoption (2026-11-13 UTC)

## Review-branch qualification against current Nodus default

Branch `codex/nodus-ray-kernel-library` was formed from current
`origin/nogodsnomasters` at `7f56f50`, then the kernel-library and authored
shader path was applied without the unrelated OpenGL/service worktree edits.
The default branch's newer fixed-buffer declaration remains in
`mem_backend.h`. `nodus_runtime` links the shared tensor core so the runtime,
canvas and shader tools do not own separate backend registries. An initial
isolated run before this link fix crashed during buffer cleanup; the repaired
build and six-job run exited zero.

The isolated build passed `test_spectral_t5_vulkan`,
`test_spectral_exact_lens_vulkan`, `test_spectral_t3_vulkan`,
`test_tensor_state_library`, `test_vulkan_state_bank`,
`test_vulkan_device_local_state`, and `edge_runtime_abi_test`. The six-job
T3→T5 run, the 26-stage fixture regression, the direct T3 record trial,
and the four-stage sensor→T1→conic T2→T3 lens trial exited zero. Logs are
under `C:/Users/alber/nodus-build/review-ray-*.log`. These are native
hardware trials; they do not establish a complete Studio exposure or the
second Studio tracer sequence.

## Authored T3 producer joined to T5 on Nodus

A second mode of `tests/spectral_t5_vulkan_test.cpp` now takes the authored
`ray_material` SPIR-V as a sixth artifact. It registers T3 as the first
`GpuShaderJobTool` in the same `KernelPipelineTool` and maps its BDPT output
descriptor to the state-library address consumed by sort, pack, scatter, and
T5. The 64 refined hits are a controlled one-bounce input (32 camera and 32
light); the BDPT vertices and side records are produced by T3 on-device, not
built by the test. All 27 stage receipts completed on each of the three
Vulkan devices. T3 emitted 64 vertices, 64 spectral side records, and 32
PDF side records; the 32 emissive endpoints carry their PDF in the vertex.
The sorted split was 32/32 and T5 yielded 1,024 pairs and linear RGB
`(1024,1024,1024)` per device. The separate 26-stage constructed-record mode
still passes. Six generated-record jobs now lease disjoint address ranges
on the same state-library card and submit concurrently, two per GPU row.
All six completed with the same output. Receipt:
`C:/Users/alber/nodus-build/t3_to_t5_nodus_six_jobs.log`.

This verifies a joined T3→T5 state-library pipeline, not a complete exposure.
T1/T2 output is not yet feeding these T3 hits, bounce feedback is not yet
scheduled, and the original Spectral queue runner has not been installed as
another Nodus sequence. These are concurrent host submissions to three GPUs,
not a measured hardware saturation or canvas scheduling trial. Correct
combination of independently sampled full
tracers requires their linear sums and effective exposure weights before
the film/display curve; the current test has one estimator and cannot certify
cross-engine weighting. T4 aperture propagation remains a separate bonus
gate after the full ray job path.

The second requested algorithm is the current Studio camera/forward shader
family in `spectral-analyzer/demo_pluck_gl.py`, identified separately in
`docs/STUDIO_NATIVE_PATH_FAMILY_INVENTORY.md`. It has not been lowered into
Nodus as a runnable sequence. Its actual camera shader, forward field pass,
and reactive queue have different root and normalization contracts from the
T1–T5 pipeline. The two families may share the state library and device
market, but a combined exposure must preserve their distinct root counts,
PDFs, shutter slices, and linear output sums. A cross-family path MIS claim
would require the proposal densities from both algorithms for each sampled
path. A weighted average of independently complete estimators can instead
combine per-pixel linear sums and corresponding effective sample weights.

## Later same-day execution receipt

The earlier stage matrix below was written before the T3 and T5 native-row
trials and is retained as a dated record. T3 now executes as an authored
Spectral shader through a Nodus Vulkan row on devices 0, 1, and 2. On device
0, one pipeline row coupled sensor raygen → T1 → exact-conic T2 → T3, with four
completed stage receipts. This is a transport-step proof, not a rendered frame.

The authored Spectral `t5_full_connect` shader now executes through a Nodus
`GpuShaderJobTool` inside `KernelPipelineTool` on all three Vulkan devices. The
work token gained an optional Y invocation count, allowing its native 16×16
camera/light tiles to dispatch in both axes. Legacy one-axis tokens retain their
original shape and behavior. A constructed pair of facing one-vertex camera
and light subpaths, each with 32 vertices, produced 1,024 evaluated pairs,
1,024 unoccluded MIS-valid contributions, and RGB `(1024,1024,1024)` in the
one-pixel linear accumulator on **each** device. The fixture uses empty BVH,
one band, sensor and emission endpoints, and exact expected energy. It proves
the full T5 shader path through Nodus, including spectral response, connection
PDFs, MIS, shadow test, reduction, and pixel atomic splat. It does not yet
consume T3-generated vertex records, test multi-bounce paths, or establish
pixel parity against the original OpenGL host. Receipt:
`C:/Users/alber/nodus-build/t5_nodus_full.log`.

T4 remains the separate native wave-continuation stage. Its original path has
known separated-arena energy and polarization xfails in the guestbook; no
Nodus T4 row is proven here. T4 is entered only for wave-region work; a
ray-only exposure can route T1/T2/T3 to T5 without T4. Wave-enabled exposures
still need the arena path, while all exposures need real side-record packing,
subpath scheduling, and a complete exposure/image contract.

The authored `bdpt_pack_t5` and `bdpt_scatter_t5` stages have also been
relocated into Vulkan descriptor contracts without changing their algorithms.
A four-stage Nodus pipeline now runs pack → spectral scatter → PDF scatter →
T5 connection on all three Vulkan devices. It uses 32 light and 32 sensor
one-vertex records in the original BDPT side-record layout, with distinct
subpath IDs above 2^24. Packer fallback betas are deliberately 0.25 and PDF
flags are empty; successful scatter must replace them with unit betas and
emission/diffuse flags before T5 can yield the exact RGB `(1024,1024,1024)`.
Each device passed these assertions and all 1,024 connection-PDF and unoccluded
counters. `C:/Users/alber/nodus-build/t5_nodus_pack_connect.log` records the run.
The side records are constructed fixture inputs, not yet emitted by the
preceding Nodus transport stages.

The authored `bdpt_sort_init` is now the first stage, generating keys,
indices, and light/camera counts from raw BDPT vertices. The resulting five
stage pipeline (sort-init → pack → spectral scatter → PDF scatter → connect)
returned five successful stage receipts, counts `(32,32)`, 1,024 accepted
pairs, and exact RGB `(1024,1024,1024)` on Vulkan devices 0/1/2. The fixture's
raw vertices happen to be in sorted order; the compare/swap sort stage is not
yet in this Nodus test. Latest receipt:
`C:/Users/alber/nodus-build/t5_nodus_sort_pack_connect.log`.

The authored `bdpt_sort_step` is now included too. The controlled input
interleaves 32 camera and 32 light vertices, so sort-init alone cannot produce
the expected packed layout. A Nodus pipeline executes all 21 bitonic passes
for 64 items, then pack → spectral scatter → PDF scatter → connect: 26 stages
total. Each Vulkan device returned the correct ascending keys and sorted
indices (the first light index is input 1; the first camera index is input 0),
32/32 counts, 1,024 unoccluded connections, and linear RGB `(1024,1024,1024)`.
The fixture still supplies raw BDPT side records; live T1/T2/T3 output and
multi-vertex MIS chains are not yet coupled. Latest receipt:
`C:/Users/alber/nodus-build/t5_nodus_full_sort_connect.log`.

The Nodus T3 direct-run test now enables its BDPT output: a single emissive
hit emitted one vertex, one spectral record, one PDF record, and one terminal
on each Vulkan device. This establishes the producer record shape without
claiming that it has been wired to the pack/sort stages. Receipt:
`C:/Users/alber/nodus-build/t3_bdpt_records.log`.

The stricter pack/scatter fixture exposed a Vulkan ID lookup failure in the
authored scatter shader: light-side records landed but camera-side records
did not when the bit-packed ID flowed directly into the binary-search key.
An exact two-16-bit reconstruction of the same 31-bit ID fixed the lookup on
all devices, including IDs above 2^24. The source fix is in Spectral's
`bdpt_scatter_t5.comp.glsl`; the Nodus adapter still relocates only uniforms.

This receipt distinguishes shader compilation, one-stage numerical execution,
and a complete camera pipeline. Builds are in `C:/Users/alber/nodus-build`;
the public service and Studio browser were not changed.

## Source ownership and current backend matrix

| Family | Authoritative source and existing execution | Nodus Vulkan result | Unfinished boundary |
|---|---|---|---|
| Studio Phong raster | Spectral `csrc/shaders/base_material.vert.glsl` and `.frag.glsl`; Turing-generated WebGL variants run in Studio | Unmodified `-V` refused non-opaque uniforms outside blocks; no Vulkan raster host | Explicit descriptors, texture/image resources, render-pass targets, actual live Studio draw/uniform contract; CPU native raster kernel absent |
| Thin aperture sensor launch | Spectral OpenGL `sensor_mip_raygen.comp.glsl` and native C++ camera launch; Turing Studio's derived analytical lens contract is separate | Raygen adapted and executed in both target directions on three Vulkan devices; one sensor ray coupled to T1 and exact-conic T2 | T3/material and sensor completion; derived Studio contract must not be substituted for real compound glass |
| Exact compound lens | Spectral `camera_designer/compound_optics.py` builds the `PLENS` payload; native C++ `csrc/kernels/ray_tracer.cpp` and original OpenGL `csrc/shaders/ray_refine.comp.glsl` traverse its conics | T2 adapted, validated, dispatched on three devices; one backward sensor ray coupled through T1→T2 | Off-axis/wavelength parity corpus, T3 and camera/sensor exposure |
| Forward and sensor-backward ray transport | Spectral native C++ ray pipeline and OpenGL T1/T2/T3 shaders; direction flag and reciprocal traversal are in T2 | T1, T2, and T3 compiled; T1 hit count and T2 lens tests run separately | T3 image-resource runner, direction-specific end-to-end results |
| BDPT connection | Spectral native T5 logic and OpenGL `t5_full_connect.comp.glsl` plus BVH/material preamble | Source-specific T5 artifact compiles and validates | T5 input/output adoption, MIS numerical parity, full bidirectional exposure |

The active GPU loader in Spectral's `ray_tracer.cpp` names T1
`ray_bvh_intersect`, T2 `ray_refine`, T3 `ray_material`, and T5
`t5_full_connect`. Its `t0_intent_pack.comp.glsl` explicitly says it is retired
and unwired; it is not used as a substitute launch stage. A thick lens is the
`PLENS` conic interception zone in T2, not Studio's derived thin-lens profile.

Unmodified Vulkan compile probes on `ray_refine`, `ray_material`,
`t5_full_connect`, and `sensor_terminal_splat` all stopped at default-block
uniform declarations. This is a Vulkan GLSL interface rule, not evidence about
the transport math or runtime speed. The Phong desktop GLSL pair passed
`glslangValidator -G --auto-map-locations --auto-map-bindings` as separate
OpenGL SPIR-V stages. Its unmodified `-V` attempts stopped at the same Vulkan
uniform-block rule. The generated WebGL pair remains the actual Studio source;
the desktop GLSL is the corresponding original shader family.

## Executable Nodus boundary

`KernelPipelineTool` is a native `ITool` row using the existing registry and
table/FIFO execution path. Its input is an `AbstractTensor` U64[N,7]. Every row
is `{component index, execution API, semantic program ID, library card, state
base, invocation count, generation}` in execution order. The component index
resolves to a registered row, never user-supplied executable source. API and
program identity are checked against that component before the first stage.
Each component receives the existing U64[4] work token and returns its own
U64[9] timed receipt. The output U64[N+1,9] contains a summary then each
attempted stage's receipt. It stops on the first failed or uncertain stage and
does not reroute/replay it. The row supports Vulkan and CPU components that
already share the tensor state library. OpenGL is refused until its owning
context/process and state-bank transfer contract exists.

`GpuShaderJobContract.binding_offsets` maps a stage's descriptor ordinal to a
logical state address. This is necessary for T1/T2/T3 to share a tensor without
pretending their binding numbers have the same meaning. Empty offsets preserve
the original contiguous-address ABI. Duplicate offsets and address overflow
are refused. The two-component test uses non-contiguous offsets and a two-row
pipeline tensor. A second test runs the existing Turing Vulkan fabric row
followed by its LLVM CPU row on the *same* addressed state and returns two
successful receipts. These are ordered native row calls, not a fused GPU
program or a measured performance win.

## Exact-conic T2 trial

`glsl_uniform_block.py` now performs the relocation for sensor raygen, T1, T2,
T3, and T5 from
explicit field lists. It refuses missing, duplicate, unexpected, opaque, or
initialized default uniforms; no string-wide deletion is allowed. Each stage
adapter retains the original source bytes, verifies SPIR-V reflection, and
keeps host packing responsible for exact offsets. In particular,
`adapt_spectral_ray_refine.py` replaces only its six OpenGL default uniforms
with one std430 storage block at binding 8. It verifies the nine reflected
SSBO bindings, 64-wide workgroup, control
member order and exact byte offsets (0,4,8,12,140,144; 148-byte block).
`glslangValidator -V`, `spirv-val`, and `spirv-cross` passed. It does not use
Nodus KernelIR translation.

The test used `simple_doublet_preset()`,
`CompoundLens.from_preset(..., wavelengths_um=preset.wavelengths,
axial_scale=-1.0)`, and the unmodified `build_gpu_payload()` bytes. One ray
entered near the first conic at X=-0.06 m with Y=0.001 m and +X direction.
Each Vulkan device returned the same exiting X=-0.0499998 m, Y=0.00125763 m,
outgoing Y direction 0.049325, teleport flag 4, and no absorption flag. The
original Python compound lens, traced from before the same assembly, gives
Y=0.00125423 m and outgoing Y direction 0.04851. This comparison is contextual,
not a parity assertion: the GPU selected the first authored spectral band and
the Python scalar trace uses its own reference index. The test asserts finite
off-axis bending and successful traversal; it does not certify chromatic or
energy parity. Warm one-ray dispatches in one run were 0.85, 2.00, and 1.20 ms
on Vulkan devices 0/1/2, respectively. These are dominated by launch/wait
cost and are not a camera throughput benchmark.

`adapt_spectral_ray_material.py` compiles the original T3 scattering source.
Its 17 scalar default uniforms move to a reflected 68-byte storage block.
The original OpenGL image and SSBO both use binding number 7 in separate
namespaces; the Vulkan image moves to descriptor binding 9 while the SSBO
remains at 7. The artifact passes `spirv-val`, but the current Nodus shader
job row leases only storage buffers, so it cannot execute T3 yet. No image
or sensor result is claimed.

`adapt_spectral_sensor_raygen.py` compiles the original sensor mip raygen with
its nineteen default controls relocated to reflected SSBO binding 8 (152
bytes). Its source already models an aperture disk target and `target_mode`
for opposite ray direction. A one-sample fixture in the native test dispatched
on all three Vulkan devices: mode 0 wrote an intent pointing +X; mode 1 wrote
an intent pointing -X. All six runs set the expected one-intent counter. The
aperture radius in this fixture was zero, so it verifies launch direction and
buffer ABI, not stochastic disk sampling, sensor film response, or a camera
image. The separate exact-conic T2 stage remains necessary for real glass.

An additional ordered three-component pipeline on Vulkan device 0 uses one
logical state-address range: raygen binding 5 writes the RayIntent tensor,
T1 binding 0 reads that same tensor and writes a HitRecord, and T2 binding 0
reads the same HitRecord. The ray starts on the sensor side of the original
doublet, crosses a one-triangle lens proxy at X=-0.05 m, and T2 traverses the
`PLENS` surfaces backward. Its receipt reported three completed stages; T1
reported one hit; T2 exited at X=-0.0600002 m with flags 5 (camera path plus
teleport, no absorption). This exercises actual state coupling and ordering,
not merely independent jobs in a list. It still uses one ray, zero aperture
radius, one proxy triangle and no T3, film accumulation, image or exposure.

`adapt_spectral_t5.py` mirrors Spectral's native T5 preamble injection (BVH
bindings 5/6/7 and material binding 8), and moves the three default uniforms
to a verified 12-byte storage block at binding 10. Its SPIR-V compiles and
validates with all eleven storage bindings reflected. No T5 invocation or
BDPT image is claimed.

`adapt_spectral_phong_raster.py` compiles the original desktop Phong vertex
and fragment pair as Vulkan raster SPIR-V. The source-derived interface moves
vertex, thermal, and other fragment scalar controls into reflected storage
blocks at bindings 17–19 and assigns samplers distinct descriptors starting
at 32. Uniform initializer expressions are recorded in the receipt for the
future host. Both stages pass `spirv-val`; samplers disabled by compile-time
branches are recorded as declared but may be absent from active reflection.
This is a compiled graphics artifact, not a rendered frame.

The frame itself needs no special image storage: Nodus already represents
images as dense `AbstractTensor` values in HWC layout, carries tensor handles
through row stacks/FIFOs, and accepts tensor payloads in the state-library
`Put` operation. A graphics runner can finish a draw, read or copy its color
attachment into such a tensor, and publish the tensor plus a frame receipt.
The receipt must identify the frame generation and interpretation (channel
order, linear or display transfer, and origin); shape and dtype already live
in the tensor descriptor. A consumer can take the FIFO tensor directly or
store it under a library address for reuse. `kpath_image_export` can encode a
compatible tensor to PNG if persistence is wanted.

What is missing is the Vulkan/OpenGL raster runner's render target and its
completed-draw transfer into the existing tensor ABI. The current Vulkan
runner owns storage buffers for compute only. A first implementation can
publish a CPU tensor after synchronized readback. Keeping the frame on the
GPU would require image-to-buffer copy and ownership/synchronization support
in that graphics runner; it does not require a second image library or an
image-specific FIFO.

For durable output, `exec_row` is the byte-span compression stage: it consumes
a tensor payload and returns a U8 compressed tensor in FIFO order. Its current
contract does not preserve the input tensor's shape/dtype, so a lossless raw
frame must serialize a small descriptor with the pixels before that row.
Nodus does have file-backed FIFO storage: `EdgeTensorFifo` can attach a
filesystem backend, and that backend writes allocated buffers as files.
`gp_meta_edge_snapshot_capture_to_edge` also publishes file-buffer handles
through a FIFO. A file-backed ring is staging storage, however: its slots can
be reused, and its generated `fb_*` paths are not a named, committed job
artifact. The job host should give the file backend an already authorized
job-scoped directory, then commit the finished compressed file and update the
job's status only after verifying it exists. `PhotoJobs.job_directory(digest)`
already resolves that directory using the `job_owners` SQL index; `service_jobs`
records running work, not image blobs. A PNG preview needs separate image
encoding because the exec compressor returns raw compressed bytes. The
graphics runner and live job-host commit are not yet connected; the
source-only Phong artifact creates no SQL job or file on its own.

The native row boundary is now implemented for an admitted job. A
`FrameTensorPacketTool` emits a versioned U8 packet containing dtype, shape,
color encoding, origin and exact tensor bytes; its decoder validates and
recreates the tensor. `FrameArtifactFileTool` accepts a compressed U8 result
and a generation token, allocates through Nodus's filesystem backend under
the host-selected job directory, closes the buffer, and publishes the file
as `frame-<generation>.nfrm.pack` with a U64 receipt. Repeating a generation
is refused; a failure after file publication does not restore the input for
replay. `register_frame_artifact_rows` gives the canvas registry a packet and
file row for one admitted digest. The host supplies separate raw-packet and
compressed-byte limits. The exec compressor itself is the existing plugin
row; the host still selects its declared program and records that codec in
the job contract.

`frame_compression_pipeline_test` passed on Windows with the actual exec row
using gzip as its external compressor: a 32×32×4 F32 test tensor went through
packet → exec pack → filesystem-backend file → exec unpack → packet decode,
and every reconstructed float matched. `frame_artifact_file_tool_test` also
passed exact file bytes, duplicate-generation refusal, and compressed-byte
quota refusal. An isolated SQL
bridge check claimed a digest for `account:7` in `job-library.sqlite`, used
`PhotoJobs.job_directory()` to resolve the owner-scoped job directory, ran
the same native pipeline there, and confirmed another account could not read
the job owner index. No real service database or user job was changed.

This proves the lower frame-storage path, not the full image path: the graphics
runner does not yet draw the original Phong shaders into an attachment or
publish its pixels as a tensor, and the live service has not attached these
rows to a PhotoJobs worker or recorded their receipt in job status. A generic
`EdgeTensorFifo` slot is not used for the variable-length compressed span;
the tool's filesystem buffer is one file per generation, while ordinary row
FIFOs carry the tensor/receipt tokens.

An actual source-rendered proof then used Spectral Analyzer's existing
`BaseGLRenderer`, which compiles the original `base_material.vert/frag.glsl`,
to draw one 256×256 RGBA Phong frame on this Windows GL context. The captured
frame had 110 distinct RGBA values and was saved for visual inspection at
`C:/Users/alber/nodus-build/frame-phong-proof/frame.png`. The integration
test's external-frame option packeted those exact 262,144 bytes as U8 HWC
with sRGB/top-left metadata, compressed them through the actual exec-row gzip
program, and verified decompression plus tensor reconstruction byte for byte.
In an isolated SQL test, `job_library.claim` assigned a fresh digest to
`account:7`; `PhotoJobs.job_directory` resolved the user's asset folder; the
native file row committed a 3,352-byte compressed frame there. Account:8 did
not pass the owner check. This is not a production Studio frame: the test
invoked the existing Spectral Python GL owner, and the Nodus protected GL
process remains link-only. The actual service job/receipt attachment and a
native Nodus raster runner still remain.

The OpenGL kernel boundary now also has `gp_gl_frame_read_rgba8`: the owner of
the current GL read framebuffer can obtain a quota-bounded U8 HWC tensor after
the draw. It checks the registered native context, uses the current read
framebuffer, and restores GL pixel pack state. The tensor retains native GL
bottom-left row order, which the frame packet records explicitly. A real WGL
test drew two distinct regions, compared every byte of Nodus's tensor to
`glReadPixels`, checked restoration of nondefault row-length/skip settings,
and verified a one-byte-short host quota refuses the transfer. This is the
missing frame-to-tensor seam; it does not create a render target or command the
protected link-only child to draw. The 256×256 Phong proof above predated this
seam and used direct `glReadPixels` capture.
The same test now optionally imports Spectral Analyzer's `BaseGLRenderer` and
`MaterialDatabase`, draws the original `base_material` shader pair, and reads
that completed draw through `gp_gl_frame_read_rgba8`. On this machine the
64×64 result contained a lit blue surface with more than two distinct colors;
Nodus's tensor bytes matched direct GL readback exactly. The inspection image
is `C:/Users/alber/nodus-build/frame-phong-proof/native-transfer.png`. This
is a caller-owned development context, not the protected child or live service.
The exact 16,384 bottom-left RGBA tensor bytes from that readback then passed
through the packet row (origin=bottom-left), the actual exec-row gzip program,
the filesystem artifact row, exec-row decompression, and packet decode. The
integration test returned zero after byte-for-byte reconstruction. Its durable
test file is `C:/Users/alber/nodus-build/frame-phong-proof/native-transfer-pipeline/frame-0000000000000027.nfrm.pack`
and is 820 bytes. This completes a source-shader → Nodus tensor → compressed
artifact proof, while the production protected graphics child and SQL job
status integration remain separate work.
The same native-readback bytes were also sent through an isolated SQL admission:
`job_library.claim` assigned digest `dddd…` to `account:7`, `PhotoJobs.job_directory`
selected that account's private assets directory, and the file row committed
the 820-byte frame there with exact roundtrip parity. `account:8` failed the
owner check. This did not touch the live service database or change a running
PhotoJobs worker.

## Next integration

1. Feed one authenticated scene's actual T3 material/sensor buffers after the
   now coupled raygen→T1→T2 pipeline; compare the
   native C++ and original OpenGL paths on identical wavelength and ray cases.
2. Add T3's image resource to the state bank and Vulkan dispatch contract;
   then exercise forward and backward camera rays and the sensor endpoint.
3. Give the Phong raster pair a reviewed Vulkan descriptor/render-pass host
   using the actual live Studio resources. A compute-row compile is not a
   substitute for a raster draw.
4. Connect native T5 buffers and MIS results, then benchmark equivalent
   OpenGL/Vulkan/native end-to-end exposures. T5 artifact compilation alone
   is not a usable BDPT pipeline.
