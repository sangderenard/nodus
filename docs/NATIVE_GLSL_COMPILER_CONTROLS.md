# Native OpenGL compiler controls: evidence and bounded test plan

Research date: 2026-11-11 UTC. This investigation made no graphics calls,
created no contexts, changed no driver profiles, and changed no frozen source
package or native binary. Source-only candidates below are not yet applied.

## Recommended first controlled candidate

Create a separate copy of the exact failing `studio.segment.v2` source with
only `#pragma optimize(off)` added immediately after its `#version 430 core`
line. Keep the original frozen source and receipt. This is a documented GLSL
control, not an invented NVIDIA environment variable or a Cg/CUDA compiler flag.

[GLSL 4.30 specification, section 3.3, printed page 13](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.30.pdf)
defines optimize(on/off) outside function definitions; optimization defaults on.
The same section defines debug(on/off) for debug information and says unknown
pragmas are ignored. The specification does not identify the particular NVIDIA
link-time passes suppressed or promise a compile-memory bound. A successful
compile is therefore not proof that every optimizer was disabled. Disabling
optimization can also increase final code/resource requirements. Do not add
`debug(on)` simultaneously: it is not an optimization-off control.

A narrower extension candidate is `#extension GL_EXT_control_flow_attributes :
require` plus `[[dont_unroll]]` on an identified expensive loop, or
`[[dont_flatten]]` on an identified branch. The
[Khronos extension text](https://raw.githubusercontent.com/KhronosGroup/GLSL/master/extensions/ext/GL_EXT_control_flow_attributes.txt)
is marked Draft, applies to GLSL 1.40 and later, and describes these as strong
requests rather than mandatory transformations. Actual NVIDIA desktop-driver
support is not established here. Using `require` fails compilation rather than
silently treating an unsupported extension as tested. Do not use dependency
attributes without proving their semantic promises. This is a secondary
candidate, not evidence that unrolling caused our failure.

Core GLSL has no usable `noinline` function keyword: it is reserved for future
use in GLSL 4.30 section 3.6. Do not add CUDA `__noinline__` or HLSL `[loop]`
syntax to the frozen GLSL source.

## Native API controls are different

[ARB_parallel_shader_compile](https://registry.khronos.org/OpenGL/extensions/ARB/ARB_parallel_shader_compile.txt)
and [KHR_parallel_shader_compile](https://registry.khronos.org/OpenGL/extensions/KHR/KHR_parallel_shader_compile.txt)
define `glMaxShaderCompilerThreadsARB/KHR(0)` as a request for no background
parallel compile/link work. The extension explicitly does not guarantee exact
obedience. Its state query returns the requested value, not measured live
thread count. `COMPLETION_STATUS_ARB/KHR` supports completion polling; it is not
an optimization-level control. Restricting compiler parallelism could reduce
peak host memory, but this is an unmeasured hypothesis. Our single-process job
limit restricts processes, not threads.

This API candidate would require a separately reviewed child implementation,
verified extension/procedure support in that child, and new frozen executable
hashes. It cannot be injected through the current CLI. Keep it separate from
the first source-only test so the variable being tested is clear.

[ARB_get_program_binary](https://registry.khronos.org/OpenGL/extensions/ARB/ARB_get_program_binary.txt)
caches a successfully compiled/linked program for later use. It does not remove
the first failing compilation. Program-binary retrieval hints concern later
binary availability, not a general optimization-off switch. The present worker
always releases its program and returns a bounded receipt; it does not export
program binaries or transferable GL handles.

NVIDIA documents an automatic shader cache for
[Jetson OpenGL ES](https://docs.nvidia.com/jetson/archives/r35.2.1/DeveloperGuide/text/SD/GraphicsProgramming/BinaryShaderProgramManagement.html),
including platform-specific environment variables. That document is not proof
of equivalent Windows desktop controls. NVIDIA's
[NVAPI settings reference](https://docs.nvidia.com/nvapi/NvApiDriverSettings_8h.html)
lists shader-cache settings, but no profile or system change is part of this
investigation. Cache hits may affect timing; no cache flushing/disabling or
additional baseline retry is authorized.

## NVIDIA-specific pragmas: evidence limit

A [2019 NVIDIA engineer's report](https://forums.developer.nvidia.com/t/bug-report-glsl-perf-dead-code-unrecognized-unrolled-code-25-times-more-costly-than-loop/69472)
documents `#pragma option inline all` for a particular historical GL inlining
issue. This establishes neither a present-day complete option grammar nor
support for guessed inverse forms on driver 610.88. It asks for more inlining,
which is not the intended memory experiment. Historical third-party mentions
of `optionNV` do not establish a supported current contract. No optionNV form,
undocumented environment variable, profile edit, or system setting is proposed
as a verified fix.

## Offline tools are not the driver compiler

[Khronos glslang](https://github.com/KhronosGroup/glslang) supplies source
validation and SPIR-V generation. The installed official glslang 16.0.0 help
was read CPU-only: `-Od` disables its optimization, `-Os` minimizes SPIR-V size,
`-G` emits SPIR-V for OpenGL, and `-V` emits it for Vulkan. Those flags do not
change NVIDIA's compiler when our helper later passes the original GLSL text
to `glShaderSource`. The recorded `-S comp` and `-E` checks are offline
syntax/preprocessing evidence, not native machine-code/link validation.

[ARB_gl_spirv](https://registry.khronos.org/OpenGL/extensions/ARB/ARB_gl_spirv.txt)
is a different input path with module validation and specialization. Our
frozen worker does not implement it, and offline optimization would not prove
that later NVIDIA specialization/linking avoids this failure. Similarly,
FXC/DXC `/Od`/`-Od`, CUDA compiler optimization/unroll flags, and Cg options are
not arguments accepted by the current OpenGL source path.

## What the existing source and receipts establish

- `src/kernels/kernel_opengl.cpp` loads ShaderSource/CompileShader/LinkProgram,
  passes unchanged bounded source bytes, checks compile/link status, and
  releases shader/program owners. It exposes no optimization flag argument and
  installs no parallel-compiler, program-binary or compiler-hint setting.
- `src/kernels/opengl_protected_wgl.cpp` verifies exact requested GL LUID/node
  before the helper. It checks the needed identity extension but does not
  persist a full extension inventory or prove compiler-control support.
- Frozen segment v2 currently contains no pragma or extension directive.
- The completed segment baseline reached combined linking and returned C9999
  after 4344 ms. Host process peak 4293746688 bytes; reported aggregate job peak
  4294991872 bytes. The latter is 24576 bytes above the configured 4 GiB cap.
  Cleanup/reaping succeeded. This is host compiler-memory evidence, not a VRAM
  measurement, and it does not identify which optimizer pass consumed memory.

Without another graphics call we can verify source syntax, existing logs,
installed tool help and executable/source hashes. We cannot claim the current
context advertises the optional extensions, that an ignored pragma took
visible effect, or that optimizer suppression prevents C9999. Offline glslang
extension support is not proof of NVIDIA extension support.

## One-variable, link-only follow-up plan

1. On approval to prepare, create a new optimize-off segment copy outside all
   frozen package files. Preserve every original byte except the added line;
   record the diff, byte count, raw hash and a unique source label.
2. Run the official bounded offline `-S comp` and `-E` validation and inspect
   the directive plus unchanged stage/state contract. Use the unchanged native
   runner `--prepare` to obtain the exact labeled-object approval digest.
3. Assign a new contract ID. Keep RTX 3060 LUID `22f1000000000000`, node mask 1,
   verified-default-dc, 4 GiB process AND aggregate job caps, 8 GiB available-RAM
   floor, 90-second watchdog, one process and no descendants.
4. Root gives the user-facing notice and explicit go-ahead for that exact
   source/digest/one attempt. Only then use the canonical native
   ThreadManager/typed FIFO/row/protected child path. Do not rerun baseline or
   automatically retry a failed/uncertain operation.
5. Compare against the already completed baseline: compile/link result,
   diagnostic, elapsed time, process/job peaks, identity and containment proof,
   GL cleanup, process reaping, exactly one commit. Record any above-cap OS peak
   honestly. No dispatch, rendering, exposure, screenshot, allocation workload,
   persistent session, deployment, or driver-profile change.

Even if the variant links, retain GP_GL_UNVERIFIED_CONTINUATION. Link success
alone does not prove physical correctness, full field data or safe execution.
