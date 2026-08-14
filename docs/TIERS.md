# Nodus tiers: what KernelIR is, and what it is not

**Status:** established 2026-08-13, from an audit of the existing design.
Until this document, the tier discipline lived entirely in two file headers
(`src/spirv_translation.cpp` and `include/bitops_lowering.h`) and was
therefore easy to violate by accident — which is exactly what happened, see
*Walked back* below.

## The layers

### Tier-0 — the portable kernel ISA (`src/kernel_isa.h`)

The minimal instruction set **every kernel emitter understands directly**.
`spirv_translation.cpp` states the contract:

> There exists a *pure* Tier-0 KernelIR and a GLSL compute emitter that only
> understands Tier-0 ops. Composites / BitStructType dispatch occur *before*
> this stage (Tier-1 lowering), not in the emitter.

Tier-0 is deliberately small. Its membership test is strict:

> An instruction belongs in Tier-0 only if it is **primitive** — it cannot be
> built from other Tier-0 instructions — and **universally implementable** by
> every emitter (SPIR-V, GLSL, CPU, Eigen, Metal, ONNX, torch, Vulkan,
> Fortran).

Adding an opcode taxes every current and future backend, forever. Adding a
*recipe* taxes nobody.

### Tier-1 — composites defined over Tier-0

Anything that is not primitive is **defined as a recipe that emits Tier-0**,
never as an opcode. `include/bitops_lowering.h` is the original example
(gray-code as AND/XOR/shift); `include/tier1_lowering.h` holds the tensor
composites (reduce, affine generate). A composite expressed here lowers to
every backend that implements Tier-0 — one definition, N emitters.

The catalog records which family a composite belongs to
(`OpDesc::tier1_class` in `include/canonical_ops.h`: `reduce`, `contract`,
`remap`, `generate`, `order`) so the knowledge is data rather than folklore,
while `kernel_op` stays null and `lowerable` stays false — an honest
statement that it is not one Tier-0 instruction.

### Tier-2 — the AbstractTensor layer

nodus's C++ `AbstractTensor` (`include/common/tensors/abstraction/`) is a
port of turing's; `python_prototype.py` sits beside it as the reference.
This is where tensor semantics live, where backends (InMemoryBackend /
`tensor_math`, Eigen, torch) execute, where the ~300-operation surface
composes, and where autograd binds.

**Tier-2 does not require Tier-0.** An AbstractTensor operation needs a
KernelIR expression *only if it must run as a device kernel*. `sum` on CPU or
Eigen never touches KernelIR at all.

## The two roads (do not confuse them)

1. **turing AbstractTensor ↔ nodus AbstractTensor — the main road.**
   Both sides hold the same operator set in their own language. Translation
   is direct, semantics are shared, and composition schemas mean the extended
   surface (linalg, FFT) is built from the fundamental set and *inherits
   autograd by construction*. The generated vocabulary tools
   (`src/vocabulary_actualizer.cpp`) and the artifact importer
   (`src/artifact_importer.cpp`) live here.

2. **KernelIR — nodus's deployment path to devices.**
   Repository SSA / AbstractTensor programs lower to Tier-0 so they can run
   as compute kernels (SPIR-V first, hence `namespace nodus::spirv`).

Breadth of the tensor surface is a **Tier-2** question. Device deployment is
a **Tier-0/1** question. They meet only where a tensor op must become a
kernel.

## Walked back (2026-08-13)

Five "structured classes" (REDUCE, CONTRACT, REMAP, GENERATE, ORDER) were
added to Tier-0, with REDUCE's loop expansion written *inside*
`kernel_spirv.cpp`. That inverted the architecture twice over: a reduction is
not primitive, and its definition ended up in an emitter. The practical cost
was immediate — REDUCE worked only on the one backend taught about it, and
the other nine would each have needed their own implementation.

Removed. In their place Tier-0 gained the primitive that was actually
missing:

- **`LOOP_BEGIN` / `LOOP_END`** — bounded, structured iteration. Nothing that
  walks a buffer could be *defined* without it, which is why composites had
  looked impossible to express and why reaching for opcodes felt necessary.
  Universally implementable: SPIR-V `OpLoopMerge`, GLSL/C `for`, a plain loop
  on CPU/Eigen.
- **`VAR`** — was declared but never implemented; now a function-local
  mutable cell whose result is a pointer, so ordinary `LOAD`/`STORE` work on
  it. This is what carries an accumulator across iterations without the IR
  needing phi nodes at all.

With those two, `sum` became a Tier-1 recipe
(`tier1::lower_reduce`) and ran correctly on an NVIDIA GTX 1660 SUPER:
`spirv-val`-clean, 16320 on device against 16320 on host.

## Cooperative execution (design, in progress)

A driver watchdog (Windows TDR) resets the GPU if one dispatch runs long, so
a kernel must be able to stop of its own accord. The mechanism belongs **in
the translation**, not in host-side chunking:

- **Flag dressing on control primitives.** Every control primitive consults a
  budget as part of its own condition — `KernelIR::budget_value_id` names a
  buffer, and `LOOP_BEGIN` folds `budget != 0` into its loop test. Because it
  rides on the primitive, *every* generated kernel is bounded by
  construction; no author or caller has to remember.
- **Per-segment skip-state flags.** Each straight-line segment between
  control points carries a resume marker, so a re-entered kernel skips
  completed segments and continues — resume through minor additions to each
  closure.

Consequence: state that must survive a yield lives in buffers, not registers;
a reduce recipe should carry a continuation index rather than one unbounded
in-kernel loop. See task #28.

## Rules of thumb

- Adding a Tier-0 opcode is a **last resort**. First ask: can this be a
  Tier-1 recipe over what exists? If not, what primitive is genuinely missing?
- An emitter must never contain a composite's definition.
- `lowerable` in the catalog means exactly "is one Tier-0 instruction" —
  never widen it to mean "we have some way to do this".
- What is valid SPIR-V is valid SPIR-V; the constraint is only that a
  construct must work across turing, nodus, and the device backends alike.
