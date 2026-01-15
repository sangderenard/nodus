# Slice + Broadcast Execution Plan (Abstract Tensor Core)

This document describes the end-to-end plan to make abstract-tensor math natively
slice-aware, broadcast-aware, and ready for persistent worker-pool execution.
The focus is to place the deepest execution layer in tensor_math so every higher
level operation inherits the discipline. This is a strict "plan-first" workflow
for function-pointer kernels and a single-call broadcast execution model.

## Goals (Hard Requirements)
- No-copy slice views remain the canonical way to access subregions.
- All math ops must accept Dense and Strided layouts without bypassing accessors.
- Broadcasting must be explicit, planned, and centralized (one plan per op).
- Function-pointer kernels must always produce and use a plan.
- Execution must be ready for a persistent worker pool (no per-call threads).

## Non-Goals (for this phase)
- GPU backend changes.
- Replacing all math ops at once (we will migrate in waves).
- Changing front-end demo behavior or API shape outside tensor_math.

## Current State (Audit Summary)
1) Slice/View:
   - AbstractTensor::slice -> backend->get_item, no copy (InMemoryBackend).
   - InMemoryBackend::get_item builds a strided TensorDesc + offset and creates
     a view handle sharing the lease (no copy).
2) TensorDesc:
   - Supports TensorLayout::Dense and TensorLayout::Strided with element strides.
3) Tensor math:
   - Most ops assume Dense and map raw contiguous memory.
   - tensor_axpby_f32 has a custom strided path (standalone).
4) Function-pointer kernels:
   - Now require KernelDispatchPlan and precompute dense kernel tensors.
   - Hot loops no longer call function pointers.

## The Mounting Point (Core Execution Layer)
The correct "deepest level" is tensor_math, not in-memory backend or kpath.
All math ops will call a shared plan+executor pipeline:

  AbstractTensor(s) -> map -> TensorOpPlan -> TensorOpExecutor -> op kernel

This ensures:
- Slice and broadcast semantics are enforced once.
- All higher-level ops inherit the discipline.
- Worker pool integration is centralized.

## New Core Types (tensor_math)
### 1) TensorOpPlan
Represents a resolved broadcast + stride plan for N tensors.
Fields (proposed):
- rank: uint32_t
- shape: vector<uint64_t> (broadcasted logical dims)
- tensors: array of TensorOpTensor (one per input/output)

TensorOpTensor:
- dtype, layout
- base_ptr (mapped pointer)
- byte_strides[rank] (0 for broadcasted dims)
- element_bytes
- is_contiguous (fast-path flag)

### 2) TensorOpTile
Represents a contiguous range of work with stride info.
Fields:
- base_offsets for each tensor
- inner_count (length of fastest-changing contiguous span)
- outer_count (number of rows/blocks)
- inner_stride_bytes for each tensor (0 for broadcast)
- outer_stride_bytes for each tensor

### 3) TensorOpExecutor
Executes TensorOpPlan by generating tiles and dispatching to a worker pool.
Features:
- single-thread fallback (baseline)
- optional persistent worker pool by op-class (Elementwise, Stencil, Scatter)
- vectorized inner kernel entry (function pointer on raw spans)

## Broadcast + Slice Semantics
Broadcast rules:
- Scalars: rank = 0 or dims == {1} broadcast to any shape.
- For each dim, sizes must match or one side is 1 (broadcast).
- Stride for broadcasted dim is 0.
Slice rules:
- Strided views are handled by byte_strides from TensorDesc.
- Strides measured in elements, converted to bytes via dtype size.

## Executor Algorithm (Detailed)
1) Map all tensors (InMemoryBackend map, no copies).
2) Build TensorOpPlan:
   - Resolve broadcasted shape (max rank, right-align dims).
   - Validate dtype/layout compatibility.
   - Compute per-tensor byte strides (0 for broadcast).
   - Detect contiguous fast-path (all tensors dense and same shape).
3) Tile generation:
   - Choose fastest-changing dimension as "inner".
   - Compute inner_count = shape[last_dim].
   - Compute outer_count = product of remaining dims.
   - If inner stride == element_bytes for all tensors (or 0), it is a
     vectorization-friendly span.
4) Execution:
   - If worker pool enabled: split outer_count into blocks and dispatch.
   - Otherwise: iterate outer_count and call inner kernel once per row.
   - Kernel signature: void* arrays for tensor bases + inner_count.

## Worker Pool Integration
We will introduce an OpClass registry with fixed worker counts:
- OpClass::Elementwise (axpby, add_scaled, mul, clamp)
- OpClass::Stencil (apply_stencil_2d_into)
- OpClass::Scatter (kernel scatter)

The pool is persistent (threads parked), configured once at startup or first use:
- m x n matrix = {op_class -> worker_count}
- Each op submits a batch of tiles tagged by op_class.
- No per-call thread creation.

## Migration Phases (Strict Order)
Phase 1: Core planning/execution
- Implement TensorOpPlan and TensorOpExecutor in tensor_math.
- Extract broadcast/strided iteration from tensor_axpby_f32 into the shared plan.

Phase 2: Replace elementwise ops
- add_scaled_inplace_f32
- tensor_axpby_f32
- scale_inplace / add_inplace (if present)

Phase 3: Stencil and diffusion
- apply_stencil_2d_into uses TensorOpExecutor for interior and border tiles.
- Optional fast-path for 3x3 still OK, but now slice-aware.

Phase 4: Scatter and kernel bank
- Kernel scatter uses precomputed kernel tensors.
- One broadcast op per kernel (no per-pixel fn calls).
- Slice views used to assemble inputs (points, values) and outputs.

Phase 5: Remove legacy paths
- Delete ad-hoc dense-only loops once executor covers them.
- Require plans for all function-pointer kernels.

## API Adjustments (Concrete)
1) tensor_math.h:
   - Add TensorOpPlan, TensorOpExecutor declarations.
   - Keep kernel function-pointer entrypoints requiring KernelDispatchPlan.
2) tensor_math.cpp:
   - Implement plan/execute.
   - Migrate axpby to plan path, then add_scaled_inplace.
3) Optional: expose minimal stats (tile count, op class, time) for debug.

## Validation and Instrumentation
Use existing timing overlay to track:
- diffusion_iter_ms
- film_pixel_loop_ms
- scatter_deposit_ms

Add optional plan stats (debug-only):
- broadcasted shape
- tile count
- inner_count (vector span length)
- worker utilization (if enabled)

## Risks and Mitigations
- Risk: stride arithmetic errors -> wrong results.
  Mitigation: compare dense vs strided outputs on known test tensors.
- Risk: broadcast mismatch -> silent errors.
  Mitigation: strict validation, return false on mismatch.
- Risk: performance regressions on dense fast path.
  Mitigation: keep dense fast path if plan detects fully contiguous inputs.

## Success Criteria
- All elementwise ops are slice-aware and broadcast-aware without raw pointer
  access outside AbstractTensor map/unmap.
- Function-pointer kernels always produce a plan and execute in a single
  broadcast call per kernel.
- Diffusion and film pixel loops can be rewritten on top of the executor.
- Measurable drop in scatter and film hot loop timings once vectorized paths
  are in place.
