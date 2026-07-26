# Nodus tensor subsystem

This directory contains the active C++ tensor subsystem derived in part from the Python abstraction work under `turing/src/common/tensors`. It is no longer an empty scaffold.

The current implementation combines a thin `AbstractTensor` handle and backend registry; in-memory allocation, pooling, descriptors, and dtype/shape metadata; a large structured-spatial `tensor_math` library; k-path geometry, stencil, splat, probe, and raster operations; and backend, scheduling, runtime, autograd, diagnostic, and model experiments.

Not every historical subdirectory represents a complete or equally active layer. Use the headers and build-target sources as the authority for what is implemented.

## Cross-module state

Tensor handles are meaningful only when their registry, backend, pool, and memory state are shared across the module boundary. The `nodus_tensor_core` shared target is intended to provide that single process-wide home while `canvas_tables` and `canvas_tables_static` consume it.

For the diagnosis, decision history, and dated implementation status, read [`../../../../research/12_substrate_blocker.md`](../../../../research/12_substrate_blocker.md) and [`../../../../NODUS_TENSOR_CORE_EXTRACTION_HANDOFF.md`](../../../../NODUS_TENSOR_CORE_EXTRACTION_HANDOFF.md). Verify the handoff against the current build before following its next step; handoffs are snapshots and may lag active work.

## Historical layout intent

The initial scaffold anticipated abstraction, backend, autograd, operations, pooling, scheduling, diagnostics, runtime, models, and tests. That taxonomy still explains many directory names, but the implementation has evolved beyond the original light-layout plan.
