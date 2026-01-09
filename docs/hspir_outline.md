# HSPIR PathTape Outline

This document consolidates the HSPIR + PathTape conversation into a single outline covering terminology, data domains, and workflows. Use this as the entry point for future agents that continue the abstract spatial program work.

## Core pillars
1. **Atlas / Token layer** – tokens are sequences of atlas edges. Each edge keeps postings for quick occurrence indexing and capacity hints for preallocation.
2. **Metric schema layer** – channels declare kinematics, tool state, semantics, and scheduling information. The schema is stable and extensible.
3. **Tape layer** – `StepTape` stores axis deltas plus channel values in SoA form with provenance per step.
4. **Resources layer** – frames, transforms, fonts, and devices bind the abstract tape to concrete deployment targets.
5. **Programs & bundles** – tokens compile into `Program` artifacts referencing schema, atlas, resources, and the final tape; sidecars carry provenance and embeddings.
6. **Backends** – raster, mesh, CNC, robot exporters consume the shared tape and honor orientation, metric, axis_mask, and tool constraints.

## Execution workflow
1. Build atlas/node graph & token paths.
2. Shape tokens (Harfbuzz) and outlines (FreeType) to produce MoveEdges with MetricSlices.
3. Compile token path into `StepTape` with schema metadata, axis deltas, and provenance.
4. Schedule tape respecting lockouts (axis_mask, lock_group, sync_id).
5. Export to desired backend (texture rasterizer, mesh builder, G-code emitter, robotic trajectory) via the shared tape format.

## Future-proof strategies
- Keep IDs and hot structs header-only with strong typedefs (TokenId, NodeId, FrameId).
- Tape is immutable once finalized; builders mutate temporary buffers.
- Serialization uses chunked binary/JSON descriptor so future schema changes remain backward compatible.
- Orientation via quaternion internally but export conversions to axis placements for STEP compatibility.
- Device models and kinematic chains live in the resources layer so post-processors can map unrealized axes to joints.

## Next additions (sketch)
- `TapeBuilder` class to populate axis/channel arrays and validate invariants (winding constancy, axis_mask non-empty).
- `ProgramBundle` container tying schema, atlas, resources, programs, and sidecars.
- Backends/interfaces for export: `RasterBackend`, `MeshBackend`, `CNCBackend`, `RobotBackend`.
- Validators for schema presence, lock group conflicts, and quaternion normalization.
- Documentation for the HSPIR file/container format (Header, Schema, Resources, Programs, Sidecars).

This outline maps directly to the conversation and should remain the unifying spec reference for the abstract spatial program feature.
