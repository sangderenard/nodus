# KPath Handoff Checklist

## Implementation TODOs
- Harden `ArmatureSolver::solve` to accumulate transforms along joints and incorporate beam-theory joint metrics (mechanical slack, bending/shear stiffness) into the pose prediction.
- Add real channel storage management in `StepTapeBuilder` (separate buckets for different dtypes, schema-driven channel indexing, etc.).
- Implement posting append logic in `AtlasBuilder::add_posting` and ensure `Atlas` stores spans for tokens.
- Flesh out `ResourceTable` with device/armature metadata (axis bindings, joint constraints) and tie into `BundleBuilder`.
- Add validators that catch winding inconsistencies, axis_mask misuse, quaternion normalization, and lock_group conflicts after scheduling.
- Provide a simple “debug backend” that dumps tape steps + channels to stdout for quick verification before wiring real exporters.

## Design reminders
- Armature metrics (mechanical slack, beam-theory stiffness) belong on `Joint` so frameworks can examine expected re-engagement friction.
- The tool kernel is the same concept across raster/mesh/machining/painting—the occupancy weight semantics unify antialiasing and partial material deposition.
- Rotation direction is semantic; encode it via the `winding_dir` channel (±1) rather than a physical axis.
- Keep orientation exportable via quaternion-to-frame conversions so STEP/STEP-NC compatibility is achievable later.
- Atlas tokens must capture HarfBuzz advance/offset metadata on edges and attach glyph outlines with winding to nodes so layout semantics survive mesh/machining/text-stencil emitters.
- Reuse the atlas graph for armatures: declare joint nodes and constraint edges (see `kpath_armature_graph.h`) so tool-path and armature planning share the same abstract tensor representation.

## Next agent handoff
 - Extend `ArmatureSolver` with a traversal of `Joint`s and accumulate transforms; the stub `solve` currently returns position only. Ensure it consumes vectorized `AbstractTensor` data so batches and autograd/backprop works.
- Build a `StepTapeBuilder` that allocates per-channel storage and exposes setters keyed by channel index or name (using `MetricSchema::find_channel`).
- Make `ProgramBundle` serializable (header + blobs) by defining `serialize`/`deserialize` on `Program` and `ProgramBundle`.
- Once tape + resources work, add the first backend that converts `Program.tape` to a textual dump or simplified G-code for sanity checking.
