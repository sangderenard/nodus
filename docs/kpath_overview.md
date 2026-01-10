# KPath Overview

`kpath` unifies all the recorded use cases: a **tool kernel** is swept along a **metric-rich path** whose **pose** comes from an **armature (kinematics) engine**. Raster fonts, meshes, machining, and robotic calligraphy are all just different backends that integrate the same kernel coverage data, so the work stops being a text vs robot debate and becomes kernel integration under generalized coordinates `q` with schema-tuned controls.

Key points:

- **Armature vector sum** (`kpath_kinematics.h`) treats CNC axes or robot joints as the same object: `ArmatureModel` is a list of `Joint`s; `ArmatureSolver` accumulates them into a tool `Pose(q)` by ingesting vectorized `AbstractTensor`-style arrays so batches of identical armatures can solve in one pass and gradients can be traced automatically. Beam-theory metrics (mechanical slack, bending stiffness, shear limits) stay with each `Joint` so the solver knows about re-engagement behavior and structural compliance.
- **Tool kernel** (`kpath_kernel.h`) captures the local footprint (analytic shapes or discrete occupancy) so downstream backends know how to compute coverage/cut/deposition.
- **Tensorized solver**: the armature solver is expected to run on vectorized `AbstractTensor` batches so multiple identical robots/CNC rigs can advance in lockstep while keeping autograd trackers aware of tool poses.
- **Metric schema** (`kpath_schema.h`) makes rotation direction a first-class channel (winding_dir) rather than a geometric axis; tool mode/power, scheduling (lock_group/axis_mask), and orientation channels live beside provenance metadata so backends can interpret the tape deterministically.
- **Step tape** (`kpath_tape.h`) is the SoA medium: per-axis deltas, per-channel spans, and provenance per step. Builders (`StepTapeBuilder`) populate it and enforce simple bounds.
- **Atlas and bundle** track tokens, postings, layout edges, and glyph metadata so you can serialize a full artifact and later replay it across backends. `AtlasEdge`s now carry HarfBuzz layout offsets (advance/offset per glyph) so realized strings know how letters align, while `AtlasNode`s own FreeType-derived `GlyphOutline`s plus winding direction so stencil backends can generate fills with holes or export to mesh/machining/kernel targets.
- **Armature graph** (`kpath_armature_graph.h`) reuses the *same* sparse-tensor graph implementation (`kpath_atlas`) but as a separate instance. Glyphs, posting-tracked tokens, and tool paths live on an atlas graph, while joint/constraint data live on a distinct armature graph that shares the same storage primitives without mixing their semantics.
- **Graph philosophy**: The two graphs are mathematically identical, but their edges record different things. HarfBuzz layout edges describe tool-tape semantics while armature constraint edges describe linkages. Both are just different labelings of an abstract sparse tensor graph, which keeps the infrastructure unified but the domains distinct.

This doc should be the crisp explanation you hand off: kernel + path + pose + metrics = the unified substrate.
