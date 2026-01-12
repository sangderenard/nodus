# RelGeo stencil pass (qualitative layout)

This pass derives a non-numeric "stencil" from a `RelProgram`. It captures partial
ordering and sign constraints without assigning coordinates. The goal is to narrow
the domain for any later numeric solver while keeping purely relational reasoning.

## Vocabulary

- **Frames**: local coordinate systems with N axes. Frames are symbolic and may be
  related (parallel/perpendicular) without a global orientation.
- **Axis order**: `a < b`, `a = b`, `a > b` along a specific axis in a frame.
- **Axis sign**: `negative`, `zero`, `positive` for a point along a frame axis.
- **Between**: `a - mid - b` along a frame axis (strict ordering).
- **Frame relations**: axis-to-axis relationships between frames (parallel or perpendicular),
  with optional "same vs opposite direction".

If `include_local_frames` is enabled, every point gets its own local frame. These
frames are unconstrained unless other rules relate them to geometry.

## Sources of constraints

- `RelAssertCoincident`: equality on all axes in the global frame.
- `RelAssertPointOnLine`: zero on the line frame's perpendicular axis.
- `RelAssertPerpAt(v,a,b)`: create a local frame at `v` with axis0 along `v->a`
  and axis1 along `v->b`; infer signs (quadrant only in that frame).
- `RelAssertParallelLines` / `RelAssertPerpendicularLines`: relate line frames.
- `RelPointLerp`: establish "between" on the segment frame for `a-b`.
- `RelPointOffset`: if numeric offsets exist, constrain relative order on global axes.
- `RelPointLineLineIntersection`: enforce on-line constraints in both line frames.

## Non-goals (current)

- No numeric placement.
- No global quadrant/absolute orientation without explicit relations.
- Minimal propagation (equality and direct constraints only).

This pass is intended to grow into a richer symbolic layout engine that cooperates
with e-saturation and numeric solving.

## Grid seeding

A follow-on grid pass can turn the stencil into a discrete lattice:

- Each frame/axis assigns an integer index to every point.
- Equality constraints collapse points to the same index.
- Order/between constraints induce a topological ordering.
- Unconstrained points get unique indices to avoid accidental equivalence.

The grid can be serialized as a sparse-graph edit log (numeric attrs only) to
feed later solver stages.
