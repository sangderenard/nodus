# KPath Shaper Layer

This document records the new focus on codepoint→cluster→glyph extraction, which feeds the rest of the kpath pipeline.

## Goals
- Accept a `CodepointSequence` (Unicode scalars) and produce clusters that group glyphs/advances (HarfBuzz semantics).
- Store glyph outlines as collections of `OutlineSegment`s produced by FreeType so the tool kernel can later sweep along them.
- Keep the shaper interfaces `AbstractTensor`-friendly: they should operate on vectorized arrays (codepoint batches) and produce metadata (cluster ranges, glyph IDs) for q-batching/autograd.

## Interfaces
- `Cluster` holds a codepoint span, glyph IDs, and advance vectors.
- `GlyphOutline` keeps draw instructions (move/line/quad/cubic/close) so downstream `Kernel` or `StepTape` builders can treat glyphs as mini armatures.
- Those segments map directly to the existing `vector_ops` kernels (move, quad, cubic, etc.), so you can materialize glyph metadata as the same language that describes other operation-coded tool paths.
- `GlyphCache` maps glyph IDs to outlines for reuse across tokens.
- `Shaper` stubs `shape()` and `extract_outline()`, which will later call HarfBuzz/FreeType while keeping the pipeline data-structure oriented (vectorized input, cluster metadata output).

## Atlas wiring
- When you build tokens, insert `AtlasNode`s for each glyph; keep their `GlyphOutline` metadata (move/line/quad/cubic/close ops) plus a `RotDir`-based winding flag so downstream renderers know how to fill holes.
- The `AtlasEdge`s between those nodes carry HarfBuzz advances and offsets, describing how each glyph sits relative to its predecessor in a realized string. That is the “layout offset edge” that lets a string token become a path plan and allows stencil-backed backends to treat glyph sequences as tool paths.

Next steps: implement `Shaper::shape` to drive HarfBuzz, use `GlyphCache` to store outlines, and feed outlines plus cluster-level contexts into `kpath_tape` and the tool kernel.
