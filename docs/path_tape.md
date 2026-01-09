# PathTape (abstract spatial program)

PathTape is the distilled subfeature of the abstract tensor suite that models atlas-driven program paths as a scheduled, metric-rich tape. It unifies tokens, atlas edges, and per-step semantics so every execution trace can be expressed as a deterministic sequence that still exposes provenance (which token and edge produced a step).

Key concepts:

1. **Atlas paths -> TokenPath**  
   Every token in the abstract tensor stack is represented as a path through composable atlas edges. `TokenPath` records the shared sequence with optional capacity hints so editors and allocators can pre-reserve room for growth.

2. **Channel-first tape**  
   `StepTape` stores axis deltas and metric channels as separate SoA arrays. Channels are labeled with `ChannelKind` so downstream components (renderers, machines, robot planners) can decide which values matter without decoding opaque blobs.

3. **Provenance awareness**  
   Each tape step can map back to its originating token/edge offset via `StepProvenance`. That makes deduplication, reverse lookups, and embedding-indexing straight-forward.

4. **Schema-friendly growth**  
   `ChannelDef` and `TapeLayout` keep the hot path narrow. New channel kinds can be introduced without breaking existing readers.

How it fits under the abstract tensor suite:  
- The atlas/paths come from tensor suites that already produce sparse graphs or COO backed tensors.  
- The tape exposes motion semantics for the same sparse data, making it easy to turn tensors into raster, mesh, CNC, or pose backends.  
- PathTape is intentionally lightweight so it can be extended later with richer metric/schema layers (e.g., HSPIR v0.1) once serialization and builders are in place.

See `include/common/tensors/abstraction/path_tape.h` for the first pass on the public surface.
