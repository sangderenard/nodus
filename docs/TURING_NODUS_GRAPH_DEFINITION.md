# Turing/Python to Nodus graph definition

## Boundary

Turing compiles computation. Nodus owns the editable execution graph.

```text
Turing source -> ProcessGraph -> SSA -> deployable row tool
                                      |
nodus.graph-definition.v1 ------------+-> native Canvas/Table execution
```

`nodus.graph-definition.v1` describes how row tools inhabit a Nodus graph. It
does not transport Turing ProcessGraph nodes and it does not duplicate Nodus's
scheduler.

## V1 contents

The immutable Python `GraphDefinition` carries:

- scheduler mode;
- modules with stable identity, label, rectangle, ordered rows, builtin/plugin
  identity, plugin stack effects, and scalar/tensor/opaque contract;
- connectors identified by stable module identity, direction, and ordinal,
  with names and logical tensor dtype/shape contracts (including explicit
  connector/axis sources for runtime extents);
- edges with output/input connector references, native type ID, and explicit
  FIFO presence or absence plus token size, capacity, top-k, layout, dtype, and
  storage backend (`host` or `vulkan`);
- named input and output ports with the same FIFO properties; and
- output reader start policy.

`to_mapping()` produces a plain nested Python value suitable for an in-process
Turing handoff or an independently chosen envelope. `from_mapping()` validates
the schema and reconstructs the immutable definition. No transport protocol is
prescribed by Nodus.

## Native adoption

`Graph.apply_definition()` accepts only an empty graph. It creates modules and
installs ordered rows through the native headless ABI, connects their native
contacts, configures native FIFOs, and declares native ports. Explicit module
rectangles and labels are restored after table-row materialization, so the
communicated properties remain authoritative rather than being replaced by
the editor's interactive auto-fit policy.

`Graph.definition()` is native readback, not a replay of the Python definition
used for adoption. It enumerates the live Canvas modules, descriptors, stable
IDs, contracts, ordered rows and plugin stack effects; reads scheduler mode,
edges and Table FIFO configuration; and enumerates named headless ports in
declaration order. Logical connector tensor metadata belongs to the Python
communication schema, but every connector reference is checked against the
actual native row-derived arity before it is returned.

The scalar acceptance test constructs a two-module add/multiply graph,
serializes and parses its mapping, adopts it into a fresh native graph, checks
exact definition equality, deliberately corrupts the Python-side scheduler,
module, edge, and port caches, checks equality again from native readback, and
executes `(2 + 3) * 4 == 20`.

The Turing crossing compiles `calculate(left, right)` through the sanctioned
SSA compiler entry to a C row product. `graph_definition_from_row_product()`
derives the module rows, stable connectors, connector names, logical tensor
contracts, typed FIFOs, and named ports from `NODUS BUFFER ABI V2`; no caller
repeats that wiring. The native graph returns the identical definition and
executes `9.5 + 4.0 == 13.5` through `AbstractTensorHandle` ports.

Programmatic row adoption also materializes the real attached Nodus table
without depending on UI focus. `Module.export_to_root()` exposes the existing
module serializer through the headless boundary; the acceptance test requires
the exported table blob to contain its ordered Input/Tool/Output row labels.
This guards the empty-table defect found during visual proof. The retained
artifact at `out/turing-web-proof-v4/turing-created-nodus-graph.png` shows two
concrete Turing-created tables and their connecting cable.

The module's communicated `abstract_tensor_pointer` execution contract—not
the FIFO `type_id`—selects the `RawStackFrame` transfer path. FIFO schema and
row-stack delivery are deliberately orthogonal; the scalar regression uses a
registered typed FIFO without activating stack transfer.

## Refusal boundary

V1 is the complete contract for the current executable `nodus_graph` headless
surface. It does not claim to carry interactive overlays, lasso/meta-rope
presentation, GUI actions, table simulation state, queued FIFO samples, reader
cursors, or in-flight effects. Those require existing native authorities to be
made inspectable before a later schema can include them.

Native CANVAS V4 is not used as this communication contract because it does
not retain FIFO configuration, named host ports, or live transport state.

## Runtime-shaped and SPIR-V crossings

The same definition derivation is proved for real runtime-rank-1 and
runtime-rank-2 C products, including their `shape_sources`. Turing's
`element_count` extent is represented explicitly as `ELEMENT_COUNT` in Buffer
ABI V2 and the native wrapper reads `NodusTensorDesc.element_count`; rank-two
execution therefore covers every element rather than confusing total count
with axis zero.

The equivalent LLVM products are crossed for runtime rank 1 and rank 2. The
Windows linker now retains the descriptor-declared row entry through a focused
`.def` file, and a mismatched rank-2 call proves validation code 77 reaches the
scheduler without publishing an output.
The direct SPIR-V lane is crossed too. Nodus strictly adopts
`turing.spirv-product.v1`, retains its exact module, validates the physical
resource and launch contract, generates an ordinary registered row, transfers
typed tensor handles through `RawStackFrame`, and dispatches through the
existing Vulkan runner. The crossing publishes 1000 values (deliberately not
divisible by local size 64), checks the numerical result, and verifies the
published past-end guard with an extra workgroup. Tensor mapping accepts arena
capacity greater than logical payload size; logical bytes remain governed by
`NodusTensorDesc` and the product record.

FIFO `storage="vulkan"` is now an additive V1 property with native round-trip
readback. The Vulkan backend keeps persistent SSBO allocations, and the
two-shader acceptance moves a 256-byte packet through `EdgeTensorFifo` by
native device-to-device transfer using the established `push_from_frame` and
`pop_to_frame` operations. Vulkan-to-Vulkan transfer is fail-closed rather
than host-staged. The ordinary generated SPIR-V row wrapper also retains the
independent `AbstractTensorHandle`/`RawStackFrame` option; it is not a missing
FIFO slot API.
