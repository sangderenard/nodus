# Turing ProcessGraph interoperability

Nodus receives Turing computation through two complementary paths:

```text
ProcessGraph -> GraphIR -> registered AbstractTensor ToolIR nodes and ports
SSA / FusedProgram -> KernelIR or Nodus in-memory calculator execution
```

They are deliberately separate. The graph/tool route preserves composition
and UI structure. The KernelIR/calculator route is an execution packet suitable
for fusion.

## Canonical operation identity

`ops/canonical_ops.json` is the operation correlation table. Every catalog
entry has an append-only canonical ID generated from its array position.
Reordering or inserting entries is an ABI break; new entries append.

`ct_value` is separate. It records the verified ordinal of operations currently
implemented by Turing's C dispatcher. This prevents missing C operations from
being unrepresentable in KernelIR.

KernelIR stores canonical IDs in `Instruction.sub_op`. Operands contain only
values and immediates. Nodus BitOps lowering follows this contract; it no longer
places a private operation enum in the operand list.

Regenerate and verify:

```powershell
python ops/generate_canonical_ops.py
python ops/generate_canonical_ops.py --check
python ops/verify_canonical_ops.py
```

## AbstractTensor graph tools

`make_abstract_tensor_graph_ir_ops` adds:

- `tensor_node(op)`
- `tensor_input(node, role)`
- `tensor_output(node, role)`

These emit ordinary Nodus `GraphEdit` records. Canonical nodes use concrete
tool ids such as `abstract_tensor.add`; structural ProcessGraph nodes remain
visible as `abstract_tensor.structural`.

`register_abstract_tensor_tool_ir` registers stack-executable ToolIR instances
for the 28 canonical F32/F64 elementwise operations currently implemented by
Nodus TensorMath. Each tool consumes one or two `AbstractTensorHandle` values,
allocates its result in the input backend, executes the same
`tensor_elementwise_*` semantics used by `InMemoryCalculator`, and pushes the
result. The graph description and the executable registry therefore agree on
the tool id rather than being correlated by an attribute-only convention.

Structural input/constant/return nodes and non-elementwise operations remain
explicit translation boundaries. They need node-instance binding/state and
must not be disguised as stateless elementwise tools.

## Verification

```powershell
cmake --build build --config Release --target test_graph_ir canonical_ops_test bitops_lowering_test
cmake --build build --config Release --target tensor_graph_execute
ctest --test-dir build -C Release -R "graph_ir|canonical_ops|bitops_lowering|turing_process_graph_intake" --output-on-failure
```

`turing_process_graph_intake_test` invokes Turing's sanctioned
`method_to_graph_ir` shallow interpreter, gives the exact emitted GraphIR to
Nodus, executes a noncommutative three-input tensor expression through the
registered tools, and verifies the numerical result. It also proves that an
unresolved two-input call remains a visible structural node and is refused
rather than guessed.

## Remaining production intake boundary

The test proves the wire grammar, canonical tool resolution, handle ABI, and
numerical execution together. It does not turn the test executable into a UI
or runtime importer. A production intake product still needs:

- a versioned manifest retaining the GraphIR text, Turing provenance, ordered
  named inputs, and explicitly named outputs;
- dtype, rank, fixed/runtime extent, and ownership contracts for each boundary
  value, using the same tensor ABI metadata as compiled row products;
- explicit result identities rather than selecting the last unconsumed node;
- multi-output and constant-tensor representation instead of the current
  scalar-constant test materialization;
- an adopter that applies the emitted `GraphEdit` records to Nodus's existing
  editable graph surface and binds canonical nodes to registered tools; and
- persistence of that adopted source/artifact relationship so a saved table
  can show where its graph and tools came from.

Control flow, unresolved calls, stateful/structural nodes, dimensional
reductions, and operations outside the registered canonical subset remain
named refusal boundaries. They must not be treated as elementwise pass-through
nodes merely to make a graph load.
