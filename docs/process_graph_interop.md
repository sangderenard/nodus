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
Nodus TensorMath. Each tool consumes one or two `AbstractTensor*` values,
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
ctest --test-dir build -C Release -R "graph_ir|canonical_ops|bitops_lowering" --output-on-failure
```
