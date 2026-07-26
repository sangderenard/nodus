# Turing ProcessGraph interoperability

Nodus receives Turing computation through two complementary paths:

```text
ProcessGraph -> GraphIR -> AbstractTensor tool nodes and ports
SSA / PrimitiveProgram -> KernelIR or native calculator execution
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

These emit ordinary Nodus `GraphEdit` records. Canonical operations are marked
with their arity and KernelIR-lowerability. Structural ProcessGraph nodes remain
visible and non-canonical.

This establishes graph composition and introspection. Binding each canonical
node to an executable `ToolIR::execute_stack` implementation remains a separate
step because tensor-handle ownership, descriptors, backend selection, and
multi-output lifetime must be explicit.

## Verification

```powershell
cmake --build build --config Release --target test_graph_ir canonical_ops_test bitops_lowering_test
ctest --test-dir build -C Release -R "graph_ir|canonical_ops|bitops_lowering" --output-on-failure
```

