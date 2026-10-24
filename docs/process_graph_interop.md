# Turing ProcessGraph interoperability (historical experiment)

The direct ProcessGraph-to-Nodus route is not a production integration target.
It proved canonical operation correlation and exposed unresolved structural
nodes, but it conflated Turing's compiler input graph with Nodus's editable
runtime graph. Turing owns ProcessGraph-to-SSA lowering and should execute the
resulting SSA product through an ordinary Nodus row tool.

The supported boundary is now:

```text
Turing source -> ProcessGraph -> SSA -> deployable row tool
Nodus graph definition -> native Canvas/Table modules, connectors, properties, and ports
```

Nodus graph composition and UI structure travel through
`nodus.graph-definition.v1`; compiler computation does not travel by turning a
ProcessGraph into Nodus modules.

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

## No production ProcessGraph intake

`turing_process_graph_intake_test` remains useful regression evidence for the
old GraphIR vocabulary and canonical operator IDs. It is not a roadmap for a
manifest, adopter, or broader ProcessGraph importer. Control flow, unresolved
calls, stateful nodes, reductions, and compiler value identity stay in Turing's
SSA lowering rather than acquiring parallel Nodus graph semantics.
