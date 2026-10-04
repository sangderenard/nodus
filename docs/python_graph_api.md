# Pythonic native canvas graphs

`nodus_graph` is an idiomatic Python construction and inspection layer over
Nodus's existing Canvas, ThreadManager, row stack, and `EdgeTensorFifo`. It is
not a second graph model or scheduler. A `Graph.step()` submits the real native
tick and waits at an explicit headless observation barrier before returning.

## A scalar row program

```python
from nodus_graph import BuiltinTool, Graph, Input, Output, Tool

with Graph() as graph:
    module = graph.add_module("(a-b)*c", stable_id=0xA11CE)
    module.set_rows(
        Input(2), Tool(BuiltinTool.SUBTRACT),
        Input(1), Tool(BuiltinTool.MULTIPLY), Output(1),
    )
    graph.bind_input("a", module.input(0))
    graph.bind_input("b", module.input(1))
    graph.bind_input("c", module.input(2))
    graph.bind_output("result", module.output(0))
    graph.push_f32("a", 9.0)
    graph.push_f32("b", 4.0)
    graph.push_f32("c", 3.0)
    graph.step()
    assert graph.pull_f32("result") == 15.0
```

Rows retain native order. Input attachments push from contact order, binary
tools pop `b` then `a` and calculate `a op b`, and Output attachments pop from
the native module stack. Python validates underflow and requires an empty final
stack before installing the complete row program through
`gp_canvas_set_module_rows`.

Contacts use the native unified index space: inputs are `0..input_count-1` and
output ordinal `n` is `input_count+n`. A semantic row index is not a contact
index. Stable module UUIDs survive CANVAS V4 save/load; after loading, prefer
`graph.module_by_id(uuid, input_count=..., output_count=...)` over a load-local
module index.

## FIFO configuration is optional, not zero-sized

`Graph.connect(source, target, fifo=FifoConfig(...))` configures the native
bounded edge FIFO. `slots` must be positive. `top_k == 0` retains the native
reject/backpressure behavior when unread data fills the ring; positive `top_k`
allows native old-reader advancement/drop behavior. Use `fifo=None` to represent
an intentionally unconfigured edge. Execution refuses such an edge explicitly;
it is never converted into a mandatory queue or a zero-capacity sentinel.

Named input/output ports are real table edges with the same fixed token shape,
dtype/layout fields, subscriptions, and first-token semantics. The initial
surface exposes exact four-byte scalar tokens as `push_f32`/`pull_f32`; raw or
tensor-token helpers should only be added with an equally exact ownership and
shape contract.

## Native interchange and boundaries

`Graph.save(path)` and `Graph.load(path)` use native CANVAS V4 persistence.
Nodus also persists full table/FIFO-side state in its existing `.gpmod`
sidecars under `module_library`; the canvas text alone is not the whole native
document. Headless creation disables interactive autosave and clears the
default UI workspace so Python execution cannot mutate that active document.
Host port bindings are process-local views and must be rebound after load.

The following limits are deliberate and raise `UnsupportedNativeContract` or
remain outside this initial surface:

- Only one live graph per process: canvas contact lookup and the ThreadManager
  global are still process-global.
- Ordinary Canvas Input/Output rows carry scalar values. Registered
  `abstract_tensor.*` ToolIR consumes owned `AbstractTensor*`; Python refuses
  to mix those pointer and scalar currencies.
- Canvas, ProcessGraph/GraphIR, SSA, and KernelIR retain different contracts.
  This API does not flatten them or claim that a native canvas save is a
  lossless GraphIR/SSA translation.
- The Turing straight-line Canvas/SSA bridge still has its documented control
  flow and shape limits. Canonical operation IDs, Turing C-dispatch ordinals,
  handler IDs, and canvas/table contact IDs remain distinct namespaces.
- `run_to_quiescence` reports unread work at its exposed black-box boundary;
  it is not proof that every internal cyclic or delayed graph has terminated.
- Native module skip cadence, slip readiness, effects, ownership, and
  transaction snapshots are not reimplemented here. Future Python properties
  must bind the existing native authority or reject unsupported use.

## Verification

```powershell
cmake --build build --config Release --target nodus_headless nodus_headless_runtime_test
ctest --test-dir build -C Release -R "nodus_headless_runtime_test|nodus_headless_bridge_smoke|nodus_python_graph_test" --output-on-failure
```

The acceptance test covers operand/stack order, first-token visibility,
repeated ticks, two-module FIFO delivery, stable-identity save/fresh-load,
pointer-contract refusal, optional/unconfigured FIFO errors, and the current
single-live-canvas restriction.
