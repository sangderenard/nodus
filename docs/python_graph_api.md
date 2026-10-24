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

The native transport state is inspectable without inventing a Python queue:

```python
state = graph.port_fifo_state("input")
assert state.config == FifoConfig.scalar_f32(slots=2, top_k=0)
assert state.unread == 0

first = graph.try_push_f32("input", 1.0)
assert first.accepted and not first.dropped
```

`try_push_f32` and `try_push_bytes` report the native publish decision. A full
lossless FIFO returns `PublishResult(accepted=False, dropped=False)`. A lossy
FIFO may return `accepted=True, dropped=True` after advancing an old reader.
`push_f32` remains the compatibility convenience and raises when a valid push
is rejected. `edge_fifo_state` returns `FifoState(config=None, unread=None)`
for a deliberately unconfigured edge; this preserves absence as distinct from
any configured capacity.

CANVAS V4 does **not** serialize a configured edge's FIFO spec, subscriptions,
reader positions, or samples. After `Graph.load`, attach every restored edge
and make the transport choice again:

```python
producer = graph.module_by_id(producer_id, input_count=2, output_count=1)
consumer = graph.module_by_id(consumer_id, input_count=2, output_count=1)
graph.attach_edge(
    producer.output(0), consumer.input(0),
    fifo=FifoConfig.scalar_f32(slots=7, top_k=0),
)
```

Execution fails while any restored edge remains unattached. Passing
`fifo=None` to `attach_edge` records deliberate absence and continues to fail
at execution; it never invents a persisted/default queue.

Named input/output ports are real table edges with the same fixed token shape,
dtype/layout fields, subscriptions, and first-token semantics. Exact fixed-size
bytes can be published and consumed with `try_push_bytes`/`push_bytes` and
`pull_bytes`; the native configured `token_bytes` remains authoritative.
These helpers copy bytes across the C ABI and make no tensor ownership claim.
`push_f32`/`pull_f32` are the exact four-byte scalar conveniences used by the
current row executor.

## Read-only native stack inspection

The native scheduler already records stack snapshots at `StackDisplay` rows
and publishes a final module stack tail. Python exposes those records directly:

```python
module.set_rows(Input(2), Tool(BuiltinTool.STACK_DISPLAY), Output(1))
graph.push_f32("input", 4.0)
graph.step()
snapshot = module.stack_snapshot(1)
assert snapshot.values == (4.0,)
assert snapshot.items[0].element_bytes > 0
assert module.stack_tail() == ()
```

Snapshots preserve native bottom-to-top value order and the recorded native
type ID/element-byte metadata. A valid non-display row has no snapshot and
returns empty tuples. This is observation after a completed headless step,
not arbitrary stack mutation and not persisted runtime state.

## Native interchange and boundaries

### Complete Python/Turing graph definitions

`Graph.definition()` returns an immutable `nodus.graph-definition.v1`
description of every property installed through this headless Python surface:

- persistent module identity, label, position, size, ordered rows, tool/plugin
  identity, stack effects, and value contract;
- connectors named by `(module stable id, input/output, ordinal)` rather than
  load-local native indices;
- edge endpoints, native type id, and the explicit optional FIFO contract,
  including `storage="host"` or `storage="vulkan"`;
- scheduler mode; and
- named input/output bindings, their FIFO contracts, and output reader start
  policy.

The definition is the communication object for Turing/Python. It does not
contain a ProcessGraph and does not execute anything itself:

```python
from nodus_graph import Graph, GraphDefinition

wire_value = graph.definition().to_mapping()  # plain Python transport value
received = GraphDefinition.from_mapping(wire_value)

with Graph(scheduler=received.scheduler) as native_graph:
    native_graph.apply_definition(received)
    assert native_graph.definition() == received
```

Applying a definition invokes the existing native module, row, edge, FIFO, and
port APIs. Nodus remains the execution authority. Unknown schemas, duplicate
module identities, missing modules, out-of-range connector ordinals, reversed
edges, mismatched port directions, and row-contract disagreement are refused.

`module.export_to_root(path)` exposes the existing native per-module table
serializer at `<path>/serialized/module_<index>.gpmod`. It is useful for
materialization and provenance checks and does not invent a second persistence
format.

This V1 contract is complete for the executable headless surface exposed by
`nodus_graph`. Interactive-only overlays, lasso/meta-rope presentation,
in-flight FIFO samples, reader cursors, GUI actions, and table simulation state
are outside it and are not silently represented. Native CANVAS V4 persistence
still has the separate limitations below.

`Graph.save(path)` and `Graph.load(path)` use native CANVAS V4 persistence.
Nodus writes table topology sidecars under the shared `module_library` path;
they are not per-canvas bundles, and their edge records do not contain FIFO
configuration or live transport state. The canvas text alone is therefore not
an independently portable native document. Host ports and edge FIFO choices
are process-local views and must be rebound after load.

Headless construction refuses to start while a native canvas singleton already
exists, so it cannot replace the UI's process-global manager. The underlying
interactive constructor still reads the default workspace before the headless
layer disables autosave and clears it in memory; no-write isolation is
enforced, but strict no-read isolation is not claimed.

The following limits are deliberate and raise `UnsupportedNativeContract` or
remain outside this initial surface:

- Only one primary canvas/headless graph per process: canvas contact lookup and
  table helpers are still process-global. A pre-existing UI/native canvas makes
  headless creation fail; a second Python `Graph` is refused as well.
- The row-stack surface includes scalar values and registered typed values.
  FIFO payload typing and module row-stack currency are independent choices.
  `AbstractTensorHandle` crosses ordinary Input/Output rows when the module's
  explicit `abstract_tensor_pointer` execution contract selects typed
  FIFO-to-`RawStackFrame` transfer; a registered FIFO `type_id` alone never
  selects that path, and the handle is never coerced through `float`.
  Scheduler-produced
  `StackDisplay` snapshots and final scalar tails remain read-only. Arbitrary
  stack mutation, clone ownership, and unresolved plugin execution are not
  exposed.
- FIFO support covers fixed-size transport tokens, native `slots`/`top_k`, and
  an explicit host or Vulkan storage backend. Setting the backend rehomes the
  actual ring allocation; Vulkan graph adoption shares one device context and
  native readback reports the effective storage. This property does not by
  itself change a generated shader row from handle-token transport to direct
  payload-SSBO binding.
  Logical tensor rank/shape can still be runtime-defined through connector
  `TensorContract.shape_sources`. Python's
  scalar executable helper is exactly four-byte `float32`; raw helpers copy an
  exactly configured number of bytes. Tensor handles retain explicit caller
  ownership; every native
  delta/by-reference/drop/slip policy, internal-edge start positions, reader
  management, and live queue checkpointing are unsupported. `type_id`, layout,
  and dtype are passed through rather than inferred.
- When typed stack transfer is selected explicitly, a FIFO sample must exactly
  match its registered `ValueType` byte width. The scheduler advances the
  reader only after the complete element reaches the module's `RawStackFrame`,
  and advances the writer only after the typed result leaves it. Typed FIFOs
  remain valid with scalar or opaque module contracts and do not imply this
  transfer mode.
- Canvas, ProcessGraph/GraphIR, SSA, and KernelIR retain different contracts.
  This API does not flatten them or claim that a native canvas save is a
  lossless GraphIR/SSA translation. Native editor topology, labels, positions,
  tool bindings, and UUIDs use native persistence; Python row metadata and host
  bindings are not automatically reconstructed.
- The Turing straight-line Canvas/SSA bridge still has its documented control
  flow and shape limits. Canonical operation IDs, Turing C-dispatch ordinals,
  handler IDs, and canvas/table contact IDs remain distinct namespaces.
- `run_to_quiescence` reports unread work at its exposed black-box boundary;
  it is not proof that every internal cyclic or delayed graph has terminated.
  Input reader identity varies by tool while undrained output remains externally
  unread, so this is best-effort boundary state, not graph completion.
- Native module skip cadence, slip readiness, effects, ownership, and
  transaction snapshots are not reimplemented here. Future Python properties
  must bind the existing native authority or reject unsupported use.
- Configuration, save/load, row edits, and port declaration require no tick in
  flight. `Graph.step()` waits through tool execution and manager snapshot
  publication, but concurrent native mutation is not made safe.
- Live runtime state is not persisted: queued requests, FIFO samples, reader
  cursors, event queues, ledgers, and in-flight effects are lost. Old Python
  `Module` handles become explicitly stale after a successful load.

## Verification

On Windows this build links some test executables against LibTorch supplied by
the existing Python package. The observed installation is PyTorch
`2.5.1+cu124`; its DLLs live under Python's `site-packages/torch/lib` and the PE
imports include `torch_cuda.dll`, `torch_cpu.dll`, and `c10.dll`. Headers and
import libraries being found at configure/link time does not put those DLLs on
the Windows runtime search path. CTest now prepends `${TORCH_INSTALL_PREFIX}/lib`
and the target's build directory **for the test process only** for GraphIR,
ThreadManager, and the headless tests. No user/system `PATH` change is needed.

For an equivalent one-off PowerShell launch outside CTest:

```powershell
$torchLib = python -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')"
$env:PATH = "$PWD\build\Release;$torchLib;$env:PATH"
& .\build\Release\test_graph_ir.exe
```

That environment proves DLL resolution and the test's own assertions only. It
is not evidence that every LibTorch/CUDA backend operation is numerically
implemented or equivalent to Python Torch.

```powershell
cmake --build build --config Release --target nodus_headless nodus_headless_runtime_test test_edge_fifo_drop_contract
ctest --test-dir build -C Release -R "nodus_headless_runtime_test|nodus_headless_bridge_smoke|nodus_python_graph_test|edge_fifo_drop_contract_test" --output-on-failure
```

The Python acceptance executable runs thirteen named cases covering operand/stack
order, first-token visibility, repeated ticks, two-module FIFO delivery,
read-only native stack snapshots, lossless rejection versus lossy dropping,
configured versus absent FIFO inspection, invalid calls,
stable-identity save/fresh-load, explicit restored-edge FIFO reattachment,
failed-load preservation, stale handles, typed-contract scheduler admission,
optional/unconfigured FIFO errors, duplicate UUID refusal, and the current
single-live-canvas restriction, plus runtime-shape source mapping.

`turing_nodus_graph_crossing_test` additionally compiles a real row through
Turing, derives its connector contracts from the retained buffer ABI, proves
mapping/native concordance, and executes tensor handles through the scheduler.
