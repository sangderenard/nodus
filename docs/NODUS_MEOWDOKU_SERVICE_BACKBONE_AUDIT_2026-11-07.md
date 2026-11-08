# Nodus and Meowdoku service-backbone audit

**Date:** 2026-11-07
**Title:** Nodus as the execution backbone for the Python-backed Meowdoku service
**Snapshot:** Nodus `3c793cc`; Turing was moving during inspection and was observed at `0e4b6d3c` before this report was written. These hashes identify the inspected period, not an integration gate. This is a read-only source audit, not a runtime acceptance result.

## Scope

Assess whether Nodus can own the service's scheduling, graph state, and tool execution while retaining authored Python for duties that Turing cannot yet lower. Examine native and Python rows, graph definition, bounded FIFOs, threaded scheduling, GPU execution, current Meowdoku duties, and both dynamical and operational forms of chaos. Identify measured capabilities, source-level risks, and the order of work needed for a service cutover.

## Methodology

- Read the workspace instructions, Speaktome guestbook guidance and relevant Nodus/Turing reports, Nodus continuation and graph-definition handoffs, and the Meowdoku web and continuation reports.
- Inspected Nodus `ThreadManager`, `EdgeTensorFifo`, `nodus_graph`, Python row deployment and interpreter, Turing row adoption, and Vulkan FIFO tests and documentation. Compared them with Meowdoku `server.py`, `startup.py`, `entities.py`, `session.py`, fountain builders, and client compute runtime.
- Queried the live TLS server only with GET requests to `/api/whoami`, `/api/ready`, and static pages. At the final check, `/api/ready` reported `all_settled=true`, 11 products ready, seven fountain CPU fallbacks skipped, and zero builds in progress. The local caller was classified as `lan/edit`; this does not establish a remote visitor's tier.
- Did not edit source, run a build or test suite, exercise a mutating endpoint, or claim runtime reproduction of the source-level findings below.

## Detailed Observations

### Existing integration is real but narrow

Nodus's `nodus.graph-definition.v1` records stable modules, ordered rows, connectors, tensor metadata, named ports, scheduler mode, and optional FIFO properties, including `host` or `vulkan` storage. Native readback rather than a Python-side cache supplies the round-trip evidence. It deliberately omits live queue samples, reader cursors, in-flight effects, and some interactive state. Native CANVAS V4 save/load does not restore the full graph transport contract. See `C:/dev/Powershell/nodus/docs/TURING_NODUS_GRAPH_DEFINITION.md` and `C:/dev/Powershell/nodus/docs/python_graph_api.md`.

Turing can supply deployable C and LLVM row tools with runtime-shaped tensor crossings, and a narrow validated SPIR-V compute product. Nodus has a real two-shader, device-local 256-byte packet crossing through its existing Vulkan-backed `EdgeTensorFifo`. The successful proof is a row and transport proof; it does not establish that Meowdoku's browser WGSL fountain law is already a server Vulkan row. See `C:/dev/Powershell/nodus/docs/SESSION_CONTINUATION_2026-10-23.md`, `C:/dev/Powershell/nodus/tests/vulkan_ssbo_fifo_test.cpp`, and `C:/dev/Powershell/turing/examples/meowdoku/fountain_build.py`.

Nodus also deploys one retained Python source and descriptor explicitly as a live embedded CPython row or as a Turing-translated C, LLVM, or Fortran row (`include/python_row_workflow.h`, `src/python_row_workflow.cpp`). A native translation failure does not silently select CPython. The interpreted ABI currently converts declared primitive scalar tensors to Python scalars and back (`include/python_row_tool.h`, `src/python_row_tool.cpp:784`). I found no established ordinary row protocol for calling a separate hosted Python process. Passing a `python_executable` to the compiler workflow is build-time invocation, not such a runtime protocol.

The live Meowdoku service still owns HTTP and WebSocket ingress, access tiers, cookie-keyed per-client tables, dt-system session save/restore, training, Lichess, product builds, and readiness. `server.py:1030` binds before independent background build lanes; `startup.py` reports per-product readiness; `entities.py` defaults to a native Eigen law product and reports a NumPy fallback on refusal. Browser fountain compute uses a generated WGSL product and a compiler cost verdict, with Wasm where built and available. The current live launch skips fountain Wasm products. See `C:/dev/Powershell/turing/examples/meowdoku/server.py`, `startup.py`, `entities.py`, and `web/compute.js`.

### Source-level service blockers

**A. Partial-cycle deadlock in free-spinning mode.** `ThreadManager::run_scheduled_tick` detects a cyclic module graph and computes an alternate sequential order (`src/thread_manager.cpp:640`). Scheduled mode uses sequential execution for a cycle, but free-spinning mode calls `run_parallel` even when cyclic (`:1594`). The parallel ready queue admits zero-indegree nodes or Slip nodes and waits until `remaining == 0` (`:1522-1589`). For `A -> B`, `B -> C`, `C -> B` with ordinary pooled modes, A runs, then B and C remain blocked by predecessor counts. The queue is empty, `remaining` is 2, and workers cannot finish. This conclusion follows directly from the code; no test was run. `tests/thread_manager_test.cpp` covers a completion barrier with empty tick requests, not this graph.

**B. Tick firing is not yet token readiness.** The scalar Input-row path initializes a value to zero and may push it even without unread FIFO data (`src/thread_manager.cpp:855-945`). The Output-row path can pop a value and ignore a failed publish (`:1420-1477`). `EdgeTensorFifo::ensure_space_for_write` refuses a full lossless FIFO when `top_k == 0` (`include/edge_tensor_fifo.h:813`). Thus a missing action can look like a zero action, and a full response edge can discard a produced result. The current per-tick canvas behavior and a service request/response firing rule are different contracts. This must be measured and resolved inside the existing ThreadManager/FIFO authority before the graph owns service truth.

**C. CPython refusal can be reported as success to the scheduler.** Generated Python-row wrapper code logs when `Interpreter::execute_tensor_stack` returns false but leaves `ToolStackContext.succeeded` at its default true (`src/python_row_tool.cpp:594`, `include/tool_api.h:110`). The scheduler only stops later output rows when that flag is false (`src/thread_manager.cpp:950-991`). The interpreter restores inputs on failure, so a downstream output row could mistake restored input values for returns. The existing plugin integration test exercises a successful call; it does not close this failure path. This is a source-level risk pending a graph-level regression.

**D. Persistence does not yet cover a live service.** `Graph.save/load` persists the native canvas, but the Python graph API explicitly excludes live FIFO samples/cursors and requires restored edges to be reattached (`nodus_graph/native.py:1191`, `docs/python_graph_api.md`). Meowdoku's `MeowdokuSession` uses the dt system's own `copy_shallow` and `restore` (`examples/meowdoku/session.py`), and the server has separate session/lifecycle persistence. A saved topology cannot be treated as a saved game. Nodus has a quiescent edge transaction facility (`include/edge_tensor_fifo_transaction.h`, `include/table_abi.h`); determine its exact fit before adding another persistence scheme.

**E. CPython's current scalar bridge is too narrow for whole sessions.** Game sessions include arrays, controller state, history, profiles, and effects. These are not representable by the current primitive-scalar descriptor. The graph needs an explicit typed ownership boundary for the values it actually passes, and a separate hosted-process protocol only if embedded CPython is insufficient. The GIL is entered for each interpreted execution (`src/python_row_tool.cpp:734`); more Nodus worker threads alone do not make Python bytecode parallel within one interpreter.

### Graph, dynamical, and operational chaos

Kahn-process-network determinacy assumes blocking reads and nonblocking writes ([Kahn's original semantics](https://www.cs.columbia.edu/~sedwards/papers/kahn1974semantics.pdf)). Bounded FIFO implementations need a scheduling policy to preserve useful behavior ([Parks's bounded-scheduling work](https://ptolemy.berkeley.edu/papers/95/parksThesis/)); Nodus's draft `docs/edge_tensor_fifo_lockfree_kpn.md` itself distinguishes its current bounded modes from that ideal. Feedback components need explicit initial state, token rates, capacity/drop policy, and termination or quiescence criteria. Topological scheduling is sufficient only for acyclic dependencies. A Slip hint bypasses predecessor readiness and changes semantics; it should be justified per edge and law.

For nonlinear simulation, distinguish **event determinism** from **numerical agreement**. The former asks whether the same admitted actions, sequence, edge decisions, and state revisions produce the same committed transition log. The latter asks whether Python/NumPy, native CPU, browser WebGPU, and server Vulkan obey their declared precision and physics contracts. The fountain's WGSL build uses the `fast` contract and f32 legalization; its Wasm build uses `develop` (`examples/meowdoku/fountain_build.py`). Different arithmetic order or rounding can change later contacts in a dense pile. The Meowdoku work list already records dropped contacts, an unsettled large pile, and other unverified physical behavior. Do not make a broad numerical tolerance hide a changed law. Measure emitted IDs, contacts/drops, energy, settle state, and stepwise state divergence under fixed inputs and a recorded device/product identity.

Operationally, retries, process restarts, FIFO loss, concurrent Python effects, and live model updates can also create divergent histories. The existing server orders a client's actions under its table lock and retains explicit lifecycle and access checks. A service graph needs a monotonic action identity, a state revision, a commit result, and a rule for effects such as profile changes or Lichess calls so replay does not duplicate them. Nodus's `GRAPH_ENTITY_AND_SPRITE_CONTRACT.md` already sketches revisioned transitions; use the native graph and existing game state authorities rather than inventing a parallel scheduler or dt snapshot convention.

## Analysis

The desired ownership split is feasible: Turing publishes compiled row products; Nodus communicates and runs its own graph; Python remains an explicitly identified row execution mode for work that has not lowered. A thin network ingress may still be a Python process while the Nodus graph is the scheduling and state backbone. The remaining question is which process owns each session and committed effect, not the language of the HTTP accept loop.

This audit should not be read as a claim that current Nodus is already a complete deterministic KPN service. Both scheduler modes enter `run_scheduled_tick`, but the tail selects different parallel/sequential behavior; free-spinning cyclic execution has the concrete deadlock path above. Current typed stack, optional FIFO, and Vulkan transport are valuable and must remain distinct. A Vulkan storage choice does not make every edge a FIFO or every row a shader.

The user's proposed row coloring is useful as execution provenance. A row should display the deployed product's actual mode (`cpython`, C, LLVM, Fortran, SPIR-V/Vulkan), source digest, and status. `plugin_id` and the deployment workflow already retain some identity, but a consistent visual execution-mode indicator was not found in the current row renderer. Color should be a projection of retained deployment metadata, never a guess from the source filename.

## Recommendations

1. **Close four service-semantics regressions in Nodus first:** partial cycle with an upstream node and a bounded wait; absent scalar input; full lossless output; failing embedded CPython row followed by an Output row. Each must yield an explicit result or refusal, with no fabricated or lost successful token. Exercise the existing native graph and FIFO authority.
2. **Specify one action transaction across the existing graph definition:** session identity, action sequence, admitted inputs, state revision, result/error, and effect commit. Preserve Meowdoku's existing dt-system save/restore, per-session isolation, access policy, and ordering.
3. **Prove one mixed vertical slice:** one Meowdoku session action enters a Nodus graph, uses existing Python state where necessary and an existing compiled pure law where available, then returns an ordered result. Compare move, undo, refusal, restart, and two-session isolation with the current server. Extend typed Python transport only to the values this slice requires.
4. **Measure a real server-GPU product before moving fountain compute.** Nodus has a Vulkan row and device-local FIFO proof, but the browser WGSL product is not automatically that row. Retain product/ABI/device identities, compare transfer plus execution latency and numerical traces, and keep the browser route until the server route is proven.
5. **Stage cutover of ownership:** computation and session state first, then lifecycle/build readiness and external effects. Keep the current HTTP/TLS boundary until the graph has replay, checkpoint, and failure receipts sufficient to replace its responsibilities.
6. **Profile thread cost and Python contention.** `run_parallel` creates and joins worker threads each tick (`src/thread_manager.cpp:1579`); determine its cost at the intended tick rate. Compare embedded CPython under the GIL with a separately hosted process only after the latter has a real protocol and ownership contract.

These are audit recommendations, not approval to change the active Nodus or Turing trees. The corresponding Speaktome `todo` stub records the next engineering entrypoint.

## Prompt History

> we got a good workflow and server setup for a python backed service that can compile and present material. lots and lots of commits. the worker is finishing up their last hour or so. I want you to get brought up to speed while also inventorying what's bespoke, what ai or users have to do compared to what is scripted or can be scripted, the compile pipeline and how modules are decided to be compiled instead of python, and client side then too, and shader or web assembly, even locally nothing stops the server from using a GPU. get caught up, explore site access, we're going to be working on it in a little while with a few more changes

> our shift in focus will turn to nodus and while someone sometimes will chip away at our compiler to one day unlock universally web assembly lowerable code, but we'll be shifting focus to nodus and compiling things into tools there so that the server's duties will be delegated by the advanced threaded kpn

> nodus also can host python or call to hosted python, I believe, and we can exploit that to make the server backbone run totally on nodus, just utilizing pythonic files or processes for things we genuinely can't lower yet, - they'll just be colored different in rows or something that identifies they are running in or using cpython binaries for python

> perform an in depth audit of the state and the next steps from the perspective of a software engineer specializing in systems, graph theory, and chaos

> save please in top level and in speaktome following the speaktome conventions please, in nodus, and in turing, and in meowdoku
