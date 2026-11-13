# Nodus bridging: contract ports (2026-11-10)

Source of record: nodus commits d16e395, 01f7d65, ddd78c5 (all 2026-11-10, branch tip ddd78c5). Measured claims below are only those the commit messages state, from build-bridging (Release). Proposal this implements: `C:/dev/Powershell/NODUS_BRIDGING_PROPOSAL_2026-11-10.md` (steps 2 and 5a, decisions A and D).

## What a contract port is

A canvas-owned transport for a contract negotiator pair:

- the negotiator pair (`<prefix>.market` / `<prefix>.worker` plugin rows) is process-owned, acquired by registry prefix (`include/contract_negotiator_registry.h`, `src/contract_negotiator_registry.cpp`); acquire is idempotent, release unregisters the pair;
- the canvas owns the result edge between the rows; that edge is bound as the negotiator's result transport;
- the canvas owns a 127.0.0.1 listener (`include/contract_listener.h`) that reads one Hello per connection.

`nodus_headless_step` polls the listener before the tick and applies engine rejections after it. Hello role 1 binds the connection as a host peer, role 2 as an engine producer.

## Wire protocol (`include/contract_socket.h`)

Fixed 72-byte little-endian frame. Bytes 0-3 magic 0x31434E4E, bytes 4-7 kind (decode accepts 1-9), then eight uint64 at offsets 8, 16, ..., 64: work.id, work.engine, work.revision, work.payload, loan.generation, loan.host, loan.deadline, result.payload.

| kind | name | meaning |
|---|---|---|
| 1 | Offer | host side, offered loan |
| 2 | Accept | host side, accepted generation |
| 3 | Result | result |
| 4 | Receipt | generation + payload 1/0 accepted |
| 5 | Hello | work.id = role (1 host, 2 engine), work.engine = participant id, work.revision = 2 (version) |
| 6 | Submit | engine to nodus, work = Contract |
| 7 | Admitted | nodus to engine, work.id = contract id |
| 8 | Route | nodus to engine, full Result |
| 9 | Decision | engine to nodus: work.id, loan.generation, payload 1 commit / 0 hold / 2 reject |

`WireSocket` is the shared nonblocking bounded-queue base; `ContractSocket` is the host side, `EngineSocket` the producer adapter. A peer exceeding capacity of unadmitted Submits is a protocol violation and the socket closes. Python peer: `tools/contract_service_peer.py` (Hello, connect_loopback, EnginePeer).

## Persistence: trailer records

`nodus_headless_save_file` appends records after the CANVAS V4 body (`src/headless/nodus_headless_abi.cpp`, writer ~565-612, reader ~470-510). Exact spellings:

- `EDGE_FIFO <a_module> <a_contact_idx> <b_module> <b_contact_idx> <fifo config fields>`
- `PORT <in|out> <module_idx> <row_idx> <start_at_head> <declaration_order> <fifo config fields> <name>`
- `CONTRACT_PORT <prefix> <capacity> <a_module> <a_contact_idx> <b_module> <b_contact_idx> <listen_port>`

`nodus_headless_load_file` restores them: re-attach and configure edges, re-declare ports in declaration order, re-declare contract ports last (they need their result edge). The canvas loader skips unknown tags, so older files and the GUI's `gp_canvas_load_from_file` are unaffected; the GUI restores topology only. Save refuses a port name containing a newline and a contract prefix containing whitespace.

## Refusal rule for unregistered plugin rows (Decision A)

`gp_canvas_set_module_rows` refuses a plugin row whose id has no registered factory before applying anything (`include/inl/canvas_abi_rope_map.inl`). Programmatic adoption only; file-load paths keep their name fallback.

## Abstract-tensor vocabulary

The static registrar is dropped when `canvas_tables_static` is linked, so `abstract_tensor.*` rows had no instance in the headless image. The vocabulary is now registered explicitly in `nodus_headless_create`.

## ABI and Python API added (`include/nodus_headless_abi.h`, `nodus_graph/native.py`)

- `nodus_headless_acquire_contract_pair(prefix, capacity)`, `nodus_headless_release_contract_pair(prefix)`
- `nodus_headless_declare_contract_port(g, prefix, capacity, edge_id, listen_port)` returns the bound port (>0) or 0 (empty/whitespace prefix, capacity <= 0, edge not attached/configured, duplicate prefix or edge, bind or listen failure); listen_port 0 = ephemeral. The edge must be configured with token_bytes == sizeof(Result), type 0x534A5231, layout 2, dtype 8, top_k 0.
- `nodus_headless_get_contract_port`, `nodus_headless_get_contract_port_count`, `nodus_headless_get_contract_port_at`
- Python: `Graph.acquire_contract_pair`, `Graph.declare_contract_port`, `Graph.contract_port`, `Graph.contract_ports`.

Release only after every graph that bound the pair has been destroyed (a negotiator holding contracts refuses a new result transport). A peer whose socket closed is removed from the negotiator at the next `pre_step` (see Removal semantics), so its id can connect again; a second LIVE connection with a still-connected id is refused as a duplicate.

## Pre-existing build repairs (d16e395)

- `nodus_runtime` gets the `src` include dir (kernel_isa.h); the Vulkan backend branch is guarded by `NODUS_HAVE_VULKAN`, so the lean runtime links again.
- ctest PATH property: the Torch lib is prepended to PATH for the contract tests (CMakeLists.txt `ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:...;${TORCH_INSTALL_PREFIX}/lib"`); the missing-DLL dialog had shown as ctest timeouts. `nodus_headless` links `nodus_runtime` and ws2_32 (01f7d65).

## Measured (per commit messages; build-bridging, Release)

- d16e395: contract_negotiator_test, contract_engine_socket_test, contract_socket_graph_test, thread_manager_test, artifact_buffer_import_test pass; `python tests/test_contract_service_peer.py` 11 passed.
- 01f7d65: nodus_headless_runtime_test (unregistered-row refusal, ports/FIFO save-load round trip, contract port loopback + save/destroy/load + reconnect), nodus_python_graph_test (contract port round trip via contract_service_peer.py), nodus_headless_bridge_smoke, contract_socket_graph_test, thread_manager_test: all pass.
- ddd78c5: the unregistered-row refusal case is now run in tests/nodus_python_graph_test.py (one-line change).

## NOT done

- GUI (`frontend_shell`) does not yet host contract ports; work in progress. It restores topology only.
- Autosave does not rewrite the trailer.
- The Spectral/Meowdoku server still uses the Python-owned `NativeMailHost` route.
- The graph-forward Turing lane (`compile_graph_forward_to_wasm` / `BrowserGraphForward`) has no API contract; see the Turing hazards entry `docs/hazards/20261110T155027Z_bridging-web-lane-api.md` (worktree C:/Temp/wtb-api, branch bridging/web-lane-api, commits 0d267080, cc1b307e).
- No restart durability of negotiator state (process-owned; save/load re-declares the port, not contracts).
- No authentication beyond loopback binding.
- Program-rows and entity concurrency (proposal steps 4 and 5) not started.

## Status update, later 2026-11-10

- GUI hosting is proven at runtime: `frontend_shell --workspace=<copy> --acquire=gui.proof:4` printed `contract_port gui.proof 127.0.0.1:47611` 2.33 s after launch; an engine and a host peer completed Submit -> Admitted -> Offer/Accept/Result -> Route -> Decision -> Receipt through the window's tick loop. Receipt and the two scripts: docs/receipts/GUI_CONTRACT_PORT_PROOF_2026-11-10.md. Not proven: autosave behaviour after a GUI edit (no autosave write occurred), rendering, clean shutdown.
- Canvas-host process: tools/run_canvas.py (tests/test_run_canvas.py).
- Lowering library reader: nodus_graph/lowering_library.py over the Meowdoku server's accounts.db tables (written on the Turing side, branch bridging/web-lane-api).

## Turing compile host (one job at a time)

`include/turing_compile_host_tool.h` (`nodus::contracts::TuringCompileHost`) is the place Turing compile requests now go. It is a negotiator HOST Peer: `negotiator.add_host(host.peer())`. It needs no row of its own; the existing `<prefix>.market` row calls every host once per market turn and each call is one non-blocking `exchange()`.

- Requests: an engine submits a contract whose `id` names the work. The request JSON is `<jobs_dir>/<id>.request.json` (same schema as `tools/turing_row_tool_compiler.py --request`); the product directory is `<jobs_dir>/<id>/`. The contract `payload` is carried untouched and not interpreted.
- One in flight: the host owns at most one `std::thread`, which runs only `gp_turing_row_tool_compile_only` (validation, driver, receipt check; touches no registry/canvas). While it runs, `exchange()` returns an empty Exchange (no offer accepted). Only after a result is delivered does it take another offer. `stats()` exposes accepted/completed/refused/in_flight (0 or 1).
- Lane-side adoption: when the job is finished, the next `exchange()` (on the market lane) joins the thread, runs `gp_turing_product_adopt_and_load` if `cfg.adopt`, writes `<product_dir>/compile-host.result.json` (status, tool id, queued/started/finished epoch seconds, wall seconds, adoption report) and returns `Result.payload = status`.
- Status codes (`CompileStatus`): 1 Adopted, 2 CompiledNotAdopted, 3 ExistingProduct, 4 EmittedNotBuilt (source-only, never adopted), 16 RequestMissing, 17 CompileFailed, 18 AdoptFailed, 19 Cancelled.
- Restart: if `<product_dir>/nodus-row-tool-product.receipt` already exists the compiler is not run again; the contract completes with ExistingProduct (adopted on the lane when `cfg.adopt`). Missing request file: refused immediately with RequestMissing.
- Cancellation: `shutdown()` / destructor sets the `GP_TuringCompileCancel` handle; the driver wait (polling 20 ms) kills the child (Windows: kill-on-close job object, so grandchildren die too) and joins. Never detached. Other callers pass nullptr and keep the INFINITE wait.
- Loan deadlines count market turns: set the negotiator's `loan_ticks` large enough to cover a compile, or a late result is dropped by the negotiator.
- Test: `tests/turing_compile_host_test.cpp` (stub driver; ordering, one-in-flight, turn latency, restart refusal, cancel).

Not done: routing of who may submit; the request upload path from the Python server (nothing yet writes `<id>.request.json`); the live canvas does not yet host the compile host.

## Compile host on a live canvas

A headless canvas can now attach the Turing compile host (above) to a negotiator prefix it has acquired.

- ABI (`include/nodus_headless_abi.h`): `nodus_headless_attach_compile_host(prefix, host_id, python_executable, nodus_root, turing_root, module_library_root, jobs_dir, adopt)` -> 1/0; `nodus_headless_detach_compile_host(prefix, host_id)` (cancels a running job, joins, destroys the host); `nodus_headless_compile_host_stats(prefix, host_id, uint64_t out5[5])` = accepted, completed, refused, acknowledged, in_flight. Refused (0, the ABI reports by return value only): no acquired pair, host_id 0, id already attached to the prefix, any empty/NULL argument. `nodus_headless_release_contract_pair` detaches every host of the prefix first. Python: `Graph.attach_compile_host / detach_compile_host / compile_host_stats` (RuntimeError on refusal).
- `tools/run_canvas.py`: `--compile-host PREFIX:HOST_ID:JOBS_DIR` (repeatable; PREFIX must also be in `--acquire`, refused before start otherwise), `--compile-python` (default: running interpreter), `--turing-root` (default: `../turing` beside this repo), `--module-library-root` (default: `<canvas>.modules`), `--compile-adopt` (adoption is off by default). Hosts are detached on shutdown.
- File convention: request `<JOBS_DIR>/<contract id>.request.json` (placed by the submitter before the contract is submitted), product dir `<JOBS_DIR>/<id>/`, outcome `<JOBS_DIR>/<id>/compile-host.result.json`; `Result.payload` is the `CompileStatus`. Market lane steps the host; only the compile wait runs on its one thread.
- Detach: after cancelling and joining, the host is removed from the negotiator (`remove_host`); no inert wrapper remains, a second live host on the prefix is offered the next contract at once, and the same id can be attached again. A loan the detached host held is re-offered (see below).
- Test: `tests/turing_compile_host_abi_test.cpp` (`turing_compile_host_abi_test`; refusals, RequestMissing, ExistingProduct with a nonexistent python so no compiler can run, idle detach, detach then a second live host served within 10 turns, same id re-attached, release-after-attach). No Python-level test was added (it needs a rebuilt `nodus_headless` DLL).
- NOT done: `build-service` has not been relinked, so the live canvas host does not have these ABI functions or flags yet. No real Turing compile has been run through an attached host.

## Removal semantics (`include/contract_negotiator.h`)

`ContractNegotiator::remove_engine(id)` / `remove_host(id)`; both return false for an unknown id. Peers are held by INDEX inside the negotiator (host/offer/engine vectors, round-robin cursors, references held across the exchange/route/request callbacks) while contracts and loans name peers by ID. Removal therefore never mutates the vectors mid-turn: from inside `market_turn()`/`worker_turn()` it is deferred to the end of that turn; between turns it applies at once.

- Engine removed: contracts not in flight (Unaccepted, Committed) are dropped. A Loaned contract stays until its host answers; the host completes normally, the result has no engine and is discarded (`discarded_results()`), and a loan that expires first is dropped instead of re-offered. A Returned contract is discarded the same way when its result is consumed. The id can be added again.
- Host removed: its retained offer goes with it; each Loaned contract it held returns to Unaccepted with the loan cleared and is re-offered to another host under a NEW generation (`reoffered_loans()`), so a late result of the old generation cannot match. A Returned contract (host result already accepted and published) is left alone and routes to its engine.
- `contract_port_canvas.cpp`: at the start of each `pre_step`, engine and host sockets found closed are removed from the negotiator and from the binding, then the listener is polled (so a reconnect with the same id in the same step succeeds).
- Tests: `contract_negotiator_test` (all removal cases, deferral, loan length), `nodus_headless_runtime_test` (same engine id reconnects through a real loopback port and gets a second contract routed; duplicate live id refused), `turing_compile_host_abi_test`.

## Loan length (`loan_ticks`)

A loan lives `loan_ticks` market turns (default 60, one second at 60 turns/s). Work held longer is re-offered under a new generation, and a late result of the old generation is dropped.

RULE: a prefix that carries a compile host needs `loan_ticks >= market rate x the longest compile you accept`. The default made a real 8 s compile re-loan up to generation 10 and finish as ExistingProduct.

- Chosen where the pair is acquired: registry `acquire(prefix, capacity, loan_ticks = 0)` (0 = unspecified: 60 for a new pair, an existing pair as it is); `nodus_headless_acquire_contract_pair_with_loan(prefix, capacity, loan_ticks)` (the old function keeps 60); `nodus_headless_contract_pair_loan_ticks(prefix)`; Python `Graph.acquire_contract_pair(prefix, capacity, loan_ticks=None)`.
- Acquiring an existing prefix with a DIFFERENT explicit loan length is refused ("already acquired with loan_ticks=N, refusing loan_ticks=M"); the same length is the ordinary shared acquire.
- `tools/run_canvas.py --acquire PREFIX:CAPACITY[:LOAN_TICKS]` (two-field form unchanged). Example, ten minutes of compile at 60 turns/s: `--acquire turing:8:36000 --compile-host turing:7:C:/jobs`.
- Persisted: the CONTRACT_PORT record gets an optional trailing field `... <listen_port> [<loan_ticks>]`, written only when the length is not 60. Reason: a canvas file that loads onto a fresh process must come back with the loan length its compile host needs. Old files (no field) load with the default; loading a record whose length differs from an already-acquired pair is refused. GUI `frontend_shell --acquire=PREFIX:CAPACITY` has no loan option yet (default 60).
- Test: `contract_negotiator_test` (a host holding a loan 100 turns completes at generation 1 with loan_ticks 600; with the default 60 the same scenario re-loans and never completes) and `nodus_headless_runtime_test` (refusal rules, trailer round trip).

## Result edge full after `slots` results (found 2026-11-10, live canvas host)

Symptom: with loan_ticks 54000, engines connecting one contract per connection under the same id: the first contracts were served, later ones were admitted but never got a Result; the compile host nevertheless finished them (product directories appeared, `completed` rose, `in_flight` 0).

Reproduction: `turing_compile_host_abi_test` (`loopback_missing_requests`): real loopback port, pair loan_ticks 54000, compile host attached, six contracts each on its own engine connection with the same id, requests missing. Before the fix contracts 1-4 routed, contract 5 stalled: negotiator entry Loaned (gen 5, deadline 54019), `discarded=4`, host `accepted=5 completed=5 refused=5 acknowledged=5 in_flight=0`. Observed inside the stall: `ContractNegotiator::accept_result` reached `publish_` and the canvas result edge refused the push (`EdgeTensorFifo::ensure_space_for_write`, edge_tensor_fifo.h:813-821, lossless, top_k 0): seq=4, min_seq=0, slots=4, readers = {0x100000000 (worker row, seq 4), 0x8000000000000000 (seq 0)}.

Cause: the second reader. `gp_canvas_clear_meta_and_overlays` (called by `nodus_headless_create`) destroys every module, including the synthetic root-reflection module, via `canvas_clear_workspace`, but left `root_module_idx` at 0. The first module the caller adds is then module 0, which `thread_manager.cpp:505-516` treats as the root: it subscribes the root reader key (the all-zero contact key, fallback 0x8000...) to every edge touching that module (the market row's result edge) and never drains it. A lossless edge therefore holds every token forever and refuses the publish after `slots` results. The negotiator then returns false from `accept_result`, the host's receipt reports not accepted, and the contract stays Loaned until the loan deadline. Not the compile host (it completed and was acknowledged), not remove_engine, not capacity.

Fix: `include/inl/canvas_abi_overlay_raster.inl`, `gp_canvas_clear_meta_and_overlays` sets `root_module_idx = -1` right after `canvas_clear_workspace`. After the fix the six-contract run routes every result 4 canvas steps after Submit, and the abandoned-engine variant (engines closing after Admitted, before any result) serves the next same-id engine in 4 steps with the negotiator empty and the host idle.

Step bound (asserted as 6): step 1 listener reads Hello, worker admits the Submit; step 2 market offers, host accepts and (missing request) finishes at once; step 3 market takes the finished Result and publishes it; step 4 worker consumes and sends Route. Two spare steps for socket timing.

RULES: (1) a result edge must have exactly one reader, the worker row; (2) a canvas that destroys its modules must forget the root module index; (3) when a host's result is refused by the edge, the contract waits for the loan deadline, so an edge with fewer slots than the negotiator capacity is a configuration error. NOT done: `canvas_clear_workspace` in the file-load path (`gp_canvas_load_from_file`) still leaves `root_module_idx` as it was; a loaded GUI canvas whose saved module 0 is the ROOT module keeps working by that coincidence and was deliberately not touched. No check yet that edge slots >= negotiator capacity at `declare`.


## Stale root module index on the LOAD path (2026-11-10, later)

Observed (test `loopback_loaded_canvas` in `tests/turing_compile_host_abi_test.cpp`, the live order: create, acquire pair capacity 4 / loan 54000, `nodus_headless_load_file` of a bare copy of a saved messenger canvas, attach compile host, eight same-id engine connections) with only the create-path fix: contracts 1-4 routed in 4 steps, contract 5 never routed. After the load `gp_canvas_get_root_module_idx()` was 0, i.e. module 0 = `market`. At the refused publish (`EdgeTensorFifo::ensure_space_for_write`): seq=4, min_seq=0, slots=4, readers = {0x100000000 (worker row) seq 4, 0x8000000000000000 (root key) seq 0}. The second reader is subscribed by `thread_manager.cpp:505-516` (root-table FIFO block) because `root_mod = gp_canvas_get_root_module_idx()` named market; nothing ever drains it.

Cause: the create-path fix reset the index in `gp_canvas_clear_meta_and_overlays`; `gp_canvas_load_from_file` also destroys all modules through `canvas_clear_workspace` and left the index as it was. Before the load the index was already -1 only because create had cleared it; the fix was in the wrong place for the owner of the invariant.

RULE (one place): `canvas_clear_workspace` (rope_map.inl) sets `root_module_idx = -1`; it is the function that destroys the modules, and create and load both go through it (the line in `gp_canvas_clear_meta_and_overlays` was removed). A GUI canvas that loads a file whose module at its previous root index is labelled "ROOT" re-adopts that index (end of the module loop in `gp_canvas_load_from_file`); the index never names an ordinary module. Headless graphs have no root module and therefore no root reader.

Validation added: `CanvasContractPorts::declare` (used by `nodus_headless_declare_contract_port`, the `CONTRACT_PORT` restore in `nodus_headless_load_file`, and the GUI `load_trailer`) refuses a result edge with fewer slots than the negotiator capacity ("result edge has N slots, fewer than the negotiator capacity M"). Reason: up to `capacity` contracts can be Returned at once, each holding one result until the worker consumes it; a lossless edge with fewer slots refuses the publish and the contract waits for its loan deadline. Test: `declare_refuses_small_edge`.

Not reproduced: the live host's exact 1-in-4 pattern with the current code. With the fix, the load-path test and `tools/run_canvas.py` (live DLL and build-bridging DLL, committing client) route every contract. Separate nondeterministic observation, not investigated: a client that closes its engine socket right after sending the Decision sometimes leaves that contract Returned (Decision not seen); the next connection with the same id is then re-sent that Route first, and a client that holds or ignores a Route for a contract it did not submit gets no result for its own. Check the live client's behaviour on stale Routes before blaming the edge.


## Close right after Decision, same-id reconnect (2026-11-10, later still)

Observed (`tests/turing_compile_host_abi_test.cpp`, engine() sends Hello+Submit and lets 20 ms pass so they sit in the server's buffers together with the previous engine's Decision+FIN): the next engine received the PREVIOUS contract's Route and the test's id check failed. Both suspects were real. (i) `CanvasContractPorts::pre_step` polled a closed engine and removed it without processing the Decision it had already received (decisions are only read when the negotiator routes), so the contract was never committed; before the fix each contract ended as "discarded" instead of committed. (ii) `ContractNegotiator::worker_turn` kept the pending Returned result and routed it to whichever engine currently had the same id, so a new registration received the old one's Route (a client that holds a Route it did not submit never gets its own result: the live 1-in-4 pattern).

Rule (`include/contract_negotiator.h`, removal semantics comment): a contract belongs to the registration (connection) that submitted it. `add_engine` stamps each engine with a fresh `registration`, `Entry.registration` records the submitter, routing and the unaccepted-work purge require the registration to match, so a new registration can never receive an old one's routes. A Returned contract whose engine is removed is retired with `discarded_results()` (at once if it is the result being routed, else at consume time). `settle_engine(id)` lets a closing peer's already-received Decision commit before `remove_engine`; `pre_step` (`src/contract_port_canvas.cpp`) calls it for every closed engine first. Tests: `contract_negotiator_test` (registration block), `loopback_*` in the abi test now assert discarded_results()==0 over 8 contracts.

Open: Committed entries are only retired by the owner (`retire`) or by removal of their engine; the canvas port never retires, so a long-lived engine connection fills capacity with Committed entries.

## ROOT_MODULE record

`gp_canvas_save_to_file` writes `ROOT_MODULE <module index>` after the NODE records, only when a root-reflection module exists (a headless graph has none, so its files have no such line); `gp_canvas_load_from_file` adopts that index. Files without the record go through `canvas_adopt_legacy_root_module_by_label` (legacy: module at the canvas's previous root index labelled "ROOT"; a headless graph's previous index is -1, so it adopts nothing). Test: `root_module_record_paths` (new file, pre-record file, headless-saved file loads with -1).

## A second contract port on a canvas (2026-11-11)

Measured: a canvas host serves only the contract ports its canvas file declares. `run_canvas.py --acquire PREFIX:CAP`
before the load creates the process-owned negotiator and its two plugin rows' behaviour, not a port; the load restores
the file's `CONTRACT_PORT` records and nothing else. Acquiring `studio.product` beside a canvas that declared only
`gui.proof` printed one port.

`tools/add_contract_port.py --canvas IN --out OUT --prefix P --capacity N --port PORT [--loan TURNS] [--keep
PREFIX:CAP[:LOAN] ...]` adds one: it loads the canvas, adds a `<prefix> market` and a `<prefix> worker` module with
the plugin rows, joins them with a result edge of the required shape, declares the port and saves to a new file. The
kept pairs' ports are bound while it runs, so the host that owns them must be stopped (or the input must be a copy
with other ports). Verified on a copy: both ports listed by `run_canvas.py`, and a Python host peer plus an engine
peer completed a contract on the new one (turing `tests/test_meowdoku_studio_pictures.py`, opt-in test).
