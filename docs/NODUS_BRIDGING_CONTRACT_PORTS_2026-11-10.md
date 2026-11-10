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

Release only after every graph that bound the pair has been destroyed (a negotiator holding contracts refuses a new result transport). A closed peer stays registered; its exchange returns an empty Exchange.

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
- Status codes (`CompileStatus`): 1 Adopted, 2 CompiledNotAdopted, 3 ExistingProduct, 16 RequestMissing, 17 CompileFailed, 18 AdoptFailed, 19 Cancelled.
- Restart: if `<product_dir>/nodus-row-tool-product.receipt` already exists the compiler is not run again; the contract completes with ExistingProduct (adopted on the lane when `cfg.adopt`). Missing request file: refused immediately with RequestMissing.
- Cancellation: `shutdown()` / destructor sets the `GP_TuringCompileCancel` handle; the driver wait (polling 20 ms) kills the child (Windows: kill-on-close job object, so grandchildren die too) and joins. Never detached. Other callers pass nullptr and keep the INFINITE wait.
- Loan deadlines count market turns: set the negotiator's `loan_ticks` large enough to cover a compile, or a late result is dropped by the negotiator.
- Test: `tests/turing_compile_host_test.cpp` (stub driver; ordering, one-in-flight, turn latency, restart refusal, cancel).

Not done: routing of who may submit; the request upload path from the Python server (nothing yet writes `<id>.request.json`); the live canvas does not yet host the compile host.
