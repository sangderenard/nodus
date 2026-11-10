# Native Studio job-pool qualification, 2026-11-09

Inspected local Nodus 6f3183e49d74d2392d889bfde740bafb72a8f680. This is a source-backed integration plan and bounded scheduler fix, not a completed service cutover.

## Existing authority to use

- `include/graph_runtime.h`: GraphRuntime owns a 1:1 ThreadManager and native plugin-row state. Do not replace this with a Python scheduler.
- `include/nodus_headless_abi.h` and `src/headless/nodus_headless_abi.cpp`: create graph, register plugin rows, configure typed FIFO edges, exchange named-port tokens, and step the actual ThreadManager without a renderer. The header's old process-global dispatch caveat is partly superseded by GraphRuntime ownership; topology still includes Canvas context.
- `include/edge_tensor_fifo.h`: fixed-size, single-writer, multiple independently subscribed readers. Multiple readers broadcast samples; they do NOT implement exclusive competing-consumer work claims. `top_k=0` refuses a full write without advancing readers. Subscribe before publishing; subscribe-at-head skips prior samples.
- `include/edge_tensor_fifo_transaction.h`: existing binary transaction snapshot preserves backing bytes, slot tags, writer binding and reader cursors. Graph-definition/Canvas topology save alone omits live queue state. Snapshot is not automatically a durable disk journal or a transaction across plugin lease state.
- `camera_software/nodus_runtime_bridge.py` in spectral-analyzer wraps the native edge ABI, but using that alone would not satisfy native ThreadManager ownership.

## Bounded change

`src/thread_manager.cpp` now requires acyclic topology for predecessor-count parallel dispatch in both modes. Previously FreeSpinning allowed upstream A -> B <-> C into the parallel loop: A completed, the ready queue emptied, and remaining B/C could never complete. The fix reuses the existing cyclic sequential ordering; it introduces no new scheduler.

Regressions added to `tests/thread_manager_test.cpp` for both modes and `tests/nodus_headless_runtime_test.cpp` for the real headless graph route. CTest timeouts bound hang failures. These tests prove completion only when executed; they do not establish token-ready KPN semantics or job durability.

## Next native slice

Use custom native admission/lease/return/reclaim rows in the existing graph. One authoritative native lease owner assigns work; browsers must not be independent competing FIFO readers. Carry job identity, owner/account reference, scene revision, work range, lease generation, worker/device identity and product identity in typed tokens. Python retains authenticated HTTP ingress and account library presentation, not pool scheduling. Reuse existing continuation/recovery protocol rather than invent new ray state.

A lease return must match its generation and scene revision, accept at most once, and leave timed-out work eligible for reassignment without losing its original retained rays. Checkpoint native FIFO state together with lease-owner state at a real quiescent transaction boundary before claiming crash recovery. Keep queue admission and GPU residency admission distinct; a yielded live GL process still holds VRAM.

## Remaining correctness gates

Current `thread_manager.cpp` ignores typed input/output transfer return values around lines 894 and 1429. Scalar inputs may fabricate zero without an available token. A one-line early return is insufficient: earlier input rows may already have consumed samples, and earlier outputs may already have published. Define and test the module transaction using the existing transaction machinery rather than hiding partial effects. Full-output, empty-input, stale-return, duplicate-return, lease expiry, restart and two-account isolation need real native graph tests.

## Verification

`git diff --check` passed. Current-source build command:
`cmake --build build --config Release --target nodus_headless_runtime_test test_thread_manager --parallel 2`
reached MSBuild and Checking File Globs, then execution was blocked by network policy for `https://dc.services.visualstudio.com:443`. No MSBuild/cmake process remained afterward; existing test executables retained October 23 timestamps. No runtime test pass is claimed and no source changes were deployed. Do not use the old Release binaries as evidence for this patch.

## Prompt history

User: "you could future proof us to the moon if you used nodus fifo for the job storage, be set up and imply that load management is probably going to be nodus handled eventually"
User correction relayed by parent: "it's not optimized for python ... what would matter more would be looking at whether you can run an instance of thread manager from nodus with the fifos from nodus with some custom nodes specifically for our job stealing pool"

## User's two-table topology refinement

Participant exchange table: one registered connection row keeps the participant's capability/credit/lease/check-in state; nonblocking ingress publishes completed bidirectional exchanges to native typed FIFO. ThreadManager schedules exchange processing on its ticks. Never wait on browser network I/O inside the scheduler thread. Offers identify compatible engine/product/scene and bounded capacity; returns identify the precise outstanding loan.

Engine routing table: consumes exchange-result tokens, validates against the authoritative outstanding lease, and routes accepted results to the correct waiting engine job FIFO. Only this authority commits a result; FIFO broadcast readers do not grant exclusive job ownership. The retained work survives a client disconnect or late result. Expired generations can be reoffered; late or duplicate generations cannot complete the replacement lease.

This topology is a mapping onto existing module rows, typed edges and ThreadManager, not an implemented custom node package. Its API must preserve the current ray continuation protocol and account authorization in the HTTP facade.

## Offer-market clarification and concrete ABI

The operator analyzes the table's stack of work contracts that nobody has accepted, then offers compatible contracts to checked-in connections. Competition is market-pressure analysis by that operator, NOT competing FIFO readers. The running server is one loopback connection through this same offer/accept/result path. Do not create a privileged server dequeue lane or silently compute work already on loan.

Actual table mapping:
1. Unaccepted work contracts remain on the native table's retained stack/state until explicit acceptance. The offer operator reads demand and participant capacity; no pricing formula is invented by this audit.
2. Participant-exchange rows carry offers, acceptances and completed returns for browser connections and server loopback. ThreadManager ticks advance the exchanges; network completion arrives through nonblocking ingress FIFOs.
3. Accepted work is retained as a loan. Return-routing rows validate identity and deliver accepted results to the originating engine job; transport submission is not estimator commit.

Reuse `turing/examples/meowdoku/client_ray_relay.py` immutable identity fields: `token`, `revision`, `sample_prefix`, `geometry_hash`, `kernel`, `query_kernel`. Its current engine-side `take_result`/`finish` split distinguishes submitted, validating, committed, rejected and uncertain. Keep that distinction and engine-owned GL continuation. This existing relay is origin-session-only; multi-participant assignment requires explicit account/participant authorization rather than weakening `_auth`.

`include/tool_api.h` documents an important concrete boundary: separately loaded plugin DLLs receive a distinct inline ValueTypeRegistry singleton. A plugin calling register_struct does not register its struct in the host. Use existing host-owned AbstractTensor crossings for contract payloads, or compile custom rows into the same native image as the host. Do not pretend a private plugin type ID is a globally registered queue schema. Native graph `ModuleStackContract::AbstractTensorPointer` and `nodus_headless_abstract_tensor_type_id` already support the shared tensor route. Raw process pointers must not be persisted as durable work references.

Prompt clarification: "the competition will just be straight market pressure analysis from the operator that makes offers based on the table's stack of work contracts that haven't been accepted by anyone yet, one connection will be loopback to the running server"

Qualification correction: added `nodus_headless_runtime_test` to the existing NODUS_TEST_TARGETS assertion-enablement list. Its previous Release configuration could compile away assert-contained graph mutations and checks. Future Release test qualification must use the rebuilt target with assertions active.

## Implemented native mailman qualification (later this session)

`include/contract_negotiator.h` implements the bounded native operator: retained unaccepted contracts, caller-supplied offer selection, one participant visited each market turn, explicit acceptance and fresh loan generation, real Nodus lossless result FIFO, one result delivered and one engine checked each worker turn. Refusing result consumers retain the pending result. Only committed entries can retire. The server joins as another Host callback, not a privileged compute path. Callback exchanges must be nonblocking mailbox operations on the owning graph lane; engine work requests peek and are acknowledged only after admission. No HTTP, currency/pricing formula, ray intersection law, or alternate scheduler is introduced.

`include/contract_negotiator_tool.h` supplies ordinary `ITool` market/worker rows and explicit registration in the host's existing ToolRegistry. Factory-bound context survives ordinary null-user initialization. Multiple pairs use separate Mailman instances/prefixes and independently scheduled graph rows; an independent multiplayer pair can share the canvas without sharing this pool state. Bind the pair as ordered rows or explicit dependent modules. Do not execute their shared state concurrently.

`tests/contract_negotiator_test.cpp` was compiled directly with the installed official MSVC compiler, `src/tool_registry.cpp`, and the existing `nodus_runtime.lib`/DLL, independently of the policy-blocked MSBuild operation. It PASSED against the real native FIFO: tool registration and execution, duplicate registration refusal, one connection per turn, acceptance, consumer refusal/retry, at-most-once result delivery, expiry/new generation, stale return rejection, capacity and committed-only retirement. This verifies the new C++ code with existing FIFO binary, not a rebuilt headless image or production HTTP integration. CTest target added for subsequent full-source qualification.

Limits: callback wiring into the live server/engine and a real headless scheduler test for these new rows remain unimplemented/unverified. No restart durability is claimed for Mailman state. The domain envelope carries opaque engine-owned payload IDs and scene revision; the engine must perform the existing full ray identity/geometry/product validation before its route callback reports commit. Transport acceptance alone never claims numerical correctness. The earlier scheduler build remains blocked; its tests have not run. No deployment occurred.

## Generic socket/contract tool clarification

Final new source names are `include/contract_negotiator.h`, `include/contract_negotiator_tool.h` and `tests/contract_negotiator_test.cpp`. The generic `nodus::contracts::NegotiatorTool` has exchange/routing roles; use separate instances and edges for rendering or multiplayer. The policy selecting unaccepted contracts stays outside generic transport; no economic formula or multiplayer fairness policy is embedded. The nonblocking exchange callback is where the application socket mailbox binds. Searches of Nodus include/src/nodus_graph for socket, WebSocket, Winsock, asio and negotiator filenames found no existing socket negotiation implementation. No actual network socket adapter is claimed by this qualification.

## Real socket and headless qualification

Implemented `include/contract_socket.h`: connected-socket adapter with a bounded, fixed 72-byte v1 little-endian protocol, nonblocking partial-send/receive handling, offer/accept/result/receipt messages and bounded work per poll. Caller supplies an already-authorized connected socket. This tool neither creates a public listener nor replaces authentication/TLS. Offer acceptance now explicitly names its generation and remains outstanding across asynchronous turns; a socket write alone is never acceptance.

`tests/contract_socket_graph_test.cpp` has two configurations. Both passed direct MSVC qualification against existing native libraries:
- native registered tools -> actual loopback TCP -> native FIFO -> producer commit and receipt;
- actual headless ThreadManager graph -> same loopback TCP/FIFO/commit path, exit 0 in an isolated headless working directory.

The graph test compiles current headless ABI source into the same native image as the registry and links the existing static canvas runtime. It does not certify that all current native sources were rebuilt. An initial attempt to connect zero-port tools with a fabricated edge crashed at step; that invalid edge was removed. Final proof uses supported ordered tool rows in one module. Separate tables with meaningful declared typed ports still need their graph binding; do not call this proof a completed two-table graph. Current source scheduler-cycle regression still needs its full-source build.

CTest targets added: `contract_negotiator_test`, `contract_socket_test`, `contract_socket_graph_test`, assertions active in Release, bounded timeouts, isolated graph-test working directory. Negotiator unit test was rerun after asynchronous-offer correction and passed. Socket proof is loopback only; no external network connection or production service change occurred. No restart durability or production engine/client wiring is claimed.

## Correction and completed two-table graph transport

Debugger evidence resolved the graph crash: NegotiatorTool::initialize wrongly treated the generic ToolInitContext.user pointer as a ContractNegotiator. Graph initialization may supply its own host/canvas pointer. The tool now preserves its explicitly factory-bound state; the unit regression initializes with an unrelated host pointer. Earlier speculation about an invalid edge or stale-library cause was not established and must not be repeated as the diagnosis.

Added narrow `nodus_headless_publish_edge_token` / `nodus_headless_consume_edge_token` calls using the existing configured edge and canonical contact reader/writer identities. ContractNegotiator can bind this graph transport before admitting work, replacing its standalone FIFO rather than copying into a parallel queue. Two separate native modules now exchange on a declared lossless graph edge. The real two-module ThreadManager/TCP/FIFO/producer-commit test passed, exit 0. Python remains a required active-development service participant, not something this native connection removes.

Latest verification: expanded actual two-module graph/socket test to two independent producer instances. Worker turns peek each producer's pending contract in round-robin order and acknowledge admission; matching result/revision is delivered exactly once to each producer. Final rebuilt test exited 0. Negotiator unit test also passed after the factory-context regression. These are producer-identity tests, not proof of production account authorization or real Meowdoku moves.

Existing Python service inspection: `turing/examples/meowdoku/server.py` accepts `/api/action`, resolves the existing cookie-owned Table after viewer/edit checks, then calls `Table.action(payload)` under `table.lock`. No batch action endpoint was found. Python callouts must reuse this actual Table/game authority and preserve authorization, not instantiate parallel game state or assume move retries are idempotent. The Python adapter task is coordinated separately with the worker owning that service.

## Current scheduler source and chosen real engine

Current `src/thread_manager.cpp` was compiled directly with `tests/thread_manager_test.cpp` using installed MSVC and existing supporting static libraries. Bounded regression execution exited 0. This qualifies the new partial-cycle scheduling fix itself. Standalone compilation exposed a missing `<functional>` include in `include/graph_runtime.h`; it is now explicitly included. Full CMake/current-dependency rebuild remains unperformed.

Selected real engine: `turing/examples/meowdoku/entity_motion.py:damped_axis`, the existing ship-motion engine shared by Python and the page predictor. The tensor/compiler worker lowered its exact source through the sanctioned compiler, with concordance detector count zero and retained C/SSA/Buffer ABI V2 under `out/entity-motion-native/product-c/`. Native artifact compilation was subsequently denied by network policy for Visual Studio telemetry, so no completed native engine DLL or complete Python/native/browser vertical slice is claimed. Do not retry that denied engine compilation via another route without new authorization.

Python remains a necessary active-development service. Native links complement it; they do not remove the Python engine/API authority. The service worker owns the existing authenticated batch action route and is independently checking Python versus the shipped browser motion.wasm while the native compile is blocked.

### Python service wire qualification (continued)
Added tools/contract_service_peer.py, an exact 72-byte native protocol peer with bounded nonblocking poll and asynchronous service dispatch. It retains result identity until FIFO receipt, rejects replayed generations, and stops on uncertain service failure instead of retrying. Tests/test_contract_service_peer.py passes two tests, including actual entity_motion.damped_axis evaluation, fragmented framing, result retention and no-replay cases.

Important authority boundary: this protocol has Offer/Accept/Result/Receipt, but no server ownership Grant. A worker can compute before its acceptance is adopted; only native result validation plus producer commit decides ownership. Consequently this adapter is safe for pure or staged computation only. It MUST NOT directly execute mutable /api/action/batch requests on an offer. That endpoint requires its existing session, epoch and sequence protection at authorized commit, or a separately specified grant protocol. Opaque engine and payload IDs provide no authorization.

This is not a production service and does not yet join Python, current compiled native engine and browser into one verified path. Existing separate native two-table/FIFO socket proof and WASM/Python parity are real; native damped_axis compilation remains blocked by the denied Visual Studio telemetry network request. No denied build was retried, no live server was changed, and no deployment is claimed.

### Operational native host boundary qualification
Added src/headless/contract_service_host.cpp and tools/nodus_mail_host.py. The DLL creates the actual two native modules, registered negotiator tools and declared lossless edge. Engine callbacks peek/admit contracts and defer, reject/requeue, or commit result refs. Python callbacks are mailbox-only, never GL operations; the owner performs real validation/commit outside graph.step. Rejected pending FIFO results explicitly return their contract to Unaccepted and release the head, preventing one invalid result from blocking all jobs. Uncertain results stay retained, never silently requeued.

Direct independent host compilation against existing qualified supporting libraries succeeded; this was not a retry of the denied engine artifact or MSBuild command. tests/test_contract_service_host.py passed an actual native graph -> connected socket -> Python pure computation -> native FIFO -> rejection -> new-generation offer -> commit -> retire cycle. The host unregisters its factories on teardown. Existing native dependencies are not a complete current-source rebuild.

The compiled local DLL is build/Release/nodus_contract_service.dll. This is an isolated development host with one configured producer and one connected peer per host, no listening port or new authentication. The Studio worker is integrating its retained full-identity lease store and owner-thread commit. Production multi-client host, restart durability, real-ray integration and server savings are NOT established by the bounded host test.

### Completed-return-only double-check qualification
Extended tests/test_contract_service_host.py with a deterministic five-poll validation delay in the engine owner, outside native callbacks. With no client response, native graph turns cause zero checks. Two completed participant results cause exactly two checks (1:1); the first is rejected/reoffered and the second commits once. The native FIFO holds pending results until those owner decisions. Both host scenarios pass. Five polls is a test clock, not a proposed production delay; no live configuration was changed. Server validation work remains real cost and is not reported as eliminated server computation.

### Direct native row benchmark, avoiding an artificial Python call
The existing compiler product out/turing-web-proof-v4 contains both native calculate DLL and generated Nodus Buffer ABI tool wrapper. tests/direct_native_row_benchmark.cpp invokes the actual ITool::execute_stack against Nodus AbstractTensor handles, not a Python delegate. 20,000 calls all returned the expected value 20. Direct kernel entry took 0.000022900s total; native tool row took 0.011240500s total, approximately 0.562 microseconds per call including output allocation/mapping/read/destruction. Python compute calls: 0. This is a tiny scalar calculate law, NOT an intercept benchmark, full graph scheduling measurement, or production speed claim. Existing product wrapper predates current refusal-reporting fix; successful valid-input path only is exercised. Current original-GLSL triangle native product identification remains with compiler worker.


### Callback stall attribution, 2026-11-10
Added native wall spans to contract_service_host.cpp and built a separate nodus_contract_service_diagnostic.dll, leaving the operational DLL unchanged. nodus_mail_timings exports totals for whole graph step, request/route/admitted callbacks, socket exchange and step count. Probe tests/contract_callback_timing_probe.py executes 100 idle graph turns with no actual kernel work, then repeats with a competing Python thread. Receipt: build/contract-callback-timings.json.

Idle: native step 0.0484998 s, request callback 0.0011171 s, socket 0.0004173 s. Competing Python: native step 1.6101415 s, request callback 1.5239233 s (94.6 percent), socket 0.0011761 s, residual 0.0850421 s. Outer Python call wall 3.124921 s additionally includes approximately 1.515 s outside the native timed span. Calling-thread CPU reported zero at its available clock resolution. This isolates waiting around Python callback/execution reacquisition as the major cost under contention, not native compute or socket blocking. Changing only the probe's Python switch interval from 5 ms to 1 ms did not meaningfully change the observed wait, so that exact cadence is not claimed as the cause. Diagnostic DLL used /O2; comparisons are within that same diagnostic binary, not against the prior full-render timing. Actual render attribution requires running the same render with these spans. No global setting, live DLL or server was changed.

### Fair multi-participant parent batch, 2026-11-10
Added serialized nodus_mail_add_participant and NativeMailHost.add_participant. First connected peer remains ID 1; additional IDs are explicit and unique. Outstanding offers reserve their contract from other peers, so peers no longer compute the same unaccepted chunk concurrently. A host has one accepted loan at a time. After expiry, other hosts receive first retry opportunity while the previous host is temporarily ineligible for that same retained chunk. Native market round-robin and result FIFO remain authoritative; Python provides identity/payload callbacks only.

Separate nodus_contract_service_multihost.dll was built without replacing live/previous host libraries. tests/test_contract_multihost.py passes: eight disjoint ranges of one 512-sample parent batch were computed exactly once and distributed four per ready participant. Recovery scenario passes: first host accepts then goes silent, same contract/payload reoffers to second host with new generation, commits once, and late first-host result cannot commit. Single-host rejection/deferred-validation tests also pass against the multi-host DLL. These are scheduling/identity fixtures; full Studio parent assembly remains in the ray worker. Latest user policy trusts authorized structurally valid participants in normal mode, with no duplicate GPU verification; earlier 1:1 verification fixtures remain optional diagnostic evidence, not the current normal-mode policy.
