# Mesh canvas: canvas ports, "up", per-node write, private canvases (2026-11-21)

Ticket `e12fd0f545c44f868591bbe7ef505cc1` (follows `e2f19532`, regional state broadcast).

## The owner's words

- "all nodes are the same"
- "the edge needs to be to the parent canvas of the mesh so that needs working out in the canvas it would be going to a
  canvas port in the top tools area then you'd need an up option to the mesh"
- "they all run the mesh but only individual machines have their write permission and we should allow some privacy from
  read for systems that have sensitive canvases"
- "just let the canvas always optionally build upper layers as used for networking, and let tables hold canvas rows"

## What exists (read 2026-11-21)

- Every canvas has a ROOT reflection module (`canvas_ensure_root_module`, canvas_abi_rope_map.inl): the canvas's own
  boundary, mirroring the root table.
- Toolbar LEDs (`ToolbarLedBox`, canvas_abi_context.inl) were meant to anchor ropes on that ROOT module; the connect
  mode is stubbed: a click prints "connect mode disabled" (canvas_abi_create.inl ~426-453).
- A row's own port is contact `GP_CANVAS_ROW_PORT_CONTACT_BASE + row x GP_CANVAS_ROW_PORT_STRIDE + port` and publishes
  only onto a canvas edge drawn from it (thread_manager.cpp `row_port_publish`).
- Edges refuse a second writer (canvas_abi_actions_edges.inl); readers subscribe independently; `top_k = 0` refuses a
  full write instead of dropping.
- A headless graph is a process singleton (`gp_canvas_get_singleton`): one canvas per process.
- `module_library_actualizer.cpp` turns one table's rows into one row with exposed argument/return ports; it does not
  nest canvases.
- The Nodus link (`<id>.control`, `<id>.bulk`) joins two canvases, in-process loopback, over TCP, wss or UDP.

## Design

1. **Canvas port.** A named contact on the canvas's ROOT module, drawn as a toolbar LED in the top tools area, saved in
   the canvas file. A row-port edge (the link bulk row's "received") or an ordinary Output row ends there. It replaces
   the interim host-side table and drain in tools/run_canvas.py (link_receive), which lost data twice: null pads filled a
   host port, and bulk-row refusals were dropped while the host counted the push as sent (63 of 271 records, live run).
   Inside the canvas a refused tensor stays on its edge.
2. **Up, into upper layers.** A canvas optionally builds UPPER LAYERS when it is used for networking; the mesh is an
   upper layer of the same canvas, not a second canvas or process (a canvas is one per process). A canvas port marked
   *up* carries its edge from layer 0's top tools area into the layer above.
   **Canvas rows.** A table can hold a CANVAS ROW: a row that is a whole canvas, whose canvas ports (and up ports) are the
   row's ports. In the mesh layer each node is a table holding that node's canvas row.
3. **Mesh.** Every machine runs the same mesh canvas: one table per node. A node writes only its own table (single
   writer per table, the edge rule lifted to tables); every other machine holds a read replica of it, fed over the link.
4. **Read privacy.** A canvas may name its readers. Its up ports and contents reach only them, sealed to their keys
   (state_broadcast's recipient model: X25519-wrapped content key, signed manifest). Others see that the node exists.
5. **State broadcast on this shape.** A generation is one node's up stream; its readers are the enrolled regions; the
   pack/accept code is unchanged, and the reader rows live in the mesh canvas, the same on every node.

## Stages

1. Canvas ports on the ROOT module, declared without rendering: today the toolbar LEDs become ROOT contacts
   (`toolbar_base + subgroup_idx`, canvas_abi_overlay_raster.inl ~1854) only while the canvas renders, so a headless
   canvas has none, and nothing reads an edge that ends there. Declare them by name, save them, give them a reader.
2. Upper layers: the canvas builds a layer above on first use of an up port; edges cross layers only through up ports.
3. Canvas rows: a table row that holds a canvas and exposes its ports; in the mesh layer one per node (own node writable,
   remote nodes read replicas fed over the link); single writer per table.
4. Read privacy: per-canvas reader lists; sealed delivery; the non-readers' view.

## Findings for stage 1 (read-only map, 2026-11-21)

- **Contact spaces per module** (all canvas edges are FIFOs in the root table, keyed `(module<<32)|(contact<<16)`, so a
  contact must stay below 65536): Input/Output attachments from 0; module frame LEDs 10000-10063
  (`kModuleFrameContactBase`, 16 x 4); toolbar 10064-10079 (render time only, ROOT only); row ports 20000 + row x 64;
  ROOT contacts 0/1 are the overlay anchors. A canvas-port range fits between 10080 and 20000.
- **A headless canvas has no ROOT module**: headless setup clears the workspace and sets `root_module_idx = -1`
  (rope_map.inl 856-862, nodus_headless_abi.cpp 289-295); save writes `ROOT_MODULE` only when one exists.
- **The scheduler swallows every edge touching ROOT**: `run_scheduled_tick` (thread_manager.cpp 1701-1840) subscribes
  and consumes any edge with `a_module` or `b_module` == root and treats it as a toolbar colour sample; non-toolbar
  contacts are consumed and dropped, and its reader shares the composed key a host reader would use. Port edges need that
  block narrowed to the toolbar range.
- **Root-table IO keys cannot be ports**: unnamed type hints, not saved, never read by the scheduler.
- **Second canvas contexts** (upper layers, canvas rows) build and get their own ThreadManager, but their ticks read and
  write the primary through `g_canvas_context_singleton`, `gp_canvas_get_root_module_idx()` (no ctx),
  `ThreadManager::global()`, the primary's `module_binding_id_map`, input/stack vectors, rope map and overlays, and the
  global sim tick. Each touchpoint needs a ctx before stage 2/3 nesting is safe.
- **Stepping**: plugin rows run through `execute_stack`; `ITool::tick` is never called by the scheduler. A row can step a
  child synchronously (separate manager thread), once the singleton touchpoints are per ctx.
- **Save/load**: CANVAS V4 text; the loader skips unknown tags; a new core record goes beside `ROOT_MODULE`, restored
  after the edge rebuild, and must enter `gp_canvas_compute_signature`. Headless trailers (`PORT`, `EDGE_FIFO`, ...)
  are parsed strictly, so a new tag must not reuse their names. A child canvas needs its own `module_library_root`.

## Stage 3: the canvas row (2026-11-21)

Owner, 2026-11-21: "we need the canvas in a table and going up from the canvas into a table"; "compartmentalize the
canvas in a table, don't use mandated global namespace just nest it"; "eliminate the use of python for nodus unless it's
for an add on development work we haven't implemented yet".

- **A canvas is a compartment.** Every tick request names its canvas (`ThreadManager::TickRequest::canvas`); the
  scheduler's contact keys, module input state, stack snapshots, frame pointers and ROOT lookup read it from there.
  Every table records the canvas that owns it (`GP_TableContext::owner_canvas`, stamped as a canvas creates or is given
  a table); its edge ids, scheduler mirror and sim frame count come from that canvas. Each canvas keeps its own sim
  frame count; only the primary also advances the process-wide pair for tables no canvas owns. Only the primary canvas
  loads or autosaves the workspace file. `nodus_headless_create` no longer refuses a second graph.
- **Rows keep their settings.** `ITool::serialize` / `deserialize` existed but no canvas called them. A canvas file now
  carries `ROW_STATE <module> <row> <hex>` for every row whose serialize answers, handed back on load once the canvas
  exists; `nodus_headless_get_row_state` / `set_row_state` read and set the same bytes.
- **canvas_row** (`module_library/source/tools/canvas_row.cpp`, built as its own plugin like every row): its state is a
  whole canvas, so the table holds the canvas. The canvas's IN ports are the row's Argument ports and its OUT ports the
  row's Return ports, in declaration order. Each execution moves waiting tokens in, steps the canvas once on its own
  scheduler, and moves arrivals out while the edge takes them; a refused token is held by the row, never dropped.
- **Going up.** A canvas's up ports reach the layer above as the ports of the row that holds it: the layer above is a
  table, and the link's ends belong on that table's edges, not in a host.
- **Proof.** `tests/canvas_row_test.cpp` (native, Oracle A1, 2026-11-21): two canvases in one process; 12 of 12 values
  through the nested canvas in order; the parent saved with its ROW_STATE and loaded into a third graph carries 12 of 12
  again.
- **Python removed from this lane.** The link ends that lived in `tools/run_canvas.py`, the canvas-port wrappers in
  `nodus_graph/native.py` and their Python tests are gone; the link's ends move into tables (next).
- **Still process-wide, by design or not yet:** the tool registry (a catalogue), `ThreadManager::global()` for tables
  no canvas owns, the meta-channel readers (`nodus_headless_meta_channel_*` read the primary canvas), and a nested
  canvas's module library root (unset: serialized tables fall back to the cwd default). The canvas signature does not
  include ROW_STATE yet.
- **The tick goes down.** Owner, 2026-11-21: "the top level gives a tick to a table gives a tick to a row so the
  contained canvas receives a tick and can respect it or free run". A row's execution carries its tick
  (`ToolStackContext::dt`, `tick_id`, from the tick request); the canvas row gives that tick to the canvas it holds. The
  contained canvas decides: a table steps on the tick, or free-runs on its own worker where its exec mode is Free (3),
  a choice saved with that canvas (`MODULE_POLICY`). `canvas_row_test` passes both ways, before and after reload.
- **The link's rows, one duty each.** Owner, 2026-11-21: "rows should not be doing double duty, if you want double duty
  bake a plugin from a table with two rows"; and "at no point did I ever say arrival should be redirected by a port
  instead of being fed into the stack". The routing of arrivals to a row port named "received" was an agent's design
  (e053ef6, 2026-11-19), recorded there as the owner's. A link now has four rows: `<id>.control.send`,
  `<id>.control.receive`, `<id>.bulk.send`, `<id>.bulk.receive`. A send row pops a tensor off its stack (dropping
  pads) and sends it, or restores it on refusal. A receive row pushes every arrival that fits onto its stack, oldest on
  top, and leaves the rest waiting in the endpoint. Sending and receiving rows go in separate tables, so a refused send
  waiting on a stack is never taken for an arrival; a table's Output rows carry arrivals on. Oracle A1 (2026-11-21):
  `nodus_headless_runtime_test` all OK; `link_rows_test` PASSED (50 control packets each way, F32 [1024,4096] each way,
  identical). `link_metrics_test` fails two window checks on this host, and so does the main branch's version, run the
  same way: its 64-step window is full before the packets go. `test_volunteer_cycle_link` builds; it needs a Vulkan
  device, which this host has none of, so it has not run.
- **File rows.** Owner, 2026-11-21: "reading and writing files belongs to reading and writing file rows"; "single
  event file reads are fine for loading actual row content but not for runtime unless it's an intentional file cache
  or process destination or source". `file_read` (a source folder: the next file by name, moved into sent/ first, one
  U8 tensor on the stack at a time) and `file_write` (a destination folder: each tensor to a new exclusively-named
  `<n>.arrival`, staged under a hidden name, never overwriting) are module_library rows. The link never touches a file.
- **Two system faults found on the way** (Oracle A1, 2026-11-21, `tests/file_link_test.cpp`):
  1. `EdgeTensorFifo::push_from_frame` (host path) published from a frame without checking it held an element of the
     edge's type, then failed to pop. An Output row on an idle tick re-sent the last tensor handle (each 8 MiB arrival
     was written twice, the second time under an id already destroyed and reused); a reset frame published null
     handles -- the "pads" rows had been dropping. It now checks the frame's top element, as the backend path did.
  2. An Output row moves one value a tick and leaves it on the stack while its edge is full; a table not in stack
     loopback resets its stack every tick, so anything a row pushed beyond that was lost. A row that produces tensors
     (`file_read`, a link receive row) pushes one at a time, only once the last has gone, and belongs in a
     stack-loopback table.
  After both: 12/12 files in one canvas, and 22/22 across the link (two generations with the same names, six 8 MiB
  each), byte for byte, sender 22 sent / receiver 22 received. canvas_row_test, nodus_headless_runtime_test and
  link_rows_test pass again. Tests with tensor-typed Output rows that need a GPU (volunteer cycle, GPU inventory, OpenGL
  rows) have not run on this change.
