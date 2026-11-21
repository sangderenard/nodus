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
