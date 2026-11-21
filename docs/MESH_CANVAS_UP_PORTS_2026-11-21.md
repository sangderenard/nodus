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
