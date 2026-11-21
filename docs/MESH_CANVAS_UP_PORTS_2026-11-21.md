# Mesh canvas: canvas ports, "up", per-node write, private canvases (2026-11-21)

Ticket `e12fd0f545c44f868591bbe7ef505cc1` (follows `e2f19532`, regional state broadcast).

## The owner's words

- "all nodes are the same"
- "the edge needs to be to the parent canvas of the mesh so that needs working out in the canvas it would be going to a
  canvas port in the top tools area then you'd need an up option to the mesh"
- "they all run the mesh but only individual machines have their write permission and we should allow some privacy from
  read for systems that have sensitive canvases"

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
2. **Up.** A canvas port marked *up* is bound to the mesh canvas. Because a canvas is one per process, the node canvas
   and the mesh canvas are separate processes on the same machine joined canvas to canvas by a link; the up port is that
   link's end on the node side, and in the mesh the node appears as a table whose contacts are its up ports.
3. **Mesh.** Every machine runs the same mesh canvas: one table per node. A node writes only its own table (single
   writer per table, the edge rule lifted to tables); every other machine holds a read replica of it, fed over the link.
4. **Read privacy.** A canvas may name its readers. Its up ports and contents reach only them, sealed to their keys
   (state_broadcast's recipient model: X25519-wrapped content key, signed manifest). Others see that the node exists.
5. **State broadcast on this shape.** A generation is one node's up stream; its readers are the enrolled regions; the
   pack/accept code is unchanged, and the reader rows live in the mesh canvas, the same on every node.

## Stages

1. Canvas ports on the ROOT module: headless ABI to declare/list/remove, edges to and from them, saved in the canvas
   file, drawn as toolbar LEDs (the stubbed connect mode becomes the real one). Test: a row-port edge into a canvas port
   carries tensors with no host drain and no loss under backpressure.
2. Up binding: a canvas port marked up opens a local link to the machine's mesh canvas; the mesh shows the node as a
   table with those contacts.
3. Mesh replication: each node's table replicated read-only to the other machines' meshes; single writer enforced.
4. Read privacy: per-canvas reader lists; sealed delivery; the non-readers' view.
