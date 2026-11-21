# Existing-host joins and rope-flow editor (design note, 2026-11-14)

## User contract

An application that needs a Nodus canvas first looks for a compatible running
host. It starts a new Nodus only with an explicit `new-instance` choice. A
Nodus host **refuses outside joins by default**; its owner must explicitly
enable admission. Both decisions are required before any outside client can
change the running canvas. A refused join is reported to the caller; it never
silently launches a second instance.

The ordinary successful operation is to place the requested subgraph in the
existing canvas, preserving the other tables, ropes, state, and running work.
The host returns a receipt containing the canvas identity and revision, the
new modules' stable identities, and the placement bounds. The UI immediately
selects and frames the imported subgraph and shows its receipt, so the user can
visually verify what changed. Reject name/identity collisions or incomplete
edges before changing the live graph.

## Current substrate and gap

`tools/run_canvas.py` has an authenticated loopback console, but it is
**read-only**. `SNAPSHOT` and `DIGEST` can prove a revision, while `GRAPH`
exposes module/edge identities and positions. It cannot admit or import a
subgraph. `nodus_graph` has a strict `GraphDefinition` and
`apply_definition`, but applying it materializes a graph; it does not provide a
transactional merge into a live canvas. `gp_canvas_load_from_file` loads a
canvas file; it is not evidence of an append operation. CANVAS V4 also does
not preserve live FIFO samples and reader cursors. A host-side, tick-boundary
admission and append transaction must be built on the existing native graph,
table, edge and FIFO APIs. No client thread should mutate `Graph` directly.

The admission protocol should bind to the running host's owner and local
capability, advertise its accept-joins flag, validate a versioned subgraph
definition and resource/port contracts, and commit at a safe graph boundary.
Use `DIGEST`/`GRAPH` before and after as the visible receipt. A client may
join only a host whose admission flag is on and whose declared contract is
compatible; an open process is not automatically a writable process.

## Rope-flow presentation

Build the comfortable editor on Nodus state, not a duplicate graph model.
Colored background regions contain subgraphs; clicking a region focuses it,
and its color follows the contained modules/rope grouping. Preserve native
stable IDs, contact ordinals, FIFO contracts and edge creation tools. Offer
direct manipulation for moving groups, placing modules, creating/reconnecting
edges, and descending into a contained canvas/table. Motion can use client-side
spring layout, but it commits positions through the host's graph mutation
transaction; physics must not silently rewire execution. The current canvas
ABI already has subgroup toolbar colors and rope meta-groups, which should be
examined before adding another grouping representation.

## Implementation order and evidence

1. Define host admission flag and authenticated local join handshake; default
   refusal is covered by a test, as is explicit client `new-instance`.
2. Add atomic append of a strict `GraphDefinition` subset to a running canvas,
   with identity/edge/FIFO validation and rollback on failure.
3. Return before/after revisions and stable IDs; show selected imported group
   in the running canvas, then verify with `GRAPH` and screenshot evidence.
4. Add the colored rope-flow interactions against those same identities.

This note does not claim that existing-host mutation or the editor is already
implemented.
