# Graph entities: the player symmetry between game and execution

Status: audited design contract. Nodus does not currently contain the complete entity registry described here.

## The idea

A Nodus entity is a stateful player inside the execution environment. It is not a process token, not a datum, not an edge, and not merely the remote session controlling it.

The entity has a location in the Nodus world. Wherever that location is visible, a sprite or another presentation of the entity appears. The entity can move under its own controller through or across the graph; it is not forced to move only when a process edge carries a moment, value, or tensor. Its state persists while it moves.

That supplies a missing symmetry:

| Game engine concept | Execution-engine counterpart |
|---|---|
| player/entity | durable graph inhabitant |
| world location | graph anchor plus optional world transform |
| sprite | projection of entity presentation at its current location |
| controller | human, LLM, script, network peer, or replay |
| interaction | sanctioned tool/row/world action at a location |
| inventory/state | typed entity state and artifact references |
| game tick | world transition coordinated with, but distinct from, graph scheduling |

With this symmetry, a game does not need to fake a player as graph data, and the execution engine does not need to pretend every act of agency is dataflow. A game controller and an LLM controller can inhabit the same entity contract. The graph can compute effects for the game world, while entities can inspect and operate graph structures as places.

## Current repository audit

Several existing facilities are useful ingredients, but none is the entity system:

- [`include/generic_object.h`](../include/generic_object.h) provides a transportable class schema, field slots, and bound methods. It is a plausible state payload or behavior-binding substrate. It has no instance registry, world location, controller lease, or renderer projection.
- [`include/canvas_abi.h`](../include/canvas_abi.h) gives modules stable UUIDs, canvas positions, and programmatic movement. Those are useful location anchors. A module is not an entity, and moving an entity must not mean moving its host module.
- [`include/subject_capture.h`](../include/subject_capture.h) models captured geometry and ray events. Its “subject” is not an agent/player abstraction.
- [`include/meta_cloud.h`](../include/meta_cloud.h) stores table/meta-group vertices, rings, and overlays, but its snapshot population is incomplete and it does not define entity agency or location.
- [`include/nodus_headless_abi.h`](../include/nodus_headless_abi.h) can construct and execute graphs without rendering. It has no entity CRUD, movement, or observation calls today.

Therefore the design must be added deliberately rather than inferred from an existing type with a similar name.

## Architectural laws

1. **Entity identity is durable.** Presentation, controller, and location may change without changing `entity_id`.
2. **Controller identity is separate.** Reassigning a human player to an LLM or vice versa changes a lease, not the entity.
3. **Location is first-class state.** It is saved, observed, versioned, and included in transition receipts.
4. **Movement is not dataflow.** It neither consumes nor produces a graph datum unless an explicit interaction does so.
5. **No new graph edge type is implied.** Existing edges retain their execution and transport meanings.
6. **The sprite is a view.** Headless execution retains the same entity and location even when no sprite is rendered.
7. **Entity actions use sanctioned Nodus operations.** Being “inside” the graph grants no pointer access or capability bypass.
8. **Stable identifiers are used across persistence.** Load-local module or row indices cannot be durable locations.
9. **Game and execution clocks may differ.** Their transitions must have an explicit ordering rule, not a shared-clock assumption.
10. **State and behavior remain inspectable.** Opaque provider session state is not the entity's canonical state.

## Minimal entity record

The semantic record should be representable as POD plus referenced typed state:

```text
EntityRecord
  entity_id:          stable 128-bit or namespaced 64-bit id
  revision:           monotonic entity-state revision
  class_id:           schema/binding identity
  lifecycle:          spawning | active | suspended | destroyed
  state_ref:          typed state-object or content-addressed snapshot
  location:           EntityLocation
  presentation_ref:   sprite/presentation descriptor
  controller_lease:   optional lease id
  last_transition:    receipt id
  tags:               bounded interned identifiers
```

`GenericObject` may eventually hold the small typed fields behind `state_ref`, provided its ownership, instance IDs, persistence, and class-binding lifetime are made explicit. Large data remains in tensors or artifacts referenced by handle/hash; it should not be copied into an entity record.

## Location model

An entity location needs both **semantic attachment** and **visual projection**.

```text
EntityLocation
  world_id
  anchor_kind: world | canvas | module | row | port | edge | overlay | free
  anchor_id: stable UUID/key when anchored
  local_transform: x, y, z, orientation, scale
  location_revision
```

Examples:

- A player standing “at” a module uses the module's stable UUID plus a local offset.
- A debugging entity inspecting one row uses module UUID plus a stable row identity once one is available.
- A packet visualization may appear on an edge, but it remains an entity visualization only if it has independent identity and agency; ordinary edge data is not promoted to an entity.
- A game character may occupy free world coordinates while still interacting with nearby graph modules.

The `anchor_kind` is a tagged location reference, not a new topological edge. If an anchor disappears, policy must be explicit: reject deletion, move the entity to a parent/world anchor, suspend it as unplaced, or destroy it. Silent reassignment is forbidden.

### Free movement

“Move freely through the graph” means entity relocation is requested by the controller and admitted by world policy. It is not limited to execution order or edge adjacency.

Policies may still impose game-specific constraints:

- unrestricted teleport among observable anchors;
- adjacency traversal over graph topology;
- continuous movement in canvas/world space;
- portal or permission-gated movement;
- movement cost proportional to world distance or graph distance.

The entity mechanism supports each policy; it must not hard-code one as the nature of entity movement.

## Presentation and sprite contract

The entity core should know only a presentation descriptor, never renderer-owned pointers:

```text
EntityPresentation
  presentation_id
  kind: sprite | glyph | mesh | hidden | custom
  asset_ref
  animation_state
  facing/orientation
  tint/layer/scale
  label and accessibility text
```

Rendering performs a join:

```text
EntityRecord + resolved EntityLocation + Presentation asset -> draw instance
```

The renderer does not own entity state. A headless observer can report the same descriptor and resolved location. A 2D canvas may draw a sprite; a 3D/game surface may instantiate a mesh; a table may show a glyph at a row. All are projections of one canonical entity.

Animation state belongs in the presentation component only when it affects appearance. Gameplay/execution truth—health, inventory, capabilities, selected tool, task state—belongs in entity state and cannot be inferred from animation frames.

## Controller and action contract

Controllers acquire leases over entities. A lease specifies:

- entity id;
- controller/session id;
- allowed observation projection;
- allowed action families and targets;
- world/graph scope;
- start and expiry revisions/times;
- resource budgets;
- exclusivity or shared-control mode.

Core actions are:

```text
spawn(entity_spec, initial_location)
observe_entity(entity_id, projection)
move(entity_id, destination, movement_policy)
interact(entity_id, target, action, arguments)
set_presentation(entity_id, approved_delta)
suspend(entity_id)
resume(entity_id, location?)
destroy(entity_id)
```

`interact` is the bridge to execution. It may inspect a row, invoke an authorized tool, push or pull through a declared port, adopt an artifact, or trigger a game action. The receipt names both the entity and the underlying sanctioned Nodus operation.

## World/execution ordering

The entity system must specify how game/world updates interleave with graph scheduling. A safe initial order is:

1. capture controller proposals for world tick `n`;
2. validate leases, revisions, and capabilities;
3. commit admitted entity movement and non-execution state changes;
4. translate admitted interactions to existing Nodus operations;
5. run a bounded scheduler step;
6. collect outputs/events;
7. apply world consequences;
8. publish one ordered transition receipt and observation.

This does not require both engines to have the same timestep. The receipt records world tick, scheduler tick range, and ordering. Later, independent clocks can be coordinated through timestamped boundaries, but the initial implementation should be deterministic.

## Persistence and replay

Canvas persistence currently saves graph topology, while some live FIFO/runtime state requires separate handling. Entity persistence should therefore be its own versioned section or companion manifest, containing:

- entity records and revisions;
- stable location anchors;
- state snapshots/references;
- presentation descriptors and asset hashes;
- lifecycle state;
- controller lease only if a scenario explicitly persists it;
- last transition receipt.

On restore, anchors are resolved only after the graph's stable UUID maps are rebuilt. Unresolved anchors produce named errors and the declared orphan policy. Replays record movement independently from graph data events.

## Environment and learning implications

For an LLM, embodiment makes location part of the observation and action space. The model can learn not only which tool to call, but where to position itself, what it can perceive from there, which interactions are local, and how to coordinate with other entities.

Programmatic rewards can include:

- reaching a requested location;
- observing only what the location and capability grant permit;
- selecting an efficient route when a scenario constrains movement;
- completing an interaction at the correct anchor;
- coordinating occupancy or access among multiple entities;
- preserving entity state across controller handoff;
- producing correct execution results and game-world consequences.

The entity is also the common unit for human demonstrations and model learning: a recorded human-controlled player trajectory can be replayed or used as training data for an LLM controller without translating between unrelated game and execution representations.

## Proposed implementation sequence

### E0: identity and registry

- Define versioned entity, location, presentation, and lease structs.
- Add a canvas/world-owned registry with stable IDs and revisions.
- Add create/read/update-lifecycle APIs without rendering.
- Test identity stability and stale-revision refusal.

### E1: anchoring and persistence

- Resolve module UUID and canvas/free-world locations.
- Persist entity records alongside a graph fixture.
- Define missing-anchor policy and test save/load.
- Do not add row/port anchors until they have durable identifiers.

### E2: movement and receipts

- Add revision-checked movement independent of scheduler execution.
- Emit ordered movement receipts and observation deltas.
- Test teleport, topology-constrained movement policy, conflicts, and invalid anchors.

### E3: sprite projection

- Add a renderer query that returns resolved draw instances.
- Draw one sprite/glyph on the canvas without giving the renderer ownership.
- Prove the same entity works headlessly.

### E4: interaction bridge

- Bind an entity interaction to existing registered tool/headless operations.
- Preserve capability checks, typed values, and underlying action receipts.
- Test an entity moving to a module, invoking a row tool, and observing the result.

### E5: game/execution integration

- Define deterministic world/scheduler ordering.
- Add a fixture where execution output changes game state and entity action changes execution state.
- Exercise controller handoff among script, human input, and model adapter.

## Acceptance criteria

- An entity persists state and identity while moving between two module anchors.
- Movement occurs with no edge token and no graph-topology mutation.
- The same entity is visible as a sprite in the GUI and as structured state headlessly.
- A controller can be replaced without replacing the entity.
- A stale or unauthorized movement is refused with no state change.
- Save/load preserves entity state and resolves stable anchors.
- Anchor deletion follows the declared orphan policy.
- Two entities can occupy or contend for locations according to scenario policy.
- An interaction invokes an existing Nodus operation and links both receipts.
- A deterministic fixture proves game-state-to-execution and execution-to-game-state flow.

## Non-goals

- Encoding entities as special graph edges.
- Treating every datum or process activation as an entity.
- Coupling entity lifetime to one LLM conversation.
- Making sprites authoritative state.
- Giving an entity ambient access to the graph because it is visually nearby.
- Replacing existing scheduler, tool, stack, or edge semantics.

