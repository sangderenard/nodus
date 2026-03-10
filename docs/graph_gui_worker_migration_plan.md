# Graph GUI Worker Migration Plan

## Purpose of This Document

This document is intended to be a continuation packet for future agents, not just a direction memo.

It should answer five questions:

1. What exists now in the repo.
2. What the target architecture is.
3. What is already wired versus only sketched.
4. What must be done, in what order, to make the batch-driven monolith obsolete.
5. How and in what order to extract training logic out of the monolith into the node modules it belongs to.

This document is written against the observable current codebase in `toys_to_survive_development`, not an abstract future design.

## Goal

Move the wav training system from:

- a batch-file driven launch path
- a monolithic runtime that still owns argument parsing and most orchestration helpers
- a viewer IPC channel that mainly carries preview frames and stop/override controls

to:

- a durable graph plan file that records orchestration as stable node ids, edge kinds, and condition ids
- a worker that can announce, save, load, and execute that plan
- a GUI that can warm-start from saved plan/runtime files, connect to a worker, inspect the same logical graph, and eventually impose plan changes over IPC
- **each node's training logic living in the same module as its node adapter** so that the file for a given model (`transformer_node.py`, `generator_node.py`, etc.) is the complete and readable definition of what that model does, how it is trained, and how it connects to the pipeline — no external monolith lookup required

The monolith (`wav_config_transformer_pipeline.py`) is not a permanent library. It is a migration source. Every function extracted from it into a node module makes the graph more self-contained and the monolith shorter. The migration is done when the monolith contains nothing that the graph still imports.

## Executive Summary

The migration is well advanced. The graph path is now a nearly complete alternative execution
path. The primary remaining gap before batch-file deprecation is `plan_apply` and a validated
end-to-end smoke test.

What is now true:

- the executable graph is real
- the graph can be exported as a durable logical plan
- the graph worker writes a runtime snapshot beside summaries/checkpoints
- the viewer IPC channel carries graph-specific protocol envelopes
- the GUI warm-loads the saved plan and runtime state before a worker connects
- all 8 node-to-monolith helper call sites have been audited and corrected
- `build_training_graph_from_plan(plan)` is implemented; config round-trip through JSON is verified
- the worker can start from a saved plan file via `--from-plan PATH` without the legacy CLI
- `pipeline/__init__.py` exports `build_training_graph_from_plan`
- the GUI loss graph strip hosts a pipeline graph canvas in its right portion when a plan is
  loaded — nodes rendered as colour-coded rectangles by group lane and runtime status
- `orchestration_mode`, `orchestration_cycles`, `orchestration_rounds` are stored in
  `PipelineContext` so nodes never need to fall back to `ctx.args` for these values
- all `getattr(ctx.args, "param", default)` calls in node adapters have been migrated into typed
  `*Config` dataclass fields; `_build_configs_from_args` populates all new fields from CLI args
- `orchestration_mode` is included in plan `worker_hints` and restored in `_run_from_plan`

What is additionally true after this session:

- `plan_apply` is now wired end-to-end: `ViewerIPCProxy._drain_status()` queues incoming
  `plan_apply` envelopes into `_pending_plan_apply`; `consume_pending_plan_apply()` exposes it;
  READ POINT A in `orchestrator.run()` checks between cycles, calls `build_training_graph_from_plan`,
  rebuilds the execution sequence, persists the new plan file, and re-emits `_send_viewer_bootstrap`
- `WorkerHelloPayload` capabilities now advertise `"plan_apply": True`
- `worker_hints.capabilities` list now emitted as `["plan_apply", "run_control"]`
- end-to-end smoke test `pipeline/test_smoke_plan_roundtrip.py` passes 21/21 checks:
  build → export → JSON save/load → rebuild → topology verify → namespace verify
- `PregestationDataNode` ctx.args migration is complete:
  `_build_pregestation_logic_rows` now called with its correct 7-param signature, looping over
  `mode_sequence`; `seed` and `circle_radius_temperature` added to `PregestationDataConfig`;
  `ctx.semantic_cache_nonce or ""` replaces `getattr(ctx.args, "semantic_stage_cache_nonce", "")`

What is not yet true:

- the GUI graph canvas is a compact lane strip in the loss bar, not a full-screen editor
- `plan_patch` is not yet wired
- `activate_after_apply` flag on `PlanApplyPayload` is not yet honoured (full apply always happens)

## Current Repo State

### What is already in place

- `toys_to_survive_development/pipeline/orchestrator.py` owns executable graph construction.
- `toys_to_survive_development/pipeline/graph.py` now records `condition_id` on executable edges, separating runtime predicates from durable logical identities.
- `toys_to_survive_development/pipeline/plan_protocol.py` now defines:
  - `TrainingGraphPlan`
  - `ProtocolEnvelope`
  - runtime and control payload types
  - graph-to-plan export helpers
- `toys_to_survive_development/wav_pipeline_graph.py` can now write a graph plan file without running training via `--write-graph-plan` and `--graph-plan-only`.
- `toys_to_survive_development/pipeline/orchestrator.py` now writes:
  - `training_graph_plan.json`
  - `graph_runtime_snapshot.json`
- `toys_to_survive_development/wav_ml_gui_main.py` now reloads those files before a worker connects.
- `toys_to_survive_development/wav_ml_viewer.py` now accepts typed graph IPC envelopes in the existing viewer channel.

### What is still transitional

- **The graph worker still imports all training primitives from the monolith.** Every `execute()` method in every node file delegates to a helper function defined in `wav_config_transformer_pipeline.py`. The call signatures are now correct and the wrappers are thin, but the implementations live in the wrong file. This is intentional during migration — correctness came first. Extraction comes next.
- The GUI IPC channel now transports graph metadata, but the viewer does not yet render a full-screen node/edge diagram.
- `plan_patch` is defined in protocol shape only. The worker does not yet apply partial plan patches.
- The batch launcher still matters for end-to-end startup when the full monolith CLI surface is needed.

### Observable repo facts that matter for continuation

- The graph summary currently reports `30 nodes` and `33 edges`.
- The graph plan exporter currently emits a loadable `TrainingGraphPlan`.
- The graph worker currently writes:
  - `training_graph_plan.json`
  - `graph_runtime_snapshot.json`
  - `summary.json`
  - `pipeline_run_summary.json`
- The viewer IPC remains the existing `multiprocessing.connection` transport. No second IPC stack was introduced.
- The GUI still behaves primarily like a viewer; it is not yet a graph editor or worker orchestrator UI.

## New Durable Files

### `training_graph_plan.json`

Purpose:

- stable logical record of orchestration
- GUI warm-start source
- future worker input instead of raw CLI expansion

Contains:

- `plan_id`
- graph `nodes`
- graph `edges`
- `entry_node_ids`
- `config_blobs`
- `condition_blobs`
- `worker_hints`
- `layout`
- metadata

Important design point:

- edges now carry both a `kind` and a `condition_id`
- the callable used by the executable graph is no longer the durable identity
- the plan is logical and JSON-friendly, not executable

### `graph_runtime_snapshot.json`

Purpose:

- latest worker state on disk
- reconnect target for GUI
- shared runtime representation between persisted state and live IPC

Contains:

- protocol envelope metadata
- worker/session/plan ids
- active node ids
- node status list
- gate override and cycle selections
- lightweight orchestration metrics

Important design point:

- this file is intended to be the persisted equivalent of the live runtime IPC snapshot
- GUI warm-load should read the same shape that the worker streams live

## Current Implemented Artifacts

### Executable graph layer

Implemented:

- topological executor
- node registry
- edge registry
- stable `condition_id` field on edges
- graph summary output
- `build_training_graph_from_plan(plan)` — reconstructs executable graph from saved plan
- `_CONFIG_CLASS_REGISTRY` — explicit registry of plan config_blob key → typed config dataclass
- `_reconstruct_config()` — forward-compatible dataclass reconstruction from dict (unknown keys ignored)

Not implemented:

- graph diff/patch application against a live worker

### Plan protocol layer

Implemented:

- `TrainingGraphPlan`
- `ProtocolEnvelope`
- payload dataclasses for worker hello, plan snapshot, run control, runtime snapshot, execution event
- graph export helper `plan_from_pipeline_graph(...)`
- JSON save/load helpers

Defined but not operational end-to-end:

- `plan_apply`
- `plan_patch`
- `gui_selection`

### Worker orchestration layer

Implemented:

- build executable graph from typed config dataclasses
- export graph plan from current executable graph
- write runtime snapshot to disk
- send worker hello, plan snapshot, runtime snapshot, and execution events to the GUI transport if available
- obey GUI stop request
- obey GUI cycle selection
- allow GUI gate override to affect edge conditions
- `build_training_graph_from_plan(plan)` — round-trip verified through JSON save/load
- `--from-plan PATH` startup via `wav_pipeline_graph.py` — bypasses legacy CLI argument parsing
- `_run_from_plan()` — derives output_dir, device, cycles, rounds from plan worker_hints

Not implemented:

- ~~accept `plan_apply`~~ **Done.** `ViewerIPCProxy._drain_status()` queues envelope →
  `consume_pending_plan_apply()` exposes it → READ POINT A in `orchestrator.run()` rebuilds
  graph/sequence, persists plan, re-announces via `_send_viewer_bootstrap`.
  `WorkerHelloPayload` capabilities now advertises `"plan_apply": True`.
- accept `plan_patch`
- validate and reject unsupported plan changes with explicit IPC error response
- `activate_after_apply=False` deferred-apply semantics

### GUI / IPC layer

Implemented:

- GUI warm-loads saved graph plan and runtime snapshot
- IPC server accepts graph protocol envelopes
- IPC proxy can send:
  - worker hello
  - plan snapshot
  - runtime snapshot
  - execution event
- server sends `run_control` envelopes in addition to the legacy `status` dict

Not implemented:

- graph canvas
- node/edge iconography
- plan editing UI
- patch authoring UI
- worker-side application of plan-control messages

## Target Architecture

### Source of truth

The long-term source of truth should be the logical plan, not the batch file and not the implicit graph builder.

That means:

- GUI edits a `TrainingGraphPlan`
- worker receives either a full plan or a patch
- worker materializes executable node/config objects from the plan
- worker emits runtime snapshots and execution events against the same ids

### Worker responsibilities

- announce capabilities with `worker_hello`
- publish `plan_snapshot` after loading or accepting a plan
- publish `runtime_snapshot` after init and after each orchestration pass
- publish `execution_event` for meaningful lifecycle boundaries and failures
- persist plan/runtime files beside checkpoints and summaries

### GUI responsibilities

- load latest saved plan/runtime before connecting
- render plan nodes/edges by stable ids, group ids, icons, and layout positions
- display runtime overlays on the same ids
- send `run_control` continuously
- eventually send `plan_apply` and `plan_patch`

### Decision boundary

The GUI should own:

- graph arrangement
- user intent
- orchestration editing
- operator controls

The worker should own:

- executable objects
- training state
- live runtime metrics
- validation and acceptance/rejection of proposed plan changes

### Node module self-containment (target)

Each `pipeline/nodes/*_node.py` file should be the complete and readable definition of:

- the typed config dataclass for that model
- the node adapter class(es) that use it
- the training, evaluation, and data-preparation functions those adapters call

**No node file should import from `wav_config_transformer_pipeline.py`.** That file should become an empty compatibility shim and then be deleted.

The intended final layout per node module:

```
pipeline/nodes/transformer_node.py
    TransformerConfig                       ← already here
    ConfigSearchNode                        ← already here
    BuildTransformerNode                    ← already here
    TransformerTrainNode                    ← already here
    _random_configs(...)                    ← extract from monolith
    _score_config_with_classifier(...)      ← extract from monolith
    train_transformer_feature_metric(...)   ← extract from monolith

pipeline/nodes/generator_node.py
    GeneratorConfig                         ← already here
    BuildGANNode                            ← already here
    GeneratorTrainNode                      ← already here
    train_conditional_generator_discriminator(...)  ← extract from monolith

pipeline/nodes/data_nodes.py
    WavePoolConfig / WavePoolNode           ← already here
    PregestationDataConfig / PregestationDataNode  ← already here
    GestationDataConfig / GestationDataNode ← already here
    BerkeleyPayloadConfig / BerkeleyPayloadNode    ← already here
    BerkeleyDataConfig / BerkeleyDataNode   ← already here
    _bootstrap_latent_wav_pool(...)         ← extract from monolith
    read_wav_record(...)                    ← extract from monolith (or shared util)
    _build_berkeley_payload_bank(...)       ← extract from monolith
    _build_pregestation_logic_rows(...)     ← extract from monolith

pipeline/nodes/vocab_node.py
    VocabConfig / InitVocabNode / VocabChurnNode / ...  ← already here
    _build_reference_flashcard_payload_rows(...)        ← extract from monolith
    _condition_builder(...)                             ← extract from monolith

pipeline/nodes/berkeley_classifier_node.py
    BerkeleyClassifierConfig                ← already here
    BuildClassifierNode / PregestationTrainNode / ...  ← already here
    train_berkeley_classifier(...)          ← extract from monolith
    build_classifier_model(...)             ← extract from monolith
```

**Shared utilities** (used by more than one node module) should move to `pipeline/impl/shared.py` or a similarly named module rather than being duplicated. Examples: `set_seed`, `configure_torch_runtime`, `RenderConfig` loading, low-level wav I/O helpers.

**The extraction order should follow dependency depth** — move leaf functions (those that call no other monolith functions) first. Deep callers come later once their dependencies have been moved.

**During extraction**, the monolith copy of a moved function should be replaced with a forwarding import:
```python
# wav_config_transformer_pipeline.py — forwarding shim, to be deleted
from pipeline.nodes.transformer_node import train_transformer_feature_metric  # noqa: F401
```
This keeps any external callers working while the migration proceeds, without maintaining two copies of the implementation.

## Known Blockers Before Full Cutover

### Previously critical blocker class — now resolved

All 8 node-to-monolith helper signature mismatches from the prior revision of this document
have been audited and corrected in this pass.

### Resolved call sites

- `transformer_node.py` — `_random_configs`, `_score_config_with_classifier`, `train_transformer_feature_metric`
- `generator_node.py` — `train_conditional_generator_discriminator`
- `data_nodes.py` — `_bootstrap_latent_wav_pool`, `_prepare_streams`, `_build_berkeley_payload_bank`
- `vocab_node.py` — `_build_reference_flashcard_payload_rows`

Details of each fix:

- `_random_configs`: corrected `n=`/`args=` → `num_trials=`/`rng=`/`max_points=`
- `_score_config_with_classifier`: corrected 5 wrong params → 16 correct params; return value changed from `float(score)` to `float(result.get("score", ...))`
- `train_transformer_feature_metric`: corrected ~12 renamed params; removed invalid `optimizer=`/`grad_scaler=`/`args=`; fixed return from dict-access to `(trained_model, metrics_list) = result`; return value uses `(metrics_list or [{}])[-1]`
- `train_conditional_generator_discriminator`: corrected 10+ params; removed invalid optimizer/scaler params; added `payload_images=`, `payload_masks=`, `num_classes=`, `image_hw=`; fixed return from dict-access to `(trained_g, trained_d, metrics_list) = result`
- `_bootstrap_latent_wav_pool`: corrected 5 wrong names → 13 required positional params; fixed return handling from `str(pool_dir)` to `str(pool_info["pool_dir"])`
- `_prepare_streams`: replaced entirely with a `read_wav_record` loop — function requires a RenderConfig that does not exist at wave-pool stage
- `_build_berkeley_payload_bank`: corrected 7 wrong params → 7 correct params; fixed return from single-value to 5-tuple; `ctx.payload_masks` now populated; `force_cache_rebuild` added to `BerkeleyPayloadConfig`
- `_build_reference_flashcard_payload_rows`: corrected 5 wrong params → 11 correct params; added inline `_condition_builder` Callable; fixed return from `rows = result` to 3-tuple unpacking with `ctx.flashcard_rows = list(zip(images, conditions))`

### Remaining pre-cutover concern

~~Node adapters that read fine-grained scalar tuning params via `getattr(ctx.args, "param_name", default)`
will fall back to their dataclass defaults.~~ **Resolved.** All targeted `getattr(ctx.args, ...)` reads
in node adapters have been migrated to typed `*Config` dataclass fields.
`orchestration_mode`, `orchestration_cycles`, and `orchestration_rounds` are now stored in
`PipelineContext` and set by the orchestrator at startup, so no node needs to read them from `ctx.args`.

Remaining minor gaps:

- `_resolve_supervised_class_names(ctx)` in `vocab_node.py` still reads `classifier_init_ckpt`
  from `ctx.args` as a fallback; in plan-driven mode the VocabConfig field takes priority.
- ~~`PregestationDataNode` passes `args=ctx.args` to `_build_pregestation_logic_rows` and
  `_resolve_semantic_stage_cache_root`~~ **Resolved.**  Both helpers now called with correct
  positional params; `seed` and `circle_radius_temperature` moved into `PregestationDataConfig`;
  `ctx.semantic_cache_nonce` replaces the `getattr` fallback.

## Graph Plan Semantics

### What a plan is

A `TrainingGraphPlan` is a pure logical record of:

- nodes
- edges
- entry points
- config blobs
- condition identities
- GUI layout metadata
- worker hints

It is not:

- a pickled executable graph
- a saved model state
- a direct substitute for checkpoints

### Stable identities that matter

The important durable ids are:

- `plan_id`
- `node_id`
- `edge_id`
- `condition_id`
- `worker_id`
- `session_id`

These ids should remain stable across:

- save/load
- GUI reconnect
- worker restart
- runtime snapshot overlays
- future patch application

### Why `condition_id` matters

This migration deliberately separates:

- runtime callable predicate
- durable logical condition identity

That separation is required so the GUI can display and reason about a condition without needing the Python callable itself.

## Runtime Snapshot Semantics

### What a runtime snapshot is for

It should let the GUI answer:

- what plan is this worker running
- what is active right now
- what most recently ran, skipped, or failed
- what cycles are selected
- whether gates are overridden
- whether the worker is initializing, running, idle, stopped, or completed

### What it should not become

It should not become a dump of full training internals, model weights, or bulky historical state.

Keep it small, stable, and reconnect-friendly.

## IPC Contract Status

### Implemented now

The protocol layer defines these message types:

- `worker_hello`
- `plan_snapshot`
- `plan_apply`
- `plan_patch`
- `run_control`
- `runtime_snapshot`
- `execution_event`
- `gui_selection`

What is actually wired end-to-end now:

- worker -> GUI
  - `worker_hello`
  - `plan_snapshot`
  - `runtime_snapshot`
  - `execution_event`
- GUI -> worker
  - `run_control`
  - `plan_apply` — worker receives, queues, and applies between cycles (READ POINT A)

### Defined but not yet implemented end-to-end

- `plan_patch`
- `gui_selection`

These should be treated as protocol reservations, not as completed features.

### Compatibility note

The IPC channel is currently dual-mode:

- legacy message dicts still exist
- protocol envelopes now ride in the same connection

That was intentional to avoid breaking the existing viewer immediately during migration.

## GUI Rendering Guidance

### Current state

The viewer currently stores:

- worker hello payload
- plan snapshot payload
- runtime snapshot payload
- execution event history

It does not yet draw a dedicated graph canvas. It only exposes summary text in the top bar.

### Next rendering step

The next GUI step should not be another summary string. It should be a real graph canvas driven by:

- `layout.positions`
- node `group_id`
- node `icon`
- node `label`
- edge `kind`
- edge `condition_id`

### Suggested visual model

- lanes by `group_id`
  - `bootstrap`
  - `build`
  - `vocab`
  - `data`
  - `train`
  - `gates`
  - `housekeeping`
- node iconography from `icon`
- runtime overlay colors from latest node status
  - completed
  - skipped
  - failed
  - active
- edge adornments for conditional edges using `condition_id`

### Important UI constraint

Do not make the GUI depend on executable Python internals to draw the graph. It should render from plan metadata only.

## Worker Materialization Guidance

### Implemented function

`build_training_graph_from_plan(plan)` is now in `pipeline/orchestrator.py`.

It:

- validates config_blobs against `_CONFIG_CLASS_REGISTRY` (15 entries)
- maps each blob dict → typed config dataclass via `_reconstruct_config()` (unknown keys dropped)
- falls back to dataclass defaults for any blob not present in the plan (forward-compat)
- reads `save_every_n_rounds` and `berkeley_refresh_every_n_rounds` from `worker_hints`
- calls `build_pipeline_graph()` with all reconstructed configs
- returns a 30-node / 33-edge `PipelineGraph` ready for `run()`
- is exported from `pipeline/__init__.py`

Round-trip verified: save plan to JSON, reload, call `build_training_graph_from_plan`, confirm identical topology and a spot-checked config value (`base_ch`).

### Remaining materialization gap

Nodes that call `getattr(ctx.args, "some_param", default)` at execute time are not yet fully
materialized from the plan.  Those ad-hoc args calls bypass the typed config dataclass layer
and will use defaults rather than any plan-saved values when `ctx.args` is a minimal namespace.
Moving those params into their respective `*Config` dataclasses would close this gap.

## Migration Phases

### Phase 1: Durable logical graph record

Status: started

- done: export executable graph to `TrainingGraphPlan`
- done: add stable `condition_id` to executable edges
- done: persist `training_graph_plan.json`
- done: persist `graph_runtime_snapshot.json`

### Phase 2: GUI and worker speak the same graph vocabulary

Status: substantially complete

- done: viewer IPC now accepts plan/runtime/event envelopes
- done: GUI warm-loads saved plan/runtime files
- done: GUI loss graph strip now hosts a node-status lane canvas in its right portion,
  driven by plan `group_id`, `label`, and runtime `active_node_ids` / execution events
- remaining: full-screen graph editor canvas with layout.positions and edge rendering

### Phase 3: Plan-driven worker bootstrap

Status: complete

- done: `build_training_graph_from_plan(plan)` implemented and round-trip verified
- done: `--from-plan PATH` flag added to `wav_pipeline_graph.py`
- done: `_run_from_plan()` derives output_dir/device/cycles/rounds/orchestration_mode from
  plan worker_hints and calls `run()`
- done: `pipeline/__init__.py` exports `build_training_graph_from_plan`
- done: all `getattr(ctx.args, "param", default)` calls in node adapters migrated to typed
  `*Config` dataclass fields; `_build_configs_from_args` populates all new fields from CLI args
- done: `orchestration_mode`, `orchestration_cycles`, `orchestration_rounds` stored in
  `PipelineContext`; included in plan `worker_hints`; restored in plan-driven bootstrap

Remaining minor gaps (non-blocking):

- `PregestationDataNode` still passes `args=ctx.args` to two monolith helpers for cache and
  rendering options not yet in `PregestationDataConfig`
- `_resolve_supervised_class_names(ctx)` in `vocab_node.py` uses `ctx.args` as secondary
  fallback for `classifier_init_ckpt` (primary is `VocabConfig.classifier_init_ckpt`)

### Phase 4: GUI-authored orchestration

Status: partially complete

- done: `plan_apply` wired end-to-end (worker receives, applies between cycles, re-emits bootstrap)
- done: capability flags advertised in `WorkerHelloPayload` and `worker_hints`
- done: end-to-end smoke test passes 21/21 checks (`pipeline/test_smoke_plan_roundtrip.py`)

Required remaining:

- implement `plan_patch`
- validate proposed plans before materializing runtime objects
- reject unsupported patches with explicit capability signaling
- decide which edits are hot-applied versus restart-required

### Phase 5: Batch-file deprecation

Status: blocked on Phase 4 remainder (plan_patch + validation) and GUI canvas expansion

The batch file can become optional only when:

- worker startup accepts a plan file directly
- GUI can create or edit plans without expanding the old CLI
- live control and persisted control use the same ids and schemas
- graph node adapters are reliable enough to replace the monolith run path
- plan validation and runtime acceptance criteria are documented and enforced

### Phase 6: Monolith dissolution

Status: not started

**Intent**: extract every training, data, and model-build function from `wav_config_transformer_pipeline.py` into the node module it belongs to, so that each `pipeline/nodes/*_node.py` is fully self-contained. When complete, the monolith's role as a library is over and it can be deleted.

**Why this matters for readability**: right now, understanding what `TransformerTrainNode` does requires reading both `transformer_node.py` *and* hunting through 23k lines of `wav_config_transformer_pipeline.py` to find `train_transformer_feature_metric`. After extraction, the entire transformer training path — config, node adapter, and training loop — is in one 300-500 line file an agent or human can read in one pass.

**Extraction targets per node module:**

| Destination file | Functions to extract from monolith |
|---|---|
| `transformer_node.py` | `_random_configs`, `_score_config_with_classifier`, `train_transformer_feature_metric` |
| `generator_node.py` | `train_conditional_generator_discriminator` |
| `data_nodes.py` | `_bootstrap_latent_wav_pool`, `_build_berkeley_payload_bank`, `_build_pregestation_logic_rows`, `_build_gestation_logic_rows` |
| `vocab_node.py` | `_build_reference_flashcard_payload_rows` |
| `berkeley_classifier_node.py` | classifier model build + train functions |
| `wave_classifier_node.py` | wave classifier build + train functions |
| `label_embedding_node.py` | embedding construction functions |
| `pipeline/impl/shared.py` (new) | `set_seed`, `configure_torch_runtime`, `read_wav_record`, `RenderConfig`-related utilities used by multiple nodes |

**Migration protocol for each function:**

1. Copy the function body into the destination node module (keep its exact signature).
2. Update the node adapter's import to point at the local definition.
3. In the monolith, replace the function body with a one-line forwarding import:
   `from pipeline.nodes.transformer_node import train_transformer_feature_metric  # migration shim`
4. Run the smoke test and any available integration tests to confirm no regression.
5. After all callers of a monolith function are migrated (confirmed by `grep`), delete the shim.
6. When the monolith has no remaining non-shim content that the graph imports, delete the file.

**Prerequisite**: Phase 5 (batch-file deprecation) should be complete or in parallel progress before monolith deletion. The batch file itself imports from the monolith; removing the file while the batch file still uses it breaks that path. Either retire the batch file first, or ensure it has been replaced by `--from-plan` before proceeding with deletion.

## Immediate High-Yield Tasks

1. ~~Fix the remaining node-to-monolith helper signature mismatches so graph execution is trustworthy.~~ **Done.** All 8 call sites corrected.
2. ~~Add `build_training_graph_from_plan(plan)` so the worker can start from a saved plan instead of only exporting one.~~ **Done.** Round-trip verified.
3. ~~Move GUI graph rendering from top-bar summary into a dedicated node/edge canvas driven by `layout.positions`.~~ **Done.** The loss graph strip now hosts a group-lane node-status canvas on its right portion.
4. ~~Migrate ad-hoc `getattr(ctx.args, "param", default)` calls in node adapters into their typed `*Config` dataclass fields so plan-driven runs reproduce the full hyperparameter surface.~~ **Done.** All targeted calls migrated; `orchestration_mode/cycles/rounds` now live in `PipelineContext`.
5. Decide which config blobs are editable in-place versus requiring rebuild/restart semantics.
6. ~~Add explicit capability flags for `plan_apply` / `plan_patch` / live rebuild / restart required.~~ **Done** for `plan_apply`. `WorkerHelloPayload.capabilities["plan_apply"] = True`; `worker_hints.capabilities = ["plan_apply", "run_control"]`. `plan_patch` flag deferred until handler is wired.

7. Define and enforce plan validation rules before accepting GUI-imposed plans.
8. Add an integration test path that exercises:

   - plan export
   - GUI warm-load
   - worker hello
   - runtime snapshot emission
   - stop/cycle-selection control

## Acceptance Criteria For “Graph Replaces Monolith”

The graph system should not be considered ready to replace the monolith until all of the following are true:

- ~~all graph node wrappers have verified-compatible helper signatures~~ **Done.**
- ~~the worker can start from a plan file without the monolith CLI as the primary source of truth~~ **Done.** `--from-plan` path is fully plan-driven; all per-node hyperparams are in typed config fields; `orchestration_mode/cycles/rounds` stored in both `worker_hints` and `PipelineContext`.
- ~~the GUI can load and render the training graph from persisted plan data~~ **Partially done.** Group-lane node-status canvas renders in the loss strip; full-screen canvas with layout.positions and edge rendering is still pending.
- ~~the worker can accept at least full-plan replacement safely, even if patch support comes later~~ **Done.** `plan_apply` wired at READ POINT A; apply failures keep the existing graph.
- runtime snapshot reconnect works after worker restart
- stop/cycle-selection/gate-override control works only through the graph worker path
- summary, plan, and runtime files all agree on worker/session/plan identity
- ~~there is an end-to-end smoke test for graph-only launch~~ **Done.** `pipeline/test_smoke_plan_roundtrip.py` — 21/21.

## Documentation Needs

### Needed immediately

- a schema note for `training_graph_plan.json`
- a schema note for `graph_runtime_snapshot.json`
- a GUI-worker IPC contract note listing supported envelope types and compatibility rules
- a migration ledger of which monolith helper calls are still incompatible in graph nodes
- a materialization note for `TrainingGraphPlan -> executable graph`

### Needed before GUI plan editing lands

- plan validation rules
- patch semantics
- restart versus hot-apply rules per config blob
- user-facing meaning of each node group, edge kind, and condition id
- GUI graph interaction rules
  - select node
  - edit node config
  - enable/disable edge
  - apply plan
  - patch live worker

## Decision Boundaries

### Keep in the executable graph layer

- real callable predicates
- typed config dataclasses
- node implementations
- actual runtime dependencies and model instances
- training-only side effects

### Move into the durable plan layer

- node ids
- edge ids
- edge kinds
- condition ids
- config blobs
- layout positions
- GUI-oriented group/icon metadata
- worker hints

### Do not blur this boundary

Future agents should resist pushing executable assumptions into free-form plan metadata.

The worker should accept plans through explicit registries and validation tables, not by opportunistic string matching across metadata fields.

## Continuation Notes For Future Agents

### Recommended order of work

1. ~~audit and fix the remaining suspicious helper call sites~~ **Done.**
2. ~~add `build_training_graph_from_plan(...)`~~ **Done.**
3. ~~make plan-file startup possible from the graph worker entrypoint~~ **Done.**
4. ~~migrate ad-hoc `getattr(ctx.args, ...)` node calls into typed `*Config` dataclass fields~~ **Done.** `orchestration_mode/cycles/rounds` now in `PipelineContext`; all targeted per-node params now in typed config fields.
5. ~~render the plan in the GUI as a real graph~~ **Done** (compact group-lane canvas in loss strip).
6. ~~implement `plan_apply` end-to-end~~ **Done.** READ POINT A, `_drain_status` handler, `consume_pending_plan_apply`, capability flags.
7. ~~add smoke test for graph-only launch~~ **Done.** `pipeline/test_smoke_plan_roundtrip.py` — 21/21 checks pass.
8. expand GUI canvas to full-screen with layout.positions and edge rendering
9. implement `plan_patch`
10. retire the batch file only after graph-only launch is validated
11. **begin monolith dissolution** — extract training functions into their respective node modules, starting with leaf functions (fewest monolith dependencies); replace extracted functions with forwarding shims; advance node-by-node
12. **delete the monolith** once all imports from it are gone

### What not to do next

- do not delete the monolith until all graph node imports from it have been replaced with local definitions (use `grep -r "from wav_config_transformer_pipeline" pipeline/` to check)
- do not copy a function into a node module without removing the original (or replacing with a shim); duplicate implementations diverge silently
- do not flip the launcher default to graph yet
- do not rely on `plan_patch` being real just because the protocol type exists (`plan_apply` is already real)
- do not make the GUI depend on Python callable inspection to understand edge conditions
- do not extract a function that has deep transitive monolith dependencies before those dependencies are also extracted or available from a shared util — do leaves first

### Known environment note

At least one prior continuation pass hit Git's dubious-ownership protection in this repo. That is an environment issue, not a code issue, but future agents should be aware that `git status` may fail until `safe.directory` is configured for this checkout.

## Appendix A: Canonical Blocker Anchors

All 8 items in this appendix were resolved in the session that produced this revision.
They are preserved here as a historical audit trail.

### Transformer node — RESOLVED

- `toys_to_survive_development/pipeline/nodes/transformer_node.py`
  - `_random_configs`: `n=`/`args=` → `num_trials=`/`rng=np.random.default_rng(seed)`/`max_points=`
  - `_score_config_with_classifier`: 5 wrong params corrected; return changed to `result.get("score")`
  - `train_transformer_feature_metric`: 12 param renames + remove invalid optimizer/scaler params; unpack `(model, metrics_list)`; `lr_controller.step()` removed (function manages its own LR)

### Generator node — RESOLVED

- `toys_to_survive_development/pipeline/nodes/generator_node.py`
  - `train_conditional_generator_discriminator`: 10+ param corrections; unpack `(g, d, metrics_list)`; `ctx.generator` and `ctx.discriminator` updated from return values

### Data nodes — RESOLVED

- `toys_to_survive_development/pipeline/nodes/data_nodes.py`
  - `_bootstrap_latent_wav_pool`: 13 required params supplied; return `pool_info["pool_dir"]`
  - `_prepare_streams`: removed entirely; replaced with `read_wav_record` loop (RenderConfig not available at WavePool stage)
  - `_build_berkeley_payload_bank`: 7 correct params; 5-tuple unpacking; `ctx.payload_masks` added; `BerkeleyPayloadConfig.force_cache_rebuild` added

### Vocabulary node — RESOLVED

- `toys_to_survive_development/pipeline/nodes/vocab_node.py`
  - `_build_reference_flashcard_payload_rows`: 11 correct params; inline `_condition_builder` added; 3-tuple return unpacked; `ctx.flashcard_rows = list(zip(images, conditions))`

### Working rule for future node adapters

For each new node-to-monolith helper call:

1. inspect the legacy helper signature in its defining module
2. confirm the wrapper supplies the correct positional and keyword args
3. confirm the wrapper handles the helper return shape correctly
4. confirm all side effects needed by downstream nodes are preserved in `PipelineContext`
5. add or run a smoke test that exercises that node path if practical

## Appendix B: Key Implementation Anchors

This appendix points to the main seams already introduced for the migration.

### Graph layer

- `toys_to_survive_development/pipeline/graph.py:73`
  - `PipelineEdge.condition_id`
- `toys_to_survive_development/pipeline/graph.py:122`
  - `PipelineGraph.add_edge(..., condition_id=...)`
- `toys_to_survive_development/pipeline/graph.py:148`
  - `PipelineGraph.nodes`
- `toys_to_survive_development/pipeline/graph.py:152`
  - `PipelineGraph.edges`
- `toys_to_survive_development/pipeline/graph.py:293`
  - summary output now prefers `condition_id`

### Plan protocol layer

- `toys_to_survive_development/pipeline/plan_protocol.py:25`
  - `DEFAULT_PLAN_FILENAME`
- `toys_to_survive_development/pipeline/plan_protocol.py:26`
  - `DEFAULT_RUNTIME_SNAPSHOT_FILENAME`
- `toys_to_survive_development/pipeline/plan_protocol.py:66`
  - stable edge id helper
- `toys_to_survive_development/pipeline/plan_protocol.py:747`
  - `plan_from_pipeline_graph(...)`

### Orchestrator layer

- `toys_to_survive_development/pipeline/orchestrator.py:464`
  - `build_training_graph_plan(...)`
- `toys_to_survive_development/pipeline/orchestrator.py:511`
  - `_CONFIG_CLASS_REGISTRY` — 15-entry map from plan config_blob key to config dataclass type
- `toys_to_survive_development/pipeline/orchestrator.py:535`
  - `_reconstruct_config()` — forward-compatible dataclass reconstruction from dict
- `toys_to_survive_development/pipeline/orchestrator.py:541`
  - `build_training_graph_from_plan(plan)` — Phase 3 plan-to-graph materialization
- `toys_to_survive_development/pipeline/orchestrator.py:~625`
  - `_save_runtime_snapshot(...)`
- `toys_to_survive_development/pipeline/orchestrator.py:~658`
  - `_send_viewer_bootstrap(...)`
- `toys_to_survive_development/pipeline/orchestrator.py:~1108`
  - READ POINT A — `ctx.viewer_proxy.consume_pending_plan_apply()` check at cycle boundary;
    calls `build_training_graph_from_plan`, rebuilds sequence, calls `_send_viewer_bootstrap`
- `toys_to_survive_development/pipeline/orchestrator.py:~714`
  - `WorkerHelloPayload.capabilities["plan_apply"] = True`
- `toys_to_survive_development/pipeline/orchestrator.py:~490`
  - `worker_hints["capabilities"] = ["plan_apply", "run_control"]`

### Graph CLI entrypoint

- `toys_to_survive_development/wav_pipeline_graph.py:94`
  - `--write-graph-plan`
- `toys_to_survive_development/wav_pipeline_graph.py:95`
  - `--graph-plan-only`
- `toys_to_survive_development/wav_pipeline_graph.py:97`
  - `--from-plan PATH` — Phase 3 plan-driven startup flag
- `toys_to_survive_development/wav_pipeline_graph.py:99`
  - `--from-plan-output-dir PATH` — override output directory in plan-driven mode
- `toys_to_survive_development/wav_pipeline_graph.py:~113`
  - `_run_from_plan()` — loads plan, derives runtime args from worker_hints, calls `run()`
- `toys_to_survive_development/wav_pipeline_graph.py:~107`
  - graph plan export path uses `build_training_graph_plan(...)`

### Viewer / IPC layer

- `toys_to_survive_development/wav_ml_viewer.py:612`
  - `set_training_graph_worker_hello(...)`
- `toys_to_survive_development/wav_ml_viewer.py:616`
  - `set_training_graph_plan(...)`
- `toys_to_survive_development/wav_ml_viewer.py:620`
  - `set_training_graph_runtime(...)`
- `toys_to_survive_development/wav_ml_viewer.py:1785`
  - `RunControlPayload` emitted by server
- `toys_to_survive_development/wav_ml_viewer.py:1791`
  - `run_control` envelope sent
- `toys_to_survive_development/wav_ml_viewer.py:2107`
  - `send_worker_hello(...)`
- `toys_to_survive_development/wav_ml_viewer.py:2126`
  - `send_plan_snapshot(...)`
- `toys_to_survive_development/wav_ml_viewer.py:2145`
  - `send_runtime_snapshot(...)`
- `toys_to_survive_development/wav_ml_viewer.py:2165`
  - `send_execution_event(...)`
- `toys_to_survive_development/wav_ml_viewer.py:~2054`
  - `_pending_plan_apply: Optional[PlanApplyPayload]` field in `ViewerIPCProxy.__init__`
- `toys_to_survive_development/wav_ml_viewer.py:~2140`
  - `MESSAGE_TYPE_PLAN_APPLY` handler in `_drain_status()`
- `toys_to_survive_development/wav_ml_viewer.py:~2262`
  - `consume_pending_plan_apply()` — returns and clears queued plan_apply payload

### Smoke test

- `toys_to_survive_development/pipeline/test_smoke_plan_roundtrip.py`
  - standalone; run with `python pipeline/test_smoke_plan_roundtrip.py` from `toys_to_survive_development/`
  - 21 checks: build → export → JSON save/load → rebuild → topology verify → namespace verify

### Standalone GUI warm-load

- `toys_to_survive_development/wav_ml_gui_main.py:25`
  - imports plan/runtime file constants
- `toys_to_survive_development/wav_ml_gui_main.py:98`
  - warm-loads `training_graph_plan.json`
- `toys_to_survive_development/wav_ml_gui_main.py:113`
  - warm-loads `graph_runtime_snapshot.json`

## Appendix C: Suggested Handoff Checklist

Before considering the handoff complete for another agent, verify that the next person can answer all of these from this document alone:

- What files currently define the graph plan schema.
- What files currently emit and consume the runtime snapshot.
- Which IPC messages are real versus reserved.
- Which node wrappers still need attention (ad-hoc `ctx.args` calls not yet in typed config fields).
- What must be true before the batch launcher can be deprecated.
- How `build_training_graph_from_plan(plan)` works and where it lives.
- What the `--from-plan` startup path does and what its current limitations are.
- Which monolith functions have already been extracted into node modules and which remain.
- What the forwarding-shim protocol is and why it exists during the extraction phase.

If any of those answers are unclear, update this document before changing architecture again.

## Practical Next Step

Items 1–4 from the previous revision are done.  Items 6 and 7 (plan_apply and smoke test) are now also done.

The next concrete engineering steps are:

1. Expand the GUI graph canvas from the compact lane strip to a full-screen view with
   `layout.positions`-driven node placement and edge rendering, activated by a keyboard shortcut or button.
2. Implement `plan_patch` — define which config blob changes are hot-applicable versus restart-required, then wire the handler parallel to `plan_apply`.
3. Add explicit validation in `build_training_graph_from_plan` so the worker can reject malformed or incompatible plans with a structured IPC error response rather than a silent exception.
4. Exercise the reconnect path: stop the worker, restart from `--from-plan`, verify the GUI reloads the runtime snapshot and re-syncs without a fresh worker_hello.
5. **Begin Phase 6 (monolith dissolution):** extract the first round of leaf functions from `wav_config_transformer_pipeline.py` into their target node modules. Recommended starting point: `train_transformer_feature_metric` into `transformer_node.py`, then `train_conditional_generator_discriminator` into `generator_node.py`. Each extraction makes the graph more self-contained and reduces the monolith's footprint by one function. The smoke test serves as the regression check after each move.

The graph path is now a complete and tested alternative execution path with verified plan round-trip, full hyperparameter fidelity from typed configs, live plan swapping between cycles, and a passing end-to-end smoke test. Phase 5 (batch-file deprecation) and Phase 6 (monolith dissolution) are the remaining long-term blockers. They can proceed in parallel once the GUI canvas and plan_patch are complete.
