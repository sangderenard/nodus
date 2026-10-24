# Nodus as an LLM entity environment

Status: architectural research report and implementation plan. This document does not claim that the proposed entity API, episode store, or trainer already exists.

## Executive conclusion

Nodus can become a provider-neutral environment in which an LLM-backed entity can inspect, operate, and learn to orchestrate a real computational system. The important opportunity is not merely to expose Nodus tools to a chat client. It is to make the Nodus state transition itself the authoritative training environment:

```text
provider/model adapter
        |
        v
entity session -> observation -> bounded action -> Nodus transition -> receipt
        ^                                                   |
        |                                                   v
        +--------------- reward and next observation <------+
```

This architecture would support interactive use, deterministic evaluation, trajectory collection, verifier training, and eventually specialization-layer training from the same contract. It would also allow different model vendors and local models to inhabit the same environment without giving any provider ownership of Nodus state.

Here, **entity** has a stronger meaning than an API client or security principal. It is a first-class stateful inhabitant of the graph: the same player may be controlled by a person, an LLM, a script, or a replay; it has an independently persisted location; and a sprite or other presentation appears wherever that location projects into the visible world. It can relocate through the graph as an act of agency rather than only riding a process edge with a datum. The API session and capability grant control the entity, but they are not the entity itself. The companion design is [`GRAPH_ENTITY_AND_SPRITE_CONTRACT.md`](GRAPH_ENTITY_AND_SPRITE_CONTRACT.md).

The separation of responsibilities is decisive:

- **Nodus owns state, execution, authorization, artifacts, and receipts.**
- **An entity adapter translates between a model's structured output and Nodus actions.**
- **A protocol such as MCP can transport observations and actions, but does not define their truth.**
- **A training system consumes immutable episode records and returns policies, adapters, or verifiers.**
- **The existing graph, table, stack, tool, package, and edge mechanisms remain authoritative.** This proposal does not require a new kind of graph edge.

The first useful product is therefore not an autonomous agent and not a training loop. It is a small, deterministic, headless `reset/observe/step` environment with capability-scoped actions and complete receipts.

## What Nodus already provides

The repository contains more of the environment substrate than a generic agent framework normally starts with.

### Tools are typed executable objects

[`include/tool_ir.h`](../include/tool_ir.h) defines `ToolIR`, including lifecycle callbacks, port descriptions, stack execution, and serialization. [`include/tool_api.h`](../include/tool_api.h) defines `ITool`, `ToolStackContext`, typed raw-stack helpers, tool capabilities, and explicit argument/internal/return port classes. [`include/tool_registry.h`](../include/tool_registry.h) provides stable registry lookup, construction, source paths, and alias resolution.

This means an agent action need not be an unstructured shell command. A tool invocation can name a registered tool, declare typed inputs, and receive typed outputs. That is the natural action vocabulary for a Nodus environment.

### Source and binary provenance already have a home

[`include/repo_package.h`](../include/repo_package.h) models a repository package as a named, listable, unloadable unit containing source or prebuilt tools and their contracts. Capability tags distinguish decomposable ISA-backed tools, opaque binaries, and tools with backward paths. The package and plugin paths preserve the distinction between source, generated artifacts, and loaded binaries instead of pretending all tools are equally portable.

The shader artifact path goes further: [`docs/TURING_SHADER_ARTIFACTS.md`](TURING_SHADER_ARTIFACTS.md) describes a receipt containing hashes, provenance, command arguments, exit codes, stdout, and stderr. That receipt pattern should be generalized for entity actions rather than replaced by a model-specific trace format.

### Headless graph operation already exists

[`include/nodus_headless_abi.h`](../include/nodus_headless_abi.h) exposes an in-process, renderer-free graph surface. It can load and save the existing canvas format, construct modules and edges, bind tools, declare typed external ports, push and pull tokens, inspect stack snapshots, run the real `ThreadManager` scheduler, and poll events.

Two limitations in that header are especially important to preserve in the environment contract:

1. Present quiescence means no unread samples at the declared black-box boundary; it is not yet a proof that every internal process is idle.
2. Edge FIFO configuration and live samples are not included in the canvas save format and must be reconstructed after a load.

An evaluator must not award completion merely because the current headless quiescence flag is true. It must use task-specific terminal predicates.

### Runtime snapshots exist at a narrower boundary

[`include/nodus_runtime_abi.h`](../include/nodus_runtime_abi.h) exposes snapshot sizing, filling, and restoration for the edge runtime. Tool serialization and canvas persistence cover other pieces of state. These are useful ingredients, but they are not yet a whole-environment checkpoint. A real episode snapshot must explicitly inventory what is and is not captured.

### Scheduling and observable outputs are real

The headless ABI drives the existing scheduler rather than a surrogate loop. Named external ports, typed stack snapshots, tool registry contents, package adoption, and event polling provide observable consequences that programmatic graders can check. That is the central advantage over a synthetic text-only agent benchmark.

## The proposed entity contract

An **entity** is a process-independent, stateful graph inhabitant. A **controller session** is an identity authorized to observe a projection of Nodus and submit constrained actions on behalf of one or more entities. A person, LLM, fixed policy, test harness, or replay file may drive the controller session. Changing the controller does not replace the entity, erase its state, or move it.

This yields two related identities that must not be collapsed:

- `entity_id`: durable in-world identity, state, location, presentation, and lifecycle;
- `session_id` / `controller_id`: temporary authority, budgets, observation projection, and model/provider metadata.

Entity movement is a control-plane world transition recorded by Nodus. It is not a token traversing an edge, does not require inventing an entity edge type, and does not mutate graph topology. An authorized relocation may follow topology, jump to another known anchor, or move in world coordinates according to the scenario's movement policy.

### Session identity

Every session should bind:

- `session_id` and `entity_id`;
- scenario and immutable scenario version;
- Nodus build/revision and enabled package hashes;
- random seed and deterministic-mode flags;
- capability grant;
- resource budget: wall time, scheduler ticks, actions, bytes, tool invocations, and optional monetary/token accounting supplied by the adapter;
- workspace or graph lease;
- parent session and checkpoint when forked;
- observation schema and action schema versions.

It should additionally name the controlled entity or entities. Every entity record should bind its durable id, lifecycle revision, state-object reference, graph/world location, presentation descriptor, controller lease, and last committed action.

The capability grant is not a prompt. It is enforced by the environment before an action can reach Nodus.

### Environment lifecycle

The public semantic shape should be familiar to environment tooling without copying Python-specific implementation details:

```text
reset(scenario, seed, grant) -> observation, info
observe(projection?)         -> observation
step(action)                 -> observation, reward_vector,
                                terminated, truncated, info
checkpoint()                 -> checkpoint_ref
restore(checkpoint_ref)      -> observation
fork(checkpoint_ref, grant)  -> child_session
close()                      -> final_receipt
```

Gymnasium's `Env` API usefully distinguishes **terminated** (the task reached a defined terminal state) from **truncated** (a time or resource boundary stopped the episode). Nodus should retain that distinction. A compile failure, permission refusal, exhausted tick budget, and successful graph result are not interchangeable endings.

`step` should be atomic at the environment level: either it yields one recorded transition receipt, or it yields a recorded refusal/failure. It does not imply that every underlying Nodus operation is transactionally reversible. Actions whose side effects cannot be rolled back must declare that property and be forbidden in replayable training tiers.

### Observation schema

Observations should be structured, versioned, and deliberately projected. A useful initial observation contains:

- session identity, tick, budget remaining, and capability grant;
- graph identity and topology summary using stable module/port UUIDs;
- controlled entity identity, state projection, location anchor, world transform, and visible nearby entities;
- module rows and tool bindings;
- declared external ports, FIFO metadata, unread counts, and drop counters;
- typed stack snapshot metadata and values where authorized;
- tool registry entries, source paths, capability tags, and package identity;
- scheduler mode and events since the previous observation;
- artifact references and hashes, not arbitrary binary payloads inline;
- task state and programmatic predicate results;
- prior action receipt and normalized diagnostics.

The observation must say when data is absent because it is unsupported, withheld by policy, or simply empty. Those cases must not collapse to null.

Large observations should be addressable resources. The agent can receive a compact index and request bounded slices. This maps cleanly to MCP resources, a local C ABI, or another transport without changing the canonical schema.

### Action schema

The first action vocabulary should wrap existing sanctioned operations:

| Family | Examples | Initial tier |
|---|---|---|
| Inspect | list tools, inspect module, read stack snapshot, read artifact receipt | read-only |
| Execute | step scheduler, run to bounded quiescence, invoke a registered row tool | sandbox |
| Embodiment | move entity, inspect location, interact at an anchor, change approved presentation state | sandbox |
| Graph authoring | add module, bind tool, set rows, add/configure existing edge type, declare port | sandbox |
| Data | push typed token, pull typed token | sandbox |
| Persistence | save canvas, create environment checkpoint, restore checkpoint | sandbox |
| Packages | inspect manifest, stage adoption, load/unload approved package | privileged |
| Operations | acquire/release lease, allocate worker budget, schedule bounded run | privileged |

Each action needs an idempotency key, expected pre-state revision, declared target, arguments, and requested budget. The result needs a status, before/after state revisions, typed outputs, emitted artifacts, diagnostics, measured resource use, and hashes.

Free-form source generation is not a primitive action. It is an artifact proposal followed by validation, compilation through an existing sanctioned path, and explicit adoption. This keeps generated text from bypassing Nodus's tool and package boundaries.

## Reward is a verified state judgment

The strongest part of the idea is that Nodus can reward orchestration from real programmatic state rather than a language model's self-report.

Reward should remain a vector in the stored episode even if a trainer later scalarizes it:

```text
reward = {
  authorization,
  correctness,
  provenance,
  safety,
  latency,
  throughput,
  memory,
  load_balance,
  makespan,
  recovery,
  action_efficiency,
  artifact_reuse
}
```

The first four dimensions are hard gates. An unauthorized action, incorrect result, missing provenance, or unsafe state transition cannot be compensated by speed. Only after gates pass should performance and efficiency be compared.

Examples of directly measurable orchestration objectives include:

- completion of all dependency-ready work;
- total makespan and critical-path delay;
- worker idle time and queue age;
- peak memory and live artifact footprint;
- redundant compilation or repeated tool calls;
- correct reuse of a content-addressed artifact;
- successful rollback or recovery after injected failure;
- absence of dropped FIFO samples where loss is forbidden;
- meeting latency or throughput service levels;
- final graph outputs matching exact or domain-approved numerical predicates.
- entity reaching the required graph location independently of whether data traversed an edge;
- valid interaction between entity state, game/world state, and execution outputs.

Every component must include its evidence. For example, `correctness: 1` should point at the output hashes and predicate results; `load_balance: 0.74` should point at the worker utilization series and formula version.

### Reward-hacking controls

Programmatic does not automatically mean sound. A policy may learn to stop early, exploit incomplete quiescence, avoid difficult work, spam cheap actions, or mutate the grader's inputs. Controls should include:

- immutable scenario definitions and read-only grader state;
- independent terminal predicates;
- held-out task and workload families;
- action and resource costs;
- explicit truncation outcomes;
- invariant checks before performance scoring;
- counterfactual replay from checkpoints;
- differential evaluation against baseline policies;
- adversarial fault injection;
- receipt validation by an independent verifier process.

PROVE's use of live, session-isolated MCP servers, state-grounded task synthesis, and a multi-component reward with an efficiency penalty is directly relevant. Nodus can improve on the general pattern by grounding rewards in its own typed runtime and artifact receipts rather than treating remote tool calls as the final authority.

## Multi-entity semantics

Nodus can host a coordinator entity and multiple worker entities, but concurrency must be represented unambiguously.

PettingZoo's Agent Environment Cycle is a useful warning: many multi-agent bugs arise when an API hides whose turn produced which state transition. Nodus should record an ordered commit log even if work executes concurrently. Each accepted action receives a monotonic transition sequence and names the entity, lease, pre-state revision, and affected resources.

Two modes are sufficient initially:

1. **Sequential/AEC mode:** one entity is eligible to commit an action at a time. Best for debugging, evaluation, and early training.
2. **Parallel-proposal mode:** several entities propose against one revision; the environment admits a non-conflicting set, records ordering, and returns explicit conflict/refusal receipts.

A coordinator is not a message queue. It is an entity with authority to assign leases, set budgets, inspect receipts, accept or reject deliverables, and decide when work is ready for integration. Worker entities cannot silently broaden their own grants.

## MCP and other transports

MCP defines three server primitives: prompts, resources, and tools. It is a good optional boundary for external models:

- Nodus observations and artifact receipts become **resources**.
- Bounded entity actions become **tools**.
- Scenario or role starters may be **prompts**.

MCP should not own session truth, capability decisions, leases, reward, or replay. A native embedding, command-line client, HTTP service, or future protocol should be able to drive the same entity API. This is what makes the environment provider-neutral.

## Episode and trajectory record

An episode should be an append-only manifest plus content-addressed blobs. Every transition records:

- schema version, session, scenario, entity, and sequence;
- observation hash and projection policy;
- raw model proposal hash where retention policy permits;
- parsed action and validation result;
- capability decision;
- state revision before and after;
- Nodus events and artifact receipts;
- timing and resource counters;
- reward vector, formulas, and evidence references;
- termination/truncation state;
- build, package, and grader hashes.

This record supports exact replay when all involved actions are deterministic, and forensic comparison otherwise. Nondeterminism must be declared, not hidden: device, driver, scheduling mode, clocks, external services, and unseeded sources belong in the manifest.

## Security and deployment tiers

The environment should advance through explicit tiers:

1. **Observer:** read-only projections of an existing graph.
2. **Replay:** execute a recorded action sequence in an isolated fixture.
3. **Sandbox:** mutate an isolated graph and write only to a session artifact store.
4. **Shadow:** observe a live workload and propose actions that are scored but not committed.
5. **Supervised live:** approved low-risk actions with rollback and human gates.
6. **Delegated live:** bounded autonomous operations under leases and service limits.

Model text, tool output, package metadata, and repository contents are all untrusted inputs. The adapter must parse against closed schemas; the environment must enforce grants; the artifact layer must retain source and hashes; and live credentials must never appear in observations or training traces.

## Implementation plan

### Phase 0: contract and read-only observer

- Define versioned POD/JSON-equivalent schemas for entity identity, observation index, action, receipt, reward evidence, and episode manifest.
- Implement a read-only observation builder over the tool registry and headless ABI.
- Add conformance tests proving the same canonical observation through in-process and serialized adapters.
- Document unsupported state, especially incomplete whole-graph checkpointing.

Exit gate: a fixture graph produces a stable, hashable observation with no model involved.

### Phase 1: deterministic single-entity episodes

- Add `reset`, bounded `step`, terminal predicates, truncation reasons, and append-only receipts.
- Permit only inspect, token I/O, scheduler step, and bounded tool invocation.
- Build fixtures for successful completion, tool failure, budget exhaustion, FIFO overflow, and capability refusal.

Exit gate: a fixed policy can replay an episode and reproduce outputs and receipts on the supported deterministic configuration.

### Phase 2: sandbox authoring and checkpoints

- Admit graph construction through existing headless operations.
- Define a whole-environment checkpoint manifest composed of canvas state, edge runtime snapshots, serializable tool state, package set, and explicit omissions.
- Add fork/restore tests and artifact-store quotas.

Exit gate: a branch can be evaluated and discarded without altering its parent session.

### Phase 3: programmatic evaluation

- Add versioned predicates and reward evidence.
- Establish baselines: random valid policy, scripted policy, greedy scheduler, and human-authored orchestration.
- Add fault injection, hidden scenarios, and differential replay.

Exit gate: reward changes can be explained from receipts and survive independent recomputation.

### Phase 4: multiple entities and external adapters

- Add leases, sequential turn ownership, proposal conflicts, and coordinator grants.
- Implement one thin MCP adapter and one direct local adapter against the same environment interface.
- Validate that both produce identical canonical actions and receipts.

Exit gate: two providers can run the same evaluation without provider-specific Nodus semantics.

### Phase 5: specialization training

- Export de-identified episode datasets.
- Train action-format adapters, verifiers, and policies outside Nodus.
- Re-admit candidates through fixed held-out evaluation and shadow deployment.

Exit gate: a specialized policy beats scripted baselines on held-out scenarios without increasing gate violations.

## Acceptance suite

The first release should not be called an environment until it proves:

- schema round-trip and forward-compatible rejection behavior;
- stable observation hashing;
- enforced capability refusals;
- deterministic fixture replay;
- precise terminated versus truncated outcomes;
- budget enforcement;
- receipt completeness and tamper detection;
- checkpoint/fork isolation within the documented state boundary;
- no mutation by read-only entities;
- provider-adapter equivalence;
- reward recomputation from retained evidence;
- concurrent proposal conflict handling;
- secret and path redaction.

## Non-goals

- Inventing agent-specific graph edges.
- Making MCP the internal Nodus architecture.
- Letting an LLM bypass tool registration, package adoption, or artifact validation.
- Treating a text transcript as sufficient execution provenance.
- Training directly against a user's live graph before deterministic sandbox evaluation exists.
- Replacing Nodus scheduling with an agent-authored scheduler loop.
- Claiming current edge quiescence is whole-graph completion.

## Research basis

- Model Context Protocol, server primitives: <https://modelcontextprotocol.io/specification/draft/server/index>
- Gymnasium `Env`, including `reset`, `step`, termination, and truncation: <https://gymnasium.farama.org/api/env/>
- Terry et al., “PettingZoo: Gym for Multi-Agent Reinforcement Learning,” Agent Environment Cycle model: <https://arxiv.org/abs/2009.14471>
- Abdelaziz et al., “Synthesize and Reward—Reinforcement Learning for Multi-Step Tool Use in Live Environments” (PROVE): <https://arxiv.org/abs/2606.03892>
- Pan et al., “Training Software Engineering Agents and Verifiers with SWE-Gym”: <https://arxiv.org/abs/2412.21139>
