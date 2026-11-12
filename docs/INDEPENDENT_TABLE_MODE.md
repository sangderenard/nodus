# Independent table execution mode

Status: design boundary, 2026-11-12. The scheduled scheduler is restored to
the faster persistent-worker, worker-side dependency release path. This file
does not claim independent execution is implemented.

## Contract

`Scheduled` continues to accept a tick request and complete all selected
tables with same-tick dependency ordering. It retains the persistent-worker
debug toggle. The current `FreeSpinning` name must not be presented as a
working independent mode: it still calls `run_scheduled_tick` for each
submitted tick.

An independent mode is a separate runtime contract. Each enabled table is a
process whose rows execute sequentially on one worker at a time. Workers
remain alive and take runnable tables from the manager's ready set; they need
not be permanently pinned one-to-one to tables. The manager can serialize
ready-set updates. It must not put disconnected subgraphs into a single
all-tables schedule or wait for a whole-graph tick before making another table
runnable. A radio source table with no incoming table dependency is runnable
independently of an unrelated shader or simulation subgraph: hardware input
enters the table and its output goes to broadcast.

A table can wait for required input tokens, output capacity, pause, or
shutdown. Those are process and channel conditions, not a global tick
barrier. A table with no required input may run at its configured cadence; it
must not busy spin merely because a consumer is slow. A table is in flight on
at most one worker, so a slow row never overlaps its own next invocation.

The existing table edge FIFOs carry ordered payloads. Independent mode needs
lossless behavior for edges declared as KPN channels: a full channel applies
backpressure instead of dropping a token. A producer waits for capacity, and
a consumer waits for its declared input without holding a manager or channel
lock. Existing bounded/drop policies remain available to expressly lossy
streams, but they do not make a lossless KPN edge. Cycles require seed tokens
or another declared source of progress.

Each worker publishes completion/telemetry into its own FIFO. The manager
greedily collects these records, updates which tables are runnable, and
exposes an ordered result pool to its observers. Channel publication makes a
waiting consumer eligible; it does not impose a whole-graph tick. A component
that can continue running does so while another component waits. Result
collation may lag table execution within a bounded observability budget, but
must not consume output capacity or pause producers. Payload channel ordering
is independent of telemetry collection ordering.

The manager owns graph versions. A graph edit requests quiescence, waits for
the relevant table workers to leave their current row sequence, applies the
change once, then resumes with the new version. Requests using a stale graph
version are rejected. Pause, shutdown, and service lease expiry wake every
worker blocked on input or output capacity and join workers before table or
tool storage is released. No callback or table pointer from a tick-local
request may outlive its request without an owned snapshot.

## Implementation seams in current source

- `ThreadManager::run_loop` currently waits for `TickRequest` and calls
  `run_scheduled_tick` for both `Scheduled` and `FreeSpinning`. An independent
  executor needs a separately owned graph snapshot and lifecycle.
- `run_scheduled_tick` contains the table row executor as a large local
  lambda, with request, graph, ledger, and canvas references. Extract that
  table execution once for reuse; do not duplicate the row interpreter.
- The persistent pool currently wakes on a generation for each tick and the
  manager waits for all pool workers before the local task dies. Independent
  dispatch needs durable table work records and no tick-local callback capture.
- Existing edge FIFO `top_k` policies can drop old tokens or reject writes.
  Specify and test the lossless channel behavior before claiming KPN.
- Canvas `FREE` UI and `--canvas-config=...:FREE` currently select different
  controls. The UI/ABI must distinguish independent tables from a canvas
  tick thread that runs freely relative to rendering.

## Qualification

Run a producer and consumer for a measured interval without another
`submit_tick`; both must progress. Prove independent tables make progress when
one table is blocked; verify FIFO order and fan-out; verify a seeded cycle;
measure throughput and end-to-end latency with real row work and queued
traffic, including manager collation lag; verify backpressure without loss;
pause, edit, resume, and shut down a graph with blocked workers. Compare
`Scheduled` to the restored baseline to ensure the new mode does not add its
handoff cost back into scheduled chains.
