# Independent table execution mode

Status: experimental native implementation, 2026-11-12. The scheduled
scheduler is restored to the faster persistent-worker, worker-side dependency
release path. Mode 3 has independent persistent table threads and atomic
channel-frontier backpressure, but its complete KPN qualification remains open.

## Contract

`Scheduled` continues to accept a tick request and complete all selected
tables with same-tick dependency ordering. It retains the persistent-worker
debug toggle. The current `FreeSpinning` name must not be presented as a
working independent mode: it still calls `run_scheduled_tick` for each
submitted tick.

An independent mode is a separate runtime contract. Each enabled table is a
process whose rows execute sequentially on one persistent worker at a time.
After start, the worker loops without a per-iteration manager handoff or
ready-set update. Disconnected tables do not share an all-tables schedule or
wait for a whole-graph tick. A radio source table with no incoming table
dependency runs independently of an unrelated shader or simulation subgraph:
hardware input enters the table and its output goes to broadcast.

A table can wait for required input tokens, output capacity, pause, or
shutdown. Those are process and channel conditions, not a global tick
barrier or manager dispatch decision. A table with no required input may run
at its configured cadence; it must not busy spin merely because a consumer
is slow. A table is in flight on at most one worker, so a slow row never
overlaps its own next invocation.

The existing table edge FIFOs carry ordered payloads. Independent mode needs
lossless behavior for edges declared as KPN channels: a full channel applies
backpressure instead of dropping a token. The producer and consumer use the
FIFO's atomic cursors directly, without manager participation, a channel
mutex, or a reader-registry update on each transfer. A worker may wait for
capacity or a declared input using the channel state. Existing bounded/drop
policies remain available to expressly lossy streams, but they do not make a
lossless KPN edge. Cycles require seed tokens or another declared source of
progress.

Each worker publishes completion/telemetry into its own FIFO. The manager
greedily collects these records and exposes an ordered result pool to its
observers. The manager does not make a table runnable after each payload
transfer. A component that can continue running does so while another
component waits. Result collation may lag table execution within a bounded
observability budget, but must not consume output capacity or pause producers.
Payload channel ordering is independent of telemetry collection ordering.

The manager owns graph versions. A graph edit requests quiescence, waits for
the relevant table workers to leave their current row sequence, applies the
change once, then resumes with the new version. Requests using a stale graph
version are rejected. Pause, shutdown, and service lease expiry wake every
worker blocked on input or output capacity and join workers before table or
tool storage is released. No callback or table pointer from a tick-local
request may outlive its request without an owned snapshot.

## Implementation seams in current source

- `ThreadManager::run_loop` starts mode 3 workers from an owned request
  snapshot after the first submission. Workers continue without another
  submitted tick and stop quiescently on a changed request or pending graph op.
- `execute_module_iteration` is the extracted row executor shared by scheduled
  and independent modes; the row interpreter has not been copied.
- The persistent pool currently wakes on a generation for each tick and the
  manager waits for all pool workers before the local task dies. Independent
  dispatch needs durable table work records and no tick-local callback capture.
- A mode 3 worker checks the atomic frontier of each declared input and
  lossless output before invoking its rows. A full output pauses that worker
  before it can produce another value. An unrelated table keeps running.
  This covers row contacts resolved from graph edges or declared boundary
  ports. The existing row executor still ignores an unexpected failed
  publish; dynamic changes outside the quiescent graph-edit path remain
  unsupported and must not be described as lossless.
- FIFO host storage is pinned after spec-first allocation. Its CPU copy hooks
  use the fixed address without a copy mutex; reconfiguration allocates a new
  ring after quiescence. `EdgeReaderRegistry` cursor updates are atomic.
  Registration and removal still use their management mutex outside normal
  transfers. This lockless claim is limited to ordinary fixed host rings with
  dirty-grid and sparse-delta options disabled. Other storage backends and
  optional dirty-grid/sparse paths retain their existing copy/metadata locks.
- A worker currently yields while an input is absent or output is full. An
  atomic channel wait with shutdown notification is still needed to avoid
  idle CPU use without adding transfer locks or manager handoffs.
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

The current native tests prove independent progress, ordered transfer over a
two-slot lossless edge, producer backpressure under a slow consumer, and an
unrelated table's progress. Direct concurrent FIFO transport delivered 4,096
ordered tokens with full-buffer refusals and retries. Fan-out, seeded cycles,
dynamic edits, and a blocked-worker shutdown under an atomic wait remain to
be qualified before a production KPN claim.

The 2026-11-12 user-filespace CPU profile compares independent mode with
max-rate synchronous scheduled ticks, three alternating 500 ms repetitions.
Median row runs/s: one table at 0/100/1000/5000 us work was 7,944,520 /
9,989 / 999 / 201 in mode 3, versus 524,133 / 8,974 / 985 / 199 scheduled.
Eight independent tables at the same work sizes gave 1,409,020 / 79,381 /
7,995 / 1,596 versus 236,460 / 51,709 / 7,529 / 1,579. Those profile
graphs have no edges; the connected edge test proves backpressure behavior,
not transfer throughput. Raw profile receipts live in the user build area at
`C:/Users/alber/nodus-build/profile/free_mode_profile_results.csv`.
