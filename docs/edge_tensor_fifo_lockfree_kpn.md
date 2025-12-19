# Lock-Free EdgeTensorFifo + KPN/SPN Scheduling Design (Draft)

This document captures the previously proposed design for a single-writer, spec-first, atomic FIFO (`EdgeTensorFifo`) and its intended integration with a future `ThreadManager` that can act as a KPN scheduler.

---

## 1) Design intent (confirmed constraints)

- Single-writer-per-edge (SW): each FIFO edge has at most one concurrent writer. Writers are trusted to be single-threaded per-edge (e.g., `gp_stage_stream_samples` or other module instance).
- Modules must declare tensor specs before any publish. `gp_table_edge_set_tensor_spec` (or during attach) is required prior to first publish. Reconfigure at runtime is disallowed unless you quiesce the edge.
- Multi-reader allowed (many readers per edge).
- No mutexes — fully lock-free per-edge.

## 2) Lock-free FIFO data structures (per-edge)

- Slot struct:
  - `struct Slot {`
    - `std::atomic<uint64_t> seq;` // monotonic sequence tag, initially = 0 or special
    - `alignas(64) float data[MAX_STRIDE];` // stride known from spec
  - `};`
  - `seq` holds the sequence number written into this slot (writer publishes by storing seq with release semantics).
  - Align slots to cache lines to avoid false sharing.
- Ring buffer:
  - `size_t slots;` // capacity (power-of-two optional)
  - `Slot* storage;` // preallocated array of `slots`
- Sequence counters:
  - `std::atomic<uint64_t> write_seq;` // global edge sequence, fetch_add for write reservation
  - Reader has `std::atomic<uint64_t> seq;` // next sequence expected to read
- Reader table:
  - Fixed-size array of `ReaderEntry { std::atomic<uint64_t> key; std::atomic<uint64_t> seq; }`
  - Reserve a `MAX_READERS` (configurable, e.g., 64). Subscribe uses CAS to claim an empty slot (`key==0` means free) and initializes `seq` to desired start (e.g., current `write_seq` if start-at-head semantics).
  - Lookup by scanning table; unsubscribe clears `key` via CAS.

### Push (writer path, single-writer)

- `seq = write_seq.fetch_add(1, memory_order_relaxed);`
- `slot_idx = seq % slots;`
- write payload into `storage[slot_idx].data` (non-atomic writes).
- `storage[slot_idx].seq.store(seq + 1, memory_order_release);` // publish: use `seq+1` to treat 0 as empty
- Optional: detect overflow by checking min reader seq (scan reader table) and if `seq - min_seq >= slots` then drop or advance min readers (policy).

### Pop (reader path, multi-reader)

- read `rseq = reader.seq.load(memory_order_relaxed);`
- `slot_idx = rseq % slots;`
- `observed = storage[slot_idx].seq.load(memory_order_acquire);`
- if `observed == rseq + 1` then payload is valid; copy data, then `reader.seq.fetch_add(1, memory_order_relaxed)` (or store `rseq+1`).
- else no data available (return 0).

### Unread

- compute `write_seq.load(acquire) - reader.seq.load(relaxed)`.
- Clamp to int32 limits.

### Memory ordering summary

- Writer publishes payload with a release store to slot.seq.
- Reader uses acquire load on slot.seq before copying payload.
- `write_seq` allocated via fetch_add; readers use atomic loads to compute unread.

## 3) Reader registration semantics

- Start-of-subscription choice (forced-spec + KPN scheduling):
  - Initialize reader.seq to current write_seq (start at head) by default.
  - For integrator/backlog, either subscribe earlier, or provide an option to initialize to `write_seq - 1` to read the last sample — but default is start-at-head.
- Reader table is centralized so a `ThreadManager` can pre-allocate/claim entries on behalf of readers; it may keep reader metadata (affinity, scheduling).

## 4) Handling tensor spec and reconfigure

- Enforce "spec-first": `gp_table_edge_publish` checks a configured flag and fails if not set.
- `gp_table_edge_set_tensor_spec` must be called before any publish. Reconfigure requires quiescence:
  - Caller must ensure no active writer/readers, or use a future `gp_table_edge_reconfigure_quiescent(edge_idx, new_spec)` that performs a versioned swap (defer).
- Implementation:
  - Allocate storage at set_spec time (slots * stride).
  - Disallow spec changes if active (`write_seq != 0` or reader entries non-empty), return failure.

## 5) Integration with ThreadManager (reader table / scheduling)

- Centralized reader table can live inside the ThreadManager (or be attached to canvas/table context but managed by manager).
- ThreadManager responsibilities:
  - Maintain global ReaderEntry pool and map `subscriber_key → entry index`.
  - Provide atomic snapshot of min reader seq for each edge (for overflow detection).
  - Offer APIs:
    - `register_reader(edge_idx, subscriber_key, start_pos) → entry_id`
    - `unregister_reader(entry_id)`
    - `poll_unread(entry_id)` / `fetch(entry_id)`
  - Optionally implement per-edge scheduling (KPN): schedule reads, control priority, or impose synchronicity policies.
- Benefit:
  - ThreadManager can ensure reader entries are allocated without scanning vectors, and coordinate cross-edge scheduling for KPN semantics.

## 6) KPN / SPN elaboration

### KPN basics

- Kahn Process Networks are deterministic dataflow models where processes (nodes) communicate via unbounded FIFO channels; blocking reads and non-blocking writes give deterministic behavior independent of execution order.
- For bounded FIFOs (slots), determinism depends on scheduling and capacity.

### In our system

- Each module is a process producing/consuming samples on edges.
- With SW + lock-free FIFOs we implement bounded FIFO channels. Determinism requires careful scheduling.

### Practical approach

- Implement lock-free FIFOs with bounded capacity and provide two modes:
  - Free-spinning mode: writers publish whenever; readers read whenever; host scheduling determines semantics (non-deterministic unless coordinated).
  - Scheduled KPN mode: ThreadManager runs a schedule (static or dynamic) that runs producers/consumers in a deterministic order each tick (e.g., tick: enable captures → publish → subscribe/consume), yielding deterministic behavior despite bounded buffers.
- Backpressure primitive:
  - If FIFO full, `push()` returns dropped/failure; scheduler can retry next tick.

## 7) API changes & header notes

- `gp_table_edge_set_tensor_spec` must be called before first publish; set `configured = true` and allocate internal buffers.
- Add new return codes to `gp_table_edge_publish` to report `PUBLISHED`, `DROPPED_FULL`, `NOT_CONFIGURED`.
- Add `gp_table_edge_subscribe_with_policy(edge, key, start_at_head)` for subscription start semantics.
- Add `gp_table_edge_max_readers(edge)` and `gp_table_edge_reader_capacity(edge)` for introspection.

## 8) Tests & validation plan

- Unit tests / harness:
  1. Single-writer, single-reader: push N; read N; validate order and contents.
  2. Single-writer, multiple-readers: different start positions; unread counts consistent.
  3. Overflow behavior: writer wraps; readers don’t read stale/overwritten data; drop policy validated.
  4. Subscription race: subscribe before/after publish; semantics validated.
- Stress test:
  - One writer thread, several reader threads at random rates; assert no UB; check consistency where applicable.
- Optional:
  - Use ThreadSanitizer (clang/gcc) during development.

## 9) Security/perf notes

- Use 64-bit sequence counters to avoid wrap-around.
- Prefer power-of-two slot count so modulo becomes bitmask.
- Cache-line alignment for slots and reader entries.
- Minimize scans: manager can maintain per-edge atomic min reader seq, or accept occasional scan cost.

