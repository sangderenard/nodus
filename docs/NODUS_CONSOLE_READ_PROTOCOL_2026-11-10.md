# Nodus console read protocol (2026-11-10)

The canvas host `tools/run_canvas.py` serves a READ-ONLY console on 127.0.0.1 when started with `--console-port N` (0 = ephemeral, reported as `console_port` in the first stdout JSON line) and `--console-token-file FILE` (first line = shared secret). The console never writes to the canvas, the graph, or the file system outside its private temp directory.

Server: `ConsoleServer` in `tools/run_canvas.py`. Document builders: `nodus_graph/console_views.py`. Client: `nodus_graph/console_client.py`. Tests: `tests/test_console_read_protocol.py`.

## Threading rule (unchanged)

Client threads never touch the `Graph`. A request marks which views it wants; the stepping thread builds them in `ConsoleServer.service()` between ticks, together with the canvas bytes, so every view of one build is consistent with one revision. Results are coalesced for 100 ms; a request that needs a view the cached build lacks forces a new build. Only SNAPSHOT/DIGEST content costs the tick anything unless GRAPH/TOOLS/ACTIVITY are asked for. If a build fails the verb answers `REFUSED unavailable`.

## Protocol v1 (unchanged)

Newline-delimited UTF-8, lines at most 4096 bytes.

| client sends | server answers |
|---|---|
| `HELLO nodus-console/1 <token>` | `OK console <canvas-basename> read-only`; otherwise `REFUSED bad hello` or `REFUSED bad token` and the connection closes. Nothing else is accepted before a good HELLO. |
| `SNAPSHOT` | `CANVAS <revision> <n>\n` + n canvas bytes + `\nSTATE <json>\n` |
| anything else | `REFUSED read-only` (connection stays open) |

Up to 8 clients; the 9th gets `REFUSED too many clients`. Idle connections close after 60 s. `revision` increments each time a build's canvas bytes differ from the previous build's.

## Protocol 2 verbs

Authenticated clients may send `DIGEST`, `GRAPH`, `TOOLS`, `ACTIVITY` (exact upper-case, no arguments). Answer framing:

```
JSON <VERB> <byte-length>\n
<exactly byte-length bytes of UTF-8 JSON, compact separators, sorted keys, no trailing newline>
```

`<VERB>` echoes the request. Unknown verbs (including lower-case) still answer `REFUSED read-only`. `REFUSED unavailable` means the host could not build that view. v1 clients are unaffected.

### DIGEST

`{"protocol":2,"canvas":<basename>,"revision":int,"sha256":hex,"bytes":int,"modules":int,"rows":int,"edges":int,"contract_ports":int,"compile_hosts":int,"tick":int,"rate":float,"started":unix seconds float}`

- `revision`, `sha256`, `bytes` describe exactly the canvas bytes a SNAPSHOT of the same build sends. Poll `sha256`/`revision` to learn whether the graph changed.
- `tick` = ticks stepped by this host at build time; `rate` = `--rate`; `started` = host start time.
- `compile_hosts` = hosts attached by `--compile-host`.
- Nothing is null.

### GRAPH

Returns graph-definition v1 (`GraphDefinition.to_mapping()`, `nodus_graph/model.py`, schema `nodus.graph-definition.v1`) with these ADDITIVE keys; no v1 key is renamed or removed:

- top level: `revision`, `contract_ports` `[{"prefix","capacity","edge","listen_port","loan_ticks"}]`. `edge` is the native edge index (same index as `edges[].index`). `loan_ticks` is null if the process reports 0 (pair not acquired in this process).
- v1 top-level keys kept: `schema`, `scheduler`, `modules`, `edges`, `connectors`, `inputs`, `outputs` (the named top-level ports; `inputs`/`outputs` replace a single `ports` list).
- each module: v1 `stable_id` (the stable identifier; a 64-bit integer, larger than 2^53 in practice: JavaScript clients must parse it as a BigInt or string-preserving parser), `label` (the name), `x`,`y` (saved position), `width`,`height`, `contract`, `rows`; added `index` (native module index), `exec_mode` (ALWAYS null: no per-module execution mode is exposed; the per-module stack currency is `contract`), `contacts` `[{"index","ordinal","direction","name"}]`. `index` is the native contact index (inputs first, then outputs); `ordinal` is the v1 `ConnectorRef` ordinal. `name` comes from logical connector metadata, which exists only on a Graph built through the Python API, so it is null for a canvas loaded from a file.
- each row: v1 mapping (`kind` = input|output|builtin|plugin; `tool` for builtin, `plugin_id`,`consumes`,`produces`,`contract` for plugin, `count` for input/output) plus `index`. The tool name of a row is `tool` (builtin) or `plugin_id` (plugin).
- each edge: v1 `source`,`target` (ConnectorRef: `module_id` = module `stable_id`, `direction`, `ordinal`), `fifo` (null or `token_bytes`,`slots`,`top_k`,`type_id`,`layout`,`dtype`,`storage`), `type_id`; added `index`, `token_bytes`, `slots` (copies of the fifo values, null when the edge has no FIFO config).

Fallback shape. `Graph.definition()` is strict: it raises when an edge names a contact that no Input/Output row of its module declares (the saved messenger canvas used by the tests is such a canvas: plugin rows only). In that case GRAPH returns the same schema string with an extra `definition_error` (the reason), modules/rows/contacts in the shape above, and edges and ports whose endpoints are native indices instead of stable ids: `source`/`target` = `{"module_index","contact_index"}`, port `connector` likewise. A client tells the two apart by the presence of `definition_error`. `connectors` is `[]` in the fallback.

### TOOLS

`{"tools":[{"name","plugin_id","source","description"}],"complete":false,"missing":"<why>"}` sorted by name.

- `source` `builtin`: every `BuiltinTool` (except NONE); `plugin_id` null.
- `source` `plugin`: the acquired contract pairs (`<prefix>.market`, `<prefix>.worker`) and every plugin id bound on a row of this canvas.
- `source` `library` is part of the contract but is never produced (see below).
- `description` is always null: no tool metadata is exposed.
- `complete` is always false.

### ACTIVITY

`{"tick":int,"edges":[{"index","fill","seq"}],"contract_ports":[{"prefix","pending","engines","hosts"}],"compile_hosts":[{"prefix","host_id","accepted","completed","refused","acknowledged","in_flight"}]}`

- `edges[].fill` = unread samples (`nodus_headless_get_edge_unread`); null when the edge has no FIFO config. `seq` is ALWAYS null (no FIFO sequence query in the ABI).
- `contract_ports[].engines`/`hosts` = connected peer counts from `nodus_headless_get_contract_port`; null if the port cannot be read. `pending` is ALWAYS null (no pending-contract query in the ABI).
- `compile_hosts`: one entry per attached host (`compile_host_stats`); a host that cannot be read is omitted.

## Client

```python
from nodus_graph.console_client import ConsoleClient, ConsoleError
with ConsoleClient("127.0.0.1", port, token, timeout=10.0) as c:   # hello() on enter
    c.digest(); c.graph(); c.tools(); c.activity()
    c.snapshot()   # {"revision", "canvas": bytes, "state": dict}
```

Stdlib only. `hello()` (re)connects and authenticates. A REFUSED answer, a closed connection, a timeout or a malformed header raises `ConsoleError`.

## What a client can and cannot learn

Cannot, without new C++:

- The full tool registry. The headless ABI has no `ToolRegistry` enumeration (no tool count/at function), so TOOLS lists only builtins, contract-pair rows and plugin ids already placed on the canvas. Plugins loaded in the process but unused, and the Turing module library (the host has no `accounts.db` path; `nodus_graph/lowering_library.py` reads it on the server side), are missing; `source: "library"` is never emitted. Tool descriptions are not exposed.
- Per-module execution mode (`exec_mode` is null).
- FIFO sequence numbers (`edges[].seq`) and the number of pending contracts in a negotiator (`contract_ports[].pending`).
- Contact names for a canvas loaded from a file (names live in Python-side metadata, which a file does not carry).
- Strict stable-id edges for canvases whose edges name undeclared contacts (served by the native-index fallback above).
- The event queue (SNAPSHOT's `events` stays empty).
- Whether the canvas will be rewritten: the revision only reflects what the host would save now; autosave and the GUI trailer behaviour are not visible.

Can: everything the saved canvas holds (modules, rows, positions, edges and FIFO configs, ports, contract ports), the live tick, FIFO fill, peer counts, and compile-host counters, plus a cheap change detector (`DIGEST.sha256`).
