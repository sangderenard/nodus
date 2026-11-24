# One native Nodus host, with the rasterizer (plan, 2026-11-23)

Owner, 2026-11-23: "I want us moving fully out of run canvas and the headless run, nodus must run with the rasterizer
and fully native, and still have all features it has now."

This is the research and the plan.

**Status:**
- **Stage A is built** (2026-11-23): `frontend_shell --canvas`, in `tools/native_host.h`, tested by `tests/native_host_test.cpp` on Windows and on Oracle A1.
- **Stage B is built** (2026-11-23): the console server, `tools/native_console.h`.
  - Parity with `run_canvas.py` was checked byte for byte on three canvases: a meta-group canvas, a contract-port canvas, and a strict graph-definition canvas.
  - Only live values differed: ticks, rope ids, ephemeral ports, queue fill.
  - Two deliberate differences:
    1. **Link ACTIONs wait for stage C.** `link_host`, `link_dial`, `link_udp` and `link_send` answer an error until stage C, because in `run_canvas.py` they place the pre-split rows. `link_report` is served.
    2. **A race is fixed.** A view requested while another client's build was already running answered `REFUSED unavailable` in `run_canvas.py`. The native console asks again until a build has tried the view.
- **Stage C is built** (2026-11-23): links declared in the canvas.
  - **Hosted links are recorded:** a hosted link is a `LINK_HOST <id> <name> <port> <flags> <key file>` record. A load reads these records before the canvas body, so the link rows resolve; a save writes them. The key stays in its file. Port 0 lets the system pick.
  - **Route and metrics:** flag 1 registers `<id>.route`, the front route row (the host's `--link-route-map`); flag 2 registers `<id>.metrics`.
  - **Tables:** `nodus_headless_compose_link_tables` puts the four link rows in their own stack-loopback tables, beside the declared ports `<id>.send`, `<id>.received`, `<id>.control` and `<id>.control.received`, plus a metrics table and a route table when flagged.
  - **Console ACTIONs:** `link_host` opens a recorded link (`nodus_headless_open_link_host`, port 0 allowed, `"route": true`). `link_dial` and `link_udp` open a dialed link and compose or rebind its tables. A dial is not recorded: its tickets are short-lived secrets. `link_send` pushes into `<id>.send`.
  - **Tested by** `tests/link_canvas_test.cpp`: open, save, destroy, reload, then a tensor across the reloaded link, byte for byte.
  - **Not yet done:**
    - Sent tensors are not returned to a mint (no Returns row in the send table), as before.
    - Oracle's hand-made `/srv/svc/link-test` canvas becomes an ordinary canvas with a `LINK_HOST` record at the stage D cut-over.
- **Stage D is prepared, not switched** (2026-11-23).
  - **Rehearsal:** a copy of the live messenger canvas was hosted by `run_canvas.py` and by the native host side by side, with the live instance's arguments: the four acquires, the compile host, cron and the console. The contract ports were set to 0 in the copy and the jobs folders were private. Every view, ACTION reply and SNAPSHOT matched. The only differences were rope ids (random per process) and Python's race refusal.
  - **Supervisor:** `"native": true` in a nodus instance runs the native host directly, without Python, with `--offscreen` unless `"window": true`, and with `"link_route_map"`. This is on Turing branch `claude/native-nodus-host`, tested by `test_nodus_native_host_command`.
  - **Build products:** the build service products carry `frontend_shell` (driver version 2).

**Switched on 2026-11-23** (owner: "do as much of the switching over as you can do on your own"):
- **Oracle's link-test canvas runs on the native host.**
  - The canvas is `/srv/svc/link-test/link-test.nodus.txt`, with the record `LINK_HOST state state 47616 0 /srv/svc/link-test/nodus_link.key`. It is started by `/srv/svc/link-test/run_native.sh`, which runs `/srv/svc/nodus-native/66dcbb5a/frontend_shell --offscreen`.
  - The old canvas and `run_canvas.py` are no longer started there.
  - The cross-machine pairs pass through it: 311/311, 926/926 after a 25 s outage, 323/323 after rotation. Its restart brings 47616 back from the record alone.
- **The Windows product is built:** `C:/dev/Powershell/srv/state/products/nodus-native/66dcbb5a/` holds `frontend_shell.exe` and its DLLs. It was rehearsed on a copy of the live messenger canvas.
- **The messenger switch is a reviewed change** (Turing `claude/native-nodus-host`, `instances.json`). It takes effect when it lands, the live tree is updated, and the stack restarts. The restart is the owner's.

## Stage D: the switches (each the owner's)

1. **Land the reviews.** Nodus `claude/mesh-canvas-rows`, Turing `claude/native-nodus-host`, Turing `claude/link-route-map`.
2. **Rebuild the service build.** Build `frontend_shell` in `C:/dev/Powershell/nodus/build-service`: `cmake --build build-service --config Release --target nodus_headless frontend_shell`.
3. **Change `nodus-messenger`** in `turing/examples/meowdoku/deploy/instances.json`:
   - `"tool"`: `"C:/dev/Powershell/nodus/build-service/Release/frontend_shell.exe"`
   - add `"native": true`
   - the `"args"` stay as they are (`--compile-host`, `--compile-python`, `--cron`).

   Restart the stack with `deploy/Restart-Stack.ps1`. Then check:
   - the console on 47612;
   - `/api/nodus/*`;
   - the cron log, `srv/state/jobs/cron/cron.log`;
   - a compile through `gui.proof`.

   To roll back, revert the two keys and restart.
4. **`nodus-training`.** `train_aapl_continuous.py` drives its own Python training loop and imports run_canvas's `ConsoleServer` to serve a console beside it. It is not a canvas host that `--canvas` replaces. Moving it is its own piece of work (the loop itself), and it needs the owner's direction.
5. **Modal.** The subordinate scripts start `run_canvas.py` inside the Modal image. Moving them means rebuilding that image with the native host and running it on Modal, which is billed, so it waits for the owner's go.
6. **Oracle's link-test canvas.** Rewrite `/srv/svc/link-test` as an ordinary canvas: one `LINK_HOST state state 47616 0 /srv/svc/link-test/nodus_link.key` record, hosted by `frontend_shell --canvas ... --offscreen` from a build of this branch. Then the pause and resume scripts need no `link_host` ACTION.
7. **Remove `run_canvas.py`.** Once nothing launches it, delete `run_canvas.py` and `nodus_graph/console_views.py`, and take `run_canvas.py` out of the build products.

**The FRAME console verb is built** (2026-11-23). It answers `FRAME <revision> <width> <height> <n>` followed by the canvas's RGBA bytes, at the viewport's size, built between steps. The window's draw and FRAME share one raster lock. It is tested on Windows and on Oracle A1, which has no display: there the canvas is rasterized offscreen.

Not built yet: a Returns row on the link's send table.

## Where things are

There is one engine. The headless library (`nodus_headless`) and the windowed shell (`frontend_shell`) both run on
`canvas_tables_static` and `nodus_runtime`. They share the same GraphRuntime, ThreadManager and `gp_canvas_step`. What
differs is the host around the engine.

| Feature | `run_canvas.py` (Python, live) | `frontend_shell` (native, windowed) |
|---|---|---|
| Load/save every canvas-file record (EDGE_FIFO, PORT, MODULE_POLICY, MODULE_STACK_LOOPBACK, CONTRACT_PORT, `<canvas>.modules`) | yes, through `nodus_headless_load_file/save_file` | EDGE_FIFO and CONTRACT_PORT only; its 2 s autosave writes the canvas without any of them (`frontend_shell.cpp:1540-1545`) |
| Fixed-rate stepping, `--rate`, `--ticks` | yes | its own dt/ratio/free modes |
| `--acquire PREFIX:CAPACITY[:LOAN]` | yes | yes |
| Compile host (`--compile-host`, `--compile-python`, `--turing-root`, `--compile-adopt`) | yes | no |
| Cron engine (`--cron`) | yes | no |
| Links (host, wss/udp dialers, link table, metrics row, `link_send`) | console ACTIONs, composed in Python | no |
| Meta groups (`meta_group/report/destroy`, edges by module label) | console ACTIONs | no |
| Console server: HELLO v1/v2, SNAPSHOT, DIGEST, GRAPH, TOOLS, ACTIVITY, META, ACTION | yes, all in Python (`run_canvas.py:86-604`, `nodus_graph/console_views.py`) | client only (`--attach`, v1 SNAPSHOT) |
| Revision tracking, save after an ACTION that changed it | yes | no |
| Ready line on stdout (`{canvas, contract_ports, links, pid, console_port, cron}`), signals, exit codes | yes; the supervisor parses the line (`apache_supervisor.py:433-445`) | no |
| Rasterizer (CPU, `gp_canvas_raster_rgba`) and input | no | yes, in an SDL window |
| Runs with no display (Oracle A1: no GPU, no X) | yes | no: it exits without a window and GL context (`frontend_shell.cpp:2968-2984, 3042-3045`) |

Who runs `run_canvas.py` today:
- the Windows supervisor's `nodus-messenger` (console 47612, compile host, cron 7201, contract ports gui.proof, studio.product, cron, userlist);
- `train_aapl_continuous.py`, which imports its ConsoleServer;
- the Modal subordinate scripts (`subordinate_home_host.py`, `subordinate_modal.py`);
- Oracle's `/srv/svc/link-test`, which is set up by hand and not tracked;
- the build bundles (`nodus_build.py`, `build_linux.sh`) and about ten tests in nodus and turing.

Its clients (`nodus_service.py`, `demo_canvas_tables.py --attach-console`, `frontend_shell --attach`) speak only the
console protocol, so they keep working if a native server speaks it byte for byte.

Found on the way:
- `run_canvas.py`'s link table still places `<id>.bulk` and `<id>.control`. Those rows no longer exist since the
  four-row split (`<id>.control.send/.receive`, `<id>.bulk.send/.receive`) on 2026-11-21. Oracle's link-test canvas
  works only because it runs an older library.
- `docs/MESH_CANVAS_UP_PORTS_2026-11-21.md:105` says the link ends left `run_canvas.py`, but its `link_*` ACTIONs are
  still there.
- 47616 is both `nodus-training`'s console port (`instances.json:221`) and the link upstream in both Apache fronts.

## The plan

The native host is `frontend_shell`, grown into the only Nodus host (one binary; the name can follow later). Each
stage ends with every existing test of that feature passing against the native host, and with nothing removed until the
stage that replaces it is live.

**Stage A: the host core.**
- Load and save only through the headless library's file path, so every record and `<canvas>.modules` survive.
- The autosave writes them too.
- Command-line parity with `run_canvas.py`: `--canvas --rate --ticks --acquire --compile-host --compile-python
  --turing-root --module-library-root --compile-adopt --cron --console-port --console-token-file`, with the same
  checks.
- Startup order, ready line, signals (Ctrl+Break on Windows), shutdown order (cron, then compile hosts, then the
  graph) and exit codes, identical.
- The rasterizer always runs. With a display it draws to the window. With `--offscreen` (no SDL video, no GL) it
  draws into a frame in memory that the console serves (Stage B), so Oracle runs the same host.

**Stage B: the console server, native.**
- HELLO v1/v2 with the token, 8 clients, 60 s idle, 4096-byte lines.
- SNAPSHOT, and the DIGEST, GRAPH, TOOLS, ACTIVITY and META documents ported from `console_views.py`, compact
  sorted-key JSON.
- ACTION with `expected_revision`, the 5 s outcome window, and save after a changed revision.
- Every op: viewport, mouse, key, meta_*, link_*.
- One new read verb, FRAME, for the rasterized canvas, so a client without the engine sees what the window shows.
- The Python console tests run against the native host, unchanged.

**Stage C: links in the canvas file.**
- New records `LINK_HOST <id> <name> <key file> [route] [metrics]` and `LINK_DIAL ...`, restored at load the way
  CONTRACT_PORT is.
- The listener takes a system-picked port, and the route row (`nodus.link.route`) is registered with the link like
  its metrics row.
- Send and receive rows go in separate tables, as the four-row design requires.
- The console's `link_*` ACTIONs write these records instead of composing tables.
- `link_send` becomes a declared port push.
- Oracle's link-test canvas becomes an ordinary canvas file.

**Stage D: cut over.**
- The supervisor's `nodus-messenger` tool becomes the native host (same arguments, same ready line).
- Then the training host (its ConsoleServer import) and the Modal subordinate scripts.
- Then Oracle.
- The build bundles drop `run_canvas.py`.
- When nothing launches it, `run_canvas.py`, `console_views.py` and the `nodus_graph` host path are deleted.

Python that stays, outside the host:
- The compile host still runs Turing's Python compile driver; that is the compiler, not the host.
- The cron tasks are performed by the task-worker process (`cron_host.py`). It is a peer on the cron contract port,
  not part of Nodus.

Live restarts in Stage D are the owner's.
