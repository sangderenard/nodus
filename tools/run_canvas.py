"""Canvas host process: load a saved canvas and step it at a fixed rate.

Prints one JSON line describing the bound contract ports, steps until
SIGINT/SIGTERM or --ticks, then prints {"ticks": n}.  Errors go to stderr, exit 2.

The native library prints diagnostics on fd 1, so fd 1 is redirected to stderr and the
JSON lines are written to a saved duplicate of the real stdout.

Optional managed console (--console-port / --console-token-file), newline-delimited
UTF-8 over 127.0.0.1. Version 1 remains read-only:
  HELLO nodus-console/1 <token>  -> OK console <canvas-basename> read-only | REFUSED <reason> (closes)
  SNAPSHOT -> CANVAS <revision> <n>\\n<n bytes>\\nSTATE <json>\\n
  DIGEST | GRAPH | TOOLS | ACTIVITY -> JSON <VERB> <n>
<n bytes of compact sorted-key UTF-8 JSON>
  anything else -> REFUSED read-only
(docs/NODUS_CONSOLE_READ_PROTOCOL_2026-11-10.md)

Version 2 uses HELLO nodus-console/2 <token> and adds
ACTION <n>\n<n bytes of JSON>. Each action supplies expected_revision and one
canvas UI operation. The stepping thread applies it, saves changed canvas state,
then replies JSON ACTION <n>\n<n bytes of JSON>.

Nodus links (research case 98d49cb8) are opened on a running canvas by the same ACTION, never by how the process was
started:
  {"op": "link_host", "id", "prefix", "port", "key_file"}   accept links on 127.0.0.1:port for /ws/link/<prefix>/
  {"op": "link_dial", "id", "host", "port", "prefix", "control_ticket", "bulk_ticket"[, "secure", "pin"]}
        dial ws(s)://host:port/ws/link/<prefix>/ with the two issued tickets (base64; from POST /api/link/ticket/<prefix>)
  {"op": "link_report", "id"}                                the endpoint's state and counters
link_host and link_dial put the rows <id>.control and <id>.bulk on the canvas table "link <id>" (made once, kept with
the canvas), so the canvas polls the link every step.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hmac
import json
import math
import os
import pathlib
import re
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nodus_graph import Graph  # noqa: E402
from nodus_graph.native import CanvasModuleDesc, _NativeFifoConfig, _NativeRow  # noqa: E402
from nodus_graph import console_views  # noqa: E402

_stop = False

CONSOLE_MAX_CLIENTS = 8
CONSOLE_IDLE_SECONDS = 60.0
CONSOLE_COALESCE_SECONDS = 0.1
CONSOLE_STEP_WAIT_SECONDS = 5.0


def _on_signal(signum, frame) -> None:
    global _stop
    _stop = True


class ConsoleServer:
    """Managed canvas console.

    Client threads never touch the Graph: they request a snapshot and the stepping
    thread builds it in service().  Snapshots are shared and coalesced to 100 ms.
    """

    def __init__(self, token: str, canvas_name: str, ticks_fn, canvas_path: str) -> None:
        self._token = token.encode("utf-8")
        self._canvas_name = canvas_name
        self._canvas_path = canvas_path
        self._ticks_fn = ticks_fn
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._wanted = threading.Event()
        self._clients = 0
        self._closing = False
        self._snap = None  # (monotonic_time, canvas_bytes, state_json_bytes, revision, views)
        self._want_views = set()  # views clients asked for; built on the stepping thread
        self._actions = []  # (validated request, result box, completion event); stepping thread owns Graph
        self._rate = 0.0
        self._started = time.time()
        self._hosts = []  # [(prefix, host_id)], read on the stepping thread only
        self._generation = 0
        self._last_bytes = None
        self._revision = 0
        self._tmp = tempfile.mkdtemp(prefix="nodus_console_")
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self, port: int) -> int:
        self._listener.bind(("127.0.0.1", port))
        self._listener.listen(16)
        self._listener.settimeout(0.2)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        return self._listener.getsockname()[1]

    def set_context(self, rate: float, hosts: list) -> None:
        self._rate = rate
        self._hosts = hosts

    def close(self) -> None:
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        try:
            self._listener.close()
        except OSError:
            pass
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ---- stepping thread -------------------------------------------------
    def service(self, graph) -> None:
        if not self._wanted.is_set():
            return
        with self._lock:
            wanted = set(self._want_views)
            self._want_views -= wanted
            actions = self._actions
            self._actions = []
            self._wanted.clear()
        for request, box, done in actions:
            try:
                box.append(self._apply_action(graph, request))
            except (ValueError, RuntimeError) as exc:
                box.append({"error": str(exc)[:200]})
            finally:
                done.set()
        fresh = (not actions and self._snap is not None
                 and time.monotonic() - self._snap[0] < CONSOLE_COALESCE_SECONDS
                 and wanted <= set(self._snap[4]))
        if fresh:
            return
        try:
            snap = self._build(graph, wanted)
        except Exception as exc:  # noqa: BLE001
            print(f"run_canvas: console snapshot failed: {exc}", file=sys.stderr, flush=True)
            snap = None
        with self._cond:
            if snap is not None:
                self._snap = snap
            self._generation += 1
            self._cond.notify_all()

    def _apply_action(self, graph, request: dict) -> dict:
        if not isinstance(request, dict) or type(request.get("expected_revision")) is not int:
            raise ValueError("expected_revision is required")
        if request["expected_revision"] != self._revision:
            return {"error": "canvas revision changed", "revision": self._revision}
        op = request.get("op")
        if op == "viewport" and set(request) == {"op", "expected_revision", "width", "height"}:
            width, height = request["width"], request["height"]
            if type(width) is not int or type(height) is not int:
                raise ValueError("viewport width and height must be integers")
            graph.ui_viewport(width, height)
            handled = True
        elif op == "mouse" and set(request) <= {"op", "expected_revision", "kind", "x", "y", "dx", "dy", "button", "scroll"}:
            kind, x, y = request.get("kind"), request.get("x"), request.get("y")
            dx, dy, button, scroll = (request.get("dx", 0), request.get("dy", 0),
                                       request.get("button", 0), request.get("scroll", 0))
            if kind not in ("click", "down", "up", "move", "scroll") or type(button) is not int or not 0 <= button <= 5:
                raise ValueError("invalid canvas mouse action")
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 4096
                   for v in (x, y, dx, dy, scroll)):
                raise ValueError("canvas mouse coordinates must be finite and bounded")
            handled = graph.ui_mouse(kind, x, y, dx=dx, dy=dy, button=button, scroll=scroll)
        elif op == "key" and set(request) <= {"op", "expected_revision", "key", "scancode", "action", "mods"}:
            values = [request.get("key"), request.get("scancode", 0), request.get("action", 1), request.get("mods", 0)]
            if any(type(v) is not int or abs(v) > 65535 for v in values):
                raise ValueError("invalid canvas key action")
            handled = graph.ui_key(*values)
        elif op in ("link_host", "link_dial", "link_report"):
            handled, report = self._apply_link(graph, request)
            if op == "link_report":
                return {"handled": True, "revision": self._revision, "canvas": self._canvas_name, "link": report}
        else:
            raise ValueError("unknown canvas UI action")
        previous_revision = self._revision
        snap = self._build(graph, {"graph"})
        if self._revision != previous_revision:
            graph.save(self._canvas_path)
        with self._cond:
            self._snap = snap
            self._generation += 1
            self._cond.notify_all()
        return {"handled": bool(handled), "revision": self._revision,
                "canvas": self._canvas_name}

    _LINK_NAME = re.compile(r"^[A-Za-z0-9._-]{1,60}$")

    def _apply_link(self, graph, request: dict):
        """Open (or report) a Nodus link on this running canvas; returns (handled, report)."""
        op = request["op"]
        link_id = request.get("id")
        if not isinstance(link_id, str) or not self._LINK_NAME.match(link_id):
            raise ValueError("link id: 1-60 of A-Z a-z 0-9 . _ -")
        if op == "link_report":
            if set(request) != {"op", "expected_revision", "id"}:
                raise ValueError("link_report takes op, expected_revision, id")
            return True, graph.link_report(link_id)
        prefix, port = request.get("prefix"), request.get("port")
        if not isinstance(prefix, str) or not self._LINK_NAME.match(prefix):
            raise ValueError("link prefix: 1-60 of A-Z a-z 0-9 . _ -")
        if type(port) is not int or not 0 < port < 65536:
            raise ValueError("link port is 1-65535")
        if op == "link_host":
            if set(request) != {"op", "expected_revision", "id", "prefix", "port", "key_file"}:
                raise ValueError("link_host takes op, expected_revision, id, prefix, port, key_file")
            with open(str(request["key_file"]), "rb") as handle:
                key = handle.read()
            graph.register_link_host(link_id, prefix, key, port)
        else:
            if not set(request) >= {"op", "expected_revision", "id", "host", "port", "prefix", "control_ticket", "bulk_ticket"} \
                    or not set(request) <= {"op", "expected_revision", "id", "host", "port", "prefix", "control_ticket",
                                            "bulk_ticket", "secure", "pin"}:
                raise ValueError("link_dial takes op, expected_revision, id, host, port, prefix, control_ticket, "
                                 "bulk_ticket[, secure, pin]")
            host = request["host"]
            if not isinstance(host, str) or not 0 < len(host) <= 253:
                raise ValueError("link host is a name or an address")
            try:
                control = base64.b64decode(request["control_ticket"], validate=True)
                bulk = base64.b64decode(request["bulk_ticket"], validate=True)
                pin = bytes.fromhex(request["pin"]) if request.get("pin") is not None else None
            except (TypeError, ValueError) as exc:
                raise ValueError(f"link tickets are base64, a pin is hex: {exc}") from None
            secure = request.get("secure", True)
            if type(secure) is not bool:
                raise ValueError("secure is true or false")
            graph.register_link_wss_dialer(link_id, host, port, prefix, control, bulk, secure=secure, pin_sha256=pin)
        self._bind_link_table(graph, link_id)
        return True, None

    @staticmethod
    def _bind_link_table(graph, link_id: str) -> None:
        """The canvas table "link <id>" holds the link's two rows; made once, rebound when the canvas already has it."""
        lib, label = graph.lib, f"link {link_id}"
        index = -1
        for i in range(max(0, int(lib.nodus_headless_get_module_count(graph._handle)))):
            desc = CanvasModuleDesc()
            if not lib.nodus_headless_get_module_desc(graph._handle, i, ctypes.byref(desc)):
                continue
            if bytes(desc.label).split(b"\0", 1)[0].decode("utf-8", "replace") == label:
                index = i
                break
        if index < 0:
            index = graph.add_module(label, x=40, y=40).index
        names = [f"{link_id}.control".encode("utf-8"), f"{link_id}.bulk".encode("utf-8")]
        rows = (_NativeRow * 2)(*[_NativeRow(1, 0, 0, 1, name) for name in names])
        if not lib.nodus_headless_set_module_rows(graph._handle, index, rows, 2):
            raise RuntimeError(f"the link rows of {link_id} could not be placed")
        if not lib.nodus_headless_set_module_stack_contract(graph._handle, index, 1):
            raise RuntimeError("the link table's stack contract was refused")

    def _build(self, graph, wanted=frozenset()):
        path = os.path.join(self._tmp, "snapshot.nodus.txt")
        graph.save(path)
        with open(path, "rb") as handle:
            data = handle.read()
        os.remove(path)
        shutil.rmtree(path + ".modules", ignore_errors=True)
        if data != self._last_bytes:
            self._last_bytes = data
            self._revision += 1
        lib = graph.lib
        modules = []
        for index in range(max(0, int(lib.nodus_headless_get_module_count(graph._handle)))):
            rows = int(lib.nodus_headless_get_module_row_count(graph._handle, index))
            try:
                tail = [float(v) for v in graph._stack_tail(index)]
            except RuntimeError:
                tail = []
            modules.append({"index": index, "rows": rows, "stack_tail": tail})
        cports = []
        for prefix, _cap, _edge, declared in graph.contract_ports():
            bound = graph.contract_port(prefix)
            cports.append({"prefix": prefix,
                           "listen_port": bound[0] if bound else declared,
                           "hosts": bound[1] if bound else 0,
                           "engines": bound[2] if bound else 0})
        ports = []
        for is_output in (0, 1):
            count = int(lib.nodus_headless_get_port_count(graph._handle, is_output))
            for ordinal in range(max(0, count)):
                name = ctypes.create_string_buffer(1024)
                mi, ci, head = ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32()
                cfg = _NativeFifoConfig()
                if not lib.nodus_headless_get_port_at(
                        graph._handle, is_output, ordinal, ctypes.cast(name, ctypes.c_char_p),
                        len(name), ctypes.byref(mi), ctypes.byref(ci), ctypes.byref(head),
                        ctypes.byref(cfg)):
                    continue
                unread = ctypes.c_int32()
                if not lib.nodus_headless_get_port_unread(
                        graph._handle, name.value, is_output, ctypes.byref(unread)):
                    unread.value = -1
                ports.append({"name": name.value.decode("utf-8"),
                              "is_output": bool(is_output), "unread": unread.value})
        state = {"revision": self._revision, "ticks": self._ticks_fn(),
                 "canvas": self._canvas_name, "modules": modules,
                 "contract_ports": cports, "ports": ports,
                 # The Python API exposes no event queue; nothing is reported.
                 "events": []}
        tick = self._ticks_fn()
        views = {"digest": console_views.dumps(console_views.digest_document(
            graph, canvas=self._canvas_name, revision=self._revision, data=data, tick=tick,
            rate=self._rate, started=self._started, compile_hosts=len(self._hosts)))}
        try:
            graph_doc = None
            if wanted & {"graph", "tools"}:
                graph_doc = console_views.graph_document(graph, revision=self._revision)
            if "graph" in wanted:
                views["graph"] = console_views.dumps(graph_doc)
            if "tools" in wanted:
                views["tools"] = console_views.dumps(console_views.tools_document(graph, graph_doc))
            if "activity" in wanted:
                views["activity"] = console_views.dumps(console_views.activity_document(
                    graph, tick=tick, compile_hosts=list(self._hosts)))
        except Exception as exc:  # noqa: BLE001
            print(f"run_canvas: console view failed: {exc}", file=sys.stderr, flush=True)
        return (time.monotonic(), data, json.dumps(state).encode("utf-8"), self._revision, views)

    # ---- client threads --------------------------------------------------
    def _accept_loop(self) -> None:
        while not self._closing:
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._lock:
                admit = self._clients < CONSOLE_MAX_CLIENTS
                if admit:
                    self._clients += 1
            if not admit:
                self._send(conn, b"REFUSED too many clients\n")
                conn.close()
                continue
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    @staticmethod
    def _send(conn, data: bytes) -> bool:
        try:
            conn.sendall(data)
            return True
        except OSError:
            return False

    def _client(self, conn) -> None:
        try:
            conn.settimeout(CONSOLE_IDLE_SECONDS)
            reader = conn.makefile("rb")
            authed = False
            actions = False
            while not self._closing:
                try:
                    raw = reader.readline(4096)
                except (OSError, socket.timeout):
                    return
                if not raw:
                    return
                line = raw.rstrip(b"\r\n")
                if not authed:
                    parts = line.split(b" ", 2)
                    if len(parts) != 3 or parts[0] != b"HELLO" or parts[1] not in (b"nodus-console/1", b"nodus-console/2"):
                        self._send(conn, b"REFUSED bad hello\n")
                        return
                    if not hmac.compare_digest(parts[2], self._token):
                        self._send(conn, b"REFUSED bad token\n")
                        return
                    authed = True
                    actions = parts[1] == b"nodus-console/2"
                    if not self._send(conn, b"OK console " + self._canvas_name.encode("utf-8")
                                      + (b" actions\n" if actions else b" read-only\n")):
                        return
                    continue
                if line == b"SNAPSHOT":
                    snap = self._snapshot()
                    if snap is None:
                        if not self._send(conn, b"REFUSED unavailable\n"):
                            return
                        continue
                    _t, data, state, revision, _views = snap
                    head = b"CANVAS %d %d\n" % (revision, len(data))
                    if not self._send(conn, head + data + b"\nSTATE " + state + b"\n"):
                        return
                elif line in self._VERBS:
                    view = self._VERBS[line]
                    snap = self._snapshot(view)
                    body = snap[4].get(view) if snap else None
                    if body is None:
                        if not self._send(conn, b"REFUSED unavailable\n"):
                            return
                        continue
                    if not self._send(conn, b"JSON %s %d\n" % (line, len(body)) + body):
                        return
                elif actions and line.startswith(b"ACTION "):
                    try:
                        size = int(line[7:])
                        if not 1 <= size <= 4096:
                            raise ValueError("invalid action size")
                        request = json.loads(reader.read(size))
                    except (ValueError, UnicodeError):
                        self._send(conn, b"REFUSED invalid action\n")
                        return
                    box, done = [], threading.Event()
                    with self._lock:
                        self._actions.append((request, box, done))
                        self._wanted.set()
                    if not done.wait(CONSOLE_STEP_WAIT_SECONDS):
                        self._send(conn, b"REFUSED action outcome uncertain\n")
                        return
                    body = json.dumps(box[0], separators=(",", ":")).encode()
                    if not self._send(conn, b"JSON ACTION %d\n" % len(body) + body):
                        return
                elif not self._send(conn, b"REFUSED read-only\n"):
                    return
        finally:
            with self._lock:
                self._clients -= 1
            try:
                conn.close()
            except OSError:
                pass

    _VERBS = {b"DIGEST": "digest", b"GRAPH": "graph", b"TOOLS": "tools", b"ACTIVITY": "activity"}

    def _snapshot(self, view="digest"):
        with self._cond:
            if (self._snap and time.monotonic() - self._snap[0] < CONSOLE_COALESCE_SECONDS
                    and view in self._snap[4]):
                return self._snap
            self._want_views.add(view)
            generation = self._generation
            self._wanted.set()
            deadline = time.monotonic() + CONSOLE_STEP_WAIT_SECONDS
            while self._generation == generation and not self._closing:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cond.wait(left)
            return self._snap


def _parse_acquire(text: str):
    # PREFIX:CAPACITY or PREFIX:CAPACITY:LOAN_TICKS (loan length in market turns).
    parts = text.split(":")
    loan = None
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        loan = int(parts.pop())
        if loan == 0:
            raise ValueError(f"--acquire LOAN_TICKS must be > 0, got {text!r}")
    prefix, cap = ":".join(parts[:-1]), parts[-1]
    if len(parts) < 2 or not prefix or not cap.isdigit():
        raise ValueError(f"--acquire expects PREFIX:CAPACITY[:LOAN_TICKS], got {text!r}")
    return prefix, int(cap), loan


def _parse_compile_host(text: str):
    # JOBS_DIR may hold a drive colon (C:/jobs); only the first two colons split.
    parts = text.split(":", 2)
    if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or int(parts[1]) == 0 or not parts[2]:
        raise ValueError(f"--compile-host expects PREFIX:HOST_ID:JOBS_DIR (HOST_ID non-zero), got {text!r}")
    return parts[0], int(parts[1]), parts[2]


def _parse_link_host(text: str):
    # KEYFILE may hold a drive colon (C:/state/secrets/nodus_link.key); only the first three colons split.
    parts = text.split(":", 3)
    if len(parts) != 4 or not parts[0] or not parts[1] or not parts[2].isdigit() or not 0 < int(parts[2]) < 65536 or not parts[3]:
        raise ValueError(f"--link-host expects ID:PREFIX:PORT:KEYFILE (PORT 1-65535), got {text!r}")
    return parts[0], parts[1], int(parts[2]), parts[3]


def _parse_cron(text: str):
    # JOBS_DIR may hold a drive colon (C:/jobs); only the first two colons split.
    parts = text.split(":", 2)
    if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or int(parts[1]) == 0 or not parts[2]:
        raise ValueError(f"--cron expects PREFIX:ENGINE_ID:JOBS_DIR (ENGINE_ID non-zero), got {text!r}")
    return parts[0], int(parts[1]), parts[2]


def _claim_stdout():
    """The native library writes diagnostics to fd 1; keep the real stdout for JSON only."""
    sys.stdout.flush()
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canvas", required=True)
    parser.add_argument("--rate", type=float, default=60.0)
    parser.add_argument("--acquire", action="append", default=[], metavar="PREFIX:CAPACITY[:LOAN_TICKS]",
                        help="acquire a contract pair; LOAN_TICKS = loan length in market turns (default 60). "
                             "A prefix carrying a compile host needs LOAN_TICKS >= --rate x the longest compile, "
                             "e.g. --acquire turing:8:36000 at 60/s allows 10 minutes")
    parser.add_argument("--ticks", type=int, default=None)
    parser.add_argument("--compile-host", action="append", default=[], metavar="PREFIX:HOST_ID:JOBS_DIR",
                        help="attach a Turing compile host to an --acquire'd prefix (repeatable); "
                             "requests <JOBS_DIR>/<id>.request.json, products <JOBS_DIR>/<id>/")
    parser.add_argument("--cron", action="append", default=[], metavar="PREFIX:ENGINE_ID:JOBS_DIR",
                        help="attach the cron tool as an engine of an acquired prefix; JOBS_DIR/cron.table is its "
                             "schedule and it keeps cron.ledger, cron.log and one request file per run there")
    parser.add_argument("--link-host", action="append", default=[], metavar="ID:PREFIX:PORT:KEYFILE",
                        help="register a Nodus link host before the canvas loads (research case 98d49cb8): rows "
                             "<ID>.control and <ID>.bulk over a 127.0.0.1:PORT listener for /ws/link/<PREFIX>/, tickets "
                             "checked with the 32-byte KEYFILE (meowdoku link_tickets KEY_FILE); the canvas binds the rows")
    parser.add_argument("--compile-python", default=None,
                        help="python for the Turing compiler driver (default: this interpreter)")
    parser.add_argument("--turing-root", default=None,
                        help="Turing checkout (default: ../turing next to this repo)")
    parser.add_argument("--module-library-root", default=None,
                        help="module library root (default: <canvas>.modules)")
    parser.add_argument("--compile-adopt", action="store_true",
                        help="adopt and load compiled products on the canvas lane (default off)")
    parser.add_argument("--console-port", type=int, default=None,
                        help="read-only admin console on 127.0.0.1 (0 = ephemeral)")
    parser.add_argument("--console-token-file", default=None,
                        help="file whose first line is the console shared secret")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    console = None
    try:
        if args.rate <= 0:
            raise ValueError("--rate must be positive")
        token = None
        if args.console_port is not None:
            if not args.console_token_file:
                raise ValueError("--console-port requires --console-token-file")
            with open(args.console_token_file, encoding="utf-8") as handle:
                token = handle.readline().rstrip("\r\n")
            if not token:
                raise ValueError("console token file is empty")
        acquires = [_parse_acquire(a) for a in args.acquire]
        compile_hosts = [_parse_compile_host(a) for a in args.compile_host]
        acquired = {prefix for prefix, _, _ in acquires}
        for prefix, _host_id, _jobs in compile_hosts:
            if prefix not in acquired:
                raise ValueError(f"--compile-host prefix {prefix!r} must also be given to --acquire")
        if len({(p, h) for p, h, _ in compile_hosts}) != len(compile_hosts):
            raise ValueError("--compile-host PREFIX:HOST_ID given twice")
        link_hosts = [_parse_link_host(a) for a in args.link_host]
        if len({i for i, _, _, _ in link_hosts}) != len(link_hosts):
            raise ValueError("--link-host ID given twice")
        crons = [_parse_cron(a) for a in args.cron]
        for prefix, _engine_id, _jobs in crons:
            if prefix not in acquired:
                raise ValueError(f"--cron prefix {prefix!r} must also be given to --acquire")
        if len({(p, e) for p, e, _ in crons}) != len(crons):
            raise ValueError("--cron PREFIX:ENGINE_ID given twice")
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        compile_python = args.compile_python or sys.executable
        turing_root = args.turing_root or str(repo_root.parent / "turing")
        module_library_root = args.module_library_root or (args.canvas + ".modules")
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _on_signal)
        out = _claim_stdout()
        period = 1.0 / args.rate
        ticks = 0
        console_port = None
        if token is not None:
            console = ConsoleServer(token, os.path.basename(args.canvas), lambda: ticks, args.canvas)
            console_port = console.start(args.console_port)
            console.set_context(args.rate, [])
        with Graph() as graph:
            for prefix, capacity, loan in acquires:
                graph.acquire_contract_pair(prefix, capacity, loan)
            links = []
            for link_id, link_prefix, link_port, key_file in link_hosts:   # before the load: the canvas binds the rows
                with open(key_file, "rb") as handle:
                    key = handle.read()
                bound = graph.register_link_host(link_id, link_prefix, key, link_port)
                links.append({"id": link_id, "prefix": link_prefix, "listen_port": bound})
            graph.load(args.canvas)
            attached = []
            try:
                for prefix, host_id, jobs_dir in compile_hosts:
                    graph.attach_compile_host(
                        prefix, host_id, python_executable=compile_python, nodus_root=str(repo_root),
                        turing_root=turing_root, module_library_root=module_library_root,
                        jobs_dir=jobs_dir, adopt=args.compile_adopt)
                    attached.append((prefix, host_id))
            except BaseException:
                for prefix, host_id in attached:
                    graph.detach_compile_host(prefix, host_id)
                raise
            scheduled, cron_report = [], []
            for prefix, engine_id, jobs_dir in crons:      # after the load: the engine joins the restored negotiator
                # A cron tool that cannot start (an unreadable table, a library built before it) is reported
                # and left off; it must not take the canvas host and its other services down with it.
                try:
                    graph.attach_cron(prefix, engine_id, jobs_dir)
                    scheduled.append((prefix, engine_id))
                    cron_report.append({"prefix": prefix, "engine": engine_id, "attached": True})
                except RuntimeError as exc:
                    print(f"run_canvas: {exc}", file=sys.stderr, flush=True)
                    cron_report.append({"prefix": prefix, "engine": engine_id, "attached": False, "reason": str(exc)})
            if console is not None:
                console.set_context(args.rate, attached)
            ports = []
            for prefix, capacity, edge, declared in graph.contract_ports():
                bound = graph.contract_port(prefix)
                ports.append({"prefix": prefix,
                              "listen_port": bound[0] if bound else declared,
                              "capacity": capacity, "edge": edge})
            print(json.dumps({"canvas": args.canvas, "contract_ports": ports, "links": links,
                              "pid": os.getpid(), "console_port": console_port, "cron": cron_report}),
                  file=out, flush=True)
            next_t = time.perf_counter()
            try:
                while not _stop and (args.ticks is None or ticks < args.ticks):
                    graph.step(period)
                    ticks += 1
                    if console is not None:
                        console.service(graph)
                    next_t += period
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_t = time.perf_counter()
            finally:
                for prefix, engine_id in scheduled:
                    try:
                        graph.detach_cron(prefix, engine_id)
                    except RuntimeError as exc:
                        print(f"run_canvas: {exc}", file=sys.stderr, flush=True)
                # Hosts end (running compile cancelled, thread joined) before any pair is released.
                for prefix, host_id in attached:
                    try:
                        graph.detach_compile_host(prefix, host_id)
                    except RuntimeError as exc:
                        print(f"run_canvas: {exc}", file=sys.stderr, flush=True)
        if console is not None:
            console.close()
        print(json.dumps({"ticks": ticks}), file=out, flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"run_canvas: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if console is not None:
            console.close()


if __name__ == "__main__":
    raise SystemExit(main())
