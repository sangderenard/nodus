"""Canvas host process: load a saved canvas and step it at a fixed rate.

Prints one JSON line describing the bound contract ports, steps until
SIGINT/SIGTERM or --ticks, then prints {"ticks": n}.  Errors go to stderr, exit 2.

The native library prints diagnostics on fd 1, so fd 1 is redirected to stderr and the
JSON lines are written to a saved duplicate of the real stdout.

Optional read-only admin console (--console-port / --console-token-file), protocol
nodus-console/1, newline-delimited UTF-8 over 127.0.0.1:
  HELLO nodus-console/1 <token>  -> OK console <canvas-basename> read-only | REFUSED <reason> (closes)
  SNAPSHOT -> CANVAS <revision> <n>\\n<n bytes>\\nSTATE <json>\\n
  anything else -> REFUSED read-only
"""

from __future__ import annotations

import argparse
import ctypes
import hmac
import json
import os
import pathlib
import shutil
import signal
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nodus_graph import Graph  # noqa: E402
from nodus_graph.native import _NativeFifoConfig  # noqa: E402

_stop = False

CONSOLE_MAX_CLIENTS = 8
CONSOLE_IDLE_SECONDS = 60.0
CONSOLE_COALESCE_SECONDS = 0.1
CONSOLE_STEP_WAIT_SECONDS = 5.0


def _on_signal(signum, frame) -> None:
    global _stop
    _stop = True


class ConsoleServer:
    """Read-only admin console.

    Client threads never touch the Graph: they request a snapshot and the stepping
    thread builds it in service().  Snapshots are shared and coalesced to 100 ms.
    """

    def __init__(self, token: str, canvas_name: str, ticks_fn) -> None:
        self._token = token.encode("utf-8")
        self._canvas_name = canvas_name
        self._ticks_fn = ticks_fn
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._wanted = threading.Event()
        self._clients = 0
        self._closing = False
        self._snap = None  # (monotonic_time, canvas_bytes, state_json_bytes, revision)
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
            fresh = (self._snap is not None
                     and time.monotonic() - self._snap[0] < CONSOLE_COALESCE_SECONDS)
        if fresh:
            return
        self._wanted.clear()
        try:
            snap = self._build(graph)
        except Exception as exc:  # noqa: BLE001
            print(f"run_canvas: console snapshot failed: {exc}", file=sys.stderr, flush=True)
            snap = None
        with self._cond:
            if snap is not None:
                self._snap = snap
            self._generation += 1
            self._cond.notify_all()

    def _build(self, graph):
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
        return (time.monotonic(), data, json.dumps(state).encode("utf-8"), self._revision)

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
                    if len(parts) != 3 or parts[0] != b"HELLO" or parts[1] != b"nodus-console/1":
                        self._send(conn, b"REFUSED bad hello\n")
                        return
                    if not hmac.compare_digest(parts[2], self._token):
                        self._send(conn, b"REFUSED bad token\n")
                        return
                    authed = True
                    if not self._send(conn, b"OK console " + self._canvas_name.encode("utf-8")
                                      + b" read-only\n"):
                        return
                    continue
                if line == b"SNAPSHOT":
                    snap = self._snapshot()
                    if snap is None:
                        if not self._send(conn, b"REFUSED unavailable\n"):
                            return
                        continue
                    _t, data, state, revision = snap
                    head = b"CANVAS %d %d\n" % (revision, len(data))
                    if not self._send(conn, head + data + b"\nSTATE " + state + b"\n"):
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

    def _snapshot(self):
        with self._cond:
            if self._snap and time.monotonic() - self._snap[0] < CONSOLE_COALESCE_SECONDS:
                return self._snap
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
    prefix, sep, cap = text.rpartition(":")
    if not sep or not prefix or not cap.isdigit():
        raise ValueError(f"--acquire expects PREFIX:CAPACITY, got {text!r}")
    return prefix, int(cap)


def _parse_compile_host(text: str):
    # JOBS_DIR may hold a drive colon (C:/jobs); only the first two colons split.
    parts = text.split(":", 2)
    if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or int(parts[1]) == 0 or not parts[2]:
        raise ValueError(f"--compile-host expects PREFIX:HOST_ID:JOBS_DIR (HOST_ID non-zero), got {text!r}")
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
    parser.add_argument("--acquire", action="append", default=[], metavar="PREFIX:CAPACITY")
    parser.add_argument("--ticks", type=int, default=None)
    parser.add_argument("--compile-host", action="append", default=[], metavar="PREFIX:HOST_ID:JOBS_DIR",
                        help="attach a Turing compile host to an --acquire'd prefix (repeatable); "
                             "requests <JOBS_DIR>/<id>.request.json, products <JOBS_DIR>/<id>/")
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
        acquired = {prefix for prefix, _ in acquires}
        for prefix, _host_id, _jobs in compile_hosts:
            if prefix not in acquired:
                raise ValueError(f"--compile-host prefix {prefix!r} must also be given to --acquire")
        if len({(p, h) for p, h, _ in compile_hosts}) != len(compile_hosts):
            raise ValueError("--compile-host PREFIX:HOST_ID given twice")
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
            console = ConsoleServer(token, os.path.basename(args.canvas), lambda: ticks)
            console_port = console.start(args.console_port)
        with Graph() as graph:
            for prefix, capacity in acquires:
                graph.acquire_contract_pair(prefix, capacity)
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
            ports = []
            for prefix, capacity, edge, declared in graph.contract_ports():
                bound = graph.contract_port(prefix)
                ports.append({"prefix": prefix,
                              "listen_port": bound[0] if bound else declared,
                              "capacity": capacity, "edge": edge})
            print(json.dumps({"canvas": args.canvas, "contract_ports": ports,
                              "pid": os.getpid(), "console_port": console_port}),
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
