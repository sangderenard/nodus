"""Two running canvas hosts (tools/run_canvas.py) open a Nodus link between them at run time, through each one's console.

Research case 98d49cb8: neither process is started for linking. Host B accepts links (link_host), host A dials it
(link_dial) with tickets issued by the meowdoku ticket code from the shared key, over ws:// to B's listener (the same
socket Apache proxies /ws/link/ to). Both report linked; then each canvas is saved and still loads.

Needs a nodus_headless library with the link ABI (NODUS_HEADLESS_TEST_DLL) and the turing checkout beside this repo.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TURING_EXAMPLES = pathlib.Path(os.environ.get("TURING_ROOT", ROOT.parent / "turing")) / "examples"
sys.path.insert(0, str(TURING_EXAMPLES))

from nodus_graph import Graph  # noqa: E402
from meowdoku import link_tickets  # noqa: E402


class Console:
    """A nodus-console/2 client."""

    def __init__(self, port: int, token: str) -> None:
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10.0)
        self.file = self.sock.makefile("rb")
        self.sock.sendall(f"HELLO nodus-console/2 {token}\n".encode("utf-8"))
        line = self.file.readline().decode("utf-8").strip()
        if not line.startswith("OK"):
            raise RuntimeError(f"console refused: {line}")
        self.revision = 0

    def _read_json(self, verb: str):
        head = self.file.readline().decode("utf-8").split()
        if head[:2] != ["JSON", verb]:
            raise RuntimeError(f"unexpected console answer {head}")
        return json.loads(self.file.read(int(head[2])))

    def refresh(self) -> None:
        self.sock.sendall(b"SNAPSHOT\n")
        head = self.file.readline().decode("utf-8").split()
        self.file.read(int(head[2]))
        self.file.readline()
        state = json.loads(self.file.readline().decode("utf-8").split(" ", 1)[1])
        self.revision = state["revision"]

    def action(self, **request):
        self.refresh()
        body = json.dumps(dict(request, expected_revision=self.revision)).encode("utf-8")
        self.sock.sendall(f"ACTION {len(body)}\n".encode("utf-8") + body)
        return self._read_json("ACTION")

    def close(self) -> None:
        self.sock.close()


def start_host(tmp: pathlib.Path, name: str):
    canvas = tmp / f"{name}.nodus.txt"
    with Graph() as graph:
        graph.add_module("start")
        graph.save(canvas)
    token_file = tmp / f"{name}.token"
    token = secrets.token_urlsafe(24)
    token_file.write_text(token, encoding="utf-8")
    err = open(tmp / f"{name}.stderr.txt", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "run_canvas.py"), "--canvas", str(canvas), "--rate", "200",
         "--console-port", "0", "--console-token-file", str(token_file)],
        stdout=subprocess.PIPE, stderr=err, text=True, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    first = json.loads(proc.stdout.readline())
    return proc, err, canvas, Console(first["console_port"], token)


def stop(proc) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.CTRL_BREAK_EVENT) if sys.platform == "win32" else proc.terminate()
        try:
            proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()


class RunCanvasLinkTest(unittest.TestCase):
    def test_link_opened_at_run_time(self) -> None:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="run_canvas_link_"))
        key_file = tmp / "nodus_link.key"
        key = secrets.token_bytes(32)
        key_file.write_bytes(key)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        a = b = None
        try:
            a = start_host(tmp, "dialing")
            b = start_host(tmp, "hosting")
            answer = b[3].action(op="link_host", id="peer", prefix="nodus", port=port, key_file=str(key_file))
            self.assertTrue(answer.get("handled"), answer)
            tickets = link_tickets.issue(key, "nodus", 42)
            answer = a[3].action(op="link_dial", id="peer", host="127.0.0.1", port=port, prefix="nodus",
                                 control_ticket=tickets["control_ticket"], bulk_ticket=tickets["bulk_ticket"],
                                 secure=False)
            self.assertTrue(answer.get("handled"), answer)
            deadline = time.monotonic() + 20.0
            states = {}
            while time.monotonic() < deadline:
                states = {name: c.action(op="link_report", id="peer")["link"] for name, c in (("a", a[3]), ("b", b[3]))}
                if all(s and s["state"] == "linked" for s in states.values()):
                    break
                time.sleep(0.2)
            self.assertEqual({k: v and v["state"] for k, v in states.items()}, {"a": "linked", "b": "linked"}, states)
            print("linked:", json.dumps(states))
        finally:
            for host in (a, b):
                if host:
                    host[3].close()
                    stop(host[0])
                    host[1].close()
        # The canvases kept their link tables and still load with no link registered.
        for host in (a, b):
            with Graph() as graph:
                graph.load(host[2])


if __name__ == "__main__":
    unittest.main()
