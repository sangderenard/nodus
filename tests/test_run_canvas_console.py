"""tools/run_canvas.py managed canvas console protocols."""

from __future__ import annotations

import json
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_run_canvas import CAPACITY, PREFIX, save_messenger_canvas  # noqa: E402

TOKEN = "s3cret-token"


class Client:
    def __init__(self, port: int) -> None:
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10.0)
        self.f = self.sock.makefile("rb")

    def send(self, text: str) -> None:
        self.sock.sendall(text.encode() + b"\n")

    def line(self) -> bytes:
        return self.f.readline()

    def hello(self, token: str = TOKEN) -> bytes:
        self.send(f"HELLO nodus-console/1 {token}")
        return self.line()

    def snapshot(self):
        self.send("SNAPSHOT")
        head = self.line().split()
        assert head[0] == b"CANVAS", head
        revision, count = int(head[1]), int(head[2])
        data = self.f.read(count)
        assert self.f.read(1) == b"\n"
        state = self.line()
        assert state.startswith(b"STATE "), state
        return revision, count, data, json.loads(state[6:])

    def close(self) -> None:
        self.f.close()
        self.sock.close()


class ConsoleTest(unittest.TestCase):
    def start(self, rate: str):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="run_canvas_console_"))
        canvas = tmp / "messenger.nodus.txt"
        save_messenger_canvas(canvas)
        token_file = tmp / "token.txt"
        token_file.write_text(TOKEN + "\n", encoding="utf-8")
        self.err_file = open(tmp / "stderr.txt", "w", encoding="utf-8")
        self.err_path = tmp / "stderr.txt"
        self.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "tools" / "run_canvas.py"), "--canvas", str(canvas),
             "--rate", rate, "--acquire", f"{PREFIX}:{CAPACITY}",
             "--console-port", "0", "--console-token-file", str(token_file)],
            stdout=subprocess.PIPE, stderr=self.err_file, text=True,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        first = json.loads(self.proc.stdout.readline())
        self.assertIsNotNone(first["console_port"])
        return first["console_port"], first

    def stop(self) -> dict:
        if sys.platform == "win32":
            self.proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            self.proc.terminate()
        try:
            out, _ = self.proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.communicate()
            self.fail("run_canvas did not exit")
        self.assertEqual(self.proc.returncode, 0, self.err_path.read_text(errors="replace")[-2000:])
        return json.loads([ln for ln in out.splitlines() if ln.strip()][-1])

    def tearDown(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        if getattr(self, "err_file", None):
            self.err_file.close()

    def test_auth_snapshot_and_refusals(self) -> None:
        port, first = self.start("200")
        bad = Client(port)
        self.assertTrue(bad.hello("wrong").startswith(b"REFUSED"))
        self.assertEqual(bad.line(), b"")  # closed
        bad.close()
        early = Client(port)
        early.send("SNAPSHOT")
        self.assertTrue(early.line().startswith(b"REFUSED"))
        self.assertEqual(early.line(), b"")
        early.close()

        c = Client(port)
        self.assertEqual(c.hello(), b"OK console messenger.nodus.txt read-only\n")
        rev1, count, data, state = c.snapshot()
        self.assertTrue(data.startswith(b"CANVAS V4"))
        self.assertIn(b"CONTRACT_PORT", data)
        self.assertEqual(len(data), count)
        self.assertEqual(state["revision"], rev1)
        self.assertEqual(state["canvas"], "messenger.nodus.txt")
        self.assertEqual(len(state["modules"]), 2)
        self.assertEqual([p["prefix"] for p in state["contract_ports"]], [PREFIX])
        self.assertEqual(state["contract_ports"][0]["listen_port"],
                         first["contract_ports"][0]["listen_port"])
        self.assertGreater(state["ticks"], 0)
        self.assertEqual(state["events"], [])
        time.sleep(0.15)
        rev2, _, data2, _ = c.snapshot()
        self.assertEqual(data2, data)
        self.assertEqual(rev2, rev1)

        c.send("RESET")
        self.assertEqual(c.line(), b"REFUSED read-only\n")
        rev3, _, _, _ = c.snapshot()
        self.assertEqual(rev3, rev1)
        c.close()
        self.stop()

    def test_tick_rate_with_polling_client(self) -> None:
        port, _ = self.start("200")
        c = Client(port)
        self.assertTrue(c.hello().startswith(b"OK"))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            c.snapshot()
        c.close()
        last = self.stop()
        self.assertGreaterEqual(last["ticks"], 300, last)

    def test_v2_ui_action_checks_revision_on_stepping_thread(self) -> None:
        port, _ = self.start("200")
        c = Client(port)
        c.send(f"HELLO nodus-console/2 {TOKEN}")
        self.assertEqual(c.line(), b"OK console messenger.nodus.txt actions\n")
        revision, _, _, _ = c.snapshot()

        def action(body):
            payload = json.dumps(body).encode()
            c.sock.sendall(b"ACTION %d\n" % len(payload) + payload)
            header = c.line().split()
            self.assertEqual(header[:2], [b"JSON", b"ACTION"])
            return json.loads(c.f.read(int(header[2])))

        changed = action({"op": "viewport", "expected_revision": revision,
                          "width": 800, "height": 600})
        self.assertTrue(changed["handled"])
        clicked = action({"op": "mouse", "expected_revision": changed["revision"],
                          "kind": "click", "x": 400, "y": 200})
        self.assertIn("handled", clicked)
        stale = action({"op": "viewport", "expected_revision": revision + 999,
                        "width": 800, "height": 600})
        self.assertEqual(stale["error"], "canvas revision changed")
        c.close()
        self.stop()


if __name__ == "__main__":
    unittest.main()
