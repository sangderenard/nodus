"""Console read protocol: v1 (HELLO/SNAPSHOT) plus DIGEST/GRAPH/TOOLS/ACTIVITY.

Runs tools/run_canvas.py as a subprocess against the headless library the
run_canvas tests use (find_headless_library); skipped when it is missing or lacks
the contract-port / compile-host ABI.  Nothing is built here.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

_SKIP = ""
try:
    from nodus_headless_bridge import find_headless_library, load_headless_lib
    _lib = load_headless_lib(str(find_headless_library()))
    for _symbol in ("nodus_headless_acquire_contract_pair", "nodus_headless_compile_host_stats",
                    "nodus_headless_contract_pair_loan_ticks"):
        if not hasattr(_lib, _symbol):
            _SKIP = f"headless library lacks {_symbol}"
except OSError as exc:
    _SKIP = f"headless library unavailable: {exc}"

if not _SKIP:
    from nodus_graph.console_client import ConsoleClient, ConsoleError
    from test_run_canvas import CAPACITY, PREFIX, save_messenger_canvas

TOKEN = "test-token-0123456789"


def _raw(port: int, *lines: bytes, read: int = 4096) -> bytes:
    """Send each line, collecting whatever the server answers; resets count as closed."""
    out = b""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.settimeout(2)
        for line in lines:
            try:
                sock.sendall(line + b"\n")
                expected = lines.index(line) + 1
                while out.count(b"\n") < expected:
                    chunk = sock.recv(read)
                    if not chunk:
                        return out
                    out += chunk
            except (OSError, socket.timeout):
                return out
    return out


@unittest.skipIf(_SKIP, _SKIP)
class ConsoleReadProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="console_read_"))
        canvas = cls.tmp / "messenger.nodus.txt"
        save_messenger_canvas(canvas)
        token_file = cls.tmp / "token.txt"
        token_file.write_text(TOKEN + "\n", encoding="utf-8")
        cls.err = open(cls.tmp / "stderr.txt", "w", encoding="utf-8")
        cls.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "tools" / "run_canvas.py"), "--canvas", str(canvas),
             "--rate", "200", "--acquire", f"{PREFIX}:{CAPACITY}",
             "--compile-host", f"{PREFIX}:7:{cls.tmp / 'jobs'}",
             "--console-port", "0", "--console-token-file", str(token_file)],
            stdout=subprocess.PIPE, stderr=cls.err, text=True,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        first = json.loads(cls.proc.stdout.readline())
        cls.port = first["console_port"]
        cls.canvas_name = "messenger.nodus.txt"

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.proc.poll() is None:
            if sys.platform == "win32":
                cls.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                cls.proc.terminate()
            try:
                cls.proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
                cls.proc.communicate()
        cls.err.close()

    def client(self) -> "ConsoleClient":
        client = ConsoleClient("127.0.0.1", self.port, TOKEN)
        client.hello()
        self.addCleanup(client.close)
        return client

    def test_hello_and_v1_snapshot(self) -> None:
        client = self.client()
        self.assertEqual(client.canvas_name, self.canvas_name)
        snap = client.snapshot()
        self.assertTrue(snap["canvas"].startswith(b"CANVAS"))
        self.assertIn("modules", snap["state"])

    def test_digest_shape_and_matches_snapshot(self) -> None:
        client = self.client()
        digest, snap = client.digest(), client.snapshot()
        self.assertEqual(digest["protocol"], 2)
        self.assertEqual(digest["canvas"], self.canvas_name)
        self.assertEqual(digest["sha256"], hashlib.sha256(snap["canvas"]).hexdigest())
        self.assertEqual(digest["bytes"], len(snap["canvas"]))
        self.assertEqual(digest["revision"], snap["revision"])
        self.assertEqual((digest["modules"], digest["edges"], digest["contract_ports"],
                          digest["compile_hosts"]), (2, 1, 1, 1))
        self.assertEqual(digest["rows"], sum(m["rows"] for m in snap["state"]["modules"]))
        self.assertEqual(digest["rate"], 200.0)
        for key in ("tick", "started"):
            self.assertIn(key, digest)

    def test_digest_stable_while_canvas_unchanged(self) -> None:
        # The host is read-only and has no way to edit its canvas, so "changes when the
        # canvas changes" cannot be driven from a client; stability is what is testable.
        client = self.client()
        first = client.digest()
        second = client.digest()
        for key in ("sha256", "revision", "bytes", "modules", "rows", "edges"):
            self.assertEqual(first[key], second[key])
        self.assertGreaterEqual(second["tick"], first["tick"])

    def test_graph_is_definition_v1_plus_contract_ports(self) -> None:
        graph = self.client().graph()
        self.assertEqual(graph["schema"], "nodus.graph-definition.v1")
        self.assertEqual(len(graph["modules"]), 2)
        module = graph["modules"][0]
        for key in ("index", "label", "stable_id", "x", "y", "rows", "contacts", "exec_mode"):
            self.assertIn(key, module)
        self.assertIsNone(module["exec_mode"])
        self.assertEqual(module["rows"][0]["index"], 0)
        edge = graph["edges"][0]
        self.assertEqual((edge["index"], edge["token_bytes"], edge["slots"]), (0, 64, 4))
        self.assertEqual(len(graph["contract_ports"]), 1)
        port = graph["contract_ports"][0]
        self.assertEqual((port["prefix"], port["capacity"], port["edge"]), (PREFIX, CAPACITY, 0))
        self.assertGreater(port["listen_port"], 0)
        self.assertIn("loan_ticks", port)
        self.assertEqual(graph["revision"], self.client().digest()["revision"])

    def test_tools_shape(self) -> None:
        tools = self.client().tools()
        names = {t["name"]: t for t in tools["tools"]}
        self.assertEqual(names["ADD"]["source"], "builtin")
        self.assertEqual(names[PREFIX + ".market"]["source"], "plugin")
        self.assertEqual(names[PREFIX + ".worker"]["plugin_id"], PREFIX + ".worker")
        for tool in tools["tools"]:
            self.assertEqual(set(tool), {"name", "plugin_id", "source", "description"})
        self.assertIs(tools["complete"], False)
        self.assertTrue(tools["missing"])

    def test_activity_shape(self) -> None:
        act = self.client().activity()
        self.assertIn("tick", act)
        self.assertEqual([e["index"] for e in act["edges"]], [0])
        self.assertIn("fill", act["edges"][0])
        self.assertIsNone(act["edges"][0]["seq"])
        port = act["contract_ports"][0]
        self.assertEqual(port["prefix"], PREFIX)
        self.assertIsNone(port["pending"])
        self.assertEqual(len(act["compile_hosts"]), 1)
        host = act["compile_hosts"][0]
        self.assertEqual((host["prefix"], host["host_id"]), (PREFIX, 7))
        for key in ("accepted", "completed", "refused", "acknowledged", "in_flight"):
            self.assertIn(key, host)

    def test_framing_and_compact_sorted_json(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as sock:
            sock.sendall(f"HELLO nodus-console/1 {TOKEN}\nDIGEST\n".encode())
            reader = sock.makefile("rb")
            self.assertTrue(reader.readline().startswith(b"OK console "))
            head = reader.readline().split()
            self.assertEqual(head[:2], [b"JSON", b"DIGEST"])
            body = reader.read(int(head[2]))
            self.assertEqual(len(body), int(head[2]))
            document = json.loads(body)
            self.assertEqual(body, json.dumps(document, sort_keys=True,
                                              separators=(",", ":")).encode())

    def test_wrong_token_refused_before_any_verb(self) -> None:
        reply = _raw(self.port, b"HELLO nodus-console/1 wrong", b"DIGEST")
        self.assertTrue(reply.startswith(b"REFUSED bad token"), reply)
        self.assertNotIn(b"JSON", reply)
        reply = _raw(self.port, b"DIGEST")
        self.assertTrue(reply.startswith(b"REFUSED bad hello"), reply)
        with self.assertRaises(ConsoleError):
            ConsoleClient("127.0.0.1", self.port, "wrong").hello()

    def test_unknown_verb_refused(self) -> None:
        reply = _raw(self.port, f"HELLO nodus-console/1 {TOKEN}".encode(), b"WRITE x", read=64)
        self.assertIn(b"REFUSED read-only\n", reply)
        client = self.client()
        with self.assertRaises(ConsoleError):
            client._json("digest")  # verbs are case sensitive

    def test_verbs_never_mutate(self) -> None:
        client = self.client()
        before = client.snapshot()["canvas"]
        client.digest(), client.graph(), client.tools(), client.activity()
        self.assertEqual(client.snapshot()["canvas"], before)

    def test_client_round_trip_with_context_manager(self) -> None:
        with ConsoleClient("127.0.0.1", self.port, TOKEN) as client:
            self.assertEqual(client.digest()["canvas"], self.canvas_name)
            self.assertEqual(client.graph()["schema"], "nodus.graph-definition.v1")


if __name__ == "__main__":
    unittest.main()
