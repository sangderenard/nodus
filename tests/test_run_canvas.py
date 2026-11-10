"""tools/run_canvas.py hosts a saved messenger canvas; a peer round trip works against it."""

from __future__ import annotations

import ctypes
import json
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from nodus_graph import FifoConfig, Graph, PluginTool  # noqa: E402
from nodus_graph.native import _fifo  # noqa: E402
from nodus_headless_bridge import CanvasEdgeDesc  # noqa: E402
from contract_service_peer import (  # noqa: E402
    EnginePeer, ServicePeer, connect_loopback, ROLE_ENGINE,
)

PREFIX = "py.runcanvas"
CAPACITY = 4


def save_messenger_canvas(path: pathlib.Path) -> None:
    with Graph() as graph:
        graph.acquire_contract_pair(PREFIX, CAPACITY)
        market = graph.add_module("market")
        worker = graph.add_module("worker")
        market.set_rows(PluginTool(PREFIX + ".market", 0, 0))
        worker.set_rows(PluginTool(PREFIX + ".worker", 0, 0))
        desc = CanvasEdgeDesc(a_module=market.index, a_contact_idx=0,
                              b_module=worker.index, b_contact_idx=0)
        edge_id = graph.lib.nodus_headless_add_edge(graph._handle, ctypes.byref(desc), 0)
        assert edge_id >= 0
        fifo = FifoConfig(token_bytes=64, slots=4, top_k=0,
                          type_id=0x534A5231, layout=2, dtype=8)
        assert graph.lib.nodus_headless_configure_edge_fifo(
            graph._handle, edge_id, ctypes.byref(_fifo(fifo)))
        assert graph.declare_contract_port(PREFIX, CAPACITY, edge_id) > 0
        graph.save(path)


class RunCanvasTest(unittest.TestCase):
    def test_host_round_trip(self) -> None:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="run_canvas_"))
        canvas = tmp / "messenger.nodus.txt"
        save_messenger_canvas(canvas)
        err_path = tmp / "stderr.txt"
        err_file = open(err_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "tools" / "run_canvas.py"), "--canvas", str(canvas),
             "--rate", "500", "--acquire", f"{PREFIX}:{CAPACITY}", "--ticks", "2000"],
            stdout=subprocess.PIPE, stderr=err_file, text=True,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        try:
            first = json.loads(proc.stdout.readline())
            self.assertEqual(first["pid"], proc.pid)
            self.assertEqual(len(first["contract_ports"]), 1)
            entry = first["contract_ports"][0]
            self.assertEqual((entry["prefix"], entry["capacity"]), (PREFIX, CAPACITY))
            port = entry["listen_port"]
            self.assertGreater(port, 0)

            events = {"admitted": [], "routed": [], "receipts": []}

            class Producer:
                def admitted(self, identifier):
                    events["admitted"].append(identifier)

                def route(self, result):
                    events["routed"].append(result)
                    return 1

            engine = EnginePeer(connect_loopback(port, ROLE_ENGINE, 7), Producer())
            conn = socket.create_connection(("127.0.0.1", port), timeout=5.0)
            with ThreadPoolExecutor(max_workers=1) as executor:
                host = ServicePeer(
                    conn, executor, lambda loan: 555,
                    lambda loan, accepted: events["receipts"].append(accepted),
                    announce=True, identity=1)
                self.assertTrue(engine.submit((100, 7, 3, 4)))
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    engine.poll()
                    host.poll()
                    if events["admitted"] and events["routed"] and events["receipts"]:
                        break
                    time.sleep(0.002)
                self.assertIsNone(engine.error)
                self.assertIsNone(host.error)
                self.assertEqual(events["admitted"], [100])
                self.assertEqual(len(events["routed"]), 1)
                self.assertEqual(events["routed"][0][7], 555)
                self.assertEqual(events["receipts"], [True])
            engine.close()
            host.close()

            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # terminate() would skip the handler
            else:
                proc.terminate()
            try:
                out, _ = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                self.fail("run_canvas did not exit after terminate")
            lines = [ln for ln in out.splitlines() if ln.strip()]
            self.assertEqual(proc.returncode, 0, err_path.read_text(errors='replace')[-2000:])
            last = json.loads(lines[-1])
            self.assertIn("ticks", last)
            self.assertGreater(last["ticks"], 0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            err_file.close()


if __name__ == "__main__":
    unittest.main()
