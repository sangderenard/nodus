"""Files across a Nodus link between the canvas ports of two running canvas hosts (run_canvas link_send_file /
link_receive), docs/MESH_CANVAS_UP_PORTS_2026-11-21.md.

The link's ends are canvas ports: <id>.send (IN) feeds the link table's Input row, and the bulk row's row port "received"
feeds <id>.received (OUT). The receiver reads nothing until the sender has had every file accepted -- the backpressure
case that lost data twice before canvas ports (a host drain filled with pads; bulk-row refusals were dropped) -- and
then every file arrives byte for byte. A spool file from an earlier run is never overwritten.
Needs a nodus_headless library with canvas ports and the link ABI, and the turing checkout (TURING_ROOT).
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import secrets
import socket
import sys
import tempfile
import time
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_run_canvas_link import link_tickets, start_host, stop  # noqa: E402


class RunCanvasLinkFilesTest(unittest.TestCase):
    def test_every_file_arrives_under_backpressure(self) -> None:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="run_canvas_link_files_"))
        key = secrets.token_bytes(32)
        key_file = tmp / "nodus_link.key"
        key_file.write_bytes(key)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        outgoing = tmp / "outgoing"
        outgoing.mkdir()
        sizes = [97, 3, 8 * 1024 * 1024 + 16] + [1024 * 1024 + 7 * i for i in range(21)]     # 24 files, ~29 MiB
        if os.environ.get("LINK_FILES_STRESS"):                         # 80 x 8 MiB: past every buffer
            sizes = [8 * 1024 * 1024 + 16 - i for i in range(80)]
        files = []
        for index, size in enumerate(sizes):
            path = outgoing / f"{index:06d}.rec"
            path.write_bytes(os.urandom(size))
            files.append(path)
        spool = tmp / "spool"
        spool.mkdir()
        earlier = spool / "00000001.arrival"
        earlier.write_bytes(b"an arrival from an earlier run")              # must survive a later receive
        a = b = None
        try:
            a = start_host(tmp, "sending")
            b = start_host(tmp, "receiving")
            self.assertTrue(b[3].action(op="link_host", id="state", prefix="state", port=port,
                                        key_file=str(key_file)).get("handled"))
            tickets = link_tickets.issue(key, "state", 42)
            self.assertTrue(a[3].action(op="link_dial", id="state", host="127.0.0.1", port=port, prefix="state",
                                        control_ticket=tickets["control_ticket"], bulk_ticket=tickets["bulk_ticket"],
                                        secure=False).get("handled"))
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                states = [c.action(op="link_report", id="state")["link"] for c in (a[3], b[3])]
                if all(s and s["state"] == "linked" for s in states):
                    break
                time.sleep(0.2)
            else:
                self.fail(f"not linked: {states}")

            # The receiver reads nothing while the sender is accepted: a full path answers "full", never a loss. Only
            # when the sender is refused does the receiver read one batch -- a slow reader under backpressure.
            pending, refusals, arrived = list(files), 0, []
            deadline = time.monotonic() + float(os.environ.get("LINK_FILES_TIMEOUT", "120"))

            def receive_once():
                got = b[3].action(op="link_receive", id="state", spool=str(spool), max=4)["link_receive"]["arrived"]
                arrived.extend(got)
                return got

            while pending and time.monotonic() < deadline:
                answer = a[3].action(op="link_send_file", id="state", path=str(pending[0]))["link_send_file"]
                if answer["sent"]:
                    pending.pop(0)
                else:
                    refusals += 1
                    if not receive_once():
                        time.sleep(0.05)
            self.assertEqual(pending, [], "the sender was never able to send every file")
            while len(arrived) < len(files) and time.monotonic() < deadline:
                if not receive_once():
                    time.sleep(0.1)
            reports = {name: c.action(op="link_report", id="state")["link"] for name, c in (("sender", a[3]), ("receiver", b[3]))}
            print("files", len(files), "arrived", len(arrived), "sender refusals at the port", refusals,
                  "bulk_refused (retried, not lost)", reports["sender"]["bulk_refused"])
            sent = sorted(hashlib.sha256(f.read_bytes()).hexdigest() for f in files)
            got = sorted(hashlib.sha256(pathlib.Path(x["file"]).read_bytes()).hexdigest() for x in arrived)
            self.assertEqual(got, sent)                                      # every file, byte for byte
            self.assertEqual(earlier.read_bytes(), b"an arrival from an earlier run")
            self.assertNotIn(str(earlier), [x["file"] for x in arrived])
        finally:
            for host in (a, b):
                if host:
                    host[3].close()
                    stop(host[0])
                    host[1].close()


if __name__ == "__main__":
    unittest.main()
