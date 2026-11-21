"""Files across a Nodus link, through the canvas (run_canvas link_send_file / link_receive).

Two running canvas hosts link (as test_run_canvas_link.py); the dialer sends files, each one U8 bulk tensor through the
declared port <id>.send; on the host, arrivals leave the bulk row on its row port "received", cross a canvas edge into
the table "link <id> received" and reach the host through the declared output port <id>.received, which link_receive
spools as files. Every file arrives, in order, byte for byte. This is the transport of the regional state broadcast
(turing examples/meowdoku/state_broadcast.py: one record file per tensor).
Needs a nodus_headless library with the link ABI and the turing checkout beside this repo (or TURING_ROOT).
"""
from __future__ import annotations

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
    def test_files_arrive_in_order_byte_for_byte(self) -> None:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="run_canvas_link_files_"))
        key = secrets.token_bytes(32)
        key_file = tmp / "nodus_link.key"
        key_file.write_bytes(key)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        outgoing = tmp / "outgoing"
        outgoing.mkdir()
        sizes = [97, 8 * 1024 * 1024 + 16, 3]                    # a header, one full sealed chunk, a short tail
        files = []
        for index, size in enumerate(sizes):
            path = outgoing / f"{index:06d}.rec"
            path.write_bytes(os.urandom(size))
            files.append(path)
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

            pending = list(files)
            spool = tmp / "spool"
            arrived, pads = [], []
            deadline = time.monotonic() + 60.0
            while (pending or len(arrived) < len(files)) and time.monotonic() < deadline:
                if pending:
                    answer = a[3].action(op="link_send_file", id="state", path=str(pending[0]))["link_send_file"]
                    if answer["sent"]:
                        pending.pop(0)
                answer = b[3].action(op="link_receive", id="state", spool=str(spool))["link_receive"]
                got = answer["arrived"]
                pads.append(answer["waiting"])
                arrived.extend(got)
                if not got and not pending:
                    time.sleep(0.1)
            reports = {name: c.action(op="link_report", id="state")["link"] for name, c in (("sender", a[3]), ("receiver", b[3]))}
            print("received:", [x["bytes"] for x in arrived], "waiting per pull:", pads[:12], "reports:", reports)
            self.assertEqual(len(arrived), len(files), arrived)
            for sent, landed in zip(files, arrived):
                self.assertEqual(pathlib.Path(landed["file"]).read_bytes(), sent.read_bytes(), landed)
        finally:
            for host in (a, b):
                if host:
                    host[3].close()
                    stop(host[0])
                    host[1].close()


if __name__ == "__main__":
    unittest.main()
