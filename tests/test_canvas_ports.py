"""Canvas ports (canvas_abi.h GP_CANVAS_PORT_*): named contacts on the canvas's ROOT module, the canvas's own boundary.

docs/MESH_CANVAS_UP_PORTS_2026-11-21.md, stage 1. A headless canvas gets its ROOT module from its first port; an edge
inside the canvas ends at an OUT port and the values arrive there, in order, read through the port (not through a
declared host port on a module); the scheduler's ROOT block leaves the port's edge alone; the port survives save/load.
Needs a nodus_headless library built with canvas ports (NODUS_HEADLESS_LIB).
"""
from __future__ import annotations

import ctypes
import pathlib
import struct
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nodus_graph import Graph  # noqa: E402
from nodus_graph.native import CanvasEdgeDesc, _NativeFifoConfig, _NativeRow  # noqa: E402


def _source_table(graph, label: str) -> int:
    """A table whose Input row is fed by a declared input port and whose Output row is its contact 1."""
    lib = graph.lib
    index = graph.add_module(label, x=40, y=40).index
    rows = (_NativeRow * 2)(_NativeRow(0, 0, 1, 0, None), _NativeRow(2, 0, 1, 0, None))
    assert lib.nodus_headless_set_module_rows(graph._handle, index, rows, 2)
    fifo = _NativeFifoConfig(4, 8, 0, -1, 2, 0, 0)                      # one float a token, lossless
    assert lib.nodus_headless_declare_input_port_ex(graph._handle, b"feed", index, 0, ctypes.byref(fifo))
    return index


def _edge(graph, a, ac, b, bc) -> int:
    desc = CanvasEdgeDesc(a_module=a, a_contact_idx=ac, b_module=b, b_contact_idx=bc)
    edge = int(graph.lib.nodus_headless_add_edge(graph._handle, ctypes.byref(desc), 0))
    fifo = _NativeFifoConfig(4, 8, 0, -1, 2, 0, 0)
    assert edge >= 0 and graph.lib.nodus_headless_configure_edge_fifo(graph._handle, edge, ctypes.byref(fifo))
    return edge


class CanvasPortsTest(unittest.TestCase):
    def test_out_port_carries_an_inside_edge_and_is_saved(self) -> None:
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="canvas_ports_"))
        path = tmp / "ports.nodus.txt"
        with Graph() as graph:
            self.assertEqual(graph.canvas_root_module(), -1)                   # headless: no ROOT until a port
            source = _source_table(graph, "source")
            contact = graph.declare_canvas_port("state.up", "out", up=True)
            root = graph.canvas_root_module()
            self.assertGreaterEqual(root, 0)
            self.assertEqual(contact, 12000)
            self.assertEqual(graph.declare_canvas_port("state.up", "out", up=True), contact)   # idempotent by name
            with self.assertRaises(ValueError):
                graph.declare_canvas_port("state.up", "in")                   # its direction is fixed
            _edge(graph, source, 1, root, contact)

            wanted = [float(v) for v in range(1, 13)]
            pending, got = list(wanted), []
            for _ in range(400):
                if pending and graph.try_push_bytes("feed", struct.pack("f", pending[0])).accepted:
                    pending.pop(0)
                graph.step(1 / 60)
                while True:
                    data = graph.canvas_port_pull("state.up", 4)
                    if data is None:
                        break
                    value = struct.unpack("f", data)[0]
                    if value != 0.0:                                      # an Input row pads an empty edge with 0
                        got.append(value)
                if not pending and len(got) >= len(wanted):
                    break
            self.assertEqual(got, wanted)
            self.assertEqual(graph.canvas_ports(), [{"name": "state.up", "direction": "out", "contact": contact, "up": True}])
            graph.save(path)

        text = path.read_text(encoding="utf-8", errors="replace")
        self.assertIn(f"CANVAS_PORT {contact} 0 1 state.up", text)
        with Graph() as again:
            again.load(path)
            self.assertGreaterEqual(again.canvas_root_module(), 0)
            self.assertEqual(again.canvas_ports(), [{"name": "state.up", "direction": "out", "contact": contact, "up": True}])


if __name__ == "__main__":
    unittest.main()
