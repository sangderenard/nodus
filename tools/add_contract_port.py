"""Add one more contract port to an existing canvas file and save the result as a new file.

    python tools/add_contract_port.py --canvas IN.canvas --out OUT.canvas --prefix studio.product \
        --capacity 4 --port 47613 [--loan 7200] [--keep gui.proof:4[:LOAN] ...]

The canvas host keeps only the contract ports its canvas declares (a pair acquired before the load is not a port).
This does what `docs/receipts/gui_contract_port_make_canvas.py` did for the first port, on a loaded canvas: it
acquires the pair, adds a `<prefix> market` and a `<prefix> worker` module holding the two plugin rows, joins them
with a result edge of the required shape, declares the port on that edge, and saves. `--keep` names the pairs the
input canvas already declares, so that the load can restore them (PREFIX:CAPACITY[:LOAN], as run_canvas --acquire).
The input file is never written. The listen ports of the kept pairs are bound while this runs, so run it while the
canvas host that owns them is stopped, or on a copy whose ports differ.
"""
from __future__ import annotations

import argparse
import ctypes
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]

from nodus_graph import FifoConfig, Graph, PluginTool  # noqa: E402
from nodus_graph.native import _fifo  # noqa: E402
from nodus_headless_bridge import CanvasEdgeDesc  # noqa: E402

RESULT_TYPE, RESULT_BYTES = 0x534A5231, 64


def _pair(text: str):
    parts = text.split(":")
    if len(parts) not in (2, 3) or not parts[0]:
        raise argparse.ArgumentTypeError("expected PREFIX:CAPACITY[:LOAN]")
    return parts[0], int(parts[1]), (int(parts[2]) if len(parts) == 3 else None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--canvas", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--loan", type=int, default=None, help="loan length in market turns (default 60)")
    parser.add_argument("--keep", action="append", default=[], type=_pair, metavar="PREFIX:CAPACITY[:LOAN]")
    args = parser.parse_args(argv)
    source, out = pathlib.Path(args.canvas).resolve(), pathlib.Path(args.out).resolve()
    if source == out:
        parser.error("--out must be a different file from --canvas")
    with Graph() as graph:
        for prefix, capacity, loan in args.keep:
            graph.acquire_contract_pair(prefix, capacity, loan)
        graph.acquire_contract_pair(args.prefix, args.capacity, args.loan)
        graph.load(source)
        if any(prefix == args.prefix for prefix, *_rest in graph.contract_ports()):
            parser.error(f"{source.name} already declares {args.prefix}")
        market = graph.add_module(f"{args.prefix} market")
        worker = graph.add_module(f"{args.prefix} worker")
        market.set_rows(PluginTool(args.prefix + ".market", 0, 0))
        worker.set_rows(PluginTool(args.prefix + ".worker", 0, 0))
        desc = CanvasEdgeDesc(a_module=market.index, a_contact_idx=0, b_module=worker.index, b_contact_idx=0)
        edge = graph.lib.nodus_headless_add_edge(graph._handle, ctypes.byref(desc), 0)
        if edge < 0:
            raise RuntimeError("the result edge was refused")
        fifo = FifoConfig(token_bytes=RESULT_BYTES, slots=args.capacity, top_k=0, type_id=RESULT_TYPE, layout=2, dtype=8)
        if not graph.lib.nodus_headless_configure_edge_fifo(graph._handle, edge, ctypes.byref(_fifo(fifo))):
            raise RuntimeError("the result edge's FIFO was refused")
        bound = graph.declare_contract_port(args.prefix, args.capacity, edge, args.port)
        graph.save(out)
        print({"out": str(out), "prefix": args.prefix, "port": bound,
               "ports": [(prefix, port) for prefix, _cap, _edge, port in graph.contract_ports()]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
