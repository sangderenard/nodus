"""Add one more contract port to an existing canvas file and save the result as a new file.

    python tools/add_contract_port.py --canvas IN.canvas --out OUT.canvas --prefix studio.product \
        --capacity 4 --port 47613 [--loan 7200] [--keep gui.proof:4[:LOAN] ...]

The canvas host keeps only the contract ports its canvas declares (a pair acquired before the load is not a port).
This does what `docs/receipts/gui_contract_port_make_canvas.py` did for the first port, on a loaded canvas (or, with
no --canvas, on an empty one): it acquires the pair, adds a `<prefix> market` and a `<prefix> worker` module holding
the two plugin rows, side by side below every table already there, joins them with the port pair
`<prefix> market.0:result > <prefix> worker.0:result` (the rows' own result ports, applied as the GUI's connect move,
so the rope is laid between them), gives that edge the required FIFO shape, declares the port on it, and saves.
`--keep` names the pairs the input canvas already declares, so that the load can restore them (PREFIX:CAPACITY[:LOAN],
as run_canvas --acquire).
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
from nodus_headless_bridge import CanvasModuleDesc  # noqa: E402

RESULT_TYPE, RESULT_BYTES = 0x534A5231, 64
TABLE_W, TABLE_H, GAP = 420, 320, 60      # Graph.add_module's default table size; the gap leaves the rope visible


def _pair(text: str):
    parts = text.split(":")
    if len(parts) not in (2, 3) or not parts[0]:
        raise argparse.ArgumentTypeError("expected PREFIX:CAPACITY[:LOAN]")
    return parts[0], int(parts[1]), (int(parts[2]) if len(parts) == 3 else None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--canvas", default=None, help="input canvas (omit to start from an empty canvas)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--loan", type=int, default=None, help="loan length in market turns (default 60)")
    parser.add_argument("--keep", action="append", default=[], type=_pair, metavar="PREFIX:CAPACITY[:LOAN]")
    args = parser.parse_args(argv)
    source = pathlib.Path(args.canvas).resolve() if args.canvas else None
    out = pathlib.Path(args.out).resolve()
    if source == out:
        parser.error("--out must be a different file from --canvas")
    with Graph() as graph:
        for prefix, capacity, loan in args.keep:
            graph.acquire_contract_pair(prefix, capacity, loan)
        graph.acquire_contract_pair(args.prefix, args.capacity, args.loan)
        if source is not None:
            graph.load(source)
        if any(prefix == args.prefix for prefix, *_rest in graph.contract_ports()):
            parser.error(f"{source.name if source else 'the canvas'} already declares {args.prefix}")
        # The new pair goes below every table already on the canvas, market and worker side by side.
        bottom = 0
        for index in range(int(graph.lib.nodus_headless_get_module_count(graph._handle))):
            desc = CanvasModuleDesc()
            if graph.lib.nodus_headless_get_module_desc(graph._handle, index, ctypes.byref(desc)):
                bottom = max(bottom, int(desc.y) + int(desc.h))
        y = bottom + GAP if bottom else 0
        market = graph.add_module(f"{args.prefix} market", x=0, y=y, width=TABLE_W, height=TABLE_H)
        worker = graph.add_module(f"{args.prefix} worker", x=TABLE_W + GAP, y=y, width=TABLE_W, height=TABLE_H)
        market.set_rows(PluginTool(args.prefix + ".market", 0, 0))
        worker.set_rows(PluginTool(args.prefix + ".worker", 0, 0))
        fifo = FifoConfig(token_bytes=RESULT_BYTES, slots=args.capacity, top_k=0, type_id=RESULT_TYPE, layout=2, dtype=8)
        edge = graph.apply_port_pair(f"{args.prefix} market.0:result > {args.prefix} worker.0:result", fifo=fifo)
        bound = graph.declare_contract_port(args.prefix, args.capacity, edge, args.port)
        graph.save(out)
        print({"out": str(out), "prefix": args.prefix, "port": bound,
               "ports": [(prefix, port) for prefix, _cap, _edge, port in graph.contract_ports()]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
