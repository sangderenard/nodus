"""Canvas host process: load a saved canvas and step it at a fixed rate.

Prints one JSON line describing the bound contract ports, steps until
SIGINT/SIGTERM or --ticks, then prints {"ticks": n}.  Errors go to stderr, exit 2.

The native library prints diagnostics on fd 1, so fd 1 is redirected to stderr and the
JSON lines are written to a saved duplicate of the real stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nodus_graph import Graph  # noqa: E402

_stop = False


def _on_signal(signum, frame) -> None:
    global _stop
    _stop = True


def _parse_acquire(text: str):
    prefix, sep, cap = text.rpartition(":")
    if not sep or not prefix or not cap.isdigit():
        raise ValueError(f"--acquire expects PREFIX:CAPACITY, got {text!r}")
    return prefix, int(cap)


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
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    try:
        if args.rate <= 0:
            raise ValueError("--rate must be positive")
        acquires = [_parse_acquire(a) for a in args.acquire]
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _on_signal)
        out = _claim_stdout()
        period = 1.0 / args.rate
        ticks = 0
        with Graph() as graph:
            for prefix, capacity in acquires:
                graph.acquire_contract_pair(prefix, capacity)
            graph.load(args.canvas)
            ports = []
            for prefix, capacity, edge, declared in graph.contract_ports():
                bound = graph.contract_port(prefix)
                ports.append({"prefix": prefix,
                              "listen_port": bound[0] if bound else declared,
                              "capacity": capacity, "edge": edge})
            print(json.dumps({"canvas": args.canvas, "contract_ports": ports,
                              "pid": os.getpid()}), file=out, flush=True)
            next_t = time.perf_counter()
            while not _stop and (args.ticks is None or ticks < args.ticks):
                graph.step(period)
                ticks += 1
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()
        print(json.dumps({"ticks": ticks}), file=out, flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"run_canvas: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
