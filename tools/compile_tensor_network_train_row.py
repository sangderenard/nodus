"""Compile the tensor network's training step as a stateful Nodus row.

examples/tensor_network_train_row.py holds the loss. The row-tool compiler's training request records it and hands
it to Turing's compile_native_training_schedule, which derives the backward graph and the native SGD update. The
product is an ordinary row-tool product (manifest, NODUS BUFFER ABI V2, LLVM library, adoption receipt). The four
weights, the step count and the learning rate are STATE: the row owns them across calls, they load through the row's
ports and they save with the canvas.

--reference also writes <output>/reference.txt: a batch, initial weights, and the weights after --reference-steps
steps of train_aapl_qualified_row.PolicyNetwork.step (AbstractTensor autograd, Python). tests/training_row_test.cpp
runs the adopted row on that batch and compares.

Stage 1 of the nodus-training move (docs/NODUS_NATIVE_HOST_PLAN_2026-11-23.md).

    python tools/compile_tensor_network_train_row.py <output> --turing-root ../turing --reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ENTRY = "tensor_network_train_loss"
PARAMETERS = ("input_weight", "hidden_bias", "output_weight", "output_bias")


def request(output: Path, turing_root: Path, nodus_root: Path, *, batch: int, learning_rate: float) -> dict:
    return {
        "source_path": str(nodus_root / "examples" / "tensor_network_train_row.py"),
        "row_entry": ENTRY,
        "product_name": "tensor_network_train_step",
        "backend": "llvm",
        "output_dir": str(output / "product"),
        "turing_root": str(turing_root),
        "allow_uncompiled": False,
        "training": {"learning_rate": learning_rate, "steps": 1},
        "columns": [
            {"name": "features", "role": "input", "dtype": "f64", "shape": [batch, 6]},
            {"name": "targets", "role": "input", "dtype": "f64", "shape": [batch, 3]},
            {"name": "input_weight", "role": "state", "dtype": "f64", "shape": [6, 4]},
            {"name": "hidden_bias", "role": "state", "dtype": "f64", "shape": [1, 4]},
            {"name": "output_weight", "role": "state", "dtype": "f64", "shape": [4, 3]},
            {"name": "output_bias", "role": "state", "dtype": "f64", "shape": [1, 3]},
            {"name": "loss", "role": "output", "dtype": "f64", "shape": []},
            {"name": "prediction", "role": "output", "dtype": "f64", "shape": [batch, 3]},
        ],
    }


def reference(path: Path, *, batch: int, learning_rate: float, steps: int) -> None:
    import numpy as np
    from src.common.tensors import AbstractTensor as AT
    from tools.train_aapl_qualified_row import PolicyNetwork

    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, (batch, 6))
    target = rng.uniform(-1, 1, (batch, 3))
    network = PolicyNetwork(AT, seed=0)
    initial = [np.asarray(p.numpy(), dtype=np.float64).copy() for p in network.parameters]
    losses = [float(network.step(x, target, learning_rate=learning_rate)) for _ in range(steps)]
    final = [np.asarray(p.numpy(), dtype=np.float64) for p in network.parameters]
    # One record per line, "<name> <count> <values...>", values as Python repr (round-trips a double exactly).
    records = [("steps", [steps]), ("features", x.ravel()), ("targets", target.ravel()),
               ("first_loss", [losses[0]]), ("last_loss", [losses[-1]])]
    records += [(f"initial.{name}", value.ravel()) for name, value in zip(PARAMETERS, initial)]
    records += [(f"final.{name}", value.ravel()) for name, value in zip(PARAMETERS, final)]
    path.write_text("".join(f"{name} {len(values)} " + " ".join(repr(float(v)) for v in values) + "\n"
                            for name, values in records), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--turing-root", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--reference-steps", type=int, default=50)
    args = parser.parse_args()
    nodus_root = Path(__file__).resolve().parents[1]
    turing_root = args.turing_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(nodus_root)]
    from tools.turing_row_tool_compiler import compile_row_tool

    manifest = compile_row_tool(request(output, turing_root, nodus_root,
                                        batch=args.batch, learning_rate=args.learning_rate))
    print(json.dumps({"manifest": str(output / "product" / "nodus-row-tool-product.json"),
                      "loadable": manifest["loadable"], "tool": manifest["buffer_abi"]["tool_id"]}))
    if args.reference:
        sys.path.insert(0, str(turing_root))
        reference(output / "reference.txt", batch=args.batch, learning_rate=args.learning_rate,
                  steps=args.reference_steps)
        print(json.dumps({"reference": str(output / "reference.txt")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
