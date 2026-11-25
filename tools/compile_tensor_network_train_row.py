"""Compile the tensor network's training step natively.

examples/tensor_network_train_row.py holds the loss. It is recorded on an
SSATensorProgram and handed to Turing's compile_native_training_schedule,
which derives the backward pass and compiles one LLVM library whose stepping
entry runs forward, loss, backward and the SGD update over the four weights.
The schedule is saved as <output>/schedule.json.

--check runs the native step beside train_aapl_qualified_row.PolicyNetwork.step
from the same weights and batch and prints the loss and weight differences.

Stage 1 of the nodus-training move (docs/NODUS_NATIVE_HOST_PLAN_2026-11-23.md).
Adopting the schedule as a Nodus row is not built yet.

    python tools/compile_tensor_network_train_row.py <output> --turing-root ../turing --check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ENTRY = "tensor_network_train_step"
PARAMETERS = ("input_weight", "hidden_bias", "output_weight", "output_bias")


def shapes(batch: int) -> tuple[tuple[str, tuple[int, int]], ...]:
    """Binding order is the value-id order the recorded program assigns."""
    return (("features", (batch, 6)), ("targets", (batch, 3)),
            ("input_weight", (6, 4)), ("hidden_bias", (1, 4)),
            ("output_weight", (4, 3)), ("output_bias", (1, 3)))


def _loss_function(nodus_root: Path):
    path = nodus_root / "examples" / "tensor_network_train_row.py"
    spec = importlib.util.spec_from_file_location("tensor_network_train_row", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.tensor_network_train_loss


def compile_schedule(output: Path, nodus_root: Path, *, batch: int, learning_rate: float):
    from src.common.tensors.accelerator_backends.ssa_backend import (
        SSATensorOperations, SSATensorProgram)
    from src.compiler.llvm_training_runtime import (
        NativeParameterGroup, compile_native_training_schedule, save_training_schedule)

    program = SSATensorProgram(ENTRY)
    values = {name: SSATensorOperations.input(program, shape) for name, shape in shapes(batch)}
    ids = {name: int(value.data.value.id) for name, value in values.items()}
    loss, prediction = _loss_function(nodus_root)(*values.values())
    schedule = compile_native_training_schedule(
        loss, bindings=values,
        parameter_groups=(NativeParameterGroup(
            "policy", tuple(ids[name] for name in PARAMETERS), learning_rate),),
        observed_outputs={"prediction": int(prediction.data.value.id)},
        name=ENTRY, directory=output / "artifacts")
    save_training_schedule(schedule, output / "schedule.json")
    (output / "bindings.json").write_text(
        json.dumps({"batch": batch, "learning_rate": learning_rate, "value_ids": ids,
                    "outputs": dict(schedule.outputs)}, indent=2), encoding="utf-8")
    return schedule, ids


def check(schedule, ids, *, batch: int, learning_rate: float, steps: int) -> float:
    import numpy as np
    from src.common.tensors import AbstractTensor as AT
    from src.compiler.llvm_training_runtime import run_parameter_group
    from tools.train_aapl_qualified_row import PolicyNetwork

    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, (batch, 6))
    target = rng.uniform(-1, 1, (batch, 3))
    network = PolicyNetwork(AT, seed=0)
    buffers = {ids["features"]: x.copy(), ids["targets"]: target.copy()}
    buffers.update({ids[name]: np.asarray(p.numpy(), dtype=np.float64).copy()
                    for name, p in zip(PARAMETERS, network.parameters)})
    worst = 0.0
    for step in range(steps):
        python_loss = network.step(x, target, learning_rate=learning_rate)
        execution = run_parameter_group(schedule, "policy", buffers, steps=1)
        native_loss = float(np.asarray(execution.buffers[schedule.outputs["loss_0"]]).ravel()[0])
        difference = max(float(np.max(np.abs(buffers[ids[name]] - np.asarray(p.numpy()))))
                         for name, p in zip(PARAMETERS, network.parameters))
        worst = max(worst, difference)
        if step in (0, steps - 1):
            print(json.dumps({"step": step, "python_loss": repr(python_loss),
                              "native_loss": repr(native_loss), "max_weight_difference": difference}))
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--turing-root", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-steps", type=int, default=50)
    args = parser.parse_args()
    nodus_root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.turing_root.resolve()), str(nodus_root)]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    schedule, ids = compile_schedule(output, nodus_root, batch=args.batch,
                                     learning_rate=args.learning_rate)
    print(json.dumps({"schedule": str(output / "schedule.json"), "outputs": dict(schedule.outputs)}))
    if args.check:
        worst = check(schedule, ids, batch=args.batch, learning_rate=args.learning_rate,
                      steps=args.check_steps)
        print(json.dumps({"steps": args.check_steps, "worst_weight_difference": worst}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
