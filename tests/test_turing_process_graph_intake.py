from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


def _environment() -> tuple[Path, Path]:
    runner = os.environ.get("NODUS_PROCESS_GRAPH_RUNNER")
    turing_root = os.environ.get("NODUS_TURING_ROOT")
    if not runner or not turing_root:
        pytest.skip(
            "Nodus runner and Turing root are required",
            allow_module_level=True,
        )
    return Path(runner).resolve(), Path(turing_root).resolve()


RUNNER, TURING_ROOT = _environment()
sys.path.insert(0, str(TURING_ROOT))

# Turing's legacy SSA module creates two helper-ledger files in the process
# working directory when imported. Keep that unrelated import side effect out
# of the Nodus checkout while exercising the sanctioned shallow interpreter.
_original_cwd = Path.cwd()
_import_workdir = tempfile.TemporaryDirectory(prefix="nodus-turing-graph-import-")
try:
    os.chdir(_import_workdir.name)
    from src.compiler.shallow_interpretation import method_to_graph_ir  # noqa: E402
finally:
    os.chdir(_original_cwd)


def _run_product(tmp_path: Path, source: str, inputs: dict[str, list[float]], expected: list[float]):
    original_cwd = Path.cwd()
    try:
        os.chdir(_import_workdir.name)
        product = method_to_graph_ir(
            source, "kernel", filename="process_graph_intake.py"
        )
    finally:
        os.chdir(original_cwd)
    graph_ir = tmp_path / "kernel.graphir"
    values = tmp_path / "kernel.values"
    graph_ir.write_text(product.graph_ir, encoding="utf-8")
    rows = [
        "input " + " ".join(str(value) for value in inputs[name])
        for name in product.inputs
    ]
    rows.append("expect " + " ".join(str(value) for value in expected))
    values.write_text("\n".join(rows) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [str(RUNNER), str(graph_ir), str(values)],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        env=os.environ.copy(),
    )
    return product, completed


def test_turing_process_graph_executes_registered_nodus_tensor_tools(tmp_path):
    product, completed = _run_product(
        tmp_path,
        """
def kernel(left, right, scale):
    return (left - right) * scale
""",
        {
            "left": [9.0, 7.0, -2.0],
            "right": [4.0, 10.0, 3.0],
            "scale": [3.0, -2.0, 0.5],
        },
        [15.0, 6.0, -2.5],
    )

    assert product.inputs == ("left", "right", "scale")
    assert set(product.operations) == {"mul", "sub"}
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "abstract_tensor tools" in completed.stdout
    assert "result matches" in completed.stdout


def test_unresolved_binary_process_node_is_refused_not_guessed(tmp_path):
    product, completed = _run_product(
        tmp_path,
        """
def kernel(left, right):
    return external_operation(left, right)
""",
        {"left": [1.0, 2.0], "right": [3.0, 4.0]},
        [0.0, 0.0],
    )

    assert "call" in product.operations
    assert completed.returncode != 0
    assert "structural node 'call' with arity 2 is not executable" in completed.stderr
