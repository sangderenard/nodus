"""Compile, install, and execute the two-layer Turing row in Nodus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from nodus_graph import Graph, compile_row_product_graph, pack_tensor_handle, unpack_tensor_handle
from tools.compile_tensor_network_row import SHAPES, request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--turing-root", type=Path, required=True)
    parser.add_argument("--native-nodus-root", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    nodus_root = Path(__file__).resolve().parents[1]
    native_nodus_root = (args.native_nodus_root or nodus_root).resolve()
    row_request = request(output, args.turing_root, nodus_root)
    request_path = output / "request.json"
    request_path.write_text(json.dumps(row_request, indent=2), encoding="utf-8")

    graph = Graph()
    handles: list[int] = []
    loaded_id = ""
    try:
        definition, report = compile_row_product_graph(
            graph, request_path, stable_id=0x4E45544E,
            python_executable=sys.executable, turing_root=args.turing_root,
            product_dir=row_request["output_dir"], nodus_root=native_nodus_root,
            module_library_root=output / "module-library",
        )
        loaded_id = str(report["loaded_tool_id"])
        assert graph.definition() == definition
        values = {
            "features": np.arange(12, dtype=np.float64).reshape(SHAPES["features"]) / 20,
            "input_weight": np.arange(24, dtype=np.float64).reshape(SHAPES["input_weight"]) / 60,
            "hidden_bias": np.full(SHAPES["hidden_bias"], 0.03),
            "output_weight": np.arange(12, dtype=np.float64).reshape(SHAPES["output_weight"]) / 40,
            "output_bias": np.full(SHAPES["output_bias"], -0.04),
        }
        for name, value in values.items():
            handle = graph.create_tensor(2, value.shape, value.tobytes(order="C"))
            handles.append(handle)
            graph.push_bytes(name, pack_tensor_handle(handle))
        graph.step()
        payload = graph.pull_bytes("prediction", 8)
        if payload is None:
            raise AssertionError("compiled row published no prediction")
        prediction_handle = unpack_tensor_handle(payload)
        handles.append(prediction_handle)
        actual = np.frombuffer(graph.read_tensor(prediction_handle), dtype=np.float64).reshape(SHAPES["prediction"])

        sys.path.insert(0, str(Path(args.turing_root).resolve()))
        from src.common.tensors import AbstractTensor as AT

        with AT.use_backend("numpy"), AT.autograd.no_grad():
            tensors = {name: AT.tensor(value) for name, value in values.items()}
            hidden = (tensors["features"] @ tensors["input_weight"] + tensors["hidden_bias"]).tanh()
            expected = ((hidden @ tensors["output_weight"] + tensors["output_bias"]).tanh()).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        print(json.dumps({"loaded_tool_id": loaded_id, "rows": len(definition.modules[0].rows),
                          "shape": list(actual.shape), "max_abs_error": float(np.max(np.abs(actual - expected))),
                          "adoption": report}, indent=2))
    finally:
        for handle in reversed(handles):
            graph.destroy_tensor(handle)
        graph.close()
        if loaded_id:
            graph.unload_plugin(loaded_id)


if __name__ == "__main__":
    main()
