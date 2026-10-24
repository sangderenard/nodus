"""Real runtime-shaped Turing C rows through the Nodus graph boundary."""

from __future__ import annotations

import json
import math
import pathlib
import shutil
import struct
import sys
import tempfile

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nodus_graph import (
    Graph,
    GraphDefinition,
    compile_row_product_graph,
    pack_tensor_handle,
    unpack_tensor_handle,
)


SOURCE = "def calculate(left, right):\n    return (left - right) * (left + right) * 0.5\n"
TENSOR_TYPE = "src.common.tensors.abstraction.AbstractTensor"


def write_contract(base: pathlib.Path, path: pathlib.Path, rank: int) -> None:
    value = yaml.safe_load(base.read_text(encoding="utf-8"))
    value["program_abi"] = {
        "records": {},
        "bindings": [],
        "values": [
            {
                "function": "calculate", "parameter": name,
                "storage": "span", "dtype": "float64", "rank": rank,
                "python_type": TENSOR_TYPE,
            }
            for name in ("left", "right")
        ],
    }
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def run_rank(
    backend: str,
    rank: int,
    shape: tuple[int, ...],
    *,
    right_shape: tuple[int, ...] | None = None,
    declare_right_shape_from_left: bool = True,
    expect_validation_failure: bool = False,
) -> None:
    nodus_root = pathlib.Path(__file__).resolve().parent.parent
    turing_root = nodus_root.parent / "turing"
    root = pathlib.Path(tempfile.mkdtemp(
        prefix=f"nodus_turing_runtime_{backend}_rank{rank}_"
    ))
    product = root / f"product-{backend}"
    module_library = root / "module-library"
    contract = root / "contract.yaml"
    request_path = root / "request.json"
    write_contract(
        turing_root / "extraction_contracts" / "program_extraction.yaml",
        contract, rank,
    )
    right_column = {
        "name": "right", "role": "input", "dtype": "f64", "rank": rank,
        "shape": None,
    }
    if declare_right_shape_from_left:
        right_column["shape_from"] = "left"
    columns = [
        {"name": "left", "role": "input", "dtype": "f64", "rank": rank, "shape": None},
        right_column,
        {"name": "result", "role": "output", "dtype": "f64", "rank": rank,
         "shape": None, "shape_from": "left"},
    ]
    request_path.write_text(json.dumps({
        "source_text": SOURCE,
        "row_entry": "calculate",
        "columns": columns,
        "backend": backend,
        "output_dir": product.as_posix(),
        "product_name": f"runtime_{backend}_rank{rank}",
        "turing_root": turing_root.as_posix(),
        "extraction_contract": contract.as_posix(),
        "allow_uncompiled": False,
        "ssa_composition": {"directory": (root / "ssa-composition").as_posix(), "options": {}},
    }, indent=2), encoding="utf-8")

    graph = Graph()
    loaded_id = ""
    handles: list[int] = []
    try:
        definition, report = compile_row_product_graph(
            graph, request_path, stable_id=0xD000 + rank,
            python_executable=sys.executable, turing_root=turing_root,
            product_dir=product, nodus_root=nodus_root,
            module_library_root=module_library,
        )
        loaded_id = str(report["loaded_tool_id"])
        assert graph.definition() == definition
        assert GraphDefinition.from_mapping(definition.to_mapping()) == definition
        assert all(item.tensor.shape == (None,) * rank for item in definition.connectors)

        count = math.prod(shape)
        actual_right_shape = right_shape or shape
        right_count = math.prod(actual_right_shape)
        left_values = tuple(float(i + 1) for i in range(count))
        right_values = tuple(float((i % 3) + 0.25) for i in range(right_count))
        left = graph.create_tensor(2, shape, struct.pack(f"={count}d", *left_values))
        right = graph.create_tensor(
            2, actual_right_shape,
            struct.pack(f"={right_count}d", *right_values),
        )
        handles.extend((left, right))
        graph.push_bytes("left", pack_tensor_handle(left))
        graph.push_bytes("right", pack_tensor_handle(right))
        graph.step()
        output_payload = graph.pull_bytes("result", 8)
        if expect_validation_failure:
            assert output_payload is None
            return
        assert output_payload is not None
        output = unpack_tensor_handle(output_payload)
        handles.append(output)
        assert graph.describe_tensor(output)["shape"] == shape
        actual = struct.unpack(f"={count}d", graph.read_tensor(output))
        expected = tuple(
            (left_value - right_value) * (left_value + right_value) * 0.5
            for left_value, right_value in zip(left_values, right_values)
        )
        for got, want in zip(actual, expected):
            assert math.isclose(got, want, rel_tol=1.0e-12, abs_tol=1.0e-12)
    finally:
        for handle in reversed(handles):
            graph.destroy_tensor(handle)
        graph.close()
        if loaded_id:
            graph.unload_plugin(loaded_id)
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    run_rank("c", 1, (7,))
    run_rank("c", 2, (2, 3))
    run_rank("llvm", 1, (7,))
    run_rank("llvm", 2, (2, 3))
    run_rank(
        "llvm", 2, (2, 3), right_shape=(3, 2),
        declare_right_shape_from_left=False,
        expect_validation_failure=True,
    )
    print("turing_nodus_runtime_shape_crossing_test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
