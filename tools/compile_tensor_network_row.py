"""Compile the two-layer AbstractTensor inference example as a Nodus row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from tools.turing_row_tool_compiler import compile_row_tool


ENTRY = "tensor_network_row"
TENSOR_TYPE = "src.common.tensors.abstraction.AbstractTensor"
SHAPES = {
    "features": (2, 6),
    "input_weight": (6, 4),
    "hidden_bias": (2, 4),
    "output_weight": (4, 3),
    "output_bias": (2, 3),
    "prediction": (2, 3),
}


def request(output: Path, turing_root: Path, nodus_root: Path) -> dict:
    output = output.resolve()
    turing_root = turing_root.resolve()
    nodus_root = nodus_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = turing_root / "extraction_contracts" / "program_extraction.yaml"
    contract = yaml.safe_load(base.read_text(encoding="utf-8"))
    contract["program_abi"] = {
        "records": {}, "bindings": [],
        "values": [
            {"function": ENTRY, "parameter": name, "storage": "span",
             "dtype": "float64", "rank": 2, "shape": list(shape),
             "python_type": TENSOR_TYPE}
            for name, shape in SHAPES.items() if name != "prediction"
        ],
    }
    contract_path = output / "extraction-contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    columns = [
        {"name": name, "role": "output" if name == "prediction" else "input",
         "dtype": "f64", "rank": 2, "shape": list(shape)}
        for name, shape in SHAPES.items()
    ]
    return {
        "source_path": str(nodus_root / "examples" / "tensor_network_row.py"),
        "row_entry": ENTRY,
        "columns": columns,
        "backend": "c",
        "output_dir": str(output / "product-c"),
        "product_name": ENTRY,
        "turing_root": str(turing_root),
        "extraction_contract": str(contract_path),
        "allow_uncompiled": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--turing-root", type=Path, required=True)
    args = parser.parse_args()
    nodus_root = Path(__file__).resolve().parents[1]
    row_request = request(args.output, args.turing_root, nodus_root)
    request_path = args.output.resolve() / "request.json"
    request_path.write_text(json.dumps(row_request, indent=2), encoding="utf-8")
    manifest = compile_row_tool(row_request)
    print(json.dumps({"manifest": str(Path(row_request["output_dir"]) / "nodus-row-tool-product.json"),
                      "loadable": manifest["loadable"], "backend": manifest["backend"]}, indent=2))


if __name__ == "__main__":
    main()
