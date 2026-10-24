"""Compile a real Turing SSA row and save a visibly wired Nodus graph.

This is a visual-proof fixture for ``demo_canvas_tables.py --canvas``.  The
module rows, connectors, contracts, and named ports are derived through the
same public crossing used by the execution acceptance test.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import pathlib
import sys


NODUS_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NODUS_ROOT))

from nodus_graph import (  # noqa: E402
    ConnectorRef,
    EdgeDefinition,
    Graph,
    GraphDefinition,
    PortDefinition,
    PythonRowPort,
    deploy_row_product_graph,
    write_python_row_descriptor,
)


SOURCE = """def calculate(left, right):
    return (left - right) * (left + right) * 0.5
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--turing-root", type=pathlib.Path,
        default=NODUS_ROOT.parent / "turing",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "turing_graph_demo.py"
    descriptor = output / "turing_graph_demo.pyabi"
    product = output / "product-c"
    module_library = output / "module-library"
    canvas = output / "turing-created-nodus.canvas"

    source.write_text(SOURCE, encoding="utf-8")
    write_python_row_descriptor(
        descriptor,
        tool_id="python.visual_turing_nodus_graph",
        tool_name="Turing SSA: calculate(left, right)",
        source_path=source,
        module_name="visual_turing_nodus_graph",
        callable_name="calculate",
        inputs=(PythonRowPort("left"), PythonRowPort("right")),
        outputs=(PythonRowPort("result"),),
    )

    previous = pathlib.Path.cwd()
    os.chdir(output)
    graph = Graph()
    loaded_id = ""
    try:
        base_definition, report = deploy_row_product_graph(
            graph,
            descriptor,
            target="c",
            stable_id=0x7A11CE,
            nodus_root=NODUS_ROOT,
            module_library_root=module_library,
            python_executable=sys.executable,
            turing_root=args.turing_root,
            extraction_contract=(
                args.turing_root / "extraction_contracts" /
                "program_extraction.yaml"
            ),
            product_dir=product,
        )
        loaded_id = str(report["loaded_tool_id"])
        if graph.definition() != base_definition:
            raise RuntimeError("native graph disagrees with Turing-derived definition")
        graph.close()

        source_module = base_definition.modules[0]
        first_id = source_module.stable_id
        second_id = first_id + 1

        def mapped(reference: ConnectorRef, stable_id: int) -> ConnectorRef:
            return ConnectorRef(stable_id, reference.direction, reference.ordinal)

        first = replace(
            source_module, label="Turing SSA stage 1", x=40, y=80,
            width=400, height=300,
        )
        second = replace(
            source_module, stable_id=second_id, label="Turing SSA stage 2",
            x=580, y=80, width=400, height=300,
        )
        connectors = tuple(
            replace(item, reference=mapped(item.reference, stable_id))
            for stable_id in (first_id, second_id)
            for item in base_definition.connectors
        )
        fifo = base_definition.inputs[0].fifo
        definition = GraphDefinition(
            modules=(first, second),
            connectors=connectors,
            edges=(EdgeDefinition(
                ConnectorRef(first_id, "output", 0),
                ConnectorRef(second_id, "input", 0),
                # The native canvas compatibility selector is the plugin's
                # declared primitive port type.  The FIFO independently keeps
                # the communicated AbstractTensor pointer payload contract.
                fifo=fifo, type_id=0,
            ),),
            inputs=(
                PortDefinition("stage1_left", ConnectorRef(first_id, "input", 0), fifo),
                PortDefinition("stage1_right", ConnectorRef(first_id, "input", 1), fifo),
                PortDefinition("stage2_right", ConnectorRef(second_id, "input", 1), fifo),
            ),
            outputs=(
                PortDefinition("stage2_result", ConnectorRef(second_id, "output", 0), fifo),
            ),
        )
        graph = Graph()
        graph.apply_definition(definition)
        if graph.definition() != definition:
            raise RuntimeError("two-stage native graph disagrees with communicated definition")
        graph.save(canvas)
        (output / "graph-definition.json").write_text(
            json.dumps(definition.to_mapping(), indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        graph.close()
        if loaded_id:
            graph.unload_plugin(loaded_id)
        os.chdir(previous)

    print(canvas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
