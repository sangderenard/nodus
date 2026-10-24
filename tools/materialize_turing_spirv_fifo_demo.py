"""Publish a Turing shader and save a two-row Vulkan-FIFO Nodus graph."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import pathlib
import subprocess
import sys

NODUS_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NODUS_ROOT))

from nodus_graph import (  # noqa: E402
    ConnectorRef, EdgeDefinition, FifoConfig, Graph, GraphDefinition,
    PortDefinition, deploy_spirv_product_graph,
)

PUBLISHER = """from pathlib import Path
from src.transmogrifier.ssa import SSAValue, Instr, BasicBlock, Function
from src.compiler.ssa_spirv_backend import emit_deployable_compute_shader
from src.compiler.shader_deployment import publish_spirv_product
left = SSAValue(0, 'float32', ())
right = SSAValue(1, 'float32', ())
result = SSAValue(2, 'float32', ())
function = Function('nodus_ssbo_add', [left, right], {'entry': BasicBlock('entry', [
    Instr('add', [left, right], result), Instr('Ret', [], None),
])})
published = publish_spirv_product(
    Path(__import__('sys').argv[1]),
    emit_deployable_compute_shader(function, outputs=[result], name='nodus_ssbo_add'),
)
print(published.manifest_path)
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--turing-root", type=pathlib.Path,
                        default=NODUS_ROOT.parent / "turing")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    publisher = output / "publish_shader.py"
    publisher.write_text(PUBLISHER, encoding="utf-8")
    product_dir = output / "turing-product"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.turing_root.resolve()) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(publisher), str(product_dir)],
        cwd=args.turing_root.resolve(), env=environment,
        capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)

    manifest = product_dir / "nodus_ssbo_add.spirv.json"
    library = output / "module-library"
    previous = pathlib.Path.cwd()
    os.chdir(output)
    graph = Graph()
    loaded_id = ""
    try:
        base, _product, report = deploy_spirv_product_graph(
            graph, manifest, stable_id=0x55B0,
            nodus_root=NODUS_ROOT, module_library_root=library,
        )
        loaded_id = str(report["loaded_tool_id"])
        graph.close()

        source = base.modules[0]
        first_id, second_id = source.stable_id, source.stable_id + 1
        first = replace(source, label="Turing SPIR-V shader A", x=40, y=90,
                        width=420, height=300)
        second = replace(source, stable_id=second_id,
                         label="Turing SPIR-V shader B", x=600, y=90,
                         width=420, height=300)
        connectors = tuple(
            replace(item, reference=ConnectorRef(
                stable_id, item.reference.direction, item.reference.ordinal))
            for stable_id in (first_id, second_id)
            for item in base.connectors
        )
        fifo = replace(base.inputs[0].fifo, slots=4, top_k=0,
                       storage="vulkan")
        definition = GraphDefinition(
            modules=(first, second), connectors=connectors,
            edges=(EdgeDefinition(
                ConnectorRef(first_id, "output", 0),
                ConnectorRef(second_id, "input", 0), fifo=fifo, type_id=0,
            ),),
            inputs=(
                PortDefinition("shader_a_left", ConnectorRef(first_id, "input", 0), fifo),
                PortDefinition("shader_a_right", ConnectorRef(first_id, "input", 1), fifo),
                PortDefinition("shader_b_right", ConnectorRef(second_id, "input", 1), fifo),
            ),
            outputs=(PortDefinition(
                "shader_b_output", ConnectorRef(second_id, "output", 0), fifo),),
        )
        graph = Graph()
        graph.apply_definition(definition)
        if graph.definition() != definition:
            raise RuntimeError("native Vulkan-FIFO graph readback disagrees")
        graph.save(output / "turing-spirv-ssbo-fifo.canvas")
        (output / "graph-definition.json").write_text(
            json.dumps(definition.to_mapping(), indent=2) + "\n", encoding="utf-8")
        (output / "plugin-id.txt").write_text(loaded_id + "\n", encoding="utf-8")
    finally:
        graph.close()
        if loaded_id:
            graph.unload_plugin(loaded_id)
        os.chdir(previous)
    print(output / "turing-spirv-ssbo-fifo.canvas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
