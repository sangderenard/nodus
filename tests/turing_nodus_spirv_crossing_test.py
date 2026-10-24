"""A real Turing-published SPIR-V product executing as an ordinary Nodus row."""

from __future__ import annotations

import math
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nodus_graph import (
    Graph,
    GraphDefinition,
    deploy_spirv_product_graph,
    pack_tensor_handle,
    unpack_tensor_handle,
)


def publish_product(root: pathlib.Path) -> pathlib.Path:
    nodus_root = pathlib.Path(__file__).resolve().parent.parent
    turing_root = nodus_root.parent / "turing"
    product = root / "turing-product"
    publisher = root / "publish.py"
    publisher.write_text(
        """from pathlib import Path
from src.transmogrifier.ssa import SSAValue, Instr, BasicBlock, Function
from src.compiler.ssa_spirv_backend import emit_deployable_compute_shader
from src.compiler.shader_deployment import publish_spirv_product
left = SSAValue(0, 'float32', ())
right = SSAValue(1, 'float32', ())
result = SSAValue(2, 'float32', ())
function = Function('nodus_spirv_add', [left, right], {'entry': BasicBlock('entry', [
    Instr('add', [left, right], result), Instr('Ret', [], None),
])})
published = publish_spirv_product(
    Path(__import__('sys').argv[1]),
    emit_deployable_compute_shader(function, outputs=[result], name='nodus_spirv_add'),
)
print(published.manifest_path)
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(turing_root) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(publisher), str(product)], cwd=turing_root,
        capture_output=True, text=True, env=environment,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    return product / "nodus_spirv_add.spirv.json"


def main() -> int:
    nodus_root = pathlib.Path(__file__).resolve().parent.parent
    root = pathlib.Path(tempfile.mkdtemp(prefix="nodus_turing_spirv_crossing_"))
    module_library = root / "module-library"
    previous = pathlib.Path.cwd()
    os.chdir(root)
    graph = Graph()
    loaded_id = ""
    handles: list[int] = []
    try:
        definition, product, report = deploy_spirv_product_graph(
            graph, publish_product(root), stable_id=0x5A17,
            nodus_root=nodus_root, module_library_root=module_library,
        )
        loaded_id = str(report["loaded_tool_id"])
        assert graph.definition() == definition
        assert GraphDefinition.from_mapping(definition.to_mapping()) == definition
        assert [connector.name for connector in definition.connectors] == [
            "feed_0", "feed_1", "output_0",
        ]
        assert [connector.tensor.shape for connector in definition.connectors] == [
            (None,), (None,), (None,),
        ]
        assert product.launch.invocations_per_group == 64

        # 1000 is deliberately not divisible by the published local size.
        # The generated row launches one additional full workgroup; its padded
        # sentinel check proves the shader's published past-end guard held.
        count = 1000
        left_values = tuple(float(index) * 0.25 for index in range(count))
        right_values = tuple(float((index % 17) - 8) for index in range(count))
        pack = lambda values: struct.pack(f"={count}f", *values)
        left = graph.create_tensor(1, (count,), pack(left_values))
        right = graph.create_tensor(1, (count,), pack(right_values))
        handles.extend((left, right))
        graph.push_bytes("feed_0", pack_tensor_handle(left))
        graph.push_bytes("feed_1", pack_tensor_handle(right))
        graph.step()
        payload = graph.pull_bytes("output_0", 8)
        assert payload is not None
        output = unpack_tensor_handle(payload)
        handles.append(output)
        assert graph.describe_tensor(output)["shape"] == (count,)
        actual = struct.unpack(f"={count}f", graph.read_tensor(output))
        for index, (got, left_value, right_value) in enumerate(zip(actual, left_values, right_values)):
            want = left_value + right_value
            assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-6), (
                index, got, want, actual[:8]
            )
    finally:
        for handle in reversed(handles):
            graph.destroy_tensor(handle)
        graph.close()
        if loaded_id:
            graph.unload_plugin(loaded_id)
        os.chdir(previous)
        shutil.rmtree(root, ignore_errors=True)
    print("turing_nodus_spirv_crossing_test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
