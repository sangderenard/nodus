"""Real Turing SSA product -> Nodus GraphDefinition -> native execution."""

from __future__ import annotations

import math
import pathlib
import shutil
import struct
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nodus_graph import (
    Graph,
    GraphDefinition,
    PythonRowPort,
    deploy_row_product_graph,
    pack_tensor_handle,
    unpack_tensor_handle,
    write_python_row_descriptor,
)


def main() -> int:
    nodus_root = pathlib.Path(__file__).resolve().parent.parent
    turing_root = nodus_root.parent / "turing"
    root = pathlib.Path(tempfile.mkdtemp(prefix="nodus_turing_graph_crossing_"))
    source = root / "arithmetic.py"
    descriptor = root / "arithmetic.pyabi"
    product = root / "product-c"
    module_library = root / "module-library"
    source.write_text(
        "def calculate(left, right):\n"
        "    return left + right\n",
        encoding="utf-8",
    )
    write_python_row_descriptor(
        descriptor,
        tool_id="python.graph_crossing_arithmetic",
        tool_name="Graph crossing arithmetic",
        source_path=source,
        module_name="graph_crossing_arithmetic",
        callable_name="calculate",
        inputs=(PythonRowPort("left"), PythonRowPort("right")),
        outputs=(PythonRowPort("result"),),
    )

    graph = Graph()
    tool_id = ""
    handles: list[int] = []
    try:
        definition, report = deploy_row_product_graph(
            graph, descriptor, target="c", stable_id=0x7A11CE,
            nodus_root=nodus_root, module_library_root=module_library,
            python_executable=sys.executable, turing_root=turing_root,
            extraction_contract=(
                turing_root / "extraction_contracts" / "program_extraction.yaml"
            ),
            product_dir=product,
        )
        tool_id = str(report["loaded_tool_id"])
        assert report["compiler_invoked"] == 1
        assert report["plugin_loaded"] == 1
        assert len(definition.connectors) == 3
        assert [item.name for item in definition.connectors] == ["left", "right", "result"]
        assert all(item.tensor.dtype == "f64" for item in definition.connectors)
        assert all(item.tensor.shape == () for item in definition.connectors)
        assert GraphDefinition.from_mapping(definition.to_mapping()) == definition
        readback = graph.definition()
        assert readback == definition, (readback.to_mapping(), definition.to_mapping())
        assert graph.modules[0].native_contract.value == "abstract_tensor_pointer"

        left = graph.create_tensor(2, (), struct.pack("=d", 9.5))
        right = graph.create_tensor(2, (), struct.pack("=d", 4.0))
        handles.extend((left, right))
        graph.push_bytes("left", pack_tensor_handle(left))
        graph.push_bytes("right", pack_tensor_handle(right))
        graph.step()
        payload = graph.pull_bytes("result", 8)
        assert payload is not None
        output = unpack_tensor_handle(payload)
        handles.append(output)
        description = graph.describe_tensor(output)
        assert description["dtype"] == 2
        assert description["shape"] == ()
        value = struct.unpack("=d", graph.read_tensor(output))[0]
        assert math.isclose(value, 13.5, rel_tol=0.0, abs_tol=1.0e-12)
    finally:
        for handle in reversed(handles):
            graph.destroy_tensor(handle)
        graph.close()
        if tool_id:
            graph.unload_plugin(tool_id)
        shutil.rmtree(root, ignore_errors=True)

    print("turing_nodus_graph_crossing_test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
