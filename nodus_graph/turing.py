from __future__ import annotations

import json
import pathlib
import struct
from dataclasses import dataclass
from typing import Mapping, Sequence

from .model import (
    ConnectorDefinition,
    ConnectorRef,
    ExecutionContract,
    FifoConfig,
    GraphDefinition,
    Input,
    ModuleDefinition,
    Output,
    PluginTool,
    PortDefinition,
    ShapeSource,
    TensorContract,
)
from .native import Graph


@dataclass(frozen=True)
class PythonRowPort:
    name: str
    primitive_type: str = "float64"
    label: str = ""
    form_control: str = ""
    required: bool = True


def write_python_row_descriptor(
    path: str | pathlib.Path, *, tool_id: str, tool_name: str,
    source_path: str | pathlib.Path, module_name: str, callable_name: str,
    inputs: Sequence[PythonRowPort], outputs: Sequence[PythonRowPort],
) -> pathlib.Path:
    """Write the existing NODUSPYROW V1 descriptor for native deployment."""

    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    lines = [
        "NODUSPYROW 1",
        "MODE INTERPRETED",
        f"TOOL_ID {quote(tool_id)}",
        f"TOOL_NAME {quote(tool_name)}",
        f"SOURCE {quote(pathlib.Path(source_path).resolve().as_posix())}",
        f"MODULE {quote(module_name)}",
        f"CALLABLE {quote(callable_name)}",
    ]
    for port in inputs:
        lines.append(f"INPUT {quote(port.name)} {port.primitive_type}")
        if port.label or port.form_control or not port.required:
            requirement = "REQUIRED" if port.required else "OPTIONAL"
            lines.append(
                f"INPUT_UI {quote(port.name)} {quote(port.label)} "
                f"{quote(port.form_control)} {requirement}"
            )
    for port in outputs:
        lines.append(f"OUTPUT {quote(port.name)} {port.primitive_type}")
    output = pathlib.Path(path)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def load_row_product(path: str | pathlib.Path) -> Mapping[str, object]:
    product = pathlib.Path(path)
    manifest_path = product if product.is_file() else product / "nodus-row-tool-product.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "nodus-turing-row-tool-product-v1":
        raise ValueError(f"unsupported Turing row product: {manifest.get('schema')!r}")
    if manifest.get("loadable") is not True or not isinstance(manifest.get("buffer_abi"), dict):
        raise ValueError("Turing row product is not loadable or lacks its buffer ABI")
    return manifest


def graph_definition_from_row_product(
    product: Mapping[str, object], *, stable_id: int, abstract_tensor_type_id: int,
    label: str | None = None, slots: int = 4, top_k: int = 1,
) -> GraphDefinition:
    """Derive one executable Nodus module from Turing's retained buffer ABI."""

    abi = product.get("buffer_abi")
    if not isinstance(abi, dict) or abi.get("schema") != "nodus-turing-buffer-abi-v2":
        raise ValueError("product lacks NODUS BUFFER ABI V2 metadata")
    buffers = {
        int(item["ssa_value_id"]): item
        for item in abi.get("buffers", ())
        if isinstance(item, dict)
    }
    argument_ids = tuple(int(value) for value in abi.get("arguments", ()))
    output_ids = tuple(int(value) for value in abi.get("outputs", ()))
    if not argument_ids or not output_ids:
        raise ValueError("row product must expose at least one input and one output")
    if any(value not in buffers for value in argument_ids + output_ids):
        raise ValueError("buffer ABI omits a public argument or output")

    refs: dict[int, ConnectorRef] = {}
    for ordinal, value_id in enumerate(argument_ids):
        refs[value_id] = ConnectorRef(stable_id, "input", ordinal)
    for ordinal, value_id in enumerate(output_ids):
        if value_id in refs:
            raise ValueError("buffer ABI aliases an input and output identity")
        refs[value_id] = ConnectorRef(stable_id, "output", ordinal)

    connector_definitions: list[ConnectorDefinition] = []
    for value_id in argument_ids + output_ids:
        buffer = buffers[value_id]
        shape = tuple(buffer.get("shape", ()))
        raw_sources = tuple(buffer.get("shape_sources", ()))
        if len(raw_sources) != len(shape):
            raise ValueError(f"buffer {value_id} shape source count disagrees with rank")
        sources: list[ShapeSource | None] = []
        for source in raw_sources:
            if source is None:
                sources.append(None)
                continue
            source_id = int(source["ssa_value_id"])
            if source_id not in refs:
                raise ValueError(f"buffer {value_id} shape source {source_id} is not a public connector")
            sources.append(ShapeSource(refs[source_id], int(source["axis"])))
        connector_definitions.append(ConnectorDefinition(
            reference=refs[value_id],
            name=str(buffer.get("column") or f"ssa_{value_id}"),
            tensor=TensorContract(
                dtype=str(buffer["dtype"]),
                shape=shape,
                shape_sources=tuple(sources),
            ),
        ))

    fifo = FifoConfig(
        token_bytes=8,
        slots=slots,
        top_k=top_k,
        type_id=abstract_tensor_type_id,
        layout=2,  # TensorLayout::Opaque
        dtype=16,  # TensorDType::Ptr
    )
    tool_id = str(abi["tool_id"])
    module = ModuleDefinition(
        stable_id=stable_id,
        label=label or str(abi.get("tool_name") or tool_id),
        rows=(
            Input(len(argument_ids)),
            PluginTool(
                tool_id, len(argument_ids), len(output_ids),
                ExecutionContract.ABSTRACT_TENSOR_POINTER,
            ),
            Output(len(output_ids)),
        ),
        contract=ExecutionContract.ABSTRACT_TENSOR_POINTER,
    )
    by_ref = {item.reference: item for item in connector_definitions}
    return GraphDefinition(
        modules=(module,),
        connectors=tuple(connector_definitions),
        inputs=tuple(
            PortDefinition(by_ref[refs[value_id]].name, refs[value_id], fifo)
            for value_id in argument_ids
        ),
        outputs=tuple(
            PortDefinition(by_ref[refs[value_id]].name, refs[value_id], fifo)
            for value_id in output_ids
        ),
    )


def deploy_row_product_graph(
    graph: Graph, descriptor_path: str | pathlib.Path, *, target: str,
    stable_id: int, nodus_root: str | pathlib.Path,
    module_library_root: str | pathlib.Path,
    python_executable: str | pathlib.Path,
    turing_root: str | pathlib.Path,
    extraction_contract: str | pathlib.Path,
    product_dir: str | pathlib.Path,
) -> tuple[GraphDefinition, dict[str, object]]:
    tool_id, report = graph.deploy_python_row(
        descriptor_path, target=target, nodus_root=nodus_root,
        module_library_root=module_library_root,
        python_executable=python_executable, turing_root=turing_root,
        extraction_contract=extraction_contract, product_dir=product_dir,
    )
    manifest = load_row_product(product_dir)
    definition = graph_definition_from_row_product(
        manifest, stable_id=stable_id,
        abstract_tensor_type_id=graph.abstract_tensor_type_id,
    )
    product_tool = definition.modules[0].rows[1]
    if not isinstance(product_tool, PluginTool) or not (
        tool_id == product_tool.plugin_id or
        tool_id.startswith(product_tool.plugin_id + "_ver_")
    ):
        graph.unload_plugin(tool_id)
        raise RuntimeError(
            f"loaded tool {tool_id!r} disagrees with product ABI {product_tool!r}"
        )
    graph.apply_definition(definition)
    return definition, report


def compile_row_product_graph(
    graph: Graph, request_json: str | pathlib.Path, *, stable_id: int,
    python_executable: str | pathlib.Path, turing_root: str | pathlib.Path,
    product_dir: str | pathlib.Path, nodus_root: str | pathlib.Path,
    module_library_root: str | pathlib.Path,
) -> tuple[GraphDefinition, dict[str, object]]:
    """Compile an explicit Turing row request and install its derived graph."""

    loaded_id, report = graph.compile_turing_row(
        request_json, python_executable=python_executable,
        turing_root=turing_root, product_dir=product_dir,
        nodus_root=nodus_root, module_library_root=module_library_root,
    )
    definition = graph_definition_from_row_product(
        load_row_product(product_dir), stable_id=stable_id,
        abstract_tensor_type_id=graph.abstract_tensor_type_id,
    )
    product_tool = definition.modules[0].rows[1]
    if not isinstance(product_tool, PluginTool) or not (
        loaded_id == product_tool.plugin_id or
        loaded_id.startswith(product_tool.plugin_id + "_ver_")
    ):
        graph.unload_plugin(loaded_id)
        raise RuntimeError(
            f"loaded tool {loaded_id!r} disagrees with product ABI {product_tool!r}"
        )
    graph.apply_definition(definition)
    return definition, report


def pack_tensor_handle(handle: int) -> bytes:
    return struct.pack("=Q", handle)


def unpack_tensor_handle(payload: bytes) -> int:
    if len(payload) != 8:
        raise ValueError(f"AbstractTensorHandle payload must be 8 bytes, got {len(payload)}")
    return struct.unpack("=Q", payload)[0]
