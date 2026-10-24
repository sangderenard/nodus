"""Strict, immutable consumer for Turing deployable SPIR-V products."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any, Mapping

import yaml


PRODUCT_SCHEMA = "turing.spirv-product.v1"
API_SCHEMA = "turing-compiled-program-api-v1"
COMPONENT_SCHEMA = "turing.shader-component.v1"
DEPLOYMENT_SCHEMA = "turing.shader-deployment.v1"


class SpirvProductError(ValueError):
    pass


@dataclass(frozen=True)
class SpirvPort:
    slot: int
    name: str
    role: str
    dtype: str
    value_id: int
    binding: int


@dataclass(frozen=True)
class SpirvPortView:
    slot: int
    value_id: int
    rank: int
    shape: tuple[int, ...] | None
    extents_from: str | int | None
    instance_shape: tuple[int, ...]
    byte_scale: int
    fixed_bytes: int | None


@dataclass(frozen=True)
class SpirvResource:
    name: str
    binding: int
    access: str
    dtype: str
    element_bytes: int
    view: SpirvPortView


@dataclass(frozen=True)
class SpirvLaunch:
    local_size: tuple[int, int, int]
    invocations_per_group: int
    control_binding: int
    extent_output_slot: int


@dataclass(frozen=True)
class ToolReceipt:
    tool: str
    arguments: tuple[str, ...]
    returncode: int
    diagnostics: str


@dataclass(frozen=True)
class SpirvProduct:
    root: Path
    manifest_path: Path
    module_path: Path
    assembly_path: Path
    api_path: Path
    component_id: str
    entrypoint: str
    target_environment: str
    spirv_version: str
    ports: tuple[SpirvPort, ...]
    resources: tuple[SpirvResource, ...]
    launch: SpirvLaunch
    tools: tuple[ToolReceipt, ...]
    provenance: tuple[tuple[str, Any], ...]


def _map(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpirvProductError(f"{label} must be a mapping")
    return value


def _fields(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing, unknown = required - set(value), set(value) - required
    if missing or unknown:
        raise SpirvProductError(
            f"{label} fields differ: missing={sorted(missing)!r} unknown={sorted(unknown)!r}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpirvProductError(f"{label} must be a non-empty string")
    return value


def _int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SpirvProductError(f"{label} must be an integer >= {minimum}")
    return value


def _file(root: Path, value: Any, label: str) -> Path:
    relative = Path(_text(value, label))
    if relative.is_absolute():
        raise SpirvProductError(f"{label} must be relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SpirvProductError(f"{label} escapes the product directory") from error
    if not path.is_file():
        raise SpirvProductError(f"{label} does not name a file")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shape(value: Any, label: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise SpirvProductError(f"{label} must be a list or null")
    return tuple(_int(item, label, 1) for item in value)


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((str(key), _freeze(member)) for key, member in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze(member) for member in value)
    return value


_DTYPE_BYTES = {"f32": 4, "i32": 4, "u32": 4, "f64": 8, "i64": 8, "u64": 8}


def _extent(value: Any, label: str, scale: int) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 0:
            raise SpirvProductError(f"{label} must be positive")
        return value
    if _map(value, label) != {"control": "element_count", "index": 0, "scale": scale}:
        raise SpirvProductError(f"{label} is not a supported byte view")
    return None


def _tools(value: Any) -> tuple[ToolReceipt, ...]:
    if not isinstance(value, list):
        raise SpirvProductError("manifest.tools must be a list")
    result = []
    for index, raw in enumerate(value):
        item = _map(raw, f"tool receipt {index}")
        _fields(item, {"tool", "arguments", "returncode", "diagnostics"}, "tool receipt")
        args = item["arguments"]
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise SpirvProductError("tool arguments must be strings")
        result.append(ToolReceipt(
            _text(item["tool"], "tool"), tuple(args),
            _int(item["returncode"], "tool returncode"), str(item["diagnostics"]),
        ))
    if {item.tool for item in result} != {"spirv-as", "spirv-val"}:
        raise SpirvProductError("complete product requires assembler and validator receipts")
    if any(item.returncode for item in result):
        raise SpirvProductError("complete product contains a failed tool receipt")
    return tuple(result)


def load_spirv_product(manifest_path: str | Path) -> SpirvProduct:
    """Adopt the narrow set-0 Vulkan compute profile without recompilation."""

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent.resolve()
    try:
        manifest = _map(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise SpirvProductError(f"cannot read manifest: {error}") from error
    _fields(manifest, {
        "schema", "product_kind", "assembly", "module", "api", "entrypoint",
        "target_env", "spirv_version", "complete", "shortfalls", "tools", "sha256",
    }, "manifest")
    if manifest["schema"] != PRODUCT_SCHEMA:
        raise SpirvProductError(f"manifest schema must be {PRODUCT_SCHEMA}")
    if manifest["product_kind"] != "deployable-shader":
        raise SpirvProductError("linkable SPIR-V libraries are not runnable shaders")
    if manifest["complete"] is not True or manifest["shortfalls"] != []:
        raise SpirvProductError("product is incomplete or publishes shortfalls")
    assembly = _file(root, manifest["assembly"], "assembly")
    module = _file(root, manifest["module"], "module")
    api_path = _file(root, manifest["api"], "api")
    entrypoint = _text(manifest["entrypoint"], "entrypoint")
    target_env = _text(manifest["target_env"], "target_env")
    version = _text(manifest["spirv_version"], "spirv_version")
    if target_env != "vulkan1.1":
        raise SpirvProductError(f"unsupported target environment {target_env!r}")

    hashes = _map(manifest["sha256"], "manifest.sha256")
    referenced = (assembly, module, api_path)
    if set(hashes) != {path.name for path in referenced}:
        raise SpirvProductError("manifest hashes must name exactly assembly, module, and API")
    for path in referenced:
        if hashes[path.name] != _sha(path):
            raise SpirvProductError(f"manifest digest mismatch for {path.name}")
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise SpirvProductError("product lacks SHA256SUMS")
    sums = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or not name or name in sums:
            raise SpirvProductError("malformed SHA256SUMS")
        sums[name] = digest
    for path in (*referenced, manifest_path):
        if sums.get(path.name) != _sha(path):
            raise SpirvProductError(f"SHA256SUMS mismatch for {path.name}")

    binary = module.read_bytes()
    if len(binary) < 20 or len(binary) % 4:
        raise SpirvProductError("module is not a complete SPIR-V word stream")
    magic, version_word = struct.unpack_from("<II", binary)
    observed = f"{version_word >> 16 & 0xff}.{version_word >> 8 & 0xff}"
    if magic != 0x07230203 or observed != version:
        raise SpirvProductError("SPIR-V header disagrees with the manifest")

    try:
        api = _map(yaml.safe_load(api_path.read_text(encoding="utf-8")), "API")
    except (OSError, yaml.YAMLError) as error:
        raise SpirvProductError(f"cannot read API: {error}") from error
    if api.get("schema") != API_SCHEMA or api.get("language") != "spirv":
        raise SpirvProductError("API is not a Turing SPIR-V program record")
    if api.get("entry") != entrypoint or not any(
        isinstance(item, Mapping) and item.get("symbol") == entrypoint
        for item in (api.get("entry_points") or [])
    ):
        raise SpirvProductError("API does not declare the manifest entrypoint")
    metadata = _map(api.get("metadata"), "API metadata")
    component = _map(metadata.get("component_abi"), "component ABI")
    deployment = _map(metadata.get("shader_deployment"), "deployment")
    component_id = _text(component.get("component_id"), "component identity")
    reference = _map(deployment.get("component_abi"), "component reference")
    if component.get("schema") != COMPONENT_SCHEMA or deployment.get("schema") != DEPLOYMENT_SCHEMA:
        raise SpirvProductError("unsupported component or deployment schema")
    if not (api.get("module") == component_id == deployment.get("component_id") == reference.get("component_id")):
        raise SpirvProductError("API, component, and deployment identities disagree")
    if reference != {"schema": COMPONENT_SCHEMA, "component_id": component_id,
                     "record": "metadata.component_abi"}:
        raise SpirvProductError("deployment does not reference its authoritative component ABI")
    if not (
        component.get("language") == deployment.get("language") == "spirv"
        and component.get("stage") == deployment.get("stage") == metadata.get("stage") == "compute"
        and component.get("entrypoint") == deployment.get("entrypoint") == entrypoint
        and deployment.get("product_kind") == "deployable-shader"
        and deployment.get("complete") is True and deployment.get("shortfalls") == []
    ):
        raise SpirvProductError("logical ABI and deployment disagree")
    target = _map(deployment.get("target"), "deployment target")
    if target != {"client": "vulkan", "environment": target_env, "spirv_version": version}:
        raise SpirvProductError("manifest and deployment targets disagree")
    source = _map(deployment.get("source"), "deployment source")
    if source != {"file": module.name, "sha256": _sha(module),
                  "assembly": assembly.name, "assembly_sha256": _sha(assembly)}:
        raise SpirvProductError("deployment source identity or digest disagrees")

    port_rows = component.get("ports")
    if not isinstance(port_rows, list) or not port_rows:
        raise SpirvProductError("component has no ports")
    ports = []
    port_fields = {"slot", "name", "role", "dtype", "value_id", "backend_binding", "transport"}
    for raw in port_rows:
        row = _map(raw, "component port")
        _fields(row, port_fields, "component port")
        role, dtype = row["role"], row["dtype"]
        if role not in {"feed", "output"} or dtype not in _DTYPE_BYTES:
            raise SpirvProductError("unsupported port role or dtype")
        if row["transport"] != "storage-buffer":
            raise SpirvProductError("port transport must be storage-buffer")
        ports.append(SpirvPort(
            _int(row["slot"], "port slot"), _text(row["name"], "port name"),
            role, dtype, _int(row["value_id"], "value id"),
            _int(row["backend_binding"], "port binding"),
        ))
    if tuple(port.slot for port in ports) != tuple(range(len(ports))):
        raise SpirvProductError("logical port slots must be contiguous and ordered")

    raw_resources = deployment.get("resources")
    if not isinstance(raw_resources, list) or len(raw_resources) != len(ports):
        raise SpirvProductError("deployment must have one data resource per logical port")
    by_slot = {port.slot: port for port in ports}
    resources = []
    for raw in raw_resources:
        row = _map(raw, "resource")
        if row.get("kind") != "storage-buffer" or row.get("set") != 0:
            raise SpirvProductError("only set-0 storage buffers are supported")
        views = row.get("realizes")
        if not isinstance(views, list) or len(views) != 1:
            raise SpirvProductError("each resource must realize exactly one port")
        view = _map(views[0], "port view")
        slot, binding = _int(view.get("slot"), "view slot"), _int(row.get("binding"), "binding")
        port = by_slot.get(slot)
        if port is None or port.binding != binding or port.dtype != row.get("dtype"):
            raise SpirvProductError("resource binding/dtype disagrees with its logical port")
        access = "read" if port.role == "feed" else "read_write"
        dtype = port.dtype
        element_bytes = _int(row.get("element_bytes"), "element bytes", 1)
        if row.get("access") != access or row.get("layout") != "std430" or _DTYPE_BYTES[dtype] != element_bytes:
            raise SpirvProductError("resource access, layout, or element size is unsupported")
        instance = _shape(view.get("instance_shape"), "instance shape")
        if instance is None:
            raise SpirvProductError("instance_shape must be present")
        scale = element_bytes
        for extent in instance:
            scale *= extent
        resource_fixed = _extent(row.get("byte_extent"), "resource byte extent", scale)
        view_fixed = _extent(view.get("byte_extent"), "view byte extent", scale)
        if resource_fixed != view_fixed:
            raise SpirvProductError("resource and logical view byte extents disagree")
        rank = _int(view.get("rank"), "view rank")
        shape = _shape(view.get("shape"), "view shape")
        extents_from = view.get("extents_from")
        if shape is not None:
            if len(shape) != rank or resource_fixed != element_bytes * _product(shape):
                raise SpirvProductError("fixed resource shape and byte extent disagree")
            if extents_from is not None:
                raise SpirvProductError("fixed resource cannot cite runtime extents")
        else:
            if resource_fixed is not None or rank <= len(instance):
                raise SpirvProductError("runtime resource must have dynamic leading extents")
            if not (extents_from == "fed" or isinstance(extents_from, int)):
                raise SpirvProductError("runtime resource has no extent authority")
        if row.get("byte_offset") != 0 or view.get("byte_offset") != 0:
            raise SpirvProductError("nonzero resource offsets are unsupported")
        resources.append(SpirvResource(
            _text(row.get("name"), "resource name"), binding, access, dtype,
            element_bytes, SpirvPortView(
                slot, _int(view.get("value_id"), "view value id"),
                rank, shape, extents_from, instance, scale, resource_fixed,
            ),
        ))
    resources.sort(key=lambda item: item.binding)
    if tuple(item.binding for item in resources) != tuple(range(len(resources))):
        raise SpirvProductError("data bindings must be contiguous from zero")
    resource_by_slot = {item.view.slot: item for item in resources}
    for resource in resources:
        source = resource.view.extents_from
        if source == "fed":
            if by_slot[resource.view.slot].role != "feed":
                raise SpirvProductError("only a feed may declare fed extents")
        elif isinstance(source, int):
            authority = resource_by_slot.get(source)
            if authority is None or authority.view.shape is not None:
                raise SpirvProductError("runtime extent authority is not a dynamic port")
    if not any(item.view.extents_from == "fed" for item in resources):
        raise SpirvProductError("runtime launch has no authoritative fed extent")
    def extent_feed(slot: int, seen: set[int]) -> int:
        if slot in seen:
            raise SpirvProductError("runtime extent authorities form a cycle")
        seen.add(slot)
        source = resource_by_slot[slot].view.extents_from
        if source == "fed":
            return slot
        if not isinstance(source, int):
            raise SpirvProductError("runtime extent authority does not reach a feed")
        return extent_feed(source, seen)
    for resource in resources:
        if resource.view.shape is None:
            extent_feed(resource.view.slot, set())

    launch = _map(deployment.get("launch"), "launch")
    raw_local = launch.get("local_size")
    if not isinstance(raw_local, list) or len(raw_local) != 3:
        raise SpirvProductError("local_size must have three entries")
    local = tuple(_int(item, "local size", 1) for item in raw_local)
    if _map(launch.get("group_count"), "group count") != {
        "rule": "ceil_div", "work": "element_count",
        "invocations_per_group": local[0], "groups": None, "axis": "x",
    }:
        raise SpirvProductError("unsupported group-count rule")
    controls = launch.get("controls")
    control_binding = len(resources)
    if not isinstance(controls, list) or controls != [{
        "name": "element_count", "storage": "storage-buffer", "dtype": "u32",
        "length": 1, "binding": control_binding, "byte_offset": 0,
        "derivation": [{"rule": "element_count"}],
    }]:
        raise SpirvProductError("unsupported launch controls")
    extents = _map(launch.get("extents"), "launch extents")
    order = extents.get("order")
    outputs = {port.slot for port in ports if port.role == "output"}
    if extents.get("type") != "uint32_t" or not isinstance(order, list) or len(order) != 1:
        raise SpirvProductError("unsupported launch extent order")
    extent = _map(order[0], "launch extent")
    if not (extent.get("slot") == 0 and extent.get("kind") == "element_count"
            and extent.get("axis") is None and extent.get("buffer") in outputs):
        raise SpirvProductError("launch extent does not name an output element count")
    count = _map(launch.get("element_count"), "element count")
    if not (count.get("source") == "storage-word" and count.get("control") == "element_count"
            and count.get("kind") == "element_count" and count.get("value") is None):
        raise SpirvProductError("unsupported element-count derivation")

    return SpirvProduct(
        root, manifest_path, module, assembly, api_path, component_id, entrypoint,
        target_env, version, tuple(ports), tuple(resources),
        SpirvLaunch(local, local[0], control_binding, int(extent["buffer"])),
        _tools(manifest["tools"]),
        _freeze(_map(deployment.get("provenance"), "provenance")),
    )


_NODUS_DTYPE = {
    "f32": "NODUS_DTYPE_F32", "f64": "NODUS_DTYPE_F64",
    "i32": "NODUS_DTYPE_I32", "i64": "NODUS_DTYPE_I64",
    "u32": "NODUS_DTYPE_U32", "u64": "NODUS_DTYPE_U64",
}


def _generated_tool_id(product: SpirvProduct) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", product.component_id)
    if not stem or stem[0].isdigit():
        stem = "_" + stem
    return stem + "_spirv"


def _extent_feed_slot(product: SpirvProduct, slot: int) -> int:
    by_slot = {resource.view.slot: resource for resource in product.resources}
    seen: set[int] = set()
    while True:
        if slot in seen:
            raise SpirvProductError("runtime extent authorities form a cycle")
        seen.add(slot)
        source = by_slot[slot].view.extents_from
        if source == "fed":
            return slot
        if not isinstance(source, int) or source not in by_slot:
            raise SpirvProductError("runtime extent authority does not reach a feed")
        slot = source


def write_spirv_row_source(
    product: SpirvProduct, path: str | Path, *, tool_id: str | None = None,
) -> Path:
    """Generate an ordinary Nodus row from the validated physical contract.

    The retained SPIR-V words are embedded byte-for-byte; this function never
    invokes a compiler for the shader.  The C++ compilation only builds the
    host row that validates Nodus tensors and calls the existing Vulkan runner.
    """

    tool_id = tool_id or _generated_tool_id(product)
    symbol = re.sub(r"[^A-Za-z0-9_]", "_", tool_id)
    if not symbol or symbol[0].isdigit():
        symbol = "_" + symbol
    ports = tuple(sorted(product.ports, key=lambda item: item.slot))
    feeds = tuple(port for port in ports if port.role == "feed")
    outputs = tuple(port for port in ports if port.role == "output")
    if not feeds or not outputs:
        raise SpirvProductError("row deployment needs at least one feed and output")
    feed_index = {port.slot: index for index, port in enumerate(feeds)}
    output_index = {port.slot: index for index, port in enumerate(outputs)}
    by_slot = {resource.view.slot: resource for resource in product.resources}
    words = struct.unpack(f"<{product.module_path.stat().st_size // 4}I", product.module_path.read_bytes())

    lines = [
        "// Generated from a strictly adopted Turing SPIR-V product -- DO NOT EDIT.",
        f"// Manifest: {product.manifest_path.as_posix()}",
        f"// Module SHA-256: {_sha(product.module_path)}",
        '#include "tool_api.h"',
        '#include "value_types.h"',
        '#include "common/tensors/abstraction/abstract_tensor_handle.h"',
        '#include "common/tensors/abstraction/tensor_abi.h"',
        '#include "kernel_vulkan_runner.h"',
        "#include <algorithm>", "#include <array>", "#include <cstdint>",
        "#include <cstdio>", "#include <cstring>", "#include <limits>",
        "#include <string>", "#include <vector>", "", "namespace {",
        "static const std::vector<std::uint32_t> kSpirv = {",
    ]
    for start in range(0, len(words), 8):
        lines.append("    " + ", ".join(f"0x{word:08x}u" for word in words[start:start + 8]) + ",")
    lines.extend([
        "};", "",
        f"class TuringSpirvTool_{symbol} final : public ITool {{",
        "public:",
        f"    const char* id_cstr() const noexcept override {{ return {json.dumps(tool_id)}; }}",
        f"    const char* name_cstr() const noexcept override {{ return {json.dumps('Turing SPIR-V ' + product.component_id)}; }}",
        "    ToolCaps caps() const override { return ToolCaps::None; }",
        "    void initialize(const ToolInitContext&) override {}",
        "    void shutdown() override {}",
        "    void tick(double, HostAPI&) override {}",
        "    void render(RenderContext&) override {}",
        "    int32_t port_count() const override { return 2; }",
        "    ToolPortSpec port_spec(int32_t index) const override {",
        f"        if (index == 0) return ToolPortSpec{{ToolPortKind::Argument, {len(feeds)}}};",
        f"        if (index == 1) return ToolPortSpec{{ToolPortKind::Return, {len(outputs)}}};",
        "        return ToolPortSpec{};",
        "    }",
        "    void execute_stack(ToolStackContext& ctx) override {",
        "        if (!ctx.stack.raw) { std::fprintf(stderr, \"[turing-spirv] missing raw stack frame\\n\"); return; }",
        "        if (!nodus::kernels::vulkan_available()) { std::fprintf(stderr, \"[turing-spirv] Vulkan unavailable in row plugin\\n\"); return; }",
        "        RawStackFrame& frame = *ctx.stack.raw;",
        f"        std::array<nodus::tensors::AbstractTensorHandle, {len(feeds)}> inputs{{}};",
        f"        std::array<nodus::tensors::AbstractTensorHandle, {len(outputs)}> outputs{{}};",
        f"        std::array<NodusTensorDesc, {len(feeds)}> input_desc{{}};",
        f"        std::array<void*, {len(feeds)}> input_data{{}};",
        f"        std::array<bool, {len(feeds)}> input_mapped{{}};",
        f"        std::array<std::vector<std::uint8_t>, {len(outputs)}> output_scratch{{}};",
        f"        std::array<std::uint64_t, {len(outputs)}> output_bytes{{}};",
        "        int popped = 0;",
        f"        for (int i = {len(feeds) - 1}; i >= 0; --i) {{",
        "            if (!raw_stack_pop_abstract_tensor(frame, inputs[static_cast<size_t>(i)])) break;",
        "            ++popped;",
        "        }",
        "        auto restore_inputs = [&]() {",
        "            for (const auto handle : inputs) if (handle.id) raw_stack_push_abstract_tensor(frame, handle);",
        "        };",
        f"        if (popped != {len(feeds)}) {{",
        f"            for (int i = {len(feeds)} - popped; i < {len(feeds)}; ++i)",
        "                if (inputs[static_cast<size_t>(i)].id) raw_stack_push_abstract_tensor(frame, inputs[static_cast<size_t>(i)]);",
        "            return;",
        "        }",
        "        auto release_inputs = [&]() {",
        "            for (size_t i = 0; i < inputs.size(); ++i) if (input_mapped[i]) {",
        "                nodus_tensor_unmap(inputs[i].id); input_mapped[i] = false;",
        "            }",
        "        };",
        f"        auto fail = [&](const char* reason) {{ std::fprintf(stderr, \"[turing-spirv:{symbol}] %s\\n\", reason);",
        "            release_inputs();",
        "            for (auto& handle : outputs) if (handle.id) { nodus_tensor_destroy(handle.id); handle.id = 0; }",
        "            restore_inputs();",
        "        };",
    ])

    # Tensor descriptor checks come only from the product's logical/physical
    # records; there is no handwritten binding convention at the call site.
    for port in feeds:
        index = feed_index[port.slot]
        resource = by_slot[port.slot]
        view = resource.view
        checks = [
            f"input_desc[{index}].dtype != {_NODUS_DTYPE[port.dtype]}",
            f"input_desc[{index}].rank != {view.rank}u",
        ]
        if view.shape is not None:
            checks.extend(
                f"input_desc[{index}].shape[{axis}] != {extent}ull"
                for axis, extent in enumerate(view.shape)
            )
            checks.append(f"input_desc[{index}].total_bytes != {view.fixed_bytes}ull")
        else:
            dynamic = view.rank - len(view.instance_shape)
            checks.extend(
                f"input_desc[{index}].shape[{dynamic + axis}] != {extent}ull"
                for axis, extent in enumerate(view.instance_shape)
            )
            checks.append(f"input_desc[{index}].total_bytes % {view.byte_scale}ull != 0")
        lines.append(
            f"        if (nodus_tensor_describe(inputs[{index}].id, &input_desc[{index}]) != NODUS_OK || "
            + " || ".join(checks) + ") { fail(\"input descriptor validation failed\"); return; }"
        )
    lines.extend([
        "        std::uint64_t count64 = 0;",
        "        bool have_count = false;",
    ])
    for port in feeds:
        resource = by_slot[port.slot]
        if resource.view.extents_from == "fed":
            index = feed_index[port.slot]
            lines.extend([
                f"        {{ const std::uint64_t candidate = input_desc[{index}].total_bytes / {resource.view.byte_scale}ull;",
                "          if (!candidate || (have_count && candidate != count64)) { fail(\"runtime extent disagreement\"); return; }",
                "          count64 = candidate; have_count = true; }",
            ])
    lines.extend([
        "        if (!have_count || count64 > std::numeric_limits<std::uint32_t>::max()) { fail(\"invalid dispatch element count\"); return; }",
        "        const std::uint32_t element_count = static_cast<std::uint32_t>(count64);",
    ])
    for port in feeds:
        index = feed_index[port.slot]
        resource = by_slot[port.slot]
        expected = (
            f"{resource.view.fixed_bytes}ull" if resource.view.fixed_bytes is not None
            else f"count64 * {resource.view.byte_scale}ull"
        )
        lines.extend([
            f"        {{ void* data = nullptr; std::uint64_t bytes = 0;",
            f"          if (nodus_tensor_map(inputs[{index}].id, &data, &bytes) != NODUS_OK || bytes < {expected}) {{ fail(\"input tensor map failed\"); return; }}",
            f"          input_data[{index}] = data; input_mapped[{index}] = true; }}",
        ])

    for port in outputs:
        index = output_index[port.slot]
        resource = by_slot[port.slot]
        view = resource.view
        if view.shape is not None:
            if view.shape:
                shape_values = ", ".join(f"{extent}ull" for extent in view.shape)
                lines.append(f"        {{ const std::array<std::uint64_t, {len(view.shape)}> shape = {{{shape_values}}};")
                lines.append(f"          outputs[{index}].id = nodus_tensor_create({_NODUS_DTYPE[port.dtype]}, {len(view.shape)}, shape.data()); }}")
            else:
                lines.append(f"        outputs[{index}].id = nodus_tensor_create({_NODUS_DTYPE[port.dtype]}, 0, nullptr);")
            lines.append(f"        output_bytes[{index}] = {view.fixed_bytes}ull;")
        else:
            authority_slot = _extent_feed_slot(product, port.slot)
            authority_index = feed_index.get(authority_slot)
            if authority_index is None:
                raise SpirvProductError("output extent authority is not a public feed")
            authority = by_slot[authority_slot].view
            dynamic = view.rank - len(view.instance_shape)
            authority_dynamic = authority.rank - len(authority.instance_shape)
            if dynamic != authority_dynamic:
                raise SpirvProductError("output and feed dynamic ranks disagree")
            shape_values = [f"input_desc[{authority_index}].shape[{axis}]" for axis in range(dynamic)]
            shape_values.extend(f"{extent}ull" for extent in view.instance_shape)
            lines.append(f"        {{ const std::array<std::uint64_t, {view.rank}> shape = {{{', '.join(shape_values)}}};")
            lines.append(f"          outputs[{index}].id = nodus_tensor_create({_NODUS_DTYPE[port.dtype]}, {view.rank}, shape.data()); }}")
            lines.append(f"        output_bytes[{index}] = count64 * {view.byte_scale}ull;")
        lines.extend([
            f"        if (!outputs[{index}].id || output_bytes[{index}] > std::numeric_limits<size_t>::max() - {view.byte_scale * product.launch.invocations_per_group}ull) {{ fail(\"output tensor allocation failed\"); return; }}",
            f"        output_scratch[{index}].assign(static_cast<size_t>(output_bytes[{index}] + {view.byte_scale * product.launch.invocations_per_group}ull), 0xA5u);",
        ])

    lines.append("        std::vector<nodus::kernels::VulkanBufferBinding> bindings;")
    lines.append(f"        bindings.reserve({len(product.resources) + 1});")
    for resource in product.resources:
        slot = resource.view.slot
        if slot in feed_index:
            index = feed_index[slot]
            expected = (
                f"{resource.view.fixed_bytes}ull" if resource.view.fixed_bytes is not None
                else f"count64 * {resource.view.byte_scale}ull"
            )
            lines.append(f"        bindings.push_back({{input_data[{index}], static_cast<size_t>({expected}), false}});")
        else:
            index = output_index[slot]
            lines.append(f"        bindings.push_back({{output_scratch[{index}].data(), output_scratch[{index}].size(), true}});")
    lines.extend([
        "        std::uint32_t control = element_count;",
        "        bindings.push_back({&control, sizeof(control), false});",
        f"        const std::uint64_t groups64 = (count64 + {product.launch.invocations_per_group - 1}ull) / {product.launch.invocations_per_group}ull + 1ull;",
        "        if (groups64 > std::numeric_limits<std::uint32_t>::max()) { fail(\"dispatch group count overflow\"); return; }",
        "        std::string error;",
        f"        if (!nodus::kernels::dispatch_spirv(kSpirv, {json.dumps(product.entrypoint)}, bindings, static_cast<std::uint32_t>(groups64), &error)) {{",
        f"            std::fprintf(stderr, \"[turing-spirv:{symbol}] dispatch failed: %s\\n\", error.c_str()); fail(\"Vulkan dispatch failed\"); return;",
        "        }",
    ])
    for port in outputs:
        index = output_index[port.slot]
        lines.extend([
            f"        if (!std::all_of(output_scratch[{index}].begin() + static_cast<size_t>(output_bytes[{index}]), output_scratch[{index}].end(), [](std::uint8_t value) {{ return value == 0xA5u; }})) {{",
            f"            std::fprintf(stderr, \"[turing-spirv:{symbol}] published past-end guard failed\\n\"); fail(\"published past-end guard failed\"); return;",
            "        }",
            "        { void* destination = nullptr; std::uint64_t bytes = 0;",
            f"          if (nodus_tensor_map(outputs[{index}].id, &destination, &bytes) != NODUS_OK || bytes < output_bytes[{index}]) {{ fail(\"output tensor map failed\"); return; }}",
            f"          std::memcpy(destination, output_scratch[{index}].data(), static_cast<size_t>(bytes));",
            f"          nodus_tensor_unmap(outputs[{index}].id); }}",
        ])
    lines.extend([
        "        release_inputs();",
        "        int pushed = 0;",
        "        for (auto& handle : outputs) {",
        "            if (!raw_stack_push_abstract_tensor(frame, handle)) {",
        "                for (int i = 0; i < pushed; ++i) { nodus::tensors::AbstractTensorHandle undo{}; if (raw_stack_pop_abstract_tensor(frame, undo)) nodus_tensor_destroy(undo.id); }",
        "                for (size_t i = static_cast<size_t>(pushed); i < outputs.size(); ++i) if (outputs[i].id) nodus_tensor_destroy(outputs[i].id);",
        "                restore_inputs(); return;",
        "            }",
        "            handle.id = 0; ++pushed;",
        "        }",
        "    }",
        "};", "}", "",
        f"extern \"C\" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() {{ return new TuringSpirvTool_{symbol}(); }}",
        "extern \"C\" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }",
        "extern \"C\" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }",
        "",
    ])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def graph_definition_from_spirv_product(
    product: SpirvProduct, *, stable_id: int, abstract_tensor_type_id: int,
    tool_id: str | None = None, label: str | None = None,
    slots: int = 4, top_k: int = 1,
):
    """Derive connectors, named ports, and tensor contracts from Turing."""

    from .model import (
        ConnectorDefinition, ConnectorRef, ExecutionContract, FifoConfig,
        GraphDefinition, Input, ModuleDefinition, Output, PluginTool,
        PortDefinition, ShapeSource, TensorContract,
    )

    feeds = tuple(port for port in product.ports if port.role == "feed")
    outputs = tuple(port for port in product.ports if port.role == "output")
    refs = {
        port.slot: ConnectorRef(stable_id, "input", index)
        for index, port in enumerate(feeds)
    }
    refs.update({
        port.slot: ConnectorRef(stable_id, "output", index)
        for index, port in enumerate(outputs)
    })
    by_slot = {resource.view.slot: resource for resource in product.resources}
    connectors = []
    for port in (*feeds, *outputs):
        view = by_slot[port.slot].view
        if view.shape is not None:
            shape = view.shape
            sources = (None,) * len(shape)
        else:
            dynamic = view.rank - len(view.instance_shape)
            shape = (None,) * dynamic + view.instance_shape
            authority = port.slot if port.role == "feed" else _extent_feed_slot(product, port.slot)
            sources = tuple(ShapeSource(refs[authority], axis) for axis in range(dynamic))
            sources += (None,) * len(view.instance_shape)
        connectors.append(ConnectorDefinition(
            refs[port.slot], port.name, TensorContract(port.dtype, shape, sources),
        ))
    fifo = FifoConfig(
        token_bytes=8, slots=slots, top_k=top_k,
        type_id=abstract_tensor_type_id, layout=2, dtype=16,
    )
    row_id = tool_id or _generated_tool_id(product)
    module = ModuleDefinition(
        stable_id=stable_id, label=label or f"Turing SPIR-V {product.component_id}",
        rows=(
            Input(len(feeds)),
            PluginTool(row_id, len(feeds), len(outputs), ExecutionContract.ABSTRACT_TENSOR_POINTER),
            Output(len(outputs)),
        ),
        contract=ExecutionContract.ABSTRACT_TENSOR_POINTER,
    )
    return GraphDefinition(
        modules=(module,), connectors=tuple(connectors),
        inputs=tuple(PortDefinition(port.name, refs[port.slot], fifo) for port in feeds),
        outputs=tuple(PortDefinition(port.name, refs[port.slot], fifo) for port in outputs),
    )


def deploy_spirv_product_graph(
    graph, manifest_path: str | Path, *, stable_id: int,
    nodus_root: str | Path, module_library_root: str | Path,
):
    """Strictly adopt, register, and materialize a Turing SPIR-V graph row."""

    product = load_spirv_product(manifest_path)
    base_id = _generated_tool_id(product)
    source = Path(module_library_root) / "source" / "tools" / f"{base_id}.cpp"
    write_spirv_row_source(product, source, tool_id=base_id)
    loaded_id = graph.build_plugin_source(
        source, nodus_root=nodus_root, module_library_root=module_library_root,
    )
    if not (loaded_id == base_id or loaded_id.startswith(base_id + "_ver_")):
        graph.unload_plugin(loaded_id)
        raise RuntimeError(
            f"loaded tool {loaded_id!r} disagrees with SPIR-V product {base_id!r}"
        )
    definition = graph_definition_from_spirv_product(
        product, stable_id=stable_id,
        abstract_tensor_type_id=graph.abstract_tensor_type_id,
        tool_id=loaded_id,
    )
    graph.apply_definition(definition)
    return definition, product, {"loaded_tool_id": loaded_id, "staged_source": str(source)}
