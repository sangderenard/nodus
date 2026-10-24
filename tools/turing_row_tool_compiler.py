"""Compile a Python row through Turing and retain a Nodus tool product.

This module is deliberately a Nodus-side adapter.  Turing is treated as an
external compiler and is entered only through
``src.compiler.fortran_c_shell.lower_ast_source_to_ssa``.  Backend emitters
consume the returned repository SSA without a second lowering path.

CLI usage::

    python tools/turing_row_tool_compiler.py --request row.json

The request is JSON.  It must contain ``row_entry``, ``columns``, ``backend``
(``c``, ``llvm`` or ``fortran``), ``output_dir``, ``extraction_contract``, and
exactly one of ``source_path`` or ``source_text``.  The extraction contract is
the caller-selected Turing contract; this adapter never synthesizes one.
Columns distinguish ``shape: []`` (scalar), a concrete extent list, and
``rank: N, shape: null`` (runtime-sized tensor).  A dynamic output names an
input/inout ``shape_from`` column whose handle supplies its allocation shape.
``global_scope: true`` wraps the submitted body in the named row function using
the input/inout columns as parameters; both the authored text and that
deterministic submitted source are retained.

An optional ``ssa_composition`` object selects Turing's existing partitioned
SSA composer::

    {"directory": "...", "options": {"jobs": 2}}

The adapter always sets ``ssa_composition`` true.  It never links SSA itself.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


PRODUCT_SCHEMA = "nodus-turing-row-tool-product-v1"
BUFFER_ABI_SCHEMA = "nodus-turing-buffer-abi-v2"
REQUEST_SCHEMA = "nodus-turing-row-tool-request-v1"
ADOPTION_RECEIPT_NAME = "nodus-row-tool-product.receipt"
_BACKENDS = {"c", "llvm", "fortran"}
_COLUMN_ROLES = {"input", "output", "inout"}
_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")


class ProductError(RuntimeError):
    """The requested product cannot be represented truthfully."""


@dataclasses.dataclass(frozen=True)
class Column:
    name: str
    role: str
    dtype: str
    rank: int = 0
    # ``()`` is a scalar, a concrete tuple is a fixed tensor, and ``None`` is
    # a rank-known tensor whose extents arrive with the Nodus handle.
    shape: tuple[int, ...] | None = ()
    # Dynamic outputs/workspaces need an allocation authority before the
    # compiled entry can run.  The named input/inout column supplies the
    # extents axis-for-axis; Turing remains unaware of the Nodus handle.
    shape_from: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Column":
        name = str(value.get("name") or "")
        role = str(value.get("role") or "")
        dtype = str(value.get("dtype") or "")
        raw_rank = value.get("rank")
        raw_shape = value.get("shape") if "shape" in value else ()
        if raw_shape is None:
            rank = int(raw_rank or 0)
            shape = None if rank else ()
        else:
            shape = tuple(int(item) for item in raw_shape)
            rank = len(shape) if raw_rank is None else int(raw_rank)
        shape_from = (
            None if value.get("shape_from") is None
            else str(value.get("shape_from"))
        )
        if not _IDENTIFIER.fullmatch(name):
            raise ValueError(f"invalid column name {name!r}")
        if role not in _COLUMN_ROLES:
            raise ValueError(f"column {name!r} has unsupported role {role!r}")
        if not dtype:
            raise ValueError(f"column {name!r} requires a dtype")
        if rank < 0:
            raise ValueError(f"column {name!r} has invalid rank {rank}")
        if shape is not None:
            if len(shape) != rank:
                raise ValueError(
                    f"column {name!r} rank {rank} disagrees with shape {shape!r}"
                )
            if any(extent <= 0 for extent in shape):
                raise ValueError(f"column {name!r} has invalid shape {shape!r}")
        elif rank == 0:
            raise ValueError(f"column {name!r} cannot have an unknown scalar shape")
        if shape_from is not None:
            if not _IDENTIFIER.fullmatch(shape_from):
                raise ValueError(
                    f"column {name!r} has invalid shape_from {shape_from!r}"
                )
            if shape is not None:
                raise ValueError(
                    f"column {name!r} may use shape_from only with shape null"
                )
        return cls(
            name=name, role=role, dtype=dtype, rank=rank,
            shape=shape, shape_from=shape_from,
        )

    def to_mapping(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "role": self.role,
            "dtype": self.dtype,
            "rank": self.rank,
            "shape": None if self.shape is None else list(self.shape),
        }
        if self.shape_from is not None:
            result["shape_from"] = self.shape_from
        return result


@dataclasses.dataclass(frozen=True)
class RowToolRequest:
    row_entry: str
    columns: tuple[Column, ...]
    backend: str
    output_dir: Path
    source_path: Path | None = None
    source_text: str | None = None
    global_scope: bool = False
    product_name: str | None = None
    turing_root: Path | None = None
    extraction_contract: Path | None = None
    ssa_composition_directory: Path | None = None
    ssa_composition_options: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )
    allow_uncompiled: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RowToolRequest":
        source_path = value.get("source_path")
        source_text = value.get("source_text")
        if (source_path is None) == (source_text is None):
            raise ValueError("request requires exactly one of source_path or source_text")
        row_entry = str(value.get("row_entry") or "")
        if not _IDENTIFIER.fullmatch(row_entry):
            raise ValueError(f"invalid row entry {row_entry!r}")
        backend = str(value.get("backend") or "").lower()
        if backend not in _BACKENDS:
            raise ValueError(f"unsupported backend {backend!r}")
        columns = tuple(Column.from_mapping(item) for item in value.get("columns") or ())
        if not columns:
            raise ValueError("request requires an ordered, non-empty column ABI")
        names = [column.name for column in columns]
        if len(names) != len(set(names)):
            raise ValueError("column names must be unique")
        columns_by_name = {column.name: column for column in columns}
        for column in columns:
            if column.shape is None and column.role in {"output", "inout"}:
                if column.shape_from is None and column.role == "output":
                    raise ValueError(
                        f"dynamic output column {column.name!r} requires shape_from"
                    )
            if column.shape_from is None:
                continue
            source = columns_by_name.get(column.shape_from)
            if source is None or source.role not in {"input", "inout"}:
                raise ValueError(
                    f"column {column.name!r} shape_from must name an input/inout column"
                )
            if source.rank != column.rank:
                raise ValueError(
                    f"column {column.name!r} rank does not match shape source "
                    f"{source.name!r}"
                )
        output_dir = value.get("output_dir")
        if not output_dir:
            raise ValueError("request requires output_dir")
        extraction_contract = value.get("extraction_contract")
        if not extraction_contract:
            raise ValueError("request requires an explicit extraction_contract")
        composition = value.get("ssa_composition")
        composition_directory = None
        composition_options: Mapping[str, Any] = {}
        if composition is not None:
            if not isinstance(composition, Mapping) or not composition.get("directory"):
                raise ValueError("ssa_composition requires a directory")
            composition_directory = Path(str(composition["directory"]))
            composition_options = dict(composition.get("options") or {})
            if composition_options.get("ssa_composition") is False:
                raise ValueError("ssa_composition cannot disable Turing's composer")
            composition_options["ssa_composition"] = True
        return cls(
            row_entry=row_entry,
            columns=columns,
            backend=backend,
            output_dir=Path(str(output_dir)),
            source_path=None if source_path is None else Path(str(source_path)),
            source_text=None if source_text is None else str(source_text),
            global_scope=bool(value.get("global_scope", False)),
            product_name=(
                None if value.get("product_name") is None
                else str(value["product_name"])
            ),
            turing_root=(
                None if value.get("turing_root") is None
                else Path(str(value["turing_root"]))
            ),
            extraction_contract=Path(str(extraction_contract)),
            ssa_composition_directory=composition_directory,
            ssa_composition_options=composition_options,
            allow_uncompiled=bool(value.get("allow_uncompiled", True)),
        )


class CompilerBoundary(Protocol):
    head: str
    toolchain: Mapping[str, Any]

    def lower(self, source: str, entrypoint: str, **kwargs: Any) -> tuple[Any, Any, Any]: ...

    def emit_c(self, module: Any, root: str, *, entry_name: str) -> Any: ...

    def compile_c(self, artifact: Any, directory: Path) -> Any: ...

    def emit_llvm(self, module: Any, root: str, *, entry_name: str) -> Any: ...

    def compile_llvm(self, artifact: Any, directory: Path) -> Any: ...

    def emit_fortran(
        self, module: Any, *, name: str, outputs: Any, exports: Sequence[str]
    ) -> Any: ...

    def compile_fortran(
        self, artifact: Any, directory: Path, *, entrypoint: str, name: str
    ) -> Any: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


@contextmanager
def _compiler_working_directory(path: Path):
    """Contain public-compiler relative side products inside the product."""

    path.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not cleaned or cleaned[0].isdigit():
        cleaned = "row_" + cleaned
    return cleaned


def _read_authored_source(request: RowToolRequest) -> tuple[str, str | None]:
    if request.source_path is not None:
        path = request.source_path.resolve()
        return path.read_text(encoding="utf-8"), path.as_posix()
    assert request.source_text is not None
    return request.source_text, None


def _submitted_source(request: RowToolRequest, authored: str) -> str:
    if not request.global_scope:
        return authored
    if re.search(r"^\s*from\s+__future__\s+import\s+", authored, re.MULTILINE):
        raise ValueError("global-scope row bodies cannot contain __future__ imports")
    parameters = [
        column.name for column in request.columns
        if column.role in {"input", "inout"}
    ]
    body = authored.splitlines()
    if not body or not any(line.strip() for line in body):
        body = ["pass"]
    indented = "\n".join("    " + line if line else "" for line in body)
    return f"def {request.row_entry}({', '.join(parameters)}):\n{indented}\n"


def _value_id(value: Any) -> int:
    return int(getattr(value, "id"))


def _root_function(module: Any, root: str) -> Any:
    functions = getattr(module, "functions", None)
    if not isinstance(functions, Mapping) or root not in functions:
        raise ProductError(f"lowered SSA has no exported root {root!r}")
    return functions[root]


def _published_outputs(outputs: Any, root: str) -> tuple[Any, ...]:
    if not isinstance(outputs, Mapping):
        return ()
    return tuple(outputs.get(root, ()))


def _render_textual_ssa(module: Any, exports: Sequence[str]) -> str | None:
    """Render a diagnostic SSA listing without claiming a reloadable format."""

    functions = getattr(module, "functions", None)
    if not isinstance(functions, Mapping):
        return None
    lines = ["; NODUS DIAGNOSTIC SSA V1 (repository-ssa.pkl is authoritative)"]
    lines.append("; exports: " + ", ".join(map(str, exports)))
    for name in sorted(functions):
        function = functions[name]
        args = tuple(getattr(function, "args", ()) or ())
        lines.append(
            f"function {name}("
            + ", ".join(f"%t{_value_id(arg)}:{getattr(arg, 'dtype', None)}" for arg in args)
            + ")"
        )
        blocks = getattr(function, "blocks", None)
        if not isinstance(blocks, Mapping):
            return None
        for block_name, block in blocks.items():
            lines.append(f"  {block_name}:")
            for instruction in tuple(getattr(block, "instrs", ()) or ()):
                result = getattr(instruction, "res", None)
                prefix = "" if result is None else f"%t{_value_id(result)} = "
                operands = " ".join(
                    f"%t{_value_id(argument)}"
                    for argument in tuple(getattr(instruction, "args", ()) or ())
                )
                lines.append(
                    f"    {prefix}{getattr(instruction, 'op', '<unknown>')} {operands}".rstrip()
                )
            successors = tuple(getattr(block, "successors", ()) or ())
            if successors:
                lines.append("    ; successors: " + ", ".join(map(str, successors)))
    return "\n".join(lines) + "\n"


def _normalize_shape(shape: Any) -> list[int]:
    return [int(item) for item in tuple(shape or ())]


def _ssa_spelling(value_id: int) -> str:
    return f"%t{int(value_id)}"


def _nodus_dtype(dtype: Any) -> str:
    spelling = str(dtype).lower()
    aliases = {
        "double": "f64", "float64": "f64", "f64": "f64",
        "float": "f32", "float32": "f32", "f32": "f32",
        "int8": "i8", "i8": "i8", "int16": "i16", "i16": "i16",
        "int32": "i32", "i32": "i32", "int64": "i64", "i64": "i64",
        "uint8": "u8", "u8": "u8", "uint16": "u16", "u16": "u16",
        "uint32": "u32", "u32": "u32", "uint64": "u64", "u64": "u64",
        "bool": "bool", "i1": "bool",
    }
    try:
        return aliases[spelling]
    except KeyError as error:
        raise ProductError(f"NODUS BUFFER ABI V2 cannot spell dtype {dtype!r}") from error


def _quoted(value: Any) -> str:
    return json.dumps("" if value is None else str(value), ensure_ascii=False)


def _buffer_abi(
    request: RowToolRequest,
    artifact: Any,
    module: Any,
    outputs: Any,
    root: str,
) -> tuple[dict[str, Any], str]:
    function = _root_function(module, root)
    argument_ids = [_value_id(value) for value in tuple(getattr(function, "args", ()) or ())]
    output_ids = [_value_id(value) for value in _published_outputs(outputs, root)]
    overlap = set(argument_ids) & set(output_ids)
    if overlap:
        raise ProductError(
            "buffer ABI cannot classify values that are both inputs and outputs: "
            + repr(sorted(overlap))
        )
    input_columns = [column for column in request.columns if column.role in {"input", "inout"}]
    output_columns = [column for column in request.columns if column.role in {"output", "inout"}]
    if len(input_columns) != len(argument_ids):
        raise ProductError(
            "declared input/inout column count does not match root argument count: "
            f"{len(input_columns)} != {len(argument_ids)}"
        )
    if len(output_columns) != len(output_ids):
        raise ProductError(
            "declared output/inout column count does not match published output count: "
            f"{len(output_columns)} != {len(output_ids)}"
        )
    input_names = {
        value_id: input_columns[index].name
        for index, value_id in enumerate(argument_ids)
        if index < len(input_columns)
    }
    output_names = {
        value_id: output_columns[index].name
        for index, value_id in enumerate(output_ids)
        if index < len(output_columns)
    }
    columns_by_name = {column.name: column for column in request.columns}
    input_ids_by_name = {
        input_columns[index].name: value_id
        for index, value_id in enumerate(argument_ids)
    }
    order = tuple(int(item) for item in getattr(artifact, "buffer_order", ()) or ())
    dtypes = tuple(getattr(artifact, "buffer_dtypes", ()) or ())
    shapes = tuple(getattr(artifact, "buffer_shapes", ()) or ())
    if len(dtypes) not in {0, len(order)} or len(shapes) not in {0, len(order)}:
        raise ProductError("backend buffer metadata does not match buffer_order")
    if len(order) != len(set(order)):
        raise ProductError("backend buffer_order contains duplicate identities")
    missing_boundary = (set(argument_ids) | set(output_ids)) - set(order)
    if missing_boundary:
        raise ProductError(
            "backend buffer_order omits public values " + repr(sorted(missing_boundary))
        )
    buffers = []
    for index, value_id in enumerate(order):
        role = (
            "inout" if value_id in input_names and value_id in output_names
            else "input" if value_id in input_names
            else "output" if value_id in output_names
            else "workspace"
        )
        artifact_shape = _normalize_shape(shapes[index]) if shapes else []
        column_name = output_names.get(value_id, input_names.get(value_id))
        column = columns_by_name.get(column_name) if column_name else None
        shape_sources: list[dict[str, int] | None] = []
        if column is None:
            if any(extent <= 0 for extent in artifact_shape):
                raise ProductError(
                    f"workspace buffer {value_id} has an invalid shape {artifact_shape!r}"
                )
            rank = len(artifact_shape)
            shape: list[int | None] = list(artifact_shape)
            shape_sources = [None] * rank
        elif column.shape is not None:
            shape = list(column.shape)
            rank = column.rank
            if artifact_shape and artifact_shape != shape:
                raise ProductError(
                    f"backend shape {artifact_shape!r} disagrees with column "
                    f"{column.name!r} shape {shape!r}"
                )
            shape_sources = [None] * rank
        else:
            # A nonempty backend shape means the emitted program was
            # specialized despite the caller asking for a runtime-sized row.
            # Advertising it as dynamic would make the wrapper unsound.
            if artifact_shape:
                raise ProductError(
                    f"backend specialized dynamic column {column.name!r} to "
                    f"shape {artifact_shape!r}"
                )
            rank = column.rank
            shape = [None] * rank
            source_name = column.shape_from
            if role == "output" and column.role == "inout" and source_name is None:
                source_name = column.name
            if role in {"output", "workspace"}:
                if source_name is None:
                    raise ProductError(
                        f"dynamic {role} column {column.name!r} requires shape_from"
                    )
                source_id = input_ids_by_name.get(source_name)
                if source_id is None:
                    raise ProductError(
                        f"dynamic column {column.name!r} names unavailable shape source "
                        f"{source_name!r}"
                    )
                shape_sources = [
                    {"ssa_value_id": int(source_id), "axis": axis}
                    for axis in range(rank)
                ]
            else:
                source_id = (
                    int(value_id)
                    if source_name is None
                    else input_ids_by_name.get(source_name)
                )
                if source_id is None:
                    raise ProductError(
                        f"dynamic column {column.name!r} names unavailable shape source "
                        f"{source_name!r}"
                    )
                shape_sources = [
                    {"ssa_value_id": int(source_id), "axis": axis}
                    for axis in range(rank)
                ]
        buffers.append({
            "position": index,
            "ssa_value_id": value_id,
            "role": role,
            "column": column_name,
            "dtype": _nodus_dtype(dtypes[index] if dtypes else "float64"),
            "rank": rank,
            "shape": shape,
            "shape_sources": shape_sources,
        })
    extent_order = [list(item) for item in tuple(
        getattr(artifact, "extent_order", ()) or ()
    )]
    for extent in extent_order:
        if len(extent) != 3:
            raise ProductError(
                f"NODUS BUFFER ABI V2 extent must have value, kind, and axis; got {extent!r}"
            )
        value_id, kind, axis = extent
        kind = str(kind).lower()
        if kind in {"shape", "dim"}:
            if axis is None:
                raise ProductError(f"axis-addressed extent lacks an axis: {extent!r}")
        elif kind in {"element_count", "numel", "rank"}:
            if axis is not None:
                raise ProductError(f"whole-descriptor extent unexpectedly names an axis: {extent!r}")
        else:
            raise ProductError(f"unsupported NODUS BUFFER ABI V2 extent kind: {extent!r}")
        try:
            extent_buffer = next(
                item for item in buffers if item["ssa_value_id"] == int(value_id)
            )
        except StopIteration:
            raise ProductError(f"extent names unknown buffer {value_id!r}")
        if kind in {"shape", "dim"} and (
            int(axis) < 0 or int(axis) >= int(extent_buffer["rank"])
        ):
            raise ProductError(f"extent has invalid axis {axis!r}")
    tool_id = f"imported.{_safe_name(request.product_name or request.row_entry)}.{request.backend}"
    descriptor = {
        "schema": BUFFER_ABI_SCHEMA,
        "backend": request.backend,
        "symbol": str(getattr(artifact, "name")),
        "row_entry": request.row_entry,
        "columns": [column.to_mapping() for column in request.columns],
        "arguments": argument_ids,
        "outputs": output_ids,
        "buffer_order": list(order),
        "buffers": buffers,
        "extent_order": extent_order,
        "tool_id": tool_id,
        "tool_name": f"compiled {request.row_entry} ({request.backend})",
        "call": (
            "void symbol(void **buffers, int64_t *extents)"
            if request.backend == "c"
            else "void symbol(void **buffers, int32_t *extents)"
        ),
    }
    extent_dtype = "i64" if request.backend == "c" else "i32"
    descriptor["extent_dtype"] = extent_dtype
    lines = [
        "NODUS BUFFER ABI V2",
        f"BACKEND {_quoted(request.backend)}",
        f"TOOL {_quoted(tool_id)} {_quoted(descriptor['tool_name'])}",
        f"ENTRY {_quoted(descriptor['symbol'])}",
        f"EXTENT_DTYPE {_quoted(extent_dtype)}",
    ]
    for buffer in buffers:
        shape = buffer["shape"]
        lines.append(
            "VALUE "
            f"{str(buffer['role']).upper()} {_quoted(_ssa_spelling(buffer['ssa_value_id']))} "
            f"{_quoted(buffer['dtype'])} {buffer['rank']}"
            + (
                " " + " ".join("?" if extent is None else str(extent) for extent in shape)
                if shape else ""
            )
        )
    for value_id in order:
        lines.append(f"BUFFER {_quoted(_ssa_spelling(value_id))}")
    for value_id, kind, axis in descriptor["extent_order"]:
        normalized_kind = str(kind).lower()
        if normalized_kind in {"shape", "dim"}:
            selector = str(int(axis))
        elif normalized_kind in {"element_count", "numel"}:
            selector = "ELEMENT_COUNT"
        else:
            selector = "RANK"
        lines.append(f"EXTENT {_quoted(_ssa_spelling(value_id))} {selector}")
    for buffer in buffers:
        for axis, source in enumerate(buffer["shape_sources"]):
            if source is None or source["ssa_value_id"] == buffer["ssa_value_id"]:
                continue
            lines.append(
                f"SHAPE {_quoted(_ssa_spelling(buffer['ssa_value_id']))} {axis} "
                f"FROM {_quoted(_ssa_spelling(source['ssa_value_id']))} "
                f"{source['axis']}"
            )
    return descriptor, "\n".join(lines) + "\n"


def _artifact_record(root: Path, path: Path, kind: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        display = resolved.as_posix()
    return {
        "kind": kind,
        "path": display,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _product_relative(root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ProductError(
            f"adoptable artifact is outside the product root: {path}"
        ) from error
    if not relative.parts or any(part == ".." for part in relative.parts):
        raise ProductError(f"invalid product-relative artifact path: {relative}")
    return relative.as_posix()


def _write_adoption_receipt(
    root: Path,
    *,
    backend: str,
    manifest_path: Path,
    abi_path: Path,
    binary_path: Path,
) -> Path:
    """Publish the adopter's closed, product-root-relative handoff."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PRODUCT_SCHEMA or manifest.get("loadable") is not True:
        raise ProductError("adoption receipt requires a loadable product manifest")
    for path in (manifest_path, abi_path, binary_path):
        if not path.is_file():
            raise ProductError(f"adoption receipt artifact is missing: {path}")
    lines = [
        "NODUS ROW TOOL PRODUCT V1",
        f"BACKEND {_quoted(backend)}",
        f"MANIFEST {_quoted(_product_relative(root, manifest_path))}",
        f"ABI {_quoted(_product_relative(root, abi_path))}",
        f"BINARY {_quoted(_product_relative(root, binary_path))}",
    ]
    receipt = root / ADOPTION_RECEIPT_NAME
    _write_text(receipt, "\n".join(lines) + "\n")
    return receipt


def _compiler_head(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class TuringBoundary:
    """Late imports keep Turing external to Nodus and easy to test."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        if not (self.root / "src" / "compiler" / "fortran_c_shell.py").is_file():
            raise FileNotFoundError(f"Turing compiler not found at {self.root}")
        sys.path.insert(0, str(self.root))
        try:
            from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
            from src.compiler.project_compilation_product import (
                compiler_toolchain_fingerprint,
            )
            from src.compiler.ssa_c_backend import emit_ssa_module_to_c
            from src.compiler.ssa_fortran_backend import emit_module
            from src.compiler.ssa_llvm_backend import (
                compile_artifact,
                emit_ssa_function_to_llvm,
            )
            from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        finally:
            sys.path.pop(0)
        self._lower = lower_ast_source_to_ssa
        self._emit_c = emit_ssa_module_to_c
        self._emit_llvm = emit_ssa_function_to_llvm
        self._compile_llvm = compile_artifact
        self._emit_fortran = emit_module
        self._compile_fortran = compile_fortran_module_c_shell
        self.head = _compiler_head(self.root)
        self.toolchain = compiler_toolchain_fingerprint()

    def lower(self, source: str, entrypoint: str, **kwargs: Any) -> tuple[Any, Any, Any]:
        return self._lower(source, entrypoint, **kwargs)

    def emit_c(self, module: Any, root: str, *, entry_name: str) -> Any:
        return self._emit_c(module, root, entry_name=entry_name)

    @staticmethod
    def compile_c(artifact: Any, directory: Path) -> Any:
        return artifact.compile(directory)

    def emit_llvm(self, module: Any, root: str, *, entry_name: str) -> Any:
        return self._emit_llvm(module, root, entry_name=entry_name)

    def compile_llvm(self, artifact: Any, directory: Path) -> Any:
        return self._compile_llvm(artifact, directory=directory)

    def emit_fortran(
        self, module: Any, *, name: str, outputs: Any, exports: Sequence[str]
    ) -> Any:
        return self._emit_fortran(
            module, name=name, outputs=outputs, extra_roots=tuple(exports)
        )

    def compile_fortran(
        self, artifact: Any, directory: Path, *, entrypoint: str, name: str
    ) -> Any:
        return self._compile_fortran(
            artifact, {}, directory, entrypoint=entrypoint, name=name,
            standalone=False, library=True,
        )


def compile_row_tool(
    request: RowToolRequest | Mapping[str, Any],
    *,
    compiler: CompilerBoundary | None = None,
) -> dict[str, Any]:
    """Compile and retain one row product, returning its manifest mapping."""

    if not isinstance(request, RowToolRequest):
        request = RowToolRequest.from_mapping(request)
    if request.extraction_contract is None:
        raise ValueError("request requires an explicit extraction_contract")
    if request.backend == "fortran" and any(
        column.shape is None for column in request.columns
    ):
        raise ProductError(
            "runtime-sized columns require the C/LLVM buffer ABI; the Fortran "
            "compiled-program importer does not yet carry dynamic output shape sources"
        )
    output = request.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A rerun that does not produce a loadable product must not leave an older
    # valid-looking adoption handoff beside its new non-loadable manifest.
    (output / ADOPTION_RECEIPT_NAME).unlink(missing_ok=True)
    extraction_source = request.extraction_contract.resolve()
    if not extraction_source.is_file():
        raise FileNotFoundError(
            f"extraction contract does not exist: {extraction_source}"
        )
    product_name = _safe_name(request.product_name or request.row_entry)
    authored, authored_origin = _read_authored_source(request)
    submitted = _submitted_source(request, authored)
    authored_path = output / "authored-source.py"
    submitted_path = output / "submitted-source.py"
    retained_contract_path = output / "extraction-contract.yaml"
    compiler_work = output / "compiler-work"
    _write_text(authored_path, authored)
    _write_text(submitted_path, submitted)
    _write_bytes(retained_contract_path, extraction_source.read_bytes())

    if compiler is None:
        turing_root = (
            request.turing_root.resolve()
            if request.turing_root is not None
            else Path(__file__).resolve().parents[2] / "turing"
        )
        # Importing Turing activates compiler bootstrap modules which may emit
        # generated helper receipts relative to the process working directory.
        # Keep those products beside the rest of this compilation rather than
        # allowing the adapter to dirty the Nodus checkout before lowering even
        # begins.
        with _compiler_working_directory(compiler_work):
            compiler = TuringBoundary(turing_root)

    lower_options: dict[str, Any] = {
        "name": product_name,
        "runtime_closure_only": True,
    }
    lower_options["extraction_contract"] = retained_contract_path
    if request.ssa_composition_directory is not None:
        composition_directory = request.ssa_composition_directory.resolve()
        composition_directory.mkdir(parents=True, exist_ok=True)
        lower_options["partitioned_directory"] = composition_directory
        lower_options["partitioned_options"] = dict(request.ssa_composition_options)
    with _compiler_working_directory(compiler_work):
        module, outputs, exports_value = compiler.lower(
            submitted, request.row_entry, **lower_options
        )
    exports = tuple(map(str, exports_value or ()))
    if not exports:
        raise ProductError("Turing lowering produced no exported row entry")
    root = exports[0]
    _root_function(module, root)

    ssa_path = output / "repository-ssa.pkl"
    with ssa_path.open("wb") as stream:
        pickle.dump((module, outputs, exports), stream, protocol=5)
    textual = _render_textual_ssa(module, exports)
    textual_path = None
    if textual is not None:
        textual_path = output / "repository-ssa.txt"
        _write_text(textual_path, textual)

    backend_dir = output / "backend"
    backend_dir.mkdir(parents=True, exist_ok=True)
    binary_path: Path | None = None
    abi_path: Path | None = None
    backend_source_path: Path
    build_error: dict[str, str] | None = None
    buffer_abi_manifest: dict[str, Any] | None = None

    with _compiler_working_directory(compiler_work):
        if request.backend == "c":
            artifact = compiler.emit_c(module, root, entry_name=product_name)
            if not bool(getattr(artifact, "complete", False)):
                raise ProductError(f"C emission incomplete: {getattr(artifact, 'shortfalls', ())!r}")
            backend_source_path = backend_dir / f"{artifact.name}.c"
            _write_text(backend_source_path, str(artifact.source))
            abi_path = backend_dir / f"{artifact.name}.nodus_buffer_abi.txt"
            buffer_abi_manifest, buffer_abi_text = _buffer_abi(
                request, artifact, module, outputs, root
            )
            _write_text(abi_path, buffer_abi_text)
            try:
                built = compiler.compile_c(artifact, backend_dir)
                candidate = getattr(built, "library_path", None)
                if candidate is not None and Path(candidate).is_file():
                    binary_path = Path(candidate)
            except Exception as error:  # build receipt is retained, never hidden
                build_error = {"type": type(error).__name__, "message": str(error)}
        elif request.backend == "llvm":
            artifact = compiler.emit_llvm(module, root, entry_name=product_name)
            if not bool(getattr(artifact, "complete", False)):
                raise ProductError(
                    f"LLVM emission incomplete: {getattr(artifact, 'shortfalls', ())!r}"
                )
            backend_source_path = backend_dir / f"{artifact.name}.ll"
            _write_text(backend_source_path, str(artifact.llvm_ir))
            abi_path = backend_dir / f"{artifact.name}.nodus_buffer_abi.txt"
            buffer_abi_manifest, buffer_abi_text = _buffer_abi(
                request, artifact, module, outputs, root
            )
            _write_text(abi_path, buffer_abi_text)
            try:
                built = compiler.compile_llvm(artifact, backend_dir)
                candidate = getattr(built, "library_path", None)
                if candidate is not None and Path(candidate).is_file():
                    binary_path = Path(candidate)
            except Exception as error:
                build_error = {"type": type(error).__name__, "message": str(error)}
        else:
            artifact = compiler.emit_fortran(
                module, name=product_name, outputs=outputs, exports=exports
            )
            if not bool(getattr(artifact, "complete", False)):
                raise ProductError(
                    f"Fortran emission incomplete: {getattr(artifact, 'shortfalls', ())!r}"
                )
            backend_source_path = Path(artifact.write(backend_dir))
            candidate_api = backend_source_path.with_suffix(".api.yaml")
            if candidate_api.is_file():
                abi_path = candidate_api
            try:
                built = compiler.compile_fortran(
                    artifact, backend_dir, entrypoint=root, name=product_name
                )
                candidate = getattr(built, "executable_path", None)
                if candidate is not None and Path(candidate).is_file():
                    binary_path = Path(candidate)
                built_api = getattr(built, "api_path", None)
                if built_api is not None and Path(built_api).is_file():
                    abi_path = Path(built_api)
            except Exception as error:
                build_error = {"type": type(error).__name__, "message": str(error)}

    if build_error is not None and not request.allow_uncompiled:
        raise ProductError(
            f"native {request.backend} build failed: "
            f"{build_error['type']}: {build_error['message']}"
        )
    if binary_path is None and build_error is None:
        build_error = {
            "type": "MissingBinary",
            "message": "backend build returned without a compiled library",
        }
        if not request.allow_uncompiled:
            raise ProductError(build_error["message"])

    artifact_records = [
        _artifact_record(output, authored_path, "authored-source"),
        _artifact_record(output, submitted_path, "submitted-source"),
        _artifact_record(output, retained_contract_path, "extraction-contract"),
        _artifact_record(output, ssa_path, "repository-ssa"),
        _artifact_record(output, backend_source_path, "backend-source"),
    ]
    if textual_path is not None:
        artifact_records.append(_artifact_record(output, textual_path, "diagnostic-ssa"))
    if abi_path is not None:
        artifact_records.append(_artifact_record(output, abi_path, "abi-descriptor"))
    if binary_path is not None:
        artifact_records.append(_artifact_record(output, binary_path, "native-library"))
    tracked_paths = {str(Path(record["path"])) for record in artifact_records}
    for side_product in sorted(path for path in compiler_work.rglob("*") if path.is_file()):
        relative = _product_relative(output, side_product)
        if str(Path(relative)) in tracked_paths:
            continue
        artifact_records.append(
            _artifact_record(output, side_product, "compiler-side-product")
        )
        tracked_paths.add(str(Path(relative)))

    loadable = binary_path is not None and abi_path is not None
    manifest = {
        "schema": PRODUCT_SCHEMA,
        "request_schema": REQUEST_SCHEMA,
        "product_name": product_name,
        "backend": request.backend,
        "row_entry": request.row_entry,
        "ssa_root": root,
        "exports": list(exports),
        "columns": [column.to_mapping() for column in request.columns],
        "source": {
            "origin": authored_origin,
            "global_scope_wrapped": request.global_scope,
            "authored_sha256": _sha256(authored_path),
            "submitted_sha256": _sha256(submitted_path),
            "extraction_contract": "extraction-contract.yaml",
            "extraction_contract_sha256": _sha256(retained_contract_path),
        },
        "composition": (
            None if request.ssa_composition_directory is None else {
                "mechanism": "turing-public-partitioned-ssa-composition",
                "directory": request.ssa_composition_directory.resolve().as_posix(),
                "options": dict(request.ssa_composition_options),
            }
        ),
        "compiler": {
            "public_entry": "src.compiler.fortran_c_shell.lower_ast_source_to_ssa",
            "git_head": str(compiler.head),
            "toolchain_fingerprint": dict(compiler.toolchain),
        },
        "loadable": loadable,
        "build_error": build_error,
        "buffer_abi": buffer_abi_manifest,
        "adoption_receipt": ADOPTION_RECEIPT_NAME if loadable else None,
        "artifacts": artifact_records,
    }
    manifest_path = output / "nodus-row-tool-product.json"
    _write_json(manifest_path, manifest)
    if loadable:
        assert abi_path is not None and binary_path is not None
        _write_adoption_receipt(
            output,
            backend=request.backend,
            manifest_path=manifest_path,
            abi_path=abi_path,
            binary_path=binary_path,
        )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help="override request output_dir")
    parser.add_argument("--turing-root", type=Path, help="override request turing_root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    request_data = json.loads(arguments.request.read_text(encoding="utf-8"))
    if arguments.output_dir is not None:
        request_data["output_dir"] = str(arguments.output_dir)
    if arguments.turing_root is not None:
        request_data["turing_root"] = str(arguments.turing_root)
    try:
        manifest = compile_row_tool(request_data)
    except Exception as error:
        print(f"turing-row-tool: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print((Path(request_data["output_dir"]).resolve() / "nodus-row-tool-product.json"))
    return 0 if manifest["loadable"] or request_data.get("allow_uncompiled", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
