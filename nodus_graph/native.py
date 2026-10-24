from __future__ import annotations

import ctypes
import pathlib
import struct
import uuid
from dataclasses import dataclass
from typing import Iterable, Optional, Union

from nodus_headless_bridge import (
    CanvasEdgeDesc,
    CanvasModuleDesc,
    load_headless_lib,
)

from .model import (
    BuiltinTool,
    ConnectorDefinition,
    ConnectorRef,
    EdgeDefinition,
    ExecutionContract,
    FifoConfig,
    FifoState,
    GraphDefinition,
    Input,
    ModuleDefinition,
    Output,
    PluginTool,
    PortDefinition,
    PublishResult,
    SchedulerMode,
    StackItem,
    StackSnapshot,
    Tool,
)


class UnsupportedNativeContract(RuntimeError):
    pass


class _NativeFifoConfig(ctypes.Structure):
    _fields_ = [
        ("token_bytes", ctypes.c_int32),
        ("slots", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("type_id", ctypes.c_int32),
        ("layout", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("backend_type", ctypes.c_int32),
    ]


class _NativeRow(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("tool", ctypes.c_int32),
        ("attachment_count", ctypes.c_int32),
        ("tool_origin", ctypes.c_int32),
        ("plugin_id", ctypes.c_char_p),
    ]


class _NativePublishResult(ctypes.Structure):
    _fields_ = [("accepted", ctypes.c_int32), ("dropped", ctypes.c_int32)]


class _DeploymentReport(ctypes.Structure):
    _fields_ = [
        ("abi_loaded", ctypes.c_int32),
        ("source_retained", ctypes.c_int32),
        ("request_written", ctypes.c_int32),
        ("compiler_invoked", ctypes.c_int32),
        ("wrapper_staged", ctypes.c_int32),
        ("plugin_loaded", ctypes.c_int32),
    ]


class _NativeRowInfo(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("tool", ctypes.c_int32),
        ("attachment_count", ctypes.c_int32),
        ("tool_origin", ctypes.c_int32),
        ("contact_idx", ctypes.c_int32),
        ("consumes", ctypes.c_int32),
        ("produces", ctypes.c_int32),
    ]


class _TuringAdoptionReport(ctypes.Structure):
    _fields_ = [
        ("receipt_validated", ctypes.c_int32),
        ("wrapper_imported", ctypes.c_int32),
        ("wrapper_staged", ctypes.c_int32),
        ("plugin_loaded", ctypes.c_int32),
    ]


class _TensorDesc(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_uint8),
        ("layout", ctypes.c_uint8),
        ("is_readonly", ctypes.c_uint8),
        ("is_buffer", ctypes.c_uint8),
        ("rank", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * 8),
        ("strides", ctypes.c_uint64 * 8),
        ("element_count", ctypes.c_uint64),
        ("element_bytes", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("total_bytes", ctypes.c_uint64),
    ]


def _fifo(config: FifoConfig) -> _NativeFifoConfig:
    return _NativeFifoConfig(
        config.token_bytes,
        config.slots,
        config.top_k,
        config.type_id,
        config.layout,
        config.dtype,
        {"host": 0, "vulkan": 3}[config.storage],
    )


def _fifo_from_native(config: _NativeFifoConfig) -> FifoConfig:
    return FifoConfig(
        token_bytes=config.token_bytes,
        slots=config.slots,
        top_k=config.top_k,
        type_id=config.type_id,
        layout=config.layout,
        dtype=config.dtype,
        storage={0: "host", 3: "vulkan"}.get(
            config.backend_type, f"unsupported:{config.backend_type}"
        ),
    )


def _contract_from_native(value: int) -> ExecutionContract:
    try:
        return {
            0: ExecutionContract.SCALAR_F32,
            1: ExecutionContract.ABSTRACT_TENSOR_POINTER,
            2: ExecutionContract.OPAQUE,
        }[value]
    except KeyError as exc:
        raise RuntimeError(f"unknown native module stack contract {value}") from exc


def _load(lib_path: Optional[str] = None):
    lib = load_headless_lib(lib_path)
    lib.nodus_headless_get_module_count.restype = ctypes.c_int32
    lib.nodus_headless_get_module_count.argtypes = (ctypes.c_void_p,)
    lib.nodus_headless_get_module_desc.restype = ctypes.c_int32
    lib.nodus_headless_get_module_desc.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(CanvasModuleDesc),
    )
    lib.nodus_headless_set_module_desc.restype = ctypes.c_int32
    lib.nodus_headless_set_module_desc.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(CanvasModuleDesc),
    )
    lib.nodus_headless_set_module_rows.restype = ctypes.c_int32
    lib.nodus_headless_set_module_rows.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(_NativeRow), ctypes.c_int32,
    )
    lib.nodus_headless_get_module_row_count.restype = ctypes.c_int32
    lib.nodus_headless_get_module_row_count.argtypes = (
        ctypes.c_void_p, ctypes.c_int32,
    )
    lib.nodus_headless_get_module_row.restype = ctypes.c_int32
    lib.nodus_headless_get_module_row.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(_NativeRowInfo), ctypes.c_char_p, ctypes.c_int32,
    )
    lib.nodus_headless_set_module_stack_contract.restype = ctypes.c_int32
    lib.nodus_headless_set_module_stack_contract.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
    )
    lib.nodus_headless_get_module_stack_contract.restype = ctypes.c_int32
    lib.nodus_headless_get_module_stack_contract.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
    )
    lib.nodus_headless_export_module_to_root.restype = ctypes.c_int32
    lib.nodus_headless_export_module_to_root.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p,
    )
    lib.nodus_headless_configure_edge_fifo.restype = ctypes.c_int32
    lib.nodus_headless_configure_edge_fifo.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(_NativeFifoConfig),
    )
    lib.nodus_headless_get_edge_count.restype = ctypes.c_int32
    lib.nodus_headless_get_edge_count.argtypes = (ctypes.c_void_p,)
    lib.nodus_headless_get_edge.restype = ctypes.c_int32
    lib.nodus_headless_get_edge.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(CanvasEdgeDesc),
        ctypes.POINTER(ctypes.c_int32),
    )
    lib.nodus_headless_attach_edge.restype = ctypes.c_int32
    lib.nodus_headless_attach_edge.argtypes = (
        ctypes.c_void_p, ctypes.POINTER(CanvasEdgeDesc),
    )
    lib.nodus_headless_get_edge_fifo_config.restype = ctypes.c_int32
    lib.nodus_headless_get_edge_fifo_config.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(_NativeFifoConfig),
    )
    lib.nodus_headless_get_edge_unread.restype = ctypes.c_int32
    lib.nodus_headless_get_edge_unread.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
    )
    lib.nodus_headless_push_token_ex.restype = ctypes.c_int32
    lib.nodus_headless_push_token_ex.argtypes = (
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(_NativePublishResult),
    )
    lib.nodus_headless_get_port_fifo_config.restype = ctypes.c_int32
    lib.nodus_headless_get_port_fifo_config.argtypes = (
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(_NativeFifoConfig),
    )
    lib.nodus_headless_get_port_count.restype = ctypes.c_int32
    lib.nodus_headless_get_port_count.argtypes = (
        ctypes.c_void_p, ctypes.c_int32,
    )
    lib.nodus_headless_get_port_at.restype = ctypes.c_int32
    lib.nodus_headless_get_port_at.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(_NativeFifoConfig),
    )
    lib.nodus_headless_get_port_unread.restype = ctypes.c_int32
    lib.nodus_headless_get_port_unread.argtypes = (
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    )
    lib.nodus_headless_get_stack_snapshot.restype = ctypes.c_int32
    lib.nodus_headless_get_stack_snapshot.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    )
    lib.nodus_headless_get_stack_snapshot_meta.restype = ctypes.c_int32
    lib.nodus_headless_get_stack_snapshot_meta.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
    )
    lib.nodus_headless_get_stack_tail.restype = ctypes.c_int32
    lib.nodus_headless_get_stack_tail.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
    )
    lib.nodus_headless_declare_input_port_ex.restype = ctypes.c_int32
    lib.nodus_headless_declare_input_port_ex.argtypes = (
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(_NativeFifoConfig),
    )
    lib.nodus_headless_declare_output_port_ex.restype = ctypes.c_int32
    lib.nodus_headless_declare_output_port_ex.argtypes = (
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(_NativeFifoConfig), ctypes.c_int32,
    )
    lib.nodus_headless_set_module_uuid.restype = ctypes.c_int32
    lib.nodus_headless_set_module_uuid.argtypes = (ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint64)
    lib.nodus_headless_get_module_uuid.restype = ctypes.c_uint64
    lib.nodus_headless_get_module_uuid.argtypes = (ctypes.c_void_p, ctypes.c_int32)
    lib.nodus_headless_find_module_by_uuid.restype = ctypes.c_int32
    lib.nodus_headless_find_module_by_uuid.argtypes = (ctypes.c_void_p, ctypes.c_uint64)
    lib.nodus_headless_set_scheduler_mode.restype = ctypes.c_int32
    lib.nodus_headless_set_scheduler_mode.argtypes = (ctypes.c_void_p, ctypes.c_int32)
    lib.nodus_headless_get_scheduler_mode.restype = ctypes.c_int32
    lib.nodus_headless_get_scheduler_mode.argtypes = (
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
    )
    lib.nodus_headless_abstract_tensor_type_id.restype = ctypes.c_int32
    lib.nodus_headless_abstract_tensor_type_id.argtypes = ()
    lib.nodus_headless_deploy_python_row.restype = ctypes.c_int32
    lib.nodus_headless_deploy_python_row.argtypes = (
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.POINTER(_DeploymentReport), ctypes.c_char_p, ctypes.c_int32,
        ctypes.c_char_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32,
    )
    lib.nodus_headless_unload_plugin.restype = ctypes.c_int32
    lib.nodus_headless_unload_plugin.argtypes = (ctypes.c_char_p,)
    lib.nodus_headless_build_plugin_source.restype = ctypes.c_int32
    lib.nodus_headless_build_plugin_source.argtypes = (
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_int32,
    )
    lib.nodus_headless_compile_turing_row.restype = ctypes.c_int32
    lib.nodus_headless_compile_turing_row.argtypes = (
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(_TuringAdoptionReport),
        ctypes.c_char_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32,
    )
    lib.nodus_headless_tensor_create.restype = ctypes.c_uint64
    lib.nodus_headless_tensor_create.argtypes = (
        ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64),
    )
    lib.nodus_headless_tensor_write.restype = ctypes.c_int64
    lib.nodus_headless_tensor_write.argtypes = (
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64,
    )
    lib.nodus_headless_tensor_read.restype = ctypes.c_int64
    lib.nodus_headless_tensor_read.argtypes = (
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64,
    )
    lib.nodus_headless_tensor_describe.restype = ctypes.c_int32
    lib.nodus_headless_tensor_describe.argtypes = (
        ctypes.c_uint64, ctypes.POINTER(_TensorDesc),
    )
    lib.nodus_headless_tensor_destroy.restype = None
    lib.nodus_headless_tensor_destroy.argtypes = (ctypes.c_uint64,)
    return lib


@dataclass(frozen=True)
class Contact:
    module: "Module"
    direction: str
    ordinal: int

    @property
    def native_index(self) -> int:
        self.module._assert_live()
        if self.direction == "input":
            return self.ordinal
        return self.module.input_count + self.ordinal

    @property
    def reference(self) -> ConnectorRef:
        return ConnectorRef(self.module.stable_id, self.direction, self.ordinal)


@dataclass(frozen=True)
class ToolRow:
    module: "Module"
    semantic_row_index: int


@dataclass(frozen=True)
class Edge:
    graph: "Graph"
    native_index: int
    source: Contact
    target: Contact
    fifo: Optional[FifoConfig]
    type_id: int = 0


Row = Union[Input, Output, Tool, PluginTool]


class Module:
    def __init__(self, graph: "Graph", index: int, stable_id: int, *,
                 label: str = "", x: int = 0, y: int = 0,
                 width: int = 420, height: int = 320):
        self.graph = graph
        self.index = index
        self.stable_id = stable_id
        self.label = label
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.input_count = 0
        self.output_count = 0
        self.contract = ExecutionContract.OPAQUE
        self.rows: tuple[Row, ...] = ()
        self._generation = graph._generation

    def _assert_live(self) -> None:
        if not self.graph._handle or self._generation != self.graph._generation:
            raise UnsupportedNativeContract("module handle is stale after graph close/load")

    def set_rows(self, *rows: Row) -> "Module":
        self._assert_live()
        stack_depth = 0
        input_count = 0
        output_count = 0
        contract: Optional[ExecutionContract] = None
        native_rows = (_NativeRow * len(rows))()
        plugin_bytes: list[bytes] = []

        for index, row in enumerate(rows):
            if isinstance(row, Input):
                if not 1 <= row.count <= 32:
                    raise ValueError("Input row attachment count must be in 1..32")
                stack_depth += row.count
                input_count += row.count
                native_rows[index] = _NativeRow(0, 0, row.count, 0, None)
            elif isinstance(row, Output):
                if not 1 <= row.count <= 32:
                    raise ValueError("Output row attachment count must be in 1..32")
                if stack_depth < row.count:
                    raise ValueError(f"output row {index} underflows the stack")
                stack_depth -= row.count
                output_count += row.count
                native_rows[index] = _NativeRow(2, 0, row.count, 0, None)
            elif isinstance(row, Tool):
                consumes, produces = row.stack_effect
                if stack_depth < consumes:
                    raise ValueError(f"tool row {index} ({row.kind.name}) underflows the stack")
                stack_depth += produces - consumes
                if contract not in (None, row.contract):
                    raise ValueError("one module cannot mix scalar and pointer stack contracts")
                contract = row.contract
                native_rows[index] = _NativeRow(1, int(row.kind), 0, 0, None)
            elif isinstance(row, PluginTool):
                if stack_depth < row.consumes:
                    raise ValueError(f"plugin row {index} ({row.plugin_id}) underflows the stack")
                stack_depth += row.produces - row.consumes
                if contract not in (None, row.contract):
                    raise ValueError("one module cannot mix scalar and pointer stack contracts")
                contract = row.contract
                encoded = row.plugin_id.encode("utf-8")
                plugin_bytes.append(encoded)
                native_rows[index] = _NativeRow(1, 0, 0, 1, encoded)
            else:
                raise TypeError(f"unsupported row {row!r}")

        if stack_depth:
            raise ValueError(f"row program leaves {stack_depth} value(s) on the stack")
        ok = self.graph.lib.nodus_headless_set_module_rows(
            self.graph._handle, self.index, native_rows, len(rows),
        )
        if not ok:
            raise RuntimeError("native module row configuration failed")
        resolved_contract = contract or ExecutionContract.OPAQUE
        native_contract = {
            ExecutionContract.SCALAR_F32: 0,
            ExecutionContract.ABSTRACT_TENSOR_POINTER: 1,
            ExecutionContract.OPAQUE: 2,
        }[resolved_contract]
        if not self.graph.lib.nodus_headless_set_module_stack_contract(
            self.graph._handle, self.index, native_contract,
        ):
            raise RuntimeError("native module stack contract configuration failed")
        self.rows = tuple(rows)
        self.input_count = input_count
        self.output_count = output_count
        self.contract = resolved_contract
        return self

    def input(self, ordinal: int) -> Contact:
        self._assert_live()
        if not 0 <= ordinal < self.input_count:
            raise IndexError("input contact ordinal out of range")
        return Contact(self, "input", ordinal)

    @property
    def native_contract(self) -> ExecutionContract:
        """Read the stack/FIFO transfer choice held by the native runtime."""
        self._assert_live()
        value = ctypes.c_int32()
        if not self.graph.lib.nodus_headless_get_module_stack_contract(
            self.graph._handle, self.index, ctypes.byref(value),
        ):
            raise RuntimeError("native module stack contract query failed")
        return _contract_from_native(value.value)

    def output(self, ordinal: int) -> Contact:
        self._assert_live()
        if not 0 <= ordinal < self.output_count:
            raise IndexError("output contact ordinal out of range")
        return Contact(self, "output", ordinal)

    def stack_snapshot(self, row_index: int) -> StackSnapshot:
        """Return the native snapshot captured by a StackDisplay row.

        Other row kinds have no snapshot entry and return empty tuples. This is
        live inspection after a completed step, not persisted graph state.
        """
        self._assert_live()
        if not 0 <= row_index < len(self.rows):
            raise IndexError("semantic row index out of range or row metadata unavailable")
        return self.graph._stack_snapshot(self.index, row_index)

    def stack_tail(self) -> tuple[float, ...]:
        self._assert_live()
        return self.graph._stack_tail(self.index)

    def export_to_root(self, output_root: Union[str, pathlib.Path]) -> pathlib.Path:
        """Export this module's concrete Nodus table serialization."""
        self._assert_live()
        root = pathlib.Path(output_root)
        if not self.graph.lib.nodus_headless_export_module_to_root(
            self.graph._handle, self.index, str(root).encode(),
        ):
            raise RuntimeError(f"native module export failed: {root}")
        return root / "serialized" / f"module_{self.index}.gpmod"


class Graph:
    _active = False

    def __init__(self, lib_path: Optional[str] = None, *, scheduler: SchedulerMode = SchedulerMode.SCHEDULED):
        if Graph._active:
            raise UnsupportedNativeContract(
                "Nodus canvas contact lookup is process-global; the Python API permits one live native graph per process"
            )
        self.lib = _load(lib_path)
        self._handle = self.lib.nodus_headless_create()
        if not self._handle:
            raise RuntimeError("nodus_headless_create failed")
        Graph._active = True
        self._generation = 0
        self.modules: list[Module] = []
        self.edges: list[Edge] = []
        self._loaded_unattached_edges: set[int] = set()
        self._unsupported: list[str] = []
        self.scheduler = scheduler
        self._input_ports: dict[str, PortDefinition] = {}
        self._output_ports: dict[str, PortDefinition] = {}
        self._connectors: dict[ConnectorRef, ConnectorDefinition] = {}
        if not self.lib.nodus_headless_set_scheduler_mode(self._handle, int(scheduler)):
            self.close()
            raise RuntimeError("native scheduler mode configuration failed")

    def __enter__(self) -> "Graph":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        if self._handle:
            self.lib.nodus_headless_destroy(self._handle)
            self._handle = None
            Graph._active = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def add_module(self, label: str, *, x: int = 0, y: int = 0,
                   width: int = 420, height: int = 320,
                   stable_id: Optional[int] = None) -> Module:
        identity = stable_id if stable_id is not None else (uuid.uuid4().int & ((1 << 64) - 1)) or 1
        if int(self.lib.nodus_headless_find_module_by_uuid(self._handle, identity)) >= 0:
            raise ValueError(f"stable module id {identity} is already in use")
        desc = CanvasModuleDesc()
        desc.x, desc.y, desc.w, desc.h = x, y, width, height
        encoded = label.encode("utf-8")[:63]
        desc.label = encoded + b"\0" * (64 - len(encoded))
        index = self.lib.nodus_headless_add_module(self._handle, ctypes.byref(desc))
        if index < 0:
            raise RuntimeError("native add_module failed")
        if not self.lib.nodus_headless_set_module_uuid(self._handle, index, identity):
            raise RuntimeError("native module UUID assignment failed")
        module = Module(
            self, index, identity, label=label, x=x, y=y,
            width=width, height=height,
        )
        self.modules.append(module)
        return module

    def attach_module(self, index: int, *, input_count: int, output_count: int,
                      contract: ExecutionContract = ExecutionContract.SCALAR_F32) -> Module:
        identity = int(self.lib.nodus_headless_get_module_uuid(self._handle, index))
        if not identity:
            raise ValueError(f"no native module at index {index}")
        module = Module(self, index, identity)
        module.input_count = input_count
        module.output_count = output_count
        module.contract = contract
        native_contract = {
            ExecutionContract.SCALAR_F32: 0,
            ExecutionContract.ABSTRACT_TENSOR_POINTER: 1,
            ExecutionContract.OPAQUE: 2,
        }[contract]
        if not self.lib.nodus_headless_set_module_stack_contract(
            self._handle, index, native_contract,
        ):
            raise RuntimeError("native module stack contract configuration failed")
        self.modules.append(module)
        return module

    def module_by_id(self, stable_id: int, *, input_count: int, output_count: int,
                     contract: ExecutionContract = ExecutionContract.SCALAR_F32) -> Module:
        """Attach to a restored module by persistent identity, not load-local index."""
        index = int(self.lib.nodus_headless_find_module_by_uuid(self._handle, stable_id))
        if index < 0:
            raise KeyError(f"no native module with stable id {stable_id}")
        return self.attach_module(
            index, input_count=input_count, output_count=output_count, contract=contract,
        )

    def connect(self, source: Contact, target: Contact, *,
                fifo: Optional[FifoConfig] = FifoConfig.scalar_f32(), type_id: int = 0) -> Edge:
        if source.direction != "output" or target.direction != "input":
            raise ValueError("connect requires an output contact followed by an input contact")
        if source.module.graph is not self or target.module.graph is not self:
            raise ValueError("both contacts must belong to this graph")
        desc = CanvasEdgeDesc(
            a_module=source.module.index,
            a_contact_idx=source.native_index,
            b_module=target.module.index,
            b_contact_idx=target.native_index,
        )
        index = self.lib.nodus_headless_add_edge(self._handle, ctypes.byref(desc), type_id)
        if index < 0:
            raise RuntimeError("native add_edge failed")
        if fifo is not None:
            native_fifo = _fifo(fifo)
            if not self.lib.nodus_headless_configure_edge_fifo(self._handle, index, ctypes.byref(native_fifo)):
                raise RuntimeError("native edge FIFO configuration failed")
        edge = Edge(self, index, source, target, fifo, type_id)
        self.edges.append(edge)
        return edge

    def attach_edge(self, source: Contact, target: Contact, *,
                    fifo: Optional[FifoConfig] = FifoConfig.scalar_f32()) -> Edge:
        """Reattach process-local FIFO semantics to a native-restored edge.

        CANVAS V4 persists topology but not FIFO specs, reader positions, or
        live samples. Every restored edge must therefore be named here before
        execution; passing ``None`` preserves deliberate absence.
        """
        if source.direction != "output" or target.direction != "input":
            raise ValueError("attach_edge requires an output contact followed by an input contact")
        if source.module.graph is not self or target.module.graph is not self:
            raise ValueError("both contacts must belong to this graph")
        desc = CanvasEdgeDesc(
            a_module=source.module.index,
            a_contact_idx=source.native_index,
            b_module=target.module.index,
            b_contact_idx=target.native_index,
        )
        index = int(self.lib.nodus_headless_attach_edge(self._handle, ctypes.byref(desc)))
        if index < 0:
            raise KeyError("no restored native edge matches those contacts")
        if fifo is not None:
            native_fifo = _fifo(fifo)
            if not self.lib.nodus_headless_configure_edge_fifo(self._handle, index, ctypes.byref(native_fifo)):
                raise RuntimeError("native restored-edge FIFO configuration failed")
        edge = Edge(self, index, source, target, fifo, 0)
        self.edges = [prior for prior in self.edges if prior.native_index != index]
        self.edges.append(edge)
        self._loaded_unattached_edges.discard(index)
        return edge

    def bind_input(self, name: str, contact: Contact, *, fifo: FifoConfig = FifoConfig.scalar_f32()) -> None:
        if contact.direction != "input":
            raise ValueError("input port requires an input contact")
        if name in self._input_ports:
            raise ValueError(f"input port {name!r} is already bound")
        config = _fifo(fifo)
        if not self.lib.nodus_headless_declare_input_port_ex(
            self._handle, name.encode(), contact.module.index, contact.native_index, ctypes.byref(config),
        ):
            raise RuntimeError(f"native input port declaration failed: {name}")
        self._input_ports[name] = PortDefinition(name, contact.reference, fifo)

    def bind_output(self, name: str, contact: Contact, *, fifo: FifoConfig = FifoConfig.scalar_f32(),
                    start_at_head: bool = True) -> None:
        if contact.direction != "output":
            raise ValueError("output port requires an output contact")
        if name in self._output_ports:
            raise ValueError(f"output port {name!r} is already bound")
        config = _fifo(fifo)
        if not self.lib.nodus_headless_declare_output_port_ex(
            self._handle, name.encode(), contact.module.index, contact.native_index,
            ctypes.byref(config), int(start_at_head),
        ):
            raise RuntimeError(f"native output port declaration failed: {name}")
        self._output_ports[name] = PortDefinition(name, contact.reference, fifo, start_at_head)

    def definition(self) -> GraphDefinition:
        """Read back the definition from the concrete native graph.

        Canvas/Table state is queried rather than copied from ``self.modules``
        and ``self.edges``. Logical connector tensor contracts remain schema
        metadata owned by this Nodus communication surface; their references
        are checked against the native row arity before being returned.
        """
        scheduler_value = ctypes.c_int32()
        if not self.lib.nodus_headless_get_scheduler_mode(
            self._handle, ctypes.byref(scheduler_value),
        ):
            raise RuntimeError("native scheduler mode query failed")
        try:
            scheduler = SchedulerMode(scheduler_value.value)
        except ValueError as exc:
            raise RuntimeError(
                f"unknown native scheduler mode {scheduler_value.value}"
            ) from exc

        module_count = int(self.lib.nodus_headless_get_module_count(self._handle))
        if module_count < 0:
            raise RuntimeError("native module count query failed")
        modules: list[ModuleDefinition] = []
        module_ids: dict[int, int] = {}
        input_counts: dict[int, int] = {}
        output_counts: dict[int, int] = {}
        for module_index in range(module_count):
            desc = CanvasModuleDesc()
            if not self.lib.nodus_headless_get_module_desc(
                self._handle, module_index, ctypes.byref(desc),
            ):
                raise RuntimeError(f"native module {module_index} descriptor query failed")
            stable_id = int(self.lib.nodus_headless_get_module_uuid(
                self._handle, module_index,
            ))
            if not stable_id:
                raise RuntimeError(f"native module {module_index} has no stable identity")
            contract_value = ctypes.c_int32()
            if not self.lib.nodus_headless_get_module_stack_contract(
                self._handle, module_index, ctypes.byref(contract_value),
            ):
                raise RuntimeError(f"native module {module_index} contract query failed")
            contract = _contract_from_native(contract_value.value)
            row_count = int(self.lib.nodus_headless_get_module_row_count(
                self._handle, module_index,
            ))
            if row_count < 0:
                raise RuntimeError(f"native module {module_index} row count query failed")
            rows: list[Row] = []
            input_count = 0
            output_count = 0
            for row_index in range(row_count):
                info = _NativeRowInfo()
                plugin_id = ctypes.create_string_buffer(1024)
                if not self.lib.nodus_headless_get_module_row(
                    self._handle, module_index, row_index, ctypes.byref(info),
                    ctypes.cast(plugin_id, ctypes.c_char_p), len(plugin_id),
                ):
                    raise RuntimeError(
                        f"native module {module_index} row {row_index} query failed"
                    )
                if info.kind == 0:
                    row = Input(info.attachment_count)
                    input_count += info.attachment_count
                elif info.kind == 2:
                    row = Output(info.attachment_count)
                    output_count += info.attachment_count
                elif info.kind == 1 and info.tool_origin == 0:
                    row = Tool(BuiltinTool(info.tool))
                elif info.kind == 1 and info.tool_origin == 1:
                    decoded = plugin_id.value.decode("utf-8")
                    row = PluginTool(decoded, info.consumes, info.produces, contract)
                else:
                    raise RuntimeError(
                        f"native module {module_index} row {row_index} has "
                        f"unknown kind/origin {info.kind}/{info.tool_origin}"
                    )
                rows.append(row)
            label = bytes(desc.label).split(b"\0", 1)[0].decode("utf-8")
            modules.append(ModuleDefinition(
                stable_id=stable_id,
                label=label,
                x=desc.x,
                y=desc.y,
                width=desc.w,
                height=desc.h,
                rows=tuple(rows),
                contract=contract,
            ))
            module_ids[module_index] = stable_id
            input_counts[module_index] = input_count
            output_counts[module_index] = output_count

        def reference(module_index: int, contact_index: int, direction: str) -> ConnectorRef:
            if module_index not in module_ids:
                raise RuntimeError(f"native endpoint names missing module {module_index}")
            if direction == "input":
                ordinal = contact_index
                limit = input_counts[module_index]
            else:
                ordinal = contact_index - input_counts[module_index]
                limit = output_counts[module_index]
            if ordinal < 0 or ordinal >= limit:
                raise RuntimeError(
                    f"native {direction} endpoint {module_index}.{contact_index} "
                    "is outside materialized row arity"
                )
            return ConnectorRef(module_ids[module_index], direction, ordinal)

        edge_count = int(self.lib.nodus_headless_get_edge_count(self._handle))
        if edge_count < 0:
            raise RuntimeError("native edge count query failed")
        edges: list[EdgeDefinition] = []
        for edge_index in range(edge_count):
            desc = CanvasEdgeDesc()
            type_id = ctypes.c_int32()
            if not self.lib.nodus_headless_get_edge(
                self._handle, edge_index, ctypes.byref(desc), ctypes.byref(type_id),
            ):
                raise RuntimeError(f"native edge {edge_index} query failed")
            fifo_value = _NativeFifoConfig()
            fifo = (
                _fifo_from_native(fifo_value)
                if self.lib.nodus_headless_get_edge_fifo_config(
                    self._handle, edge_index, ctypes.byref(fifo_value),
                )
                else None
            )
            edges.append(EdgeDefinition(
                source=reference(desc.a_module, desc.a_contact_idx, "output"),
                target=reference(desc.b_module, desc.b_contact_idx, "input"),
                fifo=fifo,
                type_id=type_id.value,
            ))

        def ports(is_output: bool) -> tuple[PortDefinition, ...]:
            count = int(self.lib.nodus_headless_get_port_count(
                self._handle, int(is_output),
            ))
            if count < 0:
                raise RuntimeError("native port count query failed")
            result: list[PortDefinition] = []
            direction = "output" if is_output else "input"
            for ordinal in range(count):
                name = ctypes.create_string_buffer(1024)
                module_index = ctypes.c_int32()
                contact_index = ctypes.c_int32()
                start_at_head = ctypes.c_int32()
                fifo_value = _NativeFifoConfig()
                if not self.lib.nodus_headless_get_port_at(
                    self._handle, int(is_output), ordinal,
                    ctypes.cast(name, ctypes.c_char_p), len(name),
                    ctypes.byref(module_index), ctypes.byref(contact_index),
                    ctypes.byref(start_at_head), ctypes.byref(fifo_value),
                ):
                    raise RuntimeError(f"native {direction} port {ordinal} query failed")
                result.append(PortDefinition(
                    name.value.decode("utf-8"),
                    reference(module_index.value, contact_index.value, direction),
                    _fifo_from_native(fifo_value),
                    bool(start_at_head.value),
                ))
            return tuple(result)

        valid_connectors = {
            ConnectorRef(stable_id, "input", ordinal)
            for index, stable_id in module_ids.items()
            for ordinal in range(input_counts[index])
        } | {
            ConnectorRef(stable_id, "output", ordinal)
            for index, stable_id in module_ids.items()
            for ordinal in range(output_counts[index])
        }
        unknown = set(self._connectors) - valid_connectors
        if unknown:
            raise RuntimeError(
                f"logical connector metadata names non-native contacts: {sorted(unknown, key=repr)!r}"
            )
        return GraphDefinition(
            scheduler=scheduler,
            modules=tuple(modules),
            edges=tuple(edges),
            inputs=ports(False),
            outputs=ports(True),
            connectors=tuple(
                self._connectors[key]
                for key in sorted(
                    self._connectors,
                    key=lambda ref: (ref.module_id, ref.direction, ref.ordinal),
                )
            ),
        )

    def apply_definition(self, definition: GraphDefinition) -> None:
        """Install a validated graph definition into an empty native graph."""
        if self.modules or self.edges or self._input_ports or self._output_ports:
            raise UnsupportedNativeContract("graph definitions can only be applied to an empty graph")
        definition.validate()
        if definition.scheduler != self.scheduler:
            if not self.lib.nodus_headless_set_scheduler_mode(self._handle, int(definition.scheduler)):
                raise RuntimeError("native scheduler mode configuration failed")
            self.scheduler = definition.scheduler

        modules: dict[int, Module] = {}
        for item in definition.modules:
            module = self.add_module(
                item.label, x=item.x, y=item.y, width=item.width,
                height=item.height, stable_id=item.stable_id,
            )
            module.set_rows(*item.rows)
            # Row materialization may grow an interactive module to fit its
            # preview. A communicated graph definition is explicit, so its
            # rectangle wins after the concrete rows have been built.
            desc = CanvasModuleDesc()
            desc.x, desc.y, desc.w, desc.h = item.x, item.y, item.width, item.height
            encoded = item.label.encode("utf-8")[:63]
            desc.label = encoded + b"\0" * (64 - len(encoded))
            if not self.lib.nodus_headless_set_module_desc(
                self._handle, module.index, ctypes.byref(desc),
            ):
                raise RuntimeError(
                    f"native module {item.stable_id} descriptor restoration failed"
                )
            if module.contract != item.contract:
                raise UnsupportedNativeContract(
                    f"module {item.stable_id} row contract {module.contract.value} "
                    f"does not match declared {item.contract.value}"
                )
            modules[item.stable_id] = module

        def contact(ref: ConnectorRef) -> Contact:
            module = modules[ref.module_id]
            return module.input(ref.ordinal) if ref.direction == "input" else module.output(ref.ordinal)

        for item in definition.edges:
            self.connect(contact(item.source), contact(item.target), fifo=item.fifo, type_id=item.type_id)
        for item in definition.inputs:
            self.bind_input(item.name, contact(item.connector), fifo=item.fifo)
        for item in definition.outputs:
            self.bind_output(
                item.name, contact(item.connector), fifo=item.fifo,
                start_at_head=item.start_at_head,
            )
        self._connectors = {
            item.reference: item for item in definition.connectors
        }

    @property
    def abstract_tensor_type_id(self) -> int:
        return int(self.lib.nodus_headless_abstract_tensor_type_id())

    def deploy_python_row(
        self, descriptor_path: Union[str, pathlib.Path], *, target: str,
        nodus_root: Union[str, pathlib.Path],
        module_library_root: Union[str, pathlib.Path],
        python_executable: Union[str, pathlib.Path, None] = None,
        turing_root: Union[str, pathlib.Path, None] = None,
        extraction_contract: Union[str, pathlib.Path, None] = None,
        product_dir: Union[str, pathlib.Path, None] = None,
    ) -> tuple[str, dict[str, object]]:
        """Deploy through the compiler/plugin registry used by this graph."""

        def encoded(value: Union[str, pathlib.Path, None]) -> bytes | None:
            return None if value is None else str(value).encode()

        report = _DeploymentReport()
        tool_id = ctypes.create_string_buffer(1024)
        retained = ctypes.create_string_buffer(4096)
        wrapper = ctypes.create_string_buffer(4096)
        ok = self.lib.nodus_headless_deploy_python_row(
            encoded(descriptor_path), target.encode(), encoded(python_executable),
            encoded(turing_root), encoded(extraction_contract), encoded(product_dir),
            encoded(nodus_root), encoded(module_library_root), ctypes.byref(report),
            tool_id, len(tool_id), retained, len(retained), wrapper, len(wrapper),
        )
        details: dict[str, object] = {
            name: int(getattr(report, name))
            for name, _ctype in report._fields_
        }
        details.update({
            "loaded_tool_id": tool_id.value.decode(),
            "retained_python": retained.value.decode(),
            "staged_wrapper": wrapper.value.decode(),
        })
        if not ok:
            raise RuntimeError(f"Python row deployment failed: {details}")
        return tool_id.value.decode(), details

    def create_tensor(self, dtype: int, shape: tuple[int, ...], data: bytes = b"") -> int:
        native_shape = (ctypes.c_uint64 * len(shape))(*shape) if shape else None
        handle = int(self.lib.nodus_headless_tensor_create(dtype, len(shape), native_shape))
        if not handle:
            raise RuntimeError("native tensor creation failed")
        if data:
            payload = ctypes.create_string_buffer(data, len(data))
            written = int(self.lib.nodus_headless_tensor_write(handle, 0, payload, len(data)))
            if written != len(data):
                self.lib.nodus_headless_tensor_destroy(handle)
                raise RuntimeError(f"native tensor write was short: {written} != {len(data)}")
        return handle

    def describe_tensor(self, handle: int) -> dict[str, object]:
        desc = _TensorDesc()
        if self.lib.nodus_headless_tensor_describe(handle, ctypes.byref(desc)) != 0:
            raise KeyError(f"unknown tensor handle {handle}")
        return {
            "dtype": int(desc.dtype),
            "layout": int(desc.layout),
            "readonly": bool(desc.is_readonly),
            "buffer": bool(desc.is_buffer),
            "shape": tuple(int(desc.shape[i]) for i in range(desc.rank)),
            "strides": tuple(int(desc.strides[i]) for i in range(desc.rank)),
            "element_count": int(desc.element_count),
            "element_bytes": int(desc.element_bytes),
            "total_bytes": int(desc.total_bytes),
        }

    def read_tensor(self, handle: int) -> bytes:
        desc = self.describe_tensor(handle)
        size = int(desc["total_bytes"])
        payload = ctypes.create_string_buffer(size)
        read = int(self.lib.nodus_headless_tensor_read(handle, 0, payload, size))
        if read != size:
            raise RuntimeError(f"native tensor read was short: {read} != {size}")
        return payload.raw

    def destroy_tensor(self, handle: int) -> None:
        self.lib.nodus_headless_tensor_destroy(handle)

    def unload_plugin(self, tool_id: str) -> None:
        if not self.lib.nodus_headless_unload_plugin(tool_id.encode()):
            raise RuntimeError(f"native plugin unload failed: {tool_id}")

    def build_plugin_source(
        self, source_path: Union[str, pathlib.Path], *,
        nodus_root: Union[str, pathlib.Path],
        module_library_root: Union[str, pathlib.Path],
    ) -> str:
        """Build/load a generated row through Nodus's ordinary registry."""

        tool_id = ctypes.create_string_buffer(1024)
        ok = self.lib.nodus_headless_build_plugin_source(
            str(source_path).encode(), str(nodus_root).encode(),
            str(module_library_root).encode(), tool_id, len(tool_id),
        )
        if not ok:
            raise RuntimeError(f"native plugin build/load failed: {source_path}")
        return tool_id.value.decode()

    def compile_turing_row(
        self, request_json: Union[str, pathlib.Path], *,
        python_executable: Union[str, pathlib.Path],
        turing_root: Union[str, pathlib.Path],
        product_dir: Union[str, pathlib.Path],
        nodus_root: Union[str, pathlib.Path],
        module_library_root: Union[str, pathlib.Path],
    ) -> tuple[str, dict[str, object]]:
        report = _TuringAdoptionReport()
        tool_id = ctypes.create_string_buffer(1024)
        staged = ctypes.create_string_buffer(4096)
        ok = self.lib.nodus_headless_compile_turing_row(
            str(python_executable).encode(), str(request_json).encode(),
            str(turing_root).encode(), str(product_dir).encode(),
            str(nodus_root).encode(), str(module_library_root).encode(),
            ctypes.byref(report), tool_id, len(tool_id), staged, len(staged),
        )
        details: dict[str, object] = {
            name: int(getattr(report, name))
            for name, _ctype in report._fields_
        }
        details.update({
            "loaded_tool_id": tool_id.value.decode(),
            "staged_source": staged.value.decode(),
        })
        if not ok:
            raise RuntimeError(f"Turing row compilation/adoption failed: {details}")
        return tool_id.value.decode(), details

    def edge_fifo_state(self, edge: Edge) -> FifoState:
        if edge.graph is not self or edge not in self.edges:
            raise ValueError("edge does not belong to this graph")
        native = _NativeFifoConfig()
        if not self.lib.nodus_headless_get_edge_fifo_config(
            self._handle, edge.native_index, ctypes.byref(native),
        ):
            return FifoState(config=None, unread=None)
        unread = ctypes.c_int32()
        if not self.lib.nodus_headless_get_edge_unread(
            self._handle, edge.native_index, ctypes.byref(unread),
        ):
            raise RuntimeError("native edge unread inspection failed")
        return FifoState(config=_fifo_from_native(native), unread=unread.value)

    def port_fifo_state(self, name: str, *, output: bool = False) -> FifoState:
        native = _NativeFifoConfig()
        encoded = name.encode()
        if not self.lib.nodus_headless_get_port_fifo_config(
            self._handle, encoded, int(output), ctypes.byref(native),
        ):
            raise KeyError(f"no native {'output' if output else 'input'} port {name!r}")
        unread = ctypes.c_int32()
        if not self.lib.nodus_headless_get_port_unread(
            self._handle, encoded, int(output), ctypes.byref(unread),
        ):
            raise RuntimeError(f"native port unread inspection failed: {name}")
        return FifoState(config=_fifo_from_native(native), unread=unread.value)

    def _ensure_scalar_execution(self) -> None:
        if self._loaded_unattached_edges:
            pending = sorted(self._loaded_unattached_edges)
            raise UnsupportedNativeContract(
                "native canvas persistence does not retain FIFO specs or live queue state; "
                f"reattach loaded edges {pending} before execution"
            )
        missing = [e.native_index for e in self.edges if e.fifo is None]
        if missing:
            raise UnsupportedNativeContract(
                f"native execution requires an explicit FIFO config on edges {missing}"
            )

    def try_push_bytes(self, name: str, data: bytes) -> PublishResult:
        self._ensure_scalar_execution()
        buf = ctypes.create_string_buffer(data, len(data))
        result = _NativePublishResult()
        if not self.lib.nodus_headless_push_token_ex(
            self._handle, name.encode(), buf, len(data), ctypes.byref(result),
        ):
            raise RuntimeError(f"native push call invalid: {name}")
        return PublishResult(bool(result.accepted), bool(result.dropped))

    def push_bytes(self, name: str, data: bytes) -> PublishResult:
        result = self.try_push_bytes(name, data)
        if not result.accepted:
            raise RuntimeError(f"native FIFO rejected push: {name}")
        return result

    def try_push_f32(self, name: str, value: float) -> PublishResult:
        return self.try_push_bytes(name, struct.pack("f", value))

    def push_f32(self, name: str, value: float) -> None:
        self.push_bytes(name, struct.pack("f", value))

    def pull_bytes(self, name: str, capacity: int) -> Optional[bytes]:
        self._ensure_scalar_execution()
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        buf = ctypes.create_string_buffer(capacity)
        written = ctypes.c_size_t()
        ok = self.lib.nodus_headless_pull_token(
            self._handle, name.encode(), buf, capacity, ctypes.byref(written),
        )
        if not ok:
            return None
        return buf.raw[:written.value]

    def pull_f32(self, name: str) -> Optional[float]:
        data = self.pull_bytes(name, 4)
        if data is None:
            return None
        if len(data) != 4:
            raise RuntimeError(f"native output {name!r} wrote {len(data)} bytes, expected 4")
        return struct.unpack("f", data)[0]

    def _stack_snapshot(self, module_idx: int, row_idx: int) -> StackSnapshot:
        value_count = ctypes.c_int32()
        if not self.lib.nodus_headless_get_stack_snapshot(
            self._handle, module_idx, row_idx, None, 0, ctypes.byref(value_count),
        ):
            raise RuntimeError("native stack snapshot size query failed")
        values_buf = (ctypes.c_float * value_count.value)()
        if value_count.value and not self.lib.nodus_headless_get_stack_snapshot(
            self._handle, module_idx, row_idx, values_buf, value_count.value,
            ctypes.byref(value_count),
        ):
            raise RuntimeError("native stack snapshot read failed")

        item_count = ctypes.c_int32()
        if not self.lib.nodus_headless_get_stack_snapshot_meta(
            self._handle, module_idx, row_idx, None, None, 0,
            ctypes.byref(item_count),
        ):
            raise RuntimeError("native stack metadata size query failed")
        type_ids = (ctypes.c_int32 * item_count.value)()
        element_bytes = (ctypes.c_int32 * item_count.value)()
        if item_count.value and not self.lib.nodus_headless_get_stack_snapshot_meta(
            self._handle, module_idx, row_idx, type_ids, element_bytes,
            item_count.value, ctypes.byref(item_count),
        ):
            raise RuntimeError("native stack metadata read failed")
        return StackSnapshot(
            values=tuple(float(values_buf[i]) for i in range(value_count.value)),
            items=tuple(
                StackItem(int(type_ids[i]), int(element_bytes[i]))
                for i in range(item_count.value)
            ),
        )

    def _stack_tail(self, module_idx: int) -> tuple[float, ...]:
        count = ctypes.c_int32()
        if not self.lib.nodus_headless_get_stack_tail(
            self._handle, module_idx, None, 0, ctypes.byref(count),
        ):
            raise RuntimeError("native stack tail size query failed")
        values = (ctypes.c_float * count.value)()
        if count.value and not self.lib.nodus_headless_get_stack_tail(
            self._handle, module_idx, values, count.value, ctypes.byref(count),
        ):
            raise RuntimeError("native stack tail read failed")
        return tuple(float(values[i]) for i in range(count.value))

    def step(self, dt: float = 1.0 / 60.0) -> None:
        self._ensure_scalar_execution()
        if not self.lib.nodus_headless_step(self._handle, dt):
            raise RuntimeError("native graph step failed")

    def save(self, path: Union[str, pathlib.Path]) -> None:
        if not self.lib.nodus_headless_save_file(self._handle, str(path).encode()):
            raise RuntimeError(f"native save failed: {path}")

    def load(self, path: Union[str, pathlib.Path]) -> None:
        if not self.lib.nodus_headless_load_file(self._handle, str(path).encode()):
            raise RuntimeError(f"native load failed: {path}")
        self._generation += 1
        self.modules.clear()
        self.edges.clear()
        self._input_ports.clear()
        self._output_ports.clear()
        self._connectors.clear()
        edge_count = int(self.lib.nodus_headless_get_edge_count(self._handle))
        if edge_count < 0:
            raise RuntimeError("native edge enumeration failed after load")
        self._loaded_unattached_edges = set(range(edge_count))
