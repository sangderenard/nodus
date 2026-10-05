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
    ExecutionContract,
    FifoConfig,
    FifoState,
    Input,
    Output,
    PluginTool,
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


def _fifo(config: FifoConfig) -> _NativeFifoConfig:
    return _NativeFifoConfig(
        config.token_bytes,
        config.slots,
        config.top_k,
        config.type_id,
        config.layout,
        config.dtype,
    )


def _fifo_from_native(config: _NativeFifoConfig) -> FifoConfig:
    return FifoConfig(
        token_bytes=config.token_bytes,
        slots=config.slots,
        top_k=config.top_k,
        type_id=config.type_id,
        layout=config.layout,
        dtype=config.dtype,
    )


def _load(lib_path: Optional[str] = None):
    lib = load_headless_lib(lib_path)
    lib.nodus_headless_set_module_rows.restype = ctypes.c_int32
    lib.nodus_headless_set_module_rows.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(_NativeRow), ctypes.c_int32,
    )
    lib.nodus_headless_configure_edge_fifo.restype = ctypes.c_int32
    lib.nodus_headless_configure_edge_fifo.argtypes = (
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(_NativeFifoConfig),
    )
    lib.nodus_headless_get_edge_count.restype = ctypes.c_int32
    lib.nodus_headless_get_edge_count.argtypes = (ctypes.c_void_p,)
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


Row = Union[Input, Output, Tool, PluginTool]


class Module:
    def __init__(self, graph: "Graph", index: int, stable_id: int):
        self.graph = graph
        self.index = index
        self.stable_id = stable_id
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
        self.rows = tuple(rows)
        self.input_count = input_count
        self.output_count = output_count
        self.contract = contract or ExecutionContract.OPAQUE
        return self

    def input(self, ordinal: int) -> Contact:
        self._assert_live()
        if not 0 <= ordinal < self.input_count:
            raise IndexError("input contact ordinal out of range")
        return Contact(self, "input", ordinal)

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
        module = Module(self, index, identity)
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
        edge = Edge(self, index, source, target, fifo)
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
        edge = Edge(self, index, source, target, fifo)
        self.edges = [prior for prior in self.edges if prior.native_index != index]
        self.edges.append(edge)
        self._loaded_unattached_edges.discard(index)
        return edge

    def bind_input(self, name: str, contact: Contact, *, fifo: FifoConfig = FifoConfig.scalar_f32()) -> None:
        if contact.direction != "input":
            raise ValueError("input port requires an input contact")
        config = _fifo(fifo)
        if not self.lib.nodus_headless_declare_input_port_ex(
            self._handle, name.encode(), contact.module.index, contact.native_index, ctypes.byref(config),
        ):
            raise RuntimeError(f"native input port declaration failed: {name}")

    def bind_output(self, name: str, contact: Contact, *, fifo: FifoConfig = FifoConfig.scalar_f32(),
                    start_at_head: bool = True) -> None:
        if contact.direction != "output":
            raise ValueError("output port requires an output contact")
        config = _fifo(fifo)
        if not self.lib.nodus_headless_declare_output_port_ex(
            self._handle, name.encode(), contact.module.index, contact.native_index,
            ctypes.byref(config), int(start_at_head),
        ):
            raise RuntimeError(f"native output port declaration failed: {name}")

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
        pointer_modules = [m.index for m in self.modules if m.contract == ExecutionContract.ABSTRACT_TENSOR_POINTER]
        if pointer_modules:
            raise UnsupportedNativeContract(
                "ordinary canvas Input/Output rows carry float scalars, while abstract_tensor.* tools "
                f"consume owned AbstractTensor pointers; native mixed execution is refused for modules {pointer_modules}"
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
        edge_count = int(self.lib.nodus_headless_get_edge_count(self._handle))
        if edge_count < 0:
            raise RuntimeError("native edge enumeration failed after load")
        self._loaded_unattached_edges = set(range(edge_count))
