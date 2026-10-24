from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Mapping


class SchedulerMode(IntEnum):
    FREE_SPINNING = 0
    SCHEDULED = 1


class ExecutionContract(Enum):
    """Value currency carried by a module's row stack."""

    SCALAR_F32 = "scalar_f32"
    ABSTRACT_TENSOR_POINTER = "abstract_tensor_pointer"
    OPAQUE = "opaque"


class BuiltinTool(IntEnum):
    NONE = 0
    ADD = 1
    SUBTRACT = 2
    MULTIPLY = 3
    DIVIDE = 4
    MODULO = 5
    KEYBOARD_LISTENER = 6
    MOUSE_LISTENER = 7
    STACK_DISPLAY = 8
    RECT_RGBA = 9
    TABLE_NUMBER = 10
    CLONE = 11
    TENSOR_ALLOCATOR = 12
    TENSOR_TOOL = 13


_SCALAR_STACK_EFFECT = {
    BuiltinTool.ADD: (2, 1),
    BuiltinTool.SUBTRACT: (2, 1),
    BuiltinTool.MULTIPLY: (2, 1),
    BuiltinTool.DIVIDE: (2, 1),
    BuiltinTool.MODULO: (2, 1),
    BuiltinTool.STACK_DISPLAY: (0, 0),
    BuiltinTool.TABLE_NUMBER: (0, 1),
    BuiltinTool.CLONE: (2, 0),
}


@dataclass(frozen=True)
class FifoConfig:
    """The existing EdgeTensorFifo configuration, without reinterpretation.

    ``top_k == 0`` rejects writes while unread data fills the bounded FIFO.
    A positive value advances old readers to retain the newest samples.
    ``None`` at an edge means no FIFO configuration is applied; zero capacity
    is deliberately not used as a synonym for absence.
    """

    token_bytes: int = 4
    slots: int = 4
    top_k: int = 1
    type_id: int = -1
    layout: int = 2  # TensorLayout::Opaque
    dtype: int = 0  # TensorDType::Unknown
    storage: str = "host"

    def __post_init__(self) -> None:
        if self.token_bytes <= 0:
            raise ValueError("token_bytes must be positive")
        if self.slots <= 0:
            raise ValueError("slots must be positive; use None for no FIFO configuration")
        if not 0 <= self.top_k <= self.slots:
            raise ValueError("top_k must be between zero and slots")
        if self.storage not in ("host", "vulkan"):
            raise ValueError("FIFO storage must be 'host' or 'vulkan'")

    @classmethod
    def scalar_f32(cls, *, slots: int = 4, top_k: int = 1) -> "FifoConfig":
        return cls(token_bytes=4, slots=slots, top_k=top_k)


@dataclass(frozen=True)
class PublishResult:
    accepted: bool
    dropped: bool


@dataclass(frozen=True)
class FifoState:
    config: FifoConfig | None
    unread: int | None


@dataclass(frozen=True)
class StackItem:
    type_id: int
    element_bytes: int


@dataclass(frozen=True)
class StackSnapshot:
    values: tuple[float, ...]
    items: tuple[StackItem, ...]


@dataclass(frozen=True)
class Input:
    count: int = 1


@dataclass(frozen=True)
class Output:
    count: int = 1


@dataclass(frozen=True)
class Tool:
    kind: BuiltinTool

    @property
    def contract(self) -> ExecutionContract:
        return ExecutionContract.SCALAR_F32

    @property
    def stack_effect(self) -> tuple[int, int]:
        try:
            return _SCALAR_STACK_EFFECT[self.kind]
        except KeyError as exc:
            raise ValueError(f"{self.kind.name} has no ordinary scalar-stack contract") from exc


@dataclass(frozen=True)
class PluginTool:
    plugin_id: str
    consumes: int
    produces: int = 1
    contract: ExecutionContract = ExecutionContract.OPAQUE

    def __post_init__(self) -> None:
        if not self.plugin_id:
            raise ValueError("plugin_id must not be empty")
        if self.consumes < 0 or self.produces < 0:
            raise ValueError("stack effects must be non-negative")


RowDefinition = Input | Output | Tool | PluginTool


@dataclass(frozen=True)
class ConnectorRef:
    """Stable reference to one native module contact.

    Native canvas indices are load-local.  A graph definition therefore names
    a connector by the module's persistent identity, direction, and ordinal.
    """

    module_id: int
    direction: str
    ordinal: int

    def __post_init__(self) -> None:
        if self.module_id <= 0:
            raise ValueError("connector module_id must be positive")
        if self.direction not in ("input", "output"):
            raise ValueError("connector direction must be 'input' or 'output'")
        if self.ordinal < 0:
            raise ValueError("connector ordinal must be non-negative")


@dataclass(frozen=True)
class ShapeSource:
    connector: ConnectorRef
    axis: int

    def __post_init__(self) -> None:
        if self.axis < 0:
            raise ValueError("shape source axis must be non-negative")


@dataclass(frozen=True)
class TensorContract:
    """Logical tensor value carried by one connector.

    FIFO configuration describes transport storage. This contract preserves
    the Turing ABI's logical dtype and shape, including runtime extents whose
    authority is another connector.
    """

    dtype: str
    shape: tuple[int | None, ...] = ()
    shape_sources: tuple[ShapeSource | None, ...] = ()

    def __post_init__(self) -> None:
        if not self.dtype:
            raise ValueError("tensor dtype must not be empty")
        if not isinstance(self.shape, tuple):
            object.__setattr__(self, "shape", tuple(self.shape))
        if not isinstance(self.shape_sources, tuple):
            object.__setattr__(self, "shape_sources", tuple(self.shape_sources))
        if self.shape_sources and len(self.shape_sources) != len(self.shape):
            raise ValueError("shape_sources must be empty or match tensor rank")
        if any(extent is not None and extent <= 0 for extent in self.shape):
            raise ValueError("fixed tensor extents must be positive")
        if not self.shape_sources:
            object.__setattr__(self, "shape_sources", (None,) * len(self.shape))
        for extent, source in zip(self.shape, self.shape_sources):
            if extent is None and source is None:
                raise ValueError("a dynamic tensor extent requires a shape source")
            if extent is not None and source is not None:
                raise ValueError("a fixed tensor extent cannot also have a shape source")


@dataclass(frozen=True)
class ConnectorDefinition:
    reference: ConnectorRef
    name: str
    tensor: TensorContract

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("connector name must not be empty")


@dataclass(frozen=True)
class ModuleDefinition:
    stable_id: int
    label: str
    rows: tuple[RowDefinition, ...]
    contract: ExecutionContract
    x: int = 0
    y: int = 0
    width: int = 420
    height: int = 320

    def __post_init__(self) -> None:
        if self.stable_id <= 0:
            raise ValueError("module stable_id must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("module width and height must be positive")
        if not isinstance(self.rows, tuple):
            object.__setattr__(self, "rows", tuple(self.rows))


@dataclass(frozen=True)
class EdgeDefinition:
    source: ConnectorRef
    target: ConnectorRef
    fifo: FifoConfig | None = FifoConfig.scalar_f32()
    type_id: int = 0

    def __post_init__(self) -> None:
        if self.source.direction != "output" or self.target.direction != "input":
            raise ValueError("edge endpoints must be output -> input")


@dataclass(frozen=True)
class PortDefinition:
    name: str
    connector: ConnectorRef
    fifo: FifoConfig = FifoConfig.scalar_f32()
    start_at_head: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("port name must not be empty")


def _row_to_mapping(row: RowDefinition) -> dict[str, Any]:
    if isinstance(row, Input):
        return {"kind": "input", "count": row.count}
    if isinstance(row, Output):
        return {"kind": "output", "count": row.count}
    if isinstance(row, Tool):
        return {"kind": "builtin", "tool": row.kind.name}
    if isinstance(row, PluginTool):
        return {
            "kind": "plugin",
            "plugin_id": row.plugin_id,
            "consumes": row.consumes,
            "produces": row.produces,
            "contract": row.contract.value,
        }
    raise TypeError(f"unsupported row definition: {row!r}")


def _row_from_mapping(value: Mapping[str, Any]) -> RowDefinition:
    kind = value.get("kind")
    if kind == "input":
        return Input(int(value["count"]))
    if kind == "output":
        return Output(int(value["count"]))
    if kind == "builtin":
        return Tool(BuiltinTool[str(value["tool"])])
    if kind == "plugin":
        return PluginTool(
            str(value["plugin_id"]),
            int(value["consumes"]),
            int(value.get("produces", 1)),
            ExecutionContract(str(value.get("contract", ExecutionContract.OPAQUE.value))),
        )
    raise ValueError(f"unknown row kind: {kind!r}")


def _connector_to_mapping(value: ConnectorRef) -> dict[str, Any]:
    return {
        "module_id": value.module_id,
        "direction": value.direction,
        "ordinal": value.ordinal,
    }


def _connector_from_mapping(value: Mapping[str, Any]) -> ConnectorRef:
    return ConnectorRef(
        module_id=int(value["module_id"]),
        direction=str(value["direction"]),
        ordinal=int(value["ordinal"]),
    )


def _fifo_to_mapping(value: FifoConfig | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "token_bytes": value.token_bytes,
        "slots": value.slots,
        "top_k": value.top_k,
        "type_id": value.type_id,
        "layout": value.layout,
        "dtype": value.dtype,
        "storage": value.storage,
    }


def _fifo_from_mapping(value: Mapping[str, Any] | None) -> FifoConfig | None:
    if value is None:
        return None
    return FifoConfig(
        token_bytes=int(value["token_bytes"]),
        slots=int(value["slots"]),
        top_k=int(value["top_k"]),
        type_id=int(value.get("type_id", -1)),
        layout=int(value.get("layout", 2)),
        dtype=int(value.get("dtype", 0)),
        storage=str(value.get("storage", "host")),
    )


@dataclass(frozen=True)
class GraphDefinition:
    """Immutable, versioned description of the executable headless graph.

    This is a transport description of native Canvas/Table configuration, not
    another scheduler or graph runtime.  Turing can construct or inspect this
    value from Python while Nodus remains authoritative for execution.
    """

    modules: tuple[ModuleDefinition, ...]
    edges: tuple[EdgeDefinition, ...] = ()
    inputs: tuple[PortDefinition, ...] = ()
    outputs: tuple[PortDefinition, ...] = ()
    connectors: tuple[ConnectorDefinition, ...] = ()
    scheduler: SchedulerMode = SchedulerMode.SCHEDULED
    schema: str = "nodus.graph-definition.v1"

    def __post_init__(self) -> None:
        for field in ("modules", "edges", "inputs", "outputs", "connectors"):
            value = getattr(self, field)
            if not isinstance(value, tuple):
                object.__setattr__(self, field, tuple(value))
        self.validate()

    def validate(self) -> None:
        if self.schema != "nodus.graph-definition.v1":
            raise ValueError(f"unsupported graph definition schema: {self.schema!r}")
        modules = {module.stable_id: module for module in self.modules}
        if len(modules) != len(self.modules):
            raise ValueError("module stable identities must be unique")

        arity: dict[int, tuple[int, int]] = {}
        for module in self.modules:
            inputs = 0
            outputs = 0
            stack_depth = 0
            contract: ExecutionContract | None = None
            for index, row in enumerate(module.rows):
                if isinstance(row, Input):
                    if not 1 <= row.count <= 32:
                        raise ValueError(f"module {module.stable_id} input row {index} count must be in 1..32")
                    inputs += row.count
                    stack_depth += row.count
                elif isinstance(row, Output):
                    if not 1 <= row.count <= 32:
                        raise ValueError(f"module {module.stable_id} output row {index} count must be in 1..32")
                    if stack_depth < row.count:
                        raise ValueError(f"module {module.stable_id} output row {index} underflows the stack")
                    outputs += row.count
                    stack_depth -= row.count
                elif isinstance(row, Tool):
                    consumes, produces = row.stack_effect
                    if stack_depth < consumes:
                        raise ValueError(f"module {module.stable_id} tool row {index} underflows the stack")
                    stack_depth += produces - consumes
                    if contract not in (None, row.contract):
                        raise ValueError(f"module {module.stable_id} mixes execution contracts")
                    contract = row.contract
                elif isinstance(row, PluginTool):
                    if stack_depth < row.consumes:
                        raise ValueError(f"module {module.stable_id} plugin row {index} underflows the stack")
                    stack_depth += row.produces - row.consumes
                    if contract not in (None, row.contract):
                        raise ValueError(f"module {module.stable_id} mixes execution contracts")
                    contract = row.contract
                else:
                    raise TypeError(f"module {module.stable_id} has unsupported row {row!r}")
            if stack_depth:
                raise ValueError(f"module {module.stable_id} leaves {stack_depth} value(s) on the stack")
            actual_contract = contract or ExecutionContract.OPAQUE
            if module.contract != actual_contract:
                raise ValueError(
                    f"module {module.stable_id} row contract {actual_contract.value} "
                    f"does not match declared {module.contract.value}"
                )
            arity[module.stable_id] = (inputs, outputs)

        def check_connector(connector: ConnectorRef) -> None:
            if connector.module_id not in arity:
                raise ValueError(f"connector references unknown module {connector.module_id}")
            input_count, output_count = arity[connector.module_id]
            limit = input_count if connector.direction == "input" else output_count
            if connector.ordinal >= limit:
                raise ValueError(
                    f"connector {connector.module_id}:{connector.direction}:{connector.ordinal} "
                    f"is outside arity {limit}"
                )

        for edge in self.edges:
            check_connector(edge.source)
            check_connector(edge.target)

        connector_map: dict[ConnectorRef, ConnectorDefinition] = {}
        connector_names: set[tuple[int, str, str]] = set()
        for connector in self.connectors:
            check_connector(connector.reference)
            if connector.reference in connector_map:
                raise ValueError(f"duplicate connector definition: {connector.reference!r}")
            name_key = (
                connector.reference.module_id,
                connector.reference.direction,
                connector.name,
            )
            if name_key in connector_names:
                raise ValueError(f"duplicate connector name: {name_key!r}")
            connector_map[connector.reference] = connector
            connector_names.add(name_key)
        if self.connectors:
            expected = {
                ConnectorRef(module_id, direction, ordinal)
                for module_id, (input_count, output_count) in arity.items()
                for direction, count in (("input", input_count), ("output", output_count))
                for ordinal in range(count)
            }
            missing = expected - set(connector_map)
            extra = set(connector_map) - expected
            if missing or extra:
                raise ValueError(
                    f"connector definitions must cover module arity; missing={sorted(missing, key=repr)!r}, "
                    f"extra={sorted(extra, key=repr)!r}"
                )
            for connector in self.connectors:
                for source in connector.tensor.shape_sources:
                    if source is None:
                        continue
                    check_connector(source.connector)
                    source_definition = connector_map[source.connector]
                    if source.axis >= len(source_definition.tensor.shape):
                        raise ValueError(
                            f"shape source axis {source.axis} is outside connector "
                            f"{source.connector!r} rank"
                        )
        names: set[tuple[str, str]] = set()
        for direction, ports in (("input", self.inputs), ("output", self.outputs)):
            for port in ports:
                if port.connector.direction != direction:
                    raise ValueError(f"{direction} port {port.name!r} targets a {port.connector.direction} connector")
                check_connector(port.connector)
                key = (direction, port.name)
                if key in names:
                    raise ValueError(f"duplicate {direction} port name: {port.name!r}")
                names.add(key)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "scheduler": self.scheduler.name,
            "modules": [
                {
                    "stable_id": module.stable_id,
                    "label": module.label,
                    "x": module.x,
                    "y": module.y,
                    "width": module.width,
                    "height": module.height,
                    "contract": module.contract.value,
                    "rows": [_row_to_mapping(row) for row in module.rows],
                }
                for module in self.modules
            ],
            "edges": [
                {
                    "source": _connector_to_mapping(edge.source),
                    "target": _connector_to_mapping(edge.target),
                    "fifo": _fifo_to_mapping(edge.fifo),
                    "type_id": edge.type_id,
                }
                for edge in self.edges
            ],
            "connectors": [
                {
                    "reference": _connector_to_mapping(connector.reference),
                    "name": connector.name,
                    "tensor": {
                        "dtype": connector.tensor.dtype,
                        "shape": list(connector.tensor.shape),
                        "shape_sources": [
                            None if source is None else {
                                "connector": _connector_to_mapping(source.connector),
                                "axis": source.axis,
                            }
                            for source in connector.tensor.shape_sources
                        ],
                    },
                }
                for connector in self.connectors
            ],
            "inputs": [
                {
                    "name": port.name,
                    "connector": _connector_to_mapping(port.connector),
                    "fifo": _fifo_to_mapping(port.fifo),
                }
                for port in self.inputs
            ],
            "outputs": [
                {
                    "name": port.name,
                    "connector": _connector_to_mapping(port.connector),
                    "fifo": _fifo_to_mapping(port.fifo),
                    "start_at_head": port.start_at_head,
                }
                for port in self.outputs
            ],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GraphDefinition":
        def port(item: Mapping[str, Any], *, output: bool) -> PortDefinition:
            fifo = _fifo_from_mapping(item["fifo"])
            if fifo is None:
                raise ValueError("named ports require an explicit FIFO configuration")
            return PortDefinition(
                name=str(item["name"]),
                connector=_connector_from_mapping(item["connector"]),
                fifo=fifo,
                start_at_head=bool(item.get("start_at_head", True)) if output else True,
            )

        modules = tuple(
            ModuleDefinition(
                stable_id=int(item["stable_id"]),
                label=str(item["label"]),
                x=int(item.get("x", 0)),
                y=int(item.get("y", 0)),
                width=int(item.get("width", 420)),
                height=int(item.get("height", 320)),
                contract=ExecutionContract(str(item["contract"])),
                rows=tuple(_row_from_mapping(row) for row in item["rows"]),
            )
            for item in value.get("modules", ())
        )
        edges = tuple(
            EdgeDefinition(
                source=_connector_from_mapping(item["source"]),
                target=_connector_from_mapping(item["target"]),
                fifo=_fifo_from_mapping(item.get("fifo")),
                type_id=int(item.get("type_id", 0)),
            )
            for item in value.get("edges", ())
        )
        connectors = tuple(
            ConnectorDefinition(
                reference=_connector_from_mapping(item["reference"]),
                name=str(item["name"]),
                tensor=TensorContract(
                    dtype=str(item["tensor"]["dtype"]),
                    shape=tuple(item["tensor"].get("shape", ())),
                    shape_sources=tuple(
                        None if source is None else ShapeSource(
                            connector=_connector_from_mapping(source["connector"]),
                            axis=int(source["axis"]),
                        )
                        for source in item["tensor"].get("shape_sources", ())
                    ),
                ),
            )
            for item in value.get("connectors", ())
        )
        return cls(
            schema=str(value.get("schema", "")),
            scheduler=SchedulerMode[str(value.get("scheduler", "SCHEDULED"))],
            modules=modules,
            edges=edges,
            connectors=connectors,
            inputs=tuple(port(item, output=False) for item in value.get("inputs", ())),
            outputs=tuple(port(item, output=True) for item in value.get("outputs", ())),
        )
