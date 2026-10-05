from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


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

    def __post_init__(self) -> None:
        if self.token_bytes <= 0:
            raise ValueError("token_bytes must be positive")
        if self.slots <= 0:
            raise ValueError("slots must be positive; use None for no FIFO configuration")
        if not 0 <= self.top_k <= self.slots:
            raise ValueError("top_k must be between zero and slots")

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
