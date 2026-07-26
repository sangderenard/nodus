from __future__ import annotations

"""
ctypes bridge for the nodus_headless ABI (include/nodus_headless_abi.h).

Mirrors the loading/prototype pattern already used by demo_canvas_tables.py
for canvas_tables.dll -- same library-discovery convention, same
ctypes.Structure-per-C-struct approach. This is the Python side of nodus's
"Prong 1" headless graph runtime: build or load a tool graph, run it to
quiescence with no window/renderer, and exchange data at named external
ports, entirely from Python via ctypes.

Example:
    g = NodusHeadlessGraph()
    a = g.add_module(0, 0, 64, 64, "alloc")
    b = g.add_module(100, 0, 64, 64, "tool")
    row_a = g.bind_builtin_tool(a, TOOL_TENSOR_ALLOCATOR)
    row_b = g.bind_builtin_tool(b, TOOL_TENSOR_TOOL)
    g.add_edge(a, row_a, b, row_b)
    g.declare_input_port("in", a, row_a, token_bytes=16)
    g.declare_output_port("out", b, row_b, token_bytes=16)
    g.on("quiescent", lambda ev: print("quiescent at tick", ev.tick_id))
    g.push_token("in", struct.pack("4f", 1.0, 2.0, 3.0, 4.0))
    g.run_to_quiescence()
    g.poll()  # dispatches any queued events to registered callbacks
"""

import ctypes
import os
import pathlib
import platform
from typing import Callable, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parent

# ModuleToolKind values (include/thread_manager.h) that don't require any
# plugin DLL -- convenience constants for callers exercising the ABI.
TOOL_TENSOR_ALLOCATOR = 12
TOOL_TENSOR_TOOL = 13

_EVENT_NAMES = {
    0: "quiescent",
    1: "node_complete",
    2: "overflow",
    3: "rejected_txn",
}


class CanvasModuleDesc(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("w", ctypes.c_int32),
        ("h", ctypes.c_int32),
        ("label", ctypes.c_char * 64),
    ]


class CanvasEdgeDesc(ctypes.Structure):
    _fields_ = [
        ("a_module", ctypes.c_int32),
        ("a_contact_idx", ctypes.c_int32),
        ("b_module", ctypes.c_int32),
        ("b_contact_idx", ctypes.c_int32),
    ]


class NodusHeadlessEvent(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("module_idx", ctypes.c_int32),
        ("edge_idx", ctypes.c_int32),
        ("tick_id", ctypes.c_uint64),
        ("code", ctypes.c_int32),
    ]


def find_headless_library() -> pathlib.Path:
    """Locate the native nodus_headless library across common build layouts."""
    env_override = os.getenv("NODUS_HEADLESS_TEST_DLL") or os.getenv("NODUS_HEADLESS_LIB")
    if env_override:
        path = pathlib.Path(env_override)
        if path.exists():
            return path

    search_roots = [
        ROOT / "build" / "Release",
        ROOT / "build" / "Debug",
        ROOT / "build",
        ROOT,
    ]
    system = platform.system().lower()
    names: List[str]
    if system == "windows":
        names = ["nodus_headless.dll"]
    elif system == "darwin":
        names = ["libnodus_headless.dylib", "nodus_headless.dylib"]
    else:
        names = ["libnodus_headless.so", "nodus_headless.so"]

    for root in search_roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate

    raise OSError(
        "nodus_headless library not found. Build it (cmake --build build --target nodus_headless) "
        "or set NODUS_HEADLESS_LIB."
    )


def load_headless_lib(lib_path: Optional[str] = None):
    path = pathlib.Path(lib_path) if lib_path is not None else find_headless_library()
    lib = ctypes.CDLL(str(path))

    lib.nodus_headless_create.restype = ctypes.c_void_p
    lib.nodus_headless_create.argtypes = ()
    lib.nodus_headless_destroy.restype = None
    lib.nodus_headless_destroy.argtypes = (ctypes.c_void_p,)

    lib.nodus_headless_load_file.restype = ctypes.c_int32
    lib.nodus_headless_load_file.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    lib.nodus_headless_save_file.restype = ctypes.c_int32
    lib.nodus_headless_save_file.argtypes = (ctypes.c_void_p, ctypes.c_char_p)

    lib.nodus_headless_add_module.restype = ctypes.c_int32
    lib.nodus_headless_add_module.argtypes = (ctypes.c_void_p, ctypes.POINTER(CanvasModuleDesc))
    lib.nodus_headless_bind_builtin_tool.restype = ctypes.c_int32
    lib.nodus_headless_bind_builtin_tool.argtypes = (ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32)
    lib.nodus_headless_bind_plugin_tool.restype = ctypes.c_int32
    lib.nodus_headless_bind_plugin_tool.argtypes = (ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p)
    lib.nodus_headless_add_edge.restype = ctypes.c_int32
    lib.nodus_headless_add_edge.argtypes = (ctypes.c_void_p, ctypes.POINTER(CanvasEdgeDesc), ctypes.c_int32)

    lib.nodus_headless_declare_input_port.restype = ctypes.c_int32
    lib.nodus_headless_declare_input_port.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
    lib.nodus_headless_declare_output_port.restype = ctypes.c_int32
    lib.nodus_headless_declare_output_port.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
    lib.nodus_headless_push_token.restype = ctypes.c_int32
    lib.nodus_headless_push_token.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t)
    lib.nodus_headless_pull_token.restype = ctypes.c_int32
    lib.nodus_headless_pull_token.argtypes = (
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    )

    lib.nodus_headless_step.restype = ctypes.c_int32
    lib.nodus_headless_step.argtypes = (ctypes.c_void_p, ctypes.c_double)
    lib.nodus_headless_run_to_quiescence.restype = ctypes.c_int32
    lib.nodus_headless_run_to_quiescence.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_int32)
    lib.nodus_headless_is_quiescent.restype = ctypes.c_int32
    lib.nodus_headless_is_quiescent.argtypes = (ctypes.c_void_p,)

    lib.nodus_headless_poll_events.restype = ctypes.c_int32
    lib.nodus_headless_poll_events.argtypes = (ctypes.c_void_p, ctypes.POINTER(NodusHeadlessEvent), ctypes.c_int32)

    return lib


class NodusHeadlessGraph:
    """Owns a native headless graph handle and dispatches polled events to
    registered Python callbacks. Events are never delivered by the C runtime
    calling directly into Python from its own scheduler thread -- call
    .poll() (or rely on run_to_quiescence()'s implicit poll) to drain and
    dispatch them, keeping the GIL out of nodus's scheduling hot path.
    """

    def __init__(self, lib_path: Optional[str] = None):
        self.lib = load_headless_lib(lib_path)
        self._handle = self.lib.nodus_headless_create()
        if not self._handle:
            raise RuntimeError("nodus_headless_create failed")
        self._callbacks: Dict[str, List[Callable[[NodusHeadlessEvent], None]]] = {
            name: [] for name in _EVENT_NAMES.values()
        }

    def close(self):
        if self._handle:
            self.lib.nodus_headless_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def load_file(self, path: str) -> bool:
        return bool(self.lib.nodus_headless_load_file(self._handle, path.encode("utf-8")))

    def save_file(self, path: str) -> bool:
        return bool(self.lib.nodus_headless_save_file(self._handle, path.encode("utf-8")))

    def add_module(self, x: int, y: int, w: int, h: int, label: str) -> int:
        desc = CanvasModuleDesc()
        desc.x, desc.y, desc.w, desc.h = int(x), int(y), int(w), int(h)
        b = label.encode("utf-8")[:63]
        desc.label = b + b"\0" * (64 - len(b))
        return self.lib.nodus_headless_add_module(self._handle, ctypes.byref(desc))

    def bind_builtin_tool(self, module_idx: int, tool_kind: int) -> int:
        return self.lib.nodus_headless_bind_builtin_tool(self._handle, module_idx, tool_kind)

    def bind_plugin_tool(self, module_idx: int, plugin_id: str) -> int:
        return self.lib.nodus_headless_bind_plugin_tool(self._handle, module_idx, plugin_id.encode("utf-8"))

    def add_edge(self, a_module: int, a_row: int, b_module: int, b_row: int, type_id: int = 0) -> int:
        desc = CanvasEdgeDesc(a_module=a_module, a_contact_idx=a_row, b_module=b_module, b_contact_idx=b_row)
        return self.lib.nodus_headless_add_edge(self._handle, ctypes.byref(desc), type_id)

    def declare_input_port(self, name: str, module_idx: int, row_idx: int, token_bytes: int) -> bool:
        return bool(self.lib.nodus_headless_declare_input_port(
            self._handle, name.encode("utf-8"), module_idx, row_idx, token_bytes
        ))

    def declare_output_port(self, name: str, module_idx: int, row_idx: int, token_bytes: int) -> bool:
        return bool(self.lib.nodus_headless_declare_output_port(
            self._handle, name.encode("utf-8"), module_idx, row_idx, token_bytes
        ))

    def push_token(self, port_name: str, data: bytes) -> bool:
        buf = ctypes.create_string_buffer(data, len(data))
        return bool(self.lib.nodus_headless_push_token(self._handle, port_name.encode("utf-8"), buf, len(data)))

    def pull_token(self, port_name: str, capacity: int = 4096) -> Optional[bytes]:
        buf = ctypes.create_string_buffer(capacity)
        written = ctypes.c_size_t(0)
        ok = self.lib.nodus_headless_pull_token(
            self._handle, port_name.encode("utf-8"), buf, capacity, ctypes.byref(written)
        )
        if not ok:
            return None
        return buf.raw[: written.value]

    def step(self, dt: float = 1.0 / 60.0) -> bool:
        return bool(self.lib.nodus_headless_step(self._handle, dt))

    def run_to_quiescence(self, dt: float = 1.0 / 60.0, max_ticks: int = 1000) -> bool:
        reached = bool(self.lib.nodus_headless_run_to_quiescence(self._handle, dt, max_ticks))
        self.poll()
        return reached

    def is_quiescent(self) -> bool:
        return bool(self.lib.nodus_headless_is_quiescent(self._handle))

    def on(self, event_name: str, callback: Callable[[NodusHeadlessEvent], None]) -> None:
        if event_name not in self._callbacks:
            raise ValueError(f"unknown event name: {event_name!r}")
        self._callbacks[event_name].append(callback)

    def poll(self, capacity: int = 64) -> int:
        """Drain queued native events and dispatch them to registered callbacks.
        Returns the number of events processed."""
        buf = (NodusHeadlessEvent * capacity)()
        n = self.lib.nodus_headless_poll_events(self._handle, buf, capacity)
        for i in range(n):
            ev = buf[i]
            name = _EVENT_NAMES.get(ev.kind)
            if name:
                for cb in self._callbacks[name]:
                    cb(ev)
        return n
