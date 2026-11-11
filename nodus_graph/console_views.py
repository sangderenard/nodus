"""Read-only documents served by the canvas host console (protocol 2 verbs).

Every function reads a live ``Graph`` and must run on the thread that steps it
(tools/run_canvas.py calls them from ConsoleServer.service, exactly where SNAPSHOT
is built).  Nothing here writes to the graph.  GRAPH reuses graph-definition v1
(``GraphDefinition.to_mapping``) and only adds keys.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from typing import Any, Iterable

from nodus_headless_bridge import CanvasEdgeDesc, CanvasModuleDesc

from .model import BuiltinTool, Input, Output, _fifo_to_mapping
from .native import (Graph, _NativeFifoConfig, _NativeRowInfo, _contract_from_native,
                     _fifo_from_native)

PROTOCOL = 2


def dumps(document: Any) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _count(lib, name: str, handle, *args) -> int:
    return max(0, int(getattr(lib, name)(handle, *args)))


def counts(graph: Graph) -> dict:
    lib, handle = graph.lib, graph._handle
    modules = _count(lib, "nodus_headless_get_module_count", handle)
    rows = sum(_count(lib, "nodus_headless_get_module_row_count", handle, i)
               for i in range(modules))
    return {"modules": modules, "rows": rows,
            "edges": _count(lib, "nodus_headless_get_edge_count", handle),
            "contract_ports": len(graph.contract_ports())}


def digest_document(graph: Graph, *, canvas: str, revision: int, data: bytes, tick: int,
                    rate: float, started: float, compile_hosts: int) -> dict:
    document = {"protocol": PROTOCOL, "canvas": canvas, "revision": revision,
                "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                "compile_hosts": compile_hosts, "tick": tick, "rate": rate,
                "started": started}
    document.update(counts(graph))
    return document


def contract_port_documents(graph: Graph) -> list:
    lib = graph._contract_abi()
    result = []
    for prefix, capacity, edge, declared in graph.contract_ports():
        bound = graph.contract_port(prefix)
        loan = int(lib.nodus_headless_contract_pair_loan_ticks(prefix.encode("utf-8")))
        result.append({"prefix": prefix, "capacity": capacity, "edge": edge,
                       "listen_port": bound[0] if bound else declared,
                       "loan_ticks": loan or None})
    return result


def graph_document(graph: Graph, *, revision: int) -> dict:
    """graph-definition v1 plus additive keys: module/row/edge ``index``, module
    ``contacts`` and ``exec_mode`` (null: not exposed), edge ``token_bytes``/``slots``,
    and top-level ``revision`` and ``contract_ports``."""
    try:
        definition = graph.definition()
    except (RuntimeError, ValueError) as exc:
        # A canvas whose edges name contacts no Input/Output row declares (the
        # messenger canvas) cannot be a strict v1 definition; serve native indices.
        return _native_document(graph, revision, str(exc))
    document = definition.to_mapping()
    names = {(c.reference.module_id, c.reference.direction, c.reference.ordinal): c.name
             for c in definition.connectors}
    for index, (module, mapped) in enumerate(zip(definition.modules, document["modules"])):
        mapped["index"] = index
        mapped["exec_mode"] = None
        inputs = sum(r.count for r in module.rows if isinstance(r, Input))
        outputs = sum(r.count for r in module.rows if isinstance(r, Output))
        contacts = []
        # Native contact index: inputs first, then outputs.
        for ordinal in range(inputs):
            contacts.append({"index": ordinal, "ordinal": ordinal, "direction": "input",
                             "name": names.get((module.stable_id, "input", ordinal))})
        for ordinal in range(outputs):
            contacts.append({"index": inputs + ordinal, "ordinal": ordinal,
                             "direction": "output",
                             "name": names.get((module.stable_id, "output", ordinal))})
        mapped["contacts"] = contacts
        for row_index, row in enumerate(mapped["rows"]):
            row["index"] = row_index
    for index, mapped in enumerate(document["edges"]):
        mapped["index"] = index
        fifo = mapped.get("fifo") or {}
        mapped["token_bytes"] = fifo.get("token_bytes")
        mapped["slots"] = fifo.get("slots")
    document["revision"] = revision
    document["contract_ports"] = contract_port_documents(graph)
    return document


def tools_document(graph: Graph, graph_doc: dict | None = None) -> dict:
    """Tools reachable from Python: the BuiltinTool vocabulary, plugin ids bound on the
    canvas rows, and the acquired contract pairs.  The process-wide ToolRegistry has no
    enumeration entry point in the headless ABI, so the list is not complete."""
    tools: dict[str, dict] = {}
    for tool in BuiltinTool:
        if tool is not BuiltinTool.NONE:
            tools[tool.name] = {"name": tool.name, "plugin_id": None,
                                "source": "builtin", "description": None}

    def plugin(plugin_id: str) -> None:
        tools.setdefault(plugin_id, {"name": plugin_id, "plugin_id": plugin_id,
                                     "source": "plugin", "description": None})

    for prefix, *_rest in graph.contract_ports():
        plugin(f"{prefix}.market")
        plugin(f"{prefix}.worker")
    if graph_doc is not None:
        for module in graph_doc["modules"]:
            for row in module["rows"]:
                if row.get("kind") == "plugin":
                    plugin(row["plugin_id"])
    return {"tools": [tools[key] for key in sorted(tools)], "complete": False,
            "missing": ("registered plugin tools that are neither placed on this canvas nor a "
                        "contract pair, and the module library: the headless ABI has no tool "
                        "registry enumeration and the host has no library path (needs new C++)")}


def activity_document(graph: Graph, *, tick: int, compile_hosts: Iterable) -> dict:
    lib, handle = graph.lib, graph._handle
    edges = []
    for index in range(_count(lib, "nodus_headless_get_edge_count", handle)):
        unread = ctypes.c_int32()
        probe = _NativeFifoConfig()
        ok = (lib.nodus_headless_get_edge_fifo_config(handle, index, ctypes.byref(probe))
              and lib.nodus_headless_get_edge_unread(handle, index, ctypes.byref(unread)))
        # seq: no sequence-number query exists in the ABI.
        edges.append({"index": index, "fill": unread.value if ok else None, "seq": None})
    ports = []
    for prefix, _cap, _edge, _declared in graph.contract_ports():
        bound = graph.contract_port(prefix)
        # pending: no pending-contract query exists in the ABI.
        ports.append({"prefix": prefix, "pending": None,
                      "engines": bound[2] if bound else None,
                      "hosts": bound[1] if bound else None})
    hosts = []
    for prefix, host_id in compile_hosts:
        try:
            stats = graph.compile_host_stats(prefix, host_id)
        except RuntimeError:
            continue
        hosts.append({"prefix": prefix, "host_id": host_id, **stats})
    return {"tick": tick, "edges": edges, "contract_ports": ports, "compile_hosts": hosts}


def _native_document(graph: Graph, revision: int, reason: str) -> dict:
    """Tolerant read of the same native queries Graph.definition() uses.  Same module and
    row field names as graph-definition v1; edges name native indices because stable
    connector references do not exist for these contacts.  ``definition_error`` says why
    the strict v1 document was not possible."""
    lib, handle = graph.lib, graph._handle
    modules = []
    for index in range(_count(lib, "nodus_headless_get_module_count", handle)):
        desc = CanvasModuleDesc()
        lib.nodus_headless_get_module_desc(handle, index, ctypes.byref(desc))
        stable = int(lib.nodus_headless_get_module_uuid(handle, index))
        contract = ctypes.c_int32()
        lib.nodus_headless_get_module_stack_contract(handle, index, ctypes.byref(contract))
        rows, inputs, outputs = [], 0, 0
        for row_index in range(_count(lib, "nodus_headless_get_module_row_count", handle, index)):
            info = _NativeRowInfo()
            plugin_id = ctypes.create_string_buffer(1024)
            if not lib.nodus_headless_get_module_row(
                    handle, index, row_index, ctypes.byref(info),
                    ctypes.cast(plugin_id, ctypes.c_char_p), len(plugin_id)):
                continue
            if info.kind == 0:
                row = {"kind": "input", "count": info.attachment_count}
                inputs += info.attachment_count
            elif info.kind == 2:
                row = {"kind": "output", "count": info.attachment_count}
                outputs += info.attachment_count
            elif info.tool_origin == 1:
                row = {"kind": "plugin", "plugin_id": plugin_id.value.decode("utf-8"),
                       "consumes": info.consumes, "produces": info.produces}
            else:
                try:
                    tool = BuiltinTool(info.tool).name
                except ValueError:
                    tool = int(info.tool)
                row = {"kind": "builtin", "tool": tool}
            row["index"] = row_index
            rows.append(row)
        contacts = ([{"index": i, "ordinal": i, "direction": "input", "name": None}
                     for i in range(inputs)]
                    + [{"index": inputs + i, "ordinal": i, "direction": "output", "name": None}
                       for i in range(outputs)])
        modules.append({
            "index": index, "stable_id": stable or None,
            "label": bytes(desc.label).split(b"\0", 1)[0].decode("utf-8"),
            "x": desc.x, "y": desc.y, "width": desc.w, "height": desc.h,
            "contract": _contract_from_native(contract.value).value,
            "exec_mode": None, "rows": rows, "contacts": contacts})
    edges = []
    for index in range(_count(lib, "nodus_headless_get_edge_count", handle)):
        desc, type_id, native = CanvasEdgeDesc(), ctypes.c_int32(), _NativeFifoConfig()
        if not lib.nodus_headless_get_edge(handle, index, ctypes.byref(desc), ctypes.byref(type_id)):
            continue
        fifo = None
        if lib.nodus_headless_get_edge_fifo_config(handle, index, ctypes.byref(native)):
            fifo = _fifo_to_mapping(_fifo_from_native(native))
        edges.append({"index": index,
                      "source": {"module_index": desc.a_module, "contact_index": desc.a_contact_idx},
                      "target": {"module_index": desc.b_module, "contact_index": desc.b_contact_idx},
                      "fifo": fifo, "type_id": type_id.value,
                      "token_bytes": fifo["token_bytes"] if fifo else None,
                      "slots": fifo["slots"] if fifo else None})
    ports = {False: [], True: []}
    for is_output in (False, True):
        for ordinal in range(_count(lib, "nodus_headless_get_port_count", handle, int(is_output))):
            name = ctypes.create_string_buffer(1024)
            mi, ci, head = ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32()
            native = _NativeFifoConfig()
            if lib.nodus_headless_get_port_at(
                    handle, int(is_output), ordinal, ctypes.cast(name, ctypes.c_char_p),
                    len(name), ctypes.byref(mi), ctypes.byref(ci), ctypes.byref(head),
                    ctypes.byref(native)):
                ports[is_output].append({
                    "name": name.value.decode("utf-8"),
                    "connector": {"module_index": mi.value, "contact_index": ci.value},
                    "fifo": _fifo_to_mapping(_fifo_from_native(native)),
                    "start_at_head": bool(head.value)})
    return {"schema": "nodus.graph-definition.v1", "definition_error": reason,
            "revision": revision, "modules": modules, "edges": edges, "connectors": [],
            "inputs": ports[False], "outputs": ports[True],
            "contract_ports": contract_port_documents(graph)}
