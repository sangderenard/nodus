"""Compile a table of Turing-compiled rows into one row: the rows' LLVM joined into one plugin (research case 6393486c).

Owner, 2026-11-18: "the llvm sources get concated together as the compiled table, you know, compiled canvases, they
become plugins that have a history of development that started with nothing more than a python program".

A table's row program runs on its stack: the Input row pushes the table's inputs, each row pops its arguments and
pushes its results, and the Output row pops the table's outputs. This tool runs that program over buffer *values*
instead of tensors, at compile time:

- a compiled row (module_library/products/<id>, NODUS BUFFER ABI V2, backend llvm) consumes its INPUT values in
  stack order and produces its OUTPUT values. Each value's dtype and shape must equal what the row's ABI declares.
- tensor_handle_dup pushes the top value again: one buffer, read twice.
- tensor_returns is the borrowed-tensor return path. In one compiled call every intermediate lives only for that call,
  so it compiles to nothing.

The joined entry ``void <table>(void** buffers, int32_t* extents)`` takes the table's inputs, then its outputs, then
one workspace per intermediate value, and calls each row's entry on its own buffer list in program order. Each row's
LLVM module is the one Turing emitted for it (backend/<id>.ll), linked unchanged with llvmlite.link_modules, the
mechanism Turing uses to link LLVM modules (llvm_signal_math.link_llvm_trig_solver). Row entries become internal, so
the library exports only the table's entry. Turing's compile_artifact builds it. The descriptor is NODUS BUFFER ABI V2:
the inputs and outputs as the table's columns, the intermediates as WORKSPACE.

The product directory keeps the history: the table program, and for each row its authored Python source and digest.

Usage: compile_row_table.py <table.json>, where table.json is
    {"table": "<entry>", "module_library_root": "...", "turing_root": "...",
     "inputs": [{"name": "x", "dtype": "f64", "shape": [4, 4]}],
     "rows": ["tensor_handle_dup", "<product id>", ...],
     "outputs": ["result"]}
It prints one JSON line: the product directory, descriptor, library and joined source.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The rows whose compiled meaning is a stack operation, by their declared ids.
STACK_ROWS = {"tensor_handle_dup": "dup", "tensor_returns": "returns"}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class TableError(ValueError):
    pass


@dataclass
class Value:
    name: str
    dtype: str
    shape: tuple[int, ...]
    role: str = "WORKSPACE"


@dataclass
class RowAbi:
    product: str
    entry: str
    values: dict[str, tuple[str, str, tuple[int, ...]]] = field(default_factory=dict)  # id -> (role, dtype, shape)
    buffers: list[str] = field(default_factory=list)
    llvm_path: Path | None = None


def read_row_abi(products: Path, product: str) -> RowAbi:
    backend = products / product / "backend"
    descriptor = backend / f"{product}.nodus_buffer_abi.txt"
    if not descriptor.is_file():
        raise TableError(f"row {product!r} has no NODUS BUFFER ABI descriptor at {descriptor}")
    abi = RowAbi(product=product, entry="", llvm_path=backend / f"{product}.ll")
    lines = descriptor.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "NODUS BUFFER ABI V2":
        raise TableError(f"row {product!r}: descriptor is not NODUS BUFFER ABI V2")
    for line in lines[1:]:
        parts = shlex.split(line)
        if not parts:
            continue
        tag = parts[0]
        if tag == "BACKEND" and parts[1] != "llvm":
            raise TableError(f"row {product!r}: backend {parts[1]!r}; a compiled table joins LLVM rows")
        elif tag == "ENTRY":
            abi.entry = parts[1]
        elif tag == "EXTENT":
            raise TableError(f"row {product!r} takes runtime extents; a compiled table joins fixed-shape rows")
        elif tag == "VALUE":
            role, vid, dtype, rank = parts[1], parts[2], parts[3], int(parts[4])
            dims = tuple(int(d) for d in parts[5:5 + rank])
            if len(dims) != rank:
                raise TableError(f"row {product!r}: value {vid} is runtime-sized; a compiled table joins fixed shapes")
            abi.values[vid] = (role, dtype, dims)
        elif tag == "BUFFER":
            abi.buffers.append(parts[1])
    if not abi.entry or not abi.llvm_path.is_file():
        raise TableError(f"row {product!r}: missing ENTRY or its LLVM module {abi.llvm_path}")
    return abi


def plan_table(spec: dict, products: Path):
    """Run the row program over values. Returns (values, calls, outputs, row abis)."""
    values: list[Value] = []
    stack: list[Value] = []
    for item in spec["inputs"]:
        v = Value(str(item["name"]), str(item["dtype"]), tuple(int(d) for d in item["shape"]), "INPUT")
        values.append(v)
        stack.append(v)
    calls: list[tuple[RowAbi, list[Value]]] = []
    abis: dict[str, RowAbi] = {}
    for position, row in enumerate(spec["rows"]):
        kind = STACK_ROWS.get(row)
        if kind == "dup":
            if not stack:
                raise TableError(f"row {position} (dup): empty stack")
            stack.append(stack[-1])
            continue
        if kind == "returns":
            continue
        abi = abis.get(row) or read_row_abi(products, row)
        abis[row] = abi
        ins = [vid for vid in abi.buffers if abi.values[vid][0] == "INPUT"]
        outs = [vid for vid in abi.buffers if abi.values[vid][0] == "OUTPUT"]
        if any(abi.values[vid][0] == "WORKSPACE" for vid in abi.buffers):
            raise TableError(f"row {position} ({row}) has workspaces; not joined yet")
        if len(stack) < len(ins):
            raise TableError(f"row {position} ({row}) needs {len(ins)} values; the stack holds {len(stack)}")
        popped = stack[len(stack) - len(ins):]
        del stack[len(stack) - len(ins):]
        bound: dict[str, Value] = {}
        for vid, value in zip(ins, popped):  # the wrapper pops its last input first: inputs in stack order
            _role, dtype, shape = abi.values[vid]
            if (value.dtype, value.shape) != (dtype, shape):
                raise TableError(f"row {position} ({row}) input {vid} is {dtype}{list(shape)}; "
                                 f"the stack gives {value.name} {value.dtype}{list(value.shape)}")
            bound[vid] = value
        for k, vid in enumerate(outs):
            _role, dtype, shape = abi.values[vid]
            v = Value(f"r{position}_{k}", dtype, shape)
            values.append(v)
            bound[vid] = v
            stack.append(v)
        calls.append((abi, [bound[vid] for vid in abi.buffers]))
    names = [str(n) for n in spec["outputs"]]
    if len(stack) != len(names):
        raise TableError(f"the program leaves {len(stack)} values for {len(names)} outputs")
    outputs = list(stack)
    for v, name in zip(outputs, names):
        if v.role == "INPUT":
            raise TableError(f"output {name!r} is the table input {v.name!r} passed through")
        v.role, v.name = "OUTPUT", name
    return values, calls, outputs, abis


def table_llvm(table: str, order: list[Value], calls) -> str:
    slot = {id(v): i for i, v in enumerate(order)}
    lines = [f'source_filename = "nodus.compiled-table.{table}"', "", f"define void @{table}(ptr %buffers, ptr %extents) {{",
             "entry:"]
    for i in range(len(order)):
        lines.append(f"  %b{i}.addr = getelementptr ptr, ptr %buffers, i64 {i}")
        lines.append(f"  %b{i} = load ptr, ptr %b{i}.addr, align 8")
    for c, (abi, args) in enumerate(calls):
        lines.append(f"  %call{c} = alloca [{len(args)} x ptr], align 8")
        for k, v in enumerate(args):
            lines.append(f"  %call{c}.{k} = getelementptr [{len(args)} x ptr], ptr %call{c}, i64 0, i64 {k}")
            lines.append(f"  store ptr %b{slot[id(v)]}, ptr %call{c}.{k}, align 8")
        lines.append(f"  call void @{abi.entry}(ptr %call{c}, ptr null)")
    lines += ["  ret void", "}", ""]
    for entry in sorted({abi.entry for abi, _ in calls}):
        lines.append(f"declare void @{entry}(ptr, ptr)")
    return "\n".join(lines) + "\n"


def descriptor_text(table: str, order: list[Value]) -> str:
    out = ["NODUS BUFFER ABI V2", 'BACKEND "llvm"', f'TOOL "compiled_table.{table}.llvm" "compiled table {table} (llvm)"',
           f'ENTRY "{table}"', 'EXTENT_DTYPE "i32"']
    for v in order:
        out.append(f'VALUE {v.role} "%{v.name}" "{v.dtype}" {len(v.shape)} ' + " ".join(str(d) for d in v.shape))
    for v in order:
        out.append(f'BUFFER "%{v.name}"')
    return "\n".join(out) + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(spec_path: str) -> dict:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    table = str(spec["table"])
    if not _IDENTIFIER.fullmatch(table):
        raise TableError(f"invalid table entry {table!r}")
    # A table program kept with the row sources (module_library/source/tools/<table>.table.json) names neither root:
    # the library is the one it sits in, and Turing is NODUS_TURING_ROOT.
    library_root = Path(spec.get("module_library_root") or Path(spec_path).resolve().parents[2]).resolve()
    spec.setdefault("turing_root", os.environ.get("NODUS_TURING_ROOT", ""))
    if not spec["turing_root"]:
        raise TableError("no Turing root: name turing_root in the table or set NODUS_TURING_ROOT")
    products = library_root / "products"
    values, calls, outputs, abis = plan_table(spec, products)
    order = [v for v in values if v.role == "INPUT"] + outputs + [v for v in values if v.role == "WORKSPACE"]

    from llvmlite import binding as llvm  # the linker Turing itself uses (parse and link need no target set-up)
    destination = llvm.parse_assembly(table_llvm(table, order, calls))
    for abi in abis.values():
        destination.link_in(llvm.parse_assembly(abi.llvm_path.read_text(encoding="utf-8")))
    for abi in abis.values():
        destination.get_function(abi.entry).linkage = "internal"
    destination.triple = llvm.parse_assembly(next(iter(abis.values())).llvm_path.read_text(encoding="utf-8")).triple
    destination.verify()
    joined = str(destination)

    product_dir = products / table
    backend = product_dir / "backend"
    backend.mkdir(parents=True, exist_ok=True)
    descriptor = backend / f"{table}.nodus_buffer_abi.txt"
    descriptor.write_text(descriptor_text(table, order), encoding="utf-8", newline="\n")

    turing = Path(spec["turing_root"]).resolve()
    sys.path.insert(0, str(turing))
    try:
        from src.compiler.ssa_llvm_backend import LLVMFunctionArtifact, compile_artifact
    finally:
        sys.path.pop(0)
    artifact = LLVMFunctionArtifact(name=table, llvm_ir=joined, buffer_order=(), buffer_shapes=(), extent_order=(),
                                    shortfalls=())
    built = compile_artifact(artifact, directory=backend)
    library = Path(built.library_path) if built.library_path else backend / f"{table}.dll"
    if not library.is_file():
        raise TableError(f"compile_artifact produced no library for {table}")

    history = {
        "schema": "nodus-compiled-table-v1",
        "table": table,
        "program": {"inputs": spec["inputs"], "rows": spec["rows"], "outputs": spec["outputs"]},
        "buffers": [{"name": v.name, "role": v.role, "dtype": v.dtype, "shape": list(v.shape)} for v in order],
        "rows": [{
            "product": abi.product, "entry": abi.entry,
            "authored_source": str((products / abi.product / "authored-source.py").relative_to(library_root).as_posix()),
            "authored_source_sha256": sha256(products / abi.product / "authored-source.py"),
            "llvm_sha256": sha256(abi.llvm_path),
        } for abi in abis.values()],
        "joined_llvm": f"backend/{table}.ll",
        "joined_llvm_sha256": sha256(backend / f"{table}.ll"),
        "library": library.relative_to(product_dir).as_posix(),
        "library_sha256": sha256(library),
    }
    (product_dir / "nodus-compiled-table.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {"product_dir": str(product_dir), "descriptor": str(descriptor), "library": str(library),
            "joined_llvm": str(backend / f"{table}.ll")}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(main(sys.argv[1])))
