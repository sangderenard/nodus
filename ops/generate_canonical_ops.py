#!/usr/bin/env python3
"""Generate the canonical-op bindings for nodus (C++) and turing (Python).

Single source of truth: ``canonical_ops.json``.

    python ops/generate_canonical_ops.py            # write outputs
    python ops/generate_canonical_ops.py --check     # fail if outputs are stale

Outputs (never hand-edit):
    include/canonical_ops.h          header-only, dependency-free C++
    ops/canonical_ops_generated.py   Python mirror + reverse lookup maps

``canonical_id`` is the append-only catalog position shared by every lowering
surface. ``ct_value`` remains a separate vendored snapshot of the subset
implemented by Turing's current CTensorOp dispatcher. Drift detection for that
subset lives in ``verify_canonical_ops.py``.

Why generated at all: see ``research/15_the_missing_function_table.md``. Every previous
attempt at this table in either repo was authored as code and left incomplete. This one
is authored as data, so the two languages cannot drift and neither copy is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOURCE = HERE / "canonical_ops.json"
CPP_OUT = REPO / "include" / "canonical_ops.h"
PY_OUT = HERE / "canonical_ops_generated.py"

OP_CLASSES = ("unary", "binary", "compare", "cast", "opaque")

# OpCode members of nodus::spirv::OpCode that entries may reference. A literal list, so a
# typo in the JSON fails here rather than at C++ compile time.
KERNEL_OPS = {
    "MODULE_BEGIN", "KERNEL_ENTRY", "TYPE", "CONST", "SPEC_CONST", "VAR", "ADDR",
    "LOAD", "STORE", "MEMCPY", "UNARY", "BINARY", "TERNARY", "CMP", "SELECT",
    "CAST", "EXTRACT", "INSERT", "SHUFFLE", "IF", "BARRIER", "ATOMIC",
    "AND", "OR", "NOT", "XOR",
}

REQUIRED_FIELDS = {
    "name", "class", "ct_op", "ct_value", "arity", "returns", "kernel_op",
    "lowerable", "handler", "sympy", "reflectable", "c_fn", "notes",
}

BANNER = """// GENERATED FILE -- DO NOT EDIT.
// Source:    ops/canonical_ops.json
// Generator: ops/generate_canonical_ops.py
// Verifier:  ops/verify_canonical_ops.py  (checks ct_value against turing's live header)
// Rationale: research/15_the_missing_function_table.md
//
// The canonical operation set shared by turing (Python) and nodus (C++). Both sides
// reduce their own op spellings to these names; anything that agrees here is
// interchangeable regardless of which side produced it.
//
// Canonical IDs are append-only catalog positions. CTensorOp ordinals remain a
// separate, verified backend field for the currently implemented C subset.
"""

PY_BANNER = '''"""GENERATED FILE -- DO NOT EDIT.

Source:    ops/canonical_ops.json
Generator: ops/generate_canonical_ops.py
Rationale: research/15_the_missing_function_table.md
"""
'''


class SchemaError(Exception):
    pass


def validate(ops: list[dict]) -> None:
    """Reject anything that would make the generated tables lie."""
    errors: list[str] = []
    seen_names: set[str] = set()
    seen_ct_values: dict[int, str] = {}
    seen_ct_ops: dict[str, str] = {}
    seen_handlers: dict[str, list[str]] = {}
    seen_sympy: dict[str, str] = {}

    for i, op in enumerate(ops):
        missing = REQUIRED_FIELDS - set(op)
        if missing:
            errors.append(f"ops[{i}]: missing fields {sorted(missing)}")
            continue
        extra = set(op) - REQUIRED_FIELDS
        if extra:
            errors.append(f"ops[{i}]: unknown fields {sorted(extra)}")

        name = op["name"]
        where = f"op '{name}'"
        if name in seen_names:
            errors.append(f"{where}: duplicate name")
        seen_names.add(name)
        if name != name.lower():
            errors.append(f"{where}: name must be lowercase")

        if op["class"] not in OP_CLASSES:
            errors.append(f"{where}: unknown class '{op['class']}'")

        # ct_op and ct_value travel together, or are both absent (a recorded hole).
        ct_op, ct_value = op["ct_op"], op["ct_value"]
        if (ct_op is None) != (ct_value is None):
            errors.append(f"{where}: ct_op and ct_value must both be set or both be null")
        if ct_op is not None:
            if not ct_op.startswith("CT_OP_"):
                errors.append(f"{where}: ct_op '{ct_op}' should start with CT_OP_")
            if ct_op in seen_ct_ops:
                errors.append(f"{where}: ct_op '{ct_op}' already used by '{seen_ct_ops[ct_op]}'")
            seen_ct_ops[ct_op] = name
            if not isinstance(ct_value, int) or ct_value < 0:
                errors.append(f"{where}: ct_value must be a non-negative int")
            elif ct_value in seen_ct_values:
                errors.append(
                    f"{where}: ct_value {ct_value} collides with '{seen_ct_values[ct_value]}'"
                )
            else:
                seen_ct_values[ct_value] = name
            if op["class"] == "opaque":
                errors.append(
                    f"{where}: class 'opaque' must not have a ct_op -- turing's dispatcher "
                    f"requires every instruction to write one equally-shaped slot"
                )

        if op["returns"] not in ("value", "bool"):
            errors.append(f"{where}: returns must be 'value' or 'bool'")
        if not isinstance(op["arity"], int) or op["arity"] < 0:
            errors.append(f"{where}: arity must be a non-negative int")
        for flag in ("lowerable", "reflectable"):
            if not isinstance(op[flag], bool):
                errors.append(f"{where}: {flag} must be a bool")

        kop, low = op["kernel_op"], op["lowerable"]
        if low and kop is None:
            errors.append(f"{where}: lowerable=true requires a kernel_op")
        if not low and kop is not None:
            errors.append(f"{where}: lowerable=false must have kernel_op null")
        if kop is not None and kop not in KERNEL_OPS:
            errors.append(f"{where}: kernel_op '{kop}' is not a nodus::spirv::OpCode member")
        if op["class"] == "opaque" and low:
            errors.append(f"{where}: class 'opaque' cannot be lowerable")

        if op["handler"] is not None:
            seen_handlers.setdefault(op["handler"], []).append(name)

        # A sympy class name must not resolve to two canonical ops: that ambiguity is
        # precisely what this table exists to eliminate.
        for s in op["sympy"]:
            if s in seen_sympy:
                errors.append(f"{where}: sympy name '{s}' already maps to '{seen_sympy[s]}'")
            seen_sympy[s] = name

    # Handler fan-out is legal only for Call, which turing uses as the catch-all for every
    # elementary function. Anything else fanning out is a modelling mistake.
    for handler, names in sorted(seen_handlers.items()):
        if len(names) > 1 and handler != "Call":
            errors.append(
                f"handler '{handler}' fans out to {sorted(names)}; only 'Call' may fan out"
            )

    # ct_values must be a gapless 0..N-1 block, mirroring a C enum with no explicit values.
    if seen_ct_values:
        expected = set(range(len(seen_ct_values)))
        if set(seen_ct_values) != expected:
            gaps = sorted(expected - set(seen_ct_values))
            errors.append(
                f"ct_value block is not gapless 0..{len(seen_ct_values) - 1}; missing {gaps}"
            )

    if errors:
        raise SchemaError("\n".join(f"  - {e}" for e in errors))


def cpp_str(value) -> str:
    if value is None:
        return '""'
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def enumerator(name: str) -> str:
    return name.upper()


def emit_cpp(data: dict) -> str:
    ops = data["ops"]
    out = [BANNER, "#pragma once", "", "#include <cstdint>", "#include <string_view>", "",
           "namespace nodus {", "namespace ops {", ""]

    out += [
        "// Sub-opcode carried by KernelIR's UNARY / BINARY / CMP / CAST instructions,",
        "// which reserve the slot but never defined the index space (research/15).",
        "// Every catalog operation has an ID, including operations not yet available",
        "// in the C dispatcher. IDs are append-only catalog positions.",
        "enum class CanonicalOp : uint16_t {",
    ]
    for canonical_id, o in enumerate(ops):
        suffix = f"  // {o['ct_op']}" if o["ct_op"] else ""
        out.append(f"    {enumerator(o['name'])} = {canonical_id},{suffix}")
    out.append(f"    COUNT = {len(ops)},")
    out.append("};")
    out.append("")

    out += [
        "enum class OpClass : uint8_t { Unary, Binary, Compare, Cast, Opaque };",
        "",
        "struct OpDesc {",
        "    std::string_view name;         // canonical name -- the one true key",
        "    uint16_t         canonical_id; // append-only shared operation ID",
        "    OpClass          op_class;",
        "    int32_t          ct_value;     // turing CTensorOp ordinal, or -1 (no C target)",
        "    std::string_view ct_op;        // CTensorOp member name, or \"\"",
        "    uint8_t          arity;",
        "    bool             returns_bool;",
        "    bool             lowerable;    // false => must stay native (CAP_BINARY)",
        "    bool             reflectable;  // has a distinct reversed-operand form",
        "    std::string_view kernel_op;    // nodus::spirv::OpCode name, or \"\"",
        "    std::string_view handler;      // turing Handler member, or \"\"",
        "    std::string_view c_fn;         // named C function outside the dispatcher, or \"\"",
        "};",
        "",
    ]

    cls_map = {"unary": "OpClass::Unary", "binary": "OpClass::Binary",
               "compare": "OpClass::Compare", "cast": "OpClass::Cast",
               "opaque": "OpClass::Opaque"}

    out.append("inline constexpr OpDesc kOps[] = {")
    for canonical_id, o in enumerate(ops):
        out.append("    {" + ", ".join([
            cpp_str(o["name"]),
            str(canonical_id),
            cls_map[o["class"]],
            str(-1 if o["ct_value"] is None else o["ct_value"]),
            cpp_str(o["ct_op"]),
            str(o["arity"]),
            "true" if o["returns"] == "bool" else "false",
            "true" if o["lowerable"] else "false",
            "true" if o["reflectable"] else "false",
            cpp_str(o["kernel_op"]),
            cpp_str(o["handler"]),
            cpp_str(o["c_fn"]),
        ]) + "},")
    out.append("};")
    out.append("")
    out.append(f"inline constexpr size_t kOpCount = {len(ops)};")
    out.append("")

    out += [
        "// Lookups return nullptr when unknown. A hard null beats a silent zero: research/06",
        "// and research/12 document what silently-defaulting lookups have already cost here.",
        "inline constexpr const OpDesc* find_op(std::string_view name) {",
        "    for (size_t i = 0; i < kOpCount; ++i) {",
        "        if (kOps[i].name == name) return &kOps[i];",
        "    }",
        "    return nullptr;",
        "}",
        "",
        "inline constexpr const OpDesc* find_op(CanonicalOp op) {",
        "    for (size_t i = 0; i < kOpCount; ++i) {",
        "        if (kOps[i].canonical_id == static_cast<uint16_t>(op)) return &kOps[i];",
        "    }",
        "    return nullptr;",
        "}",
        "",
        "// Resolve a turing Handler member. Handler::Call fans out across every elementary",
        "// function, so this returns the first match only and is unsuitable for Call --",
        "// prefer find_op(name). Present to make the correspondence expressible, not to",
        "// dispatch on.",
        "inline constexpr const OpDesc* find_op_by_handler(std::string_view handler) {",
        "    for (size_t i = 0; i < kOpCount; ++i) {",
        "        if (!kOps[i].handler.empty() && kOps[i].handler == handler) return &kOps[i];",
        "    }",
        "    return nullptr;",
        "}",
        "",
        "} // namespace ops",
        "} // namespace nodus",
        "",
    ]
    return "\n".join(out)


def emit_python(data: dict) -> str:
    ops = data["ops"]
    out = [PY_BANNER, "from __future__ import annotations", "",
           "from enum import IntEnum", "from typing import NamedTuple", "", ""]

    out.append("class CanonicalOp(IntEnum):")
    out.append('    """Append-only operation IDs shared by all lowering surfaces."""')
    for canonical_id, o in enumerate(ops):
        suffix = f"  # {o['ct_op']}" if o["ct_op"] else ""
        out.append(f"    {enumerator(o['name'])} = {canonical_id}{suffix}")
    out += ["", ""]

    out += [
        "class OpDesc(NamedTuple):",
        "    name: str",
        "    canonical_id: int",
        "    op_class: str",
        "    ct_op: str | None",
        "    ct_value: int | None",
        "    arity: int",
        "    returns: str",
        "    lowerable: bool",
        "    reflectable: bool",
        "    kernel_op: str | None",
        "    handler: str | None",
        "    sympy: tuple[str, ...]",
        "    c_fn: str | None",
        "    notes: str | None",
        "",
        "",
        "OPS: tuple[OpDesc, ...] = (",
    ]
    for canonical_id, o in enumerate(ops):
        out.append("    OpDesc(")
        out.append(f"        canonical_id={canonical_id},")
        for field, key in (("name", "name"), ("op_class", "class"), ("ct_op", "ct_op"),
                           ("ct_value", "ct_value"), ("arity", "arity"),
                           ("returns", "returns"), ("lowerable", "lowerable"),
                           ("reflectable", "reflectable"), ("kernel_op", "kernel_op"),
                           ("handler", "handler")):
            out.append(f"        {field}={o[key]!r},")
        out.append(f"        sympy={tuple(o['sympy'])!r},")
        out.append(f"        c_fn={o['c_fn']!r},")
        out.append(f"        notes={o['notes']!r},")
        out.append("    ),")
    out += [
        ")",
        "",
        "BY_NAME: dict[str, OpDesc] = {o.name: o for o in OPS}",
        "",
        "BY_CANONICAL_ID: dict[int, OpDesc] = {o.canonical_id: o for o in OPS}",
        "",
        "# sympy class name (lowercased, as keyed in transmogrifier.ssa_registry) -> canonical op.",
        "BY_SYMPY: dict[str, OpDesc] = {s: o for o in OPS for s in o.sympy}",
        "",
        "# turing CTensorOp ordinal -> canonical op.",
        "BY_CT_VALUE: dict[int, OpDesc] = {",
        "    o.ct_value: o for o in OPS if o.ct_value is not None",
        "}",
        "",
        "# turing Handler member -> canonical ops. A list, not a scalar: Handler.Call",
        "# legitimately fans out across every elementary function.",
        "BY_HANDLER: dict[str, list[OpDesc]] = {}",
        "for _o in OPS:",
        "    if _o.handler is not None:",
        "        BY_HANDLER.setdefault(_o.handler, []).append(_o)",
        "del _o",
        "",
        "",
        "def resolve(op_string: str) -> tuple[OpDesc, bool] | None:",
        '    """Resolve an AbstractTensor ``_apply_operator__`` op string.',
        "",
        "    Returns ``(desc, reversed_operands)`` or None. Mirrors the i/r-prefix",
        "    normalization in turing's c_backend, but only strips a prefix when doing so",
        "    actually yields a known op -- so 'isnan', 'invert' and 'round' cannot be",
        "    mangled into 'snan', 'nvert' and 'ound' the way blind stripping would.",
        '    """',
        "    direct = BY_NAME.get(op_string)",
        "    if direct is not None:",
        "        return direct, False",
        "    if op_string[:1] in ('i', 'r'):",
        "        base = BY_NAME.get(op_string[1:])",
        "        if base is not None:",
        "            return base, op_string[0] == 'r'",
        "    return None",
        "",
        "",
        "def lowerable_ops() -> tuple[OpDesc, ...]:",
        '    """Ops expressible as a single KernelIR instruction."""',
        "    return tuple(o for o in OPS if o.lowerable)",
        "",
        "",
        "def holes(surface: str) -> tuple[OpDesc, ...]:",
        '    """Ops with no implementation on ``surface``.',
        "",
        "    This is the 'method union with holes marked' that operator_defs.py's",
        "    mirrored_*_funcs built as data and nothing ever read.",
        '    """',
        "    if surface == 'c':",
        "        return tuple(o for o in OPS if o.ct_op is None and o.c_fn is None)",
        "    if surface == 'kernel_ir':",
        "        return tuple(o for o in OPS if not o.lowerable)",
        "    raise ValueError(f'unknown surface: {surface!r}')",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify outputs are current; write nothing, exit 1 if stale")
    args = ap.parse_args(argv)

    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    try:
        validate(data["ops"])
    except SchemaError as exc:
        print(f"canonical_ops.json failed validation:\n{exc}", file=sys.stderr)
        return 2

    targets = {CPP_OUT: emit_cpp(data), PY_OUT: emit_python(data)}

    if args.check:
        stale = [p for p, text in targets.items()
                 if not p.exists() or p.read_text(encoding="utf-8") != text]
        for p in stale:
            print(f"STALE: {p.relative_to(REPO)}", file=sys.stderr)
        if stale:
            print("run: python ops/generate_canonical_ops.py", file=sys.stderr)
            return 1
        print(f"up to date ({len(data['ops'])} ops)")
        return 0

    for path, text in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")

    ops = data["ops"]
    print(f"{len(ops)} canonical ops: "
          f"{sum(1 for o in ops if o['ct_value'] is not None)} with a C target, "
          f"{sum(1 for o in ops if o['lowerable'])} lowerable to KernelIR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
