#!/usr/bin/env python3
"""Cross-check canonical_ops.json against turing's *live* source.

    python ops/verify_canonical_ops.py [--turing PATH]

Why: canonical_ops.json vendors turing's `CTensorOp` ordinals rather than inventing a
second numbering (turing's ctensor_ops.h says outright that no second numeric opcode
table should exist). Vendoring a number is only safe if something notices when the
original moves. That is this script's whole job.

Read-only. Parses, and never writes to, turing:
  * c_backend/ctensor_ops.h  -- the CTensorOp enum: names and ordinals
  * c_backend/ctensor_ops.c  -- which named C functions actually exist
  * c_backend.py             -- the op-string -> CT_OP_* dispatch dicts
  * transmogrifier/ssa_registry.py -- the Handler enum members

Exit codes: 0 agree, 1 disagreement, 2 could not locate sources.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOURCE = HERE / "canonical_ops.json"
DEFAULT_TURING = REPO.parent / "turing"

REL = {
    "ctensor_ops.h": Path("src/common/tensors/accelerator_backends/c_backend/ctensor_ops.h"),
    "ctensor_ops.c": Path("src/common/tensors/accelerator_backends/c_backend/ctensor_ops.c"),
    "c_backend.py": Path("src/common/tensors/accelerator_backends/c_backend.py"),
    "ssa_registry.py": Path("src/transmogrifier/ssa_registry.py"),
}


def parse_ctensor_enum(text: str) -> dict[str, int]:
    """Parse `typedef enum CTensorOp { ... }`, honouring C's implicit numbering."""
    m = re.search(r"typedef\s+enum\s+CTensorOp\s*\{(.*?)\}", text, re.S)
    if not m:
        return {}
    values: dict[str, int] = {}
    nxt = 0
    for raw in m.group(1).split(","):
        item = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
        item = re.sub(r"//.*", "", item).strip()
        if not item:
            continue
        em = re.match(r"^(CT_OP_\w+)\s*(?:=\s*(-?\d+))?$", item)
        if not em:
            continue
        name, explicit = em.group(1), em.group(2)
        nxt = int(explicit) if explicit is not None else nxt
        values[name] = nxt
        nxt += 1
    return values


def check_ct_enum(live: dict[str, int], ops: list[dict], errors: list[str],
                  warnings: list[str]) -> None:
    if not live:
        errors.append("ctensor_ops.h: could not parse `typedef enum CTensorOp` -- source shape changed")
        return

    sentinel = "CT_OP_COUNT"
    real = {k: v for k, v in live.items() if k != sentinel}
    if sentinel not in live:
        warnings.append(f"ctensor_ops.h: no {sentinel} sentinel found")
    elif live[sentinel] != len(real):
        errors.append(
            f"{sentinel} is {live[sentinel]} but {len(real)} members precede it -- "
            f"the enum has explicit values this parser mishandled"
        )

    tabled = {o["ct_op"]: o for o in ops if o["ct_op"] is not None}

    for name, value in sorted(real.items(), key=lambda kv: kv[1]):
        op = tabled.get(name)
        if op is None:
            errors.append(
                f"turing defines {name} (= {value}); canonical_ops.json does not record it. "
                f"Add an entry with ct_op={name!r}, ct_value={value}."
            )
        elif op["ct_value"] != value:
            errors.append(
                f"{name}: turing says {value}, canonical_ops.json vendors "
                f"{op['ct_value']}. turing is authoritative -- update the snapshot."
            )

    for name, op in sorted(tabled.items()):
        if name not in real:
            errors.append(
                f"op '{op['name']}' vendors {name}, which turing's CTensorOp no longer defines"
            )


def check_dispatch_dicts(text: str, ops: list[dict], errors: list[str],
                         warnings: list[str]) -> None:
    """Every op string turing dispatches must resolve in the table, and agree on the code."""
    pairs = re.findall(r"""["'](\w+)["']\s*:\s*C\.(CT_OP_\w+)""", text)
    if not pairs:
        warnings.append(
            "c_backend.py: found no \"name\": C.CT_OP_* dispatch entries -- "
            "the backend may have moved to a different lowering shape"
        )
        return

    by_name = {o["name"]: o for o in ops}
    for op_string, ct_op in pairs:
        op = by_name.get(op_string)
        if op is None:
            errors.append(
                f"c_backend.py dispatches op string '{op_string}' -> {ct_op}, but "
                f"canonical_ops.json has no op of that name"
            )
        elif op["ct_op"] != ct_op:
            errors.append(
                f"op '{op_string}': c_backend.py maps it to {ct_op}, "
                f"canonical_ops.json says {op['ct_op']}"
            )


def check_c_functions(text: str, ops: list[dict], errors: list[str]) -> None:
    defined = set(re.findall(r"\b(?:void|double|int|float)\s+(\w+)\s*\(", text))
    if not defined:
        errors.append("ctensor_ops.c: parsed zero function definitions -- source shape changed")
        return
    for o in ops:
        if o["c_fn"] and o["c_fn"] not in defined:
            errors.append(f"op '{o['name']}': c_fn {o['c_fn']!r} is not defined in ctensor_ops.c")


def check_handlers(text: str, ops: list[dict], errors: list[str],
                   warnings: list[str]) -> None:
    m = re.search(r"class\s+Handler\s*\(\s*Enum\s*\)\s*:", text)
    if not m:
        errors.append("ssa_registry.py: could not locate `class Handler(Enum)`")
        return
    rest = text[m.end():]
    stop = re.search(r"^\S", rest, re.M)          # next top-level statement
    body = rest[:stop.start()] if stop else rest

    members = set(re.findall(r"^\s{4}(\w+)\s*=\s*[\"']", body, re.M))
    if not members:
        errors.append("ssa_registry.py: parsed zero Handler members -- source shape changed")
        return

    for o in ops:
        h = o["handler"]
        if h and h not in members:
            errors.append(
                f"op '{o['name']}': handler {h!r} is not a member of turing's Handler enum"
            )

    uncovered = sorted(members - {o["handler"] for o in ops if o["handler"]})
    if uncovered:
        warnings.append(
            f"{len(uncovered)} Handler members map to no canonical op -- expected, these are "
            f"control-flow / memory / SSA-only: {', '.join(uncovered)}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turing", type=Path, default=DEFAULT_TURING,
                    help=f"path to the turing repo (default: {DEFAULT_TURING})")
    args = ap.parse_args(argv)

    ops = json.loads(SOURCE.read_text(encoding="utf-8"))["ops"]

    paths = {k: args.turing / v for k, v in REL.items()}
    absent = [f"{k}: {v}" for k, v in paths.items() if not v.is_file()]
    if absent:
        print("cannot locate turing sources:", file=sys.stderr)
        for a in absent:
            print(f"  - {a}", file=sys.stderr)
        print("pass --turing PATH", file=sys.stderr)
        return 2

    read = {k: p.read_text(encoding="utf-8", errors="replace") for k, p in paths.items()}
    errors: list[str] = []
    warnings: list[str] = []

    check_ct_enum(parse_ctensor_enum(read["ctensor_ops.h"]), ops, errors, warnings)
    check_c_functions(read["ctensor_ops.c"], ops, errors)
    check_dispatch_dicts(read["c_backend.py"], ops, errors, warnings)
    check_handlers(read["ssa_registry.py"], ops, errors, warnings)

    for w in warnings:
        print(f"note: {w}")

    if errors:
        print(f"\n{len(errors)} disagreement(s) with turing's live source:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    coded = sum(1 for o in ops if o["ct_value"] is not None)
    print(f"OK: {len(ops)} canonical ops ({coded} vendored from CTensorOp, "
          f"{sum(1 for o in ops if o['lowerable'])} lowerable) agree with turing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
