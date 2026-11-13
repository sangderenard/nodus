"""Strict relocation of GLSL default-block uniforms into one std430 block.

This only changes the delivery ABI. Initializers require an explicit output
map so the host can restore their defaults. It refuses duplicate or unexpected
default-block declarations and opaque uniforms. The caller must verify
reflected offsets and supply matching host data before execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


@dataclass(frozen=True)
class UniformField:
    name: str
    glsl_type: str
    array: str = ""


@dataclass(frozen=True)
class ParsedUniform:
    field: UniformField
    initializer: str = ""


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
_ARRAY = re.compile(r"^(?:[0-9]+|[A-Za-z_][A-Za-z_0-9]*)$")
_DECL = re.compile(
    r"^\s*uniform\s+(?P<type>[A-Za-z_][A-Za-z_0-9]*)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
    r"(?:\s*\[\s*(?P<array>[A-Za-z_0-9]+)\s*\])?"
    r"\s*(?:=\s*(?P<initializer>[^;]+))?\s*;"
    r"\s*(?:(?://[^\r\n]*)|(?:/\*[^\r\n]*\*/))?\s*$"
)


def collect_uniforms(source: str) -> list[ParsedUniform]:
    declarations=[]
    for line in source.splitlines():
        match=_DECL.fullmatch(line)
        if match:
            declarations.append(ParsedUniform(UniformField(
                match.group("name"),match.group("type"),match.group("array") or ""),
                (match.group("initializer") or "").strip()))
        elif re.match(r"^\s*uniform\s+",line):
            raise ValueError("unsupported default uniform syntax")
    return declarations


def relocate_uniforms(source: str, fields: Sequence[UniformField], *,
                      binding: int, block_name: str,
                      allow_unlisted: bool = False,
                      defaults_out: dict[str,str] | None = None) -> str:
    if not fields or binding < 0 or not _IDENTIFIER.fullmatch(block_name):
        raise ValueError("invalid storage-control block")
    expected = {}
    for field in fields:
        if not _IDENTIFIER.fullmatch(field.name) or \
           not _IDENTIFIER.fullmatch(field.glsl_type) or \
           (field.array and not _ARRAY.fullmatch(field.array)) or \
           field.name in expected or "sampler" in field.glsl_type or \
           "image" in field.glsl_type:
            raise ValueError("invalid or opaque uniform field")
        expected[field.name] = field
    lines = source.splitlines(keepends=True)
    found: dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        match = _DECL.fullmatch(stripped)
        if match:
            name = match.group("name")
            if name not in expected:
                if allow_unlisted:continue
                raise ValueError(f"unexpected default uniform {name}")
            if name in found:
                raise ValueError(f"unexpected or duplicate default uniform {name}")
            field = expected[name]
            if (match.group("type"), match.group("array") or "") != \
               (field.glsl_type, field.array):
                raise ValueError(f"uniform declaration changed: {name}")
            initializer=(match.group("initializer") or "").strip()
            if initializer:
                if defaults_out is None:
                    raise ValueError(f"uniform initializer requires host default: {name}")
                defaults_out[name]=initializer
            found[name] = i
        elif re.match(r"^\s*uniform\s+", stripped):
            raise ValueError("unsupported default uniform syntax or initializer")
    if set(found) != set(expected):
        raise ValueError(f"missing default uniforms: {sorted(set(expected)-set(found))}")
    block = f"layout(std430, binding = {binding}) readonly buffer {block_name} {{\n"
    for field in fields:
        suffix = f"[{field.array}]" if field.array else ""
        block += f"    {field.glsl_type} {field.name}{suffix};\n"
    block += "};\n"
    first = min(found.values())
    for i in found.values():
        lines[i] = "\n"
    lines[first] = block
    return "".join(lines)
