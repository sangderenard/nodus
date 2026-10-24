#!/usr/bin/env python3
"""Retain and translate a Turing-authored shader artifact in Nodus.

This is deliberately not part of ``TranslationMatrix``. That registry accepts
KernelIR callbacks; this module consumes emitted shader source plus its
``turing-compiled-program-api-v1`` contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


MANIFEST_SCHEMA = "nodus-turing-shader-artifact-v1"
API_SCHEMA = "turing-compiled-program-api-v1"
COMPONENT_SCHEMA = "turing.shader-component.v1"
RECEIPT_SCHEMA = "nodus-turing-shader-artifact-receipt-v1"

_STAGES = {"compute", "vertex", "fragment", "geometry"}
_TARGETS = {"glsl", "essl", "hlsl", "msl", "wgsl"}
_GLSL_LANGUAGES = {"glsl", "glsl-430", "webgl2-glsl-es", "glsl-es"}
_HLSL_LANGUAGES = {"hlsl", "hlsl-spirv"}
_SUFFIXES = {"glsl": ".glsl", "essl": ".essl", "hlsl": ".hlsl", "msl": ".metal"}
_GLSL_STAGE = {"compute": "comp", "vertex": "vert", "fragment": "frag", "geometry": "geom"}
_HLSL_PROFILE = {"compute": "cs_6_0", "vertex": "vs_6_0", "fragment": "ps_6_0", "geometry": "gs_6_0"}


class ShaderArtifactError(RuntimeError):
    """A fail-closed shader artifact contract or toolchain failure."""


@dataclass(frozen=True)
class ShaderIdentity:
    language: str
    stage: str
    entrypoint: str
    component_id: str


@dataclass(frozen=True)
class ShaderArtifactResult:
    receipt: Path
    retained_source: Path
    retained_api: Path
    spirv: Path
    converted: Mapping[str, Path]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShaderArtifactError(f"{label} must be a mapping")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ShaderArtifactError(f"{label} must be a non-empty string")
    return value.strip()


def _resolve_member(root: Path, value: Any, label: str) -> Path:
    spelling = _text(value, label)
    candidate = Path(spelling)
    if candidate.is_absolute():
        raise ShaderArtifactError(f"{label} must be relative to the manifest")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ShaderArtifactError(f"{label} escapes the manifest directory") from error
    if not resolved.is_file():
        raise ShaderArtifactError(f"{label} does not name a file: {spelling}")
    return resolved


def _load_manifest(path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ShaderArtifactError(f"cannot read shader manifest {path}: {error}") from error
    doc = _mapping(raw, "shader manifest")
    allowed = {"schema", "source", "api", "targets"}
    unknown = set(doc) - allowed
    if unknown:
        raise ShaderArtifactError(f"shader manifest has unknown fields: {sorted(unknown)!r}")
    if doc.get("schema") != MANIFEST_SCHEMA:
        raise ShaderArtifactError(f"shader manifest schema must be {MANIFEST_SCHEMA!r}")
    targets_raw = doc.get("targets")
    if not isinstance(targets_raw, list) or not targets_raw:
        raise ShaderArtifactError("shader manifest targets must be a non-empty list")
    targets = tuple(_text(target, "target").lower() for target in targets_raw)
    if len(set(targets)) != len(targets):
        raise ShaderArtifactError("shader manifest targets must be unique")
    unknown_targets = set(targets) - _TARGETS
    if unknown_targets:
        raise ShaderArtifactError(f"unsupported shader targets: {sorted(unknown_targets)!r}")
    if "wgsl" in targets:
        raise ShaderArtifactError(
            "WGSL conversion is unavailable: installed SPIRV-Cross has no WGSL backend"
        )
    root = path.resolve().parent
    return (
        _resolve_member(root, doc.get("source"), "source"),
        _resolve_member(root, doc.get("api"), "api"),
        targets,
    )


def _load_identity(api_path: Path) -> ShaderIdentity:
    try:
        raw = yaml.safe_load(api_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ShaderArtifactError(f"cannot read Turing API {api_path}: {error}") from error
    api = _mapping(raw, "Turing API")
    if api.get("schema") != API_SCHEMA:
        raise ShaderArtifactError(f"Turing API schema must be {API_SCHEMA!r}")
    language = _text(api.get("language"), "Turing API language").lower()
    if language == "wgsl":
        raise ShaderArtifactError(
            "WGSL source ingestion is unavailable: no installed WGSL-to-SPIR-V compiler is configured"
        )
    if language not in _GLSL_LANGUAGES | _HLSL_LANGUAGES:
        raise ShaderArtifactError(f"unsupported shader source language {language!r}")
    metadata = _mapping(api.get("metadata"), "Turing API metadata")
    component = _mapping(metadata.get("component_abi"), "Turing component ABI")
    if component.get("schema") != COMPONENT_SCHEMA:
        raise ShaderArtifactError(f"component ABI schema must be {COMPONENT_SCHEMA!r}")
    component_language = _text(component.get("language"), "component ABI language").lower()
    if component_language != language:
        raise ShaderArtifactError("Turing API and component ABI languages disagree")
    stage = _text(metadata.get("stage"), "Turing API metadata.stage").lower()
    component_stage = _text(component.get("stage"), "component ABI stage").lower()
    if stage != component_stage or stage not in _STAGES:
        raise ShaderArtifactError("Turing API and component ABI stages disagree or are unsupported")
    entrypoint = _text(api.get("entry"), "Turing API entry")
    component_entrypoint = _text(component.get("entrypoint"), "component ABI entrypoint")
    if entrypoint != component_entrypoint:
        raise ShaderArtifactError("Turing API and component ABI entrypoints disagree")
    entries = api.get("entry_points")
    if not isinstance(entries, list) or not any(
        isinstance(entry, Mapping) and entry.get("symbol") == entrypoint for entry in entries
    ):
        raise ShaderArtifactError("Turing API entrypoint has no matching declared symbol")
    return ShaderIdentity(
        language=language,
        stage=stage,
        entrypoint=entrypoint,
        component_id=_text(component.get("component_id"), "component ABI component_id"),
    )


def _tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise ShaderArtifactError(f"required shader tool is unavailable: {name}")
    return executable


def _run(argv: Sequence[str], diagnostics: list[dict[str, Any]]) -> None:
    # A sequence is passed directly: no shell and no command-line interpolation.
    completed = subprocess.run(
        list(argv), check=False, capture_output=True, text=True, shell=False
    )
    diagnostics.append({
        "tool": Path(argv[0]).name,
        "argv": list(argv[1:]),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    })
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ShaderArtifactError(f"{Path(argv[0]).name} failed: {detail}")


def _compile_to_spirv(
    source: Path, output: Path, identity: ShaderIdentity,
    diagnostics: list[dict[str, Any]],
) -> str:
    if identity.language in _GLSL_LANGUAGES:
        environment = "opengl4.5" if identity.language in {"webgl2-glsl-es", "glsl-es"} else "vulkan1.2"
        mode = "-G" if environment == "opengl4.5" else "-V"
        argv = [
            _tool("glslangValidator"), mode, "--target-env", environment,
            "-S", _GLSL_STAGE[identity.stage], "-e", identity.entrypoint,
            str(source), "-o", str(output),
        ]
    else:
        environment = "vulkan1.2"
        argv = [
            _tool("dxc"), "-spirv", "-fspv-target-env=vulkan1.2",
            "-T", _HLSL_PROFILE[identity.stage], "-E", identity.entrypoint,
            "-Fo", str(output), str(source),
        ]
    _run(argv, diagnostics)
    if not output.is_file() or output.stat().st_size < 20:
        raise ShaderArtifactError("shader compiler did not produce a SPIR-V module")
    return environment


def _validate_spirv(spirv: Path, environment: str, diagnostics: list[dict[str, Any]]) -> None:
    _run([_tool("spirv-val"), "--target-env", environment, str(spirv)], diagnostics)


def _convert(spirv: Path, target: str, output: Path, diagnostics: list[dict[str, Any]]) -> None:
    argv = [_tool("spirv-cross"), str(spirv)]
    if target == "glsl":
        argv += ["--version", "450"]
    elif target == "essl":
        argv += ["--es", "--version", "310"]
    elif target == "hlsl":
        argv += ["--hlsl", "--shader-model", "50"]
    elif target == "msl":
        argv += ["--msl"]
    else:
        raise ShaderArtifactError(f"no SPIRV-Cross conversion exists for {target!r}")
    argv += ["--output", str(output)]
    _run(argv, diagnostics)
    if not output.is_file() or output.stat().st_size == 0:
        raise ShaderArtifactError(f"SPIRV-Cross did not produce {target} source")


def ingest_shader_artifact(manifest_path: str | Path, output_dir: str | Path) -> ShaderArtifactResult:
    """Ingest one strict manifest, retaining source/API and conversion receipts."""

    manifest = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    diagnostics: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "shader-artifact.receipt.json"
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA, "status": "failed",
        "manifest": str(manifest), "diagnostics": diagnostics,
    }
    try:
        source, api, targets = _load_manifest(manifest)
        identity = _load_identity(api)
        retained = output / "retained"
        intermediate = output / "intermediate"
        converted_dir = output / "converted"
        retained.mkdir(exist_ok=True)
        intermediate.mkdir(exist_ok=True)
        converted_dir.mkdir(exist_ok=True)
        retained_source = retained / ("authored" + (source.suffix or ".shader"))
        retained_api = retained / "component.api.yaml"
        shutil.copyfile(source, retained_source)
        shutil.copyfile(api, retained_api)
        spirv = intermediate / "component.spv"
        environment = _compile_to_spirv(retained_source, spirv, identity, diagnostics)
        _validate_spirv(spirv, environment, diagnostics)
        converted: dict[str, Path] = {}
        for target in targets:
            destination = converted_dir / ("component" + _SUFFIXES[target])
            _convert(spirv, target, destination, diagnostics)
            converted[target] = destination
        receipt.update({
            "status": "ok",
            "identity": {
                "component_id": identity.component_id, "language": identity.language,
                "stage": identity.stage, "entrypoint": identity.entrypoint,
            },
            "provenance": {
                "source": {"origin": str(source), "retained": str(retained_source), "sha256": _sha256(retained_source)},
                "api": {"origin": str(api), "retained": str(retained_api), "sha256": _sha256(retained_api)},
                "manifest": {"origin": str(manifest), "sha256": _sha256(manifest)},
            },
            "spirv": {"path": str(spirv), "sha256": _sha256(spirv), "target_env": environment},
            "converted": {
                target: {"path": str(path), "sha256": _sha256(path)}
                for target, path in converted.items()
            },
        })
        return ShaderArtifactResult(receipt_path, retained_source, retained_api, spirv, converted)
    except Exception as error:
        receipt["error"] = str(error)
        if not isinstance(error, ShaderArtifactError):
            receipt["error_type"] = type(error).__name__
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = ingest_shader_artifact(args.manifest, args.output)
    except ShaderArtifactError as error:
        print(f"shader artifact refused: {error}", file=sys.stderr)
        return 2
    print(result.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
