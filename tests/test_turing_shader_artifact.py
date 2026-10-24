from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from turing_shader_artifact import (  # noqa: E402
    MANIFEST_SCHEMA,
    ShaderArtifactError,
    ingest_shader_artifact,
)


def _api(*, language="glsl-430", stage="compute", entry="main"):
    return {
        "schema": "turing-compiled-program-api-v1",
        "module": "fixture",
        "language": language,
        "entry": entry,
        "metadata": {
            "stage": stage,
            "component_abi": {
                "schema": "turing.shader-component.v1",
                "component_id": "fixture",
                "language": language,
                "stage": stage,
                "entrypoint": entry,
                "sentinels": {"word_layout": [], "words": []},
                "decorations": {},
                "ports": [],
            },
        },
        "entry_points": [{"name": entry, "symbol": entry, "kind": "numerical", "parameters": []}],
    }


def _fixture(tmp_path, *, targets=("glsl", "essl", "hlsl", "msl"), api=None, source=None):
    source_path = tmp_path / "shader.comp"
    source_path.write_text(source or """#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) buffer Values { float values[]; };
void main() { values[gl_GlobalInvocationID.x] += 1.0; }
""", encoding="utf-8")
    api_path = tmp_path / "shader.api.yaml"
    api_path.write_text(yaml.safe_dump(api or _api(), sort_keys=False), encoding="utf-8")
    manifest = tmp_path / "shader-artifact.json"
    manifest.write_text(json.dumps({
        "schema": MANIFEST_SCHEMA,
        "source": source_path.name,
        "api": api_path.name,
        "targets": list(targets),
    }), encoding="utf-8")
    return manifest, source_path, api_path


def test_real_glsl_spirv_cross_roundtrip_retains_provenance(tmp_path):
    required = ["glslangValidator", "spirv-val", "spirv-cross"]
    if any(shutil.which(tool) is None for tool in required):
        pytest.skip("Vulkan SDK shader tools are unavailable")
    artifact_root = tmp_path / "artifact space & literal"
    artifact_root.mkdir()
    manifest, source, api = _fixture(artifact_root)
    result = ingest_shader_artifact(manifest, tmp_path / "output space & literal")
    receipt = json.loads(result.receipt.read_text(encoding="utf-8"))

    assert receipt["status"] == "ok"
    assert receipt["identity"] == {
        "component_id": "fixture", "language": "glsl-430",
        "stage": "compute", "entrypoint": "main",
    }
    assert result.spirv.read_bytes()[:4] == b"\x03\x02#\x07"
    assert set(result.converted) == {"glsl", "essl", "hlsl", "msl"}
    assert all(path.stat().st_size for path in result.converted.values())
    assert result.retained_source.read_bytes() == source.read_bytes()
    assert result.retained_api.read_bytes() == api.read_bytes()
    assert receipt["provenance"]["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    expected = ["glslangvalidator.exe", "spirv-val.exe"] if sys.platform == "win32" else ["glslangvalidator", "spirv-val"]
    assert [step["tool"].lower() for step in receipt["diagnostics"]][:2] == expected


def test_real_hlsl_to_spirv_to_glsl_roundtrip_when_dxc_is_available(tmp_path):
    required = ["dxc", "spirv-val", "spirv-cross"]
    if any(shutil.which(tool) is None for tool in required):
        pytest.skip("DXC/SPIR-V tools are unavailable")
    manifest, _, _ = _fixture(
        tmp_path,
        targets=("glsl",),
        api=_api(language="hlsl"),
        source="""RWStructuredBuffer<float> values : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { values[id.x] += 1.0; }
""",
    )
    result = ingest_shader_artifact(manifest, tmp_path / "out")
    receipt = json.loads(result.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "ok"
    assert receipt["identity"]["language"] == "hlsl"
    assert result.spirv.read_bytes()[:4] == b"\x03\x02#\x07"
    assert "void main" in result.converted["glsl"].read_text(encoding="utf-8")


def test_shader_compiler_failure_is_preserved_in_diagnostic_receipt(tmp_path):
    if shutil.which("glslangValidator") is None:
        pytest.skip("glslangValidator is unavailable")
    manifest, _, _ = _fixture(tmp_path, targets=("glsl",), source="#version 450\nthis is not GLSL\n")
    output = tmp_path / "out"
    with pytest.raises(ShaderArtifactError, match="glslangValidator"):
        ingest_shader_artifact(manifest, output)
    receipt = json.loads((output / "shader-artifact.receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert len(receipt["diagnostics"]) == 1
    assert receipt["diagnostics"][0]["exit_code"] != 0
    assert receipt["diagnostics"][0]["stderr"] or receipt["diagnostics"][0]["stdout"]


@pytest.mark.parametrize("mutation, message", [
    (lambda manifest, api: manifest.update(schema="wrong"), "schema"),
    (lambda manifest, api: manifest.update(extra=True), "unknown fields"),
    (lambda manifest, api: manifest.update(source="../outside.comp"), "escapes"),
    (lambda manifest, api: api.update(schema="wrong"), "API schema"),
    (lambda manifest, api: api["metadata"].update(stage="fragment"), "stages disagree"),
    (lambda manifest, api: api["metadata"]["component_abi"].update(entrypoint="other"), "entrypoints disagree"),
])
def test_malformed_contracts_are_refused_with_failure_receipt(tmp_path, mutation, message):
    manifest_path, _, api_path = _fixture(tmp_path, targets=("glsl",))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    api = yaml.safe_load(api_path.read_text(encoding="utf-8"))
    mutation(manifest, api)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    api_path.write_text(yaml.safe_dump(api, sort_keys=False), encoding="utf-8")
    output = tmp_path / "out"
    with pytest.raises(ShaderArtifactError, match=message):
        ingest_shader_artifact(manifest_path, output)
    receipt = json.loads((output / "shader-artifact.receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert message.lower() in receipt["error"].lower()


def test_wgsl_source_is_explicitly_refused(tmp_path):
    manifest, _, _ = _fixture(tmp_path, api=_api(language="wgsl"), targets=("glsl",))
    with pytest.raises(ShaderArtifactError, match="WGSL source ingestion"):
        ingest_shader_artifact(manifest, tmp_path / "out")


def test_wgsl_target_is_explicitly_refused_before_tool_invocation(tmp_path):
    manifest, _, _ = _fixture(tmp_path, targets=("wgsl",))
    with pytest.raises(ShaderArtifactError, match="SPIRV-Cross has no WGSL backend"):
        ingest_shader_artifact(manifest, tmp_path / "out")
