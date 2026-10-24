"""Strict adoption of a real Turing-published deployable SPIR-V product."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nodus_graph import SpirvProductError, load_spirv_product


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish(tmp_path: pathlib.Path) -> pathlib.Path:
    nodus_root = pathlib.Path(__file__).resolve().parent.parent
    turing_root = nodus_root.parent / "turing"
    product = tmp_path / "product"
    script = tmp_path / "publish.py"
    script.write_text(
        """from pathlib import Path
from src.transmogrifier.ssa import SSAValue, Instr, BasicBlock, Function
from src.compiler.ssa_spirv_backend import emit_deployable_compute_shader
from src.compiler.shader_deployment import publish_spirv_product
a = SSAValue(0, 'float32', ())
b = SSAValue(1, 'float32', ())
result = SSAValue(2, 'float32', ())
function = Function('add_float32', [a, b], {'entry': BasicBlock('entry', [
    Instr('add', [a, b], result), Instr('Ret', [], None),
])})
published = publish_spirv_product(
    Path(__import__('sys').argv[1]),
    emit_deployable_compute_shader(function, outputs=[result], name='turing_add'),
)
print(published.manifest_path)
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(turing_root) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(script), str(product)], cwd=turing_root,
        capture_output=True, text=True, env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    return product / "turing_add.spirv.json"


def _rehash(manifest_path: pathlib.Path) -> None:
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("assembly", "module", "api"):
        path = root / manifest[key]
        manifest["sha256"][path.name] = _sha(path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files = [root / manifest[key] for key in ("assembly", "module", "api")]
    files.append(manifest_path)
    (root / "SHA256SUMS").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in sorted(files)),
        encoding="utf-8",
    )


def test_real_turing_spirv_product_is_adopted_without_recompilation(tmp_path):
    product = load_spirv_product(_publish(tmp_path))
    assert product.component_id == "turing_add"
    assert product.entrypoint == "main"
    assert product.target_environment == "vulkan1.1"
    assert [port.role for port in product.ports] == ["feed", "feed", "output"]
    assert [(item.binding, item.access, item.dtype) for item in product.resources] == [
        (0, "read", "f32"), (1, "read", "f32"), (2, "read_write", "f32"),
    ]
    assert product.launch.local_size == (64, 1, 1)
    assert product.launch.control_binding == 3
    assert product.module_path.read_bytes()[:4] == b"\x03\x02#\x07"


def test_product_kind_hashes_and_physical_bindings_are_authoritative(tmp_path):
    original = _publish(tmp_path)

    wrong_kind = tmp_path / "wrong-kind"
    shutil.copytree(original.parent, wrong_kind)
    manifest_path = wrong_kind / original.name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["product_kind"] = "linkable-spirv-library"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SpirvProductError, match="not runnable"):
        load_spirv_product(manifest_path)

    tampered = tmp_path / "tampered"
    shutil.copytree(original.parent, tampered)
    manifest_path = tampered / original.name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    module = tampered / manifest["module"]
    data = bytearray(module.read_bytes())
    data[-1] ^= 1
    module.write_bytes(data)
    with pytest.raises(SpirvProductError, match="digest mismatch"):
        load_spirv_product(manifest_path)

    inconsistent = tmp_path / "inconsistent"
    shutil.copytree(original.parent, inconsistent)
    manifest_path = inconsistent / original.name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    api_path = inconsistent / manifest["api"]
    api = yaml.safe_load(api_path.read_text(encoding="utf-8"))
    api["metadata"]["component_abi"]["ports"][0]["backend_binding"] = 2
    api_path.write_text(yaml.safe_dump(api, sort_keys=False), encoding="utf-8")
    _rehash(manifest_path)
    with pytest.raises(SpirvProductError, match="binding/dtype disagrees"):
        load_spirv_product(manifest_path)
