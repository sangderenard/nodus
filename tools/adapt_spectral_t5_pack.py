"""Compile the authored Spectral BDPT pack/scatter stages for Nodus Vulkan.

Only default-block scalar uniforms are relocated to reflected storage blocks.
The sorting, packing, scattering, and PDF algorithms remain in their source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from glsl_uniform_block import UniformField, relocate_uniforms


STAGES = {
    "bdpt_sort_init": (
        4,
        ("nv", "npad", "base_idx"),
        ("int",) * 3,
    ),
    "bdpt_sort_step": (
        2,
        ("k", "j", "npad", "base_idx"),
        ("int",) * 4,
    ),
    "bdpt_pack_t5": (
        6,
        ("nv", "base_idx", "n_lv", "n_bands", "sensor_half_w", "sensor_half_h"),
        ("int", "int", "int", "int", "float", "float"),
    ),
    "bdpt_scatter_t5": (
        4,
        ("scatter_mode", "nv", "n_lv", "n_records", "record_offset",
         "n_bands", "bdpt_spectral_base", "bdpt_pdf_base", "bdpt_optical_base"),
        ("int",) * 9,
    ),
}


def _run(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name}: {result.stderr or result.stdout}")
    return result.stdout


def build(shader_dir: Path, output: Path) -> Path:
    shader_dir = shader_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("glslangValidator")
    validator = shutil.which("spirv-val")
    reflector = shutil.which("spirv-cross")
    if not all((compiler, validator, reflector)):
        raise RuntimeError("Vulkan shader compiler, validator, and reflector required")
    receipt = {"schema": "nodus.spectral-t5-pack-vulkan.v1", "stages": {}}
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, (control_binding, names, kinds) in STAGES.items():
        source_path = shader_dir / f"{name}.comp.glsl"
        source = source_path.read_text(encoding="utf-8")
        if source.count("#version 430 core") != 1:
            raise ValueError(f"{name}: version declaration changed")
        derived = relocate_uniforms(source,
            tuple(UniformField(field, kind) for field, kind in zip(names, kinds)),
            binding=control_binding, block_name=f"{name}_RuntimeParams")
        derived = derived.replace("#version 430 core", "#version 450 core", 1)
        glsl = output / f"{name}.vulkan.comp"
        spirv = output / f"{name}.vulkan.spv"
        glsl.write_text(derived, encoding="utf-8")
        _run(compiler, "-V", "--target-env", "vulkan1.2", "-S", "comp",
             str(glsl), "-o", str(spirv))
        _run(validator, "--target-env", "vulkan1.2", str(spirv))
        reflection = json.loads(_run(reflector, str(spirv), "--reflect"))
        rows = reflection.get("ssbos", [])
        expected = {(0, i) for i in range(control_binding + 1)}
        if {(r.get("set"), r.get("binding")) for r in rows} != expected or \
           len(rows) != len(expected) or reflection.get("ubos") or \
           reflection.get("images") or reflection.get("push_constants"):
            raise ValueError(f"{name}: descriptor set changed")
        control = next(r for r in rows if r.get("binding") == control_binding)
        members = reflection.get("types", {}).get(control["type"], {}).get("members", [])
        if control.get("block_size") != 4 * len(names) or \
           [(m.get("name"), m.get("offset")) for m in members] != \
           [(field, 4 * i) for i, field in enumerate(names)]:
            raise ValueError(f"{name}: control layout changed")
        receipt["stages"][name] = {
            "source": str(source_path), "source_sha256": digest(source_path),
            "derived": str(glsl), "derived_sha256": digest(glsl),
            "spirv": str(spirv), "spirv_sha256": digest(spirv),
            "reflection": reflection,
        }
    path = output / "t5_pack.vulkan.receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shader_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(build(args.shader_dir, args.output))
