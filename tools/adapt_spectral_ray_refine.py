"""Adopt Spectral's exact-conic T2 GLSL for Vulkan without changing its law.

The original source remains authoritative. Only Vulkan's required control
binding differs: the six OpenGL uniforms become one std430 storage block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from glsl_uniform_block import UniformField, relocate_uniforms


UNIFORMS = (
    UniformField("n_tris","int"),
    UniformField("n_groups","int"),
    UniformField("n_bands","int"),
    UniformField("freq_hz","float","MAX_GPU_BANDS"),
    UniformField("bdpt_max_optical","int"),
    UniformField("bdpt_optical_base","int"),
)


def _run(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} failed: {result.stderr or result.stdout}")
    return result.stdout


def build(source: Path, directory: Path) -> Path:
    source = source.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    original = source.read_text(encoding="utf-8")
    if original.count("#version 430 core") != 1:
        raise ValueError("expected one GLSL 430 version declaration")
    derived = relocate_uniforms(original,UNIFORMS,binding=8,
        block_name="RayRefineRuntimeParams")
    derived = derived.replace("#version 430 core", "#version 450 core", 1)
    retained = directory / "ray_refine.opengl.comp.glsl"
    glsl = directory / "ray_refine.vulkan.comp"
    spirv = directory / "ray_refine.vulkan.spv"
    retained.write_bytes(source.read_bytes())
    glsl.write_text(derived, encoding="utf-8")
    compiler = shutil.which("glslangValidator")
    validator = shutil.which("spirv-val")
    reflector = shutil.which("spirv-cross")
    if not all((compiler, validator, reflector)):
        raise RuntimeError("Vulkan SDK shader compiler, validator, and reflector are required")
    _run(compiler, "-V", "--target-env", "vulkan1.2", "-S", "comp",
         str(glsl), "-o", str(spirv))
    _run(validator, "--target-env", "vulkan1.2", str(spirv))
    reflection = json.loads(_run(reflector, str(spirv), "--reflect"))
    entries = reflection.get("entryPoints", [])
    if len(entries) != 1 or entries[0].get("name") != "main" or \
       entries[0].get("mode") != "comp" or \
       entries[0].get("workgroup_size") != [64, 1, 1]:
        raise ValueError("T2 entry point or workgroup differs")
    bindings = {(row.get("set"), row.get("binding"))
                for row in reflection.get("ssbos", [])}
    if bindings != {(0, i) for i in range(9)} or len(reflection["ssbos"]) != 9:
        raise ValueError("T2 storage bindings differ from the nine-binding contract")
    if reflection.get("ubos") or reflection.get("push_constants"):
        raise ValueError("T2 has an unaccounted control binding")
    params = next((row for row in reflection["ssbos"]
                   if row.get("binding") == 8), None)
    typ = reflection.get("types", {}).get(params.get("type") if params else None, {})
    members = typ.get("members", [])
    if [m.get("name") for m in members] != [
        "n_tris", "n_groups", "n_bands", "freq_hz",
        "bdpt_max_optical", "bdpt_optical_base"]:
        raise ValueError("T2 control member order differs")
    if [m.get("offset") for m in members] != [0, 4, 8, 12, 140, 144] or \
       members[3].get("array_stride") != 4 or \
       params.get("block_size") != 148:
        raise ValueError("T2 std430 control offsets differ")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema": "nodus.spectral-ray-refine-vulkan.v1",
        "source": str(source), "original_sha256": digest(source),
        "retained_source": str(retained), "retained_sha256": digest(retained),
        "derived_source": str(glsl), "derived_sha256": digest(glsl),
        "spirv": str(spirv), "spirv_sha256": digest(spirv),
        "control_members": members, "reflection": reflection,
    }
    output = directory / "ray_refine.vulkan.receipt.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(build(args.source, args.directory))
