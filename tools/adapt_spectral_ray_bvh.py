"""Derive a Vulkan storage-control variant of Spectral's T1 ray shader.

The transport algorithm remains the original OpenGL GLSL source. The eight
default-block integer uniforms become one std430 binding because Vulkan GLSL
does not permit standalone non-opaque uniforms. Source, binary, tool output,
and reflection are retained for the native Nodus shader-job row.
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
    "n_intents", "n_bands", "n_mats", "n_arenas", "n_tris",
    "wave_base_floats", "wave_stride", "wave_capacity",
)


def _run(*argv: str) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{Path(argv[0]).name} failed: {result.stderr or result.stdout}")
    return result.stdout


def build(source: Path, directory: Path) -> Path:
    source = source.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    original = source.read_text(encoding="utf-8")
    if original.count("#version 430 core") != 1:
        raise ValueError("expected one GLSL 430 version declaration")
    derived = relocate_uniforms(original,
        [UniformField(name,"int") for name in UNIFORMS],
        binding=8,block_name="RayRuntimeParams")
    derived = derived.replace("#version 430 core", "#version 450 core", 1)
    retained = directory / "ray_bvh_intersect.opengl.comp.glsl"
    glsl = directory / "ray_bvh_intersect.vulkan.comp"
    spirv = directory / "ray_bvh_intersect.vulkan.spv"
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
        raise ValueError("ray workgroup reflection differs from the source contract")
    bindings = {(row.get("set"), row.get("binding"))
                for row in reflection.get("ssbos", [])}
    if bindings != {(0, i) for i in range(9)} or len(reflection["ssbos"]) != 9:
        raise ValueError("ray storage bindings differ from the nine-binding contract")
    if reflection.get("ubos") or reflection.get("push_constants"):
        raise ValueError("ray shader contains an unaccounted control binding")
    params = next((row for row in reflection["ssbos"]
                   if row.get("binding") == 8), None)
    members = reflection.get("types", {}).get(
        params.get("type") if params else None, {}).get("members", [])
    if not params or params.get("block_size") != 32 or \
       [(m.get("name"),m.get("offset")) for m in members] != [
           (name,4*i) for i,name in enumerate(UNIFORMS)]:
        raise ValueError("T1 std430 control offsets differ")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema": "nodus.spectral-ray-bvh-vulkan.v1",
        "source": str(source), "original_sha256": digest(source),
        "retained_source": str(retained), "retained_sha256": digest(retained),
        "derived_source": str(glsl), "derived_sha256": digest(glsl),
        "spirv": str(spirv), "spirv_sha256": digest(spirv),
        "uniforms_moved_to_ssbo_binding_8": list(UNIFORMS),
        "reflection": reflection,
    }
    output = directory / "ray_bvh_intersect.vulkan.receipt.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(build(args.source, args.directory))
