"""Compile Spectral's authored T5 connection shader with its native preamble.

The preamble order and binding defines mirror ray_tracer.cpp's
load_with_shadow_bvh call for t5_full_connect. Only the three OpenGL default
uniforms move to a Vulkan storage-control block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from glsl_uniform_block import UniformField, relocate_uniforms


def _run(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} failed: {result.stderr or result.stdout}")
    return result.stdout


def build(shader: Path, shadow: Path, material: Path, output: Path) -> Path:
    shader, shadow, material = (p.resolve() for p in (shader,shadow,material))
    output.mkdir(parents=True, exist_ok=True)
    source=shader.read_text(encoding="utf-8")
    shadow_source=shadow.read_text(encoding="utf-8")
    material_source=material.read_text(encoding="utf-8")
    if source.count("#version 430 core")!=1:
        raise ValueError("T5 version declaration changed")
    defines="""#define SHADOW_BVH_BINDING 5
#define SHADOW_TRI_ID_BINDING 6
#define SHADOW_TRI_FULL_BINDING 7
#define MAT_EVAL_MAX_BANDS 32
#define MAT_EVAL_BAND_STRIDE 12
#define MAT_EVAL_MAT_BINDING 8
"""
    derived=source.replace("#version 430 core",
        "#version 450 core\n"+defines+shadow_source+"\n"+
        material_source+"\n",1)
    derived=relocate_uniforms(derived,(
        UniformField("t5_profile_mode","int"),
        UniformField("t5_debug_enabled","int"),
        UniformField("t5_n_mats","int")),
        binding=10,block_name="T5RuntimeParams")
    glsl=output/"t5_full_connect.vulkan.comp"
    spirv=output/"t5_full_connect.vulkan.spv"
    glsl.write_text(derived,encoding="utf-8")
    compiler=shutil.which("glslangValidator")
    validator=shutil.which("spirv-val")
    reflector=shutil.which("spirv-cross")
    if not all((compiler,validator,reflector)):
        raise RuntimeError("Vulkan SDK shader compiler, validator, and reflector are required")
    _run(compiler,"-V","--target-env","vulkan1.2","-S","comp",
         str(glsl),"-o",str(spirv))
    _run(validator,"--target-env","vulkan1.2",str(spirv))
    reflection=json.loads(_run(reflector,str(spirv),"--reflect"))
    bindings={(row.get("set"),row.get("binding")) for row in reflection.get("ssbos",[])}
    if bindings!={(0,i) for i in range(11)} or len(reflection["ssbos"])!=11:
        raise ValueError("T5 descriptor set differs from native preamble contract")
    if reflection.get("ubos") or reflection.get("push_constants"):
        raise ValueError("T5 has an unaccounted control binding")
    control=next((row for row in reflection["ssbos"] if row.get("binding")==10),None)
    members=reflection.get("types",{}).get(control.get("type") if control else None,{}).get("members",[])
    if not control or control.get("block_size")!=12 or \
       [(m.get("name"),m.get("offset")) for m in members]!=[
           ("t5_profile_mode",0),("t5_debug_enabled",4),("t5_n_mats",8)]:
        raise ValueError("T5 std430 control offsets differ")
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    receipt={"schema":"nodus.spectral-t5-vulkan.v1",
        "sources":[{"path":str(p),"sha256":digest(p)} for p in (shader,shadow,material)],
        "derived_source":str(glsl),"derived_sha256":digest(glsl),
        "spirv":str(spirv),"spirv_sha256":digest(spirv),
        "reflection":reflection}
    path=output/"t5_full_connect.vulkan.receipt.json"
    path.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    return path


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shader",type=Path)
    parser.add_argument("shadow",type=Path)
    parser.add_argument("material",type=Path)
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    print(build(a.shader,a.shadow,a.material,a.output))
