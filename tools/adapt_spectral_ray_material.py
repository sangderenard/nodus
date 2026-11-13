"""Compile Spectral's authored T3 with explicit Vulkan control/image slots.

OpenGL gives SSBO and image units separate binding namespaces. T3 uses number
7 for both; Vulkan requires the image at its own descriptor binding. This
adapter maps that image to 9 and moves non-opaque defaults to SSBO 8. No
material, scattering, RNG, or sensor algorithm is rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from glsl_uniform_block import UniformField, relocate_uniforms


FIELDS=(
    UniformField("n_bands","int"),
    UniformField("n_mats","int"),
    UniformField("max_children","int"),
    UniformField("max_children_per_hit","int"),
    UniformField("sensor_res","int"),
    UniformField("sensor_pr","float"),
    UniformField("sensor_px","float"),
    UniformField("n_uv_groups","int"),
    UniformField("uv_group_id_base","int"),
    UniformField("uv_meta_base","int"),
    UniformField("rng_seed","uint"),
    UniformField("bdpt_max_verts","int"),
    UniformField("bdpt_max_spectral","int"),
    UniformField("bdpt_max_pdfs","int"),
    UniformField("bdpt_count_base","int"),
    UniformField("bdpt_spectral_base","int"),
    UniformField("bdpt_pdf_base","int"),
)


def _run(*args: str) -> str:
    result=subprocess.run(args,capture_output=True,text=True,check=False)
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} failed: {result.stderr or result.stdout}")
    return result.stdout


def build(source: Path, directory: Path) -> Path:
    source=source.resolve();directory.mkdir(parents=True,exist_ok=True)
    original=source.read_text(encoding="utf-8")
    if original.count("#version 430 core")!=1:
        raise ValueError("T3 version declaration changed")
    old="layout(r32ui, binding = 7) coherent volatile uniform uimage2DArray sensor_image;"
    new="layout(r32ui, set = 0, binding = 9) coherent volatile uniform uimage2DArray sensor_image;"
    if original.count(old)!=1:
        raise ValueError("T3 sensor image binding differs from source contract")
    derived=original.replace(old,new,1)
    derived=relocate_uniforms(derived,FIELDS,binding=8,
        block_name="RayMaterialRuntimeParams")
    derived=derived.replace("#version 430 core","#version 450 core",1)
    retained=directory/"ray_material.opengl.comp.glsl"
    glsl=directory/"ray_material.vulkan.comp"
    spirv=directory/"ray_material.vulkan.spv"
    retained.write_bytes(source.read_bytes())
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
    ssbos={(r.get("set"),r.get("binding")) for r in reflection.get("ssbos",[])}
    if ssbos!={(0,i) for i in range(9)} or len(reflection["ssbos"])!=9:
        raise ValueError("T3 storage descriptors differ")
    images=reflection.get("images",[])
    if len(images)!=1 or images[0].get("set")!=0 or images[0].get("binding")!=9:
        raise ValueError("T3 image descriptor differs")
    if reflection.get("ubos") or reflection.get("push_constants"):
        raise ValueError("T3 has an unaccounted control binding")
    control=next((r for r in reflection["ssbos"] if r.get("binding")==8),None)
    members=reflection.get("types",{}).get(control.get("type") if control else None,{}).get("members",[])
    if not control or [m.get("name") for m in members]!=[f.name for f in FIELDS] or \
       [m.get("offset") for m in members]!=list(range(0,4*len(FIELDS),4)) or \
       control.get("block_size")!=4*len(FIELDS):
        raise ValueError("T3 std430 control layout differs")
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt={"schema":"nodus.spectral-ray-material-vulkan.v1",
             "source":str(source),"original_sha256":digest(source),
             "retained_source":str(retained),"retained_sha256":digest(retained),
             "derived_source":str(glsl),"derived_sha256":digest(glsl),
             "spirv":str(spirv),"spirv_sha256":digest(spirv),
             "control_members":members,"image_binding":9,"reflection":reflection}
    path=directory/"ray_material.vulkan.receipt.json"
    path.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    return path


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    print(build(args.source,args.directory))
