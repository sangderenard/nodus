"""Compile the original desktop Phong raster pair for Vulkan review.

This is a source/interface artifact, not a graphics runner. Every removed
initializer is recorded in the receipt; a Vulkan host must populate the
reflected controls before drawing. Samplers receive distinct descriptors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from glsl_uniform_block import collect_uniforms, relocate_uniforms


def _run(*args: str) -> str:
    result=subprocess.run(args,capture_output=True,text=True,check=False)
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} failed: {result.stderr or result.stdout}")
    return result.stdout


def _opaque(glsl_type: str) -> bool:
    return "sampler" in glsl_type or "image" in glsl_type


def build(vertex: Path, fragment: Path, directory: Path) -> Path:
    vertex=vertex.resolve();fragment=fragment.resolve()
    directory.mkdir(parents=True,exist_ok=True)
    original_v=vertex.read_text(encoding="utf-8")
    original_f=fragment.read_text(encoding="utf-8")
    if original_v.count("#version 430 core")!=1 or \
       original_f.count("#version 430 core")!=1:
        raise ValueError("Phong desktop GLSL version declaration changed")
    vertex_decls=collect_uniforms(original_v)
    fragment_decls=collect_uniforms(original_f)
    if {d.field.name for d in vertex_decls}!={"uMVP","uMV"}:
        raise ValueError("Phong vertex controls changed")
    thermal={"uThermalEmission","uThermalGain","uThermalRefK","uThermalFloorK"}
    if not thermal.issubset({d.field.name for d in fragment_decls}):
        raise ValueError("Phong thermal control group changed")
    defaults:dict[str,str]={}
    derived_v=relocate_uniforms(original_v,[d.field for d in vertex_decls],
        binding=17,block_name="PhongVertexRuntimeParams",defaults_out=defaults)
    derived_f=relocate_uniforms(original_f,
        [d.field for d in fragment_decls if d.field.name in thermal],
        binding=18,block_name="PhongThermalRuntimeParams",
        allow_unlisted=True,defaults_out=defaults)
    general=[d.field for d in fragment_decls
             if d.field.name not in thermal and not _opaque(d.field.glsl_type)]
    derived_f=relocate_uniforms(derived_f,general,
        binding=19,block_name="PhongFragmentRuntimeParams",
        allow_unlisted=True,defaults_out=defaults)
    samplers=[d for d in fragment_decls if _opaque(d.field.glsl_type)]
    sampler_bindings={}
    for index,declaration in enumerate(samplers):
        field=declaration.field
        if declaration.initializer or field.array:
            raise ValueError("Phong opaque initializer/array requires a separate contract")
        pattern=re.compile(r"(?m)^\s*uniform\s+"+re.escape(field.glsl_type)+
            r"\s+"+re.escape(field.name)+r"\s*;\s*(?://[^\r\n]*)?$")
        if len(pattern.findall(derived_f))!=1:
            raise ValueError(f"Phong sampler declaration changed: {field.name}")
        binding=32+index
        derived_f=pattern.sub(
            f"layout(set = 0, binding = {binding}) uniform {field.glsl_type} {field.name};",
            derived_f,count=1)
        sampler_bindings[field.name]=binding
    if collect_uniforms(derived_f) or collect_uniforms(derived_v):
        raise ValueError("Phong translation left default-block uniforms")
    derived_v=derived_v.replace("#version 430 core","#version 450 core",1)
    derived_f=derived_f.replace("#version 430 core","#version 450 core",1)
    vpath=directory/"base_material.vulkan.vert"
    fpath=directory/"base_material.vulkan.frag"
    vspv=directory/"base_material.vulkan.vert.spv"
    fspv=directory/"base_material.vulkan.frag.spv"
    vpath.write_text(derived_v,encoding="utf-8")
    fpath.write_text(derived_f,encoding="utf-8")
    compiler=shutil.which("glslangValidator")
    validator=shutil.which("spirv-val")
    reflector=shutil.which("spirv-cross")
    if not all((compiler,validator,reflector)):
        raise RuntimeError("Vulkan SDK shader compiler, validator, and reflector are required")
    for stage,path,spirv in (("vert",vpath,vspv),("frag",fpath,fspv)):
        _run(compiler,"-V","--target-env","vulkan1.2",
             "--auto-map-locations","-S",stage,str(path),"-o",str(spirv))
        _run(validator,"--target-env","vulkan1.2",str(spirv))
    vertex_ref=json.loads(_run(reflector,str(vspv),"--reflect"))
    fragment_ref=json.loads(_run(reflector,str(fspv),"--reflect"))
    for binding,reflection in ((17,vertex_ref),(18,fragment_ref),(19,fragment_ref)):
        if not any(row.get("set")==0 and row.get("binding")==binding
                   for row in reflection.get("ssbos",[])):
            raise ValueError(f"Phong storage control binding {binding} absent")
    active_samplers={row.get("name"):row.get("binding")
                     for row in fragment_ref.get("textures",[])
                     if row.get("set")==0}
    # SPIR-V reflection omits samplers in compile-time-disabled branches.
    if any(sampler_bindings.get(name)!=binding
           for name,binding in active_samplers.items()):
        raise ValueError("Phong sampler reflection differs")
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt={"schema":"nodus.spectral-phong-vulkan-raster.v1",
        "status":"compiled-only-no-raster-runner",
        "sources":[{"path":str(p),"sha256":digest(p)} for p in (vertex,fragment)],
        "derived":[{"path":str(p),"sha256":digest(p)} for p in (vpath,fpath)],
        "spirv":[{"path":str(p),"sha256":digest(p)} for p in (vspv,fspv)],
        "sampler_bindings":sampler_bindings,
        "active_sampler_bindings":active_samplers,
        "removed_initializer_expressions":defaults,
        "host_requirement":"Pack reflected control blocks and original defaults before drawing; provide render targets and sampled images.",
        "vertex_reflection":vertex_ref,"fragment_reflection":fragment_ref}
    path=directory/"base_material.vulkan.receipt.json"
    path.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    return path


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vertex",type=Path)
    parser.add_argument("fragment",type=Path)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    print(build(args.vertex,args.fragment,args.directory))
