"""Adopt Spectral's sensor/aperture ray generator as a Vulkan artifact.

The original GPU kernel samples a thin aperture target and carries both
sensor-to-scene and reverse target modes. This adapter only relocates its
default-block controls; the optical assembly remains a separate T2 stage.
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
    UniformField("sensor_center","vec3"),
    UniformField("sensor_right","vec3"),
    UniformField("sensor_up","vec3"),
    UniformField("sensor_half_w","float"),
    UniformField("sensor_half_h","float"),
    UniformField("camera_target","vec3"),
    UniformField("aperture_right","vec3"),
    UniformField("aperture_up","vec3"),
    UniformField("aperture_radius","float"),
    UniformField("sensor_res","uint"),
    UniformField("target_mode","uint"),
    UniformField("n_bands","uint"),
    UniformField("max_bounces","uint"),
    UniformField("min_amplitude","float"),
    UniformField("exposure_weight","float"),
    UniformField("epoch_seed","uint"),
    UniformField("continuous_lut_enabled","uint"),
    UniformField("spectral_lut_knot_count","uint"),
    UniformField("spectral_lut_profile_count","uint"),
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
        raise ValueError("sensor raygen version declaration changed")
    derived=relocate_uniforms(original,FIELDS,binding=8,
        block_name="SensorRaygenRuntimeParams")
    derived=derived.replace("#version 430 core","#version 450 core",1)
    retained=directory/"sensor_mip_raygen.opengl.comp.glsl"
    glsl=directory/"sensor_mip_raygen.vulkan.comp"
    spirv=directory/"sensor_mip_raygen.vulkan.spv"
    retained.write_bytes(source.read_bytes());glsl.write_text(derived,encoding="utf-8")
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
        raise ValueError("sensor raygen storage binding contract differs")
    control=next((r for r in reflection["ssbos"] if r.get("binding")==8),None)
    members=reflection.get("types",{}).get(control.get("type") if control else None,{}).get("members",[])
    if not control or [m.get("name") for m in members]!=[f.name for f in FIELDS] or \
       reflection.get("ubos") or reflection.get("push_constants"):
        raise ValueError("sensor raygen controls differ")
    entries=reflection.get("entryPoints",[])
    if len(entries)!=1 or entries[0].get("workgroup_size")!=[64,1,1]:
        raise ValueError("sensor raygen workgroup differs")
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt={"schema":"nodus.spectral-sensor-raygen-vulkan.v1",
        "source":str(source),"original_sha256":digest(source),
        "retained_source":str(retained),"retained_sha256":digest(retained),
        "derived_source":str(glsl),"derived_sha256":digest(glsl),
        "spirv":str(spirv),"spirv_sha256":digest(spirv),
        "control_block_bytes":control["block_size"],
        "control_members":members,"reflection":reflection}
    path=directory/"sensor_mip_raygen.vulkan.receipt.json"
    path.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    return path


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    print(build(args.source,args.directory))
