#pragma once
// CPU-only call-recording tests. No GL library/context/device is loaded.
#include "kernel_opengl_runner.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define CALL __stdcall
#else
#define CALL
#endif
namespace {
using U=uint32_t; using I=int32_t; using B=uint8_t; using P=intptr_t;
std::vector<std::string> calls,source_bytes;
U next_name=10,error_code=0;
bool compile_ok=true,link_ok=true,missing_uniform=false,bad_uniform=false;
bool release_unavailable_after_link=false;
bool fail_delete_program=false,fail_delete_shader=false;
void (*after_delete_program)()=nullptr;
void (*after_delete_shader)()=nullptr;
int fail_dispatch=0,dispatch_count=0;
uintptr_t current=77;
std::string missing_proc;
std::map<U,bool> programs;
void expect(bool value,const char* label) { if(!value) throw std::runtime_error(label); }
void reset() {
    calls.clear(); source_bytes.clear(); error_code=0; compile_ok=link_ok=true;
    missing_uniform=bad_uniform=false;fail_dispatch=dispatch_count=0;current=77;missing_proc.clear();
    release_unavailable_after_link=false;
    fail_delete_program=fail_delete_shader=false;
    after_delete_program=after_delete_shader=nullptr;
}
U CALL CreateShader(U kind) { expect(kind==0x91B9,"compute shader kind");calls.push_back("shader");return next_name++; }
void CALL ShaderSource(U,I count,const char* const* p,const I* lengths) {
    expect(count==1,"one exact byte span per source object");
    source_bytes.emplace_back(*p,*lengths);
}
void CALL CompileShader(U) { calls.push_back("compile"); }
void CALL GetShaderiv(U,U p,I* out) { *out=p==0x8B81?compile_ok:4; }
void CALL GetShaderInfoLog(U,I,I* size,char* data) { *size=3;std::memcpy(data,"bad",3); }
void CALL DeleteShader(U) {
    calls.push_back("delete-shader");if(fail_delete_shader)error_code=1282;
    if(after_delete_shader)after_delete_shader();
}
U CALL CreateProgram() { const U id=next_name++;programs[id]=true;calls.push_back("program");return id; }
void CALL AttachShader(U,U) { calls.push_back("attach"); }
void CALL DetachShader(U,U) { calls.push_back("detach"); }
void CALL LinkProgram(U) { calls.push_back("link");if(release_unavailable_after_link)missing_proc="glDeleteProgram"; }
void CALL GetProgramiv(U,U p,I* out) { *out=p==0x8B82?link_ok:4; }
void CALL GetProgramInfoLog(U,I,I* size,char* data) { *size=3;std::memcpy(data,"bad",3); }
void CALL DeleteProgram(U p) {
    calls.push_back("delete-program");
    if(fail_delete_program)error_code=1282;else programs.erase(p);
    if(after_delete_program)after_delete_program();
}
void CALL UseProgram(U p) { calls.push_back("use-"+std::to_string(p)); }
B CALL IsProgram(U p) { return programs.count(p)?1:0; }
B CALL IsBuffer(U p) { return p==501; }
B CALL IsTexture(U p) { return p==601; }
U CALL GetError() { U result=error_code;error_code=0;return result; }
void CALL GetIntegerv(U p,I* out) { *out=p==0x90DF?16:32; }
void CALL GetIntegeri_v(U,U,I* out) { *out=65535; }
void CALL BindBuffer(U,U) {}
void CALL GetBufferParameteri64v(U,U,int64_t* out) { *out=4096; }
void CALL BindBufferBase(U,U slot,U value) { calls.push_back("buffer-"+std::to_string(slot)+"-"+std::to_string(value)); }
void CALL BindBufferRange(U,U slot,U value,P offset,P size) {
    calls.push_back("range-"+std::to_string(slot)+"-"+std::to_string(value)+"-"+std::to_string(offset)+"-"+std::to_string(size));
}
void CALL BindImageTexture(U unit,U name,I,B,I,U,U) { calls.push_back("image-"+std::to_string(unit)+"-"+std::to_string(name)); }
I CALL GetUniformLocation(U,const char*) { return missing_uniform?-1:7; }
void CALL Uniformiv(I,I,const I* value) { calls.push_back("uniform-"+std::to_string(*value));if(bad_uniform)error_code=1282; }
void CALL Uniformuiv(I,I,const U*) {}
void CALL Uniformfv(I,I,const float*) {}
void CALL UniformMatrixfv(I,I,B,const float*) {}
void CALL DispatchCompute(U x,U y,U z) {
    calls.push_back("dispatch-"+std::to_string(x)+"-"+std::to_string(y)+"-"+std::to_string(z));
    ++dispatch_count;if(dispatch_count==fail_dispatch)error_code=1282;
}
void CALL MemoryBarrier(U mask) { calls.push_back("barrier-"+std::to_string(mask)); }
void* resolve(const char* name,void*) {
    if(missing_proc==name)return nullptr;
#define PROC(n) if(std::strcmp(name,"gl" #n)==0)return reinterpret_cast<void*>(&n)
    PROC(CreateShader);PROC(ShaderSource);PROC(CompileShader);PROC(GetShaderiv);
    PROC(GetShaderInfoLog);PROC(DeleteShader);PROC(CreateProgram);PROC(AttachShader);
    PROC(DetachShader);PROC(LinkProgram);PROC(GetProgramiv);PROC(GetProgramInfoLog);
    PROC(DeleteProgram);PROC(UseProgram);PROC(IsProgram);PROC(IsBuffer);PROC(IsTexture);
    PROC(GetError);PROC(GetIntegerv);PROC(GetIntegeri_v);PROC(BindBuffer);
    PROC(GetBufferParameteri64v);PROC(BindBufferBase);PROC(BindBufferRange);
    PROC(BindImageTexture);PROC(GetUniformLocation);PROC(DispatchCompute);PROC(MemoryBarrier);
#undef PROC
    const std::string n=name;
    if(n.rfind("glUniformMatrix",0)==0)return reinterpret_cast<void*>(&UniformMatrixfv);
    if(n.rfind("glUniform",0)==0) {
        if(n.ends_with("uiv"))return reinterpret_cast<void*>(&Uniformuiv);
        if(n.ends_with("iv"))return reinterpret_cast<void*>(&Uniformiv);
        if(n.ends_with("fv"))return reinterpret_cast<void*>(&Uniformfv);
    }
    return nullptr;
}
uintptr_t current_context(void*) { return current; }
gp_gl_compute_context context{5,&resolve,&current_context,nullptr};
char error[2048];
gp_gl_compute_program* make_program() {
    const char source[]="#version 430\nvoid main(){}\n";
    gp_gl_compute_source unit{"whole",source,sizeof(source)-1};
    gp_gl_compute_program* program=nullptr;
    expect(gp_gl_compute_link(&context,&unit,1,&program,error,sizeof(error)),error);
    return program;
}
size_t count(const char* value) {size_t n=0;for(const auto& x:calls)if(x==value)++n;return n;}
}

// Link-time mock of the actual Nodus context registry, deliberately no device.
extern "C" uintptr_t gp_mem_backend_get_registered_gl_context(uintptr_t id) {return id==5?77:0;}
