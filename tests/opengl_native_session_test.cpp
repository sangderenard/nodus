#include "opengl_native_session.h"
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#define CALL __stdcall
#else
#define CALL
#endif
namespace {
using U=uint32_t;using I=int32_t;using P=intptr_t;
using namespace nodus::gpu;
U next_name=1,error_code=0;uintptr_t current=77;
size_t links=0,allocations=0,opens=0,closes=0,fail_link=0,fail_allocation=0;
bool fail_program_delete=false,fail_buffer_delete=false,fail_context_close=false,mismatch_device=false;
std::set<U> programs,shaders;std::map<U,std::vector<uint8_t>> buffers;std::map<U,U> bindings;
std::vector<std::string> source_copies;
void expect(bool x,const char* why){if(!x)throw std::runtime_error(why);}
U CALL CreateShader(U){auto n=next_name++;shaders.insert(n);return n;}
void CALL ShaderSource(U,I n,const char* const* s,const I* size){expect(n==1,"one source object");source_copies.emplace_back(*s,size_t(*size));}
void CALL CompileShader(U){}
void CALL GetShaderiv(U,U p,I* out){*out=p==0x8B81?1:4;}
void CALL GetShaderInfoLog(U,I,I* n,char* p){*n=3;std::memcpy(p,"bad",3);}
void CALL DeleteShader(U n){shaders.erase(n);}
U CALL CreateProgram(){auto n=next_name++;programs.insert(n);return n;}
void CALL AttachShader(U,U){}void CALL DetachShader(U,U){}
void CALL LinkProgram(U){++links;}
void CALL GetProgramiv(U,U p,I* out){*out=p==0x8B82?(links!=fail_link):4;}
void CALL GetProgramInfoLog(U,I,I* n,char* p){*n=3;std::memcpy(p,"bad",3);}
void CALL DeleteProgram(U n){if(fail_program_delete){error_code=1282;return;}programs.erase(n);}
U CALL GetError(){auto r=error_code;error_code=0;return r;}
void CALL GenBuffers(I n,U* out){for(I i=0;i<n;++i){out[i]=next_name++;buffers[out[i]]={};}}
void CALL DeleteBuffers(I n,const U* names){if(fail_buffer_delete){error_code=1282;return;}
    for(I i=0;i<n;++i)buffers.erase(names[i]);}
void CALL BindBuffer(U target,U name){bindings[target]=name;}
void CALL BufferData(U target,P bytes,const void*,U){if(++allocations==fail_allocation){error_code=1285;return;}
    buffers.at(bindings.at(target)).resize(size_t(bytes),0xcc);}
void CALL BufferSubData(U target,P offset,P bytes,const void* source){std::memcpy(buffers.at(bindings.at(target)).data()+offset,source,size_t(bytes));}
void CALL GetBufferSubData(U target,P offset,P bytes,void* out){std::memcpy(out,buffers.at(bindings.at(target)).data()+offset,size_t(bytes));}
void CALL GetBufferParameteri64v(U target,U,int64_t* out){*out=int64_t(buffers.at(bindings.at(target)).size());}
void CALL ClearBufferSubData(U target,U,P offset,P bytes,U,U,const void* value){
    for(P p=offset;p<offset+bytes;p+=4)std::memcpy(buffers.at(bindings.at(target)).data()+p,value,4);}
void CALL CopyBufferSubData(U a,U b,P offset,P target,P bytes){std::memcpy(buffers.at(bindings.at(b)).data()+target,buffers.at(bindings.at(a)).data()+offset,size_t(bytes));}
void* resolve(const char* name,void*){
#define PROC(n) if(std::strcmp(name,"gl" #n)==0)return reinterpret_cast<void*>(&n)
    PROC(CreateShader);PROC(ShaderSource);PROC(CompileShader);PROC(GetShaderiv);PROC(GetShaderInfoLog);PROC(DeleteShader);
    PROC(CreateProgram);PROC(AttachShader);PROC(DetachShader);PROC(LinkProgram);PROC(GetProgramiv);PROC(GetProgramInfoLog);PROC(DeleteProgram);
    PROC(GetError);PROC(GenBuffers);PROC(DeleteBuffers);PROC(BindBuffer);PROC(BufferData);PROC(BufferSubData);PROC(GetBufferSubData);
    PROC(GetBufferParameteri64v);PROC(ClearBufferSubData);PROC(CopyBufferSubData);
#undef PROC
    throw std::runtime_error("Unrequested procedure; execution procedures are absent from CPU fixture");
}
uintptr_t current_context(void*){return current;}
class Context final:public SessionContextOwner {
public:
    bool open(const SessionIdentity& i,SessionContextEvidence& out,std::string&)override{
        ++opens;gp_mem_backend_register_gl_context(71,77);
        out.context={71,&resolve,&current_context,nullptr};out.observed_luid=i.adapter_luid;
        if(mismatch_device)out.observed_luid[0]^=0xff;
        out.node_mask=i.node_mask;out.selection=ProtectedContextSelection::VerifiedDefaultDc;return true;
    }
    bool close(std::string& why)override{
        if(fail_context_close){why="CPU fixture context close failure";return false;}
        expect(programs.empty() && buffers.empty() && shaders.empty(),"context cannot close before native objects");
        ++closes;gp_mem_backend_unregister_gl_context(71);return true;
    }
};
SessionIdentity identity(){SessionIdentity i;i.host_id=7;i.contract_id=8;i.generation=9;i.session_id=10;
    i.adapter_luid[0]=1;i.node_mask=1;i.context_policy=ProtectedContextPolicy::VerifiedDefaultDc;return i;}
NativeSessionPlan plan(){NativeSessionPlan p;p.opted_in=true;p.storage_limits={1024,4096,8};
    p.programs={SessionProgram{1,GP_GL_UNVERIFIED_CONTINUATION,{{"caller-source-a","#version 430\nvoid main(){}\n"}}},
                SessionProgram{2,GP_GL_COMPLETE_INVOCATION,{{"caller-source-b","#version 430\nvoid main(){ }\n"}}}};
    SessionBuffer first;first.resource_id=3;first.binding=0;first.bytes=64;first.uploads={{4,{1,2,3,4}},{24,{5,6,7,8}}};
    SessionBuffer second;second.resource_id=4;second.binding=23;second.bytes=128;p.buffers={first,second};return p;}
std::unique_ptr<Context> context(){return std::make_unique<Context>();}
}
int main(){try{using namespace nodus::gpu;auto i=identity();auto p=plan();
    {NativeOpenGLSession session(i,context());expect(session.prepare(p),session.receipt().message);
        auto r=session.receipt();expect(r.state==uint64_t(SessionState::Retained) && r.retained_programs==2 && r.retained_buffers==2 &&
            r.retained_bytes==192 && !r.cleanup_verified && r.execution_held && !r.dispatch_calls,"truthful retained ownership");
        expect(source_copies==std::vector<std::string>{p.programs[0].objects[0].source,p.programs[1].objects[0].source},"exact original source bytes");
        gp_gl_storage_binding b{};expect(session.storage(i,3,&b) && b.binding==0,"exact borrowed process-local resource");
        const auto& bytes=buffers.at(b.buffer);expect(bytes[4]==1 && bytes[7]==4 && bytes[24]==5 && bytes[27]==8 && bytes[0]==0 && bytes[63]==0,"exact producer upload slices and zero gaps");
        expect(session.program(i,1)!=nullptr,"retained linked program remains owned");
        auto wrong=i;++wrong.generation;expect(!session.close(wrong) && session.program(wrong,1)==nullptr,"wrong-generation request cannot release/borrow");
        const auto prior_links=links;expect(!session.prepare(p) && links==prior_links,"duplicate preparation never relinks");
        current=88;expect(!session.close(i) && !session.program(i,1),"wrong context retains resources and refuses borrow");current=77;
        expect(session.close(i),session.receipt().message);expect(programs.empty() && buffers.empty(),"explicit close releases actual resources");
        expect(session.receipt().cleanup_verified && !session.receipt().dispatch_calls,"cleanup is not execution");}
    {NativeOpenGLSession session(i,context());auto bad=p;bad.opted_in=false;const auto n=opens;
        expect(!session.prepare(bad) && opens==n && session.receipt().cleanup_verified,"absent opt-in before context");}
    {NativeOpenGLSession session(i,context());expect(session.close(i,true),"cancel unopened session");const auto n=opens;
        expect(!session.prepare(p) && opens==n,"cancelled request cannot replay");}
    {NativeOpenGLSession session(i,context());mismatch_device=true;const auto n=links;
        expect(!session.prepare(p) && links==n && session.receipt().cleanup_verified,"device identity mismatch before shader work");mismatch_device=false;}
    {NativeOpenGLSession session(i,context());fail_link=links+2;const auto n=links;
        expect(!session.prepare(p) && links==n+2 && programs.empty() && buffers.empty(),"partial program failure closes prior programs");
        expect(!session.prepare(p) && links==n+2,"link failure is not replayed");fail_link=0;}
    {NativeOpenGLSession session(i,context());fail_allocation=allocations+2;
        expect(!session.prepare(p) && programs.empty() && buffers.empty() && session.receipt().cleanup_verified,"partial allocation/OOM cleanup");fail_allocation=0;}
    {NativeOpenGLSession session(i,context());expect(session.prepare(p),"prepare cleanup-failure fixture");
        fail_program_delete=true;const auto n=closes;expect(!session.close(i,true) && closes==n && !programs.empty(),"failed program deletion retains context");
        const auto prior=links;fail_program_delete=false;expect(session.close(i,true) && links==prior,"cleanup retry never relinks");}
    {NativeOpenGLSession session(i,context());expect(session.prepare(p),"prepare context-close-failure fixture");
        fail_context_close=true;expect(!session.close(i) && programs.empty() && buffers.empty(),"context-close failure retains only context owner");
        fail_context_close=false;expect(session.close(i),"retry context cleanup only");}
    {NativeOpenGLSession session(i,context());expect(session.prepare(p),"prepare lane fixture");bool rejected=false;
        std::thread worker([&]{rejected=!session.close(i,true);});worker.join();expect(rejected && !programs.empty(),"foreign lane cannot clean native resources");
        expect(session.close(i,true),"owner lane cleanup after rejection");}
    std::string error;auto bad=p;bad.buffers[1].binding=0;expect(!validate_session_plan(bad,error),"duplicate binding refused");
    bad=p;bad.buffers[0].uploads[1].offset=6;expect(!validate_session_plan(bad,error),"overlapping producer slices refused");
    bad=p;bad.buffers[0].bytes=UINT64_MAX;expect(!validate_session_plan(bad,error),"overflow allocation refused");
    expect(programs.empty() && buffers.empty() && shaders.empty(),"final no leaked native fixture resources");
    std::cout<<"PASS native persistent session ownership: exact resources, no replay, cancellation, partial failure, cleanup; CPU callbacks only, no hardware or dispatch\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
