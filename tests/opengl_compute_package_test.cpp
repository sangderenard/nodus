#include "opengl_compute_mock.h"
#include <cstdlib>
#include <new>

namespace {bool fail_allocation=false;void fail_next_allocations(){fail_allocation=true;}}
void* operator new(std::size_t size) {
    if(fail_allocation)throw std::bad_alloc();
    if(void* p=std::malloc(size?size:1))return p;throw std::bad_alloc();
}
void* operator new[](std::size_t size){return ::operator new(size);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}

int main() {
    try {
        reset();
        const char first[]="#version 430\nfloat f(){return 1.;}\n";
        const char second[]="#version 430\nfloat f();void main(){f();}\n";
        gp_gl_compute_source objects[]={{"helper",first,sizeof(first)-1},{"main",second,sizeof(second)-1}};
        gp_gl_compute_program* combined=nullptr;
        expect(gp_gl_compute_link(&context,objects,2,&combined,error,sizeof(error)),error);
        expect(source_bytes==std::vector<std::string>{first,second},"source bytes unchanged");
        expect(count("compile")==2 && count("link")==1 && !dispatch_count,"two objects, one link, no dispatch");
        expect(count("detach")==2 && count("delete-shader")==2,"successful shader cleanup");
        expect(gp_gl_compute_release(&context,combined,error,sizeof(error)),error);

        reset(); compile_ok=false; combined=nullptr;
        expect(!gp_gl_compute_link(&context,objects,2,&combined,error,sizeof(error)),"compile failure");
        expect(!combined && count("link")==0 && count("delete-program")==1 && count("delete-shader")==1,"compile failure cleanup");
        expect(std::string(error).find("helper: bad")!=std::string::npos,"compile phase log");

        reset(); link_ok=false;
        expect(!gp_gl_compute_link(&context,objects,2,&combined,error,sizeof(error)),"link failure");
        expect(!combined && count("link")==1 && count("delete-shader")==2 && count("delete-program")==1,"link failure cleanup/no retry");
        expect(std::string(error).find("Combined program link: bad")!=std::string::npos,"link log");

        reset();compile_ok=false;fail_delete_program=true;
        expect(!gp_gl_compute_link(&context,objects,2,&combined,error,sizeof(error)) && combined,
            "failed compile retains owner when program cleanup fails");
        expect(programs.count(gp_gl_compute_program_name(combined)) && count("link")==0,
            "failed cleanup does not claim program release");
        fail_delete_program=false;current=88;
        expect(!gp_gl_compute_release(&context,combined,error,sizeof(error)),"failed-package cleanup still checks context");
        current=77;
        expect(gp_gl_compute_release(&context,combined,error,sizeof(error)) && programs.empty(),
            "failed package releases once on owning context without recompiling");combined=nullptr;

        reset();link_ok=false;fail_delete_shader=true;
        expect(!gp_gl_compute_link(&context,objects,2,&combined,error,sizeof(error)) && combined &&
            gp_gl_compute_program_name(combined)==0 && count("delete-program")==1,
            "shader-only cleanup owner survives after program is already released");
        fail_delete_shader=false;
        expect(gp_gl_compute_release(&context,combined,error,sizeof(error)) && count("delete-program")==1 &&
            count("link")==1,"partial cleanup retries remaining shaders only, never program/link");combined=nullptr;

        reset();combined=make_program();after_delete_program=&fail_next_allocations;
        const bool released_without_allocation=gp_gl_compute_release(&context,combined,error,sizeof(error))!=0;
        fail_allocation=false;after_delete_program=nullptr;combined=nullptr;
        expect(released_without_allocation && programs.empty(),"successful program deletion cannot allocate before clearing ownership");

        reset();after_delete_shader=&fail_next_allocations;
        const bool linked_without_allocation=gp_gl_compute_link(&context,objects,1,&combined,error,sizeof(error))!=0;
        fail_allocation=false;after_delete_shader=nullptr;
        expect(linked_without_allocation && combined,"successful shader deletion cannot allocate before clearing ownership");
        expect(gp_gl_compute_release(&context,combined,error,sizeof(error)),error);combined=nullptr;

        reset();combined=make_program();fail_delete_program=true;after_delete_program=&fail_next_allocations;
        const bool failure_contained=gp_gl_compute_release(&context,combined,error,sizeof(error))==0;
        fail_allocation=false;after_delete_program=nullptr;fail_delete_program=false;
        expect(failure_contained && error[0] && programs.count(gp_gl_compute_program_name(combined)),
            "allocation failure in error construction stays inside C ABI with owner retained");
        expect(gp_gl_compute_release(&context,combined,error,sizeof(error)),error);combined=nullptr;

        reset(); current=88;
        expect(!gp_gl_compute_link(&context,objects,2,&combined,error,sizeof(error)) && calls.empty(),"wrong context before GL");
        reset(); missing_proc="glCompileShader";
        expect(!gp_gl_compute_link(&context,objects,2,&combined,error,sizeof(error)) && calls.empty(),"missing procedure before GL");

        reset();auto a=make_program();auto b=make_program();
        I seed=1234,mode0=0,mode2=2;
        gp_gl_uniform ua[]={{"uSeed",GP_GL_UNIFORM_I32,1,1,&seed,0},{"uMode",GP_GL_UNIFORM_I32,1,1,&mode0,0}};
        gp_gl_uniform ub[]={{"uSeed",GP_GL_UNIFORM_I32,1,1,&seed,0},{"uMode",GP_GL_UNIFORM_I32,1,1,&mode2,0}};
        gp_gl_compute_stage stages[]={{"forward",a,{4,1,1},ua,2,0,0x2020,GP_GL_COMPLETE_INVOCATION},
                                     {"reactive",b,{2,1,1},ub,2,0x2000,0x2020,GP_GL_COMPLETE_INVOCATION}};
        gp_gl_storage_binding storage[]={{6,501,64,128}};
        gp_gl_image_binding images[]={{0,601,0,0,1,0x88BA,0x8236}};
        gp_gl_compute_schedule schedule{storage,1,images,1,stages,2};
        gp_gl_compute_receipt receipt{};
        calls.clear();
        expect(gp_gl_compute_submit(&context,&schedule,&receipt,error,sizeof(error)),error);
        expect(receipt.dispatch_calls==2 && receipt.stages_submitted==2 && !receipt.dispatch_may_have_run && receipt.failed_stage==SIZE_MAX,"submission receipt is not completion");
        expect(count("range-6-501-64-128")==3 && count("image-0-601")==3,"same persistent resources at every stage");
        std::vector<std::string> execution;
        for(const auto& c:calls)if(c.starts_with("dispatch-")||c.starts_with("barrier-"))execution.push_back(c);
        expect(execution==std::vector<std::string>{"dispatch-4-1-1","barrier-8224","barrier-8192","dispatch-2-1-1","barrier-8224"},"authored order and exact barriers");
        expect(count("uniform-1234")==4 && seed==1234,"uniform values copied without RNG regeneration");

        reset();stages[1].program=a;stages[0].uniform_count=0;
        expect(!gp_gl_compute_submit(&context,&schedule,&receipt,error,sizeof(error)) && !dispatch_count,
            "later same-program uniform cannot contaminate earlier missing set");
        stages[0].uniform_count=2;
        reset();
        expect(gp_gl_compute_submit(&context,&schedule,&receipt,error,sizeof(error)),error);
        expect(dispatch_count==2 && count("uniform-0")==2 && count("uniform-2")==2,
            "repeated program restores full scheduled uniform set");
        stages[1].program=b;

        reset();stages[1].boundary=GP_GL_UNVERIFIED_CONTINUATION;
        expect(!gp_gl_compute_submit(&context,&schedule,&receipt,error,sizeof(error)) && !dispatch_count,"reject unproven continuation before any dispatch");
        stages[1].boundary=GP_GL_COMPLETE_INVOCATION;
        reset();stages[1].groups[0]=65536;
        expect(!gp_gl_compute_submit(&context,&schedule,&receipt,error,sizeof(error)) && !dispatch_count,"later stage group failure before dispatch");
        stages[1].groups[0]=2;
        reset();missing_uniform=true;
        expect(!gp_gl_compute_submit(&context,&schedule,&receipt,error,sizeof(error)) && !dispatch_count,"required uniform refusal");
        reset();bad_uniform=true;
        expect(!gp_gl_compute_submit(&context,&schedule,&receipt,error,sizeof(error)) && !dispatch_count,"uniform GL error before dispatch");
        reset();storage[0].offset=65;
        expect(!gp_gl_compute_submit(&context,&schedule,&receipt,error,sizeof(error)) && !dispatch_count,"storage alignment refusal");
        storage[0].offset=64;
        reset();storage[0].bytes=4096;
        expect(!gp_gl_compute_submit(&context,&schedule,&receipt,error,sizeof(error)) && !dispatch_count,"storage extent refusal");
        storage[0].bytes=128;
        reset();fail_dispatch=2;
        expect(!gp_gl_compute_submit(&context,&schedule,&receipt,error,sizeof(error)),"second dispatch failure");
        expect(receipt.dispatch_calls==2 && receipt.stages_submitted==1 && receipt.failed_stage==1 && receipt.dispatch_may_have_run,"partial outcome retained without retry");
        expect(dispatch_count==2,"no failed-work replay");
        reset();current=88;
        expect(!gp_gl_compute_release(&context,a,error,sizeof(error)) && programs.count(gp_gl_compute_program_name(a)),"wrong context retains ownership");
        current=77;
        expect(gp_gl_compute_release(&context,a,error,sizeof(error)),error);
        expect(gp_gl_compute_release(&context,b,error,sizeof(error)),error);
        expect(programs.empty(),"all mock program ownership released");
        std::cout<<"PASS: native GL package/ordered submission, CPU call recording only; no device compilation or dispatch\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n';return 1; }
}
