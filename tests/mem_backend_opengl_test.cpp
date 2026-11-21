#include "mem_backend_opengl.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define CALL __stdcall
#else
#define CALL
#endif
namespace {
using U=uint32_t;using I=int32_t;using P=intptr_t;
constexpr U storage=0x90D2,copy_read=0x8F36,copy_write=0x8F37;
std::map<U,std::vector<unsigned char>> buffers;
std::map<U,U> bindings;
U next=1,gl_error=0;uintptr_t native_context=77;
bool fail_allocate=false,fail_delete=false;U fail_bind_target=0;
size_t creations=0,deletions=0,clears=0,allocations=0,writes=0,reads=0,copies=0;
void expect(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void CALL GenBuffers(I count,U* names){for(I i=0;i<count;++i){names[i]=next++;buffers[names[i]]={};++creations;}}
void CALL DeleteBuffers(I count,const U* names){if(fail_delete){gl_error=0x0502;return;}
    for(I i=0;i<count;++i){buffers.erase(names[i]);++deletions;}}
void CALL BindBuffer(U target,U name){if(target==fail_bind_target){gl_error=0x0502;return;}bindings[target]=name;}
void CALL BufferData(U target,P bytes,const void*,U){++allocations;if(fail_allocate){gl_error=0x0505;return;}
    buffers.at(bindings.at(target)).resize(size_t(bytes),0xcc);}
void CALL BufferSubData(U target,P offset,P bytes,const void* p){
    ++writes;auto& b=buffers.at(bindings.at(target));std::memcpy(b.data()+offset,p,size_t(bytes));}
void CALL GetBufferSubData(U target,P offset,P bytes,void* p){
    ++reads;auto& b=buffers.at(bindings.at(target));std::memcpy(p,b.data()+offset,size_t(bytes));}
void CALL GetBufferParameteri64v(U target,U,int64_t* size){*size=int64_t(buffers.at(bindings.at(target)).size());}
void CALL ClearBufferSubData(U target,U,P offset,P bytes,U,U,const void* p){
    auto& b=buffers.at(bindings.at(target));for(P at=offset;at<offset+bytes;at+=4)std::memcpy(b.data()+at,p,4);++clears;}
void CALL CopyBufferSubData(U a,U b,P so,P d,P bytes){
    ++copies;std::memcpy(buffers.at(bindings.at(b)).data()+d,buffers.at(bindings.at(a)).data()+so,size_t(bytes));}
U CALL GetError(){U e=gl_error;gl_error=0;return e;}
uintptr_t current(void*){return native_context;}
void* proc(const char* name,void*){
#define RESOLVE(fn) if(std::string(name)=="gl" #fn)return reinterpret_cast<void*>(&fn)
    RESOLVE(GenBuffers);RESOLVE(DeleteBuffers);RESOLVE(BindBuffer);RESOLVE(BufferData);
    RESOLVE(BufferSubData);RESOLVE(GetBufferSubData);RESOLVE(GetBufferParameteri64v);
    RESOLVE(ClearBufferSubData);RESOLVE(CopyBufferSubData);RESOLVE(GetError);
#undef RESOLVE
    return nullptr;
}
void* no_procedures(const char*,void*){return nullptr;}
gp_gl_compute_context context{7,&proc,&current,nullptr};
gp_mem_backend_handle_t create(size_t bytes=65536){
    gp_mem_backend_opengl_limits limits{bytes,bytes*2,4};
    auto h=gp_mem_backend_create_opengl(&context,&limits);
    expect(h!=nullptr,gp_mem_backend_opengl_last_error());return h;
}
template<class F> void refuses(F&& f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}
    expect(failed,"Invalid auxiliary request was accepted");}
}
int main(){try{
    // Actual canonical registry and host release code, injected GL callbacks:
    // there is no graphics DLL, context creation, device query or dispatch.
    expect(gp_mem_backend_register_gl_context(7,77)!=0,"register CPU context token");
    gp_mem_backend_gl_params_t legacy{41,storage,0,7};
    expect(!gp_mem_backend_create_from_params(GP_MEM_BACKEND_OPENGL,&legacy,sizeof(legacy)),
           "unowned GLuint parameters must not create a pretend allocation");
    gp_mem_backend_opengl_limits limits{65536,131072,4};
    const auto before_missing=creations;
    auto missing=context;missing.get_proc=nullptr;
    expect(!gp_mem_backend_create_opengl(&missing,&limits),"missing procedure callback refused");
    missing=context;missing.current_context=nullptr;
    expect(!gp_mem_backend_create_opengl(&missing,&limits),"missing current-context callback refused");
    missing=context;missing.get_proc=&no_procedures;
    expect(!gp_mem_backend_create_opengl(&missing,&limits),"missing storage procedure refused");
    missing=context;missing.context_id=999;
    expect(!gp_mem_backend_create_opengl(&missing,&limits),"unregistered context refused");
    native_context=88;expect(!gp_mem_backend_create_opengl(&context,&limits),"wrong current context refused");native_context=77;
    expect(creations==before_missing,"no-hook callers cannot allocate anything");
    // No-hook CPU clients keep the canonical pre-existing backend behavior.
    gp_mem_backend_cpu_params_t cpu_params{16,4,0};
    auto cpu=gp_mem_backend_create_from_params(GP_MEM_BACKEND_CPU,&cpu_params,sizeof(cpu_params));
    expect(cpu && gp_mem_backend_size(cpu)==16 && gp_mem_backend_map(cpu),"legacy CPU creation without GL hooks");
    expect(gp_mem_backend_resize(cpu,32)!=0 && gp_mem_backend_size(cpu)==32,"legacy CPU resize unaffected");
    gp_mem_backend_unmap(cpu);gp_mem_backend_release(cpu);
    auto owner=create();auto* vt=gp_mem_backend_get_vtable(owner);
    expect(vt && vt->alloc && vt->free && !vt->map && !vt->resize,"canonical bounded native allocation vtable");
    auto a=vt->alloc(owner,64,4),b=vt->alloc(owner,64,4);expect(a && b,"allocate two buffers");
    expect(gp_mem_backend_opengl_validate_owner(a,owner,&context)!=0,"exact native owner proof");
    auto unrelated_owner=create();
    expect(!gp_mem_backend_opengl_validate_owner(a,unrelated_owner,&context),"same context is not same allocation owner");
    expect(gp_mem_backend_opengl_release(unrelated_owner)!=0,"release unused distinct owner");
    auto mismatched_callbacks=context;mismatched_callbacks.user=reinterpret_cast<void*>(1);
    expect(!gp_mem_backend_opengl_validate_owner(a,owner,&mismatched_callbacks),"caller cannot assert different callback identity");
    expect(gp_mem_backend_get_type(a)==GP_MEM_BACKEND_OPENGL && gp_mem_backend_size(a)==64,"canonical type/extent");
    expect(!gp_mem_backend_write_manifest("not-a-native-resource-export.txt",&a,1),
           "filesystem manifest cannot reinterpret GLuint as a string");
    expect(gp_mem_backend_get_capabilities(a)==(GP_MEM_CAP_DEVICE_POINTER|GP_MEM_CAP_D2D_NATIVE),"truthful capabilities");
    std::array<U,16> words{};expect(vt->copy_from_backend(a,0,words.data(),64)!=0,"read zero initialized allocation");
    expect(std::all_of(words.begin(),words.end(),[](U x){return x==0;}),"new buffer initialized");
    words.fill(9);expect(vt->copy_to_backend(a,0,words.data(),64)!=0,"upload through native vtable");
    expect(gp_mem_backend_opengl_clear_u32(a,16,16,0)!=0,"bounded view clear");
    expect(vt->copy_from_backend(a,0,words.data(),64)!=0,"read cleared view");
    for(size_t i=0;i<16;++i)expect(words[i]==(i>=4 && i<8?0u:9u),"adjacent views preserved");
    expect(gp_mem_backend_transfer(a,b,0,0,64)!=0,"canonical native D2D transfer");
    const auto prior_buffers=buffers;const auto prior_allocations=allocations,prior_writes=writes,
        prior_reads=reads,prior_clears=clears,prior_copies=copies;
    fail_bind_target=storage;
    expect(!vt->alloc(owner,256,4),"failed new-buffer bind refuses allocation");
    expect(!vt->copy_to_backend(a,0,words.data(),64),"failed bind refuses upload");
    expect(!vt->copy_from_backend(a,0,words.data(),64),"failed bind refuses readback");
    expect(!gp_mem_backend_opengl_clear_u32(a,0,64,7),"failed bind refuses clear");
    fail_bind_target=copy_read;expect(!gp_mem_backend_transfer(a,b,0,0,64),"failed source bind refuses copy");
    fail_bind_target=copy_write;expect(!gp_mem_backend_transfer(a,b,0,0,64),"failed destination bind refuses copy");
    fail_bind_target=0;
    expect(allocations==prior_allocations && writes==prior_writes && reads==prior_reads &&
        clears==prior_clears && copies==prior_copies && buffers==prior_buffers,
        "failed bindings cannot mutate previous allocations or issue dependent operations");
    expect(!gp_mem_backend_transfer(a,a,0,4,16),"overlap must refuse, without host fallback");
    expect(!vt->copy_to_backend(a,SIZE_MAX,words.data(),4),"overflow-safe range");
    expect(!gp_mem_backend_opengl_clear_u32(a,1,4,0),"clear alignment");
    expect(!gp_mem_backend_resize(a,128),"persistent extents not resized implicitly");
    gp_gl_storage_binding view{};expect(gp_mem_backend_opengl_storage_view(a,23,0,0,&view)!=0 &&
        view.buffer && view.binding==23 && !view.bytes,"borrow complete native binding");
    expect(!gp_mem_backend_opengl_storage_view(a,23,4,0,&view),"zero-byte offset cannot alias base");
    expect(!gp_mem_backend_opengl_release(owner),"root cannot outlive live resource ownership");
    native_context=88;size_t deleted=deletions;gp_mem_backend_release(a);
    expect(gp_mem_backend_get_vtable(a)==vt && gp_mem_backend_size(a)==64 && deletions==deleted,
           "void release preserves registry when context is wrong");native_context=77;
    bool wrong_lane=false;std::thread other([&]{wrong_lane=!gp_mem_backend_opengl_release(a);});other.join();
    expect(wrong_lane && gp_mem_backend_get_vtable(a)==vt,"wrong lane leaves live ownership");
    fail_delete=true;gp_mem_backend_release(a);expect(gp_mem_backend_get_vtable(a)==vt,"failed delete remains registered");
    fail_delete=false;expect(gp_mem_backend_opengl_release(a)!=0,"cleanup retry only");gp_mem_backend_release(b);
    gp_mem_backend_opengl_usage usage{};expect(gp_mem_backend_opengl_get_usage(owner,&usage)!=0 &&
        !usage.live_bytes && !usage.live_buffers,"released canonical accounting");
    fail_allocate=true;fail_delete=true;expect(!vt->alloc(owner,128,4),"allocation failure");
    expect(gp_mem_backend_opengl_get_usage(owner,&usage)!=0 && usage.pending_cleanup_buffers==1,"failed allocation name retained");
    size_t created=creations;expect(!vt->alloc(owner,128,4) && creations==created,"pending cleanup prevents another allocation");
    expect(!gp_mem_backend_opengl_release(owner),"pending resources protect context owner");
    fail_allocate=false;fail_delete=false;expect(gp_mem_backend_opengl_collect(owner)!=0,"collect exact retained name");
    auto oversize=vt->alloc(owner,65540,4);expect(!oversize && creations==created,"budget refusal precedes GL name creation");

    expect(gp_mem_backend_opengl_release(owner)!=0 && buffers.empty(),"all process-local resources released");
    expect(gp_mem_backend_unregister_gl_context(7)!=0,"unregister after release");
    std::cout<<"PASS: canonical GL backend and failure ownership; CPU callbacks only; zero graphics calls\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
