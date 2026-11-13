#include "mem_backend_opengl.h"
#include "mem_backend_registry_internal.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#define GP_GL_CALL __stdcall
#else
#define GP_GL_CALL
#endif

namespace {
using U=uint32_t; using I=int32_t; using P=intptr_t;
constexpr U storage=0x90D2, dynamic_draw=0x88E8, size_query=0x8764,
    copy_read=0x8F36, copy_write=0x8F37, r32ui=0x8236,
    red_integer=0x8D94, unsigned_int=0x1405;
thread_local std::array<char,512> last_error{};
void error(const char* text) noexcept {
    if(!text)text="Unknown OpenGL backend failure";
    const auto n=std::min(last_error.size()-1,std::strlen(text));
    std::memcpy(last_error.data(),text,n);last_error[n]=0;
}
void require(bool condition,const char* text) { if(!condition)throw std::runtime_error(text); }
struct Api {
#define FN(ret,name,args) using name##T=ret(GP_GL_CALL*)args;name##T name=nullptr
    FN(void,GenBuffers,(I,U*)); FN(void,DeleteBuffers,(I,const U*));
    FN(void,BindBuffer,(U,U)); FN(void,BufferData,(U,P,const void*,U));
    FN(void,BufferSubData,(U,P,P,const void*)); FN(void,GetBufferSubData,(U,P,P,void*));
    FN(void,GetBufferParameteri64v,(U,U,int64_t*));
    FN(void,ClearBufferSubData,(U,U,P,P,U,U,const void*));
    FN(void,CopyBufferSubData,(U,U,P,P,P)); FN(U,GetError,());
#undef FN
};
struct State {
    gp_gl_compute_context context{}; uintptr_t native=0;
    std::thread::id lane; Api api; gp_mem_backend_opengl_limits limits{};
    size_t live_bytes=0; uint32_t live_buffers=0;
    // Reserved before creating any driver name: failed allocation cleanup can
    // never lose ownership because of a later host allocation failure.
    std::vector<U> pending;
};
struct Handle {
    uint32_t tag=0x4e474c42u; // Not the canonical CPU-handle magic.
    std::shared_ptr<State> state;
    U name=0; size_t bytes=0; bool root=false;
};
extern const gp_mem_backend_vtable_t vtable;
Handle& handle(gp_mem_backend_handle_t raw) {
    require(raw && gp_mem_backend_get_vtable(raw)==&vtable,"Not a live Nodus OpenGL backend handle");
    return *static_cast<Handle*>(raw);
}
void current(const State& s) {
    require(s.lane==std::this_thread::get_id(),"OpenGL storage used on another manager lane");
    require(gp_mem_backend_get_registered_gl_context(s.context.context_id)==s.native &&
            s.context.current_context(s.context.user)==s.native,
            "OpenGL storage owning registered context is not current");
}
void checked(const State& s,const char* phase) { require(!s.api.GetError(),phase); }
void range(const Handle& h,size_t offset,size_t bytes) {
    require(!h.root && h.name && offset<=h.bytes && bytes<=h.bytes-offset,
            "OpenGL storage view exceeds live allocation");
}
template<class F> F resolve(const gp_gl_compute_context& c,const char* name) {
    auto p=c.get_proc(name,c.user);auto address=reinterpret_cast<uintptr_t>(p);
    require(address>3 && address!=UINTPTR_MAX,"Required OpenGL storage procedure unavailable");
    return reinterpret_cast<F>(p);
}
Api load(const gp_gl_compute_context& c) {
    Api a;
#define LOAD(name) a.name=resolve<Api::name##T>(c,"gl" #name)
    LOAD(GenBuffers);LOAD(DeleteBuffers);LOAD(BindBuffer);LOAD(BufferData);
    LOAD(BufferSubData);LOAD(GetBufferSubData);LOAD(GetBufferParameteri64v);
    LOAD(ClearBufferSubData);LOAD(CopyBufferSubData);LOAD(GetError);
#undef LOAD
    return a;
}
template<class F> int attempt(F&& f) noexcept {
    error("");
    try {f();return 1;}catch(const std::exception& e){error(e.what());}
    catch(...){error("Unknown OpenGL backend failure");}return 0;
}
void delete_name(State& s,U& name) {
    if(!name)return;
    s.api.DeleteBuffers(1,&name);
    checked(s,"OpenGL buffer deletion not verified; ownership retained");
    name=0; // No allocation/error formatting between verified delete and record.
}
gp_mem_backend_handle_t allocate(gp_mem_backend_handle_t owner,size_t bytes,U alignment) noexcept {
    error("");std::shared_ptr<State> state;std::unique_ptr<Handle> result;U name=0;
    try {
        auto& parent=handle(owner);require(parent.root,"OpenGL allocation requires its context owner");
        state=parent.state;auto& s=*state;current(s);
        require(s.pending.empty(),"OpenGL allocation blocked by unresolved cleanup");
        require(bytes && bytes<=size_t(INTPTR_MAX) && bytes<=s.limits.max_buffer_bytes &&
                s.live_bytes<=s.limits.max_total_bytes && bytes<=s.limits.max_total_bytes-s.live_bytes &&
                s.live_buffers<s.limits.max_buffers,"OpenGL storage reservation exceeded");
        require(!alignment || (alignment<=4 && !(alignment&(alignment-1))),
                "Unsupported OpenGL host-pointer alignment request");
        require(bytes%4==0,"OpenGL storage extent must be uint32 aligned");
        result=std::make_unique<Handle>();result->state=state;result->bytes=bytes;
        checked(s,"Pending OpenGL error before allocation");
        s.api.GenBuffers(1,&name);checked(s,"OpenGL buffer-name allocation failed");
        require(name!=0,"OpenGL buffer-name allocation returned zero");
        s.api.BindBuffer(storage,name);checked(s,"OpenGL allocation binding failed");
        s.api.BufferData(storage,P(bytes),nullptr,dynamic_draw);
        checked(s,"OpenGL buffer storage allocation failed");
        int64_t actual=0;s.api.GetBufferParameteri64v(storage,size_query,&actual);
        checked(s,"OpenGL buffer extent verification failed");
        require(actual>0 && uint64_t(actual)==bytes,"OpenGL allocation extent mismatch");
        U zero=0;s.api.ClearBufferSubData(storage,r32ui,0,P(bytes),red_integer,unsigned_int,&zero);
        checked(s,"OpenGL initial storage clear failed");
        result->name=name;
        nodus::mem_backend_internal::register_external_handle(result.get(),GP_MEM_BACKEND_OPENGL,&vtable,
            bytes,GP_MEM_CAP_DEVICE_POINTER|GP_MEM_CAP_D2D_NATIVE);
        s.live_bytes+=bytes;++s.live_buffers;name=0;
        return result.release();
    }catch(const std::exception& e){error(e.what());}catch(...){error("Unknown OpenGL allocation failure");}
    if(result)nodus::mem_backend_internal::unregister_external_handle(result.get());
    if(name && state) {
        // All permitted failures happened on the owning lane. Drain the error
        // attributable to that failed operation before the single cleanup call.
        try {current(*state);state->api.GetError();delete_name(*state,name);}catch(...){}
        if(name)state->pending.push_back(name); // capacity reserved at creation.
    }
    return nullptr;
}
void free_handle(gp_mem_backend_handle_t h) { (void)gp_mem_backend_opengl_release(h); }
int write(gp_mem_backend_handle_t raw,size_t offset,const void* src,size_t bytes) {
    return attempt([&]{auto& h=handle(raw);range(h,offset,bytes);auto& s=*h.state;current(s);
        require(src || !bytes,"OpenGL upload source missing");if(!bytes)return;
        checked(s,"Pending OpenGL error before upload");s.api.BindBuffer(storage,h.name);
        checked(s,"OpenGL upload binding failed");
        s.api.BufferSubData(storage,P(offset),P(bytes),src);checked(s,"OpenGL upload failed; do not replay dependent work");});
}
int read(gp_mem_backend_handle_t raw,size_t offset,void* dst,size_t bytes) {
    return attempt([&]{auto& h=handle(raw);range(h,offset,bytes);auto& s=*h.state;current(s);
        require(dst || !bytes,"OpenGL readback destination missing");if(!bytes)return;
        checked(s,"Pending OpenGL error before readback");s.api.BindBuffer(storage,h.name);
        checked(s,"OpenGL readback binding failed");
        s.api.GetBufferSubData(storage,P(offset),P(bytes),dst);checked(s,"OpenGL readback failed");});
}
int copy(gp_mem_backend_handle_t source,gp_mem_backend_handle_t destination,size_t so,size_t d,size_t bytes) {
    return attempt([&]{auto& a=handle(source);auto& b=handle(destination);range(a,so,bytes);range(b,d,bytes);
        require(a.state==b.state,"OpenGL native copy requires the same process-local owner");
        auto& s=*a.state;current(s);
        require(source!=destination || !bytes || so+bytes<=d || d+bytes<=so,
                "Overlapping OpenGL buffer copy refused");if(!bytes)return;
        checked(s,"Pending OpenGL error before native copy");
        s.api.BindBuffer(copy_read,a.name);checked(s,"OpenGL copy source binding failed");
        s.api.BindBuffer(copy_write,b.name);checked(s,"OpenGL copy destination binding failed");
        s.api.CopyBufferSubData(copy_read,copy_write,P(so),P(d),P(bytes));
        checked(s,"OpenGL native copy failed; do not replay dependent work");});
}
uintptr_t native(gp_mem_backend_handle_t raw) {
    uintptr_t value=0;attempt([&]{auto& h=handle(raw);current(*h.state);value=h.name;});return value;
}
const gp_mem_backend_vtable_t vtable={nullptr,nullptr,&allocate,nullptr,&free_handle,&write,&read,&copy,
    &native,nullptr,nullptr,nullptr,nullptr};
}

extern "C" gp_mem_backend_handle_t gp_mem_backend_create_opengl(
    const gp_gl_compute_context* c,const gp_mem_backend_opengl_limits* limits) {
    std::unique_ptr<Handle> root;
    const auto ok=attempt([&]{
        require(c && c->context_id && c->get_proc && c->current_context && limits,
                "OpenGL storage context/limits missing");
        require(limits->max_buffer_bytes && limits->max_buffer_bytes<=size_t(INTPTR_MAX) &&
                limits->max_total_bytes>=limits->max_buffer_bytes && limits->max_buffers &&
                limits->max_buffers<=65536,"Invalid bounded OpenGL storage reservation");
        root=std::make_unique<Handle>();root->root=true;root->state=std::make_shared<State>();
        auto& s=*root->state;s.context=*c;s.native=gp_mem_backend_get_registered_gl_context(c->context_id);
        require(s.native,"OpenGL storage requires an already registered context");
        s.lane=std::this_thread::get_id();s.limits=*limits;current(s);
        s.pending.reserve(limits->max_buffers);s.api=load(*c);checked(s,"Pending OpenGL error before backend creation");
        nodus::mem_backend_internal::register_external_handle(root.get(),GP_MEM_BACKEND_OPENGL,&vtable,0,0);
    });
    if(!ok){if(root)nodus::mem_backend_internal::unregister_external_handle(root.get());return nullptr;}
    return root.release();
}
extern "C" int gp_mem_backend_opengl_release(gp_mem_backend_handle_t raw) {
    if(!raw)return 1;
    return attempt([&]{auto& h=handle(raw);auto& s=*h.state;current(s);
        if(h.root)require(!s.live_buffers && s.pending.empty(),"OpenGL context owner still has live/pending buffers");
        else {checked(s,"Pending OpenGL error before release; owner retained");delete_name(s,h.name);
            s.live_bytes-=h.bytes;--s.live_buffers;}
        nodus::mem_backend_internal::unregister_external_handle(raw);delete &h;
    });
}
extern "C" int gp_mem_backend_opengl_collect(gp_mem_backend_handle_t raw) {
    return attempt([&]{auto& h=handle(raw);require(h.root,"Cleanup collection requires context owner");
        auto& s=*h.state;current(s);checked(s,"Pending OpenGL error before cleanup collection");
        while(!s.pending.empty()){delete_name(s,s.pending.back());s.pending.pop_back();}});
}
extern "C" int gp_mem_backend_opengl_get_usage(gp_mem_backend_handle_t raw,gp_mem_backend_opengl_usage* usage) {
    return attempt([&]{auto& h=handle(raw);require(h.root && usage,"OpenGL usage requires owner and result");
        const auto& s=*h.state;require(s.lane==std::this_thread::get_id(),"OpenGL usage queried on another lane");
        *usage={s.live_bytes,s.live_buffers,uint32_t(s.pending.size())};});
}
extern "C" int gp_mem_backend_opengl_validate_owner(gp_mem_backend_handle_t buffer,
    gp_mem_backend_handle_t owner,const gp_gl_compute_context* context) {
    return attempt([&]{auto& b=handle(buffer);auto& root=handle(owner);
        require(!b.root && b.name && root.root && b.state==root.state && context,
                "OpenGL buffer does not belong to this exact native owner");
        const auto& c=root.state->context;
        require(c.context_id==context->context_id && c.get_proc==context->get_proc &&
                c.current_context==context->current_context && c.user==context->user,
                "OpenGL buffer context callback identity mismatch");
        current(*root.state);
    });
}
extern "C" const char* gp_mem_backend_opengl_last_error(void){return last_error.data();}
extern "C" int gp_mem_backend_opengl_clear_u32(gp_mem_backend_handle_t raw,size_t offset,size_t bytes,U value) {
    return attempt([&]{auto& h=handle(raw);range(h,offset,bytes);auto& s=*h.state;current(s);
        require(offset%4==0 && bytes%4==0,"OpenGL uint32 clear view is unaligned");if(!bytes)return;
        checked(s,"Pending OpenGL error before view clear");s.api.BindBuffer(storage,h.name);
        checked(s,"OpenGL clear binding failed");
        s.api.ClearBufferSubData(storage,r32ui,P(offset),P(bytes),red_integer,unsigned_int,&value);
        checked(s,"OpenGL view clear failed; do not replay dependent work");});
}
extern "C" int gp_mem_backend_opengl_storage_view(gp_mem_backend_handle_t raw,U binding,size_t offset,size_t bytes,
                                                  gp_gl_storage_binding* view) {
    return attempt([&]{auto& h=handle(raw);range(h,offset,bytes);current(*h.state);
        require(view && (bytes || !offset),"Invalid borrowed OpenGL binding");
        *view={binding,h.name,offset,bytes};});
}
