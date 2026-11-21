#include "opengl_native_session.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <thread>

namespace nodus::gpu {
namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void message(SessionReceipt& r,const char* text) noexcept {
    const auto n=std::min(sizeof(r.message)-1,std::strlen(text));std::memcpy(r.message,text,n);r.message[n]=0;
}
bool valid_identity(const SessionIdentity& i){return i.host_id && i.contract_id && i.generation && i.session_id &&
    i.node_mask && std::any_of(i.adapter_luid.begin(),i.adapter_luid.end(),[](uint8_t b){return b!=0;}) &&
    (i.context_policy==ProtectedContextPolicy::RequireNvAffinity || i.context_policy==ProtectedContextPolicy::VerifiedDefaultDc);}
}
bool session_identity_equal(const SessionIdentity& a,const SessionIdentity& b){return
    a.host_id==b.host_id && a.contract_id==b.contract_id && a.generation==b.generation && a.session_id==b.session_id &&
    a.adapter_luid==b.adapter_luid && a.node_mask==b.node_mask && a.context_policy==b.context_policy;}
bool validate_session_plan(const NativeSessionPlan& p,std::string& error){
    try {
        require(p.opted_in,"Persistent native resources require explicit opt-in");
        require(!p.programs.empty() && p.programs.size()<=64 && p.buffers.size()<=256,"Native session resource-count limit");
        const auto& l=p.storage_limits;
        require(l.max_buffer_bytes && l.max_buffer_bytes<=512ull*1024*1024 &&
            l.max_total_bytes>=l.max_buffer_bytes && l.max_total_bytes<=512ull*1024*1024 &&
            l.max_buffers && l.max_buffers<=256 && p.buffers.size()<=l.max_buffers,"Native session storage caps invalid");
        size_t source_bytes=0,objects=0;uint64_t bytes=0,upload_bytes=0;
        std::set<uint64_t> identities;std::set<uint32_t> bindings;
        for(const auto& program:p.programs){
            require(program.resource_id && identities.insert(program.resource_id).second &&
                !program.objects.empty(),"Native session program identity/objects invalid");
            require(program.boundary==GP_GL_UNVERIFIED_CONTINUATION || program.boundary==GP_GL_COMPLETE_INVOCATION,
                "Native session boundary invalid");
            for(const auto& s:program.objects){
                require(++objects<=4096 && !s.source.empty() && s.label.find('\0')==std::string::npos,
                    "Native session source shape invalid");
                require(s.source.size()<=16*1024*1024-source_bytes,"Native session source bytes exceeded");source_bytes+=s.source.size();
                require(s.label.size()<=16*1024*1024-source_bytes,"Native session label bytes exceeded");source_bytes+=s.label.size();
            }
        }
        for(const auto& b:p.buffers){
            require(b.resource_id && identities.insert(b.resource_id).second && bindings.insert(b.binding).second && b.binding<65536,
                "Native session buffer/binding identity invalid");
            require(b.bytes && b.bytes%4==0 && b.bytes<=l.max_buffer_bytes && bytes<=l.max_total_bytes &&
                b.bytes<=l.max_total_bytes-bytes,"Native session allocation budget exceeded");bytes+=b.bytes;
            require(b.uploads.size()<=4096,"Native session upload slice limit");
            uint64_t previous_end=0;
            for(const auto& u:b.uploads){
                require(!u.bytes.empty() && u.offset>=previous_end && u.offset<=b.bytes && u.bytes.size()<=b.bytes-u.offset,
                    "Native session upload spans must be ordered, nonoverlapping and in bounds");
                require(u.bytes.size()<=64ull*1024*1024-upload_bytes,"Native session upload transport budget exceeded");
                upload_bytes+=u.bytes.size();previous_end=u.offset+u.bytes.size();
            }
        }
        error.clear();return true;
    }catch(const std::exception& e){error=e.what();return false;}
    catch(...){error="Native session preflight allocation failure";return false;}
}
struct NativeOpenGLSession::Impl {
    struct Program {uint64_t id=0;gp_gl_compute_program* handle=nullptr;};
    struct Buffer {uint64_t id=0;uint32_t binding=0;gp_mem_backend_handle_t handle=nullptr;};
    SessionIdentity identity;std::unique_ptr<SessionContextOwner> context_owner;
    SessionContextEvidence evidence{};gp_mem_backend_handle_t backend=nullptr;
    std::vector<Program> programs;std::vector<Buffer> buffers;
    std::thread::id lane=std::this_thread::get_id();SessionReceipt receipt{};
    bool attempted=false,context_may_exist=false,failed=false;
    bool current_lane()const{return lane==std::this_thread::get_id();}
    void update()noexcept{
        receipt.retained_programs=0;receipt.retained_buffers=0;receipt.retained_bytes=0;
        for(const auto& p:programs)if(p.handle)++receipt.retained_programs;
        for(const auto& b:buffers)if(b.handle){++receipt.retained_buffers;receipt.retained_bytes+=gp_mem_backend_size(b.handle);}
    }
    bool cleanup(bool cancel)noexcept {
        if(!current_lane()){message(receipt,"Native session cleanup on wrong lane; ownership retained");return false;}
        receipt.cancelled|=cancel?1:0;receipt.state=uint64_t(SessionState::Closing);bool ok=true;
        for(auto it=buffers.rbegin();it!=buffers.rend();++it)if(it->handle){
            if(gp_mem_backend_opengl_release(it->handle))it->handle=nullptr;
            else {message(receipt,gp_mem_backend_opengl_last_error());ok=false;}
        }
        for(auto it=programs.rbegin();it!=programs.rend();++it)if(it->handle){char error[512]{};
            if(gp_gl_compute_release(&evidence.context,it->handle,error,sizeof(error)))it->handle=nullptr;
            else {message(receipt,error);ok=false;}
        }
        if(backend){
            if(!gp_mem_backend_opengl_collect(backend)){message(receipt,gp_mem_backend_opengl_last_error());ok=false;}
            if(ok){if(gp_mem_backend_opengl_release(backend))backend=nullptr;
                else{message(receipt,gp_mem_backend_opengl_last_error());ok=false;}}
        }
        update();
        if(ok && context_may_exist){
            try {std::string error;if(context_owner->close(error))context_may_exist=false;
                else {message(receipt,error.c_str());ok=false;}}
            catch(...){message(receipt,"Native context cleanup threw; ownership retained");ok=false;}
        }
        receipt.cleanup_verified=ok?1:0;
        receipt.state=uint64_t(ok?(failed?SessionState::Failed:SessionState::Closed):SessionState::CleanupPending);
        return ok;
    }
};
NativeOpenGLSession::NativeOpenGLSession(SessionIdentity i,std::unique_ptr<SessionContextOwner> c):impl_(new Impl){
    require(valid_identity(i) && c,"Native session requires exact identity and context owner");
    impl_->identity=i;impl_->context_owner=std::move(c);auto& r=impl_->receipt;
    r.host_id=i.host_id;r.contract_id=i.contract_id;r.generation=i.generation;r.session_id=i.session_id;r.requested_luid=i.adapter_luid;
}
NativeOpenGLSession::~NativeOpenGLSession(){
    if(impl_->context_may_exist && !impl_->cleanup(true)){
        // Destroying the context owner here could tear down a context before
        // its retained GL objects. Keep the exact cleanup owner rather than
        // freeing on a foreign lane or claiming success. Protected child exit
        // remains the final containment mechanism, not explicit GL cleanup.
        std::fputs("Native OpenGL session destroyed with retained cleanup ownership\n",stderr);
        (void)impl_.release();
    }
}
bool NativeOpenGLSession::prepare(const NativeSessionPlan& plan){auto& s=*impl_;
    if(!s.current_lane() || s.attempted)return false;
    s.attempted=true;s.receipt.state=uint64_t(SessionState::Preparing);
    try {
        std::string error;require(validate_session_plan(plan,error),error.c_str());
        // All host vectors/slots exist before a context or driver object can
        // be acquired, so recording new cleanup ownership cannot allocate.
        s.programs.resize(plan.programs.size());s.buffers.resize(plan.buffers.size());
        std::vector<std::vector<gp_gl_compute_source>> sources(plan.programs.size());
        for(size_t i=0;i<plan.programs.size();++i){s.programs[i].id=plan.programs[i].resource_id;
            sources[i].reserve(plan.programs[i].objects.size());
            for(const auto& p:plan.programs[i].objects)sources[i].push_back({p.label.c_str(),p.source.data(),p.source.size()});}
        for(size_t i=0;i<plan.buffers.size();++i){s.buffers[i].id=plan.buffers[i].resource_id;s.buffers[i].binding=plan.buffers[i].binding;}
        s.context_may_exist=true;
        require(s.context_owner->open(s.identity,s.evidence,error),error.c_str());s.receipt.context_opened=1;
        s.receipt.observed_luid=s.evidence.observed_luid;s.receipt.observed_node_mask=s.evidence.node_mask;
        s.receipt.context_selection=s.evidence.selection;
        require(s.evidence.observed_luid==s.identity.adapter_luid && s.evidence.node_mask==s.identity.node_mask &&
            s.evidence.selection==(s.identity.context_policy==ProtectedContextPolicy::VerifiedDefaultDc?
                ProtectedContextSelection::VerifiedDefaultDc:ProtectedContextSelection::NvAffinity),
            "Native session context did not prove exact requested device identity");
        s.receipt.identity_verified=1;
        s.backend=gp_mem_backend_create_opengl(&s.evidence.context,&plan.storage_limits);
        require(s.backend,gp_mem_backend_opengl_last_error());
        for(size_t i=0;i<plan.programs.size();++i){char error[2048]{};
            s.receipt.failed_resource_id=plan.programs[i].resource_id;++s.receipt.link_attempts;
            require(gp_gl_compute_link(&s.evidence.context,sources[i].data(),sources[i].size(),&s.programs[i].handle,error,sizeof(error))!=0,error);
            ++s.receipt.linked_programs;
        }
        const auto* vt=gp_mem_backend_get_vtable(s.backend);
        for(size_t i=0;i<plan.buffers.size();++i){const auto& b=plan.buffers[i];auto& owned=s.buffers[i];s.receipt.failed_resource_id=b.resource_id;
            owned.handle=vt->alloc(s.backend,size_t(b.bytes),4);require(owned.handle,gp_mem_backend_opengl_last_error());
            for(const auto& u:b.uploads)require(vt->copy_to_backend(owned.handle,size_t(u.offset),u.bytes.data(),u.bytes.size())!=0,
                gp_mem_backend_opengl_last_error());
        }
        s.receipt.failed_resource_id=0;s.receipt.state=uint64_t(SessionState::Retained);s.update();return true;
    }catch(const std::exception& e){message(s.receipt,e.what());}catch(...){message(s.receipt,"Native session preparation failed; no replay");}
    s.failed=true;s.cleanup(false);return false;
}
bool NativeOpenGLSession::close(const SessionIdentity& identity,bool cancel){
    if(!session_identity_equal(identity,impl_->identity))return false;
    impl_->attempted=true; // Closing/cancelling an unopened session is terminal.
    return impl_->cleanup(cancel);
}
SessionReceipt NativeOpenGLSession::receipt()const{return impl_->receipt;}
bool NativeOpenGLSession::storage(const SessionIdentity& i,uint64_t id,gp_gl_storage_binding* out)const{
    const auto& s=*impl_;if(!session_identity_equal(i,s.identity) || s.receipt.state!=uint64_t(SessionState::Retained) || !s.current_lane())return false;
    for(const auto& b:s.buffers)if(b.id==id)return gp_mem_backend_opengl_validate_owner(b.handle,s.backend,&s.evidence.context) &&
        gp_mem_backend_opengl_storage_view(b.handle,b.binding,0,0,out);return false;
}
gp_gl_compute_program* NativeOpenGLSession::program(const SessionIdentity& i,uint64_t id)const{
    const auto& s=*impl_;if(!session_identity_equal(i,s.identity) || s.receipt.state!=uint64_t(SessionState::Retained) || !s.current_lane())return nullptr;
    const auto& c=s.evidence.context;
    if(!c.context_id || !c.current_context || gp_mem_backend_get_registered_gl_context(c.context_id)!=c.current_context(c.user))return nullptr;
    for(const auto& p:s.programs)if(p.id==id)return p.handle;return nullptr;
}
}
