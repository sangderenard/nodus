#pragma once
// Child-local persistent ownership for one protected canonical GPU-row loan.
// It has no queue, worker thread, dispatch API, or cross-process native handles.
#include "mem_backend_opengl.h"
#include "opengl_protected_process.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace nodus::gpu {
struct SessionIdentity {
    uint64_t host_id=0,contract_id=0,generation=0,session_id=0;
    std::array<uint8_t,8> adapter_luid{};
    uint32_t node_mask=0;
    ProtectedContextPolicy context_policy=ProtectedContextPolicy::RequireNvAffinity;
};
struct SessionSource {std::string label,source;};
struct SessionProgram {
    uint64_t resource_id=0;
    gp_gl_compute_boundary boundary=GP_GL_UNVERIFIED_CONTINUATION;
    std::vector<SessionSource> objects;
};
struct SessionUpload {uint64_t offset=0;std::vector<uint8_t> bytes;};
struct SessionBuffer {
    uint64_t resource_id=0,bytes=0;
    uint32_t binding=0;
    std::vector<SessionUpload> uploads; // Exact producer bytes; no reconstruction.
};
struct NativeSessionPlan {
    bool opted_in=false;
    gp_mem_backend_opengl_limits storage_limits{};
    std::vector<SessionProgram> programs;
    std::vector<SessionBuffer> buffers;
};
enum class SessionState:uint64_t { Empty=0,Preparing=1,Retained=2,Closing=3,Closed=4,Failed=5,CleanupPending=6 };
struct SessionReceipt {
    uint64_t version=1,host_id=0,contract_id=0,generation=0,session_id=0;
    uint64_t state=0,link_attempts=0,linked_programs=0,retained_programs=0,retained_buffers=0,retained_bytes=0;
    uint64_t context_opened=0,identity_verified=0,cleanup_verified=0,cancelled=0;
    uint64_t dispatch_calls=0,execution_held=1,failed_resource_id=0;
    uint64_t child_reaped=0,containment_verified=0,process_resources_released=0,outcome_uncertain=0;
    uint64_t process_id=0,exit_code=0,elapsed_ms=0,peak_process_bytes=0,peak_job_bytes=0;
    std::array<uint8_t,8> requested_luid{},observed_luid{};
    uint32_t observed_node_mask=0;
    ProtectedContextSelection context_selection=ProtectedContextSelection::None;
    char message[512]{};
};
static_assert(std::is_trivially_copyable_v<SessionReceipt>);

struct SessionContextEvidence {
    gp_gl_compute_context context{};
    std::array<uint8_t,8> observed_luid{};
    uint32_t node_mask=0;
    ProtectedContextSelection selection=ProtectedContextSelection::None;
};
// Implemented by the actual protected child's native context owner. The caller
// retains this object through every program/buffer cleanup failure. Tests inject
// CPU callbacks only; a CPU fixture is never a real-device identity receipt.
class SessionContextOwner {
public:
    virtual ~SessionContextOwner()=default;
    virtual bool open(const SessionIdentity&,SessionContextEvidence&,std::string&)=0;
    virtual bool close(std::string&)=0;
};

bool validate_session_plan(const NativeSessionPlan&,std::string&);
bool session_identity_equal(const SessionIdentity&,const SessionIdentity&);

class NativeOpenGLSession final {
public:
    NativeOpenGLSession(SessionIdentity,std::unique_ptr<SessionContextOwner>);
    ~NativeOpenGLSession();
    NativeOpenGLSession(const NativeOpenGLSession&)=delete;
    NativeOpenGLSession& operator=(const NativeOpenGLSession&)=delete;
    // At most one attempt. Failure may retain cleanup ownership, never relink.
    bool prepare(const NativeSessionPlan&);
    // Repeatable cleanup only. Context closes strictly after all resources.
    bool close(const SessionIdentity&,bool cancel=false);
    SessionReceipt receipt() const;
    // Only child-local consumers can borrow a retained resource. No wire/API
    // receipt contains any of these pointers or GLuints. No execution is exposed.
    bool storage(const SessionIdentity&,uint64_t resource_id,gp_gl_storage_binding*) const;
    gp_gl_compute_program* program(const SessionIdentity&,uint64_t resource_id) const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
