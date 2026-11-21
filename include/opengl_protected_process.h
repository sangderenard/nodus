#pragma once
// The native row's one-request process boundary. This is not a
// scheduler. No GL handle, source file, or executable command crosses this API.
#include "kernel_opengl_runner.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace nodus::gpu {
struct ProtectedLinkRequest {
    uint64_t host_id=0, contract_id=0, generation=0;
    const gp_gl_compute_source* sources=nullptr;
    size_t source_count=0;
};
enum class ProtectedLinkStatus : uint32_t {
    Refused=0, Linked=1, LinkFailed=2, IdentityUnavailable=3,
    TimedOut=4, MemoryFloor=5, ChildFailed=6, InvalidReceipt=7,
    SupervisionFailed=8, CleanupUncertain=9, Running=10, Cancelled=11
};
enum class ProtectedContextPolicy : uint32_t { RequireNvAffinity=0, VerifiedDefaultDc=1 };
enum class ProtectedContextSelection : uint32_t { None=0, NvAffinity=1, VerifiedDefaultDc=2 };
struct ProtectedLinkOutcome {
    uint64_t host_id=0, contract_id=0, generation=0;
    ProtectedLinkStatus status=ProtectedLinkStatus::Refused;
    bool attempted=false, linked=false, resources_released=false;
    bool child_reaped=false, containment_verified=false, identity_verified=false;
    bool gl_cleanup_verified=false; // explicit GL release/context teardown, not inferred from process exit
    bool outcome_uncertain=false;
    ProtectedContextSelection context_selection=ProtectedContextSelection::None;
    uint32_t process_id=0, exit_code=0;
    uint64_t elapsed_ms=0, peak_process_bytes=0, peak_job_bytes=0;
    std::array<uint8_t,8> requested_luid{}, observed_luid{};
    std::string message;
};
class ProtectedLinkProvider {
public:
    virtual ~ProtectedLinkProvider()=default;
    // Structural admission can precede loan generation. It does not authorize
    // a link. The engine obtains the exact queued loan, then authorizes once.
    virtual bool available(std::string& reason) const {return ready(reason);}
    // Admission only: no process, context or driver invocation. Actual context
    // identity is established in the child, before the first shader call.
    virtual bool ready(std::string& reason) const=0;
    virtual bool authorized(const ProtectedLinkRequest&,std::string& reason) const {
        reason="Provider has no exact-request authorization";return false;
    }
    virtual ProtectedLinkOutcome link(const ProtectedLinkRequest& request)=0;
    // Bounded control-lane calls. A native safety owner watches only its one
    // child; it is not a queue, placement policy or tool scheduler.
    virtual bool poll(ProtectedLinkOutcome&) {return false;}
    virtual bool cancel() {return false;}
    // Only reclaim an already-started child; never resumes or relinks work.
    virtual bool poll_cleanup(ProtectedLinkOutcome&) {return false;}
};

using ProtectedSourceDigest=std::array<uint8_t,32>;
// SHA-256 over the exact labeled objects and their boundaries, not rewritten
// GLSL. Bounded to 16 MiB total and 4096 objects. No GL or process operation.
bool protected_source_digest(const ProtectedLinkRequest&,ProtectedSourceDigest&,std::string&);
struct ProtectedLinkAuthorization {
    uint64_t host_id=0, contract_id=0, generation=0;
    ProtectedSourceDigest source_digest{};
};
struct ProtectedProcessOptions {
    // Absolute path to the native nodus_opengl_link_worker, never a shell.
    std::wstring executable;
    std::array<uint8_t,8> adapter_luid{};
    uint32_t node_mask=0;
    ProtectedContextPolicy context_policy=ProtectedContextPolicy::RequireNvAffinity;
    uint64_t process_memory_bytes=4ull*1024*1024*1024;
    uint64_t job_memory_bytes=4ull*1024*1024*1024;
    uint64_t minimum_available_memory_bytes=8ull*1024*1024*1024;
    uint32_t wall_timeout_ms=90000;
    uint32_t poll_ms=25;
};
class ProtectedOpenGLProcess final : public ProtectedLinkProvider {
public:
    explicit ProtectedOpenGLProcess(ProtectedProcessOptions options);
    ~ProtectedOpenGLProcess();
    ProtectedOpenGLProcess(const ProtectedOpenGLProcess&)=delete;
    ProtectedOpenGLProcess& operator=(const ProtectedOpenGLProcess&)=delete;
    // The caller must obtain the user's explicit approval for this exact real
    // link before arming. One authorization, one child, consumed on submission;
    // any failure requires new approval. No automatic retries or device changes.
    bool authorize_once(const ProtectedLinkAuthorization&,std::string& reason);
    bool ready(std::string& reason) const override;
    bool available(std::string& reason) const override;
    bool authorized(const ProtectedLinkRequest&,std::string& reason) const override;
    ProtectedLinkOutcome link(const ProtectedLinkRequest&) override;
    bool poll(ProtectedLinkOutcome&) override;
    bool cancel() override;
    bool poll_cleanup(ProtectedLinkOutcome&) override;
private:
    struct Impl;
    ProtectedLinkOutcome run(const ProtectedLinkRequest&);
    std::unique_ptr<Impl> impl_;
};
} // namespace nodus::gpu
