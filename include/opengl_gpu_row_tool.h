#pragma once
// A native GPU HOST in the existing ContractNegotiator and an ordinary ITool
// control/telemetry row. No second queue, worker thread or placement scheduler.
// Contract.payload names engine-owned immutable source. The existing contract
// result FIFO carries Result, never a process-local GL handle.
#include "contract_negotiator.h"
#include "kernel_opengl_runner.h"
#include "opengl_protected_process.h"
#include "tool_api.h"
#include "tool_registry.h"

#include <memory>
#include <mutex>
#include <cstdio>
#include <thread>
#include <type_traits>
#include <unordered_map>

namespace nodus::gpu {

enum class Operation : uint64_t { LinkOnly=1, Execute=2 };
enum class State : uint64_t { Idle=0, Queued=1, Linking=2, Linked=3, Failed=4, Cancelled=5 };
enum class Refusal : uint64_t {
    None=0, WrongLane=1, ContextUnavailable=2, SupervisionUnavailable=3,
    ExecutionHeld=4, SourceUnavailable=5, SourceLimit=6, RetentionFull=7,
    NativeLinkFailed=8, CleanupPending=9, Busy=10, ContractOwnershipUnavailable=11,
    ProtectedLinkFailed=12, OutcomeUnknown=13
};
enum class Command : uint64_t { Query=0, Advance=1, CancelQueued=2, CancelActive=3 };

struct SourceObject { std::string label, source; };
struct Work {
    Operation operation=Operation::LinkOnly;
    std::vector<SourceObject> objects;
};
struct Control { uint64_t command=0, contract_id=0; };
struct Telemetry {
    uint64_t version=2, host_id=0, contract_id=0, generation=0;
    uint64_t state=0, refusal=0, harness_busy=0, context_current=0;
    uint64_t link_succeeded=0, program_released=0, result_fifo_accepted=0;
    uint64_t dispatch_calls=0, execution_complete=0;
    uint64_t physical_utilization_available=0; // not inferred from idle/VRAM
    uint64_t protected_process=0, containment_verified=0, identity_verified=0;
    uint64_t process_resources_released=0, gl_cleanup_verified=0, link_attempted=0;
    uint64_t child_reaped=0, outcome_uncertain=0, process_id=0, process_exit_code=0;
    uint64_t elapsed_ms=0, peak_process_bytes=0, peak_job_bytes=0;
    uint64_t context_selection=0; // protected provider's verified selection method
    std::array<uint8_t,8> requested_luid{},observed_luid{};
    char message[256]{};
};
static_assert(std::is_trivially_copyable_v<Control> && std::is_trivially_copyable_v<Telemetry>);

struct Config {
    uint64_t host_id=0;
    gp_gl_compute_context context{};
    size_t max_source_bytes=16*1024*1024, retained_outcomes=64;
    // Read only, no import/execute. The engine retains the same work through
    // FIFO publication and estimator commit; the row takes its own source copy.
    std::function<std::shared_ptr<const Work>(const contracts::Contract&)> lookup;
    // A native supervisor must verify isolation, process memory bound and wall
    // deadline before allowing the dangerous driver call. Default is REFUSED.
    // This is an admission requirement, not an implementation of supervision.
    // The existing Turing Python compile host does not satisfy this GL seam.
    std::function<bool(std::string&)> protected_link_permitted;
    // Preferred production boundary. The child owns every GL object/context;
    // this route NEVER invokes local context callbacks or local GL functions.
    std::shared_ptr<ProtectedLinkProvider> protected_provider;
};

class OpenGLGpuHost {
public:
    explicit OpenGLGpuHost(Config config):config_(std::move(config)),lane_(std::this_thread::get_id()) {
        if(!config_.host_id || !config_.lookup || !config_.max_source_bytes || !config_.retained_outcomes)
            throw std::invalid_argument("OpenGL GPU row configuration");
        telemetry_.host_id=config_.host_id;
    }
    ~OpenGLGpuHost() {
        // Correct lifecycle is shutdown()==true before releasing the bound
        // owner. If that contract is violated, never delete on a foreign GL
        // context and never claim cleanup succeeded.
        if(pending_cleanup_ && (!on_lane() || !cleanup()))
            std::fputs("Nodus OpenGL GPU row destroyed with owning-context cleanup pending; GL program not released\n",stderr);
        if(external_cleanup_pending_)
            std::fputs("Nodus OpenGL GPU row destroyed with protected child cleanup unresolved\n",stderr);
    }
    OpenGLGpuHost(const OpenGLGpuHost&)=delete;
    OpenGLGpuHost& operator=(const OpenGLGpuHost&)=delete;

    uint64_t id() const {return config_.host_id;}
    bool on_lane() const {std::lock_guard<std::recursive_mutex> guard(mutex_);return std::this_thread::get_id()==lane_;}
    // Control/telemetry can bind to the canonical manager lane even when no
    // native GL context has been installed. This confers NO GL readiness.
    bool claim_control_lane() {
        std::lock_guard<std::recursive_mutex> guard(mutex_);
        if(active_ || pending_cleanup_ || external_cleanup_pending_ || !retained_.empty())return false;
        lane_=std::this_thread::get_id();return true;
    }
    // Explicitly bind a quiescent row to an already-current registered native
    // context. Does not create, activate or move a context to another thread.
    bool claim_current_lane() {
        std::lock_guard<std::recursive_mutex> guard(mutex_);
        if(active_ || pending_cleanup_ || external_cleanup_pending_ || !retained_.empty() || !context_current())return false;
        lane_=std::this_thread::get_id();return true;
    }
    bool busy() const {std::lock_guard<std::recursive_mutex> guard(mutex_);return active_ || pending_cleanup_ || external_cleanup_pending_;}
    bool bind_contract_owner(contracts::ContractNegotiator& owner) {
        std::lock_guard<std::recursive_mutex> guard(mutex_);
        if(!on_lane() || busy() || (contract_owner_ && contract_owner_!=&owner))return false;
        contract_owner_=&owner;return true;
    }
    bool remembers(uint64_t id) const {std::lock_guard<std::recursive_mutex> guard(mutex_);return retained_.find(id)!=retained_.end() || (active_ && loan_.work.id==id);}
    Telemetry telemetry() const {std::lock_guard<std::recursive_mutex> guard(mutex_);return telemetry_;}
    bool snapshot(uint64_t id,Telemetry& out) const {
        std::lock_guard<std::recursive_mutex> guard(mutex_);
        if(!on_lane())return false;
        if(!id || id==telemetry_.contract_id){out=telemetry_;return true;}
        const auto found=retained_.find(id);if(found==retained_.end())return false;
        out=found->second;return true;
    }

    bool eligible(const contracts::Contract& contract) const {
        std::lock_guard<std::recursive_mutex> guard(mutex_);
        if(!on_lane() || !contract_owner_ || busy() || remembers(contract.id) || retained_.size()>=config_.retained_outcomes)
            return false;
        std::string reason;
        if(config_.protected_provider) {
            if(!config_.protected_provider->available(reason))return false;
        } else if(!context_current() || !config_.protected_link_permitted ||
            !config_.protected_link_permitted(reason)) return false;
        const auto work=config_.lookup(contract);
        return work && work->operation==Operation::LinkOnly && valid_source(*work);
    }

    // Called by the existing market row. It only accepts a source package into
    // this row's single in-flight slot or returns its retained result. It never
    // calls a GL compile, waits on a process, or dispatches a shader.
    contracts::Peer peer() {
        contracts::Peer peer;
        peer.id=config_.host_id;
        peer.exchange=[this](const contracts::Loan* offered) {
            std::lock_guard<std::recursive_mutex> guard(mutex_);
            contracts::Exchange reply;
            if(!on_lane())return reply;
            if(active_) {
                if(terminal() && !pending_cleanup_ && !external_cleanup_pending_) {reply.has_result=true;reply.result={loan_,static_cast<uint64_t>(telemetry_.state)};}
                return reply;
            }
            if(!offered || !eligible(offered->work))return reply;
            const auto source=config_.lookup(offered->work);
            if(!source || !valid_source(*source))return reply;
            work_=*source; loan_=*offered; active_=true;link_error_.clear();
            telemetry_={};telemetry_.host_id=id();telemetry_.contract_id=loan_.work.id;
            telemetry_.generation=loan_.generation;telemetry_.state=uint64_t(State::Queued);
            telemetry_.harness_busy=1;telemetry_.context_current=config_.protected_provider?0:1;
            telemetry_.protected_process=config_.protected_provider?1:0;
            reply.accepted_generation=loan_.generation;
            reply.hold_accepted_loan=bool(config_.protected_provider);
            return reply;
        };
        peer.receipt=[this](uint64_t generation,bool accepted) {
            std::lock_guard<std::recursive_mutex> guard(mutex_);
            if(!on_lane() || !active_ || generation!=loan_.generation)return;
            telemetry_.result_fifo_accepted=accepted?1:0;
            if(accepted && terminal() && !pending_cleanup_ && !external_cleanup_pending_) {
                // Retained until the engine explicitly confirms commit. This
                // prevents expired/rejected loans from replaying link work.
                telemetry_.harness_busy=0;retained_.emplace(loan_.work.id,telemetry_);
                active_=false;work_={};
            }
        };
        return peer;
    }

    // Same owning lane as the native context. The containing GPU row performs
    // this operation, not the market callback or an invented detached worker.
    bool control(const Control& command) {
        std::lock_guard<std::recursive_mutex> guard(mutex_);
        if(!on_lane())return false;
        telemetry_.context_current=config_.protected_provider?0:(context_current()?1:0);
        if(command.command==uint64_t(Command::Query)) {
            if(command.contract_id && command.contract_id!=telemetry_.contract_id)
                return retained_.find(command.contract_id)!=retained_.end();
            if(telemetry_.state==uint64_t(State::Idle)) {
                std::string reason;
                telemetry_.refusal=0;telemetry_.message[0]='\0';
                if(config_.protected_provider) {
                    telemetry_.protected_process=1;
                    if(!config_.protected_provider->ready(reason))
                        telemetry_.refusal=uint64_t(Refusal::SupervisionUnavailable);
                } else if(!telemetry_.context_current) {
                    telemetry_.refusal=uint64_t(Refusal::ContextUnavailable);
                    reason="Owning registered GL context is not current on this lane";
                } else if(!config_.protected_link_permitted || !config_.protected_link_permitted(reason)) {
                    telemetry_.refusal=uint64_t(Refusal::SupervisionUnavailable);
                    if(reason.empty())reason="Protected native GL process/context owner is not installed";
                }
                std::strncpy(telemetry_.message,reason.c_str(),sizeof(telemetry_.message)-1);
            }
            return true;
        }
        if(!active_ || (command.contract_id && command.contract_id!=loan_.work.id))return false;
        if(command.command==uint64_t(Command::CancelActive)) {
            if(!external_running_ || !config_.protected_provider)return false;
            if(config_.protected_provider->cancel())return true;
            // The child may have completed after the last row poll. A refused
            // cancel must still drain its final receipt and emit Telemetry;
            // restoring this Control would stall the typed downstream edge.
            return poll_protected_running();
        }
        if(command.command==uint64_t(Command::CancelQueued)) {
            if(telemetry_.state!=uint64_t(State::Queued))return false;
            telemetry_.state=uint64_t(State::Cancelled); return true;
        }
        if(command.command!=uint64_t(Command::Advance))return false;
        if(external_running_)return poll_protected_running();
        if(external_cleanup_pending_)return poll_protected_cleanup(); // never relink
        if(pending_cleanup_)return cleanup();
        if(telemetry_.state!=uint64_t(State::Queued))return false;
        if(!config_.protected_provider && !context_current()) {refuse(Refusal::ContextUnavailable,"Owning registered GL context is not current on this lane");return true;}
        std::string reason;
        if(!config_.protected_provider && (!config_.protected_link_permitted || !config_.protected_link_permitted(reason))) {
            refuse(Refusal::SupervisionUnavailable,reason.empty()?"Protected native GL process/context owner is not installed":reason);
            return true;
        }
        if(work_.operation!=Operation::LinkOnly) {refuse(Refusal::ExecutionHeld,"Shader execution is held");return true;}
        std::vector<gp_gl_compute_source> objects;
        for(const auto& object:work_.objects)objects.push_back({object.label.c_str(),object.source.data(),object.source.size()});
        if(config_.protected_provider && !config_.protected_provider->authorized(
            {id(),loan_.work.id,loan_.generation,objects.data(),objects.size()},reason)) {
            // Stay queued on the same held generation. Approval cannot be
            // inferred from admission, and waiting must not launch a process.
            telemetry_.refusal=uint64_t(Refusal::SupervisionUnavailable);
            if(reason.empty())reason="Exact queued native link requires explicit authorization";
            std::strncpy(telemetry_.message,reason.c_str(),sizeof(telemetry_.message)-1);
            telemetry_.message[sizeof(telemetry_.message)-1]='\0';return true;
        }
        if(!contract_owner_ || !contract_owner_->protect_loan(loan_.work.id,loan_.generation,loan_.host)) {
            refuse(Refusal::ContractOwnershipUnavailable,"Exact native contract loan is no longer owned; no link was attempted");
            return true;
        }
        telemetry_.state=uint64_t(State::Linking);
        if(config_.protected_provider)return protected_link(objects);
        char error[2048]{};
        telemetry_.link_attempted=1;
        if(!gp_gl_compute_link(&config_.context,objects.data(),objects.size(),&pending_cleanup_,error,sizeof(error))) {
            link_error_=error;
            telemetry_.program_released=telemetry_.gl_cleanup_verified=pending_cleanup_?0:1;
            refuse(Refusal::NativeLinkFailed,error);return true;
        }
        telemetry_.link_succeeded=1;
        return cleanup();
    }

    // Only forget after the existing contract system has committed this ID.
    // No user control command clears replay protection or in-flight state.
    bool forget_committed(uint64_t id,const contracts::ContractNegotiator& negotiator) {
        std::lock_guard<std::recursive_mutex> guard(mutex_);
        if(!on_lane() || external_cleanup_pending_)return false;
        for(const auto& entry:negotiator.contracts())if(entry.work.id==id &&
            entry.state==contracts::ContractNegotiator::State::Committed)return retained_.erase(id)>0;
        return false;
    }
    bool shutdown() {
        std::lock_guard<std::recursive_mutex> guard(mutex_);
        if(!on_lane())return false;
        if(external_running_) {
            try {config_.protected_provider->cancel();}catch(...) {return false;}
            if(!poll_protected_running() || external_running_)return false;
        }
        if(external_cleanup_pending_ && !poll_protected_cleanup())return false;
        if(pending_cleanup_ && !cleanup())return false;
        if(active_ && telemetry_.state==uint64_t(State::Queued))telemetry_.state=uint64_t(State::Cancelled);
        return !pending_cleanup_ && !external_cleanup_pending_;
    }

private:
    bool protected_link(const std::vector<gp_gl_compute_source>& objects) {
        external_cleanup_pending_=true; // fail closed even if the provider throws
        try {
            const ProtectedLinkRequest request{id(),loan_.work.id,loan_.generation,objects.data(),objects.size()};
            apply_protected_outcome(config_.protected_provider->link(request));
        } catch(...) {
            telemetry_.outcome_uncertain=1;
            refuse(Refusal::OutcomeUnknown,"Protected provider threw; submission/child ownership is unresolved, no retry");
        }
        return true;
    }
    void apply_protected_outcome(const ProtectedLinkOutcome& outcome) {
            if(outcome.host_id!=id() || outcome.contract_id!=loan_.work.id || outcome.generation!=loan_.generation) {
                telemetry_.outcome_uncertain=1;
                refuse(Refusal::OutcomeUnknown,"Protected receipt identity does not match its pinned native loan");
                return;
            }
            telemetry_.containment_verified=outcome.containment_verified;
            telemetry_.identity_verified=outcome.identity_verified;
            telemetry_.child_reaped=outcome.child_reaped;
            telemetry_.process_id=outcome.process_id;telemetry_.process_exit_code=outcome.exit_code;
            telemetry_.elapsed_ms=outcome.elapsed_ms;telemetry_.peak_process_bytes=outcome.peak_process_bytes;
            telemetry_.peak_job_bytes=outcome.peak_job_bytes;
            telemetry_.context_selection=static_cast<uint64_t>(outcome.context_selection);
            telemetry_.requested_luid=outcome.requested_luid;telemetry_.observed_luid=outcome.observed_luid;
            telemetry_.process_resources_released=outcome.resources_released;
            telemetry_.gl_cleanup_verified=outcome.gl_cleanup_verified;
            telemetry_.program_released=outcome.gl_cleanup_verified;
            telemetry_.link_attempted=outcome.attempted;
            if(outcome.status==ProtectedLinkStatus::Running) {
                external_running_=true;external_cleanup_pending_=true;
                telemetry_.state=uint64_t(State::Linking);telemetry_.refusal=0;
                telemetry_.outcome_uncertain=outcome.outcome_uncertain;
                std::strncpy(telemetry_.message,"Protected native link is running",sizeof(telemetry_.message)-1);
                return;
            }
            external_running_=false;
            external_cleanup_pending_=!outcome.child_reaped || !outcome.resources_released;
            telemetry_.outcome_uncertain=outcome.outcome_uncertain || external_cleanup_pending_;
            const bool adapter_matches=outcome.requested_luid==outcome.observed_luid &&
                std::any_of(outcome.requested_luid.begin(),outcome.requested_luid.end(),[](uint8_t byte){return byte!=0;});
            if(outcome.status==ProtectedLinkStatus::Linked && outcome.attempted && outcome.linked && outcome.resources_released &&
               outcome.child_reaped && outcome.gl_cleanup_verified && outcome.containment_verified &&
               outcome.identity_verified && adapter_matches && !telemetry_.outcome_uncertain) {
                telemetry_.link_succeeded=1;telemetry_.state=uint64_t(State::Linked);
                telemetry_.refusal=0;telemetry_.message[0]='\0';
            } else refuse(telemetry_.outcome_uncertain?Refusal::OutcomeUnknown:Refusal::ProtectedLinkFailed,
                outcome.message.empty()?"Protected native link did not prove a released, verified success":outcome.message);
    }
    bool poll_protected_running() {
        if(!config_.protected_provider)return false;
        try {
            ProtectedLinkOutcome outcome;
            if(!config_.protected_provider->poll(outcome))return true;
            apply_protected_outcome(outcome);return true;
        } catch(...) {
            external_running_=false;telemetry_.outcome_uncertain=1;
            refuse(Refusal::OutcomeUnknown,"Protected polling failed; child ownership remains unresolved, no retry");return true;
        }
    }
    bool poll_protected_cleanup() {
        if(!config_.protected_provider)return false;
        try {
            ProtectedLinkOutcome outcome;
            if(!config_.protected_provider->poll_cleanup(outcome))return false;
            apply_protected_outcome(outcome);
            return !external_cleanup_pending_;
        } catch(...) {
            telemetry_.outcome_uncertain=1;
            refuse(Refusal::OutcomeUnknown,"Protected cleanup ownership remains unresolved; no retry");return false;
        }
    }
    bool context_current() const {
        const auto& c=config_.context;
        const auto registered=gp_mem_backend_get_registered_gl_context(c.context_id);
        return c.context_id && registered && c.current_context && c.get_proc && c.current_context(c.user)==registered;
    }
    bool valid_source(const Work& work) const {
        if(work.objects.empty() || work.objects.size()>4096)return false;
        size_t bytes=0;
        for(const auto& object:work.objects) {
            if(object.source.empty() || object.source.size()>config_.max_source_bytes-bytes)return false;
            bytes+=object.source.size();
            if(object.label.size()>config_.max_source_bytes-bytes)return false;
            bytes+=object.label.size();
        }
        return true;
    }
    bool terminal() const {
        return telemetry_.state==uint64_t(State::Linked) || telemetry_.state==uint64_t(State::Failed) ||
               telemetry_.state==uint64_t(State::Cancelled);
    }
    void refuse(Refusal code,const std::string& message) {
        telemetry_.state=uint64_t(State::Failed);telemetry_.refusal=uint64_t(code);
        std::strncpy(telemetry_.message,message.c_str(),sizeof(telemetry_.message)-1);
        telemetry_.message[sizeof(telemetry_.message)-1]='\0';
    }
    bool cleanup() {
        char error[2048]{};
        if(!gp_gl_compute_release(&config_.context,pending_cleanup_,error,sizeof(error))) {
            refuse(Refusal::CleanupPending,error);return false;
        }
        pending_cleanup_=nullptr;telemetry_.program_released=telemetry_.gl_cleanup_verified=1;
        if(telemetry_.link_succeeded) {
            telemetry_.state=uint64_t(State::Linked);telemetry_.refusal=0;telemetry_.message[0]='\0';
        } else refuse(Refusal::NativeLinkFailed,link_error_);
        return true;
    }
    Config config_;
    mutable std::recursive_mutex mutex_;
    contracts::ContractNegotiator* contract_owner_=nullptr; // must outlive host
    std::thread::id lane_;
    contracts::Loan loan_{};
    Work work_{};
    std::string link_error_;
    bool active_=false;
    bool external_cleanup_pending_=false;
    bool external_running_=false;
    gp_gl_compute_program* pending_cleanup_=nullptr;
    Telemetry telemetry_{};
    std::unordered_map<uint64_t,Telemetry> retained_;
};

// Supply this policy to the EXISTING ContractNegotiator. Its host order and
// pressure/availability remain its authority. No second GPU work queue exists.
// Empty GPU row means harness capacity, not globally idle physical hardware.
inline contracts::ContractNegotiator::Offer first_suitable_free(
    std::vector<std::shared_ptr<OpenGLGpuHost>> rows) {
    return [rows=std::move(rows)](const contracts::Peer& peer,
                                const std::vector<contracts::ContractNegotiator::Entry>& entries) {
        std::shared_ptr<OpenGLGpuHost> row;
        for(const auto& candidate:rows)if(candidate && candidate->id()==peer.id)row=candidate;
        if(!row || !row->on_lane())return SIZE_MAX;
        for(size_t i=0;i<entries.size();++i) {
            const auto& e=entries[i];
            if(e.state!=contracts::ContractNegotiator::State::Unaccepted)continue;
            bool already=false;
            for(const auto& other:rows)if(other && other->remembers(e.work.id)){already=true;break;}
            if(!already && row->eligible(e.work))return i;
        }
        return SIZE_MAX;
    };
}

struct PortTypes {ValueTypeId control,telemetry;};
inline PortTypes register_port_types() {
    auto& registry=ValueTypeRegistry::global();
    return {registry.register_primitive("nodus_gpu_control_v1",sizeof(Control)),
            registry.register_primitive("nodus_gpu_telemetry_v2",sizeof(Telemetry))};
}
class OpenGLGpuTool final : public ITool {
public:
    explicit OpenGLGpuTool(std::shared_ptr<OpenGLGpuHost> owner):owner_(std::move(owner)),types_(register_port_types()){}
    const char* id_cstr() const noexcept override {return "nodus.gpu.opengl";}
    const char* name_cstr() const noexcept override {return "OpenGL GPU host";}
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {if(owner_)owner_->shutdown();}
    void tick(double,HostAPI&) override {}
    void render(RenderContext&) override {}
    int32_t port_count() const override {return 2;}
    ToolPortSpec port_spec(int32_t index) const override {
        if(index==0)return {ToolPortKind::Argument,1};
        if(index==1)return {ToolPortKind::Return,1};
        return {};
    }
    void execute_stack(ToolStackContext& context) override {
        auto* frame=context.stack.raw;
        if(!owner_ || !frame || frame->byte_count<sizeof(Control) ||
           frame->byte_count-sizeof(Control)>frame->byte_capacity ||
           sizeof(Telemetry)>frame->byte_capacity-(frame->byte_count-sizeof(Control))) {
            context.succeeded=false;return;
        }
        Control command{};
        if(!raw_stack_pop_typed(*frame,&command,types_.control)) {context.succeeded=false;return;}
        bool accepted=false;
        Telemetry output{};
        try {
            if(!owner_->on_lane() && command.command==uint64_t(Command::Query))owner_->claim_control_lane();
            accepted=owner_->control(command) && owner_->snapshot(
                command.command==uint64_t(Command::Query)?command.contract_id:0,output);
        } catch(...) {accepted=false;}
        if(!accepted) {raw_stack_push_typed(*frame,&command,types_.control);context.succeeded=false;return;}
        if(!raw_stack_push_typed(*frame,&output,types_.telemetry)) {
            // The host retains its outcome, so Query can deliver it again
            // without replaying the link. Restore the command for inspection.
            raw_stack_push_typed(*frame,&command,types_.control);context.succeeded=false;
        }
    }
private:
    std::shared_ptr<OpenGLGpuHost> owner_; PortTypes types_;
};
inline bool register_opengl_gpu_tool(ToolRegistry& registry,
    const std::shared_ptr<OpenGLGpuHost>& owner,const std::string& id) {
    if(!owner || id.empty() || registry.find(id))return false;
    ToolRegistry::Entry entry;entry.id=id;entry.name="Native OpenGL GPU row";
    entry.factory=[owner] {return std::unique_ptr<ITool,std::function<void(ITool*)>>(
        new OpenGLGpuTool(owner),[](ITool* tool){delete tool;});};
    return registry.register_tool(std::move(entry));
}
} // namespace nodus::gpu
