#include "opengl_protected_protocol.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <psapi.h>
#endif

namespace nodus::gpu {
#ifdef NODUS_PROTECTED_PROCESS_TEST_HOOKS
std::atomic<int> protected_detail::supervisor_fault=0;
#endif
namespace {
void fault(int point) {
#ifdef NODUS_PROTECTED_PROCESS_TEST_HOOKS
    int expected=point;
    if(protected_detail::supervisor_fault.compare_exchange_strong(expected,0))throw std::bad_alloc();
#else
    (void)point;
#endif
}
constexpr uint64_t GiB=1024ull*1024*1024;
ProtectedLinkOutcome refusal(std::string message) {
    ProtectedLinkOutcome out;out.child_reaped=true;out.resources_released=true;
    out.message=std::move(message);return out;
}
bool source_bounds(const ProtectedLinkRequest& request,size_t& bytes,std::string& error) {
    bytes=0;
    if(!request.host_id || !request.contract_id || !request.generation || !request.sources ||
       !request.source_count || request.source_count>protected_detail::object_limit) {
        error="Missing exact loan identity or bounded source objects";return false;
    }
    for(size_t i=0;i<request.source_count;++i) {
        const auto& s=request.sources[i];
        if(!s.source || !s.source_bytes || s.source_bytes>protected_detail::source_limit-bytes) {
            error="Shader source exceeds the 16 MiB transport bound";return false;
        }
        bytes+=s.source_bytes;
        const size_t remaining=protected_detail::source_limit-bytes;
        const size_t label=s.label?strnlen(s.label,remaining+1):0;
        if(label>remaining) {error="Shader labels exceed the transport bound";return false;}
        bytes+=label;
    }
    return true;
}
#ifdef _WIN32
struct Handle {
    HANDLE value=nullptr;
    Handle()=default;explicit Handle(HANDLE h):value(h){}
    ~Handle(){reset();}
    Handle(const Handle&)=delete;Handle& operator=(const Handle&)=delete;
    Handle(Handle&& h) noexcept:value(std::exchange(h.value,nullptr)){}
    Handle& operator=(Handle&& h) noexcept {if(this!=&h){reset();value=std::exchange(h.value,nullptr);}return *this;}
    void reset(HANDLE h=nullptr){if(value && value!=INVALID_HANDLE_VALUE)CloseHandle(value);value=h;}
    explicit operator bool()const{return value && value!=INVALID_HANDLE_VALUE;}
};
struct View {
    void* value=nullptr;
    explicit View(void* p=nullptr):value(p){}
    ~View(){if(value)UnmapViewOfFile(value);}
};
std::string win_error(const char* action){return std::string(action)+" failed (Win32 "+std::to_string(GetLastError())+")";}
bool valid_options(const ProtectedProcessOptions& o,std::string& error) {
    const bool absolute=(o.executable.size()>3 && o.executable[1]==L':' &&
                         (o.executable[2]==L'\\' || o.executable[2]==L'/'));
    const auto attrs=absolute?GetFileAttributesW(o.executable.c_str()):INVALID_FILE_ATTRIBUTES;
    if(!absolute || attrs==INVALID_FILE_ATTRIBUTES || (attrs&FILE_ATTRIBUTE_DIRECTORY)) {
        error="Protected native worker executable must be an existing absolute local path";return false;
    }
    if(std::all_of(o.adapter_luid.begin(),o.adapter_luid.end(),[](uint8_t x){return x==0;}) || !o.node_mask) {
        error="Exact adapter LUID and node mask are required; ordinals and names are not GL affinity";return false;
    }
    if(o.context_policy!=ProtectedContextPolicy::RequireNvAffinity && o.context_policy!=ProtectedContextPolicy::VerifiedDefaultDc) {
        error="Unknown native GL context selection policy";return false;
    }
    if(o.process_memory_bytes<16*1024*1024 || o.process_memory_bytes>4*GiB ||
       o.job_memory_bytes<o.process_memory_bytes || o.job_memory_bytes>4*GiB ||
       o.minimum_available_memory_bytes<8*GiB || !o.wall_timeout_ms || o.wall_timeout_ms>90000 ||
       !o.poll_ms || o.poll_ms>100) {
        error="Protection policy requires <=4 GiB process/job caps, >=8 GiB RAM floor and <=90 s deadline";return false;
    }
    return true;
}
bool memory_available(uint64_t floor) {
    MEMORYSTATUSEX status{};status.dwLength=sizeof(status);
    return GlobalMemoryStatusEx(&status) && status.ullAvailPhys>=floor;
}
std::wstring quote(const std::wstring& in) {
    std::wstring out=L"\"";size_t slashes=0;
    for(wchar_t c:in) {
        if(c==L'\\'){++slashes;continue;}
        out.append(c==L'\"'?2*slashes+1:slashes,L'\\');slashes=0;out.push_back(c);
    }
    out.append(2*slashes,L'\\');out.push_back(L'\"');return out;
}
bool verify_limits(HANDLE job,uint64_t process_bytes,uint64_t job_bytes) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION actual{};
    constexpr DWORD required=JOB_OBJECT_LIMIT_PROCESS_MEMORY|JOB_OBJECT_LIMIT_JOB_MEMORY|
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE|JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
    return QueryInformationJobObject(job,JobObjectExtendedLimitInformation,&actual,sizeof(actual),nullptr) &&
        (actual.BasicLimitInformation.LimitFlags&required)==required &&
        !(actual.BasicLimitInformation.LimitFlags&(JOB_OBJECT_LIMIT_BREAKAWAY_OK|JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK)) &&
        actual.BasicLimitInformation.ActiveProcessLimit==1 && actual.ProcessMemoryLimit==process_bytes &&
        actual.JobMemoryLimit==job_bytes;
}
void accounting(HANDLE job,HANDLE process,ProtectedLinkOutcome& out) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    if(QueryInformationJobObject(job,JobObjectExtendedLimitInformation,&info,sizeof(info),nullptr)) {
        out.peak_process_bytes=info.PeakProcessMemoryUsed;out.peak_job_bytes=info.PeakJobMemoryUsed;
    }
    DWORD code=0;if(GetExitCodeProcess(process,&code))out.exit_code=code;
}
#endif
}

bool protected_source_digest(const ProtectedLinkRequest& request,ProtectedSourceDigest& digest,std::string& error) {
    digest={};size_t bytes=0;if(!source_bounds(request,bytes,error))return false;
#ifdef _WIN32
    BCRYPT_ALG_HANDLE algorithm=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
    if(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0) {error="SHA-256 unavailable";return false;}
    bool ok=BCryptCreateHash(algorithm,&hash,nullptr,0,nullptr,0,0)>=0;
    const auto append=[&](const void* p,size_t n) {if(ok && n)ok=BCryptHashData(hash,
        const_cast<PUCHAR>(static_cast<const UCHAR*>(p)),static_cast<ULONG>(n),0)>=0;};
    const uint64_t count=request.source_count;append(&count,sizeof(count));
    for(size_t i=0;i<request.source_count && ok;++i) {
        const auto& s=request.sources[i];const uint64_t label=s.label?std::strlen(s.label):0,source=s.source_bytes;
        append(&label,sizeof(label));append(s.label,static_cast<size_t>(label));
        append(&source,sizeof(source));append(s.source,static_cast<size_t>(source));
    }
    if(ok)ok=BCryptFinishHash(hash,digest.data(),static_cast<ULONG>(digest.size()),0)>=0;
    if(hash)BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(algorithm,0);
    if(!ok)error="SHA-256 source commitment failed";return ok;
#else
    error="Protected OpenGL process requires Windows";return false;
#endif
}

bool protected_detail::encode_request(const ProtectedLinkRequest& request,const ProtectedProcessOptions& options,
                                     std::vector<uint8_t>& wire,std::string& error) {
    size_t bytes=0;if(!source_bounds(request,bytes,error))return false;
    ProtectedSourceDigest digest{};if(!protected_source_digest(request,digest,error))return false;
    wire.assign(sizeof(WireRequest)+bytes,0);
    auto* header=reinterpret_cast<WireRequest*>(wire.data());*header=WireRequest{};
    header->total_bytes=static_cast<uint32_t>(wire.size());header->source_count=static_cast<uint32_t>(request.source_count);
    header->host_id=request.host_id;header->contract_id=request.contract_id;header->generation=request.generation;
    header->adapter_luid=options.adapter_luid;header->node_mask=options.node_mask;header->digest=digest;
    header->context_policy=static_cast<uint32_t>(options.context_policy);
    header->process_memory_bytes=options.process_memory_bytes;header->job_memory_bytes=options.job_memory_bytes;
    size_t offset=sizeof(WireRequest);
    for(size_t i=0;i<request.source_count;++i) {
        const auto& s=request.sources[i];auto& range=header->objects[i];
        const size_t label=s.label?std::strlen(s.label):0;
        range.label_offset=static_cast<uint32_t>(offset);range.label_bytes=static_cast<uint32_t>(label);
        if(label)std::memcpy(wire.data()+offset,s.label,label);offset+=label;
        range.source_offset=static_cast<uint32_t>(offset);range.source_bytes=static_cast<uint32_t>(s.source_bytes);
        std::memcpy(wire.data()+offset,s.source,s.source_bytes);offset+=s.source_bytes;
    }
    return true;
}

struct ProtectedOpenGLProcess::Impl {
    explicit Impl(ProtectedProcessOptions o):options(std::move(o)){}
    ProtectedProcessOptions options;
    mutable std::mutex mutex;
    std::thread supervisor;
    std::atomic<bool> cancellation=false;
    bool running=false,has_outcome=false;
    ProtectedLinkOutcome outcome;
    ProtectedLinkAuthorization authorization{};
    bool armed=false;
#ifdef _WIN32
    Handle pending_job,pending_process;
    ProtectedLinkOutcome pending_outcome;
#endif
};
ProtectedOpenGLProcess::ProtectedOpenGLProcess(ProtectedProcessOptions options):impl_(new Impl(std::move(options))){}
ProtectedOpenGLProcess::~ProtectedOpenGLProcess() {
    impl_->cancellation.store(true);
    if(impl_->supervisor.joinable())impl_->supervisor.join();
#ifdef _WIN32
    if(impl_->pending_process) {
        TerminateJobObject(impl_->pending_job.value,0xe001);
        WaitForSingleObject(impl_->pending_process.value,5000);
        // The last job handle still enforces kill-on-close if termination is
        // delayed in the kernel. Never claim that such cleanup was observed.
    }
#endif
}
bool ProtectedOpenGLProcess::authorize_once(const ProtectedLinkAuthorization& authorization,std::string& reason) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#ifdef _WIN32
    if(impl_->running){reason="Native child owner is already running";return false;}
    if(!valid_options(impl_->options,reason))return false;
    if(impl_->pending_process || impl_->armed) {reason="Existing request or unreaped child owns this provider";return false;}
    if(!authorization.host_id || !authorization.contract_id || !authorization.generation ||
       std::all_of(authorization.source_digest.begin(),authorization.source_digest.end(),[](uint8_t x){return x==0;})) {
        reason="Authorization must bind one exact loan and source digest";return false;
    }
    impl_->authorization=authorization;impl_->armed=true;reason.clear();return true;
#else
    (void)authorization;reason="Protected OpenGL process requires Windows";return false;
#endif
}
bool ProtectedOpenGLProcess::ready(std::string& reason) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#ifdef _WIN32
    if(impl_->running){reason="Native child owner is already running";return false;}
    if(!valid_options(impl_->options,reason))return false;
    if(impl_->pending_process){reason="The prior native child has not been reaped";return false;}
    if(!impl_->armed){reason="A new explicit one-link authorization is required";return false;}
    reason.clear();return true;
#else
    reason="Protected OpenGL process requires Windows";return false;
#endif
}
bool ProtectedOpenGLProcess::available(std::string& reason) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#ifdef _WIN32
    if(impl_->running){reason="Native child owner is already running";return false;}
    if(impl_->pending_process){reason="The prior native child has not been reaped";return false;}
    return valid_options(impl_->options,reason);
#else
    reason="Protected OpenGL process requires Windows";return false;
#endif
}
bool ProtectedOpenGLProcess::authorized(const ProtectedLinkRequest& request,std::string& reason) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(impl_->running || !impl_->armed){reason="Exact queued loan is awaiting one-link approval";return false;}
    ProtectedSourceDigest digest{};
    if(!protected_source_digest(request,digest,reason))return false;
    const auto& a=impl_->authorization;
    if(a.host_id!=request.host_id || a.contract_id!=request.contract_id || a.generation!=request.generation || a.source_digest!=digest) {
        reason="Approval does not match this exact loan and source";return false;
    }
    reason.clear();return true;
}

ProtectedLinkOutcome ProtectedOpenGLProcess::run(const ProtectedLinkRequest& request) {
#ifndef _WIN32
    (void)request;return refusal("Protected OpenGL process requires Windows");
#else
    const auto& o=impl_->options;std::string error;
    if(impl_->pending_process)return refusal("Unreaped child retains process ownership; no new link permitted");
    if(!impl_->armed)return refusal("No one-link authorization; no child created");
    const auto authorization=impl_->authorization;impl_->armed=false;
    if(!valid_options(o,error))return refusal(error);
    if(impl_->cancellation.load()) {auto out=refusal("Cancelled before child creation");out.status=ProtectedLinkStatus::Cancelled;return out;}
    ProtectedSourceDigest digest{};
    if(!protected_source_digest(request,digest,error))return refusal(error);
    if(request.host_id!=authorization.host_id || request.contract_id!=authorization.contract_id ||
       request.generation!=authorization.generation || digest!=authorization.source_digest)
        return refusal("Exact loan or source differs from the consumed one-link authorization");
    if(!memory_available(o.minimum_available_memory_bytes))return refusal("Available system RAM is below the required floor or cannot be queried");
    std::vector<uint8_t> packet;
    if(!protected_detail::encode_request(request,o,packet,error))return refusal(error);
    SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,TRUE};
    Handle request_mapping(CreateFileMappingW(INVALID_HANDLE_VALUE,&sa,PAGE_READWRITE,0,
        static_cast<DWORD>(packet.size()),nullptr));
    Handle result_mapping(CreateFileMappingW(INVALID_HANDLE_VALUE,&sa,PAGE_READWRITE,0,
        sizeof(protected_detail::WireResult),nullptr));
    if(!request_mapping || !result_mapping)return refusal(win_error("CreateFileMapping"));
    View request_view(MapViewOfFile(request_mapping.value,FILE_MAP_WRITE,0,0,packet.size()));
    View result_view(MapViewOfFile(result_mapping.value,FILE_MAP_READ|FILE_MAP_WRITE,0,0,sizeof(protected_detail::WireResult)));
    if(!request_view.value || !result_view.value)return refusal(win_error("MapViewOfFile"));
    std::memcpy(request_view.value,packet.data(),packet.size());
    std::memset(result_view.value,0,sizeof(protected_detail::WireResult));
    Handle readonly_request;HANDLE duplicate=nullptr;
    if(!DuplicateHandle(GetCurrentProcess(),request_mapping.value,GetCurrentProcess(),&duplicate,FILE_MAP_READ,TRUE,0))
        return refusal(win_error("Duplicate read-only request handle"));
    readonly_request.reset(duplicate);
    // Only these two unnamed memory sections are inherited, never a job,
    // service socket, credential, parent process handle or ambient file handle.
    HANDLE inherited[]{readonly_request.value,result_mapping.value};
    SIZE_T attribute_bytes=0;InitializeProcThreadAttributeList(nullptr,1,0,&attribute_bytes);
    std::vector<uint8_t> attributes(attribute_bytes);
    auto* list=reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
    if(!InitializeProcThreadAttributeList(list,1,0,&attribute_bytes))return refusal(win_error("Initialize process attributes"));
    struct AttributeCleanup{LPPROC_THREAD_ATTRIBUTE_LIST p;~AttributeCleanup(){DeleteProcThreadAttributeList(p);}} attribute_cleanup{list};
    if(!UpdateProcThreadAttribute(list,0,PROC_THREAD_ATTRIBUTE_HANDLE_LIST,inherited,sizeof(inherited),nullptr,nullptr))
        return refusal(win_error("Restrict inherited handles"));
    Handle job(CreateJobObjectW(nullptr,nullptr));
    if(!job)return refusal(win_error("CreateJobObject"));
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_PROCESS_MEMORY|JOB_OBJECT_LIMIT_JOB_MEMORY|
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE|JOB_OBJECT_LIMIT_ACTIVE_PROCESS|JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
    limits.BasicLimitInformation.ActiveProcessLimit=1;
    limits.ProcessMemoryLimit=static_cast<SIZE_T>(o.process_memory_bytes);limits.JobMemoryLimit=static_cast<SIZE_T>(o.job_memory_bytes);
    if(!SetInformationJobObject(job.value,JobObjectExtendedLimitInformation,&limits,sizeof(limits)) ||
       !verify_limits(job.value,o.process_memory_bytes,o.job_memory_bytes))return refusal("Could not install and verify exact native JobObject memory limits");
    std::wstring command=quote(o.executable)+L" --nodus-protected-gl-worker "+
        std::to_wstring(reinterpret_cast<uintptr_t>(readonly_request.value))+L" "+
        std::to_wstring(reinterpret_cast<uintptr_t>(result_mapping.value));
    std::vector<wchar_t> mutable_command(command.begin(),command.end());mutable_command.push_back(0);
    STARTUPINFOEXW startup{};startup.StartupInfo.cb=sizeof(startup);startup.lpAttributeList=list;
    PROCESS_INFORMATION pi{};
    const ULONGLONG started=GetTickCount64();
    if(!CreateProcessW(o.executable.c_str(),mutable_command.data(),nullptr,nullptr,TRUE,
       CREATE_SUSPENDED|CREATE_NO_WINDOW|EXTENDED_STARTUPINFO_PRESENT,nullptr,nullptr,&startup.StartupInfo,&pi))
        return refusal(win_error("Create suspended native worker"));
    Handle process(pi.hProcess),thread(pi.hThread);
    ProtectedLinkOutcome out;out.process_id=pi.dwProcessId;out.requested_luid=o.adapter_luid;
    bool resumed=false;BOOL assigned=FALSE;
    std::atomic<bool> watchdog_started=false,stop=false;
    std::atomic<uint32_t> watchdog_reason=0;
    std::thread watchdog;
    struct WatchdogJoin {
        std::thread& thread;std::atomic<bool>& stop;
        ~WatchdogJoin(){stop.store(true,std::memory_order_release);if(thread.joinable())thread.join();}
    } watchdog_join{watchdog,stop};
    const auto finish=[&](ProtectedLinkOutcome result) {
        result.host_id=request.host_id;result.contract_id=request.contract_id;result.generation=request.generation;
        result.elapsed_ms=GetTickCount64()-started;
        result.child_reaped=WaitForSingleObject(process.value,0)==WAIT_OBJECT_0;
        result.resources_released=result.child_reaped;
        accounting(job.value,process.value,result);
        if(!result.child_reaped) {
            result.status=ProtectedLinkStatus::CleanupUncertain;result.outcome_uncertain=true;result.linked=false;
            result.message+="; child exit is unconfirmed, owner retained; no replay";
            impl_->pending_outcome=result;impl_->pending_job=std::move(job);impl_->pending_process=std::move(process);
        }
        return result;
    };
    try {
    fault(1); // suspended, not assigned: exception must still terminate/reap it
    if(!AssignProcessToJobObject(job.value,process.value) ||
       !IsProcessInJob(process.value,job.value,&assigned) || !assigned ||
       !verify_limits(job.value,o.process_memory_bytes,o.job_memory_bytes)) {
        out.status=ProtectedLinkStatus::SupervisionFailed;out.message="Suspended child could not be assigned to verified memory-capped JobObject";
        TerminateProcess(process.value,0xe002);WaitForSingleObject(process.value,5000);return finish(out);
    }
    out.containment_verified=true;
    try {
        watchdog=std::thread([&] {
            watchdog_started.store(true,std::memory_order_release);
            while(!stop.load(std::memory_order_acquire)) {
                const DWORD wait=WaitForSingleObject(process.value,o.poll_ms);
                if(wait==WAIT_OBJECT_0)return;
                uint32_t reason=0;
                if(wait!=WAIT_TIMEOUT)reason=3;
                else if(impl_->cancellation.load())reason=4;
                else if(GetTickCount64()-started>=o.wall_timeout_ms)reason=1;
                else if(!memory_available(o.minimum_available_memory_bytes))reason=2;
                if(reason) {watchdog_reason.store(reason);TerminateJobObject(job.value,0xe010+reason);return;}
            }
        });
    } catch(...) {
        out.status=ProtectedLinkStatus::SupervisionFailed;out.message="Independent watchdog could not start; suspended child terminated";
        TerminateJobObject(job.value,0xe003);WaitForSingleObject(process.value,5000);return finish(out);
    }
    while(!watchdog_started.load(std::memory_order_acquire))SwitchToThread();
    // No child instruction runs until caps, membership and watchdog exist.
    if(!watchdog_reason.load() && memory_available(o.minimum_available_memory_bytes) &&
       ResumeThread(thread.value)==1)resumed=true;
    if(!resumed) {
        out.status=ProtectedLinkStatus::SupervisionFailed;out.message="Resume refused or failed after supervised admission";
        TerminateJobObject(job.value,0xe004);
    }
    fault(2); // joinable watchdog: unwinding must never terminate the server
    DWORD wait=WaitForSingleObject(process.value,o.wall_timeout_ms+5000);
    if(wait!=WAIT_OBJECT_0) {
        TerminateJobObject(job.value,0xe005);TerminateProcess(process.value,0xe005);
        WaitForSingleObject(process.value,5000);
    }
    stop.store(true,std::memory_order_release);watchdog.join();
    if(!resumed)return finish(out);
    out.attempted=true;out.outcome_uncertain=true; // conservative until a final receipt is verified
    const uint32_t stop_reason=watchdog_reason.load();
    if(stop_reason || wait!=WAIT_OBJECT_0) {
        out.status=stop_reason==4?ProtectedLinkStatus::Cancelled:
            stop_reason==2?ProtectedLinkStatus::MemoryFloor:ProtectedLinkStatus::TimedOut;
        out.message=stop_reason==4?"Native child cancelled and terminated; no replay":
            stop_reason==2?"Native child terminated at the system RAM floor; no replay":
            "Native child exceeded its supervised deadline or wait failed; no replay";
        return finish(out);
    }
    DWORD exit_code=0;
    if(!GetExitCodeProcess(process.value,&exit_code) || exit_code!=0) {
        out.status=ProtectedLinkStatus::ChildFailed;out.message="Native child exited without a successful protocol completion; no replay";
        return finish(out);
    }
    protected_detail::WireResult receipt{};std::memcpy(&receipt,result_view.value,sizeof(receipt));
    if(receipt.magic!=protected_detail::result_magic || receipt.version!=protected_detail::protocol_version ||
       receipt.bytes!=sizeof(receipt) || receipt.host_id!=request.host_id || receipt.contract_id!=request.contract_id ||
       receipt.generation!=request.generation || receipt.digest!=digest || receipt.context_selection>2 ||
       receipt.status>static_cast<uint32_t>(ProtectedLinkStatus::CleanupUncertain) || receipt.attempted>1 ||
       receipt.linked>1 || receipt.identity_verified>1 || receipt.context_released>1 || !std::memchr(receipt.message,0,sizeof(receipt.message)) ||
       (receipt.identity_verified && receipt.observed_luid!=o.adapter_luid) ||
       (receipt.identity_verified && receipt.context_selection!=(o.context_policy==ProtectedContextPolicy::VerifiedDefaultDc?2u:1u)) ||
       (receipt.linked && (!receipt.attempted || !receipt.identity_verified || !receipt.context_released ||
                          receipt.status!=static_cast<uint32_t>(ProtectedLinkStatus::Linked)))) {
        out.status=ProtectedLinkStatus::InvalidReceipt;out.message="Native child receipt failed identity/shape/loan validation; no replay";
        return finish(out);
    }
    out.status=static_cast<ProtectedLinkStatus>(receipt.status);out.attempted=receipt.attempted!=0;
    out.linked=receipt.linked!=0;out.identity_verified=receipt.identity_verified!=0;
    out.observed_luid=receipt.observed_luid;out.message=receipt.message;out.outcome_uncertain=false;
    out.gl_cleanup_verified=receipt.context_released!=0;
    out.context_selection=static_cast<ProtectedContextSelection>(receipt.context_selection);
    return finish(out);
    } catch(...) {
        // No joinable thread may unwind into std::terminate, and no exceptional
        // path may drop an unobserved native child then admit a second one.
        stop.store(true,std::memory_order_release);if(watchdog.joinable())watchdog.join();
        if(job)TerminateJobObject(job.value,0xe006);
        if(process){TerminateProcess(process.value,0xe006);WaitForSingleObject(process.value,5000);}
        out.host_id=request.host_id;out.contract_id=request.contract_id;out.generation=request.generation;
        out.attempted=resumed;out.linked=false;out.outcome_uncertain=resumed;
        out.child_reaped=process && WaitForSingleObject(process.value,0)==WAIT_OBJECT_0;
        out.resources_released=out.child_reaped;out.status=ProtectedLinkStatus::SupervisionFailed;
        out.message.clear(); // avoid allocating on a potential allocation failure
        if(!out.child_reaped) {
            out.status=ProtectedLinkStatus::CleanupUncertain;out.outcome_uncertain=true;
            impl_->pending_outcome=std::move(out);impl_->pending_job=std::move(job);impl_->pending_process=std::move(process);
            // If the copy itself throws, the outer catch still leaves the OS
            // owner retained, and authorize_once/ready remain closed.
            return impl_->pending_outcome;
        }
        accounting(job.value,process.value,out);return out;
    }
#endif
}

ProtectedLinkOutcome ProtectedOpenGLProcess::link(const ProtectedLinkRequest& request) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto refused=[&](const std::string& message) {
        auto out=refusal(message);out.host_id=request.host_id;out.contract_id=request.contract_id;
        out.generation=request.generation;return out;
    };
    if(impl_->running)return refused("Provider already owns an active child; no new request admitted");
#ifdef _WIN32
    if(impl_->pending_process)return refused("Provider retains an unreaped child; no new request admitted");
#endif
    if(!impl_->armed)return refused("No explicit one-link authorization; no child created");
    std::string error;ProtectedSourceDigest digest{};
    if(!protected_source_digest(request,digest,error)) {impl_->armed=false;return refused(error);}
    const auto& a=impl_->authorization;
    if(a.host_id!=request.host_id || a.contract_id!=request.contract_id || a.generation!=request.generation || a.source_digest!=digest) {
        impl_->armed=false;return refused("Exact loan or source differs from the consumed one-link authorization");
    }
    std::vector<std::pair<std::string,std::string>> source;
    try {
        source.reserve(request.source_count);
        for(size_t i=0;i<request.source_count;++i)source.emplace_back(
            request.sources[i].label?request.sources[i].label:"",
            std::string(request.sources[i].source,request.sources[i].source_bytes));
    } catch(...) {impl_->armed=false;return refused("Bounded source copy failed; no child created");}
    if(impl_->supervisor.joinable())impl_->supervisor.join();
    impl_->outcome={};impl_->outcome.host_id=request.host_id;impl_->outcome.contract_id=request.contract_id;
    impl_->outcome.generation=request.generation;impl_->outcome.status=ProtectedLinkStatus::Running;
    impl_->outcome.requested_luid=impl_->options.adapter_luid;
    impl_->outcome.message="Native process owner accepted one exact request; awaiting supervised receipt";
    impl_->running=true;impl_->has_outcome=true;impl_->cancellation.store(false);
    try {
        impl_->supervisor=std::thread([this,source=std::move(source),host=request.host_id,id=request.contract_id,generation=request.generation]() {
            ProtectedLinkOutcome out;
            try {
                std::vector<gp_gl_compute_source> objects;objects.reserve(source.size());
                for(const auto& object:source)objects.push_back({object.first.c_str(),object.second.data(),object.second.size()});
                out=run({host,id,generation,objects.data(),objects.size()});
            } catch(...) {
                // RAII closes the kill-on-close job on an exceptional path.
                // No final process receipt was observed, so never claim cleanup.
                out.status=ProtectedLinkStatus::CleanupUncertain;out.outcome_uncertain=true;out.attempted=true;
                out.message.clear(); // allocation-free even under persistent bad_alloc
            }
            out.host_id=host;out.contract_id=id;out.generation=generation;
            std::lock_guard<std::mutex> done(impl_->mutex);
            impl_->outcome=std::move(out);impl_->running=false;impl_->armed=false;
        });
    } catch(...) {
        impl_->running=false;impl_->armed=false;impl_->outcome=refused("Native process owner thread could not start");
    }
    return impl_->outcome;
}

bool ProtectedOpenGLProcess::poll(ProtectedLinkOutcome& out) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(!impl_->has_outcome)return false;
    out=impl_->outcome;return true;
}
bool ProtectedOpenGLProcess::cancel() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(!impl_->running)return false;
    impl_->cancellation.store(true);return true;
}

bool ProtectedOpenGLProcess::poll_cleanup(ProtectedLinkOutcome& out) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#ifdef _WIN32
    if(impl_->running)return false;
    if(!impl_->pending_process)return false;
    out=impl_->pending_outcome;
    if(WaitForSingleObject(impl_->pending_process.value,0)!=WAIT_OBJECT_0)return true;
    accounting(impl_->pending_job.value,impl_->pending_process.value,out);
    out.child_reaped=true;out.resources_released=true;out.outcome_uncertain=true;
    out.message="Original native child exit now verified; link outcome remains uncertain; no replay";
    impl_->outcome=out;
    impl_->pending_process.reset();impl_->pending_job.reset();impl_->pending_outcome={};return true;
#else
    (void)out;return false;
#endif
}
} // namespace nodus::gpu
