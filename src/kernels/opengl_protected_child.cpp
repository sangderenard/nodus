#include "opengl_protected_protocol.h"
#include <algorithm>
#include <cstring>
#include <cwchar>
#include <limits>
#include <string>

// ===== PLATFORM REGION (the only OS-specific code in this file) =====
// Child-side transport and containment primitives. Windows: inherited file
// mapping handles and a verified JobObject. POSIX: inherited memfd descriptors
// and verified rlimit/process-group/parent-death containment.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
namespace nodus::gpu::protected_detail::platform {
using Native=HANDLE;
// Parses one inherited transport handle from argv. Rejects empty, signed,
// zero, overflowing or trailing-garbage text.
bool parse_handle(const wchar_t* text,Native& out) {
    if(!text || !*text || *text==L'-' || *text==L'+')return false;
    wchar_t* end=nullptr;const auto value=std::wcstoull(text,&end,10);
    if(!end || *end || !value || value>std::numeric_limits<uintptr_t>::max())return false;
    out=reinterpret_cast<HANDLE>(static_cast<uintptr_t>(value));return true;
}
void close_native(Native h){CloseHandle(h);}
// Read-only view of exactly `bytes` of the request section (nullptr on failure).
const void* map_request(Native h,size_t bytes){return MapViewOfFile(h,FILE_MAP_READ,0,0,bytes);}
void* map_result(Native h,size_t bytes){return MapViewOfFile(h,FILE_MAP_WRITE,0,0,bytes);}
void unmap(const void* p,size_t){UnmapViewOfFile(p);}
// True only when this process is inside the supervisor's verified containment
// with exactly the requested memory limits.
bool contained(uint64_t process_bytes,uint64_t job_bytes) {
    BOOL assigned=FALSE;JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    constexpr DWORD required=JOB_OBJECT_LIMIT_PROCESS_MEMORY|JOB_OBJECT_LIMIT_JOB_MEMORY|
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE|JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
    return IsProcessInJob(GetCurrentProcess(),nullptr,&assigned) && assigned &&
        QueryInformationJobObject(nullptr,JobObjectExtendedLimitInformation,&limits,sizeof(limits),nullptr) &&
        (limits.BasicLimitInformation.LimitFlags&required)==required && limits.BasicLimitInformation.ActiveProcessLimit==1 &&
        limits.ProcessMemoryLimit==process_bytes && limits.JobMemoryLimit==job_bytes;
}
}
#else
#include <csignal>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
namespace nodus::gpu::protected_detail::platform {
using Native=int;
bool parse_handle(const wchar_t* text,Native& out) {
    if(!text || !*text || *text==L'-' || *text==L'+')return false;
    wchar_t* end=nullptr;const auto value=std::wcstoull(text,&end,10);
    // 0..2 are the standard streams, never transport descriptors.
    if(!end || *end || value<3 || value>static_cast<unsigned long long>(std::numeric_limits<int>::max()))return false;
    out=static_cast<int>(value);return true;
}
void close_native(Native fd){::close(fd);}
void* map_section(Native fd,size_t bytes,int protection,int sharing) {
    struct stat info{};
    // A mapping beyond the section would fault on access, so refuse it here
    // (Windows MapViewOfFile refuses the same request).
    if(fstat(fd,&info)!=0 || info.st_size<0 || static_cast<size_t>(info.st_size)<bytes)return nullptr;
    void* p=mmap(nullptr,bytes,protection,sharing,fd,0);
    return p==MAP_FAILED?nullptr:p;
}
const void* map_request(Native fd,size_t bytes){return map_section(fd,bytes,PROT_READ,MAP_PRIVATE);}  // sealed section: a shared mapping of it is refused
void* map_result(Native fd,size_t bytes){return map_section(fd,bytes,PROT_READ|PROT_WRITE,MAP_SHARED);}
void unmap(const void* p,size_t bytes){munmap(const_cast<void*>(p),bytes);}
// The supervisor set the memory limit, own process group and parent-death
// kill before releasing this process; verify each from the inside.
bool contained(uint64_t process_bytes,uint64_t) {
    struct rlimit data{};int death=0;
    return getrlimit(RLIMIT_DATA,&data)==0 && data.rlim_cur==process_bytes && data.rlim_max==process_bytes &&
        getpgrp()==getpid() && prctl(PR_GET_PDEATHSIG,&death)==0 && death==SIGKILL;
}
}
#endif
// ===== END PLATFORM REGION =====

namespace nodus::gpu::protected_detail {
namespace {
struct Handles {
    platform::Native a,b;
    ~Handles(){platform::close_native(a);platform::close_native(b);}
};
struct Views {
    const void* a;size_t a_bytes;void* b;size_t b_bytes;
    ~Views(){if(a)platform::unmap(a,a_bytes);if(b)platform::unmap(b,b_bytes);}
};
}
int protected_child_main(int argc,wchar_t** argv,ChildLink callback) {
    // Standalone invocation cannot create a context or reach the callback.
    if(argc!=4 || !argv || std::wcscmp(argv[1],L"--nodus-protected-gl-worker") || !callback)return 91;
    platform::Native request_handle{},result_handle{};
    if(!platform::parse_handle(argv[2],request_handle) || !platform::parse_handle(argv[3],result_handle) ||
       request_handle==result_handle)return 92;
    Handles handles{request_handle,result_handle};
    // Query only the header first; never trust an input's allocation length.
    auto* initial=static_cast<const WireRequest*>(platform::map_request(request_handle,sizeof(WireRequest)));
    if(!initial)return 93;
    const uint32_t total=initial->total_bytes;
    const bool header_valid=initial->magic==request_magic && initial->version==protocol_version &&
        total>=sizeof(WireRequest) && total<=sizeof(WireRequest)+source_limit &&
        initial->source_count && initial->source_count<=object_limit &&
        initial->context_policy<=static_cast<uint32_t>(ProtectedContextPolicy::VerifiedDefaultDc) &&
        initial->host_id && initial->contract_id && initial->generation;
    platform::unmap(initial,sizeof(WireRequest));
    if(!header_valid)return 94;
    const auto* request=static_cast<const WireRequest*>(platform::map_request(request_handle,total));
    auto* response=static_cast<WireResult*>(platform::map_result(result_handle,sizeof(WireResult)));
    Views views{request,total,response,sizeof(WireResult)};
    if(!request || !response)return 95;
    if(!platform::contained(request->process_memory_bytes,request->job_memory_bytes) ||
       request->process_memory_bytes>4ull*1024*1024*1024 || request->job_memory_bytes>4ull*1024*1024*1024 ||
       request->job_memory_bytes<request->process_memory_bytes)return 96;
    try {
        std::vector<std::string> labels;labels.reserve(request->source_count);
        std::vector<gp_gl_compute_source> sources;sources.reserve(request->source_count);
        const auto* bytes=reinterpret_cast<const char*>(request);size_t offset=sizeof(WireRequest);
        for(uint32_t i=0;i<request->source_count;++i) {
            const auto& range=request->objects[i];
            if(range.label_offset!=offset || range.label_bytes>total-offset)return 97;
            if(range.label_bytes && std::memchr(bytes+offset,0,range.label_bytes))return 97;
            labels.emplace_back(bytes+offset,range.label_bytes);offset+=range.label_bytes;
            if(range.source_offset!=offset || !range.source_bytes || range.source_bytes>total-offset)return 97;
            sources.push_back({labels.back().c_str(),bytes+offset,range.source_bytes});offset+=range.source_bytes;
        }
        if(offset!=total)return 97;
        ChildRequest child{{request->host_id,request->contract_id,request->generation,sources.data(),sources.size()},
                           request->adapter_luid,request->node_mask,static_cast<ProtectedContextPolicy>(request->context_policy)};
        ProtectedSourceDigest digest{};std::string error;
        if(!protected_source_digest(child.link,digest,error) || digest!=request->digest)return 98;
        // No callback may observe unvalidated source, unbounded bytes or a child
        // that started outside verified OS memory containment.
        const auto outcome=callback(child);
        WireResult result{};result.host_id=request->host_id;result.contract_id=request->contract_id;
        result.generation=request->generation;result.digest=digest;
        result.status=static_cast<uint32_t>(outcome.status);result.attempted=outcome.attempted?1:0;
        result.linked=outcome.linked?1:0;result.identity_verified=outcome.identity_verified?1:0;
        result.context_released=outcome.resources_released?1:0;result.observed_luid=outcome.observed_luid;
        result.context_selection=static_cast<uint32_t>(outcome.context_selection);
        const size_t message=std::min(outcome.message.size(),sizeof(result.message)-1);
        std::memcpy(result.message,outcome.message.data(),message);
        // Written once. The parent reads only after observed process exit and
        // validates magic, size, version, exact loan, digest and adapter again.
        result.magic=result_magic;std::memcpy(response,&result,sizeof(result));return 0;
    } catch(...) {return 99;}
}
} // namespace nodus::gpu::protected_detail
