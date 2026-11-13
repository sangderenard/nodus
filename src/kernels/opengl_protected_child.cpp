#include "opengl_protected_protocol.h"
#include <cstring>
#include <cwchar>
#include <limits>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace nodus::gpu::protected_detail {
int protected_child_main(int argc,wchar_t** argv,ChildLink callback) {
#ifndef _WIN32
    (void)argc;(void)argv;(void)callback;return 90;
#else
    // Standalone invocation cannot create a context or reach the callback.
    if(argc!=4 || !argv || std::wcscmp(argv[1],L"--nodus-protected-gl-worker") || !callback)return 91;
    const auto parse=[](const wchar_t* text)->HANDLE {
        if(!text || !*text || *text==L'-' || *text==L'+')return nullptr;
        wchar_t* end=nullptr;const auto value=std::wcstoull(text,&end,10);
        if(!end || *end || !value || value>std::numeric_limits<uintptr_t>::max())return nullptr;
        return reinterpret_cast<HANDLE>(static_cast<uintptr_t>(value));
    };
    HANDLE request_handle=parse(argv[2]),result_handle=parse(argv[3]);
    if(!request_handle || !result_handle || request_handle==result_handle)return 92;
    struct Handles {HANDLE a,b;~Handles(){CloseHandle(a);CloseHandle(b);}} handles{request_handle,result_handle};
    // Query only the header first; never trust an input's allocation length.
    auto* initial=static_cast<const WireRequest*>(MapViewOfFile(request_handle,FILE_MAP_READ,0,0,sizeof(WireRequest)));
    if(!initial)return 93;
    const uint32_t total=initial->total_bytes;
    const bool header_valid=initial->magic==request_magic && initial->version==protocol_version &&
        total>=sizeof(WireRequest) && total<=sizeof(WireRequest)+source_limit &&
        initial->source_count && initial->source_count<=object_limit &&
        initial->context_policy<=static_cast<uint32_t>(ProtectedContextPolicy::VerifiedDefaultDc) &&
        initial->host_id && initial->contract_id && initial->generation;
    UnmapViewOfFile(initial);
    if(!header_valid)return 94;
    const auto* request=static_cast<const WireRequest*>(MapViewOfFile(request_handle,FILE_MAP_READ,0,0,total));
    auto* response=static_cast<WireResult*>(MapViewOfFile(result_handle,FILE_MAP_WRITE,0,0,sizeof(WireResult)));
    struct Views {const void* a;void* b;~Views(){if(a)UnmapViewOfFile(a);if(b)UnmapViewOfFile(b);}} views{request,response};
    if(!request || !response)return 95;
    BOOL assigned=FALSE;JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    constexpr DWORD required=JOB_OBJECT_LIMIT_PROCESS_MEMORY|JOB_OBJECT_LIMIT_JOB_MEMORY|
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE|JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
    if(!IsProcessInJob(GetCurrentProcess(),nullptr,&assigned) || !assigned ||
       !QueryInformationJobObject(nullptr,JobObjectExtendedLimitInformation,&limits,sizeof(limits),nullptr) ||
       (limits.BasicLimitInformation.LimitFlags&required)!=required || limits.BasicLimitInformation.ActiveProcessLimit!=1 ||
       limits.ProcessMemoryLimit!=request->process_memory_bytes || limits.JobMemoryLimit!=request->job_memory_bytes ||
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
#endif
}
} // namespace nodus::gpu::protected_detail
