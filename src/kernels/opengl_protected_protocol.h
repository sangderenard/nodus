#pragma once
// Private fixed-width, bounded local-process transport. No pointers on wire.
#include "opengl_protected_process.h"
#include <cstddef>
#include <vector>
#ifdef NODUS_PROTECTED_PROCESS_TEST_HOOKS
#include <atomic>
#endif

namespace nodus::gpu::protected_detail {
#ifdef NODUS_PROTECTED_PROCESS_TEST_HOOKS
// CPU-fixture build only. Not compiled into the library or native GL worker.
extern std::atomic<int> supervisor_fault;
#endif
constexpr uint32_t protocol_version=2;
constexpr uint64_t request_magic=0x315145524c47504eull;
constexpr uint64_t result_magic=0x315345524c47504eull;
constexpr size_t source_limit=16*1024*1024, object_limit=4096;
struct ObjectRange {uint32_t label_offset,label_bytes,source_offset,source_bytes;};
struct WireRequest {
    uint64_t magic=request_magic;
    uint32_t version=protocol_version, total_bytes=0;
    uint64_t host_id=0, contract_id=0, generation=0;
    uint64_t process_memory_bytes=0, job_memory_bytes=0;
    uint32_t source_count=0, node_mask=0;
    uint32_t context_policy=0;
    std::array<uint8_t,8> adapter_luid{};
    ProtectedSourceDigest digest{};
    ObjectRange objects[object_limit]{};
};
struct WireResult {
    uint64_t magic=0;
    uint32_t version=protocol_version, bytes=sizeof(WireResult);
    uint64_t host_id=0, contract_id=0, generation=0;
    ProtectedSourceDigest digest{};
    uint32_t status=0, attempted=0, linked=0, identity_verified=0;
    uint32_t context_released=0, context_selection=0;
    std::array<uint8_t,8> observed_luid{};
    char message[2048]{};
};
struct ChildRequest {
    ProtectedLinkRequest link;
    std::array<uint8_t,8> adapter_luid{};
    uint32_t node_mask=0;
    ProtectedContextPolicy context_policy=ProtectedContextPolicy::RequireNvAffinity;
};
using ChildLink=ProtectedLinkOutcome(*)(const ChildRequest&);
// Entrypoint rejects other arguments, malformed transport, missing limits and
// missing job containment before invoking the native callback. Tests compile a
// different executable with a CPU callback; production has no fixture switch.
int protected_child_main(int argc,wchar_t** argv,ChildLink callback);
ProtectedLinkOutcome link_in_affinity_context(const ChildRequest&);
bool encode_request(const ProtectedLinkRequest&,const ProtectedProcessOptions&,
                    std::vector<uint8_t>&,std::string&);
} // namespace nodus::gpu::protected_detail
