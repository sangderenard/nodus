// CPU-only real Windows containment test. This executable has NO OpenGL or
// Vulkan dependency. Its child callback is a fixture, not a GPU proof.
#include "opengl_protected_process.h"
#include "kernels/opengl_protected_protocol.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

using namespace nodus::gpu;
static void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
static ProtectedLinkOutcome cpu_child(const protected_detail::ChildRequest& request) {
    ProtectedLinkOutcome out;out.status=ProtectedLinkStatus::Linked;out.attempted=true;
    out.linked=true;out.identity_verified=true;out.resources_released=true;
    out.gl_cleanup_verified=true;out.observed_luid=request.adapter_luid;
    out.context_selection=request.context_policy==ProtectedContextPolicy::VerifiedDefaultDc?
        ProtectedContextSelection::VerifiedDefaultDc:ProtectedContextSelection::NvAffinity;
    out.message="CPU fixture receipt; no driver or GL context";
    if(request.link.source_count!=2 || request.link.sources[0].source_bytes!=5 ||
       std::memcmp(request.link.sources[0].source,"a\0b\nc",5) ||
       request.link.sources[1].source_bytes!=7 || std::memcmp(request.link.sources[1].source,"object2",7))ExitProcess(74);
    const std::string mode=request.link.sources[0].label;
    if(mode=="timeout" || mode=="cancel")Sleep(10000);
    if(mode=="crash")TerminateProcess(GetCurrentProcess(),77);
    if(mode=="identity")out.observed_luid[0]^=1;
    if(mode=="selection")out.context_selection=ProtectedContextSelection::VerifiedDefaultDc;
    if(mode=="bounded") {
        // Parent caps commit at 64 MiB. A single 128 MiB request must be denied.
        void* allocation=VirtualAlloc(nullptr,128ull*1024*1024,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
        if(allocation){VirtualFree(allocation,0,MEM_RELEASE);ExitProcess(75);}
        out.message="CPU fixture: 128 MiB commit denied by verified 64 MiB job/process cap";
    }
    if(mode=="descendant") {
        wchar_t executable[32768]{};GetModuleFileNameW(nullptr,executable,32768);
        STARTUPINFOW startup{};startup.cb=sizeof(startup);PROCESS_INFORMATION process{};
        const BOOL created=CreateProcessW(executable,nullptr,nullptr,nullptr,FALSE,
            CREATE_SUSPENDED|CREATE_NO_WINDOW,nullptr,nullptr,&startup,&process);
        if(created){TerminateProcess(process.hProcess,76);WaitForSingleObject(process.hProcess,5000);
            CloseHandle(process.hThread);CloseHandle(process.hProcess);ExitProcess(76);}
        out.message="CPU fixture: descendant creation denied by active-process limit";
    }
    return out;
}
struct Work {
    std::string label="normal",first=std::string("a\0b\nc",5),second="object2";
    gp_gl_compute_source objects[2]{};
    ProtectedLinkRequest request() {
        objects[0]={label.c_str(),first.data(),first.size()};objects[1]={"second",second.c_str(),second.size()};
        return {101,202,303,objects,2};
    }
};
static ProtectedProcessOptions options() {
    wchar_t executable[32768]{};require(GetModuleFileNameW(nullptr,executable,32768)!=0,"self path");
    ProtectedProcessOptions options;options.executable=executable;options.adapter_luid={1,2,3,4,5,6,7,8};
    options.node_mask=1;options.process_memory_bytes=64ull*1024*1024;options.job_memory_bytes=64ull*1024*1024;
    options.wall_timeout_ms=5000;return options;
}
static void arm(ProtectedOpenGLProcess& process,ProtectedLinkRequest request) {
    ProtectedSourceDigest digest{};std::string error;
    require(protected_source_digest(request,digest,error),error.c_str());
    require(process.authorize_once({request.host_id,request.contract_id,request.generation,digest},error),error.c_str());
    require(process.ready(error),error.c_str());
}
static ProtectedLinkOutcome wait(ProtectedOpenGLProcess& process,ProtectedLinkRequest request) {
    ProtectedLinkOutcome out;const auto started=GetTickCount64();
    do {
        require(process.poll(out),"poll missing receipt");
        require(out.host_id==request.host_id && out.contract_id==request.contract_id && out.generation==request.generation,"exact receipt identity");
        if(out.status!=ProtectedLinkStatus::Running)return out;
        Sleep(2);
    } while(GetTickCount64()-started<12000);
    throw std::runtime_error("CPU test deadline exceeded");
}
static void report(const char* name,const ProtectedLinkOutcome& out) {
    std::printf("%s: status=%u pid=%u exit=%u reaped=%d containment=%d uncertain=%d peak_process=%llu peak_job=%llu ms=%llu %s\n",
        name,static_cast<unsigned>(out.status),out.process_id,out.exit_code,out.child_reaped,out.containment_verified,
        out.outcome_uncertain,static_cast<unsigned long long>(out.peak_process_bytes),
        static_cast<unsigned long long>(out.peak_job_bytes),static_cast<unsigned long long>(out.elapsed_ms),out.message.c_str());
}
int wmain(int argc,wchar_t** argv) {
    if(argc>1)return protected_detail::protected_child_main(argc,argv,cpu_child);
    try {
        MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
        if(!GlobalMemoryStatusEx(&memory) ||
           memory.ullAvailPhys<options().minimum_available_memory_bytes) {
            std::puts("SKIP: protected child requires the configured available RAM floor");
            return 77;
        }
        Work work;auto request=work.request();std::string error;
        ProtectedOpenGLProcess unauthorized(options());
        require(!unauthorized.ready(error),"unarmed provider admitted");
        require(unauthorized.available(error),"structural availability depends on future loan approval");
        require(!unauthorized.authorized(request,error),"unarmed exact request authorized");
        auto out=unauthorized.link(request);
        require(out.status==ProtectedLinkStatus::Refused && !out.process_id && out.child_reaped && out.resources_released,"unarmed creation");
        require(out.host_id==request.host_id && out.contract_id==request.contract_id && out.generation==request.generation,"prelaunch receipt identity");

        ProtectedOpenGLProcess changed(options());arm(changed,request);work.first[0]='z';
        out=changed.link(work.request());require(out.status==ProtectedLinkStatus::Refused && !out.process_id,"changed source launched");
        require(!changed.ready(error),"changed source did not consume authorization");work.first[0]='a';request=work.request();

        ProtectedOpenGLProcess wrong_loan(options());arm(wrong_loan,request);auto other=request;++other.generation;
        out=wrong_loan.link(other);require(out.status==ProtectedLinkStatus::Refused && !out.process_id,"wrong generation launched");
        ProtectedSourceDigest digest{};
        gp_gl_compute_source too_large{"large","x",protected_detail::source_limit+1};
        require(!protected_source_digest({1,2,3,&too_large,1},digest,error),"oversized source accepted");
        auto bad=options();bad.job_memory_bytes=8ull*1024*1024*1024;
        ProtectedOpenGLProcess unsafe(bad);require(!unsafe.ready(error),"unsafe policy admitted");
        auto floor=options();floor.minimum_available_memory_bytes=UINT64_MAX;
        ProtectedOpenGLProcess floor_refusal(floor);arm(floor_refusal,request);floor_refusal.link(request);
        out=wait(floor_refusal,request);require(!out.process_id && out.child_reaped && !out.attempted,"RAM-floor refusal launched");

#ifdef NODUS_PROTECTED_PROCESS_TEST_HOOKS
        for(int point:{1,2}) {
            work.label="timeout";request=work.request();ProtectedOpenGLProcess process(options());arm(process,request);
            protected_detail::supervisor_fault.store(point);process.link(request);out=wait(process,request);
            report(point==1?"exception-before-assignment":"exception-after-resume",out);
            require(out.status==ProtectedLinkStatus::SupervisionFailed && out.child_reaped && out.resources_released,"exception leaked process owner");
            require(out.attempted==(point==2) && out.outcome_uncertain==(point==2),"exception attempt uncertainty wrong");
        }
#endif

        for(const char* mode:{"normal","bounded","descendant","identity","selection","crash","timeout","cancel"}) {
            work.label=mode;request=work.request();auto config=options();
            if(work.label=="timeout")config.wall_timeout_ms=150;
            if(work.label=="normal")config.context_policy=ProtectedContextPolicy::VerifiedDefaultDc;
            ProtectedOpenGLProcess process(config);arm(process,request);
            const auto start=GetTickCount64();out=process.link(request);
            require(GetTickCount64()-start<200,"start blocked manager lane");
            require(out.status==ProtectedLinkStatus::Running,"start did not return Running");
            require(!process.ready(error),"running process admitted another loan");
            if(work.label=="cancel"){Sleep(80);require(process.cancel(),"cancel request refused");}
            out=wait(process,request);report(mode,out);
            require(out.child_reaped && out.resources_released,"child/resources not reaped");
            require(out.containment_verified,"JobObject containment not verified");
            HANDLE old=OpenProcess(SYNCHRONIZE,FALSE,out.process_id);
            if(old){require(WaitForSingleObject(old,0)==WAIT_OBJECT_0,"worker still alive");CloseHandle(old);}
            if(work.label=="normal" || work.label=="bounded" || work.label=="descendant")
                require(out.status==ProtectedLinkStatus::Linked && out.linked && out.identity_verified && out.gl_cleanup_verified && !out.outcome_uncertain &&
                    out.context_selection==(config.context_policy==ProtectedContextPolicy::VerifiedDefaultDc?
                        ProtectedContextSelection::VerifiedDefaultDc:ProtectedContextSelection::NvAffinity),"CPU success receipt rejected");
            else if(work.label=="identity" || work.label=="selection")require(out.status==ProtectedLinkStatus::InvalidReceipt && !out.linked && out.outcome_uncertain,"wrong adapter/selection receipt accepted");
            else if(work.label=="crash")require(out.status==ProtectedLinkStatus::ChildFailed && out.outcome_uncertain,"crash not contained");
            else if(work.label=="timeout")require(out.status==ProtectedLinkStatus::TimedOut && out.outcome_uncertain,"watchdog did not terminate");
            else require(out.status==ProtectedLinkStatus::Cancelled && out.outcome_uncertain,"cancel did not terminate");
            const auto pid=out.process_id;
            auto replay=process.link(request);require(replay.status==ProtectedLinkStatus::Refused && !replay.process_id,"authorization replay spawned child");
            ProtectedLinkOutcome retained;require(process.poll(retained) && retained.process_id==pid,"terminal receipt not retained");
        }
        std::puts("PASS: CPU-only native suspended-child containment, nonblocking control, limits, watchdog, identity, source/loan binding and no replay");return 0;
    } catch(const std::exception& error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
