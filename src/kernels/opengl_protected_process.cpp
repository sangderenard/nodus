#include "opengl_protected_protocol.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

// ===== PLATFORM REGION (the only OS-specific code in this file) =====
// Everything below the matching END marker is platform-free. The region owns:
// the monotonic clock, the RAM floor query, executable validation, SHA-256,
// and ChildProcess (the supervised child, its two transport sections, memory
// containment, wait/terminate and accounting).
// Windows: file-mapping sections + a kill-on-close memory-capped JobObject.
// POSIX: memfd sections + fork/exec with RLIMIT_DATA, own process group,
// parent-death SIGKILL, and a start gate so no child instruction runs before
// containment is verified and the watchdog exists.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <psapi.h>
namespace nodus::gpu::platform {
enum class WaitResult {Exited,Timeout,Failed};
uint64_t tick_ms(){return GetTickCount64();}
void yield_thread(){SwitchToThread();}
bool memory_available(uint64_t floor) {
    MEMORYSTATUSEX status{};status.dwLength=sizeof(status);
    return GlobalMemoryStatusEx(&status) && status.ullAvailPhys>=floor;
}
bool executable_ok(const std::wstring& executable) {
    const bool absolute=(executable.size()>3 && executable[1]==L':' &&
                         (executable[2]==L'\\' || executable[2]==L'/'));
    const auto attrs=absolute?GetFileAttributesW(executable.c_str()):INVALID_FILE_ATTRIBUTES;
    return absolute && attrs!=INVALID_FILE_ATTRIBUTES && !(attrs&FILE_ATTRIBUTE_DIRECTORY);
}
class Sha256 {
public:
    Sha256() {
        open_=BCryptOpenAlgorithmProvider(&algorithm_,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0;
        ok_=open_ && BCryptCreateHash(algorithm_,&hash_,nullptr,0,nullptr,0,0)>=0;
    }
    Sha256(const Sha256&)=delete;Sha256& operator=(const Sha256&)=delete;
    ~Sha256(){if(hash_)BCryptDestroyHash(hash_);BCryptCloseAlgorithmProvider(algorithm_,0);}
    bool available()const{return open_;}
    void update(const void* p,size_t n) {
        if(ok_ && n)ok_=BCryptHashData(hash_,const_cast<PUCHAR>(static_cast<const UCHAR*>(p)),static_cast<ULONG>(n),0)>=0;
    }
    bool finish(uint8_t* out,size_t bytes) {
        if(ok_)ok_=BCryptFinishHash(hash_,out,static_cast<ULONG>(bytes),0)>=0;
        return ok_;
    }
private:
    BCRYPT_ALG_HANDLE algorithm_=nullptr;BCRYPT_HASH_HANDLE hash_=nullptr;bool open_=false,ok_=false;
};
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
    View()=default;explicit View(void* p):value(p){}
    ~View(){reset();}
    View(const View&)=delete;View& operator=(const View&)=delete;
    View(View&& v) noexcept:value(std::exchange(v.value,nullptr)){}
    View& operator=(View&& v) noexcept {if(this!=&v){reset();value=std::exchange(v.value,nullptr);}return *this;}
    void reset(void* p=nullptr){if(value)UnmapViewOfFile(value);value=p;}
};
std::string win_error(const char* action){return std::string(action)+" failed (Win32 "+std::to_string(GetLastError())+")";}
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
// One supervised child with its job/containment owner. Move-only. Destroying
// it closes the kill-on-close job, as the original handle owners did.
class ChildProcess {
public:
    ChildProcess()=default;
    ChildProcess(ChildProcess&&)=default;ChildProcess& operator=(ChildProcess&&)=default;
    explicit operator bool()const{return static_cast<bool>(process_);}
    uint32_t id()const{return pid_;}
    uint64_t started_ms()const{return started_;}
    // Creates the two transport sections, the capped job and the SUSPENDED child.
    bool spawn(const ProtectedProcessOptions& o,const std::vector<uint8_t>& packet,std::string& error) {
        SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,TRUE};
        request_mapping_.reset(CreateFileMappingW(INVALID_HANDLE_VALUE,&sa,PAGE_READWRITE,0,
            static_cast<DWORD>(packet.size()),nullptr));
        result_mapping_.reset(CreateFileMappingW(INVALID_HANDLE_VALUE,&sa,PAGE_READWRITE,0,
            sizeof(protected_detail::WireResult),nullptr));
        if(!request_mapping_ || !result_mapping_){error=win_error("CreateFileMapping");return false;}
        View request_view(MapViewOfFile(request_mapping_.value,FILE_MAP_WRITE,0,0,packet.size()));
        result_view_.reset(MapViewOfFile(result_mapping_.value,FILE_MAP_READ|FILE_MAP_WRITE,0,0,sizeof(protected_detail::WireResult)));
        if(!request_view.value || !result_view_.value){error=win_error("MapViewOfFile");return false;}
        std::memcpy(request_view.value,packet.data(),packet.size());
        std::memset(result_view_.value,0,sizeof(protected_detail::WireResult));
        HANDLE duplicate=nullptr;
        if(!DuplicateHandle(GetCurrentProcess(),request_mapping_.value,GetCurrentProcess(),&duplicate,FILE_MAP_READ,TRUE,0)){
            error=win_error("Duplicate read-only request handle");return false;
        }
        readonly_request_.reset(duplicate);
        // Only these two unnamed memory sections are inherited, never a job,
        // service socket, credential, parent process handle or ambient file handle.
        HANDLE inherited[]{readonly_request_.value,result_mapping_.value};
        SIZE_T attribute_bytes=0;InitializeProcThreadAttributeList(nullptr,1,0,&attribute_bytes);
        std::vector<uint8_t> attributes(attribute_bytes);
        auto* list=reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
        if(!InitializeProcThreadAttributeList(list,1,0,&attribute_bytes)){error=win_error("Initialize process attributes");return false;}
        struct AttributeCleanup{LPPROC_THREAD_ATTRIBUTE_LIST p;~AttributeCleanup(){DeleteProcThreadAttributeList(p);}} attribute_cleanup{list};
        if(!UpdateProcThreadAttribute(list,0,PROC_THREAD_ATTRIBUTE_HANDLE_LIST,inherited,sizeof(inherited),nullptr,nullptr)){
            error=win_error("Restrict inherited handles");return false;
        }
        job_.reset(CreateJobObjectW(nullptr,nullptr));
        if(!job_){error=win_error("CreateJobObject");return false;}
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_PROCESS_MEMORY|JOB_OBJECT_LIMIT_JOB_MEMORY|
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE|JOB_OBJECT_LIMIT_ACTIVE_PROCESS|JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
        limits.BasicLimitInformation.ActiveProcessLimit=1;
        limits.ProcessMemoryLimit=static_cast<SIZE_T>(o.process_memory_bytes);limits.JobMemoryLimit=static_cast<SIZE_T>(o.job_memory_bytes);
        if(!SetInformationJobObject(job_.value,JobObjectExtendedLimitInformation,&limits,sizeof(limits)) ||
           !verify_limits(job_.value,o.process_memory_bytes,o.job_memory_bytes)){
            error="Could not install and verify exact native JobObject memory limits";return false;
        }
        std::wstring command=quote(o.executable)+L" --nodus-protected-gl-worker "+
            std::to_wstring(reinterpret_cast<uintptr_t>(readonly_request_.value))+L" "+
            std::to_wstring(reinterpret_cast<uintptr_t>(result_mapping_.value));
        std::vector<wchar_t> mutable_command(command.begin(),command.end());mutable_command.push_back(0);
        STARTUPINFOEXW startup{};startup.StartupInfo.cb=sizeof(startup);startup.lpAttributeList=list;
        PROCESS_INFORMATION pi{};
        started_=GetTickCount64();
        if(!CreateProcessW(o.executable.c_str(),mutable_command.data(),nullptr,nullptr,TRUE,
           CREATE_SUSPENDED|CREATE_NO_WINDOW|EXTENDED_STARTUPINFO_PRESENT,nullptr,nullptr,&startup.StartupInfo,&pi)){
            error=win_error("Create suspended native worker");return false;
        }
        process_.reset(pi.hProcess);thread_.reset(pi.hThread);pid_=pi.dwProcessId;
        return true;
    }
    // Places the still-suspended child in the verified job.
    bool contain(const ProtectedProcessOptions& o) {
        BOOL assigned=FALSE;
        return AssignProcessToJobObject(job_.value,process_.value) &&
            IsProcessInJob(process_.value,job_.value,&assigned) && assigned &&
            verify_limits(job_.value,o.process_memory_bytes,o.job_memory_bytes);
    }
    bool resume(){return ResumeThread(thread_.value)==1;}
    WaitResult wait(uint32_t ms) const {
        const DWORD w=WaitForSingleObject(process_.value,ms);
        return w==WAIT_OBJECT_0?WaitResult::Exited:w==WAIT_TIMEOUT?WaitResult::Timeout:WaitResult::Failed;
    }
    // Terminates the whole job (the child and anything it could have created).
    void terminate_group(uint32_t code){TerminateJobObject(job_.value,code);}
    void terminate_process(uint32_t code){TerminateProcess(process_.value,code);}
    bool exit_code(uint32_t& code)const{DWORD c=0;if(!GetExitCodeProcess(process_.value,&c))return false;code=c;return true;}
    void accounting(ProtectedLinkOutcome& out) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
        if(QueryInformationJobObject(job_.value,JobObjectExtendedLimitInformation,&info,sizeof(info),nullptr)) {
            out.peak_process_bytes=info.PeakProcessMemoryUsed;out.peak_job_bytes=info.PeakJobMemoryUsed;
        }
        uint32_t code=0;if(exit_code(code))out.exit_code=code;
    }
    // Copies the child's result section; call only after observed exit.
    void read_result(protected_detail::WireResult& receipt)const{std::memcpy(&receipt,result_view_.value,sizeof(receipt));}
    // Drops the transport sections and the start thread; keeps job and process.
    void close_io(){result_view_.reset();readonly_request_.reset();result_mapping_.reset();request_mapping_.reset();thread_.reset();}
private:
    Handle job_,process_,thread_,request_mapping_,result_mapping_,readonly_request_;
    View result_view_;uint32_t pid_=0;uint64_t started_=0;
};
}
#else
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <poll.h>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
namespace nodus::gpu::platform {
enum class WaitResult {Exited,Timeout,Failed};
uint64_t tick_ms() {
    timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);
    return static_cast<uint64_t>(t.tv_sec)*1000u+static_cast<uint64_t>(t.tv_nsec)/1000000u;
}
void yield_thread(){std::this_thread::yield();}
// /proc/meminfo MemAvailable, the same "available physical RAM" the floor means.
bool memory_available(uint64_t floor) {
    FILE* f=std::fopen("/proc/meminfo","r");if(!f)return false;
    char line[256];unsigned long long kib=0;bool found=false;
    while(std::fgets(line,sizeof(line),f))if(std::sscanf(line,"MemAvailable: %llu kB",&kib)==1){found=true;break;}
    std::fclose(f);return found && kib>=floor/1024u+(floor%1024u?1u:0u);
}
// wchar_t is UTF-32 here; the executable path crosses the API as std::wstring.
bool to_utf8(const std::wstring& in,std::string& out) {
    out.clear();
    for(wchar_t w:in) {
        uint32_t c=static_cast<uint32_t>(w);
        if(c>0x10ffffu || (c>=0xd800u && c<=0xdfffu))return false;
        if(c<0x80u)out.push_back(static_cast<char>(c));
        else if(c<0x800u){out.push_back(static_cast<char>(0xc0|(c>>6)));out.push_back(static_cast<char>(0x80|(c&0x3f)));}
        else if(c<0x10000u){out.push_back(static_cast<char>(0xe0|(c>>12)));out.push_back(static_cast<char>(0x80|((c>>6)&0x3f)));out.push_back(static_cast<char>(0x80|(c&0x3f)));}
        else{out.push_back(static_cast<char>(0xf0|(c>>18)));out.push_back(static_cast<char>(0x80|((c>>12)&0x3f)));out.push_back(static_cast<char>(0x80|((c>>6)&0x3f)));out.push_back(static_cast<char>(0x80|(c&0x3f)));}
    }
    return true;
}
bool executable_ok(const std::wstring& executable) {
    std::string path;struct stat info{};
    return to_utf8(executable,path) && !path.empty() && path[0]=='/' && stat(path.c_str(),&info)==0 &&
        !S_ISDIR(info.st_mode) && access(path.c_str(),X_OK)==0;
}
// FIPS 180-4 SHA-256 (no system crypto library is assumed on the target).
class Sha256 {
public:
    Sha256(){for(int i=0;i<8;++i)h_[i]=init_[i];}
    bool available()const{return true;}
    void update(const void* p,size_t n) {
        const auto* b=static_cast<const uint8_t*>(p);total_+=n;
        while(n) {
            const size_t take=std::min<size_t>(64-fill_,n);std::memcpy(block_+fill_,b,take);
            fill_+=take;b+=take;n-=take;
            if(fill_==64){compress();fill_=0;}
        }
    }
    bool finish(uint8_t* out,size_t bytes) {
        if(bytes!=32)return false;
        const uint64_t bits=total_*8u;block_[fill_++]=0x80;
        if(fill_>56){std::memset(block_+fill_,0,64-fill_);compress();fill_=0;}
        std::memset(block_+fill_,0,56-fill_);
        for(int i=0;i<8;++i)block_[56+i]=static_cast<uint8_t>(bits>>(56-8*i));
        compress();
        for(int i=0;i<8;++i){out[4*i]=uint8_t(h_[i]>>24);out[4*i+1]=uint8_t(h_[i]>>16);out[4*i+2]=uint8_t(h_[i]>>8);out[4*i+3]=uint8_t(h_[i]);}
        return true;
    }
private:
    static uint32_t rotr(uint32_t x,int n){return (x>>n)|(x<<(32-n));}
    void compress() {
        static const uint32_t k[64]={
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for(int i=0;i<16;++i)w[i]=uint32_t(block_[4*i])<<24|uint32_t(block_[4*i+1])<<16|uint32_t(block_[4*i+2])<<8|block_[4*i+3];
        for(int i=16;i<64;++i) {
            const uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3),s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],h=h_[7];
        for(int i=0;i<64;++i) {
            const uint32_t t1=h+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            const uint32_t t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h_[0]+=a;h_[1]+=b;h_[2]+=c;h_[3]+=d;h_[4]+=e;h_[5]+=f;h_[6]+=g;h_[7]+=h;
    }
    static constexpr uint32_t init_[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint32_t h_[8];uint8_t block_[64]{};size_t fill_=0;uint64_t total_=0;
};
std::string posix_error(const char* action){return std::string(action)+" failed (errno "+std::to_string(errno)+")";}
// A descriptor at >=3 with close-on-exec, so it can never alias a std stream.
int high_fd(int fd) {
    if(fd<0)return -1;
    const int moved=fcntl(fd,F_DUPFD_CLOEXEC,3);const int saved=errno;close(fd);errno=saved;return moved;
}
void close_fd(int& fd){if(fd>=0)close(fd);fd=-1;}
uint32_t decode_status(int status){return WIFEXITED(status)?static_cast<uint32_t>(WEXITSTATUS(status)):128u+static_cast<uint32_t>(WTERMSIG(status));}
// One supervised child. Move-only. Destroying a still-running child kills its
// process group (the kill-on-close analogue) and reaps it for a bounded time.
class ChildProcess {
public:
    ChildProcess()=default;
    ChildProcess(ChildProcess&& o) noexcept{take(o);}
    ChildProcess& operator=(ChildProcess&& o) noexcept{if(this!=&o){release();take(o);}return *this;}
    ChildProcess(const ChildProcess&)=delete;ChildProcess& operator=(const ChildProcess&)=delete;
    ~ChildProcess(){release();}
    explicit operator bool()const{return pid_>0;}
    uint32_t id()const{return static_cast<uint32_t>(pid_);}
    uint64_t started_ms()const{return started_;}
    // Creates the sealed read-only request section, the writable result
    // section and the gated child. The child sets its own limits, reports
    // ready, then blocks until resume(); only then does it exec the worker.
    bool spawn(const ProtectedProcessOptions& o,const std::vector<uint8_t>& packet,std::string& error) {
        std::string path;
        if(!to_utf8(o.executable,path)){error="Protected native worker executable is not valid Unicode";return false;}
        request_fd_=high_fd(static_cast<int>(memfd_create("nodus-gl-request",MFD_CLOEXEC|MFD_ALLOW_SEALING)));
        if(request_fd_<0){error=posix_error("memfd_create request");return false;}
        for(size_t done=0;done<packet.size();) {
            const ssize_t n=write(request_fd_,packet.data()+done,packet.size()-done);
            if(n<0 && errno==EINTR)continue;
            if(n<=0){error=posix_error("Write request section");return false;}
            done+=static_cast<size_t>(n);
        }
        // Immutable for every holder, the analogue of the read-only duplicate.
        if(fcntl(request_fd_,F_ADD_SEALS,F_SEAL_SEAL|F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK)!=0){error=posix_error("Seal request section");return false;}
        result_fd_=high_fd(static_cast<int>(memfd_create("nodus-gl-result",MFD_CLOEXEC)));
        if(result_fd_<0){error=posix_error("memfd_create result");return false;}
        if(ftruncate(result_fd_,sizeof(protected_detail::WireResult))!=0){error=posix_error("Size result section");return false;}
        void* view=mmap(nullptr,sizeof(protected_detail::WireResult),PROT_READ|PROT_WRITE,MAP_SHARED,result_fd_,0);
        if(view==MAP_FAILED){error=posix_error("mmap result section");return false;}
        result_view_=view;std::memset(result_view_,0,sizeof(protected_detail::WireResult));
        int go[2]{-1,-1},ready[2]{-1,-1};
        if(pipe2(go,O_CLOEXEC)!=0){error=posix_error("pipe");return false;}
        go_r_=go[0];go_w_=go[1];
        if(pipe2(ready,O_CLOEXEC)!=0){error=posix_error("pipe");return false;}
        ready_r_=ready[0];int ready_w=ready[1];
        // Everything the child needs is built BEFORE fork: only async-signal-safe
        // calls run between fork and exec (the parent is multi-threaded).
        const std::string flag="--nodus-protected-gl-worker",request_text=std::to_string(request_fd_),result_text=std::to_string(result_fd_);
        char* argv[]{const_cast<char*>(path.c_str()),const_cast<char*>(flag.c_str()),
                     const_cast<char*>(request_text.c_str()),const_cast<char*>(result_text.c_str()),nullptr};
        struct rlimit data{static_cast<rlim_t>(o.process_memory_bytes),static_cast<rlim_t>(o.process_memory_bytes)};
        struct rlimit no_core{0,0};
        long open_max=sysconf(_SC_OPEN_MAX);if(open_max<64 || open_max>65536)open_max=65536;
        const int request_fd=request_fd_,result_fd=result_fd_,go_r=go_r_,go_w=go_w_,ready_r=ready_r_;
        const pid_t parent=getpid();
        started_=tick_ms();
        const pid_t pid=fork();
        if(pid<0){close(ready_w);error=posix_error("fork native worker");return false;}
        if(pid==0) {
            setpgid(0,0);                                   // own process group: terminate_group()
            if(prctl(PR_SET_PDEATHSIG,SIGKILL)!=0 || getppid()!=parent)_exit(120);   // kill-on-close
            if(setrlimit(RLIMIT_DATA,&data)!=0)_exit(121);  // process memory cap
            setrlimit(RLIMIT_CORE,&no_core);
            close(go_w);close(ready_r);
            const char mark='r';if(write(ready_w,&mark,1)!=1)_exit(122);
            close(ready_w);
            char gate=0;if(read(go_r,&gate,1)!=1)_exit(123);   // start gate (resume)
            // Only the two transport sections are inherited; std streams go to /dev/null.
            fcntl(request_fd,F_SETFD,0);fcntl(result_fd,F_SETFD,0);
            const int null=open("/dev/null",O_RDWR);
            if(null>=0){dup2(null,0);dup2(null,1);dup2(null,2);if(null>2)close(null);}
            for(int fd=3;fd<open_max;++fd)if(fd!=request_fd && fd!=result_fd)close(fd);
            execv(argv[0],argv);
            _exit(127);
        }
        pid_=pid;setpgid(pid,pid);                          // also from the parent: no startup race
        close(ready_w);close_fd(go_r_);
        return true;
    }
    // Waits for the child's own limit setup, then verifies containment from
    // outside: memory cap, own group and the still-closed start gate.
    bool contain(const ProtectedProcessOptions& o) {
        pollfd p{ready_r_,POLLIN,0};int rc=0;
        do{rc=poll(&p,1,5000);}while(rc<0 && errno==EINTR);
        char mark=0;
        if(rc!=1 || read(ready_r_,&mark,1)!=1 || mark!='r')return false;
        close_fd(ready_r_);
        struct rlimit actual{};
        return prlimit(pid_,RLIMIT_DATA,nullptr,&actual)==0 && actual.rlim_cur==o.process_memory_bytes &&
            actual.rlim_max==o.process_memory_bytes && getpgid(pid_)==pid_;
    }
    bool resume() {
        const char gate='g';ssize_t n=0;
        do{n=write(go_w_,&gate,1);}while(n<0 && errno==EINTR);
        close_fd(go_w_);return n==1;
    }
    // Idempotent and safe from the watchdog thread and the owner at once:
    // observation never reaps; only accounting()/release() reap.
    WaitResult wait(uint32_t ms) const {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(ms);
        for(;;) {
            {
                std::lock_guard<std::mutex> lock(*mutex_);
                if(reaped_)return WaitResult::Exited;
                siginfo_t info{};
                if(waitid(P_PID,static_cast<id_t>(pid_),&info,WEXITED|WNOHANG|WNOWAIT)==0){if(info.si_pid==pid_)return WaitResult::Exited;}
                else if(errno==ECHILD)return WaitResult::Exited;
                else if(errno!=EINTR)return WaitResult::Failed;
            }
            if(std::chrono::steady_clock::now()>=deadline)return WaitResult::Timeout;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    void terminate_group(uint32_t){
        std::lock_guard<std::mutex> lock(*mutex_);
        if(pid_>0 && !reaped_){kill(-pid_,SIGKILL);kill(pid_,SIGKILL);}
    }
    void terminate_process(uint32_t){
        std::lock_guard<std::mutex> lock(*mutex_);
        if(pid_>0 && !reaped_)kill(pid_,SIGKILL);
    }
    bool exit_code(uint32_t& code)const{
        std::lock_guard<std::mutex> lock(*mutex_);
        if(reaped_){code=decode_status(status_);return true;}
        siginfo_t info{};
        if(waitid(P_PID,static_cast<id_t>(pid_),&info,WEXITED|WNOHANG|WNOWAIT)!=0 || info.si_pid!=pid_)return false;
        code=info.si_code==CLD_EXITED?static_cast<uint32_t>(info.si_status):128u+static_cast<uint32_t>(info.si_status);return true;
    }
    // Reaps an exited child and reports its peak resident size (single-process
    // "job": the process peak is the job peak) and exit code.
    void accounting(ProtectedLinkOutcome& out) {
        std::lock_guard<std::mutex> lock(*mutex_);
        if(!reaped_ && pid_>0) {
            int status=0;rusage usage{};
            const pid_t r=wait4(pid_,&status,WNOHANG,&usage);
            if(r==pid_){status_=status;usage_=usage;reaped_=true;}
        }
        if(reaped_) {
            out.peak_process_bytes=static_cast<uint64_t>(usage_.ru_maxrss)*1024u;out.peak_job_bytes=out.peak_process_bytes;
            out.exit_code=decode_status(status_);
        }
    }
    void read_result(protected_detail::WireResult& receipt)const{std::memcpy(&receipt,result_view_,sizeof(receipt));}
    void close_io() {
        if(result_view_){munmap(result_view_,sizeof(protected_detail::WireResult));result_view_=nullptr;}
        close_fd(request_fd_);close_fd(result_fd_);close_fd(go_w_);close_fd(go_r_);close_fd(ready_r_);
    }
private:
    void take(ChildProcess& o) {
        pid_=std::exchange(o.pid_,0);started_=o.started_;request_fd_=std::exchange(o.request_fd_,-1);
        result_fd_=std::exchange(o.result_fd_,-1);go_r_=std::exchange(o.go_r_,-1);go_w_=std::exchange(o.go_w_,-1);
        ready_r_=std::exchange(o.ready_r_,-1);result_view_=std::exchange(o.result_view_,nullptr);
        mutex_=std::move(o.mutex_);o.mutex_.reset(new std::mutex);reaped_=o.reaped_;o.reaped_=false;
        status_=o.status_;usage_=o.usage_;
    }
    void release() {
        close_io();
        if(pid_<=0)return;
        if(!reaped_) {
            kill(-pid_,SIGKILL);kill(pid_,SIGKILL);
            for(int i=0;i<5000;++i){int s=0;const pid_t r=waitpid(pid_,&s,WNOHANG);if(r==pid_ || (r<0 && errno!=EINTR))break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));}
        }
        pid_=0;
    }
    pid_t pid_=0;uint64_t started_=0;
    int request_fd_=-1,result_fd_=-1,go_r_=-1,go_w_=-1,ready_r_=-1;void* result_view_=nullptr;
    std::unique_ptr<std::mutex> mutex_{new std::mutex};bool reaped_=false;int status_=0;rusage usage_{};
};
}
#endif
// ===== END PLATFORM REGION =====

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
bool valid_options(const ProtectedProcessOptions& o,std::string& error) {
    if(!platform::executable_ok(o.executable)) {
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
}

bool protected_source_digest(const ProtectedLinkRequest& request,ProtectedSourceDigest& digest,std::string& error) {
    digest={};size_t bytes=0;if(!source_bounds(request,bytes,error))return false;
    platform::Sha256 hash;
    if(!hash.available()) {error="SHA-256 unavailable";return false;}
    const uint64_t count=request.source_count;hash.update(&count,sizeof(count));
    for(size_t i=0;i<request.source_count;++i) {
        const auto& s=request.sources[i];const uint64_t label=s.label?std::strlen(s.label):0,source=s.source_bytes;
        hash.update(&label,sizeof(label));hash.update(s.label,static_cast<size_t>(label));
        hash.update(&source,sizeof(source));hash.update(s.source,static_cast<size_t>(source));
    }
    const bool ok=hash.finish(digest.data(),digest.size());
    if(!ok)error="SHA-256 source commitment failed";return ok;
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
    platform::ChildProcess pending_process; // job/containment owner included
    ProtectedLinkOutcome pending_outcome;
};
ProtectedOpenGLProcess::ProtectedOpenGLProcess(ProtectedProcessOptions options):impl_(new Impl(std::move(options))){}
ProtectedOpenGLProcess::~ProtectedOpenGLProcess() {
    impl_->cancellation.store(true);
    if(impl_->supervisor.joinable())impl_->supervisor.join();
    if(impl_->pending_process) {
        impl_->pending_process.terminate_group(0xe001);
        impl_->pending_process.wait(5000);
        // The last containment owner still enforces kill-on-close if termination is
        // delayed in the kernel. Never claim that such cleanup was observed.
    }
}
bool ProtectedOpenGLProcess::authorize_once(const ProtectedLinkAuthorization& authorization,std::string& reason) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(impl_->running){reason="Native child owner is already running";return false;}
    if(!valid_options(impl_->options,reason))return false;
    if(impl_->pending_process || impl_->armed) {reason="Existing request or unreaped child owns this provider";return false;}
    if(!authorization.host_id || !authorization.contract_id || !authorization.generation ||
       std::all_of(authorization.source_digest.begin(),authorization.source_digest.end(),[](uint8_t x){return x==0;})) {
        reason="Authorization must bind one exact loan and source digest";return false;
    }
    impl_->authorization=authorization;impl_->armed=true;reason.clear();return true;
}
bool ProtectedOpenGLProcess::ready(std::string& reason) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(impl_->running){reason="Native child owner is already running";return false;}
    if(!valid_options(impl_->options,reason))return false;
    if(impl_->pending_process){reason="The prior native child has not been reaped";return false;}
    if(!impl_->armed){reason="A new explicit one-link authorization is required";return false;}
    reason.clear();return true;
}
bool ProtectedOpenGLProcess::available(std::string& reason) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if(impl_->running){reason="Native child owner is already running";return false;}
    if(impl_->pending_process){reason="The prior native child has not been reaped";return false;}
    return valid_options(impl_->options,reason);
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
    if(!platform::memory_available(o.minimum_available_memory_bytes))return refusal("Available system RAM is below the required floor or cannot be queried");
    std::vector<uint8_t> packet;
    if(!protected_detail::encode_request(request,o,packet,error))return refusal(error);
    platform::ChildProcess child;
    if(!child.spawn(o,packet,error))return refusal(error);
    const uint64_t started=child.started_ms();
    ProtectedLinkOutcome out;out.process_id=child.id();out.requested_luid=o.adapter_luid;
    bool resumed=false;
    std::atomic<bool> watchdog_started=false,stop=false;
    std::atomic<uint32_t> watchdog_reason=0;
    std::thread watchdog;
    struct WatchdogJoin {
        std::thread& thread;std::atomic<bool>& stop;
        ~WatchdogJoin(){stop.store(true,std::memory_order_release);if(thread.joinable())thread.join();}
    } watchdog_join{watchdog,stop};
    const auto finish=[&](ProtectedLinkOutcome result) {
        result.host_id=request.host_id;result.contract_id=request.contract_id;result.generation=request.generation;
        result.elapsed_ms=platform::tick_ms()-started;
        result.child_reaped=child.wait(0)==platform::WaitResult::Exited;
        result.resources_released=result.child_reaped;
        child.accounting(result);
        if(!result.child_reaped) {
            result.status=ProtectedLinkStatus::CleanupUncertain;result.outcome_uncertain=true;result.linked=false;
            result.message+="; child exit is unconfirmed, owner retained; no replay";
            impl_->pending_outcome=result;child.close_io();impl_->pending_process=std::move(child);
        }
        return result;
    };
    try {
    fault(1); // suspended, not assigned: exception must still terminate/reap it
    if(!child.contain(o)) {
        out.status=ProtectedLinkStatus::SupervisionFailed;out.message="Suspended child could not be assigned to verified memory-capped JobObject";
        child.terminate_process(0xe002);child.wait(5000);return finish(out);
    }
    out.containment_verified=true;
    try {
        watchdog=std::thread([&] {
            watchdog_started.store(true,std::memory_order_release);
            while(!stop.load(std::memory_order_acquire)) {
                const auto wait=child.wait(o.poll_ms);
                if(wait==platform::WaitResult::Exited)return;
                uint32_t reason=0;
                if(wait!=platform::WaitResult::Timeout)reason=3;
                else if(impl_->cancellation.load())reason=4;
                else if(platform::tick_ms()-started>=o.wall_timeout_ms)reason=1;
                else if(!platform::memory_available(o.minimum_available_memory_bytes))reason=2;
                if(reason) {watchdog_reason.store(reason);child.terminate_group(0xe010+reason);return;}
            }
        });
    } catch(...) {
        out.status=ProtectedLinkStatus::SupervisionFailed;out.message="Independent watchdog could not start; suspended child terminated";
        child.terminate_group(0xe003);child.wait(5000);return finish(out);
    }
    while(!watchdog_started.load(std::memory_order_acquire))platform::yield_thread();
    // No child instruction runs until caps, membership and watchdog exist.
    if(!watchdog_reason.load() && platform::memory_available(o.minimum_available_memory_bytes) &&
       child.resume())resumed=true;
    if(!resumed) {
        out.status=ProtectedLinkStatus::SupervisionFailed;out.message="Resume refused or failed after supervised admission";
        child.terminate_group(0xe004);
    }
    fault(2); // joinable watchdog: unwinding must never terminate the server
    const auto wait=child.wait(o.wall_timeout_ms+5000);
    if(wait!=platform::WaitResult::Exited) {
        child.terminate_group(0xe005);child.terminate_process(0xe005);
        child.wait(5000);
    }
    stop.store(true,std::memory_order_release);watchdog.join();
    if(!resumed)return finish(out);
    out.attempted=true;out.outcome_uncertain=true; // conservative until a final receipt is verified
    const uint32_t stop_reason=watchdog_reason.load();
    if(stop_reason || wait!=platform::WaitResult::Exited) {
        out.status=stop_reason==4?ProtectedLinkStatus::Cancelled:
            stop_reason==2?ProtectedLinkStatus::MemoryFloor:ProtectedLinkStatus::TimedOut;
        out.message=stop_reason==4?"Native child cancelled and terminated; no replay":
            stop_reason==2?"Native child terminated at the system RAM floor; no replay":
            "Native child exceeded its supervised deadline or wait failed; no replay";
        return finish(out);
    }
    uint32_t exit_code=0;
    if(!child.exit_code(exit_code) || exit_code!=0) {
        out.status=ProtectedLinkStatus::ChildFailed;out.message="Native child exited without a successful protocol completion; no replay";
        return finish(out);
    }
    protected_detail::WireResult receipt{};child.read_result(receipt);
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
        child.terminate_group(0xe006);
        if(child){child.terminate_process(0xe006);child.wait(5000);}
        out.host_id=request.host_id;out.contract_id=request.contract_id;out.generation=request.generation;
        out.attempted=resumed;out.linked=false;out.outcome_uncertain=resumed;
        out.child_reaped=child && child.wait(0)==platform::WaitResult::Exited;
        out.resources_released=out.child_reaped;out.status=ProtectedLinkStatus::SupervisionFailed;
        out.message.clear(); // avoid allocating on a potential allocation failure
        if(!out.child_reaped) {
            out.status=ProtectedLinkStatus::CleanupUncertain;out.outcome_uncertain=true;
            child.close_io();
            impl_->pending_outcome=std::move(out);impl_->pending_process=std::move(child);
            // If the copy itself throws, the outer catch still leaves the OS
            // owner retained, and authorize_once/ready remain closed.
            return impl_->pending_outcome;
        }
        child.accounting(out);return out;
    }
}

ProtectedLinkOutcome ProtectedOpenGLProcess::link(const ProtectedLinkRequest& request) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto refused=[&](const std::string& message) {
        auto out=refusal(message);out.host_id=request.host_id;out.contract_id=request.contract_id;
        out.generation=request.generation;return out;
    };
    if(impl_->running)return refused("Provider already owns an active child; no new request admitted");
    if(impl_->pending_process)return refused("Provider retains an unreaped child; no new request admitted");
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
    if(impl_->running)return false;
    if(!impl_->pending_process)return false;
    out=impl_->pending_outcome;
    if(impl_->pending_process.wait(0)!=platform::WaitResult::Exited)return true;
    impl_->pending_process.accounting(out);
    out.child_reaped=true;out.resources_released=true;out.outcome_uncertain=true;
    out.message="Original native child exit now verified; link outcome remains uncertain; no replay";
    impl_->outcome=out;
    impl_->pending_process=platform::ChildProcess();impl_->pending_outcome={};return true;
}
} // namespace nodus::gpu
