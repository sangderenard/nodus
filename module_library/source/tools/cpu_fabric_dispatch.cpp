// cpu_fabric_dispatch: a CPU fabric law dispatch row over the tensor state library (include/cpu_fabric_dispatch_tool.h).
// The row tool's source, in the module library: the Menu lists it, the canvas compiler inlines it, it builds as its own
// plugin DLL, and the image registers it from it (src/cpu_fabric_dispatch_tool.cpp).
// Its settings are its saved state (serialize/deserialize); a row with no library refuses.
#include "cpu_fabric_dispatch_tool.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
// ---- platform region: run-time library loading (the only OS calls in this file) ----
#if defined(_WIN32)
#include <windows.h>
namespace {
void* platform_library_open(const char* path){
    return LoadLibraryExA(path,nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}
void* platform_library_symbol(void* module,const char* name){
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module),name));
}
void platform_library_close(void* module){
    if(module)FreeLibrary(static_cast<HMODULE>(module));
}
}
#else
#include <dlfcn.h>
namespace {
void* platform_library_open(const char* path){return dlopen(path,RTLD_NOW|RTLD_LOCAL);}
void* platform_library_symbol(void* module,const char* name){return dlsym(module,name);}
void platform_library_close(void* module){if(module)dlclose(module);}
}
#endif
// ---- end platform region ----

namespace nodus::gpu {
CpuFabricDispatchTool::CpuFabricDispatchTool(
    TensorStateLibraryTool& library,std::string dll,std::string entry,
    std::uint64_t batch,std::uint32_t steps_per_loadout,
    std::uint64_t program_id):library_(&library),
    dll_(std::move(dll)),entry_name_(std::move(entry)),batch_(batch),
    steps_per_loadout_(steps_per_loadout),program_id_(program_id){}
CpuFabricDispatchTool::CpuFabricDispatchTool()=default;
void CpuFabricDispatchTool::configure(std::string dll,std::string entry,
    std::uint64_t batch,std::uint32_t steps_per_loadout,std::uint64_t program_id){
    dll_=std::move(dll);entry_name_=std::move(entry);batch_=batch;
    steps_per_loadout_=steps_per_loadout;program_id_=program_id;
}
// Saved state: the row's settings (magic "CFD1", version 1, DLL path, entry name, batch, steps per loadout, program ID).
namespace {
constexpr std::uint32_t kCpuFabricSettingsMagic=0x31444643,kCpuFabricSettingsVersion=1;
}
bool CpuFabricDispatchTool::serialize(OutputArchive& out) const{
    auto u32=[&](std::uint32_t v){return out.write(&v,sizeof(v));};
    auto u64=[&](std::uint64_t v){return out.write(&v,sizeof(v));};
    auto str=[&](const std::string& v){return u32(static_cast<std::uint32_t>(v.size()))&&(v.empty()||out.write(v.data(),v.size()));};
    auto vec=[&](const auto& v){
        if(!u64(v.size()))return false;
        for(const auto x:v)if(!u64(static_cast<std::uint64_t>(x)))return false;
        return true;
    };
    (void)str;(void)vec;
    return u32(kCpuFabricSettingsMagic)&&u32(kCpuFabricSettingsVersion)&&str(dll_)&&str(entry_name_)&&
           u64(batch_)&&u64(steps_per_loadout_)&&u64(program_id_);
}
bool CpuFabricDispatchTool::deserialize(InputArchive& in){
    auto u32=[&](std::uint32_t& v){return in.read(&v,sizeof(v));};
    auto u64=[&](std::uint64_t& v){return in.read(&v,sizeof(v));};
    auto str=[&](std::string& v,std::uint32_t limit){
        std::uint32_t n=0;
        if(!u32(n)||n>limit)return false;
        v.resize(n);
        return n==0||in.read(v.data(),n);
    };
    auto vec=[&](auto& v,std::uint64_t limit){
        std::uint64_t n=0;
        if(!u64(n)||n>limit)return false;
        v.resize(static_cast<std::size_t>(n));
        for(auto& x:v){
            std::uint64_t raw=0;
            using Element=std::decay_t<decltype(x)>;
            if(!u64(raw)||raw>(std::numeric_limits<Element>::max)())return false;
            x=static_cast<Element>(raw);
        }
        return true;
    };
    (void)str;(void)vec;
    std::uint32_t magic=0,version=0;
    std::string dll,entry;
    std::uint64_t batch=0,steps=0,program=0;
    if(!u32(magic)||!u32(version)||magic!=kCpuFabricSettingsMagic||version!=kCpuFabricSettingsVersion||
       !str(dll,4096)||!str(entry,4096)||!u64(batch)||!u64(steps)||steps>4096||!u64(program))return false;
    configure(std::move(dll),std::move(entry),batch,static_cast<std::uint32_t>(steps),program);
    return true;
}
void CpuFabricDispatchTool::initialize(const ToolInitContext& ctx){
    if(ctx.serialized_path&&ctx.serialized_path[0]!='\0'){
        FileInputArchive in(ctx.serialized_path);
        if(in.ok())deserialize(in);
    }
    if(module_)return;
    module_=platform_library_open(dll_.c_str());
    if(!module_)return;
    entry_=reinterpret_cast<void (*)(void**,int32_t*)>(
        platform_library_symbol(module_,entry_name_.c_str()));
    reset_=reinterpret_cast<void (*)()>(platform_library_symbol(module_,"turing_validation_error_reset"));
    take_=reinterpret_cast<std::uint32_t (*)()>(
        platform_library_symbol(module_,"turing_validation_error_take"));
    if(!entry_||((reset_==nullptr)!=(take_==nullptr)))shutdown();
}
void CpuFabricDispatchTool::shutdown(){
    platform_library_close(module_);
    module_=nullptr;entry_=nullptr;reset_=nullptr;take_=nullptr;
}
void CpuFabricDispatchTool::execute_stack(ToolStackContext& context){
    context.succeeded=false;
    if(!context.stack.raw||!library_)return;
    auto& stack=*context.stack.raw;
    nodus::tensors::AbstractTensorHandle input{};
    if(!raw_stack_pop_abstract_tensor(stack,input))return;
    const auto total_started=std::chrono::steady_clock::now();
    std::array<std::uint64_t,4> request{};
    NodusTensorDesc desc{};
    std::uint64_t status=1,elapsed_ns=0,lease_ns=0;
    if(entry_&&nodus_tensor_describe(input.id,&desc)==NODUS_OK&&
       desc.dtype==NODUS_DTYPE_U64&&desc.rank==1&&desc.shape[0]==request.size()&&
       nodus_tensor_read(input.id,0,request.data(),sizeof(request))==sizeof(request)&&
       request[0]&&request[1]&&request[2]==batch_&&request[1]<=UINT64_MAX-25){
        const auto lease_started=std::chrono::steady_clock::now();
        std::vector<TensorStateLibraryTool::CpuLoan> leases;
        leases.reserve(27);
        std::array<void*,26> buffers{};
        std::uint32_t row_stride=0;
        bool admitted=true;
        for(std::uint64_t i=0;i<27;i++){
            auto lease=library_->borrow_cpu(request[0],request[1]+i,
                                           i==24||i==25||
                                           (steps_per_loadout_>1&&(i==17||i==23)));
            const auto expected=i==26?sizeof(std::uint32_t):batch_*sizeof(double);
            if(!lease||lease.bytes()<expected){
                std::fprintf(stderr,"CPU fabric lease refused base=%llu feed=%llu bytes=%zu expected=%llu\n",
                    static_cast<unsigned long long>(request[1]),
                    static_cast<unsigned long long>(i),lease.bytes(),
                    static_cast<unsigned long long>(expected));
                admitted=false;break;
            }
            if(i==26){
                if(steps_per_loadout_>1){
                    if(lease.bytes()<2*sizeof(std::uint32_t)){
                        admitted=false;break;
                    }
                    row_stride=static_cast<const std::uint32_t*>(lease.data())[1];
                    if(row_stride<3||row_stride>batch_){admitted=false;break;}
                }
            }else buffers[i]=lease.data();
            leases.push_back(std::move(lease));
        }
        lease_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
            std::chrono::nanoseconds>(std::chrono::steady_clock::now()-lease_started).count());
        if(admitted){
            const auto started=std::chrono::steady_clock::now();
            std::uint32_t validation=0;
            for(std::uint32_t step=0;step<steps_per_loadout_;step++){
                if(steps_per_loadout_>1){
                    auto* position=static_cast<const double*>(buffers[17]);
                    const auto* left=static_cast<const double*>(buffers[5]);
                    const auto* right=static_cast<const double*>(buffers[6]);
                    const auto* up=static_cast<const double*>(buffers[7]);
                    const auto* down=static_cast<const double*>(buffers[8]);
                    auto* neighbor_left=static_cast<double*>(buffers[11]);
                    auto* neighbor_right=static_cast<double*>(buffers[12]);
                    auto* neighbor_up=static_cast<double*>(buffers[13]);
                    auto* neighbor_down=static_cast<double*>(buffers[14]);
                    for(std::uint64_t lane=0;lane<batch_;lane++){
                        neighbor_left[lane]=left[lane]!=0&&lane>=3?
                            position[lane-3]:0;
                        neighbor_right[lane]=right[lane]!=0&&lane+3<batch_?
                            position[lane+3]:0;
                        neighbor_up[lane]=up[lane]!=0&&lane>=row_stride?
                            position[lane-row_stride]:0;
                        neighbor_down[lane]=down[lane]!=0&&lane+row_stride<batch_?
                            position[lane+row_stride]:0;
                    }
                }
                if(reset_)reset_();
                entry_(buffers.data(),nullptr);
                validation=take_?take_():0;
                if(validation)break;
                if(steps_per_loadout_>1){
                    std::memcpy(buffers[17],buffers[24],batch_*sizeof(double));
                    std::memcpy(buffers[23],buffers[25],batch_*sizeof(double));
                }
            }
            elapsed_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count());
            status=validation?2:0;
        }else status=3;
    }
    const auto total_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
        std::chrono::nanoseconds>(std::chrono::steady_clock::now()-total_started).count());
    if(status==0)library_->record_work_completion(
        TensorStateLibraryTool::ExecutionApi::Cpu,0,program_id_,request[2],
        lease_ns,elapsed_ns,total_ns);
    const std::uint64_t shape[1]={9};
    const std::array<std::uint64_t,9> receipt{
        status,UINT64_MAX,request[2],elapsed_ns,request[3],request[1],lease_ns,total_ns,
        status==0?steps_per_loadout_:0};
    const auto output=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    if(!output||nodus_tensor_write(output,0,receipt.data(),sizeof(receipt))!=sizeof(receipt)||
       !raw_stack_push_abstract_tensor(stack,{output})){
        if(output)nodus_tensor_destroy(output);
        raw_stack_push_abstract_tensor(stack,input);return;
    }
    context.succeeded=true;
}

} // namespace nodus::gpu

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::gpu::CpuFabricDispatchTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
