#include "cpu_fabric_dispatch_tool.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nodus::gpu {
CpuFabricDispatchTool::CpuFabricDispatchTool(
    TensorStateLibraryTool& library,std::string dll,std::string entry,
    std::uint64_t batch,std::uint32_t steps_per_loadout,
    std::uint64_t program_id):library_(library),
    dll_(std::move(dll)),entry_name_(std::move(entry)),batch_(batch),
    steps_per_loadout_(steps_per_loadout),program_id_(program_id){}
void CpuFabricDispatchTool::initialize(const ToolInitContext&){
    if(module_)return;
#ifdef _WIN32
    module_=LoadLibraryExA(dll_.c_str(),nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
    module_=dlopen(dll_.c_str(),RTLD_NOW|RTLD_LOCAL);
#endif
    if(!module_)return;
#ifdef _WIN32
    entry_=reinterpret_cast<void (*)(void**,int32_t*)>(
        GetProcAddress(static_cast<HMODULE>(module_),entry_name_.c_str()));
    reset_=reinterpret_cast<void (*)()>(GetProcAddress(
        static_cast<HMODULE>(module_),"turing_validation_error_reset"));
    take_=reinterpret_cast<std::uint32_t (*)()>(GetProcAddress(
        static_cast<HMODULE>(module_),"turing_validation_error_take"));
#else
    entry_=reinterpret_cast<void (*)(void**,int32_t*)>(
        dlsym(module_,entry_name_.c_str()));
    reset_=reinterpret_cast<void (*)()>(dlsym(module_,"turing_validation_error_reset"));
    take_=reinterpret_cast<std::uint32_t (*)()>(
        dlsym(module_,"turing_validation_error_take"));
#endif
    if(!entry_||((reset_==nullptr)!=(take_==nullptr)))shutdown();
}
void CpuFabricDispatchTool::shutdown(){
#ifdef _WIN32
    if(module_)FreeLibrary(static_cast<HMODULE>(module_));
#else
    if(module_)dlclose(module_);
#endif
    module_=nullptr;entry_=nullptr;reset_=nullptr;take_=nullptr;
}
void CpuFabricDispatchTool::execute_stack(ToolStackContext& context){
    context.succeeded=false;
    if(!context.stack.raw)return;
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
            auto lease=library_.borrow_cpu(request[0],request[1]+i,
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
    if(status==0)library_.record_work_completion(
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
bool register_cpu_fabric_dispatcher(ToolRegistry& registry,
    TensorStateLibraryTool& library,const std::string& dll,
    const std::string& entry,std::uint64_t batch,
    std::uint32_t steps_per_loadout,std::uint64_t program_id){
    if(dll.empty()||entry.empty()||!batch||!steps_per_loadout||
       steps_per_loadout>4096||!program_id)return false;
    ToolRegistry::Entry item;
    item.id="nodus.fabric.dispatch.cpu";item.name="CPU fabric dispatch";
    item.caps=ToolCaps::None;
    item.factory=[&library,dll,entry,batch,steps_per_loadout,program_id]{
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new CpuFabricDispatchTool(library,dll,entry,batch,
                steps_per_loadout,program_id),
            [](ITool* value){delete value;});
    };
    return registry.register_tool(std::move(item));
}
} // namespace nodus::gpu
