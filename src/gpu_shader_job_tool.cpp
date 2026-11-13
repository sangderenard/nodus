#include "gpu_shader_job_tool.h"
#include "kernel_vulkan_runner.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
#include <limits>

namespace nodus::gpu {
GpuShaderJobTool::GpuShaderJobTool(TensorStateLibraryTool& library,
    std::size_t device,GpuShaderJobContract contract)
    :library_(library),device_(device),contract_(std::move(contract)),
     id_(contract_.id+".vulkan."+std::to_string(device)){}

void GpuShaderJobTool::execute_stack(ToolStackContext& context){
    context.succeeded=false;
    if(!context.stack.raw)return;
    auto& stack=*context.stack.raw;
    nodus::tensors::AbstractTensorHandle input{};
    if(!raw_stack_pop_abstract_tensor(stack,input))return;
    const auto total_started=std::chrono::steady_clock::now();
    std::array<std::uint64_t,5> request{};
    NodusTensorDesc desc{};
    std::uint64_t status=1,elapsed_ns=0,lease_ns=0;
    bool side_effect_possible=false;
    const auto count=contract_.minimum_binding_bytes.size();
    if(nodus_tensor_describe(input.id,&desc)==NODUS_OK&&
       desc.dtype==NODUS_DTYPE_U64&&desc.rank==1&&
       (desc.shape[0]==4||desc.shape[0]==5)&&
       nodus_tensor_read(input.id,0,request.data(),desc.shape[0]*sizeof(std::uint64_t))==
           desc.shape[0]*sizeof(std::uint64_t)&&
       request[0]&&request[1]&&request[2]&&
       request[2]<=contract_.maximum_invocations&&
       (contract_.local_size_y==0 ? desc.shape[0]==4 :
           (desc.shape[0]==5&&request[4]&&
            request[4]<=contract_.maximum_invocations_y))&&
       request[1]<=UINT64_MAX-(contract_.binding_offsets.empty()?count-1:
          *std::max_element(contract_.binding_offsets.begin(),
                            contract_.binding_offsets.end()))&&
       device_<library_.device_count()&&count==contract_.writable.size()){
        const auto lease_started=std::chrono::steady_clock::now();
        std::vector<TensorStateLibraryTool::GpuLoan> loans;
        std::vector<kernels::VulkanBackendBinding> bindings;
        loans.reserve(count);bindings.reserve(count);
        bool admitted=true;
        for(std::size_t i=0;i<count;i++){
            const auto address=request[1]+(contract_.binding_offsets.empty()?i:
                                           contract_.binding_offsets[i]);
            auto loan=library_.borrow_gpu(request[0],address,device_,
                                          contract_.writable[i]!=0);
            if(!loan||loan.bytes()<contract_.minimum_binding_bytes[i]){
                admitted=false;break;
            }
            bindings.push_back({loan.handle(),0,
                static_cast<std::size_t>(loan.bytes())});
            loans.push_back(std::move(loan));
        }
        lease_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
            std::chrono::nanoseconds>(std::chrono::steady_clock::now()-lease_started).count());
        if(admitted){
            for(const auto& reset:contract_.reset_before_dispatch){
                side_effect_possible=true;
                const auto& binding=bindings[reset.binding];
                const auto* vtable=gp_mem_backend_get_vtable(binding.backend);
                std::vector<std::uint32_t> words(reset.count,reset.value);
                if(!vtable||!vtable->copy_to_backend||
                   !vtable->copy_to_backend(binding.backend,
                       static_cast<std::size_t>(reset.offset_words)*4,
                       words.data(),words.size()*4)){
                    admitted=false;break;
                }
            }
        }
        if(admitted){
            std::string error;
            const auto started=std::chrono::steady_clock::now();
            const auto groups=static_cast<std::uint32_t>(
                1+(request[2]-1)/contract_.local_size_x);
            side_effect_possible=true;
            const bool okay=contract_.local_size_y==0
                ? kernels::dispatch_spirv_backends(contract_.spirv,
                    contract_.entry.c_str(),bindings,groups,&error)
                : kernels::dispatch_spirv_backends_2d(contract_.spirv,
                    contract_.entry.c_str(),bindings,groups,
                    static_cast<std::uint32_t>(
                        1+(request[4]-1)/contract_.local_size_y),&error);
            elapsed_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count());
            if(!okay)std::fprintf(stderr,"[shader-job] %s generation=%llu: %s\n",
                id_.c_str(),static_cast<unsigned long long>(request[3]),
                error.c_str());
            status=okay?0:2;
        }else status=3;
    }
    const auto total_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
        std::chrono::nanoseconds>(std::chrono::steady_clock::now()-total_started).count());
    if(status==0)library_.record_work_completion(
        TensorStateLibraryTool::ExecutionApi::Vulkan,device_,
        contract_.program_id,request[2],
        lease_ns,elapsed_ns,total_ns);
    const std::uint64_t shape[1]={9};
    const std::array<std::uint64_t,9> receipt{
        status,static_cast<std::uint64_t>(device_),request[2],elapsed_ns,
        request[3],request[1],lease_ns,total_ns,status==0?1ull:0ull};
    const auto output=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    if(!output||nodus_tensor_write(output,0,receipt.data(),sizeof(receipt))!=sizeof(receipt)||
       !raw_stack_push_abstract_tensor(stack,{output})){
        if(output)nodus_tensor_destroy(output);
        if(!side_effect_possible)raw_stack_push_abstract_tensor(stack,input);
        else std::fprintf(stderr,
            "[shader-job] %s receipt publication failed after possible device effect; input consumed, do not replay\n",
            id_.c_str());
        return;
    }
    context.succeeded=true;
}

bool register_gpu_shader_job(ToolRegistry& registry,
    TensorStateLibraryTool& library,const GpuShaderJobContract& contract){
    if(contract.id.empty()||!contract.program_id||contract.entry.empty()||
       contract.spirv.empty()||
       contract.local_size_x==0||contract.maximum_invocations==0||
       ((contract.local_size_y==0)!=(contract.maximum_invocations_y==0))||
       contract.minimum_binding_bytes.empty()||
       contract.minimum_binding_bytes.size()!=contract.writable.size()||
       contract.minimum_binding_bytes.size()>32||
       (!contract.binding_offsets.empty()&&
        contract.binding_offsets.size()!=contract.minimum_binding_bytes.size()))return false;
    if(!contract.binding_offsets.empty()){
        auto offsets=contract.binding_offsets;
        std::sort(offsets.begin(),offsets.end());
        if(std::adjacent_find(offsets.begin(),offsets.end())!=offsets.end())return false;
    }
    for(const auto bytes:contract.minimum_binding_bytes)if(!bytes)return false;
    for(const auto& reset:contract.reset_before_dispatch){
        if(reset.binding>=contract.writable.size()||
           !contract.writable[reset.binding]||!reset.count||
           (static_cast<std::uint64_t>(reset.offset_words)+reset.count)*4>
               contract.minimum_binding_bytes[reset.binding])return false;
    }
    for(std::size_t device=0;device<library.device_count();device++){
        ToolRegistry::Entry item;
        item.id=contract.id+".vulkan."+std::to_string(device);
        item.name="GPU shader job "+contract.id;
        item.caps=ToolCaps::None;
        item.factory=[&library,device,contract]{
            return std::unique_ptr<ITool,std::function<void(ITool*)>>(
                new GpuShaderJobTool(library,device,contract),
                [](ITool* value){delete value;});
        };
        if(!registry.register_tool(std::move(item)))return false;
    }
    return true;
}
} // namespace nodus::gpu
