#include "gpu_fabric_dispatch_tool.h"
#include "kernel_vulkan_runner.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <string>

namespace nodus::gpu {

GpuFabricDispatchTool::GpuFabricDispatchTool(
    TensorStateLibraryTool& library,std::size_t device,
    std::vector<std::uint32_t> shader,std::uint32_t local_size,
    GpuShaderWorkContract contract)
    :library_(library),device_(device),shader_(std::move(shader)),
     local_size_(local_size),contract_(contract),
     id_("nodus.fabric.dispatch.vulkan."+std::to_string(device)){}

void GpuFabricDispatchTool::execute_stack(ToolStackContext& context){
    context.succeeded=false;
    if(!context.stack.raw)return;
    auto& stack=*context.stack.raw;
    nodus::tensors::AbstractTensorHandle input{};
    if(!raw_stack_pop_abstract_tensor(stack,input))return;
    const auto total_started=std::chrono::steady_clock::now();
    std::array<std::uint64_t,4> request{};
    NodusTensorDesc desc{};
    std::uint64_t status=1,elapsed_ns=0,lease_ns=0;
    std::uint32_t selected_steps=1;
    if(nodus_tensor_describe(input.id,&desc)==NODUS_OK&&
       desc.dtype==NODUS_DTYPE_U64&&desc.rank==1&&desc.shape[0]==request.size()&&
       nodus_tensor_read(input.id,0,request.data(),sizeof(request))==sizeof(request)&&
       request[0]&&request[1]&&request[2]&&local_size_&&
       request[2]<=65535ull*local_size_&&
       request[1]<=UINT64_MAX-26&&device_<library_.device_count()&&!shader_.empty()&&
       contract_.steps_per_loadout>=1&&contract_.steps_per_loadout<=4096){
        selected_steps=contract_.target_completion_ns?1:contract_.steps_per_loadout;
        if(contract_.target_completion_ns&&step_samples_){
            double total=0;
            for(std::size_t i=0;i<step_samples_;i++)total+=recent_step_ns_[i];
            const double average=total/static_cast<double>(step_samples_);
            if(average>0)selected_steps=std::clamp(
                static_cast<std::uint32_t>(std::min<double>(
                    contract_.target_completion_ns/average,
                    contract_.steps_per_loadout)),
                1u,contract_.steps_per_loadout);
        }
        const auto lease_started=std::chrono::steady_clock::now();
        std::vector<TensorStateLibraryTool::GpuLoan> leases;
        leases.reserve(27);
        std::vector<kernels::VulkanBackendBinding> bindings;
        bindings.reserve(27);
        bool admitted=true;
        for(std::uint64_t i=0;i<27;i++){
            auto lease=library_.borrow_gpu(request[0],request[1]+i,device_,
                                            i==24||i==25||
                                            (contract_.resident_state_feedback &&
                                             (i==17||i==23)));
            const std::uint64_t expected=i==26?sizeof(std::uint32_t):request[2]*sizeof(double);
            if(!lease||lease.bytes()<expected){
                std::fprintf(stderr,
                    "[fabric-dispatch] device=%zu generation=%llu address=%llu feed=%llu lease refused bytes=%llu expected=%llu\n",
                    device_,static_cast<unsigned long long>(request[3]),
                    static_cast<unsigned long long>(request[1]+i),
                    static_cast<unsigned long long>(i),
                    static_cast<unsigned long long>(lease.bytes()),
                    static_cast<unsigned long long>(expected));
                admitted=false;break;
            }
            bindings.push_back({lease.handle(),0,static_cast<std::size_t>(expected)});
            leases.push_back(std::move(lease));
        }
        lease_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
            std::chrono::nanoseconds>(std::chrono::steady_clock::now()-lease_started).count());
        if(admitted){
            const auto started=std::chrono::steady_clock::now();
            std::string error;
            // The runner waits for fence completion. A failure after submit
            // has an unknown outcome; callers must not replay this generation.
            const auto groups=static_cast<std::uint32_t>(
                (request[2]+local_size_-1)/local_size_);
            const bool okay=contract_.resident_state_feedback
                ? kernels::dispatch_spirv_resident_loop(shader_,"main",bindings,
                    groups,selected_steps,17,23,24,25,&error)
                : kernels::dispatch_spirv_backends(shader_,"main",bindings,
                    groups,&error);
            elapsed_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count());
            if(!okay)std::fprintf(stderr,"[fabric-dispatch] device=%zu generation=%llu: %s\n",
                device_,static_cast<unsigned long long>(request[3]),error.c_str());
            status=okay?0:2;
            if(okay&&contract_.resident_state_feedback){
                recent_step_ns_[step_window_next_]=
                    static_cast<double>(elapsed_ns)/selected_steps;
                step_window_next_=(step_window_next_+1)%recent_step_ns_.size();
                step_samples_=std::min(step_samples_+1,recent_step_ns_.size());
            }
        }else status=3;
    }
    const auto total_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
        std::chrono::nanoseconds>(std::chrono::steady_clock::now()-total_started).count());
    if(status==0){
        library_.record_completion(device_,lease_ns,elapsed_ns,total_ns);
        library_.record_work_completion(TensorStateLibraryTool::ExecutionApi::Vulkan,
            device_,contract_.program_id,request[2],lease_ns,elapsed_ns,total_ns);
    }
    const std::uint64_t shape[1]={9};
    std::array<std::uint64_t,9> receipt{status,static_cast<std::uint64_t>(device_),
        request[2],elapsed_ns,request[3],request[1],lease_ns,total_ns,
        status==0?selected_steps:0};
    auto output=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    if(!output||nodus_tensor_write(output,0,receipt.data(),sizeof(receipt))!=sizeof(receipt)||
       !raw_stack_push_abstract_tensor(stack,{output})){
        if(output)nodus_tensor_destroy(output);
        raw_stack_push_abstract_tensor(stack,input);return;
    }
    context.succeeded=true;
}

bool register_gpu_fabric_dispatchers(ToolRegistry& registry,
    TensorStateLibraryTool& library,const std::vector<std::uint32_t>& shader,
    std::uint32_t local_size,GpuShaderWorkContract contract){
    if(shader.empty()||!local_size||!contract.steps_per_loadout||
       contract.steps_per_loadout>4096||
       (!contract.resident_state_feedback&&
        (contract.steps_per_loadout!=1||contract.target_completion_ns)))return false;
    for(std::size_t device=0;device<library.device_count();++device){
        const auto id="nodus.fabric.dispatch.vulkan."+std::to_string(device);
        ToolRegistry::Entry entry;
        entry.id=id;entry.name="GPU fabric dispatch "+std::to_string(device);
        entry.caps=ToolCaps::None;
        entry.factory=[&library,device,shader,local_size,contract]{
            return std::unique_ptr<ITool,std::function<void(ITool*)>>(
                new GpuFabricDispatchTool(library,device,shader,local_size,contract),
                [](ITool* value){delete value;});
        };
        if(!registry.register_tool(std::move(entry)))return false;
    }
    return true;
}

} // namespace nodus::gpu
