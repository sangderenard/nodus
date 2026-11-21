// gpu_fabric_dispatch: a GPU fabric law dispatch row, one per Vulkan device over the tensor state library (include/gpu_fabric_dispatch_tool.h).
// The row tool's source, in the module library: the Menu lists it, the canvas compiler inlines it, it builds as its own
// plugin DLL, and the image registers it from it (src/gpu_fabric_dispatch_tool.cpp).
// Its settings are its saved state (serialize/deserialize); a row with no library refuses.
#include "gpu_fabric_dispatch_tool.h"
#include "kernel_vulkan_runner.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>

namespace nodus::gpu {

GpuFabricDispatchTool::GpuFabricDispatchTool(
    TensorStateLibraryTool& library,std::size_t device,
    std::vector<std::uint32_t> shader,std::uint32_t local_size,
    GpuShaderWorkContract contract)
    :library_(&library),device_(device),shader_(std::move(shader)),
     local_size_(local_size),contract_(contract),
     id_("nodus.fabric.dispatch.vulkan."+std::to_string(device)){}

GpuFabricDispatchTool::GpuFabricDispatchTool()=default;
void GpuFabricDispatchTool::configure(std::size_t device,
    std::vector<std::uint32_t> shader,std::uint32_t local_size,
    GpuShaderWorkContract contract){
    device_=device;shader_=std::move(shader);local_size_=local_size;contract_=contract;
    id_="nodus.fabric.dispatch.vulkan."+std::to_string(device_);
}
// Saved state: the row's settings (magic "GFD1", version 1, device, SPIR-V words, local size, then the work contract:
// program ID, steps per loadout, resident state feedback, target completion ns).
namespace {
constexpr std::uint32_t kGpuFabricSettingsMagic=0x31444647,kGpuFabricSettingsVersion=1;
}
bool GpuFabricDispatchTool::serialize(OutputArchive& out) const{
    auto u32=[&](std::uint32_t v){return out.write(&v,sizeof(v));};
    auto u64=[&](std::uint64_t v){return out.write(&v,sizeof(v));};
    auto str=[&](const std::string& v){return u32(static_cast<std::uint32_t>(v.size()))&&(v.empty()||out.write(v.data(),v.size()));};
    auto vec=[&](const auto& v){
        if(!u64(v.size()))return false;
        for(const auto x:v)if(!u64(static_cast<std::uint64_t>(x)))return false;
        return true;
    };
    (void)str;(void)vec;
    return u32(kGpuFabricSettingsMagic)&&u32(kGpuFabricSettingsVersion)&&u64(device_)&&vec(shader_)&&
           u64(local_size_)&&u64(contract_.program_id)&&u64(contract_.steps_per_loadout)&&
           u64(contract_.resident_state_feedback?1:0)&&u64(contract_.target_completion_ns);
}
bool GpuFabricDispatchTool::deserialize(InputArchive& in){
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
            if(!u64(raw)||raw>std::numeric_limits<Element>::max())return false;
            x=static_cast<Element>(raw);
        }
        return true;
    };
    (void)str;(void)vec;
    std::uint32_t magic=0,version=0;
    std::uint64_t device=0,local_size=0,steps=0,feedback=0;
    std::vector<std::uint32_t> shader;
    GpuShaderWorkContract contract;
    if(!u32(magic)||!u32(version)||magic!=kGpuFabricSettingsMagic||version!=kGpuFabricSettingsVersion||
       !u64(device)||device>0xFFFF||!vec(shader,1u<<22)||!u64(local_size)||local_size>0xFFFFFFFFull||
       !u64(contract.program_id)||!u64(steps)||steps>4096||!u64(feedback)||feedback>1||
       !u64(contract.target_completion_ns))return false;
    contract.steps_per_loadout=static_cast<std::uint32_t>(steps);
    contract.resident_state_feedback=feedback!=0;
    configure(static_cast<std::size_t>(device),std::move(shader),static_cast<std::uint32_t>(local_size),contract);
    return true;
}
void GpuFabricDispatchTool::initialize(const ToolInitContext& ctx){
    if(ctx.serialized_path&&ctx.serialized_path[0]!='\0'){
        FileInputArchive in(ctx.serialized_path);
        if(in.ok())deserialize(in);
    }
}

void GpuFabricDispatchTool::execute_stack(ToolStackContext& context){
    context.succeeded=false;
    if(!context.stack.raw||!library_)return;
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
       request[1]<=UINT64_MAX-26&&device_<library_->device_count()&&!shader_.empty()&&
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
            auto lease=library_->borrow_gpu(request[0],request[1]+i,device_,
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
        library_->record_completion(device_,lease_ns,elapsed_ns,total_ns);
        library_->record_work_completion(TensorStateLibraryTool::ExecutionApi::Vulkan,
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

} // namespace nodus::gpu

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::gpu::GpuFabricDispatchTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
