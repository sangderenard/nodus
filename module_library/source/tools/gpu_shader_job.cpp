// gpu_shader_job: a GPU shader job row, one per Vulkan device over the tensor state library (include/gpu_shader_job_tool.h).
// The row tool's source, in the module library: the Menu lists it, the canvas compiler inlines it, it builds as its own
// plugin DLL, and the image registers it from it (src/gpu_shader_job_tool.cpp).
// Its settings are its saved state (serialize/deserialize); a row with no library refuses.
#include "gpu_shader_job_tool.h"
#include "kernel_vulkan_runner.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>
#include <limits>

namespace nodus::gpu {
GpuShaderJobTool::GpuShaderJobTool(TensorStateLibraryTool& library,
    std::size_t device,GpuShaderJobContract contract)
    :library_(&library),device_(device),contract_(std::move(contract)),
     id_(contract_.id+".vulkan."+std::to_string(device)){}
GpuShaderJobTool::GpuShaderJobTool()=default;
void GpuShaderJobTool::configure(std::size_t device,GpuShaderJobContract contract){
    device_=device;contract_=std::move(contract);
    id_=contract_.id+".vulkan."+std::to_string(device_);
}
// Saved state: the row's settings (magic "GSJ1", version 1, device, then the contract: id, program id, entry, SPIR-V words,
// minimum binding bytes, writable flags, binding offsets, reset ranges, local sizes and invocation limits).
namespace {
constexpr std::uint32_t kShaderJobSettingsMagic=0x314a5347,kShaderJobSettingsVersion=1;
}
bool GpuShaderJobTool::serialize(OutputArchive& out) const{
    auto u32=[&](std::uint32_t v){return out.write(&v,sizeof(v));};
    auto u64=[&](std::uint64_t v){return out.write(&v,sizeof(v));};
    auto str=[&](const std::string& v){return u32(static_cast<std::uint32_t>(v.size()))&&(v.empty()||out.write(v.data(),v.size()));};
    auto vec=[&](const auto& v){
        if(!u64(v.size()))return false;
        for(const auto x:v)if(!u64(static_cast<std::uint64_t>(x)))return false;
        return true;
    };
    (void)str;(void)vec;
    if(!u32(kShaderJobSettingsMagic)||!u32(kShaderJobSettingsVersion)||!u64(device_)||
       !str(contract_.id)||!u64(contract_.program_id)||!str(contract_.entry)||
       !vec(contract_.spirv)||!vec(contract_.minimum_binding_bytes)||!vec(contract_.writable)||
       !vec(contract_.binding_offsets)||!u64(contract_.reset_before_dispatch.size()))return false;
    for(const auto& reset:contract_.reset_before_dispatch)
        if(!u64(reset.binding)||!u64(reset.offset_words)||!u64(reset.count)||!u64(reset.value))return false;
    return u64(contract_.local_size_x)&&u64(contract_.maximum_invocations)&&
           u64(contract_.local_size_y)&&u64(contract_.maximum_invocations_y);
}
bool GpuShaderJobTool::deserialize(InputArchive& in){
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
    std::uint64_t device=0,resets=0;
    GpuShaderJobContract read;
    if(!u32(magic)||!u32(version)||magic!=kShaderJobSettingsMagic||version!=kShaderJobSettingsVersion||
       !u64(device)||device>0xFFFF||
       !str(read.id,4096)||!u64(read.program_id)||!str(read.entry,4096)||
       !vec(read.spirv,1u<<22)||!vec(read.minimum_binding_bytes,32)||!vec(read.writable,32)||
       !vec(read.binding_offsets,32)||!u64(resets)||resets>1024)return false;
    read.reset_before_dispatch.resize(static_cast<std::size_t>(resets));
    for(auto& reset:read.reset_before_dispatch){
        std::uint64_t fields[4]={};
        if(!u64(fields[0])||!u64(fields[1])||!u64(fields[2])||!u64(fields[3])||
           fields[0]>0xFFFFFFFFull||fields[1]>0xFFFFFFFFull||fields[2]>0xFFFFFFFFull||fields[3]>0xFFFFFFFFull)return false;
        reset.binding=static_cast<std::uint32_t>(fields[0]);reset.offset_words=static_cast<std::uint32_t>(fields[1]);
        reset.count=static_cast<std::uint32_t>(fields[2]);reset.value=static_cast<std::uint32_t>(fields[3]);
    }
    std::uint64_t sizes[4]={};
    if(!u64(sizes[0])||!u64(sizes[1])||!u64(sizes[2])||!u64(sizes[3]))return false;
    for(const auto size:sizes)if(size>0xFFFFFFFFull)return false;
    read.local_size_x=static_cast<std::uint32_t>(sizes[0]);read.maximum_invocations=static_cast<std::uint32_t>(sizes[1]);
    read.local_size_y=static_cast<std::uint32_t>(sizes[2]);read.maximum_invocations_y=static_cast<std::uint32_t>(sizes[3]);
    configure(static_cast<std::size_t>(device),std::move(read));
    return true;
}
void GpuShaderJobTool::initialize(const ToolInitContext& ctx){
    if(ctx.serialized_path&&ctx.serialized_path[0]!='\0'){
        FileInputArchive in(ctx.serialized_path);
        if(in.ok())deserialize(in);
    }
}

void GpuShaderJobTool::execute_stack(ToolStackContext& context){
    context.succeeded=false;
    if(!context.stack.raw||!library_)return;
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
       device_<library_->device_count()&&count==contract_.writable.size()){
        const auto lease_started=std::chrono::steady_clock::now();
        std::vector<TensorStateLibraryTool::GpuLoan> loans;
        std::vector<kernels::VulkanBackendBinding> bindings;
        loans.reserve(count);bindings.reserve(count);
        bool admitted=true;
        for(std::size_t i=0;i<count;i++){
            const auto address=request[1]+(contract_.binding_offsets.empty()?i:
                                           contract_.binding_offsets[i]);
            auto loan=library_->borrow_gpu(request[0],address,device_,
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
    if(status==0)library_->record_work_completion(
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

} // namespace nodus::gpu

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::gpu::GpuShaderJobTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
