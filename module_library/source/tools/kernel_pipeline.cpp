// kernel_pipeline: a row that runs a string of registered component rows over the tensor state library (include/kernel_pipeline_tool.h).
// The row tool's source, in the module library: the Menu lists it, the canvas compiler inlines it, it builds as its own
// plugin DLL, and the image registers it from it (src/kernel_pipeline_tool.cpp).
// Its settings are its saved state (serialize/deserialize); a row with no registry refuses.
#include "kernel_pipeline_tool.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nodus::gpu {
namespace {
constexpr std::size_t kLegacyRowWords=7,kRowWords=8,kReceiptWords=9,kMaximumRows=64;
using WorkRow=std::array<std::uint64_t,kRowWords>;
using Receipt=std::array<std::uint64_t,kReceiptWords>;

bool execute_component(ITool& tool,const WorkRow& row,Receipt& receipt){
    const std::uint64_t request_shape[1]={row[7]?5ull:4ull};
    const auto request=nodus_tensor_create(NODUS_DTYPE_U64,1,request_shape);
    if(!request)return false;
    const std::array<std::uint64_t,5> work{row[3],row[4],row[5],row[6],row[7]};
    const auto work_bytes=request_shape[0]*sizeof(std::uint64_t);
    if(nodus_tensor_write(request,0,work.data(),work_bytes)!=work_bytes){
        nodus_tensor_destroy(request);return false;
    }
    RawStackFrame frame{};
    if(!raw_stack_init_frame(frame,4096)){
        nodus_tensor_destroy(request);return false;
    }
    bool okay=false;
    if(raw_stack_push_abstract_tensor(frame,{request})){
        ToolStackContext stage{};stage.stack.raw=&frame;
        tool.execute_stack(stage);
        nodus::tensors::AbstractTensorHandle result{};
        NodusTensorDesc desc{};
        if(stage.succeeded&&raw_stack_pop_abstract_tensor(frame,result)){
            if(nodus_tensor_describe(result.id,&desc)==NODUS_OK&&
               desc.dtype==NODUS_DTYPE_U64&&desc.rank==1&&
               desc.shape[0]>=kReceiptWords&&
               nodus_tensor_read(result.id,0,receipt.data(),sizeof(receipt))==
                   sizeof(receipt))okay=receipt[0]==0;
            nodus_tensor_destroy(result.id);
        }
    }
    nodus_tensor_destroy(request);
    raw_stack_free_mask(frame);
    if(frame.backend)gp_mem_backend_release(frame.backend);
    return okay;
}
}

KernelPipelineTool::KernelPipelineTool(std::string id,
    std::vector<KernelPipelineComponent> components,ToolRegistry& registry)
    :id_(std::move(id)),components_(std::move(components)),registry_(&registry){
    build_tools();
}
KernelPipelineTool::KernelPipelineTool()=default;
void KernelPipelineTool::build_tools(){
    tools_.clear();
    if(!registry_)return;
    tools_.reserve(components_.size());
    for(const auto& component:components_){
        auto tool=registry_->create(component.tool_id);
        if(!tool||tool->port_count()!=2||
           tool->port_spec(0).kind!=ToolPortKind::Argument||
           tool->port_spec(1).kind!=ToolPortKind::Return)
            throw std::invalid_argument("kernel pipeline component row unavailable");
        tools_.push_back(std::move(tool));
    }
}

void KernelPipelineTool::configure(std::string id,std::vector<KernelPipelineComponent> components){
    id_=std::move(id);components_=std::move(components);
    build_tools();
}
// Saved state: the row's settings (magic "KPL1", version 1, id, component count, then each component's row id, API and program ID).
namespace {
constexpr std::uint32_t kPipelineSettingsMagic=0x314c504b,kPipelineSettingsVersion=1;
}
bool KernelPipelineTool::serialize(OutputArchive& out) const{
    auto u32=[&](std::uint32_t v){return out.write(&v,sizeof(v));};
    auto u64=[&](std::uint64_t v){return out.write(&v,sizeof(v));};
    auto str=[&](const std::string& v){return u32(static_cast<std::uint32_t>(v.size()))&&(v.empty()||out.write(v.data(),v.size()));};
    auto vec=[&](const auto& v){
        if(!u64(v.size()))return false;
        for(const auto x:v)if(!u64(static_cast<std::uint64_t>(x)))return false;
        return true;
    };
    (void)str;(void)vec;
    if(!u32(kPipelineSettingsMagic)||!u32(kPipelineSettingsVersion)||!str(id_)||!u64(components_.size()))return false;
    for(const auto& component:components_)
        if(!str(component.tool_id)||!u64(static_cast<std::uint64_t>(component.api))||!u64(component.program_id))return false;
    return true;
}
bool KernelPipelineTool::deserialize(InputArchive& in){
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
    std::uint64_t count=0;
    std::string id;
    if(!u32(magic)||!u32(version)||magic!=kPipelineSettingsMagic||version!=kPipelineSettingsVersion||
       !str(id,4096)||!u64(count)||count==0||count>kMaximumRows)return false;
    std::vector<KernelPipelineComponent> components(static_cast<std::size_t>(count));
    for(auto& component:components){
        std::uint64_t api=0;
        if(!str(component.tool_id,4096)||!u64(api)||api>static_cast<std::uint64_t>(TensorStateLibraryTool::ExecutionApi::OpenGL)||
           !u64(component.program_id))return false;
        component.api=static_cast<TensorStateLibraryTool::ExecutionApi>(api);
    }
    try{configure(std::move(id),std::move(components));}
    catch(const std::exception& e){
        std::fprintf(stderr,"[kernel-pipeline] %s\n",e.what());
        tools_.clear();
        return false;
    }
    return true;
}

void KernelPipelineTool::initialize(const ToolInitContext& context){
    if(context.serialized_path&&context.serialized_path[0]!='\0'){
        FileInputArchive in(context.serialized_path);
        if(in.ok())deserialize(in);
    }
    // The component rows are not given this row's saved-state path: it is not theirs.
    ToolInitContext child=context;
    child.serialized_path=nullptr;
    for(auto& tool:tools_)tool->initialize(child);
}
void KernelPipelineTool::shutdown(){
    for(auto& tool:tools_)tool->shutdown();
}

void KernelPipelineTool::execute_stack(ToolStackContext& context){
    context.succeeded=false;
    if(!context.stack.raw||tools_.size()!=components_.size())return;
    auto& stack=*context.stack.raw;
    nodus::tensors::AbstractTensorHandle input{};
    if(!raw_stack_pop_abstract_tensor(stack,input))return;
    NodusTensorDesc desc{};
    if(nodus_tensor_describe(input.id,&desc)!=NODUS_OK||
       desc.dtype!=NODUS_DTYPE_U64||desc.rank!=2||
       (desc.shape[1]!=kLegacyRowWords&&desc.shape[1]!=kRowWords)||
       desc.shape[0]==0||
       desc.shape[0]>kMaximumRows){
        raw_stack_push_abstract_tensor(stack,input);return;
    }
    const auto n=static_cast<std::size_t>(desc.shape[0]);
    const auto stride=static_cast<std::size_t>(desc.shape[1]);
    std::vector<std::uint64_t> flat(n*stride);
    if(nodus_tensor_read(input.id,0,flat.data(),flat.size()*sizeof(std::uint64_t))!=
       flat.size()*sizeof(std::uint64_t)){
        raw_stack_push_abstract_tensor(stack,input);return;
    }
    std::vector<WorkRow> work(n);
    for(std::size_t i=0;i<n;i++)
        std::copy_n(flat.data()+i*stride,stride,work[i].data());
    // Preflight the entire string before allowing the first side effect.
    for(const auto& row:work){
        if(row[0]>=tools_.size()||row[3]==0||row[4]==0||row[5]==0||
           row[1]!=static_cast<std::uint64_t>(components_[row[0]].api)||
           row[2]!=components_[row[0]].program_id){
            raw_stack_push_abstract_tensor(stack,input);return;
        }
    }
    const std::uint64_t result_shape[2]={static_cast<std::uint64_t>(n+1),
                                         kReceiptWords};
    const auto output=nodus_tensor_create(NODUS_DTYPE_U64,2,result_shape);
    if(!output){raw_stack_push_abstract_tensor(stack,input);return;}
    std::vector<Receipt> receipts(n+1);
    receipts[0][0]=0;receipts[0][1]=n;
    receipts[0][4]=std::numeric_limits<std::uint64_t>::max();
    for(std::size_t i=0;i<n;i++){
        receipts[i+1].fill(std::numeric_limits<std::uint64_t>::max());
        receipts[0][2]++;
        if(!execute_component(*tools_[work[i][0]],work[i],receipts[i+1])){
            receipts[0][0]=1;receipts[0][4]=i;
            break; // Never replay a component whose submission may have run.
        }
        receipts[0][3]++;
    }
    if(nodus_tensor_write(output,0,receipts.data(),
                          receipts.size()*sizeof(Receipt))!=
                          receipts.size()*sizeof(Receipt)||
       !raw_stack_push_abstract_tensor(stack,{output})){
        nodus_tensor_destroy(output);
        if(receipts[0][2]==0)raw_stack_push_abstract_tensor(stack,input);
        else std::fprintf(stderr,
            "[kernel-pipeline] receipt publication failed after %llu attempted stages; input consumed, do not replay\n",
            static_cast<unsigned long long>(receipts[0][2]));
        return;
    }
    context.succeeded=true;
}

} // namespace nodus::gpu

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::gpu::KernelPipelineTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
