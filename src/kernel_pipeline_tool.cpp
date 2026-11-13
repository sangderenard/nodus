#include "kernel_pipeline_tool.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

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
    :id_(std::move(id)),components_(std::move(components)){
    tools_.reserve(components_.size());
    for(const auto& component:components_){
        auto tool=registry.create(component.tool_id);
        if(!tool||tool->port_count()!=2||
           tool->port_spec(0).kind!=ToolPortKind::Argument||
           tool->port_spec(1).kind!=ToolPortKind::Return)
            throw std::invalid_argument("kernel pipeline component row unavailable");
        tools_.push_back(std::move(tool));
    }
}

void KernelPipelineTool::initialize(const ToolInitContext& context){
    for(auto& tool:tools_)tool->initialize(context);
}
void KernelPipelineTool::shutdown(){
    for(auto& tool:tools_)tool->shutdown();
}

void KernelPipelineTool::execute_stack(ToolStackContext& context){
    context.succeeded=false;
    if(!context.stack.raw)return;
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

bool register_kernel_pipeline(ToolRegistry& registry,std::string id,
                              std::vector<KernelPipelineComponent> components){
    if(id.empty()||components.empty()||components.size()>kMaximumRows)return false;
    for(const auto& component:components){
        // OpenGL rows require an owning current context and a process-local
        // memory bank; they cannot be called from this pooled pipeline lane.
        if(component.tool_id.empty()||!component.program_id||
           component.api==TensorStateLibraryTool::ExecutionApi::OpenGL||
           !registry.find(component.tool_id))return false;
    }
    ToolRegistry::Entry item;
    item.id=id;item.name="Kernel pipeline";item.caps=ToolCaps::None;
    item.factory=[&registry,id,components]{
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new KernelPipelineTool(id,components,registry),
            [](ITool* value){delete value;});
    };
    return registry.register_tool(std::move(item));
}
} // namespace nodus::gpu
