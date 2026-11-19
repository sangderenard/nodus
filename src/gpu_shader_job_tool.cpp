// The GPU shader job rows in the tool registry. The row itself is the plugin source
// module_library/source/tools/gpu_shader_job.cpp, compiled into this image the way a composed plugin source is:
// NODUS_PLUGIN_COMPOSITE with its exports renamed (as src/tcp_byte_stream_tool.cpp does for its row).
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_gpu_shader_job_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_gpu_shader_job_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_gpu_shader_job_source
#include "../module_library/source/tools/gpu_shader_job.cpp"

#include "tool_registry.h"

namespace nodus::gpu {

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
        item.source_path=nodus_gpu_shader_job_source();
        item.factory=[&library,device,contract]{
            return std::unique_ptr<ITool,std::function<void(ITool*)>>(
                new GpuShaderJobTool(library,device,contract),
                [](ITool* value){nodus_gpu_shader_job_destroy(value);});
        };
        if(!registry.register_tool(std::move(item)))return false;
    }
    return true;
}
} // namespace nodus::gpu
