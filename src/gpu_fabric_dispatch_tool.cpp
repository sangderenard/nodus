// The GPU fabric dispatch rows in the tool registry. The row itself is the plugin source
// module_library/source/tools/gpu_fabric_dispatch.cpp, compiled into this image the way a composed plugin source is:
// NODUS_PLUGIN_COMPOSITE with its exports renamed (as src/tcp_byte_stream_tool.cpp does for its row).
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_gpu_fabric_dispatch_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_gpu_fabric_dispatch_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_gpu_fabric_dispatch_source
#include "../module_library/source/tools/gpu_fabric_dispatch.cpp"

#include "tool_registry.h"

namespace nodus::gpu {

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
        entry.source_path=nodus_gpu_fabric_dispatch_source();
        entry.factory=[&library,device,shader,local_size,contract]{
            return std::unique_ptr<ITool,std::function<void(ITool*)>>(
                new GpuFabricDispatchTool(library,device,shader,local_size,contract),
                [](ITool* value){nodus_gpu_fabric_dispatch_destroy(value);});
        };
        if(!registry.register_tool(std::move(entry)))return false;
    }
    return true;
}

} // namespace nodus::gpu
