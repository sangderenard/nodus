// The CPU fabric dispatch row in the tool registry. The row itself is the plugin source
// module_library/source/tools/cpu_fabric_dispatch.cpp, compiled into this image the way a composed plugin source is:
// NODUS_PLUGIN_COMPOSITE with its exports renamed (as src/tcp_byte_stream_tool.cpp does for its row).
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_cpu_fabric_dispatch_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_cpu_fabric_dispatch_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_cpu_fabric_dispatch_source
#include "../module_library/source/tools/cpu_fabric_dispatch.cpp"

namespace nodus::gpu {
bool register_cpu_fabric_dispatcher(ToolRegistry& registry,
    TensorStateLibraryTool& library,const std::string& dll,
    const std::string& entry,std::uint64_t batch,
    std::uint32_t steps_per_loadout,std::uint64_t program_id){
    if(dll.empty()||entry.empty()||!batch||!steps_per_loadout||
       steps_per_loadout>4096||!program_id)return false;
    ToolRegistry::Entry item;
    item.id="nodus.fabric.dispatch.cpu";item.name="CPU fabric dispatch";
    item.caps=ToolCaps::None;
    item.source_path=nodus_cpu_fabric_dispatch_source();
    item.factory=[&library,dll,entry,batch,steps_per_loadout,program_id]{
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new CpuFabricDispatchTool(library,dll,entry,batch,
                steps_per_loadout,program_id),
            [](ITool* value){nodus_cpu_fabric_dispatch_destroy(value);});
    };
    return registry.register_tool(std::move(item));
}
} // namespace nodus::gpu
