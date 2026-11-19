// The kernel pipeline rows in the tool registry. The row itself is the plugin source
// module_library/source/tools/kernel_pipeline.cpp, compiled into this image the way a composed plugin source is:
// NODUS_PLUGIN_COMPOSITE with its exports renamed (as src/tcp_byte_stream_tool.cpp does for its row).
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_kernel_pipeline_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_kernel_pipeline_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_kernel_pipeline_source
#include "../module_library/source/tools/kernel_pipeline.cpp"

namespace nodus::gpu {

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
    item.source_path=nodus_kernel_pipeline_source();
    item.factory=[&registry,id,components]{
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new KernelPipelineTool(id,components,registry),
            [](ITool* value){nodus_kernel_pipeline_destroy(value);});
    };
    return registry.register_tool(std::move(item));
}
} // namespace nodus::gpu
