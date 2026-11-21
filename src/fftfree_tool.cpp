// The fftfree row in the tool registry. The row itself is the plugin source
// module_library/source/tools/fftfree.cpp, compiled into this image the way a composed plugin source is:
// NODUS_PLUGIN_COMPOSITE with its exports renamed (as src/tcp_byte_stream_tool.cpp does for its row).
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_fftfree_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_fftfree_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_fftfree_source
#include "../module_library/source/tools/fftfree.cpp"

#include "tool_registry.h"

namespace nodus::gpu {
bool register_fftfree_tool(ToolRegistry& registry,
    TensorStateLibraryTool& library,const std::string& dll){
    if(dll.empty())return false;
    ToolRegistry::Entry item;
    item.id="nodus.fftfree";item.name="fftfree transform";
    item.caps=ToolCaps::None;
    item.source_path=nodus_fftfree_source();
    item.factory=[&library,dll]{
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new FftFreeTool(library,dll),[](ITool* value){nodus_fftfree_destroy(value);});
    };
    return registry.register_tool(std::move(item));
}
} // namespace nodus::gpu
