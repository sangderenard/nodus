// The T4 arena state row in the tool registry. The row itself is the plugin source
// module_library/source/tools/wave_t4_state.cpp, compiled into this image the way a composed plugin source is:
// NODUS_PLUGIN_COMPOSITE with its exports renamed (as src/tcp_byte_stream_tool.cpp does for its row).
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_wave_t4_state_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_wave_t4_state_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_wave_t4_state_source
#include "../module_library/source/tools/wave_t4_state.cpp"

#include "tool_registry.h"

namespace nodus::gpu {
bool register_wave_t4_state_tool(ToolRegistry& registry,
    TensorStateLibraryTool& library,const std::string& dll){
    if(dll.empty())return false;
    ToolRegistry::Entry item;
    item.id="nodus.wave.t4.state";item.name="T4 arena state";
    item.caps=ToolCaps::None;
    item.source_path=nodus_wave_t4_state_source();
    item.factory=[&library,dll]{
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new WaveT4StateTool(library,dll),[](ITool* value){nodus_wave_t4_state_destroy(value);});
    };
    return registry.register_tool(std::move(item));
}
} // namespace nodus::gpu
