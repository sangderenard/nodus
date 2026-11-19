// The TCP byte stream row in the tool registry. The row itself is the plugin source
// module_library/source/tools/tcp_byte_stream.cpp, compiled into this image the way a composed plugin source is:
// NODUS_PLUGIN_COMPOSITE with its exports renamed (as src/tools/exec_row_registered.cpp does for the exec row).
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_tcp_byte_stream_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_tcp_byte_stream_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_tcp_byte_stream_source
#include "../module_library/source/tools/tcp_byte_stream.cpp"

namespace nodus::net {

bool register_tcp_byte_stream(ToolRegistry& registry,TcpByteStreamConfig config){
    if(config.id.empty()||config.maximum_frame_bytes==0||
       config.maximum_frame_bytes>64*1024*1024||
       config.pending_frames==0||config.pending_frames>1024||
       config.broadcast_readers>64||
       (config.broadcast_readers&&config.pending_frames!=1))return false;
    ToolRegistry::Entry entry;
    entry.id=config.id;entry.name=config.broadcast_readers?"TCP broadcast exit":"TCP byte stream";
    entry.caps=ToolCaps::None;
    entry.source_path=nodus_tcp_byte_stream_source();
    entry.factory=[config]{
        auto* tool=new TcpByteStreamTool();
        tool->configure(config);
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(tool,[](ITool* made){nodus_tcp_byte_stream_destroy(made);});
    };
    return registry.register_tool(std::move(entry));
}
} // namespace nodus::net
