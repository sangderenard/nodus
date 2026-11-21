// The frame artifact rows in the image. The file row itself is the plugin source
// module_library/source/tools/frame_artifact_file.cpp, compiled into this image the way a composed plugin source is:
// NODUS_PLUGIN_COMPOSITE with its exports renamed. The packet row's source is compiled once, in
// src/frame_tensor_packet_tool.cpp, and is only named here.
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_frame_artifact_file_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_frame_artifact_file_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_frame_artifact_file_source
#include "../module_library/source/tools/frame_artifact_file.cpp"
#include "frame_tensor_packet_tool.h"

extern "C" const char* nodus_frame_tensor_packet_source();   // src/frame_tensor_packet_tool.cpp

namespace nodus::gpu {
bool register_frame_artifact_rows(ToolRegistry& registry,
    const std::string& digest,const std::filesystem::path& directory,
    std::uint32_t color_encoding,std::uint32_t origin,
    std::uint64_t max_packet_bytes,std::uint64_t max_compressed_bytes) {
    if(digest.size()!=64||color_encoding>2||origin>1||
       !max_packet_bytes||!max_compressed_bytes)return false;
    for(char c:digest)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
    if(!std::filesystem::is_directory(directory))return false;
    const auto resolved=std::filesystem::weakly_canonical(directory);
    ToolRegistry::Entry packet;
    packet.id="nodus.frame.packet."+digest;
    packet.name="Frame tensor packet";
    packet.source_path=nodus_frame_tensor_packet_source();
    packet.factory=[color_encoding,origin,max_packet_bytes] {
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new FrameTensorPacketTool(color_encoding,origin,max_packet_bytes),
            [](ITool* value){delete value;});
    };
    ToolRegistry::Entry file;
    file.id="nodus.frame.file."+digest;
    file.name="Frame artifact file";
    file.source_path=nodus_frame_artifact_file_source();
    file.factory=[resolved,max_compressed_bytes] {
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new FrameArtifactFileTool(resolved,max_compressed_bytes),
            [](ITool* value){nodus_frame_artifact_file_destroy(value);});
    };
    // Both names are reserved before either registration so a partial pair
    // cannot replace an existing row. Registry mutation is host-only.
    if(registry.find(packet.id)||registry.find(file.id))return false;
    if(!registry.register_tool(std::move(packet)))return false;
    if(!registry.register_tool(std::move(file))) {
        registry.unregister_tool("nodus.frame.packet."+digest);
        return false;
    }
    return true;
}
}
