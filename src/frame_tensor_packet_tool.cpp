// The frame tensor packet row in the image. The row itself is the plugin source
// module_library/source/tools/frame_tensor_packet.cpp, compiled into this image the way a composed plugin source is:
// NODUS_PLUGIN_COMPOSITE with its exports renamed (as src/tools/exec_row_registered.cpp does for the exec row).
// The row is registered by register_frame_artifact_rows (src/frame_artifact_file_tool.cpp).
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_frame_tensor_packet_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_frame_tensor_packet_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_frame_tensor_packet_source
#include "../module_library/source/tools/frame_tensor_packet.cpp"
