// The tensor state library row in the tool registry. The row itself is the plugin source
// module_library/source/tools/tensor_state_library.cpp, compiled into this image the way a composed plugin source is:
// NODUS_PLUGIN_COMPOSITE with its exports renamed (as src/tcp_byte_stream_tool.cpp does for its row).
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_tensor_state_library_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_tensor_state_library_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_tensor_state_library_source
#include "../module_library/source/tools/tensor_state_library.cpp"

#include "tool_registry.h"

using NodusTensorStateLibraryTool=nodus::gpu::TensorStateLibraryTool;
REGISTER_TOOL(NodusTensorStateLibraryTool,"nodus.tensor.state.library",
              "Tensor state library",ToolCaps::None);
// The entry's source is the row source (the registrar above takes no source path); statics of this file run in order.
static const bool g_tensor_state_library_source =
    tool_registry_global().set_source_path("nodus.tensor.state.library",nodus_tensor_state_library_source());
