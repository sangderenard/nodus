// The KPath UTF16 stack to toolpath row in the tool registry. The row itself is the plugin source
// module_library/source/tools/kpath_utf16_stack_to_toolpath.cpp, compiled into this image the way a composed plugin
// source is: NODUS_PLUGIN_COMPOSITE with its exports renamed (as src/tools/exec_row_registered.cpp does for the exec row).
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_kpath_utf16_stack_to_toolpath_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_kpath_utf16_stack_to_toolpath_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_kpath_utf16_stack_to_toolpath_source
#include "../../module_library/source/tools/kpath_utf16_stack_to_toolpath.cpp"

REGISTER_TOOL(KpathUtf16StackToToolpathTool,
              "kpath_utf16_stack_to_toolpath",
              "KPath UTF16 Stack To Toolpath",
              ToolCaps::None);

// The entry's source is the row source (the registrar above takes no source path); statics of this file run in order.
static const bool g_kpath_utf16_stack_to_toolpath_source =
    tool_registry_global().set_source_path("kpath_utf16_stack_to_toolpath",nodus_kpath_utf16_stack_to_toolpath_source());
