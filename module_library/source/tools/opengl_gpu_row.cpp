// opengl_gpu_row: the native OpenGL GPU host's control/telemetry row (include/opengl_gpu_row_tool.h).
// The row tool's source, in the module library (case 74fd8157): the Menu lists it, the canvas compiler inlines it, it
// builds as its own plugin DLL. The class is defined in its header (the image's registration, the graph binding and
// the header-only tests use it there). The exports below make a row with no host, which refuses in execute_stack
// until a registration binds one; it has no settings, so it has no saved state.
#include "opengl_gpu_row_tool.h"

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::gpu::OpenGLGpuTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
