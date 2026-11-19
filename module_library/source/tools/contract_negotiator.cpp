// contract_negotiator: the market and worker rows of a native contract pair (include/contract_negotiator_tool.h).
// The row tool's source, in the module library (case 74fd8157): the Menu lists it, the canvas compiler inlines it, it
// builds as its own plugin DLL. The class is defined in its header (the image's registration and the header-only tests
// use it there); the exports below make a row with no negotiator, which refuses in execute_stack until a registration
// binds one. Its saved state is which row it is (serialize/deserialize in the header).
#include "contract_negotiator_tool.h"

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::contracts::NegotiatorTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
