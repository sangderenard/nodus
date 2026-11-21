// opengl_native_control: the native development entry's control row (src/opengl_link_once.cpp, its own executable).
// The row tool's source, in the module library (case 74fd8157): the Menu lists it, the canvas compiler inlines it, it
// builds as its own plugin DLL, and the executable compiles it in. The row runs one function the executable gives it
// (a closure over the executable's own locals); that closure is process state, never saved state, and a row made with
// no function (the no-argument constructor) refuses in execute_stack. It has no settings, so it has no saved state.
#include "tool_api.h"

#include <functional>
#include <memory>
#include <utility>

namespace nodus::gpu {

class NativeControlTool final:public ITool {
public:
    explicit NativeControlTool(std::function<void(ToolStackContext&)> fn):fn_(std::move(fn)){}
    NativeControlTool()=default;
    void initialize(const ToolInitContext&)override{}void shutdown()override{}
    void tick(double,HostAPI&)override{}void render(RenderContext&)override{}
    void execute_stack(ToolStackContext& c)override{
        if(!fn_){c.succeeded=false;return;}
        fn_(c);
    }
private:std::function<void(ToolStackContext&)> fn_;
};

} // namespace nodus::gpu

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::gpu::NativeControlTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
