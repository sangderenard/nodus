// Generated from module module_0
#include "tool_api.h"

#include "canvas_abi.h"
#include "tool_events.h"
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

// Tool stack (in order):
//  - MouseListener (attachments=0)

class Tool_tool_module_0 : public ITool {
public:
    std::string id() const override { return "tool_module_0"; }
    std::string name() const override { return "Table"; }
    ToolCaps caps() const override { return static_cast<ToolCaps>(0); }

    void initialize(const ToolInitContext& ctx) override {
        if (ctx.serialized_path && ctx.serialized_path[0] != '\0') {
            FileInputArchive in(ctx.serialized_path);
            if (in.ok()) {
                deserialize(in);
            }
        }
    }

    void shutdown() override {
    }

    void tick(double /*dt*/, HostAPI& /*host*/) override {}

    void render(RenderContext& /*ctx*/) override {}

    void execute_stack(ToolStackContext& ctx) override {
        ToolStackFrame& stack = ctx.stack;
        const ToolInputState* input = ctx.input;
        (void)input;
        // step 0: MouseListener
        {
            float mx = 0.0f;
            float my = 0.0f;
            float down = 0.0f;
            float up = 0.0f;
            if (input) {
                mx = input->mouse_x;
                my = input->mouse_y;
                down = input->mouse_down ? 1.0f : 0.0f;
                up = input->mouse_up ? 1.0f : 0.0f;
            }
            tool_stack_push(stack, up);
            tool_stack_push(stack, down);
            tool_stack_push(stack, my);
            tool_stack_push(stack, mx);
        }
    }

    int32_t port_count() const override {
        return static_cast<int32_t>(kPortCount);
    }

    ToolPortSpec port_spec(int32_t idx) const override {
        if (idx < 0 || idx >= port_count()) return ToolPortSpec{};
        return kPorts[idx];
    }

private:
    static constexpr ToolPortSpec kPorts[] = {
        {ToolPortKind::Internal, 4},
    };
    static constexpr int kPortCount = static_cast<int>(sizeof(kPorts) / sizeof(kPorts[0]));
};

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() {
    return new Tool_tool_module_0();
}
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) {
    delete t;
}

extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() {
    return __FILE__;
}

extern "C" NODUS_PLUGIN_EXPORT int nodus_autobind_mouse_ports() { return 4; }
extern "C" NODUS_PLUGIN_EXPORT int nodus_autobind_keyboard_ports() { return 0; }

#if defined(_WIN32)
extern "C" __declspec(dllexport) int plugin_init(HostAPI* host) {
#else
extern "C" int plugin_init(HostAPI* host) {
#endif
    (void)host;
    return 1;
}

#if defined(_WIN32)
extern "C" __declspec(dllexport) void plugin_shutdown() {
#else
extern "C" void plugin_shutdown() {
#endif
}
