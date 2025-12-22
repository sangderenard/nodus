// Generated from module module_0
#include "tool_api.h"

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>

class Tool_tool_module_0 : public ITool {
public:
    std::string id() const override { return "tool_module_0"; }
    std::string name() const override { return "tool_module_0"; }
    ToolCaps caps() const override { return static_cast<ToolCaps>(0); }

    void initialize(const ToolInitContext& ctx) override {
        if (ctx.serialized_path && ctx.serialized_path[0] != '\0') {
            FileInputArchive in(ctx.serialized_path);
            if (in.ok()) {
                deserialize(in);
            }
        }
    }

    void shutdown() override {}

    void tick(double /*dt*/, HostAPI& /*host*/) override {}

    void render(RenderContext& /*ctx*/) override {}

    void execute_stack(ToolStackContext& ctx) override {
        ToolStackFrame& stack = ctx.stack;
        const ToolInputState* input = ctx.input;
        (void)input;
    }

    int32_t port_count() const override {
        return static_cast<int32_t>(kPortCount);
    }

    ToolPortSpec port_spec(int32_t idx) const override {
        if (idx < 0 || idx >= port_count()) return ToolPortSpec{};
        return kPorts[idx];
    }

private:
    static constexpr const ToolPortSpec* kPorts = nullptr;
    static constexpr int kPortCount = 0;
};

#if defined(_WIN32)
extern "C" __declspec(dllexport) ITool* create_tool() {
#else
extern "C" ITool* create_tool() {
#endif
    return new Tool_tool_module_0();
}
#if defined(_WIN32)
extern "C" __declspec(dllexport) void destroy_tool(ITool* t) {
#else
extern "C" void destroy_tool(ITool* t) {
#endif
    delete t;
}

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
