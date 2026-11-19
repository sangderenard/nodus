// profile_math_source: a profiling row that pushes the operands of a math chain
// The row tool's source, in the module library: the Menu lists it, the canvas compiler inlines it, and it builds as its
// own plugin DLL; the job timing profile (tools/profiling/nodus_job_timing_profile.cpp) includes it.
#include "profile_probe.h"

class MathSource final : public ITool {
public:
    explicit MathSource(bool chain) : chain_(chain) {}
    MathSource() = default;   // no settings: a plain two-value source
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        if (chain_) {
            for (int i = 0; i < 65; ++i) tool_stack_push(ctx.stack, 1.0f);
        } else {
            tool_stack_push(ctx.stack, 1.25f);
            tool_stack_push(ctx.stack, 2.5f);
        }
    }
private:
    bool chain_ = false;
};

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new MathSource(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
