// profile_math_sink: a profiling row that checks the result of a math chain
// The row tool's source, in the module library: the Menu lists it, the canvas compiler inlines it, and it builds as its
// own plugin DLL; the job timing profile (tools/profiling/nodus_job_timing_profile.cpp) includes it.
#include "profile_probe.h"

class MathSink final : public ITool {
public:
    MathSink(Probe& probe, float expected) : probe_(&probe), expected_(expected) {}
    MathSink() = default;   // no probe: a probe is the profiling program's own object, not saved state, so this row refuses
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        if (!probe_) { ctx.succeeded = false; return; }
        const float result = tool_stack_pop(ctx.stack);
        if (std::fabs(result - expected_) > 1e-5f)
            probe_->ordered.store(false, std::memory_order_relaxed);
        probe_->runs.fetch_add(1, std::memory_order_relaxed);
    }
private:
    Probe* probe_ = nullptr;
    float expected_ = 0.0f;
};

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new MathSink(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
