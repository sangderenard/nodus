// profile_row: a profiling row that counts its runs, checks its inputs arrive in order, and spins for a set time
// The row tool's source, in the module library: the Menu lists it, the canvas compiler inlines it, and it builds as its
// own plugin DLL; the job timing profile (tools/profiling/nodus_job_timing_profile.cpp) includes it.
#include "profile_probe.h"

class Row final : public ITool {
public:
    Row(Probe& probe, int work_us, int input_count, bool produce)
        : probe_(&probe), work_us_(work_us), input_count_(input_count), produce_(produce) {}
    Row() = default;   // no probe: a probe is the profiling program's own object, not saved state, so this row refuses
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        if (!probe_) { ctx.succeeded = false; return; }
        const uint64_t ordinal = probe_->runs.fetch_add(1, std::memory_order_relaxed) + 1;
        for (int i = 0; i < input_count_; ++i) {
            if (tool_stack_pop(ctx.stack) != static_cast<float>(ordinal))
                probe_->ordered.store(false, std::memory_order_relaxed);
        }
        if (produce_) tool_stack_push(ctx.stack, static_cast<float>(ordinal));
        if (work_us_ > 0) {
            const auto until = Clock::now() + std::chrono::microseconds(work_us_);
            while (Clock::now() < until) std::atomic_signal_fence(std::memory_order_seq_cst);
        }
    }
private:
    Probe* probe_ = nullptr;
    int work_us_ = 0;
    int input_count_ = 0;
    bool produce_ = false;
};

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new Row(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
