// radio_fft_bands: the Radio fftfree band projection row (src/headless/radio_rows.cpp registers it).
// The row tool's source, in the module library (case 74fd8157): the Menu lists it, the canvas compiler inlines it, it
// builds as its own plugin DLL, and the image registers it from there. The Radio configuration is not the row's saved
// state: it lives in Shared (radio_shared.h). Shared is process state (the registration captures it in the factory); a row made with no Shared refuses in execute_stack.
#include "radio_shared.h"

#if defined(_WIN32)
namespace nodus::radio {

class RadioFftBands final : public ITool {
public:
    explicit RadioFftBands(std::shared_ptr<Shared> shared) : shared_(std::move(shared)) {}
    RadioFftBands() = default; // no Shared: the row refuses in execute_stack
    const char* id_cstr() const noexcept override { return "radio.fft_bands"; }
    const char* name_cstr() const noexcept override { return "Radio fftfree band projection"; }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        ctx.succeeded = false;
        if (!shared_ || !ctx.stack.raw || !shared_->current_frame || !shared_->fft_states) return;
        nodus::tensors::AbstractTensorHandle token{};
        if (!raw_stack_pop_abstract_tensor(*ctx.stack.raw, token)) return;
        std::array<uint64_t, nodus::gpu::FftFreeTool::kReceiptWords> receipt{};
        const bool valid = nodus_tensor_read(token.id, 0, receipt.data(), sizeof(receipt)) == sizeof(receipt);
        nodus_tensor_destroy(token.id);
        auto& frame = *shared_->current_frame;
        const size_t n = shared_->config.fft_size;
        const size_t bins = n / 2 + 1;
        if (!valid || receipt[0] != 0 || receipt[1] != frame.captured_frames ||
            receipt[2] != 2 || receipt[3] != n || receipt[4] != bins ||
            !frame.sample_rate || frame.sample_rate <= 2 * shared_->config.min_hz) return;
        auto magnitude = shared_->fft_states->borrow_cpu(Shared::fft_card,
                                                         Shared::fft_magnitude, false);
        if (!magnitude || magnitude.bytes() < 2 * bins * sizeof(float)) return;
        const auto* left = static_cast<const float*>(magnitude.data());
        const auto* right = left + bins;
        const float low = std::max(1.0f, shared_->config.min_hz);
        const float high = std::min(std::max(low * 1.001f, shared_->config.max_hz),
                                    0.5f * frame.sample_rate);
        const double log_low = std::log(low), span = std::log(high) - log_low;
        const double scale = 4.0 / (n - 1); // Two over the exact Hann sum.
        for (size_t band = 0; band < shared_->config.bands; ++band) {
            const auto edge = [&](size_t index) {
                const double fraction = double(index) / shared_->config.bands;
                return shared_->config.linear_bands
                    ? low + (high - low) * fraction
                    : std::exp(log_low + span * fraction);
            };
            const double edge0 = edge(band), edge1 = edge(band + 1);
            size_t first = std::max<size_t>(1, static_cast<size_t>(std::ceil(edge0 * n / frame.sample_rate)));
            size_t last = std::min(n / 2, static_cast<size_t>(std::floor(edge1 * n / frame.sample_rate)));
            if (last < first) {
                first = std::max<size_t>(1, static_cast<size_t>(std::llround(
                    (shared_->config.linear_bands ? 0.5 * (edge0 + edge1) : std::sqrt(edge0 * edge1))
                    * n / frame.sample_rate)));
                last = std::min(n / 2, first);
            }
            double power = 0.0;
            for (size_t bin = first; bin <= last; ++bin)
                power += 0.5 * (double(left[bin])*left[bin] + double(right[bin])*right[bin]);
            const double db = 20.0 * std::log10(std::max(1.0e-9, scale * std::sqrt(power)));
            frame.power[band] = static_cast<float>(std::clamp((db + 80.0) / 80.0, 0.0, 1.0));
        }
        frame.band_count = shared_->config.bands;
        magnitude = {};
        void* pointer = &frame;
        const auto type = ValueTypeRegistry::global().builtin(VT_VOID_PTR);
        ctx.succeeded = raw_stack_push_typed(*ctx.stack.raw, &pointer, type);
    }
private:
    std::shared_ptr<Shared> shared_;
};

} // namespace nodus::radio

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::radio::RadioFftBands(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
#endif
