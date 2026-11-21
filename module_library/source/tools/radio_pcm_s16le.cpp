// radio_pcm_s16le: the Radio PCM s16le row (src/headless/radio_rows.cpp registers it).
// The row tool's source, in the module library (case 74fd8157): the Menu lists it, the canvas compiler inlines it, it
// builds as its own plugin DLL, and the image registers it from there. The Radio configuration is not the row's saved
// state: it lives in Shared (radio_shared.h). It has no settings and no shared state.
#include "radio_shared.h"

#if defined(_WIN32)
namespace nodus::radio {

// Stack bridge: the RadioFrame pointer stays on top for the spectrum exit;
// an s16le byte tensor sits beneath it for a later declared exec row.
class RadioPcmS16le final : public ITool {
public:
    ~RadioPcmS16le() noexcept override { shutdown(); }
    const char* id_cstr() const noexcept override { return "radio.pcm_s16le"; }
    const char* name_cstr() const noexcept override { return "Radio PCM s16le"; }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override { if (owned_) { nodus_tensor_destroy(owned_); owned_ = 0; } }
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        if (!ctx.stack.raw) { ctx.succeeded = false; return; }
        shutdown(); // The previous tick's exec row has already copied its bytes.
        void* value = nullptr;
        const auto pointer_type = ValueTypeRegistry::global().builtin(VT_VOID_PTR);
        if (!raw_stack_pop_typed(*ctx.stack.raw, &value, pointer_type)) { ctx.succeeded = false; return; }
        const auto* frame = static_cast<const RadioFrame*>(value);
        if (!frame || frame->magic != kFrameMagic || frame->pcm.frames > 256 ||
            (frame->pcm.frames && frame->pcm.sample_rate != 48000)) {
            raw_stack_push_typed(*ctx.stack.raw, &value, pointer_type);
            ctx.succeeded = false; return;
        }
        if (frame->pcm.frames && !frame->terminal) {
            const uint64_t shape[1] = {static_cast<uint64_t>(frame->pcm.frames) * 4};
            owned_ = nodus_tensor_create(NODUS_DTYPE_U8, 1, shape);
            if (!owned_) { raw_stack_push_typed(*ctx.stack.raw, &value, pointer_type); ctx.succeeded = false; return; }
            std::array<uint8_t, 1024> pcm{};
            for (size_t i = 0; i < frame->pcm.frames * 2; ++i) {
                const float sample = std::clamp(frame->pcm.stereo[i], -1.0f, 1.0f);
                const uint16_t word = static_cast<uint16_t>(static_cast<int16_t>(std::lround(sample * 32767.0f)));
                pcm[i * 2] = static_cast<uint8_t>(word);
                pcm[i * 2 + 1] = static_cast<uint8_t>(word >> 8);
            }
            if (nodus_tensor_write(owned_, 0, pcm.data(), shape[0]) != static_cast<int64_t>(shape[0]) ||
                !raw_stack_push_abstract_tensor(*ctx.stack.raw, {owned_})) {
                shutdown(); raw_stack_push_typed(*ctx.stack.raw, &value, pointer_type);
                ctx.succeeded = false; return;
            }
        }
        if (!raw_stack_push_typed(*ctx.stack.raw, &value, pointer_type)) ctx.succeeded = false;
    }
private:
    uint64_t owned_ = 0;
};

} // namespace nodus::radio

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::radio::RadioPcmS16le(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
#endif
