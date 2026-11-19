// audio.pa.sdl_output: a graph row that plays 16-bit stereo PCM blocks from a headless port through SDL.
// The row tool's source, in the module library: the Menu lists it, the canvas compiler inlines it, it builds as its
// own plugin DLL, and the headless PCM playback process registers it from it (src/headless/pcm_playback.cpp).
// Windows only, as the process is. The row has no saved settings; the owning process's State is handed in at
// construction, and a row made without it refuses every execute_stack.
#if defined(_WIN32)
#include "audio_pa_sdl_output.h"
#include "tool_registry.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <thread>

namespace {
class Output final : public ITool {
public:
    explicit Output(std::shared_ptr<State> state): s_(std::move(state)) {}
    Output() = default;   // no State: the process that owns the graph and the device is not saved state, so this row refuses
    ~Output() override { shutdown(); }
    const char* id_cstr() const noexcept override { return kTool; }
    const char* name_cstr() const noexcept override { return "Owned PCM SDL output"; }
    void initialize(const ToolInitContext&) override {
        if (!s_) return;
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) { fail(); return; }
        initialized_ = true;
        SDL_AudioSpec input{SDL_AUDIO_S16LE, 2, kRate};
        stream_ = SDL_OpenAudioDeviceStream(s_->device ? s_->device : SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK,
                                           &input, nullptr, nullptr);
        if (!stream_) { fail(); return; }
        SDL_AudioSpec actual{}; int frames = 0;
        if (SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(stream_), &actual, &frames) && actual.freq > 0)
            drain_seconds_ = 2. * frames / actual.freq;
        else { fail(); return; }
        last_ = std::chrono::steady_clock::now();
        s_->ready.store(true);
        std::printf("pcm_sink_ready device=%u rate=48000 channels=2\n", s_->device); std::fflush(stdout);
    }
    void shutdown() override {
        if (stream_) { SDL_DestroyAudioStream(stream_); stream_ = nullptr; }
        if (initialized_) { SDL_QuitSubSystem(SDL_INIT_AUDIO); initialized_ = false; }
    }
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& context) override {
        if (!s_) { context.succeeded = false; return; }
        if (!stream_ || s_->failed.load() || s_->drained.load()) { nap(); return; }
        int queued = SDL_GetAudioStreamQueued(stream_);
        if (queued < 0) { fail(); return; }
        // Limit SDL's queue to 2048 input frames plus one bounded block.
        if (queued < 2048 * kFrameBytes) {
            Block block{}; size_t written = 0;
            if (nodus_headless_pull_token(s_->graph, kPort, &block, sizeof(block), &written)) {
                if (written != sizeof(block) || !block.frames || block.frames > 256) { fail(); return; }
                if (!SDL_PutAudioStreamData(stream_, block.samples.data(), block.frames * kFrameBytes)) { fail(); return; }
                s_->frames.fetch_add(block.frames);
                queued = SDL_GetAudioStreamQueued(stream_);
            }
        }
        s_->queued.store(queued / kFrameBytes);
        s_->max_queued.store(std::max(s_->max_queued.load(), queued / kFrameBytes));
        int unread = 0;
        if (!nodus_headless_get_port_unread(s_->graph, kPort, 1, &unread)) { fail(); return; }
        const bool finish = s_->finishing.load() && !unread;
        if (finish && !flushed_) {
            if (!SDL_FlushAudioStream(stream_)) { fail(); return; }
            flushed_ = true;
        }
        if (!playing_ && (queued >= 1024 * kFrameBytes || finish)) {
            if (!SDL_ResumeAudioStreamDevice(stream_)) { fail(); return; }
            playing_ = true;
        }
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now-last_).count(); last_ = now;
        if (playing_ && !finish) {
            // Momentum and slew correct independent hardware-clock drift.
            // SDL performs the actual resampling, never repeated PCM blocks.
            const double error = queued / double(kFrameBytes * 1024) - 1.;
            filtered_ += -std::expm1(-dt/.5) * (error-filtered_);
            const double target = std::clamp(filtered_*.002, -.005, .005);
            offset_ += std::clamp(target-offset_, -.002*dt, .002*dt);
            if (!SDL_SetAudioStreamFrequencyRatio(stream_, static_cast<float>(1.+offset_))) { fail(); return; }
            s_->ratio.store(static_cast<float>(1.+offset_));
            const bool empty_now = queued == 0 && SDL_GetAudioStreamAvailable(stream_) == 0;
            if (empty_now && !was_empty_) s_->empty_events.fetch_add(1);
            was_empty_ = empty_now;
        }
        if (finish && queued == 0 && SDL_GetAudioStreamAvailable(stream_) == 0) {
            if (!empty_) { empty_ = true; empty_since_ = now; }
            // Stream drain alone omits samples already handed to the device.
            if (std::chrono::duration<double>(now-empty_since_).count() >= drain_seconds_) s_->drained.store(true);
        } else empty_ = false;
        nap();
    }
private:
    void fail() { if (!s_) return; s_->failed.store(true); std::fprintf(stderr, "pcm_sink_error=%s\n", SDL_GetError()); }
    void nap() { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    std::shared_ptr<State> s_;   // unset in a row made with no process
    SDL_AudioStream* stream_ = nullptr;
    bool initialized_ = false, playing_ = false, flushed_ = false, empty_ = false, was_empty_ = false;
    double filtered_ = 0, offset_ = 0, drain_seconds_ = 0;
    std::chrono::steady_clock::time_point last_{}, empty_since_{};
};
}

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new Output(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
#endif
