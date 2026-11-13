#include "nodus_headless_abi.h"
#include "tool_registry.h"
#if defined(_WIN32)
#include <SDL3/SDL.h>
#include <io.h>
#include <fcntl.h>
#include <array>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>

namespace {
constexpr const char* kPort = "audio.pa.pcm";
constexpr const char* kTool = "audio.pa.sdl_output";
constexpr int kFrameBytes = 4, kRate = 48000;
struct Block { uint32_t frames = 0; std::array<int16_t, 512> samples{}; };
struct State {
    NodusHeadlessGraph* graph = nullptr;
    uint32_t device = 0;
    std::atomic<bool> ready{false}, failed{false}, finishing{false}, drained{false};
    std::atomic<uint64_t> frames{0}, refused{0}, empty_events{0};
    std::atomic<int> queued{0}, max_queued{0};
    std::atomic<float> ratio{1};
};
class Output final : public ITool {
public:
    explicit Output(std::shared_ptr<State> state): s_(std::move(state)) {}
    ~Output() override { shutdown(); }
    const char* id_cstr() const noexcept override { return kTool; }
    const char* name_cstr() const noexcept override { return "Owned PCM SDL output"; }
    void initialize(const ToolInitContext&) override {
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
    void execute_stack(ToolStackContext&) override {
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
    void fail() { s_->failed.store(true); std::fprintf(stderr, "pcm_sink_error=%s\n", SDL_GetError()); }
    void nap() { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    std::shared_ptr<State> s_;
    SDL_AudioStream* stream_ = nullptr;
    bool initialized_ = false, playing_ = false, flushed_ = false, empty_ = false, was_empty_ = false;
    double filtered_ = 0, offset_ = 0, drain_seconds_ = 0;
    std::chrono::steady_clock::time_point last_{}, empty_since_{};
};
}

extern "C" int32_t nodus_headless_pcm_playback_stdin(uint32_t device_id) {
    // This process entry owns its complete graph. No shared service canvas.
    auto state = std::make_shared<State>(); state->device = device_id;
    state->graph = nodus_headless_create(); if (!state->graph) return 1;
    ToolRegistry::Entry entry;
    entry.id = kTool; entry.name = "Owned PCM SDL output"; entry.caps = ToolCaps::Table;
    entry.factory = [state] { return ToolRegistry::Factory::result_type(new Output(state), [](ITool* p){delete p;}); };
    int result = 1;
    if (!tool_registry_global().register_tool(std::move(entry))) { nodus_headless_destroy(state->graph); return 1; }
    GP_CanvasModuleDesc desc{}; desc.w=240; desc.h=120; std::strcpy(desc.label,"owned_pcm_output");
    const int module = nodus_headless_add_module(state->graph, &desc);
    GP_CanvasModuleRowDesc row{1,0,1,1,kTool};
    NodusHeadlessFifoConfig fifo{}; fifo.token_bytes=sizeof(Block); fifo.slots=16;
    fifo.top_k=0; fifo.type_id=-1; fifo.layout=2; fifo.dtype=0; fifo.backend_type=GP_MEM_BACKEND_CPU;
    if (module >= 0 && nodus_headless_set_module_rows(state->graph,module,&row,1) &&
        nodus_headless_declare_output_port_ex(state->graph,kPort,module,0,&fifo,1) &&
        nodus_headless_set_module_stack_contract(state->graph,module,2) &&
        nodus_headless_set_module_exec_mode(state->graph,module,3) && nodus_headless_step(state->graph,1./60)) {
        const auto ready_deadline = std::chrono::steady_clock::now()+std::chrono::seconds(3);
        while (!state->ready.load() && !state->failed.load() && std::chrono::steady_clock::now()<ready_deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (state->ready.load()) {
            _setmode(_fileno(stdin), _O_BINARY);
            result = 0;
            while (!state->failed.load()) {
                Block block{};
                const size_t bytes = std::fread(block.samples.data(),1,sizeof(block.samples),stdin);
                if (bytes % kFrameBytes || std::ferror(stdin)) { result=2; break; }
                if (!bytes) break;
                block.frames=static_cast<uint32_t>(bytes/kFrameBytes);
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
                while (!state->failed.load()) {
                    NodusHeadlessPublishResult publish{};
                    if (nodus_headless_publish_output_token(state->graph,kPort,&block,sizeof(block),&publish) && publish.accepted) {
                        if (publish.dropped) { result=3; state->failed.store(true); }
                        break;
                    }
                    state->refused.fetch_add(1);
                    if (std::chrono::steady_clock::now()>deadline) { result=3; state->failed.store(true); break; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            state->finishing.store(true);
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
            while (!state->drained.load() && !state->failed.load() && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (!state->drained.load() || state->failed.load()) result=1;
        }
    }
    nodus_headless_destroy(state->graph);
    tool_registry_global().unregister_tool(kTool);
    std::printf("pcm_sink_done frames=%llu max_queue_frames=%d rate=%.8f backpressure=%llu empty_stream_events=%llu drained=%d result=%d\n",
        static_cast<unsigned long long>(state->frames.load()),state->max_queued.load(),state->ratio.load(),
        static_cast<unsigned long long>(state->refused.load()),static_cast<unsigned long long>(state->empty_events.load()),
        state->drained.load()?1:0,result);
    std::fflush(stdout);
    return result;
}
#else
extern "C" int32_t nodus_headless_pcm_playback_stdin(uint32_t) { return 1; }
#endif
