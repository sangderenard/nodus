#include "nodus_headless_abi.h"
#include "tool_api.h"
#include "tool_registry.h"
#include "fftfree_tool.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"

#if defined(_WIN32)
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <SDL3/SDL.h>

#include <array>
#include <algorithm>
#include <filesystem>
#include <climits>
#include <cstdio>
#include <random>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr size_t kMaxBands = 128;
constexpr uint32_t kPacketVersion = 1;
constexpr uint32_t kFrameMagic = 0x4441524eu; // "NRAD" bytes in little-endian order

struct Config {
    std::wstring scope_dll;
    std::wstring fftfree_dll;
    std::string host;
    uint16_t port = 0;
    uint32_t fft_size = 0;
    uint32_t overlap = 0;
    uint32_t bands = 0;
    float min_hz = 0;
    float max_hz = 0;
    bool linear_bands = false;
    std::wstring wav;
    bool loop = false;
    uint16_t pcm_port = 0;
};

struct RadioFrame;
struct Shared {
    Config config;
    std::unique_ptr<nodus::gpu::TensorStateLibraryTool> fft_states;
    RadioFrame* current_frame = nullptr;
    static constexpr uint64_t fft_card = 0x524144494fULL;
    static constexpr uint64_t fft_input = 0x100;
    static constexpr uint64_t fft_real = 0x101;
    static constexpr uint64_t fft_imag = 0x102;
    static constexpr uint64_t fft_magnitude = 0x103;
    std::atomic<bool> held{false};
    NodusHeadlessGraph* pcm_graph = nullptr;
    std::atomic<uint64_t> pcm_dropped{0};
    std::atomic<uint64_t> pcm_emitted{0};
    std::atomic<uint64_t> captured{0};
    std::atomic<uint64_t> capture_overruns{0};
    std::atomic<uint64_t> emitted{0};
    std::atomic<uint64_t> dropped{0};
    std::atomic<uint32_t> sample_rate{0};
    std::atomic<int32_t> running{0};
    ~Shared() { if (fft_states) fft_states->shutdown(); }
};

std::mutex config_mutex;
std::shared_ptr<Shared> configured;

struct PcmBlock {
    uint64_t first_frame = 0;
    uint32_t sample_rate = 48000;
    uint32_t frames = 0;
    std::array<float, 512> stereo{};
};
static_assert(sizeof(PcmBlock) == 2064);
constexpr const char* kPcmPort = "radio.pcm";

struct RadioFrame {
    uint32_t magic = kFrameMagic;
    uint32_t band_count = 0;
    uint32_t sample_rate = 0;
    uint64_t captured_frames = 0;
    float rms = 0;
    std::array<float, kMaxBands> power{};
    PcmBlock pcm{}; // Same captured samples, made available to a stack conversion row.
    bool terminal = false; // Internal row control, never part of NRAD.
};

using scope_create_capture_fn = void* (__cdecl *)();
using scope_start_fn = int (__cdecl *)(void*);
using scope_read_fn = size_t (__cdecl *)(void*, uint64_t*, float*, size_t, uint32_t*, uint64_t*);
using scope_running_fn = int (__cdecl *)(void*);
using scope_frame_count_fn = uint64_t (__cdecl *)(void*);
using scope_feed_fn = int (__cdecl *)(void*, const float*, size_t, uint32_t);
using scope_free_fn = void (__cdecl *)(void*);

bool radio_state_request(nodus::gpu::TensorStateLibraryTool& library,
                         const std::array<uint64_t, 12>& words, uint64_t payload) {
    const uint64_t shape[1] = {words.size()};
    const auto request = nodus_tensor_create(NODUS_DTYPE_U64, 1, shape);
    if (!request) return false;
    bool ok = nodus_tensor_write(request, 0, words.data(), sizeof(words)) == sizeof(words);
    RawStackFrame stack{};
    if (ok) ok = raw_stack_init_frame(stack, 4096);
    if (ok) ok = raw_stack_push_abstract_tensor(stack, {request}) &&
                 raw_stack_push_abstract_tensor(stack, {payload});
    ToolStackContext context{};
    context.stack.raw = &stack;
    if (ok) library.execute_stack(context);
    nodus::tensors::AbstractTensorHandle reply{};
    if (ok) ok = context.succeeded && raw_stack_pop_abstract_tensor(stack, reply);
    uint64_t status = UINT64_MAX;
    if (ok) ok = nodus_tensor_read(reply.id, 0, &status, sizeof(status)) == sizeof(status) && status == 0;
    if (reply.id) nodus_tensor_destroy(reply.id);
    if (stack.backend) { raw_stack_free_mask(stack); gp_mem_backend_release(stack.backend); }
    nodus_tensor_destroy(request);
    return ok;
}

bool radio_register_fft_states(Shared& shared) {
    auto& library = *shared.fft_states;
    const uint64_t dummy_shape[1] = {1};
    const auto dummy = nodus_tensor_create(NODUS_DTYPE_U64, 1, dummy_shape);
    if (!dummy) return false;
    std::array<uint64_t, 12> request{};
    request[0] = 1; request[1] = Shared::fft_card; request[2] = 1;
    request[4] = 16ull << 20;
    bool ok = radio_state_request(library, request, dummy);
    nodus_tensor_destroy(dummy);
    const uint64_t n = shared.config.fft_size;
    const uint64_t bins = n / 2 + 1;
    for (uint64_t address = Shared::fft_input; ok && address <= Shared::fft_magnitude; ++address) {
        const uint64_t state_shape[1] = {address == Shared::fft_input ? 2 * n : 2 * bins};
        const auto state = nodus_tensor_create(NODUS_DTYPE_F32, 1, state_shape);
        if (!state) return false;
        request = {};
        request[0] = 2; request[1] = Shared::fft_card; request[3] = address;
        ok = radio_state_request(library, request, state);
        nodus_tensor_destroy(state);
    }
    return ok;
}

template<size_t N>
uint64_t radio_words_tensor(const std::array<uint64_t, N>& words) {
    const uint64_t shape[1] = {N};
    const auto tensor = nodus_tensor_create(NODUS_DTYPE_U64, 1, shape);
    if (!tensor) return 0;
    if (nodus_tensor_write(tensor, 0, words.data(), sizeof(words)) == sizeof(words)) return tensor;
    nodus_tensor_destroy(tensor);
    return 0;
}

class RadioSource final : public ITool {
public:
    explicit RadioSource(std::shared_ptr<Shared> shared) : shared_(std::move(shared)) {}
    ~RadioSource() noexcept override { shutdown(); }
    const char* id_cstr() const noexcept override { return shared_->config.wav.empty() ? "radio.system_output" : "radio.file_output"; }
    const char* name_cstr() const noexcept override { return "Radio PCM source"; }
    void initialize(const ToolInitContext&) override {
        module_ = LoadLibraryW(shared_->config.scope_dll.c_str());
        if (!module_) return;
        create_capture_ = reinterpret_cast<scope_create_capture_fn>(GetProcAddress(module_, "openrgb_audio_scope_create_capture"));
        start_ = reinterpret_cast<scope_start_fn>(GetProcAddress(module_, "openrgb_audio_scope_start"));
        read_ = reinterpret_cast<scope_read_fn>(GetProcAddress(module_, "openrgb_audio_scope_read_stereo"));
        running_ = reinterpret_cast<scope_running_fn>(GetProcAddress(module_, "openrgb_audio_scope_running"));
        frame_count_ = reinterpret_cast<scope_frame_count_fn>(GetProcAddress(module_, "openrgb_audio_scope_frame_count"));
        free_ = reinterpret_cast<scope_free_fn>(GetProcAddress(module_, "openrgb_audio_scope_free"));
        if (!create_capture_ || !start_ || !read_ || !running_ || !frame_count_ || !free_) return;
        const Config& c = shared_->config;
        scope_ = create_capture_();
        if (!scope_) return;
        if (c.wav.empty()) {
            if (!start_(scope_)) { free_(scope_); scope_ = nullptr; }
        } else {
            feed_ = reinterpret_cast<scope_feed_fn>(GetProcAddress(module_, "openrgb_audio_scope_feed_stereo"));
            if (!feed_ || !SDL_InitSubSystem(SDL_INIT_AUDIO)) return;
            audio_ready_ = true;
            std::error_code size_error;
            const auto bytes = std::filesystem::file_size(c.wav, size_error);
            if (size_error || bytes > 64u * 1024u * 1024u) return;
            SDL_AudioSpec spec{};
            Uint8* original = nullptr;
            Uint32 length = 0;
            const auto utf8 = std::filesystem::path(c.wav).u8string();
            if (!SDL_LoadWAV(reinterpret_cast<const char*>(utf8.c_str()), &spec, &original, &length)) return;
            const int source_frame_bytes = SDL_AUDIO_FRAMESIZE(spec);
            if (source_frame_bytes <= 0 || spec.freq <= 0 ||
                (static_cast<uint64_t>(length) / source_frame_bytes) * 8 * (c.pcm_port ? 48000 : spec.freq) / spec.freq > 64u * 1024u * 1024u) {
                SDL_free(original); return;
            }
            SDL_AudioSpec target{SDL_AUDIO_F32, 2, c.pcm_port ? 48000 : spec.freq};
            Uint8* converted = nullptr;
            int converted_length = 0;
            const bool ok = length <= INT_MAX && SDL_ConvertAudioSamples(&spec, original, static_cast<int>(length),
                &target, &converted, &converted_length);
            SDL_free(original);
            if (!ok) return;
            if (converted_length <= 0 || converted_length > 64 * 1024 * 1024 || converted_length % (2 * sizeof(float))) { SDL_free(converted); return; }
            samples_.resize(static_cast<size_t>(converted_length) / sizeof(float));
            std::memcpy(samples_.data(), converted, converted_length);
            SDL_free(converted);
            for (float value : samples_) if (!std::isfinite(value)) { samples_.clear(); return; }
            rate_ = static_cast<uint32_t>(target.freq);
            stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &target, nullptr, nullptr);
            if (!stream_ || !SDL_ResumeAudioStreamDevice(stream_)) return;
            shared_->running.store(1);
        }
        std::array<uint64_t, nodus::gpu::FftFreeTool::kConfigWords> config{};
        const uint64_t n = c.fft_size;
        config[0] = 1; config[1] = n; config[2] = 1; config[3] = 2;
        config[5] = 1; config[6] = 2; config[16] = 2;
        config[17] = n; config[18] = n; config[19] = 1;
        config[20] = 1; config[23] = 1;
        // fftfree's optional vectored crash hook is process-global and cannot
        // outlive a dynamically loaded row DLL in this long-running host.
        config[28] = 1; // silent crash reports: documented handler opt-out
        std::array<uint64_t, nodus::gpu::FftFreeTool::kWorkWords> work{};
        work[0] = 1; work[1] = Shared::fft_card; work[2] = Shared::fft_input;
        work[4] = Shared::fft_real; work[5] = Shared::fft_imag;
        work[6] = Shared::fft_magnitude; work[7] = 2 * n; work[8] = 2;
        left_.assign(n, 0.0f); right_.assign(n, 0.0f);
        config_token_ = radio_words_tensor(config);
        work_token_ = radio_words_tensor(work);
        if (config_token_ && work_token_) shared_->current_frame = &frame_;
    }
    void shutdown() override {
        if (shared_->current_frame == &frame_) shared_->current_frame = nullptr;
        if (config_token_) { nodus_tensor_destroy(config_token_); config_token_ = 0; }
        if (work_token_) { nodus_tensor_destroy(work_token_); work_token_ = 0; }
        if (stream_) { SDL_DestroyAudioStream(stream_); stream_ = nullptr; }
        if (audio_ready_) { SDL_QuitSubSystem(SDL_INIT_AUDIO); audio_ready_ = false; }
        if (scope_ && free_) { free_(scope_); scope_ = nullptr; }
        if (module_) { FreeLibrary(module_); module_ = nullptr; }
        shared_->running.store(0, std::memory_order_release);
    }
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        if (shared_->held.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ctx.succeeded = false; return;
        }
        if (!scope_ || !config_token_ || !work_token_ || !ctx.stack.raw) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            ctx.succeeded = false;
            return;
        }
        frame_.pcm.frames = 0;
        if (!shared_->config.wav.empty()) {
            if (!stream_ || !feed_) { shared_->running.store(0); std::this_thread::sleep_for(std::chrono::milliseconds(2)); ctx.succeeded = false; return; }
            const size_t hop = shared_->config.fft_size / shared_->config.overlap;
            const int queued = SDL_GetAudioStreamQueued(stream_);
            if (queued < 0) { shared_->running.store(0); ctx.succeeded = false; return; }
            if (queued >= static_cast<int>(shared_->config.fft_size * 2 * 2 * sizeof(float))) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1)); ctx.succeeded = false; return;
            }
            if (position_ == samples_.size()) {
                if (shared_->config.loop) position_ = 0;
                else {
                    // A zero-frame terminal token shares the canonical PCM FIFO:
                    // its first_frame is the exclusive end of the finite source.
                    if (!end_published_ && shared_->config.pcm_port && shared_->pcm_graph) {
                        PcmBlock end{}; end.first_frame = frame_count_(scope_);
                        NodusHeadlessPublishResult result{};
                        end_published_ = nodus_headless_publish_output_token(shared_->pcm_graph,
                            kPcmPort, &end, sizeof(end), &result) && result.accepted;
                        if (result.dropped) shared_->pcm_dropped.fetch_add(result.dropped);
                    }
                    shared_->running.store(queued > 0);
                    ctx.succeeded = false;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1)); return;
                }
            }
            const size_t count = std::min({hop, size_t{256}, (samples_.size() - position_) / 2});
            if (!SDL_PutAudioStreamData(stream_, samples_.data() + position_, static_cast<int>(count * 2 * sizeof(float))) ||
                !feed_(scope_, samples_.data() + position_, count, rate_)) {
                shared_->running.store(0); ctx.succeeded = false; return;
            }
            position_ += count * 2;
        }
        const uint64_t captured = frame_count_(scope_);
        shared_->running.store(running_(scope_), std::memory_order_release);
        std::array<float, 512> pcm{};
        uint32_t sample_rate = 0;
        uint64_t lost = 0;
        const uint64_t first_frame = read_cursor_;
        const size_t count = read_(scope_, &read_cursor_, pcm.data(), 256,
                                   &sample_rate, &lost);
        if (lost) {
            shared_->capture_overruns.fetch_add(lost, std::memory_order_relaxed);
            captured_since_fft_ = 0;
            std::fill(left_.begin(), left_.end(), 0.0f);
            std::fill(right_.begin(), right_.end(), 0.0f);
        }
        if (!count || !sample_rate) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ctx.succeeded = false; return;
        }
        if (frame_.sample_rate && frame_.sample_rate != sample_rate) {
            captured_since_fft_ = 0;
            std::fill(left_.begin(), left_.end(), 0.0f);
            std::fill(right_.begin(), right_.end(), 0.0f);
        }
        frame_.pcm = {};
        frame_.pcm.first_frame = first_frame + lost;
        frame_.pcm.sample_rate = sample_rate;
        frame_.pcm.frames = static_cast<uint32_t>(count);
        std::copy_n(pcm.data(), 2 * count, frame_.pcm.stereo.data());
        if (shared_->config.pcm_port && shared_->pcm_graph) {
            NodusHeadlessPublishResult result{};
            if (!nodus_headless_publish_output_token(shared_->pcm_graph, kPcmPort,
                    &frame_.pcm, sizeof(frame_.pcm), &result) || !result.accepted || result.dropped)
                shared_->pcm_dropped.fetch_add(1);
        }
        const size_t n = shared_->config.fft_size;
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(pcm[2*i]) || !std::isfinite(pcm[2*i+1])) {
                captured_since_fft_ = 0;
                std::fill(left_.begin(), left_.end(), 0.0f);
                std::fill(right_.begin(), right_.end(), 0.0f);
                ctx.succeeded = false; return;
            }
            left_[ring_write_] = pcm[2*i]; right_[ring_write_] = pcm[2*i+1];
            ring_write_ = (ring_write_ + 1) % n;
        }
        captured_since_fft_ += count;
        shared_->captured.store(captured, std::memory_order_relaxed);
        shared_->sample_rate.store(sample_rate, std::memory_order_relaxed);
        const uint64_t hop = n / shared_->config.overlap;
        if (captured_since_fft_ < hop) {
            ctx.succeeded = false; return;
        }
        captured_since_fft_ %= hop;
        auto input = shared_->fft_states->borrow_cpu(Shared::fft_card, Shared::fft_input, true);
        if (!input || input.bytes() < 2 * n * sizeof(float)) {
            ctx.succeeded = false; return;
        }
        auto* values = static_cast<float*>(input.data());
        double power = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const size_t at = (ring_write_ + i) % n;
            values[i] = left_[at]; values[n+i] = right_[at];
            power += 0.5 * (double(values[i])*values[i] + double(values[n+i])*values[n+i]);
            const float window = 0.5f - 0.5f * std::cos(
                6.2831853071795864769f * static_cast<float>(i) / static_cast<float>(n - 1));
            values[i] *= window; values[n+i] *= window;
        }
        input = {};
        frame_.sample_rate = sample_rate;
        frame_.captured_frames = read_cursor_;
        frame_.rms = static_cast<float>(std::sqrt(power / n));
        if (nodus_tensor_write(work_token_, 9 * sizeof(uint64_t),
                &frame_.captured_frames, sizeof(uint64_t)) != sizeof(uint64_t) ||
            !raw_stack_push_abstract_tensor(*ctx.stack.raw, {work_token_}) ||
            !raw_stack_push_abstract_tensor(*ctx.stack.raw, {config_token_})) {
            ctx.succeeded = false; return;
        }
    }
private:
    std::shared_ptr<Shared> shared_;
    HMODULE module_ = nullptr;
    void* scope_ = nullptr;
    scope_feed_fn feed_ = nullptr;
    SDL_AudioStream* stream_ = nullptr;
    bool audio_ready_ = false;
    bool end_published_ = false;
    std::vector<float> samples_;
    size_t position_ = 0;
    uint32_t rate_ = 0;
    scope_create_capture_fn create_capture_ = nullptr;
    scope_start_fn start_ = nullptr;
    scope_read_fn read_ = nullptr;
    scope_running_fn running_ = nullptr;
    scope_frame_count_fn frame_count_ = nullptr;
    scope_free_fn free_ = nullptr;
    uint64_t read_cursor_ = 0;
    uint64_t captured_since_fft_ = 0;
    size_t ring_write_ = 0;
    std::vector<float> left_, right_;
    uint64_t config_token_ = 0, work_token_ = 0;
    RadioFrame frame_{};
};

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

class RadioFftBands final : public ITool {
public:
    explicit RadioFftBands(std::shared_ptr<Shared> shared) : shared_(std::move(shared)) {}
    const char* id_cstr() const noexcept override { return "radio.fft_bands"; }
    const char* name_cstr() const noexcept override { return "Radio fftfree band projection"; }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        ctx.succeeded = false;
        if (!ctx.stack.raw || !shared_->current_frame || !shared_->fft_states) return;
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

class RadioUdp final : public ITool {
public:
    explicit RadioUdp(std::shared_ptr<Shared> shared) : shared_(std::move(shared)) {}
    ~RadioUdp() noexcept override { shutdown(); }
    const char* id_cstr() const noexcept override { return "radio.udp_broadcast"; }
    const char* name_cstr() const noexcept override { return "Radio UDP broadcast"; }
    void initialize(const ToolInitContext&) override {
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return;
        winsock_ready_ = true;
        addrinfo hints{};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_DGRAM;
        hints.ai_protocol = IPPROTO_UDP;
        addrinfo* addresses = nullptr;
        const std::string service = std::to_string(shared_->config.port);
        if (getaddrinfo(shared_->config.host.c_str(), service.c_str(), &hints, &addresses) != 0)
            return;
        socket_ = socket(addresses->ai_family, addresses->ai_socktype, addresses->ai_protocol);
        if (socket_ != INVALID_SOCKET && addresses->ai_addrlen <= sizeof(destination_)) {
            std::memcpy(&destination_, addresses->ai_addr, addresses->ai_addrlen);
            destination_bytes_ = static_cast<int>(addresses->ai_addrlen);
            sockaddr_in local{};
            local.sin_family = AF_INET;
            local.sin_addr.s_addr = htonl(INADDR_ANY);
            if (bind(socket_, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) != 0) {
                closesocket(socket_); socket_ = INVALID_SOCKET; freeaddrinfo(addresses); return;
            }
            int local_size = sizeof(local);
            if (getsockname(socket_, reinterpret_cast<sockaddr*>(&local), &local_size) == 0) {
                std::printf("radio_udp_source_port=%u\n", static_cast<unsigned>(ntohs(local.sin_port)));
                std::fflush(stdout);
            }
            u_long nonblocking = 1;
            ioctlsocket(socket_, FIONBIO, &nonblocking);
        }
        freeaddrinfo(addresses);
    }
    void shutdown() override {
        if (socket_ != INVALID_SOCKET) { closesocket(socket_); socket_ = INVALID_SOCKET; }
        if (winsock_ready_) { WSACleanup(); winsock_ready_ = false; }
    }
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        void* frame_ptr = nullptr;
        const ValueTypeId pointer_type = ValueTypeRegistry::global().builtin(VT_VOID_PTR);
        if (!ctx.stack.raw || !raw_stack_pop_typed(*ctx.stack.raw, &frame_ptr, pointer_type)) {
            ctx.succeeded = false;
            return;
        }
        const auto* frame = static_cast<const RadioFrame*>(frame_ptr);
        if (frame && frame->magic == kFrameMagic && frame->terminal) return;
        if (!frame || frame->magic != kFrameMagic || frame->band_count != shared_->config.bands ||
            socket_ == INVALID_SOCKET || !destination_bytes_) {
            shared_->dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        // NRAD v1, little-endian: magic, version, bands, sample_rate,
        // captured-frame index, RMS, then one float32 power per band.
        std::array<uint8_t, 28 + kMaxBands * sizeof(float)> packet{};
        size_t offset = 0;
        auto append = [&](const void* data, size_t bytes) {
            std::memcpy(packet.data() + offset, data, bytes);
            offset += bytes;
        };
        const uint32_t magic = kFrameMagic;
        append(&magic, sizeof(magic));
        append(&kPacketVersion, sizeof(kPacketVersion));
        append(&frame->band_count, sizeof(frame->band_count));
        append(&frame->sample_rate, sizeof(frame->sample_rate));
        append(&frame->captured_frames, sizeof(frame->captured_frames));
        append(&frame->rms, sizeof(frame->rms));
        append(frame->power.data(), frame->band_count * sizeof(float));
        const int sent = sendto(socket_, reinterpret_cast<const char*>(packet.data()),
                                static_cast<int>(offset), 0,
                                reinterpret_cast<const sockaddr*>(&destination_), destination_bytes_);
        if (sent == static_cast<int>(offset)) shared_->emitted.fetch_add(1, std::memory_order_relaxed);
        else shared_->dropped.fetch_add(1, std::memory_order_relaxed);
    }
private:
    std::shared_ptr<Shared> shared_;
    bool winsock_ready_ = false;
    SOCKET socket_ = INVALID_SOCKET;
    sockaddr_storage destination_{};
    int destination_bytes_ = 0;
};

// Independent PCM destination: RTP v2 / dynamic PT 96 / L16 48000 Hz stereo.
// NRAD is unchanged. The sender consumes the real canvas output FIFO, not FFT.
class RadioPcmUdp final : public ITool {
public:
    explicit RadioPcmUdp(std::shared_ptr<Shared> shared) : shared_(std::move(shared)) {}
    ~RadioPcmUdp() noexcept override { shutdown(); }
    const char* id_cstr() const noexcept override { return "radio.pcm_rtp"; }
    const char* name_cstr() const noexcept override { return "Radio PCM RTP"; }
    void initialize(const ToolInitContext&) override {
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data)) return;
        ready_ = true;
        socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_ == INVALID_SOCKET) return;
        sockaddr_in local{}; local.sin_family = AF_INET; local.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(socket_, reinterpret_cast<sockaddr*>(&local), sizeof(local))) return;
        u_long nonblocking = 1; ioctlsocket(socket_, FIONBIO, &nonblocking);
        int size = sizeof(local); getsockname(socket_, reinterpret_cast<sockaddr*>(&local), &size);
        destination_.sin_family = AF_INET;
        destination_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        destination_.sin_port = htons(shared_->config.pcm_port);
        std::random_device random;
        ssrc_ = random(); sequence_ = static_cast<uint16_t>(random()); timestamp_base_ = random();
        std::printf("radio_pcm_source_port=%u rtp_payload=96 encoding=L16 rate=48000 channels=2 ssrc=%u\n",
            static_cast<unsigned>(ntohs(local.sin_port)), ssrc_);
        std::fflush(stdout);
    }
    void shutdown() override {
        if (socket_ != INVALID_SOCKET) { closesocket(socket_); socket_ = INVALID_SOCKET; }
        if (ready_) { WSACleanup(); ready_ = false; }
    }
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext&) override {
        if (!shared_->pcm_graph || socket_ == INVALID_SOCKET) return;
        if (end_pending_) {
            if (std::chrono::steady_clock::now() >= next_end_) send_end();
            return;
        }
        PcmBlock block{}; size_t written = 0;
        if (!nodus_headless_pull_token(shared_->pcm_graph, kPcmPort, &block, sizeof(block), &written)) return;
        if (written != sizeof(block) || block.frames > 256 || block.sample_rate != 48000) {
            shared_->pcm_dropped.fetch_add(1); return;
        }
        if (!block.frames) {
            end_timestamp_ = timestamp_base_ + static_cast<uint32_t>(block.first_frame);
            end_pending_ = true; send_end(); return;
        }
        std::array<uint8_t, 12 + 1024> packet{};
        packet[0] = 0x80; packet[1] = 96;
        auto u16 = [&](size_t pos, uint16_t x) { packet[pos] = static_cast<uint8_t>(x >> 8); packet[pos + 1] = static_cast<uint8_t>(x); };
        auto u32 = [&](size_t pos, uint32_t x) { u16(pos, static_cast<uint16_t>(x >> 16)); u16(pos + 2, static_cast<uint16_t>(x)); };
        u16(2, sequence_++); u32(4, timestamp_base_ + static_cast<uint32_t>(block.first_frame)); u32(8, ssrc_);
        for (size_t i = 0; i < block.frames * 2; ++i) {
            const float sample = std::clamp(block.stereo[i], -1.0f, 1.0f);
            const int value = static_cast<int>(std::lround(sample * 32767.0f));
            u16(12 + i * 2, static_cast<uint16_t>(static_cast<int16_t>(value)));
        }
        const int bytes = 12 + block.frames * 4;
        if (sendto(socket_, reinterpret_cast<const char*>(packet.data()), bytes, 0,
                   reinterpret_cast<const sockaddr*>(&destination_), sizeof(destination_)) != bytes)
            shared_->pcm_dropped.fetch_add(1);
        else { shared_->pcm_emitted.fetch_add(1); octets_ += block.frames * 4; }
    }
private:
    void send_end() {
        // Standard compound RTCP (RFC3550), muxed with RTP (RFC5761):
        // SR provides the exclusive source timestamp; SDES CNAME; BYE.
        // No recipient identity or file name is carried in the CNAME.
        std::array<uint8_t, 52> packet{};
        auto u16 = [&](size_t pos, uint16_t x) { packet[pos] = static_cast<uint8_t>(x >> 8); packet[pos+1] = static_cast<uint8_t>(x); };
        auto u32 = [&](size_t pos, uint32_t x) { u16(pos, static_cast<uint16_t>(x >> 16)); u16(pos+2, static_cast<uint16_t>(x)); };
        packet[0] = 0x80; packet[1] = 200; u16(2, 6); u32(4, ssrc_);
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        u32(8, static_cast<uint32_t>(ns / 1000000000LL + 2208988800ULL));
        u32(12, static_cast<uint32_t>((static_cast<uint64_t>(ns % 1000000000LL) << 32) / 1000000000ULL));
        u32(16, end_timestamp_); u32(20, static_cast<uint32_t>(shared_->pcm_emitted.load())); u32(24, octets_);
        packet[28] = 0x81; packet[29] = 202; u16(30, 3); u32(32, ssrc_);
        packet[36] = 1; packet[37] = 5; std::memcpy(packet.data()+38, "radio", 5);
        packet[44] = 0x81; packet[45] = 203; u16(46, 1); u32(48, ssrc_);
        const int sent = sendto(socket_, reinterpret_cast<const char*>(packet.data()),
            static_cast<int>(packet.size()), 0, reinterpret_cast<const sockaddr*>(&destination_), sizeof(destination_));
        std::printf("radio_pcm_end timestamp=%u sent=%d attempt=%u\n", end_timestamp_, sent, ++end_attempts_);
        std::fflush(stdout);
        next_end_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
        if (end_attempts_ >= 3) end_pending_ = false;
    }
    std::shared_ptr<Shared> shared_;
    SOCKET socket_ = INVALID_SOCKET;
    sockaddr_in destination_{};
    bool ready_ = false;
    uint16_t sequence_ = 0;
    uint32_t ssrc_ = 0, timestamp_base_ = 0;
    uint32_t octets_ = 0, end_timestamp_ = 0, end_attempts_ = 0;
    bool end_pending_ = false;
    std::chrono::steady_clock::time_point next_end_{};
};

ToolRegistry::Entry entry(const char* id, const char* name,
                          ToolRegistry::Factory factory) {
    ToolRegistry::Entry result;
    result.id = id;
    result.name = name;
    result.caps = ToolCaps::Table;
    result.factory = std::move(factory);
    return result;
}
} // namespace

extern "C" int32_t nodus_headless_radio_configure(
    const wchar_t* scope_dll, const wchar_t* fftfree_dll,
    const char* destination_host, uint16_t destination_port,
    uint32_t fft_size, uint32_t overlap, uint32_t band_count,
    float min_hz, float max_hz) {
    if (!scope_dll || !*scope_dll || !fftfree_dll || !*fftfree_dll ||
        !destination_host || !*destination_host || !destination_port ||
        fft_size < 256 || (fft_size & (fft_size - 1)) ||
        !overlap || (overlap & (overlap - 1)) || overlap > fft_size ||
        !band_count || band_count > kMaxBands ||
        !std::isfinite(min_hz) || !std::isfinite(max_hz) ||
        min_hz <= 0 || max_hz <= min_hz) return 0;
    std::lock_guard<std::mutex> lock(config_mutex);
    if (configured) return 0;
    auto shared = std::make_shared<Shared>();
    shared->config = {scope_dll, fftfree_dll, destination_host, destination_port,
                      fft_size, overlap, band_count, min_hz, max_hz};
    shared->fft_states = std::make_unique<nodus::gpu::TensorStateLibraryTool>(
        16ull << 20, std::vector<uint64_t>{}, false); // Radio FFT is CPU-only.
    shared->fft_states->initialize({});
    if (!radio_register_fft_states(*shared)) return 0;
    auto& registry = tool_registry_global();
    if (!nodus::gpu::register_fftfree_tool(registry, *shared->fft_states,
                                           std::filesystem::path(fftfree_dll).string())) return 0;
    if (!registry.register_tool(entry("radio.system_output", "Radio PCM source",
        [shared] { return ToolRegistry::Factory::result_type(new RadioSource(shared), [](ITool* p) { delete p; }); }))) {
        registry.unregister_tool("nodus.fftfree"); return 0;
    }
    if (!registry.register_tool(entry("radio.fft_bands", "Radio fftfree band projection",
        [shared] { return ToolRegistry::Factory::result_type(new RadioFftBands(shared), [](ITool* p) { delete p; }); }))) {
        registry.unregister_tool("radio.system_output");
        registry.unregister_tool("nodus.fftfree"); return 0;
    }
    if (!registry.register_tool(entry("radio.udp_broadcast", "Radio UDP broadcast",
        [shared] { return ToolRegistry::Factory::result_type(new RadioUdp(shared), [](ITool* p) { delete p; }); }))) {
        registry.unregister_tool("radio.system_output");
        registry.unregister_tool("radio.fft_bands");
        registry.unregister_tool("nodus.fftfree");
        return 0;
    }
    configured = std::move(shared);
    return 1;
}

extern "C" int32_t nodus_headless_radio_file_source(const wchar_t* wav_path, int32_t loop) {
    if (!wav_path || !*wav_path) return 0;
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || configured->running.load() || configured->captured.load()) return 0;
    configured->config.wav = wav_path;
    configured->config.loop = loop != 0;
    auto shared = configured;
    return tool_registry_global().register_tool(entry("radio.file_output", "Radio WAV playback + fftfree",
        [shared] { return ToolRegistry::Factory::result_type(new RadioSource(shared), [](ITool* p) { delete p; }); })) ? 1 : 0;
}

extern "C" int32_t nodus_headless_radio_band_scale(int32_t linear) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || configured->running.load() || configured->captured.load() ||
        (linear != 0 && linear != 1)) return 0;
    configured->config.linear_bands = linear != 0;
    return 1;
}

extern "C" int32_t nodus_headless_radio_hold(int32_t held) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || configured->config.wav.empty()) return 0;
    configured->held.store(held != 0);
    return 1;
}

extern "C" int32_t nodus_headless_radio_pcm_configure(uint16_t port) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || !port || configured->config.wav.empty() || configured->running.load() ||
        configured->config.fft_size / configured->config.overlap != 256) return 0;
    auto shared = configured;
    if (!tool_registry_global().register_tool(entry("radio.pcm_rtp", "Radio PCM RTP",
        [shared] { return ToolRegistry::Factory::result_type(new RadioPcmUdp(shared), [](ITool* p) { delete p; }); }))) return 0;
    if (!tool_registry_global().register_tool(entry("radio.pcm_s16le", "Radio PCM s16le",
        [] { return ToolRegistry::Factory::result_type(new RadioPcmS16le(), [](ITool* p) { delete p; }); }))) {
        tool_registry_global().unregister_tool("radio.pcm_rtp");
        return 0;
    }
    configured->config.pcm_port = port;
    return 1;
}
extern "C" int32_t nodus_headless_radio_pcm_bind_at(NodusHeadlessGraph* graph, int32_t module, int32_t row) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || !configured->config.pcm_port || configured->captured.load() || row < 0) return 0;
    NodusHeadlessFifoConfig fifo{};
    fifo.token_bytes = sizeof(PcmBlock); fifo.slots = 16; fifo.top_k = 8;
    fifo.type_id = -1; fifo.layout = 2; fifo.dtype = 0; fifo.backend_type = GP_MEM_BACKEND_CPU;
    if (!nodus_headless_declare_output_port_ex(graph, kPcmPort, module, row, &fifo, 1)) return 0;
    configured->pcm_graph = graph;
    return 1;
}
extern "C" int32_t nodus_headless_radio_pcm_bind(NodusHeadlessGraph* graph, int32_t module) {
    return nodus_headless_radio_pcm_bind_at(graph, module, 2);
}

extern "C" int32_t nodus_headless_radio_pcm_status(uint64_t* emitted, uint64_t* dropped) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || !emitted || !dropped) return 0;
    *emitted = configured->pcm_emitted.load(); *dropped = configured->pcm_dropped.load();
    return 1;
}

extern "C" int32_t nodus_headless_radio_status(NodusRadioStatus* out_status) {
    if (!out_status) return 0;
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured) return 0;
    out_status->captured_frames = configured->captured.load(std::memory_order_relaxed);
    out_status->emitted_packets = configured->emitted.load(std::memory_order_relaxed);
    out_status->dropped_packets = configured->dropped.load(std::memory_order_relaxed);
    out_status->sample_rate = configured->sample_rate.load(std::memory_order_relaxed);
    out_status->capture_running = configured->running.load(std::memory_order_relaxed);
    return 1;
}
extern "C" int32_t nodus_headless_radio_capture_overruns(uint64_t* lost_frames) {
    if (!lost_frames) return 0;
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured) return 0;
    *lost_frames = configured->capture_overruns.load(std::memory_order_relaxed);
    return 1;
}
#else
extern "C" int32_t nodus_headless_radio_hold(int32_t) { return 0; }
extern "C" int32_t nodus_headless_radio_configure(
    const wchar_t*, const wchar_t*, const char*, uint16_t,
    uint32_t, uint32_t, uint32_t, float, float) { return 0; }
extern "C" int32_t nodus_headless_radio_file_source(const wchar_t*, int32_t) { return 0; }
extern "C" int32_t nodus_headless_radio_band_scale(int32_t) { return 0; }
extern "C" int32_t nodus_headless_radio_pcm_configure(uint16_t) { return 0; }
extern "C" int32_t nodus_headless_radio_pcm_bind(NodusHeadlessGraph*, int32_t) { return 0; }
extern "C" int32_t nodus_headless_radio_pcm_status(uint64_t*, uint64_t*) { return 0; }
extern "C" int32_t nodus_headless_radio_status(NodusRadioStatus*) { return 0; }
extern "C" int32_t nodus_headless_radio_capture_overruns(uint64_t*) { return 0; }
#endif
