// radio_source: the Radio PCM source rows (radio.system_output / radio.file_output) (src/headless/radio_rows.cpp registers it).
// The row tool's source, in the module library (case 74fd8157): the Menu lists it, the canvas compiler inlines it, it
// builds as its own plugin DLL, and the image registers it from there. The Radio configuration is not the row's saved
// state: it lives in Shared (radio_shared.h). Shared is process state (the registration captures it in the factory); a row made with no Shared refuses in execute_stack.
#include "radio_shared.h"

#if defined(_WIN32)
namespace nodus::radio {

class RadioSource final : public ITool {
public:
    // A 270 s stereo track at 48 kHz occupies about 104 MiB as F32 PCM.
    // Keep the existing bounded file-source path while admitting complete songs.
    static constexpr std::uint64_t kMaxFilePcmBytes = 256ull * 1024ull * 1024ull;
    explicit RadioSource(std::shared_ptr<Shared> shared) : shared_(std::move(shared)) {}
    RadioSource() = default; // no Shared: the row refuses in execute_stack
    ~RadioSource() noexcept override { shutdown(); }
    const char* id_cstr() const noexcept override { return shared_ && !shared_->config.wav.empty() ? "radio.file_output" : "radio.system_output"; }
    const char* name_cstr() const noexcept override { return "Radio PCM source"; }
    void initialize(const ToolInitContext&) override {
        if (!shared_) return;
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
            SDL_AudioSpec target{SDL_AUDIO_F32, 2, 48000};
            const auto decoders = nodus::exec::environment_paths("NODUS_RADIO_DECODER");
            if (!decoders.empty()) {
                // Operator-only executable declaration, never supplied by a song or HTTP caller.
                if (decoders.size() != 1 || !std::filesystem::path(decoders[0]).is_absolute() ||
                    !std::filesystem::is_regular_file(decoders[0]) || !std::filesystem::is_regular_file(c.wav)) return;
                decoder_program_ = {decoders[0], {"-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
                    "-i", std::filesystem::path(c.wav).string(), "-vn", "-f", "f32le", "-ar", "48000", "-ac", "2", "pipe:1"},
                    std::filesystem::current_path().string(), nodus::exec::declared_core_roots(), nodus::exec::declared_work_roots()};
                if (!start_decoder()) return;
                decoding_ = true;
                rate_ = 48000;
            } else {
            std::error_code size_error;
            const auto bytes = std::filesystem::file_size(c.wav, size_error);
            if (size_error || bytes > kMaxFilePcmBytes) return;
            SDL_AudioSpec spec{};
            Uint8* original = nullptr;
            Uint32 length = 0;
            const auto utf8 = std::filesystem::path(c.wav).u8string();
            if (!SDL_LoadWAV(reinterpret_cast<const char*>(utf8.c_str()), &spec, &original, &length)) return;
            const int source_frame_bytes = SDL_AUDIO_FRAMESIZE(spec);
            if (source_frame_bytes <= 0 || spec.freq <= 0 ||
                (static_cast<uint64_t>(length) / source_frame_bytes) * 8 * (c.pcm_port ? 48000 : spec.freq) / spec.freq > kMaxFilePcmBytes) {
                SDL_free(original); return;
            }
            target.freq = c.pcm_port ? 48000 : spec.freq;
            Uint8* converted = nullptr;
            int converted_length = 0;
            const bool ok = length <= INT_MAX && SDL_ConvertAudioSamples(&spec, original, static_cast<int>(length),
                &target, &converted, &converted_length);
            SDL_free(original);
            if (!ok) return;
            if (converted_length <= 0 || converted_length > static_cast<int>(kMaxFilePcmBytes) || converted_length % (2 * sizeof(float))) { SDL_free(converted); return; }
            samples_.resize(static_cast<size_t>(converted_length) / sizeof(float));
            std::memcpy(samples_.data(), converted, converted_length);
            SDL_free(converted);
            for (float value : samples_) if (!std::isfinite(value)) { samples_.clear(); return; }
            rate_ = static_cast<uint32_t>(target.freq);
            }
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
        if (shared_ && shared_->current_frame == &frame_) shared_->current_frame = nullptr;
        if (config_token_) { nodus_tensor_destroy(config_token_); config_token_ = 0; }
        if (work_token_) { nodus_tensor_destroy(work_token_); work_token_ = 0; }
        decoder_.stop();
        if (stream_) { SDL_DestroyAudioStream(stream_); stream_ = nullptr; }
        if (audio_ready_) { SDL_QuitSubSystem(SDL_INIT_AUDIO); audio_ready_ = false; }
        if (scope_ && free_) { free_(scope_); scope_ = nullptr; }
        if (module_) { FreeLibrary(module_); module_ = nullptr; }
        if (shared_) shared_->running.store(0, std::memory_order_release);
    }
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        if (!shared_) { ctx.succeeded = false; return; }
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
            if (decoding_ && position_ == samples_.size()) {
                const size_t bytes = hop * 2 * sizeof(float);
                decoder_.take(decoded_bytes_, bytes - decoded_bytes_.size());
                const bool ended = decoder_.output_drained() && !decoder_.running();
                if (ended && (decoder_.exit_code() != 0 || decoded_bytes_.size() % (2 * sizeof(float)))) {
                    std::fprintf(stderr, "Radio decoder failed or returned a partial stereo frame\n");
                    shared_->running.store(0); ctx.succeeded = false; return;
                }
                if (decoded_bytes_.size() < bytes && !ended) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1)); ctx.succeeded = false; return;
                }
                samples_.resize(decoded_bytes_.size() / sizeof(float));
                if (!samples_.empty()) std::memcpy(samples_.data(), decoded_bytes_.data(), decoded_bytes_.size());
                decoded_bytes_.clear(); position_ = 0;
                for (float value : samples_) if (!std::isfinite(value)) {
                    shared_->running.store(0); ctx.succeeded = false; return;
                }
                if (samples_.empty() && ended && shared_->config.loop) {
                    if (!decoder_had_samples_ || !start_decoder()) shared_->running.store(0);
                    ctx.succeeded = false; return;
                }
                if (!samples_.empty()) decoder_had_samples_ = true;
            }
            if (position_ == samples_.size()) {
                if (shared_->config.loop && !decoding_) position_ = 0;
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
    bool start_decoder() {
        std::string why;
        if (!decoder_.start(decoder_program_, why, false, 65536)) {
            std::fprintf(stderr, "Radio decoder: %s\n", why.c_str()); return false;
        }
        decoder_.finish_input();
        decoder_had_samples_ = false;
        return true;
    }
    nodus::exec::Child decoder_;
    nodus::exec::Program decoder_program_;
    bool decoding_ = false, decoder_had_samples_ = false;
    std::string decoded_bytes_;
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

} // namespace nodus::radio

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::radio::RadioSource(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
#endif
