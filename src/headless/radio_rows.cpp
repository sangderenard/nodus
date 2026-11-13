#include "nodus_headless_abi.h"
#include "tool_api.h"
#include "tool_registry.h"

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
    std::wstring wav;
    bool loop = false;
    uint16_t pcm_port = 0;
};

struct Shared {
    Config config;
    NodusHeadlessGraph* pcm_graph = nullptr;
    std::atomic<uint64_t> pcm_dropped{0};
    std::atomic<uint64_t> pcm_emitted{0};
    std::atomic<uint64_t> captured{0};
    std::atomic<uint64_t> emitted{0};
    std::atomic<uint64_t> dropped{0};
    std::atomic<uint32_t> sample_rate{0};
    std::atomic<int32_t> running{0};
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
    bool terminal = false; // Internal row control, never part of NRAD.
};

using scope_create_fn = void* (__cdecl *)(const wchar_t*, size_t, size_t, size_t, float, float);
using scope_start_fn = int (__cdecl *)(void*);
using scope_snapshot_fn = size_t (__cdecl *)(void*, float, float*, float*, float*, size_t, float*, uint32_t*);
using scope_running_fn = int (__cdecl *)(void*);
using scope_frame_count_fn = uint64_t (__cdecl *)(void*);
using scope_feed_fn = int (__cdecl *)(void*, const float*, size_t, uint32_t);
using scope_free_fn = void (__cdecl *)(void*);

class RadioSource final : public ITool {
public:
    explicit RadioSource(std::shared_ptr<Shared> shared) : shared_(std::move(shared)) {}
    ~RadioSource() noexcept override { shutdown(); }
    const char* id_cstr() const noexcept override { return shared_->config.wav.empty() ? "radio.system_output" : "radio.file_output"; }
    const char* name_cstr() const noexcept override { return "Radio system output + fftfree"; }
    void initialize(const ToolInitContext&) override {
        module_ = LoadLibraryW(shared_->config.scope_dll.c_str());
        if (!module_) return;
        create_ = reinterpret_cast<scope_create_fn>(GetProcAddress(module_, "openrgb_audio_scope_create"));
        start_ = reinterpret_cast<scope_start_fn>(GetProcAddress(module_, "openrgb_audio_scope_start"));
        snapshot_ = reinterpret_cast<scope_snapshot_fn>(GetProcAddress(module_, "openrgb_audio_scope_snapshot"));
        running_ = reinterpret_cast<scope_running_fn>(GetProcAddress(module_, "openrgb_audio_scope_running"));
        frame_count_ = reinterpret_cast<scope_frame_count_fn>(GetProcAddress(module_, "openrgb_audio_scope_frame_count"));
        free_ = reinterpret_cast<scope_free_fn>(GetProcAddress(module_, "openrgb_audio_scope_free"));
        if (!create_ || !start_ || !snapshot_ || !running_ || !frame_count_ || !free_) return;
        const Config& c = shared_->config;
        scope_ = create_(c.fftfree_dll.c_str(), c.fft_size, c.overlap,
                         c.bands, c.min_hz, c.max_hz);
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
    }
    void shutdown() override {
        if (stream_) { SDL_DestroyAudioStream(stream_); stream_ = nullptr; }
        if (audio_ready_) { SDL_QuitSubSystem(SDL_INIT_AUDIO); audio_ready_ = false; }
        if (scope_ && free_) { free_(scope_); scope_ = nullptr; }
        if (module_) { FreeLibrary(module_); module_ = nullptr; }
        shared_->running.store(0, std::memory_order_release);
    }
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        if (!scope_ || !ctx.stack.raw) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            ctx.succeeded = false;
            return;
        }
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
                    // Refusal stops following rows in ThreadManager. Supply a
                    // terminal control value so the PCM row can drain its FIFO
                    // and send BYE without emitting fictitious spectrum frames.
                    frame_.terminal = true;
                    void* terminal = &frame_;
                    const auto type = ValueTypeRegistry::global().builtin(VT_VOID_PTR);
                    ctx.succeeded = raw_stack_push_typed(*ctx.stack.raw, &terminal, type);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1)); return;
                }
            }
            const size_t count = std::min(hop, (samples_.size() - position_) / 2);
            if (!SDL_PutAudioStreamData(stream_, samples_.data() + position_, static_cast<int>(count * 2 * sizeof(float))) ||
                !feed_(scope_, samples_.data() + position_, count, rate_)) {
                shared_->running.store(0); ctx.succeeded = false; return;
            }
            if (shared_->config.pcm_port && shared_->pcm_graph) {
                PcmBlock block{};
                block.frames = static_cast<uint32_t>(count);
                block.first_frame = frame_count_(scope_) - count;
                std::copy_n(samples_.data() + position_, count * 2, block.stereo.data());
                NodusHeadlessPublishResult result{};
                if (!nodus_headless_publish_output_token(shared_->pcm_graph, kPcmPort, &block, sizeof(block), &result) ||
                    !result.accepted || result.dropped) shared_->pcm_dropped.fetch_add(1);
            }
            position_ += count * 2;
        }
        const uint64_t captured = frame_count_(scope_);
        shared_->running.store(running_(scope_), std::memory_order_release);
        const uint64_t hop = shared_->config.fft_size / shared_->config.overlap;
        if (captured < last_emitted_ || captured - last_emitted_ < hop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ctx.succeeded = false;
            return;
        }
        std::array<float, kMaxBands * 2> xy{};
        std::array<float, kMaxBands> frequencies{};
        uint32_t sample_rate = 0;
        float rms = 0;
        const size_t count = snapshot_(scope_, 1.0f, xy.data(), frame_.power.data(),
                                       frequencies.data(), kMaxBands, &rms, &sample_rate);
        if (count != shared_->config.bands || !sample_rate) {
            ctx.succeeded = false;
            return;
        }
        frame_.band_count = static_cast<uint32_t>(count);
        frame_.sample_rate = sample_rate;
        frame_.captured_frames = captured;
        frame_.rms = rms;
        void* frame_ptr = &frame_;
        const ValueTypeId pointer_type = ValueTypeRegistry::global().builtin(VT_VOID_PTR);
        if (!raw_stack_push_typed(*ctx.stack.raw, &frame_ptr, pointer_type)) {
            ctx.succeeded = false;
            return;
        }
        last_emitted_ = captured;
        shared_->captured.store(captured, std::memory_order_relaxed);
        shared_->sample_rate.store(sample_rate, std::memory_order_relaxed);
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
    scope_create_fn create_ = nullptr;
    scope_start_fn start_ = nullptr;
    scope_snapshot_fn snapshot_ = nullptr;
    scope_running_fn running_ = nullptr;
    scope_frame_count_fn frame_count_ = nullptr;
    scope_free_fn free_ = nullptr;
    uint64_t last_emitted_ = 0;
    RadioFrame frame_{};
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
    auto& registry = tool_registry_global();
    if (!registry.register_tool(entry("radio.system_output", "Radio system output + fftfree",
        [shared] { return ToolRegistry::Factory::result_type(new RadioSource(shared), [](ITool* p) { delete p; }); }))) return 0;
    if (!registry.register_tool(entry("radio.udp_broadcast", "Radio UDP broadcast",
        [shared] { return ToolRegistry::Factory::result_type(new RadioUdp(shared), [](ITool* p) { delete p; }); }))) {
        registry.unregister_tool("radio.system_output");
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

extern "C" int32_t nodus_headless_radio_pcm_configure(uint16_t port) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || !port || configured->config.wav.empty() || configured->running.load() ||
        configured->config.fft_size / configured->config.overlap != 256) return 0;
    auto shared = configured;
    if (!tool_registry_global().register_tool(entry("radio.pcm_rtp", "Radio PCM RTP",
        [shared] { return ToolRegistry::Factory::result_type(new RadioPcmUdp(shared), [](ITool* p) { delete p; }); }))) return 0;
    configured->config.pcm_port = port;
    return 1;
}
extern "C" int32_t nodus_headless_radio_pcm_bind(NodusHeadlessGraph* graph, int32_t module) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || !configured->config.pcm_port || configured->captured.load()) return 0;
    NodusHeadlessFifoConfig fifo{};
    fifo.token_bytes = sizeof(PcmBlock); fifo.slots = 16; fifo.top_k = 8;
    fifo.type_id = -1; fifo.layout = 2; fifo.dtype = 0; fifo.backend_type = GP_MEM_BACKEND_CPU;
    if (!nodus_headless_declare_output_port_ex(graph, kPcmPort, module, 2, &fifo, 1)) return 0;
    configured->pcm_graph = graph;
    return 1;
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
#else
extern "C" int32_t nodus_headless_radio_configure(
    const wchar_t*, const wchar_t*, const char*, uint16_t,
    uint32_t, uint32_t, uint32_t, float, float) { return 0; }
extern "C" int32_t nodus_headless_radio_file_source(const wchar_t*, int32_t) { return 0; }
extern "C" int32_t nodus_headless_radio_pcm_configure(uint16_t) { return 0; }
extern "C" int32_t nodus_headless_radio_pcm_bind(NodusHeadlessGraph*, int32_t) { return 0; }
extern "C" int32_t nodus_headless_radio_pcm_status(uint64_t*, uint64_t*) { return 0; }
extern "C" int32_t nodus_headless_radio_status(NodusRadioStatus*) { return 0; }
#endif
