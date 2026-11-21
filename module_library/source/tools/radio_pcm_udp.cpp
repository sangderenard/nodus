// radio_pcm_udp: the Radio PCM RTP row (src/headless/radio_rows.cpp registers it).
// The row tool's source, in the module library (case 74fd8157): the Menu lists it, the canvas compiler inlines it, it
// builds as its own plugin DLL, and the image registers it from there. The Radio configuration is not the row's saved
// state: it lives in Shared (radio_shared.h). Shared is process state (the registration captures it in the factory); a row made with no Shared refuses in execute_stack.
#include "radio_shared.h"

namespace nodus::radio {

// Independent PCM destination: RTP v2 / dynamic PT 96 / L16 48000 Hz stereo.
// NRAD is unchanged. The sender consumes the real canvas output FIFO, not FFT.
class RadioPcmUdp final : public ITool {
public:
    explicit RadioPcmUdp(std::shared_ptr<Shared> shared) : shared_(std::move(shared)) {}
    RadioPcmUdp() = default; // no Shared: the row refuses in execute_stack
    ~RadioPcmUdp() noexcept override { shutdown(); }
    const char* id_cstr() const noexcept override { return "radio.pcm_rtp"; }
    const char* name_cstr() const noexcept override { return "Radio PCM RTP"; }
    void initialize(const ToolInitContext&) override {
        if (!shared_) return;
        if (!platform::network_start()) return;
        ready_ = true;
        socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_ == platform::kInvalidSocket) return;
        sockaddr_in local{}; local.sin_family = AF_INET; local.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(socket_, reinterpret_cast<sockaddr*>(&local), sizeof(local))) return;
        platform::set_nonblocking(socket_);
        platform::socket_length size = sizeof(local); getsockname(socket_, reinterpret_cast<sockaddr*>(&local), &size);
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
        if (socket_ != platform::kInvalidSocket) { platform::close_socket(socket_); socket_ = platform::kInvalidSocket; }
        if (ready_) { platform::network_stop(); ready_ = false; }
    }
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        if (!shared_) { ctx.succeeded = false; return; }
        if (!shared_->pcm_graph || socket_ == platform::kInvalidSocket) return;
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
        if (platform::send_datagram(socket_, packet.data(), bytes,
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
        const int sent = platform::send_datagram(socket_, packet.data(),
            packet.size(), reinterpret_cast<const sockaddr*>(&destination_), sizeof(destination_));
        std::printf("radio_pcm_end timestamp=%u sent=%d attempt=%u\n", end_timestamp_, sent, ++end_attempts_);
        std::fflush(stdout);
        next_end_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
        if (end_attempts_ >= 3) end_pending_ = false;
    }
    std::shared_ptr<Shared> shared_;
    platform::socket_handle socket_ = platform::kInvalidSocket;
    sockaddr_in destination_{};
    bool ready_ = false;
    uint16_t sequence_ = 0;
    uint32_t ssrc_ = 0, timestamp_base_ = 0;
    uint32_t octets_ = 0, end_timestamp_ = 0, end_attempts_ = 0;
    bool end_pending_ = false;
    std::chrono::steady_clock::time_point next_end_{};
};

} // namespace nodus::radio

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::radio::RadioPcmUdp(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
