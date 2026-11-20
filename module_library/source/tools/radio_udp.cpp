// radio_udp: the Radio UDP broadcast row (src/headless/radio_rows.cpp registers it).
// The row tool's source, in the module library (case 74fd8157): the Menu lists it, the canvas compiler inlines it, it
// builds as its own plugin DLL, and the image registers it from there. The Radio configuration is not the row's saved
// state: it lives in Shared (radio_shared.h). Shared is process state (the registration captures it in the factory); a row made with no Shared refuses in execute_stack.
#include "radio_shared.h"

#if defined(_WIN32)
namespace nodus::radio {

class RadioUdp final : public ITool {
public:
    explicit RadioUdp(std::shared_ptr<Shared> shared) : shared_(std::move(shared)) {}
    RadioUdp() = default; // no Shared: the row refuses in execute_stack
    ~RadioUdp() noexcept override { shutdown(); }
    const char* id_cstr() const noexcept override { return "radio.udp_broadcast"; }
    const char* name_cstr() const noexcept override { return "Radio UDP broadcast"; }
    void initialize(const ToolInitContext&) override {
        if (!shared_) return;
        if (shared_->config.analysis_only) {
            file_ = CreateFileW(shared_->config.analysis_file.c_str(), GENERIC_WRITE, 0, nullptr,
                                CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            return;
        }
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
        if (file_ != INVALID_HANDLE_VALUE) {
            const bool flushed = FlushFileBuffers(file_) != 0;
            const bool closed = CloseHandle(file_) != 0;
            if (!flushed || !closed)
                shared_->dropped.fetch_add(1, std::memory_order_relaxed);
            file_ = INVALID_HANDLE_VALUE;
        }
        if (socket_ != INVALID_SOCKET) { closesocket(socket_); socket_ = INVALID_SOCKET; }
        if (winsock_ready_) { WSACleanup(); winsock_ready_ = false; }
    }
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        if (!shared_) { ctx.succeeded = false; return; }
        void* frame_ptr = nullptr;
        const ValueTypeId pointer_type = ValueTypeRegistry::global().builtin(VT_VOID_PTR);
        if (!ctx.stack.raw || !raw_stack_pop_typed(*ctx.stack.raw, &frame_ptr, pointer_type)) {
            ctx.succeeded = false;
            return;
        }
        const auto* frame = static_cast<const RadioFrame*>(frame_ptr);
        if (frame && frame->magic == kFrameMagic && frame->terminal) return;
        if (!frame || frame->magic != kFrameMagic || frame->band_count != shared_->config.bands ||
            (shared_->config.analysis_only ? file_ == INVALID_HANDLE_VALUE :
             socket_ == INVALID_SOCKET || !destination_bytes_)) {
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
        bool sent = false;
        if (shared_->config.analysis_only) {
            // 15 minutes at five bands and a 256-frame hop fits below this bound.
            if (offset <= kMaxAnalysisBytes - written_) {
                DWORD count = 0;
                sent = WriteFile(file_, packet.data(), static_cast<DWORD>(offset), &count, nullptr) && count == offset;
                if (sent) written_ += offset;
            }
        } else {
            sent = sendto(socket_, reinterpret_cast<const char*>(packet.data()),
                          static_cast<int>(offset), 0,
                          reinterpret_cast<const sockaddr*>(&destination_), destination_bytes_) == static_cast<int>(offset);
        }
        if (sent) shared_->emitted.fetch_add(1, std::memory_order_relaxed);
        else shared_->dropped.fetch_add(1, std::memory_order_relaxed);
    }
private:
    std::shared_ptr<Shared> shared_;
    bool winsock_ready_ = false;
    static constexpr size_t kMaxAnalysisBytes = 16 * 1024 * 1024;
    size_t written_ = 0;
    HANDLE file_ = INVALID_HANDLE_VALUE;
    SOCKET socket_ = INVALID_SOCKET;
    sockaddr_storage destination_{};
    int destination_bytes_ = 0;
};

} // namespace nodus::radio

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::radio::RadioUdp(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
#endif
