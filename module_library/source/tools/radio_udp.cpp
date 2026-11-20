// radio_udp: the Radio UDP broadcast row (src/headless/radio_rows.cpp registers it).
// The row tool's source, in the module library (case 74fd8157): the Menu lists it, the canvas compiler inlines it, it
// builds as its own plugin DLL, and the image registers it from there. The Radio configuration is not the row's saved
// state: it lives in Shared (radio_shared.h). Shared is process state (the registration captures it in the factory); a row made with no Shared refuses in execute_stack.
#include "radio_shared.h"
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace nodus::radio {

// ---- platform region: the offline analysis file (a new file, written whole datagrams at a time) ----
namespace platform {
#if defined(_WIN32)
using analysis_file = HANDLE;
static const analysis_file kNoFile = INVALID_HANDLE_VALUE;
inline analysis_file analysis_open(const std::wstring& path) {
    return CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
}
inline bool analysis_write(analysis_file f, const uint8_t* p, size_t n) {
    DWORD count = 0; return WriteFile(f, p, static_cast<DWORD>(n), &count, nullptr) && count == n;
}
inline bool analysis_close(analysis_file f) { const bool flushed = FlushFileBuffers(f) != 0; return CloseHandle(f) != 0 && flushed; }
#else
using analysis_file = int;
static constexpr analysis_file kNoFile = -1;
inline analysis_file analysis_open(const std::wstring& path) {
    std::string utf8; utf8.reserve(path.size());
    for (wchar_t c : path) {   // UTF-32 code points to UTF-8
        const uint32_t u = static_cast<uint32_t>(c);
        if (u < 0x80) utf8 += static_cast<char>(u);
        else if (u < 0x800) { utf8 += static_cast<char>(0xC0 | (u >> 6)); utf8 += static_cast<char>(0x80 | (u & 0x3F)); }
        else if (u < 0x10000) { utf8 += static_cast<char>(0xE0 | (u >> 12)); utf8 += static_cast<char>(0x80 | ((u >> 6) & 0x3F)); utf8 += static_cast<char>(0x80 | (u & 0x3F)); }
        else { utf8 += static_cast<char>(0xF0 | (u >> 18)); utf8 += static_cast<char>(0x80 | ((u >> 12) & 0x3F)); utf8 += static_cast<char>(0x80 | ((u >> 6) & 0x3F)); utf8 += static_cast<char>(0x80 | (u & 0x3F)); }
    }
    return ::open(utf8.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
}
inline bool analysis_write(analysis_file f, const uint8_t* p, size_t n) {
    size_t done = 0;
    while (done < n) { const ssize_t w = ::write(f, p + done, n - done); if (w <= 0) return false; done += static_cast<size_t>(w); }
    return true;
}
inline bool analysis_close(analysis_file f) { const bool flushed = ::fsync(f) == 0; return ::close(f) == 0 && flushed; }
#endif
}  // namespace platform
// ---- end platform region ----

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
            file_ = platform::analysis_open(shared_->config.analysis_file);
            return;
        }
        if (!platform::network_start()) return;
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
        if (socket_ != platform::kInvalidSocket && addresses->ai_addrlen <= sizeof(destination_)) {
            std::memcpy(&destination_, addresses->ai_addr, addresses->ai_addrlen);
            destination_bytes_ = static_cast<int>(addresses->ai_addrlen);
            sockaddr_in local{};
            local.sin_family = AF_INET;
            local.sin_addr.s_addr = htonl(INADDR_ANY);
            if (bind(socket_, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) != 0) {
                platform::close_socket(socket_); socket_ = platform::kInvalidSocket; freeaddrinfo(addresses); return;
            }
            platform::socket_length local_size = sizeof(local);
            if (getsockname(socket_, reinterpret_cast<sockaddr*>(&local), &local_size) == 0) {
                std::printf("radio_udp_source_port=%u\n", static_cast<unsigned>(ntohs(local.sin_port)));
                std::fflush(stdout);
            }
            platform::set_nonblocking(socket_);
        }
        freeaddrinfo(addresses);
    }
    void shutdown() override {
        if (file_ != platform::kNoFile) {
            if (!platform::analysis_close(file_))
                shared_->dropped.fetch_add(1, std::memory_order_relaxed);
            file_ = platform::kNoFile;
        }
        if (socket_ != platform::kInvalidSocket) { platform::close_socket(socket_); socket_ = platform::kInvalidSocket; }
        if (winsock_ready_) { platform::network_stop(); winsock_ready_ = false; }
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
            (shared_->config.analysis_only ? file_ == platform::kNoFile :
             socket_ == platform::kInvalidSocket || !destination_bytes_)) {
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
                sent = platform::analysis_write(file_, packet.data(), offset);
                if (sent) written_ += offset;
            }
        } else {
            sent = platform::send_datagram(socket_, packet.data(), offset,
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
    platform::analysis_file file_ = platform::kNoFile;
    platform::socket_handle socket_ = platform::kInvalidSocket;
    sockaddr_storage destination_{};
    int destination_bytes_ = 0;
};

} // namespace nodus::radio

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::radio::RadioUdp(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
