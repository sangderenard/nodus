#pragma once
// The Radio rows' shared definitions: the Config, the process-wide Shared state (fftfree state library, current frame,
// PCM graph and counters), the NRAD/PCM frame types and the helpers the rows use. Included by each radio_*.cpp row
// source in this directory and by src/headless/radio_rows.cpp (the nodus_headless_radio_* ABI and the registration).
// Shared is process state handed to a row by the registration; it is never a row's saved state.
#include "nodus_headless_abi.h"
#include "tool_api.h"
#include "tool_registry.h"
#include "fftfree_tool.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"

// ---- platform region: sockets and calling convention (the only OS calls in this file) ----
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#define NODUS_RADIO_CALL __cdecl
namespace nodus::radio::platform {
using socket_handle = SOCKET;
constexpr socket_handle kInvalidSocket = INVALID_SOCKET;
using socket_length = int;
inline bool network_start() { WSADATA data{}; return WSAStartup(MAKEWORD(2, 2), &data) == 0; }
inline void network_stop() { WSACleanup(); }
inline void close_socket(socket_handle s) { closesocket(s); }
inline void set_nonblocking(socket_handle s) { u_long nonblocking = 1; ioctlsocket(s, FIONBIO, &nonblocking); }
inline int send_datagram(socket_handle s, const void* data, size_t bytes, const sockaddr* to, int to_bytes) {
    return sendto(s, static_cast<const char*>(data), static_cast<int>(bytes), 0, to, to_bytes);
}
} // namespace nodus::radio::platform
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#define NODUS_RADIO_CALL
namespace nodus::radio::platform {
using socket_handle = int;
constexpr socket_handle kInvalidSocket = -1;
using socket_length = socklen_t;
inline bool network_start() { return true; }
inline void network_stop() {}
inline void close_socket(socket_handle s) { ::close(s); }
inline void set_nonblocking(socket_handle s) {
    const int flags = fcntl(s, F_GETFL, 0);
    if (flags >= 0) fcntl(s, F_SETFL, flags | O_NONBLOCK);
}
inline int send_datagram(socket_handle s, const void* data, size_t bytes, const sockaddr* to, int to_bytes) {
    return static_cast<int>(sendto(s, data, bytes, MSG_NOSIGNAL, to, static_cast<socklen_t>(to_bytes)));
}
} // namespace nodus::radio::platform
#endif
// ---- end platform region ----
#include <SDL3/SDL.h>
#include "exec_row.h"

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


namespace nodus::radio {
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

using scope_create_capture_fn = void* (NODUS_RADIO_CALL *)();
using scope_start_fn = int (NODUS_RADIO_CALL *)(void*);
using scope_read_fn = size_t (NODUS_RADIO_CALL *)(void*, uint64_t*, float*, size_t, uint32_t*, uint64_t*);
using scope_running_fn = int (NODUS_RADIO_CALL *)(void*);
using scope_frame_count_fn = uint64_t (NODUS_RADIO_CALL *)(void*);
using scope_feed_fn = int (NODUS_RADIO_CALL *)(void*, const float*, size_t, uint32_t);
using scope_free_fn = void (NODUS_RADIO_CALL *)(void*);

inline bool radio_state_request(nodus::gpu::TensorStateLibraryTool& library,
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

inline bool radio_register_fft_states(Shared& shared) {
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
inline uint64_t radio_words_tensor(const std::array<uint64_t, N>& words) {
    const uint64_t shape[1] = {N};
    const auto tensor = nodus_tensor_create(NODUS_DTYPE_U64, 1, shape);
    if (!tensor) return 0;
    if (nodus_tensor_write(tensor, 0, words.data(), sizeof(words)) == sizeof(words)) return tensor;
    nodus_tensor_destroy(tensor);
    return 0;
}

} // namespace nodus::radio
