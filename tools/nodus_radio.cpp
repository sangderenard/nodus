#include "nodus_headless_abi.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>

namespace {
std::atomic<bool> stopping{false};
void on_signal(int) { stopping.store(true, std::memory_order_relaxed); }

std::wstring path_from_utf8(const char* source) {
    return std::filesystem::u8path(source).wstring();
}
}

int main(int argc, char** argv) {
    if (argc < 5 || argc > 8) {
        std::fprintf(stderr,
            "usage: nodus_radio <audio_scope.dll> <fft_cffi.dll> <destination-host> <udp-port> [seconds=0] [bands=5] [fft_size=1024]\n");
        return 2;
    }
    const int port = std::atoi(argv[4]);
    const int seconds = argc > 5 ? std::atoi(argv[5]) : 0;
    const int bands = argc > 6 ? std::atoi(argv[6]) : 5;
    const int fft_size = argc > 7 ? std::atoi(argv[7]) : 1024;
    if (port < 1 || port > 65535 || seconds < 0 || bands < 1 || bands > 128 ||
        fft_size < 256 || (fft_size & (fft_size - 1))) return 2;
    const std::wstring scope_dll = path_from_utf8(argv[1]);
    const std::wstring fftfree_dll = path_from_utf8(argv[2]);
    if (!nodus_headless_radio_configure(scope_dll.c_str(), fftfree_dll.c_str(), argv[3],
                                       static_cast<uint16_t>(port), static_cast<uint32_t>(fft_size), 4,
                                       static_cast<uint32_t>(bands), 40.0f, 18000.0f)) {
        std::fprintf(stderr, "Radio configuration or row registration failed\n");
        return 1;
    }
    NodusHeadlessGraph* graph = nodus_headless_create();
    if (!graph) return 1;
    GP_CanvasModuleDesc module{};
    module.x = 0; module.y = 0; module.w = 240; module.h = 120;
    std::strncpy(module.label, "radio", sizeof(module.label) - 1);
    const int table = nodus_headless_add_module(graph, &module);
    const GP_CanvasModuleRowDesc rows[] = {
        {1, 0, 1, 1, "radio.system_output"},
        {1, 0, 1, 1, "radio.udp_broadcast"},
    };
    if (table < 0 || !nodus_headless_set_module_rows(graph, table, rows, 2) ||
        !nodus_headless_set_module_stack_contract(graph, table, 2) ||
        !nodus_headless_set_module_exec_mode(graph, table, 3) ||
        !nodus_headless_step(graph, 1.0 / 60.0)) {
        std::fprintf(stderr, "Radio table construction or Free launch failed\n");
        nodus_headless_destroy(graph);
        return 1;
    }
    std::signal(SIGINT, on_signal);
    std::printf("Radio: one Free table, rows=[system_output+fftfree, udp_broadcast], udp://%s:%d, bands=%d\n",
                argv[3], port, bands);
    std::fflush(stdout);
    const auto started = std::chrono::steady_clock::now();
    while (!stopping.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        NodusRadioStatus status{};
        if (nodus_headless_radio_status(&status)) {
            std::printf("capture=%d rate=%u frames=%llu packets=%llu dropped=%llu\n",
                status.capture_running, status.sample_rate,
                static_cast<unsigned long long>(status.captured_frames),
                static_cast<unsigned long long>(status.emitted_packets),
                static_cast<unsigned long long>(status.dropped_packets));
            std::fflush(stdout);
        }
        if (seconds > 0 && std::chrono::steady_clock::now() - started >= std::chrono::seconds(seconds)) break;
    }
    NodusRadioStatus status{};
    nodus_headless_radio_status(&status);
    const auto stop_begin = std::chrono::steady_clock::now();
    nodus_headless_destroy(graph);
    const auto stop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - stop_begin).count();
    std::printf("shutdown_ms=%lld\n", static_cast<long long>(stop_ms));
    return status.capture_running && status.emitted_packets ? 0 : 1;
}
