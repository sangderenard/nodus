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
    if (argc < 5 || argc > 11) {
        std::fprintf(stderr,
            "usage: nodus_radio <audio_scope.dll> <fft_cffi.dll> <destination-host> <udp-port> [seconds=0] [bands=5] [fft_size=1024] [wav-file] [loop=0] [pcm-loopback-port]\n");
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
    if (argc > 8 && !nodus_headless_radio_file_source(path_from_utf8(argv[8]).c_str(), argc > 9 ? std::atoi(argv[9]) : 0)) {
        std::fprintf(stderr, "Radio file-source configuration failed\n"); return 1;
    }
    const int pcm_port = argc > 10 ? std::atoi(argv[10]) : 0;
    if (argc > 10 && (pcm_port < 1 || pcm_port > 65535 || !nodus_headless_radio_pcm_configure(static_cast<uint16_t>(pcm_port)))) return 2;
    NodusHeadlessGraph* graph = nodus_headless_create();
    if (!graph) return 1;
    GP_CanvasModuleDesc module{};
    module.x = 0; module.y = 0; module.w = 240; module.h = 120;
    std::strncpy(module.label, "radio", sizeof(module.label) - 1);
    const int table = nodus_headless_add_module(graph, &module);
    const GP_CanvasModuleRowDesc rows[] = {
        {1, 0, 1, 1, argc > 8 ? "radio.file_output" : "radio.system_output"},
        {1, 0, 1, 1, "radio.udp_broadcast"},
        {1, 0, 1, 1, "radio.pcm_rtp"},
    };
    if (table < 0 || !nodus_headless_set_module_rows(graph, table, rows, pcm_port ? 3 : 2) ||
        !nodus_headless_set_module_stack_contract(graph, table, 2) ||
        !nodus_headless_set_module_exec_mode(graph, table, 3) ||
        (pcm_port && !nodus_headless_radio_pcm_bind(graph, table)) ||
        !nodus_headless_step(graph, 1.0 / 60.0)) {
        std::fprintf(stderr, "Radio table construction or Free launch failed\n");
        nodus_headless_destroy(graph);
        return 1;
    }
    std::signal(SIGINT, on_signal);
    std::printf("Radio: one Free table, source=%s, udp://%s:%d, bands=%d\n",
                argc > 8 ? "wav+fftfree" : "system_output+fftfree", argv[3], port, bands);
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
            if (pcm_port) {
                uint64_t sent = 0, lost = 0;
                nodus_headless_radio_pcm_status(&sent, &lost);
                std::printf("pcm_packets=%llu pcm_dropped=%llu\n", static_cast<unsigned long long>(sent), static_cast<unsigned long long>(lost));
            }
            std::fflush(stdout);
        }
        if (argc > 8 && !status.capture_running) break;
        if (seconds > 0 && std::chrono::steady_clock::now() - started >= std::chrono::seconds(seconds)) break;
    }
    NodusRadioStatus status{};
    nodus_headless_radio_status(&status);
    const auto stop_begin = std::chrono::steady_clock::now();
    nodus_headless_destroy(graph);
    const auto stop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - stop_begin).count();
    std::printf("shutdown_ms=%lld\n", static_cast<long long>(stop_ms));
    return status.emitted_packets ? 0 : 1;
}
