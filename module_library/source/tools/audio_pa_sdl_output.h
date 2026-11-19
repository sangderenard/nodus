#pragma once
// The PCM SDL output row's shared declarations: the PCM block, the port and tool names, and the State a headless
// process shares with its one row. The row's source is audio_pa_sdl_output.cpp (the Menu ignores this header).
#include "nodus_headless_abi.h"

#include <array>
#include <atomic>
#include <cstdint>

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
}
