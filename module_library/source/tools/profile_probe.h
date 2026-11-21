#pragma once
// Shared by the job timing profile rows (profile_row.cpp, profile_math_source.cpp, profile_math_sink.cpp) and the
// profiling program tools/profiling/nodus_job_timing_profile.cpp: the probe a row counts its runs into. The Menu
// ignores this header.
#include "graph_runtime.h"
#include "table_abi.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>

using Clock = std::chrono::steady_clock;

struct Probe {
    std::atomic<uint64_t> runs{0};
    std::atomic<bool> ordered{true};
};
