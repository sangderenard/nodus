// Simple deferred console logger with thread-safe cache and periodic flush
#pragma once
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <condition_variable>
#include <atomic>

namespace console_logger {
    void enqueue_printf(const char* fmt, ...);
    void enqueue_vprintf(const char* fmt, va_list ap);
    void flush_now();
    void shutdown();
}

// Backwards-compatible global symbol (some translation units reference this)
#ifdef __cplusplus
extern "C" {
#endif
void enqueue_printf(const char* fmt, ...);
#ifdef __cplusplus
}
#endif

#define CONSOLE_PRINTF(...) console_logger::enqueue_printf(__VA_ARGS__)
