#include "console_logger.h"
#include <cstdio>
#include <cstdarg>
#include <chrono>
#include <iostream>

namespace {
    std::mutex g_mutex;
    std::vector<std::string> g_buf;
    std::condition_variable g_cv;
    std::thread g_thread;
    std::atomic<bool> g_running{false};
    bool g_thread_started = false;

    void background_thread() {
        std::unique_lock<std::mutex> lk(g_mutex);
        while (g_running.load()) {
            if (g_buf.empty()) g_cv.wait_for(lk, std::chrono::milliseconds(200));
            std::vector<std::string> local;
            local.swap(g_buf);
            lk.unlock();
            for (auto &s : local) {
                fwrite(s.c_str(), 1, s.size(), stdout);
            }
            fflush(stdout);
            lk.lock();
        }
        // final flush
        std::vector<std::string> local;
        local.swap(g_buf);
        lk.unlock();
        for (auto &s : local) fwrite(s.c_str(), 1, s.size(), stdout);
        fflush(stdout);
    }

    void ensure_thread() {
        if (g_thread_started) return;
        g_running.store(true);
        g_thread = std::thread(background_thread);
        g_thread_started = true;
    }
}

namespace console_logger {
    static std::string format_from_va(const char* fmt, va_list ap_in) {
        va_list ap;
        va_copy(ap, ap_in);
        char tmp[4096];
        int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
        va_end(ap);
        std::string s;
        if (n < 0) s = "[format-error]\n";
        else {
            if (n < (int)sizeof(tmp)) s.assign(tmp, static_cast<size_t>(n));
            else {
                int sz = n + 1;
                std::string big;
                big.resize(sz);
                va_list ap2;
                va_copy(ap2, ap_in);
                vsnprintf(&big[0], sz, fmt, ap2);
                va_end(ap2);
                s.swap(big);
            }
        }
        if (s.empty() || s.back() != '\n') s.push_back('\n');
        return s;
    }

    void enqueue_vprintf(const char* fmt, va_list ap) {
        ensure_thread();
        std::string s = format_from_va(fmt, ap);
        {
            std::lock_guard<std::mutex> lk(g_mutex);
            g_buf.push_back(std::move(s));
        }
        g_cv.notify_one();
    }

    void enqueue_printf(const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        enqueue_vprintf(fmt, ap);
        va_end(ap);
    }

    void flush_now() {
        std::vector<std::string> local;
        {
            std::lock_guard<std::mutex> lk(g_mutex);
            local.swap(g_buf);
        }
        for (auto &s : local) fwrite(s.c_str(), 1, s.size(), stdout);
        fflush(stdout);
    }

    void shutdown() {
        if (!g_thread_started) return;
        g_running.store(false);
        g_cv.notify_one();
        if (g_thread.joinable()) g_thread.join();
        g_thread_started = false;
    }
}

// ensure shutdown at process exit
struct ConsoleLoggerTerminator { ~ConsoleLoggerTerminator() { console_logger::shutdown(); } } g_console_logger_terminator;

// Backwards-compatible global wrapper (some TU's reference the symbol without namespace)
extern "C" void enqueue_printf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    console_logger::enqueue_vprintf(fmt, ap);
    va_end(ap);
}
