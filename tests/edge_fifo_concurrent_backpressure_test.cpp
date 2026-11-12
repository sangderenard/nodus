#include "edge_tensor_fifo.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>

static void expect(bool value) { if (!value) std::abort(); }

int main() {
    constexpr uint64_t writer = 71;
    constexpr uint64_t reader = 91;
    constexpr uint64_t count = 4096;
    EdgeTensorFifo fifo;
    fifo.configure({1}, 2, 0, sizeof(uint64_t));
    expect(fifo.subscribe(reader, true));
    expect(gp_mem_backend_resize(fifo.get_storage_backend(), 2 * sizeof(uint64_t)));
    expect(!gp_mem_backend_resize(fifo.get_storage_backend(), 3 * sizeof(uint64_t)));

    std::atomic<bool> start{false};
    std::atomic<uint64_t> refused_full{0};
    std::atomic<bool> ordered{true};
    std::thread producer([&] {
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        for (uint64_t value = 1; value <= count;) {
            bool dropped = false;
            if (fifo.push(1, writer, &value, sizeof(value), &dropped)) {
                if (dropped) ordered.store(false, std::memory_order_relaxed);
                ++value;
            } else {
                refused_full.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::yield();
            }
        }
    });
    std::thread consumer([&] {
        start.store(true, std::memory_order_release);
        for (uint64_t expected = 1; expected <= count;) {
            uint64_t value = 0;
            size_t written = 0;
            if (!fifo.pop(reader, &value, sizeof(value), written)) {
                std::this_thread::yield();
                continue;
            }
            if (written != sizeof(value) || value != expected)
                ordered.store(false, std::memory_order_relaxed);
            ++expected;
            if ((expected & 7) == 0) {
                auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(50);
                while (std::chrono::steady_clock::now() < until)
                    std::atomic_signal_fence(std::memory_order_seq_cst);
            }
        }
    });
    producer.join();
    consumer.join();
    expect(ordered.load(std::memory_order_relaxed));
    expect(refused_full.load(std::memory_order_relaxed) > 0);
}
