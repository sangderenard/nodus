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

    // Each subscriber has its own cursor. The producer remains blocked until
    // the slowest reader advances; one reader cannot consume another's data.
    EdgeTensorFifo shared;
    shared.configure({1}, 2, 0, sizeof(uint64_t));
    expect(shared.subscribe(101, true));
    expect(shared.subscribe(102, true));
    uint64_t first = 11, second = 12, third = 13, received = 0;
    size_t written = 0;
    bool dropped = false;
    expect(shared.push(2, writer, &first, sizeof(first), &dropped) && !dropped);
    expect(shared.push(2, writer, &second, sizeof(second), &dropped) && !dropped);
    expect(!shared.push(2, writer, &third, sizeof(third), &dropped));
    expect(shared.pop(101, &received, sizeof(received), written) && received == first);
    expect(shared.pop(101, &received, sizeof(received), written) && received == second);
    expect(!shared.push(2, writer, &third, sizeof(third), &dropped));
    expect(shared.pop(102, &received, sizeof(received), written) && received == first);
    expect(shared.push(2, writer, &third, sizeof(third), &dropped) && !dropped);
    expect(shared.pop(102, &received, sizeof(received), written) && received == second);
    expect(shared.pop(102, &received, sizeof(received), written) && received == third);
    expect(shared.pop(101, &received, sizeof(received), written) && received == third);

    // A top-k edge explicitly trades continuity for bounded lag. Both
    // subscribers remain independent after the slow cursor is advanced.
    EdgeTensorFifo latest;
    latest.configure({1}, 2, 1, sizeof(uint64_t));
    expect(latest.subscribe(201, true));
    expect(latest.subscribe(202, true));
    expect(latest.push(3, writer, &first, sizeof(first), &dropped) && !dropped);
    expect(latest.push(3, writer, &second, sizeof(second), &dropped) && !dropped);
    expect(latest.pop(201, &received, sizeof(received), written) && received == first);
    expect(latest.push(3, writer, &third, sizeof(third), &dropped) && dropped);
    expect(latest.pop(201, &received, sizeof(received), written) && received == third);
    expect(latest.pop(202, &received, sizeof(received), written) && received == third);
}
