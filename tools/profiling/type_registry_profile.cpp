#include "value_types.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const int threads = std::atoi(argv[1]);
    if (threads < 1 || threads > 32) return 2;
    auto& registry = ValueTypeRegistry::global();
    const ValueTypeId id = registry.builtin(VT_FLOAT64);
    std::vector<uint64_t> counts(threads, 0);
    std::vector<uint64_t> sums(threads, 0);
    std::atomic<bool> start{false}, stop{false};
    std::vector<std::thread> workers;
    for (int i = 0; i < threads; ++i) {
        workers.emplace_back([&, i] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            uint64_t count = 0, sum = 0;
            do {
                for (int batch = 0; batch < 256; ++batch) {
                    const ValueType* type = registry.get(id);
                    sum += type ? type->size : 0;
                }
                count += 256;
            } while (!stop.load(std::memory_order_acquire));
            counts[i] = count;
            sums[i] = sum;
        });
    }
    const auto begin = std::chrono::steady_clock::now();
    start.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    stop.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    const auto end = std::chrono::steady_clock::now();
    uint64_t count = 0, sum = 0;
    for (int i = 0; i < threads; ++i) { count += counts[i]; sum += sums[i]; }
    if (sum != count * sizeof(double)) return 1;
    const double elapsed = std::chrono::duration<double>(end - begin).count();
    std::cout << threads << ',' << elapsed << ',' << count << ',' << count / elapsed << '\n';
}
