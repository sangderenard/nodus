#include "edge_reader_registry.h"

#include <cassert>
#include <atomic>
#include <cstdint>
#include <limits>
#include <thread>

int main() {
    EdgeReaderRegistry registry;
    const auto edge_a_reader_1 = registry.register_reader(11);
    const auto edge_a_reader_2 = registry.register_reader(11);
    const auto edge_b_reader = registry.register_reader(12);

    assert(edge_a_reader_1 != EdgeReaderRegistry::kInvalidSlot);
    assert(edge_a_reader_2 != EdgeReaderRegistry::kInvalidSlot);
    assert(edge_b_reader != EdgeReaderRegistry::kInvalidSlot);
    assert(registry.reader_count(11) == 2);
    assert(registry.min_reader_seq(11) == 0);

    assert(registry.update_reader(edge_a_reader_1, 9));
    assert(registry.update_reader(edge_a_reader_2, 4));
    assert(registry.update_reader(edge_b_reader, 100));
    assert(registry.min_reader_seq(11) == 4);
    assert(registry.min_reader_seq(12) == 100);

    // Removing the slow reader must expose the surviving reader's exact
    // frontier, rather than preserving the old aggregate minimum.
    registry.unregister_reader(edge_a_reader_2);
    assert(registry.min_reader_seq(11) == 9);
    assert(registry.reader_count(11) == 1);

    registry.unregister_reader(edge_a_reader_1);
    assert(registry.min_reader_seq(11) == std::numeric_limits<uint64_t>::max());
    assert(!registry.update_reader(edge_a_reader_1, 10));

    const auto live = registry.register_reader(13);
    std::atomic<bool> done{false};
    std::thread publisher([&] {
        for (uint64_t seq = 1; seq <= 100000; ++seq)
            assert(registry.update_reader(live, seq));
        done.store(true, std::memory_order_release);
    });
    uint64_t last = 0;
    while (!done.load(std::memory_order_acquire)) {
        const uint64_t current = registry.min_reader_seq(13);
        assert(current >= last && current <= 100000);
        last = current;
    }
    publisher.join();
    assert(registry.min_reader_seq(13) == 100000);
    registry.unregister_reader(live);
    return 0;
}
