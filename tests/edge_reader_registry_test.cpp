#include "edge_reader_registry.h"

#include <cassert>
#include <cstdint>
#include <limits>

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
    return 0;
}
