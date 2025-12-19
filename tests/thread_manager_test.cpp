#include <cassert>
#include <cstdint>
#include "thread_manager.h"

int main() {
    ThreadManager mgr;
    mgr.start();
    uint64_t edge = 0xDEADBEEFull;
    int slot1 = mgr.register_reader_for_edge(edge);
    assert(slot1 != 0);
    mgr.update_reader_seq(slot1, 10);
    uint64_t min = mgr.min_reader_seq_for_edge(edge);
    // Ensure manager reports a known min for the edge
    assert(min != UINT64_MAX);
    mgr.unregister_reader_slot(slot1);
    mgr.stop();
    return 0;
}
