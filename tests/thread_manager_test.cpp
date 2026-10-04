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

    // Completion barriers track submission order/count, not the maximum
    // caller-supplied tick id. Explicit and out-of-order ids are legal.
    ThreadManager::TickRequest high{};
    high.tick_id = 100;
    ThreadManager::TickRequest low{};
    low.tick_id = 2;
    mgr.submit_tick(std::move(high), false);
    mgr.submit_tick(std::move(low), false);
    assert(mgr.wait_for_submitted_ticks());
    assert(mgr.ticks_submitted() == 2);
    assert(mgr.ticks_completed() == 100);
    mgr.stop();
    return 0;
}
