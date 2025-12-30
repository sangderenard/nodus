#include "meta_edge_snapshot_wrapper.h"
#include "table_abi.h"
#include <cstdio>
#include <cstdlib>

int main() {
    GP_TableContext* t = gp_table_create(nullptr);
    if (!t) return 2;

    // Add a single edge and snapshot it. We don't need to populate payload;
    // this test verifies capture/restore plumbing only and that APIs succeed.
    uint64_t a = 1ull, b = 2ull;
    if (!gp_table_add_edge(t, a, b)) return 3;
    int edge_count = gp_table_get_edge_count(t);
    if (edge_count <= 0) return 4;
    int edge_idx = edge_count - 1;

    void* buf = nullptr;
    size_t buf_len = 0, wrote = 0;
    int32_t cap = gp_meta_edge_snapshot_capture(t, edge_idx, &buf, &buf_len, &wrote);
    if (!cap || !buf || wrote == 0) {
        if (buf) gp_meta_edge_snapshot_free(buf);
        return 5;
    }

    int32_t res = gp_meta_edge_snapshot_restore(t, edge_idx, buf, wrote);
    gp_meta_edge_snapshot_free(buf);
    return res ? 0 : 6;
}
