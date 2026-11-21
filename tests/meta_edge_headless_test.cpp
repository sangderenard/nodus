// Meta edges listen: a headless canvas with two tables and an edge, a meta group over that edge, the real scheduler
// (nodus_headless_step -> gp_canvas_step -> ThreadManager) stepped a few ticks. Checks that
//   - the group FIFO received one observation record per member edge per tick, with a growing write sequence;
//   - the snapshot lists the member (edge, rope id, channel group) and the member FIFO's refusals are reported;
//   - the listener did not change the member edge (its own sequence is what the publisher and tool rows made it);
//   - save/load keeps the group (id, channel group, member rope) and a loaded canvas is fed again;
//   - the groups can be listed, re-channelled and destroyed;
//   - edge_introspect paints the group's channel and keeps its stack contract.
// Winsock headers must precede anything that may pull in <windows.h>.
#include "nodus_link.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

#include "nodus_headless_abi.h"
#include "table_abi.h"

#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME edge_introspect_create
#define NODUS_PLUGIN_DESTROY_NAME edge_introspect_destroy
#define NODUS_PLUGIN_SOURCE_NAME edge_introspect_source
#include "../module_library/source/tools/edge_introspect.cpp"

#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")
#endif

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; std::printf("FAIL %s:%d  " #cond "  ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

namespace {
constexpr int32_t kToolTensorAllocator = 12;
constexpr int32_t kToolTensorTool = 13;
constexpr int32_t kChannel = 7;

int32_t build_two_table_graph(NodusHeadlessGraph* g) {
    GP_CanvasModuleDesc a{}; a.w = a.h = 64; std::strncpy(a.label, "alloc", sizeof(a.label) - 1);
    GP_CanvasModuleDesc b{}; b.x = 100; b.w = b.h = 64; std::strncpy(b.label, "tool", sizeof(b.label) - 1);
    const int32_t ma = nodus_headless_add_module(g, &a), mb = nodus_headless_add_module(g, &b);
    const int32_t ra = nodus_headless_bind_builtin_tool(g, ma, kToolTensorAllocator);
    const int32_t rb = nodus_headless_bind_builtin_tool(g, mb, kToolTensorTool);
    GP_CanvasEdgeDesc e{}; e.a_module = ma; e.a_contact_idx = ra; e.b_module = mb; e.b_contact_idx = rb;
    return nodus_headless_add_edge(g, &e, 0);
}
}  // namespace

int main() {
    namespace fs = std::filesystem;
    NodusHeadlessGraph* g = nodus_headless_create();
    CHECK(g, "create");
    if (!g) return 1;
    const int32_t edge = build_two_table_graph(g);
    CHECK(edge >= 0, "edge");
    NodusHeadlessFifoConfig cfg{};
    cfg.token_bytes = 16; cfg.slots = 4; cfg.top_k = 0; cfg.type_id = 0x4D455441; cfg.layout = 2; cfg.dtype = 8;
    CHECK(nodus_headless_configure_edge_fifo(g, edge, &cfg), "configure the member edge (4 slots, lossless)");

    // --- make the group; the ABI refuses what is not a canvas edge ---
    const int32_t bad = 99;
    CHECK(nodus_headless_meta_create(g, &bad, 1, kChannel) == 0, "a group over a missing edge");
    CHECK(nodus_headless_meta_count(g) == 0, "no group yet");
    const uint64_t gid = nodus_headless_meta_create(g, &edge, 1, kChannel);
    CHECK(gid != 0 && nodus_headless_meta_count(g) == 1 && nodus_headless_meta_id_at(g, 0) == gid, "group %llu", (unsigned long long)gid);
    int32_t ch = -1;
    CHECK(nodus_headless_meta_get_channel_group(g, gid, &ch) && ch == kChannel, "channel %d", ch);
    CHECK(nodus_headless_meta_member_count(g, gid) == 1, "one member");
    int32_t member_canvas = -2, member_table = -2; uint64_t rope = 0;
    CHECK(nodus_headless_meta_member(g, gid, 0, &member_canvas, &member_table, &rope) && member_canvas == edge && member_table >= 0 && rope != 0,
          "member canvas %d table %d rope %llu", member_canvas, member_table, (unsigned long long)rope);

    // Six tokens into a 4-slot lossless edge: four accepted, two refused (the member FIFO counts its refusals).
    uint8_t token[16] = {1, 2, 3, 4};
    int accepted = 0;
    for (int i = 0; i < 6; ++i) {
        NodusHeadlessPublishResult r{};
        if (nodus_headless_publish_edge_token(g, edge, token, sizeof(token), &r) && r.accepted) ++accepted;
    }
    std::printf("published 6 tokens: %d accepted\n", accepted);

    // --- step the real scheduler ---
    const int kTicks = 5;
    uint64_t prev_write_seq = 0;
    for (int t = 0; t < kTicks; ++t) {
        CHECK(nodus_headless_step(g, 1.0 / 60.0), "step %d", t);
        uint64_t words[GP_META_SNAPSHOT_HEADER_WORDS + GP_META_RECORD_WORDS] = {};
        const int32_t n = nodus_headless_meta_snapshot(g, gid, words, int32_t(sizeof(words) / sizeof(words[0])));
        CHECK(n == GP_META_SNAPSHOT_HEADER_WORDS + GP_META_RECORD_WORDS, "snapshot words %d", n);
        CHECK(words[GP_META_SNAP_RECORDS] == uint64_t(t + 1), "after tick %d the group FIFO holds %llu records", t, (unsigned long long)words[GP_META_SNAP_RECORDS]);
        const uint64_t* rec = &words[GP_META_SNAPSHOT_HEADER_WORDS];
        CHECK(rec[GP_META_REC_ROPE_ID] == rope && rec[GP_META_REC_EDGE_IDX] == uint64_t(member_table), "record names the member");
        CHECK(rec[GP_META_REC_WRITE_SEQ] >= prev_write_seq, "write sequence never goes back");
        prev_write_seq = rec[GP_META_REC_WRITE_SEQ];
        if (t == 0) {
            CHECK(rec[GP_META_REC_SLOTS] == 4, "slots %llu", (unsigned long long)rec[GP_META_REC_SLOTS]);
            CHECK(rec[GP_META_REC_REFUSED_TOTAL] >= uint64_t(6 - accepted), "refusals %llu", (unsigned long long)rec[GP_META_REC_REFUSED_TOTAL]);
            CHECK(rec[GP_META_REC_WRITE_SEQ] >= uint64_t(accepted), "write seq %llu >= accepted %d", (unsigned long long)rec[GP_META_REC_WRITE_SEQ], accepted);
        }
        // The edge's reader takes what is there, then a fresh pair goes in: the sequence the listener reports is the
        // sequence of the edge itself, so it grows by the pair every tick.
        uint8_t sink[16]; size_t got = 0;
        while (nodus_headless_consume_edge_token(g, edge, sink, sizeof(sink), &got) && got) got = 0;
        for (int k = 0; k < 2; ++k) { NodusHeadlessPublishResult r{}; nodus_headless_publish_edge_token(g, edge, token, sizeof(token), &r); }
    }
    {
        uint64_t records[kTicks * GP_META_RECORD_WORDS] = {}, next = 0;
        const int32_t n = nodus_headless_meta_read_records(g, gid, 0, records, kTicks, &next);
        CHECK(n == kTicks && next == uint64_t(kTicks), "read %d records, next %llu", n, (unsigned long long)next);
        bool ticks_rise = true, seq_rises = true;
        for (int i = 1; i < n; ++i) {
            ticks_rise = ticks_rise && records[i * GP_META_RECORD_WORDS + GP_META_REC_TICK] > records[(i - 1) * GP_META_RECORD_WORDS + GP_META_REC_TICK];
            seq_rises = seq_rises && records[i * GP_META_RECORD_WORDS + GP_META_REC_WRITE_SEQ] >= records[(i - 1) * GP_META_RECORD_WORDS + GP_META_REC_WRITE_SEQ];
        }
        CHECK(ticks_rise && seq_rises, "scheduler tick ids rise and the write sequence grows");
        CHECK(records[(n - 1) * GP_META_RECORD_WORDS + GP_META_REC_WRITE_SEQ] > records[GP_META_REC_WRITE_SEQ],
              "write sequence grew over the run (%llu -> %llu)", (unsigned long long)records[GP_META_REC_WRITE_SEQ],
              (unsigned long long)records[(n - 1) * GP_META_RECORD_WORDS + GP_META_REC_WRITE_SEQ]);
        uint64_t again = 0;
        CHECK(nodus_headless_meta_read_records(g, gid, 3, records, kTicks, &again) == 2, "reading from sequence 3 yields the last two");
        std::printf("member record 0: write_seq %llu occupancy %llu/%llu refused %llu flags %llx; record %d: write_seq %llu delta %llu\n",
            (unsigned long long)records[GP_META_REC_WRITE_SEQ], (unsigned long long)records[GP_META_REC_OCCUPANCY], (unsigned long long)records[GP_META_REC_SLOTS],
            (unsigned long long)records[GP_META_REC_REFUSED_TOTAL], (unsigned long long)records[GP_META_REC_FLAGS], n - 1,
            (unsigned long long)records[(n - 1) * GP_META_RECORD_WORDS + GP_META_REC_WRITE_SEQ], (unsigned long long)records[(n - 1) * GP_META_RECORD_WORDS + GP_META_REC_WRITE_DELTA]);
    }

    // The float snapshot (edge_introspect's reader) reports the member too.
    {
        float f[256] = {};
        const int32_t n = nodus_headless_meta_channel_floats(kChannel, 0, f, 256);
        CHECK(n > 25, "float snapshot of %d floats", n);
        CHECK(nodus_headless_meta_channel_floats(kChannel, 1, f, 256) == 0 && nodus_headless_meta_channel_floats(kChannel + 1, 0, f, 256) == 0, "only one group on the channel");
    }

    // --- edge_introspect: stack contract and picture ---
    {
        ITool* tool = edge_introspect_create();
        RawStackFrame raw{}; raw_stack_init_frame(raw, 1 << 16);
        ToolStackContext c{}; c.stack.raw = &raw; c.stack.default_type = ValueTypeRegistry::global().builtin(VT_FLOAT64);
        tool_stack_push(c.stack, float(kChannel));
        tool->execute_stack(c);
        float f[256] = {};
        const int32_t expect = nodus_headless_meta_channel_floats(kChannel, 0, f, 256);
        float top = 0; for (int32_t i = 0; i < expect; ++i) top = tool_stack_pop(c.stack);
        const float count = tool_stack_pop(c.stack);
        CHECK(int32_t(count) == expect && top == f[0], "pushed <count>=%g then %d floats, first %g (want %g)", count, expect, top, f[0]);
        const int w = 480, h = 120;
        std::vector<uint8_t> a(size_t(w) * h * 4, 0), none(size_t(w) * h * 4, 0);
        TableRenderArgs args; args.rows = 1; args.cols = 1; args.cell_w = float(w); args.cell_h = float(h); args.frame_buffer = a.data(); args.pitch = w * 4;
        tool->render_table(args);
        size_t lit = 0; for (size_t i = 0; i < a.size(); i += 4) lit += (a[i] | a[i + 1] | a[i + 2] | a[i + 3]) != 0;
        CHECK(lit > 200, "edge_introspect painted %zu pixels", lit);
        tool_stack_push(c.stack, 99.0f); tool->execute_stack(c);   // an empty channel pushes nothing; the picture follows it
        args.frame_buffer = none.data(); tool->render_table(args);
        CHECK(a != none, "the picture of channel %d differs from that of an empty channel", kChannel);
        edge_introspect_destroy(tool);
        raw_stack_free_mask(raw); if (raw.backend) gp_mem_backend_release(raw.backend);
    }

    // --- save / load keeps the group ---
    const fs::path path = fs::temp_directory_path() / "meta_edge_headless_test.txt";
    std::error_code ec; fs::remove_all(fs::path(path.string() + ".modules"), ec);
    CHECK(nodus_headless_save_file(g, path.string().c_str()), "save");
    nodus_headless_destroy(g);
    g = nodus_headless_create();
    CHECK(g && nodus_headless_load_file(g, path.string().c_str()), "load");
    if (g) {
        CHECK(nodus_headless_meta_count(g) == 1, "loaded group count %d", nodus_headless_meta_count(g));
        const uint64_t lid = nodus_headless_meta_id_at(g, 0);
        int32_t lch = -1;
        CHECK(lid == gid && nodus_headless_meta_get_channel_group(g, lid, &lch) && lch == kChannel, "loaded id %llu channel %d", (unsigned long long)lid, lch);
        int32_t lcanvas = -2, ltable = -2; uint64_t lrope = 0;
        CHECK(nodus_headless_meta_member(g, lid, 0, &lcanvas, &ltable, &lrope) && lrope == rope && lcanvas == edge, "loaded member canvas %d rope %llu (was %llu)", lcanvas, (unsigned long long)lrope, (unsigned long long)rope);
        for (int t = 0; t < 3; ++t) CHECK(nodus_headless_step(g, 1.0 / 60.0), "step after load");
        uint64_t words[GP_META_SNAPSHOT_HEADER_WORDS + GP_META_RECORD_WORDS] = {};
        CHECK(nodus_headless_meta_snapshot(g, lid, words, int32_t(sizeof(words) / sizeof(words[0]))) == GP_META_SNAPSHOT_HEADER_WORDS + GP_META_RECORD_WORDS
              && words[GP_META_SNAP_RECORDS] == 3 && words[GP_META_SNAPSHOT_HEADER_WORDS + GP_META_REC_ROPE_ID] == rope,
              "a loaded group is fed again (%llu records)", (unsigned long long)words[GP_META_SNAP_RECORDS]);

        // --- re-channel and destroy ---
        CHECK(nodus_headless_meta_set_channel_group(g, lid, 9) && uint64_t(0) == [&] { uint64_t w[GP_META_RECORD_WORDS] = {}; return uint64_t(nodus_headless_meta_channel_records(kChannel, w, GP_META_RECORD_WORDS)); }(), "re-channel");
        uint64_t one[GP_META_RECORD_WORDS * 4] = {};
        CHECK(nodus_headless_meta_channel_records(9, one, int32_t(sizeof(one) / sizeof(one[0]))) == GP_META_RECORD_WORDS, "channel 9 now holds the member's record");
        CHECK(nodus_headless_meta_destroy(g, lid) && nodus_headless_meta_count(g) == 0 && !nodus_headless_meta_destroy(g, lid), "destroy");
        CHECK(nodus_headless_step(g, 1.0 / 60.0), "a canvas with no group still steps");
        nodus_headless_destroy(g);
    }
    fs::remove(path, ec); fs::remove_all(fs::path(path.string() + ".modules"), ec);
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
