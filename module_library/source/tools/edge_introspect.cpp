// edge_introspect: hears the meta edges of a channel group. Meta groups listen to their member edges once per scheduler
// tick (include/table_abi.h "Meta edges listen to their member edges"); this row reads what they heard.
//   execute_stack: pops the channel group (a number), then for each meta group of that channel pushes
//                  <count> <v0> ... <v(count-1)>, the group's float snapshot (gp_table_meta_get_fifo_snapshot: the FIFO
//                  header, then the members block). The contract is the one this row always had.
//   render_table:  paints the groups of the channel last asked for (0 until a stack asks): per member edge its
//                  occupancy, its throughput and its refusals, from the latest observation records.
// Both read through nodus_headless's process-wide readers (include/nodus_meta_channel.h), so the same source works as a
// plugin DLL and inside the image.
#include "tool_api.h"
#include "table_abi.h"
#include "nodus_meta_channel.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

class EdgeIntrospectTool : public ITool {
public:
    EdgeIntrospectTool() {}
    ~EdgeIntrospectTool() noexcept override {}

    const char* id_cstr() const noexcept override { return "edge_introspect"; }
    const char* name_cstr() const noexcept override { return "Edge Introspect"; }
    ToolCaps caps() const override { return ToolCaps::Table | ToolCaps::Text; }

    void initialize(const ToolInitContext& ctx) override { (void)ctx; }
    void shutdown() override {}
    void tick(double dt, HostAPI& host) override { (void)dt; (void)host; }
    void render(RenderContext& ctx) override { (void)ctx; }
    void render_text(TextRenderArgs& args) override { (void)args; }

    void render_table(TableRenderArgs& args) override {
        if (!args.frame_buffer || args.cell_w < 1.0f || args.cell_h < 1.0f) return;
        const int32_t channel = channel_;
        std::vector<uint64_t> words(static_cast<size_t>(GP_META_RECORD_WORDS) * kMaxMembers);
        const int32_t got = nodus::meta::channel_records(channel, words.data(), static_cast<int32_t>(words.size()));
        const int32_t members = got / GP_META_RECORD_WORDS;

        std::vector<GP_TableRow> rows(static_cast<size_t>(members) + 2);
        std::memset(rows.data(), 0, rows.size() * sizeof(GP_TableRow));
        auto text = [](GP_TableCell& c, const char* s) { c.kind = GP_TABLE_CELL_TEXT; std::snprintf(c.text, sizeof(c.text), "%s", s); };
        {
            GP_TableRow& h = rows[0]; h.kind = GP_TABLE_ROW_HEADER; h.expanded = 1; h.cell_count = 4;
            char b[96];
            std::snprintf(b, sizeof(b), "Edge Introspect  channel %d", channel);
            text(h.cells[0], b);
            std::snprintf(b, sizeof(b), "%d member edge(s)", members);
            text(h.cells[1], b);
            text(h.cells[2], "throughput");
            text(h.cells[3], "refused");
        }
        if (members == 0) {
            GP_TableRow& r = rows[1]; r.kind = GP_TABLE_ROW_NOTE; r.expanded = 1; r.cell_count = 1;
            text(r.cells[0], "no meta group listens on this channel yet");
        }
        for (int32_t i = 0; i < members; ++i) {
            const uint64_t* w = &words[static_cast<size_t>(i) * GP_META_RECORD_WORDS];
            GP_TableRow& r = rows[static_cast<size_t>(i) + 1]; r.kind = GP_TABLE_ROW_DEVICE; r.expanded = 1; r.cell_count = 4;
            char b[96];
            if (w[GP_META_REC_EDGE_IDX] == ~0ull) std::snprintf(b, sizeof(b), "edge gone (rope %llu)", static_cast<unsigned long long>(w[GP_META_REC_ROPE_ID]));
            else std::snprintf(b, sizeof(b), "edge %llu", static_cast<unsigned long long>(w[GP_META_REC_EDGE_IDX]));
            text(r.cells[0], b);
            std::snprintf(b, sizeof(b), "%llu/%llu%s", static_cast<unsigned long long>(w[GP_META_REC_OCCUPANCY]),
                          static_cast<unsigned long long>(w[GP_META_REC_SLOTS]), (w[GP_META_REC_FLAGS] & GP_META_FLAG_FULL) ? " FULL" : "");
            text(r.cells[1], b);
            const double seconds = static_cast<double>(w[GP_META_REC_DT_NS]) * 1e-9;
            const double per_s = seconds > 0.0 ? static_cast<double>(w[GP_META_REC_BYTES_DELTA]) / seconds : 0.0;
            std::snprintf(b, sizeof(b), "%llu wr (+%llu)  %.0f B/s", static_cast<unsigned long long>(w[GP_META_REC_WRITE_SEQ]),
                          static_cast<unsigned long long>(w[GP_META_REC_WRITE_DELTA]), per_s);
            text(r.cells[2], b);
            std::snprintf(b, sizeof(b), "%llu (+%llu)", static_cast<unsigned long long>(w[GP_META_REC_REFUSED_TOTAL]),
                          static_cast<unsigned long long>(w[GP_META_REC_REFUSED_DELTA]));
            text(r.cells[3], b);
        }

        const int32_t width = std::max<int32_t>(1, static_cast<int32_t>(args.cell_w));
        const int32_t height = std::max<int32_t>(1, static_cast<int32_t>(args.cell_h));
        std::vector<GP_TableColumn> cols(4);
        for (auto& c : cols) { c.kind = GP_TABLE_CELL_TEXT; c.width_px = std::max<int32_t>(4, width / 4); c.align = 0; }
        GP_TableStyle style{};
        style.width_px = width;
        style.row_h_px = std::max<int32_t>(8, std::min<int32_t>(height / static_cast<int32_t>(rows.size()), 14));
        style.name_w_px = 0;
        const uint8_t bg[4] = {18, 22, 28, 255}, hdr[4] = {40, 54, 74, 255}, ink[4] = {222, 228, 236, 255}, hink[4] = {255, 255, 255, 255};
        std::memcpy(style.bg_rgba, bg, 4); std::memcpy(style.bg_sel_rgba, hdr, 4); std::memcpy(style.hdr_rgba, hdr, 4);
        std::memcpy(style.text_rgba, ink, 4); std::memcpy(style.text_hdr_rgba, hink, 4);
        GP_TableGeom geom{};
        if (!gp_table_calc_size(&style, static_cast<int32_t>(rows.size()), 4, &geom) || geom.width_px < 1 || geom.height_px < 1) return;
        std::vector<uint8_t> pixels(static_cast<size_t>(geom.width_px) * static_cast<size_t>(geom.height_px) * 4);
        if (!gp_table_raster_rgba(rows.data(), static_cast<int32_t>(rows.size()), cols.data(), 4, &style, pixels.data(),
                                  static_cast<int32_t>(pixels.size()), &geom)) return;
        auto* out = static_cast<uint8_t*>(args.frame_buffer);
        const int32_t copy_w = std::min(geom.width_px, width), copy_h = std::min(geom.height_px, height);
        for (int32_t y = 0; y < copy_h; ++y)
            std::memcpy(out + static_cast<size_t>(y) * static_cast<size_t>(args.pitch), pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(geom.width_px) * 4,
                        static_cast<size_t>(copy_w) * 4);
    }

    void execute_stack(ToolStackContext& ctx) override {
        // Stack form: pop the desired channel group (float -> int). For each meta group whose channel group equals it,
        // push the group's float snapshot as: <count> <v0> <v1> ... where <count> is the number of floats.
        const float chf = tool_stack_pop(ctx.stack);
        const int channel = static_cast<int>(chf);
        channel_ = channel;
        for (int32_t ordinal = 0;; ++ordinal) {
            float buf[512];
            const int32_t wrote = nodus::meta::channel_floats(channel, ordinal, buf, static_cast<int32_t>(sizeof(buf) / sizeof(buf[0])));
            if (wrote <= 0) break;
            tool_stack_push(ctx.stack, static_cast<float>(wrote));
            for (int i = 0; i < wrote; ++i) tool_stack_push(ctx.stack, buf[i]);
        }
    }

private:
    static constexpr size_t kMaxMembers = 64;
    int32_t channel_ = 0;   // the channel the last stack asked for; what the picture shows
};

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new EdgeIntrospectTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
