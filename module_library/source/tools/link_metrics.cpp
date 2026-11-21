// link_metrics: the row "nodus.link.metrics" (and, per registered link, "<link id>.metrics"): it sits on a link's edge and
// publishes that link's metrics as a matrix. Development case 2115fd107eba439fb0b5e65d6e6809e9 (Total participant map),
// stage A1; the link is research case 98d49cb8520e44d092c5d134c37c3c32 (include/nodus_link_rows.h).
//
// The row only READS: link_endpoint_report(id) (in the image) or nodus_headless_link_report (as a plugin DLL), and, when
// it has a channel group, the observations of the meta edges of that channel (include/nodus_meta_channel.h). It never
// sends on, closes or alters a link or an edge. Each execute_stack it appends one sample to a ring of N rows.
//
// Row layout (columns of the published float64 matrix [N, W]):
//   0       link state: 0 waiting, 1 linked, 2 closed, 3 failed
//   1..16   the counters of nodus_headless_link_report, in its order: listen_port, account, control_sent,
//           control_received, bulk_bytes_sent, bulk_bytes_received, bulk_tensors_sent, bulk_tensors_received,
//           control_refused, bulk_refused, bulk_dropped, peers_refused, control_read_stalls, bulk_read_stalls,
//           protocol_violations, refused_full
//   17      monotonic seconds (steady clock) when the sample was taken
// and only when the saved channel group is not 0 (W = 26; with channel group 0, W = 18), the meta-edge observations of
// that channel, summed over every member edge of every meta group on it, as of the scheduler tick before the sample
// (GP_MetaRecordWord in include/table_abi.h):
//   18      meta_members          member edges heard (0 until a group on the channel has been fed once)
//   19      meta_write_seq        sum of the members' FIFO write sequences
//   20      meta_write_delta      writes since the previous tick, summed
//   21      meta_read_delta       slowest-reader advance since the previous tick, summed
//   22      meta_occupancy        slots in use, summed
//   23      meta_slots            capacity in slots, summed
//   24      meta_refused_delta    writes refused (full) since the previous tick, summed
//   25      meta_bytes_delta      bytes written since the previous tick, summed
// Row port "metrics" (8-byte AbstractTensor handle tokens, as the link rows' "received") carries the matrix, oldest
// first: the newest sample is the last row and rows not yet filled are zero rows at the top. One tensor per execute,
// only while the port's edge can take it. The stack is neither read nor written.
// Settings are saved state ("NLM1" version 2: link id, window N (default 64), channel group (default 0 = none); version
// 1, without the channel, still loads); render_table paints the link id, the state name and, per counter, the latest
// value and its rate over the window.
#include "nodus_link_rows.h"
#include "table_abi.h"
#include "nodus_headless_abi.h"
#include "nodus_meta_channel.h"

#include "common/tensors/abstraction/tensor_abi.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace nodus::link {

class LinkMetricsRow final : public ITool {
public:
    static constexpr int32_t kCounters = 16, kLinkColumns = 18, kTimeColumn = 17;   // state + 16 counters + monotonic seconds
    static constexpr int32_t kMetaColumns = 8, kMaxMetaMembers = 64;
    static constexpr uint32_t kDefaultWindow = 64, kMaximumWindow = 4096;
    LinkMetricsRow(std::string row_id, std::string link_id, uint32_t window = kDefaultWindow, int32_t channel = 0) : row_id_(std::move(row_id)) {
        configure(std::move(link_id), window, channel);
    }
    LinkMetricsRow() : row_id_("nodus.link.metrics") { configure(std::string(), kDefaultWindow, 0); }
    void configure(std::string link_id, uint32_t window, int32_t channel = 0) {
        std::lock_guard<std::mutex> g(mutex_);
        link_id_ = std::move(link_id);
        window_ = std::clamp<uint32_t>(window, 1, kMaximumWindow);
        channel_ = std::max<int32_t>(0, channel);
        columns_ = kLinkColumns + (channel_ > 0 ? kMetaColumns : 0);
        ring_.assign(size_t(window_) * size_t(columns_), 0.0); filled_ = 0; next_ = 0;
    }
    int32_t columns() const { std::lock_guard<std::mutex> g(mutex_); return columns_; }

    // Saved state: magic "NLM1", version 2, link id length, window, channel group; then the link id (version 1: no channel).
    bool serialize(OutputArchive& out) const override {
        const uint32_t head[5] = {kMagic, kVersion, static_cast<uint32_t>(link_id_.size()), window_, static_cast<uint32_t>(channel_)};
        return out.write(head, sizeof(head)) && out.write(link_id_.data(), link_id_.size());
    }
    bool deserialize(InputArchive& in) override {
        uint32_t head[4] = {};
        if (!in.read(head, sizeof(head)) || head[0] != kMagic || (head[1] != 1 && head[1] != kVersion) || head[2] > 4096 || head[3] < 1 || head[3] > kMaximumWindow) return false;
        uint32_t channel = 0;
        if (head[1] >= 2 && (!in.read(&channel, sizeof(channel)) || channel > 0x7FFFFFFFu)) return false;
        std::string id; id.resize(head[2]);
        if (!in.read(id.data(), id.size())) return false;
        configure(std::move(id), head[3], static_cast<int32_t>(channel));
        return true;
    }
    const char* id_cstr() const noexcept override { return row_id_.c_str(); }
    const char* name_cstr() const noexcept override { return "Link metrics"; }
    ToolCaps caps() const override { return ToolCaps::Table; }
    void initialize(const ToolInitContext& ctx) override {
        if (ctx.serialized_path && ctx.serialized_path[0] != '\0') {
            FileInputArchive in(ctx.serialized_path);
            if (in.ok()) deserialize(in);
        }
    }
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    int32_t row_port_count() const override { return 1; }
    ToolRowPort row_port(int32_t index) const override {
        if (index == 0) return ToolRowPort{ToolPortKind::Return, 1.0f, 0.5f, "metrics", 8};
        return ToolRowPort{};
    }
    int32_t row_height_px() const override { return 18 * 12; }
    // It pops and pushes nothing; the tensors it mints leave on its row port and belong to whoever consumes them.
    ToolTensorRole tensor_role() const override { return ToolTensorRole::Borrower; }

    void execute_stack(ToolStackContext& context) override {
        context.succeeded = false;
        if (!sample()) return;
        context.succeeded = true;
        const ToolRowPortAccess* ports = context.row_ports;
        if (!ports || !ports->publish || !ports->can_publish || !ports->can_publish(ports->host, 0, 1)) return;
        const uint64_t t = matrix_tensor();
        if (!t) return;
        if (!ports->publish(ports->host, 0, &t, sizeof(t))) nodus_tensor_destroy(t);
    }

    void render_table(TableRenderArgs& args) override {
        if (!args.frame_buffer || args.cell_w < 1.0f || args.cell_h < 1.0f) return;
        static const char* const names[kCounters] = {"listen_port", "account", "control_sent", "control_received", "bulk_bytes_sent",
            "bulk_bytes_received", "bulk_tensors_sent", "bulk_tensors_received", "control_refused", "bulk_refused", "bulk_dropped",
            "peers_refused", "control_read_stalls", "bulk_read_stalls", "protocol_violations", "refused_full"};
        std::vector<double> latest, oldest;
        uint32_t have = 0, window = 0; std::string link; int32_t channel = 0;
        {
            std::lock_guard<std::mutex> g(mutex_);
            link = link_id_; have = filled_; window = window_; channel = channel_;
            if (have) { latest = row_at(have - 1); oldest = row_at(0); }
        }
        if (!have) { latest.assign(size_t(kLinkColumns + kMetaColumns), 0.0); oldest = latest; }
        const double span = have > 1 ? latest[kTimeColumn] - oldest[kTimeColumn] : 0.0;
        const int32_t line_count = 1 + kCounters + (channel > 0 ? 1 : 0);
        std::vector<GP_TableRow> rows(static_cast<size_t>(line_count));
        std::memset(rows.data(), 0, rows.size() * sizeof(GP_TableRow));
        auto text = [](GP_TableCell& c, const char* s) { c.kind = GP_TABLE_CELL_TEXT; std::snprintf(c.text, sizeof(c.text), "%s", s); };
        {
            GP_TableRow& h = rows[0]; h.kind = GP_TABLE_ROW_HEADER; h.expanded = 1; h.cell_count = 3;
            char b[96];
            std::snprintf(b, sizeof(b), "link %s", link.empty() ? "(none)" : link.c_str());
            text(h.cells[0], b);
            text(h.cells[1], have ? state_name(int32_t(latest[0])) : "no sample");
            std::snprintf(b, sizeof(b), "window %u/%u", have, window);
            text(h.cells[2], b);
        }
        for (int32_t i = 0; i < kCounters; ++i) {
            GP_TableRow& r = rows[size_t(i) + 1]; r.kind = GP_TABLE_ROW_DEVICE; r.expanded = 1; r.cell_count = 3;
            char b[96];
            text(r.cells[0], names[i]);
            std::snprintf(b, sizeof(b), "%.0f", have ? latest[size_t(i) + 1] : 0.0);
            text(r.cells[1], b);
            std::snprintf(b, sizeof(b), "%.3f/s", span > 0.0 ? (latest[size_t(i) + 1] - oldest[size_t(i) + 1]) / span : 0.0);
            text(r.cells[2], b);
        }
        if (channel > 0) {
            GP_TableRow& r = rows[size_t(kCounters) + 1]; r.kind = GP_TABLE_ROW_DEVICE; r.expanded = 1; r.cell_count = 3;
            char b[96];
            std::snprintf(b, sizeof(b), "meta channel %d", channel);
            text(r.cells[0], b);
            const bool heard = have && latest.size() >= size_t(kLinkColumns + kMetaColumns);
            std::snprintf(b, sizeof(b), "%.0f edge(s) occ %.0f/%.0f", heard ? latest[18] : 0.0, heard ? latest[22] : 0.0, heard ? latest[23] : 0.0);
            text(r.cells[1], b);
            std::snprintf(b, sizeof(b), "%.0f wr +%.0f ref +%.0f", heard ? latest[19] : 0.0, heard ? latest[20] : 0.0, heard ? latest[24] : 0.0);
            text(r.cells[2], b);
        }
        const int32_t width = std::max<int32_t>(1, int32_t(args.cell_w)), height = std::max<int32_t>(1, int32_t(args.cell_h));
        std::vector<GP_TableColumn> cols(3);
        for (auto& c : cols) { c.kind = GP_TABLE_CELL_TEXT; c.width_px = std::max<int32_t>(4, width / 3); c.align = 0; }
        GP_TableStyle style{};
        style.width_px = width; style.row_h_px = std::max<int32_t>(8, height / int32_t(rows.size())); style.name_w_px = 0;
        const uint8_t bg[4] = {18, 22, 28, 255}, hdr[4] = {40, 54, 74, 255}, ink[4] = {222, 228, 236, 255}, hink[4] = {255, 255, 255, 255};
        std::memcpy(style.bg_rgba, bg, 4); std::memcpy(style.bg_sel_rgba, hdr, 4); std::memcpy(style.hdr_rgba, hdr, 4);
        std::memcpy(style.text_rgba, ink, 4); std::memcpy(style.text_hdr_rgba, hink, 4);
        GP_TableGeom geom{};
        if (!gp_table_calc_size(&style, int32_t(rows.size()), 3, &geom) || geom.width_px < 1 || geom.height_px < 1) return;
        std::vector<uint8_t> pixels(size_t(geom.width_px) * size_t(geom.height_px) * 4);
        if (!gp_table_raster_rgba(rows.data(), int32_t(rows.size()), cols.data(), 3, &style, pixels.data(), int32_t(pixels.size()), &geom)) return;
        auto* out = static_cast<uint8_t*>(args.frame_buffer);
        const int32_t copy_w = std::min(geom.width_px, width), copy_h = std::min(geom.height_px, height);
        for (int32_t y = 0; y < copy_h; ++y)
            std::memcpy(out + size_t(y) * size_t(args.pitch), pixels.data() + size_t(y) * size_t(geom.width_px) * 4, size_t(copy_w) * 4);
    }

private:
    static constexpr uint32_t kMagic = 0x314D4C4E, kVersion = 2;   // "NLM1"
    static const char* state_name(int32_t s) {
        switch (s) { case 0: return "waiting"; case 1: return "linked"; case 2: return "closed"; case 3: return "failed"; default: return "unknown"; }
    }
    std::vector<double> row_at(uint32_t oldest_first) const {   // caller holds the mutex; oldest_first < filled_
        const uint32_t slot = (next_ + window_ - filled_ + oldest_first) % window_;
        const double* p = &ring_[size_t(slot) * size_t(columns_)];
        return std::vector<double>(p, p + columns_);
    }
    // One read of the link's state and counters (and the channel's meta observations), appended to the ring. False: no such link.
    bool sample() {
        std::string id; int32_t channel = 0;
        { std::lock_guard<std::mutex> g(mutex_); id = link_id_; channel = channel_; }
        if (id.empty()) return false;
        int32_t state = 0; uint64_t c[kCounters] = {};
#if defined(NODUS_PLUGIN_COMPOSITE)
        EndpointReport report;
        if (!link_endpoint_report(id, &report)) return false;
        state = static_cast<int32_t>(report.state); link_report_counters(report, c);
#else
        if (!nodus_headless_link_report(id.c_str(), &state, c)) return false;
#endif
        double meta[kMetaColumns] = {};
        if (channel > 0) {
            std::vector<uint64_t> words(size_t(GP_META_RECORD_WORDS) * kMaxMetaMembers);
            const int32_t n = nodus::meta::channel_records(channel, words.data(), int32_t(words.size())) / GP_META_RECORD_WORDS;
            meta[0] = double(n);
            for (int32_t m = 0; m < n; ++m) {
                const uint64_t* w = &words[size_t(m) * GP_META_RECORD_WORDS];
                meta[1] += double(w[GP_META_REC_WRITE_SEQ]);
                meta[2] += double(w[GP_META_REC_WRITE_DELTA]);
                meta[3] += double(w[GP_META_REC_READ_DELTA]);
                meta[4] += double(w[GP_META_REC_OCCUPANCY]);
                meta[5] += double(w[GP_META_REC_SLOTS]);
                meta[6] += double(w[GP_META_REC_REFUSED_DELTA]);
                meta[7] += double(w[GP_META_REC_BYTES_DELTA]);
            }
        }
        const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::lock_guard<std::mutex> g(mutex_);
        if (channel != channel_) return false;   // reconfigured while reading: this sample belongs to the old shape
        double* row = &ring_[size_t(next_) * size_t(columns_)];
        row[0] = double(state);
        for (int32_t i = 0; i < kCounters; ++i) row[1 + i] = double(c[i]);
        row[kTimeColumn] = now;
        if (channel_ > 0) for (int32_t i = 0; i < kMetaColumns; ++i) row[kLinkColumns + i] = meta[i];
        next_ = (next_ + 1) % window_;
        if (filled_ < window_) ++filled_;
        return true;
    }
    uint64_t matrix_tensor() {
        std::vector<double> m;
        uint32_t n = 0; int32_t cols = 0;
        {
            std::lock_guard<std::mutex> g(mutex_);
            n = window_; cols = columns_; m.assign(size_t(n) * size_t(cols), 0.0);
            for (uint32_t i = 0; i < filled_; ++i) {   // oldest first, ending on the last row
                const auto r = row_at(i);
                std::memcpy(&m[size_t(n - filled_ + i) * size_t(cols)], r.data(), size_t(cols) * sizeof(double));
            }
        }
        const uint64_t shape[2] = {n, uint64_t(cols)};
        const uint64_t t = nodus_tensor_create(NODUS_DTYPE_F64, 2, shape);
        if (!t) return 0;
        if (nodus_tensor_write(t, 0, m.data(), m.size() * sizeof(double)) != int64_t(m.size() * sizeof(double))) { nodus_tensor_destroy(t); return 0; }
        return t;
    }

    std::string row_id_;
    mutable std::mutex mutex_;
    std::string link_id_;
    uint32_t window_ = kDefaultWindow, filled_ = 0, next_ = 0;
    int32_t channel_ = 0, columns_ = kLinkColumns;
    std::vector<double> ring_;
};

}  // namespace nodus::link

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::link::LinkMetricsRow(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
