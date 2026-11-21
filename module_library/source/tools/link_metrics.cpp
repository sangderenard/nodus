// link_metrics: the row "nodus.link.metrics" (and, per registered link, "<link id>.metrics"): it sits on a link's edge and
// publishes that link's metrics as a matrix. Development case 2115fd107eba439fb0b5e65d6e6809e9 (Total participant map),
// stage A1; the link is research case 98d49cb8520e44d092c5d134c37c3c32 (include/nodus_link_rows.h).
//
// The row only READS: link_endpoint_report(id) (in the image) or nodus_headless_link_report (as a plugin DLL). It never
// sends on, closes or alters a link. Each execute_stack it appends one sample [state, the 16 counters, monotonic
// seconds] to a ring of N (the counters in the order of nodus_headless_link_report: listen_port, account, control_sent,
// control_received, bulk_bytes_sent, bulk_bytes_received, bulk_tensors_sent, bulk_tensors_received, control_refused,
// bulk_refused, bulk_dropped, peers_refused, control_read_stalls, bulk_read_stalls, protocol_violations, refused_full).
// Row port "metrics" (8-byte AbstractTensor handle tokens, as the link rows' "received") carries a float64 tensor of
// shape [N, 18], oldest first: the newest sample is the last row and rows not yet filled are zero rows at the top.
// One tensor per execute, only while the port's edge can take it. The stack is neither read nor written.
// Settings are saved state ("NLM1": link id, window N, default 64); render_table paints the link id, the state name
// and, per counter, the latest value and its rate over the window.
#include "nodus_link_rows.h"
#include "table_abi.h"
#include "nodus_headless_abi.h"

#include "common/tensors/abstraction/tensor_abi.h"

#include <algorithm>
#include <array>
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
    static constexpr int32_t kCounters = 16, kColumns = 18;   // state + 16 counters + monotonic seconds
    static constexpr uint32_t kDefaultWindow = 64, kMaximumWindow = 4096;
    LinkMetricsRow(std::string row_id, std::string link_id, uint32_t window = kDefaultWindow) : row_id_(std::move(row_id)) {
        configure(std::move(link_id), window);
    }
    LinkMetricsRow() : row_id_("nodus.link.metrics") { configure(std::string(), kDefaultWindow); }
    void configure(std::string link_id, uint32_t window) {
        std::lock_guard<std::mutex> g(mutex_);
        link_id_ = std::move(link_id);
        window_ = std::clamp<uint32_t>(window, 1, kMaximumWindow);
        ring_.assign(size_t(window_) * kColumns, 0.0); filled_ = 0; next_ = 0;
    }

    // Saved state: magic "NLM1", version 1, link id length, window; then the link id.
    bool serialize(OutputArchive& out) const override {
        const uint32_t head[4] = {kMagic, kVersion, static_cast<uint32_t>(link_id_.size()), window_};
        return out.write(head, sizeof(head)) && out.write(link_id_.data(), link_id_.size());
    }
    bool deserialize(InputArchive& in) override {
        uint32_t head[4] = {};
        if (!in.read(head, sizeof(head)) || head[0] != kMagic || head[1] != kVersion || head[2] > 4096 || head[3] < 1 || head[3] > kMaximumWindow) return false;
        std::string id; id.resize(head[2]);
        if (!in.read(id.data(), id.size())) return false;
        configure(std::move(id), head[3]);
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
        std::array<double, kColumns> latest{}, oldest{};
        uint32_t have = 0, window = 0; std::string link;
        {
            std::lock_guard<std::mutex> g(mutex_);
            link = link_id_; have = filled_; window = window_;
            if (have) { latest = row_at(have - 1); oldest = row_at(0); }
        }
        const double span = have > 1 ? latest[kColumns - 1] - oldest[kColumns - 1] : 0.0;
        std::vector<GP_TableRow> rows(size_t(kCounters) + 1);
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
    static constexpr uint32_t kMagic = 0x314D4C4E, kVersion = 1;   // "NLM1"
    static const char* state_name(int32_t s) {
        switch (s) { case 0: return "waiting"; case 1: return "linked"; case 2: return "closed"; case 3: return "failed"; default: return "unknown"; }
    }
    std::array<double, kColumns> row_at(uint32_t oldest_first) const {   // caller holds the mutex; oldest_first < filled_
        const uint32_t slot = (next_ + window_ - filled_ + oldest_first) % window_;
        std::array<double, kColumns> r{};
        std::memcpy(r.data(), &ring_[size_t(slot) * kColumns], sizeof(r));
        return r;
    }
    // One read of the link's state and counters, appended to the ring. False: no such link.
    bool sample() {
        std::string id;
        { std::lock_guard<std::mutex> g(mutex_); id = link_id_; }
        if (id.empty()) return false;
        int32_t state = 0; uint64_t c[kCounters] = {};
#if defined(NODUS_PLUGIN_COMPOSITE)
        EndpointReport report;
        if (!link_endpoint_report(id, &report)) return false;
        state = static_cast<int32_t>(report.state); link_report_counters(report, c);
#else
        if (!nodus_headless_link_report(id.c_str(), &state, c)) return false;
#endif
        const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::lock_guard<std::mutex> g(mutex_);
        double* row = &ring_[size_t(next_) * kColumns];
        row[0] = double(state);
        for (int32_t i = 0; i < kCounters; ++i) row[1 + i] = double(c[i]);
        row[kColumns - 1] = now;
        next_ = (next_ + 1) % window_;
        if (filled_ < window_) ++filled_;
        return true;
    }
    uint64_t matrix_tensor() {
        std::vector<double> m;
        uint32_t n = 0;
        {
            std::lock_guard<std::mutex> g(mutex_);
            n = window_; m.assign(size_t(n) * kColumns, 0.0);
            for (uint32_t i = 0; i < filled_; ++i) {   // oldest first, ending on the last row
                const auto r = row_at(i);
                std::memcpy(&m[size_t(n - filled_ + i) * kColumns], r.data(), sizeof(r));
            }
        }
        const uint64_t shape[2] = {n, uint64_t(kColumns)};
        const uint64_t t = nodus_tensor_create(NODUS_DTYPE_F64, 2, shape);
        if (!t) return 0;
        if (nodus_tensor_write(t, 0, m.data(), m.size() * sizeof(double)) != int64_t(m.size() * sizeof(double))) { nodus_tensor_destroy(t); return 0; }
        return t;
    }

    std::string row_id_;
    mutable std::mutex mutex_;
    std::string link_id_;
    uint32_t window_ = kDefaultWindow, filled_ = 0, next_ = 0;
    std::vector<double> ring_;
};

}  // namespace nodus::link

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::link::LinkMetricsRow(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
