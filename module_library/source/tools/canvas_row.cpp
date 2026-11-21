// canvas_row: a table row that holds a whole canvas (include/canvas_row_tool.h). The row tool's source, in the module
// library: the Menu lists it, the canvas compiler inlines it, and the image registers it (src/canvas_row_tool.cpp).
// docs/MESH_CANVAS_UP_PORTS_2026-11-21.md, stage 3.
#include "canvas_row_tool.h"
#include "nodus_headless_abi.h"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

namespace nodus::canvas_row {

namespace {
// Saved state: magic "NCR1", version 1, canvas byte count (u64), the canvas file's bytes.
constexpr uint32_t kCanvasRowMagic = 0x3152434E, kCanvasRowVersion = 1;
constexpr uint64_t kCanvasRowMaxBytes = uint64_t(1) << 32;
constexpr size_t kTokenBytesMax = 256;

// The canvas file the headless graph reads and writes, in a private scratch name for this one call.
std::filesystem::path scratch_path() {
    static std::atomic<uint64_t> serial{0};
    std::error_code ec;
    auto dir = std::filesystem::temp_directory_path(ec);
    if (ec) dir = std::filesystem::current_path();
    char name[96];
    std::snprintf(name, sizeof(name), "nodus_canvas_row_%p_%llu.nodus.txt", static_cast<void*>(&serial),
                  static_cast<unsigned long long>(serial.fetch_add(1)));
    return dir / name;
}
}  // namespace

CanvasRow::CanvasRow() = default;
CanvasRow::~CanvasRow() { release(); }

void CanvasRow::release() {
    if (canvas_) nodus_headless_destroy(canvas_);
    canvas_ = nullptr;
    ports_.clear();
}

void CanvasRow::initialize(const ToolInitContext& ctx) {
    if (ctx.serialized_path && ctx.serialized_path[0] != '\0') {
        FileInputArchive in(ctx.serialized_path);
        if (in.ok()) deserialize(in);
    }
}

void CanvasRow::shutdown() { release(); }

// The canvas, from its saved bytes: a nested graph of its own, its ports become the row's ports.
bool CanvasRow::adopt(const std::vector<uint8_t>& saved) {
    NodusHeadlessGraph* graph = nodus_headless_create();
    if (!graph) return false;
    const auto path = scratch_path();
    bool loaded = false;
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(saved.data()), static_cast<std::streamsize>(saved.size()));
        loaded = out.good();
    }
    if (loaded) loaded = nodus_headless_load_file(graph, path.string().c_str()) != 0;
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (!loaded) { nodus_headless_destroy(graph); return false; }
    std::vector<Port> ports;
    const int32_t n = nodus_headless_canvas_port_count(graph);
    for (int32_t i = 0; i < n; ++i) {
        char name[128] = {};
        int32_t direction = 0, contact = 0, up = 0;
        if (!nodus_headless_canvas_port_at(graph, i, name, sizeof(name), &direction, &contact, &up)) continue;
        Port p;
        p.name = name;
        p.into_canvas = direction == 1;     // GP_CANVAS_PORT_IN
        ports.push_back(std::move(p));
    }
    if (ports.size() > 64) { nodus_headless_destroy(graph); return false; }   // GP_CANVAS_ROW_PORT_STRIDE
    release();
    canvas_ = graph;
    ports_ = std::move(ports);
    saved_ = saved;
    return true;
}

bool CanvasRow::serialize(OutputArchive& out) const {
    std::vector<uint8_t> bytes = saved_;
    if (canvas_) {
        const auto path = scratch_path();
        if (nodus_headless_save_file(canvas_, path.string().c_str())) {
            std::ifstream in(path, std::ios::binary);
            bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    if (bytes.empty()) return false;        // a row that holds no canvas has nothing to save
    const uint32_t head[2] = {kCanvasRowMagic, kCanvasRowVersion};
    const uint64_t size = bytes.size();
    return out.write(head, sizeof(head)) && out.write(&size, sizeof(size)) && out.write(bytes.data(), bytes.size());
}

bool CanvasRow::deserialize(InputArchive& in) {
    uint32_t head[2] = {};
    uint64_t size = 0;
    if (!in.read(head, sizeof(head)) || head[0] != kCanvasRowMagic || head[1] != kCanvasRowVersion) return false;
    if (!in.read(&size, sizeof(size)) || !size || size > kCanvasRowMaxBytes) return false;
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!in.read(bytes.data(), bytes.size())) return false;
    return adopt(bytes);
}

ToolRowPort CanvasRow::row_port(int32_t index) const {
    if (index < 0 || index >= static_cast<int32_t>(ports_.size())) return ToolRowPort{};
    const auto& p = ports_[static_cast<size_t>(index)];
    return ToolRowPort{p.into_canvas ? ToolPortKind::Argument : ToolPortKind::Return, p.into_canvas ? 0.0f : 1.0f,
                       (static_cast<float>(index) + 0.5f) / static_cast<float>(ports_.size()), p.name.c_str(), 0};
}

void CanvasRow::execute_stack(ToolStackContext& context) {
    context.succeeded = false;
    if (!canvas_) return;
    const ToolRowPortAccess* rp = context.row_ports;
    // Into the canvas: what waits on each Argument port, while the canvas takes it.
    for (size_t i = 0; i < ports_.size(); ++i) {
        Port& p = ports_[i];
        if (!p.into_canvas) continue;
        for (;;) {
            if (p.held.empty()) {
                if (!rp || !rp->consume) break;
                uint8_t token[kTokenBytesMax];
                size_t got = 0;
                if (!rp->consume(rp->host, static_cast<int32_t>(i), token, sizeof(token), &got) || !got) break;
                p.held.assign(token, token + got);
            }
            if (!nodus_headless_canvas_port_publish(canvas_, p.name.c_str(), p.held.data(), p.held.size())) break;
            p.held.clear();
        }
    }
    // One step of the canvas, on its own scheduler; it returns once that tick has run.
    if (!nodus_headless_step(canvas_, 0.0)) return;
    // Out of the canvas: what reached each OUT port, while the row's Return port edge can take it.
    for (size_t i = 0; i < ports_.size(); ++i) {
        Port& p = ports_[i];
        if (p.into_canvas || !rp || !rp->publish || !rp->can_publish) continue;
        while (rp->can_publish(rp->host, static_cast<int32_t>(i), 1)) {
            if (p.held.empty()) {
                uint8_t token[kTokenBytesMax];
                size_t got = 0;
                if (!nodus_headless_canvas_port_consume(canvas_, p.name.c_str(), token, sizeof(token), &got) || !got)
                    break;
                p.held.assign(token, token + got);
            }
            if (!rp->publish(rp->host, static_cast<int32_t>(i), p.held.data(), p.held.size())) break;   // held for next time
            p.held.clear();
        }
    }
    context.succeeded = true;
}

}  // namespace nodus::canvas_row

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::canvas_row::CanvasRow(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
