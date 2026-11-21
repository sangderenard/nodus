// canvas_row: a table row that holds a whole canvas (include/canvas_row_tool.h). The row tool's source, in the module
// library: the Menu lists it, the canvas compiler inlines it, and it builds as its own plugin.
// docs/MESH_CANVAS_UP_PORTS_2026-11-21.md, stage 3.
#include "canvas_row_tool.h"
#include "nodus_headless_abi.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

namespace nodus::canvas_row {

namespace {
// Saved state: magic "NCR1". Version 2: role (u32), publish interval (u32), snapshot byte count (u64), the snapshot
// (CanvasRow::snapshot). Version 1 (read only): canvas byte count (u64) and the canvas file's bytes, a host.
constexpr uint32_t kCanvasRowMagic = 0x3152434E, kCanvasRowVersion = 2;
constexpr uint32_t kSnapshotMagic = 0x3153434E, kSnapshotVersion = 2;
constexpr uint64_t kCanvasRowMaxBytes = uint64_t(1) << 32;
constexpr size_t kTokenBytesMax = 256;
constexpr int32_t kMetaRecordWords = 16;            // GP_META_RECORD_WORDS
constexpr int32_t kMetaRecordsKept = 1024;          // a group FIFO's ring (table_abi_meta_feed.inl)

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

// A save writes the canvas file and, beside it, <file>.modules/ (its tables' serialized state: the container table's
// rope ids, which a meta group's members resolve through). A snapshot carries both; these move the folder in and out.
std::filesystem::path modules_of(const std::filesystem::path& file) { return file.string() + ".modules"; }
void remove_saved(const std::filesystem::path& file) {
    std::error_code ec;
    std::filesystem::remove(file, ec);
    std::filesystem::remove_all(modules_of(file), ec);
}

template <typename T> void put(std::vector<uint8_t>& out, const T& v) {
    const auto* b = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), b, b + sizeof(T));
}
struct Reader {
    const std::vector<uint8_t>& bytes;
    size_t at = 0;
    template <typename T> bool get(T& v) {
        if (sizeof(T) > bytes.size() - at) return false;
        std::memcpy(&v, bytes.data() + at, sizeof(T));
        at += sizeof(T);
        return true;
    }
    bool take(size_t n, const uint8_t*& p) {
        if (n > bytes.size() - at) return false;
        p = bytes.data() + at;
        at += n;
        return true;
    }
};
}  // namespace

CanvasRow::CanvasRow() = default;
CanvasRow::~CanvasRow() { release(); }

void CanvasRow::release() {
    if (canvas_) nodus_headless_destroy(canvas_);
    canvas_ = nullptr;
    ports_.clear();
    if (held_state_) nodus_tensor_destroy(held_state_);
    held_state_ = 0;
}

void CanvasRow::initialize(const ToolInitContext& ctx) {
    if (ctx.serialized_path && ctx.serialized_path[0] != '\0') {
        FileInputArchive in(ctx.serialized_path);
        if (in.ok()) deserialize(in);
    }
}

void CanvasRow::shutdown() { release(); }

// The canvas's snapshot: its file, and every meta group's records still in its ring, oldest first.
std::vector<uint8_t> CanvasRow::snapshot() const {
    if (!canvas_) return held_snapshot_;
    std::vector<uint8_t> canvas;
    const auto path = scratch_path();
    std::vector<std::pair<std::string, std::vector<uint8_t>>> files;
    if (nodus_headless_save_file(canvas_, path.string().c_str())) {
        {
            std::ifstream in(path, std::ios::binary);
            canvas.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        std::error_code ec;
        const auto dir = modules_of(path);
        for (std::filesystem::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            std::ifstream in(it->path(), std::ios::binary);
            files.emplace_back(std::filesystem::relative(it->path(), dir, ec).generic_string(),
                               std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()));
        }
    }
    remove_saved(path);
    if (canvas.empty()) return held_snapshot_;
    std::vector<uint8_t> out;
    put(out, kSnapshotMagic);
    put(out, kSnapshotVersion);
    put(out, static_cast<uint64_t>(canvas.size()));
    out.insert(out.end(), canvas.begin(), canvas.end());
    put(out, static_cast<uint32_t>(files.size()));
    for (const auto& f : files) {
        put(out, static_cast<uint32_t>(f.first.size()));
        out.insert(out.end(), f.first.begin(), f.first.end());
        put(out, static_cast<uint64_t>(f.second.size()));
        out.insert(out.end(), f.second.begin(), f.second.end());
    }
    const int32_t groups = std::max(0, nodus_headless_meta_count(canvas_));
    put(out, static_cast<uint32_t>(groups));
    std::vector<uint64_t> words(static_cast<size_t>(kMetaRecordsKept) * kMetaRecordWords);
    for (int32_t g = 0; g < groups; ++g) {
        const uint64_t id = nodus_headless_meta_id_at(canvas_, g);
        uint64_t head = 0;
        const int32_t n = id ? nodus_headless_meta_read_records(canvas_, id, 0, words.data(), kMetaRecordsKept, &head) : 0;
        const uint32_t kept = static_cast<uint32_t>(std::max(0, n));
        put(out, id);
        put(out, kept);
        const auto* b = reinterpret_cast<const uint8_t*>(words.data());
        out.insert(out.end(), b, b + static_cast<size_t>(kept) * kMetaRecordWords * sizeof(uint64_t));
    }
    return out;
}

// A snapshot becomes the canvas this row holds: a nested graph of its own, its meta groups carrying the snapshot's
// records; a host's canvas ports become the row's ports.
bool CanvasRow::adopt(const std::vector<uint8_t>& snap) {
    Reader r{snap};
    uint32_t magic = 0, version = 0;
    uint64_t canvas_size = 0;
    const uint8_t* canvas_bytes = nullptr;
    if (!r.get(magic) || magic != kSnapshotMagic || !r.get(version) || version != kSnapshotVersion ||
        !r.get(canvas_size) || !canvas_size || !r.take(static_cast<size_t>(canvas_size), canvas_bytes))
        return false;
    const auto path = scratch_path();
    bool loaded = false;
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(canvas_bytes), static_cast<std::streamsize>(canvas_size));
        loaded = out.good();
    }
    uint32_t file_count = 0;
    if (!r.get(file_count)) loaded = false;
    for (uint32_t f = 0; loaded && f < file_count; ++f) {
        uint32_t name_size = 0;
        uint64_t size = 0;
        const uint8_t* name = nullptr;
        const uint8_t* bytes = nullptr;
        if (!r.get(name_size) || name_size > 4096 || !r.take(name_size, name) || !r.get(size) ||
            !r.take(static_cast<size_t>(size), bytes)) { loaded = false; break; }
        const std::filesystem::path rel(std::string(reinterpret_cast<const char*>(name), name_size));
        if (rel.empty() || rel.is_absolute() || rel.lexically_normal().string().rfind("..", 0) == 0) { loaded = false; break; }
        const auto target = modules_of(path) / rel;
        std::error_code ec;
        std::filesystem::create_directories(target.parent_path(), ec);
        std::ofstream out(target, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(size));
        loaded = out.good();
    }
    NodusHeadlessGraph* graph = loaded ? nodus_headless_create() : nullptr;
    if (graph) loaded = nodus_headless_load_file(graph, path.string().c_str()) != 0;
    remove_saved(path);
    if (!graph) return false;
    uint32_t groups = 0;
    if (!loaded || !r.get(groups)) { nodus_headless_destroy(graph); return false; }
    for (uint32_t g = 0; g < groups; ++g) {
        uint64_t id = 0;
        uint32_t n = 0;
        const uint8_t* recs = nullptr;
        if (!r.get(id) || !r.get(n) || !r.take(static_cast<size_t>(n) * kMetaRecordWords * sizeof(uint64_t), recs)) {
            nodus_headless_destroy(graph);
            return false;
        }
        if (n) {
            std::vector<uint64_t> words(static_cast<size_t>(n) * kMetaRecordWords);
            std::memcpy(words.data(), recs, words.size() * sizeof(uint64_t));
            nodus_headless_meta_append_records(graph, id, words.data(), static_cast<int32_t>(n));
        }
    }
    std::vector<Port> ports;
    if (role_ == CanvasRowRole::Host) {
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
        if (ports.size() > 63) { nodus_headless_destroy(graph); return false; }   // GP_CANVAS_ROW_PORT_STRIDE, less "state"
    }
    release();
    canvas_ = graph;
    ports_ = std::move(ports);
    held_snapshot_ = snap;
    return true;
}

bool CanvasRow::serialize(OutputArchive& out) const {
    // Rebuilt from the canvas held, a replica's too: its meta groups hold the records the snapshot carried. A replica
    // that has received nothing yet saves its role alone.
    const std::vector<uint8_t> snap = snapshot();
    if (snap.empty() && role_ == CanvasRowRole::Host) return false;   // a host that holds no canvas has nothing to save
    const uint32_t head[4] = {kCanvasRowMagic, kCanvasRowVersion, static_cast<uint32_t>(role_), publish_every_};
    const uint64_t size = snap.size();
    return out.write(head, sizeof(head)) && out.write(&size, sizeof(size)) && out.write(snap.data(), snap.size());
}

bool CanvasRow::deserialize(InputArchive& in) {
    uint32_t head[2] = {};
    if (!in.read(head, sizeof(head)) || head[0] != kCanvasRowMagic || (head[1] != 1 && head[1] != kCanvasRowVersion)) return false;
    uint32_t role = 0, every = 0;
    if (head[1] == kCanvasRowVersion && (!in.read(&role, sizeof(role)) || !in.read(&every, sizeof(every)) || role > 1)) return false;
    uint64_t size = 0;
    if (!in.read(&size, sizeof(size)) || size > kCanvasRowMaxBytes) return false;
    if (!size) {                            // a replica that has received nothing yet
        if (head[1] != kCanvasRowVersion || role != static_cast<uint32_t>(CanvasRowRole::Replica)) return false;
        release();
        held_snapshot_.clear();
        role_ = CanvasRowRole::Replica;
        publish_every_ = 0;
        return true;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!in.read(bytes.data(), bytes.size())) return false;
    if (head[1] == 1) {                     // a bare canvas file: a host snapshot with no records
        std::vector<uint8_t> snap;
        put(snap, kSnapshotMagic);
        put(snap, kSnapshotVersion);
        put(snap, size);
        snap.insert(snap.end(), bytes.begin(), bytes.end());
        put(snap, uint32_t(0));             // no table files
        put(snap, uint32_t(0));             // no meta groups
        bytes = std::move(snap);
    }
    const CanvasRowRole previous_role = role_;
    const uint32_t previous_every = publish_every_;
    role_ = static_cast<CanvasRowRole>(role);
    publish_every_ = every;
    if (adopt(bytes)) return true;
    role_ = previous_role;
    publish_every_ = previous_every;
    return false;
}

int32_t CanvasRow::row_port_count() const {
    if (role_ == CanvasRowRole::Replica) return 1;
    return static_cast<int32_t>(ports_.size()) + (publish_every_ ? 1 : 0);
}

ToolRowPort CanvasRow::row_port(int32_t index) const {
    const int32_t count = row_port_count();
    if (index < 0 || index >= count) return ToolRowPort{};
    const float y = (static_cast<float>(index) + 0.5f) / static_cast<float>(count);
    if (role_ == CanvasRowRole::Replica) return ToolRowPort{ToolPortKind::Argument, 0.0f, y, "state", 8};
    if (index == static_cast<int32_t>(ports_.size())) return ToolRowPort{ToolPortKind::Return, 1.0f, y, "state", 8};
    const auto& p = ports_[static_cast<size_t>(index)];
    return ToolRowPort{p.into_canvas ? ToolPortKind::Argument : ToolPortKind::Return, p.into_canvas ? 0.0f : 1.0f, y,
                       p.name.c_str(), 0};
}

void CanvasRow::execute_stack(ToolStackContext& context) {
    context.succeeded = false;
    if (role_ == CanvasRowRole::Replica) replicate(context);
    else if (canvas_) host(context);
}

void CanvasRow::host(ToolStackContext& context) {
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
    // The tick this row was given goes on to the canvas it holds: its tables step on it, or free-run on their own workers
    // if their exec mode is Free (the canvas's own choice, saved with it). The step returns once that tick has run.
    if (!nodus_headless_step(canvas_, context.dt)) return;
    ++executions_;
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
    // The snapshot for the mesh, on "state", between ticks of the canvas (it is quiescent here). One waits at a time:
    // a refused one is offered again, and no newer one is taken until it is gone.
    const int32_t state_port = static_cast<int32_t>(ports_.size());
    if (publish_every_ && rp && rp->publish && rp->can_publish && (held_state_ || executions_ % publish_every_ == 0) &&
        rp->can_publish(rp->host, state_port, 1)) {
        if (!held_state_) {
            const std::vector<uint8_t> snap = snapshot();
            const uint64_t shape[1] = {static_cast<uint64_t>(snap.size())};
            const uint64_t t = snap.empty() ? 0 : nodus_tensor_create(NODUS_DTYPE_U8, 1, shape);
            if (t && nodus_tensor_write(t, 0, snap.data(), snap.size()) == static_cast<int64_t>(snap.size())) held_state_ = t;
            else if (t) nodus_tensor_destroy(t);
        }
        if (held_state_ && rp->publish(rp->host, state_port, &held_state_, sizeof(held_state_))) held_state_ = 0;
    }
    context.succeeded = true;
}

// A replica: each snapshot that arrives on "state" replaces the canvas; it is never stepped. The arrived tensor is the
// row's: it is destroyed once read. A snapshot that cannot be adopted leaves the previous replica.
void CanvasRow::replicate(ToolStackContext& context) {
    const ToolRowPortAccess* rp = context.row_ports;
    context.succeeded = true;
    if (!rp || !rp->consume) return;
    for (;;) {
        uint64_t t = 0;
        size_t got = 0;
        if (!rp->consume(rp->host, 0, &t, sizeof(t), &got) || got != sizeof(t)) break;
        if (!t) continue;
        NodusTensorDesc d{};
        std::vector<uint8_t> snap;
        if (nodus_tensor_describe(t, &d) == NODUS_OK && d.total_bytes && d.total_bytes <= kCanvasRowMaxBytes) {
            snap.resize(static_cast<size_t>(d.total_bytes));
            if (nodus_tensor_read(t, 0, snap.data(), snap.size()) != static_cast<int64_t>(snap.size())) snap.clear();
        }
        nodus_tensor_destroy(t);
        if (snap.empty() || !adopt(snap)) context.succeeded = false;
    }
}

}  // namespace nodus::canvas_row

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::canvas_row::CanvasRow(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
