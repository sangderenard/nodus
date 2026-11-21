// canvas_row: a table row that holds a whole canvas (docs/MESH_CANVAS_UP_PORTS_2026-11-21.md, stage 3). Owner,
// 2026-11-21: "compartmentalize the canvas in a table, don't use mandated global namespace just nest it".
//
// The row owns its canvas outright: a nested graph with its own scheduler, its own tables and its own sim frame count
// (ThreadManager::TickRequest::canvas, GP_TableContext::owner_canvas). It has one of two roles.
//
// Host: the canvas runs here. Its declared ports are the row's ports, in declaration order: an IN canvas port is a row
// Argument port (into the row, on into the canvas), an OUT canvas port a row Return port. Owner, 2026-11-21: "the top
// level gives a tick to a table gives a tick to a row so the contained canvas receives a tick and can respect it or
// free run". Each execution moves what waits on the row's Argument ports into the canvas, gives the canvas the row's
// tick (ToolStackContext::dt / tick_id) -- its tables step on it, or free-run on their own workers where their exec
// mode is Free -- and moves what reached its OUT ports onto the row's Return ports while their edges can take it.
// Nothing is dropped: a token the canvas refuses is held by the row and offered again next execution, and the row
// reads no more from that port until it is taken. With a publish interval, every that many executions the row also
// puts the canvas's snapshot on one more Return port, "state", for the mesh: only the node that runs a canvas writes it.
//
// Replica: another node's canvas, read only. The row never steps it. Its one port is an Argument port "state"; each
// snapshot that arrives there replaces the replica. Owner, 2026-11-21: "by having the canvas state we should collect
// also any meta edges supplying traffic metrics": a snapshot is the canvas file and every meta group's records, and the
// replica's meta groups carry those records (gp_table_meta_append_records), so every reader of a meta edge sees the
// writer's traffic.
//
// The mesh, secured (docs/MESH_CANVAS_UP_PORTS_2026-11-21.md, stages 3-4; module_library/source/tools/mesh_crypto.py).
// A host in mesh mode names its readers (node kids, public): it puts "NRS1" | n | reader kids | snapshot on "state",
// and the rows above it seal (to those readers; none = public) and sign (with the node's keys) on its way out. A
// replica names its one writer (a kid) and takes what the verify and open rows hand it, "NMO1" | writer | generation |
// readable | snapshot: only that writer, and only a later generation than the last one taken -- one writer per canvas,
// no replays. A replica that is no reader of the canvas learns only that its writer exists and its generation. A
// replica with no writer named takes plain snapshots, unsigned (the unsecured mesh of mesh_replica_test).
//
// The row's saved state (serialize/deserialize, the canvas file's ROW_STATE record) is its role, its publish interval,
// its mesh settings (readers, or the writer and the last generation taken) and the snapshot it holds, so a table's
// canvas rows are saved inside the canvas that holds the table.
// The source is module_library/source/tools/canvas_row.cpp.
#pragma once

#include "tool_api.h"

#include <cstdint>
#include <string>
#include <vector>

struct NodusHeadlessGraph;

namespace nodus::canvas_row {

inline constexpr const char* kCanvasRowToolId = "nodus.canvas_row";

enum class CanvasRowRole : uint32_t { Host = 0, Replica = 1 };

class CanvasRow : public ITool {
public:
    CanvasRow();
    ~CanvasRow() override;

    const char* id_cstr() const noexcept override { return kCanvasRowToolId; }
    const char* name_cstr() const noexcept override { return "Canvas row"; }
    void initialize(const ToolInitContext& ctx) override;
    void shutdown() override;
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& context) override;
    bool serialize(OutputArchive& out) const override;
    bool deserialize(InputArchive& in) override;
    int32_t row_port_count() const override;
    ToolRowPort row_port(int32_t index) const override;

    // The canvas it holds (null until a saved canvas is given to it).
    NodusHeadlessGraph* canvas() const { return canvas_; }

private:
    struct Port {
        std::string name;
        bool into_canvas = false;           // canvas IN port: the row's Argument port
        std::vector<uint8_t> held;          // a token the canvas refused, offered again next execution
    };
    // Snapshot: magic "NCS1", version 2, canvas byte count (u64), the canvas file's bytes; the files a save writes beside
    // it in <file>.modules/ (count u32; each: relative name length u32, name, byte count u64, bytes); meta group count
    // (u32), then per group its id (u64), its record count (u32) and those records (GP_META_RECORD_WORDS u64 each),
    // oldest first.
    std::vector<uint8_t> snapshot() const;
    bool adopt(const std::vector<uint8_t>& snapshot);
    void release();
    void host(ToolStackContext& context);
    void replicate(ToolStackContext& context);
    bool take(const std::vector<uint8_t>& arrived);

    CanvasRowRole role_ = CanvasRowRole::Host;
    uint32_t publish_every_ = 0;            // executions between snapshots on "state" (0 = none)
    uint64_t executions_ = 0;
    NodusHeadlessGraph* canvas_ = nullptr;
    std::vector<uint8_t> held_snapshot_;    // the snapshot as last given (a replica's state; a host's if it cannot save)
    uint64_t held_state_ = 0;               // a snapshot tensor the "state" edge refused, offered again
    std::vector<Port> ports_;
    bool mesh_ = false;                     // host: "state" carries "NRS1" (readers + snapshot) for the seal row
    std::vector<uint8_t> readers_;          // host: reader kids, 32 bytes each (none = public)
    std::vector<uint8_t> writer_;           // replica: the one writer's kid (32 bytes), or empty (unsecured)
    uint64_t last_generation_ = 0;          // replica: the generation last taken from that writer
    bool readable_ = false;                 // replica: the last snapshot taken could be opened here
};

}  // namespace nodus::canvas_row
