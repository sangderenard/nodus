// canvas_row: a table row that holds a whole canvas (docs/MESH_CANVAS_UP_PORTS_2026-11-21.md, stage 3). Owner,
// 2026-11-21: "compartmentalize the canvas in a table, don't use mandated global namespace just nest it".
//
// The row owns its canvas outright: a nested graph with its own scheduler, its own tables and its own sim frame count
// (ThreadManager::TickRequest::canvas, GP_TableContext::owner_canvas). The canvas's declared ports are the row's ports,
// in declaration order: an IN canvas port is a row Argument port (into the row, on into the canvas), an OUT canvas
// port a row Return port. Owner, 2026-11-21: "the top level gives a tick to a table gives a tick to a row so the
// contained canvas receives a tick and can respect it or free run". Each execution moves what waits on the row's
// Argument ports into the canvas, gives the canvas the row's tick (ToolStackContext::dt / tick_id) -- its tables step on
// it, or free-run on their own workers where their exec mode is Free -- and moves what reached its OUT ports onto the
// row's Return ports while their edges can take it. Nothing is
// dropped: a token the canvas refuses is held by the row and offered again next execution, and the row reads no more
// from that port until it is taken.
//
// The row's saved state (serialize/deserialize, the canvas file's ROW_STATE record) is the canvas it holds, so a table's
// canvas rows are saved inside the canvas that holds the table. The source is module_library/source/tools/canvas_row.cpp.
#pragma once

#include "tool_api.h"

#include <cstdint>
#include <string>
#include <vector>

struct NodusHeadlessGraph;

namespace nodus::canvas_row {

inline constexpr const char* kCanvasRowToolId = "nodus.canvas_row";

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
    int32_t row_port_count() const override { return static_cast<int32_t>(ports_.size()); }
    ToolRowPort row_port(int32_t index) const override;

    // The canvas it holds (null until a saved canvas is given to it).
    NodusHeadlessGraph* canvas() const { return canvas_; }

private:
    struct Port {
        std::string name;
        bool into_canvas = false;           // canvas IN port: the row's Argument port
        std::vector<uint8_t> held;          // a token the canvas refused, offered again next execution
    };
    bool adopt(const std::vector<uint8_t>& saved);
    void release();

    NodusHeadlessGraph* canvas_ = nullptr;
    std::vector<uint8_t> saved_;            // the canvas as last given (returned by serialize when it cannot save)
    std::vector<Port> ports_;
};

}  // namespace nodus::canvas_row
