#include "tool_registry.h"
#include "tool_api.h"
#include "table_abi.h"
#include "canvas_abi.h"
#include <string>
#include <vector>
#include <sstream>

// Source string for later piecewise compiling/exporting
static const char* kEdgeIntrospectToolSource = R"SRC(
// Edge Introspection Tool (source)
// This source is provided as a string for later piecewise compilation.
// It inspects per-edge FIFO state and renders a multi-row report.

// NOTE: This is a placeholder textual source; the compiled tool below
// implements similar functionality in C++ directly linked into the
// executable for convenience.

)SRC";

class EdgeIntrospectTool : public ITool {
public:
    EdgeIntrospectTool() {}
    ~EdgeIntrospectTool() noexcept override {}

    std::string id() const override { return "edge_introspect"; }
    std::string name() const override { return "Edge Introspect"; }
    ToolCaps caps() const override { return ToolCaps::Table | ToolCaps::Text; }

    void initialize(const ToolInitContext& ctx) override {
        (void)ctx;
    }
    void shutdown() override {}

    void tick(double dt, HostAPI& host) override {
        (void)dt; (void)host;
    }

    void render(RenderContext& ctx) override {
        // Not used; canvas will call render_table/render_text as appropriate.
        (void)ctx;
    }

    void render_text(TextRenderArgs& args) override {
        // Minimal: draw a static help string when the tool's text rendering is invoked.
        if (!args.text) return;
        // Use HostAPI::draw_text if available via a host, but here we just ignore.
        (void)args;
    }

    void render_table(TableRenderArgs& args) override {
        if (!args.frame_buffer || args.rows <= 0 || args.cols <= 0) return;

        // Do NOT enumerate or read all edges by default. The intended
        // workflow is that this tool will be configured to target a small
        // list of meta-edges (the 'meta group') which are the only edges
        // allowed to affix to ropes. Until such a configuration exists we
        // render a helpful placeholder and avoid scanning the full edge
        // list which can be large.
        int32_t ccount = args.cols;
        int32_t rcount = args.rows;
        std::vector<GP_TableColumn> cols(static_cast<size_t>(ccount));
        for (int32_t c = 0; c < ccount; ++c) {
            cols[static_cast<size_t>(c)].kind = GP_TABLE_CELL_TEXT;
            cols[static_cast<size_t>(c)].width_px = static_cast<int32_t>(std::max(4.0f, args.cell_w));
            cols[static_cast<size_t>(c)].align = 0;
        }

        std::vector<GP_TableRow> rows(static_cast<size_t>(rcount));

        // Render a small instruction until the tool is explicitly provided
        // with a list of target meta-edges (via future UI/API). This avoids
        // scanning or reading the full edge table.
        if (rcount > 0) {
            GP_TableRow h; memset(&h, 0, sizeof(h));
            h.kind = GP_TABLE_ROW_HEADER; h.depth = 0; h.expanded = 1; h.cell_count = std::min(8, ccount);
            std::snprintf(h.cells[0].text, sizeof(h.cells[0].text), "Edge Introspect (meta-group only)");
            rows[0] = h;
        }
        if (rcount > 1) {
            GP_TableRow pr; memset(&pr, 0, sizeof(pr));
            pr.kind = GP_TABLE_ROW_DEVICE; pr.depth = 0; pr.expanded = 1; pr.cell_count = std::min(8, ccount);
            pr.cells[0].kind = GP_TABLE_CELL_TEXT;
            std::snprintf(pr.cells[0].text, sizeof(pr.cells[0].text), "No meta-edge group configured. Use the tool UI to bind a meta group.");
            rows[1] = pr;
        }

        GP_TableStyle st{};
        st.width_px = static_cast<int32_t>(std::max(1.0f, args.cell_w * args.cols));
        st.row_h_px = static_cast<int32_t>(std::max(8.0f, args.cell_h));
        st.name_w_px = 140;

        GP_TableGeom geom{};
        uint8_t* out_rgba = reinterpret_cast<uint8_t*>(args.frame_buffer);
        int32_t out_len = args.pitch * static_cast<int32_t>(std::max(1, static_cast<int>(args.rows)));
        gp_table_raster_rgba(rows.data(), rcount, cols.data(), ccount, &st, out_rgba, out_len, &geom);
    }

    void execute_stack(ToolStackContext& ctx) override {
        // Not a stack tool.
        (void)ctx;
    }
};

REGISTER_TOOL(EdgeIntrospectTool, "edge_introspect", "Edge Introspect", ToolCaps::Table | ToolCaps::Text);

