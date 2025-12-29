#include "tool_registry.h"
#include "tool_api.h"
#include "table_abi.h"
#include "canvas_abi.h"
#include <vector>
#include <string>

class MetaEdgeLassoTool : public ITool {
public:
    MetaEdgeLassoTool() {}
    ~MetaEdgeLassoTool() noexcept override {}

    const char* id_cstr() const noexcept override { return "meta_edge_lasso"; }
    const char* name_cstr() const noexcept override { return "Meta-Edge Lasso"; }
    // rely on base `id()`/`name()` which call `id_cstr()`/`name_cstr()`
    ToolCaps caps() const override { return ToolCaps::Table | ToolCaps::Text; }

    void initialize(const ToolInitContext& ctx) override { (void)ctx; }
    void shutdown() override {}

    void tick(double dt, HostAPI& host) override { (void)dt; (void)host; }
    void render(RenderContext& ctx) override { (void)ctx; }

    void render_text(TextRenderArgs& args) override {
        if (!args.text) return;
        (void)args;
    }

    void render_table(TableRenderArgs& args) override {
        // no-op: visual lasso rendering handled by canvas UI directly
        (void)args;
    }

    void execute_stack(ToolStackContext& ctx) override { (void)ctx; }
};

REGISTER_TOOL(MetaEdgeLassoTool, "meta_edge_lasso", "Meta-Edge Lasso", ToolCaps::Table | ToolCaps::Text);
