#include "tool_registry.h"
#include "tool_ir.h"

static bool register_example_ir_tool() {
    auto ir = std::make_shared<ToolIR>();
    ir->id = "ir_example";
    ir->name = "IR Example Tool";
    ir->caps = ToolCaps::Table | ToolCaps::Text;

    ir->initialize = [](const ToolInitContext& ctx) { (void)ctx; };
    ir->shutdown = []() {};
    ir->tick = [](double, HostAPI&){ };
    ir->render = [](RenderContext&){ };
    ir->render_text = [](TextRenderArgs&){ };
    ir->execute_stack = [](ToolStackContext&){ };

    return register_tool_ir(ir);
}

static bool g_registered = register_example_ir_tool();
