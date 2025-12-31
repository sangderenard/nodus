#include "tool_ir.h"
#include "tool_registry.h"

namespace tool_ir {

    ToolRegistry::Entry make_registry_entry(std::shared_ptr<ToolIR> ir) {
        ToolRegistry::Entry entry;
        if (!ir) return entry;
        entry.id = ir->id;
        entry.name = ir->name;
        entry.caps = ir->caps;
        // factory captures the shared_ptr so the IR data stays alive for tool instances
        entry.factory = [ir]() -> std::unique_ptr<ITool, std::function<void(ITool*)>> {
            auto ptr = new BuiltinTool(ir);
            return std::unique_ptr<ITool, std::function<void(ITool*)>>(ptr, [](ITool* p){ delete p; });
        };
        return entry;
    }

}
