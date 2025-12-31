#include "tool_ir.h"
#include "tool_registry.h"

#include "mem_backend.h" // For SPIR-V translation integration
#include "kernel_isa.h" // Integrate KernelISA for KernelIR usage

namespace tool_ir {

// Example stub: ToolIR integration with SPIR-V translation
// (Replace or extend this with actual logic as needed)
void tool_ir_translate_kernel_to_spirv(const nodus::spirv::KernelIR& ir) {
    gp_mem_backend_translate_to_spirv(ir);
    // TODO: Use/store result as needed for the tool
}

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
