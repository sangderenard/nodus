#pragma once
#include "tensor_state_library_tool.h"
#include "tool_registry.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nodus::gpu {

// Each component receives U64[4] {card, state base, X invocation count,
// generation}, or U64[5] with a Y invocation count for tiled shaders. The
// pipeline input is U64[N,7] or U64[N,8]: {component index, API, program ID,
// card, state base, X invocation count, generation[, Y invocation count]}.
// API and program ID must agree
// with the registered component. It carries no shader source or duplicate
// transport implementation. Its receipt is U64[N+1,9]: a summary row,
// followed by each existing component receipt in execution order.
struct KernelPipelineComponent {
    std::string tool_id;
    TensorStateLibraryTool::ExecutionApi api;
    std::uint64_t program_id=0;
};

class KernelPipelineTool final : public ITool {
public:
    KernelPipelineTool(std::string id,std::vector<KernelPipelineComponent> components,
                       ToolRegistry& registry);
    const char* id_cstr() const noexcept override {return id_.c_str();}
    const char* name_cstr() const noexcept override {return "Kernel pipeline";}
    ToolCaps caps() const override {return ToolCaps::None;}
    int32_t port_count() const override {return 2;}
    ToolPortSpec port_spec(int32_t index) const override {
        if(index==0)return {ToolPortKind::Argument,1};
        if(index==1)return {ToolPortKind::Return,1};
        return {};
    }
    void initialize(const ToolInitContext&) override;
    void shutdown() override;
    void tick(double,HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext&) override;
private:
    using ToolPtr=std::unique_ptr<ITool,std::function<void(ITool*)>>;
    std::string id_;
    std::vector<KernelPipelineComponent> components_;
    std::vector<ToolPtr> tools_;
};

bool register_kernel_pipeline(ToolRegistry& registry,std::string id,
                              std::vector<KernelPipelineComponent> components);

} // namespace nodus::gpu
