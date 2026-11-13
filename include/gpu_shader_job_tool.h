#pragma once
#include "tensor_state_library_tool.h"
#include "tool_registry.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nodus::gpu {

// A validated compute artifact's storage contract. Binding order is the
// SPIR-V descriptor order; addresses are base + binding index in the common
// state library. The caller retains the artifact receipt and source hash.
struct GpuShaderJobContract {
    struct ResetU32 {std::uint32_t binding=0,offset_words=0,count=0,value=0;};
    std::string id;
    std::uint64_t program_id=0;
    std::string entry="main";
    std::vector<std::uint32_t> spirv;
    std::vector<std::uint64_t> minimum_binding_bytes;
    std::vector<std::uint8_t> writable;
    // Optional logical state address for each descriptor, relative to the
    // work token's base. This lets adjacent shader stages share one state
    // tensor despite different GLSL binding ordinals. Empty means 0..N-1.
    std::vector<std::uint64_t> binding_offsets;
    std::vector<ResetU32> reset_before_dispatch;
    std::uint32_t local_size_x=0;
    std::uint32_t maximum_invocations=0;
    // A five-word work token adds the Y invocation count. Zero retains the
    // four-word, one-dimensional row ABI used by existing shader jobs.
    std::uint32_t local_size_y=0;
    std::uint32_t maximum_invocations_y=0;
};

class GpuShaderJobTool final : public ITool {
public:
    GpuShaderJobTool(TensorStateLibraryTool& library,std::size_t device,
                     GpuShaderJobContract contract);
    const char* id_cstr() const noexcept override {return id_.c_str();}
    const char* name_cstr() const noexcept override {return "GPU shader job";}
    ToolCaps caps() const override {return ToolCaps::None;}
    int32_t port_count() const override {return 2;}
    ToolPortSpec port_spec(int32_t index) const override {
        if(index==0)return {ToolPortKind::Argument,1};
        if(index==1)return {ToolPortKind::Return,1};
        return {};
    }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double,HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext&) override;
private:
    TensorStateLibraryTool& library_;
    std::size_t device_;
    GpuShaderJobContract contract_;
    std::string id_;
};

bool register_gpu_shader_job(ToolRegistry& registry,
                             TensorStateLibraryTool& library,
                             const GpuShaderJobContract& contract);

} // namespace nodus::gpu
