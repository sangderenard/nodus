#pragma once
#include "tensor_state_library_tool.h"
#include "tool_registry.h"

#include <cstdint>
#include <array>
#include <string>
#include <vector>

namespace nodus::gpu {

struct GpuShaderWorkContract {
    // Semantic program identity supplied by the artifact owner. CPU and GPU
    // products of the same law use the same identity in performance records.
    std::uint64_t program_id = 1;
    // One queued work token owns this many completed device steps before its
    // resident loadout expires and the FIFO may offer another token.
    std::uint32_t steps_per_loadout = 1;
    // This shader reads current position/velocity and publishes the next
    // pair at bindings 24/25. The runner copies them to 17/23 between steps.
    bool resident_state_feedback = false;
    // Optional host-observed dispatch completion target. This includes
    // submission and fence wait; it is not a GPU utilization measurement.
    // The row uses a rolling per-step average, as the existing spring row does.
    std::uint64_t target_completion_ns = 0;
};

// One ordinary native row per Vulkan device. The table's FIFO supplies a
// compact work token; tensor payloads stay in the library's addressed bank.
class GpuFabricDispatchTool final : public ITool {
public:
    GpuFabricDispatchTool(TensorStateLibraryTool& library, std::size_t device,
                          std::vector<std::uint32_t> shader,
                          std::uint32_t local_size,
                          GpuShaderWorkContract contract = {});
    const char* id_cstr() const noexcept override {return id_.c_str();}
    const char* name_cstr() const noexcept override {return "GPU fabric dispatch";}
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
    std::vector<std::uint32_t> shader_;
    std::uint32_t local_size_;
    GpuShaderWorkContract contract_;
    std::array<double,8> recent_step_ns_{};
    std::size_t step_samples_=0,step_window_next_=0;
    std::string id_;
};

// Called by the native owner after the library row has enumerated its GPU
// banks. The same shader product is admitted to each compatible device row.
bool register_gpu_fabric_dispatchers(ToolRegistry& registry,
    TensorStateLibraryTool& library,const std::vector<std::uint32_t>& shader,
    std::uint32_t local_size, GpuShaderWorkContract contract = {});
} // namespace nodus::gpu
