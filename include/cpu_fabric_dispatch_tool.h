#pragma once
#include "tensor_state_library_tool.h"
#include "tool_registry.h"

#include <cstdint>
#include <string>

namespace nodus::gpu {
class CpuFabricDispatchTool final : public ITool {
public:
    CpuFabricDispatchTool(TensorStateLibraryTool& library,std::string dll,
                          std::string entry,std::uint64_t batch,
                          std::uint32_t steps_per_loadout=1,
                          std::uint64_t program_id=1);
    const char* id_cstr() const noexcept override {return "nodus.fabric.dispatch.cpu";}
    const char* name_cstr() const noexcept override {return "CPU fabric dispatch";}
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
    TensorStateLibraryTool& library_;
    std::string dll_,entry_name_;
    std::uint64_t batch_;
    std::uint32_t steps_per_loadout_;
    std::uint64_t program_id_;
    void* module_=nullptr;
    void (*entry_)(void**,int32_t*)=nullptr;
    void (*reset_)()=nullptr;
    std::uint32_t (*take_)()=nullptr;
};
bool register_cpu_fabric_dispatcher(ToolRegistry& registry,
    TensorStateLibraryTool& library,const std::string& dll,
    const std::string& entry,std::uint64_t batch,
    std::uint32_t steps_per_loadout=1,std::uint64_t program_id=1);
} // namespace nodus::gpu
