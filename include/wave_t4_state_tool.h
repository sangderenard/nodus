#pragma once

#include "tensor_state_library_tool.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>

class ToolRegistry;

namespace nodus::gpu {

// Runs the authored fftfree-backed T4 angular-spectrum stage on addressed
// arena field planes. One row instance owns one non-reentrant T4 plan.
class WaveT4StateTool final : public ITool {
public:
    static constexpr std::size_t kControlWords=8,kReceiptWords=9;
    // Control U64[8]: schema=1, nx, ny, arena count, bands per arena,
    // IEEE F64 bits for dx and dz, signed direction (+1 or -1).
    // Work U64[4+2*arena count]: card, wavelengths F64 state address,
    // frame identity, continuation generation, then each arena's real and
    // imaginary F32 state addresses. Every plane is modified in place.
    WaveT4StateTool(TensorStateLibraryTool& library,std::string dll);
    ~WaveT4StateTool() override {shutdown();}
    const char* id_cstr() const noexcept override {return "nodus.wave.t4.state";}
    const char* name_cstr() const noexcept override {return "T4 arena state";}
    ToolCaps caps() const override {return ToolCaps::None;}
    int32_t port_count() const override {return 3;}
    ToolPortSpec port_spec(int32_t i) const override {
        if(i==0||i==1)return {ToolPortKind::Argument,1};
        if(i==2)return {ToolPortKind::Return,1};
        return {};
    }
    void initialize(const ToolInitContext&) override;
    void shutdown() override;
    void tick(double,HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext&) override;
private:
    TensorStateLibraryTool& library_;
    std::string dll_;
    std::mutex mutex_;
    void* module_=nullptr;
    void* plan_=nullptr;
    int plan_nx_=0,plan_ny_=0;
    void* (*create_)(int,int)=nullptr;
    void (*destroy_)(void*)=nullptr;
    int (*step_)(void*,int,int,double,double,int,const double*,
        float* const*,float* const*)=nullptr;
};

bool register_wave_t4_state_tool(ToolRegistry& registry,
    TensorStateLibraryTool& library,const std::string& dll);

} // namespace nodus::gpu
