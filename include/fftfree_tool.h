#pragma once

#include "tensor_state_library_tool.h"
#include "tool_api.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>

class ToolRegistry;

namespace nodus::gpu {

// The configuration words mirror fft_cffi.hpp::fft_init_full_v2 arguments.
// Enum-valued controls are passed to fftfree, not reinterpreted by Nodus.
// Word 0 is schema version 1; words 8..15 are a mixed-radix pattern.
// Floating window parameters in words 32,33,35,36 use the low 32-bit IEEE
// float representation. Word 39 requests fftfree's experimental backup path.
// All other words are integer arguments in the order documented below.
class FftFreeTool final : public ITool {
public:
    static constexpr std::size_t kConfigWords=40, kWorkWords=10,
                                 kReceiptWords=9;
    // Work U64[10]: operation (1=real batched, 2=complex batched), card,
    // input real, input imag (0 for real), output real/PCM, output imag,
    // output magnitude, input samples/frames, max output frames, job id.
    // For complex operation only output real/PCM is used.
    explicit FftFreeTool(TensorStateLibraryTool& library,std::string dll);
    ~FftFreeTool() override { shutdown(); }
    const char* id_cstr() const noexcept override {return "nodus.fftfree";}
    const char* name_cstr() const noexcept override {return "fftfree transform";}
    ToolCaps caps() const override {return ToolCaps::None;}
    int32_t port_count() const override {return 3;}
    ToolPortSpec port_spec(int32_t index) const override {
        if(index==0||index==1)return {ToolPortKind::Argument,1};
        if(index==2)return {ToolPortKind::Return,1};
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
    std::array<std::uint64_t,kConfigWords> cached_{};
    bool cached_valid_=false;
    using Init=void* (*)(std::size_t,int,int,int,int,int,const int*,
        std::size_t,int,int,int,int,int,int,int,int,int,int,int,int,int,int,
        int,int,float,float,int,float,float,int,int);
    using ExecuteReal=std::size_t (*)(void*,const float*,std::size_t,
        float*,float*,float*,int,int,std::size_t);
    using ExecuteComplex=std::size_t (*)(void*,const float*,const float*,
        std::size_t,float*,int,int,std::size_t);
    Init init_=nullptr;
    ExecuteReal execute_real_=nullptr;
    ExecuteComplex execute_complex_=nullptr;
    std::size_t (*plan_size_)(void*)=nullptr;
    void (*free_)(void*)=nullptr;
};

bool register_fftfree_tool(ToolRegistry& registry,
    TensorStateLibraryTool& library,const std::string& dll);

} // namespace nodus::gpu
