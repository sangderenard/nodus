#pragma once
// Lossless tensor envelope before a byte-span compressor. The producer owns
// interpretation fields; the packet keeps the tensor's exact dtype and shape.
#include "tool_api.h"
#include <cstdint>

namespace nodus::gpu {
struct FrameTensorPacketMetadata {
    std::uint32_t color_encoding=0,origin=0;
};
// Returns a new tensor handle (caller destroys it), or zero for malformed
// packet. This is the inverse of the packet row after exec decompression.
std::uint64_t decode_frame_tensor_packet(std::uint64_t packet_handle,
    FrameTensorPacketMetadata* metadata);
class FrameTensorPacketTool final : public ITool {
public:
    // 0 unknown, 1 linear, 2 sRGB; origin: 0 top-left, 1 bottom-left.
    FrameTensorPacketTool(std::uint32_t color_encoding,std::uint32_t origin,
                          std::uint64_t max_packet_bytes)
        :color_encoding_(color_encoding),origin_(origin),max_bytes_(max_packet_bytes){}
    const char* id_cstr() const noexcept override {return "nodus.frame.tensor.packet";}
    const char* name_cstr() const noexcept override {return "Frame tensor packet";}
    ToolCaps caps() const override {return ToolCaps::None;}
    int32_t port_count() const override {return 2;}
    ToolPortSpec port_spec(int32_t i) const override {
        if(i==0)return {ToolPortKind::Argument,1};
        if(i==1)return {ToolPortKind::Return,1};
        return {};
    }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double,HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext&) override;
private:
    std::uint32_t color_encoding_,origin_;
    std::uint64_t max_bytes_;
};
}
