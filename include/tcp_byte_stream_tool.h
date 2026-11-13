#pragma once

#include "tool_registry.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nodus::net {

// A graph row for a bounded, framed TCP byte stream. The row's Argument and
// Return ports carry U8 AbstractTensor values through ordinary Nodus edges.
// The connection is loopback-only and uses ContractListener's existing Hello
// admission before switching to u64 little-endian length + payload frames.
struct TcpByteStreamConfig {
    std::string id;
    std::uint16_t port=0;
    std::size_t maximum_frame_bytes=4*1024*1024;
    std::size_t pending_frames=8;
};

class TcpByteStreamTool final : public ITool {
public:
    explicit TcpByteStreamTool(TcpByteStreamConfig config);
    ~TcpByteStreamTool() noexcept override;
    const char* id_cstr() const noexcept override{return config_.id.c_str();}
    const char* name_cstr() const noexcept override{return "TCP byte stream";}
    ToolCaps caps() const override{return ToolCaps::None;}
    int32_t port_count() const override{return 2;}
    ToolPortSpec port_spec(int32_t index) const override{
        if(index==0)return {ToolPortKind::Argument,1};
        if(index==1)return {ToolPortKind::Return,1};
        return {};
    }
    void initialize(const ToolInitContext&) override;
    void shutdown() override;
    void tick(double,HostAPI&) override;
    void render(RenderContext&) override{}
    void execute_stack(ToolStackContext&) override;
    std::uint16_t listen_port() const noexcept;
    bool uncertain_delivery() const noexcept;
private:
    struct Impl;
    TcpByteStreamConfig config_;
    std::unique_ptr<Impl> impl_;
};

bool register_tcp_byte_stream(ToolRegistry& registry,TcpByteStreamConfig config);

} // namespace nodus::net
