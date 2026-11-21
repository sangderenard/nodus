// frame_tensor_packet: a graph row that wraps a tensor, exactly, in a framed packet before a byte-span compressor
// (include/frame_tensor_packet_tool.h). The row tool's source, in the module library: the Menu lists it, the canvas
// compiler inlines it, it builds as its own plugin DLL, and the image registers configured rows from it
// (src/frame_tensor_packet_tool.cpp). Its settings are its saved state (serialize/deserialize); a row made with none
// has a zero packet limit and refuses every tensor.
#include "frame_tensor_packet_tool.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "value_types.h"
#include <array>
#include <cstring>
#include <limits>
#include <vector>

namespace nodus::gpu {
namespace {
constexpr std::array<std::uint8_t,8> kMagic={'N','F','R','M','0','0','1',0};
void write_u32(std::uint8_t*& p,std::uint32_t value) {
    for(int i=0;i<4;++i){*p++=std::uint8_t(value);value>>=8;}
}
void write_u64(std::uint8_t*& p,std::uint64_t value) {
    for(int i=0;i<8;++i){*p++=std::uint8_t(value);value>>=8;}
}
std::uint32_t read_u32(const std::uint8_t*& p) {
    std::uint32_t value=0;
    for(int i=0;i<4;++i)value|=std::uint32_t(*p++)<<(8*i);
    return value;
}
std::uint64_t read_u64(const std::uint8_t*& p) {
    std::uint64_t value=0;
    for(int i=0;i<8;++i)value|=std::uint64_t(*p++)<<(8*i);
    return value;
}
}
// Saved state: the row's settings (magic "FTP1", version 1, color encoding, origin, packet byte limit).
namespace {
constexpr std::uint32_t kPacketSettingsMagic=0x31505446,kPacketSettingsVersion=1;
}
bool FrameTensorPacketTool::serialize(OutputArchive& out) const{
    const std::uint32_t head[4]={kPacketSettingsMagic,kPacketSettingsVersion,color_encoding_,origin_};
    const std::uint64_t value=max_bytes_;
    return out.write(head,sizeof(head))&&out.write(&value,sizeof(value));
}
bool FrameTensorPacketTool::deserialize(InputArchive& in){
    std::uint32_t head[4]={};
    std::uint64_t value=0;
    if(!in.read(head,sizeof(head))||head[0]!=kPacketSettingsMagic||head[1]!=kPacketSettingsVersion||
       head[2]>2||head[3]>1||!in.read(&value,sizeof(value)))return false;
    configure(head[2],head[3],value);
    return true;
}
void FrameTensorPacketTool::initialize(const ToolInitContext& ctx){
    if(ctx.serialized_path&&ctx.serialized_path[0]!='\0'){
        FileInputArchive in(ctx.serialized_path);
        if(in.ok())deserialize(in);
    }
}
void FrameTensorPacketTool::execute_stack(ToolStackContext& context) {
    if(!context.stack.raw)return;
    auto& stack=*context.stack.raw;
    nodus::tensors::AbstractTensorHandle source{};
    if(!raw_stack_pop_abstract_tensor(stack,source))return;
    auto refuse=[&] {raw_stack_push_abstract_tensor(stack,source);context.succeeded=false;};
    NodusTensorDesc desc{};
    if(color_encoding_>2||origin_>1||source.id==0||
       nodus_tensor_describe(source.id,&desc)!=NODUS_OK||
       !desc.rank||desc.rank>NODUS_TENSOR_MAX_RANK||
       desc.layout!=0||
       desc.total_bytes>std::numeric_limits<std::uint64_t>::max()-(32+8ull*desc.rank)) {
        refuse();return;
    }
    // Fixed header: magic, dtype, rank, color, origin, byte count, then dims.
    const std::uint64_t header=32+8ull*desc.rank;
    const std::uint64_t total=header+desc.total_bytes;
    if(!max_bytes_||total>max_bytes_||
       total>std::numeric_limits<std::size_t>::max()) {refuse();return;}
    const std::uint64_t shape[1]={total};
    const auto packed=nodus_tensor_create(NODUS_DTYPE_U8,1,shape);
    if(!packed){refuse();return;}
    std::vector<std::uint8_t> bytes;
    try {bytes.resize(static_cast<std::size_t>(total));}
    catch (...) {nodus_tensor_destroy(packed);refuse();return;}
    auto* p=bytes.data();
    std::memcpy(p,kMagic.data(),kMagic.size());p+=kMagic.size();
    write_u32(p,desc.dtype);write_u32(p,desc.rank);
    write_u32(p,color_encoding_);write_u32(p,origin_);
    write_u64(p,desc.total_bytes);
    for(std::uint32_t i=0;i<desc.rank;++i)write_u64(p,desc.shape[i]);
    if(nodus_tensor_read(source.id,0,p,desc.total_bytes)!=
       static_cast<std::int64_t>(desc.total_bytes) ||
       nodus_tensor_write(packed,0,bytes.data(),total)!=
       static_cast<std::int64_t>(total) ||
       !raw_stack_push_abstract_tensor(stack,{packed})) {
        nodus_tensor_destroy(packed);refuse();return;
    }
}

std::uint64_t decode_frame_tensor_packet(std::uint64_t handle,
    FrameTensorPacketMetadata* metadata) {
    if(!metadata)return 0;
    NodusTensorDesc packet{};
    if(nodus_tensor_describe(handle,&packet)!=NODUS_OK||
       packet.dtype!=NODUS_DTYPE_U8||packet.rank!=1||packet.total_bytes<40||
       packet.total_bytes>std::numeric_limits<std::size_t>::max())return 0;
    std::vector<std::uint8_t> bytes;
    try {bytes.resize(static_cast<std::size_t>(packet.total_bytes));}
    catch (...) {return 0;}
    if(nodus_tensor_read(handle,0,bytes.data(),bytes.size())!=
       static_cast<std::int64_t>(bytes.size())||
       std::memcmp(bytes.data(),kMagic.data(),kMagic.size()))return 0;
    const auto* p=bytes.data()+kMagic.size();
    const auto dtype=read_u32(p),rank=read_u32(p);
    const auto encoding=read_u32(p),origin=read_u32(p);
    const auto payload_bytes=read_u64(p);
    if(!rank||rank>NODUS_TENSOR_MAX_RANK||encoding>2||origin>1||
       packet.total_bytes<32+8ull*rank||
       payload_bytes!=packet.total_bytes-(32+8ull*rank))return 0;
    std::array<std::uint64_t,NODUS_TENSOR_MAX_RANK> shape{};
    std::uint64_t count=1;
    for(std::uint32_t i=0;i<rank;++i) {
        shape[i]=read_u64(p);
        if(!shape[i]||count>std::numeric_limits<std::uint64_t>::max()/shape[i])return 0;
        count*=shape[i];
    }
    const auto element_bytes=nodus_tensor_dtype_size(dtype);
    if(!element_bytes||count>std::numeric_limits<std::uint64_t>::max()/element_bytes||
       count*element_bytes!=payload_bytes)return 0;
    const auto result=nodus_tensor_create(dtype,rank,shape.data());
    if(!result)return 0;
    if(nodus_tensor_write(result,0,p,payload_bytes)!=
       static_cast<std::int64_t>(payload_bytes)) {
        nodus_tensor_destroy(result);return 0;
    }
    *metadata={encoding,origin};
    return result;
}
}

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::gpu::FrameTensorPacketTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
