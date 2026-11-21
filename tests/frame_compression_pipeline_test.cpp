#include "frame_tensor_packet_tool.h"
#include "frame_artifact_file_tool.h"
#include "exec_row.h"
#include "tool_registry.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "value_types.h"
#include <windows.h>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace {
using Handle=nodus::tensors::AbstractTensorHandle;
std::uint64_t invoke(ITool& tool,std::vector<std::uint64_t> inputs,bool* okay=nullptr) {
    RawStackFrame frame{};
    if(!raw_stack_init_frame(frame,1024))return 0;
    if(inputs.empty()) {
        const double idle=0;
        raw_stack_push_typed(frame,&idle,ValueTypeRegistry::global().builtin(VT_FLOAT64));
    }
    for(auto input:inputs)raw_stack_push_abstract_tensor(frame,{input});
    ToolStackContext context{};context.stack.raw=&frame;
    tool.execute_stack(context);
    Handle result{};
    raw_stack_pop_abstract_tensor(frame,result);
    if(okay)*okay=context.succeeded;
    raw_stack_free_mask(frame);
    if(frame.backend)gp_mem_backend_release(frame.backend);
    return result.id;
}
std::uint64_t wait_result(ITool& tool,std::uint64_t immediate) {
    if(immediate)return immediate;
    const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(std::chrono::steady_clock::now()<end) {
        if(auto next=invoke(tool,{}))return next;
        Sleep(1);
    }
    return 0;
}
}

int main(int argc,char** argv) {
    if(argc!=5&&argc!=8&&argc!=9)return 2; // optional raw RGBA, width, height, GL origin
    const std::filesystem::path directory=argv[1];
    std::filesystem::create_directories(directory);
    SetEnvironmentVariableA("NODUS_EXEC_CORE_ROOTS",argv[4]);
    HMODULE dll=LoadLibraryA(argv[2]);
    if(!dll)return 3;
    auto create=reinterpret_cast<CreateToolFn>(GetProcAddress(dll,"create_tool"));
    auto destroy=reinterpret_cast<void (*)(ITool*)>(GetProcAddress(dll,"destroy_tool"));
    if(!create||!destroy)return 4;
    auto make_exec=[&](const char* name,std::vector<std::string> args) {
        auto settings=nodus::exec::RowSettings{argv[3],std::move(args),directory.string(),1};
        const auto path=directory/name;
        {FileOutputArchive out(path.string().c_str());
         if(!nodus::exec::write_settings(out,settings))return std::pair<ITool*,std::filesystem::path>{nullptr,path};}
        ITool* tool=create();
        const auto saved=path.string();
        ToolInitContext context{};context.serialized_path=saved.c_str();
        tool->initialize(context);
        return std::pair{tool,path};
    };
    auto [compress,compress_settings]=make_exec("pack.settings",{"-c"});
    auto [decompress,decompress_settings]=make_exec("unpack.settings",{"-d","-c"});
    if(!compress||!decompress)return 5;

    const bool external=argc>=8;
    const std::uint32_t origin=argc==9?std::uint32_t(std::stoul(argv[8])):0;
    if(origin>1)return 2;
    const std::uint64_t height=external?std::stoull(argv[7]):32;
    const std::uint64_t width=external?std::stoull(argv[6]):32;
    constexpr std::uint64_t channels=4;
    if(!height||!width||height>4096||width>4096)return 6;
    const std::uint64_t shape[3]={height,width,channels};
    const auto dtype=external?NODUS_DTYPE_U8:NODUS_DTYPE_F32;
    const auto image=nodus_tensor_create(dtype,3,shape);
    if(!image)return 6;
    std::vector<std::uint8_t> pixels;
    if(external) {
        std::ifstream raw(argv[5],std::ios::binary);
        pixels.assign(std::istreambuf_iterator<char>{raw},{});
        if(pixels.size()!=height*width*channels)return 7;
    }else{
        std::vector<float> values(height*width*channels);
        for(std::size_t i=0;i<values.size();++i)values[i]=float(i%17)/17.0f;
        pixels.resize(values.size()*sizeof(float));
        std::memcpy(pixels.data(),values.data(),pixels.size());
    }
    if(nodus_tensor_write(image,0,pixels.data(),pixels.size())!=pixels.size())return 7;
    ToolRegistry registry;
    const std::string digest(64,'a');
    if(!nodus::gpu::register_frame_artifact_rows(registry,digest,directory,external?2:1,origin,
        1024*1024,1024*1024))return 8;
    auto packet=registry.create("nodus.frame.packet."+digest);
    auto sink=registry.create("nodus.frame.file."+digest);
    if(!packet||!sink)return 8;
    const auto envelope=invoke(*packet,{image});
    if(!envelope)return 8;
    NodusTensorDesc envelope_desc{};
    if(nodus_tensor_describe(envelope,&envelope_desc)!=NODUS_OK||
       envelope_desc.dtype!=NODUS_DTYPE_U8||envelope_desc.rank!=1)return 9;
    std::vector<std::uint8_t> expected(envelope_desc.total_bytes);
    if(nodus_tensor_read(envelope,0,expected.data(),expected.size())!=expected.size())return 10;
    const auto compressed=wait_result(*compress,invoke(*compress,{envelope}));
    if(!compressed)return 11;
    const std::uint64_t generation=27,one[1]={1};
    const auto control=nodus_tensor_create(NODUS_DTYPE_U64,1,one);
    nodus_tensor_write(control,0,&generation,sizeof(generation));
    bool okay=false;
    const auto receipt=invoke(*sink,{control,compressed},&okay);
    if(!okay||!receipt)return 12;
    std::array<std::uint64_t,4> status{};
    if(nodus_tensor_read(receipt,0,status.data(),sizeof(status))!=sizeof(status)||
       status[0]||status[1]!=generation||!status[2]||status[3]!=1)return 13;
    const auto* file_row=dynamic_cast<nodus::gpu::FrameArtifactFileTool*>(sink.get());
    if(!file_row)return 14;
    std::ifstream file(file_row->file_for(generation),std::ios::binary);
    const std::string stored(std::istreambuf_iterator<char>{file},{});
    if(stored.size()!=status[2]||stored.empty())return 14;
    const std::uint64_t compressed_shape[1]={stored.size()};
    const auto stored_tensor=nodus_tensor_create(NODUS_DTYPE_U8,1,compressed_shape);
    nodus_tensor_write(stored_tensor,0,stored.data(),stored.size());
    const auto restored=wait_result(*decompress,invoke(*decompress,{stored_tensor}));
    if(!restored)return 15;
    std::vector<std::uint8_t> actual(expected.size());
    NodusTensorDesc restored_desc{};
    if(nodus_tensor_describe(restored,&restored_desc)!=NODUS_OK||
       restored_desc.total_bytes!=expected.size()||
       nodus_tensor_read(restored,0,actual.data(),actual.size())!=actual.size()||
       actual!=expected)return 16;
    nodus::gpu::FrameTensorPacketMetadata metadata{};
    const auto reconstructed=nodus::gpu::decode_frame_tensor_packet(restored,&metadata);
    if(!reconstructed||metadata.color_encoding!=(external?2u:1u)||metadata.origin!=origin)return 17;
    NodusTensorDesc reconstructed_desc{};
    std::vector<std::uint8_t> reconstructed_pixels(pixels.size());
    if(nodus_tensor_describe(reconstructed,&reconstructed_desc)!=NODUS_OK||
       reconstructed_desc.dtype!=dtype||reconstructed_desc.rank!=3||
       reconstructed_desc.shape[0]!=height||reconstructed_desc.shape[1]!=width||
       reconstructed_desc.shape[2]!=channels||
       nodus_tensor_read(reconstructed,0,reconstructed_pixels.data(),
           reconstructed_pixels.size())!=reconstructed_pixels.size()||
       reconstructed_pixels!=pixels)return 18;
    nodus_tensor_destroy(reconstructed);
    nodus_tensor_destroy(restored);
    nodus_tensor_destroy(stored_tensor);
    nodus_tensor_destroy(receipt);
    nodus_tensor_destroy(control);
    nodus_tensor_destroy(compressed);
    nodus_tensor_destroy(envelope);
    nodus_tensor_destroy(image);
    compress->shutdown();decompress->shutdown();
    destroy(compress);destroy(decompress);
    FreeLibrary(dll);
    return 0;
}
