#include "frame_artifact_file_tool.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "value_types.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

int main(int argc,char** argv) {
    if(argc!=2)return 2;
    const std::filesystem::path directory=argv[1];
    std::filesystem::create_directories(directory);
    nodus::gpu::FrameArtifactFileTool sink(directory,1024);
    if(sink.port_count()!=3)return 3;
    const std::uint64_t shape1[1]={1},shape8[1]={8};
    const auto control=nodus_tensor_create(NODUS_DTYPE_U64,1,shape1);
    const auto compressed=nodus_tensor_create(NODUS_DTYPE_U8,1,shape8);
    const std::uint64_t generation=1701;
    const std::array<std::uint8_t,8> bytes={31,139,8,0,0,0,0,0};
    if(!control||!compressed ||
       nodus_tensor_write(control,0,&generation,sizeof(generation))!=sizeof(generation) ||
       nodus_tensor_write(compressed,0,bytes.data(),bytes.size())!=bytes.size())return 4;
    RawStackFrame stack{};
    if(!raw_stack_init_frame(stack,1024))return 5;
    if(!raw_stack_push_abstract_tensor(stack,{control}) ||
       !raw_stack_push_abstract_tensor(stack,{compressed}))return 6;
    ToolStackContext context{};context.stack.raw=&stack;
    sink.execute_stack(context);
    nodus::tensors::AbstractTensorHandle result{};
    if(!context.succeeded||!raw_stack_pop_abstract_tensor(stack,result))return 7;
    std::array<std::uint64_t,4> receipt{};
    if(nodus_tensor_read(result.id,0,receipt.data(),sizeof(receipt))!=sizeof(receipt) ||
       receipt!=std::array<std::uint64_t,4>{0,generation,bytes.size(),1})return 8;
    const auto path=sink.file_for(generation);
    std::ifstream file(path,std::ios::binary);
    const std::string actual(std::istreambuf_iterator<char>{file},{});
    if(actual!=std::string(reinterpret_cast<const char*>(bytes.data()),bytes.size()))return 9;
    if(!raw_stack_push_abstract_tensor(stack,{control}) ||
       !raw_stack_push_abstract_tensor(stack,{compressed}))return 10;
    context.succeeded=true;
    sink.execute_stack(context);
    if(context.succeeded||sink.last_error()!="frame generation already committed")return 11;
    const std::uint64_t large_shape[1]={1025};
    const auto oversized=nodus_tensor_create(NODUS_DTYPE_U8,1,large_shape);
    const std::uint64_t next_generation=1702;
    nodus_tensor_write(control,0,&next_generation,sizeof(next_generation));
    raw_stack_reset(stack);
    raw_stack_push_abstract_tensor(stack,{control});
    raw_stack_push_abstract_tensor(stack,{oversized});
    context.succeeded=true;
    sink.execute_stack(context);
    if(context.succeeded||std::filesystem::exists(sink.file_for(next_generation)))return 12;
    nodus_tensor_destroy(oversized);
    nodus_tensor_destroy(result.id);
    nodus_tensor_destroy(compressed);
    nodus_tensor_destroy(control);
    raw_stack_free_mask(stack);
    if(stack.backend)gp_mem_backend_release(stack.backend);
    return 0;
}
