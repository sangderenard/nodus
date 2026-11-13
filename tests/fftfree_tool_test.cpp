#include "fftfree_tool.h"
#include "tool_registry.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

using nodus::gpu::FftFreeTool;
using nodus::gpu::TensorStateLibraryTool;

namespace {
bool invoke(ITool& tool,std::uint64_t first,std::uint64_t second,
            std::uint64_t& output){
    RawStackFrame frame{};
    if(!raw_stack_init_frame(frame,4096))return false;
    bool ok=raw_stack_push_abstract_tensor(frame,{first})&&
            raw_stack_push_abstract_tensor(frame,{second});
    ToolStackContext context{};context.stack.raw=&frame;
    if(ok)tool.execute_stack(context);
    nodus::tensors::AbstractTensorHandle result{};
    ok=ok&&context.succeeded&&raw_stack_pop_abstract_tensor(frame,result);
    output=ok?result.id:0;
    raw_stack_free_mask(frame);
    if(frame.backend)gp_mem_backend_release(frame.backend);
    return ok;
}
template<std::size_t N>
std::uint64_t tensor(const std::array<std::uint64_t,N>& values){
    const std::uint64_t shape[1]={N};
    auto id=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    if(id)nodus_tensor_write(id,0,values.data(),sizeof(values));
    return id;
}
bool put(TensorStateLibraryTool& lib,std::uint64_t card,
         std::uint64_t address,std::uint64_t payload){
    std::array<std::uint64_t,12> words{};
    words[0]=2;words[1]=card;words[3]=address;
    auto command=tensor(words);
    std::uint64_t reply=0,result=UINT64_MAX;
    const bool sent=invoke(lib,command,payload,reply);
    if(sent)nodus_tensor_read(reply,0,&result,sizeof(result));
    if(reply)nodus_tensor_destroy(reply);
    if(command)nodus_tensor_destroy(command);
    return sent&&result==0;
}
}

int main(int argc,char** argv){
    if(argc!=2)return 2;
    TensorStateLibraryTool library(16*1024*1024,{});
    library.initialize({});
    constexpr std::uint64_t card=17;
    const std::uint64_t dummy_shape[1]={1};
    auto dummy=nodus_tensor_create(NODUS_DTYPE_U64,1,dummy_shape);
    std::array<std::uint64_t,12> card_words{};
    card_words[0]=1;card_words[1]=card;card_words[2]=901;
    card_words[4]=1024*1024;
    auto card_request=tensor(card_words);
    std::uint64_t card_reply=0;
    if(!invoke(library,card_request,dummy,card_reply))return 3;
    std::uint64_t card_status=UINT64_MAX;
    nodus_tensor_read(card_reply,0,&card_status,sizeof(card_status));
    if(card_status!=0)return 4;
    nodus_tensor_destroy(card_reply);
    nodus_tensor_destroy(card_request);
    const std::uint64_t input_shape[1]={8},output_shape[1]={5};
    float input_values[8]={1,0,0,0,0,0,0,0};
    auto input=nodus_tensor_create(NODUS_DTYPE_F32,1,input_shape);
    nodus_tensor_write(input,0,input_values,sizeof(input_values));
    if(!put(library,card,101,input))return 5;
    nodus_tensor_destroy(input);
    for(std::uint64_t address=102;address<=104;++address){
        auto target=nodus_tensor_create(NODUS_DTYPE_F32,1,output_shape);
        if(!put(library,card,address,target))return 6;
        nodus_tensor_destroy(target);
    }
    ToolRegistry registry;
    if(!nodus::gpu::register_fftfree_tool(registry,library,argv[1]))return 7;
    auto row=registry.create("nodus.fftfree");
    if(!row||row->port_count()!=3)return 8;
    row->initialize({});
    std::array<std::uint64_t,FftFreeTool::kConfigWords> config{};
    config[0]=1;config[1]=8;config[2]=1;config[3]=1;
    config[5]=1;config[6]=2;config[16]=2;
    config[17]=8;config[18]=8;config[19]=1;
    config[20]=1;config[23]=1;
    std::array<std::uint64_t,FftFreeTool::kWorkWords> work{};
    work[0]=1;work[1]=card;work[2]=101;
    work[4]=102;work[5]=103;work[6]=104;
    work[7]=8;work[8]=1;work[9]=7001;
    auto work_token=tensor(work),config_token=tensor(config);
    std::uint64_t reply=0;
    if(!invoke(*row,work_token,config_token,reply))return 9;
    std::array<std::uint64_t,FftFreeTool::kReceiptWords> receipt{};
    nodus_tensor_read(reply,0,receipt.data(),sizeof(receipt));
    if(receipt[0]!=0||receipt[1]!=7001||receipt[2]!=1||
       receipt[3]!=8||receipt[4]!=5){
        std::cerr<<"fftfree receipt "<<receipt[0]<<" frames="<<receipt[2]
                 <<" n="<<receipt[3]<<" bins="<<receipt[4]<<'\n';
        return 10;
    }
    for(std::uint64_t address=102;address<=104;++address){
        auto loan=library.borrow_cpu(card,address,false);
        if(!loan||loan.bytes()<5*sizeof(float))return 11;
        const auto* values=static_cast<const float*>(loan.data());
        for(int i=0;i<5;++i){
            const float expected=address==103?0.0f:1.0f;
            if(!std::isfinite(values[i])||std::fabs(values[i]-expected)>1e-4f)
                return 12;
        }
    }
    auto reconstructed=nodus_tensor_create(NODUS_DTYPE_F32,1,input_shape);
    if(!put(library,card,105,reconstructed))return 13;
    nodus_tensor_destroy(reconstructed);
    nodus_tensor_destroy(reply);
    nodus_tensor_destroy(work_token);
    nodus_tensor_destroy(config_token);
    config[4]=1;config[20]=2;
    work[0]=2;work[2]=102;work[3]=103;work[4]=105;
    work[5]=work[6]=0;work[7]=1;work[9]=7002;
    work_token=tensor(work);config_token=tensor(config);
    if(!invoke(*row,work_token,config_token,reply))return 14;
    receipt={};nodus_tensor_read(reply,0,receipt.data(),sizeof(receipt));
    if(receipt[0]!=0||receipt[2]!=1)return 15;
    {
        auto loan=library.borrow_cpu(card,105,false);
        if(!loan||loan.bytes()<sizeof(input_values))return 16;
        const auto* values=static_cast<const float*>(loan.data());
        for(int i=0;i<8;++i){
            const float expected=i==0?1.0f:0.0f;
            if(!std::isfinite(values[i])||std::fabs(values[i]-expected)>1e-4f)
                return 17;
        }
    }
    nodus_tensor_destroy(reply);
    nodus_tensor_destroy(work_token);
    nodus_tensor_destroy(config_token);
    config[4]=0;config[5]=2;config[20]=1;config[39]=1;
    work[0]=1;work[2]=101;work[3]=0;
    work[4]=102;work[5]=103;work[6]=104;
    work[7]=8;work[9]=7003;
    work_token=tensor(work);config_token=tensor(config);
    if(!invoke(*row,work_token,config_token,reply))return 18;
    receipt={};nodus_tensor_read(reply,0,receipt.data(),sizeof(receipt));
    if(receipt[0]!=0||receipt[2]!=1)return 19;
    {
        auto loan=library.borrow_cpu(card,104,false);
        if(!loan)return 20;
        const auto* values=static_cast<const float*>(loan.data());
        for(int i=0;i<5;++i)
            if(!std::isfinite(values[i])||std::fabs(values[i]-1.0f)>1e-4f)
                return 21;
    }
    const std::uint64_t long_shape[1]={16},spectrum_shape[1]={15};
    float long_values[16]={};long_values[0]=long_values[8]=1.0f;
    auto long_input=nodus_tensor_create(NODUS_DTYPE_F32,1,long_shape);
    nodus_tensor_write(long_input,0,long_values,sizeof(long_values));
    if(!put(library,card,106,long_input))return 22;
    nodus_tensor_destroy(long_input);
    for(std::uint64_t address=107;address<=109;++address){
        auto target=nodus_tensor_create(NODUS_DTYPE_F32,1,spectrum_shape);
        if(!put(library,card,address,target))return 23;
        nodus_tensor_destroy(target);
    }
    nodus_tensor_destroy(reply);
    nodus_tensor_destroy(work_token);
    nodus_tensor_destroy(config_token);
    config[5]=1;config[18]=4;config[39]=0;
    work[2]=106;work[4]=107;work[5]=108;work[6]=109;
    work[7]=16;work[8]=3;work[9]=7004;
    work_token=tensor(work);config_token=tensor(config);
    if(!invoke(*row,work_token,config_token,reply))return 24;
    receipt={};nodus_tensor_read(reply,0,receipt.data(),sizeof(receipt));
    if(receipt[0]!=0||receipt[2]!=3)return 25;
    {
        auto loan=library.borrow_cpu(card,109,false);
        if(!loan)return 26;
        const auto* values=static_cast<const float*>(loan.data());
        for(int i=0;i<15;++i)
            if(!std::isfinite(values[i])||std::fabs(values[i]-1.0f)>1e-4f)
                return 27;
    }
    const auto profile=library.work_performance(
        TensorStateLibraryTool::ExecutionApi::Cpu,0,
        (1ull<<48)|(1ull<<40)|8ull,3);
    if(!profile||profile->samples!=1||!profile->average_dispatch_ns)return 28;
    nodus_tensor_destroy(reply);
    nodus_tensor_destroy(work_token);
    nodus_tensor_destroy(config_token);
    nodus_tensor_destroy(dummy);
    row->shutdown();
    library.shutdown();
    return 0;
}
