#include "wave_t4_state_tool.h"
#include "edge_tensor_fifo.h"
#include "tool_registry.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using nodus::gpu::TensorStateLibraryTool;

namespace {
bool invoke(ITool& tool,std::uint64_t first,std::uint64_t second,
            std::uint64_t& output){
    RawStackFrame frame{};
    if(!raw_stack_init_frame(frame,4096))return false;
    bool ok=raw_stack_push_abstract_tensor(frame,{first})&&
            raw_stack_push_abstract_tensor(frame,{second});
    ToolStackContext ctx{};ctx.stack.raw=&frame;
    if(ok)tool.execute_stack(ctx);
    nodus::tensors::AbstractTensorHandle result{};
    ok=ok&&ctx.succeeded&&raw_stack_pop_abstract_tensor(frame,result);
    output=ok?result.id:0;
    raw_stack_free_mask(frame);
    if(frame.backend)gp_mem_backend_release(frame.backend);
    return ok;
}
std::uint64_t u64_tensor(const std::vector<std::uint64_t>& values){
    const std::uint64_t shape[1]={values.size()};
    auto id=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    if(id)nodus_tensor_write(id,0,values.data(),values.size()*sizeof(values[0]));
    return id;
}
bool put(TensorStateLibraryTool& library,std::uint64_t card,
         std::uint64_t address,std::uint64_t payload){
    std::vector<std::uint64_t> request(12);
    request[0]=2;request[1]=card;request[3]=address;
    auto command=u64_tensor(request);
    std::uint64_t reply=0,status=UINT64_MAX;
    const bool ok=invoke(library,command,payload,reply);
    if(ok)nodus_tensor_read(reply,0,&status,sizeof(status));
    if(reply)nodus_tensor_destroy(reply);
    nodus_tensor_destroy(command);
    return ok&&status==0;
}
std::uint64_t bits(double value){
    std::uint64_t result=0;
    std::memcpy(&result,&value,sizeof(result));
    return result;
}
}

int main(int argc,char** argv){
    if(argc!=2)return 2;
    TensorStateLibraryTool library(16*1024*1024,{});
    library.initialize({});
    constexpr std::uint64_t card=17;
    std::vector<std::uint64_t> card_words(12);
    card_words[0]=1;card_words[1]=card;card_words[2]=904;
    card_words[4]=1024*1024;
    auto card_request=u64_tensor(card_words);
    const std::uint64_t one_shape[1]={1};
    auto dummy=nodus_tensor_create(NODUS_DTYPE_U64,1,one_shape);
    std::uint64_t reply=0,card_status=UINT64_MAX;
    if(!invoke(library,card_request,dummy,reply))return 3;
    nodus_tensor_read(reply,0,&card_status,sizeof(card_status));
    if(card_status!=0)return 4;
    nodus_tensor_destroy(reply);nodus_tensor_destroy(card_request);
    const std::uint64_t wavelength_shape[1]={2};
    auto wavelengths=nodus_tensor_create(NODUS_DTYPE_F64,1,wavelength_shape);
    const double wavelength_values[2]={450e-9,850e-9};
    nodus_tensor_write(wavelengths,0,wavelength_values,sizeof(wavelength_values));
    if(!put(library,card,100,wavelengths))return 5;
    nodus_tensor_destroy(wavelengths);
    const std::uint64_t field_shape[1]={256};
    for(std::uint64_t address=101;address<=104;++address){
        auto field=nodus_tensor_create(NODUS_DTYPE_F32,1,field_shape);
        float values[256]={};
        if(address==101||address==103)values[8*16+8]=1;
        nodus_tensor_write(field,0,values,sizeof(values));
        if(!put(library,card,address,field))return 6;
        nodus_tensor_destroy(field);
    }
    ToolRegistry registry;
    if(!nodus::gpu::register_wave_t4_state_tool(registry,library,argv[1]))return 7;
    auto row=registry.create("nodus.wave.t4.state");
    if(!row||row->port_count()!=3)return 8;
    row->initialize({});
    std::vector<std::uint64_t> control{
        1,16,16,2,1,bits(0.5e-6),bits(0.25e-6),1};
    std::vector<std::uint64_t> work{
        card,100,777,42,101,102,103,104};
    auto control_token=u64_tensor(control),work_token=u64_tensor(work);
    if(!invoke(*row,work_token,control_token,reply))return 9;
    std::array<std::uint64_t,9> receipt{};
    nodus_tensor_read(reply,0,receipt.data(),sizeof(receipt));
    if(receipt[0]!=0||receipt[1]!=777||receipt[2]!=42||
       receipt[3]!=2||receipt[4]!=1)return 10;
    EdgeTensorFifo fifo;
    fifo.configure({9},2,0,sizeof(std::uint64_t));
    if(!fifo.subscribe(201,true)||!fifo.subscribe(202,true))return 11;
    bool dropped=false;
    if(!fifo.push(7,301,receipt.data(),sizeof(receipt),&dropped)||dropped)
        return 12;
    for(std::uint64_t reader:{201ull,202ull}){
        std::array<std::uint64_t,9> seen{};
        std::size_t bytes=0;
        if(!fifo.pop(reader,seen.data(),sizeof(seen),bytes)||
           bytes!=sizeof(seen)||seen!=receipt)return 13;
    }
    for(std::uint64_t address:{101ull,103ull}){
        auto field=library.borrow_cpu(card,address,false);
        if(!field||field.bytes()<256*sizeof(float))return 14;
        const auto* values=static_cast<const float*>(field.data());
        double total=0;
        for(int i=0;i<256;++i){
            if(!std::isfinite(values[i]))return 15;
            total+=std::fabs(values[i]);
        }
        if(total<=0||std::fabs(values[8*16+8]-1.0f)<1e-5f)return 16;
    }
    nodus_tensor_destroy(reply);nodus_tensor_destroy(control_token);
    nodus_tensor_destroy(work_token);nodus_tensor_destroy(dummy);
    row->shutdown();library.shutdown();
    return 0;
}
