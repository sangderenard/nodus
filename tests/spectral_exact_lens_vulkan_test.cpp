#include "tensor_state_library_tool.h"
#include "gpu_shader_job_tool.h"
#include "kernel_pipeline_tool.h"
#include "tool_registry.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {
constexpr std::uint64_t kCard=173,kBase=17000;

bool send_library(ITool& tool,const std::array<std::uint64_t,12>& words,
                  std::uint64_t payload){
    const std::uint64_t shape[1]={words.size()};
    auto request=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    if(!request||nodus_tensor_write(request,0,words.data(),sizeof(words))!=sizeof(words))return false;
    RawStackFrame frame{};
    if(!raw_stack_init_frame(frame,4096))return false;
    raw_stack_push_abstract_tensor(frame,{request});
    raw_stack_push_abstract_tensor(frame,{payload});
    ToolStackContext context{};context.stack.raw=&frame;
    tool.execute_stack(context);
    nodus::tensors::AbstractTensorHandle result{};
    std::array<std::uint64_t,32> receipt{};
    const bool okay=context.succeeded&&
        raw_stack_pop_abstract_tensor(frame,result)&&
        nodus_tensor_read(result.id,0,receipt.data(),sizeof(receipt))==sizeof(receipt)&&
        receipt[0]==0;
    if(result.id)nodus_tensor_destroy(result.id);
    nodus_tensor_destroy(request);
    raw_stack_free_mask(frame);
    if(frame.backend)gp_mem_backend_release(frame.backend);
    return okay;
}

bool put(ITool& tool,std::uint64_t address,const std::vector<std::uint32_t>& data){
    const std::uint64_t shape[1]={data.size()};
    auto tensor=nodus_tensor_create(NODUS_DTYPE_U32,1,shape);
    if(!tensor||nodus_tensor_write(tensor,0,data.data(),data.size()*4)!=data.size()*4)return false;
    std::array<std::uint64_t,12> command{};
    command[0]=2;command[1]=kCard;command[3]=address;
    const bool okay=send_library(tool,command,tensor);
    nodus_tensor_destroy(tensor);
    return okay;
}

float f(std::uint32_t v){return std::bit_cast<float>(v);}
}

int main(int argc,char** argv){
    if(argc<3||argc>6)return 2;
    std::ifstream shader_source(argv[1],std::ios::binary);
    std::vector<char> shader_bytes{std::istreambuf_iterator<char>(shader_source),{}};
    std::ifstream payload_source(argv[2],std::ios::binary);
    std::vector<char> payload_bytes{std::istreambuf_iterator<char>(payload_source),{}};
    if(shader_bytes.empty()||shader_bytes.size()%4||payload_bytes.size()<32||
       payload_bytes.size()%4)return 2;
    std::vector<std::uint32_t> shader(shader_bytes.size()/4);
    std::memcpy(shader.data(),shader_bytes.data(),shader_bytes.size());
    std::vector<std::uint32_t> payload(payload_bytes.size()/4);
    std::memcpy(payload.data(),payload_bytes.data(),payload_bytes.size());
    const float first=f(payload[8]);
    const int surfaces=static_cast<int>(f(payload[1]));
    const float last=f(payload[8+(surfaces-1)*8]);
    if(f(payload[0])!=14949.0f||surfaces<2||!(last>first))return 3;

    auto library_tool=tool_registry_global().create("nodus.tensor.state.library");
    if(!library_tool)return 4;
    library_tool->initialize({});
    auto* library=dynamic_cast<nodus::gpu::TensorStateLibraryTool*>(library_tool.get());
    if(!library||library->device_count()==0)return 4;
    std::array<std::uint64_t,12> card{};
    card[0]=1;card[1]=kCard;card[2]=1800;card[4]=8*1024*1024;
    card[5]=card[6]=card[7]=4*1024*1024;
    const std::uint64_t dummy_shape[1]={1};
    auto dummy=nodus_tensor_create(NODUS_DTYPE_U64,1,dummy_shape);
    if(!send_library(*library_tool,card,dummy))return 5;

    constexpr std::size_t hit_stride=93;
    std::vector<std::uint32_t> hit(hit_stride,0);
    hit[0]=std::bit_cast<std::uint32_t>(first+0.0001f);
    hit[1]=std::bit_cast<std::uint32_t>(0.001f);
    hit[3]=std::bit_cast<std::uint32_t>(-1.0f);
    hit[6]=std::bit_cast<std::uint32_t>(1.0f);
    hit[28]=std::bit_cast<std::uint32_t>(1.0f);
    std::vector<std::uint32_t> counter(24,0);counter[0]=1;
    std::vector<std::uint32_t> control(37,0);
    control[0]=1;control[1]=1;control[2]=3;
    control[3]=std::bit_cast<std::uint32_t>(299792458.0f/(0.486e-6f));
    control[4]=std::bit_cast<std::uint32_t>(299792458.0f/(0.587e-6f));
    control[5]=std::bit_cast<std::uint32_t>(299792458.0f/(0.656e-6f));
    const std::array<std::vector<std::uint32_t>,9> inputs{
        hit, std::vector<std::uint32_t>(16,0), std::vector<std::uint32_t>{0},
        std::vector<std::uint32_t>{5}, std::vector<std::uint32_t>(16,0),
        counter,payload,std::vector<std::uint32_t>(28,0),control};
    const std::array<std::uint64_t,9> offsets{1,5,10,11,12,2,13,14,15};
    for(std::size_t i=0;i<inputs.size();i++)
        if(!put(*library_tool,kBase+offsets[i],inputs[i]))return 6;
    nodus::gpu::GpuShaderJobContract contract;
    contract.id="nodus.shader.spectral.exact_lens_t2";
    // First 64 bits of the authored ray_refine.comp.glsl SHA-256 receipt.
    contract.program_id=0xcf93afc7099c2071ull;
    contract.spirv=std::move(shader);
    contract.minimum_binding_bytes.reserve(9);
    for(const auto& row:inputs)contract.minimum_binding_bytes.push_back(row.size()*4);
    contract.writable={1,0,0,0,0,1,0,1,0};
    contract.binding_offsets.assign(offsets.begin(),offsets.end());
    contract.local_size_x=64;contract.maximum_invocations=1;
    if(!nodus::gpu::register_gpu_shader_job(tool_registry_global(),*library,contract))return 7;

    for(std::size_t device=0;device<library->device_count();device++){
        {
            auto initial=library->borrow_cpu(kCard,kBase+offsets[0],true);
            auto counters=library->borrow_cpu(kCard,kBase+offsets[5],true);
            if(!initial||!counters||initial.bytes()!=hit.size()*4||
               counters.bytes()!=counter.size()*4)return 8;
            std::memcpy(initial.data(),hit.data(),hit.size()*4);
            std::memcpy(counters.data(),counter.data(),counter.size()*4);
        }
        auto worker=tool_registry_global().create(contract.id+".vulkan."+std::to_string(device));
        if(!worker)return 8;
        const std::uint64_t work_shape[1]={4};
        auto work=nodus_tensor_create(NODUS_DTYPE_U64,1,work_shape);
        const std::uint64_t command[4]={kCard,kBase,1,device+1};
        nodus_tensor_write(work,0,command,sizeof(command));
        RawStackFrame frame{};
        if(!raw_stack_init_frame(frame,4096))return 9;
        raw_stack_push_abstract_tensor(frame,{work});
        ToolStackContext context{};context.stack.raw=&frame;
        worker->execute_stack(context);
        nodus::tensors::AbstractTensorHandle result{};
        std::array<std::uint64_t,9> receipt{};
        if(!context.succeeded||!raw_stack_pop_abstract_tensor(frame,result)||
           nodus_tensor_read(result.id,0,receipt.data(),sizeof(receipt))!=sizeof(receipt)||
           receipt[0]!=0)return 10;
        nodus_tensor_destroy(result.id);nodus_tensor_destroy(work);
        raw_stack_free_mask(frame);if(frame.backend)gp_mem_backend_release(frame.backend);
        auto output=library->borrow_cpu(kCard,kBase+offsets[0],false);
        if(!output||output.bytes()!=hit.size()*4)return 11;
        const auto* values=static_cast<const std::uint32_t*>(output.data());
        const float x=f(values[0]);
        const float y=f(values[1]);
        const float dy=f(values[7]);
        const std::uint32_t flags=values[16];
        std::cout<<"device="<<device<<" first="<<first<<" last="<<last
                 <<" exit_x="<<x<<" exit_y="<<y<<" exit_dy="<<dy
                 <<" flags="<<flags
                 <<" dispatch_ms="<<receipt[3]*1e-6<<"\n";
        if(!std::isfinite(x)||std::abs(x-last)>0.002f||
           !std::isfinite(y)||!std::isfinite(dy)||
           y<=0.001f||dy<=0.001f||
           (flags&4u)==0||(flags&8u)!=0)return 12;
    }
    for(std::size_t i=0;i<inputs.size();i++)
        if(!put(*library_tool,kBase+100+offsets[i],inputs[i]))return 13;
    {
        auto initial=library->borrow_cpu(kCard,kBase+offsets[0],true);
        auto counters=library->borrow_cpu(kCard,kBase+offsets[5],true);
        if(!initial||!counters)return 14;
        std::memcpy(initial.data(),hit.data(),hit.size()*4);
        std::memcpy(counters.data(),counter.data(),counter.size()*4);
    }
    using Api=nodus::gpu::TensorStateLibraryTool::ExecutionApi;
    const std::string pipeline_id="nodus.pipeline.spectral.exact_lens_doublet";
    if(!nodus::gpu::register_kernel_pipeline(tool_registry_global(),pipeline_id,
        {{contract.id+".vulkan.0",Api::Vulkan,contract.program_id},
         {contract.id+".vulkan.1",Api::Vulkan,contract.program_id}}))return 15;
    auto pipeline=tool_registry_global().create(pipeline_id);
    if(!pipeline)return 16;
    pipeline->initialize({});
    const std::uint64_t pipeline_shape[2]={2,7};
    auto pipeline_input=nodus_tensor_create(NODUS_DTYPE_U64,2,pipeline_shape);
    const std::uint64_t rows[14]={0,1,contract.program_id,kCard,kBase,1,10,
                                  1,1,contract.program_id,kCard,kBase+100,1,11};
    if(!pipeline_input||nodus_tensor_write(pipeline_input,0,rows,sizeof(rows))!=sizeof(rows))return 17;
    RawStackFrame pipeline_frame{};
    if(!raw_stack_init_frame(pipeline_frame,4096))return 17;
    raw_stack_push_abstract_tensor(pipeline_frame,{pipeline_input});
    ToolStackContext pipeline_context{};pipeline_context.stack.raw=&pipeline_frame;
    pipeline->execute_stack(pipeline_context);
    nodus::tensors::AbstractTensorHandle pipeline_output{};
    std::array<std::uint64_t,27> pipeline_receipt{};
    if(!pipeline_context.succeeded||
       !raw_stack_pop_abstract_tensor(pipeline_frame,pipeline_output)||
       nodus_tensor_read(pipeline_output.id,0,pipeline_receipt.data(),
                         sizeof(pipeline_receipt))!=sizeof(pipeline_receipt)||
       pipeline_receipt[0]!=0||pipeline_receipt[2]!=2||
       pipeline_receipt[3]!=2||pipeline_receipt[9]!=0||
       pipeline_receipt[18]!=0)return 18;
    auto second=library->borrow_cpu(kCard,kBase+100+offsets[0],false);
    if(!second||std::abs(f(static_cast<const std::uint32_t*>(second.data())[0])-last)>
       0.002f)return 19;
    std::cout<<"pipeline_stages="<<pipeline_receipt[3]<<" ordered=1\n";
    nodus_tensor_destroy(pipeline_output.id);nodus_tensor_destroy(pipeline_input);
    raw_stack_free_mask(pipeline_frame);
    if(pipeline_frame.backend)gp_mem_backend_release(pipeline_frame.backend);
    pipeline->shutdown();
    if(argc>=4){
        std::ifstream sensor_source(argv[3],std::ios::binary);
        std::vector<char> sensor_bytes{std::istreambuf_iterator<char>(sensor_source),{}};
        if(sensor_bytes.empty()||sensor_bytes.size()%4)return 20;
        std::vector<std::uint32_t> sensor_shader(sensor_bytes.size()/4);
        std::memcpy(sensor_shader.data(),sensor_bytes.data(),sensor_bytes.size());
        constexpr std::uint64_t sensor_base=kBase+1000;
        std::vector<std::uint32_t> node(16,0),lineage(8,0),control(8,0),
            counters(12,0),intent(86,0),runtime(38,0);
        node[12]=std::bit_cast<std::uint32_t>(1.0f);
        lineage[2]=lineage[3]=std::bit_cast<std::uint32_t>(0.5f);
        lineage[6]=std::bit_cast<std::uint32_t>(1.0f);
        control[7]=1;
        runtime[5]=runtime[10]=std::bit_cast<std::uint32_t>(1.0f);
        runtime[28]=1; // sensor_res
        runtime[30]=1; // n_bands
        runtime[31]=1; // max_bounces
        runtime[32]=std::bit_cast<std::uint32_t>(0.001f);
        runtime[33]=std::bit_cast<std::uint32_t>(1.0f);
        // std430 control offsets from the reflected raygen contract:
        // target.x at byte 64, half widths at bytes 44/48.
        runtime[11]=runtime[12]=std::bit_cast<std::uint32_t>(0.01f);
        runtime[16]=std::bit_cast<std::uint32_t>(1.0f);
        const std::array<std::vector<std::uint32_t>,9> sensor_inputs{
            node,lineage,control,std::vector<std::uint32_t>{1},counters,
            intent,std::vector<std::uint32_t>{0},
            std::vector<std::uint32_t>{0},runtime};
        for(std::size_t i=0;i<sensor_inputs.size();i++)
            if(!put(*library_tool,sensor_base+i,sensor_inputs[i]))return 21;
        nodus::gpu::GpuShaderJobContract sensor_contract;
        sensor_contract.id="nodus.shader.spectral.sensor_raygen";
        sensor_contract.program_id=0x82a58c07ed647f57ull;
        sensor_contract.spirv=std::move(sensor_shader);
        for(const auto& row:sensor_inputs)
            sensor_contract.minimum_binding_bytes.push_back(row.size()*4);
        sensor_contract.writable={0,0,0,0,1,1,1,0,0};
        sensor_contract.local_size_x=64;
        sensor_contract.maximum_invocations=1;
        if(!nodus::gpu::register_gpu_shader_job(tool_registry_global(),*library,
                                                 sensor_contract))return 22;
        for(std::uint32_t mode=0;mode<2;mode++){
            runtime[29]=mode; // target_mode: sensor -> target, then reciprocal
            {
                auto runtime_loan=library->borrow_cpu(kCard,sensor_base+8,true);
                if(!runtime_loan||runtime_loan.bytes()!=runtime.size()*4)return 23;
                std::memcpy(runtime_loan.data(),runtime.data(),runtime.size()*4);
            }
        for(std::size_t device=0;device<library->device_count();device++){
            {
                auto counter_loan=library->borrow_cpu(kCard,sensor_base+4,true);
                auto intent_loan=library->borrow_cpu(kCard,sensor_base+5,true);
                auto weight_loan=library->borrow_cpu(kCard,sensor_base+6,true);
                if(!counter_loan||!intent_loan||!weight_loan)return 23;
                std::memcpy(counter_loan.data(),counters.data(),counters.size()*4);
                std::memcpy(intent_loan.data(),intent.data(),intent.size()*4);
                *static_cast<std::uint32_t*>(weight_loan.data())=0;
            }
            auto row=tool_registry_global().create(sensor_contract.id+".vulkan."+
                                                   std::to_string(device));
            if(!row)return 24;
            const std::uint64_t shape[1]={4};
            auto work=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
            const std::uint64_t command[4]={kCard,sensor_base,1,device+1};
            nodus_tensor_write(work,0,command,sizeof(command));
            RawStackFrame frame{};
            if(!raw_stack_init_frame(frame,4096))return 25;
            raw_stack_push_abstract_tensor(frame,{work});
            ToolStackContext context{};context.stack.raw=&frame;
            row->execute_stack(context);
            nodus::tensors::AbstractTensorHandle result{};
            std::array<std::uint64_t,9> receipt{};
            if(!context.succeeded||!raw_stack_pop_abstract_tensor(frame,result)||
               nodus_tensor_read(result.id,0,receipt.data(),sizeof(receipt))!=
                   sizeof(receipt)||receipt[0]!=0)return 26;
            nodus_tensor_destroy(result.id);nodus_tensor_destroy(work);
            raw_stack_free_mask(frame);
            if(frame.backend)gp_mem_backend_release(frame.backend);
            auto output=library->borrow_cpu(kCard,sensor_base+5,false);
            auto count=library->borrow_cpu(kCard,sensor_base+4,false);
            if(!output||!count)return 27;
            const auto* values=static_cast<const std::uint32_t*>(output.data());
            const auto* counts=static_cast<const std::uint32_t*>(count.data());
            std::cout<<"raygen_device="<<device<<" mode="<<mode
                     <<" origin="<<f(values[0])<<','
                     <<f(values[1])<<','<<f(values[2])<<" direction="
                     <<f(values[3])<<','<<f(values[4])<<','<<f(values[5])
                     <<" count="<<counts[8]<<"\n";
            const float expected_x=mode?-1.0f:1.0f;
            if(counts[8]!=1||std::abs(f(values[3])-expected_x)>0.001f||
               std::abs(f(values[4]))>0.001f||
               std::abs(f(values[5]))>0.001f)return 28;
        }
        }
    }
    if(argc>=5){
        std::ifstream t1_source(argv[4],std::ios::binary);
        std::vector<char> t1_bytes{std::istreambuf_iterator<char>(t1_source),{}};
        if(t1_bytes.empty()||t1_bytes.size()%4)return 29;
        std::vector<std::uint32_t> t1_shader(t1_bytes.size()/4);
        std::memcpy(t1_shader.data(),t1_bytes.data(),t1_bytes.size());
        constexpr std::uint64_t coupled_base=kBase+2000;
        std::array<std::vector<std::uint32_t>,29> state;
        state[0].resize(86); // raygen output / T1 RayIntent input
        state[1].resize(93); // T1 HitRecord output / T2 input
        state[2].resize(24); // shared stage counters
        const auto fb=[](float value){return std::bit_cast<std::uint32_t>(value);};
        state[3]={fb(-.051f),fb(-.01f),fb(-.01f),
                  fb(-.049f),fb(.01f),fb(.01f),
                  static_cast<std::uint32_t>(-1),
                  static_cast<std::uint32_t>(-1),0,1}; // one-leaf BVH
        state[4]={0}; // triangle ID
        state[5]={fb(-.05f),fb(-.01f),fb(-.01f),
                  fb(0),fb(.02f),fb(0),
                  fb(0),fb(0),fb(.02f),
                  fb(1),fb(0),fb(0),
                  0,0,static_cast<std::uint32_t>(-1),
                  static_cast<std::uint32_t>(-1)};
        state[6].resize(32*12);state[7].resize(2);
        state[8]={1,1,1,0,1,0,0,0}; // T1 controls
        state[9]={0};state[10]={5};
        state[11].resize(16);state[12]=payload;state[13].resize(28);
        state[14].resize(37);state[14][0]=1;state[14][1]=1;
        state[14][2]=1;
        state[14][3]=fb(299792458.0f/(0.486e-6f));
        state[15].resize(16);state[15][12]=fb(1);
        state[16].resize(8);state[16][2]=state[16][3]=fb(.5f);
        state[16][6]=fb(1);
        state[17].resize(8);state[17][7]=1;
        state[18]={1};state[19]={0};state[20]={0};
        state[21].resize(38);
        state[21][0]=fb(-.04f); // sensor after the last lens surface
        state[21][5]=state[21][10]=fb(1);
        state[21][11]=state[21][12]=fb(.01f);
        state[21][16]=fb(-.08f); // target beyond the first surface
        state[21][28]=1;state[21][30]=1;state[21][31]=1;
        state[21][32]=fb(.001f);state[21][33]=fb(1);
        if(argc==6){
            state[22].resize(86+94); // T3 child intent followed by terminal
            state[23].resize(24);state[23][6]=1; // one refined hit
            state[24].resize(32*12); // one 32-band material
            state[25].resize(1);state[26].resize(1);state[27].resize(1);
            state[28].resize(17);
            state[28][0]=1;state[28][1]=1;state[28][2]=1;
            state[28][3]=1;state[28][14]=3; // BDPT disabled, hit-count slot 6
        }
        for(std::size_t i=0;i<state.size();i++)
            if(!state[i].empty()&&!put(*library_tool,coupled_base+i,state[i]))return 30;
        auto raygen_contract=nodus::gpu::GpuShaderJobContract{};
        raygen_contract.id="nodus.shader.spectral.coupled_raygen";
        raygen_contract.program_id=0x82a58c07ed647f57ull;
        {
            std::ifstream f(argv[3],std::ios::binary);
            std::vector<char> data{std::istreambuf_iterator<char>(f),{}};
            raygen_contract.spirv.resize(data.size()/4);
            std::memcpy(raygen_contract.spirv.data(),data.data(),data.size());
        }
        raygen_contract.binding_offsets={15,16,17,18,2,0,19,20,21};
        raygen_contract.writable={0,0,0,0,1,1,1,0,0};
        raygen_contract.local_size_x=64;raygen_contract.maximum_invocations=1;
        for(auto offset:raygen_contract.binding_offsets)
            raygen_contract.minimum_binding_bytes.push_back(state[offset].size()*4);
        auto t1_contract=nodus::gpu::GpuShaderJobContract{};
        t1_contract.id="nodus.shader.spectral.coupled_t1";
        t1_contract.program_id=0xc4c823b095706d0cull;
        t1_contract.spirv=std::move(t1_shader);
        t1_contract.binding_offsets={0,1,2,3,4,5,6,7,8};
        t1_contract.writable={0,1,1,0,0,0,0,0,0};
        t1_contract.local_size_x=64;t1_contract.maximum_invocations=1;
        for(auto offset:t1_contract.binding_offsets)
            t1_contract.minimum_binding_bytes.push_back(state[offset].size()*4);
        auto t2_contract=contract;
        t2_contract.id="nodus.shader.spectral.coupled_t2";
        t2_contract.binding_offsets={1,5,9,10,11,2,12,13,14};
        t2_contract.minimum_binding_bytes.clear();
        for(auto offset:t2_contract.binding_offsets)
            t2_contract.minimum_binding_bytes.push_back(state[offset].size()*4);
        auto t3_contract=nodus::gpu::GpuShaderJobContract{};
        if(argc==6){
            std::ifstream f(argv[5],std::ios::binary);
            std::vector<char> data{std::istreambuf_iterator<char>(f),{}};
            if(data.empty()||data.size()%4)return 38;
            t3_contract.id="nodus.shader.spectral.coupled_t3";
            t3_contract.program_id=0x443cf187a1852b46ull;
            t3_contract.spirv.resize(data.size()/4);
            std::memcpy(t3_contract.spirv.data(),data.data(),data.size());
            t3_contract.binding_offsets={1,22,23,5,24,25,26,27,28};
            t3_contract.writable={0,1,1,0,0,0,1,1,0};
            t3_contract.local_size_x=64;t3_contract.maximum_invocations=1;
            for(auto offset:t3_contract.binding_offsets)
                t3_contract.minimum_binding_bytes.push_back(state[offset].size()*4);
        }
        for(const auto* item:{&raygen_contract,&t1_contract,&t2_contract})
            if(!nodus::gpu::register_gpu_shader_job(tool_registry_global(),
                                                    *library,*item))return 31;
        if(argc==6&&!nodus::gpu::register_gpu_shader_job(tool_registry_global(),
                                                        *library,t3_contract))return 39;
        const std::string id="nodus.pipeline.spectral.sensor_t1_conic_t2";
        const auto suffix=".vulkan.0";
        std::vector<nodus::gpu::KernelPipelineComponent> components{
            {raygen_contract.id+suffix,Api::Vulkan,raygen_contract.program_id},
            {t1_contract.id+suffix,Api::Vulkan,t1_contract.program_id},
            {t2_contract.id+suffix,Api::Vulkan,t2_contract.program_id}};
        if(argc==6)components.push_back(
            {t3_contract.id+suffix,Api::Vulkan,t3_contract.program_id});
        if(!nodus::gpu::register_kernel_pipeline(tool_registry_global(),id,
                                                 components))return 32;
        auto coupled=tool_registry_global().create(id);
        if(!coupled)return 33;
        coupled->initialize({});
        const std::uint64_t shape[2]={static_cast<std::uint64_t>(components.size()),7};
        auto work=nodus_tensor_create(NODUS_DTYPE_U64,2,shape);
        const std::uint64_t rows[28]={
            0,1,raygen_contract.program_id,kCard,coupled_base,1,1,
            1,1,t1_contract.program_id,kCard,coupled_base,1,2,
            2,1,t2_contract.program_id,kCard,coupled_base,1,3,
            3,1,t3_contract.program_id,kCard,coupled_base,1,4};
        nodus_tensor_write(work,0,rows,components.size()*7*sizeof(std::uint64_t));
        RawStackFrame frame{};
        if(!raw_stack_init_frame(frame,4096))return 34;
        raw_stack_push_abstract_tensor(frame,{work});
        ToolStackContext context{};context.stack.raw=&frame;
        coupled->execute_stack(context);
        nodus::tensors::AbstractTensorHandle result{};
        std::array<std::uint64_t,45> receipt{};
        if(!context.succeeded||!raw_stack_pop_abstract_tensor(frame,result)||
           nodus_tensor_read(result.id,0,receipt.data(),
                             (components.size()+1)*9*sizeof(std::uint64_t))!=
               (components.size()+1)*9*sizeof(std::uint64_t))return 35;
        {
        auto hit_loan=library->borrow_cpu(kCard,coupled_base+1,false);
        auto count_loan=library->borrow_cpu(kCard,coupled_base+2,false);
        if(!hit_loan||!count_loan)return 36;
        const auto* h=static_cast<const std::uint32_t*>(hit_loan.data());
        const auto* c=static_cast<const std::uint32_t*>(count_loan.data());
        std::cout<<"coupled_status="<<receipt[0]<<" completed="<<receipt[3]
                 <<" t1_hits="<<c[0]<<" lens_exit_x="<<f(h[0])
                 <<" lens_flags="<<h[16]<<"\n";
        if(receipt[0]!=0||receipt[3]!=components.size()||c[0]!=1||
           std::abs(f(h[0])-first)>0.002f||
           (h[16]&4u)==0||(h[16]&8u)!=0)return 37;
        if(argc==6){
            auto meta=library->borrow_cpu(kCard,coupled_base+23,false);
            auto output=library->borrow_cpu(kCard,coupled_base+22,false);
            if(!meta||!output)return 40;
            const auto* m=static_cast<const std::uint32_t*>(meta.data());
            const auto* o=static_cast<const std::uint32_t*>(output.data());
            std::cout<<"t3_child_count="<<m[0]<<" terminal_count="<<m[1]
                     <<" child_x="<<f(o[0])<<"\n";
            if(m[0]!=1||m[1]!=0||std::abs(f(o[0])-f(h[0]))>.002f)return 41;
        }
        }
        if(argc==6){
            {
                auto hit=library->borrow_cpu(kCard,coupled_base+1,true);
                auto meta=library->borrow_cpu(kCard,coupled_base+23,true);
                auto output=library->borrow_cpu(kCard,coupled_base+22,true);
                if(!hit||!meta||!output)return 42;
                auto* h=static_cast<std::uint32_t*>(hit.data());
                auto* m=static_cast<std::uint32_t*>(meta.data());
                h[16]&=~4u; // ordinary material branch instead of conic continuation
                h[18]=0;    // authored bounce budget is exhausted
                m[0]=m[1]=0;m[6]=1;
                std::memset(output.data(),0,output.bytes());
            }
            for(std::size_t device=0;device<library->device_count();device++){
            if(device){
                auto meta=library->borrow_cpu(kCard,coupled_base+23,true);
                auto output=library->borrow_cpu(kCard,coupled_base+22,true);
                if(!meta||!output)return 48;
                auto* m=static_cast<std::uint32_t*>(meta.data());
                m[0]=m[1]=0;m[6]=1;
                std::memset(output.data(),0,output.bytes());
            }
            auto row=tool_registry_global().create(t3_contract.id+".vulkan."+
                                                    std::to_string(device));
            if(!row)return 43;
            const std::uint64_t token_shape[1]={4};
            auto token=nodus_tensor_create(NODUS_DTYPE_U64,1,token_shape);
            const std::uint64_t token_words[4]={kCard,coupled_base,1,5+device};
            nodus_tensor_write(token,0,token_words,sizeof(token_words));
            RawStackFrame terminal_frame{};
            if(!raw_stack_init_frame(terminal_frame,4096))return 44;
            raw_stack_push_abstract_tensor(terminal_frame,{token});
            ToolStackContext terminal_context{};
            terminal_context.stack.raw=&terminal_frame;
            row->execute_stack(terminal_context);
            nodus::tensors::AbstractTensorHandle terminal_receipt{};
            std::array<std::uint64_t,9> receipt_words{};
            if(!terminal_context.succeeded||
               !raw_stack_pop_abstract_tensor(terminal_frame,terminal_receipt)||
               nodus_tensor_read(terminal_receipt.id,0,receipt_words.data(),
                                 sizeof(receipt_words))!=sizeof(receipt_words)||
               receipt_words[0]!=0)return 45;
            {
                auto meta=library->borrow_cpu(kCard,coupled_base+23,false);
                auto output=library->borrow_cpu(kCard,coupled_base+22,false);
                if(!meta||!output)return 46;
                const auto* m=static_cast<const std::uint32_t*>(meta.data());
                const auto* o=static_cast<const std::uint32_t*>(output.data());
                std::cout<<"t3_device="<<device<<" terminal_count="<<m[1]
                         <<" terminal_x="<<f(o[86])<<"\n";
                if(m[0]!=0||m[1]!=1||std::abs(f(o[86])-first)>.002f)return 47;
            }
            nodus_tensor_destroy(terminal_receipt.id);
            nodus_tensor_destroy(token);
            raw_stack_free_mask(terminal_frame);
            if(terminal_frame.backend)gp_mem_backend_release(terminal_frame.backend);
            }
        }
        nodus_tensor_destroy(result.id);nodus_tensor_destroy(work);
        raw_stack_free_mask(frame);
        if(frame.backend)gp_mem_backend_release(frame.backend);
        coupled->shutdown();
    }
    nodus_tensor_destroy(dummy);
    library_tool->shutdown();
    return 0;
}
