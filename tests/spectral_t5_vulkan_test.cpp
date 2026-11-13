#include "tensor_state_library_tool.h"
#include "gpu_shader_job_tool.h"
#include "kernel_pipeline_tool.h"
#include "tool_registry.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <vector>

namespace {
constexpr std::uint64_t card_id=501,base=31000;
bool call_library(ITool& tool,const std::array<std::uint64_t,12>& words,
                  std::uint64_t payload){
    const std::uint64_t shape[1]={words.size()};
    auto request=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    if(!request||nodus_tensor_write(request,0,words.data(),sizeof(words))!=sizeof(words))
        return false;
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
bool put(ITool& tool,std::uint64_t address,const std::vector<std::uint32_t>& values){
    const std::uint64_t shape[1]={values.size()};
    auto tensor=nodus_tensor_create(NODUS_DTYPE_U32,1,shape);
    if(!tensor||nodus_tensor_write(tensor,0,values.data(),values.size()*4)!=
        values.size()*4)return false;
    std::array<std::uint64_t,12> command{};
    command[0]=2;command[1]=card_id;command[3]=address;
    const bool okay=call_library(tool,command,tensor);
    nodus_tensor_destroy(tensor);
    return okay;
}
}

int main(int argc,char** argv){
    const bool generated_records=argc==7;
    if(argc!=6&&!generated_records)return 2;
    std::array<std::vector<std::uint32_t>,5> programs;
    for(int i=0;i<5;i++){
        std::ifstream source(argv[i+1],std::ios::binary);
        std::vector<char> bytes{std::istreambuf_iterator<char>(source),{}};
        if(bytes.empty()||bytes.size()%4)return 2;
        programs[i].resize(bytes.size()/4);
        std::memcpy(programs[i].data(),bytes.data(),bytes.size());
    }
    std::vector<std::uint32_t> t3_program;
    if(generated_records){
        std::ifstream source(argv[6],std::ios::binary);
        std::vector<char> bytes{std::istreambuf_iterator<char>(source),{}};
        if(bytes.empty()||bytes.size()%4)return 2;
        t3_program.resize(bytes.size()/4);
        std::memcpy(t3_program.data(),bytes.data(),bytes.size());
    }
    auto tool=tool_registry_global().create("nodus.tensor.state.library");
    if(!tool)return 3;
    tool->initialize({});
    auto* library=dynamic_cast<nodus::gpu::TensorStateLibraryTool*>(tool.get());
    if(!library||library->device_count()<3)return 3;
    std::array<std::uint64_t,12> card{};
    card[0]=1;card[1]=card_id;card[2]=64;card[4]=1<<20;
    card[5]=card[6]=card[7]=1<<20;
    const std::uint64_t dummy_shape[1]={1};
    auto dummy=nodus_tensor_create(NODUS_DTYPE_U64,1,dummy_shape);
    if(!call_library(*tool,card,dummy))return 4;
    std::array<std::vector<std::uint32_t>,11> buffers;
    buffers[0].resize(32*57);buffers[1].resize(32*72);
    buffers[2].resize(3);buffers[3].resize(17);buffers[4].resize(3);
    buffers[5].resize(10);buffers[6].resize(1);buffers[7].resize(16);
    buffers[8].resize(32*12);buffers[9].resize(128);buffers[10].resize(3);
    auto fb=[](float v){return std::bit_cast<std::uint32_t>(v);};
    buffers[3][0]=fb(1e-8f);buffers[3][1]=buffers[3][2]=fb(.5f);
    buffers[3][4]=32;buffers[3][5]=32;buffers[3][6]=1;
    buffers[3][7]=32;buffers[3][9]=1;
    buffers[3][14]=1;buffers[3][15]=fb(1.f);
    buffers[4]={fb(1.f),fb(1.f),fb(1.f)};
    buffers[10]={0,1,1};
    // Authored pack/scatter shaders consume two sorted one-vertex path sets.
    // They, rather than this host fixture, construct T5's LGV/CGV rows.
    std::vector<std::uint32_t> raw(64*28+64*8+64*12),keys(64*2),idx(64);
    for(std::size_t i=0;i<64&&!generated_records;i++){
        const bool camera=(i&1u)==0u;
        const auto sid=0x01000000u+
            static_cast<std::uint32_t>(camera?(i/2)+101:(i/2)+1);
        const auto flags=camera?((1u<<22)|(1u<<18)):(1u<<21);
        const auto vb=i*28;
        raw[vb+0]=sid;raw[vb+1]=camera?(1u<<8):0u;
        raw[vb+2]=camera?0u:1u;
        raw[vb+4]=0;raw[vb+6]=std::bit_cast<std::uint32_t>(-1);
        raw[vb+7]=fb(camera?0.f:1.f);
        raw[vb+10]=fb(camera?1.f:-1.f);
        raw[vb+21]=raw[vb+22]=fb(1.f/3.14159265358979323846f);
        // The pack fallback is intentionally dim and has no light PDF flag.
        // Spectral and PDF scatter must replace both for T5 to contribute.
        raw[vb+23]=0;raw[vb+25]=fb(.25f);
        const auto sb=64*28+i*8;
        raw[sb+0]=sid;raw[sb+2]=fb(1.f);raw[sb+5]=fb(1.f);
        const auto pb=64*28+64*8+i*12;
        raw[pb+0]=sid;raw[pb+2]=raw[pb+3]=raw[pb+5]=
            fb(1.f/3.14159265358979323846f);
        raw[pb+8]=flags;
    }
    std::vector<std::uint32_t> t3_hits,t3_out,t3_meta,t3_tri,t3_mat,
        t3_uv,t3_uv_accum,t3_control;
    if(generated_records){
        t3_hits.resize(64*93);t3_out.resize(64*92*2);
        t3_meta.resize(24);t3_tri.resize(32);t3_mat.resize(32*12);
        t3_uv.resize(1);t3_uv_accum.resize(1);t3_control.resize(17);
        t3_meta[6]=64;
        t3_tri[9]=std::bit_cast<std::uint32_t>(-1.f);
        t3_tri[12]=1;t3_tri[13]=0;
        t3_tri[16+9]=fb(1.f);t3_tri[16+13]=0;
        t3_mat[2]=fb(1.f);t3_mat[4]=fb(1.f);
        t3_mat[5]=fb(1.f);t3_mat[7]=fb(1.f);
        t3_control[0]=1;t3_control[1]=1;t3_control[2]=64;
        t3_control[3]=1;t3_control[11]=64;
        t3_control[12]=64;t3_control[13]=64;
        t3_control[14]=3;t3_control[15]=64*28;
        t3_control[16]=64*28+64*8;
        for(std::size_t i=0;i<64;i++){
            const bool camera=(i&1u)==0u;
            const auto h=i*93;
            t3_hits[h+0]=fb(camera?0.f:1.f);
            t3_hits[h+3]=fb(camera?1.f:-1.f);
            t3_hits[h+6]=fb(camera?-1.f:1.f);
            t3_hits[h+14]=camera?1u:0u;
            t3_hits[h+15]=0;
            t3_hits[h+16]=camera?1u:0u;
            t3_hits[h+18]=0;
            t3_hits[h+27]=fb(1.f);t3_hits[h+28]=fb(1.f);
            t3_hits[h+92]=0x01000000u+
                static_cast<std::uint32_t>(camera?(i/2)+101:(i/2)+1);
        }
    }
    std::vector<std::uint32_t> pack_control={64,0,32,1,fb(.5f),fb(.5f)};
    std::vector<std::uint32_t> spectral_control={0,64,32,64,0,1,64*28,
                                                64*28+64*8,0};
    auto pdf_control=spectral_control;pdf_control[0]=2;
    std::vector<std::uint32_t> sort_counts(2),sort_control={64,64,0};
    std::vector<std::vector<std::uint32_t>> sort_steps;
    for(std::uint32_t k=2;k<=64;k*=2)
        for(std::uint32_t j=k/2;j>0;j/=2)
            sort_steps.push_back({k,j,64,0});
    buffers[5][0]=fb(100.f);buffers[5][3]=fb(101.f);
    buffers[5][6]=std::bit_cast<std::uint32_t>(-1);
    // Full connection, MIS, shadow, reduction and pixel splat.
    nodus::gpu::GpuShaderJobContract contract;
    contract.id="nodus.shader.spectral.t5";
    contract.program_id=0xe61fe2ab3b0d3644ull;
    contract.spirv=programs[0];
    for(const auto& buffer:buffers)
        contract.minimum_binding_bytes.push_back(buffer.size()*4);
    contract.writable={0,0,1,0,0,0,0,0,0,1,0};
    contract.local_size_x=16;contract.local_size_y=16;
    contract.maximum_invocations=32;contract.maximum_invocations_y=32;
    if(!nodus::gpu::register_gpu_shader_job(tool_registry_global(),*library,contract))
        return 6;
    nodus::gpu::GpuShaderJobContract t3;
    if(generated_records){
        t3.id="nodus.shader.spectral.t3.producer";
        t3.program_id=0x743370726f647563ull;
        t3.spirv=t3_program;
        t3.binding_offsets={60,61,62,63,64,65,66,20,67};
        t3.minimum_binding_bytes={t3_hits.size()*4,t3_out.size()*4,
            t3_meta.size()*4,t3_tri.size()*4,t3_mat.size()*4,
            t3_uv.size()*4,t3_uv_accum.size()*4,raw.size()*4,
            t3_control.size()*4};
        t3.writable={0,1,1,0,0,0,1,1,0};
        t3.local_size_x=64;t3.maximum_invocations=64;
        if(!nodus::gpu::register_gpu_shader_job(tool_registry_global(),*library,t3))
            return 29;
    }
    nodus::gpu::GpuShaderJobContract sort;
    sort.id="nodus.shader.spectral.t5.sort.init";
    sort.program_id=0x736f727474350001ull;sort.spirv=programs[3];
    sort.binding_offsets={20,21,22,26,27};
    sort.minimum_binding_bytes={raw.size()*4,keys.size()*4,idx.size()*4,
        sort_counts.size()*4,sort_control.size()*4};
    sort.writable={0,1,1,1,0};
    sort.local_size_x=64;sort.maximum_invocations=64;
    if(!nodus::gpu::register_gpu_shader_job(tool_registry_global(),*library,sort))
        return 22;
    nodus::gpu::GpuShaderJobContract step;
    step.program_id=0x736f727474350002ull;step.spirv=programs[4];
    step.minimum_binding_bytes={keys.size()*4,idx.size()*4,4*4};
    step.writable={1,1,0};
    step.local_size_x=64;step.maximum_invocations=32;
    std::vector<std::string> step_ids;
    for(std::size_t i=0;i<sort_steps.size();i++){
        step.id="nodus.shader.spectral.t5.sort.step."+std::to_string(i);
        step.binding_offsets={21,22,30+i};
        if(!nodus::gpu::register_gpu_shader_job(tool_registry_global(),*library,step))
            return 25;
        step_ids.push_back(step.id);
    }
    nodus::gpu::GpuShaderJobContract pack;
    pack.id="nodus.shader.spectral.t5.pack";pack.program_id=0x7061636b74350001ull;
    pack.spirv=programs[1];
    pack.binding_offsets={21,22,20,0,1,8,23};
    pack.minimum_binding_bytes={keys.size()*4,idx.size()*4,raw.size()*4,
        buffers[0].size()*4,buffers[1].size()*4,buffers[8].size()*4,
        pack_control.size()*4};
    pack.writable={0,0,0,1,1,0,0};
    pack.local_size_x=64;pack.maximum_invocations=64;
    if(!nodus::gpu::register_gpu_shader_job(tool_registry_global(),*library,pack))
        return 17;
    auto scatter=pack;
    scatter.id="nodus.shader.spectral.t5.scatter.spectral";
    scatter.program_id=0x7363617474350001ull;
    scatter.spirv=programs[2];
    scatter.binding_offsets={21,20,0,1,24};
    scatter.minimum_binding_bytes={keys.size()*4,raw.size()*4,
        buffers[0].size()*4,buffers[1].size()*4,spectral_control.size()*4};
    scatter.writable={0,0,1,1,0};
    if(!nodus::gpu::register_gpu_shader_job(tool_registry_global(),*library,scatter))
        return 18;
    scatter.id="nodus.shader.spectral.t5.scatter.pdf";
    scatter.program_id=0x7363617474350002ull;
    scatter.binding_offsets[4]=25;
    if(!nodus::gpu::register_gpu_shader_job(tool_registry_global(),*library,scatter))
        return 19;
    std::vector<std::vector<nodus::gpu::KernelPipelineComponent>>
        device_components(3);
    for(std::size_t device=0;device<3;device++){
        const std::string pipeline_id="nodus.pipeline.spectral.t5."+
                                      std::to_string(device);
        const auto vk=nodus::gpu::TensorStateLibraryTool::ExecutionApi::Vulkan;
        const auto suffix=".vulkan."+std::to_string(device);
        std::vector<nodus::gpu::KernelPipelineComponent> components;
        if(generated_records)
            components.push_back({t3.id+suffix,vk,t3.program_id});
        components.push_back({sort.id+suffix,vk,sort.program_id});
        for(const auto& id:step_ids)
            components.push_back({id+suffix,vk,step.program_id});
        components.push_back({pack.id+suffix,vk,pack.program_id});
        components.push_back({"nodus.shader.spectral.t5.scatter.spectral"+suffix,
                              vk,0x7363617474350001ull});
        components.push_back({"nodus.shader.spectral.t5.scatter.pdf"+suffix,
                              vk,0x7363617474350002ull});
        components.push_back({contract.id+suffix,vk,contract.program_id});
        if(!nodus::gpu::register_kernel_pipeline(tool_registry_global(),pipeline_id,
                                                 components))return 7;
        device_components[device]=std::move(components);
    }
    const auto run_device=[&](std::size_t device,std::size_t job_id)->int{
        const std::uint64_t base=31000+job_id*1000;
        for(std::size_t i=0;i<buffers.size();i++)
            if(!put(*tool,base+i,buffers[i]))return 5;
        if(!put(*tool,base+20,raw)||!put(*tool,base+21,keys)||
           !put(*tool,base+22,idx)||!put(*tool,base+23,pack_control)||
           !put(*tool,base+24,spectral_control)||
           !put(*tool,base+25,pdf_control)||
           !put(*tool,base+26,sort_counts)||
           !put(*tool,base+27,sort_control))return 5;
        for(std::size_t i=0;i<sort_steps.size();i++)
            if(!put(*tool,base+30+i,sort_steps[i]))return 5;
        if(generated_records&&
           (!put(*tool,base+60,t3_hits)||!put(*tool,base+61,t3_out)||
            !put(*tool,base+62,t3_meta)||!put(*tool,base+63,t3_tri)||
            !put(*tool,base+64,t3_mat)||!put(*tool,base+65,t3_uv)||
            !put(*tool,base+66,t3_uv_accum)||
            !put(*tool,base+67,t3_control)))return 5;
        const auto pipeline_id="nodus.pipeline.spectral.t5."+
                               std::to_string(device);
        const auto& components=device_components[device];
        auto pipeline=tool_registry_global().create(pipeline_id);
        if(!pipeline)return 8;
        const std::uint64_t shape[2]={components.size(),8};
        auto request=nodus_tensor_create(NODUS_DTYPE_U64,2,shape);
        std::vector<std::uint64_t> rows(components.size()*8);
        for(std::size_t i=0;i<components.size();i++){
            const bool connector=i+1==components.size();
            const bool sort_step=i>(generated_records?1u:0u)&&
                i<=(generated_records?1u:0u)+sort_steps.size();
            const std::uint64_t row[8]={i,1,components[i].program_id,card_id,
                base,connector||sort_step?32ull:64ull,device+1,
                connector?32ull:0ull};
            std::copy_n(row,8,rows.begin()+i*8);
        }
        nodus_tensor_write(request,0,rows.data(),rows.size()*8);
        RawStackFrame frame{};
        if(!raw_stack_init_frame(frame,4096))return 9;
        raw_stack_push_abstract_tensor(frame,{request});
        ToolStackContext context{};context.stack.raw=&frame;
        pipeline->execute_stack(context);
        nodus::tensors::AbstractTensorHandle result{};
        std::vector<std::uint64_t> receipt((components.size()+1)*9);
        if(!context.succeeded||!raw_stack_pop_abstract_tensor(frame,result)||
           nodus_tensor_read(result.id,0,receipt.data(),receipt.size()*8)!=
               receipt.size()*8||receipt[0]!=0||
           receipt[3]!=components.size())return 10;
        auto debug=library->borrow_cpu(card_id,base+9,false);
        if(!debug)return 11;
        const auto count=static_cast<const std::uint32_t*>(debug.data())[0];
        auto pixels=library->borrow_cpu(card_id,base+2,false);
        if(!pixels)return 14;
        const auto* pixel_words=static_cast<const std::uint32_t*>(pixels.data());
        const float red=std::bit_cast<float>(pixel_words[0]);
        const float green=std::bit_cast<float>(pixel_words[1]);
        const float blue=std::bit_cast<float>(pixel_words[2]);
        std::cout<<"t5_device="<<device<<" job="<<job_id<<" pairs="<<count
                 <<" stages="<<receipt[3]<<" rgb="<<red<<","<<green
                 <<","<<blue;
        auto light=library->borrow_cpu(card_id,base+0,false);
        auto camera=library->borrow_cpu(card_id,base+1,false);
        if(!light||!camera)return 20;
        auto counts=library->borrow_cpu(card_id,base+26,false);
        if(!counts)return 23;
        auto sorted_keys=library->borrow_cpu(card_id,base+21,false);
        auto sorted_idx=library->borrow_cpu(card_id,base+22,false);
        if(!sorted_keys||!sorted_idx)return 26;
        const auto* kp=static_cast<const std::uint32_t*>(sorted_keys.data());
        const auto* ip=static_cast<const std::uint32_t*>(sorted_idx.data());
        if(!generated_records&&(ip[0]!=1||ip[32]!=0))return 27;
        for(std::size_t i=1;i<64;i++)
            if(kp[2*i-2]>kp[2*i]||
               (kp[2*i-2]==kp[2*i]&&kp[2*i-1]>kp[2*i+1]))return 28;
        const auto* lp=static_cast<const std::uint32_t*>(light.data());
        const auto* cp=static_cast<const std::uint32_t*>(camera.data());
        std::cout<<" light_beta="<<std::bit_cast<float>(lp[16])
                 <<" camera_beta="<<std::bit_cast<float>(cp[22])
                 <<" light_flags="<<lp[13]<<" camera_flags="<<cp[17];
        for(std::size_t i=1;i<=12;i++)
            std::cout<<" debug"<<i<<"="<<static_cast<const std::uint32_t*>(debug.data())[i];
        std::cout<<'\n';
        if(count!=1024)return 12;
        if(generated_records){
            auto produced=library->borrow_cpu(card_id,base+62,false);
            if(!produced)return 30;
            const auto* m=static_cast<const std::uint32_t*>(produced.data());
            std::cout<<"t3_job="<<job_id<<" produced="<<m[3]<<","<<m[4]<<","<<m[5]
                     <<" sorted="<<static_cast<const std::uint32_t*>(counts.data())[0]
                     <<","<<static_cast<const std::uint32_t*>(counts.data())[1]<<'\n';
            // Emissive light terminals do not emit a separate PDF record;
            // their discrete endpoint PDF is already carried by the vertex.
            if(m[3]!=64||m[4]!=64||m[5]!=32||
               static_cast<const std::uint32_t*>(counts.data())[0]!=32||
               static_cast<const std::uint32_t*>(counts.data())[1]!=32)
                return 31;
        }
        if(!std::isfinite(red)||!std::isfinite(green)||!std::isfinite(blue)||
           std::abs(red-1024.f)>0.1f||std::abs(green-1024.f)>0.1f||
           std::abs(blue-1024.f)>0.1f||
           std::abs(std::bit_cast<float>(lp[16])-1.f)>1e-5f||
           std::abs(std::bit_cast<float>(cp[22])-1.f)>1e-5f||
           (!generated_records&&
            (lp[13]!=(1u<<21)||cp[17]!=((1u<<22)|(1u<<18))))||
           static_cast<const std::uint32_t*>(counts.data())[0]!=32||
           static_cast<const std::uint32_t*>(counts.data())[1]!=32||
           (!generated_records&&
            static_cast<const std::uint32_t*>(debug.data())[5]!=1024)||
           static_cast<const std::uint32_t*>(debug.data())[8]!=1024)
            return 15;
        nodus_tensor_destroy(result.id);nodus_tensor_destroy(request);
        raw_stack_free_mask(frame);
        if(frame.backend)gp_mem_backend_release(frame.backend);
        pipeline->shutdown();
        // The next device reads the same library state with a fresh counter.
        debug={};
        auto clear=library->borrow_cpu(card_id,base+9,true);
        if(!clear)return 13;
        std::memset(clear.data(),0,128*sizeof(std::uint32_t));
        pixels={};
        auto clear_pixels=library->borrow_cpu(card_id,base+2,true);
        if(!clear_pixels)return 16;
        std::memset(clear_pixels.data(),0,3*sizeof(std::uint32_t));
        counts={};
        auto clear_counts=library->borrow_cpu(card_id,base+26,true);
        if(!clear_counts)return 24;
        std::memset(clear_counts.data(),0,2*sizeof(std::uint32_t));
        if(generated_records){
            auto clear_meta=library->borrow_cpu(card_id,base+62,true);
            auto clear_raw=library->borrow_cpu(card_id,base+20,true);
            if(!clear_meta||!clear_raw)return 32;
            std::memset(static_cast<std::uint32_t*>(clear_meta.data())+3,0,
                        3*sizeof(std::uint32_t));
            std::memset(clear_raw.data(),0,raw.size()*sizeof(std::uint32_t));
        }
        return 0;
    };
    if(generated_records){
        std::array<std::future<int>,6> jobs;
        for(std::size_t job_id=0;job_id<jobs.size();job_id++)
            jobs[job_id]=std::async(std::launch::async,run_device,
                                    job_id%3,job_id);
        for(auto& job:jobs){
            const int result=job.get();
            if(result)return result;
        }
    }else{
        for(std::size_t device=0;device<3;device++){
            const int result=run_device(device,device);
            if(result)return result;
        }
    }
    nodus_tensor_destroy(dummy);
    tool->shutdown();
    return 0;
}
