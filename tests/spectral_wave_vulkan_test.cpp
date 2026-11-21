#include "gpu_shader_job_tool.h"
#include "kernel_library_session.h"
#include "kernel_pipeline_tool.h"
#include "common/tensors/abstraction/kpath/kpath_raster.h"
#include "tensor_state_library_tool.h"
#include "tool_registry.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <vector>

namespace {
constexpr std::uint64_t card=907, base=51000;

bool request_library(ITool& tool,const std::array<std::uint64_t,12>& words,
                     std::uint64_t payload){
    const std::uint64_t shape[1]={words.size()};
    const auto request=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
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
    const auto tensor=nodus_tensor_create(NODUS_DTYPE_U32,1,shape);
    if(!tensor||nodus_tensor_write(tensor,0,values.data(),values.size()*4)!=values.size()*4)
        return false;
    std::array<std::uint64_t,12> request{};
    request[0]=2;request[1]=card;request[3]=address;
    const bool okay=request_library(tool,request,tensor);
    nodus_tensor_destroy(tensor);
    return okay;
}
}

int main(int argc,char** argv){
    const bool joined=(argc==4||argc==5)&&
        std::string(argv[1])=="angular_propagating";
    if(argc!=3&&!joined)return 2;
    const std::string kind=argv[1];
    const std::size_t feeds=kind=="angular_propagating"?7:
        kind=="angular_evanescent"?4:
        kind=="aperture_material"?8:
        kind=="jones_interface"?12:
        kind=="volume3d"?15:0;
    if(!feeds)return 2;
    const std::size_t outputs=kind=="jones_interface"?4:
        kind=="volume3d"?1:2;
    std::ifstream input(argv[2],std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(input),{}};
    if(bytes.empty()||bytes.size()%4)return 2;
    std::vector<std::uint32_t> shader(bytes.size()/4);
    std::memcpy(shader.data(),bytes.data(),bytes.size());
    std::vector<std::uint32_t> material_shader;
    if(joined){
        std::ifstream next(argv[3],std::ios::binary);
        std::vector<char> material_bytes{std::istreambuf_iterator<char>(next),{}};
        if(material_bytes.empty()||material_bytes.size()%4)return 2;
        material_shader.resize(material_bytes.size()/4);
        std::memcpy(material_shader.data(),material_bytes.data(),material_bytes.size());
    }

    nodus::gpu::KernelLibrarySession session(tool_registry_global());
    if(!session)return 3;
    auto& library_row=session.library_row();
    auto* library=&session.library();
    if(!library||library->device_count()==0)return 3;
    std::array<std::uint64_t,12> card_request{};
    card_request[0]=1;card_request[1]=card;card_request[2]=1907;
    card_request[4]=card_request[5]=card_request[6]=card_request[7]=64<<20;
    const std::uint64_t dummy_shape[1]={1};
    const auto dummy=nodus_tensor_create(NODUS_DTYPE_U64,1,dummy_shape);
    if(!request_library(library_row,card_request,dummy))return 4;

    // Binding order is each Turing product's published argument order,
    // followed by its outputs and element-count control.
    // One 512x512 field plane or a 64x64x64 volume, not a tiny shader probe.
    constexpr std::size_t lanes=512*512;
    std::vector<std::vector<std::uint32_t>> state(joined?15:feeds+outputs+1);
    const auto fb=[](float value){return std::bit_cast<std::uint32_t>(value);};
    for(std::size_t i=0;i<state.size();i++)
        if(i!=(feeds+outputs))state[i].resize(lanes);
    state[feeds+outputs]={static_cast<std::uint32_t>(lanes)};
    for(std::size_t lane=0;lane<lanes;lane++){
        const float ar=.25f+static_cast<float>(lane%17)*.03125f;
        const float ai=-.125f+static_cast<float>(lane%11)*.025f;
        if(kind=="angular_propagating"){
            // ai,ar,direction,distance,kx,ky,wavelength
            const float values[]={ai,ar,lane&1?-1.f:1.f,1e-6f,
                                  1e6f,.2e6f,550e-9f};
            for(std::size_t i=0;i<feeds;i++)state[i][lane]=fb(values[i]);
            if(joined){
                state[10][lane]=fb(1.f); // background index
                state[11][lane]=fb(.01f); // material extinction
                state[12][lane]=fb(1.5f); // material real index
            }
        }else if(kind=="angular_evanescent"){
            // ai,ar,distance,gap. Backward travel decays too.
            const float values[]={ai,ar,1e-7f,1e12f};
            for(std::size_t i=0;i<feeds;i++)state[i][lane]=fb(values[i]);
        }else if(kind=="aperture_material"){
            // ai,ar,background_index,direction,distance,
            // imaginary_index,real_index,wavelength
            const float values[]={ai,ar,1.f,lane&1?-1.f:1.f,1e-6f,
                                  .01f,1.5f,550e-9f};
            for(std::size_t i=0;i<feeds;i++)state[i][lane]=fb(values[i]);
        }else if(kind=="volume3d"){
            // Binding order: current,damping,dt,dx,dy,dz,previous,
            // source,speed,x-,x+,y-,y+,z-,z+. A stationary central
            // impulse must propagate equally to all six axial neighbors.
            constexpr std::size_t center=(32*64+32)*64+32;
            const bool central=lane==center;
            const auto value=[&](std::size_t neighbor){
                return neighbor==center?1.f:0.f;
            };
            const std::size_t x=lane/(64*64),y=(lane/64)%64,z=lane%64;
            const float values[]={
                central?1.f:0.f,0.f,.5f,1.f,1.f,1.f,
                central?1.f:0.f,0.f,.5f,
                x>0?value(lane-64*64):0.f,
                x<63?value(lane+64*64):0.f,
                y>0?value(lane-64):0.f,
                y<63?value(lane+64):0.f,
                z>0?value(lane-1):0.f,
                z<63?value(lane+1):0.f};
            for(std::size_t i=0;i<feeds;i++)state[i][lane]=fb(values[i]);
        }else{
            // a_im,a_re,b_im,b_re,c_im,c_re,d_im,d_re,pi,pr,si,sr
            state[3][lane]=fb(1); // b=1; swap S and P
            state[5][lane]=fb(1); // c=1
            state[8][lane]=fb(ai);
            state[9][lane]=fb(ar);
            state[10][lane]=fb(ai*.5f);
            state[11][lane]=fb(ar*.5f);
        }
    }
    for(std::size_t i=0;i<state.size();i++)
        if(!put(library_row,base+i,state[i]))return 5;

    nodus::gpu::GpuShaderJobContract contract;
    contract.id="nodus.shader.turing.wave."+kind;
    contract.program_id=0x776176654a6f6e65ull;
    contract.spirv=std::move(shader);
    for(std::size_t i=0;i<feeds+outputs+1;i++)
        contract.minimum_binding_bytes.push_back(state[i].size()*4);
    contract.writable.assign(feeds+outputs+1,0);
    for(std::size_t i=feeds;i<feeds+outputs;i++)contract.writable[i]=1;
    contract.local_size_x=64;contract.maximum_invocations=lanes;
    if(!session.register_shader(contract))return 6;
    nodus::gpu::GpuShaderJobContract material;
    if(joined){
        material.id="nodus.shader.turing.wave.aperture_material.joined";
        material.program_id=0x776176654a6f6e66ull;
        material.spirv=std::move(material_shader);
        // The first row's complex outputs are the second row's inputs.
        material.binding_offsets={8,7,10,2,3,11,12,6,13,14,9};
        for(auto offset:material.binding_offsets)
            material.minimum_binding_bytes.push_back(state[offset].size()*4);
        material.writable.assign(11,0);
        material.writable[8]=material.writable[9]=1;
        material.local_size_x=64;material.maximum_invocations=lanes;
        if(!session.register_shader(material))return 12;
    }
    for(std::size_t device=0;device<library->device_count();device++){
        const auto suffix=".vulkan."+std::to_string(device);
        const auto pipeline_id="nodus.pipeline.turing.wave.angular_material."+
                               std::to_string(device);
        if(joined&&!session.register_pipeline(pipeline_id,
            {{contract.id+suffix,nodus::gpu::TensorStateLibraryTool::ExecutionApi::Vulkan,
              contract.program_id},
             {material.id+suffix,nodus::gpu::TensorStateLibraryTool::ExecutionApi::Vulkan,
              material.program_id}}))return 13;
        auto row=tool_registry_global().create(joined?pipeline_id:
                                               contract.id+suffix);
        if(!row)return 7;
        const std::uint64_t work[4]={card,base,lanes,device+1};
        const std::uint64_t pipeline_work[14]={
            0,1,contract.program_id,card,base,lanes,device+1,
            1,1,material.program_id,card,base,lanes,device+1};
        const std::uint64_t shape[2]={joined?2ull:4ull,joined?7ull:0ull};
        const auto token=nodus_tensor_create(NODUS_DTYPE_U64,joined?2:1,shape);
        nodus_tensor_write(token,0,joined?static_cast<const void*>(pipeline_work):
                           static_cast<const void*>(work),
                           joined?sizeof(pipeline_work):sizeof(work));
        RawStackFrame frame{};
        if(!raw_stack_init_frame(frame,4096))return 8;
        raw_stack_push_abstract_tensor(frame,{token});
        ToolStackContext context{};context.stack.raw=&frame;
        row->execute_stack(context);
        nodus::tensors::AbstractTensorHandle result{};
        std::array<std::uint64_t,27> receipt{};
        if(!context.succeeded||!raw_stack_pop_abstract_tensor(frame,result)||
           nodus_tensor_read(result.id,0,receipt.data(),
                             joined?sizeof(receipt):9*sizeof(std::uint64_t))!=
               (joined?sizeof(receipt):9*sizeof(std::uint64_t))||receipt[0]!=0||
           (joined&&receipt[3]!=2))return 9;
        nodus_tensor_destroy(result.id);nodus_tensor_destroy(token);
        raw_stack_free_mask(frame);
        if(frame.backend)gp_mem_backend_release(frame.backend);
        std::vector<nodus::gpu::TensorStateLibraryTool::CpuLoan> output(outputs);
        for(std::size_t i=0;i<outputs;i++){
            output[i]=library->borrow_cpu(card,base+feeds+i,false);
            if(!output[i]||output[i].bytes()!=lanes*4)return 10;
        }
        auto final_re=joined?library->borrow_cpu(card,base+13,false):
                             nodus::gpu::TensorStateLibraryTool::CpuLoan{};
        auto final_im=joined?library->borrow_cpu(card,base+14,false):
                             nodus::gpu::TensorStateLibraryTool::CpuLoan{};
        if(joined&&(!final_re||!final_im))return 14;
        for(std::size_t lane=0;lane<lanes;lane++){
            const auto get=[&](std::size_t i){
                return std::bit_cast<float>(state[i][lane]);};
            float expected[4]{};
            if(kind=="angular_propagating"){
                const float k=6.2831853071795864769f/get(6);
                const float phase=get(3)*std::sqrt(k*k-get(4)*get(4)-get(5)*get(5));
                const float tr=std::cos(phase),ti=get(2)*std::sin(phase);
                expected[0]=get(1)*tr-get(0)*ti;
                expected[1]=get(1)*ti+get(0)*tr;
            }else if(kind=="angular_evanescent"){
                const float decay=std::exp(-get(2)*std::sqrt(get(3)));
                expected[0]=get(1)*decay;expected[1]=get(0)*decay;
            }else if(kind=="aperture_material"){
                const float k=6.2831853071795864769f/get(7);
                const float phase=get(3)*k*(get(6)-get(2))*get(4);
                const float attenuation=std::exp(-k*get(5)*get(4));
                const float tr=attenuation*std::cos(phase);
                const float ti=attenuation*std::sin(phase);
                expected[0]=get(1)*tr-get(0)*ti;
                expected[1]=get(1)*ti+get(0)*tr;
            }else if(kind=="volume3d"){
                constexpr std::size_t center=(32*64+32)*64+32;
                const std::size_t x=lane/(64*64),y=(lane/64)%64,z=lane%64;
                const int distance=std::abs(static_cast<int>(x)-32)
                    +std::abs(static_cast<int>(y)-32)
                    +std::abs(static_cast<int>(z)-32);
                expected[0]=lane==center?.625f:distance==1?.0625f:0.f;
            }else{
                expected[0]=get(9);expected[1]=get(8);
                expected[2]=get(11);expected[3]=get(10);
            }
            for(std::size_t i=0;i<outputs;i++){
                const auto* value=static_cast<const std::uint32_t*>(output[i].data());
                if(!std::isfinite(std::bit_cast<float>(value[lane]))||
                   std::abs(std::bit_cast<float>(value[lane])-expected[i])>2e-5f)
                    return 11;
            }
            if(joined){
                const float k=6.2831853071795864769f/get(6);
                const float phase=get(2)*k*(get(12)-get(10))*get(3);
                const float attenuation=std::exp(-k*get(11)*get(3));
                const float tr=attenuation*std::cos(phase);
                const float ti=attenuation*std::sin(phase);
                const float want_re=expected[0]*tr-expected[1]*ti;
                const float want_im=expected[0]*ti+expected[1]*tr;
                const auto* re=static_cast<const std::uint32_t*>(final_re.data());
                const auto* im=static_cast<const std::uint32_t*>(final_im.data());
                if(std::abs(std::bit_cast<float>(re[lane])-want_re)>2e-5f||
                   std::abs(std::bit_cast<float>(im[lane])-want_im)>2e-5f)
                    return 15;
            }
        }
        if(joined&&argc==5&&device==0){
            const auto* re=static_cast<const std::uint32_t*>(final_re.data());
            const auto* im=static_cast<const std::uint32_t*>(final_im.data());
            float peak=0.f;
            for(std::size_t lane=0;lane<lanes;lane++)
                peak=std::max(peak,std::hypot(std::bit_cast<float>(re[lane]),
                                              std::bit_cast<float>(im[lane])));
            if(!(peak>0.f))return 16;
            std::vector<std::uint8_t> pixels(lanes*3);
            for(std::size_t lane=0;lane<lanes;lane++){
                const float real=std::bit_cast<float>(re[lane])/peak;
                const float imag=std::bit_cast<float>(im[lane])/peak;
                const auto channel=[](float value){return static_cast<std::uint8_t>(
                    std::clamp(value,0.f,1.f)*255.f);};
                pixels[3*lane]=channel(.5f+.5f*real);
                pixels[3*lane+1]=channel(.5f+.5f*imag);
                pixels[3*lane+2]=channel(real*real+imag*imag);
            }
            if(!nodus::tensors::kpath::write_png_rgb_u8(
                   argv[4],512,512,
                   std::span<const std::uint8_t>(pixels.data(),pixels.size())))
                return 17;
        }
        std::cout<<"wave_kind="<<kind<<" device="<<device<<" lanes="<<lanes
                 <<" dispatch_ms="<<(joined?receipt[12]+receipt[21]:receipt[3])*1e-6
                 <<" stages="<<(joined?2:1)<<" verified=1\n";
    }
    nodus_tensor_destroy(dummy);
    return 0;
}
