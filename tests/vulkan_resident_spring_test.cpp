#include "kernel_vulkan_runner.h"
#include "mem_backend.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>
#define NOMINMAX
#include <windows.h>

namespace {
constexpr std::uint32_t width=4,height=4,axes=3,lanes=width*height*axes;
using Buffer=std::unique_ptr<void,decltype(&gp_mem_backend_release)>;
using Columns=std::array<std::vector<double>,26>;
void init(Columns& columns){
    for(auto& column:columns)column.resize(lanes,0.0);
    for(std::uint32_t lane=0;lane<lanes;lane++){
        const auto node=lane/axes,axis=lane%axes,x=node%width,y=node/width;
        const double position=axis==0?x*.1:axis==1?(height-y)*.1:0.0;
        columns[0][lane]=position;
        columns[3][lane]=.05;columns[4][lane]=.01;
        columns[5][lane]=x>0;columns[6][lane]=x+1<width;
        columns[7][lane]=y>0;columns[8][lane]=y+1<height;
        columns[9][lane]=axis==1?-9.8:0.0;
        columns[10][lane]=1.0;columns[15][lane]=1.0;
        columns[16][lane]=y==0;columns[17][lane]=position;
        columns[18][lane]=axis==0?-.1:0.0;
        columns[19][lane]=axis==0?.1:0.0;
        columns[20][lane]=axis==1?.1:0.0;
        columns[21][lane]=axis==1?-.1:0.0;
        columns[22][lane]=10.0;
        columns[23][lane]=axis==0?.02:0.0;
    }
}
void gather(Columns& columns){
    for(std::uint32_t lane=0;lane<lanes;lane++){
        const auto node=lane/axes,x=node%width,y=node/width;
        columns[11][lane]=x>0?columns[17][lane-axes]:0;
        columns[12][lane]=x+1<width?columns[17][lane+axes]:0;
        columns[13][lane]=y>0?columns[17][lane-width*axes]:0;
        columns[14][lane]=y+1<height?columns[17][lane+width*axes]:0;
    }
}
}

int main(int argc,char** argv){
    if(argc!=3)return 2;
    std::ifstream file(argv[1],std::ios::binary);
    std::vector<char> binary{std::istreambuf_iterator<char>(file),{}};
    if(binary.empty()||binary.size()%4)return 3;
    std::vector<std::uint32_t> shader(binary.size()/4);
    std::memcpy(shader.data(),binary.data(),binary.size());
    const auto module=LoadLibraryExA(argv[2],nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if(!module)return 4;
    const auto entry=reinterpret_cast<void (*)(void**,int32_t*)>(
        GetProcAddress(module,"spring_network_step_batched__tick"));
    const auto reset=reinterpret_cast<void (*)()>(
        GetProcAddress(module,"turing_validation_error_reset"));
    const auto take=reinterpret_cast<std::uint32_t (*)()>(
        GetProcAddress(module,"turing_validation_error_take"));
    if(!entry||((reset==nullptr)!=(take==nullptr)))return 5;
    Columns initial,cpu;
    init(initial);cpu=initial;
    gp_mem_backend_vulkan_params_t params{0,GP_VULKAN_ALLOC_DEVICE_LOCAL};
    std::string error;
    Buffer context(nodus::kernels::create_vulkan_backend(&params,nullptr,&error),
                   &gp_mem_backend_release);
    if(!context){std::cerr<<error<<'\n';return 6;}
    const auto* vtable=gp_mem_backend_get_vtable(context.get());
    std::vector<Buffer> buffers;
    std::vector<nodus::kernels::VulkanBackendBinding> bindings;
    for(std::uint32_t i=0;i<27;i++){
        const auto bytes=i==26?sizeof(std::uint32_t)*2:lanes*sizeof(double);
        Buffer buffer(vtable->alloc(context.get(),bytes,0),&gp_mem_backend_release);
        if(!buffer)return 7;
        if(i<26&&!vtable->copy_to_backend(buffer.get(),0,
               initial[i].data(),bytes))return 8;
        if(i==26){
            const std::uint32_t control[2]={lanes,width*axes};
            if(!vtable->copy_to_backend(buffer.get(),0,control,sizeof(control)))return 8;
        }
        bindings.push_back({buffer.get(),0,bytes});
        buffers.push_back(std::move(buffer));
    }
    constexpr std::uint32_t steps=4,quantums=2;
    for(std::uint32_t quantum=0;quantum<quantums;quantum++)
        if(!nodus::kernels::dispatch_spirv_resident_loop(shader,"main",bindings,
                1,steps,17,23,24,25,&error)){
            std::cerr<<error<<'\n';return 9;
        }
    nodus::kernels::VulkanPipelineCacheStats pipeline{};
    if(!nodus::kernels::query_vulkan_pipeline_cache_stats(
           context.get(),&pipeline,&error)||pipeline.created!=1||
       pipeline.hits<1||pipeline.resident!=1)return 13;
    for(std::uint32_t step=0;step<steps*quantums;step++){
        gather(cpu);
        std::array<void*,26> pointers{};
        for(int i=0;i<26;i++)pointers[i]=cpu[i].data();
        if(reset)reset();
        entry(pointers.data(),nullptr);
        if(take&&take())return 10;
        cpu[17]=cpu[24];cpu[23]=cpu[25];
    }
    std::vector<double> position(lanes),velocity(lanes);
    if(!vtable->copy_from_backend(buffers[17].get(),0,position.data(),
                                   lanes*sizeof(double))||
       !vtable->copy_from_backend(buffers[23].get(),0,velocity.data(),
                                   lanes*sizeof(double)))return 11;
    double max_error=0;
    for(std::uint32_t i=0;i<lanes;i++){
        max_error=std::max(max_error,std::abs(position[i]-cpu[17][i]));
        max_error=std::max(max_error,std::abs(velocity[i]-cpu[23][i]));
    }
    if(!std::isfinite(max_error)||max_error>1e-11)return 12;
    std::cout<<"resident_steps="<<steps*quantums<<" lanes="<<lanes
             <<" max_llvm_difference="<<max_error
             <<" pipeline_created="<<pipeline.created
             <<" pipeline_hits="<<pipeline.hits<<" pass\n";
    FreeLibrary(module);
    return 0;
}
