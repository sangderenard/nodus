#include "kernel_vulkan_runner.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc,char** argv){
    if(argc!=2)return 2;
    std::ifstream input(argv[1],std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(input),{}};
    if(bytes.empty()||bytes.size()%4)return 2;
    std::vector<std::uint32_t> spirv(bytes.size()/4);
    std::memcpy(spirv.data(),bytes.data(),bytes.size());
    const auto fb=[](float v){return std::bit_cast<std::uint32_t>(v);};
    std::vector<std::uint32_t> hit(93),out(86+92),meta(24),tri(16),
        material(32*12),uv(1),uv_accum(1),bdpt(28+8+12),control(17);
    hit[0]=fb(.25f);hit[1]=fb(.5f);hit[2]=fb(0.f);
    hit[5]=fb(1.f);hit[8]=fb(-1.f);
    hit[14]=0;hit[15]=0;hit[18]=0;
    hit[27]=fb(1.f);hit[28]=fb(1.f);hit[92]=7;
    tri[11]=fb(1.f);tri[12]=1;tri[13]=0;
    material[5]=fb(1.f);material[7]=fb(1.f);
    meta[6]=1;
    control[0]=1;control[1]=1;control[2]=1;control[3]=1;
    control[11]=1;control[12]=1;control[13]=1;
    control[14]=3;control[15]=28;control[16]=36;
    std::array<std::vector<std::uint32_t>*,9> values{
        &hit,&out,&meta,&tri,&material,&uv,&uv_accum,&bdpt,&control};
    std::vector<nodus::kernels::VulkanDeviceInfo> devices;
    std::string error;
    if(!nodus::kernels::enumerate_vulkan_devices(&devices,&error)){
        std::cerr<<error<<'\n';return 3;
    }
    std::size_t executed=0;
    for(const auto& device:devices){
        if(!device.compute)continue;
        std::fill(out.begin(),out.end(),0);
        meta[0]=meta[1]=0;
        meta[3]=meta[4]=meta[5]=0;
        std::fill(bdpt.begin(),bdpt.end(),0);
        std::vector<nodus::kernels::VulkanBufferBinding> bindings;
        for(auto* v:values)bindings.push_back({v->data(),v->size()*4,true});
        gp_mem_backend_vulkan_params_t select{};
        select.device_index=device.index;
        if(!nodus::kernels::dispatch_spirv_on_device(spirv,"main",bindings,1,
                                                     &select,&error)){
            std::cerr<<"device="<<device.index<<" T3 failed: "<<error<<'\n';
            return 4;
        }
        const float x=std::bit_cast<float>(out[86]);
        std::cout<<"device="<<device.index<<" name="<<device.name
                 <<" child="<<meta[0]<<" terminal="<<meta[1]
                 <<" terminal_x="<<x<<" bdpt="<<meta[3]<<","<<meta[4]
                 <<","<<meta[5]<<'\n';
        if(meta[0]!=0||meta[1]!=1||std::abs(x-.25f)>1e-5f||
           meta[3]!=1||meta[4]!=1||meta[5]!=1||
           bdpt[0]!=7||bdpt[4]!=0||
           std::abs(std::bit_cast<float>(bdpt[7])-.25f)>1e-5f||
           std::abs(std::bit_cast<float>(bdpt[28+2])-1.f)>1e-5f)
            return 5;
        ++executed;
    }
    return executed?0:6;
}
