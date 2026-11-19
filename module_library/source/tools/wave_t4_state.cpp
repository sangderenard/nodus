// wave_t4_state: the T4 angular-spectrum arena stage row over the tensor state library, through the fftfree-backed DLL (include/wave_t4_state_tool.h).
// The row tool's source, in the module library: the Menu lists it, the canvas compiler inlines it, it builds as its own
// plugin DLL, and the image registers it from it (src/wave_t4_state_tool.cpp).
// Its settings are its saved state (serialize/deserialize); a row with no library refuses.
#include "wave_t4_state_tool.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nodus::gpu {
namespace {
void* symbol(void* module,const char* name){
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module),name));
#else
    return dlsym(module,name);
#endif
}
double double_word(std::uint64_t value){
    double result=0;
    std::memcpy(&result,&value,sizeof(result));
    return result;
}
bool f32_state(TensorStateLibraryTool& library,std::uint64_t card,
               std::uint64_t address,std::uint64_t count){
    const auto spec=library.describe_state(card,address);
    if(!spec||spec->dtype!=nodus::tensors::TensorDType::F32)return false;
    std::uint64_t available=1;
    for(auto dim:spec->shape){
        if(!dim||available>UINT64_MAX/dim)return false;
        available*=dim;
    }
    return available>=count;
}
bool f64_state(TensorStateLibraryTool& library,std::uint64_t card,
               std::uint64_t address,std::uint64_t count){
    const auto spec=library.describe_state(card,address);
    if(!spec||spec->dtype!=nodus::tensors::TensorDType::F64)return false;
    std::uint64_t available=1;
    for(auto dim:spec->shape){
        if(!dim||available>UINT64_MAX/dim)return false;
        available*=dim;
    }
    return available>=count;
}
}

WaveT4StateTool::WaveT4StateTool(TensorStateLibraryTool& library,std::string dll)
    :library_(&library),dll_(std::move(dll)){}
WaveT4StateTool::WaveT4StateTool()=default;
void WaveT4StateTool::configure(std::string dll){
    std::lock_guard<std::mutex> lock(mutex_);
    dll_=std::move(dll);
}
// Saved state: the row's settings (magic "WT41", version 1, DLL path).
namespace {
constexpr std::uint32_t kWaveT4SettingsMagic=0x31345457,kWaveT4SettingsVersion=1;
}
bool WaveT4StateTool::serialize(OutputArchive& out) const{
    auto u32=[&](std::uint32_t v){return out.write(&v,sizeof(v));};
    auto u64=[&](std::uint64_t v){return out.write(&v,sizeof(v));};
    auto str=[&](const std::string& v){return u32(static_cast<std::uint32_t>(v.size()))&&(v.empty()||out.write(v.data(),v.size()));};
    auto vec=[&](const auto& v){
        if(!u64(v.size()))return false;
        for(const auto x:v)if(!u64(static_cast<std::uint64_t>(x)))return false;
        return true;
    };
    (void)vec;
    return u32(kWaveT4SettingsMagic)&&u32(kWaveT4SettingsVersion)&&str(dll_);
}
bool WaveT4StateTool::deserialize(InputArchive& in){
    auto u32=[&](std::uint32_t& v){return in.read(&v,sizeof(v));};
    auto u64=[&](std::uint64_t& v){return in.read(&v,sizeof(v));};
    auto str=[&](std::string& v,std::uint32_t limit){
        std::uint32_t n=0;
        if(!u32(n)||n>limit)return false;
        v.resize(n);
        return n==0||in.read(v.data(),n);
    };
    auto vec=[&](auto& v,std::uint64_t limit){
        std::uint64_t n=0;
        if(!u64(n)||n>limit)return false;
        v.resize(static_cast<std::size_t>(n));
        for(auto& x:v){
            std::uint64_t raw=0;
            using Element=std::decay_t<decltype(x)>;
            if(!u64(raw)||raw>std::numeric_limits<Element>::max())return false;
            x=static_cast<Element>(raw);
        }
        return true;
    };
    (void)vec;
    std::uint32_t magic=0,version=0;
    std::string dll;
    if(!u32(magic)||!u32(version)||magic!=kWaveT4SettingsMagic||version!=kWaveT4SettingsVersion||
       !str(dll,4096))return false;
    configure(std::move(dll));
    return true;
}
void WaveT4StateTool::initialize(const ToolInitContext& ctx){
    if(ctx.serialized_path&&ctx.serialized_path[0]!='\0'){
        FileInputArchive in(ctx.serialized_path);
        if(in.ok())deserialize(in);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if(module_||dll_.empty())return;
#ifdef _WIN32
    module_=LoadLibraryExA(dll_.c_str(),nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
    module_=dlopen(dll_.c_str(),RTLD_NOW|RTLD_LOCAL);
#endif
    if(!module_)return;
    create_=reinterpret_cast<decltype(create_)>(symbol(module_,"wave_t4_batch_plan_create"));
    destroy_=reinterpret_cast<decltype(destroy_)>(symbol(module_,"wave_t4_batch_plan_destroy"));
    step_=reinterpret_cast<decltype(step_)>(symbol(module_,"wave_t4_batch_angular_step"));
    if(create_&&destroy_&&step_)return;
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(module_));
#else
    dlclose(module_);
#endif
    module_=nullptr;create_=nullptr;destroy_=nullptr;step_=nullptr;
}
void WaveT4StateTool::shutdown(){
    std::lock_guard<std::mutex> lock(mutex_);
    if(plan_&&destroy_)destroy_(plan_);
    plan_=nullptr;plan_nx_=plan_ny_=0;
#ifdef _WIN32
    if(module_)FreeLibrary(static_cast<HMODULE>(module_));
#else
    if(module_)dlclose(module_);
#endif
    module_=nullptr;create_=nullptr;destroy_=nullptr;step_=nullptr;
}
void WaveT4StateTool::execute_stack(ToolStackContext& context){
    context.succeeded=false;
    if(!context.stack.raw||!library_)return;
    auto& stack=*context.stack.raw;
    nodus::tensors::AbstractTensorHandle control{},work{};
    if(!raw_stack_pop_abstract_tensor(stack,control))return;
    if(!raw_stack_pop_abstract_tensor(stack,work)){
        raw_stack_push_abstract_tensor(stack,control);return;
    }
    const auto started=std::chrono::steady_clock::now();
    std::array<std::uint64_t,kControlWords> c{};
    std::vector<std::uint64_t> w;
    std::uint64_t status=1,lease_ns=0,dispatch_ns=0;
    NodusTensorDesc control_desc{},work_desc{};
    if(nodus_tensor_describe(control.id,&control_desc)==NODUS_OK&&
       control_desc.dtype==NODUS_DTYPE_U64&&control_desc.rank==1&&
       control_desc.shape[0]==c.size()&&
       nodus_tensor_read(control.id,0,c.data(),sizeof(c))==sizeof(c)&&
       nodus_tensor_describe(work.id,&work_desc)==NODUS_OK&&
       work_desc.dtype==NODUS_DTYPE_U64&&work_desc.rank==1&&
       work_desc.shape[0]>=6&&work_desc.shape[0]<=68){
        w.resize(work_desc.shape[0]);
        if(nodus_tensor_read(work.id,0,w.data(),w.size()*sizeof(std::uint64_t))==
           static_cast<std::int64_t>(w.size()*sizeof(std::uint64_t))){
            status=2;
            const auto arenas=c[3],bands=c[4],nx=c[1],ny=c[2];
            const double dx=double_word(c[5]),dz=double_word(c[6]);
            const auto direction=static_cast<std::int64_t>(c[7]);
            if(c[0]==1&&arenas>=1&&arenas<=32&&bands>=1&&
               bands<=32&&arenas*bands<=32&&nx>=2&&nx<=4096&&
               ny>=2&&ny<=4096&&(nx&(nx-1))==0&&
               (ny&(ny-1))==0&&nx<=INT_MAX&&ny<=INT_MAX&&
               std::isfinite(dx)&&dx>0&&std::isfinite(dz)&&dz!=0&&
               (direction==1||direction==-1)&&w.size()==4+2*arenas&&
               w[0]&&w[1]&&nx*ny*arenas*bands<=
                   (512ull<<20)/sizeof(float)/2u){
                const auto samples=nx*ny*bands;
                status=3;
                std::unordered_set<std::uint64_t> unique{w[1]};
                bool valid=f64_state(*library_,w[0],w[1],arenas*bands);
                for(std::size_t i=0;i<arenas&&valid;++i){
                    const auto real=w[4+2*i],imag=w[5+2*i];
                    valid=real&&imag&&unique.insert(real).second&&
                          unique.insert(imag).second&&
                          f32_state(*library_,w[0],real,samples)&&
                          f32_state(*library_,w[0],imag,samples);
                }
                if(valid){
                    std::lock_guard<std::mutex> lock(mutex_);
                    status=4;
                    if(module_&&create_&&destroy_&&step_){
                        if(!plan_||plan_nx_!=static_cast<int>(nx)||
                           plan_ny_!=static_cast<int>(ny)){
                            if(plan_)destroy_(plan_);
                            plan_=create_(static_cast<int>(nx),static_cast<int>(ny));
                            plan_nx_=static_cast<int>(nx);
                            plan_ny_=static_cast<int>(ny);
                        }
                        if(plan_){
                            const auto lease_started=std::chrono::steady_clock::now();
                            auto waves=library_->borrow_cpu(w[0],w[1],false);
                            bool admitted=bool(waves);
                            if(admitted){
                                const auto* wavelengths=static_cast<const double*>(waves.data());
                                for(std::size_t i=0;i<arenas*bands;++i)
                                    admitted=admitted&&std::isfinite(wavelengths[i])&&
                                             wavelengths[i]>0&&dx<=0.5*wavelengths[i];
                                if(!admitted)status=3;
                            }
                            std::vector<TensorStateLibraryTool::CpuLoan> fields;
                            std::vector<float*> real,imag;
                            fields.reserve(2*arenas);real.reserve(arenas);imag.reserve(arenas);
                            for(std::size_t i=0;i<arenas&&admitted;++i){
                                auto re=library_->borrow_cpu(w[0],w[4+2*i],true);
                                auto im=library_->borrow_cpu(w[0],w[5+2*i],true);
                                admitted=bool(re)&&bool(im);
                                if(admitted){
                                    real.push_back(static_cast<float*>(re.data()));
                                    imag.push_back(static_cast<float*>(im.data()));
                                    fields.push_back(std::move(re));
                                    fields.push_back(std::move(im));
                                }
                            }
                            lease_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
                                std::chrono::nanoseconds>(std::chrono::steady_clock::now()-lease_started).count());
                            if(admitted){
                                const auto* wavelengths=static_cast<const double*>(waves.data());
                                const auto dispatch_started=std::chrono::steady_clock::now();
                                const auto result=step_(plan_,static_cast<int>(arenas),
                                    static_cast<int>(bands),dx,dz,
                                    static_cast<int>(direction),wavelengths,
                                    real.data(),imag.data());
                                dispatch_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
                                    std::chrono::nanoseconds>(std::chrono::steady_clock::now()-dispatch_started).count());
                                status=result?6:0;
                            }else if(status!=3)status=5;
                        }
                    }
                }
            }
        }
    }
    std::array<std::uint64_t,kReceiptWords> receipt{
        status,w.size()>2?w[2]:0,w.size()>3?w[3]:0,
        c[3],c[4],c[1],c[2],
        static_cast<std::uint64_t>(std::chrono::duration_cast<
            std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count()),
        w.empty()?0:w[0]};
    if(status==0)library_->record_work_completion(
        TensorStateLibraryTool::ExecutionApi::Cpu,0,
        (0x5434ull<<48)|(c[1]<<24)|c[2],c[3]*c[4],
        lease_ns,dispatch_ns,receipt[7]);
    const std::uint64_t shape[1]={kReceiptWords};
    const auto output=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    if(!output||nodus_tensor_write(output,0,receipt.data(),sizeof(receipt))!=
           sizeof(receipt)||!raw_stack_push_abstract_tensor(stack,{output})){
        if(output)nodus_tensor_destroy(output);
        raw_stack_push_abstract_tensor(stack,work);
        raw_stack_push_abstract_tensor(stack,control);return;
    }
    context.succeeded=true;
}

} // namespace nodus::gpu

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::gpu::WaveT4StateTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
