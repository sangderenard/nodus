#include "wave_t4_state_tool.h"
#include "tool_registry.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
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
    :library_(library),dll_(std::move(dll)){}
void WaveT4StateTool::initialize(const ToolInitContext&){
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
    if(!context.stack.raw)return;
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
                bool valid=f64_state(library_,w[0],w[1],arenas*bands);
                for(std::size_t i=0;i<arenas&&valid;++i){
                    const auto real=w[4+2*i],imag=w[5+2*i];
                    valid=real&&imag&&unique.insert(real).second&&
                          unique.insert(imag).second&&
                          f32_state(library_,w[0],real,samples)&&
                          f32_state(library_,w[0],imag,samples);
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
                            auto waves=library_.borrow_cpu(w[0],w[1],false);
                            std::vector<TensorStateLibraryTool::CpuLoan> fields;
                            std::vector<float*> real,imag;
                            fields.reserve(2*arenas);real.reserve(arenas);imag.reserve(arenas);
                            bool admitted=bool(waves);
                            for(std::size_t i=0;i<arenas&&admitted;++i){
                                auto re=library_.borrow_cpu(w[0],w[4+2*i],true);
                                auto im=library_.borrow_cpu(w[0],w[5+2*i],true);
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
                            status=5;
                            if(admitted){
                                const auto* wavelengths=static_cast<const double*>(waves.data());
                                bool finite=true;
                                for(std::size_t i=0;i<arenas*bands;++i)
                                    finite=finite&&std::isfinite(wavelengths[i])&&
                                           wavelengths[i]>0;
                                if(finite){
                                    const auto dispatch_started=std::chrono::steady_clock::now();
                                    const auto result=step_(plan_,static_cast<int>(arenas),
                                        static_cast<int>(bands),dx,dz,
                                        static_cast<int>(direction),wavelengths,
                                        real.data(),imag.data());
                                    dispatch_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
                                        std::chrono::nanoseconds>(std::chrono::steady_clock::now()-dispatch_started).count());
                                    status=result?6:0;
                                }else status=3;
                            }
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
    if(status==0)library_.record_work_completion(
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
bool register_wave_t4_state_tool(ToolRegistry& registry,
    TensorStateLibraryTool& library,const std::string& dll){
    if(dll.empty())return false;
    ToolRegistry::Entry item;
    item.id="nodus.wave.t4.state";item.name="T4 arena state";
    item.caps=ToolCaps::None;
    item.factory=[&library,dll]{
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new WaveT4StateTool(library,dll),[](ITool* value){delete value;});
    };
    return registry.register_tool(std::move(item));
}
} // namespace nodus::gpu
