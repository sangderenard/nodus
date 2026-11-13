#include "fftfree_tool.h"
#include "tool_registry.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
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
bool read_words(std::uint64_t handle,std::uint64_t* words,std::size_t count){
    NodusTensorDesc desc{};
    return nodus_tensor_describe(handle,&desc)==NODUS_OK&&
        desc.dtype==NODUS_DTYPE_U64&&desc.rank==1&&desc.shape[0]==count&&
        nodus_tensor_read(handle,0,words,count*sizeof(*words))==
            static_cast<std::int64_t>(count*sizeof(*words));
}
float float_word(std::uint64_t value){
    std::uint32_t bits=static_cast<std::uint32_t>(value);
    float result=0;
    std::memcpy(&result,&bits,sizeof(result));
    return result;
}
bool integer_word(std::uint64_t value){return value<=INT_MAX;}
}

FftFreeTool::FftFreeTool(TensorStateLibraryTool& library,std::string dll)
    :library_(library),dll_(std::move(dll)){}

void FftFreeTool::initialize(const ToolInitContext&){
    std::lock_guard<std::mutex> lock(mutex_);
    if(module_||dll_.empty())return;
#ifdef _WIN32
    module_=LoadLibraryExA(dll_.c_str(),nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
    module_=dlopen(dll_.c_str(),RTLD_NOW|RTLD_LOCAL);
#endif
    if(!module_)return;
    init_=reinterpret_cast<Init>(symbol(module_,"fft_init_full_v2"));
    execute_real_=reinterpret_cast<ExecuteReal>(symbol(module_,"fft_execute_batched"));
    execute_complex_=reinterpret_cast<ExecuteComplex>(symbol(module_,"fft_execute_complex_batched"));
    plan_size_=reinterpret_cast<decltype(plan_size_)>(symbol(module_,"fft_ctx_size"));
    free_=reinterpret_cast<decltype(free_)>(symbol(module_,"fft_free"));
    if(init_&&execute_real_&&execute_complex_&&plan_size_&&free_)return;
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(module_));
#else
    dlclose(module_);
#endif
    module_=nullptr;init_=nullptr;execute_real_=nullptr;
    execute_complex_=nullptr;plan_size_=nullptr;free_=nullptr;
}
void FftFreeTool::shutdown(){
    std::lock_guard<std::mutex> lock(mutex_);
    if(plan_&&free_)free_(plan_);
    plan_=nullptr;cached_valid_=false;
#ifdef _WIN32
    if(module_)FreeLibrary(static_cast<HMODULE>(module_));
#else
    if(module_)dlclose(module_);
#endif
    module_=nullptr;init_=nullptr;execute_real_=nullptr;
    execute_complex_=nullptr;plan_size_=nullptr;free_=nullptr;
}
void FftFreeTool::execute_stack(ToolStackContext& context){
    context.succeeded=false;
    if(!context.stack.raw)return;
    auto& stack=*context.stack.raw;
    nodus::tensors::AbstractTensorHandle control{},work{};
    if(!raw_stack_pop_abstract_tensor(stack,control))return;
    if(!raw_stack_pop_abstract_tensor(stack,work)){
        raw_stack_push_abstract_tensor(stack,control);return;
    }
    std::array<std::uint64_t,kConfigWords> c{};
    std::array<std::uint64_t,kWorkWords> w{};
    std::array<std::uint64_t,kReceiptWords> receipt{};
    const auto started=std::chrono::steady_clock::now();
    std::uint64_t status=1,produced=0,actual_n=0,bins=0;
    std::uint64_t lease_ns=0,dispatch_ns=0;
    if(read_words(control.id,c.data(),c.size())&&
       read_words(work.id,w.data(),w.size())){
        status=2;
        const auto n=c[1];
        if(c[0]==1 && (w[0]==1||w[0]==2) && n>=2 &&
           n<=1'048'576 && (n&(n-1))==0 && c[7]<=8 &&
           w[1] && w[2] && w[4] && w[7] && w[8] &&
           (w[0]==1 ? (w[5]&&w[6]&&w[2]!=w[4]&&
                        w[2]!=w[5]&&w[2]!=w[6]&&
                        w[4]!=w[5]&&w[4]!=w[6]&&w[5]!=w[6])
                    : (w[3]&&w[3]!=w[2]&&w[4]!=w[2]&&w[4]!=w[3])) &&
           std::all_of(c.begin()+2,c.begin()+8,integer_word)&&
           std::all_of(c.begin()+8,c.begin()+32,integer_word)&&
           integer_word(c[34])&&integer_word(c[37])&&integer_word(c[38])&&
           std::isfinite(float_word(c[32]))&&
           std::isfinite(float_word(c[33]))&&
           std::isfinite(float_word(c[35]))&&
           std::isfinite(float_word(c[36]))&&
           c[2]>=1&&c[2]<=64&&c[3]>=1&&c[3]<=65536&&
           c[17]<=n&&c[18]<=n&&c[19]<=2&&c[39]<=1 &&
           w[8]<=std::numeric_limits<std::size_t>::max()/n){
            std::lock_guard<std::mutex> lock(mutex_);
            if(module_&&init_&&execute_real_&&execute_complex_){
                status=3;
                if(!cached_valid_||c!=cached_){
                    if(plan_&&free_)free_(plan_);
                    plan_=nullptr;cached_valid_=false;
                    int pattern[8]{};
                    for(std::size_t i=0;i<c[7];++i)pattern[i]=static_cast<int>(c[8+i]);
                    plan_=init_(static_cast<std::size_t>(n),
                        static_cast<int>(c[2]),static_cast<int>(c[3]),
                        static_cast<int>(c[4]),static_cast<int>(c[5]),
                        static_cast<int>(c[6]),c[7]?pattern:nullptr,
                        static_cast<std::size_t>(c[7]),
                        static_cast<int>(c[16]),static_cast<int>(c[17]),
                        static_cast<int>(c[18]),static_cast<int>(c[19]),
                        static_cast<int>(c[20]),static_cast<int>(c[21]),
                        static_cast<int>(c[22]),static_cast<int>(c[23]),
                        static_cast<int>(c[24]),static_cast<int>(c[25]),
                        static_cast<int>(c[26]),static_cast<int>(c[27]),
                        static_cast<int>(c[28]),static_cast<int>(c[29]),
                        static_cast<int>(c[30]),static_cast<int>(c[31]),
                        float_word(c[32]),float_word(c[33]),
                        static_cast<int>(c[34]),float_word(c[35]),
                        float_word(c[36]),static_cast<int>(c[37]),
                        static_cast<int>(c[38]));
                    if(plan_){cached_=c;cached_valid_=true;}
                }
                if(plan_){
                    actual_n=plan_size_(plan_);
                    bins=c[23]?actual_n/2+1:actual_n;
                    status=4;
                    auto valid=[&](std::uint64_t address,std::uint64_t samples){
                        const auto spec=library_.describe_state(w[1],address);
                        if(!spec||spec->dtype!=nodus::tensors::TensorDType::F32)return false;
                        std::uint64_t count=1;
                        for(const auto dim:spec->shape){
                            if(!dim||count>UINT64_MAX/dim)return false;
                            count*=dim;
                        }
                        return count>=samples;
                    };
                    if(actual_n>=n && actual_n<=1'048'576 && bins &&
                       w[8]<=UINT64_MAX/bins &&
                       (w[0]==1 ?
                           valid(w[2],w[7])&&
                           valid(w[4],w[8]*bins)&&
                           valid(w[5],w[8]*bins)&&
                           valid(w[6],w[8]*bins) :
                           w[7]<=w[8]&&w[7]<=UINT64_MAX/bins&&
                           w[7]<=UINT64_MAX/actual_n&&
                           valid(w[2],w[7]*bins)&&
                           valid(w[3],w[7]*bins)&&
                           valid(w[4],w[7]*actual_n))){
                        const auto lease_started=std::chrono::steady_clock::now();
                        auto in=library_.borrow_cpu(w[1],w[2],false);
                        TensorStateLibraryTool::CpuLoan imag_in;
                        if(w[0]==2)imag_in=library_.borrow_cpu(w[1],w[3],false);
                        auto out=library_.borrow_cpu(w[1],w[4],true);
                        TensorStateLibraryTool::CpuLoan imag_out,mag_out;
                        if(w[0]==1){
                            imag_out=library_.borrow_cpu(w[1],w[5],true);
                            mag_out=library_.borrow_cpu(w[1],w[6],true);
                        }
                        lease_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
                            std::chrono::nanoseconds>(std::chrono::steady_clock::now()-lease_started).count());
                        status=5;
                        if(in&&out&&(w[0]==1 ? (imag_out&&mag_out) : bool(imag_in))){
                            const auto dispatch_started=std::chrono::steady_clock::now();
                            produced=w[0]==1 ? execute_real_(plan_,
                                static_cast<const float*>(in.data()),
                                static_cast<std::size_t>(w[7]),
                                static_cast<float*>(out.data()),
                                static_cast<float*>(imag_out.data()),
                                static_cast<float*>(mag_out.data()),
                                static_cast<int>(c[16]),static_cast<int>(c[39]),
                                static_cast<std::size_t>(w[8])) :
                                execute_complex_(plan_,
                                static_cast<const float*>(in.data()),
                                static_cast<const float*>(imag_in.data()),
                                static_cast<std::size_t>(w[7]),
                                static_cast<float*>(out.data()),
                                static_cast<int>(c[16]),static_cast<int>(c[39]),
                                static_cast<std::size_t>(w[8]));
                            dispatch_ns=static_cast<std::uint64_t>(std::chrono::duration_cast<
                                std::chrono::nanoseconds>(std::chrono::steady_clock::now()-dispatch_started).count());
                            status=produced?0:6;
                        }
                    }
                }
            }
        }
    }
    receipt={status,w[9],produced,actual_n,bins,w[4],
        static_cast<std::uint64_t>(std::chrono::duration_cast<
            std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count()),
        w[1],w[0]};
    if(status==0){
        const std::uint64_t program=(c[5]&255ull)<<48|
            (c[20]&255ull)<<40|(c[4]&255ull)<<32|actual_n;
        library_.record_work_completion(
            TensorStateLibraryTool::ExecutionApi::Cpu,0,program,produced,
            lease_ns,dispatch_ns,receipt[6]);
    }
    const std::uint64_t shape[1]={kReceiptWords};
    const auto output=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    if(!output||nodus_tensor_write(output,0,receipt.data(),sizeof(receipt))!=
            sizeof(receipt)||!raw_stack_push_abstract_tensor(stack,{output})){
        if(output)nodus_tensor_destroy(output);
        raw_stack_push_abstract_tensor(stack,work);
        raw_stack_push_abstract_tensor(stack,control);
        return;
    }
    context.succeeded=true;
}
bool register_fftfree_tool(ToolRegistry& registry,
    TensorStateLibraryTool& library,const std::string& dll){
    if(dll.empty())return false;
    ToolRegistry::Entry item;
    item.id="nodus.fftfree";item.name="fftfree transform";
    item.caps=ToolCaps::None;
    item.factory=[&library,dll]{
        return std::unique_ptr<ITool,std::function<void(ITool*)>>(
            new FftFreeTool(library,dll),[](ITool* value){delete value;});
    };
    return registry.register_tool(std::move(item));
}
} // namespace nodus::gpu
