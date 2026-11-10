// Direct native row benchmark of an existing compiler product, not ray throughput.
#include "tool_api.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include <windows.h>
#include <cassert>
#include <chrono>
#include <cstdio>
int main(int argc,char**argv){
 assert(argc==3);
 auto module=LoadLibraryA(argv[1]);assert(module);
 auto create=reinterpret_cast<ITool*(*)()>(GetProcAddress(module,"create_tool"));
 auto destroy=reinterpret_cast<void(*)(ITool*)>(GetProcAddress(module,"destroy_tool"));assert(create && destroy);
 ITool* tool=create();tool->initialize({});
 auto kernel=LoadLibraryA(argv[2]);assert(kernel);
 auto entry=reinterpret_cast<void(*)(void**,int64_t*)>(GetProcAddress(kernel,"visual_turing_nodus_graph"));assert(entry);
 double left=7,right=3,output=0;void* buffers[]={&left,&right,&output};
 constexpr int count=20000;
 auto start=std::chrono::steady_clock::now();
 for(int i=0;i<count;++i){entry(buffers,nullptr);assert(output==20);}
 auto direct=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 uint64_t x=nodus_tensor_create(NODUS_DTYPE_F64,0,nullptr),y=nodus_tensor_create(NODUS_DTYPE_F64,0,nullptr);assert(x&&y);
 assert(nodus_tensor_write(x,0,&left,8)==8 && nodus_tensor_write(y,0,&right,8)==8);
 RawStackFrame raw{};assert(raw_stack_init_frame(raw,4096));
 ToolStackContext context{};context.stack.raw=&raw;
 start=std::chrono::steady_clock::now();
 for(int i=0;i<count;++i){
   assert(raw_stack_push_abstract_tensor(raw,{x}) && raw_stack_push_abstract_tensor(raw,{y}));
   context.succeeded=true;tool->execute_stack(context);assert(context.succeeded);
   nodus::tensors::AbstractTensorHandle result{};assert(raw_stack_pop_abstract_tensor(raw,result));
   assert(nodus_tensor_read(result.id,0,&output,8)==8 && output==20);nodus_tensor_destroy(result.id);
   assert(raw.byte_count==0);
 }
 auto row=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::printf("{\"iterations\":%d,\"direct_kernel_seconds\":%.9f,\"native_tool_row_seconds\":%.9f,\"python_compute_calls\":0,\"output\":%.1f}\n",count,direct,row,output);
 nodus_tensor_destroy(x);nodus_tensor_destroy(y);raw_stack_free_mask(raw);if(raw.backend)gp_mem_backend_release(raw.backend);
 tool->shutdown();destroy(tool);FreeLibrary(module);FreeLibrary(kernel);
}
