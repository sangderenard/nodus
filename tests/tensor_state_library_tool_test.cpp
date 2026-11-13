#include "tensor_state_library_tool.h"
#include "gpu_fabric_dispatch_tool.h"
#include "cpu_fabric_dispatch_tool.h"
#include "kernel_pipeline_tool.h"
#include "fabric_resident_offer.h"
#include "tool_registry.h"
#include "edge_tensor_fifo.h"

#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {
constexpr std::uint64_t card=17,address=29;
std::array<std::uint64_t,32> send(ITool& tool,EdgeTensorFifo& command_edge,
                                  EdgeTensorFifo& payload_edge,EdgeTensorFifo& reply_edge,
                                  const std::array<std::uint64_t,12>& request,
                                  std::uint64_t payload){
    const std::uint64_t shape[1]={12};
    const auto command=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    nodus_tensor_write(command,0,request.data(),sizeof(request));
    bool dropped=false;std::size_t read=0;
    if(!command_edge.push(1,1,&command,sizeof(command),&dropped)||dropped||
       !payload_edge.push(2,2,&payload,sizeof(payload),&dropped)||dropped)
        return {UINT64_MAX};
    std::uint64_t received_command=0,received_payload=0;
    if(!command_edge.pop(11,&received_command,8,read)||read!=8||
       !payload_edge.pop(12,&received_payload,8,read)||read!=8)
        return {UINT64_MAX};
    RawStackFrame frame{};
    if(!raw_stack_init_frame(frame,4096))return {UINT64_MAX};
    raw_stack_push_abstract_tensor(frame,{received_command});
    raw_stack_push_abstract_tensor(frame,{received_payload});
    ToolStackContext context{};context.stack.raw=&frame;
    tool.execute_stack(context);
    nodus::tensors::AbstractTensorHandle response{};
    if(!context.succeeded||!raw_stack_pop_abstract_tensor(frame,response)||
       !reply_edge.push(3,3,&response.id,8,&dropped)||dropped)
        return {UINT64_MAX};
    std::uint64_t received_response=0;
    if(!reply_edge.pop(13,&received_response,8,read)||read!=8)
        return {UINT64_MAX};
    std::array<std::uint64_t,32> values{};
    if(nodus_tensor_read(received_response,0,values.data(),sizeof(values))!=
       sizeof(values))return {UINT64_MAX};
    nodus_tensor_destroy(response.id);nodus_tensor_destroy(command);
    raw_stack_free_mask(frame);
    if(frame.backend)gp_mem_backend_release(frame.backend);
    return values;
}
}

int main(int argc,char** argv){
    if(argc!=3)return 2;
    std::ifstream source(argv[1],std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(source),{}};
    if(bytes.empty()||bytes.size()%4)return 2;
    std::vector<std::uint32_t> shader(bytes.size()/4);
    std::memcpy(shader.data(),bytes.data(),bytes.size());
    auto tool=tool_registry_global().create("nodus.tensor.state.library");
    if(!tool||tool->port_count()!=3||
       tool->port_spec(0).kind!=ToolPortKind::Argument||
       tool->port_spec(1).kind!=ToolPortKind::Argument||
       tool->port_spec(2).kind!=ToolPortKind::Return)return 1;
    tool->initialize({});
    auto* library=dynamic_cast<nodus::gpu::TensorStateLibraryTool*>(tool.get());
    if(!library||library->device_count()<3)return 2;
    auto type=ValueTypeRegistry::global().register_primitive("tensor_library_token",8);
    EdgeTensorFifo command_edge,payload_edge,reply_edge;
    for(auto* edge:{&command_edge,&payload_edge,&reply_edge}){
        edge->configure({1},4,0,8,type);
    }
    if(!command_edge.subscribe(11,true)||!payload_edge.subscribe(12,true)||
       !reply_edge.subscribe(13,true))return 3;
    const std::uint64_t dummy_shape[1]={1};
    auto dummy=nodus_tensor_create(NODUS_DTYPE_U64,1,dummy_shape);
    std::array<std::uint64_t,12> request{};
    request[0]=1;request[1]=card;request[2]=901;request[4]=8*1024*1024;
    request[5]=request[6]=request[7]=4*1024*1024;
    auto receipt=send(*tool,command_edge,payload_edge,reply_edge,request,dummy);
    if(receipt[0]!=0||receipt[5]!=3)return 4;
    const std::uint64_t data_shape[2]={2,3};
    auto data=nodus_tensor_create(NODUS_DTYPE_F64,2,data_shape);
    const double values[6]={1,2,3,4,5,6};
    nodus_tensor_write(data,0,values,sizeof(values));
    request={};request[0]=2;request[1]=card;request[3]=address;
    receipt=send(*tool,command_edge,payload_edge,reply_edge,request,data);
    if(receipt[0]!=0||receipt[3]!=sizeof(values))return 5;
    {
        auto loan=library->borrow_gpu(card,address,0,true);
        if(!loan)return 6;
        const auto* vt=gp_mem_backend_get_vtable(loan.handle());
        const double changed=37;
        if(!vt||!vt->copy_to_backend(loan.handle(),0,&changed,sizeof(changed)))return 7;
        if(library->borrow_gpu(card,address,1,false))return 8;
    }
    {
        auto loan=library->borrow_gpu(card,address,2,false);
        if(!loan)return 9;
        double observed=0;
        const auto* vt=gp_mem_backend_get_vtable(loan.handle());
        if(!vt||!vt->copy_from_backend(loan.handle(),0,&observed,sizeof(observed))||
           observed!=37)return 10;
    }
    request={};request[0]=3;request[1]=card;request[3]=address;
    receipt=send(*tool,command_edge,payload_edge,reply_edge,request,dummy);
    if(receipt[0]!=0||receipt[3]!=0)return 11;
    if(!nodus::gpu::register_gpu_fabric_dispatchers(
           tool_registry_global(),*library,shader,256))return 12;
    if(!nodus::gpu::register_cpu_fabric_dispatcher(
           tool_registry_global(),*library,argv[2],
           "spring_network_step_batched__tick",64))return 12;
    constexpr std::uint64_t base=1000,lanes=64;
    const std::uint64_t column_shape[1]={lanes};
    std::vector<double> column(lanes);
    for(int i=0;i<27;i++){
        std::uint64_t tensor=0;
        if(i==26){
            const std::uint64_t control_shape[1]={1};
            tensor=nodus_tensor_create(NODUS_DTYPE_U32,1,control_shape);
            const std::uint32_t count=lanes;
            nodus_tensor_write(tensor,0,&count,sizeof(count));
        }else{
            tensor=nodus_tensor_create(NODUS_DTYPE_F64,1,column_shape);
            std::fill(column.begin(),column.end(),0.0);
            if(i==0||i==17)std::fill(column.begin(),column.end(),1.0);
            if(i==3)std::fill(column.begin(),column.end(),.05);
            if(i==4)std::fill(column.begin(),column.end(),.001);
            if(i==9)std::fill(column.begin(),column.end(),-9.8);
            if(i==10||i==15)std::fill(column.begin(),column.end(),1.0);
            if(i==22)std::fill(column.begin(),column.end(),10.0);
            nodus_tensor_write(tensor,0,column.data(),column.size()*sizeof(double));
        }
        request={};request[0]=2;request[1]=card;request[3]=base+i;
        receipt=send(*tool,command_edge,payload_edge,reply_edge,request,tensor);
        nodus_tensor_destroy(tensor);
        if(receipt[0]!=0)return 13;
    }
    std::array<std::uint64_t,3> elapsed{};
    for(std::size_t device=0;device<3;device++){
        auto worker=tool_registry_global().create(
            "nodus.fabric.dispatch.vulkan."+std::to_string(device));
        if(!worker)return 14;
        const std::uint64_t work_shape[1]={4};
        auto work=nodus_tensor_create(NODUS_DTYPE_U64,1,work_shape);
        const std::uint64_t command[4]={card,base,lanes,device+1};
        nodus_tensor_write(work,0,command,sizeof(command));
        RawStackFrame frame{};
        if(!raw_stack_init_frame(frame,4096))return 15;
        raw_stack_push_abstract_tensor(frame,{work});
        ToolStackContext invocation{};invocation.stack.raw=&frame;
        worker->execute_stack(invocation);
        nodus::tensors::AbstractTensorHandle result{};
        if(!invocation.succeeded||!raw_stack_pop_abstract_tensor(frame,result))return 16;
        std::array<std::uint64_t,6> event{};
        if(nodus_tensor_read(result.id,0,event.data(),sizeof(event))!=sizeof(event)||
           event[0]!=0||event[1]!=device||event[2]!=lanes)return 17;
        elapsed[device]=event[3];
        bool dropped=false;std::size_t read=0;
        if(!reply_edge.push(3,3,&result.id,8,&dropped)||dropped)return 18;
        std::uint64_t receipt_id=0;
        if(!reply_edge.pop(13,&receipt_id,8,read)||read!=8||receipt_id!=result.id)return 19;
        nodus_tensor_destroy(result.id);nodus_tensor_destroy(work);
        raw_stack_free_mask(frame);if(frame.backend)gp_mem_backend_release(frame.backend);
    }
    std::uint64_t cpu_ns=0;
    {
        auto worker=tool_registry_global().create("nodus.fabric.dispatch.cpu");
        if(!worker)return 22;
        worker->initialize({});
        const std::uint64_t work_shape[1]={4};
        auto work=nodus_tensor_create(NODUS_DTYPE_U64,1,work_shape);
        const std::uint64_t command[4]={card,base,lanes,4};
        nodus_tensor_write(work,0,command,sizeof(command));
        RawStackFrame frame{};
        if(!raw_stack_init_frame(frame,4096))return 23;
        raw_stack_push_abstract_tensor(frame,{work});
        ToolStackContext invocation{};invocation.stack.raw=&frame;
        worker->execute_stack(invocation);
        nodus::tensors::AbstractTensorHandle result{};
        if(!invocation.succeeded||!raw_stack_pop_abstract_tensor(frame,result))return 24;
        std::array<std::uint64_t,6> event{};
        if(nodus_tensor_read(result.id,0,event.data(),sizeof(event))!=sizeof(event)||
           event[0]!=0||event[2]!=lanes)return 25;
        cpu_ns=event[3];
        bool dropped=false;std::size_t read=0;
        if(!reply_edge.push(3,3,&result.id,8,&dropped)||dropped)return 26;
        std::uint64_t receipt_id=0;
        if(!reply_edge.pop(13,&receipt_id,8,read)||read!=8||receipt_id!=result.id)return 27;
        nodus_tensor_destroy(result.id);nodus_tensor_destroy(work);
        raw_stack_free_mask(frame);if(frame.backend)gp_mem_backend_release(frame.backend);
        worker->shutdown();
    }
    {
        auto output=library->borrow_gpu(card,base+24,0,false);
        double observed=0;
        const auto* vt=output?gp_mem_backend_get_vtable(output.handle()):nullptr;
        if(!vt||!vt->copy_from_backend(output.handle(),0,&observed,sizeof(observed))||
           !std::isfinite(observed)||observed>=1.0||observed<.99)return 20;
    }
    {
        using Api=nodus::gpu::TensorStateLibraryTool::ExecutionApi;
        const std::string pipeline_id="nodus.pipeline.test.gpu_then_cpu";
        if(!nodus::gpu::register_kernel_pipeline(tool_registry_global(),pipeline_id,
            {{"nodus.fabric.dispatch.vulkan.0",Api::Vulkan,1},
             {"nodus.fabric.dispatch.cpu",Api::Cpu,1}}))return 54;
        auto pipeline=tool_registry_global().create(pipeline_id);
        if(!pipeline)return 55;
        pipeline->initialize({});
        const std::uint64_t shape[2]={2,7};
        auto work=nodus_tensor_create(NODUS_DTYPE_U64,2,shape);
        const std::uint64_t rows[14]={0,1,1,card,base,lanes,1,
                                      1,0,1,card,base,lanes,2};
        if(!work||nodus_tensor_write(work,0,rows,sizeof(rows))!=sizeof(rows))return 56;
        RawStackFrame frame{};
        if(!raw_stack_init_frame(frame,4096))return 57;
        raw_stack_push_abstract_tensor(frame,{work});
        ToolStackContext invocation{};invocation.stack.raw=&frame;
        pipeline->execute_stack(invocation);
        nodus::tensors::AbstractTensorHandle result{};
        std::array<std::uint64_t,27> event{};
        if(!invocation.succeeded||!raw_stack_pop_abstract_tensor(frame,result)||
           nodus_tensor_read(result.id,0,event.data(),sizeof(event))!=sizeof(event)||
           event[0]!=0||event[2]!=2||event[3]!=2||
           event[9]!=0||event[18]!=0)return 58;
        nodus_tensor_destroy(result.id);nodus_tensor_destroy(work);
        raw_stack_free_mask(frame);if(frame.backend)gp_mem_backend_release(frame.backend);
        pipeline->shutdown();
    }
    {
        auto warm=library->borrow_gpu(card,base+17,0,false);
        if(!warm)return 28;
    }
    request={};request[0]=2;request[1]=card;request[3]=base+100;
    receipt=send(*tool,command_edge,payload_edge,reply_edge,request,data);
    if(receipt[0])return 29;
    const std::uint64_t candidate_shape[2]={2,2};
    auto candidates=nodus_tensor_create(NODUS_DTYPE_U64,2,candidate_shape);
    const std::uint64_t choices[4]={base+100,1,base+17,1};
    nodus_tensor_write(candidates,0,choices,sizeof(choices));
    request={};request[0]=5;request[1]=card;request[5]=0;
    receipt=send(*tool,command_edge,payload_edge,reply_edge,request,candidates);
    nodus_tensor_destroy(candidates);
    if(receipt[0]||receipt[2]!=base+17||receipt[21]!=1||
       receipt[22]!=lanes*sizeof(double)||receipt[23]!=lanes*sizeof(double))
        return 30;
    const std::array<nodus::gpu::TensorStateLibraryTool::Candidate,2> ready{{
        {base+100,1},{base+17,1}}};
    const auto direct_choice=library->choose_resident(card,0,ready);
    if(!direct_choice||direct_choice->index!=1)return 34;
    std::uint64_t cold_expiry=5000;
    const auto offer=nodus::gpu::resident_fabric_offer(*library,
        [](std::uint64_t peer)->std::optional<std::size_t>{
            if(peer==500)return 0;return {};
        },[&](std::uint64_t payload)->std::optional<nodus::gpu::FabricStateRef>{
            if(payload==1)return nodus::gpu::FabricStateRef{
                card,base+100,1,cold_expiry,100};
            if(payload==2||payload==3)
                return nodus::gpu::FabricStateRef{card,base+17,1,5000,100};
            return {};
        },[]{return std::uint64_t{1000};});
    using Entry=nodus::contracts::ContractNegotiator::Entry;
    std::vector<Entry> pending(3);
    pending[0].work.payload=1;
    pending[1].work.payload=2;
    pending[2].work.payload=3;
    nodus::contracts::Peer peer{};peer.id=500;
    if(offer(peer,pending)!=1)return 35; // warm beats older cold
    auto market_choice=[&](std::uint64_t expected){
        nodus::contracts::ContractNegotiator market(3,offer,10,false);
        std::uint64_t offered=0;
        nodus::contracts::Producer producer{};
        producer.id=42;
        producer.route=[](const nodus::contracts::Result&){return true;};
        producer.request=[](nodus::contracts::Contract&){return false;};
        peer.exchange=[&](const nodus::contracts::Loan* loan){
            nodus::contracts::Exchange reply{};
            if(loan){offered=loan->work.payload;
                     reply.accepted_generation=loan->generation;}
            return reply;
        };
        if(!market.add_engine(producer)||!market.add_host(peer))return false;
        nodus::contracts::Contract cold{};
        cold.id=101;cold.engine=42;cold.payload=1;
        auto warm=cold;warm.id=102;warm.payload=2;
        if(!market.submit(cold)||!market.submit(warm))return false;
        market.market_turn();
        return offered==expected;
    };
    if(!market_choice(2))return 40;
    cold_expiry=1050;
    if(offer(peer,pending)!=0)return 39; // deadline pressure beats residency
    if(!market_choice(1))return 41;
    cold_expiry=5000;
    pending[1].state=nodus::contracts::ContractNegotiator::State::Loaned;
    if(offer(peer,pending)!=2)return 36; // only other warm entry is ready
    pending[1].state=nodus::contracts::ContractNegotiator::State::Unaccepted;
    pending[0].state=nodus::contracts::ContractNegotiator::State::Loaned;
    if(offer(peer,pending)!=1)return 37; // warm tie retains FIFO
    peer.id=501;
    if(offer(peer,pending)!=SIZE_MAX)return 38;
    for(int sample=0;sample<3;sample++){
        library->record_work_completion(
            nodus::gpu::TensorStateLibraryTool::ExecutionApi::Vulkan,
            0,99123,64,100,5000,5100);
        library->record_work_completion(
            nodus::gpu::TensorStateLibraryTool::ExecutionApi::Vulkan,
            1,99123,64,100,3000,3100);
        library->record_work_completion(
            nodus::gpu::TensorStateLibraryTool::ExecutionApi::Vulkan,
            2,99123,64,100,1000,1100);
    }
    const auto measured=nodus::gpu::resident_fabric_offer(*library,
        [](std::uint64_t host)->std::optional<std::size_t>{
            if(host>=500&&host<=502)return static_cast<std::size_t>(host-500);
            return {};
        },[&](std::uint64_t payload)->std::optional<nodus::gpu::FabricStateRef>{
            if(payload!=4)return {};
            return nodus::gpu::FabricStateRef{card,base+100,1,5000,100,
                                             99123,64};
        },[]{return std::uint64_t{1000};},{{500,0},{501,1},{502,2}});
    std::vector<Entry> measured_pending(1);
    measured_pending[0].work.payload=4;
    peer.id=500;if(measured(peer,measured_pending)!=SIZE_MAX)return 42;
    peer.id=501;if(measured(peer,measured_pending)!=SIZE_MAX)return 43;
    peer.id=502;if(measured(peer,measured_pending)!=0)return 44;
    {
        nodus::contracts::ContractNegotiator market(3,measured,10,false);
        nodus::contracts::Producer producer{};producer.id=77;
        producer.route=[](const nodus::contracts::Result&){return true;};
        producer.request=[](nodus::contracts::Contract&){return false;};
        if(!market.add_engine(producer))return 48;
        std::vector<std::uint64_t> accepted_hosts;
        for(std::uint64_t host=500;host<=502;host++){
            nodus::contracts::Peer candidate{};candidate.id=host;
            candidate.exchange=[host,&accepted_hosts](const nodus::contracts::Loan* loan){
                nodus::contracts::Exchange reply{};
                if(loan){accepted_hosts.push_back(host);
                         reply.accepted_generation=loan->generation;}
                return reply;
            };
            if(!market.add_host(candidate))return 49;
        }
        nodus::contracts::Contract work{};work.engine=77;work.payload=4;
        work.id=201;if(!market.submit(work))return 50;
        for(int i=0;i<3;i++)market.market_turn();
        if(accepted_hosts!=std::vector<std::uint64_t>{502})return 51;
        work.id=202;if(!market.submit(work))return 52;
        for(int i=0;i<3;i++)market.market_turn();
        if(accepted_hosts!=std::vector<std::uint64_t>{502,501})return 53;
    }
    {
        auto warm=library->borrow_gpu(card,base+100,0,false);
        if(!warm)return 45;
    }
    peer.id=500;if(measured(peer,measured_pending)!=0)return 46;
    peer.id=502;if(measured(peer,measured_pending)!=SIZE_MAX)return 47;
    request={};request[0]=4;
    const auto restores_before=send(*tool,command_edge,payload_edge,
                                    reply_edge,request,dummy)[26];
    {
        auto relend=library->borrow_gpu(card,receipt[2],0,false);
        if(!relend)return 32;
    }
    request={};request[0]=4;
    if(send(*tool,command_edge,payload_edge,reply_edge,request,dummy)[26]!=
       restores_before)return 33;
    request={};request[0]=3;request[1]=card;request[3]=base+100;
    if(send(*tool,command_edge,payload_edge,reply_edge,request,dummy)[0])return 31;
    for(int i=0;i<27;i++){
        request={};request[0]=3;request[1]=card;request[3]=base+i;
        receipt=send(*tool,command_edge,payload_edge,reply_edge,request,dummy);
        if(receipt[0]!=0)return 21;
    }
    nodus_tensor_destroy(data);nodus_tensor_destroy(dummy);
    const auto device_rows=library->device_count();
    tool->shutdown();
    std::cout<<"registered_tool=pass fifo_control_payload_receipt=pass"
             <<" device_rows="<<device_rows
             <<" card_lease_cross_device=pass"
             <<" gpu_dispatch_ns="<<elapsed[0]<<','<<elapsed[1]<<','<<elapsed[2]
             <<" cpu_dispatch_ns="<<cpu_ns
             <<" common_result_fifo=pass\n";
    return 0;
}
