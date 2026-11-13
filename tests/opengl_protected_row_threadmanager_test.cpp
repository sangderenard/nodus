// Actual ThreadManager/FIFOs/row/provider/Windows child integration. The child
// callback is CPU-only. No WGL implementation is linked or invoked here.
#include "opengl_compute_mock.h"
#include "opengl_gpu_graph_binding.h"
#include "kernels/opengl_protected_protocol.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <array>

namespace {
using namespace nodus::gpu;
class TestTool final:public ITool {
public:
    explicit TestTool(std::function<void(ToolStackContext&)> f):fn(std::move(f)){}
    void initialize(const ToolInitContext&)override{}void shutdown()override{}
    void tick(double,HostAPI&)override{}void render(RenderContext&)override{}
    void execute_stack(ToolStackContext& c)override{fn(c);}
private:std::function<void(ToolStackContext&)> fn;
};
using Table=std::unique_ptr<GP_TableContext,decltype(&gp_table_destroy)>;
Table table(){Table t(gp_table_create(nullptr),gp_table_destroy);expect(bool(t),"table");gp_table_set_sim_enabled(t.get(),0);return t;}
uint64_t key(int module,int contact){return uint64_t(module)<<32|uint64_t(contact)<<16;}
void io(GraphRuntime& graph,int module,ModuleRowKind kind,int contact){
    graph.module_io_rows[module].push_back({kind,contact,ModuleToolKind::None,1});
    graph.module_plugin_instances[module].emplace_back(nullptr,[](ITool*){});
}
void tool(GraphRuntime& graph,int module,std::function<void(ToolStackContext&)> fn){
    ModuleIORow row;row.tool_origin=ModuleToolOrigin::Plugin;graph.module_io_rows[module].push_back(row);
    graph.module_plugin_instances[module].emplace_back(new TestTool(std::move(fn)),[](ITool* p){delete p;});
}
ProtectedLinkOutcome cpu_child(const protected_detail::ChildRequest& request) {
    if(request.link.source_count!=1 || request.link.sources[0].source_bytes!=25 ||
       std::memcmp(request.link.sources[0].source,"CPU ONLY immutable source",25))ExitProcess(81);
    Sleep(100); // ensure control/telemetry can progress during the owned child
    ProtectedLinkOutcome out;out.status=ProtectedLinkStatus::Linked;
    out.attempted=out.linked=out.identity_verified=out.resources_released=true;
    out.gl_cleanup_verified=true;out.observed_luid=request.adapter_luid;
    out.context_selection=request.context_policy==ProtectedContextPolicy::VerifiedDefaultDc?
        ProtectedContextSelection::VerifiedDefaultDc:ProtectedContextSelection::NvAffinity;
    out.message="CPU fixture completed; no native driver call";return out;
}
uintptr_t forbidden_context(void*) {throw std::runtime_error("parent context queried on protected path");}
void* forbidden_proc(const char*,void*) {throw std::runtime_error("parent GL queried on protected path");}
}
int wmain(int argc,wchar_t** argv) {
    if(argc>1)return nodus::gpu::protected_detail::protected_child_main(argc,argv,cpu_child);
    using namespace nodus::gpu;using namespace nodus::contracts;
    try {
        reset();wchar_t executable[32768]{};expect(GetModuleFileNameW(nullptr,executable,32768)!=0,"self executable");
        ProtectedProcessOptions options;options.executable=executable;options.adapter_luid={1,2,3,4,5,6,7,8};options.node_mask=1;
        options.process_memory_bytes=512ull*1024*1024;options.job_memory_bytes=512ull*1024*1024;options.wall_timeout_ms=5000;
        options.context_policy=ProtectedContextPolicy::VerifiedDefaultDc;
        auto provider=std::make_shared<ProtectedOpenGLProcess>(options);
        auto work=std::make_shared<Work>();work->objects={{"CPU fixture object","CPU ONLY immutable source"}};
        Config config;config.host_id=501;config.context={77,forbidden_proc,forbidden_context,nullptr};config.protected_provider=provider;
        config.lookup=[&](const Contract& c)->std::shared_ptr<const Work>{return c.payload==99?work:nullptr;};
        auto host=std::make_shared<OpenGLGpuHost>(config);
        ContractNegotiator market(2,first_suitable_free({host}),1);
        expect(host->bind_contract_owner(market),"bind contract owner");
        int commits=0;expect(market.add_host(host->peer()) && market.add_engine({17,[&](const Result& result){
            expect(result.loan.work.id==9 && result.payload==uint64_t(State::Linked),"exact result FIFO outcome");++commits;return true;
        },[](Contract&){return false;}}) && market.submit({9,17,8,99}),"canonical source contract");
        ToolRegistry registry;expect(register_opengl_gpu_tool(registry,host,"native.protected.cpu"),"row registry");
        auto root=table();std::array<Table,3> tables{table(),table(),table()};
        GraphRuntime graph;graph.module_io_rows.resize(3);graph.module_plugin_instances.resize(3);
        ThreadManager::TickRequest tick_request;tick_request.root_table=root.get();
        for(int i=0;i<3;++i){ThreadManager::ModuleContract module;module.module_idx=i;module.table=tables[i].get();module.sim_enabled=0;
            module.exec_mode=i==2?ThreadManager::ExecMode::Pooled:ThreadManager::ExecMode::Sequential;
            module.stack_contract=ModuleStackContract::AbstractTensorPointer;tick_request.modules.push_back(module);}
        const auto types=register_port_types();
        for(int i=0;i<2;++i){
            expect(gp_table_add_edge(root.get(),key(i,1),key(i+1,0))!=0,"FIFO edge");
            GP_TableEdgeTensorSpecTyped spec{};spec.dims[0]=1;spec.dim_count=1;spec.slots=4;
            spec.elem_size=i?sizeof(Telemetry):sizeof(Control);spec.type_id=i?types.telemetry:types.control;
            expect(gp_table_edge_set_tensor_spec(root.get(),i,&spec)!=0,"exact typed FIFO");
            expect(gp_table_edge_subscribe_ex(root.get(),i,key(i+1,0),1)!=0,"FIFO subscriber");
            ThreadManager::EdgeContract edge;edge.edge_idx=i;edge.a_module=i;edge.a_contact_idx=1;edge.b_module=i+1;edge.b_contact_idx=0;
            tick_request.edges.push_back(edge);
        }
        std::string error;
        expect(bind_opengl_gpu_module(graph,registry,"native.protected.cpu",tick_request.modules[1],root.get(),0,1,0,1,&error),error.c_str());
        const auto caller=std::this_thread::get_id();std::thread::id lane;
        int tick=0,received=0,running_ticks=0;uint64_t generation=0,max_tick_ms=0;Telemetry latest{};
        tool(graph,0,[&](ToolStackContext& context){
            if(!tick)lane=std::this_thread::get_id();
            expect(lane==std::this_thread::get_id() && lane!=caller,"canonical manager lane");
            Control command{};
            if(tick==1){market.market_turn();generation=host->telemetry().generation;expect(generation!=0,"actual assigned generation");}
            if(tick>=1 && tick<5){market.market_turn();expect(host->telemetry().generation==generation,"same approval-wait generation");command={uint64_t(Command::Advance),9};}
            if(tick==5){
                gp_gl_compute_source source{work->objects[0].label.c_str(),work->objects[0].source.data(),work->objects[0].source.size()};
                ProtectedSourceDigest digest{};ProtectedLinkRequest request{501,9,generation,&source,1};
                expect(protected_source_digest(request,digest,error),error.c_str());
                expect(provider->authorize_once({501,9,generation,digest},error),error.c_str());
            }
            if(tick>=5)command={uint64_t(Command::Advance),9};
            if(latest.state==uint64_t(State::Linked)){
                if(!commits){market.market_turn();market.worker_turn();expect(host->forget_committed(9,market),"retire committed result");}
                command={uint64_t(Command::Query),9};
            }
            expect(raw_stack_push_typed(*context.stack.raw,&command,types.control),"Control producer");
        });io(graph,0,ModuleRowKind::Output,1);
        io(graph,2,ModuleRowKind::Input,0);tool(graph,2,[&](ToolStackContext& context){
            expect(raw_stack_pop_typed(*context.stack.raw,&latest,types.telemetry),"Telemetry FIFO consumer");
            expect(latest.host_id==501 && latest.version==2 && !latest.context_current && !latest.dispatch_calls && !latest.execution_complete,"no parent GL/execution claim");
            if(tick>=1 && tick<5)expect(latest.state==uint64_t(State::Queued) && !latest.process_id && !latest.link_attempted,"approval wait launched child");
            if(latest.state==uint64_t(State::Linking))++running_ticks;
            if(latest.state==uint64_t(State::Linked))expect(latest.containment_verified && latest.child_reaped && latest.identity_verified &&
                latest.process_resources_released && latest.gl_cleanup_verified && latest.process_id && !latest.outcome_uncertain,"complete protected fixture proof");
            ++received;
        });
        auto& manager=*graph.thread_mgr;manager.start();
        for(tick=0;tick<500 && !commits;++tick){const auto before=GetTickCount64();manager.submit_tick(tick_request,true);
            max_tick_ms=std::max<uint64_t>(max_tick_ms,GetTickCount64()-before);Sleep(5);}
        manager.stop();
        expect(commits==1 && received==tick && running_ticks>=2 && latest.result_fifo_accepted && !latest.harness_busy,"progress + single FIFO commit");
        expect(max_tick_ms<500 && count("compile")==0 && count("link")==0 && !dispatch_count,"nonblocking manager and zero local GPU calls");
        std::cout<<"PASS: canonical manager="<<lane<<" caller="<<caller<<" child="<<latest.process_id
            <<" ticks="<<tick<<" running_ticks="<<running_ticks<<" max_tick_ms="<<max_tick_ms
            <<" -> actual protected CPU child -> typed Telemetry FIFO -> exact result/commit; zero GPU calls\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
}
