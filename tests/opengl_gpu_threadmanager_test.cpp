// Real ThreadManager -> typed native FIFOs -> registered GPU ITool. GL only is
// mocked. No GL context, shader device operation, canvas or renderer is created.
#include "opengl_compute_mock.h"
#include "opengl_gpu_graph_binding.h"
#include <array>

namespace {
class TestTool final:public ITool {
public:
    explicit TestTool(std::function<void(ToolStackContext&)> f):fn(std::move(f)){}
    void initialize(const ToolInitContext&)override{} void shutdown()override{}
    void tick(double,HostAPI&)override{}void render(RenderContext&)override{}
    void execute_stack(ToolStackContext& c)override{fn(c);}
private:std::function<void(ToolStackContext&)> fn;
};
using Table=std::unique_ptr<GP_TableContext,decltype(&gp_table_destroy)>;
Table make_table(){Table t(gp_table_create(nullptr),gp_table_destroy);expect(bool(t),"native table");gp_table_set_sim_enabled(t.get(),0);return t;}
uint64_t contact_key(int module,int contact){return uint64_t(module)<<32 | uint64_t(contact)<<16;}
void io(GraphRuntime& graph,int module,ModuleRowKind kind,int contact){
    graph.module_io_rows[module].push_back({kind,contact,ModuleToolKind::None,1});
    graph.module_plugin_instances[module].emplace_back(nullptr,[](ITool*){});
}
void tool(GraphRuntime& graph,int module,std::function<void(ToolStackContext&)> fn){
    ModuleIORow row;row.tool_origin=ModuleToolOrigin::Plugin;
    graph.module_io_rows[module].push_back(row);
    graph.module_plugin_instances[module].emplace_back(new TestTool(std::move(fn)),[](ITool* p){delete p;});
}
}
int main(){
    using namespace nodus::gpu;using namespace nodus::contracts;
    try{
        reset();current=0;bool protected_process=false;
        auto work=std::make_shared<Work>();work->objects={{"native source object","#version 430\nvoid main(){}\n"}};
        Config cfg;cfg.host_id=501;cfg.context=context;
        cfg.lookup=[&](const Contract& c)->std::shared_ptr<const Work>{return c.payload==99?work:nullptr;};
        cfg.protected_link_permitted=[&](std::string& why){why="No native protected context worker installed (CPU mock test)";return protected_process;};
        auto host=std::make_shared<OpenGLGpuHost>(cfg);
        ContractNegotiator market(2,first_suitable_free({host}),100);
        expect(host->bind_contract_owner(market),"bind existing native owner");
        int commits=0;
        expect(market.add_host(host->peer()) && market.add_engine({17,[&](const Result& r){
            expect(r.loan.work.id==9 && r.payload==uint64_t(State::Linked),"native FIFO result exact identity");++commits;return true;
        },[](Contract&){return false;}}) && market.submit({9,17,8,99}),"canonical work routing");
        ToolRegistry registry;expect(register_opengl_gpu_tool(registry,host,"native.gpu.test"),"GPU tool factory");
        auto root=make_table();std::array<Table,3> tables{make_table(),make_table(),make_table()};
        GraphRuntime graph;graph.module_io_rows.resize(3);graph.module_plugin_instances.resize(3);
        ThreadManager::TickRequest request;request.root_table=root.get();
        for(int i=0;i<3;++i){ThreadManager::ModuleContract module;module.module_idx=i;module.table=tables[i].get();
            module.sim_enabled=0;module.exec_mode=i==2?ThreadManager::ExecMode::Pooled:ThreadManager::ExecMode::Sequential;
            module.stack_contract=ModuleStackContract::AbstractTensorPointer;request.modules.push_back(module);}
        const auto types=register_port_types();
        for(int i=0;i<2;++i){
            expect(gp_table_add_edge(root.get(),contact_key(i,1),contact_key(i+1,0))!=0,"real typed FIFO edge");
            GP_TableEdgeTensorSpecTyped spec{};spec.dims[0]=1;spec.dim_count=1;spec.slots=4;
            spec.elem_size=i?sizeof(Telemetry):sizeof(Control);spec.type_id=i?types.telemetry:types.control;
            expect(gp_table_edge_set_tensor_spec(root.get(),i,&spec)!=0,"exact token layout");
            expect(gp_table_edge_subscribe_ex(root.get(),i,contact_key(i+1,0),1)!=0,"subscribe actual FIFO before producer");
            ThreadManager::EdgeContract edge;edge.edge_idx=i;edge.a_module=i;edge.a_contact_idx=1;edge.b_module=i+1;edge.b_contact_idx=0;
            request.edges.push_back(edge);
        }
        std::string reason;
        expect(bind_opengl_gpu_module(graph,registry,"native.gpu.test",request.modules[1],root.get(),0,1,0,1,&reason),reason.c_str());
        expect(request.modules[1].stack_contract==ModuleStackContract::AbstractTensorPointer &&
            request.modules[1].exec_mode==ThreadManager::ExecMode::Sequential,"binding installs actual typed manager-lane contract");
        const auto caller=std::this_thread::get_id();std::thread::id manager_lane;
        int tick=0,received=0;
        tool(graph,0,[&](ToolStackContext& c){
            if(!tick)manager_lane=std::this_thread::get_id();
            expect(manager_lane==std::this_thread::get_id() && manager_lane!=caller,"persistent canonical manager lane");
            Control command{};
            if(tick==1)current=77; // simulated native context availability, NOT real activation
            if(tick==2){protected_process=true;market.market_turn();command={uint64_t(Command::Advance),9};}
            if(tick==3){market.market_turn();market.worker_turn();expect(host->forget_committed(9,market),"retire after native engine commit");}
            if(tick>2)command.contract_id=9;
            expect(raw_stack_push_typed(*c.stack.raw,&command,types.control),"typed control producer");
        });io(graph,0,ModuleRowKind::Output,1);
        io(graph,2,ModuleRowKind::Input,0);tool(graph,2,[&](ToolStackContext& c){
            expect(std::this_thread::get_id()!=manager_lane,"telemetry consumer can be pooled");
            Telemetry value{};expect(raw_stack_pop_typed(*c.stack.raw,&value,types.telemetry),"typed telemetry arrives via real edge");
            expect(value.version==2 && value.host_id==501 && !value.dispatch_calls && !value.execution_complete &&
                !value.physical_utilization_available,"honest typed GPU telemetry");
            if(tick==0)expect(value.refusal==uint64_t(Refusal::ContextUnavailable) && count("compile")==0,"control works without GL context");
            if(tick==1)expect(value.refusal==uint64_t(Refusal::SupervisionUnavailable) && count("compile")==0,"context alone never authorizes link");
            if(tick>=2)expect(value.state==uint64_t(State::Linked) && value.link_succeeded && value.program_released &&
                value.link_attempted && value.gl_cleanup_verified,"link-only outcome");
            if(tick>=3)expect(value.result_fifo_accepted && !value.harness_busy,"FIFO acceptance separate from initial link");
            ++received;
        });
        auto& manager=*graph.thread_mgr;manager.start();
        for(tick=0;tick<8;++tick)manager.submit_tick(request,true);
        expect(received==8 && commits==1 && count("compile")==1 && count("link")==1 && !dispatch_count && programs.empty(),
            "actual scheduler submits link once and no shader dispatch");
        manager.stop();
        std::cout<<"PASS: real ThreadManager manager="<<manager_lane<<" caller="<<caller
            <<" -> typed Control FIFO -> registered GPU ITool -> typed Telemetry FIFO; GL mocked, real context/supervisor absent\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
