// CPU-only supplied inventory. Real registry, tools, contract FIFO, module
// library serializer and graph binding; no enumeration/process/context/GPU.
#include "opengl_compute_mock.h"
#include "gpu_inventory_rows.h"
#include <algorithm>

namespace {
using namespace nodus::gpu;using namespace nodus::contracts;
struct Provider final:ProtectedLinkProvider {
    bool available_now=true;int calls=0;std::array<uint8_t,8> target{};
    bool available(std::string& why)const override {why=available_now?"":"CPU provider unavailable";return available_now;}
    bool ready(std::string& why)const override {why="Exact operation remains unapproved";return false;}
    ProtectedLinkOutcome link(const ProtectedLinkRequest&)override {++calls;throw std::runtime_error("No provider work authorized by inventory tests");}
};
InventoryRowConfig device(uint8_t uuid,uint8_t luid,uint32_t ordinal,uint64_t host,bool configured) {
    InventoryRowConfig out;out.host_id=host;out.device.uuid[0]=uuid;out.device.luid[0]=luid;
    out.device.uuid_valid=out.device.luid_valid=true;out.device.index=ordinal;out.device.name="CPU inventory "+std::to_string(uuid);
    if(configured){out.process.emplace();out.process->node_mask=1;out.process->executable=L"C:\\CPU-fixture-never-run.exe";}
    return out;
}
Telemetry control(ITool& tool,Control command) {
    const auto types=register_port_types();RawStackFrame frame{};expect(raw_stack_init_frame(frame,1024),"typed frame");
    expect(raw_stack_push_typed(frame,&command,types.control),"typed Control");ToolStackContext ctx;ctx.stack.raw=&frame;tool.execute_stack(ctx);
    Telemetry out{};expect(ctx.succeeded && raw_stack_pop_typed(frame,&out,types.telemetry),"typed Telemetry");
    raw_stack_free_mask(frame);gp_mem_backend_release(frame.backend);return out;
}
void turn(ITool& tool){ToolStackContext context;tool.execute_stack(context);expect(context.succeeded,"canonical market/worker row");}
using Table=std::unique_ptr<GP_TableContext,decltype(&gp_table_destroy)>;
Table table(){Table out(gp_table_create(nullptr),gp_table_destroy);expect(bool(out),"native table");gp_table_set_sim_enabled(out.get(),0);return out;}
}
int main(int argc,char** argv) {
    try {
        reset();const std::string output=argc>1?argv[1]:"gpu-inventory-module-library.test";
        auto shader=std::make_shared<Work>();shader->objects={{"unchanged","CPU source, never compiled"}};
        auto unsupported=std::make_shared<Work>(*shader);unsupported->operation=Operation::Execute;
        auto lookup=[&](const Contract& c)->std::shared_ptr<const Work>{return c.payload==2?unsupported:shader;};
        std::vector<std::shared_ptr<Provider>> providers;
        auto factory=[&](const ProtectedProcessOptions& options)->std::shared_ptr<ProtectedLinkProvider>{
            auto provider=std::make_shared<Provider>();provider->target=options.adapter_luid;providers.push_back(provider);return provider;
        };
        std::vector<InventoryRowConfig> configs{device(1,11,0,101,false),device(2,22,1,102,true),device(3,33,2,103,true)};
        auto fleet=GpuInventoryRows::create("inventory.cpu",configs,lookup,8,2,factory);
        expect(providers.size()==2 && providers[0]->target==configs[1].device.luid && providers[1]->target==configs[2].device.luid,
            "native provider targets exact supplied LUID, not Vulkan ordinal");
        const auto row0=fleet->rows()[0].tool_id,row1=fleet->rows()[1].tool_id,row2=fleet->rows()[2].tool_id;
        auto first=tool_registry_global().create(row0),second=tool_registry_global().create(row1),third=tool_registry_global().create(row2);
        const auto unavailable=control(*first,{});
        expect(unavailable.refusal==uint64_t(Refusal::SupervisionUnavailable) && std::string(unavailable.message).find("No protected")!=std::string::npos,
            "unconfigured inventory is visible unavailable telemetry, not GL readiness");
        const auto available=fleet->status(row1);
        expect(available.uuid==configs[1].device.uuid && available.inventory_luid==configs[1].device.luid &&
            available.provider_available && !available.physical_utilization_available && !available.operation.identity_verified,
            "inventory identity, structural provider availability and hardware proof remain separate");
        auto market_tool=tool_registry_global().create("inventory.cpu.market"),worker_tool=tool_registry_global().create("inventory.cpu.worker");
        auto& market=fleet->negotiator();
        expect(&acquire("inventory.cpu",8,2)==&market,"canonical saved-prefix acquire preserves custom Offer");
        bool replaced=false;try{create_with_offer("inventory.cpu",8,[](const Peer&,const std::vector<ContractNegotiator::Entry>&){return size_t(0);});}
        catch(const std::runtime_error&){replaced=true;}expect(replaced,"cannot replace live custom Offer");
        std::unique_ptr<NodusEdgeRuntime,decltype(&nodus_edge_runtime_destroy)> fifo(nodus_edge_runtime_create(),nodus_edge_runtime_destroy);
        const int32_t dims=1;expect(fifo && nodus_edge_runtime_configure(fifo.get(),&dims,1,8,0,sizeof(Result),0x534A5231,2,8) &&
            nodus_edge_runtime_subscribe(fifo.get(),1,1),"actual native result FIFO");
        expect(market.bind_result_transport([&](const Result& result){int32_t dropped=0;return nodus_edge_runtime_publish(fifo.get(),1,&result,sizeof(result),&dropped)!=0 && !dropped;},
            [&](Result& result){size_t bytes=0;return nodus_edge_runtime_consume(fifo.get(),1,&result,sizeof(result),&bytes)!=0 && bytes==sizeof(result);}),"canonical result transport");
        int commits=0;expect(market.add_engine({17,[&](const Result& result){expect(result.payload==uint64_t(State::Cancelled),"only unexecuted cancellation results");++commits;return true;},[](Contract&){return false;}}),"native engine");
        expect(market.submit({90,17,1,2}) && market.submit({91,17,1,1}) && market.submit({92,17,1,1}) && market.submit({93,17,1,1}),"native input contracts");
        turn(*market_tool);
        expect(!fleet->rows()[0].host->busy() && market.contracts()[0].loan.generation==0 && market.contracts()[1].loan.generation==0,
            "unsuitable row does not reserve any work");
        turn(*market_tool);expect(fleet->rows()[1].host->telemetry().contract_id==91,"first structurally suitable free row");
        turn(*market_tool);expect(fleet->rows()[2].host->telemetry().contract_id==92,"next free row, busy row not reused");
        const auto generation=market.contracts()[1].loan.generation;
        for(int i=0;i<6;++i)turn(*market_tool);
        expect(market.contracts()[0].state==ContractNegotiator::State::Unaccepted && market.contracts()[3].state==ContractNegotiator::State::Unaccepted &&
            market.contracts()[1].loan.generation==generation,"unsupported program skipped and held busy ownership never reoffered");
        std::string reason;expect(!fleet->close(reason),"cannot unregister unfinished native ownership");
        expect(market.remove_host(102),"detach existing callback");for(int i=0;i<4;++i)turn(*market_tool);
        expect(market.contracts()[1].loan.host==102 && market.contracts()[1].loan.generation==generation,
            "removal does not reroute protected accepted identity");
        expect(market.add_host(fleet->rows()[1].host->peer()),"reattach same retained native owner");
        control(*second,{uint64_t(Command::CancelQueued),91});control(*third,{uint64_t(Command::CancelQueued),92});
        for(int i=0;i<6 && commits<2;++i){turn(*market_tool);turn(*worker_tool);}
        expect(commits==2 && fleet->rows()[1].host->forget_committed(91,market) && fleet->rows()[2].host->forget_committed(92,market),"native FIFO cancellation commit");
        GP_ModuleLibrary library;library.root_dir="CPU inventory fixture";
        for(size_t i=0;i<fleet->rows().size();++i){const auto& id=fleet->rows()[i].tool_id;library.tool_registry.push_back(fleet->library_tool(id));library.modules.push_back(fleet->library_module(id,int(i)));}
        expect(gp_module_library_write_to_file(library,output.c_str()),"existing module library serializer");
        GP_ModuleLibrary restored;expect(gp_module_library_read_from_file(output.c_str(),&restored) && restored.modules.size()==3,"existing module library reader");
        expect(market.remove_engine(17),"remove unsubmitted unsupported/test tail only after committed outcomes");
        first.reset();second.reset();third.reset();market_tool.reset();worker_tool.reset();expect(fleet->close(reason),reason.c_str());fleet.reset();
        configs[1].device.index=8;configs[2].device.index=0;std::rotate(configs.begin(),configs.begin()+2,configs.end());
        fleet=GpuInventoryRows::create("inventory.cpu",configs,lookup,8,2,factory);
        const auto* reidentified=fleet->find_tool(restored.modules[1].tool_instances[0].tool_id);
        expect(reidentified && reidentified->device.uuid[0]==2 && reidentified->device.index==8 && reidentified->host_id==102,
            "persisted UUID factory rebinds same device after ordinal/order changes");
        {
            auto root=table();auto module_table=table();GraphRuntime graph;ThreadManager::ModuleContract module;module.module_idx=1;module.table=module_table.get();
            const auto types=register_port_types();
            for(int i=0;i<2;++i){expect(gp_table_add_edge(root.get(),uint64_t(i)<<32|(uint64_t(1)<<16),uint64_t(i+1)<<32)!=0,"binding edge");
                GP_TableEdgeTensorSpecTyped spec{};spec.dim_count=1;spec.dims[0]=1;spec.slots=4;spec.elem_size=i?sizeof(Telemetry):sizeof(Control);spec.type_id=i?types.telemetry:types.control;
                expect(gp_table_edge_set_tensor_spec(root.get(),i,&spec)!=0,"typed binding metadata");}
            expect(fleet->bind_module(reidentified->tool_id,graph,module,root.get(),0,1,0,1,&reason),reason.c_str());
            graph.module_stack_contract[1]=ModuleStackContract::Opaque;graph.module_exec_mode[1]=int(ThreadManager::ExecMode::Pooled);module={};module.module_idx=1;
            expect(fleet->restore_module(reidentified->tool_id,graph,module,root.get(),0,1,0,1,&reason) &&
                module.stack_contract==ModuleStackContract::AbstractTensorPointer && module.exec_mode==ThreadManager::ExecMode::Sequential,
                "explicit post-load restore reinstates typed canonical manager lane");
            expect(!fleet->restore_module(row0,graph,module,root.get(),0,1,0,1,&reason),"wrong saved identity cannot take another row");
            auto correct=std::move(graph.module_plugin_instances[1][1]);
            graph.module_plugin_instances[1][1]=tool_registry_global().create(row0);
            expect(!fleet->restore_module(reidentified->tool_id,graph,module,root.get(),0,1,0,1,&reason),
                "unchanged UUID metadata cannot disguise an actual instance from another GPU");
            graph.module_plugin_instances[1][1]=std::move(correct);
            expect(!fleet->close(reason) && reason.find("instances")!=std::string::npos,
                "close refuses live bound GPU instances before removing any factory");
            GP_TableEdgeTensorSpecTyped wrong{};wrong.dim_count=1;wrong.dims[0]=1;wrong.slots=4;wrong.elem_size=sizeof(Telemetry);wrong.type_id=types.control;
            expect(gp_table_edge_set_tensor_spec(root.get(),1,&wrong)!=0 &&
                !fleet->restore_module(reidentified->tool_id,graph,module,root.get(),0,1,0,1,&reason),"stale persisted type ID refused, not reinterpreted");
        }
        expect(fleet->close(reason),reason.c_str());fleet.reset();
        auto duplicate=configs;duplicate[1].device.uuid=duplicate[0].device.uuid;bool refused=false;
        try{auto invalid=GpuInventoryRows::create("duplicate.cpu",duplicate,lookup,8,2,factory);}catch(const std::invalid_argument&){refused=true;}
        expect(refused && !nodus::contracts::find("duplicate.cpu"),"duplicate physical identity refused before registration");
        for(const auto& provider:providers)expect(!provider->calls,"inventory routing never starts native work");
        expect(calls.empty() && !dispatch_count,"no local GL or shader API calls");
        std::cout<<"PASS: supplied UUID inventory -> existing factories/custom Offer/native FIFO; unsuitable/busy exclusion, no replay, native config identity roundtrip, actual-instance provenance and explicit graph restoration; zero GPU/process calls\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;}
}
