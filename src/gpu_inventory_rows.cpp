#include "gpu_inventory_rows.h"
#include <set>
#include <cstdio>
#include <map>
#include <mutex>

namespace nodus::gpu {
// Factory provenance follows the actual instance lifetime. Saved metadata alone
// cannot establish which host a pre-existing OpenGLGpuTool actually owns.
struct GpuInventoryRows::InstanceBindings {
    std::mutex mutex;
    std::map<ITool*,const OpenGLGpuHost*> hosts;
    bool matches(ITool* tool,const OpenGLGpuHost* host) {
        std::lock_guard<std::mutex> guard(mutex);
        const auto found=hosts.find(tool);return found!=hosts.end() && found->second==host;
    }
    bool empty() {std::lock_guard<std::mutex> guard(mutex);return hosts.empty();}
};
namespace {
template<size_t N> bool nonzero(const std::array<uint8_t,N>& value) {
    return std::any_of(value.begin(),value.end(),[](uint8_t x){return x!=0;});
}
std::string uuid_text(const std::array<uint8_t,16>& uuid) {
    const char* hex="0123456789abcdef";std::string text;text.reserve(32);
    for(auto byte:uuid){text.push_back(hex[byte>>4]);text.push_back(hex[byte&15]);}return text;
}
class UnavailableProvider final:public ProtectedLinkProvider {
public:
    explicit UnavailableProvider(std::string reason):reason_(std::move(reason)){}
    bool available(std::string& reason)const override {reason=reason_;return false;}
    bool ready(std::string& reason)const override {reason=reason_;return false;}
    ProtectedLinkOutcome link(const ProtectedLinkRequest& request)override {
        ProtectedLinkOutcome out;out.host_id=request.host_id;out.contract_id=request.contract_id;out.generation=request.generation;
        out.resources_released=out.child_reaped=true;out.message=reason_;return out;
    }
private:std::string reason_;
};
bool fifo_matches(GP_TableContext* root,int edge,ValueTypeId type,size_t bytes) {
    GP_TableEdgeTensorSpecTyped spec{};
    return root && edge>=0 && gp_table_edge_get_tensor_spec(root,edge,&spec) &&
        spec.dim_count==1 && spec.dims[0]==1 && spec.elem_size==int32_t(bytes) &&
        spec.type_id==type && spec.slots>0 && spec.top_k==0;
}
}
std::unique_ptr<GpuInventoryRows> GpuInventoryRows::create(const std::string& prefix,
    const std::vector<InventoryRowConfig>& configs,
    std::function<std::shared_ptr<const Work>(const contracts::Contract&)> lookup,
    size_t capacity,uint64_t loan_ticks,ProviderFactory factory) {
    if(prefix.empty() || prefix.size()>128 || prefix.find_first_of(" \t\r\n")!=std::string::npos ||
       configs.empty() || configs.size()>64 || !lookup || !capacity || !loan_ticks)
        throw std::invalid_argument("invalid bounded GPU inventory registration");
    if(contracts::find(prefix))throw std::runtime_error("GPU contract prefix already exists; no policy/work replacement");
    std::set<std::array<uint8_t,16>> identities;std::set<uint64_t> hosts;
    for(const auto& config:configs) {
        if(!config.device.uuid_valid || !nonzero(config.device.uuid) || config.device.name.size()>1024 ||
           !config.host_id || !identities.insert(config.device.uuid).second || !hosts.insert(config.host_id).second)
            throw std::invalid_argument("GPU rows require distinct nonzero UUIDs and host identities");
        if(tool_registry_global().find(prefix+".gpu."+uuid_text(config.device.uuid)))
            throw std::runtime_error("GPU factory already registered");
        if(config.process && nonzero(config.process->adapter_luid) && config.process->adapter_luid!=config.device.luid)
            throw std::invalid_argument("configured OpenGL target differs from inventory LUID");
    }
    if(!factory)factory=[](const ProtectedProcessOptions& options){return std::make_shared<ProtectedOpenGLProcess>(options);};
    auto result=std::unique_ptr<GpuInventoryRows>(new GpuInventoryRows);result->prefix_=prefix;
    result->instances_=std::make_shared<InstanceBindings>();
    std::vector<std::shared_ptr<OpenGLGpuHost>> hosts_for_offer;
    for(const auto& config:configs) {
        InventoryRow row;row.device=config.device;row.host_id=config.host_id;
        row.tool_id=prefix+".gpu."+uuid_text(config.device.uuid);
        std::shared_ptr<ProtectedLinkProvider> provider;
        if(!config.device.luid_valid || !nonzero(config.device.luid))row.unavailable_reason="Windows adapter LUID is unavailable; Vulkan inventory does not prove OpenGL affinity";
        else if(!config.process)row.unavailable_reason="No protected OpenGL provider is configured for this device UUID";
        else if(!config.process->node_mask)row.unavailable_reason="Verified Windows device node mask is missing";
        else {
            auto options=*config.process;options.adapter_luid=config.device.luid;
            provider=factory(options);
            if(!provider)row.unavailable_reason="Protected OpenGL provider factory is unavailable";
        }
        if(!provider)provider=std::make_shared<UnavailableProvider>(row.unavailable_reason);
        Config native;native.host_id=row.host_id;native.lookup=lookup;native.protected_provider=provider;
        row.host=std::make_shared<OpenGLGpuHost>(std::move(native));
        hosts_for_offer.push_back(row.host);result->providers_.push_back(std::move(provider));result->rows_.push_back(std::move(row));
    }
    result->negotiator_=&contracts::create_with_offer(prefix,capacity,first_suitable_free(std::move(hosts_for_offer)),loan_ticks);
    size_t registered=0;
    try {
        for(auto& row:result->rows_) {
            if(!row.host->bind_contract_owner(*result->negotiator_))throw std::runtime_error("GPU native owner binding failed");
            ToolRegistry::Entry entry;entry.id=row.tool_id;entry.name="Native OpenGL GPU row";
            entry.factory=[owner=row.host,instances=result->instances_] {
                auto tool=std::make_unique<OpenGLGpuTool>(owner);
                // Allocate the deleter before recording identity. Once recorded,
                // transfer to the ordinary registry handle is nonthrowing.
                std::function<void(ITool*)> deleter=[instances](ITool* ptr) {
                    {std::lock_guard<std::mutex> guard(instances->mutex);instances->hosts.erase(ptr);}
                    delete ptr;
                };
                {std::lock_guard<std::mutex> guard(instances->mutex);instances->hosts.emplace(tool.get(),owner.get());}
                return std::unique_ptr<ITool,std::function<void(ITool*)>>(tool.release(),std::move(deleter));
            };
            if(!tool_registry_global().register_tool(std::move(entry)))throw std::runtime_error("GPU native factory binding failed");
            ++registered;
            if(!result->negotiator_->add_host(row.host->peer()))throw std::runtime_error("GPU native host registration failed");
        }
    } catch(...) {
        for(size_t i=0;i<registered;++i)tool_registry_global().unregister_tool(result->rows_[i].tool_id);
        contracts::release(prefix);result->negotiator_=nullptr;throw;
    }
    return result;
}
GpuInventoryRows::~GpuInventoryRows() {
    if(negotiator_)std::fputs("GPU inventory registration requires explicit close after graph teardown; canonical owners remain registered\n",stderr);
}
contracts::ContractNegotiator& GpuInventoryRows::negotiator()const {
    if(!negotiator_ || contracts::find(prefix_)!=negotiator_)throw std::runtime_error("GPU contract owner is no longer registered");
    return *negotiator_;
}
const InventoryRow* GpuInventoryRows::find_tool(const std::string& id)const {
    for(const auto& row:rows_)if(row.tool_id==id)return &row;return nullptr;
}
InventoryRowStatus GpuInventoryRows::status(const std::string& id)const {
    for(size_t i=0;i<rows_.size();++i)if(rows_[i].tool_id==id) {
        const auto& row=rows_[i];InventoryRowStatus out;out.host_id=row.host_id;out.uuid=row.device.uuid;
        out.inventory_luid=row.device.luid;out.vulkan_snapshot_index=row.device.index;
        out.harness_busy=row.host->busy();out.provider_available=providers_[i]->available(out.reason);
        out.operation=row.host->telemetry();return out;
    }
    throw std::invalid_argument("unknown GPU inventory tool identity");
}
GP_ModuleLibraryTool GpuInventoryRows::library_tool(const std::string& id)const {
    const auto* row=find_tool(id);if(!row)throw std::invalid_argument("unknown GPU tool");
    return {row->tool_id,ModuleToolKind::None,"GPU: "+row->device.name,{}};
}
GP_ModuleLibraryModule GpuInventoryRows::library_module(const std::string& id,int index)const {
    const auto* row=find_tool(id);if(!row || index<0)throw std::invalid_argument("unknown GPU module");
    GP_ModuleLibraryModule module;module.id=gp_module_library_module_id(index);module.module_idx=index;
    module.label="GPU: "+row->device.name;module.input_count=module.output_count=1;
    module.tool_instances.push_back({1,0,row->tool_id});return module;
}
bool GpuInventoryRows::bind_module(const std::string& id,GraphRuntime& graph,ThreadManager::ModuleContract& module,
    GP_TableContext* root,int control_edge,int telemetry_edge,int control_contact,int telemetry_contact,std::string* error)const {
    if(!find_tool(id)){if(error)*error="Persisted GPU UUID has no current inventory row";return false;}
    return bind_opengl_gpu_module(graph,tool_registry_global(),id,module,root,control_edge,telemetry_edge,control_contact,telemetry_contact,error);
}
bool GpuInventoryRows::restore_module(const std::string& id,GraphRuntime& graph,ThreadManager::ModuleContract& module,
    GP_TableContext* root,int control_edge,int telemetry_edge,int control_contact,int telemetry_contact,std::string* error)const {
    auto fail=[&](const char* text){if(error)*error=text;return false;};
    const auto* inventory=find_tool(id);
    if(!inventory || module.module_idx<0 || size_t(module.module_idx)>=graph.module_io_rows.size())return fail("Persisted GPU UUID/module is not registered");
    const auto types=register_port_types();
    if(control_edge==telemetry_edge || control_contact<0 || telemetry_contact<0 || control_contact==telemetry_contact ||
       !fifo_matches(root,control_edge,types.control,sizeof(Control)) || !fifo_matches(root,telemetry_edge,types.telemetry,sizeof(Telemetry)))
        return fail("Restored GPU FIFO types/extents do not match this native image; no reinterpretation");
    const auto index=size_t(module.module_idx);const auto& rows=graph.module_io_rows[index];
    if(rows.size()!=3 || rows[0].kind!=ModuleRowKind::Input || rows[0].contact_idx!=control_contact ||
       rows[1].kind!=ModuleRowKind::Tool || rows[1].tool_origin!=ModuleToolOrigin::Plugin || rows[1].plugin_id!=id ||
       rows[2].kind!=ModuleRowKind::Output || rows[2].contact_idx!=telemetry_contact ||
       index>=graph.module_plugin_instances.size() || graph.module_plugin_instances[index].size()!=3 ||
       dynamic_cast<OpenGLGpuTool*>(graph.module_plugin_instances[index][1].get())==nullptr ||
       !instances_->matches(graph.module_plugin_instances[index][1].get(),inventory->host.get()))
        return fail("Restored GPU row program/factory does not match its saved UUID");
    graph.module_stack_contract.resize(std::max(index+1,graph.module_stack_contract.size()),ModuleStackContract::Opaque);
    graph.module_exec_mode.resize(std::max(index+1,graph.module_exec_mode.size()),int(ThreadManager::ExecMode::Pooled));
    graph.module_io_in_count.resize(std::max(index+1,graph.module_io_in_count.size()),0);
    graph.module_io_out_count.resize(std::max(index+1,graph.module_io_out_count.size()),0);
    graph.module_stack_contract[index]=module.stack_contract=ModuleStackContract::AbstractTensorPointer;
    graph.module_exec_mode[index]=int(module.exec_mode=ThreadManager::ExecMode::Sequential);
    graph.module_io_in_count[index]=module.in_count=1;graph.module_io_out_count[index]=module.out_count=1;
    if(error)error->clear();return true;
}
bool GpuInventoryRows::close(std::string& reason) {
    if(!negotiator_){reason.clear();return true;}
    if(contracts::find(prefix_)!=negotiator_){reason="Canonical GPU contract owner was replaced externally";return false;}
    if(!instances_->empty()){reason="GPU factory instances are still bound; destroy the stopped graph/tools before close";return false;}
    for(const auto& entry:negotiator_->contracts())if(entry.state!=contracts::ContractNegotiator::State::Committed) {
        reason="GPU contract is unfinished; removing its owner would discard replay protection";return false;
    }
    for(const auto& row:rows_)if(!row.host->on_lane() || row.host->busy() || !row.host->shutdown()) {
        reason="GPU row still owns work/cleanup or is on another manager lane";return false;
    }
    for(const auto& row:rows_){negotiator_->remove_host(row.host_id);tool_registry_global().unregister_tool(row.tool_id);}
    if(!contracts::release(prefix_)){reason="Canonical GPU pair release failed";return false;}
    negotiator_=nullptr;reason.clear();return true;
}
} // namespace nodus::gpu
