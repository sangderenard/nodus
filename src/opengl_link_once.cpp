// Bounded native development entry for ONE explicitly approved link. The
// existing ThreadManager, GPU ITool, typed FIFOs and ContractNegotiator own the
// flow. No direct worker invocation, replacement scheduler or dispatch path.
#include "opengl_gpu_graph_binding.h"
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace {
using namespace nodus::gpu;
std::atomic<bool> cancelled=false;
BOOL WINAPI console_control(DWORD event) {
    if(event==CTRL_C_EVENT || event==CTRL_BREAK_EVENT){cancelled.store(true);return TRUE;}return FALSE;
}
void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
std::string utf8(const std::wstring& value) {
    if(value.empty())return {};
    const int count=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,value.data(),static_cast<int>(value.size()),nullptr,0,nullptr,nullptr);
    require(count>0,"Invalid Unicode argument");std::string out(count,'\0');
    require(WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,value.data(),static_cast<int>(value.size()),out.data(),count,nullptr,nullptr)==count,"UTF-8 argument conversion failed");return out;
}
uint64_t number(const std::wstring& value) {
    require(!value.empty() && value[0]!=L'-' && value[0]!=L'+',"Invalid unsigned number");size_t used=0;
    const auto out=std::stoull(value,&used,0);require(used==value.size() && out,"Expected a nonzero integer");return out;
}
template<size_t N> std::array<uint8_t,N> unhex(const std::wstring& value) {
    require(value.size()==2*N,"Hex field has wrong byte count");std::array<uint8_t,N> out{};
    const auto digit=[](wchar_t c)->uint8_t {
        if(c>=L'0' && c<=L'9')return uint8_t(c-L'0');if(c>=L'a' && c<=L'f')return uint8_t(c-L'a'+10);
        if(c>=L'A' && c<=L'F')return uint8_t(c-L'A'+10);throw std::runtime_error("Non-hexadecimal identity byte");
    };
    for(size_t i=0;i<N;++i)out[i]=uint8_t(digit(value[2*i])*16+digit(value[2*i+1]));return out;
}
template<size_t N> std::string hex(const std::array<uint8_t,N>& bytes) {
    constexpr char alphabet[]="0123456789abcdef";std::string out;out.reserve(2*N);
    for(auto byte:bytes){out+=alphabet[byte>>4];out+=alphabet[byte&15];}return out;
}
class NativeControlTool final:public ITool {
public:
    explicit NativeControlTool(std::function<void(ToolStackContext&)> fn):fn_(std::move(fn)){}
    void initialize(const ToolInitContext&)override{}void shutdown()override{}
    void tick(double,HostAPI&)override{}void render(RenderContext&)override{}
    void execute_stack(ToolStackContext& c)override{fn_(c);}
private:std::function<void(ToolStackContext&)> fn_;
};
using Table=std::unique_ptr<GP_TableContext,decltype(&gp_table_destroy)>;
Table make_table(){Table t(gp_table_create(nullptr),gp_table_destroy);require(bool(t),"Native table allocation failed");gp_table_set_sim_enabled(t.get(),0);return t;}
uint64_t contact(int module,int port){return uint64_t(module)<<32|uint64_t(port)<<16;}
void io(GraphRuntime& graph,int module,ModuleRowKind kind,int port){
    graph.module_io_rows[module].push_back({kind,port,ModuleToolKind::None,1});
    graph.module_plugin_instances[module].emplace_back(nullptr,[](ITool*){});
}
void control_tool(GraphRuntime& graph,int module,std::function<void(ToolStackContext&)> fn){
    ModuleIORow row;row.tool_origin=ModuleToolOrigin::Plugin;graph.module_io_rows[module].push_back(row);
    graph.module_plugin_instances[module].emplace_back(new NativeControlTool(std::move(fn)),[](ITool* p){delete p;});
}
void print_telemetry(const Telemetry& t) {
    std::cout<<"telemetry version="<<t.version<<" host="<<t.host_id<<" contract="<<t.contract_id<<" generation="<<t.generation
        <<" state="<<t.state<<" refusal="<<t.refusal<<" pid="<<t.process_id<<" exit="<<t.process_exit_code
        <<" attempted="<<t.link_attempted<<" linked="<<t.link_succeeded<<" child_reaped="<<t.child_reaped
        <<" containment="<<t.containment_verified<<" identity="<<t.identity_verified
        <<" context_selection="<<(t.context_selection==2?"verified-default-dc":t.context_selection==1?"nv-affinity":"none")
        <<" process_released="<<t.process_resources_released<<" gl_cleanup="<<t.gl_cleanup_verified
        <<" uncertain="<<t.outcome_uncertain<<" result_fifo="<<t.result_fifo_accepted
        <<" requested_luid="<<hex(t.requested_luid)<<" observed_luid="<<hex(t.observed_luid)
        <<" elapsed_ms="<<t.elapsed_ms<<" peak_process="<<t.peak_process_bytes<<" peak_job="<<t.peak_job_bytes
        <<" dispatch_calls="<<t.dispatch_calls<<" execution_complete="<<t.execution_complete
        <<" message="<<t.message<<'\n'<<std::flush;
}
int run_once(const std::shared_ptr<const Work>& work,const ProtectedProcessOptions& options,
             const ProtectedSourceDigest& digest,uint64_t host_id,uint64_t contract_id) {
    using namespace nodus::contracts;
    auto provider=std::make_shared<ProtectedOpenGLProcess>(options);std::string error;
    require(provider->available(error),error);
    Config config;config.host_id=host_id;config.protected_provider=provider;
    config.lookup=[work,contract_id](const Contract& c)->std::shared_ptr<const Work>{return c.id==contract_id?work:nullptr;};
    auto host=std::make_shared<OpenGLGpuHost>(config);
    ContractNegotiator market(2,first_suitable_free({host}),10);
    require(host->bind_contract_owner(market),"Native contract owner binding failed");
    int commits=0;
    require(market.add_host(host->peer()) && market.add_engine({1,[&](const Result& result){
        require(result.loan.work.id==contract_id && result.loan.host==host_id,"Result FIFO loan mismatch");
        ++commits;return true;
    },[](Contract&){return false;}}) && market.submit({contract_id,1,1,1}),"Native contract admission failed");
    ToolRegistry registry;require(register_opengl_gpu_tool(registry,host,"nodus.development.protected-link"),"GPU row registry binding failed");
    auto root=make_table();std::array<Table,3> tables{make_table(),make_table(),make_table()};
    GraphRuntime graph;graph.module_io_rows.resize(3);graph.module_plugin_instances.resize(3);
    ThreadManager::TickRequest request;request.root_table=root.get();
    for(int i=0;i<3;++i){ThreadManager::ModuleContract module;module.module_idx=i;module.table=tables[i].get();module.sim_enabled=0;
        module.exec_mode=i==2?ThreadManager::ExecMode::Pooled:ThreadManager::ExecMode::Sequential;
        module.stack_contract=ModuleStackContract::AbstractTensorPointer;request.modules.push_back(module);}
    const auto types=register_port_types();
    for(int i=0;i<2;++i){
        require(gp_table_add_edge(root.get(),contact(i,1),contact(i+1,0))!=0,"Native FIFO edge allocation failed");
        GP_TableEdgeTensorSpecTyped spec{};spec.dims[0]=1;spec.dim_count=1;spec.slots=4;
        spec.elem_size=i?sizeof(Telemetry):sizeof(Control);spec.type_id=i?types.telemetry:types.control;
        require(gp_table_edge_set_tensor_spec(root.get(),i,&spec)!=0,"Native typed FIFO specification failed");
        require(gp_table_edge_subscribe_ex(root.get(),i,contact(i+1,0),1)!=0,"Native FIFO subscription failed");
        ThreadManager::EdgeContract edge;edge.edge_idx=i;edge.a_module=i;edge.a_contact_idx=1;edge.b_module=i+1;edge.b_contact_idx=0;request.edges.push_back(edge);
    }
    require(bind_opengl_gpu_module(graph,registry,"nodus.development.protected-link",request.modules[1],root.get(),0,1,0,1,&error),error);
    int tick=0;bool submitted=false,cancel_sent=false;Telemetry latest{};uint64_t last_state=UINT64_MAX,last_refusal=UINT64_MAX;
    std::string callback_error;
    control_tool(graph,0,[&](ToolStackContext& context){
        try {
            Control command{};
            if(tick==1){
                require(!cancelled.load(),"Cancelled before native loan admission; no child created");
                market.market_turn();const auto queued=host->telemetry();
                require(queued.state==uint64_t(State::Queued) && queued.generation,"Canonical row did not assign an exact queued loan");
                // CLI approval committed the exact source/target, not a guessed
                // generation. Bind it once to the actual canonical admission.
                if(!cancelled.load()) {
                    require(provider->authorize_once({host_id,contract_id,queued.generation,digest},error),error);
                    std::cout<<"authorized_exact_loan host="<<host_id<<" contract="<<contract_id<<" generation="<<queued.generation<<'\n'<<std::flush;
                    submitted=true;
                }
            }
            if(tick>=1)command={uint64_t(Command::Advance),contract_id};
            if(cancelled.load() && !cancel_sent && latest.state==uint64_t(State::Linking)){
                command={uint64_t(Command::CancelActive),contract_id};cancel_sent=true;
            }
            if(cancelled.load() && host->telemetry().state==uint64_t(State::Queued))
                command={uint64_t(Command::CancelQueued),contract_id};
            const bool terminal=latest.state==uint64_t(State::Linked) || latest.state==uint64_t(State::Failed) || latest.state==uint64_t(State::Cancelled);
            const bool no_child_cancel=latest.state==uint64_t(State::Cancelled) && !latest.link_attempted && !latest.process_id;
            if(terminal && ((latest.child_reaped && latest.process_resources_released) || no_child_cancel) && !commits){market.market_turn();market.worker_turn();}
            if(commits)command={uint64_t(Command::Query),contract_id};
            require(raw_stack_push_typed(*context.stack.raw,&command,types.control),"Control FIFO push failed");
        } catch(const std::exception& e){callback_error=e.what();context.succeeded=false;}
    });io(graph,0,ModuleRowKind::Output,1);
    io(graph,2,ModuleRowKind::Input,0);control_tool(graph,2,[&](ToolStackContext& context){
        if(!raw_stack_pop_typed(*context.stack.raw,&latest,types.telemetry)){callback_error="Telemetry FIFO pop failed";context.succeeded=false;return;}
        if(latest.state!=last_state || latest.refusal!=last_refusal || latest.result_fifo_accepted){print_telemetry(latest);last_state=latest.state;last_refusal=latest.refusal;}
        if(latest.dispatch_calls || latest.execution_complete || latest.context_current)callback_error="Link-only protected-row invariant violated";
    });
    auto& manager=*graph.thread_mgr;
    struct ManagerLifetime {
        ThreadManager& manager;bool handler=false;
        void stop(){manager.stop();if(handler){SetConsoleCtrlHandler(console_control,FALSE);handler=false;}}
        ~ManagerLifetime(){stop();}
    } manager_lifetime{manager}; // destroyed before every callback-captured local
    manager_lifetime.handler=SetConsoleCtrlHandler(console_control,TRUE)!=0;
    require(manager_lifetime.handler,"Could not install native cancellation handler; no child created");
    manager.start();const auto start=GetTickCount64();
    // This is graph driving, not a second scheduling queue. Every operation
    // still traverses the same manager lane and typed Control/Telemetry FIFOs.
    for(tick=0;!commits && callback_error.empty() && GetTickCount64()-start<105000;++tick){
        manager.submit_tick(request,true);Sleep(25);
    }
    if(!commits){provider->cancel();std::cerr<<"No committed final result; retained/uncertain ownership is not retry permission\n";}
    manager_lifetime.stop();
    print_telemetry(latest);
    std::cout<<"summary submitted="<<submitted<<" commits="<<commits<<" ticks="<<tick
        <<" callback_error="<<callback_error<<" no_retry=1 dispatch_held=1\n";
    return commits==1 && latest.state==uint64_t(State::Linked) && latest.link_succeeded &&
        latest.gl_cleanup_verified && !latest.outcome_uncertain && callback_error.empty()?0:4;
}
}

int wmain(int argc,wchar_t** argv) {
    try {
        require(argc>=2,"Specify --prepare or --link-once and explicit source/worker/device/loan fields");
        const std::wstring mode=argv[1];require(mode==L"--prepare" || mode==L"--link-once","Unknown mode; no operation performed");
        std::map<std::wstring,std::wstring> args;bool approved=false;
        for(int i=2;i<argc;++i){const std::wstring key=argv[i];if(key==L"--approve-one-link"){require(!approved,"Duplicate approval flag");approved=true;continue;}
            require(i+1<argc && !args.count(key),"Missing or duplicate option value");args.emplace(key,argv[++i]);}
        for(const auto& [key,value]:args)require(key==L"--source-file" || key==L"--source-label" || key==L"--worker" ||
            key==L"--adapter-luid" || key==L"--node-mask" || key==L"--context-policy" || key==L"--host-id" || key==L"--contract-id" || key==L"--source-digest","Unknown option");
        const auto get=[&](const wchar_t* key)->const std::wstring& {const auto found=args.find(key);require(found!=args.end(),"Missing required explicit option: "+utf8(key));return found->second;};
        const auto host=number(get(L"--host-id")),contract=number(get(L"--contract-id"));
        ProtectedProcessOptions options;options.executable=get(L"--worker");options.adapter_luid=unhex<8>(get(L"--adapter-luid"));
        const auto mask=number(get(L"--node-mask"));require(mask<=UINT32_MAX,"Node mask out of range");options.node_mask=static_cast<uint32_t>(mask);
        const auto& policy=get(L"--context-policy");
        require(policy==L"verified-default-dc" || policy==L"nv-affinity","Unknown explicit context policy");
        options.context_policy=policy==L"verified-default-dc"?ProtectedContextPolicy::VerifiedDefaultDc:ProtectedContextPolicy::RequireNvAffinity;
        const std::filesystem::path path(get(L"--source-file"));require(path.is_absolute(),"Source file must be an absolute path");
        const auto bytes=std::filesystem::file_size(path);const auto label=utf8(get(L"--source-label"));
        require(!label.empty() && label.size()<=1024 && bytes && bytes<=16*1024*1024-label.size(),"Source/label exceeds bounded native input");
        std::ifstream input(path,std::ios::binary);require(bool(input),"Cannot read source file");
        std::string source(static_cast<size_t>(bytes),'\0');input.read(source.data(),static_cast<std::streamsize>(source.size()));
        require(input.gcount()==static_cast<std::streamsize>(source.size()) && input.peek()==std::char_traits<char>::eof(),"Source file changed or could not be read exactly");
        auto work=std::make_shared<Work>();work->objects={{label,std::move(source)}};
        gp_gl_compute_source object{work->objects[0].label.c_str(),work->objects[0].source.data(),work->objects[0].source.size()};
        ProtectedSourceDigest digest{};std::string error;
        require(protected_source_digest({host,contract,1,&object,1},digest,error),error); // digest does not include loan IDs
        std::cout<<"backend=OpenGL operation=link-only source_file="<<utf8(path.wstring())<<" source_label="<<label
            <<" source_bytes="<<bytes<<" source_digest="<<hex(digest)<<" requested_luid="<<hex(options.adapter_luid)
            <<" expected_node_mask="<<options.node_mask<<" host="<<host<<" contract="<<contract
            <<" context_policy="<<utf8(policy)
            <<" process_cap="<<options.process_memory_bytes<<" job_cap="<<options.job_memory_bytes
            <<" minimum_available_ram="<<options.minimum_available_memory_bytes<<" deadline_ms="<<options.wall_timeout_ms
            <<" active_process_limit=1 dispatch_held=1\n"<<std::flush;
        if(mode==L"--prepare"){
            require(!approved && !args.count(L"--source-digest"),"Preparation never accepts approval arguments");
            std::cout<<"PREPARED ONLY: no graph, child, context or shader call. Syntax validation and explicit per-link approval remain required.\n";return 0;
        }
        require(approved,"--link-once requires --approve-one-link after explicit user authorization");
        require(unhex<32>(get(L"--source-digest"))==digest,"Source label/bytes differ from the approved digest; no child created");
        return run_once(work,options,digest,host,contract);
    }catch(const std::exception& error){std::cerr<<"REFUSED: "<<error.what()<<'\n';return 2;}
}
