// Actual native ContractNegotiator/FIFO + ITool/raw stack. Only GL is mocked.
#include "opengl_compute_mock.h"
#include "opengl_gpu_row_tool.h"
#include <thread>

namespace {
struct MockProtectedProvider final : nodus::gpu::ProtectedLinkProvider {
    nodus::gpu::ProtectedLinkOutcome result;
    int links=0,polls=0,cancels=0;bool throws=false,can_cleanup=false,poll_available=false,mismatched=false,armed=true,cancel_accepts=true;
    bool available(std::string& reason) const override {reason.clear();return true;}
    bool ready(std::string& reason) const override {reason.clear();return true;}
    bool authorized(const nodus::gpu::ProtectedLinkRequest&,std::string& reason) const override {
        reason=armed?"":"CPU test awaiting explicit exact-loan approval";return armed;
    }
    nodus::gpu::ProtectedLinkOutcome link(const nodus::gpu::ProtectedLinkRequest& request) override {
        ++links;
        expect(request.source_count==1 && std::string(request.sources[0].source,request.sources[0].source_bytes)==
            "#version 430\nvoid main(){}\n","provider receives exact unchanged source");
        result.host_id=request.host_id;result.contract_id=request.contract_id;result.generation=request.generation;
        if(throws)throw std::runtime_error("provider fault after possible submission");
        auto out=result;if(mismatched)++out.contract_id;return out;
    }
    bool poll_cleanup(nodus::gpu::ProtectedLinkOutcome& out) override {
        ++polls;if(!can_cleanup)return false;
        result.child_reaped=result.resources_released=true;result.outcome_uncertain=true;
        out=result;return true;
    }
    bool poll(nodus::gpu::ProtectedLinkOutcome& out) override {
        ++polls;if(!poll_available)return false;out=result;return true;
    }
    bool cancel() override {++cancels;return cancel_accepts;}
};
uintptr_t forbidden_local_context(void*) {throw std::runtime_error("external provider called parent context");}
void* forbidden_local_proc(const char*,void*) {throw std::runtime_error("external provider called parent GL");}
nodus::gpu::Telemetry command(ITool& tool,nodus::gpu::Control control) {
    const auto types=nodus::gpu::register_port_types();
    RawStackFrame frame{};
    expect(raw_stack_init_frame(frame,1024),"native typed stack allocation");
    expect(raw_stack_push_typed(frame,&control,types.control),"typed control input");
    ToolStackContext context;context.stack.raw=&frame;
    tool.execute_stack(context);
    expect(context.succeeded,"native tool row command accepted");
    nodus::gpu::Telemetry telemetry{};
    expect(raw_stack_pop_typed(frame,&telemetry,types.telemetry),"typed telemetry output");
    raw_stack_free_mask(frame);gp_mem_backend_release(frame.backend);
    return telemetry;
}
}

int main() {
    using namespace nodus::gpu;
    using namespace nodus::contracts;
    try {
        reset();
        auto work=std::make_shared<Work>();
        work->objects={{"canonical object","#version 430\nvoid main(){}\n"}};
        bool protected_process=false;
        Config config;
        config.host_id=101;config.context=context;
        config.lookup=[&](const Contract& c)->std::shared_ptr<const Work>{return c.payload==99?work:nullptr;};
        config.protected_link_permitted=[&](std::string& why){why="CPU test supervisor unavailable";return protected_process;};
        auto host=std::make_shared<OpenGLGpuHost>(config);
        Config second=config;second.host_id=102;
        auto host2=std::make_shared<OpenGLGpuHost>(second);
        ToolRegistry registry;
        expect(register_opengl_gpu_tool(registry,host,"gpu.test"),"bound native GPU registry entry");
        auto tool=registry.create("gpu.test");
        expect(tool && tool->port_count()==2 && tool->port_spec(0).kind==ToolPortKind::Argument &&
            tool->port_spec(1).kind==ToolPortKind::Return,"control and telemetry ports");
        const auto query=command(*tool,{});
        expect(query.refusal==uint64_t(Refusal::SupervisionUnavailable) && !dispatch_count,"missing protection is explicit");
        const Contract work_token{1,17,8,99};
        expect(!host->eligible(work_token),"not ready without verified protection");
        protected_process=true;

        // Existing native result FIFO. No fake transport or independent queue.
        ContractNegotiator market(4,first_suitable_free({host,host2}),20);
        expect(host->bind_contract_owner(market) && host2->bind_contract_owner(market),"canonical contract owner binding");
        int committed=0;uint64_t result_state=0;bool accept_result=false;
        expect(market.add_engine({17,[&](const Result& r){++committed;result_state=r.payload;return accept_result;},
            [](Contract&){return false;}}),"source engine registration");
        expect(market.add_host(host->peer()) && market.add_host(host2->peer()),"GPU host registration");
        expect(market.submit(work_token),"source contract admitted");
        market.market_turn();
        expect(host->busy() && host->telemetry().state==uint64_t(State::Queued),"first suitable row queued");
        expect(count("compile")==0 && !dispatch_count,"queue admission is not execution");
        auto before=command(*tool,{});
        expect(before.harness_busy && !before.link_succeeded && !before.execution_complete &&
            !before.physical_utilization_available,"queued telemetry distinguishes physical utilization");
        auto linked=command(*tool,{uint64_t(Command::Advance),1});
        expect(linked.state==uint64_t(State::Linked) && linked.link_succeeded && linked.program_released &&
            !linked.dispatch_calls && !linked.execution_complete,"link-only success never dispatch/completion");
        expect(count("compile")==1 && count("link")==1 && programs.empty(),"native helper and release on owner");
        market.market_turn();market.market_turn();
        expect(market.contracts()[0].state==ContractNegotiator::State::Returned &&
            host->telemetry().result_fifo_accepted,"result accepted into actual FIFO");
        market.worker_turn();
        expect(committed==1 && result_state==uint64_t(State::Linked) &&
            market.contracts()[0].state==ContractNegotiator::State::Returned,"FIFO result is not engine commit");
        expect(!host->forget_committed(1,market),"cannot forget uncommitted outcome");
        // Rejection/lease reoffer cannot replay an already handled program.
        const auto generation=market.contracts()[0].loan.generation;
        expect(!market.reject_pending(1,generation),"protected result cannot be reoffered");
        expect(market.remove_host(101),"removed GPU callbacks detach safely");
        const auto replacement_policy=first_suitable_free({host2});
        expect(replacement_policy(host2->peer(),market.contracts())==SIZE_MAX,
            "replacement policy without old GPU still observes canonical protected ownership");
        for(int i=0;i<6;++i)market.market_turn();
        expect(market.contracts()[0].state==ContractNegotiator::State::Returned &&
            count("compile")==1,"rejected result cannot replay on another GPU");
        accept_result=true;market.worker_turn();
        expect(host->forget_committed(1,market),"release replay retention only after commit");

        // Independent next work can use an available row, while old outcome
        // remains explicitly retained rather than silently replayed.
        expect(market.submit({2,17,8,99}),"second contract");
        for(int i=0;i<2;++i)market.market_turn();
        auto owner=host->busy()?host:host2;
        auto peer=owner->peer();
        expect(owner->busy(),"second job admitted to free row");
        const size_t calls_before=calls.size();
        bool wrong_lane_accepted=true;
        std::thread other([&]{wrong_lane_accepted=owner->control({uint64_t(Command::Advance),2});});other.join();
        expect(!wrong_lane_accepted && calls.size()==calls_before,"wrong ThreadManager lane makes no GL call");
        expect(owner->control({uint64_t(Command::CancelQueued),2}),"cancel queued work");
        expect(owner->telemetry().state==uint64_t(State::Cancelled) && count("compile")==1,"cancel has no link");
        for(int i=0;i<2;++i)market.market_turn();
        accept_result=true;market.worker_turn();
        expect(owner->forget_committed(2,market),"explicit retirement after engine commit");

        // A removed/expired host keeps its outstanding identity; the Offer
        // policy will not place that exact work on a second GPU for replay.
        Config short_cfg=config;short_cfg.host_id=103;
        auto short_host=std::make_shared<OpenGLGpuHost>(short_cfg);
        Config alt_cfg=config;alt_cfg.host_id=104;
        auto alt=std::make_shared<OpenGLGpuHost>(alt_cfg);
        ContractNegotiator expiry(2,first_suitable_free({short_host,alt}),1);
        expect(short_host->bind_contract_owner(expiry) && alt->bind_contract_owner(expiry),"expiry owner binding");
        expect(expiry.add_engine({17,[](const Result&){return true;},[](Contract&){return false;}}),"expiry engine");
        expect(expiry.add_host(short_host->peer()) && expiry.add_host(alt->peer()),"expiry hosts");
        expect(expiry.submit({3,17,8,99}),"expiry work");expiry.market_turn();
        expect(short_host->busy(),"expiry queued");
        expiry.market_turn();
        expect(!alt->busy(),"lease expiry cannot give held source to second row");
        expect(expiry.remove_host(103),"remove host from routing");expiry.market_turn();
        expect(!alt->busy(),"host removal still does not replay held source");
        expect(short_host->control({uint64_t(Command::CancelQueued),3}),"cancel removed host without GL");

        // A context can be lost AFTER admission. That is a terminal refusal,
        // not permission to activate another context or try another device.
        Config lost_cfg=config;lost_cfg.host_id=105;
        auto lost=std::make_shared<OpenGLGpuHost>(lost_cfg);
        ContractNegotiator loss(1,first_suitable_free({lost}),100);
        expect(lost->bind_contract_owner(loss) && loss.add_host(lost->peer()) &&
            loss.add_engine({17,[](const Result&){return true;},[](Contract&){return false;}}) &&
            loss.submit({4,17,8,99}),"context-loss owner");loss.market_turn();
        expect(lost->busy(),"context-loss admission");
        current=88;
        expect(lost->control({uint64_t(Command::Advance),4}),"context-loss result");
        expect(lost->telemetry().refusal==uint64_t(Refusal::ContextUnavailable) && count("compile")==1,"context-loss no native call");
        current=77;

        // Callback exceptions restore the consumed typed input as required by
        // ThreadManager's failed-tool contract, without dispatch or replay.
        Config throwing_cfg=config;throwing_cfg.host_id=106;
        throwing_cfg.protected_link_permitted=[](std::string&)->bool{throw std::runtime_error("test gate fault");};
        auto throwing=std::make_shared<OpenGLGpuHost>(throwing_cfg);
        OpenGLGpuTool throwing_tool(throwing);
        RawStackFrame exception_frame{};
        expect(raw_stack_init_frame(exception_frame,1024),"exception stack");
        const auto port_types=register_port_types();Control original{};
        expect(raw_stack_push_typed(exception_frame,&original,port_types.control),"exception input");
        ToolStackContext exception_context;exception_context.stack.raw=&exception_frame;
        throwing_tool.execute_stack(exception_context);
        Control restored{};
        expect(!exception_context.succeeded && raw_stack_pop_typed(exception_frame,&restored,port_types.control) &&
            restored.command==original.command,"exception restores typed Control");
        raw_stack_free_mask(exception_frame);gp_mem_backend_release(exception_frame.backend);

        Config fail_cfg=config;fail_cfg.host_id=107;
        auto failing=std::make_shared<OpenGLGpuHost>(fail_cfg);OpenGLGpuTool failing_tool(failing);
        ContractNegotiator failures(2,first_suitable_free({failing}),100);
        expect(failing->bind_contract_owner(failures) && failures.add_host(failing->peer()) &&
            failures.add_engine({17,[](const Result&){return false;},[](Contract&){return false;}}) &&
            failures.submit({7,17,8,99}),"failure owner");failures.market_turn();
        compile_ok=false;
        auto failed=command(failing_tool,{uint64_t(Command::Advance),7});
        expect(failed.refusal==uint64_t(Refusal::NativeLinkFailed) && failed.message[0],"native failure diagnostic");
        compile_ok=true;failures.market_turn();
        const auto failure_query=command(failing_tool,{uint64_t(Command::Query),7});
        expect(failure_query.refusal==failed.refusal && std::string(failure_query.message)==failed.message,
            "Query after FIFO acceptance preserves terminal diagnostic");
        expect(failures.submit({8,17,8,99}),"next work beside retained failure");failures.market_turn();
        const auto old_query=command(failing_tool,{uint64_t(Command::Query),7});
        expect(old_query.contract_id==7 && old_query.refusal==failed.refusal &&
            failing->telemetry().contract_id==8,"query retained outcome does not overwrite active work");
        expect(failing->control({uint64_t(Command::CancelQueued),8}),"cancel diagnostic-test tail");

        Config cleanup_cfg=config;cleanup_cfg.host_id=108;
        auto cleanup_host=std::make_shared<OpenGLGpuHost>(cleanup_cfg);
        ContractNegotiator cleanups(1,first_suitable_free({cleanup_host}),100);
        expect(cleanup_host->bind_contract_owner(cleanups) && cleanups.add_host(cleanup_host->peer()) &&
            cleanups.add_engine({17,[](const Result&){return true;},[](Contract&){return false;}}) &&
            cleanups.submit({9,17,8,99}),"cleanup owner");cleanups.market_turn();
        release_unavailable_after_link=true;
        const auto links_before=count("link");
        expect(!cleanup_host->control({uint64_t(Command::Advance),9}) && cleanup_host->busy() &&
            cleanup_host->telemetry().refusal==uint64_t(Refusal::CleanupPending),"failed cleanup retains GL owner");
        cleanups.market_turn();
        expect(cleanups.contracts()[0].state==ContractNegotiator::State::Loaned,"cleanup-pending result not published");
        missing_proc.clear();release_unavailable_after_link=false;
        expect(cleanup_host->control({uint64_t(Command::Advance),9}) &&
            count("link")==links_before+1 && cleanup_host->telemetry().program_released,
            "deferred cleanup retries release only, never shader link");

        Config failed_cleanup_cfg=config;failed_cleanup_cfg.host_id=109;
        auto failed_cleanup_host=std::make_shared<OpenGLGpuHost>(failed_cleanup_cfg);
        ContractNegotiator failed_cleanups(1,first_suitable_free({failed_cleanup_host}),100);
        expect(failed_cleanup_host->bind_contract_owner(failed_cleanups) && failed_cleanups.add_host(failed_cleanup_host->peer()) &&
            failed_cleanups.add_engine({17,[](const Result&){return true;},[](Contract&){return false;}}) &&
            failed_cleanups.submit({10,17,8,99}),"failed-link cleanup owner");failed_cleanups.market_turn();
        link_ok=false;release_unavailable_after_link=true;
        const auto failed_links_before=count("link");
        expect(failed_cleanup_host->control({uint64_t(Command::Advance),10}) && failed_cleanup_host->busy() &&
            failed_cleanup_host->telemetry().refusal==uint64_t(Refusal::NativeLinkFailed),"failed link keeps cleanup owner");
        failed_cleanups.market_turn();
        expect(failed_cleanups.contracts()[0].state==ContractNegotiator::State::Loaned,"failed-link cleanup holds result publication");
        link_ok=true;missing_proc.clear();release_unavailable_after_link=false;
        expect(failed_cleanup_host->control({uint64_t(Command::Advance),10}) &&
            count("link")==failed_links_before+1 && failed_cleanup_host->telemetry().program_released &&
            failed_cleanup_host->telemetry().state==uint64_t(State::Failed) &&
            failed_cleanup_host->telemetry().refusal==uint64_t(Refusal::NativeLinkFailed) &&
            !failed_cleanup_host->telemetry().link_succeeded,"cleanup preserves original failed-link outcome, no retry");

        auto provider=std::make_shared<MockProtectedProvider>();
        provider->result.status=ProtectedLinkStatus::Linked;provider->result.linked=true;
        provider->result.resources_released=provider->result.child_reaped=true;
        provider->result.gl_cleanup_verified=true;
        provider->result.attempted=true;
        provider->result.requested_luid[0]=provider->result.observed_luid[0]=42;
        provider->result.containment_verified=provider->result.identity_verified=true;
        Config remote_cfg=config;remote_cfg.host_id=110;remote_cfg.protected_provider=provider;
        remote_cfg.context={5,&forbidden_local_proc,&forbidden_local_context,nullptr};
        remote_cfg.protected_link_permitted=[](std::string&)->bool{throw std::runtime_error("parent local gate used by provider");};
        auto remote=std::make_shared<OpenGLGpuHost>(remote_cfg);OpenGLGpuTool remote_tool(remote);
        ContractNegotiator remote_market(1,first_suitable_free({remote}),100);
        expect(remote->bind_contract_owner(remote_market) && remote_market.add_host(remote->peer()) &&
            remote_market.add_engine({17,[](const Result&){return true;},[](Contract&){return false;}}) &&
            remote_market.submit({11,17,8,99}),"protected provider owner");
        const auto native_calls_before=calls.size();
        expect(command(remote_tool,{}).protected_process,"provider query needs no local context");
        remote_market.market_turn();
        const auto protected_result=command(remote_tool,{uint64_t(Command::Advance),11});
        expect(protected_result.state==uint64_t(State::Linked) && protected_result.containment_verified &&
            protected_result.identity_verified && protected_result.child_reaped && protected_result.link_attempted && !protected_result.context_current &&
            provider->links==1 && calls.size()==native_calls_before,"protected provider bypasses all parent GL");
        remote_market.market_turn();remote_market.worker_turn();
        expect(remote->forget_committed(11,remote_market),"protected result commits via existing native FIFO");
        expect(remote_market.retire(11),"retire completed test contract");provider->result.attempted=false;
        expect(remote_market.submit({15,17,8,99}),"contradictory receipt work");remote_market.market_turn();
        const auto contradictory=command(remote_tool,{uint64_t(Command::Advance),15});
        expect(contradictory.state==uint64_t(State::Failed) && !contradictory.link_succeeded && !contradictory.link_attempted,
            "cannot claim linked success when provider says no attempt occurred");
        remote_market.market_turn();remote_market.worker_turn();expect(remote->forget_committed(15,remote_market),"contradictory receipt commit");

        auto uncertain=std::make_shared<MockProtectedProvider>();uncertain->throws=true;
        remote_cfg.host_id=111;remote_cfg.protected_provider=uncertain;
        auto unknown_host=std::make_shared<OpenGLGpuHost>(remote_cfg);
        ContractNegotiator unknown_market(1,first_suitable_free({unknown_host}),2);
        expect(unknown_host->bind_contract_owner(unknown_market) && unknown_market.add_host(unknown_host->peer()) &&
            unknown_market.add_engine({17,[](const Result&){return true;},[](Contract&){return false;}}) &&
            unknown_market.submit({12,17,8,99}),"uncertain protected owner");unknown_market.market_turn();
        expect(unknown_host->control({uint64_t(Command::Advance),12}) && unknown_host->telemetry().outcome_uncertain,
            "provider exception records uncertain outcome rather than restoring a relink command");
        for(int i=0;i<4;++i)unknown_market.market_turn();
        expect(unknown_market.contracts()[0].state==ContractNegotiator::State::Loaned &&
            !unknown_host->control({uint64_t(Command::Advance),12}) && !unknown_host->shutdown() &&
            uncertain->links==1,"unknown child ownership stays pinned and cannot relink or publish");
        uncertain->can_cleanup=true;
        expect(unknown_host->control({uint64_t(Command::Advance),12}) &&
            unknown_host->telemetry().child_reaped && unknown_host->telemetry().outcome_uncertain &&
            unknown_host->telemetry().process_resources_released && !unknown_host->telemetry().program_released &&
            unknown_host->telemetry().state==uint64_t(State::Failed) && uncertain->links==1,
            "cleanup polling reclaims original child without retry or invented success");
        unknown_market.market_turn();unknown_market.worker_turn();
        expect(unknown_host->forget_committed(12,unknown_market) && calls.size()==native_calls_before,
            "reaped uncertain failure can commit with no parent GL calls");

        auto asynchronous=std::make_shared<MockProtectedProvider>();
        asynchronous->result.status=ProtectedLinkStatus::Running;
        asynchronous->armed=false;
        remote_cfg.host_id=112;remote_cfg.protected_provider=asynchronous;
        auto running=std::make_shared<OpenGLGpuHost>(remote_cfg);
        ContractNegotiator running_market(1,first_suitable_free({running}),2);
        expect(running->bind_contract_owner(running_market) && running_market.add_host(running->peer()) &&
            running_market.add_engine({17,[](const Result&){return true;},[](Contract&){return false;}}) &&
            running_market.submit({13,17,8,99}),"nonblocking provider owner");running_market.market_turn();
        const auto held_generation=running_market.contracts()[0].loan.generation;
        for(int i=0;i<5;++i) {
            expect(running->control({uint64_t(Command::Advance),13}) &&
                running->telemetry().state==uint64_t(State::Queued),"unapproved native work remains queued");
            running_market.market_turn();
        }
        expect(!asynchronous->links && running_market.contracts()[0].state==ContractNegotiator::State::Loaned &&
            running_market.contracts()[0].loan.generation==held_generation,
            "approval wait holds exact canonical generation beyond lease, no native attempt");
        asynchronous->armed=true;
        expect(running->control({uint64_t(Command::Advance),13}) &&
            running->telemetry().state==uint64_t(State::Linking) && !running->telemetry().outcome_uncertain,
            "accepted running work is not an uncertain failure");
        for(int i=0;i<4;++i){running_market.market_turn();expect(running->control({uint64_t(Command::Advance),13}),"poll running owner");}
        expect(asynchronous->links==1 && asynchronous->polls==4 &&
            running_market.contracts()[0].state==ContractNegotiator::State::Loaned,
            "manager-lane polling does not relink or expire nonreplayable work");
        expect(running->control({uint64_t(Command::CancelActive),13}) && asynchronous->cancels==1 &&
            running->busy(),"cancel requests original child termination without clearing its ownership");
        asynchronous->result.status=ProtectedLinkStatus::Cancelled;
        asynchronous->result.child_reaped=asynchronous->result.resources_released=true;
        asynchronous->result.outcome_uncertain=true;asynchronous->poll_available=true;
        expect(running->control({uint64_t(Command::Advance),13}) && running->telemetry().outcome_uncertain &&
            running->telemetry().state==uint64_t(State::Failed) && asynchronous->links==1,
            "cancelled uncertain child reports failure only after reaping, never success/replay");
        running_market.market_turn();running_market.worker_turn();
        expect(running->forget_committed(13,running_market) && calls.size()==native_calls_before,
            "cancelled protected outcome commits with no parent GL calls");

        auto mismatch=std::make_shared<MockProtectedProvider>();mismatch->result=provider->result;mismatch->mismatched=true;
        remote_cfg.host_id=113;remote_cfg.protected_provider=mismatch;
        auto mismatch_host=std::make_shared<OpenGLGpuHost>(remote_cfg);
        ContractNegotiator mismatch_market(1,first_suitable_free({mismatch_host}),2);
        expect(mismatch_host->bind_contract_owner(mismatch_market) && mismatch_market.add_host(mismatch_host->peer()) &&
            mismatch_market.add_engine({17,[](const Result&){return true;},[](Contract&){return false;}}) &&
            mismatch_market.submit({14,17,8,99}),"mismatched receipt owner");mismatch_market.market_turn();
        expect(mismatch_host->control({uint64_t(Command::Advance),14}) &&
            mismatch_host->telemetry().refusal==uint64_t(Refusal::OutcomeUnknown) &&
            !mismatch_host->telemetry().link_succeeded,"foreign receipt cannot complete local pinned loan");
        mismatch_market.market_turn();
        expect(mismatch_market.contracts()[0].state==ContractNegotiator::State::Loaned,"foreign receipt is not published");
        mismatch->can_cleanup=true;
        expect(mismatch_host->control({uint64_t(Command::Advance),14}) && mismatch->links==1 &&
            mismatch_host->telemetry().state==uint64_t(State::Failed),"matching cleanup receipt resolves failure without replay");

        auto cancel_race=std::make_shared<MockProtectedProvider>();cancel_race->result.status=ProtectedLinkStatus::Running;
        remote_cfg.host_id=114;remote_cfg.protected_provider=cancel_race;
        auto race_host=std::make_shared<OpenGLGpuHost>(remote_cfg);OpenGLGpuTool race_tool(race_host);
        int race_commits=0;ContractNegotiator race_market(1,first_suitable_free({race_host}),2);
        expect(race_host->bind_contract_owner(race_market) && race_market.add_host(race_host->peer()) &&
            race_market.add_engine({17,[&](const Result& r){++race_commits;return r.payload==uint64_t(State::Linked);},[](Contract&){return false;}}) &&
            race_market.submit({16,17,8,99}),"cancel completion race owner");race_market.market_turn();
        expect(command(race_tool,{uint64_t(Command::Advance),16}).state==uint64_t(State::Linking),"race starts once");
        cancel_race->result.status=ProtectedLinkStatus::Linked;
        cancel_race->result.attempted=cancel_race->result.linked=cancel_race->result.resources_released=true;
        cancel_race->result.child_reaped=cancel_race->result.gl_cleanup_verified=true;
        cancel_race->result.containment_verified=cancel_race->result.identity_verified=true;
        cancel_race->result.requested_luid[0]=cancel_race->result.observed_luid[0]=42;
        cancel_race->cancel_accepts=false;cancel_race->poll_available=true;
        const auto raced=command(race_tool,{uint64_t(Command::CancelActive),16});
        expect(raced.state==uint64_t(State::Linked) && raced.link_succeeded && cancel_race->links==1 &&
            cancel_race->cancels==1 && cancel_race->polls==1,
            "completed-child cancel emits final typed Telemetry without relink or restored Control");
        race_market.market_turn();race_market.worker_turn();
        expect(race_commits==1 && race_host->forget_committed(16,race_market),"cancel completion race commits exact result once");
        expect(host->shutdown() && host2->shutdown() && short_host->shutdown() && alt->shutdown() && lost->shutdown(),"native row shutdown");
        expect(!dispatch_count,"GPU row test never dispatched a shader");
        std::cout<<"PASS: actual native FIFO -> bound GPU row -> mock GL link-only -> typed telemetry; no renderer/device calls\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
