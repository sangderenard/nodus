#include "contract_negotiator_tool.h"
#include "contract_negotiator_registry.h"
#include <cassert>
#include <cstdio>
#include <deque>
#include <memory>
using namespace nodus::contracts;
int main(){
    auto choose=[](const Peer&,const std::vector<ContractNegotiator::Entry>& entries){
        for(size_t i=0;i<entries.size();++i) if(entries[i].state==ContractNegotiator::State::Unaccepted)return i;
        return size_t(-1);
    };
    ContractNegotiator box(3,choose,3);
    int checks=0, routed=0; bool allow=false; Loan loan{};
    assert(box.add_engine({10,[&](const Result& r){++routed;assert(r.payload==99);return allow;},
        [&](Contract&){++checks;return false;}}));
    int host1=0,host2=0;
    assert(box.add_host({1,[&](const Loan* offered){++host1;Exchange e;if(offered){loan=*offered;e.accepted_generation=offered->generation;}
        else if(loan.generation){e.has_result=true;e.result={loan,99};}return e;}}));
    assert(box.add_host({2,[&](const Loan*){++host2;return Exchange{};}})); // loopback same contract
    assert(box.submit({100,10,7,88}));assert(!box.submit({100,10,7,88}));
    ToolRegistry registry;
    assert(register_negotiator_tools(registry,box,"test.mail"));
    assert(!register_negotiator_tools(registry,box,"test.mail"));
    auto market=registry.create("test.mail.market");
    auto worker=registry.create("test.mail.worker");
    assert(market && worker);
    int unrelated_host=0; ToolInitContext init{};init.user=&unrelated_host;market->initialize(init);worker->initialize(init);
    ToolStackContext stack{};market->execute_stack(stack);
    assert(stack.succeeded && host1==1 && host2==0);
    box.market_turn();assert(host1==1 && host2==1);
    box.market_turn();assert(host1==2 && box.contracts()[0].state==ContractNegotiator::State::Returned);
    worker->execute_stack(stack);assert(stack.succeeded && routed==1 && checks==1); // refusal retains result
    allow=true;box.worker_turn();assert(routed==2);assert(box.contracts()[0].state==ContractNegotiator::State::Committed);
    box.market_turn();box.market_turn();box.worker_turn();assert(routed==2); // duplicate refused
    assert(box.retire(100));
    // Expiry reoffers with a fresh generation; old completion cannot win.
    Loan old{},fresh{};int visits=0;
    ContractNegotiator retry(1,choose,1);
    assert(retry.add_engine({10,[&](const Result&){assert(false);return false;},[](Contract&){return false;}}));
    assert(retry.add_host({3,[&](const Loan* offer){Exchange e;assert(offer);e.accepted_generation=offer->generation;
        if(++visits==1)old=*offer;else {fresh=*offer;e.has_result=true;e.result={old,99};}return e;}}));
    assert(retry.submit({200,10,8,90}));retry.market_turn();retry.market_turn();retry.worker_turn();
    assert(fresh.generation!=old.generation);assert(retry.contracts()[0].state==ContractNegotiator::State::Loaned);
    assert(!retry.retire(200));assert(!retry.submit({201,10,8,90}));
    // Process registry: idempotent acquire, live rows, release unregisters.
    {
        auto& a=acquire("registry.proof",2);auto& b=acquire("registry.proof",2);
        assert(&a==&b && find("registry.proof")==&a);
        auto m=tool_registry_global().create("registry.proof.market");
        auto w=tool_registry_global().create("registry.proof.worker");
        assert(m && w);
        ToolStackContext rs{};m->execute_stack(rs);assert(rs.succeeded); // execute BEFORE release only
        assert(release("registry.proof"));
        assert(!tool_registry_global().find("registry.proof.market") && !tool_registry_global().find("registry.proof.worker"));
        assert(find("registry.proof")==nullptr && !release("registry.proof"));
    }
    // ---- Removal API ----
    {
        auto fresh=[&](size_t cap=4,uint64_t ticks=50){return std::make_unique<ContractNegotiator>(cap,choose,ticks);};
        auto never=[](const Result&){return false;};
        auto none=[](Contract&){return false;};
        // Engine removal with nothing in flight; id reusable; unknown id refused.
        auto n=fresh();
        assert(n->add_engine({10,never,none}) && n->engine_count()==1);
        assert(!n->remove_engine(11));
        assert(n->remove_engine(10) && n->engine_count()==0 && !n->remove_engine(10));
        assert(!n->submit({1,10,1,1}));
        assert(n->add_engine({10,never,none}) && n->submit({1,10,1,1}));
        // Engine removal with pending (unaccepted) contracts: dropped, other engines' work kept.
        assert(n->add_engine({11,never,none}) && n->submit({2,10,1,1}) && n->submit({3,11,1,1}));
        assert(n->remove_engine(10) && n->contracts().size()==1 && n->contracts()[0].work.id==3);
        assert(n->discarded_results()==0);
        // Engine removal with a loan in flight: host completes, result discarded and counted.
        {
            auto b=fresh(); Loan held{}; int stage=0; int routed=0;
            assert(b->add_engine({10,[&](const Result&){++routed;return true;},none}));
            assert(b->add_host({1,[&](const Loan* o){Exchange e;
                if(o && stage==0){held=*o;e.accepted_generation=o->generation;stage=1;}
                else if(stage==1 && !o){e.has_result=true;e.result={held,5};stage=2;}
                return e;}}));
            assert(b->submit({7,10,1,1}));b->market_turn();
            assert(b->contracts()[0].state==ContractNegotiator::State::Loaned);
            assert(b->remove_engine(10));
            assert(b->contracts().size()==1 && b->contracts()[0].state==ContractNegotiator::State::Loaned);
            b->market_turn();                                   // host returns its result
            assert(b->contracts()[0].state==ContractNegotiator::State::Returned);
            b->worker_turn();
            assert(routed==0 && b->discarded_results()==1 && b->contracts().empty());
            b->worker_turn();assert(b->discarded_results()==1);
        }
        // Host removal with a loan in flight: re-offered to another host under a new generation.
        {
            auto b=fresh(); Loan held1{},held2{}; int stage2=0; int routed=0; uint64_t routed_host=0,routed_gen=0;
            assert(b->add_engine({10,[&](const Result& r){++routed;routed_host=r.loan.host;routed_gen=r.loan.generation;return true;},none}));
            assert(b->add_host({1,[&](const Loan* o){Exchange e;if(o){held1=*o;e.accepted_generation=o->generation;}return e;}}));
            assert(b->add_host({2,[&](const Loan* o){Exchange e;
                if(o && stage2==0){held2=*o;e.accepted_generation=o->generation;stage2=1;}
                else if(stage2==1 && !o){e.has_result=true;e.result={held2,6};stage2=2;}
                return e;}}));
            assert(b->submit({8,10,1,1}));b->market_turn();      // host 1 takes it, never answers
            assert(held1.generation==1 && b->contracts()[0].state==ContractNegotiator::State::Loaned);
            assert(b->remove_host(1) && !b->remove_host(1) && b->host_count()==1);
            assert(b->reoffered_loans()==1 && b->contracts()[0].state==ContractNegotiator::State::Unaccepted);
            b->market_turn();b->market_turn();                   // host 2: offer+accept, then result
            assert(b->contracts()[0].state==ContractNegotiator::State::Returned);
            b->worker_turn();
            assert(routed==1 && routed_host==2 && routed_gen==held2.generation && held2.generation>held1.generation);
        }
        // Removal requested from inside a turn is deferred to its end.
        {
            auto b=fresh(); Loan held{}; int stage=0; ContractNegotiator* self=b.get(); bool inside_true=false;
            assert(b->add_engine({10,[&](const Result&){inside_true=self->remove_engine(10);assert(self->engine_count()==1);return true;},none}));
            assert(b->add_host({1,[&](const Loan* o){Exchange e;
                if(o && stage==0){held=*o;e.accepted_generation=o->generation;stage=1;}
                else if(stage==1 && !o){e.has_result=true;e.result={held,5};stage=2;}
                return e;}}));
            assert(b->submit({9,10,1,1}));b->market_turn();b->market_turn();b->worker_turn();
            assert(inside_true && b->engine_count()==0 && b->contracts().empty());
        }
    }
    // ---- loan_ticks chosen at acquire ----
    {
        auto scenario=[&](const char* prefix,uint64_t loan,uint64_t* generation_out,int* routed_out){
            auto& n=loan?acquire(prefix,2,loan):acquire(prefix,2);
            std::deque<Result> fifo; int routed=0; uint64_t gen=0; Loan held{}; uint64_t accepted_at=0;
            assert(n.bind_result_transport([&](const Result& r){fifo.push_back(r);return true;},
                [&](Result& r){if(fifo.empty())return false;r=fifo.front();fifo.pop_front();return true;}));
            assert(n.add_engine({10,[&](const Result& r){++routed;gen=r.loan.generation;return true;},[](Contract&){return false;}}));
            assert(n.add_host({1,[&](const Loan* o){Exchange e;                       // holds each loan 100 turns
                if(o){held=*o;accepted_at=n.turns();e.accepted_generation=o->generation;}
                else if(held.generation && n.turns()-accepted_at>=100){e.has_result=true;e.result={held,5};held={};}
                return e;}}));
            assert(n.submit({1,10,1,1}));
            for(int i=0;i<150 && !routed;++i){n.market_turn();n.worker_turn();}
            *generation_out=routed?gen:n.contracts().empty()?0:n.contracts()[0].loan.generation;*routed_out=routed;
            assert(release(prefix));
        };
        uint64_t g=0;int r=0;
        scenario("loan.long",600,&g,&r);assert(r==1 && g==1);            // completes at generation 1, no re-loan
        scenario("loan.default",0,&g,&r);assert(r==0 && g>=2);           // default 60: re-loaned, never completes
        auto& a=acquire("loan.rule",2,300);
        assert(a.loan_ticks()==300 && &acquire("loan.rule",2,300)==&a && &acquire("loan.rule",2)==&a);
        bool refused=false;try{acquire("loan.rule",2,301);}catch(const std::runtime_error&){refused=true;}
        assert(refused && release("loan.rule"));
        assert(acquire("loan.dflt",2).loan_ticks()==60 && release("loan.dflt"));
    }
    std::puts("native contract negotiator FIFO tests passed");
}
