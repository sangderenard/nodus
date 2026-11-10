#include "contract_negotiator_tool.h"
#include <cassert>
#include <cstdio>
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
    std::puts("native contract negotiator FIFO tests passed");
}
