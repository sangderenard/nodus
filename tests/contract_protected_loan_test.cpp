#include "contract_negotiator.h"
#include <iostream>
#include <stdexcept>

using namespace nodus::contracts;
static void check(bool value,const char* why){if(!value)throw std::runtime_error(why);}
int main(){try{
    auto choose=[](const Peer&,const std::vector<ContractNegotiator::Entry>& entries){
        for(size_t i=0;i<entries.size();++i)if(entries[i].state==ContractNegotiator::State::Unaccepted)return i;
        return SIZE_MAX;
    };
    ContractNegotiator owner(2,choose,2);
    Loan held{};bool deliver=false,commit=false;int admissions=0,other_admissions=0;
    check(owner.add_engine({10,[&](const Result&){return commit;},[](Contract&){return false;}}),"engine");
    check(owner.add_host({20,[&](const Loan* offer){Exchange r;
        if(offer){held=*offer;r.accepted_generation=offer->generation;++admissions;}
        if(deliver && held.generation){r.has_result=true;r.result={held,123};}
        return r;}}),"host");
    check(owner.add_host({21,[&](const Loan* offer){Exchange r;if(offer){++other_admissions;r.accepted_generation=offer->generation;}return r;}}),"other host");
    check(owner.submit({1,10,7,44}),"submit");owner.market_turn();
    check(!owner.protect_loan(1,held.generation+1,20) && !owner.protect_loan(1,held.generation,21),"only exact loan may protect");
    check(owner.protect_loan(1,held.generation,20),"protect exact native owner before operation");
    for(int i=0;i<20;++i)owner.market_turn();
    check(admissions==1 && other_admissions==0 && owner.contracts()[0].state==ContractNegotiator::State::Loaned,
        "expired protected loan is not replayed");
    deliver=true;owner.market_turn();owner.market_turn();
    check(owner.contracts()[0].state==ContractNegotiator::State::Returned,"original late result still accepted");
    owner.worker_turn();
    check(!owner.reject_pending(1,held.generation),"protected result rejection cannot replay");
    commit=true;owner.worker_turn();
    check(owner.contracts()[0].state==ContractNegotiator::State::Committed,"engine commit");
    check(owner.remove_host(20) && owner.retire(1) && owner.remove_engine(10),"normal cleanup after commit");
    // Socket owners destroy callbacks immediately after remove_*. Removal
    // must succeed while the authoritative non-replayable entry stays behind.
    ContractNegotiator detached(2,choose,2);Loan old{};int old_callbacks=0,new_callbacks=0;
    check(detached.add_engine({30,[](const Result&){return false;},[](Contract&){return false;}}),"detached engine");
    check(detached.add_host({40,[&](const Loan* offer){++old_callbacks;Exchange r;if(offer){old=*offer;r.accepted_generation=offer->generation;}return r;}}),"detached host");
    check(detached.submit({2,30,9,55}),"detached work");detached.market_turn();
    check(detached.protect_loan(2,old.generation,40),"detached protection");
    check(detached.remove_host(40) && detached.remove_engine(30),"callbacks safely removed");
    check(detached.add_host({40,[&](const Loan* offer){++new_callbacks;check(!offer,"replacement received protected work");return Exchange{};}}),"replacement host");
    check(detached.add_engine({30,[](const Result&){return true;},[](Contract&){return false;}}),"replacement source");
    for(int i=0;i<12;++i){detached.market_turn();detached.worker_turn();}
    check(old_callbacks==1 && new_callbacks==12 && detached.contracts().size()==1 &&
        detached.contracts()[0].replay_protected && detached.contracts()[0].state==ContractNegotiator::State::Loaned,
        "replacement callbacks cannot erase/replay canonical tombstone");
    check(!detached.submit({2,30,9,55}),"retained identity prevents resubmission");
    std::cout<<"PASS: canonical protected loan survives expiry/removal/rejection until commit; actual native FIFO\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
