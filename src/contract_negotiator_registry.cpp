#include "contract_negotiator_registry.h"
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
namespace nodus::contracts {
namespace {
struct Pair {
    std::unique_ptr<ContractNegotiator> owner;
    std::string market_id,worker_id; // allocated before publication; release is allocation-free
};
struct Store { std::mutex m; std::map<std::string,Pair> by_prefix; };
Store& store(){ static Store s; return s; }
ContractNegotiator& create_locked(Store& s,const std::string& prefix,size_t capacity,
    ContractNegotiator::Offer offer,uint64_t loan_ticks) {
    if(prefix.empty() || !offer || !loan_ticks)throw std::runtime_error("invalid native contract pair configuration");
    auto neg=std::make_unique<ContractNegotiator>(capacity,std::move(offer),loan_ticks,false);
    // Canonical ownership must exist before any factory captures the instance.
    // The store lock keeps this insertion private until the transaction returns.
    Pair pair{std::move(neg),prefix+".market",prefix+".worker"};
    const auto inserted=s.by_prefix.emplace(prefix,std::move(pair));
    if(!inserted.second)throw std::runtime_error("contract pair already exists");
    try {
        if(!register_negotiator_tools(tool_registry_global(),*inserted.first->second.owner,prefix))
            throw std::runtime_error("contract negotiator registration failed for prefix '"+prefix+"'");
    } catch(...) {s.by_prefix.erase(inserted.first);throw;}
    // No allocating publication step remains after factories become visible.
    return *inserted.first->second.owner;
}
}
ContractNegotiator& acquire(const std::string& prefix, size_t capacity, uint64_t loan_ticks){
    auto& s=store(); std::lock_guard<std::mutex> lock(s.m);
    auto it=s.by_prefix.find(prefix);
    if(it!=s.by_prefix.end()){
        if(loan_ticks && loan_ticks!=it->second.owner->loan_ticks())
            throw std::runtime_error("contract pair '"+prefix+"' is already acquired with loan_ticks="+
                std::to_string(it->second.owner->loan_ticks())+", refusing loan_ticks="+std::to_string(loan_ticks));
        return *it->second.owner;
    }
    // Same default offer selection as src/headless/contract_service_host.cpp:
    // first Unaccepted entry; graph-bound (non-standalone), default loan ticks.
    return create_locked(s,prefix,capacity,
        [](const Peer&,const std::vector<ContractNegotiator::Entry>& entries){
            for(size_t i=0;i<entries.size();++i)
                if(entries[i].state==ContractNegotiator::State::Unaccepted) return i;
            return size_t(-1);
        },loan_ticks?loan_ticks:60);
}
ContractNegotiator& create_with_offer(const std::string& prefix,size_t capacity,
    ContractNegotiator::Offer offer,uint64_t loan_ticks) {
    auto& s=store();std::lock_guard<std::mutex> lock(s.m);
    if(s.by_prefix.find(prefix)!=s.by_prefix.end())
        throw std::runtime_error("contract pair '"+prefix+"' already exists; Offer replacement refused");
    return create_locked(s,prefix,capacity,std::move(offer),loan_ticks);
}
ContractNegotiator* find(const std::string& prefix){
    auto& s=store(); std::lock_guard<std::mutex> lock(s.m);
    auto it=s.by_prefix.find(prefix); return it==s.by_prefix.end()?nullptr:it->second.owner.get();
}
bool release(const std::string& prefix){
    auto& s=store(); std::lock_guard<std::mutex> lock(s.m);
    auto it=s.by_prefix.find(prefix); if(it==s.by_prefix.end()) return false;
    tool_registry_global().unregister_tool(it->second.market_id);
    tool_registry_global().unregister_tool(it->second.worker_id);
    s.by_prefix.erase(it); return true;
}
}
