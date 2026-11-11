#include "contract_negotiator_registry.h"
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
namespace nodus::contracts {
namespace {
struct Store { std::mutex m; std::map<std::string,std::unique_ptr<ContractNegotiator>> by_prefix; };
Store& store(){ static Store s; return s; }
}
ContractNegotiator& acquire(const std::string& prefix, size_t capacity, uint64_t loan_ticks){
    auto& s=store(); std::lock_guard<std::mutex> lock(s.m);
    auto it=s.by_prefix.find(prefix);
    if(it!=s.by_prefix.end()){
        if(loan_ticks && loan_ticks!=it->second->loan_ticks())
            throw std::runtime_error("contract pair '"+prefix+"' is already acquired with loan_ticks="+
                std::to_string(it->second->loan_ticks())+", refusing loan_ticks="+std::to_string(loan_ticks));
        return *it->second;
    }
    // Same default offer selection as src/headless/contract_service_host.cpp:
    // first Unaccepted entry; graph-bound (non-standalone), default loan ticks.
    auto neg=std::make_unique<ContractNegotiator>(capacity,
        [](const Peer&,const std::vector<ContractNegotiator::Entry>& entries){
            for(size_t i=0;i<entries.size();++i)
                if(entries[i].state==ContractNegotiator::State::Unaccepted) return i;
            return size_t(-1);
        },loan_ticks?loan_ticks:60,false);
    if(!register_negotiator_tools(tool_registry_global(),*neg,prefix))
        throw std::runtime_error("contract negotiator registration failed for prefix '"+prefix+"'");
    return *(s.by_prefix[prefix]=std::move(neg));
}
ContractNegotiator* find(const std::string& prefix){
    auto& s=store(); std::lock_guard<std::mutex> lock(s.m);
    auto it=s.by_prefix.find(prefix); return it==s.by_prefix.end()?nullptr:it->second.get();
}
bool release(const std::string& prefix){
    auto& s=store(); std::lock_guard<std::mutex> lock(s.m);
    auto it=s.by_prefix.find(prefix); if(it==s.by_prefix.end()) return false;
    tool_registry_global().unregister_tool(prefix+".market");
    tool_registry_global().unregister_tool(prefix+".worker");
    s.by_prefix.erase(it); return true;
}
}
