#pragma once
#include "contract_negotiator.h"
#include "tool_api.h"
#include "tool_registry.h"
namespace nodus::contracts {
// In-image tools: initialization binds the same native ContractNegotiator to both rows.
// No private ValueType registration and no second scheduler.
class NegotiatorTool : public ITool {
public:
    explicit NegotiatorTool(bool market, ContractNegotiator* mailman=nullptr):market_(market),mailman_(mailman){}
    const char* id_cstr() const noexcept override {return market_?"nodus.contract.exchange":"nodus.contract.route";}
    const char* name_cstr() const noexcept override {return market_?"Contract exchange":"Contract routing";}
    // Graph initialization may supply its own host/canvas user pointer. It is
    // not a domain ContractNegotiator and must never replace factory binding.
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {mailman_=nullptr;}
    void tick(double,HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& context) override {
        if(!mailman_){context.succeeded=false;return;}
        if(market_) mailman_->market_turn(); else mailman_->worker_turn();
    }
private: bool market_; ContractNegotiator* mailman_=nullptr;
};
// Register in the SAME native image/registry used by the graph, before ticks.
// Bind the returned IDs as ordinary plugin rows in market -> worker order (or
// with a native dependency edge). ContractNegotiator must outlive every bound row instance.
// Multiple independent pairs, including multiplayer, use distinct prefixes and
// ContractNegotiator instances; no global singleton or ray-specific transport is created.
inline bool register_negotiator_tools(ToolRegistry& registry, ContractNegotiator& mailman,
                                   const std::string& prefix) {
    if(prefix.empty() || registry.find(prefix+".market") || registry.find(prefix+".worker")) return false;
    for(bool market : {true,false}){
        ToolRegistry::Entry entry;
        entry.id=prefix+(market?".market":".worker");
        entry.name=market?"Contract market":"Producer mailman";
        entry.factory=[&mailman,market](){
            return std::unique_ptr<ITool,std::function<void(ITool*)>>(
                new NegotiatorTool(market,&mailman),[](ITool* tool){delete tool;});
        };
        if(!registry.register_tool(std::move(entry))){
            registry.unregister_tool(prefix+".market");return false;
        }
    }
    return true;
}
} // namespace nodus::contracts
