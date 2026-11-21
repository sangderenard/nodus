#pragma once
#include "contract_negotiator.h"
#include <cstdint>
#include "tool_api.h"
#include "tool_registry.h"
namespace nodus::contracts {
// In-image tools: initialization binds the same native ContractNegotiator to both rows.
// No private ValueType registration and no second scheduler.
// The row's plugin source is module_library/source/tools/contract_negotiator.cpp (the Menu lists it, the canvas compiler
// inlines it, it builds as its own plugin DLL). The class stays defined here because the registration below and the
// tests that build with this header alone need it. Saved state: which row it is (market or worker). The
// ContractNegotiator is process state registered by the pair's factory, never saved state; a row made with the
// no-argument constructor has none and refuses in execute_stack.
class NegotiatorTool : public ITool {
public:
    explicit NegotiatorTool(bool market, ContractNegotiator* mailman=nullptr):market_(market),mailman_(mailman){}
    NegotiatorTool()=default;                              // worker row, no negotiator: configure() or the saved state decides the row
    void configure(bool market){market_=market;}
    // Saved state: magic "NGR1", version 1, market (u64 0/1).
    bool serialize(OutputArchive& out) const override {
        const std::uint32_t head[2]={0x31524E47u,1u};
        const std::uint64_t values[1]={market_?1u:0u};
        return out.write(head,sizeof(head))&&out.write(values,sizeof(values));
    }
    bool deserialize(InputArchive& in) override {
        std::uint32_t head[2]={};std::uint64_t values[1]={};
        if(!in.read(head,sizeof(head))||head[0]!=0x31524E47u||head[1]!=1u||!in.read(values,sizeof(values))||values[0]>1)return false;
        market_=values[0]==1;
        return true;
    }
    const char* id_cstr() const noexcept override {return market_?"nodus.contract.exchange":"nodus.contract.route";}
    const char* name_cstr() const noexcept override {return market_?"Contract exchange":"Contract routing";}
    // Graph initialization may supply its own host/canvas user pointer. It is
    // not a domain ContractNegotiator and must never replace factory binding.
    void initialize(const ToolInitContext& ctx) override {
        if(ctx.serialized_path&&ctx.serialized_path[0]!='\0'){
            FileInputArchive in(ctx.serialized_path);
            if(in.ok())deserialize(in);
        }
    }
    void shutdown() override {mailman_=nullptr;}
    void tick(double,HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& context) override {
        if(!mailman_){context.succeeded=false;return;}
        if(market_) mailman_->market_turn(); else mailman_->worker_turn();
    }
    // The pair's result edge is the rows' own channel (research case 075069677cb74e3f9794914c9009cec5): the market row's
    // output port "result" at its right edge, the worker row's input port "result" at its left edge. The edge between
    // them is the pair "<market table>.<row>:result > <worker table>.<row>:result"; the negotiator's bound transport
    // publishes and consumes on it.
    int32_t row_port_count() const override {return 1;}
    ToolRowPort row_port(int32_t idx) const override {
        ToolRowPort p;
        if(idx!=0)return p;
        p.kind=market_?ToolPortKind::Return:ToolPortKind::Argument;
        p.x=market_?1.0f:0.0f; p.y=0.5f; p.name="result"; p.token_bytes=sizeof(Result);
        return p;
    }
private: bool market_=false; ContractNegotiator* mailman_=nullptr;
};
// Register in the SAME native image/registry used by the graph, before ticks.
// Bind the returned IDs as ordinary plugin rows in market -> worker order (or
// with a native dependency edge). ContractNegotiator must outlive every bound row instance.
// Multiple independent pairs, including multiplayer, use distinct prefixes and
// ContractNegotiator instances; no global singleton or ray-specific transport is created.
inline bool register_negotiator_tools(ToolRegistry& registry, ContractNegotiator& mailman,
                                   const std::string& prefix) {
    // Allocate every rollback key before publishing the first factory. The
    // registry is configured on a stopped graph, not concurrently mutated.
    const std::string market_id=prefix+".market",worker_id=prefix+".worker";
    if(prefix.empty() || registry.find(market_id) || registry.find(worker_id)) return false;
    bool market_attempted=false,worker_attempted=false;
    auto rollback=[&] {
        // register_tool can throw after insertion (for example its diagnostics).
        // These names were absent before this transaction; never erase a
        // pre-existing entry or alias when registration is refused.
        if(worker_attempted)registry.unregister_tool(worker_id);
        if(market_attempted)registry.unregister_tool(market_id);
    };
    try {for(bool market : {true,false}){
        ToolRegistry::Entry entry;
        entry.id=market?market_id:worker_id;
        entry.name=market?"Contract market":"Producer mailman";
        entry.source_path="source/tools/contract_negotiator.cpp";
        entry.factory=[&mailman,market](){
            return std::unique_ptr<ITool,std::function<void(ITool*)>>(
                new NegotiatorTool(market,&mailman),[](ITool* tool){delete tool;});
        };
        (market?market_attempted:worker_attempted)=true;
        if(!registry.register_tool(std::move(entry))){rollback();return false;}
    }} catch(...) {rollback();throw;}
    return true;
}
} // namespace nodus::contracts
