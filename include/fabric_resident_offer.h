#pragma once
#include "contract_negotiator.h"
#include "tensor_state_library_tool.h"

#include <functional>
#include <algorithm>
#include <optional>
#include <chrono>
#include <limits>
#include <vector>
#include <utility>

namespace nodus::gpu {

struct FabricStateRef {
    std::uint64_t card=0, base=0, fields=0;
    // Absolute steady-clock nanoseconds and the recent end-to-end completion
    // average for this job/device, including a cold transfer when applicable.
    std::uint64_t expires_at_ns=0, recent_completion_ns=0;
    // Exact semantic program and invocation shape for comparable measurements.
    // Zero preserves the original residency/deadline-only policy.
    std::uint64_t program_id=0,lanes=0;
};

// Policy for the existing ContractNegotiator. The caller retains the mapping
// from opaque contract payload IDs to library states and peers to GPU slots.
// This only reads residency: the selected host borrows after accepting a loan.
inline contracts::ContractNegotiator::Offer resident_fabric_offer(
    TensorStateLibraryTool& library,
    std::function<std::optional<std::size_t>(std::uint64_t)> peer_device,
    std::function<std::optional<FabricStateRef>(std::uint64_t)> payload_state,
    std::function<std::uint64_t()> now_ns=[] {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    },
    std::vector<std::pair<std::uint64_t,std::size_t>> known_peers={}) {
    return [&library,peer_device=std::move(peer_device),
            payload_state=std::move(payload_state),now_ns=std::move(now_ns),
            known_peers=std::move(known_peers)](
        const contracts::Peer& peer,
        const std::vector<contracts::ContractNegotiator::Entry>& entries) {
        if(!peer_device||!payload_state||!now_ns)return SIZE_MAX;
        const auto device=peer_device(peer.id);
        if(!device)return SIZE_MAX;
        const auto now=now_ns();
        std::size_t best=SIZE_MAX;
        std::uint64_t best_bytes=0;
        std::uint64_t best_lateness=0;
        bool best_full=false,best_at_risk=false;
        for(std::size_t i=0;i<entries.size();++i) {
            const auto& entry=entries[i];
            if(entry.state!=contracts::ContractNegotiator::State::Unaccepted)
                continue;
            const auto state=payload_state(entry.work.payload);
            if(!state||!state->expires_at_ns||!state->recent_completion_ns)
                continue;
            const TensorStateLibraryTool::Candidate candidate{
                state->base,state->fields};
            const auto choice=library.choose_resident(state->card,*device,
                std::span<const TensorStateLibraryTool::Candidate>(&candidate,1));
            if(!choice)continue;
            if(state->program_id&&state->lanes&&!known_peers.empty()){
                struct PeerCost {std::uint64_t id,total_ns,resident,total;};
                std::vector<PeerCost> costs;
                bool all_measured=true;
                for(const auto& [host_id,slot]:known_peers){
                    bool busy=false;
                    for(const auto& other:entries)
                        if(other.state==contracts::ContractNegotiator::State::Loaned&&
                           other.loan.host==host_id){busy=true;break;}
                    if(busy)continue;
                    const auto available=library.choose_resident(state->card,slot,
                        std::span<const TensorStateLibraryTool::Candidate>(&candidate,1));
                    if(!available)continue;
                    const auto metric=library.work_performance(
                        TensorStateLibraryTool::ExecutionApi::Vulkan,slot,
                        state->program_id,state->lanes);
                    if(!metric||metric->samples<3){all_measured=false;break;}
                    costs.push_back({host_id,metric->average_total_ns,
                        available->resident_bytes,available->total_bytes});
                }
                // Until all compatible idle hosts have matching observations,
                // every host gets its ordinary market turn for exploration.
                if(all_measured&&!costs.empty()){
                    const bool warm_exists=std::any_of(costs.begin(),costs.end(),
                        [](const PeerCost& cost){return cost.resident==cost.total;});
                    const auto best_peer=std::min_element(costs.begin(),costs.end(),
                        [warm_exists](const PeerCost& a,const PeerCost& b){
                            const bool a_warm=a.resident==a.total;
                            const bool b_warm=b.resident==b.total;
                            if(warm_exists&&a_warm!=b_warm)return a_warm;
                            if(a.total_ns!=b.total_ns)return a.total_ns<b.total_ns;
                            return a.id<b.id;
                        });
                    if(best_peer->id!=peer.id)continue;
                }
            }
            const auto completion=state->recent_completion_ns>
                std::numeric_limits<std::uint64_t>::max()-now
                ?std::numeric_limits<std::uint64_t>::max()
                :now+state->recent_completion_ns;
            const bool at_risk=completion>=state->expires_at_ns;
            const auto lateness=at_risk?completion-state->expires_at_ns:0;
            const bool full=choice->resident_bytes==choice->total_bytes;
            if(best==SIZE_MAX||(at_risk&&!best_at_risk)||
               (at_risk==best_at_risk &&
                ((at_risk&&lateness>best_lateness)||
                 (!at_risk&&((full&&!best_full)||
                   (full==best_full&&choice->resident_bytes>best_bytes)))))) {
                best=i;best_at_risk=at_risk;best_lateness=lateness;
                best_full=full;best_bytes=choice->resident_bytes;
            }
        }
        return best;
    };
}

} // namespace nodus::gpu
