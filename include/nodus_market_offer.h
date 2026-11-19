#pragma once
// The market's placement across local devices and volunteers (volunteer case a11b5c9b1be74c9b9fb684790f7fd5a3, stage 4).
// resident_fabric_offer (fabric_resident_offer.h) ranks a visited host's candidate work: deadline risk first (largest
// projected lateness), then full residency, then most resident bytes, FIFO otherwise; and once every idle compatible
// host has at least three matching timings, only the fastest of them is offered the job (warm hosts first). It reads
// residency and timings from the in-process library, so it cannot see a remote host. market_offer keeps that ranking
// unchanged and reads the same facts through a MarketView, which the server fills for local and volunteer hosts alike.
#include "contract_negotiator.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace nodus::contracts {

struct MarketJob {
    // Absolute steady-clock deadline and this host's recent end-to-end completion for this job (0 = unknown).
    uint64_t expires_at_ns = 0, recent_completion_ns = 0;
    // The job's state on this host: bytes already resident of the bytes it needs (0 of 0 = nothing to keep).
    uint64_t resident_bytes = 0, total_bytes = 0;
};
struct MarketTiming { uint64_t samples = 0, average_total_ns = 0; };
struct MarketView {
    std::vector<uint64_t> hosts;                                                       // every host in the market
    std::function<bool(uint64_t host, const ContractNegotiator::Entry&)> compatible;   // may this host run it at all
    std::function<MarketJob(uint64_t host, const ContractNegotiator::Entry&)> job;
    std::function<MarketTiming(uint64_t host, const ContractNegotiator::Entry&)> timing;
    std::function<uint64_t()> now_ns = [] {
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    };
};

inline ContractNegotiator::Offer market_offer(MarketView view) {
    return [view = std::move(view)](const Peer& peer, const std::vector<ContractNegotiator::Entry>& entries) {
        const auto now = view.now_ns();
        size_t best = SIZE_MAX;
        uint64_t best_bytes = 0, best_lateness = 0;
        bool best_full = false, best_at_risk = false;
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto& entry = entries[i];
            if (entry.state != ContractNegotiator::State::Unaccepted || !view.compatible(peer.id, entry)) continue;
            const MarketJob job = view.job(peer.id, entry);
            // The speed gate: among the idle compatible hosts, once all are measured, only the fastest gets the job.
            struct Cost { uint64_t id, total_ns; bool warm; };
            std::vector<Cost> costs; bool all_measured = true;
            for (uint64_t host : view.hosts) {
                if (!view.compatible(host, entry)) continue;
                bool busy = false;
                for (const auto& other : entries) if (other.state == ContractNegotiator::State::Loaned && other.loan.host == host) { busy = true; break; }
                if (busy) continue;
                const auto t = view.timing(host, entry);
                if (t.samples < 3) { all_measured = false; break; }
                const auto j = view.job(host, entry);
                costs.push_back({host, t.average_total_ns, j.total_bytes && j.resident_bytes == j.total_bytes});
            }
            if (all_measured && !costs.empty()) {
                const bool warm_exists = std::any_of(costs.begin(), costs.end(), [](const Cost& c) { return c.warm; });
                const auto fastest = std::min_element(costs.begin(), costs.end(), [warm_exists](const Cost& a, const Cost& b) {
                    if (warm_exists && a.warm != b.warm) return a.warm;
                    if (a.total_ns != b.total_ns) return a.total_ns < b.total_ns;
                    return a.id < b.id;
                });
                if (fastest->id != peer.id) continue;
            }
            const uint64_t completion = job.recent_completion_ns > std::numeric_limits<uint64_t>::max() - now
                                        ? std::numeric_limits<uint64_t>::max() : now + job.recent_completion_ns;
            const bool at_risk = job.expires_at_ns && completion >= job.expires_at_ns;
            const uint64_t lateness = at_risk ? completion - job.expires_at_ns : 0;
            const bool full = job.total_bytes && job.resident_bytes == job.total_bytes;
            if (best == SIZE_MAX || (at_risk && !best_at_risk) ||
                (at_risk == best_at_risk && ((at_risk && lateness > best_lateness) ||
                                             (!at_risk && ((full && !best_full) || (full == best_full && job.resident_bytes > best_bytes)))))) {
                best = i; best_at_risk = at_risk; best_lateness = lateness; best_full = full; best_bytes = job.resident_bytes;
            }
        }
        return best;
    };
}

}  // namespace nodus::contracts
