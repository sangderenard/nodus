#pragma once
// Native domain policy; transport is the existing lossless Nodus FIFO.
// Call all methods on the owning ThreadManager lane. Callbacks are nonblocking
// mailbox exchanges, never synchronous HTTP. Payload refs are engine-owned IDs,
// not pointers; engines retain their real ray state until route() commits it.
#include "nodus_runtime_abi.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nodus::contracts {
struct Contract { uint64_t id=0, engine=0, revision=0, payload=0; };
struct Loan { Contract work{}; uint64_t generation=0, host=0, deadline=0; };
struct Result { Loan loan{}; uint64_t payload=0; };
struct Exchange { uint64_t accepted_generation=0; bool has_result=false; Result result{}; };
// An exchange first returns the host's previous completed result, and may
// accept the new offer. Rejected offers leave the contract unaccepted.
struct Peer {
    uint64_t id=0;
    std::function<Exchange(const Loan*)> exchange;
    // Receipt acknowledges FIFO acceptance only, never estimator commit.
    std::function<void(uint64_t, bool)> receipt;
};
struct Producer {
    uint64_t id=0;
    std::function<bool(const Result&)> route; // true only after engine commit
    std::function<bool(Contract&)> request; // peek only; engine retains until admitted
    std::function<void(uint64_t)> admitted; // optional admission acknowledgement
};
class ContractNegotiator {
public:
    enum class State { Unaccepted, Loaned, Returned, Committed };
    struct Entry { Contract work; State state=State::Unaccepted; Loan loan{}; };
    // Operator chooses an unaccepted index from the retained stack. SIZE_MAX
    // means no offer. Pricing/pressure is policy, not guessed by transport.
    using Offer = std::function<size_t(const Peer&, const std::vector<Entry>&)>;
    ContractNegotiator(size_t capacity, Offer offer, uint64_t loan_ticks=60, bool standalone=true)
        : capacity_(capacity), offer_(std::move(offer)), loan_ticks_(loan_ticks) {
        if (!capacity || !offer_ || !loan_ticks) throw std::invalid_argument("mailman configuration");
        if(!standalone)return;
        fifo_=nodus_edge_runtime_create();
        int32_t dims[1]={1};
        if (!fifo_ || !nodus_edge_runtime_configure(fifo_,dims,1,capacity,0,sizeof(Result),0x534A5231,2,8)
            || !nodus_edge_runtime_subscribe(fifo_,1,1)) {
            if(fifo_) nodus_edge_runtime_destroy(fifo_);
            throw std::runtime_error("Nodus result FIFO unavailable");
        }
    }
    ~ContractNegotiator(){ nodus_edge_runtime_destroy(fifo_); }
    ContractNegotiator(const ContractNegotiator&)=delete; ContractNegotiator& operator=(const ContractNegotiator&)=delete;
    // Bind declared graph-edge transport before any contracts are admitted.
    // This replaces, rather than mirrors, the standalone native edge.
    bool bind_result_transport(std::function<bool(const Result&)> publish,
                               std::function<bool(Result&)> consume){
        if(!entries_.empty() || pending_ || !publish || !consume)return false;
        publish_=std::move(publish);consume_=std::move(consume);
        if(fifo_){nodus_edge_runtime_destroy(fifo_);fifo_=nullptr;}
        return true;
    }
    bool add_host(Peer host){
        if(!host.id || !host.exchange) return false;
        for(auto& h:hosts_) if(h.id==host.id) return false;
        hosts_.push_back(std::move(host)); offers_.push_back({}); return true;
    }
    bool add_engine(Producer engine){
        if(!engine.id || !engine.route || !engine.request) return false;
        for(auto& e:engines_) if(e.id==engine.id) return false;
        engines_.push_back(std::move(engine)); return true;
    }
    // REMOVAL. Peers are held by index (hosts_/offers_/engines_ vectors, round-robin
    // cursors, and references held across the exchange/route/request callbacks), while
    // contracts and loans name peers by id. So removal never mutates the vectors
    // mid-turn: called from inside market_turn()/worker_turn() (e.g. from a callback)
    // it is DEFERRED to the end of that turn; called between turns it applies at once.
    // Both return true if the id is registered (or scheduled) and false otherwise.
    //
    // remove_engine(id): the engine is unregistered; its contracts that are not in
    // flight (Unaccepted, Committed) are dropped. A contract that is Loaned stays until
    // its host answers: the host completes normally, but the result has no engine to
    // route to and is DISCARDED (discarded_results() counts it; the entry is erased); a
    // loan that expires meanwhile is dropped instead of re-offered. A Returned contract
    // (result already published, not yet routed) is discarded the same way when its
    // result is consumed. The id may then be added again.
    //
    // remove_host(id): the host is unregistered together with its retained offer. Each
    // Loaned contract it held returns to Unaccepted with its loan cleared and is
    // re-offered to another host under a NEW generation (reoffered_loans() counts
    // them); the removed host can no longer deliver, and a late result carrying the old
    // generation could not match anyway. One case is left untouched: a Returned
    // contract (the host's result was already accepted and published) keeps going to
    // its engine, since that work is complete.
    bool remove_engine(uint64_t id){
        if(in_turn_){ if(!has_engine(id))return false; deferred_.push_back({false,id}); return true; }
        return apply_remove_engine(id);
    }
    bool remove_host(uint64_t id){
        if(in_turn_){ if(!has_host(id))return false; deferred_.push_back({true,id}); return true; }
        return apply_remove_host(id);
    }
    uint64_t discarded_results() const{return discarded_results_;}
    uint64_t reoffered_loans() const{return reoffered_loans_;}
    uint64_t loan_ticks() const{return loan_ticks_;}
    size_t host_count() const{return hosts_.size();}
    size_t engine_count() const{return engines_.size();}
    bool submit(Contract work){
        if(!work.id || entries_.size()>=capacity_) return false;
        bool known=false; for(auto& e:engines_) known|=e.id==work.engine;
        if(!known || find(work.id)) return false;
        entries_.push_back({work}); return true;
    }
    // Owner rejects a validated-but-uncommitted result; release FIFO head and reoffer.
    bool reject_pending(uint64_t id,uint64_t generation){
        if(!pending_ || pending_result_.loan.work.id!=id || pending_result_.loan.generation!=generation)return false;
        auto* e=find(id);if(!e || e->state!=State::Returned)return false;
        e->state=State::Unaccepted;pending_=false;return true;
    }
    bool retire(uint64_t id){
        for(auto it=entries_.begin();it!=entries_.end();++it) if(it->work.id==id){
            if(it->state!=State::Committed) return false;
            entries_.erase(it); return true;
        } return false;
    }
    const std::vector<Entry>& contracts() const {return entries_;}
    uint64_t turns() const{return turn_;}
    // One connection per turn, including server loopback. Deadline is measured
    // in these market turns, independent of callback duration or wall-clock.
    void market_turn(){
        TurnGuard guard(*this);
        ++turn_;
        for(auto& e:entries_) if(e.state==State::Loaned && turn_>=e.loan.deadline)
            e.state=State::Unaccepted;
        // Work of a removed engine whose loan expired has nowhere to go.
        entries_.erase(std::remove_if(entries_.begin(),entries_.end(),[&](const Entry& e){
            return e.state==State::Unaccepted && !has_engine(e.work.engine);}),entries_.end());
        if(hosts_.empty()) return;
        const size_t host_index=host_cursor_++ % hosts_.size();
        auto& host=hosts_[host_index];
        auto& retained=offers_[host_index];
        auto* candidate=find(retained.work.id);
        if(!candidate || candidate->state!=State::Unaccepted || turn_>=retained.deadline)
            retained={};
        bool host_busy=false;
        for(const auto& e:entries_)if(e.state==State::Loaned && e.loan.host==host.id)host_busy=true;
        if(!retained.generation && !host_busy){
            auto eligible=entries_;
            // Give another participant first chance after this host missed its lease.
            if(hosts_.size()>1)for(auto& e:eligible)
                if(e.state==State::Unaccepted && e.loan.host==host.id && e.loan.generation && turn_>=e.loan.deadline && turn_-e.loan.deadline<loan_ticks_)e.state=State::Loaned;
            for(size_t other=0;other<offers_.size();++other){
                const auto& reserved=offers_[other];
                if(other==host_index || !reserved.generation || turn_>=reserved.deadline)continue;
                for(auto& e:eligible)if(e.work.id==reserved.work.id && e.state==State::Unaccepted)e.state=State::Loaned;
            }
            size_t chosen=offer_(host,eligible);
            if(chosen<entries_.size() && eligible[chosen].state==State::Unaccepted && entries_[chosen].state==State::Unaccepted)
                retained={entries_[chosen].work,++generation_,host.id,turn_+loan_ticks_};
        }
        const Loan offered=retained;
        Exchange reply=host.exchange(offered.generation?&offered:nullptr);
        // Async replies name their exact proposal; another peer may have
        // accepted this contract while this response was in flight.
        candidate=find(offered.work.id);
        if(offered.generation && reply.accepted_generation==offered.generation &&
           candidate && candidate->state==State::Unaccepted && turn_<offered.deadline){
            candidate->loan=offered;candidate->state=State::Loaned;retained={};
        }
        if(reply.has_result){
            bool received=reply.result.loan.host==host.id && accept_result(reply.result);
            if(host.receipt) host.receipt(reply.result.loan.generation,received);
        }
    }
    // Route one completed FIFO result and check in with one engine per turn.
    // A refusing engine leaves the result retained, rather than losing it.
    void worker_turn(){
        TurnGuard guard(*this);
        if(!pending_){size_t written=0;
            if(consume_){pending_=consume_(pending_result_);written=pending_?sizeof(Result):0;}
            else pending_=nodus_edge_runtime_consume(fifo_,1,&pending_result_,sizeof(Result),&written)!=0;
            if(pending_ && written!=sizeof(Result)) throw std::runtime_error("Nodus token size mismatch");
        }
        if(pending_){
            bool routed_to_known=false;
            for(auto& engine:engines_) if(engine.id==pending_result_.loan.work.engine){
                routed_to_known=true;
                if(engine.route(pending_result_)){
                    auto* entry=find(pending_result_.loan.work.id);
                    if(entry) entry->state=State::Committed;
                    pending_=false;
                } break;
            }
            if(!routed_to_known){ // engine removed while its work was in flight
                const uint64_t id=pending_result_.loan.work.id;
                for(auto it=entries_.begin();it!=entries_.end();++it)if(it->work.id==id){entries_.erase(it);break;}
                pending_=false;++discarded_results_;
            }
        }
        if(engines_.empty() || entries_.size()>=capacity_) return;
        auto& engine=engines_[engine_cursor_++ % engines_.size()]; Contract work{};
        if(engine.request(work) && work.engine==engine.id && submit(work) && engine.admitted)
            engine.admitted(work.id);
    }
private:
    struct Removal{bool host;uint64_t id;};
    struct TurnGuard{
        ContractNegotiator& n; bool outer;
        explicit TurnGuard(ContractNegotiator& x):n(x),outer(!x.in_turn_){x.in_turn_=true;}
        ~TurnGuard(){
            if(!outer)return;
            n.in_turn_=false;
            auto todo=std::move(n.deferred_);n.deferred_.clear();
            for(const auto& r:todo){ if(r.host)n.apply_remove_host(r.id); else n.apply_remove_engine(r.id); }
        }
    };
    bool has_engine(uint64_t id) const{for(auto& e:engines_)if(e.id==id)return true;return false;}
    bool has_host(uint64_t id) const{for(auto& h:hosts_)if(h.id==id)return true;return false;}
    bool apply_remove_engine(uint64_t id){
        for(auto it=engines_.begin();it!=engines_.end();++it) if(it->id==id){
            engines_.erase(it);
            entries_.erase(std::remove_if(entries_.begin(),entries_.end(),[&](const Entry& e){
                return e.work.engine==id && (e.state==State::Unaccepted || e.state==State::Committed);}),entries_.end());
            return true;
        } return false;
    }
    bool apply_remove_host(uint64_t id){
        for(size_t i=0;i<hosts_.size();++i) if(hosts_[i].id==id){
            hosts_.erase(hosts_.begin()+i); offers_.erase(offers_.begin()+i);
            for(auto& e:entries_) if(e.state==State::Loaned && e.loan.host==id){
                e.state=State::Unaccepted; e.loan={}; ++reoffered_loans_;
            }
            return true;
        } return false;
    }
    Entry* find(uint64_t id){for(auto& e:entries_)if(e.work.id==id)return &e;return nullptr;}
    bool accept_result(const Result& result){
        auto* e=find(result.loan.work.id);
        if(!e || e->state!=State::Loaned) return false;
        const auto& a=e->loan; const auto& b=result.loan;
        if(a.generation!=b.generation || a.host!=b.host || a.deadline!=b.deadline ||
           a.work.engine!=b.work.engine || a.work.revision!=b.work.revision ||
           a.work.payload!=b.work.payload || turn_>=a.deadline) return false;
        int32_t dropped=0;
        if(publish_){if(!publish_(result)) return false;}
        else if(!nodus_edge_runtime_publish(fifo_,1,&result,sizeof(result),&dropped)) return false;
        if(dropped) throw std::runtime_error("lossless result FIFO dropped a token");
        e->state=State::Returned; return true;
    }
    std::function<bool(const Result&)> publish_; std::function<bool(Result&)> consume_;
    size_t capacity_; Offer offer_; uint64_t loan_ticks_,turn_=0,generation_=0;
    size_t host_cursor_=0,engine_cursor_=0; NodusEdgeRuntime* fifo_=nullptr;
    std::vector<Peer> hosts_; std::vector<Loan> offers_; std::vector<Producer> engines_; std::vector<Entry> entries_;
    bool pending_=false; Result pending_result_{};
    bool in_turn_=false; std::vector<Removal> deferred_; uint64_t discarded_results_=0,reoffered_loans_=0;
};
} // namespace nodus::contracts




