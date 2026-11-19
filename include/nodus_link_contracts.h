#pragma once
// The contract-negotiator work market over the Nodus link (volunteer case a11b5c9b1be74c9b9fb684790f7fd5a3, stage 1).
// A volunteer joins the server's market as a HOST peer, exactly as a loopback ContractSocket host does: the server's
// ContractNegotiator offers it loans, it accepts and returns results, and the server sends receipts. The frames are the
// contract socket's own 72-byte encoding (WireSocket::encode/decode), each carried as one ControlKind::Contract packet
// on the link's control pair; the link has already authenticated the peer (its ticket names the account), which takes
// the place of the loopback Hello. Large inputs and outputs named by a contract travel on the bulk pair (later stages).
//
//   server:    LinkHostPeer adapts a Link to ContractNegotiator::Peer (ContractSocket::exchange, transport swapped)
//   volunteer: LinkContractHost answers offers on its Link: accept one loan at a time, run it, return the result
// Both share a ControlMux, so contract frames and other control packets can use the same pair. Nothing blocks.
#include "nodus_link.h"
#include "contract_socket.h"

#include <deque>
#include <functional>
#include <map>
#include <optional>

namespace nodus::link {
using contracts::Exchange;
using contracts::Loan;
using contracts::Peer;
using contracts::Result;
using contracts::WireKind;
using contracts::WireMessage;
using contracts::WireSocket;

// Splits a link's control.in by declared ControlKind; sends are unchanged.
class ControlMux {
public:
    explicit ControlMux(Link& link) : link_(link) {}
    Link& link() { return link_; }
    void pump() {
        ControlPacket p;
        while (link_.receive_control(p)) {
            auto& q = queues_[p.kind];
            if (q.size() >= kControlQueuePackets) { ++overflow_; continue; }   // bounded; counted, never silent
            q.push_back(p);
        }
    }
    bool take(ControlKind kind, ControlPacket& p) {
        pump();
        auto it = queues_.find(uint16_t(kind));
        if (it == queues_.end() || it->second.empty()) return false;
        p = it->second.front(); it->second.pop_front(); return true;
    }
    uint64_t overflow() const { return overflow_; }

private:
    Link& link_;
    std::map<uint16_t, std::deque<ControlPacket>> queues_;
    uint64_t overflow_ = 0;
};

inline bool send_frame(Link& link, const WireMessage& m) {
    auto frame = WireSocket::encode(m);
    ControlPacket p; p.kind = uint16_t(ControlKind::Contract); p.length = uint32_t(frame.size());
    std::memcpy(p.payload.data(), frame.data(), frame.size());
    return link.send_control(p);
}
inline bool take_frame(ControlMux& mux, WireMessage& m) {
    ControlPacket p;
    if (!mux.take(ControlKind::Contract, p)) return false;
    if (p.length != WireSocket::Frame{}.size()) return false;
    WireSocket::Frame f{}; std::memcpy(f.data(), p.payload.data(), f.size());
    return WireSocket::decode(f, m);
}

// Many market hosts share one link (one per device, stage 4): frames are routed by the loan's host id, which the
// negotiator sets on every offer and every frame here carries back (Accept, Result, Receipt).
class ContractRouter {
public:
    explicit ContractRouter(ControlMux& mux) : mux_(mux) {}
    Link& link() { return mux_.link(); }
    bool take(uint64_t host, WireMessage& m) {
        WireMessage f{};
        while (take_frame(mux_, f)) queues_[f.result.loan.host].push_back(f);
        auto it = queues_.find(host);
        if (it == queues_.end() || it->second.empty()) return false;
        m = it->second.front(); it->second.pop_front(); return true;
    }

private:
    ControlMux& mux_;
    std::map<uint64_t, std::deque<WireMessage>> queues_;
};

// ---- server side: the volunteer as a ContractNegotiator::Peer ---------------------------------------------------
// ContractSocket::exchange with the link as transport: send an offer once per generation, return one response per
// visit (an Accept or a Result), forward receipts. `host_id` is the account the link's ticket names.
class LinkHostPeer {
public:
    // `on_accept` runs when the volunteer accepts a loan: the moment its inputs must go to it (the bulk pair).
    LinkHostPeer(ContractRouter& router, uint64_t host_id, std::function<void(const Loan&)> on_accept = {})
        : router_(router), id_(host_id), on_accept_(std::move(on_accept)) {}
    Peer peer() {
        Peer p; p.id = id_;
        p.exchange = [this](const Loan* offer) { return exchange(offer); };
        p.receipt = [this](uint64_t generation, bool accepted) { receipt(generation, accepted); };
        return p;
    }
    Exchange exchange(const Loan* offer) {
        if (offer && offer->generation != last_offer_) {
            if (send_frame(router_.link(), {WireKind::Offer, {*offer, 0}})) { last_offer_ = offer->generation; offered_ = *offer; }
        }
        Exchange reply{};
        WireMessage m{};
        if (router_.take(id_, m)) {
            if (m.kind == WireKind::Accept) {
                reply.accepted_generation = m.result.loan.generation;
                if (on_accept_ && reply.accepted_generation == offered_.generation) on_accept_(offered_);
            }
            else if (m.kind == WireKind::Result) { reply.has_result = true; reply.result = m.result; }
        }
        return reply;
    }
    void receipt(uint64_t generation, bool accepted) {
        WireMessage m{WireKind::Receipt, {}}; m.result.loan.generation = generation; m.result.loan.host = id_; m.result.payload = accepted ? 1 : 0;
        send_frame(router_.link(), m);
    }
    bool closed() { return router_.link().closed(); }

private:
    ContractRouter& router_;
    uint64_t id_;
    uint64_t last_offer_ = 0;
    Loan offered_{};
    std::function<void(const Loan&)> on_accept_;
};

// ---- volunteer side: a host that serves the server's market over its link --------------------------------------
// One loan at a time, like ServicePeer. `start(loan)` begins the work and returns false to decline it (the offer stays
// unaccepted on the server); `poll(loan, payload)` returns true once the work is done with the result payload.
class LinkContractHost {
public:
    using Start = std::function<bool(const Loan&)>;
    using Poll = std::function<bool(const Loan&, uint64_t& result_payload)>;
    // `host_id` is this host's id in the server's market (one per device); 0 serves every frame (a single host).
    LinkContractHost(ContractRouter& router, uint64_t host_id, Start start, Poll poll)
        : router_(router), host_(host_id), start_(std::move(start)), poll_(std::move(poll)) {}
    uint64_t id() const { return host_; }
    void poll() {
        WireMessage m{};
        while (router_.take(host_, m)) {
            if (m.kind == WireKind::Offer) {
                const Loan& offer = m.result.loan;
                if (working_ || waiting_receipt_) continue;                 // one loan at a time; the offer expires
                if (offer.generation == declined_) continue;
                if (!start_(offer)) { declined_ = offer.generation; continue; }
                if (!send_frame(router_.link(), {WireKind::Accept, {offer, 0}})) continue;
                loan_ = offer; working_ = true; ++accepted_;
            } else if (m.kind == WireKind::Receipt) {
                if (waiting_receipt_ && m.result.loan.generation == loan_.generation) { waiting_receipt_ = false; ++receipts_; }
            }
        }
        uint64_t payload = 0;
        if (working_ && poll_(loan_, payload)) {
            if (send_frame(router_.link(), {WireKind::Result, {loan_, payload}})) { working_ = false; waiting_receipt_ = true; ++results_; }
        }
    }
    bool busy() const { return working_ || waiting_receipt_; }
    uint64_t accepted() const { return accepted_; }
    uint64_t results() const { return results_; }
    uint64_t receipts() const { return receipts_; }

private:
    ContractRouter& router_;
    uint64_t host_;
    Start start_; Poll poll_;
    Loan loan_{};
    bool working_ = false, waiting_receipt_ = false;
    uint64_t declined_ = 0, accepted_ = 0, results_ = 0, receipts_ = 0;
};

}  // namespace nodus::link
