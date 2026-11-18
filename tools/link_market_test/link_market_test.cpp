// link_market_test: the contract-negotiator work market over the Nodus link (volunteer case
// a11b5c9b1be74c9b9fb684790f7fd5a3, stage 1). The real ContractNegotiator runs on the "server"; one engine submits
// contracts; two hosts compete for them: a local host and a volunteer that answers over a link (LinkContractHost on the
// dialer side, LinkHostPeer on the server side). Every result must route back to the engine, from both hosts.
#include "nodus_link_contracts.h"
#include "contract_negotiator.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <map>
#include <set>

using namespace nodus::link;
using nodus::contracts::Contract;
using nodus::contracts::ContractNegotiator;
using nodus::contracts::Producer;
using Clock = std::chrono::steady_clock;

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) { if (!ok) { ++failures; std::printf("FAIL %s\n", what); } };

    // The link: listener (server) and dialer (volunteer) on loopback, tickets for account 42.
    Key key{}; random_bytes(key.data(), key.size());
    ListenerConfig lc; lc.prefix = "market"; lc.key = key;
    LinkListener listener(lc);
    const auto id = new_link_id(); const uint64_t now = uint64_t(std::time(nullptr));
    const auto ct = sign_ticket(make_ticket(Channel::Control, Role::Dialer, id, "market", 42, now, 60), key);
    const auto bt = sign_ticket(make_ticket(Channel::Bulk, Role::Dialer, id, "market", 42, now, 60), key);
    DialTarget t; t.port = listener.port(); t.prefix = "market";
    LinkDialer dialer(t, ct, bt);
    std::unique_ptr<Link> server_link, volunteer_link;
    for (auto end = Clock::now() + std::chrono::seconds(5); Clock::now() < end && !(server_link && volunteer_link);) {
        dialer.poll(); listener.poll(uint64_t(std::time(nullptr)));
        if (!volunteer_link && dialer.ready()) volunteer_link = dialer.take();
        if (!server_link) server_link = listener.take();
        if (volunteer_link) volunteer_link->poll();
    }
    check(server_link && volunteer_link, "link up");
    if (!server_link || !volunteer_link) return 1;

    // The server's market: the real negotiator, first-unaccepted offers, one engine, two hosts.
    ContractNegotiator market(16, [](const nodus::contracts::Peer&, const std::vector<ContractNegotiator::Entry>& entries) {
        for (size_t i = 0; i < entries.size(); ++i) if (entries[i].state == ContractNegotiator::State::Unaccepted) return i;
        return std::numeric_limits<size_t>::max();
    }, 10000);
    ControlMux server_mux(*server_link), volunteer_mux(*volunteer_link);
    LinkHostPeer volunteer_peer(server_mux, server_link->account());   // host id = the account the ticket names
    check(market.add_host(volunteer_peer.peer()), "volunteer host added");

    // A local host: accepts at once, returns 3 x payload on its next visit.
    std::optional<Loan> local_loan; bool local_result_due = false;
    nodus::contracts::Peer local; local.id = 7;
    local.exchange = [&](const Loan* offer) {
        Exchange e{};
        if (local_loan && local_result_due) { e.has_result = true; e.result = {*local_loan, local_loan->work.payload * 3}; local_loan.reset(); local_result_due = false; }
        else if (local_loan) local_result_due = true;
        if (offer && !local_loan) { local_loan = *offer; e.accepted_generation = offer->generation; }
        return e;
    };
    check(market.add_host(local), "local host added");

    // The volunteer's work: 3 x payload, done a few polls after it starts (it is not instant).
    std::map<uint64_t, int> volunteer_progress;
    LinkContractHost volunteer(volunteer_mux,
        [&](const Loan& loan) { volunteer_progress[loan.generation] = 0; return true; },
        [&](const Loan& loan, uint64_t& out) {
            if (++volunteer_progress[loan.generation] < 20) return false;
            out = loan.work.payload * 3; return true;
        });

    // The engine: 12 contracts, payloads 100..111; route checks each result.
    constexpr int kContracts = 12;
    int submitted = 0, routed = 0, wrong = 0;
    std::map<uint64_t, int> by_host;
    Producer engine; engine.id = 900;
    engine.request = [&](Contract& c) {
        if (submitted >= kContracts) return false;
        c = {uint64_t(5000 + submitted), 900, 1, uint64_t(100 + submitted)}; return true;
    };
    engine.admitted = [&](uint64_t) { ++submitted; };
    engine.route = [&](const Result& r) {
        if (r.payload != r.loan.work.payload * 3) ++wrong;
        ++by_host[r.loan.host]; ++routed; return true;
    };
    check(market.add_engine(engine), "engine added");

    const auto t0 = Clock::now();
    for (auto end = t0 + std::chrono::seconds(20); Clock::now() < end && routed < kContracts;) {
        market.worker_turn(); market.market_turn();
        server_link->poll(); volunteer_link->poll();
        volunteer.poll();
    }
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("market over the link: %d contracts routed in %.3f s; local host %d, volunteer (account %llu) %d; wrong results %d\n",
                routed, s, by_host[7], (unsigned long long)server_link->account(), by_host[server_link->account()], wrong);
    std::printf("volunteer: accepted %llu, results %llu, receipts %llu\n", (unsigned long long)volunteer.accepted(),
                (unsigned long long)volunteer.results(), (unsigned long long)volunteer.receipts());
    check(routed == kContracts && wrong == 0, "every contract routed with the right result");
    check(by_host[server_link->account()] > 0 && by_host[7] > 0, "both hosts did work");
    check(volunteer.receipts() == volunteer.results(), "every volunteer result receipted");
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
