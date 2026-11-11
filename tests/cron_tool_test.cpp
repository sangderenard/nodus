// Cron tool against the real ContractNegotiator: a fake host performs the work and a fake
// clock moves time, so a three-day gap and an interrupted run happen in milliseconds.
#include "cron_tool.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

using namespace nodus::contracts;
using namespace nodus::cron;
namespace fs = std::filesystem;

#define CHECK(x) do { if (!(x)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

namespace {
std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary); std::stringstream s; s << in.rdbuf(); return s.str();
}
bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

// One negotiator, one host that answers each accepted loan with `status` on its next exchange
// (or never, while `hold` is set), and the cron tool as the only engine.
struct Rig {
    int64_t& clock;
    fs::path dir;
    uint64_t status = 7;
    bool hold = false;
    std::vector<uint64_t> performed;     // contract ids the host finished, in order
    Loan loan{};
    std::unique_ptr<ContractNegotiator> box;
    std::unique_ptr<CronTool> cron;

    Rig(int64_t& c, fs::path d) : clock(c), dir(std::move(d)) {
        auto choose = [](const Peer&, const std::vector<ContractNegotiator::Entry>& entries) {
            for (size_t i = 0; i < entries.size(); ++i)
                if (entries[i].state == ContractNegotiator::State::Unaccepted) return i;
            return size_t(-1);
        };
        box = std::make_unique<ContractNegotiator>(4, choose, 1000);
        Peer host; host.id = 1;
        host.exchange = [this](const Loan* offered) {
            Exchange e;
            if (loan.generation && !hold) {
                e.has_result = true; e.result = {loan, status};
                performed.push_back(loan.work.id); loan = {};
            } else if (offered && !loan.generation) {
                loan = *offered; e.accepted_generation = offered->generation;
            }
            return e;
        };
        box->add_host(std::move(host));
    }
    bool attach(std::string* why = nullptr) {
        CronConfig cfg;
        cfg.jobs_dir = dir.string(); cfg.engine_id = 9;
        cfg.now = [this] { return clock; };
        cfg.retire = [this](uint64_t id) { return box->retire(id); };
        cron = std::make_unique<CronTool>(std::move(cfg));
        if (!cron->open(why)) return false;
        return box->add_engine(cron->producer());
    }
    void turns(int n) { for (int i = 0; i < n; ++i) { box->worker_turn(); box->market_turn(); } }
    std::string request_of(uint64_t id) const { return slurp(dir / (std::to_string(id) + ".request.json")); }
};
}

int main() {
    const fs::path root = fs::temp_directory_path() / "nodus_cron_tool_test";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);

    // 2026-11-11 12:00:30 UTC. 20768 days from 1970-01-01 to 2026-11-11.
    const int64_t day = 86400, noon = 20768 * day + 12 * 3600;
    int64_t clock = noon + 30;

    // ---- a missing or malformed table is refused; the tool never guesses a schedule
    {
        Rig rig(clock, root / "empty");
        std::string why;
        CHECK(!rig.attach(&why) && has(why, "cron.table"));
        fs::create_directories(root / "bad");
        std::ofstream(root / "bad" / "cron.table") << "nightly 99 3 * * *\n";
        Rig bad(clock, root / "bad");
        CHECK(!bad.attach(&why) && has(why, "cron.table"));
    }

    const fs::path dir = root / "jobs";
    fs::create_directories(dir);
    std::ofstream(dir / "cron.table")
        << "# three tasks, three policies\n"
        << "tick    * * * * *\n"
        << "nightly 0 3 * * *   missed=once\n"
        << "ping    */5 * * * * missed=skip\n"
        << "report  0 6 * * *   missed=all max_catch_up=2\n";

    // ---- a new schedule starts now: nothing earlier is missed, nothing fires at once
    {
        Rig rig(clock, dir);
        CHECK(rig.attach());
        CHECK(rig.cron->stats().tasks == 4 && rig.cron->stats().gap_seconds == 0);
        rig.turns(6);
        CHECK(rig.performed.empty() && rig.cron->stats().fired == 0);

        // ---- one minute later the every-minute task fires once, on time
        clock = noon + 65;
        rig.turns(8);
        CHECK(rig.performed.size() == 1 && rig.cron->stats().fired == 1 && rig.cron->stats().finished == 1);
        const std::string request = rig.request_of(rig.performed[0]);
        CHECK(has(request, "\"task\":\"tick\"") && has(request, "\"catch_up\":false"));
        CHECK(has(request, "\"scheduled_for\":" + std::to_string(noon + 60)));
        CHECK(rig.cron->stats().last_status == 7);
        CHECK(rig.box->contracts().empty());            // the committed contract was retired
        const auto& state = rig.cron->ledger().tasks.at("tick");
        CHECK(state.last_due == noon + 60 && state.last_status == 7 && state.last_finished >= state.last_started);

        // ---- the same minute does not fire twice
        rig.turns(8);
        CHECK(rig.performed.size() == 1);

        // ---- ping comes due at 12:05 and is on time: skip only applies to missed runs
        clock = noon + 5 * 60 + 10;
        rig.turns(16);
        bool ping_ran = false;
        for (uint64_t id : rig.performed) ping_ran |= has(rig.request_of(id), "\"task\":\"ping\"");
        CHECK(ping_ran);
    }

    // ---- three days with nothing running, then a new process reasons about the gap
    {
        const size_t before = 0;
        clock = noon + 3 * day + 20;                     // 12:00:20, three days on
        Rig rig(clock, dir);
        CHECK(rig.attach());
        CHECK(rig.cron->stats().gap_seconds > 3 * day - 3600);
        rig.turns(60);
        int nightly = 0, tick = 0, ping = 0, report = 0;
        std::string nightly_request, tick_request;
        std::vector<int64_t> report_times;
        for (size_t i = before; i < rig.performed.size(); ++i) {
            const std::string r = rig.request_of(rig.performed[i]);
            if (has(r, "\"task\":\"nightly\"")) { ++nightly; nightly_request = r; }
            if (has(r, "\"task\":\"tick\"")) { ++tick; tick_request = r; }
            if (has(r, "\"task\":\"ping\"")) ++ping;
            if (has(r, "\"task\":\"report\"")) {
                ++report;
                const auto at = r.find("\"scheduled_for\":");
                report_times.push_back(std::stoll(r.substr(at + 16)));
            }
        }
        // once: one run stands for the three nights (03:00 on days 1, 2, 3)
        CHECK(nightly == 1 && has(nightly_request, "\"catch_up\":true") && has(nightly_request, "\"stands_for\":2"));
        CHECK(has(nightly_request, "\"scheduled_for\":" + std::to_string(noon - 9 * 3600 + 3 * day)));
        // default policy is once: the every-minute task runs once, not 4,300 times
        CHECK(tick == 1 && has(tick_request, "\"stands_for\":"));
        // skip: the missed five-minute pings never run; the on-time 12:00 one does
        CHECK(ping == 1);
        CHECK(rig.cron->ledger().tasks.at("ping").skipped > 800);
        // all with a cap of 2: the two most recent mornings, oldest first; the third is skipped
        CHECK(report == 2 && report_times[0] == noon - 6 * 3600 + 2 * day && report_times[1] == noon - 6 * 3600 + 3 * day);
        CHECK(rig.cron->ledger().tasks.at("report").skipped == 1);
        CHECK(has(slurp(dir / "cron.log"), "resumed after a gap"));
        CHECK(rig.box->contracts().empty());

        // ---- a run that is started and never finished
        clock += 60;
        rig.hold = true;
        rig.turns(10);
        CHECK(rig.cron->stats().in_flight == 1);
        const auto& state = rig.cron->ledger().tasks.at("tick");
        CHECK(state.last_started > state.last_finished);
    }
    {
        clock += 30;                                     // the process comes back half a minute later
        Rig rig(clock, dir);
        CHECK(rig.attach());
        CHECK(rig.cron->stats().interrupted == 1);
        rig.turns(10);
        CHECK(rig.performed.size() == 1);
        const std::string again = rig.request_of(rig.performed[0]);
        CHECK(has(again, "\"task\":\"tick\"") && has(again, "\"catch_up\":true") && has(again, "interrupted"));
        const auto& state = rig.cron->ledger().tasks.at("tick");
        CHECK(state.last_finished >= state.last_started);
    }

    // ---- a request is offered again, unchanged, until it is admitted
    {
        clock += 120;
        CronConfig cfg; cfg.jobs_dir = dir.string(); cfg.engine_id = 9; cfg.now = [&] { return clock; };
        CronTool tool(std::move(cfg));
        CHECK(tool.open());
        Producer p = tool.producer();
        Contract a{}, b{};
        CHECK(p.request(a) && p.request(b) && a.id == b.id && a.id != 0 && a.engine == 9);
        p.admitted(a.id);
        Contract c{};
        CHECK(!p.request(c));                            // one run at a time
        Result done{}; done.loan.work = a; done.payload = 3;
        CHECK(p.route(done) && tool.stats().finished == 1 && tool.stats().last_status == 3);
    }

    fs::remove_all(root, ec);
    std::puts("OK cron tool: new schedule, on-time run, gap reasoning by policy, interrupted run, retained request");
    return 0;
}
