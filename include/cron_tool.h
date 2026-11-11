#pragma once
// Cron tool: a contract-negotiator ENGINE (Producer) that submits one contract for every
// scheduled task that comes due, and reasons about the runs a gap made it miss.
//
// It is the mirror of the Turing compile host. That one is a HOST Peer that performs work;
// this one is a Producer that asks for work to be performed. It needs no row of its own:
// `negotiator.add_engine(cron.producer())` is enough, and the existing `<prefix>.worker` row
// calls every engine once per worker turn. Whoever is a host on the prefix performs the task.
//
// Files, all in one directory (cfg.jobs_dir):
//   cron.table              the schedule; one task per line (include/cron_schedule.h)
//   cron.ledger             what has been handled, started and finished; rewritten atomically
//   cron.log                one line per decision the tool made, with its reason
//   <id>.request.json       the request for contract <id>, written BEFORE the contract is
//                           submitted (the compile host's convention). A host reads it by the
//                           contract's id; the contract payload is 0 and is not interpreted.
// Result.payload is the host's status for the run. The tool records it and commits; what a
// status means belongs to the hosts of the prefix.
//
// One run is in flight at a time. While a run is out, later occurrences simply come due and
// are reasoned about when the tool is free again, exactly as occurrences missed while the
// process was down are: `cron::decide` does not care why the tool was not looking.
//
// What "missed" covers:
//   - occurrences that passed while nothing was running (the ledger's `alive` stamp is how a
//     gap is measured and reported);
//   - occurrences that passed while an earlier run held the tool, or while no host took it;
//   - a run that was started and never finished (last_started > last_finished in the ledger):
//     on attach it is run again once unless the task's policy is skip or it is too late.
// A new task starts from the moment it is first seen: nothing before that counts as missed.
//
// Time comes from cfg.now (epoch seconds, UTC). The default is the system clock; a system
// clock port can be bound here when one exists. All methods run on the lane that drives the
// negotiator; nothing here blocks or starts a thread.
#include "contract_negotiator.h"
#include "cron_schedule.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace nodus::cron {

struct CronConfig {
    std::string jobs_dir;
    uint64_t engine_id = 0;                  // must be non-zero; becomes Producer::id
    std::function<int64_t()> now;            // epoch seconds, UTC; default: the system clock
    std::function<bool(uint64_t)> retire;    // retire a committed contract id from the negotiator
    int64_t on_time_s = 45;                  // an occurrence older than this when decided is "missed" (under one minute,
                                             // so an every-minute task has one on-time occurrence, not two)
    int64_t heartbeat_s = 30;                // how often the ledger's `alive` stamp is refreshed
    int64_t gap_s = 120;                     // an `alive` stamp older than this at attach is a gap
};

struct CronStats {
    uint64_t fired = 0, finished = 0, skipped = 0, interrupted = 0;
    uint64_t tasks = 0;
    int64_t gap_seconds = 0;   // how long nothing was running before this attach; 0 = no gap seen
    int in_flight = 0;         // 0 or 1; includes a request prepared but not yet admitted
    uint64_t last_status = 0;
};

class CronTool {
public:
    explicit CronTool(CronConfig cfg) : cfg_(std::move(cfg)) {
        if (!cfg_.now) cfg_.now = [] {
            return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        };
    }
    CronTool(const CronTool&) = delete;
    CronTool& operator=(const CronTool&) = delete;

    // Read the table and the ledger and reason about the gap. False (with a reason) when the
    // table is missing or malformed or the ledger is corrupt: a cron that cannot read its own
    // schedule must not guess one.
    bool open(std::string* why = nullptr) {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(cfg_.jobs_dir, ec);
        std::ifstream in(path("cron.table"), std::ios::binary);
        if (!in) return fail(why, "cron.table is missing in " + cfg_.jobs_dir);
        std::stringstream text; text << in.rdbuf();
        std::string reason;
        if (!parse_table(text.str(), tasks_, &reason)) return fail(why, "cron.table: " + reason);
        if (!load_ledger(path("cron.ledger"), ledger_, &reason)) return fail(why, "cron.ledger: " + reason);
        const int64_t now = cfg_.now();
        if (ledger_.alive > 0 && now - ledger_.alive > cfg_.gap_s) {
            stats_.gap_seconds = now - ledger_.alive;
            log(now, "-", "resumed after a gap of " + std::to_string(stats_.gap_seconds) +
                         " s; nothing was running since " + std::to_string(ledger_.alive) + ".");
        }
        for (const auto& task : tasks_) {
            auto found = ledger_.tasks.find(task.name);
            if (found == ledger_.tasks.end()) {            // a new task starts now
                ledger_.tasks[task.name].last_due = now;
                log(now, task.name, "is new; its schedule starts now and nothing earlier counts as missed.");
                continue;
            }
            const TaskState& state = found->second;
            if (state.last_started > state.last_finished) { // started, never finished
                ++stats_.interrupted;
                const bool too_late = task.max_late_s > 0 && now - state.last_due > task.max_late_s;
                if (task.missed == Missed::Skip || too_late) {
                    log(now, task.name, std::string("had a run interrupted before it finished; it will not be run again (") +
                                       (too_late ? "too late" : "policy is skip") + ").");
                    found->second.last_finished = now;      // settled: do not report it again
                    ++found->second.skipped; ++stats_.skipped;
                } else {
                    Firing again; again.scheduled_for = state.last_due; again.catch_up = true;
                    interrupted_.push_back({task.name, again,
                        "had a run interrupted before it finished; running it again once."});
                    log(now, task.name, "had a run interrupted before it finished; running it again once.");
                }
            }
        }
        stats_.tasks = tasks_.size();
        ledger_.alive = now;
        save();
        open_ = true;
        return true;
    }

    // The Producer borrows *this; keep the tool alive as long as the negotiator has it.
    nodus::contracts::Producer producer() {
        nodus::contracts::Producer p;
        p.id = cfg_.engine_id;
        p.request = [this](nodus::contracts::Contract& out) { return request(out); };
        p.admitted = [this](uint64_t id) { admitted(id); };
        p.route = [this](const nodus::contracts::Result& result) { return route(result); };
        return p;
    }

    CronStats stats() const {
        CronStats s = stats_;
        s.in_flight = (prepared_ || flying_) ? 1 : 0;
        return s;
    }
    const std::vector<Task>& tasks() const { return tasks_; }
    const Ledger& ledger() const { return ledger_; }

private:
    struct Planned { std::string task; Firing firing; std::string reason; };

    std::string path(const std::string& name) const {
        return (std::filesystem::path(cfg_.jobs_dir) / name).string();
    }
    static bool fail(std::string* why, const std::string& text) { if (why) *why = text; return false; }
    void save() { std::string ignored; save_ledger(path("cron.ledger"), ledger_, &ignored); }
    void log(int64_t now, const std::string& task, const std::string& text) {
        std::ofstream out(path("cron.log"), std::ios::binary | std::ios::app);
        if (out) out << now << ' ' << task << ' ' << text << '\n';
    }
    static std::string escape(const std::string& s) {
        std::string o;
        for (char c : s) {
            if (c == '"' || c == '\\') { o += '\\'; o += c; }
            else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
            else o += c;
        }
        return o;
    }

    void retire_committed() {
        if (!cfg_.retire) { committed_.clear(); return; }
        while (!committed_.empty()) {
            if (!cfg_.retire(committed_.front())) break;   // not committed yet: try again next turn
            committed_.pop_front();
        }
    }

    // The next run to ask for, or false when nothing is due. Advances the ledger for decisions
    // that fire nothing (skips); a decision that fires is recorded when its contract is admitted.
    bool plan(int64_t now, Planned& out) {
        if (!interrupted_.empty()) { out = interrupted_.front(); interrupted_.pop_front(); return true; }
        for (const auto& task : tasks_) {
            TaskState& state = ledger_.tasks[task.name];
            Decision d = decide(task, state.last_due, now, cfg_.on_time_s);
            if (d.fire.empty()) {
                if (d.last_due != state.last_due || d.skipped) {
                    state.last_due = d.last_due;
                    state.skipped += d.skipped; stats_.skipped += d.skipped;
                    if (!d.reason.empty()) log(now, task.name, d.reason);
                    save();
                }
                continue;
            }
            out.task = task.name; out.firing = d.fire.front(); out.reason = d.reason;
            // One run now. When a decision holds several runs (policy all), only the first is
            // taken and the ledger advances just past it, so the rest are decided again next time.
            next_last_due_ = d.fire.size() == 1 ? d.last_due : d.fire.front().scheduled_for;
            next_skipped_ = d.skipped;
            return true;
        }
        return false;
    }

    bool request(nodus::contracts::Contract& out) {
        if (!open_) return false;
        retire_committed();
        const int64_t now = cfg_.now();
        if (now - ledger_.alive >= cfg_.heartbeat_s) { ledger_.alive = now; save(); }
        if (flying_) return false;
        if (!prepared_) {
            Planned planned;
            next_last_due_ = 0; next_skipped_ = 0; from_interrupted_ = !interrupted_.empty();
            if (!plan(now, planned)) return false;
            const uint64_t id = (static_cast<uint64_t>(now) << 20) | (++serial_ & 0xFFFFFu);
            std::ofstream f(path(std::to_string(id) + ".request.json"), std::ios::binary | std::ios::trunc);
            if (!f) { log(now, planned.task, "could not write its request file; it will be tried again."); return false; }
            f << "{\"schema\":\"nodus.cron.request.v1\",\"contract\":" << id
              << ",\"task\":\"" << escape(planned.task) << "\""
              << ",\"scheduled_for\":" << planned.firing.scheduled_for
              << ",\"fired_at\":" << now
              << ",\"catch_up\":" << (planned.firing.catch_up ? "true" : "false")
              << ",\"stands_for\":" << planned.firing.stands_for
              << ",\"reason\":\"" << escape(planned.reason) << "\"}\n";
            f.close();
            planned_ = planned; work_ = {id, cfg_.engine_id, 1, 0}; prepared_ = true;
        }
        out = work_;       // peek: the same contract is offered again until it is admitted
        return true;
    }

    void admitted(uint64_t id) {
        if (!prepared_ || id != work_.id) return;
        const int64_t now = cfg_.now();
        TaskState& state = ledger_.tasks[planned_.task];
        if (!from_interrupted_) {
            state.last_due = next_last_due_;
            state.skipped += next_skipped_; stats_.skipped += next_skipped_;
        }
        state.last_started = now;
        ++state.fired; ++stats_.fired;
        if (!planned_.reason.empty()) log(now, planned_.task, planned_.reason);
        ledger_.alive = now;
        save();
        prepared_ = false; flying_ = true;
    }

    bool route(const nodus::contracts::Result& result) {
        const uint64_t id = result.loan.work.id;
        if (flying_ && id == work_.id) {
            const int64_t now = cfg_.now();
            TaskState& state = ledger_.tasks[planned_.task];
            state.last_finished = now < state.last_started ? state.last_started : now;
            state.last_status = result.payload;
            stats_.last_status = result.payload; ++stats_.finished;
            log(now, planned_.task, "finished with status " + std::to_string(result.payload) + ".");
            ledger_.alive = now;
            save();
            flying_ = false;
        }
        committed_.push_back(id);   // ours or stale: it is decided, and is retired on the next turn
        return true;
    }

    CronConfig cfg_;
    std::vector<Task> tasks_;
    Ledger ledger_;
    CronStats stats_;
    std::deque<Planned> interrupted_;
    std::deque<uint64_t> committed_;
    Planned planned_;
    nodus::contracts::Contract work_{};
    int64_t next_last_due_ = 0;
    uint32_t next_skipped_ = 0;
    uint32_t serial_ = 0;
    bool open_ = false, prepared_ = false, flying_ = false, from_interrupted_ = false;
};

} // namespace nodus::cron
