#pragma once
// Turing compile host: a contract-negotiator HOST Peer that consumes Turing
// compile requests strictly one at a time, never eagerly, never blocking the
// lane that drives it.
//
// Wiring: no row of its own. `negotiator.add_host(host.peer())` is enough; the
// existing `<prefix>.market` row calls every host once per market turn, and each
// call is one non-blocking exchange().
//
// Request convention: the contract's `id` names the work. The request JSON for
// a contract is `<jobs_dir>/<id>.request.json` (the contract payload is carried
// through untouched and is NOT interpreted). The product directory is
// `<jobs_dir>/<id>/`. Whoever submits contract id N must first place the
// request file; a missing request is refused with RequestMissing.
//
// Per exchange (all on the calling lane; only the compile driver wait runs on
// the single worker thread):
//   1. a finished job waits  -> lane-side adoption (if cfg.adopt), join the
//      thread, return that job's Result (payload = Status). No new offer taken.
//   2. a job is running      -> empty Exchange (offer not accepted).
//   3. offer non-null        -> start the job (or finish it immediately when the
//      product receipt already exists / the request is missing) and accept
//      the offer's generation. The result arrives on a LATER exchange.
// Loan deadlines count market turns: pick loan_ticks large enough for a compile.
//
// Restart: if `<product_dir>/nodus-row-tool-product.receipt` already exists the
// compiler is NOT run again (work is not repeated; the contract still completes
// with ExistingProduct, and is adopted on the lane when cfg.adopt).
// `<product_dir>/compile-host.result.json` records the outcome.
#include "contract_negotiator.h"
#include "turing_product_adopter.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace nodus::contracts {

struct CompileHostConfig {
    std::string python_executable, nodus_root, turing_root, module_library_root, jobs_dir;
    uint64_t host_id = 0;   // must be non-zero; becomes Peer::id
    bool adopt = true;      // lane-side gp_turing_product_adopt_and_load after compile
};

enum class CompileStatus : uint64_t {
    Adopted = 1,
    CompiledNotAdopted = 2,
    ExistingProduct = 3,
    RequestMissing = 16,
    CompileFailed = 17,
    AdoptFailed = 18,
    Cancelled = 19,
};

struct CompileHostStats {
    uint64_t accepted = 0, completed = 0, refused = 0, acknowledged = 0, rejected_receipts = 0;
    int in_flight = 0; // 0 or 1; includes a finished-but-not-yet-delivered job
};

class TuringCompileHost {
public:
    explicit TuringCompileHost(CompileHostConfig cfg) : cfg_(std::move(cfg)) {}
    ~TuringCompileHost() { shutdown(); }
    TuringCompileHost(const TuringCompileHost&) = delete;
    TuringCompileHost& operator=(const TuringCompileHost&) = delete;

    // The Peer borrows *this; keep the host alive as long as the negotiator.
    Peer peer() {
        Peer p;
        p.id = cfg_.host_id;
        p.exchange = [this](const Loan* offer) { return exchange(offer); };
        p.receipt = [this](uint64_t, bool accepted) {
            ++stats_.acknowledged;
            if (!accepted) ++stats_.rejected_receipts;
        };
        return p;
    }

    // Cancel the running child, join the worker. Never detaches.
    void shutdown() {
        if (thread_.joinable()) {
            cancel_.requested.store(true);
            thread_.join();
        }
        active_ = false;
        done_.store(false);
    }

    CompileHostStats stats() const {
        CompileHostStats s = stats_;
        s.in_flight = active_ ? 1 : 0;
        return s;
    }

private:
    using Clock = std::chrono::system_clock;
    static double epoch(Clock::time_point t) {
        return std::chrono::duration<double>(t.time_since_epoch()).count();
    }

    Exchange exchange(const Loan* offer) {
        Exchange out;
        if (active_) {
            if (!done_.load()) return out; // running: accept nothing
            return deliver();
        }
        if (!offer) return out;
        namespace fs = std::filesystem;
        job_ = Job{};
        job_.loan = *offer;
        job_.queued = Clock::now();
        const std::string id = std::to_string(offer->work.id);
        job_.request = (fs::path(cfg_.jobs_dir) / (id + ".request.json")).string();
        job_.product = (fs::path(cfg_.jobs_dir) / id).string();
        std::error_code ec;
        slot_ = Slot{};
        job_.started = job_.queued;
        if (fs::is_regular_file(fs::path(job_.product) / "nodus-row-tool-product.receipt", ec)) {
            slot_.rc = 1; job_.existing = true;
            finish_now();
        } else if (!fs::is_regular_file(job_.request, ec)) {
            slot_.rc = -1; // request missing
            ++stats_.refused;
            finish_now();
        } else {
            cancel_.requested.store(false);
            done_.store(false);
            active_ = true;
            job_.started = Clock::now();
            thread_ = std::thread([this] {
                char err[512] = {};
                int rc = gp_turing_row_tool_compile_only(
                    cfg_.python_executable.c_str(), cfg_.nodus_root.c_str(), job_.request.c_str(),
                    cfg_.turing_root.c_str(), job_.product.c_str(), &cancel_, err, sizeof(err));
                {
                    std::lock_guard<std::mutex> lock(slot_mutex_);
                    slot_.rc = rc; slot_.error = err;
                }
                done_.store(true);
            });
        }
        ++stats_.accepted;
        out.accepted_generation = offer->generation;
        return out;
    }

    void finish_now() { active_ = true; done_.store(true); }

    Exchange deliver() {
        Exchange out;
        if (thread_.joinable()) thread_.join();
        Slot slot;
        { std::lock_guard<std::mutex> lock(slot_mutex_); slot = slot_; }
        CompileStatus status;
        GP_TuringProductAdoptionReport report{};
        char tool_id[1024] = {}, staged[2048] = {};
        std::string error = slot.error;
        if (slot.rc == -1) { status = CompileStatus::RequestMissing; error = "request file missing"; }
        else if (slot.rc == 2) status = CompileStatus::Cancelled;
        else if (slot.rc != 1) status = CompileStatus::CompileFailed;
        else if (!cfg_.adopt) status = job_.existing ? CompileStatus::ExistingProduct : CompileStatus::CompiledNotAdopted;
        else {
            const auto receipt = (std::filesystem::path(job_.product) / "nodus-row-tool-product.receipt").string();
            const int ok = gp_turing_product_adopt_and_load(
                receipt.c_str(), cfg_.nodus_root.c_str(), cfg_.module_library_root.c_str(),
                &report, tool_id, sizeof(tool_id), staged, sizeof(staged));
            status = !ok ? CompileStatus::AdoptFailed
                         : (job_.existing ? CompileStatus::ExistingProduct : CompileStatus::Adopted);
        }
        const auto finished = Clock::now();
        write_result_json(status, tool_id, error, report, finished);
        ++stats_.completed;
        active_ = false; done_.store(false);
        out.has_result = true;
        out.result = Result{job_.loan, static_cast<uint64_t>(status)};
        return out;
    }

    void write_result_json(CompileStatus status, const char* tool_id, const std::string& error,
                           const GP_TuringProductAdoptionReport& r, Clock::time_point finished) {
        std::error_code ec;
        std::filesystem::create_directories(job_.product, ec);
        std::ofstream f(std::filesystem::path(job_.product) / "compile-host.result.json",
                        std::ios::binary | std::ios::trunc);
        if (!f) return;
        auto esc = [](const std::string& s) {
            std::string o;
            for (char c : s) { if (c == '"' || c == '\\') { o += '\\'; o += c; }
                               else if (static_cast<unsigned char>(c) < 0x20) o += ' '; else o += c; }
            return o;
        };
        f << "{\"status\":" << static_cast<uint64_t>(status)
          << ",\"contract_id\":" << job_.loan.work.id
          << ",\"generation\":" << job_.loan.generation
          << ",\"tool_id\":\"" << esc(tool_id) << "\""
          << ",\"error\":\"" << esc(error) << "\""
          << ",\"queued_epoch_s\":" << epoch(job_.queued)
          << ",\"started_epoch_s\":" << epoch(job_.started)
          << ",\"finished_epoch_s\":" << epoch(finished)
          << ",\"wall_seconds\":" << std::chrono::duration<double>(finished - job_.started).count()
          << ",\"adoption\":{\"receipt_validated\":" << r.receipt_validated
          << ",\"wrapper_imported\":" << r.wrapper_imported
          << ",\"wrapper_staged\":" << r.wrapper_staged
          << ",\"plugin_loaded\":" << r.plugin_loaded << "}}\n";
    }

    struct Job { Loan loan{}; std::string request, product; Clock::time_point queued{}, started{}; bool existing = false; };
    struct Slot { int rc = 0; std::string error; };

    CompileHostConfig cfg_;
    Job job_;
    Slot slot_;                       // written by the worker only under slot_mutex_ while running
    std::mutex slot_mutex_;
    std::atomic<bool> done_{false};
    bool active_ = false;             // lane-only
    GP_TuringCompileCancel cancel_;
    std::thread thread_;
    CompileHostStats stats_;          // lane-only
};

} // namespace nodus::contracts
