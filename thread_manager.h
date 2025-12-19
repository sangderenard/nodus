#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

struct GP_TableContext;

class ThreadManager {
public:
    enum class Mode : int32_t {
        FreeSpinning = 0,
        Scheduled = 1,
    };

    struct ModuleContract {
        int32_t module_idx = -1;
        GP_TableContext* table = nullptr; // non-owning
        int32_t in_count = 0;
        int32_t out_count = 0;
    };

    struct EdgeContract {
        int32_t edge_idx = -1;
        int32_t type_id = 0;
        int32_t a_module = -1;
        int32_t a_contact_idx = -1;
        int32_t b_module = -1;
        int32_t b_contact_idx = -1;
    };

    struct StageContract {
        int32_t module_idx = -1;
        void* stage = nullptr; // GP_StageContext* (opaque here)
        GP_TableContext* table = nullptr; // non-owning
        uint8_t* out_rgba = nullptr; // buffer to write final RGBA into (tight or with pitch)
        int32_t out_pitch = 0; // bytes per row
        int32_t width = 0;
        int32_t height = 0;
        std::mutex* cache_mu = nullptr; // optional mutex protecting the cache buffer
    };

    struct TickRequest {
        uint64_t tick_id = 0;
        double dt = 0.0;
        std::vector<ModuleContract> modules;
        std::vector<EdgeContract> edges;
            std::vector<StageContract> stages;
    };

    ThreadManager();
    ~ThreadManager();

    ThreadManager(const ThreadManager&) = delete;
    ThreadManager& operator=(const ThreadManager&) = delete;

    void start();
    void stop();

    void set_mode(Mode mode);
    Mode mode() const;

    // Submit a tick. If `wait` is true, the call blocks until that tick is completed
    // by the manager thread (safe for now: avoids concurrent mutation with render).
    void submit_tick(TickRequest req, bool wait);

    uint64_t ticks_submitted() const;
    uint64_t ticks_completed() const;

    // Reader-table bridge API: register a reader for a given edge id (opaque),
    // returns a small integer slot >= 1 on success, or 0 on failure.
    int register_reader_for_edge(uint64_t edge_id);
    void unregister_reader_slot(int slot);
    // Return the last-known minimum reader sequence for an edge, or UINT64_MAX
    // if unknown.
    uint64_t min_reader_seq_for_edge(uint64_t edge_id) const;

    // Update the per-slot sequence (called by readers when they advance).
    void update_reader_seq(int slot, uint64_t seq);

    // Global accessor: set/get the process-global ThreadManager instance.
    static void set_global(ThreadManager* mgr);
    static ThreadManager* global();

private:
    struct ModuleLedger {
        uint64_t ticks = 0;
        uint64_t last_tick_id = 0;
        double last_dt = 0.0;
    };

    void run_loop();
    void run_scheduled_tick(const TickRequest& req);

    std::atomic<Mode> mode_{Mode::Scheduled};
    std::atomic<bool> running_{false};
    std::thread worker_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable cv_done_;
    std::deque<TickRequest> queue_;

    uint64_t next_tick_id_ = 1;
    uint64_t submitted_ = 0;
    uint64_t completed_ = 0;

    // "Safety deposit boxes": last-known network snapshot, plus some derived indexing.
    std::vector<ModuleContract> last_modules_;
    std::vector<EdgeContract> last_edges_;
    std::unordered_map<int32_t, std::vector<int32_t>> edges_by_writer_module_;

    // "Time cards": per-module tick ledger.
    std::vector<ModuleLedger> module_ledger_;
    // Reader-table scaffold: map slot -> edge_id and per-edge min seq snapshot.
    int next_reader_slot_ = 0;
    std::unordered_map<int, uint64_t> reader_slot_to_edge_;
    std::unordered_map<int, uint64_t> reader_slot_seq_;
    std::unordered_map<uint64_t, uint64_t> reader_min_seq_by_edge_;
};
