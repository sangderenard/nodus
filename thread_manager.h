#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <array>
#include "table_abi.h"

struct GP_TableContext;

enum class ModuleRowKind : int8_t {
    Input = 0,
    Tool = 1,
    Output = 2,
};

enum class ModuleToolKind : int8_t {
    None = 0,
    Add = 1,
    Subtract = 2,
    Multiply = 3,
    Divide = 4,
    Modulo = 5,
    KeyboardListener = 6,
    MouseListener = 7,
    StackDisplay = 8,
    RectRgba = 9,
    TableNumber = 10,
    Clone = 11,
    TensorAllocator = 12,
    FontRenderer = 13,
};

enum class ModuleToolOrigin : int8_t {
    Builtin = 0,
    Plugin = 1,
};

struct ModuleIORow {
    ModuleRowKind kind = ModuleRowKind::Tool;
    int contact_idx = 0;
    ModuleToolKind tool = ModuleToolKind::None;
    int attachment_count = 1;
    ModuleToolOrigin tool_origin = ModuleToolOrigin::Builtin;
    std::string plugin_id; // non-empty for plugin-origin rows to identify which plugin
};

struct ModuleInputState {
    float mouse_x = 0.0f;
    float mouse_y = 0.0f;
    int32_t mouse_down = 0;
    int32_t mouse_up = 0;
    float mouse_dx = 0.0f;
    float mouse_dy = 0.0f;
    int32_t mouse_button = 0;
    float mouse_scroll = 0.0f;
        uint32_t mouse_button_mask_down = 0u;
        uint32_t mouse_button_mask_up = 0u;
        int32_t mouse_device_id = 0;
    int32_t key = 0;
    int32_t key_event = 0;
};

struct ToolStackSpec {
    int consumes = 0;
    int produces = 0;
};

inline ToolStackSpec module_tool_stack_spec(ModuleToolKind tool) {
    switch (tool) {
        case ModuleToolKind::Add:
        case ModuleToolKind::Subtract:
        case ModuleToolKind::Multiply:
        case ModuleToolKind::Divide:
        case ModuleToolKind::Modulo:
            return {2, 1};
        case ModuleToolKind::KeyboardListener:
            return {0, 1};
        case ModuleToolKind::MouseListener:
            // produces: button_up_mask, button_down_mask, device_id, up, down, y, x, mdx, mdy
            return {0, 9};
        case ModuleToolKind::StackDisplay:
            return {0, 0};
        case ModuleToolKind::RectRgba:
            return {12, 1};
        case ModuleToolKind::TableNumber:
            return {0, 1};
        case ModuleToolKind::Clone:
            return {2, 0};
        case ModuleToolKind::TensorAllocator:
            return {0, 0};
        case ModuleToolKind::None:
        default:
            return {0, 0};
    }
}

class ThreadManager {
public:
    enum class Mode : int32_t {
        FreeSpinning = 0,
        Scheduled = 1,
    };

    // Per-module execution mode hints. 'Slip' allows the module to be
    // executed immediately on worker threads regardless of predecessor
    // readiness (use at your own risk). 'Pooled' permits dispatch into
    // the worker pool when ready. 'Sequential' forces manager-thread
    // execution.
    enum class ExecMode : int8_t {
        Sequential = 0,
        Pooled = 1,
        Slip = 2,
    };

        struct ModuleContract {
            int32_t module_idx = -1;
            GP_TableContext* table = nullptr; // non-owning
            int32_t in_count = 0;
            int32_t out_count = 0;
            int32_t sim_enabled = 1; // 1 = simulate, 0 = skip (rope sim enabled)
            int32_t module_skip = 0; // 1 = skip entire module (no tool/table work)
            int32_t exec_skip_count = 0; // number of frames to skip between executions (0 = every frame)
            ExecMode exec_mode = ExecMode::Pooled;
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
        GP_TableContext* root_table = nullptr;
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
    // Optional per-module timing snapshot structure
    struct ModuleTiming {
        uint64_t run_count = 0;
        double last_run_wall_time = 0.0; // seconds
        double total_run_wall_time = 0.0; // seconds
    };
    // Enable/disable the optional per-module timing database
    void set_timing_enabled(bool v) { timing_enabled_.store(v, std::memory_order_relaxed); }
    bool timing_enabled() const { return timing_enabled_.load(std::memory_order_relaxed); }
    // Retrieve timing snapshot for a module (returns false if index invalid)
    bool get_module_timing(int module_idx, ModuleTiming* out) const;
    // Per-table immutable network snapshots published by the manager thread.
    struct NetworkSnapshot {
        std::vector<uint64_t> nodes;
        std::vector<GP_TableEdgeSnapshot> edges;
        uint64_t stamp = 0;
        std::unordered_map<uint64_t, uint32_t> node_index_map;
    };

    // Retrieve the most recent immutable network snapshot for a table (may return nullptr).
    std::shared_ptr<NetworkSnapshot> get_table_snapshot(GP_TableContext* table) const;

private:
    struct ModuleLedger {
        uint64_t ticks = 0;
        uint64_t last_tick_id = 0;
        double last_dt = 0.0;
        uint64_t tool_cycle = 0;
        uint64_t exec_tick_counter = 0; // persistent counter used to implement exec cadence
        // timing DB fields (updated when timing enabled). Use atomics so updates
        // can be performed without taking the ThreadManager mutex.
        std::atomic<uint64_t> run_count{0};
        // Store wall-time values as integer microseconds to allow atomic adds.
        std::atomic<uint64_t> last_run_wall_time_us{0}; // microseconds
        std::atomic<uint64_t> total_run_wall_time_us{0}; // microseconds
        // Sleep control: until which tick id or until what wall-time (us) this module should be dormant.
        std::atomic<uint64_t> sleep_until_tick{0};
        std::atomic<uint64_t> sleep_until_time_us{0};
        // Provide copy/move semantics for container use (atomics are not copyable by default).
        ModuleLedger() = default;
        ModuleLedger(const ModuleLedger& o) {
            ticks = o.ticks;
            last_tick_id = o.last_tick_id;
            last_dt = o.last_dt;
            tool_cycle = o.tool_cycle;
            exec_tick_counter = o.exec_tick_counter;
            run_count.store(o.run_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            last_run_wall_time_us.store(o.last_run_wall_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
            total_run_wall_time_us.store(o.total_run_wall_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
            sleep_until_tick.store(o.sleep_until_tick.load(std::memory_order_relaxed), std::memory_order_relaxed);
            sleep_until_time_us.store(o.sleep_until_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        ModuleLedger& operator=(const ModuleLedger& o) {
            if (this == &o) return *this;
            ticks = o.ticks;
            last_tick_id = o.last_tick_id;
            last_dt = o.last_dt;
            tool_cycle = o.tool_cycle;
            exec_tick_counter = o.exec_tick_counter;
            run_count.store(o.run_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            last_run_wall_time_us.store(o.last_run_wall_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
            total_run_wall_time_us.store(o.total_run_wall_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
            sleep_until_tick.store(o.sleep_until_tick.load(std::memory_order_relaxed), std::memory_order_relaxed);
            sleep_until_time_us.store(o.sleep_until_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
        ModuleLedger(ModuleLedger&& o) noexcept {
            ticks = o.ticks;
            last_tick_id = o.last_tick_id;
            last_dt = o.last_dt;
            tool_cycle = o.tool_cycle;
            exec_tick_counter = o.exec_tick_counter;
            run_count.store(o.run_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            last_run_wall_time_us.store(o.last_run_wall_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
            total_run_wall_time_us.store(o.total_run_wall_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
            sleep_until_tick.store(o.sleep_until_tick.load(std::memory_order_relaxed), std::memory_order_relaxed);
            sleep_until_time_us.store(o.sleep_until_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
            // leave source in a valid zeroed state
            o.ticks = 0; o.last_tick_id = 0; o.last_dt = 0.0; o.tool_cycle = 0; o.exec_tick_counter = 0;
            o.run_count.store(0, std::memory_order_relaxed);
            o.last_run_wall_time_us.store(0, std::memory_order_relaxed);
            o.total_run_wall_time_us.store(0, std::memory_order_relaxed);
            o.sleep_until_tick.store(0, std::memory_order_relaxed);
            o.sleep_until_time_us.store(0, std::memory_order_relaxed);
        }
        ModuleLedger& operator=(ModuleLedger&& o) noexcept {
            if (this == &o) return *this;
            ticks = o.ticks;
            last_tick_id = o.last_tick_id;
            last_dt = o.last_dt;
            tool_cycle = o.tool_cycle;
            exec_tick_counter = o.exec_tick_counter;
            run_count.store(o.run_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            last_run_wall_time_us.store(o.last_run_wall_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
            total_run_wall_time_us.store(o.total_run_wall_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
            sleep_until_tick.store(o.sleep_until_tick.load(std::memory_order_relaxed), std::memory_order_relaxed);
            sleep_until_time_us.store(o.sleep_until_time_us.load(std::memory_order_relaxed), std::memory_order_relaxed);
            o.ticks = 0; o.last_tick_id = 0; o.last_dt = 0.0; o.tool_cycle = 0; o.exec_tick_counter = 0;
            o.run_count.store(0, std::memory_order_relaxed);
            o.last_run_wall_time_us.store(0, std::memory_order_relaxed);
            o.total_run_wall_time_us.store(0, std::memory_order_relaxed);
            o.sleep_until_tick.store(0, std::memory_order_relaxed);
            o.sleep_until_time_us.store(0, std::memory_order_relaxed);
            return *this;
        }
    };

    // Put a module to sleep for a combination of ticks and/or seconds.
    // If delay_ticks > 0, the module will be skipped until current-next-tick + delay_ticks.
    // If delay_seconds > 0.0, the module will be skipped until the specified wall time has passed.
    void set_module_sleep_delay(int module_idx, uint64_t delay_ticks, double delay_seconds);

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
    
    std::unordered_map<GP_TableContext*, std::shared_ptr<NetworkSnapshot>> table_snapshots_;

    // Helper: find the node index for a given endpoint key in a table snapshot.
    // Returns -1 if not found.
    int32_t find_node_index(GP_TableContext* table, uint64_t key) const;

    // "Time cards": per-module tick ledger.
    std::vector<ModuleLedger> module_ledger_;
    // Optional timing enable flag (controls whether we record wall-time per run)
    std::atomic<bool> timing_enabled_{false};
    // Reader-table scaffold: map slot -> edge_id and per-edge min seq snapshot.
    int next_reader_slot_ = 0;
    std::unordered_map<int, uint64_t> reader_slot_to_edge_;
    std::unordered_map<int, uint64_t> reader_slot_seq_;
    std::unordered_map<uint64_t, uint64_t> reader_min_seq_by_edge_;
    // Last applied RGBA per toolbar subgroup (to avoid repeated reapplication when peeking)
    std::unordered_map<int, std::array<float,4>> last_applied_rgba_subgroup_;
    // Per-edge assembly buffer when producer emits stride<4 and color is encoded
    // across multiple successive samples (e.g., stride==1 sends R,G,B,A as four samples).
    std::unordered_map<int, std::vector<float>> edge_assemble_buf_;
    // Cache of last-known stride per edge (0 = unknown)
    std::unordered_map<int, int> edge_stride_cache_;
};
