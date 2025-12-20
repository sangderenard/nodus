#include "thread_manager.h"

#include "table_abi.h"
#include "stage_abi.h"
#include <chrono>
#include "stage_abi.h"

#include <algorithm>
#include <cmath>
#include <limits>

extern const std::vector<ModuleIORow>* canvas_get_module_io_rows(int module_idx);
extern bool canvas_get_module_input_state(int module_idx, ModuleInputState* out_state);
extern void canvas_clear_module_input_pulses(int module_idx);

// Scheduling helpers used inside run_scheduled_tick.
namespace {

// Kahn topo sort. Returns empty vector on cycle.
static std::vector<int> topo_kahn(const std::vector<std::vector<int>>& succ) {
    int N = (int)succ.size();
    std::vector<int> indeg(N, 0);
    for (int i = 0; i < N; ++i) for (int j : succ[i]) if (j >= 0 && j < N) ++indeg[j];
    std::vector<int> q;
    q.reserve(N);
    for (int i = 0; i < N; ++i) if (indeg[i] == 0) q.push_back(i);
    std::vector<int> order;
    order.reserve(N);
    for (size_t idx = 0; idx < q.size(); ++idx) {
        int v = q[idx];
        order.push_back(v);
        for (int u : succ[v]) {
            if (u < 0 || u >= N) continue;
            --indeg[u];
            if (indeg[u] == 0) q.push_back(u);
        }
    }
    if ((int)order.size() != N) return {};
    return order;
}

// Compute ASAP times using iterative relaxation (works for cyclic graphs bounded by N iterations).
static std::vector<int> compute_asap_iter(int N, const std::vector<std::vector<int>>& succ) {
    std::vector<int> asap(N, 0);
    for (int iter = 0; iter < N; ++iter) {
        bool changed = false;
        for (int v = 0; v < N; ++v) {
            for (int u : succ[v]) {
                if (u < 0 || u >= N) continue;
                int want = asap[v] + 1;
                if (want > asap[u]) { asap[u] = want; changed = true; }
            }
        }
        if (!changed) break;
    }
    return asap;
}

static std::vector<int> compute_alap_iter(int N, const std::vector<std::vector<int>>& succ, int makespan) {
    std::vector<int> alap(N, makespan);
    for (int iter = 0; iter < N; ++iter) {
        bool changed = false;
        for (int v = 0; v < N; ++v) {
            for (int u : succ[v]) {
                if (u < 0 || u >= N) continue;
                int want = alap[u] - 1;
                if (want < alap[v]) { alap[v] = want; changed = true; }
            }
        }
        if (!changed) break;
    }
    return alap;
}

} // namespace


ThreadManager::ThreadManager() = default;

ThreadManager::~ThreadManager() {
    stop();
}

std::shared_ptr<ThreadManager::NetworkSnapshot> ThreadManager::get_table_snapshot(GP_TableContext* table) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = table_snapshots_.find(table);
    if (it == table_snapshots_.end()) return nullptr;
    return it->second;
}

int32_t ThreadManager::find_node_index(GP_TableContext* table, uint64_t key) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = table_snapshots_.find(table);
    if (it == table_snapshots_.end()) return -1;
    const auto &snap = it->second;
    auto nit = snap->node_index_map.find(key);
    if (nit == snap->node_index_map.end()) return -1;
    return static_cast<int32_t>(nit->second);
}

void ThreadManager::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    worker_ = std::thread([this]() { run_loop(); });
}

void ThreadManager::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.clear();
    }
    cv_.notify_all();
    cv_done_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void ThreadManager::set_mode(Mode mode) {
    mode_.store(mode, std::memory_order_relaxed);
}

ThreadManager::Mode ThreadManager::mode() const {
    return mode_.load(std::memory_order_relaxed);
}

void ThreadManager::submit_tick(TickRequest req, bool wait) {
    if (req.tick_id == 0) {
        std::lock_guard<std::mutex> lk(mu_);
        req.tick_id = next_tick_id_++;
    }
    uint64_t wait_for = req.tick_id;
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back(std::move(req));
        ++submitted_;
    }
    cv_.notify_one();
    if (!wait) return;
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&]() { return completed_ >= wait_for || !running_.load(std::memory_order_relaxed); });
}

uint64_t ThreadManager::ticks_submitted() const {
    std::lock_guard<std::mutex> lk(mu_);
    return submitted_;
}

uint64_t ThreadManager::ticks_completed() const {
    std::lock_guard<std::mutex> lk(mu_);
    return completed_;
}

void ThreadManager::run_loop() {
    for (;;) {
        TickRequest req;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&]() { return !queue_.empty() || !running_.load(std::memory_order_relaxed); });
            if (!running_.load(std::memory_order_relaxed) && queue_.empty()) break;
            req = std::move(queue_.front());
            queue_.pop_front();
        }

        // For now both modes execute the same per-tick work; the mode flag is
        // reserved for upcoming scheduling semantics (free-spinning vs imposed
        // deterministic schedule).
        run_scheduled_tick(req);

        {
            std::lock_guard<std::mutex> lk(mu_);
            // Persist snapshot and derived indexing (edges grouped under their writer module).
            last_modules_ = req.modules;
            last_edges_ = req.edges;
            edges_by_writer_module_.clear();
            for (const auto& e : last_edges_) {
                edges_by_writer_module_[e.a_module].push_back(e.edge_idx);
            }
            completed_ = std::max(completed_, req.tick_id);
        }
        // Build and publish per-table network snapshots (manager thread only).
        for (const auto& mod : req.modules) {
            GP_TableContext* t = mod.table;
            if (!t) continue;
            int32_t node_count = 0, edge_count = 0; uint64_t stamp = 0;
            if (!gp_table_snapshot_network_size(t, &node_count, &edge_count, &stamp)) continue;
            auto snap = std::make_shared<ThreadManager::NetworkSnapshot>();
            snap->nodes.resize(static_cast<size_t>(node_count));
            snap->edges.resize(static_cast<size_t>(edge_count));
            if (!gp_table_snapshot_network_fill(t, snap->nodes.data(), node_count, snap->edges.data(), edge_count, stamp)) continue;
            snap->stamp = stamp;
            // Build quick lookup map from node key -> index for UI consumers.
            snap->node_index_map.clear();
            for (uint32_t ni = 0; ni < snap->nodes.size(); ++ni) {
                snap->node_index_map.emplace(snap->nodes[ni], ni);
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                table_snapshots_[t] = snap;
            }
        }
        cv_done_.notify_all();
    }
}

// Reader-table bridge implementation
int ThreadManager::register_reader_for_edge(uint64_t edge_id) {
    std::lock_guard<std::mutex> lk(mu_);
    int slot = ++next_reader_slot_;
    reader_slot_to_edge_.emplace(slot, edge_id);
    // Ensure entry exists
    if (reader_min_seq_by_edge_.find(edge_id) == reader_min_seq_by_edge_.end()) {
        reader_min_seq_by_edge_[edge_id] = 0ull;
    }
    return slot;
}

void ThreadManager::unregister_reader_slot(int slot) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = reader_slot_to_edge_.find(slot);
    if (it == reader_slot_to_edge_.end()) return;
    uint64_t edge_id = it->second;
    reader_slot_to_edge_.erase(it);
    // recompute min for this edge conservatively
    uint64_t minv = UINT64_MAX;
    for (const auto& kv : reader_slot_to_edge_) {
        if (kv.second == edge_id) {
            // unknown per-slot seq; leave as 0 for scaffold
            minv = std::min(minv, reader_min_seq_by_edge_[edge_id]);
        }
    }
    if (minv == UINT64_MAX) reader_min_seq_by_edge_.erase(edge_id);
    else reader_min_seq_by_edge_[edge_id] = minv;
}

uint64_t ThreadManager::min_reader_seq_for_edge(uint64_t edge_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = reader_min_seq_by_edge_.find(edge_id);
    if (it == reader_min_seq_by_edge_.end()) return UINT64_MAX;
    return it->second;
}

void ThreadManager::update_reader_seq(int slot, uint64_t seq) {
    std::lock_guard<std::mutex> lk(mu_);
    reader_slot_seq_[slot] = seq;
    auto it = reader_slot_to_edge_.find(slot);
    if (it == reader_slot_to_edge_.end()) return;
    uint64_t edge_id = it->second;
    uint64_t minv = UINT64_MAX;
    for (const auto &kv : reader_slot_to_edge_) {
        if (kv.second != edge_id) continue;
        int s = kv.first;
        auto sit = reader_slot_seq_.find(s);
        uint64_t sv = (sit != reader_slot_seq_.end()) ? sit->second : 0ull;
        minv = std::min(minv, sv);
    }
    if (minv == UINT64_MAX) reader_min_seq_by_edge_.erase(edge_id);
    else reader_min_seq_by_edge_[edge_id] = minv;
}

// Global instance
static ThreadManager* g_thread_manager_instance = nullptr;
void ThreadManager::set_global(ThreadManager* mgr) { g_thread_manager_instance = mgr; }
ThreadManager* ThreadManager::global() { return g_thread_manager_instance; }

void ThreadManager::run_scheduled_tick(const TickRequest& req) {
        // Apply any queued UI ops on each module's table before building the successor
        // adjacency. This ensures UI-side edits are applied on the manager thread
        // and will be visible to scheduling logic in this tick.
        for (const auto& m : req.modules) {
            if (m.table) gp_table_apply_pending_ops(reinterpret_cast<GP_TableContext*>(m.table));
        }

        // Build successor adjacency from edges (writer -> reader modules) using
    // module vector indices as canonical ids.
    int N = static_cast<int>(req.modules.size());
    std::vector<std::vector<int>> succ(N);
    for (const auto& e : req.edges) {
        if (e.a_module >= 0 && e.a_module < N && e.b_module >= 0 && e.b_module < N) {
            succ[e.a_module].push_back(e.b_module);
        }
    }

    // Prefer topological order when acyclic, otherwise use ASAP/ALAP slack heuristic.
    std::vector<int> order = topo_kahn(succ);
    if (order.empty()) {
        // cyclic: compute asap/alap and order by slack
        auto asap = compute_asap_iter(N, succ);
        int max_asap = 0; for (int v : asap) max_asap = std::max(max_asap, v);
        int makespan = max_asap + N;
        auto alap = compute_alap_iter(N, succ, makespan);
        struct Node { int idx; int slack; int a; };
        std::vector<Node> nodes; nodes.reserve(N);
        for (int i = 0; i < N; ++i) nodes.push_back({i, alap[i] - asap[i], asap[i]});
        std::sort(nodes.begin(), nodes.end(), [](const Node& A, const Node& B){
            if (A.slack != B.slack) return A.slack < B.slack;
            return A.a < B.a;
        });
        order.clear(); order.reserve(N);
        for (auto &n : nodes) order.push_back(n.idx);
    }

    // Execute modules in chosen order.
    // First, execute any stage tasks so their streamed samples arrive into
    // tables before table steps run in this tick. Stage tasks may write
    // into canvas-provided cache buffers (protected by optional mutex).
    for (const auto &st : req.stages) {
        if (!st.stage) continue;
        uint64_t batch_id = static_cast<uint64_t>(std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
        if (st.cache_mu) {
            std::lock_guard<std::mutex> lk(*st.cache_mu);
            gp_stage_perform_tick(reinterpret_cast<GP_StageContext*>(st.stage), st.table, batch_id, nullptr, st.out_rgba, st.out_pitch, st.width, st.height);
        } else {
            gp_stage_perform_tick(reinterpret_cast<GP_StageContext*>(st.stage), st.table, batch_id, nullptr, st.out_rgba, st.out_pitch, st.width, st.height);
        }
    }
    // Per-module, row-sequential, stack-based tool pipeline execution
    // For each module, wire up IO row metadata and FIFO consume/publish
    for (int mod_idx : order) {
        if (mod_idx < 0 || mod_idx >= (int)req.modules.size()) continue;
        const auto& mod = req.modules[static_cast<size_t>(mod_idx)];
        if (!mod.table) continue;
        std::vector<ModuleIORow> io_rows;
        {
            if (const auto* rows = canvas_get_module_io_rows(mod_idx)) {
                io_rows = *rows;
            } else {
                int row_count = gp_table_get_row_count(mod.table);
                io_rows.resize(row_count);
                int output_offset = std::max(0, mod.in_count);
                for (int i = 0; i < row_count; ++i) {
                    ModuleRowKind kind = ModuleRowKind::Tool;
                    if (i < mod.in_count) {
                        kind = ModuleRowKind::Input;
                    } else if (i >= row_count - mod.out_count) {
                        kind = ModuleRowKind::Output;
                    }
                    int contact_idx = (kind == ModuleRowKind::Output)
                        ? (output_offset + (i - (row_count - mod.out_count)))
                        : i;
                    io_rows[i] = ModuleIORow{kind, contact_idx, ModuleToolKind::None, 1};
                }
            }
        }
        int row_count = gp_table_get_row_count(mod.table);
        std::vector<float> stack;
        stack.reserve(32);
        ModuleInputState input_state{};
        bool input_state_loaded = false;
        bool input_state_used = false;
        auto load_input_state = [&]() -> bool {
            if (!input_state_loaded) {
                input_state_loaded = canvas_get_module_input_state(mod_idx, &input_state);
            }
            return input_state_loaded;
        };
        auto pop_value = [&stack]() -> float {
            if (stack.empty()) return 0.0f;
            float v = stack.back();
            stack.pop_back();
            return v;
        };
        for (int row = 0; row < row_count; ++row) {
            ModuleIORow meta = (row < (int)io_rows.size()) ? io_rows[row] : ModuleIORow{ModuleRowKind::Tool, row, ModuleToolKind::None, 0};
            if (meta.kind == ModuleRowKind::Input) {
                // Consume from FIFO for this input contact
                int attachments = std::max(1, meta.attachment_count);
                for (int ai = 0; ai < attachments; ++ai) {
                    float val = 0.0f;
                    int contact_idx = meta.contact_idx + ai;
                    // Find the edge index for this input (mod_idx is the consumer)
                    int edge_idx = -1;
                    uint64_t reader_key = ((uint64_t)mod_idx << 32) | ((uint64_t)contact_idx << 16) | 0u;
                    // Find the edge in req.edges where b_module == mod_idx and b_contact_idx == contact_idx
                    for (const auto& e : req.edges) {
                        if (e.b_module == mod_idx && e.b_contact_idx == contact_idx) {
                            edge_idx = e.edge_idx;
                            break;
                        }
                    }
                    if (edge_idx >= 0) {
                        int32_t unread = 0;
                        gp_table_edge_unread(mod.table, edge_idx, reader_key, &unread);
                        if (unread > 0) {
                            float sample[1] = {0.0f};
                            int32_t written = 0;
                            if (gp_table_edge_consume(mod.table, edge_idx, reader_key, sample, 1, &written) && written > 0) {
                                val = sample[0];
                            }
                        }
                    }
                    stack.push_back(val);
                }
            }
            // Tool row: pass stack through (no-op for now)
            if (meta.kind == ModuleRowKind::Tool) {
                switch (meta.tool) {
                    case ModuleToolKind::KeyboardListener: {
                        float val = 0.0f;
                        if (load_input_state() && input_state.key_event) {
                            val = static_cast<float>(input_state.key);
                        }
                        stack.push_back(val);
                        if (input_state_loaded) input_state_used = true;
                        break;
                    }
                    case ModuleToolKind::MouseListener: {
                        float mx = 0.0f;
                        float my = 0.0f;
                        float down = 0.0f;
                        float up = 0.0f;
                        if (load_input_state()) {
                            mx = input_state.mouse_x;
                            my = input_state.mouse_y;
                            down = input_state.mouse_down ? 1.0f : 0.0f;
                            up = input_state.mouse_up ? 1.0f : 0.0f;
                            input_state_used = true;
                        }
                        stack.push_back(up);
                        stack.push_back(down);
                        stack.push_back(my);
                        stack.push_back(mx);
                        break;
                    }
                    case ModuleToolKind::Add:
                    case ModuleToolKind::Subtract:
                    case ModuleToolKind::Multiply:
                    case ModuleToolKind::Divide:
                    case ModuleToolKind::Modulo:
                    case ModuleToolKind::None:
                    default: {
                        size_t before = stack.size();
                        float b = pop_value();
                        float a = pop_value();
                        switch (meta.tool) {
                            case ModuleToolKind::Add:
                                stack.push_back(a + b);
                                break;
                            case ModuleToolKind::Subtract:
                                stack.push_back(a - b);
                                break;
                            case ModuleToolKind::Multiply:
                                stack.push_back(a * b);
                                break;
                            case ModuleToolKind::Divide:
                                stack.push_back((b == 0.0f) ? 0.0f : (a / b));
                                break;
                            case ModuleToolKind::Modulo:
                                stack.push_back((b == 0.0f) ? 0.0f : std::fmod(a, b));
                                break;
                            case ModuleToolKind::None:
                            default:
                                if (before >= 2) {
                                    stack.push_back(a);
                                    stack.push_back(b);
                                } else if (before == 1) {
                                    stack.push_back(a);
                                }
                                break;
                        }
                        break;
                    }
                }
            }
            if (meta.kind == ModuleRowKind::Output) {
                // Output row: publish from stack to FIFO
                int attachments = std::max(1, meta.attachment_count);
                for (int ai = 0; ai < attachments; ++ai) {
                    float val = pop_value();
                    int contact_idx = meta.contact_idx + ai;
                    int edge_idx = -1;
                    uint64_t writer_key = ((uint64_t)mod_idx << 32) | ((uint64_t)contact_idx << 16) | 0u;
                    // Find the edge in req.edges where a_module == mod_idx and a_contact_idx == contact_idx
                    for (const auto& e : req.edges) {
                        if (e.a_module == mod_idx && e.a_contact_idx == contact_idx) {
                            edge_idx = e.edge_idx;
                            break;
                        }
                    }
                    if (edge_idx >= 0) {
                        int dropped = 0;
                        float payload[1] = {val};
                        gp_table_edge_publish(mod.table, edge_idx, writer_key, payload, 1, &dropped);
                    }
                }
            }
        }
        if (input_state_used) {
            canvas_clear_module_input_pulses(mod_idx);
        }
        if (mod.module_idx >= 0) {
            std::lock_guard<std::mutex> lk(mu_);
            if (static_cast<size_t>(mod.module_idx) >= module_ledger_.size()) {
                module_ledger_.resize(static_cast<size_t>(mod.module_idx) + 1);
            }
            auto& ledger = module_ledger_[static_cast<size_t>(mod.module_idx)];
            ledger.ticks += 1;
            ledger.last_tick_id = req.tick_id;
            ledger.last_dt = req.dt;
        }
    }
}
