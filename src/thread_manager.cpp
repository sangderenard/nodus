#include "thread_manager.h"

#include "graph_runtime.h"
#include "table_abi.h"
#include "stage_abi.h"
#include "canvas_abi.h"
#include "tool_api.h"
#include "table_tensor_tool.h"
#include "value_types.h"
#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_registry.h"
#include "common/tensors/abstraction/tensor_types.h"

#ifndef NODUS_ENABLE_TORCH
#define NODUS_ENABLE_TORCH 0
#endif

#include <chrono>
#include "stage_abi.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include "console_logger.h"
#ifndef printf
#define printf(...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
#ifndef fprintf
#define fprintf(file, ...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
#ifndef NODUS_DEBUG_MOUSE_LISTENER
#define NODUS_DEBUG_MOUSE_LISTENER 0
#endif
#if NODUS_DEBUG_MOUSE_LISTENER
#define ML_DEBUGF(...) fprintf(stderr, __VA_ARGS__)
#else
#define ML_DEBUGF(...) do {} while (0)
#endif

// EventPayload is declared in canvas_abi.h and used for pointer-mode FIFO
// events published by the canvas.

extern bool canvas_get_module_input_state(int module_idx, ModuleInputState* out_state);
extern void canvas_clear_module_input_pulses(int module_idx);
extern void canvas_set_module_stack_snapshot(int module_idx, int row_idx, const float* values, int count);
extern void canvas_set_module_stack_snapshot_meta(int module_idx, int row_idx, const int* types, const int* counts, int count);

namespace {
GP_TableTensorToolState* get_tensor_tool_state() {
    static GP_TableTensorToolState* state = gp_table_tensor_tool_create(nullptr);
    return state;
}
} // namespace
extern void canvas_set_module_stack_tail(int module_idx, const float* values, int count);

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

static bool point_in_rounded_rect(float px, float py, float w, float h, float r) {
    if (w <= 0.0f || h <= 0.0f) return false;
    r = std::clamp(r, 0.0f, 0.5f * std::min(w, h));
    float dx = 0.0f;
    float dy = 0.0f;
    if (px < r) {
        dx = r - px;
    } else if (px > w - r) {
        dx = px - (w - r);
    }
    if (py < r) {
        dy = r - py;
    } else if (py > h - r) {
        dy = py - (h - r);
    }
    return (dx * dx + dy * dy) <= (r * r);
}

// Helper: check whether an edge's tensor stride is large enough to safely carry a pointer value.
static bool edge_stride_allows_pointer(GP_TableContext* table, int edge_idx) {
    if (!table || edge_idx < 0) return false;
    GP_TableEdgeTensorSpecTyped spec{};
    if (!gp_table_edge_get_tensor_spec(table, edge_idx, &spec)) return false;
    uint64_t stride = 1;
    for (int32_t di = 0; di < spec.dim_count; ++di) {
        stride *= static_cast<uint64_t>(std::max(1, spec.dims[di]));
    }
    uint64_t bytes = stride * static_cast<uint64_t>(spec.elem_size);
    if (bytes >= static_cast<uint64_t>(sizeof(void*))) return true;
    // If the FIFO has an attached memory backend that explicitly supports
    // BYREF/pointer semantics, allow pointer-sized payloads even if the
    // configured elem_size is smaller than a host pointer.
    gp_mem_backend_handle_t h = gp_table_edge_get_backend(table, edge_idx);
    if (h && gp_mem_backend_supports_byref(h)) return true;
    return false;
}

static bool is_valid_event_payload_ptr(void* p) {
    uintptr_t v = reinterpret_cast<uintptr_t>(p);
    // Filter obvious invalids (null, all-ones) before dereferencing.
    if (v == 0 || v == static_cast<uintptr_t>(~0ULL)) return false;
    // Basic alignment check: payload contains pointers/ints so require pointer alignment.
    if ((v & (alignof(void*) - 1)) != 0) return false;
    return true;
}

static inline int32_t clamp_size_to_int32(size_t value) {
    constexpr size_t kMaxInt32 = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    return static_cast<int32_t>(std::min(value, kMaxInt32));
}

// Helper: compute canonical table key for a module/contact, preferring
// the stable per-port UUID when the contact maps to a module frame port.
static uint64_t canvas_contact_key(int module_idx, int contact_idx) {
    // Fallback canonical composition
    uint64_t composed = ((uint64_t)static_cast<uint32_t>(module_idx) << 32) | ((uint64_t)static_cast<uint32_t>(contact_idx) << 16) | 0ull;
    GP_CanvasContext* cvs = gp_canvas_get_singleton();
    if (!cvs) return composed;
    // Only consider module frame contacts
    const int frame_base = kModuleFrameContactBase;
    const int frame_end = frame_base + kModuleExtraLedCount * kModuleExtraLedRows;
    if (contact_idx < frame_base || contact_idx >= frame_end) return 0ull;
    // Resolve the stable tool-row id only. Do NOT fall back to canonical
    // composed keys — the system now relies exclusively on the generated ids.
    uint64_t trid = gp_canvas_get_tool_row_id(cvs, module_idx, contact_idx);
    return trid;
}

static float raw_stack_value_as_float(const uint8_t* bytes, size_t size) {
    if (!bytes || size == 0) return 0.0f;
    if (size == sizeof(float)) {
        float v = 0.0f;
        std::memcpy(&v, bytes, sizeof(float));
        return v;
    }
    if (size == sizeof(double)) {
        double d = 0.0;
        std::memcpy(&d, bytes, sizeof(double));
        return static_cast<float>(d);
    }
    return 0.0f;
}

static void raw_stack_to_float_values(const RawStackFrame& frame, ValueTypeId default_type, std::vector<float>& out) {
    out.clear();
    if (!frame.types_per_byte || frame.byte_count == 0) return;
    // Read frame bytes into a host buffer via backend copy/map so we can
    // safely interpret typed elements. If reading fails, treat bytes as zeros.
    std::vector<uint8_t> tmp;
    try { tmp.resize(frame.byte_count); } catch (...) { return; }
    bool read_ok = false;
    if (frame.backend) {
        const gp_mem_backend_vtable_t* bvt = gp_mem_backend_get_vtable(frame.backend);
        if (bvt && bvt->copy_from_backend) {
            if (bvt->copy_from_backend(frame.backend, 0, tmp.data(), frame.byte_count)) read_ok = true;
        }
        if (!read_ok) {
            void* m = gp_mem_backend_map_or_null(frame.backend);
            if (m) {
                std::memcpy(tmp.data(), m, frame.byte_count);
                gp_mem_backend_unmap(frame.backend);
                read_ok = true;
            }
        }
    }
    if (!read_ok) std::memset(tmp.data(), 0, tmp.size());

    size_t idx = 0;
    while (idx < frame.byte_count) {
        ValueTypeId tid = frame.types_per_byte[idx];
        if (tid == kInvalidValueTypeId) break;
        const ValueType* vt = ValueTypeRegistry::global().get(tid);
        if (!vt || vt->size == 0 || idx + vt->size > frame.byte_count) break;
        if ((vt->size == sizeof(double) && tid == default_type) || vt->size == sizeof(float)) {
            float value = raw_stack_value_as_float(tmp.data() + idx, vt->size);
            out.push_back(value);
        } else {
            out.push_back(0.0f);
        }
        idx += vt->size;
    }
}

} // namespace


ThreadManager::ThreadManager()
    : module_tool_stack_default_type_(ValueTypeRegistry::global().builtin(VT_FLOAT64)) {}

ThreadManager::~ThreadManager() {
    stop();
}

void ThreadManager::set_module_tool_stack_default_type(ValueTypeId tid) {
    module_tool_stack_default_type_.store(tid, std::memory_order_relaxed);
}

ValueTypeId ThreadManager::module_tool_stack_default_type() const {
    return module_tool_stack_default_type_.load(std::memory_order_relaxed);
}

RawStackFrame* ThreadManager::ensure_module_tool_stack(int module_idx) {
    if (module_idx < 0) return nullptr;
    std::lock_guard<std::mutex> lk(module_tool_stacks_mu_);
    if (static_cast<size_t>(module_idx) >= module_tool_stacks_.size()) {
        module_tool_stacks_.resize(static_cast<size_t>(module_idx) + 1);
    }
    auto& ptr = module_tool_stacks_[static_cast<size_t>(module_idx)];
    if (!ptr) {
        ptr.reset(new RawStackFrame());
        if (!raw_stack_init_frame(*ptr, module_tool_stack_capacity_bytes_, module_tool_stack_default_type_.load(std::memory_order_relaxed))) {
            ptr.reset();
            return nullptr;
        }
    }
    return ptr.get();
}

void ThreadManager::reset_module_tool_stack(RawStackFrame& frame) {
    raw_stack_reset(frame);
}

void ThreadManager::set_module_sleep_delay(int module_idx, uint64_t delay_ticks, double delay_seconds) {
    if (module_idx < 0) return;
    uint64_t until_tick = 0;
    uint64_t until_time_us = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        // next_tick_id_ is the id that will be assigned to the next submitted tick.
        // Use that as a baseline for relative tick delays.
        until_tick = next_tick_id_ + delay_ticks;
        if (delay_seconds > 0.0) {
            using namespace std::chrono;
            until_time_us = static_cast<uint64_t>(duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count() + static_cast<int64_t>(std::lround(delay_seconds * 1e6)));
        }
        if (static_cast<size_t>(module_idx) >= module_ledger_.size()) module_ledger_.resize(static_cast<size_t>(module_idx) + 1);
        auto &ledger = module_ledger_[static_cast<size_t>(module_idx)];
        ledger.sleep_until_tick.store(until_tick, std::memory_order_relaxed);
        ledger.sleep_until_time_us.store(until_time_us, std::memory_order_relaxed);
    }
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
    return clamp_size_to_int32(nit->second);
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
    return edge_reader_registry_.register_reader(edge_id);
}

void ThreadManager::unregister_reader_slot(int slot) {
    edge_reader_registry_.unregister_reader(slot);
}

uint64_t ThreadManager::min_reader_seq_for_edge(uint64_t edge_id) const {
    return edge_reader_registry_.min_reader_seq(edge_id);
}

void ThreadManager::update_reader_seq(int slot, uint64_t seq) {
    (void)edge_reader_registry_.update_reader(slot, seq);
}

// Global instance
static ThreadManager* g_thread_manager_instance = nullptr;
void ThreadManager::set_global(ThreadManager* mgr) { g_thread_manager_instance = mgr; }
ThreadManager* ThreadManager::global() { return g_thread_manager_instance; }

bool ThreadManager::get_module_timing(int module_idx, ModuleTiming* out) const {
    if (!out) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (module_idx < 0 || static_cast<size_t>(module_idx) >= module_ledger_.size()) return false;
    const auto &ledger = module_ledger_[static_cast<size_t>(module_idx)];
    out->run_count = ledger.run_count.load(std::memory_order_relaxed);
    // convert microseconds back to seconds
    uint64_t last_us = ledger.last_run_wall_time_us.load(std::memory_order_relaxed);
    uint64_t total_us = ledger.total_run_wall_time_us.load(std::memory_order_relaxed);
    out->last_run_wall_time = static_cast<double>(last_us) / 1e6;
    out->total_run_wall_time = static_cast<double>(total_us) / 1e6;
    return true;
}

void ThreadManager::run_scheduled_tick(const TickRequest& req) {
        // Apply any queued UI ops on each module's table before building the successor
        // adjacency. This ensures UI-side edits are applied on the manager thread
        // and will be visible to scheduling logic in this tick.
        for (const auto& m : req.modules) {
            if (m.table) gp_table_apply_pending_ops(reinterpret_cast<GP_TableContext*>(m.table));
        }
        if (req.root_table) {
            gp_table_apply_pending_ops(req.root_table);
        }

        // Build successor adjacency from edges (writer -> reader modules) using
    // module vector indices as canonical ids.
    int N = static_cast<int>(req.modules.size());
    std::vector<std::vector<int>> succ(N);
    // Note: GUI-originated events for bound ports are published into root
    // table FIFOs (pointer-mode). They are consumed during the normal per-row
    // FIFO consumption below; the legacy canvas pop-queue path was removed.
    for (const auto& e : req.edges) {
        if (e.a_module >= 0 && e.a_module < N && e.b_module >= 0 && e.b_module < N) {
            succ[e.a_module].push_back(e.b_module);
        }
    }

        // Special-case: consume root-table FIFO samples targeted at the canvas
        // synthetic root-reflection module and use them to update the canonical
        // subgroup toolbar RGBA values. We expect producers to publish 4-float
        // RGBA samples to the root-edge consuming the toolbar contact index.
        if (req.root_table) {
            GP_CanvasContext* canvas_single = gp_canvas_get_singleton();
            int root_mod = gp_canvas_get_root_module_idx();
            if (canvas_single && root_mod >= 0) {
                for (const auto &e : req.edges) {
                    // consume samples for edges where either endpoint is the synthetic root module
                    int edge_idx = e.edge_idx;
                    int contact_idx = -1;
                    if (e.b_module == root_mod) contact_idx = e.b_contact_idx;
                    else if (e.a_module == root_mod) contact_idx = e.a_contact_idx;
                    else continue;
                    uint64_t reader_key = canvas_contact_key(root_mod, contact_idx);
                    if (reader_key == 0) reader_key = 0x8000000000000000ull;
                    gp_table_edge_subscribe_ex(req.root_table, edge_idx, reader_key, /*start_at_head=*/1);
                    int32_t unread = 0;
                    gp_table_edge_unread(req.root_table, edge_idx, reader_key, &unread);
                            if (unread > 0) {
                                // Determine edge stride (cache if available).
                        int stride = 0;
                        auto itc = edge_stride_cache_.find(edge_idx);
                        if (itc != edge_stride_cache_.end()) stride = itc->second;
                        else {
                                GP_TableEdgeTensorSpecTyped spec{};
                                if (gp_table_edge_get_tensor_spec(req.root_table, edge_idx, &spec)) {
                                    stride = 1;
                                    for (int di = 0; di < spec.dim_count; ++di) stride *= std::max(1, spec.dims[di]);
                                } else stride = 1;
                            edge_stride_cache_[edge_idx] = stride;
                        }

                        int subgroup_idx = contact_idx;
                        const int toolbar_base = kModuleFrameContactBase + kModuleExtraLedCount * kModuleExtraLedRows;
                        if (contact_idx >= toolbar_base && contact_idx < toolbar_base + kModuleExtraLedCount) {
                            subgroup_idx = contact_idx - toolbar_base;
                        }

                        auto almost_equal = [](float a, float b) {
                            return std::fabs(a - b) <= 1e-6f;
                        };

                        if (stride == 4) {
                            float sample[4] = {0.0f,0.0f,0.0f,0.0f};
                            int32_t written = 0;
                            if (gp_table_edge_peek(req.root_table, edge_idx, reader_key, sample, static_cast<int32_t>(4 * sizeof(float)), &written) && written >= static_cast<int32_t>(4 * sizeof(float))) {
                                float rgba[4];
                                int32_t written2 = 0;
                                if (gp_table_edge_consume(req.root_table, edge_idx, reader_key, rgba, static_cast<int32_t>(4 * sizeof(float)), &written2) && written2 >= static_cast<int32_t>(4 * sizeof(float))) {
                                    bool changed = false;
                                    auto it = last_applied_rgba_subgroup_.find(subgroup_idx);
                                    if (it == last_applied_rgba_subgroup_.end()) changed = true;
                                    else {
                                        for (int i = 0; i < 4; ++i) if (!almost_equal(it->second[i], rgba[i])) { changed = true; break; }
                                    }
                                    if (changed) {
                                        printf("ThreadManager: consumed full sample for edge %d contact=%d mapped_subgroup=%d\n", edge_idx, contact_idx, subgroup_idx);
                                        gp_canvas_set_subgroup_toolbar_rgba_at(canvas_single, subgroup_idx, rgba);
                                        last_applied_rgba_subgroup_[subgroup_idx] = {rgba[0], rgba[1], rgba[2], rgba[3]};
                                    }
                                }
                            }
                        } else if (stride == 1) {
                            // Assemble RGBA from successive single-float samples.
                            while (true) {
                                // First check for pointer-mode events on stride-1 assembly pathway.
                                void* maybe_p = nullptr;
                                if (edge_stride_allows_pointer(req.root_table, edge_idx) &&
                                    gp_table_edge_consume_ptr(req.root_table, edge_idx, reader_key, &maybe_p) && is_valid_event_payload_ptr(maybe_p)) {
                                    // consumed a pointer EventPayload
                                    EventPayload* ep = reinterpret_cast<EventPayload*>(maybe_p);
                                    void* pa = ep->pending;
                                    if (canvas_single) {
                                        // clear the originating frame ptrs (both send/receive)
                                        gp_canvas_set_module_frame_ptr(canvas_single, ep->src_module, 1, ep->frame_idx, nullptr);
                                        gp_canvas_set_module_frame_ptr(canvas_single, ep->src_module, 0, ep->frame_idx, nullptr);
                                        // invoke and free the pending action
                                        gp_canvas_invoke_pending_action(canvas_single, pa);
                                        gp_canvas_free_pending_action(canvas_single, pa);
                                    }
                                    delete ep;
                                    continue; // continue consuming until FIFO empty
                                }
                                int32_t written = 0;
                                float s[1] = {0.0f};
                                if (!(gp_table_edge_consume(req.root_table, edge_idx, reader_key, s, static_cast<int32_t>(sizeof(float)), &written) && written == static_cast<int32_t>(sizeof(float)))) break;
                                auto &buf = edge_assemble_buf_[edge_idx];
                                buf.push_back(s[0]);
                                if (buf.size() >= 4) {
                                    float rgba[4] = {buf[0], buf[1], buf[2], buf[3]};
                                    bool changed = false;
                                    auto it = last_applied_rgba_subgroup_.find(subgroup_idx);
                                    if (it == last_applied_rgba_subgroup_.end()) changed = true;
                                    else {
                                        for (int i = 0; i < 4; ++i) if (!almost_equal(it->second[i], rgba[i])) { changed = true; break; }
                                    }
                                    if (changed) {
                                        printf("ThreadManager: assembled RGBA from stride-1 for edge %d contact=%d mapped_subgroup=%d\n", edge_idx, contact_idx, subgroup_idx);
                                        gp_canvas_set_subgroup_toolbar_rgba_at(canvas_single, subgroup_idx, rgba);
                                        last_applied_rgba_subgroup_[subgroup_idx] = {rgba[0], rgba[1], rgba[2], rgba[3]};
                                    }
                                    // remove consumed values
                                    if (buf.size() > 4) {
                                        std::vector<float> leftover(buf.begin() + 4, buf.end());
                                        buf.swap(leftover);
                                    } else buf.clear();
                                }
                            }
                        } else {
                            // Fallback: attempt to read stride (if >4 we ignore extras) and apply first 4 components.
                            int cap = std::max(1, std::min(4, stride));
                            std::vector<float> sample(cap, 0.0f);
                            int32_t written = 0;
                            if (gp_table_edge_peek(req.root_table, edge_idx, reader_key, sample.data(), static_cast<int32_t>(cap * sizeof(float)), &written) && written > 0) {
                                // consume only if we can read the full stride
                                std::vector<float> consumed(stride, 0.0f);
                                int32_t consumed_written = 0;
                                if (gp_table_edge_consume(req.root_table, edge_idx, reader_key, consumed.data(), static_cast<int32_t>(stride * sizeof(float)), &consumed_written) && consumed_written == static_cast<int32_t>(stride * sizeof(float))) {
                                    float rgba[4] = {0.0f,0.0f,0.0f,1.0f};
                                    for (int i = 0; i < std::min(4, stride); ++i) rgba[i] = consumed[i];
                                    bool changed = false;
                                    auto it = last_applied_rgba_subgroup_.find(subgroup_idx);
                                    if (it == last_applied_rgba_subgroup_.end()) changed = true;
                                    else {
                                        for (int i = 0; i < 4; ++i) if (!almost_equal(it->second[i], rgba[i])) { changed = true; break; }
                                    }
                                    if (changed) {
                                        printf("ThreadManager: consumed stride-%d sample for edge %d contact=%d mapped_subgroup=%d\n", stride, edge_idx, contact_idx, subgroup_idx);
                                        gp_canvas_set_subgroup_toolbar_rgba_at(canvas_single, subgroup_idx, rgba);
                                        last_applied_rgba_subgroup_[subgroup_idx] = {rgba[0], rgba[1], rgba[2], rgba[3]};
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    // Prefer topological order when acyclic, otherwise use ASAP/ALAP slack heuristic.
    bool acyclic = true;
    std::vector<int> order = topo_kahn(succ);
    if (order.empty()) {
        acyclic = false;
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
    auto execute_module = [&](int mod_idx) {
        if (mod_idx < 0 || mod_idx >= (int)req.modules.size()) return;
        const auto& mod = req.modules[static_cast<size_t>(mod_idx)];
        // Skip modules that have no table or are marked to skip entirely by canvas UI
        if (!mod.table) return;
        if (mod.module_skip != 0) return;
        // Handle per-module execution cadence (exec_skip_count): manager maintains a persistent
        // counter per module and only executes the module once every (exec_skip_count+1) frames.
        if (mod.module_idx >= 0) {
            std::lock_guard<std::mutex> lk(mu_);
            if (static_cast<size_t>(mod.module_idx) >= module_ledger_.size()) {
                module_ledger_.resize(static_cast<size_t>(mod.module_idx) + 1);
            }
        }
        uint64_t exec_skip = 0;
        if (mod.exec_skip_count > 0) exec_skip = static_cast<uint64_t>(mod.exec_skip_count);
        if (exec_skip > 0) {
            // check ledger counter and possibly skip this module execution this tick
            bool should_execute = false;
            if (mod.module_idx >= 0) {
                std::lock_guard<std::mutex> lk(mu_);
                auto &ledger = module_ledger_[static_cast<size_t>(mod.module_idx)];
                // increment counter and test
                ledger.exec_tick_counter = ledger.exec_tick_counter + 1;
                should_execute = ((ledger.exec_tick_counter % (exec_skip + 1ull)) == 0ull);
            } else {
                // module has no ledger index; fall back to executing every frame
                should_execute = true;
            }
            if (!should_execute) return;
        }
        std::vector<ModuleIORow> io_rows;
        {
            // Tool/plugin bindings are GraphRuntime-owned (not a canvas/
            // rendering concern) -- read them directly from this
            // ThreadManager's own GraphRuntime instead of a canvas
            // singleton, so scheduling never depends on a live canvas.
            const std::vector<ModuleIORow>* rows = nullptr;
            if (graph_runtime_ && mod_idx >= 0 &&
                mod_idx < static_cast<int>(graph_runtime_->module_io_rows.size())) {
                rows = &graph_runtime_->module_io_rows[static_cast<size_t>(mod_idx)];
            }
            if (rows) {
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
        RawStackFrame* module_stack = (mod.module_idx >= 0) ? ensure_module_tool_stack(mod.module_idx) : nullptr;
        if (!module_stack) return;
        ValueTypeId stack_default_type = module_tool_stack_default_type();
        const ValueType* stack_type_info = ValueTypeRegistry::global().get(stack_default_type);
        size_t stack_elem_size = (stack_type_info && stack_type_info->size > 0) ? stack_type_info->size : sizeof(double);
        reset_module_tool_stack(*module_stack);
        std::vector<float> stack_snapshot;
        auto push_value = [&](float value) {
            double dv = static_cast<double>(value);
            raw_stack_push_typed(*module_stack, &dv, stack_default_type);
        };
        auto pop_value = [&]() -> float {
            if (!module_stack || module_stack->byte_count == 0) {
#if defined(NODUS_MODULE_ACTIVITY_DEBUG) && (NODUS_MODULE_ACTIVITY_DEBUG != 0)
                fprintf(stderr, "[DEBUG] pop_value mod=%d stack_empty\n", mod_idx);
#endif
                return 0.0f;
            }
            ValueTypeId top_tid = kInvalidValueTypeId;
            if (!raw_stack_peek_type(*module_stack, top_tid) || top_tid != stack_default_type) {
                return 0.0f;
            }
            size_t before = module_stack->byte_count / (stack_elem_size ? stack_elem_size : 1);
            double dv = 0.0;
            if (!raw_stack_pop_typed(*module_stack, &dv, stack_default_type)) {
                return 0.0f;
            }
#if defined(NODUS_MODULE_ACTIVITY_DEBUG) && (NODUS_MODULE_ACTIVITY_DEBUG != 0)
            size_t after = module_stack->byte_count / (stack_elem_size ? stack_elem_size : 1);
            fprintf(stderr, "[DEBUG] pop_value mod=%d popped=%f stack_before=%zu stack_after=%zu\n", mod_idx, static_cast<float>(dv), before, after);
#endif
            return static_cast<float>(dv);
        };
        ModuleInputState input_state{};
        bool input_state_loaded = false;
        bool input_state_used = false;
        uint64_t tool_cycle = 0;
        bool has_ledger = false;
        int ledger_idx = mod.module_idx;
        if (ledger_idx >= 0) {
            std::lock_guard<std::mutex> lk(mu_);
            if (static_cast<size_t>(ledger_idx) >= module_ledger_.size()) {
                module_ledger_.resize(static_cast<size_t>(ledger_idx) + 1);
            }
            tool_cycle = module_ledger_[static_cast<size_t>(ledger_idx)].tool_cycle;
            has_ledger = true;
        }
        // If module requested a sleep/delay, honor it and skip execution until
        // either the tick threshold or the wall-time threshold has passed.
        if (has_ledger && mod.module_idx >= 0) {
            uint64_t sleep_tick = 0;
            uint64_t sleep_time_us = 0;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (static_cast<size_t>(ledger_idx) < module_ledger_.size()) {
                    const auto &ledger = module_ledger_[static_cast<size_t>(ledger_idx)];
                    sleep_tick = ledger.sleep_until_tick.load(std::memory_order_relaxed);
                    sleep_time_us = ledger.sleep_until_time_us.load(std::memory_order_relaxed);
                }
            }
            bool tick_block = (sleep_tick > req.tick_id);
            bool time_block = false;
            if (sleep_time_us > 0) {
                using namespace std::chrono;
                uint64_t now_us = static_cast<uint64_t>(duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count());
                time_block = (now_us < sleep_time_us);
            }
            if (tick_block || time_block) {
                return; // skip module while sleeping
            }
        }
        auto load_input_state = [&]() -> bool {
            if (!input_state_loaded) {
                input_state_loaded = canvas_get_module_input_state(mod_idx, &input_state);
            }
            return input_state_loaded;
        };
        GP_TableContext* fifo_table = req.root_table ? req.root_table : mod.table;
        // Optional timing measurement for this module execution
        bool _tm_measured = false;
        double _tm_last_run_secs = 0.0;
        bool _tm_enabled = timing_enabled_.load(std::memory_order_relaxed);
        std::chrono::high_resolution_clock::time_point _tm_start;
        if (_tm_enabled) {
            _tm_start = std::chrono::high_resolution_clock::now();
        }
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
                    uint64_t reader_key = canvas_contact_key(mod_idx, contact_idx);
                    if (reader_key == 0) reader_key = 0x8000000000000000ull;
                    // Find the edge in req.edges where b_module == mod_idx and b_contact_idx == contact_idx
                    for (const auto& e : req.edges) {
                        if (e.b_module == mod_idx && e.b_contact_idx == contact_idx) {
                            edge_idx = e.edge_idx;
                            break;
                        }
                    }
                    if (edge_idx >= 0 && fifo_table) {
                        int32_t unread = 0;
                        gp_table_edge_subscribe_ex(fifo_table, edge_idx, reader_key, /*start_at_head=*/1);
                        gp_table_edge_unread(fifo_table, edge_idx, reader_key, &unread);
                        if (unread > 0) {
                            // Check for pointer-mode marker on this row: reserved0>0
                            GP_TableRow tbl_row{};
                            bool tried_row = false;
                            if (fifo_table && gp_table_get_row(fifo_table, row, &tbl_row)) {
                                tried_row = true;
                            }
                            if (tried_row && tbl_row.reserved0 > 0) {
                                // Pointer mode: consume opaque EventPayload when the FIFO stride can safely hold a pointer.
                                if (edge_stride_allows_pointer(fifo_table, edge_idx)) {
                                    // Consume all available pointer-mode EventPayloads for bulk handling.
                                    while (true) {
                                        void* maybe_p = nullptr;
                                        if (!(gp_edge_consume_ptr(fifo_table, edge_idx, reader_key, &maybe_p) && is_valid_event_payload_ptr(maybe_p))) break;
                                        EventPayload* ep = reinterpret_cast<EventPayload*>(maybe_p);
                                        void* pa = ep->pending;
                                        GP_CanvasContext* canvas_single = gp_canvas_get_singleton();
                                        if (canvas_single && pa) {
                                            // If this pending action encodes a text/tensor action
                                            // deliver its payload into the module frame port and
                                            // stash it for the tool to consume and free.
                                            struct MinimalPending { int32_t action_id; };
                                            auto *ptest = reinterpret_cast<MinimalPending*>(pa);
                                            if (ptest && (ptest->action_id == CANVAS_ACT_TEXT_RENDER || ptest->action_id == CANVAS_ACT_TENSOR_ALLOC)) {
                                                // set module receive frame ptr to pending payload
                                                gp_canvas_set_module_frame_ptr(canvas_single, ep->src_module, 0, ep->frame_idx, pa);
                                                // stash under managed map so tool can pop and free
                                                gp_canvas_stash_managed_event_payload(canvas_single, ep->src_module, ep->frame_idx, pa);
                                                // do not invoke or free here; tool will handle
                                            } else {
                                                gp_canvas_set_module_frame_ptr(canvas_single, ep->src_module, 1, ep->frame_idx, nullptr);
                                                gp_canvas_set_module_frame_ptr(canvas_single, ep->src_module, 0, ep->frame_idx, nullptr);
                                                gp_canvas_invoke_pending_action(canvas_single, pa);
                                                gp_canvas_free_pending_action(canvas_single, pa);
                                            }
                                        }
                                        delete ep;
                                    }
                                }
                            } else {
                                float sample[1] = {0.0f};
                                int32_t written = 0;
                                if (gp_table_edge_consume(fifo_table, edge_idx, reader_key, sample, static_cast<int32_t>(sizeof(float)), &written) && written > 0) {
                                    val = sample[0];
                                }
                            }
                        }
                    }
                    push_value(val);
                }
            }
            // Tool row: pass stack through (no-op for now)
            if (meta.kind == ModuleRowKind::Tool) {
                // If this is a plugin-origin row, attempt to dispatch to the live plugin instance
                if (meta.tool_origin == ModuleToolOrigin::Plugin) {
                    ITool* inst = nullptr;
                    if (graph_runtime_ && mod_idx >= 0 &&
                        mod_idx < static_cast<int>(graph_runtime_->module_plugin_instances.size()) &&
                        row >= 0 &&
                        row < static_cast<int>(graph_runtime_->module_plugin_instances[static_cast<size_t>(mod_idx)].size())) {
                        inst = graph_runtime_->module_plugin_instances[static_cast<size_t>(mod_idx)][static_cast<size_t>(row)].get();
                    }
                    if (inst) {
                        ToolStackFrame frame{};
                        frame.raw = module_stack;
                        frame.default_type = stack_default_type;
                        ToolStackContext tctx;
                        tctx.stack = frame;
                        ToolInputState tinp;
                        if (load_input_state()) {
                            tinp.mouse_x = input_state.mouse_x;
                            tinp.mouse_y = input_state.mouse_y;
                            tinp.mouse_down = input_state.mouse_down;
                            tinp.mouse_up = input_state.mouse_up;
                            tinp.mouse_dx = input_state.mouse_dx;
                            tinp.mouse_dy = input_state.mouse_dy;
                            tinp.mouse_button = input_state.mouse_button;
                            tinp.mouse_scroll = input_state.mouse_scroll;
                            tinp.mouse_button_mask_down = input_state.mouse_button_mask_down;
                            tinp.mouse_button_mask_up = input_state.mouse_button_mask_up;
                            tinp.mouse_device_id = input_state.mouse_device_id;
                            tinp.key = input_state.key;
                            tinp.key_event = input_state.key_event;
                            tctx.input = &tinp;
                        } else {
                            tctx.input = nullptr;
                        }
                        try {
                            inst->execute_stack(tctx);
                        } catch (...) {}
                        continue; // plugin handled this row
                    }
                }
                switch (meta.tool) {
                    case ModuleToolKind::KeyboardListener: {
                        float val = 0.0f;
                        if (load_input_state() && input_state.key_event) {
                            val = static_cast<float>(input_state.key);
                        }
                        push_value(val);
                        if (input_state_loaded) input_state_used = true;
                        break;
                    }
                    case ModuleToolKind::MouseListener: {
                        ML_DEBUGF("[DBG] MouseListener: executing for mod=%d contact=%d\n", mod_idx, meta.contact_idx);
                        float mx = 0.0f;
                        float my = 0.0f;
                        float mdx = 0.0f;
                        float mdy = 0.0f;
                        float scroll = 0.0f;
                        uint32_t button_mask_down = 0u;
                        uint32_t button_mask_up = 0u;
                        int32_t device_id = 0;
                        // Prefer FIFO-backed events published into root table FIFOs
                        // for module-frame bindings. If none available, fall back
                        // to the GUI-captured input_state as a last resort.
                        bool handled_fifo = false;
                        GP_TableContext* fifo_table = req.root_table ? req.root_table : mod.table;
                        if (fifo_table) {
                            // Only consider module-frame contacts that actually have
                            // an edge bound to this module (respect the binding)
                            GP_CanvasContext* canvas_single = gp_canvas_get_singleton();
                            for (const auto &e : req.edges) {
                                if (e.b_module != mod_idx) continue;
                                int contact_idx = e.b_contact_idx;
                                int edge_idx = e.edge_idx;
                                // Check whether this contact has a bound pending action
                                void* bound_ptr = canvas_single ? gp_canvas_get_module_frame_ptr_for_contact(canvas_single, mod_idx, contact_idx) : nullptr;
                                if (!bound_ptr) {
                                    ML_DEBUGF("[DBG] MouseListener: no bound_ptr for mod=%d contact=%d\n", mod_idx, contact_idx);
                                    continue;
                                }
                                struct MinimalPending { int32_t action_id; };
                                auto *bound_min = reinterpret_cast<MinimalPending*>(bound_ptr);
                                int32_t bound_action = bound_min ? bound_min->action_id : 0;
                                // Only process contacts bound to mouse actions
                                if (!(bound_action >= CANVAS_ACT_MOUSE_DOWN && bound_action <= CANVAS_ACT_MOUSE_SCROLL_DOWN)) continue;

                                uint64_t reader_key = canvas_contact_key(mod_idx, contact_idx);
                                if (reader_key == 0) reader_key = 0x8000000000000000ull;
                                // Debug: report computed reader_key and target edge so we can
                                // compare it with the writer_key printed at publish time.
                                ML_DEBUGF("[DBG] sub edge=%d -> mod=%d contact=%d reader_key=%llu (0x%llx)\n", edge_idx, mod_idx, contact_idx, (unsigned long long)reader_key, (unsigned long long)reader_key);
                                int32_t unread = 0;
                                gp_table_edge_subscribe_ex(fifo_table, edge_idx, reader_key, /*start_at_head=*/1);
                                gp_table_edge_unread(fifo_table, edge_idx, reader_key, &unread);
                                ML_DEBUGF("[DBG] unread edge=%d reader_key=%llu -> %d\n", edge_idx, (unsigned long long)reader_key, unread);
                                if (unread <= 0) continue;
                                if (!edge_stride_allows_pointer(fifo_table, edge_idx)) continue;

                                // Consume all pointer-mode EventPayloads for this bound contact
                                bool any = false;
                                while (true) {
                                    void* maybe_p = nullptr;
                                    if (!(gp_edge_consume_ptr(fifo_table, edge_idx, reader_key, &maybe_p) && is_valid_event_payload_ptr(maybe_p))) break;
                                    any = true;
                                    EventPayload* ep = reinterpret_cast<EventPayload*>(maybe_p);
                                    if (!ep) { delete ep; continue; }
                                    void* pa_void = ep->pending;
                                    struct LocalPendingAction { int32_t action_id; GP_TableHitBox hit; float dx; float dy; int32_t button; float scroll; uint32_t button_mask_down; uint32_t button_mask_up; int32_t device_id; };
                                    auto *pa = reinterpret_cast<LocalPendingAction*>(pa_void);
                                    if (!pa) {
                                        // nothing to do
                                        if (ep) delete ep;
                                        continue;
                                    }
                                    // Only deliver payloads that match the contact's bound action
                                    if (pa->action_id != bound_action) {
                                        // Not our action: log and stash it back as managed payload
                                        ML_DEBUGF("[DBG] stash not-matching action=%d bound=%d mod=%d contact=%d edge=%d src_mod=%d frame=%d\n", pa->action_id, bound_action, mod_idx, contact_idx, edge_idx, ep->src_module, ep->frame_idx);
                                        GP_CanvasContext* c_single = gp_canvas_get_singleton();
                                        if (c_single) {
                                            gp_canvas_set_module_frame_ptr(c_single, ep->src_module, 0, ep->frame_idx, pa_void);
                                            gp_canvas_stash_managed_event_payload(c_single, ep->src_module, ep->frame_idx, pa_void);
                                        }
                                        if (ep) delete ep;
                                        continue;
                                    }

                                    // extract payload fields into locals
                                    if (pa->action_id == CANVAS_ACT_MOUSE_SCROLL_UP || pa->action_id == CANVAS_ACT_MOUSE_SCROLL_DOWN) scroll = pa->scroll;
                                    mx = static_cast<float>(pa->hit.x0);
                                    my = static_cast<float>(pa->hit.y0);
                                    mdx = pa->dx;
                                    mdy = pa->dy;
                                    if (pa->button_mask_down) button_mask_down = pa->button_mask_down;
                                    else if (pa->button > 0) button_mask_down = (1u << static_cast<uint32_t>(pa->button));
                                    if (pa->button_mask_up) button_mask_up = pa->button_mask_up;
                                    else if (pa->button > 0) button_mask_up = (1u << static_cast<uint32_t>(pa->button));
                                    device_id = pa->device_id;

                                    // deliver this payload to the module's MouseListener stack
                                    ML_DEBUGF("[DBG] deliver action=%d -> mod=%d contact=%d edge=%d src_mod=%d frame=%d mx=%f my=%f mdx=%f mdy=%f scroll=%f dev=%d down=0x%X up=0x%X\n", pa->action_id, mod_idx, contact_idx, edge_idx, ep->src_module, ep->frame_idx, mx, my, mdx, mdy, scroll, device_id, button_mask_down, button_mask_up);

                                    // Push typed values into the module's raw byte stack
                                    // (the single canonical stack). We require a RawStackFrame
                                    // to be present for this contact; otherwise the event is
                                    // dropped. Use the registry builtin id for float32.
                                    GP_CanvasContext* canvas_for_push = gp_canvas_get_singleton();
                                    RawStackFrame* rs_for_push = nullptr;
                                    if (canvas_for_push) {
                                        void* maybe_raw = gp_canvas_get_module_frame_ptr_for_contact(canvas_for_push, mod_idx, contact_idx);
                                        rs_for_push = reinterpret_cast<RawStackFrame*>(maybe_raw);
                                    }
                                    if (!rs_for_push) {
                                        ML_DEBUGF("[DBG] no raw stack frame for mod=%d contact=%d -> dropping event\n", mod_idx, contact_idx);
                                    } else {
                                        ValueTypeId vt_float = ValueTypeRegistry::global().builtin(VT_FLOAT32);
                                        ValueTypeId vt_uint32 = ValueTypeRegistry::global().builtin(VT_UINT32);
                                        ValueTypeId vt_int32 = ValueTypeRegistry::global().builtin(VT_INT32);

                                        uint32_t up_v = button_mask_up;
                                        uint32_t down_v = button_mask_down;
                                        int32_t dev_v = device_id;

                                        bool ok = raw_stack_push_typed(*rs_for_push, &up_v, vt_uint32)
                                               && raw_stack_push_typed(*rs_for_push, &down_v, vt_uint32)
                                               && raw_stack_push_typed(*rs_for_push, &dev_v, vt_int32)
                                               && raw_stack_push_typed(*rs_for_push, &scroll, vt_float)
                                               && raw_stack_push_typed(*rs_for_push, &my, vt_float)
                                               && raw_stack_push_typed(*rs_for_push, &mx, vt_float)
                                               && raw_stack_push_typed(*rs_for_push, &mdx, vt_float)
                                               && raw_stack_push_typed(*rs_for_push, &mdy, vt_float);
                                        if (!ok) {
                                            ML_DEBUGF("[DBG] raw_stack_push_typed failed for mod=%d contact=%d\n", mod_idx, contact_idx);
                                        }
                                    }

                                    if (canvas_single) {
                                        gp_canvas_set_module_frame_ptr(canvas_single, ep->src_module, 1, ep->frame_idx, nullptr);
                                        gp_canvas_set_module_frame_ptr(canvas_single, ep->src_module, 0, ep->frame_idx, nullptr);
                                        gp_canvas_free_pending_action(canvas_single, pa_void);
                                    }

                                    if (ep) delete ep;
                                }
                                if (any) {
                                    handled_fifo = true;
                                    input_state_used = true;
                                    break;
                                }
                            }
                        }
                        // Nothing further: per-payload pushes delivered inside the FIFO loop.
                        break;
                    }
                    case ModuleToolKind::StackDisplay: {
                        // Build float snapshot (for legacy preview) and metadata (type id + element count per item)
                        raw_stack_to_float_values(*module_stack, stack_default_type, stack_snapshot);
                        const float* snap_ptr = stack_snapshot.empty() ? nullptr : stack_snapshot.data();
                        canvas_set_module_stack_snapshot(mod_idx, row, snap_ptr, static_cast<int>(stack_snapshot.size()));

                        // Build metadata by scanning the per-byte type mask
                        std::vector<int> types;
                        std::vector<int> counts;
                        if (module_stack && module_stack->types_per_byte) {
                            size_t idx = 0;
                            while (idx < module_stack->byte_count) {
                                int tid = static_cast<int>(module_stack->types_per_byte[idx]);
                                // Lookup element size via registry
                                const ValueType* vt = ValueTypeRegistry::global().get(module_stack->types_per_byte[idx]);
                                if (!vt || vt->size == 0) break;
                                int elem_size = static_cast<int>(vt->size);
                                // Append type id and element count (in bytes/units: 1 element)
                                types.push_back(tid);
                                counts.push_back(elem_size);
                                idx += static_cast<size_t>(elem_size);
                            }
                        }
                        if (!types.empty()) {
                            canvas_set_module_stack_snapshot_meta(mod_idx, row, types.data(), counts.data(), static_cast<int>(types.size()));
                        }
                        break;
                    }
                    case ModuleToolKind::TableNumber: {
                        float val = static_cast<float>(std::max(0, meta.attachment_count));
                        push_value(val);
                        break;
                    }
                    case ModuleToolKind::Clone: {
                        float count_f = pop_value();
                        float value = pop_value();
                        int count = std::max(0, static_cast<int>(std::lround(count_f)));
                        for (int i = 0; i < count; ++i) {
                        push_value(value);
                        }
                        break;
                    }
                    case ModuleToolKind::TensorAllocator: {
                        // Built-in tensor allocator: operate on a RawStackFrame
                        // stored in the module frame pointer for this contact.
                        GP_CanvasContext* canvas_single = gp_canvas_get_singleton();
                        if (!canvas_single) break;
                        void* maybe_raw = gp_canvas_get_module_frame_ptr_for_contact(canvas_single, mod_idx, meta.contact_idx);
                        if (!maybe_raw) break;
                        RawStackFrame* rs = reinterpret_cast<RawStackFrame*>(maybe_raw);
                        if (!rs) break;

                        nodus::tensors::TensorBackend* backend = nodus::tensors::default_backend();
                        if (!backend) break;

                        // Collect old handles and shapes
                        std::vector<nodus::tensors::AbstractTensorHandle> old_handles;
                        std::vector<std::vector<uint32_t>> shapes;

                        ValueTypeId top_tid = kInvalidValueTypeId;
                        while (true) {
                            if (!raw_stack_peek_type(*rs, top_tid)) break;
                            if (top_tid != ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR)) break;

                            // Pop the tensor handle
                            nodus::tensors::AbstractTensorHandle h{};
                            if (!raw_stack_pop_abstract_tensor(*rs, h)) break;
                            old_handles.push_back(h);

                            // Pop consecutive int64 dims below the handle (top-first order)
                            std::vector<int64_t> dims_raw;
                            raw_stack_pop_int64s_while(*rs, dims_raw);
                            std::reverse(dims_raw.begin(), dims_raw.end()); // restore original push order
                            std::vector<uint32_t> dims;
                            dims.reserve(dims_raw.size());
                            for (int64_t v : dims_raw) {
                                int64_t clamped = std::max<int64_t>(1, v);
                                dims.push_back(static_cast<uint32_t>(clamped));
                            }
                            shapes.push_back(std::move(dims));
                        }

                        if (old_handles.empty()) break;

                        // Destroy old tensors
                        for (const auto& h : old_handles) {
                            if (nodus::tensors::abstract_tensor_handle_is_valid(h)) {
                                backend->destroy(h);
                            }
                        }

                        // Allocate new tensors for each recorded shape (same logical order)
                        std::vector<nodus::tensors::AbstractTensorHandle> new_handles;
                        new_handles.reserve(shapes.size());
                        bool alloc_ok = true;
                        for (const auto& dims : shapes) {
                            nodus::tensors::TensorDesc desc{};
                            desc.dtype = nodus::tensors::TensorDType::F32;
                            desc.layout = nodus::tensors::TensorLayout::Dense;
                            if (dims.empty()) {
                                desc.shape.dims = {1};
                            } else {
                                desc.shape.dims = dims;
                            }
                            nodus::tensors::AbstractTensorHandle h = backend->create(desc);
                            if (!nodus::tensors::abstract_tensor_handle_is_valid(h)) {
                                alloc_ok = false;
                                break;
                            }
                            new_handles.push_back(h);
                        }
                        if (!alloc_ok) {
                            for (const auto& h : new_handles) {
                                if (nodus::tensors::abstract_tensor_handle_is_valid(h)) {
                                    backend->destroy(h);
                                }
                            }
                            break;
                        }

                        // Push new handles back onto the raw stack in reverse so top ordering preserved
                        for (auto it = new_handles.rbegin(); it != new_handles.rend(); ++it) {
                            if (!raw_stack_push_abstract_tensor(*rs, *it)) {
                                // push failed: cleanup and abort
                                for (const auto& h : new_handles) {
                                    if (nodus::tensors::abstract_tensor_handle_is_valid(h)) {
                                        backend->destroy(h);
                                    }
                                }
                                alloc_ok = false;
                                break;
                            }
                        }
                        if (!alloc_ok) break;

                        // Clear and free any managed event payload that delivered this frame
                        {
                            int led_idx = (meta.contact_idx - kModuleFrameContactBase) % kModuleExtraLedCount;
                            GP_CanvasContext* canvas_single2 = gp_canvas_get_singleton();
                            if (canvas_single2) {
                                gp_canvas_set_module_frame_ptr(canvas_single2, mod_idx, 1, led_idx, nullptr);
                                gp_canvas_set_module_frame_ptr(canvas_single2, mod_idx, 0, led_idx, nullptr);
                                void* mp = gp_canvas_pop_managed_event_payload(canvas_single2, mod_idx, led_idx);
                                if (mp) {
                                    // payload for tensor allocator is expected to be a RawStackFrame*
                                    raw_stack_destroy_frame(reinterpret_cast<RawStackFrame*>(mp));
                                }
                            }
                        }
                        break;
                    }
                    case ModuleToolKind::TensorTool: {
                        GP_TableContext* table = req.root_table ? req.root_table : mod.table;
                        gp_table_tensor_tool_tick(get_tensor_tool_state(), table, mod_idx, row);
                        break;
                    }
                    case ModuleToolKind::RectRgba: {
                        float height = pop_value();
                        float width = pop_value();
                        float bg[4] = { pop_value(), pop_value(), pop_value(), pop_value() };
                        float border[4] = { pop_value(), pop_value(), pop_value(), pop_value() };
                        float border_width = pop_value();
                        float corner_radius = pop_value();
                        int raster_h = std::max(0, static_cast<int>(std::lround(height)));
                        int raster_w = std::max(0, static_cast<int>(std::lround(width)));
                        float out = 0.0f;
                        uint64_t cycle = tool_cycle++;
                        if (raster_w > 0 && raster_h > 0) {
                            uint64_t pixel_count = static_cast<uint64_t>(raster_w) * static_cast<uint64_t>(raster_h);
                            uint64_t total = pixel_count * 4ull;
                            if (total > 0) {
                                uint64_t idx = cycle % total;
                                int channel = static_cast<int>(idx % 4ull);
                                uint64_t pix = idx / 4ull;
                                int x = static_cast<int>(pix % static_cast<uint64_t>(raster_w));
                                int y = static_cast<int>(pix / static_cast<uint64_t>(raster_w));
                                float px = static_cast<float>(x) + 0.5f;
                                float py = static_cast<float>(y) + 0.5f;
                                float fw = static_cast<float>(raster_w);
                                float fh = static_cast<float>(raster_h);
                                float bw = std::max(0.0f, border_width);
                                float cr = std::max(0.0f, corner_radius);
                                bool inside = point_in_rounded_rect(px, py, fw, fh, cr);
                                if (inside) {
                                    bool use_bg = true;
                                    float inner_w = fw - 2.0f * bw;
                                    float inner_h = fh - 2.0f * bw;
                                    if (bw > 0.0f && inner_w > 0.0f && inner_h > 0.0f) {
                                        float inner_r = std::max(0.0f, cr - bw);
                                        use_bg = point_in_rounded_rect(px - bw, py - bw, inner_w, inner_h, inner_r);
                                    }
                                    const float* src = use_bg ? bg : border;
                                    out = src[channel];
                                }
                            }
                        }
                        push_value(out);
                        break;
                    }
                    case ModuleToolKind::Add:
                    case ModuleToolKind::Subtract:
                    case ModuleToolKind::Multiply:
                    case ModuleToolKind::Divide:
                    case ModuleToolKind::Modulo:
                    case ModuleToolKind::None:
                    default: {
                        ToolStackSpec spec = module_tool_stack_spec(meta.tool);
                        float b = (spec.consumes >= 1) ? pop_value() : 0.0f;
                        float a = (spec.consumes >= 2) ? pop_value() : 0.0f;
                        switch (meta.tool) {
                            case ModuleToolKind::Add:
                                push_value(a + b);
                                break;
                            case ModuleToolKind::Subtract:
                                push_value(a - b);
                                break;
                            case ModuleToolKind::Multiply:
                                push_value(a * b);
                                break;
                            case ModuleToolKind::Divide:
                                push_value((b == 0.0f) ? 0.0f : (a / b));
                                break;
                            case ModuleToolKind::Modulo:
                                push_value((b == 0.0f) ? 0.0f : std::fmod(a, b));
                                break;
                            case ModuleToolKind::None:
                            default:
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
                    uint64_t writer_key = canvas_contact_key(mod_idx, contact_idx);
                    // Find the edge in req.edges where a_module == mod_idx and a_contact_idx == contact_idx
                    for (const auto& e : req.edges) {
                        if (e.a_module == mod_idx && e.a_contact_idx == contact_idx) {
                            edge_idx = e.edge_idx;
                            break;
                        }
                    }
                    if (edge_idx >= 0 && fifo_table) {
                        int dropped = 0;
                        // If this output contact maps to a module-frame contact, publish the bound pointer instead of a float.
                        const int frame_base = kModuleFrameContactBase;
                        const int frame_end = frame_base + kModuleExtraLedCount * kModuleExtraLedRows;
                        if (contact_idx >= frame_base && contact_idx < frame_end) {
                            GP_CanvasContext* canvas_single = gp_canvas_get_singleton();
                            void* p = nullptr;
                            if (canvas_single) p = gp_canvas_get_module_frame_ptr_for_contact(canvas_single, mod_idx, contact_idx);
                            if (p && edge_stride_allows_pointer(fifo_table, edge_idx) && is_valid_event_payload_ptr(p)) {
                                // If pointer references a RawStackFrame and the FIFO has
                                // an attached backend, attempt an optimized in-backend
                                // transfer from the frame into the FIFO storage. Fall
                                // back to the existing EventPayload pointer publishing
                                // semantics if transfer not possible.
                                RawStackFrame* rs = reinterpret_cast<RawStackFrame*>(p);
                                gp_mem_backend_handle_t fifo_b = gp_table_edge_get_backend(fifo_table, edge_idx);
                                bool transferred = false;
                                if (rs && rs->backend && fifo_b) {
                                    // Determine top type id and try to publish from frame
                                    ValueTypeId top_tid = kInvalidValueTypeId;
                                    if (raw_stack_peek_type(*rs, top_tid)) {
                                        int dropped_local = 0;
                                        if (gp_table_edge_publish_from_frame(fifo_table, edge_idx, writer_key, rs, static_cast<int32_t>(top_tid), &dropped_local)) {
                                            transferred = true;
                                            dropped = dropped_local;
                                        }
                                    }
                                }
                                if (!transferred) {
                                    int local = contact_idx - frame_base;
                                    int frame_idx = local % kModuleExtraLedCount;
                                    // Wrap into EventPayload so consumers can clear the frame slot after handling
                                    EventPayload* ep = new EventPayload{p, mod_idx, frame_idx};
                                    gp_edge_publish_ptr(fifo_table, edge_idx, writer_key, reinterpret_cast<void*>(ep), &dropped);
                                }
                            } else {
                                float payload[1] = {val};
                                gp_edge_publish(fifo_table, edge_idx, writer_key, payload, static_cast<int32_t>(sizeof(payload)), &dropped);
                            }
                        } else {
                            float payload[1] = {val};
                            gp_edge_publish(fifo_table, edge_idx, writer_key, payload, static_cast<int32_t>(sizeof(payload)), &dropped);
                        }
                    }
                }
            }
        }
        if (input_state_used) {
            canvas_clear_module_input_pulses(mod_idx);
        }
        raw_stack_to_float_values(*module_stack, stack_default_type, stack_snapshot);
        const float* tail_ptr = stack_snapshot.empty() ? nullptr : stack_snapshot.data();
        canvas_set_module_stack_tail(mod_idx, tail_ptr, static_cast<int>(stack_snapshot.size()));
        if (_tm_enabled) {
            auto _tm_end = std::chrono::high_resolution_clock::now();
            _tm_last_run_secs = std::chrono::duration<double>(_tm_end - _tm_start).count();
            _tm_measured = true;
        }
        if (mod.module_idx >= 0) {
            // We must hold the mutex to safely resize/access the ledger vector,
            // but timing value updates use atomics to avoid prolonged locking.
            std::lock_guard<std::mutex> lk(mu_);
            if (static_cast<size_t>(mod.module_idx) >= module_ledger_.size()) {
                module_ledger_.resize(static_cast<size_t>(mod.module_idx) + 1);
            }
            auto& ledger = module_ledger_[static_cast<size_t>(mod.module_idx)];
            ledger.ticks += 1;
            ledger.last_tick_id = req.tick_id;
            ledger.last_dt = req.dt;
            if (has_ledger) {
                ledger.tool_cycle = tool_cycle;
            }
            if (_tm_measured) {
                // store times as integer microseconds to allow atomic fetch_add
                uint64_t us = static_cast<uint64_t>(std::lround(_tm_last_run_secs * 1e6));
                ledger.run_count.fetch_add(1, std::memory_order_relaxed);
                ledger.last_run_wall_time_us.store(us, std::memory_order_relaxed);
                ledger.total_run_wall_time_us.fetch_add(us, std::memory_order_relaxed);
            }
        }
    };

    auto run_sequential = [&]() {
        for (int mod_idx : order) {
            execute_module(mod_idx);
        }
    };

    auto run_parallel = [&]() {
        std::vector<std::atomic<int>> pred_count(N);
        for (int i = 0; i < N; ++i) pred_count[i].store(0, std::memory_order_relaxed);
        for (int from = 0; from < N; ++from) {
            for (int to : succ[from]) {
                if (to >= 0 && to < N) pred_count[to].fetch_add(1, std::memory_order_relaxed);
            }
        }
        std::deque<int> ready;
        for (int i = 0; i < N; ++i) {
            // Slip modules are enqueued immediately regardless of predecessor
            // counts. Pooled modules join the ready set when their predecessors
            // are satisfied. Sequential modules remain manager-thread-only
            // unless forced into parallel by other policies.
            if (pred_count[i].load(std::memory_order_relaxed) == 0 || req.modules[static_cast<size_t>(i)].exec_mode == ThreadManager::ExecMode::Slip) {
                ready.push_back(i);
            }
        }
        if (ready.empty()) {
            run_sequential();
            return;
        }
        int hw = static_cast<int>(std::thread::hardware_concurrency());
        if (hw <= 0) hw = 1;
        int thread_count = std::min(hw, std::max(1, static_cast<int>(order.size())));
        std::mutex ready_mu;
        std::condition_variable ready_cv;
        std::atomic<int> remaining(static_cast<int>(order.size()));
        auto worker = [&]() {
            while (true) {
                int idx = -1;
                {
                    std::unique_lock<std::mutex> lk(ready_mu);
                    ready_cv.wait(lk, [&]() { return !ready.empty() || remaining.load(std::memory_order_relaxed) == 0; });
                    if (ready.empty()) {
                        if (remaining.load(std::memory_order_relaxed) == 0) break;
                        continue;
                    }
                    idx = ready.front();
                    ready.pop_front();
                }
                execute_module(idx);
                for (int succ_idx : succ[idx]) {
                    if (succ_idx < 0 || succ_idx >= N) continue;
                    int prev = pred_count[succ_idx].fetch_sub(1, std::memory_order_relaxed);
                    if (prev == 1) {
                        std::lock_guard<std::mutex> lk(ready_mu);
                        ready.push_back(succ_idx);
                        ready_cv.notify_one();
                    }
                }
                if (remaining.fetch_sub(1, std::memory_order_relaxed) == 1) {
                    std::lock_guard<std::mutex> lk(ready_mu);
                    ready_cv.notify_all();
                }
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(thread_count);
        for (int ti = 0; ti < thread_count; ++ti) {
            workers.emplace_back(worker);
        }
        {
            std::lock_guard<std::mutex> lk(ready_mu);
            ready_cv.notify_all();
        }
        for (auto& worker_thread : workers) {
            if (worker_thread.joinable()) worker_thread.join();
        }
    };

        // Parallel execution enabled when scheduled+acyclic or if the
        // manager is in FreeSpinning mode.
        auto cur_mode = mode_.load(std::memory_order_relaxed);
        if ((cur_mode == Mode::Scheduled && acyclic && order.size() > 1) || (cur_mode == Mode::FreeSpinning)) {
            run_parallel();
        } else {
            run_sequential();
        }
}
