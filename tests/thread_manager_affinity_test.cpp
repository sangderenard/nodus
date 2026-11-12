// CPU-only regression of the real ThreadManager, row stack, and table FIFO.
// No canvas, graphics context, shader, dispatch, draw, or device is created.
#include "graph_runtime.h"
#include "table_abi.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        // Fail from either scheduler lane without invoking a platform crash
        // reporter or waiting for other workers during negative regressions.
        std::_Exit(EXIT_FAILURE);
    }
}

class ProbeTool final : public ITool {
public:
    explicit ProbeTool(std::function<void(ToolStackContext&)> fn) : fn_(std::move(fn)) {}
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override { fn_(ctx); }
private:
    std::function<void(ToolStackContext&)> fn_;
};

using Table = std::unique_ptr<GP_TableContext, decltype(&gp_table_destroy)>;
Table table() {
    Table result(gp_table_create(nullptr), gp_table_destroy);
    expect(result != nullptr, "create native CPU table");
    gp_table_set_sim_enabled(result.get(), 0);
    return result;
}

void add_probe(GraphRuntime& graph, int module, std::function<void(ToolStackContext&)> fn) {
    ModuleIORow row{};
    row.tool_origin = ModuleToolOrigin::Plugin;
    graph.module_io_rows[module].push_back(row);
    graph.module_plugin_instances[module].emplace_back(new ProbeTool(std::move(fn)),
        [](ITool* tool) { delete tool; });
}

void add_io(GraphRuntime& graph, int module, ModuleRowKind kind, int contact) {
    ModuleIORow row{};
    row.kind = kind;
    row.contact_idx = contact;
    graph.module_io_rows[module].push_back(row);
    graph.module_plugin_instances[module].emplace_back(nullptr, [](ITool*) {});
}

uint64_t contact_key(int module, int contact) {
    return (uint64_t(module) << 32) | (uint64_t(contact) << 16);
}

ThreadManager::EdgeContract dependency(int from, int to, int edge = -1) {
    ThreadManager::EdgeContract result{};
    result.a_module = from;
    result.a_contact_idx = 1;
    result.b_module = to;
    result.b_contact_idx = 0;
    result.edge_idx = edge;
    return result;
}
}

int main() {
    using ExecMode = ThreadManager::ExecMode;
    auto fifo = table();
    std::array<Table, 4> tables{table(), table(), table(), table()};
    GraphRuntime graph;
    auto& manager = *graph.thread_mgr;
    graph.module_io_rows.resize(4);
    graph.module_plugin_instances.resize(4);
    ThreadManager::TickRequest request{};
    request.root_table = fifo.get();
    for (int i = 0; i < 4; ++i) {
        ThreadManager::ModuleContract module{};
        module.module_idx = i;
        module.table = tables[i].get();
        module.sim_enabled = 0;
        module.exec_mode = i % 2 == 0 ? ExecMode::Sequential : ExecMode::Pooled;
        module.stack_contract = ModuleStackContract::ScalarF32;
        request.modules.push_back(module);
    }

    // A one-module Scheduled tick already runs directly on the manager. Use
    // that as the independently established thread identity for mixed ticks.
    std::thread::id manager_id;
    const auto caller_id = std::this_thread::get_id();
    add_probe(graph, 0, [&](ToolStackContext&) { manager_id = std::this_thread::get_id(); });
    manager.start();
    auto warmup = request;
    warmup.modules.resize(1);
    manager.submit_tick(warmup, true);
    expect(manager_id != caller_id, "manager is not the submitting thread");
    graph.module_io_rows[0].clear();
    graph.module_plugin_instances[0].clear();

    std::array<RawStackFrame*, 4> frames{};
    std::array<std::thread::id, 4> last_lane{};
    std::array<std::atomic<int>, 4> executions{};
    for (int i = 0; i < 4; ++i) {
        if (i > 0) add_io(graph, i, ModuleRowKind::Input, 0);
        add_probe(graph, i, [&, i](ToolStackContext& ctx) {
            last_lane[i] = std::this_thread::get_id();
            expect((last_lane[i] == manager_id) == (i % 2 == 0), "module uses its declared lane");
            expect(last_lane[i] != caller_id, "rows do not run on submitter");
            expect(ctx.stack.raw != nullptr, "canonical module stack exists");
            if (frames[i]) expect(frames[i] == ctx.stack.raw, "module stack persists across ticks");
            frames[i] = ctx.stack.raw;
            if (i > 0) {
                expect(executions[i - 1].load() == executions[i].load() + 1, "predecessor completes before successor");
                expect(tool_stack_pop(ctx.stack) == float(20 + i - 1), "FIFO value crosses mixed lanes");
            }
            expect(ctx.stack.raw->byte_count == 0, "stack is reset or input consumed exactly once");
            tool_stack_push(ctx.stack, float(20 + i));
            ++executions[i];
        });
        if (i < 3) {
            add_io(graph, i, ModuleRowKind::Output, 1);
            expect(gp_table_add_edge(fifo.get(), contact_key(i, 1), contact_key(i + 1, 0)) != 0, "create native FIFO edge");
            GP_TableEdgeTensorSpecTyped spec{};
            spec.dims[0] = 1;
            spec.dim_count = 1;
            spec.slots = 4;
            spec.elem_size = sizeof(float);
            spec.type_id = ValueTypeRegistry::global().builtin(VT_FLOAT32);
            expect(gp_table_edge_set_tensor_spec(fifo.get(), i, &spec) != 0, "configure native FIFO");
            expect(gp_table_edge_subscribe_ex(fifo.get(), i, contact_key(i + 1, 0), 1) != 0, "subscribe before first output");
            request.edges.push_back(dependency(i, i + 1, i));
        }
    }

    for (auto mode : {ThreadManager::Mode::Scheduled, ThreadManager::Mode::FreeSpinning}) {
        manager.set_mode(mode);
        for (int tick = 0; tick < 16; ++tick) manager.submit_tick(request, true);
    }
    for (int i = 0; i < 4; ++i) {
        expect(executions[i] == 32, "exactly one execution per module per tick");
        for (int j = 0; j < i; ++j) expect(frames[i] != frames[j], "stacks remain module-owned");
        if (i < 3) {
            int32_t unread = -1;
            expect(gp_table_edge_unread(fifo.get(), i, contact_key(i + 1, 0), &unread) != 0 && unread == 0,
                   "FIFO consumed once with no leftover sample");
        }
    }
    std::cout << "manager=" << manager_id << " caller=" << caller_id
              << " sequential=" << last_lane[0] << ',' << last_lane[2]
              << " pooled=" << last_lane[1] << ',' << last_lane[3] << '\n';

    // An independent pooled row must make progress while a manager row is
    // executing. This would time out if the repair serialized the whole tick.
    for (auto& rows : graph.module_io_rows) rows.clear();
    for (auto& instances : graph.module_plugin_instances) instances.clear();
    request.modules.resize(2);
    request.edges.clear();
    std::mutex gate_mu;
    std::condition_variable gate_cv;
    bool sequential_entered = false, pooled_entered = false;
    add_probe(graph, 0, [&](ToolStackContext&) {
        std::unique_lock<std::mutex> lock(gate_mu);
        sequential_entered = true;
        gate_cv.notify_all();
        expect(gate_cv.wait_for(lock, std::chrono::seconds(5), [&] { return pooled_entered; }), "pooled work overlaps manager lane");
    });
    add_probe(graph, 1, [&](ToolStackContext&) {
        std::unique_lock<std::mutex> lock(gate_mu);
        expect(gate_cv.wait_for(lock, std::chrono::seconds(5), [&] { return sequential_entered; }), "manager lane progresses beside pool");
        pooled_entered = true;
        gate_cv.notify_all();
    });
    manager.submit_tick(request, true);

    // Slip remains worker-owned and bypasses its predecessor, but must not be
    // enqueued a second time when that predecessor later completes.
    request.modules[1].exec_mode = ExecMode::Slip;
    request.edges.push_back(dependency(0, 1));
    graph.module_plugin_instances[1].clear();
    graph.module_io_rows[1].clear();
    std::atomic<int> slip_runs{0};
    pooled_entered = false;
    add_probe(graph, 1, [&](ToolStackContext&) {
        expect(std::this_thread::get_id() != manager_id, "Slip stays on worker lane");
        ++slip_runs;
        std::lock_guard<std::mutex> lock(gate_mu);
        pooled_entered = true;
        gate_cv.notify_all();
    });
    manager.submit_tick(request, true);
    expect(slip_runs == 1, "Slip executes once despite later predecessor completion");

    // A pooled predecessor can finish before or after Slip; neither timing
    // may duplicate Slip or release its Sequential successor twice.
    for (auto& rows : graph.module_io_rows) rows.clear();
    for (auto& instances : graph.module_plugin_instances) instances.clear();
    auto sequential_successor = request.modules[0];
    sequential_successor.module_idx = 2;
    sequential_successor.table = tables[2].get();
    request.modules.push_back(sequential_successor);
    request.modules[0].exec_mode = ExecMode::Pooled;
    request.edges = {dependency(0, 1), dependency(1, 2)};
    std::array<std::atomic<int>, 3> slip_chain_runs{};
    for (int i = 0; i < 3; ++i) {
        add_probe(graph, i, [&, i](ToolStackContext&) {
            expect((std::this_thread::get_id() == manager_id) == (i == 2), "Pooled-Slip-Sequential lane identity");
            if (i == 2) expect(slip_chain_runs[1] == slip_chain_runs[2] + 1, "Sequential successor waits for Slip completion");
            ++slip_chain_runs[i];
        });
    }
    for (auto mode : {ThreadManager::Mode::Scheduled, ThreadManager::Mode::FreeSpinning}) {
        manager.set_mode(mode);
        manager.submit_tick(request, true);
    }
    for (const auto& runs : slip_chain_runs) expect(runs == 2, "Pooled-Slip-Sequential executes each module once per tick");

    // Keep existing refusal and exception semantics on the repaired lane:
    // later tool and output rows must not reinterpret the retained input.
    request.modules.resize(2);
    request.modules[0].exec_mode = ExecMode::Sequential;
    request.modules[1].exec_mode = ExecMode::Pooled;
    request.edges = {dependency(0, 1, 0)};
    graph.module_io_rows[1].clear();
    graph.module_plugin_instances[1].clear();
    add_probe(graph, 1, [&](ToolStackContext&) {
        expect(std::this_thread::get_id() != manager_id, "pooled successor after refusal");
    });
    for (bool throws : {false, true}) {
        graph.module_io_rows[0].clear();
        graph.module_plugin_instances[0].clear();
        add_probe(graph, 0, [&, throws](ToolStackContext& ctx) {
            expect(std::this_thread::get_id() == manager_id, "failure runs on manager lane");
            tool_stack_push(ctx.stack, 99);
            if (throws) throw std::runtime_error("expected CPU test refusal");
            ctx.succeeded = false;
        });
        std::atomic<int> later_rows{0};
        add_probe(graph, 0, [&](ToolStackContext&) { ++later_rows; });
        add_io(graph, 0, ModuleRowKind::Output, 1);
        manager.submit_tick(request, true);
        expect(later_rows == 0, "failure suppresses later plugin rows");
        int32_t unread = -1;
        expect(gp_table_edge_unread(fifo.get(), 0, contact_key(1, 0), &unread) != 0 && unread == 0,
               "failure suppresses output publication");
    }

    // All-Sequential readiness, including a cyclic fallback, stays on the
    // same lane and does not strand workers waiting for manager-only work.
    for (auto& rows : graph.module_io_rows) rows.clear();
    for (auto& instances : graph.module_plugin_instances) instances.clear();
    for (int i = 0; i < 2; ++i) {
        request.modules[i].exec_mode = ExecMode::Sequential;
        add_probe(graph, i, [&](ToolStackContext&) {
            expect(std::this_thread::get_id() == manager_id, "all-Sequential and cyclic lane identity");
        });
    }
    manager.submit_tick(request, true);
    request.edges.push_back(dependency(1, 0));
    manager.submit_tick(request, true);
    expect(manager.wait_for_submitted_ticks(), "completion barrier remains intact");
    manager.stop();

    // With one pooled module, a persistent pool has exactly one worker lane.
    // Its thread identity must survive tick boundaries and a debug-mode run.
    GraphRuntime persistent_graph;
    auto& persistent_manager = *persistent_graph.thread_mgr;
    auto persistent_table = table();
    persistent_graph.module_io_rows.resize(1);
    persistent_graph.module_plugin_instances.resize(1);
    std::thread::id persistent_id;
    int persistent_runs = 0;
    add_probe(persistent_graph, 0, [&](ToolStackContext&) {
        const auto current = std::this_thread::get_id();
        if (persistent_id == std::thread::id{}) persistent_id = current;
        expect(current == persistent_id, "persistent pooled worker survives ticks");
        ++persistent_runs;
    });
    ThreadManager::TickRequest persistent_tick{};
    persistent_tick.modules.resize(1);
    persistent_tick.modules[0].module_idx = 0;
    persistent_tick.modules[0].table = persistent_table.get();
    persistent_tick.modules[0].sim_enabled = 0;
    persistent_tick.modules[0].exec_mode = ExecMode::Pooled;
    persistent_manager.set_mode(ThreadManager::Mode::FreeSpinning);
    persistent_manager.start();
    expect(persistent_manager.persistent_workers(), "persistent workers default on");
    for (int tick = 0; tick < 32; ++tick) persistent_manager.submit_tick(persistent_tick, true);
    expect(persistent_runs == 32, "persistent worker executed every tick");
    persistent_graph.module_io_rows[0].clear();
    persistent_graph.module_plugin_instances[0].clear();
    persistent_manager.set_persistent_workers(false);
    expect(!persistent_manager.persistent_workers(), "debug toggle turns persistence off");
    persistent_manager.submit_tick(persistent_tick, true);
    persistent_manager.set_persistent_workers(true);
    add_probe(persistent_graph, 0, [&](ToolStackContext&) {
        expect(std::this_thread::get_id() == persistent_id,
               "persistent lane resumes after debug tick");
        ++persistent_runs;
    });
    persistent_manager.submit_tick(persistent_tick, true);
    expect(persistent_runs == 33, "debug and resumed ticks complete");
    persistent_manager.stop();
    std::cout << "PASS: mixed-lane FIFO, stable stacks, parallel overlap, Slip once, refusal, exception, cycle\n";
}
