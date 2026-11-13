#include "graph_runtime.h"
#include "table_abi.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using Table = std::unique_ptr<GP_TableContext, decltype(&gp_table_destroy)>;

struct Probe {
    std::atomic<uint64_t> runs{0};
    std::atomic<bool> ordered{true};
};

class Row final : public ITool {
public:
    Row(Probe& probe, int work_us, int input_count, bool produce)
        : probe_(probe), work_us_(work_us), input_count_(input_count), produce_(produce) {}
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        const uint64_t ordinal = probe_.runs.fetch_add(1, std::memory_order_relaxed) + 1;
        for (int i = 0; i < input_count_; ++i) {
            if (tool_stack_pop(ctx.stack) != static_cast<float>(ordinal))
                probe_.ordered.store(false, std::memory_order_relaxed);
        }
        if (produce_) tool_stack_push(ctx.stack, static_cast<float>(ordinal));
        if (work_us_ > 0) {
            const auto until = Clock::now() + std::chrono::microseconds(work_us_);
            while (Clock::now() < until) std::atomic_signal_fence(std::memory_order_seq_cst);
        }
    }
private:
    Probe& probe_;
    int work_us_;
    int input_count_;
    bool produce_;
};

class MathSource final : public ITool {
public:
    explicit MathSource(bool chain) : chain_(chain) {}
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        if (chain_) {
            for (int i = 0; i < 65; ++i) tool_stack_push(ctx.stack, 1.0f);
        } else {
            tool_stack_push(ctx.stack, 1.25f);
            tool_stack_push(ctx.stack, 2.5f);
        }
    }
private:
    bool chain_;
};

class MathSink final : public ITool {
public:
    MathSink(Probe& probe, float expected) : probe_(probe), expected_(expected) {}
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override {
        const float result = tool_stack_pop(ctx.stack);
        if (std::fabs(result - expected_) > 1e-5f)
            probe_.ordered.store(false, std::memory_order_relaxed);
        probe_.runs.fetch_add(1, std::memory_order_relaxed);
    }
private:
    Probe& probe_;
    float expected_;
};

static void add_row(GraphRuntime& graph, int module, ModuleRowKind kind, int contact, ITool* tool = nullptr) {
    ModuleIORow row{};
    row.kind = kind;
    row.contact_idx = contact;
    if (tool) row.tool_origin = ModuleToolOrigin::Plugin;
    graph.module_io_rows[module].push_back(row);
    graph.module_plugin_instances[module].emplace_back(tool, [](ITool* p) { delete p; });
}

static uint64_t contact_key(int module, int contact) {
    return (uint64_t(module) << 32) | (uint64_t(contact) << 16);
}

static void add_builtin(GraphRuntime& graph, int module, ModuleToolKind tool) {
    ModuleIORow row{};
    row.kind = ModuleRowKind::Tool;
    row.tool = tool;
    row.tool_origin = ModuleToolOrigin::Builtin;
    graph.module_io_rows[module].push_back(row);
    graph.module_plugin_instances[module].emplace_back(nullptr, [](ITool*) {});
}

static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error("usage: job_profile scenario mode work_us capacity timed repeat");
        const std::string scenario = argv[1];
        const std::string mode_name = argv[2];
        const int work_us = std::atoi(argv[3]);
        const int capacity = std::atoi(argv[4]);
        const bool timed = std::atoi(argv[5]) != 0;
        const int repeat = std::atoi(argv[6]);
        const bool free = mode_name == "free";
        ThreadManager::ExecMode exec_mode = ThreadManager::ExecMode::Pooled;
        if (mode_name == "sequential") exec_mode = ThreadManager::ExecMode::Sequential;
        else if (mode_name == "slip") exec_mode = ThreadManager::ExecMode::Slip;
        else if (free) exec_mode = ThreadManager::ExecMode::Free;
        else if (mode_name != "pooled") throw std::runtime_error("unknown mode");
        if (work_us < 0 || capacity < 1) throw std::runtime_error("invalid work/capacity");
        const bool math = scenario.rfind("math_", 0) == 0;
        const int producers = scenario == "fanin" ? 4 : scenario == "pipeline" || scenario == "pipeline_other" ? 1 : 0;
        const int independent = scenario == "independent8" || scenario == "math_chain8" ? 8 :
                                scenario == "math_chain4" ? 4 : scenario == "math_chain2" ? 2 :
                                scenario == "pipeline_other" || scenario == "independent1" || math ? 1 : 0;
        if (producers == 0 && independent == 0) throw std::runtime_error("unknown scenario");
        const int consumer_idx = producers ? producers : -1;
        const int module_count = producers ? producers + 1 + independent : independent;

        GraphRuntime graph;
        graph.module_io_rows.resize(module_count);
        graph.module_plugin_instances.resize(module_count);
        std::vector<std::unique_ptr<Probe>> probes;
        std::vector<Table> tables;
        ThreadManager::TickRequest request{};
        for (int i = 0; i < module_count; ++i) {
            probes.emplace_back(std::make_unique<Probe>());
            tables.emplace_back(gp_table_create(nullptr), gp_table_destroy);
            if (!tables.back()) throw std::runtime_error("table create failed");
            gp_table_set_sim_enabled(tables.back().get(), 0);
            ThreadManager::ModuleContract module{};
            module.module_idx = i;
            module.table = tables.back().get();
            module.sim_enabled = 0;
            module.exec_mode = exec_mode;
            module.stack_contract = ModuleStackContract::ScalarF32;
            request.modules.push_back(module);
        }
        Table root(nullptr, gp_table_destroy);
        if (producers) {
            root.reset(gp_table_create(nullptr));
            if (!root) throw std::runtime_error("root table create failed");
            request.root_table = root.get();
            for (int i = 0; i < producers; ++i) {
                add_row(graph, i, ModuleRowKind::Tool, 0, new Row(*probes[i], 0, 0, true));
                add_row(graph, i, ModuleRowKind::Output, 1);
                if (!gp_table_add_edge(root.get(), contact_key(i, 1), contact_key(consumer_idx, i)))
                    throw std::runtime_error("edge create failed");
                GP_TableEdgeTensorSpecTyped spec{};
                spec.dims[0] = 1;
                spec.dim_count = 1;
                spec.slots = capacity;
                spec.top_k = 0;
                spec.elem_size = sizeof(float);
                spec.type_id = ValueTypeRegistry::global().builtin(VT_FLOAT32);
                if (!gp_table_edge_set_tensor_spec(root.get(), i, &spec))
                    throw std::runtime_error("edge spec failed");
                if (!gp_table_edge_subscribe_ex(root.get(), i, contact_key(consumer_idx, i), 1))
                    throw std::runtime_error("edge subscribe failed");
                ThreadManager::EdgeContract edge{};
                edge.edge_idx = i;
                edge.a_module = i;
                edge.a_contact_idx = 1;
                edge.b_module = consumer_idx;
                edge.b_contact_idx = i;
                request.edges.push_back(edge);
            }
            for (int i = 0; i < producers; ++i)
                add_row(graph, consumer_idx, ModuleRowKind::Input, i);
            add_row(graph, consumer_idx, ModuleRowKind::Tool, 0,
                    new Row(*probes[consumer_idx], work_us, producers, false));
        }
        const int independent_start = producers ? consumer_idx + 1 : 0;
        for (int i = independent_start; i < module_count; ++i) {
            if (math) {
                const bool chain = scenario.rfind("math_chain", 0) == 0;
                add_row(graph, i, ModuleRowKind::Tool, 0, new MathSource(chain));
                ModuleToolKind operation = ModuleToolKind::Add;
                float expected = chain ? 65.0f : 3.75f;
                if (scenario == "math_multiply") { operation = ModuleToolKind::Multiply; expected = 3.125f; }
                else if (scenario == "math_divide") { operation = ModuleToolKind::Divide; expected = 0.5f; }
                else if (scenario == "math_modulo") { operation = ModuleToolKind::Modulo; expected = 1.25f; }
                const int operations = chain ? 64 : 1;
                for (int op = 0; op < operations; ++op) add_builtin(graph, i, operation);
                add_row(graph, i, ModuleRowKind::Tool, 0, new MathSink(*probes[i], expected));
            } else {
                add_row(graph, i, ModuleRowKind::Tool, 0,
                        new Row(*probes[i], producers ? 0 : work_us, 0, false));
            }
        }

        auto& manager = *graph.thread_mgr;
        manager.set_timing_enabled(timed);
        const auto start_begin = Clock::now();
        manager.start();
        const auto start_end = Clock::now();
        const auto first_begin = Clock::now();
        manager.submit_tick(request, true);
        const auto first_end = Clock::now();
        if (free) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        else for (int i = 0; i < 20; ++i) manager.submit_tick(request, true);

        std::vector<uint64_t> before(module_count), after(module_count);
        std::vector<ThreadManager::ModuleTiming> timing_before(module_count), timing_after(module_count);
        for (int i = 0; i < module_count; ++i) {
            before[i] = probes[i]->runs.load(std::memory_order_relaxed);
            if (timed) manager.get_module_timing(i, &timing_before[i]);
        }
        const auto measure_begin = Clock::now();
        if (free) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        } else {
            do { manager.submit_tick(request, true); }
            while (Clock::now() - measure_begin < std::chrono::milliseconds(500));
        }
        const auto measure_end = Clock::now();
        for (int i = 0; i < module_count; ++i) {
            after[i] = probes[i]->runs.load(std::memory_order_relaxed);
            if (timed) manager.get_module_timing(i, &timing_after[i]);
        }
        const auto stop_begin = Clock::now();
        manager.stop();
        const auto stop_end = Clock::now();
        for (int i = 0; i < module_count; ++i) {
            if (!probes[i]->ordered.load(std::memory_order_relaxed))
                throw std::runtime_error("FIFO order or input value failure");
            const char* role = math ? "math" : producers && i < producers ? "producer" :
                               producers && i == consumer_idx ? "consumer" : "independent";
            const uint64_t runs = after[i] - before[i];
            const double row_wall = timed ? timing_after[i].total_run_wall_time - timing_before[i].total_run_wall_time : 0.0;
            const uint64_t timed_runs = timed ? timing_after[i].run_count - timing_before[i].run_count : 0;
            std::cout << repeat << ',' << scenario << ',' << mode_name << ',' << work_us << ',' << capacity << ','
                      << int(timed) << ',' << i << ',' << role << ','
                      << seconds(measure_begin, measure_end) << ',' << runs << ','
                      << row_wall << ',' << timed_runs << ','
                      << seconds(start_begin, start_end) << ',' << seconds(first_begin, first_end) << ','
                      << seconds(stop_begin, stop_end) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
