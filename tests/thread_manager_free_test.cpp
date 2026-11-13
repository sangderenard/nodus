#include "graph_runtime.h"
#include "table_abi.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>
#include <functional>

class CountingTool final : public ITool {
public:
    CountingTool(std::atomic<uint64_t>& count, std::chrono::microseconds work)
        : count_(count), work_(work) {}
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext&) override {
        const auto deadline = std::chrono::steady_clock::now() + work_;
        while (std::chrono::steady_clock::now() < deadline)
            std::atomic_signal_fence(std::memory_order_seq_cst);
        count_.fetch_add(1, std::memory_order_relaxed);
    }
private:
    std::atomic<uint64_t>& count_;
    std::chrono::microseconds work_;
};

class StackTool final : public ITool {
public:
    explicit StackTool(std::function<void(ToolStackContext&)> fn) : fn_(std::move(fn)) {}
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& ctx) override { fn_(ctx); }
private:
    std::function<void(ToolStackContext&)> fn_;
};

static void expect(bool condition) { if (!condition) std::abort(); }

int main() {
    GraphRuntime graph;
    graph.module_io_rows.resize(3);
    graph.module_plugin_instances.resize(3);
    std::atomic<uint64_t> fast{0}, slow{0}, scheduled{0};
    std::unique_ptr<GP_TableContext, decltype(&gp_table_destroy)> tables[] = {
        {gp_table_create(nullptr), gp_table_destroy},
        {gp_table_create(nullptr), gp_table_destroy},
        {gp_table_create(nullptr), gp_table_destroy},
    };
    ThreadManager::TickRequest req{};
    for (int i = 0; i < 3; ++i) {
        expect(bool(tables[i]));
        gp_table_set_sim_enabled(tables[i].get(), 0);
        ThreadManager::ModuleContract mod{};
        mod.module_idx = i;
        mod.table = tables[i].get();
        mod.sim_enabled = 0;
        mod.exec_mode = i == 2 ? ThreadManager::ExecMode::Sequential : ThreadManager::ExecMode::Free;
        req.modules.push_back(mod);
        ModuleIORow row{};
        row.tool_origin = ModuleToolOrigin::Plugin;
        graph.module_io_rows[i].push_back(row);
        auto* count = i == 0 ? &fast : i == 1 ? &slow : &scheduled;
        auto work = i == 1 ? std::chrono::microseconds(500) : std::chrono::microseconds(0);
        graph.module_plugin_instances[i].emplace_back(new CountingTool(*count, work), [](ITool* p) { delete p; });
    }
    auto& manager = *graph.thread_mgr;
    manager.start();
    manager.submit_tick(req, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    const uint64_t first_fast = fast.load();
    const uint64_t first_slow = slow.load();
    expect(first_fast > 0 && first_slow > 0);
    expect(scheduled.load() == 1);
    manager.submit_tick(req, true);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    expect(fast.load() > first_fast && slow.load() > first_slow);
    expect(scheduled.load() == 2);
    manager.stop();
    const auto stopped_fast = fast.load();
    const auto stopped_slow = slow.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    expect(fast.load() == stopped_fast && slow.load() == stopped_slow);

    // A slow consumer fills a two-slot lossless edge. The producer must pause
    // before invoking its next row, while an unrelated table keeps running.
    GraphRuntime connected;
    connected.module_io_rows.resize(3);
    connected.module_plugin_instances.resize(3);
    std::unique_ptr<GP_TableContext, decltype(&gp_table_destroy)> root(gp_table_create(nullptr), gp_table_destroy);
    std::unique_ptr<GP_TableContext, decltype(&gp_table_destroy)> nodes[] = {
        {gp_table_create(nullptr), gp_table_destroy},
        {gp_table_create(nullptr), gp_table_destroy},
        {gp_table_create(nullptr), gp_table_destroy},
    };
    expect(bool(root) && bool(nodes[0]) && bool(nodes[1]) && bool(nodes[2]));
    std::atomic<uint64_t> produced{0}, consumed{0}, independent{0};
    std::atomic<bool> in_order{true};
    auto add_row = [&](int module, ModuleRowKind kind, int contact, ITool* tool = nullptr) {
        ModuleIORow row{};
        row.kind = kind;
        row.contact_idx = contact;
        if (tool) row.tool_origin = ModuleToolOrigin::Plugin;
        connected.module_io_rows[module].push_back(row);
        connected.module_plugin_instances[module].emplace_back(tool, [](ITool* p) { delete p; });
    };
    add_row(0, ModuleRowKind::Tool, 0, new StackTool([&](ToolStackContext& ctx) {
        tool_stack_push(ctx.stack, static_cast<float>(produced.fetch_add(1) + 1));
    }));
    add_row(0, ModuleRowKind::Output, 1);
    add_row(1, ModuleRowKind::Input, 0);
    add_row(1, ModuleRowKind::Tool, 0, new StackTool([&](ToolStackContext& ctx) {
        const float value = tool_stack_pop(ctx.stack);
        if (value != static_cast<float>(consumed.fetch_add(1) + 1))
            in_order.store(false);
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
        while (std::chrono::steady_clock::now() < until)
            std::atomic_signal_fence(std::memory_order_seq_cst);
    }));
    add_row(2, ModuleRowKind::Tool, 0, new StackTool([&](ToolStackContext&) { ++independent; }));
    const uint64_t output_key = uint64_t(1) << 16;
    const uint64_t input_key = uint64_t(1) << 32;
    expect(gp_table_add_edge(root.get(), output_key, input_key));
    GP_TableEdgeTensorSpecTyped spec{};
    spec.dims[0] = 1;
    spec.dim_count = 1;
    spec.slots = 2;
    spec.top_k = 0;
    spec.elem_size = sizeof(float);
    spec.type_id = ValueTypeRegistry::global().builtin(VT_FLOAT32);
    expect(gp_table_edge_set_tensor_spec(root.get(), 0, &spec));
    ThreadManager::TickRequest connected_req{};
    connected_req.root_table = root.get();
    for (int i = 0; i < 3; ++i) {
        gp_table_set_sim_enabled(nodes[i].get(), 0);
        ThreadManager::ModuleContract module{};
        module.module_idx = i;
        module.table = nodes[i].get();
        module.sim_enabled = 0;
        module.exec_mode = ThreadManager::ExecMode::Free;
        module.stack_contract = ModuleStackContract::ScalarF32;
        connected_req.modules.push_back(module);
    }
    ThreadManager::EdgeContract edge{};
    edge.edge_idx = 0;
    edge.a_module = 0;
    edge.a_contact_idx = 1;
    edge.b_module = 1;
    edge.b_contact_idx = 0;
    connected_req.edges.push_back(edge);
    auto& connected_manager = *connected.thread_mgr;
    connected_manager.start();
    connected_manager.submit_tick(connected_req, true);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (consumed.load() < 500 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    connected_manager.stop();
    expect(consumed.load() >= 500);
    expect(in_order.load());
    expect(produced.load() >= consumed.load());
    expect(produced.load() <= consumed.load() + 2);
    expect(independent.load() > consumed.load());
}
