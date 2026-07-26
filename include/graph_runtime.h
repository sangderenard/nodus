#pragma once
// GraphRuntime owns nodus tool-graph state that is not, in itself, a
// rendering concern: which tool/plugin is bound to each module row, the
// module tool-kind stack, per-module IO/exec-policy counts, graph node
// contracts, and the ThreadManager instance that schedules all of it.
//
// This is Phase B.1 of separating "canvas" (the pixel rasterizer) from
// nodus's own graph-runtime complexity: GP_CanvasContextImpl still exists
// and still owns rendering state (module positions, rope/cable visuals, LED
// grids, click hit-testing, menus), but the fields relocated here move
// ownership to GraphRuntime, and ThreadManager talks to its owning
// GraphRuntime directly instead of reaching into a process-global canvas
// singleton. GP_CanvasContextImpl keeps reference-member aliases to this
// same storage so existing canvas-side code (rendering, which genuinely
// needs to read tool-binding data to lay out pixels -- see
// sync_module_table_io_layout) continues to compile and behave unchanged.
//
// Deliberately NOT relocated in this phase (see the plan's roadmap for why):
// edges/EdgeInfo (fuses topology with rope-rendering state -- splitting
// that is Phase B.3), module_frame_links (bundles event-routing pointers
// with port UUIDs in one struct -- splitting it is deferred alongside
// EdgeInfo for the same reason), and module_input_state/module_stack_tail/
// module_stack_snapshot* (live GUI input/display-feedback plumbing for the
// MouseListener/KeyboardListener/StackDisplay tool kinds -- meaningful only
// with a live canvas, not a graph-execution concern in the headless sense).

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "thread_manager.h"
#include "tool_api.h"

class GraphRuntime {
public:
    GraphRuntime();
    ~GraphRuntime();

    GraphRuntime(const GraphRuntime&) = delete;
    GraphRuntime& operator=(const GraphRuntime&) = delete;

    // The scheduler this GraphRuntime owns 1:1. Constructed and bound
    // (via ThreadManager::set_graph_runtime) in the GraphRuntime constructor.
    std::unique_ptr<ThreadManager> thread_mgr;

    // Tool-binding state: which tool/plugin is bound to each module row.
    std::vector<std::vector<ModuleIORow>> module_io_rows;
    // Parallel to module_io_rows: instantiated plugin tool objects for
    // plugin-origin rows; null entries indicate no instance.
    std::vector<std::vector<std::unique_ptr<ITool, std::function<void(ITool*)>>>> module_plugin_instances;
    // Ordered tool-kind stack per module.
    std::vector<std::vector<ModuleToolKind>> module_tool_stack;

    // Stable per-(module,contact) binding id map, backing the FIFO
    // reader/writer key resolution used on essentially every edge
    // publish/consume during a tick.
    std::unordered_map<uint64_t, uint64_t> module_binding_id_map;

    // Event payloads stashed mid-delivery to a module frame port. Keyed by
    // (module_idx<<32)|led_idx.
    std::unordered_map<uint64_t, void*> managed_event_payloads;

    // Per-module IO arity and execution policy -- feeds
    // ThreadManager::ModuleContract directly.
    std::vector<int> module_io_in_count;
    std::vector<int> module_io_out_count;
    std::vector<int> module_sim_enabled;
    std::vector<int> module_skip;
    std::vector<int> module_exec_skip_count;
    std::vector<int> module_exec_mode;

    // Graph node/contract representation.
    struct NodeContract {
        int node_id = -1;
        int module_idx = -1;
        std::vector<int> input_types;
        std::vector<int> output_types;
    };
    std::vector<NodeContract> nodes;
    int next_node_id = 1;
    std::vector<int> module_node_id;
};
