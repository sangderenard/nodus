#include "canvas_abi.h"
#include "rope_sim.h"
#include "table_abi.h"
#include "raytrace_2d.h"
#include "stage_abi.h"
#include "text_render_helper.h"
#include "thread_manager.h"
#include "graph_runtime.h"
#include "module_library.h"
#include "module_library_actualizer.h"
#include "labels.h"
#include "module_preview.h"
#include "plugin_manager.h"
#include "tool_registry.h"
#include "tool_api.h"
#include "table_node_groups.h"

#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <set>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iterator>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <array>
#include <atomic>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <Eigen/Dense>
#include <chrono>
#include <mutex>
#include <cstdlib>


// Segmented implementation pieces (see .inl files for details).
#include "inl/canvas_abi_primitives.inl"
#include "inl/canvas_abi_context.inl"
#include "inl/canvas_abi_rope_map.inl"
#include "inl/canvas_abi_create.inl"
#include "inl/canvas_abi_table_step.inl"
#include "inl/canvas_abi_actions_edges.inl"
#include "inl/canvas_abi_save_load.inl"
#include "inl/canvas_abi_overlay_raster.inl"

extern "C" int gp_canvas_set_module_stack_tail(
    GP_CanvasContext* ctx, int module_idx, const float* values, int count) {
    if (!ctx || count < 0 || count > 4096 || (count && !values)) return 0;
    auto* canvas = reinterpret_cast<GP_CanvasContextImpl*>(ctx);
    if (module_idx < 0 || module_idx >= static_cast<int>(canvas->modules.size())) return 0;
    module_stack_tail_write(canvas, module_idx, values, count);
    return 1;
}
