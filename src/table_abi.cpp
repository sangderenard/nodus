#include "table_abi.h"
#include "canvas_abi.h"
#include <cstdlib>
#include "menu_waveform_abi.h"
// Local copy of LassoConfig layout (kept here to avoid cross-translation-unit
// include path issues). The public header forward-declares `LassoConfig`.
typedef struct LassoConfig {
    uint32_t flags;
    uint8_t widget_type;
    uint8_t reserved[3];
    // Edge-spring parameters recorded when a lasso/meta-group enables springs
    float spring_min_rest;    // conservative rest length added when enabling springs
    float spring_reduce_rate; // rate used to reduce rest back to normal
    uint8_t spring_mode;      // reserved mode field for future behaviors
    uint8_t spring_reserved[3];
} LassoConfig;

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <cstdio>
#include <memory>
#include <fstream>
#include <iterator>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <filesystem>
#include <limits>
#include <mutex>
#include <condition_variable>
#include <tuple>
#include "text_render_helper.h"
#include "thread_manager.h"
#include "table_node_groups.h"
#include "rope_sim.h"


// Segmented implementation pieces (see .inl files for details).
#include "inl/table_abi_primitives.inl"
#include "inl/table_abi_core.inl"
#include "inl/table_abi_context_meta.inl"
#include "inl/table_abi_pending_ops.inl"
#include "inl/table_abi_rope_draw.inl"
#include "inl/table_abi_edge_io.inl"
#include "inl/table_abi_serialization.inl"
#include "inl/table_abi_relax_prospective.inl"
#include "inl/table_abi_cpp_helpers.inl"
