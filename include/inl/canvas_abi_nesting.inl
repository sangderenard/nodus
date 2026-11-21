// A canvas as a compartment (docs/MESH_CANVAS_UP_PORTS_2026-11-21.md, stage 3): a canvas a table row holds runs on its
// own scheduler, owns its own tables and keeps its own sim frame count, so nothing it does reaches the process's primary
// canvas through the global singleton. These are the lookups the table and scheduler layers use to find their canvas.

ThreadManager* canvas_thread_manager(GP_CanvasContext* canvas) {
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(canvas);
    return c ? c->thread_mgr.get() : nullptr;
}

void canvas_sim_frame_state(GP_CanvasContext* canvas, int* out_skip_count, uint64_t* out_tick) {
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(canvas);
    if (!c) return;
    if (out_skip_count) *out_skip_count = c->sim_frame_skip_count;
    if (out_tick) *out_tick = c->sim_frame_tick;
}

// One canvas frame: the canvas's own count advances; the primary canvas also keeps the process-wide pair current for
// tables no canvas owns.
static void canvas_advance_sim_frame(GP_CanvasContextImpl* c) {
    if (!c) return;
    c->sim_frame_tick = (c->sim_frame_tick + 1) % 0xFFFFFFFFFFFFu;
    if (c == g_canvas_context_singleton) {
        gp_table_set_global_sim_frame_skip_count(c->sim_frame_skip_count);
        gp_table_advance_global_sim_tick();
    }
}

// Every table a canvas creates or is given is that canvas's.
static GP_TableContext* canvas_own_table(GP_CanvasContextImpl* c, GP_TableContext* t) {
    if (c && t) gp_table_set_owner_canvas(t, reinterpret_cast<GP_CanvasContext*>(c));
    return t;
}

// A row's saved state (ITool::serialize / deserialize), the bytes a ROW_STATE record carries. Set it at a quiescent
// boundary (no tick running), as rows are bound. A canvas row's state is the canvas it holds.
extern "C" int gp_canvas_row_state_get(GP_CanvasContext* ctx_, int module_idx, int row, uint8_t* out, int capacity,
                                       int* out_size) {
    if (out_size) *out_size = 0;
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!c || module_idx < 0 || row < 0 || module_idx >= static_cast<int>(c->module_plugin_instances.size())) return 0;
    auto& insts = c->module_plugin_instances[static_cast<size_t>(module_idx)];
    if (row >= static_cast<int>(insts.size()) || !insts[static_cast<size_t>(row)]) return 0;
    MemoryOutputArchive state;
    bool saved = false;
    try { saved = insts[static_cast<size_t>(row)]->serialize(state); } catch (...) { saved = false; }
    if (!saved) return 0;
    if (out_size) *out_size = static_cast<int>(state.bytes.size());
    if (!out || capacity < static_cast<int>(state.bytes.size())) return 0;   // size reported: call again with room
    std::memcpy(out, state.bytes.data(), state.bytes.size());
    return 1;
}

extern "C" int gp_canvas_row_state_set(GP_CanvasContext* ctx_, int module_idx, int row, const uint8_t* bytes, int size) {
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!c || module_idx < 0 || row < 0 || size < 0 || (size > 0 && !bytes) ||
        module_idx >= static_cast<int>(c->module_plugin_instances.size())) return 0;
    auto& insts = c->module_plugin_instances[static_cast<size_t>(module_idx)];
    if (row >= static_cast<int>(insts.size()) || !insts[static_cast<size_t>(row)]) return 0;
    const std::vector<uint8_t> state(bytes, bytes + size);
    MemoryInputArchive in(state);
    try { return insts[static_cast<size_t>(row)]->deserialize(in) ? 1 : 0; } catch (...) { return 0; }
}
