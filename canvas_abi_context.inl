// Minimal internal canvas context implementation
// Central ID hook type used by canvas-wide ID generation.
typedef uint64_t (*GP_CanvasIdHookFn)(GP_CanvasContext* ctx, uint64_t hint);
struct GP_CanvasContextImpl {
    int width=0, height=0;
    std::vector<GP_CanvasModuleDesc> modules;
    // internal edge info bundles the desc, rope index, and per-edge hues
    struct EdgeInfo {
        GP_CanvasEdgeDesc desc;
        int rope_idx = -1;
            uint64_t rope_uid = 0ull;
        int type_id = 0; // 0 == wildcard / untyped
        uint32_t subgroup_flags = 0;
        std::vector<float> hues;
        float hue_intensity = 0.0f;
        // Optional overlay keys: when non-zero, these 64-bit keys override
        // the usual module/contact-derived keys and reference custom canvas
        // overlays created by `gp_canvas_create_overlay_with_leds`.
        unsigned long long overlay_key_a = 0ull;
        unsigned long long overlay_key_b = 0ull;
    };
    std::vector<EdgeInfo> edges;
    // optional attached table per module (aligned with `modules` by index)
    std::vector<GP_TableContext*> module_tables;
    std::vector<int> module_table_owned; // 1 if canvas should destroy
    // Optional stage per module (for light-sim demo modules).
    std::vector<GP_StageContext*> module_stages;
    std::vector<int> module_stage_owned; // 1 if canvas should destroy
    std::vector<int> module_is_stage;    // 1 if this module is a stage module
    std::vector<GP_TableImage> module_stage_images; // live image descriptors per stage module
    // per-module cached completed stage RGBA buffers (tight RGBA8 rows)
    std::vector<std::vector<uint8_t>> module_stage_cache_rgba;
    std::vector<int> module_stage_cache_w;
    std::vector<int> module_stage_cache_h;
    std::vector<int> module_stage_cache_pitch;
    std::vector<std::unique_ptr<std::mutex>> module_stage_cache_mu;
    std::vector<uint8_t> module_stage_integrator_mode; // integrator lock per stage module
    std::vector<std::vector<float>> module_stage_integrator_accum; // per-stage integrator accumulators
    struct ModuleBg {
        GP_CanvasModuleBgFn cb = nullptr;
        void* user = nullptr;
        int mode = 0;
        int rays = 512;
        int reflections = 2;
        float blur_sigma = 2.0f;
        int oversample = 2;
        float temporal_decay = 0.92f;
        float temporal_max = 1.0f;
        uint32_t temporal_frame = 0;
        float ray_exposure = 1.0f;
        float ray_bounce_decay = 0.75f;
        float ray_air_decay = 0.0025f;
        uint8_t table_alpha = 255;
        uint8_t table_alpha_ray = 48;
        int header_margin_px = 0; // reserved vertical strip for overlay UI (e.g., stage table header)
        Raytrace2D* ray = nullptr;
        std::vector<float> accum;
        std::vector<float> temporal;
        std::vector<uint8_t> scratch;
        std::vector<uint8_t> layer;
    };
    std::vector<ModuleBg> module_bg;
    // per-module IO counts (inputs, outputs) exposed in the control bar
    std::vector<int> module_io_in_count;
    std::vector<int> module_io_out_count;
    std::vector<int> module_sim_enabled; // per-module sim enable flag (1=simulate,0=skip)
    std::vector<int> module_skip; // per-module full-skip flag (1=skip entire module work)
    std::vector<int> module_exec_skip_count; // per-module execution cadence skip count (0 = every frame)
    std::vector<std::vector<int>> module_io_input_rows;
    std::vector<std::vector<int>> module_io_output_rows;
    std::vector<MolexLayoutInfo> module_input_layout;
    std::vector<MolexLayoutInfo> module_output_layout;
    std::vector<std::vector<ModuleIORow>> module_io_rows;
    // parallel structure to `module_io_rows` storing instantiated plugin tool
    // instances for plugin-origin rows; null entries indicate no instance.
    std::vector<std::vector<std::unique_ptr<ITool, std::function<void(ITool*)>>>> module_plugin_instances;
    std::vector<std::vector<ModuleIORow>> module_table_rows;
    std::vector<ModuleFrameLedGroup> module_frame_leds;
    std::vector<ModuleFrameLink> module_frame_links;
    std::vector<ModulePreviewBuffer> module_preview_buffers;
    std::vector<ModuleStackTail> module_stack_tail;
    std::vector<std::unordered_map<int, std::vector<float>>> module_stack_snapshots;
    std::vector<std::vector<ModuleToolKind>> module_tool_stack;
    // Action subscriber registry: map action_id -> list of (callback,user)
    std::unordered_map<int32_t, std::vector<std::pair<GP_CanvasActionSubscriberFn, void*>>> action_subscribers;
    std::mutex action_subscribers_mu;
    // Action -> bound ports (module,row,col,led,pending_ptr)
    // Per-binding edge payload published into root table FIFOs. Layout is
    // declared in `canvas_abi.h` as `EventPayload` and used by the manager.
    struct ActionBinding { int module_idx; int row; int col; int led_idx; void* pending_ptr; int root_edge_idx = -1; unsigned long long writer_key = 0ull; };
    std::unordered_map<int32_t, std::vector<ActionBinding>> action_port_bindings;
    // optional per-module key-recorder state pointer
    std::vector<void*> module_key_recorder_state;
    std::vector<ModuleInputState> module_input_state;
    // per-module lightweight chat state (used to visually confirm rope traffic)
    std::vector<std::string> module_chat_text;
    std::vector<ChatCol> module_chat_color;
    std::vector<int> module_chat_ttl; // frames remaining to show chat highlight
    // cable style/hues
    int jacket_px = 4;
    int jacket_border = 2;
    std::vector<float> hues;
    float hue_intensity = 0.0f;
    // selection state for click-to-connect behavior. `anchor_x/anchor_y` are
    // pixel coordinates (canvas space) for the selected contact when the
    // selection originates from a table hitbox; they remain -1 when unset.
    struct Sel { int module = -1; int contact_idx = -1; int left = -1; int anchor_x = -1; int anchor_y = -1; } selected;
    // click-listen: when true, root-table click actions are captured rather
    // than immediately dispatched. `pending_action` holds an allocated
    // copy of the action intent and can be bound into module frame ptrs.
    struct PendingAction { int32_t action_id = 0; GP_TableHitBox hit{}; uint64_t aux_uid = 0ull; };
    bool click_listen_mode = false;
    PendingAction* pending_action = nullptr; // owned when non-null
    // transient module index used by root-table action dispatch
    int dispatch_module_idx = -1;
    // per-canvas drag state (moved here to avoid a global map)
    DragState drag;
    // provisional rope index while user is selecting a contact and moving the mouse
    int prospective_rope_idx = -1;
    // lasso meta rope/module created for the current lasso action
    int lasso_rope_idx = -1;
    int lasso_module_idx = -1;
    // rope simulation tuning parameters and UI bar height
    int rope_bar_h = 28; // extra bar above control bar
    int sim_segs = 8;
    float sim_slack = 0.0f;
    int sim_iters = 8;
    float sim_damping = 0.01f;
    float sim_maxforce = 800.0f;
    // global rope-sim frame-skip count (0 => step every frame).
    // The count is the number of frames to skip between steps (step every count+1 frames).
    int sim_frame_skip_count = 0;
    int sim_root_paused = 0; // toolbar-level pause for the root/global rope sim
    uint32_t debug_flags = GP_CANVAS_DEBUG_NORENDER_MODE; // debug render/sim overrides (see GP_CanvasDebugFlags)
    // tool selection state: separate groups (exclusive within group)
    // canvas tool group: 0 = select, 1 = new table, 2 = edge mode, 3 = new stage
    int selected_tool_canvas = 0;
    // edge tool group: 0=create, 1=destroy, 2=on-change, 3=continuous (-1 = none)
    int selected_tool_edge = 0;
    int edge_order_value = 0;
    int edge_order_tool_active = 0;
    // table tool group: 0 = neutral, 1 = select, 2 = menu
    int selected_tool_table = 0;
    // kpn tool group: 0..2 (K, P, N), -1 = none
    int selected_tool_kpn = -1;
    // fifo policy subgroup selector: bitmask of enabled subgroup flags
    uint32_t selected_tool_subgroup_flags = 0u;

    // lasso mode: when true, canvas will route mouse drags to the
    // meta-edge lasso handler (keyboard-toggleable). This is intentionally
    // independent from the toolbar/menu tool groups so it does not alter
    // visible toolbar layout.
    bool lasso_mode = false;
    // transient lasso path points (canvas-local coords). Collected while
    // `lasso_mode` is enabled and the user drags the mouse; processed on
    // mouse-up to create a meta-group in the attached table.
    std::vector<std::pair<float,float>> lasso_points;
    // optional lasso event callback (registered via gp_canvas_set_lasso_callback)
    GP_CanvasLassoFn lasso_cb = nullptr;
    void* lasso_cb_user = nullptr;
    // optional overlay button callback (registered via gp_canvas_set_overlay_button_callback)
    GP_CanvasOverlayButtonFn overlay_button_cb = nullptr;
    void* overlay_button_user = nullptr;
    // optional click-drag callback (registered via gp_canvas_set_click_drag_callback)
    GP_CanvasClickDragFn click_drag_cb = nullptr;
    void* click_drag_cb_user = nullptr;
    // transient click-drag tracking (independent of tool state)
    bool click_drag_active = false;
    float click_drag_start_x = 0.0f;
    float click_drag_start_y = 0.0f;
    std::array<std::array<float, 4>, kSubgroupBinCount> subgroup_base_rgba{};
    std::array<Color, kSubgroupBinCount> subgroup_base_colors{};
    std::array<std::atomic<uint32_t>, kSubgroupBinCount> subgroup_target_rgba{};
    std::array<Color, 1 << kSubgroupBinCount> subgroup_palette{};
    std::array<float, 1 << kSubgroupBinCount> subgroup_palette_hue{};
    struct ToolbarLedBox {
        int x0=0, y0=0, x1=0, y1=0; // view coords
        int wx0=0, wy0=0, wx1=0, wy1=0; // world coords (with offset)
        int subgroup_idx=0;
    };
    std::vector<ToolbarLedBox> toolbar_leds;
    bool tool_menu_open = false;
    bool rope_menu_open = false;
    // Custom canvas overlays keyed by small integer id. Overlays are
    // lightweight rectangle-only panes carrying two LED positions used for
    // anchoring ropes and interactions. Keys returned to callers encode the
    // overlay id and led index.
    struct OverlayEntry {
        int id = 0;
        float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
        unsigned long long key_a = 0ull;
        unsigned long long key_b = 0ull;
        // stable per-port UUIDs for the two contact positions
        uint64_t port_uuid_a = 0ull;
        uint64_t port_uuid_b = 0ull;
        // cached numeric-control rect in screen/view coords (pixels)
        int ctrl_x = -1, ctrl_y = -1, ctrl_w = 0, ctrl_h = 0;
        // authoritative meta-group binding: when an overlay represents a
        // meta-group widget, these fields are set so clicks map directly
        // to the table-side meta-group without rescanning ropes.
        GP_TableContext* meta_table = nullptr;
        GP_MetaGroup* meta_mg = nullptr;
    };
    std::unordered_map<int, OverlayEntry> overlays;
    // map port UUID -> (overlay id, led index 0/1)
    std::unordered_map<uint64_t, std::pair<int,int>> overlay_port_uuid_map;
    // map module port UUID -> (module_idx, row, led_idx)
    std::unordered_map<uint64_t, std::array<int,3>> module_port_uuid_map;
    // map overlay sentinel keys -> overlay id for fast lookup and to
    // ensure we never create duplicate overlays for the same keys
    std::unordered_map<unsigned long long, int> overlay_key_map;
    // pending snapshots waiting for overlays or backing tables to exist
    std::vector<CanvasMetaSnapshot> pending_meta_snapshots;
    // Post-load pending meta groups that need deterministic finalize pass
    struct PostLoadMetaPending { GP_TableContext* table; GP_MetaGroup* mg; float ring_u; int ring_mode; unsigned long long overlay_a; unsigned long long overlay_b; };
    std::vector<PostLoadMetaPending> post_load_meta_pending;
    // map canonical root table keys -> overlay sentinel keys so table-led
    // queries can resolve to overlay positions when appropriate.
    std::unordered_map<uint64_t, unsigned long long> canonical_to_overlay;
    // module UUID -> module_idx
    std::unordered_map<uint64_t, int> module_uuid_map;
    // mapping from persisted rope id -> (table, rope_id) for deterministic restore
    std::unordered_map<uint64_t, RopeIdEntry> rope_id_map;
    bool rope_map_dirty = false;
    // mapping from persisted meta-group id (lasso id) -> sim_group_idx
    std::unordered_map<uint64_t, int> lasso_id_map;
    // next stable port UUID (monotonic)
    uint64_t next_port_uuid = 1;
    // next stable module UUID (monotonic)
    uint64_t next_module_uuid = 1;
    // Central ID hook: callers may install a custom ID generator that
    // receives an optional hint. If null, `gp_canvas_generate_id` will
    // produce a default id using a simple hash+counter.
    GP_CanvasIdHookFn id_hook = nullptr;
    // fallback counter used by default generator to reduce colliding outputs
    std::atomic<uint64_t> next_id_counter{1};
    int next_overlay_id = 1;
    bool plugin_menu_open = false;
    int plugin_menu_module_idx = -1;
    std::vector<ModuleToolKind> plugin_tool_kinds;
    std::vector<std::string> plugin_tool_ids;
    std::vector<std::string> plugin_tool_labels;
    int io_attachment_count = 1;
    // number of send/receive row pairs to display (each pair == one grid-row)
    int module_frame_pair_count = 2;
    int table_tool_number = 1;
    // which module (if any) has keyboard/focus for table editing
    int focused_module = -1;
    // registered host windows (opaque pointers)
    std::vector<void*> windows;
    // mapping from window pointer to stable node id for backing graph
    std::unordered_map<void*, int> window_node_ids;
    // next unique node id for graph nodes
    int next_node_id = 1;
    // per-module node id (aligned with `modules`) or -1 if none
    std::vector<int> module_node_id;
    // per-module stable UUIDs (monotonic 64-bit). 0 == unset
    std::vector<uint64_t> module_uuids;
    // graph node/contract representation
    struct NodeContract {
        int node_id = -1;
        int module_idx = -1; // which module this node belongs to (-1 if none)
        std::vector<int> input_types; // supported input type ids
        std::vector<int> output_types; // supported output type ids
    };
    std::vector<NodeContract> nodes;
    // UI control bar height (in canvas-local pixels)
    int control_bar_h = 56;
    // viewport offset (world origin visible at (0,0) in screen space)
    int offset_x = 0;
    int offset_y = 0;
    int scroll_x_needed = 0;
    int scroll_y_needed = 0;
    // optional table container (non-owning unless marked)
    GP_TableContext* container_table = nullptr;
    int container_table_owned = 0;
    int root_actions_installed = 0;
    int root_module_idx = -1; // synthetic module that mirrors the canvas root table
    // autosave parameters (path may be empty to disable)
    std::string autosave_path;
    double autosave_interval_s = 0.0;
    double autosave_accum_s = 0.0;
    bool thread_mgr_paused = true;
    int thread_mgr_delay_ms = 0;
    double thread_mgr_delay_accum_s = 0.0;
    std::unique_ptr<ThreadManager> thread_mgr;
    GP_CanvasContextImpl(int w, int h): width(w), height(h) {}
};

// (global drag map removed; each canvas has its own DragState member)

static GP_CanvasContextImpl* g_canvas_context_singleton = nullptr;

static void canvas_recompute_subgroup_palette(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    constexpr int kPaletteSize = 1 << kSubgroupBinCount;
    for (int mask = 0; mask < kPaletteSize; ++mask) {
        if (mask == 0) {
            Color neutral{120, 120, 130, 255};
            ctx->subgroup_palette[0] = neutral;
            ctx->subgroup_palette_hue[0] = 0.0f;
            continue;
        }
        float sum_r = 0.0f;
        float sum_g = 0.0f;
        float sum_b = 0.0f;
        float sum_a = 0.0f;
        int count = 0;
        for (int i = 0; i < kSubgroupBinCount; ++i) {
            if ((mask & (1 << i)) == 0) continue;
            const Color base = ctx->subgroup_base_colors[static_cast<size_t>(i)];
            sum_r += static_cast<float>(base.r);
            sum_g += static_cast<float>(base.g);
            sum_b += static_cast<float>(base.b);
            sum_a += static_cast<float>(base.a);
            ++count;
        }
        if (count <= 0) count = 1;
        float inv = 1.0f / static_cast<float>(count);
        Color mixed{
            static_cast<uint8_t>(std::lround(std::clamp(sum_r * inv, 0.0f, 255.0f))),
            static_cast<uint8_t>(std::lround(std::clamp(sum_g * inv, 0.0f, 255.0f))),
            static_cast<uint8_t>(std::lround(std::clamp(sum_b * inv, 0.0f, 255.0f))),
            static_cast<uint8_t>(std::lround(std::clamp(sum_a * inv, 0.0f, 255.0f)))
        };
        ctx->subgroup_palette[static_cast<size_t>(mask)] = mixed;
        ctx->subgroup_palette_hue[static_cast<size_t>(mask)] = rgb_to_hue(mixed);
    }
}

static void canvas_init_subgroup_palette(GP_CanvasContextImpl* ctx) {
    if (!ctx) return;
    for (int i = 0; i < kSubgroupBinCount; ++i) {
        float hue = static_cast<float>(i) / static_cast<float>(kSubgroupBinCount);
        Color base = hsv_to_color(hue, 0.9f, 0.95f, 255);
        ctx->subgroup_base_rgba[static_cast<size_t>(i)] = {float(base.r), float(base.g), float(base.b), float(base.a)};
        ctx->subgroup_base_colors[static_cast<size_t>(i)] = base;
        ctx->subgroup_target_rgba[static_cast<size_t>(i)].store(pack_rgba(base), std::memory_order_relaxed);
    }
    canvas_recompute_subgroup_palette(ctx);
}

static int canvas_resolve_rope_id_to_sim_index(GP_TableContext* table, uint64_t rope_id) {
    if (!table || rope_id == 0ull) return -1;
    if (!gp_table_get_rope_sim(table)) return -1;
    return gp_table_resolve_rope_id_to_sim_index(table, rope_id);
}

static uint64_t canvas_find_rope_id_for_sim(GP_CanvasContextImpl* c, GP_TableContext* table, int sim_idx) {
    if (!c || !table || sim_idx < 0) return 0ull;
    if (!gp_table_get_rope_sim(table)) return 0ull;
    for (const auto &kv : c->rope_id_map) {
        if (kv.second.table != table) continue;
        int resolved = gp_table_resolve_rope_id_to_sim_index(table, kv.second.rope_id);
        if (resolved == sim_idx) return kv.second.rope_id;
    }
    return 0ull;
}

static void canvas_refresh_rope_map(GP_CanvasContextImpl* c) {
    if (!c || !c->rope_map_dirty) return;
    std::unordered_set<GP_TableContext*> tables;
    tables.reserve(c->rope_id_map.size());
    for (const auto &kv : c->rope_id_map) {
        if (kv.second.table) tables.insert(kv.second.table);
    }
    for (auto *table : tables) {
        if (!table) continue;
        if (!gp_table_get_rope_sim(table)) continue;
        bool needs_rebuild = false;
        for (const auto &kv : c->rope_id_map) {
            if (kv.second.table != table) continue;
            if (gp_table_resolve_rope_id_to_sim_index(table, kv.second.rope_id) < 0) {
                needs_rebuild = true;
                break;
            }
        }
        if (!needs_rebuild) continue;
        int cnt = gp_table_get_rope_id_count(table);
        if (cnt <= 0) continue;
        std::vector<uint64_t> tmp(static_cast<size_t>(cnt));
        int got = gp_table_get_rope_ids(table, tmp.data(), cnt);
        if (got > 0) {
            gp_table_set_rope_ids_from_array(table, tmp.data(), got);
            printf("canvas: refreshed rope map for table=%p entries=%d\n", (void*)table, got);
        }
    }
    c->rope_map_dirty = false;
}

