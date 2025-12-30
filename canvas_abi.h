#ifndef NODUS_CANVAS_ABI_H
#define NODUS_CANVAS_ABI_H
// Minimal canvas API for placing modules and drawing droopy rope edges between contacts
#pragma once

#include <stdint.h>

// Expose module/frame constants used across compilation units.
// Declarations only; definitions live in canvas_abi.cpp.
extern const int kModuleExtraLedCount;
extern const int kModuleExtraLedRows;
extern const int kModuleFrameContactBase;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GP_CanvasContext GP_CanvasContext;
typedef struct GP_TableContext GP_TableContext; // forward from table_abi
typedef struct GP_TableHitBox GP_TableHitBox; // forward

// Pointer-mode event payload published into root table FIFOs for bound
// canvas actions. Consumers should cast pointer to this layout and free
// contents when handled.
typedef struct EventPayload { void* pending; int src_module; int frame_idx; } EventPayload;

// Module background rasterizer hook.
typedef void(*GP_CanvasModuleBgFn)(void* user, int module_idx, int width, int height, uint8_t* out_rgba, int32_t out_pitch);

// Lasso event callback: event_type is 1=start, 2=sample, 3=end (finalized).
// `points` is a contiguous XY float array (2*count) in canvas/world coords.
typedef void(*GP_CanvasLassoFn)(void* user, int event_type, const float* points, int count);

// Click-drag event callback: event_type is 1=start, 2=move, 3=end.
// `sx,sy` are the drag start in canvas/world coords, `ex,ey` are the current/end coords.
typedef void(*GP_CanvasClickDragFn)(void* user, int event_type, float sx, float sy, float ex, float ey);

// Overlay button callback: invoked when an overlay's mode button is clicked.
// Returns 1 if the event was handled (no further action), 0 to let canvas
// perform the default sim-only toggle. `button_id` identifies which button
// inside the overlay was clicked (0 = mode 'R' button).
typedef int(*GP_CanvasOverlayButtonFn)(void* user, unsigned long long overlay_key_a, unsigned long long overlay_key_b, int rope_idx, int button_id);

// Register an overlay button callback for this canvas. Passing NULL clears.
int gp_canvas_set_overlay_button_callback(GP_CanvasContext* ctx, GP_CanvasOverlayButtonFn cb, void* user);

// Register a click-drag event callback for this canvas. Passing NULL clears.
int gp_canvas_set_click_drag_callback(GP_CanvasContext* ctx, GP_CanvasClickDragFn cb, void* user);

// Register a lasso event callback for this canvas. Passing NULL clears.
int gp_canvas_set_lasso_callback(GP_CanvasContext* ctx, GP_CanvasLassoFn cb, void* user);

// Module descriptor
typedef struct {
    int32_t x, y; // top-left
    int32_t w, h;
    char label[64];
} GP_CanvasModuleDesc;

// Edge reference: module index and contact index (0..n-1)
typedef struct {
    int32_t a_module;
    int32_t a_contact_idx; // 0 = topmost
    int32_t b_module;
    int32_t b_contact_idx;
} GP_CanvasEdgeDesc;

GP_CanvasContext* gp_canvas_create(int width, int height);
void gp_canvas_destroy(GP_CanvasContext* ctx);

// add module, returns module index or -1
int gp_canvas_add_module(GP_CanvasContext* ctx, const GP_CanvasModuleDesc* desc);
// move module
int gp_canvas_move_module(GP_CanvasContext* ctx, int module_idx, int x, int y);

// add edge, returns edge index or -1
int gp_canvas_add_edge(GP_CanvasContext* ctx, const GP_CanvasEdgeDesc* desc);

// rasterize into RGBA buffer sized width*height*4; returns 1 on success
int gp_canvas_raster_rgba(GP_CanvasContext* ctx, uint8_t* out_rgba, int32_t out_len_bytes);

// Set cable rendering style for this canvas (jacket radius and inner border)
int gp_canvas_set_cable_style(GP_CanvasContext* ctx, int jacket_px, int jacket_border);

// Set global hue array for colored core rendering. `hues` is an array of float in [0..1].
// The function copies the hue data internally. Passing nullptr clears hues.
int gp_canvas_set_edge_hues(GP_CanvasContext* ctx, const float* hues, int hue_count, float hue_intensity);

// Update subgroup toolbar LED colors. `rgba` is a float array in stride-of-4 RGBA values.
int gp_canvas_set_subgroup_toolbar_rgba(GP_CanvasContext* ctx, const float* rgba, int value_count);
// Set a single subgroup toolbar RGBA color at index `idx` (rgba = 4 floats)
int gp_canvas_set_subgroup_toolbar_rgba_at(GP_CanvasContext* ctx, int idx, const float* rgba);

// Debug flags for simple render/sim control (all disabled by default).
typedef enum GP_CanvasDebugFlags {
    GP_CANVAS_DEBUG_SIMPLE_RENDER = 1u << 0,
    GP_CANVAS_DEBUG_SEGMENTS_1 = 1u << 1,
    GP_CANVAS_DEBUG_NO_LIGHTING = 1u << 2,
    GP_CANVAS_DEBUG_NO_GRAVITY = 1u << 3,
    GP_CANVAS_DEBUG_NO_SPRINGS = 1u << 4,
    GP_CANVAS_DEBUG_RING_STATIC = 1u << 5,
    GP_CANVAS_DEBUG_NORENDER_MODE = (GP_CANVAS_DEBUG_SIMPLE_RENDER |
                                     GP_CANVAS_DEBUG_SEGMENTS_1 |
                                     GP_CANVAS_DEBUG_NO_LIGHTING |
                                     GP_CANVAS_DEBUG_NO_GRAVITY |
                                     GP_CANVAS_DEBUG_NO_SPRINGS |
                                     GP_CANVAS_DEBUG_RING_STATIC)
} GP_CanvasDebugFlags;

int gp_canvas_set_debug_flags(GP_CanvasContext* ctx, uint32_t flags);
int gp_canvas_get_debug_flags(GP_CanvasContext* ctx, uint32_t* out_flags);

// Click-listen mode: when enabled, root-table click actions will be captured
// instead of acted upon. Use `gp_canvas_bind_pending_action_to_module` to
// bind the retained action pointer into a module's frame receive slot.
int gp_canvas_set_click_listen_mode(GP_CanvasContext* ctx, int enable);
int gp_canvas_get_click_listen_mode(GP_CanvasContext* ctx, int* out_enabled);
// Bind retained action (if any) to the most-left unused receive frame ptr of module.
int gp_canvas_bind_pending_action_to_module(GP_CanvasContext* ctx, int module_idx);
// Query a module frame ptr previously set via gp_canvas_set_module_frame_ptr.
void* gp_canvas_get_module_frame_ptr(GP_CanvasContext* ctx, int module_idx, int is_send, int led_idx);
// Return the bound module-frame pointer for a given module contact id.
// `contact_idx` should be in the frame contact range (kModuleFrameContactBase..).
void* gp_canvas_get_module_frame_ptr_for_contact(GP_CanvasContext* ctx, int module_idx, int contact_idx);
// Free a pending action pointer that was previously created by click-listen
// and bound into a module frame. Returns 1 on success.
int gp_canvas_free_pending_action(GP_CanvasContext* ctx, void* pending_ptr);
// Invoke a bound pending action pointer (as stored in module-frame ptrs).
// The pointer should be a `GP_CanvasContextImpl::PendingAction*` previously
// created by click-listen capture and bound into a module frame slot. This
// function will execute the action semantics on the canvas (same as a root
// table action). Returns 1 on success.
int gp_canvas_invoke_pending_action(GP_CanvasContext* ctx, void* pending_ptr);

// Note: the queued pending-action pop API was removed. Canvas bindings now
// publish pointer-mode EventPayloads into root table FIFOs using
// `gp_edge_publish_ptr`; the manager consumes them via normal FIFO APIs.

// Action subscriber callback. Return 1 if the subscriber handled the action
// (suppressing default canvas dispatch), 0 to allow further processing.
typedef int(*GP_CanvasActionSubscriberFn)(void* user, GP_CanvasContext* ctx, int32_t action_id, const GP_TableHitBox* hit);

// Subscribe/unsubscribe to canvas actions. `action_id` is the numeric
// action identifier (e.g., GP_TABLE_ACTION_* or application-defined ids).
// Returns 1 on success, 0 on failure.
int gp_canvas_subscribe_action(GP_CanvasContext* ctx, int32_t action_id, GP_CanvasActionSubscriberFn cb, void* user);
int gp_canvas_unsubscribe_action(GP_CanvasContext* ctx, int32_t action_id, GP_CanvasActionSubscriberFn cb, void* user);

// Return pointer to singleton canvas (may be null)
GP_CanvasContext* gp_canvas_get_singleton();

// Return the synthetic root-reflection module index, or -1 if none
int gp_canvas_get_root_module_idx();

// Forward a mouse click (canvas-local coords). Returns 1 if handled.
int gp_canvas_on_click(GP_CanvasContext* ctx, int x, int y);

// Mouse drag handlers to support click-and-dragging modules.
// Mouse event APIs now use floating-point coordinates and deltas (SDL3 provides
// higher-precision mouse values). Each event supplies local canvas coords and
// motion deltas (dx, dy).
int gp_canvas_on_mouse_down(GP_CanvasContext* ctx, float x, float y, float dx, float dy, int button);
int gp_canvas_on_mouse_move(GP_CanvasContext* ctx, float x, float y, float dx, float dy);
int gp_canvas_on_mouse_up(GP_CanvasContext* ctx, float x, float y, float dx, float dy, int button);
// Scroll events: `scroll` positive for up, negative for down.
int gp_canvas_on_mouse_scroll(GP_CanvasContext* ctx, float x, float y, float dx, float dy, float scroll);
// Deliver a keyboard event to the canvas host. Returns 1 if handled.
int gp_canvas_on_key(GP_CanvasContext* ctx, int key, int scancode, int action, int mods);
// Set/get the canvas viewport offset (world origin mapped to local (0,0)).
// Offsets are expressed in canvas-space pixels and can be updated by user
// panning or by a containing table.
int gp_canvas_set_offset(GP_CanvasContext* ctx, int offx, int offy);
int gp_canvas_get_offset(GP_CanvasContext* ctx, int* out_offx, int* out_offy);
// Query whether content overflows the viewport (useful for drawing scrollbars).
int gp_canvas_get_scroll_flags(GP_CanvasContext* ctx, int* out_has_h, int* out_has_v);

// Set the canvas logical pixel size (background/work area). This updates
// internal viewport sizes used by rasterization and scroll calculations.
int gp_canvas_set_size(GP_CanvasContext* ctx, int width, int height);

// Create/destroy a canvas-owned table attached to module. The created table
// will be owned by the canvas and destroyed when detached or when canvas
// is destroyed. Returns 1 on success.
int gp_canvas_create_table(GP_CanvasContext* ctx, int module_idx);
int gp_canvas_destroy_table(GP_CanvasContext* ctx, int module_idx);

// Step the canvas' internal simulation by dt seconds. Returns 1 on success.
int gp_canvas_step(GP_CanvasContext* ctx, float dt);

// Return the RopeSim pointer used by the canvas' root/container table (or NULL if none).
// Useful to attach table contexts to the shared simulator so tables defer
// simulation to the root table. The returned pointer is an opaque `RopeSim*`.
void* gp_canvas_get_rope_sim(GP_CanvasContext* ctx);

// Create a custom overlay rectangle with two LED contact positions. Returns
// two 64-bit keys in `out_key_a` and `out_key_b` that can be used with
// canvas/edge APIs. Keys follow internal sentinel encoding and are valid for
// the lifetime of the canvas. Returns 1 on success.
int gp_canvas_create_overlay_with_leds(GP_CanvasContext* ctx, float x1, float y1, float x2, float y2, unsigned long long* out_key_a, unsigned long long* out_key_b);
int gp_canvas_set_overlay_meta(GP_CanvasContext* ctx, unsigned long long key_a, unsigned long long key_b, GP_TableContext* table, void* meta_mg);
// Register an overlay using persisted overlay keys during table deserialization.
// Creates the overlay entry only if the keys are not already known. Returns 1 on success.
// Register an overlay using persisted overlay keys during table deserialization.
// `port_uuid_a` and `port_uuid_b` are optional persisted per-port UUIDs; pass 0 to let canvas
// assign deterministic UUIDs. Creates the overlay entry only if the keys are not already known.
// Returns 1 on success.
int gp_canvas_register_table_overlay(GP_CanvasContext* ctx, unsigned long long key_a, unsigned long long key_b, uint64_t port_uuid_a, uint64_t port_uuid_b);

// Retrieve the persisted/runtime port UUIDs for an overlay identified by keys.
// Returns 1 and fills out_port_a/out_port_b on success, 0 on failure.
int gp_canvas_get_overlay_port_uuids(GP_CanvasContext* ctx, unsigned long long key_a, unsigned long long key_b, uint64_t* out_port_a, uint64_t* out_port_b);

// Attach an existing rope index to an overlay by supplying the two overlay
// keys and the rope index. This creates a canvas edge record that links the
// rope to the overlay so rendering and interaction bind to the rope.
int gp_canvas_attach_rope_to_overlay(GP_CanvasContext* ctx, unsigned long long key_a, unsigned long long key_b, int rope_idx);
// Variant that attaches an overlay's rope to a specified module/contact indices
// instead of the canvas root module. Returns edge index on success or -1.
int gp_canvas_attach_rope_to_overlay_with_module(GP_CanvasContext* ctx, unsigned long long key_a, unsigned long long key_b, int rope_idx, int module_idx, int a_contact_idx, int b_contact_idx);
// Resolve an overlay key into canvas pixel coordinates. Returns 1 on success.
int gp_canvas_resolve_overlay_key(GP_CanvasContext* ctx, unsigned long long key, int* out_x, int* out_y);
// Resolve a canonical root key (packed as (module<<32)|(col<<16)|led) to
// overlay pixel coords if it maps to an overlay. Returns 1 on success.
int gp_canvas_resolve_canonical_key(GP_CanvasContext* ctx, unsigned long long key, int* out_x, int* out_y);

// Attach a `GP_TableContext` to a canvas module so the canvas will render
// the table inside the module rectangle and forward clicks. `take_ownership`
// indicates whether the canvas should destroy the table when the module is
// removed/destroyed (1 = canvas destroys it). Returns 1 on success.
int gp_canvas_attach_table(GP_CanvasContext* ctx, int module_idx, GP_TableContext* table, int take_ownership);

// Detach any table from the given module. If the canvas owned the table,
// it will destroy it. Returns 1 on success.
int gp_canvas_detach_table(GP_CanvasContext* ctx, int module_idx);

// Attach a pointer to a module-frame LED cell (send row if is_send!=0, receive row otherwise).
int gp_canvas_set_module_frame_ptr(GP_CanvasContext* ctx, int module_idx, int is_send, int led_idx, void* ptr);

// Module and module-port UUID management
// Generate a new stable module UUID (monotonic 64-bit id).
uint64_t gp_canvas_generate_module_uuid(GP_CanvasContext* ctx);
// Central ID generator (pluggable): returns a non-zero stable id.
uint64_t gp_canvas_generate_id(GP_CanvasContext* ctx, uint64_t hint);
// Set/get a module's stable UUID. Returns 1 on success for setter.
int gp_canvas_set_module_uuid(GP_CanvasContext* ctx, int module_idx, uint64_t module_uuid);
uint64_t gp_canvas_get_module_uuid(GP_CanvasContext* ctx, int module_idx);
// Set/get a module frame port's stable UUID. `is_send` selects send(1)/receive(0),
// `col` selects left(0)/right(1) column pair, `led_idx` selects the LED index.
int gp_canvas_set_module_frame_port_uuid(GP_CanvasContext* ctx, int module_idx, int is_send, int col, int led_idx, uint64_t port_uuid);
uint64_t gp_canvas_get_module_frame_port_uuid(GP_CanvasContext* ctx, int module_idx, int is_send, int col, int led_idx);

// Register module UUID and frame-port UUIDs using a table pointer. These
// convenience APIs let table deserialization register persistent UUIDs
// without needing the module index. The canvas will map the table to its
// module_idx internally. Returns 1 on success, 0 on failure.
int gp_canvas_register_table_module_uuid(GP_CanvasContext* ctx, GP_TableContext* table, uint64_t module_uuid);
int gp_canvas_register_table_frame_port_uuid(GP_CanvasContext* ctx, GP_TableContext* table, int row, int col_idx, uint64_t port_uuid);

// Rope metadata exposure: query rope ownership and participation for dense ring/spring networks.
int gp_canvas_get_rope_meta_owner(GP_CanvasContext* ctx, GP_TableContext* table, int rope_idx);
int gp_canvas_get_rope_rings(GP_CanvasContext* ctx, GP_TableContext* table, int rope_idx, int* out_ids, int max_count);
int gp_canvas_get_rope_meta_groups(GP_CanvasContext* ctx, GP_TableContext* table, int rope_idx, int* out_groups, int max_count);
int gp_canvas_is_meta_rope(GP_CanvasContext* ctx, GP_TableContext* table, int rope_idx);
int gp_canvas_get_meta_vertices(GP_CanvasContext* ctx, GP_TableContext* table, int rope_idx, int* out_vertices, int max_count);

// Create a retained pending-action object from a numeric action id. Returned
// pointer has ownership transferred to caller and may be bound into a
// module frame with `gp_canvas_bind_action_ptr_to_module_port` or freed via
// `gp_canvas_free_pending_action`.
void* gp_canvas_create_action_from_enum(GP_CanvasContext* ctx, int32_t action_id);

// Bind a previously-created pending-action pointer into a specific module
// frame port. `is_send` selects send(1)/receive(0); `col` selects the
// left(0)/right(1) pair within the frame; `led_idx` is 0..kModuleExtraLedCount-1.
// The function will set the internal frame ptr and update the module's
// attached table LED glow so the port appears illuminated. Returns 1 on
// success.
int gp_canvas_bind_action_ptr_to_module_port(GP_CanvasContext* ctx, int module_idx, int is_send, int col, int led_idx, void* pending_ptr);

// Convenience: create-and-bind in one call from an action enum id.
int gp_canvas_bind_action_enum_to_module_port(GP_CanvasContext* ctx, int module_idx, int is_send, int col, int led_idx, int32_t action_id);

// Convenience: unbind a previously-bound action from a module frame port.
// This will clear the frame ptr, clear visual LED glow, remove the
// action->port binding, and free the pending action pointer. Returns 1
// on success, 0 on failure.
int gp_canvas_unbind_action_from_module_port(GP_CanvasContext* ctx, int module_idx, int is_send, int col, int led_idx);
// Pop a managed event payload previously stashed by the manager for a
// delivered action. Returns the payload pointer or NULL if none. Caller
// is responsible for freeing the returned payload as appropriate for its
// type.
void* gp_canvas_pop_managed_event_payload(GP_CanvasContext* ctx, int module_idx, int led_idx);
// Stash a managed event payload pointer for later popping by the tool.
int gp_canvas_stash_managed_event_payload(GP_CanvasContext* ctx, int module_idx, int led_idx, void* payload);
// Per-module background rasterizer hook. Returns 1 on success.
int gp_canvas_set_module_bg_callback(GP_CanvasContext* ctx, int module_idx, GP_CanvasModuleBgFn cb, void* user);
int gp_canvas_clear_module_bg_callback(GP_CanvasContext* ctx, int module_idx);
// Background mode: 0=none (solid fill), 1=raytrace (input LED lights).
int gp_canvas_set_module_bg_mode(GP_CanvasContext* ctx, int module_idx, int mode);
int gp_canvas_get_module_bg_mode(GP_CanvasContext* ctx, int module_idx, int* out_mode);
// Configure raytrace parameters for background mode.
int gp_canvas_set_module_raytrace_params(GP_CanvasContext* ctx, int module_idx, int ray_count, int max_reflections, float blur_sigma);
// Configure raytrace tuning: bounce decay, distance decay, and exposure scale.
int gp_canvas_set_module_raytrace_tuning(GP_CanvasContext* ctx, int module_idx, float bounce_decay, float distance_decay, float exposure);
// Set per-module table alpha (0..1). When raytrace mode is active, alpha is
// clamped to `raytrace_alpha`.
int gp_canvas_set_module_table_alpha(GP_CanvasContext* ctx, int module_idx, float alpha, float raytrace_alpha);

// Standard mouse event action ids usable by bindings/subscribers.
#define CANVAS_ACT_MOUSE_DOWN (3001)
#define CANVAS_ACT_MOUSE_UP (3002)
#define CANVAS_ACT_MOUSE_MOVE (3003)
// Mouse wheel actions
#define CANVAS_ACT_MOUSE_SCROLL_UP (3004)
#define CANVAS_ACT_MOUSE_SCROLL_DOWN (3005)

// Action IDs for module-side event bindings: text render and tensor allocation
// These are published as pointer-mode EventPayloads into the root table FIFO
// so module-frame bound ports can observe and react to them.
#define CANVAS_ACT_TEXT_RENDER (3101)
#define CANVAS_ACT_TENSOR_ALLOC (3102)

// Register a host window pointer with the canvas so the canvas can retain
// references to windows it will handle (opaque pointer). Returns 1 on success.
int gp_canvas_register_window(GP_CanvasContext* ctx, void* window_ptr);
int gp_canvas_unregister_window(GP_CanvasContext* ctx, void* window_ptr);

// Get the backing graph node id for a registered window pointer, or -1 if none.
int gp_canvas_get_window_node_id(GP_CanvasContext* ctx, void* window_ptr);

// Typed-edge API: add an edge with an explicit type id (0 == untyped/wildcard).
int gp_canvas_add_edge_with_type(GP_CanvasContext* ctx, const GP_CanvasEdgeDesc* desc, int type_id);

// Set the module's supported input/output type lists. Each type id is an int.
int gp_canvas_set_module_io_types(GP_CanvasContext* ctx, int module_idx, const int* input_types, int input_count, const int* output_types, int output_count);

// Return the backing graph node id for a module, or -1 if none.
int gp_canvas_get_module_node_id(GP_CanvasContext* ctx, int module_idx);

// Clear and remove all overlays and meta-groups. This will unbind overlays
// from any tables and destroy table-side meta-groups (and their rings).
// Returns 1 on success.
int gp_canvas_clear_meta_and_overlays(GP_CanvasContext* ctx);
// Register per-table rope UIDs with the canvas so the canvas can map
// persistent rope UIDs to runtime rope indices for deterministic restore.
int gp_canvas_register_table_rope_ids_from_array(GP_CanvasContext* ctx, GP_TableContext* table, const uint64_t* ids, int count);
// Add a meta-group vertex by persistent rope id. Returns 1 on success, 0 on failure.
int gp_canvas_table_meta_add_vertex_by_id(GP_CanvasContext* ctx, GP_TableContext* table, void* meta_mg, uint64_t rope_id, int vertex_idx);
// Resolve a persisted rope id to the runtime rope index for a given table.
// Returns >=0 rope index on success, -1 if not found.
int gp_canvas_resolve_rope_id_to_index(GP_CanvasContext* ctx, GP_TableContext* table, uint64_t rope_id);
// Mark the canvas rope-id map dirty so callers can trigger a re-resolution pass.
void gp_canvas_mark_rope_map_dirty(GP_CanvasContext* ctx);
// Find a persistent rope id for a runtime rope index on a table.
// Lookup order: local table -> container/root -> breadth-first across modules.
uint64_t gp_canvas_find_persistent_rope_id(GP_CanvasContext* ctx, GP_TableContext* table, int sim_idx);
// Generate a new stable port UUID for the canvas (monotonic 64-bit id).
uint64_t gp_canvas_generate_port_uuid(GP_CanvasContext* ctx);
// Query canvas for edges that belong entirely to `table` (both endpoints' modules map to the same table).
// If `out_edges` or `out_rope_ids` are NULL, the function returns the required count without writing.
// Returns the number of entries written (or required).
int gp_canvas_get_table_local_edges_and_rope_ids(GP_CanvasContext* ctx, GP_TableContext* table, GP_CanvasEdgeDesc* out_edges, uint64_t* out_rope_ids, int max_entries);
// Persist/restore canvas state to a simple text file. Returns 1 on success.
int gp_canvas_save_to_file(GP_CanvasContext* ctx, const char* path);
int gp_canvas_load_from_file(GP_CanvasContext* ctx, const char* path);
// Export a module library serialization with tool/source references. Returns 1 on success.
int gp_canvas_export_module_library(GP_CanvasContext* ctx, const char* path);

// Actualize the current canvas into an output root directory (create
// module_library/{serialized,source} placeholders and tool sources).
// Returns 1 on success.
int gp_canvas_actualize_to_root(GP_CanvasContext* ctx, const char* output_root);
// Export a single module's serialized blob to the given output root.
// If output_root is NULL, uses the default `module_library` root.
int gp_canvas_export_module_to_root(GP_CanvasContext* ctx, int module_idx, const char* output_root);

// Templates directory: set a global library directory for templates (used by
// table template APIs if they are invoked with dir==NULL).
int gp_canvas_set_templates_dir(const char* dir);
int gp_canvas_get_templates_dir(char* out_buf, int out_len);

// Optional containment: attach a GP_TableContext that will act as the root table
// for canvas actions and will mirror the canvas' scroll fractions (both axes).
// Ownership is controlled by `take_ownership`.
int gp_canvas_set_container_table(GP_CanvasContext* ctx, GP_TableContext* table, int take_ownership);
GP_TableContext* gp_canvas_get_container_table(GP_CanvasContext* ctx);

// Per-canvas autosave: set autosave path and interval in seconds. If path is
// NULL or empty, autosave is disabled. Autosave will trigger during
// `gp_canvas_step` when enough time has accumulated.
int gp_canvas_set_autosave(GP_CanvasContext* ctx, const char* path, double interval_s);
int gp_canvas_get_autosave(GP_CanvasContext* ctx, char* out_path, int out_len, double* out_interval_s);
// Control autosave policy: 0=disabled,1=timer,2=action-listener
int gp_canvas_set_autosave_policy(GP_CanvasContext* ctx, int policy);
int gp_canvas_get_autosave_policy(GP_CanvasContext* ctx, int* out_policy);
// Update the in-memory last-save signature from current canvas state.
int gp_canvas_update_last_save_signature(GP_CanvasContext* ctx);
// Configure action-list autosave: whitelist actions and timing (delay before acting, cooldown between saves)
int gp_canvas_set_autosave_action_whitelist(GP_CanvasContext* ctx, const int* action_ids, int count);
int gp_canvas_set_autosave_action_timing(GP_CanvasContext* ctx, double delay_s, double cooldown_s);

// Table tool number control (tool menu). Value is clamped to 0..99.
int gp_canvas_set_table_tool_number(GP_CanvasContext* ctx, int value);
int gp_canvas_get_table_tool_number(GP_CanvasContext* ctx, int* out_value);

// Auto-bind helpers for event-driven tools/plugins. They bind up to
// `max_ports` receive-side frame ports that are currently unbound.
// Returns how many ports were bound.
int gp_canvas_autobind_actions(GP_CanvasContext* ctx, int module_idx, const int32_t* action_ids, int action_count, int max_ports);
int gp_canvas_autobind_mouse_ports(GP_CanvasContext* ctx, int module_idx, int max_ports);
int gp_canvas_autobind_keyboard_ports(GP_CanvasContext* ctx, int module_idx, int max_ports);

// Thread manager / scheduler mode (0 = free-spinning, 1 = scheduled).
// Current behavior: gp_canvas_step always waits for scheduled ticks to finish
// before returning, so raster/input can remain single-threaded and safe.
int gp_canvas_set_thread_manager_mode(GP_CanvasContext* ctx, int mode);
int gp_canvas_get_thread_manager_mode(GP_CanvasContext* ctx, int* out_mode);
int gp_canvas_get_thread_manager_paused(GP_CanvasContext* ctx, int* out_paused);
int gp_canvas_set_thread_manager_paused(GP_CanvasContext* ctx, int paused);

#ifdef __cplusplus
}
#endif

#endif // NODUS_CANVAS_ABI_H
