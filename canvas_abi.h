// Minimal canvas API for placing modules and drawing droopy rope edges between contacts
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GP_CanvasContext GP_CanvasContext;
typedef struct GP_TableContext GP_TableContext; // forward from table_abi

// Module background rasterizer hook.
typedef void(*GP_CanvasModuleBgFn)(void* user, int module_idx, int width, int height, uint8_t* out_rgba, int32_t out_pitch);

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

// Forward a mouse click (canvas-local coords). Returns 1 if handled.
int gp_canvas_on_click(GP_CanvasContext* ctx, int x, int y);

// Mouse drag handlers to support click-and-dragging modules.
int gp_canvas_on_mouse_down(GP_CanvasContext* ctx, int x, int y);
int gp_canvas_on_mouse_move(GP_CanvasContext* ctx, int x, int y);
int gp_canvas_on_mouse_up(GP_CanvasContext* ctx, int x, int y);
// Deliver a keyboard event to the canvas host. Returns 1 if handled.
int gp_canvas_on_key(GP_CanvasContext* ctx, int key, int scancode, int action, int mods);
// Set/get the canvas viewport offset (world origin mapped to local (0,0)).
// Offsets are expressed in canvas-space pixels and can be updated by user
// panning or by a containing table.
int gp_canvas_set_offset(GP_CanvasContext* ctx, int offx, int offy);
int gp_canvas_get_offset(GP_CanvasContext* ctx, int* out_offx, int* out_offy);
// Query whether content overflows the viewport (useful for drawing scrollbars).
int gp_canvas_get_scroll_flags(GP_CanvasContext* ctx, int* out_has_h, int* out_has_v);

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

// Table tool number control (tool menu). Value is clamped to 0..99.
int gp_canvas_set_table_tool_number(GP_CanvasContext* ctx, int value);
int gp_canvas_get_table_tool_number(GP_CanvasContext* ctx, int* out_value);

// Thread manager / scheduler mode (0 = free-spinning, 1 = scheduled).
// Current behavior: gp_canvas_step always waits for scheduled ticks to finish
// before returning, so raster/input can remain single-threaded and safe.
int gp_canvas_set_thread_manager_mode(GP_CanvasContext* ctx, int mode);
int gp_canvas_get_thread_manager_mode(GP_CanvasContext* ctx, int* out_mode);

#ifdef __cplusplus
}
#endif
