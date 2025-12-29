#pragma once

// Generic C ABI for rasterizing dense tables with hierarchical rows, LEDs, axis bars,
// timers, optional waveforms, and simple expand/collapse affordances.
// Text is carried through the API (label + per-cell text) but this minimal raster
// does not render text; callers can overlay text separately using their text system.

#include <stdint.h>
// Forward-declare LassoConfig (defined in lasso_config.h). We avoid including
// the header here to keep this file friendly to C/C++ consumers and
// to prevent requiring the header search path at compile time for all units.
typedef struct LassoConfig LassoConfig;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RopeSim RopeSim; // forward-declare rope sim type for attachment API
struct GP_StageContext; // forward declare stage for port bindings

struct GP_MenuWaveform; // from menu_waveform_abi.h

// Cell kinds
typedef enum GP_TableCellKind {
    GP_TABLE_CELL_TEXT = 0,
    GP_TABLE_CELL_LEDS = 1,
    GP_TABLE_CELL_AXIS = 2,
    GP_TABLE_CELL_TIMERS = 3,
    GP_TABLE_CELL_WAVE = 4,
    GP_TABLE_CELL_CALIB = 5, // compact calibration strip (INV/TRIM/CAP/DED/RST)
    GP_TABLE_CELL_LEDS_ARG = 6,    // signal arg LEDs (required vs linked)
    GP_TABLE_CELL_LEDS_TABLE = 7,  // stacked LED strips encoded inside one cell
    GP_TABLE_CELL_COUNTER = 8,     // minus/number/plus control
    GP_TABLE_CELL_SCROLL = 9,      // vertical scrollbar (arrows + track + thumb)
    GP_TABLE_CELL_IMAGE = 10,      // RGBA image content
} GP_TableCellKind;

// Hitbox parts for interactive regions.
typedef enum GP_TableHitPart {
    GP_TABLE_HIT_CELL = 0,
    GP_TABLE_HIT_EXPAND = 1,
    GP_TABLE_HIT_LED = 2,
    GP_TABLE_HIT_AXIS = 3,
    GP_TABLE_HIT_CALIB_SLOT = 4,
    GP_TABLE_HIT_SCROLL_UP = 5,
    GP_TABLE_HIT_SCROLL_DOWN = 6,
    GP_TABLE_HIT_SCROLL_THUMB = 7,
    GP_TABLE_HIT_LED_TABLE = 8,
    GP_TABLE_HIT_LED_ARG = 9,
    GP_TABLE_HIT_COUNTER_DEC = 10,
    GP_TABLE_HIT_COUNTER_INC = 11,
} GP_TableHitPart;

// Hitbox emitted per interactive sub-element (pixel coords in table-local space).
typedef struct GP_TableHitBox {
    int32_t x0, y0, x1, y1; // inclusive-exclusive rectangle
    int32_t row_idx;        // row in input array
    int32_t col_idx;        // column in input array
    int32_t cell_kind;      // GP_TableCellKind
    int32_t part;           // GP_TableHitPart
    int32_t aux0;           // e.g., led index, slot index
    int32_t aux1;           // spare
    uint32_t flags;         // optional
} GP_TableHitBox;

// Action hooks for click dispatch. Use GP_TABLE_ACTION_ANY (-1) for wildcards.
#define GP_TABLE_ACTION_ANY (-1)
#define GP_TABLE_ACTION_EXPAND_TOGGLE (1000)
#define GP_TABLE_ACTION_SCROLL_UP (1001)
#define GP_TABLE_ACTION_SCROLL_DOWN (1002)
#define GP_TABLE_ACTION_LED_TOGGLE (1003)

typedef struct GP_TableAction {
    int32_t row_idx;    // row index or GP_TABLE_ACTION_ANY
    int32_t col_idx;    // column index or GP_TABLE_ACTION_ANY
    int32_t part;       // GP_TableHitPart or GP_TABLE_ACTION_ANY
    int32_t aux0;       // aux0 match or GP_TABLE_ACTION_ANY
    int32_t action_id;  // caller-defined action id
} GP_TableAction;

typedef void(*GP_TableActionFn)(void* user, int32_t action_id, const GP_TableHitBox* hit);

// Keyboard callback signature (mirrors common window callbacks: key, scancode, action, mods)
typedef void(*GP_TableKeyFn)(void* user, int32_t key, int32_t scancode, int32_t action, int32_t mods);

// Row kinds
typedef enum GP_TableRowKind {
    GP_TABLE_ROW_HEADER = 0,
    GP_TABLE_ROW_DEVICE = 1,
    GP_TABLE_ROW_AXIS = 2,
    GP_TABLE_ROW_BUTTON = 3,
    GP_TABLE_ROW_NOTE = 4,
} GP_TableRowKind;

// Column meta
typedef struct GP_TableColumn {
    int32_t kind;      // GP_TableCellKind
    int32_t width_px;  // fixed width; 0 => auto (not implemented, treated as fixed)
    int32_t align;     // 0=left,1=center,2=right (used for text when rendered elsewhere)
} GP_TableColumn;

// One cell
typedef struct GP_TableCell {
    int32_t kind;          // GP_TableCellKind
    char text[96];         // for TEXT; AXIS: optional "cap_min cap_max trim deadzone"
    uint32_t flags;        // for LEDS; AXIS: bit0/bit1 mark seen_min/seen_max
    float value;           // for AXIS (current)
    float hold_s;          // for TIMERS; AXIS: observed min
    float last_s;          // for TIMERS; AXIS: observed max
    const struct GP_MenuWaveform* wave; // for WAVE (nullable)
    const struct GP_TableImage* image;  // for IMAGE (nullable)
    int32_t reserved0;
} GP_TableCell;

// External image descriptor used by GP_TABLE_CELL_IMAGE.
typedef struct GP_TableImage {
    const uint8_t* rgba;   // pointer to tightly-packed RGBA8 pixels
    int32_t width_px;      // image width in pixels
    int32_t height_px;     // image height in pixels
    int32_t pitch_bytes;   // bytes per row (>= width_px*4)
} GP_TableImage;

// One row
typedef struct GP_TableRow {
    int32_t kind;          // GP_TableRowKind
    int32_t depth;         // indentation level (0=root)
    int32_t expanded;      // non-zero => expanded; renderer draws +/- box
    int32_t selected;      // highlight row if non-zero
    char label[96];        // primary label (not rendered in this minimal raster)
    GP_TableCell cells[8]; // fixed cell array
    int32_t cell_count;    // how many cells are valid
    int32_t reserved0;
} GP_TableRow;

// Style
typedef struct GP_TableStyle {
    int32_t width_px;
    int32_t row_h_px;
    int32_t indent_px;
    int32_t expand_w_px;
    int32_t name_w_px;     // label gutter width (used for +/- and potential text overlay)

    uint8_t bg_rgba[4];
    uint8_t bg_sel_rgba[4];
    uint8_t hdr_rgba[4];
    uint8_t text_rgba[4];
    uint8_t text_hdr_rgba[4];
    uint8_t led_on_rgba[4];
    uint8_t led_off_rgba[4];
    uint8_t led_edge_rgba[4];
    uint8_t axis_bg_rgba[4];
    uint8_t axis_tick_rgba[4];
    uint8_t axis_val_rgba[4];
    uint8_t timer_rgba[4];
    uint8_t wave_bg_rgba[4];
    uint8_t wave_fg_rgba[4];

    uint32_t led_mask[9];
    uint32_t led_edge_mask;
    // Cable FIFO lighting mode: when enabled, rope glow can be driven by
    // FIFO read/write head movement.
    int32_t cable_fifo_light_mode;
    float cable_fifo_friction_half_life; // seconds; <=0 uses internal default
    float cable_fifo_friction_gain;      // scalar gain for friction glow
    int32_t cable_fifo_friction_regions; // quantization regions for head movement
    float cable_fifo_friction_tint;      // hue shift intensity from head angle (0..1)
} GP_TableStyle;

// Geometry returned to the caller.
typedef struct GP_TableGeom {
    int32_t width_px;
    int32_t height_px;
    int32_t col_x0[8];
    int32_t col_w[8];
} GP_TableGeom;

// Optional transient render state supplied by callers that want the C renderer
// to apply highlights or mouse-based overlays at render time. Any field set
// to -1/0 acts as 'none'. Color is RGBA (0-255); if all zero, a default is used.
typedef struct GP_TableRenderState {
    int32_t mouse_x;    // table-local pixel x or -1 if unused
    int32_t mouse_y;    // table-local pixel y or -1 if unused
    int32_t highlight_row; // row index or -1
    int32_t highlight_col; // col index or -1
    int32_t highlight_part; // GP_TableHitPart or -1
    int32_t highlight_aux0; // aux0 (e.g., led index or slot) or -1
    uint8_t highlight_color[4]; // RGBA; if all zero, default highlight color used
} GP_TableRenderState;

// Opaque stateful table context. Holds style/columns/rows so callers can update incrementally
// and render repeatedly without repassing everything.
typedef struct GP_TableContext GP_TableContext;
typedef struct GP_MetaGroup GP_MetaGroup;

// Ring topology selection for meta-groups
#define GP_META_RING_RIBBON 0 // simple chain between consecutive members
#define GP_META_RING_CLOSED 1 // chain + close last->first
#define GP_META_RING_DENSE 2  // closed + additional cross-links for small groups

// Calculate output size; returns 1 on success.
int32_t gp_table_calc_size(const GP_TableStyle* style, int32_t row_count, int32_t col_count, GP_TableGeom* out_geom);

// Rasterize rows into RGBA8 buffer; buffer must be >= width*height*4 bytes as reported by calc_size.
// Returns 1 on success.
int32_t gp_table_raster_rgba(
    const GP_TableRow* rows,
    int32_t row_count,
    const GP_TableColumn* cols,
    int32_t col_count,
    const GP_TableStyle* style,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom);

// Rasterize and optionally emit hitboxes; hitboxes_out may be NULL to skip.
int32_t gp_table_raster_rgba_with_hits(
    const GP_TableRow* rows,
    int32_t row_count,
    const GP_TableColumn* cols,
    int32_t col_count,
    const GP_TableStyle* style,
    const GP_TableRenderState* render_state,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom,
    GP_TableHitBox* hitboxes_out,
    int32_t hitboxes_cap,
    int32_t* hitboxes_written);

// Stateful context helpers --------------------------------------------------

// Create/destroy a table context. Style may be NULL to use defaults.
GP_TableContext* gp_table_create(const GP_TableStyle* style);
void gp_table_destroy(GP_TableContext* ctx);

// Update style/columns/rows on an existing context. Any call will recompute
// geometry. Passing NULL for style leaves it unchanged. Returns 1 on success.
int32_t gp_table_set_style(GP_TableContext* ctx, const GP_TableStyle* style);
int32_t gp_table_set_columns(GP_TableContext* ctx, const GP_TableColumn* cols, int32_t col_count);
int32_t gp_table_set_rows(GP_TableContext* ctx, const GP_TableRow* rows, int32_t row_count);

// Retrieve current geometry (width/height/columns). Returns 1 on success.
int32_t gp_table_get_geom(const GP_TableContext* ctx, GP_TableGeom* out_geom);

// Render using stored rows/columns/style. Hitboxes are optional (may be NULL).
int32_t gp_table_render_rgba_with_hits(
    GP_TableContext* ctx,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom,
    GP_TableHitBox* hitboxes_out,
    int32_t hitboxes_cap,
    int32_t* hitboxes_written);

// Stateful render variant: supply a transient GP_TableRenderState to have the
// C raster apply highlights/overlays based on instantaneous state (mouse,
// selection, focused LED, etc.). Backwards-compatible: callers can pass NULL.
int32_t gp_table_render_rgba_with_state(
    GP_TableContext* ctx,
    const GP_TableRenderState* render_state,
    uint8_t* out_rgba,
    int32_t out_len_bytes,
    GP_TableGeom* out_geom,
    GP_TableHitBox* hitboxes_out,
    int32_t hitboxes_cap,
    int32_t* hitboxes_written);

// Interaction and state helpers ------------------------------------------------

// Get projected 2D rope vertices for the specified rope index in table-local
// coordinates. Projection uses the table's internal cable tilt (orthographic
// tilt) so callers can test against the same coordinates the renderer uses.
// `out_xy` should be a float array sized >= 2 * vertex_count. Returns vertex
// count on success, or 0 on failure.
int32_t gp_table_get_projected_rope_vertices(GP_TableContext* ctx, int32_t rope_idx, float* out_xy, int32_t max_count);

// Meta-group enumeration/accessors
int32_t gp_table_get_meta_group_count(const GP_TableContext* ctx);
GP_MetaGroup* gp_table_get_meta_group(GP_TableContext* ctx, int32_t idx);
int32_t gp_table_meta_get_vertex(const GP_TableContext* ctx, GP_MetaGroup* mg, int32_t idx, int32_t* out_rope_idx, int32_t* out_vertex_idx);
int32_t gp_table_meta_get_vertex_u(const GP_TableContext* ctx, GP_MetaGroup* mg, int32_t idx, float* out_u);
int32_t gp_table_meta_set_vertex_u(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t rope_idx, int32_t vertex_idx, float u);
int32_t gp_table_meta_get_vertex_count(GP_TableContext* ctx, GP_MetaGroup* mg);
int32_t gp_table_meta_get_lasso_config(GP_TableContext* ctx, GP_MetaGroup* mg, LassoConfig* out_cfg);

// Debug helper: print a meta-group's vertices and lasso config with a tag
int32_t gp_table_debug_dump_meta_group(GP_TableContext* ctx, GP_MetaGroup* mg, const char* tag);
int32_t gp_table_meta_get_subgroup_flags(GP_TableContext* ctx, GP_MetaGroup* mg, uint32_t* out_flags);
// If the meta-group has a dangling widget rope, return its rope index and vertex index.
int32_t gp_table_meta_get_dangling_rope_info(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_rope_idx, int32_t* out_vertex_idx);
// If the meta-group is registered with a RopeSim, return its sim group index.
// Returns 1 on success and sets out_sim_idx, 0 on failure.
int32_t gp_table_meta_get_sim_group_index(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_sim_idx);

// Widget helpers
int32_t gp_table_get_widget_position(GP_TableContext* ctx, int32_t widget_id, float* out_xyz);
int32_t gp_table_meta_get_dangling_widget_id(const GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_widget_id);

// Set or clear the table's prospective rope index used for rendering a
// temporary rope while the user is interacting. Pass -1 to clear.
int32_t gp_table_set_prospective_rope_index(GP_TableContext* ctx, int32_t rope_idx);

// Insert a vertex into the simulator rope at segment `seg_index` and param t (0..1).
// Returns new vertex id >=0 on success or -1 on failure.
int32_t gp_table_rope_insert_vertex(GP_TableContext* ctx, int32_t rope_idx, int32_t seg_index, float t);

// Create/destroy a sliding ring attached to a rope. `u` is parametric 0..1
// along the rope. Returns ring id >=0 on success or -1 on failure.
int32_t gp_table_create_ring(GP_TableContext* ctx, int32_t rope_idx, float u);
// Create a ring by persisted rope UID. Resolves the UID to a runtime rope index
// using the canvas mapping and table-local fallback, then calls
// `gp_table_create_ring`. Returns ring id >=0 on success or -1 on failure.
int32_t gp_table_create_ring_by_id(GP_TableContext* ctx, uint64_t rope_id, float u);
int32_t gp_table_destroy_ring(GP_TableContext* ctx, int32_t ring_id);
// Move a ring towards `target_u` at `speed` (parametric units per second).
int32_t gp_table_set_ring_target(GP_TableContext* ctx, int32_t ring_id, float target_u, float speed);
// Query current ring u parameter. Returns 1 on success.
int32_t gp_table_get_ring_u(GP_TableContext* ctx, int32_t ring_id, float* out_u);

// Register a ring id created in the simulator with the table so it will be
// enumerated and rendered like an edge. Returns ring entry index or -1 on error.
int32_t gp_table_register_ring_edge(GP_TableContext* ctx, int32_t ring_id, unsigned long long ring_key);


// Deliver a click (table-local pixel coords) to the context. If a hit was
// found and processed, fills `out_hit` (if non-null) with the hit info (row/col
// are the original indices) and returns 1. Returns 0 if nothing was hit.
int32_t gp_table_on_click(GP_TableContext* ctx, int32_t x, int32_t y, GP_TableHitBox* out_hit);

// Register a click-action table and callback. Actions are matched on row/col/part/aux0,
// using GP_TABLE_ACTION_ANY (-1) as a wildcard. If a match is found, the callback
// is invoked with the action_id and hitbox. Returns 1 on success.
int32_t gp_table_set_actions(GP_TableContext* ctx, const GP_TableAction* actions, int32_t count);
int32_t gp_table_set_action_callback(GP_TableContext* ctx, GP_TableActionFn cb, void* user);
int32_t gp_table_clear_action_callback(GP_TableContext* ctx);
// Dispatch a pre-resolved hit through the action registry (no hit-testing).
// Returns 1 if any action matched, 0 otherwise.
int32_t gp_table_dispatch_hit(GP_TableContext* ctx, const GP_TableHitBox* hit);

// Keyboard event delivery to a table context. Returns 1 if delivered, 0 otherwise.
int32_t gp_table_on_key(GP_TableContext* ctx, int32_t key, int32_t scancode, int32_t action, int32_t mods);

// Register / clear a keyboard callback for a table context. Returns 1 on success.
int32_t gp_table_set_key_callback(GP_TableContext* ctx, GP_TableKeyFn cb, void* user);
int32_t gp_table_clear_key_callback(GP_TableContext* ctx);

// Set/get scroll position as a fraction 0..1 (0 -> top). Returns 1 on success.
int32_t gp_table_set_scroll_fraction(GP_TableContext* ctx, float frac);
int32_t gp_table_get_scroll_fraction(GP_TableContext* ctx, float* out_frac);
// Set/get scroll position for both axes. Horizontal fraction is stored on the
// context for embedding/containment scenarios (e.g. canvas viewports) and does
// not currently alter table rendering.
int32_t gp_table_set_scroll_fraction_xy(GP_TableContext* ctx, float frac_x, float frac_y);
int32_t gp_table_get_scroll_fraction_xy(GP_TableContext* ctx, float* out_frac_x, float* out_frac_y);

// Get row count and fetch a copy of a row by original index.
int32_t gp_table_get_row_count(const GP_TableContext* ctx);
int32_t gp_table_get_row(const GP_TableContext* ctx, int32_t idx, GP_TableRow* out_row);
// Accessors for persistent rope UIDs (safe C API so other modules may
// register per-table UIDs with the canvas without exposing internal types).
int gp_table_get_rope_id_count(GP_TableContext* ctx);
int gp_table_get_rope_ids(GP_TableContext* ctx, uint64_t* out_ids, int cap);
// Resolve a persisted rope id to the attached RopeSim index for this table.
// Returns -1 if not present.
int gp_table_resolve_rope_id_to_sim_index(GP_TableContext* ctx, uint64_t id);
// Set the table's persistent rope id vector from an array. Returns 1 on success.
int gp_table_set_rope_ids_from_array(GP_TableContext* ctx, const uint64_t* ids, int count);

// Per-LED selection API: set/get selection state for a specific LED (row/col/index).
// Selected LEDs are rendered with a selection ring; selection is independent of
// the LED "on" state encoded in cell.flags.
int32_t gp_table_set_led_selected(GP_TableContext* ctx, int32_t row_idx, int32_t col_idx, int32_t led_index, int32_t selected);
int32_t gp_table_get_led_selected(const GP_TableContext* ctx, int32_t row_idx, int32_t col_idx, int32_t led_index);

// Optional glow values per LED (0..1). When set, the renderer will perform an
// additional glow pass around the LED using its on/off color and the supplied
// normalized strength. Glow is applied only when rendering through a table
// context with gp_table_render_rgba_with_state.
int32_t gp_table_set_led_glow(GP_TableContext* ctx, int32_t row_idx, int32_t col_idx, int32_t led_index, float glow01);
int32_t gp_table_get_led_glow(const GP_TableContext* ctx, int32_t row_idx, int32_t col_idx, int32_t led_index, float* out_glow01);
int32_t gp_table_clear_led_glow(GP_TableContext* ctx);
// Query LED state for a specific row/col/led index.
int32_t gp_table_get_led_info(const GP_TableContext* ctx, int32_t row_idx, int32_t col_idx, int32_t led_index,
    int32_t* out_on, int32_t* out_active, int32_t* out_is_input, int32_t* out_is_output);

// Fetch the current raw style from the table context.
int32_t gp_table_get_style(const GP_TableContext* ctx, GP_TableStyle* out_style);

// Edge list API: LEDs are identified by the same packed 64-bit key used
// by the selection APIs: (row<<32)|(col<<16)|led_index.
int32_t gp_table_add_edge(GP_TableContext* ctx, unsigned long long a, unsigned long long b);
int32_t gp_table_clear_edges(GP_TableContext* ctx);
int32_t gp_table_get_edge_count(const GP_TableContext* ctx);
int32_t gp_table_get_edge(const GP_TableContext* ctx, int32_t idx, unsigned long long* out_a, unsigned long long* out_b);

// Edge companion tensor FIFO -------------------------------------------------
// Each rope edge owns a companion FIFO that stores an n-D tensor per sample.
// A single writer appends samples; any number of subscribers can read through
// their own circular heads. When the buffer is full, `top_k` controls whether
// the oldest unread samples are advanced (drop old data, keep last top_k) or
// writes are rejected (top_k == 0).
typedef struct GP_TableEdgeTensorSpec {
    int32_t dims[8];    // up to 8 dimensions; each <= INT32_MAX, clamped to >=1
    int32_t dim_count;  // number of dims used in `dims`
    int32_t slots;      // how many tensor samples to keep in the FIFO (capacity)
    int32_t top_k;      // if >0, keep this many of the newest samples when overwriting; 0 = hold until readers consume
} GP_TableEdgeTensorSpec;

// Extended tensor spec: element size (bytes) and type/schema id must be
// supplied so FIFOs are byte-oriented and typed. `elem_size` is the size
// in bytes of a single element; `type_id` indexes the ValueTypeRegistry.
typedef struct GP_TableEdgeTensorSpecTyped {
    int32_t dims[8];
    int32_t dim_count;
    int32_t slots;
    int32_t top_k;
    int32_t elem_size; // bytes per element
    int32_t type_id;   // schema id (ValueTypeRegistry)
} GP_TableEdgeTensorSpecTyped;

typedef struct GP_TableEdgeBatchMetadata {
    uint64_t batch_id;
    double timestamp;
    // `sample_count` is the number of elements per sample; `stride` is
    // reserved for compatibility but callers should use `elem_size` on
    // the edge spec for byte size. `schema_id` is now `type_id`.
    uint32_t sample_count;
    uint32_t stride;
    uint32_t type_id;
} GP_TableEdgeBatchMetadata;

// Typed spec setters/getters. Old float-centric spec is removed in favor
// of typed tensor spec. Callers must provide `elem_size` and `type_id`.
int32_t gp_table_edge_set_tensor_spec(GP_TableContext* ctx, int32_t edge_idx, const GP_TableEdgeTensorSpecTyped* spec);
int32_t gp_table_edge_get_tensor_spec(GP_TableContext* ctx, int32_t edge_idx, GP_TableEdgeTensorSpecTyped* out_spec);

// Enable or disable simulator stepping for this table. When disabled the
// table's `RopeSim` will not be advanced by table-side ticks. Defaults to enabled (1).
int32_t gp_table_set_sim_enabled(GP_TableContext* ctx, int32_t enabled);
int32_t gp_table_get_sim_enabled(GP_TableContext* ctx, int32_t* out_enabled);
// Per-table frame-skip percent for rope sim (0..100). When >0 the table's
// rope sim will only be stepped on a subset of frames to reduce CPU load.
// Global rope-sim frame-skip percent (0..100). Applied across all table
// and canvas rope simulators; advance tick once per canvas frame via
// `gp_table_advance_global_sim_tick()` to produce deterministic stepping.
// Global rope-sim frame-skip count (0..). When set to N, each rope sim will
// be stepped once every (N+1) frames. 0 = step every frame.
int32_t gp_table_set_global_sim_frame_skip_count(int32_t count);
int32_t gp_table_get_global_sim_frame_skip_count(int32_t* out_count);
// Advance the global sim frame tick (called once per canvas frame).
void gp_table_advance_global_sim_tick();
// Query whether the table's rope sim should step this frame (honors per-table
// `sim_enabled` and the global sim frame-skip count). Returns 1 if should step, 0 otherwise.
int32_t gp_table_should_step_sim(GP_TableContext* ctx);
// Debug flags propagated from canvas (0 = default behavior).
int32_t gp_table_set_debug_flags(GP_TableContext* ctx, uint32_t flags);
int32_t gp_table_get_debug_flags(const GP_TableContext* ctx, uint32_t* out_flags);
// Override cable segment count used for rope simulation and rendering.
int32_t gp_table_set_cable_segments(GP_TableContext* ctx, int32_t segments);
int32_t gp_table_edge_subscribe(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key);
// Subscribe with explicit start policy. If `start_at_head` is non-zero, the subscriber
// begins at the current write head (new samples only). If zero, it begins at the
// earliest sample still addressable within the ring capacity (best-effort).
int32_t gp_table_edge_subscribe_ex(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, int32_t start_at_head);
int32_t gp_table_edge_unsubscribe(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key);
// Typed byte-oriented FIFO APIs. Samples are passed as raw bytes and lengths
// are in bytes. Callers must ensure the provided byte length matches the
// per-edge `elem_size * sample_count` configured in the tensor spec.
// Returns 1 on success; `out_dropped` is set to 1 when old unread samples
// were advanced to admit the write.
int32_t gp_table_edge_publish(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const void* sample_bytes, int32_t sample_len_bytes, int32_t* out_dropped);
int32_t gp_table_edge_publish_blocking(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const void* sample_bytes, int32_t sample_len_bytes, int32_t* out_dropped, int32_t timeout_ms);
// Consume the next available sample as raw bytes. `out_len_bytes` must be
// large enough to hold the sample; `out_written` is set to the number of
// bytes written.
int32_t gp_table_edge_consume(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void* out_sample_bytes, int32_t out_len_bytes, int32_t* out_written);
int32_t gp_table_edge_peek(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void* out_sample_bytes, int32_t out_len_bytes, int32_t* out_written);

// Pointer-oriented edge publish/consume helpers.
// These mirror the float-based APIs but carry opaque pointers across the
// per-edge FIFO. The FIFO retains the same sequencing/reader semantics.
int32_t gp_table_edge_publish_ptr(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, void* ptr, int32_t* out_dropped);
int32_t gp_table_edge_consume_ptr(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void** out_ptr);
// Blocking consume with optional timeout (ms). timeout_ms < 0 waits forever.
// Returns 1 on success, 0 on failure/timeout.
int32_t gp_table_edge_consume_blocking(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void* out_sample_bytes, int32_t out_len_bytes, int32_t* out_written, int32_t timeout_ms);
// Query unread sample count for a subscriber on an edge.
int32_t gp_table_edge_unread(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, int32_t* out_count);
int32_t gp_table_edge_set_batch_metadata(GP_TableContext* ctx, int32_t edge_idx, const GP_TableEdgeBatchMetadata* metadata);
int32_t gp_table_edge_get_batch_metadata(GP_TableContext* ctx, int32_t edge_idx, GP_TableEdgeBatchMetadata* out_metadata);
// Set/get policy subgroup flags for a specific edge. These flags are used to
// derive color wheel hues for rendering and policy grouping.
int32_t gp_table_edge_set_subgroup_flags(GP_TableContext* ctx, int32_t edge_idx, uint32_t flags);
// Query the RopeSim index associated with an edge, or -1 if none.
int32_t gp_table_get_edge_rope_index(const GP_TableContext* ctx, int32_t edge_idx, int32_t* out_rope_idx);
// Resolve a persistent rope id to the attached RopeSim index for this table.
// Returns -1 if not present.
int gp_table_resolve_rope_id_to_sim_index(GP_TableContext* ctx, uint64_t id);
// Bind an existing rope id to a specific RopeSim index.
int32_t gp_table_bind_rope_id_to_sim_index(GP_TableContext* ctx, uint64_t id, int32_t rope_idx);
int32_t gp_table_edge_get_subgroup_flags(GP_TableContext* ctx, int32_t edge_idx, uint32_t* out_flags);
int32_t gp_table_edge_index_for_key(GP_TableContext* ctx, unsigned long long led_key, int32_t* out_edge_idx);
int32_t gp_table_edge_index_for_pair(GP_TableContext* ctx, unsigned long long a, unsigned long long b, int32_t* out_edge_idx);
int32_t gp_table_edge_set_delta_mode(GP_TableContext* ctx, int32_t edge_idx, int32_t delta_mode);
int32_t gp_table_edge_set_order_mode(GP_TableContext* ctx, int32_t edge_idx, int32_t order_mode);
int32_t gp_table_remove_edge(GP_TableContext* ctx, int32_t edge_idx);
int32_t gp_table_remove_edge_pair(GP_TableContext* ctx, unsigned long long a, unsigned long long b);

// Ring-as-edge API: register a simulated ring (created by rope_sim_create_ring)
// as a drawable/inspectable element with its own FIFO/color metadata. Returns
// a ring_entry index >=0 on success, or -1 on failure.
int32_t gp_table_register_ring_edge(GP_TableContext* ctx, int32_t ring_id, unsigned long long ring_key);
int32_t gp_table_unregister_ring_edge(GP_TableContext* ctx, int32_t ring_entry_idx);
int32_t gp_table_get_ring_edge_count(const GP_TableContext* ctx);
int32_t gp_table_get_ring_edge(const GP_TableContext* ctx, int32_t idx, int32_t* out_ring_id, unsigned long long* out_key);
int32_t gp_table_ring_set_subgroup_flags(GP_TableContext* ctx, int32_t ring_entry_idx, uint32_t flags);
int32_t gp_table_ring_get_subgroup_flags(GP_TableContext* ctx, int32_t ring_entry_idx, uint32_t* out_flags);
int32_t gp_table_ring_set_tensor_spec(GP_TableContext* ctx, int32_t ring_entry_idx, const GP_TableEdgeTensorSpecTyped* spec);
// Ring FIFO publish/subscribe APIs (mirror per-edge APIs)
int32_t gp_table_ring_subscribe(GP_TableContext* ctx, int32_t ring_entry_idx, unsigned long long subscriber_key);
int32_t gp_table_ring_subscribe_ex(GP_TableContext* ctx, int32_t ring_entry_idx, unsigned long long subscriber_key, int32_t start_at_head);
int32_t gp_table_ring_unsubscribe(GP_TableContext* ctx, int32_t ring_entry_idx, unsigned long long subscriber_key);
int32_t gp_table_ring_publish(GP_TableContext* ctx, int32_t ring_entry_idx, unsigned long long writer_key, const void* sample_bytes, int32_t sample_len_bytes, int32_t* out_dropped);
int32_t gp_table_ring_get_tensor_spec(GP_TableContext* ctx, int32_t ring_entry_idx, GP_TableEdgeTensorSpecTyped* out_spec);

// Queued UI operations: enqueue structural edits from UI threads to be applied
// by the manager thread. These mirror immediate APIs but defer application.
int32_t gp_table_enqueue_add_edge(GP_TableContext* ctx, unsigned long long a, unsigned long long b);
int32_t gp_table_enqueue_clear_edges(GP_TableContext* ctx);
int32_t gp_table_enqueue_edge_subscribe_ex(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, int32_t start_at_head);
int32_t gp_table_enqueue_edge_unsubscribe(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key);
int32_t gp_table_enqueue_bind_stage_port(GP_TableContext* ctx, unsigned long long led_key, GP_StageContext* stage, int32_t is_output, int32_t channel);
int32_t gp_table_enqueue_unbind_stage_port(GP_TableContext* ctx, unsigned long long led_key);

// Apply any pending queued ops on the current thread (manager should call this).
int32_t gp_table_apply_pending_ops(GP_TableContext* ctx);

// Network-only immutable snapshot (NO UI data). Two-call pattern:
// 1) gp_table_snapshot_network_size -> returns node and edge counts + stamp
// 2) gp_table_snapshot_network_fill -> caller supplies buffers sized from step (1)
// The snapshot contains only endpoint keys and edges (node indices + edge_uid).
// These functions perform no locking and assume the caller is the manager
// thread responsible for touching the table context.
typedef struct GP_TableEdgeSnapshot {
    uint32_t a_idx;
    uint32_t b_idx;
    uint64_t edge_uid;
} GP_TableEdgeSnapshot;

int32_t gp_table_snapshot_network_size(GP_TableContext* ctx, int32_t* out_node_count, int32_t* out_edge_count, uint64_t* out_stamp);
int32_t gp_table_snapshot_network_fill(GP_TableContext* ctx, uint64_t* node_buf, int32_t node_buf_len, GP_TableEdgeSnapshot* edge_buf, int32_t edge_buf_len, uint64_t expected_stamp);

// Stage port bindings --------------------------------------------------------
// Bind a stage instance to a specific LED key so edges can auto-provision
// FIFO writers/readers between stage output/input ports.
int32_t gp_table_bind_stage_port(GP_TableContext* ctx, unsigned long long led_key, GP_StageContext* stage, int32_t is_output, int32_t channel);
int32_t gp_table_unbind_stage_port(GP_TableContext* ctx, unsigned long long led_key);
int32_t gp_table_get_stage_port(GP_TableContext* ctx, unsigned long long led_key, GP_StageContext** out_stage, int32_t* out_is_output, int32_t* out_channel);

// Selected LED query helpers
int32_t gp_table_get_selected_count(const GP_TableContext* ctx);
int32_t gp_table_get_selected_key(const GP_TableContext* ctx, int32_t idx, unsigned long long* out_key);

// Prospective mode: when enabled and exactly one LED is selected, the renderer will
// optionally render a live prospective edge whose free end is relaxed towards the
// supplied mouse position. Python can enable this mode and should supply mouse
// coordinates each frame via GP_TableRenderState to drive the live target.
int32_t gp_table_set_prospective_mode(GP_TableContext* ctx, int32_t enabled);
int32_t gp_table_get_prospective_mode(GP_TableContext* ctx, int32_t* out_enabled);

// Prospective parameters: max_history (how many mouse targets to queue), slack
// (distance in pixels to consider a target cleared), and rope_length (visual
// slack length, currently unused but reserved). Returns 1 on success.
int32_t gp_table_prospective_set_params(GP_TableContext* ctx, int32_t max_history, float slack, float rope_length);
int32_t gp_table_prospective_get_params(GP_TableContext* ctx, int32_t* out_max_history, float* out_slack, float* out_rope_length);

// Serialization API -------------------------------------------------------
// Serialize the table context into a binary blob. If `out_buf` is NULL, the
// function returns the number of bytes that would be written. Otherwise it
// writes up to `out_len` bytes and returns the number of bytes written, or 0
// on failure.
int32_t gp_table_serialize(GP_TableContext* ctx, char* out_buf, int32_t out_len);

// Deserialize from a binary blob produced by gp_table_serialize. Returns 1 on
// success, 0 on failure. Deserialization will replace rows/cols/edges and
// recompute geometry; callers are responsible for attaching rope sims if
// desired afterward.
int32_t gp_table_deserialize(GP_TableContext* ctx, const char* in_buf, int32_t in_len);

// Module/port UUID helpers embedded into serialized blobs
// Set a module UUID onto a table context so it will be included in serialization.
int32_t gp_table_set_module_uuid(GP_TableContext* ctx, uint64_t module_uuid);
// Set a module frame port UUID (row,row_idx) so it will be included in serialization.
int32_t gp_table_set_frame_port_uuid(GP_TableContext* ctx, int row, int idx, uint64_t port_uuid);

// Editable flag: tables may be marked editable to expose them to an editor.
// Default is editable (1). Callers can toggle editability at will.
int32_t gp_table_set_editable(GP_TableContext* ctx, int32_t editable);
int32_t gp_table_get_editable(GP_TableContext* ctx, int32_t* out_editable);

// Chain relaxer modes for animating/relaxing edges. Use the gp_table_relax_* APIs
// to configure and drive the relaxer.
typedef enum GP_TableRelaxMode {
    GP_TABLE_RELAX_OFF = 0,
    GP_TABLE_RELAX_DT = 1,         // step using supplied dt (gp_table_relax_step)
    GP_TABLE_RELAX_WALL_TIME = 2,  // step automatically using wall-clock time (gp_table_relax_update)
    GP_TABLE_RELAX_UNTIL_STABLE = 3 // run until change below threshold (gp_table_relax_run_until_stable)
} GP_TableRelaxMode;

// Relaxation control APIs
int32_t gp_table_relax_set_mode(GP_TableContext* ctx, int32_t mode);
int32_t gp_table_relax_get_mode(GP_TableContext* ctx, int32_t* out_mode);
int32_t gp_table_relax_set_params(GP_TableContext* ctx, float stiffness, float damping, float threshold, int32_t max_iters);
int32_t gp_table_relax_step(GP_TableContext* ctx, float dt);
int32_t gp_table_relax_update(GP_TableContext* ctx); // uses wall time internally
int32_t gp_table_relax_run_until_stable(GP_TableContext* ctx);

// Meta-group API: groups of simulated rope vertices used by the Meta-Edge Lasso
// tool. A meta-group owns a list of (rope_idx, vertex_idx) bindings and a
// confinement parameter. These APIs are lightweight: groups are created and
// owned by the caller (the table retains a record for runtime queries).
GP_MetaGroup* gp_table_meta_create(GP_TableContext* ctx);
int32_t gp_table_meta_destroy(GP_TableContext* ctx, GP_MetaGroup* mg);
int32_t gp_table_meta_add_vertex(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t rope_idx, int32_t vertex_idx);
int32_t gp_table_meta_get_vertex_count(GP_TableContext* ctx, GP_MetaGroup* mg);

// Set/get a preferred anchor for a meta-group so helpers (widgets) attach
// to the specified `(rope_idx, vertex_idx)`. Returns 1 on success.
int32_t gp_table_meta_set_anchor(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t rope_idx, int32_t vertex_idx);
int32_t gp_table_meta_get_anchor(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_rope_idx, int32_t* out_vertex_idx);

// Configure lasso meta-group behavior. `cfg` may be NULL to clear and
// resets to defaults. Returns 1 on success.
int32_t gp_table_meta_set_lasso_config(GP_TableContext* ctx, GP_MetaGroup* mg, const LassoConfig* cfg);
int32_t gp_table_meta_get_lasso_config(GP_TableContext* ctx, GP_MetaGroup* mg, LassoConfig* out_cfg);

// Store edge-spring parameters for a meta-group so they survive save/load.
int32_t gp_table_meta_set_edge_spring_params(GP_TableContext* ctx, GP_MetaGroup* mg, float min_rest, float reduce_rate, int32_t mode);
// Create a dangling widget attached to a meta-group. The widget is created
// in the attached RopeSim (if any) and follows the first vertex in the
// meta-group. Returns 1 on success.
int32_t gp_table_meta_create_widget(GP_TableContext* ctx, GP_MetaGroup* mg);
int32_t gp_table_meta_destroy_widget(GP_TableContext* ctx, GP_MetaGroup* mg);
// Associate canvas overlay keys with a meta-group so they will be serialized.
int32_t gp_table_meta_set_overlay_keys(GP_TableContext* ctx, GP_MetaGroup* mg, unsigned long long key_a, unsigned long long key_b);
int32_t gp_table_meta_get_overlay_keys(GP_TableContext* ctx, GP_MetaGroup* mg, unsigned long long* out_key_a, unsigned long long* out_key_b);
// Set/get the integer "channel group" for a meta-group. Returns 1 on success.
int32_t gp_table_meta_set_channel_group(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t channel_group);
int32_t gp_table_meta_get_channel_group(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_channel_group);
// Additional accessors for canvas-level serialization
int32_t gp_table_meta_get_confinement(const GP_TableContext* ctx, GP_MetaGroup* mg, float* out_conf);
int32_t gp_table_meta_set_confinement(GP_TableContext* ctx, GP_MetaGroup* mg, float conf);
int32_t gp_table_meta_get_id(const GP_TableContext* ctx, GP_MetaGroup* mg, unsigned long long* out_id);
int32_t gp_table_meta_set_id(GP_TableContext* ctx, GP_MetaGroup* mg, unsigned long long id);
int32_t gp_table_meta_get_dangling_hang_len(const GP_TableContext* ctx, GP_MetaGroup* mg, float* out_len);
int32_t gp_table_meta_set_dangling_hang_len(GP_TableContext* ctx, GP_MetaGroup* mg, float len);
// Lasso fields (flags + widget type) accessor helpers (avoid exposing full struct in headers)
int32_t gp_table_meta_get_lasso_fields(const GP_TableContext* ctx, GP_MetaGroup* mg, unsigned int* out_flags, int32_t* out_widget_type);
int32_t gp_table_meta_set_lasso_fields(GP_TableContext* ctx, GP_MetaGroup* mg, unsigned int flags, int32_t widget_type);
// Ring mode setter (getter already exists)
int32_t gp_table_meta_set_ring_mode(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t mode);
// Snapshot the FIFO metadata for a meta-group into a float array.
// Returns the number of floats written (or 0 on error). The buffer should be
// large enough to hold: id, vertex_count, channel_group, stride, slots,
// top_k, write_seq, writer_low, last_write_seq, last_read_seq,
// write_friction, read_friction, last_write_region, last_read_region,
// write_phase, read_phase, friction_regions, configured(0/1), shape_len, shape[...]
int32_t gp_table_meta_get_fifo_snapshot(GP_TableContext* ctx, GP_MetaGroup* mg, float* out_buf, int32_t out_len);
// Enable/disable edge-springs for a meta-group. Springs connect consecutive
// vertices in the group's member list and will reduce their rest length
// over time until `min_rest` at the given `reduce_rate` (units per second).
int32_t gp_table_meta_enable_edge_springs(GP_TableContext* ctx, GP_MetaGroup* mg, float min_rest, float reduce_rate);
int32_t gp_table_meta_disable_edge_springs(GP_TableContext* ctx, GP_MetaGroup* mg);
// Set ring topology mode for an existing meta-group. Mode should be one of
// `GP_META_RING_RIBBON`, `GP_META_RING_CLOSED`, or `GP_META_RING_DENSE`.
// Removed table-side ring-mode API — use sim-only toggle instead.
// NOTE: sim-only meta-group operations should be done via `rope_sim` APIs directly.
// Retrieve the current ring mode for a meta-group (0=ribbon,1=closed,2=dense)
int32_t gp_table_meta_get_ring_mode(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t* out_mode);
// Per-rope rest-length control wrappers
int32_t gp_table_rope_modify_rest_length(GP_TableContext* ctx, int32_t rope_idx, float delta);
int32_t gp_table_rope_set_rest_target(GP_TableContext* ctx, int32_t rope_idx, float target_rest, float rate, float delay);
int32_t gp_table_rope_get_rest_length(GP_TableContext* ctx, int32_t rope_idx, float* out_rest);
// Per-rope radius accessors
int32_t gp_table_rope_set_radius(GP_TableContext* ctx, int32_t rope_idx, float radius);
int32_t gp_table_rope_get_radius(GP_TableContext* ctx, int32_t rope_idx, float* out_radius);
// Insert a vertex into a rope at fractional position along segment (seg_index,t).
// Returns new vertex index or -1.
int32_t gp_table_rope_insert_vertex(GP_TableContext* ctx, int32_t rope_idx, int32_t seg_index, float t);

// Simulator-only helper: create a transient sim meta-group connecting the
// endpoints of `rope_idx` and enable edge-springs so additional physics
// edges are evaluated. Returns 1 on success, 0 on failure.
int32_t gp_table_sim_add_meta_group_for_rope(GP_TableContext* ctx, int32_t rope_idx);


// Attach or detach an external RopeSim instance to the table context.
// If `sim` is non-null the table will use that simulator for all rope
// allocations and updates. `take_ownership` indicates whether the table
// should destroy the provided simulator when the table is destroyed
// (1 = table destroys the sim, 0 = caller retains ownership). Passing
// `sim == NULL` detaches any external simulator; the table may create
// its own simulator later on demand. Returns 1 on success.
int32_t gp_table_attach_rope_sim(GP_TableContext* ctx, RopeSim* sim, int32_t take_ownership);

// IO and type-hint helpers -----------------------------------------------
// Packed LED key is (row<<32)|(col<<16)|led_index as used elsewhere.
// Set a type hint for a specific LED key. `is_input`/`is_output` are booleans
// indicating whether this key can act as an input and/or output. Returns 1
// on success.
int32_t gp_table_set_key_type_hint(GP_TableContext* ctx, unsigned long long key, int32_t type_id, int32_t is_input, int32_t is_output);
int32_t gp_table_get_key_type_hint(GP_TableContext* ctx, unsigned long long key, int32_t* out_type_id, int32_t* out_is_input, int32_t* out_is_output);

// Query whether the table exposes at least one input and one output key.
int32_t gp_table_has_io_sections(GP_TableContext* ctx);

// Enumerate IO keys: fill `out_keys` with up to `cap` keys for the requested
// direction (0 = inputs, 1 = outputs). Returns number written.
int32_t gp_table_enumerate_io_keys(GP_TableContext* ctx, int32_t direction, unsigned long long* out_keys, int32_t cap);

// Reading direction for input/output sides. `dir` is one of:
// 0 = LeftToRight, 1 = TopToBottom, 2 = RightToLeft, 3 = BottomToTop
int32_t gp_table_set_side_reading_direction(GP_TableContext* ctx, int32_t side /*0=input,1=output*/, int32_t dir);
int32_t gp_table_get_side_reading_direction(GP_TableContext* ctx, int32_t side /*0=input,1=output*/, int32_t* out_dir);

// LED grid layout preference: prefer `pref_cols x pref_rows` or a given aspect
// ratio when rendering LED-table blocks. These are hints only. Returns 1 on success.
int32_t gp_table_set_led_grid_preference(GP_TableContext* ctx, int32_t pref_cols, int32_t pref_rows, float pref_aspect);
int32_t gp_table_get_led_grid_preference(GP_TableContext* ctx, int32_t* out_pref_cols, int32_t* out_pref_rows, float* out_pref_aspect);

// Table object definition (rows/columns/actions) for external modules to consume.
typedef struct GP_TableObjectDef {
    const GP_TableColumn* cols;
    int32_t col_count;
    const GP_TableRow* rows;
    int32_t row_count;
    const GP_TableAction* actions;
    int32_t action_count;
} GP_TableObjectDef;

// Apply a table object definition onto an existing table context.
int32_t gp_table_apply_object_def(GP_TableContext* ctx, const GP_TableObjectDef* def);

// Return the RopeSim instance owned/used by this table (may be NULL).
RopeSim* gp_table_get_rope_sim(GP_TableContext* ctx);

// Table step callback and runtime invocation ---------------------------------
// A table shim may provide a step function which consumes `in_count` input
// values and writes `out_count` outputs. The callback receives an opaque
// `user` pointer supplied by the caller and is invoked with arrays of floats.
// dt is the timestep in seconds. The callback must not block.
typedef void(*GP_TableStepFn)(void* user, const float* inputs, int32_t in_count, float* outputs, int32_t out_count, double dt);

// Install / clear a step callback on a table context. Returns 1 on success.
int32_t gp_table_set_step_callback(GP_TableContext* ctx, GP_TableStepFn cb, void* user);
int32_t gp_table_clear_step_callback(GP_TableContext* ctx);

// Invoke the step for a table context. `inputs`/`outputs` are arrays of
// floats with counts matching the table's enumerated IO keys (caller is
// responsible for matching sizes). Returns 1 on success (callback invoked),
// 0 if ctx is NULL or no callback is installed.
int32_t gp_table_step(GP_TableContext* ctx, const float* inputs, int32_t in_count, float* outputs, int32_t out_count, double dt);

// Template / filesystem helpers ------------------------------------------
// Save the current table context as a named template into `dir` (if dir is
// NULL the library directory set via gp_canvas_set_templates_dir is used).
// Template files are written as binary blobs produced by gp_table_serialize
// and named `<name>.gptbl`. Returns 1 on success.
int32_t gp_table_save_template(GP_TableContext* ctx, const char* dir, const char* name);

// Load a named template from `dir` into the provided table context (overwrites
// rows/cols/edges/etc). Returns 1 on success.
int32_t gp_table_load_template(GP_TableContext* ctx, const char* dir, const char* name);

// List available templates in `dir` (or library dir if NULL). If `out_buf` is
// NULL the function returns the number of bytes required; otherwise writes up
// to `out_len` bytes of a newline-separated UTF-8 list and returns bytes written.
int32_t gp_table_list_templates(const char* dir, char* out_buf, int32_t out_len);

// Set/get the global template library directory used when `dir==NULL` in
// template APIs. Passing NULL or empty string clears the library dir.
int32_t gp_table_set_library_dir(const char* dir);
int32_t gp_table_get_library_dir(char* out_buf, int32_t out_len);

// Generic edge API wrappers (convenience). These forward to the table-specific
// implementations so external callers can use a stable gp_edge_* surface.
int32_t gp_edge_publish(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const void* sample_bytes, int32_t sample_len_bytes, int32_t* out_dropped);
int32_t gp_edge_publish_blocking(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const void* sample_bytes, int32_t sample_len_bytes, int32_t* out_dropped, int32_t timeout_ms);
int32_t gp_edge_publish_ptr(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, void* ptr, int32_t* out_dropped);
int32_t gp_edge_consume_ptr(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void** out_ptr);

// Configure mapping from subgroup color index (0..4) to FIFO flags.
// This lets users map UI subgroup colors to edge FIFO behavior (e.g. BYREF).
extern "C" int32_t gp_table_set_subgroup_color_mapping(int32_t color_idx, uint32_t fifo_flags);

#ifdef __cplusplus
}
#endif
