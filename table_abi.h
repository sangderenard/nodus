#pragma once

// Generic C ABI for rasterizing dense tables with hierarchical rows, LEDs, axis bars,
// timers, optional waveforms, and simple expand/collapse affordances.
// Text is carried through the API (label + per-cell text) but this minimal raster
// does not render text; callers can overlay text separately using their text system.

#include <stdint.h>

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
    GP_TABLE_CELL_SCROLL = 8,      // vertical scrollbar (arrows + track + thumb)
    GP_TABLE_CELL_IMAGE = 9,       // RGBA image content
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

typedef struct GP_TableEdgeBatchMetadata {
    uint64_t batch_id;
    double timestamp;
    uint32_t sample_count;
    uint32_t stride;
    uint32_t schema_id;
} GP_TableEdgeBatchMetadata;

int32_t gp_table_edge_set_tensor_spec(GP_TableContext* ctx, int32_t edge_idx, const GP_TableEdgeTensorSpec* spec);
int32_t gp_table_edge_get_tensor_spec(GP_TableContext* ctx, int32_t edge_idx, GP_TableEdgeTensorSpec* out_spec);
int32_t gp_table_edge_subscribe(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key);
// Subscribe with explicit start policy. If `start_at_head` is non-zero, the subscriber
// begins at the current write head (new samples only). If zero, it begins at the
// earliest sample still addressable within the ring capacity (best-effort).
int32_t gp_table_edge_subscribe_ex(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, int32_t start_at_head);
int32_t gp_table_edge_unsubscribe(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key);
// Publish one tensor sample into the FIFO for an edge. Returns 1 on success.
// If the buffer advanced slow readers to admit the write (top-k overwrite),
// `out_dropped` is set to 1; otherwise 0. Returns 0 if no write occurred.
int32_t gp_table_edge_publish(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const float* sample, int32_t sample_len, int32_t* out_dropped);
// Blocking publish with optional timeout (ms). timeout_ms < 0 waits forever.
// Returns 1 on success, 0 on failure/timeout.
int32_t gp_table_edge_publish_blocking(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const float* sample, int32_t sample_len, int32_t* out_dropped, int32_t timeout_ms);
// Consume the next available tensor sample for a subscriber. Returns 1 if a
// sample was written to `out_sample` (length must match the tensor stride).
int32_t gp_table_edge_consume(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, float* out_sample, int32_t out_len, int32_t* out_written);
// Blocking consume with optional timeout (ms). timeout_ms < 0 waits forever.
// Returns 1 on success, 0 on failure/timeout.
int32_t gp_table_edge_consume_blocking(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, float* out_sample, int32_t out_len, int32_t* out_written, int32_t timeout_ms);
// Query unread sample count for a subscriber on an edge.
int32_t gp_table_edge_unread(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, int32_t* out_count);
int32_t gp_table_edge_set_batch_metadata(GP_TableContext* ctx, int32_t edge_idx, const GP_TableEdgeBatchMetadata* metadata);
int32_t gp_table_edge_get_batch_metadata(GP_TableContext* ctx, int32_t edge_idx, GP_TableEdgeBatchMetadata* out_metadata);
int32_t gp_table_edge_index_for_key(GP_TableContext* ctx, unsigned long long led_key, int32_t* out_edge_idx);

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

#ifdef __cplusplus
}
#endif
