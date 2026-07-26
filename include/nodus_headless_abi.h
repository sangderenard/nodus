#pragma once
// Headless (no window/renderer) nodus graph runtime C ABI.
//
// Wraps a GP_CanvasContext purely as an in-process graph container and tool
// dispatch mechanism: nothing here ever calls gp_canvas_raster_rgba or any
// other rendering entry point, and no window/SDL/GL handle is ever created.
// This is the "Prong 1" headless entry point: build or load a tool graph,
// run ThreadManager's real scheduler to quiescence, and exchange data at a
// small set of named external ports, all from a plain console process (or
// from Python via ctypes).
//
// Scope note: a GP_CanvasContext still exists in-process because tool
// dispatch (ThreadManager::run_scheduled_tick) currently reads tool/plugin
// bindings through a process-global canvas singleton rather than through
// TickRequest itself. Decoupling that is tracked as later work, not part of
// this ABI. "Headless" here means: no window, no renderer, no GUI, and the
// caller never needs to know internal topology -- not "zero canvas object
// in the process."
//
// Quiescence note: nodus_headless_run_to_quiescence reports quiescence when
// all declared external ports have zero unread samples after a tick -- i.e.
// no pending work at the black-box boundary this ABI exposes. It is not a
// whole-graph idle proof; process-level quiescence accounting beyond
// per-edge unread counts is not yet implemented anywhere in nodus.

#include <stddef.h>
#include <stdint.h>

#include "canvas_abi.h"

#if defined(_WIN32)
#  if defined(NODUS_HEADLESS_BUILD)
#    define NODUS_HEADLESS_API __declspec(dllexport)
#  else
#    define NODUS_HEADLESS_API __declspec(dllimport)
#  endif
#else
#  define NODUS_HEADLESS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NodusHeadlessGraph NodusHeadlessGraph;

// NODUS_EVT_QUIESCENT and NODUS_EVT_OVERFLOW are fired by this Phase A
// implementation (see src/headless/nodus_headless_abi.cpp). NODUS_EVT_
// NODE_COMPLETE and NODUS_EVT_REJECTED_TXN are reserved in this enum for API
// stability but are not fired yet: per-tool completion has no generic
// signal to hook today, and rejected/rolled-back transactions depend on the
// quiescent-transaction snapshot/restore machinery (gp_table_edge_
// transaction_snapshot_*), which nothing in this ABI drives yet. Both are
// real future work, not silently-dropped features.
typedef enum NodusHeadlessEventKind {
    NODUS_EVT_QUIESCENT = 0,
    NODUS_EVT_NODE_COMPLETE = 1,
    NODUS_EVT_OVERFLOW = 2,
    NODUS_EVT_REJECTED_TXN = 3,
} NodusHeadlessEventKind;

// Fixed, trivially-copyable event record. Never crosses back into C++/STL --
// the ABI boundary stays POD-only so callers (including Python via ctypes)
// never need to understand nodus's internal types.
typedef struct NodusHeadlessEvent {
    int32_t kind;        // NodusHeadlessEventKind
    int32_t module_idx;  // -1 if not applicable
    int32_t edge_idx;    // -1 if not applicable
    uint64_t tick_id;
    int32_t code;        // reserved (drop reason, rejection reason, ...)
} NodusHeadlessEvent;

NODUS_HEADLESS_API NodusHeadlessGraph* nodus_headless_create(void);
NODUS_HEADLESS_API void nodus_headless_destroy(NodusHeadlessGraph* g);

// Load/save the full graph topology (modules, tool/plugin bindings, edges)
// using nodus's existing canvas save format -- no new format is introduced.
NODUS_HEADLESS_API int32_t nodus_headless_load_file(NodusHeadlessGraph* g, const char* path);
NODUS_HEADLESS_API int32_t nodus_headless_save_file(NodusHeadlessGraph* g, const char* path);

// Programmatic graph construction, for building a graph in-process instead
// of authoring a save file. Returns the new index, or -1 on failure.
NODUS_HEADLESS_API int32_t nodus_headless_add_module(NodusHeadlessGraph* g, const GP_CanvasModuleDesc* desc);
NODUS_HEADLESS_API int32_t nodus_headless_bind_builtin_tool(NodusHeadlessGraph* g, int32_t module_idx, int32_t tool_kind);
NODUS_HEADLESS_API int32_t nodus_headless_bind_plugin_tool(NodusHeadlessGraph* g, int32_t module_idx, const char* plugin_id);
NODUS_HEADLESS_API int32_t nodus_headless_add_edge(NodusHeadlessGraph* g, const GP_CanvasEdgeDesc* desc, int32_t type_id);

// Named external ports: the opaque black-box boundary. Each port is bound to
// one module's tool row and rides the same table-edge transport
// (gp_table_edge_publish/consume) that real graph edges use internally --
// declaring a port creates (or reuses) a table edge for that row, configures
// it as a `token_bytes`-byte-per-sample opaque FIFO (nodus's edge transport
// is fixed-shape and typed, configured once before use -- every push/pull on
// this port must be exactly `token_bytes` bytes), and registers this runtime
// as a reader/writer on it. Returns 1 on success.
NODUS_HEADLESS_API int32_t nodus_headless_declare_input_port(NodusHeadlessGraph* g, const char* name, int32_t module_idx, int32_t row_idx, int32_t token_bytes);
NODUS_HEADLESS_API int32_t nodus_headless_declare_output_port(NodusHeadlessGraph* g, const char* name, int32_t module_idx, int32_t row_idx, int32_t token_bytes);
NODUS_HEADLESS_API int32_t nodus_headless_push_token(NodusHeadlessGraph* g, const char* port_name, const void* bytes, size_t len);
NODUS_HEADLESS_API int32_t nodus_headless_pull_token(NodusHeadlessGraph* g, const char* port_name, void* out_bytes, size_t cap, size_t* out_written);

// Scheduling. Both drive the real ThreadManager frontier via gp_canvas_step
// -- never a hand-written per-tool tick loop.
NODUS_HEADLESS_API int32_t nodus_headless_step(NodusHeadlessGraph* g, double dt);
NODUS_HEADLESS_API int32_t nodus_headless_run_to_quiescence(NodusHeadlessGraph* g, double dt, int32_t max_ticks);
NODUS_HEADLESS_API int32_t nodus_headless_is_quiescent(const NodusHeadlessGraph* g);

// Event polling. Events are queued internally as they occur and drained
// here explicitly -- never delivered via a direct callback from the
// scheduler thread, so Python callers never need to acquire the GIL from a
// nodus-owned thread. Returns the number of events written into `out`
// (up to `capacity`).
NODUS_HEADLESS_API int32_t nodus_headless_poll_events(NodusHeadlessGraph* g, NodusHeadlessEvent* out, int32_t capacity);

#ifdef __cplusplus
}
#endif
