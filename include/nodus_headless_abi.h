#pragma once
// Headless (no window/renderer) nodus graph runtime C ABI.
//
// Wraps a GP_CanvasContext purely as an in-process graph container and tool
// dispatch mechanism: nothing here ever calls gp_canvas_raster_rgba or any
// other rendering entry point, and no window or GL rendering context is created by this graph API. Optional
// audio tool rows may own SDL audio-device handles.
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
#include "python_row_workflow.h"
#include "turing_product_adopter.h"
#include "common/tensors/abstraction/tensor_abi.h"

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

typedef struct NodusHeadlessFifoConfig {
    int32_t token_bytes;
    int32_t slots;
    int32_t top_k;
    int32_t type_id;
    int32_t layout;
    int32_t dtype;
    int32_t backend_type; // GP_MemBackendType; CPU or Vulkan in graph-definition v1
} NodusHeadlessFifoConfig;

typedef struct NodusHeadlessPublishResult {
    int32_t accepted;
    int32_t dropped;
} NodusHeadlessPublishResult;

NODUS_HEADLESS_API NodusHeadlessGraph* nodus_headless_create(void);
NODUS_HEADLESS_API void nodus_headless_destroy(NodusHeadlessGraph* g);

// Load/save the full graph topology (modules, tool/plugin bindings, edges)
// using nodus's existing canvas save format -- no new format is introduced.
// The per-module serialized tables (.gpmod) travel beside the canvas file in
// the directory "<path>.modules" (created on save): both calls point the
// canvas module-library root there for the duration of the call and restore
// the previous root afterward, so nothing is written to or read from the cwd.
NODUS_HEADLESS_API int32_t nodus_headless_load_file(NodusHeadlessGraph* g, const char* path);
NODUS_HEADLESS_API int32_t nodus_headless_save_file(NodusHeadlessGraph* g, const char* path);
// Export one module's concrete table serialization under
// <output_root>/serialized/module_<index>.gpmod.
NODUS_HEADLESS_API int32_t nodus_headless_export_module_to_root(
    NodusHeadlessGraph* g, int32_t module_idx, const char* output_root);

// Programmatic graph construction, for building a graph in-process instead
// of authoring a save file. Returns the new index, or -1 on failure.
NODUS_HEADLESS_API int32_t nodus_headless_add_module(NodusHeadlessGraph* g, const GP_CanvasModuleDesc* desc);
NODUS_HEADLESS_API int32_t nodus_headless_get_module_count(NodusHeadlessGraph* g);
NODUS_HEADLESS_API int32_t nodus_headless_get_module_desc(
    NodusHeadlessGraph* g, int32_t module_idx, GP_CanvasModuleDesc* out_desc);
NODUS_HEADLESS_API int32_t nodus_headless_set_module_desc(
    NodusHeadlessGraph* g, int32_t module_idx, const GP_CanvasModuleDesc* desc);
NODUS_HEADLESS_API int32_t nodus_headless_bind_builtin_tool(NodusHeadlessGraph* g, int32_t module_idx, int32_t tool_kind);
NODUS_HEADLESS_API int32_t nodus_headless_bind_plugin_tool(NodusHeadlessGraph* g, int32_t module_idx, const char* plugin_id);
// Admit one native TCP byte-stream row into this graph's registry. The host
// then binds its id with nodus_headless_bind_plugin_tool and connects its
// Argument/Return contacts to ordinary typed FIFOs. The port is explicit so
// a saved graph and its host contract can name the same loopback endpoint.
// The row is unregistered after graph destruction.
NODUS_HEADLESS_API int32_t nodus_headless_register_tcp_byte_stream(
    NodusHeadlessGraph* g, const char* tool_id, uint16_t port,
    uint64_t maximum_frame_bytes, uint32_t pending_frames);
NODUS_HEADLESS_API int32_t nodus_headless_set_module_rows(
    NodusHeadlessGraph* g, int32_t module_idx,
    const GP_CanvasModuleRowDesc* rows, int32_t row_count);
NODUS_HEADLESS_API int32_t nodus_headless_get_module_row_count(
    NodusHeadlessGraph* g, int32_t module_idx);
NODUS_HEADLESS_API int32_t nodus_headless_get_module_row(
    NodusHeadlessGraph* g, int32_t module_idx, int32_t row_idx,
    GP_CanvasModuleRowInfo* out_info, char* plugin_id,
    int32_t plugin_id_capacity);
NODUS_HEADLESS_API int32_t nodus_headless_set_module_stack_contract(
    NodusHeadlessGraph* g, int32_t module_idx, int32_t contract);
NODUS_HEADLESS_API int32_t nodus_headless_get_module_stack_contract(
    NodusHeadlessGraph* g, int32_t module_idx, int32_t* out_contract);
NODUS_HEADLESS_API int32_t nodus_headless_set_module_exec_mode(
    NodusHeadlessGraph* g, int32_t module_idx, int32_t mode);
NODUS_HEADLESS_API int32_t nodus_headless_get_module_exec_mode(
    NodusHeadlessGraph* g, int32_t module_idx, int32_t* out_mode);

// Register the native Radio source and UDP broadcast rows. Configure before
// binding either row. Both rows belong to one Opaque Free table, in that
// order. The source reuses the native WASAPI/fftfree scope named here.
NODUS_HEADLESS_API int32_t nodus_headless_radio_configure(
    const wchar_t* scope_dll, const wchar_t* fftfree_dll,
    const char* destination_host, uint16_t destination_port,
    uint32_t fft_size, uint32_t overlap, uint32_t band_count,
    float min_hz, float max_hz);
typedef struct NodusRadioStatus {
    uint64_t captured_frames;
    uint64_t emitted_packets;
    uint64_t dropped_packets;
    uint32_t sample_rate;
    int32_t capture_running;
} NodusRadioStatus;
// Select WAV playback before constructing/launching the Radio table. No loopback capture.
NODUS_HEADLESS_API int32_t nodus_headless_radio_file_source(const wchar_t* wav_path, int32_t loop);
// Opt-in private PCM RTP sender; configure before binding rows, then bind its
// canonical canvas output FIFO before launching. Requires file source, hop 256.
// Hold source advancement until a trusted receiver has pinned the announced stream.
NODUS_HEADLESS_API int32_t nodus_headless_radio_hold(int32_t held);
NODUS_HEADLESS_API int32_t nodus_headless_radio_pcm_configure(uint16_t loopback_port);
NODUS_HEADLESS_API int32_t nodus_headless_radio_pcm_status(uint64_t* emitted, uint64_t* dropped);
NODUS_HEADLESS_API int32_t nodus_headless_radio_pcm_bind(NodusHeadlessGraph* graph, int32_t module_idx);
NODUS_HEADLESS_API int32_t nodus_headless_radio_status(NodusRadioStatus* out_status);
NODUS_HEADLESS_API int32_t nodus_headless_add_edge(NodusHeadlessGraph* g, const GP_CanvasEdgeDesc* desc, int32_t type_id);
// Drive the same canvas UI handlers used by the GUI, on the graph's owning
// thread. Mouse kinds: 0 click, 1 down, 2 up, 3 move, 4 scroll.
NODUS_HEADLESS_API int32_t nodus_headless_ui_viewport(NodusHeadlessGraph* g, int32_t width, int32_t height);
NODUS_HEADLESS_API int32_t nodus_headless_ui_mouse(NodusHeadlessGraph* g, int32_t kind,
    float x, float y, float dx, float dy, int32_t button, float scroll);
NODUS_HEADLESS_API int32_t nodus_headless_ui_key(NodusHeadlessGraph* g,
    int32_t key, int32_t scancode, int32_t action, int32_t mods);
// Configured edge FIFOs and named ports are persisted by
// nodus_headless_save_file as EDGE_FIFO / PORT trailer records appended after
// the CANVAS V4 body, and restored by nodus_headless_load_file (load returns 0
// if a record cannot be re-attached). CANVAS V4 itself does not carry them, so
// gp_canvas_load_from_file alone (as frontend_shell uses it) restores topology
// only. Live samples are never serialized.
NODUS_HEADLESS_API int32_t nodus_headless_get_edge_count(NodusHeadlessGraph* g);
NODUS_HEADLESS_API int32_t nodus_headless_get_edge(
    NodusHeadlessGraph* g, int32_t edge_idx,
    GP_CanvasEdgeDesc* out_desc, int32_t* out_type_id);
NODUS_HEADLESS_API int32_t nodus_headless_attach_edge(
    NodusHeadlessGraph* g, const GP_CanvasEdgeDesc* desc);
NODUS_HEADLESS_API int32_t nodus_headless_configure_edge_fifo(
    NodusHeadlessGraph* g, int32_t canvas_edge_idx,
    const NodusHeadlessFifoConfig* config);
NODUS_HEADLESS_API int32_t nodus_headless_get_edge_fifo_config(
    NodusHeadlessGraph* g, int32_t canvas_edge_idx,
    NodusHeadlessFifoConfig* out_config);
// Custom native tools can exchange bytes on their configured declared edge.
// Uses the same producer/consumer contact identity as ordinary scheduled rows.
// Call from ordered tool rows or at a completed-step boundary, not concurrently.
NODUS_HEADLESS_API int32_t nodus_headless_publish_edge_token(
    NodusHeadlessGraph* g, int32_t edge_idx, const void* bytes, size_t len,
    NodusHeadlessPublishResult* result);
NODUS_HEADLESS_API int32_t nodus_headless_consume_edge_token(
    NodusHeadlessGraph* g, int32_t edge_idx, void* bytes, size_t capacity,
    size_t* written);
NODUS_HEADLESS_API int32_t nodus_headless_get_edge_unread(
    NodusHeadlessGraph* g, int32_t canvas_edge_idx, int32_t* out_unread);
NODUS_HEADLESS_API int32_t nodus_headless_set_module_uuid(
    NodusHeadlessGraph* g, int32_t module_idx, uint64_t module_uuid);
NODUS_HEADLESS_API uint64_t nodus_headless_get_module_uuid(
    NodusHeadlessGraph* g, int32_t module_idx);
NODUS_HEADLESS_API int32_t nodus_headless_find_module_by_uuid(
    NodusHeadlessGraph* g, uint64_t module_uuid);
NODUS_HEADLESS_API int32_t nodus_headless_set_scheduler_mode(
    NodusHeadlessGraph* g, int32_t mode);
NODUS_HEADLESS_API int32_t nodus_headless_get_scheduler_mode(
    NodusHeadlessGraph* g, int32_t* out_mode);
NODUS_HEADLESS_API int32_t nodus_headless_set_persistent_workers(
    NodusHeadlessGraph* g, int32_t enabled);
NODUS_HEADLESS_API int32_t nodus_headless_get_persistent_workers(
    NodusHeadlessGraph* g, int32_t* out_enabled);

// Compiler/tool and tensor services exposed by this same runtime image. This
// keeps plugin registration and ValueType ids in the registry instance used
// by the graph scheduler rather than asking a foreign caller to load another
// copy of those process-local authorities.
NODUS_HEADLESS_API int32_t nodus_headless_abstract_tensor_type_id(void);
NODUS_HEADLESS_API int32_t nodus_headless_deploy_python_row(
    const char* descriptor_path, const char* target,
    const char* python_executable, const char* turing_root,
    const char* extraction_contract, const char* product_dir,
    const char* nodus_root, const char* module_library_root,
    GP_PythonRowDeploymentReport* report,
    char* out_tool_id, int32_t out_tool_id_cap,
    char* out_retained_python, int32_t out_retained_python_cap,
    char* out_staged_wrapper, int32_t out_staged_wrapper_cap);
NODUS_HEADLESS_API int32_t nodus_headless_unload_plugin(const char* tool_id);
// Build and load one already-generated ordinary Nodus plugin source.  This is
// the narrow native half used by strict product adopters: product validation
// and source generation happen before this call; registration still goes
// through the same PluginLoader/ToolRegistry authority as every other row.
NODUS_HEADLESS_API int32_t nodus_headless_build_plugin_source(
    const char* source_path, const char* nodus_root,
    const char* module_library_root,
    char* out_tool_id, int32_t out_tool_id_cap);
NODUS_HEADLESS_API int32_t nodus_headless_compile_turing_row(
    const char* python_executable, const char* request_json,
    const char* turing_root, const char* product_dir,
    const char* nodus_root, const char* module_library_root,
    GP_TuringProductAdoptionReport* report,
    char* out_tool_id, int32_t out_tool_id_cap,
    char* out_staged_source, int32_t out_staged_source_cap);
NODUS_HEADLESS_API uint64_t nodus_headless_tensor_create(
    int32_t dtype, int32_t rank, const uint64_t* shape);
NODUS_HEADLESS_API int64_t nodus_headless_tensor_write(
    uint64_t handle, uint64_t byte_offset, const void* src, uint64_t byte_count);
NODUS_HEADLESS_API int64_t nodus_headless_tensor_read(
    uint64_t handle, uint64_t byte_offset, void* dst, uint64_t byte_count);
NODUS_HEADLESS_API int32_t nodus_headless_tensor_describe(
    uint64_t handle, NodusTensorDesc* out_desc);
NODUS_HEADLESS_API void nodus_headless_tensor_destroy(uint64_t handle);

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
NODUS_HEADLESS_API int32_t nodus_headless_declare_input_port_ex(
    NodusHeadlessGraph* g, const char* name, int32_t module_idx,
    int32_t contact_idx, const NodusHeadlessFifoConfig* config);
NODUS_HEADLESS_API int32_t nodus_headless_declare_output_port_ex(
    NodusHeadlessGraph* g, const char* name, int32_t module_idx,
    int32_t contact_idx, const NodusHeadlessFifoConfig* config,
    int32_t start_at_head);
NODUS_HEADLESS_API int32_t nodus_headless_push_token(NodusHeadlessGraph* g, const char* port_name, const void* bytes, size_t len);
NODUS_HEADLESS_API int32_t nodus_headless_push_token_ex(
    NodusHeadlessGraph* g, const char* port_name, const void* bytes, size_t len,
    NodusHeadlessPublishResult* out_result);
// Native source rows may publish onto a declared output using the same edge
// Explicit process-owned PA consumer. Blocking raw stereo s16le/48000 stdin
// until EOF; starts a device only when invoked. Run in its own process because
// the current headless scheduler has process-global canvas ownership.
// device_id=0 selects SDL's default playback device. No network listener.
NODUS_HEADLESS_API int32_t nodus_headless_pcm_playback_stdin(uint32_t device_id);

// Native source rows may publish onto a declared output using the same edge
// transport. Declare/configure all ports before launching persistent workers.
NODUS_HEADLESS_API int32_t nodus_headless_publish_output_token(
    NodusHeadlessGraph* g, const char* port_name, const void* bytes, size_t len,
    NodusHeadlessPublishResult* out_result);
NODUS_HEADLESS_API int32_t nodus_headless_pull_token(NodusHeadlessGraph* g, const char* port_name, void* out_bytes, size_t cap, size_t* out_written);
NODUS_HEADLESS_API int32_t nodus_headless_get_port_fifo_config(
    NodusHeadlessGraph* g, const char* port_name, int32_t is_output,
    NodusHeadlessFifoConfig* out_config);
// Deterministic name-sorted enumeration of process-local named ports. The
// endpoint and start policy come from the same binding used for transport;
// FIFO values are read from the live table edge.
NODUS_HEADLESS_API int32_t nodus_headless_get_port_count(
    NodusHeadlessGraph* g, int32_t is_output);
NODUS_HEADLESS_API int32_t nodus_headless_get_port_at(
    NodusHeadlessGraph* g, int32_t is_output, int32_t ordinal,
    char* out_name, int32_t out_name_capacity,
    int32_t* out_module_idx, int32_t* out_contact_idx,
    int32_t* out_start_at_head, NodusHeadlessFifoConfig* out_config);
NODUS_HEADLESS_API int32_t nodus_headless_get_port_unread(
    NodusHeadlessGraph* g, const char* port_name, int32_t is_output,
    int32_t* out_unread);

// Contract ports: a canvas-owned loopback-socket transport for a contract
// negotiator pair. The negotiator and its `<prefix>.market` / `<prefix>.worker`
// plugin rows are process-owned (contract_negotiator_registry.h); the canvas
// owns the result edge between the rows and a 127.0.0.1 listener. Hosts
// (Hello role 1) and engines (Hello role 2) that connect and say Hello are
// bound to that negotiator inside nodus_headless_step. Persisted by
// nodus_headless_save_file (CONTRACT_PORT trailer record) and re-declared live
// by nodus_headless_load_file. A peer whose socket closed is removed from the
// negotiator at the next step (ContractNegotiator::remove_engine/remove_host), so
// its id can connect again.
//
// Create the process-owned pair (market/worker rows) so a caller that cannot
// reach the C++ registry (e.g. Python) can bind plugin rows by prefix.
// Idempotent; returns 1 on success.
NODUS_HEADLESS_API int32_t nodus_headless_acquire_contract_pair(const char* prefix, int32_t capacity);
// Same, with the loan length in market turns (>0; the plain function uses 60). A
// loan that outlives its length is re-offered, so a prefix carrying a compile host
// needs loan_ticks >= market rate x the longest compile it accepts. Acquiring an
// existing prefix with a DIFFERENT loan length returns 0 (the same length is the
// normal shared acquire).
NODUS_HEADLESS_API int32_t nodus_headless_acquire_contract_pair_with_loan(const char* prefix, int32_t capacity, uint64_t loan_ticks);
// Loan length of an acquired pair; 0 if the prefix is unknown.
NODUS_HEADLESS_API uint64_t nodus_headless_contract_pair_loan_ticks(const char* prefix);
// Detaches every compile host of the prefix first, then unregister and destroy the process-owned pair; 1 if it existed. Only call
// after every graph that bound its rows or declared a contract port on it has
// been destroyed (a reload in the same process needs this, because a
// negotiator that already holds contracts refuses a new result transport).
NODUS_HEADLESS_API int32_t nodus_headless_release_contract_pair(const char* prefix);
// Compile host (include/turing_compile_host_tool.h) attached to the negotiator
// of an acquired pair; its Peer is added to that negotiator, so the existing
// `<prefix>.market` row drives it on the lane that steps the canvas. Only the
// compile wait runs on the host's single worker thread. Returns 1 on success; 0
// (the ABI reports failure by return value only) when the prefix has no acquired
// pair, host_id is 0, a host with that id is already attached to the prefix, or
// any string argument is NULL/empty. After a detach the same id may be attached
// again. adopt != 0 adopts the product on the lane after a compile.
NODUS_HEADLESS_API int32_t nodus_headless_attach_compile_host(
    const char* prefix, uint64_t host_id, const char* python_executable,
    const char* nodus_root, const char* turing_root, const char* module_library_root,
    const char* jobs_dir, int32_t adopt);
// Cancels a running job (the host's cancel path), joins its thread, destroys the
// host before returning. 1 if a host was attached.
NODUS_HEADLESS_API int32_t nodus_headless_detach_compile_host(const char* prefix, uint64_t host_id);
// out5 = {accepted, completed, refused, acknowledged, in_flight}; 0 if not attached.
NODUS_HEADLESS_API int32_t nodus_headless_compile_host_stats(
    const char* prefix, uint64_t host_id, uint64_t* out5);
// Cron tool (include/cron_tool.h) attached to the negotiator of an acquired pair as an
// ENGINE: its Producer is added to that negotiator, so the existing `<prefix>.worker`
// row asks it for due work on the lane that steps the canvas, and any host on the
// prefix performs the task. jobs_dir holds cron.table (the schedule, read here),
// cron.ledger, cron.log and one <contract id>.request.json per run. Returns 1 on
// success; 0 when the prefix has no acquired pair, engine_id is 0 or already an engine
// of the prefix, jobs_dir is NULL/empty, or the table is missing or malformed (the
// reason is then the last line of <jobs_dir>/cron.log when that file can be written).
NODUS_HEADLESS_API int32_t nodus_headless_attach_cron(
    const char* prefix, uint64_t engine_id, const char* jobs_dir);
// Removes the engine from the negotiator and destroys the tool. A run that is out
// stays recorded as started in the ledger and is reasoned about at the next attach.
NODUS_HEADLESS_API int32_t nodus_headless_detach_cron(const char* prefix, uint64_t engine_id);
// out8 = {fired, finished, skipped, interrupted, tasks, gap_seconds, in_flight, last_status}.
NODUS_HEADLESS_API int32_t nodus_headless_cron_stats(
    const char* prefix, uint64_t engine_id, uint64_t* out8);
// Declare a contract port on a configured result edge (token_bytes ==
// sizeof(Result), type 0x534A5231, layout 2, dtype 8, top_k 0). listen_port 0
// picks an ephemeral loopback port. Returns the bound port (>0), or 0 when the
// prefix is empty/contains whitespace, capacity <= 0, the edge is not attached
// and configured, the prefix or edge is already declared, the transport cannot
// be bound, or listening fails.
NODUS_HEADLESS_API int32_t nodus_headless_declare_contract_port(
    NodusHeadlessGraph* g, const char* prefix, int32_t capacity,
    int32_t edge_id, int32_t listen_port);
// Returns 1 if `prefix` is declared; any out pointer may be NULL.
NODUS_HEADLESS_API int32_t nodus_headless_get_contract_port(
    NodusHeadlessGraph* g, const char* prefix, int32_t* out_listen_port,
    int32_t* out_hosts, int32_t* out_engines);
// Declaration-order enumeration of contract ports.
NODUS_HEADLESS_API int32_t nodus_headless_get_contract_port_count(NodusHeadlessGraph* g);
NODUS_HEADLESS_API int32_t nodus_headless_get_contract_port_at(
    NodusHeadlessGraph* g, int32_t ordinal, char* out_prefix, int32_t cap,
    int32_t* out_capacity, int32_t* out_edge_idx, int32_t* out_listen_port);

// Scheduler-produced scalar stack inspection. Snapshot rows exist for native
// StackDisplay rows. These calls are read-only and must follow a completed
// headless step; they do not persist or restore runtime stack state.
NODUS_HEADLESS_API int32_t nodus_headless_get_stack_snapshot(
    NodusHeadlessGraph* g, int32_t module_idx, int32_t row_idx,
    float* out_values, int32_t capacity, int32_t* out_count);
NODUS_HEADLESS_API int32_t nodus_headless_get_stack_snapshot_meta(
    NodusHeadlessGraph* g, int32_t module_idx, int32_t row_idx,
    int32_t* out_type_ids, int32_t* out_element_bytes,
    int32_t capacity, int32_t* out_count);
NODUS_HEADLESS_API int32_t nodus_headless_get_stack_tail(
    NodusHeadlessGraph* g, int32_t module_idx,
    float* out_values, int32_t capacity, int32_t* out_count);

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
