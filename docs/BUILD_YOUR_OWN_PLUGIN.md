# Build Your Own Plugin (Tools)

This guide explains how to write a plugin tool for nodus by hand. It covers the minimal ITool implementation, lifecycle, reserving module frame ports (UUIDs), and customizing the tool row/table rendering.

**Quick summary**
- Implement `ITool` (see `tool_api.h`).
- Export `create_tool()` and `destroy_tool()` (and optionally `plugin_init(HostAPI*)`/`plugin_shutdown()`).
- Declare ports via `port_count()` and `port_spec()` and use `execute_stack()` to consume/produce values.
- Draw table rows by implementing `render_table(TableRenderArgs&)` and calling `gp_table_raster_rgba()`.
- Reserve/advertise module frame ports using `gp_canvas_generate_port_uuid()` and `gp_canvas_set_module_frame_port_uuid()` where appropriate.

## Minimal plugin skeleton

```cpp
#include "tool_api.h"
#include "table_abi.h"   // optional: for GP_TableRow/Gp_TableColumn helpers
#include "canvas_abi.h"  // optional: to call gp_canvas_* helpers

class MyTool : public ITool {
public:
    std::string id() const override { return "my_custom_tool"; }
    std::string name() const override { return "My Custom Tool"; }
    ToolCaps caps() const override { return ToolCaps::Table | ToolCaps::Text; }

    void initialize(const ToolInitContext& ctx) override {
        // ctx.user may be provided by host; you can call gp_canvas_get_singleton()
        // to get a GP_CanvasContext* if you need to reserve ports.
        (void)ctx;
    }

    void shutdown() override {}

    void tick(double /*dt*/, HostAPI& /*host*/) override {}

    void render(RenderContext& /*ctx*/) override {}

    void render_table(TableRenderArgs& args) override {
        // Fill GP_TableRow array then rasterize into args.frame_buffer
        int rows = args.rows;
        int cols = args.cols;
        std::vector<GP_TableColumn> columns(static_cast<size_t>(cols));
        for (int i = 0; i < cols; ++i) {
            columns[static_cast<size_t>(i)].kind = GP_TABLE_CELL_TEXT;
            columns[static_cast<size_t>(i)].width_px = static_cast<int>(std::max(8.0f, args.cell_w));
        }

        std::vector<GP_TableRow> r(static_cast<size_t>(rows));
        // populate rows[...].cells[...] etc.

        GP_TableStyle st{};
        st.width_px = static_cast<int>(std::max(1.0f, args.cell_w * args.cols));
        st.row_h_px = static_cast<int>(std::max(8.0f, args.cell_h));

        GP_TableGeom geom{};
        gp_table_raster_rgba(r.data(), rows, columns.data(), cols, &st,
                             reinterpret_cast<uint8_t*>(args.frame_buffer), args.pitch, &geom);
    }

    int32_t port_count() const override { return static_cast<int32_t>(kPortCount); }
    ToolPortSpec port_spec(int32_t idx) const override { return kPorts[idx]; }

    void execute_stack(ToolStackContext& ctx) override {
        // read args via tool_stack_pop(ctx.stack); push results with tool_stack_push
    }

private:
    static constexpr ToolPortSpec kPorts[] = {
        {ToolPortKind::Argument, 2},
        {ToolPortKind::Internal, 4},
        {ToolPortKind::Return, 1},
    };
    static constexpr int kPortCount = static_cast<int>(sizeof(kPorts) / sizeof(kPorts[0]));
};

#if defined(_WIN32)
extern "C" __declspec(dllexport) ITool* create_tool() { return new MyTool(); }
extern "C" __declspec(dllexport) void destroy_tool(ITool* t) { delete t; }
extern "C" __declspec(dllexport) int plugin_init(HostAPI* host) { (void)host; return 1; }
extern "C" __declspec(dllexport) void plugin_shutdown() { }
#else
extern "C" ITool* create_tool() { return new MyTool(); }
extern "C" void destroy_tool(ITool* t) { delete t; }
int plugin_init(HostAPI* host) { (void)host; return 1; }
void plugin_shutdown() { }
#endif
```

## Lifecycle notes
- `create_tool()` is called by the host (see `PluginLoader::load_module`). The host will instantiate a temporary tool to discover `id()/name()/caps()` and later register a factory that calls your `create_tool()`.
- `plugin_init(HostAPI*)` is invoked once after the DLL is loaded; the pointer can be retained to draw text via `HostAPI::draw_text()` or to `log()`.
- The host will call your tool instance methods (e.g., `render_table`) when displaying tool rows.

## Declaring and reserving ports

Tool ports allow your tool to provide argument/return/internal slots the host or other components may bind to.

- Implement `port_count()` and `port_spec(idx)` to return a short array of `ToolPortSpec` entries. Each entry is a contiguous block of floats with a `ToolPortKind`:
  - `Argument` — inputs to the tool from connected contacts
  - `Internal` — internal scratch/register storage
  - `Return` — outputs produced by the tool

Example declaration (see skeleton above):

- After the host creates a tool instance and places it into a module's stack/rows, the host may allocate or map your declared ports to physical frame LED ports. To reserve a module frame port (so it shows up as an addressable LED slot), use the canvas API:

```cpp
GP_CanvasContext* cvs = gp_canvas_get_singleton();
uint64_t pu = gp_canvas_generate_port_uuid(cvs);
// module_idx, is_send(1)/receive(0), col(0=left,1=right), led_idx (0..15)
gp_canvas_set_module_frame_port_uuid(cvs, module_idx, /*is_send*/0, /*col*/0, /*led_idx*/3, pu);
```

- The host keeps a mapping from frame-port UUIDs to module/row/LED indices to enable deterministic wiring and serialization. Plugins should generate UUIDs centrally via `gp_canvas_generate_port_uuid()` to avoid collisions.

## Customizing the tool row display in tables

- Implement `render_table(TableRenderArgs& args)` to supply an RGBA raster for the tool's table rows. The arguments include:
  - `rows, cols` — the logical grid the renderer should produce
  - `cell_w, cell_h` — per-cell geometry
  - `frame_buffer` — pointer to the caller-provided RGBA buffer
  - `pitch` — bytes per row

- Typical approach:
  1. Build an array of `GP_TableColumn` and `GP_TableRow` structures and populate `rows[].cells[]` with `GP_TABLE_CELL_*` kinds and values.
  2. Configure a `GP_TableStyle` (width, row height, palette colors, name width) to match the host's table style.
  3. Call `gp_table_raster_rgba(rows.data(), row_count, cols.data(), col_count, &style, out_rgba, out_len, &geom)` to rasterize into `args.frame_buffer`.

See `tools/edge_introspect_tool.cpp` for a concrete example of `render_table()` that builds rows and calls `gp_table_raster_rgba()`.

## LED rendering and per-LED hooks

- For finer LED-region drawing, implement `render_led_cell(LEDRenderArgs& args)`. The host will invoke this when painting LED strips for modules or table cells. The args provide a small region (region_x/region_y/region_w/region_h), a palette pointer, and a `connection_handle` you can use as an opaque id.

## Stack execution

- Tools participate in the module stack VM by implementing `execute_stack(ToolStackContext& ctx)`.
- Use `tool_stack_pop(ctx.stack)` to pop floats and `tool_stack_push(ctx.stack, value)` to push results. The order and meaning of stack values must match the `ToolPortSpec` layout your tool declares.

## Serialization

- Implement `serialize(OutputArchive&)` and `deserialize(InputArchive&)` if your tool has persistent state to save/restore. The generated module sources show a pattern reading `ctx.serialized_path` in `initialize()` and calling `deserialize`.

## Building and loading

- You can build a plugin DLL manually (CMake or MSVC). The repository includes `gp_plugin_build_module_and_load()` helper for dynamic single-file building in development workflows.
- The host's `PluginLoader::load_module(path, host_ptr)` will:
  1. Load your DLL, look up `create_tool()` and optional `plugin_init()`.
  2. Instantiate a temporary `ITool` to discover `id()`/`name()`/`caps()`.
  3. Register a unique id and factory in the `ToolRegistry` so your tool appears as a selectable plugin.

## Example: reserving 4 receive ports for mouse handling

Place this in `initialize()` or when you know the target `module_idx`:

```cpp
GP_CanvasContext* cvs = gp_canvas_get_singleton();
if (!cvs) return;
int module_idx = /* choose module index (e.g., focused or root) */;
for (int li = 0; li < 4; ++li) {
    uint64_t pu = gp_canvas_generate_port_uuid(cvs);
    // bind to receive left column, led index li
    gp_canvas_set_module_frame_port_uuid(cvs, module_idx, 0 /*receive*/, 0 /*left*/, li, pu);
}
```

Notes: it's best to coordinate module selection with the host UI (for example, reserve ports after user explicitly attaches the plugin to a focused module) so the plugin does not attempt to claim ports for arbitrary modules.

## Tips and common pitfalls
- Keep `id()` stable and short; host will append a timestamp/version when loading the DLL so multiple loads can coexist.
- Use `gp_canvas_generate_port_uuid()` for all generated UUIDs (host maintains maps and serialization relies on them).
- Keep `port_spec()` small and predictable — plugin port ordering must match stack semantics implemented in `execute_stack()`.
- If you need canvas runtime pointers, call `gp_canvas_get_singleton()` from the plugin — many `gp_canvas_*` helpers are available in `canvas_abi.h`.

## References and examples in this repository
- Tool API: `tool_api.h`
- Canvas/table helpers: `canvas_abi.h`, `table_abi.h`
- Plugin loader: `plugin_loader.cpp`, `plugin_manager.h`
- Example tool showing table rendering: `tools/edge_introspect_tool.cpp`
- Generated module tool example: see `module_library_actualizer.cpp` for generated tool sources and the pattern used for ports and exports.

If you want, I can generate a ready-to-build sample plugin CMakeLists and a minimal plugin source file that demonstrates reserving ports and rendering a custom row. Want me to add that to the repo?
