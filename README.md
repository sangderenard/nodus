# Nodus

Nodus is a C++20 graph editor and runtime. What began as a compact canvas/table ABI snapshot now includes graph execution, plugin and repository ingestion, headless access, tensor-backed edge data, portable kernel IR, and browser-facing inspection surfaces.

## Major surfaces

- `include/` and `src/`: canvas, table, graph-runtime, tool, memory-backend, and headless APIs and implementations.
- `src/common/tensors/`: handle-based tensor abstraction, backends, pooling, structured tensor mathematics, and k-path geometry/raster operations.
- `plugins/`: dynamically loaded tools and package examples.
- `module_library/`: graph-collapse/module tooling.
- `tools/`: native frontends and demos.
- `tests/`: C++ and Python regression coverage.
- `docs/`: focused design notes and integration contracts.

The shared library retains the historical name `canvas_tables`; do not infer from that name that the project is limited to canvas and table widgets.

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
```

Vcpkg is discovered through `VCPKG_ROOT`, `VCPKG_INSTALLATION_ROOT`, `vcpkg` on `PATH`, or common installation locations. CMake presets are also available. The working tree is under active development, so build targeted components before assuming a full-tree build is green.

## Run and integrate

- `python demo_canvas_tables.py --mode server --port 8000` runs the historical canvas/table demo.
- `nodus_headless_bridge.py` exposes the headless runtime to Python.
- [`docs/BUILD_YOUR_OWN_PLUGIN.md`](docs/BUILD_YOUR_OWN_PLUGIN.md) introduces the plugin boundary.
- [`docs/module_library_build.md`](docs/module_library_build.md) covers generated module tools.

Set `CANVAS_TABLES_LIB` to override shared-library discovery when needed.

## Cross-repository context

- [`../NODUS_PLUCK_HANDOFF.md`](../NODUS_PLUCK_HANDOFF.md): Nodus ↔ Pluck integration.
- [`../NODUS_TENSOR_CORE_EXTRACTION_HANDOFF.md`](../NODUS_TENSOR_CORE_EXTRACTION_HANDOFF.md): dated tensor-substrate extraction status.
- [`../research/README.md`](../research/README.md): Turing ↔ Nodus tensor and translation research. Its conclusions are orientation, not a substitute for checking current source.

## Historical note

The old description of Nodus as a “standalone snapshot of the canvas + table ABIs” describes its origin, not its present scope. `README_SCAFFOLD.md` preserves related early context.
