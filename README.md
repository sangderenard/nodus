Purpose
-------
This folder is a standalone snapshot of the canvas + table ABIs and their minimal implementations.
You can copy this folder into a new repository and build a small shared library plus the demo.

What's included
- ABI headers: `table_abi.h`, `canvas_abi.h`, `menu_waveform_abi.h`
- Implementations: `table_abi.cpp`, `canvas_abi.cpp`, `menu_waveform_abi.cpp`, `menu_waveform.cpp`, `rope_sim.cpp`, `text_render_helper.cpp`, `table_node_groups.cpp`
- Helpers: `rope_sim.h`, `text_render_helper.h`, `table_node_groups.h`, `menu_waveform.hpp`
- Demo: `demo_canvas_tables.py`
- Build: `CMakeLists.txt`
- Third-party: `stb_easy_font.h` (from https://github.com/nothings/stb)

Build (from this folder)
1. `cmake -S . -B build`
2. `cmake --build build --config Release`

Run the demo (from this folder)
- `python demo_canvas_tables.py --mode server --port 8000`
- Optional: set `CANVAS_TABLES_LIB` to a full path if you want to override library discovery.

Notes
- The library name is `canvas_tables` (e.g., `canvas_tables.dll`, `libcanvas_tables.so`).
- `text_render_helper.cpp` requires `stb_easy_font.h`, which is included in this folder.
