# Nodus GUI control map (2026-11-10)

Read from source only. Nothing was run, built or launched. "INFERRED" = derived by reading or by arithmetic on the coded geometry, not observed.
All line numbers are in the working tree at HEAD adc3adb (+ uncommitted contract/negotiator edits, which were not touched).

## 0. Where the GUI actually lives

There are two hosts. Both call the same native library; the controls are NOT drawn by either host.

| Host | File | What it does |
|---|---|---|
| Web (the one people use) | `demo_canvas_tables.py` | `python demo_canvas_tables.py --mode server --host H --port N [--width 1000 --height 720 --step-dt 0.016667 --canvas FILE --plugin DLL]` (parse_args 818-835). Loads `canvas_tables.dll` (find_canvas_library 86-116: `$CANVAS_TABLES_LIB`, build/Release, build/Debug, build, repo root; the Windows names tried are canvas_tables.dll then c_menu.dll). |
| SDL3/GL | `tools/frontend_shell.cpp` | Window "Canvas Frontend Shell" 1280x720 (1659). Blits `gp_canvas_raster_rgba` into an SDL texture, forwards input to the same `gp_canvas_on_*` calls. Section 7. |
| Native library (draws + hit-tests EVERYTHING) | `include/inl/canvas_abi_*.inl` (included by `src/canvas_abi.cpp`), labels in `include/labels.h` | Sections 1-6. |

### Web server protocol (demo_canvas_tables.py)

- `GET /` HTML page (img + JS). `GET /frame` PNG of `gp_canvas_raster_rgba` (raster_png 308-315; each GET re-rasterizes). `GET /meta` -> `{"width","height","step_dt"}` (542-553).
- `POST /event` (623-654). Body JSON fields: `type` (string), `x` (int, default -1), `y` (int, default -1), `dx` (float 0), `dy` (float 0), `button` (int 0), `scroll` (float 0). Events with x,y outside `[0,width)x[0,height)` are silently dropped (`handled:false`). Reply `{"handled": bool}` (the return of the native call). 503 if library unloaded, 400 on malformed JSON.
- `type` values honoured by `CanvasBackend.handle_mouse` (282-306): `down` -> `gp_canvas_on_mouse_down(x,y,dx,dy,button)`; `up` -> `gp_canvas_on_mouse_up`; `move` -> `gp_canvas_on_mouse_move(x,y,dx,dy)`; `scroll` -> `gp_canvas_on_mouse_scroll(x,y,dx,dy,scroll)`; `click` -> `gp_canvas_on_click(x,y)` (int coords). Any other `type` returns handled=false.
- No key event route exists in the server (`gp_canvas_on_key` is never called from this file). The shipped page JS (426-440) only sends `{type,x,y}` for down/up and for move while the button is held; it never sends dx/dy, button or scroll.
- To press a control programmatically: POST `{"type":"down","x":X,"y":Y}` then `{"type":"up","x":X,"y":Y}` (or one `{"type":"click",...}`). Note `down` already calls `gp_canvas_on_click` internally (canvas_abi_create.inl:894), so a down+up+click triple fires a toggle control up to twice; the page's own comment (JS ~441-446) says the same. Use `down` alone (or `click` alone) for a button. `up` on an overlay/lasso/edge has separate meaning (section 2 gestures).
- `POST /unload`, `/load`, `/reload`: free/reopen/rebuild the native DLL (354-620). Not canvas controls.
- Time: the server owns a Python thread `_step_loop` (267-271) calling `gp_canvas_step(ctx, step_dt)` then `time.sleep(step_dt)`; default step_dt = 1/60 s. This is the only clock the web GUI has. Style calls at startup: `gp_canvas_set_cable_style(5,2)`, `gp_canvas_set_edge_hues` 3 hues (254-256).
- Canvas size: `gp_canvas_create(width,height)` (default 1000x720). If `--canvas` is given the file is loaded then `gp_canvas_set_size` re-applied (237-247). `gp_canvas_create` itself (canvas_abi_create.inl:16-41) ALSO auto-loads `canvas_workspace.txt` from the cwd if present, sets autosave to that file every 2.0 s, creates the ROOT module, starts the ThreadManager in `Mode::Scheduled`.

## 1. Layout, in drawing order (all in `gp_canvas_raster_rgba`, canvas_abi_overlay_raster.inl:698)

Coordinates assume a 1000-wide frame; they are my arithmetic from the coded formulas (INFERRED numbers, formulas are read). Heights: rope bar `rope_bar_h`=28 (canvas_abi_context.inl:149), control bar `control_bar_h`=56 (:325).

1. Rope bar, y 0..28 (overlay_raster:717-840). Colour {22,22,28}. Buttons 20x20, spacing 8, y=4, fill {60,60,72}.
2. Control bar, y 28..84, colour {28,28,34} (843). Two rows: row 1 y=32 (`by0`), row 2 y=58 (`by1`); each button is square, side `bh` = (48-4)/2 = 22, spacing 8.
   - Row 1 (848-1330): canvas tools, edge tools, [order, C/P, counter] groups, Save/Clear, table tools.
   - Row 2 (1331-1635): KPN tools, 8 subgroup swatches, ROOT, delay, TIME, exec modes, CLN/CLR/X (decorative), SIM|play.
3. Modules (1637-2236): first pass renders tables and gathers hit boxes; second pass draws each module background, its table, then `draw_module_top_ui` (canvas_abi_rope_map.inl:2452) with its header controls, focus frame (blue {60,120,220}, 2px), preview.
4. Overlays (2238), ropes/edges (2639-2760), rings (2787), meta-edge splines (2920), dangling widgets (3120).
5. Floating menus, drawn last = on top: rope menu (3340), tool menu (3401), module/LIB menu (3560), plugin/Menu menu (3605).

### Why labels overlap (answer to the owner's observation)

Each toolbar button is a 22 px square but is drawn with TWO text blits: the FULL label ("Select", "New Table", "Edge Mode", "New Stage", "Edge Create", ... "Meta-Edge Lasso", "Save", "Clear", "Tbl Select", "Tbl Edit", "Tbl More") at scale 0.85-0.95, centred on the button (`tx = bx_i + (bw - lbm.width)/2`, e.g. overlay_raster:~870-880, ~950-960, ~1070), then a SHORT letter ("S","T","E","G"; "C","X","O","N","L"; "SV","CL"; "s","e","m") at scale 1.05-1.1 on top. The full text is far wider than 22 px and the gap is 8 px, so neighbouring labels overprint each other and the short letters. Strings are in `include/labels.h`. This is the source of "Tug/Elec/Chip/Sort/Chan" style fragments: none of those strings exist in the source (git grep of `"Tug"`, `"Elec"`, `"Chip"`, `"Sort"`, `"Chan"`, `"CAT"`, `"elm"`, `"Clear Tbl"`, `"Select Tbl"`, `"Edit Tbl"` over tracked files found nothing); INFERRED they are mis-readings of overprinted real labels (e.g. "Clear Tbl" = "Clear" over "Tbl Select"; "CAT" = "CMT"; "K" = "X"; "elm" = lowercase "sim"; "+0" = "+" next to the order value box 0). If a DLL built from another tree was run, the labels could differ: unverified.

## 2. Control table (native canvas). Handler = file:line; call = what it ends in

Hit-test lives in `gp_canvas_on_click` (canvas_abi_create.inl:194-888), re-derives the same geometry as the raster, and for toolbar buttons calls `canvas_dispatch_root_action(c, ACTION)` (canvas_abi_rope_map.inl:4918) = `gp_table_dispatch_hit(root_table, hit{aux0=ACTION})` -> the callback installed at `canvas_install_root_actions` (4360), switch at 4379-4813. `gp_canvas_on_mouse_down` (890) calls on_click first. Action ids are the enum at rope_map.inl:36-110. "Toggle" tools: clicking the selected one deselects.

### 2a. Top rope bar (click 208-238)

| Control (as drawn) | Position | Look | Action id / handler | Does |
|---|---|---|---|---|
| "Seg -" | x 8..28 | grey {60,60,72} 20px | SIM_SEGS_DEC 2000 -> rope_map:4380 | `c->sim_segs = max(2, sim_segs-1)`; `canvas_reinstantiate_all_ropes` |
| "Seg +" | x 36..56 | same | SIM_SEGS_INC 2001 -> 4385 | `sim_segs = min(64, +1)`; reinstantiate ropes |
| "SIM" | x 64..100 (36 wide) | lighter {70,70,84} when open | ROPE_MENU_TOGGLE -> 4398 | toggles `rope_menu_open`, closes other menus. NOT a simulation switch: it opens the "Rope Sim" menu (2e) |
| readout "segs:8 slack:0.000000" | centred | text {220,220,220} | none (display) | prints `sim_segs`, `sim_slack` (overlay_raster:~729) |
| "Slack -" | x w-56..w-36 | grey | SIM_SLACK_DEC 2002 -> 4390 | `sim_slack = max(0, -0.1)` (not re-instantiated) |
| "Slack +" | x w-28..w-8 | grey | SIM_SLACK_INC 2003 -> 4394 | `sim_slack = min(8, +0.1)` |

### 2b. Control bar row 1 (y 32..54). Order left to right

| Control | x | Colour (unselected / selected) | Action -> handler | Does |
|---|---|---|---|---|
| Canvas tool 0 "Select" ("S") | 8 | {60,60,72} / {90,90,110} | TOOL_CANVAS_0 2010 -> 4433 | `selected_tool_canvas` toggle (0 = default select) |
| Canvas tool 1 "New Table" ("T") | 38 | same | TOOL_CANVAS_1 -> 4433 | while selected, a click on empty canvas spawns a "Table" module (on_click 706-723: `gp_canvas_add_module` + `gp_canvas_create_table`) |
| Canvas tool 2 "Edge Mode" ("E") | 68 | same | TOOL_CANVAS_2 -> 4433 | enables LED-to-LED edge making without table selection side effects (on_click 649) and background drag = pan viewport (mouse_down 1041-1051) |
| Canvas tool 3 "New Stage" ("G") | 98 | same | TOOL_CANVAS_3 -> 4433 | click on empty canvas spawns a "Stage" module (724-739, `canvas_configure_stage_module`) |
| Edge tool 0 "Edge Create" ("C") | 136 | {60,54,48} / {90,80,70} | TOOL_EDGE_0 2014 -> 4442 | `selected_tool_edge` toggle; 0 = plain create (clicking two LEDs makes an edge) |
| Edge tool 1 "Edge Destroy" ("X") | 166 | same | TOOL_EDGE_1 -> 4442 | click a rope or an LED with edges -> `canvas_remove_edge_at` (on_click 678-688; LED path rope_map ~5883) |
| Edge tool 2 "Edge On-Change" ("O") | 196 | same | TOOL_EDGE_2 -> 4442 | `canvas_set_delta_mode_for_edge` -> `gp_table_edge_set_delta_mode(root, idx, 1)` |
| Edge tool 3 "Edge Continuous" ("N") | 226 | same | TOOL_EDGE_3 -> 4442 | same call with delta_mode 0 |
| Edge tool 4 "Meta-Edge Lasso" ("L") | 256 | {70,58,52} / {110,70,80} | TOOL_EDGE_4 2018 -> 4453 | toggles `lasso_mode`; with it on, mouse drag collects `lasso_points`, mouse-up builds a meta-group over crossed ropes (rope_map:1186-~1500) |
| Order "-" | 492 | {50,50,56} | EDGE_ORDER_DEC 2019 -> 4464 | `edge_order_value` clamp [-4,4] |
| Order value box | 524 | {36,36,42}, shows `+0` style value | display | |
| Order "+" | 578 | | EDGE_ORDER_INC 2020 | |
| "Order" | 610 | {70,58,52}/active | EDGE_ORDER_TOOL 2021 -> 4471 | toggles `edge_order_tool_active`; while on, clicking a rope (on_click 663-669) or an LED with edges applies `canvas_set_order_mode_for_edge` -> `gp_table_edge_set_order_mode` |
| "C" (consumer add) | 648 | | IO_CONSUMER_ADD 2042 -> 4513 | on FOCUSED module: `canvas_append_io_row(is_input=true, io_attachment_count)` (4945) |
| "P" (producer add) | 680 | | IO_PRODUCER_ADD 2043 | same, output row |
| counter "-" | 718 | | IO_COUNT_DEC 2040 -> 4498 | `io_attachment_count` clamp [1,32] |
| counter value box, shows "1", short label "n" | 750 | | display | attachments per new IO row |
| counter "+" | 804 | | IO_COUNT_INC 2041 | |
| "Save" ("SV") | 842 | {50,62,54} | SAVE 2004 -> 4421 | `gp_canvas_save_to_file(autosave_path or "canvas_workspace.txt")`. Writes CANVAS V4 only, no EDGE_FIFO/CONTRACT_PORT trailer |
| "Clear" ("CL") | 872 | same | CLEAR 2005 -> 4428 | `gp_canvas_clear_meta_and_overlays` (overlay_raster:539): clears overlays, meta-groups, modules. No confirmation |
| "Tbl Select" ("s") | 910 | {60,50,60} / {90,70,90} | TOOL_TABLE_0 2030 -> 4477 | `selected_tool_table` toggle (mode flag read by table-side tools) |
| "Tbl Edit" ("e") | 940 | same | TOOL_TABLE_1 | same |
| "Tbl More" ("m") | 970 | same | TOOL_TABLE_2 -> 4483 | toggles `tool_menu_open` = the Tool Menu (2f); closes other menus |

### 2c. Control bar row 2 (y 58..80)

| Control | x | Look | Action -> handler | Does |
|---|---|---|---|---|
| KPN "K" / "P" / "N" (labels "KPN K" etc.) | 8, 38, 68 | {52,58,70}/{80,90,110} | TOOL_KPN_0..2 2060-2 -> 4576 | sets `selected_tool_kpn` (-1 none). INFERRED: only stored/saved, no reader found in this pass; I did not trace consumers |
| Swatch row: 8 buttons each with a coloured LED disc to its right | 106 + 30*i (i 0..7) | button grey, white border when selected; LED = `subgroup_target_rgba[i]`; red, amber, green, green, cyan, blue, purple, magenta per the owner's read (palette set in `canvas_init_subgroup_palette`, canvas_abi_primitives.inl / create.inl:20) | TOOL_SUBGROUP_0..7 2080-7 -> 4584 | toggles bit i of `selected_tool_subgroup_flags`. While non-zero, clicking a rope/LED applies the flags to the edge: `canvas_apply_subgroup_to_edge_idx` -> `gp_table_edge_set_subgroup_flags(root, idx, flags)` (on_click 670-677). This is the FIFO "subgroup" selector for edges |
| LED discs at the swatch row ends / "small boxes at both ends" | | | click on a disc: `c->toolbar_leds` hit (on_click 428-453) | prints "connect mode disabled" and returns 1: INERT (rope-from-toolbar disabled by design, comment 429-433). The boxes at the ends of the row are the "ROOT" button and the delay control, below |
| "ROOT" | 346 (only drawn if it fits left of the delay control) | {70,70,90} | SPAWN_ROOT 2090 -> `canvas_dispatch_root_action` fast path (rope_map:4922) | `canvas_ensure_root_module` (4856): ensures the synthetic "ROOT" module mirroring the root table (already created by `gp_canvas_create`) |
| delay "-" / value / "+" with short label "ms" | DRAWN at 660 / 692 / 746 (draw_io_group at `bx_delay_new`, overlay_raster:~1546) but HIT-TESTED at 760..782 and 868..890 (on_click 380-397, 476-481) | | THREAD_DELAY_DEC/INC 2071/2072 -> 4726 | `thread_mgr_delay_ms` += -10 / +10, clamp [0,2000]. Used in `gp_canvas_step` to submit a ThreadManager tick only every N ms of accumulated dt (table_step.inl:274-290). INFERRED: draw and hit rectangles do not coincide; the clickable "-" sits on top of the SEQ/POOL button, whose toolbar-LED hit is tested first (on_click 428-452) and wins |
| "TIME" (global clock toggle) | 748 | green {64,120,60} when on, else {48,48,62} | TIMING_TOGGLE 2089 -> 4618 | `ThreadManager::global()->set_timing_enabled(!)`. Only turns on per-module wall-time collection (`ModuleTiming`: run_count, last/total wall seconds), shown appended to each module title. It is a profiler switch, not a clock source |
| "SEQ" "POOL" "SLIP" "FREE" | 774, 800, 826, 852 | {48,48,62} / {80,90,110} | on_click 428-445 sets `thread_mgr_global_exec_mode` directly (0..3, click again = -1) | GLOBAL override of every module's `ThreadManager::ExecMode` (table_step.inl:~308-320): 0 Sequential (manager thread), 1 Pooled (worker pool when ready), 2 Slip (run immediately on workers, ignoring predecessor readiness; "use at your own risk", thread_manager.h:~130). 3 FREE: the code comment says it is a "canvas-only toggle handled elsewhere" and leaves `exec_mode` at the default (Pooled); INFERRED no effect found. A per-module choice beats the global one. None of these is memory/arena related |
| "CLN" "CLR" "X" (toolbar) | 880, 908, 936 | {58,58,70} {64,56,52} {70,52,52} | none | DECORATIVE: drawn by `draw_action_button` (overlay_raster:1280) but `gp_canvas_on_click` has no hit-test for them (comment at on_click 475: "moved into module top UI"). INFERRED from absence |
| "SIM" half of the last button | 970..981 | {46,46,56} | CANVAS_ROOT_SIM_TOGGLE 2088 -> 4698 | toggles root table sim (`gp_table_set_sim_enabled`), sets `sim_root_paused`, and `gp_table_set_global_sim_frame_skip_count(INT32_MAX/4)` to freeze every table's rope sim |
| ">" / "||" half (label "Play"/"Pause") | 981..992 | {70,60,70} paused / {60,80,60} running | KPN_GLOBAL_TOGGLE 2079 -> 4627 | toggles `thread_mgr_paused` (the whole ThreadManager tick submit, table_step:~273). Shows ">" when paused |

### 2d. Module header (per module, `draw_module_top_ui` rope_map:2452; hits on_click 545-870). Control row at module-local y = 6+24+16+4 = 50, buttons 22 high

Module defaults 420x620 (primitives:402); ROOT is `max(160,w-32)` x `max(120,h/3)` at (16,64) (rope_map:4862). Left to right:

| Control | Look | Action -> handler | Does |
|---|---|---|---|
| Title row: label + `[table=..]`, plus timing text when global TIME is on | text 1.05 {220,220,230} | drag: mouse_down 900-921 (drag area `layout.drag_y/drag_h`) | drags the module (`modules[i].x,y`) |
| Thumb row (grey bars) | decor | none | |
| "-" / "0" / "+" (left) | {52,52,64}; "0" is a HARD-CODED string (rope_map:2597) | FRAME_PAIRS_DEC/INC 2200/2201 (on_click 564-571) -> 4505 | `module_frame_pair_count` clamp [1, 32] (this is a CANVAS-WIDE value, `kModuleExtraLedCount*2`); the box never shows it |
| "SEQ" "POOL" "SLIP" "FREE" | {46,46,56}; selected = module-local override | EXEC_MODE_MODULE_0..3 2095-8 (on_click 779-793) -> 4661 | `module_exec_mode[mi]` toggle (-1 = inherit global). Same meaning as 2c |
| "TIME" | {52,52,64}/green | hit box pushed at overlay_raster:2151-2190 with aux0=0 | INFERRED INERT AND HIDDEN: it is drawn at `btns_left-28`, exactly where the "FREE" button is drawn afterwards (same x and width), so FREE covers it; and aux0=0 matches no root action |
| "CLN" | {52,52,64} | MODULE_CLONE 2073 -> 4733 | `canvas_clone_module(focused)` (NB acts on FOCUSED module) |
| "CLR" | | MODULE_CLEAR 2074 -> 4741 | `canvas_clear_module_table(focused)` |
| "X" | | MODULE_DESTROY 2075 -> 4748 | `canvas_destroy_module(focused)` |
| "CMT" | {48,48,56} | MODULE_COMMIT 2077 -> 4755 | `canvas_clear_module_bindings` then queue `pending_module_commits`; processed in `gp_canvas_step` (table_step:~213-258): `gp_canvas_export_module_to_root`, manifest write, `gp_module_library_actualize_sources`, then `gp_plugin_build_module_and_load` (a compiler run) and `canvas_refresh_plugin_tools` |
| "EXP" | {44,60,48} | MODULE_EXPORT 2076 -> 4765 | `gp_canvas_export_module_to_root` + `canvas_write_module_library_manifest` (writes the module into the module library) |
| "sim"/"SIM" (left half of 44px) | | THREAD_SIM_TOGGLE 2078 -> 4637 | `module_sim_enabled[mi]` flip -> `gp_table_set_sim_enabled(module table)`; lower case = off |
| ">" / "||" (right half) | | THREAD_TOGGLE 2070 -> 4600 | `module_skip[mi]` flip = skip the whole module's tool/table work in ThreadManager ticks (shows ">" when skipped) |
| "LIB" | {46,54,52} | on_click 819-832 | toggles `module_menu_open`, `canvas_refresh_module_library` (3435) |
| "Menu" | {50,50,62} | on_click 833-846 | toggles `plugin_menu_open`, `canvas_refresh_plugin_tools` (3762) |

POOL, FREE, CLN, CLR, SEQ, SLIP, CAT(=CMT), EXP, LIB, K(=X) are therefore execution-mode (SEQ/POOL/SLIP/FREE), module edit (CLN/CLR/X), library-build (CMT/EXP/LIB) controls. None allocates tensors or touches the arena.

"Small square glyph at the module's lower left": NOT TRACED. Candidates are the module-frame LED cells (`module_frame_leds`, 16 per row, `kModuleExtraLedCount`, canvas_abi_primitives.inl:412) or IO-row LEDs; I did not map a single glyph to a handler.

Module body (not headed buttons): table cells are handled by the table ABI (`gp_table_on_click`, on_click 871-883; `gp_table_on_key` for keys, create.inl:1587). LED cells (`GP_TABLE_HIT_LED`, `_LED_ARG`, `_LED_TABLE`) go to `canvas_handle_module_led_hit` (rope_map:~5864): first click on a producer/consumer LED selects it and creates a prospective rope (`rope_sim_add_rope`); click on an opposite-role LED calls `gp_canvas_add_edge_with_type(type 0)` and attaches the rope. Counter cells of a TableNumber tool row: `canvas_handle_module_counter_hit` (5846) edits `attachment_count` [0,99].

### 2e. "SIM" menu = Rope Sim floating menu (draw 3340, hit on_click 240-268)

| Row | Does |
|---|---|
| "No Sim / No Frills" | `gp_canvas_set_debug_flags(GP_CANVAS_DEBUG_NORENDER_MODE)` + reinstantiate ropes (rope_map:4409) |
| "Full Sim" | `gp_canvas_set_debug_flags(0)` (4415) |
| segs "-" "+" | same as Seg -/+ |
| slack "-" "+" | same as Slack -/+ |

Click outside the menu closes it.

### 2f. "Tbl More" = Tool Menu (draw 3401, hit on_click 484-508). 200 px wide, at top-right below the bars. Items in `kToolMenuItems` (rope_map:3544), each a root action that calls `canvas_push_tool_to_focused(kind, Builtin)` and appends a Tool row to the FOCUSED module:

| Item (LABEL_TOOL_*) | Action | ModuleToolKind | Stack effect (thread_manager.h `module_tool_stack_spec`) |
|---|---|---|---|
| Add / Subtract / Multiply / Divide / Modulo | MENU_TOOL_ADD..MOD 2101-5 | 1-5 | consumes 2 produces 1 |
| TensorAllocator | MENU_TOOL_ALLOCATOR | 12 | spec {0,0}; behaviour in section 5 |
| Keyboard | MENU_TOOL_KEYBOARD 2106 | 6 | produces 1 |
| Mouse | 2107 | 7 | produces 9 |
| Stack | 2108 | 8 | display |
| Clone | 2109 (INFERRED id) | 11 | consumes 2 |
| RectRgba | 2110 (INFERRED) | 9 | consumes 12 produces 1 |
| Number | 2111 (INFERRED) | 10 | produces `attachment_count` |

Below the items: a "Stack" section listing the focused module's rows (clicks consumed, no handler: INFERRED inert), and "Table Number" with "-" "+" and short "n" -> TABLE_TOOL_NUM_DEC/INC 2044/5 (4526) -> `table_tool_number` clamp [0,99]. `ModuleToolKind::TensorTool` (13) has NO menu item; it can only be bound programmatically (headless `nodus_headless_bind_builtin_tool`).

### 2g. "Menu" (plugin menu, "Plugin Tools", draw 3605, hit 509-523)
Items = `plugin_tool_ids` from `canvas_refresh_plugin_tools` (3762): the `*.cpp` stems in `<module library root>/source/tools` that also have a compiled artifact on disk. Empty shows "No plugin tools". Click -> `canvas_push_plugin_tool_to_focused(id)` (5570): adds a Tool row with `ModuleToolOrigin::Plugin` and `tool_registry_global().create(id)`. The registry also holds default-registered rows that do NOT appear here (section 5). Library root: `gp_module_library_default_root()` (src/module_library.cpp:23).

### 2h. "LIB" (module menu, "Modules", draw 3560, hit 524-544)
Items = `*.gpmod` files in `<root>/serialized`, label from the manifest (`canvas_refresh_module_library`, 3435). "No modules" if empty. Click -> `canvas_apply_module_library_entry` (3839) replaces the target module's content with the saved module (rows, edges, table). Not an arena or memory control: it is the saved-module library that CMT/EXP write into.

### 2i. Gestures and keys with no button

| Input | Handler | Does |
|---|---|---|
| Left down on module title/drag area, then move | mouse_down 900-921, mouse_move 1099 | moves module (`modules[i].x/y`, world coords) |
| Down on an overlay (custom LED pair created by lasso), move | 923-1040 / 1112 | moves overlay |
| Overlay numeric box "-"/"+" (below the overlay's mode button) | mouse_down 928-1000; mouse_up 1170-1260 | `CANVAS_ACT_META_CHAN_DEC/INC` 2500/2501 (rope_map:4533) -> `gp_table_meta_set_channel_group(table, mg, n)` clamp [-32768, 32767] |
| Click within 12 px of an overlay endpoint on mouse-up | mouse_up 1156-1170 | synthesizes a ROOT-module LED hit (contacts 0/1) -> edge creation |
| Edge Mode tool + drag on empty background | mouse_down 1041-1051, move 1080-1090 | pans the viewport (`offset_x/offset_y`) |
| Lasso mode + drag | rope_map:1186+ | builds a meta-group |
| Mouse wheel | on_mouse_scroll create.inl:52 | ONLY forwards CANVAS_ACT_MOUSE_SCROLL_UP/DOWN to bound ports; no zoom or scroll of the canvas |
| Any mouse down/up/move not consumed | `canvas_dispatch_event_to_bound_ports` (actions_edges.inl:344) | publishes CANVAS_ACT_MOUSE_DOWN/UP/MOVE (3001-3003) as pointer EventPayloads to module-frame ports bound to those actions (see 3) |
| Keyboard (SDL host only) | frontend_shell -> `gp_canvas_on_key` (create.inl:1587) | `canvas_record_key_input`; forwarded to the focused module's table key callback (`gp_table_on_key`); the "Key Recorder" table publishes the key on a root-table edge (rope_map:4225-4262). No global shortcuts exist in the canvas. Not reachable from the web server |

## 3. Ports as the GUI knows them

| Kind | Structure | Created/configured by |
|---|---|---|
| Module contacts (LED rows) | `ModuleIORow` {kind Input/Tool/Output, contact_idx, tool, attachment_count, plugin_id} (thread_manager.h:~58), `module_io_rows`, `module_table_rows`; LED cells in module tables | toolbar "C"/"P" + counter (2b); `gp_canvas_set_module_rows`; contact id = index or `kModuleFrameContactBase`(10000)+row*16+led |
| Module frame ports (16 LEDs x send/receive x left/right) | `module_frame_links[m].ptrs[4][16]`, `module_frame_leds` (`kModuleExtraLedCount`=16) | `gp_canvas_bind_pending_action_to_module`, `gp_canvas_bind_action_enum_to_module_port`, `gp_canvas_unbind_action_from_module_port` (canvas_abi.h:~300-312); click-listen mode (`gp_canvas_set_click_listen_mode`, overlay_raster:603) captures the next root action as a pending action to bind. Not reachable from the GUI buttons |
| Event ports | action ids CANVAS_ACT_MOUSE_DOWN/UP/MOVE/SCROLL_UP/DOWN 3001-3005 (canvas_abi.h:336-341) | bound frame ports receive pointer-mode `EventPayload`s through the root table FIFO (thread_manager.cpp:905-935). The canvas therefore PROVIDES input events as FIFO traffic only to ports someone bound |
| Tool-row ids | `gp_canvas_get_tool_row_id` (a 64-bit key per tool row) | assigned at bind time; used as FIFO edge keys (table_step:~340-370) |
| Edges | `GP_CanvasEdgeDesc{a_module,a_contact_idx,b_module,b_contact_idx}`, `EdgeInfo` (context.inl:20) incl. `type_id`, `subgroup_flags`, `overlay_key_a/b`; backed by root-table FIFO edges (`gp_table_add_edge`, key = module<<32 \| contact<<16) | LED clicks (2d); typed via `gp_canvas_add_edge_with_type`; GUI always type 0 |
| Overlay ports | overlay with two LEDs: `gp_canvas_create_overlay_with_leds` (overlay_raster:10) -> key_a/key_b, `gp_canvas_register_table_overlay(.., port_uuid_a, port_uuid_b)` (298), `gp_canvas_get_overlay_port_uuids` (323), `gp_canvas_attach_rope_to_overlay` (343) | made by the meta-edge lasso and by file load (save_load.inl:966, 1046) |
| Named external ports (headless only) | `nodus_headless_declare_input_port/_output_port[_ex]`, `push_token/pull_token`, FIFO config `NodusHeadlessFifoConfig` (nodus_headless_abi.h:219-246); persisted as `PORT` trailer lines | NOT in the GUI: no button creates or shows them |
| Contract ports | `nodus::contracts::CanvasContractPorts` (include/contract_port_canvas.h:20): per-canvas loopback listener + negotiator `<prefix>.market/.worker` rows; trailer `CONTRACT_PORT`; wire protocol docs/NODUS_BRIDGING_CONTRACT_PORTS_2026-11-10.md | SDL shell only: `--acquire=PREFIX:CAPACITY` then `load_trailer` on workspace load (frontend_shell.cpp:1546-1557); ports pre_step/post_step around `gp_canvas_step` (1463-1465). Printed to stdout as `contract_port PREFIX 127.0.0.1:PORT`; NOT drawn in the canvas. The web server (demo_canvas_tables.py) never calls any of this |
| Attach mode (SDL) | `--attach=127.0.0.1:P --console-token-file=F` read-only viewer of a headless graph's console (protocol nodus-console/1), frontend_shell.cpp:112-425, 2743 | status line on a 16 px black bar (2809) |

There is NO port kind named "system" in the GUI or runtime. Searched: `git grep -i` for "system port", "system_port", "privileg", "emergency", "analytics", "telemetry", "heartbeat", "gratis", "time port", "time_port" across docs/, include/, src/, tools/, plugins/, README*.md. Hits were unrelated (docs/LLM_ENTITY_ENVIRONMENT.md lines 148-149 use "privileged" for an action tier; kernel watchdog comments; a "(hook if you later want more telemetry)" in stream_add.h:323).

## 4. Time as the GUI and runtime know it

| Clock | Where set/shown | Owner |
|---|---|---|
| Frame step `dt` passed to `gp_canvas_step` | web: `--step-dt` (default 1/60) in a Python thread with `time.sleep(step_dt)` (demo_canvas_tables.py:267-271). SDL: `CanvasTickController::thread_loop` (frontend_shell.cpp:1433-1476) calls `gp_canvas_step(ctx, head_thread_dt_[i])`, default 1/60, set by `--canvas-config=IDX:DT:RATIO:FREE`; ticks per render frame = `ratio`; `FREE`=spin without waiting for frames; loop sleeps `--thread-delay` (default 1/60 s); GUI loop sleeps `--gui-delay` (default 1/60) | the host process, not the canvas |
| Tick submission gate | `thread_mgr_delay_ms` (toolbar delay control, 2c; table_step.inl:274-290): accumulates `dt`, submits a `ThreadManager::TickRequest{dt}` only when accumulated >= delay | canvas context |
| Per-row clock | each tick carries ONE `dt` (`TickRequest.dt`, thread_manager.h:~165) to every module/table/stage. This is the "dt on each row's tick" the earlier header search found | ThreadManager |
| Pause | toolbar ">"/"||" `thread_mgr_paused`; per-module `module_skip`; SDL `--pause-canvas`; `gp_canvas_set_thread_manager_paused` | canvas context |
| Rope-sim cadence | `sim_frame_skip_count` / `gp_table_set_global_sim_frame_skip_count`, `gp_table_advance_global_sim_tick` each frame (table_step.inl:205-208, 326-328); toolbar SIM half freezes it | canvas context |
| Per-module cadence | `module_exec_skip_count` (frames to skip, 0 = every frame) propagated into `ModuleContract.exec_skip_count` (table_step:~300). No GUI control sets it | canvas context |
| Scheduling mode | `ThreadManager::Mode::{FreeSpinning, Scheduled}`; `gp_canvas_create` sets Scheduled; `gp_canvas_set_thread_manager_mode`, headless `nodus_headless_set_scheduler_mode`. No GUI control | ThreadManager |
| Execution mode | SEQ/POOL/SLIP/FREE (2c/2d) | canvas context -> ModuleContract |
| Timing profile | "TIME" toggle -> `set_timing_enabled`; `ModuleTiming` wall seconds (thread_manager.h:~222-236) | ThreadManager |
| Display rate (SDL only) | `DisplayState` interval (default 1/30 s) or ratio, `--canvas-display-interval/-ratio` | shell |
| Autosave | 2.0 s (create.inl:34; shell `kCanvasAutosaveIntervalSeconds`) | canvas context |
| Loan deadlines | counted in market turns, default 60 (`loan_ticks`), docs/NODUS_BRIDGING_CONTRACT_PORTS_2026-11-10.md | negotiator |
| Kpath demo `phase_step` | frontend_shell.cpp:473 (Ctrl+Shift+K mode only) | shell |

No privileged or system clock exists as an object. There is no time port, no wall-clock row, no tick counter exposed to tools except `dt`. The only monotonically increasing counters are `ThreadManager::ticks_submitted/completed` (thread_manager.h:~196) and a global sim tick (`gp_table_advance_global_sim_tick`).

## 5. System-provided things

| Thing | Status | Evidence |
|---|---|---|
| Built-in tool kinds `ModuleToolKind` 1-13 | Implemented 1-12 (switch in `ThreadManager` tool dispatch, thread_manager.cpp ~1100-1330); 13 TensorTool is a STUB | `gp_table_tensor_tool_tick` body is only a comment: "Placeholder: tie table tool execution into tensor runtime once defined." (src/table_tensor_tool.cpp:22-23) |
| TensorAllocator row (the closest thing to "a tool that mints tensors") | Implemented, but it RE-MINTS rather than creates from nothing, and silently no-ops without a default backend | thread_manager.cpp:1193-1296: reads the module frame `RawStackFrame` for the contact (`gp_canvas_get_module_frame_ptr_for_contact`), pops every `VT_ABSTRACT_TENSOR` handle plus the int64 dims beneath it, `backend->destroy(old)`, then `default_backend()->create(TensorDesc{F32, Dense, dims})` per recorded shape, pushes new handles back, frees the managed event payload. `if (!backend) break;` (1203). Menu stack spec is {0,0} (thread_manager.h:110). It is fed by a CANVAS_ACT_TENSOR_ALLOC (3102) pending action delivered to a bound frame port (thread_manager.cpp:919); NOTHING in the tree publishes that action (git grep of 3102/CANVAS_ACT_TENSOR_ALLOC: canvas_abi.h:347 and thread_manager.cpp:919 only). INFERRED: with the SDL/web GUI, `default_backend()` is null unless something registered it: `register_in_memory_backend` is called by `Runtime::reset` (tensor_runtime.cpp:13, reached only via the TensorTool state, thread_manager.cpp:52) and by the kpath demo (frontend_shell.cpp:890) |
| Who mints an `AbstractTensorHandle` ({uint64 id}, abstract_tensor_handle.h) | `TensorBackend::create(TensorDesc)` (tensor_backend.h:17); concrete `InMemoryBackend` singleton (in_memory_backend.cpp:2328). Callers: C ABI `nodus_tensor_create` (tensor_abi.cpp:~182) and headless `nodus_headless_tensor_create` (nodus_headless_abi.h:201); `AbstractTensor::create` (abstract_tensor.cpp:85-88) and `AbstractTensorPool` (abstract_tensor_pool.cpp:75,197); `abstract_tensor.*` tool rows mint OUTPUT tensors per call via `nodus_tensor_create` (abstract_tensor_graph_ir.cpp:~163, ~232); kpath code via `in_memory_backend_singleton()`; TensorAllocator via `default_backend()` | implemented |
| Arena / memory manager | Implemented: one process-wide arena, best-fit leases, 64-byte alignment, coalescing free list, clean-lease swap for zeroing; configure before first allocation; env `NODUS_INMEM_ARENA_MB` (default 1024), `NODUS_INMEM_ARENA_SPANS` (262144); static defaults reserve 10 GiB, min commit 256 MiB (in_memory_backend.cpp:48-165). C ABI: `nodus_tensor_arena_configure/_stats/_system_available_bytes/_reset`, `nodus_tensor_map/unmap/zero/write/read/allocation`, unary/binary/scalar/reduce/matmul ops (tensor_abi.h:96-175). No GUI control; not a port |
| Generic tensor FIFO per edge | Implemented: `edge_tensor_fifo.h` (uses `in_memory_backend_singleton()` at :451), docs/edge_tensor_fifo_lockfree_kpn.md, `table_edge_tensor_bridge.*`. Configured per edge, not "system-fed" |
| `abstract_tensor.<op>` tool rows | Implemented, registered at static init (abstract_tensor_graph_ir.cpp:263) and explicitly in `nodus_headless_create` (the static registrar is dropped when `canvas_tables_static` links). Not listed in "Menu" (only on-disk plugin sources are). Coverage deliberately limited to ops with a CT code, arity 1-2, plus `sum`/`mean` |
| Other default-registered rows (`REGISTER_TOOL`) | `edge_introspect` (tools/edge_introspect_tool.cpp:128), `meta_edge_lasso` (tools/meta_edge_lasso.cpp:37), `kpath_utf16_stack_to_toolpath` (src/tools/...:265); registered only if the translation unit is linked into the running image (INFERRED for the web DLL) |
| Contract negotiator rows `<prefix>.market` / `.worker`, Turing compile host | Implemented, opt-in (`--acquire`, `nodus_headless_acquire_contract_pair`); docs/NODUS_BRIDGING_CONTRACT_PORTS_2026-11-10.md |
| "fifos in and out, configurable, on a privileged clock" | PARTLY: configurable FIFOs exist for edges and named headless ports (`NodusHeadlessFifoConfig`, `nodus_headless_configure_edge_fifo`); they run on the ordinary tick `dt`, no privileged clock | |
| Emergency messages, analytics channel, system-fed FIFO to every table | NOT FOUND. No code, no document sentence. Searched terms: emergency, analytics, privileg, gratis, system port/system_port, time port/time_port, heartbeat, watchdog, telemetry, diagnostic over docs/, include/, src/, tools/, plugins/, README.md, README_SCAFFOLD.md; file names matching `*port*`, `*system*`, `*clock*`, `*mint*`, `*arena*`, `*overlay*`, `*tensor*tool*` (git ls-files only; `table_tensor_tool.*` and `contract_port_canvas.*` matched, `overlay` only as `canvas_abi_overlay_raster.inl`). Closest design statement is the intent log src/common/tensors/abstraction/INTENTION_2026-01-07_164145.md ("move TensorAllocator and TensorTool onto an abstract tensor handle"), which describes only that refactor |
| Intention document | Only the above log, plus docs/LLM_ENTITY_ENVIRONMENT.md (action tiers, line 145-149) and docs/GRAPH_ENTITY_AND_SPRITE_CONTRACT.md. Neither describes a system port or a tensor-minting tool |

## 6. Before touching Nodus timing or tensor allocation

1. The tick `dt` is chosen by the HOST (Python `--step-dt` thread, or the SDL tick controller); the canvas only forwards it. A new clock facility has no existing home: there is no time port, only `TickRequest.dt` and `thread_mgr_delay_ms`.
2. Delay control draw and hit rectangles differ (drawn x~660, hit x 760/868) and the toolbar CLN/CLR/X are not wired: do not trust a click-test of those without reading both sides.
3. Global TIME toggles profiling only; module "TIME" is hidden under FREE and wired to nothing. FREE has no scheduling effect in `gp_canvas_step`.
4. The toolbar SIM freezes rope sim cadence globally via a process-global (`gp_table_set_global_sim_frame_skip_count`), so two canvases in one process share it.
5. TensorAllocator is a re-allocator keyed on a shape recorded on the frame stack, silently does nothing without `default_backend()`, and its trigger action (3102) has no producer. TensorTool is a stub. The working minting path is `nodus_tensor_create` / `nodus_headless_tensor_create` on the InMemoryBackend arena.
6. Arena settings must be applied before the first allocation; `nodus_tensor_arena_reset` is test-only.
7. GUI file Save drops EDGE_FIFO/PORT/CONTRACT_PORT trailers (autosave and the Save button write CANVAS V4 only; comment frontend_shell.cpp:1540-1545). Contract ports exist only in the SDL shell with `--acquire`.
8. `gp_canvas_create` auto-loads and autosaves `canvas_workspace.txt` in the cwd every 2 s: running the web server in the repo root edits that tracked file.
9. The web page sends no keys, no scroll, and no dx/dy.

## 7. frontend_shell.cpp (SDL host), short

- Options (2555-2578): `--device`, `--width/--height`, `--workspace`, `--acquire=PREFIX:CAP`, `--attach`, `--console-token-file`, `--thread-delay`, `--gui-delay`, `--canvas-config=IDX:DT:RATIO:FREE`, `--canvas-display-interval/-ratio`, `--pause-canvas`, `--disable-keyboard/-mouse`, `--enable-gamepad`.
- Layout: window clear colour {20,24,34}; canvas texture top-centred (`on_window_resized` 2149); no widgets of its own except the attach status bar (2809).
- Input -> canvas: key `gp_canvas_on_key` (1996), mouse down/up/move/scroll to `gp_canvas_on_mouse_*` with coordinates offset by canvas_x/y (2027-2082). Debug `fprintf` on every click/scroll.
- Shell-only shortcuts: Ctrl+Shift+Tab then digits 1-8 within 3 s switches among 8 canvas heads (`handle_combo_key` 2187; with Alt held it first copies the active workspace file over the target: `complete_combo` 2221-2262); state in `frontend_canvases.txt`. Ctrl+Shift+K toggles the kpath raster demo (1888); in that mode Space toggles scatter and `[ ] - = O P K L H J , . N M` adjust heat decay, energy scale, exposure gain, film decay, heat intensity, heat max Kelvin, energy/heat normalization (1915-1983). Gamepad buttons only print.
- Attach mode drops all keyboard/mouse/gamepad input (1866-1882).
