#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Event/action ids used by tools and plugins for module-frame bindings.
// These mirror the canvas action ids so plugins can bind/listen without
// depending on internal canvas headers.
enum GP_ToolEventId {
    GP_TOOL_EVENT_KEYBOARD = 2106, // matches CANVAS_ACT_MENU_TOOL_KEYBOARD
    GP_TOOL_EVENT_MOUSE    = 2107, // matches CANVAS_ACT_MENU_TOOL_MOUSE
};

#ifdef __cplusplus
}
#endif
