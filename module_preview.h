// Module preview builder interface (module frame send/receive layout workspace).
#pragma once

#include <stdint.h>

#include "table_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Input description for module preview generation.
typedef struct GP_ModulePreviewInput {
    int32_t rows;
    int32_t cols;
    int32_t layout_strategy;
    const void* send_ptrs[16];
    const void* receive_ptrs[16];
    float stack_tail[64];
    int32_t stack_tail_count;
} GP_ModulePreviewInput;

// Output buffer for module preview generation.
typedef struct GP_ModulePreviewOutput {
    uint8_t* rgba;
    int32_t width_px;
    int32_t height_px;
    int32_t pitch_bytes;
    GP_TableHitBox* hitboxes;
    int32_t hitbox_capacity;
    int32_t hitbox_count;
} GP_ModulePreviewOutput;

// Build the module preview layer (image + hitboxes). Intended to be called
// from the UI tick; fill in here for layout experiments.
void gp_module_preview_build(const GP_ModulePreviewInput* input, GP_ModulePreviewOutput* output);

#ifdef __cplusplus
}
#endif
