// Module preview builder stub for send/receive strip experimentation.

#include "module_preview.h"

#include <algorithm>
#include <cstring>

void gp_module_preview_build(const GP_ModulePreviewInput* input, GP_ModulePreviewOutput* output) {
    if (!output) return;
    output->hitbox_count = 0;
    if (!output->rgba || output->width_px <= 0 || output->height_px <= 0 || output->pitch_bytes <= 0) {
        return;
    }
    const int rows = output->height_px;
    const int cols = output->width_px * 4;
    for (int y = 0; y < rows; ++y) {
        std::memset(output->rgba + y * output->pitch_bytes, 0, static_cast<size_t>(std::min(cols, output->pitch_bytes)));
    }
    (void)input;
}
