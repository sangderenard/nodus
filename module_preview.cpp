// Module preview builder stub for send/receive strip experimentation.

#include "module_preview.h"

#include <algorithm>
#include <cstring>
#include <vector>
#include <cmath>

void gp_module_preview_build(const GP_ModulePreviewInput* input, GP_ModulePreviewOutput* output) {
    if (!output) return;
    output->hitbox_count = 0;
    if (!output->rgba || output->width_px <= 0 || output->height_px <= 0 || output->pitch_bytes <= 0) {
        return;
    }
    const int out_h = output->height_px;
    const int out_w = output->width_px;
    const int out_pitch = output->pitch_bytes;

    // If no stack data, draw a checker pattern so the preview region is visible.
    if (!input || input->stack_tail_count <= 0) {
        const int tile = 8;
        const uint8_t c1 = 200, c2 = 160;
        for (int y = 0; y < out_h; ++y) {
            uint8_t* dst = output->rgba + y * out_pitch;
            for (int x = 0; x < out_w; ++x) {
                int tx = (x / tile) & 1;
                int ty = (y / tile) & 1;
                uint8_t v = (tx ^ ty) ? c1 : c2;
                int i = x * 4;
                if (i + 3 < out_pitch) {
                    dst[i + 0] = v;
                    dst[i + 1] = v;
                    dst[i + 2] = v;
                    dst[i + 3] = 255;
                }
            }
        }
        return;
    }

    // Determine largest square we can form from the tail values
    int available = std::clamp(input->stack_tail_count, 0, 64);
    int s = static_cast<int>(std::floor(std::sqrt(static_cast<float>(available))));
    if (s <= 0) {
        const int cols = out_w * 4;
        for (int y = 0; y < out_h; ++y) {
            std::memset(output->rgba + y * out_pitch, 0, static_cast<size_t>(std::min(cols, out_pitch)));
        }
        return;
    }

    const int src_pixels = s * s;
    std::vector<uint8_t> src_rgba(static_cast<size_t>(src_pixels) * 4);

    // Prefer the most recent contiguous block of values for the square
    int start_idx = 0;
    if (available > src_pixels) start_idx = available - src_pixels;

    // find min/max in the used tail slice
    float vmin = input->stack_tail[start_idx];
    float vmax = input->stack_tail[start_idx];
    for (int i = 0; i < src_pixels; ++i) {
        float v = input->stack_tail[start_idx + i];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    float denom = vmax - vmin;

    // fill source square as grayscale RGBA; if flat signal, render mid-gray
    if (std::fabs(denom) < 1e-6f) {
        uint8_t mid = 128;
        for (int i = 0; i < src_pixels; ++i) {
            src_rgba[static_cast<size_t>(i) * 4 + 0] = mid;
            src_rgba[static_cast<size_t>(i) * 4 + 1] = mid;
            src_rgba[static_cast<size_t>(i) * 4 + 2] = mid;
            src_rgba[static_cast<size_t>(i) * 4 + 3] = 255;
        }
    } else {
        for (int i = 0; i < src_pixels; ++i) {
            float v = input->stack_tail[start_idx + i];
            float n = (v - vmin) / denom;
            n = std::clamp(n, 0.0f, 1.0f);
            uint8_t g = static_cast<uint8_t>(std::lround(n * 255.0f));
            src_rgba[static_cast<size_t>(i) * 4 + 0] = g;
            src_rgba[static_cast<size_t>(i) * 4 + 1] = g;
            src_rgba[static_cast<size_t>(i) * 4 + 2] = g;
            src_rgba[static_cast<size_t>(i) * 4 + 3] = 255;
        }
    }

    // Nearest-neighbor scale from src (s x s) into output (out_w x out_h)
    for (int y = 0; y < out_h; ++y) {
        uint8_t* dst_row = output->rgba + y * out_pitch;
        for (int x = 0; x < out_w; ++x) {
            // map output pixel center to source coordinate, then nearest
            float src_fx = ((x + 0.5f) * static_cast<float>(s) / static_cast<float>(out_w)) - 0.5f;
            float src_fy = ((y + 0.5f) * static_cast<float>(s) / static_cast<float>(out_h)) - 0.5f;
            int sx = static_cast<int>(std::lround(src_fx));
            int sy = static_cast<int>(std::lround(src_fy));
            if (sx < 0) sx = 0; if (sx >= s) sx = s - 1;
            if (sy < 0) sy = 0; if (sy >= s) sy = s - 1;
            int si = sy * s + sx;
            const uint8_t* src_px = &src_rgba[static_cast<size_t>(si) * 4];
            int dst_i = x * 4;
            if (dst_i + 3 < out_pitch) {
                dst_row[dst_i + 0] = src_px[0];
                dst_row[dst_i + 1] = src_px[1];
                dst_row[dst_i + 2] = src_px[2];
                dst_row[dst_i + 3] = src_px[3];
            }
        }
    }
}
