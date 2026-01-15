#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Fused: out += activation(M * x + b)
// - out: [T, C]
// - x:   [T, K]
// - M:   [K, C]
// - b:   [C] (b_stride=0) or [T, C] (b_stride=C)
// - act_id: 0=ReLU, 1=Sigmoid, 2=Tanh, else Linear
void nodus_fused_addmul_f32(float* out,
                            const float* x,
                            const float* m,
                            const float* b,
                            uint32_t t,
                            uint32_t k,
                            uint32_t c,
                            uint32_t b_stride,
                            uint32_t act_id);

// Strided variant for tiled updates.
// - out_row_stride, x_row_stride in elements.
// - rows, cols describe the tile shape.
// - If m is NULL and k == 1, multiplier is treated as 1.0.
// - If b is NULL, bias is treated as 0.0.
// - If saturate != 0, out is clamped to threshold after activation.
void nodus_fused_addmul_f32_strided(float* out,
                                    uint32_t out_row_stride,
                                    const float* x,
                                    uint32_t x_row_stride,
                                    const float* m,
                                    const float* b,
                                    uint32_t rows,
                                    uint32_t cols,
                                    uint32_t k,
                                    uint32_t c,
                                    uint32_t b_stride,
                                    uint32_t act_id,
                                    uint32_t saturate,
                                    float threshold);

// Strided add for tiled updates: out += x (x is [rows, cols, c]).
// - out_row_stride, x_row_stride in elements.
// - If saturate != 0, out is clamped to threshold after addition.
void nodus_fused_add_f32_strided(float* out,
                                 uint32_t out_row_stride,
                                 const float* x,
                                 uint32_t x_row_stride,
                                 uint32_t rows,
                                 uint32_t cols,
                                 uint32_t c,
                                 uint32_t saturate,
                                 float threshold);

#ifdef __cplusplus
} // extern "C"
#endif
