#include <math.h>
#include <stdint.h>

#include "common/tensors/abstraction/microkernels.h"

static inline float act_relu(float x) { return x > 0.0f ? x : 0.0f; }
static inline float act_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float act_tanh(float x) { return tanhf(x); }

void nodus_fused_addmul_f32(float* out,
                            const float* x,
                            const float* m,
                            const float* b,
                            uint32_t t,
                            uint32_t k,
                            uint32_t c,
                            uint32_t b_stride,
                            uint32_t act_id) {
    if (!out || !x || !m || !b || t == 0 || c == 0) return;

    if (k == 1) {
        switch (act_id) {
            case 0: { // ReLU
                for (uint32_t i = 0; i < t; ++i) {
                    const float xv = x[i];
                    const uint32_t b_off = (b_stride == 0) ? 0u : i * b_stride;
                    float* dst = out + i * c;
                    for (uint32_t j = 0; j < c; ++j) {
                        const float z = xv * m[j] + b[b_off + j];
                        dst[j] += act_relu(z);
                    }
                }
            } break;
            case 1: { // Sigmoid
                for (uint32_t i = 0; i < t; ++i) {
                    const float xv = x[i];
                    const uint32_t b_off = (b_stride == 0) ? 0u : i * b_stride;
                    float* dst = out + i * c;
                    for (uint32_t j = 0; j < c; ++j) {
                        const float z = xv * m[j] + b[b_off + j];
                        dst[j] += act_sigmoid(z);
                    }
                }
            } break;
            case 2: { // Tanh
                for (uint32_t i = 0; i < t; ++i) {
                    const float xv = x[i];
                    const uint32_t b_off = (b_stride == 0) ? 0u : i * b_stride;
                    float* dst = out + i * c;
                    for (uint32_t j = 0; j < c; ++j) {
                        const float z = xv * m[j] + b[b_off + j];
                        dst[j] += act_tanh(z);
                    }
                }
            } break;
            default: { // Linear
                for (uint32_t i = 0; i < t; ++i) {
                    const float xv = x[i];
                    const uint32_t b_off = (b_stride == 0) ? 0u : i * b_stride;
                    float* dst = out + i * c;
                    for (uint32_t j = 0; j < c; ++j) {
                        const float z = xv * m[j] + b[b_off + j];
                        dst[j] += z;
                    }
                }
            } break;
        }
        return;
    }

    switch (act_id) {
        case 0: { // ReLU
            for (uint32_t i = 0; i < t; ++i) {
                const uint32_t x_off = i * k;
                const uint32_t b_off = (b_stride == 0) ? 0u : i * b_stride;
                float* dst = out + i * c;
                for (uint32_t j = 0; j < c; ++j) {
                    float acc = b[b_off + j];
                    const float* mp = m + j;
                    for (uint32_t kk = 0; kk < k; ++kk) {
                        acc += x[x_off + kk] * mp[kk * c];
                    }
                    dst[j] += act_relu(acc);
                }
            }
        } break;
        case 1: { // Sigmoid
            for (uint32_t i = 0; i < t; ++i) {
                const uint32_t x_off = i * k;
                const uint32_t b_off = (b_stride == 0) ? 0u : i * b_stride;
                float* dst = out + i * c;
                for (uint32_t j = 0; j < c; ++j) {
                    float acc = b[b_off + j];
                    const float* mp = m + j;
                    for (uint32_t kk = 0; kk < k; ++kk) {
                        acc += x[x_off + kk] * mp[kk * c];
                    }
                    dst[j] += act_sigmoid(acc);
                }
            }
        } break;
        case 2: { // Tanh
            for (uint32_t i = 0; i < t; ++i) {
                const uint32_t x_off = i * k;
                const uint32_t b_off = (b_stride == 0) ? 0u : i * b_stride;
                float* dst = out + i * c;
                for (uint32_t j = 0; j < c; ++j) {
                    float acc = b[b_off + j];
                    const float* mp = m + j;
                    for (uint32_t kk = 0; kk < k; ++kk) {
                        acc += x[x_off + kk] * mp[kk * c];
                    }
                    dst[j] += act_tanh(acc);
                }
            }
        } break;
        default: { // Linear
            for (uint32_t i = 0; i < t; ++i) {
                const uint32_t x_off = i * k;
                const uint32_t b_off = (b_stride == 0) ? 0u : i * b_stride;
                float* dst = out + i * c;
                for (uint32_t j = 0; j < c; ++j) {
                    float acc = b[b_off + j];
                    const float* mp = m + j;
                    for (uint32_t kk = 0; kk < k; ++kk) {
                        acc += x[x_off + kk] * mp[kk * c];
                    }
                    dst[j] += acc;
                }
            }
        } break;
    }
}

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
                                    float threshold) {
    if (!out || !x || rows == 0 || cols == 0 || c == 0) return;
    const float* mm = m;
    const float* bb = b;
    const uint32_t tile_elems = cols * c;

    if (k == 1 && !mm) {
        static const float one = 1.0f;
        mm = &one;
    }

    for (uint32_t r = 0; r < rows; ++r) {
        float* dst = out + r * out_row_stride;
        const float* xv = x + r * x_row_stride;
        const uint32_t b_off = (bb && b_stride != 0) ? r * b_stride : 0u;
        (void)tile_elems;
        if (k == 1) {
            const float mul = mm ? mm[0] : 1.0f;
            switch (act_id) {
                case 0:
                    for (uint32_t j = 0; j < cols; ++j) {
                        const float xv0 = xv[j];
                        float* drow = dst + j * c;
                        for (uint32_t cc = 0; cc < c; ++cc) {
                            float z = xv0 * mul + (bb ? bb[b_off + cc] : 0.0f);
                            z = act_relu(z);
                            drow[cc] += z;
                            if (saturate && drow[cc] > threshold) drow[cc] = threshold;
                        }
                    }
                    break;
                case 1:
                    for (uint32_t j = 0; j < cols; ++j) {
                        const float xv0 = xv[j];
                        float* drow = dst + j * c;
                        for (uint32_t cc = 0; cc < c; ++cc) {
                            float z = xv0 * mul + (bb ? bb[b_off + cc] : 0.0f);
                            z = act_sigmoid(z);
                            drow[cc] += z;
                            if (saturate && drow[cc] > threshold) drow[cc] = threshold;
                        }
                    }
                    break;
                case 2:
                    for (uint32_t j = 0; j < cols; ++j) {
                        const float xv0 = xv[j];
                        float* drow = dst + j * c;
                        for (uint32_t cc = 0; cc < c; ++cc) {
                            float z = xv0 * mul + (bb ? bb[b_off + cc] : 0.0f);
                            z = act_tanh(z);
                            drow[cc] += z;
                            if (saturate && drow[cc] > threshold) drow[cc] = threshold;
                        }
                    }
                    break;
                default:
                    for (uint32_t j = 0; j < cols; ++j) {
                        const float xv0 = xv[j];
                        float* drow = dst + j * c;
                        for (uint32_t cc = 0; cc < c; ++cc) {
                            float z = xv0 * mul + (bb ? bb[b_off + cc] : 0.0f);
                            drow[cc] += z;
                            if (saturate && drow[cc] > threshold) drow[cc] = threshold;
                        }
                    }
                    break;
            }
            continue;
        }

        switch (act_id) {
            case 0:
                for (uint32_t j = 0; j < cols; ++j) {
                    float* drow = dst + j * c;
                    for (uint32_t cc = 0; cc < c; ++cc) {
                        float acc = (bb ? bb[b_off + cc] : 0.0f);
                        const float* mp = m + cc;
                        for (uint32_t kk = 0; kk < k; ++kk) {
                            acc += xv[j * k + kk] * mp[kk * c];
                        }
                        acc = act_relu(acc);
                        drow[cc] += acc;
                        if (saturate && drow[cc] > threshold) drow[cc] = threshold;
                    }
                }
                break;
            case 1:
                for (uint32_t j = 0; j < cols; ++j) {
                    float* drow = dst + j * c;
                    for (uint32_t cc = 0; cc < c; ++cc) {
                        float acc = (bb ? bb[b_off + cc] : 0.0f);
                        const float* mp = m + cc;
                        for (uint32_t kk = 0; kk < k; ++kk) {
                            acc += xv[j * k + kk] * mp[kk * c];
                        }
                        acc = act_sigmoid(acc);
                        drow[cc] += acc;
                        if (saturate && drow[cc] > threshold) drow[cc] = threshold;
                    }
                }
                break;
            case 2:
                for (uint32_t j = 0; j < cols; ++j) {
                    float* drow = dst + j * c;
                    for (uint32_t cc = 0; cc < c; ++cc) {
                        float acc = (bb ? bb[b_off + cc] : 0.0f);
                        const float* mp = m + cc;
                        for (uint32_t kk = 0; kk < k; ++kk) {
                            acc += xv[j * k + kk] * mp[kk * c];
                        }
                        acc = act_tanh(acc);
                        drow[cc] += acc;
                        if (saturate && drow[cc] > threshold) drow[cc] = threshold;
                    }
                }
                break;
            default:
                for (uint32_t j = 0; j < cols; ++j) {
                    float* drow = dst + j * c;
                    for (uint32_t cc = 0; cc < c; ++cc) {
                        float acc = (bb ? bb[b_off + cc] : 0.0f);
                        const float* mp = m + cc;
                        for (uint32_t kk = 0; kk < k; ++kk) {
                            acc += xv[j * k + kk] * mp[kk * c];
                        }
                        drow[cc] += acc;
                        if (saturate && drow[cc] > threshold) drow[cc] = threshold;
                    }
                }
                break;
        }
    }
}

void nodus_fused_add_f32_strided(float* out,
                                 uint32_t out_row_stride,
                                 const float* x,
                                 uint32_t x_row_stride,
                                 uint32_t rows,
                                 uint32_t cols,
                                 uint32_t c,
                                 uint32_t saturate,
                                 float threshold) {
    if (!out || !x || rows == 0 || cols == 0 || c == 0) return;

    for (uint32_t r = 0; r < rows; ++r) {
        float* dst = out + r * out_row_stride;
        const float* xv = x + r * x_row_stride;
        for (uint32_t j = 0; j < cols; ++j) {
            float* drow = dst + j * c;
            const float* xrow = xv + j * c;
            if (!saturate) {
                for (uint32_t cc = 0; cc < c; ++cc) {
                    drow[cc] += xrow[cc];
                }
            } else {
                for (uint32_t cc = 0; cc < c; ++cc) {
                    float v = drow[cc] + xrow[cc];
                    if (v > threshold) v = threshold;
                    drow[cc] = v;
                }
            }
        }
    }
}
