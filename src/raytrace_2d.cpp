#include "raytrace_2d.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;

static uint32_t lcg_next(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

static float rand01(uint32_t& s) {
    return (lcg_next(s) >> 8) * (1.0f / 16777216.0f);
}

struct Raytrace2DImpl {
    int width = 0;
    int height = 0;
    float room_min_x = 0.0f;
    float room_min_y = 0.0f;
    float room_max_x = 1.0f;
    float room_max_y = 1.0f;
    float light_x = 0.5f;
    float light_y = 0.5f;
    float light_radius = 0.1f;
    int rays = 1024;
    int max_reflections = 2;
    float blur_sigma = 2.0f;
    float bounce_decay = 0.75f;
    float distance_decay = 0.0025f;
    uint32_t seed = 1;
    std::vector<float> accum;
    std::vector<float> tmp;
    std::vector<float> blur;
};

static void ensure_buffers(Raytrace2DImpl* rt) {
    if (!rt) return;
    const size_t count = static_cast<size_t>(std::max(0, rt->width)) * static_cast<size_t>(std::max(0, rt->height));
    rt->accum.assign(count, 0.0f);
    rt->tmp.assign(count, 0.0f);
    rt->blur.assign(count, 0.0f);
}

static void add_segment(Raytrace2DImpl* rt, float x0, float y0, float x1, float y1, float weight) {
    if (!rt || rt->width <= 0 || rt->height <= 0) return;
    const float rw = rt->room_max_x - rt->room_min_x;
    const float rh = rt->room_max_y - rt->room_min_y;
    if (rw <= 0.0f || rh <= 0.0f) return;
    const float fx0 = (x0 - rt->room_min_x) / rw * float(rt->width - 1);
    const float fy0 = (y0 - rt->room_min_y) / rh * float(rt->height - 1);
    const float fx1 = (x1 - rt->room_min_x) / rw * float(rt->width - 1);
    const float fy1 = (y1 - rt->room_min_y) / rh * float(rt->height - 1);
    const float dx = fx1 - fx0;
    const float dy = fy1 - fy0;
    int steps = static_cast<int>(std::ceil(std::max(std::fabs(dx), std::fabs(dy))));
    if (steps < 1) steps = 1;
    const float sx = dx / float(steps);
    const float sy = dy / float(steps);
    float x = fx0;
    float y = fy0;
    for (int i = 0; i <= steps; ++i) {
        int px = static_cast<int>(std::floor(x + 0.5f));
        int py = static_cast<int>(std::floor(y + 0.5f));
        if (px >= 0 && px < rt->width && py >= 0 && py < rt->height) {
            rt->accum[static_cast<size_t>(py) * static_cast<size_t>(rt->width) + static_cast<size_t>(px)] += weight;
        }
        x += sx;
        y += sy;
    }
}

static void gaussian_blur(Raytrace2DImpl* rt) {
    if (!rt || rt->width <= 0 || rt->height <= 0) return;
    const int w = rt->width;
    const int h = rt->height;
    const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (rt->blur_sigma <= 0.0f) {
        for (size_t i = 0; i < count; ++i) rt->blur[i] = rt->accum[i];
        return;
    }

    const float sigma = std::max(0.1f, rt->blur_sigma);
    const int radius = std::max(1, static_cast<int>(std::ceil(sigma * 3.0f)));
    const int ksize = radius * 2 + 1;
    std::vector<float> kernel(static_cast<size_t>(ksize));
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        float v = std::exp(-0.5f * (float(i) * float(i)) / (sigma * sigma));
        kernel[static_cast<size_t>(i + radius)] = v;
        sum += v;
    }
    if (sum > 0.0f) {
        for (float& v : kernel) v /= sum;
    }

    // horizontal pass
    for (int y = 0; y < h; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int xx = std::clamp(x + k, 0, w - 1);
                acc += rt->accum[row + static_cast<size_t>(xx)] * kernel[static_cast<size_t>(k + radius)];
            }
            rt->tmp[row + static_cast<size_t>(x)] = acc;
        }
    }

    // vertical pass
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int yy = std::clamp(y + k, 0, h - 1);
                acc += rt->tmp[static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(x)] * kernel[static_cast<size_t>(k + radius)];
            }
            rt->blur[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = acc;
        }
    }
}

} // namespace

extern "C" {

Raytrace2D* raytrace2d_create(int width, int height) {
    if (width <= 0 || height <= 0) return nullptr;
    Raytrace2DImpl* rt = new Raytrace2DImpl();
    rt->width = width;
    rt->height = height;
    rt->room_min_x = 0.0f;
    rt->room_min_y = 0.0f;
    rt->room_max_x = static_cast<float>(width);
    rt->room_max_y = static_cast<float>(height);
    rt->light_x = 0.5f * (rt->room_min_x + rt->room_max_x);
    rt->light_y = 0.5f * (rt->room_min_y + rt->room_max_y);
    rt->light_radius = 0.1f * std::min(rt->room_max_x - rt->room_min_x, rt->room_max_y - rt->room_min_y);
    ensure_buffers(rt);
    return reinterpret_cast<Raytrace2D*>(rt);
}

void raytrace2d_destroy(Raytrace2D* rt) {
    if (!rt) return;
    auto* impl = reinterpret_cast<Raytrace2DImpl*>(rt);
    delete impl;
}

int raytrace2d_resize(Raytrace2D* rt, int width, int height) {
    if (!rt || width <= 0 || height <= 0) return 0;
    auto* impl = reinterpret_cast<Raytrace2DImpl*>(rt);
    impl->width = width;
    impl->height = height;
    ensure_buffers(impl);
    return 1;
}

int raytrace2d_set_room(Raytrace2D* rt, float min_x, float min_y, float max_x, float max_y) {
    if (!rt) return 0;
    if (!(max_x > min_x && max_y > min_y)) return 0;
    auto* impl = reinterpret_cast<Raytrace2DImpl*>(rt);
    impl->room_min_x = min_x;
    impl->room_min_y = min_y;
    impl->room_max_x = max_x;
    impl->room_max_y = max_y;
    return 1;
}

int raytrace2d_set_light(Raytrace2D* rt, float cx, float cy, float radius) {
    if (!rt) return 0;
    auto* impl = reinterpret_cast<Raytrace2DImpl*>(rt);
    impl->light_x = cx;
    impl->light_y = cy;
    impl->light_radius = std::max(0.0f, radius);
    return 1;
}

int raytrace2d_set_params(Raytrace2D* rt, int ray_count, int max_reflections, float blur_sigma) {
    if (!rt) return 0;
    auto* impl = reinterpret_cast<Raytrace2DImpl*>(rt);
    impl->rays = std::max(1, ray_count);
    impl->max_reflections = std::max(0, max_reflections);
    impl->blur_sigma = std::max(0.0f, blur_sigma);
    return 1;
}

int raytrace2d_set_attenuation(Raytrace2D* rt, float bounce_decay, float distance_decay) {
    if (!rt) return 0;
    auto* impl = reinterpret_cast<Raytrace2DImpl*>(rt);
    impl->bounce_decay = std::clamp(bounce_decay, 0.0f, 1.0f);
    impl->distance_decay = std::max(0.0f, distance_decay);
    return 1;
}

int raytrace2d_set_seed(Raytrace2D* rt, uint32_t seed) {
    if (!rt) return 0;
    auto* impl = reinterpret_cast<Raytrace2DImpl*>(rt);
    impl->seed = seed ? seed : 1u;
    return 1;
}

int raytrace2d_render_rgba(Raytrace2D* rt, uint8_t* out_rgba, int32_t out_len_bytes) {
    if (!rt || !out_rgba) return 0;
    auto* impl = reinterpret_cast<Raytrace2DImpl*>(rt);
    if (impl->width <= 0 || impl->height <= 0) return 0;
    const int32_t need = impl->width * impl->height * 4;
    if (out_len_bytes < need) return 0;

    std::fill(impl->accum.begin(), impl->accum.end(), 0.0f);

    const float min_x = impl->room_min_x;
    const float min_y = impl->room_min_y;
    const float max_x = impl->room_max_x;
    const float max_y = impl->room_max_y;
    if (!(max_x > min_x && max_y > min_y)) return 0;

    uint32_t seed = impl->seed;
    const float eps = 1e-3f;
    const int rays = std::max(1, impl->rays);
    const int max_bounces = std::max(0, impl->max_reflections);
    for (int i = 0; i < rays; ++i) {
        const float dir_ang = (float(i) + 0.5f) / float(rays) * (2.0f * kPi);
        const float pos_ang = rand01(seed) * (2.0f * kPi);
        float x = impl->light_x + std::cos(pos_ang) * impl->light_radius;
        float y = impl->light_y + std::sin(pos_ang) * impl->light_radius;
        x = std::clamp(x, min_x + eps, max_x - eps);
        y = std::clamp(y, min_y + eps, max_y - eps);
        float dx = std::cos(dir_ang);
        float dy = std::sin(dir_ang);
        float total_len = 0.0f;
        for (int bounce = 0; bounce <= max_bounces; ++bounce) {
            float tx = std::numeric_limits<float>::infinity();
            float ty = std::numeric_limits<float>::infinity();
            if (dx > 0.0f) tx = (max_x - x) / dx;
            else if (dx < 0.0f) tx = (min_x - x) / dx;
            if (dy > 0.0f) ty = (max_y - y) / dy;
            else if (dy < 0.0f) ty = (min_y - y) / dy;
            float t = std::min(tx, ty);
            if (!std::isfinite(t) || t <= 0.0f) break;
            float nx = x + dx * t;
            float ny = y + dy * t;
            float seg_len = t;
            float avg_len = total_len + 0.5f * seg_len;
            float bounce_att = std::pow(impl->bounce_decay, static_cast<float>(bounce));
            float dist_att = std::exp(-impl->distance_decay * avg_len);
            float weight = std::max(0.0f, bounce_att * dist_att);
            add_segment(impl, x, y, nx, ny, weight);
            bool hit_x = std::fabs(tx - ty) <= 1e-6f || tx < ty;
            bool hit_y = std::fabs(tx - ty) <= 1e-6f || ty < tx;
            if (hit_x) dx = -dx;
            if (hit_y) dy = -dy;
            x = std::clamp(nx + dx * eps, min_x + eps, max_x - eps);
            y = std::clamp(ny + dy * eps, min_y + eps, max_y - eps);
            total_len += seg_len;
        }
    }

    gaussian_blur(impl);

    float max_val = 0.0f;
    for (float v : impl->blur) max_val = std::max(max_val, v);
    float scale = (max_val > 0.0f) ? (255.0f / max_val) : 0.0f;
    for (int y = 0; y < impl->height; ++y) {
        for (int x = 0; x < impl->width; ++x) {
            float v = impl->blur[static_cast<size_t>(y) * static_cast<size_t>(impl->width) + static_cast<size_t>(x)];
            int iv = static_cast<int>(std::lround(v * scale));
            iv = std::clamp(iv, 0, 255);
            size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(impl->width) + static_cast<size_t>(x)) * 4u;
            out_rgba[idx + 0] = static_cast<uint8_t>(iv);
            out_rgba[idx + 1] = static_cast<uint8_t>(iv);
            out_rgba[idx + 2] = static_cast<uint8_t>(iv);
            out_rgba[idx + 3] = 255;
        }
    }
    return 1;
}

} // extern "C"
