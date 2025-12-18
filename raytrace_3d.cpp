#include "raytrace_3d.h"

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

struct Raytrace3DImpl {
    int width = 0;
    int height = 0;
    float room_min_x = 0.0f;
    float room_min_y = 0.0f;
    float room_min_z = 0.0f;
    float room_max_x = 1.0f;
    float room_max_y = 1.0f;
    float room_max_z = 1.0f;
    float light_x = 0.5f;
    float light_y = 0.5f;
    float light_z = 0.5f;
    float light_radius = 0.1f;
    int rays = 1024;
    int max_reflections = 2;
    float blur_sigma = 2.0f;
    float bounce_decay = 0.75f;
    float distance_decay = 0.0025f;
    float wave_frequency = 0.0f;
    float wave_phase0 = 0.0f;
    uint32_t seed = 1;

    struct Tri {
        float v0[3];
        float v1[3];
        float v2[3];
        float n[3];
        float reflectivity;
    };
    std::vector<Tri> tris;

    std::vector<float> accum;
    std::vector<float> tmp;
    std::vector<float> blur;
};

static void ensure_buffers(Raytrace3DImpl* rt, size_t count) {
    if (!rt) return;
    if (rt->accum.size() != count) rt->accum.assign(count, 0.0f);
    if (rt->tmp.size() != count) rt->tmp.assign(count, 0.0f);
    if (rt->blur.size() != count) rt->blur.assign(count, 0.0f);
}

static void gaussian_blur_2d(float* io, float* tmp, int w, int h, float sigma) {
    if (!io || !tmp || w <= 0 || h <= 0) return;
    const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (sigma <= 0.0f) return;

    sigma = std::max(0.1f, sigma);
    const int radius = std::max(1, static_cast<int>(std::ceil(sigma * 3.0f)));
    const int ksize = radius * 2 + 1;
    std::vector<float> kernel(static_cast<size_t>(ksize));
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        float v = std::exp(-0.5f * (float(i) * float(i)) / (sigma * sigma));
        kernel[static_cast<size_t>(i + radius)] = v;
        sum += v;
    }
    if (sum > 0.0f) for (float& v : kernel) v /= sum;

    // horizontal
    for (int y = 0; y < h; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int xx = std::clamp(x + k, 0, w - 1);
                acc += io[row + static_cast<size_t>(xx)] * kernel[static_cast<size_t>(k + radius)];
            }
            tmp[row + static_cast<size_t>(x)] = acc;
        }
    }
    // vertical
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int yy = std::clamp(y + k, 0, h - 1);
                acc += tmp[static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(x)] * kernel[static_cast<size_t>(k + radius)];
            }
            io[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = acc;
        }
    }
}

static int quantize_layer(float z, int layer_count, const float* centers, float min_z, float max_z) {
    if (layer_count <= 1) return 0;
    if (centers) {
        int best = 0;
        float best_d = std::fabs(z - centers[0]);
        for (int i = 1; i < layer_count; ++i) {
            float d = std::fabs(z - centers[i]);
            if (d < best_d) { best_d = d; best = i; }
        }
        return best;
    }
    float t = 0.0f;
    if (max_z > min_z) t = (z - min_z) / (max_z - min_z);
    int idx = static_cast<int>(std::floor(t * layer_count));
    return std::clamp(idx, 0, layer_count - 1);
}

static void add_segment_quantized(
    Raytrace3DImpl* rt,
    float x0, float y0, float z0,
    float x1, float y1, float z1,
    float weight,
    int layer_count,
    const float* layer_centers_z,
    float* out_layers) {
    if (!rt || !out_layers) return;
    if (rt->width <= 0 || rt->height <= 0) return;

    const float rw = rt->room_max_x - rt->room_min_x;
    const float rh = rt->room_max_y - rt->room_min_y;
    if (rw <= 0.0f || rh <= 0.0f) return;

    const float fx0 = (x0 - rt->room_min_x) / rw * float(rt->width - 1);
    const float fy0 = (y0 - rt->room_min_y) / rh * float(rt->height - 1);
    const float fx1 = (x1 - rt->room_min_x) / rw * float(rt->width - 1);
    const float fy1 = (y1 - rt->room_min_y) / rh * float(rt->height - 1);

    const float dx = fx1 - fx0;
    const float dy = fy1 - fy0;
    const float dz = z1 - z0;

    int steps = static_cast<int>(std::ceil(std::max({std::fabs(dx), std::fabs(dy), std::fabs(dz)})));
    if (steps < 1) steps = 1;
    const float sx = dx / float(steps);
    const float sy = dy / float(steps);
    const float sz = dz / float(steps);

    float x = fx0;
    float y = fy0;
    float z = z0;
    const float lmin = rt->room_min_z;
    const float lmax = rt->room_max_z;
    const size_t px = static_cast<size_t>(rt->width) * static_cast<size_t>(rt->height);
    for (int i = 0; i <= steps; ++i) {
        int px_i = static_cast<int>(std::floor(x + 0.5f));
        int py_i = static_cast<int>(std::floor(y + 0.5f));
        if (px_i >= 0 && px_i < rt->width && py_i >= 0 && py_i < rt->height) {
            int li = quantize_layer(z, layer_count, layer_centers_z, lmin, lmax);
            out_layers[static_cast<size_t>(li) * px + static_cast<size_t>(py_i) * static_cast<size_t>(rt->width) + static_cast<size_t>(px_i)] += weight;
        }
        x += sx;
        y += sy;
        z += sz;
    }
}

static void add_segment_field_quantized(
    Raytrace3DImpl* rt,
    float x0, float y0, float z0,
    float x1, float y1, float z1,
    float dirx, float diry, float dirz,
    float weight,
    float phase,
    int layer_count,
    const float* layer_centers_z,
    float* out_field) {
    if (!rt || !out_field) return;
    if (rt->width <= 0 || rt->height <= 0) return;

    const float rw = rt->room_max_x - rt->room_min_x;
    const float rh = rt->room_max_y - rt->room_min_y;
    if (rw <= 0.0f || rh <= 0.0f) return;

    const float fx0 = (x0 - rt->room_min_x) / rw * float(rt->width - 1);
    const float fy0 = (y0 - rt->room_min_y) / rh * float(rt->height - 1);
    const float fx1 = (x1 - rt->room_min_x) / rw * float(rt->width - 1);
    const float fy1 = (y1 - rt->room_min_y) / rh * float(rt->height - 1);

    const float dx = fx1 - fx0;
    const float dy = fy1 - fy0;
    const float dz = z1 - z0;
    int steps = static_cast<int>(std::ceil(std::max({std::fabs(dx), std::fabs(dy), std::fabs(dz)})));
    if (steps < 1) steps = 1;
    const float sx = dx / float(steps);
    const float sy = dy / float(steps);
    const float sz = dz / float(steps);

    float x = fx0;
    float y = fy0;
    float z = z0;

    const float lmin = rt->room_min_z;
    const float lmax = rt->room_max_z;
    const size_t px = static_cast<size_t>(rt->width) * static_cast<size_t>(rt->height);
    const float re = std::cos(phase) * weight;
    const float im = std::sin(phase) * weight;
    for (int i = 0; i <= steps; ++i) {
        int px_i = static_cast<int>(std::floor(x + 0.5f));
        int py_i = static_cast<int>(std::floor(y + 0.5f));
        if (px_i >= 0 && px_i < rt->width && py_i >= 0 && py_i < rt->height) {
            int li = quantize_layer(z, layer_count, layer_centers_z, lmin, lmax);
            const size_t base = (static_cast<size_t>(li) * px + static_cast<size_t>(py_i) * static_cast<size_t>(rt->width) + static_cast<size_t>(px_i)) * 6u;
            out_field[base + 0] += weight;
            out_field[base + 1] += dirx * weight;
            out_field[base + 2] += diry * weight;
            out_field[base + 3] += dirz * weight;
            out_field[base + 4] += re;
            out_field[base + 5] += im;
        }
        x += sx;
        y += sy;
        z += sz;
    }
}

static bool tri_intersect(const Raytrace3DImpl::Tri& t, float ox, float oy, float oz, float dx, float dy, float dz, float& out_t, float& nx, float& ny, float& nz) {
    // Moller-Trumbore
    float v0x = t.v0[0], v0y = t.v0[1], v0z = t.v0[2];
    float v1x = t.v1[0], v1y = t.v1[1], v1z = t.v1[2];
    float v2x = t.v2[0], v2y = t.v2[1], v2z = t.v2[2];
    float e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    float e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;
    float px = dy * e2z - dz * e2y;
    float py = dz * e2x - dx * e2z;
    float pz = dx * e2y - dy * e2x;
    float det = e1x * px + e1y * py + e1z * pz;
    if (std::fabs(det) < 1e-7f) return false;
    float inv_det = 1.0f / det;
    float tx = ox - v0x, ty = oy - v0y, tz = oz - v0z;
    float u = (tx * px + ty * py + tz * pz) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;
    float qx = ty * e1z - tz * e1y;
    float qy = tz * e1x - tx * e1z;
    float qz = tx * e1y - ty * e1x;
    float v = (dx * qx + dy * qy + dz * qz) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;
    float tval = (e2x * qx + e2y * qy + e2z * qz) * inv_det;
    if (tval <= 1e-6f) return false;
    out_t = tval;
    nx = t.n[0]; ny = t.n[1]; nz = t.n[2];
    return true;
}

static void compute_normal(Raytrace3DImpl::Tri& t) {
    float e1x = t.v1[0] - t.v0[0];
    float e1y = t.v1[1] - t.v0[1];
    float e1z = t.v1[2] - t.v0[2];
    float e2x = t.v2[0] - t.v0[0];
    float e2y = t.v2[1] - t.v0[1];
    float e2z = t.v2[2] - t.v0[2];
    float nx = e1y * e2z - e1z * e2y;
    float ny = e1z * e2x - e1x * e2z;
    float nz = e1x * e2y - e1y * e2x;
    float len = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (len < 1e-6f) { t.n[0]=0; t.n[1]=0; t.n[2]=1; return; }
    t.n[0] = nx / len; t.n[1] = ny / len; t.n[2] = nz / len;
}

} // namespace

extern "C" {

Raytrace3D* raytrace3d_create(int width, int height) {
    if (width <= 0 || height <= 0) return nullptr;
    auto* rt = new Raytrace3DImpl();
    rt->width = width;
    rt->height = height;
    rt->room_min_x = 0.0f;
    rt->room_min_y = 0.0f;
    rt->room_min_z = 0.0f;
    rt->room_max_x = static_cast<float>(width);
    rt->room_max_y = static_cast<float>(height);
    rt->room_max_z = static_cast<float>(std::max(1, std::min(width, height)));
    rt->light_x = 0.5f * (rt->room_min_x + rt->room_max_x);
    rt->light_y = 0.5f * (rt->room_min_y + rt->room_max_y);
    rt->light_z = 0.5f * (rt->room_min_z + rt->room_max_z);
    rt->light_radius = 0.1f * std::min(rt->room_max_x - rt->room_min_x, rt->room_max_y - rt->room_min_y);
    return reinterpret_cast<Raytrace3D*>(rt);
}

void raytrace3d_destroy(Raytrace3D* rt) {
    if (!rt) return;
    delete reinterpret_cast<Raytrace3DImpl*>(rt);
}

int raytrace3d_resize(Raytrace3D* rt, int width, int height) {
    if (!rt || width <= 0 || height <= 0) return 0;
    auto* impl = reinterpret_cast<Raytrace3DImpl*>(rt);
    impl->width = width;
    impl->height = height;
    return 1;
}

int raytrace3d_set_room(Raytrace3D* rt,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z) {
    if (!rt) return 0;
    if (!(max_x > min_x && max_y > min_y && max_z > min_z)) return 0;
    auto* impl = reinterpret_cast<Raytrace3DImpl*>(rt);
    impl->room_min_x = min_x;
    impl->room_min_y = min_y;
    impl->room_min_z = min_z;
    impl->room_max_x = max_x;
    impl->room_max_y = max_y;
    impl->room_max_z = max_z;
    return 1;
}

int raytrace3d_set_light(Raytrace3D* rt, float cx, float cy, float cz, float radius) {
    if (!rt) return 0;
    auto* impl = reinterpret_cast<Raytrace3DImpl*>(rt);
    impl->light_x = cx;
    impl->light_y = cy;
    impl->light_z = cz;
    impl->light_radius = std::max(0.0f, radius);
    return 1;
}

int raytrace3d_set_params(Raytrace3D* rt, int ray_count, int max_reflections, float blur_sigma) {
    if (!rt) return 0;
    auto* impl = reinterpret_cast<Raytrace3DImpl*>(rt);
    impl->rays = std::max(1, ray_count);
    impl->max_reflections = std::max(0, max_reflections);
    impl->blur_sigma = std::max(0.0f, blur_sigma);
    return 1;
}

int raytrace3d_set_attenuation(Raytrace3D* rt, float bounce_decay, float distance_decay) {
    if (!rt) return 0;
    auto* impl = reinterpret_cast<Raytrace3DImpl*>(rt);
    impl->bounce_decay = std::clamp(bounce_decay, 0.0f, 1.0f);
    impl->distance_decay = std::max(0.0f, distance_decay);
    return 1;
}

int raytrace3d_set_mesh(Raytrace3D* rt, const Raytrace3DMeshTriangle* tris, int count) {
    if (!rt) return 0;
    auto* impl = reinterpret_cast<Raytrace3DImpl*>(rt);
    impl->tris.clear();
    if (count <= 0) return 1;
    impl->tris.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        Raytrace3DImpl::Tri t{};
        std::memcpy(t.v0, tris[i].v0, sizeof(t.v0));
        std::memcpy(t.v1, tris[i].v1, sizeof(t.v1));
        std::memcpy(t.v2, tris[i].v2, sizeof(t.v2));
        std::memcpy(t.n, tris[i].normal, sizeof(t.n));
        t.reflectivity = std::clamp(tris[i].reflectivity, 0.0f, 1.0f);
        if (std::fabs(t.n[0]) < 1e-6f && std::fabs(t.n[1]) < 1e-6f && std::fabs(t.n[2]) < 1e-6f) {
            compute_normal(t);
        } else {
            float len = std::sqrt(t.n[0]*t.n[0] + t.n[1]*t.n[1] + t.n[2]*t.n[2]);
            if (len < 1e-6f) compute_normal(t);
            else { t.n[0]/=len; t.n[1]/=len; t.n[2]/=len; }
        }
        impl->tris.push_back(t);
    }
    return 1;
}

int raytrace3d_set_seed(Raytrace3D* rt, uint32_t seed) {
    if (!rt) return 0;
    auto* impl = reinterpret_cast<Raytrace3DImpl*>(rt);
    impl->seed = seed ? seed : 1u;
    return 1;
}

int raytrace3d_set_wave(Raytrace3D* rt, float frequency, float phase0) {
    if (!rt) return 0;
    auto* impl = reinterpret_cast<Raytrace3DImpl*>(rt);
    impl->wave_frequency = std::max(0.0f, frequency);
    impl->wave_phase0 = phase0;
    return 1;
}

int raytrace3d_render_layers_f32(
    Raytrace3D* rt,
    int layer_count,
    const float* layer_centers_z,
    float* out_layers_f32,
    int32_t out_len_floats) {
    if (!rt || !out_layers_f32) return 0;
    auto* impl = reinterpret_cast<Raytrace3DImpl*>(rt);
    if (impl->width <= 0 || impl->height <= 0) return 0;
    const int lc = std::max(1, layer_count);
    const int32_t need = impl->width * impl->height * lc;
    if (out_len_floats < need) return 0;

    const float min_x = impl->room_min_x;
    const float min_y = impl->room_min_y;
    const float min_z = impl->room_min_z;
    const float max_x = impl->room_max_x;
    const float max_y = impl->room_max_y;
    const float max_z = impl->room_max_z;
    if (!(max_x > min_x && max_y > min_y && max_z > min_z)) return 0;

    const size_t per_layer = static_cast<size_t>(impl->width) * static_cast<size_t>(impl->height);
    const size_t total = per_layer * static_cast<size_t>(lc);
    ensure_buffers(impl, total);
    std::fill(out_layers_f32, out_layers_f32 + total, 0.0f);

    uint32_t seed = impl->seed;
    const float eps = 1e-3f;
    const int rays = std::max(1, impl->rays);
    const int max_bounces = std::max(0, impl->max_reflections);

    for (int i = 0; i < rays; ++i) {
        // Uniform-ish sample over sphere using two randoms for direction, while keeping some determinism by index.
        float u = (float(i) + 0.5f) / float(rays);
        float v = rand01(seed);
        float theta = 2.0f * kPi * u;
        float zdir = 2.0f * v - 1.0f;
        float rxy = std::sqrt(std::max(0.0f, 1.0f - zdir * zdir));
        float dx = rxy * std::cos(theta);
        float dy = rxy * std::sin(theta);
        float dz = zdir;

        // Random point on light sphere surface
        float a = rand01(seed) * 2.0f * kPi;
        float b = std::acos(std::clamp(2.0f * rand01(seed) - 1.0f, -1.0f, 1.0f));
        float sx = std::sin(b) * std::cos(a);
        float sy = std::sin(b) * std::sin(a);
        float sz = std::cos(b);
        float x = impl->light_x + sx * impl->light_radius;
        float y = impl->light_y + sy * impl->light_radius;
        float z = impl->light_z + sz * impl->light_radius;
        x = std::clamp(x, min_x + eps, max_x - eps);
        y = std::clamp(y, min_y + eps, max_y - eps);
        z = std::clamp(z, min_z + eps, max_z - eps);

        float total_len = 0.0f;
        for (int bounce = 0; bounce <= max_bounces; ++bounce) {
            float tx = std::numeric_limits<float>::infinity();
            float ty = std::numeric_limits<float>::infinity();
            float tz = std::numeric_limits<float>::infinity();
            if (dx > 0.0f) tx = (max_x - x) / dx;
            else if (dx < 0.0f) tx = (min_x - x) / dx;
            if (dy > 0.0f) ty = (max_y - y) / dy;
            else if (dy < 0.0f) ty = (min_y - y) / dy;
            if (dz > 0.0f) tz = (max_z - z) / dz;
            else if (dz < 0.0f) tz = (min_z - z) / dz;

            float t = std::min({tx, ty, tz});
            if (!std::isfinite(t) || t <= 0.0f) break;

            float nx = x + dx * t;
            float ny = y + dy * t;
            float nz = z + dz * t;

            float seg_len = t;
            float avg_len = total_len + 0.5f * seg_len;
            float bounce_att = std::pow(impl->bounce_decay, static_cast<float>(bounce));
            float dist_att = std::exp(-impl->distance_decay * avg_len);
            float weight = std::max(0.0f, bounce_att * dist_att);

            add_segment_quantized(impl, x, y, z, nx, ny, nz, weight, lc, layer_centers_z, out_layers_f32);

            bool hit_x = std::fabs(t - tx) <= 1e-6f || tx < std::min(ty, tz);
            bool hit_y = std::fabs(t - ty) <= 1e-6f || ty < std::min(tx, tz);
            bool hit_z = std::fabs(t - tz) <= 1e-6f || tz < std::min(tx, ty);
            if (hit_x) dx = -dx;
            if (hit_y) dy = -dy;
            if (hit_z) dz = -dz;

            x = std::clamp(nx + dx * eps, min_x + eps, max_x - eps);
            y = std::clamp(ny + dy * eps, min_y + eps, max_y - eps);
            z = std::clamp(nz + dz * eps, min_z + eps, max_z - eps);
            total_len += seg_len;
        }
    }

    // Blur per layer.
    if (impl->blur_sigma > 0.0f) {
        for (int li = 0; li < lc; ++li) {
            float* layer = out_layers_f32 + static_cast<size_t>(li) * per_layer;
            // reuse tmp slice from impl
            if (impl->tmp.size() < per_layer) impl->tmp.assign(per_layer, 0.0f);
            gaussian_blur_2d(layer, impl->tmp.data(), impl->width, impl->height, impl->blur_sigma);
        }
    }

    return 1;
}

int raytrace3d_render_field_f32(
    Raytrace3D* rt,
    int layer_count,
    const float* layer_centers_z,
    float* out_field_f32,
    int32_t out_len_floats) {
    if (!rt || !out_field_f32) return 0;
    auto* impl = reinterpret_cast<Raytrace3DImpl*>(rt);
    if (impl->width <= 0 || impl->height <= 0) return 0;
    const int lc = std::max(1, layer_count);
    const int32_t need = impl->width * impl->height * lc * 6;
    if (out_len_floats < need) return 0;

    const float min_x = impl->room_min_x;
    const float min_y = impl->room_min_y;
    const float min_z = impl->room_min_z;
    const float max_x = impl->room_max_x;
    const float max_y = impl->room_max_y;
    const float max_z = impl->room_max_z;
    if (!(max_x > min_x && max_y > min_y && max_z > min_z)) return 0;

    const size_t px = static_cast<size_t>(impl->width) * static_cast<size_t>(impl->height);
    const size_t total = px * static_cast<size_t>(lc) * 6u;
    std::fill(out_field_f32, out_field_f32 + total, 0.0f);

    uint32_t seed = impl->seed;
    const float eps = 1e-3f;
    const int rays = std::max(1, impl->rays);
    const int max_bounces = std::max(0, impl->max_reflections);
    const float k = 2.0f * kPi * impl->wave_frequency;

    for (int i = 0; i < rays; ++i) {
        float u = (float(i) + 0.5f) / float(rays);
        float v = rand01(seed);
        float theta = 2.0f * kPi * u;
        float zdir = 2.0f * v - 1.0f;
        float rxy = std::sqrt(std::max(0.0f, 1.0f - zdir * zdir));
        float dx = rxy * std::cos(theta);
        float dy = rxy * std::sin(theta);
        float dz = zdir;

        float a = rand01(seed) * 2.0f * kPi;
        float b = std::acos(std::clamp(2.0f * rand01(seed) - 1.0f, -1.0f, 1.0f));
        float sx = std::sin(b) * std::cos(a);
        float sy = std::sin(b) * std::sin(a);
        float sz = std::cos(b);
        float x = impl->light_x + sx * impl->light_radius;
        float y = impl->light_y + sy * impl->light_radius;
        float z = impl->light_z + sz * impl->light_radius;
        x = std::clamp(x, min_x + eps, max_x - eps);
        y = std::clamp(y, min_y + eps, max_y - eps);
        z = std::clamp(z, min_z + eps, max_z - eps);

        float total_len = 0.0f;
        for (int bounce = 0; bounce <= max_bounces; ++bounce) {
            float tx = std::numeric_limits<float>::infinity();
            float ty = std::numeric_limits<float>::infinity();
            float tz = std::numeric_limits<float>::infinity();
            if (dx > 0.0f) tx = (max_x - x) / dx;
            else if (dx < 0.0f) tx = (min_x - x) / dx;
            if (dy > 0.0f) ty = (max_y - y) / dy;
            else if (dy < 0.0f) ty = (min_y - y) / dy;
            if (dz > 0.0f) tz = (max_z - z) / dz;
            else if (dz < 0.0f) tz = (min_z - z) / dz;

            float t_hit = std::min({tx, ty, tz});
            float hit_nx = 0.0f, hit_ny = 0.0f, hit_nz = 0.0f;
            float hit_reflect = 1.0f;
            // Mesh intersection
            for (const auto& tri : impl->tris) {
                float ttri = 0.0f, nx_hit=0.0f, ny_hit=0.0f, nz_hit=0.0f;
                if (!tri_intersect(tri, x, y, z, dx, dy, dz, ttri, nx_hit, ny_hit, nz_hit)) continue;
                if (ttri < t_hit) {
                    t_hit = ttri;
                    hit_nx = nx_hit; hit_ny = ny_hit; hit_nz = nz_hit;
                    hit_reflect = tri.reflectivity;
                }
            }

            if (!std::isfinite(t_hit) || t_hit <= 0.0f) break;
            float nx = x + dx * t_hit;
            float ny = y + dy * t_hit;
            float nz = z + dz * t_hit;

            float seg_len = t_hit;
            float avg_len = total_len + 0.5f * seg_len;
            float bounce_att = std::pow(impl->bounce_decay, static_cast<float>(bounce));
            float dist_att = std::exp(-impl->distance_decay * avg_len);
            float weight = std::max(0.0f, bounce_att * dist_att);
            float phase = impl->wave_phase0 + k * avg_len;

            add_segment_field_quantized(impl, x, y, z, nx, ny, nz, dx, dy, dz, weight, phase, lc, layer_centers_z, out_field_f32);

            bool hit_mesh = (hit_reflect < 1.1f) && (t_hit < std::min({tx, ty, tz}) - 1e-6f);
            if (hit_mesh) {
                // Reflect or absorb
                if (hit_reflect <= 0.0f) break;
                float dot = dx*hit_nx + dy*hit_ny + dz*hit_nz;
                dx = dx - 2.0f * dot * hit_nx;
                dy = dy - 2.0f * dot * hit_ny;
                dz = dz - 2.0f * dot * hit_nz;
                float lenv = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (lenv > 1e-6f) { dx/=lenv; dy/=lenv; dz/=lenv; }
                weight *= hit_reflect;
            } else {
                bool hit_x = std::fabs(t_hit - tx) <= 1e-6f || tx < std::min(ty, tz);
                bool hit_y = std::fabs(t_hit - ty) <= 1e-6f || ty < std::min(tx, tz);
                bool hit_z = std::fabs(t_hit - tz) <= 1e-6f || tz < std::min(tx, ty);
                if (hit_x) dx = -dx;
                if (hit_y) dy = -dy;
                if (hit_z) dz = -dz;
            }

            x = std::clamp(nx + dx * eps, min_x + eps, max_x - eps);
            y = std::clamp(ny + dy * eps, min_y + eps, max_y - eps);
            z = std::clamp(nz + dz * eps, min_z + eps, max_z - eps);
            total_len += seg_len;
        }
    }

    if (impl->blur_sigma > 0.0f) {
        // Blur I, Dx, Dy, Dz, Re, Im independently per layer.
        if (impl->accum.size() < px) impl->accum.assign(px, 0.0f);
        if (impl->tmp.size() < px) impl->tmp.assign(px, 0.0f);
        for (int li = 0; li < lc; ++li) {
            for (int ch = 0; ch < 6; ++ch) {
                for (size_t p = 0; p < px; ++p) {
                    impl->accum[p] = out_field_f32[(static_cast<size_t>(li) * px + p) * 6u + static_cast<size_t>(ch)];
                }
                gaussian_blur_2d(impl->accum.data(), impl->tmp.data(), impl->width, impl->height, impl->blur_sigma);
                for (size_t p = 0; p < px; ++p) {
                    out_field_f32[(static_cast<size_t>(li) * px + p) * 6u + static_cast<size_t>(ch)] = impl->accum[p];
                }
            }
        }
    }

    return 1;
}

} // extern "C"
