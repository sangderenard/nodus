#include "stage_abi.h"

#include "raytrace_2d.h"
#include "raytrace_3d.h"
#include "table_abi.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

struct StageLayer {
    std::vector<float> temporal;
    uint8_t tint[4] = {255, 255, 255, 255};
};

struct GP_StageContextImpl {
    int32_t w = 0;
    int32_t h = 0;
    int32_t oversample = 2;
    int32_t downsample = 2;
    float depth_units = 0.0f;
    std::vector<float> layer_centers_z;
    int32_t field_mode = GP_STAGE_FIELD_SCALAR;

    float temporal_decay = 0.92f;
    float temporal_max = 1.0f;
    float exposure = 1.0f;

    int32_t rays = 512;
    int32_t reflections = 2;
    float blur_sigma = 2.0f;
    float bounce_decay = 0.75f;
    float distance_decay = 0.0025f;

    uint32_t frame = 0;

    GP_StageOcclusionFn occlusion_cb = nullptr;
    void* occlusion_user = nullptr;
    GP_StageOcclusionFieldFn occlusion_field_cb = nullptr;
    void* occlusion_field_user = nullptr;

    Raytrace2D* rt = nullptr;
    Raytrace3D* rt3 = nullptr;
    std::vector<uint8_t> rt_rgba_hi;
    std::vector<float> rt_layers_hi;
    std::vector<float> rt_field_hi;
    std::vector<float> frame_accum; // flattened (layers * w*h)
    std::vector<StageLayer> layers;
    std::vector<GP_StageLight> lights;
    std::vector<GP_StageLight3D> lights3d;
    std::vector<GP_StageEmitter3D> emitters3d;
    std::vector<GP_StageMeshTriangle> mesh;

    std::vector<float> field_frame;
    std::vector<float> field_temporal;

    std::vector<uint8_t> emission_alpha;
    int32_t emission_pitch = 0;
    int32_t emission_samples_per_frame = 0;
    float emission_z = 0.0f;
    float emission_radius = 3.0f;
    float emission_intensity = 0.2f;
    float emission_frequency = 0.0f;
    float emission_phase0 = 0.0f;

    // Internal-resolution (oversampled) 3D volumes, layer-major.
    std::vector<uint8_t> occlusion_alpha_hi;
    std::vector<uint8_t> negative_damp_hi;
    std::vector<uint8_t> out_rgba;

    GP_SurfaceHitFn hit_cb = nullptr;
    void* hit_user = nullptr;
    bool capture_samples = false;
    std::vector<GP_SurfaceHit> staged_hits;
    size_t staged_capture_limit = 0;
    uint64_t pending_batch_id = 0;
    double pending_batch_time = 0.0;
    uint32_t pending_sample_count = 0;
    bool batch_committed = false;
};

namespace {

static inline uint32_t mix_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x ? x : 1u;
}

static inline uint32_t lcg_next(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return s;
}

static inline float rand01(uint32_t& s) {
    return (lcg_next(s) >> 8) * (1.0f / 16777216.0f);
}

static constexpr uint32_t kStageSampleStride = 18;

static inline float pack_u32_as_float(uint32_t value) {
    float storage = 0.0f;
    std::memcpy(&storage, &value, sizeof(storage));
    return storage;
}

static void stage_surface_hit_trampoline(void* user, const GP_SurfaceHit* hit) {
    if (!user || !hit) return;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(user);
    if (s->capture_samples) {
        if (s->staged_capture_limit == 0 || s->staged_hits.size() < s->staged_capture_limit) {
            s->staged_hits.push_back(*hit);
        }
    }
    if (s->hit_cb) {
        s->hit_cb(s->hit_user, hit);
    }
}

static void fill_sample_payload(const GP_SurfaceHit& hit, float* dst, uint32_t stride) {
    if (!dst || stride == 0) return;
    std::fill(dst, dst + stride, 0.0f);
    auto assign = [&](uint32_t idx, float value) {
        if (idx < stride) dst[idx] = value;
    };
    assign(0, hit.px);
    assign(1, hit.py);
    assign(2, hit.pz);
    assign(3, hit.nx);
    assign(4, hit.ny);
    assign(5, hit.nz);
    assign(6, hit.dirx);
    assign(7, hit.diry);
    assign(8, hit.dirz);
    assign(9, hit.wavelength);
    assign(10, hit.phase);
    assign(11, hit.time);
    assign(12, pack_u32_as_float(hit.surface_id));
    assign(13, pack_u32_as_float(hit.material_id));
    assign(14, hit.radiance_r);
    assign(15, hit.radiance_g);
    assign(16, hit.radiance_b);
    assign(17, hit.weight);
}

static void ensure_rt(GP_StageContextImpl* s) {
    if (!s) return;
    int os = std::max(1, s->oversample);
    int iw = std::max(0, s->w) * os;
    int ih = std::max(0, s->h) * os;
    if (iw <= 0 || ih <= 0) {
        if (s->rt) {
            raytrace2d_destroy(s->rt);
            s->rt = nullptr;
        }
        if (s->rt3) {
            raytrace3d_destroy(s->rt3);
            s->rt3 = nullptr;
        }
        s->rt_rgba_hi.clear();
        s->rt_layers_hi.clear();
        s->rt_field_hi.clear();
        return;
    }
    const size_t need = static_cast<size_t>(iw) * static_cast<size_t>(ih) * 4u;
    s->rt_rgba_hi.assign(need, 0);
    if (!s->rt) {
        s->rt = raytrace2d_create(iw, ih);
    } else {
        raytrace2d_resize(s->rt, iw, ih);
    }
    if (!s->rt3) {
        s->rt3 = raytrace3d_create(iw, ih);
    } else {
        raytrace3d_resize(s->rt3, iw, ih);
    }
}

static void ensure_mask_volumes(GP_StageContextImpl* s) {
    if (!s) return;
    const int os = std::max(1, s->oversample);
    const int iw = std::max(0, s->w) * os;
    const int ih = std::max(0, s->h) * os;
    const size_t layers = s->layers.size();
    const size_t px_hi = static_cast<size_t>(iw) * static_cast<size_t>(ih);
    const size_t total = px_hi * layers;
    if (s->occlusion_alpha_hi.size() != total) s->occlusion_alpha_hi.assign(total, 0);
    if (s->negative_damp_hi.size() != total) s->negative_damp_hi.assign(total, 0);
}

static void ensure_buffers(GP_StageContextImpl* s) {
    if (!s) return;
    const size_t px = static_cast<size_t>(std::max(0, s->w)) * static_cast<size_t>(std::max(0, s->h));
    const size_t layer_count = static_cast<size_t>(s->layers.size());
    s->frame_accum.assign(px * layer_count, 0.0f);
    s->out_rgba.assign(px * 4u, 0);
    for (auto& l : s->layers) {
        l.temporal.assign(px, 0.0f);
    }
    if (s->field_mode == GP_STAGE_FIELD_VECTOR6) {
        s->field_frame.assign(px * layer_count * 6u, 0.0f);
        s->field_temporal.assign(px * layer_count * 6u, 0.0f);
    } else {
        s->field_frame.clear();
        s->field_temporal.clear();
    }
    s->emission_alpha.assign(px, 0);
    s->emission_pitch = (s->w > 0) ? s->w : 0;
    ensure_rt(s);
    ensure_mask_volumes(s);
}

static inline size_t idx2(int32_t w, int32_t x, int32_t y) {
    return static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
}

} // namespace

extern "C" {

GP_StageContext* gp_stage_create(int32_t width_px, int32_t height_px, int32_t layer_count) {
    if (width_px <= 0 || height_px <= 0) return nullptr;
    int lc = std::max(1, layer_count);
    auto* s = new GP_StageContextImpl();
    s->w = width_px;
    s->h = height_px;
    s->layers.resize(static_cast<size_t>(lc));
    // Default tints: subtle cool/warm ramp.
    for (int i = 0; i < lc; ++i) {
        float t = (lc <= 1) ? 0.0f : (float(i) / float(lc - 1));
        uint8_t r = static_cast<uint8_t>(std::lround(120.0f + t * 80.0f));
        uint8_t g = static_cast<uint8_t>(std::lround(140.0f + t * 60.0f));
        uint8_t b = static_cast<uint8_t>(std::lround(220.0f - t * 100.0f));
        s->layers[static_cast<size_t>(i)].tint[0] = r;
        s->layers[static_cast<size_t>(i)].tint[1] = g;
        s->layers[static_cast<size_t>(i)].tint[2] = b;
        s->layers[static_cast<size_t>(i)].tint[3] = 255;
    }
    ensure_buffers(s);
    return reinterpret_cast<GP_StageContext*>(s);
}

void gp_stage_destroy(GP_StageContext* st) {
    if (!st) return;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    if (s->rt) {
        raytrace2d_destroy(s->rt);
        s->rt = nullptr;
    }
    if (s->rt3) {
        raytrace3d_destroy(s->rt3);
        s->rt3 = nullptr;
    }
    delete s;
}

int32_t gp_stage_resize(GP_StageContext* st, int32_t width_px, int32_t height_px) {
    if (!st) return 0;
    if (width_px <= 0 || height_px <= 0) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    if (s->w == width_px && s->h == height_px) return 1;
    s->w = width_px;
    s->h = height_px;
    s->frame = 0;
    ensure_buffers(s);
    return 1;
}

int32_t gp_stage_get_dims(const GP_StageContext* st, GP_StageDims* out_dims) {
    if (!st || !out_dims) return 0;
    const auto* s = reinterpret_cast<const GP_StageContextImpl*>(st);
    out_dims->width_px = s->w;
    out_dims->height_px = s->h;
    out_dims->internal_width_px = s->w * std::max(1, s->oversample);
    out_dims->internal_height_px = s->h * std::max(1, s->oversample);
    out_dims->layer_count = static_cast<int32_t>(s->layers.size());
    out_dims->oversample = s->oversample;
    out_dims->downsample = s->downsample;
    return 1;
}

int32_t gp_stage_push_layer(GP_StageContext* st) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    StageLayer nl;
    // Copy last tint by default.
    if (!s->layers.empty()) std::memcpy(nl.tint, s->layers.back().tint, 4);
    s->layers.push_back(nl);
    ensure_buffers(s);
    return static_cast<int32_t>(s->layers.size() - 1);
}

int32_t gp_stage_pop_layer(GP_StageContext* st) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    if (s->layers.size() <= 1) return 0;
    s->layers.pop_back();
    ensure_buffers(s);
    return 1;
}

int32_t gp_stage_set_sampling(GP_StageContext* st, int32_t oversample, int32_t downsample) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->oversample = std::max(1, oversample);
    s->downsample = std::max(1, downsample);
    s->frame = 0;
    ensure_buffers(s);
    return 1;
}

int32_t gp_stage_set_occlusion_callback(GP_StageContext* st, GP_StageOcclusionFn cb, void* user) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->occlusion_cb = cb;
    s->occlusion_user = user;
    return 1;
}

int32_t gp_stage_set_depth(GP_StageContext* st, float depth_units) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->depth_units = depth_units;
    return 1;
}

int32_t gp_stage_set_layer_depths(GP_StageContext* st, const float* layer_centers_z, int32_t count) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    if (!layer_centers_z || count <= 0) {
        s->layer_centers_z.clear();
        return 1;
    }
    if (count != static_cast<int32_t>(s->layers.size())) return 0;
    s->layer_centers_z.assign(layer_centers_z, layer_centers_z + count);
    return 1;
}

int32_t gp_stage_set_temporal(GP_StageContext* st, float decay, float max_intensity) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->temporal_decay = std::clamp(decay, 0.0f, 1.0f);
    s->temporal_max = std::max(0.0f, max_intensity);
    return 1;
}

int32_t gp_stage_set_ray_params(GP_StageContext* st, int32_t ray_count, int32_t max_reflections, float blur_sigma) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->rays = std::max(1, ray_count);
    s->reflections = std::max(0, max_reflections);
    s->blur_sigma = std::max(0.0f, blur_sigma);
    return 1;
}

int32_t gp_stage_set_ray_attenuation(GP_StageContext* st, float bounce_decay, float distance_decay) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->bounce_decay = std::clamp(bounce_decay, 0.0f, 1.0f);
    s->distance_decay = std::max(0.0f, distance_decay);
    return 1;
}

int32_t gp_stage_set_exposure(GP_StageContext* st, float exposure) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->exposure = std::max(0.0f, exposure);
    return 1;
}

int32_t gp_stage_set_layer_tint(GP_StageContext* st, int32_t layer, const uint8_t rgba[4]) {
    if (!st || !rgba) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    if (layer < 0 || layer >= static_cast<int32_t>(s->layers.size())) return 0;
    std::memcpy(s->layers[static_cast<size_t>(layer)].tint, rgba, 4);
    return 1;
}

int32_t gp_stage_clear(GP_StageContext* st) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    for (auto& l : s->layers) std::fill(l.temporal.begin(), l.temporal.end(), 0.0f);
    std::fill(s->out_rgba.begin(), s->out_rgba.end(), 0);
    s->staged_hits.clear();
    s->batch_committed = false;
    s->pending_sample_count = 0;
    s->pending_batch_time = 0.0;
    s->pending_batch_id = 0;
    return 1;
}

int32_t gp_stage_clear_lights(GP_StageContext* st) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->lights.clear();
    return 1;
}

int32_t gp_stage_add_light(GP_StageContext* st, const GP_StageLight* light) {
    if (!st || !light) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    if (light->layer < 0 || light->layer >= static_cast<int32_t>(s->layers.size())) return 0;
    s->lights.push_back(*light);
    return 1;
}

int32_t gp_stage_clear_lights3d(GP_StageContext* st) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->lights3d.clear();
    return 1;
}

int32_t gp_stage_add_light3d(GP_StageContext* st, const GP_StageLight3D* light) {
    if (!st || !light) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->lights3d.push_back(*light);
    return 1;
}

int32_t gp_stage_clear_emitters3d(GP_StageContext* st) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->emitters3d.clear();
    return 1;
}

int32_t gp_stage_add_emitter3d(GP_StageContext* st, const GP_StageEmitter3D* emitter) {
    if (!st || !emitter) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->emitters3d.push_back(*emitter);
    return 1;
}

int32_t gp_stage_set_field_mode(GP_StageContext* st, int32_t field_mode) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    int32_t m = (field_mode == GP_STAGE_FIELD_VECTOR6) ? GP_STAGE_FIELD_VECTOR6 : GP_STAGE_FIELD_SCALAR;
    if (s->field_mode == m) return 1;
    s->field_mode = m;
    ensure_buffers(s);
    return 1;
}

int32_t gp_stage_get_field_desc(const GP_StageContext* st, GP_StageFieldDesc* out_desc) {
    if (!st || !out_desc) return 0;
    const auto* s = reinterpret_cast<const GP_StageContextImpl*>(st);
    out_desc->field_mode = s->field_mode;
    out_desc->channels = (s->field_mode == GP_STAGE_FIELD_VECTOR6) ? 6 : 0;
    out_desc->width_px = s->w;
    out_desc->height_px = s->h;
    out_desc->layer_count = static_cast<int32_t>(s->layers.size());
    return 1;
}

int32_t gp_stage_set_occlusion_field_callback(GP_StageContext* st, GP_StageOcclusionFieldFn cb, void* user) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->occlusion_field_cb = cb;
    s->occlusion_field_user = user;
    return 1;
}

int32_t gp_stage_copy_field_f32(const GP_StageContext* st, float* out_field_f32, int32_t out_len_floats) {
    if (!st || !out_field_f32) return 0;
    const auto* s = reinterpret_cast<const GP_StageContextImpl*>(st);
    if (s->field_mode != GP_STAGE_FIELD_VECTOR6) return 0;
    const int32_t need = static_cast<int32_t>(s->field_temporal.size());
    if (need <= 0) return 0;
    if (out_len_floats < need) return 0;
    std::memcpy(out_field_f32, s->field_temporal.data(), static_cast<size_t>(need) * sizeof(float));
    return need;
}

int32_t gp_stage_set_emission_alpha_mask(GP_StageContext* st, const uint8_t* alpha, int32_t pitch_bytes) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    const size_t px = static_cast<size_t>(std::max(0, s->w)) * static_cast<size_t>(std::max(0, s->h));
    if (px == 0) return 0;
    if (!alpha) {
        std::fill(s->emission_alpha.begin(), s->emission_alpha.end(), 0);
        return 1;
    }
    int pitch = pitch_bytes;
    if (pitch <= 0) pitch = s->w;
    s->emission_alpha.assign(px, 0);
    for (int y = 0; y < s->h; ++y) {
        const uint8_t* row = alpha + static_cast<size_t>(y) * static_cast<size_t>(pitch);
        for (int x = 0; x < s->w; ++x) {
            s->emission_alpha[idx2(s->w, x, y)] = row[x];
        }
    }
    s->emission_pitch = s->w;
    return 1;
}

int32_t gp_stage_set_emission_mask_sampling(GP_StageContext* st, int32_t samples_per_frame, float z, float radius, float intensity, float frequency, float phase0) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->emission_samples_per_frame = std::max(0, samples_per_frame);
    s->emission_z = z;
    s->emission_radius = std::max(0.0f, radius);
    s->emission_intensity = std::max(0.0f, intensity);
    s->emission_frequency = std::max(0.0f, frequency);
    s->emission_phase0 = phase0;
    return 1;
}

int32_t gp_stage_set_mesh(GP_StageContext* st, const GP_StageMeshTriangle* tris, int32_t count) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->mesh.clear();
    if (count <= 0 || !tris) {
        raytrace3d_set_mesh(s->rt3, nullptr, 0);
        return 1;
    }
    s->mesh.assign(tris, tris + count);
    if (s->rt3) {
        raytrace3d_set_mesh(s->rt3, reinterpret_cast<const Raytrace3DMeshTriangle*>(tris), count);
    }
    return 1;
}
int32_t gp_stage_set_occlusion_alpha_volume(GP_StageContext* st, const uint8_t* alpha, int32_t row_pitch_bytes, int32_t slice_pitch_bytes) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    ensure_mask_volumes(s);
    const int os = std::max(1, s->oversample);
    const int iw = std::max(0, s->w) * os;
    const int ih = std::max(0, s->h) * os;
    const int layers = static_cast<int>(s->layers.size());
    if (iw <= 0 || ih <= 0 || layers <= 0) return 0;
    if (!alpha) {
        std::fill(s->occlusion_alpha_hi.begin(), s->occlusion_alpha_hi.end(), 0);
        return 1;
    }
    int row_pitch = row_pitch_bytes;
    if (row_pitch <= 0) row_pitch = iw;
    int slice_pitch = slice_pitch_bytes;
    if (slice_pitch <= 0) slice_pitch = row_pitch * ih;
    const size_t px_hi = static_cast<size_t>(iw) * static_cast<size_t>(ih);
    for (int z = 0; z < layers; ++z) {
        const uint8_t* slice = alpha + static_cast<size_t>(z) * static_cast<size_t>(slice_pitch);
        for (int y = 0; y < ih; ++y) {
            const uint8_t* row = slice + static_cast<size_t>(y) * static_cast<size_t>(row_pitch);
            for (int x = 0; x < iw; ++x) {
                s->occlusion_alpha_hi[static_cast<size_t>(z) * px_hi + idx2(iw, x, y)] = row[x];
            }
        }
    }
    return 1;
}

int32_t gp_stage_set_negative_emission_volume(GP_StageContext* st, const uint8_t* damp, int32_t row_pitch_bytes, int32_t slice_pitch_bytes) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    ensure_mask_volumes(s);
    const int os = std::max(1, s->oversample);
    const int iw = std::max(0, s->w) * os;
    const int ih = std::max(0, s->h) * os;
    const int layers = static_cast<int>(s->layers.size());
    if (iw <= 0 || ih <= 0 || layers <= 0) return 0;
    if (!damp) {
        std::fill(s->negative_damp_hi.begin(), s->negative_damp_hi.end(), 0);
        return 1;
    }
    int row_pitch = row_pitch_bytes;
    if (row_pitch <= 0) row_pitch = iw;
    int slice_pitch = slice_pitch_bytes;
    if (slice_pitch <= 0) slice_pitch = row_pitch * ih;
    const size_t px_hi = static_cast<size_t>(iw) * static_cast<size_t>(ih);
    for (int z = 0; z < layers; ++z) {
        const uint8_t* slice = damp + static_cast<size_t>(z) * static_cast<size_t>(slice_pitch);
        for (int y = 0; y < ih; ++y) {
            const uint8_t* row = slice + static_cast<size_t>(y) * static_cast<size_t>(row_pitch);
            for (int x = 0; x < iw; ++x) {
                s->negative_damp_hi[static_cast<size_t>(z) * px_hi + idx2(iw, x, y)] = row[x];
            }
        }
    }
    return 1;
}

int32_t gp_stage_get_value(const GP_StageContext* st, int32_t layer, int32_t x, int32_t y, float* out_value) {
    if (!st || !out_value) return 0;
    const auto* s = reinterpret_cast<const GP_StageContextImpl*>(st);
    if (layer < 0 || layer >= static_cast<int32_t>(s->layers.size())) return 0;
    if (x < 0 || y < 0 || x >= s->w || y >= s->h) return 0;
    const auto& l = s->layers[static_cast<size_t>(layer)];
    *out_value = l.temporal[idx2(s->w, x, y)];
    return 1;
}

int32_t gp_stage_set_value(GP_StageContext* st, int32_t layer, int32_t x, int32_t y, float value) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    if (layer < 0 || layer >= static_cast<int32_t>(s->layers.size())) return 0;
    if (x < 0 || y < 0 || x >= s->w || y >= s->h) return 0;
    auto& l = s->layers[static_cast<size_t>(layer)];
    l.temporal[idx2(s->w, x, y)] = value;
    return 1;
}

int32_t gp_stage_render(GP_StageContext* st) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    if (s->w <= 0 || s->h <= 0) return 0;
    if (!s->rt || !s->rt3) ensure_rt(s);
    if (!s->rt) return 0;

    const int os = std::max(1, s->oversample);
    const int ds = std::max(1, std::min(s->downsample, os));
    const int iw = s->w * os;
    const int ih = s->h * os;
    if (iw <= 0 || ih <= 0) return 0;

    std::fill(s->frame_accum.begin(), s->frame_accum.end(), 0.0f);
    ++s->frame;

    const bool use_3d = (s->depth_units > 0.0f) && (s->rt3 != nullptr);
    if (use_3d) {
        raytrace3d_set_room(s->rt3, 0.0f, 0.0f, 0.0f, static_cast<float>(s->w), static_cast<float>(s->h), s->depth_units);
        raytrace3d_set_params(s->rt3, s->rays, s->reflections, s->blur_sigma);
        raytrace3d_set_attenuation(s->rt3, s->bounce_decay, s->distance_decay);
        if (!s->mesh.empty()) {
            raytrace3d_set_mesh(s->rt3, reinterpret_cast<const Raytrace3DMeshTriangle*>(s->mesh.data()), static_cast<int>(s->mesh.size()));
        } else {
            raytrace3d_set_mesh(s->rt3, nullptr, 0);
        }
        if (s->capture_samples || s->hit_cb) {
            raytrace3d_set_surface_hit_callback(s->rt3, stage_surface_hit_trampoline, s);
        } else {
            raytrace3d_set_surface_hit_callback(s->rt3, nullptr, nullptr);
        }
    } else {
        raytrace2d_set_room(s->rt, 0.0f, 0.0f, static_cast<float>(s->w), static_cast<float>(s->h));
        raytrace2d_set_params(s->rt, s->rays, s->reflections, s->blur_sigma);
        raytrace2d_set_attenuation(s->rt, s->bounce_decay, s->distance_decay);
    }

    const size_t px = static_cast<size_t>(s->w) * static_cast<size_t>(s->h);
    const size_t layer_count = s->layers.size();
    ensure_mask_volumes(s);
    const size_t px_hi = static_cast<size_t>(iw) * static_cast<size_t>(ih);

    if (use_3d) {
        const int lc = static_cast<int>(layer_count);
        const size_t need_f = static_cast<size_t>(iw) * static_cast<size_t>(ih) * static_cast<size_t>(lc);
        if (s->rt_layers_hi.size() != need_f) s->rt_layers_hi.assign(need_f, 0.0f);

        const float* centers = nullptr;
        std::vector<float> auto_centers;
        if (!s->layer_centers_z.empty() && s->layer_centers_z.size() == layer_count) {
            centers = s->layer_centers_z.data();
        } else if (s->depth_units > 0.0f && layer_count > 0) {
            auto_centers.resize(layer_count);
            for (size_t i = 0; i < layer_count; ++i) {
                float t = (float(i) + 0.5f) / float(layer_count);
                auto_centers[i] = t * s->depth_units;
            }
            centers = auto_centers.data();
        }

        if (s->field_mode == GP_STAGE_FIELD_VECTOR6) {
            const size_t need_field = static_cast<size_t>(iw) * static_cast<size_t>(ih) * static_cast<size_t>(lc) * 6u;
            if (s->rt_field_hi.size() != need_field) s->rt_field_hi.assign(need_field, 0.0f);
            if (s->field_frame.size() != px * layer_count * 6u) s->field_frame.assign(px * layer_count * 6u, 0.0f);
            if (s->field_temporal.size() != s->field_frame.size()) s->field_temporal.assign(s->field_frame.size(), 0.0f);
            std::fill(s->field_frame.begin(), s->field_frame.end(), 0.0f);

            auto accumulate_field_hi_to_lo = [&](float emitter_intensity) {
                for (size_t layer = 0; layer < layer_count; ++layer) {
                    for (int y = 0; y < s->h; ++y) {
                        const int hy = y * ds;
                        const size_t row_in = static_cast<size_t>(hy) * static_cast<size_t>(iw);
                        const size_t row_out = static_cast<size_t>(y) * static_cast<size_t>(s->w);
                        for (int x = 0; x < s->w; ++x) {
                            const int hx = x * ds;
                            const size_t p_in = row_in + static_cast<size_t>(hx);
                            const size_t p_out = (layer * px + row_out + static_cast<size_t>(x)) * 6u;
                            const size_t p_in_base = (layer * px_hi + p_in) * 6u;
                            float occ = 0.0f;
                            if (!s->occlusion_alpha_hi.empty()) {
                                occ = s->occlusion_alpha_hi[layer * px_hi + p_in] / 255.0f;
                            }
                            float occ_mul = 1.0f - std::clamp(occ, 0.0f, 1.0f);
                            for (size_t ch = 0; ch < 6u; ++ch) {
                                s->field_frame[p_out + ch] += s->rt_field_hi[p_in_base + ch] * emitter_intensity * occ_mul;
                            }
                        }
                    }
                }
            };

            // Coherent emitters.
            for (size_t ei = 0; ei < s->emitters3d.size(); ++ei) {
                const auto& E = s->emitters3d[ei];
                if (E.intensity <= 0.0f) continue;
                uint32_t seed = mix_u32(static_cast<uint32_t>(ei + 1) ^ (s->frame * 0x85EBCA6Bu));
                raytrace3d_set_seed(s->rt3, seed);
                raytrace3d_set_wave(s->rt3, E.frequency, E.phase0);
                raytrace3d_set_light(s->rt3, E.x, E.y, E.z, std::max(0.0f, E.radius));
                if (!raytrace3d_render_field_f32(s->rt3, lc, centers, s->rt_field_hi.data(), static_cast<int32_t>(s->rt_field_hi.size()))) continue;
                accumulate_field_hi_to_lo(E.intensity);
            }

            // Back-compat lights3d list (treated as incoherent emitters with frequency=0).
            for (size_t li = 0; li < s->lights3d.size(); ++li) {
                const auto& L = s->lights3d[li];
                if (L.intensity <= 0.0f) continue;
                uint32_t seed = mix_u32(static_cast<uint32_t>(li + 1) ^ (s->frame * 0xC2B2AE35u));
                raytrace3d_set_seed(s->rt3, seed);
                raytrace3d_set_wave(s->rt3, /*frequency=*/0.0f, /*phase0=*/0.0f);
                raytrace3d_set_light(s->rt3, L.x, L.y, L.z, std::max(0.0f, L.radius));
                if (!raytrace3d_render_field_f32(s->rt3, lc, centers, s->rt_field_hi.data(), static_cast<int32_t>(s->rt_field_hi.size()))) continue;
                accumulate_field_hi_to_lo(L.intensity);
            }

            // Emission alpha mask sampling (scene-sized).
            if (s->emission_samples_per_frame > 0 && s->emission_intensity > 0.0f && !s->emission_alpha.empty()) {
                uint32_t rng = mix_u32(0x1234ABCDu ^ (s->frame * 0x9E3779B1u));
                int attempts = std::max(16, s->emission_samples_per_frame * 12);
                int spawned = 0;
                for (int a = 0; a < attempts && spawned < s->emission_samples_per_frame; ++a) {
                    int x = static_cast<int>(std::floor(rand01(rng) * float(s->w)));
                    int y = static_cast<int>(std::floor(rand01(rng) * float(s->h)));
                    x = std::clamp(x, 0, s->w - 1);
                    y = std::clamp(y, 0, s->h - 1);
                    uint8_t alpha = s->emission_alpha[idx2(s->w, x, y)];
                    if (alpha == 0) continue;
                    if (rand01(rng) > (alpha / 255.0f)) continue;
                    ++spawned;
                    raytrace3d_set_seed(s->rt3, mix_u32(rng ^ static_cast<uint32_t>(spawned)));
                    raytrace3d_set_wave(s->rt3, s->emission_frequency, s->emission_phase0);
                    raytrace3d_set_light(s->rt3, float(x) + 0.5f, float(y) + 0.5f, s->emission_z, s->emission_radius);
                    if (!raytrace3d_render_field_f32(s->rt3, lc, centers, s->rt_field_hi.data(), static_cast<int32_t>(s->rt_field_hi.size()))) continue;
                    accumulate_field_hi_to_lo(s->emission_intensity);
                }
            }

            if (s->occlusion_field_cb) {
                GP_StageFieldDesc desc{};
                desc.field_mode = s->field_mode;
                desc.channels = 6;
                desc.width_px = s->w;
                desc.height_px = s->h;
                desc.layer_count = static_cast<int32_t>(layer_count);
                s->occlusion_field_cb(s->occlusion_field_user, &desc, s->field_frame.data());
            }

            // Temporal integrate with negative emission damping (per-voxel).
            for (size_t layer = 0; layer < layer_count; ++layer) {
                for (int y = 0; y < s->h; ++y) {
                    const int hy = y * ds;
                    const size_t row_in = static_cast<size_t>(hy) * static_cast<size_t>(iw);
                    const size_t row_out = static_cast<size_t>(y) * static_cast<size_t>(s->w);
                    for (int x = 0; x < s->w; ++x) {
                        const int hx = x * ds;
                        const size_t p_in = row_in + static_cast<size_t>(hx);
                        float damp = 0.0f;
                        if (!s->negative_damp_hi.empty()) {
                            damp = s->negative_damp_hi[layer * px_hi + p_in] / 255.0f;
                        }
                        float damp_mul = 1.0f - std::clamp(damp, 0.0f, 1.0f);
                        const size_t base = (layer * px + row_out + static_cast<size_t>(x)) * 6u;
                        for (size_t ch = 0; ch < 6u; ++ch) {
                            float v = s->field_temporal[base + ch] * s->temporal_decay + s->field_frame[base + ch];
                            v *= damp_mul;
                            if (ch == 0u) v = std::min(s->temporal_max, v);
                            s->field_temporal[base + ch] = v;
                        }
                    }
                }
            }

            // Project to RGBA.
            const uint8_t base_r = 24, base_g = 24, base_b = 30;
            for (int y = 0; y < s->h; ++y) {
                for (int x = 0; x < s->w; ++x) {
                    const size_t p = idx2(s->w, x, y);
                    float r = base_r;
                    float g = base_g;
                    float b = base_b;
                    for (size_t layer = 0; layer < layer_count; ++layer) {
                        const size_t off = (layer * px + p) * 6u;
                        float I = std::clamp(s->field_temporal[off + 0] * s->exposure, 0.0f, 1.0f);
                        float dx = s->field_temporal[off + 1];
                        float dy = s->field_temporal[off + 2];
                        float dz = s->field_temporal[off + 3];
                        float dmag = std::sqrt(std::max(0.0f, dx*dx + dy*dy + dz*dz));
                        float nx = (dmag > 1e-6f) ? (dx / dmag) : 0.0f;
                        float ny = (dmag > 1e-6f) ? (dy / dmag) : 0.0f;
                        float nz = (dmag > 1e-6f) ? (dz / dmag) : 0.0f;
                        float dir_r = 0.5f + 0.5f * nx;
                        float dir_g = 0.5f + 0.5f * ny;
                        float dir_b = 0.5f + 0.5f * nz;
                        const auto& Lt = s->layers[layer];
                        float a = (Lt.tint[3] / 255.0f);
                        float tr = Lt.tint[0] * dir_r;
                        float tg = Lt.tint[1] * dir_g;
                        float tb = Lt.tint[2] * dir_b;
                        r += I * a * (tr - base_r);
                        g += I * a * (tg - base_g);
                        b += I * a * (tb - base_b);
                    }
                    const size_t o = p * 4u;
                    s->out_rgba[o + 0] = static_cast<uint8_t>(std::clamp(std::lround(r), 0l, 255l));
                    s->out_rgba[o + 1] = static_cast<uint8_t>(std::clamp(std::lround(g), 0l, 255l));
                    s->out_rgba[o + 2] = static_cast<uint8_t>(std::clamp(std::lround(b), 0l, 255l));
                    s->out_rgba[o + 3] = 255;
                }
            }
            return 1;
        }

        for (size_t li = 0; li < s->lights3d.size(); ++li) {
            const auto& L = s->lights3d[li];
            if (L.intensity <= 0.0f) continue;
            uint32_t seed = static_cast<uint32_t>(li + 1);
            seed ^= s->frame * 0x85EBCA6Bu;
            seed = mix_u32(seed);
            raytrace3d_set_seed(s->rt3, seed);
            raytrace3d_set_light(s->rt3, L.x, L.y, L.z, std::max(0.0f, L.radius));
            if (!raytrace3d_render_layers_f32(s->rt3, lc, centers, s->rt_layers_hi.data(), static_cast<int32_t>(s->rt_layers_hi.size()))) continue;

            for (size_t layer = 0; layer < layer_count; ++layer) {
                const size_t base_out = layer * px;
                const size_t base_in = layer * static_cast<size_t>(iw) * static_cast<size_t>(ih);
                for (int y = 0; y < s->h; ++y) {
                    const int hy = y * ds;
                    const size_t row_in = static_cast<size_t>(hy) * static_cast<size_t>(iw);
                    const size_t row_out = static_cast<size_t>(y) * static_cast<size_t>(s->w);
                    for (int x = 0; x < s->w; ++x) {
                        const int hx = x * ds;
                        const size_t p_in = row_in + static_cast<size_t>(hx);
                        float occ = 0.0f;
                        if (!s->occlusion_alpha_hi.empty()) occ = s->occlusion_alpha_hi[layer * px_hi + p_in] / 255.0f;
                        float occ_mul = 1.0f - std::clamp(occ, 0.0f, 1.0f);
                        float v = s->rt_layers_hi[base_in + p_in] * occ_mul;
                        s->frame_accum[base_out + row_out + static_cast<size_t>(x)] += v * L.intensity;
                    }
                }
            }
        }
    } else {
        for (size_t li = 0; li < s->lights.size(); ++li) {
            const auto& L = s->lights[li];
            if (L.intensity <= 0.0f) continue;
            if (L.layer < 0 || static_cast<size_t>(L.layer) >= layer_count) continue;
            uint32_t seed = static_cast<uint32_t>(L.layer + 1);
            seed ^= static_cast<uint32_t>(li + 1) * 0x9E3779B1u;
            seed ^= s->frame * 0x85EBCA6Bu;
            seed = mix_u32(seed);

            raytrace2d_set_seed(s->rt, seed);
            raytrace2d_set_light(s->rt, L.x, L.y, std::max(0.0f, L.radius));
            if (!raytrace2d_render_rgba(s->rt, s->rt_rgba_hi.data(), static_cast<int32_t>(s->rt_rgba_hi.size()))) continue;

            const size_t base = static_cast<size_t>(L.layer) * px;
            for (int y = 0; y < s->h; ++y) {
                const int hy = y * ds;
                const size_t row_hi = static_cast<size_t>(hy) * static_cast<size_t>(iw);
                const size_t row_lo = static_cast<size_t>(y) * static_cast<size_t>(s->w);
                for (int x = 0; x < s->w; ++x) {
                    const int hx = x * ds;
                    const size_t idx_hi = (row_hi + static_cast<size_t>(hx)) * 4u;
                    float v = s->rt_rgba_hi[idx_hi + 0] / 255.0f;
                    s->frame_accum[base + row_lo + static_cast<size_t>(x)] += v * L.intensity;
                }
            }
        }
    }

    if (s->occlusion_cb) {
        s->occlusion_cb(s->occlusion_user, s->w, s->h, static_cast<int32_t>(s->layers.size()), s->frame_accum.data());
    }

    // Temporal accumulate per layer.
    for (size_t layer = 0; layer < layer_count; ++layer) {
        auto& dst = s->layers[layer].temporal;
        const size_t base = layer * px;
        for (int y = 0; y < s->h; ++y) {
            const int hy = y * ds;
            const size_t row_in = static_cast<size_t>(hy) * static_cast<size_t>(iw);
            const size_t row_out = static_cast<size_t>(y) * static_cast<size_t>(s->w);
            for (int x = 0; x < s->w; ++x) {
                const int hx = x * ds;
                const size_t p_in = row_in + static_cast<size_t>(hx);
                float damp = 0.0f;
                if (!s->negative_damp_hi.empty()) damp = s->negative_damp_hi[layer * px_hi + p_in] / 255.0f;
                float damp_mul = 1.0f - std::clamp(damp, 0.0f, 1.0f);
                const size_t i = row_out + static_cast<size_t>(x);
                float v = dst[i] * s->temporal_decay + s->frame_accum[base + i];
                v *= damp_mul;
                dst[i] = std::min(s->temporal_max, v);
            }
        }
    }

    // Flatten.
    const uint8_t base_r = 24, base_g = 24, base_b = 30;
    for (int y = 0; y < s->h; ++y) {
        for (int x = 0; x < s->w; ++x) {
            const size_t p = idx2(s->w, x, y);
            float r = base_r;
            float g = base_g;
            float b = base_b;
            for (size_t layer = 0; layer < layer_count; ++layer) {
                const auto& L = s->layers[layer];
                float t = std::clamp(L.temporal[p] * s->exposure, 0.0f, 1.0f);
                float a = (L.tint[3] / 255.0f);
                r += t * a * (L.tint[0] - base_r);
                g += t * a * (L.tint[1] - base_g);
                b += t * a * (L.tint[2] - base_b);
            }
            const size_t o = p * 4u;
            s->out_rgba[o + 0] = static_cast<uint8_t>(std::clamp(std::lround(r), 0l, 255l));
            s->out_rgba[o + 1] = static_cast<uint8_t>(std::clamp(std::lround(g), 0l, 255l));
            s->out_rgba[o + 2] = static_cast<uint8_t>(std::clamp(std::lround(b), 0l, 255l));
            s->out_rgba[o + 3] = 255;
        }
    }
    return 1;
}

int32_t gp_stage_copy_rgba(const GP_StageContext* st, uint8_t* out_rgba, int32_t out_len_bytes) {
    if (!st || !out_rgba) return 0;
    const auto* s = reinterpret_cast<const GP_StageContextImpl*>(st);
    const int32_t need = s->w * s->h * 4;
    if (out_len_bytes < need) return 0;
    if (s->out_rgba.empty()) return 0;
    std::memcpy(out_rgba, s->out_rgba.data(), static_cast<size_t>(need));
    return 1;
}

int32_t gp_stage_set_hit_callback(GP_StageContext* st, GP_SurfaceHitFn cb, void* user) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->hit_cb = cb;
    s->hit_user = user;
    return 1;
}

int32_t gp_stage_enable_sample_capture(GP_StageContext* st, int32_t enable) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    bool new_state = (enable != 0);
    if (new_state && !s->capture_samples) {
        s->staged_hits.clear();
        s->batch_committed = false;
        s->pending_sample_count = 0;
        s->pending_batch_id = 0;
        s->pending_batch_time = 0.0;
    }
    if (!new_state) {
        s->staged_hits.clear();
    }
    s->capture_samples = new_state;
    return 1;
}

int32_t gp_stage_commit_staged_samples(GP_StageContext* st, uint64_t batch_id) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->pending_batch_id = batch_id;
    s->pending_batch_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    const size_t sample_count = s->staged_hits.size();
    s->pending_sample_count = static_cast<uint32_t>(std::min<size_t>(sample_count, static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
    s->batch_committed = true;
    return 1;
}

int32_t gp_stage_stream_samples(GP_StageContext* st, GP_TableContext* table_ctx, unsigned long long led_key, const GP_StageBatchOptions* opts) {
    if (!st || !table_ctx || !opts) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    if (!s->batch_committed) return 0;
    int32_t edge_idx = -1;
    if (!gp_table_edge_index_for_key(table_ctx, led_key, &edge_idx) || edge_idx < 0) return 0;

    uint32_t stride = opts->stride ? opts->stride : kStageSampleStride;
    stride = std::max<uint32_t>(stride, kStageSampleStride);
    if (stride > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) return 0;

    uint32_t sample_limit = opts->sample_limit ? opts->sample_limit : static_cast<uint32_t>(s->staged_hits.size());
    sample_limit = static_cast<uint32_t>(std::min<size_t>(sample_limit, s->staged_hits.size()));

    // Query tensor spec for debugging
    GP_TableEdgeTensorSpec spec{};
    (void)gp_table_edge_get_tensor_spec(table_ctx, edge_idx, &spec);

    // If tensor stride doesn't match our payload stride, try to set it to match.
    if ((spec.dim_count == 0 || spec.dims[0] != static_cast<int32_t>(stride))) {
        GP_TableEdgeTensorSpec new_spec{};
        new_spec.dim_count = 1;
        new_spec.dims[0] = static_cast<int32_t>(stride);
        // Choose a sensible slot count: prefer sample_limit if small, otherwise cap.
        uint32_t desired_slots = sample_limit > 0 ? sample_limit : 64u;
        const uint32_t kMaxSlots = 4096u;
        new_spec.slots = static_cast<int32_t>(std::max<uint32_t>(16u, std::min<uint32_t>(desired_slots, kMaxSlots)));
        new_spec.top_k = 0;
        if (gp_table_edge_set_tensor_spec(table_ctx, edge_idx, &new_spec)) {
            gp_table_edge_get_tensor_spec(table_ctx, edge_idx, &spec);
        }
    }

    std::vector<float> payload(stride);
    size_t pushed = 0;
    bool ok = true;
    for (size_t i = 0; i < sample_limit; ++i) {
        fill_sample_payload(s->staged_hits[i], payload.data(), stride);
        int32_t dropped = 0;
        if (!gp_table_edge_publish(table_ctx, edge_idx, led_key, payload.data(), static_cast<int32_t>(stride), &dropped)) {
            ok = false;
            break;
        }
        ++pushed;
    }

    GP_TableEdgeBatchMetadata meta{};
    meta.batch_id = s->pending_batch_id;
    meta.timestamp = s->pending_batch_time;
    meta.sample_count = static_cast<uint32_t>(pushed);
    meta.stride = stride;
    meta.schema_id = opts->schema_id;
    gp_table_edge_set_batch_metadata(table_ctx, edge_idx, &meta);

    s->batch_committed = false;
    s->staged_hits.clear();
    s->pending_sample_count = 0;
    return ok ? 1 : 0;
}

int32_t gp_stage_clear_staged_samples(GP_StageContext* st) {
    if (!st) return 0;
    auto* s = reinterpret_cast<GP_StageContextImpl*>(st);
    s->staged_hits.clear();
    s->batch_committed = false;
    s->pending_sample_count = 0;
    s->pending_batch_time = 0.0;
    s->pending_batch_id = 0;
    return 1;
}

int32_t gp_stage_perform_tick(GP_StageContext* st, GP_TableContext* table_ctx, uint64_t batch_id, const GP_StageBatchOptions* opts, uint8_t* out_rgba, int32_t out_pitch, int32_t width, int32_t height) {
    if (!st) return 0;
    // Ensure stage is sized appropriately for the requested output buffer.
    if (width > 0 && height > 0) gp_stage_resize(st, width, height);

    // Enable capture, render, commit and stream samples into the provided table.
    gp_stage_enable_sample_capture(st, 1);
    gp_stage_render(st);
    if (batch_id == 0) batch_id = static_cast<uint64_t>(std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
    gp_stage_commit_staged_samples(st, batch_id);
    // If opts is null, stream with defaults.
    GP_StageBatchOptions local_opts{};
    if (!opts) {
        local_opts.stride = 18;
        local_opts.sample_limit = 256;
        opts = &local_opts;
    }
    if (table_ctx) gp_stage_stream_samples(st, table_ctx, ((uint64_t)0 << 32) | ((uint64_t)1 << 16) | 0ull, opts);
    gp_stage_enable_sample_capture(st, 0);

    // Copy RGBA into out buffer if provided
    if (out_rgba && out_pitch >= width * 4 && width > 0 && height > 0) {
        gp_stage_copy_rgba(st, out_rgba, out_pitch * height);
    }
    return 1;
}

} // extern "C"
