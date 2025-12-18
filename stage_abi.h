#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GP_StageContext GP_StageContext;

typedef struct GP_StageDims {
    int32_t width_px;
    int32_t height_px;
    int32_t internal_width_px;
    int32_t internal_height_px;
    int32_t layer_count;
    int32_t oversample;
    int32_t downsample;
} GP_StageDims;

typedef struct GP_StageLight {
    float x;
    float y;
    float radius;
    float intensity; // 0..1 nominal
    int32_t layer;   // 0..layer_count-1
} GP_StageLight;

typedef struct GP_StageLight3D {
    float x;
    float y;
    float z;
    float radius;
    float intensity; // 0..1 nominal
} GP_StageLight3D;

typedef struct GP_StageEmitter3D {
    float x;
    float y;
    float z;
    float radius;
    float intensity;   // 0..1 nominal
    float frequency;   // arbitrary units; phase uses 2π*frequency*distance
    float phase0;      // radians
} GP_StageEmitter3D;

typedef struct GP_StageMeshTriangle {
    float v0[3];
    float v1[3];
    float v2[3];
    float normal[3];    // optional, auto-computed if zero
    float reflectivity; // 0..1
} GP_StageMeshTriangle;

typedef enum GP_StageFieldMode {
    GP_STAGE_FIELD_SCALAR = 0,
    GP_STAGE_FIELD_VECTOR6 = 1, // [I, Dx, Dy, Dz, Re, Im]
} GP_StageFieldMode;

typedef struct GP_StageFieldDesc {
    int32_t field_mode;     // GP_StageFieldMode
    int32_t channels;       // 0 or 6
    int32_t width_px;
    int32_t height_px;
    int32_t layer_count;
} GP_StageFieldDesc;

typedef void(*GP_StageOcclusionFieldFn)(void* user, const GP_StageFieldDesc* desc, float* io_field_layers);

// Optional occlusion reduction hook: called after per-frame accumulation and before
// temporal integration. `io_frame_layers` is a flattened tensor (layers * w*h).
typedef void(*GP_StageOcclusionFn)(void* user, int32_t width_px, int32_t height_px, int32_t layer_count, float* io_frame_layers);

// Create/destroy a stage. `width_px/height_px` are the output dimensions.
GP_StageContext* gp_stage_create(int32_t width_px, int32_t height_px, int32_t layer_count);
void gp_stage_destroy(GP_StageContext* st);

// Resize output dimensions (keeps params, clears buffers).
int32_t gp_stage_resize(GP_StageContext* st, int32_t width_px, int32_t height_px);

// Get stage dimensions and configuration.
int32_t gp_stage_get_dims(const GP_StageContext* st, GP_StageDims* out_dims);

// Layer stack helpers (adds/removes the top-most layer).
// `gp_stage_pop_layer` will not pop the last remaining layer.
int32_t gp_stage_push_layer(GP_StageContext* st);
int32_t gp_stage_pop_layer(GP_StageContext* st);

// Configure oversample/downsample. Oversample controls internal raytrace resolution
// relative to output. Downsample controls internal->output; typically equal to oversample.
int32_t gp_stage_set_sampling(GP_StageContext* st, int32_t oversample, int32_t downsample);
int32_t gp_stage_set_occlusion_callback(GP_StageContext* st, GP_StageOcclusionFn cb, void* user);

// Configure depth (world units) used by the 3D ray tracer. If <=0, stage uses a
// 2D tracer per layer. If >0, stage uses the 3D tracer and quantizes along z
// into its layers.
int32_t gp_stage_set_depth(GP_StageContext* st, float depth_units);

// Optional per-layer z centers (length == layer_count). If not set, layers are
// split equidistant across [0, depth_units].
int32_t gp_stage_set_layer_depths(GP_StageContext* st, const float* layer_centers_z, int32_t count);

// Configure temporal accumulation: per-redraw decay and max intensity clamp.
int32_t gp_stage_set_temporal(GP_StageContext* st, float decay, float max_intensity);

// Configure raytrace parameters.
int32_t gp_stage_set_ray_params(GP_StageContext* st, int32_t ray_count, int32_t max_reflections, float blur_sigma);
int32_t gp_stage_set_ray_attenuation(GP_StageContext* st, float bounce_decay, float distance_decay);
int32_t gp_stage_set_exposure(GP_StageContext* st, float exposure);

// Layer tint (RGBA 0..255); alpha is treated as a multiplier for that layer.
int32_t gp_stage_set_layer_tint(GP_StageContext* st, int32_t layer, const uint8_t rgba[4]);

// Clear all layers to zero intensity.
int32_t gp_stage_clear(GP_StageContext* st);

// Light list management.
int32_t gp_stage_clear_lights(GP_StageContext* st);
int32_t gp_stage_add_light(GP_StageContext* st, const GP_StageLight* light);
int32_t gp_stage_clear_lights3d(GP_StageContext* st);
int32_t gp_stage_add_light3d(GP_StageContext* st, const GP_StageLight3D* light);
int32_t gp_stage_clear_emitters3d(GP_StageContext* st);
int32_t gp_stage_add_emitter3d(GP_StageContext* st, const GP_StageEmitter3D* emitter);

// Field mode and accessors.
int32_t gp_stage_set_field_mode(GP_StageContext* st, int32_t field_mode);
int32_t gp_stage_get_field_desc(const GP_StageContext* st, GP_StageFieldDesc* out_desc);
int32_t gp_stage_set_occlusion_field_callback(GP_StageContext* st, GP_StageOcclusionFieldFn cb, void* user);
int32_t gp_stage_copy_field_f32(const GP_StageContext* st, float* out_field_f32, int32_t out_len_floats);

// Emission alpha mask (scene-sized, 0..255) used to spawn additional emitters each frame.
// The mask is copied into the stage and resized with the stage.
int32_t gp_stage_set_emission_alpha_mask(GP_StageContext* st, const uint8_t* alpha, int32_t pitch_bytes);
int32_t gp_stage_set_emission_mask_sampling(GP_StageContext* st, int32_t samples_per_frame, float z, float radius, float intensity, float frequency, float phase0);

// Mesh: opaque triangles (copied). Passing count=0 clears.
int32_t gp_stage_set_mesh(GP_StageContext* st, const GP_StageMeshTriangle* tris, int32_t count);

// 3D masks at internal (oversampled) resolution.
// Volume layout is layer-major: slice_pitch bytes per z-layer, row_pitch bytes per y-row.
// Each voxel is a uint8 value in [0..255].
// - Occlusion alpha: 0 = empty, 255 = fully opaque (attenuates contributions).
// - Negative emission damping: 0 = no damping, 255 = fully damp (multiplies integration result by (1-damp)).
int32_t gp_stage_set_occlusion_alpha_volume(GP_StageContext* st, const uint8_t* alpha, int32_t row_pitch_bytes, int32_t slice_pitch_bytes);
int32_t gp_stage_set_negative_emission_volume(GP_StageContext* st, const uint8_t* damp, int32_t row_pitch_bytes, int32_t slice_pitch_bytes);

// Direct per-layer tensor access (output resolution).
int32_t gp_stage_get_value(const GP_StageContext* st, int32_t layer, int32_t x, int32_t y, float* out_value);
int32_t gp_stage_set_value(GP_StageContext* st, int32_t layer, int32_t x, int32_t y, float value);

// Render the stage into its internal output RGBA buffer.
int32_t gp_stage_render(GP_StageContext* st);

// Copy the rendered RGBA8 output into `out_rgba` (pitch is tightly packed).
int32_t gp_stage_copy_rgba(const GP_StageContext* st, uint8_t* out_rgba, int32_t out_len_bytes);

#ifdef __cplusplus
}
#endif
