#pragma once
#include <cstdint>

extern "C" {

typedef struct Raytrace3D Raytrace3D;

typedef struct Raytrace3DMeshTriangle {
    float v0[3];
    float v1[3];
    float v2[3];
    float normal[3];      // optional; if all zero, computed automatically
    float reflectivity;   // 0..1 (1=perfect mirror, 0=absorb)
} Raytrace3DMeshTriangle;

Raytrace3D* raytrace3d_create(int width, int height);
void raytrace3d_destroy(Raytrace3D* rt);
int raytrace3d_resize(Raytrace3D* rt, int width, int height);

// Set the rectangular room bounds (world coordinates).
int raytrace3d_set_room(Raytrace3D* rt,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z);

// Set the spherical light (center + radius, world coordinates).
int raytrace3d_set_light(Raytrace3D* rt, float cx, float cy, float cz, float radius);

// Set ray-tracing parameters: number of rays, max reflections, blur sigma (pixels).
int raytrace3d_set_params(Raytrace3D* rt, int ray_count, int max_reflections, float blur_sigma);

// Set attenuation tuning: bounce decay (0..1) and distance decay (>=0, per unit).
int raytrace3d_set_attenuation(Raytrace3D* rt, float bounce_decay, float distance_decay);

// Set opaque triangle mesh (copied). Passing count=0 clears mesh.
int raytrace3d_set_mesh(Raytrace3D* rt, const Raytrace3DMeshTriangle* tris, int count);

// Set a deterministic seed for ray sampling.
int raytrace3d_set_seed(Raytrace3D* rt, uint32_t seed);

// Configure wave parameters for phase accumulation (phase = phase0 + 2π*frequency*distance).
int raytrace3d_set_wave(Raytrace3D* rt, float frequency, float phase0);

// Render into layered float output (layers * width * height).
// `layer_centers_z` is an array of `layer_count` z values (world units) used to
// quantize each ray sample to the nearest layer; if NULL, layers are equidistant
// across room z extents.
int raytrace3d_render_layers_f32(
    Raytrace3D* rt,
    int layer_count,
    const float* layer_centers_z,
    float* out_layers_f32,
    int32_t out_len_floats);

// Render into layered vector-field output: (layers * width * height * 6).
// Layout per pixel: [I, Dx, Dy, Dz, Re, Im]
// - I: scalar intensity accumulator (>=0)
// - D*: accumulated direction (not normalized)
// - Re/Im: accumulated coherent sum using wave params
int raytrace3d_render_field_f32(
    Raytrace3D* rt,
    int layer_count,
    const float* layer_centers_z,
    float* out_field_f32,
    int32_t out_len_floats);

} // extern "C"
