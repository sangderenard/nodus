#pragma once
#include <cstdint>

extern "C" {

typedef struct Raytrace2D Raytrace2D;

// Create/destroy a 2D ray tracer that renders into an RGBA buffer of size width*height.
Raytrace2D* raytrace2d_create(int width, int height);
void raytrace2d_destroy(Raytrace2D* rt);

// Resize the output buffer (reallocates internal buffers).
int raytrace2d_resize(Raytrace2D* rt, int width, int height);

// Set the rectangular room bounds (world coordinates).
int raytrace2d_set_room(Raytrace2D* rt, float min_x, float min_y, float max_x, float max_y);

// Set the circular light (center + radius, world coordinates).
int raytrace2d_set_light(Raytrace2D* rt, float cx, float cy, float radius);

// Set ray-tracing parameters: number of rays, max reflections, blur sigma (pixels).
int raytrace2d_set_params(Raytrace2D* rt, int ray_count, int max_reflections, float blur_sigma);
// Set attenuation tuning: bounce decay (0..1) and distance decay (>=0, per unit).
int raytrace2d_set_attenuation(Raytrace2D* rt, float bounce_decay, float distance_decay);

// Set a deterministic seed for ray sampling.
int raytrace2d_set_seed(Raytrace2D* rt, uint32_t seed);

// Render into RGBA8 buffer. Returns 1 on success.
int raytrace2d_render_rgba(Raytrace2D* rt, uint8_t* out_rgba, int32_t out_len_bytes);

} // extern "C"
