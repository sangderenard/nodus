#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Describes a single surface interaction produced by the tracer.
typedef struct GP_SurfaceHit {
    float px;
    float py;
    float pz;
    float nx;
    float ny;
    float nz;
    float dirx;
    float diry;
    float dirz;
    float wavelength; // nanometers
    float phase;
    float time; // seconds
    uint32_t surface_id;
    uint32_t material_id;
    float radiance_r;
    float radiance_g;
    float radiance_b;
    float weight; // optional confidence/weight
} GP_SurfaceHit;

typedef void(*GP_SurfaceHitFn)(void* user, const GP_SurfaceHit* hit);

#ifdef __cplusplus
}
#endif
